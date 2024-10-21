import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_one_over_42_l578_57813

def my_sequence (n : ℕ+) : ℚ := 1 / ((n : ℚ) * (n + 1))

theorem sixth_term_is_one_over_42 : my_sequence 6 = 1 / 42 := by
  -- Unfold the definition of my_sequence
  unfold my_sequence
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_one_over_42_l578_57813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_20_equals_2_pow_4181_l578_57868

/-- Sequence b_n defined recursively -/
def b : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => b (n + 2) * b (n + 1)

/-- Theorem stating that the 20th term of sequence b_n equals 2^4181 -/
theorem b_20_equals_2_pow_4181 : b 20 = 2^4181 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_20_equals_2_pow_4181_l578_57868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_minimization_l578_57818

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the family of lines l
def line_l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the minimizing line
def min_line (x y : ℝ) : Prop := 2*x - y - 5 = 0

-- Theorem statement
theorem chord_length_minimization :
  ∀ (m : ℝ), 
  (∃ (x y : ℝ), circle_C x y ∧ line_l m x y) →
  (∀ (x y : ℝ), circle_C x y ∧ line_l m x y → 
    ∃ (x' y' : ℝ), circle_C x' y' ∧ min_line x' y' ∧ 
      ((x - x')^2 + (y - y')^2 ≤ (x - 1)^2 + (y - 2)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_minimization_l578_57818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_inequality_l578_57864

theorem factorial_inequality (n : ℕ) : 
  ((n.factorial ^ 3 : ℝ) ^ (1 / n : ℝ)) ≤ (n * (n + 1)^2 : ℝ) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_inequality_l578_57864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l578_57819

-- Define the point F
def F : ℝ × ℝ := (1, 1)

-- Define the line L: 3x + y - 4 = 0
def L (x y : ℝ) : Prop := 3 * x + y - 4 = 0

-- Define the distance from a point to F
noncomputable def dist_to_F (x y : ℝ) : ℝ := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)

-- Define the distance from a point to line L
noncomputable def dist_to_L (x y : ℝ) : ℝ := 
  abs (3 * x + y - 4) / Real.sqrt (3^2 + 1^2)

-- State the theorem
theorem trajectory_of_P : 
  ∀ x y : ℝ, dist_to_F x y = dist_to_L x y → x - 3 * y + 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l578_57819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angles_l578_57829

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ

-- Define the condition for the altitude
def altitude_condition (t : IsoscelesTriangle) : Prop :=
  ∃ side : ℝ, (t.leg * Real.sin (t.vertex_angle / 2) = side / 2) ∧ (side = t.base ∨ side = t.leg)

-- Theorem statement
theorem isosceles_triangle_vertex_angles 
  (t : IsoscelesTriangle) 
  (h : altitude_condition t) : 
  t.vertex_angle = 30 ∨ t.vertex_angle = 120 ∨ t.vertex_angle = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angles_l578_57829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellen_croissants_l578_57870

/-- The price of a can of cola in pence -/
def cola_price : ℚ := sorry

/-- The price of a croissant in pence -/
def croissant_price : ℚ := sorry

/-- Ellen's total money in pence -/
def total_money : ℚ := sorry

/-- Condition: Ellen can spend all her money on 6 cans of cola and 7 croissants -/
axiom condition1 : total_money = 6 * cola_price + 7 * croissant_price

/-- Condition: Ellen can spend all her money on 8 cans of cola and 4 croissants -/
axiom condition2 : total_money = 8 * cola_price + 4 * croissant_price

/-- Theorem: Ellen can buy 16 croissants if she spends all her money on croissants -/
theorem ellen_croissants : total_money / croissant_price = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellen_croissants_l578_57870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l578_57876

-- Define the square root of 180
noncomputable def sqrt_180 : ℝ := 6 * Real.sqrt 5

-- Theorem statement
theorem rationalize_denominator :
  5 / (sqrt_180 - 2) = (5 * (3 * Real.sqrt 5 + 1)) / 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l578_57876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l578_57826

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (h_prime : Nat.Prime p) (h_not_div : ¬(↑p ∣ a)) :
  ↑p ∣ (a^(p - 1) - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l578_57826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_number_theorem_l578_57880

def is_prime (n : ℕ) := Nat.Prime n

def is_power_of_prime (n : ℕ) := ∃ (p k : ℕ), is_prime p ∧ n = p^k

def has_digit_7 (n : ℕ) := ∃ (a b : ℕ), n = 10 * a + 7 ∨ n = 10 * b + 70

def ExactlyThreeTrue (a b c d : Prop) : Prop :=
  (a ∧ b ∧ c ∧ ¬d) ∨
  (a ∧ b ∧ ¬c ∧ d) ∨
  (a ∧ ¬b ∧ c ∧ d) ∨
  (¬a ∧ b ∧ c ∧ d)

theorem house_number_theorem :
  ∃! (n : ℕ),
    10 ≤ n ∧ n < 100 ∧
    (ExactlyThreeTrue
      (is_prime n)
      (is_power_of_prime n)
      (n % 5 = 0)
      (has_digit_7 n)) ∧
    n % 10 = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_number_theorem_l578_57880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l578_57888

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 6
  | (n + 3) => (a (n + 2))^2 + 9 / a (n + 1)

/-- Theorem stating the properties of the sequence a_n -/
theorem a_properties :
  (∀ n : ℕ, a n > 0) ∧ 
  (∀ n : ℕ, ∃ k : ℕ, a n = k) ∧
  ¬(∃ m : ℕ, 2109 ∣ a m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l578_57888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_arch_angle_theorem_l578_57872

/-- Represents a circular arrangement of congruent isosceles trapezoids -/
structure TrapezoidArch where
  num_trapezoids : ℕ
  central_angle : ℚ
  base_angle : ℚ

/-- The smaller interior angle adjacent to the longer base of one trapezoid -/
def smaller_interior_angle (arch : TrapezoidArch) : ℚ :=
  (180 - arch.central_angle / 2) / 2

/-- Theorem statement for the trapezoid arch problem -/
theorem trapezoid_arch_angle_theorem (arch : TrapezoidArch) :
  arch.num_trapezoids = 8 ∧
  arch.central_angle = 360 / arch.num_trapezoids ∧
  arch.base_angle = arch.central_angle / 2 →
  smaller_interior_angle arch = 315 / 4 := by
  sorry

#eval (315 : ℚ) / 4  -- This should output 78.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_arch_angle_theorem_l578_57872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_1002_solutions_l578_57828

-- Define the function g₁
noncomputable def g₁ (x : ℝ) : ℝ := 1/2 - 4/(4*x + 2)

-- Define the recursive function gₙ
noncomputable def g : ℕ → ℝ → ℝ
| 0 => λ x => x
| 1 => g₁
| (n+1) => λ x => g₁ (g n x)

-- State the theorem
theorem g_1002_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g 1002 x₁ = x₁ - 4 ∧ g 1002 x₂ = x₂ - 4 ∧ 
  (x₁ = -1/2 ∨ x₁ = 7) ∧ (x₂ = -1/2 ∨ x₂ = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_1002_solutions_l578_57828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l578_57895

noncomputable section

/-- A function that is monotonically decreasing on (0,+∞) -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f y < f x

/-- A function that is monotonically increasing on (0,+∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The power function -/
noncomputable def PowerFunction (m : ℤ) : ℝ → ℝ :=
  fun x ↦ x ^ m

theorem power_function_properties (m n : ℤ) :
  MonoDecreasing (PowerFunction m) →
  MonoIncreasing (PowerFunction n) →
  OddFunction (fun x ↦ PowerFunction m x + PowerFunction n x) →
  (m < 0 ∧ Odd m) ∧ (n > 0 ∧ Odd n) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l578_57895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_f_g_magnitudes_l578_57830

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := lg (1 - x)
noncomputable def g (x : ℝ) : ℝ := lg (1 + x)

-- State the theorem
theorem compare_f_g_magnitudes :
  (∀ x ∈ Set.Ioo 0 1, |f x| > |g x|) ∧
  (|f 0| = |g 0|) ∧
  (∀ x ∈ Set.Ioo (-1) 0, |f x| < |g x|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_f_g_magnitudes_l578_57830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l578_57850

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Vector m is defined as (a-c, a-b) -/
def m (a b c : ℝ) : ℝ × ℝ := (a - c, a - b)

/-- Vector n is defined as (a+b, c) -/
def n (a b c : ℝ) : ℝ × ℝ := (a + b, c)

/-- Vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Area of a triangle given side lengths -/
noncomputable def areaTriangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_properties (a b c : ℝ) (h1 : Triangle a b c) 
    (h2 : parallel (m a b c) (n a b c)) :
  ∃ (A B C : ℝ), 
    A + B + C = π ∧ 
    B = π / 3 ∧
    (a = 1 ∧ b = Real.sqrt 7 → areaTriangle a b c = 3 * Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l578_57850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_g_4_l578_57807

def g (x : ℕ) : ℕ :=
  if x % 5 = 0 ∧ x % 7 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_power : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => g (g_power n x)

theorem smallest_b_for_g_4 :
  ∀ b : ℕ, b > 1 → (g 4 = g_power b 4 ↔ b ≥ 21) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_g_4_l578_57807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_area_l578_57840

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (A B C : Point) : ℝ := sorry

/-- Function to find intersection points of an ellipse and a line -/
noncomputable def intersectionPoints (e : Ellipse) (l : Line) : Set Point := sorry

/-- Theorem stating the relationship between k and the area of triangle AMN -/
theorem ellipse_line_intersection_area 
  (e : Ellipse)
  (l : Line)
  (h_ellipse : e.a = 2 ∧ e.b = Real.sqrt 2)
  (A : Point)
  (h_A : A.x = 2 ∧ A.y = 0)
  (h_intersect : ∃ M N, M ∈ intersectionPoints e l ∧ N ∈ intersectionPoints e l ∧ M ≠ N)
  (h_area : ∀ M N, M ∈ intersectionPoints e l → N ∈ intersectionPoints e l → M ≠ N → 
            triangleArea A M N = 4 * Real.sqrt 2 / 5) :
  l.k = Real.sqrt 2 ∨ l.k = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_area_l578_57840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_below_x_axis_l578_57889

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- Calculates the y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (f : QuadraticFunction) : ℝ := f.c - (f.b^2) / (4 * f.a)

/-- Theorem: The value of c that places the vertex of y = 2x^2 - 6x + c just below the x-axis is 3.5 -/
theorem parabola_vertex_below_x_axis :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  let f : QuadraticFunction := ⟨2, -6, 3.5⟩
  vertex_y f = -ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_below_x_axis_l578_57889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_naturals_covered_l578_57814

def a (n : ℕ) : ℕ := 3 * n^2 - 2 * n

noncomputable def f (n : ℕ) : ℕ := ⌊(n : ℝ) + Real.sqrt (n / 3 : ℝ) + 1/2⌋.toNat

theorem all_naturals_covered (m : ℕ) :
  (∃ n : ℕ, a n = m) ∨ (∃ k : ℕ, f k = m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_naturals_covered_l578_57814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l578_57821

-- Define the points on the original line
noncomputable def point1 : ℝ × ℝ := (3, -3)
noncomputable def point2 : ℝ × ℝ := (-4, 5)

-- Define the slope of the original line
noncomputable def original_slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)

-- Define the slope of the perpendicular line
noncomputable def perpendicular_slope : ℝ := -1 / original_slope

-- Theorem statement
theorem perpendicular_line_slope :
  perpendicular_slope = 7/8 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l578_57821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_elimination_tournament_games_tournament_with_23_teams_l578_57857

/-- The number of games played in a single-elimination tournament with n teams. -/
def number_of_games (n : ℕ) : ℕ :=
  n - 1

/-- In a single-elimination tournament with no ties, the number of games played
    is equal to the number of teams minus 1. -/
theorem single_elimination_tournament_games (n : ℕ) (h : n > 0) :
  number_of_games n = n - 1 := by
  rfl

/-- For a tournament with 23 teams, 22 games are played. -/
theorem tournament_with_23_teams :
  number_of_games 23 = 22 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_elimination_tournament_games_tournament_with_23_teams_l578_57857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_side_heavier_l578_57852

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Theorem stating that the volume of a sphere with radius 8 is greater than
    the sum of volumes of spheres with radii 3 and 5 -/
theorem right_side_heavier :
  sphere_volume 8 > sphere_volume 3 + sphere_volume 5 := by
  -- Expand the definition of sphere_volume
  unfold sphere_volume
  -- Simplify the inequality
  simp [Real.pi_pos]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_side_heavier_l578_57852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_percentage_theorem_l578_57806

/-- The percentage of Tom's income put aside for other purposes -/
noncomputable def income_percentage_put_aside (item_cost1 item_cost2 item_cost3 hourly_wage work_hours : ℝ) : ℝ :=
  let total_cost := item_cost1 + item_cost2 + item_cost3
  let total_earned := hourly_wage * work_hours
  let amount_put_aside := total_earned - total_cost
  (amount_put_aside / total_earned) * 100

/-- Theorem stating that the percentage of Tom's income put aside is approximately 9.68% -/
theorem income_percentage_theorem :
  ∃ ε > 0, |income_percentage_put_aside 25.35 70.69 85.96 6.50 31 - 9.68| < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_percentage_theorem_l578_57806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l578_57844

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log (1/2) - 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l578_57844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_in_target_function_l578_57833

noncomputable section

open Real

/-- Definition of the original function -/
def f (x : ℝ) : ℝ := sin x

/-- Definition of the target function -/
def g (x : ℝ) : ℝ := (1/2) * sin (x/2 + π/3)

/-- Transformation 6: Shift left by π/3 -/
def transform6 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x + π/3)

/-- Transformation 2: Double horizontal coordinates -/
def transform2 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x/2)

/-- Transformation 3: Halve vertical coordinates -/
def transform3 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ (1/2) * f x

/-- Theorem stating that applying transformations 6, 2, and 3 to f results in g -/
theorem transformations_result_in_target_function :
  ∀ x, (transform3 (transform2 (transform6 f))) x = g x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_in_target_function_l578_57833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l578_57835

/-- The area of a triangle given by three points in 2D space -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let (x₃, y₃) := R
  (1/2) * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

/-- The area of triangle PQR is 19 square units -/
theorem triangle_PQR_area :
  triangleArea (-3, 4) (1, 7) (3, -1) = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l578_57835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_workshop_skills_l578_57815

-- Define the total number of participants
def total : ℕ := 120

-- Define the skills as predicates
def poetry : (Fin total) → Prop := sorry
def painting : (Fin total) → Prop := sorry
def photography : (Fin total) → Prop := sorry

theorem theater_workshop_skills 
  (no_poetry : ℕ) 
  (no_painting : ℕ) 
  (no_photo : ℕ) 
  (h_no_poetry : no_poetry = 52)
  (h_no_painting : no_painting = 75)
  (h_no_photo : no_photo = 38)
  (h_no_all_three : ∀ p : Fin total, ¬(poetry p ∧ painting p ∧ photography p)) :
  (total - no_poetry) + (total - no_painting) + (total - no_photo) - total = 75 := by
  sorry

#check theater_workshop_skills

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_workshop_skills_l578_57815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lego_figures_proof_l578_57845

noncomputable def figure_pieces : Nat → Nat := sorry

theorem lego_figures_proof (total_pieces : Nat) 
  (smallest_three_sum : Nat) (largest_three_sum : Nat) 
  (h1 : total_pieces = 80)
  (h2 : smallest_three_sum = 14)
  (h3 : largest_three_sum = 43)
  (h4 : ∀ (i j : Nat), i ≠ j → figure_pieces i ≠ figure_pieces j) :
  ∃ (n : Nat), n = 8 ∧ 
    (∀ (i : Nat), i < n → figure_pieces i > 0) ∧
    figure_pieces 0 + figure_pieces 1 + figure_pieces 2 = smallest_three_sum ∧
    figure_pieces (n-3) + figure_pieces (n-2) + figure_pieces (n-1) = largest_three_sum ∧
    (∀ (i : Nat), i < n-1 → figure_pieces i < figure_pieces (i+1)) ∧
    figure_pieces (n-1) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lego_figures_proof_l578_57845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l578_57866

def initial_investment : ℝ := 1000
def investment_period : ℕ := 6
def interest_rates : List ℝ := [0.05, 0.06, 0.04, 0.05, 0.06, 0.07]

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def final_amount : ℝ :=
  List.foldl compound_interest initial_investment interest_rates

theorem investment_growth :
  |final_amount - 1378.50| < 0.01 := by
  -- Proof goes here
  sorry

#eval final_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l578_57866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_minus_b_range_l578_57896

theorem cos_2a_minus_b_range (α β : ℝ) (h1 : 0 ≤ α ∧ α ≤ π) (h2 : 0 ≤ β ∧ β ≤ π)
  (h3 : Real.sin α * Real.cos β - Real.cos α * Real.sin β = 1) :
  ∃ x, Real.cos (2 * α - β) = x ∧ -1 ≤ x ∧ x ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_minus_b_range_l578_57896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_felix_hit_twelve_l578_57803

/-- Represents a player in the dart game -/
inductive Player
| Adam
| Bella
| Carlos
| Diana
| Eva
| Felix

/-- Represents a score in the dart game (1 to 15) -/
def Score := Fin 15

/-- A function that returns the total score for a given player -/
def total_score (p : Player) : ℕ :=
  match p with
  | Player.Adam => 23
  | Player.Bella => 19
  | Player.Carlos => 21
  | Player.Diana => 30
  | Player.Eva => 24
  | Player.Felix => 27

/-- A function that returns the three scores for a given player -/
def player_scores (p : Player) : (Score × Score × Score) :=
  sorry

/-- The theorem stating that Felix is the only player who hit the 12-point region -/
theorem felix_hit_twelve :
  ∃! (p : Player), ∃ (s1 s2 : Score), 
    (player_scores p).1 = ⟨11, sorry⟩ ∧ 
    s1 ≠ s2 ∧ 
    s1 ≠ ⟨11, sorry⟩ ∧ 
    s2 ≠ ⟨11, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_felix_hit_twelve_l578_57803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l578_57825

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (μ σ : ℝ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_probability 
  (σ : ℝ) 
  (ξ : normal_distribution 3 σ) 
  (h : P 3 σ {x | x > 4} = 0.2) : 
  P 3 σ {x | 3 < x ∧ x ≤ 4} = 0.3 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l578_57825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_sum_l578_57893

/-- Triangle ABC with centroid G -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices -/
noncomputable def s₁ (t : Triangle) : ℝ :=
  distance t.G t.A + distance t.G t.B + distance t.G t.C

/-- Perimeter of the triangle -/
noncomputable def s₂ (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: s₁ is equal to one-third of s₂ for any triangle -/
theorem centroid_distance_sum (t : Triangle) : s₁ t = (1/3) * s₂ t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_sum_l578_57893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l578_57898

/-- The optimal speed that minimizes transportation cost -/
noncomputable def optimal_speed (s a : ℝ) : ℝ :=
  if 0 < a ∧ a < 144 then 5 * Real.sqrt a else 60

theorem optimal_speed_minimizes_cost (s a : ℝ) (hs : s > 0) (ha : a > 0) :
  let y : ℝ → ℝ := λ v => s * (a / v + v / 25)
  let v := optimal_speed s a
  ∀ u, 0 < u ∧ u ≤ 60 → y v ≤ y u := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l578_57898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l578_57892

/-- The probability that no two points out of four randomly chosen points on a circle 
    form an obtuse triangle with the circle's center -/
noncomputable def probability_no_obtuse_triangle : ℝ := 1 / 64

/-- Four points chosen uniformly at random on a circle -/
def random_points_on_circle : ℕ := 4

/-- Theorem stating that the probability of no obtuse triangle is 1/64 -/
theorem probability_theorem :
  probability_no_obtuse_triangle = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l578_57892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l578_57871

noncomputable def vector_a : ℝ × ℝ := (2, -1)
def point_A : ℝ × ℝ := (1, -2)

def same_direction (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = v.2 * w.1 ∧ k ≠ 0

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
    let AB := (B.1 - point_A.1, B.2 - point_A.2)
    same_direction AB vector_a →
    vector_length AB = 3 * Real.sqrt 5 →
    B = (7, -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l578_57871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_product_bound_l578_57846

theorem permutation_product_bound (a : Fin 1985 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 1986) : 
  (Finset.range 1985).sup (λ k ↦ (k + 1) * a k) ≥ 993^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_product_bound_l578_57846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l578_57883

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 3

/-- Represents the number of elderly people -/
def num_elderly : ℕ := 2

/-- Represents the total number of people -/
def total_people : ℕ := num_volunteers + num_elderly

/-- Represents the number of arrangements -/
def num_arrangements : ℕ := (total_people - 1) * (total_people - 2) * (total_people - 3) * num_elderly

theorem arrangement_count : num_arrangements = 48 := by
  rfl

#eval num_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l578_57883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_right_triangular_prism_l578_57860

/-- A right triangular prism with base sides a and b, and height h -/
structure RightTriangularPrism where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- The sum of areas of two lateral faces and one base -/
noncomputable def sumOfAreas (p : RightTriangularPrism) : ℝ :=
  p.a * p.h + p.b * p.h + 1/2 * p.a * p.b

/-- The volume of the prism -/
noncomputable def volume (p : RightTriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h

/-- Theorem stating that the maximum volume of a right triangular prism
    with sum of areas of two lateral faces and one base equal to 30 is 50 -/
theorem max_volume_of_right_triangular_prism :
  ∃ (p : RightTriangularPrism), sumOfAreas p = 30 ∧
  ∀ (q : RightTriangularPrism), sumOfAreas q = 30 → volume q ≤ volume p ∧
  volume p = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_right_triangular_prism_l578_57860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_next_birthday_is_14_l578_57817

-- Define the ages as real numbers
variable (m s d : ℝ)

-- Define the relationships between ages
def mary_sally_relation (m s : ℝ) : Prop := m = 1.25 * s
def sally_danielle_relation (s d : ℝ) : Prop := s = 0.5 * d
def sum_of_ages (m s d : ℝ) : Prop := m + s + d = 42

-- Define Mary's next birthday age
noncomputable def mary_next_birthday (m : ℝ) : ℤ := ⌊m⌋ + 1

-- Theorem statement
theorem mary_next_birthday_is_14 
  (h1 : mary_sally_relation m s)
  (h2 : sally_danielle_relation s d)
  (h3 : sum_of_ages m s d) :
  mary_next_birthday m = 14 := by
  sorry

-- Note: The proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_next_birthday_is_14_l578_57817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_right_triangle_l578_57869

noncomputable def point1 : ℝ × ℝ := (5, -3)
noncomputable def point2 : ℝ × ℝ := (-7, 4)
noncomputable def point3 : ℝ × ℝ := (5, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def is_right_triangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2) ∨ (c^2 + a^2 = b^2)

theorem distance_and_right_triangle :
  (distance point1 point2 = Real.sqrt 193) ∧
  is_right_triangle point1 point2 point3 := by
  sorry

#check distance_and_right_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_right_triangle_l578_57869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l578_57847

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (1 + |x|) - 1 / (1 + x^2)

/-- Theorem stating the range of x for which f(x) > f(2x-1) -/
theorem f_inequality_range (x : ℝ) : f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l578_57847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_measurement_theorem_l578_57877

/-- Represents the properties of the rectangle measurement process -/
structure RectangleMeasurement where
  nominal_length : ℝ
  nominal_width : ℝ
  length_std_dev : ℝ
  width_std_dev : ℝ

/-- The expected area of the rectangle -/
def expected_area (r : RectangleMeasurement) : ℝ :=
  r.nominal_length * r.nominal_width

/-- The standard deviation of the area in square centimeters -/
noncomputable def area_std_dev (r : RectangleMeasurement) : ℝ :=
  100 * Real.sqrt ((r.nominal_width^2 * r.length_std_dev^2) + 
                   (r.nominal_length^2 * r.width_std_dev^2))

/-- The main theorem about the rectangle measurements -/
theorem rectangle_measurement_theorem (r : RectangleMeasurement) 
  (h1 : r.nominal_length = 2)
  (h2 : r.nominal_width = 1)
  (h3 : r.length_std_dev = 0.003)
  (h4 : r.width_std_dev = 0.002) :
  expected_area r = 2 ∧ 
  abs (area_std_dev r - 63) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_measurement_theorem_l578_57877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_problems_l578_57865

structure AMC10 where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unanswered_points : ℤ
  attempted_problems : ℕ
  target_score : ℤ

def score (exam : AMC10) (correct : ℕ) : ℤ :=
  correct * exam.correct_points +
  (exam.attempted_problems - correct) * exam.incorrect_points +
  (exam.total_problems - exam.attempted_problems) * exam.unanswered_points

theorem min_correct_problems (exam : AMC10) :
  exam.total_problems = 30 ∧
  exam.correct_points = 7 ∧
  exam.incorrect_points = -1 ∧
  exam.unanswered_points = 2 ∧
  exam.attempted_problems = 28 ∧
  exam.target_score = 150 →
  ∃ n : ℕ, n = 22 ∧ ∀ m : ℕ, score exam m ≥ exam.target_score ↔ m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_problems_l578_57865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_range_l578_57862

noncomputable def f (x a : ℝ) : ℝ := (4 : ℝ)^x - 2 * (2 : ℝ)^(x + 1) + a

theorem function_properties (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f x a ≥ 1 ∧ (∃ y ∈ Set.Icc 0 3, f y a = 1)) ↔ a = 5 :=
by sorry

theorem function_range (a : ℝ) :
  (∃ x ∈ Set.Icc 0 3, f x a ≥ 33) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_range_l578_57862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l578_57804

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 9) + 1 / (x^2 - 6*x + 21) + Real.cos (2 * Real.pi * x)

theorem f_max_value : ∀ x : ℝ, f x ≤ 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l578_57804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_velocity_calculation_l578_57822

/-- Represents the velocity of a current in a river. -/
def current_velocity : ℝ → Prop := sorry

/-- Represents the rowing speed of a person in still water. -/
def rowing_speed : ℝ → Prop := sorry

/-- Represents the total time taken for a round trip. -/
def total_time : ℝ → Prop := sorry

/-- Represents the total distance of the round trip. -/
def total_distance : ℝ → Prop := sorry

/-- 
Given:
- A person can row at 10 kmph in still water
- It takes 20 hours for a round trip
- The total distance of the round trip is 96 km
Prove that the velocity of the current is √52 kmph.
-/
theorem current_velocity_calculation 
  (h1 : rowing_speed 10)
  (h2 : total_time 20)
  (h3 : total_distance 96) :
  current_velocity (Real.sqrt 52) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_velocity_calculation_l578_57822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l578_57867

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem graph_translation (x : ℝ) :
  g x = f (x + Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l578_57867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l578_57812

/-- The complex number -3 - i√8 -/
noncomputable def z : ℂ := -3 - Complex.I * Real.sqrt 8

/-- The monic quadratic polynomial with real coefficients -/
def p (x : ℂ) : ℂ := x^2 + 6*x + 17

theorem monic_quadratic_with_complex_root :
  (∀ x : ℝ, p x = x^2 + 6*x + 17) ∧
  (∀ a b : ℝ, (∀ x : ℝ, p x = x^2 + a*x + b) → a = 6 ∧ b = 17) ∧
  (p z = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l578_57812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l578_57885

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the ellipse
def is_valid_ellipse (e : Ellipse) : Prop :=
  e.a > 0 ∧ e.b > 0 ∧ e.a > e.b ∧ e.c^2 = e.a^2 - e.b^2

def has_eccentricity (e : Ellipse) (ecc : ℝ) : Prop :=
  e.c / e.a = ecc

def passes_through (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the theorem
theorem ellipse_properties (e : Ellipse) :
  is_valid_ellipse e →
  has_eccentricity e (Real.sqrt 3 / 2) →
  passes_through e (-Real.sqrt 3) (1/2) →
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (let f := Real.sqrt 3;
   let chord_length := 2 * e.a - (Real.sqrt 3 / 2) * (8 * Real.sqrt 3 / 5);
   chord_length = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l578_57885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_transformation_l578_57811

/-- The transformation that maps the ellipse to the circle -/
structure Transformation where
  lambda : ℝ
  mu : ℝ
  lambda_pos : lambda > 0
  mu_pos : mu > 0

/-- The ellipse equation -/
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- The circle equation after transformation -/
def is_circle (x_prime y_prime : ℝ) : Prop :=
  x_prime^2 + y_prime^2 = 9

/-- The theorem stating the correct transformation parameters -/
theorem correct_transformation :
  ∃ (φ : Transformation),
    (∀ x y x_prime y_prime, is_ellipse x y → x_prime = φ.lambda * x → y_prime = φ.mu * y → is_circle x_prime y_prime) ∧
    φ.lambda = 1 ∧ φ.mu = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_transformation_l578_57811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l578_57891

theorem sin_alpha_plus_pi_fourth (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : 0 < α ∧ α < π/2) :
  Real.sin (α + π/4) = 17 * Real.sqrt 2 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l578_57891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_error_probability_l578_57890

-- Define the normal distribution
def normal_distribution (μ σ : ℝ) : Type := sorry

-- Define the probability function
noncomputable def probability {α : Type} (X : Set α) : ℝ := sorry

-- Define the random variable ξ
def ξ : ℝ := sorry

-- Theorem statement
theorem length_error_probability :
  let σ : ℝ := 3
  let μ : ℝ := 0
  let X := normal_distribution μ (σ^2)
  probability {x : ℝ | -σ < x ∧ x < σ} = 0.6826 →
  probability {x : ℝ | -2*σ < x ∧ x < 2*σ} = 0.9544 →
  probability {x : ℝ | 3 < x ∧ x < 6} = 0.1359 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_error_probability_l578_57890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l578_57808

/-- A line in 2D space --/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space --/
structure Point where
  x : ℚ
  y : ℚ

/-- The area of a triangle formed by three points --/
def triangleArea (p1 p2 p3 : Point) : ℚ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem triangle_area_zero (l1 l2 : Line) (p : Point) :
  l1.slope = 1/3 →
  l2.slope = 3 →
  p.x = 3 →
  p.y = 3 →
  let p1 := p
  let p2 : Point := { x := 8, y := l1.slope * 8 + l1.intercept }
  let p3 : Point := { x := 8, y := l2.slope * 8 + l2.intercept }
  triangleArea p1 p2 p3 = 0 := by
  sorry

#eval triangleArea { x := 3, y := 3 } { x := 8, y := 14/3 } { x := 8, y := 18 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l578_57808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_fraction_sum_representations_l578_57832

theorem unit_fraction_sum_representations :
  {(x, y, z) : ℕ × ℕ × ℕ | 1 = (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z} =
  {(2, 3, 6), (2, 4, 4), (3, 3, 3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_fraction_sum_representations_l578_57832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l578_57881

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the conditions
axiom a_positive : a > 0
axiom b_positive : b > 0

-- Define the asymptote condition
axiom asymptote_point : b / a * Real.sqrt 2 = Real.sqrt 6

-- Define eccentricity
noncomputable def eccentricity := Real.sqrt (1 + (b / a)^2)

-- Theorem statement
theorem hyperbola_eccentricity : eccentricity a b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l578_57881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l578_57894

/-- Given a line y = (3/4)x + 6, prove that a parallel line 4 units away has the equation y = (3/4)x + 1 -/
theorem parallel_line_equation (x y : ℝ) : 
  let original_line := fun x => (3/4) * x + 6
  let parallel_line := fun x => (3/4) * x + 1
  let distance := fun (f g : ℝ → ℝ) => |f 0 - g 0| / Real.sqrt (1 + ((3/4) ^ 2))
  (∀ x, y = original_line x) →
  (∀ x, y = parallel_line x) →
  distance original_line parallel_line = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l578_57894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_interval_l578_57851

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then
    -x^2 + 10*x + 5
  else if 5 < x ∧ x ≤ 20 then
    -2*x + 40
  else
    0  -- undefined for other x values

theorem tangent_line_and_interval :
  -- Part 1: Equation of the tangent line at (1, f(1))
  (∃ (m b : ℝ), m * 1 - f 1 + b = 0 ∧
    ∀ x, m * x - f x + b = 0 ↔ x = 1) ∧
  -- Part 2: Interval where f(x) > 14
  (∀ x, 1 < x ∧ x < 13 ↔ f x > 14) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_interval_l578_57851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_5cos_values_l578_57874

theorem sin_plus_5cos_values (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) :
  Real.sin x + 5 * Real.cos x = -1/2 ∨ Real.sin x + 5 * Real.cos x = 17/13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_5cos_values_l578_57874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_inspected_half_l578_57841

/-- The fraction of products Jane inspected -/
noncomputable def jane_fraction (john_reject_rate jane_reject_rate total_reject_rate : ℝ) : ℝ :=
  1 - (total_reject_rate - jane_reject_rate) / (john_reject_rate - jane_reject_rate)

/-- Theorem stating that Jane inspected half of the products -/
theorem jane_inspected_half :
  jane_fraction 0.007 0.008 0.0075 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_inspected_half_l578_57841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gatorade_bottle_price_l578_57810

-- Define the sales and conditions
def cupcakes_sold : ℕ := 120
def cupcake_price : ℚ := 2
def cookies_sold : ℕ := 100
def cookie_price : ℚ := 1/2
def candy_bars_sold : ℕ := 25
def candy_bar_price : ℚ := 3/2
def granola_bars_sold : ℕ := 15
def granola_bar_price : ℚ := 1
def cupcake_discount : ℚ := 1/10
def granola_discount : ℚ := 1/20
def exchange_rate : ℚ := 11/10
def sales_tax : ℚ := 1/20
def soccer_team_share : ℚ := 1/10
def basketballs_bought : ℕ := 4
def gatorade_bottles : ℕ := 35
def basketball_discount : ℚ := 3/20
def basketball_original_price : ℚ := 60

-- Define the theorem
theorem gatorade_bottle_price :
  ∃ (price : ℚ), (price * 100).num / (price * 100).den = 271 ∧
  (let total_sales := cupcakes_sold * cupcake_price * (1 - cupcake_discount) +
                      cookies_sold * cookie_price +
                      candy_bars_sold * candy_bar_price +
                      granola_bars_sold * granola_bar_price * (1 - granola_discount);
   let usd_sales := total_sales * exchange_rate;
   let after_tax := usd_sales * (1 - sales_tax);
   let after_sharing := after_tax * (1 - soccer_team_share);
   let basketball_cost := basketballs_bought * basketball_original_price * (1 - basketball_discount);
   let gatorade_total := after_sharing - basketball_cost;
   gatorade_total / gatorade_bottles = price) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gatorade_bottle_price_l578_57810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_sum_of_f_geq_4_l578_57836

/-- The function f(x) defined as |x + a| + |x + 1/a| where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 1/a|

/-- Theorem stating the solution set of f(x) > 3 when a = 2 -/
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x > 3} = {x : ℝ | x < -11/4 ∨ x > 1/4} := by sorry

/-- Theorem proving that f(m) + f(-1/m) ≥ 4 for any real m and a > 0 -/
theorem sum_of_f_geq_4 (a : ℝ) (h : a > 0) (m : ℝ) :
  f a m + f a (-1/m) ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_sum_of_f_geq_4_l578_57836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutual_card_senders_l578_57838

/-- Represents a class of pupils and their card-sending behavior -/
structure CardSendingClass where
  n : ℕ  -- number of pupils
  m : ℕ  -- number of cards each pupil sends
  h1 : n > 1  -- there are at least 2 pupils
  h2 : m < n  -- each pupil sends fewer cards than the total number of pupils

/-- Predicate for a pupil sending a card to another pupil -/
def sends_card (c : CardSendingClass) (i j : ℕ) : Prop := sorry

/-- Predicate for whether at least two pupils send each other a card -/
def hasMutualCardSenders (c : CardSendingClass) : Prop :=
  ∃ i j, i ≠ j ∧ i < c.n ∧ j < c.n ∧ (sends_card c i j ∧ sends_card c j i)

/-- Theorem stating the condition for mutual card senders -/
theorem mutual_card_senders (c : CardSendingClass) :
  hasMutualCardSenders c ↔ c.m > (c.n - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutual_card_senders_l578_57838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escape_possible_l578_57863

/-- Represents a door type -/
inductive DoorType
  | Normal
  | Inversive

/-- Represents a grid of rooms -/
def Grid (n : ℕ) := Fin n → Fin n → Type

/-- Represents a configuration of doors in the grid -/
def DoorConfig (n : ℕ) := 
  (Fin n → Fin n → DoorType) × (Fin n → Fin n → DoorType)

/-- A path through the grid -/
def GridPath (n : ℕ) := List (Fin n × Fin n)

/-- Checks if a path is Hamiltonian (visits each room exactly once) -/
def isHamiltonian (n : ℕ) (path : GridPath n) : Prop :=
  path.length = n * n ∧ path.Nodup

/-- Counts the number of inversive doors passed in a path -/
def countInversiveDoors (n : ℕ) (config : DoorConfig n) (path : GridPath n) : ℕ :=
  sorry

/-- The main theorem: there exists a Hamiltonian path and a door configuration
    such that passing through an even number of inversive doors is possible -/
theorem escape_possible (n : ℕ) : 
  ∃ (path : GridPath n) (config : DoorConfig n), 
    isHamiltonian n path ∧ 
    Even (countInversiveDoors n config path) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escape_possible_l578_57863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l578_57899

-- Define the points
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (-4, 2)
def R : ℝ × ℝ := (-3, 7)
def S : ℝ × ℝ := (2, 6)

-- Define the square
def square : Set (ℝ × ℝ) := {P, Q, R, S}

-- Define a function to calculate the area of the square
def squareArea (s : Set (ℝ × ℝ)) : ℝ := 26

-- Theorem statement
theorem square_area : squareArea square = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l578_57899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_of_equation_l578_57843

theorem root_interval_of_equation : ∃! x : ℝ, x ∈ Set.Ioo 0 1 ∧ 2^x = 2 - x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_of_equation_l578_57843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_function_l578_57820

theorem max_value_trigonometric_function (φ : ℝ) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, Real.sin (x + φ) - 2 * Real.sin φ * Real.cos x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_function_l578_57820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_l578_57842

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (1 + x)) / (x^2 * Real.sqrt x)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := -2/3 * (Real.sqrt ((1 + x) / x))^3

-- Theorem statement
theorem integral_of_f (x : ℝ) (hx : x > 0) : 
  deriv F x = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_l578_57842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l578_57837

-- Define the parametric equations for lines l₁ and l₂
noncomputable def l₁ (t k : ℝ) : ℝ × ℝ := (2 - t, k * t)
noncomputable def l₂ (m k : ℝ) : ℝ × ℝ := (-2 + m, m / k)

-- Define the locus C₁
def C₁ : Set (ℝ × ℝ) := {p | ∃ (k t m : ℝ), k ≠ 0 ∧ l₁ t k = l₂ m k ∧ p = l₁ t k}

-- Define the curve C₂ in polar form
def C₂ : Set (ℝ × ℝ) := {p | ∃ (r θ : ℝ), r = 4 * Real.sin θ ∧ p = (r * Real.cos θ, r * Real.sin θ)}

-- Theorem statement
theorem intersection_points : 
  (C₁ = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}) ∧ 
  (C₁ ∩ C₂ = {(2 * Real.cos (π/6), 2 * Real.sin (π/6)), (2 * Real.cos (5*π/6), 2 * Real.sin (5*π/6))}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l578_57837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pair_probability_l578_57884

/-- The number of socks -/
def total_socks : ℕ := 8

/-- The number of colors -/
def num_colors : ℕ := 4

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The number of socks drawn -/
def socks_drawn : ℕ := 4

/-- The probability of drawing exactly one pair of socks with the same color -/
theorem one_pair_probability : 
  (Nat.choose total_socks socks_drawn : ℚ)⁻¹ * 
  (Nat.choose num_colors 3 * Nat.choose 3 1 * socks_per_color * socks_per_color : ℚ) = 24/35 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pair_probability_l578_57884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_divisibility_conditions_l578_57805

theorem smallest_number_with_divisibility_conditions : ∃! x : ℕ, 
  (∀ d : ℕ, d ∈ [12, 16, 18, 21, 28, 35, 39] → (x - 3) % d = 0) ∧ 
  (x + 5) % 45 = 0 ∧
  (∀ y : ℕ, y < x → ¬((∀ d : ℕ, d ∈ [12, 16, 18, 21, 28, 35, 39] → (y - 3) % d = 0) ∧ (y + 5) % 45 = 0)) ∧
  x = 65523 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_divisibility_conditions_l578_57805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_period_pi_f_is_even_l578_57854

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := |Real.cos (2 * x)| + Real.cos (|x|)

-- Statement 1: π is not a period of f(x)
theorem not_period_pi : ∃ x : ℝ, f (x + Real.pi) ≠ f x := by sorry

-- Statement 2: f(x) is an even function (symmetric about the y-axis)
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_period_pi_f_is_even_l578_57854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l578_57859

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  short_side : ℝ
  long_side : ℝ
  is_isosceles : short_side + short_side > long_side

/-- Calculates the perimeter of a triangle given its side lengths -/
noncomputable def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.short_side + t.long_side

/-- Represents the scaling factor between two similar triangles -/
noncomputable def scaling_factor (t1 t2 : IsoscelesTriangle) : ℝ := t2.short_side / t1.short_side

theorem similar_triangle_perimeter 
  (small : IsoscelesTriangle) 
  (large : IsoscelesTriangle) 
  (h_small : small.short_side = 12 ∧ small.long_side = 24) 
  (h_large : large.short_side = 18) 
  (h_similar : scaling_factor small large = large.short_side / small.short_side) :
  perimeter large = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l578_57859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cylinder_cone_l578_57873

/-- Sphere type with necessary properties --/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  touches_cylinder : ℝ → ℝ → Prop
  touches_cone : ℝ → ℝ → Prop
  touches_cylinder_base : ℝ → Prop

/-- The radius of a sphere inscribed in a cylinder and cone configuration --/
theorem sphere_radius_in_cylinder_cone (cylinder_radius cylinder_height : ℝ) 
  (hr : cylinder_radius = 12) (hh : cylinder_height = 30) :
  let cone_height := cylinder_height
  let sphere_radius := 21 - (1/2) * Real.sqrt 1044
  ∃ (s : Sphere), 
    s.radius = sphere_radius ∧ 
    s.touches_cylinder cylinder_radius cylinder_height ∧
    s.touches_cone cylinder_radius cone_height ∧
    s.touches_cylinder_base cylinder_radius :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cylinder_cone_l578_57873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_l578_57801

theorem circle_arrangement (r R x : ℝ) (hr : 0 < r) (hR : r < R) : 
  (x = (R - r) / 2 ∧ 
   12 * x = R * (Real.sqrt 6 - Real.sqrt 2) / 2) → 
  (R / r = (4 + Real.sqrt 6 - Real.sqrt 2) / (4 - Real.sqrt 6 + Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_l578_57801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteNestedRadical_value_l578_57858

/-- The value of the infinite nested radical √(3 - √(3 - √(3 - √(3 - ...)))) -/
noncomputable def infiniteNestedRadical : ℝ := Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3)))

/-- The infinite nested radical equals (√13 - 1) / 2 -/
theorem infiniteNestedRadical_value : infiniteNestedRadical = (Real.sqrt 13 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteNestedRadical_value_l578_57858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l578_57831

/-- The positive difference between the roots of 5x^2 - 11x + 2 = 0 -/
noncomputable def root_difference : ℝ :=
  |((11 + Real.sqrt 81) / 10) - ((11 - Real.sqrt 81) / 10)|

/-- p is the square of the numerator in the simplified root difference -/
def p : ℕ := 81

/-- q is the denominator in the simplified root difference -/
def q : ℕ := 5

/-- p is not divisible by the square of any prime number -/
axiom p_square_free : ∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ p)

theorem quadratic_root_difference :
  root_difference = Real.sqrt p / q ∧ p + q = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l578_57831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_m_range_l578_57849

/-- The function f(x) = (m + ln x) / x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

/-- The theorem stating the equivalence between f(x) < mx for all x > 1 and m ≥ 1/2 -/
theorem f_inequality_iff_m_range (m : ℝ) :
  (∀ x > 1, f m x < m * x) ↔ m ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_m_range_l578_57849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_one_ninth_l578_57816

-- Define the points
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (8, 12)
def C : ℝ × ℝ := (14, 0)
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem area_ratio_one_ninth :
  (triangleArea X Y Z) / (triangleArea A B C) = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_one_ninth_l578_57816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_median_length_l578_57861

theorem min_median_length (h : ℝ) (h_pos : h > 0) : 
  ∃ (median : ℝ), median = (3/2) * h ∧ 
  ∀ (other_median : ℝ), other_median ≥ median := by
  -- Proof goes here
  sorry

#check min_median_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_median_length_l578_57861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_range_of_m_part2_range_of_m_l578_57886

-- Define the determinant function
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define f(x) using the determinant
def f (m x : ℝ) : ℝ := det (m * x) m (2 * x) (x + 1)

-- Part 1: Range of m for which f(x) < 1 for all real x
theorem part1_range_of_m : 
  {m : ℝ | ∀ x, f m x < 1} = Set.Ioc (-4) 0 := by sorry

-- Part 2: Range of m for which f(x) < 6 - m for all x in [1,3]
theorem part2_range_of_m :
  {m : ℝ | ∀ x ∈ Set.Icc 1 3, f m x < 6 - m} = Set.Iio (6/7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_range_of_m_part2_range_of_m_l578_57886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_is_zero_l578_57875

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define points F and P
def F : ℝ × ℝ := (1, 0)
def P : ℝ × ℝ := (4, 0)

-- Define a line with non-zero slope passing through F
def line_through_F (t : ℝ) (x y : ℝ) : Prop := x = t * y + 1 ∧ t ≠ 0

-- Define the slope of a line passing through P and another point
noncomputable def slope_from_P (x y : ℝ) : ℝ := y / (x - 4)

-- Theorem statement
theorem sum_of_slopes_is_zero 
  (t : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : C A.1 A.2 ∧ line_through_F t A.1 A.2) 
  (hB : C B.1 B.2 ∧ line_through_F t B.1 B.2) 
  (hAB : A ≠ B) :
  slope_from_P A.1 A.2 + slope_from_P B.1 B.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_is_zero_l578_57875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_properties_l578_57827

/-- A triangular pyramid with three edges of length 1 and three edges of length a -/
structure TriangularPyramid (a : ℝ) where
  edge_1 : ℝ := 1
  edge_2 : ℝ := 1
  edge_3 : ℝ := 1
  edge_4 : ℝ := a
  edge_5 : ℝ := a
  edge_6 : ℝ := a

/-- Condition that none of the faces is an equilateral triangle -/
def no_equilateral_face (a : ℝ) : Prop :=
  a ≠ 1 ∧ a > 0 ∧ a < 2

/-- The range of a for which the pyramid has no equilateral faces -/
noncomputable def valid_range (a : ℝ) : Prop :=
  ((-1 + Real.sqrt 5) / 2 < a) ∧ (a < (1 + Real.sqrt 5) / 2) ∧ a ≠ 1

/-- The volume of the pyramid -/
noncomputable def volume (a : ℝ) : ℝ :=
  (1 / 12) * Real.sqrt ((a^2 + 1) * (3 * a^2 - 1 - a^4))

theorem triangular_pyramid_properties (a : ℝ) (p : TriangularPyramid a) 
    (h : no_equilateral_face a) : 
  valid_range a ∧ volume a = (1 / 12) * Real.sqrt ((a^2 + 1) * (3 * a^2 - 1 - a^4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_properties_l578_57827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_portion_is_25_percent_l578_57882

/-- Represents a square that can be divided into smaller squares -/
structure DivisibleSquare where
  area : ℕ
  num_divisions : ℕ

/-- Represents the shaded portion of a square -/
def shaded_portion (s : DivisibleSquare) : ℚ := 1 / s.num_divisions

/-- The theorem stating that for a square divided into 4 equal parts,
    with one part shaded, the shaded portion is 25% of the total area -/
theorem shaded_portion_is_25_percent (s : DivisibleSquare) 
  (h : s.num_divisions = 4) : shaded_portion s = 1/4 := by
  rw [shaded_portion, h]
  norm_num

#check shaded_portion_is_25_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_portion_is_25_percent_l578_57882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_zero_l578_57823

theorem existence_of_zero (f : ℝ → ℝ) (a b : ℝ) :
  ContinuousOn f (Set.Icc a b) →
  a < b →
  f a * f b < 0 →
  ∃ x ∈ Set.Ioo a b, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_zero_l578_57823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l578_57802

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c

def is_acute_triangle (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  (Real.sin t.A)^2 + (Real.sin t.C)^2 = (Real.sin t.B)^2 + (Real.sin t.A) * (Real.sin t.C)

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : is_acute_triangle t)
  (h3 : satisfies_condition t)
  (h4 : t.b = Real.sqrt 3) :
  t.B = Real.pi/3 ∧ 0 < 2*t.a - t.c ∧ 2*t.a - t.c < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l578_57802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equality_exists_l578_57856

-- Define the set S
def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a partition type
def Partition := Nat → Nat

-- Theorem statement
theorem partition_equality_exists (π π' : Partition) : 
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ π x = π y ∧ π' x = π' y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equality_exists_l578_57856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sphere_distance_theorem_l578_57848

/-- The distance between a vertex of an equilateral triangle and the center of a sphere -/
noncomputable def triangle_sphere_distance (l r : ℝ) : ℝ :=
  Real.sqrt (r^2 + l^2 / 4)

/-- Theorem: The distance between any vertex of an equilateral triangle with side length l
    placed horizontally over a sphere of radius r is √(r² + l²/4) -/
theorem triangle_sphere_distance_theorem (l r : ℝ) (hl : l > 0) (hr : r > 0) :
  ∃ D : ℝ, D = triangle_sphere_distance l r ∧ 
  D = Real.sqrt (r^2 + l^2 / 4) :=
by
  -- Existence of D
  use triangle_sphere_distance l r
  
  constructor
  -- First part of the conjunction
  · rfl
  
  -- Second part of the conjunction
  · rfl

#check triangle_sphere_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sphere_distance_theorem_l578_57848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_tile_position_l578_57887

/-- Represents a position on the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents a tile on the grid -/
inductive Tile
  | Small : Tile  -- 1x1 tile
  | Large : Tile  -- 1x3 tile

/-- Represents the grid configuration -/
def GridConfig := Position → Option Tile

/-- Checks if a position is in the center or adjacent to the boundaries -/
def isValidPosition (p : Position) : Prop :=
  p.row = 0 ∨ p.row = 3 ∨ p.row = 6 ∨
  p.col = 0 ∨ p.col = 3 ∨ p.col = 6

/-- Main theorem: The 1x1 tile must be in a valid position -/
theorem small_tile_position 
  (config : GridConfig)
  (h_small_tile : ∃! p, config p = some Tile.Small)
  (h_large_tiles : ∃ s : Finset Position, s.card = 16 ∧ ∀ p ∈ s, config p = some Tile.Large)
  : ∃ p, config p = some Tile.Small ∧ isValidPosition p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_tile_position_l578_57887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_theorem_l578_57809

-- Define the types for planes and lines
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Subspace ℝ V
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Subspace ℝ V

-- Define the relations
def perpendicular {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (L : Line V) (P : Plane V) : Prop := sorry
def parallel {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (L1 L2 : Line V) : Prop := sorry
def contained_in {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (L : Line V) (P : Plane V) : Prop := sorry
def planes_perpendicular {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (P1 P2 : Plane V) : Prop := sorry

-- State the theorem
theorem perpendicular_planes_theorem 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (α β : Plane V) (m n : Line V) : 
  perpendicular m α → parallel m n → contained_in n β → 
  planes_perpendicular α β :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_theorem_l578_57809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l578_57897

/-- The volume of a pyramid with a rectangular base and equal edge lengths from apex to corners. -/
theorem pyramid_volume (base_length base_width edge_length : ℝ) 
  (h_positive : 0 < base_length ∧ 0 < base_width ∧ 0 < edge_length) :
  (let base_area := base_length * base_width
   let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
   let height := Real.sqrt (edge_length^2 - (base_diagonal / 2)^2)
   (1/3 : ℝ) * base_area * height = 105 * Real.sqrt 7)
  ↔ base_length = 7 ∧ base_width = 9 ∧ edge_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l578_57897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l578_57834

-- Define complex number addition
def add (z w : ℂ) : ℂ := z + w

-- Define complex number multiplication
def mul (z w : ℂ) : ℂ := z * w

-- State the theorem
theorem complex_multiplication :
  mul (add 1 (2 * Complex.I)) (add 3 (-2 * Complex.I)) = 7 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l578_57834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_good_set_l578_57824

theorem at_least_one_good_set (A : Fin 100 → Set ℕ) 
  (h_union : (⋃ i, A i) = Set.univ) :
  ∃ i : Fin 100, ∃ n : ℕ, n > 0 ∧ Set.Infinite {p : ℕ × ℕ | p.1 ∈ A i ∧ p.2 ∈ A i ∧ p.1 - p.2 = n} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_good_set_l578_57824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l578_57800

noncomputable def circle_equation (x y : ℝ) := (x - 2)^2 + (y + 1)^2 = 9
def line_equation (x y : ℝ) := 3*x - 4*y + 5 = 0

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A*x₀ + B*y₀ + C| / Real.sqrt (A^2 + B^2)

theorem circle_tangent_to_line :
  ∀ x y : ℝ,
  circle_equation x y ↔
    (x - 2)^2 + (y + 1)^2 = (distance_point_to_line 2 (-1) 3 (-4) 5)^2 ∧
    distance_point_to_line 2 (-1) 3 (-4) 5 = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l578_57800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_l578_57878

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The distance between B' and C' is 3 -/
theorem distance_B'C'_equals_3 
  (A B C : Circle)
  (B' C' : Point)
  (h1 : distance A.center B.center = 3)
  (h2 : distance B.center C.center = 3)
  (h3 : distance C.center A.center = 3)
  (h4 : A.radius = 2)
  (h5 : B.radius = 1.5)
  (h6 : C.radius = 1.5)
  (h7 : B' ∈ {p | distance p A.center = A.radius ∧ distance p B.center = B.radius})
  (h8 : C' ∈ {p | distance p A.center = A.radius ∧ distance p C.center = C.radius})
  (h9 : distance B' C.center > C.radius)
  (h10 : distance C' B.center > B.radius)
  : distance B' C' = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_l578_57878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_400_units_l578_57879

-- Define the cost function G(x)
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the revenue function R(x)
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define the profit function f(x)
noncomputable def f (x : ℝ) : ℝ := R x - G x

-- Theorem statement
theorem max_profit_at_400_units :
  ∃ (max_profit : ℝ), f 4 = max_profit ∧ ∀ x, f x ≤ max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_400_units_l578_57879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l578_57853

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d₁ d₂ : ℝ) : ℝ :=
  |d₁ - d₂| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance between the planes x - 4y + 4z = 10 and 2x - 8y + 8z = 4 is 8/√33 -/
theorem distance_between_specific_planes :
  distance_between_planes 1 (-4) 4 10 2 = 8 / Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l578_57853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l578_57855

/-- Calculates the profit percentage without discounts given the production fee percentage,
    discount percentages, and profit percentage after discounts. -/
noncomputable def profit_percentage_without_discounts (production_fee_percent : ℝ) 
                                        (discount1_percent : ℝ) 
                                        (discount2_percent : ℝ) 
                                        (profit_after_discounts_percent : ℝ) : ℝ :=
  let cost_with_fee := 1 + production_fee_percent / 100
  let discount_factor := (1 - discount1_percent / 100) * (1 - discount2_percent / 100)
  let selling_price_with_discounts := cost_with_fee * (1 + profit_after_discounts_percent / 100)
  let selling_price_without_discounts := selling_price_with_discounts / discount_factor
  (selling_price_without_discounts / cost_with_fee - 1) * 100

/-- Theorem stating that given the specified conditions, the profit percentage
    without discounts is approximately 46.19%. -/
theorem profit_percentage_approx : 
  ∃ ε > 0, |profit_percentage_without_discounts 10 10 5 25 - 46.19| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l578_57855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_five_samples_correct_l578_57839

/-- Represents a row in the random number table -/
def RandomTableRow := List Nat

/-- Check if a number is valid for sample (between 000 and 799 inclusive) -/
def isValidSample (n : Nat) : Bool :=
  n ≤ 799

/-- Find the first n valid samples from a list of numbers -/
def findValidSamples (numbers : List Nat) (n : Nat) : List Nat :=
  (numbers.filter isValidSample).take n

theorem first_five_samples_correct (row8 row9 : List Nat) : 
  findValidSamples (row8 ++ row9) 5 = [785, 667, 199, 507, 175] :=
by
  sorry

#eval findValidSamples ([63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79] ++
                        [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54]) 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_five_samples_correct_l578_57839
