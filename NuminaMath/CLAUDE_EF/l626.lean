import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l626_62692

theorem sin_double_angle (x : Real) (h : Real.sin x - Real.cos x = 1/2) : Real.sin (2 * x) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l626_62692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l626_62602

theorem simplify_and_evaluate (a : ℝ) (h : (a - 1) / 2 ≤ 1) 
  (ha : a > 0) (ha2 : a ≠ 2) (ha3 : a ≠ 3) :
  (a^2 - 6*a + 9) / (a - 2) / (a + 2 + 5 / (2 - a)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l626_62602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l626_62649

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Calculates the distance from a point to a line --/
noncomputable def Line.distanceToPoint (l : Line) (x y : ℝ) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Theorem: The equation of line l passing through the intersection of two given lines
    and having a specific distance from a point --/
theorem line_equation (l1 l2 : Line) (l : Line) :
  (∀ x y : ℝ, l1.contains x y ∧ l2.contains x y → l.contains x y) →
  l.distanceToPoint 5 0 = 3 →
  (l.a = 1 ∧ l.b = 0 ∧ l.c = -2) ∨ (l.a = 4 ∧ l.b = -3 ∧ l.c = -5) := by
  sorry

/-- Given lines --/
def line1 : Line := ⟨2, 1, -5⟩
def line2 : Line := ⟨1, -2, 0⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l626_62649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_min_side_a_l626_62610

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * t.c * Real.cos t.A = t.a * Real.sin t.C ∧
  4 * Real.sin t.C = t.c^2 * Real.sin t.B

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : t.area = Real.sqrt 3 := by
  sorry

-- Theorem for the minimum value of a
theorem min_side_a (t : Triangle) (h : triangle_conditions t) (dot_product : t.b * t.c * Real.cos t.A = 4) :
  ∃ (a_min : ℝ), a_min = 2 * Real.sqrt 2 ∧ ∀ (a : ℝ), t.a ≥ a_min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_min_side_a_l626_62610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_l626_62652

/-- The area between two equal parallel chords in a circle -/
theorem area_between_chords (r d : ℝ) (h1 : r = 10) (h2 : d = 12) :
  let θ := Real.arccos (1 - d^2 / (2 * r^2))
  let area := 2 * (θ / (2 * Real.pi) * Real.pi * r^2 - d * Real.sqrt (r^2 - (d/2)^2) / 2)
  area = 20 * 12/25 * Real.pi - 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_l626_62652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solutions_cosine_equation_l626_62685

theorem smallest_solutions_cosine_equation :
  ∃ (k₁ k₂ : ℕ+), k₁ < k₂ ∧
  (∀ (k : ℕ+), k < k₁ → (Real.cos ((k.val ^ 2 + 7 ^ 2 : ℝ) * π / 180))^2 ≠ 1) ∧
  (Real.cos ((k₁.val ^ 2 + 7 ^ 2 : ℝ) * π / 180))^2 = 1 ∧
  (Real.cos ((k₂.val ^ 2 + 7 ^ 2 : ℝ) * π / 180))^2 = 1 ∧
  k₁ = 13 ∧ k₂ = 23 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solutions_cosine_equation_l626_62685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l626_62631

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Applies a dilation to a point -/
def dilate (p : Point) (center : Point) (factor : ℝ) : Point :=
  { x := center.x + factor * (p.x - center.x),
    y := center.y + factor * (p.y - center.y) }

/-- Theorem: The farthest vertex after dilation -/
theorem farthest_vertex_after_dilation (s : Square)
  (h1 : s.center = { x := 5, y := -5 })
  (h2 : s.area = 16)
  (dilation_center : Point)
  (h3 : dilation_center = { x := 0, y := 0 })
  (dilation_factor : ℝ)
  (h4 : dilation_factor = 3) :
  ∃ (v : Point), v = { x := 21, y := -21 } ∧
  ∀ (other : Point), other ≠ v →
  distance dilation_center v ≥ distance dilation_center other :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l626_62631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l626_62656

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, f x = f (-2 - x)) ∧
  f 1 = 1 ∧
  (∀ x, f x ≥ 0) ∧
  (∃ x, f x = 0)

/-- The specific quadratic function we're proving about -/
noncomputable def f : ℝ → ℝ := λ x => (1/4) * x^2 + (1/2) * x + (1/4)

/-- The theorem stating our claims about the quadratic function -/
theorem quadratic_function_properties :
  QuadraticFunction f ∧
  (∀ m > 1, (∃ t, ∀ x ∈ Set.Icc 1 m, f (x + t) ≤ x) ↔ m ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l626_62656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l626_62634

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem about a specific trapezium -/
theorem trapezium_properties (t : Trapezium) 
    (h1 : t.side1 = 20)
    (h2 : t.side2 = 18)
    (h3 : t.height = 14) :
    area t = 266 ∧ max t.side1 t.side2 = 20 := by
  sorry

#check trapezium_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l626_62634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l626_62671

theorem equation_solution :
  ∃ x : ℝ, (2 : ℝ)^((32 : ℝ)^x) = (32 : ℝ)^((2 : ℝ)^x) ↔ x = (Real.log 5) / (4 * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l626_62671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_is_eight_l626_62626

def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 17}

def is_bijection (f : ℕ → ℕ) : Prop :=
  Function.Injective f ∧ Function.Surjective f

def f_iter (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, x => x
| (Nat.succ k), x => f (f_iter f k x)

def satisfies_conditions (f : ℕ → ℕ) (M : ℕ) : Prop :=
  is_bijection f ∧
  (∀ m < M, ∀ i ∈ A, i < 17 →
    (f_iter f m (i+1) - f_iter f m i) % 17 ≠ 1 ∧
    (f_iter f m (i+1) - f_iter f m i) % 17 ≠ 16 ∧
    (f_iter f m 1 - f_iter f m 17) % 17 ≠ 1 ∧
    (f_iter f m 1 - f_iter f m 17) % 17 ≠ 16) ∧
  (∀ i ∈ A, i < 17 →
    ((f_iter f M (i+1) - f_iter f M i) % 17 = 1 ∨
     (f_iter f M (i+1) - f_iter f M i) % 17 = 16) ∧
    ((f_iter f M 1 - f_iter f M 17) % 17 = 1 ∨
     (f_iter f M 1 - f_iter f M 17) % 17 = 16))

theorem max_M_is_eight :
  ∀ f : ℕ → ℕ, (∃ M, satisfies_conditions f M) →
  (∀ N, satisfies_conditions f N → N ≤ 8) ∧
  (∃ f', satisfies_conditions f' 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_is_eight_l626_62626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_cans_display_l626_62614

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℤ) : ℕ :=
  let n : ℕ := (a₁ - aₙ) / d.natAbs + 1
  n * (a₁ + aₙ) / 2

/-- The problem statement -/
theorem supermarket_cans_display : arithmetic_sequence_sum 35 3 (-4) = 171 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_cans_display_l626_62614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_properties_l626_62632

theorem power_of_two_properties (n : ℕ) :
  (∃ k : ℕ, n = 3 * k) ↔ (7 ∣ (2^n - 1)) ∧ ¬(7 ∣ (2^n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_properties_l626_62632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ian_debt_calculation_l626_62624

/-- Represents the amount of money Ian won in the lottery -/
noncomputable def lottery_winnings : ℚ := 100

/-- Represents the amount Ian paid to Colin -/
noncomputable def colin_debt : ℚ := 20

/-- Represents the amount Ian paid to Helen -/
noncomputable def helen_debt : ℚ := 2 * colin_debt

/-- Represents the amount Ian paid to Benedict -/
noncomputable def benedict_debt : ℚ := helen_debt / 2

/-- Represents the initial amount Ian owed to Emma -/
noncomputable def emma_initial_debt : ℚ := 15

/-- Represents the interest rate Emma charged -/
noncomputable def emma_interest_rate : ℚ := 1 / 10

/-- Represents the total amount Ian paid to Emma -/
noncomputable def emma_total_debt : ℚ := emma_initial_debt * (1 + emma_interest_rate)

/-- Represents the initial amount Ian owed to Ava in euros -/
noncomputable def ava_initial_debt_euros : ℚ := 8

/-- Represents the percentage of debt Ava forgave -/
noncomputable def ava_forgiveness_rate : ℚ := 1 / 4

/-- Represents the exchange rate from euros to dollars -/
noncomputable def euro_to_dollar_rate : ℚ := 6 / 5

/-- Represents the total amount Ian paid to Ava in dollars -/
noncomputable def ava_total_debt_dollars : ℚ := ava_initial_debt_euros * (1 - ava_forgiveness_rate) * euro_to_dollar_rate

/-- Theorem stating that Ian needs an additional $3.70 to pay off all debts -/
theorem ian_debt_calculation :
  lottery_winnings - (colin_debt + helen_debt + benedict_debt + emma_total_debt + ava_total_debt_dollars) = -37/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ian_debt_calculation_l626_62624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_max_power_of_two_l626_62698

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => 5 * sequence_a n + 3^n

theorem sequence_a_formula (n : ℕ) :
  sequence_a n = (5^n - 3^n) / 2 := by sorry

theorem max_power_of_two (k : ℕ) (h : k = 2^2019) :
  ∃ m : ℕ, m = 2021 ∧ 2^m ∣ sequence_a k ∧ ∀ l : ℕ, l > m → ¬(2^l ∣ sequence_a k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_max_power_of_two_l626_62698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l626_62694

/-- Given that 12 workers take 6 days to build a wall, prove that 18 workers
    working at the same rate will take 4 days to build the same wall. -/
theorem wall_building_time
  (workers₁ : ℕ) (days₁ : ℕ) (workers₂ : ℕ) (days₂ : ℝ)
  (h₁ : workers₁ = 12)
  (h₂ : days₁ = 6)
  (h₃ : workers₂ = 18)
  (h₄ : (workers₁ : ℝ) * days₁ = workers₂ * days₂) :
  days₂ = 4 := by
  sorry

#check wall_building_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l626_62694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_A_joins_Hanfu_Society_l626_62664

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the clubs
inductive Club : Type
| Couplet : Club
| Calligraphy : Club
| Hanfu : Club

-- Define the grades
inductive Grade : Type
| First : Grade
| Second : Grade
| Third : Grade

-- Define the function that assigns a club to each student
variable (club_assignment : Student → Club)

-- Define the function that assigns a grade to each student
variable (grade_assignment : Student → Grade)

-- State the theorem
theorem student_A_joins_Hanfu_Society 
  (h1 : ∀ s1 s2, s1 ≠ s2 → club_assignment s1 ≠ club_assignment s2)
  (h2 : ∀ s1 s2, s1 ≠ s2 → grade_assignment s1 ≠ grade_assignment s2)
  (h3 : club_assignment Student.A ≠ Club.Couplet)
  (h4 : club_assignment Student.B ≠ Club.Hanfu)
  (h5 : ∀ s, club_assignment s = Club.Couplet → grade_assignment s ≠ Grade.Second)
  (h6 : ∀ s, club_assignment s = Club.Hanfu → grade_assignment s = Grade.First)
  (h7 : grade_assignment Student.B ≠ Grade.Third) :
  club_assignment Student.A = Club.Hanfu :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_A_joins_Hanfu_Society_l626_62664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l626_62611

def A : Set ℤ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l626_62611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_pi_plus_2x_l626_62630

open Real MeasureTheory

theorem integral_exp_pi_plus_2x : ∫ x in Set.Icc 0 1, (Real.exp π + 2 * x) = Real.exp π + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_pi_plus_2x_l626_62630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_z_percentage_l626_62612

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of Brand Z gasoline
  y : ℚ  -- Amount of Brand Y gasoline

/-- Fills the tank with Brand Y gasoline -/
def fillY (s : TankState) : TankState :=
  { z := s.z, y := 1 - s.z }

/-- Fills the tank with Brand Z gasoline -/
def fillZ (s : TankState) : TankState :=
  { z := 1 - s.y, y := s.y }

/-- Simulates half of the tank becoming empty -/
def halfEmpty (s : TankState) : TankState :=
  { z := s.z / 2, y := s.y / 2 }

/-- The sequence of operations described in the problem -/
def finalState : TankState :=
  fillY (halfEmpty (fillZ (halfEmpty (fillY { z := 1/4, y := 0 }))))

/-- The theorem to be proved -/
theorem brand_z_percentage :
  finalState.z = 5/16 ∧ finalState.z / (finalState.z + finalState.y) = 5/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_z_percentage_l626_62612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_diagonal_ratio_l626_62691

/-- A regular hexagon is a polygon with 6 equal sides and 6 equal angles -/
structure RegularHexagon where
  -- We don't need to define the structure explicitly for this problem

/-- The ratio of the shorter diagonal to the longer diagonal in a regular hexagon -/
noncomputable def diagonal_ratio (h : RegularHexagon) : ℝ :=
  Real.sqrt 3 / 2

/-- Theorem: The ratio of the shorter diagonal to the longer diagonal in a regular hexagon is √3/2 -/
theorem regular_hexagon_diagonal_ratio (h : RegularHexagon) : 
  diagonal_ratio h = Real.sqrt 3 / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_diagonal_ratio_l626_62691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_negative_l626_62655

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * x^2 - 3*x - 3) * Real.exp x

noncomputable def M (m : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * f m x - m * (x^2 - x)

noncomputable def N (x : ℝ) : ℝ := Real.log x - 4*x - 3

noncomputable def F (m : ℝ) (x : ℝ) : ℝ := M m x - N x

theorem f_derivative_negative (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < x₂) 
  (h₃ : F m x₁ = 0) (h₄ : F m x₂ = 0) :
  let x₀ := Real.sqrt (x₁ * x₂)
  (deriv (F m)) x₀ < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_negative_l626_62655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_cut_pyramid_l626_62676

/-- Given a regular triangular pyramid with height H and a plane through one base vertex
    inclined at angle α to the base, the volume of the part of the pyramid between
    the base and the cutting plane is (3√3 / 8) * H³ * tan²(α) * sin²(α) -/
theorem volume_cut_pyramid (H α : ℝ) (H_pos : H > 0) (α_pos : α > 0) (α_lt_pi_half : α < π / 2) :
  let volume := (3 * Real.sqrt 3 / 8) * H^3 * Real.tan α^2 * Real.sin α^2
  volume > 0 ∧ volume < (Real.sqrt 3 / 4) * H^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_cut_pyramid_l626_62676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_range_l626_62688

-- Define the function g(x) = x^2 - ax + 1
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Define the logarithmic function y = log_a(g(x))
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := Real.log (g a x) / Real.log a

-- Theorem statement
theorem function_minimum_implies_a_range (a : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_min : ∃ (x_min : ℝ), ∀ (x : ℝ), y a x_min ≤ y a x) :
  1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_range_l626_62688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_cannot_form_triangle_perpendicular_distance_l626_62699

-- Define the lines
def l1 (x y : ℝ) : Prop := 4 * x + y - 4 = 0
def l2 (m x y : ℝ) : Prop := m * x + y = 0
def l3 (m x y : ℝ) : Prop := x - m * y - 4 = 0

-- Define when lines are parallel
def parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

-- Define when a line is perpendicular to another line
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Define the distance between two parallel lines
noncomputable def distance_parallel_lines (a b c1 c2 : ℝ) : ℝ := 
  |c2 - c1| / Real.sqrt (a^2 + b^2)

theorem lines_cannot_form_triangle (m : ℝ) : 
  (∃ x y, l1 x y ∧ l2 m x y ∧ l3 m x y) → (m = 4 ∨ m = -1/4) := by
  sorry

theorem perpendicular_distance (m : ℝ) :
  (perpendicular 4 1 1 (-m) ∧ perpendicular (-m) 1 1 (-m)) →
  (distance_parallel_lines 4 1 (-4) 0 = 4 * Real.sqrt 17 / 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_cannot_form_triangle_perpendicular_distance_l626_62699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_santinos_papaya_trees_l626_62681

/-- The number of papaya trees Santino has -/
def papaya_trees : ℕ := 2

/-- The number of mango trees Santino has -/
def mango_trees : ℕ := 3

/-- The number of papayas each papaya tree produces -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos each mango tree produces -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := 80

theorem santinos_papaya_trees :
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits ∧
  papaya_trees = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_santinos_papaya_trees_l626_62681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_max_at_arccos_neg_third_l626_62636

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 3 + (Real.sin x) ^ 2 - Real.cos x

-- State the theorem
theorem f_max_value : 
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 32/27 := by
  sorry

-- Additional lemma to show the maximum occurs at x = arccos(-1/3)
theorem f_max_at_arccos_neg_third :
  ∃ x₀, Real.cos x₀ = -1/3 ∧ ∀ x, f x ≤ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_max_at_arccos_neg_third_l626_62636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_l626_62638

/-- Two intersecting circles with centers O and O', and a point X at their intersection -/
structure IntersectingCircles where
  O : ℝ × ℝ
  O' : ℝ × ℝ
  R : ℝ
  R' : ℝ
  X : ℝ × ℝ
  h1 : (X.1 - O.1)^2 + (X.2 - O.2)^2 = R^2
  h2 : (X.1 - O'.1)^2 + (X.2 - O'.2)^2 = R'^2

/-- Y is a point on the circle with center O -/
def Y (c : IntersectingCircles) := 
  {y : ℝ × ℝ // (y.1 - c.O.1)^2 + (y.2 - c.O.2)^2 = c.R^2}

/-- Z is a point on the circle with center O' -/
def Z (c : IntersectingCircles) := 
  {z : ℝ × ℝ // (z.1 - c.O'.1)^2 + (z.2 - c.O'.2)^2 = c.R'^2}

/-- X, Y, and Z are collinear -/
def AreCollinear (c : IntersectingCircles) (y : Y c) (z : Z c) :=
  ∃ t : ℝ, (y.1.1 - c.X.1) * (z.1.2 - c.X.2) = (y.1.2 - c.X.2) * (z.1.1 - c.X.1)

/-- YO and ZO' are collinear and symmetrical about the line connecting O and O' -/
def Symmetrical (c : IntersectingCircles) (y : Y c) (z : Z c) :=
  ∃ t : ℝ, y.1 = c.O + t • (c.O' - c.O) ∧ z.1 = c.O' + t • (c.O - c.O')

/-- The product XY · XZ -/
def Product (c : IntersectingCircles) (y : Y c) (z : Z c) : ℝ :=
  ((y.1.1 - c.X.1)^2 + (y.1.2 - c.X.2)^2) * ((z.1.1 - c.X.1)^2 + (z.1.2 - c.X.2)^2)

/-- The main theorem -/
theorem max_product (c : IntersectingCircles) :
  ∃ (y : Y c) (z : Z c), AreCollinear c y z ∧
    (∀ (y' : Y c) (z' : Z c), AreCollinear c y' z' → Product c y z ≥ Product c y' z') ∧
    Symmetrical c y z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_l626_62638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_unique_factor_count_l626_62673

def is_in_range (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 15

def factor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem twelve_unique_factor_count :
  ∃! n : ℕ, is_in_range n ∧ factor_count n = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_unique_factor_count_l626_62673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equality_l626_62667

noncomputable def f_A (x : ℝ) : ℝ := Real.sqrt (-2 * x^3)
noncomputable def g_A (x : ℝ) : ℝ := x * Real.sqrt (-2 * x)

def f_B (x : ℝ) : ℝ := abs x
noncomputable def g_B (x : ℝ) : ℝ := Real.sqrt (x^2)

noncomputable def f_D (x : ℝ) : ℝ := x / x
def g_D (x : ℝ) : ℝ := x^0

theorem functions_equality :
  (∀ x ≤ 0, f_A x = g_A x) ∧
  (∀ x : ℝ, f_B x = g_B x) ∧
  (∀ x ≠ 0, f_D x = g_D x) := by
  sorry

#check functions_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equality_l626_62667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l626_62605

/-- The area of a pentagon formed by cutting a right-angled triangle from a rectangle -/
theorem pentagon_area (a b c d e r s : ℕ) : 
  a ∈ ({14, 21, 22, 28, 35} : Set ℕ) →
  b ∈ ({14, 21, 22, 28, 35} : Set ℕ) →
  c ∈ ({14, 21, 22, 28, 35} : Set ℕ) →
  d ∈ ({14, 21, 22, 28, 35} : Set ℕ) →
  e ∈ ({14, 21, 22, 28, 35} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  r = b - d →
  s = c - a →
  r^2 + s^2 = e^2 →
  b * c - (r * s) / 2 = 931 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l626_62605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_transport_cost_l626_62646

/-- The cost in dollars to transport 1 kg to the International Space Station -/
noncomputable def transport_rate : ℝ := 25000

/-- The weight of the communication device in grams -/
noncomputable def comm_device_weight : ℝ := 500

/-- The weight of the sensor unit in grams -/
noncomputable def sensor_weight : ℝ := 300

/-- The additional cost percentage for transporting the sensor unit -/
noncomputable def sensor_additional_cost_percent : ℝ := 10

/-- Converts grams to kilograms -/
noncomputable def grams_to_kg (g : ℝ) : ℝ := g / 1000

/-- Calculates the cost to transport a given weight in kg -/
noncomputable def transport_cost (weight_kg : ℝ) : ℝ := weight_kg * transport_rate

/-- Calculates the total cost including the additional percentage -/
noncomputable def total_cost_with_additional (base_cost : ℝ) (additional_percent : ℝ) : ℝ :=
  base_cost * (1 + additional_percent / 100)

theorem total_transport_cost :
  let comm_device_cost := transport_cost (grams_to_kg comm_device_weight)
  let sensor_base_cost := transport_cost (grams_to_kg sensor_weight)
  let sensor_total_cost := total_cost_with_additional sensor_base_cost sensor_additional_cost_percent
  comm_device_cost + sensor_total_cost = 20750 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_transport_cost_l626_62646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bearing_savings_l626_62659

/-- Calculates the savings when buying ball bearings on sale with a bulk discount --/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (normal_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : normal_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : (num_machines * bearings_per_machine * normal_price : ℚ) -
    (num_machines * bearings_per_machine * sale_price * (1 - bulk_discount)) = 120 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bearing_savings_l626_62659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l626_62642

theorem root_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 2 3 ∧ Real.log x = -x + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l626_62642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_c_cheapest_with_discount_l626_62680

/-- Represents a brand of car wash soap -/
structure CarWashSoap where
  price : ℚ
  washes : ℕ

/-- Calculates the cost for a given number of washes using a specific soap -/
def cost_for_washes (soap : CarWashSoap) (num_washes : ℕ) : ℚ :=
  (num_washes : ℚ) * soap.price / (soap.washes : ℚ)

/-- Applies a bulk discount to the soap price -/
def apply_bulk_discount (soap : CarWashSoap) (discount_percent : ℚ) : CarWashSoap :=
  { price := soap.price * (1 - discount_percent), washes := soap.washes }

/-- Theorem stating that Brand C with bulk discount is the cheapest for 20 washes -/
theorem brand_c_cheapest_with_discount (brand_a brand_b brand_c : CarWashSoap)
  (h_a : brand_a = ⟨4, 4⟩)
  (h_b : brand_b = ⟨6, 6⟩)
  (h_c : brand_c = ⟨8, 9⟩)
  (discount : ℚ)
  (h_discount : discount = 1/10)
  (num_washes : ℕ)
  (h_num_washes : num_washes = 20) :
  cost_for_washes (apply_bulk_discount brand_c discount) num_washes = 16 ∧
  cost_for_washes (apply_bulk_discount brand_c discount) num_washes ≤ 
    min (cost_for_washes (apply_bulk_discount brand_a discount) num_washes)
        (cost_for_washes (apply_bulk_discount brand_b discount) num_washes) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_c_cheapest_with_discount_l626_62680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l626_62607

theorem problem_solution (x a : ℝ) 
  (h1 : (7 : ℝ)^(2*x) = 36) 
  (h2 : (7 : ℝ)^(-x) = (6 : ℝ)^(-a/2)) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l626_62607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_7398_to_hundredth_l626_62644

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_24_7398_to_hundredth :
  roundToHundredth 24.7398 = 24.74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_7398_to_hundredth_l626_62644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l626_62645

theorem trigonometric_identities (x : Real) (h : Real.tan x = 3) :
  ((2 * Real.sin (Real.pi - x) + 3 * Real.cos (-x)) / (Real.sin (x + Real.pi/2) - Real.sin (x + Real.pi)) = 9/4) ∧
  (2 * Real.sin x ^ 2 - Real.sin (2 * x) + Real.cos x ^ 2 = 13/10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l626_62645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_area_binomial_coefficient_l626_62669

/-- Given a parabola and a line forming a closed figure, prove the coefficient of x^(-16) in a binomial expansion --/
theorem parabola_line_area_binomial_coefficient 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_area : ∫ (x : ℝ) in Set.Icc 0 1, 2 * Real.sqrt (a * x) = 4/3) :
  Nat.choose 20 18 = 190 := by
  sorry

#check parabola_line_area_binomial_coefficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_area_binomial_coefficient_l626_62669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l626_62663

/-- Calculates the time (in seconds) for a train to pass a stationary point -/
noncomputable def time_to_pass (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

theorem train_passing_time :
  let train_length := (500 : ℝ)
  let train_speed := (60 : ℝ)
  let passing_time := time_to_pass train_length train_speed
  ∃ ε > 0, |passing_time - 30| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l626_62663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2a_minus_ab_l626_62666

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) : 
  (∀ x y : ℕ, 0 < x ∧ x < 10 → 0 < y ∧ y < 10 → (2 * x : ℤ) - x * y ≥ (2 * a : ℤ) - a * b) → 
  (2 * a : ℤ) - a * b = -63 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2a_minus_ab_l626_62666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_greater_than_sqrt2_over_2_l626_62622

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

-- Define the period
noncomputable def period : ℝ := Real.pi

-- Theorem for monotonically decreasing interval
theorem f_monotone_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi)) :=
by sorry

-- Theorem for the set of values where f(x) > √2/2
theorem f_greater_than_sqrt2_over_2 (x : ℝ) :
  f x > Real.sqrt 2 / 2 ↔ ∃ k : ℤ, -Real.pi / 24 + k * Real.pi < x ∧ x < 5 * Real.pi / 24 + k * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_greater_than_sqrt2_over_2_l626_62622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sine_problem_l626_62660

theorem inverse_sine_problem (x : ℝ) 
  (h1 : Real.sin x = -2/5) 
  (h2 : π < x) 
  (h3 : x < 3*π/2) : 
  x = π + Real.arcsin (2/5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sine_problem_l626_62660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_circumscribing_sphere_radius_l626_62617

/-- The radius of the inscribed ball at the center of the pyramid -/
noncomputable def r : ℝ := Real.sqrt 2 - 1

/-- The number of balls in the pyramid -/
def n : ℕ := 10

/-- The minimum number of balls each ball touches -/
def min_touches : ℕ := 3

/-- The number of balls the inscribed ball touches -/
def inscribed_touches : ℕ := 6

/-- The radius of the circumscribing sphere -/
noncomputable def R : ℝ := Real.sqrt 6 + 1

theorem pyramid_circumscribing_sphere_radius :
  ∀ (balls : ℕ) (min_contact : ℕ) (inscribed_contacts : ℕ) (inscribed_radius : ℝ),
    balls = n →
    min_contact = min_touches →
    inscribed_contacts = inscribed_touches →
    inscribed_radius = r →
    R = Real.sqrt 6 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_circumscribing_sphere_radius_l626_62617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_third_quadrant_l626_62639

-- Define the inverse proportion function
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m - 1) / x

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (x : ℝ) (y : ℝ) : Prop := x < 0 ∧ y < 0

-- State the theorem
theorem inverse_proportion_third_quadrant (m : ℝ) :
  (∃ x y : ℝ, y = inverse_proportion m x ∧ in_third_quadrant x y) → m > 1 := by
  intro h
  -- The proof goes here
  sorry

#check inverse_proportion_third_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_third_quadrant_l626_62639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_three_digit_numbers_l626_62696

def is_valid_arrangement (a b c : ℕ) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧
  b ≥ 100 ∧ b < 1000 ∧
  c ≥ 100 ∧ c < 1000 ∧
  a % 4 = 0 ∧ b % 4 = 0 ∧ c % 4 = 0 ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → 
    (d ∈ Nat.digits 10 a ∨ 
     d ∈ Nat.digits 10 b ∨ 
     d ∈ Nat.digits 10 c)) ∧
  (∀ d : ℕ, d ∈ Nat.digits 10 a → d ∉ Nat.digits 10 b ∧ d ∉ Nat.digits 10 c) ∧
  (∀ d : ℕ, d ∈ Nat.digits 10 b → d ∉ Nat.digits 10 c)

theorem min_max_three_digit_numbers :
  ∀ a b c : ℕ, is_valid_arrangement a b c →
  ∃ x y z : ℕ, is_valid_arrangement x y z ∧
  max x (max y z) = 896 ∧
  max a (max b c) ≥ 896 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_three_digit_numbers_l626_62696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l626_62618

open Real

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  cos (2 * C) - cos (2 * A) = 2 * sin (π / 3 + C) * sin (π / 3 - C) ∧
  a = Real.sqrt 3 ∧
  b ≥ a ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C →
  A = π / 3 ∧ 
  Real.sqrt 3 ≤ 2 * b - c ∧ 
  2 * b - c < 2 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l626_62618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_of_liberty_scale_models_l626_62606

/-- Represents the scale of a model in relation to the original statue -/
structure ModelScale where
  statue_height : ℝ
  model_height : ℝ
  scale : ℝ

/-- The scale of a model is the ratio of the statue's height to the model's height -/
noncomputable def calculate_scale (m : ModelScale) : ℝ :=
  m.statue_height / m.model_height

theorem statue_of_liberty_scale_models 
  (statue_height : ℝ)
  (large_model_height : ℝ)
  (small_model_height : ℝ)
  (h1 : statue_height = 305)
  (h2 : large_model_height = 10)
  (h3 : small_model_height = 5) :
  let large_scale := calculate_scale ⟨statue_height, large_model_height, 0⟩
  let small_scale := calculate_scale ⟨statue_height, small_model_height, 0⟩
  large_scale = 30.5 ∧ small_scale = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_of_liberty_scale_models_l626_62606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_solution_set_l626_62633

-- Define the constants a and b
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the axioms
axiom a_gt_one : a > 1
axiom b_gt_zero : b > 0
axiom one_gt_b : 1 > b
axiom a_squared : a^2 = b^2 + 1

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (a^x - b^x)

-- State the theorem
theorem f_positive_solution_set :
  {x : ℝ | f x > 0} = Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_solution_set_l626_62633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_divisors_l626_62675

theorem least_k_for_divisors (a b : ℕ) (h : Nat.gcd a b = 1) :
  (∃ k : ℕ, ∀ r : ℕ, r ≥ k → 
    ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ 
      (Nat.card {y : ℕ | y ∣ (m^a * n^b)} = r + 1)) ∧
  (∀ k' : ℕ, (∀ r : ℕ, r ≥ k' → 
    ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ 
      (Nat.card {y : ℕ | y ∣ (m^a * n^b)} = r + 1)) → 
   k' ≥ a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_divisors_l626_62675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_l626_62678

/-- Population growth model over 10 years -/
theorem population_growth
  (initial_population : ℕ)
  (birth_rate : ℚ)
  (emigration_rate : ℕ)
  (immigration_rate : ℕ)
  (time_period : ℕ)
  (h1 : initial_population = 100000)
  (h2 : birth_rate = 0.6)
  (h3 : emigration_rate = 2000)
  (h4 : immigration_rate = 2500)
  (h5 : time_period = 10) :
  initial_population +
  (↑initial_population * birth_rate).floor +
  (immigration_rate - emigration_rate) * time_period = 165000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_l626_62678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l626_62657

-- Define triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define vectors m and n
noncomputable def m : ℝ × ℝ := (-1, 1)
noncomputable def n (B C : ℝ) : ℝ × ℝ := (Real.cos B * Real.cos C, Real.sin B * Real.sin C - Real.sqrt 3 / 2)

-- State the theorem
theorem triangle_abc_properties :
  (m.1 * (n B C).1 + m.2 * (n B C).2 = 0) → -- m perpendicular to n
  (A + B + C = π) →
  (A = π / 6) ∧
  ((a = 1 ∧ 2 * c - (Real.sqrt 3 + 1) * b = 0) ∨ (a = 1 ∧ B = π / 4)) →
  (1 / 2 * a * b * Real.sin C = (Real.sqrt 3 + 1) / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l626_62657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_segment_length_l626_62684

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
noncomputable def is_equilateral (t : Triangle) : Prop :=
  let AB := ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2).sqrt
  let BC := ((t.C.x - t.B.x)^2 + (t.C.y - t.B.y)^2).sqrt
  let CA := ((t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2).sqrt
  AB = BC ∧ BC = CA

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x)^2 + (p2.y - p1.y)^2).sqrt

/-- Main theorem -/
theorem equilateral_triangle_segment_length 
  (ABC : Triangle) 
  (D E F G H J : Point) :
  is_equilateral ABC →
  distance ABC.A ABC.B = 28 →
  D.x ∈ Set.Icc ABC.B.x ABC.C.x →
  E.x ∈ Set.Icc ABC.C.x ABC.A.x →
  F.x ∈ Set.Icc ABC.A.x ABC.B.x →
  distance ABC.A G = 3 →
  distance G F = 14 →
  distance F H = 9 →
  distance H ABC.C = 2 →
  J.x ∈ Set.Icc H.x F.x →
  distance H J = distance J F →
  distance ABC.B J = 13.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_segment_length_l626_62684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l626_62677

-- Define the center of the circle
def center : ℝ × ℝ := (5, 1)

-- Define the radius of the circle
def radius : ℝ := 7

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_to_origin :
  ∃ (p : ℝ × ℝ), distance p (0, 0) = distance center (0, 0) + radius ∧
  ∀ (q : ℝ × ℝ), distance q center ≤ radius → distance q (0, 0) ≤ distance p (0, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l626_62677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l626_62625

/-- Ellipse E with equation x^2/2 + y^2 = 1 -/
def ellipse_E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Circle with equation x^2 + y^2 = 3 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- Left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Line l passing through F₂ -/
def line_l (t : ℝ) (x y : ℝ) : Prop := x = t*y + 1

/-- Condition for the dot product of F₁A and F₁B to be 1 -/
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (x₁ + 1)*(x₂ + 1) + y₁*y₂ = 1

/-- Main theorem -/
theorem ellipse_triangle_area :
  ∃ (t : ℝ) (A B C D : ℝ × ℝ),
    line_l t A.1 A.2 ∧ line_l t B.1 B.2 ∧
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    line_l t C.1 C.2 ∧ line_l t D.1 D.2 ∧
    ellipse_E C.1 C.2 ∧ ellipse_E D.1 D.2 ∧
    dot_product_condition A B ∧
    abs (C.2 - D.2) * 2 = 4*Real.sqrt 6 / 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l626_62625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completes_work_in_8_days_l626_62650

/-- Represents the time it takes for A to complete the work alone -/
noncomputable def time_A : ℝ := sorry

/-- Represents the total amount of work to be done -/
noncomputable def total_work : ℝ := sorry

/-- Represents the amount of work A can do in one day -/
noncomputable def work_rate_A : ℝ := total_work / time_A

/-- Represents the amount of work B can do in one day -/
noncomputable def work_rate_B : ℝ := sorry

theorem A_completes_work_in_8_days :
  (4 * work_rate_A + 6 * work_rate_B = total_work) →
  ((work_rate_A + work_rate_B) * 4.8 = total_work) →
  time_A = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completes_work_in_8_days_l626_62650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_value_l626_62643

theorem binomial_coefficient_value (a : ℝ) (X : ℝ) : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k) * a^k * X^(7-k)) = 7 * X^6 + X * (Finset.range 7).sum (λ k => (Nat.choose 7 (k+1)) * a^(k+1) * X^(6-k)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_value_l626_62643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_two_l626_62601

/-- The function f(x) = x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := (x - c)^2 + 2 * x * (x - c)

theorem max_at_two (c : ℝ) :
  (f_derivative c 2 = 0 ∧
   ∀ ε > 0, ∃ δ₁ δ₂ : ℝ, δ₁ > 0 ∧ δ₂ > 0 ∧
     (∀ x, 2 - δ₁ < x ∧ x < 2 → f_derivative c x > 0) ∧
     (∀ x, 2 < x ∧ x < 2 + δ₂ → f_derivative c x < 0)) →
  c = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_two_l626_62601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_multiplier_is_one_l626_62623

-- Define the multiplier as a parameter
variable (multiplier : ℚ)

-- Define the floor function
def floor (m : ℕ) : ℚ :=
  if m % 2 = 0 then (1 / 2) * m else multiplier * m

-- Define the conditions
axiom floor_odd (m : ℕ) (h : m % 2 = 1) : floor multiplier m = multiplier * m
axiom floor_even (m : ℕ) (h : m % 2 = 0) : floor multiplier m = (1 / 2) * m
axiom floor_product : floor multiplier 9 * floor multiplier 10 = 45

-- Theorem to prove
theorem odd_multiplier_is_one : multiplier = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_multiplier_is_one_l626_62623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_expression_l626_62620

/-- The given function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

/-- Recursive definition of f_n(x) -/
noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => f x
| (n + 1), x => f (f_n n x)

/-- The main theorem to prove -/
theorem f_16_expression (x : ℝ) (h : x ≠ 0) :
  (f_n 13 x = f_n 31 x) → f_n 16 x = (x - 1) / x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_expression_l626_62620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_root_equals_two_l626_62637

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem floor_of_root_equals_two :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ Int.floor x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_root_equals_two_l626_62637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_lesson_cost_l626_62641

theorem dance_lesson_cost (studio_a_price studio_b_price studio_c_price : ℚ)
  (studio_a_lessons studio_b_lessons studio_c_lessons : ℕ)
  (studio_b_discount : ℚ) (studio_c_free_lessons : ℕ) :
  studio_a_price = 15 ∧ 
  studio_b_price = 12 ∧ 
  studio_c_price = 18 ∧
  studio_a_lessons = 4 ∧
  studio_b_lessons = 3 ∧
  studio_c_lessons = 3 ∧
  studio_b_discount = 1/5 ∧
  studio_c_free_lessons = 1 →
  (studio_a_price * studio_a_lessons +
   studio_b_price * studio_b_lessons * (1 - studio_b_discount) +
   studio_c_price * (studio_c_lessons - studio_c_free_lessons)) = 124.8 := by
  sorry

#check dance_lesson_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_lesson_cost_l626_62641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_areas_theorem_l626_62608

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk :=
  (n : ℕ)  -- Half the number of equally spaced radii
  (secant_lines : Fin 2 → Set ℝ × Set ℝ)  -- Two secant lines represented as pairs of sets

/-- Predicate to check if secant lines do not intersect outside the circle -/
def secant_lines_valid (disk : DividedDisk) : Prop :=
  ∀ (i j : Fin 2), i ≠ j → (disk.secant_lines i).1 ∩ (disk.secant_lines j).1 ⊆ {x : ℝ | x^2 ≤ 1}

/-- The maximum number of non-overlapping areas in the divided disk -/
def max_areas (disk : DividedDisk) : ℕ := 5 * disk.n + 1

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_theorem (disk : DividedDisk) (h : secant_lines_valid disk) :
  max_areas disk = 5 * disk.n + 1 :=
by
  -- The proof would go here, but we'll use sorry for now
  sorry

#check max_areas_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_areas_theorem_l626_62608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_one_fourth_l626_62613

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x^3 - 3) / 2

-- State the theorem
theorem inverse_g_one_fourth :
  g (Real.rpow (7/2) (1/3)) = 1/4 :=
by
  -- Expand the definition of g
  unfold g
  -- Simplify the expression
  simp [Real.rpow_nat_cast, Real.rpow_mul]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_one_fourth_l626_62613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_values_l626_62683

noncomputable def mySequence : ℕ → ℚ
  | 0 => 1/2
  | 1 => 5/8
  | 2 => 3/4
  | 3 => 7/8
  | n + 4 => (n + 8) / 8

theorem sequence_values :
  mySequence 4 = 1 ∧ mySequence 5 = 9/8 ∧ mySequence 6 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_values_l626_62683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friction_coefficient_approx_l626_62686

/-- The coefficient of friction between a rod and a surface --/
noncomputable def friction_coefficient (α : Real) (normal_force_ratio : Real) : Real :=
  (1 - normal_force_ratio * Real.cos α) / (normal_force_ratio * Real.sin α)

/-- Theorem stating that the coefficient of friction is approximately 0.17 --/
theorem friction_coefficient_approx :
  let α : Real := 80 * Real.pi / 180  -- Convert 80° to radians
  let normal_force_ratio : Real := 11
  abs (friction_coefficient α normal_force_ratio - 0.17) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friction_coefficient_approx_l626_62686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_period_omega_l626_62628

theorem cos_period_omega (ω : ℝ) : 
  (∀ x : ℝ, Real.cos (ω * x) = Real.cos (ω * (x + π / 2))) →
  (∀ T : ℝ, T > 0 → T < π / 2 → ∃ x : ℝ, Real.cos (ω * x) ≠ Real.cos (ω * (x + T))) →
  ω = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_period_omega_l626_62628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_implies_a_range_l626_62674

/-- A function f(x) with a cubic term, quadratic term, linear term, and constant term. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + x^2 + a * x + 1

/-- The derivative of f(x) with respect to x. -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + a

/-- Theorem stating that if f(x) has both a maximum and a minimum value, 
    then a is in the open interval (-1,0) union (0,1). -/
theorem f_max_min_implies_a_range (a : ℝ) : 
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) → 
  (a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_implies_a_range_l626_62674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_bound_l626_62693

/-- The function f(x) = ln x + a(x^2 - x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - x)

/-- The statement that f has two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x > 0, f a x ≤ f a x₁ ∨ f a x ≤ f a x₂)

/-- The main theorem -/
theorem extreme_points_sum_bound (a : ℝ) :
  has_two_extreme_points a →
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    f a x₁ + f a x₂ < -3 - 4 * Real.log 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_bound_l626_62693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l626_62682

/-- Vector a as defined in the problem -/
def a : Fin 3 → ℝ := ![-7, 0, 1]

/-- Vector b as defined in the problem -/
def b : Fin 3 → ℝ := ![4, 2, -1]

/-- Theorem stating that a - 3 • b equals the expected result -/
theorem vector_subtraction : a - 3 • b = ![-19, -6, 4] := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l626_62682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l626_62672

/-- A function that checks if a number has three different digits -/
def has_three_different_digits (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 && d2 ≠ d3 && d1 ≠ d3

/-- A function that checks if the digits of a number are in strictly increasing order -/
def digits_increasing (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 < d2 && d2 < d3

/-- A function that checks if the digits of a number are in strictly decreasing order -/
def digits_decreasing (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 > d2 && d2 > d3

/-- A function that checks if the sum of digits of a number is even -/
def sum_of_digits_even (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  (d1 + d2 + d3) % 2 == 0

/-- The main theorem to be proved -/
theorem count_special_numbers : 
  (Finset.filter (λ n : ℕ => 
    200 ≤ n && n ≤ 999 && 
    has_three_different_digits n &&
    (digits_increasing n || digits_decreasing n) &&
    sum_of_digits_even n) 
    (Finset.range 1000)).card = 56 := by
  sorry

#eval (Finset.filter (λ n : ℕ => 
    200 ≤ n && n ≤ 999 && 
    has_three_different_digits n &&
    (digits_increasing n || digits_decreasing n) &&
    sum_of_digits_even n) 
    (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l626_62672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cl_approx_l626_62654

/-- The molar mass of sodium in g/mol -/
noncomputable def molar_mass_Na : ℝ := 22.99

/-- The molar mass of chlorine in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The mass of NaClO produced in grams -/
noncomputable def mass_NaClO : ℝ := 100

/-- The molar mass of NaClO in g/mol -/
noncomputable def molar_mass_NaClO : ℝ := molar_mass_Na + molar_mass_Cl + molar_mass_O

/-- The mass percentage of Cl in NaClO -/
noncomputable def mass_percentage_Cl : ℝ := (molar_mass_Cl / molar_mass_NaClO) * 100

theorem mass_percentage_Cl_approx :
  |mass_percentage_Cl - 47.61| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cl_approx_l626_62654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l626_62697

def is_solution (a b c : ℕ+) : Prop :=
  (1 + 1 / (a : ℝ)) * (1 + 1 / (b : ℝ)) * (1 + 1 / (c : ℝ)) = 2

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(⟨2, by norm_num⟩, ⟨4, by norm_num⟩, ⟨15, by norm_num⟩),
   (⟨2, by norm_num⟩, ⟨5, by norm_num⟩, ⟨9, by norm_num⟩),
   (⟨2, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩),
   (⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩),
   (⟨3, by norm_num⟩, ⟨3, by norm_num⟩, ⟨8, by norm_num⟩)}

def permutations (s : Set (ℕ+ × ℕ+ × ℕ+)) : Set (ℕ+ × ℕ+ × ℕ+) :=
  s ∪ (s.image (λ (a, b, c) => (a, c, b)))
    ∪ (s.image (λ (a, b, c) => (b, a, c)))
    ∪ (s.image (λ (a, b, c) => (b, c, a)))
    ∪ (s.image (λ (a, b, c) => (c, a, b)))
    ∪ (s.image (λ (a, b, c) => (c, b, a)))

theorem solution_characterization :
  ∀ a b c : ℕ+, is_solution a b c ↔ (a, b, c) ∈ permutations solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l626_62697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_formulas_equivalent_l626_62653

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | 2 => Real.sqrt 2
  | 3 => 0
  | n + 4 => sequence_a n

noncomputable def formula1 (n : ℕ) : ℝ := (Real.sqrt 2 / 2) * (1 + (-1)^n)

noncomputable def formula2 (n : ℕ) : ℝ := Real.sqrt (1 + (-1)^n)

noncomputable def formula3 (n : ℕ) : ℝ := if n % 2 = 0 then Real.sqrt 2 else 0

theorem all_formulas_equivalent (n : ℕ) :
  sequence_a n = formula1 n ∧ sequence_a n = formula2 n ∧ sequence_a n = formula3 n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_formulas_equivalent_l626_62653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_sides_l626_62661

/-- 
An isosceles right triangle ABC with hypotenuse on the line 3x - y + 2 = 0 
and right-angle vertex C at (3, -2) has right-angle sides AC and BC 
with equations x + 2y - 7 = 0 and x - 2y - 7 = 0.
-/
theorem isosceles_right_triangle_sides (A B C : ℝ × ℝ) : 
  let hypotenuse := {(x, y) : ℝ × ℝ | 3 * x - y + 2 = 0}
  let right_angle_vertex := (3, -2)
  C = right_angle_vertex →
  (∃ (a b : ℝ × ℝ), a ∈ hypotenuse ∧ b ∈ hypotenuse ∧ A = a ∧ B = b) →
  (∀ (x y : ℝ), x + 2 * y - 7 = 0 ↔ (x - 3) * (2) + (y + 2) * (-1) = 0) ∧
  (∀ (x y : ℝ), x - 2 * y - 7 = 0 ↔ (x - 3) * (-2) + (y + 2) * (-1) = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_sides_l626_62661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_mutually_gazing_pairs_l626_62640

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The function f(x) = 1/(1-x) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

/-- The function g(x) = 2sin(πx) -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x)

/-- Check if a point is on the graph of a function within a given interval -/
def isOnGraph (p : Point) (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ p.x ∧ p.x ≤ b ∧ p.y = h p.x

/-- Check if two points are symmetric about (1,0) -/
def areSymmetric (p q : Point) : Prop :=
  q.x - 1 = 1 - p.x ∧ q.y = -p.y

/-- A mutually gazing point pair -/
structure MutuallyGazingPair where
  p : Point
  q : Point
  h_p : isOnGraph p f (-2) 4
  h_q : isOnGraph q g (-2) 4
  h_sym : areSymmetric p q

/-- The main theorem -/
theorem four_mutually_gazing_pairs :
  ∃ (s : Finset MutuallyGazingPair), s.card = 4 ∧
  (∀ pair : MutuallyGazingPair, pair ∈ s) ∧
  (∀ p q : MutuallyGazingPair, p ∈ s → q ∈ s → p ≠ q →
    (p.p ≠ q.p ∨ p.q ≠ q.q)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_mutually_gazing_pairs_l626_62640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_twelve_edges_l626_62635

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- The number of edges in a geometric shape -/
def num_edges (shape : Type) : ℕ := sorry

/-- Theorem: A cube has 12 edges -/
theorem cube_has_twelve_edges : num_edges Cube = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_twelve_edges_l626_62635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_twelve_l626_62629

/-- Represents the number of radios purchased -/
def n : ℕ := sorry

/-- Represents the total cost of radios in dollars -/
def d : ℕ := sorry

/-- Represents that d is positive -/
axiom h_d_pos : d > 0

/-- Represents the profit function based on the problem conditions -/
def profit (n d : ℕ) : ℚ :=
  10 * n - 20 - (4 * d) / (3 * n)

/-- Theorem stating that 12 is the smallest n that satisfies the conditions -/
theorem smallest_n_is_twelve :
  (∀ k < 12, profit k d ≠ 100) ∧ profit 12 d = 100 := by
  sorry

#check smallest_n_is_twelve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_twelve_l626_62629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_equals_one_sixteenth_l626_62670

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.sqrt (-x)
  else (x - 1/2)^4

-- Theorem statement
theorem f_of_f_neg_one_equals_one_sixteenth :
  f (f (-1)) = 1/16 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_equals_one_sixteenth_l626_62670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l626_62662

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the right focus of the ellipse
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line with slope 1 passing through the right focus
noncomputable def line (x y : ℝ) : Prop := y = x - Real.sqrt 3

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B → Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l626_62662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_key_arrangements_l626_62690

/-- Represents the number of keys on the keychain --/
def total_keys : ℕ := 6

/-- Represents the number of ways to arrange the house-car-office group --/
def group_arrangements : ℕ := 4

/-- Represents the number of ways to place the office key relative to the house-car pair --/
def office_placements : ℕ := 2

/-- Represents the number of ways to arrange the remaining keys --/
def remaining_key_arrangements : ℕ := Nat.factorial 3

/-- Theorem stating the number of distinct key arrangements --/
theorem distinct_key_arrangements :
  total_keys = 6 →
  group_arrangements = 4 →
  office_placements = 2 →
  remaining_key_arrangements = Nat.factorial 3 →
  (group_arrangements * office_placements * remaining_key_arrangements) = 48 := by
  intros h1 h2 h3 h4
  rw [h2, h3, h4]
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_key_arrangements_l626_62690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_field_dimensions_l626_62621

/-- Represents a rectangular plot with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the diagonal of a rectangle using the Pythagorean theorem -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.length ^ 2 + r.width ^ 2)

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem: If a rectangular plot with diagonal 185 meters has its length and width
    reduced by 4 meters each, and the resulting area is reduced by 1012 m²,
    then the dimensions of the new rectangle are 153 m × 104 m -/
theorem sports_field_dimensions (original : Rectangle) (new : Rectangle) :
  original.diagonal = 185 ∧
  new.length = original.length - 4 ∧
  new.width = original.width - 4 ∧
  new.area = original.area - 1012 →
  (new.length = 153 ∧ new.width = 104) ∨ (new.length = 104 ∧ new.width = 153) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_field_dimensions_l626_62621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_parabola_properties_l626_62600

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 4 * p * x

/-- Represents an ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1

/-- Represents a circle in the xy-plane -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  equation : ℝ → ℝ → Prop := fun x y => (x - h)^2 + (y - k)^2 = r^2

theorem hyperbola_properties (h : Hyperbola) (e : Ellipse) :
  h.a = 6 ∧ h.b = 2 * Real.sqrt 3 ∧ e.a = 8 ∧ e.b = 4 →
  (∀ x y, x - Real.sqrt 3 * y = 0 → h.equation x y) ∧
  (∀ x, h.equation (4 * Real.sqrt 3) x ↔ x = 0) :=
by sorry

theorem parabola_properties (p : Parabola) (c : Circle) :
  p.p = 1 ∧ c.h = 3 ∧ c.k = 0 ∧ c.r = 4 →
  (∀ x y, x = -1 → p.equation x y) ∧
  (∃ x y, c.equation x y ∧ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_parabola_properties_l626_62600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l626_62616

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
structure PointOnEllipse (ε : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / ε.a^2 + y^2 / ε.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (ε : Ellipse) : ℝ := 
  Real.sqrt (1 - ε.b^2 / ε.a^2)

/-- The theorem statement -/
theorem ellipse_properties (ε : Ellipse) 
  (A B C : PointOnEllipse ε) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_not_x_axis : A.y ≠ 0 ∧ B.y ≠ 0 ∧ C.y ≠ 0) 
  (h_AC_through_F1 : ∃ t : ℝ, A.x + t * (C.x - A.x) = -ε.a * eccentricity ε ∧ A.y + t * (C.y - A.y) = 0)
  (h_BC_through_F2 : ∃ t : ℝ, B.x + t * (C.x - B.x) = ε.a * eccentricity ε ∧ B.y + t * (C.y - B.y) = 0) :
  (2 * ε.a = 4 ∧ eccentricity ε = 1/2 → ε.b = Real.sqrt 3) ∧
  (C.x = 0 ∧ C.y = 1 ∧ abs ((A.x * B.y - A.y * B.x) / 2) = 64/(49 * Real.sqrt 3) → ε.a = 2) ∧
  (ε.a = Real.sqrt 2 ∧ ε.b = 1 ∧ ∃ t : ℝ, A.x + t * (B.x - A.x) = 3/2 ∧ A.y + t * (B.y - A.y) = 0 → 
   (C.x = -4/3 ∧ C.y = -1/3) ∨ (C.x = -4/3 ∧ C.y = 1/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l626_62616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_rent_calculation_rent_approximation_l626_62609

/-- Calculates the required monthly rent for an apartment investment -/
theorem apartment_rent_calculation 
  (purchase_price : ℝ) 
  (repair_percentage : ℝ) 
  (annual_taxes : ℝ) 
  (return_rate : ℝ) : ℝ :=
  let annual_return := purchase_price * return_rate
  let total_annual_cost := annual_return + annual_taxes
  let monthly_earnings := total_annual_cost / 12
  let rent := monthly_earnings / (1 - repair_percentage)
  rent

/-- Proves that the calculated rent is approximately 106.48 -/
theorem rent_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |apartment_rent_calculation 12000 0.125 395 0.06 - 106.48| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_rent_calculation_rent_approximation_l626_62609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_spheres_l626_62604

/-- Helper function to calculate the volume of intersection of two spheres -/
noncomputable def volume_of_intersection (r₁ r₂ d : ℝ) : ℝ :=
  sorry -- Implementation details omitted as per instructions

/-- The volume of the intersection of two spheres -/
theorem intersection_volume_of_spheres 
  (r₁ r₂ d : ℝ) 
  (hr₁ : r₁ = 5) 
  (hr₂ : r₂ = 3) 
  (hd : d = 4) :
  let V := (68 * Real.pi) / 3
  ∃ (V_intersection : ℝ), V_intersection = V ∧ 
    V_intersection = volume_of_intersection r₁ r₂ d :=
by
  sorry -- Proof details omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_spheres_l626_62604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_l626_62687

-- Define the polynomial as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2023 * x^3 - 4050 * x^2 + 16 * x - 4

-- State the theorem
theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ →
  x₂ * (x₁ + x₃) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_l626_62687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passenger_encounters_three_freight_trains_l626_62615

/-- Represents a train on the railway system -/
structure Train where
  speed : ℚ
  startTime : ℚ
  startStation : ℕ

/-- Represents the railway system -/
structure Railway where
  numStations : ℕ
  stationDistance : ℚ
  freightTrains : List Train
  passengerTrain : Train

/-- The specific railway system described in the problem -/
noncomputable def problemRailway : Railway :=
  { numStations := 11
  , stationDistance := 7
  , freightTrains := List.range 18 |>.map (λ i =>
      { speed := 60
      , startTime := 0 + (5 / 60) * i
      , startStation := 11
      })
  , passengerTrain :=
      { speed := 100
      , startTime := 1
      , startStation := 1
      }
  }

/-- Calculates the position of a train at a given time -/
noncomputable def trainPosition (train : Train) (time : ℚ) : ℚ :=
  train.speed * (time - train.startTime)

/-- Theorem stating where the passenger train encounters 3 freight trains -/
theorem passenger_encounters_three_freight_trains :
  ∃ (x : ℚ), 5 * problemRailway.stationDistance < x ∧ x < 6 * problemRailway.stationDistance ∧
  (∃ (t₁ t₂ t₃ : ℚ) (f₁ f₂ f₃ : Train),
    t₁ < t₂ ∧ t₂ < t₃ ∧
    f₁ ∈ problemRailway.freightTrains ∧
    f₂ ∈ problemRailway.freightTrains ∧
    f₃ ∈ problemRailway.freightTrains ∧
    trainPosition problemRailway.passengerTrain t₁ = trainPosition f₁ t₁ ∧
    trainPosition problemRailway.passengerTrain t₂ = trainPosition f₂ t₂ ∧
    trainPosition problemRailway.passengerTrain t₃ = trainPosition f₃ t₃ ∧
    x = trainPosition problemRailway.passengerTrain t₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_passenger_encounters_three_freight_trains_l626_62615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_theorem_l626_62648

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of the largest face of a rectangular box -/
noncomputable def largestFaceArea (d : BoxDimensions) : ℝ :=
  max (d.x * d.y) (max (d.x * d.z) (d.y * d.z))

/-- Theorem: For a rectangular box with dimensions x by 2 by 3 inches, 
    if the face of greatest area has an area of 15 square inches, then x = 5 -/
theorem box_dimension_theorem (d : BoxDimensions) 
    (h1 : d.y = 2) 
    (h2 : d.z = 3) 
    (h3 : largestFaceArea d = 15) : 
  d.x = 5 := by
  sorry

#check box_dimension_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_theorem_l626_62648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_area_difference_l626_62651

/-- Approximation relation -/
def approx (a b : ℝ) : Prop := abs (a - b) < 0.05

notation a " ≈ " b => approx a b

/-- The difference between the area of a circle with diameter 8 inches
    and the area of a square with diagonal 6 inches -/
theorem circle_square_area_difference : ∃ (diff : ℝ),
  (approx diff 32.3) ∧
  (∀ (circle_area square_area : ℝ),
    (∃ (r : ℝ), circle_area = π * r^2 ∧ 2 * r = 8) →
    (∃ (s : ℝ), square_area = s^2 ∧ 2 * s^2 = 6^2) →
    diff = circle_area - square_area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_area_difference_l626_62651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_triangle_area_l626_62679

/-- Given a hyperbola C with eccentricity √3 passing through (√3, 0),
    and a line l passing through its right focus at an angle of π/3,
    prove that the area of the triangle formed by the origin and
    the two intersection points of the hyperbola and the line is 36. -/
theorem hyperbola_intersection_triangle_area :
  ∀ (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) (O F₁ F₂ : ℝ × ℝ),
  (∃ (a b : ℝ), C = {(x, y) | x^2/a^2 - y^2/b^2 = 1 ∧ a^2 + b^2 = 9}) →
  (∃ (m : ℝ), l = {(x, y) | y = m * (x - 3)}) →
  (Real.sqrt 3, 0) ∈ C →
  F₂ ∈ l →
  (∃ (θ : ℝ), θ = π / 3 ∧ ∀ (x y : ℝ), (x, y) ∈ l → y = (Real.tan θ) * (x - F₂.1)) →
  O = (0, 0) →
  F₁.1 < O.1 →
  F₂.1 > O.1 →
  F₁ ∈ C ∧ F₂ ∈ C →
  ∃ (A B : ℝ × ℝ),
    A ∈ C ∧ B ∈ C ∧
    A ∈ l ∧ B ∈ l ∧
    abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2)) / 2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_triangle_area_l626_62679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_4_6_8_l626_62689

/-- Heron's formula for the area of a triangle -/
noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

/-- The area of a triangle with sides 4, 6, and 8 is 3√15 -/
theorem triangle_area_4_6_8 :
  herons_formula 4 6 8 = 3 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_4_6_8_l626_62689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l626_62647

-- Define the constants
noncomputable def a : ℝ := Real.exp 0.9 + 1
noncomputable def b : ℝ := 29 / 10
noncomputable def c : ℝ := Real.log (0.9 * Real.exp 3)

-- State the theorem
theorem a_gt_b_gt_c : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l626_62647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_property_l626_62665

/-- Represents the sum of the first n terms of a geometric sequence. -/
def GeometricSum (n : ℕ) : ℝ := sorry

/-- Given a geometric sequence with sum S_n, if S_2 = 2 and S_4 = 8, then S_6 = 26. -/
theorem geometric_sum_property :
  (GeometricSum 2 = 2) →
  (GeometricSum 4 = 8) →
  (GeometricSum 6 = 26) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_property_l626_62665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_iff_a_ge_2_l626_62668

-- Define the function f(x) = 2^(x(x-a))
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(x*(x-a))

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem f_monotonically_decreasing_iff_a_ge_2 :
  ∀ a : ℝ, (monotonically_decreasing (f a) 0 1) ↔ a ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_iff_a_ge_2_l626_62668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_false_l626_62619

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | ∃ y, y = x^2 - 4}
def B : Set ℝ := {y : ℝ | ∃ x, y = x^2 - 4}
def C : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | y = x^2 - 4}

-- Define the statements
def statement1 : Prop := (A.prod Set.univ) ∩ C = ∅
def statement2 : Prop := A.prod Set.univ = C
def statement3 : Prop := A = B
def statement4 : Prop := B.prod Set.univ = C

-- Theorem stating that exactly 3 of the statements 2, 3, and 4 are false
theorem three_statements_false :
  (¬statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement2 ∧ ¬statement3 ∧ statement4) ∨
  (¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement2 ∧ ¬statement3 ∧ ¬statement4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_false_l626_62619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_charges_calculation_l626_62658

/-- Given a purchase price, repair cost, profit percentage, and selling price,
    calculate the transportation charges. -/
def calculate_transportation_charges (purchase_price repair_cost : ℕ) 
                                     (profit_percentage : ℚ) 
                                     (selling_price : ℕ) : ℕ :=
  let total_cost_before_transport := purchase_price + repair_cost
  let profit_multiplier := 1 + profit_percentage
  ((selling_price : ℚ) / profit_multiplier - total_cost_before_transport).floor.toNat

theorem transportation_charges_calculation :
  calculate_transportation_charges 11000 5000 (1/2) 25500 = 1000 := by
  sorry

#eval calculate_transportation_charges 11000 5000 (1/2) 25500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_charges_calculation_l626_62658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_length_is_six_l626_62627

/-- Represents the dimensions and properties of a rectangular cistern -/
structure Cistern where
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the length of a cistern given its properties -/
noncomputable def cisternLength (c : Cistern) : ℝ :=
  (c.wetSurfaceArea - 2 * c.width * c.depth) / (c.width + 2 * c.depth)

/-- Theorem stating that a cistern with given properties has a length of 6 meters -/
theorem cistern_length_is_six (c : Cistern) 
    (h_width : c.width = 5)
    (h_depth : c.depth = 1.25)
    (h_area : c.wetSurfaceArea = 57.5) :
    cisternLength c = 6 := by
  sorry

#check cistern_length_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_length_is_six_l626_62627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_and_lines_l626_62695

-- Define the types for planes and lines in 3D space
variable (Point Line Plane : Type)

-- Define the perpendicular relation
variable (perp : ∀ (A B : Type), A → B → Prop)

-- Define the "outside of" relation
variable (outside : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_and_lines
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : m ≠ n)
  (h_m_outside : outside m α ∧ outside m β)
  (h_n_outside : outside n α ∧ outside n β) :
  ((perp Plane Plane α β ∧ perp Line Plane m β ∧ perp Line Plane n α) → perp Line Line m n) ∨
  ((perp Line Line m n ∧ perp Line Plane m β ∧ perp Line Plane n α) → perp Plane Plane α β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_and_lines_l626_62695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_membership_l626_62603

/-- Represents a committee in the organization -/
def Committee : Type := ℕ

/-- Represents a member of the organization -/
def Member : Type := ℕ

/-- The number of committees in the organization -/
def num_committees : ℕ := 5

/-- A function that returns the committees a member belongs to -/
def member_committees : Member → Finset Committee := sorry

/-- A function that returns the common member between two committees -/
def common_member : Committee → Committee → Member := sorry

/-- The total number of members in the organization -/
def total_members : ℕ := sorry

theorem organization_membership :
  (∀ m : Member, (member_committees m).card = 2) →
  (∀ c₁ c₂ : Committee, c₁ ≠ c₂ → ∃! m : Member, m ∈ (member_committees m) ∧ c₁ ∈ (member_committees m) ∧ c₂ ∈ (member_committees m)) →
  total_members = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_membership_l626_62603
