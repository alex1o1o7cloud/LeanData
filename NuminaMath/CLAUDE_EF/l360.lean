import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_side_sum_l360_36035

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧  -- Positive sides
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧  -- Positive angles
  t.A + t.B + t.C = Real.pi ∧  -- Sum of angles
  t.b^2 = t.a * t.c ∧  -- Geometric progression
  Real.cos t.B = 3/5

-- Theorem 1
theorem cosine_sine_sum (t : Triangle) (h : triangle_conditions t) :
  (Real.cos t.A / Real.sin t.A) + (Real.cos t.C / Real.sin t.C) = 5/4 := by
  sorry

-- Theorem 2
theorem side_sum (t : Triangle) (h : triangle_conditions t) 
  (h2 : t.a * t.c * Real.cos t.B = 3) :
  t.a + t.c = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_side_sum_l360_36035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_circular_sections_imply_sphere_l360_36057

/-- A solid is a three-dimensional geometric shape -/
structure Solid where
  -- Add necessary fields here

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- Add necessary fields here

/-- A circle is a curved line with points always the same distance from a center point -/
structure Circle where
  -- Add necessary fields here

/-- A point in three-dimensional space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the cross-section of a solid when cut by a plane -/
def CrossSection (s : Solid) (p : Plane) : Set Point := sorry

/-- Predicate to check if a set of points forms a circle -/
def IsCircle (s : Set Point) : Prop := sorry

/-- Predicate to check if a solid is a sphere -/
def IsSphere (s : Solid) : Prop := sorry

/-- Theorem stating that if all cross-sections of a solid are circles, then it must be a sphere -/
theorem all_circular_sections_imply_sphere (s : Solid) :
  (∀ p : Plane, IsCircle (CrossSection s p)) → IsSphere s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_circular_sections_imply_sphere_l360_36057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l360_36017

/-- An odd function f with specific properties -/
noncomputable def f (a : ℝ) : ℝ → ℝ :=
  fun x => if x > 0 ∧ x < 2 then Real.log x - a * x else 0

theorem odd_function_property (a : ℝ) (h : a > 1/2) :
  (∀ x, f a x = -f a (-x)) →
  (∀ x ∈ Set.Ioo (-2 : ℝ) 0, f a x ≥ 1) →
  (∃ x ∈ Set.Ioo (-2 : ℝ) 0, f a x = 1) →
  a = 1 := by
  sorry

#check odd_function_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l360_36017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_books_purchasers_l360_36078

-- Define variables
def only_A : ℕ := 1000  -- Number of people who purchased only book A
def only_B : ℕ := 250   -- Number of people who purchased only book B
def both : ℕ := 500     -- Number of people who purchased both books A and B
def total_A : ℕ := only_A + both  -- Total number of people who purchased book A
def total_B : ℕ := only_B + both  -- Total number of people who purchased book B

-- State the theorem
theorem both_books_purchasers : 
  both = 2 * only_B ∧                -- Condition 2
  total_A = 2 * total_B ∧            -- Condition 1
  total_A = only_A + both ∧          -- Definition of total_A
  total_B = only_B + both ∧          -- Definition of total_B
  only_A = 1000                      -- Condition 3
  → both = 500 := by                 -- Conclusion to prove
  intro h
  sorry  -- Skip the proof for now

#eval both  -- This will evaluate to 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_books_purchasers_l360_36078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_a_2n_equals_f_n_squared_l360_36025

/-- Sequence a_n representing the number of positive integers with digit sum n using digits 1, 3, 4 -/
def a : ℕ → ℕ :=
  sorry

/-- Fibonacci-like sequence f_n -/
def f : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => f (n + 1) + f n

/-- The main theorem: a_{2n} is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k, a (2 * n) = k ^ 2 := by
  use f n
  sorry

/-- The specific form of a_{2n} -/
theorem a_2n_equals_f_n_squared (n : ℕ) : a (2 * n) = (f n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_a_2n_equals_f_n_squared_l360_36025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l360_36093

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem sine_function_properties (φ : ℝ) (h1 : -π < φ) (h2 : φ < 0) 
  (h3 : ∀ x, f x φ = f (-π/6 - x) φ) : 
  (φ = -π/3) ∧ 
  (∀ k : ℤ, ∀ x : ℝ, k * π - π/12 ≤ x ∧ x ≤ k * π + 5*π/12 → 
    ∀ y : ℝ, k * π - π/12 ≤ y ∧ y ≤ x → f x φ ≥ f y φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l360_36093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PM_is_5_l360_36074

/-- Square ABCD with side length 10 and point E on DC -/
structure SquareWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  h_square : A = (0, 10) ∧ B = (10, 10) ∧ C = (10, 0) ∧ D = (0, 0)
  h_E_on_DC : E.1 = 3 ∧ E.2 = 0

/-- M is the midpoint of AE -/
noncomputable def M (s : SquareWithPoint) : ℝ × ℝ :=
  ((s.A.1 + s.E.1) / 2, (s.A.2 + s.E.2) / 2)

/-- P is the intersection of the perpendicular bisector of AE with AD -/
noncomputable def P (s : SquareWithPoint) : ℝ × ℝ :=
  (53/3, 10)

/-- The length of PM is 5 -/
theorem length_PM_is_5 (s : SquareWithPoint) : 
  Real.sqrt ((P s).1 - (M s).1)^2 + ((P s).2 - (M s).2)^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PM_is_5_l360_36074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l360_36092

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2*x^2 + 2*x + 41) - Real.sqrt (2*x^2 + 4*x + 4)

/-- Theorem stating that the maximum value of f(x) is 5 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l360_36092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pirate_coins_l360_36075

/-- Represents the number of pirates --/
def num_pirates : ℕ := 10

/-- Calculates the fraction of remaining coins taken by the kth pirate --/
def pirate_share (k : ℕ) : ℚ := (2 * k) / 10

/-- Calculates the remaining coins after the kth pirate takes their share --/
def remaining_coins (initial_coins : ℕ) : ℕ → ℚ
  | 0 => initial_coins
  | k + 1 => (1 - pirate_share (k + 1)) * remaining_coins initial_coins k

/-- The smallest initial number of coins such that each pirate receives a whole number --/
def initial_coins : ℕ := 1953125  -- 5^9

/-- The theorem to be proved --/
theorem tenth_pirate_coins : 
  (remaining_coins initial_coins (num_pirates - 1)).num = 362880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pirate_coins_l360_36075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_expression_l360_36012

theorem integer_part_of_expression (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  ⌊(3 : ℝ)^((Real.cos x)^2) + (3 : ℝ)^((Real.sin x)^5)⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_expression_l360_36012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_sums_l360_36052

variable (A : Set ℝ)
variable (x₁ x₂ x₃ x₄ : ℝ)

def A_minus (A : Set ℝ) : Set ℝ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = |a - b|}

theorem equality_of_sums (h₁ : A = {x₁, x₂, x₃, x₄})
                         (h₂ : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
                         (h₃ : A_minus A = A) :
  x₁ + x₄ = x₂ + x₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_sums_l360_36052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_radius_is_8_l360_36069

/-- The radius of a sphere -/
def sphere_radius : ℝ := 12

/-- The length of the wire -/
def wire_length : ℝ := 36

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder (wire) -/
noncomputable def wire_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The theorem stating that the radius of the wire's cross section is 8 cm -/
theorem wire_radius_is_8 :
  ∃ (r : ℝ), r = 8 ∧ sphere_volume sphere_radius = wire_volume r wire_length := by
  sorry

#check wire_radius_is_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_radius_is_8_l360_36069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_concentric_circles_l360_36090

/-- Two concentric circles with center O -/
structure ConcentricCircles where
  center : Point
  inner_radius : ℝ
  outer_radius : ℝ
  inner_radius_pos : 0 < inner_radius
  outer_radius_pos : 0 < outer_radius
  inner_smaller : inner_radius < outer_radius

/-- The length of an arc given its central angle and circle radius -/
noncomputable def arcLength (angle : ℝ) (radius : ℝ) : ℝ := (angle / (2 * Real.pi)) * (2 * Real.pi * radius)

/-- The area of a circle given its radius -/
noncomputable def circleArea (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Theorem stating the ratio of areas of concentric circles -/
theorem area_ratio_of_concentric_circles (c : ConcentricCircles) 
  (h : arcLength (Real.pi / 3) c.inner_radius = arcLength (Real.pi / 6) c.outer_radius) :
  circleArea c.inner_radius / circleArea c.outer_radius = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_concentric_circles_l360_36090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l360_36044

/-- The number of digits in the decimal representation of a natural number -/
def numDigits (a : ℕ) : ℕ :=
  if a = 0 then 1 else Nat.log a 10 + 1

/-- Theorem stating that there are no natural numbers a, n, and m satisfying the given conditions -/
theorem no_solution_exists : ¬ ∃ (a n m : ℕ), 
  (numDigits a = n) ∧ 
  (numDigits (a^3) = m) ∧ 
  (n + m = 2001) := by
  sorry

#check no_solution_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l360_36044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l360_36008

/-- An ellipse with semi-major axis 5 and semi-minor axis 3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 9) = 1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The two foci of the ellipse -/
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

/-- A point on the ellipse with x-coordinate 4 -/
def M : ℝ × ℝ := (4, 3)

theorem ellipse_triangle_perimeter :
  M ∈ Ellipse →
  distance F₁ M + distance M F₂ + distance F₁ F₂ = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l360_36008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l360_36036

/-- Definition of line l₁ -/
noncomputable def l₁ (t k : ℝ) : ℝ × ℝ := (1 - 2*t, 2 + k*t)

/-- Definition of line l₂ -/
noncomputable def l₂ (s : ℝ) : ℝ × ℝ := (s, 1 - 2*s)

/-- Slope of line l₁ -/
noncomputable def slope_l₁ (k : ℝ) : ℝ := k / 2

/-- Slope of line l₂ -/
noncomputable def slope_l₂ : ℝ := -2

/-- Theorem: If l₁ is parallel to l₂, then k = 4 -/
theorem parallel_lines (k : ℝ) : slope_l₁ k = slope_l₂ → k = 4 := by
  sorry

/-- Theorem: If l₁ is perpendicular to l₂, then k = -1 -/
theorem perpendicular_lines (k : ℝ) : slope_l₁ k * slope_l₂ = -1 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l360_36036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l360_36030

def M : Set ℝ := {x : ℝ | x^2 < 36}
def N : Set ℝ := {2, 4, 6, 8}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l360_36030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l360_36009

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x - Real.cos x)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : f t.B = 1/2)
  (h2 : t.a + t.c = 1) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (t.b ≥ 1/2 ∧ t.b < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l360_36009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l360_36014

/-- The distance between two parallel lines is equal to the absolute difference of their constants divided by the square root of the sum of squares of their coefficients. -/
noncomputable def distance_parallel_lines (A B c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (A^2 + B^2)

/-- The distance between the parallel lines 4x + 3y + 1 = 0 and 4x + 3y - 9 = 0 is 2. -/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 4 3 1 (-9) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l360_36014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l360_36050

theorem solve_exponential_equation (x : ℝ) :
  3 * (2 : ℝ)^x = 768 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l360_36050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_count_l360_36085

/-- The number of baskets Lilibeth filled -/
def baskets : ℕ := 6

/-- The number of strawberries in each basket -/
def strawberries_per_basket : ℕ := 50

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := 1200

/-- The number of strawberries Lilibeth picked -/
def lilibeth_strawberries : ℕ := baskets * strawberries_per_basket

/-- The number of friends who picked strawberries with Lilibeth -/
theorem friends_count : (total_strawberries - lilibeth_strawberries) / lilibeth_strawberries = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_count_l360_36085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorizable_n_l360_36026

/-- The largest value of n for which 6x^2 + nx + 72 can be factored with integer coefficients -/
def largest_n : ℕ := 433

/-- Predicate to check if a quadratic expression can be factored with integer coefficients -/
def can_be_factored (a b c : ℤ) : Prop :=
  ∃ (A B : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (a * x + A) * (x + B)

theorem largest_factorizable_n :
  (∀ n : ℕ, n > largest_n → ¬(can_be_factored 6 (n : ℤ) 72)) ∧
  (can_be_factored 6 (largest_n : ℤ) 72) := by
  sorry

#check largest_factorizable_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorizable_n_l360_36026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_reciprocal_l360_36056

theorem x_power_n_plus_reciprocal (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) : 
  x ^ n + (1 / x) ^ n = 2 * Real.sin (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_reciprocal_l360_36056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_def_l360_36099

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Calculate the area of a triangle given three points -/
noncomputable def triangle_area (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- The theorem to be proved -/
theorem area_of_triangle_def (d e f : Square) : 
  d.side_length = 2 → 
  e.side_length = 2 → 
  f.side_length = 2 → 
  (e.center.x - d.center.x)^2 + (e.center.y - d.center.y)^2 = 4 → 
  (f.center.x - e.center.x)^2 + (f.center.y - e.center.y)^2 = 4 → 
  triangle_area d.center e.center f.center = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_def_l360_36099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l360_36031

theorem rectangle_area_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (((L * 1.03) * (B * 1.06) - L * B) / (L * B)) * 100 = 9.18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l360_36031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_negative_is_correct_l360_36068

/-- The original proposition -/
def original_prop (x : ℝ) : Prop := x ≥ 1 → (2 : ℝ)^x + 1 ≥ 3

/-- The converse negative proposition -/
def converse_negative_prop (x : ℝ) : Prop := (2 : ℝ)^x + 1 < 3 → x < 1

/-- Theorem stating that the converse negative proposition is correct -/
theorem converse_negative_is_correct :
  (∀ x : ℝ, original_prop x) ↔ (∀ x : ℝ, converse_negative_prop x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_negative_is_correct_l360_36068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l360_36038

/-- Given two distinct real roots of the equation x^2 + x/tan(θ) - 1/sin(θ) = 0,
    prove that the line passing through the points (a, a^2) and (b, b^2)
    is tangent to the circle x^2 + y^2 = 1 -/
theorem line_tangent_to_circle (θ : ℝ) (a b : ℝ) :
  a ≠ b →
  (a^2 + a / Real.tan θ - 1 / Real.sin θ = 0) →
  (b^2 + b / Real.tan θ - 1 / Real.sin θ = 0) →
  let line_eq := λ (x y : ℝ) => y = (a + b) * (x - (a + b) / 2) + (a^2 + b^2) / 2
  ∃! (p : ℝ × ℝ), line_eq p.1 p.2 ∧ p.1^2 + p.2^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l360_36038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l360_36094

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / x

theorem f_is_even : ∀ x : ℝ, x ≠ 0 → f (-x) = f x := by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l360_36094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l360_36081

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  a_1_eq_2 : a 1 = 2
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a (n + 2) - a (n + 1)
  is_geometric : (a 5 / a 3) = (a 8 / a 5)
  nonzero_diff : ∀ n : ℕ+, a (n + 1) - a n ≠ 0

/-- The b_n sequence derived from a_n -/
noncomputable def b (a : ℕ+ → ℝ) : ℕ+ → ℝ := fun n => a n + (2 : ℝ)^(a n)

/-- The sum of the first n terms of b_n -/
noncomputable def T (a : ℕ+ → ℝ) : ℕ → ℝ := fun n => 
  Finset.sum (Finset.range n) (fun i => b a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = n + 1) ∧
  (∀ n : ℕ, T seq.a n = (2 : ℝ)^(n + 2) + (n^2 : ℝ)/2 + (3 * n : ℝ)/2 - 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l360_36081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birgit_speed_difference_l360_36082

-- Define the hiking parameters
noncomputable def hiking_time : ℝ := 3.5
noncomputable def hiking_distance : ℝ := 21

-- Define Birgit's parameters
noncomputable def birgit_distance : ℝ := 8
noncomputable def birgit_time_minutes : ℝ := 48

-- Calculate average group speed
noncomputable def average_speed : ℝ := hiking_distance / hiking_time

-- Calculate Birgit's speed
noncomputable def birgit_speed : ℝ := birgit_distance / (birgit_time_minutes / 60)

-- Theorem to prove
theorem birgit_speed_difference : birgit_speed - average_speed = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birgit_speed_difference_l360_36082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosB_is_seven_twentyfifth_l360_36058

/-- A right triangle ABC with altitude CD meeting AB at D -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  altitude_meets_AB : D.1 = C.1 ∧ (D.2 - A.2) / (B.2 - A.2) = (D.1 - A.1) / (B.1 - A.1)

/-- The lengths of sides AB, BC, AC are integers -/
def integer_sides (t : RightTriangle) : Prop :=
  ∃ (ab bc ac : ℤ), 
    (t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2 = ab^2 ∧
    (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2 = bc^2 ∧
    (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = ac^2

/-- The length of BD is 7³ -/
def BD_length (t : RightTriangle) : Prop :=
  (t.D.1 - t.B.1)^2 + (t.D.2 - t.B.2)^2 = 7^6

theorem cosB_is_seven_twentyfifth (t : RightTriangle) 
  (h1 : integer_sides t) (h2 : BD_length t) : 
  (t.C.1 - t.B.1) / Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 7/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosB_is_seven_twentyfifth_l360_36058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l360_36045

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- The x-axis -/
def XAxis : Set Point := {p : Point | p.2 = 0}

/-- Tangent line to a circle at a point -/
noncomputable def TangentLine (ω : Circle) (p : Point) : Set Point := sorry

/-- Intersection of two sets of points -/
def Intersect (s t : Set Point) : Set Point := s ∩ t

/-- Area of a circle -/
noncomputable def Area (ω : Circle) : ℝ := Real.pi * ω.radius ^ 2

/-- Main theorem -/
theorem circle_area_theorem (ω : Circle) (A B : Point) :
  A ∈ {p : Point | (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2} ∧
  A.1 = 4 ∧ A.2 = 12 ∧
  B ∈ {p : Point | (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2} ∧
  B.1 = 8 ∧ B.2 = 8 ∧
  (Intersect (TangentLine ω A) (TangentLine ω B) ⊆ XAxis) →
  Area ω = 208 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l360_36045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l360_36043

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℝ) : ℝ := (1/4) * n * (n+1) * (n+2) * (n+3)

/-- Theorem stating that g(r) - g(r-1) = r * (r+1) * (r+2) -/
theorem g_difference (r : ℝ) : g r - g (r-1) = r * (r+1) * (r+2) := by
  -- Expand the definition of g
  unfold g
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l360_36043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tobacco_acreage_increase_l360_36088

/-- Calculates the increase in tobacco acreage after changing crop ratios --/
theorem tobacco_acreage_increase (total_land : ℝ) (initial_ratio : Fin 3 → ℝ) (new_ratio : Fin 3 → ℝ) : 
  total_land = 1350 ∧ 
  initial_ratio = ![5, 2, 2] ∧ 
  new_ratio = ![2, 2, 5] →
  (new_ratio 2 / (new_ratio 0 + new_ratio 1 + new_ratio 2) * total_land) - 
  (initial_ratio 2 / (initial_ratio 0 + initial_ratio 1 + initial_ratio 2) * total_land) = 450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tobacco_acreage_increase_l360_36088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_fraction_approx_l360_36077

/-- Represents the fraction of milk remaining after one replacement operation -/
noncomputable def replace_milk (milk_fraction : ℝ) (replace_percent : ℝ) : ℝ :=
  milk_fraction * (1 - replace_percent / 100)

/-- Represents the fraction of milk remaining after one cycle of three operations -/
noncomputable def cycle (milk_fraction : ℝ) : ℝ :=
  replace_milk (replace_milk (replace_milk milk_fraction 20) 30) 40

/-- The final fraction of milk after 9 operations (3 cycles) -/
noncomputable def final_milk_fraction : ℝ :=
  cycle (cycle (cycle 1))

/-- Theorem stating that the final milk fraction is approximately 0.037933056 -/
theorem final_milk_fraction_approx :
  |final_milk_fraction - 0.037933056| < 0.000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_fraction_approx_l360_36077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_first_year_prob_distribution_X_expectation_X_l360_36097

-- Define the total number of students and the number of students from each year
def total_students : ℕ := 15
def first_year_students : ℕ := 7
def second_year_students : ℕ := 6
def third_year_students : ℕ := 2

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define the probabilities for Zhang and Wang
def zhang_prob : ℚ := 1/2
def wang_prob : ℚ := 2/3

-- Define the number of questions in the exam
def num_questions : ℕ := 3

-- Define the minimum number of correct answers to pass
def pass_threshold : ℕ := 2

-- Define X as the number of students who pass the exam
def X : Fin 3 → ℚ
| 0 => 7/54
| 1 => 1/2
| 2 => 10/27

-- Theorem for the probability of selecting exactly 2 first-year students
theorem prob_two_first_year : 
  (Nat.choose first_year_students 2 * Nat.choose (second_year_students + third_year_students) 1) / 
  (Nat.choose total_students selected_students) = 24/65 := by sorry

-- Theorem for the probability distribution of X
theorem prob_distribution_X : 
  X 0 = 7/54 ∧ X 1 = 1/2 ∧ X 2 = 10/27 := by sorry

-- Theorem for the mathematical expectation of X
theorem expectation_X : 
  (0 * X 0 + 1 * X 1 + 2 * X 2) = 67/54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_first_year_prob_distribution_X_expectation_X_l360_36097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_matching_pair_standard_deck_l360_36051

/-- Represents a standard deck of playing cards -/
structure Deck where
  total_cards : Nat
  ranks : Nat
  cards_per_rank : Nat
  h1 : total_cards = ranks * cards_per_rank

/-- Represents the deck after a pair is removed -/
def RemainingDeck (d : Deck) : Deck where
  total_cards := d.total_cards - 2
  ranks := d.ranks
  cards_per_rank := d.cards_per_rank
  h1 := by sorry  -- Proof that the equation still holds

/-- The probability of selecting a matching pair from the remaining deck -/
def probability_matching_pair (d : Deck) : ℚ :=
  let remaining := RemainingDeck d
  let total_combinations := Nat.choose remaining.total_cards 2
  let matching_combinations := (remaining.ranks - 1) * Nat.choose remaining.cards_per_rank 2 + 1
  matching_combinations / total_combinations

def standard_deck : Deck where
  total_cards := 52
  ranks := 13
  cards_per_rank := 4
  h1 := rfl

theorem probability_matching_pair_standard_deck :
  probability_matching_pair standard_deck = 73 / 1225 := by
  sorry

#eval (73 + 1225 : Nat)  -- Sum of numerator and denominator

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_matching_pair_standard_deck_l360_36051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_impossible_l360_36033

/-- A bag containing balls of different colors -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a ball of a specific color from a bag -/
noncomputable def probability (bag : Bag) (color : String) : ℝ :=
  match color with
  | "white" => (bag.white : ℝ) / ((bag.white + bag.red) : ℝ)
  | "red" => (bag.red : ℝ) / ((bag.white + bag.red) : ℝ)
  | _ => 0

/-- Theorem: The probability of drawing a red ball from a bag containing only 8 white balls is 0 -/
theorem red_ball_impossible (bag : Bag) (h : bag.white = 8 ∧ bag.red = 0) :
  probability bag "red" = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_impossible_l360_36033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_arrives_20_minutes_before_jack_l360_36048

/-- The distance to the pool in miles -/
noncomputable def distance_to_pool : ℝ := 2

/-- Jill's speed in miles per hour -/
noncomputable def jill_speed : ℝ := 12

/-- Jack's speed in miles per hour -/
noncomputable def jack_speed : ℝ := 4

/-- Convert hours to minutes -/
noncomputable def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Calculate travel time in hours given distance and speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem jill_arrives_20_minutes_before_jack :
  let jill_time := travel_time distance_to_pool jill_speed
  let jack_time := travel_time distance_to_pool jack_speed
  hours_to_minutes (jack_time - jill_time) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_arrives_20_minutes_before_jack_l360_36048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ao_limit_theorem_l360_36039

/-- A convex quadrilateral with given side lengths and diagonal intersection properties -/
structure ConvexQuadrilateral (a b p : ℝ) where
  convex : Bool
  ab_length : ℝ := a
  ad_length : ℝ := b
  bc_length : ℝ := p - a
  dc_length : ℝ := p - b
  o_is_diagonal_intersection : Bool

/-- The limit of |AO| as angle BAC approaches zero in a ConvexQuadrilateral -/
noncomputable def ao_limit (a b p : ℝ) (quad : ConvexQuadrilateral a b p) : ℝ :=
  (p * Real.sqrt (a * b)) / (Real.sqrt (a * b) + Real.sqrt ((p - a) * (p - b)))

/-- The length of AO as a function of angle α -/
noncomputable def ao_length (a b p : ℝ) (quad : ConvexQuadrilateral a b p) (α : ℝ) : ℝ :=
  sorry -- This function would be defined based on the geometry of the quadrilateral

/-- Theorem stating the limit of |AO| as angle BAC approaches zero -/
theorem ao_limit_theorem (a b p : ℝ) (quad : ConvexQuadrilateral a b p) :
  ∀ ε > 0, ∃ δ > 0, ∀ α, 0 < α ∧ α < δ → |ao_length a b p quad α - ao_limit a b p quad| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ao_limit_theorem_l360_36039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_minimum_length_l360_36070

-- Define the curves C1 and C2
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (3 + Real.cos θ, 4 + Real.sin θ)

noncomputable def C2 (k : ℝ) (θ : ℝ) : ℝ := 3 / (Real.sin θ - k * Real.cos θ)

-- Define the distance function between a point and C1's center
noncomputable def distance_to_C1_center (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 3)^2 + (y - 4)^2)

-- Define the tangent length function
noncomputable def tangent_length (k : ℝ) : ℝ :=
  let d := |4 - 3*k| / Real.sqrt (k^2 + 1)
  Real.sqrt (d^2 - 1)

-- State the theorem
theorem tangent_minimum_length (k : ℝ) :
  (∃ θ : ℝ, distance_to_C1_center (C2 k θ * Real.cos θ) (C2 k θ * Real.sin θ) > 1) →
  tangent_length k ≥ 2 * Real.sqrt 2 →
  k = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_minimum_length_l360_36070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dawn_hourly_rate_l360_36006

/-- Dawn's painting rate in hours per painting -/
noncomputable def painting_rate : ℚ := 2

/-- Number of paintings commissioned -/
def num_paintings : ℕ := 12

/-- Total earnings for the commissioned paintings in dollars -/
noncomputable def total_earnings : ℚ := 3600

/-- Calculate Dawn's hourly rate in dollars per hour -/
noncomputable def hourly_rate : ℚ := total_earnings / (painting_rate * num_paintings)

theorem dawn_hourly_rate :
  hourly_rate = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dawn_hourly_rate_l360_36006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l360_36066

/-- A function f(x) = (x-b)ln(x) is monotonically increasing on [1,e] if and only if b ∈ (-∞, 1] -/
theorem monotone_increasing_condition (b : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), Monotone (fun x => (x - b) * Real.log x)) ↔ b ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l360_36066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_properties_l360_36060

/-- Represents a cubical water tank -/
structure CubicalTank where
  initialVolume : ℝ
  initialLevel : ℝ
  fillRate : ℝ

/-- Calculates the capacity of the tank -/
noncomputable def tankCapacity (tank : CubicalTank) : ℝ :=
  (tank.initialVolume / tank.initialLevel) ^ 3

/-- Calculates the fraction of the tank filled initially -/
noncomputable def initialFraction (tank : CubicalTank) : ℝ :=
  tank.initialVolume / tankCapacity tank

/-- Calculates the time to fill the tank completely -/
noncomputable def timeToFill (tank : CubicalTank) : ℝ :=
  (tankCapacity tank - tank.initialVolume) / tank.fillRate

/-- Theorem stating the properties of the specific tank described in the problem -/
theorem water_tank_properties (tank : CubicalTank) 
  (h1 : tank.initialVolume = 16)
  (h2 : tank.initialLevel = 1)
  (h3 : tank.fillRate = 2) :
  initialFraction tank = 1/4 ∧ timeToFill tank = 24 := by
  sorry

-- Remove the #eval line as it's not necessary for the proof and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_properties_l360_36060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l360_36076

/-- The distance between each pair of points A, B, and C -/
def S : ℝ := 210

/-- The speed of the car traveling from A to B in km/h -/
def car_speed : ℝ := 90

/-- The speed of the train traveling from B to C in km/h -/
def train_speed : ℝ := 60

/-- The time at which the car and train are at their shortest distance in hours -/
def t : ℝ := 2

/-- The distance between the car and the train as a function of time -/
noncomputable def r (t : ℝ) : ℝ := 
  Real.sqrt (S^2 - 120 * S * t + 6300 * t^2)

theorem distance_between_points : 
  (∀ t : ℝ, r t ≥ r 2) → S = 210 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l360_36076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_functions_l360_36055

-- Define the interval (2, +∞)
def interval := Set.Ioi (2 : ℝ)

-- Define the functions
noncomputable def f_A (x : ℝ) := x + 1/x
noncomputable def f_B (x : ℝ) := x - 1/x
noncomputable def f_C (x : ℝ) := 1/(4-x)
noncomputable def f_D (x : ℝ) := Real.sqrt (x^2 - 4*x + 3)

-- Statement to prove
theorem monotone_increasing_functions :
  (StrictMono (fun x => f_A x)) ∧
  (StrictMono (fun x => f_B x)) ∧
  ¬(StrictMono (fun x => f_C x)) ∧
  ¬(StrictMono (fun x => f_D x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_functions_l360_36055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l360_36016

/-- The function f(x) = 1 / (x-1)^2 + 1 -/
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)^2 + 1

theorem f_range :
  Set.range f = Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l360_36016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_after_two_hours_l360_36049

/-- The time when two drivers meet, given their speeds and a delay for the second driver. -/
noncomputable def meeting_time (speed1 speed2 delay : ℝ) : ℝ :=
  (speed1 * delay) / (speed2 - speed1)

/-- Theorem stating that under the given conditions, the meeting time is 2 hours. -/
theorem meeting_after_two_hours (man_speed wife_speed delay : ℝ) 
  (h1 : man_speed = 40)
  (h2 : wife_speed = 50)
  (h3 : delay = 0.5) :
  meeting_time man_speed wife_speed delay = 2 := by
  sorry

#check meeting_after_two_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_after_two_hours_l360_36049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_specific_triangle_l360_36072

/-- A right triangle with sides a, b, and c, where c is the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0

/-- The length of the crease when folding a right triangle along its shortest side -/
noncomputable def creaseLength (t : RightTriangle) : ℝ := min t.a t.b / 2

theorem crease_length_specific_triangle :
  ∃ t : RightTriangle, t.a = 5 ∧ t.b = 12 ∧ t.c = 13 ∧ creaseLength t = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_specific_triangle_l360_36072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_three_max_profit_value_l360_36005

/-- Profit function for a company's technological reform -/
noncomputable def profit (m : ℝ) : ℝ := 28 - m - 16 / (m + 1)

/-- The maximum profit occurs when the investment is 3 -/
theorem max_profit_at_three :
  ∀ m : ℝ, m ≥ 0 → profit m ≤ profit 3 := by
  sorry

/-- The maximum profit is 21 -/
theorem max_profit_value :
  profit 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_three_max_profit_value_l360_36005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_minimal_area_condition_minimal_area_line_equation_l360_36004

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the point E where the directrix intersects the x-axis
def E : ℝ × ℝ := (-1, 0)

-- Define the line l passing through F
def l (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - m * p.2 - 1 = 0}

-- Define points A and B as the intersections of l with C
noncomputable def A (m : ℝ) : ℝ × ℝ := sorry
noncomputable def B (m : ℝ) : ℝ × ℝ := sorry

-- Define the moving point P
noncomputable def P (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (p q r s : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: The trajectory of P satisfies y² = 4x - 12
theorem trajectory_of_P :
  ∀ m : ℝ, (P m).2^2 = 4 * (P m).1 - 12 :=
by sorry

-- Theorem 2: The area of EAPB is minimal when m = 0
theorem minimal_area_condition :
  ∃ m₀ : ℝ, ∀ m : ℝ,
    area_quadrilateral E (A m) (P m) (B m) ≥ area_quadrilateral E (A m₀) (P m₀) (B m₀) ∧
    m₀ = 0 :=
by sorry

-- Theorem 3: When the area is minimal, the equation of l is x - 1 = 0
theorem minimal_area_line_equation :
  ∀ m : ℝ,
    (∀ m' : ℝ, area_quadrilateral E (A m) (P m) (B m) ≤ area_quadrilateral E (A m') (P m') (B m')) →
    l m = {p : ℝ × ℝ | p.1 - 1 = 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_minimal_area_condition_minimal_area_line_equation_l360_36004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l360_36062

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 ≥ a

-- Define proposition q
def q (a : ℝ) : Prop := ∃ ρ α : ℝ, (ρ * Real.cos α)^2 - (ρ * Real.sin α)^2 = a + 2

-- We'll remove the IsHyperbolaWithFociOnXAxis definition as it's not essential for the structure of the problem

theorem problem_statement (a : ℝ) : 
  (p a → a ∈ Set.Iic 1) ∧ 
  (p a ∧ q a → a ∈ Set.Ioo (-2) 1) :=
by
  sorry -- We'll use sorry to skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l360_36062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_shaded_area_l360_36067

/-- Given a square with side length x and midpoints on two adjacent sides,
    the fraction of the square's area not covered by the triangle formed
    by these midpoints and the opposite corner is 7/8 -/
theorem square_shaded_area (x : ℝ) (hx : x > 0) : 
  (x^2 - (1/2) * (x/2) * (x/2)) / x^2 = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_shaded_area_l360_36067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_third_pair_is_basis_l360_36029

/-- Definition of the vector pairs --/
def vector_pairs : List ((ℝ × ℝ) × (ℝ × ℝ)) :=
  [((0, 0), (1, -2)),
   ((-1, -2), (3, 6)),
   ((3, -5), (6, 10)),
   ((2, -3), (-2, 3))]

/-- Function to check if two vectors are linearly independent --/
def is_linearly_independent (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 ≠ v1.2 * v2.1

/-- Theorem stating that only the third pair forms a basis --/
theorem only_third_pair_is_basis :
  ∃! i : Fin 4, is_linearly_independent (vector_pairs[i].1) (vector_pairs[i].2) ∧ i = 2 := by
  sorry

#check only_third_pair_is_basis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_third_pair_is_basis_l360_36029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equal_sums_l360_36046

theorem partition_equal_sums (n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1) ∧ a (i + 1) ≤ 2 * a i)
  (h3 : Even (Finset.sum (Finset.range n) (λ i => a (i + 1)))) :
  ∃ (S : Finset ℕ), 
    S ⊆ Finset.range n ∧ 
    Finset.sum S (λ i => a (i + 1)) = Finset.sum (Finset.range n \ S) (λ i => a (i + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equal_sums_l360_36046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_l360_36003

/-- Two lines in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A point lies on a line -/
def pointOnLine (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = l.point + t • l.direction

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Skew relation between two lines -/
def skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ p : ℝ × ℝ × ℝ, pointOnLine p l1 ∧ pointOnLine p l2) ∧ ¬ parallel l1 l2

/-- Intersect relation between two lines -/
def intersect (l1 l2 : Line3D) : Prop :=
  ∃ p : ℝ × ℝ × ℝ, pointOnLine p l1 ∧ pointOnLine p l2

theorem line_relationship (a b l : Line3D) 
  (h_parallel : parallel a b) 
  (h_skew : skew l a) : 
  intersect l b ∨ skew l b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_l360_36003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l360_36047

/-- The speed of sound in feet per second -/
def speed_of_sound : ℚ := 1100

/-- The time delay between lightning flash and thunder in seconds -/
def time_delay : ℚ := 12

/-- The number of feet in a mile -/
def feet_per_mile : ℚ := 5280

/-- Rounds a rational number to the nearest quarter -/
def round_to_nearest_quarter (x : ℚ) : ℚ :=
  ⌊x * 4 + 1/2⌋ / 4

/-- The theorem stating the distance to the lightning strike -/
theorem lightning_distance : 
  round_to_nearest_quarter ((speed_of_sound * time_delay) / feet_per_mile) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l360_36047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l360_36053

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  Real.arccos ((distance p1 p2)^2 + (distance p2 p3)^2 - (distance p1 p3)^2) / (2 * distance p1 p2 * distance p2 p3)

theorem ellipse_angle_theorem (e : Ellipse) (f1 f2 p : Point) :
  e.a^2 = 9 ∧ e.b^2 = 2 ∧
  isOnEllipse p e ∧
  distance p f1 = 4 →
  angle f1 p f2 = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l360_36053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_6_is_negative_21_l360_36083

def geometric_sequence (a₁ : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^(n - 1)

def sum_geometric_sequence (a₁ : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  if r = 1 then
    a₁ * n
  else
    a₁ * (1 - r^n) / (1 - r)

theorem sum_first_6_is_negative_21 :
  sum_geometric_sequence 1 (-2) 6 = -21 := by
  -- Proof goes here
  sorry

#eval sum_geometric_sequence 1 (-2) 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_6_is_negative_21_l360_36083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_line_parabola_intersection_l360_36022

/-- The sum of distances from a point on a line to its intersections with a parabola --/
theorem distance_sum_line_parabola_intersection :
  ∀ (a : ℝ),
  let l : Set (ℝ × ℝ) := {p | p.1 + p.2 = a}
  let C : Set (ℝ × ℝ) := {p | p.2^2 = 2 * p.1}
  let P : ℝ × ℝ := (1, 1)
  ∃ A B : ℝ × ℝ,
    A ∈ l ∩ C ∧
    B ∈ l ∩ C ∧
    A ≠ B ∧
    P ∈ l →
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) =
    2 * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_line_parabola_intersection_l360_36022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_and_slopes_sum_zero_l360_36037

/-- The locus of points P such that the product of slopes of PA and PB is -3/4 -/
def Locus (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x ≠ 2 ∧ x ≠ -2 ∧ ((y / (x + 2)) * (y / (x - 2))) = (-3/4)

/-- The ellipse defined by x²/4 + y²/3 = 1 -/
def Ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + (y^2 / 3) = 1

/-- A point is inside the ellipse if x²/4 + y²/3 < 1 -/
def InsideEllipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + (y^2 / 3) < 1

/-- Definition of a line passing through a point with a given slope -/
def Line (P : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | let (x₁, y₁) := P; let (x₂, y₂) := Q; y₂ - y₁ = m * (x₂ - x₁)}

/-- Four points form a cyclic quadrilateral -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ P, P ∈ ({A, B, C, D} : Set (ℝ × ℝ)) → dist O P = r

theorem locus_is_ellipse_and_slopes_sum_zero :
  ∀ (P : ℝ × ℝ),
    Locus P ↔ Ellipse P ∧
    ∀ (D : ℝ × ℝ) (m₁ m₂ : ℝ),
      InsideEllipse D →
      (∃ (M N E F : ℝ × ℝ),
        M ∈ Line D m₁ ∧ N ∈ Line D m₁ ∧ E ∈ Line D m₂ ∧ F ∈ Line D m₂ ∧
        Ellipse M ∧ Ellipse N ∧ Ellipse E ∧ Ellipse F ∧
        CyclicQuadrilateral M N E F) →
      m₁ + m₂ = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_and_slopes_sum_zero_l360_36037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l360_36098

-- Define the equation
def equation (x : ℝ) : Prop :=
  1 + (Real.sin x) / (Real.sin (4 * x)) = (Real.sin (3 * x)) / (Real.sin (2 * x))

-- Define the set of solutions
def solutions : Set ℝ :=
  {x | 0 < x ∧ x < Real.pi ∧ equation x}

-- Theorem statement
theorem sum_of_solutions :
  ∃ (S : Finset ℝ), S.toSet = solutions ∧ S.sum id = 320 * Real.pi / 180 := by
  sorry

#check sum_of_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l360_36098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_guess_l360_36023

/-- Represents the color of a bear -/
inductive BearColor
  | White
  | Brown
  | Black
deriving Repr, DecidableEq

/-- A sequence of bear colors -/
def BearSequence := Nat → BearColor

/-- Checks if a subsequence of three bears contains all three colors -/
def hasAllColors (seq : BearSequence) (start : Nat) : Prop :=
  ∃ (c1 c2 c3 : BearColor), 
    ({c1, c2, c3} : Finset BearColor) = {BearColor.White, BearColor.Brown, BearColor.Black} ∧
    ({seq start, seq (start + 1), seq (start + 2)} : Finset BearColor) = {c1, c2, c3}

/-- The main theorem -/
theorem inconsistent_guess (seq : BearSequence) : 
  (∀ n, n < 998 → hasAllColors seq n) →  -- For any 3 consecutive bears, all colors are present
  seq 2 = BearColor.White →              -- 2nd bear is white
  seq 800 = BearColor.White →            -- 800th bear is white
  seq 20 ≠ BearColor.Brown               -- 20th bear cannot be brown
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_guess_l360_36023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l360_36061

/-- Definition of the function f(x) = (x + 2) / (x - 1) -/
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem properties_of_f :
  (∀ x : ℝ, x ≠ 1 → f x ≠ 1) ∧
  (∀ x : ℝ, x ≠ 1 → f (1 + (1 - x)) = f (1 + (x - 1))) ∧
  (∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x > f y) ∧
  (∀ x y : ℝ, x < y ∧ x > 1 ∧ y > 1 → f x > f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l360_36061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_log_l360_36000

theorem arithmetic_sequence_log (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) :
  (∃ d : ℝ, Real.log (a^4 * b^9) + d = Real.log (a^7 * b^17) ∧
            Real.log (a^7 * b^17) + d = Real.log (a^11 * b^26)) →
  (∃ k : ℝ, Real.log (b^n) = Real.log (a^4 * b^9) + 14 * k) →
  n = 167 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_log_l360_36000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_colorings_l360_36034

-- Define a 2x3 grid
def Grid := Fin 2 → Fin 3 → Bool

-- Define a function to check if a grid has exactly 4 colored squares
def has_four_colored (g : Grid) : Prop :=
  (Finset.univ.filter (λ (i : Fin 2 × Fin 3) => g i.1 i.2)).card = 4

-- Define a function to check if each row has at least one colored square
def rows_have_colored (g : Grid) : Prop :=
  ∀ i : Fin 2, ∃ j : Fin 3, g i j

-- Define a function to check if each column has at least one colored square
def cols_have_colored (g : Grid) : Prop :=
  ∀ j : Fin 3, ∃ i : Fin 2, g i j

-- Define rotational equivalence
def rotationally_equivalent (g1 g2 : Grid) : Prop :=
  ∃ n : Fin 4, g1 = (λ i j => g2 ((i + n) % 2) ((j + n) % 3))

-- Define a valid coloring
def valid_coloring (g : Grid) : Prop :=
  has_four_colored g ∧ rows_have_colored g ∧ cols_have_colored g

-- The main theorem
theorem distinct_colorings :
  ∃ (colorings : Finset Grid),
    colorings.card = 5 ∧
    (∀ g ∈ colorings, valid_coloring g) ∧
    (∀ g1 ∈ colorings, ∀ g2 ∈ colorings, g1 ≠ g2 → ¬rotationally_equivalent g1 g2) ∧
    (∀ g, valid_coloring g → ∃ g' ∈ colorings, rotationally_equivalent g g') :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_colorings_l360_36034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l360_36019

/-- A sequence defined by a(1) = 2 and a(n+1) = 3a(n) - 2 for n ≥ 1 -/
def a : ℕ → ℤ
  | 0 => 2  -- We define a(0) to be 2 as well, to cover all natural numbers
  | n + 1 => 3 * a n - 2

/-- The theorem stating that the general term of the sequence is 3^(n-1) + 1 for n ≥ 1 -/
theorem a_general_term (n : ℕ) (h : n ≥ 1) : a n = 3^(n-1) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l360_36019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l360_36091

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := x - 1 + 9 / (x + 1)

-- State the theorem
theorem min_value_of_f :
  ∃ (a : ℝ), a > -1 ∧ (∀ x > -1, f x ≥ f a) ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l360_36091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l360_36063

theorem min_value_of_function :
  ∃ (min_val : ℝ), 
    (∀ (x y : ℝ), -2 < x ∧ x < 2 ∧ -2 < y ∧ y < 2 ∧ x * y = -1 → 
      4 / (4 - x^2) + 9 / (9 - y^2) ≥ min_val) ∧
    (∃ (x y : ℝ), -2 < x ∧ x < 2 ∧ -2 < y ∧ y < 2 ∧ x * y = -1 ∧ 
      4 / (4 - x^2) + 9 / (9 - y^2) = min_val) ∧
    min_val = 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l360_36063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_negative_four_in_second_quadrant_l360_36079

/-- The quadrant in which an angle lies -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines the quadrant of an angle given in radians -/
noncomputable def angle_quadrant (α : ℝ) : Quadrant :=
  if -Real.pi < α && α ≤ -Real.pi/2 then Quadrant.second
  else if -Real.pi/2 < α && α ≤ 0 then Quadrant.fourth
  else if 0 < α && α ≤ Real.pi/2 then Quadrant.first
  else if Real.pi/2 < α && α ≤ Real.pi then Quadrant.second
  else if Real.pi < α && α ≤ 3*Real.pi/2 then Quadrant.third
  else Quadrant.fourth

theorem angle_negative_four_in_second_quadrant :
  angle_quadrant (-4 : ℝ) = Quadrant.second := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_negative_four_in_second_quadrant_l360_36079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_square_perimeters_l360_36027

theorem sum_of_square_perimeters (x : ℝ) (h : x = 3) : 
  (4 * Real.sqrt (x^2 + 4*x + 4)) + (4 * Real.sqrt (4*x^2 - 12*x + 9)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_square_perimeters_l360_36027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l360_36071

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

theorem point_not_on_graph :
  f (-3/2) ≠ 1 ∧
  f 0 = -1/2 ∧
  f 1 = 0 ∧
  f (-1) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l360_36071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_exists_l360_36020

theorem unique_divisor_exists : ∃! D : ℕ, 
  (242 % D = 11) ∧ 
  (698 % D = 18) ∧ 
  ((242 + 698) % D = 9) ∧ 
  (D > 18) ∧
  (D = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_exists_l360_36020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_goals_l360_36021

theorem soccer_goals :
  ∃ K : ℚ,
    K + 2 * K + K / 2 + 4 * K = 15 ∧ K = 2 := by
  use 2
  constructor
  · norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_goals_l360_36021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_interrupt_points_l360_36054

/-- A penalty point system for student misbehavior --/
structure PenaltySystem where
  interrupt_points : ℕ
  insult_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- A student's misbehavior record --/
structure StudentRecord where
  interruptions : ℕ
  insults : ℕ
  throws : ℕ

/-- Theorem: Given Mrs. Carlton's penalty system and Jerry's misbehavior record, 
    we can determine the number of points for interrupting. --/
theorem find_interrupt_points 
  (system : PenaltySystem)
  (jerry : StudentRecord)
  (h1 : system.insult_points = 10)
  (h2 : system.throw_points = 25)
  (h3 : system.office_threshold = 100)
  (h4 : jerry.interruptions = 2)
  (h5 : jerry.insults = 4)
  (h6 : jerry.throws = 2) :
  system.interrupt_points = 5 := by
  sorry

#check find_interrupt_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_interrupt_points_l360_36054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_leq_neg_one_f_monotonicity_when_a_gt_neg_one_inequality_when_a_lt_one_l360_36095

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 + a) * x^2 - Real.log x - a + 1

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * (1 + a) * x^2 - 1) / x

-- Theorem for the monotonicity of f when a ≤ -1
theorem f_increasing_when_a_leq_neg_one (a : ℝ) (x : ℝ) (ha : a ≤ -1) (hx : x > 0) :
  f_deriv a x > 0 := by sorry

-- Theorem for the monotonicity of f when a > -1
theorem f_monotonicity_when_a_gt_neg_one (a : ℝ) (x : ℝ) (ha : a > -1) (hx : x > 0) :
  (x < Real.sqrt (2 * (1 + a)) / (2 * (1 + a)) → f_deriv a x < 0) ∧
  (x > Real.sqrt (2 * (1 + a)) / (2 * (1 + a)) → f_deriv a x > 0) := by sorry

-- Theorem for the inequality when a < 1
theorem inequality_when_a_lt_one (a : ℝ) (x : ℝ) (ha : a < 1) (hx : x > 0) :
  x * f a x > Real.log x + (1 + a) * x^3 - x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_leq_neg_one_f_monotonicity_when_a_gt_neg_one_inequality_when_a_lt_one_l360_36095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nominal_rate_calculation_l360_36032

-- Define the effective annual rate
def ear : ℝ := 0.0609

-- Define the number of compounding periods per year
def n : ℕ := 2

-- Define the nominal rate of interest per annum
noncomputable def nominal_rate : ℝ := 2 * (Real.sqrt (1 + ear) - 1)

-- Theorem to prove
theorem nominal_rate_calculation :
  ∀ ε > 0, |nominal_rate - 0.0598| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nominal_rate_calculation_l360_36032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l360_36002

def sequence_a : ℕ → ℚ
  | 0 => -2
  | n + 1 => 2 + (2 * sequence_a n) / (1 - sequence_a n)

theorem fourth_term_value : sequence_a 3 = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l360_36002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l360_36065

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Ioo (-2) 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x / Real.sqrt x

-- Theorem stating the domain of g
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.univ} = Set.Ioo 0 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l360_36065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_7_8_l360_36001

/-- The area of a triangle given its side lengths. -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 5, 7, and 8 is 10√3. -/
theorem triangle_area_5_7_8 : triangleArea 5 7 8 = 10 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_7_8_l360_36001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_count_l360_36010

theorem equality_condition_count : 
  ∃! (S : Set (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), ∀ (a b c : ℝ), (a, b, c) ∈ S → 
      |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ∧ 
    Finite S ∧ 
    Nat.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_count_l360_36010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l360_36040

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}
def C (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the condition for C to be a sufficient but not necessary condition for A ∩ B
def C_sufficient_not_necessary (m : ℝ) : Prop :=
  C m ⊆ A_intersect_B ∧ C m ≠ A_intersect_B

-- Theorem statement
theorem range_of_m :
  {m : ℝ | C_sufficient_not_necessary m} = Set.Iic 2 := by sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l360_36040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_when_k_zero_monotonicity_intervals_when_k_positive_l360_36089

-- Define the function f(x) with parameter k
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 - 3 * x^2 + 3

-- Define the line y = x - 1
def line (x : ℝ) : ℝ := x - 1

-- Theorem for the enclosed area when k = 0
theorem enclosed_area_when_k_zero :
  let f₀ := f 0
  let a := -4/3
  let b := 1
  (∫ x in a..b, f₀ x - line x) = 343/54 := by sorry

-- Theorem for monotonicity intervals when k > 0
theorem monotonicity_intervals_when_k_positive (k : ℝ) (h : k > 0) :
  let f' := λ x => 3 * k * x^2 - 6 * x
  (∀ x, x < 0 → f' x > 0) ∧
  (∀ x, x > 2/k → f' x > 0) ∧
  (∀ x, 0 < x ∧ x < 2/k → f' x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_when_k_zero_monotonicity_intervals_when_k_positive_l360_36089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l360_36028

/-- The focus of a parabola given by its equation -/
noncomputable def focus_of_parabola (x y : ℝ) : ℝ × ℝ := sorry

/-- Given a parabola defined by x = -1/8 * y^2, its focus is at (1/2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  x = -1/8 * y^2 → focus_of_parabola x y = (1/2, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l360_36028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l360_36041

theorem relationship_abc (a b c : ℝ) 
  (h1 : (3 : ℝ)^a = 4) 
  (h2 : (4 : ℝ)^b = 5) 
  (h3 : a^c = b) : 
  a > b ∧ b > c := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l360_36041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_points_l360_36084

def point_a : ℝ × ℝ × ℝ := (-3, 4, -2)
def point_b : ℝ × ℝ × ℝ := (1, 4, 0)
def point_c : ℝ × ℝ × ℝ := (3, 2, -1)

def plane_equation (x y z : ℝ) : ℝ := x + 2*y - 2*z - 9

theorem plane_through_points :
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ 
    (∃ (s t : ℝ), (x, y, z) = 
      (point_a.1 + s*(point_b.1 - point_a.1) + t*(point_c.1 - point_a.1),
       point_a.2.1 + s*(point_b.2.1 - point_a.2.1) + t*(point_c.2.1 - point_a.2.1),
       point_a.2.2 + s*(point_b.2.2 - point_a.2.2) + t*(point_c.2.2 - point_a.2.2)))) ∧
  (plane_equation point_a.1 point_a.2.1 point_a.2.2 = 0) ∧
  (plane_equation point_b.1 point_b.2.1 point_b.2.2 = 0) ∧
  (plane_equation point_c.1 point_c.2.1 point_c.2.2 = 0) ∧
  (∃ (A B C D : ℤ), ∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
  (∃ (A : ℤ), A > 0 ∧ ∀ (x y z : ℝ), plane_equation x y z = A*x + 2*A*y - 2*A*z - 9*A) ∧
  (Nat.gcd (Nat.gcd 1 2) (Nat.gcd 2 9) = 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_points_l360_36084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l360_36086

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point (2, 0) to the line y = x + 2 is 2√2 -/
theorem distance_to_line : distance_point_to_line 2 0 1 (-1) 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l360_36086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_T_formula_l360_36073

-- Define the sequence a_n
def a : ℕ → ℕ := sorry

-- Define the sum S_n of the first n terms of a_n
def S : ℕ → ℕ := sorry

-- Define the sequence b_n
def b (n : ℕ) : ℤ := (-1)^n + 2^(a n)

-- Define the sum T_n of the first n terms of b_n
noncomputable def T : ℕ → ℚ := sorry

-- Axioms
axiom a_1 : a 1 = 3
axiom S_relation (n : ℕ) : S n + 1 = a n + n^2

-- Theorem 1: General formula for a_n
theorem a_formula (n : ℕ) : a n = 2 * n + 1 := by sorry

-- Theorem 2: Formula for T_n
theorem T_formula (n : ℕ) : T n = ((-1)^n - 1) / 2 + 8 * (4^n - 1) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_T_formula_l360_36073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l360_36015

def sequenceA (a : ℕ) : ℕ → ℕ
  | 0 => 0  -- arbitrary value for n₀
  | 1 => 0  -- arbitrary value for n₁
  | (i + 2) => 2 * sequenceA a (i + 1) + a

theorem sequence_fifth_term (a : ℕ) :
  sequenceA a 2 = 5 → sequenceA a 8 = 257 → sequenceA a 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l360_36015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_2015_location_l360_36024

/-- Represents a point on a line segment --/
structure Point where
  position : ℝ

/-- Represents a moving object --/
structure MovingObject where
  speed : ℝ
  startPoint : Point

/-- Represents the system of two objects moving back and forth --/
structure TravelSystem where
  object1 : MovingObject
  object2 : MovingObject
  segmentLength : ℝ
  timeDifference : ℝ

/-- Represents a meeting point --/
def MeetingPoint := Point

/-- Function to determine the meeting point given a meeting number --/
noncomputable def meetingLocation (system : TravelSystem) (meetingNumber : ℕ) : MeetingPoint :=
  sorry

/-- Theorem stating that the 2015th meeting occurs at the same point as the first meeting --/
theorem meeting_2015_location (system : TravelSystem) :
  meetingLocation system 2015 = meetingLocation system 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_2015_location_l360_36024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dimensions_l360_36087

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The perimeter of a triangle formed by two points on the ellipse and one focus -/
def triangle_perimeter (e : Ellipse) : ℝ := 4 * e.a

theorem ellipse_dimensions (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 3 / 3)
  (h_per : triangle_perimeter e = 4 * Real.sqrt 3) :
  e.a^2 = 3 ∧ e.b^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dimensions_l360_36087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_sqrt_l360_36096

theorem cube_sum_sqrt : Real.sqrt (4^3 + 4^3 + 4^3) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_sqrt_l360_36096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_iff_a_in_interval_l360_36059

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

theorem range_of_f_iff_a_in_interval :
  ∀ a : ℝ, (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ∈ Set.Ici (-1) ∩ Set.Iio (1/2) :=
by
  sorry

#check range_of_f_iff_a_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_iff_a_in_interval_l360_36059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l360_36018

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 3 * x + 4) / (x^2 - x + 1)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (3 < y ∧ y ≤ 13/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l360_36018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l360_36007

-- Define the temperatures for each day
def monday_temp : ℝ := 42
variable (tuesday_temp : ℝ)
variable (wednesday_temp : ℝ)
variable (thursday_temp : ℝ)
variable (friday_temp : ℝ)

-- Define the average temperatures
def avg_mon_to_thu : ℝ := 48
def avg_tue_to_fri : ℝ := 40

-- Theorem statement
theorem friday_temperature :
  (monday_temp + tuesday_temp + wednesday_temp + thursday_temp) / 4 = avg_mon_to_thu ∧
  (tuesday_temp + wednesday_temp + thursday_temp + friday_temp) / 4 = avg_tue_to_fri →
  friday_temp = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l360_36007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_footballs_is_12_l360_36011

/-- The number of footballs in a school given certain conditions -/
def number_of_footballs : ℕ :=
  let total_balls : ℕ := 20
  let total_students : ℕ := 96
  let students_per_football : ℕ := 6
  let students_per_basketball : ℕ := 3
  let footballs : ℕ := 12
  let basketballs : ℕ := total_balls - footballs

  have h1 : footballs + basketballs = total_balls := by sorry
  have h2 : students_per_football * footballs + students_per_basketball * basketballs = total_students := by sorry

  footballs

theorem number_of_footballs_is_12 : number_of_footballs = 12 := by
  rfl

#eval number_of_footballs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_footballs_is_12_l360_36011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_theorem_l360_36013

/-- Rounds a real number to the nearest thousandth -/
noncomputable def round_to_thousandth (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

theorem sum_and_round_theorem :
  round_to_thousandth (75.126 + 8.0034) = 83.129 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_theorem_l360_36013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_exp_log_l360_36080

-- Define the functions
noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (x : ℝ) := Real.log x

-- Define the set of points on each graph
def P : Set (ℝ × ℝ) := {p | p.2 = f p.1}
def Q : Set (ℝ × ℝ) := {q | q.2 = g q.1}

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_exp_log :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ ∀ (p q : ℝ × ℝ), p ∈ P → q ∈ Q → distance p q ≥ d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_exp_log_l360_36080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_alignment_l360_36064

noncomputable def a : ℝ × ℝ × ℝ := (2, -3, 1)
noncomputable def b : ℝ × ℝ × ℝ := (-1, 1, -1)
noncomputable def v : ℝ × ℝ × ℝ := (-27/20, 31/20, -27/20)

theorem vector_alignment :
  (‖v‖ = 1) ∧ (∃ k : ℝ, k > 0 ∧ b = k • ((1 / 2 : ℝ) • (a + 7 • v))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_alignment_l360_36064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l360_36042

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  -1 ≤ a ∧ a < 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l360_36042
