import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3500_350056

/-- Given two lines L1 and L2, returns true if they are parallel but not coincident -/
def are_parallel_not_coincident (L1 L2 : ℝ → ℝ → Prop) : Prop :=
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, L1 x y ↔ L2 (k * x) (k * y)) ∧
  ¬(∀ x y, L1 x y ↔ L2 x y)

/-- The first line: ax + 2y + 6 = 0 -/
def L1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

/-- The second line: x + (a - 1)y + (a^2 - 1) = 0 -/
def L2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0

theorem parallel_lines_a_value :
  ∃ a : ℝ, are_parallel_not_coincident (L1 a) (L2 a) ∧ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3500_350056


namespace NUMINAMATH_CALUDE_sequence_inequality_l3500_350015

theorem sequence_inequality (a : ℕ → ℕ) (h1 : a 1 < a 2) 
  (h2 : ∀ k ≥ 3, a k = 4 * a (k - 1) - 3 * a (k - 2)) : 
  a 45 > 3^43 := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3500_350015


namespace NUMINAMATH_CALUDE_marble_count_l3500_350098

theorem marble_count (allison_marbles angela_marbles albert_marbles : ℕ) : 
  allison_marbles = 28 →
  angela_marbles = allison_marbles + 8 →
  albert_marbles = 3 * angela_marbles →
  albert_marbles + allison_marbles = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_count_l3500_350098


namespace NUMINAMATH_CALUDE_bill_donut_order_combinations_l3500_350038

/-- The number of combinations for selecting donuts satisfying the given conditions -/
def donut_combinations (total_donuts : ℕ) (donut_types : ℕ) (types_to_select : ℕ) : ℕ :=
  (donut_types.choose types_to_select) * 
  ((total_donuts - types_to_select + types_to_select - 1).choose (types_to_select - 1))

/-- Theorem stating that the number of combinations for Bill's donut order is 100 -/
theorem bill_donut_order_combinations : 
  donut_combinations 7 5 4 = 100 := by
sorry

end NUMINAMATH_CALUDE_bill_donut_order_combinations_l3500_350038


namespace NUMINAMATH_CALUDE_bottle_production_l3500_350024

/-- Given that 4 identical machines produce 16 bottles per minute at a constant rate,
    prove that 8 such machines will produce 96 bottles in 3 minutes. -/
theorem bottle_production (machines : ℕ) (bottles_per_minute : ℕ) (time : ℕ) : 
  machines = 4 → bottles_per_minute = 16 → time = 3 →
  (2 * machines) * (bottles_per_minute / machines) * time = 96 := by
  sorry

#check bottle_production

end NUMINAMATH_CALUDE_bottle_production_l3500_350024


namespace NUMINAMATH_CALUDE_perfect_square_count_l3500_350022

theorem perfect_square_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n > 0 ∧ n ≤ 2000 ∧ ∃ k : Nat, 21 * n = k * k) ∧ 
  S.card = 9 ∧
  (∀ n : Nat, n > 0 ∧ n ≤ 2000 ∧ (∃ k : Nat, 21 * n = k * k) → n ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_count_l3500_350022


namespace NUMINAMATH_CALUDE_arrangements_count_l3500_350039

/-- The number of different arrangements for 6 students where two specific students cannot stand together -/
def number_of_arrangements : ℕ := 480

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students that can be arranged freely -/
def free_students : ℕ := 4

/-- The number of gaps after arranging the free students -/
def number_of_gaps : ℕ := 5

/-- The number of students that cannot stand together -/
def restricted_students : ℕ := 2

theorem arrangements_count :
  number_of_arrangements = 
    (Nat.factorial free_students) * (number_of_gaps * (number_of_gaps - 1)) :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l3500_350039


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3500_350003

theorem lcm_hcf_problem (a b : ℕ+) (h1 : b = 15) (h2 : Nat.lcm a b = 60) (h3 : Nat.gcd a b = 3) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3500_350003


namespace NUMINAMATH_CALUDE_two_train_problem_l3500_350096

/-- Prove that given the conditions of the two-train problem, the speed of the second train is 40 km/hr -/
theorem two_train_problem (v : ℝ) : 
  (∀ t : ℝ, 50 * t = v * t + 100) →  -- First train travels 100 km more
  (∀ t : ℝ, 50 * t + v * t = 900) →  -- Total distance is 900 km
  v = 40 := by
  sorry

end NUMINAMATH_CALUDE_two_train_problem_l3500_350096


namespace NUMINAMATH_CALUDE_cosine_period_problem_l3500_350093

theorem cosine_period_problem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * x + c) + d) →
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * (x + 2 * π) + c) + d) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_period_problem_l3500_350093


namespace NUMINAMATH_CALUDE_two_white_prob_correct_at_least_one_white_prob_correct_l3500_350048

/-- Represents the outcome of drawing a ball -/
inductive Ball
| White
| Black

/-- Represents the state of the bag of balls -/
structure BagState where
  total : Nat
  white : Nat
  black : Nat

/-- The initial state of the bag -/
def initialBag : BagState :=
  { total := 5, white := 3, black := 2 }

/-- Calculates the probability of drawing two white balls in succession -/
def probTwoWhite (bag : BagState) : Rat :=
  (bag.white / bag.total) * ((bag.white - 1) / (bag.total - 1))

/-- Calculates the probability of drawing at least one white ball in two draws -/
def probAtLeastOneWhite (bag : BagState) : Rat :=
  1 - (bag.black / bag.total) * ((bag.black - 1) / (bag.total - 1))

theorem two_white_prob_correct :
  probTwoWhite initialBag = 3 / 10 := by sorry

theorem at_least_one_white_prob_correct :
  probAtLeastOneWhite initialBag = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_two_white_prob_correct_at_least_one_white_prob_correct_l3500_350048


namespace NUMINAMATH_CALUDE_f_min_at_three_l3500_350083

/-- The quadratic function to be minimized -/
def f (c : ℝ) : ℝ := 3 * c^2 - 18 * c + 20

/-- Theorem stating that f is minimized at c = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f 3 ≤ f x := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l3500_350083


namespace NUMINAMATH_CALUDE_tyler_meal_combinations_correct_l3500_350013

/-- The number of different meal combinations Tyler can choose at a buffet. -/
def tyler_meal_combinations : ℕ := 150

/-- The number of meat options available. -/
def meat_options : ℕ := 3

/-- The number of vegetable options available. -/
def vegetable_options : ℕ := 5

/-- The number of vegetables Tyler must choose. -/
def vegetables_to_choose : ℕ := 3

/-- The number of dessert options available. -/
def dessert_options : ℕ := 5

/-- Theorem stating that the number of meal combinations Tyler can choose is correct. -/
theorem tyler_meal_combinations_correct :
  tyler_meal_combinations = meat_options * (Nat.choose vegetable_options vegetables_to_choose) * dessert_options :=
by sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_correct_l3500_350013


namespace NUMINAMATH_CALUDE_expression_simplification_l3500_350065

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  ((x * y + 2) * (x * y - 2) + (x * y - 2)^2) / (x * y) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3500_350065


namespace NUMINAMATH_CALUDE_vector_collinearity_l3500_350018

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c, then k = -26/15 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, 3)) (h3 : c = (4, -7)) :
  (∃ (k : ℝ), ∃ (t : ℝ), t • c = k • a + b) → 
  (∃ (k : ℝ), k • a + b = (-26/15) • a + b) :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3500_350018


namespace NUMINAMATH_CALUDE_gcd_83_power_plus_one_l3500_350017

theorem gcd_83_power_plus_one (h : Prime 83) : 
  Nat.gcd (83^9 + 1) (83^9 + 83^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_83_power_plus_one_l3500_350017


namespace NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l3500_350009

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (planeParallel : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (lineInPlane : Line → Plane → Prop)

theorem line_parallel_plane_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : lineInPlane m α) :
  (∀ (α β : Plane) (m : Line), planeParallel α β → lineInPlane m α → lineParallelPlane m β) ∧
  (∃ (α β : Plane) (m : Line), lineParallelPlane m β ∧ lineInPlane m α ∧ ¬planeParallel α β) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l3500_350009


namespace NUMINAMATH_CALUDE_employee_hours_proof_l3500_350068

/-- The number of hours worked per week by both employees -/
def hours : ℕ := 40

/-- The hourly rate of the first employee -/
def rate1 : ℕ := 20

/-- The hourly rate of the second employee -/
def rate2 : ℕ := 22

/-- The hourly subsidy for hiring a disabled worker -/
def subsidy : ℕ := 6

/-- The weekly savings by hiring the cheaper employee -/
def savings : ℕ := 160

theorem employee_hours_proof :
  (rate1 * hours) - ((rate2 * hours) - (subsidy * hours)) = savings :=
by sorry

end NUMINAMATH_CALUDE_employee_hours_proof_l3500_350068


namespace NUMINAMATH_CALUDE_folded_rectangle_length_l3500_350099

/-- Given a rectangular strip of paper with dimensions 4 × 13, folded to form two rectangles
    with areas P and Q such that P = 2Q, prove that the length of one of the resulting rectangles is 6. -/
theorem folded_rectangle_length (x y : ℝ) (P Q : ℝ) : 
  x + y = 9 →  -- Sum of lengths of the two rectangles
  x + 4 + y = 13 →  -- Total length of the original rectangle
  P = 4 * x →  -- Area of rectangle P
  Q = 4 * y →  -- Area of rectangle Q
  P = 2 * Q →  -- Relationship between areas P and Q
  x = 6 := by sorry

end NUMINAMATH_CALUDE_folded_rectangle_length_l3500_350099


namespace NUMINAMATH_CALUDE_DE_DB_ratio_l3500_350020

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (right_angle_ABC : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)
variable (right_angle_ABD : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
variable (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 2)
variable (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3)
variable (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 7)
variable (C_D_opposite : (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1) * (B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1) < 0)
variable (D_parallel_AC : (D.2 - A.2) * (C.1 - A.1) = (D.1 - A.1) * (C.2 - A.2))
variable (E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2)) ∧ t > 1)

-- Theorem statement
theorem DE_DB_ratio :
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 5 / 9 :=
sorry

end NUMINAMATH_CALUDE_DE_DB_ratio_l3500_350020


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3500_350053

theorem arithmetic_sequence_common_difference :
  let a : ℕ → ℤ := λ n => 2 - 3 * n
  ∀ n : ℕ, a (n + 1) - a n = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3500_350053


namespace NUMINAMATH_CALUDE_triangle_angle_value_l3500_350058

/-- Given a triangle ABC with angle C = 60°, angle A = x, and angle B = 2x,
    where x is also an alternate interior angle formed by a line intersecting two parallel lines,
    prove that x = 40°. -/
theorem triangle_angle_value (A B C : ℝ) (x : ℝ) : 
  A = x → B = 2*x → C = 60 → A + B + C = 180 → x = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l3500_350058


namespace NUMINAMATH_CALUDE_triangle_area_is_two_symmetric_point_correct_line_equal_intercepts_l3500_350082

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  sorry

-- Function to find the symmetric point
def symmetricPoint (p : Point) (l : Line) : Point :=
  sorry

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line) : Prop :=
  sorry

-- Theorem 1
theorem triangle_area_is_two :
  let l : Line := { a := 1, b := -1, c := -2 }
  let t : Triangle := { p1 := { x := 0, y := 0 }, p2 := { x := 2, y := 0 }, p3 := { x := 0, y := -2 } }
  triangleArea t = 2 :=
sorry

-- Theorem 2
theorem symmetric_point_correct :
  let l : Line := { a := -1, b := 1, c := 1 }
  let p : Point := { x := 0, y := 2 }
  symmetricPoint p l = { x := 1, y := 1 } :=
sorry

-- Theorem 3
theorem line_equal_intercepts :
  let l : Line := { a := 1, b := 1, c := -2 }
  pointOnLine { x := 1, y := 1 } l ∧ hasEqualIntercepts l :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_symmetric_point_correct_line_equal_intercepts_l3500_350082


namespace NUMINAMATH_CALUDE_seven_nanometers_in_meters_l3500_350043

-- Define the conversion factor for nanometers to meters
def nanometer_to_meter : ℝ := 1e-9

-- Theorem statement
theorem seven_nanometers_in_meters :
  7 * nanometer_to_meter = 7e-9 := by
  sorry

end NUMINAMATH_CALUDE_seven_nanometers_in_meters_l3500_350043


namespace NUMINAMATH_CALUDE_cage_cost_proof_l3500_350007

def cat_toy_cost : ℝ := 10.22
def total_cost : ℝ := 21.95

theorem cage_cost_proof : total_cost - cat_toy_cost = 11.73 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_proof_l3500_350007


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l3500_350045

def sum_odd (n : ℕ) : ℕ := n^2

def sum_even (n : ℕ) : ℕ := n * (n + 1)

def odd_terms (max : ℕ) : ℕ := (max - 1) / 2 + 1

def even_terms (max : ℕ) : ℕ := (max - 2) / 2 + 1

theorem odd_even_sum_difference :
  sum_odd (odd_terms 2023) - sum_even (even_terms 2020) = 3034 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l3500_350045


namespace NUMINAMATH_CALUDE_f_properties_l3500_350080

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - (m / 2) * x^2 + x

theorem f_properties (m : ℝ) :
  (m > 0 ∧ (∀ x > 0, f m x ≤ m * x - 1/2) → m ≥ 1) ∧
  (m = -1 → ∀ x₁ > 0, ∀ x₂ > 0, f m x₁ + f m x₂ = 0 → x₁ + x₂ ≥ Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3500_350080


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3500_350005

/-- Given two natural numbers m and n, returns true if their product has a units digit of 1 -/
def product_has_units_digit_one (m n : ℕ) : Prop :=
  (m * n) % 10 = 1

/-- Given a natural number m, returns true if it has a units digit of 9 -/
def has_units_digit_nine (m : ℕ) : Prop :=
  m % 10 = 9

theorem units_digit_of_n (m n : ℕ) 
  (h1 : m * n = 11^4)
  (h2 : has_units_digit_nine m) :
  n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3500_350005


namespace NUMINAMATH_CALUDE_subset_of_A_l3500_350025

def A : Set ℝ := {x | x ≤ 10}

theorem subset_of_A : {2} ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_subset_of_A_l3500_350025


namespace NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l3500_350086

theorem container_capacity : ℝ → Prop :=
  fun C =>
    (C > 0) ∧                   -- Capacity is positive
    (1/2 * C + 20 = 3/4 * C) →  -- Adding 20 liters to half-full makes it 3/4 full
    C = 80                      -- The capacity is 80 liters

-- Proof
theorem container_capacity_proof : ∃ C, container_capacity C :=
  sorry

end NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l3500_350086


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l3500_350046

/-- The line equation -/
def line (a x y : ℝ) : Prop := (a + 1) * x + (3 * a - 1) * y - (6 * a + 2) = 0

/-- The ellipse equation -/
def ellipse (x y m : ℝ) : Prop := x^2 / 16 + y^2 / m = 1

/-- The theorem stating the conditions for the line and ellipse to always have a common point -/
theorem line_ellipse_intersection (a m : ℝ) :
  (∀ x y : ℝ, line a x y → ellipse x y m → False) ↔ 
  (m ∈ Set.Icc (16/7) 16 ∪ Set.Ioi 16) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l3500_350046


namespace NUMINAMATH_CALUDE_circle_equation_tangent_lines_l3500_350042

-- Define the circle C
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the point M
def M : ℝ × ℝ := (0, 2)

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Theorem for the equation of circle C
theorem circle_equation : 
  ∃ (r : ℝ), M ∈ Circle r ∧ Circle r = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} :=
sorry

-- Define a tangent line
def TangentLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 - 3 * k + 2 = 0}

-- Theorem for the equations of tangent lines
theorem tangent_lines :
  ∃ (k₁ k₂ : ℝ), 
    (TangentLine k₁ = {p : ℝ × ℝ | p.2 = 2}) ∧
    (TangentLine k₂ = {p : ℝ × ℝ | 12 * p.1 - 5 * p.2 - 26 = 0}) ∧
    P ∈ TangentLine k₁ ∧ P ∈ TangentLine k₂ ∧
    (∀ (p : ℝ × ℝ), p ∈ TangentLine k₁ ∩ Circle 2 → p = P) ∧
    (∀ (p : ℝ × ℝ), p ∈ TangentLine k₂ ∩ Circle 2 → p = P) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_lines_l3500_350042


namespace NUMINAMATH_CALUDE_quadratic_through_points_l3500_350090

/-- A quadratic function that passes through (-1, 2) and (1, y) must have y = 2 -/
theorem quadratic_through_points (a : ℝ) (y : ℝ) : 
  a ≠ 0 → (2 = a * (-1)^2) → (y = a * 1^2) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_points_l3500_350090


namespace NUMINAMATH_CALUDE_scale_multiplication_l3500_350033

theorem scale_multiplication (a b c : ℝ) (h : a * b = c) :
  (a / 100) * (b / 100) = c / 10000 := by
  sorry

end NUMINAMATH_CALUDE_scale_multiplication_l3500_350033


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3500_350071

/-- The position function of a particle -/
def s (t : ℝ) : ℝ := 3 * t^2 + t

/-- The velocity function of a particle -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_3_seconds : v 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3500_350071


namespace NUMINAMATH_CALUDE_sin_10_cos_20_cos_40_l3500_350052

theorem sin_10_cos_20_cos_40 :
  Real.sin (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (40 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_cos_20_cos_40_l3500_350052


namespace NUMINAMATH_CALUDE_division_problem_l3500_350004

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 22 →
  divisor = 3 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 7 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3500_350004


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_l3500_350001

/-- The total amount Jason spent on clothing -/
def total_spent : ℚ := 19.02

/-- The amount Jason spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount Jason spent on a jacket -/
def jacket_cost : ℚ := 4.74

/-- Theorem stating that the total amount spent equals the sum of shorts and jacket costs -/
theorem total_spent_equals_sum : total_spent = shorts_cost + jacket_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_l3500_350001


namespace NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_simplify_expression_3_l3500_350055

-- Part 1
theorem simplify_expression_1 (m n : ℝ) :
  15 * m * n^2 + 5 * m * n * m^3 * n = 15 * m * n^2 + 5 * m^4 * n^2 := by sorry

-- Part 2
theorem expand_expression_2 (x : ℝ) :
  (3 * x + 1) * (2 * x - 5) = 6 * x^2 - 13 * x - 5 := by sorry

-- Part 3
theorem simplify_expression_3 :
  (-0.25)^2024 * 4^2023 = 0.25 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_simplify_expression_3_l3500_350055


namespace NUMINAMATH_CALUDE_youseff_walk_time_l3500_350076

/-- The number of blocks from Youseff's home to his office -/
def distance : ℕ := 12

/-- The time in seconds it takes Youseff to ride his bike one block -/
def bike_time : ℕ := 20

/-- The additional time in minutes it takes Youseff to walk compared to biking -/
def additional_time : ℕ := 8

/-- The time in seconds it takes Youseff to walk one block -/
def walk_time : ℕ := sorry

theorem youseff_walk_time :
  walk_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_youseff_walk_time_l3500_350076


namespace NUMINAMATH_CALUDE_a_equals_one_range_of_f_final_no_fixed_points_l3500_350030

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + (a^2 - 1) * x + 1

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Theorem 1: If f is an even function, then a = 1 -/
theorem a_equals_one (a : ℝ) (h : is_even_function (f a)) : a = 1 := by
  sorry

/-- The quadratic function f(x) with a = 1 -/
def f_final (x : ℝ) : ℝ := 2 * x^2 + 1

/-- Theorem 2: If x ∈ [-1, 2], then the range of f_final(x) is [1, 9] -/
theorem range_of_f_final : 
  ∀ y ∈ Set.range f_final, y ∈ Set.Icc 1 9 ∧ 
  ∃ x ∈ Set.Icc (-1) 2, f_final x = 1 ∧
  ∃ x ∈ Set.Icc (-1) 2, f_final x = 9 := by
  sorry

/-- Theorem 3: The equation 2x^2 + 1 = x has no real solutions -/
theorem no_fixed_points : ¬ ∃ x : ℝ, f_final x = x := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_range_of_f_final_no_fixed_points_l3500_350030


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3500_350094

/-- Given the equation 2x + y = 6, prove that y can be expressed as 6 - 2x. -/
theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 6) : y = 6 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3500_350094


namespace NUMINAMATH_CALUDE_negation_of_existence_exp_minus_x_minus_one_negation_l3500_350075

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem exp_minus_x_minus_one_negation :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_exp_minus_x_minus_one_negation_l3500_350075


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3500_350026

theorem inequality_system_solution (x : ℝ) : 
  (2 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 20) →
  (1 / (x - 2) > 1 / 5) →
  (x ∈ Set.Ioo 2 3) ∨ (x ∈ Set.Ioo 4 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3500_350026


namespace NUMINAMATH_CALUDE_zipline_configurations_count_l3500_350014

/-- The number of stories in each building -/
def n : ℕ := 5

/-- The total number of steps (right + up) -/
def total_steps : ℕ := n + n

/-- The number of ways to string ziplines between two n-story buildings
    satisfying the given conditions -/
def num_zipline_configurations : ℕ := Nat.choose total_steps n

/-- Theorem stating that the number of zipline configurations
    is equal to 252 -/
theorem zipline_configurations_count :
  num_zipline_configurations = 252 := by sorry

end NUMINAMATH_CALUDE_zipline_configurations_count_l3500_350014


namespace NUMINAMATH_CALUDE_inscribed_polygon_sides_l3500_350066

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the configuration of polygons in the problem -/
structure PolygonConfiguration :=
  (central : RegularPolygon)
  (inscribed : RegularPolygon)
  (num_inscribed : ℕ)

/-- The sum of interior angles at a contact point -/
def contact_angle_sum : ℝ := 360

/-- The condition that the vertices of the central polygon touch the centers of the inscribed polygons -/
def touches_centers (config : PolygonConfiguration) : Prop :=
  sorry

/-- The theorem stating that in the given configuration, the number of sides of the inscribed polygons must be 6 -/
theorem inscribed_polygon_sides
  (config : PolygonConfiguration)
  (h1 : config.central.sides = 12)
  (h2 : config.num_inscribed = 6)
  (h3 : touches_centers config)
  (h4 : contact_angle_sum = 360) :
  config.inscribed.sides = 6 :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_sides_l3500_350066


namespace NUMINAMATH_CALUDE_michaels_initial_money_l3500_350002

def total_cost : ℕ := 61
def additional_needed : ℕ := 11

theorem michaels_initial_money :
  total_cost - additional_needed = 50 := by
  sorry

end NUMINAMATH_CALUDE_michaels_initial_money_l3500_350002


namespace NUMINAMATH_CALUDE_two_rooks_non_attacking_placements_l3500_350081

/-- The size of a standard chessboard --/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard --/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares a rook can attack (excluding its own square) --/
def rookAttackSquares : Nat := 2 * boardSize - 1

/-- The number of ways to place two rooks on a chessboard without attacking each other --/
def twoRooksPlacement : Nat := totalSquares * (totalSquares - rookAttackSquares)

theorem two_rooks_non_attacking_placements :
  twoRooksPlacement = 3136 := by
  sorry

end NUMINAMATH_CALUDE_two_rooks_non_attacking_placements_l3500_350081


namespace NUMINAMATH_CALUDE_watch_cost_price_l3500_350057

/-- Proves that the cost price of a watch is 1500 Rs. given the conditions of the problem -/
theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 10 / 100 →
  gain_percentage = 5 / 100 →
  price_difference = 225 →
  ∃ (cost_price : ℚ), 
    (1 - loss_percentage) * cost_price + price_difference = (1 + gain_percentage) * cost_price ∧
    cost_price = 1500 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l3500_350057


namespace NUMINAMATH_CALUDE_afternoon_rowers_l3500_350063

theorem afternoon_rowers (total : ℕ) (morning : ℕ) (h1 : total = 60) (h2 : morning = 53) :
  total - morning = 7 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowers_l3500_350063


namespace NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l3500_350028

theorem ab_plus_cd_eq_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l3500_350028


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l3500_350041

theorem sum_reciprocal_inequality (u v w : ℝ) (h : u + v + w = 3) :
  1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l3500_350041


namespace NUMINAMATH_CALUDE_candy_distribution_solution_l3500_350091

def candy_distribution (n : ℕ) : Prop :=
  let initial_candy : ℕ := 120
  let first_phase_passes : ℕ := 40
  let first_phase_candy := first_phase_passes
  let second_phase_candy := initial_candy - first_phase_candy
  let total_passes := first_phase_passes + (second_phase_candy / 2)
  (n ∣ total_passes) ∧ (n > 0) ∧ (n ≤ total_passes)

theorem candy_distribution_solution :
  candy_distribution 40 ∧ ∀ m : ℕ, m ≠ 40 → ¬(candy_distribution m) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_solution_l3500_350091


namespace NUMINAMATH_CALUDE_library_visit_equation_l3500_350010

/-- Represents the growth of library visits over three months -/
def library_visit_growth (initial_visits : ℕ) (final_visits : ℕ) (growth_rate : ℝ) : Prop :=
  initial_visits * (1 + growth_rate)^2 = final_visits

/-- Theorem stating that the given equation accurately represents the library visit growth -/
theorem library_visit_equation : 
  ∃ (x : ℝ), library_visit_growth 560 830 x :=
sorry

end NUMINAMATH_CALUDE_library_visit_equation_l3500_350010


namespace NUMINAMATH_CALUDE_red_books_probability_l3500_350000

-- Define the number of red books and total books
def num_red_books : ℕ := 4
def total_books : ℕ := 8

-- Define the number of books to be selected
def books_selected : ℕ := 2

-- Define the probability function
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem red_books_probability :
  probability (combination num_red_books books_selected) (combination total_books books_selected) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_red_books_probability_l3500_350000


namespace NUMINAMATH_CALUDE_pencil_cost_is_0_602_l3500_350023

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a ruler in dollars -/
def ruler_cost : ℝ := sorry

/-- The total cost of six notebooks and four pencils is $7.44 -/
axiom six_notebooks_four_pencils : 6 * notebook_cost + 4 * pencil_cost = 7.44

/-- The total cost of three notebooks and seven pencils is $6.73 -/
axiom three_notebooks_seven_pencils : 3 * notebook_cost + 7 * pencil_cost = 6.73

/-- The total cost of one notebook, two pencils, and a ruler is $3.36 -/
axiom one_notebook_two_pencils_ruler : notebook_cost + 2 * pencil_cost + ruler_cost = 3.36

/-- The cost of each pencil is $0.602 -/
theorem pencil_cost_is_0_602 : pencil_cost = 0.602 := by sorry

end NUMINAMATH_CALUDE_pencil_cost_is_0_602_l3500_350023


namespace NUMINAMATH_CALUDE_work_rate_ratio_l3500_350032

/-- 
Theorem: Given a job that P can complete in 4 days, and P and Q together can complete in 3 days, 
the ratio of Q's work rate to P's work rate is 1/3.
-/
theorem work_rate_ratio (p q : ℝ) : 
  p > 0 ∧ q > 0 →  -- Ensure positive work rates
  (1 / p = 1 / 4) →  -- P completes the job in 4 days
  (1 / (p + q) = 1 / 3) →  -- P and Q together complete the job in 3 days
  q / p = 1 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_work_rate_ratio_l3500_350032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3500_350029

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Last term of an arithmetic sequence -/
def last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  ∃ n : ℕ, n > 0 ∧ last_term 1 2 n = 21 ∧ arithmetic_sum 1 2 n = 121 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3500_350029


namespace NUMINAMATH_CALUDE_prime_sequence_implies_composite_l3500_350051

theorem prime_sequence_implies_composite (p : ℕ) 
  (h1 : Nat.Prime p)
  (h2 : Nat.Prime (3*p + 2))
  (h3 : Nat.Prime (5*p + 4))
  (h4 : Nat.Prime (7*p + 6))
  (h5 : Nat.Prime (9*p + 8))
  (h6 : Nat.Prime (11*p + 10)) :
  ¬(Nat.Prime (6*p + 11)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_implies_composite_l3500_350051


namespace NUMINAMATH_CALUDE_set_operations_and_subset_condition_l3500_350016

def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_subset_condition (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ (C a ∪ B) = {x | x < 4}) ∧
  (A ⊆ C a → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_condition_l3500_350016


namespace NUMINAMATH_CALUDE_division_problem_l3500_350064

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * remainder + 3 →
  dividend = 113 →
  dividend = divisor * quotient + remainder →
  (divisor : ℚ) / quotient = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3500_350064


namespace NUMINAMATH_CALUDE_trig_identity_l3500_350060

theorem trig_identity : 
  1 / Real.sin (70 * π / 180) - Real.sqrt 2 / Real.cos (70 * π / 180) = 
  -2 * Real.sin (25 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3500_350060


namespace NUMINAMATH_CALUDE_correct_propositions_l3500_350054

theorem correct_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l3500_350054


namespace NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l3500_350072

-- Define the concept of corresponding angles
def corresponding_angles (α β : ℝ) : Prop := sorry

-- Theorem stating that the proposition "corresponding angles are equal" is false
theorem corresponding_angles_not_always_equal :
  ¬ ∀ α β : ℝ, corresponding_angles α β → α = β :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l3500_350072


namespace NUMINAMATH_CALUDE_expression_value_l3500_350061

theorem expression_value (a b c d m : ℝ)
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : m = -1)     -- m equals -1
  : 2 * a * b - (c + d) + m^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3500_350061


namespace NUMINAMATH_CALUDE_lewis_earnings_l3500_350087

/-- Calculates the weekly earnings of a person given the number of weeks worked,
    weekly rent, and final savings. -/
def weekly_earnings (weeks : ℕ) (rent : ℚ) (final_savings : ℚ) : ℚ :=
  (final_savings + weeks * rent) / weeks

theorem lewis_earnings :
  let weeks : ℕ := 1181
  let rent : ℚ := 216
  let final_savings : ℚ := 324775
  weekly_earnings weeks rent final_savings = 490.75 := by sorry

end NUMINAMATH_CALUDE_lewis_earnings_l3500_350087


namespace NUMINAMATH_CALUDE_inner_polygon_perimeter_less_than_outer_l3500_350092

-- Define a type for convex polygons
structure ConvexPolygon where
  -- Add necessary fields (this is a simplified representation)
  perimeter : ℝ

-- Define a relation for one polygon being inside another
def IsInside (inner outer : ConvexPolygon) : Prop :=
  -- Add necessary conditions for one polygon being inside another
  sorry

-- Theorem statement
theorem inner_polygon_perimeter_less_than_outer
  (inner outer : ConvexPolygon)
  (h : IsInside inner outer) :
  inner.perimeter < outer.perimeter :=
sorry

end NUMINAMATH_CALUDE_inner_polygon_perimeter_less_than_outer_l3500_350092


namespace NUMINAMATH_CALUDE_tribe_leadership_structure_l3500_350037

theorem tribe_leadership_structure (n : ℕ) (h : n = 12) : 
  n * (n - 1) * (n - 2) * (Nat.choose (n - 3) 3) * (Nat.choose (n - 6) 3) = 2217600 :=
by sorry

end NUMINAMATH_CALUDE_tribe_leadership_structure_l3500_350037


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_9000_l3500_350067

theorem last_four_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 1250]) :
  5^9000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_9000_l3500_350067


namespace NUMINAMATH_CALUDE_carl_payment_percentage_l3500_350062

theorem carl_payment_percentage (property_damage medical_bills insurance_percentage carl_owes : ℚ)
  (h1 : property_damage = 40000)
  (h2 : medical_bills = 70000)
  (h3 : insurance_percentage = 80/100)
  (h4 : carl_owes = 22000) :
  carl_owes / (property_damage + medical_bills) = 20/100 := by
  sorry

end NUMINAMATH_CALUDE_carl_payment_percentage_l3500_350062


namespace NUMINAMATH_CALUDE_exists_acute_triangle_with_large_intersection_area_l3500_350050

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle. -/
def area (t : Triangle) : ℝ := sorry

/-- A point is the median of a triangle if it is the midpoint of a side. -/
def is_median (M : Point) (t : Triangle) : Prop := sorry

/-- A point is on the angle bisector if it is equidistant from the two sides forming the angle. -/
def is_angle_bisector (K : Point) (t : Triangle) : Prop := sorry

/-- A point is on the altitude if it forms a right angle with the base of the triangle. -/
def is_altitude (H : Point) (t : Triangle) : Prop := sorry

/-- A triangle is acute if all its angles are less than 90 degrees. -/
def is_acute (t : Triangle) : Prop := sorry

/-- The area of the triangle formed by the intersection points of the median, angle bisector, and altitude. -/
def area_intersection (t : Triangle) (M K H : Point) : ℝ := sorry

/-- There exists an acute triangle where the area of the triangle formed by the intersection points
    of its median, angle bisector, and altitude is greater than 0.499 times the area of the original triangle. -/
theorem exists_acute_triangle_with_large_intersection_area :
  ∃ (t : Triangle) (M K H : Point),
    is_acute t ∧
    is_median M t ∧
    is_angle_bisector K t ∧
    is_altitude H t ∧
    area_intersection t M K H > 0.499 * area t :=
sorry

end NUMINAMATH_CALUDE_exists_acute_triangle_with_large_intersection_area_l3500_350050


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3500_350006

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 12) :
  (7 * x - 3) / (x^2 - 8 * x - 48) = 11 / (x + 4) + 0 / (x - 12) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3500_350006


namespace NUMINAMATH_CALUDE_inverse_function_condition_l3500_350027

noncomputable def g (a b c d x : ℝ) : ℝ := (2*a*x + b) / (2*c*x - d)

theorem inverse_function_condition (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : ∀ x, x ∈ {x | 2*c*x - d ≠ 0} → g a b c d (g a b c d x) = x) : 
  2*a - d = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_condition_l3500_350027


namespace NUMINAMATH_CALUDE_tangent_line_and_root_condition_l3500_350040

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- State the theorem
theorem tangent_line_and_root_condition (x : ℝ) :
  -- The tangent line at (2, 7)
  (∃ (m b : ℝ), f 2 = 7 ∧ 
    (∀ x, f x = m * x + b) ∧
    m = 12 ∧ b = -17) ∧
  -- Condition for three distinct real roots
  (∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔
    -3 < m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_root_condition_l3500_350040


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3500_350044

theorem perpendicular_vectors_x_value (x y : ℝ) :
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (3, 2 - x)
  (a.1 * b.1 + a.2 * b.2 = 0) → (x = 3 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3500_350044


namespace NUMINAMATH_CALUDE_faulty_odometer_distance_l3500_350077

/-- Represents an odometer that skips certain digits --/
structure SkippingOdometer where
  reading : Nat
  skipped_digits : List Nat

/-- Calculates the actual distance traveled given a skipping odometer --/
def actual_distance (o : SkippingOdometer) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem faulty_odometer_distance :
  let o : SkippingOdometer := { reading := 3509, skipped_digits := [4, 6] }
  actual_distance o = 2964 :=
sorry

end NUMINAMATH_CALUDE_faulty_odometer_distance_l3500_350077


namespace NUMINAMATH_CALUDE_chair_count_difference_l3500_350074

/-- Represents the number of chairs of each color in a classroom. -/
structure ClassroomChairs where
  blue : Nat
  green : Nat
  white : Nat

/-- Theorem about the difference in chair counts in a classroom. -/
theorem chair_count_difference 
  (chairs : ClassroomChairs) 
  (h1 : chairs.blue = 10)
  (h2 : chairs.green = 3 * chairs.blue)
  (h3 : chairs.blue + chairs.green + chairs.white = 67) :
  chairs.blue + chairs.green - chairs.white = 13 := by
  sorry


end NUMINAMATH_CALUDE_chair_count_difference_l3500_350074


namespace NUMINAMATH_CALUDE_brick_height_l3500_350089

/-- The surface area of a rectangular prism given its length, width, and height -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The height of a rectangular prism with length 8 cm, width 6 cm, 
    and surface area 152 cm² is 2 cm -/
theorem brick_height : 
  ∃ (h : ℝ), h > 0 ∧ surface_area 8 6 h = 152 → h = 2 := by
sorry

end NUMINAMATH_CALUDE_brick_height_l3500_350089


namespace NUMINAMATH_CALUDE_remainder_3125_div_98_l3500_350088

theorem remainder_3125_div_98 : 3125 % 98 = 87 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3125_div_98_l3500_350088


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3500_350012

-- Define the necessary structures
structure Line :=
  (id : String)

structure Plane :=
  (id : String)

-- Define the relationships
def parallel (l : Line) (p : Plane) : Prop :=
  sorry

def incident (l : Line) (p : Plane) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

def skew_lines (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel a α) (h2 : incident b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3500_350012


namespace NUMINAMATH_CALUDE_farm_animals_l3500_350019

/-- Given a farm with chickens and pigs, prove the number of chickens. -/
theorem farm_animals (total_legs : ℕ) (num_pigs : ℕ) (num_chickens : ℕ) : 
  total_legs = 48 → num_pigs = 9 → 2 * num_chickens + 4 * num_pigs = total_legs → num_chickens = 6 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3500_350019


namespace NUMINAMATH_CALUDE_inconvenient_transportation_probability_l3500_350069

/-- The probability of selecting exactly 4 villages with inconvenient transportation
    out of 10 randomly selected villages from a group of 15 villages,
    where 7 have inconvenient transportation, is equal to 1/30. -/
theorem inconvenient_transportation_probability :
  let total_villages : ℕ := 15
  let inconvenient_villages : ℕ := 7
  let selected_villages : ℕ := 10
  let target_inconvenient : ℕ := 4
  
  Fintype.card {s : Finset (Fin total_villages) //
    s.card = selected_villages ∧
    (s.filter (λ i => i.val < inconvenient_villages)).card = target_inconvenient} /
  Fintype.card {s : Finset (Fin total_villages) // s.card = selected_villages} = 1 / 30 :=
by sorry

end NUMINAMATH_CALUDE_inconvenient_transportation_probability_l3500_350069


namespace NUMINAMATH_CALUDE_system_solutions_l3500_350079

theorem system_solutions :
  ∀ x y : ℝ,
  (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) ∨ 
   (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3500_350079


namespace NUMINAMATH_CALUDE_add_base6_35_14_l3500_350035

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 --/
def add_base6 (a b : ℕ) : ℕ :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem add_base6_35_14 : add_base6 35 14 = 53 := by sorry

end NUMINAMATH_CALUDE_add_base6_35_14_l3500_350035


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3500_350078

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 24 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3500_350078


namespace NUMINAMATH_CALUDE_no_integer_roots_l3500_350011

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0)
  (h0 : Odd (c))
  (h1 : Odd (a + b + c)) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3500_350011


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3500_350034

theorem quadratic_inequality_condition (x : ℝ) :
  (∀ x, 0 < x ∧ x < 2 → x^2 - x - 6 < 0) ∧
  (∃ x, x^2 - x - 6 < 0 ∧ ¬(0 < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3500_350034


namespace NUMINAMATH_CALUDE_distance_center_to_plane_l3500_350070

/-- Given a sphere and three points on its surface, calculate the distance from the center to the plane of the triangle formed by the points. -/
theorem distance_center_to_plane (S : Real) (AB BC AC : Real) (h1 : S = 20 * Real.pi) (h2 : BC = 2 * Real.sqrt 3) (h3 : AB = 2) (h4 : AC = 2) : 
  ∃ d : Real, d = 1 ∧ d = Real.sqrt (((S / (4 * Real.pi))^(1/2 : Real))^2 - (BC / (2 * Real.sin (Real.arccos ((AC^2 + AB^2 - BC^2) / (2 * AC * AB)))))^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_plane_l3500_350070


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l3500_350084

theorem triangle_arithmetic_sequence (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a = 2 * Real.sin A) ∧ (b = 2 * Real.sin B) ∧ (c = 2 * Real.sin C) →
  -- a*cos(C), b*cos(B), c*cos(A) form an arithmetic sequence
  (a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) →
  -- Conclusions
  (B = π / 3) ∧
  (∀ x, x ∈ Set.Icc (-1/2) (1 + Real.sqrt 3) ↔ 
    ∃ A C, (0 < A) ∧ (A < 2*π/3) ∧ (C = 2*π/3 - A) ∧
    (x = 2 * Real.sin A * Real.sin A + Real.cos (A - C))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l3500_350084


namespace NUMINAMATH_CALUDE_square_two_minus_sqrt_three_l3500_350073

theorem square_two_minus_sqrt_three (a b : ℚ) :
  (2 - Real.sqrt 3)^2 = a + b * Real.sqrt 3 → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_two_minus_sqrt_three_l3500_350073


namespace NUMINAMATH_CALUDE_triangle_sum_bounds_l3500_350036

theorem triangle_sum_bounds (A B C : Real) (hsum : A + B + C = Real.pi) (hpos : 0 < A ∧ 0 < B ∧ 0 < C) :
  let S := Real.sqrt (3 * Real.tan (A/2) * Real.tan (B/2) + 1) +
           Real.sqrt (3 * Real.tan (B/2) * Real.tan (C/2) + 1) +
           Real.sqrt (3 * Real.tan (C/2) * Real.tan (A/2) + 1)
  4 ≤ S ∧ S < 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_bounds_l3500_350036


namespace NUMINAMATH_CALUDE_integer_between_bounds_l3500_350031

theorem integer_between_bounds (x : ℤ) :
  (-4.5 : ℝ) < (x : ℝ) ∧ (x : ℝ) < (-4 : ℝ) / 3 →
  x = -4 ∨ x = -3 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_integer_between_bounds_l3500_350031


namespace NUMINAMATH_CALUDE_f_properties_l3500_350085

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + a

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f_derivative a x + (2 * a - 1) * x

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) (f_derivative a x) x) ∧
  (∀ x > 0, HasDerivAt (g a) ((1 - x) / x) x) ∧
  (∃ x₀ > 0, IsLocalMax (g a) x₀ ∧ g a x₀ = 0) ∧
  (∀ x₀ > 0, ¬ IsLocalMin (g a) x₀) ∧
  (∀ x > 1, f a x < 0) ↔ a ≥ (1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3500_350085


namespace NUMINAMATH_CALUDE_positive_terms_count_l3500_350049

/-- The number of positive terms in an arithmetic sequence with general term a_n = 90 - 2n -/
theorem positive_terms_count : ∃ k : ℕ, k = 44 ∧ 
  ∀ n : ℕ+, (90 : ℝ) - 2 * (n : ℝ) > 0 ↔ n ≤ k := by sorry

end NUMINAMATH_CALUDE_positive_terms_count_l3500_350049


namespace NUMINAMATH_CALUDE_curve_transformation_l3500_350008

/-- Given a curve y = sin(2x) and a scaling transformation x' = 2x, y' = 3y,
    prove that the resulting curve has the equation y' = 3sin(x'). -/
theorem curve_transformation (x x' y y' : ℝ) : 
  y = Real.sin (2 * x) → 
  x' = 2 * x → 
  y' = 3 * y → 
  y' = 3 * Real.sin x' :=
by sorry

end NUMINAMATH_CALUDE_curve_transformation_l3500_350008


namespace NUMINAMATH_CALUDE_john_replacement_cost_l3500_350095

/-- Represents the genre of a movie --/
inductive Genre
  | Action
  | Comedy
  | Drama

/-- Represents the popularity of a movie --/
inductive Popularity
  | Popular
  | ModeratelyPopular
  | Unpopular

/-- Represents a movie with its genre and popularity --/
structure Movie where
  genre : Genre
  popularity : Popularity

/-- The trade-in value for a VHS movie based on its genre --/
def tradeInValue (g : Genre) : ℕ :=
  match g with
  | Genre.Action => 3
  | Genre.Comedy => 2
  | Genre.Drama => 1

/-- The purchase price for a DVD based on its popularity --/
def purchasePrice (p : Popularity) : ℕ :=
  match p with
  | Popularity.Popular => 12
  | Popularity.ModeratelyPopular => 8
  | Popularity.Unpopular => 5

/-- The collection of movies John has --/
def johnMovies : List Movie :=
  (List.replicate 20 ⟨Genre.Action, Popularity.Popular⟩) ++
  (List.replicate 30 ⟨Genre.Comedy, Popularity.ModeratelyPopular⟩) ++
  (List.replicate 10 ⟨Genre.Drama, Popularity.Unpopular⟩) ++
  (List.replicate 15 ⟨Genre.Comedy, Popularity.Popular⟩) ++
  (List.replicate 25 ⟨Genre.Action, Popularity.ModeratelyPopular⟩)

/-- The total cost to replace all movies --/
def replacementCost (movies : List Movie) : ℕ :=
  (movies.map (fun m => purchasePrice m.popularity)).sum -
  (movies.map (fun m => tradeInValue m.genre)).sum

/-- Theorem stating the cost to replace all of John's movies --/
theorem john_replacement_cost :
  replacementCost johnMovies = 675 := by
  sorry

end NUMINAMATH_CALUDE_john_replacement_cost_l3500_350095


namespace NUMINAMATH_CALUDE_number_with_specific_divisor_sum_l3500_350059

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem number_with_specific_divisor_sum :
  ∀ l m : ℕ,
  let n := 2^l * 3^m
  sum_of_divisors n = 403 →
  n = 144 := by
sorry

end NUMINAMATH_CALUDE_number_with_specific_divisor_sum_l3500_350059


namespace NUMINAMATH_CALUDE_gcd_of_squares_l3500_350097

theorem gcd_of_squares : Nat.gcd (101^2 + 203^2 + 307^2) (100^2 + 202^2 + 308^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l3500_350097


namespace NUMINAMATH_CALUDE_palindrome_decomposition_l3500_350047

/-- A word is a list of characters -/
def Word := List Char

/-- A palindrome is a word that reads the same forward and backward -/
def isPalindrome (w : Word) : Prop :=
  w = w.reverse

/-- X is a word of length 2014 consisting of only 'A' and 'B' -/
def X : Word :=
  List.replicate 2014 'A'  -- Example word, actual content doesn't matter for the theorem

/-- Theorem: There exist at least 806 palindromes whose concatenation forms X -/
theorem palindrome_decomposition :
  ∃ (palindromes : List Word),
    palindromes.length ≥ 806 ∧
    (∀ p ∈ palindromes, isPalindrome p) ∧
    palindromes.join = X :=
  sorry


end NUMINAMATH_CALUDE_palindrome_decomposition_l3500_350047


namespace NUMINAMATH_CALUDE_min_distance_line_ellipse_l3500_350021

noncomputable def minDistance : ℝ := (24 - 2 * Real.sqrt 41) / 5

/-- The minimum distance between a point on the line 4x + 3y = 24
    and a point on the ellipse (x²/8) + (y²/4) = 1 is (24 - 2√41) / 5 -/
theorem min_distance_line_ellipse :
  let line := {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 24}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  ∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ),
    p₁ ∈ line ∧ p₂ ∈ ellipse ∧
    ∀ (q₁ : ℝ × ℝ) (q₂ : ℝ × ℝ),
      q₁ ∈ line → q₂ ∈ ellipse →
      Real.sqrt ((q₁.1 - q₂.1)^2 + (q₁.2 - q₂.2)^2) ≥ minDistance :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_ellipse_l3500_350021
