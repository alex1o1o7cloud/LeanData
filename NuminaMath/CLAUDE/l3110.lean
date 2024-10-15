import Mathlib

namespace NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l3110_311090

theorem cubic_sum_of_quadratic_roots : ∀ r s : ℝ,
  r^2 - 5*r + 6 = 0 →
  s^2 - 5*s + 6 = 0 →
  r^3 + s^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l3110_311090


namespace NUMINAMATH_CALUDE_odd_function_period_4_symmetric_exists_a_inequality_f_is_odd_not_unique_a_for_odd_g_l3110_311070

-- Define an odd function with period 4
def OddFunctionPeriod4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x)

-- Define symmetry about (2,0)
def SymmetricAbout2_0 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (4 - x) = f x

-- Statement 1
theorem odd_function_period_4_symmetric :
  ∀ f : ℝ → ℝ, OddFunctionPeriod4 f → SymmetricAbout2_0 f :=
sorry

-- Statement 2
theorem exists_a_inequality :
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ a^(1 + a) ≥ a^(1 + 1/a) :=
sorry

-- Define the logarithmic function
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Statement 3
theorem f_is_odd :
  ∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x :=
sorry

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + Real.sqrt (2 * x^2 + 1))

-- Statement 4
theorem not_unique_a_for_odd_g :
  ¬ ∃! a : ℝ, ∀ x : ℝ, g a (-x) = -g a x :=
sorry

end NUMINAMATH_CALUDE_odd_function_period_4_symmetric_exists_a_inequality_f_is_odd_not_unique_a_for_odd_g_l3110_311070


namespace NUMINAMATH_CALUDE_no_x_axis_intersection_implies_m_bound_l3110_311015

/-- A quadratic function of the form f(x) = x^2 - x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - x + m

/-- The discriminant of the quadratic function f(x) = x^2 - x + m -/
def discriminant (m : ℝ) : ℝ := 1 - 4*m

theorem no_x_axis_intersection_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, f m x ≠ 0) → m > (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_no_x_axis_intersection_implies_m_bound_l3110_311015


namespace NUMINAMATH_CALUDE_student_number_problem_l3110_311020

theorem student_number_problem (x : ℝ) : 4 * x - 142 = 110 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3110_311020


namespace NUMINAMATH_CALUDE_geometric_sum_7_terms_l3110_311053

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_7_terms :
  let a : ℚ := 1/2
  let r : ℚ := -1/2
  let n : ℕ := 7
  geometric_sum a r n = 129/384 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_7_terms_l3110_311053


namespace NUMINAMATH_CALUDE_bus_people_count_l3110_311043

/-- Represents the number of people who got off the bus -/
def people_off : ℕ := 47

/-- Represents the number of people remaining on the bus -/
def people_remaining : ℕ := 43

/-- Represents the total number of people on the bus before -/
def total_people : ℕ := people_off + people_remaining

theorem bus_people_count : total_people = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_people_count_l3110_311043


namespace NUMINAMATH_CALUDE_tanα_eq_2_implies_reciprocal_sin2α_eq_5_4_l3110_311024

theorem tanα_eq_2_implies_reciprocal_sin2α_eq_5_4 (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_tanα_eq_2_implies_reciprocal_sin2α_eq_5_4_l3110_311024


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l3110_311017

theorem smallest_three_digit_divisible_by_4_and_5 :
  ∃ n : ℕ, n = 100 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 4 ∣ m ∧ 5 ∣ m → n ≤ m) ∧
  4 ∣ n ∧ 5 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l3110_311017


namespace NUMINAMATH_CALUDE_sweet_distribution_l3110_311006

theorem sweet_distribution (total_sweets : ℕ) (initial_children : ℕ) : 
  (initial_children * 15 = total_sweets) → 
  ((initial_children - 32) * 21 = total_sweets) →
  initial_children = 112 := by
sorry

end NUMINAMATH_CALUDE_sweet_distribution_l3110_311006


namespace NUMINAMATH_CALUDE_nine_points_chords_l3110_311029

/-- The number of different chords that can be drawn by connecting two of n points on a circle. -/
def number_of_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting two of nine points
    on the circumference of a circle is equal to 36. -/
theorem nine_points_chords :
  number_of_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_points_chords_l3110_311029


namespace NUMINAMATH_CALUDE_min_seating_circular_table_l3110_311083

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  chairs : ℕ
  seated : ℕ

/-- Predicate to check if a seating arrangement is valid. -/
def validSeating (table : CircularTable) : Prop :=
  table.seated ≤ table.chairs ∧
  ∀ k : ℕ, k < table.seated → ∃ j : ℕ, j < table.seated ∧ j ≠ k ∧
    (((k + 1) % table.chairs = j) ∨ ((k + table.chairs - 1) % table.chairs = j))

/-- The theorem to be proved. -/
theorem min_seating_circular_table :
  ∃ (n : ℕ), n = 20 ∧
  validSeating ⟨60, n⟩ ∧
  ∀ m : ℕ, m < n → ¬validSeating ⟨60, m⟩ := by
  sorry

end NUMINAMATH_CALUDE_min_seating_circular_table_l3110_311083


namespace NUMINAMATH_CALUDE_georgia_black_buttons_l3110_311003

theorem georgia_black_buttons
  (yellow_buttons : Nat)
  (green_buttons : Nat)
  (buttons_given : Nat)
  (buttons_left : Nat)
  (h1 : yellow_buttons = 4)
  (h2 : green_buttons = 3)
  (h3 : buttons_given = 4)
  (h4 : buttons_left = 5) :
  ∃ (black_buttons : Nat), black_buttons = 2 ∧
    yellow_buttons + black_buttons + green_buttons = buttons_left + buttons_given :=
by sorry

end NUMINAMATH_CALUDE_georgia_black_buttons_l3110_311003


namespace NUMINAMATH_CALUDE_range_of_f_l3110_311046

def f (x : ℤ) : ℤ := x^2 + 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 1}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3110_311046


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3110_311092

/-- Given a > 0, if in the expansion of (1+a√x)^n, the coefficient of x^2 is 9 times
    the coefficient of x, and the third term is 135x, then a = 3 -/
theorem binomial_expansion_coefficient (a n : ℝ) (ha : a > 0) : 
  (∃ k₁ k₂ : ℝ, k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ 
    k₁ * a^4 = 9 * k₂ * a^2 ∧
    k₂ * a^2 = 135) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3110_311092


namespace NUMINAMATH_CALUDE_andrew_appointments_l3110_311042

/-- Calculates the number of 3-hour appointments given total work hours, permits stamped per hour, and total permits stamped. -/
def appointments (total_hours : ℕ) (permits_per_hour : ℕ) (total_permits : ℕ) : ℕ :=
  (total_hours - (total_permits / permits_per_hour)) / 3

/-- Theorem stating that given the problem conditions, Andrew has 2 appointments. -/
theorem andrew_appointments : appointments 8 50 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_andrew_appointments_l3110_311042


namespace NUMINAMATH_CALUDE_sequence_properties_l3110_311016

/-- Given a sequence {a_n}, where S_n is the sum of the first n terms,
    a_1 = a (a ≠ 4), and a_{n+1} = 2S_n + 4^n for n ∈ ℕ* -/
def Sequence (a : ℝ) (a_n : ℕ+ → ℝ) (S_n : ℕ+ → ℝ) : Prop :=
  a ≠ 4 ∧
  a_n 1 = a ∧
  ∀ n : ℕ+, a_n (n + 1) = 2 * S_n n + 4^(n : ℕ)

/-- Definition of b_n -/
def b_n (S_n : ℕ+ → ℝ) : ℕ+ → ℝ :=
  λ n => S_n n - 4^(n : ℕ)

theorem sequence_properties {a : ℝ} {a_n : ℕ+ → ℝ} {S_n : ℕ+ → ℝ}
    (h : Sequence a a_n S_n) :
    /- 1. {b_n} forms a geometric progression with common ratio 3 -/
    (∀ n : ℕ+, b_n S_n (n + 1) = 3 * b_n S_n n) ∧
    /- 2. General formula for {a_n} -/
    (∀ n : ℕ+, n = 1 → a_n n = a) ∧
    (∀ n : ℕ+, n ≥ 2 → a_n n = 3 * 4^(n - 1 : ℕ) + 2 * (a - 4) * 3^(n - 2 : ℕ)) ∧
    /- 3. Range of a that satisfies a_{n+1} ≥ a_n for n ∈ ℕ* -/
    (∀ n : ℕ+, a_n (n + 1) ≥ a_n n ↔ a ∈ Set.Icc (-4 : ℝ) 4 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3110_311016


namespace NUMINAMATH_CALUDE_expand_product_l3110_311047

theorem expand_product (x : ℝ) : (x + 3) * (x - 2) * (x + 4) = x^3 + 5*x^2 - 2*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3110_311047


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_C_internal_tangency_of_circles_l3110_311007

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - 2*m)^2 = m^2

-- Define the circle E
def circle_E (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ x = 0

-- Theorem for part (I)
theorem tangent_line_to_circle_C :
  ∀ x y : ℝ, circle_C 2 x y → tangent_line x y → (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0) :=
sorry

-- Theorem for part (II)
theorem internal_tangency_of_circles :
  ∃ x y : ℝ, circle_C ((Real.sqrt 29 - 1) / 4) x y ∧ circle_E x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_C_internal_tangency_of_circles_l3110_311007


namespace NUMINAMATH_CALUDE_eighteen_hundred_is_interesting_smallest_interesting_number_l3110_311073

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a^2 ∧ 15 * n = b^3

/-- 1800 is an interesting number. -/
theorem eighteen_hundred_is_interesting : IsInteresting 1800 :=
  sorry

/-- 1800 is the smallest interesting number. -/
theorem smallest_interesting_number :
  IsInteresting 1800 ∧ ∀ m < 1800, ¬IsInteresting m :=
  sorry

end NUMINAMATH_CALUDE_eighteen_hundred_is_interesting_smallest_interesting_number_l3110_311073


namespace NUMINAMATH_CALUDE_twenty_is_least_pieces_l3110_311031

/-- The number of expected guests -/
def expected_guests : Set Nat := {10, 11}

/-- A function to check if a number of pieces can be equally divided among a given number of guests -/
def can_divide_equally (pieces : Nat) (guests : Nat) : Prop :=
  ∃ (share : Nat), pieces = guests * share

/-- The proposition that a given number of pieces is the least number that can be equally divided among either 10 or 11 guests -/
def is_least_pieces (pieces : Nat) : Prop :=
  (∀ g ∈ expected_guests, can_divide_equally pieces g) ∧
  (∀ p < pieces, ∃ g ∈ expected_guests, ¬can_divide_equally p g)

/-- Theorem stating that 20 is the least number of pieces that can be equally divided among either 10 or 11 guests -/
theorem twenty_is_least_pieces : is_least_pieces 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_least_pieces_l3110_311031


namespace NUMINAMATH_CALUDE_cube_sum_ge_sqrt_product_square_sum_l3110_311074

theorem cube_sum_ge_sqrt_product_square_sum {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_sqrt_product_square_sum_l3110_311074


namespace NUMINAMATH_CALUDE_successor_arrangements_l3110_311095

/-- The number of distinct arrangements of letters in a word -/
def word_arrangements (total_letters : ℕ) (repeated_letters : List (Char × ℕ)) : ℕ :=
  Nat.factorial total_letters / (repeated_letters.map (λ (_, count) => Nat.factorial count)).prod

/-- Theorem: The number of distinct arrangements of SUCCESSOR is 30,240 -/
theorem successor_arrangements :
  word_arrangements 9 [('S', 3), ('C', 2)] = 30240 := by
  sorry

end NUMINAMATH_CALUDE_successor_arrangements_l3110_311095


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3110_311045

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_fourth : a 4 = 16)
  (h_tenth : a 10 = 2) :
  a 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3110_311045


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3110_311037

theorem algebraic_expression_equality (x : ℝ) (h : x^2 - 4*x + 1 = 3) :
  3*x^2 - 12*x - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3110_311037


namespace NUMINAMATH_CALUDE_symmetric_point_l3110_311088

/-- The point symmetric to P(2,-3) with respect to the origin is (-2,3). -/
theorem symmetric_point : 
  let P : ℝ × ℝ := (2, -3)
  let symmetric_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_wrt_origin P = (-2, 3) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_l3110_311088


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l3110_311023

/-- Represents the daily wage in rupees --/
def daily_wage : ℚ := 25

/-- Represents the daily fine in rupees --/
def daily_fine : ℚ := 7.5

/-- Represents the total amount received in rupees --/
def total_amount : ℚ := 685

/-- Represents the number of days absent --/
def days_absent : ℕ := 2

/-- Proves that the contractor was engaged for 28 days --/
theorem contractor_engagement_days : 
  ∃ (days_worked : ℕ), 
    (daily_wage * days_worked - daily_fine * days_absent = total_amount) ∧ 
    (days_worked + days_absent = 28) := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l3110_311023


namespace NUMINAMATH_CALUDE_complement_of_B_l3110_311082

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the universal set U
def U (x : ℝ) : Set ℝ := A x ∪ B x

-- State the theorem
theorem complement_of_B (x : ℝ) :
  (B x ∪ (U x \ B x) = A x) →
  ((x = 0 ∧ U x \ B x = {3}) ∨
   (x = Real.sqrt 3 ∧ U x \ B x = {Real.sqrt 3}) ∨
   (x = -Real.sqrt 3 ∧ U x \ B x = {-Real.sqrt 3})) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_B_l3110_311082


namespace NUMINAMATH_CALUDE_complement_intersection_when_a_is_3_range_of_a_when_union_equals_B_range_of_a_when_intersection_is_empty_l3110_311022

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Theorem 1: When a=3, ℂᴿ(A∩B) = {x | x < 2 or x > 4}
theorem complement_intersection_when_a_is_3 :
  (Set.univ \ (A 3 ∩ B)) = {x | x < 2 ∨ x > 4} := by sorry

-- Theorem 2: When A∪B=B, the range of a is (-∞,-2)∪[-1,3/2]
theorem range_of_a_when_union_equals_B :
  (∀ a, A a ∪ B = B) ↔ (∀ a, a < -2 ∨ (-1 ≤ a ∧ a ≤ 3/2)) := by sorry

-- Theorem 3: When A∩B=∅, the range of a is (-∞,-3/2)∪(5,+∞)
theorem range_of_a_when_intersection_is_empty :
  (∀ a, A a ∩ B = ∅) ↔ (∀ a, a < -3/2 ∨ a > 5) := by sorry

end NUMINAMATH_CALUDE_complement_intersection_when_a_is_3_range_of_a_when_union_equals_B_range_of_a_when_intersection_is_empty_l3110_311022


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l3110_311018

theorem fixed_point_of_exponential_translation (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l3110_311018


namespace NUMINAMATH_CALUDE_square_equation_proof_l3110_311025

theorem square_equation_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ (k : ℚ), (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = k^2 ∧ 
              a^2 + b^2 - c^2 = k^2 ∧
              k = (4*a - 3*b : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_proof_l3110_311025


namespace NUMINAMATH_CALUDE_farm_ratio_l3110_311004

theorem farm_ratio (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 17 / 7)
  (h2 : H - 15 = C + 15 + 50) :
  H / C = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_l3110_311004


namespace NUMINAMATH_CALUDE_specific_system_is_linear_l3110_311067

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ a * x + b * y = c

/-- A system of two equations -/
structure EquationSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system of equations we want to prove is linear -/
def specificSystem : EquationSystem where
  eq1 := {
    a := 1
    b := 1
    c := 1
    eq := λ x y => x + y = 1
    eq_def := by sorry
  }
  eq2 := {
    a := 1
    b := -1
    c := 2
    eq := λ x y => x - y = 2
    eq_def := by sorry
  }

/-- Definition of a system of two linear equations -/
def isSystemOfTwoLinearEquations (system : EquationSystem) : Prop :=
  ∃ (x y : ℝ), 
    system.eq1.eq x y ∧ 
    system.eq2.eq x y ∧
    (∀ z, system.eq1.eq x z ↔ system.eq1.a * x + system.eq1.b * z = system.eq1.c) ∧
    (∀ z, system.eq2.eq x z ↔ system.eq2.a * x + system.eq2.b * z = system.eq2.c)

theorem specific_system_is_linear : isSystemOfTwoLinearEquations specificSystem := by
  sorry

end NUMINAMATH_CALUDE_specific_system_is_linear_l3110_311067


namespace NUMINAMATH_CALUDE_expression_simplification_l3110_311078

theorem expression_simplification :
  (12 - 2 * Real.sqrt 35 + Real.sqrt 14 + Real.sqrt 10) / (Real.sqrt 7 - Real.sqrt 5 + Real.sqrt 2) = 2 * Real.sqrt 7 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3110_311078


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l3110_311008

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- Theorem: In a rhombus with one diagonal of 25 m and an area of 625 m², the other diagonal is 50 m -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.d1 = 25) (h2 : r.area = 625) : r.d2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l3110_311008


namespace NUMINAMATH_CALUDE_negative_reals_inequality_l3110_311054

theorem negative_reals_inequality (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + b + c ≤ (a^2 + b^2) / (2*c) + (b^2 + c^2) / (2*a) + (c^2 + a^2) / (2*b) ∧
  (a^2 + b^2) / (2*c) + (b^2 + c^2) / (2*a) + (c^2 + a^2) / (2*b) ≤ a^2 / (b*c) + b^2 / (c*a) + c^2 / (a*b) :=
by sorry

end NUMINAMATH_CALUDE_negative_reals_inequality_l3110_311054


namespace NUMINAMATH_CALUDE_binomial_18_10_l3110_311056

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l3110_311056


namespace NUMINAMATH_CALUDE_product_of_four_six_seven_fourteen_l3110_311011

theorem product_of_four_six_seven_fourteen : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_six_seven_fourteen_l3110_311011


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3110_311060

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (d : RepeatingDecimal) : ℚ := sorry

/-- The repeating decimal 7.316316316... -/
def ourDecimal : RepeatingDecimal := { integerPart := 7, repeatingPart := 316 }

theorem repeating_decimal_equals_fraction :
  repeatingDecimalToRational ourDecimal = 7309 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3110_311060


namespace NUMINAMATH_CALUDE_power_two_ge_cube_l3110_311061

theorem power_two_ge_cube (n : ℕ) (h : n ≥ 10) : 2^n ≥ n^3 := by sorry

end NUMINAMATH_CALUDE_power_two_ge_cube_l3110_311061


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_vector_sum_l3110_311096

/-- An isosceles right triangle with hypotenuse of length 6 -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  isIsosceles : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  hypotenuseLength : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 36

def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vectorSum (t : IsoscelesRightTriangle) : ℝ :=
  let AB := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  let AC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  let BA := (-AB.1, -AB.2)
  let CA := (-AC.1, -AC.2)
  let CB := (-BC.1, -BC.2)
  dotProduct AB AC + dotProduct BC BA + dotProduct CA CB

theorem isosceles_right_triangle_vector_sum (t : IsoscelesRightTriangle) :
  vectorSum t = 36 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_vector_sum_l3110_311096


namespace NUMINAMATH_CALUDE_remaining_aces_probability_l3110_311085

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in each hand -/
def HandSize : ℕ := 13

/-- Represents the number of aces in a standard deck -/
def TotalAces : ℕ := 4

/-- Computes the probability of a specific person having the remaining aces
    given that one person has one ace -/
def probabilityOfRemainingAces (deck : ℕ) (handSize : ℕ) (totalAces : ℕ) : ℚ :=
  22 / 703

theorem remaining_aces_probability :
  probabilityOfRemainingAces StandardDeck HandSize TotalAces = 22 / 703 := by
  sorry

end NUMINAMATH_CALUDE_remaining_aces_probability_l3110_311085


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3110_311059

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 26 + 12 * Real.sqrt 6 :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀ + 3 / b₀) = 26 + 12 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3110_311059


namespace NUMINAMATH_CALUDE_tetrahedron_cut_vertices_l3110_311010

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Finset (Fin 4)

/-- The result of cutting off a vertex from a polyhedron -/
def cutVertex (p : RegularTetrahedron) (v : Fin 4) : ℕ := 3

/-- The number of vertices in the shape resulting from cutting off all vertices of a regular tetrahedron -/
def verticesAfterCutting (t : RegularTetrahedron) : ℕ :=
  t.vertices.sum (λ v => cutVertex t v)

/-- Theorem: Cutting off all vertices of a regular tetrahedron results in a shape with 12 vertices -/
theorem tetrahedron_cut_vertices (t : RegularTetrahedron) :
  verticesAfterCutting t = 12 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_cut_vertices_l3110_311010


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l3110_311079

theorem sin_plus_cos_value (α : ℝ) (h : (Real.sin (α - π/4)) / (Real.cos (2*α)) = -Real.sqrt 2) : 
  Real.sin α + Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l3110_311079


namespace NUMINAMATH_CALUDE_budget_research_development_l3110_311038

theorem budget_research_development (transportation utilities equipment supplies salaries research_development : ℝ) : 
  transportation = 20 →
  utilities = 5 →
  equipment = 4 →
  supplies = 2 →
  salaries = 216 / 360 * 100 →
  transportation + utilities + equipment + supplies + salaries + research_development = 100 →
  research_development = 9 := by
sorry

end NUMINAMATH_CALUDE_budget_research_development_l3110_311038


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3110_311094

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > a) ∧ (∃ a, a ≤ 1 ∧ a^2 > a) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3110_311094


namespace NUMINAMATH_CALUDE_x_eq_two_is_axis_of_symmetry_l3110_311065

-- Define a function f with the given property
def f (x : ℝ) : ℝ := sorry

-- State the condition that f(x) = f(4-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (4 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem stating that x = 2 is an axis of symmetry
theorem x_eq_two_is_axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end NUMINAMATH_CALUDE_x_eq_two_is_axis_of_symmetry_l3110_311065


namespace NUMINAMATH_CALUDE_double_sides_same_perimeter_l3110_311077

/-- A regular polygon with n sides and side length s -/
structure RegularPolygon where
  n : ℕ
  s : ℝ
  h_n : n ≥ 3

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ := p.n * p.s

theorem double_sides_same_perimeter (p : RegularPolygon) :
  ∃ (q : RegularPolygon), q.n = 2 * p.n ∧ perimeter q = perimeter p ∧ q.s = p.s / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_sides_same_perimeter_l3110_311077


namespace NUMINAMATH_CALUDE_derivative_e_cubed_l3110_311019

-- e is the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Statement: The derivative of e^3 is e^3
theorem derivative_e_cubed : 
  deriv (fun x : ℝ => e^3) = fun x : ℝ => e^3 :=
sorry

end NUMINAMATH_CALUDE_derivative_e_cubed_l3110_311019


namespace NUMINAMATH_CALUDE_sum_of_cumulative_sums_geometric_sequence_l3110_311049

/-- The sum of cumulative sums of a geometric sequence -/
theorem sum_of_cumulative_sums_geometric_sequence (a₁ q : ℝ) (h : |q| < 1) :
  ∃ (S : ℕ → ℝ), (∀ n, S n = a₁ * (1 - q^n) / (1 - q)) ∧
  (∑' n, S n) = a₁ / (1 - q)^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cumulative_sums_geometric_sequence_l3110_311049


namespace NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l3110_311091

/-- Given two parabolas C₁ and C₂ with specific properties, prove that C₂ passes through a fixed point. -/
theorem parabola_intersection_fixed_point 
  (C₁_vertex : ℝ × ℝ) 
  (C₁_focus : ℝ × ℝ)
  (a b : ℝ) :
  let C₁_vertex_x := Real.sqrt 2 - 1
  let C₁_vertex_y := 1
  let C₁_focus_x := Real.sqrt 2 - 3/4
  let C₁_focus_y := 1
  let C₂_eq (x y : ℝ) := y^2 - a*y + x + 2*b = 0
  let fixed_point := (Real.sqrt 2 - 1/2, 1)
  C₁_vertex = (C₁_vertex_x, C₁_vertex_y) →
  C₁_focus = (C₁_focus_x, C₁_focus_y) →
  (∃ (x₀ y₀ : ℝ), 
    (y₀^2 - 2*y₀ - x₀ + Real.sqrt 2 = 0) ∧ 
    (C₂_eq x₀ y₀) ∧ 
    ((2*y₀ - 2) * (2*y₀ - a) = -1)) →
  C₂_eq fixed_point.1 fixed_point.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l3110_311091


namespace NUMINAMATH_CALUDE_divisibility_condition_l3110_311080

theorem divisibility_condition (n : ℕ+) :
  (6^n.val - 1) ∣ (7^n.val - 1) ↔ ∃ k : ℕ, n.val = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3110_311080


namespace NUMINAMATH_CALUDE_grid_toothpick_count_l3110_311057

/-- Calculates the number of toothpicks in a grid with a missing center block -/
def toothpick_count (length width missing_size : ℕ) : ℕ :=
  let vertical := (length + 1) * width - missing_size * missing_size
  let horizontal := (width + 1) * length - missing_size * missing_size
  vertical + horizontal

/-- Theorem stating the correct number of toothpicks for the given grid -/
theorem grid_toothpick_count :
  toothpick_count 30 20 2 = 1242 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpick_count_l3110_311057


namespace NUMINAMATH_CALUDE_zebra_stripes_l3110_311028

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes = white stripes + 1
  b = w + 7 →      -- White stripes = wide black stripes + 7
  n = 8 :=         -- Number of narrow black stripes is 8
by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l3110_311028


namespace NUMINAMATH_CALUDE_st_length_l3110_311072

/-- Rectangle WXYZ with parallelogram PQRS inside -/
structure RectangleWithParallelogram where
  /-- Width of the rectangle -/
  width : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- Length of PW -/
  pw : ℝ
  /-- Length of WS -/
  ws : ℝ
  /-- Length of SZ -/
  sz : ℝ
  /-- Length of ZR -/
  zr : ℝ
  /-- PT is perpendicular to SR -/
  pt_perp_sr : Bool

/-- The main theorem -/
theorem st_length (rect : RectangleWithParallelogram) 
  (h1 : rect.width = 15)
  (h2 : rect.height = 9)
  (h3 : rect.pw = 3)
  (h4 : rect.ws = 4)
  (h5 : rect.sz = 5)
  (h6 : rect.zr = 12)
  (h7 : rect.pt_perp_sr = true) :
  ∃ (st : ℝ), st = 16 / 13 := by sorry

end NUMINAMATH_CALUDE_st_length_l3110_311072


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l3110_311093

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_proof (a b c : ℝ) :
  (∀ x, f a b c x ≥ 0) ∧  -- Minimum value is 0
  (∀ x, f a b c x = f a b c (-2 - x)) ∧  -- Symmetric about x = -1
  (∀ x ∈ Set.Ioo 0 5, x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1) →
  ∀ x, f a b c x = (1/4) * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l3110_311093


namespace NUMINAMATH_CALUDE_largest_number_proof_l3110_311002

theorem largest_number_proof (a b c d e : ℝ) 
  (ha : a = 0.997) (hb : b = 0.979) (hc : c = 0.99) (hd : d = 0.9709) (he : e = 0.999) :
  e = max a (max b (max c (max d e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3110_311002


namespace NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l3110_311089

theorem cos_seventeen_pi_sixths : Real.cos (17 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l3110_311089


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l3110_311099

theorem polynomial_root_sum (p q : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + p * (Complex.I * Real.sqrt 2 + 2) + q = 0 → 
  p + q = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l3110_311099


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3110_311026

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 5500

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  coefficient := 5.5
  exponent := 3
  h_coefficient := by sorry
}

/-- Theorem stating that the scientific notation is correct -/
theorem scientific_notation_correct : 
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3110_311026


namespace NUMINAMATH_CALUDE_smallest_n_for_equation_l3110_311000

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_n_for_equation : 
  ∃ (n : ℕ), n > 0 ∧ 2 * n * factorial n + 3 * factorial n = 5040 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → 2 * m * factorial m + 3 * factorial m ≠ 5040 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_equation_l3110_311000


namespace NUMINAMATH_CALUDE_average_seeds_per_grape_l3110_311013

/-- Theorem: Average number of seeds per grape -/
theorem average_seeds_per_grape 
  (total_seeds : ℕ) 
  (apple_seeds : ℕ) 
  (pear_seeds : ℕ) 
  (apples : ℕ) 
  (pears : ℕ) 
  (grapes : ℕ) 
  (seeds_needed : ℕ) 
  (h1 : total_seeds = 60)
  (h2 : apple_seeds = 6)
  (h3 : pear_seeds = 2)
  (h4 : apples = 4)
  (h5 : pears = 3)
  (h6 : grapes = 9)
  (h7 : seeds_needed = 3)
  : (total_seeds - (apples * apple_seeds + pears * pear_seeds) - seeds_needed) / grapes = 3 :=
by sorry

end NUMINAMATH_CALUDE_average_seeds_per_grape_l3110_311013


namespace NUMINAMATH_CALUDE_chess_tournament_red_pairs_l3110_311005

/-- Represents the number of pairs in a chess tournament where both players wear red hats. -/
def red_red_pairs (green_players : ℕ) (red_players : ℕ) (total_pairs : ℕ) (green_green_pairs : ℕ) : ℕ :=
  (red_players - (total_pairs * 2 - green_players - red_players)) / 2

/-- Theorem stating that in the given chess tournament scenario, there are 27 pairs where both players wear red hats. -/
theorem chess_tournament_red_pairs : 
  red_red_pairs 64 68 66 25 = 27 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_red_pairs_l3110_311005


namespace NUMINAMATH_CALUDE_no_even_increasing_function_l3110_311068

open Function

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem stating that no function can be both even and increasing
theorem no_even_increasing_function : ¬ ∃ f : ℝ → ℝ, IsEven f ∧ IsIncreasing f := by
  sorry

end NUMINAMATH_CALUDE_no_even_increasing_function_l3110_311068


namespace NUMINAMATH_CALUDE_intercept_sum_mod_50_l3110_311066

theorem intercept_sum_mod_50 : ∃! (x₀ y₀ : ℕ), 
  x₀ < 50 ∧ y₀ < 50 ∧ 
  (7 * x₀ ≡ 2 [MOD 50]) ∧
  (3 * y₀ ≡ 48 [MOD 50]) ∧
  ((x₀ + y₀) ≡ 2 [MOD 50]) := by
sorry

end NUMINAMATH_CALUDE_intercept_sum_mod_50_l3110_311066


namespace NUMINAMATH_CALUDE_cookie_store_spending_l3110_311009

theorem cookie_store_spending : ∀ (ben david : ℝ),
  (david = 0.6 * ben) →  -- For every dollar Ben spent, David spent 40 cents less
  (ben = david + 16) →   -- Ben paid $16.00 more than David
  (ben + david = 64) :=  -- The total amount they spent together
by
  sorry

end NUMINAMATH_CALUDE_cookie_store_spending_l3110_311009


namespace NUMINAMATH_CALUDE_marks_remaining_money_l3110_311021

def initial_amount : ℕ := 85
def books_seven_dollars : ℕ := 3
def books_five_dollars : ℕ := 4
def books_nine_dollars : ℕ := 2

def cost_seven_dollars : ℕ := 7
def cost_five_dollars : ℕ := 5
def cost_nine_dollars : ℕ := 9

theorem marks_remaining_money :
  initial_amount - 
  (books_seven_dollars * cost_seven_dollars + 
   books_five_dollars * cost_five_dollars + 
   books_nine_dollars * cost_nine_dollars) = 26 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l3110_311021


namespace NUMINAMATH_CALUDE_a_divisible_by_power_of_three_l3110_311097

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (3 * (a n)^2 + 1) / 2 - a n

theorem a_divisible_by_power_of_three (k : ℕ) : 
  ∃ m : ℕ, a (3^k) = m * (3^k) := by sorry

end NUMINAMATH_CALUDE_a_divisible_by_power_of_three_l3110_311097


namespace NUMINAMATH_CALUDE_exponent_division_l3110_311081

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3110_311081


namespace NUMINAMATH_CALUDE_angel_food_cake_egg_whites_angel_food_cake_proof_l3110_311087

theorem angel_food_cake_egg_whites (aquafaba_per_egg_white : ℕ) 
  (num_cakes : ℕ) (total_aquafaba : ℕ) : ℕ :=
  let egg_whites_per_cake := (total_aquafaba / aquafaba_per_egg_white) / num_cakes
  egg_whites_per_cake

theorem angel_food_cake_proof : 
  angel_food_cake_egg_whites 2 2 32 = 8 := by
  sorry

end NUMINAMATH_CALUDE_angel_food_cake_egg_whites_angel_food_cake_proof_l3110_311087


namespace NUMINAMATH_CALUDE_greatest_multiple_of_8_no_repeats_remainder_l3110_311040

/-- The greatest integer multiple of 8 with no repeating digits -/
def N : ℕ :=
  sorry

/-- Predicate to check if a natural number has no repeating digits -/
def has_no_repeating_digits (n : ℕ) : Prop :=
  sorry

theorem greatest_multiple_of_8_no_repeats_remainder : 
  N % 1000 = 120 ∧ 
  N % 8 = 0 ∧
  has_no_repeating_digits N ∧
  ∀ m : ℕ, m % 8 = 0 → has_no_repeating_digits m → m ≤ N :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_8_no_repeats_remainder_l3110_311040


namespace NUMINAMATH_CALUDE_bicycle_owners_without_cars_proof_l3110_311075

/-- Represents the number of adults who own bicycles but not cars in a population where every adult owns either a bicycle, a car, or both. -/
def bicycle_owners_without_cars (total_adults bicycle_owners car_owners : ℕ) : ℕ :=
  bicycle_owners - (bicycle_owners + car_owners - total_adults)

/-- Theorem stating that in a population of 500 adults where each adult owns either a bicycle, a car, or both, 
    given that 450 adults own bicycles and 120 adults own cars, the number of bicycle owners who do not own a car is 380. -/
theorem bicycle_owners_without_cars_proof :
  bicycle_owners_without_cars 500 450 120 = 380 := by
  sorry

#eval bicycle_owners_without_cars 500 450 120

end NUMINAMATH_CALUDE_bicycle_owners_without_cars_proof_l3110_311075


namespace NUMINAMATH_CALUDE_duck_eggs_sum_l3110_311071

theorem duck_eggs_sum (yesterday_eggs : ℕ) (fewer_today : ℕ) : 
  yesterday_eggs = 1925 →
  fewer_today = 138 →
  yesterday_eggs + (yesterday_eggs - fewer_today) = 3712 := by
  sorry

end NUMINAMATH_CALUDE_duck_eggs_sum_l3110_311071


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3110_311058

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3110_311058


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3110_311030

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (8 * x₁^2 + 12 * x₁ - 14 = 0) → 
  (8 * x₂^2 + 12 * x₂ - 14 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 23/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3110_311030


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l3110_311084

theorem solution_set_of_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l3110_311084


namespace NUMINAMATH_CALUDE_factorization_3x2_minus_27y2_l3110_311001

theorem factorization_3x2_minus_27y2 (x y : ℝ) : 3 * x^2 - 27 * y^2 = 3 * (x + 3*y) * (x - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x2_minus_27y2_l3110_311001


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3110_311050

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 3) and (4, -3) is 7. -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, 3)
  let p₂ : ℝ × ℝ := (4, -3)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 7 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3110_311050


namespace NUMINAMATH_CALUDE_sum_of_squares_l3110_311027

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_cubes_eq_sum_fifth : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3110_311027


namespace NUMINAMATH_CALUDE_count_lines_with_integer_chord_l3110_311062

/-- Represents a line in the form kx - y - 4k + 1 = 0 --/
structure Line where
  k : ℝ

/-- Represents the circle x^2 + (y + 1)^2 = 25 --/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 + 1)^2 = 25}

/-- Returns true if the line intersects the circle with a chord of integer length --/
def hasIntegerChord (l : Line) : Prop :=
  ∃ n : ℕ, ∃ p q : ℝ × ℝ,
    p ∈ Circle ∧ q ∈ Circle ∧
    l.k * p.1 - p.2 - 4 * l.k + 1 = 0 ∧
    l.k * q.1 - q.2 - 4 * l.k + 1 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = n^2

/-- The theorem to be proved --/
theorem count_lines_with_integer_chord :
  ∃! (s : Finset Line), s.card = 10 ∧ ∀ l ∈ s, hasIntegerChord l :=
sorry

end NUMINAMATH_CALUDE_count_lines_with_integer_chord_l3110_311062


namespace NUMINAMATH_CALUDE_system_solution_and_sum_l3110_311076

theorem system_solution_and_sum :
  ∃ (x y : ℚ),
    (4 * x - 6 * y = -3) ∧
    (8 * x + 3 * y = 6) ∧
    (x = 9/20) ∧
    (y = 4/5) ∧
    (x + y = 5/4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_and_sum_l3110_311076


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3110_311052

theorem quadratic_solution_property (a b : ℝ) : 
  (3 * a^2 - 9 * a + 21 = 0) ∧ 
  (3 * b^2 - 9 * b + 21 = 0) →
  (3 * a - 4) * (6 * b - 8) = 50 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3110_311052


namespace NUMINAMATH_CALUDE_quadratic_order_l3110_311036

/-- Given m < -2 and points on a quadratic function, prove y3 < y2 < y1 -/
theorem quadratic_order (m : ℝ) (y1 y2 y3 : ℝ)
  (h_m : m < -2)
  (h_y1 : y1 = (m - 1)^2 + 2*(m - 1))
  (h_y2 : y2 = m^2 + 2*m)
  (h_y3 : y3 = (m + 1)^2 + 2*(m + 1)) :
  y3 < y2 ∧ y2 < y1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_order_l3110_311036


namespace NUMINAMATH_CALUDE_first_box_weight_proof_l3110_311032

/-- The weight of the first box given the conditions in the problem -/
def first_box_weight : ℝ := 24

/-- The weight of the third box -/
def third_box_weight : ℝ := 13

/-- The difference between the weight of the first and third box -/
def weight_difference : ℝ := 11

theorem first_box_weight_proof :
  first_box_weight = third_box_weight + weight_difference := by
  sorry

end NUMINAMATH_CALUDE_first_box_weight_proof_l3110_311032


namespace NUMINAMATH_CALUDE_symmedian_circle_theorem_l3110_311064

/-- A triangle with side lengths a, b, and c is non-isosceles if no two sides are equal. -/
def NonIsoscelesTriangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- A circle passes through the feet of the symmedians of a triangle if it intersects
    each side of the triangle at the point where the symmedian meets that side. -/
def CircleThroughSymmedianFeet (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∃ (x y z : ℝ), x^2 + y^2 = r^2 ∧ y^2 + z^2 = r^2 ∧ z^2 + x^2 = r^2

/-- A circle is tangent to one side of a triangle if it touches that side at exactly one point. -/
def CircleTangentToSide (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ (∃ (x : ℝ), x^2 = r^2 ∨ ∃ (y : ℝ), y^2 = r^2 ∨ ∃ (z : ℝ), z^2 = r^2)

/-- Three positive real numbers form a geometric progression if the ratio between
    consecutive terms is constant. -/
def GeometricProgression (x y z : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ y = x * r ∧ z = y * r

/-- Main theorem: If a circle passes through the feet of the symmedians of a non-isosceles
    triangle and is tangent to one side, then the sums of squares of side lengths taken
    pairwise form a geometric progression. -/
theorem symmedian_circle_theorem (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  NonIsoscelesTriangle a b c →
  CircleThroughSymmedianFeet a b c →
  CircleTangentToSide a b c →
  GeometricProgression (a^2 + b^2) (b^2 + c^2) (c^2 + a^2) ∨
  GeometricProgression (b^2 + c^2) (c^2 + a^2) (a^2 + b^2) ∨
  GeometricProgression (c^2 + a^2) (a^2 + b^2) (b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_symmedian_circle_theorem_l3110_311064


namespace NUMINAMATH_CALUDE_max_candies_equals_complete_graph_edges_l3110_311044

/-- The number of ones initially on the board -/
def initial_ones : Nat := 30

/-- The number of minutes the process continues -/
def total_minutes : Nat := 30

/-- Represents the board state at any given time -/
structure Board where
  numbers : List Nat

/-- Represents a single operation of erasing two numbers and writing their sum -/
def erase_and_sum (b : Board) (i j : Nat) : Board := sorry

/-- The number of candies eaten in a single operation -/
def candies_eaten (b : Board) (i j : Nat) : Nat := sorry

/-- The maximum number of candies that can be eaten -/
def max_candies : Nat := (initial_ones * (initial_ones - 1)) / 2

/-- Theorem stating that the maximum number of candies eaten is equal to
    the number of edges in a complete graph with 'initial_ones' vertices -/
theorem max_candies_equals_complete_graph_edges :
  max_candies = (initial_ones * (initial_ones - 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_max_candies_equals_complete_graph_edges_l3110_311044


namespace NUMINAMATH_CALUDE_slope_does_not_exist_for_vertical_line_l3110_311035

/-- A line is vertical if its equation can be written in the form x = constant -/
def IsVerticalLine (a b : ℝ) : Prop := a ≠ 0 ∧ ∀ x y : ℝ, a * x + b = 0 → x = -b / a

/-- The slope of a line does not exist if the line is vertical -/
def SlopeDoesNotExist (a b : ℝ) : Prop := IsVerticalLine a b

theorem slope_does_not_exist_for_vertical_line (a b : ℝ) :
  a * x + b = 0 → a ≠ 0 → SlopeDoesNotExist a b := by sorry

end NUMINAMATH_CALUDE_slope_does_not_exist_for_vertical_line_l3110_311035


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3110_311034

/-- Sum of a geometric series with n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The given geometric series -/
def givenSeries : List ℚ := [1/4, -1/16, 1/64, -1/256, 1/1024]

theorem geometric_series_sum :
  let a₁ : ℚ := 1/4
  let r : ℚ := -1/4
  let n : ℕ := 5
  geometricSum a₁ r n = 205/1024 ∧ givenSeries.sum = 205/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3110_311034


namespace NUMINAMATH_CALUDE_solution_difference_l3110_311041

def is_solution (x : ℝ) : Prop :=
  (4 * x - 12) / (x^2 + 2*x - 15) = x + 2

theorem solution_difference (p q : ℝ) 
  (hp : is_solution p) 
  (hq : is_solution q) 
  (hdistinct : p ≠ q) 
  (horder : p > q) : 
  p - q = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l3110_311041


namespace NUMINAMATH_CALUDE_i_power_2010_l3110_311063

theorem i_power_2010 : (Complex.I : ℂ) ^ 2010 = -1 := by sorry

end NUMINAMATH_CALUDE_i_power_2010_l3110_311063


namespace NUMINAMATH_CALUDE_sum_equals_5186_l3110_311012

theorem sum_equals_5186 : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_5186_l3110_311012


namespace NUMINAMATH_CALUDE_equation_solutions_l3110_311033

theorem equation_solutions (k : ℤ) (x₁ x₂ x₃ x₄ y₁ : ℤ) :
  (y₁^2 - k = x₁^3) ∧
  ((y₁ - 1)^2 - k = x₂^3) ∧
  ((y₁ - 2)^2 - k = x₃^3) ∧
  ((y₁ - 3)^2 - k = x₄^3) →
  k ≡ 17 [ZMOD 63] :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3110_311033


namespace NUMINAMATH_CALUDE_three_power_fraction_equals_five_fourths_l3110_311039

theorem three_power_fraction_equals_five_fourths :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_three_power_fraction_equals_five_fourths_l3110_311039


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3110_311069

theorem polynomial_division_remainder :
  let f (x : ℝ) := x^4 - 7*x^3 + 18*x^2 - 28*x + 15
  let g (x : ℝ) := x^2 - 3*x + 16/3
  let q (x : ℝ) := x^2 - 4*x + 10/3
  let r (x : ℝ) := 2*x + 103/9
  ∀ x, f x = g x * q x + r x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3110_311069


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l3110_311086

theorem quadratic_no_solution (a : ℝ) : 
  ({x : ℝ | x^2 - x + a = 0} : Set ℝ) = ∅ → a > 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l3110_311086


namespace NUMINAMATH_CALUDE_factors_of_sixty_l3110_311014

theorem factors_of_sixty : Nat.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_sixty_l3110_311014


namespace NUMINAMATH_CALUDE_exam_marks_l3110_311055

theorem exam_marks (full_marks : ℕ) (A B C D : ℕ) : 
  full_marks = 500 →
  A = B - B / 10 →
  B = C + C / 4 →
  C = D - D / 5 →
  D = full_marks * 4 / 5 →
  A = 360 := by sorry

end NUMINAMATH_CALUDE_exam_marks_l3110_311055


namespace NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3110_311048

/-- The radius of the smallest sphere centered at the origin that contains
    ten spheres of radius 2 positioned at the corners of a cube with side length 4 -/
theorem smallest_enclosing_sphere_radius (r : ℝ) (s : ℝ) : r = 2 ∧ s = 4 →
  (2 * Real.sqrt 3 + 2 : ℝ) = 
    (s * Real.sqrt 3 / 2 + r : ℝ) := by sorry

end NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3110_311048


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3110_311098

theorem triangle_angle_measure (A B : Real) (a b : Real) : 
  0 < A ∧ 0 < B ∧ 0 < a ∧ 0 < b →  -- Ensure positive values
  A = 2 * B →                      -- Condition: A = 2B
  a / b = Real.sqrt 2 →            -- Condition: a:b = √2:1
  A = 90 * (π / 180) :=            -- Conclusion: A = 90° (in radians)
by
  sorry

#check triangle_angle_measure

end NUMINAMATH_CALUDE_triangle_angle_measure_l3110_311098


namespace NUMINAMATH_CALUDE_isabella_hair_growth_l3110_311051

def monthly_growth : List Float := [0.5, 1, 0.75, 1.25, 1, 0.5, 1.5, 1, 0.25, 1.5, 1.25, 0.75]

theorem isabella_hair_growth :
  monthly_growth.sum = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_growth_l3110_311051
