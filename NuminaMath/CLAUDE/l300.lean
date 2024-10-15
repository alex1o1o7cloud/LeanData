import Mathlib

namespace NUMINAMATH_CALUDE_stating_tour_cost_is_correct_l300_30009

/-- Represents the cost of a tour at an aqua park -/
def tour_cost : ℝ := 6

/-- Represents the admission fee for the aqua park -/
def admission_fee : ℝ := 12

/-- Represents the total number of people in the first group -/
def group1_size : ℕ := 10

/-- Represents the total number of people in the second group -/
def group2_size : ℕ := 5

/-- Represents the total earnings of the aqua park -/
def total_earnings : ℝ := 240

/-- 
Theorem stating that the tour cost is correct given the problem conditions
-/
theorem tour_cost_is_correct : 
  group1_size * (admission_fee + tour_cost) + group2_size * admission_fee = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_stating_tour_cost_is_correct_l300_30009


namespace NUMINAMATH_CALUDE_tangent_condition_l300_30007

/-- A line with equation kx - y - 3√2 = 0 is tangent to the circle x² + y² = 9 -/
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), k*x - y - 3*Real.sqrt 2 = 0 ∧ x^2 + y^2 = 9 ∧
  ∀ (x' y' : ℝ), k*x' - y' - 3*Real.sqrt 2 = 0 → x'^2 + y'^2 ≥ 9

/-- k = 1 is a sufficient but not necessary condition for the line to be tangent -/
theorem tangent_condition : 
  (is_tangent 1) ∧ (∃ (k : ℝ), k ≠ 1 ∧ is_tangent k) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_l300_30007


namespace NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l300_30085

theorem smallest_factor_of_32_with_sum_3 (a b c : Int) : 
  a * b * c = 32 → a + b + c = 3 → min a (min b c) = -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l300_30085


namespace NUMINAMATH_CALUDE_coin_stack_theorem_l300_30039

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

def coin_arrangements (n : ℕ) : ℕ := fibonacci (n + 2)

theorem coin_stack_theorem :
  coin_arrangements 10 = 233 :=
by sorry

end NUMINAMATH_CALUDE_coin_stack_theorem_l300_30039


namespace NUMINAMATH_CALUDE_matts_age_l300_30052

theorem matts_age (john_age matt_age : ℕ) 
  (h1 : matt_age = 4 * john_age - 3)
  (h2 : matt_age + john_age = 52) :
  matt_age = 41 := by
  sorry

end NUMINAMATH_CALUDE_matts_age_l300_30052


namespace NUMINAMATH_CALUDE_water_left_in_cooler_l300_30018

/-- Proves that the amount of water left in the cooler after filling all cups is 84 ounces -/
theorem water_left_in_cooler : 
  let initial_gallons : ℕ := 3
  let ounces_per_cup : ℕ := 6
  let rows : ℕ := 5
  let chairs_per_row : ℕ := 10
  let ounces_per_gallon : ℕ := 128
  
  let total_chairs : ℕ := rows * chairs_per_row
  let initial_ounces : ℕ := initial_gallons * ounces_per_gallon
  let ounces_used : ℕ := total_chairs * ounces_per_cup
  let ounces_left : ℕ := initial_ounces - ounces_used

  ounces_left = 84 := by sorry

end NUMINAMATH_CALUDE_water_left_in_cooler_l300_30018


namespace NUMINAMATH_CALUDE_english_teachers_count_l300_30040

def committee_size (E : ℕ) : ℕ := E + 4 + 2

def probability_two_math_teachers (E : ℕ) : ℚ :=
  6 / (committee_size E * (committee_size E - 1) / 2)

theorem english_teachers_count :
  ∃ E : ℕ, probability_two_math_teachers E = 1/12 ∧ E = 3 :=
sorry

end NUMINAMATH_CALUDE_english_teachers_count_l300_30040


namespace NUMINAMATH_CALUDE_keith_turnips_l300_30072

theorem keith_turnips (total : ℕ) (alyssa : ℕ) (keith : ℕ)
  (h1 : total = 15)
  (h2 : alyssa = 9)
  (h3 : keith + alyssa = total) :
  keith = 15 - 9 :=
by sorry

end NUMINAMATH_CALUDE_keith_turnips_l300_30072


namespace NUMINAMATH_CALUDE_evaluate_otimes_expression_l300_30028

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem evaluate_otimes_expression :
  (otimes (otimes 5 3) 2) = 293/15 := by sorry

end NUMINAMATH_CALUDE_evaluate_otimes_expression_l300_30028


namespace NUMINAMATH_CALUDE_line_l1_parallel_line_l1_perpendicular_l300_30050

-- Define the line l passing through points A(4,0) and B(0,3)
def line_l (x y : ℝ) : Prop := x / 4 + y / 3 = 1

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y = 2

-- Define the intersection point of line1 and line2
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define a parallel line to l
def parallel_line (m : ℝ) (x y : ℝ) : Prop := x / 4 + y / 3 = m

-- Define a perpendicular line to l
def perpendicular_line (n : ℝ) (x y : ℝ) : Prop := x / 3 - y / 4 = n

theorem line_l1_parallel :
  ∃ m : ℝ, (∀ x y : ℝ, intersection_point x y → parallel_line m x y) ∧
           (∀ x y : ℝ, parallel_line m x y ↔ 3 * x + 4 * y - 9 = 0) :=
sorry

theorem line_l1_perpendicular :
  ∃ n1 n2 : ℝ, n1 ≠ n2 ∧
    (∀ x y : ℝ, perpendicular_line n1 x y ↔ 4 * x - 3 * y - 12 = 0) ∧
    (∀ x y : ℝ, perpendicular_line n2 x y ↔ 4 * x - 3 * y + 12 = 0) ∧
    (∀ n : ℝ, n = n1 ∨ n = n2 →
      ∃ x1 y1 x2 y2 : ℝ,
        perpendicular_line n x1 0 ∧ perpendicular_line n 0 y2 ∧
        x1 * y2 / 2 = 6) :=
sorry

end NUMINAMATH_CALUDE_line_l1_parallel_line_l1_perpendicular_l300_30050


namespace NUMINAMATH_CALUDE_small_circle_radius_l300_30076

theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 →  -- radius of large circle is 10 meters
  (3 * (2 * r) = 2 * R) →  -- three diameters of small circles equal diameter of large circle
  r = 10 / 3 :=  -- radius of small circle is 10/3 meters
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l300_30076


namespace NUMINAMATH_CALUDE_range_of_m_existence_of_a_b_l300_30094

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Part 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 2 4, f m x ≥ -1) ↔ m ≤ 4 :=
sorry

-- Part 2: Existence of integers a and b
theorem existence_of_a_b :
  ∃ (a b : ℤ), a < b ∧
  (∀ x : ℝ, a ≤ f (↑a + ↑b - 1) x ∧ f (↑a + ↑b - 1) x ≤ b ↔ a ≤ x ∧ x ≤ b) ∧
  a = 0 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_existence_of_a_b_l300_30094


namespace NUMINAMATH_CALUDE_first_divisible_by_three_and_seven_l300_30064

theorem first_divisible_by_three_and_seven (lower_bound upper_bound : ℕ) 
  (h_lower : lower_bound = 100) (h_upper : upper_bound = 600) :
  ∃ (n : ℕ), 
    n ≥ lower_bound ∧ 
    n ≤ upper_bound ∧ 
    n % 3 = 0 ∧ 
    n % 7 = 0 ∧
    ∀ (m : ℕ), m ≥ lower_bound ∧ m < n → m % 3 ≠ 0 ∨ m % 7 ≠ 0 :=
by
  -- Proof goes here
  sorry

#eval (105 : ℕ)

end NUMINAMATH_CALUDE_first_divisible_by_three_and_seven_l300_30064


namespace NUMINAMATH_CALUDE_sum_always_positive_l300_30078

/-- An increasing function on ℝ -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_incr : IncreasingFunction f)
  (h_odd : OddFunction f)
  (h_arith : ArithmeticSequence a)
  (h_a3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l300_30078


namespace NUMINAMATH_CALUDE_degrees_120_45_equals_120_75_l300_30095

/-- Converts degrees and minutes to decimal degrees -/
def degreesMinutesToDecimal (degrees : ℕ) (minutes : ℕ) : ℚ :=
  degrees + (minutes : ℚ) / 60

/-- Theorem stating that 120°45' is equal to 120.75° -/
theorem degrees_120_45_equals_120_75 :
  degreesMinutesToDecimal 120 45 = 120.75 := by
  sorry

end NUMINAMATH_CALUDE_degrees_120_45_equals_120_75_l300_30095


namespace NUMINAMATH_CALUDE_max_surface_area_inscribed_cylinder_l300_30044

/-- Given a cone with height h and base radius r, where h > 2r, 
    the maximum total surface area of an inscribed cylinder is πh²r / (2(h - r)). -/
theorem max_surface_area_inscribed_cylinder (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) (h_gt_2r : h > 2 * r) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    x ≤ r ∧ y ≤ h ∧
    (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' ≤ r → y' ≤ h →
      2 * π * x' * (x' + y') ≤ 2 * π * x * (x + y)) ∧
    2 * π * x * (x + y) = π * h^2 * r / (2 * (h - r)) :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_inscribed_cylinder_l300_30044


namespace NUMINAMATH_CALUDE_all_rooms_on_same_hall_l300_30013

/-- A type representing a hall in the castle -/
def Hall : Type := ℕ

/-- A function that assigns a hall to each room number -/
def room_to_hall : ℕ → Hall := sorry

/-- The property that room n is on the same hall as rooms 2n+1 and 3n+1 -/
def hall_property (room_to_hall : ℕ → Hall) : Prop :=
  ∀ n : ℕ, (room_to_hall n = room_to_hall (2*n + 1)) ∧ (room_to_hall n = room_to_hall (3*n + 1))

/-- The theorem stating that all rooms must be on the same hall -/
theorem all_rooms_on_same_hall (room_to_hall : ℕ → Hall) 
  (h : hall_property room_to_hall) : 
  ∀ m n : ℕ, room_to_hall m = room_to_hall n :=
by sorry

end NUMINAMATH_CALUDE_all_rooms_on_same_hall_l300_30013


namespace NUMINAMATH_CALUDE_donkey_elephant_weight_difference_l300_30053

-- Define the weights and conversion factor
def elephant_weight_tons : ℝ := 3
def pounds_per_ton : ℝ := 2000
def combined_weight_pounds : ℝ := 6600

-- Define the theorem
theorem donkey_elephant_weight_difference : 
  let elephant_weight_pounds := elephant_weight_tons * pounds_per_ton
  let donkey_weight_pounds := combined_weight_pounds - elephant_weight_pounds
  let weight_difference_percentage := (elephant_weight_pounds - donkey_weight_pounds) / elephant_weight_pounds * 100
  weight_difference_percentage = 90 := by
sorry

end NUMINAMATH_CALUDE_donkey_elephant_weight_difference_l300_30053


namespace NUMINAMATH_CALUDE_z1_div_z2_equals_one_minus_two_i_l300_30020

-- Define complex numbers z1 and z2
def z1 : ℂ := Complex.mk 2 1
def z2 : ℂ := Complex.mk 0 1

-- Theorem statement
theorem z1_div_z2_equals_one_minus_two_i :
  z1 / z2 = Complex.mk 1 (-2) :=
sorry

end NUMINAMATH_CALUDE_z1_div_z2_equals_one_minus_two_i_l300_30020


namespace NUMINAMATH_CALUDE_magnitude_sum_perpendicular_vectors_l300_30023

/-- Given two vectors a and b in R², where a is perpendicular to b,
    prove that the magnitude of a + 2b is 2√10 -/
theorem magnitude_sum_perpendicular_vectors
  (a b : ℝ × ℝ)
  (h1 : a.1 = 4)
  (h2 : b = (1, -2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ‖a + 2 • b‖ = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_sum_perpendicular_vectors_l300_30023


namespace NUMINAMATH_CALUDE_simultaneous_colonies_count_l300_30004

/-- Represents the growth of bacteria colonies over time -/
def bacteriaGrowth (n : ℕ) (t : ℕ) : ℕ := n * 2^t

/-- The number of days it takes for a single colony to reach the habitat limit -/
def singleColonyLimit : ℕ := 25

/-- The number of days it takes for multiple colonies to reach the habitat limit -/
def multipleColoniesLimit : ℕ := 24

/-- Theorem stating that the number of simultaneously growing colonies is 2 -/
theorem simultaneous_colonies_count :
  ∃ (n : ℕ), n > 0 ∧ 
    bacteriaGrowth n multipleColoniesLimit = bacteriaGrowth 1 singleColonyLimit ∧ 
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_colonies_count_l300_30004


namespace NUMINAMATH_CALUDE_smallest_b_value_l300_30041

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1/a + 1/b ≤ 1/2) →
  b ≥ (7 + Real.sqrt 17) / 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l300_30041


namespace NUMINAMATH_CALUDE_grilled_cheese_slices_l300_30010

/-- The number of ham sandwiches made -/
def num_ham_sandwiches : ℕ := 10

/-- The number of grilled cheese sandwiches made -/
def num_grilled_cheese : ℕ := 10

/-- The number of cheese slices used in one ham sandwich -/
def cheese_per_ham : ℕ := 2

/-- The total number of cheese slices used -/
def total_cheese : ℕ := 50

/-- The number of cheese slices in one grilled cheese sandwich -/
def cheese_per_grilled_cheese : ℕ := (total_cheese - num_ham_sandwiches * cheese_per_ham) / num_grilled_cheese

theorem grilled_cheese_slices :
  cheese_per_grilled_cheese = 3 :=
sorry

end NUMINAMATH_CALUDE_grilled_cheese_slices_l300_30010


namespace NUMINAMATH_CALUDE_complex_number_sum_of_parts_l300_30030

theorem complex_number_sum_of_parts (a : ℝ) :
  let z : ℂ := a / (2 + Complex.I) + (2 + Complex.I) / 5
  (z.re + z.im = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_of_parts_l300_30030


namespace NUMINAMATH_CALUDE_vector_at_t_5_l300_30060

/-- A line in 3D space parameterized by t -/
def Line := ℝ → ℝ × ℝ × ℝ

/-- The given line satisfying the conditions -/
def givenLine : Line := sorry

theorem vector_at_t_5 (h1 : givenLine 1 = (2, -1, 3))
                      (h2 : givenLine 4 = (8, -5, 11)) :
  givenLine 5 = (10, -19/3, 41/3) := by sorry

end NUMINAMATH_CALUDE_vector_at_t_5_l300_30060


namespace NUMINAMATH_CALUDE_set_operations_l300_30034

def U : Set ℝ := {x | x ≤ 5}
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -4 < x ∧ x ≤ 2}

theorem set_operations :
  (A ∩ B = {x | -2 < x ∧ x ≤ 2}) ∧
  (A ∩ (U \ B) = {x | 2 < x ∧ x < 3}) ∧
  ((U \ A) ∪ B = {x | x ≤ 2 ∨ (3 ≤ x ∧ x ≤ 5)}) ∧
  ((U \ A) ∪ (U \ B) = {x | x ≤ -2 ∨ (2 < x ∧ x ≤ 5)}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l300_30034


namespace NUMINAMATH_CALUDE_opposite_roots_implies_n_eq_neg_two_l300_30051

/-- The equation has roots that are numerically equal but of opposite signs -/
def has_opposite_roots (k b d e n : ℝ) : Prop :=
  ∃ x : ℝ, (k * x^2 - b * x) / (d * x - e) = (n - 2) / (n + 2) ∧
            ∃ y : ℝ, y = -x ∧ (k * y^2 - b * y) / (d * y - e) = (n - 2) / (n + 2)

/-- Theorem stating that if the equation has roots that are numerically equal but of opposite signs, then n = -2 -/
theorem opposite_roots_implies_n_eq_neg_two (k b d e : ℝ) :
  ∀ n : ℝ, has_opposite_roots k b d e n → n = -2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_roots_implies_n_eq_neg_two_l300_30051


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l300_30033

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 5 → m ≤ n :=
  by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l300_30033


namespace NUMINAMATH_CALUDE_complex_equation_solution_l300_30029

theorem complex_equation_solution :
  ∃ (z : ℂ), ∃ (a b : ℝ),
    z = Complex.mk a b ∧
    z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I ∧
    a = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l300_30029


namespace NUMINAMATH_CALUDE_last_locker_opened_l300_30014

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the set of lockers -/
def Lockers := Fin 2048 → LockerState

/-- Defines the opening pattern for the lockers -/
def openingPattern (n : Nat) (lockers : Lockers) : Lockers :=
  sorry

/-- Defines the process of opening lockers until all are open -/
def openAllLockers (lockers : Lockers) : Nat :=
  sorry

/-- Theorem stating that the last locker to be opened is 1999 -/
theorem last_locker_opened (initialLockers : Lockers) 
    (h : ∀ i, initialLockers i = LockerState.Closed) : 
    openAllLockers initialLockers = 1999 :=
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l300_30014


namespace NUMINAMATH_CALUDE_union_A_complement_B_l300_30074

def I : Set ℤ := {x | |x| < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_A_complement_B : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l300_30074


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l300_30077

/-- Given a polynomial ax^4 + bx^3 + 45x^2 - 18x + 10 with a factor of 5x^2 - 3x + 2,
    prove that a = 151.25 and b = -98.25 -/
theorem polynomial_factor_coefficients :
  ∀ (a b : ℝ),
  (∃ (c d : ℝ), ∀ (x : ℝ),
    a * x^4 + b * x^3 + 45 * x^2 - 18 * x + 10 =
    (5 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 5)) →
  a = 151.25 ∧ b = -98.25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l300_30077


namespace NUMINAMATH_CALUDE_kangaroo_six_hops_l300_30035

def hop_distance (n : ℕ) : ℚ :=
  1 - (3/4)^n

theorem kangaroo_six_hops :
  hop_distance 6 = 3367 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_six_hops_l300_30035


namespace NUMINAMATH_CALUDE_weight_loss_percentage_l300_30045

def weight_before : ℝ := 840
def weight_after : ℝ := 546

theorem weight_loss_percentage : 
  (weight_before - weight_after) / weight_before * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_percentage_l300_30045


namespace NUMINAMATH_CALUDE_christmas_decorations_distribution_l300_30097

theorem christmas_decorations_distribution :
  let total_decorations : ℕ := 455
  let valid_student_count (n : ℕ) : Prop := 10 < n ∧ n < 70
  let valid_distribution (students : ℕ) (per_student : ℕ) : Prop :=
    valid_student_count students ∧
    students * per_student = total_decorations

  (∀ students per_student, valid_distribution students per_student →
    (students = 65 ∧ per_student = 7) ∨
    (students = 35 ∧ per_student = 13) ∨
    (students = 13 ∧ per_student = 35)) ∧
  (valid_distribution 65 7 ∧
   valid_distribution 35 13 ∧
   valid_distribution 13 35) :=
by sorry

end NUMINAMATH_CALUDE_christmas_decorations_distribution_l300_30097


namespace NUMINAMATH_CALUDE_root_product_of_quartic_l300_30006

theorem root_product_of_quartic (p q r s : ℂ) : 
  (3 * p^4 - 8 * p^3 - 15 * p^2 + 10 * p - 2 = 0) →
  (3 * q^4 - 8 * q^3 - 15 * q^2 + 10 * q - 2 = 0) →
  (3 * r^4 - 8 * r^3 - 15 * r^2 + 10 * r - 2 = 0) →
  (3 * s^4 - 8 * s^3 - 15 * s^2 + 10 * s - 2 = 0) →
  p * q * r * s = 2/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_of_quartic_l300_30006


namespace NUMINAMATH_CALUDE_lcm_12_18_30_l300_30088

theorem lcm_12_18_30 : Nat.lcm 12 (Nat.lcm 18 30) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_30_l300_30088


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l300_30038

-- Define the original line
def original_line (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

-- Define a point on the x-axis
def x_axis_point (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define symmetry with respect to x-axis
def symmetric_wrt_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- State the theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ),
  (∃ (x₀ : ℝ), original_line x₀ 0) →
  (∀ (p q : ℝ × ℝ),
    original_line p.1 p.2 →
    symmetric_wrt_x_axis p q →
    4 * q.1 + 3 * q.2 + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l300_30038


namespace NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l300_30046

theorem minimum_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2 : ℝ) = Real.sqrt (2^a * 2^b) → 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ) = Real.sqrt (2^x * 2^y) → 1/a + 1/b ≤ 1/x + 1/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (2 : ℝ) = Real.sqrt (2^x * 2^y) ∧ 1/a + 1/b = 1/x + 1/y) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l300_30046


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l300_30093

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.sqrt (1 + Real.log (1 + x^2 * Real.sin (1/x))) - 1
  else
    0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l300_30093


namespace NUMINAMATH_CALUDE_polynomial_common_factor_l300_30017

-- Define variables
variable (x y m n : ℝ)

-- Define the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4*x*(m-n) + 2*y*(m-n)^2

-- Define the common factor
def common_factor (m n : ℝ) : ℝ := 2*(m-n)

-- Theorem statement
theorem polynomial_common_factor :
  ∃ (a b : ℝ), polynomial x y m n = common_factor m n * (a*x + b*y*(m-n)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_common_factor_l300_30017


namespace NUMINAMATH_CALUDE_triangle_side_length_l300_30061

/-- Given a triangle ABC with side lengths a, b, and c opposite to angles A, B, and C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  B = π/3 →
  a^2 + c^2 = 3*a*c →
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l300_30061


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l300_30098

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 20 → F = 68 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l300_30098


namespace NUMINAMATH_CALUDE_money_division_l300_30055

/-- The problem of dividing money among A, B, and C -/
theorem money_division (a b c : ℚ) : 
  a + b + c = 720 →  -- Total amount is $720
  a = (1/3) * (b + c) →  -- A gets 1/3 of what B and C get
  b = (2/7) * (a + c) →  -- B gets 2/7 of what A and C get
  a > b →  -- A receives more than B
  a - b = 20 :=  -- Prove that A receives $20 more than B
by sorry

end NUMINAMATH_CALUDE_money_division_l300_30055


namespace NUMINAMATH_CALUDE_three_positions_from_eight_people_l300_30082

def number_of_people : ℕ := 8
def number_of_positions : ℕ := 3

theorem three_positions_from_eight_people :
  (number_of_people.factorial) / ((number_of_people - number_of_positions).factorial) = 336 :=
sorry

end NUMINAMATH_CALUDE_three_positions_from_eight_people_l300_30082


namespace NUMINAMATH_CALUDE_division_remainder_proof_l300_30008

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 265 →
  divisor = 22 →
  quotient = 12 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l300_30008


namespace NUMINAMATH_CALUDE_age_score_ratio_l300_30071

def almas_age : ℕ := 20
def melinas_age : ℕ := 60
def almas_score : ℕ := 40

theorem age_score_ratio :
  (almas_age + melinas_age) / almas_score = 2 ∧
  melinas_age = 3 * almas_age ∧
  melinas_age = 60 ∧
  almas_score = 40 := by
  sorry

end NUMINAMATH_CALUDE_age_score_ratio_l300_30071


namespace NUMINAMATH_CALUDE_area_between_tangents_and_curve_l300_30075

noncomputable section

-- Define the curve
def C (x : ℝ) : ℝ := 1 / x

-- Define the points P and Q
def P (a : ℝ) : ℝ × ℝ := (a, C a)
def Q (a : ℝ) : ℝ × ℝ := (2*a, C (2*a))

-- Define the tangent lines at P and Q
def l (a : ℝ) (x : ℝ) : ℝ := -1/(a^2) * x + 2/a
def m (a : ℝ) (x : ℝ) : ℝ := -1/(4*a^2) * x + 1/a

-- Define the area function
def area (a : ℝ) : ℝ :=
  ∫ x in a..(2*a), (C x - l a x) + (C x - m a x)

-- State the theorem
theorem area_between_tangents_and_curve (a : ℝ) (h : a > 0) :
  area a = 2 * Real.log 2 - 9/8 :=
sorry

end

end NUMINAMATH_CALUDE_area_between_tangents_and_curve_l300_30075


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l300_30032

theorem square_sum_lower_bound (x y : ℝ) (h : |x - 2*y| = 5) : x^2 + y^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l300_30032


namespace NUMINAMATH_CALUDE_total_wood_needed_l300_30092

def bench1_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

def bench2_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

def bench3_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

theorem total_wood_needed :
  let bench1 := bench1_wood 4 2 6 2
  let bench2 := bench2_wood 3 1.5 8 5
  let bench3 := bench3_wood 5 2.5 4 3
  bench1 + bench2 + bench3 = 87 := by sorry

end NUMINAMATH_CALUDE_total_wood_needed_l300_30092


namespace NUMINAMATH_CALUDE_sum_of_possible_DE_values_l300_30084

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A function that constructs the number D465E32 from digits D and E -/
def constructNumber (D E : ℕ) : ℕ := D * 100000 + 465000 + E * 100 + 32

/-- The theorem stating the sum of all possible values of D+E is 24 -/
theorem sum_of_possible_DE_values : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, ∃ D E : ℕ, isSingleDigit D ∧ isSingleDigit E ∧ 
      7 ∣ constructNumber D E ∧ n = D + E) ∧
    (∀ D E : ℕ, isSingleDigit D → isSingleDigit E → 
      7 ∣ constructNumber D E → D + E ∈ S) ∧
    S.sum id = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_DE_values_l300_30084


namespace NUMINAMATH_CALUDE_barn_paint_area_l300_30099

/-- Calculates the total area to be painted in a rectangular barn -/
def total_paint_area (width length height : ℝ) : ℝ :=
  2 * (2 * width * height + 2 * length * height) + 2 * width * length

/-- Theorem stating the total area to be painted for a specific barn -/
theorem barn_paint_area :
  let width : ℝ := 15
  let length : ℝ := 20
  let height : ℝ := 8
  total_paint_area width length height = 1720 := by
  sorry

end NUMINAMATH_CALUDE_barn_paint_area_l300_30099


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l300_30073

theorem five_topping_pizzas (n k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l300_30073


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l300_30011

theorem gcd_n_cube_plus_25_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l300_30011


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l300_30069

theorem point_on_unit_circle (s : ℝ) :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := 2*s / (s^2 + 1)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l300_30069


namespace NUMINAMATH_CALUDE_license_plate_difference_l300_30081

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible Sunland license plates -/
def sunland_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible Moonland license plates -/
def moonland_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of possible license plates between Sunland and Moonland -/
theorem license_plate_difference :
  sunland_plates - moonland_plates = 7321600 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_difference_l300_30081


namespace NUMINAMATH_CALUDE_archie_record_l300_30083

/-- The number of games in a season -/
def season_length : ℕ := 16

/-- Richard's average touchdowns per game in the first 14 games -/
def richard_average : ℕ := 6

/-- The number of games Richard has played so far -/
def games_played : ℕ := 14

/-- The number of remaining games -/
def remaining_games : ℕ := season_length - games_played

/-- The average number of touchdowns Richard needs in the remaining games to beat Archie's record -/
def needed_average : ℕ := 3

theorem archie_record :
  let richard_total := richard_average * games_played + needed_average * remaining_games
  richard_total - 1 = 89 := by sorry

end NUMINAMATH_CALUDE_archie_record_l300_30083


namespace NUMINAMATH_CALUDE_only_one_correct_statement_l300_30090

theorem only_one_correct_statement :
  (∃! n : Nat, n = 1 ∧
    (¬ (∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/b < 1/a)) ∧
    (¬ (∀ a b c : ℝ, a < b → a*c < b*c)) ∧
    ((∀ a b c : ℝ, a < b → a + c < b + c)) ∧
    (¬ (∀ a b : ℝ, a^2 < b^2 → a < b))) :=
by sorry

end NUMINAMATH_CALUDE_only_one_correct_statement_l300_30090


namespace NUMINAMATH_CALUDE_divisibility_condition_l300_30067

theorem divisibility_condition (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l300_30067


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l300_30025

/-- The value of k for which vectors a and b are perpendicular --/
theorem perpendicular_vectors (i j a b : ℝ × ℝ) (k : ℝ) : 
  i = (1, 0) →
  j = (0, 1) →
  a = (2 * i.1 + 0 * i.2, 0 * j.1 + 3 * j.2) →
  b = (k * i.1 + 0 * i.2, 0 * j.1 + (-4) * j.2) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l300_30025


namespace NUMINAMATH_CALUDE_larry_expression_equality_l300_30049

theorem larry_expression_equality (e : ℝ) : 
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := 4
  let d : ℝ := 5
  (a + (b - (c + (d + e)))) = (a + b - c - d - e) :=
by sorry

end NUMINAMATH_CALUDE_larry_expression_equality_l300_30049


namespace NUMINAMATH_CALUDE_equation_solutions_l300_30087

theorem equation_solutions :
  (∃ x : ℚ, x + 2 * (x - 3) = 3 * (1 - x) ∧ x = 3/2) ∧
  (∃ x : ℚ, 1 - (2*x - 1)/3 = (3 + x)/6 ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l300_30087


namespace NUMINAMATH_CALUDE_minimal_force_to_submerge_cube_l300_30031

-- Define constants
def cube_volume : Real := 10e-6  -- 10 cm³ in m³
def cube_density : Real := 500   -- kg/m³
def water_density : Real := 1000 -- kg/m³
def gravity : Real := 10         -- m/s²

-- Define the minimal force function
def minimal_force (v : Real) (ρ_cube : Real) (ρ_water : Real) (g : Real) : Real :=
  (ρ_water - ρ_cube) * v * g

-- Theorem statement
theorem minimal_force_to_submerge_cube :
  minimal_force cube_volume cube_density water_density gravity = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_minimal_force_to_submerge_cube_l300_30031


namespace NUMINAMATH_CALUDE_square_circle_octagon_l300_30059

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Square type
structure Square :=
  (a b c d : Point)

-- Define a Circle type
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define an Octagon type
structure Octagon :=
  (vertices : List Point)

-- Function to check if a square can be circumscribed by a circle
def can_circumscribe (s : Square) (c : Circle) : Prop :=
  sorry

-- Function to check if an octagon is regular and inscribed in a circle
def is_regular_inscribed_octagon (o : Octagon) (c : Circle) : Prop :=
  sorry

-- Main theorem
theorem square_circle_octagon (s : Square) :
  ∃ (c : Circle) (o : Octagon),
    can_circumscribe s c ∧ is_regular_inscribed_octagon o c :=
  sorry

end NUMINAMATH_CALUDE_square_circle_octagon_l300_30059


namespace NUMINAMATH_CALUDE_hyperbola_equation_l300_30027

/-- 
Given a hyperbola with equation (x²/a² - y²/b² = 1) that passes through the point (√2, √3) 
and has eccentricity 2, prove that its equation is x² - y²/3 = 1.
-/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 / a^2 - 3 / b^2 = 1) →  -- The hyperbola passes through (√2, √3)
  ((a^2 + b^2) / a^2 = 4) →  -- The eccentricity is 2
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l300_30027


namespace NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersection_points_l300_30063

/-- The number of interior intersection points in an n by n grid of squares -/
def interior_intersection_points (n : ℕ) : ℕ := (n - 1) * (n - 1)

/-- Theorem: The number of interior intersection points in a 12 by 12 grid of squares is 121 -/
theorem twelve_by_twelve_grid_intersection_points :
  interior_intersection_points 12 = 121 := by
  sorry

end NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersection_points_l300_30063


namespace NUMINAMATH_CALUDE_wednesday_bags_theorem_l300_30037

/-- Represents the leaf raking business of Bob and Johnny --/
structure LeafRakingBusiness where
  price_per_bag : ℕ
  monday_bags : ℕ
  tuesday_bags : ℕ
  total_money : ℕ

/-- Calculates the number of bags raked on Wednesday --/
def bags_raked_wednesday (business : LeafRakingBusiness) : ℕ :=
  (business.total_money - business.price_per_bag * (business.monday_bags + business.tuesday_bags)) / business.price_per_bag

/-- Theorem stating that given the conditions, the number of bags raked on Wednesday is 9 --/
theorem wednesday_bags_theorem (business : LeafRakingBusiness) 
  (h1 : business.price_per_bag = 4)
  (h2 : business.monday_bags = 5)
  (h3 : business.tuesday_bags = 3)
  (h4 : business.total_money = 68) :
  bags_raked_wednesday business = 9 := by
  sorry

#eval bags_raked_wednesday { price_per_bag := 4, monday_bags := 5, tuesday_bags := 3, total_money := 68 }

end NUMINAMATH_CALUDE_wednesday_bags_theorem_l300_30037


namespace NUMINAMATH_CALUDE_unique_third_shot_combination_l300_30042

/-- Represents the scores of a player's shots -/
structure PlayerScores :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- The set of all possible scores -/
def possibleScores : Finset ℕ := {10, 9, 9, 8, 8, 5, 4, 4, 3, 2}

/-- Conditions for the shooting problem -/
structure ShootingProblem :=
  (petya : PlayerScores)
  (vasya : PlayerScores)
  (first_three_equal : petya.first + petya.second + petya.third = vasya.first + vasya.second + vasya.third)
  (last_three_relation : petya.third + petya.fourth + petya.fifth = 3 * (vasya.third + vasya.fourth + vasya.fifth))
  (all_scores_valid : ∀ s, s ∈ [petya.first, petya.second, petya.third, petya.fourth, petya.fifth,
                               vasya.first, vasya.second, vasya.third, vasya.fourth, vasya.fifth] → s ∈ possibleScores)

/-- The theorem to be proved -/
theorem unique_third_shot_combination (problem : ShootingProblem) : 
  problem.petya.third = 10 ∧ problem.vasya.third = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_third_shot_combination_l300_30042


namespace NUMINAMATH_CALUDE_bennys_card_collection_l300_30057

theorem bennys_card_collection (original_cards : ℕ) (remaining_cards : ℕ) : 
  (remaining_cards = original_cards / 2) → (remaining_cards = 34) → (original_cards = 68) := by
  sorry

end NUMINAMATH_CALUDE_bennys_card_collection_l300_30057


namespace NUMINAMATH_CALUDE_badminton_survey_k_squared_l300_30054

/-- Represents the contingency table for the badminton survey --/
structure ContingencyTable :=
  (male_like : ℕ)
  (male_dislike : ℕ)
  (female_like : ℕ)
  (female_dislike : ℕ)

/-- Calculates the K² statistic for a given contingency table --/
def calculate_k_squared (table : ContingencyTable) : ℚ :=
  let n := table.male_like + table.male_dislike + table.female_like + table.female_dislike
  let a := table.male_like
  let b := table.male_dislike
  let c := table.female_like
  let d := table.female_dislike
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem badminton_survey_k_squared :
  ∀ (table : ContingencyTable),
    table.male_like + table.male_dislike = 100 →
    table.female_like + table.female_dislike = 100 →
    table.male_like = 40 →
    table.female_dislike = 90 →
    calculate_k_squared table = 24 := by
  sorry

end NUMINAMATH_CALUDE_badminton_survey_k_squared_l300_30054


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l300_30005

/-- Given a three-digit number ABC divisible by 37, prove that BCA + CAB is also divisible by 37 -/
theorem three_digit_divisibility (A B C : ℕ) (h : 37 ∣ (100 * A + 10 * B + C)) :
  37 ∣ ((100 * B + 10 * C + A) + (100 * C + 10 * A + B)) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l300_30005


namespace NUMINAMATH_CALUDE_evaluate_expression_l300_30024

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l300_30024


namespace NUMINAMATH_CALUDE_john_post_break_time_l300_30080

/-- The number of hours John danced before the break -/
def john_pre_break : ℝ := 3

/-- The number of hours John took for break -/
def john_break : ℝ := 1

/-- The total dancing time of both John and James (excluding John's break) -/
def total_dance_time : ℝ := 20

/-- The number of hours John danced after the break -/
def john_post_break : ℝ := 5

theorem john_post_break_time : 
  john_post_break = 
    (total_dance_time - john_pre_break - 
      (john_pre_break + john_break + john_post_break + 
        (1/3) * (john_pre_break + john_break + john_post_break))) / 
    (7/3) := by sorry

end NUMINAMATH_CALUDE_john_post_break_time_l300_30080


namespace NUMINAMATH_CALUDE_product_equality_l300_30062

/-- The product of 12, -0.5, 3/4, and 0.20 is equal to -9/10 -/
theorem product_equality : 12 * (-0.5) * (3/4 : ℚ) * 0.20 = -9/10 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l300_30062


namespace NUMINAMATH_CALUDE_principal_amount_satisfies_conditions_l300_30021

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := 6400

/-- The annual interest rate -/
def interest_rate : ℝ := 0.05

/-- The time period in years -/
def time_period : ℝ := 2

/-- The difference between compound interest and simple interest -/
def interest_difference : ℝ := 16

/-- Theorem stating that the principal amount satisfies the given conditions -/
theorem principal_amount_satisfies_conditions :
  let compound_interest := principal_amount * (1 + interest_rate) ^ time_period - principal_amount
  let simple_interest := principal_amount * interest_rate * time_period
  compound_interest - simple_interest = interest_difference :=
by sorry

end NUMINAMATH_CALUDE_principal_amount_satisfies_conditions_l300_30021


namespace NUMINAMATH_CALUDE_election_votes_l300_30022

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (excess_percent : ℚ) 
  (h_total : total_votes = 6720)
  (h_invalid : invalid_percent = 1/5)
  (h_excess : excess_percent = 3/20) :
  ∃ (votes_b : ℕ), votes_b = 2184 ∧ 
  (↑votes_b : ℚ) + (↑votes_b + excess_percent * total_votes) = (1 - invalid_percent) * total_votes :=
sorry

end NUMINAMATH_CALUDE_election_votes_l300_30022


namespace NUMINAMATH_CALUDE_expression_evaluation_l300_30070

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -1
  ((x - 2*y)^2 - (2*x + y)*(x - 4*y) - (-x + 3*y)*(x + 3*y)) / (-y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l300_30070


namespace NUMINAMATH_CALUDE_range_of_a_l300_30036

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, Real.exp x - a ≥ 0) → a ≤ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l300_30036


namespace NUMINAMATH_CALUDE_carbon_neutrality_time_l300_30001

/-- Carbon neutrality time calculation -/
theorem carbon_neutrality_time (a : ℝ) (b : ℝ) :
  a > 0 →
  a * b^7 = (4/5) * a →
  ∃ t : ℝ, t ≥ 42 ∧ a * b^t = (1/4) * a :=
by
  sorry

end NUMINAMATH_CALUDE_carbon_neutrality_time_l300_30001


namespace NUMINAMATH_CALUDE_segment_ratio_in_quadrilateral_l300_30058

/-- Given four distinct points on a plane with segment lengths a, a, a, b, b, and c,
    prove that the ratio of c to a is √3/2 -/
theorem segment_ratio_in_quadrilateral (a b c : ℝ) :
  (∃ A B C D : ℝ × ℝ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    dist A B = a ∧ dist A C = a ∧ dist B C = a ∧
    dist A D = b ∧ dist B D = b ∧ dist C D = c) →
  c / a = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_segment_ratio_in_quadrilateral_l300_30058


namespace NUMINAMATH_CALUDE_work_time_ratio_l300_30016

/-- Given two workers A and B, this theorem proves the ratio of their individual work times
    based on their combined work time and B's individual work time. -/
theorem work_time_ratio (time_together time_B : ℝ) (h1 : time_together = 4) (h2 : time_B = 24) :
  ∃ time_A : ℝ, time_A / time_B = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l300_30016


namespace NUMINAMATH_CALUDE_triangle_base_calculation_l300_30043

theorem triangle_base_calculation (base1 height1 height2 : ℝ) :
  base1 = 15 ∧ height1 = 12 ∧ height2 = 18 →
  ∃ base2 : ℝ, base2 = 20 ∧ base2 * height2 = 2 * (base1 * height1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_base_calculation_l300_30043


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l300_30096

theorem power_of_eight_sum_equals_power_of_two : ∃ x : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l300_30096


namespace NUMINAMATH_CALUDE_total_stamps_l300_30089

def stamps_problem (snowflake truck rose : ℕ) : Prop :=
  (snowflake = 11) ∧
  (truck = snowflake + 9) ∧
  (rose = truck - 13)

theorem total_stamps :
  ∀ snowflake truck rose : ℕ,
    stamps_problem snowflake truck rose →
    snowflake + truck + rose = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l300_30089


namespace NUMINAMATH_CALUDE_steve_height_after_growth_l300_30065

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Represents a person's height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Calculates the new height after growth -/
def new_height_after_growth (initial_height : Height) (growth_inches : ℕ) : ℕ :=
  feet_inches_to_inches initial_height.feet initial_height.inches + growth_inches

theorem steve_height_after_growth :
  let initial_height : Height := ⟨5, 6⟩
  let growth_inches : ℕ := 6
  new_height_after_growth initial_height growth_inches = 72 := by
  sorry

end NUMINAMATH_CALUDE_steve_height_after_growth_l300_30065


namespace NUMINAMATH_CALUDE_sin_plus_tan_10_deg_l300_30026

theorem sin_plus_tan_10_deg : 
  Real.sin (10 * π / 180) + (Real.sqrt 3 / 4) * Real.tan (10 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_tan_10_deg_l300_30026


namespace NUMINAMATH_CALUDE_work_completion_time_l300_30091

theorem work_completion_time (total_work : ℝ) (raja_rate : ℝ) (ram_rate : ℝ) :
  raja_rate + ram_rate = total_work / 4 →
  raja_rate = total_work / 12 →
  ram_rate = total_work / 6 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l300_30091


namespace NUMINAMATH_CALUDE_log_equation_solution_l300_30048

theorem log_equation_solution (y : ℝ) (h : y > 0) : 
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l300_30048


namespace NUMINAMATH_CALUDE_min_value_sin_cos_squared_l300_30000

theorem min_value_sin_cos_squared (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (m : ℝ), m = -1/9 ∧ ∀ z, Real.sin z + Real.sin (y + z - x) = 1/3 →
    m ≤ Real.sin z + (Real.cos z)^2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_squared_l300_30000


namespace NUMINAMATH_CALUDE_tom_balloons_l300_30079

theorem tom_balloons (initial_balloons : ℕ) (given_balloons : ℕ) : 
  initial_balloons = 30 → given_balloons = 16 → initial_balloons - given_balloons = 14 := by
  sorry

end NUMINAMATH_CALUDE_tom_balloons_l300_30079


namespace NUMINAMATH_CALUDE_tangent_lines_for_cubic_curve_l300_30012

def f (x : ℝ) := x^3 - x

theorem tangent_lines_for_cubic_curve :
  let C := f
  -- Tangent line at (2, f(2))
  ∃ (m b : ℝ), (∀ x y, y = m*x + b ↔ m*x - y + b = 0) ∧
               m = 3*2^2 - 1 ∧
               b = f 2 - m*2 ∧
               m*2 - f 2 + b = 0
  ∧
  -- Tangent lines parallel to y = 5x + 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
                 x₁^2 = 2 ∧ x₂^2 = 2 ∧
                 (∀ x y, y - f x₁ = 5*(x - x₁) ↔ 5*x - y - 4*Real.sqrt 2 = 0) ∧
                 (∀ x y, y - f x₂ = 5*(x - x₂) ↔ 5*x - y + 4*Real.sqrt 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_lines_for_cubic_curve_l300_30012


namespace NUMINAMATH_CALUDE_total_apples_eaten_l300_30003

def simone_daily_consumption : ℚ := 1/2
def simone_days : ℕ := 16
def lauri_daily_consumption : ℚ := 1/3
def lauri_days : ℕ := 15

theorem total_apples_eaten :
  simone_daily_consumption * simone_days + lauri_daily_consumption * lauri_days = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_eaten_l300_30003


namespace NUMINAMATH_CALUDE_polygon_perimeter_sum_l300_30002

theorem polygon_perimeter_sum (n : ℕ) (x y c : ℝ) : 
  n ≥ 3 →
  x = (2 * n : ℝ) * (Real.tan (π / (n : ℝ))) * (c / (2 * π)) →
  y = (2 * n : ℝ) * (Real.sin (π / (n : ℝ))) * (c / (2 * π)) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 → Real.tan θ ≥ θ) →
  x + y ≥ 2 * c :=
by sorry

end NUMINAMATH_CALUDE_polygon_perimeter_sum_l300_30002


namespace NUMINAMATH_CALUDE_board_tiling_divisibility_l300_30066

/-- Represents a square on the board -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a domino placement -/
inductive Domino
  | horizontal : Square → Domino
  | vertical : Square → Domino

/-- Represents a tiling of the board -/
def Tiling := List Domino

/-- Represents an assignment of integers to squares -/
def Assignment := Square → Int

/-- Checks if a tiling is valid for a 2n × 2n board -/
def is_valid_tiling (n : Nat) (t : Tiling) : Prop := sorry

/-- Checks if an assignment satisfies the neighbor condition -/
def satisfies_neighbor_condition (n : Nat) (red_tiling blue_tiling : Tiling) (assignment : Assignment) : Prop := sorry

theorem board_tiling_divisibility (n : Nat) 
  (red_tiling blue_tiling : Tiling) 
  (h_red_valid : is_valid_tiling n red_tiling)
  (h_blue_valid : is_valid_tiling n blue_tiling)
  (assignment : Assignment)
  (h_nonzero : ∀ s, assignment s ≠ 0)
  (h_satisfies : satisfies_neighbor_condition n red_tiling blue_tiling assignment) :
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_board_tiling_divisibility_l300_30066


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l300_30019

theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (4 * x₁^2 - 2 * x₁ = (1/4 : ℝ)) ∧ (4 * x₂^2 - 2 * x₂ = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l300_30019


namespace NUMINAMATH_CALUDE_andrew_jeffrey_walk_l300_30015

/-- Calculates the number of steps Andrew walks given Jeffrey's steps and their step ratio -/
def andrews_steps (jeffreys_steps : ℕ) (andrew_ratio jeffrey_ratio : ℕ) : ℕ :=
  (andrew_ratio * jeffreys_steps) / jeffrey_ratio

/-- Theorem stating that if Jeffrey walks 200 steps and the ratio of Andrew's to Jeffrey's steps is 3:4, then Andrew walks 150 steps -/
theorem andrew_jeffrey_walk :
  andrews_steps 200 3 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_andrew_jeffrey_walk_l300_30015


namespace NUMINAMATH_CALUDE_complex_number_parts_opposite_l300_30068

theorem complex_number_parts_opposite (b : ℝ) : 
  let z : ℂ := (2 - b * I) / (1 + 2 * I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_parts_opposite_l300_30068


namespace NUMINAMATH_CALUDE_line_translation_l300_30056

theorem line_translation (x : ℝ) : 
  let original_line := λ x : ℝ => x / 3
  let translated_line := λ x : ℝ => (x + 5) / 3
  translated_line x - original_line x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l300_30056


namespace NUMINAMATH_CALUDE_square_root_equality_implies_zero_product_l300_30086

theorem square_root_equality_implies_zero_product (x y z : ℝ) 
  (h : Real.sqrt (x - y + z) = Real.sqrt x - Real.sqrt y + Real.sqrt z) : 
  (x - y) * (y - z) * (z - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_implies_zero_product_l300_30086


namespace NUMINAMATH_CALUDE_bailey_towel_cost_l300_30047

def guest_sets : ℕ := 2
def master_sets : ℕ := 4
def guest_price : ℚ := 40
def master_price : ℚ := 50
def discount_rate : ℚ := 0.20

theorem bailey_towel_cost : 
  let total_before_discount := guest_sets * guest_price + master_sets * master_price
  let discount_amount := discount_rate * total_before_discount
  let final_cost := total_before_discount - discount_amount
  final_cost = 224 := by sorry

end NUMINAMATH_CALUDE_bailey_towel_cost_l300_30047
