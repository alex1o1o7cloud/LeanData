import Mathlib

namespace NUMINAMATH_CALUDE_average_weight_decrease_l138_13815

theorem average_weight_decrease (n : ℕ) (old_weight new_weight : ℝ) :
  n = 6 →
  old_weight = 80 →
  new_weight = 62 →
  (old_weight - new_weight) / n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l138_13815


namespace NUMINAMATH_CALUDE_ratio_is_zero_l138_13842

-- Define the rectangle JKLM
def Rectangle (J L M : ℝ × ℝ) : Prop :=
  (L.1 - J.1 = 8) ∧ (L.2 - M.2 = 6) ∧ (J.2 = L.2) ∧ (J.1 = M.1)

-- Define points N, P, Q
def PointN (J L N : ℝ × ℝ) : Prop :=
  N.2 = J.2 ∧ L.1 - N.1 = 2

def PointP (L M P : ℝ × ℝ) : Prop :=
  P.1 = L.1 ∧ M.2 - P.2 = 2

def PointQ (M J Q : ℝ × ℝ) : Prop :=
  Q.2 = J.2 ∧ M.1 - Q.1 = 3

-- Define the intersection points R and S
def IntersectionR (J P N Q R : ℝ × ℝ) : Prop :=
  (R.2 - J.2) / (R.1 - J.1) = (P.2 - J.2) / (P.1 - J.1) ∧
  R.2 = N.2

def IntersectionS (J M N Q S : ℝ × ℝ) : Prop :=
  (S.2 - J.2) / (S.1 - J.1) = (M.2 - J.2) / (M.1 - J.1) ∧
  S.2 = N.2

-- Theorem statement
theorem ratio_is_zero
  (J K L M N P Q R S : ℝ × ℝ)
  (h_rect : Rectangle J L M)
  (h_N : PointN J L N)
  (h_P : PointP L M P)
  (h_Q : PointQ M J Q)
  (h_R : IntersectionR J P N Q R)
  (h_S : IntersectionS J M N Q S) :
  (R.1 - S.1) / (N.1 - Q.1) = 0 :=
sorry

end NUMINAMATH_CALUDE_ratio_is_zero_l138_13842


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l138_13852

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l138_13852


namespace NUMINAMATH_CALUDE_mike_changed_tires_on_12_motorcycles_l138_13887

/-- The number of motorcycles Mike changed tires on -/
def num_motorcycles (total_tires num_cars tires_per_car tires_per_motorcycle : ℕ) : ℕ :=
  (total_tires - num_cars * tires_per_car) / tires_per_motorcycle

theorem mike_changed_tires_on_12_motorcycles :
  num_motorcycles 64 10 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_changed_tires_on_12_motorcycles_l138_13887


namespace NUMINAMATH_CALUDE_area_ABDE_l138_13881

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- The quadrilateral ABDE formed by four vertices of the regular hexagon -/
def ABDE (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {(2, 0), (1, Real.sqrt 3), (-2, 0), (-1, -Real.sqrt 3)}

/-- The area of a set of points in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of ABDE in a regular hexagon with side length 2 is 4√3 -/
theorem area_ABDE (h : RegularHexagon) : area (ABDE h) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ABDE_l138_13881


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l138_13825

theorem sum_of_absolute_roots (x : ℂ) : 
  x^4 - 6*x^3 + 13*x^2 - 12*x + 4 = 0 →
  ∃ r1 r2 r3 r4 : ℂ, 
    (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4) ∧
    (Complex.abs r1 + Complex.abs r2 + Complex.abs r3 + Complex.abs r4 = 2 * Real.sqrt 6 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l138_13825


namespace NUMINAMATH_CALUDE_no_odd_integer_solution_l138_13807

theorem no_odd_integer_solution :
  ¬∃ (x y z : ℤ), Odd x ∧ Odd y ∧ Odd z ∧ (x + y)^2 + (x + z)^2 = (y + z)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_odd_integer_solution_l138_13807


namespace NUMINAMATH_CALUDE_x_intercepts_count_l138_13872

theorem x_intercepts_count (x : ℝ) :
  (∃! x, (x - 5) * (x^2 + 6*x + 10) = 0) :=
sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l138_13872


namespace NUMINAMATH_CALUDE_yuna_position_l138_13837

/-- Given Eunji's position and Yuna's relative position after Eunji, 
    calculate Yuna's absolute position on the train. -/
theorem yuna_position (eunji_pos yuna_after : ℕ) : 
  eunji_pos = 100 → yuna_after = 11 → eunji_pos + yuna_after = 111 := by
  sorry

#check yuna_position

end NUMINAMATH_CALUDE_yuna_position_l138_13837


namespace NUMINAMATH_CALUDE_unique_arithmetic_triangle_l138_13858

/-- A triangle with integer angles in arithmetic progression -/
structure ArithmeticTriangle where
  a : ℕ
  d : ℕ
  sum_180 : a + (a + d) + (a + 2*d) = 180
  distinct : a ≠ a + d ∧ a ≠ a + 2*d ∧ a + d ≠ a + 2*d

/-- Theorem stating there's exactly one valid arithmetic triangle with possibly zero angle -/
theorem unique_arithmetic_triangle : 
  ∃! t : ArithmeticTriangle, t.a = 0 ∨ t.a + t.d = 0 ∨ t.a + 2*t.d = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_arithmetic_triangle_l138_13858


namespace NUMINAMATH_CALUDE_lunch_probability_l138_13821

def total_school_days : ℕ := 5

def ham_sandwich_days : ℕ := 3
def cake_days : ℕ := 1
def carrot_sticks_days : ℕ := 3

def prob_ham_sandwich : ℚ := ham_sandwich_days / total_school_days
def prob_cake : ℚ := cake_days / total_school_days
def prob_carrot_sticks : ℚ := carrot_sticks_days / total_school_days

theorem lunch_probability : 
  prob_ham_sandwich * prob_cake * prob_carrot_sticks = 3 / 125 := by
  sorry

end NUMINAMATH_CALUDE_lunch_probability_l138_13821


namespace NUMINAMATH_CALUDE_scale_drawing_conversion_l138_13895

/-- Proves that a 6.5 cm line segment in a scale drawing where 1 cm represents 250 meters
    is equivalent to 5332.125 feet, given that 1 meter equals approximately 3.281 feet. -/
theorem scale_drawing_conversion (scale : ℝ) (length : ℝ) (meter_to_feet : ℝ) :
  scale = 250 →
  length = 6.5 →
  meter_to_feet = 3.281 →
  length * scale * meter_to_feet = 5332.125 := by
sorry

end NUMINAMATH_CALUDE_scale_drawing_conversion_l138_13895


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l138_13859

theorem cube_sum_divisibility (x y z t : ℤ) (h : x^3 + y^3 = 3*(z^3 + t^3)) :
  3 ∣ x ∧ 3 ∣ y := by sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l138_13859


namespace NUMINAMATH_CALUDE_not_proportional_l138_13871

theorem not_proportional (x y : ℝ) : 
  (3 * x - y = 7 ∨ 4 * x + 3 * y = 13) → 
  ¬(∃ (k₁ k₂ : ℝ), (y = k₁ * x ∨ x * y = k₂)) :=
by sorry

end NUMINAMATH_CALUDE_not_proportional_l138_13871


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l138_13849

theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := (s^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l138_13849


namespace NUMINAMATH_CALUDE_linear_equation_values_l138_13822

/-- Given that x^(a-2) - 2y^(a-b+5) = 1 is a linear equation in x and y, prove that a = 3 and b = 7 -/
theorem linear_equation_values (a b : ℤ) : 
  (∀ x y : ℝ, ∃ m n c : ℝ, x^(a-2) - 2*y^(a-b+5) = m*x + n*y + c) → 
  a = 3 ∧ b = 7 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_values_l138_13822


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l138_13869

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ x_max) ∧ (-x_max^2 + 9*x_max - 18 ≥ 0) ∧ x_max = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l138_13869


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l138_13893

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 14

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := 6

/-- The total number of people who can ride the Ferris wheel at the same time -/
def total_people : ℕ := num_seats * people_per_seat

theorem ferris_wheel_capacity : total_people = 84 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l138_13893


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l138_13862

-- Define the displacement function
def s (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t^2

-- Theorem: The instantaneous velocity at t = 3 is 54
theorem instantaneous_velocity_at_3 : v 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l138_13862


namespace NUMINAMATH_CALUDE_exists_diff_same_num_prime_divisors_l138_13841

/-- The number of distinct prime divisors of a natural number -/
def numDistinctPrimeDivisors (n : ℕ) : ℕ := sorry

/-- For any natural number n, there exist natural numbers a and b such that
    n = a - b and they have the same number of distinct prime divisors -/
theorem exists_diff_same_num_prime_divisors (n : ℕ) :
  ∃ a b : ℕ, n = a - b ∧ numDistinctPrimeDivisors a = numDistinctPrimeDivisors b := by
  sorry

end NUMINAMATH_CALUDE_exists_diff_same_num_prime_divisors_l138_13841


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l138_13831

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l138_13831


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l138_13823

def sum_of_powers (x₁ x₂ x₃ x₄ x₅ : ℕ) : ℕ :=
  x₁^3 + x₂^5 + x₃^7 + x₄^9 + x₅^11

def representable (n : ℕ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ x₅ : ℕ, sum_of_powers x₁ x₂ x₃ x₄ x₅ = n

theorem infinitely_many_non_representable :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ n ∈ S, ¬representable n) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l138_13823


namespace NUMINAMATH_CALUDE_three_a_equals_30_l138_13802

theorem three_a_equals_30 
  (h1 : 3 * a - 2 * b - 2 * c = 30)
  (h2 : Real.sqrt (3 * a) - Real.sqrt (2 * b + 2 * c) = 4)
  (h3 : a + b + c = 10)
  : 3 * a = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_a_equals_30_l138_13802


namespace NUMINAMATH_CALUDE_f_2013_l138_13843

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 6) ≤ f (x + 2) + 4) ∧
  (∀ x : ℝ, f (x + 4) ≥ f (x + 2) + 2) ∧
  (f 1 = 1)

theorem f_2013 (f : ℝ → ℝ) (h : f_properties f) : f 2013 = 2013 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_l138_13843


namespace NUMINAMATH_CALUDE_units_digit_of_F_F10_l138_13809

-- Define the sequence F_n
def F : ℕ → ℕ
| 0 => 3
| 1 => 2
| (n + 2) => F (n + 1) + F n

-- Theorem statement
theorem units_digit_of_F_F10 : ∃ k : ℕ, F (F 10) = 10 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F10_l138_13809


namespace NUMINAMATH_CALUDE_sarah_trip_characteristics_l138_13819

/-- Represents a segment of Sarah's trip --/
structure TripSegment where
  start_time : ℝ
  end_time : ℝ
  start_distance : ℝ
  end_distance : ℝ
  pace : ℝ
  is_stationary : Bool

/-- Represents Sarah's entire trip --/
def SarahTrip : List TripSegment := sorry

/-- Checks if a trip has exactly two stationary periods --/
def has_two_stationary_periods (trip : List TripSegment) : Prop :=
  (trip.filter (λ seg => seg.is_stationary)).length = 2

/-- Checks if a trip has varying walking paces --/
def has_varying_paces (trip : List TripSegment) : Prop :=
  ∃ (seg1 seg2 : TripSegment), seg1 ∈ trip ∧ seg2 ∈ trip ∧ seg1.pace ≠ seg2.pace

/-- Theorem stating that Sarah's trip has two stationary periods and varying paces --/
theorem sarah_trip_characteristics :
  has_two_stationary_periods SarahTrip ∧ has_varying_paces SarahTrip := by sorry

end NUMINAMATH_CALUDE_sarah_trip_characteristics_l138_13819


namespace NUMINAMATH_CALUDE_min_value_of_function_l138_13868

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  ∃ y : ℝ, y = x + 4 / x ∧ y ≥ 4 ∧ (∀ z : ℝ, z = x + 4 / x → z ≥ y) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l138_13868


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l138_13857

/-- A function that checks if a number has only even digits -/
def hasOnlyEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

/-- The largest positive integer with only even digits that is less than 1000 and is a multiple of 9 -/
def largestEvenDigitMultipleOf9Under1000 : ℕ := 864

theorem largest_even_digit_multiple_of_9_under_1000 :
  (largestEvenDigitMultipleOf9Under1000 < 1000) ∧
  (largestEvenDigitMultipleOf9Under1000 % 9 = 0) ∧
  (hasOnlyEvenDigits largestEvenDigitMultipleOf9Under1000) ∧
  (∀ n : ℕ, n < 1000 → n % 9 = 0 → hasOnlyEvenDigits n → n ≤ largestEvenDigitMultipleOf9Under1000) :=
by sorry

#eval largestEvenDigitMultipleOf9Under1000

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l138_13857


namespace NUMINAMATH_CALUDE_calculation_proof_l138_13834

theorem calculation_proof :
  ((-1 : ℝ) ^ 2021 + |-(Real.sqrt 3)| + (8 : ℝ) ^ (1/3) - Real.sqrt 16 = -3 + Real.sqrt 3) ∧
  (-(1 : ℝ) ^ 2 - (27 : ℝ) ^ (1/3) + |1 - Real.sqrt 2| = -5 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l138_13834


namespace NUMINAMATH_CALUDE_quadratic_fit_coefficient_l138_13854

/-- Given three points (1, y₁), (2, y₂), and (3, y₃), the coefficient 'a' of the 
    quadratic equation y = ax² + bx + c that best fits these points is equal to 
    (y₃ - 2y₂ + y₁) / 2. -/
theorem quadratic_fit_coefficient (y₁ y₂ y₃ : ℝ) : 
  ∃ (a b c : ℝ), 
    (a * 1^2 + b * 1 + c = y₁) ∧ 
    (a * 2^2 + b * 2 + c = y₂) ∧ 
    (a * 3^2 + b * 3 + c = y₃) ∧ 
    (a = (y₃ - 2 * y₂ + y₁) / 2) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_fit_coefficient_l138_13854


namespace NUMINAMATH_CALUDE_lateral_surface_area_square_pyramid_l138_13889

/-- Lateral surface area of a regular square pyramid -/
theorem lateral_surface_area_square_pyramid 
  (base_edge : ℝ) 
  (height : ℝ) 
  (h : base_edge = 2 * Real.sqrt 3) 
  (h' : height = 1) : 
  4 * (1/2 * base_edge * Real.sqrt (base_edge^2/4 + height^2)) = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_square_pyramid_l138_13889


namespace NUMINAMATH_CALUDE_max_b_rectangular_prism_l138_13826

theorem max_b_rectangular_prism (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_rectangular_prism_l138_13826


namespace NUMINAMATH_CALUDE_vacation_savings_proof_l138_13875

/-- Calculates the amount to save per paycheck given a savings goal, time frame, and number of paychecks per month. -/
def amount_per_paycheck (savings_goal : ℚ) (months : ℕ) (paychecks_per_month : ℕ) : ℚ :=
  savings_goal / (months * paychecks_per_month)

/-- Proves that given a savings goal of $3,000.00 over 15 months with 2 paychecks per month, 
    the amount to save per paycheck is $100.00. -/
theorem vacation_savings_proof : 
  amount_per_paycheck 3000 15 2 = 100 := by
sorry

end NUMINAMATH_CALUDE_vacation_savings_proof_l138_13875


namespace NUMINAMATH_CALUDE_george_remaining_eggs_l138_13811

/-- Calculates the remaining number of eggs given the initial inventory and sold amount. -/
def remaining_eggs (cases : ℕ) (boxes_per_case : ℕ) (eggs_per_box : ℕ) (boxes_sold : ℕ) : ℕ :=
  cases * boxes_per_case * eggs_per_box - boxes_sold * eggs_per_box

/-- Proves that George has 648 eggs remaining after selling 3 boxes. -/
theorem george_remaining_eggs :
  remaining_eggs 7 12 8 3 = 648 := by
  sorry

end NUMINAMATH_CALUDE_george_remaining_eggs_l138_13811


namespace NUMINAMATH_CALUDE_zeros_of_quadratic_function_l138_13882

theorem zeros_of_quadratic_function (x : ℝ) :
  x^2 = 0 → x ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_of_quadratic_function_l138_13882


namespace NUMINAMATH_CALUDE_function_characterization_l138_13844

def is_positive_integer (n : ℕ) : Prop := n > 0

def satisfies_equation (f : ℕ → ℕ) : Prop :=
  ∀ n, is_positive_integer n → f (f n) + f n = 2 * n + 6

theorem function_characterization (f : ℕ → ℕ) :
  (∀ n, is_positive_integer n → is_positive_integer (f n)) →
  satisfies_equation f →
  ∀ n, is_positive_integer n → f n = n + 2 :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l138_13844


namespace NUMINAMATH_CALUDE_greatest_x_value_l138_13886

theorem greatest_x_value (x y : ℕ+) (a b : ℝ) : 
  a > 1 → 
  b > 1 → 
  a = 2.75 → 
  b = 4.26 → 
  ((a * x.val^2) / (y.val^3 : ℝ)) + b < 800000 → 
  Nat.Prime y.val → 
  Nat.gcd x.val y.val = 1 → 
  x.val + y.val = (x.val + y.val).min → 
  x.val ≤ 2801 ∧ ∃ (x' : ℕ+), x'.val = 2801 ∧ 
    ((a * x'.val^2) / (y.val^3 : ℝ)) + b < 800000 ∧
    Nat.gcd x'.val y.val = 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l138_13886


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l138_13864

theorem solve_cubic_equation (n : ℝ) :
  (n - 5) ^ 3 = (1 / 9)⁻¹ ↔ n = 5 + 3 ^ (2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l138_13864


namespace NUMINAMATH_CALUDE_common_solution_proof_l138_13851

theorem common_solution_proof :
  let x : ℝ := Real.rpow 2 (1/3) - 2
  let y : ℝ := Real.rpow 2 (2/3)
  (y = (x + 2)^2) ∧ (x * y + y = 2) := by
  sorry

end NUMINAMATH_CALUDE_common_solution_proof_l138_13851


namespace NUMINAMATH_CALUDE_jerry_spent_two_tickets_l138_13891

def tickets_spent (initial_tickets : ℕ) (won_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  initial_tickets + won_tickets - final_tickets

theorem jerry_spent_two_tickets :
  tickets_spent 4 47 49 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_spent_two_tickets_l138_13891


namespace NUMINAMATH_CALUDE_adams_students_l138_13866

/-- The number of students Adam teaches in 10 years -/
def students_in_ten_years (normal_students_per_year : ℕ) (first_year_students : ℕ) (total_years : ℕ) : ℕ :=
  first_year_students + (total_years - 1) * normal_students_per_year

/-- Theorem stating the total number of students Adam will teach in 10 years -/
theorem adams_students : 
  students_in_ten_years 50 40 10 = 490 := by
  sorry

end NUMINAMATH_CALUDE_adams_students_l138_13866


namespace NUMINAMATH_CALUDE_circle_area_increase_l138_13817

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l138_13817


namespace NUMINAMATH_CALUDE_arccos_cos_fifteen_l138_13874

theorem arccos_cos_fifteen (x : Real) (h : x = 15) :
  Real.arccos (Real.cos x) = x % (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_fifteen_l138_13874


namespace NUMINAMATH_CALUDE_sphere_enclosed_by_truncated_cone_l138_13838

theorem sphere_enclosed_by_truncated_cone (R r r' ζ : ℝ) 
  (h_positive : R > 0)
  (h_volume : (4/3) * π * R^3 * 2 = (4/3) * π * (r^2 + r * r' + r'^2) * R)
  (h_generator : (r + r')^2 = 4 * R^2 + (r - r')^2)
  (h_contact : ζ = (2 * r * r') / (r + r')) :
  r = (R/2) * (Real.sqrt 5 + 1) ∧ 
  r' = (R/2) * (Real.sqrt 5 - 1) ∧ 
  ζ = (2 * R * Real.sqrt 5) / 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_enclosed_by_truncated_cone_l138_13838


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l138_13861

theorem sqrt_product_simplification (p : ℝ) (hp : p ≥ 0) :
  Real.sqrt (40 * p^2) * Real.sqrt (10 * p^3) * Real.sqrt (8 * p^2) = 40 * p^3 * Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l138_13861


namespace NUMINAMATH_CALUDE_paragraph_writing_time_l138_13888

/-- Represents the time in minutes for various writing assignments -/
structure WritingTimes where
  short_answer : ℕ  -- Time for one short-answer question
  essay : ℕ         -- Time for one essay
  total : ℕ         -- Total homework time
  paragraph : ℕ     -- Time for one paragraph (to be proved)

/-- Represents the number of assignments -/
structure AssignmentCounts where
  essays : ℕ
  paragraphs : ℕ
  short_answers : ℕ

theorem paragraph_writing_time 
  (wt : WritingTimes) 
  (ac : AssignmentCounts) 
  (h1 : wt.short_answer = 3)
  (h2 : wt.essay = 60)
  (h3 : wt.total = 4 * 60)
  (h4 : ac.essays = 2)
  (h5 : ac.paragraphs = 5)
  (h6 : ac.short_answers = 15)
  (h7 : wt.total = ac.essays * wt.essay + ac.paragraphs * wt.paragraph + ac.short_answers * wt.short_answer) :
  wt.paragraph = 15 := by
  sorry

end NUMINAMATH_CALUDE_paragraph_writing_time_l138_13888


namespace NUMINAMATH_CALUDE_x_minus_y_value_l138_13897

theorem x_minus_y_value (x y : ℝ) (h1 : |x| = 2) (h2 : y^2 = 9) (h3 : x + y < 0) :
  x - y = 1 ∨ x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l138_13897


namespace NUMINAMATH_CALUDE_first_day_is_wednesday_l138_13885

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that the 22nd day of a month is a Wednesday, 
    prove that the 1st day of that month is also a Wednesday -/
theorem first_day_is_wednesday 
  (h : ∃ (m : List DayInMonth), 
    m.length = 31 ∧ 
    (∃ (d : DayInMonth), d ∈ m ∧ d.day = 22 ∧ d.dayOfWeek = DayOfWeek.Wednesday)) :
  ∃ (m : List DayInMonth),
    m.length = 31 ∧
    (∃ (d : DayInMonth), d ∈ m ∧ d.day = 1 ∧ d.dayOfWeek = DayOfWeek.Wednesday) :=
by sorry

end NUMINAMATH_CALUDE_first_day_is_wednesday_l138_13885


namespace NUMINAMATH_CALUDE_opposite_of_2023_l138_13856

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l138_13856


namespace NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l138_13824

/-- A regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- The dihedral angle between a lateral face and the base -/
  α : ℝ
  /-- The dihedral angle between two adjacent lateral faces -/
  β : ℝ

/-- Theorem: For a regular quadrilateral pyramid, 2 cos β + cos 2α = -1 -/
theorem regular_quad_pyramid_angle_relation (P : RegularQuadPyramid) : 
  2 * Real.cos P.β + Real.cos (2 * P.α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l138_13824


namespace NUMINAMATH_CALUDE_hannah_mug_collection_l138_13813

theorem hannah_mug_collection (total_mugs : ℕ) (num_colors : ℕ) (yellow_mugs : ℕ) :
  total_mugs = 40 →
  num_colors = 4 →
  yellow_mugs = 12 →
  let red_mugs := yellow_mugs / 2
  let blue_mugs := 3 * red_mugs
  total_mugs = blue_mugs + red_mugs + yellow_mugs + (total_mugs - (blue_mugs + red_mugs + yellow_mugs)) →
  (total_mugs - (blue_mugs + red_mugs + yellow_mugs)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_hannah_mug_collection_l138_13813


namespace NUMINAMATH_CALUDE_f_properties_l138_13808

-- Define the function f
def f (x : ℝ) : ℝ := -x - x^3

-- State the theorem
theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l138_13808


namespace NUMINAMATH_CALUDE_cosine_power_relation_l138_13803

theorem cosine_power_relation (z : ℂ) (α : ℝ) (h : z + 1/z = 2 * Real.cos α) :
  ∀ n : ℕ, z^n + 1/z^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_power_relation_l138_13803


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l138_13816

def systematicSampling (totalProducts : Nat) (sampleSize : Nat) (firstSample : Nat) : List Nat :=
  let interval := totalProducts / sampleSize
  List.range sampleSize |>.map (fun i => firstSample + i * interval)

theorem systematic_sampling_result :
  systematicSampling 60 5 5 = [5, 17, 29, 41, 53] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l138_13816


namespace NUMINAMATH_CALUDE_orchestra_overlap_l138_13810

theorem orchestra_overlap (total : ℕ) (violin : ℕ) (keyboard : ℕ) (neither : ℕ) : 
  total = 42 → violin = 25 → keyboard = 22 → neither = 3 →
  violin + keyboard - (total - neither) = 8 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_overlap_l138_13810


namespace NUMINAMATH_CALUDE_book_cost_calculation_l138_13832

theorem book_cost_calculation (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  num_books = 9 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / num_books = 7 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l138_13832


namespace NUMINAMATH_CALUDE_specific_project_time_l138_13827

/-- A project requires workers to complete it. The number of workers and time can change during the project. -/
structure Project where
  initial_workers : ℕ
  initial_days : ℕ
  additional_workers : ℕ
  days_before_addition : ℕ

/-- Calculate the total time required to complete the project -/
def total_time (p : Project) : ℕ :=
  sorry

/-- The specific project described in the problem -/
def specific_project : Project :=
  { initial_workers := 10
  , initial_days := 15
  , additional_workers := 5
  , days_before_addition := 5 }

/-- Theorem stating that the total time for the specific project is 6 days -/
theorem specific_project_time : total_time specific_project = 6 :=
  sorry

end NUMINAMATH_CALUDE_specific_project_time_l138_13827


namespace NUMINAMATH_CALUDE_equidistant_function_l138_13850

def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

theorem equidistant_function (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equidistant : ∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z - 1))
  (h_norm : Complex.abs (a + b * Complex.I) = 5) :
  b^2 = 99/4 := by
sorry

end NUMINAMATH_CALUDE_equidistant_function_l138_13850


namespace NUMINAMATH_CALUDE_airplane_passengers_l138_13829

theorem airplane_passengers (total_passengers men : ℕ) 
  (h1 : total_passengers = 80)
  (h2 : men = 30)
  (h3 : ∃ women : ℕ, men = women ∧ men + women + (total_passengers - (men + women)) = total_passengers) :
  total_passengers - 2 * men = 20 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l138_13829


namespace NUMINAMATH_CALUDE_derivative_of_constant_cosine_l138_13879

theorem derivative_of_constant_cosine (x : ℝ) : 
  deriv (λ _ : ℝ => Real.cos (π / 3)) x = 0 :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_constant_cosine_l138_13879


namespace NUMINAMATH_CALUDE_triangulation_count_l138_13873

/-- A triangulation of a non-self-intersecting n-gon using m interior vertices -/
structure Triangulation where
  n : ℕ  -- number of vertices in the n-gon
  m : ℕ  -- number of interior vertices
  N : ℕ  -- number of triangles in the triangulation
  h1 : N > 0  -- there is at least one triangle
  h2 : n ≥ 3  -- n-gon has at least 3 vertices

/-- The number of triangles in a triangulation of an n-gon with m interior vertices -/
theorem triangulation_count (T : Triangulation) : T.N = T.n + 2 * T.m - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangulation_count_l138_13873


namespace NUMINAMATH_CALUDE_sum_of_solutions_x_minus_4_squared_equals_16_l138_13840

theorem sum_of_solutions_x_minus_4_squared_equals_16 : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 4)^2 = 16 ∧ (x₂ - 4)^2 = 16 ∧ x₁ + x₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_x_minus_4_squared_equals_16_l138_13840


namespace NUMINAMATH_CALUDE_multiply_power_rule_l138_13839

theorem multiply_power_rule (x : ℝ) : x * x^4 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_power_rule_l138_13839


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l138_13878

/-- Given a geometric sequence {aₙ}, prove that a₃ + a₁₁ = 17 when a₇ = 4 and a₅ + a₉ = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * r) →  -- Geometric sequence definition
  a 7 = 4 →                         -- Given condition
  a 5 + a 9 = 10 →                  -- Given condition
  a 3 + a 11 = 17 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l138_13878


namespace NUMINAMATH_CALUDE_min_zeros_in_interval_l138_13846

theorem min_zeros_in_interval (f : ℝ → ℝ) 
  (h0 : f 0 = 0)
  (h1 : ∀ x, f (9 + x) = f (9 - x))
  (h2 : ∀ x, f (x - 10) = f (-x - 10)) :
  ∃ n : ℕ, n = 107 ∧ 
    (∀ m : ℕ, (∃ S : Finset ℝ, S.card = m ∧ 
      (∀ x ∈ S, x ∈ Set.Icc 0 2014 ∧ f x = 0)) → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_min_zeros_in_interval_l138_13846


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l138_13899

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 9
  let boat_breadth : ℝ := 3
  let boat_sinking : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000 -- kg/m³
  mass_of_man boat_length boat_breadth boat_sinking water_density = 270 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l138_13899


namespace NUMINAMATH_CALUDE_kite_long_diagonal_angle_in_circular_arrangement_l138_13845

/-- Represents a symmetrical kite in a circular arrangement -/
structure Kite where
  long_diagonal_angle : ℝ
  short_diagonal_angle : ℝ

/-- Represents a circular arrangement of kites -/
structure CircularArrangement where
  num_kites : ℕ
  kites : Fin num_kites → Kite
  covers_circle : Bool
  long_diagonals_meet_center : Bool

/-- The theorem stating the long diagonal angle in the specific arrangement -/
theorem kite_long_diagonal_angle_in_circular_arrangement 
  (arr : CircularArrangement) 
  (h1 : arr.num_kites = 10) 
  (h2 : arr.covers_circle = true) 
  (h3 : arr.long_diagonals_meet_center = true) :
  ∀ i, (arr.kites i).long_diagonal_angle = 162 :=
sorry

end NUMINAMATH_CALUDE_kite_long_diagonal_angle_in_circular_arrangement_l138_13845


namespace NUMINAMATH_CALUDE_total_earnings_l138_13848

def price_per_kg : ℝ := 0.50
def rooster1_weight : ℝ := 30
def rooster2_weight : ℝ := 40

theorem total_earnings :
  price_per_kg * rooster1_weight + price_per_kg * rooster2_weight = 35 := by
sorry

end NUMINAMATH_CALUDE_total_earnings_l138_13848


namespace NUMINAMATH_CALUDE_tangent_line_value_l138_13853

/-- The value of a when the line 2x - y + 1 = 0 is tangent to the curve y = ae^x + x -/
theorem tangent_line_value (a : ℝ) : 
  (∃ x y : ℝ, 2*x - y + 1 = 0 ∧ y = a*(Real.exp x) + x ∧ 
    (∀ h : ℝ, h ≠ 0 → (a*(Real.exp (x + h)) + (x + h) - y) / h ≠ 2)) → 
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_value_l138_13853


namespace NUMINAMATH_CALUDE_only_subtraction_negative_positive_l138_13818

theorem only_subtraction_negative_positive : 
  (1 + (-2) ≤ 0) ∧ 
  (1 - (-2) > 0) ∧ 
  (1 * (-2) ≤ 0) ∧ 
  (1 / (-2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_subtraction_negative_positive_l138_13818


namespace NUMINAMATH_CALUDE_amys_garden_space_l138_13894

/-- Calculates the total square feet of growing space for Amy's garden beds -/
theorem amys_garden_space : 
  let small_bed_length : ℝ := 3
  let small_bed_width : ℝ := 3
  let large_bed_length : ℝ := 4
  let large_bed_width : ℝ := 3
  let num_small_beds : ℕ := 2
  let num_large_beds : ℕ := 2
  
  let small_bed_area := small_bed_length * small_bed_width
  let large_bed_area := large_bed_length * large_bed_width
  let total_area := (num_small_beds : ℝ) * small_bed_area + (num_large_beds : ℝ) * large_bed_area
  
  total_area = 42 := by sorry

end NUMINAMATH_CALUDE_amys_garden_space_l138_13894


namespace NUMINAMATH_CALUDE_fraction_equality_l138_13870

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 20)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 8) :
  a / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l138_13870


namespace NUMINAMATH_CALUDE_event_B_more_likely_l138_13880

/-- Represents the number of sides on a fair die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 3

/-- Probability of event A: some number appears at least twice in three throws -/
def probA : ℚ := 4 / 9

/-- Probability of event B: three different numbers appear in three throws -/
def probB : ℚ := 5 / 9

/-- Theorem stating that event B is more likely than event A -/
theorem event_B_more_likely : probB > probA := by
  sorry

end NUMINAMATH_CALUDE_event_B_more_likely_l138_13880


namespace NUMINAMATH_CALUDE_optimal_gcd_l138_13812

/-- The number of integers to choose from (0 to 81 inclusive) -/
def n : ℕ := 82

/-- The set of numbers to choose from -/
def S : Finset ℕ := Finset.range n

/-- Amy's strategy: A function that takes the current state and returns Amy's choice -/
def amy_strategy : S → S → ℕ → ℕ := sorry

/-- Bob's strategy: A function that takes the current state and returns Bob's choice -/
def bob_strategy : S → S → ℕ → ℕ := sorry

/-- The sum of Amy's chosen numbers -/
def A (amy_nums : Finset ℕ) : ℕ := amy_nums.sum id

/-- The sum of Bob's chosen numbers -/
def B (bob_nums : Finset ℕ) : ℕ := bob_nums.sum id

/-- The game result when Amy and Bob play optimally -/
def optimal_play : Finset ℕ × Finset ℕ := sorry

/-- The theorem stating the optimal gcd when Amy and Bob play optimally -/
theorem optimal_gcd :
  let (amy_nums, bob_nums) := optimal_play
  Nat.gcd (A amy_nums) (B bob_nums) = 41 := by sorry

end NUMINAMATH_CALUDE_optimal_gcd_l138_13812


namespace NUMINAMATH_CALUDE_horse_lap_time_l138_13877

-- Define the given parameters
def field_area : Real := 625
def horse_speed : Real := 25

-- Define the theorem
theorem horse_lap_time : 
  ∀ (side_length perimeter time : Real),
  side_length^2 = field_area →
  perimeter = 4 * side_length →
  time = perimeter / horse_speed →
  time = 4 := by
  sorry

end NUMINAMATH_CALUDE_horse_lap_time_l138_13877


namespace NUMINAMATH_CALUDE_martha_has_115_cards_l138_13835

/-- The number of cards Martha has at the end of the transactions -/
def martha_final_cards : ℕ :=
  let initial_cards : ℕ := 3
  let cards_from_emily : ℕ := 25
  let cards_from_alex : ℕ := 43
  let cards_from_jenny : ℕ := 58
  let cards_given_to_sam : ℕ := 14
  initial_cards + cards_from_emily + cards_from_alex + cards_from_jenny - cards_given_to_sam

/-- Theorem stating that Martha ends up with 115 cards -/
theorem martha_has_115_cards : martha_final_cards = 115 := by
  sorry

end NUMINAMATH_CALUDE_martha_has_115_cards_l138_13835


namespace NUMINAMATH_CALUDE_total_chairs_l138_13863

/-- Represents the number of chairs of each color in a classroom. -/
structure Classroom where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Defines a classroom with the given conditions. -/
def classroom : Classroom where
  blue := 10
  green := 3 * 10
  white := (3 * 10 + 10) - 13

/-- Theorem stating the total number of chairs in the classroom. -/
theorem total_chairs : classroom.blue + classroom.green + classroom.white = 67 := by
  sorry

#eval classroom.blue + classroom.green + classroom.white

end NUMINAMATH_CALUDE_total_chairs_l138_13863


namespace NUMINAMATH_CALUDE_reflect_M_x_axis_l138_13855

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point M -/
def M : ℝ × ℝ := (3, -4)

/-- Theorem stating that reflecting M across the x-axis results in (3, 4) -/
theorem reflect_M_x_axis : reflect_x M = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_reflect_M_x_axis_l138_13855


namespace NUMINAMATH_CALUDE_divisibility_property_l138_13883

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def sequence_property (A : ℕ → ℕ) : Prop :=
  ∀ n k : ℕ, is_divisible (A (n + k) - A k) (A n)

def B (A : ℕ → ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => B A n * A (n + 1)

theorem divisibility_property (A : ℕ → ℕ) (h : sequence_property A) :
  ∀ n k : ℕ, is_divisible (B A (n + k)) ((B A n) * (B A k)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l138_13883


namespace NUMINAMATH_CALUDE_environmental_policy_survey_l138_13876

theorem environmental_policy_survey (group_a_size : ℕ) (group_b_size : ℕ) 
  (group_a_favor_percent : ℚ) (group_b_favor_percent : ℚ) : 
  group_a_size = 200 →
  group_b_size = 800 →
  group_a_favor_percent = 70 / 100 →
  group_b_favor_percent = 75 / 100 →
  (group_a_size * group_a_favor_percent + group_b_size * group_b_favor_percent) / 
  (group_a_size + group_b_size) = 74 / 100 := by
  sorry

end NUMINAMATH_CALUDE_environmental_policy_survey_l138_13876


namespace NUMINAMATH_CALUDE_player_arrangement_count_l138_13820

def num_players_alpha : ℕ := 4
def num_players_beta : ℕ := 4
def num_players_gamma : ℕ := 2
def total_players : ℕ := num_players_alpha + num_players_beta + num_players_gamma

theorem player_arrangement_count :
  (Nat.factorial 3) * (Nat.factorial num_players_alpha) * (Nat.factorial num_players_beta) * (Nat.factorial num_players_gamma) = 6912 :=
by sorry

end NUMINAMATH_CALUDE_player_arrangement_count_l138_13820


namespace NUMINAMATH_CALUDE_hyperbola_equivalence_l138_13836

theorem hyperbola_equivalence (x y : ℝ) :
  (4 * x^2 * y^2 = 4 * x * y + 3) ↔ (x * y = 3/2 ∨ x * y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equivalence_l138_13836


namespace NUMINAMATH_CALUDE_sin_difference_l138_13801

theorem sin_difference (A B : ℝ) : 
  Real.sin (A - B) = Real.sin A * Real.cos B - Real.cos A * Real.sin B := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_l138_13801


namespace NUMINAMATH_CALUDE_multiple_of_four_is_multiple_of_two_l138_13860

theorem multiple_of_four_is_multiple_of_two (n : ℕ) :
  (∀ k : ℕ, 4 ∣ k → 2 ∣ k) →
  4 ∣ n →
  2 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_four_is_multiple_of_two_l138_13860


namespace NUMINAMATH_CALUDE_largest_angle_is_right_angle_l138_13800

/-- Given a triangle with sides a, b, and c, if its area is (a+b+c)(a+b-c)/4, 
    then its largest angle is 90°. -/
theorem largest_angle_is_right_angle 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a + b + c) * (a + b - c) / 4 = (a + b + c) * (a + b - c) / 4) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) 
                                    (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
                                         (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_right_angle_l138_13800


namespace NUMINAMATH_CALUDE_power_multiplication_l138_13867

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l138_13867


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_l138_13805

theorem sqrt_two_times_sqrt_six : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_l138_13805


namespace NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l138_13865

theorem abs_one_point_five_minus_sqrt_two : |1.5 - Real.sqrt 2| = 1.5 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l138_13865


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l138_13833

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -20 + 15*I ∧ (4 + 3*I)^2 = -20 + 15*I →
  (-4 - 3*I)^2 = -20 + 15*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l138_13833


namespace NUMINAMATH_CALUDE_chord_length_for_max_distance_l138_13847

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define the chord AB
def Chord (A B : ℝ × ℝ) := A ∈ Circle ∧ B ∈ Circle

-- Define the semicircle ACB
def Semicircle (A B C : ℝ × ℝ) := 
  Chord A B ∧ 
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Define the point C as the farthest point on semicircle ACB from O
def FarthestPoint (O A B C : ℝ × ℝ) := 
  Semicircle A B C ∧
  ∀ D, Semicircle A B D → (C.1 - O.1)^2 + (C.2 - O.2)^2 ≥ (D.1 - O.1)^2 + (D.2 - O.2)^2

-- Define OC perpendicular to AB
def Perpendicular (O A B C : ℝ × ℝ) := 
  (C.1 - O.1) * (B.1 - A.1) + (C.2 - O.2) * (B.2 - A.2) = 0

-- The main theorem
theorem chord_length_for_max_distance (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  Chord A B →
  FarthestPoint O A B C →
  Perpendicular O A B C →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_for_max_distance_l138_13847


namespace NUMINAMATH_CALUDE_least_partition_size_formula_l138_13892

/-- A move that can be applied to a permutation -/
inductive Move
  | MedianFirst : Move  -- If a is the median, replace a,b,c with b,c,a
  | MedianLast : Move   -- If c is the median, replace a,b,c with c,a,b

/-- A permutation of 1, 2, 3, ..., n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of inversions in a permutation -/
def inversions (n : ℕ) (σ : Permutation n) : ℕ :=
  sorry

/-- Whether two permutations are obtainable from each other by a sequence of moves -/
def obtainable (n : ℕ) (σ τ : Permutation n) : Prop :=
  sorry

/-- The least number of sets in a partition of all n! permutations -/
def least_partition_size (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: the least number of sets in the partition is n^2 - 3n + 4 -/
theorem least_partition_size_formula (n : ℕ) :
  least_partition_size n = n^2 - 3*n + 4 :=
sorry

end NUMINAMATH_CALUDE_least_partition_size_formula_l138_13892


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l138_13890

theorem nested_sqrt_equality : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * (3 ^ (1/4)) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l138_13890


namespace NUMINAMATH_CALUDE_prime_equality_l138_13806

theorem prime_equality (p q r n : ℕ) : 
  Prime p → Prime q → Prime r → n > 0 →
  (∃ k₁ k₂ k₃ : ℕ, (p + n) = k₁ * q * r ∧ 
                   (q + n) = k₂ * r * p ∧ 
                   (r + n) = k₃ * p * q) →
  p = q ∧ q = r :=
sorry

end NUMINAMATH_CALUDE_prime_equality_l138_13806


namespace NUMINAMATH_CALUDE_power_six_expression_l138_13830

theorem power_six_expression (m n : ℕ) (P Q : ℕ) 
  (h1 : P = 2^m) (h2 : Q = 5^n) : 
  6^(m+n) = P * 2^n * 3^(m+n) := by
  sorry

end NUMINAMATH_CALUDE_power_six_expression_l138_13830


namespace NUMINAMATH_CALUDE_root_sum_cubes_l138_13898

theorem root_sum_cubes (a b c d : ℝ) : 
  (3 * a^4 + 6 * a^3 + 1002 * a^2 + 2005 * a + 4010 = 0) →
  (3 * b^4 + 6 * b^3 + 1002 * b^2 + 2005 * b + 4010 = 0) →
  (3 * c^4 + 6 * c^3 + 1002 * c^2 + 2005 * c + 4010 = 0) →
  (3 * d^4 + 6 * d^3 + 1002 * d^2 + 2005 * d + 4010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l138_13898


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l138_13828

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- Points A and B on the ellipse -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_property :
  ellipse_eq A.1 A.2 ∧ 
  ellipse_eq B.1 B.2 ∧ 
  (∃ (t : ℝ), A = F₂ + t • (B - F₂) ∨ B = F₂ + t • (A - F₂)) ∧
  distance A B = 5 →
  distance A F₁ + distance B F₁ = 11 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l138_13828


namespace NUMINAMATH_CALUDE_closest_to_370_l138_13896

def calculation : ℝ := 3.1 * 9.1 * (5.92 + 4.08) + 100

def options : List ℝ := [300, 350, 370, 400, 430]

theorem closest_to_370 : 
  ∀ x ∈ options, |calculation - 370| ≤ |calculation - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_370_l138_13896


namespace NUMINAMATH_CALUDE_julia_tag_game_l138_13814

theorem julia_tag_game (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 18) 
  (h2 : monday = 4) 
  (h3 : total = monday + tuesday) : 
  tuesday = 14 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l138_13814


namespace NUMINAMATH_CALUDE_girls_in_class_l138_13884

theorem girls_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 70 → 
  4 * boys = 3 * girls → 
  total = girls + boys → 
  girls = 40 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l138_13884


namespace NUMINAMATH_CALUDE_height_estimate_correct_l138_13804

/-- Represents the regression line for student height based on foot length -/
structure HeightRegression where
  n : ℕ              -- number of students in the sample
  sum_x : ℝ          -- sum of foot lengths
  sum_y : ℝ          -- sum of heights
  slope : ℝ          -- slope of the regression line
  intercept : ℝ      -- y-intercept of the regression line

/-- Calculates the estimated height for a given foot length -/
def estimate_height (reg : HeightRegression) (x : ℝ) : ℝ :=
  reg.slope * x + reg.intercept

/-- Theorem stating that the estimated height for a foot length of 24 cm is 166 cm -/
theorem height_estimate_correct (reg : HeightRegression) : 
  reg.n = 10 ∧ 
  reg.sum_x = 225 ∧ 
  reg.sum_y = 1600 ∧ 
  reg.slope = 4 ∧
  reg.intercept = reg.sum_y / reg.n - reg.slope * (reg.sum_x / reg.n) →
  estimate_height reg 24 = 166 := by
  sorry

end NUMINAMATH_CALUDE_height_estimate_correct_l138_13804
