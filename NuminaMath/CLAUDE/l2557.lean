import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_l2557_255748

theorem fraction_equality : (5 * 7 + 3) / (3 * 5) = 38 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2557_255748


namespace NUMINAMATH_CALUDE_equation_solution_l2557_255732

theorem equation_solution : ∃ y : ℝ, y > 0 ∧ 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4) ∧ y = 1296 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2557_255732


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2557_255714

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2557_255714


namespace NUMINAMATH_CALUDE_estimate_larger_than_actual_l2557_255798

theorem estimate_larger_than_actual (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  ⌈x⌉ - ⌊y⌋ > x - y :=
sorry

end NUMINAMATH_CALUDE_estimate_larger_than_actual_l2557_255798


namespace NUMINAMATH_CALUDE_max_pages_for_25_dollars_l2557_255705

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the total amount available in cents
def total_amount : ℕ := 2500

-- Define the function to calculate the maximum number of pages
def max_pages (cost : ℕ) (total : ℕ) : ℕ := 
  (total / cost : ℕ)

-- Theorem statement
theorem max_pages_for_25_dollars : 
  max_pages cost_per_page total_amount = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_25_dollars_l2557_255705


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2557_255719

-- Define the item price
def item_price : ℝ := 50

-- Define the tax rates
def tax_rate_high : ℝ := 0.075
def tax_rate_low : ℝ := 0.07

-- Define the sales tax calculation function
def sales_tax (price : ℝ) (rate : ℝ) : ℝ := price * rate

-- Theorem statement
theorem sales_tax_difference :
  sales_tax item_price tax_rate_high - sales_tax item_price tax_rate_low = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_sales_tax_difference_l2557_255719


namespace NUMINAMATH_CALUDE_sqrt_6_plus_sqrt_6_equals_3_l2557_255736

theorem sqrt_6_plus_sqrt_6_equals_3 :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (6 + x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_plus_sqrt_6_equals_3_l2557_255736


namespace NUMINAMATH_CALUDE_circumradius_of_special_triangle_l2557_255725

/-- The radius of the circumcircle of a triangle with sides 8, 15, and 17 is 17/2 -/
theorem circumradius_of_special_triangle :
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := 17
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (area * 4) / (a * b * c) = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_special_triangle_l2557_255725


namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_is_one_l2557_255741

/-- Given a man's swimming speed and the relative time to swim upstream vs downstream, 
    calculate the speed of the stream. -/
theorem stream_speed (mans_speed : ℝ) (upstream_time_ratio : ℝ) : ℝ :=
  let stream_speed := (mans_speed * (upstream_time_ratio - 1)) / (upstream_time_ratio + 1)
  stream_speed

/-- Prove that given the conditions, the stream speed is 1 km/h -/
theorem stream_speed_is_one :
  stream_speed 3 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_is_one_l2557_255741


namespace NUMINAMATH_CALUDE_quotient_minus_fraction_number_plus_half_l2557_255739

-- Question 1
theorem quotient_minus_fraction (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (c / a) / (c / b) - c = c * (b - a) / a / b :=
sorry

-- Question 2
theorem number_plus_half (x : ℚ) :
  x + (1/2) * x = 12/5 ↔ x = (12/5 - 1/2) / (3/2) :=
sorry

end NUMINAMATH_CALUDE_quotient_minus_fraction_number_plus_half_l2557_255739


namespace NUMINAMATH_CALUDE_train_length_calculation_l2557_255766

-- Define the given parameters
def bridge_length : ℝ := 120
def crossing_time : ℝ := 20
def train_speed : ℝ := 66.6

-- State the theorem
theorem train_length_calculation :
  let total_distance := train_speed * crossing_time
  let train_length := total_distance - bridge_length
  train_length = 1212 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2557_255766


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2557_255756

theorem complex_modulus_equation (m : ℝ) (h1 : m > 0) :
  Complex.abs (5 + m * Complex.I) = 5 * Real.sqrt 26 → m = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2557_255756


namespace NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l2557_255793

theorem unique_solution_implies_negative_a (a : ℝ) :
  (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) → a < 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l2557_255793


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_congruences_l2557_255733

theorem smallest_integer_satisfying_congruences : ∃ b : ℕ, b > 0 ∧
  b % 3 = 2 ∧
  b % 4 = 3 ∧
  b % 5 = 4 ∧
  b % 6 = 5 ∧
  ∀ k : ℕ, k > 0 ∧ k % 3 = 2 ∧ k % 4 = 3 ∧ k % 5 = 4 ∧ k % 6 = 5 → k ≥ b :=
by
  -- Proof goes here
  sorry

#eval 59 % 3  -- Should output 2
#eval 59 % 4  -- Should output 3
#eval 59 % 5  -- Should output 4
#eval 59 % 6  -- Should output 5

end NUMINAMATH_CALUDE_smallest_integer_satisfying_congruences_l2557_255733


namespace NUMINAMATH_CALUDE_not_certain_rain_beijing_no_rain_shanghai_l2557_255721

-- Define the probabilities of rainfall
def probability_rain_beijing : ℝ := 0.8
def probability_rain_shanghai : ℝ := 0.2

-- Theorem to prove
theorem not_certain_rain_beijing_no_rain_shanghai :
  ¬(probability_rain_beijing = 1 ∧ probability_rain_shanghai = 0) :=
sorry

end NUMINAMATH_CALUDE_not_certain_rain_beijing_no_rain_shanghai_l2557_255721


namespace NUMINAMATH_CALUDE_angle_properties_l2557_255755

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-3/5, -4/5), prove properties about α and β. -/
theorem angle_properties (α β : Real) : 
  (∃ (P : Real × Real), P.1 = -3/5 ∧ P.2 = -4/5 ∧ 
   Real.cos α = -3/5 ∧ Real.sin α = -4/5) →
  Real.sin (α + π) = 4/5 ∧
  (Real.sin (α + β) = 5/13 → Real.cos β = -56/65 ∨ Real.cos β = 16/65) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l2557_255755


namespace NUMINAMATH_CALUDE_points_to_office_theorem_l2557_255706

/-- The number of points needed to be sent to the office -/
def points_to_office : ℕ := 100

/-- Points for interrupting -/
def interrupt_points : ℕ := 5

/-- Points for insulting classmates -/
def insult_points : ℕ := 10

/-- Points for throwing things -/
def throw_points : ℕ := 25

/-- Number of times Jerry interrupted -/
def jerry_interrupts : ℕ := 2

/-- Number of times Jerry insulted classmates -/
def jerry_insults : ℕ := 4

/-- Number of times Jerry can throw things before being sent to office -/
def jerry_throws_left : ℕ := 2

/-- Theorem stating the number of points needed to be sent to the office -/
theorem points_to_office_theorem :
  points_to_office = 
    jerry_interrupts * interrupt_points +
    jerry_insults * insult_points +
    jerry_throws_left * throw_points :=
by sorry

end NUMINAMATH_CALUDE_points_to_office_theorem_l2557_255706


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2557_255770

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 5) : 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 5 → 
    9/x + 16/y + 25/z ≤ 9/a + 16/b + 25/c) ∧
  9/x + 16/y + 25/z = 28.8 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2557_255770


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l2557_255744

theorem new_ratio_after_addition : 
  ∀ (x y : ℤ), 
    x * 4 = y →  -- The two integers are in the ratio of 1 to 4
    y = 48 →     -- The larger integer is 48
    (x + 12) * 2 = y  -- The new ratio after adding 12 to the smaller integer is 1:2
    := by sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l2557_255744


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l2557_255715

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else x^2 - 1

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l2557_255715


namespace NUMINAMATH_CALUDE_bike_rental_cost_l2557_255763

theorem bike_rental_cost 
  (daily_rate : ℝ) 
  (mileage_rate : ℝ) 
  (rental_days : ℕ) 
  (miles_biked : ℝ) 
  (h1 : daily_rate = 15)
  (h2 : mileage_rate = 0.1)
  (h3 : rental_days = 3)
  (h4 : miles_biked = 300) :
  daily_rate * ↑rental_days + mileage_rate * miles_biked = 75 :=
by sorry

end NUMINAMATH_CALUDE_bike_rental_cost_l2557_255763


namespace NUMINAMATH_CALUDE_lucky_years_2020_to_2024_l2557_255757

def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem lucky_years_2020_to_2024 :
  isLuckyYear 2020 ∧
  isLuckyYear 2021 ∧
  isLuckyYear 2022 ∧
  ¬isLuckyYear 2023 ∧
  isLuckyYear 2024 := by
  sorry

end NUMINAMATH_CALUDE_lucky_years_2020_to_2024_l2557_255757


namespace NUMINAMATH_CALUDE_solve_equation_l2557_255731

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2557_255731


namespace NUMINAMATH_CALUDE_line_equation_l2557_255750

/-- Given a line passing through (a, 0) and cutting a triangular region
    with area T in the first quadrant, prove its equation. -/
theorem line_equation (a T : ℝ) (h₁ : a ≠ 0) (h₂ : T > 0) :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = a ∧ y = 0) ∨ (x = 0 ∧ y > 0) →
    (y = m * x + b ↔ a^2 * y + 2 * T * x - 2 * a * T = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2557_255750


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l2557_255712

theorem sqrt_plus_square_zero_implies_diff (x y : ℝ) :
  Real.sqrt (y - 3) + (2 * x - 4)^2 = 0 → 2 * x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l2557_255712


namespace NUMINAMATH_CALUDE_bus_max_capacity_l2557_255711

/-- Represents the seating capacity of a bus with specific arrangements -/
structure BusCapacity where
  left_regular : Nat
  left_priority : Nat
  right_regular : Nat
  right_priority : Nat
  back_row : Nat
  standing : Nat
  regular_capacity : Nat
  priority_capacity : Nat

/-- Calculates the total capacity of the bus -/
def total_capacity (bus : BusCapacity) : Nat :=
  bus.left_regular * bus.regular_capacity +
  bus.left_priority * bus.priority_capacity +
  bus.right_regular * bus.regular_capacity +
  bus.right_priority * bus.priority_capacity +
  bus.back_row +
  bus.standing

/-- Theorem stating that the maximum capacity of the bus is 94 -/
theorem bus_max_capacity :
  ∀ (bus : BusCapacity),
    bus.left_regular = 12 →
    bus.left_priority = 3 →
    bus.right_regular = 9 →
    bus.right_priority = 2 →
    bus.back_row = 7 →
    bus.standing = 14 →
    bus.regular_capacity = 3 →
    bus.priority_capacity = 2 →
    total_capacity bus = 94 := by
  sorry


end NUMINAMATH_CALUDE_bus_max_capacity_l2557_255711


namespace NUMINAMATH_CALUDE_girls_in_class_l2557_255703

theorem girls_in_class (total : Nat) (prob : Rat) : 
  total = 25 → 
  prob = 3/25 → 
  (fun n : Nat => n * (n - 1) = prob * (total * (total - 1))) 9 → 
  total - 9 = 16 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_class_l2557_255703


namespace NUMINAMATH_CALUDE_sequence_general_term_l2557_255724

theorem sequence_general_term (n : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ k, S k = 2 * k^2 - 3 * k) →
  (∀ k, k ≥ 1 → a k = S k - S (k - 1)) →
  (∀ k, a k = 4 * k - 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2557_255724


namespace NUMINAMATH_CALUDE_range_of_m_l2557_255746

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - m)^x < (3 - m)^y

-- Define the theorem
theorem range_of_m : 
  ∃ m_min m_max : ℝ, 
    (m_min = 1 ∧ m_max = 2) ∧ 
    (∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (m_min ≤ m ∧ m < m_max)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2557_255746


namespace NUMINAMATH_CALUDE_divisible_by_38_count_l2557_255716

def numbers : List Nat := [3624, 36024, 360924, 3609924, 36099924, 360999924, 3609999924]

theorem divisible_by_38_count :
  (numbers.filter (·.mod 38 = 0)).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_38_count_l2557_255716


namespace NUMINAMATH_CALUDE_correct_product_l2557_255778

theorem correct_product (a b : ℚ) (a_int b_int : ℕ) (result : ℕ) : 
  a = 0.125 →
  b = 5.12 →
  a_int = 125 →
  b_int = 512 →
  result = 64000 →
  a_int * b_int = result →
  a * b = 0.64 := by
sorry

end NUMINAMATH_CALUDE_correct_product_l2557_255778


namespace NUMINAMATH_CALUDE_license_advantages_18_vs_30_l2557_255729

/-- Represents the age at which a person gets a driver's license -/
inductive LicenseAge
| Age18 : LicenseAge
| Age30 : LicenseAge

/-- Represents the advantages of getting a driver's license -/
structure LicenseAdvantages where
  insuranceCostSavings : Bool
  rentalCarFlexibility : Bool
  employmentOpportunities : Bool

/-- Theorem stating that getting a license at 18 has more advantages than at 30 -/
theorem license_advantages_18_vs_30 :
  ∃ (adv18 adv30 : LicenseAdvantages),
    (adv18.insuranceCostSavings = true ∧
     adv18.rentalCarFlexibility = true ∧
     adv18.employmentOpportunities = true) ∧
    (adv30.insuranceCostSavings = false ∨
     adv30.rentalCarFlexibility = false ∨
     adv30.employmentOpportunities = false) :=
by sorry

end NUMINAMATH_CALUDE_license_advantages_18_vs_30_l2557_255729


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l2557_255701

/-- p is a sufficient but not necessary condition for q -/
def is_sufficient_not_necessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

theorem sufficient_condition_range (a : ℝ) :
  is_sufficient_not_necessary (∀ x : ℝ, 4 - x ≤ 6) (∀ x : ℝ, x > a - 1) →
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l2557_255701


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l2557_255787

-- Define the condition for factoring quadratic trinomials
def is_factorizable (p q m n : ℤ) : Prop :=
  q = m * n ∧ p = m + n

-- Theorem 1
theorem factorization_1 : ∀ x : ℤ, x^2 - 7*x + 12 = (x - 3) * (x - 4) :=
  sorry

-- Theorem 2
theorem factorization_2 : ∀ x y : ℤ, (x - y)^2 + 4*(x - y) + 3 = (x - y + 1) * (x - y + 3) :=
  sorry

-- Theorem 3
theorem factorization_3 : ∀ a b : ℤ, (a + b) * (a + b - 2) - 3 = (a + b - 3) * (a + b + 1) :=
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l2557_255787


namespace NUMINAMATH_CALUDE_unscreened_percentage_l2557_255702

/-- Calculates the percentage of unscreened part of a TV -/
theorem unscreened_percentage (tv_length tv_width screen_length screen_width : ℕ) 
  (h1 : tv_length = 6) (h2 : tv_width = 5) (h3 : screen_length = 5) (h4 : screen_width = 4) :
  (1 : ℚ) / 3 * 100 = 
    (tv_length * tv_width - screen_length * screen_width : ℚ) / (tv_length * tv_width) * 100 := by
  sorry

end NUMINAMATH_CALUDE_unscreened_percentage_l2557_255702


namespace NUMINAMATH_CALUDE_prime_power_of_two_l2557_255722

theorem prime_power_of_two (n : ℕ) : 
  Prime (2^n + 1) → ∃ k : ℕ, n = 2^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_of_two_l2557_255722


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2557_255728

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250) 
  (h2 : bridge_length = 150) 
  (h3 : crossing_time = 20) : 
  (train_length + bridge_length) / crossing_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2557_255728


namespace NUMINAMATH_CALUDE_only_A_has_zero_constant_term_l2557_255747

def equation_A (x : ℝ) : ℝ := x^2 + x
def equation_B (x : ℝ) : ℝ := 2*x^2 - x - 12
def equation_C (x : ℝ) : ℝ := 2*(x^2 - 1) - 3*(x - 1)
def equation_D (x : ℝ) : ℝ := 2*(x^2 + 1) - (x + 4)

def has_zero_constant_term (f : ℝ → ℝ) : Prop := f 0 = 0

theorem only_A_has_zero_constant_term :
  has_zero_constant_term equation_A ∧
  ¬has_zero_constant_term equation_B ∧
  ¬has_zero_constant_term equation_C ∧
  ¬has_zero_constant_term equation_D :=
by sorry

end NUMINAMATH_CALUDE_only_A_has_zero_constant_term_l2557_255747


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l2557_255772

/- Define the total number of cards -/
def total_cards : ℕ := 50

/- Define the number of different numbers on the cards -/
def distinct_numbers : ℕ := 10

/- Define the number of cards for each number -/
def cards_per_number : ℕ := 5

/- Define the number of cards drawn -/
def cards_drawn : ℕ := 5

/- Function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/- Probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / choose total_cards cards_drawn

/- Probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (2250 : ℚ) / choose total_cards cards_drawn

/- Theorem stating the ratio of q to p -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l2557_255772


namespace NUMINAMATH_CALUDE_fencing_match_prob_increase_correct_l2557_255760

def fencing_match_prob_increase 
  (k l : ℕ) 
  (hk : k < 15) 
  (hl : l < 15) 
  (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  Nat.choose (k + l) k * p^k * (1 - p)^(l + 1)

theorem fencing_match_prob_increase_correct 
  (k l : ℕ) 
  (hk : k < 15) 
  (hl : l < 15) 
  (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) :
  fencing_match_prob_increase k l hk hl p hp = 
    Nat.choose (k + l) k * p^k * (1 - p)^(l + 1) := by
  sorry

end NUMINAMATH_CALUDE_fencing_match_prob_increase_correct_l2557_255760


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2557_255776

theorem arithmetic_calculation : 6 / (-3) + 2^2 * (1 - 4) = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2557_255776


namespace NUMINAMATH_CALUDE_vector_equality_vector_equation_solution_l2557_255704

/-- Two vectors in ℝ² are equal if their corresponding components are equal -/
theorem vector_equality (a b c d : ℝ) : (a, b) = (c, d) ↔ a = c ∧ b = d := by sorry

/-- Definition of Vector1 -/
def Vector1 (u : ℝ) : ℝ × ℝ := (3 + 5*u, -1 - 3*u)

/-- Definition of Vector2 -/
def Vector2 (v : ℝ) : ℝ × ℝ := (0 - 3*v, 2 + 4*v)

theorem vector_equation_solution :
  ∃ (u v : ℝ), Vector1 u = Vector2 v ∧ u = -3/11 ∧ v = -16/11 := by sorry

end NUMINAMATH_CALUDE_vector_equality_vector_equation_solution_l2557_255704


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l2557_255707

def bacteria_count (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem first_day_exceeding_500 :
  let initial_count := 4
  let growth_factor := 3
  let target := 500
  (∀ d : ℕ, d < 6 → bacteria_count initial_count growth_factor d ≤ target) ∧
  (bacteria_count initial_count growth_factor 6 > target) :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l2557_255707


namespace NUMINAMATH_CALUDE_no_snow_probability_l2557_255783

theorem no_snow_probability (p : ℝ) (h : p = 3/4) :
  (1 - p)^3 = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l2557_255783


namespace NUMINAMATH_CALUDE_solve_equation_l2557_255790

theorem solve_equation (x : ℝ) : 4 * (x - 1) - 5 * (1 + x) = 3 ↔ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2557_255790


namespace NUMINAMATH_CALUDE_prime_sum_floor_squared_l2557_255794

theorem prime_sum_floor_squared : ∃! (p₁ p₂ : ℕ), 
  Prime p₁ ∧ Prime p₂ ∧ p₁ ≠ p₂ ∧
  (∃ n₁ : ℕ+, 5 * p₁ = ⌊(n₁.val^2 : ℚ) / 5⌋) ∧
  (∃ n₂ : ℕ+, 5 * p₂ = ⌊(n₂.val^2 : ℚ) / 5⌋) ∧
  p₁ + p₂ = 52 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_floor_squared_l2557_255794


namespace NUMINAMATH_CALUDE_married_men_fraction_l2557_255742

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : 0 < total_women) :
  let single_women := (3 * total_women) / 7
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  (single_women : ℚ) / total_women = 3 / 7 →
  (married_men : ℚ) / total_people = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l2557_255742


namespace NUMINAMATH_CALUDE_product_of_special_n_values_l2557_255726

theorem product_of_special_n_values : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧ n^2 - 40*n + 399 = p) ∧ 
  (∀ n : ℕ, (∃ p : ℕ, Nat.Prime p ∧ n^2 - 40*n + 399 = p) → n ∈ S) ∧
  S.card > 0 ∧
  (S.prod id = 396) := by
sorry

end NUMINAMATH_CALUDE_product_of_special_n_values_l2557_255726


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2557_255753

-- Define the polynomial
def cubic_polynomial (p q : ℚ) (x : ℝ) : ℝ := x^3 + p*x + q

-- State the theorem
theorem integer_root_of_cubic (p q : ℚ) : 
  (∃ (n : ℤ), cubic_polynomial p q n = 0) →
  (cubic_polynomial p q (3 - Real.sqrt 5) = 0) →
  (∃ (n : ℤ), cubic_polynomial p q n = 0 ∧ n = -6) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2557_255753


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l2557_255713

theorem opposite_of_negative_one_half :
  -(-(1/2)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l2557_255713


namespace NUMINAMATH_CALUDE_monthly_profit_calculation_l2557_255797

/-- Calculates the monthly profit for John's computer assembly business --/
theorem monthly_profit_calculation (cost_per_computer : ℝ) (markup : ℝ) 
  (computers_per_month : ℕ) (monthly_rent : ℝ) (monthly_non_rent_expenses : ℝ) :
  cost_per_computer = 800 →
  markup = 1.4 →
  computers_per_month = 60 →
  monthly_rent = 5000 →
  monthly_non_rent_expenses = 3000 →
  let selling_price := cost_per_computer * markup
  let total_revenue := selling_price * computers_per_month
  let total_component_cost := cost_per_computer * computers_per_month
  let total_expenses := monthly_rent + monthly_non_rent_expenses
  let profit := total_revenue - total_component_cost - total_expenses
  profit = 11200 := by
sorry

end NUMINAMATH_CALUDE_monthly_profit_calculation_l2557_255797


namespace NUMINAMATH_CALUDE_bottles_per_case_is_13_l2557_255700

/-- The number of bottles of water a company produces per day -/
def daily_production : ℕ := 65000

/-- The number of cases required to hold the daily production -/
def cases_required : ℕ := 5000

/-- The number of bottles that a single case can hold -/
def bottles_per_case : ℕ := daily_production / cases_required

theorem bottles_per_case_is_13 : bottles_per_case = 13 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_is_13_l2557_255700


namespace NUMINAMATH_CALUDE_complete_square_l2557_255708

theorem complete_square (b : ℝ) : ∀ x : ℝ, x^2 + b*x = (x + b/2)^2 - (b/2)^2 := by sorry

end NUMINAMATH_CALUDE_complete_square_l2557_255708


namespace NUMINAMATH_CALUDE_roots_sum_l2557_255734

theorem roots_sum (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = 0 ↔ x = m ∨ x = n) → 
  m = 2*n → 
  m + n = 3/2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_l2557_255734


namespace NUMINAMATH_CALUDE_cat_stairs_ways_l2557_255774

def stair_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | m + 4 => stair_ways m + stair_ways (m + 1) + stair_ways (m + 2)

theorem cat_stairs_ways :
  stair_ways 10 = 12 :=
by sorry

end NUMINAMATH_CALUDE_cat_stairs_ways_l2557_255774


namespace NUMINAMATH_CALUDE_bicycle_speed_l2557_255782

/-- Proves that given a 400 km trip where the first 100 km is traveled at speed v km/h
    and the remaining 300 km at 15 km/h, if the average speed for the entire trip is 16 km/h,
    then v = 20 km/h. -/
theorem bicycle_speed (v : ℝ) :
  v > 0 →
  (100 / v + 300 / 15 = 400 / 16) →
  v = 20 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_l2557_255782


namespace NUMINAMATH_CALUDE_floor_painting_rate_l2557_255749

/-- Proves that the painting rate for a rectangular floor is 5 Rs/sq m given specific conditions -/
theorem floor_painting_rate (length : ℝ) (total_cost : ℝ) : 
  length = 13.416407864998739 →
  total_cost = 300 →
  ∃ (breadth : ℝ), 
    length = 3 * breadth ∧ 
    (5 : ℝ) = total_cost / (length * breadth) := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_rate_l2557_255749


namespace NUMINAMATH_CALUDE_least_xy_value_l2557_255789

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 9) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 9 → (a : ℕ) * b ≥ (x : ℕ) * y) ∧
  (x : ℕ) * y = 108 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l2557_255789


namespace NUMINAMATH_CALUDE_race_distance_l2557_255717

theorem race_distance (p q : ℝ) (d : ℝ) : 
  p = 1.2 * q →  -- p is 20% faster than q
  d = q * (d + 50) / p →  -- race ends in a tie
  d + 50 = 300 :=  -- p runs 300 meters
by sorry

end NUMINAMATH_CALUDE_race_distance_l2557_255717


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2557_255799

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2557_255799


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l2557_255780

/-- The number of bottle caps Joshua bought -/
def bottle_caps_bought (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the number of bottle caps Joshua bought
    is the difference between his final and initial counts -/
theorem joshua_bottle_caps 
  (initial : ℕ) 
  (final : ℕ) 
  (h1 : initial = 40) 
  (h2 : final = 47) :
  bottle_caps_bought initial final = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l2557_255780


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2557_255771

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := 2 * t^2 - 4

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6 * t^2 + 11 * t - 6

/-- The theorem stating that the curve intersects itself at (18, -44√11 - 6) -/
theorem curve_self_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    x t₁ = x t₂ ∧ 
    y t₁ = y t₂ ∧ 
    x t₁ = 18 ∧ 
    y t₁ = -44 * Real.sqrt 11 - 6 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2557_255771


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l2557_255740

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 3*x - 18 ≤ 0}
def B : Set ℝ := {x : ℝ | 1 / (x + 1) ≤ -1}

-- Define the complement of B
def complement_B : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem for part (1)
theorem complement_B_intersect_A :
  (complement_B ∩ A) = {x : ℝ | (-6 ≤ x ∧ x < -2) ∨ (-1 ≤ x ∧ x ≤ 3)} :=
sorry

-- Define set C
def C (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 1}

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (B ∪ C a = B) ↔ (a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l2557_255740


namespace NUMINAMATH_CALUDE_triplet_equality_l2557_255786

theorem triplet_equality (a b c : ℝ) :
  a * (b^2 + c) = c * (c + a * b) →
  b * (c^2 + a) = a * (a + b * c) →
  c * (a^2 + b) = b * (b + a * c) →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triplet_equality_l2557_255786


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l2557_255768

theorem three_digit_number_operation (a b c : ℕ) : 
  a = c - 3 → 
  0 ≤ a ∧ a < 10 → 
  0 ≤ b ∧ b < 10 → 
  0 ≤ c ∧ c < 10 → 
  (2 * (100 * a + 10 * b + c) - (100 * c + 10 * b + a)) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l2557_255768


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2557_255751

def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_for_integral_solutions :
  (∀ m : ℤ, m > 0 ∧ m < 170 → ¬ has_integral_solutions 10 (-m) 720) ∧
  has_integral_solutions 10 (-170) 720 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2557_255751


namespace NUMINAMATH_CALUDE_water_speed_calculation_l2557_255720

/-- Given a person who can swim in still water at 10 km/h and takes 2 hours to swim 12 km against
    the current, prove that the speed of the water is 4 km/h. -/
theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ) :
  still_water_speed = 10 →
  distance = 12 →
  time = 2 →
  distance = (still_water_speed - water_speed) * time →
  water_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l2557_255720


namespace NUMINAMATH_CALUDE_factoring_expression_l2557_255710

theorem factoring_expression (x : ℝ) : x * (x + 4) + 2 * (x + 4) + (x + 4) = (x + 3) * (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2557_255710


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2557_255752

theorem quadratic_equation_roots (x : ℝ) :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ (x = r₁ ∨ x = r₂) ↔ x^2 - 2*x - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2557_255752


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l2557_255788

theorem cement_mixture_weight (sand water gravel cement limestone total : ℚ) : 
  sand = 2/9 →
  water = 5/18 →
  gravel = 1/6 →
  cement = 7/36 →
  limestone = 1 - (sand + water + gravel + cement) →
  limestone * total = 12 →
  total = 86.4 := by
sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l2557_255788


namespace NUMINAMATH_CALUDE_sum_first_102_remainder_l2557_255781

theorem sum_first_102_remainder (n : Nat) (h : n = 102) : 
  (n * (n + 1) / 2) % 5250 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_102_remainder_l2557_255781


namespace NUMINAMATH_CALUDE_goldfish_problem_l2557_255737

/-- The number of goldfish that died -/
def goldfish_died (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem goldfish_problem :
  let initial : ℕ := 89
  let remaining : ℕ := 57
  goldfish_died initial remaining = 32 := by
sorry

end NUMINAMATH_CALUDE_goldfish_problem_l2557_255737


namespace NUMINAMATH_CALUDE_employee_pay_l2557_255723

/-- Given two employees with a total pay of 528 and one paid 120% of the other, prove the lower-paid employee's wage --/
theorem employee_pay (x y : ℝ) (h1 : x + y = 528) (h2 : x = 1.2 * y) : y = 240 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l2557_255723


namespace NUMINAMATH_CALUDE_greatest_k_for_100_factorial_l2557_255795

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_of_2 (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1).log2) 0

def highest_power_of_5 (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (Nat.log 5 (x + 1))) 0

theorem greatest_k_for_100_factorial (a b : ℕ) (k : ℕ) :
  a = factorial 100 →
  b = 100^k →
  (∀ m : ℕ, m > k → ¬(100^m ∣ a)) →
  (100^k ∣ a) →
  k = 12 := by sorry

end NUMINAMATH_CALUDE_greatest_k_for_100_factorial_l2557_255795


namespace NUMINAMATH_CALUDE_inequality_proof_l2557_255769

theorem inequality_proof (x y : ℝ) (h : x > y) : 
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2557_255769


namespace NUMINAMATH_CALUDE_locus_of_P_B_l2557_255779

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
def Point := ℝ × ℝ

-- Define the given circle and points
variable (c : Circle)
variable (A : Point)
variable (B : Point)

-- Define P_B as a function of B
def P_B (B : Point) : Point := sorry

-- Define the condition that A and B are on the circle
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the condition that B is not on line OA
def not_on_line_OA (B : Point) (c : Circle) (A : Point) : Prop := sorry

-- Define the condition that P_B is on the internal bisector of ∠AOB
def on_internal_bisector (P : Point) (O : Point) (A : Point) (B : Point) : Prop := sorry

-- State the theorem
theorem locus_of_P_B (c : Circle) (A B : Point) 
  (h1 : on_circle A c)
  (h2 : on_circle B c)
  (h3 : not_on_line_OA B c A)
  (h4 : on_internal_bisector (P_B B) c.center A B) :
  ∃ (r : ℝ), ∀ B, on_circle (P_B B) { center := c.center, radius := r } :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_B_l2557_255779


namespace NUMINAMATH_CALUDE_correct_number_l2557_255709

theorem correct_number (x : ℤ) (h1 : x - 152 = 346) : x + 152 = 650 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_l2557_255709


namespace NUMINAMATH_CALUDE_phi_equals_theta_is_plane_l2557_255761

/-- Spherical coordinates in 3D space -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A generalized plane in 3D space -/
structure GeneralizedPlane where
  equation : SphericalCoord → Prop

/-- The specific equation φ = θ -/
def phiEqualsThetaPlane : GeneralizedPlane where
  equation := fun coord => coord.φ = coord.θ

/-- Theorem: The equation φ = θ in spherical coordinates describes a generalized plane -/
theorem phi_equals_theta_is_plane : 
  ∃ (p : GeneralizedPlane), p = phiEqualsThetaPlane :=
sorry

end NUMINAMATH_CALUDE_phi_equals_theta_is_plane_l2557_255761


namespace NUMINAMATH_CALUDE_managers_in_game_l2557_255773

/-- The number of managers participating in a volleyball game --/
def num_managers (total_teams : ℕ) (people_per_team : ℕ) (num_employees : ℕ) : ℕ :=
  total_teams * people_per_team - num_employees

/-- Theorem stating that the number of managers in the game is 3 --/
theorem managers_in_game :
  num_managers 3 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_managers_in_game_l2557_255773


namespace NUMINAMATH_CALUDE_equation_solution_l2557_255784

theorem equation_solution : ∃ y : ℚ, y - 1/2 = 1/6 - 2/3 + 1/4 ∧ y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2557_255784


namespace NUMINAMATH_CALUDE_combination_permutation_equality_l2557_255727

theorem combination_permutation_equality (n : ℕ) (hn : n > 0) :
  3 * (Nat.choose (n - 1) (n - 5)) = 5 * (Nat.factorial (n - 2) / Nat.factorial (n - 4)) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_combination_permutation_equality_l2557_255727


namespace NUMINAMATH_CALUDE_solution_set_is_two_l2557_255791

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := log10 (2 * x + 1) + log10 x = 1

-- Theorem statement
theorem solution_set_is_two :
  ∃! x : ℝ, x > 0 ∧ 2 * x + 1 > 0 ∧ equation x := by sorry

end NUMINAMATH_CALUDE_solution_set_is_two_l2557_255791


namespace NUMINAMATH_CALUDE_expression_value_l2557_255764

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  4 * (x - y)^2 - x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2557_255764


namespace NUMINAMATH_CALUDE_muffins_per_box_l2557_255759

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) (muffins_per_box : ℕ) : 
  total_muffins = 96 →
  num_boxes = 8 →
  total_muffins = num_boxes * muffins_per_box →
  muffins_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_muffins_per_box_l2557_255759


namespace NUMINAMATH_CALUDE_square_perimeter_equals_area_l2557_255765

theorem square_perimeter_equals_area (x : ℝ) (h : x > 0) :
  4 * x = x^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_equals_area_l2557_255765


namespace NUMINAMATH_CALUDE_oddFactorsOf252_eq_six_l2557_255745

/-- The number of odd factors of 252 -/
def oddFactorsOf252 : ℕ :=
  let n : ℕ := 252
  let primeFactors : List (ℕ × ℕ) := [(3, 2), (7, 1)]  -- List of (prime, exponent) pairs for odd primes
  (primeFactors.map (fun (p, e) => e + 1)).prod

/-- Theorem: The number of odd factors of 252 is 6 -/
theorem oddFactorsOf252_eq_six : oddFactorsOf252 = 6 := by
  sorry

end NUMINAMATH_CALUDE_oddFactorsOf252_eq_six_l2557_255745


namespace NUMINAMATH_CALUDE_inequality_proof_l2557_255767

theorem inequality_proof (t : ℝ) (h : t > 0) : (1 + 2/t) * Real.log (1 + t) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2557_255767


namespace NUMINAMATH_CALUDE_cecil_money_problem_l2557_255785

theorem cecil_money_problem (cecil : ℝ) (catherine : ℝ) (carmela : ℝ) 
  (h1 : catherine = 2 * cecil - 250)
  (h2 : carmela = 2 * cecil + 50)
  (h3 : cecil + catherine + carmela = 2800) :
  cecil = 600 := by
sorry

end NUMINAMATH_CALUDE_cecil_money_problem_l2557_255785


namespace NUMINAMATH_CALUDE_elizabeth_granola_profit_l2557_255762

/-- Calculates the net profit for Elizabeth's granola bag sales --/
theorem elizabeth_granola_profit : 
  let full_price : ℝ := 6.00
  let low_cost : ℝ := 2.50
  let high_cost : ℝ := 3.50
  let low_cost_bags : ℕ := 10
  let high_cost_bags : ℕ := 10
  let full_price_low_cost_sold : ℕ := 7
  let full_price_high_cost_sold : ℕ := 8
  let discounted_low_cost_bags : ℕ := 3
  let discounted_high_cost_bags : ℕ := 2
  let low_cost_discount : ℝ := 0.20
  let high_cost_discount : ℝ := 0.30

  let total_cost : ℝ := low_cost * low_cost_bags + high_cost * high_cost_bags
  let full_price_revenue : ℝ := full_price * (full_price_low_cost_sold + full_price_high_cost_sold)
  let discounted_low_price : ℝ := full_price * (1 - low_cost_discount)
  let discounted_high_price : ℝ := full_price * (1 - high_cost_discount)
  let discounted_revenue : ℝ := discounted_low_price * discounted_low_cost_bags + 
                                 discounted_high_price * discounted_high_cost_bags
  let total_revenue : ℝ := full_price_revenue + discounted_revenue
  let net_profit : ℝ := total_revenue - total_cost

  net_profit = 52.80 := by sorry

end NUMINAMATH_CALUDE_elizabeth_granola_profit_l2557_255762


namespace NUMINAMATH_CALUDE_comparison_of_powers_l2557_255743

theorem comparison_of_powers (a b c : ℕ) : 
  a = 81^31 → b = 27^41 → c = 9^61 → a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l2557_255743


namespace NUMINAMATH_CALUDE_car_speed_problem_l2557_255754

/-- Proves that Car B's speed is 50 mph given the problem conditions -/
theorem car_speed_problem (speed_A speed_B initial_distance overtake_time final_distance : ℝ) :
  speed_A = 58 ∧ 
  initial_distance = 16 ∧ 
  overtake_time = 3 ∧ 
  final_distance = 8 ∧
  speed_A * overtake_time = speed_B * overtake_time + initial_distance + final_distance →
  speed_B = 50 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2557_255754


namespace NUMINAMATH_CALUDE_pencil_cost_l2557_255735

theorem pencil_cost (x y : ℚ) 
  (eq1 : 5 * x + 2 * y = 286)
  (eq2 : 3 * x + 4 * y = 204) :
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2557_255735


namespace NUMINAMATH_CALUDE_water_consumption_l2557_255718

theorem water_consumption (initial_water : ℝ) : 
  initial_water > 0 →
  let remaining_day1 := initial_water * (1 - 7/15)
  let remaining_day2 := remaining_day1 * (1 - 5/8)
  let remaining_day3 := remaining_day2 * (1 - 2/3)
  remaining_day3 = 2.6 →
  initial_water = 39 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_l2557_255718


namespace NUMINAMATH_CALUDE_system_solution_existence_l2557_255777

theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ 
  b ≤ 2 * Real.sqrt 2 + 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2557_255777


namespace NUMINAMATH_CALUDE_car_average_speed_l2557_255738

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 50) :
  (speed1 + speed2) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2557_255738


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2557_255775

theorem quadratic_equation_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + k^2 - 1 = 0) ∧ 
  ((k - 1) * 0^2 + 6 * 0 + k^2 - 1 = 0) → 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2557_255775


namespace NUMINAMATH_CALUDE_james_played_five_rounds_l2557_255792

/-- Represents the quiz bowl scoring system and James' performance -/
structure QuizBowl where
  pointsPerCorrectAnswer : ℕ
  questionsPerRound : ℕ
  bonusPoints : ℕ
  totalPoints : ℕ
  missedQuestions : ℕ

/-- Calculates the number of rounds played given the quiz bowl parameters -/
def calculateRounds (qb : QuizBowl) : ℕ :=
  sorry

/-- Theorem stating that James played 5 rounds -/
theorem james_played_five_rounds (qb : QuizBowl) 
  (h1 : qb.pointsPerCorrectAnswer = 2)
  (h2 : qb.questionsPerRound = 5)
  (h3 : qb.bonusPoints = 4)
  (h4 : qb.totalPoints = 66)
  (h5 : qb.missedQuestions = 1) :
  calculateRounds qb = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_played_five_rounds_l2557_255792


namespace NUMINAMATH_CALUDE_more_non_products_than_products_l2557_255758

/-- The number of ten-digit numbers -/
def ten_digit_count : ℕ := 9 * 10^9

/-- The number of five-digit numbers -/
def five_digit_count : ℕ := 90000

/-- The estimated number of products of two five-digit numbers that are ten-digit numbers -/
def ten_digit_products : ℕ := (five_digit_count * (five_digit_count - 1) / 2 + five_digit_count) / 2

theorem more_non_products_than_products : ten_digit_count - ten_digit_products > ten_digit_products := by
  sorry

end NUMINAMATH_CALUDE_more_non_products_than_products_l2557_255758


namespace NUMINAMATH_CALUDE_greatest_b_satisfying_inequality_l2557_255796

def quadratic_inequality (b : ℝ) : Prop :=
  b^2 - 14*b + 45 ≤ 0

theorem greatest_b_satisfying_inequality :
  ∃ (b : ℝ), quadratic_inequality b ∧
    ∀ (x : ℝ), quadratic_inequality x → x ≤ b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_b_satisfying_inequality_l2557_255796


namespace NUMINAMATH_CALUDE_travis_apple_sales_proof_l2557_255730

/-- Calculates the total money Travis will take home from selling apples -/
def travis_apple_sales (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_apples / apples_per_box) * price_per_box

/-- Proves that Travis will take home $7000 from selling his apples -/
theorem travis_apple_sales_proof :
  travis_apple_sales 10000 50 35 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_travis_apple_sales_proof_l2557_255730
