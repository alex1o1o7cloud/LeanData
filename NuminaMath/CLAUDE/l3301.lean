import Mathlib

namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3301_330175

theorem solve_exponential_equation :
  ∃ y : ℝ, (5 : ℝ)^9 = 25^y ∧ y = (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3301_330175


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l3301_330105

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2) + f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l3301_330105


namespace NUMINAMATH_CALUDE_queen_diamond_probability_l3301_330147

/-- Represents a standard deck of 52 playing cards -/
def Deck : Type := Unit

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of diamonds in a standard deck -/
def num_diamonds : ℕ := 13

/-- Represents the event of drawing a Queen as the first card and a diamond as the second card -/
def queen_then_diamond (d : Deck) : Prop := sorry

/-- The probability of the queen_then_diamond event -/
def prob_queen_then_diamond (d : Deck) : ℚ := sorry

theorem queen_diamond_probability (d : Deck) : 
  prob_queen_then_diamond d = 1 / deck_size := by sorry

end NUMINAMATH_CALUDE_queen_diamond_probability_l3301_330147


namespace NUMINAMATH_CALUDE_fence_cost_l3301_330163

-- Define the side lengths of the pentagon
def side1 : ℕ := 10
def side2 : ℕ := 14
def side3 : ℕ := 12
def side4 : ℕ := 8
def side5 : ℕ := 6

-- Define the prices per foot for each group of sides
def price1 : ℕ := 45  -- Price for first two sides
def price2 : ℕ := 55  -- Price for third and fourth sides
def price3 : ℕ := 60  -- Price for last side

-- Define the total cost function
def totalCost : ℕ := 
  side1 * price1 + side2 * price1 + 
  side3 * price2 + side4 * price2 + 
  side5 * price3

-- Theorem stating that the total cost is 2540
theorem fence_cost : totalCost = 2540 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l3301_330163


namespace NUMINAMATH_CALUDE_scout_troop_profit_l3301_330138

/-- Calculates the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_rate : ℚ) (sell_rate : ℚ) : 
  num_bars = 1200 → 
  buy_rate = 1/3 → 
  sell_rate = 3/5 → 
  (sell_rate * num_bars : ℚ) - (buy_rate * num_bars : ℚ) = 320 := by
  sorry

#check scout_troop_profit

end NUMINAMATH_CALUDE_scout_troop_profit_l3301_330138


namespace NUMINAMATH_CALUDE_hundreds_digit_of_binomial_12_6_times_6_factorial_l3301_330118

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function to get the hundreds digit
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

-- Theorem statement
theorem hundreds_digit_of_binomial_12_6_times_6_factorial :
  hundreds_digit (binomial 12 6 * Nat.factorial 6) = 8 := by
  sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_binomial_12_6_times_6_factorial_l3301_330118


namespace NUMINAMATH_CALUDE_two_digit_multiple_of_35_l3301_330162

theorem two_digit_multiple_of_35 (n : ℕ) (h1 : 10 ≤ n ∧ n < 100) (h2 : n % 35 = 0) : 
  n % 10 = 5 :=
sorry

end NUMINAMATH_CALUDE_two_digit_multiple_of_35_l3301_330162


namespace NUMINAMATH_CALUDE_existence_of_m_n_l3301_330169

theorem existence_of_m_n (p s : ℕ) (hp : Nat.Prime p) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧
    (m * s % p : ℚ) / p < (n * s % p : ℚ) / p ∧ (n * s % p : ℚ) / p < (s : ℚ) / p) ↔
  ¬(s ∣ p - 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3301_330169


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3301_330174

theorem simple_interest_rate_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 10000) 
  (h2 : time = 1) 
  (h3 : interest = 800) : 
  (interest / (principal * time)) * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3301_330174


namespace NUMINAMATH_CALUDE_red_beads_count_l3301_330120

/-- The total number of beads in the string -/
def total_beads : ℕ := 85

/-- The number of green beads in one pattern cycle -/
def green_in_cycle : ℕ := 3

/-- The number of red beads in one pattern cycle -/
def red_in_cycle : ℕ := 4

/-- The number of yellow beads in one pattern cycle -/
def yellow_in_cycle : ℕ := 1

/-- The total number of beads in one pattern cycle -/
def beads_per_cycle : ℕ := green_in_cycle + red_in_cycle + yellow_in_cycle

/-- The number of complete cycles in the string -/
def complete_cycles : ℕ := total_beads / beads_per_cycle

/-- The number of beads remaining after complete cycles -/
def remaining_beads : ℕ := total_beads % beads_per_cycle

/-- The number of red beads in the remaining portion -/
def red_in_remaining : ℕ := min remaining_beads (red_in_cycle)

/-- Theorem: The total number of red beads in the string is 42 -/
theorem red_beads_count : 
  complete_cycles * red_in_cycle + red_in_remaining = 42 := by
sorry

end NUMINAMATH_CALUDE_red_beads_count_l3301_330120


namespace NUMINAMATH_CALUDE_jane_usable_a4_sheets_l3301_330184

/-- Represents the different types of paper sheets -/
inductive SheetType
  | BrownA4
  | YellowA4
  | YellowA3
  | PinkA2

/-- Calculates the number of usable sheets given the total and damaged counts -/
def usableSheets (total : ℕ) (damaged : ℕ) : ℕ :=
  total - damaged + (damaged / 2)

/-- Theorem: Jane has 40 total usable A4 sheets for sketching -/
theorem jane_usable_a4_sheets :
  let brown_a4_total := 28
  let yellow_a4_total := 18
  let yellow_a3_total := 9
  let pink_a2_total := 10
  let brown_a4_damaged := 3
  let yellow_a4_damaged := 5
  let yellow_a3_damaged := 2
  let pink_a2_damaged := 2
  let brown_a4_usable := usableSheets brown_a4_total brown_a4_damaged
  let yellow_a4_usable := usableSheets yellow_a4_total yellow_a4_damaged
  brown_a4_usable + yellow_a4_usable = 40 := by
    sorry


end NUMINAMATH_CALUDE_jane_usable_a4_sheets_l3301_330184


namespace NUMINAMATH_CALUDE_indeterminate_remainder_l3301_330126

theorem indeterminate_remainder (a b c d m n x y : ℤ) 
  (eq1 : a * x + b * y = m)
  (eq2 : c * x + d * y = n)
  (rem64 : ∃ k : ℤ, a * x + b * y = 64 * k + 37) :
  ∀ r : ℤ, ¬ (∀ k : ℤ, c * x + d * y = 5 * k + r ∧ 0 ≤ r ∧ r < 5) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_remainder_l3301_330126


namespace NUMINAMATH_CALUDE_eva_total_marks_l3301_330152

def eva_marks (maths_second science_second arts_second : ℕ) : Prop :=
  let maths_first := maths_second + 10
  let arts_first := arts_second - 15
  let science_first := science_second - (science_second / 3)
  let total_first := maths_first + arts_first + science_first
  let total_second := maths_second + science_second + arts_second
  total_first + total_second = 485

theorem eva_total_marks :
  eva_marks 80 90 90 := by sorry

end NUMINAMATH_CALUDE_eva_total_marks_l3301_330152


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3301_330122

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - x + a < 0 ↔ -1 < x ∧ x < 2) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3301_330122


namespace NUMINAMATH_CALUDE_fraction_power_calculation_l3301_330170

theorem fraction_power_calculation (x y : ℚ) 
  (hx : x = 2/3) (hy : y = 3/2) : 
  (3/4) * x^8 * y^9 = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_calculation_l3301_330170


namespace NUMINAMATH_CALUDE_min_range_for_largest_angle_l3301_330156

-- Define the triangle sides as functions of x
def side_a (x : ℝ) := 2 * x
def side_b (x : ℝ) := x + 3
def side_c (x : ℝ) := x + 6

-- Define the triangle inequality conditions
def triangle_inequality (x : ℝ) : Prop :=
  side_a x + side_b x > side_c x ∧
  side_a x + side_c x > side_b x ∧
  side_b x + side_c x > side_a x

-- Define the condition for ∠A to be the largest angle
def angle_a_largest (x : ℝ) : Prop :=
  side_c x > side_a x ∧ side_c x > side_b x

-- Theorem stating the minimum range for x
theorem min_range_for_largest_angle :
  ∃ (m n : ℝ), m < n ∧
  (∀ x, m < x ∧ x < n → triangle_inequality x ∧ angle_a_largest x) ∧
  (∀ m' n', m' < n' →
    (∀ x, m' < x ∧ x < n' → triangle_inequality x ∧ angle_a_largest x) →
    n - m ≤ n' - m') ∧
  n - m = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_range_for_largest_angle_l3301_330156


namespace NUMINAMATH_CALUDE_hotel_room_charges_percentage_increase_l3301_330178

/-- Proves that if the charge for a single room at hotel P is 70% less than hotel R
    and 10% less than hotel G, then the charge for a single room at hotel R is 170%
    greater than hotel G. -/
theorem hotel_room_charges (P R G : ℝ) 
    (h1 : P = R * 0.3)  -- P is 70% less than R
    (h2 : P = G * 0.9)  -- P is 10% less than G
    : R = G * 2.7 := by
  sorry

/-- Proves that if R = G * 2.7, then R is 170% greater than G. -/
theorem percentage_increase (R G : ℝ) (h : R = G * 2.7) 
    : (R - G) / G * 100 = 170 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charges_percentage_increase_l3301_330178


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3301_330104

theorem trig_identity_proof : 
  (Real.cos (63 * π / 180) * Real.cos (3 * π / 180) - 
   Real.cos (87 * π / 180) * Real.cos (27 * π / 180)) / 
  (Real.cos (132 * π / 180) * Real.cos (72 * π / 180) - 
   Real.cos (42 * π / 180) * Real.cos (18 * π / 180)) = 
  -Real.tan (24 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3301_330104


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l3301_330121

/-- Given a line x - 2y + a = 0 and a circle (x-2)^2 + y^2 = 1 with common points,
    the range of values for the real number a is [-2-√5, -2+√5]. -/
theorem line_circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - 2*y + a = 0 ∧ (x-2)^2 + y^2 = 1) →
  a ∈ Set.Icc (-2 - Real.sqrt 5) (-2 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l3301_330121


namespace NUMINAMATH_CALUDE_trig_identity_l3301_330106

theorem trig_identity (a b c : ℝ) (θ : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b + (Real.sin θ)^2 * (Real.cos θ)^2 / c = 1 / (a + b + c) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 + ((Real.sin θ)^2 * (Real.cos θ)^2)^3 / c^5 = 
    (a + b + (a*b)^3/c^5) / (a + b + c)^6 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l3301_330106


namespace NUMINAMATH_CALUDE_factor_expression_l3301_330182

theorem factor_expression (y : ℝ) : 3 * y^2 - 75 = 3 * (y - 5) * (y + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3301_330182


namespace NUMINAMATH_CALUDE_largest_root_of_quadratic_l3301_330167

theorem largest_root_of_quadratic (y : ℝ) :
  (6 * y ^ 2 - 31 * y + 35 = 0) → y ≤ (5 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_root_of_quadratic_l3301_330167


namespace NUMINAMATH_CALUDE_ratio_problem_l3301_330183

theorem ratio_problem (p q r s : ℚ) 
  (h1 : p / q = 4)
  (h2 : q / r = 3)
  (h3 : r / s = 1 / 5) :
  s / p = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3301_330183


namespace NUMINAMATH_CALUDE_beef_to_steaks_l3301_330141

/-- Given 15 pounds of beef cut into 12-ounce steaks, prove that the number of steaks obtained is 20. -/
theorem beef_to_steaks :
  let pounds_of_beef : ℕ := 15
  let ounces_per_pound : ℕ := 16
  let ounces_per_steak : ℕ := 12
  let total_ounces : ℕ := pounds_of_beef * ounces_per_pound
  let number_of_steaks : ℕ := total_ounces / ounces_per_steak
  number_of_steaks = 20 :=
by sorry

end NUMINAMATH_CALUDE_beef_to_steaks_l3301_330141


namespace NUMINAMATH_CALUDE_qt_plus_q_plus_t_not_two_l3301_330127

theorem qt_plus_q_plus_t_not_two :
  ∀ q t : ℕ+, q * t + q + t ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_qt_plus_q_plus_t_not_two_l3301_330127


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_192_l3301_330149

/-- The sum of all integer coefficients in the factorization of 216x^9 - 1000y^9 -/
def sum_of_coefficients (x y : ℚ) : ℤ :=
  let expression := 216 * x^9 - 1000 * y^9
  -- The actual computation of the sum is not implemented here
  192

/-- Theorem stating that the sum of all integer coefficients in the factorization of 216x^9 - 1000y^9 is 192 -/
theorem sum_of_coefficients_is_192 (x y : ℚ) : sum_of_coefficients x y = 192 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_192_l3301_330149


namespace NUMINAMATH_CALUDE_peach_count_l3301_330179

/-- The number of peaches Sally had initially -/
def initial_peaches : ℕ := 13

/-- The number of peaches Sally picked -/
def picked_peaches : ℕ := 55

/-- The total number of peaches Sally has now -/
def total_peaches : ℕ := initial_peaches + picked_peaches

theorem peach_count : total_peaches = 68 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_l3301_330179


namespace NUMINAMATH_CALUDE_equal_diagonals_only_in_quad_and_pent_l3301_330130

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  diagonals : ℕ
  has_equal_diagonals : Bool

/-- Definition: A polygon has equal diagonals if all its diagonals have the same length. -/
def has_equal_diagonals (p : Polygon) : Prop :=
  p.has_equal_diagonals = true

/-- Theorem: Among polygons with 3 or more sides, only quadrilaterals and pentagons can have all diagonals equal. -/
theorem equal_diagonals_only_in_quad_and_pent (p : Polygon) :
  p.sides ≥ 3 → (has_equal_diagonals p ↔ p.sides = 4 ∨ p.sides = 5) := by
  sorry

#check equal_diagonals_only_in_quad_and_pent

end NUMINAMATH_CALUDE_equal_diagonals_only_in_quad_and_pent_l3301_330130


namespace NUMINAMATH_CALUDE_bank_account_balance_l3301_330102

theorem bank_account_balance 
  (transferred_amount : ℕ) 
  (remaining_balance : ℕ) 
  (original_balance : ℕ) : 
  transferred_amount = 69 → 
  remaining_balance = 26935 → 
  original_balance = remaining_balance + transferred_amount → 
  original_balance = 27004 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_balance_l3301_330102


namespace NUMINAMATH_CALUDE_maria_carrots_next_day_l3301_330101

/-- The number of carrots Maria picked the next day -/
def carrots_picked_next_day (initial_carrots thrown_out final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - thrown_out)

/-- Theorem stating that Maria picked 15 carrots the next day -/
theorem maria_carrots_next_day : 
  carrots_picked_next_day 48 11 52 = 15 := by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_next_day_l3301_330101


namespace NUMINAMATH_CALUDE_log_equation_solution_l3301_330133

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y ^ 2 / Real.log 3 + Real.log y / Real.log (1/3) = 6 →
  y = 729 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3301_330133


namespace NUMINAMATH_CALUDE_absolute_value_square_l3301_330100

theorem absolute_value_square (a b : ℚ) : |a| = b → a^2 = (-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_l3301_330100


namespace NUMINAMATH_CALUDE_expression_evaluation_l3301_330137

theorem expression_evaluation (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a * x + y / b ≠ 0) :
  (a * x + y / b)⁻¹ * ((a * x)⁻¹ + (y / b)⁻¹) = (a * x * y)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3301_330137


namespace NUMINAMATH_CALUDE_smallest_value_of_complex_expression_l3301_330109

theorem smallest_value_of_complex_expression (a b c d : ℤ) (ω : ℂ) (ζ : ℂ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ω^4 = 1 →
  ω ≠ 1 →
  ζ = ω^2 →
  ∃ (x y z w : ℤ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    |Complex.abs (↑x + ↑y * ω + ↑z * ζ + ↑w * ω^3)| = Real.sqrt 2 ∧
    ∀ (p q r s : ℤ), p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
      |Complex.abs (↑p + ↑q * ω + ↑r * ζ + ↑s * ω^3)| ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_complex_expression_l3301_330109


namespace NUMINAMATH_CALUDE_problem_solution_l3301_330123

def proposition (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - 2*m*x - 3*m^2 < 0

def set_A : Set ℝ := {m | proposition m}

def set_B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

theorem problem_solution :
  (set_A = Set.Ioi (-2) ∪ Set.Iio (2/3)) ∧
  {a | set_A ⊆ set_B a ∧ set_A ≠ set_B a} = Set.Iic (-3) ∪ Set.Ici (5/3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3301_330123


namespace NUMINAMATH_CALUDE_susan_single_digit_in_ten_steps_l3301_330192

/-- Represents a multi-digit number as a list of digits -/
def MultiDigitNumber := List Nat

/-- Represents a position where a plus sign can be inserted -/
def PlusPosition := Nat

/-- Represents a set of positions where plus signs are inserted -/
def PlusPositions := List PlusPosition

/-- Performs one step of Susan's operation -/
def performStep (n : MultiDigitNumber) (positions : PlusPositions) : MultiDigitNumber :=
  sorry

/-- Checks if a number is a single digit -/
def isSingleDigit (n : MultiDigitNumber) : Prop :=
  n.length = 1

/-- Main theorem: Susan can always obtain a single-digit number in at most ten steps -/
theorem susan_single_digit_in_ten_steps (n : MultiDigitNumber) :
  ∃ (steps : List PlusPositions),
    steps.length ≤ 10 ∧
    isSingleDigit (steps.foldl performStep n) :=
  sorry

end NUMINAMATH_CALUDE_susan_single_digit_in_ten_steps_l3301_330192


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3301_330131

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3301_330131


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_three_digits_l3301_330119

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3 ∨ d = 6

def contains_all_required_digits (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 3 ∈ n.digits 10 ∧ 6 ∈ n.digits 10

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem smallest_valid_number_last_three_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 2 = 0 ∧
    m % 3 = 0 ∧
    is_valid_number m ∧
    contains_all_required_digits m ∧
    (∀ k : ℕ, k > 0 ∧ k % 2 = 0 ∧ k % 3 = 0 ∧ is_valid_number k ∧ contains_all_required_digits k → m ≤ k) ∧
    last_three_digits m = 326 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_three_digits_l3301_330119


namespace NUMINAMATH_CALUDE_car_motorcycle_transaction_result_l3301_330187

theorem car_motorcycle_transaction_result :
  let car_selling_price : ℚ := 20000
  let motorcycle_selling_price : ℚ := 10000
  let car_loss_percentage : ℚ := 25 / 100
  let motorcycle_gain_percentage : ℚ := 25 / 100
  let car_cost : ℚ := car_selling_price / (1 - car_loss_percentage)
  let motorcycle_cost : ℚ := motorcycle_selling_price / (1 + motorcycle_gain_percentage)
  let total_cost : ℚ := car_cost + motorcycle_cost
  let total_selling_price : ℚ := car_selling_price + motorcycle_selling_price
  let transaction_result : ℚ := total_cost - total_selling_price
  transaction_result = 4667 / 1 := by sorry

end NUMINAMATH_CALUDE_car_motorcycle_transaction_result_l3301_330187


namespace NUMINAMATH_CALUDE_beta_value_l3301_330108

theorem beta_value (β : ℂ) 
  (h1 : β ≠ 1)
  (h2 : Complex.abs (β^2 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^4 - 1) = 5 * Complex.abs (β - 1)) :
  β = 2 := by
  sorry

end NUMINAMATH_CALUDE_beta_value_l3301_330108


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3301_330115

theorem least_positive_integer_with_remainders : 
  ∃! n : ℕ, n > 0 ∧ 
    n % 3 = 2 ∧ 
    n % 4 = 3 ∧ 
    n % 5 = 4 ∧ 
    n % 6 = 5 ∧
    ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 → n ≤ m :=
by sorry

#eval 119 % 3  -- Expected: 2
#eval 119 % 4  -- Expected: 3
#eval 119 % 5  -- Expected: 4
#eval 119 % 6  -- Expected: 5

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3301_330115


namespace NUMINAMATH_CALUDE_sector_area_l3301_330111

/-- Given a sector with central angle 2 radians and arc length 4, its area is equal to 4 -/
theorem sector_area (θ : Real) (L : Real) (r : Real) (A : Real) : 
  θ = 2 → L = 4 → L = θ * r → A = 1/2 * θ * r^2 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3301_330111


namespace NUMINAMATH_CALUDE_vector_perpendicular_and_obtuse_angle_l3301_330168

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

-- Define x and y as functions of k
def x (k : ℝ) : Fin 2 → ℝ := ![k - 3, 2*k + 2]
def y : Fin 2 → ℝ := ![10, -4]

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the theorem
theorem vector_perpendicular_and_obtuse_angle (k : ℝ) :
  (dot_product (x k) y = 0 ↔ k = 19) ∧
  (dot_product (x k) y < 0 ↔ k < 19 ∧ k ≠ -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_and_obtuse_angle_l3301_330168


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l3301_330155

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_num : Nat := 56

-- Theorem statement
theorem binary_to_octal_conversion :
  (binary_num.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0) = octal_num * 8 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l3301_330155


namespace NUMINAMATH_CALUDE_test_score_calculation_l3301_330198

theorem test_score_calculation (total_questions : ℕ) (first_half : ℕ) (second_half : ℕ)
  (first_correct_rate : ℚ) (second_correct_rate : ℚ)
  (h1 : total_questions = 80)
  (h2 : first_half = 40)
  (h3 : second_half = 40)
  (h4 : first_correct_rate = 9/10)
  (h5 : second_correct_rate = 19/20)
  (h6 : total_questions = first_half + second_half) :
  ⌊first_correct_rate * first_half⌋ + ⌊second_correct_rate * second_half⌋ = 74 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l3301_330198


namespace NUMINAMATH_CALUDE_beetle_speed_l3301_330116

/-- Proves that a beetle's speed is 2.7 km/h given specific conditions --/
theorem beetle_speed : 
  let ant_distance : ℝ := 600 -- meters
  let ant_time : ℝ := 10 -- minutes
  let beetle_distance_ratio : ℝ := 0.75 -- 25% less than ant
  let beetle_distance : ℝ := ant_distance * beetle_distance_ratio
  let km_per_meter : ℝ := 1 / 1000
  let hours_per_minute : ℝ := 1 / 60
  beetle_distance * km_per_meter / (ant_time * hours_per_minute) = 2.7 := by
sorry

end NUMINAMATH_CALUDE_beetle_speed_l3301_330116


namespace NUMINAMATH_CALUDE_birds_on_fence_l3301_330148

/-- The number of birds initially sitting on the fence -/
def initial_birds : ℕ := 4

/-- The initial number of storks -/
def initial_storks : ℕ := 3

/-- The number of additional storks that joined -/
def additional_storks : ℕ := 6

theorem birds_on_fence :
  initial_birds = 4 ∧
  initial_storks = 3 ∧
  additional_storks = 6 ∧
  initial_storks + additional_storks = initial_birds + 5 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3301_330148


namespace NUMINAMATH_CALUDE_monotonic_increasing_cubic_l3301_330145

/-- A cubic function with parameters m and n -/
def f (m n : ℝ) (x : ℝ) : ℝ := 4 * x^3 + m * x^2 + (m - 3) * x + n

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * m * x + (m - 3)

theorem monotonic_increasing_cubic (m n : ℝ) :
  (∀ x : ℝ, Monotone (f m n)) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_cubic_l3301_330145


namespace NUMINAMATH_CALUDE_min_value_of_f_l3301_330180

def f (x : ℝ) := 27 * x - x^3

theorem min_value_of_f :
  ∃ (min : ℝ), min = -54 ∧
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3301_330180


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l3301_330117

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The sample center point of a dataset -/
structure SampleCenterPoint where
  x : ℝ
  y : ℝ

/-- Theorem: Two regression lines with the same sample center point intersect -/
theorem regression_lines_intersect
  (l₁ l₂ : RegressionLine)
  (center : SampleCenterPoint)
  (h₁ : center.y = l₁.slope * center.x + l₁.intercept)
  (h₂ : center.y = l₂.slope * center.x + l₂.intercept) :
  ∃ (x y : ℝ), y = l₁.slope * x + l₁.intercept ∧ y = l₂.slope * x + l₂.intercept :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_l3301_330117


namespace NUMINAMATH_CALUDE_rectangle_area_l3301_330193

/-- The area of a rectangle bounded by lines y = 2a, y = 3b, x = 4c, and x = 5d,
    where a, b, c, and d are positive numbers. -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (2 * a - 3 * b) * (5 * d - 4 * c) = 10 * a * d - 8 * a * c - 15 * b * d + 12 * b * c := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3301_330193


namespace NUMINAMATH_CALUDE_successive_discounts_result_l3301_330124

/-- Calculates the final price after applying successive discounts -/
def finalPrice (initialPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that applying successive discounts of 20%, 10%, and 5% to a good 
    with an actual price of Rs. 9941.52 results in a final selling price of Rs. 6800.00 -/
theorem successive_discounts_result (ε : ℝ) (h : ε > 0) :
  ∃ (result : ℝ), abs (finalPrice 9941.52 0.20 0.10 0.05 - 6800.00) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_successive_discounts_result_l3301_330124


namespace NUMINAMATH_CALUDE_cone_base_radius_l3301_330196

/-- Given a sector paper with a central angle of 90° and a radius of 20 cm
    used to form the lateral surface of a cone, the radius of the base of the cone is 5 cm. -/
theorem cone_base_radius (θ : Real) (R : Real) (r : Real) : 
  θ = 90 → R = 20 → 2 * π * r = (θ / 360) * 2 * π * R → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3301_330196


namespace NUMINAMATH_CALUDE_total_amount_pigs_and_hens_l3301_330153

/-- The total amount spent on buying pigs and hens -/
def total_amount (num_pigs : ℕ) (num_hens : ℕ) (price_pig : ℕ) (price_hen : ℕ) : ℕ :=
  num_pigs * price_pig + num_hens * price_hen

/-- Theorem stating that the total amount spent on 3 pigs at Rs. 300 each and 10 hens at Rs. 30 each is Rs. 1200 -/
theorem total_amount_pigs_and_hens :
  total_amount 3 10 300 30 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_pigs_and_hens_l3301_330153


namespace NUMINAMATH_CALUDE_smaller_angle_is_45_degrees_l3301_330157

/-- A parallelogram with a specific angle ratio -/
structure AngleRatioParallelogram where
  -- The measure of the smaller interior angle
  small_angle : ℝ
  -- The measure of the larger interior angle
  large_angle : ℝ
  -- The ratio of the angles is 1:3
  angle_ratio : small_angle * 3 = large_angle
  -- The angles are supplementary (add up to 180°)
  supplementary : small_angle + large_angle = 180

/-- The theorem stating that the smaller angle in the parallelogram is 45° -/
theorem smaller_angle_is_45_degrees (p : AngleRatioParallelogram) : p.small_angle = 45 := by
  sorry


end NUMINAMATH_CALUDE_smaller_angle_is_45_degrees_l3301_330157


namespace NUMINAMATH_CALUDE_min_value_theorem_l3301_330129

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ (2 + Real.rpow 4 (1/3)) / Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3301_330129


namespace NUMINAMATH_CALUDE_smallest_stairs_l3301_330150

theorem smallest_stairs (n : ℕ) : 
  (n > 10) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 10 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_stairs_l3301_330150


namespace NUMINAMATH_CALUDE_salary_increase_l3301_330136

/-- Regression line for worker's salary with respect to labor productivity -/
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

/-- Theorem: When labor productivity increases by 1000 Yuan (1 unit in x), 
    the salary increases by 90 Yuan -/
theorem salary_increase (x : ℝ) : 
  regression_line (x + 1) - regression_line x = 90 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3301_330136


namespace NUMINAMATH_CALUDE_bicycle_time_saved_l3301_330139

/-- The time in minutes it takes Mike to walk to school -/
def walking_time : ℕ := 98

/-- The time in minutes Mike saved by riding a bicycle -/
def time_saved : ℕ := 34

/-- Theorem: The time saved by riding a bicycle compared to walking is 34 minutes -/
theorem bicycle_time_saved : time_saved = 34 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_time_saved_l3301_330139


namespace NUMINAMATH_CALUDE_water_jars_problem_l3301_330146

/-- Proves that given 28 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars, the total number of water-filled jars is 48. -/
theorem water_jars_problem (total_water : ℚ) (num_each_jar : ℕ) : 
  total_water = 28 →
  (1/4 : ℚ) * num_each_jar + (1/2 : ℚ) * num_each_jar + 1 * num_each_jar = total_water →
  3 * num_each_jar = 48 := by
  sorry

end NUMINAMATH_CALUDE_water_jars_problem_l3301_330146


namespace NUMINAMATH_CALUDE_scholarship_sum_l3301_330159

theorem scholarship_sum (wendy kelly nina : ℕ) : 
  wendy = 20000 →
  kelly = 2 * wendy →
  nina = kelly - 8000 →
  wendy + kelly + nina = 92000 := by
  sorry

end NUMINAMATH_CALUDE_scholarship_sum_l3301_330159


namespace NUMINAMATH_CALUDE_lucca_bread_problem_l3301_330113

/-- The fraction of remaining bread Lucca ate on the second day -/
def second_day_fraction (initial_bread : ℕ) (first_day_fraction : ℚ) (third_day_fraction : ℚ) (remaining_bread : ℕ) : ℚ :=
  let remaining_after_first := initial_bread - initial_bread * first_day_fraction
  2 / 5

/-- Theorem stating the fraction of remaining bread Lucca ate on the second day -/
theorem lucca_bread_problem (initial_bread : ℕ) (first_day_fraction : ℚ) (third_day_fraction : ℚ) (remaining_bread : ℕ)
    (h1 : initial_bread = 200)
    (h2 : first_day_fraction = 1 / 4)
    (h3 : third_day_fraction = 1 / 2)
    (h4 : remaining_bread = 45) :
  second_day_fraction initial_bread first_day_fraction third_day_fraction remaining_bread = 2 / 5 := by
  sorry

#eval second_day_fraction 200 (1/4) (1/2) 45

end NUMINAMATH_CALUDE_lucca_bread_problem_l3301_330113


namespace NUMINAMATH_CALUDE_chinese_character_equation_l3301_330166

theorem chinese_character_equation :
  ∃! (a b c d : Nat),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    d + d + d + d = 48 ∧
    1000 * a + 100 * b + 10 * c + d = 1468 :=
by
  sorry

end NUMINAMATH_CALUDE_chinese_character_equation_l3301_330166


namespace NUMINAMATH_CALUDE_park_visitors_l3301_330165

theorem park_visitors (bike_riders : ℕ) (hikers : ℕ) : 
  bike_riders = 249 →
  hikers = bike_riders + 178 →
  bike_riders + hikers = 676 :=
by sorry

end NUMINAMATH_CALUDE_park_visitors_l3301_330165


namespace NUMINAMATH_CALUDE_intersection_line_l3301_330125

/-- The line of intersection of two planes -/
def line_of_intersection (t : ℝ) : ℝ × ℝ × ℝ := (t, 2 - t, t + 1)

/-- First plane equation -/
def plane1 (x y z : ℝ) : Prop := 2 * x - y - 3 * z + 5 = 0

/-- Second plane equation -/
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

theorem intersection_line (t : ℝ) :
  let (x, y, z) := line_of_intersection t
  plane1 x y z ∧ plane2 x y z := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_l3301_330125


namespace NUMINAMATH_CALUDE_function_lower_bound_l3301_330199

open Real

theorem function_lower_bound (x : ℝ) (h : x > 0) : Real.exp x - Real.log x > 2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3301_330199


namespace NUMINAMATH_CALUDE_food_box_shipment_l3301_330140

theorem food_box_shipment (total_food : ℝ) (max_shipping_weight : ℝ) :
  total_food = 777.5 ∧ max_shipping_weight = 2 →
  ⌊total_food / max_shipping_weight⌋ = 388 := by
  sorry

end NUMINAMATH_CALUDE_food_box_shipment_l3301_330140


namespace NUMINAMATH_CALUDE_peach_difference_l3301_330190

/-- Given a basket of peaches with specific counts for each color, 
    prove the difference between green and red peaches. -/
theorem peach_difference (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h_red : red = 7)
  (h_yellow : yellow = 71)
  (h_green : green = 8) :
  green - red = 1 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3301_330190


namespace NUMINAMATH_CALUDE_rabbit_population_estimate_l3301_330132

/-- Capture-recapture estimation of rabbit population -/
theorem rabbit_population_estimate :
  ∀ (total_population : ℕ)
    (first_capture second_capture recaptured_tagged : ℕ),
  first_capture = 10 →
  second_capture = 10 →
  recaptured_tagged = 2 →
  total_population = (first_capture * second_capture) / recaptured_tagged →
  total_population = 50 :=
by
  sorry

#check rabbit_population_estimate

end NUMINAMATH_CALUDE_rabbit_population_estimate_l3301_330132


namespace NUMINAMATH_CALUDE_pool_filling_time_l3301_330158

/-- Proves that it takes 33 hours to fill a 30,000-gallon pool with 5 hoses supplying 3 gallons per minute each -/
theorem pool_filling_time : 
  let pool_capacity : ℕ := 30000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℕ := 3
  let minutes_per_hour : ℕ := 60
  let total_flow_rate_per_hour : ℕ := num_hoses * flow_rate_per_hose * minutes_per_hour
  let filling_time_hours : ℕ := pool_capacity / total_flow_rate_per_hour
  filling_time_hours = 33 := by
  sorry


end NUMINAMATH_CALUDE_pool_filling_time_l3301_330158


namespace NUMINAMATH_CALUDE_line_slope_angle_l3301_330173

theorem line_slope_angle (x y : ℝ) : 
  x + Real.sqrt 3 * y = 0 → 
  Real.tan (150 * π / 180) = -(1 / Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l3301_330173


namespace NUMINAMATH_CALUDE_limit_exists_l3301_330185

/-- Prove the existence of δ(ε) for the limit of (5x^2 - 24x - 5) / (x - 5) as x approaches 5 -/
theorem limit_exists (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, x ≠ 5 → |x - 5| < δ →
    |(5 * x^2 - 24 * x - 5) / (x - 5) - 26| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_exists_l3301_330185


namespace NUMINAMATH_CALUDE_english_speakers_l3301_330154

theorem english_speakers (total : ℕ) (hindi : ℕ) (both : ℕ) (english : ℕ) : 
  total = 40 → 
  hindi = 30 → 
  both ≥ 10 → 
  total = hindi + english - both → 
  english = 20 := by
sorry

end NUMINAMATH_CALUDE_english_speakers_l3301_330154


namespace NUMINAMATH_CALUDE_expression_factorization_l3301_330194

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 - 9) = 6 * x^4 * (2 * x^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3301_330194


namespace NUMINAMATH_CALUDE_inequality_proof_l3301_330160

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3301_330160


namespace NUMINAMATH_CALUDE_monster_family_kids_eyes_l3301_330197

/-- Represents a monster family with parents and kids -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  total_eyes : ℕ

/-- Calculates the number of eyes each kid has in a monster family -/
def eyes_per_kid (family : MonsterFamily) : ℕ :=
  (family.total_eyes - family.mom_eyes - family.dad_eyes) / family.num_kids

/-- Theorem: In the specific monster family, each kid has 4 eyes -/
theorem monster_family_kids_eyes :
  let family : MonsterFamily := {
    mom_eyes := 1,
    dad_eyes := 3,
    num_kids := 3,
    total_eyes := 16
  }
  eyes_per_kid family = 4 := by sorry

end NUMINAMATH_CALUDE_monster_family_kids_eyes_l3301_330197


namespace NUMINAMATH_CALUDE_tea_leaf_problem_l3301_330189

theorem tea_leaf_problem (num_plants : ℕ) (remaining_fraction : ℚ) (total_remaining : ℕ) :
  num_plants = 3 →
  remaining_fraction = 2/3 →
  total_remaining = 36 →
  ∃ initial_per_plant : ℕ,
    initial_per_plant * num_plants * remaining_fraction = total_remaining ∧
    initial_per_plant = 18 :=
by sorry

end NUMINAMATH_CALUDE_tea_leaf_problem_l3301_330189


namespace NUMINAMATH_CALUDE_gcd_1215_1995_l3301_330181

theorem gcd_1215_1995 : Nat.gcd 1215 1995 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1215_1995_l3301_330181


namespace NUMINAMATH_CALUDE_elvins_phone_bill_l3301_330143

/-- Elvin's monthly telephone bill -/
def monthly_bill (call_charge : ℕ) (internet_charge : ℕ) : ℕ :=
  call_charge + internet_charge

theorem elvins_phone_bill 
  (internet_charge : ℕ) 
  (first_month_call_charge : ℕ) 
  (h1 : monthly_bill first_month_call_charge internet_charge = 50)
  (h2 : monthly_bill (2 * first_month_call_charge) internet_charge = 76) :
  monthly_bill (2 * first_month_call_charge) internet_charge = 76 :=
by
  sorry

#check elvins_phone_bill

end NUMINAMATH_CALUDE_elvins_phone_bill_l3301_330143


namespace NUMINAMATH_CALUDE_beam_buying_problem_l3301_330128

/-- Represents the problem of buying beams as described in "Si Yuan Yu Jian" -/
theorem beam_buying_problem (x : ℕ) :
  (3 * x * (x - 1) = 6210) ↔
  (x > 0 ∧
   3 * x = 6210 / x +
   3 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_beam_buying_problem_l3301_330128


namespace NUMINAMATH_CALUDE_sum_first_three_terms_l3301_330134

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem sum_first_three_terms (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 4 = 8 ∧
  arithmetic_sequence a d 5 = 12 ∧
  arithmetic_sequence a d 6 = 16 →
  arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_first_three_terms_l3301_330134


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3301_330195

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  Nat.factorial total / Nat.factorial identical

/-- Theorem: Arranging 6 books with 3 identical copies results in 120 ways -/
theorem book_arrangement_theorem :
  arrange_books 6 3 = 120 := by
  sorry

#eval arrange_books 6 3

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3301_330195


namespace NUMINAMATH_CALUDE_increasing_function_range_l3301_330114

theorem increasing_function_range (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → x ∈ [-1, 3] → y ∈ [-1, 3] → f x < f y) :
  ∀ a : ℝ, f a > f (1 - 2 * a) → a ∈ Set.Ioo (1/3) 1 := by
sorry

end NUMINAMATH_CALUDE_increasing_function_range_l3301_330114


namespace NUMINAMATH_CALUDE_mixture_volume_l3301_330112

/-- Given a mixture of liquids p and q with an initial ratio and a change in ratio after adding more of q, 
    calculate the initial volume of the mixture. -/
theorem mixture_volume (initial_p initial_q added_q : ℝ) 
  (h1 : initial_p / initial_q = 4 / 3) 
  (h2 : initial_p / (initial_q + added_q) = 5 / 7)
  (h3 : added_q = 13) : 
  initial_p + initial_q = 35 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_l3301_330112


namespace NUMINAMATH_CALUDE_solution_t_l3301_330172

theorem solution_t (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) → t = 37/10 := by
  sorry

end NUMINAMATH_CALUDE_solution_t_l3301_330172


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l3301_330171

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem two_digit_number_puzzle (n : ℕ) :
  is_two_digit n ∧ 
  (digit_sum n) % 3 = 0 ∧ 
  n - 27 = reverse_digits n → 
  n = 63 ∨ n = 96 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l3301_330171


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3301_330135

theorem cubic_root_sum (a b c : ℝ) (p q r : ℕ+) : 
  a^3 - 3*a^2 - 7*a - 1 = 0 →
  b^3 - 3*b^2 - 7*b - 1 = 0 →
  c^3 - 3*c^2 - 7*c - 1 = 0 →
  a ≠ b →
  b ≠ c →
  c ≠ a →
  (1 / (a^(1/3) - b^(1/3)) + 1 / (b^(1/3) - c^(1/3)) + 1 / (c^(1/3) - a^(1/3)))^2 = p * q^(1/3) / r →
  Nat.gcd p.val r.val = 1 →
  ∀ (prime : ℕ), prime.Prime → ¬(∃ (k : ℕ), q = prime^3 * k) →
  100 * p + 10 * q + r = 1913 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3301_330135


namespace NUMINAMATH_CALUDE_range_of_a_l3301_330110

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) : p a ∧ q a → a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3301_330110


namespace NUMINAMATH_CALUDE_king_queen_prob_l3301_330188

/-- Represents a standard deck of cards -/
def StandardDeck : Type := Unit

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The number of Queens in a standard deck -/
def numQueens : ℕ := 4

/-- Calculates the probability of drawing a King followed by a Queen from a standard deck -/
def probKingQueen (deck : StandardDeck) : ℚ :=
  (numKings * numQueens : ℚ) / (deckSize * (deckSize - 1))

/-- Theorem stating that the probability of drawing a King followed by a Queen is 4/663 -/
theorem king_queen_prob : 
  ∀ (deck : StandardDeck), probKingQueen deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_king_queen_prob_l3301_330188


namespace NUMINAMATH_CALUDE_ContrapositiveDual_l3301_330176

-- Define what it means for a number to be even
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k

-- The original proposition
def OriginalProposition : Prop :=
  ∀ a b : Int, IsEven a ∧ IsEven b → IsEven (a + b)

-- The contrapositive we want to prove
def Contrapositive : Prop :=
  ∀ a b : Int, ¬IsEven (a + b) → ¬(IsEven a ∧ IsEven b)

-- The theorem stating that the contrapositive is correct
theorem ContrapositiveDual : OriginalProposition ↔ Contrapositive := by
  sorry

end NUMINAMATH_CALUDE_ContrapositiveDual_l3301_330176


namespace NUMINAMATH_CALUDE_age_ratio_correct_l3301_330151

/-- Represents the ages and relationship between a mother and daughter -/
structure FamilyAges where
  mother_current_age : ℕ
  daughter_future_age : ℕ
  years_to_future : ℕ
  multiple : ℝ

/-- Calculates the ratio of mother's age to daughter's age at a past time -/
def age_ratio (f : FamilyAges) : ℝ × ℝ :=
  (f.multiple, 1)

/-- Theorem stating that the age ratio is correct given the family ages -/
theorem age_ratio_correct (f : FamilyAges) 
  (h1 : f.mother_current_age = 41)
  (h2 : f.daughter_future_age = 26)
  (h3 : f.years_to_future = 3)
  (h4 : ∃ (x : ℕ), f.mother_current_age - x = f.multiple * (f.daughter_future_age - f.years_to_future - x)) :
  age_ratio f = (f.multiple, 1) := by
  sorry

#check age_ratio_correct

end NUMINAMATH_CALUDE_age_ratio_correct_l3301_330151


namespace NUMINAMATH_CALUDE_three_squared_sum_equals_three_cubed_l3301_330177

theorem three_squared_sum_equals_three_cubed (a : ℕ) :
  3^2 + 3^2 + 3^2 = 3^a → a = 3 := by
sorry

end NUMINAMATH_CALUDE_three_squared_sum_equals_three_cubed_l3301_330177


namespace NUMINAMATH_CALUDE_third_difference_of_cubic_is_six_l3301_330144

/-- Finite difference operator -/
def finiteDifference (f : ℕ → ℝ) : ℕ → ℝ := fun n ↦ f (n + 1) - f n

/-- Third finite difference -/
def thirdFiniteDifference (f : ℕ → ℝ) : ℕ → ℝ :=
  finiteDifference (finiteDifference (finiteDifference f))

/-- Cubic function -/
def cubicFunction : ℕ → ℝ := fun n ↦ (n : ℝ) ^ 3

theorem third_difference_of_cubic_is_six :
  ∀ n, thirdFiniteDifference cubicFunction n = 6 := by sorry

end NUMINAMATH_CALUDE_third_difference_of_cubic_is_six_l3301_330144


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l3301_330191

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ
  -- The radius of the tangent circle
  s : ℝ
  -- The circle is tangent to three sides
  tangent_to_sides : length = 2 * s ∧ width = s
  -- The circle passes through the midpoint of the diagonal
  passes_through_midpoint : s = Real.sqrt (s^2 + (length/2)^2)

/-- The area of a TangentCircleRectangle is 2s^2 -/
theorem tangent_circle_rectangle_area (r : TangentCircleRectangle) : 
  r.length * r.width = 2 * r.s^2 := by
  sorry

#check tangent_circle_rectangle_area

end NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l3301_330191


namespace NUMINAMATH_CALUDE_four_variable_inequality_l3301_330142

theorem four_variable_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)^2 ≤ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_four_variable_inequality_l3301_330142


namespace NUMINAMATH_CALUDE_cooking_participants_l3301_330161

/-- The number of people who practice yoga -/
def yoga : ℕ := 35

/-- The number of people who study weaving -/
def weaving : ℕ := 15

/-- The number of people who study cooking only -/
def cooking_only : ℕ := 7

/-- The number of people who study both cooking and yoga -/
def cooking_and_yoga : ℕ := 5

/-- The number of people who participate in all curriculums -/
def all_curriculums : ℕ := 3

/-- The number of people who study both cooking and weaving -/
def cooking_and_weaving : ℕ := 5

/-- The total number of people who study cooking -/
def total_cooking : ℕ := cooking_only + (cooking_and_yoga - all_curriculums) + (cooking_and_weaving - all_curriculums) + all_curriculums

theorem cooking_participants : total_cooking = 14 := by
  sorry

end NUMINAMATH_CALUDE_cooking_participants_l3301_330161


namespace NUMINAMATH_CALUDE_angle_through_point_l3301_330164

theorem angle_through_point (α : Real) : 
  0 ≤ α → α < 2 * Real.pi → 
  let P : Real × Real := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (Real.cos α = P.1 ∧ Real.sin α = P.2) →
  α = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l3301_330164


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3301_330103

/-- Represents a stratified sample from a high school population -/
structure StratifiedSample where
  total_students : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sample_size : ℕ
  sampled_freshmen : ℕ
  sampled_sophomores : ℕ
  sampled_juniors : ℕ

/-- Checks if a stratified sample is valid according to the problem conditions -/
def is_valid_sample (s : StratifiedSample) : Prop :=
  s.total_students = 2000 ∧
  s.freshmen = 800 ∧
  s.sophomores = 600 ∧
  s.juniors = 600 ∧
  s.sample_size = 50 ∧
  s.sampled_freshmen + s.sampled_sophomores + s.sampled_juniors = s.sample_size

/-- Theorem stating that the correct stratified sample is 20 freshmen, 15 sophomores, and 15 juniors -/
theorem correct_stratified_sample (s : StratifiedSample) :
  is_valid_sample s →
  s.sampled_freshmen = 20 ∧ s.sampled_sophomores = 15 ∧ s.sampled_juniors = 15 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l3301_330103


namespace NUMINAMATH_CALUDE_time_to_run_around_field_l3301_330107

-- Define the side length of the square field
def side_length : ℝ := 50

-- Define the boy's running speed in km/hr
def running_speed : ℝ := 9

-- Theorem statement
theorem time_to_run_around_field : 
  let perimeter : ℝ := 4 * side_length
  let speed_in_mps : ℝ := running_speed * 1000 / 3600
  let time : ℝ := perimeter / speed_in_mps
  time = 80 := by sorry

end NUMINAMATH_CALUDE_time_to_run_around_field_l3301_330107


namespace NUMINAMATH_CALUDE_students_in_class_g_l3301_330186

theorem students_in_class_g (total_students : ℕ) (class_a class_b class_c class_d class_e class_f class_g : ℕ) : 
  total_students = 1500 ∧
  class_a = 188 ∧
  class_b = 115 ∧
  class_c = class_b + 80 ∧
  class_d = 2 * class_b ∧
  class_e = class_a + class_b ∧
  class_f = (class_c + class_d) / 2 ∧
  class_g = total_students - (class_a + class_b + class_c + class_d + class_e + class_f) →
  class_g = 256 :=
by sorry

end NUMINAMATH_CALUDE_students_in_class_g_l3301_330186
