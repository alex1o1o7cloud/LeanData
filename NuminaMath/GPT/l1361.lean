import Mathlib

namespace NUMINAMATH_GPT_probability_function_has_zero_point_l1361_136136

noncomputable def probability_of_zero_point : ℚ :=
by
  let S := ({-1, 1, 2} : Finset ℤ).product ({-1, 1, 2} : Finset ℤ)
  let zero_point_pairs := S.filter (λ p => (p.1 * p.2 ≤ 1))
  let favorable_outcomes := zero_point_pairs.card
  let total_outcomes := S.card
  exact favorable_outcomes / total_outcomes

theorem probability_function_has_zero_point :
  probability_of_zero_point = (2 / 3 : ℚ) :=
  sorry

end NUMINAMATH_GPT_probability_function_has_zero_point_l1361_136136


namespace NUMINAMATH_GPT_g_min_value_l1361_136147

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end NUMINAMATH_GPT_g_min_value_l1361_136147


namespace NUMINAMATH_GPT_n_value_l1361_136159

theorem n_value (n : ℕ) (h1 : ∃ a b : ℕ, a = (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7 ∧ b = 2 * n ∧ a ^ 2 - b ^ 2 = 0) : n = 10 := 
  by sorry

end NUMINAMATH_GPT_n_value_l1361_136159


namespace NUMINAMATH_GPT_smallest_next_divisor_l1361_136119

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor (m : ℕ) (h_even : is_even m)
  (h_four_digit : is_four_digit m)
  (h_div_437 : is_divisor 437 m) :
  ∃ next_div : ℕ, next_div > 437 ∧ is_divisor next_div m ∧ 
  ∀ d, d > 437 ∧ is_divisor d m → next_div ≤ d :=
sorry

end NUMINAMATH_GPT_smallest_next_divisor_l1361_136119


namespace NUMINAMATH_GPT_find_d_l1361_136173

theorem find_d (d : ℝ) (h1 : ∃ (x y : ℝ), y = x + d ∧ x = -y + d ∧ x = d-1 ∧ y = d) : d = 1 :=
sorry

end NUMINAMATH_GPT_find_d_l1361_136173


namespace NUMINAMATH_GPT_average_computation_l1361_136133

variable {a b c X Y Z : ℝ}

theorem average_computation 
  (h1 : a + b + c = 15)
  (h2 : X + Y + Z = 21) :
  ((2 * a + 3 * X) + (2 * b + 3 * Y) + (2 * c + 3 * Z)) / 3 = 31 :=
by
  sorry

end NUMINAMATH_GPT_average_computation_l1361_136133


namespace NUMINAMATH_GPT_smallest_integer_with_divisors_l1361_136122

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end NUMINAMATH_GPT_smallest_integer_with_divisors_l1361_136122


namespace NUMINAMATH_GPT_final_amount_l1361_136170

-- Definitions for the initial amount, price per pound, and quantity purchased.
def initial_amount : ℕ := 20
def price_per_pound : ℕ := 2
def quantity_purchased : ℕ := 3

-- Formalizing the statement
theorem final_amount (A P Q : ℕ) (hA : A = initial_amount) (hP : P = price_per_pound) (hQ : Q = quantity_purchased) :
  A - P * Q = 14 :=
by
  sorry

end NUMINAMATH_GPT_final_amount_l1361_136170


namespace NUMINAMATH_GPT_evaluate_at_3_l1361_136103

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 + x + 6

theorem evaluate_at_3 : g 3 = 135 := 
  by
  sorry

end NUMINAMATH_GPT_evaluate_at_3_l1361_136103


namespace NUMINAMATH_GPT_smallest_number_of_white_marbles_l1361_136102

theorem smallest_number_of_white_marbles
  (n : ℕ)
  (hn1 : n > 0)
  (orange_marbles : ℕ := n / 5)
  (hn_orange : n % 5 = 0)
  (purple_marbles : ℕ := n / 6)
  (hn_purple : n % 6 = 0)
  (green_marbles : ℕ := 9)
  : (n - (orange_marbles + purple_marbles + green_marbles)) = 10 → n = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_white_marbles_l1361_136102


namespace NUMINAMATH_GPT_largest_expression_l1361_136150

def U := 2 * 2004^2005
def V := 2004^2005
def W := 2003 * 2004^2004
def X := 2 * 2004^2004
def Y := 2004^2004
def Z := 2004^2003

theorem largest_expression :
  U - V > V - W ∧
  U - V > W - X ∧
  U - V > X - Y ∧
  U - V > Y - Z :=
by
  sorry

end NUMINAMATH_GPT_largest_expression_l1361_136150


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1361_136109

def ellipse_condition (m : ℝ) : Prop :=
  m + 1 > 4 - m ∧ 4 - m > 0

def circle_condition (m : ℝ) : Prop :=
  m^2 - 4 > 0

theorem problem_part1 (m : ℝ) :
  ellipse_condition m → (3 / 2 < m ∧ m < 4) :=
sorry

theorem problem_part2 (m : ℝ) :
  ellipse_condition m ∧ circle_condition m → (2 < m ∧ m < 4) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1361_136109


namespace NUMINAMATH_GPT_bottles_left_after_purchase_l1361_136110

def initial_bottles : ℕ := 35
def jason_bottles : ℕ := 5
def harry_bottles : ℕ := 6
def jason_effective_bottles (n : ℕ) : ℕ := n  -- Jason buys 5 bottles
def harry_effective_bottles (n : ℕ) : ℕ := n + 1 -- Harry gets one additional free bottle

theorem bottles_left_after_purchase (j_b h_b i_b : ℕ) (j_effective h_effective : ℕ → ℕ) :
  j_b = 5 → h_b = 6 → i_b = 35 → j_effective j_b = 5 → h_effective h_b = 7 →
  i_b - (j_effective j_b + h_effective h_b) = 23 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bottles_left_after_purchase_l1361_136110


namespace NUMINAMATH_GPT_find_m_l1361_136111

theorem find_m (m : ℝ) :
  (m - 2013 = 0) → (m = 2013) ∧ (m - 1 ≠ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l1361_136111


namespace NUMINAMATH_GPT_find_a_l1361_136184

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a^2 - 1

theorem find_a (a : ℝ) (h : ∀ x ∈ (Set.Icc 1 2), f x a ≤ 16 ∧ ∃ y ∈ (Set.Icc 1 2), f y a = 16) : a = 3 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1361_136184


namespace NUMINAMATH_GPT_ratio_of_black_to_white_after_border_l1361_136156

def original_tiles (black white : ℕ) : Prop := black = 14 ∧ white = 21
def original_dimensions (length width : ℕ) : Prop := length = 5 ∧ width = 7

def border_added (length width l w : ℕ) : Prop := l = length + 2 ∧ w = width + 2

def total_white_tiles (initial_white new_white total_white : ℕ) : Prop :=
  total_white = initial_white + new_white

def black_white_ratio (black_tiles white_tiles : ℕ) (ratio : ℚ) : Prop :=
  ratio = black_tiles / white_tiles

theorem ratio_of_black_to_white_after_border 
  (black_white_tiles : ℕ → ℕ → Prop)
  (dimensions : ℕ → ℕ → Prop)
  (border : ℕ → ℕ → ℕ → ℕ → Prop)
  (total_white : ℕ → ℕ → ℕ → Prop)
  (ratio : ℕ → ℕ → ℚ → Prop)
  (black_tiles white_tiles initial_white total_white_new length width l w : ℕ)
  (rat : ℚ) :
  black_white_tiles black_tiles initial_white →
  dimensions length width →
  border length width l w →
  total_white initial_white (l * w - length * width) white_tiles →
  ratio black_tiles white_tiles rat →
  rat = 2 / 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_of_black_to_white_after_border_l1361_136156


namespace NUMINAMATH_GPT_circle_chords_intersect_radius_square_l1361_136157

theorem circle_chords_intersect_radius_square
  (r : ℝ) -- The radius of the circle
  (AB CD BP : ℝ) -- The lengths of chords AB, CD, and segment BP
  (angle_APD : ℝ) -- The angle ∠APD in degrees
  (AB_len : AB = 8)
  (CD_len : CD = 12)
  (BP_len : BP = 10)
  (angle_APD_val : angle_APD = 60) :
  r^2 = 91 := 
sorry

end NUMINAMATH_GPT_circle_chords_intersect_radius_square_l1361_136157


namespace NUMINAMATH_GPT_compute_fg_l1361_136118

def g (x : ℕ) : ℕ := 2 * x + 6
def f (x : ℕ) : ℕ := 4 * x - 8
def x : ℕ := 10

theorem compute_fg : f (g x) = 96 := by
  sorry

end NUMINAMATH_GPT_compute_fg_l1361_136118


namespace NUMINAMATH_GPT_nesbitts_inequality_l1361_136193

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_nesbitts_inequality_l1361_136193


namespace NUMINAMATH_GPT_eval_expression_l1361_136100

theorem eval_expression : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := 
  sorry

end NUMINAMATH_GPT_eval_expression_l1361_136100


namespace NUMINAMATH_GPT_simplify_expression_l1361_136140

theorem simplify_expression : (8^(1/3) / 8^(1/6)) = 8^(1/6) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1361_136140


namespace NUMINAMATH_GPT_commission_percentage_l1361_136153

-- Define the given conditions
def cost_of_item : ℝ := 17
def observed_price : ℝ := 25.50
def desired_profit_percentage : ℝ := 0.20

-- Calculate the desired profit in dollars
def desired_profit : ℝ := desired_profit_percentage * cost_of_item

-- Calculate the total desired price for the distributor
def total_desired_price : ℝ := cost_of_item + desired_profit

-- Calculate the commission in dollars
def commission_in_dollars : ℝ := observed_price - total_desired_price

-- Prove that commission percentage taken by the online store is 20%
theorem commission_percentage :
  (commission_in_dollars / observed_price) * 100 = 20 := 
by
  -- This is the placeholder for the proof
  sorry

end NUMINAMATH_GPT_commission_percentage_l1361_136153


namespace NUMINAMATH_GPT_Evelyn_bottle_caps_problem_l1361_136145

theorem Evelyn_bottle_caps_problem (E : ℝ) (H1 : E - 18.0 = 45) : E = 63.0 := 
by
  sorry


end NUMINAMATH_GPT_Evelyn_bottle_caps_problem_l1361_136145


namespace NUMINAMATH_GPT_kurt_savings_l1361_136117

def daily_cost_old : ℝ := 0.85
def daily_cost_new : ℝ := 0.45
def days : ℕ := 30

theorem kurt_savings : (daily_cost_old * days) - (daily_cost_new * days) = 12.00 := by
  sorry

end NUMINAMATH_GPT_kurt_savings_l1361_136117


namespace NUMINAMATH_GPT_Jason_earned_60_dollars_l1361_136168

-- Define initial and final amounts of money
variable (Jason_initial Jason_final : ℕ)

-- State the assumption about Jason's initial and final amounts of money
variable (h_initial : Jason_initial = 3) (h_final : Jason_final = 63)

-- Define the amount of money Jason earned
def Jason_earn := Jason_final - Jason_initial

-- Prove that Jason earned 60 dollars by delivering newspapers
theorem Jason_earned_60_dollars : Jason_earn Jason_initial Jason_final = 60 := by
  sorry

end NUMINAMATH_GPT_Jason_earned_60_dollars_l1361_136168


namespace NUMINAMATH_GPT_job_candidates_excel_nights_l1361_136194

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end NUMINAMATH_GPT_job_candidates_excel_nights_l1361_136194


namespace NUMINAMATH_GPT_binom_30_3_is_4060_l1361_136174

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_GPT_binom_30_3_is_4060_l1361_136174


namespace NUMINAMATH_GPT_automobile_travel_distance_l1361_136113

theorem automobile_travel_distance (a r : ℝ) :
  (2 * a / 5) / (2 * r) * 5 * 60 / 3 = 20 * a / r :=
by 
  -- skipping proof details
  sorry

end NUMINAMATH_GPT_automobile_travel_distance_l1361_136113


namespace NUMINAMATH_GPT_num_intersecting_chords_on_circle_l1361_136130

theorem num_intersecting_chords_on_circle (points : Fin 20 → Prop) : 
  ∃ num_chords : ℕ, num_chords = 156180 :=
by
  sorry

end NUMINAMATH_GPT_num_intersecting_chords_on_circle_l1361_136130


namespace NUMINAMATH_GPT_largest_square_not_divisible_by_100_l1361_136101

theorem largest_square_not_divisible_by_100
  (n : ℕ) (h1 : ∃ a : ℕ, a^2 = n) 
  (h2 : n % 100 ≠ 0)
  (h3 : ∃ m : ℕ, m * 100 + n % 100 = n ∧ ∃ b : ℕ, b^2 = m) :
  n = 1681 := sorry

end NUMINAMATH_GPT_largest_square_not_divisible_by_100_l1361_136101


namespace NUMINAMATH_GPT_cab_to_bus_ratio_l1361_136112

noncomputable def train_distance : ℤ := 300
noncomputable def bus_distance : ℤ := train_distance / 2
noncomputable def total_distance : ℤ := 500
noncomputable def cab_distance : ℤ := total_distance - (train_distance + bus_distance)
noncomputable def ratio : ℚ := cab_distance / bus_distance

theorem cab_to_bus_ratio :
  ratio = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_cab_to_bus_ratio_l1361_136112


namespace NUMINAMATH_GPT_lcm_of_ratio_hcf_l1361_136160

theorem lcm_of_ratio_hcf {a b : ℕ} (ratioCond : a = 14 * 28) (ratioCond2 : b = 21 * 28) (hcfCond : Nat.gcd a b = 28) : Nat.lcm a b = 1176 := by
  sorry

end NUMINAMATH_GPT_lcm_of_ratio_hcf_l1361_136160


namespace NUMINAMATH_GPT_pencil_pen_cost_l1361_136149

theorem pencil_pen_cost 
  (p q : ℝ) 
  (h1 : 6 * p + 3 * q = 3.90) 
  (h2 : 2 * p + 5 * q = 4.45) :
  3 * p + 4 * q = 3.92 :=
by
  sorry

end NUMINAMATH_GPT_pencil_pen_cost_l1361_136149


namespace NUMINAMATH_GPT_division_remainder_l1361_136175

noncomputable def remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem division_remainder :
  remainder (Polynomial.X ^ 3) (Polynomial.X ^ 2 + 7 * Polynomial.X + 2) = 47 * Polynomial.X + 14 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1361_136175


namespace NUMINAMATH_GPT_count_8_digit_odd_last_l1361_136195

-- Define the constraints for the digits of the 8-digit number
def first_digit_choices := 9
def next_six_digits_choices := 10 ^ 6
def last_digit_choices := 5

-- State the theorem based on the given conditions and the solution
theorem count_8_digit_odd_last : first_digit_choices * next_six_digits_choices * last_digit_choices = 45000000 :=
by
  sorry

end NUMINAMATH_GPT_count_8_digit_odd_last_l1361_136195


namespace NUMINAMATH_GPT_exists_prime_and_positive_integer_l1361_136171

theorem exists_prime_and_positive_integer (a : ℕ) (h : a = 9) : 
  ∃ (p : ℕ) (hp : Nat.Prime p) (b : ℕ) (hb : b ≥ 2), (a^p - a) / p = b^2 := 
  by
  sorry

end NUMINAMATH_GPT_exists_prime_and_positive_integer_l1361_136171


namespace NUMINAMATH_GPT_calculate_x_l1361_136154

theorem calculate_x : 121 + 2 * 11 * 8 + 64 = 361 :=
by
  sorry

end NUMINAMATH_GPT_calculate_x_l1361_136154


namespace NUMINAMATH_GPT_sum_of_fractions_is_514_l1361_136120

theorem sum_of_fractions_is_514 : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 5 / 14 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_is_514_l1361_136120


namespace NUMINAMATH_GPT_arithmetic_sequence_a15_l1361_136186

variable {α : Type*} [LinearOrderedField α]

-- Conditions for the arithmetic sequence
variable (a : ℕ → α)
variable (d : α)
variable (a1 : α)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 5)
variable (h_a10 : a 10 = 15)

-- To prove that a15 = 25
theorem arithmetic_sequence_a15 : a 15 = 25 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a15_l1361_136186


namespace NUMINAMATH_GPT_sum_of_ages_l1361_136178

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1361_136178


namespace NUMINAMATH_GPT_problem1_problem2_l1361_136177

-- Define sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x : ℝ | x < -2 ∨ x > 6}

-- Define the two proof problems as Lean statements
theorem problem1 (a : ℝ) : setA a ∩ setB = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

theorem problem2 (a : ℝ) : setA a ⊆ setB ↔ (a < -5 ∨ a > 6) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1361_136177


namespace NUMINAMATH_GPT_track_meet_total_people_l1361_136197

theorem track_meet_total_people (B G : ℕ) (H1 : B = 30)
  (H2 : ∃ G, (3 * G) / 5 + (2 * G) / 5 = G)
  (H3 : ∀ G, 2 * G / 5 = 10) :
  B + G = 55 :=
by
  sorry

end NUMINAMATH_GPT_track_meet_total_people_l1361_136197


namespace NUMINAMATH_GPT_standard_parts_bounds_l1361_136187

noncomputable def n : ℕ := 900
noncomputable def p : ℝ := 0.9
noncomputable def confidence_level : ℝ := 0.95
noncomputable def lower_bound : ℝ := 792
noncomputable def upper_bound : ℝ := 828

theorem standard_parts_bounds : 
  792 ≤ n * p - 1.96 * (n * p * (1 - p)).sqrt ∧ 
  n * p + 1.96 * (n * p * (1 - p)).sqrt ≤ 828 :=
sorry

end NUMINAMATH_GPT_standard_parts_bounds_l1361_136187


namespace NUMINAMATH_GPT_cooper_saved_days_l1361_136127

variable (daily_saving : ℕ) (total_saving : ℕ) (n : ℕ)

-- Conditions
def cooper_saved (daily_saving total_saving n : ℕ) : Prop :=
  total_saving = daily_saving * n

-- Theorem stating the question equals the correct answer
theorem cooper_saved_days :
  cooper_saved 34 12410 365 :=
by
  sorry

end NUMINAMATH_GPT_cooper_saved_days_l1361_136127


namespace NUMINAMATH_GPT_fraction_value_l1361_136126

theorem fraction_value (a b : ℚ) (h₁ : b / (a - 2) = 3 / 4) (h₂ : b / (a + 9) = 5 / 7) : b / a = 165 / 222 := 
by sorry

end NUMINAMATH_GPT_fraction_value_l1361_136126


namespace NUMINAMATH_GPT_combined_yells_l1361_136165

def yells_at_obedient : ℕ := 12
def yells_at_stubborn (y_obedient : ℕ) : ℕ := 4 * y_obedient
def total_yells (y_obedient : ℕ) (y_stubborn : ℕ) : ℕ := y_obedient + y_stubborn

theorem combined_yells : total_yells yells_at_obedient (yells_at_stubborn yells_at_obedient) = 60 := 
by
  sorry

end NUMINAMATH_GPT_combined_yells_l1361_136165


namespace NUMINAMATH_GPT_greatest_product_of_two_even_integers_whose_sum_is_300_l1361_136144

theorem greatest_product_of_two_even_integers_whose_sum_is_300 :
  ∃ (x y : ℕ), (2 ∣ x) ∧ (2 ∣ y) ∧ (x + y = 300) ∧ (x * y = 22500) :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_of_two_even_integers_whose_sum_is_300_l1361_136144


namespace NUMINAMATH_GPT_vikki_take_home_pay_l1361_136137

-- Define the conditions
def hours_worked : ℕ := 42
def pay_rate : ℝ := 10
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5

-- Define the gross earnings function
def gross_earnings (hours_worked : ℕ) (pay_rate : ℝ) : ℝ := hours_worked * pay_rate

-- Define the deductions functions
def tax_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def insurance_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def total_deductions (tax : ℝ) (insurance : ℝ) (dues : ℝ) : ℝ := tax + insurance + dues

-- Define the take-home pay function
def take_home_pay (gross : ℝ) (deductions : ℝ) : ℝ := gross - deductions

theorem vikki_take_home_pay :
  take_home_pay (gross_earnings hours_worked pay_rate)
    (total_deductions (tax_deduction (gross_earnings hours_worked pay_rate) tax_rate)
                      (insurance_deduction (gross_earnings hours_worked pay_rate) insurance_rate)
                      union_dues) = 310 :=
by
  sorry

end NUMINAMATH_GPT_vikki_take_home_pay_l1361_136137


namespace NUMINAMATH_GPT_find_price_per_backpack_l1361_136166

noncomputable def original_price_of_each_backpack
  (total_backpacks : ℕ)
  (monogram_cost : ℕ)
  (total_cost : ℕ)
  (backpacks_cost_before_discount : ℕ) : ℕ :=
total_cost - (total_backpacks * monogram_cost)

theorem find_price_per_backpack
  (total_backpacks : ℕ := 5)
  (monogram_cost : ℕ := 12)
  (total_cost : ℕ := 140)
  (expected_price_per_backpack : ℕ := 16) :
  original_price_of_each_backpack total_backpacks monogram_cost total_cost / total_backpacks = expected_price_per_backpack :=
by
  sorry

end NUMINAMATH_GPT_find_price_per_backpack_l1361_136166


namespace NUMINAMATH_GPT_terminating_decimal_multiples_l1361_136124

theorem terminating_decimal_multiples :
  (∃ n : ℕ, 20 = n ∧ ∀ m, 1 ≤ m ∧ m ≤ 180 → 
  (∃ k : ℕ, m = 9 * k)) :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_multiples_l1361_136124


namespace NUMINAMATH_GPT_largest_among_numbers_l1361_136183

theorem largest_among_numbers :
  ∀ (a b c d e : ℝ), 
  a = 0.997 ∧ b = 0.9799 ∧ c = 0.999 ∧ d = 0.9979 ∧ e = 0.979 →
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by intros a b c d e habcde
   rcases habcde with ⟨ha, hb, hc, hd, he⟩
   simp [ha, hb, hc, hd, he]
   sorry

end NUMINAMATH_GPT_largest_among_numbers_l1361_136183


namespace NUMINAMATH_GPT_factory_days_worked_l1361_136198

-- Define the number of refrigerators produced per hour
def refrigerators_per_hour : ℕ := 90

-- Define the number of coolers produced per hour
def coolers_per_hour : ℕ := refrigerators_per_hour + 70

-- Define the number of working hours per day
def working_hours_per_day : ℕ := 9

-- Define the total products produced per hour
def products_per_hour : ℕ := refrigerators_per_hour + coolers_per_hour

-- Define the total products produced in a day
def products_per_day : ℕ := products_per_hour * working_hours_per_day

-- Define the total number of products produced in given days
def total_products : ℕ := 11250

-- Define the number of days worked
def days_worked : ℕ := total_products / products_per_day

-- Prove that the number of days worked equals 5
theorem factory_days_worked : days_worked = 5 :=
by
  sorry

end NUMINAMATH_GPT_factory_days_worked_l1361_136198


namespace NUMINAMATH_GPT_calculate_interest_rate_l1361_136104

theorem calculate_interest_rate
  (total_investment : ℝ)
  (invested_at_eleven_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_first_type : ℝ) :
  total_investment = 100000 ∧ 
  invested_at_eleven_percent = 30000 ∧ 
  total_interest = 9.6 → 
  interest_rate_first_type = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calculate_interest_rate_l1361_136104


namespace NUMINAMATH_GPT_train_speed_comparison_l1361_136172

variables (V_A V_B : ℝ)

open Classical

theorem train_speed_comparison
  (distance_AB : ℝ)
  (h_distance : distance_AB = 360)
  (h_time_limit : V_A ≤ 72)
  (h_meeting_time : 3 * V_A + 2 * V_B > 360) :
  V_B > V_A :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_comparison_l1361_136172


namespace NUMINAMATH_GPT_comprehensive_score_l1361_136115

variable (regularAssessmentScore : ℕ)
variable (finalExamScore : ℕ)
variable (regularAssessmentWeighting : ℝ)
variable (finalExamWeighting : ℝ)

theorem comprehensive_score 
  (h1 : regularAssessmentScore = 95)
  (h2 : finalExamScore = 90)
  (h3 : regularAssessmentWeighting = 0.20)
  (h4 : finalExamWeighting = 0.80) :
  (regularAssessmentScore * regularAssessmentWeighting + finalExamScore * finalExamWeighting) = 91 :=
sorry

end NUMINAMATH_GPT_comprehensive_score_l1361_136115


namespace NUMINAMATH_GPT_pizza_slices_left_per_person_l1361_136191

def total_slices (small: Nat) (large: Nat) : Nat := small + large

def total_eaten (phil: Nat) (andre: Nat) : Nat := phil + andre

def slices_left (total: Nat) (eaten: Nat) : Nat := total - eaten

def pieces_per_person (left: Nat) (people: Nat) : Nat := left / people

theorem pizza_slices_left_per_person :
  ∀ (small large phil andre people: Nat),
  small = 8 → large = 14 → phil = 9 → andre = 9 → people = 2 →
  pieces_per_person (slices_left (total_slices small large) (total_eaten phil andre)) people = 2 :=
by
  intros small large phil andre people h_small h_large h_phil h_andre h_people
  rw [h_small, h_large, h_phil, h_andre, h_people]
  /-
  Here we conclude the proof.
  -/
  sorry

end NUMINAMATH_GPT_pizza_slices_left_per_person_l1361_136191


namespace NUMINAMATH_GPT_number_of_free_ranging_chickens_l1361_136129

-- Define the conditions as constants
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def barn_chickens : ℕ := coop_chickens / 2
def total_chickens_in_coop_and_run : ℕ := coop_chickens + run_chickens    
def free_ranging_chickens_condition : ℕ := 2 * run_chickens - 4
def ratio_condition : Prop := total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + free_ranging_chickens_condition)
def target_free_ranging_chickens : ℕ := 105

-- The proof statement
theorem number_of_free_ranging_chickens : 
  total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + target_free_ranging_chickens) →
  free_ranging_chickens_condition = target_free_ranging_chickens :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_free_ranging_chickens_l1361_136129


namespace NUMINAMATH_GPT_polynomial_bound_l1361_136125

theorem polynomial_bound (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end NUMINAMATH_GPT_polynomial_bound_l1361_136125


namespace NUMINAMATH_GPT_find_coefficients_l1361_136182

def polynomial (a b : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 3 - 3 * x ^ 2 + b * x - 7

theorem find_coefficients (a b : ℝ) :
  polynomial a b 2 = -17 ∧ polynomial a b (-1) = -11 → a = 0 ∧ b = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_l1361_136182


namespace NUMINAMATH_GPT_absolute_value_equality_l1361_136155

variables {a b c d : ℝ}

theorem absolute_value_equality (h1 : |a - b| + |c - d| = 99) (h2 : |a - c| + |b - d| = 1) : |a - d| + |b - c| = 99 :=
sorry

end NUMINAMATH_GPT_absolute_value_equality_l1361_136155


namespace NUMINAMATH_GPT_right_triangle_m_c_l1361_136108

theorem right_triangle_m_c (a b c : ℝ) (m_c : ℝ) 
  (h : (1 / a) + (1 / b) = 3 / c) : 
  m_c = (c * (1 + Real.sqrt 10)) / 9 :=
sorry

end NUMINAMATH_GPT_right_triangle_m_c_l1361_136108


namespace NUMINAMATH_GPT_cost_of_running_tv_for_week_l1361_136106

def powerUsage : ℕ := 125
def hoursPerDay : ℕ := 4
def costPerkWh : ℕ := 14

theorem cost_of_running_tv_for_week :
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  weeklyCost = 49 := by
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  sorry

end NUMINAMATH_GPT_cost_of_running_tv_for_week_l1361_136106


namespace NUMINAMATH_GPT_ratio_of_sides_l1361_136164

theorem ratio_of_sides (
  perimeter_triangle perimeter_square : ℕ)
  (h_triangle : perimeter_triangle = 48)
  (h_square : perimeter_square = 64) :
  (perimeter_triangle / 3) / (perimeter_square / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l1361_136164


namespace NUMINAMATH_GPT_percentage_equivalence_l1361_136192

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end NUMINAMATH_GPT_percentage_equivalence_l1361_136192


namespace NUMINAMATH_GPT_problem_statement_l1361_136146

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 4))

theorem problem_statement :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-Real.sqrt 2 / 2) 1) ∧
  (f (Real.pi / 2) = -Real.sqrt 2 / 2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8), 
    ∃ δ > 0, ∀ y ∈ Set.Ioc x (x + δ), f x < f y) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1361_136146


namespace NUMINAMATH_GPT_salary_of_thomas_l1361_136107

variable (R Ro T : ℕ)

theorem salary_of_thomas 
  (h1 : R + Ro = 8000) 
  (h2 : R + Ro + T = 15000) : 
  T = 7000 := by
  sorry

end NUMINAMATH_GPT_salary_of_thomas_l1361_136107


namespace NUMINAMATH_GPT_base_salary_at_least_l1361_136135

-- Definitions for the conditions.
def previous_salary : ℕ := 75000
def commission_rate : ℚ := 0.15
def sale_value : ℕ := 750
def min_sales_required : ℚ := 266.67

-- Calculate the commission per sale
def commission_per_sale : ℚ := commission_rate * sale_value

-- Calculate the total commission for the minimum sales required
def total_commission : ℚ := min_sales_required * commission_per_sale

-- The base salary S required to not lose money
theorem base_salary_at_least (S : ℚ) : S + total_commission ≥ previous_salary ↔ S ≥ 45000 := 
by
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_base_salary_at_least_l1361_136135


namespace NUMINAMATH_GPT_polar_to_rectangular_inequality_range_l1361_136189

-- Part A: Transforming a polar coordinate equation to a rectangular coordinate equation
theorem polar_to_rectangular (ρ θ : ℝ) : 
  (ρ^2 * Real.cos θ - ρ = 0) ↔ ((ρ = 0 ∧ 0 = 1) ∨ (ρ ≠ 0 ∧ Real.cos θ = 1 / ρ)) := 
sorry

-- Part B: Determining range for an inequality
theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2-x| + |x+1| ≤ a) ↔ (a ≥ 9) := 
sorry

end NUMINAMATH_GPT_polar_to_rectangular_inequality_range_l1361_136189


namespace NUMINAMATH_GPT_geometric_sequence_properties_l1361_136163

/-- Given {a_n} is a geometric sequence, a_1 = 1 and a_4 = 1/8, 
the common ratio q of {a_n} is 1/2 and the sum of the first 5 terms of {1/a_n} is 31. -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (h1 : a 1 = 1) (h4 : a 4 = 1 / 8) : 
  (∃ q : ℝ, (∀ n : ℕ, a n = a 1 * q ^ (n - 1)) ∧ q = 1 / 2) ∧ 
  (∃ S : ℝ, S = 31 ∧ S = (1 - 2^5) / (1 - 2)) :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l1361_136163


namespace NUMINAMATH_GPT_online_game_months_l1361_136179

theorem online_game_months (m : ℕ) (initial_cost monthly_cost total_cost : ℕ) 
  (h1 : initial_cost = 5) (h2 : monthly_cost = 8) (h3 : total_cost = 21) 
  (h_equation : initial_cost + monthly_cost * m = total_cost) : m = 2 :=
by {
  -- Placeholder for the proof, as we don't need to include it
  sorry
}

end NUMINAMATH_GPT_online_game_months_l1361_136179


namespace NUMINAMATH_GPT_symmetrical_parabola_eq_l1361_136181

/-- 
  Given a parabola y = (x-1)^2 + 3, prove that its symmetrical parabola 
  about the x-axis is y = -(x-1)^2 - 3.
-/
theorem symmetrical_parabola_eq (x : ℝ) : 
  (x-1)^2 + 3 = -(x-1)^2 - 3 ↔ y = -(x-1)^2 - 3 := 
sorry

end NUMINAMATH_GPT_symmetrical_parabola_eq_l1361_136181


namespace NUMINAMATH_GPT_find_c_share_l1361_136142

theorem find_c_share (a b c : ℕ) 
  (h1 : a + b + c = 1760)
  (h2 : ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x)
  (h3 : 6 * a = 8 * b ∧ 8 * b = 20 * c) : 
  c = 250 :=
by
  sorry

end NUMINAMATH_GPT_find_c_share_l1361_136142


namespace NUMINAMATH_GPT_mass_scientific_notation_l1361_136158

def mass := 37e-6

theorem mass_scientific_notation : mass = 3.7 * 10^(-5) :=
by
  sorry

end NUMINAMATH_GPT_mass_scientific_notation_l1361_136158


namespace NUMINAMATH_GPT_min_third_side_triangle_l1361_136134

theorem min_third_side_triangle (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_distinct_1 : 42 * a ≠ 72 * b) (h_distinct_2 : 42 * a ≠ c) (h_distinct_3 : 72 * b ≠ c) :
    (42 * a + 72 * b > c) ∧ (42 * a + c > 72 * b) ∧ (72 * b + c > 42 * a) → c ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_third_side_triangle_l1361_136134


namespace NUMINAMATH_GPT_problem_equiv_proof_l1361_136176

theorem problem_equiv_proof :
  2015 * (1 + 1999 / 2015) * (1 / 4) - (2011 / 2015) = 503 := 
by
  sorry

end NUMINAMATH_GPT_problem_equiv_proof_l1361_136176


namespace NUMINAMATH_GPT_player_winning_strategy_l1361_136180

-- Define the game conditions
def Sn (n : ℕ) : Type := Equiv.Perm (Fin n)

def game_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ G : Set (Sn n), ∃ x : Sn n, x ∈ G → G ≠ (Set.univ : Set (Sn n)))

-- Statement of the proof problem
theorem player_winning_strategy (n : ℕ) (hn : n > 1) : 
  ((n = 2 ∨ n = 3) → (∃ strategyA : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyA x x)) ∧ 
  ((n ≥ 4 ∧ n % 2 = 1) → (∃ strategyB : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyB x x)) :=
by
  sorry

end NUMINAMATH_GPT_player_winning_strategy_l1361_136180


namespace NUMINAMATH_GPT_inequality_solution_correct_l1361_136169

variable (f : ℝ → ℝ)

def f_one : Prop := f 1 = 1

def f_prime_half : Prop := ∀ x : ℝ, (deriv f x) > (1 / 2)

def inequality_solution_set : Prop := ∀ x : ℝ, f (x^2) < (x^2 / 2 + 1 / 2) ↔ -1 < x ∧ x < 1

theorem inequality_solution_correct (h1 : f_one f) (h2 : f_prime_half f) : inequality_solution_set f := sorry

end NUMINAMATH_GPT_inequality_solution_correct_l1361_136169


namespace NUMINAMATH_GPT_number_of_valid_permutations_l1361_136148

noncomputable def count_valid_permutations : Nat :=
  let multiples_of_77 := [154, 231, 308, 385, 462, 539, 616, 693, 770, 847, 924]
  let total_count := multiples_of_77.foldl (fun acc x =>
    if x == 770 then
      acc + 3
    else if x == 308 then
      acc + 6 - 2
    else
      acc + 6) 0
  total_count

theorem number_of_valid_permutations : count_valid_permutations = 61 :=
  sorry

end NUMINAMATH_GPT_number_of_valid_permutations_l1361_136148


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1361_136116

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a n = a 0 + n * d)
  (h1 : a 0 + a 3 + a 6 = 45)
  (h2 : a 1 + a 4 + a 7 = 39) :
  a 2 + a 5 + a 8 = 33 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1361_136116


namespace NUMINAMATH_GPT_simplify_exponent_multiplication_l1361_136139

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end NUMINAMATH_GPT_simplify_exponent_multiplication_l1361_136139


namespace NUMINAMATH_GPT_opposite_sides_of_line_l1361_136167

theorem opposite_sides_of_line 
  (x₀ y₀ : ℝ) 
  (h : (3 * x₀ + 2 * y₀ - 8) * (3 * 1 + 2 * 2 - 8) < 0) :
  3 * x₀ + 2 * y₀ > 8 :=
by
  sorry

end NUMINAMATH_GPT_opposite_sides_of_line_l1361_136167


namespace NUMINAMATH_GPT_probability_of_sum_14_l1361_136188

-- Define the set of faces on a tetrahedral die
def faces : Set ℕ := {2, 4, 6, 8}

-- Define the event where the sum of two rolls equals 14
def event_sum_14 (a b : ℕ) : Prop := a + b = 14 ∧ a ∈ faces ∧ b ∈ faces

-- Define the total number of outcomes when rolling two dice
def total_outcomes : ℕ := 16

-- Define the number of successful outcomes for the event where the sum is 14
def successful_outcomes : ℕ := 2

-- The probability of rolling a sum of 14 with two such tetrahedral dice
def probability_sum_14 : ℚ := successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_sum_14 : probability_sum_14 = 1 / 8 := 
by sorry

end NUMINAMATH_GPT_probability_of_sum_14_l1361_136188


namespace NUMINAMATH_GPT_geometric_mean_of_4_and_9_l1361_136199

theorem geometric_mean_of_4_and_9 : ∃ (G : ℝ), G = 6 ∨ G = -6 :=
by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_4_and_9_l1361_136199


namespace NUMINAMATH_GPT_fred_paid_amount_l1361_136121

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def borrowed_movie_price : ℝ := 6.79
def change_received : ℝ := 1.37

def total_cost : ℝ := (number_of_tickets : ℝ) * ticket_price + borrowed_movie_price
def amount_paid : ℝ := total_cost + change_received

theorem fred_paid_amount : amount_paid = 20.00 := sorry

end NUMINAMATH_GPT_fred_paid_amount_l1361_136121


namespace NUMINAMATH_GPT_num_two_digit_palindromes_l1361_136114

theorem num_two_digit_palindromes : 
  let is_palindrome (n : ℕ) : Prop := (n / 10) = (n % 10)
  ∃ n : ℕ, 10 ≤ n ∧ n < 90 ∧ is_palindrome n →
  ∃ count : ℕ, count = 9 := 
sorry

end NUMINAMATH_GPT_num_two_digit_palindromes_l1361_136114


namespace NUMINAMATH_GPT_vacation_costs_l1361_136123

variable (Anne_paid Beth_paid Carlos_paid : ℕ) (a b : ℕ)

theorem vacation_costs (hAnne : Anne_paid = 120) (hBeth : Beth_paid = 180) (hCarlos : Carlos_paid = 150)
  (h_a : a = 30) (h_b : b = 30) :
  a - b = 0 := sorry

end NUMINAMATH_GPT_vacation_costs_l1361_136123


namespace NUMINAMATH_GPT_total_words_in_poem_l1361_136128

theorem total_words_in_poem (s l w : ℕ) (h1 : s = 35) (h2 : l = 15) (h3 : w = 12) : 
  s * l * w = 6300 := 
by 
  -- the proof will be inserted here
  sorry

end NUMINAMATH_GPT_total_words_in_poem_l1361_136128


namespace NUMINAMATH_GPT_minimum_embrasure_length_l1361_136138

theorem minimum_embrasure_length : ∀ (s : ℝ), 
  (∀ t : ℝ, (∃ k : ℤ, t = k / 2 ∧ k % 2 = 0) ∨ (∃ k : ℤ, t = (k + 1) / 2 ∧ k % 2 = 1)) → 
  (∃ z : ℝ, z = 2 / 3) := 
sorry

end NUMINAMATH_GPT_minimum_embrasure_length_l1361_136138


namespace NUMINAMATH_GPT_seed_mixture_ryegrass_percent_l1361_136190

theorem seed_mixture_ryegrass_percent (R : ℝ) :
  let X := 0.40
  let percentage_X_in_mixture := 1 / 3
  let percentage_Y_in_mixture := 2 / 3
  let final_ryegrass := 0.30
  (final_ryegrass = percentage_X_in_mixture * X + percentage_Y_in_mixture * R) → 
  R = 0.25 :=
by
  intros X percentage_X_in_mixture percentage_Y_in_mixture final_ryegrass H
  sorry

end NUMINAMATH_GPT_seed_mixture_ryegrass_percent_l1361_136190


namespace NUMINAMATH_GPT_problem1_problem2_l1361_136151

def f (x : ℝ) : ℝ := |x - 3|

theorem problem1 :
  {x : ℝ | f x < 2 + |x + 1|} = {x : ℝ | 0 < x} := sorry

theorem problem2 (m n : ℝ) (h_mn : m > 0) (h_nn : n > 0) (h : (1 / m) + (1 / n) = 2 * m * n) :
  m * f n + n * f (-m) ≥ 6 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1361_136151


namespace NUMINAMATH_GPT_smallest_n_for_factorization_l1361_136141

theorem smallest_n_for_factorization :
  ∃ n : ℤ, (∀ A B : ℤ, A * B = 60 ↔ n = 5 * B + A) ∧ n = 56 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_factorization_l1361_136141


namespace NUMINAMATH_GPT_geometric_sequence_k_value_l1361_136105

theorem geometric_sequence_k_value :
  ∀ {S : ℕ → ℤ} (a : ℕ → ℤ) (k : ℤ),
    (∀ n, S n = 3 * 2^n + k) → 
    (∀ n ≥ 2, a n = S n - S (n - 1)) → 
    (∀ n ≥ 2, a n ^ 2 = a 1 * a 3) → 
    k = -3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_k_value_l1361_136105


namespace NUMINAMATH_GPT_rotated_line_l1_l1361_136161

-- Define the original line equation and the point around which the line is rotated
def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def point_A : ℝ × ℝ := (2, 3)

-- Define the line equation that needs to be proven
def line_l1 (x y : ℝ) : Prop := x + y - 5 = 0

-- The theorem stating that after a 90-degree rotation of line l around point A, the new line is equation l1
theorem rotated_line_l1 : 
  ∀ (x y : ℝ), 
  (∃ (k : ℝ), k = 1 ∧ ∀ (x y), line_l x y ∧ ∀ (x y), line_l1 x y) ∧ 
  ∀ (a b : ℝ), (a, b) = point_A → 
  x + y - 5 = 0 := 
by
  sorry

end NUMINAMATH_GPT_rotated_line_l1_l1361_136161


namespace NUMINAMATH_GPT_total_spent_correct_l1361_136196

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end NUMINAMATH_GPT_total_spent_correct_l1361_136196


namespace NUMINAMATH_GPT_inclination_angle_of_line_l1361_136152

noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  Real.arctan m

theorem inclination_angle_of_line (α : ℝ) :
  angle_of_inclination (-1) = 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l1361_136152


namespace NUMINAMATH_GPT_carol_meets_alice_in_30_minutes_l1361_136132

def time_to_meet (alice_speed carol_speed initial_distance : ℕ) : ℕ :=
((initial_distance * 60) / (alice_speed + carol_speed))

theorem carol_meets_alice_in_30_minutes :
  time_to_meet 4 6 5 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_carol_meets_alice_in_30_minutes_l1361_136132


namespace NUMINAMATH_GPT_max_value_of_f_l1361_136162

noncomputable def f (x : ℝ) := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ x, f x = Real.sqrt 5 := sorry

end NUMINAMATH_GPT_max_value_of_f_l1361_136162


namespace NUMINAMATH_GPT_total_investment_sum_l1361_136143

-- Definitions of the problem
variable (Raghu Trishul Vishal : ℕ)
variable (h1 : Raghu = 2000)
variable (h2 : Trishul = Nat.div (Raghu * 9) 10)
variable (h3 : Vishal = Nat.div (Trishul * 11) 10)

-- The theorem to prove
theorem total_investment_sum :
  Vishal + Trishul + Raghu = 5780 :=
by
  sorry

end NUMINAMATH_GPT_total_investment_sum_l1361_136143


namespace NUMINAMATH_GPT_exists_cubic_polynomial_with_cubed_roots_l1361_136185

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

-- Statement that we need to prove
theorem exists_cubic_polynomial_with_cubed_roots :
  ∃ (b c d : ℝ), ∀ (x : ℝ),
  (f x = 0) → (x^3 = y → x^3^3 + b * x^3^2 + c * x^3 + d = 0) :=
sorry

end NUMINAMATH_GPT_exists_cubic_polynomial_with_cubed_roots_l1361_136185


namespace NUMINAMATH_GPT_find_a_value_l1361_136131

theorem find_a_value 
  (A : Set ℤ := {-1, 0, 1})
  (a : ℤ) 
  (B : Set ℤ := {a, a^2}) 
  (h_union : A ∪ B = A) : 
  a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_value_l1361_136131
