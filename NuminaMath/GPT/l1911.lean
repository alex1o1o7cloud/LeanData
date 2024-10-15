import Mathlib

namespace NUMINAMATH_GPT_min_value_abs_b_minus_c_l1911_191190

-- Define the problem conditions
def condition1 (a b c : ℝ) : Prop :=
  (a - 2 * b - 1)^2 + (a - c - Real.log c)^2 = 0

-- Define the theorem to be proved
theorem min_value_abs_b_minus_c {a b c : ℝ} (h : condition1 a b c) : |b - c| = 1 :=
sorry

end NUMINAMATH_GPT_min_value_abs_b_minus_c_l1911_191190


namespace NUMINAMATH_GPT_find_multiple_l1911_191185

/-- 
Given:
1. Hank Aaron hit 755 home runs.
2. Dave Winfield hit 465 home runs.
3. Hank Aaron has 175 fewer home runs than a certain multiple of the number that Dave Winfield has.

Prove:
The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to is 2.
-/
def multiple_of_dave_hr (ha_hr dw_hr diff : ℕ) (m : ℕ) : Prop :=
  ha_hr + diff = m * dw_hr

theorem find_multiple :
  multiple_of_dave_hr 755 465 175 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1911_191185


namespace NUMINAMATH_GPT_fifty_third_number_is_2_pow_53_l1911_191162

theorem fifty_third_number_is_2_pow_53 :
  ∀ n : ℕ, (n = 53) → ∃ seq : ℕ → ℕ, (seq 1 = 2) ∧ (∀ k : ℕ, seq (k+1) = 2 * seq k) ∧ (seq n = 2 ^ 53) :=
  sorry

end NUMINAMATH_GPT_fifty_third_number_is_2_pow_53_l1911_191162


namespace NUMINAMATH_GPT_balance_scale_measurements_l1911_191159

theorem balance_scale_measurements {a b c : ℕ}
    (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
    ∀ w : ℕ, 1 ≤ w ∧ w ≤ 11 → ∃ (x y z : ℤ), w = abs (x * a + y * b + z * c) :=
sorry

end NUMINAMATH_GPT_balance_scale_measurements_l1911_191159


namespace NUMINAMATH_GPT_rectangular_prism_surface_area_l1911_191154

/-- The surface area of a rectangular prism with edge lengths 2, 3, and 4 is 52. -/
theorem rectangular_prism_surface_area :
  let a := 2
  let b := 3
  let c := 4
  2 * (a * b + a * c + b * c) = 52 :=
by
  let a := 2
  let b := 3
  let c := 4
  show 2 * (a * b + a * c + b * c) = 52
  sorry

end NUMINAMATH_GPT_rectangular_prism_surface_area_l1911_191154


namespace NUMINAMATH_GPT_marys_garbage_bill_l1911_191125

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end NUMINAMATH_GPT_marys_garbage_bill_l1911_191125


namespace NUMINAMATH_GPT_binom_sum_l1911_191120

theorem binom_sum :
  (Nat.choose 15 12) + 10 = 465 := by
  sorry

end NUMINAMATH_GPT_binom_sum_l1911_191120


namespace NUMINAMATH_GPT_carla_cream_volume_l1911_191123

-- Definitions of the given conditions and problem
def watermelon_puree_volume : ℕ := 500
def servings_count : ℕ := 4
def volume_per_serving : ℕ := 150
def total_smoothies_volume := servings_count * volume_per_serving
def cream_volume := total_smoothies_volume - watermelon_puree_volume

-- Statement of the proposition we want to prove
theorem carla_cream_volume : cream_volume = 100 := by
  sorry

end NUMINAMATH_GPT_carla_cream_volume_l1911_191123


namespace NUMINAMATH_GPT_cranberry_juice_cost_l1911_191168

theorem cranberry_juice_cost 
  (cost_per_ounce : ℕ) (number_of_ounces : ℕ) 
  (h1 : cost_per_ounce = 7) 
  (h2 : number_of_ounces = 12) : 
  cost_per_ounce * number_of_ounces = 84 := 
by 
  sorry

end NUMINAMATH_GPT_cranberry_juice_cost_l1911_191168


namespace NUMINAMATH_GPT_find_savings_l1911_191183

noncomputable def savings (income expenditure : ℕ) : ℕ :=
  income - expenditure

theorem find_savings (I E : ℕ) (h_ratio : I = 9 * E) (h_income : I = 18000) : savings I E = 2000 :=
by
  sorry

end NUMINAMATH_GPT_find_savings_l1911_191183


namespace NUMINAMATH_GPT_new_average_age_l1911_191149

theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) :
  avg_age = 40 →
  num_people = 8 →
  leaving_age = 25 →
  remaining_people = 7 →
  (avg_age * num_people - leaving_age) / remaining_people = 42 :=
by
  sorry

end NUMINAMATH_GPT_new_average_age_l1911_191149


namespace NUMINAMATH_GPT_range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l1911_191124

-- Define the propositions p and q
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := x^2 - 5 * x + 6 < 0

-- Question 1: When a = 1, if p ∧ q is true, determine the range of x
theorem range_of_x_when_a_is_1_and_p_and_q_are_true :
  ∀ x, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
by
  sorry

-- Question 2: If p is a necessary but not sufficient condition for q, determine the range of a
theorem range_of_a_when_p_necessary_for_q :
  ∀ a, (∀ x, q x → p x a) ∧ ¬ (∀ x, p x a → q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l1911_191124


namespace NUMINAMATH_GPT_mary_has_more_money_than_marco_l1911_191177

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end NUMINAMATH_GPT_mary_has_more_money_than_marco_l1911_191177


namespace NUMINAMATH_GPT_product_of_numbers_l1911_191143

theorem product_of_numbers (a b c m : ℚ) (h_sum : a + b + c = 240)
    (h_m_a : 6 * a = m) (h_m_b : m = b - 12) (h_m_c : m = c + 12) :
    a * b * c = 490108320 / 2197 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1911_191143


namespace NUMINAMATH_GPT_sequence_formula_correct_l1911_191142

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 2

-- Define the general term of the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 1 then -1 else 2 * n - 1

-- Theorem to prove that for the given S_n, the defined a_n is correct
theorem sequence_formula_correct (n : ℕ) (h : n > 0) : 
  a n = if n = 1 then -1 else S n - S (n - 1) :=
by sorry

end NUMINAMATH_GPT_sequence_formula_correct_l1911_191142


namespace NUMINAMATH_GPT_range_of_k_l1911_191163

theorem range_of_k (k : ℝ) : (x^2 + k * y^2 = 2) ∧ (k > 0) ∧ (k < 1) ↔ (0 < k ∧ k < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1911_191163


namespace NUMINAMATH_GPT_find_stock_rate_l1911_191131

theorem find_stock_rate (annual_income : ℝ) (investment_amount : ℝ) (R : ℝ) 
  (h1 : annual_income = 2000) (h2 : investment_amount = 6800) : 
  R = 2000 / 6800 :=
by
  sorry

end NUMINAMATH_GPT_find_stock_rate_l1911_191131


namespace NUMINAMATH_GPT_problem_statement_l1911_191176

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x - Real.sqrt x ≤ y - 1 / 4 ∧ y - 1 / 4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1 / 4 ∧ x - 1 / 4 ≤ y + Real.sqrt y :=
sorry

end NUMINAMATH_GPT_problem_statement_l1911_191176


namespace NUMINAMATH_GPT_Dave_won_tickets_l1911_191198

theorem Dave_won_tickets :
  ∀ (tickets_toys tickets_clothes total_tickets : ℕ),
  (tickets_toys = 8) →
  (tickets_clothes = 18) →
  (tickets_clothes = tickets_toys + 10) →
  (total_tickets = tickets_toys + tickets_clothes) →
  total_tickets = 26 :=
by
  intros tickets_toys tickets_clothes total_tickets h1 h2 h3 h4
  have h5 : tickets_clothes = 8 + 10 := by sorry
  have h6 : tickets_clothes = 18 := by sorry
  have h7 : tickets_clothes = 18 := by sorry
  exact sorry

end NUMINAMATH_GPT_Dave_won_tickets_l1911_191198


namespace NUMINAMATH_GPT_smallest_three_digit_number_multiple_of_conditions_l1911_191110

theorem smallest_three_digit_number_multiple_of_conditions :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
  (x % 2 = 0) ∧ ((x + 1) % 3 = 0) ∧ ((x + 2) % 4 = 0) ∧ ((x + 3) % 5 = 0) ∧ ((x + 4) % 6 = 0) 
  ∧ x = 122 := 
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_number_multiple_of_conditions_l1911_191110


namespace NUMINAMATH_GPT_purely_imaginary_iff_l1911_191112

noncomputable def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0

theorem purely_imaginary_iff (a : ℝ) :
  isPurelyImaginary (Complex.mk ((a * (a + 2)) / (a - 1)) (a ^ 2 + 2 * a - 3))
  ↔ a = 0 ∨ a = -2 := by
  sorry

end NUMINAMATH_GPT_purely_imaginary_iff_l1911_191112


namespace NUMINAMATH_GPT_water_consumption_eq_l1911_191146

-- Define all conditions
variables (x : ℝ) (improvement : ℝ := 0.8) (water : ℝ := 80) (days_difference : ℝ := 5)

-- State the theorem
theorem water_consumption_eq (h : improvement = 0.8) (initial_water := 80) (difference := 5) : 
  initial_water / x - (initial_water * improvement) / x = difference :=
sorry

end NUMINAMATH_GPT_water_consumption_eq_l1911_191146


namespace NUMINAMATH_GPT_xy_square_value_l1911_191147

theorem xy_square_value (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) : (x + y)^2 = 96 :=
by
  sorry

end NUMINAMATH_GPT_xy_square_value_l1911_191147


namespace NUMINAMATH_GPT_Pete_latest_time_to_LA_l1911_191137

def minutesInHour := 60
def minutesOfWalk := 10
def minutesOfTrain := 80
def departureTime := 7 * minutesInHour + 30

def latestArrivalTime : Prop :=
  9 * minutesInHour = departureTime + minutesOfWalk + minutesOfTrain 

theorem Pete_latest_time_to_LA : latestArrivalTime :=
by
  sorry

end NUMINAMATH_GPT_Pete_latest_time_to_LA_l1911_191137


namespace NUMINAMATH_GPT_abs_add_eq_abs_sub_implies_mul_eq_zero_l1911_191145

variable {a b : ℝ}

theorem abs_add_eq_abs_sub_implies_mul_eq_zero (h : |a + b| = |a - b|) : a * b = 0 :=
sorry

end NUMINAMATH_GPT_abs_add_eq_abs_sub_implies_mul_eq_zero_l1911_191145


namespace NUMINAMATH_GPT_arithmetic_sequence_x_value_l1911_191140

theorem arithmetic_sequence_x_value (x : ℝ) (a2 a1 d : ℝ)
  (h1 : a1 = 1 / 3)
  (h2 : a2 = x - 2)
  (h3 : d = 4 * x + 1 - a2)
  (h2_eq_d_a1 : a2 - a1 = d) : x = - (8 / 3) :=
by
  -- Proof yet to be completed
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_x_value_l1911_191140


namespace NUMINAMATH_GPT_root_interval_l1911_191158

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h₁ : f 1 < 0) (h₂ : f 1.5 > 0) (h₃ : f 1.25 < 0) (h₄ : f 2 > 0) :
  ∃ x, 1.25 < x ∧ x < 1.5 ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_root_interval_l1911_191158


namespace NUMINAMATH_GPT_find_a_for_chord_length_l1911_191171

theorem find_a_for_chord_length :
  ∀ a : ℝ, ((∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 ∧ (2 * x - y + a = 0)) 
  → ((2 * 1 - 1 + a = 0) → a = -1)) :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_chord_length_l1911_191171


namespace NUMINAMATH_GPT_abs_neg_three_l1911_191170

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1911_191170


namespace NUMINAMATH_GPT_five_digit_divisibility_l1911_191108

-- Definitions of n and m
def n (a b c d e : ℕ) := 10000 * a + 1000 * b + 100 * c + 10 * d + e
def m (a b d e : ℕ) := 1000 * a + 100 * b + 10 * d + e

-- Condition that n is a five-digit number whose first digit is non-zero and n/m is an integer
theorem five_digit_divisibility (a b c d e : ℕ):
  1 <= a ∧ a <= 9 → 0 <= b ∧ b <= 9 → 0 <= c ∧ c <= 9 → 0 <= d ∧ d <= 9 → 0 <= e ∧ e <= 9 →
  m a b d e ∣ n a b c d e →
  ∃ x y : ℕ, a = x ∧ b = y ∧ c = 0 ∧ d = 0 ∧ e = 0 :=
by
  sorry

end NUMINAMATH_GPT_five_digit_divisibility_l1911_191108


namespace NUMINAMATH_GPT_number_of_blocks_l1911_191180

theorem number_of_blocks (total_amount : ℕ) (gift_worth : ℕ) (workers_per_block : ℕ) (h1 : total_amount = 4000) (h2 : gift_worth = 4) (h3 : workers_per_block = 100) :
  (total_amount / gift_worth) / workers_per_block = 10 :=
by
-- This part will be proven later, hence using sorry for now
sorry

end NUMINAMATH_GPT_number_of_blocks_l1911_191180


namespace NUMINAMATH_GPT_largest_common_in_range_l1911_191188

-- Definitions for the problem's conditions
def first_seq (n : ℕ) : ℕ := 3 + 8 * n
def second_seq (m : ℕ) : ℕ := 5 + 9 * m

-- Statement of the theorem we are proving
theorem largest_common_in_range : 
  ∃ n m : ℕ, first_seq n = second_seq m ∧ 1 ≤ first_seq n ∧ first_seq n ≤ 200 ∧ first_seq n = 131 := by
  sorry

end NUMINAMATH_GPT_largest_common_in_range_l1911_191188


namespace NUMINAMATH_GPT_union_sets_l1911_191174

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets : A ∪ B = {x | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_union_sets_l1911_191174


namespace NUMINAMATH_GPT_avg_of_second_largest_and_second_smallest_is_eight_l1911_191153

theorem avg_of_second_largest_and_second_smallest_is_eight :
  ∀ (a b c d e : ℕ), 
  a + b + c + d + e = 40 → 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  ((d + b) / 2 : ℕ) = 8 := 
by
  intro a b c d e hsum horder
  /- the proof goes here, but we use sorry to skip it -/
  sorry

end NUMINAMATH_GPT_avg_of_second_largest_and_second_smallest_is_eight_l1911_191153


namespace NUMINAMATH_GPT_determine_g_l1911_191100

noncomputable def g (x : ℝ) : ℝ := -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8

theorem determine_g (x : ℝ) : 
  4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1 := by
  sorry

end NUMINAMATH_GPT_determine_g_l1911_191100


namespace NUMINAMATH_GPT_simplify_fraction_l1911_191189

theorem simplify_fraction :
  (18 / 462) + (35 / 77) = 38 / 77 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1911_191189


namespace NUMINAMATH_GPT_right_triangle_sides_l1911_191122

theorem right_triangle_sides (a b c : ℝ) (h_ratio : ∃ x : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x) 
(h_area : 1 / 2 * a * b = 24) : a = 6 ∧ b = 8 ∧ c = 10 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l1911_191122


namespace NUMINAMATH_GPT_monomial_same_type_l1911_191132

theorem monomial_same_type (a b : ℕ) (h1 : a + 1 = 3) (h2 : b = 3) : a + b = 5 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_monomial_same_type_l1911_191132


namespace NUMINAMATH_GPT_smallest_number_of_marbles_l1911_191184

theorem smallest_number_of_marbles 
  (r w b bl n : ℕ) 
  (h : r + w + b + bl = n)
  (h1 : r * (r - 1) * (r - 2) * (r - 3) = 24 * w * b * (r * (r - 1) / 2))
  (h2 : r * (r - 1) * (r - 2) * (r - 3) = 24 * bl * b * (r * (r - 1) / 2))
  (h_no_neg : 4 ≤ r):
  n = 18 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_marbles_l1911_191184


namespace NUMINAMATH_GPT_mary_total_nickels_l1911_191178

theorem mary_total_nickels : (7 + 12 + 9 = 28) :=
by
  sorry

end NUMINAMATH_GPT_mary_total_nickels_l1911_191178


namespace NUMINAMATH_GPT_polynomial_roots_absolute_sum_l1911_191119

theorem polynomial_roots_absolute_sum (p q r : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2027) :
  |p| + |q| + |r| = 98 := 
sorry

end NUMINAMATH_GPT_polynomial_roots_absolute_sum_l1911_191119


namespace NUMINAMATH_GPT_arithmetic_expression_result_l1911_191192

theorem arithmetic_expression_result :
  (24 / (8 + 2 - 5)) * 7 = 33.6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_result_l1911_191192


namespace NUMINAMATH_GPT_nancy_potatoes_l1911_191155

theorem nancy_potatoes (sandy_potatoes total_potatoes : ℕ) (h1 : sandy_potatoes = 7) (h2 : total_potatoes = 13) :
    total_potatoes - sandy_potatoes = 6 :=
by
  sorry

end NUMINAMATH_GPT_nancy_potatoes_l1911_191155


namespace NUMINAMATH_GPT_probability_not_green_l1911_191166

theorem probability_not_green :
  let red_balls := 6
  let yellow_balls := 3
  let black_balls := 4
  let green_balls := 5
  let total_balls := red_balls + yellow_balls + black_balls + green_balls
  let not_green_balls := red_balls + yellow_balls + black_balls
  total_balls = 18 ∧ not_green_balls = 13 → (not_green_balls : ℚ) / total_balls = 13 / 18 := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_not_green_l1911_191166


namespace NUMINAMATH_GPT_heaps_never_empty_l1911_191194

-- Define initial conditions
def initial_heaps := (1993, 199, 19)

-- Allowed operations
def add_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a + b + c, b, c)
else if b = 199 then (a, b + a + c, c)
else (a, b, c + a + b)

def remove_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a - (b + c), b, c)
else if b = 199 then (a, b - (a + c), c)
else (a, b, c - (a + b))

-- The proof statement
theorem heaps_never_empty :
  ∀ a b c : ℕ, a = 1993 ∧ b = 199 ∧ c = 19 ∧ (∀ n : ℕ, (a + b + c) % 2 = 1) ∧ (a - (b + c) % 2 = 1) → ¬(a = 0 ∨ b = 0 ∨ c = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_heaps_never_empty_l1911_191194


namespace NUMINAMATH_GPT_cubic_sum_l1911_191139

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_l1911_191139


namespace NUMINAMATH_GPT_common_number_is_eleven_l1911_191175

theorem common_number_is_eleven 
  (a b c d e f g h i : ℝ)
  (H1 : (a + b + c + d + e) / 5 = 7)
  (H2 : (e + f + g + h + i) / 5 = 10)
  (H3 : (a + b + c + d + e + f + g + h + i) / 9 = 74 / 9) :
  e = 11 := 
sorry

end NUMINAMATH_GPT_common_number_is_eleven_l1911_191175


namespace NUMINAMATH_GPT_minimum_omega_l1911_191130

open Real

theorem minimum_omega (ω : ℕ) (h_ω_pos : ω > 0) :
  (∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + (π / 2)) → ω = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_omega_l1911_191130


namespace NUMINAMATH_GPT_tim_balloons_proof_l1911_191133

-- Define the number of balloons Dan has
def dan_balloons : ℕ := 29

-- Define the relationship between Tim's and Dan's balloons
def balloons_ratio : ℕ := 7

-- Define the number of balloons Tim has
def tim_balloons : ℕ := balloons_ratio * dan_balloons

-- Prove that the number of balloons Tim has is 203
theorem tim_balloons_proof : tim_balloons = 203 :=
sorry

end NUMINAMATH_GPT_tim_balloons_proof_l1911_191133


namespace NUMINAMATH_GPT_least_positive_integer_to_add_l1911_191187

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_positive_integer_to_add_l1911_191187


namespace NUMINAMATH_GPT_part_a_l1911_191191

theorem part_a (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) :
  ∃ (m : ℕ) (x1 x2 x3 x4 : ℤ), m < p ∧ (x1^2 + x2^2 + x3^2 + x4^2 = m * p) :=
sorry

end NUMINAMATH_GPT_part_a_l1911_191191


namespace NUMINAMATH_GPT_tan_of_angle_in_third_quadrant_l1911_191121

open Real

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : sin (π + α) = 3/5) 
  (h2 : π < α ∧ α < 3 * π / 2) : 
  tan α = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_angle_in_third_quadrant_l1911_191121


namespace NUMINAMATH_GPT_sum_reciprocals_eq_three_l1911_191152

-- Define nonzero real numbers x and y with their given condition
variables (x y : ℝ) (hx : x ≠ 0) (hy: y ≠ 0) (h : x + y = 3 * x * y)

-- State the theorem to prove the sum of reciprocals of x and y is 3
theorem sum_reciprocals_eq_three (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : (1 / x) + (1 / y) = 3 :=
sorry

end NUMINAMATH_GPT_sum_reciprocals_eq_three_l1911_191152


namespace NUMINAMATH_GPT_percentage_x_y_l1911_191135

variable (x y P : ℝ)

theorem percentage_x_y 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y))
  (h2 : y = (1 / 9) * x) : 
  P = 40 :=
sorry

end NUMINAMATH_GPT_percentage_x_y_l1911_191135


namespace NUMINAMATH_GPT_determine_percentage_of_yellow_in_darker_green_paint_l1911_191165

noncomputable def percentage_of_yellow_in_darker_green_paint : Real :=
  let volume_light_green := 5
  let volume_darker_green := 1.66666666667
  let percentage_light_green := 0.20
  let final_percentage := 0.25
  let total_volume := volume_light_green + volume_darker_green
  let total_yellow_required := final_percentage * total_volume
  let yellow_in_light_green := percentage_light_green * volume_light_green
  (total_yellow_required - yellow_in_light_green) / volume_darker_green

theorem determine_percentage_of_yellow_in_darker_green_paint :
  percentage_of_yellow_in_darker_green_paint = 0.4 := by
  sorry

end NUMINAMATH_GPT_determine_percentage_of_yellow_in_darker_green_paint_l1911_191165


namespace NUMINAMATH_GPT_value_of_x_l1911_191181

theorem value_of_x (x : ℝ) (h : x = 88 * 1.2) : x = 105.6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1911_191181


namespace NUMINAMATH_GPT_johns_previous_salary_l1911_191167

-- Conditions
def johns_new_salary : ℝ := 70
def percent_increase : ℝ := 0.16666666666666664

-- Statement
theorem johns_previous_salary :
  ∃ x : ℝ, x + percent_increase * x = johns_new_salary ∧ x = 60 :=
by
  sorry

end NUMINAMATH_GPT_johns_previous_salary_l1911_191167


namespace NUMINAMATH_GPT_number_of_oxygen_atoms_l1911_191105

/-- Given a compound has 1 H, 1 Cl, and a certain number of O atoms and the molecular weight of the compound is 68 g/mol,
    prove that the number of O atoms is 2. -/
theorem number_of_oxygen_atoms (atomic_weight_H: ℝ) (atomic_weight_Cl: ℝ) (atomic_weight_O: ℝ) (molecular_weight: ℝ) (n : ℕ):
    atomic_weight_H = 1.0 →
    atomic_weight_Cl = 35.5 →
    atomic_weight_O = 16.0 →
    molecular_weight = 68.0 →
    molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O →
    n = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_oxygen_atoms_l1911_191105


namespace NUMINAMATH_GPT_negative_sixty_represents_expenditure_l1911_191126

def positive_represents_income (x : ℤ) : Prop := x > 0
def negative_represents_expenditure (x : ℤ) : Prop := x < 0

theorem negative_sixty_represents_expenditure :
  negative_represents_expenditure (-60) ∧ abs (-60) = 60 :=
by
  sorry

end NUMINAMATH_GPT_negative_sixty_represents_expenditure_l1911_191126


namespace NUMINAMATH_GPT_max_integer_value_l1911_191128

theorem max_integer_value (x : ℝ) : 
  ∃ (n : ℤ), n = 15 ∧ ∀ x : ℝ, 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ n :=
by
  sorry

end NUMINAMATH_GPT_max_integer_value_l1911_191128


namespace NUMINAMATH_GPT_equiv_proof_problem_l1911_191182

theorem equiv_proof_problem (b c : ℝ) (h1 : b ≠ 1 ∨ c ≠ 1) (h2 : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2 * n) (h3 : b * 1 = c * c) : 
  100 * (b - c) = 75 := 
by sorry

end NUMINAMATH_GPT_equiv_proof_problem_l1911_191182


namespace NUMINAMATH_GPT_average_age_of_others_when_youngest_was_born_l1911_191179

noncomputable def average_age_when_youngest_was_born (total_people : ℕ) (average_age : ℕ) (youngest_age : ℕ) : ℚ :=
  let total_age := total_people * average_age
  let age_without_youngest := total_age - youngest_age
  age_without_youngest / (total_people - 1)

theorem average_age_of_others_when_youngest_was_born :
  average_age_when_youngest_was_born 7 30 7 = 33.833 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_others_when_youngest_was_born_l1911_191179


namespace NUMINAMATH_GPT_roots_subtraction_l1911_191161

theorem roots_subtraction (a b : ℝ) (h_roots : a * b = 20 ∧ a + b = 12) (h_order : a > b) : a - b = 8 :=
sorry

end NUMINAMATH_GPT_roots_subtraction_l1911_191161


namespace NUMINAMATH_GPT_tourists_count_l1911_191160

theorem tourists_count :
  ∃ (n : ℕ), (1 / 2 * n + 1 / 3 * n + 1 / 4 * n = 39) :=
by
  use 36
  sorry

end NUMINAMATH_GPT_tourists_count_l1911_191160


namespace NUMINAMATH_GPT_GP_GQ_GR_proof_l1911_191164

open Real

noncomputable def GP_GQ_GR_sum (XY XZ YZ : ℝ) (G : (ℝ × ℝ × ℝ)) (P Q R : (ℝ × ℝ × ℝ)) : ℝ :=
  let GP := dist G P
  let GQ := dist G Q
  let GR := dist G R
  GP + GQ + GR

theorem GP_GQ_GR_proof (XY XZ YZ : ℝ) (hXY : XY = 4) (hXZ : XZ = 3) (hYZ : YZ = 5)
  (G P Q R : (ℝ × ℝ × ℝ))
  (GP := dist G P) (GQ := dist G Q) (GR := dist G R)
  (hG : GP_GQ_GR_sum XY XZ YZ G P Q R = GP + GQ + GR) :
  GP + GQ + GR = 47 / 15 :=
sorry

end NUMINAMATH_GPT_GP_GQ_GR_proof_l1911_191164


namespace NUMINAMATH_GPT_cookie_cost_1_l1911_191172

theorem cookie_cost_1 (C : ℝ) 
  (h1 : ∀ c, c > 0 → 1.2 * c = c + 0.2 * c)
  (h2 : 50 * (1.2 * C) = 60) :
  C = 1 :=
by
  sorry

end NUMINAMATH_GPT_cookie_cost_1_l1911_191172


namespace NUMINAMATH_GPT_intersection_A_B_l1911_191103

-- Definitions for sets A and B
def A : Set ℝ := { x | ∃ y : ℝ, x + y^2 = 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

-- The proof goal to show the intersection of sets A and B
theorem intersection_A_B : A ∩ B = { z | -1 ≤ z ∧ z ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1911_191103


namespace NUMINAMATH_GPT_right_triangle_integers_solutions_l1911_191148

theorem right_triangle_integers_solutions :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a^2 + b^2 = c^2 ∧ (a + b + c : ℕ) = (1 / 2 * a * b : ℚ) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
sorry

end NUMINAMATH_GPT_right_triangle_integers_solutions_l1911_191148


namespace NUMINAMATH_GPT_domino_chain_can_be_built_l1911_191101

def domino_chain_possible : Prop :=
  let total_pieces := 28
  let pieces_with_sixes_removed := 7
  let remaining_pieces := total_pieces - pieces_with_sixes_removed
  (∀ n : ℕ, n < 6 → (∃ k : ℕ, k = 6) → (remaining_pieces % 2 = 0))

theorem domino_chain_can_be_built (h : domino_chain_possible) : Prop :=
  sorry

end NUMINAMATH_GPT_domino_chain_can_be_built_l1911_191101


namespace NUMINAMATH_GPT_polar_to_cartesian_l1911_191193

-- Definitions for the polar coordinates conversion
noncomputable def polar_to_cartesian_eq (C : ℝ → ℝ → Prop) :=
  ∀ (ρ θ : ℝ), (ρ^2 * (1 + 3 * (Real.sin θ)^2) = 4) → C (ρ * (Real.cos θ)) (ρ * (Real.sin θ))

-- Define the Cartesian equation
def cartesian_eq (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 = 1)

-- The main theorem
theorem polar_to_cartesian 
  (C : ℝ → ℝ → Prop)
  (h : polar_to_cartesian_eq C) :
  ∀ x y : ℝ, C x y ↔ cartesian_eq x y :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l1911_191193


namespace NUMINAMATH_GPT_factorize_expression_l1911_191151

theorem factorize_expression (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1911_191151


namespace NUMINAMATH_GPT_larry_wins_prob_l1911_191196

def probability_larry_wins (pLarry pJulius : ℚ) : ℚ :=
  let r := (1 - pLarry) * (1 - pJulius)
  pLarry * (1 / (1 - r))

theorem larry_wins_prob : probability_larry_wins (2 / 3) (1 / 3) = 6 / 7 :=
by
  -- Definitions for probabilities
  let pLarry := 2 / 3
  let pJulius := 1 / 3
  have r := (1 - pLarry) * (1 - pJulius)
  have S := pLarry * (1 / (1 - r))
  -- Expected result
  have expected := 6 / 7
  -- Prove the result equals the expected
  sorry

end NUMINAMATH_GPT_larry_wins_prob_l1911_191196


namespace NUMINAMATH_GPT_division_by_negative_divisor_l1911_191169

theorem division_by_negative_divisor : 15 / (-3) = -5 :=
by sorry

end NUMINAMATH_GPT_division_by_negative_divisor_l1911_191169


namespace NUMINAMATH_GPT_solve_m_value_l1911_191118

-- Definitions for conditions
def hyperbola_eq (m : ℝ) : Prop := ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3
def has_focus (m : ℝ) : Prop := (∃ f1 f2 : ℝ, f1 = 0 ∧ f2 = 2)

-- Statement of the problem to prove
theorem solve_m_value (m : ℝ) (h_eq : hyperbola_eq m) (h_focus : has_focus m) : m = -1 :=
sorry

end NUMINAMATH_GPT_solve_m_value_l1911_191118


namespace NUMINAMATH_GPT_quadratic_equal_roots_l1911_191136

theorem quadratic_equal_roots (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 1 = 0 → x = -k / 2) ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equal_roots_l1911_191136


namespace NUMINAMATH_GPT_point_in_third_quadrant_l1911_191186

def quadrant_of_point (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "first"
  else if x < 0 ∧ y > 0 then "second"
  else if x < 0 ∧ y < 0 then "third"
  else if x > 0 ∧ y < 0 then "fourth"
  else "on_axis"

theorem point_in_third_quadrant : quadrant_of_point (-2) (-3) = "third" :=
  by sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l1911_191186


namespace NUMINAMATH_GPT_smaller_than_negative_one_l1911_191134

theorem smaller_than_negative_one :
  ∃ x ∈ ({0, -1/2, 1, -2} : Set ℝ), x < -1 ∧ x = -2 :=
by
  -- the proof part is skipped
  sorry

end NUMINAMATH_GPT_smaller_than_negative_one_l1911_191134


namespace NUMINAMATH_GPT_simplify_expression_l1911_191199

variable (x y : ℝ)

theorem simplify_expression:
  3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1911_191199


namespace NUMINAMATH_GPT_butterfly_cocoon_l1911_191117

theorem butterfly_cocoon (c l : ℕ) (h1 : l + c = 120) (h2 : l = 3 * c) : c = 30 :=
by
  sorry

end NUMINAMATH_GPT_butterfly_cocoon_l1911_191117


namespace NUMINAMATH_GPT_francis_violin_count_l1911_191111

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end NUMINAMATH_GPT_francis_violin_count_l1911_191111


namespace NUMINAMATH_GPT_canadian_ratio_correct_l1911_191107

-- The total number of scientists
def total_scientists : ℕ := 70

-- Half of the scientists are from Europe
def european_scientists : ℕ := total_scientists / 2

-- The number of scientists from the USA
def usa_scientists : ℕ := 21

-- The number of Canadian scientists
def canadian_scientists : ℕ := total_scientists - european_scientists - usa_scientists

-- The ratio of the number of Canadian scientists to the total number of scientists
def canadian_ratio : ℚ := canadian_scientists / total_scientists

-- Prove that the ratio is 1:5
theorem canadian_ratio_correct : canadian_ratio = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_canadian_ratio_correct_l1911_191107


namespace NUMINAMATH_GPT_bill_due_in_months_l1911_191156

theorem bill_due_in_months
  (TD : ℝ) (FV : ℝ) (R_annual : ℝ) (m : ℝ) 
  (h₀ : TD = 270)
  (h₁ : FV = 2520)
  (h₂ : R_annual = 16) :
  m = 9 :=
by
  sorry

end NUMINAMATH_GPT_bill_due_in_months_l1911_191156


namespace NUMINAMATH_GPT_selling_price_before_brokerage_l1911_191195

variables (CR BR SP : ℝ)
variables (hCR : CR = 120.50) (hBR : BR = 1 / 400)

theorem selling_price_before_brokerage :
  SP = (CR * 400) / (399) := 
by
  sorry

end NUMINAMATH_GPT_selling_price_before_brokerage_l1911_191195


namespace NUMINAMATH_GPT_standard_deviation_bound_l1911_191109

theorem standard_deviation_bound (mu sigma : ℝ) (h_mu : mu = 51) (h_ineq : mu - 3 * sigma > 44) : sigma < 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_standard_deviation_bound_l1911_191109


namespace NUMINAMATH_GPT_joggers_difference_l1911_191127

theorem joggers_difference (Tyson_joggers Alexander_joggers Christopher_joggers : ℕ) 
  (h1 : Alexander_joggers = Tyson_joggers + 22) 
  (h2 : Christopher_joggers = 20 * Tyson_joggers)
  (h3 : Christopher_joggers = 80) : 
  Christopher_joggers - Alexander_joggers = 54 :=
by 
  sorry

end NUMINAMATH_GPT_joggers_difference_l1911_191127


namespace NUMINAMATH_GPT_total_parents_surveyed_l1911_191150

-- Define the given conditions
def percent_agree : ℝ := 0.20
def percent_disagree : ℝ := 0.80
def disagreeing_parents : ℕ := 640

-- Define the statement to prove
theorem total_parents_surveyed :
  ∃ (total_parents : ℕ), disagreeing_parents = (percent_disagree * total_parents) ∧ total_parents = 800 :=
by
  sorry

end NUMINAMATH_GPT_total_parents_surveyed_l1911_191150


namespace NUMINAMATH_GPT_selling_prices_max_profit_strategy_l1911_191141

theorem selling_prices (x y : ℕ) (hx : y - x = 30) (hy : 2 * x + 3 * y = 740) : x = 130 ∧ y = 160 :=
by
  sorry

theorem max_profit_strategy (m : ℕ) (hm : 20 ≤ m ∧ m ≤ 80) 
(hcost : 90 * m + 110 * (80 - m) ≤ 8400) : m = 20 ∧ (80 - m) = 60 :=
by
  sorry

end NUMINAMATH_GPT_selling_prices_max_profit_strategy_l1911_191141


namespace NUMINAMATH_GPT_f_neg1_plus_f_2_l1911_191173

def f (x : ℤ) : ℤ :=
  if x ≤ 0 then 4 * x else 2 * x

theorem f_neg1_plus_f_2 : f (-1) + f 2 = 0 := 
by
  -- Definition of f is provided above and conditions are met in that.
  sorry

end NUMINAMATH_GPT_f_neg1_plus_f_2_l1911_191173


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1911_191115

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 6

theorem solution_set_of_inequality (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ (-1/3 : ℝ) < m ∧ m < 3 :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1911_191115


namespace NUMINAMATH_GPT_max_area_quadrilateral_l1911_191114

theorem max_area_quadrilateral (a b c d : ℝ) (h1 : a = 1) (h2 : b = 4) (h3 : c = 7) (h4 : d = 8) : 
  ∃ A : ℝ, (A ≤ (1/2) * 1 * 8 + (1/2) * 4 * 7) ∧ (A = 18) :=
by
  sorry

end NUMINAMATH_GPT_max_area_quadrilateral_l1911_191114


namespace NUMINAMATH_GPT_outfit_choices_l1911_191129

-- Define the numbers of shirts, pants, and hats.
def num_shirts : ℕ := 6
def num_pants : ℕ := 7
def num_hats : ℕ := 6

-- Define the number of colors and the constraints.
def num_colors : ℕ := 6

-- The total number of outfits without restrictions.
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- Number of outfits where all items are the same color.
def same_color_outfits : ℕ := num_colors

-- Number of outfits where the shirt and pants are the same color.
def same_shirt_pants_color_outfits : ℕ := num_colors + 1  -- accounting for the extra pair of pants

-- The total number of valid outfits calculated.
def valid_outfits : ℕ :=
  total_outfits - same_color_outfits - same_shirt_pants_color_outfits

-- The theorem statement asserting the correct answer.
theorem outfit_choices : valid_outfits = 239 := by
  sorry

end NUMINAMATH_GPT_outfit_choices_l1911_191129


namespace NUMINAMATH_GPT_total_students_l1911_191197

theorem total_students (ratio_boys : ℕ) (ratio_girls : ℕ) (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8) (h_ratio_girls : ratio_girls = 5) (h_num_girls : num_girls = 175) : 
  ratio_boys * (num_girls / ratio_girls) + num_girls = 455 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1911_191197


namespace NUMINAMATH_GPT_ratio_y_x_l1911_191138

variable {c x y : ℝ}

-- Conditions stated as assumptions
theorem ratio_y_x (h1 : x = 0.80 * c) (h2 : y = 1.25 * c) : y / x = 25 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_y_x_l1911_191138


namespace NUMINAMATH_GPT_point_is_in_second_quadrant_l1911_191116

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end NUMINAMATH_GPT_point_is_in_second_quadrant_l1911_191116


namespace NUMINAMATH_GPT_snack_cost_is_five_l1911_191144

-- Define the cost of one ticket
def ticket_cost : ℕ := 18

-- Define the total number of people
def total_people : ℕ := 4

-- Define the total cost for tickets and snacks
def total_cost : ℕ := 92

-- Define the unknown cost of one set of snacks
def snack_cost := 92 - 4 * 18

-- Statement asserting that the cost of one set of snacks is $5
theorem snack_cost_is_five : snack_cost = 5 := by
  sorry

end NUMINAMATH_GPT_snack_cost_is_five_l1911_191144


namespace NUMINAMATH_GPT_max_students_gcd_l1911_191113

def numPens : Nat := 1802
def numPencils : Nat := 1203
def numErasers : Nat := 1508
def numNotebooks : Nat := 2400

theorem max_students_gcd : Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numErasers) numNotebooks = 1 := by
  sorry

end NUMINAMATH_GPT_max_students_gcd_l1911_191113


namespace NUMINAMATH_GPT_zoe_total_money_l1911_191102

def numberOfPeople : ℕ := 6
def sodaCostPerBottle : ℝ := 0.5
def pizzaCostPerSlice : ℝ := 1.0

theorem zoe_total_money :
  numberOfPeople * sodaCostPerBottle + numberOfPeople * pizzaCostPerSlice = 9 := 
by
  sorry

end NUMINAMATH_GPT_zoe_total_money_l1911_191102


namespace NUMINAMATH_GPT_max_cubes_fit_l1911_191157

theorem max_cubes_fit (L S : ℕ) (hL : L = 10) (hS : S = 2) : (L * L * L) / (S * S * S) = 125 := by
  sorry

end NUMINAMATH_GPT_max_cubes_fit_l1911_191157


namespace NUMINAMATH_GPT_adjacent_probability_is_2_over_7_l1911_191104

variable (n : Nat := 5) -- number of student performances
variable (m : Nat := 2) -- number of teacher performances

/-- Total number of ways to insert two performances
    (ignoring adjacency constraints) into the program list. -/
def total_insertion_ways : Nat :=
  Fintype.card (Fin (n + m))

/-- Number of ways to insert two performances such that they are adjacent. -/
def adjacent_insertion_ways : Nat :=
  Fintype.card (Fin (n + 1))

/-- Probability that two specific performances are adjacent in a program list. -/
def adjacent_probability : ℚ :=
  adjacent_insertion_ways / total_insertion_ways

theorem adjacent_probability_is_2_over_7 :
  adjacent_probability = (2 : ℚ) / 7 := by
  sorry

end NUMINAMATH_GPT_adjacent_probability_is_2_over_7_l1911_191104


namespace NUMINAMATH_GPT_find_side_b_l1911_191106

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_side_b_l1911_191106
