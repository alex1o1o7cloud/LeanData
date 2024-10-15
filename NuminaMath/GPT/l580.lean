import Mathlib

namespace NUMINAMATH_GPT_waiter_earnings_l580_58016

theorem waiter_earnings
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (customers_tipped : total_customers - no_tip_customers = 3)
  (tips_per_customer : tip_amount = 9) :
  (total_customers - no_tip_customers) * tip_amount = 27 := by
  sorry

end NUMINAMATH_GPT_waiter_earnings_l580_58016


namespace NUMINAMATH_GPT_find_c_l580_58008

theorem find_c (b c : ℤ) (H : (b - 4) / (2 * b + 42) = c / 6) : c = 2 := 
sorry

end NUMINAMATH_GPT_find_c_l580_58008


namespace NUMINAMATH_GPT_problem1_problem2_l580_58002

open Set

variable {x y z a b : ℝ}

-- Problem 1: Prove the inequality
theorem problem1 (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 :=
by
  sorry

-- Problem 2: Prove the range of 10a - 5b is [−1, 20]
theorem problem2 (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b ∧ 2 * a + b ≤ 4)
  (h2 : -1 ≤ a - 2 * b ∧ a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l580_58002


namespace NUMINAMATH_GPT_billy_age_l580_58019

variable (B J : ℕ)

theorem billy_age (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end NUMINAMATH_GPT_billy_age_l580_58019


namespace NUMINAMATH_GPT_area_inside_arcs_outside_square_l580_58083

theorem area_inside_arcs_outside_square (r : ℝ) (θ : ℝ) (L : ℝ) (a b c d : ℝ) :
  r = 6 ∧ θ = 45 ∧ L = 12 ∧ a = 15 ∧ b = 0 ∧ c = 15 ∧ d = 144 →
  (a + b + c + d = 174) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_area_inside_arcs_outside_square_l580_58083


namespace NUMINAMATH_GPT_find_f7_l580_58080

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 7

theorem find_f7 (a b : ℝ) (h : f (-7) a b = -17) : f (7) a b = 31 := 
by
  sorry

end NUMINAMATH_GPT_find_f7_l580_58080


namespace NUMINAMATH_GPT_find_number_subtract_four_l580_58074

theorem find_number_subtract_four (x : ℤ) (h : 35 + 3 * x = 50) : x - 4 = 1 := by
  sorry

end NUMINAMATH_GPT_find_number_subtract_four_l580_58074


namespace NUMINAMATH_GPT_population_in_terms_of_t_l580_58005

noncomputable def boys_girls_teachers_total (b g t : ℕ) : ℕ :=
  b + g + t

theorem population_in_terms_of_t (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) :
  boys_girls_teachers_total b g t = 26 * t :=
by
  sorry

end NUMINAMATH_GPT_population_in_terms_of_t_l580_58005


namespace NUMINAMATH_GPT_remainder_sequences_mod_1000_l580_58014

theorem remainder_sequences_mod_1000 :
  ∃ m, (m = 752) ∧ (m % 1000 = 752) ∧ 
  (∃ (a : ℕ → ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ 6 → (a i) - i % 2 = 1), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 6 → a i ≤ a j) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 6 → 1 ≤ a i ∧ a i ≤ 1500)
  ) := by
    -- proof would go here
    sorry

end NUMINAMATH_GPT_remainder_sequences_mod_1000_l580_58014


namespace NUMINAMATH_GPT_largest_number_of_positive_consecutive_integers_l580_58066

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end NUMINAMATH_GPT_largest_number_of_positive_consecutive_integers_l580_58066


namespace NUMINAMATH_GPT_find_range_of_a_l580_58097

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x => a * (x - 2 * Real.exp 1) * Real.log x + 1

def range_of_a (a : ℝ) : Prop :=
  (a < 0 ∨ a > 1 / Real.exp 1)

theorem find_range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ range_of_a a := by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l580_58097


namespace NUMINAMATH_GPT_value_of_expression_l580_58037

theorem value_of_expression {p q : ℝ} (hp : 3 * p^2 + 9 * p - 21 = 0) (hq : 3 * q^2 + 9 * q - 21 = 0) : 
  (3 * p - 4) * (6 * q - 8) = 122 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l580_58037


namespace NUMINAMATH_GPT_equation_1_equation_2_l580_58044

theorem equation_1 (x : ℝ) : x^2 - 1 = 8 ↔ x = 3 ∨ x = -3 :=
by sorry

theorem equation_2 (x : ℝ) : (x + 4)^3 = -64 ↔ x = -8 :=
by sorry

end NUMINAMATH_GPT_equation_1_equation_2_l580_58044


namespace NUMINAMATH_GPT_three_digit_powers_of_two_count_l580_58096

theorem three_digit_powers_of_two_count : 
  ∃ n_count : ℕ, (∀ n : ℕ, (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9)) ∧ n_count = 3 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_powers_of_two_count_l580_58096


namespace NUMINAMATH_GPT_rebecca_haircut_charge_l580_58025

-- Define the conditions
variable (H : ℕ) -- Charge for a haircut
def perm_charge : ℕ := 40
def dye_charge : ℕ := 60
def dye_cost : ℕ := 10
def haircuts_today : ℕ := 4
def perms_today : ℕ := 1
def dye_jobs_today : ℕ := 2
def tips_today : ℕ := 50
def total_amount_end_day : ℕ := 310

-- State the proof problem
theorem rebecca_haircut_charge :
  4 * H + perms_today * perm_charge + dye_jobs_today * dye_charge + tips_today - dye_jobs_today * dye_cost = total_amount_end_day →
  H = 30 :=
by
  sorry

end NUMINAMATH_GPT_rebecca_haircut_charge_l580_58025


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l580_58073

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 2 + a 12 = 32) : a 3 + a 11 = 32 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l580_58073


namespace NUMINAMATH_GPT_max_difference_proof_l580_58063

-- Define the revenue function R(x)
def R (x : ℕ+) : ℝ := 3000 * (x : ℝ) - 20 * (x : ℝ) ^ 2

-- Define the cost function C(x)
def C (x : ℕ+) : ℝ := 500 * (x : ℝ) + 4000

-- Define the profit function P(x) as revenue minus cost
def P (x : ℕ+) : ℝ := R x - C x

-- Define the marginal function M
def M (f : ℕ+ → ℝ) (x : ℕ+) : ℝ := f (⟨x + 1, Nat.succ_pos x⟩) - f x

-- Define the marginal profit function MP(x)
def MP (x : ℕ+) : ℝ := M P x

-- Statement of the proof
theorem max_difference_proof : 
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → P x ≤ P x_max) → -- P achieves its maximum at some x_max within constraints
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → MP x ≤ MP x_max) → -- MP achieves its maximum at some x_max within constraints
  (P x_max - MP x_max = 71680) := 
sorry -- proof omitted

end NUMINAMATH_GPT_max_difference_proof_l580_58063


namespace NUMINAMATH_GPT_equation_solution_l580_58092

theorem equation_solution (x : ℝ) :
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) + 1 / (x + 8) = 0) →
  (x = -4 - 2 * Real.sqrt 3) ∨ (x = 2 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_equation_solution_l580_58092


namespace NUMINAMATH_GPT_inequality_proof_l580_58023

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l580_58023


namespace NUMINAMATH_GPT_cos_30_eq_sqrt3_div_2_l580_58093

theorem cos_30_eq_sqrt3_div_2 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_30_eq_sqrt3_div_2_l580_58093


namespace NUMINAMATH_GPT_right_angled_triangle_area_l580_58034

theorem right_angled_triangle_area 
  (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 18) (h3 : a^2 + b^2 + c^2 = 128) : 
  (1/2) * a * b = 9 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_right_angled_triangle_area_l580_58034


namespace NUMINAMATH_GPT_triangle_heights_inequality_l580_58018

variable {R : Type} [OrderedRing R]

theorem triangle_heights_inequality (m_a m_b m_c s : R) 
  (h_m_a_nonneg : 0 ≤ m_a) (h_m_b_nonneg : 0 ≤ m_b) (h_m_c_nonneg : 0 ≤ m_c)
  (h_s_nonneg : 0 ≤ s) : 
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_heights_inequality_l580_58018


namespace NUMINAMATH_GPT_calculate_total_tulips_l580_58001

def number_of_red_tulips_for_eyes := 8 * 2
def number_of_purple_tulips_for_eyebrows := 5 * 2
def number_of_red_tulips_for_nose := 12
def number_of_red_tulips_for_smile := 18
def number_of_yellow_tulips_for_background := 9 * number_of_red_tulips_for_smile

def total_number_of_tulips : ℕ :=
  number_of_red_tulips_for_eyes + 
  number_of_red_tulips_for_nose + 
  number_of_red_tulips_for_smile + 
  number_of_purple_tulips_for_eyebrows + 
  number_of_yellow_tulips_for_background

theorem calculate_total_tulips : total_number_of_tulips = 218 := by
  sorry

end NUMINAMATH_GPT_calculate_total_tulips_l580_58001


namespace NUMINAMATH_GPT_polynomial_has_real_root_l580_58054

open Real Polynomial

variable {c d : ℝ}
variable {P : Polynomial ℝ}

theorem polynomial_has_real_root (hP1 : ∀ n : ℕ, c * |(n : ℝ)|^3 ≤ |P.eval (n : ℝ)|)
                                (hP2 : ∀ n : ℕ, |P.eval (n : ℝ)| ≤ d * |(n : ℝ)|^3)
                                (hc : 0 < c) (hd : 0 < d) : 
                                ∃ x : ℝ, P.eval x = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_has_real_root_l580_58054


namespace NUMINAMATH_GPT_probability_non_obtuse_l580_58059

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end NUMINAMATH_GPT_probability_non_obtuse_l580_58059


namespace NUMINAMATH_GPT_compare_neg_fractions_l580_58048

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (5 / 7 : ℝ) := 
by 
  sorry

end NUMINAMATH_GPT_compare_neg_fractions_l580_58048


namespace NUMINAMATH_GPT_sum_of_common_divisors_36_48_l580_58061

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end NUMINAMATH_GPT_sum_of_common_divisors_36_48_l580_58061


namespace NUMINAMATH_GPT_ellipse_semi_focal_distance_range_l580_58089

theorem ellipse_semi_focal_distance_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (h_ellipse : a^2 = b^2 + c^2) :
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_ellipse_semi_focal_distance_range_l580_58089


namespace NUMINAMATH_GPT_simple_interest_l580_58027

theorem simple_interest (P R T : ℝ) (hP : P = 8965) (hR : R = 9) (hT : T = 5) : 
    (P * R * T) / 100 = 806.85 := 
by 
  sorry

end NUMINAMATH_GPT_simple_interest_l580_58027


namespace NUMINAMATH_GPT_no_integer_roots_l580_58017

theorem no_integer_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) : ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l580_58017


namespace NUMINAMATH_GPT_true_proposition_is_D_l580_58076

open Real

theorem true_proposition_is_D :
  (∃ x_0 : ℝ, exp x_0 ≤ 0) = False ∧
  (∀ x : ℝ, 2 ^ x > x ^ 2) = False ∧
  (∀ a b : ℝ, a + b = 0 ↔ a / b = -1) = False ∧
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) = True :=
by
    sorry

end NUMINAMATH_GPT_true_proposition_is_D_l580_58076


namespace NUMINAMATH_GPT_negate_proposition_l580_58062

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^2 + 2 > 6)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) :=
by sorry

end NUMINAMATH_GPT_negate_proposition_l580_58062


namespace NUMINAMATH_GPT_y1_greater_than_y2_l580_58087

-- Define the function and points
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m

-- Define the points A and B on the parabola
def A_y1 (m : ℝ) : ℝ := parabola 0 m
def B_y2 (m : ℝ) : ℝ := parabola 1 m

-- Theorem statement
theorem y1_greater_than_y2 (m : ℝ) : A_y1 m > B_y2 m := 
  sorry

end NUMINAMATH_GPT_y1_greater_than_y2_l580_58087


namespace NUMINAMATH_GPT_number_of_x_intercepts_l580_58036

def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem number_of_x_intercepts : ∃! (x : ℝ), ∃ (y : ℝ), parabola y = x ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_x_intercepts_l580_58036


namespace NUMINAMATH_GPT_tangerines_more_than_oranges_l580_58026

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end NUMINAMATH_GPT_tangerines_more_than_oranges_l580_58026


namespace NUMINAMATH_GPT_one_room_cheaper_by_l580_58090

-- Define the initial prices of the apartments
variables (a b : ℝ)

-- Define the increase rates and the new prices
def new_price_one_room := 1.21 * a
def new_price_two_room := 1.11 * b
def new_total_price := 1.15 * (a + b)

-- The main theorem encapsulating the problem
theorem one_room_cheaper_by : a + b ≠ 0 → 1.21 * a + 1.11 * b = 1.15 * (a + b) → b / a = 1.5 :=
by
  intro h_non_zero h_prices
  -- we assume the main theorem is true to structure the goal state
  sorry

end NUMINAMATH_GPT_one_room_cheaper_by_l580_58090


namespace NUMINAMATH_GPT_zachary_crunches_more_than_pushups_l580_58000

def push_ups_zachary : ℕ := 46
def crunches_zachary : ℕ := 58

theorem zachary_crunches_more_than_pushups : crunches_zachary - push_ups_zachary = 12 := by
  sorry

end NUMINAMATH_GPT_zachary_crunches_more_than_pushups_l580_58000


namespace NUMINAMATH_GPT_find_x_l580_58007

theorem find_x (x : ℚ) (h : (3 - x) / (2 - x) - 1 / (x - 2) = 3) : x = 1 := 
  sorry

end NUMINAMATH_GPT_find_x_l580_58007


namespace NUMINAMATH_GPT_average_speed_l580_58012

theorem average_speed (uphill_speed downhill_speed : ℚ) (t : ℚ) (v : ℚ) :
  uphill_speed = 4 →
  downhill_speed = 6 →
  (1 / uphill_speed + 1 / downhill_speed = t) →
  (v * t = 2) →
  v = 4.8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_speed_l580_58012


namespace NUMINAMATH_GPT_value_of_a_minus_b_l580_58006

theorem value_of_a_minus_b (a b : ℝ) (h₁ : |a| = 2) (h₂ : |b| = 5) (h₃ : a < b) :
  a - b = -3 ∨ a - b = -7 := 
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l580_58006


namespace NUMINAMATH_GPT_greatest_value_sum_eq_24_l580_58079

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end NUMINAMATH_GPT_greatest_value_sum_eq_24_l580_58079


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_l580_58013

-- Given the equation of the ellipse x^2 + k * y^2 = 2 with foci on the y-axis,
-- prove that the range of k such that the ellipse is oriented with foci on the y-axis is (0, 1).
theorem ellipse_foci_y_axis (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ a > 0 ∧ b > 0 ∧ b / a = k ∧ x^2 + k * y^2 = 2 :=
sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_l580_58013


namespace NUMINAMATH_GPT_last_three_digits_2005_pow_2005_l580_58047

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_digits_2005_pow_2005 :
  last_three_digits (2005 ^ 2005) = 125 :=
sorry

end NUMINAMATH_GPT_last_three_digits_2005_pow_2005_l580_58047


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_rational_expression_of_trig_l580_58010

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  Real.tan (α + Real.pi / 4) = -1 / 7 := 
by 
  sorry

theorem rational_expression_of_trig (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_rational_expression_of_trig_l580_58010


namespace NUMINAMATH_GPT_goose_eggs_count_l580_58009

theorem goose_eggs_count (E : ℕ)
  (h1 : (2/3 : ℚ) * E ≥ 0)
  (h2 : (3/4 : ℚ) * (2/3 : ℚ) * E ≥ 0)
  (h3 : 100 = (2/5 : ℚ) * (3/4 : ℚ) * (2/3 : ℚ) * E) :
  E = 500 := by
  sorry

end NUMINAMATH_GPT_goose_eggs_count_l580_58009


namespace NUMINAMATH_GPT_shop_owner_cheat_percentage_l580_58022

def CP : ℝ := 100
def cheating_buying : ℝ := 0.15  -- 15% cheating
def actual_cost_price : ℝ := CP * (1 + cheating_buying)  -- $115
def profit_percentage : ℝ := 43.75

theorem shop_owner_cheat_percentage :
  ∃ x : ℝ, profit_percentage = ((CP - x * CP / 100 - actual_cost_price) / actual_cost_price * 100) ∧ x = 65.26 :=
by
  sorry

end NUMINAMATH_GPT_shop_owner_cheat_percentage_l580_58022


namespace NUMINAMATH_GPT_number_of_tests_in_series_l580_58032

theorem number_of_tests_in_series (S : ℝ) (n : ℝ) :
  (S + 97) / n = 90 →
  (S + 73) / n = 87 →
  n = 8 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_tests_in_series_l580_58032


namespace NUMINAMATH_GPT_inequality_x_pow_n_ge_n_x_l580_58072

theorem inequality_x_pow_n_ge_n_x (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x > -1) (h3 : n > 0) : 
  (1 + x)^n ≥ n * x := by
  sorry

end NUMINAMATH_GPT_inequality_x_pow_n_ge_n_x_l580_58072


namespace NUMINAMATH_GPT_angle_sum_eq_pi_div_2_l580_58065

open Real

theorem angle_sum_eq_pi_div_2 (θ1 θ2 : ℝ) (h1 : 0 < θ1 ∧ θ1 < π / 2) (h2 : 0 < θ2 ∧ θ2 < π / 2)
  (h : (sin θ1)^2020 / (cos θ2)^2018 + (cos θ1)^2020 / (sin θ2)^2018 = 1) :
  θ1 + θ2 = π / 2 :=
sorry

end NUMINAMATH_GPT_angle_sum_eq_pi_div_2_l580_58065


namespace NUMINAMATH_GPT_sin_squared_alpha_eq_one_add_sin_squared_beta_l580_58085

variable {α θ β : ℝ}

theorem sin_squared_alpha_eq_one_add_sin_squared_beta
  (h1 : Real.sin α = Real.sin θ + Real.cos θ)
  (h2 : Real.sin β ^ 2 = 2 * Real.sin θ * Real.cos θ) :
  Real.sin α ^ 2 = 1 + Real.sin β ^ 2 := 
sorry

end NUMINAMATH_GPT_sin_squared_alpha_eq_one_add_sin_squared_beta_l580_58085


namespace NUMINAMATH_GPT_expression_equals_minus_0p125_l580_58038

-- Define the expression
def compute_expression : ℝ := 0.125^8 * (-8)^7

-- State the theorem to prove
theorem expression_equals_minus_0p125 : compute_expression = -0.125 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_equals_minus_0p125_l580_58038


namespace NUMINAMATH_GPT_sample_size_is_10_l580_58031

def product := Type

noncomputable def number_of_products : ℕ := 80
noncomputable def selected_products_for_quality_inspection : ℕ := 10

theorem sample_size_is_10 
  (N : ℕ) (sample_size : ℕ) 
  (hN : N = 80) 
  (h_sample_size : sample_size = 10) : 
  sample_size = 10 :=
by 
  sorry

end NUMINAMATH_GPT_sample_size_is_10_l580_58031


namespace NUMINAMATH_GPT_value_of_7_star_3_l580_58069

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - a * b

theorem value_of_7_star_3 : star 7 3 = 16 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_value_of_7_star_3_l580_58069


namespace NUMINAMATH_GPT_solve_for_constants_l580_58084

def f (x : ℤ) (a b c : ℤ) : ℤ :=
if x > 0 then 2 * a * x + 4
else if x = 0 then a + b
else 3 * b * x + 2 * c

theorem solve_for_constants :
  ∃ a b c : ℤ, 
    f 1 a b c = 6 ∧ 
    f 0 a b c = 7 ∧ 
    f (-1) a b c = -4 ∧ 
    a + b + c = 14 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_constants_l580_58084


namespace NUMINAMATH_GPT_sky_color_change_l580_58046

theorem sky_color_change (hours: ℕ) (colors: ℕ) (minutes_per_hour: ℕ) 
                          (H1: hours = 2) 
                          (H2: colors = 12) 
                          (H3: minutes_per_hour = 60) : 
                          (hours * minutes_per_hour) / colors = 10 := 
by
  sorry

end NUMINAMATH_GPT_sky_color_change_l580_58046


namespace NUMINAMATH_GPT_Hillary_reading_time_on_sunday_l580_58049

-- Define the assigned reading times for both books
def assigned_time_book_a : ℕ := 60 -- minutes
def assigned_time_book_b : ℕ := 45 -- minutes

-- Define the reading times already spent on each book
def time_spent_friday_book_a : ℕ := 16 -- minutes
def time_spent_saturday_book_a : ℕ := 28 -- minutes
def time_spent_saturday_book_b : ℕ := 15 -- minutes

-- Calculate the total time already read for each book
def total_time_read_book_a : ℕ := time_spent_friday_book_a + time_spent_saturday_book_a
def total_time_read_book_b : ℕ := time_spent_saturday_book_b

-- Calculate the remaining time needed for each book
def remaining_time_book_a : ℕ := assigned_time_book_a - total_time_read_book_a
def remaining_time_book_b : ℕ := assigned_time_book_b - total_time_read_book_b

-- Calculate the total remaining time and the equal time division
def total_remaining_time : ℕ := remaining_time_book_a + remaining_time_book_b
def equal_time_division : ℕ := total_remaining_time / 2

-- Theorem statement to prove Hillary's reading time for each book on Sunday
theorem Hillary_reading_time_on_sunday : equal_time_division = 23 := by
  sorry

end NUMINAMATH_GPT_Hillary_reading_time_on_sunday_l580_58049


namespace NUMINAMATH_GPT_problem_statement_l580_58029

def system_eq1 (x y : ℝ) := x^3 - 5 * x * y^2 = 21
def system_eq2 (y x : ℝ) := y^3 - 5 * x^2 * y = 28

theorem problem_statement
(x1 y1 x2 y2 x3 y3 : ℝ)
(h1 : system_eq1 x1 y1)
(h2 : system_eq2 y1 x1)
(h3 : system_eq1 x2 y2)
(h4 : system_eq2 y2 x2)
(h5 : system_eq1 x3 y3)
(h6 : system_eq2 y3 x3)
(h_distinct : (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :
  (11 - x1 / y1) * (11 - x2 / y2) * (11 - x3 / y3) = 1729 :=
sorry

end NUMINAMATH_GPT_problem_statement_l580_58029


namespace NUMINAMATH_GPT_percentage_of_Indian_women_l580_58058

-- Definitions of conditions
def total_people := 700 + 500 + 800
def indian_men := (20 / 100) * 700
def indian_children := (10 / 100) * 800
def total_indian_people := (21 / 100) * total_people
def indian_women := total_indian_people - indian_men - indian_children

-- Statement of the theorem
theorem percentage_of_Indian_women : 
  (indian_women / 500) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_Indian_women_l580_58058


namespace NUMINAMATH_GPT_b_c_value_l580_58098

theorem b_c_value (a b c d : ℕ) 
  (h₁ : a + b = 12) 
  (h₂ : c + d = 3) 
  (h₃ : a + d = 6) : 
  b + c = 9 :=
sorry

end NUMINAMATH_GPT_b_c_value_l580_58098


namespace NUMINAMATH_GPT_find_f_2_l580_58099

variable (f : ℤ → ℤ)

-- Definitions of the conditions
def is_monic_quartic (f : ℤ → ℤ) : Prop :=
  ∃ a b c d, ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d

variable (hf_monic : is_monic_quartic f)
variable (hf_conditions : f (-2) = -4 ∧ f 1 = -1 ∧ f 3 = -9 ∧ f (-4) = -16)

-- The main statement to prove
theorem find_f_2 : f 2 = -28 := sorry

end NUMINAMATH_GPT_find_f_2_l580_58099


namespace NUMINAMATH_GPT_percentage_of_number_l580_58043

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_number_l580_58043


namespace NUMINAMATH_GPT_smallest_n_inequality_l580_58091

-- Define the main statement based on the identified conditions and answer.
theorem smallest_n_inequality (x y z w : ℝ) : 
  (x^2 + y^2 + z^2 + w^2)^2 ≤ 4 * (x^4 + y^4 + z^4 + w^4) :=
sorry

end NUMINAMATH_GPT_smallest_n_inequality_l580_58091


namespace NUMINAMATH_GPT_least_number_of_marbles_l580_58050

theorem least_number_of_marbles :
  ∃ n, (∀ d ∈ ({3, 4, 5, 7, 8} : Set ℕ), d ∣ n) ∧ n = 840 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_marbles_l580_58050


namespace NUMINAMATH_GPT_angle_y_in_triangle_l580_58028

theorem angle_y_in_triangle (y : ℝ) (h1 : ∀ a b c : ℝ, a + b + c = 180) (h2 : 3 * y + y + 40 = 180) : y = 35 :=
sorry

end NUMINAMATH_GPT_angle_y_in_triangle_l580_58028


namespace NUMINAMATH_GPT_find_prime_p_l580_58095

theorem find_prime_p
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h : Nat.Prime (p^3 + p^2 + 11 * p + 2)) :
  p = 3 :=
sorry

end NUMINAMATH_GPT_find_prime_p_l580_58095


namespace NUMINAMATH_GPT_dan_has_remaining_cards_l580_58024

-- Define the initial conditions
def initial_cards : ℕ := 97
def cards_sold_to_sam : ℕ := 15

-- Define the expected result
def remaining_cards (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- State the theorem to prove
theorem dan_has_remaining_cards : remaining_cards initial_cards cards_sold_to_sam = 82 :=
by
  -- This insertion is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_dan_has_remaining_cards_l580_58024


namespace NUMINAMATH_GPT_chinese_pig_problem_l580_58055

variable (x : ℕ)

theorem chinese_pig_problem :
  100 * x - 90 * x = 100 :=
sorry

end NUMINAMATH_GPT_chinese_pig_problem_l580_58055


namespace NUMINAMATH_GPT_Kimiko_age_proof_l580_58041

variables (Kimiko_age Kayla_age : ℕ)
variables (min_driving_age wait_years : ℕ)

def is_half_age (a b : ℕ) : Prop := a = b / 2
def minimum_driving_age (a b : ℕ) : Prop := a + b = 18

theorem Kimiko_age_proof
  (h1 : is_half_age Kayla_age Kimiko_age)
  (h2 : wait_years = 5)
  (h3 : minimum_driving_age Kayla_age wait_years) :
  Kimiko_age = 26 :=
sorry

end NUMINAMATH_GPT_Kimiko_age_proof_l580_58041


namespace NUMINAMATH_GPT_find_positive_number_l580_58078

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_positive_number_l580_58078


namespace NUMINAMATH_GPT_cost_of_fencing_l580_58056

-- Define the conditions
def width_garden : ℕ := 12
def length_playground : ℕ := 16
def width_playground : ℕ := 12
def price_per_meter : ℕ := 15
def area_playground : ℕ := length_playground * width_playground
def area_garden : ℕ := area_playground
def length_garden : ℕ := area_garden / width_garden
def perimeter_garden : ℕ := 2 * (length_garden + width_garden)
def cost_fencing : ℕ := perimeter_garden * price_per_meter

-- State the theorem
theorem cost_of_fencing : cost_fencing = 840 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l580_58056


namespace NUMINAMATH_GPT_harry_lost_sea_creatures_l580_58057

def initial_sea_stars := 34
def initial_seashells := 21
def initial_snails := 29
def initial_crabs := 17

def sea_stars_reproduced := 5
def seashells_reproduced := 3
def snails_reproduced := 4

def final_items := 105

def sea_stars_after_reproduction := initial_sea_stars + (sea_stars_reproduced * 2 - sea_stars_reproduced)
def seashells_after_reproduction := initial_seashells + (seashells_reproduced * 2 - seashells_reproduced)
def snails_after_reproduction := initial_snails + (snails_reproduced * 2 - snails_reproduced)
def crabs_after_reproduction := initial_crabs

def total_after_reproduction := sea_stars_after_reproduction + seashells_after_reproduction + snails_after_reproduction + crabs_after_reproduction

theorem harry_lost_sea_creatures : total_after_reproduction - final_items = 8 :=
by
  sorry

end NUMINAMATH_GPT_harry_lost_sea_creatures_l580_58057


namespace NUMINAMATH_GPT_absolute_value_zero_l580_58003

theorem absolute_value_zero (x : ℝ) (h : |4 * x + 6| = 0) : x = -3 / 2 :=
sorry

end NUMINAMATH_GPT_absolute_value_zero_l580_58003


namespace NUMINAMATH_GPT_slope_of_line_l580_58094

theorem slope_of_line (x y : ℝ) (h : 3 * y = 4 * x + 9) : 4 / 3 = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_slope_of_line_l580_58094


namespace NUMINAMATH_GPT_problem_sign_of_trig_product_l580_58052

open Real

theorem problem_sign_of_trig_product (θ : ℝ) (hθ : π / 2 < θ ∧ θ < π) :
  sin (cos θ) * cos (sin (2 * θ)) < 0 :=
sorry

end NUMINAMATH_GPT_problem_sign_of_trig_product_l580_58052


namespace NUMINAMATH_GPT_slope_negative_l580_58068

theorem slope_negative (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → mx1 + 5 > mx2 + 5) → m < 0 :=
by
  sorry

end NUMINAMATH_GPT_slope_negative_l580_58068


namespace NUMINAMATH_GPT_large_block_volume_l580_58015

theorem large_block_volume (W D L : ℝ) (h : W * D * L = 4) :
    (2 * W) * (2 * D) * (2 * L) = 32 :=
by
  sorry

end NUMINAMATH_GPT_large_block_volume_l580_58015


namespace NUMINAMATH_GPT_thin_film_radius_volume_l580_58064

theorem thin_film_radius_volume :
  ∀ (r : ℝ) (V : ℝ) (t : ℝ), 
    V = 216 → t = 0.1 → π * r^2 * t = V → r = Real.sqrt (2160 / π) :=
by
  sorry

end NUMINAMATH_GPT_thin_film_radius_volume_l580_58064


namespace NUMINAMATH_GPT_min_value_inverse_sum_l580_58088

variable (m n : ℝ)
variable (hm : 0 < m)
variable (hn : 0 < n)
variable (b : ℝ) (hb : b = 2)
variable (hline : 3 * m + n = 1)

theorem min_value_inverse_sum : 
  (1 / m + 4 / n) = 7 + 4 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l580_58088


namespace NUMINAMATH_GPT_sufficient_conditions_for_sum_positive_l580_58030

variable {a b : ℝ}

theorem sufficient_conditions_for_sum_positive (h₃ : a + b > 2) (h₄ : a > 0 ∧ b > 0) : a + b > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_conditions_for_sum_positive_l580_58030


namespace NUMINAMATH_GPT_incorrect_membership_l580_58033

-- Let's define the sets involved.
def a : Set ℕ := {1}             -- singleton set {a}
def ab : Set (Set ℕ) := {{1}, {2}}  -- set {a, b}

-- Now, the proof statement.
theorem incorrect_membership : ¬ (a ∈ ab) := 
by { sorry }

end NUMINAMATH_GPT_incorrect_membership_l580_58033


namespace NUMINAMATH_GPT_product_of_two_numbers_is_320_l580_58045

theorem product_of_two_numbers_is_320 (x y : ℕ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x = 5 * (y / 4)) : x * y = 320 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_two_numbers_is_320_l580_58045


namespace NUMINAMATH_GPT_molecular_weight_of_7_moles_of_CaO_l580_58081

/-- The molecular weight of 7 moles of calcium oxide (CaO) -/
def Ca_atomic_weight : Float := 40.08
def O_atomic_weight : Float := 16.00
def CaO_molecular_weight : Float := Ca_atomic_weight + O_atomic_weight

theorem molecular_weight_of_7_moles_of_CaO : 
    7 * CaO_molecular_weight = 392.56 := by 
sorry

end NUMINAMATH_GPT_molecular_weight_of_7_moles_of_CaO_l580_58081


namespace NUMINAMATH_GPT_largest_number_le_1_1_from_set_l580_58070

def is_largest_le (n : ℚ) (l : List ℚ) (bound : ℚ) : Prop :=
  (n ∈ l ∧ n ≤ bound) ∧ ∀ m ∈ l, m ≤ bound → m ≤ n

theorem largest_number_le_1_1_from_set : 
  is_largest_le (9/10) [14/10, 9/10, 12/10, 5/10, 13/10] (11/10) :=
by 
  sorry

end NUMINAMATH_GPT_largest_number_le_1_1_from_set_l580_58070


namespace NUMINAMATH_GPT_division_multiplication_identity_l580_58053

theorem division_multiplication_identity :
  24 / (-6) * (3 / 2) / (- (4 / 3)) = 9 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_division_multiplication_identity_l580_58053


namespace NUMINAMATH_GPT_problem_statement_l580_58060

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem problem_statement : same_terminal_side (-510) 210 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l580_58060


namespace NUMINAMATH_GPT_sin_product_identity_l580_58082

theorem sin_product_identity :
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * (Real.sin (72 * Real.pi / 180)) = 1 / 16 := 
by 
  sorry

end NUMINAMATH_GPT_sin_product_identity_l580_58082


namespace NUMINAMATH_GPT_gcd_180_308_l580_58021

theorem gcd_180_308 : Nat.gcd 180 308 = 4 :=
by
  sorry

end NUMINAMATH_GPT_gcd_180_308_l580_58021


namespace NUMINAMATH_GPT_triangle_proof_l580_58067

noncomputable def triangle_math_proof (A B C : ℝ) (AA1 BB1 CC1 : ℝ) : Prop :=
  AA1 = 2 * Real.sin (B + A / 2) ∧
  BB1 = 2 * Real.sin (C + B / 2) ∧
  CC1 = 2 * Real.sin (A + C / 2) ∧
  (Real.sin A + Real.sin B + Real.sin C) ≠ 0 ∧
  ∀ x, x = (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) / (Real.sin A + Real.sin B + Real.sin C) → x = 2

theorem triangle_proof (A B C AA1 BB1 CC1 : ℝ) (h : triangle_math_proof A B C AA1 BB1 CC1) :
  (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) /
  (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end NUMINAMATH_GPT_triangle_proof_l580_58067


namespace NUMINAMATH_GPT_fractional_inequality_solution_l580_58004

theorem fractional_inequality_solution :
  ∃ (m n : ℕ), n = m^2 - 1 ∧ 
               (m + 2) / (n + 2 : ℝ) > 1 / 3 ∧ 
               (m - 3) / (n - 3 : ℝ) < 1 / 10 ∧ 
               1 ≤ m ∧ m ≤ 9 ∧ 1 ≤ n ∧ n ≤ 9 ∧ 
               (m = 3) ∧ (n = 8) := 
by
  sorry

end NUMINAMATH_GPT_fractional_inequality_solution_l580_58004


namespace NUMINAMATH_GPT_leak_empty_time_l580_58040

theorem leak_empty_time
  (pump_fill_time : ℝ)
  (leak_fill_time : ℝ)
  (pump_fill_rate : pump_fill_time = 5)
  (leak_fill_rate : leak_fill_time = 10)
  : (1 / 5 - 1 / leak_fill_time)⁻¹ = 10 :=
by
  -- you can fill in the proof here
  sorry

end NUMINAMATH_GPT_leak_empty_time_l580_58040


namespace NUMINAMATH_GPT_monotonicity_of_f_extremum_of_f_on_interval_l580_58077

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem monotonicity_of_f : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → 1 ≤ x₂ → f x₁ < f x₂ := by
  sorry

theorem extremum_of_f_on_interval : 
  f 1 = 3 / 2 ∧ f 4 = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_monotonicity_of_f_extremum_of_f_on_interval_l580_58077


namespace NUMINAMATH_GPT_math_problem_l580_58011

theorem math_problem {x y : ℕ} (h1 : 1059 % x = y) (h2 : 1417 % x = y) (h3 : 2312 % x = y) : x - y = 15 := by
  sorry

end NUMINAMATH_GPT_math_problem_l580_58011


namespace NUMINAMATH_GPT_value_of_sum_l580_58035

theorem value_of_sum (a b c d : ℤ) 
  (h1 : a - b + c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 12 := 
  sorry

end NUMINAMATH_GPT_value_of_sum_l580_58035


namespace NUMINAMATH_GPT_intersection_point_interval_l580_58020

theorem intersection_point_interval (x₀ : ℝ) (h : x₀^3 = 2^x₀ + 1) : 
  1 < x₀ ∧ x₀ < 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_interval_l580_58020


namespace NUMINAMATH_GPT_number_of_roses_sold_l580_58071

def initial_roses : ℕ := 50
def picked_roses : ℕ := 21
def final_roses : ℕ := 56

theorem number_of_roses_sold : ∃ x : ℕ, initial_roses - x + picked_roses = final_roses ∧ x = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_roses_sold_l580_58071


namespace NUMINAMATH_GPT_Jessica_has_3_dozens_l580_58042

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end NUMINAMATH_GPT_Jessica_has_3_dozens_l580_58042


namespace NUMINAMATH_GPT_find_x_l580_58039

noncomputable def geometric_series_sum (x: ℝ) : ℝ := 
  1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + ∑' n: ℕ, (n + 1) * x^(n + 1)

theorem find_x (x: ℝ) (hx : geometric_series_sum x = 16) : x = 15 / 16 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l580_58039


namespace NUMINAMATH_GPT_car_C_has_highest_average_speed_l580_58075

-- Define the distances traveled by each car
def distance_car_A_1st_hour := 140
def distance_car_A_2nd_hour := 130
def distance_car_A_3rd_hour := 120

def distance_car_B_1st_hour := 170
def distance_car_B_2nd_hour := 90
def distance_car_B_3rd_hour := 130

def distance_car_C_1st_hour := 120
def distance_car_C_2nd_hour := 140
def distance_car_C_3rd_hour := 150

-- Define the total distance and average speed calculations
def total_distance_car_A := distance_car_A_1st_hour + distance_car_A_2nd_hour + distance_car_A_3rd_hour
def total_distance_car_B := distance_car_B_1st_hour + distance_car_B_2nd_hour + distance_car_B_3rd_hour
def total_distance_car_C := distance_car_C_1st_hour + distance_car_C_2nd_hour + distance_car_C_3rd_hour

def total_time := 3

def average_speed_car_A := total_distance_car_A / total_time
def average_speed_car_B := total_distance_car_B / total_time
def average_speed_car_C := total_distance_car_C / total_time

-- Lean proof statement
theorem car_C_has_highest_average_speed :
  average_speed_car_C > average_speed_car_A ∧ average_speed_car_C > average_speed_car_B :=
by
  sorry

end NUMINAMATH_GPT_car_C_has_highest_average_speed_l580_58075


namespace NUMINAMATH_GPT_scientific_notation_of_tourists_l580_58051

theorem scientific_notation_of_tourists : 
  (23766400 : ℝ) = 2.37664 * 10^7 :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_tourists_l580_58051


namespace NUMINAMATH_GPT_sum_mod_6_l580_58086

theorem sum_mod_6 :
  (60123 + 60124 + 60125 + 60126 + 60127 + 60128 + 60129 + 60130) % 6 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_6_l580_58086
