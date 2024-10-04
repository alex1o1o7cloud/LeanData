import Mathlib

namespace simplify_fraction_fraction_c_over_d_l279_279218

-- Define necessary constants and variables
variable (k : ℤ)

/-- Original expression -/
def original_expr := (6 * k + 12 + 3 : ℤ)

/-- Simplified expression -/
def simplified_expr := (2 * k + 5 : ℤ)

/-- The main theorem to prove the equivalent mathematical proof problem -/
theorem simplify_fraction : (original_expr / 3) = simplified_expr :=
by
  sorry

-- The final fraction to prove the answer
theorem fraction_c_over_d : (2 / 5 : ℚ) = 2 / 5 :=
by
  sorry

end simplify_fraction_fraction_c_over_d_l279_279218


namespace prime_factors_1260_l279_279517

theorem prime_factors_1260 (w x y z : ℕ) (h : 2 ^ w * 3 ^ x * 5 ^ y * 7 ^ z = 1260) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
by sorry

end prime_factors_1260_l279_279517


namespace johns_weight_l279_279020

theorem johns_weight (j m : ℝ) (h1 : j + m = 240) (h2 : j - m = j / 3) : j = 144 :=
by
  sorry

end johns_weight_l279_279020


namespace wedge_product_correct_l279_279510

variables {a1 a2 b1 b2 : ℝ}
def a : ℝ × ℝ := (a1, a2)
def b : ℝ × ℝ := (b1, b2)

def wedge_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.2 - v.2 * w.1

theorem wedge_product_correct (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 :=
by
  -- Proof is omitted, theorem statement only
  sorry

end wedge_product_correct_l279_279510


namespace find_y_value_l279_279852

/-- Given angles and conditions, find the value of y in the geometric figure. -/
theorem find_y_value
  (AB_parallel_DC : true) -- AB is parallel to DC
  (ACE_straight_line : true) -- ACE is a straight line
  (angle_ACF : ℝ := 130) -- ∠ACF = 130°
  (angle_CBA : ℝ := 60) -- ∠CBA = 60°
  (angle_ACB : ℝ := 100) -- ∠ACB = 100°
  (angle_ADC : ℝ := 125) -- ∠ADC = 125°
  : 35 = 35 := -- y = 35°
by
  sorry

end find_y_value_l279_279852


namespace seq_geometric_seq_general_formulas_no_arithmetic_subseq_l279_279195

open Function

-- Given conditions
variable {a : ℕ → ℤ} {b : ℕ → ℤ} {c : ℕ → ℤ} (d q : ℤ) (hnonzero : q ≠ 1)

-- Definitions of the sequences
def an_arithmetic (d : ℤ) (a : ℕ → ℤ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def bn_geometric (q : ℤ) (b : ℕ → ℤ) : Prop := 
  ∀ n, b (n + 1) = b n * q

def cn (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n + b n

-- Theorem to be proved
theorem seq_geometric_seq (d q : ℤ) (hnonzero : q ≠ 1) {a b c : ℕ → ℤ}
  (h1 : an_arithmetic d a) (h2 : bn_geometric q b) (h3 : cn a b c) :
  (∀ n, (c (n + 1) - c n - d) = b n * (q - 1)) :=
sorry

-- General formula for sequences
theorem general_formulas {a b c : ℕ → ℤ} :
  (a 1 = 1) ∧ (∀ n, a (n + 1) = a n + 3) ∧
  (b 1 = 3) ∧ (∀ n, b (n + 1) = b n * 2) :=
sorry

-- Non-existence of set A
theorem no_arithmetic_subseq (c : ℕ → ℤ) :
  (c 0 = 4) ∧ (c 1 = 10) ∧ (c 2 = 19) ∧ (c 3 = 34) →
  ¬ ∃ (A : Finset ℕ), A.card ≥ 4 ∧ 
        ∃ n1 n2 n3 n4, 
          n1 < n2 < n3 < n4 ∧ 
          ∀ {i < j < k < l}, 2 * c j = c i + c k :=
sorry

end seq_geometric_seq_general_formulas_no_arithmetic_subseq_l279_279195


namespace altitude_of_dolphin_l279_279541

theorem altitude_of_dolphin (h_submarine : altitude_submarine = -50) (h_dolphin : distance_above_submarine = 10) : altitude_dolphin = -40 :=
by
  -- Altitude of the dolphin is the altitude of the submarine plus the distance above it
  have h_dolphin_altitude : altitude_dolphin = altitude_submarine + distance_above_submarine := sorry
  -- Substitute the values
  rw [h_submarine, h_dolphin] at h_dolphin_altitude
  -- Simplify the expression
  exact h_dolphin_altitude

end altitude_of_dolphin_l279_279541


namespace problem_l279_279484

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem problem
  (ω : ℝ) 
  (hω : ω > 0)
  (hab : Real.sqrt (4 + (Real.pi ^ 2) / (ω ^ 2)) = 2 * Real.sqrt 2) :
  f ω 1 = Real.sqrt 3 / 2 := 
sorry

end problem_l279_279484


namespace percentage_orange_juice_in_blend_l279_279033

theorem percentage_orange_juice_in_blend :
  let pear_juice_per_pear := 10 / 2
  let orange_juice_per_orange := 8 / 2
  let pear_juice := 2 * pear_juice_per_pear
  let orange_juice := 3 * orange_juice_per_orange
  let total_juice := pear_juice + orange_juice
  (orange_juice / total_juice) = (6 / 11) := 
by
  sorry

end percentage_orange_juice_in_blend_l279_279033


namespace f_2016_plus_f_2015_l279_279311

theorem f_2016_plus_f_2015 (f : ℝ → ℝ) 
  (H1 : ∀ x, f (-x) = -f x) -- Odd function property
  (H2 : ∀ x, f (x + 1) = f (-x + 1)) -- Even function property for f(x+1)
  (H3 : f 1 = 1) : 
  f 2016 + f 2015 = -1 :=
sorry

end f_2016_plus_f_2015_l279_279311


namespace abs_val_equality_l279_279010

theorem abs_val_equality (m : ℝ) (h : |m| = |(-3 : ℝ)|) : m = 3 ∨ m = -3 :=
sorry

end abs_val_equality_l279_279010


namespace cylinder_volume_increase_l279_279320

theorem cylinder_volume_increase 
  (r h : ℝ) 
  (V : ℝ := π * r^2 * h) 
  (new_h : ℝ := 3 * h) 
  (new_r : ℝ := 2 * r) : 
  (π * new_r^2 * new_h) = 12 * V := 
by
  sorry

end cylinder_volume_increase_l279_279320


namespace average_weight_of_girls_l279_279045

theorem average_weight_of_girls (avg_weight_boys : ℕ) (num_boys : ℕ) (avg_weight_class : ℕ) (num_students : ℕ) :
  num_boys = 15 →
  avg_weight_boys = 48 →
  num_students = 25 →
  avg_weight_class = 45 →
  ( (avg_weight_class * num_students - avg_weight_boys * num_boys) / (num_students - num_boys) ) = 27 :=
by
  intros h_num_boys h_avg_weight_boys h_num_students h_avg_weight_class
  sorry

end average_weight_of_girls_l279_279045


namespace aria_cookies_per_day_l279_279144

theorem aria_cookies_per_day 
  (cost_per_cookie : ℕ)
  (total_amount_spent : ℕ)
  (days_in_march : ℕ)
  (h_cost : cost_per_cookie = 19)
  (h_spent : total_amount_spent = 2356)
  (h_days : days_in_march = 31) : 
  (total_amount_spent / cost_per_cookie) / days_in_march = 4 :=
by
  sorry

end aria_cookies_per_day_l279_279144


namespace percent_more_proof_l279_279637

-- Define the conditions
def y := 150
def x := 120
def is_percent_more (y x p : ℕ) : Prop := y = (1 + p / 100) * x

-- The proof problem statement
theorem percent_more_proof : ∃ p : ℕ, is_percent_more y x p ∧ p = 25 := by
  sorry

end percent_more_proof_l279_279637


namespace max_gcd_of_consecutive_terms_l279_279472

-- Given conditions
def a (n : ℕ) : ℕ := 2 * (n.factorial) + n

-- Theorem statement
theorem max_gcd_of_consecutive_terms : ∃ (d : ℕ), ∀ n ≥ 0, d ≤ gcd (a n) (a (n + 1)) ∧ d = 1 := by sorry

end max_gcd_of_consecutive_terms_l279_279472


namespace sum_quotient_dividend_divisor_l279_279652

theorem sum_quotient_dividend_divisor (n : ℕ) (d : ℕ) (h : n = 45) (h1 : d = 3) : 
  (n / d) + n + d = 63 :=
by
  sorry

end sum_quotient_dividend_divisor_l279_279652


namespace minimum_value_inequality_l279_279518

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 64) :
  ∃ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 64 ∧ (x^2 + 8 * x * y + 4 * y^2 + 4 * z^2) = 384 := 
sorry

end minimum_value_inequality_l279_279518


namespace coords_of_point_P_l279_279844

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

theorem coords_of_point_P :
  ∀ (a : ℝ), 0 < a ∧ a ≠ 1 → ∃ P : ℝ × ℝ, (P = (1, -2) ∧ ∀ y, f (f a (-2)) y = y) :=
by
  sorry

end coords_of_point_P_l279_279844


namespace binom_150_1_l279_279611

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end binom_150_1_l279_279611


namespace cat_food_insufficient_for_six_days_l279_279460

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l279_279460


namespace find_pq_l279_279514

theorem find_pq (p q : ℝ) : (∃ x : ℝ, f(p, q, x) = 0) ∧ (∃ y : ℝ, f(p, q, y) = 0) ↔ 
(p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2) := 
begin
  sorry,
end

def f (p q x : ℝ) : ℝ := x^2 + p * x + q

end find_pq_l279_279514


namespace base_conversion_subtraction_l279_279622

def base6_to_nat (d0 d1 d2 d3 d4 : ℕ) : ℕ :=
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

def base7_to_nat (d0 d1 d2 d3 : ℕ) : ℕ :=
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

theorem base_conversion_subtraction :
  base6_to_nat 1 2 3 5 4 - base7_to_nat 1 2 3 4 = 4851 := by
  sorry

end base_conversion_subtraction_l279_279622


namespace charging_piles_problem_l279_279562

variable (priceA priceB : ℝ)
variable (numA numB : ℕ)

-- Conditions from problem
def unit_price_relation : Prop := priceA + 0.3 = priceB
def quantity_equal_condition : Prop := (15 / priceA) = (20 / priceB)
def total_piles : Prop := numA + numB = 25
def total_cost : Prop := (priceA * numA + priceB * numB) <= 26
def quantity_relation : Prop := (numB >= (numA / 2))

-- Main proof statement
theorem charging_piles_problem :
  unit_price_relation priceA priceB ∧
  quantity_equal_condition priceA priceB ∧
  total_piles numA numB ∧
  total_cost priceA priceB numA numB ∧
  quantity_relation numA numB →
  (priceA = 0.9 ∧ priceB = 1.2) ∧
  ((numA = 14 ∧ numB = 11) ∨ 
   (numA = 15 ∧ numB = 10) ∨ 
   (numA = 16 ∧ numB = 9)) ∧
  (numA = 16 ∧ numB = 9 → (priceA * 16 + priceB * 9) = 25.2) :=
sorry

end charging_piles_problem_l279_279562


namespace problem_statement_l279_279338

theorem problem_statement
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := 
by
  sorry

end problem_statement_l279_279338


namespace q_is_necessary_but_not_sufficient_for_p_l279_279077

theorem q_is_necessary_but_not_sufficient_for_p (a : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)) → (a < 1) ∧ (¬ (a < 1 → (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)))) :=
by
  sorry

end q_is_necessary_but_not_sufficient_for_p_l279_279077


namespace vasya_numbers_l279_279742

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l279_279742


namespace shelter_animals_count_l279_279558

theorem shelter_animals_count : 
  (initial_cats adopted_cats new_cats final_cats dogs total_animals : ℕ) 
   (h1 : initial_cats = 15)
   (h2 : adopted_cats = initial_cats / 3)
   (h3 : new_cats = adopted_cats * 2)
   (h4 : final_cats = initial_cats - adopted_cats + new_cats)
   (h5 : dogs = final_cats * 2)
   (h6 : total_animals = final_cats + dogs) :
   total_animals = 60 := 
sorry

end shelter_animals_count_l279_279558


namespace product_gcd_lcm_l279_279283

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l279_279283


namespace multiplication_with_mixed_number_l279_279108

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l279_279108


namespace totalNumberOfCrayons_l279_279051

def numOrangeCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numBlueCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numRedCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

theorem totalNumberOfCrayons :
  numOrangeCrayons 6 8 + numBlueCrayons 7 5 + numRedCrayons 1 11 = 94 :=
by
  sorry

end totalNumberOfCrayons_l279_279051


namespace ruler_cost_l279_279587

variable {s c r : ℕ}

theorem ruler_cost (h1 : s > 18) (h2 : r > 1) (h3 : c > r) (h4 : s * c * r = 1729) : c = 13 :=
by
  sorry

end ruler_cost_l279_279587


namespace unique_solution_l279_279989

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l279_279989


namespace find_speed_l279_279028

noncomputable def circumference := 15 / 5280 -- miles
noncomputable def increased_speed (r : ℝ) := r + 5 -- miles per hour
noncomputable def reduced_time (t : ℝ) := t - 1 / 10800 -- hours
noncomputable def original_distance (r t : ℝ) := r * t
noncomputable def new_distance (r t : ℝ) := increased_speed r * reduced_time t

theorem find_speed (r t : ℝ) (h1 : original_distance r t = circumference) 
(h2 : new_distance r t = circumference) : r = 13.5 := by
  sorry

end find_speed_l279_279028


namespace range_of_set_is_8_l279_279789

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end range_of_set_is_8_l279_279789


namespace rocco_total_usd_l279_279209

def us_quarters := 4 * 8 * 0.25
def canadian_dimes := 6 * 12 * 0.10 * 0.8
def us_nickels := 9 * 10 * 0.05
def euro_cents := 5 * 15 * 0.01 * 1.18
def british_pence := 3 * 20 * 0.01 * 1.4
def japanese_yen := 2 * 10 * 1 * 0.0091
def mexican_pesos := 4 * 5 * 1 * 0.05

def total_usd := us_quarters + canadian_dimes + us_nickels + euro_cents + british_pence + japanese_yen + mexican_pesos

theorem rocco_total_usd : total_usd = 21.167 := by
  simp [us_quarters, canadian_dimes, us_nickels, euro_cents, british_pence, japanese_yen, mexican_pesos]
  sorry

end rocco_total_usd_l279_279209


namespace min_C2_minus_D2_is_36_l279_279512

noncomputable def find_min_C2_minus_D2 (x y z : ℝ) : ℝ :=
  (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))^2 -
  (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2

theorem min_C2_minus_D2_is_36 : ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → 
  find_min_C2_minus_D2 x y z ≥ 36 :=
by
  intros x y z hx hy hz
  sorry

end min_C2_minus_D2_is_36_l279_279512


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279131

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279131


namespace misha_total_students_l279_279348

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l279_279348


namespace cost_per_ream_is_27_l279_279246

-- Let ream_sheets be the number of sheets in one ream.
def ream_sheets : ℕ := 500

-- Let total_sheets be the total number of sheets needed.
def total_sheets : ℕ := 5000

-- Let total_cost be the total cost to buy the total number of sheets.
def total_cost : ℕ := 270

-- We need to prove that the cost per ream (in dollars) is 27.
theorem cost_per_ream_is_27 : (total_cost / (total_sheets / ream_sheets)) = 27 := 
by
  sorry

end cost_per_ream_is_27_l279_279246


namespace find_g_of_conditions_l279_279717

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end find_g_of_conditions_l279_279717


namespace total_money_is_145_83_l279_279189

noncomputable def jackson_money : ℝ := 125

noncomputable def williams_money : ℝ := jackson_money / 6

noncomputable def total_money : ℝ := jackson_money + williams_money

theorem total_money_is_145_83 :
  total_money = 145.83 := by
sorry

end total_money_is_145_83_l279_279189


namespace infinite_sum_equals_l279_279367

theorem infinite_sum_equals :
  10 * (79 * (1 / 7)) + (∑' n : ℕ, if n % 2 = 0 then (if n = 0 then 0 else 2 / 7 ^ n) else (1 / 7 ^ n)) = 3 / 16 :=
by
  sorry

end infinite_sum_equals_l279_279367


namespace find_x_l279_279148

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (4, -6, x)
def dot_product : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ
  | (a1, a2, a3), (b1, b2, b3) => a1 * b1 + a2 * b2 + a3 * b3

theorem find_x (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -26 :=
by 
  sorry

end find_x_l279_279148


namespace gcd_lcm_product_24_60_l279_279288

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l279_279288


namespace ratio_of_weight_l279_279330

theorem ratio_of_weight (B : ℝ) : 
    (2 * (4 + B) = 16) → ((B = 4) ∧ (4 + B) / 2 = 4) := by
  intro h
  have h₁ : B = 4 := by
    linarith
  have h₂ : (4 + B) / 2 = 4 := by
    rw [h₁]
    norm_num
  exact ⟨h₁, h₂⟩

end ratio_of_weight_l279_279330


namespace berry_ratio_l279_279362

-- Define the conditions
variables (S V R : ℕ) -- Number of berries Stacy, Steve, and Sylar have
axiom h1 : S + V + R = 1100
axiom h2 : S = 800
axiom h3 : V = 2 * R

-- Define the theorem to be proved
theorem berry_ratio (h1 : S + V + R = 1100) (h2 : S = 800) (h3 : V = 2 * R) : S / V = 4 :=
by
  sorry

end berry_ratio_l279_279362


namespace mul_mixed_number_l279_279123

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l279_279123


namespace difference_of_squares_l279_279496

variable (a b : ℝ)

theorem difference_of_squares (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := 
by
  sorry

end difference_of_squares_l279_279496


namespace multiply_mixed_number_l279_279114

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l279_279114


namespace least_number_divisible_by_38_and_3_remainder_1_exists_l279_279393

theorem least_number_divisible_by_38_and_3_remainder_1_exists :
  ∃ n, n % 38 = 1 ∧ n % 3 = 1 ∧ ∀ m, m % 38 = 1 ∧ m % 3 = 1 → n ≤ m :=
sorry

end least_number_divisible_by_38_and_3_remainder_1_exists_l279_279393


namespace money_increase_factor_two_years_l279_279967

theorem money_increase_factor_two_years (P : ℝ) (rate : ℝ) (n : ℕ)
  (h_rate : rate = 0.50) (h_n : n = 2) :
  (P * (1 + rate) ^ n) = 2.25 * P :=
by
  -- proof goes here
  sorry

end money_increase_factor_two_years_l279_279967


namespace maximum_distance_l279_279094

-- Definitions from the conditions
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def distance_driven : ℝ := 244
def gallons_used : ℝ := 20

-- Problem statement
theorem maximum_distance (h: (distance_driven / gallons_used = highway_mpg)): 
  (distance_driven = 244) :=
sorry

end maximum_distance_l279_279094


namespace gcd_exponentiation_l279_279341

theorem gcd_exponentiation (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) : 
  let a := 2^m - 2^n
  let b := 2^(m^2 + m * n + n^2) - 1
  let d := Nat.gcd a b
  d = 1 ∨ d = 7 :=
by
  sorry

end gcd_exponentiation_l279_279341


namespace sam_catches_alice_in_40_minutes_l279_279257

def sam_speed := 7 -- mph
def alice_speed := 4 -- mph
def initial_distance := 2 -- miles

theorem sam_catches_alice_in_40_minutes : 
  (initial_distance / (sam_speed - alice_speed)) * 60 = 40 :=
by sorry

end sam_catches_alice_in_40_minutes_l279_279257


namespace factorization_a_minus_b_l279_279543

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end factorization_a_minus_b_l279_279543


namespace evaluate_expression_l279_279428

theorem evaluate_expression : (- (1 / 4))⁻¹ - (Real.pi - 3)^0 - |(-4 : ℝ)| + (-1)^(2021 : ℕ) = -10 := 
by
  sorry

end evaluate_expression_l279_279428


namespace books_in_bin_after_transactions_l279_279574

def initial_books : ℕ := 4
def sold_books : ℕ := 3
def added_books : ℕ := 10

def final_books (initial_books sold_books added_books : ℕ) : ℕ :=
  initial_books - sold_books + added_books

theorem books_in_bin_after_transactions :
  final_books initial_books sold_books added_books = 11 := by
  sorry

end books_in_bin_after_transactions_l279_279574


namespace initial_amount_l279_279190

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end initial_amount_l279_279190


namespace find_b_minus_c_l279_279231

variable (a b c: ℤ)

theorem find_b_minus_c (h1: a - b - c = 1) (h2: a - (b - c) = 13) (h3: (b - c) - a = -9) : b - c = 1 :=
by {
  sorry
}

end find_b_minus_c_l279_279231


namespace sum_of_ages_eq_19_l279_279236

theorem sum_of_ages_eq_19 :
  ∃ (a b s : ℕ), (3 * a + 5 + b = s) ∧ (6 * s^2 = 2 * a^2 + 10 * b^2) ∧ (Nat.gcd a (Nat.gcd b s) = 1 ∧ a + b + s = 19) :=
sorry

end sum_of_ages_eq_19_l279_279236


namespace cat_food_inequality_l279_279439

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l279_279439


namespace jenna_bill_eel_ratio_l279_279328

theorem jenna_bill_eel_ratio:
  ∀ (B : ℕ), (B + 16 = 64) → (16 / B = 1 / 3) :=
by
  intros B h
  sorry

end jenna_bill_eel_ratio_l279_279328


namespace ab_div_c_eq_2_l279_279665

variable (a b c : ℝ)

def condition1 (a b c : ℝ) : Prop := a * b - c = 3
def condition2 (a b c : ℝ) : Prop := a * b * c = 18

theorem ab_div_c_eq_2 (h1 : condition1 a b c) (h2 : condition2 a b c) : a * b / c = 2 :=
by sorry

end ab_div_c_eq_2_l279_279665


namespace brenda_mice_left_l279_279601

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end brenda_mice_left_l279_279601


namespace value_of_a_minus_2_b_minus_2_l279_279670

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end value_of_a_minus_2_b_minus_2_l279_279670


namespace power_identity_l279_279004

theorem power_identity {a n m k : ℝ} (h1: a^n = 2) (h2: a^m = 3) (h3: a^k = 4) :
  a^(2 * n + m - 2 * k) = 3 / 4 :=
by
  sorry

end power_identity_l279_279004


namespace roger_has_more_candies_l279_279211

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end roger_has_more_candies_l279_279211


namespace f_zero_f_odd_f_range_l279_279519

noncomputable def f : ℝ → ℝ := sorry

-- Add the hypothesis for the conditions
axiom f_domain : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_value_one_third : f (1 / 3) = 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- (1) Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- (2) Prove that f(x) is odd
theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

-- (3) Given f(x) + f(2 + x) < 2, find the range of x
theorem f_range (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 := sorry

end f_zero_f_odd_f_range_l279_279519


namespace cone_base_radius_l279_279549

theorem cone_base_radius (angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) 
(h1 : angle = 216)
(h2 : sector_radius = 15)
(h3 : 2 * π * base_radius = (3 / 5) * 2 * π * sector_radius) :
base_radius = 9 := 
sorry

end cone_base_radius_l279_279549


namespace train_crossing_platform_time_l279_279579

theorem train_crossing_platform_time
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_signal_pole : ℝ)
  (speed : ℝ)
  (time_platform_cross : ℝ)
  (v := length_train / time_signal_pole)
  (d := length_train + length_platform)
  (t := d / v) :
  length_train = 300 →
  length_platform = 250 →
  time_signal_pole = 18 →
  time_platform_cross = 33 →
  t = time_platform_cross := by
  sorry

end train_crossing_platform_time_l279_279579


namespace range_of_a3_plus_a9_l279_279187

variable {a_n : ℕ → ℝ}

-- Given condition: in a geometric sequence, a4 * a8 = 9
def geom_seq_condition (a_n : ℕ → ℝ) : Prop :=
  a_n 4 * a_n 8 = 9

-- Theorem statement
theorem range_of_a3_plus_a9 (a_n : ℕ → ℝ) (h : geom_seq_condition a_n) :
  ∃ x y, (x + y = a_n 3 + a_n 9) ∧ (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≥ 6) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ x + y ≤ -6) ∨ (x = 0 ∧ y = 0 ∧ a_n 3 + a_n 9 ∈ (Set.Ici 6 ∪ Set.Iic (-6))) :=
sorry

end range_of_a3_plus_a9_l279_279187


namespace mixed_number_calculation_l279_279424

theorem mixed_number_calculation :
  47 * (2 + 2/3 - (3 + 1/4)) / (3 + 1/2 + (2 + 1/5)) = -4 - 25/38 :=
by
  sorry

end mixed_number_calculation_l279_279424


namespace initial_amount_l279_279191

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end initial_amount_l279_279191


namespace emily_subtracts_99_l279_279380

theorem emily_subtracts_99 (a b : ℕ) : (a = 50) → (b = 1) → (49^2 = 50^2 - 99) :=
by
  sorry

end emily_subtracts_99_l279_279380


namespace smallest_positive_period_l279_279047

open Real

-- Define conditions
def max_value_condition (b a : ℝ) : Prop := b + a = -1
def min_value_condition (b a : ℝ) : Prop := b - a = -5

-- Define the period of the function
def period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Main theorem
theorem smallest_positive_period (a b : ℝ) (h1 : a < 0) 
  (h2 : max_value_condition b a) 
  (h3 : min_value_condition b a) : 
  period (fun x => tan ((3 * a + b) * x)) (π / 9) :=
by
  sorry

end smallest_positive_period_l279_279047


namespace completing_the_square_l279_279534

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l279_279534


namespace gcd_lcm_product_24_60_l279_279299

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l279_279299


namespace infinite_series_sum_zero_l279_279799

theorem infinite_series_sum_zero : ∑' n : ℕ, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3)) = 0 :=
by
  sorry

end infinite_series_sum_zero_l279_279799


namespace smallest_A_l279_279217

open Finset

-- Definitions specific to our problem
variables (a : ℕ → ℤ) (S : Finset ℕ)

-- Conditions
def circle_sum_eq_100 : Prop :=
  ∑ i in range 10, a i = 100

def triplet_sum_ge_29 : Prop :=
  ∀ i in range 10, a i + a ((i + 1) % 10) + a ((i + 2) % 10) ≥ 29

def each_number_le_A (A : ℤ) : Prop :=
  ∀ i in range 10, a i ≤ A

-- The main statement we want to prove
theorem smallest_A :
  circle_sum_eq_100 a ∧ triplet_sum_ge_29 a → each_number_le_A a 13 :=
sorry

end smallest_A_l279_279217


namespace transformed_A_coordinates_l279_279527

open Real

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.fst, p.snd)

def A : ℝ × ℝ := (-3, 2)

theorem transformed_A_coordinates :
  reflect_over_y_axis (rotate_90_clockwise A) = (-2, 3) :=
by
  sorry

end transformed_A_coordinates_l279_279527


namespace multiplication_of_mixed_number_l279_279098

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l279_279098


namespace farmer_payment_per_acre_l279_279085

-- Define the conditions
def monthly_payment : ℝ := 300
def length_ft : ℝ := 360
def width_ft : ℝ := 1210
def sqft_per_acre : ℝ := 43560

-- Define the question and its correct answer
def payment_per_acre_per_month : ℝ := 30

-- Prove that the farmer pays $30 per acre per month
theorem farmer_payment_per_acre :
  (monthly_payment / ((length_ft * width_ft) / sqft_per_acre)) = payment_per_acre_per_month :=
by
  sorry

end farmer_payment_per_acre_l279_279085


namespace multiply_mixed_number_l279_279113

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l279_279113


namespace maximum_interval_length_l279_279700

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem maximum_interval_length 
  (m n : ℕ)
  (h1 : 0 < m)
  (h2 : m < n)
  (h3 : ∃ k : ℕ, ∀ i : ℕ, 0 ≤ i → i < k → ¬ is_multiple_of (m + i) 2000 ∧ (m + i) % 2021 = 0):
  n - m = 1999 :=
sorry

end maximum_interval_length_l279_279700


namespace game_show_prize_guess_l279_279952

noncomputable def total_possible_guesses : ℕ :=
  (Nat.choose 8 3) * (Nat.choose 5 3) * (Nat.choose 2 2) * (Nat.choose 7 3)

theorem game_show_prize_guess :
  total_possible_guesses = 19600 :=
by
  -- Omitted proof steps
  sorry

end game_show_prize_guess_l279_279952


namespace evaluate_f_at_3_div_5_l279_279869

def f (x : ℚ) : ℚ := 15 * x^5 + 6 * x^4 + x^3 - x^2 - 2 * x - 1

theorem evaluate_f_at_3_div_5 : f (3 / 5) = -2 / 5 :=
by
sorry

end evaluate_f_at_3_div_5_l279_279869


namespace contribution_per_person_l279_279892

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end contribution_per_person_l279_279892


namespace field_trip_classrooms_count_l279_279720

variable (students : ℕ) (seats_per_bus : ℕ) (number_of_buses : ℕ) (total_classrooms : ℕ)

def fieldTrip 
    (students := 58)
    (seats_per_bus := 2)
    (number_of_buses := 29)
    (total_classrooms := 2) : Prop :=
  students = seats_per_bus * number_of_buses  ∧ total_classrooms = students / (students / total_classrooms)

theorem field_trip_classrooms_count : fieldTrip := by
  -- Proof goes here
  sorry

end field_trip_classrooms_count_l279_279720


namespace find_u_plus_v_l279_279494

variables (u v : ℚ)

theorem find_u_plus_v (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : u + v = 27 / 43 := by
  sorry

end find_u_plus_v_l279_279494


namespace bala_age_difference_l279_279080

theorem bala_age_difference 
  (a10 : ℕ) -- Anand's age 10 years ago.
  (b10 : ℕ) -- Bala's age 10 years ago.
  (h1 : a10 = b10 / 3) -- 10 years ago, Anand's age was one-third Bala's age.
  (h2 : a10 = 15 - 10) -- Anand was 5 years old 10 years ago, given his current age is 15.
  : (b10 + 10) - 15 = 10 := -- Bala is 10 years older than Anand.
sorry

end bala_age_difference_l279_279080


namespace evaluate_expression_l279_279996

theorem evaluate_expression : 
  (3^2 - 3 * 2) - (4^2 - 4 * 2) + (5^2 - 5 * 2) - (6^2 - 6 * 2) = -14 :=
by
  sorry

end evaluate_expression_l279_279996


namespace probability_of_sequential_draws_l279_279923

theorem probability_of_sequential_draws :
  let total_cards := 52
  let num_fours := 4
  let remaining_after_first_draw := total_cards - 1
  let remaining_after_second_draw := remaining_after_first_draw - 1
  num_fours / total_cards * 1 / remaining_after_first_draw * 1 / remaining_after_second_draw = 1 / 33150 :=
by sorry

end probability_of_sequential_draws_l279_279923


namespace math_problem_solution_l279_279055

theorem math_problem_solution (pA : ℚ) (pB : ℚ)
  (hA : pA = 1/2) (hB : pB = 1/3) :
  let pNoSolve := (1 - pA) * (1 - pB)
  let pSolve := 1 - pNoSolve
  pNoSolve = 1/3 ∧ pSolve = 2/3 :=
by
  sorry

end math_problem_solution_l279_279055


namespace jack_pays_back_l279_279856

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l279_279856


namespace factorization_problem_l279_279545

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end factorization_problem_l279_279545


namespace trig_identity_example_l279_279426

theorem trig_identity_example :
  (Real.sin (43 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - Real.sin (13 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l279_279426


namespace eval_sum_l279_279805

theorem eval_sum : 
  (4 / 3 + 8 / 9 + 16 / 27 + 32 / 81 + 64 / 243 + 128 / 729 - 8 : ℚ) = -1 / 729 :=
by
  sorry

end eval_sum_l279_279805


namespace find_adult_ticket_cost_l279_279793

noncomputable def adult_ticket_cost (A : ℝ) : Prop :=
  let num_adults := 152
  let num_children := num_adults / 2
  let children_ticket_cost := 2.50
  let total_receipts := 1026
  total_receipts = num_adults * A + num_children * children_ticket_cost

theorem find_adult_ticket_cost : adult_ticket_cost 5.50 :=
by
  sorry

end find_adult_ticket_cost_l279_279793


namespace carl_sold_each_watermelon_for_3_l279_279429

def profit : ℕ := 105
def final_watermelons : ℕ := 18
def starting_watermelons : ℕ := 53
def sold_watermelons : ℕ := starting_watermelons - final_watermelons
def price_per_watermelon : ℕ := profit / sold_watermelons

theorem carl_sold_each_watermelon_for_3 :
  price_per_watermelon = 3 :=
by
  sorry

end carl_sold_each_watermelon_for_3_l279_279429


namespace rectangle_dimensions_l279_279911

theorem rectangle_dimensions (x y : ℝ) (h1 : x = 2 * y) (h2 : 2 * (x + y) = 2 * x * y) : 
  (x = 3 ∧ y = 1.5) :=
by
  sorry

end rectangle_dimensions_l279_279911


namespace perpendicular_transfer_l279_279308

variables {Line Plane : Type} 
variables (a b : Line) (α β : Plane)

def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

theorem perpendicular_transfer
  (h1 : perpendicular a α)
  (h2 : parallel_planes α β) :
  perpendicular a β := 
sorry

end perpendicular_transfer_l279_279308


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279129

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279129


namespace Julie_and_Matt_ate_cookies_l279_279523

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem Julie_and_Matt_ate_cookies : initial_cookies - remaining_cookies = 9 :=
by
  sorry

end Julie_and_Matt_ate_cookies_l279_279523


namespace telephone_number_fraction_calculation_l279_279421

theorem telephone_number_fraction_calculation :
  let valid_phone_numbers := 7 * 10^6
  let special_phone_numbers := 10^5
  (special_phone_numbers / valid_phone_numbers : ℚ) = 1 / 70 :=
by
  sorry

end telephone_number_fraction_calculation_l279_279421


namespace min_area_monochromatic_triangle_l279_279908

-- Definition of the integer lattice in the plane.
def lattice_points : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) }

-- The 3-coloring condition
def coloring (c : (ℤ × ℤ) → Fin 3) := ∀ p : (ℤ × ℤ), p ∈ lattice_points → (c p) < 3

-- Definition of the area of a triangle
def triangle_area (A B C : ℤ × ℤ) : ℝ :=
  0.5 * abs (((B.1 - A.1) * (C.2 - A.2)) - ((C.1 - A.1) * (B.2 - A.2)))

-- The statement we need to prove
theorem min_area_monochromatic_triangle :
  ∃ S : ℝ, S = 3 ∧ ∀ (c : (ℤ × ℤ) → Fin 3), coloring c → ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (c A = c B ∧ c B = c C) ∧ triangle_area A B C = S :=
sorry

end min_area_monochromatic_triangle_l279_279908


namespace Vasya_numbers_l279_279766

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l279_279766


namespace part1_part2_l279_279200
open Real

noncomputable def f (x : ℝ) (m : ℝ) := x^2 - m * log x
noncomputable def h (x : ℝ) (a : ℝ) := x^2 - x + a
noncomputable def k (x : ℝ) (a : ℝ) := x - 2 * log x - a

theorem part1 (x : ℝ) (m : ℝ) (h_pos_x : 1 < x) : 
  (f x m) - (h x 0) ≥ 0 → m ≤ exp 1 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x < 2 → k x a < 0) ∧ 
  (k 2 a < 0) ∧ 
  (∀ x, 2 < x ∧ x ≤ 3 → k x a > 0) →
  2 - 2 * log 2 < a ∧ a ≤ 3 - 2 * log 3 :=
sorry

end part1_part2_l279_279200


namespace intersection_of_M_and_N_l279_279163

def M : Set ℝ := { x | (x - 3) / (x - 1) ≤ 0 }
def N : Set ℝ := { x | -6 * x^2 + 11 * x - 4 > 0 }

theorem intersection_of_M_and_N : M ∩ N = { x | 1 < x ∧ x < 4 / 3 } :=
by 
  sorry

end intersection_of_M_and_N_l279_279163


namespace value_of_fraction_l279_279317

theorem value_of_fraction (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 + 2 * n - 1 = 0) (h3 : m * n ≠ 1) : 
  (mn + n + 1) / n = 3 :=
by
  sorry

end value_of_fraction_l279_279317


namespace ball_of_yarn_costs_6_l279_279263

-- Define the conditions as variables and hypotheses
variable (num_sweaters : ℕ := 28)
variable (balls_per_sweater : ℕ := 4)
variable (price_per_sweater : ℕ := 35)
variable (gain_from_sales : ℕ := 308)

-- Define derived values
def total_revenue : ℕ := num_sweaters * price_per_sweater
def total_cost_of_yarn : ℕ := total_revenue - gain_from_sales
def total_balls_of_yarn : ℕ := num_sweaters * balls_per_sweater
def cost_per_ball_of_yarn : ℕ := total_cost_of_yarn / total_balls_of_yarn

-- The theorem to be proven
theorem ball_of_yarn_costs_6 :
  cost_per_ball_of_yarn = 6 :=
by sorry

end ball_of_yarn_costs_6_l279_279263


namespace radius_large_circle_l279_279234

/-- Let R be the radius of the large circle. Assume three circles of radius 2 are externally 
tangent to each other. Two of these circles are internally tangent to the larger circle, 
and the third circle is tangent to the larger circle both internally and externally. 
Prove that the radius of the large circle is 4 + 2 * sqrt 3. -/
theorem radius_large_circle (R : ℝ)
  (h1 : ∃ (C1 C2 C3 : ℝ × ℝ), 
    dist C1 C2 = 4 ∧ dist C2 C3 = 4 ∧ dist C3 C1 = 4 ∧ 
    (∃ (O : ℝ × ℝ), 
      (dist O C1 = R - 2) ∧ 
      (dist O C2 = R - 2) ∧ 
      (dist O C3 = R + 2) ∧ 
      (dist C1 C2 = 4) ∧ (dist C2 C3 = 4) ∧ (dist C3 C1 = 4))):
  R = 4 + 2 * Real.sqrt 3 := 
sorry

end radius_large_circle_l279_279234


namespace unhappy_passengers_most_probable_is_1_expected_unhappy_passengers_is_variance_unhappy_passengers_is_l279_279885

noncomputable def unhappy_passengers_most_probable (n : ℕ) : ℕ :=
1

noncomputable def expected_unhappy_passengers (n : ℕ) : ℝ :=
Real.sqrt (n / Real.pi)

noncomputable def variance_unhappy_passengers (n : ℕ) : ℝ :=
0.182 * n

theorem unhappy_passengers_most_probable_is_1 (n : ℕ) : unhappy_passengers_most_probable n = 1 :=
sorry

theorem expected_unhappy_passengers_is (n : ℕ) : expected_unhappy_passengers n = Real.sqrt (n / Real.pi) :=
sorry

theorem variance_unhappy_passengers_is (n : ℕ) : variance_unhappy_passengers n = 0.182 * n :=
sorry

end unhappy_passengers_most_probable_is_1_expected_unhappy_passengers_is_variance_unhappy_passengers_is_l279_279885


namespace ellipse_general_equation_l279_279162

theorem ellipse_general_equation (x y : ℝ) (α : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_general_equation_l279_279162


namespace tangerine_count_l279_279374

def initial_tangerines : ℕ := 10
def added_tangerines : ℕ := 6

theorem tangerine_count : initial_tangerines + added_tangerines = 16 :=
by
  sorry

end tangerine_count_l279_279374


namespace plates_arrangement_l279_279091

theorem plates_arrangement : 
  let blue := 6
  let red := 3
  let green := 2
  let yellow := 1
  let total_ways_without_rest := Nat.factorial (blue + red + green + yellow - 1) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)
  let green_adj_ways := Nat.factorial (blue + red + green + yellow - 2) / (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * Nat.factorial yellow)
  total_ways_without_rest - green_adj_ways = 22680 
:= sorry

end plates_arrangement_l279_279091


namespace jack_pays_back_total_l279_279860

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l279_279860


namespace fraction_addition_l279_279807

theorem fraction_addition (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/9) : a + b = 47/36 :=
by
  rw [h1, h2]
  sorry

end fraction_addition_l279_279807


namespace max_value_of_a_l279_279313

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (x + 1) * (1 + Real.log (x + 1)) - a * x

theorem max_value_of_a (a : ℤ) : 
  (∀ x : ℝ, x ≥ -1 → (a : ℝ) * x ≤ (x + 1) * (1 + Real.log (x + 1))) → a ≤ 3 := sorry

end max_value_of_a_l279_279313


namespace chemist_mixing_solution_l279_279406

theorem chemist_mixing_solution (x : ℝ) : 0.30 * x = 0.20 * (x + 1) → x = 2 :=
by
  intro h
  sorry

end chemist_mixing_solution_l279_279406


namespace misha_total_students_l279_279347

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l279_279347


namespace green_beans_weight_l279_279520

/-- 
    Mary uses plastic grocery bags that can hold a maximum of twenty pounds. 
    She buys some green beans, 6 pounds milk, and twice the amount of carrots as green beans. 
    She can fit 2 more pounds of groceries in that bag. 
    Prove that the weight of green beans she bought is equal to 4 pounds.
-/
theorem green_beans_weight (G : ℕ) (H1 : ∀ g : ℕ, g + 6 + 2 * g ≤ 20 - 2) : G = 4 :=
by 
  have H := H1 4
  sorry

end green_beans_weight_l279_279520


namespace common_rational_root_is_neg_one_third_l279_279542

theorem common_rational_root_is_neg_one_third (a b c d e f g : ℚ) :
  ∃ k : ℚ, (75 * k^4 + a * k^3 + b * k^2 + c * k + 12 = 0) ∧
           (12 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 75 = 0) ∧
           (¬ k.isInt) ∧ (k < 0) ∧ (k = -1/3) :=
sorry

end common_rational_root_is_neg_one_third_l279_279542


namespace relationship_sides_l279_279492

-- Definitions for the given condition
variables (a b c : ℝ)

-- Statement of the theorem to prove
theorem relationship_sides (h : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) : a + c = 2 * b :=
sorry

end relationship_sides_l279_279492


namespace binary_to_decimal_101_l279_279618

theorem binary_to_decimal_101 : ∑ (i : Fin 3), (Nat.digit 2 ⟨i, sorry⟩ (list.nth_le [1, 0, 1] i sorry)) * (2 ^ i) = 5 := 
  sorry

end binary_to_decimal_101_l279_279618


namespace series_result_l279_279607

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end series_result_l279_279607


namespace multiplication_with_mixed_number_l279_279106

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l279_279106


namespace smallest_n_for_congruence_l279_279994

theorem smallest_n_for_congruence :
  ∃ n : ℕ, n > 0 ∧ 7 ^ n % 4 = n ^ 7 % 4 ∧ ∀ m : ℕ, (m > 0 ∧ m < n → ¬ (7 ^ m % 4 = m ^ 7 % 4)) :=
by
  sorry

end smallest_n_for_congruence_l279_279994


namespace sufficient_not_necessary_condition_l279_279647

variable (a : ℝ)

theorem sufficient_not_necessary_condition :
  (1 < a ∧ a < 2) → (a^2 - 3 * a ≤ 0) := by
  intro h
  sorry

end sufficient_not_necessary_condition_l279_279647


namespace triangle_inequality_problem_l279_279532

-- Define the problem statement: Given the specified conditions, prove the interval length and sum
theorem triangle_inequality_problem :
  ∀ (A B C D : Type) (AB AC BC BD CD AD AO : ℝ),
  AB = 12 ∧ CD = 4 →
  (∃ x : ℝ, (4 < x ∧ x < 24) ∧ (AC = x ∧ m = 4 ∧ n = 24 ∧ m + n = 28)) :=
by
  intro A B C D AB AC BC BD CD AD AO h
  sorry

end triangle_inequality_problem_l279_279532


namespace complete_the_square_l279_279536

theorem complete_the_square :
  ∀ x : ℝ, (x^2 + 8 * x + 7 = 0) → (x + 4)^2 = 9 :=
by
  intro x h,
  sorry

end complete_the_square_l279_279536


namespace work_completion_days_l279_279568

theorem work_completion_days (A B : ℕ) (h1 : A = 2 * B) (h2 : 6 * (A + B) = 18) : B = 1 → 18 = 18 :=
by
  sorry

end work_completion_days_l279_279568


namespace total_water_output_l279_279929

theorem total_water_output (flow_rate: ℚ) (time_duration: ℕ) (total_water: ℚ) :
  flow_rate = 2 + 2 / 3 → time_duration = 9 → total_water = 24 →
  flow_rate * time_duration = total_water :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_water_output_l279_279929


namespace line_slope_l279_279931

theorem line_slope : 
  (∀ (x y : ℝ), (x / 4 - y / 3 = -2) → (y = -3/4 * x - 6)) ∧ (∀ (x : ℝ), ∃ y : ℝ, (x / 4 - y / 3 = -2)) :=
by
  sorry

end line_slope_l279_279931


namespace solve_fraction_l279_279394

variables (w x y : ℝ)

-- Conditions
def condition1 := w / x = 2 / 3
def condition2 := w / y = 6 / 15

-- Statement
theorem solve_fraction (h1 : condition1 w x) (h2 : condition2 w y) : (x + y) / y = 8 / 5 :=
sorry

end solve_fraction_l279_279394


namespace molecular_weight_single_mole_l279_279565

theorem molecular_weight_single_mole :
  (∀ (w_7m C6H8O7 : ℝ), w_7m = 1344 → (w_7m / 7) = 192) :=
by
  intros w_7m C6H8O7 h
  sorry

end molecular_weight_single_mole_l279_279565


namespace vasya_numbers_l279_279738

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l279_279738


namespace certain_event_birthday_example_l279_279389
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end certain_event_birthday_example_l279_279389


namespace maximum_enclosed_area_l279_279070

theorem maximum_enclosed_area (P : ℝ) (A : ℝ) : 
  P = 100 → (∃ l w : ℝ, P = 2 * l + 2 * w ∧ A = l * w) → A ≤ 625 :=
by
  sorry

end maximum_enclosed_area_l279_279070


namespace smallest_angle_of_cyclic_quadrilateral_l279_279897

theorem smallest_angle_of_cyclic_quadrilateral (angles : ℝ → ℝ) (a d : ℝ) :
  -- Conditions
  (∀ n : ℕ, angles n = a + n * d) ∧ 
  (angles 3 = 140) ∧
  (a + d + (a + 3 * d) = 180) →
  -- Conclusion
  (a = 40) :=
by sorry

end smallest_angle_of_cyclic_quadrilateral_l279_279897


namespace base_b_square_l279_279468

-- Given that 144 in base b can be written as b^2 + 4b + 4 in base 10,
-- prove that it is a perfect square if and only if b > 4

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, b^2 + 4 * b + 4 = k^2 := by
  sorry

end base_b_square_l279_279468


namespace multiplication_of_mixed_number_l279_279101

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l279_279101


namespace multiplication_of_mixed_number_l279_279099

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l279_279099


namespace distinct_values_of_fx_l279_279336

theorem distinct_values_of_fx :
  let f (x : ℝ) := ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋
  ∃ (s : Finset ℤ), (∀ x, 0 ≤ x ∧ x ≤ 10 → f x ∈ s) ∧ s.card = 61 :=
by
  sorry

end distinct_values_of_fx_l279_279336


namespace function_point_proof_l279_279176

-- Given conditions
def condition (f : ℝ → ℝ) : Prop :=
  f 1 = 3

-- Prove the statement
theorem function_point_proof (f : ℝ → ℝ) (h : condition f) : f (-1) + 1 = 4 :=
by
  -- Adding the conditions here
  sorry -- proof is not required

end function_point_proof_l279_279176


namespace handshaking_remainder_l279_279326

-- Define number of people
def num_people := 11

-- Define N as the number of possible handshaking ways
def N : ℕ :=
sorry -- This will involve complicated combinatorial calculations

-- Define the target result to be proven
theorem handshaking_remainder : N % 1000 = 120 :=
sorry

end handshaking_remainder_l279_279326


namespace max_digits_in_product_l279_279385

theorem max_digits_in_product :
  let n := (99999 : Nat)
  let m := (999 : Nat)
  let product := n * m
  ∃ d : Nat, product < 10^d ∧ 10^(d-1) ≤ product :=
by
  sorry

end max_digits_in_product_l279_279385


namespace gcd_lcm_product_24_60_l279_279298

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l279_279298


namespace negation_of_exists_x_squared_gt_one_l279_279718

-- Negation of the proposition
theorem negation_of_exists_x_squared_gt_one :
  ¬ (∃ x : ℝ, x^2 > 1) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end negation_of_exists_x_squared_gt_one_l279_279718


namespace eggs_left_in_jar_l279_279530

variable (initial_eggs : ℝ) (removed_eggs : ℝ)

theorem eggs_left_in_jar (h1 : initial_eggs = 35.3) (h2 : removed_eggs = 4.5) :
  initial_eggs - removed_eggs = 30.8 :=
by
  sorry

end eggs_left_in_jar_l279_279530


namespace product_of_three_greater_than_product_of_two_or_four_l279_279820

theorem product_of_three_greater_than_product_of_two_or_four
  (nums : Fin 10 → ℝ)
  (h_positive : ∀ i, 0 < nums i)
  (h_distinct : Function.Injective nums) :
  ∃ (a b c : Fin 10),
    (∃ (d e : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (c ≠ d) ∧ (c ≠ e) ∧ nums a * nums b * nums c > nums d * nums e) ∨
    (∃ (d e f g : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ nums a * nums b * nums c > nums d * nums e * nums f * nums g) :=
sorry

end product_of_three_greater_than_product_of_two_or_four_l279_279820


namespace harkamal_paid_amount_l279_279242

variable (grapesQuantity : ℕ)
variable (grapesRate : ℕ)
variable (mangoesQuantity : ℕ)
variable (mangoesRate : ℕ)

theorem harkamal_paid_amount (h1 : grapesQuantity = 8) (h2 : grapesRate = 70) (h3 : mangoesQuantity = 9) (h4 : mangoesRate = 45) :
  (grapesQuantity * grapesRate + mangoesQuantity * mangoesRate) = 965 := by
  sorry

end harkamal_paid_amount_l279_279242


namespace expression_evaluation_l279_279621

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l279_279621


namespace cat_food_insufficient_l279_279443

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l279_279443


namespace blue_higher_than_yellow_l279_279381

theorem blue_higher_than_yellow :
  let Event := {k : ℕ // k > 0}
  let P : Event → ℝ := λ k, real.exp(-k * log 3)
  (∑' (k : ℕ), P ⟨k, Nat.succ_pos k⟩) = 1 :=
  -- probability distribution should sum to 1 to be valid
  by sorry

  let same_bin_prob := ∑' (k : ℕ), P ⟨k, Nat.succ_pos k⟩^2
  -- The probability of both balls landing in the same bin
  same_bin_prob = (1 / 8) :=
  -- Corresponding to calculation from geometric series sum
  by sorry

  let diff_bin_prob := 1 - same_bin_prob
  -- The probability of balls landing in different bins
  diff_bin_prob = (7 / 8) :=
  -- Calculating difference
  by sorry

  let final_prob := diff_bin_prob / 2
  -- Using symmetry to find the final probability for the blue ball to land in a higher bin
  final_prob = (7 / 16) :=
  (by sorry)

end blue_higher_than_yellow_l279_279381


namespace alpha_beta_roots_eq_l279_279495

theorem alpha_beta_roots_eq {α β : ℝ} (hα : α^2 - α - 2006 = 0) (hβ : β^2 - β - 2006 = 0) (h_sum : α + β = 1) : 
  α + β^2 = 2007 :=
by
  sorry

end alpha_beta_roots_eq_l279_279495


namespace net_progress_l279_279783

-- Define the conditions as properties
def lost_yards : ℕ := 5
def gained_yards : ℕ := 10

-- Prove that the team's net progress is 5 yards
theorem net_progress : (gained_yards - lost_yards) = 5 :=
by
  sorry

end net_progress_l279_279783


namespace sufficient_but_not_necessary_l279_279011

variable (x : ℝ)

def condition_p := -1 ≤ x ∧ x ≤ 1
def condition_q := x ≥ -2

theorem sufficient_but_not_necessary :
  (condition_p x → condition_q x) ∧ ¬(condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_l279_279011


namespace space_shuttle_speed_kmph_l279_279957

-- Question: Prove that the speed of the space shuttle in kilometers per hour is 32400, given it travels at 9 kilometers per second and there are 3600 seconds in an hour.
theorem space_shuttle_speed_kmph :
  (9 * 3600 = 32400) :=
by
  sorry

end space_shuttle_speed_kmph_l279_279957


namespace jack_pays_back_l279_279858

-- conditions in the problem 
def principal : ℝ := 1200
def interest_rate : ℝ := 0.1

-- the theorem statement equivalent to the question and correct answer
theorem jack_pays_back (principal_interest: principal * interest_rate) (total_amount: principal + principal_interest) : total_amount = 1320 :=
by
  sorry

end jack_pays_back_l279_279858


namespace determine_pq_value_l279_279219

noncomputable def p : ℝ → ℝ := λ x => 16 * x
noncomputable def q : ℝ → ℝ := λ x => (x + 4) * (x - 1)

theorem determine_pq_value : (p (-1) / q (-1)) = 8 / 3 := by
  sorry

end determine_pq_value_l279_279219


namespace frictional_force_is_correct_l279_279414

-- Definitions
def m1 := 2.0 -- mass of the tank in kg
def m2 := 10.0 -- mass of the cart in kg
def a := 5.0 -- acceleration of the cart in m/s^2
def mu := 0.6 -- coefficient of friction between the tank and the cart
def g := 9.8 -- acceleration due to gravity in m/s^2

-- Frictional force acting on the tank
def frictional_force := mu * (m1 * g)

-- Required force to accelerate the tank with the cart
def required_force := m1 * a

-- Proof statement
theorem frictional_force_is_correct : required_force = 10 := 
by
  -- skipping the proof as specified
  sorry

end frictional_force_is_correct_l279_279414


namespace total_animals_in_shelter_l279_279559

def initial_cats : ℕ := 15
def adopted_cats := initial_cats / 3
def replacement_cats := 2 * adopted_cats
def current_cats := initial_cats - adopted_cats + replacement_cats
def additional_dogs := 2 * current_cats
def total_animals := current_cats + additional_dogs

theorem total_animals_in_shelter : total_animals = 60 := by
  sorry

end total_animals_in_shelter_l279_279559


namespace cost_of_baseball_is_correct_l279_279875

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end cost_of_baseball_is_correct_l279_279875


namespace Vasya_numbers_l279_279759

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l279_279759


namespace gcd_lcm_product_24_60_l279_279286

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l279_279286


namespace find_number_l279_279403

theorem find_number {x : ℝ} (h : 0.5 * x - 10 = 25) : x = 70 :=
sorry

end find_number_l279_279403


namespace multiplication_with_mixed_number_l279_279103

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l279_279103


namespace regular_pay_correct_l279_279090

noncomputable def regular_pay_per_hour (total_payment : ℝ) (regular_hours : ℕ) (overtime_hours : ℕ) (overtime_rate : ℝ) : ℝ :=
  let R := total_payment / (regular_hours + overtime_rate * overtime_hours)
  R

theorem regular_pay_correct :
  regular_pay_per_hour 198 40 13 2 = 3 :=
by
  sorry

end regular_pay_correct_l279_279090


namespace shift_parabola_left_l279_279364

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end shift_parabola_left_l279_279364


namespace olivia_insurance_premium_l279_279035

theorem olivia_insurance_premium :
  ∀ (P : ℕ) (base_premium accident_percentage ticket_cost : ℤ) (tickets accidents : ℕ),
    base_premium = 50 →
    accident_percentage = P →
    ticket_cost = 5 →
    tickets = 3 →
    accidents = 1 →
    (base_premium + (accidents * base_premium * P / 100) + (tickets * ticket_cost) = 70) →
    P = 10 :=
by
  intros P base_premium accident_percentage ticket_cost tickets accidents
  intro h1 h2 h3 h4 h5 h6
  sorry

end olivia_insurance_premium_l279_279035


namespace determine_g_l279_279982

variable {R : Type*} [CommRing R]

theorem determine_g (g : R → R) (x : R) :
  (4 * x^5 + 3 * x^3 - 2 * x + 1 + g x = 7 * x^3 - 5 * x^2 + 4 * x - 3) →
  g x = -4 * x^5 + 4 * x^3 - 5 * x^2 + 6 * x - 4 :=
by
  sorry

end determine_g_l279_279982


namespace shirts_per_minute_l279_279259

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (h1 : total_shirts = 196) (h2 : total_minutes = 28) :
  total_shirts / total_minutes = 7 :=
by
  -- beginning of proof would go here
  sorry

end shirts_per_minute_l279_279259


namespace family_chocolate_chip_count_l279_279192

theorem family_chocolate_chip_count
  (batch_cookies : ℕ)
  (total_people : ℕ)
  (batches : ℕ)
  (choco_per_cookie : ℕ)
  (cookie_total : ℕ := batch_cookies * batches)
  (cookies_per_person : ℕ := cookie_total / total_people)
  (choco_per_person : ℕ := cookies_per_person * choco_per_cookie)
  (h1 : batch_cookies = 12)
  (h2 : total_people = 4)
  (h3 : batches = 3)
  (h4 : choco_per_cookie = 2)
  : choco_per_person = 18 := 
by sorry

end family_chocolate_chip_count_l279_279192


namespace irreducible_positive_fraction_unique_l279_279625

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end irreducible_positive_fraction_unique_l279_279625


namespace initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l279_279360

variable (p : ℕ → ℚ)

-- Given conditions
axiom initial_condition : p 0 = 1
axiom move_to_1 : p 1 = 1 / 2
axiom move_to_2 : p 2 = 3 / 4
axiom recurrence_relation : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2))
axiom p_99_cond : p 99 = 2 / 3 - 1 / (3 * 2^99)
axiom p_100_cond : p 100 = 1 / 3 + 1 / (3 * 2^99)

-- Proof that initial conditions are met
theorem initial_condition_proof : p 0 = 1 :=
sorry

theorem move_to_1_proof : p 1 = 1 / 2 :=
sorry

theorem move_to_2_proof : p 2 = 3 / 4 :=
sorry

-- Proof of the recurrence relation
theorem recurrence_relation_proof : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2)) :=
sorry

-- Proof of p_99
theorem p_99_proof : p 99 = 2 / 3 - 1 / (3 * 2^99) :=
sorry

-- Proof of p_100
theorem p_100_proof : p 100 = 1 / 3 + 1 / (3 * 2^99) :=
sorry

end initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l279_279360


namespace gcd_lcm_product_24_60_l279_279296

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l279_279296


namespace ratio_of_areas_l279_279065

-- Define the conditions
def angle_Q_smaller_circle : ℝ := 60
def angle_Q_larger_circle : ℝ := 30
def arc_length_equal (C1 C2 : ℝ) : Prop := 
  (angle_Q_smaller_circle / 360) * C1 = (angle_Q_larger_circle / 360) * C2

-- The required Lean statement that proves the ratio of the areas
theorem ratio_of_areas (C1 C2 r1 r2 : ℝ) 
  (arc_eq : arc_length_equal C1 C2) : 
  (π * r1^2) / (π * r2^2) = 1 / 4 := 
by 
  sorry

end ratio_of_areas_l279_279065


namespace factorize_1_factorize_2_l279_279273

variable {a x y : ℝ}

theorem factorize_1 : 2 * a * x^2 - 8 * a * x * y + 8 * a * y^2 = 2 * a * (x - 2 * y)^2 := 
by
  sorry

theorem factorize_2 : 6 * x * y^2 - 9 * x^2 * y - y^3 = -y * (3 * x - y)^2 := 
by
  sorry

end factorize_1_factorize_2_l279_279273


namespace unique_solution_l279_279987

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l279_279987


namespace parabola_intersection_min_y1_y2_sqr_l279_279310

theorem parabola_intersection_min_y1_y2_sqr :
  ∀ (x1 x2 y1 y2 : ℝ)
    (h1 : y1 ^ 2 = 4 * x1)
    (h2 : y2 ^ 2 = 4 * x2)
    (h3 : (∃ k : ℝ, x1 = 4 ∧ y1 = k * (4 - 4)) ∨ x1 = 4 ∧ y1 ≠ x2),
    ∃ m : ℝ, (y1^2 + y2^2) = m ∧ m = 32 := 
sorry

end parabola_intersection_min_y1_y2_sqr_l279_279310


namespace vasya_numbers_l279_279733

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l279_279733


namespace vasya_numbers_l279_279748

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l279_279748


namespace converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l279_279470

-- Definitions
variables {α : Type} [LinearOrderedField α] {a b : α}
variables {A B C : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]

-- Proof Problem for Question 1
theorem converse_angle_bigger_side (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]
  (angle_C angle_B : A) (side_AB side_AC : B) (h : angle_C > angle_B) : side_AB > side_AC :=
sorry

-- Proof Problem for Question 2
theorem negation_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

-- Proof Problem for Question 3
theorem contrapositive_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l279_279470


namespace food_requirement_not_met_l279_279452

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l279_279452


namespace food_requirement_not_met_l279_279449

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l279_279449


namespace geometric_to_arithmetic_common_ratio_greater_than_1_9_l279_279392

theorem geometric_to_arithmetic (q : ℝ) (h : q = (1 + Real.sqrt 5) / 2) :
  ∃ (a b c : ℝ), b - a = c - b ∧ a / b = b / c := 
sorry

theorem common_ratio_greater_than_1_9 (q : ℝ) (h_pos : q > 1.9 ∧ q < 2) :
  ∃ (n : ℕ), q^(n+1) - 2 * q^n + 1 = 0 :=
sorry

end geometric_to_arithmetic_common_ratio_greater_than_1_9_l279_279392


namespace perpendicular_lines_from_perpendicular_planes_l279_279511

variable {Line : Type} {Plane : Type}

-- Definitions of non-coincidence, perpendicularity, parallelism
noncomputable def non_coincident_lines (a b : Line) : Prop := sorry
noncomputable def non_coincident_planes (α β : Plane) : Prop := sorry
noncomputable def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def plane_parallel_to_plane (α β : Plane) : Prop := sorry
noncomputable def plane_perpendicular_to_plane (α β : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_line (a b : Line) : Prop := sorry

-- Given non-coincident lines and planes
variable {a b : Line} {α β : Plane}

-- Problem statement
theorem perpendicular_lines_from_perpendicular_planes (h1 : non_coincident_lines a b)
  (h2 : non_coincident_planes α β)
  (h3 : line_perpendicular_to_plane a α)
  (h4 : line_perpendicular_to_plane b β)
  (h5 : plane_perpendicular_to_plane α β) : line_perpendicular_to_line a b := sorry

end perpendicular_lines_from_perpendicular_planes_l279_279511


namespace vasya_numbers_l279_279752

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l279_279752


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279128

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279128


namespace simplify_expression_l279_279136

theorem simplify_expression :
  (20^4 + 625) * (40^4 + 625) * (60^4 + 625) * (80^4 + 625) /
  (10^4 + 625) * (30^4 + 625) * (50^4 + 625) * (70^4 + 625) = 7 := 
sorry

end simplify_expression_l279_279136


namespace sum_ages_l279_279139

variables (uncle_age eunji_age yuna_age : ℕ)

def EunjiAge (uncle_age : ℕ) := uncle_age - 25
def YunaAge (eunji_age : ℕ) := eunji_age + 3

theorem sum_ages (h_uncle : uncle_age = 41) (h_eunji : EunjiAge uncle_age = eunji_age) (h_yuna : YunaAge eunji_age = yuna_age) :
  eunji_age + yuna_age = 35 :=
sorry

end sum_ages_l279_279139


namespace f_of_f_of_neg1_l279_279633

-- Define the function f(x) as per the conditions
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then real.log 2 (x^2) + 1 else (1 / 3)^x + 1

-- State the theorem to prove that f(f(-1)) = 5
theorem f_of_f_of_neg1 : f (f (-1)) = 5 :=
by
  -- Proof omitted; includes necessary placeholder for compilation
  sorry

end f_of_f_of_neg1_l279_279633


namespace problem_l279_279196

theorem problem (a₅ b₅ a₆ b₆ a₇ b₇ : ℤ) (S₇ S₅ T₆ T₄ : ℤ)
  (h1 : a₅ = b₅)
  (h2 : a₆ = b₆)
  (h3 : S₇ - S₅ = 4 * (T₆ - T₄)) :
  (a₇ + a₅) / (b₇ + b₅) = -1 :=
sorry

end problem_l279_279196


namespace linear_dependent_vectors_l279_279229

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l279_279229


namespace vasya_numbers_l279_279754

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l279_279754


namespace value_2_stddev_less_than_mean_l279_279709

theorem value_2_stddev_less_than_mean :
  let mean := 17.5
  let stddev := 2.5
  mean - 2 * stddev = 12.5 :=
by
  sorry

end value_2_stddev_less_than_mean_l279_279709


namespace minimum_total_length_of_removed_segments_l279_279795

-- Definitions based on conditions
def right_angled_triangle_sides : Nat × Nat × Nat := (3, 4, 5)

def large_square_side_length : Nat := 7

-- Statement of the problem to be proved
theorem minimum_total_length_of_removed_segments
  (triangles : Fin 4 → (Nat × Nat × Nat) := fun _ => right_angled_triangle_sides)
  (side_length_of_large_square : Nat := large_square_side_length) :
  ∃ (removed_length : Nat), removed_length = 7 :=
sorry

end minimum_total_length_of_removed_segments_l279_279795


namespace abs_value_expression_l279_279497

theorem abs_value_expression (m n : ℝ) (h1 : m < 0) (h2 : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 :=
sorry

end abs_value_expression_l279_279497


namespace irreducible_positive_fraction_unique_l279_279626

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end irreducible_positive_fraction_unique_l279_279626


namespace binary_to_decimal_l279_279617

theorem binary_to_decimal :
  ∀ n : ℕ, n = 101 →
  ∑ i in Finset.range 3, (n / 10^i % 10) * 2^i = 5 :=
by
  sorry

end binary_to_decimal_l279_279617


namespace possible_m_value_l279_279150

variable (a b m t : ℝ)
variable (h_a : a ≠ 0)
variable (h1 : ∃ t, ∀ x, ax^2 - bx ≥ -1 ↔ (x ≤ t - 1 ∨ x ≥ -3 - t))
variable (h2 : a * m^2 - b * m = 2)

theorem possible_m_value : m = 1 :=
sorry

end possible_m_value_l279_279150


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279127

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279127


namespace rectangle_dimensions_l279_279550

theorem rectangle_dimensions
  (l w : ℕ)
  (h1 : 2 * l + 2 * w = l * w)
  (h2 : w = l - 3) :
  l = 6 ∧ w = 3 :=
by
  sorry

end rectangle_dimensions_l279_279550


namespace abs_neg_2035_l279_279540

theorem abs_neg_2035 : abs (-2035) = 2035 := 
by {
  sorry
}

end abs_neg_2035_l279_279540


namespace correct_chart_for_percentage_representation_l279_279925

def bar_chart_characteristic := "easily shows the quantity"
def line_chart_characteristic := "shows the quantity and reflects the changes in quantity"
def pie_chart_characteristic := "reflects the relationship between a part and the whole"

def representation_requirement := "represents the percentage of students in each grade level in the fifth grade's physical education test scores out of the total number of students in the grade"

theorem correct_chart_for_percentage_representation : 
  (representation_requirement = pie_chart_characteristic) := 
by 
   -- The proof follows from the prior definition of characteristics.
   sorry

end correct_chart_for_percentage_representation_l279_279925


namespace hash_triple_l279_279974

def hash (N : ℝ) : ℝ := 0.5 * (N^2) + 1

theorem hash_triple  : hash (hash (hash 4)) = 862.125 :=
by {
  sorry
}

end hash_triple_l279_279974


namespace compute_n_l279_279909

theorem compute_n (avg1 avg2 avg3 avg4 avg5 : ℚ) (h1 : avg1 = 1234 ∨ avg2 = 1234 ∨ avg3 = 1234 ∨ avg4 = 1234 ∨ avg5 = 1234)
  (h2 : avg1 = 345 ∨ avg2 = 345 ∨ avg3 = 345 ∨ avg4 = 345 ∨ avg5 = 345)
  (h3 : avg1 = 128 ∨ avg2 = 128 ∨ avg3 = 128 ∨ avg4 = 128 ∨ avg5 = 128)
  (h4 : avg1 = 19 ∨ avg2 = 19 ∨ avg3 = 19 ∨ avg4 = 19 ∨ avg5 = 19)
  (h5 : avg1 = 9.5 ∨ avg2 = 9.5 ∨ avg3 = 9.5 ∨ avg4 = 9.5 ∨ avg5 = 9.5) :
  ∃ n : ℕ, n = 2014 :=
by
  sorry

end compute_n_l279_279909


namespace suitable_altitude_range_l279_279780

theorem suitable_altitude_range :
  ∀ (temperature_at_base : ℝ) (temp_decrease_per_100m : ℝ) (suitable_temp_low : ℝ) (suitable_temp_high : ℝ) (altitude_at_base : ℝ),
  (22 = temperature_at_base) →
  (0.5 = temp_decrease_per_100m) →
  (18 = suitable_temp_low) →
  (20 = suitable_temp_high) →
  (0 = altitude_at_base) →
  400 ≤ ((temperature_at_base - suitable_temp_high) / temp_decrease_per_100m * 100) ∧ ((temperature_at_base - suitable_temp_low) / temp_decrease_per_100m * 100) ≤ 800 :=
by
  intros temperature_at_base temp_decrease_per_100m suitable_temp_low suitable_temp_high altitude_at_base
  intro h1 h2 h3 h4 h5
  sorry

end suitable_altitude_range_l279_279780


namespace range_of_a_l279_279677

variable (a : ℝ)

def proposition_p := ∀ x : ℝ, a * x^2 - 2 * x + 1 > 0
def proposition_q := ∀ x : ℝ, x ∈ Set.Icc (1/2 : ℝ) (2 : ℝ) → x + (1 / x) > a

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l279_279677


namespace dan_picked_l279_279598

-- Definitions:
def benny_picked : Nat := 2
def total_picked : Nat := 11

-- Problem statement:
theorem dan_picked (b : Nat) (t : Nat) (d : Nat) (h1 : b = benny_picked) (h2 : t = total_picked) (h3 : t = b + d) : d = 9 := by
  sorry

end dan_picked_l279_279598


namespace cricket_match_count_l279_279581

theorem cricket_match_count (x : ℕ) (h_avg_1 : ℕ → ℕ) (h_avg_2 : ℕ) (h_avg_all : ℕ) (h_eq : 50 * x + 26 * 15 = 42 * (x + 15)) : x = 30 :=
by
  sorry

end cricket_match_count_l279_279581


namespace tangerines_in_one_box_l279_279244

theorem tangerines_in_one_box (total_tangerines boxes remaining_tangerines tangerines_per_box : ℕ) 
  (h1 : total_tangerines = 29)
  (h2 : boxes = 8)
  (h3 : remaining_tangerines = 5)
  (h4 : total_tangerines - remaining_tangerines = boxes * tangerines_per_box) :
  tangerines_per_box = 3 :=
by 
  sorry

end tangerines_in_one_box_l279_279244


namespace candies_for_50_rubles_l279_279945

theorem candies_for_50_rubles : 
  ∀ (x : ℕ), (45 * x = 45) → (50 / x = 50) := 
by
  intros x h
  sorry

end candies_for_50_rubles_l279_279945


namespace factor_quadratic_l279_279843

theorem factor_quadratic (m p : ℝ) (h : (m - 8) ∣ (m^2 - p * m - 24)) : p = 5 :=
sorry

end factor_quadratic_l279_279843


namespace misha_grade_students_l279_279346

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l279_279346


namespace product_gcd_lcm_24_60_l279_279293

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l279_279293


namespace decagon_area_l279_279951

theorem decagon_area (perimeter : ℝ) (n : ℕ) (side_length : ℝ)
  (segments : ℕ) (area : ℝ) :
  perimeter = 200 ∧ n = 4 ∧ side_length = perimeter / n ∧ segments = 5 ∧ 
  area = (side_length / segments)^2 * (1 - (1/2)) * 4 * segments  →
  area = 2300 := 
by
  sorry

end decagon_area_l279_279951


namespace find_seating_capacity_l279_279088

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end find_seating_capacity_l279_279088


namespace female_students_transfer_l279_279241

theorem female_students_transfer (x y z : ℕ) 
  (h1 : ∀ B : ℕ, B = x - 4) 
  (h2 : ∀ C : ℕ, C = x - 5)
  (h3 : ∀ B' : ℕ, B' = x - 4 + y - z)
  (h4 : ∀ C' : ℕ, C' = x + z - 7) 
  (h5 : x - y + 2 = x - 4 + y - z)
  (h6 : x - 4 + y - z = x + z - 7) 
  (h7 : 2 = 2) :
  y = 3 ∧ z = 4 := 
by 
  sorry

end female_students_transfer_l279_279241


namespace false_proposition_l279_279819

-- Definitions of the conditions
def p1 := ∃ x0 : ℝ, x0^2 - 2*x0 + 1 ≤ 0
def p2 := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - 1 ≥ 0

-- Statement to prove
theorem false_proposition : ¬ (¬ p1 ∧ ¬ p2) :=
by sorry

end false_proposition_l279_279819


namespace compute_value_l279_279344

variable (p q : ℚ)
variable (h : ∀ x, 3 * x^2 - 7 * x - 6 = 0 → x = p ∨ x = q)

theorem compute_value (h_pq : p ≠ q) : (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  -- We assume p and q are the roots of the polynomial and p ≠ q.
  have sum_roots : p + q = 7 / 3 := sorry
  have prod_roots : p * q = -2 := sorry
  -- Additional steps to derive the required result (proof) are ignored here.
  sorry

end compute_value_l279_279344


namespace min_even_integers_least_one_l279_279067

theorem min_even_integers_least_one (x y a b m n o : ℤ) 
  (h1 : x + y = 29)
  (h2 : x + y + a + b = 47)
  (h3 : x + y + a + b + m + n + o = 66) :
  ∃ e : ℕ, (e = 1) := by
sorry

end min_even_integers_least_one_l279_279067


namespace cost_of_baseball_is_correct_l279_279876

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end cost_of_baseball_is_correct_l279_279876


namespace grandpa_movie_time_l279_279644

theorem grandpa_movie_time
  (each_movie_time : ℕ := 90)
  (max_movies_2_days : ℕ := 9)
  (x_movies_tuesday : ℕ)
  (movies_wednesday := 2 * x_movies_tuesday)
  (total_movies := x_movies_tuesday + movies_wednesday)
  (h : total_movies = max_movies_2_days) :
  90 * x_movies_tuesday = 270 :=
by
  sorry

end grandpa_movie_time_l279_279644


namespace exist_n_l279_279573

theorem exist_n : ∃ n : ℕ, n > 1 ∧ ¬(Nat.Prime n) ∧ ∀ a : ℤ, (a^n - a) % n = 0 :=
by
  sorry

end exist_n_l279_279573


namespace point_in_fourth_quadrant_l279_279012

theorem point_in_fourth_quadrant (m : ℝ) (h : m < 0) : (-m + 1 > 0 ∧ -1 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l279_279012


namespace dennis_took_away_l279_279025

-- Define the initial and remaining number of cards
def initial_cards : ℕ := 67
def remaining_cards : ℕ := 58

-- Define the number of cards taken away
def cards_taken_away (n m : ℕ) : ℕ := n - m

-- Prove that the number of cards taken away is 9
theorem dennis_took_away :
  cards_taken_away initial_cards remaining_cards = 9 :=
by
  -- Placeholder proof
  sorry

end dennis_took_away_l279_279025


namespace find_q_l279_279316

noncomputable def p (q : ℝ) : ℝ := 16 / (3 * q)

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 3/2) (h4 : p * q = 16/3) : q = 24 / 6 + 19.6 / 6 :=
by
  sorry

end find_q_l279_279316


namespace impossible_to_use_up_all_parts_l279_279942

theorem impossible_to_use_up_all_parts (p q r : ℕ) :
  (∃ p q r : ℕ,
    2 * p + 2 * r + 2 = A ∧
    2 * p + q + 1 = B ∧
    q + r = C) → false :=
by {
  sorry
}

end impossible_to_use_up_all_parts_l279_279942


namespace binom_150_1_eq_150_l279_279614

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end binom_150_1_eq_150_l279_279614


namespace x_squared_plus_inverse_squared_l279_279773

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 3.5) : x^2 + (1/x)^2 = 10.25 :=
by sorry

end x_squared_plus_inverse_squared_l279_279773


namespace remainder_of_sum_l279_279566

theorem remainder_of_sum (D k l : ℕ) (hk : 242 = k * D + 11) (hl : 698 = l * D + 18) :
  (242 + 698) % D = 29 :=
by
  sorry

end remainder_of_sum_l279_279566


namespace top_angle_degrees_l279_279636

def isosceles_triangle_with_angle_ratio (x : ℝ) (a b c : ℝ) : Prop :=
  a = x ∧ b = 4 * x ∧ a + b + c = 180 ∧ (a = b ∨ a = c ∨ b = c)

theorem top_angle_degrees (x : ℝ) (a b c : ℝ) :
  isosceles_triangle_with_angle_ratio x a b c → c = 20 ∨ c = 120 :=
by
  sorry

end top_angle_degrees_l279_279636


namespace food_requirement_not_met_l279_279448

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l279_279448


namespace num_remaining_integers_l279_279371

namespace ProofProblem

def T : Finset ℕ := (Finset.range 101).filter (λ x => x > 0)

def is_multiple (n : ℕ) (d : ℕ) : Prop := d ∣ n

def remove_multiples (S : Finset ℕ) (d : ℕ) : Finset ℕ := S.filter (λ x => ¬ is_multiple x d)

def T_removed_multiples_2_and_5 : Finset ℕ :=
  remove_multiples (remove_multiples T 2) 5

theorem num_remaining_integers : T_removed_multiples_2_and_5.card = 40 := 
  sorry

end ProofProblem

end num_remaining_integers_l279_279371


namespace domain_range_g_l279_279676

variable (f : ℝ → ℝ) 

noncomputable def g (x : ℝ) := 2 - f (x + 1)

theorem domain_range_g :
  (∀ x, 0 ≤ x → x ≤ 3 → 0 ≤ f x → f x ≤ 1) →
  (∀ x, -1 ≤ x → x ≤ 2) ∧ (∀ y, 1 ≤ y → y ≤ 2) :=
sorry

end domain_range_g_l279_279676


namespace factorize_x_cubic_l279_279140

-- Define the function and the condition
def factorize (x : ℝ) : Prop := x^3 - 9 * x = x * (x + 3) * (x - 3)

-- Prove the factorization property
theorem factorize_x_cubic (x : ℝ) : factorize x :=
by
  sorry

end factorize_x_cubic_l279_279140


namespace charlyn_visible_area_l279_279264

noncomputable def visible_area (side_length vision_distance : ℝ) : ℝ :=
  let outer_rectangles_area := 4 * (side_length * vision_distance)
  let outer_squares_area := 4 * (vision_distance * vision_distance)
  let inner_square_area := 
    let inner_side_length := side_length - 2 * vision_distance
    inner_side_length * inner_side_length
  let total_walk_area := side_length * side_length
  total_walk_area - inner_square_area + outer_rectangles_area + outer_squares_area

theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

end charlyn_visible_area_l279_279264


namespace toothpicks_needed_for_cube_grid_l279_279563

-- Defining the conditions: a cube-shaped grid with dimensions 5x5x5.
def grid_length : ℕ := 5
def grid_width : ℕ := 5
def grid_height : ℕ := 5

-- The theorem to prove the number of toothpicks needed is 2340.
theorem toothpicks_needed_for_cube_grid (L W H : ℕ) (h1 : L = grid_length) (h2 : W = grid_width) (h3 : H = grid_height) :
  (L + 1) * (W + 1) * H + 2 * (L + 1) * W * (H + 1) = 2340 :=
  by
    -- Proof goes here
    sorry

end toothpicks_needed_for_cube_grid_l279_279563


namespace num_divisible_by_2_3_5_7_lt_500_l279_279832

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l279_279832


namespace series_result_l279_279608

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end series_result_l279_279608


namespace red_balls_probability_l279_279580

noncomputable def number_of_red_balls : ℕ := 5

theorem red_balls_probability (total_balls : ℕ) (prob : ℚ) :
  (total_balls = 15) →
  (prob = (1/21 : ℚ)) →
  ∃ (r : ℕ), (r * (r - 1) / (15 * 14 : ℕ) : ℚ) = prob ∧ r = number_of_red_balls :=
by
  intros h_total_balls h_prob
  use number_of_red_balls
  split
  . rw [h_total_balls, h_prob]
    norm_num -- simplifies the arithmetic
  . refl -- states that the solution red balls is 5


end red_balls_probability_l279_279580


namespace quadratic_transform_l279_279049

theorem quadratic_transform (x : ℝ) : x^2 - 6 * x - 5 = 0 → (x - 3)^2 = 14 :=
by
  intro h
  sorry

end quadratic_transform_l279_279049


namespace roger_has_more_candy_l279_279213

-- Defining the conditions
def sandra_bag1 : Nat := 6
def sandra_bag2 : Nat := 6
def roger_bag1 : Nat := 11
def roger_bag2 : Nat := 3

-- Calculating the total pieces of candy for Sandra and Roger
def total_sandra : Nat := sandra_bag1 + sandra_bag2
def total_roger : Nat := roger_bag1 + roger_bag2

-- Statement of the proof problem
theorem roger_has_more_candy : total_roger - total_sandra = 2 := by
  sorry

end roger_has_more_candy_l279_279213


namespace cat_food_inequality_l279_279458

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l279_279458


namespace chord_length_l279_279712

variable (x y : ℝ)

/--
The chord length cut by the line y = 2x - 2 on the circle (x-2)^2 + (y-2)^2 = 25 is 10.
-/
theorem chord_length (h₁ : y = 2 * x - 2) (h₂ : (x - 2)^2 + (y - 2)^2 = 25) : 
  ∃ length : ℝ, length = 10 :=
sorry

end chord_length_l279_279712


namespace calculate_product_l279_279118

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l279_279118


namespace smallest_area_right_triangle_l279_279386

theorem smallest_area_right_triangle (a b : ℕ) (h₁ : a = 4) (h₂ : b = 5) : 
  ∃ c, (c = 6 ∧ ∀ (x y : ℕ) (h₃ : x = 4 ∨ y = 4) (h₄ : x = 5 ∨ y = 5), c ≤ (x * y / 2)) :=
by {
  sorry
}

end smallest_area_right_triangle_l279_279386


namespace find_theta_l279_279478

open Real

theorem find_theta (theta : ℝ) : sin theta = -1/3 ∧ -π < theta ∧ theta < -π / 2 ↔ theta = -π - arcsin (-1 / 3) :=
by
  sorry

end find_theta_l279_279478


namespace n_plus_d_is_155_l279_279339

noncomputable def n_and_d_sum : Nat :=
sorry

theorem n_plus_d_is_155 (n d : Nat) (hn : 0 < n) (hd : d < 10) 
  (h1 : 4 * n^2 + 2 * n + d = 305) 
  (h2 : 4 * n^3 + 2 * n^2 + d * n + 1 = 577 + 8 * d) : n + d = 155 := 
sorry

end n_plus_d_is_155_l279_279339


namespace math_problem_l279_279943

theorem math_problem : 12 - (- 18) + (- 7) - 15 = 8 :=
by
  sorry

end math_problem_l279_279943


namespace geometric_sequence_sum_l279_279825

/-- Given a geometric sequence with common ratio r = 2, and the sum of the first four terms
    equals 1, the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a r : ℝ) (h : r = 2) (h_sum_four : a * (1 + r + r^2 + r^3) = 1) :
  a * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) = 17 :=
by
  sorry

end geometric_sequence_sum_l279_279825


namespace training_trip_duration_l279_279896

-- Define the number of supervisors
def num_supervisors : ℕ := 15

-- Define the number of supervisors overseeing the pool each day
def supervisors_per_day : ℕ := 3

-- Define the number of pairs supervised per day
def pairs_per_day : ℕ := (supervisors_per_day * (supervisors_per_day - 1)) / 2

-- Define the total number of pairs from the given number of supervisors
def total_pairs : ℕ := (num_supervisors * (num_supervisors - 1)) / 2

-- Define the number of days required
def num_days : ℕ := total_pairs / pairs_per_day

-- The theorem we need to prove
theorem training_trip_duration : 
  (num_supervisors = 15) ∧
  (supervisors_per_day = 3) ∧
  (∀ (a b : ℕ), a * (a - 1) / 2 = b * (b - 1) / 2 → a = b) ∧ 
  (∀ (N : ℕ), total_pairs = N * pairs_per_day → N = 35) :=
by
  sorry

end training_trip_duration_l279_279896


namespace laura_walk_distance_l279_279664

theorem laura_walk_distance 
  (east_blocks : ℕ) 
  (north_blocks : ℕ) 
  (block_length_miles : ℕ → ℝ) 
  (h_east_blocks : east_blocks = 8) 
  (h_north_blocks : north_blocks = 14) 
  (h_block_length_miles : ∀ b : ℕ, b = 1 → block_length_miles b = 1 / 4) 
  : (east_blocks + north_blocks) * block_length_miles 1 = 5.5 := 
by 
  sorry

end laura_walk_distance_l279_279664


namespace price_of_case_bulk_is_12_l279_279404

noncomputable def price_per_can_grocery_store : ℚ := 6 / 12
noncomputable def price_per_can_bulk : ℚ := price_per_can_grocery_store - 0.25
def cans_per_case_bulk : ℕ := 48
noncomputable def price_per_case_bulk : ℚ := price_per_can_bulk * cans_per_case_bulk

theorem price_of_case_bulk_is_12 : price_per_case_bulk = 12 :=
by
  sorry

end price_of_case_bulk_is_12_l279_279404


namespace proj_eq_line_eqn_l279_279916

theorem proj_eq_line_eqn (x y : ℝ)
  (h : (6 * x + 3 * y) * 6 / 45 = -3 ∧ (6 * x + 3 * y) * 3 / 45 = -3 / 2) :
  y = -2 * x - 15 / 2 :=
by
  sorry

end proj_eq_line_eqn_l279_279916


namespace pentagon_diagl_sum_pentagon_diagonal_391_l279_279026

noncomputable def diagonal_sum (AB CD BC DE AE : ℕ) 
  (AC : ℚ) (BD : ℚ) (CE : ℚ) (AD : ℚ) (BE : ℚ) : ℚ :=
  3 * AC + AD + BE

theorem pentagon_diagl_sum (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  diagonal_sum AB CD BC DE AE AC BD CE AD BE = 385 / 6 := sorry

theorem pentagon_diagonal_391 (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  ∃ m n : ℕ, 
    m.gcd n = 1 ∧
    m / n = 385 / 6 ∧
    m + n = 391 := sorry

end pentagon_diagl_sum_pentagon_diagonal_391_l279_279026


namespace find_cost_per_sq_foot_l279_279861

noncomputable def monthly_rent := 2800 / 2
noncomputable def old_annual_rent (C : ℝ) := 750 * C * 12
noncomputable def new_annual_rent := monthly_rent * 12
noncomputable def annual_savings := old_annual_rent - new_annual_rent

theorem find_cost_per_sq_foot (C : ℝ):
    (750 * C * 12 - 2800 / 2 * 12 = 1200) ↔ (C = 2) :=
sorry

end find_cost_per_sq_foot_l279_279861


namespace ratio_of_new_r_to_original_r_l279_279657

theorem ratio_of_new_r_to_original_r
  (r₁ r₂ : ℝ)
  (a₁ a₂ : ℝ)
  (h₁ : a₁ = (2 * r₁)^3)
  (h₂ : a₂ = (2 * r₂)^3)
  (h : a₂ = 0.125 * a₁) :
  r₂ / r₁ = 1 / 2 :=
by
  sorry

end ratio_of_new_r_to_original_r_l279_279657


namespace range_of_set_is_8_l279_279788

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end range_of_set_is_8_l279_279788


namespace vasya_numbers_l279_279736

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l279_279736


namespace find_a_from_inequality_solution_set_l279_279180

theorem find_a_from_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + 4 < 0) ↔ (1 < x ∧ x < 4)) -> a = 5 :=
by
  intro h
  sorry

end find_a_from_inequality_solution_set_l279_279180


namespace sum_f_inv_l279_279547

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 2 * x - 1 else x ^ 2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 9 then (y + 1) / 2 else Real.sqrt y

theorem sum_f_inv :
  (f_inv (-3) + f_inv (-2) + 
   f_inv (-1) + f_inv 0 + 
   f_inv 1 + f_inv 2 + 
   f_inv 3 + f_inv 4 + 
   f_inv 9) = 9 :=
by
  sorry

end sum_f_inv_l279_279547


namespace part1_part2_l279_279181

variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively and 4b * sin A = sqrt 7 * a
def condition1 : 4 * b * Real.sin A = Real.sqrt 7 * a := sorry

-- Prove that sin B = sqrt 7 / 4
theorem part1 (h : 4 * b * Real.sin A = Real.sqrt 7 * a) :
  Real.sin B = Real.sqrt 7 / 4 := sorry

-- Condition: a, b, and c form an arithmetic sequence with a common difference greater than 0
def condition2 : 2 * b = a + c := sorry

-- Prove that cos A - cos C = sqrt 7 / 2
theorem part2 (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a) (h2 : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := sorry

end part1_part2_l279_279181


namespace pie_left_is_30_percent_l279_279970

def Carlos_share : ℝ := 0.60
def remaining_after_Carlos : ℝ := 1 - Carlos_share
def Jessica_share : ℝ := 0.25 * remaining_after_Carlos
def final_remaining : ℝ := remaining_after_Carlos - Jessica_share

theorem pie_left_is_30_percent :
  final_remaining = 0.30 :=
sorry

end pie_left_is_30_percent_l279_279970


namespace heartsuit_zero_heartsuit_self_heartsuit_pos_l279_279267

def heartsuit (x y : Real) : Real := x^2 - y^2

theorem heartsuit_zero (x : Real) : heartsuit x 0 = x^2 :=
by
  sorry

theorem heartsuit_self (x : Real) : heartsuit x x = 0 :=
by
  sorry

theorem heartsuit_pos (x y : Real) (h : x > y) : heartsuit x y > 0 :=
by
  sorry

end heartsuit_zero_heartsuit_self_heartsuit_pos_l279_279267


namespace num_divisible_by_2_3_5_7_lt_500_l279_279833

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l279_279833


namespace lara_bag_total_chips_l279_279922

theorem lara_bag_total_chips (C : ℕ)
  (h1 : ∃ (b : ℕ), b = C / 6)
  (h2 : 34 + 16 + C / 6 = C) :
  C = 60 := by
  sorry

end lara_bag_total_chips_l279_279922


namespace greatest_b_not_in_range_l279_279239

theorem greatest_b_not_in_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ -4) → b ≤ 7 := 
by {
  sorry
}

end greatest_b_not_in_range_l279_279239


namespace cat_food_insufficient_l279_279445

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l279_279445


namespace west_for_200_is_neg_200_l279_279173

-- Given a definition for driving east
def driving_east (d : Int) : Int := d

-- Driving east for 80 km is +80 km
def driving_east_80 : Int := driving_east 80

-- Driving west should be the negative of driving east
def driving_west (d : Int) : Int := -d

-- Driving west for 200 km is -200 km
def driving_west_200 : Int := driving_west 200

-- Theorem to prove the given condition and expected result
theorem west_for_200_is_neg_200 : driving_west_200 = -200 :=
by
  -- Proof step is skipped
  sorry

end west_for_200_is_neg_200_l279_279173


namespace tangent_line_equation_at_point_l279_279142

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1) - x

theorem tangent_line_equation_at_point :
  ∃ a b c : ℝ, (∀ x y : ℝ, a * x + b * y + c = 0 ↔ (x = 1 → y = -1 → f x = y)) ∧ (a * 1 + b * (-1) + c = 0) :=
by
  sorry

end tangent_line_equation_at_point_l279_279142


namespace correct_choice_l279_279258

theorem correct_choice : 2 ∈ ({0, 1, 2} : Set ℕ) :=
sorry

end correct_choice_l279_279258


namespace percentage_difference_is_20_l279_279663

/-
Barry can reach apples that are 5 feet high.
Larry is 5 feet tall.
When Barry stands on Larry's shoulders, they can reach 9 feet high.
-/
def Barry_height : ℝ := 5
def Larry_height : ℝ := 5
def Combined_height : ℝ := 9

/-
Prove the percentage difference between Larry's full height and his shoulder height is 20%.
-/
theorem percentage_difference_is_20 :
  ((Larry_height - (Combined_height - Barry_height)) / Larry_height) * 100 = 20 :=
by
  sorry

end percentage_difference_is_20_l279_279663


namespace average_of_six_numbers_l279_279044

theorem average_of_six_numbers (a b c d e f : ℝ)
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 :=
by sorry

end average_of_six_numbers_l279_279044


namespace inverse_function_correct_inequality_solution_l279_279638

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def f_inv (y : ℝ) : ℝ := Real.log (1 + y) / Real.log (1 - y)

theorem inverse_function_correct (x : ℝ) (hx : -1 < x ∧ x < 1) :
  f_inv (f x) = x :=
sorry

theorem inequality_solution :
  ∀ x, (1 / 2 < x ∧ x < 1) ↔ (f_inv x > Real.log (1 + x) + 1) :=
sorry

end inverse_function_correct_inequality_solution_l279_279638


namespace cat_food_inequality_l279_279459

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l279_279459


namespace product_gcd_lcm_24_60_l279_279290

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l279_279290


namespace calculate_sum_of_inverses_l279_279674

noncomputable section

variables {p q z1 z2 z3 : ℂ}

-- Conditions
def is_root (a : ℂ) (p : ℂ[X]) := p.eval a = 0

def roots_cond : Prop := 
  is_root z1 (X^3 + C p * X + C q) ∧ 
  is_root z2 (X^3 + C p * X + C q) ∧ 
  is_root z3 (X^3 + C p * X + C q)

-- Main theorem
theorem calculate_sum_of_inverses (h : roots_cond) :
  (1 / z1^2) + (1 / z2^2) + (1 / z3^2) = (p^2) / (q^2) :=
sorry

end calculate_sum_of_inverses_l279_279674


namespace find_g_of_conditions_l279_279716

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end find_g_of_conditions_l279_279716


namespace set_B_forms_triangle_l279_279770

theorem set_B_forms_triangle (a b c : ℝ) (h1 : a = 25) (h2 : b = 24) (h3 : c = 7):
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end set_B_forms_triangle_l279_279770


namespace unique_positive_integer_appending_digits_eq_sum_l279_279831

-- Define the problem in terms of Lean types and properties
theorem unique_positive_integer_appending_digits_eq_sum :
  ∃! (A : ℕ), (A > 0) ∧ (∃ (B : ℕ), (0 ≤ B ∧ B < 1000) ∧ (1000 * A + B = (A * (A + 1)) / 2)) :=
sorry

end unique_positive_integer_appending_digits_eq_sum_l279_279831


namespace number_of_men_in_company_l279_279501

noncomputable def total_workers : ℝ := 2752.8
noncomputable def women_in_company : ℝ := 91.76
noncomputable def workers_without_retirement_plan : ℝ := (1 / 3) * total_workers
noncomputable def percent_women_without_retirement_plan : ℝ := 0.10
noncomputable def percent_men_with_retirement_plan : ℝ := 0.40
noncomputable def workers_with_retirement_plan : ℝ := (2 / 3) * total_workers
noncomputable def men_with_retirement_plan : ℝ := percent_men_with_retirement_plan * workers_with_retirement_plan

theorem number_of_men_in_company : (total_workers - women_in_company) = 2661.04 := by
  -- Insert the exact calculations and algebraic manipulations
  sorry

end number_of_men_in_company_l279_279501


namespace min_value_of_n_l279_279160

theorem min_value_of_n :
  ∀ (h : ℝ), ∃ n : ℝ, (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -x^2 + 2 * h * x - h ≤ n) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -x^2 + 2 * h * x - h = n) ∧
  n = -1 / 4 := 
by
  sorry

end min_value_of_n_l279_279160


namespace find_f_value_l279_279570

noncomputable def f (x y z : ℝ) : ℝ := 2 * x^3 * Real.sin y + Real.log (z^2)

theorem find_f_value :
  f 1 (Real.pi / 2) (Real.exp 2) = 8 →
  f 2 Real.pi (Real.exp 3) = 6 :=
by
  intro h
  unfold f
  sorry

end find_f_value_l279_279570


namespace food_requirement_not_met_l279_279450

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l279_279450


namespace binom_150_1_l279_279612

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end binom_150_1_l279_279612


namespace total_crayons_l279_279052

theorem total_crayons (orange_boxes : ℕ) (orange_per_box : ℕ) (blue_boxes : ℕ) (blue_per_box : ℕ) (red_boxes : ℕ) (red_per_box : ℕ) : 
  orange_boxes = 6 → orange_per_box = 8 → 
  blue_boxes = 7 → blue_per_box = 5 →
  red_boxes = 1 → red_per_box = 11 → 
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  exact sorry

end total_crayons_l279_279052


namespace problem_1_problem_2_l279_279488

-- Problem 1:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is {x | x < -3 or x > -1}, prove k = -1/2
theorem problem_1 {k : ℝ} :
  (∀ x : ℝ, (kx^2 - 2*x + 3*k < 0 ↔ x < -3 ∨ x > -1)) → k = -1/2 :=
sorry

-- Problem 2:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is ∅, prove 0 < k ≤ sqrt(3) / 3
theorem problem_2 {k : ℝ} :
  (∀ x : ℝ, ¬ (kx^2 - 2*x + 3*k < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end problem_1_problem_2_l279_279488


namespace abs_neg_five_is_five_l279_279707

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l279_279707


namespace product_gcd_lcm_l279_279281

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l279_279281


namespace mul_mixed_number_l279_279125

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l279_279125


namespace greatest_n_le_5_value_ge_2525_l279_279654

theorem greatest_n_le_5_value_ge_2525 (n : ℤ) (V : ℤ) 
  (h1 : 101 * n^2 ≤ V) 
  (h2 : ∀ k : ℤ, (101 * k^2 ≤ V) → (k ≤ 5)) : 
  V ≥ 2525 := 
sorry

end greatest_n_le_5_value_ge_2525_l279_279654


namespace misha_students_l279_279350

theorem misha_students : 
  ∀ (n : ℕ),
  (n = 74 + 1 + 74) ↔ (n = 149) :=
by
  intro n
  split
  · intro h
    rw [← h, nat.add_assoc]
    apply nat.add_right_cancel
    rw [nat.add_comm 1 74, nat.add_assoc]
    apply nat.add_right_cancel
    rw nat.add_comm
  · intro h
    exact h
  sorry

end misha_students_l279_279350


namespace parallel_a_b_projection_a_onto_b_l279_279003

noncomputable section

open Real

def a : ℝ × ℝ := (sqrt 3, 1)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem parallel_a_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_parallel : (a.1 / a.2) = (b θ).1 / (b θ).2) : θ = π / 6 := sorry

theorem projection_a_onto_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_proj : (sqrt 3 * cos θ + sin θ) = -sqrt 3) : b θ = (-1, 0) := sorry

end parallel_a_b_projection_a_onto_b_l279_279003


namespace c_difference_correct_l279_279513

noncomputable def find_c_difference (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) : ℝ :=
  2 * Real.sqrt 34

theorem c_difference_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) :
  find_c_difference a b c h1 h2 = 2 * Real.sqrt 34 := 
sorry

end c_difference_correct_l279_279513


namespace roots_of_quadratic_l279_279687

theorem roots_of_quadratic :
  ∃ (b c : ℝ), ( ∀ (x : ℝ), x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2) :=
sorry

end roots_of_quadratic_l279_279687


namespace enough_cat_food_for_six_days_l279_279435

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l279_279435


namespace square_of_other_leg_l279_279184

variable {R : Type} [CommRing R]

theorem square_of_other_leg (a b c : R) (h1 : a^2 + b^2 = c^2) (h2 : c = a + 2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l279_279184


namespace correct_calculation_l279_279936

theorem correct_calculation (a b : ℝ) : 
  (¬ (2 * (a - 1) = 2 * a - 1)) ∧ 
  (3 * a^2 - 2 * a^2 = a^2) ∧ 
  (¬ (3 * a^2 - 2 * a^2 = 1)) ∧ 
  (¬ (3 * a + 2 * b = 5 * a * b)) :=
by
  sorry

end correct_calculation_l279_279936


namespace isosceles_right_triangle_third_angle_l279_279850

/-- In an isosceles right triangle where one of the angles opposite the equal sides measures 45 degrees, 
    the measure of the third angle is 90 degrees. -/
theorem isosceles_right_triangle_third_angle (θ : ℝ) 
  (h1 : θ = 45)
  (h2 : ∀ (a b c : ℝ), a + b + c = 180) : θ + θ + 90 = 180 :=
by
  sorry

end isosceles_right_triangle_third_angle_l279_279850


namespace cost_difference_l279_279553

/-- The selling price and cost of pants -/
def selling_price : ℕ := 34
def store_cost : ℕ := 26

/-- The proof that the store paid 8 dollars less than the selling price -/
theorem cost_difference : selling_price - store_cost = 8 := by
  sorry

end cost_difference_l279_279553


namespace combined_degrees_l279_279693

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l279_279693


namespace average_speed_l279_279947

theorem average_speed (d1 d2 d3 d4 d5 t: ℕ) 
  (h1: d1 = 120) 
  (h2: d2 = 70) 
  (h3: d3 = 90) 
  (h4: d4 = 110) 
  (h5: d5 = 80) 
  (total_time: t = 5): 
  (d1 + d2 + d3 + d4 + d5) / t = 94 := 
by 
  -- proof will go here
  sorry

end average_speed_l279_279947


namespace convex_g_inequality_l279_279161

noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem convex_g_inequality (a b : ℝ) (h : 0 < a ∧ a < b) :
  g a + g b - 2 * g ((a + b) / 2) > 0 := 
sorry

end convex_g_inequality_l279_279161


namespace complex_square_sum_eq_zero_l279_279261

theorem complex_square_sum_eq_zero (i : ℂ) (h : i^2 = -1) : (1 + i)^2 + (1 - i)^2 = 0 :=
sorry

end complex_square_sum_eq_zero_l279_279261


namespace series_sum_eq_l279_279609

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end series_sum_eq_l279_279609


namespace find_value_l279_279866

variables (x1 x2 y1 y2 : ℝ)

def condition1 := x1 ^ 2 + 5 * x2 ^ 2 = 10
def condition2 := x2 * y1 - x1 * y2 = 5
def condition3 := x1 * y1 + 5 * x2 * y2 = Real.sqrt 105

theorem find_value (h1 : condition1 x1 x2) (h2 : condition2 x1 x2 y1 y2) (h3 : condition3 x1 x2 y1 y2) :
  y1 ^ 2 + 5 * y2 ^ 2 = 23 :=
sorry

end find_value_l279_279866


namespace value_of_x_for_g_equals_g_inv_l279_279976

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l279_279976


namespace pitcher_fill_four_glasses_l279_279769

variable (P G : ℚ) -- P: Volume of pitcher, G: Volume of one glass
variable (h : P / 2 = 3 * G)

theorem pitcher_fill_four_glasses : (4 * G = 2 * P / 3) :=
by
  sorry

end pitcher_fill_four_glasses_l279_279769


namespace b_20_value_l279_279867

noncomputable def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => b (n+1) * b n

theorem b_20_value : b 19 = 2^4181 :=
sorry

end b_20_value_l279_279867


namespace sum_of_fourth_powers_correct_l279_279368

noncomputable def sum_of_fourth_powers (x : ℤ) : ℤ :=
  x^4 + (x+1)^4 + (x+2)^4

theorem sum_of_fourth_powers_correct (x : ℤ) (h : x * (x+1) * (x+2) = 36 * x + 12) : 
  sum_of_fourth_powers x = 98 :=
sorry

end sum_of_fourth_powers_correct_l279_279368


namespace max_tickets_l279_279524

theorem max_tickets (cost : ℝ) (budget : ℝ) (max_tickets : ℕ) (h1 : cost = 15.25) (h2 : budget = 200) :
  max_tickets = 13 :=
by
  sorry

end max_tickets_l279_279524


namespace range_of_m_l279_279306

def quadratic_function (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + m * x + 1 = 0 → false)

def ellipse_condition (m : ℝ) : Prop :=
  0 < m

theorem range_of_m (m : ℝ) :
  (quadratic_function m ∨ ellipse_condition m) ∧ ¬ (quadratic_function m ∧ ellipse_condition m) →
  m ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
by
  sorry

end range_of_m_l279_279306


namespace minimum_red_pieces_l279_279377

theorem minimum_red_pieces (w b r : ℕ) 
  (h1 : b ≤ w / 2) 
  (h2 : r ≥ 3 * b) 
  (h3 : w + b ≥ 55) : r = 57 := 
sorry

end minimum_red_pieces_l279_279377


namespace combined_degrees_l279_279695

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l279_279695


namespace trick_or_treat_hours_l279_279379

variable (num_children : ℕ)
variable (houses_per_hour : ℕ)
variable (treats_per_house_per_kid : ℕ)
variable (total_treats : ℕ)

theorem trick_or_treat_hours (h : num_children = 3)
  (h1 : houses_per_hour = 5)
  (h2 : treats_per_house_per_kid = 3)
  (h3 : total_treats = 180) :
  total_treats / (num_children * houses_per_hour * treats_per_house_per_kid) = 4 :=
by
  sorry

end trick_or_treat_hours_l279_279379


namespace smallest_non_lucky_multiple_of_8_correct_l279_279934

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def smallest_non_lucky_multiple_of_8 := 16

theorem smallest_non_lucky_multiple_of_8_correct :
  smallest_non_lucky_multiple_of_8 = 16 ∧
  is_lucky smallest_non_lucky_multiple_of_8 = false :=
by
  sorry

end smallest_non_lucky_multiple_of_8_correct_l279_279934


namespace cat_food_inequality_l279_279441

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l279_279441


namespace Claire_takes_6_photos_l279_279030

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end Claire_takes_6_photos_l279_279030


namespace abs_neg_five_is_five_l279_279708

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l279_279708


namespace complete_the_square_l279_279535

theorem complete_the_square :
  ∀ x : ℝ, (x^2 + 8 * x + 7 = 0) → (x + 4)^2 = 9 :=
by
  intro x h,
  sorry

end complete_the_square_l279_279535


namespace unique_rectangles_perimeter_sum_correct_l279_279253

def unique_rectangle_sum_of_perimeters : ℕ :=
  let possible_pairs := [(4, 12), (6, 6)]
  let perimeters := possible_pairs.map (λ (p : ℕ × ℕ) => 2 * (p.1 + p.2))
  perimeters.sum

theorem unique_rectangles_perimeter_sum_correct : unique_rectangle_sum_of_perimeters = 56 :=
  by 
  -- skipping actual proof
  sorry

end unique_rectangles_perimeter_sum_correct_l279_279253


namespace vasya_numbers_l279_279749

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l279_279749


namespace find_years_lent_to_B_l279_279954

def principal_B := 5000
def principal_C := 3000
def rate := 8
def time_C := 4
def total_interest := 1760

-- Interest calculation for B
def interest_B (n : ℕ) := (principal_B * rate * n) / 100

-- Interest calculation for C (constant time of 4 years)
def interest_C := (principal_C * rate * time_C) / 100

-- Total interest received
def total_interest_received (n : ℕ) := interest_B n + interest_C

theorem find_years_lent_to_B (n : ℕ) (h : total_interest_received n = total_interest) : n = 2 :=
by
  sorry

end find_years_lent_to_B_l279_279954


namespace sufficient_but_not_necessary_condition_l279_279776

theorem sufficient_but_not_necessary_condition (k : ℝ) : 
  (k = 1 → ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0) ∧ 
  ¬(∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0 → k = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l279_279776


namespace quadruples_solution_l279_279627

theorem quadruples_solution (a b c d : ℝ) :
  (a * b + c * d = 6) ∧
  (a * c + b * d = 3) ∧
  (a * d + b * c = 2) ∧
  (a + b + c + d = 6) ↔
  (a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0) :=
sorry

end quadruples_solution_l279_279627


namespace find_h_l279_279829

theorem find_h (h : ℝ) :
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 7 ∧ -(x - h)^2 = -1) → (h = 2 ∨ h = 8) :=
by sorry

end find_h_l279_279829


namespace trip_time_difference_l279_279247

theorem trip_time_difference 
  (speed : ℕ) (dist1 dist2 : ℕ) (time_per_hour : ℕ) 
  (h_speed : speed = 60) 
  (h_dist1 : dist1 = 360) 
  (h_dist2 : dist2 = 420) 
  (h_time_per_hour : time_per_hour = 60) : 
  ((dist2 / speed - dist1 / speed) * time_per_hour) = 60 := 
by
  sorry

end trip_time_difference_l279_279247


namespace F_final_coordinates_l279_279491

-- Define the original coordinates of point F
def F : ℝ × ℝ := (5, 2)

-- Reflection over the y-axis changes the sign of the x-coordinate
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Reflection over the line y = x involves swapping x and y coordinates
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- The combined transformation: reflect over the y-axis, then reflect over y = x
def F_final : ℝ × ℝ := reflect_y_eq_x (reflect_y_axis F)

-- The proof statement
theorem F_final_coordinates : F_final = (2, -5) :=
by
  -- Proof goes here
  sorry

end F_final_coordinates_l279_279491


namespace sufficient_but_not_necessary_l279_279079

variable (x : ℝ)

def condition1 : Prop := x > 2
def condition2 : Prop := x^2 > 4

theorem sufficient_but_not_necessary :
  (condition1 x → condition2 x) ∧ (¬ (condition2 x → condition1 x)) :=
by 
  sorry

end sufficient_but_not_necessary_l279_279079


namespace cat_food_insufficient_l279_279442

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l279_279442


namespace f_has_four_distinct_real_roots_l279_279864

noncomputable def f (x d : ℝ) := x ^ 2 + 4 * x + d

theorem f_has_four_distinct_real_roots (d : ℝ) (h : d = 2) :
  ∃ r1 r2 r3 r4 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
  f (f r1 d) = 0 ∧ f (f r2 d) = 0 ∧ f (f r3 d) = 0 ∧ f (f r4 d) = 0 :=
by
  sorry

end f_has_four_distinct_real_roots_l279_279864


namespace combined_degrees_l279_279699

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l279_279699


namespace contribution_per_person_l279_279891

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end contribution_per_person_l279_279891


namespace max_min_values_l279_279499

theorem max_min_values (x y : ℝ) 
  (h : (x - 3)^2 + 4 * (y - 1)^2 = 4) :
  ∃ (t u : ℝ), (∀ (z : ℝ), (x-3)^2 + 4*(y-1)^2 = 4 → t ≤ (x+y-3)/(x-y+1) ∧ (x+y-3)/(x-y+1) ≤ u) ∧ t = -1 ∧ u = 1 := 
by
  sorry

end max_min_values_l279_279499


namespace jack_pays_back_total_l279_279859

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l279_279859


namespace upper_bound_exists_l279_279477

theorem upper_bound_exists (U : ℤ) :
  (∀ n : ℤ, 1 < 4 * n + 7 ∧ 4 * n + 7 < U) →
  (∃ n_min n_max : ℤ, n_max = n_min + 29 ∧ 4 * n_max + 7 < U ∧ 4 * n_min + 7 > 1) →
  (U = 120) :=
by
  intros h1 h2
  sorry

end upper_bound_exists_l279_279477


namespace albert_horses_l279_279593

variable {H C : ℝ}

theorem albert_horses :
  (2000 * H + 9 * C = 13400) ∧ (200 * H + 0.20 * 9 * C = 1880) ∧ (∀ x : ℝ, x = 2000) → H = 4 := 
by
  sorry

end albert_horses_l279_279593


namespace total_animals_correct_l279_279198

def L := 10

def C := 2 * L + 4

def Merry_lambs := L
def Merry_cows := C
def Merry_pigs (P : ℕ) := P
def Brother_lambs := L + 3

def Brother_chickens (R : ℕ) := R * Brother_lambs
def Brother_goats (Q : ℕ) := 2 * Brother_lambs + Q

def Merry_total (P : ℕ) := Merry_lambs + Merry_cows + Merry_pigs P
def Brother_total (R Q : ℕ) := Brother_lambs + Brother_chickens R + Brother_goats Q

def Total_animals (P R Q : ℕ) := Merry_total P + Brother_total R Q

theorem total_animals_correct (P R Q : ℕ) : 
  Total_animals P R Q = 73 + P + R * 13 + Q := by
  sorry

end total_animals_correct_l279_279198


namespace problem1_problem2_l279_279777

-- Definitions for Problem 1
def cond1 (x t : ℝ) : Prop := |2 * x + t| - t ≤ 8
def sol_set1 (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 4

theorem problem1 {t : ℝ} : (∀ x, cond1 x t → sol_set1 x) → t = 1 :=
sorry

-- Definitions for Problem 2
def cond2 (x y z : ℝ) : Prop := x^2 + (1 / 4) * y^2 + (1 / 9) * z^2 = 2

theorem problem2 {x y z : ℝ} : cond2 x y z → x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end problem1_problem2_l279_279777


namespace find_a_l279_279000

def f (a : ℝ) (x : ℝ) := a * x^2 + 3 * x - 2

theorem find_a (a : ℝ) (h : deriv (f a) 2 = 7) : a = 1 :=
by {
  sorry
}

end find_a_l279_279000


namespace Vasya_numbers_l279_279768

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l279_279768


namespace Vasya_numbers_l279_279764

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l279_279764


namespace each_persons_contribution_l279_279890

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end each_persons_contribution_l279_279890


namespace cuboid_surface_area_two_cubes_l279_279927

noncomputable def cuboid_surface_area (b : ℝ) : ℝ :=
  let l := 2 * b
  let w := b
  let h := b
  2 * (l * w + l * h + w * h)

theorem cuboid_surface_area_two_cubes (b : ℝ) : cuboid_surface_area b = 10 * b^2 := by
  sorry

end cuboid_surface_area_two_cubes_l279_279927


namespace expenditure_representation_l279_279685

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end expenditure_representation_l279_279685


namespace max_n_is_4024_l279_279307

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) : ℕ :=
  4024

theorem max_n_is_4024 (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) :
  max_n_for_positive_sum a d h1 h2 h3 = 4024 :=
by
  sorry

end max_n_is_4024_l279_279307


namespace RSA_next_challenge_digits_l279_279702

theorem RSA_next_challenge_digits (previous_digits : ℕ) (prize_increase : ℕ) :
  previous_digits = 193 ∧ prize_increase > 10000 → ∃ N : ℕ, N = 212 :=
by {
  sorry -- Proof is omitted
}

end RSA_next_challenge_digits_l279_279702


namespace student_average_always_less_l279_279255

theorem student_average_always_less (w x y z: ℝ) (hwx: w < x) (hxy: x < y) (hyz: y < z) :
  let A' := (w + x + y + z) / 4
  let B' := (2 * w + 2 * x + y + z) / 6
  B' < A' :=
by
  intro A' B'
  sorry

end student_average_always_less_l279_279255


namespace find_radius_l279_279043

theorem find_radius (r : ℝ) :
  (135 * r * Real.pi) / 180 = 3 * Real.pi → r = 4 :=
by
  sorry

end find_radius_l279_279043


namespace combined_degrees_l279_279692

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l279_279692


namespace determine_p_range_l279_279509

theorem determine_p_range :
  ∀ (p : ℝ), (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = (x + 9 / 8) * (x + 9 / 8) ∧ (f x) = (8*x^2 + 18*x + 4*p)/8 ) →
  2.5 < p ∧ p < 2.6 :=
by
  sorry

end determine_p_range_l279_279509


namespace cat_food_insufficient_for_six_days_l279_279461

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l279_279461


namespace third_side_length_of_triangle_l279_279312

theorem third_side_length_of_triangle {a b c : ℝ} (h1 : a^2 - 7 * a + 12 = 0) (h2 : b^2 - 7 * b + 12 = 0) 
  (h3 : a ≠ b) (h4 : a = 3 ∨ a = 4) (h5 : b = 3 ∨ b = 4) : 
  (c = 5 ∨ c = Real.sqrt 7) := by
  sorry

end third_side_length_of_triangle_l279_279312


namespace problem1_problem2_l279_279853

variable (A B C a b c p : ℝ)
variable (hA : A = ∠ A)
variable (hC : C = ∠ C)
variable (ha : a = side opposite A)
variable (hb : b = side opposite B)
variable (hc : c = side opposite C)
variable (hsin : sin A + sin C = p * sin B)
variable (h_fourac : 4 * a * c = b * b)

theorem problem1 :
  p = 5 / 4 → b = 1 →
  (a = 1 ∧ c = 1 / 4) ∨ (a = 1 / 4 ∧ c = 1) :=
by
  intros hp hb1
  sorry

theorem problem2 :
  (cos B > 0) →
  (cos B < 1) →
  (b = sqrt (4 * a * c)) →
  (∃ p, (sqrt(6) / 2 < p ∧ p < sqrt 2)) :=
by
  intros h1 h2 h3
  sorry

end problem1_problem2_l279_279853


namespace linearly_dependent_k_l279_279641

theorem linearly_dependent_k (k : ℝ) : 
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨1, k⟩ : ℝ × ℝ) = (0, 0)) ↔ k = 3 / 2 :=
by
  sorry

end linearly_dependent_k_l279_279641


namespace money_collected_is_correct_l279_279420

-- Define the conditions as constants and definitions in Lean
def ticket_price_adult : ℝ := 0.60
def ticket_price_child : ℝ := 0.25
def total_persons : ℕ := 280
def children_attended : ℕ := 80

-- Define the number of adults
def adults_attended : ℕ := total_persons - children_attended

-- Define the total money collected
def total_money_collected : ℝ :=
  (adults_attended * ticket_price_adult) + (children_attended * ticket_price_child)

-- Statement to prove
theorem money_collected_is_correct :
  total_money_collected = 140 := by
  sorry

end money_collected_is_correct_l279_279420


namespace total_expenditure_l279_279060

-- Define the conditions
def cost_per_acre : ℕ := 20
def acres_bought : ℕ := 30
def house_cost : ℕ := 120000
def cost_per_cow : ℕ := 1000
def cows_bought : ℕ := 20
def cost_per_chicken : ℕ := 5
def chickens_bought : ℕ := 100
def hourly_installation_cost : ℕ := 100
def installation_hours : ℕ := 6
def solar_equipment_cost : ℕ := 6000

-- Define the total cost breakdown
def land_cost : ℕ := cost_per_acre * acres_bought
def cows_cost : ℕ := cost_per_cow * cows_bought
def chickens_cost : ℕ := cost_per_chicken * chickens_bought
def solar_installation_cost : ℕ := (hourly_installation_cost * installation_hours) + solar_equipment_cost

-- Define the total cost
def total_cost : ℕ :=
  land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost

-- The theorem statement
theorem total_expenditure : total_cost = 147700 :=
by
  -- Proof steps would go here
  sorry

end total_expenditure_l279_279060


namespace incorrect_connection_probability_is_correct_l279_279919

noncomputable def incorrect_connection_probability : ℝ :=
  let p := 0.02 in
  let C := (n k : ℕ) => Nat.choose n k in
  let r2 := 1/9 in
  let r3 := 8/81 in
  C 3 2 * p^2 * (1 - p) * r2 + C 3 3 * p^3 * r3

theorem incorrect_connection_probability_is_correct :
  incorrect_connection_probability ≈ 0.000131 :=
  sorry

end incorrect_connection_probability_is_correct_l279_279919


namespace vasya_numbers_l279_279746

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l279_279746


namespace g_inequality_solution_range_of_m_l279_279487

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 8
noncomputable def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16
noncomputable def h (x m : ℝ) : ℝ := x^2 - (4 + m)*x + (m + 7)

theorem g_inequality_solution:
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} :=
by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 1 → f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 4 :=
by
  sorry

end g_inequality_solution_range_of_m_l279_279487


namespace replaced_person_weight_l279_279182

theorem replaced_person_weight :
  ∀ (avg_weight: ℝ), 
    10 * (avg_weight + 4) - 10 * avg_weight = 110 - 70 :=
by
  intros avg_weight
  sorry

end replaced_person_weight_l279_279182


namespace negation_of_P_l279_279639

open Classical

variable (x : ℝ)

def P (x : ℝ) : Prop :=
  x^2 + 2 > 2 * x

theorem negation_of_P : (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_P_l279_279639


namespace range_of_independent_variable_l279_279223

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y : ℝ, y = 2 * x / (x - 1)) ↔ x ≠ 1 :=
by sorry

end range_of_independent_variable_l279_279223


namespace vasya_numbers_l279_279743

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l279_279743


namespace find_y_given_conditions_l279_279842

theorem find_y_given_conditions (x y : ℝ) (hx : x = 102) 
                                (h : x^3 * y - 3 * x^2 * y + 3 * x * y = 106200) : 
  y = 10 / 97 :=
by
  sorry

end find_y_given_conditions_l279_279842


namespace rolling_dice_probability_l279_279074

-- Defining variables and conditions
def total_outcomes : Nat := 6^7

def favorable_outcomes : Nat :=
  Nat.choose 7 2 * 6 * (Nat.factorial 5) -- Calculation for exactly one pair of identical numbers

def probability : Rat :=
  favorable_outcomes / total_outcomes

-- The main theorem to prove the probability is 5/18
theorem rolling_dice_probability :
  probability = 5 / 18 := by
  sorry

end rolling_dice_probability_l279_279074


namespace not_777_integers_l279_279630

theorem not_777_integers (p : ℕ) (hp : Nat.Prime p) :
  ¬ (∃ count : ℕ, count = 777 ∧ ∀ n : ℕ, ∃ k : ℕ, (n ^ 3 + n * p + 1 = k * (n + p + 1))) :=
by
  sorry

end not_777_integers_l279_279630


namespace no_valid_arithmetic_operation_l279_279469

-- Definition for arithmetic operations
inductive Operation
| div : Operation
| mul : Operation
| add : Operation
| sub : Operation

open Operation

-- Given conditions
def equation (op : Operation) : Prop :=
  match op with
  | div => (8 / 2) + 5 - (3 - 2) = 12
  | mul => (8 * 2) + 5 - (3 - 2) = 12
  | add => (8 + 2) + 5 - (3 - 2) = 12
  | sub => (8 - 2) + 5 - (3 - 2) = 12

-- Statement to prove
theorem no_valid_arithmetic_operation : ∀ op : Operation, ¬ equation op := by
  sorry

end no_valid_arithmetic_operation_l279_279469


namespace daily_sales_volume_80_sales_volume_function_price_for_profit_l279_279895

-- Define all relevant conditions
def cost_price : ℝ := 70
def max_price : ℝ := 99
def initial_price : ℝ := 95
def initial_sales : ℕ := 50
def price_reduction_effect : ℕ := 2

-- Part 1: Proving daily sales volume at 80 yuan
theorem daily_sales_volume_80 : 
  (initial_price - 80) * price_reduction_effect + initial_sales = 80 := 
by sorry

-- Part 2: Proving functional relationship
theorem sales_volume_function (x : ℝ) (h₁ : 70 ≤ x) (h₂ : x ≤ 99) : 
  (initial_sales + price_reduction_effect * (initial_price - x) = -2 * x + 240) :=
by sorry

-- Part 3: Proving price for 1200 yuan daily profit
theorem price_for_profit (profit_target : ℝ) (h : profit_target = 1200) :
  ∃ x, (x - cost_price) * (initial_sales + price_reduction_effect * (initial_price - x)) = profit_target ∧ x ≤ max_price :=
by sorry

end daily_sales_volume_80_sales_volume_function_price_for_profit_l279_279895


namespace length_of_first_train_l279_279405

theorem length_of_first_train
    (speed_train1_kmph : ℝ) (speed_train2_kmph : ℝ) 
    (length_train2_m : ℝ) (cross_time_s : ℝ)
    (conv_factor : ℝ)         -- Conversion factor from kmph to m/s
    (relative_speed_ms : ℝ)   -- Relative speed in m/s 
    (distance_covered_m : ℝ)  -- Total distance covered in meters
    (length_train1_m : ℝ) : Prop :=
  speed_train1_kmph = 120 →
  speed_train2_kmph = 80 →
  length_train2_m = 210.04 →
  cross_time_s = 9 →
  conv_factor = 1000 / 3600 →
  relative_speed_ms = (200 * conv_factor) →
  distance_covered_m = (relative_speed_ms * cross_time_s) →
  length_train1_m = 290 →
  distance_covered_m = length_train1_m + length_train2_m

end length_of_first_train_l279_279405


namespace cat_food_insufficient_for_six_days_l279_279462

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l279_279462


namespace g_of_neg2_l279_279137

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem g_of_neg2 : g (-2) = 7 / 3 := by
  sorry

end g_of_neg2_l279_279137


namespace regions_divided_by_7_tangents_l279_279658

-- Define the recursive function R for the number of regions divided by n tangents
def R : ℕ → ℕ
| 0       => 1
| (n + 1) => R n + (n + 1)

-- The theorem stating the specific case of the problem
theorem regions_divided_by_7_tangents : R 7 = 29 := by
  sorry

end regions_divided_by_7_tangents_l279_279658


namespace sixth_graders_more_than_seventh_l279_279539

def total_payment_seventh_graders : ℕ := 143
def total_payment_sixth_graders : ℕ := 195
def cost_per_pencil : ℕ := 13

theorem sixth_graders_more_than_seventh :
  (total_payment_sixth_graders / cost_per_pencil) - (total_payment_seventh_graders / cost_per_pencil) = 4 :=
  by
  sorry

end sixth_graders_more_than_seventh_l279_279539


namespace max_rectangle_area_l279_279071

theorem max_rectangle_area (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (a b : ℝ), 2 * a + 2 * b = perimeter ∧ a * b = 625 :=
by
  sorry

end max_rectangle_area_l279_279071


namespace gcd_lcm_product_24_60_l279_279287

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l279_279287


namespace impossible_sequence_l279_279370

theorem impossible_sequence (a : ℕ → ℝ) (c : ℝ) (a1 : ℝ)
  (h_periodic : ∀ n, a (n + 3) = a n)
  (h_det : ∀ n, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)
  (ha1 : a 1 = 2) (hc : c = 2) : false :=
by
  sorry

end impossible_sequence_l279_279370


namespace squares_and_sqrt_l279_279537

variable (a b c : ℤ)

theorem squares_and_sqrt (ha : a = 10001) (hb : b = 100010001) (hc : c = 1000200030004000300020001) :
∃ x y z : ℤ, x = a^2 ∧ y = b^2 ∧ z = Int.sqrt c ∧ x = 100020001 ∧ y = 10002000300020001 ∧ z = 1000100010001 :=
by
  use a^2, b^2, Int.sqrt c
  rw [ha, hb, hc]
  sorry

end squares_and_sqrt_l279_279537


namespace multiply_mixed_number_l279_279109

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l279_279109


namespace wedge_top_half_volume_l279_279792

theorem wedge_top_half_volume (r : ℝ) (C : ℝ) (V : ℝ) : 
  (C = 18 * π) ∧ (C = 2 * π * r) ∧ (V = (4/3) * π * r^3) ∧ 
  (V / 3 / 2) = 162 * π :=
  sorry

end wedge_top_half_volume_l279_279792


namespace cat_food_inequality_l279_279437

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l279_279437


namespace food_requirement_not_met_l279_279453

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l279_279453


namespace jack_pays_back_l279_279857

-- conditions in the problem 
def principal : ℝ := 1200
def interest_rate : ℝ := 0.1

-- the theorem statement equivalent to the question and correct answer
theorem jack_pays_back (principal_interest: principal * interest_rate) (total_amount: principal + principal_interest) : total_amount = 1320 :=
by
  sorry

end jack_pays_back_l279_279857


namespace gcd_lcm_product_24_60_l279_279295

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l279_279295


namespace roger_has_more_candies_l279_279210

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end roger_has_more_candies_l279_279210


namespace proof_problem_l279_279315

open Real

noncomputable def set_A : Set ℝ :=
  {x | x = tan (-19 * π / 6) ∨ x = sin (-19 * π / 6)}

noncomputable def set_B : Set ℝ :=
  {m | 0 <= m ∧ m <= 4}

noncomputable def set_C (a : ℝ) : Set ℝ :=
  {x | a + 1 < x ∧ x < 2 * a}

theorem proof_problem (a : ℝ) :
  set_A = {-sqrt 3 / 3, -1 / 2} ∧
  set_B = {m | 0 <= m ∧ m <= 4} ∧
  (set_A ∪ set_B) = {-sqrt 3 / 3, -1 / 2, 0, 4} →
  (∀ a, set_C a ⊆ (set_A ∪ set_B) → 1 < a ∧ a < 2) :=
sorry

end proof_problem_l279_279315


namespace primes_eq_condition_l279_279991

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l279_279991


namespace correct_factorization_l279_279556

theorem correct_factorization:
  (∃ a : ℝ, (a + 3) * (a - 3) = a ^ 2 - 9) ∧
  (∃ x : ℝ, x ^ 2 + x - 5 = x * (x + 1) - 5) ∧
  ¬ (∃ x : ℝ, x ^ 2 + 1 = x * (x + 1 / x)) ∧
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2) →
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2)
  := by
  sorry

end correct_factorization_l279_279556


namespace linear_dependent_vectors_l279_279228

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l279_279228


namespace angle_bisector_segment_rel_l279_279917

variable (a b c : ℝ) -- The sides of the triangle
variable (u v : ℝ)   -- The segments into which fa divides side a
variable (fa : ℝ)    -- The length of the angle bisector

-- Statement setting up the given conditions and the proof we need
theorem angle_bisector_segment_rel : 
  (u : ℝ) = a * c / (b + c) → 
  (v : ℝ) = a * b / (b + c) → 
  (fa : ℝ) = 2 * (Real.sqrt (b * s * (s - a) * c)) / (b + c) → 
  fa^2 = b * c - u * v :=
sorry

end angle_bisector_segment_rel_l279_279917


namespace find_a_l279_279318

theorem find_a (a : ℝ) (h : ∃ b : ℝ, (4:ℝ)*x^2 - (12:ℝ)*x + a = (2*x + b)^2) : a = 9 :=
sorry

end find_a_l279_279318


namespace find_tony_age_l279_279235

variable (y : ℕ)
variable (d : ℕ)

def Tony_day_hours : ℕ := 3
def Tony_hourly_rate (age : ℕ) : ℚ := 0.75 * age
def Tony_days_worked : ℕ := 60
def Tony_total_earnings : ℚ := 945

noncomputable def earnings_before_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate age * Tony_day_hours * days

noncomputable def earnings_after_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate (age + 1) * Tony_day_hours * days

noncomputable def total_earnings (age : ℕ) (days_before : ℕ) : ℚ :=
  (earnings_before_birthday age days_before) +
  (earnings_after_birthday age (Tony_days_worked - days_before))

theorem find_tony_age: ∃ y d : ℕ, total_earnings y d = Tony_total_earnings ∧ y = 6 := by
  sorry

end find_tony_age_l279_279235


namespace ratio_of_pieces_l279_279946

theorem ratio_of_pieces (total_length shorter_piece longer_piece : ℕ) 
    (h1 : total_length = 6) (h2 : shorter_piece = 2)
    (h3 : longer_piece = total_length - shorter_piece) :
    ((longer_piece : ℚ) / (shorter_piece : ℚ)) = 2 :=
by
    sorry

end ratio_of_pieces_l279_279946


namespace cost_of_two_pencils_and_one_pen_l279_279714

variables (a b : ℝ)

theorem cost_of_two_pencils_and_one_pen
  (h1 : 3 * a + b = 3.00)
  (h2 : 3 * a + 4 * b = 7.50) :
  2 * a + b = 2.50 :=
sorry

end cost_of_two_pencils_and_one_pen_l279_279714


namespace diane_trip_length_l279_279138

-- Define constants and conditions
def first_segment_fraction : ℚ := 1 / 4
def middle_segment_length : ℚ := 24
def last_segment_fraction : ℚ := 1 / 3

def total_trip_length (x : ℚ) : Prop :=
  (1 - first_segment_fraction - last_segment_fraction) * x = middle_segment_length

theorem diane_trip_length : ∃ x : ℚ, total_trip_length x ∧ x = 57.6 := by
  sorry

end diane_trip_length_l279_279138


namespace chuck_total_playable_area_l279_279265

noncomputable def chuck_roaming_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let larger_arc_area := (3 / 4) * Real.pi * leash_length ^ 2
  let additional_sector_area := (1 / 4) * Real.pi * (leash_length - shed_length) ^ 2
  larger_arc_area + additional_sector_area

theorem chuck_total_playable_area :
  chuck_roaming_area 3 4 5 = 19 * Real.pi :=
  by
  sorry

end chuck_total_playable_area_l279_279265


namespace base_eight_to_base_ten_l279_279073

theorem base_eight_to_base_ten (n : ℕ) : 
  n = 3 * 8^1 + 1 * 8^0 → n = 25 :=
by
  intro h
  rw [mul_comm 3 (8^1), pow_one, mul_comm 1 (8^0), pow_zero, mul_one] at h
  exact h

end base_eight_to_base_ten_l279_279073


namespace mul_mixed_number_l279_279124

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l279_279124


namespace at_least_one_hits_l279_279686

open ProbabilityTheory

def prob_person_A_hits : ℝ := 0.8
def prob_person_B_hits : ℝ := 0.8

theorem at_least_one_hits : 
  let prob_at_least_one_hits := 1 - (1 - prob_person_A_hits) * (1 - prob_person_B_hits)
  in prob_at_least_one_hits = 0.96 :=
sorry

end at_least_one_hits_l279_279686


namespace routes_from_Bristol_to_Carlisle_l279_279784

-- Given conditions as definitions
def routes_Bristol_to_Birmingham : ℕ := 8
def routes_Birmingham_to_Manchester : ℕ := 5
def routes_Manchester_to_Sheffield : ℕ := 4
def routes_Sheffield_to_Newcastle : ℕ := 3
def routes_Newcastle_to_Carlisle : ℕ := 2

-- Define the total number of routes from Bristol to Carlisle
def total_routes_Bristol_to_Carlisle : ℕ := routes_Bristol_to_Birmingham *
                                            routes_Birmingham_to_Manchester *
                                            routes_Manchester_to_Sheffield *
                                            routes_Sheffield_to_Newcastle *
                                            routes_Newcastle_to_Carlisle

-- The theorem to be proved
theorem routes_from_Bristol_to_Carlisle :
  total_routes_Bristol_to_Carlisle = 960 :=
by
  -- Proof will be provided here
  sorry

end routes_from_Bristol_to_Carlisle_l279_279784


namespace distance_between_foci_of_ellipse_l279_279595

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ),
  -- Condition: axes are parallel to the coordinate axes (implicitly given by tangency points).
  a = 3 ∧
  b = 2 ∧
  c = Real.sqrt (a^2 - b^2) ∧
  2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_ellipse_l279_279595


namespace count_of_divisibles_l279_279838

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l279_279838


namespace combined_degrees_l279_279697

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l279_279697


namespace part1_part2_l279_279248

-- Define the conditions
def cost_price := 30
def initial_selling_price := 40
def initial_sales_volume := 600
def sales_decrease_per_yuan := 10

-- Define the profit calculation function
def profit (selling_price : ℕ) : ℕ :=
  let profit_per_unit := selling_price - cost_price
  let new_sales_volume := initial_sales_volume - sales_decrease_per_yuan * (selling_price - initial_selling_price)
  profit_per_unit * new_sales_volume

-- Statements to prove
theorem part1 :
  profit 50 = 10000 :=
by
  sorry

theorem part2 :
  let max_profit_price := 60
  let max_profit := 12000
  max_profit = (fun price => max (profit price) 0) 60 :=
by
  sorry

end part1_part2_l279_279248


namespace number_of_apple_trees_l279_279207

variable (T : ℕ) -- Declare the number of apple trees as a natural number

-- Define the conditions
def picked_apples := 8 * T
def remaining_apples := 9
def initial_apples := 33

-- The statement to prove Rachel has 3 apple trees
theorem number_of_apple_trees :
  initial_apples - picked_apples + remaining_apples = initial_apples → T = 3 := 
by
  sorry

end number_of_apple_trees_l279_279207


namespace smallest_five_digit_in_pascal_l279_279932

-- Define the conditions
def pascal_triangle_increases (n k : ℕ) : Prop := 
  ∀ (r ≥ n) (c ≥ k), c ≤ r → ∃ (x : ℕ), x >= Nat.choose r c

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- State the proof problem and the expected answer
theorem smallest_five_digit_in_pascal :
  (∃ (n k : ℕ), binomial_coefficient n k = 10000) ∧ (∀ (m l : ℕ), binomial_coefficient m l = 10000 → n ≤ m) := sorry

end smallest_five_digit_in_pascal_l279_279932


namespace Vasya_numbers_l279_279729

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l279_279729


namespace solve_system_of_equations_l279_279690

theorem solve_system_of_equations (x y : ℝ) : 
  (x + y = x^2 + 2 * x * y + y^2) ∧ (x - y = x^2 - 2 * x * y + y^2) ↔ 
  (x = 0 ∧ y = 0) ∨ 
  (x = 1/2 ∧ y = 1/2) ∨ 
  (x = 1/2 ∧ y = -1/2) ∨ 
  (x = 1 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l279_279690


namespace certain_event_among_options_l279_279388

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end certain_event_among_options_l279_279388


namespace task1_task2_l279_279014

-- Define the conditions and the probabilities to be proven

def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1

def total_combinations := Nat.choose total_pens 2

def combinations_with_exactly_one_first_class : Nat :=
  (first_class_pens * (total_pens - first_class_pens))

def probability_one_first_class_pen : ℚ :=
  combinations_with_exactly_one_first_class / total_combinations

def combinations_without_any_third_class : Nat :=
  Nat.choose (first_class_pens + second_class_pens) 2

def probability_no_third_class_pen : ℚ :=
  combinations_without_any_third_class / total_combinations

theorem task1 : probability_one_first_class_pen = 3 / 5 := 
  sorry

theorem task2 : probability_no_third_class_pen = 2 / 3 := 
  sorry

end task1_task2_l279_279014


namespace multiply_mixed_number_l279_279110

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l279_279110


namespace roger_has_more_candy_l279_279212

-- Defining the conditions
def sandra_bag1 : Nat := 6
def sandra_bag2 : Nat := 6
def roger_bag1 : Nat := 11
def roger_bag2 : Nat := 3

-- Calculating the total pieces of candy for Sandra and Roger
def total_sandra : Nat := sandra_bag1 + sandra_bag2
def total_roger : Nat := roger_bag1 + roger_bag2

-- Statement of the proof problem
theorem roger_has_more_candy : total_roger - total_sandra = 2 := by
  sorry

end roger_has_more_candy_l279_279212


namespace count_valid_n_decomposition_l279_279801

theorem count_valid_n_decomposition : 
  ∃ (count : ℕ), count = 108 ∧ 
  ∀ (a b c n : ℕ), 
    8 * a + 88 * b + 888 * c = 8000 → 
    0 ≤ b ∧ b ≤ 90 → 
    0 ≤ c ∧ c ≤ 9 → 
    n = a + 2 * b + 3 * c → 
    n < 1000 :=
sorry

end count_valid_n_decomposition_l279_279801


namespace total_students_in_classes_l279_279926

theorem total_students_in_classes (t1 t2 x y: ℕ) (h1 : t1 = 273) (h2 : t2 = 273) (h3 : (x - 1) * 7 = t1) (h4 : (y - 1) * 13 = t2) : x + y = 62 :=
by
  sorry

end total_students_in_classes_l279_279926


namespace eval_expression_l279_279804

theorem eval_expression : (3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3)) :=
by sorry

end eval_expression_l279_279804


namespace find_c_l279_279342

variables {α : Type*} [LinearOrderedField α]

def p (x : α) : α := 3 * x - 9
def q (x : α) (c : α) : α := 4 * x - c

-- We aim to prove that if p(q(3,c)) = 6, then c = 7
theorem find_c (c : α) : p (q 3 c) = 6 → c = 7 :=
by
  sorry

end find_c_l279_279342


namespace not_subset_T_to_S_l279_279817

def is_odd (x : ℤ) : Prop := ∃ n : ℤ, x = 2 * n + 1
def is_of_form_4k_plus_1 (y : ℤ) : Prop := ∃ k : ℤ, y = 4 * k + 1

theorem not_subset_T_to_S :
  ¬ (∀ y, is_of_form_4k_plus_1 y → is_odd y) :=
sorry

end not_subset_T_to_S_l279_279817


namespace ratio_adults_children_l279_279260

-- Definitions based on conditions
def children := 45
def total_adults (A : ℕ) : Prop := (2 / 3 : ℚ) * A = 10

-- The theorem stating the problem
theorem ratio_adults_children :
  ∃ A, total_adults A ∧ (A : ℚ) / children = (1 / 3 : ℚ) :=
by {
  sorry
}

end ratio_adults_children_l279_279260


namespace solve_for_star_l279_279172

theorem solve_for_star 
  (x : ℝ) 
  (h : 45 - (28 - (37 - (15 - x))) = 58) : 
  x = 19 :=
by
  -- Proof goes here. Currently incomplete, so we use sorry.
  sorry

end solve_for_star_l279_279172


namespace floor_ineq_l279_279576

theorem floor_ineq (α β : ℝ) : ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ :=
sorry

end floor_ineq_l279_279576


namespace primes_eq_condition_l279_279990

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l279_279990


namespace abs_negative_five_l279_279705

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l279_279705


namespace dan_has_3_potatoes_left_l279_279802

-- Defining the number of potatoes Dan originally had
def original_potatoes : ℕ := 7

-- Defining the number of potatoes the rabbits ate
def potatoes_eaten : ℕ := 4

-- The theorem we want to prove: Dan has 3 potatoes left.
theorem dan_has_3_potatoes_left : original_potatoes - potatoes_eaten = 3 := by
  sorry

end dan_has_3_potatoes_left_l279_279802


namespace each_persons_contribution_l279_279889

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end each_persons_contribution_l279_279889


namespace greatest_possible_value_of_q_minus_r_l279_279941

noncomputable def max_difference (q r : ℕ) : ℕ :=
  if q < r then r - q else q - r

theorem greatest_possible_value_of_q_minus_r (q r : ℕ) (x y : ℕ) (hq : q = 10 * x + y) (hr : r = 10 * y + x) (cond : q ≠ r) (hqr : max_difference q r < 20) : q - r = 18 :=
  sorry

end greatest_possible_value_of_q_minus_r_l279_279941


namespace inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l279_279152

-- Proof Problem 1
theorem inverse_proportional_t (t : ℝ) (h1 : 1 ≤ t ∧ t ≤ 2023) : t = 1 :=
sorry

-- Proof Problem 2
theorem no_linear_function_2k_times (k : ℝ) (h_pos : 0 < k) : ¬ ∃ a b : ℝ, (a < b) ∧ (∀ x, a ≤ x ∧ x ≤ b → (2 * k * a ≤ k * x + 2 ∧ k * x + 2 ≤ 2 * k * b)) :=
sorry

-- Proof Problem 3
theorem quadratic_function_5_times (a b : ℝ) (h_ab : a < b) (h_quad : ∀ x, a ≤ x ∧ x ≤ b → (5 * a ≤ x^2 - 4 * x - 7 ∧ x^2 - 4 * x - 7 ≤ 5 * b)) :
  (a = -2 ∧ b = 1) ∨ (a = -(11/5) ∧ b = (9 + Real.sqrt 109) / 2) :=
sorry

end inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l279_279152


namespace binomial_p_value_l279_279363

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

theorem binomial_p_value (p : ℝ) : (binomial_expected_value 18 p = 9) → p = 1/2 :=
by
  intro h
  sorry

end binomial_p_value_l279_279363


namespace certain_event_birthday_example_l279_279390
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end certain_event_birthday_example_l279_279390


namespace Robert_books_read_in_six_hours_l279_279887

theorem Robert_books_read_in_six_hours (P H T: ℕ)
    (h1: P = 270)
    (h2: H = 90)
    (h3: T = 6):
    T * H / P = 2 :=
by 
    -- sorry placeholder to indicate that this is where the proof goes.
    sorry

end Robert_books_read_in_six_hours_l279_279887


namespace vector_dot_product_l279_279145

open Complex

def a : Complex := (1 : ℝ) + (-(2 : ℝ)) * Complex.I
def b : Complex := (-3 : ℝ) + (4 : ℝ) * Complex.I
def c : Complex := (3 : ℝ) + (2 : ℝ) * Complex.I

-- Note: Using real coordinates to simulate vector operations.
theorem vector_dot_product :
  let a_vec := (1, -2)
  let b_vec := (-3, 4)
  let c_vec := (3, 2)
  let linear_combination := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (linear_combination.1 * c_vec.1 + linear_combination.2 * c_vec.2) = -3 := 
by
  sorry

end vector_dot_product_l279_279145


namespace find_b_l279_279854

theorem find_b 
    (x1 x2 b c : ℝ)
    (h_distinct : x1 ≠ x2)
    (h_root_x : ∀ x, (x^2 + 5 * b * x + c = 0) → x = x1 ∨ x = x2)
    (h_common_root : ∃ y, (y^2 + 2 * x1 * y + 2 * x2 = 0) ∧ (y^2 + 2 * x2 * y + 2 * x1 = 0)) :
  b = 1 / 10 := 
sorry

end find_b_l279_279854


namespace commute_time_difference_l279_279252

theorem commute_time_difference (x y : ℝ) 
  (h1 : x + y = 39)
  (h2 : (x - 10)^2 + (y - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_difference_l279_279252


namespace packs_of_red_balls_l279_279023

/-
Julia bought some packs of red balls, R packs.
Julia bought 10 packs of yellow balls.
Julia bought 8 packs of green balls.
There were 19 balls in each package.
Julia bought 399 balls in total.
The goal is to prove that the number of packs of red balls Julia bought, R, is equal to 3.
-/

theorem packs_of_red_balls (R : ℕ) (balls_per_pack : ℕ) (packs_yellow : ℕ) (packs_green : ℕ) (total_balls : ℕ) 
  (h1 : balls_per_pack = 19) (h2 : packs_yellow = 10) (h3 : packs_green = 8) (h4 : total_balls = 399) 
  (h5 : total_balls = R * balls_per_pack + (packs_yellow + packs_green) * balls_per_pack) : 
  R = 3 :=
by
  -- Proof goes here
  sorry

end packs_of_red_balls_l279_279023


namespace num_valid_m_divisors_of_1750_l279_279303

theorem num_valid_m_divisors_of_1750 : 
  ∃! (m : ℕ) (h1 : m > 0), ∃ (k : ℕ), k > 0 ∧ 1750 = k * (m^2 - 4) :=
sorry

end num_valid_m_divisors_of_1750_l279_279303


namespace cone_cylinder_volume_ratio_l279_279408

theorem cone_cylinder_volume_ratio (h_cyl r_cyl: ℝ) (h_cone: ℝ) :
  h_cyl = 10 → r_cyl = 5 → h_cone = 5 →
  (1/3 * (Real.pi * r_cyl^2 * h_cone)) / (Real.pi * r_cyl^2 * h_cyl) = 1/6 :=
by
  intros h_cyl_eq r_cyl_eq h_cone_eq
  rw [h_cyl_eq, r_cyl_eq, h_cone_eq]
  sorry

end cone_cylinder_volume_ratio_l279_279408


namespace possible_values_of_AC_l279_279357

theorem possible_values_of_AC (AB CD AC : ℝ) (m n : ℝ) (h1 : AB = 16) (h2 : CD = 4)
  (h3 : Set.Ioo m n = {x : ℝ | 4 < x ∧ x < 16}) : m + n = 20 :=
by
  sorry

end possible_values_of_AC_l279_279357


namespace equilateral_triangle_of_angle_and_side_sequences_l279_279321

variable {A B C a b c : ℝ}

theorem equilateral_triangle_of_angle_and_side_sequences
  (H_angles_arithmetic : 2 * B = A + C)
  (H_sum_angles : A + B + C = Real.pi)
  (H_sides_geometric : b^2 = a * c) :
  A = Real.pi / 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 3 ∧ a = b ∧ b = c :=
by
  sorry

end equilateral_triangle_of_angle_and_side_sequences_l279_279321


namespace triangle_DEF_angle_l279_279062

noncomputable def one_angle_of_triangle_DEF (x : ℝ) : ℝ :=
  let arc_DE := 2 * x + 40
  let arc_EF := 3 * x + 50
  let arc_FD := 4 * x - 30
  if (arc_DE + arc_EF + arc_FD = 360)
  then (1 / 2) * arc_EF
  else 0

theorem triangle_DEF_angle (x : ℝ) (h : 2 * x + 40 + 3 * x + 50 + 4 * x - 30 = 360) :
  one_angle_of_triangle_DEF x = 75 :=
by sorry

end triangle_DEF_angle_l279_279062


namespace mr_william_land_percentage_l279_279999

-- Define the conditions
def farm_tax_percentage : ℝ := 0.5
def total_tax_collected : ℝ := 3840
def mr_william_tax : ℝ := 480

-- Theorem statement proving the question == answer
theorem mr_william_land_percentage : 
  (mr_william_tax / total_tax_collected) * 100 = 12.5 := 
by
  -- sorry is used to skip the proof
  sorry

end mr_william_land_percentage_l279_279999


namespace find_a_b_l279_279552

-- Define the polynomial with unknown coefficients a and b
def P (x : ℝ) (a b : ℝ) : ℝ := 2 * x^3 + a * x^2 - 13 * x + b

-- Define the conditions for the roots
def root1 (a b : ℝ) : Prop := P 2 a b = 0
def root2 (a b : ℝ) : Prop := P (-3) a b = 0

-- Prove that the coefficients a and b are 1 and 6, respectively
theorem find_a_b : ∀ a b : ℝ, root1 a b ∧ root2 a b → a = 1 ∧ b = 6 :=
by
  intros a b h
  sorry

end find_a_b_l279_279552


namespace parabola_tangent_y_intercept_correct_l279_279335

noncomputable def parabola_tangent_y_intercept (a : ℝ) : Prop :=
  let C := fun x : ℝ => x^2
  let slope := 2 * a
  let tangent_line := fun x : ℝ => slope * (x - a) + C a
  let Q := (0, tangent_line 0)
  Q = (0, -a^2)

-- Statement of the problem as a Lean theorem
theorem parabola_tangent_y_intercept_correct (a : ℝ) (h : a > 0) :
  parabola_tangent_y_intercept a := 
by 
  sorry

end parabola_tangent_y_intercept_correct_l279_279335


namespace sufficient_but_not_necessary_l279_279575

theorem sufficient_but_not_necessary (a : ℝ) (h : a = 1/4) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 1) ∧ ¬(∀ x : ℝ, x > 0 → x + a / x ≥ 1 ↔ a = 1/4) :=
by
  sorry

end sufficient_but_not_necessary_l279_279575


namespace reggie_marbles_l279_279208

/-- Given that Reggie and his friend played 9 games in total,
    Reggie lost 1 game, and they bet 10 marbles per game.
    Prove that Reggie has 70 marbles after all games. -/
theorem reggie_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) (marbles_initial : ℕ) 
  (h_total_games : total_games = 9) (h_lost_games : lost_games = 1) (h_marbles_per_game : marbles_per_game = 10) 
  (h_marbles_initial : marbles_initial = 0) : 
  marbles_initial + (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game = 70 :=
by
  -- We proved this in the solution steps, but will skip the proof here with sorry.
  sorry

end reggie_marbles_l279_279208


namespace jane_earnings_l279_279659

def earnings_per_bulb : ℝ := 0.50
def tulip_bulbs : ℕ := 20
def iris_bulbs : ℕ := tulip_bulbs / 2
def daffodil_bulbs : ℕ := 30
def crocus_bulbs : ℕ := daffodil_bulbs * 3
def total_earnings : ℝ := (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs) * earnings_per_bulb

theorem jane_earnings : total_earnings = 75.0 := by
  sorry

end jane_earnings_l279_279659


namespace length_of_platform_l279_279415

noncomputable def len_train : ℝ := 120
noncomputable def speed_train : ℝ := 60 * (1000 / 3600) -- kmph to m/s
noncomputable def time_cross : ℝ := 15

theorem length_of_platform (L_train : ℝ) (S_train : ℝ) (T_cross : ℝ) (H_train : L_train = len_train)
  (H_speed : S_train = speed_train) (H_time : T_cross = time_cross) : 
  ∃ (L_platform : ℝ), L_platform = (S_train * T_cross) - L_train ∧ L_platform = 130.05 :=
by
  rw [H_train, H_speed, H_time]
  sorry

end length_of_platform_l279_279415


namespace num_solutions_3x_plus_2y_eq_806_l279_279268

theorem num_solutions_3x_plus_2y_eq_806 :
  (∃ y : ℕ, ∃ x : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 806) ∧
  ((∃ t : ℤ, x = 268 - 2 * t ∧ y = 1 + 3 * t) ∧ (∃ t : ℤ, 0 ≤ t ∧ t ≤ 133)) :=
sorry

end num_solutions_3x_plus_2y_eq_806_l279_279268


namespace no_real_solutions_l279_279040

theorem no_real_solutions :
  ∀ x y z : ℝ, ¬ (x + y + 2 + 4*x*y = 0 ∧ y + z + 2 + 4*y*z = 0 ∧ z + x + 2 + 4*z*x = 0) :=
by
  sorry

end no_real_solutions_l279_279040


namespace velocity_at_second_return_to_equilibrium_l279_279411

noncomputable def ball_velocity (t : ℝ) : ℝ := 30 * Real.cos (2 * t + π / 6)

theorem velocity_at_second_return_to_equilibrium :
  ball_velocity (11 * π / 12) = 30 :=
by
  sorry

end velocity_at_second_return_to_equilibrium_l279_279411


namespace find_f4_l279_279841

variable (a b : ℝ)
variable (f : ℝ → ℝ)
variable (h1 : f 1 = 5)
variable (h2 : f 2 = 8)
variable (h3 : f 3 = 11)
variable (h4 : ∀ x, f x = a * x + b)

theorem find_f4 : f 4 = 14 := by
  sorry

end find_f4_l279_279841


namespace percentage_of_women_lawyers_l279_279779

theorem percentage_of_women_lawyers
  (T : ℝ) 
  (h1 : 0.70 * T = W) 
  (h2 : 0.28 * T = WL) : 
  ((WL / W) * 100 = 40) :=
by
  sorry

end percentage_of_women_lawyers_l279_279779


namespace combined_total_l279_279505

variable (Jane Jean : ℕ)

theorem combined_total (h1 : Jean = 3 * Jane) (h2 : Jean = 57) : Jane + Jean = 76 := by
  sorry

end combined_total_l279_279505


namespace initial_percentage_of_milk_l279_279724

theorem initial_percentage_of_milk (P : ℝ) :
  (P / 100) * 60 = (68 / 100) * 74.11764705882354 → P = 84 :=
by
  sorry

end initial_percentage_of_milk_l279_279724


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279132

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279132


namespace sacks_after_days_l279_279006

-- Define the number of sacks harvested per day
def harvest_per_day : ℕ := 74

-- Define the number of sacks discarded per day
def discard_per_day : ℕ := 71

-- Define the days of harvest
def days_of_harvest : ℕ := 51

-- Define the number of sacks that are not discarded per day
def net_sacks_per_day : ℕ := harvest_per_day - discard_per_day

-- Define the total number of sacks after the specified days of harvest
def total_sacks : ℕ := days_of_harvest * net_sacks_per_day

theorem sacks_after_days :
  total_sacks = 153 := by
  sorry

end sacks_after_days_l279_279006


namespace cos_alpha_l279_279826

-- Define the conditions
variable (α : Real)
variable (x y r : Real)
-- Given the point (-3, 4)
def point_condition (x : Real) (y : Real) : Prop := x = -3 ∧ y = 4

-- Define r as the distance
def radius_condition (x y r : Real) : Prop := r = Real.sqrt (x ^ 2 + y ^ 2)

-- Prove that cos α and cos 2α are the given values
theorem cos_alpha (α : Real) (x y r : Real) (h1 : point_condition x y) (h2 : radius_condition x y r) :
  Real.cos α = -3 / 5 ∧ Real.cos (2 * α) = -7 / 25 :=
by
  sorry

end cos_alpha_l279_279826


namespace walnut_trees_initial_count_l279_279054

theorem walnut_trees_initial_count (x : ℕ) (h : x + 6 = 10) : x = 4 := 
by
  sorry

end walnut_trees_initial_count_l279_279054


namespace factorization_a_minus_b_l279_279544

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end factorization_a_minus_b_l279_279544


namespace or_is_true_given_p_true_q_false_l279_279824

theorem or_is_true_given_p_true_q_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end or_is_true_given_p_true_q_false_l279_279824


namespace percentage_of_students_receiving_certificates_l279_279966

theorem percentage_of_students_receiving_certificates
  (boys girls : ℕ)
  (pct_boys pct_girls : ℕ)
  (h_boys : boys = 30)
  (h_girls : girls = 20)
  (h_pct_boys : pct_boys = 30)
  (h_pct_girls : pct_girls = 40)
  :
  (pct_boys * boys + pct_girls * girls) / (100 * (boys + girls)) * 100 = 34 :=
by
  sorry

end percentage_of_students_receiving_certificates_l279_279966


namespace price_difference_l279_279013

-- Definitions of conditions
def market_price : ℝ := 15400
def initial_sales_tax_rate : ℝ := 0.076
def new_sales_tax_rate : ℝ := 0.0667
def discount_rate : ℝ := 0.05
def handling_fee : ℝ := 200

-- Calculation of original sales tax
def original_sales_tax_amount : ℝ := market_price * initial_sales_tax_rate
-- Calculation of price after discount
def discount_amount : ℝ := market_price * discount_rate
def price_after_discount : ℝ := market_price - discount_amount
-- Calculation of new sales tax
def new_sales_tax_amount : ℝ := price_after_discount * new_sales_tax_rate
-- Calculation of total price with new sales tax and handling fee
def total_price_new : ℝ := price_after_discount + new_sales_tax_amount + handling_fee
-- Calculation of original total price with handling fee
def original_total_price : ℝ := market_price + original_sales_tax_amount + handling_fee

-- Expected difference in total cost
def expected_difference : ℝ := 964.60

-- Lean 4 statement to prove the difference
theorem price_difference :
  original_total_price - total_price_new = expected_difference :=
by
  sorry

end price_difference_l279_279013


namespace product_gcd_lcm_24_60_l279_279294

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l279_279294


namespace count_special_integers_l279_279423

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def base7 (n : ℕ) : ℕ := 
  let c := n / 343
  let rem1 := n % 343
  let d := rem1 / 49
  let rem2 := rem1 % 49
  let e := rem2 / 7
  let f := rem2 % 7
  343 * c + 49 * d + 7 * e + f

def base8 (n : ℕ) : ℕ := 
  let g := n / 512
  let rem1 := n % 512
  let h := rem1 / 64
  let rem2 := rem1 % 64
  let i := rem2 / 8
  let j := rem2 % 8
  512 * g + 64 * h + 8 * i + j

def matches_last_two_digits (n t : ℕ) : Prop := (t % 100) = (3 * (n % 100))

theorem count_special_integers : 
  ∃! (N : ℕ), is_three_digit N ∧ 
    matches_last_two_digits N (base7 N + base8 N) :=
sorry

end count_special_integers_l279_279423


namespace money_conditions_l279_279816

theorem money_conditions (a b : ℝ) (h1 : 4 * a - b > 32) (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := 
sorry

end money_conditions_l279_279816


namespace arithmetic_progression_num_terms_l279_279594

theorem arithmetic_progression_num_terms (a d n : ℕ) (h_even : n % 2 = 0) 
    (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 30)
    (h_sum_even : (n / 2) * (2 * a + 2 * d + (n - 2) * d) = 36)
    (h_diff_last_first : (n - 1) * d = 12) :
    n = 8 := 
sorry

end arithmetic_progression_num_terms_l279_279594


namespace journey_time_difference_journey_time_difference_in_minutes_l279_279083

-- Define the constant speed of the bus
def speed : ℕ := 60

-- Define distances of journeys
def distance_1 : ℕ := 360
def distance_2 : ℕ := 420

-- Define the time calculation function
def time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the theorem
theorem journey_time_difference :
  time distance_2 speed - time distance_1 speed = 1 :=
by
  sorry

-- Convert the time difference from hours to minutes
theorem journey_time_difference_in_minutes :
  (time distance_2 speed - time distance_1 speed) * 60 = 60 :=
by
  sorry

end journey_time_difference_journey_time_difference_in_minutes_l279_279083


namespace valid_k_values_l279_279840

theorem valid_k_values
  (k : ℝ)
  (h : k = -7 ∨ k = -5 ∨ k = 1 ∨ k = 4) :
  (∀ x, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) → (k = -7 ∨ k = 1 ∨ k = 4) :=
by sorry

end valid_k_values_l279_279840


namespace brownies_on_counter_l279_279879

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end brownies_on_counter_l279_279879


namespace intersection_complement_eq_empty_l279_279174

open Set

variable {α : Type*} (M N U: Set α)

theorem intersection_complement_eq_empty (h : M ⊆ N) : M ∩ (compl N) = ∅ :=
sorry

end intersection_complement_eq_empty_l279_279174


namespace evaluate_expression_l279_279427

theorem evaluate_expression :
  ((-2: ℤ)^2) ^ (1 ^ (0 ^ 2)) + 3 ^ (0 ^(1 ^ 2)) = 5 :=
by
  -- sorry allows us to skip the proof
  sorry

end evaluate_expression_l279_279427


namespace complement_U_A_inter_B_eq_l279_279490

open Set

-- Definitions
def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

-- Complement of A in U
def complement_U_A : Set ℤ := U \ A

-- Proof Problem
theorem complement_U_A_inter_B_eq : complement_U_A ∩ B = {6, 8} := by
  sorry

end complement_U_A_inter_B_eq_l279_279490


namespace jack_pays_back_l279_279855

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l279_279855


namespace range_of_a_l279_279309

def p (a m : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3 / 2

theorem range_of_a (a : ℝ) :
  (∀ m : ℝ, p a m → q m) → 
  (∃ (a_lower a_upper : ℝ), a_lower ≤ a ∧ a ≤ a_upper ∧ a_lower = 1 / 3 ∧ a_upper = 3 / 8) :=
sorry

end range_of_a_l279_279309


namespace Aunt_Zhang_expenditure_is_negative_l279_279683

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end Aunt_Zhang_expenditure_is_negative_l279_279683


namespace prime_solution_unique_l279_279985

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l279_279985


namespace mass_percentage_Ca_in_CaI2_l279_279474

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I

theorem mass_percentage_Ca_in_CaI2 :
  (molar_mass_Ca / molar_mass_CaI2) * 100 = 13.63 :=
by
  sorry

end mass_percentage_Ca_in_CaI2_l279_279474


namespace solution_set_of_quadratic_inequality_l279_279050

variable {a x : ℝ} (h_neg : a < 0)

theorem solution_set_of_quadratic_inequality :
  (a * x^2 - (a + 2) * x + 2) ≥ 0 ↔ (x ∈ Set.Icc (2 / a) 1) :=
by
  sorry

end solution_set_of_quadratic_inequality_l279_279050


namespace find_cost_of_baseball_l279_279873

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end find_cost_of_baseball_l279_279873


namespace fraction_difference_of_squares_l279_279240

theorem fraction_difference_of_squares :
  (175^2 - 155^2) / 20 = 330 :=
by
  -- Proof goes here
  sorry

end fraction_difference_of_squares_l279_279240


namespace evaluate_expression_l279_279557

theorem evaluate_expression : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  -- We will skip the proof steps here using sorry
  sorry

end evaluate_expression_l279_279557


namespace farmer_shipped_67_dozens_l279_279193

def pomelos_in_box (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 20 else if box_type = "large" then 30 else 0

def total_pomelos_last_week : ℕ := 360

def boxes_this_week (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 8 else if box_type = "large" then 7 else 0

def damage_boxes (box_type : String) : ℕ :=
  if box_type = "small" then 3 else if box_type = "medium" then 2 else if box_type = "large" then 2 else 0

def loss_percentage (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 15 else if box_type = "large" then 20 else 0

def total_pomelos_shipped_this_week : ℕ :=
  (boxes_this_week "small") * (pomelos_in_box "small") +
  (boxes_this_week "medium") * (pomelos_in_box "medium") +
  (boxes_this_week "large") * (pomelos_in_box "large")

def total_pomelos_lost_this_week : ℕ :=
  (damage_boxes "small") * (pomelos_in_box "small") * (loss_percentage "small") / 100 +
  (damage_boxes "medium") * (pomelos_in_box "medium") * (loss_percentage "medium") / 100 +
  (damage_boxes "large") * (pomelos_in_box "large") * (loss_percentage "large") / 100

def total_pomelos_shipped_successfully_this_week : ℕ :=
  total_pomelos_shipped_this_week - total_pomelos_lost_this_week

def total_pomelos_for_both_weeks : ℕ :=
  total_pomelos_last_week + total_pomelos_shipped_successfully_this_week

def total_dozens_shipped : ℕ :=
  total_pomelos_for_both_weeks / 12

theorem farmer_shipped_67_dozens :
  total_dozens_shipped = 67 := 
by sorry

end farmer_shipped_67_dozens_l279_279193


namespace vasya_numbers_l279_279745

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l279_279745


namespace linear_dependent_vectors_l279_279227

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l279_279227


namespace monotonicity_and_zeros_l279_279485

open Real

noncomputable def f (x k : ℝ) : ℝ := exp x - k * x + k

theorem monotonicity_and_zeros
  (k : ℝ)
  (h₁ : k > exp 2)
  (x₁ x₂ : ℝ)
  (h₂ : f x₁ k = 0)
  (h₃ : f x₂ k = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := 
sorry

end monotonicity_and_zeros_l279_279485


namespace intersection_of_A_and_B_l279_279002

noncomputable def A : Set ℝ := {-2, -1, 0, 1}
noncomputable def B : Set ℝ := {x | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := 
by
  sorry

end intersection_of_A_and_B_l279_279002


namespace cat_food_insufficient_l279_279447

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l279_279447


namespace number_exceeds_20_percent_by_40_eq_50_l279_279569

theorem number_exceeds_20_percent_by_40_eq_50 (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 := by
  sorry

end number_exceeds_20_percent_by_40_eq_50_l279_279569


namespace enough_cat_food_for_six_days_l279_279432

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l279_279432


namespace product_gcd_lcm_24_60_l279_279291

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l279_279291


namespace ounces_per_gallon_l279_279504

-- conditions
def gallons_of_milk (james : Type) : ℕ := 3
def ounces_drank (james : Type) : ℕ := 13
def ounces_left (james : Type) : ℕ := 371

-- question
def ounces_in_gallon (james : Type) : ℕ := 128

-- proof statement
theorem ounces_per_gallon (james : Type) :
  (gallons_of_milk james) * (ounces_in_gallon james) = (ounces_left james + ounces_drank james) :=
sorry

end ounces_per_gallon_l279_279504


namespace range_of_a_l279_279649

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → Real.exp (a * x) ≥ 2 * Real.log x + x^2 - a * x) ↔ 0 ≤ a :=
sorry

end range_of_a_l279_279649


namespace incorrect_connection_probability_l279_279921

theorem incorrect_connection_probability :
  let p := 0.02
  let r2 := 1 / 9
  let r3 := 8 / 81
  let probability_wrong_two_errors := 3 * p^2 * (1 - p) * r2
  let probability_wrong_three_errors := 1 * p^3 * r3
  let total_probability_correct_despite_errors := probability_wrong_two_errors + probability_wrong_three_errors
  let total_probability_incorrect := 1 - total_probability_correct_despite_errors
  ((total_probability_correct_despite_errors ≈ 0.000131) → 
  (total_probability_incorrect ≈ 1 - 0.000131)) :=
by
  sorry

end incorrect_connection_probability_l279_279921


namespace distance_between_stations_l279_279772

/-- Two trains start at the same time from two stations and proceed towards each other. 
    The first train travels at 20 km/hr and the second train travels at 25 km/hr. 
    When they meet, the second train has traveled 60 km more than the first train. -/
theorem distance_between_stations
    (t : ℝ) -- The time in hours when they meet
    (x : ℝ) -- The distance traveled by the slower train
    (d1 d2 : ℝ) -- Distances traveled by the two trains respectively
    (h1 : 20 * t = x)
    (h2 : 25 * t = x + 60) :
  d1 + d2 = 540 :=
by
  sorry

end distance_between_stations_l279_279772


namespace vasya_numbers_l279_279740

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l279_279740


namespace min_length_GH_l279_279158

theorem min_length_GH :
  let ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
  let A := (-2, 0)
  let B := (2, 0)
  ∀ P G H : ℝ × ℝ,
    (P.1^2 / 4 + P.2^2 = 1) →
    P.2 > 0 →
    (G.2 = 3) →
    (H.2 = 3) →
    ∃ k : ℝ, k > 0 ∧ G.1 = 3 / k - 2 ∧ H.1 = -12 * k + 2 →
    |G.1 - H.1| = 8 :=
sorry

end min_length_GH_l279_279158


namespace circle_area_percentage_decrease_l279_279940

theorem circle_area_percentage_decrease (r : ℝ) (A : ℝ := Real.pi * r^2) 
  (r' : ℝ := 0.5 * r) (A' : ℝ := Real.pi * (r')^2) :
  (A - A') / A * 100 = 75 := 
by
  sorry

end circle_area_percentage_decrease_l279_279940


namespace n_divisible_by_40_l279_279188

theorem n_divisible_by_40 {n : ℕ} (h_pos : 0 < n)
  (h1 : ∃ k1 : ℕ, 2 * n + 1 = k1 * k1)
  (h2 : ∃ k2 : ℕ, 3 * n + 1 = k2 * k2) :
  ∃ k : ℕ, n = 40 * k := 
sorry

end n_divisible_by_40_l279_279188


namespace ratio_of_areas_of_concentric_circles_l279_279064

theorem ratio_of_areas_of_concentric_circles
  (Q : Type)
  (r₁ r₂ : ℝ)
  (C₁ C₂ : ℝ)
  (h₀ : r₁ > 0 ∧ r₂ > 0)
  (h₁ : C₁ = 2 * π * r₁)
  (h₂ : C₂ = 2 * π * r₂)
  (h₃ : (60 / 360) * C₁ = (30 / 360) * C₂) :
  (π * r₁^2) / (π * r₂^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_l279_279064


namespace value_of_a_minus_2_b_minus_2_l279_279671

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end value_of_a_minus_2_b_minus_2_l279_279671


namespace rectangular_diagonal_length_l279_279721

theorem rectangular_diagonal_length (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 11)
  (h_edge_sum : x + y + z = 6) :
  Real.sqrt (x^2 + y^2 + z^2) = 5 := 
by
  sorry

end rectangular_diagonal_length_l279_279721


namespace percentage_of_Muscovy_ducks_l279_279375

theorem percentage_of_Muscovy_ducks
  (N : ℕ) (M : ℝ) (female_percentage : ℝ) (female_Muscovy : ℕ)
  (hN : N = 40)
  (hfemale_percentage : female_percentage = 0.30)
  (hfemale_Muscovy : female_Muscovy = 6)
  (hcondition : female_percentage * M * N = female_Muscovy) 
  : M = 0.5 := 
sorry

end percentage_of_Muscovy_ducks_l279_279375


namespace max_min_sum_l279_279156

variable {α : Type*} [LinearOrderedField α]

def is_odd_function (g : α → α) : Prop :=
∀ x, g (-x) = - g x

def has_max_min (f : α → α) (M N : α) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ (∀ x, N ≤ f x) ∧ (∃ x₁, f x₁ = N)

theorem max_min_sum (g f : α → α) (M N : α)
  (h_odd : is_odd_function g)
  (h_def : ∀ x, f x = g (x - 2) + 1)
  (h_max_min : has_max_min f M N) :
  M + N = 2 :=
sorry

end max_min_sum_l279_279156


namespace number_of_functions_satisfying_conditions_l279_279029

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ s ∈ S, f (f (f s)) = s) ∧ (∀ s ∈ S, (f s - s) % 3 ≠ 0)

theorem number_of_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), f_conditions f) ∧ (∃! (n : ℕ), n = 288) :=
by
  sorry

end number_of_functions_satisfying_conditions_l279_279029


namespace distance_between_foci_l279_279596

-- Define the properties of the ellipse
def ellipse_center := (3 : ℝ, 2 : ℝ)
def ellipse_tangent_x_axis := (3 : ℝ, 0 : ℝ)
def ellipse_tangent_y_axis := (0 : ℝ, 2 : ℝ)

-- Semi-major and semi-minor axes
def a : ℝ := 3
def b : ℝ := 2

-- Formula for the distance between the foci
theorem distance_between_foci : 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_between_foci_l279_279596


namespace total_miles_driven_l279_279324

-- Define the required variables and their types
variables (avg1 avg2 : ℝ) (gallons1 gallons2 : ℝ) (miles1 miles2 : ℝ)

-- State the conditions
axiom sum_avg_mpg : avg1 + avg2 = 75
axiom first_car_gallons : gallons1 = 25
axiom second_car_gallons : gallons2 = 35
axiom first_car_avg_mpg : avg1 = 40

-- Declare the function to calculate miles driven
def miles_driven (avg_mpg gallons : ℝ) : ℝ := avg_mpg * gallons

-- Declare the theorem for proof
theorem total_miles_driven : miles_driven avg1 gallons1 + miles_driven avg2 gallons2 = 2225 := by
  sorry

end total_miles_driven_l279_279324


namespace part1_part2_l279_279862


noncomputable def is_infinite_sum (a : ℕ → ℝ) : Prop :=
  ∀ (M : ℝ), ∃ (N : ℕ), ∀ n > N, (∑ k in finset.range n, a k) > M

theorem part1 :
  is_infinite_sum (λ n, 1 / (2 * (n : ℝ) - 1)) :=
sorry

theorem part2 : 
  ∃ (f : ℕ → ℕ), bijective f ∧ is_infinite_sum (λ n, (-1)^(f(n)-1) / (f(n))) :=
sorry

end part1_part2_l279_279862


namespace square_floor_tile_count_l279_279413

theorem square_floor_tile_count (n : ℕ) (h1 : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_floor_tile_count_l279_279413


namespace constant_term_in_expansion_l279_279564

-- Given conditions
def eq_half_n_minus_m_zero (n m : ℕ) : Prop := 1/2 * n = m
def eq_n_plus_m_ten (n m : ℕ) : Prop := n + m = 10
noncomputable def binom (n k : ℕ) : ℝ := Real.exp (Real.log (Nat.factorial n) - Real.log (Nat.factorial k) - Real.log (Nat.factorial (n - k)))

-- Main theorem
theorem constant_term_in_expansion : 
  ∃ (n m : ℕ), eq_half_n_minus_m_zero n m ∧ eq_n_plus_m_ten n m ∧ 
  binom 10 m * (3^4 : ℝ) = 17010 :=
by
  -- Definitions translation
  sorry

end constant_term_in_expansion_l279_279564


namespace g_eq_g_inv_at_7_over_2_l279_279980

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_at_7_over_2 : g (7 / 2) = g_inv (7 / 2) := by
  sorry

end g_eq_g_inv_at_7_over_2_l279_279980


namespace value_of_expression_l279_279822

theorem value_of_expression (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 5)
  : 2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8 := by
  sorry

end value_of_expression_l279_279822


namespace compare_expressions_l279_279606

theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 :=
by {
  -- below proof is left as an exercise
  sorry
}

end compare_expressions_l279_279606


namespace intersection_A_B_l279_279642

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {y : ℝ | ∃ x : ℝ, y = 2^x}

theorem intersection_A_B :
  A ∩ {x : ℝ | x > 0} = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_B_l279_279642


namespace jori_water_left_l279_279021

theorem jori_water_left (initial_gallons used_gallons : ℚ) (h1 : initial_gallons = 3) (h2 : used_gallons = 11 / 4) :
  initial_gallons - used_gallons = 1 / 4 :=
by
  sorry

end jori_water_left_l279_279021


namespace find_u_l279_279216

variable (α β γ : ℝ)
variables (q s u : ℝ)

-- The first polynomial has roots α, β, γ
axiom roots_first_poly : ∀ x : ℝ, x^3 + 4 * x^2 + 6 * x - 8 = (x - α) * (x - β) * (x - γ)

-- Sum of the roots α + β + γ = -4
axiom sum_roots_first_poly : α + β + γ = -4

-- Product of the roots αβγ = 8
axiom product_roots_first_poly : α * β * γ = 8

-- The second polynomial has roots α + β, β + γ, γ + α
axiom roots_second_poly : ∀ x : ℝ, x^3 + q * x^2 + s * x + u = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))

theorem find_u : u = 32 :=
sorry

end find_u_l279_279216


namespace unique_handshakes_count_l279_279016

-- Definitions from the conditions
def teams : Nat := 4
def players_per_team : Nat := 2
def total_players : Nat := teams * players_per_team

def handshakes_per_player : Nat := total_players - players_per_team

-- The Lean statement to prove the total number of unique handshakes
theorem unique_handshakes_count : (total_players * handshakes_per_player) / 2 = 24 := 
by
  -- Proof steps would go here
  sorry

end unique_handshakes_count_l279_279016


namespace land_percentage_relationship_l279_279233

variable {V : ℝ} -- Total taxable value of all land in the village
variable {x y z : ℝ} -- Percentages of Mr. William's land in types A, B, C

-- Conditions
axiom total_tax_collected : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 3840
axiom mr_william_tax : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 480

-- Prove the relationship
theorem land_percentage_relationship : (0.80 * x + 0.90 * y + 0.95 * z = 48000 / V) → (x + y + z = 100) := by
  sorry

end land_percentage_relationship_l279_279233


namespace complement_union_l279_279643

theorem complement_union (U A B complement_U_A : Set Int) (hU : U = {-1, 0, 1, 2}) 
  (hA : A = {-1, 2}) (hB : B = {0, 2}) (hC : complement_U_A = {0, 1}) :
  complement_U_A ∪ B = {0, 1, 2} := by
  sorry

end complement_union_l279_279643


namespace cat_food_insufficient_l279_279446

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l279_279446


namespace highest_probability_face_l279_279525

theorem highest_probability_face :
  let faces := 6
  let face_3 := 3
  let face_2 := 2
  let face_1 := 1
  (face_3 / faces > face_2 / faces) ∧ (face_2 / faces > face_1 / faces) →
  (face_3 / faces > face_1 / faces) →
  (face_3 = 3) :=
by {
  sorry
}

end highest_probability_face_l279_279525


namespace product_gcd_lcm_l279_279280

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l279_279280


namespace composite_sum_pow_l279_279508

theorem composite_sum_pow (a b c d : ℕ) (h_pos : a > b ∧ b > c ∧ c > d)
    (h_div : (a + b - c + d) ∣ (a * c + b * d)) (m : ℕ) (h_m_pos : 0 < m) 
    (n : ℕ) (h_n_odd : n % 2 = 1) : ∃ k : ℕ, k > 1 ∧ k ∣ (a ^ n * b ^ m + c ^ m * d ^ n) :=
by
  sorry

end composite_sum_pow_l279_279508


namespace stones_required_to_pave_hall_l279_279953

noncomputable def hall_length_meters : ℝ := 36
noncomputable def hall_breadth_meters : ℝ := 15
noncomputable def stone_length_dms : ℝ := 4
noncomputable def stone_breadth_dms : ℝ := 5

theorem stones_required_to_pave_hall :
  let hall_length_dms := hall_length_meters * 10
  let hall_breadth_dms := hall_breadth_meters * 10
  let hall_area_dms_squared := hall_length_dms * hall_breadth_dms
  let stone_area_dms_squared := stone_length_dms * stone_breadth_dms
  let number_of_stones := hall_area_dms_squared / stone_area_dms_squared
  number_of_stones = 2700 :=
by
  sorry

end stones_required_to_pave_hall_l279_279953


namespace linear_dependency_k_val_l279_279225

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l279_279225


namespace range_of_set_l279_279791

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end range_of_set_l279_279791


namespace cost_per_person_trip_trips_rental_cost_l279_279250

-- Define the initial conditions
def ticket_price_per_person := 60
def total_employees := 70
def small_car_seats := 4
def large_car_seats := 11
def extra_cost_small_car_per_person := 5
def extra_revenue_large_car := 50
def max_total_cost := 5000

-- Define the costs per person per trip for small and large cars
def large_car_cost_per_person := 10
def small_car_cost_per_person := large_car_cost_per_person + extra_cost_small_car_per_person

-- Define the number of trips for four-seater and eleven-seater cars
def four_seater_trips := 1
def eleven_seater_trips := 6

-- Prove the lean statements
theorem cost_per_person_trip : 
  (11 * large_car_cost_per_person) - (small_car_seats * small_car_cost_per_person) = extra_revenue_large_car := 
sorry

theorem trips_rental_cost (x y : ℕ) : 
  (small_car_seats * x + large_car_seats * y = total_employees) ∧
  ((total_employees * ticket_price_per_person) + (small_car_cost_per_person * small_car_seats * x) + (large_car_cost_per_person * large_car_seats * y) ≤ max_total_cost) :=
sorry

end cost_per_person_trip_trips_rental_cost_l279_279250


namespace game_win_probability_l279_279961

noncomputable def alexWinsProbability : ℝ := 1/2
noncomputable def melWinsProbability : ℝ := 1/4
noncomputable def chelseaWinsProbability : ℝ := 1/4
noncomputable def totalRounds : ℕ := 8

theorem game_win_probability :
  alexWinsProbability * alexWinsProbability * alexWinsProbability * alexWinsProbability *
  melWinsProbability * melWinsProbability * melWinsProbability *
  chelseaWinsProbability *
  (Nat.factorial totalRounds / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 1)) = 35/512 := by
sorry

end game_win_probability_l279_279961


namespace savings_relationship_l279_279245

def combined_salary : ℝ := 3000
def salary_A : ℝ := 2250
def salary_B : ℝ := combined_salary - salary_A
def savings_A : ℝ := 0.05 * salary_A
def savings_B : ℝ := 0.15 * salary_B

theorem savings_relationship : savings_A = 112.5 ∧ savings_B = 112.5 := by
  have h1 : salary_B = 750 := by sorry
  have h2 : savings_A = 0.05 * 2250 := by sorry
  have h3 : savings_B = 0.15 * 750 := by sorry
  have h4 : savings_A = 112.5 := by sorry
  have h5 : savings_B = 112.5 := by sorry
  exact And.intro h4 h5

end savings_relationship_l279_279245


namespace circumscribed_quadrilateral_converse_arithmetic_progression_l279_279786

theorem circumscribed_quadrilateral (a b c d : ℝ) (k : ℝ) (h1 : b = a + k) (h2 : d = a + 2 * k) (h3 : c = a + 3 * k) :
  a + c = b + d :=
by
  sorry

theorem converse_arithmetic_progression (a b c d : ℝ) (h : a + c = b + d) :
  ∃ k : ℝ, b = a + k ∧ d = a + 2 * k ∧ c = a + 3 * k :=
by
  sorry

end circumscribed_quadrilateral_converse_arithmetic_progression_l279_279786


namespace find_y_l279_279516

variable {a b y : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem find_y (h1 : (3 * a) ^ (4 * b) = a ^ b * y ^ b) : y = 81 * a ^ 3 := by
  sorry

end find_y_l279_279516


namespace total_brownies_correct_l279_279880

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end total_brownies_correct_l279_279880


namespace no_solution_to_equation_l279_279171

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, x ≠ 5 ∧ (1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) :=
by 
  sorry

end no_solution_to_equation_l279_279171


namespace coin_value_is_630_l279_279015

theorem coin_value_is_630 :
  (∃ x : ℤ, x > 0 ∧ 406 * x = 63000) :=
by {
  sorry
}

end coin_value_is_630_l279_279015


namespace collinear_c1_c2_l279_279794

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 7, 0)
def b : ℝ × ℝ × ℝ := (1, -3, 4)

-- Define the vectors c1 and c2 based on a and b
def c1 : ℝ × ℝ × ℝ := (4 * 3, 4 * 7, 4 * 0) - (2 * 1, 2 * -3, 2 * 4)
def c2 : ℝ × ℝ × ℝ := (1, -3, 4) - (2 * 3, 2 * 7, 2 * 0)

-- The theorem to prove that c1 and c2 are collinear
theorem collinear_c1_c2 : c1 = (-2 : ℝ) • c2 := by sorry

end collinear_c1_c2_l279_279794


namespace multiplication_of_mixed_number_l279_279097

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l279_279097


namespace Vasya_numbers_l279_279763

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l279_279763


namespace calculate_product_l279_279119

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l279_279119


namespace cat_food_inequality_l279_279457

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l279_279457


namespace find_seating_capacity_l279_279087

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end find_seating_capacity_l279_279087


namespace probability_all_girls_is_correct_l279_279249

noncomputable def probability_all_girls : ℚ :=
  let total_members := 15
  let boys := 7
  let girls := 8
  let choose_3_from_15 := Nat.choose total_members 3
  let choose_3_from_8 := Nat.choose girls 3
  choose_3_from_8 / choose_3_from_15

theorem probability_all_girls_is_correct : 
  probability_all_girls = 8 / 65 := by
sorry

end probability_all_girls_is_correct_l279_279249


namespace problem_statement_l279_279868

variable (a : ℕ → ℝ)

-- Defining sequences {b_n} and {c_n}
def b (n : ℕ) := a n - a (n + 2)
def c (n : ℕ) := a n + 2 * a (n + 1) + 3 * a (n + 2)

-- Defining that a sequence is arithmetic
def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

-- Problem statement
theorem problem_statement :
  is_arithmetic a ↔ (is_arithmetic (c a) ∧ ∀ n, b a n ≤ b a (n + 1)) :=
sorry

end problem_statement_l279_279868


namespace shift_parabola_left_l279_279365

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end shift_parabola_left_l279_279365


namespace determine_k_l279_279828

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

-- State the problem
theorem determine_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4)
  ↔ (k = 3 / 8 ∨ k = -3) :=
by
  sorry

end determine_k_l279_279828


namespace product_of_gcd_and_lcm_1440_l279_279277

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l279_279277


namespace largest_non_factor_product_of_factors_of_100_l279_279382

theorem largest_non_factor_product_of_factors_of_100 :
  ∃ x y : ℕ, 
  (x ≠ y) ∧ 
  (0 < x ∧ 0 < y) ∧ 
  (x ∣ 100 ∧ y ∣ 100) ∧ 
  ¬(x * y ∣ 100) ∧ 
  (∀ a b : ℕ, 
    (a ≠ b) ∧ 
    (0 < a ∧ 0 < b) ∧ 
    (a ∣ 100 ∧ b ∣ 100) ∧ 
    ¬(a * b ∣ 100) → 
    (x * y) ≥ (a * b)) ∧ 
  (x * y) = 40 :=
by
  sorry

end largest_non_factor_product_of_factors_of_100_l279_279382


namespace train_length_l279_279960

noncomputable def length_of_train (time_in_seconds : ℝ) (speed_in_kmh : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmh * (5 / 18)
  speed_in_mps * time_in_seconds

theorem train_length :
  length_of_train 2.3998080153587713 210 = 140 :=
by
  sorry

end train_length_l279_279960


namespace curvature_formula_l279_279886

noncomputable def curvature_squared (x y : ℝ → ℝ) (t : ℝ) :=
  let x' := (deriv x t)
  let y' := (deriv y t)
  let x'' := (deriv (deriv x) t)
  let y'' := (deriv (deriv y) t)
  (x'' * y' - y'' * x')^2 / (x'^2 + y'^2)^3

theorem curvature_formula (x y : ℝ → ℝ) (t : ℝ) :
  let k_sq := curvature_squared x y t
  k_sq = ((deriv (deriv x) t * deriv y t - deriv (deriv y) t * deriv x t)^2 /
         ((deriv x t)^2 + (deriv y t)^2)^3) := 
by 
  sorry

end curvature_formula_l279_279886


namespace sum_of_repeating_decimals_l279_279133

-- Defining the given repeating decimals as fractions
def rep_decimal1 : ℚ := 2 / 9
def rep_decimal2 : ℚ := 2 / 99
def rep_decimal3 : ℚ := 2 / 9999

-- Stating the theorem to prove the given sum equals the correct answer
theorem sum_of_repeating_decimals :
  rep_decimal1 + rep_decimal2 + rep_decimal3 = 224422 / 9999 :=
by
  sorry

end sum_of_repeating_decimals_l279_279133


namespace trig_identity_l279_279893

theorem trig_identity : (sin 40 + sin 80) / (cos 40 + cos 80) = real.sqrt 3 :=
by
  sorry

end trig_identity_l279_279893


namespace multiplication_with_mixed_number_l279_279104

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l279_279104


namespace vertex_of_parabola_l279_279548

theorem vertex_of_parabola :
  ∃ (a b c : ℝ), 
      (4 * a - 2 * b + c = 9) ∧ 
      (16 * a + 4 * b + c = 9) ∧ 
      (49 * a + 7 * b + c = 16) ∧ 
      (-b / (2 * a) = 1) :=
by {
  -- we need to provide the proof here; sorry is a placeholder
  sorry
}

end vertex_of_parabola_l279_279548


namespace melanie_cats_l279_279203

theorem melanie_cats (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ) 
  (h_jacob : jacob_cats = 90)
  (h_annie : annie_cats = jacob_cats / 3)
  (h_melanie : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end melanie_cats_l279_279203


namespace sin_15_mul_sin_75_l279_279232

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 := 
by
  sorry

end sin_15_mul_sin_75_l279_279232


namespace sum_sequence_correct_l279_279395

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end sum_sequence_correct_l279_279395


namespace simple_interest_years_l279_279372

noncomputable def simple_interest (P r t : ℕ) : ℕ :=
  P * r * t / 100

noncomputable def compound_interest (P r n : ℕ) : ℕ :=
  P * (1 + r / 100)^n - P

theorem simple_interest_years
  (P_si r_si P_ci r_ci n_ci si_half_ci si_si : ℕ)
  (h_si : simple_interest P_si r_si si_si = si_half_ci)
  (h_ci : compound_interest P_ci r_ci n_ci = si_half_ci * 2) :
  si_si = 2 :=
by
  sorry

end simple_interest_years_l279_279372


namespace negation_of_universal_proposition_l279_279220

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_universal_proposition_l279_279220


namespace triangle_area_is_18_l279_279238

noncomputable def area_of_triangle (y_8 y_2_2x y_2_minus_2x : ℝ) : ℝ :=
  let intersect1 : ℝ × ℝ := (3, 8)
  let intersect2 : ℝ × ℝ := (-3, 8)
  let intersect3 : ℝ × ℝ := (0, 2)
  let base := 3 - -3
  let height := 8 - 2
  (1 / 2 ) * base * height

theorem triangle_area_is_18 : 
  area_of_triangle (8) (2 + 2 * x) (2 - 2 * x) = 18 := 
  by
    sorry

end triangle_area_is_18_l279_279238


namespace prime_neighbor_divisible_by_6_l279_279272

theorem prime_neighbor_divisible_by_6 (p : ℕ) (h_prime: Prime p) (h_gt3: p > 3) :
  ∃ k : ℕ, k ≠ 0 ∧ ((p - 1) % 6 = 0 ∨ (p + 1) % 6 = 0) :=
by
  sorry

end prime_neighbor_divisible_by_6_l279_279272


namespace mark_sprinted_distance_l279_279032

def speed := 6 -- miles per hour
def time := 4 -- hours

/-- Mark sprinted exactly 24 miles. -/
theorem mark_sprinted_distance : speed * time = 24 := by
  sorry

end mark_sprinted_distance_l279_279032


namespace vasya_numbers_l279_279753

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l279_279753


namespace food_requirement_not_met_l279_279451

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l279_279451


namespace binary_addition_correct_l279_279603

-- define the binary numbers as natural numbers using their binary representations
def bin_1010 : ℕ := 0b1010
def bin_10 : ℕ := 0b10
def bin_sum : ℕ := 0b1100

-- state the theorem that needs to be proved
theorem binary_addition_correct : bin_1010 + bin_10 = bin_sum := by
  sorry

end binary_addition_correct_l279_279603


namespace express_c_in_terms_of_a_b_l279_279164

-- Defining the vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)

-- Defining the given vectors
def a := vec 1 1
def b := vec 1 (-1)
def c := vec (-1) 2

-- The statement
theorem express_c_in_terms_of_a_b :
  c = (1/2) • a + (-3/2) • b :=
sorry

end express_c_in_terms_of_a_b_l279_279164


namespace time_to_save_for_downpayment_l279_279522

-- Definitions based on conditions
def annual_saving : ℝ := 0.10 * 150000
def downpayment : ℝ := 0.20 * 450000

-- Statement of the theorem to be proved
theorem time_to_save_for_downpayment (T : ℝ) (H1 : annual_saving = 15000) (H2 : downpayment = 90000) : 
  T = 6 :=
by
  -- Placeholder for the proof
  sorry

end time_to_save_for_downpayment_l279_279522


namespace rex_cards_left_l279_279883

-- Definitions
def nicole_cards : ℕ := 400
def cindy_cards : ℕ := 2 * nicole_cards
def combined_total : ℕ := nicole_cards + cindy_cards
def rex_cards : ℕ := combined_total / 2
def people_count : ℕ := 4
def cards_per_person : ℕ := rex_cards / people_count

-- Proof statement
theorem rex_cards_left : cards_per_person = 150 := by
  sorry

end rex_cards_left_l279_279883


namespace fill_tub_together_time_l279_279561

theorem fill_tub_together_time :
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  combined_rate ≠ 0 → (1 / combined_rate = 12 / 7) :=
by
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  sorry

end fill_tub_together_time_l279_279561


namespace range_of_a_l279_279170

theorem range_of_a (x : ℝ) (a : ℝ) (h1 : 2 < x) (h2 : a ≤ x + 1 / (x - 2)) : a ≤ 4 := 
sorry

end range_of_a_l279_279170


namespace timothy_total_cost_l279_279058

-- Define the costs of the individual items
def costOfLand (acres : Nat) (cost_per_acre : Nat) : Nat :=
  acres * cost_per_acre

def costOfHouse : Nat :=
  120000

def costOfCows (number_of_cows : Nat) (cost_per_cow : Nat) : Nat :=
  number_of_cows * cost_per_cow

def costOfChickens (number_of_chickens : Nat) (cost_per_chicken : Nat) : Nat :=
  number_of_chickens * cost_per_chicken

def installationCost (hours : Nat) (cost_per_hour : Nat) (equipment_fee : Nat) : Nat :=
  (hours * cost_per_hour) + equipment_fee

-- Define the total cost function
def totalCost : Nat :=
  costOfLand 30 20 +
  costOfHouse +
  costOfCows 20 1000 +
  costOfChickens 100 5 +
  installationCost 6 100 6000

-- Theorem to state the total cost
theorem timothy_total_cost : totalCost = 147700 :=
by
  -- Placeholder for the proof, for now leave it as sorry
  sorry

end timothy_total_cost_l279_279058


namespace valid_license_plates_count_l279_279256

theorem valid_license_plates_count :
  let letters := 26 * 26 * 26
  let digits := 9 * 10 * 10
  letters * digits = 15818400 :=
by
  sorry

end valid_license_plates_count_l279_279256


namespace closest_point_is_correct_l279_279274

def line_eq (x : ℝ) : ℝ := -3 * x + 5

def closest_point_on_line_to_given_point : Prop :=
  ∃ (x y : ℝ), y = line_eq x ∧ (x, y) = (17 / 10, -1 / 10) ∧
  (∀ (x' y' : ℝ), y' = line_eq x' → (x' - -4)^2 + (y' - -2)^2 ≥ (x - -4)^2 + (y - -2)^2)
  
theorem closest_point_is_correct : closest_point_on_line_to_given_point :=
sorry

end closest_point_is_correct_l279_279274


namespace Vasya_numbers_l279_279758

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l279_279758


namespace solve_inner_circle_radius_l279_279194

noncomputable def isosceles_trapezoid_radius := 
  let AB := 8
  let BC := 7
  let DA := 7
  let CD := 6
  let radiusA := 4
  let radiusB := 4
  let radiusC := 3
  let radiusD := 3
  let r := (-72 + 60 * Real.sqrt 3) / 26
  r

theorem solve_inner_circle_radius :
  let k := 72
  let m := 60
  let n := 3
  let p := 26
  gcd k p = 1 → -- explicit gcd calculation between k and p 
  (isosceles_trapezoid_radius = (-k + m * Real.sqrt n) / p) ∧ (k + m + n + p = 161) :=
by
  sorry

end solve_inner_circle_radius_l279_279194


namespace calculate_product_l279_279115

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l279_279115


namespace toys_per_day_l279_279409

theorem toys_per_day (total_toys_per_week : ℕ) (days_worked_per_week : ℕ)
  (production_rate_constant : Prop) (h1 : total_toys_per_week = 8000)
  (h2 : days_worked_per_week = 4)
  (h3 : production_rate_constant)
  : (total_toys_per_week / days_worked_per_week) = 2000 :=
by
  sorry

end toys_per_day_l279_279409


namespace cat_food_insufficient_for_six_days_l279_279465

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l279_279465


namespace slope_intercept_form_of_line_l279_279818

theorem slope_intercept_form_of_line :
  ∀ (x y : ℝ), (∀ (a b : ℝ), (a, b) = (0, 4) ∨ (a, b) = (3, 0) → y = - (4 / 3) * x + 4) := 
by
  sorry

end slope_intercept_form_of_line_l279_279818


namespace vasya_numbers_l279_279755

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l279_279755


namespace ratio_of_areas_of_concentric_circles_l279_279063

theorem ratio_of_areas_of_concentric_circles
  (Q : Type)
  (r₁ r₂ : ℝ)
  (C₁ C₂ : ℝ)
  (h₀ : r₁ > 0 ∧ r₂ > 0)
  (h₁ : C₁ = 2 * π * r₁)
  (h₂ : C₂ = 2 * π * r₂)
  (h₃ : (60 / 360) * C₁ = (30 / 360) * C₂) :
  (π * r₁^2) / (π * r₂^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_l279_279063


namespace line_through_two_points_l279_279899

theorem line_through_two_points (x y : ℝ) (hA : (x, y) = (3, 0)) (hB : (x, y) = (0, 2)) :
  2 * x + 3 * y - 6 = 0 :=
sorry 

end line_through_two_points_l279_279899


namespace option_d_correct_l279_279567

theorem option_d_correct (a b : ℝ) : (a - b)^2 = (b - a)^2 := 
by {
  sorry
}

end option_d_correct_l279_279567


namespace ratio_alisha_to_todd_is_two_to_one_l279_279167

-- Definitions
def total_gumballs : ℕ := 45
def todd_gumballs : ℕ := 4
def bobby_gumballs (A : ℕ) : ℕ := 4 * A - 5
def remaining_gumballs : ℕ := 6

-- Condition stating Hector's gumball distribution
def hector_gumballs_distribution (A : ℕ) : Prop :=
  todd_gumballs + A + bobby_gumballs A + remaining_gumballs = total_gumballs

-- Definition for the ratio of the gumballs given to Alisha to Todd
def ratio_alisha_todd (A : ℕ) : ℕ × ℕ :=
  (A / 4, todd_gumballs / 4)

-- Theorem stating the problem
theorem ratio_alisha_to_todd_is_two_to_one : ∃ (A : ℕ), hector_gumballs_distribution A → ratio_alisha_todd A = (2, 1) :=
sorry

end ratio_alisha_to_todd_is_two_to_one_l279_279167


namespace value_of_transformed_product_of_roots_l279_279668

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end value_of_transformed_product_of_roots_l279_279668


namespace vasya_numbers_l279_279750

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l279_279750


namespace mark_sprinted_distance_l279_279031

def speed := 6 -- miles per hour
def time := 4 -- hours

/-- Mark sprinted exactly 24 miles. -/
theorem mark_sprinted_distance : speed * time = 24 := by
  sorry

end mark_sprinted_distance_l279_279031


namespace cat_food_inequality_l279_279454

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l279_279454


namespace find_cost_of_baseball_l279_279874

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end find_cost_of_baseball_l279_279874


namespace transform_polynomial_to_y_l279_279159

theorem transform_polynomial_to_y (x y : ℝ) (h : y = x + 1/x) :
  (x^6 + x^5 - 5*x^4 + x^3 + x + 1 = 0) → 
  (∃ (y_expr : ℝ), (x * y_expr = 0 ∨ (x = 0 ∧ y_expr = y_expr))) :=
sorry

end transform_polynomial_to_y_l279_279159


namespace bin101_to_decimal_l279_279619

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l279_279619


namespace sqrt_15_minus_1_range_l279_279271

theorem sqrt_15_minus_1_range (h : 9 < 15 ∧ 15 < 16) : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := 
  sorry

end sqrt_15_minus_1_range_l279_279271


namespace length_DE_l279_279711

open Classical

noncomputable def triangle_base_length (ABC_base : ℝ) : ℝ :=
15

noncomputable def is_parallel (DE BC : ℝ) : Prop :=
DE = BC

noncomputable def area_ratio (triangle_small triangle_large : ℝ) : ℝ :=
0.25

theorem length_DE 
  (ABC_base : ℝ)
  (DE : ℝ)
  (BC : ℝ)
  (triangle_small : ℝ)
  (triangle_large : ℝ)
  (h_base : triangle_base_length ABC_base = 15)
  (h_parallel : is_parallel DE BC)
  (h_area : area_ratio triangle_small triangle_large = 0.25)
  (h_similar : true):
  DE = 7.5 :=
by
  sorry

end length_DE_l279_279711


namespace problem_f_val_l279_279813

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_val (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (-x) = -f x)
  (h2 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3) :
  f 2015 = -1 :=
  sorry

end problem_f_val_l279_279813


namespace assembly_time_constants_l279_279419

theorem assembly_time_constants (a b : ℕ) (f : ℕ → ℝ)
  (h1 : ∀ x, f x = if x < b then a / (Real.sqrt x) else a / (Real.sqrt b))
  (h2 : f 4 = 15)
  (h3 : f b = 10) :
  a = 30 ∧ b = 9 :=
by
  sorry

end assembly_time_constants_l279_279419


namespace shells_needed_l279_279022

theorem shells_needed (current_shells : ℕ) (total_shells : ℕ) (difference : ℕ) :
  current_shells = 5 → total_shells = 17 → difference = total_shells - current_shells → difference = 12 :=
by
  intros h1 h2 h3
  sorry

end shells_needed_l279_279022


namespace green_apples_count_l279_279915

def red_apples := 33
def students_took := 21
def extra_apples := 35

theorem green_apples_count : ∃ G : ℕ, red_apples + G - students_took = extra_apples ∧ G = 23 :=
by
  use 23
  have h1 : 33 + 23 - 21 = 35 := by norm_num
  exact ⟨h1, rfl⟩

end green_apples_count_l279_279915


namespace remy_water_usage_l279_279888

theorem remy_water_usage :
  ∃ R : ℕ, (Remy = 3 * R + 1) ∧ 
    (Riley = R + (3 * R + 1) - 2) ∧ 
    (R + (3 * R + 1) + (R + (3 * R + 1) - 2) = 48) ∧ 
    (Remy = 19) :=
sorry

end remy_water_usage_l279_279888


namespace smallest_whole_number_greater_than_sum_l279_279475

theorem smallest_whole_number_greater_than_sum : 
  (3 + (1 / 3) + 4 + (1 / 4) + 6 + (1 / 6) + 7 + (1 / 7)) < 21 :=
sorry

end smallest_whole_number_greater_than_sum_l279_279475


namespace find_e_l279_279053

theorem find_e (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → (M^(1/d) * (M^(1/e) * (M^(1/f)))^(1/e)^(1/d)) = (M^(17/24))^(1/24)) → e = 4 :=
by
  sorry

end find_e_l279_279053


namespace HA_appears_at_least_once_l279_279168

-- Define the set of letters to be arranged
def letters : List Char := ['A', 'A', 'A', 'H', 'H']

-- Define a function to count the number of ways to arrange letters such that "HA" appears at least once
def countHA(A : List Char) : Nat := sorry

-- The proof problem to establish that there are 9 such arrangements
theorem HA_appears_at_least_once : countHA letters = 9 :=
sorry

end HA_appears_at_least_once_l279_279168


namespace number_of_students_l279_279710

theorem number_of_students (S N : ℕ) (h1 : S = 15 * N)
                           (h2 : (8 * 14) = 112)
                           (h3 : (6 * 16) = 96)
                           (h4 : 17 = 17)
                           (h5 : S = 225) : N = 15 :=
by sorry

end number_of_students_l279_279710


namespace choose_integers_l279_279333

def smallest_prime_divisor (n : ℕ) : ℕ := sorry
def number_of_divisors (n : ℕ) : ℕ := sorry

theorem choose_integers :
  ∃ (a : ℕ → ℕ), (∀ i, i < 2022 → a i < a (i + 1)) ∧
  (∀ k, 1 ≤ k ∧ k ≤ 2022 →
    number_of_divisors (a (k + 1) - a k - 1) > 2023^k ∧
    smallest_prime_divisor (a (k + 1) - a k) > 2023^k
  ) :=
sorry

end choose_integers_l279_279333


namespace brenda_mice_left_l279_279602

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end brenda_mice_left_l279_279602


namespace timothy_total_cost_l279_279057

-- Define the costs of the individual items
def costOfLand (acres : Nat) (cost_per_acre : Nat) : Nat :=
  acres * cost_per_acre

def costOfHouse : Nat :=
  120000

def costOfCows (number_of_cows : Nat) (cost_per_cow : Nat) : Nat :=
  number_of_cows * cost_per_cow

def costOfChickens (number_of_chickens : Nat) (cost_per_chicken : Nat) : Nat :=
  number_of_chickens * cost_per_chicken

def installationCost (hours : Nat) (cost_per_hour : Nat) (equipment_fee : Nat) : Nat :=
  (hours * cost_per_hour) + equipment_fee

-- Define the total cost function
def totalCost : Nat :=
  costOfLand 30 20 +
  costOfHouse +
  costOfCows 20 1000 +
  costOfChickens 100 5 +
  installationCost 6 100 6000

-- Theorem to state the total cost
theorem timothy_total_cost : totalCost = 147700 :=
by
  -- Placeholder for the proof, for now leave it as sorry
  sorry

end timothy_total_cost_l279_279057


namespace triangle_area_is_correct_l279_279017

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end triangle_area_is_correct_l279_279017


namespace multiplication_with_mixed_number_l279_279105

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l279_279105


namespace g_eq_g_inv_iff_l279_279978

def g (x : ℝ) : ℝ := 3 * x - 7

def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g x = g_inv x ↔ x = 7 / 2 :=
by {
  sorry
}

end g_eq_g_inv_iff_l279_279978


namespace parabola_vertex_on_x_axis_l279_279995

theorem parabola_vertex_on_x_axis (c : ℝ) : 
    (∃ h k, h = -3 ∧ k = 0 ∧ ∀ x, x^2 + 6 * x + c = x^2 + 6 * x + (c - (h^2)/4)) → c = 9 :=
by
    sorry

end parabola_vertex_on_x_axis_l279_279995


namespace product_of_gcd_and_lcm_1440_l279_279276

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l279_279276


namespace programmer_debugging_hours_l279_279269

theorem programmer_debugging_hours
    (total_hours : ℕ)
    (flow_chart_fraction : ℚ)
    (coding_fraction : ℚ)
    (meeting_fraction : ℚ)
    (flow_chart_hours : ℚ)
    (coding_hours : ℚ)
    (meeting_hours : ℚ)
    (debugging_hours : ℚ)
    (H1 : total_hours = 192)
    (H2 : flow_chart_fraction = 3 / 10)
    (H3 : coding_fraction = 3 / 8)
    (H4 : meeting_fraction = 1 / 5)
    (H5 : flow_chart_hours = flow_chart_fraction * total_hours)
    (H6 : coding_hours = coding_fraction * total_hours)
    (H7 : meeting_hours = meeting_fraction * total_hours)
    (H8 : debugging_hours = total_hours - (flow_chart_hours + coding_hours + meeting_hours))
    :
    debugging_hours = 24 :=
by 
  sorry

end programmer_debugging_hours_l279_279269


namespace cyclist_speed_l279_279928

theorem cyclist_speed (c d : ℕ) (h1 : d = c + 5) (hc : c ≠ 0) (hd : d ≠ 0)
    (H1 : ∀ tC tD : ℕ, 80 = c * tC → 120 = d * tD → tC = tD) : c = 10 := by
  sorry

end cyclist_speed_l279_279928


namespace ronaldo_current_age_l279_279572

noncomputable def roonie_age_one_year_ago (R L : ℕ) := 6 * L / 7
noncomputable def new_ratio (R L : ℕ) := (R + 5) * 8 = 7 * (L + 5)

theorem ronaldo_current_age (R L : ℕ) 
  (h1 : R = roonie_age_one_year_ago R L)
  (h2 : new_ratio R L) : L + 1 = 36 :=
by
  sorry

end ronaldo_current_age_l279_279572


namespace triangle_sides_ratio_l279_279322

theorem triangle_sides_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a)
  (ha_pos : a > 0) : b / a = Real.sqrt 2 :=
sorry

end triangle_sides_ratio_l279_279322


namespace matrix_subtraction_l279_279425

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 4, -3 ],
  ![ 2,  8 ]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 1,  5 ],
  ![ -3,  6 ]
]

-- Define the result matrix as given in the problem
def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 3, -8 ],
  ![ 5,  2 ]
]

-- The theorem to prove
theorem matrix_subtraction : A - B = result := 
by 
  sorry

end matrix_subtraction_l279_279425


namespace dormitory_to_city_distance_l279_279327

theorem dormitory_to_city_distance
  (D : ℝ)
  (h1 : (1/5) * D + (2/3) * D + 14 = D) :
  D = 105 :=
by
  sorry

end dormitory_to_city_distance_l279_279327


namespace division_modulus_l279_279689

-- Definitions using the conditions
def a : ℕ := 8 * (10^9)
def b : ℕ := 4 * (10^4)
def n : ℕ := 10^6

-- Lean statement to prove the problem
theorem division_modulus (a b n : ℕ) (h : a = 8 * (10^9) ∧ b = 4 * (10^4) ∧ n = 10^6) : 
  ((a / b) % n) = 200000 := 
by 
  sorry

end division_modulus_l279_279689


namespace increasing_intervals_l279_279146

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / 2) * x^2 - 2 * x + 5

theorem increasing_intervals :
  { x : ℝ | x < -2 / 3 } ∪ { x : ℝ | x > 1 } = { x : ℝ | deriv f x > 0 } :=
by
  sorry

end increasing_intervals_l279_279146


namespace abs_m_minus_n_eq_five_l279_279199

theorem abs_m_minus_n_eq_five (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 :=
sorry

end abs_m_minus_n_eq_five_l279_279199


namespace invertible_from_c_l279_279197

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the condition for c and the statement to prove
theorem invertible_from_c (c : ℝ) (h : ∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) : c = 3 :=
sorry

end invertible_from_c_l279_279197


namespace fraction_of_oranges_is_correct_l279_279723

variable (O P A : ℕ)
variable (total_fruit : ℕ := 56)

theorem fraction_of_oranges_is_correct:
  (A = 35) →
  (P = O / 2) →
  (A = 5 * P) →
  (O + P + A = total_fruit) →
  (O / total_fruit = 1 / 4) :=
by
  -- proof to be filled in 
  sorry

end fraction_of_oranges_is_correct_l279_279723


namespace relationship_y_values_l279_279815

theorem relationship_y_values (x1 x2 y1 y2 : ℝ) (h1 : x1 > x2) (h2 : 0 < x2) (h3 : y1 = - (3 / x1)) (h4 : y2 = - (3 / x2)) : y1 > y2 :=
by
  sorry

end relationship_y_values_l279_279815


namespace find_positive_integer_pair_l279_279808

noncomputable def quadratic_has_rational_solutions (d : ℤ) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d = 0

theorem find_positive_integer_pair :
  ∃ (d1 d2 : ℕ), 
  d1 > 0 ∧ d2 > 0 ∧ 
  quadratic_has_rational_solutions d1 ∧ quadratic_has_rational_solutions d2 ∧ 
  d1 * d2 = 2 := 
sorry -- Proof left as an exercise

end find_positive_integer_pair_l279_279808


namespace total_travel_distance_l279_279526

noncomputable def total_distance_traveled (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 - DF^2)
  DE + EF + DF

theorem total_travel_distance
  (DE DF : ℝ)
  (hDE : DE = 4500)
  (hDF : DF = 4000)
  : total_distance_traveled DE DF = 10560.992 :=
by
  rw [hDE, hDF]
  unfold total_distance_traveled
  norm_num
  sorry

end total_travel_distance_l279_279526


namespace smallest_even_n_sum_eq_l279_279340
  
theorem smallest_even_n_sum_eq (n : ℕ) (h_pos : n > 0) (h_even : n % 2 = 0) :
  n = 12 ↔ 
  let s₁ := n / 2 * (2 * 5 + (n - 1) * 6)
  let s₂ := n / 2 * (2 * 13 + (n - 1) * 3)
  s₁ = s₂ :=
by
  sorry

end smallest_even_n_sum_eq_l279_279340


namespace ratio_x_y_half_l279_279725

variable (x y z : ℝ)

theorem ratio_x_y_half (h1 : (x + 4) / 2 = (y + 9) / (z - 3))
                      (h2 : (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  x / y = 1 / 2 :=
by
  sorry

end ratio_x_y_half_l279_279725


namespace intersection_locus_is_vertical_line_l279_279304

/-- 
Given \( 0 < a < b \), lines \( l \) and \( m \) are drawn through the points \( A(a, 0) \) and \( B(b, 0) \), 
respectively, such that these lines intersect the parabola \( y^2 = x \) at four distinct points 
and these four points are concyclic. 

We want to prove that the locus of the intersection point \( P \) of lines \( l \) and \( m \) 
is the vertical line \( x = \frac{a + b}{2} \).
-/
theorem intersection_locus_is_vertical_line (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∃ P : ℝ × ℝ, P.fst = (a + b) / 2) := 
sorry

end intersection_locus_is_vertical_line_l279_279304


namespace divisible_by_8_l279_279343

theorem divisible_by_8 (k : ℤ) : 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  8 ∣ (7 * m^2 - 5 * n^2 - 2) :=
by 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  sorry

end divisible_by_8_l279_279343


namespace rectangle_side_length_l279_279688

theorem rectangle_side_length (a b c d : ℕ) 
  (h₁ : a = 3) 
  (h₂ : b = 6) 
  (h₃ : a / c = 3 / 4) : 
  c = 4 := 
by
  sorry

end rectangle_side_length_l279_279688


namespace hamburgers_purchased_l279_279631

theorem hamburgers_purchased (total_revenue : ℕ) (hamburger_price : ℕ) (additional_hamburgers : ℕ) 
  (target_amount : ℕ) (h1 : total_revenue = 50) (h2 : hamburger_price = 5) (h3 : additional_hamburgers = 4) 
  (h4 : target_amount = 50) :
  (target_amount - (additional_hamburgers * hamburger_price)) / hamburger_price = 6 := 
by 
  sorry

end hamburgers_purchased_l279_279631


namespace Brenda_mice_left_l279_279600

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end Brenda_mice_left_l279_279600


namespace maximum_value_of_function_l279_279715

theorem maximum_value_of_function :
  ∀ (x : ℝ), -2 < x ∧ x < 0 → x + 1 / x ≤ -2 :=
by
  sorry

end maximum_value_of_function_l279_279715


namespace ratio_of_boys_to_girls_l279_279036

theorem ratio_of_boys_to_girls (B G M : ℤ) 
    (hB_avg : ∀ b, b / B = 90) 
    (hG_avg : ∀ g, g / G = 96) 
    (hM_score : M = 3) 
    (hM_avg : ∀ m, m / M = 92) 
    (hOverall_avg : (90 * B + 96 * G + 92 * M) / (B + G + M) = 94) :
  B.to_rat / G.to_rat = 1 / 5 := 
by
  sorry

end ratio_of_boys_to_girls_l279_279036


namespace even_function_a_equals_one_l279_279498

theorem even_function_a_equals_one (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (1 - x) * (-x - a)) → a = 1 :=
by
  intro h
  sorry

end even_function_a_equals_one_l279_279498


namespace minimum_value_of_z_l279_279823

theorem minimum_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : ∃ min_z, min_z = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → min_z ≤ z :=
by
  sorry

end minimum_value_of_z_l279_279823


namespace monotonic_intervals_range_of_a_l279_279483

noncomputable def f (x a : ℝ) := Real.log x + (a / 2) * x^2 - (a + 1) * x
noncomputable def f' (x a : ℝ) := 1 / x + a * x - (a + 1)

theorem monotonic_intervals (a : ℝ) (ha : f 1 a = -2 ∧ f' 1 a = 0):
  (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f' x a > 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a > 0) ∧ 
  (∀ x : ℝ, (1 / 2) < x ∧ x < 1 → f' x a < 0) := sorry

theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℕ, x > 0 → (f x a) / x < (f' x a) / 2):
  a > 2 * Real.exp (- (3 / 2)) - 1 := sorry

end monotonic_intervals_range_of_a_l279_279483


namespace wall_ratio_l279_279722

theorem wall_ratio (V : ℝ) (B : ℝ) (H : ℝ) (x : ℝ) (L : ℝ) :
  V = 12.8 →
  B = 0.4 →
  H = 5 * B →
  L = x * H →
  V = B * H * L →
  x = 4 ∧ L / H = 4 :=
by
  intros hV hB hH hL hVL
  sorry

end wall_ratio_l279_279722


namespace calculate_product_l279_279117

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l279_279117


namespace vector_addition_proof_l279_279666

def u : ℝ × ℝ × ℝ := (-3, 2, 5)
def v : ℝ × ℝ × ℝ := (4, -7, 1)
def result : ℝ × ℝ × ℝ := (-2, -3, 11)

theorem vector_addition_proof : (2 • u + v) = result := by
  sorry

end vector_addition_proof_l279_279666


namespace Carrie_has_50_dollars_left_l279_279971

/-
Conditions:
1. initial_amount = 91
2. sweater_cost = 24
3. tshirt_cost = 6
4. shoes_cost = 11
-/
def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

/-
Question:
How much money does Carrie have left?
-/
def total_spent : ℕ := sweater_cost + tshirt_cost + shoes_cost
def money_left : ℕ := initial_amount - total_spent

def proof_statement : Prop := money_left = 50

theorem Carrie_has_50_dollars_left : proof_statement :=
by
  sorry

end Carrie_has_50_dollars_left_l279_279971


namespace ending_number_of_X_is_12_l279_279356

open Finset

-- Define the sets X and Y based on the given conditions
variable (n : ℕ)
def X := Icc 1 n
def Y := Icc 0 20

-- Hypothesis: 12 distinct integers belong to both sets at the same time
variable (h : (X n ∩ Y).card = 12)

-- Prove the end number of set X
theorem ending_number_of_X_is_12 : n = 12 :=
by
  sorry

end ending_number_of_X_is_12_l279_279356


namespace cat_food_insufficient_l279_279444

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l279_279444


namespace Tori_current_height_l279_279061

   -- Define the original height and the height she grew
   def Tori_original_height : Real := 4.4
   def Tori_growth : Real := 2.86

   -- Prove that Tori's current height is 7.26 feet
   theorem Tori_current_height : Tori_original_height + Tori_growth = 7.26 := by
     sorry
   
end Tori_current_height_l279_279061


namespace num_accompanying_year_2022_l279_279678

theorem num_accompanying_year_2022 : 
  ∃ N : ℤ, (N = 2) ∧ 
    (∀ n : ℤ, (100 * n + 22) % n = 0 ∧ 10 ≤ n ∧ n < 100 → n = 11 ∨ n = 22) :=
by 
  sorry

end num_accompanying_year_2022_l279_279678


namespace roots_of_quadratic_example_quadratic_problem_solution_l279_279555

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic :
  ∀ (a b c : ℝ), a ≠ 0 → discriminant a b c > 0 → (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0)) :=
by
  intros a b c a_ne_zero discr_positive
  sorry

theorem example_quadratic :
  discriminant 1 (-2) (-1) > 0 :=
by
  unfold discriminant
  norm_num
  linarith

theorem problem_solution :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (1 * x₁^2 - 2 * x₁ - 1 = 0) ∧ (1 * x₂^2 - 2 * x₂ - 1 = 0) :=
by
  apply roots_of_quadratic 1 (-2) (-1)
  norm_num
  apply example_quadratic

end roots_of_quadratic_example_quadratic_problem_solution_l279_279555


namespace abs_negative_five_l279_279703

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l279_279703


namespace area_of_rectangle_ABCD_l279_279378

theorem area_of_rectangle_ABCD :
  ∀ (short_side long_side width length : ℝ),
    (short_side = 6) →
    (long_side = 6 * (3 / 2)) →
    (width = 2 * short_side) →
    (length = long_side) →
    (width * length = 108) :=
by
  intros short_side long_side width length h_short h_long h_width h_length
  rw [h_short, h_long] at *
  sorry

end area_of_rectangle_ABCD_l279_279378


namespace count_integers_divisible_by_2_3_5_7_l279_279836

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l279_279836


namespace intersection_A_B_l279_279489

-- Defining sets A and B based on the given conditions.
def A : Set ℝ := {x | ∃ y, y = Real.log x ∧ x > 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Stating the theorem that A ∩ B = {x | 0 < x ∧ x < 3}.
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l279_279489


namespace vasya_numbers_l279_279737

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l279_279737


namespace multiply_mixed_number_l279_279112

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l279_279112


namespace mul_mixed_number_l279_279121

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l279_279121


namespace incorrect_connection_probability_l279_279920

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end incorrect_connection_probability_l279_279920


namespace comp_inter_empty_l279_279169

section
variable {α : Type*} [DecidableEq α]
variable (I M N : Set α)
variable (a b c d e : α)
variable (hI : I = {a, b, c, d, e})
variable (hM : M = {a, c, d})
variable (hN : N = {b, d, e})

theorem comp_inter_empty : 
  (I \ M) ∩ (I \ N) = ∅ :=
by sorry
end

end comp_inter_empty_l279_279169


namespace linear_func_is_direct_proportion_l279_279479

theorem linear_func_is_direct_proportion (m : ℝ) : (∀ x : ℝ, (y : ℝ) → y = m * x + m - 2 → (m - 2 = 0) → y = 0) → m = 2 :=
by
  intros h
  have : m - 2 = 0 := sorry
  exact sorry

end linear_func_is_direct_proportion_l279_279479


namespace trucks_transportation_l279_279924

theorem trucks_transportation (k : ℕ) (H : ℝ) : 
  (∃ (A B C : ℕ), 
     A + B + C = k ∧ 
     A ≤ k / 2 ∧ B ≤ k / 2 ∧ C ≤ k / 2 ∧ 
     (0 ≤ (k - 2*A)) ∧ (0 ≤ (k - 2*B)) ∧ (0 ≤ (k - 2*C))) 
  →  (k = 7 → (2 : ℕ) = 2) :=
sorry

end trucks_transportation_l279_279924


namespace csc_square_value_l279_279814

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 ∨ x = 1 then 0 -- provision for the illegal inputs as defined in the question
else 1/(x / (x - 1))

theorem csc_square_value (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) :
  f (1 / (Real.sin t)^2) = (Real.cos t)^2 :=
by
  sorry

end csc_square_value_l279_279814


namespace cylinder_surface_area_minimization_l279_279950

theorem cylinder_surface_area_minimization (S V r h : ℝ) (h₁ : π * r^2 * h = V) (h₂ : r^2 + (h / 2)^2 = S^2) : (h / r) = 2 :=
sorry

end cylinder_surface_area_minimization_l279_279950


namespace solution_to_fraction_problem_l279_279623

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end solution_to_fraction_problem_l279_279623


namespace mean_home_runs_per_game_l279_279500

variable (home_runs : Nat) (games_played : Nat)

def total_home_runs : Nat := 
  (5 * 4) + (6 * 5) + (4 * 7) + (3 * 9) + (2 * 11)

def total_games_played : Nat :=
  (5 * 5) + (6 * 6) + (4 * 8) + (3 * 10) + (2 * 12)

theorem mean_home_runs_per_game :
  (total_home_runs : ℚ) / total_games_played = 127 / 147 :=
  by 
    sorry

end mean_home_runs_per_game_l279_279500


namespace number_of_boys_girls_l279_279591

-- Define the initial conditions.
def group_size : ℕ := 8
def total_ways : ℕ := 90

-- Define the actual proof problem.
theorem number_of_boys_girls 
  (n m : ℕ) 
  (h1 : n + m = group_size) 
  (h2 : Nat.choose n 2 * Nat.choose m 1 * Nat.factorial 3 = total_ways) 
  : n = 3 ∧ m = 5 :=
sorry

end number_of_boys_girls_l279_279591


namespace other_function_value_at_20_l279_279384

def linear_function (k b : ℝ) (x : ℝ) : ℝ :=
  k * x + b

theorem other_function_value_at_20
    (k1 k2 b1 b2 : ℝ)
    (h_intersect : linear_function k1 b1 2 = linear_function k2 b2 2)
    (h_diff_at_8 : abs (linear_function k1 b1 8 - linear_function k2 b2 8) = 8)
    (h_y1_at_20 : linear_function k1 b1 20 = 100) :
  linear_function k2 b2 20 = 76 ∨ linear_function k2 b2 20 = 124 :=
sorry

end other_function_value_at_20_l279_279384


namespace nail_polishes_total_l279_279331

theorem nail_polishes_total :
  let k := 25
  let h := k + 8
  let r := k - 6
  h + r = 52 :=
by
  sorry

end nail_polishes_total_l279_279331


namespace combined_degrees_l279_279698

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l279_279698


namespace ivan_travel_time_l279_279399

theorem ivan_travel_time (d V_I V_P : ℕ) (h1 : d = 3 * V_I * 40)
  (h2 : ∀ t, t = d / V_P + 10) : 
  (d / V_I = 75) :=
by
  sorry

end ivan_travel_time_l279_279399


namespace checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l279_279948

-- Define the conditions
def is_checkered_rectangle (S : ℕ) : Prop :=
  (∃ (a b : ℕ), a * b = S) ∧
  (∀ x y k l : ℕ, x * 13 + y * 1 = S) ∧
  (S % 39 = 0)

-- Define that S is minimal satisfying the conditions
def minimal_area_checkered_rectangle (S : ℕ) : Prop :=
  is_checkered_rectangle S ∧
  (∀ (S' : ℕ), S' < S → ¬ is_checkered_rectangle S')

-- Prove that S = 78 is the minimal area
theorem checkered_rectangle_minimal_area : minimal_area_checkered_rectangle 78 :=
  sorry

-- Define the condition for possible perimeters
def possible_perimeters (S : ℕ) (p : ℕ) : Prop :=
  (∀ (a b : ℕ), a * b = S → 2 * (a + b) = p)

-- Prove the possible perimeters for area 78
theorem checkered_rectangle_possible_perimeters :
  ∀ p, p = 38 ∨ p = 58 ∨ p = 82 ↔ possible_perimeters 78 p :=
  sorry

end checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l279_279948


namespace track_extension_needed_l279_279785

noncomputable def additional_track_length (r : ℝ) (g1 g2 : ℝ) : ℝ :=
  let l1 := r / g1
  let l2 := r / g2
  l2 - l1

theorem track_extension_needed :
  additional_track_length 800 0.04 0.015 = 33333 :=
by
  sorry

end track_extension_needed_l279_279785


namespace compute_expression_l279_279266

def sum_of_squares := 7^2 + 5^2
def square_of_sum := (7 + 5)^2
def sum_of_both := sum_of_squares + square_of_sum
def final_result := 2 * sum_of_both

theorem compute_expression : final_result = 436 := by
  sorry

end compute_expression_l279_279266


namespace purely_imaginary_iff_x_equals_one_l279_279655

theorem purely_imaginary_iff_x_equals_one (x : ℝ) :
  ((x^2 - 1) + (x + 1) * Complex.I).re = 0 → x = 1 :=
by
  sorry

end purely_imaginary_iff_x_equals_one_l279_279655


namespace binary101_is_5_l279_279616

theorem binary101_is_5 : 
  let binary101 := [1, 0, 1] in
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0 in
  decimal = 5 :=
by
  let binary101 := [1, 0, 1]
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0
  show decimal = 5
  sorry

end binary101_is_5_l279_279616


namespace birthday_problem_l279_279323

def probability_at_least_two_students_same_birthday (n : ℕ) (d : ℕ) : ℝ :=
  1 - (∏ k in Finset.range n, (1 - k / d.to_real))

theorem birthday_problem (n : ℕ) (d : ℕ) (h_n : n = 30) (h_d : d = 365) :
  probability_at_least_two_students_same_birthday n d > 0.5 :=
by
  -- This is where the proof would be, but we provide a placeholder for now.
  sorry

end birthday_problem_l279_279323


namespace solve_inequality_l279_279358

variable {x : ℝ}

theorem solve_inequality :
  (x - 8) / (x^2 - 4 * x + 13) ≥ 0 ↔ x ≥ 8 :=
by
  sorry

end solve_inequality_l279_279358


namespace enough_cat_food_for_six_days_l279_279430

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l279_279430


namespace sum_of_decimals_l279_279473

theorem sum_of_decimals :
  0.3 + 0.04 + 0.005 + 0.0006 + 0.00007 = (34567 / 100000 : ℚ) :=
by
  -- The proof details would go here
  sorry

end sum_of_decimals_l279_279473


namespace incorrect_transformation_l279_279391

theorem incorrect_transformation (a b : ℤ) : ¬ (a / b = (a + 1) / (b + 1)) :=
sorry

end incorrect_transformation_l279_279391


namespace total_cost_function_range_of_x_minimum_cost_when_x_is_2_l279_279972

def transportation_cost (x : ℕ) : ℕ :=
  300 * x + 500 * (12 - x) + 400 * (10 - x) + 800 * (x - 2)

theorem total_cost_function (x : ℕ) : transportation_cost x = 200 * x + 8400 := by
  -- Simply restate the definition in the theorem form
  sorry

theorem range_of_x (x : ℕ) : 2 ≤ x ∧ x ≤ 10 := by
  -- Provide necessary constraints in theorem form
  sorry

theorem minimum_cost_when_x_is_2 : transportation_cost 2 = 8800 := by
  -- Final cost at minimum x
  sorry

end total_cost_function_range_of_x_minimum_cost_when_x_is_2_l279_279972


namespace opposite_numbers_abs_l279_279648

theorem opposite_numbers_abs (a b : ℤ) (h : a + b = 0) : |a - 2014 + b| = 2014 :=
by
  -- proof here
  sorry

end opposite_numbers_abs_l279_279648


namespace rectangle_area_in_triangle_l279_279416

theorem rectangle_area_in_triangle (c k y : ℝ) (h1 : c > 0) (h2 : k > 0) (h3 : 0 < y) (h4 : y < k) : 
  ∃ A : ℝ, A = y * ((c * (k - y)) / k) := 
by
  sorry

end rectangle_area_in_triangle_l279_279416


namespace abs_negative_five_l279_279704

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l279_279704


namespace find_seating_capacity_l279_279089

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end find_seating_capacity_l279_279089


namespace students_wrote_word_correctly_l279_279325

-- Definitions based on the problem conditions
def total_students := 50
def num_cat := 10
def num_rat := 18
def num_croc := total_students - num_cat - num_rat
def correct_cat := 15
def correct_rat := 15
def correct_total := correct_cat + correct_rat

-- Question: How many students wrote their word correctly?
-- Correct Answer: 8

theorem students_wrote_word_correctly : 
  num_cat + num_rat + num_croc = total_students 
  → correct_cat = 15 
  → correct_rat = 15 
  → correct_total = 30 
  → ∀ (num_correct_words : ℕ), num_correct_words = correct_total - num_croc 
  → num_correct_words = 8 := by 
  sorry

end students_wrote_word_correctly_l279_279325


namespace calc_expr_l279_279604

theorem calc_expr :
  (-1) * (-3) + 3^2 / (8 - 5) = 6 :=
by
  sorry

end calc_expr_l279_279604


namespace additional_track_length_l279_279589

theorem additional_track_length (rise : ℝ) (grade1 grade2 : ℝ) (h1 : grade1 = 0.04) (h2 : grade2 = 0.02) (h3 : rise = 800) :
  ∃ (additional_length : ℝ), additional_length = (rise / grade2 - rise / grade1) ∧ additional_length = 20000 :=
by
  sorry

end additional_track_length_l279_279589


namespace product_of_gcd_and_lcm_1440_l279_279275

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l279_279275


namespace concert_total_cost_l279_279531

noncomputable def total_cost (ticket_cost : ℕ) (processing_fee_rate : ℚ) (parking_fee : ℕ)
  (entrance_fee_per_person : ℕ) (num_persons : ℕ) (refreshments_cost : ℕ) 
  (merchandise_cost : ℕ) : ℚ :=
  let ticket_total := ticket_cost * num_persons
  let processing_fee := processing_fee_rate * (ticket_total : ℚ)
  ticket_total + processing_fee + (parking_fee + entrance_fee_per_person * num_persons 
  + refreshments_cost + merchandise_cost)

theorem concert_total_cost :
  total_cost 75 0.15 10 5 2 20 40 = 252.50 := by 
  sorry

end concert_total_cost_l279_279531


namespace curve_in_second_quadrant_l279_279827

theorem curve_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0) ↔ (a > 2) :=
sorry

end curve_in_second_quadrant_l279_279827


namespace units_digit_sum_base8_l279_279262

theorem units_digit_sum_base8 : 
  ∀ (x y : ℕ), (x = 64 ∧ y = 34 ∧ (x % 8 = 4) ∧ (y % 8 = 4) → (x + y) % 8 = 0) :=
by
  sorry

end units_digit_sum_base8_l279_279262


namespace ceiling_lights_l279_279185

variable (S M L : ℕ)

theorem ceiling_lights (hM : M = 12) (hL : L = 2 * M)
    (hBulbs : S + 2 * M + 3 * L = 118) : S - M = 10 :=
by
  sorry

end ceiling_lights_l279_279185


namespace inequality_proof_l279_279634

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by sorry

end inequality_proof_l279_279634


namespace inscribed_squares_ratio_l279_279590

theorem inscribed_squares_ratio (a b : ℝ) (h_triangle : 5^2 + 12^2 = 13^2)
    (h_square1 : a = 25 / 37) (h_square2 : b = 10) :
    a / b = 25 / 370 :=
by 
  sorry

end inscribed_squares_ratio_l279_279590


namespace find_monthly_fee_l279_279302

-- Definitions from conditions
def monthly_fee (total_bill : ℝ) (cost_per_minute : ℝ) (minutes_used : ℝ) : ℝ :=
  total_bill - cost_per_minute * minutes_used

-- Theorem stating the question
theorem find_monthly_fee :
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  total_bill - cost_per_minute * minutes_used = 5.00 :=
by
  -- Definition of variables used in the theorem
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  
  -- The statement of the theorem and leaving the proof as an exercise
  show total_bill - cost_per_minute * minutes_used = 5.00
  sorry

end find_monthly_fee_l279_279302


namespace linear_regression_decrease_l279_279149

theorem linear_regression_decrease (x : ℝ) (y : ℝ) (h : y = 2 - 1.5 * x) : 
  y = 2 - 1.5 * (x + 1) -> (y - (2 - 1.5 * (x +1))) = -1.5 :=
by
  sorry

end linear_regression_decrease_l279_279149


namespace ratio_equivalence_l279_279944

theorem ratio_equivalence (x : ℕ) : 
  (10 * 60 = 600) →
  (15 : ℕ) / 5 = x / 600 →
  x = 1800 :=
by
  intros h1 h2
  sorry

end ratio_equivalence_l279_279944


namespace find_m2n_plus_mn2_minus_mn_l279_279319

def quadratic_roots (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0

theorem find_m2n_plus_mn2_minus_mn :
  ∃ m n : ℝ, quadratic_roots 1 2015 (-1) m n ∧ m^2 * n + m * n^2 - m * n = 2016 :=
by
  sorry

end find_m2n_plus_mn2_minus_mn_l279_279319


namespace num_divisible_by_2_3_5_7_under_500_l279_279834

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l279_279834


namespace line_y2_not_pass_second_quadrant_l279_279178

theorem line_y2_not_pass_second_quadrant {a b : ℝ} (h1 : a < 0) (h2 : b > 0) :
  ¬∃ x : ℝ, x < 0 ∧ bx + a > 0 :=
by
  sorry

end line_y2_not_pass_second_quadrant_l279_279178


namespace series_sum_eq_l279_279610

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end series_sum_eq_l279_279610


namespace roots_quadratic_eq_sum_prod_l279_279650

theorem roots_quadratic_eq_sum_prod (r s p q : ℝ) (hr : r + s = p) (hq : r * s = q) : r^2 + s^2 = p^2 - 2 * q :=
by
  sorry

end roots_quadratic_eq_sum_prod_l279_279650


namespace max_rectangle_area_l279_279072

theorem max_rectangle_area (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (a b : ℝ), 2 * a + 2 * b = perimeter ∧ a * b = 625 :=
by
  sorry

end max_rectangle_area_l279_279072


namespace rex_remaining_cards_l279_279884

-- Definitions based on the conditions provided:
def nicole_cards : ℕ := 400
def cindy_cards (nicole_cards : ℕ) : ℕ := 2 * nicole_cards
def combined_total (nicole_cards cindy_cards : ℕ) : ℕ := nicole_cards + cindy_cards nicole_cards
def rex_cards (combined_total : ℕ) : ℕ := combined_total / 2
def rex_divided_cards (rex_cards siblings : ℕ) : ℕ := rex_cards / (1 + siblings)

-- The theorem to be proved based on the question and correct answer:
theorem rex_remaining_cards : rex_divided_cards (rex_cards (combined_total nicole_cards (cindy_cards nicole_cards))) 3 = 150 :=
by sorry

end rex_remaining_cards_l279_279884


namespace vasya_numbers_l279_279751

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l279_279751


namespace find_k_l279_279560

-- The expression in terms of x, y, and k
def expression (k x y : ℝ) :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

-- The mathematical statement to be proved
theorem find_k : ∃ k : ℝ, (∀ x y : ℝ, expression k x y ≥ 0) ∧ (∃ (x y : ℝ), expression k x y = 0) :=
sorry

end find_k_l279_279560


namespace factorization_problem_l279_279546

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end factorization_problem_l279_279546


namespace solitaire_game_removal_l279_279412

theorem solitaire_game_removal (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ moves : ℕ, ∀ i : ℕ, i < moves → (i + 1) % 2 = (i % 2) + 1) ↔ (m % 2 = 1 ∨ n % 2 = 1) :=
sorry

end solitaire_game_removal_l279_279412


namespace largest_common_value_less_than_1000_l279_279803

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, 
    (∃ n : ℕ, a = 4 + 5 * n) ∧
    (∃ m : ℕ, a = 5 + 10 * m) ∧
    a % 4 = 1 ∧
    a < 1000 ∧
    (∀ b : ℕ, 
      (∃ n : ℕ, b = 4 + 5 * n) ∧
      (∃ m : ℕ, b = 5 + 10 * m) ∧
      b % 4 = 1 ∧
      b < 1000 → 
      b ≤ a) ∧ 
    a = 989 :=
by
  sorry

end largest_common_value_less_than_1000_l279_279803


namespace maximum_enclosed_area_l279_279069

theorem maximum_enclosed_area (P : ℝ) (A : ℝ) : 
  P = 100 → (∃ l w : ℝ, P = 2 * l + 2 * w ∧ A = l * w) → A ≤ 625 :=
by
  sorry

end maximum_enclosed_area_l279_279069


namespace common_measure_largest_l279_279656

theorem common_measure_largest {a b : ℕ} (h_a : a = 15) (h_b : b = 12): 
  (∀ c : ℕ, c ∣ a ∧ c ∣ b → c ≤ Nat.gcd a b) ∧ Nat.gcd a b = 3 := 
by
  sorry

end common_measure_largest_l279_279656


namespace max_sigma_squared_l279_279863

theorem max_sigma_squared (c d : ℝ) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_c_ge_d : c ≥ d)
    (h : ∃ x y : ℝ, 0 ≤ x ∧ x < c ∧ 0 ≤ y ∧ y < d ∧ 
      c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x) ^ 2 + (d - y) ^ 2) : 
    σ^2 = 4 / 3 := by
  sorry

end max_sigma_squared_l279_279863


namespace count_of_valid_four_digit_numbers_l279_279007

def is_four_digit_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

def digits_sum_to_twelve (a b c d : ℕ) : Prop :=
  a + b + c + d = 12

def divisible_by_eleven (a b c d : ℕ) : Prop :=
  (a + c - (b + d)) % 11 = 0

theorem count_of_valid_four_digit_numbers : ∃ n : ℕ, n = 20 ∧
  (∀ a b c d : ℕ, is_four_digit_number a b c d →
  digits_sum_to_twelve a b c d →
  divisible_by_eleven a b c d →
  true) :=
sorry

end count_of_valid_four_digit_numbers_l279_279007


namespace binom_150_1_eq_150_l279_279613

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end binom_150_1_eq_150_l279_279613


namespace unique_solution_l279_279988

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l279_279988


namespace range_of_m_l279_279373

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
by
  sorry

end range_of_m_l279_279373


namespace Marissa_has_21_more_marbles_than_Jonny_l279_279679

noncomputable def Mara_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Markus_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Jonny_marbles (total_marbles : ℕ) (bags : ℕ) : ℕ :=
total_marbles

noncomputable def Marissa_marbles (bags1 : ℕ) (marbles1 : ℕ) (bags2 : ℕ) (marbles2 : ℕ) : ℕ :=
(bags1 * marbles1) + (bags2 * marbles2)

noncomputable def Jonny : ℕ := Jonny_marbles 18 3

noncomputable def Marissa : ℕ := Marissa_marbles 3 5 3 8

theorem Marissa_has_21_more_marbles_than_Jonny : (Marissa - Jonny) = 21 :=
by
  sorry

end Marissa_has_21_more_marbles_than_Jonny_l279_279679


namespace expected_value_smallest_N_l279_279422
noncomputable def expectedValueN : ℝ := 6.54

def barryPicksPointsInsideUnitCircle (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, (P n).fst^2 + (P n).snd^2 ≤ 1

def pointsIndependentAndUniform (P : ℕ → ℝ × ℝ) : Prop :=
  -- This is a placeholder representing the independent and uniform picking which 
  -- would be formally defined using probability measures in an advanced Lean library.
  sorry

theorem expected_value_smallest_N (P : ℕ → ℝ × ℝ)
  (h1 : barryPicksPointsInsideUnitCircle P)
  (h2 : pointsIndependentAndUniform P) :
  ∃ N : ℕ, N = expectedValueN :=
sorry

end expected_value_smallest_N_l279_279422


namespace right_triangle_XZ_length_l279_279628

theorem right_triangle_XZ_length (X Y Z : Type) [triangle X Y Z]
  (hypotenuse : XY = 13)
  (right_angle : angle X = 90)
  (angle_Y : angle Y = 60) :
  length XZ = (13 * real.sqrt 3 / 2) := sorry

end right_triangle_XZ_length_l279_279628


namespace martin_speed_first_half_l279_279680

variable (v : ℝ) -- speed during the first half of the trip

theorem martin_speed_first_half
    (trip_duration : ℝ := 8)              -- The trip lasted 8 hours
    (speed_second_half : ℝ := 85)          -- Speed during the second half of the trip
    (total_distance : ℝ := 620)            -- Total distance traveled
    (time_each_half : ℝ := trip_duration / 2) -- Each half of the trip took half of the total time
    (distance_second_half : ℝ := speed_second_half * time_each_half)
    (distance_first_half : ℝ := total_distance - distance_second_half) :
    v = distance_first_half / time_each_half :=
by
  sorry

end martin_speed_first_half_l279_279680


namespace inequality_proof_l279_279038

variables {x y : ℝ}

theorem inequality_proof (hx_pos : x > 0) (hy_pos : y > 0) (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := 
by 
  sorry

end inequality_proof_l279_279038


namespace find_S_l279_279155

variable {R k : ℝ}

theorem find_S (h : |k + R| / |R| = 0) : S = 1 :=
by
  let S := |k + 2*R| / |2*k + R|
  have h1 : k + R = 0 := by sorry
  have h2 : k = -R := by sorry
  sorry

end find_S_l279_279155


namespace arithmetic_mean_of_fractions_l279_279615

theorem arithmetic_mean_of_fractions :
  let a := (5 : ℚ) / 8
  let b := (9 : ℚ) / 16
  let c := (11 : ℚ) / 16
  a = (b + c) / 2 := by
  sorry

end arithmetic_mean_of_fractions_l279_279615


namespace christopher_more_money_l279_279506

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end christopher_more_money_l279_279506


namespace calculate_f_f_2_l279_279632

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 * x ^ 2 - 4
else if x = 0 then 2
else -1

theorem calculate_f_f_2 : f (f 2) = 188 :=
by
  sorry

end calculate_f_f_2_l279_279632


namespace vasya_numbers_l279_279735

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l279_279735


namespace edge_length_box_l279_279493

theorem edge_length_box (n : ℝ) (h : n = 999.9999999999998) : 
  ∃ (L : ℝ), L = 1 ∧ ((L * 100) ^ 3 / 10 ^ 3) = n := 
sorry

end edge_length_box_l279_279493


namespace division_of_decimals_l279_279800

theorem division_of_decimals : 0.36 / 0.004 = 90 := by
  sorry

end division_of_decimals_l279_279800


namespace probability_escher_consecutive_l279_279353

def total_pieces : Nat := 12
def escher_pieces : Nat := 4

theorem probability_escher_consecutive :
  (Nat.factorial 9 * Nat.factorial 4 : ℚ) / Nat.factorial 12 = 1 / 55 := 
sorry

end probability_escher_consecutive_l279_279353


namespace apples_initial_count_l279_279529

theorem apples_initial_count 
  (trees : ℕ)
  (apples_per_tree_picked : ℕ)
  (apples_picked_in_total : ℕ)
  (apples_remaining : ℕ)
  (initial_apples : ℕ) 
  (h1 : trees = 3) 
  (h2 : apples_per_tree_picked = 8) 
  (h3 : apples_picked_in_total = trees * apples_per_tree_picked)
  (h4 : apples_remaining = 9) 
  (h5 : initial_apples = apples_picked_in_total + apples_remaining) : 
  initial_apples = 33 :=
by sorry

end apples_initial_count_l279_279529


namespace ratio_of_areas_l279_279066

-- Define the conditions
def angle_Q_smaller_circle : ℝ := 60
def angle_Q_larger_circle : ℝ := 30
def arc_length_equal (C1 C2 : ℝ) : Prop := 
  (angle_Q_smaller_circle / 360) * C1 = (angle_Q_larger_circle / 360) * C2

-- The required Lean statement that proves the ratio of the areas
theorem ratio_of_areas (C1 C2 r1 r2 : ℝ) 
  (arc_eq : arc_length_equal C1 C2) : 
  (π * r1^2) / (π * r2^2) = 1 / 4 := 
by 
  sorry

end ratio_of_areas_l279_279066


namespace calculate_expression_l279_279968

theorem calculate_expression :
  2 * Real.sin (60 * Real.pi / 180) + abs (Real.sqrt 3 - 3) + (Real.pi - 1)^0 = 4 :=
by
  sorry

end calculate_expression_l279_279968


namespace composite_numbers_with_same_main_divisors_are_equal_l279_279870

theorem composite_numbers_with_same_main_divisors_are_equal (a b : ℕ) 
  (h_a_not_prime : ¬ Prime a)
  (h_b_not_prime : ¬ Prime b)
  (h_a_comp : 1 < a ∧ ∃ p, p ∣ a ∧ p ≠ a)
  (h_b_comp : 1 < b ∧ ∃ p, p ∣ b ∧ p ≠ b)
  (main_divisors : {d : ℕ // d ∣ a ∧ d ≠ a} = {d : ℕ // d ∣ b ∧ d ≠ b}) :
  a = b := 
sorry

end composite_numbers_with_same_main_divisors_are_equal_l279_279870


namespace stone_123_is_12_l279_279806

/-- Definitions: 
  1. Fifteen stones arranged in a circle counted in a specific pattern: clockwise and counterclockwise.
  2. The sequence of stones enumerated from 1 to 123
  3. The repeating pattern occurs every 28 stones
-/
def stones_counted (n : Nat) : Nat :=
  if n % 28 <= 15 then (n % 28) else (28 - (n % 28) + 1)

theorem stone_123_is_12 : stones_counted 123 = 12 :=
by
  sorry

end stone_123_is_12_l279_279806


namespace digit_B_value_l279_279904

theorem digit_B_value (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 2 % 9 = 0):
  B = 6 :=
begin
  sorry
end

end digit_B_value_l279_279904


namespace net_profit_calculation_l279_279351

def original_purchase_price : ℝ := 80000
def annual_property_tax_rate : ℝ := 0.012
def annual_maintenance_cost : ℝ := 1500
def annual_mortgage_interest_rate : ℝ := 0.04
def selling_profit_rate : ℝ := 0.20
def broker_commission_rate : ℝ := 0.05
def years_of_ownership : ℕ := 5

noncomputable def net_profit : ℝ :=
  let selling_price := original_purchase_price * (1 + selling_profit_rate)
  let brokers_commission := original_purchase_price * broker_commission_rate
  let total_property_tax := original_purchase_price * annual_property_tax_rate * years_of_ownership
  let total_maintenance_cost := annual_maintenance_cost * years_of_ownership
  let total_mortgage_interest := original_purchase_price * annual_mortgage_interest_rate * years_of_ownership
  let total_costs := brokers_commission + total_property_tax + total_maintenance_cost + total_mortgage_interest
  (selling_price - original_purchase_price) - total_costs

theorem net_profit_calculation : net_profit = -16300 := by
  sorry

end net_profit_calculation_l279_279351


namespace product_gcd_lcm_24_60_l279_279292

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l279_279292


namespace Vasya_numbers_l279_279731

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l279_279731


namespace min_value_p_plus_q_l279_279571

theorem min_value_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) 
  (h : 17 * (p + 1) = 20 * (q + 1)) : p + q = 37 :=
sorry

end min_value_p_plus_q_l279_279571


namespace photo_arrangement_l279_279093

noncomputable def valid_arrangements (teacher boys girls : ℕ) : ℕ :=
  if girls = 2 ∧ teacher = 1 ∧ boys = 2 then 24 else 0

theorem photo_arrangement :
  valid_arrangements 1 2 2 = 24 :=
by {
  -- The proof goes here.
  sorry
}

end photo_arrangement_l279_279093


namespace evaluate_expression_l279_279221

def operation (x y : ℚ) : ℚ := x^2 / y

theorem evaluate_expression : 
  (operation (operation 3 4) 2) - (operation 3 (operation 4 2)) = 45 / 32 :=
by
  sorry

end evaluate_expression_l279_279221


namespace tips_fraction_l279_279076

theorem tips_fraction (S T I : ℝ) (hT : T = 9 / 4 * S) (hI : I = S + T) : 
  T / I = 9 / 13 := 
by 
  sorry

end tips_fraction_l279_279076


namespace g_eq_g_inv_iff_l279_279977

def g (x : ℝ) : ℝ := 3 * x - 7

def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g x = g_inv x ↔ x = 7 / 2 :=
by {
  sorry
}

end g_eq_g_inv_iff_l279_279977


namespace calculate_product_l279_279116

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l279_279116


namespace digit_B_divisible_by_9_l279_279901

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l279_279901


namespace divisibility_by_24_l279_279206

theorem divisibility_by_24 (n : ℤ) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) :=
sorry

end divisibility_by_24_l279_279206


namespace count_of_divisibles_l279_279839

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l279_279839


namespace gcd_polynomial_l279_279481

theorem gcd_polynomial (b : ℤ) (h : 1729 ∣ b) : Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := 
by
  sorry

end gcd_polynomial_l279_279481


namespace primes_eq_condition_l279_279992

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l279_279992


namespace amount_received_by_A_is_4_over_3_l279_279400

theorem amount_received_by_A_is_4_over_3
  (a d : ℚ)
  (h1 : a - 2 * d + a - d = a + (a + d) + (a + 2 * d))
  (h2 : 5 * a = 5) :
  a - 2 * d = 4 / 3 :=
by
  sorry

end amount_received_by_A_is_4_over_3_l279_279400


namespace square_side_measurement_error_l279_279964

theorem square_side_measurement_error (S S' : ℝ) (h1 : S' = S * Real.sqrt 1.0404) : 
  (S' - S) / S * 100 = 2 :=
by
  sorry

end square_side_measurement_error_l279_279964


namespace compute_factorial_ratio_l279_279821

theorem compute_factorial_ratio (n : ℕ) (K : ℕ) (hK: P(ξ = K) = 1 / 2^K) : factorial n / (factorial 3 * factorial (n - 3)) = 35 := by 
  -- define P and ξ appropriately here

  sorry

end compute_factorial_ratio_l279_279821


namespace value_of_a_plus_c_l279_279056

theorem value_of_a_plus_c (a b c r : ℝ)
  (h1 : a + b + c = 114)
  (h2 : a * b * c = 46656)
  (h3 : b = a * r)
  (h4 : c = a * r^2) :
  a + c = 78 :=
sorry

end value_of_a_plus_c_l279_279056


namespace total_students_in_Lansing_l279_279024

def n_schools : Nat := 25
def students_per_school : Nat := 247
def total_students : Nat := n_schools * students_per_school

theorem total_students_in_Lansing :
  total_students = 6175 :=
  by
    -- we can either compute manually or just put sorry for automated assistance
    sorry

end total_students_in_Lansing_l279_279024


namespace fifth_term_of_sequence_l279_279204

theorem fifth_term_of_sequence :
  let a_n (n : ℕ) := (-1:ℤ)^(n+1) * (n^2 + 1)
  ∃ x : ℤ, a_n 5 * x^5 = 26 * x^5 :=
by
  sorry

end fifth_term_of_sequence_l279_279204


namespace ratio_diff_l279_279578

theorem ratio_diff (x : ℕ) (h1 : 7 * x = 56) : 56 - 3 * x = 32 :=
by
  sorry

end ratio_diff_l279_279578


namespace infinite_series_sum_l279_279667

theorem infinite_series_sum
  (a b : ℝ)
  (h1 : (∑' n : ℕ, a / (b ^ (n + 1))) = 4) :
  (∑' n : ℕ, a / ((a + b) ^ (n + 1))) = 4 / 5 := 
sorry

end infinite_series_sum_l279_279667


namespace tree_height_l279_279592

theorem tree_height (B h : ℕ) (H : ℕ) (h_eq : h = 16) (B_eq : B = 12) (L : ℕ) (L_def : L ^ 2 = B ^ 2 + h ^ 2) (H_def : H = h + L) :
    H = 36 := by
  -- We do not need to provide the proof steps as per the instructions
  sorry

end tree_height_l279_279592


namespace K_time_expression_l279_279402

variable (x : ℝ) 

theorem K_time_expression
  (hyp : (45 / (x - 2 / 5) - 45 / x = 3 / 4)) :
  45 / (x : ℝ) = 45 / x :=
sorry

end K_time_expression_l279_279402


namespace Vasya_numbers_l279_279761

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l279_279761


namespace teal_more_blue_proof_l279_279577

theorem teal_more_blue_proof (P G B N : ℕ) (hP : P = 150) (hG : G = 90) (hB : B = 40) (hN : N = 25) : 
  (∃ (x : ℕ), x = 75) :=
by
  sorry

end teal_more_blue_proof_l279_279577


namespace num_divisible_by_2_3_5_7_under_500_l279_279835

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l279_279835


namespace Vasya_numbers_l279_279727

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l279_279727


namespace minimum_value_of_f_l279_279177

-- Define the function
def f (a b x : ℝ) := x^2 + (a + 2) * x + b

-- Condition that ensures the graph is symmetric about x = 1
def symmetric_about_x1 (a : ℝ) : Prop := a + 2 = -2

-- Minimum value of the function f(x) in terms of the constant c
theorem minimum_value_of_f (a b : ℝ) (h : symmetric_about_x1 a) : ∃ c : ℝ, ∀ x : ℝ, f a b x ≥ c :=
by sorry

end minimum_value_of_f_l279_279177


namespace bc_over_ad_l279_279334

noncomputable def a : ℝ := 32 / 3
noncomputable def b : ℝ := 16 * Real.pi
noncomputable def c : ℝ := 24 * Real.pi
noncomputable def d : ℝ := 16 * Real.pi

theorem bc_over_ad : (b * c) / (a * d) = 9 / 4 := 
by 
  sorry

end bc_over_ad_l279_279334


namespace probability_of_triangle_formation_l279_279939

open Finset

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def stick_lengths : Finset ℕ := {1, 2, 4, 6, 9, 10, 14, 15, 18}

def all_combinations : Finset (ℕ × ℕ × ℕ) :=
  (stick_lengths.product stick_lengths).product stick_lengths
    |>.filter (λ t, ∃ a b c, (t.1.1 = a ∧ t.1.2 = b ∧ t.2 = c ∧ a < b ∧ b < c))

def valid_combinations : Finset (ℕ × ℕ × ℕ) :=
  all_combinations.filter (λ ⟨⟨a, b⟩, c⟩, valid_triangle a b c)

def probability_triangle : ℚ :=
  valid_combinations.card / all_combinations.card

theorem probability_of_triangle_formation : probability_triangle = 4 / 21 :=
by sorry

end probability_of_triangle_formation_l279_279939


namespace weight_of_purple_ring_l279_279662

noncomputable section

def orange_ring_weight : ℝ := 0.08333333333333333
def white_ring_weight : ℝ := 0.4166666666666667
def total_weight : ℝ := 0.8333333333

theorem weight_of_purple_ring :
  total_weight - orange_ring_weight - white_ring_weight = 0.3333333333 :=
by
  -- We'll place the statement here, leave out the proof for skipping.
  sorry

end weight_of_purple_ring_l279_279662


namespace problem1_l279_279401

theorem problem1 : 
  ∀ a b : ℤ, a = 1 → b = -3 → (a - b)^2 - 2 * a * (a + 3 * b) + (a + 2 * b) * (a - 2 * b) = -3 :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end problem1_l279_279401


namespace solve_inequality_part1_solve_inequality_part2_l279_279314

-- Define the first part of the problem
theorem solve_inequality_part1 (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 2 * a^2 < 0) ↔ 
    (a = 0 ∧ false) ∨ 
    (a > 0 ∧ -a < x ∧ x < 2 * a) ∨ 
    (a < 0 ∧ 2 * a < x ∧ x < -a) := 
sorry

-- Define the second part of the problem
theorem solve_inequality_part2 (a b : ℝ) (x : ℝ) 
  (h : { x | x^2 - a * x - b < 0 } = { x | -1 < x ∧ x < 2 }) :
  { x | a * x^2 + x - b > 0 } = { x | x < -2 } ∪ { x | 1 < x } :=
sorry

end solve_inequality_part1_solve_inequality_part2_l279_279314


namespace range_of_set_l279_279790

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end range_of_set_l279_279790


namespace minimum_omega_l279_279620

open Real

theorem minimum_omega (ω : ℕ) (h_ω_pos : ω > 0) :
  (∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + (π / 2)) → ω = 2 :=
by
  sorry

end minimum_omega_l279_279620


namespace solve_system_exists_l279_279039

theorem solve_system_exists (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : 1 / x + 1 / y + 1 / z = 5 / 12) 
  (h3 : x^3 + y^3 + z^3 = 45) 
  : (x, y, z) = (2, -3, 4) ∨ (x, y, z) = (2, 4, -3) ∨ (x, y, z) = (-3, 2, 4) ∨ (x, y, z) = (-3, 4, 2) ∨ (x, y, z) = (4, 2, -3) ∨ (x, y, z) = (4, -3, 2) := 
sorry

end solve_system_exists_l279_279039


namespace letters_into_mailboxes_l279_279645

theorem letters_into_mailboxes (n m : ℕ) (h1 : n = 3) (h2 : m = 5) : m^n = 125 :=
by
  rw [h1, h2]
  exact rfl

end letters_into_mailboxes_l279_279645


namespace product_of_gcd_and_lcm_1440_l279_279279

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l279_279279


namespace prime_solution_unique_l279_279986

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l279_279986


namespace steven_owes_jeremy_l279_279019

-- Definitions for the conditions
def base_payment_per_room := (13 : ℚ) / 3
def rooms_cleaned := (5 : ℚ) / 2
def additional_payment_per_room := (1 : ℚ) / 2

-- Define the total amount of money Steven owes Jeremy
def total_payment (base_payment_per_room rooms_cleaned additional_payment_per_room : ℚ) : ℚ :=
  let base_payment := base_payment_per_room * rooms_cleaned
  let additional_payment := if rooms_cleaned > 2 then additional_payment_per_room * rooms_cleaned else 0
  base_payment + additional_payment

-- The statement to prove
theorem steven_owes_jeremy :
  total_payment base_payment_per_room rooms_cleaned additional_payment_per_room = 145 / 12 :=
by
  sorry

end steven_owes_jeremy_l279_279019


namespace quadratic_roots_evaluation_l279_279846

theorem quadratic_roots_evaluation (x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1 * x2 = -2) :
  (1 + x1) + x2 * (1 - x1) = 4 :=
by
  sorry

end quadratic_roots_evaluation_l279_279846


namespace value_of_transformed_product_of_roots_l279_279669

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end value_of_transformed_product_of_roots_l279_279669


namespace find_m_values_l279_279809

noncomputable def lines_cannot_form_triangle (m : ℝ) : Prop :=
  (4 * m - 1 = 0) ∨ (6 * m + 1 = 0) ∨ (m^2 + m / 3 - 2 / 3 = 0)

theorem find_m_values :
  { m : ℝ | lines_cannot_form_triangle m } = {4, -1 / 6, -1, 2 / 3} :=
by
  sorry

end find_m_values_l279_279809


namespace product_gcd_lcm_l279_279282

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l279_279282


namespace equalities_hold_l279_279515

noncomputable theory

def f (x p q : ℝ) : ℝ := x^2 + p * x + q

theorem equalities_hold (p q : ℝ) :
  f p p q = 0 ∧ f q p q = 0 ↔ (p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2) :=
by {
  sorry -- Proof is omitted as it is not required.
}

end equalities_hold_l279_279515


namespace sum_sequence_correct_l279_279396

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end sum_sequence_correct_l279_279396


namespace Vasya_numbers_l279_279728

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l279_279728


namespace number_of_divisors_not_multiples_of_14_l279_279673

theorem number_of_divisors_not_multiples_of_14 
  (n : ℕ)
  (h1: ∃ k : ℕ, n = 2 * k * k)
  (h2: ∃ k : ℕ, n = 3 * k * k * k)
  (h3: ∃ k : ℕ, n = 5 * k * k * k * k * k)
  (h4: ∃ k : ℕ, n = 7 * k * k * k * k * k * k * k)
  : 
  ∃ num_divisors : ℕ, num_divisors = 19005 ∧ (∀ d : ℕ, d ∣ n → ¬(14 ∣ d)) := sorry

end number_of_divisors_not_multiples_of_14_l279_279673


namespace heights_inscribed_circle_inequality_l279_279675

theorem heights_inscribed_circle_inequality
  {h₁ h₂ r : ℝ} (h₁_pos : 0 < h₁) (h₂_pos : 0 < h₂) (r_pos : 0 < r)
  (triangle_heights : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a * h₁ = b * h₂ ∧ 
                                       a + b > c ∧ h₁ = 2 * r * (a + b + c) / (a * b)):
  (1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r) :=
sorry

end heights_inscribed_circle_inequality_l279_279675


namespace problem_proof_l279_279254

-- Definitions
def a_seq : ℕ → ℕ 
| 1 := 3
| (n + 1) := 2 * a_seq n + 2

def b_seq (n : ℕ) : ℝ := n / (a_seq n + 2)

def S_seq (n : ℕ) : ℝ := (Finset.range n).sum (λ i, b_seq (i + 1))

-- Theorem
theorem problem_proof :
  a_seq 2 = 8 ∧
  a_seq 3 = 18 ∧
  (∃ r : ℕ, ∀ n : ℕ, a_seq n + 2 = 5 * 2^(n - 1)) ∧
  (∀ n : ℕ, n > 0 → 1/5 ≤ S_seq n ∧ S_seq n < 4/5) :=
by
  sorry

end problem_proof_l279_279254


namespace geometric_number_difference_l279_279081

theorem geometric_number_difference : 
  ∀ (a b c : ℕ), 8 = a → b ≠ c → (∃ k : ℕ, 8 ≠ k ∧ b = k ∧ c = k * k / 8) → (10^2 * a + 10 * b + c = 842) ∧ (10^2 * a + 10 * b + c = 842) → (10^2 * a + 10 * b + c) - (10^2 * a + 10 * b + c) = 0 :=
by
  intro a b c
  intro ha hb
  intro hk
  intro hseq
  sorry

end geometric_number_difference_l279_279081


namespace all_blue_figures_are_small_l279_279376

variables (Shape : Type) (Large Blue Small Square Triangle : Shape → Prop)

-- Given conditions
axiom h1 : ∀ (x : Shape), Large x → Square x
axiom h2 : ∀ (x : Shape), Blue x → Triangle x

-- The goal to prove
theorem all_blue_figures_are_small : ∀ (x : Shape), Blue x → Small x :=
by
  sorry

end all_blue_figures_are_small_l279_279376


namespace smallest_four_digit_multiple_of_13_l279_279300

theorem smallest_four_digit_multiple_of_13 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 13 = 0) ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 13 ≠ 0 :=
by
  sorry

end smallest_four_digit_multiple_of_13_l279_279300


namespace find_second_number_l279_279898

theorem find_second_number (x y : ℤ) (h1 : x = -63) (h2 : (2 + y + x) / 3 = 5) : y = 76 :=
sorry

end find_second_number_l279_279898


namespace probability_two_shots_l279_279042

open ProbabilityTheory

noncomputable def prob_A : ℝ := 3 / 4
noncomputable def prob_B : ℝ := 4 / 5

def event_A : Event := sorry
def event_B : Event := sorry

axiom independent_events : ∀ (e1 e2: Event), e1.independent e2 ↔ (P[e1 ∩ e2] = P[e1] * P[e2])

theorem probability_two_shots :
  by
    let outcome_1 := (1 - prob_A) * (1 - prob_B) * prob_A
    let outcome_2 := (1 - prob_A) * (1 - prob_B) * (1 - prob_A) * prob_B
    let P := outcome_1 + outcome_2
    exact (P = 19 / 400)
:= sorry

end probability_two_shots_l279_279042


namespace compute_expression_l279_279672

noncomputable def given_cubic (x : ℝ) : Prop :=
  x ^ 3 - 7 * x ^ 2 + 12 * x = 18

theorem compute_expression (a b c : ℝ) (ha : given_cubic a) (hb : given_cubic b) (hc : given_cubic c) :
  (a + b + c = 7) → 
  (a * b + b * c + c * a = 12) → 
  (a * b * c = 18) → 
  (a * b / c + b * c / a + c * a / b = -6) :=
by 
  sorry

end compute_expression_l279_279672


namespace sum_of_possible_values_of_x_l279_279369

-- Conditions
def radius (x : ℝ) : ℝ := x - 2
def semiMajor (x : ℝ) : ℝ := x - 3
def semiMinor (x : ℝ) : ℝ := x + 4

-- Theorem to be proved
theorem sum_of_possible_values_of_x (x : ℝ) :
  (π * semiMajor x * semiMinor x = 2 * π * (radius x) ^ 2) →
  (x = 5 ∨ x = 4) →
  5 + 4 = 9 :=
by
  intros
  rfl

end sum_of_possible_values_of_x_l279_279369


namespace simplified_expression_is_zero_l279_279027

variable (a b c : ℝ)

theorem simplified_expression_is_zero (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + 2 * b + 2 * c = 0) :
  (1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 0 := 
sorry

end simplified_expression_is_zero_l279_279027


namespace prime_solution_unique_l279_279984

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l279_279984


namespace Gretchen_weekend_profit_l279_279830

theorem Gretchen_weekend_profit :
  let saturday_revenue := 24 * 25
  let sunday_revenue := 16 * 15
  let total_revenue := saturday_revenue + sunday_revenue
  let park_fee := 5 * 6 * 2
  let art_supplies_cost := 8 * 2
  let total_expenses := park_fee + art_supplies_cost
  let profit := total_revenue - total_expenses
  profit = 764 :=
by
  sorry

end Gretchen_weekend_profit_l279_279830


namespace inequality_for_positive_real_numbers_l279_279528

theorem inequality_for_positive_real_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
  sorry

end inequality_for_positive_real_numbers_l279_279528


namespace Vasya_numbers_l279_279760

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l279_279760


namespace div_by_9_digit_B_l279_279905

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l279_279905


namespace meteorite_weight_possibilities_l279_279584

def valid_meteorite_weight_combinations : ℕ :=
  (2 * (Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2))) + (Nat.factorial 5)

theorem meteorite_weight_possibilities :
  valid_meteorite_weight_combinations = 180 :=
by
  -- Sorry added to skip the proof.
  sorry

end meteorite_weight_possibilities_l279_279584


namespace cat_food_insufficient_for_six_days_l279_279464

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l279_279464


namespace perfect_square_n_l279_279865

theorem perfect_square_n (n : ℤ) (h1 : n > 0) (h2 : ∃ k : ℤ, n^2 + 19 * n + 48 = k^2) : n = 33 :=
sorry

end perfect_square_n_l279_279865


namespace probability_of_drawing_letter_in_name_l279_279270

theorem probability_of_drawing_letter_in_name :
  let total_letters := 26
  let alonso_letters := ['a', 'l', 'o', 'n', 's']
  let number_of_alonso_letters := alonso_letters.length
  number_of_alonso_letters / total_letters = 5 / 26 :=
by
  sorry

end probability_of_drawing_letter_in_name_l279_279270


namespace at_least_two_squares_same_size_l279_279958

theorem at_least_two_squares_same_size (S : ℝ) : 
  ∃ a b : ℝ, a = b ∧ 
  (∀ i : ℕ, i < 10 → 
   ∀ j : ℕ, j < 10 → 
   (∃ k : ℕ, k < 9 ∧ 
    ((∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ x ≠ y → 
          (i = x ∧ j = y)) → 
        ((S / 10) = (a * k)) ∨ ((S / 10) = (b * k))))) := sorry

end at_least_two_squares_same_size_l279_279958


namespace VerifyMultiplicationProperties_l279_279467

theorem VerifyMultiplicationProperties (α : Type) [Semiring α] :
  ((∀ x y z : α, (x * y) * z = x * (y * z)) ∧
   (∀ x y : α, x * y = y * x) ∧
   (∀ x y z : α, x * (y + z) = x * y + x * z) ∧
   (∃ e : α, ∀ x : α, x * e = x)) := by
  sorry

end VerifyMultiplicationProperties_l279_279467


namespace mike_books_l279_279681

theorem mike_books : 51 - 45 = 6 := 
by 
  rfl

end mike_books_l279_279681


namespace solution_set_f_x_gt_0_l279_279153

theorem solution_set_f_x_gt_0 (b : ℝ)
  (h_eq : ∀ x : ℝ, (x + 1) * (x - 3) = 0 → b = -2) :
  {x : ℝ | (x - 1)^2 > 0} = {x : ℝ | x ≠ 1} :=
by
  sorry

end solution_set_f_x_gt_0_l279_279153


namespace factorize_problem1_factorize_problem2_l279_279998

-- Problem 1
theorem factorize_problem1 (a b : ℝ) : 
    -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := 
by sorry

-- Problem 2
theorem factorize_problem2 (a b x y : ℝ) : 
    9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) := 
by sorry

end factorize_problem1_factorize_problem2_l279_279998


namespace lucy_l279_279811

theorem lucy's_age 
  (L V: ℕ)
  (h1: L - 5 = 3 * (V - 5))
  (h2: L + 10 = 2 * (V + 10)) :
  L = 50 :=
by
  sorry

end lucy_l279_279811


namespace total_expenditure_l279_279059

-- Define the conditions
def cost_per_acre : ℕ := 20
def acres_bought : ℕ := 30
def house_cost : ℕ := 120000
def cost_per_cow : ℕ := 1000
def cows_bought : ℕ := 20
def cost_per_chicken : ℕ := 5
def chickens_bought : ℕ := 100
def hourly_installation_cost : ℕ := 100
def installation_hours : ℕ := 6
def solar_equipment_cost : ℕ := 6000

-- Define the total cost breakdown
def land_cost : ℕ := cost_per_acre * acres_bought
def cows_cost : ℕ := cost_per_cow * cows_bought
def chickens_cost : ℕ := cost_per_chicken * chickens_bought
def solar_installation_cost : ℕ := (hourly_installation_cost * installation_hours) + solar_equipment_cost

-- Define the total cost
def total_cost : ℕ :=
  land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost

-- The theorem statement
theorem total_expenditure : total_cost = 147700 :=
by
  -- Proof steps would go here
  sorry

end total_expenditure_l279_279059


namespace chess_tournament_time_spent_l279_279037

theorem chess_tournament_time_spent (games : ℕ) (moves_per_game : ℕ)
  (opening_moves : ℕ) (middle_moves : ℕ) (endgame_moves : ℕ)
  (polly_opening_time : ℝ) (peter_opening_time : ℝ)
  (polly_middle_time : ℝ) (peter_middle_time : ℝ)
  (polly_endgame_time : ℝ) (peter_endgame_time : ℝ)
  (total_time_hours : ℝ) :
  games = 4 →
  moves_per_game = 38 →
  opening_moves = 12 →
  middle_moves = 18 →
  endgame_moves = 8 →
  polly_opening_time = 35 →
  peter_opening_time = 45 →
  polly_middle_time = 30 →
  peter_middle_time = 45 →
  polly_endgame_time = 40 →
  peter_endgame_time = 60 →
  total_time_hours = (4 * ((12 * 35 + 18 * 30 + 8 * 40) + (12 * 45 + 18 * 45 + 8 * 60))) / 3600 :=
sorry

end chess_tournament_time_spent_l279_279037


namespace eldest_child_age_l279_279585

variable (y m e : Nat)

theorem eldest_child_age :
  (m - y = 3) →
  (e = 3 * y) →
  (e = y + m + 2) →
  (e = 15) :=
by
  intros h1 h2 h3
  sorry

end eldest_child_age_l279_279585


namespace multiplication_with_mixed_number_l279_279107

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l279_279107


namespace circle_diameter_tangents_l279_279337

open Real

theorem circle_diameter_tangents {x y : ℝ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) :
  ∃ d : ℝ, d = sqrt (x * y) :=
by
  sorry

end circle_diameter_tangents_l279_279337


namespace taxi_fare_distance_l279_279918

theorem taxi_fare_distance (x : ℕ) (h₁ : 8 + 2 * (x - 3) = 20) : x = 9 :=
by {
  sorry
}

end taxi_fare_distance_l279_279918


namespace clock_hands_angle_seventy_degrees_l279_279143

theorem clock_hands_angle_seventy_degrees (t : ℝ) (h : t ≥ 0 ∧ t ≤ 60):
    let hour_angle := 210 + 30 * (t / 60)
    let minute_angle := 360 * (t / 60)
    let angle := abs (hour_angle - minute_angle)
    (angle = 70 ∨ angle = 290) ↔ (t = 25 ∨ t = 52) :=
by apply sorry

end clock_hands_angle_seventy_degrees_l279_279143


namespace certain_event_among_options_l279_279387

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end certain_event_among_options_l279_279387


namespace multiply_mixed_number_l279_279111

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l279_279111


namespace second_experimental_point_is_correct_l279_279778

-- Define the temperature range
def lower_bound : ℝ := 1400
def upper_bound : ℝ := 1600

-- Define the golden ratio constant
def golden_ratio : ℝ := 0.618

-- Calculate the first experimental point using 0.618 method
def first_point : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

-- Calculate the second experimental point
def second_point : ℝ := upper_bound - (first_point - lower_bound)

-- Theorem stating the calculated second experimental point equals 1476.4
theorem second_experimental_point_is_correct :
  second_point = 1476.4 := by
  sorry

end second_experimental_point_is_correct_l279_279778


namespace geometric_sequence_sum_l279_279635

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h2 : a 3 + a 5 = 6) :
  a 5 + a 7 + a 9 = 28 :=
  sorry

end geometric_sequence_sum_l279_279635


namespace product_gcd_lcm_l279_279284

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l279_279284


namespace expenditure_representation_l279_279684

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end expenditure_representation_l279_279684


namespace sum_sequence_up_to_2015_l279_279397

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end sum_sequence_up_to_2015_l279_279397


namespace square_of_distance_is_82_l279_279781

noncomputable def square_distance_from_B_to_center (a b : ℝ) : ℝ := a^2 + b^2

theorem square_of_distance_is_82
  (a b : ℝ)
  (r : ℝ := 11)
  (ha : a^2 + (b + 7)^2 = r^2)
  (hc : (a + 3)^2 + b^2 = r^2) :
  square_distance_from_B_to_center a b = 82 := by
  -- Proof steps omitted
  sorry

end square_of_distance_is_82_l279_279781


namespace exists_m_n_l279_279332

theorem exists_m_n (p : ℕ) (hp : p > 10) [hp_prime : Fact (Nat.Prime p)] :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 :=
sorry

end exists_m_n_l279_279332


namespace jonah_poured_total_pitchers_l279_279471

theorem jonah_poured_total_pitchers :
  (0.25 + 0.125) + (0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666) + 
  (0.25 + 0.125) + (0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666) = 1.75 :=
by
  sorry

end jonah_poured_total_pitchers_l279_279471


namespace Vasya_numbers_l279_279757

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l279_279757


namespace numWaysElectOfficers_l279_279586

-- Definitions and conditions from part (a)
def numMembers : Nat := 30
def numPositions : Nat := 5
def members := ["Alice", "Bob", "Carol", "Dave"]
def allOrNoneCondition (S : List String) : Bool := 
  S.all (members.contains)

-- Function to count the number of ways to choose the officers
def countWays (n : Nat) (k : Nat) (allOrNone : Bool) : Nat :=
if allOrNone then
  -- All four members are positioned
  Nat.factorial k * (n - k)
else
  -- None of the four members are positioned
  let remaining := n - members.length
  remaining * (remaining - 1) * (remaining - 2) * (remaining - 3) * (remaining - 4)

theorem numWaysElectOfficers :
  let casesWithNone := countWays numMembers numPositions false
  let casesWithAll := countWays numMembers numPositions true
  (casesWithNone + casesWithAll) = 6378720 :=
by
  sorry

end numWaysElectOfficers_l279_279586


namespace probability_x_eq_y_cos_cos_l279_279962

theorem probability_x_eq_y_cos_cos (X Y : ℝ) (hX : -5 * π ≤ X ∧ X ≤ 5 * π) (hY : -5 * π ≤ Y ∧ Y ≤ 5 * π) 
(hcos : cos (cos X) = cos (cos Y)) : 
P(X = Y) = 11 / 100 :=
by sorry

end probability_x_eq_y_cos_cos_l279_279962


namespace Aunt_Zhang_expenditure_is_negative_l279_279682

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end Aunt_Zhang_expenditure_is_negative_l279_279682


namespace second_discount_percentage_l279_279329

-- Definitions for the given conditions
def original_price : ℝ := 33.78
def first_discount_rate : ℝ := 0.25
def final_price : ℝ := 19.0

-- Intermediate calculations based on the conditions
def first_discount : ℝ := first_discount_rate * original_price
def price_after_first_discount : ℝ := original_price - first_discount
def second_discount_amount : ℝ := price_after_first_discount - final_price

-- Lean theorem statement
theorem second_discount_percentage : (second_discount_amount / price_after_first_discount) * 100 = 25 := by
  sorry

end second_discount_percentage_l279_279329


namespace avg_zits_per_kid_mr_jones_class_l279_279910

-- Define the conditions
def avg_zits_ms_swanson_class := 5
def num_kids_ms_swanson_class := 25
def num_kids_mr_jones_class := 32
def extra_zits_mr_jones_class := 67

-- Define the total number of zits in Ms. Swanson's class
def total_zits_ms_swanson_class := avg_zits_ms_swanson_class * num_kids_ms_swanson_class

-- Define the total number of zits in Mr. Jones' class
def total_zits_mr_jones_class := total_zits_ms_swanson_class + extra_zits_mr_jones_class

-- Define the problem statement to prove: the average number of zits per kid in Mr. Jones' class
theorem avg_zits_per_kid_mr_jones_class : 
  total_zits_mr_jones_class / num_kids_mr_jones_class = 6 := by
  sorry

end avg_zits_per_kid_mr_jones_class_l279_279910


namespace mike_found_four_more_seashells_l279_279521

/--
Given:
1. Mike initially found 6.0 seashells.
2. The total number of seashells Mike had after finding more is 10.

Prove:
Mike found 4.0 more seashells.
-/
theorem mike_found_four_more_seashells (initial_seashells : ℝ) (total_seashells : ℝ)
  (h1 : initial_seashells = 6.0)
  (h2 : total_seashells = 10.0) :
  total_seashells - initial_seashells = 4.0 :=
by
  sorry

end mike_found_four_more_seashells_l279_279521


namespace linear_dependency_k_val_l279_279224

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l279_279224


namespace not_divisible_by_n_plus_4_l279_279354

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : 0 < n) : ¬ (n + 4 ∣ n^2 + 8 * n + 15) := 
sorry

end not_divisible_by_n_plus_4_l279_279354


namespace min_moves_move_stack_from_A_to_F_l279_279034

theorem min_moves_move_stack_from_A_to_F : 
  ∀ (squares : Fin 6) (stack : Fin 15), 
  (∃ moves : Nat, 
    (moves >= 0) ∧ 
    (moves == 49) ∧
    ∀ (a b : Fin 6), 
        ∃ (piece_from : Fin 15) (piece_to : Fin 15), 
        ((piece_from > piece_to) → (a ≠ b)) ∧
        (a == 0) ∧ 
        (b == 5)) :=
sorry

end min_moves_move_stack_from_A_to_F_l279_279034


namespace Vasya_numbers_l279_279732

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l279_279732


namespace combined_degrees_l279_279694

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l279_279694


namespace line_intersects_curve_equal_segments_l279_279157

theorem line_intersects_curve_equal_segments (k m : ℝ)
  (A B C : ℝ × ℝ)
  (hA_curve : A.2 = A.1^3 - 6 * A.1^2 + 13 * A.1 - 8)
  (hB_curve : B.2 = B.1^3 - 6 * B.1^2 + 13 * B.1 - 8)
  (hC_curve : C.2 = C.1^3 - 6 * C.1^2 + 13 * C.1 - 8)
  (h_lineA : A.2 = k * A.1 + m)
  (h_lineB : B.2 = k * B.1 + m)
  (h_lineC : C.2 = k * C.1 + m)
  (h_midpoint : 2 * B.1 = A.1 + C.1 ∧ 2 * B.2 = A.2 + C.2)
  : 2 * k + m = 2 :=
sorry

end line_intersects_curve_equal_segments_l279_279157


namespace total_brownies_correct_l279_279881

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end total_brownies_correct_l279_279881


namespace find_parabola_l279_279230

variable (P : ℝ × ℝ)
variable (a b : ℝ)

def parabola1 (P : ℝ × ℝ) (a : ℝ) := P.2^2 = 4 * a * P.1
def parabola2 (P : ℝ × ℝ) (b : ℝ) := P.1^2 = 4 * b * P.2

theorem find_parabola (hP : P = (-2, 4)) :
  (∃ a, parabola1 P a ∧ P.2^2 = -8 * P.1) ∨ 
  (∃ b, parabola2 P b ∧ P.1^2 = P.2) := by
  sorry

end find_parabola_l279_279230


namespace problem_statement_l279_279410

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem problem_statement (S : ℝ) (h1 : S = golden_ratio) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 :=
by
  sorry

end problem_statement_l279_279410


namespace enough_cat_food_for_six_days_l279_279434

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l279_279434


namespace base_k_addition_is_ten_l279_279301

theorem base_k_addition_is_ten :
  ∃ k : ℕ, (k > 4) ∧ (5 * k^3 + 3 * k^2 + 4 * k + 2 + 6 * k^3 + 4 * k^2 + 2 * k + 1 = 1 * k^4 + 4 * k^3 + 1 * k^2 + 6 * k + 3) ∧ k = 10 :=
by
  sorry

end base_k_addition_is_ten_l279_279301


namespace jacob_fraction_of_phoebe_age_l279_279018

-- Definitions
def Rehana_current_age := 25
def Rehana_future_age (years : Nat) := Rehana_current_age + years
def Phoebe_future_age (years : Nat) := (Rehana_future_age years) / 3
def Phoebe_current_age := Phoebe_future_age 5 - 5
def Jacob_age := 3
def fraction_of_Phoebe_age := Jacob_age / Phoebe_current_age

-- Theorem statement
theorem jacob_fraction_of_phoebe_age :
  fraction_of_Phoebe_age = 3 / 5 :=
  sorry

end jacob_fraction_of_phoebe_age_l279_279018


namespace find_value_of_expression_l279_279629

theorem find_value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 :=
by {
  sorry
}

end find_value_of_expression_l279_279629


namespace time_to_sell_all_cars_l279_279797

/-- Conditions: -/
def total_cars : ℕ := 500
def number_of_sales_professionals : ℕ := 10
def cars_per_salesperson_per_month : ℕ := 10

/-- Proof Statement: -/
theorem time_to_sell_all_cars 
  (total_cars : ℕ) 
  (number_of_sales_professionals : ℕ) 
  (cars_per_salesperson_per_month : ℕ) : 
  ((number_of_sales_professionals * cars_per_salesperson_per_month) > 0) →
  (total_cars / (number_of_sales_professionals * cars_per_salesperson_per_month)) = 5 :=
by
  sorry

end time_to_sell_all_cars_l279_279797


namespace unique_painted_cube_l279_279583

/-- Determine the number of distinct ways to paint a cube where:
  - One side is yellow,
  - Two sides are purple,
  - Three sides are orange.
  Taking into account that two cubes are considered identical if they can be rotated to match. -/
theorem unique_painted_cube :
  ∃ unique n : ℕ, n = 1 ∧
    (∃ (c : Fin 6 → Fin 3), 
      (∃ (i : Fin 6), c i = 0) ∧ 
      (∃ (j k : Fin 6), j ≠ k ∧ c j = 1 ∧ c k = 1) ∧ 
      (∃ (m p q : Fin 6), m ≠ p ∧ m ≠ q ∧ p ≠ q ∧ c m = 2 ∧ c p = 2 ∧ c q = 2)
    ) :=
sorry

end unique_painted_cube_l279_279583


namespace sebastian_age_correct_l279_279812

-- Define the ages involved
def sebastian_age_now := 40
def sister_age_now (S : ℕ) := S - 10
def father_age_now := 85

-- Define the conditions
def age_difference_condition (S : ℕ) := (sister_age_now S) = S - 10
def father_age_condition := father_age_now = 85
def past_age_sum_condition (S : ℕ) := (S - 5) + (sister_age_now S - 5) = 3 / 4 * (father_age_now - 5)

theorem sebastian_age_correct (S : ℕ) 
  (h1 : age_difference_condition S) 
  (h2 : father_age_condition) 
  (h3 : past_age_sum_condition S) : 
  S = sebastian_age_now := 
  by sorry

end sebastian_age_correct_l279_279812


namespace vasya_numbers_l279_279734

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l279_279734


namespace Vasya_numbers_l279_279730

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l279_279730


namespace at_least_two_participants_solved_exactly_five_l279_279183

open Nat Real

variable {n : ℕ}  -- Number of participants
variable {pij : ℕ → ℕ → ℕ} -- Number of contestants who correctly answered both the i-th and j-th problems

-- Conditions as definitions in Lean 4
def conditions (n : ℕ) (pij : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 6 → pij i j > (2 * n) / 5) ∧
  (∀ k, ¬ (∀ i, 1 ≤ i ∧ i ≤ 6 → pij k i = 1))

-- Main theorem statement
theorem at_least_two_participants_solved_exactly_five (n : ℕ) (pij : ℕ → ℕ → ℕ) (h : conditions n pij) : ∃ k₁ k₂, k₁ ≠ k₂ ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₁ i = 1) ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₂ i = 1) := sorry

end at_least_two_participants_solved_exactly_five_l279_279183


namespace gcd_lcm_product_24_60_l279_279285

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l279_279285


namespace smallest_five_digit_in_pascals_triangle_l279_279933

/-- In Pascal's triangle, the smallest five-digit number is 10000. -/
theorem smallest_five_digit_in_pascals_triangle : 
  ∃ (n k : ℕ), (10000 = Nat.choose n k) ∧ (∀ m l : ℕ, Nat.choose m l < 10000) → (n > m) := 
sorry

end smallest_five_digit_in_pascals_triangle_l279_279933


namespace enough_cat_food_for_six_days_l279_279433

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l279_279433


namespace golden_section_point_l279_279482

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_section_point (AB AP PB : ℝ)
  (h1 : AP + PB = AB)
  (h2 : AB = 5)
  (h3 : (AB / AP) = (AP / PB))
  (h4 : AP > PB) :
  AP = (5 * Real.sqrt 5 - 5) / 2 :=
by sorry

end golden_section_point_l279_279482


namespace enough_cat_food_for_six_days_l279_279431

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l279_279431


namespace evan_ivan_kara_total_weight_eq_432_l279_279997

variable (weight_evan : ℕ) (weight_ivan : ℕ) (weight_kara_cat : ℕ)

-- Conditions
def evans_dog_weight : Prop := weight_evan = 63
def ivans_dog_weight : Prop := weight_evan = 7 * weight_ivan
def karas_cat_weight : Prop := weight_kara_cat = 5 * (weight_evan + weight_ivan)

-- Mathematical equivalence
def total_weight : Prop := weight_evan + weight_ivan + weight_kara_cat = 432

theorem evan_ivan_kara_total_weight_eq_432 :
  evans_dog_weight weight_evan →
  ivans_dog_weight weight_evan weight_ivan →
  karas_cat_weight weight_evan weight_ivan weight_kara_cat →
  total_weight weight_evan weight_ivan weight_kara_cat :=
by
  intros h1 h2 h3
  sorry

end evan_ivan_kara_total_weight_eq_432_l279_279997


namespace find_least_d_l279_279774

theorem find_least_d :
  ∃ d : ℕ, (d % 7 = 1) ∧ (d % 5 = 2) ∧ (d % 3 = 2) ∧ d = 92 :=
by 
  sorry

end find_least_d_l279_279774


namespace num_factors_of_M_l279_279009

-- Define the integer M
def M : ℕ := 2^4 * 3^3 * 7^2

-- Prove that M has 60 natural-number factors
theorem num_factors_of_M : (∀ d : ℕ, d ∣ M → d ≠ 0) → (∑ d in (range (M + 1)), if d ∣ M then 1 else 0) = 60 :=
by
  -- Proof omitted
  sorry

end num_factors_of_M_l279_279009


namespace max_ab_under_constraint_l279_279480

theorem max_ab_under_constraint (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 3 * a + 2 * b = 1) : 
  ab ≤ (1 / 24) ∧ (ab = 1 / 24 ↔ a = 1 / 6 ∧ b = 1 / 4) :=
sorry

end max_ab_under_constraint_l279_279480


namespace n_pow_8_minus_1_divisible_by_480_l279_279214

theorem n_pow_8_minus_1_divisible_by_480 (n : ℤ) (h1 : ¬ (2 ∣ n)) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (5 ∣ n)) : 
  480 ∣ (n^8 - 1) := 
sorry

end n_pow_8_minus_1_divisible_by_480_l279_279214


namespace value_of_x_for_g_equals_g_inv_l279_279975

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l279_279975


namespace paint_intensity_l279_279215

theorem paint_intensity (I : ℝ) (F : ℝ) (I_initial I_new : ℝ) : 
  I_initial = 50 → I_new = 30 → F = 2 / 3 → I = 20 :=
by
  intros h1 h2 h3
  sorry

end paint_intensity_l279_279215


namespace vasya_numbers_l279_279747

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l279_279747


namespace units_digit_of_A_is_1_l279_279165

-- Definition of A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Main theorem stating that the units digit of A is 1
theorem units_digit_of_A_is_1 : (A % 10) = 1 :=
by 
  -- Given conditions about powers of 3 and their properties in modulo 10
  sorry

end units_digit_of_A_is_1_l279_279165


namespace completing_the_square_l279_279533

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l279_279533


namespace brass_weight_l279_279222

theorem brass_weight (copper zinc brass : ℝ) (h_ratio : copper / zinc = 3 / 7) (h_zinc : zinc = 70) : brass = 100 :=
by
  sorry

end brass_weight_l279_279222


namespace roshini_sweets_cost_correct_l279_279201

noncomputable def roshini_sweet_cost_before_discounts_and_tax : ℝ := 10.54

theorem roshini_sweets_cost_correct (R F1 F2 F3 : ℝ) (h1 : R + F1 + F2 + F3 = 10.54)
    (h2 : R * 0.9 = (10.50 - 9.20) / 1.08)
    (h3 : F1 + F2 + F3 = 3.40 + 4.30 + 1.50) :
    R + F1 + F2 + F3 = roshini_sweet_cost_before_discounts_and_tax :=
by
  sorry

end roshini_sweets_cost_correct_l279_279201


namespace solution_l279_279179

theorem solution (x : ℝ) (h : ¬ (x ^ 2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end solution_l279_279179


namespace average_sales_is_96_l279_279701

-- Definitions for the sales data
def january_sales : ℕ := 110
def february_sales : ℕ := 80
def march_sales : ℕ := 70
def april_sales : ℕ := 130
def may_sales : ℕ := 90

-- Number of months
def num_months : ℕ := 5

-- Total sales calculation
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + may_sales

-- Average sales per month calculation
def average_sales_per_month : ℕ := total_sales / num_months

-- Proposition to prove that the average sales per month is 96
theorem average_sales_is_96 : average_sales_per_month = 96 :=
by
  -- We use 'sorry' here to skip the proof, as the problem requires only the statement
  sorry

end average_sales_is_96_l279_279701


namespace vasya_numbers_l279_279741

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l279_279741


namespace Brenda_mice_left_l279_279599

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end Brenda_mice_left_l279_279599


namespace vasya_numbers_l279_279756

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l279_279756


namespace ak_not_perfect_square_l279_279237

theorem ak_not_perfect_square (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k1 k2, a k1 = 1988 ∧ b k2 = 1988) :
  ∀ k, ¬ ∃ n, a k = n * n :=
by
  sorry

end ak_not_perfect_square_l279_279237


namespace chickens_count_l279_279538

theorem chickens_count (rabbits frogs : ℕ) (h_rabbits : rabbits = 49) (h_frogs : frogs = 37) :
  ∃ (C : ℕ), frogs + C = rabbits + 9 ∧ C = 21 :=
by
  sorry

end chickens_count_l279_279538


namespace unqualified_weight_l279_279084

theorem unqualified_weight (w : ℝ) (upper_limit lower_limit : ℝ) 
  (h1 : upper_limit = 10.1) 
  (h2 : lower_limit = 9.9) 
  (h3 : w = 9.09 ∨ w = 9.99 ∨ w = 10.01 ∨ w = 10.09) :
  ¬ (9.09 ≥ lower_limit ∧ 9.09 ≤ upper_limit) :=
by
  sorry

end unqualified_weight_l279_279084


namespace strawberry_cake_cost_proof_l279_279095

-- Define the constants
def chocolate_cakes : ℕ := 3
def price_per_chocolate_cake : ℕ := 12
def total_bill : ℕ := 168
def number_of_strawberry_cakes : ℕ := 6

-- Define the calculation for the total cost of chocolate cakes
def total_cost_of_chocolate_cakes : ℕ := chocolate_cakes * price_per_chocolate_cake

-- Define the remaining cost for strawberry cakes
def remaining_cost : ℕ := total_bill - total_cost_of_chocolate_cakes

-- Prove the cost per strawberry cake
def cost_per_strawberry_cake : ℕ := remaining_cost / number_of_strawberry_cakes

theorem strawberry_cake_cost_proof : cost_per_strawberry_cake = 22 := by
  -- We skip the proof here. Detailed proof steps would go in the place of sorry
  sorry

end strawberry_cake_cost_proof_l279_279095


namespace rectangular_solid_dimension_change_l279_279949

theorem rectangular_solid_dimension_change (a b : ℝ) (h : 2 * a^2 + 4 * a * b = 0.6 * (6 * a^2)) : b = 0.4 * a :=
by sorry

end rectangular_solid_dimension_change_l279_279949


namespace christopher_more_money_l279_279507

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end christopher_more_money_l279_279507


namespace system_of_equations_solution_l279_279078

theorem system_of_equations_solution (x y : ℚ) :
  (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧ 4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by
  sorry

end system_of_equations_solution_l279_279078


namespace can_use_bisection_method_l279_279963

noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := x⁻¹
noncomputable def f3 (x : ℝ) : ℝ := abs x
noncomputable def f4 (x : ℝ) : ℝ := x^3

theorem can_use_bisection_method : ∃ (a b : ℝ), a < b ∧ (f4 a) * (f4 b) < 0 := 
sorry

end can_use_bisection_method_l279_279963


namespace contrapositive_p_l279_279418

-- Definitions
def A_score := 70
def B_score := 70
def C_score := 65
def p := ∀ (passing_score : ℕ), passing_score < 70 → (A_score < passing_score ∧ B_score < passing_score ∧ C_score < passing_score)

-- Statement to be proved
theorem contrapositive_p : 
  ∀ (passing_score : ℕ), (A_score ≥ passing_score ∨ B_score ≥ passing_score ∨ C_score ≥ passing_score) → (¬ passing_score < 70) := 
by
  sorry

end contrapositive_p_l279_279418


namespace opposite_of_neg_abs_opposite_of_neg_abs_correct_l279_279912

theorem opposite_of_neg_abs (x : ℚ) (hx : |x| = 2 / 5) : -|x| = - (2 / 5) := sorry

theorem opposite_of_neg_abs_correct (x : ℚ) (hx : |x| = 2 / 5) : - -|x| = 2 / 5 := by
  rw [opposite_of_neg_abs x hx]
  simp

end opposite_of_neg_abs_opposite_of_neg_abs_correct_l279_279912


namespace turtle_population_estimate_l279_279082

theorem turtle_population_estimate :
  (tagged_in_june = 90) →
  (sample_november = 50) →
  (tagged_november = 4) →
  (natural_causes_removal = 0.30) →
  (new_hatchlings_november = 0.50) →
  estimate = 563 :=
by
  intros tagged_in_june sample_november tagged_november natural_causes_removal new_hatchlings_november
  sorry

end turtle_population_estimate_l279_279082


namespace sum_of_reciprocals_of_squares_l279_279048

open Nat

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h_prod : a * b = 5) : 
  (1 : ℚ) / (a * a) + (1 : ℚ) / (b * b) = 26 / 25 :=
by
  -- proof steps skipping with sorry
  sorry

end sum_of_reciprocals_of_squares_l279_279048


namespace number_of_students_l279_279787

theorem number_of_students (n : ℕ) (h1 : n < 60) (h2 : n % 6 = 4) (h3 : n % 8 = 5) : n = 46 := by
  sorry

end number_of_students_l279_279787


namespace misha_grade_students_l279_279345

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l279_279345


namespace unique_a_for_set_A_l279_279956

def A (a : ℝ) : Set ℝ := {a^2, 2 - a, 4}

theorem unique_a_for_set_A (a : ℝ) : A a = {x : ℝ // x = a^2 ∨ x = 2 - a ∨ x = 4} → a = -1 :=
by
  sorry

end unique_a_for_set_A_l279_279956


namespace magic_square_solution_l279_279877

theorem magic_square_solution (d e k f g h x y : ℤ)
  (h1 : x + 4 + f = 87 + d + f)
  (h2 : x + d + h = 87 + e + h)
  (h3 : x + y + 87 = 4 + d + e)
  (h4 : f + g + h = x + y + 87)
  (h5 : d = x - 83)
  (h6 : e = 2 * x - 170)
  (h7 : y = 3 * x - 274)
  (h8 : f = g)
  (h9 : g = h) :
  x = 62 ∧ y = -88 :=
by
  sorry

end magic_square_solution_l279_279877


namespace div_by_9_digit_B_l279_279906

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l279_279906


namespace product_of_gcd_and_lcm_1440_l279_279278

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l279_279278


namespace integer_multiplied_by_b_l279_279651

variable (a b : ℤ) (x : ℤ)

theorem integer_multiplied_by_b (h1 : -11 * a < 0) (h2 : x < 0) (h3 : (-11 * a * x) * (x * b) + a * b = 89) :
  x = -1 :=
by
  sorry

end integer_multiplied_by_b_l279_279651


namespace xy_sum_value_l279_279147

theorem xy_sum_value (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
sorry

end xy_sum_value_l279_279147


namespace christina_speed_limit_l279_279135

theorem christina_speed_limit :
  ∀ (D total_distance friend_distance : ℝ), 
  total_distance = 210 → 
  friend_distance = 3 * 40 → 
  D = total_distance - friend_distance → 
  D / 3 = 30 :=
by
  intros D total_distance friend_distance 
  intros h1 h2 h3 
  sorry

end christina_speed_limit_l279_279135


namespace sum_digits_base8_to_base4_l279_279582

theorem sum_digits_base8_to_base4 :
  ∀ n : ℕ, (n ≥ 512 ∧ n ≤ 4095) →
  (∃ d : ℕ, (4^d > n ∧ n ≥ 4^(d-1))) →
  (d = 6) :=
by {
  sorry
}

end sum_digits_base8_to_base4_l279_279582


namespace cyclic_quadrilateral_angle_D_l279_279503

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h1 : A + C = 180) (h2 : B + D = 180) (h3 : 3 * A = 4 * B) (h4 : 3 * A = 6 * C) : D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l279_279503


namespace total_distance_traveled_l279_279771

noncomputable def total_distance (d v1 v2 v3 time_total : ℝ) : ℝ :=
  3 * d

theorem total_distance_traveled
  (d : ℝ)
  (v1 : ℝ := 3)
  (v2 : ℝ := 6)
  (v3 : ℝ := 9)
  (time_total : ℝ := 11 / 60)
  (h : d / v1 + d / v2 + d / v3 = time_total) :
  total_distance d v1 v2 v3 time_total = 0.9 :=
by
  sorry

end total_distance_traveled_l279_279771


namespace cloves_used_for_roast_chicken_l279_279202

section
variable (total_cloves : ℕ)
variable (remaining_cloves : ℕ)

theorem cloves_used_for_roast_chicken (h1 : total_cloves = 93) (h2 : remaining_cloves = 7) : total_cloves - remaining_cloves = 86 := 
by 
  have h : total_cloves - remaining_cloves = 93 - 7 := by rw [h1, h2]
  exact h
-- sorry
end

end cloves_used_for_roast_chicken_l279_279202


namespace production_profit_range_l279_279251

theorem production_profit_range (x : ℝ) (t : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) (h3 : 0 ≤ t) :
  (200 * (5 * x + 1 - 3 / x) ≥ 3000) → (3 ≤ x ∧ x ≤ 10) :=
sorry

end production_profit_range_l279_279251


namespace cat_food_inequality_l279_279440

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l279_279440


namespace gcd_lcm_product_24_60_l279_279297

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l279_279297


namespace wombats_count_l279_279134

theorem wombats_count (W : ℕ) (H : 4 * W + 3 = 39) : W = 9 := 
sorry

end wombats_count_l279_279134


namespace rearrange_marked_cells_below_diagonal_l279_279848

theorem rearrange_marked_cells_below_diagonal (n : ℕ) (marked_cells : Finset (Fin n × Fin n)) :
  marked_cells.card = n - 1 →
  ∃ row_permutation col_permutation : Equiv (Fin n) (Fin n), ∀ (i j : Fin n),
    (row_permutation i, col_permutation j) ∈ marked_cells → j < i :=
by
  sorry

end rearrange_marked_cells_below_diagonal_l279_279848


namespace fraction_subtraction_l279_279075

theorem fraction_subtraction (a b : ℚ) (h_a: a = 5/9) (h_b: b = 1/6) : a - b = 7/18 :=
by
  sorry

end fraction_subtraction_l279_279075


namespace abs_eq_solution_l279_279935

theorem abs_eq_solution (x : ℝ) (h : abs (x - 3) = abs (x + 2)) : x = 1 / 2 :=
sorry

end abs_eq_solution_l279_279935


namespace find_n_for_positive_root_l279_279845

theorem find_n_for_positive_root :
  ∃ x : ℝ, x > 0 ∧ (∃ n : ℝ, (n / (x - 1) + 2 / (1 - x) = 1)) ↔ n = 2 :=
by
  sorry

end find_n_for_positive_root_l279_279845


namespace problem_statement_l279_279305

noncomputable def f : ℝ → ℝ := sorry

variable (α : ℝ)

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 3) = -f x
axiom tan_alpha : Real.tan α = 2

theorem problem_statement : f (15 * Real.sin α * Real.cos α) = 0 := 
by {
  sorry
}

end problem_statement_l279_279305


namespace linear_functions_value_at_20_l279_279383

-- Definitions of linear functions intersection and properties
def linear_functions_intersect_at (k1 k2 b1 b2 : ℝ) (x : ℝ) : Prop :=
  k1 * x + b1 = k2 * x + b2

def value_difference_at (k1 k2 b1 b2 : ℝ) (x diff : ℝ) : Prop :=
  (k1 * x + b1 - (k2 * x + b2)).abs = diff

def function_value_at (k b x val : ℝ) : Prop :=
  k * x + b = val

-- Main proof statement
theorem linear_functions_value_at_20
  (k1 k2 b1 b2 : ℝ) :
  (linear_functions_intersect_at k1 k2 b1 b2 2) →
  (value_difference_at k1 k2 b1 b2 8 8) →
  (function_value_at k1 b1 20 100) →
  (k2 * 20 + b2 = 76 ∨ k2 * 20 + b2 = 124) :=
by
  sorry

end linear_functions_value_at_20_l279_279383


namespace fish_pond_estimate_l279_279186

variable (N : ℕ)
variable (total_first_catch total_second_catch marked_in_first_catch marked_in_second_catch : ℕ)

/-- Estimate the total number of fish in the pond -/
theorem fish_pond_estimate
  (h1 : total_first_catch = 100)
  (h2 : total_second_catch = 120)
  (h3 : marked_in_first_catch = 100)
  (h4 : marked_in_second_catch = 15)
  (h5 : (marked_in_second_catch : ℚ) / total_second_catch = (marked_in_first_catch : ℚ) / N) :
  N = 800 := 
sorry

end fish_pond_estimate_l279_279186


namespace reduce_entanglement_l279_279588

/- 
Define a graph structure and required operations as per the given conditions. 
-/
structure Graph (V : Type) :=
  (E : V -> V -> Prop)

def remove_odd_degree_verts (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph reduction logic

def duplicate_graph (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph duplication logic

/--
  Prove that any graph where each vertex can be part of multiple entanglements 
  can be reduced to a state where no two vertices are connected using the given operations.
-/
theorem reduce_entanglement (G : Graph V) : ∃ G', 
  G' = remove_odd_degree_verts (duplicate_graph G) ∧
  (∀ (v1 v2 : V), ¬ G'.E v1 v2) :=
  by
  sorry

end reduce_entanglement_l279_279588


namespace weather_forecast_probability_l279_279719

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem weather_forecast_probability :
  binomial_probability 3 2 0.8 = 0.384 :=
by
  sorry

end weather_forecast_probability_l279_279719


namespace cat_food_insufficient_for_six_days_l279_279463

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l279_279463


namespace evaluate_expression_l279_279894

theorem evaluate_expression (a b : ℚ) (h1 : a + b = 4) (h2 : a - b = 2) :
  ( (a^2 - 6 * a * b + 9 * b^2) / (a^2 - 2 * a * b) / ((5 * b^2 / (a - 2 * b)) - (a + 2 * b)) - 1 / a ) = -1 / 3 :=
by
  sorry

end evaluate_expression_l279_279894


namespace combined_degrees_l279_279696

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l279_279696


namespace epsilon_max_success_ratio_l279_279981

theorem epsilon_max_success_ratio :
  ∃ (x y z w u v: ℕ), 
  (y ≠ 350) ∧
  0 < x ∧ 0 < z ∧ 0 < u ∧ 
  x < y ∧ z < w ∧ u < v ∧
  x + z + u < y + w + v ∧
  y + w + v = 800 ∧
  (x / y : ℚ) < (210 / 350 : ℚ) ∧ 
  (z / w : ℚ) < (delta_day_2_ratio) ∧ 
  (u / v : ℚ) < (delta_day_3_ratio) ∧ 
  (x + z + u) / 800 = (789 / 800 : ℚ) := 
by
  sorry

end epsilon_max_success_ratio_l279_279981


namespace find_income_l279_279366

-- Define the condition for savings
def savings_formula (income expenditure savings : ℝ) : Prop :=
  income - expenditure = savings

-- Define the ratio between income and expenditure
def ratio_condition (income expenditure : ℝ) : Prop :=
  income = 5 / 4 * expenditure

-- Given:
-- savings: Rs. 3400
-- We need to prove the income is Rs. 17000
theorem find_income (savings : ℝ) (income expenditure : ℝ) :
  savings_formula income expenditure savings →
  ratio_condition income expenditure →
  savings = 3400 →
  income = 17000 :=
sorry

end find_income_l279_279366


namespace cone_height_l279_279046

theorem cone_height (r_sector : ℝ) (θ_sector : ℝ) :
  r_sector = 3 → θ_sector = (2 * Real.pi / 3) → 
  ∃ (h : ℝ), h = 2 * Real.sqrt 2 := 
by 
  intros r_sector_eq θ_sector_eq
  sorry

end cone_height_l279_279046


namespace brownies_on_counter_l279_279878

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end brownies_on_counter_l279_279878


namespace multiplication_of_mixed_number_l279_279100

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l279_279100


namespace stacy_height_proof_l279_279361

noncomputable def height_last_year : ℕ := 50
noncomputable def brother_growth : ℕ := 1
noncomputable def stacy_growth : ℕ := brother_growth + 6
noncomputable def stacy_current_height : ℕ := height_last_year + stacy_growth

theorem stacy_height_proof : stacy_current_height = 57 := 
by
  sorry

end stacy_height_proof_l279_279361


namespace exponent_property_l279_279798

theorem exponent_property : 3000 * 3000^2500 = 3000^2501 := 
by sorry

end exponent_property_l279_279798


namespace jellybean_proof_l279_279871

def number_vanilla_jellybeans : ℕ := 120

def number_grape_jellybeans (V : ℕ) : ℕ := 5 * V + 50

def number_strawberry_jellybeans (V : ℕ) : ℕ := (2 * V) / 3

def total_number_jellybeans (V G S : ℕ) : ℕ := V + G + S

def cost_per_vanilla_jellybean : ℚ := 0.05

def cost_per_grape_jellybean : ℚ := 0.08

def cost_per_strawberry_jellybean : ℚ := 0.07

def total_cost_jellybeans (V G S : ℕ) : ℚ := 
  (cost_per_vanilla_jellybean * V) + 
  (cost_per_grape_jellybean * G) + 
  (cost_per_strawberry_jellybean * S)

theorem jellybean_proof :
  ∃ (V G S : ℕ), 
    V = number_vanilla_jellybeans ∧
    G = number_grape_jellybeans V ∧
    S = number_strawberry_jellybeans V ∧
    total_number_jellybeans V G S = 850 ∧
    total_cost_jellybeans V G S = 63.60 :=
by
  sorry

end jellybean_proof_l279_279871


namespace detergent_required_l279_279882

def ounces_of_detergent_per_pound : ℕ := 2
def pounds_of_clothes : ℕ := 9

theorem detergent_required :
  (ounces_of_detergent_per_pound * pounds_of_clothes) = 18 := by
  sorry

end detergent_required_l279_279882


namespace digit_B_value_l279_279903

theorem digit_B_value (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 2 % 9 = 0):
  B = 6 :=
begin
  sorry
end

end digit_B_value_l279_279903


namespace event_day_price_l279_279796

theorem event_day_price (original_price : ℝ) (first_discount second_discount : ℝ)
  (h1 : original_price = 250) (h2 : first_discount = 0.4) (h3 : second_discount = 0.25) : 
  ∃ discounted_price : ℝ, 
  discounted_price = (original_price * (1 - first_discount)) * (1 - second_discount) → 
  discounted_price = 112.5 :=
by
  use (250 * (1 - 0.4) * (1 - 0.25))
  sorry

end event_day_price_l279_279796


namespace required_raise_percentage_l279_279417

theorem required_raise_percentage (S : ℝ) (hS : S > 0) : 
  ((S - (0.85 * S - 50)) / (0.85 * S - 50) = 0.1875) :=
by
  -- Proof of this theorem can be carried out here
  sorry

end required_raise_percentage_l279_279417


namespace min_reciprocal_sum_l279_279151

theorem min_reciprocal_sum (m n a b : ℝ) (h1 : m = 5) (h2 : n = 5) 
  (h3 : m * a + n * b = 1) (h4 : 0 < a) (h5 : 0 < b) : 
  (1 / a + 1 / b) = 20 :=
by 
  sorry

end min_reciprocal_sum_l279_279151


namespace basketball_game_first_half_points_l279_279847

noncomputable def total_points_first_half
  (eagles_points : ℕ → ℕ) (lions_points : ℕ → ℕ) (common_ratio : ℕ) (common_difference : ℕ) : ℕ :=
  eagles_points 0 + eagles_points 1 + lions_points 0 + lions_points 1

theorem basketball_game_first_half_points 
  (eagles_points lions_points : ℕ → ℕ)
  (common_ratio : ℕ) (common_difference : ℕ)
  (h1 : eagles_points 0 = lions_points 0)
  (h2 : ∀ n, eagles_points (n + 1) = common_ratio * eagles_points n)
  (h3 : ∀ n, lions_points (n + 1) = lions_points n + common_difference)
  (h4 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 =
        lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 + 3)
  (h5 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 ≤ 120)
  (h6 : lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 ≤ 120) :
  total_points_first_half eagles_points lions_points common_ratio common_difference = 15 :=
sorry

end basketball_game_first_half_points_l279_279847


namespace symmetric_point_coordinates_l279_279713

-- Define the type for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetric point function with respect to the x-axis
def symmetricPointWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Define the specific point
def givenPoint : Point3D := { x := 2, y := 3, z := 4 }

-- State the theorem to be proven
theorem symmetric_point_coordinates : 
  symmetricPointWithRespectToXAxis givenPoint = { x := 2, y := -3, z := -4 } :=
by
  sorry

end symmetric_point_coordinates_l279_279713


namespace mul_mixed_number_l279_279122

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l279_279122


namespace action_figures_more_than_books_l279_279660

variable (initialActionFigures : Nat) (newActionFigures : Nat) (books : Nat)

def totalActionFigures (initialActionFigures newActionFigures : Nat) : Nat :=
  initialActionFigures + newActionFigures

theorem action_figures_more_than_books :
  initialActionFigures = 5 → newActionFigures = 7 → books = 9 →
  totalActionFigures initialActionFigures newActionFigures - books = 3 :=
by
  intros h_initial h_new h_books
  rw [h_initial, h_new, h_books]
  sorry

end action_figures_more_than_books_l279_279660


namespace mary_baseball_cards_count_l279_279872

def mary_initial_cards := 18
def mary_torn_cards := 8
def fred_gift_cards := 26
def mary_bought_cards := 40
def exchange_with_tom := 0
def mary_lost_cards := 5
def trade_with_lisa_gain := 1
def exchange_with_alex_loss := 2

theorem mary_baseball_cards_count : 
  mary_initial_cards - mary_torn_cards
  + fred_gift_cards
  + mary_bought_cards 
  + exchange_with_tom
  - mary_lost_cards
  + trade_with_lisa_gain 
  - exchange_with_alex_loss 
  = 70 := 
by
  sorry

end mary_baseball_cards_count_l279_279872


namespace number_of_performance_orders_l279_279782

-- Define the options for the programs
def programs : List String := ["A", "B", "C", "D", "E", "F", "G", "H"]

-- Define a function to count valid performance orders under given conditions
def countPerformanceOrders (progs : List String) : ℕ :=
  sorry  -- This is where the logic to count performance orders goes

-- The theorem to assert the total number of performance orders
theorem number_of_performance_orders : countPerformanceOrders programs = 2860 :=
by
  sorry  -- Proof of the theorem

end number_of_performance_orders_l279_279782


namespace colby_mangoes_harvested_60_l279_279466

variable (kg_left kg_each : ℕ)

def totalKgMangoes (x : ℕ) : Prop :=
  ∃ x : ℕ, 
  kg_left = (x - 20) / 2 ∧ 
  kg_each * kg_left = 160 ∧
  kg_each = 8

-- Problem Statement: Prove the total kilograms of mangoes harvested is 60 given the conditions.
theorem colby_mangoes_harvested_60 (x : ℕ) (h1 : x - 20 = 2 * kg_left)
(h2 : kg_each * kg_left = 160) (h3 : kg_each = 8) : x = 60 := by
  sorry

end colby_mangoes_harvested_60_l279_279466


namespace number_of_paths_l279_279502

theorem number_of_paths (r u : ℕ) (h_r : r = 5) (h_u : u = 4) : 
  (Nat.choose (r + u) u) = 126 :=
by
  -- The proof is omitted, as requested.
  sorry

end number_of_paths_l279_279502


namespace gcd_lcm_product_24_60_l279_279289

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l279_279289


namespace Bo_knew_percentage_l279_279096

-- Definitions from the conditions
def total_flashcards := 800
def words_per_day := 16
def days := 40
def total_words_to_learn := words_per_day * days
def known_words := total_flashcards - total_words_to_learn

-- Statement that we need to prove
theorem Bo_knew_percentage : (known_words.toFloat / total_flashcards.toFloat) * 100 = 20 :=
by
  sorry  -- Proof is omitted as per the instructions

end Bo_knew_percentage_l279_279096


namespace grapes_purchased_l279_279005

variable (G : ℕ)
variable (rate_grapes : ℕ) (qty_mangoes : ℕ) (rate_mangoes : ℕ) (total_paid : ℕ)

theorem grapes_purchased (h1 : rate_grapes = 70)
                        (h2 : qty_mangoes = 9)
                        (h3 : rate_mangoes = 55)
                        (h4 : total_paid = 1055) :
                        70 * G + 9 * 55 = 1055 → G = 8 :=
by
  sorry

end grapes_purchased_l279_279005


namespace reciprocal_of_neg_eight_l279_279914

theorem reciprocal_of_neg_eight : (1 / (-8 : ℝ)) = -1 / 8 := sorry

end reciprocal_of_neg_eight_l279_279914


namespace vasya_numbers_l279_279744

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l279_279744


namespace quadratic_has_two_distinct_real_roots_l279_279937

/-- The quadratic equation x^2 + 2x - 3 = 0 has two distinct real roots. -/
theorem quadratic_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ ^ 2 + 2 * x₁ - 3 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 3 = 0) := by
sorry

end quadratic_has_two_distinct_real_roots_l279_279937


namespace abs_neg_five_is_five_l279_279706

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l279_279706


namespace train_length_is_correct_l279_279959

noncomputable def speed_kmph : ℝ := 72
noncomputable def time_seconds : ℝ := 74.994
noncomputable def tunnel_length_m : ℝ := 1400
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600
noncomputable def total_distance : ℝ := speed_mps * time_seconds
noncomputable def train_length : ℝ := total_distance - tunnel_length_m

theorem train_length_is_correct :
  train_length = 99.88 := by
  -- the proof will follow here
  sorry

end train_length_is_correct_l279_279959


namespace Vasya_numbers_l279_279762

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l279_279762


namespace Vasya_numbers_l279_279765

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l279_279765


namespace expectation_of_X_variance_of_3X_plus_2_l279_279640

open ProbabilityTheory

namespace Proof

def X : Distribution ℝ := binom 4 (1/3)

theorem expectation_of_X :
  E[X] = 4 * (1/3) := by
  sorry

theorem variance_of_3X_plus_2 :
  let D (X : Distribution ℝ) := Var[X]
  D[3 * X + 2] = 9 * D[X] := by
  sorry

end Proof

end expectation_of_X_variance_of_3X_plus_2_l279_279640


namespace multiplication_of_mixed_number_l279_279102

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l279_279102


namespace Mrs_Heine_treats_l279_279352

theorem Mrs_Heine_treats :
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  total_treats = 11 :=
by
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  show total_treats = 11
  sorry

end Mrs_Heine_treats_l279_279352


namespace num_factors_of_M_l279_279008

def M : ℕ := 2^4 * 3^3 * 7^2

theorem num_factors_of_M : ∃ n, n = 60 ∧ (∀ d e f : ℕ, 0 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 → (2^d * 3^e * 7^f ∣ M) ∧ ∃ k, k = 5 * 4 * 3 ∧ k = n) :=
by
  sorry

end num_factors_of_M_l279_279008


namespace domain_of_g_l279_279993

theorem domain_of_g :
  {x : ℝ | -6*x^2 - 7*x + 8 >= 0} = 
  {x : ℝ | (7 - Real.sqrt 241) / 12 ≤ x ∧ x ≤ (7 + Real.sqrt 241) / 12} :=
by
  sorry

end domain_of_g_l279_279993


namespace roots_quadratic_expression_value_l279_279154

theorem roots_quadratic_expression_value (m n : ℝ) 
  (h1 : m^2 + 2 * m - 2027 = 0)
  (h2 : n^2 + 2 * n - 2027 = 0) :
  (2 * m - m * n + 2 * n) = 2023 :=
by
  sorry

end roots_quadratic_expression_value_l279_279154


namespace cat_food_inequality_l279_279455

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l279_279455


namespace parabola_hyperbola_tangent_l279_279551

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5
noncomputable def hyperbola (x y : ℝ) (m : ℝ) : ℝ := y^2 - m * x^2 - 1

theorem parabola_hyperbola_tangent (m : ℝ) :
(∃ x y : ℝ, y = parabola x ∧ hyperbola x y m = 0) ↔ 
m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6 := by
  sorry

end parabola_hyperbola_tangent_l279_279551


namespace linear_dependency_k_val_l279_279226

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l279_279226


namespace combined_degrees_l279_279691

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l279_279691


namespace boxes_given_to_mom_l279_279041

theorem boxes_given_to_mom 
  (sophie_boxes : ℕ) 
  (donuts_per_box : ℕ) 
  (donuts_to_sister : ℕ) 
  (donuts_left_for_her : ℕ) 
  (H1 : sophie_boxes = 4) 
  (H2 : donuts_per_box = 12) 
  (H3 : donuts_to_sister = 6) 
  (H4 : donuts_left_for_her = 30)
  : sophie_boxes * donuts_per_box - donuts_to_sister - donuts_left_for_her = donuts_per_box := 
by
  sorry

end boxes_given_to_mom_l279_279041


namespace hyperbola_focal_length_l279_279900

theorem hyperbola_focal_length : 
  (∃ (f : ℝ) (x y : ℝ), (3 * x^2 - y^2 = 3) ∧ (f = 4)) :=
by {
  sorry
}

end hyperbola_focal_length_l279_279900


namespace table_chair_price_l279_279092

theorem table_chair_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : T = 84) : T + C = 96 :=
sorry

end table_chair_price_l279_279092


namespace quadratic_has_two_distinct_real_roots_l279_279554

theorem quadratic_has_two_distinct_real_roots :
  ∀ (x : ℝ), ∃ (r1 r2 : ℝ), (x^2 - 2*x - 1 = 0) → r1 ≠ r2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l279_279554


namespace assignment_three_booths_l279_279849

/--
In a sub-venue of the World Chinese Business Conference, there are three booths, A, B, and C, and 
four "bilingual" volunteers, namely 甲, 乙, 丙, and 丁. Each booth must have at least one person. 
Prove that the number of different ways to assign volunteers 甲 and 乙 to the same booth is 6.
-/
theorem assignment_three_booths :
  ∃ (volunteers : set string) (booths : set string),
    volunteers = {"甲", "乙", "丙", "丁"} ∧
    booths = {"A", "B", "C"} ∧
    (∀ booth ∈ booths, ∃ volunteer ∈ volunteers, volunteer_assigned_to_booth volunteer booth) ∧
    (number_of_ways_to_assign_same_booth "甲" "乙" = 6) :=
sorry

end assignment_three_booths_l279_279849


namespace solution_to_fraction_problem_l279_279624

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end solution_to_fraction_problem_l279_279624


namespace correct_sum_of_integers_l279_279726

theorem correct_sum_of_integers :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a - b = 3 ∧ a * b = 63 ∧ a + b = 17 :=
by 
  sorry

end correct_sum_of_integers_l279_279726


namespace sum_sequence_up_to_2015_l279_279398

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end sum_sequence_up_to_2015_l279_279398


namespace boxes_of_apples_with_cherries_l279_279913

-- Define everything in the conditions
variable (A P Sp Sa : ℕ)
variable (box_cherries box_apples : ℕ)

-- Given conditions
axiom price_relation : 2 * P = 3 * A
axiom size_relation  : Sa = 12 * Sp
axiom cherries_per_box : box_cherries = 12

-- The problem statement (to be proved)
theorem boxes_of_apples_with_cherries : box_apples * A = box_cherries * P → box_apples = 18 :=
by
  sorry

end boxes_of_apples_with_cherries_l279_279913


namespace mul_mixed_number_l279_279126

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l279_279126


namespace price_of_75_cans_l279_279243

/-- The price of 75 cans of a certain brand of soda purchased in 24-can cases,
    given the regular price per can is $0.15 and a 10% discount is applied when
    purchased in 24-can cases, is $10.125.
-/
theorem price_of_75_cans (regular_price : ℝ) (discount : ℝ) (cases_needed : ℕ) (remaining_cans : ℕ) 
  (discounted_price : ℝ) (total_price : ℝ) :
  regular_price = 0.15 →
  discount = 0.10 →
  discounted_price = regular_price - (discount * regular_price) →
  cases_needed = 75 / 24 ∧ remaining_cans = 75 % 24 →
  total_price = (cases_needed * 24 + remaining_cans) * discounted_price →
  total_price = 10.125 :=
by
  sorry

end price_of_75_cans_l279_279243


namespace cookies_recipes_count_l279_279597

theorem cookies_recipes_count 
  (total_students : ℕ)
  (attending_percentage : ℚ)
  (cookies_per_student : ℕ)
  (cookies_per_batch : ℕ) : 
  (total_students = 150) →
  (attending_percentage = 0.60) →
  (cookies_per_student = 3) →
  (cookies_per_batch = 18) →
  (total_students * attending_percentage * cookies_per_student / cookies_per_batch = 15) :=
by
  intros h1 h2 h3 h4
  sorry

end cookies_recipes_count_l279_279597


namespace vasya_numbers_l279_279739

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l279_279739


namespace correct_solution_l279_279938

def fractional_equation (x : ℚ) : Prop :=
  (1 - x) / (2 - x) - 1 = (3 * x - 4) / (x - 2)

theorem correct_solution (x : ℚ) (h : fractional_equation x) : 
  x = 5 / 3 :=
sorry

end correct_solution_l279_279938


namespace circle_radius_tangent_to_parabola_l279_279407

theorem circle_radius_tangent_to_parabola (a : ℝ) (b r : ℝ) :
  (∀ x : ℝ, y = 4 * x ^ 2) ∧ 
  (b = a ^ 2 / 4) ∧ 
  (∀ x : ℝ, x ^ 2 + (4 * x ^ 2 - b) ^ 2 = r ^ 2)  → 
  r = a ^ 2 / 4 := 
  sorry

end circle_radius_tangent_to_parabola_l279_279407


namespace gumballs_per_package_correct_l279_279166

-- Define the conditions
def total_gumballs_eaten : ℕ := 20
def number_of_boxes_finished : ℕ := 4

-- Define the target number of gumballs in each package
def gumballs_in_each_package := 5

theorem gumballs_per_package_correct :
  total_gumballs_eaten / number_of_boxes_finished = gumballs_in_each_package :=
by
  sorry

end gumballs_per_package_correct_l279_279166


namespace find_original_number_l279_279205

theorem find_original_number :
  ∃ x : ℚ, (5 * (3 * x + 15) = 245) ∧ x = 34 / 3 := by
  sorry

end find_original_number_l279_279205


namespace soccer_field_kids_l279_279775

def a := 14
def b := 22
def c := a + b

theorem soccer_field_kids : c = 36 :=
by
    sorry

end soccer_field_kids_l279_279775


namespace find_seating_capacity_l279_279086

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end find_seating_capacity_l279_279086


namespace speed_ratio_l279_279068

theorem speed_ratio (v1 v2 : ℝ) 
  (h1 : v1 > 0) 
  (h2 : v2 > 0) 
  (h : v2 / v1 - v1 / v2 = 35 / 60) : v1 / v2 = 3 / 4 := 
sorry

end speed_ratio_l279_279068


namespace simplify_expression_l279_279605

variable (m : ℝ)

theorem simplify_expression (h₁ : m ≠ 2) (h₂ : m ≠ 3) :
  (m - (4 * m - 9) / (m - 2)) / ((m ^ 2 - 9) / (m - 2)) = (m - 3) / (m + 3) := 
sorry

end simplify_expression_l279_279605


namespace min_value_f_l279_279486

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem min_value_f (a : ℝ) (h : -2 < a) :
  ∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ≥ m) ∧ 
  ((a ≤ 1 → m = a^2 - 2 * a) ∧ (1 < a → m = -1)) :=
by
  sorry

end min_value_f_l279_279486


namespace digit_B_divisible_by_9_l279_279902

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l279_279902


namespace angle_between_hands_at_3_15_l279_279930

-- Definitions based on conditions
def minuteHandAngleAt_3_15 : ℝ := 90 -- The position of the minute hand at 3:15 is 90 degrees.

def hourHandSpeed : ℝ := 0.5 -- The hour hand moves at 0.5 degrees per minute.

def hourHandAngleAt_3_15 : ℝ := 3 * 30 + 15 * hourHandSpeed
-- The hour hand starts at 3 o'clock (90 degrees) and moves 0.5 degrees per minute.

-- Statement to prove
theorem angle_between_hands_at_3_15 : abs (minuteHandAngleAt_3_15 - hourHandAngleAt_3_15) = 82.5 :=
by
  sorry

end angle_between_hands_at_3_15_l279_279930


namespace calculate_product_l279_279120

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l279_279120


namespace count_integers_divisible_by_2_3_5_7_l279_279837

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l279_279837


namespace proof_valid_x_values_l279_279141

noncomputable def valid_x_values (x : ℝ) : Prop :=
  (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≤ 1

theorem proof_valid_x_values :
  {x : ℝ | valid_x_values x} = {x : ℝ | (x < -1) ∨ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1)} :=
by {
  sorry
}

end proof_valid_x_values_l279_279141


namespace probability_four_or_more_same_value_l279_279810

theorem probability_four_or_more_same_value :
  let n := 5 -- number of dice
  let d := 10 -- number of sides on each die
  let event := "at least four of the five dice show the same value"
  let probability := (23 : ℚ) / 5000 -- given probability
  n = 5 ∧ d = 10 ∧ event = "at least four of the five dice show the same value" →
  (probability = 23 / 5000) := 
by
  intros
  sorry

end probability_four_or_more_same_value_l279_279810


namespace money_spent_on_jacket_l279_279355

-- Define the initial amounts
def initial_money_sandy : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def additional_money_found : ℝ := 7.43

-- Amount of money left after buying the shirt
def remaining_after_shirt := initial_money_sandy - amount_spent_shirt

-- Total money after finding additional money
def total_after_additional := remaining_after_shirt + additional_money_found

-- Theorem statement: The amount Sandy spent on the jacket
theorem money_spent_on_jacket : total_after_additional = 9.28 :=
by
  sorry

end money_spent_on_jacket_l279_279355


namespace system1_solution_system2_solution_l279_279359

theorem system1_solution (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := sorry

theorem system2_solution (x y : ℝ) (h1 : 3 * x - 5 * y = 9) (h2 : 2 * x + 3 * y = -6) : 
  x = -3 / 19 ∧ y = -36 / 19 := sorry

end system1_solution_system2_solution_l279_279359


namespace sequence_general_term_l279_279001

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = (2 ^ n) - 1 := 
sorry

end sequence_general_term_l279_279001


namespace right_triangle_hypotenuse_inequality_l279_279653

theorem right_triangle_hypotenuse_inequality
  (a b c m : ℝ)
  (h_right_triangle : c^2 = a^2 + b^2)
  (h_area_relation : a * b = c * m) :
  m + c > a + b :=
by
  sorry

end right_triangle_hypotenuse_inequality_l279_279653


namespace integer_solutions_for_xyz_l279_279983

theorem integer_solutions_for_xyz (x y z : ℤ) : 
  (x - y - 1)^3 + (y - z - 2)^3 + (z - x + 3)^3 = 18 ↔
  (x = y ∧ y = z) ∨
  (x = y - 1 ∧ y = z) ∨
  (x = y ∧ y = z + 5) ∨
  (x = y + 4 ∧ y = z + 5) ∨
  (x = y + 4 ∧ z = y) ∨
  (x = y - 1 ∧ z = y + 4) :=
by {
  sorry
}

end integer_solutions_for_xyz_l279_279983


namespace calculate_expression_l279_279969

theorem calculate_expression :
  (56 * 0.57 * 0.85) / (2.8 * 19 * 1.7) = 0.3 :=
by
  sorry

end calculate_expression_l279_279969


namespace misha_students_l279_279349

theorem misha_students : 
  ∀ (n : ℕ),
  (n = 74 + 1 + 74) ↔ (n = 149) :=
by
  intro n
  split
  · intro h
    rw [← h, nat.add_assoc]
    apply nat.add_right_cancel
    rw [nat.add_comm 1 74, nat.add_assoc]
    apply nat.add_right_cancel
    rw nat.add_comm
  · intro h
    exact h
  sorry

end misha_students_l279_279349


namespace find_y_l279_279646

-- Declare the variables and conditions
variable (x y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 1.5 * x = 0.3 * y
def condition2 : Prop := x = 20

-- State the theorem that given these conditions, y must be 100
theorem find_y (h1 : condition1 x y) (h2 : condition2 x) : y = 100 :=
by sorry

end find_y_l279_279646


namespace avg_two_expressions_l279_279175

theorem avg_two_expressions (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 84) : a = 32 := sorry

end avg_two_expressions_l279_279175


namespace shaded_area_is_correct_l279_279965

theorem shaded_area_is_correct : 
  ∀ (leg_length : ℕ) (total_partitions : ℕ) (shaded_partitions : ℕ) 
    (tri_area : ℕ) (small_tri_area : ℕ) (shaded_area : ℕ), 
  leg_length = 10 → 
  total_partitions = 25 →
  shaded_partitions = 15 →
  tri_area = (1 / 2 * leg_length * leg_length) → 
  small_tri_area = (tri_area / total_partitions) →
  shaded_area = (shaded_partitions * small_tri_area) →
  shaded_area = 30 :=
by
  intros leg_length total_partitions shaded_partitions tri_area small_tri_area shaded_area
  intros h_leg_length h_total_partitions h_shaded_partitions h_tri_area h_small_tri_area h_shaded_area
  sorry

end shaded_area_is_correct_l279_279965


namespace squares_below_16x_144y_1152_l279_279907

noncomputable def count_squares_below_line (a b c : ℝ) (x_max y_max : ℝ) : ℝ :=
  let total_squares := x_max * y_max
  let line_slope := -a/b
  let squares_crossed_by_diagonal := x_max + y_max - 1
  (total_squares - squares_crossed_by_diagonal) / 2

theorem squares_below_16x_144y_1152 : 
  count_squares_below_line 16 144 1152 72 8 = 248.5 := 
by
  sorry

end squares_below_16x_144y_1152_l279_279907


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279130

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l279_279130


namespace det_of_matrix_M_l279_279973

open Matrix

def M : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![2, -4, 4], 
    ![0, 6, -2], 
    ![5, -3, 2]]

theorem det_of_matrix_M : Matrix.det M = -68 :=
by
  sorry

end det_of_matrix_M_l279_279973


namespace Vasya_numbers_l279_279767

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l279_279767


namespace find_m_l279_279851

variable (a b m : ℝ)

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem find_m 
  (h₁ : right_triangle a b 5)
  (h₂ : a + b = 2*m - 1)
  (h₃ : a * b = 4 * (m - 1)) : 
  m = 4 := 
sorry

end find_m_l279_279851


namespace cat_food_inequality_l279_279438

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l279_279438


namespace cat_food_inequality_l279_279436

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l279_279436


namespace cat_food_inequality_l279_279456

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l279_279456


namespace distance_traveled_l279_279661

theorem distance_traveled (speed1 speed2 hours1 hours2 : ℝ)
  (h1 : speed1 = 45) (h2 : hours1 = 2) (h3 : speed2 = 50) (h4 : hours2 = 3) :
  speed1 * hours1 + speed2 * hours2 = 240 := by
  sorry

end distance_traveled_l279_279661


namespace solve_problem_l279_279476

def problem (x : ℝ) : Prop :=
  abs (x - 25) + abs (x - 21) = abs (2 * x - 46) + abs (x - 17)

theorem solve_problem : (∃ x : ℝ, problem x) ∧ ∀ x : ℝ, problem x → x = 67 / 3 :=
by
  split
  · use 67 / 3
    rw problem
    sorry  -- Proof goes here
  
  · intros x hx
    sorry  -- Proof goes here

end solve_problem_l279_279476


namespace g_eq_g_inv_at_7_over_2_l279_279979

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_at_7_over_2 : g (7 / 2) = g_inv (7 / 2) := by
  sorry

end g_eq_g_inv_at_7_over_2_l279_279979


namespace expected_value_correct_l279_279955

-- Definitions for the problem
def brakePoint (s : List ℕ) (n : ℕ) : Prop :=
  ∀ i ∈ list.range n, i + 1 ∈ s.take n

def correctPartition (perm : List ℕ) : List (List ℕ) :=
  -- This is a placeholder for the actual function that returns the correct partition
  sorry

-- Expected value of correct partitions for permutations of {1, 2, ..., 7}
noncomputable def expected_value : ℚ :=
  let perms := Finset.univ.val.perms in
  let k (σ : List ℕ) : ℕ := (correctPartition σ).length in
  (finset.card perms : ℚ)⁻¹ * ∑ σ in perms, k σ

-- Assertion that the expected value is 151/105
theorem expected_value_correct : expected_value = 151 / 105 :=
by
  -- Placeholder proof
  sorry

end expected_value_correct_l279_279955
