import Mathlib

namespace solution_set_l153_153894

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom monotone_decreasing_f : ∀ {a b : ℝ}, 0 ≤ a → a ≤ b → f b ≤ f a
axiom f_half_eq_zero : f (1 / 2) = 0

theorem solution_set :
  { x : ℝ | f (Real.log x / Real.log (1 / 4)) < 0 } = 
  { x : ℝ | 0 < x ∧ x < 1 / 2 } ∪ { x : ℝ | 2 < x } :=
by
  sorry

end solution_set_l153_153894


namespace smallest_integer_solution_l153_153684

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) → y = 2 := by
  sorry

end smallest_integer_solution_l153_153684


namespace triangle_area_ab_l153_153232

theorem triangle_area_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0.5 * (12 / a) * (12 / b) = 12) : a * b = 6 :=
by
  sorry

end triangle_area_ab_l153_153232


namespace total_members_in_sports_club_l153_153780

-- Definitions as per the conditions
def B : ℕ := 20 -- number of members who play badminton
def T : ℕ := 23 -- number of members who play tennis
def Both : ℕ := 7 -- number of members who play both badminton and tennis
def Neither : ℕ := 6 -- number of members who do not play either sport

-- Theorem statement to prove the correct answer
theorem total_members_in_sports_club : B + T - Both + Neither = 42 :=
by
  sorry

end total_members_in_sports_club_l153_153780


namespace solution_set_for_inequality_l153_153218

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem solution_set_for_inequality
  (h1 : is_odd f)
  (h2 : f 2 = 0)
  (h3 : ∀ x > 0, x * deriv f x - f x < 0) :
  {x : ℝ | f x / x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_for_inequality_l153_153218


namespace volume_of_tetrahedron_ABCD_l153_153518

noncomputable def tetrahedron_volume_proof (S: ℝ) (AB AD BD: ℝ) 
    (angle_ABD_DBC_CBA angle_ADB_BDC_CDA angle_ACB_ACD_BCD: ℝ) : ℝ :=
if h1 : S = 1 ∧ AB = AD ∧ BD = (Real.sqrt 2) / 2
    ∧ angle_ABD_DBC_CBA = 180 ∧ angle_ADB_BDC_CDA = 180 
    ∧ angle_ACB_ACD_BCD = 90 then
  (1 / 24)
else
  0

-- Statement to prove
theorem volume_of_tetrahedron_ABCD : tetrahedron_volume_proof 1 AB AD ((Real.sqrt 2) / 2) 180 180 90 = (1 / 24) :=
by sorry

end volume_of_tetrahedron_ABCD_l153_153518


namespace expression_divisible_by_x_minus_1_squared_l153_153596

theorem expression_divisible_by_x_minus_1_squared :
  ∀ (n : ℕ) (x : ℝ), x ≠ 1 →
  (n * x^(n + 1) * (1 - 1 / x) - x^n * (1 - 1 / x^n)) / (x - 1)^2 = 
  (n * x^(n + 1) - n * x^n - x^n + 1) / (x - 1)^2 :=
by
  intro n x hx_ne_1
  sorry

end expression_divisible_by_x_minus_1_squared_l153_153596


namespace fourth_function_form_l153_153374

variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)
variable (hf : Function.LeftInverse f_inv f)
variable (hf_inv : Function.RightInverse f_inv f)

theorem fourth_function_form :
  (∀ x, y = (-(f (-x - 1)) + 2) ↔ y = f_inv (x + 2) + 1 ↔ -(x + y) = 0) :=
  sorry

end fourth_function_form_l153_153374


namespace polynomial_prime_is_11_l153_153921

def P (a : ℕ) : ℕ := a^4 - 4 * a^3 + 15 * a^2 - 30 * a + 27

theorem polynomial_prime_is_11 (a : ℕ) (hp : Nat.Prime (P a)) : P a = 11 := 
by {
  sorry
}

end polynomial_prime_is_11_l153_153921


namespace quad_relation_l153_153617

theorem quad_relation
  (α AI BI CI DI : ℝ)
  (h1 : AB = α * (AI / CI + BI / DI))
  (h2 : BC = α * (BI / DI + CI / AI))
  (h3 : CD = α * (CI / AI + DI / BI))
  (h4 : DA = α * (DI / BI + AI / CI)) :
  AB + CD = AD + BC := by
  sorry

end quad_relation_l153_153617


namespace pencils_given_l153_153772

-- Define the conditions
def a : Nat := 9
def b : Nat := 65

-- Define the goal statement: the number of pencils Kathryn gave to Anthony
theorem pencils_given (a b : Nat) (h₁ : a = 9) (h₂ : b = 65) : b - a = 56 :=
by
  -- Omitted proof part
  sorry

end pencils_given_l153_153772


namespace arithmetic_geometric_sequence_l153_153504

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 1 = 3)
    (h2 : a 1 + a 3 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 * a 4 = 36 := 
sorry

end arithmetic_geometric_sequence_l153_153504


namespace crayons_problem_l153_153306

theorem crayons_problem
  (S M L : ℕ)
  (hS_condition : (3 / 5 : ℚ) * S = 60)
  (hM_condition : (1 / 4 : ℚ) * M = 98)
  (hL_condition : (4 / 7 : ℚ) * L = 168) :
  S = 100 ∧ M = 392 ∧ L = 294 ∧ ((2 / 5 : ℚ) * S + (3 / 4 : ℚ) * M + (3 / 7 : ℚ) * L = 460) := 
by
  sorry

end crayons_problem_l153_153306


namespace max_value_expression_l153_153118

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (eq_condition : x^2 - 3 * x * y + 4 * y^2 - z = 0) : 
  ∃ (M : ℝ), M = 1 ∧ (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x^2 - 3 * x * y + 4 * y^2 - z = 0 → (2/x + 1/y - 2/z) ≤ M) := 
by
  sorry

end max_value_expression_l153_153118


namespace TimTotalRunHoursPerWeek_l153_153710

def TimUsedToRunTimesPerWeek : ℕ := 3
def TimAddedExtraDaysPerWeek : ℕ := 2
def MorningRunHours : ℕ := 1
def EveningRunHours : ℕ := 1

theorem TimTotalRunHoursPerWeek :
  (TimUsedToRunTimesPerWeek + TimAddedExtraDaysPerWeek) * (MorningRunHours + EveningRunHours) = 10 :=
by
  sorry

end TimTotalRunHoursPerWeek_l153_153710


namespace abs_sum_zero_implies_diff_eq_five_l153_153204

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end abs_sum_zero_implies_diff_eq_five_l153_153204


namespace num_tickets_bought_l153_153611

-- Defining the cost and discount conditions
def ticket_cost : ℝ := 40
def discount_rate : ℝ := 0.05
def total_paid : ℝ := 476
def base_tickets : ℕ := 10

-- Definition to calculate the cost of the first 10 tickets
def cost_first_10_tickets : ℝ := base_tickets * ticket_cost
-- Definition of the discounted price for tickets exceeding 10
def discounted_ticket_cost : ℝ := ticket_cost * (1 - discount_rate)
-- Definition of the total cost for the tickets exceeding 10
def cost_discounted_tickets (num_tickets_exceeding_10 : ℕ) : ℝ := num_tickets_exceeding_10 * discounted_ticket_cost
-- Total amount spent on the tickets exceeding 10
def amount_spent_on_discounted_tickets : ℝ := total_paid - cost_first_10_tickets

-- Main theorem statement proving the total number of tickets Mr. Benson bought
theorem num_tickets_bought : ∃ x : ℕ, x = base_tickets + (amount_spent_on_discounted_tickets / discounted_ticket_cost) ∧ x = 12 := 
by
  sorry

end num_tickets_bought_l153_153611


namespace ferry_distance_l153_153154

theorem ferry_distance 
  (x : ℝ)
  (v_w : ℝ := 3)  -- speed of water flow in km/h
  (t_downstream : ℝ := 5)  -- time taken to travel downstream in hours
  (t_upstream : ℝ := 7)  -- time taken to travel upstream in hours
  (eqn : x / t_downstream - v_w = x / t_upstream + v_w) :
  x = 105 :=
sorry

end ferry_distance_l153_153154


namespace min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l153_153916

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Prove the minimum value for a = 1 and x in [-1, 0]
theorem min_f_a_eq_1 : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f 1 x ≥ 5 :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when a ≤ -1
theorem min_f_a_le_neg1 (h : ∀ a : ℝ, a ≤ -1) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a (-1) ≤ f a x :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when -1 < a < 0
theorem min_f_neg1_lt_a_lt_0 (h : ∀ a : ℝ, -1 < a ∧ a < 0) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a a ≤ f a x :=
by
  sorry

end min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l153_153916


namespace inequality_problem_l153_153705

theorem inequality_problem (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_sum : a + b + c + d = 4) : 
    a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := 
sorry

end inequality_problem_l153_153705


namespace ladder_slip_l153_153652

theorem ladder_slip 
  (ladder_length : ℝ) 
  (initial_base : ℝ) 
  (slip_height : ℝ) 
  (h_length : ladder_length = 30) 
  (h_base : initial_base = 11) 
  (h_slip : slip_height = 6) 
  : ∃ (slide_distance : ℝ), abs (slide_distance - 9.49) < 0.01 :=
by
  let initial_height := Real.sqrt (ladder_length^2 - initial_base^2)
  let new_height := initial_height - slip_height
  let new_base := Real.sqrt (ladder_length^2 - new_height^2)
  let slide_distance := new_base - initial_base
  use slide_distance
  have h_approx : abs (slide_distance - 9.49) < 0.01 := sorry
  exact h_approx

end ladder_slip_l153_153652


namespace solve_inequality_l153_153868

theorem solve_inequality (x : ℝ) (h : 0 < x ∧ x < 2) : abs (2 * x - 1) < abs x + 1 :=
by
  sorry

end solve_inequality_l153_153868


namespace angle_trig_identity_l153_153302

theorem angle_trig_identity
  (A B C : ℝ)
  (h_sum : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 = Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 - 
                       2 * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2) :=
by
  sorry

end angle_trig_identity_l153_153302


namespace talia_drives_total_distance_l153_153133

-- Define the distances for each leg of the trip
def distance_house_to_park : ℕ := 5
def distance_park_to_store : ℕ := 3
def distance_store_to_friend : ℕ := 6
def distance_friend_to_house : ℕ := 4

-- Define the total distance Talia drives
def total_distance := distance_house_to_park + distance_park_to_store + distance_store_to_friend + distance_friend_to_house

-- Prove that the total distance is 18 miles
theorem talia_drives_total_distance : total_distance = 18 := by
  sorry

end talia_drives_total_distance_l153_153133


namespace number_of_biscuits_per_day_l153_153090

theorem number_of_biscuits_per_day 
  (price_cupcake : ℝ) (price_cookie : ℝ) (price_biscuit : ℝ)
  (cupcakes_per_day : ℕ) (cookies_per_day : ℕ) (total_earnings_five_days : ℝ) :
  price_cupcake = 1.5 → 
  price_cookie = 2 → 
  price_biscuit = 1 → 
  cupcakes_per_day = 20 → 
  cookies_per_day = 10 → 
  total_earnings_five_days = 350 →
  (total_earnings_five_days - 
   (5 * (cupcakes_per_day * price_cupcake + cookies_per_day * price_cookie))) / (5 * price_biscuit) = 20 :=
by
  intros price_cupcake_eq price_cookie_eq price_biscuit_eq cupcakes_per_day_eq cookies_per_day_eq total_earnings_five_days_eq
  sorry

end number_of_biscuits_per_day_l153_153090


namespace smallest_top_block_number_l153_153373

-- Define the pyramid structure and number assignment problem
def block_pyramid : Type := sorry

-- Given conditions:
-- 4 layers, specific numberings, and block support structure.
structure Pyramid :=
  (Layer1 : Fin 16 → ℕ)
  (Layer2 : Fin 9 → ℕ)
  (Layer3 : Fin 4 → ℕ)
  (TopBlock : ℕ)

-- Constraints on block numbers
def is_valid (P : Pyramid) : Prop :=
  -- base layer numbers are from 1 to 16
  (∀ i, 1 ≤ P.Layer1 i ∧ P.Layer1 i ≤ 16) ∧
  -- each above block is the sum of directly underlying neighboring blocks
  (∀ i, P.Layer2 i = P.Layer1 (i * 3) + P.Layer1 (i * 3 + 1) + P.Layer1 (i * 3 + 2)) ∧
  (∀ i, P.Layer3 i = P.Layer2 (i * 3) + P.Layer2 (i * 3 + 1) + P.Layer2 (i * 3 + 2)) ∧
  P.TopBlock = P.Layer3 0 + P.Layer3 1 + P.Layer3 2 + P.Layer3 3

-- Statement of the theorem
theorem smallest_top_block_number : ∃ P : Pyramid, is_valid P ∧ P.TopBlock = ComputedValue := sorry

end smallest_top_block_number_l153_153373


namespace x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l153_153982

variable (x : ℝ)

theorem x_positive_implies_abs_positive (hx : x > 0) : |x| > 0 := sorry

theorem abs_positive_not_necessiarily_x_positive : (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := sorry

theorem x_positive_is_sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ 
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := 
  ⟨x_positive_implies_abs_positive, abs_positive_not_necessiarily_x_positive⟩

end x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l153_153982


namespace irrational_b_eq_neg_one_l153_153964

theorem irrational_b_eq_neg_one
  (a : ℝ) (b : ℝ)
  (h_irrational : ¬ ∃ q : ℚ, a = (q : ℝ))
  (h_eq : ab + a - b = 1) :
  b = -1 :=
sorry

end irrational_b_eq_neg_one_l153_153964


namespace exponent_form_l153_153840

theorem exponent_form (y : ℕ) (w : ℕ) (k : ℕ) : w = 3 ^ y → w % 10 = 7 → ∃ (k : ℕ), y = 4 * k + 3 :=
by
  intros h1 h2
  sorry

end exponent_form_l153_153840


namespace sad_girls_count_l153_153171

-- Given definitions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def boys_neither_happy_nor_sad : ℕ := 10

-- Intermediate definitions
def sad_boys : ℕ := boys - happy_boys - boys_neither_happy_nor_sad
def sad_girls : ℕ := sad_children - sad_boys

-- Theorem to prove that the number of sad girls is 4
theorem sad_girls_count : sad_girls = 4 := by
  sorry

end sad_girls_count_l153_153171


namespace smallest_norm_value_l153_153458

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l153_153458


namespace yola_past_weight_l153_153665

-- Definitions based on the conditions
def current_weight_yola : ℕ := 220
def weight_difference_current (D : ℕ) : ℕ := 30
def weight_difference_past (D : ℕ) : ℕ := D

-- Main statement
theorem yola_past_weight (D : ℕ) :
  (250 - D) = (current_weight_yola + weight_difference_current D - weight_difference_past D) :=
by
  sorry

end yola_past_weight_l153_153665


namespace ordering_9_8_4_12_3_16_l153_153919

theorem ordering_9_8_4_12_3_16 : (4 ^ 12 < 9 ^ 8) ∧ (9 ^ 8 = 3 ^ 16) :=
by {
  sorry
}

end ordering_9_8_4_12_3_16_l153_153919


namespace weight_cut_percentage_unknown_l153_153826

-- Define the initial conditions
def original_speed : ℝ := 150
def new_speed : ℝ := 205
def increase_supercharge : ℝ := original_speed * 0.3
def speed_after_supercharge : ℝ := original_speed + increase_supercharge
def increase_weight_cut : ℝ := new_speed - speed_after_supercharge

-- Theorem statement
theorem weight_cut_percentage_unknown : 
  (original_speed = 150) →
  (new_speed = 205) →
  (increase_supercharge = 150 * 0.3) →
  (speed_after_supercharge = 150 + increase_supercharge) →
  (increase_weight_cut = 205 - speed_after_supercharge) →
  increase_weight_cut = 10 →
  sorry := 
by
  intros h_orig h_new h_inc_scharge h_speed_scharge h_inc_weight h_inc_10
  sorry

end weight_cut_percentage_unknown_l153_153826


namespace inequality_proof_l153_153877

noncomputable def given_condition_1 (a b c u : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ x, (a * x^2 - b * x + c = 0)) ∧
  a * u^2 - b * u + c ≤ 0

noncomputable def given_condition_2 (A B C v : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ (∃ x, (A * x^2 - B * x + C = 0)) ∧
  A * v^2 - B * v + C ≤ 0

theorem inequality_proof (a b c A B C u v : ℝ) (h1 : given_condition_1 a b c u) (h2 : given_condition_2 A B C v) :
  (a * u + A * v) * (c / u + C / v) ≤ (b + B) ^ 2 / 4 :=
by
    sorry

end inequality_proof_l153_153877


namespace tyson_one_point_count_l153_153699

def tyson_three_points := 3 * 15
def tyson_two_points := 2 * 12
def total_points := 75
def points_from_three_and_two := tyson_three_points + tyson_two_points

theorem tyson_one_point_count :
  ∃ n : ℕ, n % 2 = 0 ∧ (n = total_points - points_from_three_and_two) :=
sorry

end tyson_one_point_count_l153_153699


namespace andy_max_cookies_l153_153290

theorem andy_max_cookies (total_cookies : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ)
  (h1 : total_cookies = 30)
  (h2 : bella_cookies = 2 * andy_cookies)
  (h3 : andy_cookies + bella_cookies = total_cookies) :
  andy_cookies = 10 := by
  sorry

end andy_max_cookies_l153_153290


namespace shaded_fraction_is_one_eighth_l153_153408

noncomputable def total_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

noncomputable def half_area (length : ℕ) (width : ℕ) : ℚ :=
  total_area length width / 2

noncomputable def shaded_area (length : ℕ) (width : ℕ) : ℚ :=
  half_area length width / 4

theorem shaded_fraction_is_one_eighth : 
  ∀ (length width : ℕ), length = 15 → width = 21 → shaded_area length width / total_area length width = 1 / 8 :=
by
  sorry

end shaded_fraction_is_one_eighth_l153_153408


namespace remainder_24_2377_mod_15_l153_153575

theorem remainder_24_2377_mod_15 :
  24^2377 % 15 = 9 :=
sorry

end remainder_24_2377_mod_15_l153_153575


namespace check_true_propositions_l153_153835

open Set

theorem check_true_propositions : 
  ∀ (Prop1 Prop2 Prop3 : Prop),
    (Prop1 ↔ (∀ x : ℝ, x^2 > 0)) →
    (Prop2 ↔ ∃ x : ℝ, x^2 ≤ x) →
    (Prop3 ↔ ∀ (M N : Set ℝ) (x : ℝ), x ∈ (M ∩ N) → x ∈ M ∧ x ∈ N) →
    (¬Prop1 ∧ Prop2 ∧ Prop3) →
    (2 = 2) := sorry

end check_true_propositions_l153_153835


namespace four_digit_perfect_square_exists_l153_153935

theorem four_digit_perfect_square_exists (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) (h3 : 101 * x + 100 = y^2) : 
  ∃ n, n = 8281 ∧ n = y^2 ∧ (((n / 100) : ℕ) = ((n % 100) : ℕ) + 1) :=
by 
  sorry

end four_digit_perfect_square_exists_l153_153935


namespace c_work_rate_l153_153126

variable {W : ℝ} -- Denoting the work by W
variable {a_rate : ℝ} -- Work rate of a
variable {b_rate : ℝ} -- Work rate of b
variable {c_rate : ℝ} -- Work rate of c
variable {combined_rate : ℝ} -- Combined work rate of a, b, and c

theorem c_work_rate (W a_rate b_rate c_rate combined_rate : ℝ)
  (h1 : a_rate = W / 12)
  (h2 : b_rate = W / 24)
  (h3 : combined_rate = W / 4)
  (h4 : combined_rate = a_rate + b_rate + c_rate) :
  c_rate = W / 4.5 :=
by
  sorry

end c_work_rate_l153_153126


namespace jenny_money_l153_153646

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end jenny_money_l153_153646


namespace greatest_value_q_minus_r_l153_153616

theorem greatest_value_q_minus_r : ∃ q r : ℕ, 1043 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 37) :=
by {
  sorry
}

end greatest_value_q_minus_r_l153_153616


namespace melted_mixture_weight_l153_153910

theorem melted_mixture_weight (Z C : ℝ) (ratio : 9 / 11 = Z / C) (zinc_weight : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l153_153910


namespace total_students_l153_153209

theorem total_students (T : ℝ) (h1 : 0.3 * T =  0.7 * T - 616) : T = 880 :=
by sorry

end total_students_l153_153209


namespace geometric_sequence_sum_l153_153541

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (ha1 : q ≠ 0)
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 3 + a 4 = (a 1 + a 2) * q^2)
  : a 5 + a 6 = 48 :=
by
  sorry

end geometric_sequence_sum_l153_153541


namespace kangaroo_can_jump_1000_units_l153_153410

noncomputable def distance (x y : ℕ) : ℕ := x + y

def valid_small_jump (x y : ℕ) : Prop :=
  x + 1 ≥ 0 ∧ y - 1 ≥ 0

def valid_big_jump (x y : ℕ) : Prop :=
  x - 5 ≥ 0 ∧ y + 7 ≥ 0

theorem kangaroo_can_jump_1000_units (x y : ℕ) (h : x + y > 6) :
  distance x y ≥ 1000 :=
sorry

end kangaroo_can_jump_1000_units_l153_153410


namespace prove_equation_l153_153740

theorem prove_equation (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (2 * x + 5) = 3 / 5 :=
by
  sorry

end prove_equation_l153_153740


namespace hall_volume_l153_153010

theorem hall_volume (length breadth : ℝ) (height : ℝ := 20 / 3)
  (h1 : length = 15)
  (h2 : breadth = 12)
  (h3 : 2 * (length * breadth) = 54 * height) :
  length * breadth * height = 8004 :=
by
  sorry

end hall_volume_l153_153010


namespace value_of_expression_l153_153217

theorem value_of_expression
  (a b c : ℝ)
  (h1 : |a - b| = 1)
  (h2 : |b - c| = 1)
  (h3 : |c - a| = 2)
  (h4 : a * b * c = 60) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c) = 1 / 10 :=
sorry

end value_of_expression_l153_153217


namespace blue_pill_cost_l153_153673

theorem blue_pill_cost (y : ℕ) :
  -- Conditions
  (∀ t d : ℕ, t = 21 → 
     d = 14 → 
     (735 - d * 2 = t * ((2 * y) + (y + 2)) / t) →
     2 * y + (y + 2) = 35) →
  -- Conclusion
  y = 11 :=
by
  sorry

end blue_pill_cost_l153_153673


namespace find_number_l153_153106

noncomputable def number_with_point_one_percent (x : ℝ) : Prop :=
  0.1 * x / 100 = 12.356

theorem find_number :
  ∃ x : ℝ, number_with_point_one_percent x ∧ x = 12356 :=
by
  sorry

end find_number_l153_153106


namespace number_of_green_eyes_l153_153000

-- Definitions based on conditions
def total_people : Nat := 100
def blue_eyes : Nat := 19
def brown_eyes : Nat := total_people / 2
def black_eyes : Nat := total_people / 4

-- Theorem stating the main question and its answer
theorem number_of_green_eyes : 
  (total_people - (blue_eyes + brown_eyes + black_eyes)) = 6 := by
  sorry

end number_of_green_eyes_l153_153000


namespace Mina_digits_l153_153793

theorem Mina_digits (Carlos Sam Mina : ℕ) 
  (h1 : Sam = Carlos + 6) 
  (h2 : Mina = 6 * Carlos) 
  (h3 : Sam = 10) : 
  Mina = 24 := 
sorry

end Mina_digits_l153_153793


namespace probability_exactly_two_heads_and_two_tails_l153_153765

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end probability_exactly_two_heads_and_two_tails_l153_153765


namespace interval_of_y_l153_153244

theorem interval_of_y (y : ℝ) (h : y = (1 / y) * (-y) - 5) : -6 ≤ y ∧ y ≤ -4 :=
by sorry

end interval_of_y_l153_153244


namespace jerrys_breakfast_calories_l153_153901

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end jerrys_breakfast_calories_l153_153901


namespace b5_plus_b9_l153_153348

variable {a : ℕ → ℕ} -- Geometric sequence
variable {b : ℕ → ℕ} -- Arithmetic sequence

axiom geom_progression {r x y : ℕ} : a x = a 1 * r^(x - 1) ∧ a y = a 1 * r^(y - 1)
axiom arith_progression {d x y : ℕ} : b x = b 1 + d * (x - 1) ∧ b y = b 1 + d * (y - 1)

axiom a3a11_equals_4a7 : a 3 * a 11 = 4 * a 7
axiom a7_equals_b7 : a 7 = b 7

theorem b5_plus_b9 : b 5 + b 9 = 8 := by
  apply sorry

end b5_plus_b9_l153_153348


namespace plane_intercept_equation_l153_153158

-- Define the conditions in Lean 4
variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- State the main theorem
theorem plane_intercept_equation :
  ∃ (p : ℝ → ℝ → ℝ → ℝ), (∀ x y z, p x y z = x / a + y / b + z / c) :=
sorry

end plane_intercept_equation_l153_153158


namespace ratio_equation_solution_l153_153593

theorem ratio_equation_solution (x : ℝ) :
  (4 + 2 * x) / (6 + 3 * x) = (2 + x) / (3 + 2 * x) → (x = 0 ∨ x = 4) :=
by
  -- the proof steps would go here
  sorry

end ratio_equation_solution_l153_153593


namespace marble_problem_l153_153052

theorem marble_problem
  (h1 : ∀ x : ℕ, x > 0 → (x + 2) * ((220 / x) - 1) = 220) :
  ∃ x : ℕ, x > 0 ∧ (x + 2) * ((220 / ↑x) - 1) = 220 ∧ x = 20 :=
by
  sorry

end marble_problem_l153_153052


namespace new_average_of_remaining_students_l153_153149

theorem new_average_of_remaining_students 
  (avg_initial_score : ℝ)
  (num_initial_students : ℕ)
  (dropped_score : ℝ)
  (num_remaining_students : ℕ)
  (new_avg_score : ℝ) 
  (h_avg : avg_initial_score = 62.5)
  (h_num_initial : num_initial_students = 16)
  (h_dropped : dropped_score = 55)
  (h_num_remaining : num_remaining_students = 15)
  (h_new_avg : new_avg_score = 63) :
  let total_initial_score := avg_initial_score * num_initial_students
  let total_remaining_score := total_initial_score - dropped_score
  let calculated_new_avg := total_remaining_score / num_remaining_students
  calculated_new_avg = new_avg_score := 
by
  -- The proof will be provided here
  sorry

end new_average_of_remaining_students_l153_153149


namespace ratio_elephants_to_others_l153_153513

theorem ratio_elephants_to_others (L P E : ℕ) (h1 : L = 2 * P) (h2 : L = 200) (h3 : L + P + E = 450) :
  E / (L + P) = 1 / 2 :=
by
  sorry

end ratio_elephants_to_others_l153_153513


namespace people_in_each_column_l153_153491

theorem people_in_each_column
  (P : ℕ)
  (x : ℕ)
  (h1 : P = 16 * x)
  (h2 : P = 12 * 40) :
  x = 30 :=
sorry

end people_in_each_column_l153_153491


namespace work_completion_time_l153_153255

/-
Conditions:
1. A man alone can do the work in 6 days.
2. A woman alone can do the work in 18 days.
3. A boy alone can do the work in 9 days.

Question:
How long will they take to complete the work together?

Correct Answer:
3 days
-/

theorem work_completion_time (M W B : ℕ) (hM : M = 6) (hW : W = 18) (hB : B = 9) : 1 / (1/M + 1/W + 1/B) = 3 := 
by
  sorry

end work_completion_time_l153_153255


namespace lakeside_fitness_center_ratio_l153_153797

theorem lakeside_fitness_center_ratio (f m c : ℕ)
  (h_avg_age : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  f = 3 * (m / 6) ∧ f = 3 * (c / 2) :=
by
  sorry

end lakeside_fitness_center_ratio_l153_153797


namespace ellipse_equation_minimum_distance_l153_153372

-- Define the conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2) / (a^2) + (y^2) / (b^2) = 1)

def eccentricity (a c : ℝ) : Prop :=
  c = a / 2

def focal_distance (c : ℝ) : Prop :=
  2 * c = 4

def foci_parallel (F1 A B C D : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := F1;
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (yA - y1) / (xA - x1) = (yC - y1) / (xC - x1) ∧ 
  (yB - y1) / (xB - x1) = (yD - y1) / (xD - x1)

def orthogonal_vectors (A C B D : ℝ × ℝ) : Prop :=
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (xC - xA) * (xD - xB) + (yC - yA) * (yD - yB) = 0

-- Prove equation of ellipse E
theorem ellipse_equation (a b : ℝ) (x y : ℝ) (c : ℝ)
  (h1 : ellipse a b x y)
  (h2 : eccentricity a c)
  (h3 : focal_distance c) :
  (a = 4) ∧ (b^2 = 12) ∧ (x^2 / 16 + y^2 / 12 = 1) :=
sorry

-- Prove minimum value of |AC| + |BD|
theorem minimum_distance (A B C D : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (h1 : foci_parallel F1 A B C D)
  (h2 : orthogonal_vectors A C B D) :
  |(AC : ℝ)| + |(BD : ℝ)| = 96 / 7 :=
sorry

end ellipse_equation_minimum_distance_l153_153372


namespace subset_of_primes_is_all_primes_l153_153589

theorem subset_of_primes_is_all_primes
  (P : Set ℕ)
  (M : Set ℕ)
  (hP : ∀ n, n ∈ P ↔ Nat.Prime n)
  (hM : ∀ S : Finset ℕ, (∀ p ∈ S, p ∈ M) → ∀ p, p ∣ (Finset.prod S id + 1) → p ∈ M) :
  M = P :=
sorry

end subset_of_primes_is_all_primes_l153_153589


namespace quadratic_equation_in_one_variable_l153_153351

def is_quadratic_in_one_variable (eq : String) : Prop :=
  match eq with
  | "2x^2 + 5y + 1 = 0" => False
  | "ax^2 + bx - c = 0" => ∃ (a b c : ℝ), a ≠ 0
  | "1/x^2 + x = 2" => False
  | "x^2 = 0" => True
  | _ => False

theorem quadratic_equation_in_one_variable :
  is_quadratic_in_one_variable "x^2 = 0" := by
  sorry

end quadratic_equation_in_one_variable_l153_153351


namespace center_of_circumcircle_lies_on_AK_l153_153440

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l153_153440


namespace vertex_of_parabola_l153_153696

theorem vertex_of_parabola :
  ∃ (x y : ℝ), (∀ x : ℝ, y = x^2 - 12 * x + 9) → (x, y) = (6, -27) :=
sorry

end vertex_of_parabola_l153_153696


namespace finance_specialization_percentage_l153_153774

theorem finance_specialization_percentage (F : ℝ) :
  (76 - 43.333333333333336) = (90 - F) → 
  F = 57.333333333333336 :=
by
  sorry

end finance_specialization_percentage_l153_153774


namespace perimeter_ACFHK_is_correct_l153_153389

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the points of the pentagon within the dodecagon
def ACFHK_points : ℕ := 5

-- Define the perimeter of the pentagon ACFHK in the dodecagon
noncomputable def perimeter_of_ACFHK : ℝ :=
  let triangle_side := radius
  let isosceles_right_triangle_side := radius * Real.sqrt 2
  3 * triangle_side + 2 * isosceles_right_triangle_side

-- Verify that the calculated perimeter matches the expected value
theorem perimeter_ACFHK_is_correct : perimeter_of_ACFHK = 18 + 12 * Real.sqrt 2 :=
  sorry

end perimeter_ACFHK_is_correct_l153_153389


namespace find_added_number_l153_153465

def original_number : ℕ := 5
def doubled : ℕ := 2 * original_number
def resultant (added : ℕ) : ℕ := 3 * (doubled + added)
def final_result : ℕ := 57

theorem find_added_number (added : ℕ) (h : resultant added = final_result) : added = 9 :=
sorry

end find_added_number_l153_153465


namespace ny_mets_fans_count_l153_153233

theorem ny_mets_fans_count (Y M R : ℕ) (h1 : 3 * M = 2 * Y) (h2 : 4 * R = 5 * M) (h3 : Y + M + R = 390) : M = 104 := 
by
  sorry

end ny_mets_fans_count_l153_153233


namespace kids_still_awake_l153_153747

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end kids_still_awake_l153_153747


namespace expectation_variance_comparison_l153_153226

variable {p1 p2 : ℝ}
variable {ξ1 ξ2 : ℝ}

theorem expectation_variance_comparison
  (h_p1 : 0 < p1)
  (h_p2 : p1 < p2)
  (h_p3 : p2 < 1 / 2)
  (h_ξ1 : ξ1 = p1)
  (h_ξ2 : ξ2 = p2):
  (ξ1 < ξ2) ∧ (ξ1 * (1 - ξ1) < ξ2 * (1 - ξ2)) := by
  sorry

end expectation_variance_comparison_l153_153226


namespace hall_reunion_attendees_l153_153542

noncomputable def Oates : ℕ := 40
noncomputable def both : ℕ := 10
noncomputable def total : ℕ := 100
noncomputable def onlyOates := Oates - both
noncomputable def onlyHall := total - onlyOates - both
noncomputable def Hall := onlyHall + both

theorem hall_reunion_attendees : Hall = 70 := by {
  sorry
}

end hall_reunion_attendees_l153_153542


namespace triangle_inequality_l153_153529

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by {
  sorry
}

end triangle_inequality_l153_153529


namespace cube_points_l153_153062

theorem cube_points (A B C D E F : ℕ) 
  (h1 : A + B = 13)
  (h2 : C + D = 13)
  (h3 : E + F = 13)
  (h4 : A + C + E = 16)
  (h5 : B + D + E = 24) :
  F = 6 :=
by
  sorry  -- Proof to be filled in by the user

end cube_points_l153_153062


namespace cube_product_l153_153252

/-- A cube is a three-dimensional shape with a specific number of vertices and faces. -/
structure Cube where
  vertices : ℕ
  faces : ℕ

theorem cube_product (C : Cube) (h1: C.vertices = 8) (h2: C.faces = 6) : 
  (C.vertices * C.faces = 48) :=
by sorry

end cube_product_l153_153252


namespace find_f_neg_2_l153_153021

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 2: f is defined on ℝ
-- This is implicitly handled as f : ℝ → ℝ

-- Condition 3: f(x+2) = -f(x)
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_neg_2 (h₁ : odd_function f) (h₂ : periodic_function f) : f (-2) = 0 :=
  sorry

end find_f_neg_2_l153_153021


namespace find_b_minus_a_l153_153211

theorem find_b_minus_a (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a - 9 * b + 18 * a * b = 2018) : b - a = 223 :=
sorry

end find_b_minus_a_l153_153211


namespace value_of_k_for_square_of_binomial_l153_153134

theorem value_of_k_for_square_of_binomial (a k : ℝ) : (x : ℝ) → x^2 - 14 * x + k = (x - a)^2 → k = 49 :=
by
  intro x h
  sorry

end value_of_k_for_square_of_binomial_l153_153134


namespace greatest_possible_value_of_a_l153_153478

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l153_153478


namespace contrapositive_statement_l153_153449

-- Condition definitions
def P (x : ℝ) := x^2 < 1
def Q (x : ℝ) := -1 < x ∧ x < 1
def not_Q (x : ℝ) := x ≤ -1 ∨ x ≥ 1
def not_P (x : ℝ) := x^2 ≥ 1

theorem contrapositive_statement (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_statement_l153_153449


namespace simplify_expression_l153_153219

theorem simplify_expression : ((3 * 2 + 4 + 6) / 3 - 2 / 3) = 14 / 3 := by
  sorry

end simplify_expression_l153_153219


namespace measure_of_angle_D_in_scalene_triangle_l153_153520

-- Define the conditions
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ E ≠ F ∧ D ≠ F

-- Define the measure of angles based on the given conditions
def measure_of_angle_D (D E F : ℝ) : Prop :=
  E = 2 * D ∧ F = 40

-- Define the sum of angles in a triangle
def triangle_angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

theorem measure_of_angle_D_in_scalene_triangle (D E F : ℝ) (h_scalene : is_scalene D E F) 
  (h_measures : measure_of_angle_D D E F) (h_sum : triangle_angle_sum D E F) : D = 140 / 3 :=
by 
  sorry

end measure_of_angle_D_in_scalene_triangle_l153_153520


namespace not_cheap_is_necessary_condition_l153_153003

-- Define propositions for "good quality" and "not cheap"
variables {P: Prop} {Q: Prop} 

-- Statement "You get what you pay for" implies "good quality is not cheap"
axiom H : P → Q 

-- The proof problem
theorem not_cheap_is_necessary_condition (H : P → Q) : Q → P :=
by sorry

end not_cheap_is_necessary_condition_l153_153003


namespace sufficient_but_not_necessary_l153_153750

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 1) : (1 / a < 1) := 
by
  sorry

end sufficient_but_not_necessary_l153_153750


namespace dave_apps_left_l153_153602

theorem dave_apps_left (A : ℕ) 
  (h1 : 24 = A + 22) : A = 2 :=
by
  sorry

end dave_apps_left_l153_153602


namespace min_nS_n_eq_neg32_l153_153658

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d : ℤ) (a_1 : ℤ)

-- Conditions
axiom arithmetic_sequence_def : ∀ n : ℕ, a n = a_1 + (n - 1) * d
axiom sum_first_n_def : ∀ n : ℕ, S n = n * a_1 + (n * (n - 1) / 2) * d

axiom a5_eq_3 : a 5 = 3
axiom S10_eq_40 : S 10 = 40

theorem min_nS_n_eq_neg32 : ∃ n : ℕ, n * S n = -32 :=
sorry

end min_nS_n_eq_neg32_l153_153658


namespace MarlySoupBags_l153_153358

theorem MarlySoupBags :
  ∀ (milk chicken_stock vegetables bag_capacity total_soup total_bags : ℚ),
    milk = 6 ∧
    chicken_stock = 3 * milk ∧
    vegetables = 3 ∧
    bag_capacity = 2 ∧
    total_soup = milk + chicken_stock + vegetables ∧
    total_bags = total_soup / bag_capacity ∧
    total_bags.ceil = 14 :=
by
  intros
  sorry

end MarlySoupBags_l153_153358


namespace problem_statement_l153_153293

noncomputable def f1 (x : ℝ) := x + (1 / x)
noncomputable def f2 (x : ℝ) := 1 / (x ^ 2)
noncomputable def f3 (x : ℝ) := x ^ 3 - 2 * x
noncomputable def f4 (x : ℝ) := x ^ 2

theorem problem_statement : ∀ (x : ℝ), f2 (-x) = f2 x := by 
  sorry

end problem_statement_l153_153293


namespace range_of_omega_l153_153917

theorem range_of_omega (ω : ℝ) (hω : ω > 2/3) :
  (∀ x : ℝ, x = (k : ℤ) * π / ω + 3 * π / (4 * ω) → (x ≤ π ∨ x ≥ 2 * π) ) →
  ω ∈ Set.Icc (3/4 : ℝ) (7/8 : ℝ) :=
by
  sorry

end range_of_omega_l153_153917


namespace division_pairs_l153_153392

theorem division_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (ab^2 + b + 7) ∣ (a^2 * b + a + b) →
  (∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k) ∨ (a, b) = (11, 1) ∨ (a, b) = (49, 1) :=
sorry

end division_pairs_l153_153392


namespace range_of_a_l153_153824

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → x - Real.log x - a > 0) → a < 1 :=
sorry

end range_of_a_l153_153824


namespace range_of_t_l153_153246

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem range_of_t (t : ℝ) :  
  (∀ a > 0, ∀ x₀ y₀, 
    (a - a * Real.log x₀) / x₀^2 = 1 / 2 ∧ 
    y₀ = (a * Real.log x₀) / x₀ ∧ 
    x₀ = 2 * y₀ ∧ 
    a = Real.exp 1 ∧ 
    f (f x) = t -> t = 0) :=
by
  sorry

end range_of_t_l153_153246


namespace sum_A_B_l153_153864

noncomputable def num_four_digit_odd_numbers_divisible_by_3 : ℕ := 1500
noncomputable def num_four_digit_multiples_of_7 : ℕ := 1286

theorem sum_A_B (A B : ℕ) :
  A = num_four_digit_odd_numbers_divisible_by_3 →
  B = num_four_digit_multiples_of_7 →
  A + B = 2786 :=
by
  intros hA hB
  rw [hA, hB]
  exact rfl

end sum_A_B_l153_153864


namespace equation_solution_count_l153_153277

theorem equation_solution_count (n : ℕ) (h_pos : n > 0)
    (h_solutions : ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 28 ∧ ∀ (x y z : ℕ), (x, y, z) ∈ s → 2 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) :
    n = 17 ∨ n = 18 :=
sorry

end equation_solution_count_l153_153277


namespace sum_circumferences_of_small_circles_l153_153642

theorem sum_circumferences_of_small_circles (R : ℝ) (n : ℕ) (hR : R > 0) (hn : n > 0) :
  let original_circumference := 2 * Real.pi * R
  let part_length := original_circumference / n
  let small_circle_radius := part_length / Real.pi
  let small_circle_circumference := 2 * Real.pi * small_circle_radius
  let total_circumference := n * small_circle_circumference
  total_circumference = 2 * Real.pi ^ 2 * R :=
by {
  sorry
}

end sum_circumferences_of_small_circles_l153_153642


namespace total_count_not_47_l153_153604

theorem total_count_not_47 (h c : ℕ) : 11 * h + 6 * c ≠ 47 := by
  sorry

end total_count_not_47_l153_153604


namespace math_problem_solution_l153_153510

theorem math_problem_solution (x y : ℝ) : 
  abs x + x + 5 * y = 2 ∧ abs y - y + x = 7 → x + y + 2009 = 2012 :=
by {
  sorry
}

end math_problem_solution_l153_153510


namespace isosceles_triangle_perimeter_l153_153082

theorem isosceles_triangle_perimeter (a b : ℕ) (h_a : a = 8 ∨ a = 9) (h_b : b = 8 ∨ b = 9) 
(h_iso : a = a) (h_tri_ineq : a + a > b ∧ a + b > a ∧ b + a > a) :
  a + a + b = 25 ∨ a + a + b = 26 := 
by
  sorry

end isosceles_triangle_perimeter_l153_153082


namespace solve_system_l153_153949

theorem solve_system (x y z u : ℝ) :
  x^3 * y^2 * z = 2 ∧
  z^3 * u^2 * x = 32 ∧
  y^3 * z^2 * u = 8 ∧
  u^3 * x^2 * y = 8 →
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
  (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
  (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
  (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2) :=
sorry

end solve_system_l153_153949


namespace M_minus_N_l153_153135

theorem M_minus_N (a b c d : ℕ) (h1 : a + b = 20) (h2 : a + c = 24) (h3 : a + d = 22) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  let M := 2 * b + 26
  let N := 2 * 1 + 26
  (M - N) = 36 :=
by
  sorry

end M_minus_N_l153_153135


namespace fixed_monthly_fee_l153_153725

-- Define the problem parameters and assumptions
variables (x y : ℝ)
axiom february_bill : x + y = 20.72
axiom march_bill : x + 3 * y = 35.28

-- State the Lean theorem that we want to prove
theorem fixed_monthly_fee : x = 13.44 :=
by
  sorry

end fixed_monthly_fee_l153_153725


namespace brownies_maximum_l153_153457

theorem brownies_maximum (m n : ℕ) (h1 : (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)) :
  m * n ≤ 144 :=
sorry

end brownies_maximum_l153_153457


namespace fry_sausage_time_l153_153812

variable (time_per_sausage : ℕ)

noncomputable def time_for_sausages (sausages : ℕ) (tps : ℕ) : ℕ :=
  sausages * tps

noncomputable def time_for_eggs (eggs : ℕ) (minutes_per_egg : ℕ) : ℕ :=
  eggs * minutes_per_egg

noncomputable def total_time (time_sausages : ℕ) (time_eggs : ℕ) : ℕ :=
  time_sausages + time_eggs

theorem fry_sausage_time :
  let sausages := 3
  let eggs := 6
  let minutes_per_egg := 4
  let total_time_taken := 39
  total_time (time_for_sausages sausages time_per_sausage) (time_for_eggs eggs minutes_per_egg) = total_time_taken
  → time_per_sausage = 5 := by
  sorry

end fry_sausage_time_l153_153812


namespace tom_travel_time_to_virgo_island_l153_153375

-- Definitions based on conditions
def boat_trip_time : ℕ := 2
def plane_trip_time : ℕ := 4 * boat_trip_time
def total_trip_time : ℕ := plane_trip_time + boat_trip_time

-- Theorem we need to prove
theorem tom_travel_time_to_virgo_island : total_trip_time = 10 := by
  sorry

end tom_travel_time_to_virgo_island_l153_153375


namespace ratio_y_to_x_l153_153974

theorem ratio_y_to_x (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : y / x = 13 / 2 :=
by
  sorry

end ratio_y_to_x_l153_153974


namespace discriminant_eq_perfect_square_l153_153537

variables (a b c t : ℝ)

-- Conditions
axiom a_nonzero : a ≠ 0
axiom t_root : a * t^2 + b * t + c = 0

-- Goal
theorem discriminant_eq_perfect_square :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 :=
by
  -- Conditions and goal are stated, proof to be filled.
  sorry

end discriminant_eq_perfect_square_l153_153537


namespace paytons_score_l153_153929

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end paytons_score_l153_153929


namespace three_digit_solutions_exist_l153_153647

theorem three_digit_solutions_exist :
  ∃ (x y z : ℤ), 100 ≤ x ∧ x ≤ 999 ∧ 
                 100 ≤ y ∧ y ≤ 999 ∧
                 100 ≤ z ∧ z ≤ 999 ∧
                 17 * x + 15 * y - 28 * z = 61 ∧
                 19 * x - 25 * y + 12 * z = 31 :=
by
    sorry

end three_digit_solutions_exist_l153_153647


namespace intersection_complement_l153_153086

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 5, 6})
variable (hB : B = {1, 3, 4, 6, 7})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 5} :=
sorry

end intersection_complement_l153_153086


namespace number_of_appointments_l153_153860

-- Define the conditions
variables {hours_in_workday : ℕ} {appointments_duration : ℕ} {permit_rate : ℕ} {total_permits : ℕ}
variables (H1 : hours_in_workday = 8) (H2 : appointments_duration = 3) (H3 : permit_rate = 50) (H4: total_permits = 100)

-- Define the question as a theorem with the correct answer
theorem number_of_appointments : 
  (hours_in_workday - (total_permits / permit_rate)) / appointments_duration = 2 :=
by
  -- Proof is not required
  sorry

end number_of_appointments_l153_153860


namespace remainder_division_lemma_l153_153054

theorem remainder_division_lemma (j : ℕ) (hj : 0 < j) (hmod : 132 % (j^2) = 12) : 250 % j = 0 :=
sorry

end remainder_division_lemma_l153_153054


namespace quadratic_real_roots_l153_153012

theorem quadratic_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) : ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_real_roots_l153_153012


namespace quadrilateral_area_l153_153201

noncomputable def AB : ℝ := 3
noncomputable def BC : ℝ := 3
noncomputable def CD : ℝ := 4
noncomputable def DA : ℝ := 8
noncomputable def angle_DAB_add_angle_ABC : ℝ := 180

theorem quadrilateral_area :
  AB = 3 ∧ BC = 3 ∧ CD = 4 ∧ DA = 8 ∧ angle_DAB_add_angle_ABC = 180 →
  ∃ area : ℝ, area = 13.2 :=
by {
  sorry
}

end quadrilateral_area_l153_153201


namespace lowest_price_correct_l153_153889

noncomputable def lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components : ℕ) : ℕ :=
(cost_per_component + shipping_cost_per_unit) * number_of_components + fixed_costs

theorem lowest_price_correct :
  lowest_price 80 5 16500 150 / 150 = 195 :=
by
  sorry

end lowest_price_correct_l153_153889


namespace multiply_neg_reverse_inequality_l153_153023

theorem multiply_neg_reverse_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end multiply_neg_reverse_inequality_l153_153023


namespace panthers_score_points_l153_153485

theorem panthers_score_points (C P : ℕ) (h1 : C + P = 34) (h2 : C = P + 14) : P = 10 :=
by
  sorry

end panthers_score_points_l153_153485


namespace bunches_with_new_distribution_l153_153966

-- Given conditions
def bunches_initial := 8
def flowers_per_bunch_initial := 9
def total_flowers := bunches_initial * flowers_per_bunch_initial

-- New condition and proof requirement
def flowers_per_bunch_new := 12
def bunches_new := total_flowers / flowers_per_bunch_new

theorem bunches_with_new_distribution : bunches_new = 6 := by
  sorry

end bunches_with_new_distribution_l153_153966


namespace Iain_pennies_left_l153_153411

theorem Iain_pennies_left (initial_pennies older_pennies : ℕ) (percentage : ℝ)
  (h_initial : initial_pennies = 200)
  (h_older : older_pennies = 30)
  (h_percentage : percentage = 0.20) :
  initial_pennies - older_pennies - (percentage * (initial_pennies - older_pennies)) = 136 :=
by
  sorry

end Iain_pennies_left_l153_153411


namespace recreation_percentage_this_week_l153_153480

variable (W : ℝ) -- David's last week wages
variable (R_last_week : ℝ) -- Recreation spending last week
variable (W_this_week : ℝ) -- This week's wages
variable (R_this_week : ℝ) -- Recreation spending this week

-- Conditions
def wages_last_week : R_last_week = 0.4 * W := sorry
def wages_this_week : W_this_week = 0.95 * W := sorry
def recreation_spending_this_week : R_this_week = 1.1875 * R_last_week := sorry

-- Theorem to prove
theorem recreation_percentage_this_week :
  (R_this_week / W_this_week) = 0.5 := sorry

end recreation_percentage_this_week_l153_153480


namespace points_above_line_l153_153446

theorem points_above_line {t : ℝ} (hP : 1 + t - 1 > 0) (hQ : t^2 + (t - 1) - 1 > 0) : t > 1 :=
by
  sorry

end points_above_line_l153_153446


namespace correct_quadratic_equation_l153_153955

-- Definitions based on conditions
def root_sum (α β : ℝ) := α + β = 8
def root_product (α β : ℝ) := α * β = 24

-- Main statement to be proven
theorem correct_quadratic_equation (α β : ℝ) (h1 : root_sum 5 3) (h2 : root_product (-6) (-4)) :
    (α - 5) * (α - 3) = 0 ∧ (α + 6) * (α + 4) = 0 → α * α - 8 * α + 24 = 0 :=
sorry

end correct_quadratic_equation_l153_153955


namespace solution_set_lg2_l153_153838

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_deriv_lt : ∀ x : ℝ, deriv f x < 1

theorem solution_set_lg2 : { x : ℝ | f (Real.log x ^ 2) < Real.log x ^ 2 } = { x : ℝ | (1/10 : ℝ) < x ∧ x < 10 } :=
by
  sorry

end solution_set_lg2_l153_153838


namespace necessary_and_sufficient_condition_l153_153762

def U (a : ℕ) : Set ℕ := { x | x > 0 ∧ x ≤ a }
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}
def C_U (S : Set ℕ) (a : ℕ) : Set ℕ := U a ∩ Sᶜ

theorem necessary_and_sufficient_condition (a : ℕ) (h : 6 ≤ a ∧ a < 7) : 
  C_U P a = Q ↔ (6 ≤ a ∧ a < 7) :=
by
  sorry

end necessary_and_sufficient_condition_l153_153762


namespace probability_distribution_correct_l153_153064

noncomputable def numCombinations (n k : ℕ) : ℕ :=
  (Nat.choose n k)

theorem probability_distribution_correct :
  let totalCombinations := numCombinations 5 2
  let prob_two_red := (numCombinations 3 2 : ℚ) / totalCombinations
  let prob_two_white := (numCombinations 2 2 : ℚ) / totalCombinations
  let prob_one_red_one_white := ((numCombinations 3 1) * (numCombinations 2 1) : ℚ) / totalCombinations
  (prob_two_red, prob_one_red_one_white, prob_two_white) = (0.3, 0.6, 0.1) :=
by
  sorry

end probability_distribution_correct_l153_153064


namespace compare_neg_two_cubed_l153_153810

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end compare_neg_two_cubed_l153_153810


namespace friday_lending_tuesday_vs_thursday_total_lending_l153_153391

def standard_lending_rate : ℕ := 50
def monday_excess : ℤ := 0
def tuesday_excess : ℤ := 8
def wednesday_excess : ℤ := 6
def thursday_shortfall : ℤ := -3
def friday_shortfall : ℤ := -7

theorem friday_lending : (standard_lending_rate + friday_shortfall) = 43 := by
  sorry

theorem tuesday_vs_thursday : (tuesday_excess - thursday_shortfall) = 11 := by
  sorry

theorem total_lending : 
  (5 * standard_lending_rate + (monday_excess + tuesday_excess + wednesday_excess + thursday_shortfall + friday_shortfall)) = 254 := by
  sorry

end friday_lending_tuesday_vs_thursday_total_lending_l153_153391


namespace prime_divides_expression_l153_153156

theorem prime_divides_expression (p : ℕ) (hp : Nat.Prime p) : ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := 
by
  sorry

end prime_divides_expression_l153_153156


namespace candy_problem_l153_153116

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l153_153116


namespace no_15_students_with_unique_colors_l153_153970

-- Conditions as definitions
def num_students : Nat := 30
def num_colors : Nat := 15

-- The main statement
theorem no_15_students_with_unique_colors
  (students : Fin num_students → (Fin num_colors × Fin num_colors)) :
  ¬ ∃ (subset : Fin 15 → Fin num_students),
    ∀ i j (hi : i ≠ j), (students (subset i)).1 ≠ (students (subset j)).1 ∧
                         (students (subset i)).2 ≠ (students (subset j)).2 :=
by sorry

end no_15_students_with_unique_colors_l153_153970


namespace marks_age_more_than_thrice_aarons_l153_153395

theorem marks_age_more_than_thrice_aarons :
  ∃ (A : ℕ)(X : ℕ), 28 = A + 17 ∧ 25 = 3 * (A - 3) + X ∧ 32 = 2 * (A + 4) + 2 ∧ X = 1 :=
by
  sorry

end marks_age_more_than_thrice_aarons_l153_153395


namespace subset_A_B_l153_153079

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem subset_A_B : A ⊆ B := sorry

end subset_A_B_l153_153079


namespace spinner_sections_equal_size_l153_153760

theorem spinner_sections_equal_size 
  (p : ℕ → Prop)
  (h1 : ∀ n, p n ↔ (1 - (1: ℝ) / n) ^ 2 = 0.5625) : 
  p 4 :=
by
  sorry

end spinner_sections_equal_size_l153_153760


namespace neg_P_is_univ_l153_153124

noncomputable def P : Prop :=
  ∃ x0 : ℝ, x0^2 + 2 * x0 + 2 ≤ 0

theorem neg_P_is_univ :
  ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by {
  sorry
}

end neg_P_is_univ_l153_153124


namespace complete_square_solution_l153_153870

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end complete_square_solution_l153_153870


namespace gcd_factorial_8_6_squared_l153_153573

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l153_153573


namespace math_problem_l153_153558

theorem math_problem :
  18 * 35 + 45 * 18 - 18 * 10 = 1260 :=
by
  sorry

end math_problem_l153_153558


namespace remainder_2001_to_2005_mod_19_l153_153686

theorem remainder_2001_to_2005_mod_19 :
  (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 :=
by
  -- Use modular arithmetic properties to convert each factor
  have h2001 : 2001 % 19 = 6 := by sorry
  have h2002 : 2002 % 19 = 7 := by sorry
  have h2003 : 2003 % 19 = 8 := by sorry
  have h2004 : 2004 % 19 = 9 := by sorry
  have h2005 : 2005 % 19 = 10 := by sorry

  -- Compute the product modulo 19
  have h_prod : (6 * 7 * 8 * 9 * 10) % 19 = 11 := by sorry

  -- Combining these results
  have h_final : ((2001 * 2002 * 2003 * 2004 * 2005) % 19) = (6 * 7 * 8 * 9 * 10) % 19 := by sorry
  exact Eq.trans h_final h_prod

end remainder_2001_to_2005_mod_19_l153_153686


namespace dog_catches_rabbit_in_4_minutes_l153_153771

def dog_speed_mph : ℝ := 24
def rabbit_speed_mph : ℝ := 15
def rabbit_head_start : ℝ := 0.6

theorem dog_catches_rabbit_in_4_minutes : 
  (∃ t : ℝ, t > 0 ∧ 0.4 * t = 0.25 * t + 0.6) → ∃ t : ℝ, t = 4 :=
sorry

end dog_catches_rabbit_in_4_minutes_l153_153771


namespace least_subtraction_for_divisibility_l153_153630

/-- 
  Theorem: The least number that must be subtracted from 9857621 so that 
  the result is divisible by 17 is 8.
-/
theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, 9857621 % 17 = k ∧ k = 8 :=
by
  sorry

end least_subtraction_for_divisibility_l153_153630


namespace arithmetic_mean_of_q_and_r_l153_153827

theorem arithmetic_mean_of_q_and_r (p q r : ℝ) 
  (h₁: (p + q) / 2 = 10) 
  (h₂: r - p = 20) : 
  (q + r) / 2 = 20 :=
sorry

end arithmetic_mean_of_q_and_r_l153_153827


namespace vertices_of_equilateral_triangle_l153_153473

noncomputable def a : ℝ := 52 / 3
noncomputable def b : ℝ := -13 / 3 - 15 * Real.sqrt 3 / 2

theorem vertices_of_equilateral_triangle (a b : ℝ)
  (h₀ : (0, 0) = (0, 0))
  (h₁ : (a, 15) = (52 / 3, 15))
  (h₂ : (b, 41) = (-13 / 3 - 15 * Real.sqrt 3 / 2, 41)) :
  a * b = -676 / 9 := 
by
  sorry

end vertices_of_equilateral_triangle_l153_153473


namespace area_of_triangle_F1PF2_l153_153902

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 25) + (y^2 / 16) = 1

def is_focus (f : ℝ × ℝ) : Prop := 
  f = (3, 0) ∨ f = (-3, 0)

def right_angle_at_P (F1 P F2 : ℝ × ℝ) : Prop := 
  let a1 := (F1.1 - P.1, F1.2 - P.2)
  let a2 := (F2.1 - P.1, F2.2 - P.2)
  a1.1 * a2.1 + a1.2 * a2.2 = 0

theorem area_of_triangle_F1PF2
  (P F1 F2 : ℝ × ℝ)
  (hP : point_on_ellipse P)
  (hF1 : is_focus F1)
  (hF2 : is_focus F2)
  (h_angle : right_angle_at_P F1 P F2) :
  1/2 * (P.1 - F1.1) * (P.2 - F2.2) = 16 :=
sorry

end area_of_triangle_F1PF2_l153_153902


namespace john_pays_12_dollars_l153_153556

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end john_pays_12_dollars_l153_153556


namespace find_x2_plus_y2_l153_153085

theorem find_x2_plus_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 90) 
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 :=
sorry

end find_x2_plus_y2_l153_153085


namespace car_speed_in_kmph_l153_153489

def speed_mps : ℝ := 10  -- The speed of the car in meters per second
def conversion_factor : ℝ := 3.6  -- The conversion factor from m/s to km/h

theorem car_speed_in_kmph : speed_mps * conversion_factor = 36 := 
by
  sorry

end car_speed_in_kmph_l153_153489


namespace largest_triangle_perimeter_l153_153001

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem largest_triangle_perimeter : 
  ∃ (x : ℕ), x ≤ 14 ∧ 2 ≤ x ∧ is_valid_triangle 7 8 x ∧ (7 + 8 + x = 29) :=
sorry

end largest_triangle_perimeter_l153_153001


namespace mooncake_packaging_problem_l153_153066

theorem mooncake_packaging_problem :
  ∃ x y : ℕ, 9 * x + 4 * y = 35 ∧ x + y = 5 :=
by
  -- Proof is omitted
  sorry

end mooncake_packaging_problem_l153_153066


namespace converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l153_153905

theorem converse_of_P (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
by
  intro h
  exact sorry

theorem inverse_of_P (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

theorem contrapositive_of_P (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
by
  intro h
  exact sorry

theorem negation_of_P (a b : ℤ) : (a > b) → ¬ (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

end converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l153_153905


namespace k_plus_m_eq_27_l153_153216

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end k_plus_m_eq_27_l153_153216


namespace lion_cub_birth_rate_l153_153208

theorem lion_cub_birth_rate :
  ∀ (x : ℕ), 100 + 12 * (x - 1) = 148 → x = 5 :=
by
  intros x h
  sorry

end lion_cub_birth_rate_l153_153208


namespace area_of_rectangle_l153_153533

theorem area_of_rectangle (side_small_squares : ℝ) (side_smaller_square : ℝ) (side_larger_square : ℝ) 
  (h_small_squares : side_small_squares ^ 2 = 4) 
  (h_smaller_square : side_smaller_square ^ 2 = 1) 
  (h_larger_square : side_larger_square = 2 * side_smaller_square) :
  let horizontal_length := 2 * side_small_squares
  let vertical_length := side_small_squares + side_smaller_square
  let area := horizontal_length * vertical_length
  area = 12 
:= by 
  sorry

end area_of_rectangle_l153_153533


namespace has_exactly_one_solution_l153_153781

theorem has_exactly_one_solution (a : ℝ) : 
  (∀ x : ℝ, 5^(x^2 + 2 * a * x + a^2) = a * x^2 + 2 * a^2 * x + a^3 + a^2 - 6 * a + 6) ↔ (a = 1) :=
sorry

end has_exactly_one_solution_l153_153781


namespace direct_proportion_solution_l153_153431

theorem direct_proportion_solution (m : ℝ) (h1 : m + 3 ≠ 0) (h2 : m^2 - 8 = 1) : m = 3 :=
sorry

end direct_proportion_solution_l153_153431


namespace find_functions_l153_153025

noncomputable def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (p q r s : ℝ), p > 0 → q > 0 → r > 0 → s > 0 →
  (p * q = r * s) →
  (f p ^ 2 + f q ^ 2) / (f (r ^ 2) + f (s ^ 2)) = 
  (p ^ 2 + q ^ 2) / (r ^ 2 + s ^ 2)

theorem find_functions :
  ∀ (f : ℝ → ℝ),
  (satisfies_condition f) → 
  (∀ x : ℝ, x > 0 → f x = x ∨ f x = 1 / x) :=
by
  sorry

end find_functions_l153_153025


namespace log_three_div_square_l153_153396

theorem log_three_div_square (x y : ℝ) (h₁ : x ≠ 1) (h₂ : y ≠ 1) (h₃ : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h₄ : x * y = 243) :
  (Real.log (x / y) / Real.log 3) ^ 2 = 9 := 
sorry

end log_three_div_square_l153_153396


namespace math_problem_l153_153805

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x * y = -3)
  : x + (x^3 / y^2) + (y^3 / x^2) + y = 590.5 :=
sorry

end math_problem_l153_153805


namespace christmas_sale_pricing_l153_153027

theorem christmas_sale_pricing (a b : ℝ) : 
  (forall (c : ℝ), c = a * (3 / 5)) ∧ (forall (d : ℝ), d = b * (5 / 3)) :=
by
  sorry  -- proof goes here

end christmas_sale_pricing_l153_153027


namespace double_rooms_booked_l153_153711

theorem double_rooms_booked (S D : ℕ) 
(rooms_booked : S + D = 260) 
(single_room_cost : 35 * S + 60 * D = 14000) : 
D = 196 := 
sorry

end double_rooms_booked_l153_153711


namespace moon_speed_kmh_l153_153978

theorem moon_speed_kmh (speed_kms : ℝ) (h : speed_kms = 0.9) : speed_kms * 3600 = 3240 :=
by
  rw [h]
  norm_num

end moon_speed_kmh_l153_153978


namespace log10_sum_diff_l153_153237

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log10_sum_diff :
  log10 32 + log10 50 - log10 8 = 2.301 :=
by
  sorry

end log10_sum_diff_l153_153237


namespace minimum_rectangles_needed_l153_153900

/-- The theorem that defines the minimum number of rectangles needed to cover the specified figure -/
theorem minimum_rectangles_needed 
    (rectangles : ℕ) 
    (figure : Type)
    (covers : figure → Prop) :
  rectangles = 12 :=
sorry

end minimum_rectangles_needed_l153_153900


namespace balls_in_boxes_ways_l153_153147

theorem balls_in_boxes_ways : ∃ (ways : ℕ), ways = 56 :=
by
  let n := 5
  let m := 4
  let ways := 56
  sorry

end balls_in_boxes_ways_l153_153147


namespace distributor_cost_l153_153248

variable (C : ℝ) -- Cost of the item for the distributor
variable (P_observed : ℝ) -- Observed price
variable (commission_rate : ℝ) -- Commission rate
variable (profit_rate : ℝ) -- Desired profit rate

-- Conditions
def is_observed_price_correct (C : ℝ) (P_observed : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) : Prop :=
  let SP := C * (1 + profit_rate)
  let observed := SP * (1 - commission_rate)
  observed = P_observed

-- The proof goal
theorem distributor_cost (h : is_observed_price_correct C 30 0.20 0.20) : C = 31.25 := sorry

end distributor_cost_l153_153248


namespace domain_of_f_monotonicity_of_f_inequality_solution_l153_153806

open Real

noncomputable def f (x : ℝ) : ℝ := log ((1 - x) / (1 + x))

theorem domain_of_f :
  ∀ x, -1 < x ∧ x < 1 → ∃ y, y = f x :=
by
  intro x h
  use log ((1 - x) / (1 + x))
  simp [f]

theorem monotonicity_of_f :
  ∀ x y, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → x < y → f x > f y :=
sorry

theorem inequality_solution :
  ∀ x, f (2 * x - 1) < 0 ↔ (1 / 2 < x ∧ x < 1) :=
sorry

end domain_of_f_monotonicity_of_f_inequality_solution_l153_153806


namespace expected_teachers_with_masters_degree_l153_153508

theorem expected_teachers_with_masters_degree
  (prob: ℚ) (teachers: ℕ) (h_prob: prob = 1/4) (h_teachers: teachers = 320) :
  prob * teachers = 80 :=
by
  sorry

end expected_teachers_with_masters_degree_l153_153508


namespace find_value_of_k_l153_153940

noncomputable def line_parallel_and_point_condition (k : ℝ) :=
  ∃ (m : ℝ), m = -5/4 ∧ (22 - (-8)) / (k - 3) = m

theorem find_value_of_k : ∃ k : ℝ, line_parallel_and_point_condition k ∧ k = -21 :=
by
  sorry

end find_value_of_k_l153_153940


namespace running_laps_l153_153352

theorem running_laps (A B : ℕ)
  (h_ratio : ∀ t : ℕ, (A * t) = 5 * (B * t) / 3)
  (h_start : A = 5 ∧ B = 3 ∧ ∀ t : ℕ, (A * t) - (B * t) = 4) :
  (B * 2 = 6) ∧ (A * 2 = 10) :=
by
  sorry

end running_laps_l153_153352


namespace tan_105_eq_neg2_sub_sqrt3_l153_153074

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l153_153074


namespace possible_values_2a_b_l153_153179

theorem possible_values_2a_b (a b x y z : ℕ) (h1: a^x = 1994^z) (h2: b^y = 1994^z) (h3: 1/x + 1/y = 1/z) : 
  (2 * a + b = 1001) ∨ (2 * a + b = 1996) :=
by
  sorry

end possible_values_2a_b_l153_153179


namespace trigonometric_identity_l153_153261

theorem trigonometric_identity :
  (Real.sqrt 3 / Real.cos (10 * Real.pi / 180) - 1 / Real.sin (170 * Real.pi / 180) = -4) :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_l153_153261


namespace problem_statement_l153_153836

-- Define that f is an even function and decreasing on (0, +∞)
variables {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def is_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- Main statement: Prove the specific inequality under the given conditions
theorem problem_statement (f_even : is_even_function f) (f_decreasing : is_decreasing_on_pos f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) :=
by
  sorry

end problem_statement_l153_153836


namespace product_of_nine_integers_16_to_30_equals_15_factorial_l153_153912

noncomputable def factorial (n : Nat) : Nat :=
match n with
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem product_of_nine_integers_16_to_30_equals_15_factorial :
  (16 * 18 * 20 * 21 * 22 * 25 * 26 * 27 * 28) = factorial 15 := 
by sorry

end product_of_nine_integers_16_to_30_equals_15_factorial_l153_153912


namespace kim_distance_traveled_l153_153922

-- Definitions based on the problem conditions:
def infantry_column_length : ℝ := 1  -- The length of the infantry column in km.
def distance_inf_covered : ℝ := 2.4  -- Distance the infantrymen covered in km.

-- Theorem statement:
theorem kim_distance_traveled (column_length : ℝ) (inf_covered : ℝ) :
  column_length = 1 →
  inf_covered = 2.4 →
  ∃ d : ℝ, d = 3.6 :=
by
  sorry

end kim_distance_traveled_l153_153922


namespace part_I_part_II_part_III_l153_153615

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) - (1 / (2^x + 1))

theorem part_I :
  ∃ a : ℝ, ∀ x : ℝ, f x = a - (1 / (2^x + 1)) → a = (1 / 2) :=
by sorry

theorem part_II :
  ∀ y : ℝ, y = f x → (-1 / 2) < y ∧ y < (1 / 2) :=
by sorry

theorem part_III :
  ∀ m n : ℝ, m + n ≠ 0 → (f m + f n) / (m^3 + n^3) > f 0 :=
by sorry

end part_I_part_II_part_III_l153_153615


namespace sum_of_fractions_l153_153245

-- Definitions of parameters and conditions
variables {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

-- The statement of the proof problem
theorem sum_of_fractions (hx : x ≠ 0) (hy : y ≠ 0) : 
  (3 / x) + (2 / y) = (3 * y + 2 * x) / (x * y) :=
sorry

end sum_of_fractions_l153_153245


namespace number_of_floors_l153_153709

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end number_of_floors_l153_153709


namespace find_inradius_l153_153874

-- Define variables and constants
variables (P A : ℝ)
variables (s r : ℝ)

-- Given conditions as definitions
def perimeter_triangle : Prop := P = 36
def area_triangle : Prop := A = 45

-- Semi-perimeter definition
def semi_perimeter : Prop := s = P / 2

-- Inradius and area relationship
def inradius_area_relation : Prop := A = r * s

-- Theorem statement
theorem find_inradius (hP : perimeter_triangle P) (hA : area_triangle A) (hs : semi_perimeter P s) (har : inradius_area_relation A r s) :
  r = 2.5 :=
by
  sorry

end find_inradius_l153_153874


namespace x_y_sum_vals_l153_153178

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end x_y_sum_vals_l153_153178


namespace union_of_M_and_N_l153_153345

open Set

theorem union_of_M_and_N :
  let M := {x : ℝ | x^2 - 4 * x < 0}
  let N := {x : ℝ | |x| ≤ 2}
  M ∪ N = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end union_of_M_and_N_l153_153345


namespace coffee_shop_cups_l153_153927

variables (A B X Y : ℕ) (Z : ℕ)

theorem coffee_shop_cups (h1 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := 
by
  sorry

end coffee_shop_cups_l153_153927


namespace parallel_lines_slope_l153_153099

theorem parallel_lines_slope (a : ℝ) :
  (∀ (x y : ℝ), x + a * y + 6 = 0 ∧ (a - 2) * x + 3 * y + 2 * a = 0 → (1 / (a - 2) = a / 3)) →
  a = -1 :=
by {
  sorry
}

end parallel_lines_slope_l153_153099


namespace length_of_CD_l153_153301

theorem length_of_CD (x y : ℝ) (h1 : x = (1/5) * (4 + y))
  (h2 : (x + 4) / y = 2 / 3) (h3 : 4 = 4) : x + y + 4 = 17.143 :=
sorry

end length_of_CD_l153_153301


namespace moles_H2O_formed_l153_153749

-- Define the balanced equation as a struct
structure Reaction :=
(reactants : List (String × ℕ)) -- List of reactants with their stoichiometric coefficients
(products : List (String × ℕ)) -- List of products with their stoichiometric coefficients

-- Example reaction: NaHCO3 + HC2H3O2 -> NaC2H3O2 + H2O + CO2
def example_reaction : Reaction :=
{ reactants := [("NaHCO3", 1), ("HC2H3O2", 1)],
  products := [("NaC2H3O2", 1), ("H2O", 1), ("CO2", 1)] }

-- We need a predicate to determine the number of moles of a product based on the reaction
def moles_of_product (reaction : Reaction) (product : String) (moles_reactant₁ moles_reactant₂ : ℕ) : ℕ :=
if product = "H2O" then moles_reactant₁ else 0  -- Only considering H2O for simplicity

-- Now we define our main theorem
theorem moles_H2O_formed : 
  moles_of_product example_reaction "H2O" 3 3 = 3 :=
by
  -- The proof will go here; for now, we use sorry to skip it
  sorry

end moles_H2O_formed_l153_153749


namespace cube_volume_given_surface_area_l153_153911

theorem cube_volume_given_surface_area (A : ℝ) (V : ℝ) :
  A = 96 → V = 64 :=
by
  sorry

end cube_volume_given_surface_area_l153_153911


namespace evaluate_expression_x_eq_3_l153_153324

theorem evaluate_expression_x_eq_3 : (3^5 - 5 * 3 + 7 * 3^3) = 417 := by
  sorry

end evaluate_expression_x_eq_3_l153_153324


namespace tub_drain_time_l153_153061

theorem tub_drain_time (t : ℝ) (p q : ℝ) (h1 : t = 4) (h2 : p = 5 / 7) (h3 : q = 2 / 7) :
  q * t / p = 1.6 := by
  sorry

end tub_drain_time_l153_153061


namespace Kims_final_score_l153_153522

def easy_points : ℕ := 2
def average_points : ℕ := 3
def hard_points : ℕ := 5
def expert_points : ℕ := 7

def easy_correct : ℕ := 6
def average_correct : ℕ := 2
def hard_correct : ℕ := 4
def expert_correct : ℕ := 3

def complex_problems_bonus : ℕ := 1
def complex_problems_solved : ℕ := 2

def penalty_per_incorrect : ℕ := 1
def easy_incorrect : ℕ := 1
def average_incorrect : ℕ := 2
def hard_incorrect : ℕ := 2
def expert_incorrect : ℕ := 3

theorem Kims_final_score : 
  (easy_correct * easy_points + 
   average_correct * average_points + 
   hard_correct * hard_points + 
   expert_correct * expert_points + 
   complex_problems_solved * complex_problems_bonus) - 
   (easy_incorrect * penalty_per_incorrect + 
    average_incorrect * penalty_per_incorrect + 
    hard_incorrect * penalty_per_incorrect + 
    expert_incorrect * penalty_per_incorrect) = 53 :=
by 
  sorry

end Kims_final_score_l153_153522


namespace intersection_A_B_union_A_complement_B_subset_C_B_range_l153_153782

def set_A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 9 }
def set_C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 2 < x ∧ x < 6 } :=
sorry

theorem union_A_complement_B :
  set_A ∪ (compl set_B) = { x | x < 6 } ∪ { x | x ≥ 9 } :=
sorry

theorem subset_C_B_range (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end intersection_A_B_union_A_complement_B_subset_C_B_range_l153_153782


namespace wood_not_heavier_than_brick_l153_153482

-- Define the weights of the wood and the brick
def block_weight_kg : ℝ := 8
def brick_weight_g : ℝ := 8000

-- Conversion function from kg to g
def kg_to_g (kg : ℝ) : ℝ := kg * 1000

-- State the proof problem
theorem wood_not_heavier_than_brick : ¬ (kg_to_g block_weight_kg > brick_weight_g) :=
by
  -- Begin the proof
  sorry

end wood_not_heavier_than_brick_l153_153482


namespace total_students_class_l153_153843

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end total_students_class_l153_153843


namespace circuit_boards_fail_inspection_l153_153857

theorem circuit_boards_fail_inspection (P F : ℝ) (h1 : P + F = 3200)
    (h2 : (1 / 8) * P + F = 456) : F = 64 :=
by
  sorry

end circuit_boards_fail_inspection_l153_153857


namespace intersecting_rectangles_shaded_area_l153_153890

theorem intersecting_rectangles_shaded_area 
  (a_w : ℕ) (a_l : ℕ) (b_w : ℕ) (b_l : ℕ) (c_w : ℕ) (c_l : ℕ)
  (overlap_ab_w : ℕ) (overlap_ab_h : ℕ)
  (overlap_ac_w : ℕ) (overlap_ac_h : ℕ)
  (overlap_bc_w : ℕ) (overlap_bc_h : ℕ)
  (triple_overlap_w : ℕ) (triple_overlap_h : ℕ) :
  a_w = 4 → a_l = 12 →
  b_w = 5 → b_l = 10 →
  c_w = 3 → c_l = 6 →
  overlap_ab_w = 4 → overlap_ab_h = 5 →
  overlap_ac_w = 3 → overlap_ac_h = 4 →
  overlap_bc_w = 3 → overlap_bc_h = 3 →
  triple_overlap_w = 3 → triple_overlap_h = 3 →
  ((a_w * a_l) + (b_w * b_l) + (c_w * c_l)) - 
  ((overlap_ab_w * overlap_ab_h) + (overlap_ac_w * overlap_ac_h) + (overlap_bc_w * overlap_bc_h)) + 
  (triple_overlap_w * triple_overlap_h) = 84 :=
by 
  sorry

end intersecting_rectangles_shaded_area_l153_153890


namespace machines_complete_order_l153_153662

theorem machines_complete_order (h1 : ℝ) (h2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ)
  (h1_def : h1 = 9)
  (h2_def : h2 = 8)
  (rate1_def : rate1 = 1 / h1)
  (rate2_def : rate2 = 1 / h2)
  (combined_rate : ℝ := rate1 + rate2) :
  time = 72 / 17 :=
by
  sorry

end machines_complete_order_l153_153662


namespace angle_sum_around_point_l153_153289

theorem angle_sum_around_point (y : ℕ) (h1 : 210 + 3 * y = 360) : y = 50 := 
by 
  sorry

end angle_sum_around_point_l153_153289


namespace dandelion_average_l153_153469

theorem dandelion_average :
  let Billy_initial := 36
  let George_initial := Billy_initial / 3
  let Billy_total := Billy_initial + 10
  let George_total := George_initial + 10
  let total := Billy_total + George_total
  let average := total / 2
  average = 34 :=
by
  -- placeholder for the proof
  sorry

end dandelion_average_l153_153469


namespace cos_double_angle_l153_153407

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 5) = Real.sqrt 3 / 3) :
  Real.cos (2 * α + 2 * Real.pi / 5) = 1 / 3 :=
by
  sorry

end cos_double_angle_l153_153407


namespace solve_problem_l153_153660

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem solve_problem : spadesuit 3 (spadesuit 5 (spadesuit 8 11)) = 1 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l153_153660


namespace initial_balloons_correct_l153_153791

-- Define the variables corresponding to the conditions given in the problem
def boy_balloon_count := 3
def girl_balloon_count := 12
def balloons_sold := boy_balloon_count + girl_balloon_count
def balloons_remaining := 21

-- State the theorem asserting the initial number of balloons
theorem initial_balloons_correct :
  balloons_sold + balloons_remaining = 36 := sorry

end initial_balloons_correct_l153_153791


namespace journey_total_distance_l153_153129

theorem journey_total_distance (D : ℝ) 
  (h1 : (D / 3) / 21 + (D / 3) / 14 + (D / 3) / 6 = 12) : 
  D = 126 :=
sorry

end journey_total_distance_l153_153129


namespace savings_value_l153_153322

def total_cost_individual (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let cost (n : ℕ) : ℝ := 
    let paid_windows := n - (n / 6) -- one free window per five
    cost_per_window * paid_windows
  let discount (amount : ℝ) : ℝ :=
    if s > 10 then 0.95 * amount else amount
  discount (cost g) + discount (cost s)

def total_cost_joint (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let n := g + s
  let paid_windows := n - (n / 6) -- one free window per five
  let joint_cost := cost_per_window * paid_windows
  if n > 10 then 0.95 * joint_cost else joint_cost

def savings (g : ℕ) (s : ℕ) : ℝ :=
  total_cost_individual g s - total_cost_joint g s

theorem savings_value (g s : ℕ) (hg : g = 9) (hs : s = 13) : savings g s = 162 := 
by 
  simp [savings, total_cost_individual, total_cost_joint, hg, hs]
  -- Detailed calculation is omitted, since it's not required according to the instructions.
  sorry

end savings_value_l153_153322


namespace find_y_l153_153614

theorem find_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end find_y_l153_153614


namespace average_weight_of_all_girls_l153_153623

theorem average_weight_of_all_girls 
    (avg_weight_group1 : ℝ) (avg_weight_group2 : ℝ) 
    (num_girls_group1 : ℕ) (num_girls_group2 : ℕ) 
    (h1 : avg_weight_group1 = 50.25) 
    (h2 : avg_weight_group2 = 45.15) 
    (h3 : num_girls_group1 = 16) 
    (h4 : num_girls_group2 = 8) : 
    (avg_weight_group1 * num_girls_group1 + avg_weight_group2 * num_girls_group2) / (num_girls_group1 + num_girls_group2) = 48.55 := 
by 
    sorry

end average_weight_of_all_girls_l153_153623


namespace no_real_solutions_l153_153285

open Real

theorem no_real_solutions :
  ¬(∃ x : ℝ, (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 = 0) := by
  sorry

end no_real_solutions_l153_153285


namespace packets_in_box_l153_153093

theorem packets_in_box 
  (coffees_per_day : ℕ) 
  (packets_per_coffee : ℕ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (days : ℕ) 
  (P : ℕ) 
  (h_coffees_per_day : coffees_per_day = 2)
  (h_packets_per_coffee : packets_per_coffee = 1)
  (h_cost_per_box : cost_per_box = 4)
  (h_total_cost : total_cost = 24)
  (h_days : days = 90)
  : P = 30 := 
by
  sorry

end packets_in_box_l153_153093


namespace max_AMC_AM_MC_CA_l153_153759

theorem max_AMC_AM_MC_CA (A M C : ℕ) (h_sum : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_AMC_AM_MC_CA_l153_153759


namespace gabriel_month_days_l153_153325

theorem gabriel_month_days (forgot_days took_days : ℕ) (h_forgot : forgot_days = 3) (h_took : took_days = 28) : 
  forgot_days + took_days = 31 :=
by
  sorry

end gabriel_month_days_l153_153325


namespace exists_positive_integers_abc_l153_153842

theorem exists_positive_integers_abc (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_m_gt_one : 1 < m) (h_n_gt_one : 1 < n) :
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ m^a = 1 + n^b * c ∧ Nat.gcd c n = 1 :=
by
  sorry

end exists_positive_integers_abc_l153_153842


namespace distinct_integers_sum_of_three_elems_l153_153338

-- Define the set S and the property of its elements
def S : Set ℕ := {1, 4, 7, 10, 13, 16, 19}

-- Define the property that each element in S is of the form 3k + 1
def is_form_3k_plus_1 (x : ℕ) : Prop := ∃ k : ℤ, x = 3 * k + 1

theorem distinct_integers_sum_of_three_elems (h₁ : ∀ x ∈ S, is_form_3k_plus_1 x) :
  (∃! n, n = 13) :=
by
  sorry

end distinct_integers_sum_of_three_elems_l153_153338


namespace number_of_non_degenerate_rectangles_excluding_center_l153_153014

/-!
# Problem Statement
We want to find the number of non-degenerate rectangles in a 7x7 grid that do not fully cover the center point (4, 4).
-/

def num_rectangles_excluding_center : Nat :=
  let total_rectangles := (Nat.choose 7 2) * (Nat.choose 7 2)
  let rectangles_including_center := 4 * ((3 * 3 * 3) + (3 * 3))
  total_rectangles - rectangles_including_center

theorem number_of_non_degenerate_rectangles_excluding_center :
  num_rectangles_excluding_center = 297 :=
by
  sorry -- proof goes here

end number_of_non_degenerate_rectangles_excluding_center_l153_153014


namespace catering_budget_total_l153_153702

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end catering_budget_total_l153_153702


namespace solveNumberOfWaysToChooseSeats_l153_153633

/--
Define the problem of professors choosing their seats among 9 chairs with specific constraints.
-/
noncomputable def numberOfWaysToChooseSeats : ℕ :=
  let totalChairs := 9
  let endChairChoices := 2 * (7 * (7 - 2))  -- (2 end chairs, 7 for 2nd prof, 5 for 3rd prof)
  let middleChairChoices := 7 * (6 * (6 - 2))  -- (7 non-end chairs, 6 for 2nd prof, 4 for 3rd prof)
  endChairChoices + middleChairChoices

/--
The final result should be 238
-/
theorem solveNumberOfWaysToChooseSeats : numberOfWaysToChooseSeats = 238 := by
  sorry

end solveNumberOfWaysToChooseSeats_l153_153633


namespace intersection_A_B_l153_153006

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_A_B :
  A ∩ B = {-3, 1} := by
  sorry

end intersection_A_B_l153_153006


namespace three_digit_number_constraint_l153_153936

theorem three_digit_number_constraint (B : ℕ) (h1 : 30 ≤ B ∧ B < 40) (h2 : (330 + B) % 3 = 0) (h3 : (330 + B) % 7 = 0) : B = 6 :=
sorry

end three_digit_number_constraint_l153_153936


namespace modulus_remainder_l153_153264

namespace Proof

def a (n : ℕ) : ℕ := 88134 + n

theorem modulus_remainder :
  (2 * ((a 0)^2 + (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2)) % 11 = 3 := by
  sorry

end Proof

end modulus_remainder_l153_153264


namespace find_b_eq_five_l153_153814

/--
Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
and the condition that the distances from O (the origin) to B and from B to A are equal,
prove that b = 5.
-/
theorem find_b_eq_five : ∃ b : ℝ, (dist (0, 0) (0, b) = dist (0, b) (4, 2)) ∧ b = 5 :=
by
  sorry

end find_b_eq_five_l153_153814


namespace clinton_earnings_correct_l153_153786

-- Define the conditions as variables/constants
def num_students_Arlington : ℕ := 8
def days_Arlington : ℕ := 4

def num_students_Bradford : ℕ := 6
def days_Bradford : ℕ := 7

def num_students_Clinton : ℕ := 7
def days_Clinton : ℕ := 8

def total_compensation : ℝ := 1456

noncomputable def total_student_days : ℕ :=
  num_students_Arlington * days_Arlington + num_students_Bradford * days_Bradford + num_students_Clinton * days_Clinton

noncomputable def daily_wage : ℝ :=
  total_compensation / total_student_days

noncomputable def earnings_Clinton : ℝ :=
  daily_wage * (num_students_Clinton * days_Clinton)

theorem clinton_earnings_correct : earnings_Clinton = 627.2 := by 
  sorry

end clinton_earnings_correct_l153_153786


namespace probability_composite_is_correct_l153_153983

noncomputable def probability_composite : ℚ :=
  1 - (25 / (8^6))

theorem probability_composite_is_correct :
  probability_composite = 262119 / 262144 :=
by
  sorry

end probability_composite_is_correct_l153_153983


namespace second_quadrant_point_l153_153399

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end second_quadrant_point_l153_153399


namespace part1_part2_part3_l153_153951

-- Part 1
theorem part1 :
  ∀ x : ℝ, (4 * x - 3 = 1) → (x = 1) ↔ 
    (¬(x - 3 > 3 * x - 1) ∧ (4 * (x - 1) ≤ 2) ∧ (x + 2 > 0 ∧ 3 * x - 3 ≤ 1)) :=
by sorry

-- Part 2
theorem part2 :
  ∀ (m n q : ℝ), (m + 2 * n = 6) → (2 * m + n = 3 * q) → (m + n > 1) → q > -1 :=
by sorry

-- Part 3
theorem part3 :
  ∀ (k m n : ℝ), (k < 3) → (∃ x : ℝ, (3 * (x - 1) = k) ∧ (4 * x + n < x + 2 * m)) → 
    (m + n ≥ 0) → (∃! n : ℝ, ∀ x : ℝ, (2 ≤ m ∧ m < 5 / 2)) :=
by sorry

end part1_part2_part3_l153_153951


namespace quadratic_has_distinct_real_roots_l153_153534

theorem quadratic_has_distinct_real_roots {k : ℝ} (hk : k < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ + k = 0) ∧ (x₂^2 - x₂ + k = 0) :=
by
  -- Proof goes here.
  sorry

end quadratic_has_distinct_real_roots_l153_153534


namespace total_bees_in_hive_at_end_of_7_days_l153_153197

-- Definitions of given conditions
def daily_hatch : Nat := 3000
def daily_loss : Nat := 900
def initial_bees : Nat := 12500
def days : Nat := 7
def queen_count : Nat := 1

-- Statement to prove
theorem total_bees_in_hive_at_end_of_7_days :
  initial_bees + daily_hatch * days - daily_loss * days + queen_count = 27201 := by
  sorry

end total_bees_in_hive_at_end_of_7_days_l153_153197


namespace avg_of_first_5_multiples_of_5_l153_153174

theorem avg_of_first_5_multiples_of_5 : (5 + 10 + 15 + 20 + 25) / 5 = 15 := 
by {
  sorry
}

end avg_of_first_5_multiples_of_5_l153_153174


namespace find_number_l153_153899

theorem find_number (x : ℝ) (h : (5 / 6) * x = (5 / 16) * x + 300) : x = 576 :=
sorry

end find_number_l153_153899


namespace distinct_points_count_l153_153609

theorem distinct_points_count :
  ∃ (P : Finset (ℝ × ℝ)), 
    (∀ p ∈ P, p.1^2 + p.2^2 = 1 ∧ p.1^2 + 9 * p.2^2 = 9) ∧ P.card = 2 :=
by
  sorry

end distinct_points_count_l153_153609


namespace cyclist_speed_ratio_l153_153801

variables (k r t v1 v2 : ℝ)
variable (h1 : v1 = 2 * v2) -- Condition 5

-- When traveling in the same direction, relative speed is v1 - v2 and they cover 2k miles in 3r hours
variable (h2 : 2 * k = (v1 - v2) * 3 * r)

-- When traveling in opposite directions, relative speed is v1 + v2 and they pass each other in 2t hours
variable (h3 : 2 * k = (v1 + v2) * 2 * t)

theorem cyclist_speed_ratio (h1 : v1 = 2 * v2) (h2 : 2 * k = (v1 - v2) * 3 * r) (h3 : 2 * k = (v1 + v2) * 2 * t) :
  v1 / v2 = 2 :=
sorry

end cyclist_speed_ratio_l153_153801


namespace baker_cakes_l153_153205

theorem baker_cakes : (62.5 + 149.25 - 144.75 = 67) :=
by
  sorry

end baker_cakes_l153_153205


namespace slower_train_speed_l153_153975

noncomputable def speed_of_slower_train (v_f : ℕ) (l1 l2 : ℚ) (t : ℚ) : ℚ :=
  let total_distance := l1 + l2
  let time_in_hours := t / 3600
  let relative_speed := total_distance / time_in_hours
  relative_speed - v_f

theorem slower_train_speed :
  speed_of_slower_train 210 (11 / 10) (9 / 10) 24 = 90 := by
  sorry

end slower_train_speed_l153_153975


namespace no_solutions_for_a_gt_1_l153_153369

theorem no_solutions_for_a_gt_1 (a b : ℝ) (h_a_gt_1 : 1 < a) :
  ¬∃ x : ℝ, a^(2-2*x^2) + (b+4) * a^(1-x^2) + 3*b + 4 = 0 ↔ 0 < b ∧ b < 4 :=
by
  sorry

end no_solutions_for_a_gt_1_l153_153369


namespace tails_and_die_1_or_2_l153_153224

noncomputable def fairCoinFlipProbability : ℚ := 1 / 2
noncomputable def fairDieRollProbability : ℚ := 1 / 6
noncomputable def combinedProbability : ℚ := fairCoinFlipProbability * (fairDieRollProbability + fairDieRollProbability)

theorem tails_and_die_1_or_2 :
  combinedProbability = 1 / 6 :=
by
  sorry

end tails_and_die_1_or_2_l153_153224


namespace mean_of_set_l153_153355

theorem mean_of_set (n : ℤ) (h_median : n + 7 = 14) : (n + (n + 4) + (n + 7) + (n + 10) + (n + 14)) / 5 = 14 := by
  sorry

end mean_of_set_l153_153355


namespace largest_n_divides_1005_fact_l153_153585

theorem largest_n_divides_1005_fact (n : ℕ) : (∃ n, 10^n ∣ (Nat.factorial 1005)) ↔ n = 250 :=
by
  sorry

end largest_n_divides_1005_fact_l153_153585


namespace fish_pond_estimate_l153_153311

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

end fish_pond_estimate_l153_153311


namespace train_crossing_time_l153_153362

def train_length : ℕ := 320
def train_speed_kmh : ℕ := 72
def kmh_to_ms (v : ℕ) : ℕ := v * 1000 / 3600
def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh
def crossing_time (length : ℕ) (speed : ℕ) : ℕ := length / speed

theorem train_crossing_time : crossing_time train_length train_speed_ms = 16 := 
by {
  sorry
}

end train_crossing_time_l153_153362


namespace common_measure_of_segments_l153_153243

theorem common_measure_of_segments (a b : ℚ) (h₁ : a = 4 / 15) (h₂ : b = 8 / 21) : 
  (∃ (c : ℚ), c = 1 / 105 ∧ ∃ (n₁ n₂ : ℕ), a = n₁ * c ∧ b = n₂ * c) := 
by {
  sorry
}

end common_measure_of_segments_l153_153243


namespace probability_not_yellow_l153_153831

-- Define the conditions
def red_jelly_beans : Nat := 4
def green_jelly_beans : Nat := 7
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Definitions used in the proof problem
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_yellow_jelly_beans : Nat := total_jelly_beans - yellow_jelly_beans

-- Lean statement of the probability problem
theorem probability_not_yellow : 
  (non_yellow_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = 7 / 10 := 
by 
  sorry

end probability_not_yellow_l153_153831


namespace problem_l153_153164

theorem problem
  (x y : ℝ)
  (h₁ : x - 2 * y = -5)
  (h₂ : x * y = -2) :
  2 * x^2 * y - 4 * x * y^2 = 20 := 
by
  sorry

end problem_l153_153164


namespace percentage_cities_in_range_l153_153422

-- Definitions of percentages as given conditions
def percentage_cities_between_50k_200k : ℕ := 40
def percentage_cities_below_50k : ℕ := 35
def percentage_cities_above_200k : ℕ := 25

-- Statement of the problem
theorem percentage_cities_in_range :
  percentage_cities_between_50k_200k = 40 := 
by
  sorry

end percentage_cities_in_range_l153_153422


namespace man_l153_153960

noncomputable def man_saves (S : ℝ) : ℝ :=
0.20 * S

noncomputable def initial_expenses (S : ℝ) : ℝ :=
0.80 * S

noncomputable def new_expenses (S : ℝ) : ℝ :=
1.10 * (0.80 * S)

noncomputable def said_savings (S : ℝ) : ℝ :=
S - new_expenses S

theorem man's_monthly_salary (S : ℝ) (h : said_savings S = 500) : S = 4166.67 :=
by
  sorry

end man_l153_153960


namespace y_less_than_z_by_40_percent_l153_153914

variable {x y z : ℝ}

theorem y_less_than_z_by_40_percent (h1 : x = 1.3 * y) (h2 : x = 0.78 * z) : y = 0.6 * z :=
by
  -- The proof will be provided here
  -- We are demonstrating that y = 0.6 * z is a consequence of h1 and h2
  sorry

end y_less_than_z_by_40_percent_l153_153914


namespace max_students_distribution_l153_153669

theorem max_students_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  Nat.gcd pens toys = 41 :=
by
  sorry

end max_students_distribution_l153_153669


namespace sum_of_roots_l153_153184

-- Defined the equation x^2 - 7x + 2 - 16 = 0 as x^2 - 7x - 14 = 0
def equation (x : ℝ) := x^2 - 7 * x - 14 = 0 

-- State the theorem leveraging the above condition
theorem sum_of_roots : 
  (∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 ≠ x2) →
  (∃ sum : ℝ, sum = 7) := by
  sorry

end sum_of_roots_l153_153184


namespace probability_A_to_B_in_8_moves_l153_153693

-- Define vertices
inductive Vertex : Type
| A | B | C | D | E | F

open Vertex

-- Define the probability of ending up at Vertex B after 8 moves starting from Vertex A
noncomputable def probability_at_B_after_8_moves : ℚ :=
  let prob := (3 : ℚ) / 16
  prob

-- Theorem statement
theorem probability_A_to_B_in_8_moves :
  (probability_at_B_after_8_moves = (3 : ℚ) / 16) :=
by
  -- Proof to be provided
  sorry

end probability_A_to_B_in_8_moves_l153_153693


namespace length_third_altitude_l153_153741

theorem length_third_altitude (a b c : ℝ) (S : ℝ) 
  (h_altitude_a : 4 = 2 * S / a)
  (h_altitude_b : 12 = 2 * S / b)
  (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_third_integer : ∃ n : ℕ, h = n):
  h = 5 :=
by
  -- Proof is omitted
  sorry

end length_third_altitude_l153_153741


namespace john_spent_fraction_on_snacks_l153_153053

theorem john_spent_fraction_on_snacks (x : ℚ) :
  (∀ (x : ℚ), (1 - x) * 20 - (3 / 4) * (1 - x) * 20 = 4) → (x = 1 / 5) :=
by sorry

end john_spent_fraction_on_snacks_l153_153053


namespace evelyn_found_caps_l153_153923

theorem evelyn_found_caps (start_caps end_caps found_caps : ℕ) 
    (h1 : start_caps = 18) 
    (h2 : end_caps = 81) 
    (h3 : found_caps = end_caps - start_caps) :
  found_caps = 63 := by
  sorry

end evelyn_found_caps_l153_153923


namespace number_of_people_in_group_l153_153706

theorem number_of_people_in_group (P : ℕ) : 
  (∃ (P : ℕ), 0 < P ∧ (364 / P - 1 = 364 / (P + 2))) → P = 26 :=
by
  sorry

end number_of_people_in_group_l153_153706


namespace plane_speed_in_still_air_l153_153943

theorem plane_speed_in_still_air (P W : ℝ) 
  (h1 : (P + W) * 3 = 900) 
  (h2 : (P - W) * 4 = 900) 
  : P = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l153_153943


namespace divide_circle_three_equal_areas_l153_153783

theorem divide_circle_three_equal_areas (OA : ℝ) (r1 r2 : ℝ) 
  (hr1 : r1 = (OA * Real.sqrt 3) / 3) 
  (hr2 : r2 = (OA * Real.sqrt 6) / 3) : 
  ∀ (r : ℝ), r = OA → 
  (∀ (A1 A2 A3 : ℝ), A1 = π * r1 ^ 2 ∧ A2 = π * (r2 ^ 2 - r1 ^ 2) ∧ A3 = π * (r ^ 2 - r2 ^ 2) →
  A1 = A2 ∧ A2 = A3) :=
by
  sorry

end divide_circle_three_equal_areas_l153_153783


namespace number_of_triangles_l153_153214

-- Define a structure representing a triangle with integer angles.
structure Triangle :=
  (A B C : ℕ) -- angles in integer degrees
  (angle_sum : A + B + C = 180)
  (obtuse_A : A > 90)

-- Define a structure representing point D on side BC of triangle ABC such that triangle ABD is right-angled
-- and triangle ADC is isosceles.
structure PointOnBC (ABC : Triangle) :=
  (D : ℕ) -- angle at D in triangle ABC
  (right_ABD : ABC.A = 90 ∨ ABC.B = 90 ∨ ABC.C = 90)
  (isosceles_ADC : ABC.A = ABC.B ∨ ABC.A = ABC.C ∨ ABC.B = ABC.C)

-- Problem Statement:
theorem number_of_triangles (t : Triangle) (d : PointOnBC t): ∃ n : ℕ, n = 88 :=
by
  sorry

end number_of_triangles_l153_153214


namespace fibonacci_invariant_abs_difference_l153_153651

-- Given the sequence defined by the recurrence relation
def mArithmetical_fibonacci (u_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u_n n = u_n (n - 2) + u_n (n - 1)

theorem fibonacci_invariant_abs_difference (u : ℕ → ℤ) 
  (h : mArithmetical_fibonacci u) :
  ∃ c : ℤ, ∀ n : ℕ, |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c := 
sorry

end fibonacci_invariant_abs_difference_l153_153651


namespace savings_example_l153_153144

def window_cost : ℕ → ℕ := λ n => n * 120

def discount_windows (n : ℕ) : ℕ := (n / 6) * 2 + n

def effective_cost (needed : ℕ) : ℕ := 
  let free_windows := (needed / 8) * 2
  (needed - free_windows) * 120

def combined_cost (n m : ℕ) : ℕ :=
  effective_cost (n + m)

def separate_cost (needed1 needed2 : ℕ) : ℕ :=
  effective_cost needed1 + effective_cost needed2

def savings_if_combined (n m : ℕ) : ℕ :=
  separate_cost n m - combined_cost n m

theorem savings_example : savings_if_combined 12 9 = 360 := by
  sorry

end savings_example_l153_153144


namespace probability_no_shaded_square_l153_153720

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l153_153720


namespace longest_segment_in_cylinder_l153_153726

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt (h^2 + (2*r)^2) :=
by
  sorry

end longest_segment_in_cylinder_l153_153726


namespace susans_total_chairs_l153_153035

def number_of_red_chairs := 5
def number_of_yellow_chairs := 4 * number_of_red_chairs
def number_of_blue_chairs := number_of_yellow_chairs - 2
def total_chairs := number_of_red_chairs + number_of_yellow_chairs + number_of_blue_chairs

theorem susans_total_chairs : total_chairs = 43 :=
by
  sorry

end susans_total_chairs_l153_153035


namespace proof_problem_l153_153498

open Set Real

noncomputable def f (x : ℝ) : ℝ := sin x
noncomputable def g (x : ℝ) : ℝ := cos x
def U : Set ℝ := univ
def M : Set ℝ := {x | f x ≠ 0}
def N : Set ℝ := {x | g x ≠ 0}
def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem proof_problem :
  {x : ℝ | f x * g x = 0} = (C_U M) ∪ (C_U N) :=
by
  sorry

end proof_problem_l153_153498


namespace range_is_correct_l153_153464

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x

def domain : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

def range_of_function : Set ℝ := {y | ∃ x ∈ domain, quadratic_function x = y}

theorem range_is_correct : range_of_function = Set.Icc (-4) 21 :=
by {
  sorry
}

end range_is_correct_l153_153464


namespace find_range_a_l153_153525

noncomputable def sincos_inequality (x a θ : ℝ) : Prop :=
  (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1 / 8

theorem find_range_a :
  (∀ (x : ℝ) (θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → sincos_inequality x a θ)
  ↔ a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end find_range_a_l153_153525


namespace completing_square_eq_sum_l153_153294

theorem completing_square_eq_sum :
  ∃ (a b c : ℤ), a > 0 ∧ (∀ (x : ℝ), 36 * x^2 - 60 * x + 25 = (a * x + b)^2 - c) ∧ a + b + c = 26 :=
by
  sorry

end completing_square_eq_sum_l153_153294


namespace leastCookies_l153_153852

theorem leastCookies (b : ℕ) :
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) →
  b = 179 :=
by
  sorry

end leastCookies_l153_153852


namespace trajectory_of_M_l153_153800

theorem trajectory_of_M (x y t : ℝ) (M P F : ℝ × ℝ)
    (hF : F = (1, 0))
    (hP : P = (1/4 * t^2, t))
    (hFP : (P.1 - F.1, P.2 - F.2) = (1/4 * t^2 - 1, t))
    (hFM : (M.1 - F.1, M.2 - F.2) = (x - 1, y))
    (hFP_FM : (P.1 - F.1, P.2 - F.2) = (2 * (M.1 - F.1), 2 * (M.2 - F.2))) :
  y^2 = 2 * x - 1 :=
by
  sorry

end trajectory_of_M_l153_153800


namespace last_two_digits_of_1976_pow_100_l153_153500

theorem last_two_digits_of_1976_pow_100 :
  (1976 ^ 100) % 100 = 76 :=
by
  sorry

end last_two_digits_of_1976_pow_100_l153_153500


namespace value_of_a1_plus_a10_l153_153002

noncomputable def geometric_sequence {α : Type*} [Field α] (a : ℕ → α) :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem value_of_a1_plus_a10 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : a 4 + a 7 = 2) 
  (h3 : a 5 * a 6 = -8) 
  : a 1 + a 10 = -7 := 
by
  sorry

end value_of_a1_plus_a10_l153_153002


namespace average_age_of_second_group_is_16_l153_153606

theorem average_age_of_second_group_is_16
  (total_age_15_students : ℕ := 225)
  (total_age_first_group_7_students : ℕ := 98)
  (age_15th_student : ℕ := 15) :
  (total_age_15_students - total_age_first_group_7_students - age_15th_student) / 7 = 16 := 
by
  sorry

end average_age_of_second_group_is_16_l153_153606


namespace angle_ABD_30_degrees_l153_153992

theorem angle_ABD_30_degrees (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB BD : ℝ) (angle_DBC : ℝ)
  (h1 : BD = AB * (Real.sqrt 3 / 2))
  (h2 : angle_DBC = 90) : 
  ∃ angle_ABD, angle_ABD = 30 :=
by
  sorry

end angle_ABD_30_degrees_l153_153992


namespace volume_ratio_l153_153225

namespace Geometry

variables {Point : Type} [MetricSpace Point]

noncomputable def volume_pyramid (A B1 C1 D1 : Point) : ℝ := sorry

theorem volume_ratio 
  (A B1 B2 C1 C2 D1 D2 : Point) 
  (hA_B1: dist A B1 ≠ 0) (hA_B2: dist A B2 ≠ 0)
  (hA_C1: dist A C1 ≠ 0) (hA_C2: dist A C2 ≠ 0)
  (hA_D1: dist A D1 ≠ 0) (hA_D2: dist A D2 ≠ 0) :
  (volume_pyramid A B1 C1 D1 / volume_pyramid A B2 C2 D2) = 
    (dist A B1 * dist A C1 * dist A D1) / (dist A B2 * dist A C2 * dist A D2) := 
sorry

end Geometry

end volume_ratio_l153_153225


namespace math_group_question_count_l153_153202

theorem math_group_question_count (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m = 6 ∧ n = 3 := 
sorry

end math_group_question_count_l153_153202


namespace sqrt_mixed_number_simplify_l153_153678

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l153_153678


namespace gg_of_3_is_107_l153_153009

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l153_153009


namespace amusement_park_people_l153_153947

theorem amusement_park_people (students adults free : ℕ) (total_people paid : ℕ) :
  students = 194 →
  adults = 235 →
  free = 68 →
  total_people = students + adults →
  paid = total_people - free →
  paid - free = 293 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end amusement_park_people_l153_153947


namespace part_1_part_2_l153_153745

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def A_def : A = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  ext x
  sorry
  
def B_def : B = {x : ℝ | x^2 + 2*x - 3 > 0} := by
  ext x
  sorry

theorem part_1 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0} := by
  rw [hA, hB]
  sorry

theorem part_2 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  (compl A ∩ B) = {x | x > 1 ∨ x < -3} := by
  rw [hA, hB]
  sorry

end part_1_part_2_l153_153745


namespace shirley_cases_needed_l153_153292

-- Define the given conditions
def trefoils_boxes := 54
def samoas_boxes := 36
def boxes_per_case := 6

-- The statement to prove
theorem shirley_cases_needed : trefoils_boxes / boxes_per_case >= samoas_boxes / boxes_per_case ∧ 
                               samoas_boxes / boxes_per_case = 6 :=
by
  let n_cases := samoas_boxes / boxes_per_case
  have h1 : trefoils_boxes / boxes_per_case = 9 := sorry
  have h2 : samoas_boxes / boxes_per_case = 6 := sorry
  have h3 : 9 >= 6 := by linarith
  exact ⟨h3, h2⟩


end shirley_cases_needed_l153_153292


namespace min_score_to_achieve_average_l153_153859

theorem min_score_to_achieve_average (a b c : ℕ) (h₁ : a = 76) (h₂ : b = 94) (h₃ : c = 87) :
  ∃ d e : ℕ, d + e = 148 ∧ d ≤ 100 ∧ e ≤ 100 ∧ min d e = 48 :=
by sorry

end min_score_to_achieve_average_l153_153859


namespace coconut_grove_l153_153802

theorem coconut_grove (x : ℕ) :
  (40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x) → 
  x = 7 := by
  sorry

end coconut_grove_l153_153802


namespace min_value_fraction_l153_153798

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 
  ( (x + 1) * (y + 1) / (x * y) ) >= 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_fraction_l153_153798


namespace tangent_line_through_point_and_circle_l153_153239

noncomputable def tangent_line_equation : String :=
  "y - 1 = 0"

theorem tangent_line_through_point_and_circle :
  ∀ (line_eq: String), 
  (∀ (x y: ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ (x, y) = (1, 1) → y - 1 = 0) →
  line_eq = tangent_line_equation :=
by
  intro line_eq h
  sorry

end tangent_line_through_point_and_circle_l153_153239


namespace find_fourth_term_l153_153727

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (a_1 a_4 d : ℕ)

-- Conditions
axiom sum_first_5 : S_n 5 = 35
axiom sum_first_9 : S_n 9 = 117
axiom sum_closed_form_first_5 : 5 * a_1 + (5 * (5 - 1)) / 2 * d = 35
axiom sum_closed_form_first_9 : 9 * a_1 + (9 * (9 - 1)) / 2 * d = 117
axiom nth_term_closed_form : ∀ n, a_n n = a_1 + (n-1)*d

-- Target
theorem find_fourth_term : a_4 = 10 := by
  sorry

end find_fourth_term_l153_153727


namespace jack_evening_emails_l153_153051

theorem jack_evening_emails (ema_morning ema_afternoon ema_afternoon_evening ema_evening : ℕ)
  (h1 : ema_morning = 4)
  (h2 : ema_afternoon = 5)
  (h3 : ema_afternoon_evening = 13)
  (h4 : ema_afternoon_evening = ema_afternoon + ema_evening) :
  ema_evening = 8 :=
by
  sorry

end jack_evening_emails_l153_153051


namespace class_speeds_relationship_l153_153169

theorem class_speeds_relationship (x : ℝ) (hx : 0 < x) :
    (15 / (1.2 * x)) = ((15 / x) - (1 / 2)) :=
sorry

end class_speeds_relationship_l153_153169


namespace initial_interest_rate_l153_153380

variable (P r : ℕ)

theorem initial_interest_rate (h1 : 405 = (P * r) / 100) (h2 : 450 = (P * (r + 5)) / 100) : r = 45 :=
sorry

end initial_interest_rate_l153_153380


namespace cars_meet_and_crush_fly_l153_153985

noncomputable def time_to_meet (L v_A v_B : ℝ) : ℝ := L / (v_A + v_B)

theorem cars_meet_and_crush_fly :
  ∀ (L v_A v_B v_fly : ℝ), L = 300 → v_A = 50 → v_B = 100 → v_fly = 150 → time_to_meet L v_A v_B = 2 :=
by
  intros L v_A v_B v_fly L_eq v_A_eq v_B_eq v_fly_eq
  rw [L_eq, v_A_eq, v_B_eq]
  simp [time_to_meet]
  norm_num

end cars_meet_and_crush_fly_l153_153985


namespace unsold_percentage_l153_153044

def total_harvested : ℝ := 340.2
def sold_mm : ℝ := 125.5  -- Weight sold to Mrs. Maxwell
def sold_mw : ℝ := 78.25  -- Weight sold to Mr. Wilson
def sold_mb : ℝ := 43.8   -- Weight sold to Ms. Brown
def sold_mj : ℝ := 56.65  -- Weight sold to Mr. Johnson

noncomputable def percentage_unsold (total_harvested : ℝ) 
                                   (sold_mm : ℝ) 
                                   (sold_mw : ℝ)
                                   (sold_mb : ℝ) 
                                   (sold_mj : ℝ) : ℝ :=
  let total_sold := sold_mm + sold_mw + sold_mb + sold_mj
  let unsold := total_harvested - total_sold
  (unsold / total_harvested) * 100

theorem unsold_percentage : percentage_unsold total_harvested sold_mm sold_mw sold_mb sold_mj = 10.58 :=
by
  sorry

end unsold_percentage_l153_153044


namespace tan_half_angle_product_zero_l153_153387

theorem tan_half_angle_product_zero (a b : ℝ) 
  (h: 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) 
  : Real.tan (a / 2) * Real.tan (b / 2) = 0 := 
by 
  sorry

end tan_half_angle_product_zero_l153_153387


namespace right_triangle_condition_l153_153474

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end right_triangle_condition_l153_153474


namespace gold_bars_total_worth_l153_153336

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end gold_bars_total_worth_l153_153336


namespace total_kids_attended_camp_l153_153550

theorem total_kids_attended_camp :
  let n1 := 34044
  let n2 := 424944
  n1 + n2 = 458988 := 
by {
  sorry
}

end total_kids_attended_camp_l153_153550


namespace remainder_proof_l153_153367

-- Definitions and conditions
variables {x y u v : ℕ}
variables (hx : x = u * y + v)

-- Problem statement in Lean 4
theorem remainder_proof (hx : x = u * y + v) : ((x + 3 * u * y + y) % y) = v :=
sorry

end remainder_proof_l153_153367


namespace find_x_l153_153228

theorem find_x (x : ℝ) (y : ℝ) : 
  (10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) ↔ x = (3 / 2) :=
by
  sorry

end find_x_l153_153228


namespace ratio_p_q_is_minus_one_l153_153950

theorem ratio_p_q_is_minus_one (p q : ℤ) (h : (25 / 7 : ℝ) + ((2 * q - p) / (2 * q + p) : ℝ) = 4) : (p / q : ℝ) = -1 := 
sorry

end ratio_p_q_is_minus_one_l153_153950


namespace shirley_sold_10_boxes_l153_153426

variable (cases boxes_per_case : ℕ)

-- Define the conditions
def number_of_cases := 5
def boxes_in_each_case := 2

-- Prove the total number of boxes is 10
theorem shirley_sold_10_boxes (H1 : cases = number_of_cases) (H2 : boxes_per_case = boxes_in_each_case) :
  cases * boxes_per_case = 10 := by
  sorry

end shirley_sold_10_boxes_l153_153426


namespace g_of_f_of_3_eq_1902_l153_153568

def f (x : ℕ) := x^3 - 2
def g (x : ℕ) := 3 * x^2 + x + 2

theorem g_of_f_of_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_of_3_eq_1902_l153_153568


namespace ratio_of_female_to_male_members_l153_153820

theorem ratio_of_female_to_male_members 
  (f m : ℕ) 
  (avg_age_female : ℕ) 
  (avg_age_male : ℕ)
  (avg_age_all : ℕ) 
  (H1 : avg_age_female = 45)
  (H2 : avg_age_male = 25)
  (H3 : avg_age_all = 35)
  (H4 : (f + m) ≠ 0) :
  (45 * f + 25 * m) / (f + m) = 35 → f = m :=
by sorry

end ratio_of_female_to_male_members_l153_153820


namespace trains_crossing_time_l153_153687

theorem trains_crossing_time :
  let length_first_train := 500
  let length_second_train := 800
  let speed_first_train := 80 * (5/18 : ℚ)  -- convert km/hr to m/s
  let speed_second_train := 100 * (5/18 : ℚ)  -- convert km/hr to m/s
  let relative_speed := speed_first_train + speed_second_train
  let total_distance := length_first_train + length_second_train
  let time_taken := total_distance / relative_speed
  time_taken = 26 :=
by
  sorry

end trains_crossing_time_l153_153687


namespace length_and_width_of_prism_l153_153274

theorem length_and_width_of_prism (w l h d : ℝ) (h_cond : h = 12) (d_cond : d = 15) (length_cond : l = 3 * w) :
  (w = 3) ∧ (l = 9) :=
by
  -- The proof is omitted as instructed in the task description.
  sorry

end length_and_width_of_prism_l153_153274


namespace mo_rainy_days_last_week_l153_153283

theorem mo_rainy_days_last_week (R NR n : ℕ) (h1 : n * R + 4 * NR = 26) (h2 : 4 * NR - n * R = 14) (h3 : R + NR = 7) : R = 2 :=
sorry

end mo_rainy_days_last_week_l153_153283


namespace garden_strawberry_area_l153_153996

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end garden_strawberry_area_l153_153996


namespace sasha_took_right_triangle_l153_153578

-- Define types of triangles
inductive Triangle
| acute
| right
| obtuse

open Triangle

-- Define the function that determines if Borya can form a triangle identical to Sasha's
def can_form_identical_triangle (t1 t2 t3: Triangle) : Bool :=
match t1, t2, t3 with
| right, acute, obtuse => true
| _ , _ , _ => false

-- Define the main theorem
theorem sasha_took_right_triangle : 
  ∀ (sasha_takes borya_takes1 borya_takes2 : Triangle),
  (sasha_takes ≠ borya_takes1 ∧ sasha_takes ≠ borya_takes2 ∧ borya_takes1 ≠ borya_takes2) →
  can_form_identical_triangle sasha_takes borya_takes1 borya_takes2 →
  sasha_takes = right :=
by sorry

end sasha_took_right_triangle_l153_153578


namespace velocity_ratio_proof_l153_153340

noncomputable def velocity_ratio (V U : ℝ) : ℝ := V / U

-- The conditions:
-- 1. A smooth horizontal surface.
-- 2. The speed of the ball is perpendicular to the face of the block.
-- 3. The mass of the ball is much smaller than the mass of the block.
-- 4. The collision is elastic.
-- 5. After the collision, the ball’s speed is halved and it moves in the opposite direction.

def ball_block_collision 
    (V U U_final : ℝ) 
    (smooth_surface : Prop) 
    (perpendicular_impact : Prop) 
    (ball_much_smaller : Prop) 
    (elastic_collision : Prop) 
    (speed_halved : Prop) : Prop :=
  U_final = U ∧ V / U = 4

theorem velocity_ratio_proof : 
  ∀ (V U U_final : ℝ)
    (smooth_surface : Prop)
    (perpendicular_impact : Prop)
    (ball_much_smaller : Prop)
    (elastic_collision : Prop)
    (speed_halved : Prop),
    ball_block_collision V U U_final smooth_surface perpendicular_impact ball_much_smaller elastic_collision speed_halved := 
sorry

end velocity_ratio_proof_l153_153340


namespace Megan_finish_all_problems_in_8_hours_l153_153863

theorem Megan_finish_all_problems_in_8_hours :
  ∀ (math_problems spelling_problems problems_per_hour : ℕ),
    math_problems = 36 →
    spelling_problems = 28 →
    problems_per_hour = 8 →
    (math_problems + spelling_problems) / problems_per_hour = 8 :=
by
  intros
  sorry

end Megan_finish_all_problems_in_8_hours_l153_153863


namespace mean_of_set_l153_153839

theorem mean_of_set (m : ℝ) (h : m + 7 = 12) :
  (m + (m + 6) + (m + 7) + (m + 11) + (m + 18)) / 5 = 13.4 :=
by sorry

end mean_of_set_l153_153839


namespace girl_travel_distance_l153_153163

def speed : ℝ := 6 -- meters per second
def time : ℕ := 16 -- seconds

def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem girl_travel_distance : distance speed time = 96 :=
by 
  unfold distance
  sorry

end girl_travel_distance_l153_153163


namespace max_regular_hours_l153_153388

/-- A man's regular pay is $3 per hour up to a certain number of hours, and his overtime pay rate
    is twice the regular pay rate. The man was paid $180 and worked 10 hours overtime.
    Prove that the maximum number of hours he can work at his regular pay rate is 40 hours.
-/
theorem max_regular_hours (P R OT : ℕ) (hP : P = 180) (hOT : OT = 10) (reg_rate overtime_rate : ℕ)
  (hreg_rate : reg_rate = 3) (hovertime_rate : overtime_rate = 2 * reg_rate) :
  P = reg_rate * R + overtime_rate * OT → R = 40 :=
by
  sorry

end max_regular_hours_l153_153388


namespace A_serves_on_50th_week_is_Friday_l153_153657

-- Define the people involved in the rotation
inductive Person
| A | B | C | D | E | F

open Person

-- Define the function that computes the day A serves on given the number of weeks
def day_A_serves (weeks : ℕ) : ℕ :=
  let days := weeks * 7
  (days % 6 + 0) % 7 -- 0 is the offset for the initial day when A serves (Sunday)

theorem A_serves_on_50th_week_is_Friday :
  day_A_serves 50 = 5 :=
by
  -- We provide the proof here
  sorry

end A_serves_on_50th_week_is_Friday_l153_153657


namespace Billy_weight_is_159_l153_153769

def Carl_weight : ℕ := 145
def Brad_weight : ℕ := Carl_weight + 5
def Billy_weight : ℕ := Brad_weight + 9

theorem Billy_weight_is_159 : Billy_weight = 159 := by
  sorry

end Billy_weight_is_159_l153_153769


namespace find_g_53_l153_153454

variable (g : ℝ → ℝ)

axiom functional_eq (x y : ℝ) : g (x * y) = y * g x
axiom g_one : g 1 = 10

theorem find_g_53 : g 53 = 530 :=
by
  sorry

end find_g_53_l153_153454


namespace pizza_slices_left_l153_153737

def initial_slices : ℕ := 16
def eaten_during_dinner : ℕ := initial_slices / 4
def remaining_after_dinner : ℕ := initial_slices - eaten_during_dinner
def yves_eaten : ℕ := remaining_after_dinner / 4
def remaining_after_yves : ℕ := remaining_after_dinner - yves_eaten
def siblings_eaten : ℕ := 2 * 2
def remaining_after_siblings : ℕ := remaining_after_yves - siblings_eaten

theorem pizza_slices_left : remaining_after_siblings = 5 := by
  sorry

end pizza_slices_left_l153_153737


namespace profit_per_tire_l153_153308

theorem profit_per_tire
  (fixed_cost : ℝ)
  (variable_cost_per_tire : ℝ)
  (selling_price_per_tire : ℝ)
  (batch_size : ℕ)
  (total_cost : ℝ)
  (total_revenue : ℝ)
  (total_profit : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : variable_cost_per_tire = 8)
  (h3 : selling_price_per_tire = 20)
  (h4 : batch_size = 15000)
  (h5 : total_cost = fixed_cost + variable_cost_per_tire * batch_size)
  (h6 : total_revenue = selling_price_per_tire * batch_size)
  (h7 : total_profit = total_revenue - total_cost)
  (h8 : profit_per_tire = total_profit / batch_size) :
  profit_per_tire = 10.50 :=
sorry

end profit_per_tire_l153_153308


namespace alice_ride_average_speed_l153_153471

theorem alice_ride_average_speed
    (d1 d2 : ℝ) 
    (s1 s2 : ℝ)
    (h_d1 : d1 = 40)
    (h_d2 : d2 = 20)
    (h_s1 : s1 = 8)
    (h_s2 : s2 = 40) :
    (d1 + d2) / (d1 / s1 + d2 / s2) = 10.909 :=
by
  simp [h_d1, h_d2, h_s1, h_s2]
  norm_num
  sorry

end alice_ride_average_speed_l153_153471


namespace find_all_waldo_time_l153_153744

theorem find_all_waldo_time (b : ℕ) (p : ℕ) (t : ℕ) :
  b = 15 → p = 30 → t = 3 → b * p * t = 1350 := by
sorry

end find_all_waldo_time_l153_153744


namespace thirty_three_and_one_third_percent_of_330_l153_153055

theorem thirty_three_and_one_third_percent_of_330 :
  (33 + 1 / 3) / 100 * 330 = 110 :=
sorry

end thirty_three_and_one_third_percent_of_330_l153_153055


namespace unique_solution_of_inequality_l153_153549

open Real

theorem unique_solution_of_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2 * b * x + 2 * b| ≤ 1) ↔ b = 1 := 
by exact sorry

end unique_solution_of_inequality_l153_153549


namespace cinnamon_balls_required_l153_153941

theorem cinnamon_balls_required 
  (num_family_members : ℕ) 
  (cinnamon_balls_per_day : ℕ) 
  (num_days : ℕ) 
  (h_family : num_family_members = 5) 
  (h_balls_per_day : cinnamon_balls_per_day = 5) 
  (h_days : num_days = 10) : 
  num_family_members * cinnamon_balls_per_day * num_days = 50 := by
  sorry

end cinnamon_balls_required_l153_153941


namespace minimum_value_f_l153_153644

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem minimum_value_f :
  ∃ x > 0, (∀ y > 0, f x ≤ f y) ∧ f x = 1 :=
sorry

end minimum_value_f_l153_153644


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_4_l153_153632

-- Define the function representing the quadratic equation
def quadratic_eq (m x : ℝ) := x^2 + (2 - m) * x + 1 - m

-- Part 1
theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
sorry

-- Part 2
theorem find_m_for_roots_difference_4 (m : ℝ) (H : m < 0) :
  (∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 ∧ x1 - x2 = 4) → m = -4 :=
sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_4_l153_153632


namespace triangle_angles_inequality_l153_153672

theorem triangle_angles_inequality (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) 
(h5 : A < Real.pi) (h6 : B < Real.pi) (h7 : C < Real.pi) : 
  A * Real.cos B + Real.sin A * Real.sin C > 0 := 
by 
  sorry

end triangle_angles_inequality_l153_153672


namespace valentines_cards_count_l153_153443

theorem valentines_cards_count (x y : ℕ) (h1 : x * y = x + y + 30) : x * y = 64 :=
by {
    sorry
}

end valentines_cards_count_l153_153443


namespace simplify_fraction_l153_153417

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end simplify_fraction_l153_153417


namespace Alex_is_26_l153_153150

-- Define the ages as integers
variable (Alex Jose Zack Inez : ℤ)

-- Conditions of the problem
variable (h1 : Alex = Jose + 6)
variable (h2 : Zack = Inez + 5)
variable (h3 : Inez = 18)
variable (h4 : Jose = Zack - 3)

-- Theorem we need to prove
theorem Alex_is_26 (h1: Alex = Jose + 6) (h2 : Zack = Inez + 5) (h3 : Inez = 18) (h4 : Jose = Zack - 3) : Alex = 26 :=
by
  sorry

end Alex_is_26_l153_153150


namespace opposite_of_neg_2023_l153_153999

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l153_153999


namespace total_shirts_correct_l153_153305

def machine_A_production_rate := 6
def machine_A_yesterday_minutes := 12
def machine_A_today_minutes := 10

def machine_B_production_rate := 8
def machine_B_yesterday_minutes := 10
def machine_B_today_minutes := 15

def machine_C_production_rate := 5
def machine_C_yesterday_minutes := 20
def machine_C_today_minutes := 0

def total_shirts_produced : Nat :=
  (machine_A_production_rate * machine_A_yesterday_minutes +
  machine_A_production_rate * machine_A_today_minutes) +
  (machine_B_production_rate * machine_B_yesterday_minutes +
  machine_B_production_rate * machine_B_today_minutes) +
  (machine_C_production_rate * machine_C_yesterday_minutes +
  machine_C_production_rate * machine_C_today_minutes)

theorem total_shirts_correct : total_shirts_produced = 432 :=
by 
  sorry 

end total_shirts_correct_l153_153305


namespace lines_with_equal_intercepts_l153_153603

theorem lines_with_equal_intercepts (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (n : ℕ), n = 3 ∧ (∀ l : ℝ → ℝ, (l 1 = 2) → ((l 0 = l (-0)) ∨ (l (-0) = l 0))) :=
by
  sorry

end lines_with_equal_intercepts_l153_153603


namespace decimal_to_fraction_l153_153856

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l153_153856


namespace unique_root_of_quadratic_eq_l153_153809

theorem unique_root_of_quadratic_eq (a b c : ℝ) (d : ℝ) 
  (h_seq : b = a - d ∧ c = a - 2 * d) 
  (h_nonneg : a ≥ b ∧ b ≥ c ∧ c ≥ 0) 
  (h_discriminant : (-(a - d))^2 - 4 * a * (a - 2 * d) = 0) :
  ∃ x : ℝ, (ax^2 - bx + c = 0) ∧ x = 1 / 2 :=
by
  sorry

end unique_root_of_quadratic_eq_l153_153809


namespace companyKW_price_percentage_l153_153042

theorem companyKW_price_percentage (A B P : ℝ) (h1 : P = 1.40 * A) (h2 : P = 2.00 * B) : 
  P / ((P / 1.40) + (P / 2.00)) * 100 = 82.35 :=
by sorry

end companyKW_price_percentage_l153_153042


namespace number_identification_l153_153928

theorem number_identification (x : ℝ) (h : x ^ 655 / x ^ 650 = 100000) : x = 10 :=
by
  sorry

end number_identification_l153_153928


namespace quadratic_reciprocal_squares_l153_153924

theorem quadratic_reciprocal_squares :
  (∃ p q : ℝ, (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = p ∨ x = q)) ∧ (1 / p^2 + 1 / q^2 = 13 / 4)) :=
by
  have quadratic_eq : (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = 1 ∨ x = 2 / 3)) := sorry
  have identity_eq : 1 / (1:ℝ)^2 + 1 / (2 / 3)^2 = 13 / 4 := sorry
  exact ⟨1, 2 / 3, quadratic_eq, identity_eq⟩

end quadratic_reciprocal_squares_l153_153924


namespace find_t_l153_153321

theorem find_t :
  ∃ (B : ℝ × ℝ) (t : ℝ), 
  B.1^2 + B.2^2 = 100 ∧ 
  B.1 - 2 * B.2 + 10 = 0 ∧ 
  B.1 > 0 ∧ B.2 > 0 ∧ 
  t = 20 ∧ 
  (∃ m : ℝ, 
    m = -2 ∧ 
    B.2 = m * B.1 + (8 + 2 * B.1 - m * B.1)) := 
by
  sorry

end find_t_l153_153321


namespace pencils_in_drawer_l153_153934

theorem pencils_in_drawer (P : ℕ) (h1 : P + 19 + 16 = 78) : P = 43 :=
by
  sorry

end pencils_in_drawer_l153_153934


namespace extreme_point_f_l153_153981

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 1)

theorem extreme_point_f :
  ∃ x : ℝ, (∀ y : ℝ, y ≠ 0 → (Real.exp y * y < 0 ↔ y < x)) ∧ x = 0 :=
by
  sorry

end extreme_point_f_l153_153981


namespace fabric_sales_fraction_l153_153683

def total_sales := 36
def stationery_sales := 15
def jewelry_sales := total_sales / 4
def fabric_sales := total_sales - jewelry_sales - stationery_sales

theorem fabric_sales_fraction:
  (fabric_sales : ℝ) / total_sales = 1 / 3 :=
by
  sorry

end fabric_sales_fraction_l153_153683


namespace joshua_total_bottle_caps_l153_153100

def initial_bottle_caps : ℕ := 40
def bought_bottle_caps : ℕ := 7

theorem joshua_total_bottle_caps : initial_bottle_caps + bought_bottle_caps = 47 := 
by
  sorry

end joshua_total_bottle_caps_l153_153100


namespace complex_number_eq_l153_153122

theorem complex_number_eq (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 :=
sorry

end complex_number_eq_l153_153122


namespace triangle_inequality_min_diff_l153_153360

theorem triangle_inequality_min_diff
  (DE EF FD : ℕ) 
  (h1 : DE + EF + FD = 398)
  (h2 : DE < EF ∧ EF ≤ FD) : 
  EF - DE = 1 :=
by
  sorry

end triangle_inequality_min_diff_l153_153360


namespace rational_number_div_l153_153038

theorem rational_number_div (x : ℚ) (h : -2 / x = 8) : x = -1 / 4 := 
by
  sorry

end rational_number_div_l153_153038


namespace problem_conditions_and_inequalities_l153_153463

open Real

theorem problem_conditions_and_inequalities (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 2 * b = a * b) :
  (a + 2 * b ≥ 8) ∧ (2 * a + b ≥ 9) ∧ (a ^ 2 + 4 * b ^ 2 + 5 * a * b ≥ 72) ∧ ¬(logb 2 a + logb 2 b < 3) :=
by
  sorry

end problem_conditions_and_inequalities_l153_153463


namespace smallest_number_am_median_l153_153249

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end smallest_number_am_median_l153_153249


namespace compound_interest_rate_l153_153113

theorem compound_interest_rate :
  ∃ r : ℝ, (1000 * (1 + r)^3 = 1331.0000000000005) ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l153_153113


namespace largest_divisor_of_five_consecutive_integers_product_correct_l153_153784

noncomputable def largest_divisor_of_five_consecutive_integers_product : ℕ :=
  120

theorem largest_divisor_of_five_consecutive_integers_product_correct :
  ∀ (n : ℕ), (∃ k : ℕ, k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ∧ 120 ∣ k) :=
sorry

end largest_divisor_of_five_consecutive_integers_product_correct_l153_153784


namespace find_parabola_equation_l153_153347

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
    (y = a * x ^ 2 + b * x + c) ∧ 
    (y = (x - 3) ^ 2 - 2) ∧
    (a * (4 - 3) ^ 2 - 2 = 2)

theorem find_parabola_equation :
  ∃ (a b c : ℝ), parabola_equation a b c ∧ a = 4 ∧ b = -24 ∧ c = 34 :=
sorry

end find_parabola_equation_l153_153347


namespace line_equation_and_inclination_l153_153386

variable (t : ℝ)
variable (x y : ℝ)
variable (α : ℝ)
variable (l : x = -3 + t ∧ y = 1 + sqrt 3 * t)

theorem line_equation_and_inclination 
  (H : l) : 
  (∃ a b c : ℝ, a = sqrt 3 ∧ b = -1 ∧ c = 3 * sqrt 3 + 1 ∧ a * x + b * y + c = 0) ∧
  α = Real.pi / 3 :=
by
  sorry

end line_equation_and_inclination_l153_153386


namespace length_base_bc_l153_153070

theorem length_base_bc {A B C D : Type} [Inhabited A]
  (AB AC : ℕ)
  (BD : ℕ → ℕ → ℕ → ℕ) -- function for the median on AC
  (perimeter1 perimeter2 : ℕ)
  (h1 : AB = AC)
  (h2 : perimeter1 = 24 ∨ perimeter2 = 30)
  (AD CD : ℕ) :
  (AD = CD ∧ (∃ ab ad cd, ab + ad = perimeter1 ∧ cd + ad = perimeter2 ∧ ((AB = 2 * AD ∧ BC = 30 - CD) ∨ (AB = 2 * AD ∧ BC = 24 - CD)))) →
  (BC = 22 ∨ BC = 14) := 
sorry

end length_base_bc_l153_153070


namespace problem_xyz_l153_153883

theorem problem_xyz (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -8) :
  x^2 + y^2 = 32 :=
by
  sorry

end problem_xyz_l153_153883


namespace enclosed_polygons_l153_153961

theorem enclosed_polygons (n : ℕ) :
  (∃ α β : ℝ, (15 * β) = 360 ∧ β = 180 - α ∧ (15 * α) = 180 * (n - 2) / n) ↔ n = 15 :=
by sorry

end enclosed_polygons_l153_153961


namespace find_n_l153_153020

theorem find_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = (Real.log (s / P)) / (Real.log (1 + k + m)) :=
sorry

end find_n_l153_153020


namespace minimum_distance_between_tracks_l153_153822

-- Problem statement as Lean definitions and theorem to prove
noncomputable def rational_man_track (t : ℝ) : ℝ × ℝ :=
  (Real.cos t, Real.sin t)

noncomputable def hyperbolic_man_track (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 * Real.cos (t / 2), 5 * Real.sin (t / 2))

noncomputable def circle_eq := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def ellipse_eq := {p : ℝ × ℝ | (p.1 + 1)^2 / 9 + p.2^2 / 25 = 1}

theorem minimum_distance_between_tracks : 
  ∃ A ∈ circle_eq, ∃ B ∈ ellipse_eq, dist A B = Real.sqrt 14 - 1 := 
sorry

end minimum_distance_between_tracks_l153_153822


namespace can_construct_segment_l153_153997

noncomputable def constructSegment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P

theorem can_construct_segment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P) :=
sorry

end can_construct_segment_l153_153997


namespace a_minus_3d_eq_zero_l153_153664

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x - 3 * d)

theorem a_minus_3d_eq_zero (a b c d : ℝ) (h : f a b c d ≠ 0)
  (h1 : ∀ x, f a b c d x = x) :
  a - 3 * d = 0 :=
sorry

end a_minus_3d_eq_zero_l153_153664


namespace total_sheep_l153_153930

-- Define the conditions as hypotheses
variables (Aaron_sheep Beth_sheep : ℕ)
def condition1 := Aaron_sheep = 7 * Beth_sheep
def condition2 := Aaron_sheep = 532
def condition3 := Beth_sheep = 76

-- Assert that under these conditions, the total number of sheep is 608.
theorem total_sheep
  (h1 : condition1 Aaron_sheep Beth_sheep)
  (h2 : condition2 Aaron_sheep)
  (h3 : condition3 Beth_sheep) :
  Aaron_sheep + Beth_sheep = 608 :=
by sorry

end total_sheep_l153_153930


namespace shopping_people_count_l153_153398

theorem shopping_people_count :
  ∃ P : ℕ, P = 10 ∧
  ∃ (stores : ℕ) (total_visits : ℕ) (two_store_visitors : ℕ) 
    (at_least_one_store_visitors : ℕ) (max_stores_visited : ℕ),
    stores = 8 ∧
    total_visits = 22 ∧
    two_store_visitors = 8 ∧
    at_least_one_store_visitors = P ∧
    max_stores_visited = 3 ∧
    total_visits = (two_store_visitors * 2) + 6 ∧
    P = two_store_visitors + 2 :=
by {
    sorry
}

end shopping_people_count_l153_153398


namespace kia_vehicle_count_l153_153908

theorem kia_vehicle_count (total_vehicles : Nat) (dodge_vehicles : Nat) (hyundai_vehicles : Nat) 
    (h1 : total_vehicles = 400)
    (h2 : dodge_vehicles = total_vehicles / 2)
    (h3 : hyundai_vehicles = dodge_vehicles / 2) : 
    (total_vehicles - dodge_vehicles - hyundai_vehicles) = 100 := 
by sorry

end kia_vehicle_count_l153_153908


namespace symmetric_circle_l153_153276

variable (x y : ℝ)

def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem symmetric_circle :
  (∃ x y, original_circle x y) → (x^2 + (y + 2)^2 = 5) :=
sorry

end symmetric_circle_l153_153276


namespace core_temperature_calculation_l153_153579

-- Define the core temperature of the Sun, given in degrees Celsius
def T_Sun : ℝ := 19200000

-- Define the multiple factor
def factor : ℝ := 312.5

-- The expected result in scientific notation
def expected_temperature : ℝ := 6.0 * (10 ^ 9)

-- Prove that the calculated temperature is equal to the expected temperature
theorem core_temperature_calculation : (factor * T_Sun) = expected_temperature := by
  sorry

end core_temperature_calculation_l153_153579


namespace bus_stoppage_time_per_hour_l153_153141

theorem bus_stoppage_time_per_hour
  (speed_excluding_stoppages : ℕ) 
  (speed_including_stoppages : ℕ)
  (h1 : speed_excluding_stoppages = 54) 
  (h2 : speed_including_stoppages = 45) 
  : (60 * (speed_excluding_stoppages - speed_including_stoppages) / speed_excluding_stoppages) = 10 :=
by sorry

end bus_stoppage_time_per_hour_l153_153141


namespace complement_A_in_U_l153_153272

def U := {x : ℝ | -4 < x ∧ x < 4}
def A := {x : ℝ | -3 ≤ x ∧ x < 2}

theorem complement_A_in_U :
  {x : ℝ | x ∈ U ∧ x ∉ A} = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by {
  sorry
}

end complement_A_in_U_l153_153272


namespace evaluate_expression_l153_153920

theorem evaluate_expression (x z : ℤ) (h1 : x = 2) (h2 : z = 1) : z * (z - 4 * x) = -7 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l153_153920


namespace y_completion_days_l153_153238

theorem y_completion_days (d : ℕ) (h : (12 : ℚ) / d + 1 / 4 = 1) : d = 16 :=
by
  sorry

end y_completion_days_l153_153238


namespace nonempty_solution_iff_a_gt_one_l153_153896

theorem nonempty_solution_iff_a_gt_one (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
sorry

end nonempty_solution_iff_a_gt_one_l153_153896


namespace annabelle_savings_l153_153530

noncomputable def weeklyAllowance : ℕ := 30
noncomputable def junkFoodFraction : ℚ := 1 / 3
noncomputable def sweetsCost : ℕ := 8

theorem annabelle_savings :
  let junkFoodCost := weeklyAllowance * junkFoodFraction
  let totalSpent := junkFoodCost + sweetsCost
  let savings := weeklyAllowance - totalSpent
  savings = 12 := 
by
  sorry

end annabelle_savings_l153_153530


namespace max_value_of_expression_l153_153235

theorem max_value_of_expression :
  ∃ x : ℝ, ∀ y : ℝ, -x^2 + 4*x + 10 ≤ -y^2 + 4*y + 10 ∧ -x^2 + 4*x + 10 = 14 :=
sorry

end max_value_of_expression_l153_153235


namespace triangle_inequality_equality_condition_l153_153584

variables {A B C a b c : ℝ}

theorem triangle_inequality (A a B b C c : ℝ) :
  A * a + B * b + C * c ≥ 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b) :=
sorry

theorem equality_condition (A B C a b c : ℝ) :
  (A * a + B * b + C * c = 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b)) ↔ (a = b ∧ b = c ∧ A = B ∧ B = C) :=
sorry

end triangle_inequality_equality_condition_l153_153584


namespace bellas_score_l153_153675

theorem bellas_score (sum_19 : ℕ) (sum_20 : ℕ) (avg_19 : ℕ) (avg_20 : ℕ) (n_19 : ℕ) (n_20 : ℕ) :
  avg_19 = 82 → avg_20 = 85 → n_19 = 19 → n_20 = 20 → sum_19 = n_19 * avg_19 → sum_20 = n_20 * avg_20 →
  sum_20 - sum_19 = 142 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end bellas_score_l153_153675


namespace garden_dimensions_l153_153161

theorem garden_dimensions (l b : ℝ) (walkway_width total_area perimeter : ℝ) 
  (h1 : l = 3 * b)
  (h2 : perimeter = 2 * l + 2 * b)
  (h3 : walkway_width = 1)
  (h4 : total_area = (l + 2 * walkway_width) * (b + 2 * walkway_width))
  (h5 : perimeter = 40)
  (h6 : total_area = 120) : 
  l = 15 ∧ b = 5 ∧ total_area - l * b = 45 :=  
  by
  sorry

end garden_dimensions_l153_153161


namespace perpendicular_line_through_point_l153_153674

open Real

def line (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) :
  (line 2 1 (-5) x y) → (x = 3) ∧ (y = 0) → (line 1 (-2) 3 x y) := by
sorry

end perpendicular_line_through_point_l153_153674


namespace f_11_5_equals_neg_1_l153_153557

-- Define the function f with the given properties
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom periodic_function (f : ℝ → ℝ) : ∀ x, f (x + 2) = f x
axiom f_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x

-- State the theorem to be proved
theorem f_11_5_equals_neg_1 (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (periodic_f : ∀ x, f (x + 2) = f x)
  (f_int : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (11.5) = -1 :=
sorry

end f_11_5_equals_neg_1_l153_153557


namespace perimeter_of_triangle_l153_153050

theorem perimeter_of_triangle
  {r A P : ℝ} (hr : r = 2.5) (hA : A = 25) :
  P = 20 :=
by
  sorry

end perimeter_of_triangle_l153_153050


namespace perimeter_shaded_region_l153_153789

-- Definitions based on conditions
def circle_radius : ℝ := 10
def central_angle : ℝ := 300

-- Statement: Perimeter of the shaded region
theorem perimeter_shaded_region 
  : (10 : ℝ) + (10 : ℝ) + ((5 / 6) * (2 * Real.pi * 10)) = (20 : ℝ) + (50 / 3) * Real.pi :=
by
  sorry

end perimeter_shaded_region_l153_153789


namespace parallelogram_area_l153_153397

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (hb : b = 15) (hh : h = 2 * b) (hA : A = b * h) : A = 450 := 
by
  rw [hb, hh] at hA
  rw [hA]
  sorry

end parallelogram_area_l153_153397


namespace total_cost_is_correct_l153_153032

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end total_cost_is_correct_l153_153032


namespace batsman_average_after_17th_inning_l153_153848

theorem batsman_average_after_17th_inning
  (A : ℕ)
  (h1 : (16 * A + 88) / 17 = A + 3) :
  37 + 3 = 40 :=
by sorry

end batsman_average_after_17th_inning_l153_153848


namespace rectangle_sides_l153_153384

theorem rectangle_sides (k : ℝ) (μ : ℝ) (a b : ℝ) 
  (h₀ : k = 8) 
  (h₁ : μ = 3/10) 
  (h₂ : 2 * (a + b) = k) 
  (h₃ : a * b = μ * (a^2 + b^2)) : 
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) :=
sorry

end rectangle_sides_l153_153384


namespace k_for_circle_radius_7_l153_153127

theorem k_for_circle_radius_7 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0) →
  (∃ x y : ℝ, (x + 4)^2 + (y + 2)^2 = 49) →
  k = 29 :=
by
  sorry

end k_for_circle_radius_7_l153_153127


namespace A_is_7056_l153_153190

-- Define the variables and conditions
def D : ℕ := 4 * 3
def E : ℕ := 7 * 3
def B : ℕ := 4 * D
def C : ℕ := 7 * E
def A : ℕ := B * C

-- Prove that A = 7056 given the conditions
theorem A_is_7056 : A = 7056 := by
  -- We will skip the proof steps with 'sorry'
  sorry

end A_is_7056_l153_153190


namespace length_of_platform_l153_153962

-- Given conditions
def train_length : ℝ := 100
def time_pole : ℝ := 15
def time_platform : ℝ := 40

-- Theorem to prove the length of the platform
theorem length_of_platform (L : ℝ) 
    (h_train_length : train_length = 100)
    (h_time_pole : time_pole = 15)
    (h_time_platform : time_platform = 40)
    (h_speed : (train_length / time_pole) = (100 + L) / time_platform) : 
    L = 500 / 3 :=
by
  sorry

end length_of_platform_l153_153962


namespace spelling_bee_participants_l153_153915

theorem spelling_bee_participants (n : ℕ)
  (h1 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (k - 1 < 74 ∨ k - 1 > 74))
  (h2 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (75 - k > 0 ∨ k - 1 > 74)) :
  n = 149 := by
  sorry

end spelling_bee_participants_l153_153915


namespace total_spent_amount_l153_153667

-- Define the conditions
def spent_relation (B D : ℝ) : Prop := D = 0.75 * B
def payment_difference (B D : ℝ) : Prop := B = D + 12.50

-- Define the theorem to prove
theorem total_spent_amount (B D : ℝ) 
  (h1 : spent_relation B D) 
  (h2 : payment_difference B D) : 
  B + D = 87.50 :=
sorry

end total_spent_amount_l153_153667


namespace count_special_integers_l153_153563

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

end count_special_integers_l153_153563


namespace right_triangle_set_C_l153_153577

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_set_C_l153_153577


namespace find_x_l153_153472

def operation_eur (x y : ℕ) : ℕ := 3 * x * y

theorem find_x (y x : ℕ) (h1 : y = 3) (h2 : operation_eur y (operation_eur x 5) = 540) : x = 4 :=
by
  sorry

end find_x_l153_153472


namespace find_x_plus_one_over_x_l153_153635

open Real

theorem find_x_plus_one_over_x (x : ℝ) (h : x ^ 3 + 1 / x ^ 3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_plus_one_over_x_l153_153635


namespace split_cost_evenly_l153_153973

noncomputable def cupcake_cost : ℝ := 1.50
noncomputable def number_of_cupcakes : ℝ := 12
noncomputable def total_cost : ℝ := number_of_cupcakes * cupcake_cost
noncomputable def total_people : ℝ := 2

theorem split_cost_evenly : (total_cost / total_people) = 9 :=
by
  -- Skipping the proof for now
  sorry

end split_cost_evenly_l153_153973


namespace average_increase_l153_153160

theorem average_increase (x : ℝ) (y : ℝ) (h : y = 0.245 * x + 0.321) : 
  ∀ x_increase : ℝ, x_increase = 1 → (0.245 * (x + x_increase) + 0.321) - (0.245 * x + 0.321) = 0.245 :=
by
  intro x_increase
  intro hx
  rw [hx]
  simp
  sorry

end average_increase_l153_153160


namespace georgia_total_carnation_cost_l153_153329

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l153_153329


namespace pets_remaining_l153_153866

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end pets_remaining_l153_153866


namespace isosceles_triangle_third_vertex_y_coord_l153_153984

theorem isosceles_triangle_third_vertex_y_coord :
  ∀ (A B : ℝ × ℝ) (θ : ℝ), 
  A = (0, 5) → B = (8, 5) → θ = 60 → 
  ∃ (C : ℝ × ℝ), C.fst > 0 ∧ C.snd > 5 ∧ C.snd = 5 + 4 * Real.sqrt 3 :=
by
  intros A B θ hA hB hθ
  use (4, 5 + 4 * Real.sqrt 3)
  sorry

end isosceles_triangle_third_vertex_y_coord_l153_153984


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l153_153376

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l153_153376


namespace hot_dog_cost_l153_153655

theorem hot_dog_cost : 
  ∃ h d : ℝ, (3 * h + 4 * d = 10) ∧ (2 * h + 3 * d = 7) ∧ (d = 1) := 
by 
  sorry

end hot_dog_cost_l153_153655


namespace cube_rolling_impossible_l153_153162

-- Definitions
def paintedCube : Type := sorry   -- Define a painted black-and-white cube.
def chessboard : Type := sorry    -- Define the chessboard.
def roll (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the rolling over the board visiting each square exactly once.
def matchColors (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the condition that colors match on contact.

-- Theorem
theorem cube_rolling_impossible (c : paintedCube) (b : chessboard)
  (h1 : roll c b) : ¬ matchColors c b := sorry

end cube_rolling_impossible_l153_153162


namespace f_neg_def_l153_153198

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def f_pos_def (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x = x * (1 - x)

theorem f_neg_def (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f_pos_def f) :
  ∀ x : ℝ, x < 0 → f x = x * (1 + x) :=
by
  sorry

end f_neg_def_l153_153198


namespace root_conditions_imply_sum_l153_153409

-- Define the variables a and b in the context that their values fit the given conditions.
def a : ℝ := 5
def b : ℝ := -6

-- Define the quadratic equation and conditions on roots.
def quadratic_eq (x : ℝ) := x^2 - a * x - b

-- Given that 2 and 3 are the roots of the quadratic equation.
def roots_condition := (quadratic_eq 2 = 0) ∧ (quadratic_eq 3 = 0)

-- The theorem to prove.
theorem root_conditions_imply_sum :
  roots_condition → a + b = -1 :=
by
sorry

end root_conditions_imply_sum_l153_153409


namespace find_x_l153_153821

variables {a b : EuclideanSpace ℝ (Fin 2)} {x : ℝ}

theorem find_x (h1 : ‖a + b‖ = 1) (h2 : ‖a - b‖ = x) (h3 : inner a b = -(3 / 8) * x) : x = 2 ∨ x = -(1 / 2) :=
sorry

end find_x_l153_153821


namespace parabola_distance_relation_l153_153804

theorem parabola_distance_relation {n : ℝ} {x₁ x₂ y₁ y₂ : ℝ}
  (h₁ : y₁ = x₁^2 - 4 * x₁ + n)
  (h₂ : y₂ = x₂^2 - 4 * x₂ + n)
  (h : y₁ > y₂) :
  |x₁ - 2| > |x₂ - 2| := 
sorry

end parabola_distance_relation_l153_153804


namespace expected_volunteers_2008_l153_153275

theorem expected_volunteers_2008 (initial_volunteers: ℕ) (annual_increase: ℚ) (h1: initial_volunteers = 500) (h2: annual_increase = 1.2) : 
  let volunteers_2006 := initial_volunteers * annual_increase
  let volunteers_2007 := volunteers_2006 * annual_increase
  let volunteers_2008 := volunteers_2007 * annual_increase
  volunteers_2008 = 864 := 
by
  sorry

end expected_volunteers_2008_l153_153275


namespace probability_of_composite_l153_153155

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l153_153155


namespace f_20_value_l153_153438

noncomputable def f (n : ℕ) : ℚ := sorry

axiom f_initial : f 1 = 3 / 2
axiom f_eq : ∀ x y : ℕ, 
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_20_value : f 20 = 4305 := 
by {
  sorry 
}

end f_20_value_l153_153438


namespace abs_a_gt_b_l153_153059

theorem abs_a_gt_b (a b : ℝ) (h : a > b) : |a| > b :=
sorry

end abs_a_gt_b_l153_153059


namespace compound_percentage_increase_l153_153018

noncomputable def weeklyEarningsAfterRaises (initial : ℝ) (raises : List ℝ) : ℝ :=
  raises.foldl (λ sal raise_rate => sal * (1 + raise_rate / 100)) initial

theorem compound_percentage_increase :
  let initial := 60
  let raises := [10, 15, 12, 8]
  weeklyEarningsAfterRaises initial raises = 91.80864 ∧
  ((weeklyEarningsAfterRaises initial raises - initial) / initial * 100 = 53.0144) :=
by
  sorry

end compound_percentage_increase_l153_153018


namespace total_number_of_matches_l153_153755

-- Define the total number of teams
def numberOfTeams : ℕ := 10

-- Define the number of matches each team competes against each other team
def matchesPerPair : ℕ := 4

-- Calculate the total number of unique matches
def calculateUniqueMatches (teams : ℕ) : ℕ :=
  (teams * (teams - 1)) / 2

-- Main statement to be proved
theorem total_number_of_matches : calculateUniqueMatches numberOfTeams * matchesPerPair = 180 := by
  -- Placeholder for the proof
  sorry

end total_number_of_matches_l153_153755


namespace consecutive_odd_numbers_sum_power_fourth_l153_153401

theorem consecutive_odd_numbers_sum_power_fourth :
  ∃ x1 x2 x3 : ℕ, 
  x1 % 2 = 1 ∧ x2 % 2 = 1 ∧ x3 % 2 = 1 ∧ 
  x1 + 2 = x2 ∧ x2 + 2 = x3 ∧ 
  (∃ n : ℕ, n < 10 ∧ (x1 + x2 + x3 = n^4)) :=
sorry

end consecutive_odd_numbers_sum_power_fourth_l153_153401


namespace rectangle_perimeter_l153_153679

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * (2 * a + 2 * b)) : 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l153_153679


namespace percent_of_whole_is_fifty_l153_153619

theorem percent_of_whole_is_fifty (part whole : ℝ) (h1 : part = 180) (h2 : whole = 360) : 
  ((part / whole) * 100) = 50 := 
by 
  rw [h1, h2] 
  sorry

end percent_of_whole_is_fifty_l153_153619


namespace train_cross_time_l153_153862

noncomputable def speed_conversion (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def time_to_cross_pole (length_m speed_kmh : ℝ) : ℝ :=
  length_m / speed_conversion speed_kmh

theorem train_cross_time (length_m : ℝ) (speed_kmh : ℝ) :
  length_m = 225 → speed_kmh = 250 → time_to_cross_pole length_m speed_kmh = 3.24 := by
  intros hlen hspeed
  simp [time_to_cross_pole, speed_conversion, hlen, hspeed]
  sorry

end train_cross_time_l153_153862


namespace equality_of_arithmetic_sums_l153_153829

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem equality_of_arithmetic_sums (n : ℕ) (h : n ≠ 0) :
  sum_arithmetic_sequence 8 4 n = sum_arithmetic_sequence 17 2 n ↔ n = 10 :=
by
  sorry

end equality_of_arithmetic_sums_l153_153829


namespace car_repair_cost_l153_153442

noncomputable def total_cost (first_mechanic_rate: ℝ) (first_mechanic_hours: ℕ) 
    (first_mechanic_days: ℕ) (second_mechanic_rate: ℝ) 
    (second_mechanic_hours: ℕ) (second_mechanic_days: ℕ) 
    (discount_first: ℝ) (discount_second: ℝ) 
    (parts_cost: ℝ) (sales_tax_rate: ℝ): ℝ :=
  let first_mechanic_cost := first_mechanic_rate * first_mechanic_hours * first_mechanic_days
  let second_mechanic_cost := second_mechanic_rate * second_mechanic_hours * second_mechanic_days
  let first_mechanic_discounted := first_mechanic_cost - (discount_first * first_mechanic_cost)
  let second_mechanic_discounted := second_mechanic_cost - (discount_second * second_mechanic_cost)
  let total_before_tax := first_mechanic_discounted + second_mechanic_discounted + parts_cost
  let sales_tax := sales_tax_rate * total_before_tax
  total_before_tax + sales_tax

theorem car_repair_cost :
  total_cost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end car_repair_cost_l153_153442


namespace bottle_capacity_l153_153624

theorem bottle_capacity
  (num_boxes : ℕ)
  (bottles_per_box : ℕ)
  (fill_fraction : ℚ)
  (total_volume : ℚ)
  (total_bottles : ℕ)
  (filled_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_fraction = 3 / 4 →
  total_volume = 4500 →
  total_bottles = num_boxes * bottles_per_box →
  filled_volume = (total_bottles : ℚ) * (fill_fraction * (12 : ℚ)) →
  12 = 4500 / (total_bottles * fill_fraction) := 
by 
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end bottle_capacity_l153_153624


namespace solution_set_of_inequality_l153_153528

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1 / 3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l153_153528


namespace roadRepairDays_l153_153203

-- Definitions from the conditions
def dailyRepairLength1 : ℕ := 6
def daysToFinish1 : ℕ := 8
def totalLengthOfRoad : ℕ := dailyRepairLength1 * daysToFinish1
def dailyRepairLength2 : ℕ := 8
def daysToFinish2 : ℕ := totalLengthOfRoad / dailyRepairLength2

-- Theorem to be proven
theorem roadRepairDays :
  daysToFinish2 = 6 :=
by
  sorry

end roadRepairDays_l153_153203


namespace remaining_games_win_percent_l153_153948

variable (totalGames : ℕ) (firstGames : ℕ) (firstWinPercent : ℕ) (seasonWinPercent : ℕ)

-- Given conditions expressed as assumptions:
-- The total number of games played in a season is 40
axiom total_games_condition : totalGames = 40
-- The number of first games played is 30
axiom first_games_condition : firstGames = 30
-- The team won 40% of the first 30 games
axiom first_win_percent_condition : firstWinPercent = 40
-- The team won 50% of all its games in the season
axiom season_win_percent_condition : seasonWinPercent = 50

-- We need to prove that the percentage of the remaining games that the team won is 80%
theorem remaining_games_win_percent {remainingWinPercent : ℕ} :
  totalGames = 40 →
  firstGames = 30 →
  firstWinPercent = 40 →
  seasonWinPercent = 50 →
  remainingWinPercent = 80 :=
by
  intros
  sorry

end remaining_games_win_percent_l153_153948


namespace complete_square_solution_l153_153590

theorem complete_square_solution (a b c : ℤ) (h1 : a^2 = 25) (h2 : 10 * b = 30) (h3 : (a * x + b)^2 = 25 * x^2 + 30 * x + c) :
  a + b + c = -58 :=
by
  sorry

end complete_square_solution_l153_153590


namespace math_problem_l153_153307

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0):
  ((2^(2^n) + 1) * (2^(2^m) + 1)) % (m * n) = 0 →
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) :=
by
  sorry

end math_problem_l153_153307


namespace muffin_count_l153_153146

theorem muffin_count (doughnuts cookies muffins : ℕ) (h1 : doughnuts = 50) (h2 : cookies = (3 * doughnuts) / 5) (h3 : muffins = (1 * doughnuts) / 5) : muffins = 10 :=
by sorry

end muffin_count_l153_153146


namespace innings_question_l153_153125

theorem innings_question (n : ℕ) (runs_in_inning : ℕ) (avg_increase : ℕ) (new_avg : ℕ) 
  (h_runs_in_inning : runs_in_inning = 88) 
  (h_avg_increase : avg_increase = 3) 
  (h_new_avg : new_avg = 40)
  (h_eq : 37 * n + runs_in_inning = new_avg * (n + 1)): n + 1 = 17 :=
by
  -- Proof to be filled in here
  sorry

end innings_question_l153_153125


namespace problem_part1_problem_part2_problem_part3_l153_153893

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x^2 - x

theorem problem_part1 :
  (∀ x, 0 < x -> x < 1 / Real.exp 1 -> f (Real.log x + 1) < 0) ∧ 
  (∀ x, x > 1 / Real.exp 1 -> f (Real.log x + 1) > 0) ∧ 
  (f (1 / Real.exp 1) = 1 / Real.exp 1 * Real.log (1 / Real.exp 1)) :=
sorry

theorem problem_part2 (a : ℝ) :
  (∀ x, x > 0 -> f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem problem_part3 (a : ℝ) (m : ℝ) (ha : a = 1/8) :
  (∃ m, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ (3 * f x / (4 * x) + m + g a x = 0))) ↔ 
  (7/8 < m ∧ m < (15/8 - 3/4 * Real.log 3)) :=
sorry

end problem_part1_problem_part2_problem_part3_l153_153893


namespace fraction_addition_l153_153547

-- Definitions from conditions
def frac1 : ℚ := 18 / 42
def frac2 : ℚ := 2 / 9
def simplified_frac1 : ℚ := 3 / 7
def simplified_frac2 : ℚ := frac2
def common_denom_frac1 : ℚ := 27 / 63
def common_denom_frac2 : ℚ := 14 / 63

-- The problem statement to prove
theorem fraction_addition :
  frac1 + frac2 = 41 / 63 := by
  sorry

end fraction_addition_l153_153547


namespace min_value_reciprocal_sum_l153_153346

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 3) :
  (1 / x) + (1 / y) + (1 / z) ≥ 3 :=
sorry

end min_value_reciprocal_sum_l153_153346


namespace hash_four_times_l153_153115

noncomputable def hash (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_four_times (N : ℝ) : hash (hash (hash (hash N))) = 11.8688 :=
  sorry

end hash_four_times_l153_153115


namespace total_pages_read_l153_153026

theorem total_pages_read (J A C D : ℝ) 
  (hJ : J = 20)
  (hA : A = 2 * J + 2)
  (hC : C = J * A - 17)
  (hD : D = (C + J) / 2) :
  J + A + C + D = 1306.5 :=
by
  sorry

end total_pages_read_l153_153026


namespace part_a_part_b_l153_153871

-- Part (a)
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^m > (1 + 1 / (n:ℝ))^n :=
by sorry

-- Part (b)
theorem part_b (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^(m + 1) < (1 + 1 / (n:ℝ))^(n + 1) :=
by sorry

end part_a_part_b_l153_153871


namespace jason_total_expenditure_l153_153405

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end jason_total_expenditure_l153_153405


namespace second_dog_average_miles_l153_153733

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end second_dog_average_miles_l153_153733


namespace combined_tax_rate_33_33_l153_153861

-- Define the necessary conditions
def mork_tax_rate : ℝ := 0.40
def mindy_tax_rate : ℝ := 0.30
def mindy_income_ratio : ℝ := 2.0

-- Main theorem statement
theorem combined_tax_rate_33_33 :
  ∀ (X : ℝ), ((mork_tax_rate * X + mindy_income_ratio * mindy_tax_rate * X) / (X + mindy_income_ratio * X) * 100) = 100 / 3 :=
by
  intro X
  sorry

end combined_tax_rate_33_33_l153_153861


namespace common_integer_solutions_l153_153499

theorem common_integer_solutions
    (y : ℤ)
    (h1 : -4 * y ≥ 2 * y + 10)
    (h2 : -3 * y ≤ 15)
    (h3 : -5 * y ≥ 3 * y + 24)
    (h4 : y ≤ -1) :
  y = -3 ∨ y = -4 ∨ y = -5 :=
by 
  sorry

end common_integer_solutions_l153_153499


namespace average_age_decrease_l153_153004

theorem average_age_decrease :
  let avg_original := 40
  let new_students := 15
  let avg_new_students := 32
  let original_strength := 15
  let total_age_original := original_strength * avg_original
  let total_age_new_students := new_students * avg_new_students
  let total_strength := original_strength + new_students
  let total_age := total_age_original + total_age_new_students
  let avg_new := total_age / total_strength
  avg_original - avg_new = 4 :=
by
  sorry

end average_age_decrease_l153_153004


namespace total_roses_tom_sent_l153_153768

theorem total_roses_tom_sent
  (roses_in_dozen : ℕ := 12)
  (dozens_per_day : ℕ := 2)
  (days_in_week : ℕ := 7) :
  7 * (2 * 12) = 168 := by
  sorry

end total_roses_tom_sent_l153_153768


namespace students_not_in_same_column_or_row_l153_153794

-- Define the positions of student A and student B as conditions
structure Position where
  row : Nat
  col : Nat

-- Student A's position is in the 3rd row and 6th column
def StudentA : Position := {row := 3, col := 6}

-- Student B's position is described in a relative manner in terms of columns and rows
def StudentB : Position := {row := 6, col := 3}

-- Formalize the proof statement
theorem students_not_in_same_column_or_row :
  StudentA.row ≠ StudentB.row ∧ StudentA.col ≠ StudentB.col :=
by {
  sorry
}

end students_not_in_same_column_or_row_l153_153794


namespace set_intersection_nonempty_implies_m_le_neg1_l153_153334

theorem set_intersection_nonempty_implies_m_le_neg1
  (m : ℝ)
  (A : Set ℝ := {x | x^2 - 4 * m * x + 2 * m + 6 = 0})
  (B : Set ℝ := {x | x < 0}) :
  (A ∩ B).Nonempty → m ≤ -1 := 
sorry

end set_intersection_nonempty_implies_m_le_neg1_l153_153334


namespace circle_radius_l153_153653

theorem circle_radius (x y d : ℝ) (h₁ : x = π * r^2) (h₂ : y = 2 * π * r) (h₃ : d = 2 * r) (h₄ : x + y + d = 164 * π) : r = 10 :=
by sorry

end circle_radius_l153_153653


namespace longer_side_of_rectangle_l153_153288

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l153_153288


namespace find_x_for_f_eq_f_inv_l153_153078

def f (x : ℝ) : ℝ := 3 * x - 8

noncomputable def f_inv (x : ℝ) : ℝ := (x + 8) / 3

theorem find_x_for_f_eq_f_inv : ∃ x : ℝ, f x = f_inv x ∧ x = 4 :=
by
  sorry

end find_x_for_f_eq_f_inv_l153_153078


namespace liam_markers_liam_first_markers_over_500_l153_153356

def seq (n : ℕ) : ℕ := 5 * 3^n

theorem liam_markers (n : ℕ) (h1 : seq 0 = 5) (h2 : seq 1 = 10) (h3 : ∀ k < n, 5 * 3^k ≤ 500) : 
  seq n > 500 := by sorry

theorem liam_first_markers_over_500 (h1 : seq 0 = 5) (h2 : seq 1 = 10) :
  ∃ n, seq n > 500 ∧ ∀ k < n, seq k ≤ 500 := by sorry

end liam_markers_liam_first_markers_over_500_l153_153356


namespace algebraic_expression_value_l153_153746

noncomputable def a : ℝ := 2 * Real.sin (Real.pi / 4) + 1
noncomputable def b : ℝ := 2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_value :
  ((a^2 + b^2) / (2 * a * b) - 1) / ((a^2 - b^2) / (a^2 * b + a * b^2)) = 1 :=
by sorry

end algebraic_expression_value_l153_153746


namespace jerry_age_l153_153108

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 16) : J = 11 :=
sorry

end jerry_age_l153_153108


namespace divides_polynomial_l153_153191

theorem divides_polynomial (n : ℕ) (x : ℤ) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(n+2) + (x+1)^(2*n+1)) :=
sorry

end divides_polynomial_l153_153191


namespace prove_A_plus_B_l153_153114

variable (A B : ℝ)

theorem prove_A_plus_B (h : ∀ x : ℝ, x ≠ 2 → (A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2))) : A + B = 9 := by
  sorry

end prove_A_plus_B_l153_153114


namespace highest_score_of_batsman_l153_153451

theorem highest_score_of_batsman
  (avg : ℕ)
  (inn : ℕ)
  (diff_high_low : ℕ)
  (sum_high_low : ℕ)
  (avg_excl : ℕ)
  (inn_excl : ℕ)
  (h_l_avg : avg = 60)
  (h_l_inn : inn = 46)
  (h_l_diff : diff_high_low = 140)
  (h_l_sum : sum_high_low = 208)
  (h_l_avg_excl : avg_excl = 58)
  (h_l_inn_excl : inn_excl = 44) :
  ∃ H L : ℕ, H = 174 :=
by
  sorry

end highest_score_of_batsman_l153_153451


namespace expectation_equality_variance_inequality_l153_153501

noncomputable def X1_expectation : ℚ :=
  2 * (2 / 5 : ℚ)

noncomputable def X1_variance : ℚ :=
  2 * (2 / 5) * (1 - 2 / 5)

noncomputable def P_X2_0 : ℚ :=
  (3 * 2) / (5 * 4)

noncomputable def P_X2_1 : ℚ :=
  (2 * 3) / (5 * 4)

noncomputable def P_X2_2 : ℚ :=
  (2 * 1) / (5 * 4)

noncomputable def X2_expectation : ℚ :=
  0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2

noncomputable def X2_variance : ℚ :=
  P_X2_0 * (0 - X2_expectation)^2 + P_X2_1 * (1 - X2_expectation)^2 + P_X2_2 * (2 - X2_expectation)^2

theorem expectation_equality : X1_expectation = X2_expectation :=
  by sorry

theorem variance_inequality : X1_variance > X2_variance :=
  by sorry

end expectation_equality_variance_inequality_l153_153501


namespace bird_families_left_near_mountain_l153_153512

def total_bird_families : ℕ := 85
def bird_families_flew_to_africa : ℕ := 23
def bird_families_flew_to_asia : ℕ := 37

theorem bird_families_left_near_mountain : total_bird_families - (bird_families_flew_to_africa + bird_families_flew_to_asia) = 25 := by
  sorry

end bird_families_left_near_mountain_l153_153512


namespace infinite_sum_problem_l153_153326

theorem infinite_sum_problem :
  (∑' n : ℕ, if n = 0 then 0 else (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = (1 / 4) := 
by
  sorry

end infinite_sum_problem_l153_153326


namespace intersection_A_B_l153_153845

-- Define sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | ∃ y ∈ A, |y| = x}

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_A_B :
  A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l153_153845


namespace functional_solution_l153_153742

def functional_property (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (x * f y + 2 * x) = x * y + 2 * f x

theorem functional_solution (f : ℝ → ℝ) (h : functional_property f) : f 1 = 0 :=
by sorry

end functional_solution_l153_153742


namespace teacher_age_l153_153194

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (new_avg_with_teacher : ℕ) (num_total : ℕ) 
  (total_age_students : ℕ)
  (h1 : avg_age_students = 10)
  (h2 : num_students = 15)
  (h3 : new_avg_with_teacher = 11)
  (h4 : num_total = 16)
  (h5 : total_age_students = num_students * avg_age_students) :
  num_total * new_avg_with_teacher - total_age_students = 26 :=
by sorry

end teacher_age_l153_153194


namespace part_I_period_part_I_monotonicity_interval_part_II_range_l153_153490

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem part_I_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem part_I_monotonicity_interval (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → f (x + Real.pi) = f x := by
  sorry

theorem part_II_range :
  ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 → f x ∈ Set.Icc (-1) 2 := by
  sorry

end part_I_period_part_I_monotonicity_interval_part_II_range_l153_153490


namespace fraction_strawberries_remaining_l153_153058

theorem fraction_strawberries_remaining 
  (baskets : ℕ)
  (strawberries_per_basket : ℕ)
  (hedgehogs : ℕ)
  (strawberries_per_hedgehog : ℕ)
  (h1 : baskets = 3)
  (h2 : strawberries_per_basket = 900)
  (h3 : hedgehogs = 2)
  (h4 : strawberries_per_hedgehog = 1050) :
  (baskets * strawberries_per_basket - hedgehogs * strawberries_per_hedgehog) / (baskets * strawberries_per_basket) = 2 / 9 :=
by
  sorry

end fraction_strawberries_remaining_l153_153058


namespace smallest_z_value_l153_153637

theorem smallest_z_value :
  ∀ w x y z : ℤ, (∃ k : ℤ, w = 2 * k - 1 ∧ x = 2 * k + 1 ∧ y = 2 * k + 3 ∧ z = 2 * k + 5) ∧
    w^3 + x^3 + y^3 = z^3 →
    z = 9 :=
sorry

end smallest_z_value_l153_153637


namespace real_solutions_l153_153891

theorem real_solutions:
  ∀ x: ℝ, 
    (x ≠ 2) ∧ (x ≠ 4) ∧ 
    ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1)) / 
    ((x - 2) * (x - 4) * (x - 2)) = 1 
    → (x = 2 + Real.sqrt 2) ∨ (x = 2 - Real.sqrt 2) :=
by
  sorry

end real_solutions_l153_153891


namespace factor_polynomials_l153_153381

theorem factor_polynomials :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) :=
by
  sorry

end factor_polynomials_l153_153381


namespace average_speed_joey_round_trip_l153_153884

noncomputable def average_speed_round_trip
  (d : ℝ) (t₁ : ℝ) (r : ℝ) (s₂ : ℝ) : ℝ :=
  2 * d / (t₁ + d / s₂)

-- Lean statement for the proof problem
theorem average_speed_joey_round_trip :
  average_speed_round_trip 6 1 6 12 = 8 := sorry

end average_speed_joey_round_trip_l153_153884


namespace length_of_one_side_l153_153089

-- Definitions according to the conditions
def perimeter (nonagon : Type) : ℝ := 171
def sides (nonagon : Type) : ℕ := 9

-- Math proof problem to prove
theorem length_of_one_side (nonagon : Type) : perimeter nonagon / sides nonagon = 19 :=
by
  sorry

end length_of_one_side_l153_153089


namespace sequence_perfect_square_l153_153722

variable (a : ℕ → ℤ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 1
axiom recurrence : ∀ n ≥ 3, a n = 7 * (a (n - 1)) - (a (n - 2))

theorem sequence_perfect_square (n : ℕ) (hn : n > 0) : ∃ k : ℤ, a n + a (n + 1) + 2 = k * k :=
by
  sorry

end sequence_perfect_square_l153_153722


namespace projection_of_3_neg2_onto_v_l153_153138

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2)
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

def v : ℝ × ℝ := (2, -8)

theorem projection_of_3_neg2_onto_v :
  projection (3, -2) v = (11/17, -44/17) :=
by sorry

end projection_of_3_neg2_onto_v_l153_153138


namespace eval_difference_of_squares_l153_153065

theorem eval_difference_of_squares :
  (81^2 - 49^2 = 4160) :=
by
  -- Since the exact mathematical content is established in a formal context, 
  -- we omit the detailed proof steps.
  sorry

end eval_difference_of_squares_l153_153065


namespace square_of_larger_number_is_1156_l153_153423

theorem square_of_larger_number_is_1156
  (x y : ℕ)
  (h1 : x + y = 60)
  (h2 : x - y = 8) :
  x^2 = 1156 := by
  sorry

end square_of_larger_number_is_1156_l153_153423


namespace range_of_a_l153_153735

noncomputable def f (a x : ℝ) : ℝ := x^3 + x^2 - a * x - 4
noncomputable def f_derivative (a x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

def has_exactly_one_extremum_in_interval (a : ℝ) : Prop :=
  (f_derivative a (-1)) * (f_derivative a 1) < 0

theorem range_of_a (a : ℝ) :
  has_exactly_one_extremum_in_interval a ↔ (1 < a ∧ a < 5) :=
sorry

end range_of_a_l153_153735


namespace correct_quotient_l153_153895

variable (D : ℕ) (q1 q2 : ℕ)
variable (h1 : q1 = 4900) (h2 : D - 1000 = 1200 * q1)

theorem correct_quotient : q2 = D / 2100 → q2 = 2800 :=
by
  sorry

end correct_quotient_l153_153895


namespace union_of_intervals_l153_153030

theorem union_of_intervals :
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  P ∪ Q = {x : ℝ | -2 < x ∧ x < 1} :=
by
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  have h : P ∪ Q = {x : ℝ | -2 < x ∧ x < 1}
  {
     sorry
  }
  exact h

end union_of_intervals_l153_153030


namespace value_of_g_neg2_l153_153005

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem value_of_g_neg2 : g (-2) = -1 := by
  sorry

end value_of_g_neg2_l153_153005


namespace divisor_count_l153_153370

theorem divisor_count (m : ℕ) (h : m = 2^15 * 5^12) :
  let m_squared := m * m
  let num_divisors_m := (15 + 1) * (12 + 1)
  let num_divisors_m_squared := (30 + 1) * (24 + 1)
  let divisors_of_m_squared_less_than_m := (num_divisors_m_squared - 1) / 2
  num_divisors_m_squared - num_divisors_m = 179 :=
by
  subst h
  sorry

end divisor_count_l153_153370


namespace weight_of_second_piece_l153_153612

-- Define the uniform density of the metal.
def density : ℝ := 0.5  -- ounces per square inch

-- Define the side lengths of the two pieces of metal.
def side_length1 : ℝ := 4  -- inches
def side_length2 : ℝ := 7  -- inches

-- Define the weights of the first piece of metal.
def weight1 : ℝ := 8  -- ounces

-- Define the areas of the pieces of metal.
def area1 : ℝ := side_length1^2  -- square inches
def area2 : ℝ := side_length2^2  -- square inches

-- The theorem to prove: the weight of the second piece of metal.
theorem weight_of_second_piece : (area2 * density) = 24.5 :=
by
  sorry

end weight_of_second_piece_l153_153612


namespace turtle_feeding_cost_l153_153430

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l153_153430


namespace time_to_paint_remaining_rooms_l153_153666

-- Definitions for the conditions
def total_rooms : ℕ := 11
def time_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Statement of the problem
theorem time_to_paint_remaining_rooms : 
  total_rooms - painted_rooms = 9 →
  (total_rooms - painted_rooms) * time_per_room = 63 := 
by 
  intros h1
  sorry

end time_to_paint_remaining_rooms_l153_153666


namespace rest_stop_location_l153_153343

theorem rest_stop_location (km_A km_B : ℕ) (fraction : ℚ) (difference := km_B - km_A) 
  (rest_stop_distance := fraction * difference) : 
  km_A = 30 → km_B = 210 → fraction = 4 / 5 → rest_stop_distance + km_A = 174 :=
by 
  intros h1 h2 h3
  sorry

end rest_stop_location_l153_153343


namespace students_suggested_tomatoes_79_l153_153263

theorem students_suggested_tomatoes_79 (T : ℕ)
  (mashed_potatoes : ℕ)
  (h1 : mashed_potatoes = 144)
  (h2 : mashed_potatoes = T + 65) :
  T = 79 :=
by {
  -- Proof steps will go here
  sorry
}

end students_suggested_tomatoes_79_l153_153263


namespace reflect_point_value_l153_153229

theorem reflect_point_value (mx b : ℝ) 
  (start end_ : ℝ × ℝ)
  (Hstart : start = (2, 3))
  (Hend : end_ = (10, 7))
  (Hreflection : ∃ m b: ℝ, (end_.fst, end_.snd) = 
              (2 * ((5 / 2) - (1 / 2) * 3 * m - b), 2 * ((5 / 2) + (1 / 2) * 3)) ∧ m = -2)
  : m + b = 15 :=
sorry

end reflect_point_value_l153_153229


namespace children_too_heavy_l153_153406

def Kelly_weight : ℝ := 34
def Sam_weight : ℝ := 40
def Daisy_weight : ℝ := 28
def Megan_weight := 1.1 * Kelly_weight
def Mike_weight := Megan_weight + 5

def Total_weight := Kelly_weight + Sam_weight + Daisy_weight + Megan_weight + Mike_weight
def Bridge_limit : ℝ := 130

theorem children_too_heavy :
  Total_weight - Bridge_limit = 51.8 :=
by
  sorry

end children_too_heavy_l153_153406


namespace parabola_focus_directrix_distance_l153_153254

theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), 
    y^2 = 8 * x → 
    ∃ p : ℝ, 2 * p = 8 ∧ p = 4 := by
  sorry

end parabola_focus_directrix_distance_l153_153254


namespace no_four_distinct_nat_dividing_pairs_l153_153413

theorem no_four_distinct_nat_dividing_pairs (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∣ (b - c)) (h8 : a ∣ (b - d))
  (h9 : a ∣ (c - d)) (h10 : b ∣ (a - c)) (h11 : b ∣ (a - d)) (h12 : b ∣ (c - d))
  (h13 : c ∣ (a - b)) (h14 : c ∣ (a - d)) (h15 : c ∣ (b - d)) (h16 : d ∣ (a - b))
  (h17 : d ∣ (a - c)) (h18 : d ∣ (b - c)) : False := 
sorry

end no_four_distinct_nat_dividing_pairs_l153_153413


namespace sin_sum_of_acute_l153_153165

open Real

theorem sin_sum_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α + β) ≤ sin α + sin β := 
by
  sorry

end sin_sum_of_acute_l153_153165


namespace square_side_increase_factor_l153_153416

theorem square_side_increase_factor (s k : ℕ) (x new_x : ℕ) (h1 : x = 4 * s) (h2 : new_x = 4 * x) (h3 : new_x = 4 * (k * s)) : k = 4 :=
by
  sorry

end square_side_increase_factor_l153_153416


namespace mike_total_hours_l153_153691

-- Define the number of hours Mike worked each day.
def hours_per_day : ℕ := 3

-- Define the number of days Mike worked.
def days : ℕ := 5

-- Define the total number of hours Mike worked.
def total_hours : ℕ := hours_per_day * days

-- State and prove that the total hours Mike worked is 15.
theorem mike_total_hours : total_hours = 15 := by
  -- Proof goes here
  sorry

end mike_total_hours_l153_153691


namespace math_proof_statement_l153_153357

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  let a := (cos x, sin x)
  let b := (sqrt 2, sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8 / 5) ∧ (π / 4 < x ∧ x < π / 2) ∧ 
  (cos (x - π / 4) = 4 / 5) ∧ (tan (x - π / 4) = 3 / 4) ∧ 
  (sin (2 * x) * (1 - tan x) / (1 + tan x) = -21 / 100)

theorem math_proof_statement (x : ℝ) : proof_problem x := 
by
  unfold proof_problem
  sorry

end math_proof_statement_l153_153357


namespace find_s_over_r_l153_153303

-- Define the function
def f (k : ℝ) : ℝ := 9 * k ^ 2 - 6 * k + 15

-- Define constants
variables (d r s : ℝ)

-- Define the main theorem to be proved
theorem find_s_over_r : 
  (∀ k : ℝ, f k = d * (k + r) ^ 2 + s) → s / r = -42 :=
by
  sorry

end find_s_over_r_l153_153303


namespace patty_weighs_more_l153_153222

variable (R : ℝ) (P_0 : ℝ) (L : ℝ) (P : ℝ) (D : ℝ)

theorem patty_weighs_more :
  (R = 100) →
  (P_0 = 4.5 * R) →
  (L = 235) →
  (P = P_0 - L) →
  (D = P - R) →
  D = 115 := by
  sorry

end patty_weighs_more_l153_153222


namespace tournament_committees_count_l153_153977

-- Definitions corresponding to the conditions
def num_teams : ℕ := 4
def team_size : ℕ := 8
def members_selected_by_winning_team : ℕ := 3
def members_selected_by_other_teams : ℕ := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Counting the number of possible committees
def total_committees : ℕ :=
  let num_ways_winning_team := binom team_size members_selected_by_winning_team
  let num_ways_other_teams := binom team_size members_selected_by_other_teams
  num_teams * num_ways_winning_team * (num_ways_other_teams ^ (num_teams - 1))

-- The statement to be proved
theorem tournament_committees_count : total_committees = 4917248 := by
  sorry

end tournament_committees_count_l153_153977


namespace probability_of_sunglasses_given_caps_l153_153453

theorem probability_of_sunglasses_given_caps
  (s c sc : ℕ) 
  (h₀ : s = 60) 
  (h₁ : c = 40)
  (h₂ : sc = 20)
  (h₃ : sc = 1 / 3 * s) : 
  (sc / c) = 1 / 2 :=
by
  sorry

end probability_of_sunglasses_given_caps_l153_153453


namespace determine_p_l153_153318

def is_tangent (circle_eq : ℝ → ℝ → Prop) (parabola_eq : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq x y ∧ circle_eq x y ∧ x = -p / 2 

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

theorem determine_p (p : ℝ) (hpos : p > 0) :
  (is_tangent circle_eq (parabola_eq p) p) ↔ p = 2 := 
sorry

end determine_p_l153_153318


namespace boat_speed_in_still_water_l153_153467

/-- In one hour, a boat goes 9 km along the stream and 5 km against the stream.
Prove that the speed of the boat in still water is 7 km/hr. -/
theorem boat_speed_in_still_water (B S : ℝ) 
  (h1 : B + S = 9) 
  (h2 : B - S = 5) : 
  B = 7 :=
by
  sorry

end boat_speed_in_still_water_l153_153467


namespace solve_inequality_range_of_m_l153_153516

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := - abs (x + 3) + m

theorem solve_inequality (x a : ℝ) :
  (f x + a - 1 > 0) ↔
  (a = 1 → x ≠ 2) ∧
  (a > 1 → true) ∧
  (a < 1 → x < a + 1 ∨ x > 3 - a) := by sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := by sorry

end solve_inequality_range_of_m_l153_153516


namespace xy_eq_119_imp_sum_values_l153_153033

theorem xy_eq_119_imp_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0)
(hx_lt_30 : x < 30) (hy_lt_30 : y < 30) (h : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := 
sorry

end xy_eq_119_imp_sum_values_l153_153033


namespace cos_of_angle_between_lines_l153_153730

noncomputable def cosTheta (a b : ℝ × ℝ) : ℝ :=
  let dotProduct := a.1 * b.1 + a.2 * b.2
  let magA := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magB := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dotProduct / (magA * magB)

theorem cos_of_angle_between_lines :
  cosTheta (3, 4) (1, 3) = 3 / Real.sqrt 10 :=
by
  sorry

end cos_of_angle_between_lines_l153_153730


namespace problem_solution_l153_153976

theorem problem_solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 :=
by
  sorry

end problem_solution_l153_153976


namespace alex_silver_tokens_count_l153_153481

-- Conditions
def initial_red_tokens := 90
def initial_blue_tokens := 80

def red_exchange (x : ℕ) (y : ℕ) : ℕ := 90 - 3 * x + y
def blue_exchange (x : ℕ) (y : ℕ) : ℕ := 80 + 2 * x - 4 * y

-- Boundaries where exchanges stop
def red_bound (x : ℕ) (y : ℕ) : Prop := red_exchange x y < 3
def blue_bound (x : ℕ) (y : ℕ) : Prop := blue_exchange x y < 4

-- Proof statement
theorem alex_silver_tokens_count (x y : ℕ) :
    red_bound x y → blue_bound x y → (x + y) = 52 :=
    by
    sorry

end alex_silver_tokens_count_l153_153481


namespace tens_digit_of_9_pow_1024_l153_153925

theorem tens_digit_of_9_pow_1024 : 
  (9^1024 % 100) / 10 % 10 = 6 := 
sorry

end tens_digit_of_9_pow_1024_l153_153925


namespace find_divisor_l153_153858

theorem find_divisor (dividend quotient remainder : ℕ) (h₁ : dividend = 176) (h₂ : quotient = 9) (h₃ : remainder = 5) : 
  ∃ divisor, dividend = (divisor * quotient) + remainder ∧ divisor = 19 := by
sorry

end find_divisor_l153_153858


namespace volume_of_Q_3_l153_153535

noncomputable def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 2       -- 1 + 1
  | 2 => 2 + 3 / 16
  | 3 => (2 + 3 / 16) + 3 / 64
  | _ => sorry -- This handles cases n >= 4, which we don't need.

theorem volume_of_Q_3 : Q 3 = 143 / 64 := by
  sorry

end volume_of_Q_3_l153_153535


namespace ivanka_woody_total_months_l153_153187

theorem ivanka_woody_total_months
  (woody_years : ℝ)
  (months_per_year : ℝ)
  (additional_months : ℕ)
  (woody_months : ℝ)
  (ivanka_months : ℝ)
  (total_months : ℝ)
  (h1 : woody_years = 1.5)
  (h2 : months_per_year = 12)
  (h3 : additional_months = 3)
  (h4 : woody_months = woody_years * months_per_year)
  (h5 : ivanka_months = woody_months + additional_months)
  (h6 : total_months = woody_months + ivanka_months) :
  total_months = 39 := by
  sorry

end ivanka_woody_total_months_l153_153187


namespace vinces_bus_ride_length_l153_153199

theorem vinces_bus_ride_length (zachary_ride : ℝ) (vince_extra : ℝ) (vince_ride : ℝ) :
  zachary_ride = 0.5 →
  vince_extra = 0.13 →
  vince_ride = zachary_ride + vince_extra →
  vince_ride = 0.63 :=
by
  intros hz hv he
  -- proof steps here
  sorry

end vinces_bus_ride_length_l153_153199


namespace newspapers_sold_correct_l153_153532

def total_sales : ℝ := 425.0
def magazines_sold : ℝ := 150
def newspapers_sold : ℝ := total_sales - magazines_sold

theorem newspapers_sold_correct : newspapers_sold = 275.0 := by
  sorry

end newspapers_sold_correct_l153_153532


namespace yang_hui_problem_solution_l153_153876

theorem yang_hui_problem_solution (x : ℕ) (h : x * (x - 1) = 650) : x * (x - 1) = 650 :=
by
  exact h

end yang_hui_problem_solution_l153_153876


namespace units_digit_fraction_l153_153580

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end units_digit_fraction_l153_153580


namespace g_composed_g_has_exactly_two_distinct_real_roots_l153_153496

theorem g_composed_g_has_exactly_two_distinct_real_roots (d : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 + 4 * x + d) = 0 ∧ (y^2 + 4 * y + d) = 0) ↔ d = 8 :=
sorry

end g_composed_g_has_exactly_two_distinct_real_roots_l153_153496


namespace pieces_by_first_team_correct_l153_153692

-- Define the number of pieces required.
def total_pieces : ℕ := 500

-- Define the number of pieces made by the second team.
def pieces_by_second_team : ℕ := 131

-- Define the number of pieces made by the third team.
def pieces_by_third_team : ℕ := 180

-- Define the number of pieces made by the first team.
def pieces_by_first_team : ℕ := total_pieces - (pieces_by_second_team + pieces_by_third_team)

-- Statement to prove
theorem pieces_by_first_team_correct : pieces_by_first_team = 189 := 
by 
  -- Proof to be filled in
  sorry

end pieces_by_first_team_correct_l153_153692


namespace distinct_constructions_l153_153721

def num_cube_constructions (white_cubes : Nat) (blue_cubes : Nat) : Nat :=
  if white_cubes = 5 ∧ blue_cubes = 3 then 5 else 0

theorem distinct_constructions : num_cube_constructions 5 3 = 5 :=
by
  sorry

end distinct_constructions_l153_153721


namespace range_of_a_l153_153092

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 + 2 * a * x + 2 < 0) ↔ 0 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l153_153092


namespace g_g_g_g_2_eq_16_l153_153166

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem g_g_g_g_2_eq_16 : g (g (g (g 2))) = 16 := by
  sorry

end g_g_g_g_2_eq_16_l153_153166


namespace determine_sum_of_squares_l153_153436

theorem determine_sum_of_squares
  (x y z : ℝ)
  (h1 : x + y + z = 13)
  (h2 : x * y * z = 72)
  (h3 : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := 
sorry

end determine_sum_of_squares_l153_153436


namespace remainder_when_3_pow_305_div_13_l153_153714

theorem remainder_when_3_pow_305_div_13 :
  (3 ^ 305) % 13 = 9 := 
by {
  sorry
}

end remainder_when_3_pow_305_div_13_l153_153714


namespace value_of_b_l153_153444

theorem value_of_b (b : ℝ) (h : 4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : b = 0.48 :=
by {
  sorry
}

end value_of_b_l153_153444


namespace Jims_apples_fits_into_average_l153_153795

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end Jims_apples_fits_into_average_l153_153795


namespace initial_amount_l153_153847

theorem initial_amount (A : ℝ) (h : (9 / 8) * (9 / 8) * A = 40500) : 
  A = 32000 :=
sorry

end initial_amount_l153_153847


namespace time_to_count_envelopes_l153_153483

theorem time_to_count_envelopes (r : ℕ) : (r / 10 = 1) → (r * 60 / r = 60) ∧ (r * 90 / r = 90) :=
by sorry

end time_to_count_envelopes_l153_153483


namespace minimum_value_l153_153680

noncomputable def min_expression (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a^2 * b^2)

theorem minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 + 3 ∧ min_expression a b ≥ c :=
by
  use 2 * Real.sqrt 2 + 3
  sorry

end minimum_value_l153_153680


namespace triangle_is_obtuse_l153_153182

-- Define the conditions of the problem
def angles (x : ℝ) : Prop :=
  2 * x + 3 * x + 6 * x = 180

def obtuse_angle (x : ℝ) : Prop :=
  6 * x > 90

-- State the theorem
theorem triangle_is_obtuse (x : ℝ) (hx : angles x) : obtuse_angle x :=
sorry

end triangle_is_obtuse_l153_153182


namespace final_price_of_coat_is_correct_l153_153849

-- Define the conditions as constants
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Define the discounted amount calculation
def discount_amount : ℝ := original_price * discount_rate

-- Define the sale price after the discount
def sale_price : ℝ := original_price - discount_amount

-- Define the tax amount calculation on the sale price
def tax_amount : ℝ := sale_price * tax_rate

-- Define the total selling price
def total_selling_price : ℝ := sale_price + tax_amount

-- The theorem that needs to be proven
theorem final_price_of_coat_is_correct : total_selling_price = 96.6 :=
by
  sorry

end final_price_of_coat_is_correct_l153_153849


namespace more_movies_than_books_l153_153618

-- Conditions
def books_read := 15
def movies_watched := 29

-- Question: How many more movies than books have you watched?
theorem more_movies_than_books : (movies_watched - books_read) = 14 := sorry

end more_movies_than_books_l153_153618


namespace sally_quarters_total_l153_153084

/--
Sally originally had 760 quarters. She received 418 more quarters. 
Prove that the total number of quarters Sally has now is 1178.
-/
theorem sally_quarters_total : 
  let original_quarters := 760
  let additional_quarters := 418
  original_quarters + additional_quarters = 1178 :=
by
  let original_quarters := 760
  let additional_quarters := 418
  show original_quarters + additional_quarters = 1178
  sorry

end sally_quarters_total_l153_153084


namespace largest_multiple_of_9_less_than_100_l153_153991

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l153_153991


namespace Lisa_goal_achievable_l153_153785

open Nat

theorem Lisa_goal_achievable :
  ∀ (total_quizzes quizzes_with_A goal_percentage : ℕ),
  total_quizzes = 60 →
  quizzes_with_A = 25 →
  goal_percentage = 85 →
  (quizzes_with_A < goal_percentage * total_quizzes / 100) →
  (∃ remaining_quizzes, goal_percentage * total_quizzes / 100 - quizzes_with_A > remaining_quizzes) :=
by
  intros total_quizzes quizzes_with_A goal_percentage h_total h_A h_goal h_lack
  let needed_quizzes := goal_percentage * total_quizzes / 100
  let remaining_quizzes := total_quizzes - 35
  have h_needed := needed_quizzes - quizzes_with_A
  use remaining_quizzes
  sorry

end Lisa_goal_achievable_l153_153785


namespace find_c_l153_153188

theorem find_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y / 20) + (c * y / 10) = 0.6 * y) : c = 3 :=
by 
  -- Skipping the proof
  sorry

end find_c_l153_153188


namespace expression_value_l153_153016

theorem expression_value (x y : ℤ) (h1 : x = 2) (h2 : y = 5) : 
  (x^4 + 2 * y^2) / 6 = 11 := by
  sorry

end expression_value_l153_153016


namespace minimum_value_l153_153663

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (y : ℝ), y = (c / (a + b)) + (b / c) ∧ y ≥ (Real.sqrt 2) - (1 / 2) :=
sorry

end minimum_value_l153_153663


namespace price_of_basic_computer_l153_153808

theorem price_of_basic_computer (C P : ℝ) 
    (h1 : C + P = 2500) 
    (h2 : P = (1/8) * (C + 500 + P)) :
    C = 2125 :=
by
  sorry

end price_of_basic_computer_l153_153808


namespace avg_weight_of_a_b_c_l153_153497

theorem avg_weight_of_a_b_c (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by
  sorry

end avg_weight_of_a_b_c_l153_153497


namespace average_sales_is_104_l153_153097

-- Define the sales data for the months January to May
def january_sales : ℕ := 150
def february_sales : ℕ := 90
def march_sales : ℕ := 60
def april_sales : ℕ := 140
def may_sales : ℕ := 100
def may_discount : ℕ := 20

-- Define the adjusted sales for May after applying the discount
def adjusted_may_sales : ℕ := may_sales - (may_sales * may_discount / 100)

-- Define the total sales from January to May
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + adjusted_may_sales

-- Define the number of months
def number_of_months : ℕ := 5

-- Define the average sales per month
def average_sales_per_month : ℕ := total_sales / number_of_months

-- Prove that the average sales per month is equal to 104
theorem average_sales_is_104 : average_sales_per_month = 104 := by
  -- Here, we'd write the proof, but we'll leave it as 'sorry' for now
  sorry

end average_sales_is_104_l153_153097


namespace pine_cones_on_roof_l153_153213

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l153_153213


namespace symmetric_point_origin_l153_153420

-- Define the original point P with given coordinates
def P : ℝ × ℝ := (-2, 3)

-- Define the symmetric point P' with respect to the origin
def P'_symmetric (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The theorem states that the symmetric point of P is (2, -3)
theorem symmetric_point_origin : P'_symmetric P = (2, -3) := 
by
  sorry

end symmetric_point_origin_l153_153420


namespace max_a_value_l153_153157

theorem max_a_value : 
  (∀ (x : ℝ), (x - 1) * x - (a - 2) * (a + 1) ≥ 1) → a ≤ 3 / 2 :=
sorry

end max_a_value_l153_153157


namespace quadratic_inequality_condition_l153_153574

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ x ∈ Set.Ioo (-1) 3 := 
sorry

end quadratic_inequality_condition_l153_153574


namespace kit_time_to_ticket_window_l153_153087

theorem kit_time_to_ticket_window 
  (rate : ℝ)
  (remaining_distance : ℝ)
  (yard_to_feet_conv : ℝ)
  (new_rate : rate = 90 / 30)
  (remaining_distance_in_feet : remaining_distance = 100 * yard_to_feet_conv)
  (yard_to_feet_conv_val : yard_to_feet_conv = 3) :
  (remaining_distance / rate = 100) := 
by 
  simp [new_rate, remaining_distance_in_feet, yard_to_feet_conv_val]
  sorry

end kit_time_to_ticket_window_l153_153087


namespace wage_increase_l153_153253

-- Definition: Regression line equation
def regression_line (x : ℝ) : ℝ := 80 * x + 50

-- Theorem: On average, when the labor productivity increases by 1000 yuan, the wage increases by 80 yuan
theorem wage_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 80 :=
by
  sorry

end wage_increase_l153_153253


namespace game_promises_total_hours_l153_153475

open Real

noncomputable def total_gameplay_hours (T : ℝ) : Prop :=
  let boring_gameplay := 0.80 * T
  let enjoyable_gameplay := 0.20 * T
  let expansion_hours := 30
  (enjoyable_gameplay + expansion_hours = 50) → (T = 100)

theorem game_promises_total_hours (T : ℝ) : total_gameplay_hours T :=
  sorry

end game_promises_total_hours_l153_153475


namespace math_vs_english_time_difference_l153_153963

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l153_153963


namespace arithmetic_difference_l153_153310

variable (S : ℕ → ℤ)
variable (n : ℕ)

-- Definitions as conditions from the problem
def is_arithmetic_sum (s : ℕ → ℤ) :=
  ∀ n : ℕ, s n = 2 * n ^ 2 - 5 * n

theorem arithmetic_difference :
  is_arithmetic_sum S →
  S 10 - S 7 = 87 :=
by
  intro h
  sorry

end arithmetic_difference_l153_153310


namespace inequality_inequality_hold_l153_153269

theorem inequality_inequality_hold (k : ℕ) (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_sum : x + y + z = 1) : 
  (x ^ (k + 2) / (x ^ (k + 1) + y ^ k + z ^ k) 
  + y ^ (k + 2) / (y ^ (k + 1) + z ^ k + x ^ k) 
  + z ^ (k + 2) / (z ^ (k + 1) + x ^ k + y ^ k)) 
  ≥ (1 / 7) :=
sorry

end inequality_inequality_hold_l153_153269


namespace probability_point_closer_to_7_than_0_l153_153363

noncomputable def segment_length (a b : ℝ) : ℝ := b - a
noncomputable def closer_segment (a c b : ℝ) : ℝ := segment_length c b

theorem probability_point_closer_to_7_than_0 :
  let a := 0
  let b := 10
  let c := 7
  let midpoint := (a + c) / 2
  let total_length := b - a
  let closer_length := segment_length midpoint b
  (closer_length / total_length) = 0.7 :=
by
  sorry

end probability_point_closer_to_7_than_0_l153_153363


namespace trigonometric_identity_l153_153729

theorem trigonometric_identity
    (α φ : ℝ) :
    4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = Real.cos (2 * α) :=
by
  sorry

end trigonometric_identity_l153_153729


namespace probability_window_opens_correct_l153_153723

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l153_153723


namespace no_real_quadruples_solutions_l153_153242

theorem no_real_quadruples_solutions :
  ¬ ∃ (a b c d : ℝ),
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := 
sorry

end no_real_quadruples_solutions_l153_153242


namespace sequence_periodic_a2014_l153_153543

theorem sequence_periodic_a2014 (a : ℕ → ℚ) 
  (h1 : a 1 = -1/4) 
  (h2 : ∀ n > 1, a n = 1 - (1 / (a (n - 1)))) : 
  a 2014 = -1/4 :=
sorry

end sequence_periodic_a2014_l153_153543


namespace find_fraction_l153_153479

theorem find_fraction (f : ℝ) (h₁ : f * 50.0 - 4 = 6) : f = 0.2 :=
by
  sorry

end find_fraction_l153_153479


namespace relationship_between_a_b_c_l153_153572

noncomputable def a := 33
noncomputable def b := 5 * 6^1 + 2 * 6^0
noncomputable def c := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l153_153572


namespace infinite_primes_l153_153592

theorem infinite_primes : ∀ (p : ℕ), Prime p → ¬ (∃ q : ℕ, Prime q ∧ q > p) := sorry

end infinite_primes_l153_153592


namespace contribution_required_l153_153241

-- Definitions corresponding to the problem statement
def total_amount : ℝ := 2000
def number_of_friends : ℝ := 7
def your_contribution_factor : ℝ := 2

-- Prove that the amount each friend needs to raise is approximately 222.22
theorem contribution_required (x : ℝ) 
  (h : 9 * x = total_amount) :
  x = 2000 / 9 := 
  by sorry

end contribution_required_l153_153241


namespace min_value_frac_sum_l153_153265

-- Define the main problem
theorem min_value_frac_sum (a b : ℝ) (h1 : 2 * a + 3 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ x : ℝ, (x = 25) ∧ ∀ y, (y = (2 / a + 3 / b)) → y ≥ x :=
sorry

end min_value_frac_sum_l153_153265


namespace div_recurring_decimal_l153_153539

def recurringDecimalToFraction (q : ℚ) (h : q = 36/99) : ℚ := by
  sorry

theorem div_recurring_decimal : 12 / recurringDecimalToFraction 0.36 sorry = 33 :=
by
  sorry

end div_recurring_decimal_l153_153539


namespace mixed_number_sum_l153_153907

theorem mixed_number_sum : (2 + (1 / 10 : ℝ)) + (3 + (11 / 100 : ℝ)) = 5.21 := by
  sorry

end mixed_number_sum_l153_153907


namespace ending_time_proof_l153_153939

def starting_time_seconds : ℕ := (1 * 3600) + (57 * 60) + 58
def glow_interval : ℕ := 13
def total_glow_count : ℕ := 382
def total_glow_duration : ℕ := total_glow_count * glow_interval
def ending_time_seconds : ℕ := starting_time_seconds + total_glow_duration

theorem ending_time_proof : 
ending_time_seconds = (3 * 3600) + (14 * 60) + 4 := by
  -- Proof starts here
  sorry

end ending_time_proof_l153_153939


namespace cookies_guests_l153_153270

theorem cookies_guests (cc_cookies : ℕ) (oc_cookies : ℕ) (sc_cookies : ℕ) (cc_per_guest : ℚ) (oc_per_guest : ℚ) (sc_per_guest : ℕ)
    (cc_total : cc_cookies = 45) (oc_total : oc_cookies = 62) (sc_total : sc_cookies = 38) (cc_ratio : cc_per_guest = 1.5)
    (oc_ratio : oc_per_guest = 2.25) (sc_ratio : sc_per_guest = 1) :
    (cc_cookies / cc_per_guest) ≥ 0 ∧ (oc_cookies / oc_per_guest) ≥ 0 ∧ (sc_cookies / sc_per_guest) ≥ 0 → 
    Nat.floor (oc_cookies / oc_per_guest) = 27 :=
by
  sorry

end cookies_guests_l153_153270


namespace find_digit_sum_l153_153331

theorem find_digit_sum (A B X D C Y : ℕ) :
  (A * 100 + B * 10 + X) + (C * 100 + D * 10 + Y) = Y * 1010 + X * 1010 →
  A + D = 6 :=
by
  sorry

end find_digit_sum_l153_153331


namespace negation_of_p_is_neg_p_l153_153873

def p (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

def neg_p (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0

theorem negation_of_p_is_neg_p (f : ℝ → ℝ) : ¬ p f ↔ neg_p f :=
by
  sorry -- Proof of this theorem

end negation_of_p_is_neg_p_l153_153873


namespace problem_statement_l153_153281

open Nat

theorem problem_statement (k : ℕ) (hk : k > 0) : 
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * (factorial (k*n) / factorial n) ≤ (factorial (k*n) / factorial n))
  ↔ ∃ a : ℕ, k = 2^a := 
sorry

end problem_statement_l153_153281


namespace pow_mod_eq_l153_153117

theorem pow_mod_eq (h : 101 % 100 = 1) : (101 ^ 50) % 100 = 1 :=
by
  -- Proof omitted
  sorry

end pow_mod_eq_l153_153117


namespace inradius_of_triangle_l153_153296

theorem inradius_of_triangle (P A : ℝ) (hP : P = 40) (hA : A = 50) : 
  ∃ r : ℝ, r = 2.5 ∧ A = r * (P / 2) :=
by
  sorry

end inradius_of_triangle_l153_153296


namespace number_of_candies_bought_on_Tuesday_l153_153487

theorem number_of_candies_bought_on_Tuesday (T : ℕ) 
  (thursday_candies : ℕ := 5) 
  (friday_candies : ℕ := 2) 
  (candies_left : ℕ := 4) 
  (candies_eaten : ℕ := 6) 
  (total_initial_candies : T + thursday_candies + friday_candies = candies_left + candies_eaten) 
  : T = 3 := by
  sorry

end number_of_candies_bought_on_Tuesday_l153_153487


namespace average_increase_l153_153095

-- Definitions
def runs_11 := 90
def avg_11 := 40

-- Conditions
def total_runs_before (A : ℕ) := A * 10
def total_runs_after (runs_11 : ℕ) (total_runs_before : ℕ) := total_runs_before + runs_11
def increased_average (avg_11 : ℕ) (avg_before : ℕ) := avg_11 = avg_before + 5

-- Theorem stating the equivalent proof problem
theorem average_increase
  (A : ℕ)
  (H1 : total_runs_after runs_11 (total_runs_before A) = 40 * 11)
  (H2 : avg_11 = 40) :
  increased_average 40 A := 
sorry

end average_increase_l153_153095


namespace nancy_other_albums_count_l153_153049

-- Definitions based on the given conditions
def total_pictures : ℕ := 51
def pics_in_first_album : ℕ := 11
def pics_per_other_album : ℕ := 5

-- Theorem to prove the question's answer
theorem nancy_other_albums_count : 
  (total_pictures - pics_in_first_album) / pics_per_other_album = 8 := by
  sorry

end nancy_other_albums_count_l153_153049


namespace solve_inequality_l153_153954

theorem solve_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end solve_inequality_l153_153954


namespace arithmetic_sequence_a101_eq_52_l153_153776

theorem arithmetic_sequence_a101_eq_52 (a : ℕ → ℝ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = 1 / 2) :
  a 101 = 52 :=
by
  sorry

end arithmetic_sequence_a101_eq_52_l153_153776


namespace total_legs_arms_proof_l153_153957

/-
There are 4 birds, each with 2 legs.
There are 6 dogs, each with 4 legs.
There are 5 snakes, each with no legs.
There are 2 spiders, each with 8 legs.
There are 3 horses, each with 4 legs.
There are 7 rabbits, each with 4 legs.
There are 2 octopuses, each with 8 arms.
There are 8 ants, each with 6 legs.
There is 1 unique creature with 12 legs.
We need to prove that the total number of legs and arms is 164.
-/

def total_legs_arms : Nat := 
  (4 * 2) + (6 * 4) + (5 * 0) + (2 * 8) + (3 * 4) + (7 * 4) + (2 * 8) + (8 * 6) + (1 * 12)

theorem total_legs_arms_proof : total_legs_arms = 164 := by
  sorry

end total_legs_arms_proof_l153_153957


namespace walking_speed_proof_l153_153432

-- Definitions based on the problem's conditions
def rest_time_per_period : ℕ := 5
def distance_per_rest : ℕ := 10
def total_distance : ℕ := 50
def total_time : ℕ := 320

-- The man's walking speed
def walking_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The main statement to be proved
theorem walking_speed_proof : 
  walking_speed total_distance ((total_time - ((total_distance / distance_per_rest) * rest_time_per_period)) / 60) = 10 := 
by
  sorry

end walking_speed_proof_l153_153432


namespace distinct_real_numbers_satisfying_system_l153_153766

theorem distinct_real_numbers_satisfying_system :
  ∃! (x y z : ℝ),
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x^2 + y^2 = -x + 3 * y + z) ∧
  (y^2 + z^2 = x + 3 * y - z) ∧
  (x^2 + z^2 = 2 * x + 2 * y - z) ∧
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
sorry

end distinct_real_numbers_satisfying_system_l153_153766


namespace minimum_value_l153_153434

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
  37.5 ≤ (9 / x + 25 / y + 49 / z) :=
sorry

end minimum_value_l153_153434


namespace find_range_of_a_l153_153815

def p (a : ℝ) : Prop := 
  a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)

def q (a : ℝ) : Prop := 
  a^2 - 2 * a - 3 < 0

theorem find_range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end find_range_of_a_l153_153815


namespace abs_add_lt_abs_sub_l153_153414

-- Define the conditions
variables {a b : ℝ} (h1 : a * b < 0)

-- Prove the statement
theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| := sorry

end abs_add_lt_abs_sub_l153_153414


namespace rachel_biology_homework_pages_l153_153034

-- Declare the known quantities
def math_pages : ℕ := 8
def total_math_biology_pages : ℕ := 11

-- Define biology_pages
def biology_pages : ℕ := total_math_biology_pages - math_pages

-- Assert the main theorem
theorem rachel_biology_homework_pages : biology_pages = 3 :=
by 
  -- Proof is omitted as instructed
  sorry

end rachel_biology_homework_pages_l153_153034


namespace f_log₂_20_l153_153736

noncomputable def f (x : ℝ) : ℝ := sorry -- This is a placeholder for the function f.

lemma f_neg (x : ℝ) : f (-x) = -f (x) := sorry
lemma f_shift (x : ℝ) : f (x + 1) = f (1 - x) := sorry
lemma f_special (x : ℝ) (hx : -1 < x ∧ x < 0) : f (x) = 2^x + 6 / 5 := sorry

theorem f_log₂_20 : f (Real.log 20 / Real.log 2) = -2 := by
  -- Proof details would go here.
  sorry

end f_log₂_20_l153_153736


namespace fraction_value_l153_153937

theorem fraction_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 :=
by
  sorry

end fraction_value_l153_153937


namespace train_boxcars_capacity_l153_153476

theorem train_boxcars_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_capacity := 4000
  let blue_capacity := black_capacity * 2
  let red_capacity := blue_capacity * 3
  (black_boxcars * black_capacity) + (blue_boxcars * blue_capacity) + (red_boxcars * red_capacity) = 132000 := by
  sorry

end train_boxcars_capacity_l153_153476


namespace friend_pays_correct_percentage_l153_153850

theorem friend_pays_correct_percentage (adoption_fee : ℝ) (james_payment : ℝ) (friend_payment : ℝ) 
  (h1 : adoption_fee = 200) 
  (h2 : james_payment = 150)
  (h3 : friend_payment = adoption_fee - james_payment) : 
  (friend_payment / adoption_fee) * 100 = 25 :=
by
  sorry

end friend_pays_correct_percentage_l153_153850


namespace solve_trig_problem_l153_153728

theorem solve_trig_problem (α : ℝ) (h : Real.tan α = 1 / 3) :
  (Real.cos α)^2 - 2 * (Real.sin α)^2 / (Real.cos α)^2 = 7 / 9 := 
sorry

end solve_trig_problem_l153_153728


namespace area_curve_is_correct_l153_153332

-- Define the initial conditions
structure Rectangle :=
  (vertices : Fin 4 → ℝ × ℝ)
  (point : ℝ × ℝ)

-- Define the rotation transformation
def rotate_clockwise_90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := point
  (cx + (py - cy), cy - (px - cx))

-- Given initial rectangle and the point to track
def initial_rectangle : Rectangle :=
  { vertices := ![(0, 0), (2, 0), (0, 3), (2, 3)],
    point := (1, 1) }

-- Perform the four specified rotations
def rotated_points : List (ℝ × ℝ) :=
  let r1 := rotate_clockwise_90 (2, 0) initial_rectangle.point
  let r2 := rotate_clockwise_90 (5, 0) r1
  let r3 := rotate_clockwise_90 (7, 0) r2
  let r4 := rotate_clockwise_90 (10, 0) r3
  [initial_rectangle.point, r1, r2, r3, r4]

-- Calculate the area below the curve and above the x-axis
noncomputable def area_below_curve : ℝ :=
  6 + (7 * Real.pi / 2)

-- The theorem statement
theorem area_curve_is_correct : 
  area_below_curve = 6 + (7 * Real.pi / 2) :=
  by trivial

end area_curve_is_correct_l153_153332


namespace maximum_value_of_expression_l153_153701

noncomputable def max_value (x y z w : ℝ) : ℝ := 2 * x + 3 * y + 5 * z - 4 * w

theorem maximum_value_of_expression 
  (x y z w : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 + 16 * w^2 = 4) : 
  max_value x y z w ≤ 6 * Real.sqrt 6 :=
sorry

end maximum_value_of_expression_l153_153701


namespace fifteenth_entry_is_21_l153_153227

def r_9 (n : ℕ) : ℕ := n % 9

def condition (n : ℕ) : Prop := (7 * n) % 9 ≤ 5

def sequence_elements (k : ℕ) : ℕ := 
  if k = 0 then 0
  else if k = 1 then 2
  else if k = 2 then 3
  else if k = 3 then 4
  else if k = 4 then 7
  else if k = 5 then 8
  else if k = 6 then 9
  else if k = 7 then 11
  else if k = 8 then 12
  else if k = 9 then 13
  else if k = 10 then 16
  else if k = 11 then 17
  else if k = 12 then 18
  else if k = 13 then 20
  else if k = 14 then 21
  else 0 -- for the sake of ensuring completeness

theorem fifteenth_entry_is_21 : sequence_elements 14 = 21 :=
by
  -- Mathematical proof omitted.
  sorry

end fifteenth_entry_is_21_l153_153227


namespace binom_10_4_l153_153753

theorem binom_10_4 : Nat.choose 10 4 = 210 := 
by sorry

end binom_10_4_l153_153753


namespace opposite_number_subtraction_l153_153441

variable (a b : ℝ)

theorem opposite_number_subtraction : -(a - b) = b - a := 
sorry

end opposite_number_subtraction_l153_153441


namespace intersection_M_N_l153_153333

-- Define the set M
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the condition for set N
def N : Set ℤ := {x | x + 2 ≥ x^2}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l153_153333


namespace parallel_vectors_implies_value_of_t_l153_153811

theorem parallel_vectors_implies_value_of_t (t : ℝ) :
  let a := (1, t)
  let b := (t, 9)
  (1 * 9 - t^2 = 0) → (t = 3 ∨ t = -3) := 
by sorry

end parallel_vectors_implies_value_of_t_l153_153811


namespace gcd_solution_l153_153400

noncomputable def gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : ℤ :=
  Int.gcd (4 * b^2 + 35 * b + 72) (3 * b + 8)

theorem gcd_solution (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : gcd_problem b h = 2 :=
by
  sorry

end gcd_solution_l153_153400


namespace model2_best_fit_l153_153989
-- Import necessary tools from Mathlib

-- Define the coefficients of determination for the four models
def R2_model1 : ℝ := 0.75
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.28
def R2_model4 : ℝ := 0.55

-- Define the best fitting model
def best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ) : Prop :=
  R2_2 > R2_1 ∧ R2_2 > R2_3 ∧ R2_2 > R2_4

-- Statement to prove
theorem model2_best_fit : best_fitting_model R2_model1 R2_model2 R2_model3 R2_model4 :=
  by
  -- Proof goes here
  sorry

end model2_best_fit_l153_153989


namespace iron_ii_sulfate_moles_l153_153628

/-- Given the balanced chemical equation for the reaction between iron (Fe) and sulfuric acid (H2SO4)
    to form Iron (II) sulfate (FeSO4) and hydrogen gas (H2) and the 1:1 molar ratio between iron and
    sulfuric acid, determine the number of moles of Iron (II) sulfate formed when 3 moles of Iron and
    2 moles of Sulfuric acid are combined. This is a limiting reactant problem with the final 
    product being 2 moles of Iron (II) sulfate (FeSO4). -/
theorem iron_ii_sulfate_moles (Fe moles_H2SO4 : Nat) (reaction_ratio : Nat) (FeSO4 moles_formed : Nat) :
  Fe = 3 → moles_H2SO4 = 2 → reaction_ratio = 1 → moles_formed = 2 :=
by
  intros hFe hH2SO4 hRatio
  apply sorry

end iron_ii_sulfate_moles_l153_153628


namespace Courtney_total_marbles_l153_153751

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l153_153751


namespace sale_in_fifth_month_l153_153337

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 avg_sale num_months total_sales known_sales_five_months sale5: ℕ) :
  sale1 = 6400 →
  sale2 = 7000 →
  sale3 = 6800 →
  sale4 = 7200 →
  sale6 = 5100 →
  avg_sale = 6500 →
  num_months = 6 →
  total_sales = avg_sale * num_months →
  known_sales_five_months = sale1 + sale2 + sale3 + sale4 + sale6 →
  sale5 = total_sales - known_sales_five_months →
  sale5 = 6500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end sale_in_fifth_month_l153_153337


namespace smallest_positive_integer_l153_153339

theorem smallest_positive_integer (n : ℕ) : 3 * n ≡ 568 [MOD 34] → n = 18 := 
sorry

end smallest_positive_integer_l153_153339


namespace division_correct_l153_153312

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end division_correct_l153_153312


namespace find_r4_l153_153075

-- Definitions of the problem conditions
variable (r1 r2 r3 r4 r5 r6 r7 : ℝ)
-- Given radius of the smallest circle
axiom smallest_circle : r1 = 6
-- Given radius of the largest circle
axiom largest_circle : r7 = 24
-- Given that radii of circles form a geometric sequence
axiom geometric_sequence : r2 = r1 * (r7 / r1)^(1/6) ∧ 
                            r3 = r1 * (r7 / r1)^(2/6) ∧
                            r4 = r1 * (r7 / r1)^(3/6) ∧
                            r5 = r1 * (r7 / r1)^(4/6) ∧
                            r6 = r1 * (r7 / r1)^(5/6)

-- Statement to prove
theorem find_r4 : r4 = 12 :=
by
  sorry

end find_r4_l153_153075


namespace find_original_numbers_l153_153952

theorem find_original_numbers (x y : ℕ) (hx : x + y = 2022) 
  (hy : (x - 5) / 10 + 10 * y + 1 = 2252) : x = 1815 ∧ y = 207 :=
by sorry

end find_original_numbers_l153_153952


namespace minimize_quadratic_expression_l153_153509

noncomputable def quadratic_expression (b : ℝ) : ℝ :=
  (1 / 3) * b^2 + 7 * b - 6

theorem minimize_quadratic_expression : ∃ b : ℝ, quadratic_expression b = -10.5 :=
  sorry

end minimize_quadratic_expression_l153_153509


namespace percentage_small_bottles_sold_l153_153560

theorem percentage_small_bottles_sold :
  ∀ (x : ℕ), (6000 - (x * 60)) + 8500 = 13780 → x = 12 :=
by
  intro x h
  sorry

end percentage_small_bottles_sold_l153_153560


namespace ones_digit_of_73_pow_351_l153_153096

-- Definition of the problem in Lean 4
theorem ones_digit_of_73_pow_351 : (73 ^ 351) % 10 = 7 := by
  sorry

end ones_digit_of_73_pow_351_l153_153096


namespace find_a_n_plus_b_n_l153_153990

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else if n = 2 then 3 
  else sorry -- Placeholder for proper recursive implementation

noncomputable def b (n : ℕ) : ℕ := 
  if n = 1 then 5
  else sorry -- Placeholder for proper recursive implementation

theorem find_a_n_plus_b_n (n : ℕ) (i j k l : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : b 1 = 5) 
  (h4 : i + j = k + l) (h5 : a i + b j = a k + b l) : a n + b n = 4 * n + 2 := 
by
  sorry

end find_a_n_plus_b_n_l153_153990


namespace major_axis_length_l153_153505

def length_of_major_axis 
  (tangent_x : ℝ) (f1 : ℝ × ℝ) (f2 : ℝ × ℝ) : ℝ :=
  sorry

theorem major_axis_length 
  (hx_tangent : (4, 0) = (4, 0)) 
  (foci : (4, 2 + 2 * Real.sqrt 2) = (4, 2 + 2 * Real.sqrt 2) ∧ 
         (4, 2 - 2 * Real.sqrt 2) = (4, 2 - 2 * Real.sqrt 2)) :
  length_of_major_axis 4 
  (4, 2 + 2 * Real.sqrt 2) (4, 2 - 2 * Real.sqrt 2) = 4 :=
sorry

end major_axis_length_l153_153505


namespace find_number_l153_153024

noncomputable def number := 115.2 / 0.32

theorem find_number : number = 360 := 
by
  sorry

end find_number_l153_153024


namespace pesticide_residue_comparison_l153_153694

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem pesticide_residue_comparison (a : ℝ) (ha : a > 0) :
  (f a = (1 / (1 + a^2))) ∧ 
  (if a = 2 * Real.sqrt 2 then f a = 16 / (4 + a^2)^2 else 
   if a > 2 * Real.sqrt 2 then f a > 16 / (4 + a^2)^2 else 
   f a < 16 / (4 + a^2)^2) ∧
  (f 0 = 1) ∧ 
  (f 1 = 1 / 2) := sorry

end pesticide_residue_comparison_l153_153694


namespace difference_of_squares_l153_153629

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 :=
by
  sorry

end difference_of_squares_l153_153629


namespace tan_equals_three_l153_153605

variable (α : ℝ)

theorem tan_equals_three : 
  (Real.tan α = 3) → (1 / (Real.sin α * Real.sin α + 2 * Real.sin α * Real.cos α) = 2 / 3) :=
by
  intro h
  sorry

end tan_equals_three_l153_153605


namespace find_c_value_l153_153715

theorem find_c_value (A B C : ℝ) (S1_area S2_area : ℝ) (b : ℝ) :
  S1_area = 40 * b + 1 →
  S2_area = 40 * b →
  ∃ c, AC + CB = c ∧ c = 462 :=
by
  intro hS1 hS2
  sorry

end find_c_value_l153_153715


namespace jason_steps_is_8_l153_153098

-- Definition of the problem conditions
def nancy_steps (jason_steps : ℕ) := 3 * jason_steps -- Nancy steps 3 times as often as Jason

def together_steps (jason_steps nancy_steps : ℕ) := jason_steps + nancy_steps -- Total steps

-- Lean statement of the problem to prove
theorem jason_steps_is_8 (J : ℕ) (h₁ : together_steps J (nancy_steps J) = 32) : J = 8 :=
sorry

end jason_steps_is_8_l153_153098


namespace store_profit_l153_153724

theorem store_profit {C : ℝ} (h₁ : C > 0) : 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  SPF - C = 0.20 * C := 
by 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  sorry

end store_profit_l153_153724


namespace curves_intersect_at_three_points_l153_153177

theorem curves_intersect_at_three_points (b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = b^2 ∧ y = 2 * x^2 - b) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₁^2 + y₁^2 = b^2) ∧ (x₂^2 + y₂^2 = b^2) ∧ (x₃^2 + y₃^2 = b^2) ∧
    (y₁ = 2 * x₁^2 - b) ∧ (y₂ = 2 * x₂^2 - b) ∧ (y₃ = 2 * x₃^2 - b)) ↔ b > 1 / 4 :=
by
  sorry

end curves_intersect_at_three_points_l153_153177


namespace erdos_ginzburg_ziv_2047_l153_153818

open Finset

theorem erdos_ginzburg_ziv_2047 (s : Finset ℕ) (h : s.card = 2047) : 
  ∃ t ⊆ s, t.card = 1024 ∧ (t.sum id) % 1024 = 0 :=
sorry

end erdos_ginzburg_ziv_2047_l153_153818


namespace even_function_l153_153435

noncomputable def f : ℝ → ℝ :=
sorry

theorem even_function (f : ℝ → ℝ) (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = x - 1) : f (1/2) = -3/2 :=
sorry

end even_function_l153_153435


namespace imo_inequality_l153_153298

variable {a b c : ℝ}

theorem imo_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : (a + b) * (b + c) * (c + a) = 1) :
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ (1 / 2) := 
sorry

end imo_inequality_l153_153298


namespace maximal_n_is_k_minus_1_l153_153121

section
variable (k : ℕ) (n : ℕ)
variable (cards : Finset ℕ)
variable (red : List ℕ) (blue : List (List ℕ))

-- Conditions
axiom h_k_pos : k > 1
axiom h_card_count : cards = Finset.range (2 * n + 1)
axiom h_initial_red : red = (List.range' 1 (2 * n)).reverse
axiom h_initial_blue : blue.length = k

-- Question translated to a goal
theorem maximal_n_is_k_minus_1 (h : ∀ (n' : ℕ), n' ≤ (k - 1)) : n = k - 1 :=
sorry
end

end maximal_n_is_k_minus_1_l153_153121


namespace even_function_exists_l153_153770

def f (x m : ℝ) : ℝ := x^2 + m * x

theorem even_function_exists : ∃ m : ℝ, ∀ x : ℝ, f x m = f (-x) m :=
by
  use 0
  intros x
  unfold f
  simp

end even_function_exists_l153_153770


namespace hyperbola_eccentricity_l153_153007

theorem hyperbola_eccentricity (C : Type) (a b c e : ℝ)
  (h_asymptotes : ∀ x : ℝ, (∃ y : ℝ, y = x ∨ y = -x)) :
  a = b ∧ c = Real.sqrt (a^2 + b^2) ∧ e = c / a → e = Real.sqrt 2 := 
by
  sorry

end hyperbola_eccentricity_l153_153007


namespace coconut_grove_problem_l153_153282

variable (x : ℝ)

-- Conditions
def trees_yield_40_nuts_per_year : ℝ := 40 * (x + 2)
def trees_yield_120_nuts_per_year : ℝ := 120 * x
def trees_yield_180_nuts_per_year : ℝ := 180 * (x - 2)
def average_yield_per_tree_per_year : ℝ := 100

-- Problem Statement
theorem coconut_grove_problem
  (yield_40_trees : trees_yield_40_nuts_per_year x = 40 * (x + 2))
  (yield_120_trees : trees_yield_120_nuts_per_year x = 120 * x)
  (yield_180_trees : trees_yield_180_nuts_per_year x = 180 * (x - 2))
  (average_yield : average_yield_per_tree_per_year = 100) :
  x = 7 :=
by
  sorry

end coconut_grove_problem_l153_153282


namespace coating_profit_l153_153502

theorem coating_profit (x y : ℝ) (h1 : 0.6 * x + 0.9 * (150 - x) ≤ 120)
  (h2 : 0.7 * x + 0.4 * (150 - x) ≤ 90) :
  (50 ≤ x ∧ x ≤ 100) → (y = -50 * x + 75000) → (x = 50 → y = 72500) :=
by
  intros hx hy hx_val
  sorry

end coating_profit_l153_153502


namespace net_profit_calculation_l153_153640

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

end net_profit_calculation_l153_153640


namespace exists_prime_q_l153_153361

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) (h2 : 2 < p) : 
  ∃ q : ℕ, Nat.Prime q ∧ q < p ∧ ¬ (p ^ 2 ∣ q ^ (p - 1) - 1) := 
sorry

end exists_prime_q_l153_153361


namespace circumradius_of_triangle_l153_153364

theorem circumradius_of_triangle (a b S : ℝ) (A : a = 2) (B : b = 3) (Area : S = 3 * Real.sqrt 15 / 4)
  (median_cond : ∃ c m, m = (a^2 + b^2 - c^2) / (2*a*b) ∧ m < c / 2) :
  ∃ R, R = 8 / Real.sqrt 15 :=
by
  sorry

end circumradius_of_triangle_l153_153364


namespace ratio_proof_l153_153555

variable (a b c d : ℚ)

theorem ratio_proof 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
  sorry

end ratio_proof_l153_153555


namespace abs_expression_eq_6500_l153_153698

def given_expression (x : ℝ) : ℝ := 
  abs (abs x - x - abs x + 500) - x

theorem abs_expression_eq_6500 (x : ℝ) (h : x = -3000) : given_expression x = 6500 := by
  sorry

end abs_expression_eq_6500_l153_153698


namespace athlete_distance_l153_153523

theorem athlete_distance (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) (d : ℝ)
  (h1 : t = 24)
  (h2 : v_kmh = 30.000000000000004)
  (h3 : v_ms = v_kmh * 1000 / 3600)
  (h4 : d = v_ms * t) :
  d = 200 := 
sorry

end athlete_distance_l153_153523


namespace positive_integer_solution_of_inequality_l153_153008

theorem positive_integer_solution_of_inequality :
  {x : ℕ // 0 < x ∧ x < 2} → x = 1 :=
by
  sorry

end positive_integer_solution_of_inequality_l153_153008


namespace practice_problems_total_l153_153258

theorem practice_problems_total :
  let marvin_yesterday := 40
  let marvin_today := 3 * marvin_yesterday
  let arvin_yesterday := 2 * marvin_yesterday
  let arvin_today := 2 * marvin_today
  let kevin_yesterday := 30
  let kevin_today := kevin_yesterday + 10
  let total_problems := (marvin_yesterday + marvin_today) + (arvin_yesterday + arvin_today) + (kevin_yesterday + kevin_today)
  total_problems = 550 :=
by
  sorry

end practice_problems_total_l153_153258


namespace expression_value_l153_153888

theorem expression_value :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by
  sorry

end expression_value_l153_153888


namespace max_ab_l153_153796

noncomputable def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

theorem max_ab (a b : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ x, f a x ≥ - (1/2) * x^2 + a * x + b) : 
  ab ≤ ((Real.exp 1) / 2) :=
sorry

end max_ab_l153_153796


namespace tan_ratio_l153_153507

theorem tan_ratio (a b : ℝ) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : Real.sin (a + b) = 5/8) (h2 : Real.sin (a - b) = 3/8) :
  (Real.tan a) / (Real.tan b) = 4 :=
by
  sorry

end tan_ratio_l153_153507


namespace mathematicians_correctness_l153_153682

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  (3 / 8 < 2 / 5 ∧ 2 / 5 < 17 / 40) → false :=
by 
  sorry

end mathematicians_correctness_l153_153682


namespace functional_equation_solution_l153_153832

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y, f (f (f x)) + f (f y) = f y + x) → (∀ x, f x = x) :=
by
  intros f h x
  -- Proof goes here
  sorry

end functional_equation_solution_l153_153832


namespace jackson_star_fish_count_l153_153151

def total_starfish_per_spiral_shell (hermit_crabs : ℕ) (shells_per_crab : ℕ) (total_souvenirs : ℕ) : ℕ :=
  (total_souvenirs - (hermit_crabs + hermit_crabs * shells_per_crab)) / (hermit_crabs * shells_per_crab)

theorem jackson_star_fish_count :
  total_starfish_per_spiral_shell 45 3 450 = 2 :=
by
  -- The proof will be filled in here
  sorry

end jackson_star_fish_count_l153_153151


namespace sum_first_n_natural_numbers_l153_153817

theorem sum_first_n_natural_numbers (n : ℕ) (h : (n * (n + 1)) / 2 = 1035) : n = 46 :=
sorry

end sum_first_n_natural_numbers_l153_153817


namespace johnny_marbles_l153_153656

noncomputable def choose_at_least_one_red : ℕ :=
  let total_marbles := 8
  let red_marbles := 1
  let other_marbles := 7
  let choose_4_out_of_8 := Nat.choose total_marbles 4
  let choose_3_out_of_7 := Nat.choose other_marbles 3
  let choose_4_with_at_least_1_red := choose_3_out_of_7
  choose_4_with_at_least_1_red

theorem johnny_marbles : choose_at_least_one_red = 35 :=
by
  -- Sorry, proof is omitted
  sorry

end johnny_marbles_l153_153656


namespace fifth_friend_paid_13_l153_153447

noncomputable def fifth_friend_payment (a b c d e : ℝ) : Prop :=
a = (1/3) * (b + c + d + e) ∧
b = (1/4) * (a + c + d + e) ∧
c = (1/5) * (a + b + d + e) ∧
a + b + c + d + e = 120 ∧
e = 13

theorem fifth_friend_paid_13 : 
  ∃ (a b c d e : ℝ), fifth_friend_payment a b c d e := 
sorry

end fifth_friend_paid_13_l153_153447


namespace truncated_cone_volume_l153_153077

theorem truncated_cone_volume 
  (V_initial : ℝ)
  (r_ratio : ℝ)
  (V_final : ℝ)
  (r_ratio_eq : r_ratio = 1 / 2)
  (V_initial_eq : V_initial = 1) :
  V_final = 7 / 8 :=
  sorry

end truncated_cone_volume_l153_153077


namespace tangent_line_at_one_extreme_points_and_inequality_l153_153846

noncomputable def f (x a : ℝ) := x^2 - 2*x + a * Real.log x

-- Question 1: Tangent Line
theorem tangent_line_at_one (x a : ℝ) (h_a : a = 2) (hx_pos : x > 0) :
    2*x - Real.log x - (2*x - Real.log 1 - 1) = 0 := by
  sorry

-- Question 2: Extreme Points and Inequality
theorem extreme_points_and_inequality (a x1 x2 : ℝ) (h1 : 2*x1^2 - 2*x1 + a = 0)
    (h2 : 2*x2^2 - 2*x2 + a = 0) (hx12 : x1 < x2) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
    0 < a ∧ a < 1/2 ∧ (f x1 a) / x2 > -3/2 - Real.log 2 := by
  sorry

end tangent_line_at_one_extreme_points_and_inequality_l153_153846


namespace complementary_angles_positive_difference_l153_153993

theorem complementary_angles_positive_difference
  (x : ℝ)
  (h1 : 3 * x + x = 90): 
  |(3 * x) - x| = 45 := 
by
  -- Proof would go here (details skipped)
  sorry

end complementary_angles_positive_difference_l153_153993


namespace minimum_small_droppers_l153_153183

/-
Given:
1. A total volume to be filled: V = 265 milliliters.
2. Small droppers can hold: s = 19 milliliters each.
3. No large droppers are used.

Prove:
The minimum number of small droppers required to fill the container completely is 14.
-/

theorem minimum_small_droppers (V s: ℕ) (hV: V = 265) (hs: s = 19) : 
  ∃ n: ℕ, n = 14 ∧ n * s ≥ V ∧ (n - 1) * s < V :=
by
  sorry  -- proof to be provided

end minimum_small_droppers_l153_153183


namespace game_A_probability_greater_than_B_l153_153421

-- Defining the probabilities of heads and tails for the biased coin
def prob_heads : ℚ := 2 / 3
def prob_tails : ℚ := 1 / 3

-- Defining the winning probabilities for Game A
def prob_winning_A : ℚ := (prob_heads^4) + (prob_tails^4)

-- Defining the winning probabilities for Game B
def prob_winning_B : ℚ := (prob_heads^3 * prob_tails) + (prob_tails^3 * prob_heads)

-- The statement we want to prove
theorem game_A_probability_greater_than_B : prob_winning_A - prob_winning_B = 7 / 81 := by
  sorry

end game_A_probability_greater_than_B_l153_153421


namespace hyperbola_condition_l153_153511

theorem hyperbola_condition (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1 → (-1 < k ∧ k < 1)) ∧ 
  ((-1 < k ∧ k < 1) → ∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1) :=
sorry

end hyperbola_condition_l153_153511


namespace age_sum_l153_153280

variables (A B C : ℕ)

theorem age_sum (h1 : A = 20 + B + C) (h2 : A^2 = 2000 + (B + C)^2) : A + B + C = 100 :=
by
  -- Assume the subsequent proof follows here
  sorry

end age_sum_l153_153280


namespace correct_calculation_l153_153371

theorem correct_calculation :
  ∃ (a : ℤ), (a^2 + a^2 = 2 * a^2) ∧ 
  (¬(3*a + 4*(a : ℤ) = 12*a*(a : ℤ))) ∧ 
  (¬((a*(a : ℤ)^2)^3 = a*(a : ℤ)^6)) ∧ 
  (¬((a + 3)^2 = a^2 + 9)) :=
by
  sorry

end correct_calculation_l153_153371


namespace parallel_lines_equal_slopes_l153_153581

theorem parallel_lines_equal_slopes (a : ℝ) :
  (∀ x y, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = -7 + a) →
  a = 3 := sorry

end parallel_lines_equal_slopes_l153_153581


namespace part1_part2_l153_153586

def partsProcessedA : ℕ → ℕ
| 0 => 10
| (n + 1) => if n = 0 then 8 else partsProcessedA n - 2

def partsProcessedB : ℕ → ℕ
| 0 => 8
| (n + 1) => if n = 0 then 7 else partsProcessedB n - 1

def partsProcessedLineB_A (n : ℕ) := 7 * n
def partsProcessedLineB_B (n : ℕ) := 8 * n

def maxSetsIn14Days : ℕ := 
  let aLineA := 2 * (10 + 8 + 6) + (10 + 8)
  let aLineB := 2 * (8 + 7 + 6) + (8 + 8)
  min aLineA aLineB

theorem part1 :
  partsProcessedA 0 + partsProcessedA 1 + partsProcessedA 2 = 24 := 
by sorry

theorem part2 :
  maxSetsIn14Days = 106 :=
by sorry

end part1_part2_l153_153586


namespace distinct_terms_in_expansion_l153_153792

theorem distinct_terms_in_expansion :
  let n1 := 2 -- number of terms in (x + y)
  let n2 := 3 -- number of terms in (a + b + c)
  let n3 := 3 -- number of terms in (d + e + f)
  (n1 * n2 * n3) = 18 :=
by
  sorry

end distinct_terms_in_expansion_l153_153792


namespace brinley_animals_count_l153_153230

theorem brinley_animals_count :
  let snakes := 100
  let arctic_foxes := 80
  let leopards := 20
  let bee_eaters := 10 * ((snakes / 2) + (2 * leopards))
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 481340 := by
  sorry

end brinley_animals_count_l153_153230


namespace candy_bars_given_to_sister_first_time_l153_153854

theorem candy_bars_given_to_sister_first_time (x : ℕ) :
  (7 - x) + 30 - 4 * x = 22 → x = 3 :=
by
  sorry

end candy_bars_given_to_sister_first_time_l153_153854


namespace average_weight_increase_l153_153043

theorem average_weight_increase (n : ℕ) (w_old w_new : ℝ) (h1 : n = 9) (h2 : w_old = 65) (h3 : w_new = 87.5) :
  (w_new - w_old) / n = 2.5 :=
by
  rw [h1, h2, h3]
  norm_num

end average_weight_increase_l153_153043


namespace car_speed_l153_153576

theorem car_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 495) (h_time : time = 5) : 
  distance / time = 99 :=
by
  rw [h_distance, h_time]
  norm_num

end car_speed_l153_153576


namespace sale_coupon_discount_l153_153807

theorem sale_coupon_discount
  (original_price : ℝ)
  (sale_price : ℝ)
  (price_after_coupon : ℝ)
  (h1 : sale_price = 0.5 * original_price)
  (h2 : price_after_coupon = 0.8 * sale_price) :
  (original_price - price_after_coupon) / original_price * 100 = 60 := by
sorry

end sale_coupon_discount_l153_153807


namespace geometric_series_problem_l153_153515

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_problem
  (c d : ℝ)
  (h : geometric_series_sum (c/d) (1/d) = 6) :
  geometric_series_sum (c/(c + 2 * d)) (1/(c + 2 * d)) = 3 / 4 := by
  sorry

end geometric_series_problem_l153_153515


namespace gray_region_area_l153_153041

theorem gray_region_area
  (center_C : ℝ × ℝ) (r_C : ℝ)
  (center_D : ℝ × ℝ) (r_D : ℝ)
  (C_center : center_C = (3, 5)) (C_radius : r_C = 5)
  (D_center : center_D = (13, 5)) (D_radius : r_D = 5) :
  let rect_area := 10 * 5
  let semi_circle_area := 12.5 * π
  rect_area - 2 * semi_circle_area = 50 - 25 * π := 
by 
  sorry

end gray_region_area_l153_153041


namespace x_equals_y_l153_153825

-- Conditions
def x := 2 * 20212021 * 1011 * 202320232023
def y := 43 * 47 * 20232023 * 202220222022

-- Proof statement
theorem x_equals_y : x = y := sorry

end x_equals_y_l153_153825


namespace total_miles_l153_153309

-- Define the variables and equations as given in the conditions
variables (a b c d e : ℝ)
axiom h1 : a + b = 36
axiom h2 : b + c + d = 45
axiom h3 : c + d + e = 45
axiom h4 : a + c + e = 38

-- The conjecture we aim to prove
theorem total_miles : a + b + c + d + e = 83 :=
sorry

end total_miles_l153_153309


namespace find_b_l153_153546

theorem find_b (a b c : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 1 < c):
  (∀ N : ℝ, N ≠ 1 → (N^(3/a) * N^(2/(ab)) * N^(1/(abc)) = N^(39/48))) → b = 4 :=
  by
  sorry

end find_b_l153_153546


namespace girls_ran_miles_l153_153257

def boys_laps : ℕ := 34
def extra_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6
def girls_laps : ℕ := boys_laps + extra_laps

theorem girls_ran_miles : girls_laps * lap_distance = 9 := 
by 
  sorry

end girls_ran_miles_l153_153257


namespace length_of_BC_is_7_l153_153841

noncomputable def triangle_length_BC (a b c : ℝ) (A : ℝ) (S : ℝ) (P : ℝ) : Prop :=
  (P = a + b + c) ∧ (P = 20) ∧ (S = 1 / 2 * b * c * Real.sin A) ∧ (S = 10 * Real.sqrt 3) ∧ (A = Real.pi / 3) ∧ (b * c = 20)

theorem length_of_BC_is_7 : ∃ a b c, triangle_length_BC a b c (Real.pi / 3) (10 * Real.sqrt 3) 20 ∧ a = 7 := 
by
  -- proof omitted
  sorry

end length_of_BC_is_7_l153_153841


namespace tank_full_capacity_l153_153904

theorem tank_full_capacity (C : ℝ) (H1 : 0.4 * C + 36 = 0.7 * C) : C = 120 :=
by
  sorry

end tank_full_capacity_l153_153904


namespace Amy_work_hours_l153_153582

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end Amy_work_hours_l153_153582


namespace find_prime_p_l153_153813

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_prime_p (p : ℕ) (hp : p.Prime) (hsquare : isPerfectSquare (5^p + 12^p)) : p = 2 := 
sorry

end find_prime_p_l153_153813


namespace cost_of_fencing_field_l153_153437

def ratio (a b : ℕ) : Prop := ∃ k : ℕ, (b = k * a)

def assume_fields : Prop :=
  ∃ (x : ℚ), (ratio 3 4) ∧ (3 * 4 * x^2 = 9408) ∧ (0.25 > 0)

theorem cost_of_fencing_field :
  assume_fields → 98 = 98 := by
  sorry

end cost_of_fencing_field_l153_153437


namespace find_initial_apples_l153_153761

theorem find_initial_apples (A : ℤ)
  (h1 : 6 * ((A / 8) + 8 - 30) = 12) :
  A = 192 :=
sorry

end find_initial_apples_l153_153761


namespace minimum_cubes_required_l153_153867

def cube_snaps_visible (n : Nat) : Prop := 
  ∀ (cubes : Fin n → Fin 6 → Bool),
    (∀ i, (cubes i 0 ∧ cubes i 1) ∨ ¬(cubes i 0 ∨ cubes i 1)) → 
    ∃ i j, (i ≠ j) ∧ 
            (cubes i 0 ↔ ¬ cubes j 0) ∧ 
            (cubes i 1 ↔ ¬ cubes j 1)

theorem minimum_cubes_required : 
  ∃ n, cube_snaps_visible n ∧ n = 4 := 
  by sorry

end minimum_cubes_required_l153_153867


namespace unique_solution_mod_37_system_l153_153892

theorem unique_solution_mod_37_system :
  ∃! (a b c d : ℤ), 
  (a^2 + b * c ≡ a [ZMOD 37]) ∧
  (b * (a + d) ≡ b [ZMOD 37]) ∧
  (c * (a + d) ≡ c [ZMOD 37]) ∧
  (b * c + d^2 ≡ d [ZMOD 37]) ∧
  (a * d - b * c ≡ 1 [ZMOD 37]) :=
sorry

end unique_solution_mod_37_system_l153_153892


namespace abc_zero_l153_153906

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) 
  : a * b * c = 0 := by
  sorry

end abc_zero_l153_153906


namespace committee_selection_l153_153267

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection :
  let seniors := 10
  let members := 30
  let non_seniors := members - seniors
  let choices := binom seniors 2 * binom non_seniors 3 +
                 binom seniors 3 * binom non_seniors 2 +
                 binom seniors 4 * binom non_seniors 1 +
                 binom seniors 5
  choices = 78552 :=
by
  sorry

end committee_selection_l153_153267


namespace kelseys_sister_age_in_2021_l153_153886

-- Definitions based on given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3

-- Prove that Kelsey's older sister is 50 years old in 2021
theorem kelseys_sister_age_in_2021 : (2021 - sister_birth_year) = 50 :=
by
  -- Add proof here
  sorry

end kelseys_sister_age_in_2021_l153_153886


namespace clock_820_angle_is_130_degrees_l153_153304

def angle_at_8_20 : ℝ :=
  let degrees_per_hour := 30.0
  let degrees_per_minute_hour_hand := 0.5
  let num_hour_sections := 4.0
  let minutes := 20.0
  let hour_angle := num_hour_sections * degrees_per_hour
  let minute_addition := minutes * degrees_per_minute_hour_hand
  hour_angle + minute_addition

theorem clock_820_angle_is_130_degrees :
  angle_at_8_20 = 130 :=
by
  sorry

end clock_820_angle_is_130_degrees_l153_153304


namespace sum_of_solutions_l153_153080

theorem sum_of_solutions (y1 y2 : ℝ) (h1 : y1 + 16 / y1 = 12) (h2 : y2 + 16 / y2 = 12) : 
  y1 + y2 = 12 :=
by
  sorry

end sum_of_solutions_l153_153080


namespace last_digit_p_adic_l153_153382

theorem last_digit_p_adic (a : ℤ) (p : ℕ) (hp : Nat.Prime p) (h_last_digit_nonzero : a % p ≠ 0) : (a ^ (p - 1) - 1) % p = 0 :=
by
  sorry

end last_digit_p_adic_l153_153382


namespace women_at_each_table_l153_153536

/-- A waiter had 5 tables, each with 3 men and some women, and a total of 40 customers.
    Prove that there are 5 women at each table. -/
theorem women_at_each_table (W : ℕ) (total_customers : ℕ) (men_per_table : ℕ) (tables : ℕ)
  (h1 : total_customers = 40) (h2 : men_per_table = 3) (h3 : tables = 5) :
  (W * tables + men_per_table * tables = total_customers) → (W = 5) :=
by
  sorry

end women_at_each_table_l153_153536


namespace football_defense_stats_l153_153689

/-- Given:
1. Team 1 has an average of 1.5 goals conceded per match.
2. Team 1 has a standard deviation of 1.1 for the total number of goals conceded throughout the year.
3. Team 2 has an average of 2.1 goals conceded per match.
4. Team 2 has a standard deviation of 0.4 for the total number of goals conceded throughout the year.

Prove:
There are exactly 3 correct statements out of the 4 listed statements. -/
theorem football_defense_stats
  (avg_goals_team1 : ℝ := 1.5)
  (std_dev_team1 : ℝ := 1.1)
  (avg_goals_team2 : ℝ := 2.1)
  (std_dev_team2 : ℝ := 0.4) :
  ∃ correct_statements : ℕ, correct_statements = 3 := 
by
  sorry

end football_defense_stats_l153_153689


namespace ratio_monkeys_snakes_l153_153788

def parrots : ℕ := 8
def snakes : ℕ := 3 * parrots
def elephants : ℕ := (parrots + snakes) / 2
def zebras : ℕ := elephants - 3
def monkeys : ℕ := zebras + 35

theorem ratio_monkeys_snakes : (monkeys : ℕ) / (snakes : ℕ) = 2 / 1 :=
by
  sorry

end ratio_monkeys_snakes_l153_153788


namespace integer_solution_count_l153_153056

theorem integer_solution_count :
  (∃ x : ℤ, -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24) ↔
  (∃ n : ℕ, n = 3) :=
by
  sorry

end integer_solution_count_l153_153056


namespace increment_in_radius_l153_153045

theorem increment_in_radius (C1 C2 : ℝ) (hC1 : C1 = 50) (hC2 : C2 = 60) : 
  ((C2 / (2 * Real.pi)) - (C1 / (2 * Real.pi)) = (5 / Real.pi)) :=
by
  sorry

end increment_in_radius_l153_153045


namespace cat_finishes_food_on_next_monday_l153_153758

noncomputable def cat_food_consumption_per_day : ℚ := (1 / 4) + (1 / 6)

theorem cat_finishes_food_on_next_monday :
  ∃ n : ℕ, n = 8 ∧ (n * cat_food_consumption_per_day > 8) := sorry

end cat_finishes_food_on_next_monday_l153_153758


namespace imaginary_part_of_1_minus_2i_l153_153231

def i := Complex.I

theorem imaginary_part_of_1_minus_2i : Complex.im (1 - 2 * i) = -2 :=
by
  sorry

end imaginary_part_of_1_minus_2i_l153_153231


namespace overall_class_average_proof_l153_153819

noncomputable def group_1_weighted_average := (0.40 * 80) + (0.60 * 80)
noncomputable def group_2_weighted_average := (0.30 * 60) + (0.70 * 60)
noncomputable def group_3_weighted_average := (0.50 * 40) + (0.50 * 40)
noncomputable def group_4_weighted_average := (0.20 * 50) + (0.80 * 50)

noncomputable def overall_class_average := (0.20 * group_1_weighted_average) + 
                                           (0.50 * group_2_weighted_average) + 
                                           (0.25 * group_3_weighted_average) + 
                                           (0.05 * group_4_weighted_average)

theorem overall_class_average_proof : overall_class_average = 58.5 :=
by 
  unfold overall_class_average
  unfold group_1_weighted_average
  unfold group_2_weighted_average
  unfold group_3_weighted_average
  unfold group_4_weighted_average
  -- now perform the arithmetic calculations
  sorry

end overall_class_average_proof_l153_153819


namespace arithmetic_mean_fraction_l153_153494

theorem arithmetic_mean_fraction :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  let c := (9 : ℚ) / 10
  (1 / 3) * (a + b + c) = 149 / 180 :=
by 
  sorry

end arithmetic_mean_fraction_l153_153494


namespace sqrt_mul_sqrt_l153_153131

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l153_153131


namespace proof_problem_l153_153855

theorem proof_problem (x y : ℝ) (h1 : 3 * x ^ 2 - 5 * x + 4 * y + 6 = 0) 
                      (h2 : 3 * x - 2 * y + 1 = 0) : 
                      4 * y ^ 2 - 2 * y + 24 = 0 := 
by 
  sorry

end proof_problem_l153_153855


namespace exists_three_with_gcd_d_l153_153424

theorem exists_three_with_gcd_d (n : ℕ) (nums : Fin n.succ → ℕ) (d : ℕ)
  (h1 : n ≥ 2)  -- because n+1 (number of elements nums : Fin n.succ) ≥ 3 given that n ≥ 2
  (h2 : ∀ i, nums i > 0) 
  (h3 : ∀ i, nums i ≤ 100) 
  (h4 : Nat.gcd (nums 0) (Nat.gcd (nums 1) (nums 2)) = d) : 
  ∃ i j k : Fin n.succ, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ Nat.gcd (nums i) (Nat.gcd (nums j) (nums k)) = d :=
by
  sorry

end exists_three_with_gcd_d_l153_153424


namespace find_C_l153_153342

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : 
  C = 10 := 
by
  sorry

end find_C_l153_153342


namespace a1d1_a2d2_a3d3_eq_neg1_l153_153175

theorem a1d1_a2d2_a3d3_eq_neg1 (a1 a2 a3 d1 d2 d3 : ℝ) (h : ∀ x : ℝ, 
  x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 + 1)) : 
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end a1d1_a2d2_a3d3_eq_neg1_l153_153175


namespace zhou_yu_age_eq_l153_153266

-- Define the conditions based on the problem statement
variable (x : ℕ)  -- x represents the tens digit of Zhou Yu's age

-- Condition: The tens digit is three less than the units digit
def units_digit := x + 3

-- Define Zhou Yu's age based on the tens and units digits
def zhou_yu_age := 10 * x + units_digit x

-- Prove the correct equation representing Zhou Yu's lifespan
theorem zhou_yu_age_eq : zhou_yu_age x = (units_digit x) ^ 2 :=
by sorry

end zhou_yu_age_eq_l153_153266


namespace fence_cost_l153_153538

noncomputable def price_per_foot (total_cost : ℝ) (perimeter : ℝ) : ℝ :=
  total_cost / perimeter

theorem fence_cost (area : ℝ) (total_cost : ℝ) (price : ℝ) :
  area = 289 → total_cost = 4012 → price = price_per_foot 4012 (4 * (Real.sqrt 289)) → price = 59 :=
by
  intros h_area h_cost h_price
  sorry

end fence_cost_l153_153538


namespace find_cos2α_l153_153081

noncomputable def cos2α (tanα : ℚ) : ℚ :=
  (1 - tanα^2) / (1 + tanα^2)

theorem find_cos2α (h : tanα = (3 / 4)) : cos2α tanα = (7 / 25) :=
by
  rw [cos2α, h]
  -- here the simplification steps would be performed
  sorry

end find_cos2α_l153_153081


namespace gcf_45_75_90_l153_153833

-- Definitions as conditions
def number1 : Nat := 45
def number2 : Nat := 75
def number3 : Nat := 90

def factors_45 : Nat × Nat := (3, 2) -- represents 3^2 * 5^1 {prime factor 3, prime factor 5}
def factors_75 : Nat × Nat := (5, 1) -- represents 3^1 * 5^2 {prime factor 3, prime factor 5}
def factors_90 : Nat × Nat := (3, 2) -- represents 2^1 * 3^2 * 5^1 {prime factor 3, prime factor 5}

-- Theorems to be proved
theorem gcf_45_75_90 : Nat.gcd (Nat.gcd number1 number2) number3 = 15 :=
by {
  -- This is here as placeholder for actual proof
  sorry
}

end gcf_45_75_90_l153_153833


namespace arithmetic_example_l153_153291

theorem arithmetic_example : (2468 * 629) / (1234 * 37) = 34 :=
by
  sorry

end arithmetic_example_l153_153291


namespace minimum_value_l153_153019

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + (1 / y)) * (x + (1 / y) - 1024) +
  (y + (1 / x)) * (y + (1 / x) - 1024) ≥ -524288 :=
by sorry

end minimum_value_l153_153019


namespace x_value_for_divisibility_l153_153031

theorem x_value_for_divisibility (x : ℕ) (h1 : x = 0 ∨ x = 5) (h2 : (8 * 10 + x) % 4 = 0) : x = 0 :=
by
  sorry

end x_value_for_divisibility_l153_153031


namespace square_root_ratio_area_l153_153754

theorem square_root_ratio_area (side_length_C side_length_D : ℕ) (hC : side_length_C = 45) (hD : side_length_D = 60) : 
  Real.sqrt ((side_length_C^2 : ℝ) / (side_length_D^2 : ℝ)) = 3 / 4 :=
by
  rw [hC, hD]
  sorry

end square_root_ratio_area_l153_153754


namespace rectangle_difference_length_width_l153_153268

theorem rectangle_difference_length_width (x y p d : ℝ) (h1 : x + y = p / 2) (h2 : x^2 + y^2 = d^2) (h3 : x > y) : 
  x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 := sorry

end rectangle_difference_length_width_l153_153268


namespace count_monomials_l153_153284

def isMonomial (expr : String) : Bool :=
  match expr with
  | "m+n" => false
  | "2x^2y" => true
  | "1/x" => true
  | "-5" => true
  | "a" => true
  | _ => false

theorem count_monomials :
  let expressions := ["m+n", "2x^2y", "1/x", "-5", "a"]
  (expressions.filter isMonomial).length = 3 :=
by { sorry }

end count_monomials_l153_153284


namespace delta_eq_bullet_l153_153717

-- Definitions of all variables involved
variables (Δ Θ σ : ℕ)

-- Condition 1: Δ + Δ = σ
def cond1 : Prop := Δ + Δ = σ

-- Condition 2: σ + Δ = Θ
def cond2 : Prop := σ + Δ = Θ

-- Condition 3: Θ = 3Δ
def cond3 : Prop := Θ = 3 * Δ

-- The proof problem
theorem delta_eq_bullet (Δ Θ σ : ℕ) (h1 : Δ + Δ = σ) (h2 : σ + Δ = Θ) (h3 : Θ = 3 * Δ) : 3 * Δ = Θ :=
by
  -- Simply restate the conditions and ensure the proof
  sorry

end delta_eq_bullet_l153_153717


namespace quadrilateral_angle_B_l153_153552

/-- In quadrilateral ABCD,
given that angle A + angle C = 150 degrees,
prove that angle B = 105 degrees. -/
theorem quadrilateral_angle_B (A C : ℝ) (B : ℝ) (h1 : A + C = 150) (h2 : A + B = 180) : B = 105 :=
by
  sorry

end quadrilateral_angle_B_l153_153552


namespace boys_in_choir_l153_153193

theorem boys_in_choir
  (h1 : 20 + 2 * 20 + 16 + b = 88)
  : b = 12 :=
by
  sorry

end boys_in_choir_l153_153193


namespace sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l153_153527

noncomputable def volume_of_spheres (V : ℝ) : ℝ :=
  V * (27 / 26)

noncomputable def volume_of_tetrahedrons (V : ℝ) : ℝ :=
  (3 * V * Real.sqrt 3) / (13 * Real.pi)

theorem sum_volumes_of_spheres (V : ℝ) : 
  (∑' n : ℕ, (V * (1/27)^n)) = volume_of_spheres V :=
sorry

theorem sum_volumes_of_tetrahedrons (V : ℝ) (r : ℝ) : 
  (∑' n : ℕ, (8/9 / Real.sqrt 3 * (r^3) * (1/27)^n * (1/26))) = volume_of_tetrahedrons V :=
sorry

end sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l153_153527


namespace number_of_sides_of_polygon_l153_153170

theorem number_of_sides_of_polygon (n : ℕ) (h : 3 * (n * (n - 3) / 2) - n = 21) : n = 6 :=
by sorry

end number_of_sides_of_polygon_l153_153170


namespace tiger_distance_traveled_l153_153152

theorem tiger_distance_traveled :
  let distance1 := 25 * 1
  let distance2 := 35 * 2
  let distance3 := 20 * 1.5
  let distance4 := 10 * 1
  let distance5 := 50 * 0.5
  distance1 + distance2 + distance3 + distance4 + distance5 = 160 := by
sorry

end tiger_distance_traveled_l153_153152


namespace problem_l153_153428

theorem problem (a b : ℝ) (h1 : |a - 2| + (b + 1)^2 = 0) : a - b = 3 := by
  sorry

end problem_l153_153428


namespace number_of_possible_m_values_l153_153639

theorem number_of_possible_m_values :
  ∃ m_set : Finset ℤ, (∀ x1 x2 : ℤ, x1 * x2 = 40 → (x1 + x2) ∈ m_set) ∧ m_set.card = 8 :=
sorry

end number_of_possible_m_values_l153_153639


namespace line_y_intercept_l153_153196

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end line_y_intercept_l153_153196


namespace find_n_l153_153681

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 14) : n ≡ 14567 [MOD 15] → n = 2 := 
by
  sorry

end find_n_l153_153681


namespace calculate_value_l153_153650

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

variable (f : ℝ → ℝ)

axiom h : odd_function f
axiom h1 : increasing_on_interval f 3 7
axiom h2 : f 3 = -1
axiom h3 : f 6 = 8

theorem calculate_value : 2 * f (-6) + f (-3) = -15 := by
  sorry

end calculate_value_l153_153650


namespace rational_solution_l153_153110

theorem rational_solution (m n : ℤ) (h : a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2)) : 
  ∃ a : ℚ, a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2) :=
by {
  sorry
}

end rational_solution_l153_153110


namespace range_of_m_l153_153607

noncomputable def set_M (m : ℝ) : Set ℝ := {x | x < m}
noncomputable def set_N : Set ℝ := {y | ∃ (x : ℝ), y = Real.log x / Real.log 2 - 1 ∧ 4 ≤ x}

theorem range_of_m (m : ℝ) : set_M m ∩ set_N = ∅ → m < 1 
:= by
  sorry

end range_of_m_l153_153607


namespace maximum_value_l153_153551

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y^2 + x^3 * y^3 + x^2 * y^4 + x * y^5

theorem maximum_value (x y : ℝ) (h : x + y = 5) : maxValue x y h ≤ 625 / 4 :=
sorry

end maximum_value_l153_153551


namespace same_cost_for_same_sheets_l153_153112

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end same_cost_for_same_sheets_l153_153112


namespace candy_distribution_impossible_l153_153506

theorem candy_distribution_impossible :
  ∀ (candies : Fin 6 → ℕ),
  (candies 0 = 0 ∧ candies 1 = 1 ∧ candies 2 = 0 ∧ candies 3 = 0 ∧ candies 4 = 0 ∧ candies 5 = 1) →
  (∀ t, ∃ i, (i < 6) ∧ candies ((i+t)%6) = candies ((i+t+1)%6)) →
  ∃ (i : Fin 6), candies i ≠ candies ((i + 1) % 6) :=
by
  sorry

end candy_distribution_impossible_l153_153506


namespace findCostPrices_l153_153036

def costPriceOfApple (sp_a : ℝ) (cp_a : ℝ) : Prop :=
  sp_a = (5 / 6) * cp_a

def costPriceOfOrange (sp_o : ℝ) (cp_o : ℝ) : Prop :=
  sp_o = (3 / 4) * cp_o

def costPriceOfBanana (sp_b : ℝ) (cp_b : ℝ) : Prop :=
  sp_b = (9 / 8) * cp_b

theorem findCostPrices (sp_a sp_o sp_b : ℝ) (cp_a cp_o cp_b : ℝ) :
  costPriceOfApple sp_a cp_a → 
  costPriceOfOrange sp_o cp_o → 
  costPriceOfBanana sp_b cp_b → 
  sp_a = 20 → sp_o = 15 → sp_b = 6 → 
  cp_a = 24 ∧ cp_o = 20 ∧ cp_b = 16 / 3 :=
by 
  intro h1 h2 h3 sp_a_eq sp_o_eq sp_b_eq
  -- proof goes here
  sorry

end findCostPrices_l153_153036


namespace compute_expression_l153_153379

theorem compute_expression :
  20 * ((144 / 3) + (36 / 6) + (16 / 32) + 2) = 1130 := sorry

end compute_expression_l153_153379


namespace derivative_of_odd_function_is_even_l153_153598

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the main theorem
theorem derivative_of_odd_function_is_even (f g : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, g x = deriv f x) :
  ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_function_is_even_l153_153598


namespace ab_ac_bc_all_real_l153_153676

theorem ab_ac_bc_all_real (a b c : ℝ) (h : a + b + c = 1) : ∃ x : ℝ, ab + ac + bc = x := by
  sorry

end ab_ac_bc_all_real_l153_153676


namespace boxes_left_to_sell_l153_153661

def sales_goal : ℕ := 150
def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def total_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer

theorem boxes_left_to_sell : sales_goal - total_sold = 75 := by
  sorry

end boxes_left_to_sell_l153_153661


namespace discriminant_positive_l153_153060

theorem discriminant_positive
  (a b c : ℝ)
  (h : (a + b + c) * c < 0) : b^2 - 4 * a * c > 0 :=
sorry

end discriminant_positive_l153_153060


namespace cost_price_of_book_l153_153641

theorem cost_price_of_book 
  (C : ℝ) 
  (h1 : 1.10 * C = sp10) 
  (h2 : 1.15 * C = sp15)
  (h3 : sp15 - sp10 = 90) : 
  C = 1800 := 
sorry

end cost_price_of_book_l153_153641


namespace guilt_proof_l153_153390

variables (E F G : Prop)

theorem guilt_proof
  (h1 : ¬G → F)
  (h2 : ¬E → G)
  (h3 : G → E)
  (h4 : E → ¬F)
  : E ∧ G :=
by
  sorry

end guilt_proof_l153_153390


namespace count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l153_153712

-- Setup the basic context
def Pocket := Finset (Fin 11)

-- The pocket contains 4 red balls and 7 white balls
def red_balls : Finset (Fin 11) := {0, 1, 2, 3}
def white_balls : Finset (Fin 11) := {4, 5, 6, 7, 8, 9, 10}

-- Question 1
theorem count_selection_4_balls :
  (red_balls.card.choose 4) + (red_balls.card.choose 3 * white_balls.card.choose 1) +
  (red_balls.card.choose 2 * white_balls.card.choose 2) = 115 := 
sorry

-- Question 2
theorem count_selection_5_balls_score_at_least_7_points :
  (red_balls.card.choose 2 * white_balls.card.choose 3) +
  (red_balls.card.choose 3 * white_balls.card.choose 2) +
  (red_balls.card.choose 4 * white_balls.card.choose 1) = 301 := 
sorry

end count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l153_153712


namespace circle_inscribed_radius_l153_153958

theorem circle_inscribed_radius (R α : ℝ) (hα : α < Real.pi) : 
  ∃ x : ℝ, x = R * (Real.sin (α / 4))^2 :=
sorry

end circle_inscribed_radius_l153_153958


namespace monica_study_ratio_l153_153648

theorem monica_study_ratio :
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  weekend = wednesday + thursday + friday :=
by
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  sorry

end monica_study_ratio_l153_153648


namespace correctStatement_l153_153192

def isValidInput : String → Bool
| "INPUT a, b, c;" => true
| "INPUT x=3;" => false
| _ => false

def isValidOutput : String → Bool
| "PRINT 20,3*2." => true
| "PRINT A=4;" => false
| _ => false

def isValidStatement : String → Bool
| stmt => (isValidInput stmt ∨ isValidOutput stmt)

theorem correctStatement : isValidStatement "PRINT 20,3*2." = true ∧ 
                           ¬(isValidStatement "INPUT a; b; c;" = true) ∧ 
                           ¬(isValidStatement "INPUT x=3;" = true) ∧ 
                           ¬(isValidStatement "PRINT A=4;" = true) := 
by sorry

end correctStatement_l153_153192


namespace find_x0_l153_153046

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^2 + c
noncomputable def int_f (a c : ℝ) : ℝ := ∫ x in (0 : ℝ)..1, f x a c

theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1)
  (h_eq : int_f a c = f x0 a c) : x0 = Real.sqrt 3 / 3 := sorry

end find_x0_l153_153046


namespace find_R_l153_153105

theorem find_R (a b : ℝ) (Q R : ℝ) (hQ : Q = 4)
  (h1 : 1/a + 1/b = Q/(a + b))
  (h2 : a/b + b/a = R) : R = 2 :=
by
  sorry

end find_R_l153_153105


namespace correct_operation_l153_153130

variable (a b : ℝ)

theorem correct_operation (h1 : a^2 + a^3 ≠ a^5)
                          (h2 : (-a^2)^3 ≠ a^6)
                          (h3 : -2*a^3*b / (a*b) ≠ -2*a^2*b) :
                          a^2 * a^3 = a^5 :=
by sorry

end correct_operation_l153_153130


namespace a_lt_1_sufficient_but_not_necessary_l153_153559

noncomputable def represents_circle (a : ℝ) : Prop :=
  a^2 - 10 * a + 9 > 0

theorem a_lt_1_sufficient_but_not_necessary (a : ℝ) :
  represents_circle a → ((a < 1) ∨ (a > 9)) :=
sorry

end a_lt_1_sufficient_but_not_necessary_l153_153559


namespace production_bottles_l153_153881

-- Definitions from the problem conditions
def machines_production_rate (machines : ℕ) (rate : ℕ) : ℕ := rate / machines
def total_production (machines rate minutes : ℕ) : ℕ := machines * rate * minutes

-- Theorem to prove the solution
theorem production_bottles :
  machines_production_rate 6 300 = 50 →
  total_production 10 50 4 = 2000 :=
by
  intro h
  have : 10 * 50 * 4 = 2000 := by norm_num
  exact this

end production_bottles_l153_153881


namespace hiker_final_distance_l153_153013

-- Definitions of the movements
def northward_movement : ℤ := 20
def southward_movement : ℤ := 8
def westward_movement : ℤ := 15
def eastward_movement : ℤ := 10

-- Definitions of the net movements
def net_north_south_movement : ℤ := northward_movement - southward_movement
def net_east_west_movement : ℤ := westward_movement - eastward_movement

-- The proof statement
theorem hiker_final_distance : 
  (net_north_south_movement^2 + net_east_west_movement^2) = 13^2 := by 
    sorry

end hiker_final_distance_l153_153013


namespace polynomial_sum_l153_153076

def f (x : ℝ) : ℝ := -4 * x^3 - 3 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + x - 7
def h (x : ℝ) : ℝ := 3 * x^3 + 6 * x^2 + 3 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = x^3 - 2 * x^2 + 6 * x - 10 := by
  sorry

end polynomial_sum_l153_153076


namespace probability_different_colors_l153_153101

theorem probability_different_colors :
  let total_chips := 18
  let blue_chips := 7
  let red_chips := 6
  let yellow_chips := 5
  let prob_first_blue := blue_chips / total_chips
  let prob_first_red := red_chips / total_chips
  let prob_first_yellow := yellow_chips / total_chips
  let prob_second_not_blue := (red_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_red := (blue_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_yellow := (blue_chips + red_chips) / (total_chips - 1)
  (
    prob_first_blue * prob_second_not_blue +
    prob_first_red * prob_second_not_red +
    prob_first_yellow * prob_second_not_yellow
  ) = 122 / 153 :=
by sorry

end probability_different_colors_l153_153101


namespace symmetry_axes_condition_l153_153140

/-- Define the property of having axes of symmetry for a geometric figure -/
def has_symmetry_axes (bounded : Bool) (two_parallel_axes : Bool) : Prop :=
  if bounded then 
    ¬ two_parallel_axes 
  else 
    true

/-- Main theorem stating the condition on symmetry axes for bounded and unbounded geometric figures -/
theorem symmetry_axes_condition (bounded : Bool) : 
  ∃ two_parallel_axes : Bool, has_symmetry_axes bounded two_parallel_axes :=
by
  -- The proof itself is not necessary as per the problem statement
  sorry

end symmetry_axes_condition_l153_153140


namespace smallest_number_increased_by_3_divisible_by_divisors_l153_153987

theorem smallest_number_increased_by_3_divisible_by_divisors
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 27)
  (h2 : d2 = 35)
  (h3 : d3 = 25)
  (h4 : d4 = 21) :
  (n + 3) % d1 = 0 →
  (n + 3) % d2 = 0 →
  (n + 3) % d3 = 0 →
  (n + 3) % d4 = 0 →
  n = 4722 :=
by
  sorry

end smallest_number_increased_by_3_divisible_by_divisors_l153_153987


namespace brand_tangyuan_purchase_l153_153553

theorem brand_tangyuan_purchase (x y : ℕ) 
  (h1 : x + y = 1000) 
  (h2 : x = 2 * y + 20) : 
  x = 670 ∧ y = 330 := 
sorry

end brand_tangyuan_purchase_l153_153553


namespace rain_all_three_days_is_six_percent_l153_153212

-- Definitions based on conditions from step a)
def P_rain_friday : ℚ := 2 / 5
def P_rain_saturday : ℚ := 1 / 2
def P_rain_sunday : ℚ := 3 / 10

-- The probability it will rain on all three days
def P_rain_all_three_days : ℚ := P_rain_friday * P_rain_saturday * P_rain_sunday

-- The Lean 4 theorem statement
theorem rain_all_three_days_is_six_percent : P_rain_all_three_days * 100 = 6 := by
  sorry

end rain_all_three_days_is_six_percent_l153_153212


namespace fraction_zero_l153_153315

theorem fraction_zero (x : ℝ) (h : x ≠ -1) (h₀ : (x^2 - 1) / (x + 1) = 0) : x = 1 :=
by {
  sorry
}

end fraction_zero_l153_153315


namespace mabel_total_tomatoes_l153_153816

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l153_153816


namespace order_of_m_n_p_q_l153_153195

variable {m n p q : ℝ} -- Define the variables as real numbers

theorem order_of_m_n_p_q (h1 : m < n) 
                         (h2 : p < q) 
                         (h3 : (p - m) * (p - n) < 0) 
                         (h4 : (q - m) * (q - n) < 0) : 
    m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_m_n_p_q_l153_153195


namespace percentage_failed_in_english_l153_153120

theorem percentage_failed_in_english (total_students : ℕ) (hindi_failed : ℕ) (both_failed : ℕ) (both_passed : ℕ) 
  (H1 : hindi_failed = total_students * 25 / 100)
  (H2 : both_failed = total_students * 25 / 100)
  (H3 : both_passed = total_students * 50 / 100)
  : (total_students * 50 / 100) = (total_students * 75 / 100) + (both_failed) - both_passed
:= sorry

end percentage_failed_in_english_l153_153120


namespace distinct_rational_numbers_count_l153_153668

theorem distinct_rational_numbers_count :
  ∃ N : ℕ, 
    (N = 49) ∧
    ∀ (k : ℚ), |k| < 50 →
      (∃ x : ℤ, x^2 - k * x + 18 = 0) →
        ∃ m: ℤ, k = 2 * m ∧ |m| < 25 :=
sorry

end distinct_rational_numbers_count_l153_153668


namespace linear_dependent_vectors_l153_153278

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l153_153278


namespace coefficient_x2y2_l153_153402

theorem coefficient_x2y2 : 
  let expr1 := (1 + x) ^ 3
  let expr2 := (1 + y) ^ 4
  let C3_2 := Nat.choose 3 2
  let C4_2 := Nat.choose 4 2
  (C3_2 * C4_2 = 18) := by
    sorry

end coefficient_x2y2_l153_153402


namespace right_triangle_property_l153_153878

theorem right_triangle_property
  (a b c x : ℝ)
  (h1 : c^2 = a^2 + b^2)
  (h2 : 1/2 * a * b = 1/2 * c * x)
  : 1/x^2 = 1/a^2 + 1/b^2 :=
sorry

end right_triangle_property_l153_153878


namespace mean_of_set_l153_153597

theorem mean_of_set {m : ℝ} 
  (median_condition : (m + 8 + m + 11) / 2 = 19) : 
  (m + (m + 6) + (m + 8) + (m + 11) + (m + 18) + (m + 20)) / 6 = 20 := 
by 
  sorry

end mean_of_set_l153_153597


namespace parallel_lines_m_values_l153_153872

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 → 2 * x + (5 + m) * y = 8) →
  (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l153_153872


namespace find_a_of_perpendicular_tangent_and_line_l153_153210

open Real

theorem find_a_of_perpendicular_tangent_and_line :
  let e := Real.exp 1
  let slope_tangent := 1 / e
  let slope_line (a : ℝ) := a
  let tangent_perpendicular := ∀ (a : ℝ), slope_tangent * slope_line a = -1
  tangent_perpendicular -> ∃ a : ℝ, a = -e :=
by {
  sorry
}

end find_a_of_perpendicular_tangent_and_line_l153_153210


namespace balloon_permutations_count_l153_153383

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l153_153383


namespace valid_raise_percentage_l153_153279

-- Define the conditions
def raise_between (x : ℝ) : Prop :=
  0.05 ≤ x ∧ x ≤ 0.10

def salary_increase_by_fraction (x : ℝ) : Prop :=
  x = 0.06

-- Define the main theorem 
theorem valid_raise_percentage (x : ℝ) (hx_between : raise_between x) (hx_fraction : salary_increase_by_fraction x) :
  x = 0.06 :=
sorry

end valid_raise_percentage_l153_153279


namespace parabola_tangent_midpoint_l153_153366

theorem parabola_tangent_midpoint (p : ℝ) (h : p > 0) :
    (∃ M : ℝ × ℝ, M = (2, -2*p)) ∧ 
    (∃ A B : ℝ × ℝ, A ≠ B ∧ 
                      (∃ yA yB : ℝ, yA = (A.1^2)/(2*p) ∧ yB = (B.1^2)/(2*p)) ∧ 
                      (0.5 * (A.2 + B.2) = 6)) → p = 1 := by sorry

end parabola_tangent_midpoint_l153_153366


namespace total_cartons_used_l153_153123

theorem total_cartons_used (x : ℕ) (y : ℕ) (h1 : y = 24) (h2 : 2 * x + 3 * y = 100) : x + y = 38 :=
sorry

end total_cartons_used_l153_153123


namespace max_rock_value_l153_153979

/-- Carl discovers a cave with three types of rocks:
    - 6-pound rocks worth $16 each,
    - 3-pound rocks worth $9 each,
    - 2-pound rocks worth $3 each.
    There are at least 15 of each type.
    He can carry a maximum of 20 pounds and no more than 5 rocks in total.
    Prove that the maximum value, in dollars, of the rocks he can carry is $52. -/
theorem max_rock_value :
  ∃ (max_value: ℕ),
  (∀ (c6 c3 c2: ℕ),
    (c6 + c3 + c2 ≤ 5) ∧
    (6 * c6 + 3 * c3 + 2 * c2 ≤ 20) →
    max_value ≥ 16 * c6 + 9 * c3 + 3 * c2) ∧
  max_value = 52 :=
by
  sorry

end max_rock_value_l153_153979


namespace total_spent_on_entertainment_l153_153459

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end total_spent_on_entertainment_l153_153459


namespace numbers_difference_l153_153995

theorem numbers_difference (A B C : ℝ) (h1 : B = 10) (h2 : B - A = C - B) (h3 : A * B = 85) (h4 : B * C = 115) : 
  B - A = 1.5 ∧ C - B = 1.5 :=
by
  sorry

end numbers_difference_l153_153995


namespace particular_solutions_of_diff_eq_l153_153697

variable {x y : ℝ}

theorem particular_solutions_of_diff_eq
  (h₁ : ∀ C : ℝ, x^2 = C * (y - C))
  (h₂ : x > 0) :
  (y = 2 * x ∨ y = -2 * x) ↔ (x * (y')^2 - 2 * y * y' + 4 * x = 0) := 
sorry

end particular_solutions_of_diff_eq_l153_153697


namespace boys_variance_greater_than_girls_l153_153988

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := (scores.sum / n)
  let squared_diff := scores.map (λ x => (x - mean) ^ 2)
  (squared_diff.sum) / n

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end boys_variance_greater_than_girls_l153_153988


namespace definite_integral_solution_l153_153968

noncomputable def integral_problem : ℝ := 
  by 
    sorry

theorem definite_integral_solution :
  integral_problem = (1/6 : ℝ) + Real.log 2 - Real.log 3 := 
by
  sorry

end definite_integral_solution_l153_153968


namespace annual_rent_per_square_foot_l153_153887

-- Given conditions
def dimensions_length : ℕ := 10
def dimensions_width : ℕ := 10
def monthly_rent : ℕ := 1300

-- Derived conditions
def area : ℕ := dimensions_length * dimensions_width
def annual_rent : ℕ := monthly_rent * 12

-- The problem statement as a theorem in Lean 4
theorem annual_rent_per_square_foot :
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_l153_153887


namespace perimeter_of_square_l153_153132

-- Defining the context and proving the equivalence.
theorem perimeter_of_square (x y : ℕ) (h : Nat.gcd x y = 3) (area : ℕ) :
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  perimeter = 24 * Real.sqrt 5 :=
by
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  sorry

end perimeter_of_square_l153_153132


namespace bus_people_next_pickup_point_l153_153072

theorem bus_people_next_pickup_point (bus_capacity : ℕ) (fraction_first_pickup : ℚ) (cannot_board : ℕ)
  (h1 : bus_capacity = 80)
  (h2 : fraction_first_pickup = 3 / 5)
  (h3 : cannot_board = 18) : 
  ∃ people_next_pickup : ℕ, people_next_pickup = 50 :=
by
  sorry

end bus_people_next_pickup_point_l153_153072


namespace find_positive_real_solution_l153_153460

theorem find_positive_real_solution (x : ℝ) : 
  0 < x ∧ (1 / 2 * (4 * x^2 - 1) = (x^2 - 60 * x - 20) * (x^2 + 30 * x + 10)) ↔ 
  (x = 30 + Real.sqrt 919 ∨ x = -15 + Real.sqrt 216 ∧ 0 < -15 + Real.sqrt 216) :=
by sorry

end find_positive_real_solution_l153_153460


namespace find_x_l153_153757

open Real

theorem find_x (x : ℝ) (h : (x / 6) / 3 = 6 / (x / 3)) : x = 18 ∨ x = -18 :=
by
  sorry

end find_x_l153_153757


namespace rational_solutions_count_l153_153823

theorem rational_solutions_count :
  ∃ (sols : Finset (ℚ × ℚ × ℚ)), 
    (∀ (x y z : ℚ), (x + y + z = 0) ∧ (x * y * z + z = 0) ∧ (x * y + y * z + x * z + y = 0) ↔ (x, y, z) ∈ sols) ∧
    sols.card = 3 :=
by
  sorry

end rational_solutions_count_l153_153823


namespace erasers_pens_markers_cost_l153_153837

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end erasers_pens_markers_cost_l153_153837


namespace find_percentage_l153_153427

theorem find_percentage (x p : ℝ) (h1 : x = 840) (h2 : 0.25 * x + 15 = p / 100 * 1500) : p = 15 := 
by
  sorry

end find_percentage_l153_153427


namespace dabbies_turkey_cost_l153_153627

noncomputable def first_turkey_weight : ℕ := 6
noncomputable def second_turkey_weight : ℕ := 9
noncomputable def third_turkey_weight : ℕ := 2 * second_turkey_weight
noncomputable def cost_per_kg : ℕ := 2

noncomputable def total_cost : ℕ :=
  first_turkey_weight * cost_per_kg +
  second_turkey_weight * cost_per_kg +
  third_turkey_weight * cost_per_kg

theorem dabbies_turkey_cost : total_cost = 66 :=
by
  sorry

end dabbies_turkey_cost_l153_153627


namespace rich_avg_time_per_mile_l153_153799

-- Define the total time in minutes and the total distance
def total_minutes : ℕ := 517
def total_miles : ℕ := 50

-- Define a function to calculate the average time per mile
def avg_time_per_mile (total_time : ℕ) (distance : ℕ) : ℚ :=
  total_time / distance

-- Theorem statement
theorem rich_avg_time_per_mile :
  avg_time_per_mile total_minutes total_miles = 10.34 :=
by
  -- Proof steps go here
  sorry

end rich_avg_time_per_mile_l153_153799


namespace remainder_of_sum_div_10_l153_153425

theorem remainder_of_sum_div_10 : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 :=
by
  sorry

end remainder_of_sum_div_10_l153_153425


namespace map_distance_to_actual_distance_l153_153777

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_inches : ℝ)
  (scale_miles : ℝ)
  (actual_distance : ℝ)
  (h_scale : scale_inches = 0.5)
  (h_scale_miles : scale_miles = 10)
  (h_map_distance : map_distance = 20) :
  actual_distance = 400 :=
by
  sorry

end map_distance_to_actual_distance_l153_153777


namespace problem_statement_l153_153286

namespace MathProof

def p : Prop := (2 + 4 = 7)
def q : Prop := ∀ x : ℝ, x = 1 → x^2 ≠ 1

theorem problem_statement : ¬ (p ∧ q) ∧ (p ∨ q) :=
by
  -- To be filled in
  sorry

end MathProof

end problem_statement_l153_153286


namespace alcohol_water_ratio_l153_153779

theorem alcohol_water_ratio (A W A_new W_new : ℝ) (ha1 : A / W = 4 / 3) (ha2: A = 5) (ha3: W_new = W + 7) : A / W_new = 1 / 2.15 :=
by
  sorry

end alcohol_water_ratio_l153_153779


namespace major_premise_wrong_l153_153569

-- Definition of the problem conditions and the proof goal
theorem major_premise_wrong :
  (∀ a : ℝ, |a| > 0) ↔ false :=
by {
  sorry  -- the proof goes here but is omitted as per the instructions
}

end major_premise_wrong_l153_153569


namespace sum_of_coordinates_l153_153595

theorem sum_of_coordinates (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 20) : x + y = 2 :=
sorry

end sum_of_coordinates_l153_153595


namespace bookcase_length_in_inches_l153_153365

theorem bookcase_length_in_inches (feet_length : ℕ) (inches_per_foot : ℕ) (h1 : feet_length = 4) (h2 : inches_per_foot = 12) : (feet_length * inches_per_foot) = 48 :=
by
  sorry

end bookcase_length_in_inches_l153_153365


namespace cost_per_tree_l153_153094

theorem cost_per_tree
    (initial_temperature : ℝ := 80)
    (final_temperature : ℝ := 78.2)
    (total_cost : ℝ := 108)
    (temperature_drop_per_tree : ℝ := 0.1) :
    total_cost / ((initial_temperature - final_temperature) / temperature_drop_per_tree) = 6 :=
by sorry

end cost_per_tree_l153_153094


namespace range_of_m_l153_153971

theorem range_of_m (m : ℝ) (x : ℝ) :
  (¬ (|1 - (x - 1) / 3| ≤ 2) → ¬ (x^2 - 2 * x + (1 - m^2) ≤ 0)) → 
  (|m| ≥ 9) :=
by
  sorry

end range_of_m_l153_153971


namespace maximize_product_minimize_product_l153_153300

-- Define lists of the digits to be used
def digits : List ℕ := [2, 4, 6, 8]

-- Function to calculate the number from a list of digits
def toNumber (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * 10 + d) 0

-- Function to calculate the product given two numbers represented as lists of digits
def product (digits1 digits2 : List ℕ) : ℕ :=
  toNumber digits1 * toNumber digits2

-- Definitions of specific permutations to be used
def maxDigits1 : List ℕ := [8, 6, 4]
def maxDigit2 : List ℕ := [2]
def minDigits1 : List ℕ := [2, 4, 6]
def minDigit2 : List ℕ := [8]

-- Theorem statements
theorem maximize_product : product maxDigits1 maxDigit2 = 864 * 2 := by
  sorry

theorem minimize_product : product minDigits1 minDigit2 = 246 * 8 := by
  sorry

end maximize_product_minimize_product_l153_153300


namespace log_a_plus_b_eq_zero_l153_153713

open Complex

noncomputable def a_b_expression : ℂ := (⟨2, 1⟩ / ⟨1, 1⟩ : ℂ)

noncomputable def a : ℝ := a_b_expression.re

noncomputable def b : ℝ := a_b_expression.im

theorem log_a_plus_b_eq_zero : log (a + b) = 0 := by
  sorry

end log_a_plus_b_eq_zero_l153_153713


namespace geom_seq_min_value_l153_153748

theorem geom_seq_min_value (r : ℝ) : 
  (1 : ℝ) = a_1 → a_2 = r → a_3 = r^2 → ∃ r : ℝ, 6 * a_2 + 7 * a_3 = -9/7 := 
by 
  intros h1 h2 h3 
  use -3/7 
  rw [h2, h3] 
  ring 
  sorry

end geom_seq_min_value_l153_153748


namespace system_of_two_linear_equations_l153_153844

theorem system_of_two_linear_equations :
  ((∃ x y z, x + z = 5 ∧ x - 2 * y = 6) → False) ∧
  ((∃ x y, x * y = 5 ∧ x - 4 * y = 2) → False) ∧
  ((∃ x y, x + y = 5 ∧ 3 * x - 4 * y = 12) → True) ∧
  ((∃ x y, x^2 + y = 2 ∧ x - y = 9) → False) :=
by {
  sorry
}

end system_of_two_linear_equations_l153_153844


namespace area_inner_square_l153_153088

theorem area_inner_square (ABCD_side : ℝ) (BE : ℝ) (EFGH_area : ℝ) 
  (h1 : ABCD_side = Real.sqrt 50) 
  (h2 : BE = 1) :
  EFGH_area = 36 :=
by
  sorry

end area_inner_square_l153_153088


namespace sphere_pyramid_problem_l153_153926

theorem sphere_pyramid_problem (n m : ℕ) :
  (n * (n + 1) * (2 * n + 1)) / 6 + (m * (m + 1) * (m + 2)) / 6 = 605 → n = 10 ∧ m = 10 :=
by
  sorry

end sphere_pyramid_problem_l153_153926


namespace saturday_price_is_correct_l153_153419

-- Define Thursday's price
def thursday_price : ℝ := 50

-- Define the price increase rate on Friday
def friday_increase_rate : ℝ := 0.2

-- Define the discount rate on Saturday
def saturday_discount_rate : ℝ := 0.15

-- Calculate the price on Friday
def friday_price : ℝ := thursday_price * (1 + friday_increase_rate)

-- Calculate the discount amount on Saturday
def saturday_discount : ℝ := friday_price * saturday_discount_rate

-- Calculate the price on Saturday
def saturday_price : ℝ := friday_price - saturday_discount

-- Theorem stating the price on Saturday
theorem saturday_price_is_correct : saturday_price = 51 := by
  -- Definitions are already embedded into the conditions
  -- so here we only state the property to be proved.
  sorry

end saturday_price_is_correct_l153_153419


namespace housewife_more_kgs_l153_153335

theorem housewife_more_kgs (P R money more_kgs : ℝ)
  (hR: R = 40)
  (hReduction: R = P - 0.25 * P)
  (hMoney: money = 800)
  (hMoreKgs: more_kgs = (money / R) - (money / P)) :
  more_kgs = 5 :=
  by
    sorry

end housewife_more_kgs_l153_153335


namespace fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l153_153565

theorem fourth_vertex_of_regular_tetrahedron_exists_and_is_unique :
  ∃ (x y z : ℤ),
    (x, y, z) ≠ (1, 2, 3) ∧ (x, y, z) ≠ (5, 3, 2) ∧ (x, y, z) ≠ (4, 2, 6) ∧
    (x - 1)^2 + (y - 2)^2 + (z - 3)^2 = 18 ∧
    (x - 5)^2 + (y - 3)^2 + (z - 2)^2 = 18 ∧
    (x - 4)^2 + (y - 2)^2 + (z - 6)^2 = 18 ∧
    (x, y, z) = (2, 3, 5) :=
by
  -- Proof goes here
  sorry

end fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l153_153565


namespace number_of_friends_l153_153221

def money_emma : ℕ := 8

def money_daya : ℕ := money_emma + (money_emma * 25 / 100)

def money_jeff : ℕ := (2 * money_daya) / 5

def money_brenda : ℕ := money_jeff + 4

def money_brenda_condition : Prop := money_brenda = 8

def friends_pooling_pizza : ℕ := 4

theorem number_of_friends (h : money_brenda_condition) : friends_pooling_pizza = 4 := by
  sorry

end number_of_friends_l153_153221


namespace wood_length_equation_l153_153247

-- Define the conditions as hypotheses
def length_of_wood_problem (x : ℝ) :=
  (1 / 2) * (x + 4.5) = x - 1

-- Now we state the theorem we want to prove, which is equivalent to the question == answer
theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l153_153247


namespace max_value_a7_b7_c7_d7_l153_153743

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end max_value_a7_b7_c7_d7_l153_153743


namespace sum_of_a_and_b_is_24_l153_153545

theorem sum_of_a_and_b_is_24 
  (a b : ℕ) 
  (h_a_pos : a > 0) 
  (h_b_gt_one : b > 1) 
  (h_maximal : ∀ (a' b' : ℕ), (a' > 0) → (b' > 1) → (a'^b' < 500) → (a'^b' ≤ a^b)) :
  a + b = 24 := 
sorry

end sum_of_a_and_b_is_24_l153_153545


namespace exists_x0_l153_153913

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3 * x))^2 - 2 * a * (x + 3 * Real.log (3 * x)) + 10 * a^2

theorem exists_x0 (a : ℝ) (h : a = 1 / 30) : ∃ x0 : ℝ, f x0 a ≤ 1 / 10 := 
by
  sorry

end exists_x0_l153_153913


namespace evaluate_expression_l153_153350

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end evaluate_expression_l153_153350


namespace triangle_proof_l153_153554

noncomputable def length_DC (AB DA BC DB : ℝ) : ℝ :=
  Real.sqrt (BC^2 - DB^2)

theorem triangle_proof :
  ∀ (AB DA BC DB : ℝ), AB = 30 → DA = 24 → BC = 22.5 → DB = 18 →
  length_DC AB DA BC DB = 13.5 :=
by
  intros AB DA BC DB hAB hDA hBC hDB
  rw [length_DC]
  sorry

end triangle_proof_l153_153554


namespace return_trip_time_l153_153071

variables (d p w : ℝ)
-- Condition 1: The outbound trip against the wind took 120 minutes.
axiom h1 : d = 120 * (p - w)
-- Condition 2: The return trip with the wind took 15 minutes less than it would in still air.
axiom h2 : d / (p + w) = d / p - 15

-- Translate the conclusion that needs to be proven in Lean 4
theorem return_trip_time (h1 : d = 120 * (p - w)) (h2 : d / (p + w) = d / p - 15) : (d / (p + w) = 15) ∨ (d / (p + w) = 85) :=
sorry

end return_trip_time_l153_153071


namespace macy_hit_ball_50_times_l153_153613

-- Definitions and conditions
def token_pitches : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def piper_hits : ℕ := 55
def missed_pitches : ℕ := 315

-- Calculation based on conditions
def total_pitches : ℕ := (macy_tokens + piper_tokens) * token_pitches
def total_hits : ℕ := total_pitches - missed_pitches
def macy_hits : ℕ := total_hits - piper_hits

-- Prove that Macy hit 50 times
theorem macy_hit_ball_50_times : macy_hits = 50 := 
by
  sorry

end macy_hit_ball_50_times_l153_153613


namespace marble_draw_probability_l153_153775

theorem marble_draw_probability :
  let total_marbles := 12
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 3

  let p_red_first := (red_marbles / total_marbles : ℚ)
  let p_white_second := (white_marbles / (total_marbles - 1) : ℚ)
  let p_blue_third := (blue_marbles / (total_marbles - 2) : ℚ)
  
  p_red_first * p_white_second * p_blue_third = (1/22 : ℚ) :=
by
  sorry

end marble_draw_probability_l153_153775


namespace evaluations_total_l153_153594

theorem evaluations_total :
    let class_A_students := 30
    let class_A_mc := 12
    let class_A_essay := 3
    let class_A_presentation := 1

    let class_B_students := 25
    let class_B_mc := 15
    let class_B_short_answer := 5
    let class_B_essay := 2

    let class_C_students := 35
    let class_C_mc := 10
    let class_C_essay := 3
    let class_C_presentation_groups := class_C_students / 5 -- groups of 5

    let class_D_students := 40
    let class_D_mc := 11
    let class_D_short_answer := 4
    let class_D_essay := 3

    let class_E_students := 20
    let class_E_mc := 14
    let class_E_short_answer := 5
    let class_E_essay := 2

    let total_mc := (class_A_students * class_A_mc) +
                    (class_B_students * class_B_mc) +
                    (class_C_students * class_C_mc) +
                    (class_D_students * class_D_mc) +
                    (class_E_students * class_E_mc)

    let total_short_answer := (class_B_students * class_B_short_answer) +
                              (class_D_students * class_D_short_answer) +
                              (class_E_students * class_E_short_answer)

    let total_essay := (class_A_students * class_A_essay) +
                       (class_B_students * class_B_essay) +
                       (class_C_students * class_C_essay) +
                       (class_D_students * class_D_essay) +
                       (class_E_students * class_E_essay)

    let total_presentation := (class_A_students * class_A_presentation) +
                              class_C_presentation_groups

    total_mc + total_short_answer + total_essay + total_presentation = 2632 := by
    sorry

end evaluations_total_l153_153594


namespace cn_squared_eq_28_l153_153359

theorem cn_squared_eq_28 (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end cn_squared_eq_28_l153_153359


namespace workers_not_worked_days_l153_153643

theorem workers_not_worked_days (W N : ℤ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 := 
by
  sorry

end workers_not_worked_days_l153_153643


namespace carrie_money_left_l153_153452

/-- Carrie was given $91. She bought a sweater for $24, 
    a T-shirt for $6, a pair of shoes for $11,
    and a pair of jeans originally costing $30 with a 25% discount. 
    Prove that she has $27.50 left. -/
theorem carrie_money_left :
  let init_money := 91
  let sweater := 24
  let t_shirt := 6
  let shoes := 11
  let jeans := 30
  let discount := 25 / 100
  let jeans_discounted_price := jeans * (1 - discount)
  let total_cost := sweater + t_shirt + shoes + jeans_discounted_price
  let money_left := init_money - total_cost
  money_left = 27.50 :=
by
  intros
  sorry

end carrie_money_left_l153_153452


namespace compute_fraction_product_l153_153273

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l153_153273


namespace alice_questions_wrong_l153_153083

theorem alice_questions_wrong (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 3) 
  (h3 : c = 7) : 
  a = 8.5 := 
by
  sorry

end alice_questions_wrong_l153_153083


namespace isosceles_obtuse_triangle_smallest_angle_l153_153610

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β : ℝ), 0 < α ∧ α = 1.5 * 90 ∧ α + 2 * β = 180 ∧ β = 22.5 := by
  sorry

end isosceles_obtuse_triangle_smallest_angle_l153_153610


namespace directly_above_156_is_133_l153_153998

def row_numbers (k : ℕ) : ℕ := 2 * k - 1

def total_numbers_up_to_row (k : ℕ) : ℕ := k * k

def find_row (n : ℕ) : ℕ :=
  Nat.sqrt (n + 1)

def position_in_row (n k : ℕ) : ℕ :=
  n - (total_numbers_up_to_row (k - 1)) + 1

def number_directly_above (n : ℕ) : ℕ :=
  let k := find_row n
  let pos := position_in_row n k
  (total_numbers_up_to_row (k - 1) - row_numbers (k - 1)) + pos + 1

theorem directly_above_156_is_133 : number_directly_above 156 = 133 := 
  by
  sorry

end directly_above_156_is_133_l153_153998


namespace calc_expression_l153_153531

-- Define the fractions and whole number in the problem
def frac1 : ℚ := 5/6
def frac2 : ℚ := 1 + 1/6
def whole : ℚ := 2

-- Define the expression to be proved
def expression : ℚ := (frac1) - (-whole) + (frac2)

-- The theorem to be proved
theorem calc_expression : expression = 4 :=
by { sorry }

end calc_expression_l153_153531


namespace yoojeong_rabbits_l153_153716

theorem yoojeong_rabbits :
  ∀ (R C : ℕ), 
  let minyoung_dogs := 9
  let minyoung_cats := 3
  let minyoung_rabbits := 5
  let minyoung_total := minyoung_dogs + minyoung_cats + minyoung_rabbits
  let yoojeong_total := minyoung_total + 2
  let yoojeong_dogs := 7
  let yoojeong_cats := R - 2
  yoojeong_total = yoojeong_dogs + (R - 2) + R → 
  R = 7 :=
by
  intros R C minyoung_dogs minyoung_cats minyoung_rabbits minyoung_total yoojeong_total yoojeong_dogs yoojeong_cats
  have h1 : minyoung_total = 9 + 3 + 5 := rfl
  have h2 : yoojeong_total = minyoung_total + 2 := by sorry
  have h3 : yoojeong_dogs = 7 := rfl
  have h4 : yoojeong_cats = R - 2 := by sorry
  sorry

end yoojeong_rabbits_l153_153716


namespace meadow_trees_count_l153_153236

theorem meadow_trees_count (n : ℕ) (f s m : ℕ → ℕ) :
  (f 20 = s 7) ∧ (f 7 = s 94) ∧ (s 7 > f 20) → 
  n = 100 :=
by
  sorry

end meadow_trees_count_l153_153236


namespace final_amount_in_account_l153_153787

noncomputable def initial_deposit : ℝ := 1000
noncomputable def first_year_interest_rate : ℝ := 0.2
noncomputable def first_year_balance : ℝ := initial_deposit * (1 + first_year_interest_rate)
noncomputable def withdrawal_amount : ℝ := first_year_balance / 2
noncomputable def after_withdrawal_balance : ℝ := first_year_balance - withdrawal_amount
noncomputable def second_year_interest_rate : ℝ := 0.15
noncomputable def final_balance : ℝ := after_withdrawal_balance * (1 + second_year_interest_rate)

theorem final_amount_in_account : final_balance = 690 := by
  sorry

end final_amount_in_account_l153_153787


namespace gcd_linear_combination_l153_153393

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := by
  sorry

end gcd_linear_combination_l153_153393


namespace evaluate_expression_l153_153349

theorem evaluate_expression : 3 - 5 * (6 - 2^3) / 2 = 8 :=
by
  sorry

end evaluate_expression_l153_153349


namespace quadratic_factorization_sum_l153_153649

theorem quadratic_factorization_sum (d e f : ℤ) (h1 : ∀ x, x^2 + 18 * x + 80 = (x + d) * (x + e)) 
                                     (h2 : ∀ x, x^2 - 20 * x + 96 = (x - e) * (x - f)) : 
                                     d + e + f = 30 :=
by
  sorry

end quadratic_factorization_sum_l153_153649


namespace total_distance_driven_l153_153946

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end total_distance_driven_l153_153946


namespace half_abs_diff_of_squares_l153_153739

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l153_153739


namespace slope_of_line_through_points_l153_153600

theorem slope_of_line_through_points :
  let x1 := 1
  let y1 := 3
  let x2 := 5
  let y2 := 7
  let m := (y2 - y1) / (x2 - x1)
  m = 1 := by
  sorry

end slope_of_line_through_points_l153_153600


namespace sara_change_l153_153142

-- Define the costs of individual items
def cost_book_1 : ℝ := 5.5
def cost_book_2 : ℝ := 6.5
def cost_notebook : ℝ := 3
def cost_bookmarks : ℝ := 2

-- Define the discounts and taxes
def discount_books : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Define the payment amount
def amount_given : ℝ := 20

-- Calculate the total cost, discount, and final amount
def discounted_book_cost := (cost_book_1 + cost_book_2) * (1 - discount_books)
def subtotal := discounted_book_cost + cost_notebook + cost_bookmarks
def total_with_tax := subtotal * (1 + sales_tax)
def change := amount_given - total_with_tax

-- State the theorem
theorem sara_change : change = 3.41 := by
  sorry

end sara_change_l153_153142


namespace no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l153_153731

theorem no_integer_solution_2_to_2x_minus_3_to_2y_eq_58
  (x y : ℕ)
  (h1 : 2 ^ (2 * x) - 3 ^ (2 * y) = 58) : false :=
by
  sorry

end no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l153_153731


namespace sin_tan_relation_l153_153671

theorem sin_tan_relation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -(2 / 5) := 
sorry

end sin_tan_relation_l153_153671


namespace water_breaks_vs_sitting_breaks_l153_153834

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end water_breaks_vs_sitting_breaks_l153_153834


namespace quadratic_sum_of_b_and_c_l153_153143

theorem quadratic_sum_of_b_and_c :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 - 20 * x + 36 = (x + b)^2 + c) ∧ b + c = -74 :=
by
  sorry

end quadratic_sum_of_b_and_c_l153_153143


namespace anne_bob_total_difference_l153_153412

-- Define specific values as constants
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.08

-- Define the calculations according to Anne's method
def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)

-- Define the calculations according to Bob's method
def bob_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

-- State the theorem that the difference between Anne's and Bob's totals is zero
theorem anne_bob_total_difference : anne_total - bob_total = 0 :=
by sorry  -- Proof not required

end anne_bob_total_difference_l153_153412


namespace parametric_to_ordinary_l153_153462

theorem parametric_to_ordinary (θ : ℝ) (x y : ℝ) : 
  x = Real.cos θ ^ 2 →
  y = 2 * Real.sin θ ^ 2 →
  (x ∈ Set.Icc 0 1) → 
  2 * x + y - 2 = 0 :=
by
  intros hx hy h_range
  sorry

end parametric_to_ordinary_l153_153462


namespace find_a10_l153_153959

theorem find_a10 (a_n : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h2 : 5 * a_n 3 = a_n 3 ^ 2)
  (h3 : (a_n 3 + 2 * d) ^ 2 = (a_n 3 - d) * (a_n 3 + 11 * d))
  (h_nonzero : d ≠ 0) :
  a_n 10 = 23 :=
sorry

end find_a10_l153_153959


namespace num_partition_sets_correct_l153_153625

noncomputable def num_partition_sets (n : ℕ) : ℕ :=
  2^(n-1) - 1

theorem num_partition_sets_correct (n : ℕ) (hn : n ≥ 2) : 
  num_partition_sets n = 2^(n-1) - 1 := 
by sorry

end num_partition_sets_correct_l153_153625


namespace intersection_with_complement_l153_153445

-- Define the universal set U, sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- The equivalent proof problem in Lean 4
theorem intersection_with_complement :
  A ∩ complement_B = {0, 2} :=
by
  sorry

end intersection_with_complement_l153_153445


namespace reams_paper_l153_153953

theorem reams_paper (total_reams reams_haley reams_sister : Nat) 
    (h1 : total_reams = 5)
    (h2 : reams_haley = 2)
    (h3 : total_reams = reams_haley + reams_sister) : 
    reams_sister = 3 := by
  sorry

end reams_paper_l153_153953


namespace draw_two_green_marbles_probability_l153_153503

theorem draw_two_green_marbles_probability :
  let red := 5
  let green := 3
  let white := 7
  let total := red + green + white
  (green / total) * ((green - 1) / (total - 1)) = 1 / 35 :=
by
  sorry

end draw_two_green_marbles_probability_l153_153503


namespace JakeMowingEarnings_l153_153415

theorem JakeMowingEarnings :
  (∀ rate hours_mowing hours_planting (total_charge : ℝ),
      rate = 20 →
      hours_mowing = 1 →
      hours_planting = 2 →
      total_charge = 45 →
      (total_charge = hours_planting * rate + 5) →
      hours_mowing * rate = 20) :=
by
  intros rate hours_mowing hours_planting total_charge
  sorry

end JakeMowingEarnings_l153_153415


namespace planes_count_l153_153439

-- Define the conditions as given in the problem.
def total_wings : ℕ := 90
def wings_per_plane : ℕ := 2

-- Define the number of planes calculation based on conditions.
def number_of_planes : ℕ := total_wings / wings_per_plane

-- Prove that the number of planes is 45.
theorem planes_count : number_of_planes = 45 :=
by 
  -- The proof steps are omitted as specified.
  sorry

end planes_count_l153_153439


namespace original_strip_length_l153_153851

theorem original_strip_length (x : ℝ) 
  (h1 : 3 + x + 3 + x + 3 + x + 3 + x + 3 = 27) : 
  4 * 9 + 4 * 3 = 57 := 
  sorry

end original_strip_length_l153_153851


namespace small_pump_filling_time_l153_153048

theorem small_pump_filling_time :
  ∃ S : ℝ, (L = 2) → 
         (1 / 0.4444444444444444 = S + L) → 
         (1 / S = 4) :=
by 
  sorry

end small_pump_filling_time_l153_153048


namespace calc_root_diff_l153_153591

theorem calc_root_diff : 81^(1/4) - 16^(1/2) = -1 := by
  sorry

end calc_root_diff_l153_153591


namespace find_x_l153_153317

variable {a b x r : ℝ}
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (h₂ : r = (4 * a)^(2 * b))
variable (h₃ : r = (a^b * x^b)^2)
variable (h₄ : 0 < x)

theorem find_x : x = 4 := by
  sorry

end find_x_l153_153317


namespace john_average_speed_l153_153145

noncomputable def time_uphill : ℝ := 45 / 60 -- 45 minutes converted to hours
noncomputable def distance_uphill : ℝ := 2   -- 2 km

noncomputable def time_downhill : ℝ := 15 / 60 -- 15 minutes converted to hours
noncomputable def distance_downhill : ℝ := 2   -- 2 km

noncomputable def total_distance : ℝ := distance_uphill + distance_downhill
noncomputable def total_time : ℝ := time_uphill + time_downhill

theorem john_average_speed : total_distance / total_time = 4 :=
by
  have h1 : total_distance = 4 := by sorry
  have h2 : total_time = 1 := by sorry
  rw [h1, h2]
  norm_num

end john_average_speed_l153_153145


namespace correct_statements_are_C_and_D_l153_153932

theorem correct_statements_are_C_and_D
  (a b c m : ℝ)
  (ha1 : -1 < a) (ha2 : a < 5)
  (hb1 : -2 < b) (hb2 : b < 3)
  (hab : a > b)
  (h_ac2bc2 : a * c^2 > b * c^2) (hc2_pos : c^2 > 0)
  (h_ab_pos : a > b) (h_b_pos : b > 0) (hm_pos : m > 0) :
  (¬(1 < a - b ∧ a - b < 2)) ∧ (¬(a^2 > b^2)) ∧ (a > b) ∧ ((b + m) / (a + m) > b / a) :=
by sorry

end correct_statements_are_C_and_D_l153_153932


namespace ratio_eq_neg_1009_l153_153107

theorem ratio_eq_neg_1009 (p q : ℝ) (h : (1 / p + 1 / q) / (1 / p - 1 / q) = 1009) : (p + q) / (p - q) = -1009 := 
by 
  sorry

end ratio_eq_neg_1009_l153_153107


namespace solve_for_y_l153_153704

theorem solve_for_y (x y : ℝ) (h1 : x * y = 1) (h2 : x / y = 36) (h3 : 0 < x) (h4 : 0 < y) : 
  y = 1 / 6 := 
sorry

end solve_for_y_l153_153704


namespace carrots_cost_l153_153067

/-
Define the problem conditions and parameters.
-/
def num_third_grade_classes := 5
def students_per_third_grade_class := 30
def num_fourth_grade_classes := 4
def students_per_fourth_grade_class := 28
def num_fifth_grade_classes := 4
def students_per_fifth_grade_class := 27

def cost_per_hamburger : ℝ := 2.10
def cost_per_cookie : ℝ := 0.20
def total_lunch_cost : ℝ := 1036

/-
Calculate the total number of students.
-/
def total_students : ℕ :=
  (num_third_grade_classes * students_per_third_grade_class) +
  (num_fourth_grade_classes * students_per_fourth_grade_class) +
  (num_fifth_grade_classes * students_per_fifth_grade_class)

/-
Calculate the cost of hamburgers and cookies.
-/
def hamburgers_cost : ℝ := total_students * cost_per_hamburger
def cookies_cost : ℝ := total_students * cost_per_cookie
def total_hamburgers_and_cookies_cost : ℝ := hamburgers_cost + cookies_cost

/-
State the proof problem: How much do the carrots cost?
-/
theorem carrots_cost : total_lunch_cost - total_hamburgers_and_cookies_cost = 185 :=
by
  -- Proof is omitted
  sorry

end carrots_cost_l153_153067


namespace avg_weight_b_c_l153_153638

variables (A B C : ℝ)

-- Given Conditions
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := B = 37

-- Statement to prove
theorem avg_weight_b_c 
  (h1 : condition1 A B C)
  (h2 : condition2 A B)
  (h3 : condition3 B) : 
  (B + C) / 2 = 46 :=
sorry

end avg_weight_b_c_l153_153638


namespace distinct_roots_difference_l153_153418

theorem distinct_roots_difference (r s : ℝ) (h₀ : r ≠ s) (h₁ : r > s) (h₂ : ∀ x, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) :
  r - s = Real.sqrt 29 :=
by
  sorry

end distinct_roots_difference_l153_153418


namespace walls_painted_purple_l153_153047

theorem walls_painted_purple :
  (10 - (3 * 10 / 5)) * 8 = 32 := by
  sorry

end walls_painted_purple_l153_153047


namespace sequence_may_or_may_not_be_arithmetic_l153_153626

theorem sequence_may_or_may_not_be_arithmetic (a : ℕ → ℕ) 
  (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : a 2 = 3) 
  (h4 : a 3 = 4) (h5 : a 4 = 5) : 
  ¬(∀ n, a (n + 1) - a n = 1) → 
  (∀ n, a (n + 1) - a n = 1) ∨ ¬(∀ n, a (n + 1) - a n = 1) :=
by
  sorry

end sequence_may_or_may_not_be_arithmetic_l153_153626


namespace digit_2567_l153_153763

def nth_digit_in_concatenation (n : ℕ) : ℕ :=
  sorry

theorem digit_2567 : nth_digit_in_concatenation 2567 = 8 :=
by
  sorry

end digit_2567_l153_153763


namespace sum_of_arithmetic_sequence_l153_153493

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) 
  (h₁ : S 4 = 2) 
  (h₂ : S 8 = 6) 
  : S 12 = 12 := 
by
  sorry

end sum_of_arithmetic_sequence_l153_153493


namespace coplanar_points_l153_153488

theorem coplanar_points (a : ℝ) :
  ∀ (V : ℝ), V = 2 + a^3 → V = 0 → a = -((2:ℝ)^(1/3)) :=
by
  sorry

end coplanar_points_l153_153488


namespace odd_increasing_min_5_then_neg5_max_on_neg_interval_l153_153137

-- Definitions using the conditions given in the problem statement
variable {f : ℝ → ℝ}

-- Condition 1: f is odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Condition 2: f is increasing on the interval [3, 7]
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f (x) ≤ f (y)

-- Condition 3: Minimum value of f on [3, 7] is 5
def min_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (min_val : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f (x) = min_val

-- Lean statement for the proof problem
theorem odd_increasing_min_5_then_neg5_max_on_neg_interval
  (f_odd: odd_function f)
  (f_increasing: increasing_on_interval f 3 7)
  (min_val: min_value_on_interval f 3 7 5) :
  increasing_on_interval f (-7) (-3) ∧ min_value_on_interval f (-7) (-3) (-5) :=
by sorry

end odd_increasing_min_5_then_neg5_max_on_neg_interval_l153_153137


namespace ratio_difference_l153_153767

theorem ratio_difference (x : ℕ) (h : 7 * x = 70) : 70 - 3 * x = 40 :=
by
  -- proof would go here
  sorry

end ratio_difference_l153_153767


namespace ship_lighthouse_distance_l153_153102

-- Definitions for conditions
def speed : ℝ := 15 -- speed of the ship in km/h
def time : ℝ := 4  -- time the ship sails eastward in hours
def angle_A : ℝ := 60 -- angle at point A in degrees
def angle_C : ℝ := 30 -- angle at point C in degrees

-- Main theorem statement
theorem ship_lighthouse_distance (d_A_C : ℝ) (d_C_B : ℝ) : d_A_C = speed * time → d_C_B = 60 := 
by sorry

end ship_lighthouse_distance_l153_153102


namespace james_bike_ride_total_distance_l153_153176

theorem james_bike_ride_total_distance 
  (d1 d2 d3 : ℝ)
  (H1 : d2 = 12)
  (H2 : d2 = 1.2 * d1)
  (H3 : d3 = 1.25 * d2) :
  d1 + d2 + d3 = 37 :=
by
  -- additional proof steps would go here
  sorry

end james_bike_ride_total_distance_l153_153176


namespace triplet_sums_to_two_l153_153540

theorem triplet_sums_to_two :
  (3 / 4 + 1 / 4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3 / 5 + 4 / 5 + 3 / 5 = 2) ∧
  (2 - 3 + 3 = 2) :=
by
  sorry

end triplet_sums_to_two_l153_153540


namespace find_y_l153_153588

theorem find_y (a b c x : ℝ) (p q r y: ℝ) (hx : x ≠ 1) 
  (h₁ : (Real.log a) / p = Real.log x) 
  (h₂ : (Real.log b) / q = Real.log x) 
  (h₃ : (Real.log c) / r = Real.log x)
  (h₄ : (b^3) / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := 
by {
  sorry
}

end find_y_l153_153588


namespace train_distance_30_minutes_l153_153690

theorem train_distance_30_minutes (h : ∀ (t : ℝ), 0 < t → (1 / 2) * t = 1 / 2 * t) : 
  (1 / 2) * 30 = 15 :=
by
  sorry

end train_distance_30_minutes_l153_153690


namespace min_xy_value_l153_153495

theorem min_xy_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (hlog : Real.log x / Real.log 2 * Real.log y / Real.log 2 = 1) : x * y = 4 :=
by sorry

end min_xy_value_l153_153495


namespace spherical_to_rectangular_coords_l153_153956

noncomputable def sphericalToRectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * (Real.sin phi) * (Real.cos theta), 
   rho * (Real.sin phi) * (Real.sin theta), 
   rho * (Real.cos phi))

theorem spherical_to_rectangular_coords :
  sphericalToRectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coords_l153_153956


namespace Nicole_fish_tanks_l153_153128

-- Definition to express the conditions
def first_tank_water := 8 -- gallons
def second_tank_difference := 2 -- fewer gallons than first tanks
def num_first_tanks := 2
def num_second_tanks := 2
def total_water_four_weeks := 112 -- gallons
def weeks := 4

-- Calculate the total water per week
def water_per_week := (num_first_tanks * first_tank_water) + (num_second_tanks * (first_tank_water - second_tank_difference))

-- Calculate the total number of tanks
def total_tanks := num_first_tanks + num_second_tanks

-- Proof statement
theorem Nicole_fish_tanks : total_water_four_weeks / water_per_week = weeks → total_tanks = 4 := by
  -- Proof goes here
  sorry

end Nicole_fish_tanks_l153_153128


namespace children_tickets_sold_l153_153207

theorem children_tickets_sold (A C : ℝ) (h1 : A + C = 400) (h2 : 6 * A + 4.5 * C = 2100) : C = 200 :=
sorry

end children_tickets_sold_l153_153207


namespace union_complement_set_l153_153708

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l153_153708


namespace total_handshakes_l153_153353

theorem total_handshakes (team1 team2 refs : ℕ) (players_per_team : ℕ) :
  team1 = 11 → team2 = 11 → refs = 3 → players_per_team = 11 →
  (players_per_team * players_per_team + (players_per_team * 2 * refs) = 187) :=
by
  intros h_team1 h_team2 h_refs h_players_per_team
  -- Now we want to prove that
  -- 11 * 11 + (11 * 2 * 3) = 187
  -- However, we can just add sorry here as the purpose is to write the statement
  sorry

end total_handshakes_l153_153353


namespace work_done_by_force_l153_153622

noncomputable def displacement (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem work_done_by_force :
  let F := (5, 2)
  let A := (-1, 3)
  let B := (2, 6)
  let AB := displacement A B
  dot_product F AB = 21 := by
  sorry

end work_done_by_force_l153_153622


namespace simplest_form_l153_153738

theorem simplest_form (b : ℝ) (h : b ≠ 2) : 2 - (2 / (2 + b / (2 - b))) = 4 / (4 - b) :=
by sorry

end simplest_form_l153_153738


namespace divisible_by_6_l153_153587

theorem divisible_by_6 (n : ℕ) : 6 ∣ ((n - 1) * n * (n^3 + 1)) := sorry

end divisible_by_6_l153_153587


namespace number_of_n_l153_153620

theorem number_of_n (n : ℕ) (h1 : n ≤ 1000) (h2 : ∃ k : ℕ, 18 * n = k^2) : 
  ∃ K : ℕ, K = 7 :=
sorry

end number_of_n_l153_153620


namespace find_m_of_quadratic_fn_l153_153564

theorem find_m_of_quadratic_fn (m : ℚ) (h : 2 * m - 1 = 2) : m = 3 / 2 :=
by
  sorry

end find_m_of_quadratic_fn_l153_153564


namespace problem1_problem2_l153_153897

noncomputable def h (x a : ℝ) : ℝ := (x - a) * Real.exp x + a
noncomputable def f (x b : ℝ) : ℝ := x^2 - 2 * b * x - 3 * Real.exp 1 + Real.exp 1 + 15 / 2

theorem problem1 (a : ℝ) :
  ∃ c, ∀ x ∈ Set.Icc (-1:ℝ) (1:ℝ), h x a ≥ c :=
by
  sorry

theorem problem2 (b : ℝ) :
  (∀ x1 ∈ Set.Icc (-1:ℝ) (1:ℝ), ∃ x2 ∈ Set.Icc (1:ℝ) (2:ℝ), h x1 3 ≥ f x2 b) →
  b ≥ 17 / 8 :=
by
  sorry

end problem1_problem2_l153_153897


namespace ratio_of_ages_l153_153068

variable (J L M : ℕ)

def louis_age := L = 14
def matilda_age := M = 35
def matilda_older := M = J + 7
def jerica_multiple := ∃ k : ℕ, J = k * L

theorem ratio_of_ages
  (hL : louis_age L)
  (hM : matilda_age M)
  (hMO : matilda_older J M)
  : J / L = 2 :=
by
  sorry

end ratio_of_ages_l153_153068


namespace monthly_revenue_l153_153700

variable (R : ℝ) -- The monthly revenue

-- Conditions
def after_taxes (R : ℝ) : ℝ := R * 0.90
def after_marketing (R : ℝ) : ℝ := (after_taxes R) * 0.95
def after_operational_costs (R : ℝ) : ℝ := (after_marketing R) * 0.80
def total_employee_wages (R : ℝ) : ℝ := (after_operational_costs R) * 0.15

-- Number of employees and their wages
def number_of_employees : ℝ := 10
def wage_per_employee : ℝ := 4104
def total_wages : ℝ := number_of_employees * wage_per_employee

-- Proof problem
theorem monthly_revenue : R = 400000 ↔ total_employee_wages R = total_wages := by
  sorry

end monthly_revenue_l153_153700


namespace baron_munchausen_claim_l153_153631

-- Given conditions and question:
def weight_partition_problem (weights : Finset ℕ) (h_card : weights.card = 50) (h_distinct : ∀ w ∈ weights,  1 ≤ w ∧ w ≤ 100) (h_sum_even : weights.sum id % 2 = 0) : Prop :=
  ¬(∃ (s1 s2 : Finset ℕ), s1 ∪ s2 = weights ∧ s1 ∩ s2 = ∅ ∧ s1.sum id = s2.sum id)

-- We need to prove that the above statement is true.
theorem baron_munchausen_claim :
  ∀ (weights : Finset ℕ), weights.card = 50 ∧ (∀ w ∈ weights, 1 ≤ w ∧ w ≤ 100) ∧ weights.sum id % 2 = 0 → weight_partition_problem weights (by sorry) (by sorry) (by sorry) :=
sorry

end baron_munchausen_claim_l153_153631


namespace second_store_earns_at_least_72000_more_l153_153719

-- Conditions as definitions in Lean.
def discount_price := 900000 -- 10% discount on 1 million yuan.
def full_price := 1000000 -- Full price for 1 million yuan without discount.

-- Prize calculation for the second department store.
def prize_first := 1000 * 5
def prize_second := 500 * 10
def prize_third := 200 * 20
def prize_fourth := 100 * 40
def prize_fifth := 10 * 1000

def total_prizes := prize_first + prize_second + prize_third + prize_fourth + prize_fifth

def second_store_net_income := full_price - total_prizes -- Net income after subtracting prizes.

-- The proof problem statement.
theorem second_store_earns_at_least_72000_more :
  second_store_net_income - discount_price >= 72000 := sorry

end second_store_earns_at_least_72000_more_l153_153719


namespace midpoint_C_l153_153206

variables (A B C : ℝ × ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (AC CB : ℝ)

def segment_division (A B C : ℝ × ℝ) (m n : ℝ) : Prop :=
  C = ((m * B.1 + n * A.1) / (m + n), (m * B.2 + n * A.2) / (m + n))

theorem midpoint_C :
  A = (-2, 1) →
  B = (4, 9) →
  AC = 2 * CB →
  segment_division A B C 2 1 →
  C = (2, 19 / 3) :=
by
  sorry

end midpoint_C_l153_153206


namespace xy_equation_solution_l153_153830

theorem xy_equation_solution (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 11980 / 121 :=
by
  sorry

end xy_equation_solution_l153_153830


namespace max_value_of_quadratic_expression_l153_153260

theorem max_value_of_quadratic_expression (s : ℝ) : ∃ x : ℝ, -3 * s^2 + 24 * s - 8 ≤ x ∧ x = 40 :=
sorry

end max_value_of_quadratic_expression_l153_153260


namespace find_second_number_l153_153069

theorem find_second_number 
  (k : ℕ)
  (h_k_is_1 : k = 1)
  (h_div_1657 : ∃ q1 : ℕ, 1657 = k * q1 + 10)
  (h_div_x : ∃ q2 : ℕ, ∀ x : ℕ, x = k * q2 + 7 → x = 1655) 
: ∃ x : ℕ, x = 1655 :=
by
  sorry

end find_second_number_l153_153069


namespace train_length_is_correct_l153_153316

noncomputable def length_of_train (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := (speed_train_kmph + speed_man_kmph)
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5 / 18)
  relative_speed_mps * (time_seconds : ℝ)

theorem train_length_is_correct :
  length_of_train 60 6 3 = 54.99 := 
by
  sorry

end train_length_is_correct_l153_153316


namespace number_of_male_rabbits_l153_153865

-- Definitions based on the conditions
def white_rabbits : ℕ := 12
def black_rabbits : ℕ := 9
def female_rabbits : ℕ := 8

-- The question and proof goal
theorem number_of_male_rabbits : 
  (white_rabbits + black_rabbits - female_rabbits) = 13 :=
by
  sorry

end number_of_male_rabbits_l153_153865


namespace min_value_complex_mod_one_l153_153259

/-- Given that the modulus of the complex number \( z \) is 1, prove that the minimum value of
    \( |z - 4|^2 + |z + 3 * Complex.I|^2 \) is \( 17 \). -/
theorem min_value_complex_mod_one (z : ℂ) (h : ‖z‖ = 1) : 
  ∃ α : ℝ, (‖z - 4‖^2 + ‖z + 3 * Complex.I‖^2) = 17 :=
sorry

end min_value_complex_mod_one_l153_153259


namespace domain_M_complement_domain_M_l153_153328

noncomputable def f (x : ℝ) : ℝ :=
  1 / Real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ :=
  Real.log (1 + x)

def M : Set ℝ :=
  {x | 1 - x > 0}

def N : Set ℝ :=
  {x | 1 + x > 0}

def complement_M : Set ℝ :=
  {x | 1 - x ≤ 0}

theorem domain_M :
  M = {x | x < 1} := by
  sorry

theorem complement_domain_M :
  complement_M = {x | x ≥ 1} := by
  sorry

end domain_M_complement_domain_M_l153_153328


namespace Nancy_hourly_wage_l153_153986

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l153_153986


namespace initial_pencils_sold_l153_153029

theorem initial_pencils_sold (x : ℕ) (P : ℝ)
  (h1 : 1 = 0.9 * (x * P))
  (h2 : 1 = 1.2 * (8.25 * P))
  : x = 11 :=
by sorry

end initial_pencils_sold_l153_153029


namespace find_n_l153_153695

theorem find_n :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 120 ∧ (n % 8 = 0) ∧ (n % 7 = 5) ∧ (n % 6 = 3) ∧ n = 208 := 
by {
  sorry
}

end find_n_l153_153695


namespace sides_ratio_of_arithmetic_sequence_l153_153461

theorem sides_ratio_of_arithmetic_sequence (A B C : ℝ) (a b c : ℝ) 
  (h_arith_sequence : (A = B - (B - C)) ∧ (B = C + (C - A))) 
  (h_angle_B : B = 60)  
  (h_cosine_rule : a^2 + c^2 - b^2 = 2 * a * c * (Real.cos B)) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end sides_ratio_of_arithmetic_sequence_l153_153461


namespace sufficient_but_not_necessary_l153_153271

theorem sufficient_but_not_necessary (x : ℝ) : 
  (1 < x ∧ x < 2) → (x > 0) ∧ ¬((x > 0) → (1 < x ∧ x < 2)) := 
by 
  sorry

end sufficient_but_not_necessary_l153_153271


namespace n_not_2_7_l153_153853

open Set

variable (M N : Set ℕ)

-- Define the given set M
def M_def : Prop := M = {1, 4, 7}

-- Define the condition M ∪ N = M
def union_condition : Prop := M ∪ N = M

-- The main statement to be proved
theorem n_not_2_7 (M_def : M = {1, 4, 7}) (union_condition : M ∪ N = M) : N ≠ {2, 7} :=
  sorry

end n_not_2_7_l153_153853


namespace line_parallel_not_passing_through_point_l153_153470

noncomputable def point_outside_line (A B C x0 y0 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (A * x0 + B * y0 + C = k)

theorem line_parallel_not_passing_through_point 
  (A B C x0 y0 : ℝ) (h : point_outside_line A B C x0 y0) :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, Ax + By + C + k = 0 → Ax_0 + By_0 + C + k ≠ 0) :=
sorry

end line_parallel_not_passing_through_point_l153_153470


namespace shift_down_equation_l153_153341

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f x - 3

theorem shift_down_equation : ∀ x : ℝ, g x = 2 * x := by
  sorry

end shift_down_equation_l153_153341


namespace inequality_for_positive_real_l153_153969

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end inequality_for_positive_real_l153_153969


namespace find_m_l153_153931

theorem find_m (A B : Set ℝ) (m : ℝ) (hA: A = {2, m}) (hB: B = {1, m^2}) (hU: A ∪ B = {1, 2, 3, 9}) : m = 3 :=
by 
  sorry

end find_m_l153_153931


namespace largest_digit_for_divisibility_l153_153022

theorem largest_digit_for_divisibility (N : ℕ) (h1 : N % 2 = 0) (h2 : (3 + 6 + 7 + 2 + N) % 3 = 0) : N = 6 :=
sorry

end largest_digit_for_divisibility_l153_153022


namespace extreme_point_a_zero_l153_153670

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end extreme_point_a_zero_l153_153670


namespace s9_s3_ratio_l153_153327

variable {a_n : ℕ → ℝ}
variable {s_n : ℕ → ℝ}
variable {a : ℝ}

-- Conditions
axiom h_s6_s3_ratio : s_n 6 / s_n 3 = 1 / 2

-- Theorem to prove
theorem s9_s3_ratio (h : s_n 3 = a) : s_n 9 / s_n 3 = 3 / 4 := 
sorry

end s9_s3_ratio_l153_153327


namespace rectangle_ratio_l153_153223

theorem rectangle_ratio (L B : ℕ) (hL : L = 250) (hB : B = 160) : L / B = 25 / 16 := by
  sorry

end rectangle_ratio_l153_153223


namespace op_neg2_3_l153_153378

def op (a b : ℤ) : ℤ := a^2 + 2 * a * b

theorem op_neg2_3 : op (-2) 3 = -8 :=
by
  -- proof
  sorry

end op_neg2_3_l153_153378


namespace derivative_at_neg_one_l153_153567

def f (x : ℝ) : ℝ := List.prod (List.map (λ k => (x^3 + k)) (List.range' 1 100))

theorem derivative_at_neg_one : deriv f (-1) = 3 * Nat.factorial 99 := by
  sorry

end derivative_at_neg_one_l153_153567


namespace divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l153_153297

open Nat

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (factors n).eraseDups.length

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

variables {p q m n : ℕ}
variables (hp : is_prime p) (hq : is_prime q) (hdist : p ≠ q) (hm : 0 ≤ m) (hn : 0 ≤ n)

-- a) Prove the number of divisors of pq is 4
theorem divisors_pq : num_divisors (p * q) = 4 :=
sorry

-- b) Prove the number of divisors of p^2 q is 6
theorem divisors_p2q : num_divisors (p^2 * q) = 6 :=
sorry

-- c) Prove the number of divisors of p^2 q^2 is 9
theorem divisors_p2q2 : num_divisors (p^2 * q^2) = 9 :=
sorry

-- d) Prove the number of divisors of p^m q^n is (m + 1)(n + 1)
theorem divisors_pmqn : num_divisors (p^m * q^n) = (m + 1) * (n + 1) :=
sorry

end divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l153_153297


namespace f_100_eq_11_l153_153566
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def f (n : ℕ) : ℕ := sum_of_digits (n^2 + 1)

def f_iter : ℕ → ℕ → ℕ
| 0,    n => f n
| k+1,  n => f (f_iter k n)

theorem f_100_eq_11 (n : ℕ) (h : n = 1990) : f_iter 100 n = 11 := by
  sorry

end f_100_eq_11_l153_153566


namespace tom_age_ratio_l153_153994

theorem tom_age_ratio (T N : ℕ) (h1 : T = 2 * (T / 2)) (h2 : T - N = 3 * ((T / 2) - 3 * N)) : T / N = 16 :=
  sorry

end tom_age_ratio_l153_153994


namespace moles_NaClO4_formed_l153_153703

-- Condition: Balanced chemical reaction
def reaction : Prop := ∀ (NaOH HClO4 NaClO4 H2O : ℕ), NaOH + HClO4 = NaClO4 + H2O

-- Given: 3 moles of NaOH and 3 moles of HClO4
def initial_moles_NaOH : ℕ := 3
def initial_moles_HClO4 : ℕ := 3

-- Question: number of moles of NaClO4 formed
def final_moles_NaClO4 : ℕ := 3

-- Proof Problem: Given the balanced chemical reaction and initial moles, prove the final moles of NaClO4
theorem moles_NaClO4_formed : reaction → initial_moles_NaOH = 3 → initial_moles_HClO4 = 3 → final_moles_NaClO4 = 3 :=
by
  intros
  sorry

end moles_NaClO4_formed_l153_153703


namespace parabola_point_dot_product_eq_neg4_l153_153119

-- Definition of the parabola
def is_parabola_point (A : ℝ × ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

-- Definition of the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Coordinates of origin
def origin : ℝ × ℝ := (0, 0)

-- Vector from origin to point A
def vector_OA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, A.2)

-- Vector from point A to the focus
def vector_AF (A : ℝ × ℝ) : ℝ × ℝ :=
  (focus.1 - A.1, focus.2 - A.2)

-- Theorem statement
theorem parabola_point_dot_product_eq_neg4 (A : ℝ × ℝ) 
  (hA : is_parabola_point A) 
  (h_dot : dot_product (vector_OA A) (vector_AF A) = -4) :
  A = (1, 2) ∨ A = (1, -2) :=
sorry

end parabola_point_dot_product_eq_neg4_l153_153119


namespace set_subtraction_M_N_l153_153942

-- Definitions
def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def B : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }
def M : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

-- Statement
theorem set_subtraction_M_N : (M \ N) = { x | x < 0 } := by
  sorry

end set_subtraction_M_N_l153_153942


namespace remainder_problem_l153_153484

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 15 = 11) (hy : y % 15 = 13) (hz : z % 15 = 14) : 
  (y + z - x) % 15 = 1 := 
by 
  sorry

end remainder_problem_l153_153484


namespace student_calls_out_2005th_l153_153028

theorem student_calls_out_2005th : 
  ∀ (n : ℕ), n = 2005 → ∃ k : ℕ, k ∈ [1, 2, 3, 4, 3, 2, 1] ∧ k = 1 := 
by
  sorry

end student_calls_out_2005th_l153_153028


namespace tangency_point_of_parabolas_l153_153189

theorem tangency_point_of_parabolas :
  ∃ (x y : ℝ), y = x^2 + 17 * x + 40 ∧ x = y^2 + 51 * y + 650 ∧ x = -7 ∧ y = -25 :=
by
  sorry

end tangency_point_of_parabolas_l153_153189


namespace trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l153_153450

namespace Trapezoid

def isosceles_trapezoid (AD BC : ℝ) := 
  AD = 20 ∧ BC = 12

def diagonal (AD BC : ℝ) (AC : ℝ) := 
  isosceles_trapezoid AD BC → AC = 8 * Real.sqrt 5

def leg (AD BC : ℝ) (CD : ℝ) := 
  isosceles_trapezoid AD BC → CD = 4 * Real.sqrt 5

theorem trapezoid_diagonal_is_8sqrt5 (AD BC AC : ℝ) : 
  diagonal AD BC AC :=
by
  intros
  sorry

theorem trapezoid_leg_is_4sqrt5 (AD BC CD : ℝ) : 
  leg AD BC CD :=
by
  intros
  sorry

end Trapezoid

end trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l153_153450


namespace cos_sq_sub_sin_sq_l153_153764

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end cos_sq_sub_sin_sq_l153_153764


namespace obtuse_scalene_triangle_l153_153918

theorem obtuse_scalene_triangle {k : ℕ} (h1 : 13 < k + 17) (h2 : 17 < 13 + k)
  (h3 : 13 < k + 17) (h4 : k ≠ 13) (h5 : k ≠ 17) 
  (h6 : 17^2 > 13^2 + k^2 ∨ k^2 > 13^2 + 17^2) 
  (h7 : (k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 22 ∨ 
        k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 ∨ k = 27 ∨ k = 28 ∨ k = 29)) :
  ∃ n, n = 14 := 
by
  sorry

end obtuse_scalene_triangle_l153_153918


namespace sum_sequence_a_b_eq_1033_l153_153570

def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

theorem sum_sequence_a_b_eq_1033 : 
  (a (b 1)) + (a (b 2)) + (a (b 3)) + (a (b 4)) + (a (b 5)) + 
  (a (b 6)) + (a (b 7)) + (a (b 8)) + (a (b 9)) + (a (b 10)) = 1033 := by
  sorry

end sum_sequence_a_b_eq_1033_l153_153570


namespace factorization_exists_l153_153524

-- Define the polynomial f(x)
def f (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 12

-- Definition for polynomial g(x)
def g (a : ℤ) (x : ℚ) : ℚ := x^2 + a*x + 3

-- Definition for polynomial h(x)
def h (b : ℤ) (x : ℚ) : ℚ := x^2 + b*x + 4

-- The main statement to prove
theorem factorization_exists :
  ∃ (a b : ℤ), (∀ x, f x = (g a x) * (h b x)) :=
by
  sorry

end factorization_exists_l153_153524


namespace A_B_days_together_l153_153486

variable (W : ℝ) -- total work
variable (x : ℝ) -- days A and B worked together
variable (A_B_rate : ℝ) -- combined work rate of A and B
variable (A_rate : ℝ) -- work rate of A
variable (B_days : ℝ) -- days A worked alone after B left

-- Conditions:
axiom condition1 : A_B_rate = W / 40
axiom condition2 : A_rate = W / 80
axiom condition3 : B_days = 6
axiom condition4 : (x * A_B_rate + B_days * A_rate = W)

-- We want to prove that x = 37:
theorem A_B_days_together : x = 37 :=
by
  sorry

end A_B_days_together_l153_153486


namespace paul_spent_252_dollars_l153_153109

noncomputable def total_cost_before_discounts : ℝ :=
  let dress_shirts := 4 * 15
  let pants := 2 * 40
  let suit := 150
  let sweaters := 2 * 30
  dress_shirts + pants + suit + sweaters

noncomputable def store_discount : ℝ := 0.20

noncomputable def coupon_discount : ℝ := 0.10

noncomputable def total_cost_after_store_discount : ℝ :=
  let initial_total := total_cost_before_discounts
  initial_total - store_discount * initial_total

noncomputable def final_total : ℝ :=
  let intermediate_total := total_cost_after_store_discount
  intermediate_total - coupon_discount * intermediate_total

theorem paul_spent_252_dollars :
  final_total = 252 := by
  sorry

end paul_spent_252_dollars_l153_153109


namespace find_f_2017_div_2_l153_153295

noncomputable def is_odd_function {X Y : Type*} [AddGroup X] [AddGroup Y] (f : X → Y) :=
  ∀ x : X, f (-x) = -f x

noncomputable def is_periodic_function {X Y : Type*} [AddGroup X] [AddGroup Y] (p : X) (f : X → Y) :=
  ∀ x : X, f (x + p) = f x

noncomputable def f : ℝ → ℝ 
| x => if -1 ≤ x ∧ x ≤ 0 then x * x + x else sorry

theorem find_f_2017_div_2 : f (2017 / 2) = 1 / 4 :=
by
  have h_odd : is_odd_function f := sorry
  have h_period : is_periodic_function 2 f := sorry
  unfold f
  sorry

end find_f_2017_div_2_l153_153295


namespace temperature_difference_l153_153168

theorem temperature_difference (T_high T_low : ℝ) (h1 : T_high = 8) (h2 : T_low = -2) : T_high - T_low = 10 :=
by
  sorry

end temperature_difference_l153_153168


namespace probability_two_balls_red_l153_153455

variables (total_balls red_balls blue_balls green_balls picked_balls : ℕ)

def probability_of_both_red
  (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2) : ℚ :=
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem probability_two_balls_red (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2)
  (h_prob : probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28) : 
  probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28 := 
sorry

end probability_two_balls_red_l153_153455


namespace y_intercept_of_line_l153_153039

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end y_intercept_of_line_l153_153039


namespace prove_arithmetic_sequence_l153_153057

def arithmetic_sequence (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2 * x + 3
| n => sorry

theorem prove_arithmetic_sequence {x : ℝ} (a : ℕ → ℝ)
  (h_terms : a 0 = x - 1 ∧ a 1 = x + 1 ∧ a 2 = 2 * x + 3)
  (h_arithmetic : ∀ n, a n = a 0 + n * (a 1 - a 0)) :
  x = 0 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end prove_arithmetic_sequence_l153_153057


namespace find_n_in_arithmetic_sequence_l153_153153

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 2) = a (n + 1) + d

theorem find_n_in_arithmetic_sequence (x : ℝ) (n : ℕ) (b : ℕ → ℝ)
  (h1 : b 1 = Real.exp x) 
  (h2 : b 2 = x) 
  (h3 : is_arithmetic_sequence b) : 
  b n = 1 + Real.exp x ↔ n = (1 + x) / (x - Real.exp x) :=
sorry

end find_n_in_arithmetic_sequence_l153_153153


namespace triangle_sides_proportional_l153_153344

theorem triangle_sides_proportional (a b c r d : ℝ)
  (h1 : 2 * r < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : a = 2 * r + d)
  (h5 : b = 2 * r + 2 * d)
  (h6 : c = 2 * r + 3 * d)
  (hr_pos : r > 0)
  (hd_pos : d > 0) :
  ∃ k : ℝ, k > 0 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * k :=
sorry

end triangle_sides_proportional_l153_153344


namespace sheila_saving_years_l153_153803

theorem sheila_saving_years 
  (initial_amount : ℝ) 
  (monthly_saving : ℝ) 
  (secret_addition : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : 
  initial_amount = 3000 ∧ 
  monthly_saving = 276 ∧ 
  secret_addition = 7000 ∧ 
  final_amount = 23248 → 
  years = 4 := 
sorry

end sheila_saving_years_l153_153803


namespace colin_speed_l153_153562

variable (B T Br C : ℝ)

def Bruce := B = 1
def Tony := T = 2 * B
def Brandon := Br = T / 3
def Colin := C = 6 * Br

theorem colin_speed : Bruce B → Tony B T → Brandon T Br → Colin Br C → C = 4 := by
  sorry

end colin_speed_l153_153562


namespace family_eggs_count_l153_153571

theorem family_eggs_count : 
  ∀ (initial_eggs parent_use child_use : ℝ) (chicken1 chicken2 chicken3 chicken4 : ℝ), 
    initial_eggs = 25 →
    parent_use = 7.5 + 2.5 →
    chicken1 = 2.5 →
    chicken2 = 3 →
    chicken3 = 4.5 →
    chicken4 = 1 →
    child_use = 1.5 + 0.5 →
    (initial_eggs - parent_use + (chicken1 + chicken2 + chicken3 + chicken4) - child_use) = 24 :=
by
  intros initial_eggs parent_use child_use chicken1 chicken2 chicken3 chicken4 
         h_initial_eggs h_parent_use h_chicken1 h_chicken2 h_chicken3 h_chicken4 h_child_use
  -- Proof goes here
  sorry

end family_eggs_count_l153_153571


namespace sample_size_correct_l153_153828

def sample_size (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ) (S : ℕ) : Prop :=
  sum_frequencies = 20 ∧ frequency_sum_ratio = 0.4 → S = 50

theorem sample_size_correct :
  ∀ (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ),
    sample_size sum_frequencies frequency_sum_ratio 50 :=
by
  intros sum_frequencies frequency_sum_ratio
  sorry

end sample_size_correct_l153_153828


namespace prob_white_ball_is_0_25_l153_153320

-- Let's define the conditions and the statement for the proof
variable (P_red P_white P_yellow : ℝ)

-- The given conditions 
def prob_red_or_white : Prop := P_red + P_white = 0.65
def prob_yellow_or_white : Prop := P_yellow + P_white = 0.6

-- The statement we want to prove
theorem prob_white_ball_is_0_25 (h1 : prob_red_or_white P_red P_white)
                               (h2 : prob_yellow_or_white P_yellow P_white) :
  P_white = 0.25 :=
sorry

end prob_white_ball_is_0_25_l153_153320


namespace find_a1_l153_153433

theorem find_a1 (a : ℕ → ℕ) (h1 : a 5 = 14) (h2 : ∀ n, a (n+1) - a n = n + 1) : a 1 = 0 :=
by
  sorry

end find_a1_l153_153433


namespace min_value_of_x3y2z_l153_153448

noncomputable def min_value_of_polynomial (x y z : ℝ) : ℝ :=
  x^3 * y^2 * z

theorem min_value_of_x3y2z
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1 / x + 1 / y + 1 / z = 9) :
  min_value_of_polynomial x y z = 1 / 46656 :=
sorry

end min_value_of_x3y2z_l153_153448


namespace weight_of_new_person_l153_153688

variable (avg_increase : ℝ) (n_persons : ℕ) (weight_replaced : ℝ)

theorem weight_of_new_person (h1 : avg_increase = 3.5) (h2 : n_persons = 8) (h3 : weight_replaced = 65) :
  let total_weight_increase := n_persons * avg_increase
  let weight_new := weight_replaced + total_weight_increase
  weight_new = 93 := by
  sorry

end weight_of_new_person_l153_153688


namespace geometric_sequence_product_l153_153526

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product 
  (h : is_geometric_sequence a r)
  (h_cond : a 4 * a 6 = 10) :
  a 2 * a 8 = 10 := 
sorry

end geometric_sequence_product_l153_153526


namespace cylinder_sphere_ratio_l153_153621

theorem cylinder_sphere_ratio (r R : ℝ) (h : 8 * r^2 = 4 * R^2) : R / r = Real.sqrt 2 :=
by
  sorry

end cylinder_sphere_ratio_l153_153621


namespace arithmetic_sequence_problem_l153_153139

variable (a : ℕ → ℝ) (d : ℝ) (m : ℕ)

noncomputable def a_seq := ∀ n, a n = a 1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : a 1 = 0)
  (h2 : d ≠ 0)
  (h3 : a m = a 1 + a 2 + a 3 + a 4 + a 5) :
  m = 11 :=
sorry

end arithmetic_sequence_problem_l153_153139


namespace part_a_part_b_l153_153492

noncomputable section

open Real

theorem part_a (x y z : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x-1)^2) + (y^2 / (y-1)^2) + (z^2 / (z-1)^2) ≥ 1 :=
sorry

theorem part_b : ∃ (infinitely_many : ℕ → (ℚ × ℚ × ℚ)), 
  ∀ n, ((infinitely_many n).1.1 ≠ 1) ∧ ((infinitely_many n).1.2 ≠ 1) ∧ ((infinitely_many n).2 ≠ 1) ∧ 
  ((infinitely_many n).1.1 * (infinitely_many n).1.2 * (infinitely_many n).2 = 1) ∧ 
  ((infinitely_many n).1.1^2 / ((infinitely_many n).1.1 - 1)^2 + 
   (infinitely_many n).1.2^2 / ((infinitely_many n).1.2 - 1)^2 + 
   (infinitely_many n).2^2 / ((infinitely_many n).2 - 1)^2 = 1) :=
sorry

end part_a_part_b_l153_153492


namespace circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l153_153967

-- Problem (1)
theorem circle_a_lt_8 (x y a : ℝ) (h : x^2 + y^2 - 4*x - 4*y + a = 0) : 
  a < 8 :=
by
  sorry

-- Problem (2)
theorem tangent_lines (a : ℝ) (h : a = -17) : 
  ∃ (k : ℝ), k * 7 - 6 - 7 * k = 0 ∧
  ((39 * k + 80 * (-7) - 207 = 0) ∨ (k = 7)) :=
by
  sorry

-- Problem (3)
theorem perpendicular_circle_intersection (x1 x2 y1 y2 a : ℝ) 
  (h1: 2 * x1 - y1 - 3 = 0) 
  (h2: 2 * x2 - y2 - 3 = 0) 
  (h3: x1 * x2 + y1 * y2 = 0) 
  (hpoly : 5 * x1 * x2 - 6 * (x1 + x2) + 9 = 0): 
  a = -6 / 5 :=
by
  sorry

end circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l153_153967


namespace club_membership_l153_153707

theorem club_membership:
  (∃ (committee : ℕ → Prop) (member_assign : (ℕ × ℕ) → ℕ → Prop),
    (∀ i, i < 5 → ∃! m, member_assign (i, m) 2) ∧
    (∀ i j, i < 5 ∧ j < 5 ∧ i ≠ j → ∃! m, m < 10 ∧ member_assign (i, j) m)
  ) → 
  ∃ n, n = 10 :=
by
  sorry

end club_membership_l153_153707


namespace min_le_mult_l153_153879

theorem min_le_mult {x y z m : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
    (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : m ≤ x * y^2 * z^3 :=
by
  sorry

end min_le_mult_l153_153879


namespace equivalent_fraction_l153_153200

theorem equivalent_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = -3) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 38 / 13 :=
by
  sorry

end equivalent_fraction_l153_153200


namespace deductible_increase_l153_153377

theorem deductible_increase (current_deductible : ℝ) (increase_fraction : ℝ) (next_year_deductible : ℝ) : 
  current_deductible = 3000 ∧ increase_fraction = 2 / 3 ∧ next_year_deductible = (1 + increase_fraction) * current_deductible →
  next_year_deductible - current_deductible = 2000 :=
by
  intros h
  sorry

end deductible_increase_l153_153377


namespace median_of_64_consecutive_integers_l153_153215

theorem median_of_64_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 64) (h2 : S = 8^4) :
  S / n = 64 :=
by
  -- to skip the proof
  sorry

end median_of_64_consecutive_integers_l153_153215


namespace simplify_f_l153_153368

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) (h : Real.sin (α - 3 / 2 * Real.pi) = 1 / 5) : f α = -1 / 5 := by
  sorry

end simplify_f_l153_153368


namespace isabel_camera_pics_l153_153091

-- Conditions
def phone_pics := 2
def albums := 3
def pics_per_album := 2

-- Define the total pictures and camera pictures
def total_pics := albums * pics_per_album
def camera_pics := total_pics - phone_pics

theorem isabel_camera_pics : camera_pics = 4 :=
by
  -- The goal is translated from the correct answer in step b)
  sorry

end isabel_camera_pics_l153_153091


namespace pentagon_square_ratio_l153_153015

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l153_153015


namespace possible_values_y_l153_153636

theorem possible_values_y (x : ℝ) (h : x^2 + 4 * (x / (x - 2))^2 = 45) : 
  ∃ y : ℝ, y = 2 ∨ y = 16 :=
sorry

end possible_values_y_l153_153636


namespace sqrt_trig_identity_l153_153885

theorem sqrt_trig_identity
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP: P = (Real.sin 2, Real.cos 2))
  (h_terminal: ∃ (θ : ℝ), P = (Real.cos θ, Real.sin θ)) :
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := 
sorry

end sqrt_trig_identity_l153_153885


namespace arrival_time_l153_153773

def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem arrival_time (departure_time : ℕ) (stop1 stop2 stop3 travel_hours : ℕ) (stops_total_time := stop1 + stop2 + stop3) (stops_total_hours := minutes_to_hours stops_total_time) : 
  departure_time = 7 → 
  stop1 = 25 → 
  stop2 = 10 → 
  stop3 = 25 → 
  travel_hours = 12 → 
  (departure_time + (travel_hours - stops_total_hours)) % 24 = 18 :=
by
  sorry

end arrival_time_l153_153773


namespace value_of_a_l153_153385

-- Define the variables and conditions as lean definitions/constants
variable (a b c : ℝ)
variable (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variable (h2 : a * 15 * 11 = 1)

-- Statement to prove
theorem value_of_a : a = 6 :=
by
  sorry

end value_of_a_l153_153385


namespace average_score_l153_153965

theorem average_score (avg1 avg2 : ℕ) (n1 n2 total_matches : ℕ) (total_avg : ℕ) 
  (h1 : avg1 = 60) 
  (h2 : avg2 = 70) 
  (h3 : n1 = 10) 
  (h4 : n2 = 15) 
  (h5 : total_matches = 25) 
  (h6 : total_avg = 66) :
  (( (avg1 * n1) + (avg2 * n2) ) / total_matches = total_avg) :=
by
  sorry

end average_score_l153_153965


namespace number_of_boxes_ordered_l153_153262

-- Definitions based on the conditions
def boxes_contain_matchboxes : Nat := 20
def matchboxes_contain_sticks : Nat := 300
def total_match_sticks : Nat := 24000

-- Statement of the proof problem
theorem number_of_boxes_ordered :
  (total_match_sticks / matchboxes_contain_sticks) / boxes_contain_matchboxes = 4 := 
sorry

end number_of_boxes_ordered_l153_153262


namespace snooker_tournament_l153_153251

theorem snooker_tournament : 
  ∀ (V G : ℝ),
    V + G = 320 →
    40 * V + 15 * G = 7500 →
    V ≥ 80 →
    G ≥ 100 →
    G - V = 104 :=
by
  intros V G h1 h2 h3 h4
  sorry

end snooker_tournament_l153_153251


namespace sum_of_squares_expressible_l153_153972

theorem sum_of_squares_expressible (a b c : ℕ) (h1 : c^2 = a^2 + b^2) : 
  ∃ x y : ℕ, x^2 + y^2 = c^2 + a*b ∧ ∃ u v : ℕ, u^2 + v^2 = c^2 - a*b :=
by
  sorry

end sum_of_squares_expressible_l153_153972


namespace jessica_total_payment_l153_153756

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end jessica_total_payment_l153_153756


namespace fraction_identity_l153_153037

theorem fraction_identity (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := 
by
  sorry

end fraction_identity_l153_153037


namespace regression_decrease_by_5_l153_153561

theorem regression_decrease_by_5 (x y : ℝ) (h : y = 2 - 2.5 * x) :
  y = 2 - 2.5 * (x + 2) → y ≠ 2 - 2.5 * x - 5 :=
by sorry

end regression_decrease_by_5_l153_153561


namespace find_triples_l153_153011

theorem find_triples (x y z : ℝ) 
  (h1 : (1/3 : ℝ) * min x y + (2/3 : ℝ) * max x y = 2017)
  (h2 : (1/3 : ℝ) * min y z + (2/3 : ℝ) * max y z = 2018)
  (h3 : (1/3 : ℝ) * min z x + (2/3 : ℝ) * max z x = 2019) :
  (x = 2019) ∧ (y = 2016) ∧ (z = 2019) :=
sorry

end find_triples_l153_153011


namespace each_girl_gets_2_dollars_after_debt_l153_153111

variable (Lulu_saved : ℕ)
variable (Nora_saved : ℕ)
variable (Tamara_saved : ℕ)
variable (debt : ℕ)
variable (remaining : ℕ)
variable (each_girl_share : ℕ)

-- Conditions
axiom Lulu_saved_cond : Lulu_saved = 6
axiom Nora_saved_cond : Nora_saved = 5 * Lulu_saved
axiom Nora_Tamara_relation : Nora_saved = 3 * Tamara_saved
axiom debt_cond : debt = 40

-- Question == Answer to prove
theorem each_girl_gets_2_dollars_after_debt (total_saved : ℕ) (remaining: ℕ) (each_girl_share: ℕ) :
  total_saved = Tamara_saved + Nora_saved + Lulu_saved →
  remaining = total_saved - debt →
  each_girl_share = remaining / 3 →
  each_girl_share = 2 := 
sorry

end each_girl_gets_2_dollars_after_debt_l153_153111


namespace min_distance_between_lines_t_l153_153732

theorem min_distance_between_lines_t (t : ℝ) :
  (∀ x y : ℝ, x + 2 * y + t^2 = 0) ∧ (∀ x y : ℝ, 2 * x + 4 * y + 2 * t - 3 = 0) →
  t = 1 / 2 := by
  sorry

end min_distance_between_lines_t_l153_153732


namespace molecular_weight_calculation_l153_153752

theorem molecular_weight_calculation :
  let atomic_weight_K := 39.10
  let atomic_weight_Br := 79.90
  let atomic_weight_O := 16.00
  let num_K := 1
  let num_Br := 1
  let num_O := 3
  let molecular_weight := (num_K * atomic_weight_K) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)
  molecular_weight = 167.00 :=
by
  sorry

end molecular_weight_calculation_l153_153752


namespace abs_ineq_solution_set_l153_153220

theorem abs_ineq_solution_set {x : ℝ} : |x + 1| - |x - 3| ≥ 2 ↔ x ≥ 2 :=
by
  sorry

end abs_ineq_solution_set_l153_153220


namespace temp_neg_represents_below_zero_l153_153548

-- Definitions based on the conditions in a)
def above_zero (x: ℤ) : Prop := x > 0
def below_zero (x: ℤ) : Prop := x < 0

-- Proof problem derived from c)
theorem temp_neg_represents_below_zero (t1 t2: ℤ) 
  (h1: above_zero t1) (h2: t1 = 10) 
  (h3: below_zero t2) (h4: t2 = -3) : 
  -t2 = 3 :=
by
  sorry

end temp_neg_represents_below_zero_l153_153548


namespace man_l153_153404

theorem man's_rate_in_still_water 
  (V_s V_m : ℝ)
  (with_stream : V_m + V_s = 24)  -- Condition 1
  (against_stream : V_m - V_s = 10) -- Condition 2
  : V_m = 17 := 
by
  sorry

end man_l153_153404


namespace g_88_value_l153_153287

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n m : ℕ) (h : n < m) : g n < g m
axiom g_multiplicative (m n : ℕ) : g (m * n) = g m * g n
axiom g_exponential_condition (m n : ℕ) (h : m ≠ n ∧ m ^ n = n ^ m) : g m = n ∨ g n = m

theorem g_88_value : g 88 = 7744 :=
sorry

end g_88_value_l153_153287


namespace valid_base6_number_2015_l153_153319

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end valid_base6_number_2015_l153_153319


namespace part1_part2_part3_l153_153330

/-- Proof for part (1): If the point P lies on the x-axis, then m = -1. -/
theorem part1 (m : ℝ) (hx : 3 * m + 3 = 0) : m = -1 := 
by {
  sorry
}

/-- Proof for part (2): If point P lies on a line passing through A(-5, 1) and parallel to the y-axis, 
then the coordinates of point P are (-5, -12). -/
theorem part2 (m : ℝ) (hy : 2 * m + 5 = -5) : (2 * m + 5, 3 * m + 3) = (-5, -12) := 
by {
  sorry
}

/-- Proof for part (3): If point P is moved 2 right and 3 up to point M, 
and point M lies in the third quadrant with a distance of 7 from the y-axis, then the coordinates of M are (-7, -15). -/
theorem part3 (m : ℝ) 
  (hc : 2 * m + 7 = -7)
  (config : 3 * m + 6 < 0) : (2 * m + 7, 3 * m + 6) = (-7, -15) := 
by {
  sorry
}

end part1_part2_part3_l153_153330


namespace general_formula_for_sequence_l153_153938

noncomputable def S := ℕ → ℚ
noncomputable def a := ℕ → ℚ

theorem general_formula_for_sequence (a : a) (S : S) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, S (n + 1) = (2 / 3) * a (n + 1) + 1 / 3) :
  ∀ n : ℕ, a n = 
  if n = 1 then 2 
  else -5 * (-2)^(n-2) := 
by 
  sorry

end general_formula_for_sequence_l153_153938


namespace parkway_school_students_l153_153517

theorem parkway_school_students (total_boys total_soccer soccer_boys_percentage girls_not_playing_soccer : ℕ)
  (h1 : total_boys = 320)
  (h2 : total_soccer = 250)
  (h3 : soccer_boys_percentage = 86)
  (h4 : girls_not_playing_soccer = 95)
  (h5 : total_soccer * soccer_boys_percentage / 100 = 215) :
  total_boys + total_soccer - (total_soccer * soccer_boys_percentage / 100) + girls_not_playing_soccer = 450 :=
by
  sorry

end parkway_school_students_l153_153517


namespace unique_cd_exists_l153_153933

open Real

theorem unique_cd_exists (h₀ : 0 < π / 2):
  ∃! (c d : ℝ), (0 < c) ∧ (c < π / 2) ∧ (0 < d) ∧ (d < π / 2) ∧ (c < d) ∧ 
  (sin (cos c) = c) ∧ (cos (sin d) = d) := sorry

end unique_cd_exists_l153_153933


namespace calculate_fraction_l153_153468

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end calculate_fraction_l153_153468


namespace num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l153_153181

-- Problem 1: Number of white and black balls
theorem num_white_black_balls (n m : ℕ) (h1 : n + m = 10)
  (h2 : (10 - m) = 4) : n = 4 ∧ m = 6 :=
by sorry

-- Problem 2: Probability of drawing exactly 2 black balls with replacement
theorem prob_2_black_balls (p_black_draw : ℕ → ℕ → ℚ)
  (h1 : ∀ n m, p_black_draw n m = (6/10)^(n-m) * (4/10)^m)
  (h2 : p_black_draw 2 3 = 54/125) : p_black_draw 2 3 = 54 / 125 :=
by sorry

-- Problem 3: Distribution and Expectation of number of black balls drawn without replacement
theorem dist_exp_black_balls (prob_X : ℕ → ℚ) (expect_X : ℚ)
  (h1 : prob_X 0 = 2/15) (h2 : prob_X 1 = 8/15) (h3 : prob_X 2 = 1/3)
  (h4 : expect_X = 6 / 5) : ∀ k, prob_X k = match k with
    | 0 => 2/15
    | 1 => 8/15
    | 2 => 1/3
    | _ => 0 :=
by sorry

end num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l153_153181


namespace area_of_garden_l153_153017

theorem area_of_garden :
  ∃ (short_posts long_posts : ℕ), short_posts + long_posts - 4 = 24 → long_posts = 3 * short_posts →
  ∃ (short_length long_length : ℕ), short_length = (short_posts - 1) * 5 → long_length = (long_posts - 1) * 5 →
  (short_length * long_length = 3000) :=
by {
  sorry
}

end area_of_garden_l153_153017


namespace three_digit_number_l153_153354

theorem three_digit_number (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9) 
  (h : 100 * a + 10 * b + c = 3 * (10 * (a + b) + c)) : 100 * a + 10 * b + c = 135 :=
  sorry

end three_digit_number_l153_153354


namespace g_value_at_2_l153_153599

theorem g_value_at_2 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2 - 2) : g 2 = 11 / 28 :=
sorry

end g_value_at_2_l153_153599


namespace solve_inequality_system_l153_153103

theorem solve_inequality_system (y : ℝ) :
  (2 * (y + 1) < 5 * y - 7) ∧ ((y + 2) / 2 < 5) ↔ (3 < y) ∧ (y < 8) := 
by
  sorry

end solve_inequality_system_l153_153103


namespace smallest_prime_perimeter_l153_153718

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

def is_prime_perimeter_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧ is_prime (a + b + c)

theorem smallest_prime_perimeter (a b c : ℕ) :
  (a = 5 ∧ a < b ∧ a < c ∧ is_prime_perimeter_scalene_triangle a b c) →
  (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l153_153718


namespace min_length_M_intersect_N_l153_153944

-- Define the sets M and N with the given conditions
def M (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2/3}
def N (n : ℝ) : Set ℝ := {x | n - 3/4 ≤ x ∧ x ≤ n}
def M_intersect_N (m n : ℝ) : Set ℝ := M m ∩ N n

-- Define the condition that M and N are subsets of [0, 1]
def in_interval (m n : ℝ) := (M m ⊆ {x | 0 ≤ x ∧ x ≤ 1}) ∧ (N n ⊆ {x | 0 ≤ x ∧ x ≤ 1})

-- Define the length of a set given by an interval [a, b]
def length_interval (a b : ℝ) := b - a

-- Define the length of the intersection of M and N
noncomputable def length_M_intersect_N (m n : ℝ) : ℝ :=
  let a := max m (n - 3/4)
  let b := min (m + 2/3) n
  length_interval a b

-- Prove that the minimum length of M ∩ N is 5/12
theorem min_length_M_intersect_N (m n : ℝ) (h : in_interval m n) : length_M_intersect_N m n = 5 / 12 :=
by
  sorry

end min_length_M_intersect_N_l153_153944


namespace ceil_y_squared_possibilities_l153_153185

theorem ceil_y_squared_possibilities (y : ℝ) (h : ⌈y⌉ = 15) : 
  ∃ n : ℕ, (n = 29) ∧ (∀ z : ℕ, ⌈y^2⌉ = z → (197 ≤ z ∧ z ≤ 225)) :=
by
  sorry

end ceil_y_squared_possibilities_l153_153185


namespace min_final_exam_score_l153_153403

theorem min_final_exam_score (q1 q2 q3 q4 final_exam : ℤ)
    (H1 : q1 = 90) (H2 : q2 = 85) (H3 : q3 = 77) (H4 : q4 = 96) :
    (1/2) * (q1 + q2 + q3 + q4) / 4 + (1/2) * final_exam ≥ 90 ↔ final_exam ≥ 93 :=
by
    sorry

end min_final_exam_score_l153_153403


namespace sum_of_squares_pentagon_greater_icosagon_l153_153256

noncomputable def compare_sum_of_squares (R : ℝ) : Prop :=
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  4 * a_20^2 < a_5^2

theorem sum_of_squares_pentagon_greater_icosagon (R : ℝ) : 
  compare_sum_of_squares R :=
  sorry

end sum_of_squares_pentagon_greater_icosagon_l153_153256


namespace parabola_sum_is_neg_fourteen_l153_153173

noncomputable def parabola_sum (a b c : ℝ) : ℝ := a + b + c

theorem parabola_sum_is_neg_fourteen :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = -(x + 3)^2 + 2) ∧
    ((-1)^2 = a * (-1 + 3)^2 + 6) ∧ 
    ((-3)^2 = a * (-3 + 3)^2 + 2) ∧
    (parabola_sum a b c = -14) :=
sorry

end parabola_sum_is_neg_fourteen_l153_153173


namespace a_plus_c_eq_neg_300_l153_153323

namespace Polynomials

variable {α : Type*} [LinearOrderedField α]

def f (a b x : α) := x^2 + a * x + b
def g (c d x : α) := x^2 + c * x + d

theorem a_plus_c_eq_neg_300 
  {a b c d : α}
  (h1 : ∀ x, f a b x ≥ -144) 
  (h2 : ∀ x, g c d x ≥ -144)
  (h3 : f a b 150 = -200) 
  (h4 : g c d 150 = -200)
  (h5 : ∃ x, (2*x + a = 0) ∧ g c d x = 0)
  (h6 : ∃ x, (2*x + c = 0) ∧ f a b x = 0) :
  a + c = -300 := 
sorry

end Polynomials

end a_plus_c_eq_neg_300_l153_153323


namespace product_eq_neg_one_l153_153063

theorem product_eq_neg_one (m b : ℚ) (hm : m = -2 / 3) (hb : b = 3 / 2) : m * b = -1 :=
by
  rw [hm, hb]
  sorry

end product_eq_neg_one_l153_153063


namespace probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l153_153898

noncomputable def probability_sum_is_multiple_of_3 : ℝ :=
  let total_events := 36
  let favorable_events := 12
  favorable_events / total_events

noncomputable def probability_sum_is_prime : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

noncomputable def probability_second_greater_than_first : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

theorem probability_sum_multiple_of_3_eq_one_third :
  probability_sum_is_multiple_of_3 = 1 / 3 :=
by sorry

theorem probability_sum_prime_eq_five_twelfths :
  probability_sum_is_prime = 5 / 12 :=
by sorry

theorem probability_second_greater_than_first_eq_five_twelfths :
  probability_second_greater_than_first = 5 / 12 :=
by sorry

end probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l153_153898


namespace vertex_angle_of_isosceles_with_angle_30_l153_153882

def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ a + b + c = 180

theorem vertex_angle_of_isosceles_with_angle_30 (a b c : ℝ) 
  (ha : isosceles_triangle a b c) 
  (h1 : a = 30 ∨ b = 30 ∨ c = 30) :
  (a = 30 ∨ b = 30 ∨ c = 30) ∨ (a = 120 ∨ b = 120 ∨ c = 120) := 
sorry

end vertex_angle_of_isosceles_with_angle_30_l153_153882


namespace cistern_problem_l153_153250

noncomputable def cistern_problem_statement : Prop :=
∀ (x : ℝ),
  (1 / 5 - 1 / x = 1 / 11.25) → x = 9

theorem cistern_problem : cistern_problem_statement :=
sorry

end cistern_problem_l153_153250


namespace third_angle_in_triangle_sum_of_angles_in_triangle_l153_153148

theorem third_angle_in_triangle (a b : ℝ) (h₁ : a = 50) (h₂ : b = 80) : 180 - a - b = 50 :=
by
  rw [h₁, h₂]
  norm_num

-- Adding this to demonstrate the constraint of the problem: Sum of angles in a triangle is 180°
theorem sum_of_angles_in_triangle (a b c : ℝ) (h₁: a + b + c = 180) : true :=
by
  trivial

end third_angle_in_triangle_sum_of_angles_in_triangle_l153_153148


namespace blue_pill_cost_l153_153429

theorem blue_pill_cost
  (days : ℕ)
  (total_cost : ℤ)
  (cost_diff : ℤ)
  (daily_cost : ℤ)
  (y : ℤ) : 
  days = 21 →
  total_cost = 966 →
  cost_diff = 2 →
  daily_cost = total_cost / days →
  daily_cost = 46 →
  2 * y - cost_diff = daily_cost →
  y = 24 := 
by
  intros days_eq total_cost_eq cost_diff_eq daily_cost_eq d_cost_eq daily_eq_46;
  sorry

end blue_pill_cost_l153_153429


namespace factor_expression_correct_l153_153734

variable (y : ℝ)

def expression := 4 * y * (y + 2) + 6 * (y + 2)

theorem factor_expression_correct : expression y = (y + 2) * (2 * (2 * y + 3)) :=
by
  sorry

end factor_expression_correct_l153_153734


namespace total_cost_football_games_l153_153875

-- Define the initial conditions
def games_this_year := 14
def games_last_year := 29
def price_this_year := 45
def price_lowest := 40
def price_highest := 65
def one_third_games_last_year := games_last_year / 3
def one_fourth_games_last_year := games_last_year / 4

-- Define the assertions derived from the conditions
def games_lowest_price := 9  -- rounded down from games_last_year / 3
def games_highest_price := 7  -- rounded down from games_last_year / 4
def remaining_games := games_last_year - (games_lowest_price + games_highest_price)

-- Define the costs calculation
def cost_this_year := games_this_year * price_this_year
def cost_lowest_price_games := games_lowest_price * price_lowest
def cost_highest_price_games := games_highest_price * price_highest
def total_cost := cost_this_year + cost_lowest_price_games + cost_highest_price_games

-- The theorem statement
theorem total_cost_football_games (h1 : games_lowest_price = 9) (h2 : games_highest_price = 7) 
  (h3 : cost_this_year = 630) (h4 : cost_lowest_price_games = 360) (h5 : cost_highest_price_games = 455) :
  total_cost = 1445 :=
by
  -- Since this is just the statement, we can simply put 'sorry' here.
  sorry

end total_cost_football_games_l153_153875


namespace mrs_randall_total_teaching_years_l153_153521

def years_teaching_third_grade : ℕ := 18
def years_teaching_second_grade : ℕ := 8

theorem mrs_randall_total_teaching_years : years_teaching_third_grade + years_teaching_second_grade = 26 :=
by
  sorry

end mrs_randall_total_teaching_years_l153_153521


namespace find_f_six_l153_153544

theorem find_f_six (f : ℕ → ℤ) (h : ∀ (x : ℕ), f (x + 1) = x^2 - 4) : f 6 = 21 :=
by
sorry

end find_f_six_l153_153544


namespace product_of_solutions_l153_153654

theorem product_of_solutions :
  let a := 2
  let b := 4
  let c := -6
  let discriminant := b^2 - 4*a*c
  ∃ (x₁ x₂ : ℝ), 2*x₁^2 + 4*x₁ - 6 = 0 ∧ 2*x₂^2 + 4*x₂ - 6 = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -3 :=
sorry

end product_of_solutions_l153_153654


namespace isosceles_triangle_apex_angle_l153_153180

theorem isosceles_triangle_apex_angle (a b c : ℝ) (ha : a = 40) (hb : b = 40) (hc : b = c) :
  (a + b + c = 180) → (c = 100 ∨ a = 40) :=
by
-- We start the proof and provide the conditions.
  sorry  -- Lean expects the proof here.

end isosceles_triangle_apex_angle_l153_153180


namespace parabola_equation_l153_153313

theorem parabola_equation (h k : ℝ) (p : ℝ × ℝ) (a b c : ℝ) :
  h = 3 ∧ k = -2 ∧ p = (4, -5) ∧
  (∀ x y : ℝ, y = a * (x - h) ^ 2 + k → p.2 = a * (p.1 - h) ^ 2 + k) →
  -(3:ℝ) = a ∧ 18 = b ∧ -29 = c :=
by sorry

end parabola_equation_l153_153313


namespace remainder_div_product_l153_153073

theorem remainder_div_product (P D D' D'' Q R Q' R' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = Q' * D' + R') 
  (h3 : Q' = Q'' * D'' + R'') :
  P % (D * D' * D'') = D * D' * R'' + D * R' + R := 
sorry

end remainder_div_product_l153_153073


namespace trinomial_identity_l153_153167

theorem trinomial_identity :
  let a := 23
  let b := 15
  let c := 7
  (a + b + c)^2 - (a^2 + b^2 + c^2) = 1222 :=
by
  let a := 23
  let b := 15
  let c := 7
  sorry

end trinomial_identity_l153_153167


namespace domain_of_f_l153_153136

noncomputable def f (x : ℝ) := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ x + 1 ≠ 1 ∧ 4 - x ≥ 0} = { x : ℝ | (-1 < x ∧ x ≤ 4) ∧ x ≠ 0 } :=
sorry

end domain_of_f_l153_153136


namespace intersection_is_as_expected_l153_153240

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

noncomputable def logarithmic_condition : Set ℝ :=
  { x | x > 0 ∧ x ≠ 1 }

noncomputable def intersection_of_sets : Set ℝ :=
  (quadratic_inequality_solution ∩ logarithmic_condition)

theorem intersection_is_as_expected :
  intersection_of_sets = { x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end intersection_is_as_expected_l153_153240


namespace tangent_line_through_origin_l153_153104

theorem tangent_line_through_origin (x : ℝ) (h₁ : 0 < x) (h₂ : ∀ x, ∃ y, y = 2 * Real.log x) (h₃ : ∀ x, y = 2 * Real.log x) :
  x = Real.exp 1 :=
sorry

end tangent_line_through_origin_l153_153104


namespace product_identity_l153_153880

theorem product_identity : 
  (7^3 - 1) / (7^3 + 1) * 
  (8^3 - 1) / (8^3 + 1) * 
  (9^3 - 1) / (9^3 + 1) * 
  (10^3 - 1) / (10^3 + 1) * 
  (11^3 - 1) / (11^3 + 1) = 
  133 / 946 := 
by
  sorry

end product_identity_l153_153880


namespace value_of_frac_l153_153685

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l153_153685


namespace largest_real_number_l153_153159

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end largest_real_number_l153_153159


namespace theta_range_l153_153790

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem theta_range (k : ℤ) (θ : ℝ) : 
  (2 * ↑k * π - 5 * π / 6 < θ ∧ θ < 2 * ↑k * π - π / 6) →
  (f (1 / (Real.sin θ)) + f (Real.cos (2 * θ)) < f π - f (1 / π)) :=
by
  intros h
  sorry

end theta_range_l153_153790


namespace george_slices_l153_153186

def num_small_pizzas := 3
def num_large_pizzas := 2
def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def slices_leftover := 10
def slices_per_person := 3
def total_pizza_slices := (num_small_pizzas * slices_per_small_pizza) + (num_large_pizzas * slices_per_large_pizza)
def slices_eaten := total_pizza_slices - slices_leftover
def G := 6 -- Slices George would like to eat

theorem george_slices :
  G + (G + 1) + ((G + 1) / 2) + (3 * slices_per_person) = slices_eaten :=
by
  sorry

end george_slices_l153_153186


namespace game_result_l153_153394

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_sequence : List ℕ := [2, 5, 6, 4, 3]
def jill_sequence : List ℕ := [1, 6, 3, 2, 5]

def calculate_score (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc + g x) 0

theorem game_result : calculate_score jack_sequence * calculate_score jill_sequence = 420 :=
by
  sorry

end game_result_l153_153394


namespace polar_bear_trout_l153_153172

/-
Question: How many buckets of trout does the polar bear eat daily?
Conditions:
  1. The polar bear eats some amount of trout and 0.4 bucket of salmon daily.
  2. The polar bear eats a total of 0.6 buckets of fish daily.
Answer: 0.2 buckets of trout daily.
-/

theorem polar_bear_trout (trout salmon total : ℝ) 
  (h1 : salmon = 0.4)
  (h2 : total = 0.6)
  (h3 : trout + salmon = total) :
  trout = 0.2 :=
by
  -- The proof will be provided here
  sorry

end polar_bear_trout_l153_153172


namespace vertices_form_vertical_line_l153_153659

theorem vertices_form_vertical_line (a b k d : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ∃ x, ∀ t : ℝ, ∃ y, (x = -b / (2 * a) ∧ y = - (b^2) / (4 * a) + k * t + d) :=
sorry

end vertices_form_vertical_line_l153_153659


namespace sum_arithmetic_sequence_l153_153634

theorem sum_arithmetic_sequence :
  let n := 21
  let a := 100
  let l := 120
  (n / 2) * (a + l) = 2310 :=
by
  -- define n, a, and l based on the conditions
  let n := 21
  let a := 100
  let l := 120
  -- state the goal
  have h : (n / 2) * (a + l) = 2310 := sorry
  exact h

end sum_arithmetic_sequence_l153_153634


namespace total_cars_l153_153980

theorem total_cars (Tommy_cars Jessie_cars : ℕ) (older_brother_cars : ℕ) 
  (h1 : Tommy_cars = 3) 
  (h2 : Jessie_cars = 3)
  (h3 : older_brother_cars = Tommy_cars + Jessie_cars + 5) : 
  Tommy_cars + Jessie_cars + older_brother_cars = 17 := by
  sorry

end total_cars_l153_153980


namespace graph_not_through_third_quadrant_l153_153299

theorem graph_not_through_third_quadrant (k : ℝ) (h_nonzero : k ≠ 0) (h_decreasing : k < 0) : 
  ¬(∃ x y : ℝ, y = k * x - k ∧ x < 0 ∧ y < 0) :=
sorry

end graph_not_through_third_quadrant_l153_153299


namespace monotonic_increase_interval_l153_153677

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_increase_interval : ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (Real.log x) / x :=
by sorry

end monotonic_increase_interval_l153_153677


namespace max_digit_product_l153_153466

theorem max_digit_product (N : ℕ) (digits : List ℕ) (h1 : 0 < N) (h2 : digits.sum = 23) (h3 : digits.prod < 433) : 
  digits.prod ≤ 432 :=
sorry

end max_digit_product_l153_153466


namespace total_amount_paid_l153_153514

/-- Conditions -/
def days_in_may : Nat := 31
def rate_per_day : ℚ := 0.5
def days_book1_borrowed : Nat := 20
def days_book2_borrowed : Nat := 31
def days_book3_borrowed : Nat := 31

/-- Question and Proof -/
theorem total_amount_paid : rate_per_day * (days_book1_borrowed + days_book2_borrowed + days_book3_borrowed) = 41 := by
  sorry

end total_amount_paid_l153_153514


namespace find_two_digit_numbers_l153_153945

def first_two_digit_number (x y : ℕ) : ℕ := 10 * x + y
def second_two_digit_number (x y : ℕ) : ℕ := 10 * (x + 5) + y

theorem find_two_digit_numbers :
  ∃ (x_2 y : ℕ), 
  (first_two_digit_number x_2 y = x_2^2 + x_2 * y + y^2) ∧ 
  (second_two_digit_number x_2 y = (x_2 + 5)^2 + (x_2 + 5) * y + y^2) ∧ 
  (second_two_digit_number x_2 y - first_two_digit_number x_2 y = 50) ∧ 
  (y = 1 ∨ y = 3) := 
sorry

end find_two_digit_numbers_l153_153945


namespace equation_of_line_AB_l153_153869

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def on_circle (C : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) ^ 2 + P.2 ^ 2 = r ^ 2

theorem equation_of_line_AB : 
  ∃ A B : ℝ × ℝ, 
    is_midpoint (2, -1) A B ∧ 
    on_circle (1, 0) 5 A ∧ 
    on_circle (1, 0) 5 B ∧ 
    ∀ x y : ℝ, (x - y - 3 = 0) ∧ 
    ∃ t : ℝ, ∃ u : ℝ, (t - u - 3 = 0) := 
sorry

end equation_of_line_AB_l153_153869


namespace cos_sin_equation_solution_l153_153477

noncomputable def solve_cos_sin_equation (x : ℝ) (n : ℤ) : Prop :=
  let lhs := (Real.cos x) / (Real.sqrt 3)
  let rhs := Real.sqrt ((1 - (Real.cos (2*x)) - 2 * (Real.sin x)^3) / (6 * Real.sin x - 2))
  (lhs = rhs) ∧ (Real.cos x ≥ 0)

theorem cos_sin_equation_solution:
  (∃ (x : ℝ) (n : ℤ), solve_cos_sin_equation x n) ↔ 
  ∃ (n : ℤ), (x = (π / 2) + 2 * π * n) ∨ (x = (π / 6) + 2 * π * n) :=
by
  sorry

end cos_sin_equation_solution_l153_153477


namespace man_swims_distance_back_l153_153234

def swimming_speed_still_water : ℝ := 8
def speed_of_water : ℝ := 4
def time_taken_against_current : ℝ := 2
def distance_swum : ℝ := 8

theorem man_swims_distance_back :
  (distance_swum = (swimming_speed_still_water - speed_of_water) * time_taken_against_current) :=
by
  -- The proof will be filled in later.
  sorry

end man_swims_distance_back_l153_153234


namespace mean_age_of_seven_friends_l153_153778

theorem mean_age_of_seven_friends 
  (mean_age_group1: ℕ)
  (mean_age_group2: ℕ)
  (n1: ℕ)
  (n2: ℕ)
  (total_friends: ℕ) :
  mean_age_group1 = 147 → 
  mean_age_group2 = 161 →
  n1 = 3 → 
  n2 = 4 →
  total_friends = 7 →
  (mean_age_group1 * n1 + mean_age_group2 * n2) / total_friends = 155 := by
  sorry

end mean_age_of_seven_friends_l153_153778


namespace angle_between_hands_at_3_40_l153_153583

def degrees_per_minute_minute_hand := 360 / 60
def minutes_passed := 40
def degrees_minute_hand := degrees_per_minute_minute_hand * minutes_passed -- 240 degrees

def degrees_per_hour_hour_hand := 360 / 12
def hours_passed := 3
def degrees_hour_hand_at_hour := degrees_per_hour_hour_hand * hours_passed -- 90 degrees

def degrees_per_minute_hour_hand := degrees_per_hour_hour_hand / 60
def degrees_hour_hand_additional := degrees_per_minute_hour_hand * minutes_passed -- 20 degrees

def total_degrees_hour_hand := degrees_hour_hand_at_hour + degrees_hour_hand_additional -- 110 degrees

def expected_angle_between_hands := 130

theorem angle_between_hands_at_3_40
  (h1: degrees_minute_hand = 240)
  (h2: total_degrees_hour_hand = 110):
  (degrees_minute_hand - total_degrees_hour_hand = expected_angle_between_hands) :=
by
  sorry

end angle_between_hands_at_3_40_l153_153583


namespace harmonic_arithmetic_sequence_common_difference_l153_153909

theorem harmonic_arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * d)) →
  (∀ n, a n = a 1 + (n - 1) * d) →
  (a 1 = 1) →
  (d ≠ 0) →
  (∃ k, ∀ n, S n / S (2 * n) = k) →
  d = 2 :=
by
  sorry

end harmonic_arithmetic_sequence_common_difference_l153_153909


namespace find_incorrect_value_l153_153601

theorem find_incorrect_value (n : ℕ) (mean_initial mean_correct : ℕ) (wrongly_copied correct_value incorrect_value : ℕ) 
  (h1 : n = 30) 
  (h2 : mean_initial = 150) 
  (h3 : mean_correct = 151) 
  (h4 : correct_value = 165) 
  (h5 : n * mean_initial = 4500) 
  (h6 : n * mean_correct = 4530) 
  (h7 : n * mean_correct - n * mean_initial = 30) 
  (h8 : correct_value - (n * mean_correct - n * mean_initial) = incorrect_value) : 
  incorrect_value = 135 :=
by
  sorry

end find_incorrect_value_l153_153601


namespace find_x_if_alpha_beta_eq_4_l153_153314

def alpha (x : ℝ) : ℝ := 4 * x + 9
def beta (x : ℝ) : ℝ := 9 * x + 6

theorem find_x_if_alpha_beta_eq_4 :
  (∃ x : ℝ, alpha (beta x) = 4 ∧ x = -29 / 36) :=
by
  sorry

end find_x_if_alpha_beta_eq_4_l153_153314


namespace find_OH_squared_l153_153608

variables {O H : Type} {a b c R : ℝ}

-- Given conditions
def is_circumcenter (O : Type) (ABC : Type) := true -- Placeholder definition
def is_orthocenter (H : Type) (ABC : Type) := true -- Placeholder definition
def circumradius (O : Type) (R : ℝ) := true -- Placeholder definition
def sides_squared_sum (a b c : ℝ) := a^2 + b^2 + c^2

-- The theorem to be proven
theorem find_OH_squared (O H : Type) (a b c : ℝ) (R : ℝ) 
  (circ : is_circumcenter O ABC) 
  (orth: is_orthocenter H ABC) 
  (radius : circumradius O R) 
  (terms_sum : sides_squared_sum a b c = 50)
  (R_val : R = 10) 
  : OH^2 = 850 := 
sorry

end find_OH_squared_l153_153608


namespace area_of_mirror_l153_153040

theorem area_of_mirror (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) (mirror_area : ℝ) :
  outer_width = 70 → outer_height = 100 → frame_width = 15 → mirror_area = (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width) → mirror_area = 2800 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h4]
  sorry

end area_of_mirror_l153_153040


namespace problem_statement_l153_153519

theorem problem_statement :
  ∀ (x : ℝ),
    (5 * x - 10 = 15 * x + 5) →
    (5 * (x + 3) = 15 / 2) :=
by
  intros x h
  sorry

end problem_statement_l153_153519


namespace chickens_in_zoo_l153_153456

theorem chickens_in_zoo (c e : ℕ) (h_legs : 2 * c + 4 * e = 66) (h_heads : c + e = 24) : c = 15 :=
by
  sorry

end chickens_in_zoo_l153_153456


namespace phosphorus_atoms_l153_153645

theorem phosphorus_atoms (x : ℝ) : 122 = 26.98 + 30.97 * x + 64 → x = 1 := by
sorry

end phosphorus_atoms_l153_153645


namespace number_of_tangents_l153_153903

-- Define the points and conditions
variable (A B : ℝ × ℝ)
variable (dist_AB : dist A B = 8)
variable (radius_A : ℝ := 3)
variable (radius_B : ℝ := 2)

-- The goal
theorem number_of_tangents (dist_condition : dist A B = 8) : 
  ∃ n, n = 2 :=
by
  -- skipping the proof
  sorry

end number_of_tangents_l153_153903
