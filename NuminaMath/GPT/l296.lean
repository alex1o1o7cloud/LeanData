import Mathlib

namespace min_roots_sin_eq_zero_l296_296634

open Real

theorem min_roots_sin_eq_zero (k0 k1 k2 : ℕ) (h : k0 < k1 ∧ k1 < k2) (A1 A2 : ℝ) :
  ∃ x ∈ Ico 0 (2 * π), 
  (sin (k0 * x) + A1 * sin (k1 * x) + A2 * sin (k2 * x) = 0) :=
sorry

end min_roots_sin_eq_zero_l296_296634


namespace inequality_my_problem_l296_296953

theorem inequality_my_problem (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a * b + b * c + c * a = 1) :
  (Real.sqrt ((1 / a) + 6 * b)) + (Real.sqrt ((1 / b) + 6 * c)) + (Real.sqrt ((1 / c) + 6 * a)) ≤ (1 / (a * b * c)) :=
  sorry

end inequality_my_problem_l296_296953


namespace solve_p_l296_296413

theorem solve_p (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 2 * p + 5 * q = 8) : 
  p = 11 / 19 :=
by
  sorry

end solve_p_l296_296413


namespace x_must_be_negative_l296_296505

theorem x_must_be_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 :=
by 
  sorry

end x_must_be_negative_l296_296505


namespace chip_sheets_per_pack_l296_296366

noncomputable def sheets_per_pack (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) 
                                  (weeks : ℕ) (packs : ℕ) : ℕ :=
(pages_per_day * days_per_week * classes * weeks) / packs

theorem chip_sheets_per_pack :
  sheets_per_pack 2 5 5 6 3 = 100 :=
sorry

end chip_sheets_per_pack_l296_296366


namespace abc_one_eq_sum_l296_296245

theorem abc_one_eq_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b * c = 1) :
  (a^2 * b^2) / ((a^2 + b * c) * (b^2 + a * c))
  + (a^2 * c^2) / ((a^2 + b * c) * (c^2 + a * b))
  + (b^2 * c^2) / ((b^2 + a * c) * (c^2 + a * b))
  = 1 / (a^2 + 1 / a) + 1 / (b^2 + 1 / b) + 1 / (c^2 + 1 / c) := by
  sorry

end abc_one_eq_sum_l296_296245


namespace problem_may_not_be_equal_l296_296771

-- Define the four pairs of expressions
def expr_A (a b : ℕ) := (a + b) = (b + a)
def expr_B (a : ℕ) := (3 * a) = (a + a + a)
def expr_C (a b : ℕ) := (3 * (a + b)) ≠ (3 * a + b)
def expr_D (a : ℕ) := (a ^ 3) = (a * a * a)

-- State the theorem stating that the expression in condition C may not be equal
theorem problem_may_not_be_equal (a b : ℕ) : (3 * (a + b)) ≠ (3 * a + b) :=
by
  sorry

end problem_may_not_be_equal_l296_296771


namespace blue_to_red_ratio_l296_296989

-- Define the conditions as given in the problem
def initial_red_balls : ℕ := 16
def lost_red_balls : ℕ := 6
def bought_yellow_balls : ℕ := 32
def total_balls_after_events : ℕ := 74

-- Based on the conditions, we define the remaining red balls and the total balls equation
def remaining_red_balls := initial_red_balls - lost_red_balls

-- Suppose B is the number of blue balls
def blue_balls (B : ℕ) : Prop :=
  remaining_red_balls + B + bought_yellow_balls = total_balls_after_events

-- Now, state the theorem to prove the ratio of blue balls to red balls is 16:5
theorem blue_to_red_ratio (B : ℕ) (h : blue_balls B) : B = 32 → B / remaining_red_balls = 16 / 5 :=
by
  intro B_eq
  subst B_eq
  have h1 : remaining_red_balls = 10 := rfl
  have h2 : 32 / 10  = 16 / 5 := by rfl
  exact h2

-- Note: The proof itself is skipped, so the statement is left with sorry.

end blue_to_red_ratio_l296_296989


namespace max_students_distribute_pens_pencils_l296_296633

noncomputable def gcd_example : ℕ :=
  Nat.gcd 1340 1280

theorem max_students_distribute_pens_pencils : gcd_example = 20 :=
sorry

end max_students_distribute_pens_pencils_l296_296633


namespace count_even_digits_in_512_base_7_l296_296945

def base7_representation (n : ℕ) : ℕ := 
  sorry  -- Assuming this function correctly computes the base-7 representation of a natural number

def even_digits_count (n : ℕ) : ℕ :=
  sorry  -- Assuming this function correctly counts the even digits in the base-7 representation

theorem count_even_digits_in_512_base_7 : 
  even_digits_count (base7_representation 512) = 0 :=
by
  sorry

end count_even_digits_in_512_base_7_l296_296945


namespace Δy_over_Δx_l296_296223

-- Conditions
def f (x : ℝ) : ℝ := 2 * x^2 - 4
def y1 : ℝ := f 1
def y2 (Δx : ℝ) : ℝ := f (1 + Δx)
def Δy (Δx : ℝ) : ℝ := y2 Δx - y1

-- Theorem statement
theorem Δy_over_Δx (Δx : ℝ) : Δy Δx / Δx = 4 + 2 * Δx := by
  sorry

end Δy_over_Δx_l296_296223


namespace only_one_true_l296_296222

-- Definitions based on conditions
def line := Type
def plane := Type
def parallel (m n : line) : Prop := sorry
def perpendicular (m n : line) : Prop := sorry
def subset (m : line) (alpha : plane) : Prop := sorry

-- Propositions derived from conditions
def prop1 (m n : line) (alpha : plane) : Prop := parallel m alpha ∧ parallel n alpha → ¬ parallel m n
def prop2 (m n : line) (alpha : plane) : Prop := perpendicular m alpha ∧ perpendicular n alpha → parallel m n
def prop3 (m n : line) (alpha beta : plane) : Prop := parallel alpha beta ∧ subset m alpha ∧ subset n beta → parallel m n
def prop4 (m n : line) (alpha beta : plane) : Prop := perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular m alpha → perpendicular n beta

-- Theorem statement that only one proposition is true
theorem only_one_true (m n : line) (alpha beta : plane) :
  (prop1 m n alpha = false) ∧
  (prop2 m n alpha = true) ∧
  (prop3 m n alpha beta = false) ∧
  (prop4 m n alpha beta = false) :=
by sorry

end only_one_true_l296_296222


namespace division_of_repeating_decimal_l296_296321

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l296_296321


namespace find_boys_l296_296230

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end find_boys_l296_296230


namespace problem_a_problem_b_problem_c_problem_d_l296_296474

-- Problem a
theorem problem_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 :=
by sorry

-- Problem b
theorem problem_b (a : ℝ) : (2 * a + 3) * (2 * a - 3) = 4 * a^2 - 9 :=
by sorry

-- Problem c
theorem problem_c (m n : ℝ) : (m^3 - n^5) * (n^5 + m^3) = m^6 - n^10 :=
by sorry

-- Problem d
theorem problem_d (m n : ℝ) : (3 * m^2 - 5 * n^2) * (3 * m^2 + 5 * n^2) = 9 * m^4 - 25 * n^4 :=
by sorry

end problem_a_problem_b_problem_c_problem_d_l296_296474


namespace chimes_1000_on_march_7_l296_296067

theorem chimes_1000_on_march_7 : 
  ∀ (initial_time : Nat) (start_date : Nat) (chimes_before_noon : Nat) 
  (chimes_per_day : Nat) (target_chime : Nat) (final_date : Nat),
  initial_time = 10 * 60 + 15 ∧
  start_date = 26 ∧
  chimes_before_noon = 25 ∧
  chimes_per_day = 103 ∧
  target_chime = 1000 ∧
  final_date = start_date + (target_chime - chimes_before_noon) / chimes_per_day ∧
  (target_chime - chimes_before_noon) % chimes_per_day ≤ chimes_per_day
  → final_date = 7 := 
by
  intros
  sorry

end chimes_1000_on_march_7_l296_296067


namespace eight_div_repeat_three_l296_296319

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l296_296319


namespace cousins_initial_money_l296_296645

theorem cousins_initial_money (x : ℕ) :
  let Carmela_initial := 7
  let num_cousins := 4
  let gift_each := 1
  Carmela_initial - num_cousins * gift_each = x + gift_each →
  x = 2 :=
by
  intro h
  sorry

end cousins_initial_money_l296_296645


namespace highest_nitrogen_percentage_l296_296056

-- Define molar masses for each compound
def molar_mass_NH2OH : Float := 33.0
def molar_mass_NH4NO2 : Float := 64.1 
def molar_mass_N2O3 : Float := 76.0
def molar_mass_NH4NH2CO2 : Float := 78.1

-- Define mass of nitrogen atoms
def mass_of_nitrogen : Float := 14.0

-- Define the percentage calculations
def percentage_NH2OH : Float := (mass_of_nitrogen / molar_mass_NH2OH) * 100.0
def percentage_NH4NO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NO2) * 100.0
def percentage_N2O3 : Float := (2 * mass_of_nitrogen / molar_mass_N2O3) * 100.0
def percentage_NH4NH2CO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NH2CO2) * 100.0

-- Define the proof problem
theorem highest_nitrogen_percentage : percentage_NH4NO2 > percentage_NH2OH ∧
                                      percentage_NH4NO2 > percentage_N2O3 ∧
                                      percentage_NH4NO2 > percentage_NH4NH2CO2 :=
by 
  sorry

end highest_nitrogen_percentage_l296_296056


namespace hypotenuse_length_right_triangle_l296_296559

theorem hypotenuse_length_right_triangle :
  ∃ (x : ℝ), (x > 7) ∧ ((x - 7)^2 + x^2 = (x + 2)^2) ∧ (x + 2 = 17) :=
by {
  sorry
}

end hypotenuse_length_right_triangle_l296_296559


namespace min_value_a1_plus_a7_l296_296830

theorem min_value_a1_plus_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a n > 0) 
  (h2 : ∀ n, a (n+1) = a n * r) 
  (h3 : a 3 * a 5 = 64) : 
  a 1 + a 7 ≥ 16 := 
sorry

end min_value_a1_plus_a7_l296_296830


namespace fruit_problem_l296_296637

theorem fruit_problem :
  let apples_initial := 7
  let oranges_initial := 8
  let mangoes_initial := 15
  let grapes_initial := 12
  let strawberries_initial := 5
  let apples_taken := 3
  let oranges_taken := 4
  let mangoes_taken := 4
  let grapes_taken := 7
  let strawberries_taken := 3
  let apples_remaining := apples_initial - apples_taken
  let oranges_remaining := oranges_initial - oranges_taken
  let mangoes_remaining := mangoes_initial - mangoes_taken
  let grapes_remaining := grapes_initial - grapes_taken
  let strawberries_remaining := strawberries_initial - strawberries_taken
  let total_remaining := apples_remaining + oranges_remaining + mangoes_remaining + grapes_remaining + strawberries_remaining
  let total_taken := apples_taken + oranges_taken + mangoes_taken + grapes_taken + strawberries_taken
  total_remaining = 26 ∧ total_taken = 21 := by
    sorry

end fruit_problem_l296_296637


namespace g_1993_at_4_l296_296994

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → ℚ → ℚ
  | 0, x     => x
  | (n+1), x => g (g_n n x)

theorem g_1993_at_4 : g_n 1993 4 = 11 / 20 :=
by
  sorry

end g_1993_at_4_l296_296994


namespace cindy_olaf_earnings_l296_296783
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end cindy_olaf_earnings_l296_296783


namespace find_a_if_lines_perpendicular_l296_296958

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end find_a_if_lines_perpendicular_l296_296958


namespace study_days_needed_l296_296925

theorem study_days_needed
    (chapters : ℕ) (worksheets : ℕ)
    (hours_per_chapter : ℕ) (hours_per_worksheet : ℕ)
    (max_hours_per_day : ℕ)
    (break_minutes_per_hour : ℕ) (snack_breaks_per_day : ℕ)
    (snack_break_minutes : ℕ) (lunch_break_minutes : ℕ) :
    chapters = 2 →
    worksheets = 4 →
    hours_per_chapter = 3 →
    hours_per_worksheet = 1.5 →
    max_hours_per_day = 4 →
    break_minutes_per_hour = 10 →
    snack_breaks_per_day = 3 →
    snack_break_minutes = 10 →
    lunch_break_minutes = 30 →
    (15 / 4).ceil = 4 := by 
  sorry

end study_days_needed_l296_296925


namespace expression_value_l296_296435

theorem expression_value (x : ℤ) (hx : x = 1729) : abs (abs (abs x + x) + abs x) + x = 6916 :=
by
  rw [hx]
  sorry

end expression_value_l296_296435


namespace study_days_l296_296926

theorem study_days (chapters worksheets : ℕ) (chapter_hours worksheet_hours daily_study_hours hourly_break
                     snack_breaks_count snack_break time_lunch effective_hours : ℝ)
  (h1 : chapters = 2) 
  (h2 : worksheets = 4) 
  (h3 : chapter_hours = 3) 
  (h4 : worksheet_hours = 1.5) 
  (h5 : daily_study_hours = 4) 
  (h6 : hourly_break = 10 / 60) 
  (h7 : snack_breaks_count = 3) 
  (h8 : snack_break = 10 / 60) 
  (h9 : time_lunch = 30 / 60)
  (h10 : effective_hours = daily_study_hours - (hourly_break * (daily_study_hours - 1)) - (snack_breaks_count * snack_break) - time_lunch)
  : (chapters * chapter_hours + worksheets * worksheet_hours) / effective_hours = 4.8 :=
by
  sorry

end study_days_l296_296926


namespace revenue_increase_l296_296346

theorem revenue_increase (R : ℕ) (r2000 r2003 r2005 : ℝ) (h1 : r2003 = r2000 * 1.50) (h2 : r2005 = r2000 * 1.80) :
  ((r2005 - r2003) / r2003) * 100 = 20 :=
by sorry

end revenue_increase_l296_296346


namespace probability_yellow_chalk_is_three_fifths_l296_296828

open Nat

theorem probability_yellow_chalk_is_three_fifths
  (yellow_chalks : ℕ) (red_chalks : ℕ) (total_chalks : ℕ)
  (h_yellow : yellow_chalks = 3) (h_red : red_chalks = 2) (h_total : total_chalks = yellow_chalks + red_chalks) :
  (yellow_chalks : ℚ) / (total_chalks : ℚ) = 3 / 5 := by
  sorry

end probability_yellow_chalk_is_three_fifths_l296_296828


namespace total_coins_Zain_l296_296337

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l296_296337


namespace problem_l296_296977

theorem problem (x y : ℚ) (h1 : x + y = 10 / 21) (h2 : x - y = 1 / 63) : 
  x^2 - y^2 = 10 / 1323 := 
by 
  sorry

end problem_l296_296977


namespace prob_product_greater_than_10_l296_296114

theorem prob_product_greater_than_10 :
  let s := {1, 2, 3, 4}
  let pairs := (s.product s).filter (λ x, x.fst < x.snd)
  let favorable pairs := pairs.filter (λ x, x.fst * x.snd > 10)
  let total_pairs := pairs.length
  let num_favorable := favorable_pairs.length
  (num_favorable : ℝ) / (total_pairs : ℝ) = 1 / 6 := by
begin
    sorry
end

end prob_product_greater_than_10_l296_296114


namespace total_cost_mulch_l296_296636

-- Define the conditions
def tons_to_pounds (tons : ℕ) : ℕ := tons * 2000

def price_per_pound : ℝ := 2.5

-- Define the statement to prove
theorem total_cost_mulch (mulch_in_tons : ℕ) (h₁ : mulch_in_tons = 3) : 
  tons_to_pounds mulch_in_tons * price_per_pound = 15000 :=
by
  -- The proof would normally go here.
  sorry

end total_cost_mulch_l296_296636


namespace solution_to_absolute_value_equation_l296_296606

theorem solution_to_absolute_value_equation (x : ℝ) : 
    abs x - 2 - abs (-1) = 2 ↔ x = 5 ∨ x = -5 :=
by
  sorry

end solution_to_absolute_value_equation_l296_296606


namespace amc_problem_l296_296677

theorem amc_problem (a b : ℕ) (h : ∀ n : ℕ, 0 < n → a^n + n ∣ b^n + n) : a = b :=
sorry

end amc_problem_l296_296677


namespace max_expr_value_l296_296274

noncomputable def expr (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_expr_value : 
  ∃ (a b c d : ℝ),
    a ∈ Set.Icc (-5 : ℝ) 5 ∧
    b ∈ Set.Icc (-5 : ℝ) 5 ∧
    c ∈ Set.Icc (-5 : ℝ) 5 ∧
    d ∈ Set.Icc (-5 : ℝ) 5 ∧
    expr a b c d = 110 :=
by
  -- Proof omitted
  sorry

end max_expr_value_l296_296274


namespace algebraic_expression_zero_iff_x_eq_2_l296_296326

theorem algebraic_expression_zero_iff_x_eq_2 (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (1 / (x - 1) + 3 / (1 - x^2) = 0) ↔ (x = 2) :=
by
  sorry

end algebraic_expression_zero_iff_x_eq_2_l296_296326


namespace find_y_l296_296064

theorem find_y (x y : ℕ) (h1 : x > 0 ∧ y > 0) (h2 : x % y = 9) (h3 : (x:ℝ) / (y:ℝ) = 96.45) : y = 20 :=
by
  sorry

end find_y_l296_296064


namespace area_of_quadrilateral_ABDF_l296_296068

theorem area_of_quadrilateral_ABDF :
  let length := 40
  let width := 30
  let rectangle_area := length * width
  let B := (1/4 : ℝ) * length
  let F := (1/2 : ℝ) * width
  let area_BCD := (1/2 : ℝ) * (3/4 : ℝ) * length * width
  let area_EFD := (1/2 : ℝ) * F * length
  rectangle_area - area_BCD - area_EFD = 450 := sorry

end area_of_quadrilateral_ABDF_l296_296068


namespace new_average_age_l296_296725

theorem new_average_age:
  ∀ (initial_avg_age new_persons_avg_age : ℝ) (initial_count new_persons_count : ℕ),
    initial_avg_age = 16 →
    new_persons_avg_age = 15 →
    initial_count = 20 →
    new_persons_count = 20 →
    (initial_avg_age * initial_count + new_persons_avg_age * new_persons_count) / 
    (initial_count + new_persons_count) = 15.5 :=
by
  intros initial_avg_age new_persons_avg_age initial_count new_persons_count
  intros h1 h2 h3 h4
  
  sorry

end new_average_age_l296_296725


namespace weight_of_b_is_37_l296_296726

variables {a b c : ℝ}

-- Conditions
def average_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def average_bc (b c : ℝ) : Prop := (b + c) / 2 = 46

-- Statement to prove
theorem weight_of_b_is_37 (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 37 :=
by {
  sorry
}

end weight_of_b_is_37_l296_296726


namespace opposite_of_neg_frac_seven_thirds_is_pos_frac_seven_thirds_l296_296038

theorem opposite_of_neg_frac_seven_thirds_is_pos_frac_seven_thirds :
  (- (7 / 3) + (7 / 3) = 0) → (7 / 3) :=
by
    intro h
    exact 7 / 3


end opposite_of_neg_frac_seven_thirds_is_pos_frac_seven_thirds_l296_296038


namespace binomial_prime_divisor_l296_296574

theorem binomial_prime_divisor (p k : ℕ) (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
by
  sorry

end binomial_prime_divisor_l296_296574


namespace line_intersects_circle_chord_min_length_l296_296103

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L based on parameter m
def L (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that for any real number m, line L intersects circle C at two points.
theorem line_intersects_circle (m : ℝ) : 
  ∃ x y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ C x y₁ ∧ C x y₂ ∧ L m x y₁ ∧ L m x y₂ :=
sorry

-- Prove the equation of line L in slope-intercept form when the chord cut by circle C has minimum length.
theorem chord_min_length : ∃ (m : ℝ), ∀ x y : ℝ, 
  L m x y ↔ y = 2 * x - 5 :=
sorry

end line_intersects_circle_chord_min_length_l296_296103


namespace roots_poly_eq_l296_296612

theorem roots_poly_eq (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : d = 0) (root1_eq : 64 * a + 16 * b + 4 * c = 0) (root2_eq : -27 * a + 9 * b - 3 * c = 0) :
  (b + c) / a = -13 :=
by {
  sorry
}

end roots_poly_eq_l296_296612


namespace badges_initial_count_l296_296173

variable {V T : ℕ}

-- conditions
def initial_condition : Prop := V = T + 5
def exchange_condition : Prop := 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1

-- result
theorem badges_initial_count (h1 : initial_condition) (h2 : exchange_condition) : V = 50 ∧ T = 45 := 
  sorry

end badges_initial_count_l296_296173


namespace least_value_difference_l296_296129

noncomputable def least_difference (x : ℝ) : ℝ := 6 - 13/5

theorem least_value_difference (x n m : ℝ) (h1 : 2*x + 5 + 4*x - 3 > x + 15)
                               (h2 : 2*x + 5 + x + 15 > 4*x - 3)
                               (h3 : 4*x - 3 + x + 15 > 2*x + 5)
                               (h4 : x + 15 > 2*x + 5)
                               (h5 : x + 15 > 4*x - 3)
                               (h_m : m = 13/5) (h_n : n = 6)
                               (hx : m < x ∧ x < n) :
  n - m = 17 / 5 :=
  by sorry

end least_value_difference_l296_296129


namespace jacob_age_l296_296832

/- Conditions:
1. Rehana's current age is 25.
2. In five years, Rehana's age is three times Phoebe's age.
3. Jacob's current age is 3/5 of Phoebe's current age.

Prove that Jacob's current age is 3.
-/

theorem jacob_age (R P J : ℕ) (h1 : R = 25) (h2 : R + 5 = 3 * (P + 5)) (h3 : J = 3 / 5 * P) : J = 3 :=
by
  sorry

end jacob_age_l296_296832


namespace Dan_has_five_limes_l296_296508

-- Define the initial condition of limes Dan had
def initial_limes : Nat := 9

-- Define the limes Dan gave to Sara
def limes_given : Nat := 4

-- Define the remaining limes Dan has
def remaining_limes : Nat := initial_limes - limes_given

-- The theorem we need to prove, i.e., the remaining limes Dan has is 5
theorem Dan_has_five_limes : remaining_limes = 5 := by
  sorry

end Dan_has_five_limes_l296_296508


namespace sqrt_of_16_is_4_l296_296910

def arithmetic_square_root (x : ℕ) : ℕ :=
  if x = 0 then 0 else Nat.sqrt x

theorem sqrt_of_16_is_4 : arithmetic_square_root 16 = 4 :=
by
  sorry

end sqrt_of_16_is_4_l296_296910


namespace smallest_x_y_sum_299_l296_296372

theorem smallest_x_y_sum_299 : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x < y ∧ (100 + (x / y : ℚ) = 2 * (100 * x / y : ℚ)) ∧ (x + y = 299) :=
by
  sorry

end smallest_x_y_sum_299_l296_296372


namespace decimal_to_fraction_sum_l296_296898

def recurring_decimal_fraction_sum : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ gcd a b = 1 ∧ (a / b : ℚ) = (0.345345345 : ℚ) ∧ a + b = 226

theorem decimal_to_fraction_sum :
  recurring_decimal_fraction_sum :=
sorry

end decimal_to_fraction_sum_l296_296898


namespace division_of_decimal_l296_296312

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l296_296312


namespace total_boxes_l296_296869

theorem total_boxes (r_cost y_cost : ℝ) (avg_cost : ℝ) (R Y : ℕ) (hc_r : r_cost = 1.30) (hc_y : y_cost = 2.00) 
                    (hc_avg : avg_cost = 1.72) (hc_R : R = 4) (hc_Y : Y = 4) : 
  R + Y = 8 :=
by
  sorry

end total_boxes_l296_296869


namespace line_through_P_with_opposite_sign_intercepts_l296_296729

theorem line_through_P_with_opposite_sign_intercepts 
  (P : ℝ × ℝ) (hP : P = (3, -2)) 
  (h : ∀ (A B : ℝ), A ≠ 0 → B ≠ 0 → A * B < 0) : 
  (∀ (x y : ℝ), (x = 5 ∧ y = -5) → (5 * x - 5 * y - 25 = 0)) ∨ (∀ (x y : ℝ), (3 * y = -2) → (y = - (2 / 3) * x)) :=
sorry

end line_through_P_with_opposite_sign_intercepts_l296_296729


namespace sum_lent_is_correct_l296_296351

variable (P : ℝ) -- Sum lent
variable (R : ℝ) -- Interest rate
variable (T : ℝ) -- Time period
variable (I : ℝ) -- Simple interest

-- Conditions
axiom interest_rate : R = 8
axiom time_period : T = 8
axiom simple_interest_formula : I = (P * R * T) / 100
axiom interest_condition : I = P - 900

-- The proof problem
theorem sum_lent_is_correct : P = 2500 := by
  -- The proof is skipped
  sorry

end sum_lent_is_correct_l296_296351


namespace solution_proof_problem_l296_296143

open Real

noncomputable def proof_problem : Prop :=
  ∀ (a b c : ℝ),
  b + c = 17 →
  c + a = 18 →
  a + b = 19 →
  sqrt (a * b * c * (a + b + c)) = 36 * sqrt 5

theorem solution_proof_problem : proof_problem := 
by sorry

end solution_proof_problem_l296_296143


namespace find_angle_A_l296_296703

noncomputable def exists_angle_A (A B C : ℝ) (a b : ℝ) : Prop :=
  C = (A + B) / 2 ∧ 
  A + B + C = 180 ∧ 
  (a + b) / 2 = Real.sqrt 3 + 1 ∧ 
  C = 2 * Real.sqrt 2

theorem find_angle_A : ∃ A B C a b, 
  exists_angle_A A B C a b ∧ (A = 75 ∨ A = 45) :=
by
  -- This is where the detailed proof would go
  sorry

end find_angle_A_l296_296703


namespace min_sum_of_positive_real_solution_l296_296821

theorem min_sum_of_positive_real_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y = 6 := 
by {
  sorry
}

end min_sum_of_positive_real_solution_l296_296821


namespace Jungkook_has_the_largest_number_l296_296179

theorem Jungkook_has_the_largest_number :
  let Yoongi := 4
  let Yuna := 5
  let Jungkook := 6 + 3
  Jungkook > Yoongi ∧ Jungkook > Yuna := by
    sorry

end Jungkook_has_the_largest_number_l296_296179


namespace sum_of_arithmetic_sequence_l296_296041

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : S n = n * (a 1 + a n) / 2)
  (h3 : a 2 + a 5 + a 11 = 6) :
  S 11 = 22 :=
sorry

end sum_of_arithmetic_sequence_l296_296041


namespace combination_problem_l296_296007

noncomputable def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

theorem combination_problem (x : ℕ) (h : combination 25 (2 * x) = combination 25 (x + 4)) : x = 4 ∨ x = 7 :=
by {
  sorry
}

end combination_problem_l296_296007


namespace mila_hours_to_match_agnes_monthly_earnings_l296_296462

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end mila_hours_to_match_agnes_monthly_earnings_l296_296462


namespace gwendolyn_read_time_l296_296406

theorem gwendolyn_read_time :
  let rate := 200 -- sentences per hour
  let paragraphs_per_page := 30
  let sentences_per_paragraph := 15
  let pages := 100
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  let total_time := total_sentences / rate
  total_time = 225 :=
by
  sorry

end gwendolyn_read_time_l296_296406


namespace cage_chicken_problem_l296_296416

theorem cage_chicken_problem :
  (∃ x : ℕ, 6 ≤ x ∧ x ≤ 10 ∧ (4 * x + 1 = 5 * (x - 1))) ∧
  (∀ x : ℕ, 6 ≤ x ∧ x ≤ 10 → (4 * x + 1 ≥ 25 ∧ 4 * x + 1 ≤ 41)) :=
by
  sorry

end cage_chicken_problem_l296_296416


namespace C_share_l296_296340

theorem C_share (a b c : ℕ) (h1 : a + b + c = 1010)
                (h2 : ∃ k : ℕ, a = 3 * k + 25 ∧ b = 2 * k + 10 ∧ c = 5 * k + 15) : c = 495 :=
by
  -- Sorry is used to skip the proof
  sorry

end C_share_l296_296340


namespace perfect_squares_perfect_square_plus_one_l296_296792

theorem perfect_squares : (∃ n : ℕ, 2^n + 3 = (x : ℕ)^2) ↔ n = 0 ∨ n = 3 :=
by
  sorry

theorem perfect_square_plus_one : (∃ n : ℕ, 2^n + 1 = (x : ℕ)^2) ↔ n = 3 :=
by
  sorry

end perfect_squares_perfect_square_plus_one_l296_296792


namespace total_rattlesnakes_l296_296280

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end total_rattlesnakes_l296_296280


namespace smallest_number_l296_296909

theorem smallest_number (n : ℕ) : 
  (∀ k ∈ [12, 16, 18, 21, 28, 35, 39], ∃ m : ℕ, (n - 3) = k * m) → 
  n = 65517 := by
  sorry

end smallest_number_l296_296909


namespace tan_5pi_over_4_l296_296662

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l296_296662


namespace find_average_speed_l296_296186

theorem find_average_speed :
  ∃ v : ℝ, (880 / v) - (880 / (v + 10)) = 2 ∧ v = 61.5 :=
by
  sorry

end find_average_speed_l296_296186


namespace mark_initial_fries_l296_296264

variable (Sally_fries_before : ℕ)
variable (Sally_fries_after : ℕ)
variable (Mark_fries_given : ℕ)
variable (Mark_fries_initial : ℕ)

theorem mark_initial_fries (h1 : Sally_fries_before = 14) (h2 : Sally_fries_after = 26) (h3 : Mark_fries_given = Sally_fries_after - Sally_fries_before) (h4 : Mark_fries_given = 1/3 * Mark_fries_initial) : Mark_fries_initial = 36 :=
by sorry

end mark_initial_fries_l296_296264


namespace simplify_polynomials_l296_296029

-- Define the polynomials
def poly1 (q : ℝ) : ℝ := 5 * q^4 + 3 * q^3 - 7 * q + 8
def poly2 (q : ℝ) : ℝ := 6 - 9 * q^3 + 4 * q - 3 * q^4

-- The goal is to prove that the sum of poly1 and poly2 simplifies correctly
theorem simplify_polynomials (q : ℝ) : 
  poly1 q + poly2 q = 2 * q^4 - 6 * q^3 - 3 * q + 14 := 
by 
  sorry

end simplify_polynomials_l296_296029


namespace minimum_value_of_expression_l296_296212

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    ∃ (c : ℝ), (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x^3 + y^3 - 5 * x * y ≥ c) ∧ c = -125 / 27 :=
by
  sorry

end minimum_value_of_expression_l296_296212


namespace distance_between_towns_in_kilometers_l296_296882

theorem distance_between_towns_in_kilometers :
  (20 * 5) * 1.60934 = 160.934 :=
by
  sorry

end distance_between_towns_in_kilometers_l296_296882


namespace sam_average_speed_l296_296871

theorem sam_average_speed :
  let total_time := 7 -- total time from 7 a.m. to 2 p.m.
  let rest_time := 1 -- rest period from 9 a.m. to 10 a.m.
  let effective_time := total_time - rest_time
  let total_distance := 200 -- total miles covered
  let avg_speed := total_distance / effective_time
  avg_speed = 33.3 :=
sorry

end sam_average_speed_l296_296871


namespace probability_no_adjacent_same_rolls_l296_296948

theorem probability_no_adjacent_same_rolls :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6)
  let no_adjacent_same := outcomes.filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ a)
  (no_adjacent_same.card : ℚ) / outcomes.card = 25 / 108 :=
by
  sorry

end probability_no_adjacent_same_rolls_l296_296948


namespace range_of_k_l296_296975

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x^2 + k * y^2 = 2) ∧ (∀ x y : ℝ, y ≠ 0 → x^2 + k * y^2 = 2 → (x = 0 ∧ (∃ a : ℝ, a > 1 ∧ y = a))) → 0 < k ∧ k < 1 :=
sorry

end range_of_k_l296_296975


namespace Jillian_largest_apartment_size_l296_296495

noncomputable def largest_apartment_size (budget rent_per_sqft: ℝ) : ℝ :=
  budget / rent_per_sqft

theorem Jillian_largest_apartment_size :
  largest_apartment_size 720 1.20 = 600 := 
by
  sorry

end Jillian_largest_apartment_size_l296_296495


namespace base4_more_digits_than_base9_l296_296970

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l296_296970


namespace record_jump_l296_296018

theorem record_jump (standard_jump jump : Float) (h_standard : standard_jump = 4.00) (h_jump : jump = 3.85) : (jump - standard_jump : Float) = -0.15 := 
by
  rw [h_standard, h_jump]
  simp
  sorry

end record_jump_l296_296018


namespace triangle_area_l296_296741

theorem triangle_area (base height : ℕ) (h_base : base = 35) (h_height : height = 12) :
  (1 / 2 : ℚ) * base * height = 210 := by
  sorry

end triangle_area_l296_296741


namespace robin_initial_gum_l296_296587

theorem robin_initial_gum (x : ℕ) (h1 : x + 26 = 44) : x = 18 := 
by 
  sorry

end robin_initial_gum_l296_296587


namespace find_y_z_l296_296170

def abs_diff (x y : ℝ) := abs (x - y)

noncomputable def seq_stabilize (x y z : ℝ) (n : ℕ) : Prop :=
  let x1 := abs_diff x y 
  let y1 := abs_diff y z 
  let z1 := abs_diff z x
  ∃ k : ℕ, k ≥ n ∧ abs_diff x1 y1 = x ∧ abs_diff y1 z1 = y ∧ abs_diff z1 x1 = z

theorem find_y_z (x y z : ℝ) (hx : x = 1) (hstab : ∃ n : ℕ, seq_stabilize x y z n) : y = 0 ∧ z = 0 :=
sorry

end find_y_z_l296_296170


namespace distinct_nonzero_reals_xy_six_l296_296995

theorem distinct_nonzero_reals_xy_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 6/x = y + 6/y) (h_distinct : x ≠ y) : x * y = 6 := 
sorry

end distinct_nonzero_reals_xy_six_l296_296995


namespace number_of_pears_in_fruit_gift_set_l296_296457

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end number_of_pears_in_fruit_gift_set_l296_296457


namespace find_a_if_lines_perpendicular_l296_296957

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end find_a_if_lines_perpendicular_l296_296957


namespace arcsin_neg_sqrt_two_over_two_l296_296503

theorem arcsin_neg_sqrt_two_over_two : Real.arcsin (-Real.sqrt 2 / 2) = -Real.pi / 4 :=
  sorry

end arcsin_neg_sqrt_two_over_two_l296_296503


namespace cole_round_trip_time_l296_296785

/-- Prove that the total round trip time is 2 hours given the conditions -/
theorem cole_round_trip_time :
  ∀ (speed_to_work : ℝ) (speed_back_home : ℝ) (time_to_work_min : ℝ),
  speed_to_work = 50 → speed_back_home = 110 → time_to_work_min = 82.5 →
  ((time_to_work_min / 60) * speed_to_work + (time_to_work_min * speed_to_work / speed_back_home) / 60) = 2 :=
by
  intros
  sorry

end cole_round_trip_time_l296_296785


namespace time_saved_is_35_minutes_l296_296877

-- Define the speed and distances for each day
def monday_distance := 3
def wednesday_distance := 3
def friday_distance := 3
def sunday_distance := 4
def speed_monday := 6
def speed_wednesday := 4
def speed_friday := 5
def speed_sunday := 3
def speed_uniform := 5

-- Calculate the total time spent on the treadmill originally
def time_monday := monday_distance / speed_monday
def time_wednesday := wednesday_distance / speed_wednesday
def time_friday := friday_distance / speed_friday
def time_sunday := sunday_distance / speed_sunday
def total_time := time_monday + time_wednesday + time_friday + time_sunday

-- Calculate the total time if speed was uniformly 5 mph 
def total_distance := monday_distance + wednesday_distance + friday_distance + sunday_distance
def total_time_uniform := total_distance / speed_uniform

-- Time saved if walking at 5 mph every day
def time_saved := total_time - total_time_uniform

-- Convert time saved to minutes
def minutes_saved := time_saved * 60

theorem time_saved_is_35_minutes : minutes_saved = 35 := by
  sorry

end time_saved_is_35_minutes_l296_296877


namespace range_subset_pos_iff_l296_296163

theorem range_subset_pos_iff (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_subset_pos_iff_l296_296163


namespace cubes_difference_l296_296111

-- Given conditions
variables (a b : ℝ)
hypothesis h1 : a - b = 7
hypothesis h2 : a^2 + b^2 = 50

-- The theorem statement
theorem cubes_difference : a^3 - b^3 = 353.5 :=
by
  sorry

end cubes_difference_l296_296111


namespace base4_more_digits_than_base9_l296_296971

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l296_296971


namespace boys_on_playground_l296_296168

theorem boys_on_playground (total_children girls : ℕ) (h1 : total_children = 117) (h2 : girls = 77) :
  ∃ boys : ℕ, boys = total_children - girls ∧ boys = 40 :=
by
  have boys := total_children - girls
  use boys
  split
  . exact rfl
  . rw [h1, h2]

-- Proof skipped (Lean 4 statement only)

end boys_on_playground_l296_296168


namespace correct_quadratic_equation_l296_296057

def is_quadratic_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + 1 = 0"

theorem correct_quadratic_equation :
  is_quadratic_with_one_variable "x^2 + 1 = 0" :=
by {
  sorry
}

end correct_quadratic_equation_l296_296057


namespace line_intersects_axes_l296_296916

theorem line_intersects_axes (a b : ℝ) (x1 y1 x2 y2 : ℝ) (h_points : (x1, y1) = (8, 2) ∧ (x2, y2) = (4, 6)) :
  (∃ x_intercept : ℝ, (x_intercept, 0) = (10, 0)) ∧ (∃ y_intercept : ℝ, (0, y_intercept) = (0, 10)) :=
by
  sorry

end line_intersects_axes_l296_296916


namespace mike_sold_song_book_for_correct_amount_l296_296438

-- Define the constants for the cost of the trumpet and the net amount spent
def cost_of_trumpet : ℝ := 145.16
def net_amount_spent : ℝ := 139.32

-- Define the amount received from selling the song book
def amount_received_from_selling_song_book : ℝ :=
  cost_of_trumpet - net_amount_spent

-- The theorem stating the amount Mike sold the song book for
theorem mike_sold_song_book_for_correct_amount :
  amount_received_from_selling_song_book = 5.84 :=
sorry

end mike_sold_song_book_for_correct_amount_l296_296438


namespace eliot_account_balance_l296_296870

variable (A E : ℝ)

-- Condition 1: Al has more money than Eliot.
axiom h1 : A > E

-- Condition 2: The difference between their accounts is 1/12 of the sum of their accounts.
axiom h2 : A - E = (1 / 12) * (A + E)

-- Condition 3: If Al's account were increased by 10% and Eliot's by 20%, Al would have exactly $21 more than Eliot.
axiom h3 : 1.1 * A = 1.2 * E + 21

-- Conjecture: Eliot has $210 in his account.
theorem eliot_account_balance : E = 210 :=
by
  sorry

end eliot_account_balance_l296_296870


namespace qiuqiu_servings_l296_296790

-- Define the volume metrics
def bottles : ℕ := 1
def cups_per_bottle_kangkang : ℕ := 4
def foam_expansion : ℕ := 3
def foam_fraction : ℚ := 1 / 2

-- Calculate the effective cup volume under Qiuqiu's serving method
def beer_fraction_per_cup_qiuqiu : ℚ := 1 / 2 + (1 / foam_expansion) * foam_fraction

-- Calculate the number of cups Qiuqiu can serve from one bottle
def qiuqiu_cups_from_bottle : ℚ := cups_per_bottle_kangkang / beer_fraction_per_cup_qiuqiu

-- The theorem statement
theorem qiuqiu_servings :
  qiuqiu_cups_from_bottle = 6 := by
  sorry

end qiuqiu_servings_l296_296790


namespace complement_intersection_l296_296003

open Set

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := {x | x + 1 < 0}
noncomputable def B : Set ℝ := {x | x - 3 < 0}
noncomputable def C_UA : Set ℝ := {x | x >= -1}

theorem complement_intersection (U A B C_UA) :
  (C_UA ∩ B) = {x | -1 ≤ x ∧ x < 3} :=
by 
  sorry

end complement_intersection_l296_296003


namespace bolton_class_students_l296_296981

theorem bolton_class_students 
  (S : ℕ) 
  (H1 : 2/5 < 1)
  (H2 : 1/3 < 1)
  (C1 : (2 / 5) * (S:ℝ) + (2 / 5) * (S:ℝ) = 20) : 
  S = 25 := 
by
  sorry

end bolton_class_students_l296_296981


namespace sqrt_400_div_2_l296_296324

theorem sqrt_400_div_2 : (Nat.sqrt 400) / 2 = 10 := by
  sorry

end sqrt_400_div_2_l296_296324


namespace number_of_students_suggested_mashed_potatoes_l296_296592

theorem number_of_students_suggested_mashed_potatoes 
    (students_suggested_bacon : ℕ := 374) 
    (students_suggested_tomatoes : ℕ := 128) 
    (total_students_participated : ℕ := 826) : 
    (total_students_participated - (students_suggested_bacon + students_suggested_tomatoes)) = 324 :=
by sorry

end number_of_students_suggested_mashed_potatoes_l296_296592


namespace remainder_div_l296_296934

theorem remainder_div (n : ℕ) : (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 + 
  90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 - 90^7 * Nat.choose 10 7 + 
  90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 + 90^10 * Nat.choose 10 10) % 88 = 1 := by
  sorry

end remainder_div_l296_296934


namespace compute_value_of_expression_l296_296853

theorem compute_value_of_expression :
  ∃ p q : ℝ, (3 * p^2 - 3 * q^2) / (p - q) = 5 ∧ 3 * p^2 - 5 * p - 14 = 0 ∧ 3 * q^2 - 5 * q - 14 = 0 :=
sorry

end compute_value_of_expression_l296_296853


namespace linear_in_one_variable_linear_in_two_variables_l296_296221

namespace MathProof

-- Definition of the equation
def equation (k x y : ℝ) : ℝ := (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y - (k + 2)

-- Theorem for linear equation in one variable
theorem linear_in_one_variable (k : ℝ) (x y : ℝ) :
  k = -1 → equation k x y = 0 → ∃ y' : ℝ, equation k 0 y' = 0 :=
by
  sorry

-- Theorem for linear equation in two variables
theorem linear_in_two_variables (k : ℝ) (x y : ℝ) :
  k = 1 → equation k x y = 0 → ∃ x' y' : ℝ, equation k x' y' = 0 :=
by
  sorry

end MathProof

end linear_in_one_variable_linear_in_two_variables_l296_296221


namespace diana_owes_amount_l296_296375

def principal : ℝ := 60
def rate : ℝ := 0.06
def time : ℝ := 1
def interest := principal * rate * time
def original_amount := principal
def total_amount := original_amount + interest

theorem diana_owes_amount :
  total_amount = 63.60 :=
by
  -- Placeholder for actual proof
  sorry

end diana_owes_amount_l296_296375


namespace solution_set_equivalence_l296_296200

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative : ∀ x : ℝ, deriv f x > 1 - f x
axiom f_at_0 : f 0 = 3

theorem solution_set_equivalence :
  {x : ℝ | (Real.exp x) * f x > (Real.exp x) + 2} = {x : ℝ | x > 0} :=
by sorry

end solution_set_equivalence_l296_296200


namespace salary_in_April_after_changes_l296_296501

def salary_in_January : ℝ := 3000
def raise_percentage : ℝ := 0.10
def pay_cut_percentage : ℝ := 0.15
def bonus : ℝ := 200

theorem salary_in_April_after_changes :
  let s_Feb := salary_in_January * (1 + raise_percentage)
  let s_Mar := s_Feb * (1 - pay_cut_percentage)
  let s_Apr := s_Mar + bonus
  s_Apr = 3005 :=
by
  sorry

end salary_in_April_after_changes_l296_296501


namespace total_distance_l296_296284

noncomputable def total_distance_covered 
  (radius1 radius2 radius3 : ℝ) 
  (rev1 rev2 rev3 : ℕ) : ℝ :=
  let π := Real.pi
  let circumference r := 2 * π * r
  let distance r rev := circumference r * rev
  distance radius1 rev1 + distance radius2 rev2 + distance radius3 rev3

theorem total_distance
  (h1 : radius1 = 20.4) 
  (h2 : radius2 = 15.3) 
  (h3 : radius3 = 25.6) 
  (h4 : rev1 = 400) 
  (h5 : rev2 = 320) 
  (h6 : rev3 = 500) :
  total_distance_covered 20.4 15.3 25.6 400 320 500 = 162436.6848 := 
sorry

end total_distance_l296_296284


namespace union_of_A_and_B_l296_296400

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B : Set ℝ := {x | x < 3}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 3} := by
  sorry

end union_of_A_and_B_l296_296400


namespace white_area_correct_l296_296045

/-- The dimensions of the sign and the letter components -/
def sign_width : ℕ := 18
def sign_height : ℕ := 6
def vertical_bar_height : ℕ := 6
def vertical_bar_width : ℕ := 1
def horizontal_bar_length : ℕ := 4
def horizontal_bar_width : ℕ := 1

/-- The areas of the components of each letter -/
def area_C : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)
def area_O : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + 2 * (horizontal_bar_length * horizontal_bar_width)
def area_L : ℕ := (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)

/-- The total area of the sign -/
def total_sign_area : ℕ := sign_height * sign_width

/-- The total black area covered by the letters "COOL" -/
def total_black_area : ℕ := area_C + 2 * area_O + area_L

/-- The area of the white portion of the sign -/
def white_area : ℕ := total_sign_area - total_black_area

/-- Proof that the area of the white portion of the sign is 42 square units -/
theorem white_area_correct : white_area = 42 := by
  -- Calculation steps (skipped, though the result is expected to be 42)
  sorry

end white_area_correct_l296_296045


namespace cube_difference_div_l296_296620

theorem cube_difference_div (a b : ℕ) (h_a : a = 64) (h_b : b = 27) : 
  (a^3 - b^3) / (a - b) = 6553 := by
  sorry

end cube_difference_div_l296_296620


namespace sally_garden_area_l296_296263

theorem sally_garden_area :
  ∃ (a b : ℕ), 2 * (a + b) = 24 ∧ b + 1 = 3 * (a + 1) ∧ 
     (3 * (a - 1) * 3 * (b - 1) = 297) :=
by {
  sorry
}

end sally_garden_area_l296_296263


namespace smallest_lcm_of_4_digit_integers_with_gcd_5_l296_296122

-- Definition of the given integers k and l
def positive_4_digit_integers (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- The main theorem we want to prove
theorem smallest_lcm_of_4_digit_integers_with_gcd_5 :
  ∃ (k l : ℕ), positive_4_digit_integers k ∧ positive_4_digit_integers l ∧ gcd k l = 5 ∧ lcm k l = 201000 :=
by {
  sorry
}

end smallest_lcm_of_4_digit_integers_with_gcd_5_l296_296122


namespace find_monthly_fee_l296_296518

-- Define the given conditions
def monthly_fee (fee_per_minute : ℝ) (total_bill : ℝ) (minutes_used : ℕ) : ℝ :=
  total_bill - (fee_per_minute * minutes_used)

-- Define the values from the condition
def fee_per_minute := 0.12 -- 12 cents in dollars
def total_bill := 23.36 -- total bill in dollars
def minutes_used := 178 -- total minutes billed

-- Define the expected monthly fee
def expected_monthly_fee := 2.0 -- expected monthly fee in dollars

-- Problem statement: Prove that the monthly fee is equal to the expected monthly fee
theorem find_monthly_fee : 
  monthly_fee fee_per_minute total_bill minutes_used = expected_monthly_fee := by
  sorry

end find_monthly_fee_l296_296518


namespace find_m_l296_296965

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)
noncomputable def dot_prod (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) (h : dot_prod a (b m) / magnitude a = 3) : m = sqrt 3 :=
by
  sorry

end find_m_l296_296965


namespace Eight_div_by_repeating_decimal_0_3_l296_296315

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l296_296315


namespace flower_cost_l296_296236

-- Given conditions
variables {x y : ℕ} -- costs of type A and type B flowers respectively

-- Costs equations
def cost_equation_1 : Prop := 3 * x + 4 * y = 360
def cost_equation_2 : Prop := 4 * x + 3 * y = 340

-- Given the necessary planted pots and rates
variables {m n : ℕ} (Hmn : m + n = 600) 
-- Percentage survivals
def survival_rate_A : ℚ := 0.70
def survival_rate_B : ℚ := 0.90

-- Replacement condition
def replacement_cond : Prop := (1 - survival_rate_A) * m + (1 - survival_rate_B) * n ≤ 100

-- Minimum cost condition
def min_cost (m_plant : ℕ) (n_plant : ℕ) : ℕ := 40 * m_plant + 60 * n_plant

theorem flower_cost 
  (H1 : cost_equation_1)
  (H2 : cost_equation_2)
  (H3 : x = 40)
  (H4 : y = 60) 
  (Hmn : m + n = 600)
  (Hsurv : replacement_cond) : 
  (m = 200 ∧ n = 400) ∧ 
  (min_cost 200 400 = 32000) := 
sorry

end flower_cost_l296_296236


namespace problem_1_problem_2_l296_296544

open Set -- to work with sets conveniently

noncomputable section -- to allow the use of real numbers and other non-constructive elements

-- Define U as the set of all real numbers
def U : Set ℝ := univ

-- Define M as the set of all x such that y = sqrt(x - 2)
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 2) }

-- Define N as the set of all x such that x < 1 or x > 3
def N : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Statement to prove (1)
theorem problem_1 : M ∪ N = {x : ℝ | x < 1 ∨ x ≥ 2} := sorry

-- Statement to prove (2)
theorem problem_2 : M ∩ (compl N) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end problem_1_problem_2_l296_296544


namespace specific_value_of_n_l296_296674

theorem specific_value_of_n (n : ℕ) 
  (A_n : ℕ → ℕ)
  (C_n : ℕ → ℕ → ℕ)
  (h1 : A_n n ^ 2 = C_n n (n-3)) :
  n = 8 :=
sorry

end specific_value_of_n_l296_296674


namespace eight_div_repeating_three_l296_296297

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l296_296297


namespace tan_sub_sin_eq_sq3_div2_l296_296775

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end tan_sub_sin_eq_sq3_div2_l296_296775


namespace derivative_at_two_l296_296715

theorem derivative_at_two {f : ℝ → ℝ} (f_deriv : ∀x, deriv f x = 2 * x - 4) : deriv f 2 = 0 := 
by sorry

end derivative_at_two_l296_296715


namespace max_gcd_13n_plus_4_8n_plus_3_l296_296928

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 9 ∧ gcd (13 * n + 4) (8 * n + 3) = k := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l296_296928


namespace calculate_fraction_l296_296363

theorem calculate_fraction (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) : 
  (1 / (x - 1)) - (2 / (x^2 - 1)) = 1 / (x + 1) :=
by
  sorry

end calculate_fraction_l296_296363


namespace number_of_people_eating_both_l296_296016

variable (A B C : Nat)

theorem number_of_people_eating_both (hA : A = 13) (hB : B = 19) (hC : C = B - A) : C = 6 :=
by 
  sorry

end number_of_people_eating_both_l296_296016


namespace distinct_real_roots_imply_sum_greater_than_two_l296_296540

noncomputable def function_f (x: ℝ) : ℝ := abs (Real.log x)

theorem distinct_real_roots_imply_sum_greater_than_two {k α β : ℝ} 
  (h₁ : function_f α = k) 
  (h₂ : function_f β = k) 
  (h₃ : α ≠ β) 
  (h4 : 0 < α ∧ α < 1)
  (h5 : 1 < β) :
  (1 / α) + (1 / β) > 2 :=
sorry

end distinct_real_roots_imply_sum_greater_than_two_l296_296540


namespace find_sqrt_abc_sum_l296_296142

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end find_sqrt_abc_sum_l296_296142


namespace simplify_expression_inequality_solution_l296_296184

-- Simplification part
theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2):
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10 * x + 25) / (x^2 - 4)) = 
  (x - 2) / (x + 5) :=
sorry

-- Inequality system part
theorem inequality_solution (x : ℝ):
  (2 * x + 7 > 3) ∧ ((x + 1) / 3 > (x - 1) / 2) → -2 < x ∧ x < 5 :=
sorry

end simplify_expression_inequality_solution_l296_296184


namespace find_y_given_x_zero_l296_296032

theorem find_y_given_x_zero (t : ℝ) (y : ℝ) : 
  (3 - 2 * t = 0) → (y = 3 * t + 6) → y = 21 / 2 := 
by 
  sorry

end find_y_given_x_zero_l296_296032


namespace sponsorship_prob_zero_sponsorship_prob_gt_150k_l296_296188

noncomputable def prob_sponsorship_amount_zero
  (supports : Fin 3 → ℕ → ℕ → Prop)
  (prob : supports.has_probability (supports _ _ (1/2))) : ℙ :=
begin
  let empty_support := λ (student_supports : Fin 3 → ℕ), ∑ i, student_supports i = 0,
  exact
    @independent_product_prob _ _ _ supports (by_probability) (empty_support) sorry
end

noncomputable def prob_sponsorship_amount_gt_150k
  (supports : Fin 3 → ℕ → ℕ → Prop)
  (prob : supports.has_probability (supports _ _ (1/2))) : ℙ :=
begin
  let excess_support := λ (student_supports : Fin 3 → ℕ), ∑ i, student_supports i > 150000,
  exact
    @independent_product_prob _ _ _ supports (by_probability) (excess_support) sorry
end

theorem sponsorship_prob_zero :
  prob_sponsorship_amount_zero = 1 / 64 :=
sorry

theorem sponsorship_prob_gt_150k :
  prob_sponsorship_amount_gt_150k = 11 / 32 :=
sorry

end sponsorship_prob_zero_sponsorship_prob_gt_150k_l296_296188


namespace distinct_connected_stamps_l296_296796

theorem distinct_connected_stamps (n : ℕ) : 
  ∃ d : ℕ → ℝ, 
    d (n+1) = 1 / 4 * (1 + Real.sqrt 2)^(n + 3) + 1 / 4 * (1 - Real.sqrt 2)^(n + 3) - 2 * n - 7 / 2 :=
sorry

end distinct_connected_stamps_l296_296796


namespace proof_problem_l296_296370

def diamond (a b : ℚ) := a - (1 / b)

theorem proof_problem :
  ((diamond (diamond 2 4) 5) - (diamond 2 (diamond 4 5))) = (-71 / 380) := by
  sorry

end proof_problem_l296_296370


namespace area_of_original_rectangle_l296_296908

theorem area_of_original_rectangle 
  (L W : ℝ)
  (h1 : 2 * L * (3 * W) = 1800) :
  L * W = 300 :=
by
  sorry

end area_of_original_rectangle_l296_296908


namespace focus_of_parabola_l296_296728

theorem focus_of_parabola (x y : ℝ) (h : x^2 = 16 * y) : (0, 4) = (0, 4) :=
by {
  sorry
}

end focus_of_parabola_l296_296728


namespace ratio_of_shirt_to_pants_l296_296579

theorem ratio_of_shirt_to_pants
    (total_cost : ℕ)
    (price_pants : ℕ)
    (price_shoes : ℕ)
    (price_shirt : ℕ)
    (h1 : total_cost = 340)
    (h2 : price_pants = 120)
    (h3 : price_shoes = price_pants + 10)
    (h4 : price_shirt = total_cost - (price_pants + price_shoes)) :
    price_shirt * 4 = price_pants * 3 := sorry

end ratio_of_shirt_to_pants_l296_296579


namespace total_flower_petals_l296_296939

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end total_flower_petals_l296_296939


namespace stickers_given_l296_296623

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end stickers_given_l296_296623


namespace sum_of_solutions_eqn_l296_296384

theorem sum_of_solutions_eqn : 
  (∀ x : ℝ, -48 * x^2 + 100 * x + 200 = 0 → False) → 
  (-100 / -48) = (25 / 12) :=
by
  intros
  sorry

end sum_of_solutions_eqn_l296_296384


namespace equidistant_P_AP_BP_CP_DP_l296_296712

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2

def A : Point := ⟨10, 0, 0⟩
def B : Point := ⟨0, -6, 0⟩
def C : Point := ⟨0, 0, 8⟩
def D : Point := ⟨0, 0, 0⟩
def P : Point := ⟨5, -3, 4⟩

theorem equidistant_P_AP_BP_CP_DP :
  distance P A = distance P B ∧ distance P B = distance P C ∧ distance P C = distance P D := 
sorry

end equidistant_P_AP_BP_CP_DP_l296_296712


namespace alpha_numeric_puzzle_l296_296700

theorem alpha_numeric_puzzle : 
  ∀ (a b c d e f g h i : ℕ),
  (∀ x y : ℕ, x ≠ 0 → y ≠ 0 → x ≠ y) →
  100 * a + 10 * b + c + 100 * d + 10 * e + f + 100 * g + 10 * h + i = 1665 → 
  c + f + i = 15 →
  b + e + h = 15 :=
by
  intros a b c d e f g h i distinct nonzero_sum unit_digits_sum
  sorry

end alpha_numeric_puzzle_l296_296700


namespace eight_div_repeating_three_l296_296298

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l296_296298


namespace find_min_value_l296_296430

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1)

theorem find_min_value :
  (1 / (2 * a + 3 * b) + 1 / (2 * b + 3 * c) + 1 / (2 * c + 3 * a)) ≥ (9 / 5) :=
sorry

end find_min_value_l296_296430


namespace vertical_angles_are_congruent_l296_296471

def supplementary_angles (a b : ℝ) : Prop := a + b = 180
def corresponding_angles (l1 l2 t : ℝ) : Prop := l1 = l2
def exterior_angle_greater (ext int1 int2 : ℝ) : Prop := ext = int1 + int2
def vertical_angles_congruent (a b : ℝ) : Prop := a = b

theorem vertical_angles_are_congruent (a b : ℝ) (h : vertical_angles_congruent a b) : a = b := by
  sorry

end vertical_angles_are_congruent_l296_296471


namespace area_of_new_geometric_figure_correct_l296_296528

noncomputable def area_of_new_geometric_figure (a b : ℝ) : ℝ := 
  let d := Real.sqrt (a^2 + b^2)
  a * b + (b * d) / 4

theorem area_of_new_geometric_figure_correct (a b : ℝ) :
  area_of_new_geometric_figure a b = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 :=
by 
  sorry

end area_of_new_geometric_figure_correct_l296_296528


namespace entrance_exit_ways_equal_49_l296_296740

-- Define the number of gates on each side
def south_gates : ℕ := 4
def north_gates : ℕ := 3

-- Define the total number of gates
def total_gates : ℕ := south_gates + north_gates

-- State the theorem and provide the expected proof structure
theorem entrance_exit_ways_equal_49 : (total_gates * total_gates) = 49 := 
by {
  sorry
}

end entrance_exit_ways_equal_49_l296_296740


namespace find_triangle_value_l296_296215

variables (triangle q r : ℝ)
variables (h1 : triangle + q = 75) (h2 : triangle + q + r = 138) (h3 : r = q / 3)

theorem find_triangle_value : triangle = -114 :=
by
  sorry

end find_triangle_value_l296_296215


namespace additional_birds_flew_up_l296_296635

-- Defining the conditions from the problem
def original_birds : ℕ := 179
def total_birds : ℕ := 217

-- Defining the question to be proved as a theorem
theorem additional_birds_flew_up : 
  total_birds - original_birds = 38 :=
by
  sorry

end additional_birds_flew_up_l296_296635


namespace num_of_elements_l296_296599

-- Lean statement to define and prove the problem condition
theorem num_of_elements (n S : ℕ) (h1 : (S + 26) / n = 5) (h2 : (S + 36) / n = 6) : n = 10 := by
  sorry

end num_of_elements_l296_296599


namespace total_coins_Zain_l296_296338

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l296_296338


namespace find_number_l296_296121

theorem find_number
  (x : ℝ)
  (h : 0.90 * x = 0.50 * 1080) :
  x = 600 :=
by
  sorry

end find_number_l296_296121


namespace exam_standard_deviation_l296_296951

-- Define the mean score
def mean_score : ℝ := 74

-- Define the standard deviation and conditions
def standard_deviation (σ : ℝ) : Prop :=
  mean_score - 2 * σ = 58

-- Define the condition to prove
def standard_deviation_above_mean (σ : ℝ) : Prop :=
  (98 - mean_score) / σ = 3

theorem exam_standard_deviation {σ : ℝ} (h1 : standard_deviation σ) : standard_deviation_above_mean σ :=
by
  -- proof is omitted
  sorry

end exam_standard_deviation_l296_296951


namespace correct_values_correct_result_l296_296053

theorem correct_values (a b : ℝ) :
  ((2 * x - a) * (3 * x + b) = 6 * x^2 + 11 * x - 10) ∧
  ((2 * x + a) * (x + b) = 2 * x^2 - 9 * x + 10) →
  (a = -5) ∧ (b = -2) :=
sorry

theorem correct_result :
  (2 * x - 5) * (3 * x - 2) = 6 * x^2 - 19 * x + 10 :=
sorry

end correct_values_correct_result_l296_296053


namespace greatest_number_of_balloons_l296_296260

-- Let p be the regular price of one balloon, and M be the total amount of money Orvin has
variable (p M : ℝ)

-- Initial condition: Orvin can buy 45 balloons at the regular price.
-- Thus, he has money M = 45 * p
def orvin_has_enough_money : Prop :=
  M = 45 * p

-- Special Sale condition: The first balloon costs p and the second balloon costs p/2,
-- so total cost for 2 balloons = 1.5 * p
def special_sale_condition : Prop :=
  ∀ pairs : ℝ, M / (1.5 * p) = pairs ∧ pairs * 2 = 60

-- Given the initial condition and the special sale condition, prove the greatest 
-- number of balloons Orvin could purchase is 60
theorem greatest_number_of_balloons (p : ℝ) (M : ℝ) (h1 : orvin_has_enough_money p M) (h2 : special_sale_condition p M) : 
∀ N : ℝ, N = 60 :=
sorry

end greatest_number_of_balloons_l296_296260


namespace division_by_repeating_decimal_l296_296291

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l296_296291


namespace gold_problem_proof_l296_296240

noncomputable def solve_gold_problem : Prop :=
  ∃ (a : ℕ → ℝ), 
  (a 1) + (a 2) + (a 3) = 4 ∧ 
  (a 8) + (a 9) + (a 10) = 3 ∧
  (a 5) + (a 6) = 7 / 3

theorem gold_problem_proof : solve_gold_problem := 
  sorry

end gold_problem_proof_l296_296240


namespace inheritance_amount_l296_296135

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end inheritance_amount_l296_296135


namespace tan_five_pi_over_four_l296_296660

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l296_296660


namespace new_price_of_sugar_l296_296275

theorem new_price_of_sugar (C : ℝ) (H : 10 * C = P * (0.7692307692307693 * C)) : P = 13 := by
  sorry

end new_price_of_sugar_l296_296275


namespace solve_3x_plus_5_squared_l296_296692

theorem solve_3x_plus_5_squared (x : ℝ) (h : 5 * x - 6 = 15 * x + 21) : 
  3 * (x + 5) ^ 2 = 2523 / 100 :=
by
  sorry

end solve_3x_plus_5_squared_l296_296692


namespace problem_l296_296956

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) (h : ∀ x : ℝ, f (4 * x) = 4) : f (2 * x) = 4 :=
by
  sorry

end problem_l296_296956


namespace largest_consecutive_sum_is_nine_l296_296616

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end largest_consecutive_sum_is_nine_l296_296616


namespace find_g5_l296_296730

variable (g : ℝ → ℝ)

-- Formal definition of the condition for the function g in the problem statement.
def functional_eq_condition :=
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

-- The main statement to prove g(5) = 1 under the given condition.
theorem find_g5 (h : functional_eq_condition g) :
  g 5 = 1 :=
sorry

end find_g5_l296_296730


namespace cube_volume_increase_l296_296695

theorem cube_volume_increase (s : ℝ) (h : s > 0) :
  let new_volume := (1.4 * s) ^ 3
  let original_volume := s ^ 3
  let increase_percentage := ((new_volume - original_volume) / original_volume) * 100
  increase_percentage = 174.4 := by
  sorry

end cube_volume_increase_l296_296695


namespace domain_of_function_correct_l296_296650

noncomputable def domain_of_function (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (2 - x > 0) ∧ (Real.logb 10 (2 - x) ≠ 0)

theorem domain_of_function_correct :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ∈ Set.Icc (-1 : ℝ) 1 \ {1}} ∪ {x : ℝ | x ∈ Set.Ioc 1 2} :=
by
  sorry

end domain_of_function_correct_l296_296650


namespace division_of_decimal_l296_296311

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l296_296311


namespace inequality_proof_l296_296680

theorem inequality_proof {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1) :
    (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 := sorry

end inequality_proof_l296_296680


namespace divisibility_by_10_l296_296445

theorem divisibility_by_10 (a : ℤ) (n : ℕ) (h : n ≥ 2) : 
  (a^(2^n + 1) - a) % 10 = 0 :=
by
  sorry

end divisibility_by_10_l296_296445


namespace circle_placement_possible_l296_296831

theorem circle_placement_possible
  (length : ℕ)
  (width : ℕ)
  (n : ℕ)
  (area_ci : ℕ)
  (ne_int_lt : length = 20)
  (ne_wid_lt : width = 25)
  (ne_squares : n = 120)
  (sm_area_lt : area_ci = 456) :
  120 * (1 + (Real.pi / 4)) < area_ci :=
by sorry

end circle_placement_possible_l296_296831


namespace mila_needs_48_hours_to_earn_as_much_as_agnes_l296_296459

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end mila_needs_48_hours_to_earn_as_much_as_agnes_l296_296459


namespace stateA_issues_more_than_stateB_l296_296576

-- Definitions based on conditions
def stateA_format : ℕ := 26^5 * 10^1
def stateB_format : ℕ := 26^3 * 10^3

-- Proof problem statement
theorem stateA_issues_more_than_stateB : stateA_format - stateB_format = 10123776 := by
  sorry

end stateA_issues_more_than_stateB_l296_296576


namespace eight_div_repeating_three_l296_296300

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l296_296300


namespace board_officer_election_l296_296485

def num_ways_choose_officers (total_members : ℕ) (elect_officers : ℕ) : ℕ :=
  -- This will represent the number of ways to choose 4 officers given 30 members
  -- with the conditions on Alice, Bob, Chris, and Dana.
  if total_members = 30 ∧ elect_officers = 4 then
    358800 + 7800 + 7800 + 24
  else
    0

theorem board_officer_election : num_ways_choose_officers 30 4 = 374424 :=
by {
  -- Proof would go here
  sorry
}

end board_officer_election_l296_296485


namespace tan_five_pi_over_four_l296_296658

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l296_296658


namespace car_sales_decrease_l296_296496

theorem car_sales_decrease (P N : ℝ) (h1 : 1.30 * P / (N * (1 - D / 100)) = 1.8571 * (P / N)) : D = 30 :=
by
  sorry

end car_sales_decrease_l296_296496


namespace sum_of_solutions_l296_296095

theorem sum_of_solutions (x : ℝ) (h : x + (25 / x) = 10) : x = 5 :=
by
  sorry

end sum_of_solutions_l296_296095


namespace trivia_team_students_l296_296739

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (total_students : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  total_students = not_picked + groups * students_per_group →
  total_students = 65 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end trivia_team_students_l296_296739


namespace second_workshop_production_l296_296125

theorem second_workshop_production (a b c : ℕ) (h₁ : a + b + c = 3600) (h₂ : a + c = 2 * b) : b * 3 = 3600 := 
by 
  sorry

end second_workshop_production_l296_296125


namespace volume_of_tetrahedron_ABCD_l296_296396

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

end volume_of_tetrahedron_ABCD_l296_296396


namespace maximum_value_at_2001_l296_296795
noncomputable def a_n (n : ℕ) : ℝ := n^2 / (1.001^n)

theorem maximum_value_at_2001 : ∃ n : ℕ, n = 2001 ∧ ∀ k : ℕ, a_n k ≤ a_n 2001 := by
  sorry

end maximum_value_at_2001_l296_296795


namespace joan_books_l296_296566

theorem joan_books (initial_books sold_books result_books : ℕ) 
  (h_initial : initial_books = 33) 
  (h_sold : sold_books = 26) 
  (h_result : initial_books - sold_books = result_books) : 
  result_books = 7 := 
by
  sorry

end joan_books_l296_296566


namespace division_of_repeating_decimal_l296_296303

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l296_296303


namespace largest_of_w_l296_296232

variable {x y z w : ℝ}

namespace MathProof

theorem largest_of_w
  (h1 : x + 3 = y - 1)
  (h2 : x + 3 = z + 5)
  (h3 : x + 3 = w - 2) :  
  w > y ∧ w > x ∧ w > z :=
by
  sorry

end MathProof

end largest_of_w_l296_296232


namespace min_value_y_l296_296374

theorem min_value_y : ∃ x : ℝ, (y = 2 * x^2 + 8 * x + 18) ∧ (∀ x : ℝ, y ≥ 10) :=
by
  sorry

end min_value_y_l296_296374


namespace minimal_ab_l296_296936

theorem minimal_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
(h : 1 / (a : ℝ) + 1 / (3 * b : ℝ) = 1 / 9) : a * b = 60 :=
sorry

end minimal_ab_l296_296936


namespace least_n_for_distance_l296_296847

-- Definitions ensuring our points and distances
def A_0 : (ℝ × ℝ) := (0, 0)

-- Assume we have distance function and equilateral triangles on given coordinates
def is_on_x_axis (p : ℕ → ℝ × ℝ) : Prop := ∀ n, (p n).snd = 0
def is_on_parabola (q : ℕ → ℝ × ℝ) : Prop := ∀ n, (q n).snd = (q n).fst^2
def is_equilateral (p : ℕ → ℝ × ℝ) (q : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  let d1 := dist (p (n-1)) (q n)
  let d2 := dist (q n) (p n)
  let d3 := dist (p (n-1)) (p n)
  d1 = d2 ∧ d2 = d3

-- Define the main property we want to prove
def main_property (n : ℕ) (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  A 0 = A_0 ∧ is_on_x_axis A ∧ is_on_parabola B ∧
  (∀ k, is_equilateral A B (k+1)) ∧
  dist A_0 (A n) ≥ 200

-- Final theorem statement
theorem least_n_for_distance (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) :
  (∃ n, main_property n A B ∧ (∀ m, main_property m A B → n ≤ m)) ↔ n = 24 := by
  sorry

end least_n_for_distance_l296_296847


namespace elias_purchased_50cent_items_l296_296378

theorem elias_purchased_50cent_items :
  ∃ (a b c : ℕ), a + b + c = 50 ∧ (50 * a + 250 * b + 400 * c = 5000) ∧ (a = 40) :=
by {
  sorry
}

end elias_purchased_50cent_items_l296_296378


namespace problem_l296_296802

theorem problem 
  (a b A B : ℝ)
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
by sorry

end problem_l296_296802


namespace sum_infinite_series_l296_296368

theorem sum_infinite_series :
  ∑ k in (Finset.range ∞), (12^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
sorry

end sum_infinite_series_l296_296368


namespace average_marks_l296_296477

theorem average_marks :
  let a1 := 76
  let a2 := 65
  let a3 := 82
  let a4 := 67
  let a5 := 75
  let n := 5
  let total_marks := a1 + a2 + a3 + a4 + a5
  let avg_marks := total_marks / n
  avg_marks = 73 :=
by
  sorry

end average_marks_l296_296477


namespace vertex_of_parabola_l296_296270

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l296_296270


namespace andy_initial_cookies_l296_296078

-- Define the conditions as constants
constant andy_ate : ℕ := 3
constant brother_received : ℕ := 5
constant team_members : ℕ := 8

-- Define the function for cookies taken by the basketball team using the problem's condition
def cookies_taken_by_nth_player (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Calculate the total cookies taken by the basketball team
def total_cookies_taken_by_team : ℕ :=
  (list.range team_members).sum (λ n, cookies_taken_by_nth_player (n + 1))

-- Summing up the initial eats, gives, and taken by team
def total_cookies (andy_ate brother_received total_cookies_taken_by_team : ℕ) : ℕ :=
  andy_ate + brother_received + total_cookies_taken_by_team

theorem andy_initial_cookies : total_cookies andy_ate brother_received total_cookies_taken_by_team = 72 :=
by
  -- Introduce a known mathematical result: sum of arithmetic sequence
  have sum_arith_seq_8 : (list.range 8).sum (λ n, cookies_taken_by_nth_player (n + 1)) = 64 := 
    by sorry -- this is actually the result we assume from the solution
  simp [total_cookies, andy_ate, brother_received, sum_arith_seq_8]

end andy_initial_cookies_l296_296078


namespace class_weighted_average_l296_296822

theorem class_weighted_average
    (num_students : ℕ)
    (sect1_avg sect2_avg sect3_avg remainder_avg : ℝ)
    (sect1_pct sect2_pct sect3_pct remainder_pct : ℝ)
    (weight1 weight2 weight3 weight4 : ℝ)
    (h_total_students : num_students = 120)
    (h_sect1_avg : sect1_avg = 96.5)
    (h_sect2_avg : sect2_avg = 78.4)
    (h_sect3_avg : sect3_avg = 88.2)
    (h_remainder_avg : remainder_avg = 64.7)
    (h_sect1_pct : sect1_pct = 0.187)
    (h_sect2_pct : sect2_pct = 0.355)
    (h_sect3_pct : sect3_pct = 0.258)
    (h_remainder_pct : remainder_pct = 1 - (sect1_pct + sect2_pct + sect3_pct))
    (h_weight1 : weight1 = 0.35)
    (h_weight2 : weight2 = 0.25)
    (h_weight3 : weight3 = 0.30)
    (h_weight4 : weight4 = 0.10) :
    (sect1_avg * weight1 + sect2_avg * weight2 + sect3_avg * weight3 + remainder_avg * weight4) * 100 = 86 := 
sorry

end class_weighted_average_l296_296822


namespace smallest_N_divisibility_l296_296091

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end smallest_N_divisibility_l296_296091


namespace eccentricity_of_given_ellipse_l296_296515

noncomputable def eccentricity_of_ellipse : ℝ :=
  let a : ℝ := 1
  let b : ℝ := 1 / 2
  let c : ℝ := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_given_ellipse :
  eccentricity_of_ellipse = Real.sqrt (3) / 2 :=
by
  -- Proof is omitted.
  sorry

end eccentricity_of_given_ellipse_l296_296515


namespace right_triangle_third_side_l296_296423

theorem right_triangle_third_side (a b : ℕ) (c : ℝ) (h₁: a = 3) (h₂: b = 4) (h₃: ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2)) ∨ (c^2 + b^2 = a^2)):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end right_triangle_third_side_l296_296423


namespace mod_calculation_l296_296932

theorem mod_calculation :
  (3 * 43 + 6 * 37) % 60 = 51 :=
by
  sorry

end mod_calculation_l296_296932


namespace number_of_monomials_l296_296425

def isMonomial (expr : String) : Bool :=
  match expr with
  | "-(2 / 3) * a^3 * b" => true
  | "(x * y) / 2" => true
  | "-4" => true
  | "0" => true
  | _ => false

def countMonomials (expressions : List String) : Nat :=
  expressions.foldl (fun acc expr => if isMonomial expr then acc + 1 else acc) 0

theorem number_of_monomials : countMonomials ["-(2 / 3) * a^3 * b", "(x * y) / 2", "-4", "-(2 / a)", "0", "x - y"] = 4 :=
by
  sorry

end number_of_monomials_l296_296425


namespace sum_arithmetic_sequence_max_l296_296106

theorem sum_arithmetic_sequence_max (d : ℝ) (a : ℕ → ℝ) 
  (h1 : d < 0) (h2 : (a 1)^2 = (a 13)^2) :
  ∃ n, n = 6 ∨ n = 7 :=
by
  sorry

end sum_arithmetic_sequence_max_l296_296106


namespace complement_P_relative_to_U_l296_296437

variable (U : Set ℝ) (P : Set ℝ)

theorem complement_P_relative_to_U (hU : U = Set.univ) (hP : P = {x : ℝ | x < 1}) : 
  U \ P = {x : ℝ | x ≥ 1} := by
  sorry

end complement_P_relative_to_U_l296_296437


namespace second_trial_addition_amount_l296_296706

variable (optimal_min optimal_max: ℝ) (phi: ℝ)

def method_618 (optimal_min optimal_max phi: ℝ) :=
  let x1 := optimal_min + (optimal_max - optimal_min) * phi
  let x2 := optimal_max + optimal_min - x1
  x2

theorem second_trial_addition_amount:
  optimal_min = 10 ∧ optimal_max = 110 ∧ phi = 0.618 →
  method_618 10 110 0.618 = 48.2 :=
by
  intro h
  simp [method_618, h]
  sorry

end second_trial_addition_amount_l296_296706


namespace speed_last_segment_l296_296709

-- Definitions corresponding to conditions
def drove_total_distance : ℝ := 150
def total_time_minutes : ℝ := 120
def time_first_segment_minutes : ℝ := 40
def speed_first_segment_mph : ℝ := 70
def speed_second_segment_mph : ℝ := 75

-- The statement of the problem
theorem speed_last_segment :
  let total_distance : ℝ := drove_total_distance
  let total_time : ℝ := total_time_minutes / 60
  let time_first_segment : ℝ := time_first_segment_minutes / 60
  let time_second_segment : ℝ := time_first_segment
  let time_last_segment : ℝ := time_first_segment
  let distance_first_segment : ℝ := speed_first_segment_mph * time_first_segment
  let distance_second_segment : ℝ := speed_second_segment_mph * time_second_segment
  let distance_two_segments : ℝ := distance_first_segment + distance_second_segment
  let distance_last_segment : ℝ := total_distance - distance_two_segments
  let speed_last_segment := distance_last_segment / time_last_segment
  speed_last_segment = 80 := 
  sorry

end speed_last_segment_l296_296709


namespace initial_observations_l296_296600

theorem initial_observations (n : ℕ) (S : ℕ) (new_obs : ℕ) :
  (S = 12 * n) → (new_obs = 5) → (S + new_obs = 11 * (n + 1)) → n = 6 :=
by
  intro h1 h2 h3
  sorry

end initial_observations_l296_296600


namespace inequality_not_always_hold_l296_296996

theorem inequality_not_always_hold (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) : ¬(∀ a b, a^3 + b^3 ≥ 2 * a * b^2) :=
sorry

end inequality_not_always_hold_l296_296996


namespace divisible_l296_296585

def P (x : ℝ) : ℝ := 6 * x^3 + x^2 - 1
def Q (x : ℝ) : ℝ := 2 * x - 1

theorem divisible : ∃ R : ℝ → ℝ, ∀ x : ℝ, P x = Q x * R x :=
sorry

end divisible_l296_296585


namespace part1_part2_l296_296424

noncomputable def cost_prices (x y : ℕ) : Prop := 
  8800 / (y + 4) = 2 * (4000 / x) ∧ 
  x = 40 ∧ 
  y = 44

theorem part1 : ∃ x y : ℕ, cost_prices x y := sorry

noncomputable def minimum_lucky_rabbits (m : ℕ) : Prop := 
  26 * m + 20 * (200 - m) ≥ 4120 ∧ 
  m = 20

theorem part2 : ∃ m : ℕ, minimum_lucky_rabbits m := sorry

end part1_part2_l296_296424


namespace intersection_A_B_l296_296810

open Set

def setA : Set ℕ := {x | x - 4 < 0}
def setB : Set ℕ := {0, 1, 3, 4}

theorem intersection_A_B : setA ∩ setB = {0, 1, 3} := by
  sorry

end intersection_A_B_l296_296810


namespace smaller_circle_radius_l296_296562

theorem smaller_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : R = (2 * r) / Real.sqrt 3) : r = 5 * Real.sqrt 3 :=
by
  sorry

end smaller_circle_radius_l296_296562


namespace tan_five_pi_over_four_l296_296653

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l296_296653


namespace rattlesnakes_count_l296_296282

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end rattlesnakes_count_l296_296282


namespace proof_set_intersection_l296_296116

def set_M := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def set_N := {x : ℝ | Real.log x ≥ 0}
def set_answer := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem proof_set_intersection : 
  (set_M ∩ set_N) = set_answer := 
by 
  sorry

end proof_set_intersection_l296_296116


namespace problem1_problem2_l296_296778

theorem problem1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 :=
  sorry

theorem problem2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 :=
  sorry

end problem1_problem2_l296_296778


namespace max_expression_value_l296_296415

theorem max_expression_value :
  ∀ (a b : ℝ), (100 ≤ a ∧ a ≤ 500) → (500 ≤ b ∧ b ≤ 1500) → 
  (∃ x, x = (b - 100) / (a + 50) ∧ ∀ y, y = (b - 100) / (a + 50) → y ≤ (28 / 3)) :=
by
  sorry

end max_expression_value_l296_296415


namespace opposite_of_neg_seven_thirds_l296_296039

def opposite (x : ℚ) : ℚ := -x

theorem opposite_of_neg_seven_thirds : opposite (-7 / 3) = 7 / 3 := 
by
  -- Proof of this theorem
  sorry

end opposite_of_neg_seven_thirds_l296_296039


namespace binary_multiplication_l296_296090

theorem binary_multiplication :
  let a := 0b1101101
  let b := 0b1011
  let product := 0b10001001111
  a * b = product :=
sorry

end binary_multiplication_l296_296090


namespace minimum_perimeter_is_728_l296_296891

noncomputable def minimum_common_perimeter (a b c : ℤ) (h1 : 2 * a + 18 * c = 2 * b + 20 * c)
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : ℤ :=
2 * a + 18 * c

theorem minimum_perimeter_is_728 (a b c : ℤ) 
  (h1 : 2 * a + 18 * c = 2 * b + 20 * c) 
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : 
  minimum_common_perimeter a b c h1 h2 h3 = 728 :=
sorry

end minimum_perimeter_is_728_l296_296891


namespace simplify_fraction_sum_l296_296265

theorem simplify_fraction_sum :
  (3 / 462) + (17 / 42) + (1 / 11) = 116 / 231 := 
by
  sorry

end simplify_fraction_sum_l296_296265


namespace josh_500_coins_impossible_l296_296752

theorem josh_500_coins_impossible : ¬ ∃ (x y : ℕ), x + y ≤ 500 ∧ 36 * x + 6 * y + (500 - x - y) = 3564 := 
sorry

end josh_500_coins_impossible_l296_296752


namespace sin_identity_l296_296525

theorem sin_identity (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := sorry

end sin_identity_l296_296525


namespace factor_expression_l296_296088

theorem factor_expression (x : ℝ) : 2 * x * (x + 3) + (x + 3) = (2 * x + 1) * (x + 3) :=
by
  sorry

end factor_expression_l296_296088


namespace rowing_distance_l296_296765

theorem rowing_distance
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (total_time : ℝ)
  (D : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : total_time = 15)
  (h4 : D / (rowing_speed + current_speed) + D / (rowing_speed - current_speed) = total_time) :
  D = 72 := 
sorry

end rowing_distance_l296_296765


namespace find_x_with_conditions_l296_296803

theorem find_x_with_conditions (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1)
  (h2 : (nat.factors x).to_finset.card = 3) (h3 : 11 ∈ (nat.factors x).to_finset) :
  x = 59048 := 
by {
  sorry
}

end find_x_with_conditions_l296_296803


namespace correct_division_l296_296470

theorem correct_division (a : ℝ) : a^8 / a^2 = a^6 := by 
  sorry

end correct_division_l296_296470


namespace probability_of_odd_numbers_and_twos_l296_296176

open Finset BigOperators

noncomputable def probability_odd_numbers_exactly_six_and_two_of_them_are_three : ℚ :=
  let binom := λ n k: ℕ, (finset.range (n + 1)).powerset.filter (λ s, s.card = k).card
  let probability := (binom 8 6 : ℚ) * (1 / 2) ^ 8 * (binom 6 2 : ℚ) * (1 / 3) ^ 2 * (2 / 3) ^ 4
  in probability
  
theorem probability_of_odd_numbers_and_twos : probability_odd_numbers_exactly_six_and_two_of_them_are_three = 35 / 972 := by
  sorry

end probability_of_odd_numbers_and_twos_l296_296176


namespace findTwoHeaviestStonesWith35Weighings_l296_296286

-- Define the problem with conditions
def canFindTwoHeaviestStones (stones : Fin 32 → ℝ) (weighings : ℕ) : Prop :=
  ∀ (balanceScale : (Fin 32 × Fin 32) → Bool), weighings ≤ 35 → 
  ∃ (heaviest : Fin 32) (secondHeaviest : Fin 32), 
  (heaviest ≠ secondHeaviest) ∧ 
  (∀ i : Fin 32, stones heaviest ≥ stones i) ∧ 
  (∀ j : Fin 32, j ≠ heaviest → stones secondHeaviest ≥ stones j)

-- Formally state the theorem
theorem findTwoHeaviestStonesWith35Weighings (stones : Fin 32 → ℝ) :
  canFindTwoHeaviestStones stones 35 :=
sorry -- Proof is omitted

end findTwoHeaviestStonesWith35Weighings_l296_296286


namespace min_value_of_M_l296_296542

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem min_value_of_M (M : ℝ) (hM : M = Real.sqrt 2) :
  ∀ (a b c : ℝ), a > M → b > M → c > M → a^2 + b^2 = c^2 → 
  (f a) + (f b) > f c ∧ (f a) + (f c) > f b ∧ (f b) + (f c) > f a :=
by
  sorry

end min_value_of_M_l296_296542


namespace roots_reciprocal_sum_eq_three_halves_l296_296109

theorem roots_reciprocal_sum_eq_three_halves
  {a b : ℝ}
  (h1 : a^2 - 6 * a + 4 = 0)
  (h2 : b^2 - 6 * b + 4 = 0)
  (h_roots : a ≠ b) :
  1/a + 1/b = 3/2 := by
  sorry

end roots_reciprocal_sum_eq_three_halves_l296_296109


namespace tiffany_initial_lives_l296_296610

variable (x : ℝ) -- Define the variable x representing the initial number of lives

-- Define the conditions
def condition1 : Prop := x + 14.0 + 27.0 = 84.0

-- Prove the initial number of lives
theorem tiffany_initial_lives (h : condition1 x) : x = 43.0 := by
  sorry

end tiffany_initial_lives_l296_296610


namespace mikko_should_attempt_least_questions_l296_296864

theorem mikko_should_attempt_least_questions (p : ℝ) (h_p : 0 < p ∧ p < 1) : 
  ∃ (x : ℕ), x ≥ ⌈1 / (2 * p - 1)⌉ :=
by
  sorry

end mikko_should_attempt_least_questions_l296_296864


namespace function_is_odd_and_increasing_l296_296058

theorem function_is_odd_and_increasing :
  (∀ x : ℝ, (x^(1/3) : ℝ) = -( (-x)^(1/3) : ℝ)) ∧ (∀ x y : ℝ, x < y → (x^(1/3) : ℝ) < (y^(1/3) : ℝ)) :=
by
  sorry

end function_is_odd_and_increasing_l296_296058


namespace petals_in_garden_l296_296941

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end petals_in_garden_l296_296941


namespace certain_number_eq_0_08_l296_296753

theorem certain_number_eq_0_08 (x : ℝ) (h : 1 / x = 12.5) : x = 0.08 :=
by
  sorry

end certain_number_eq_0_08_l296_296753


namespace simple_interest_years_l296_296492

theorem simple_interest_years (P : ℝ) (difference : ℝ) (N : ℝ) : 
  P = 2300 → difference = 69 → (23 * N = 69) → N = 3 :=
by
  intros hP hdifference heq
  sorry

end simple_interest_years_l296_296492


namespace choosing_top_cases_l296_296609

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end choosing_top_cases_l296_296609


namespace bound_seq_l296_296026

def is_triplet (x y z : ℕ) : Prop := 
  x = (y + z) / 2 ∨ y = (x + z) / 2 ∨ z = (x + y) / 2 

def seq_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ 
  ∀ n > 2, a n = (Minimal z, ∀ i j k < n, ¬is_triplet (a i) (a j) (a k))

theorem bound_seq (a : ℕ → ℕ) (h : seq_condition a) : ∀ n, a n ≤ (n^2 + 7) / 8 :=
by
  sorry

end bound_seq_l296_296026


namespace tim_used_to_run_days_l296_296611

def hours_per_day := 2
def total_hours_per_week := 10
def added_days := 2

theorem tim_used_to_run_days (runs_per_day : ℕ) (total_weekly_runs : ℕ) (additional_runs : ℕ) : 
  runs_per_day = hours_per_day →
  total_weekly_runs = total_hours_per_week →
  additional_runs = added_days →
  (total_weekly_runs / runs_per_day) - additional_runs = 3 :=
by
  intros h1 h2 h3
  sorry

end tim_used_to_run_days_l296_296611


namespace find_x_exists_unique_l296_296806

theorem find_x_exists_unique (n : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ x = p * q * r) (h3 : 11 ∣ x) : x = 59048 :=
sorry

end find_x_exists_unique_l296_296806


namespace equation1_solution_equation2_solution_l296_296447

-- Equation 1 Statement
theorem equation1_solution (x : ℝ) : 
  (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ↔ x = -20 :=
by sorry

-- Equation 2 Statement
theorem equation2_solution (x : ℝ) : 
  (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ↔ x = 67 / 23 :=
by sorry

end equation1_solution_equation2_solution_l296_296447


namespace find_sum_of_min_area_ks_l296_296108

def point := ℝ × ℝ

def A : point := (2, 9)
def B : point := (14, 18)

def is_int (k : ℝ) : Prop := ∃ (n : ℤ), k = n

def min_triangle_area (P Q R : point) : ℝ := sorry
-- Placeholder for the area formula of a triangle given three points

def valid_ks (k : ℝ) : Prop :=
  is_int k ∧ min_triangle_area A B (6, k) ≠ 0

theorem find_sum_of_min_area_ks :
  (∃ k1 k2 : ℤ, valid_ks k1 ∧ valid_ks k2 ∧ (k1 + k2) = 31) :=
sorry

end find_sum_of_min_area_ks_l296_296108


namespace find_integer_k_l296_296220

theorem find_integer_k (k x : ℤ) (h : (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) (hx : x > 0) :
  k = 1 ∨ k = 2 ∨ k = 3 :=
sorry

end find_integer_k_l296_296220


namespace range_of_m_l296_296535

variable {α : Type*} [LinearOrder α]

def increasing (f : α → α) : Prop :=
  ∀ ⦃x y : α⦄, x < y → f x < f y

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h_inc : increasing f) 
  (h_cond : ∀ m : ℝ, f (m + 3) ≤ f 5) : 
  {m : ℝ | f (m + 3) ≤ f 5} = {m : ℝ | m ≤ 2} := 
sorry

end range_of_m_l296_296535


namespace Louie_monthly_payment_l296_296577

noncomputable def monthly_payment (P : ℕ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  let A := P * (1 + r) ^ n
  A / 3

theorem Louie_monthly_payment : 
  monthly_payment 1000 0.10 3 (3 / 12) = 444 := 
by
  -- computation and rounding
  sorry

end Louie_monthly_payment_l296_296577


namespace find_some_number_l296_296011

theorem find_some_number (a : ℕ) (some_number : ℕ)
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 35 * some_number * 35) :
  some_number = 21 :=
by
  sorry

end find_some_number_l296_296011


namespace cost_price_of_article_l296_296629

-- Definitions based on the conditions
def sellingPrice : ℝ := 800
def profitPercentage : ℝ := 25

-- Statement to prove the cost price
theorem cost_price_of_article :
  ∃ cp : ℝ, profitPercentage = ((sellingPrice - cp) / cp) * 100 ∧ cp = 640 :=
by
  sorry

end cost_price_of_article_l296_296629


namespace remainder_3_pow_20_mod_5_l296_296893

theorem remainder_3_pow_20_mod_5 : (3 ^ 20) % 5 = 1 := by
  sorry

end remainder_3_pow_20_mod_5_l296_296893


namespace percentage_markup_l296_296896

variable (W R : ℝ) -- W is the wholesale cost, R is the normal retail price

-- The condition that, at 60% discount, the sale price nets a 35% profit on the wholesale cost
variable (h : 0.4 * R = 1.35 * W)

-- The goal statement to prove
theorem percentage_markup (h : 0.4 * R = 1.35 * W) : ((R - W) / W) * 100 = 237.5 :=
by
  sorry

end percentage_markup_l296_296896


namespace chord_length_circle_M_eq_l296_296394

noncomputable def circle_eq : Float -> Float -> Prop := fun x y => x^2 + y^2 = 8

def point_P0 : Prod Float Float := (-1, 2)
def point_C : Prod Float Float := (3, 0)

def eq_chord_len (α : Float) : Float := 
  if α = 135 then (2 * Float.sqrt((2 * Float.sqrt(2))^2 - (Float.sqrt(2) / 2)^2))
  else 0

def eq_circle_M (M_x M_y : Float) (M_r : Float) : Prop := 
  (M_x - 1/4)^2 + (M_y + 1/2)^2 = M_r^2

theorem chord_length (α : Float) : Prop :=
  α = 135 -> eq_chord_len α = Float.sqrt 30

theorem circle_M_eq : Prop :=
  eq_circle_M (1/4) (-1/2) (Float.sqrt (125 / 16))

end chord_length_circle_M_eq_l296_296394


namespace Zain_coins_total_l296_296332

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l296_296332


namespace shopkeeper_total_cards_l296_296072

-- Definition of the number of cards in standard, Uno, and tarot decks.
def std_deck := 52
def uno_deck := 108
def tarot_deck := 78

-- Number of complete decks and additional cards.
def std_decks := 4
def uno_decks := 3
def tarot_decks := 5
def additional_std := 12
def additional_uno := 7
def additional_tarot := 9

-- Calculate the total number of cards.
def total_standard_cards := (std_decks * std_deck) + additional_std
def total_uno_cards := (uno_decks * uno_deck) + additional_uno
def total_tarot_cards := (tarot_decks * tarot_deck) + additional_tarot

def total_cards := total_standard_cards + total_uno_cards + total_tarot_cards

theorem shopkeeper_total_cards : total_cards = 950 := by
  sorry

end shopkeeper_total_cards_l296_296072


namespace theta_third_quadrant_l296_296798

theorem theta_third_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  π < θ ∧ θ < 3 * π / 2 :=
by 
  sorry

end theta_third_quadrant_l296_296798


namespace inequality_proof_l296_296342

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end inequality_proof_l296_296342


namespace probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l296_296915

-- Define the total number of ways to choose 3 leaders from 6 students
def total_ways : ℕ := Nat.choose 6 3

-- Calculate the number of ways in which boy A or girl B is chosen
def boy_A_chosen_ways : ℕ := Nat.choose 4 2 + 4 * 2
def girl_B_chosen_ways : ℕ := Nat.choose 4 1 + Nat.choose 4 2
def either_boy_A_or_girl_B_chosen_ways : ℕ := boy_A_chosen_ways + girl_B_chosen_ways

-- Calculate the probability that either boy A or girl B is chosen
def probability_either_boy_A_or_girl_B : ℚ := either_boy_A_or_girl_B_chosen_ways / total_ways

-- Calculate the probability that girl B is chosen
def girl_B_total_ways : ℕ := Nat.choose 5 2
def probability_B : ℚ := girl_B_total_ways / total_ways

-- Calculate the probability that both boy A and girl B are chosen
def both_A_and_B_chosen_ways : ℕ := Nat.choose 4 1
def probability_AB : ℚ := both_A_and_B_chosen_ways / total_ways

-- Calculate the conditional probability P(A|B) given P(B)
def conditional_probability_A_given_B : ℚ := probability_AB / probability_B

-- Theorem statements
theorem probability_either_boy_A_or_girl_B_correct : probability_either_boy_A_or_girl_B = (4 / 5) := sorry
theorem probability_B_correct : probability_B = (1 / 2) := sorry
theorem conditional_probability_A_given_B_correct : conditional_probability_A_given_B = (2 / 5) := sorry

end probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l296_296915


namespace number_of_small_jars_l296_296283

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 :=
by
  sorry

end number_of_small_jars_l296_296283


namespace largest_number_of_positive_consecutive_integers_l296_296614

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end largest_number_of_positive_consecutive_integers_l296_296614


namespace max_angle_line_plane_l296_296978

theorem max_angle_line_plane (θ : ℝ) (h_angle : θ = 72) :
  ∃ φ : ℝ, φ = 90 ∧ (72 ≤ φ ∧ φ ≤ 90) :=
by sorry

end max_angle_line_plane_l296_296978


namespace regular_octagon_exterior_angle_l296_296426

theorem regular_octagon_exterior_angle : 
  ∀ (n : ℕ), n = 8 → (180 * (n - 2) / n) + (180 - (180 * (n - 2) / n)) = 180 := by
  sorry

end regular_octagon_exterior_angle_l296_296426


namespace number_of_rectangular_arrays_l296_296355

theorem number_of_rectangular_arrays (n : ℕ) (h : n = 48) : 
  ∃ k : ℕ, (k = 6 ∧ ∀ m p : ℕ, m * p = n → m ≥ 3 → p ≥ 3 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 8 ∨ m = 12 ∨ m = 16 ∨ m = 24) :=
by
  sorry

end number_of_rectangular_arrays_l296_296355


namespace jan_drove_more_than_ian_l296_296748

variables (d t s : ℝ)

-- Ian's distance relation
def ian_distance : Prop := d = s * t

-- Han's driving relation derived by simplifying d + 100 = (s + 10) * (t + 2)
def han_condition : Prop := 5 * t + s = 40

-- Jan's distance relation
def jan_condition : Prop := m = (s + 15) * (t + 3)

-- The final proposition we need to prove
def jan_drove_165_more : Prop := (s + 15) * (t + 3) - d = 165

-- Final theorem statement
theorem jan_drove_more_than_ian (h1 : ian_distance d t s) (h2 : han_condition d t s) (h3 : jan_condition d t s) :
  jan_drove_165_more d t s :=
sorry

end jan_drove_more_than_ian_l296_296748


namespace min_value_of_reciprocal_sum_l296_296572

theorem min_value_of_reciprocal_sum (a b c : ℝ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) (h2 : a + b + c = 3) : 
  ∀ x: ℝ, (x = a ∨ x = b ∨ x = c) → ( ∃ m : ℝ, m = 3 ∧ ∀ x, ( ∑ i in {a, b, c}, 1 / i ) ≥ m) :=
begin
  sorry  -- proof not required
end

end min_value_of_reciprocal_sum_l296_296572


namespace find_m_l296_296800

open Real

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) (m : ℝ) : ℝ :=
  2 * cos (ω * x + ϕ) + m

theorem find_m (ω ϕ : ℝ) (hω : 0 < ω)
  (symmetry : ∀ t : ℝ,  f (π / 4 - t) ω ϕ m = f t ω ϕ m)
  (f_π_8 : f (π / 8) ω ϕ m = -1) :
  m = -3 ∨ m = 1 := 
sorry

end find_m_l296_296800


namespace handshakes_at_gathering_l296_296591

def total_handshakes (num_couples : ℕ) (exceptions : ℕ) : ℕ :=
  let num_people := 2 * num_couples
  let handshakes_per_person := num_people - exceptions - 1
  num_people * handshakes_per_person / 2

theorem handshakes_at_gathering : total_handshakes 6 2 = 54 := by
  sorry

end handshakes_at_gathering_l296_296591


namespace trapezoid_base_lengths_l296_296493

noncomputable def trapezoid_bases (d h : Real) : Real × Real :=
  let b := h - 2 * d
  let B := h + 2 * d
  (b, B)

theorem trapezoid_base_lengths :
  ∀ (d : Real), d = Real.sqrt 3 →
  ∀ (h : Real), h = Real.sqrt 48 →
  ∃ (b B : Real), trapezoid_bases d h = (b, B) ∧ b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ B = Real.sqrt 48 + 2 * Real.sqrt 3 := by 
  sorry

end trapezoid_base_lengths_l296_296493


namespace money_left_after_purchases_is_correct_l296_296581

noncomputable def initial_amount : ℝ := 12.50
noncomputable def cost_pencil : ℝ := 1.25
noncomputable def cost_notebook : ℝ := 3.45
noncomputable def cost_pens : ℝ := 4.80

noncomputable def total_cost : ℝ := cost_pencil + cost_notebook + cost_pens
noncomputable def money_left : ℝ := initial_amount - total_cost

theorem money_left_after_purchases_is_correct : money_left = 3.00 :=
by
  -- proof goes here, skipping with sorry for now
  sorry

end money_left_after_purchases_is_correct_l296_296581


namespace find_k_l296_296923

theorem find_k (k : ℝ) : 
  (∃ c1 c2 : ℝ, (2 * c1^2 + 5 * c1 = k) ∧ 
                (2 * c2^2 + 5 * c2 = k) ∧ 
                (c1 > c2) ∧ 
                (c1 - c2 = 5.5)) → 
  k = 12 := 
by
  intros h
  obtain ⟨c1, c2, h1, h2, h3, h4⟩ := h
  sorry

end find_k_l296_296923


namespace proj_a_b_l296_296117

open Real

def vector (α : Type*) := (α × α)

noncomputable def dot_product (a b: vector ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v: vector ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def projection (a b: vector ℝ) : ℝ := (dot_product a b) / (magnitude b)

-- Define the vectors a and b
def a : vector ℝ := (-1, 3)
def b : vector ℝ := (3, 4)

-- The projection of a in the direction of b
theorem proj_a_b : projection a b = 9 / 5 := 
  by sorry

end proj_a_b_l296_296117


namespace num_valid_n_l296_296388

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end num_valid_n_l296_296388


namespace cost_per_person_l296_296786

-- Definitions based on conditions
def totalCost : ℕ := 13500
def numberOfFriends : ℕ := 15

-- Main statement
theorem cost_per_person : totalCost / numberOfFriends = 900 :=
by sorry

end cost_per_person_l296_296786


namespace average_of_other_two_numbers_l296_296880

theorem average_of_other_two_numbers
  (avg_5_numbers : ℕ → ℚ)
  (sum_3_numbers : ℕ → ℚ)
  (h1 : ∀ n, avg_5_numbers n = 20)
  (h2 : ∀ n, sum_3_numbers n = 48)
  (h3 : ∀ n, ∃ x y z p q : ℚ, avg_5_numbers n = (x + y + z + p + q) / 5)
  (h4 : ∀ n, sum_3_numbers n = x + y + z) :
  ∃ u v : ℚ, ((u + v) / 2 = 26) :=
by sorry

end average_of_other_two_numbers_l296_296880


namespace depth_of_ship_l296_296491

-- Condition definitions
def rate : ℝ := 80  -- feet per minute
def time : ℝ := 50  -- minutes

-- Problem Statement
theorem depth_of_ship : rate * time = 4000 :=
by
  sorry

end depth_of_ship_l296_296491


namespace gumball_problem_l296_296708
-- Step d: Lean 4 statement conversion

/-- 
  Suppose Joanna initially had 40 gumballs, Jacques had 60 gumballs, 
  and Julia had 80 gumballs.
  Joanna purchased 5 times the number of gumballs she initially had,
  Jacques purchased 3 times the number of gumballs he initially had,
  and Julia purchased 2 times the number of gumballs she initially had.
  Prove that after adding their purchases:
  1. Each person will have 240 gumballs.
  2. If they combine all their gumballs and share them equally, 
     each person will still get 240 gumballs.
-/
theorem gumball_problem :
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  (joanna_final = 240) ∧ (jacques_final = 240) ∧ (julia_final = 240) ∧ 
  (total_gumballs / 3 = 240) :=
by
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  
  have h_joanna : joanna_final = 240 := sorry
  have h_jacques : jacques_final = 240 := sorry
  have h_julia : julia_final = 240 := sorry
  have h_total : total_gumballs / 3 = 240 := sorry
  
  exact ⟨h_joanna, h_jacques, h_julia, h_total⟩

end gumball_problem_l296_296708


namespace tan_five_pi_over_four_l296_296668

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l296_296668


namespace sqrt_abc_sum_is_72_l296_296144

noncomputable def abc_sqrt_calculation (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) : ℝ :=
  sqrt (a * b * c * (a + b + c))

theorem sqrt_abc_sum_is_72 (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) :
  abc_sqrt_calculation a b c h1 h2 h3 = 72 :=
by
  sorry

end sqrt_abc_sum_is_72_l296_296144


namespace excluded_angle_sum_1680_degrees_l296_296766

theorem excluded_angle_sum_1680_degrees (sum_except_one : ℝ) (h : sum_except_one = 1680) : 
  (180 - (1680 % 180)) = 120 :=
by
  have mod_eq : 1680 % 180 = 60 := by sorry
  rw [mod_eq]

end excluded_angle_sum_1680_degrees_l296_296766


namespace paperclips_exceed_target_in_days_l296_296844

def initial_paperclips := 3
def ratio := 2
def target_paperclips := 200

theorem paperclips_exceed_target_in_days :
  ∃ k : ℕ, initial_paperclips * ratio ^ k > target_paperclips ∧ k = 8 :=
by {
  sorry
}

end paperclips_exceed_target_in_days_l296_296844


namespace min_max_area_of_CDM_l296_296838

theorem min_max_area_of_CDM (x y z : ℕ) (h1 : 2 * x + y = 4) (h2 : 2 * y + z = 8) :
  z = 4 :=
by
  sorry

end min_max_area_of_CDM_l296_296838


namespace intersection_complement_l296_296545

def set_M : Set ℝ := {x : ℝ | x^2 - x = 0}

def set_N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = 2 * n + 1}

theorem intersection_complement (h : UniversalSet = Set.univ) :
  set_M ∩ (UniversalSet \ set_N) = {0} := 
sorry

end intersection_complement_l296_296545


namespace first_discount_percentage_l296_296732

noncomputable def saree_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100)

theorem first_discount_percentage (x : ℝ) : saree_price 400 x 20 = 240 → x = 25 :=
by sorry

end first_discount_percentage_l296_296732


namespace impossible_relationships_l296_296399

theorem impossible_relationships (a b : ℝ) (h : (1 / a) = (1 / b)) :
  (¬ (0 < a ∧ a < b)) ∧ (¬ (b < a ∧ a < 0)) :=
by
  sorry

end impossible_relationships_l296_296399


namespace other_bill_denomination_l296_296467

-- Define the conditions of the problem
def cost_shirt : ℕ := 80
def ten_dollar_bills : ℕ := 2
def other_bills (x : ℕ) : ℕ := ten_dollar_bills + 1

-- The amount paid with $10 bills
def amount_with_ten_dollar_bills : ℕ := ten_dollar_bills * 10

-- The total amount should match the cost of the shirt
def total_amount (x : ℕ) : ℕ := amount_with_ten_dollar_bills + (other_bills x) * x

-- Statement to prove
theorem other_bill_denomination : 
  ∃ (x : ℕ), total_amount x = cost_shirt ∧ x = 20 :=
by
  sorry

end other_bill_denomination_l296_296467


namespace proof_A_cap_complement_B_l296_296402

variable (A B U : Set ℕ) (h1 : A ⊆ U) (h2 : B ⊆ U)
variable (h3 : U = {1, 2, 3, 4})
variable (h4 : (U \ (A ∪ B)) = {4}) -- \ represents set difference, complement in the universal set
variable (h5 : B = {1, 2})

theorem proof_A_cap_complement_B : A ∩ (U \ B) = {3} := by
  sorry

end proof_A_cap_complement_B_l296_296402


namespace ones_digit_of_power_l296_296787

theorem ones_digit_of_power (n : ℕ) : 
  (13 ^ (13 * (12 ^ 12)) % 10) = 9 :=
by
  sorry

end ones_digit_of_power_l296_296787


namespace decimal_to_fraction_sum_l296_296897

def recurring_decimal_fraction_sum : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ gcd a b = 1 ∧ (a / b : ℚ) = (0.345345345 : ℚ) ∧ a + b = 226

theorem decimal_to_fraction_sum :
  recurring_decimal_fraction_sum :=
sorry

end decimal_to_fraction_sum_l296_296897


namespace infinite_series_equals_3_l296_296367

noncomputable def infinite_series_sum := ∑' (k : ℕ), (12^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

theorem infinite_series_equals_3 : infinite_series_sum = 3 := by
  sorry

end infinite_series_equals_3_l296_296367


namespace average_first_n_numbers_eq_10_l296_296381

theorem average_first_n_numbers_eq_10 (n : ℕ) 
  (h : (n * (n + 1)) / (2 * n) = 10) : n = 19 :=
  sorry

end average_first_n_numbers_eq_10_l296_296381


namespace runners_meet_l296_296285

theorem runners_meet (T : ℕ) 
  (h1 : T > 4) 
  (h2 : Nat.lcm 2 (Nat.lcm 4 T) = 44) : 
  T = 11 := 
sorry

end runners_meet_l296_296285


namespace problem1_solution_problem2_expected_value_solution_problem2_variance_solution_l296_296237

noncomputable def problem1 (P : ProbabilityMassFunction (Fin 6)) : ℝ :=
let P_white := 1 / 3 in
let P_black := 1 - P_white in
P_white * P_black + P_black * P_white

theorem problem1_solution : 
  problem1 = 4 / 9 := 
sorry

noncomputable def problem2_expected_value (P : ProbabilityMassFunction (Fin 2)) : ℝ :=
let P_xi_0 := (4 / 6) * (3 / 5) in
let P_xi_1 := (4 / 6) * (2 / 5) + (2 / 6) * (4 / 5) in
let P_xi_2 := (2 / 6) * (1 / 5) in
0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2

theorem problem2_expected_value_solution : 
  problem2_expected_value = 2 / 3 := 
sorry

noncomputable def problem2_variance (P : ProbabilityMassFunction (Fin 2)) : ℝ :=
let expected_value := problem2_expected_value in
let P_xi_0 := (4 / 6) * (3 / 5) in
let P_xi_1 := (4 / 6) * (2 / 5) + (2 / 6) * (4 / 5) in
let P_xi_2 := (2 / 6) * (1 / 5) in
(0 - expected_value)^2 * P_xi_0 + (1 - expected_value)^2 * P_xi_1 + (2 - expected_value)^2 * P_xi_2

theorem problem2_variance_solution : 
  problem2_variance = 16 / 45 := 
sorry

end problem1_solution_problem2_expected_value_solution_problem2_variance_solution_l296_296237


namespace total_cost_family_visit_l296_296835

/-
Conditions:
1. entrance_ticket_cost: $5 per person
2. attraction_ticket_cost_kid: $2 per kid
3. attraction_ticket_cost_parent: $4 per parent
4. family_discount_threshold: A family of 6 or more gets a 10% discount on entrance tickets
5. senior_discount: Senior citizens get a 50% discount on attraction tickets
6. family_composition: 4 children, 2 parents, and 1 grandmother
7. visit_attraction: The family plans to visit at least one attraction
-/

def entrance_ticket_cost : ℝ := 5
def attraction_ticket_cost_kid : ℝ := 2
def attraction_ticket_cost_parent : ℝ := 4
def family_discount_threshold : ℕ := 6
def family_discount_rate : ℝ := 0.10
def senior_discount_rate : ℝ := 0.50
def number_of_kids : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_seniors : ℕ := 1

theorem total_cost_family_visit : 
  let total_entrance_fee := (number_of_kids + number_of_parents + number_of_seniors) * entrance_ticket_cost 
  let total_entrance_fee_discounted := total_entrance_fee * (1 - family_discount_rate)
  let total_attraction_fee_kids := number_of_kids * attraction_ticket_cost_kid
  let total_attraction_fee_parents := number_of_parents * attraction_ticket_cost_parent
  let total_attraction_fee_seniors := number_of_seniors * attraction_ticket_cost_parent * (1 - senior_discount_rate)
  let total_attraction_fee := total_attraction_fee_kids + total_attraction_fee_parents + total_attraction_fee_seniors
  (number_of_kids + number_of_parents + number_of_seniors ≥ family_discount_threshold) → 
  (total_entrance_fee_discounted + total_attraction_fee = 49.50) :=
by
  -- Assuming we calculate entrance fee and attraction fee correctly, state the theorem
  sorry

end total_cost_family_visit_l296_296835


namespace inequality_proof_l296_296432

open Real

theorem inequality_proof (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l296_296432


namespace range_of_a_l296_296678

variable (a x y : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q : Prop :=
  (1 - a) * (a - 3) < 0

theorem range_of_a (h1 : proposition_p a) (h2 : proposition_q a) : 
  (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4) :=
by
  sorry

end range_of_a_l296_296678


namespace not_q_true_l296_296694

theorem not_q_true (p q : Prop) (hp : p = true) (hq : q = false) : ¬q = true :=
by
  sorry

end not_q_true_l296_296694


namespace last_four_digits_of_5_pow_2017_l296_296719

theorem last_four_digits_of_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_of_5_pow_2017_l296_296719


namespace smallest_satisfying_N_is_2520_l296_296093

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end smallest_satisfying_N_is_2520_l296_296093


namespace probability_no_adjacent_same_rolls_l296_296949

theorem probability_no_adjacent_same_rolls : 
  let A := [0, 1, 2, 3, 4, 5] -- Representing six faces of a die
  let rollings : List (A → ℕ) -- Each person rolls and the result is represented as a map from faces to counts (a distribution in effect)
  ∃ rollings : List (A → ℕ), 
    (∀ (i : Fin 5), rollings[i] ≠ rollings[(i + 1) % 5]) →
      probability rollings
    = 375 / 2592 :=
by
  sorry

end probability_no_adjacent_same_rolls_l296_296949


namespace fourth_power_sqrt_eq_256_l296_296205

theorem fourth_power_sqrt_eq_256 (x : ℝ) (h : (x^(1/2))^4 = 256) : x = 16 := by sorry

end fourth_power_sqrt_eq_256_l296_296205


namespace original_employee_count_l296_296190

theorem original_employee_count (employees_operations : ℝ) 
                                (employees_sales : ℝ) 
                                (employees_finance : ℝ) 
                                (employees_hr : ℝ) 
                                (employees_it : ℝ) 
                                (h1 : employees_operations / 0.82 = 192)
                                (h2 : employees_sales / 0.75 = 135)
                                (h3 : employees_finance / 0.85 = 123)
                                (h4 : employees_hr / 0.88 = 66)
                                (h5 : employees_it / 0.90 = 90) : 
                                employees_operations + employees_sales + employees_finance + employees_hr + employees_it = 734 :=
sorry

end original_employee_count_l296_296190


namespace train_speed_84_kmph_l296_296065

theorem train_speed_84_kmph (length : ℕ) (time : ℕ) (conversion_factor : ℚ)
  (h_length : length = 140) (h_time : time = 6) (h_conversion_factor : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 84 :=
  sorry

end train_speed_84_kmph_l296_296065


namespace matrix_multiplication_l296_296569

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_multiplication :
  (A - B = A * B) →
  (A * B = ![![7, -2], ![4, -3]]) →
  (B * A = ![![6, -2], ![4, -4]]) :=
by
  intros h₁ h₂
  sorry

end matrix_multiplication_l296_296569


namespace man_mass_calculation_l296_296185

/-- A boat has a length of 4 m, a breadth of 2 m, and a weight of 300 kg.
    The density of the water is 1000 kg/m³.
    When the man gets on the boat, it sinks by 1 cm.
    Prove that the mass of the man is 80 kg. -/
theorem man_mass_calculation :
  let length_boat := 4     -- in meters
  let breadth_boat := 2    -- in meters
  let weight_boat := 300   -- in kg
  let density_water := 1000  -- in kg/m³
  let additional_depth := 0.01 -- in meters (1 cm)
  volume_displaced = length_boat * breadth_boat * additional_depth →
  mass_water_displaced = volume_displaced * density_water →
  mass_of_man = mass_water_displaced →
  mass_of_man = 80 :=
by 
  intros length_boat breadth_boat weight_boat density_water additional_depth volume_displaced
  intros mass_water_displaced mass_of_man
  sorry

end man_mass_calculation_l296_296185


namespace total_students_correct_l296_296049

def students_in_general_hall : ℕ := 30
def students_in_biology_hall : ℕ := 2 * students_in_general_hall
def combined_students_general_biology : ℕ := students_in_general_hall + students_in_biology_hall
def students_in_math_hall : ℕ := (3 * combined_students_general_biology) / 5
def total_students_in_all_halls : ℕ := students_in_general_hall + students_in_biology_hall + students_in_math_hall

theorem total_students_correct : total_students_in_all_halls = 144 := by
  -- Proof omitted, it should be
  sorry

end total_students_correct_l296_296049


namespace equilateral_triangles_circle_l296_296943

-- Definitions and conditions
structure Triangle :=
  (A B C : ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 12)

structure Circle :=
  (S : ℝ)

def PointOnArc (P1 P2 P : ℝ) : Prop :=
  -- Definition to describe P lies on the arc P1P2
  sorry

-- Theorem stating the proof problem
theorem equilateral_triangles_circle
  (S : Circle)
  (T1 T2 : Triangle)
  (H1 : T1.side_length = 12)
  (H2 : T2.side_length = 12)
  (HAonArc : PointOnArc T2.B T2.C T1.A)
  (HBonArc : PointOnArc T2.A T2.B T1.B) :
  (T1.A - T2.A) ^ 2 + (T1.B - T2.B) ^ 2 + (T1.C - T2.C) ^ 2 = 288 :=
sorry

end equilateral_triangles_circle_l296_296943


namespace find_monthly_fee_l296_296517

variable (monthly_fee : ℝ) (cost_per_minute : ℝ := 0.12) (minutes_used : ℕ := 178) (total_bill : ℝ := 23.36)

theorem find_monthly_fee
  (h1 : total_bill = monthly_fee + (cost_per_minute * minutes_used)) :
  monthly_fee = 2 :=
by
  sorry

end find_monthly_fee_l296_296517


namespace meal_cost_with_tip_l296_296497

theorem meal_cost_with_tip 
  (cost_samosas : ℕ := 3 * 2)
  (cost_pakoras : ℕ := 4 * 3)
  (cost_lassi : ℕ := 2)
  (total_cost_before_tip := cost_samosas + cost_pakoras + cost_lassi)
  (tip : ℝ := 0.25 * total_cost_before_tip) :
  (total_cost_before_tip + tip = 25) :=
sorry

end meal_cost_with_tip_l296_296497


namespace sequence_divisible_by_11_l296_296213

theorem sequence_divisible_by_11 {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : a 2 = 3)
    (h_rec : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
    (a 4 % 11 = 0) ∧ (a 8 % 11 = 0) ∧ (a 10 % 11 = 0) ∧ (∀ n, n ≥ 11 → a n % 11 = 0) :=
by
  sorry

end sequence_divisible_by_11_l296_296213


namespace tan_five_pi_over_four_l296_296665

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l296_296665


namespace largest_reciprocal_l296_296745

-- Definitions for the given numbers
def a := 1/4
def b := 3/7
def c := 2
def d := 10
def e := 2023

-- Statement to prove the problem
theorem largest_reciprocal :
  (1/a) > (1/b) ∧ (1/a) > (1/c) ∧ (1/a) > (1/d) ∧ (1/a) > (1/e) :=
by
  sorry

end largest_reciprocal_l296_296745


namespace problem_statement_l296_296233

theorem problem_statement (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end problem_statement_l296_296233


namespace percentage_difference_l296_296181

theorem percentage_difference (w x y z : ℝ) (h1 : w = 0.6 * x) (h2 : x = 0.6 * y) (h3 : z = 0.54 * y) : 
  ((z - w) / w) * 100 = 50 :=
by
  sorry

end percentage_difference_l296_296181


namespace min_overlap_l296_296874

noncomputable def drinks_coffee := 0.60
noncomputable def drinks_tea := 0.50
noncomputable def drinks_neither := 0.10
noncomputable def drinks_either := 1 - drinks_neither
noncomputable def total_overlap := drinks_coffee + drinks_tea - drinks_either

theorem min_overlap (hcoffee : drinks_coffee = 0.60) (htea : drinks_tea = 0.50) (hneither : drinks_neither = 0.10) :
  total_overlap = 0.20 :=
by
  sorry

end min_overlap_l296_296874


namespace batsman_average_after_25th_innings_l296_296755

theorem batsman_average_after_25th_innings (A : ℝ) (h_pre_avg : (25 * (A + 3)) = (24 * A + 80))
  : A + 3 = 8 := 
by
  sorry

end batsman_average_after_25th_innings_l296_296755


namespace mission_total_days_l296_296021

theorem mission_total_days :
  let first_mission_planned := 5 : ℕ
  let first_mission_additional := 3 : ℕ  -- 60% of 5 days is 3 days
  let second_mission := 3 : ℕ
  let first_mission_total := first_mission_planned + first_mission_additional
  let total_mission_days := first_mission_total + second_mission
  total_mission_days = 11 :=
by
  sorry

end mission_total_days_l296_296021


namespace joan_books_correct_l296_296589

def sam_books : ℕ := 110
def total_books : ℕ := 212

def joan_books : ℕ := total_books - sam_books

theorem joan_books_correct : joan_books = 102 := by
  sorry

end joan_books_correct_l296_296589


namespace Peter_initially_had_33_marbles_l296_296721

-- Definitions based on conditions
def lostMarbles : Nat := 15
def currentMarbles : Nat := 18

-- Definition for the initial marbles calculation
def initialMarbles (lostMarbles : Nat) (currentMarbles : Nat) : Nat :=
  lostMarbles + currentMarbles

-- Theorem statement
theorem Peter_initially_had_33_marbles : initialMarbles lostMarbles currentMarbles = 33 := by
  sorry

end Peter_initially_had_33_marbles_l296_296721


namespace family_work_solution_l296_296736

noncomputable def family_work_problem : Prop :=
  ∃ (M W : ℕ),
    M + W = 15 ∧
    (M * (9/120) + W * (6/180) = 1) ∧
    W = 3

theorem family_work_solution : family_work_problem :=
by
  sorry

end family_work_solution_l296_296736


namespace sum_of_numerator_and_denominator_of_decimal_0_345_l296_296900

def repeating_decimal_to_fraction_sum (x : ℚ) : ℕ :=
if h : x = 115 / 333 then 115 + 333 else 0

theorem sum_of_numerator_and_denominator_of_decimal_0_345 :
  repeating_decimal_to_fraction_sum 345 / 999 = 448 :=
by {
  -- Given: 0.\overline{345} = 345 / 999, simplified to 115 / 333
  -- hence the sum of numerator and denominator = 115 + 333
  -- We don't need the proof steps here, just conclude with the sum
  sorry }

end sum_of_numerator_and_denominator_of_decimal_0_345_l296_296900


namespace sufficient_but_not_necessary_condition_l296_296229

variable {x k : ℝ}

def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

theorem sufficient_but_not_necessary_condition (h_suff : ∀ x, p x k → q x) (h_not_necessary : ∃ x, q x ∧ ¬p x k) : k > 2 :=
sorry

end sufficient_but_not_necessary_condition_l296_296229


namespace vasya_tolya_badges_l296_296175

-- Let V be the number of badges Vasya had before the exchange.
-- Let T be the number of badges Tolya had before the exchange.
theorem vasya_tolya_badges (V T : ℕ) 
  (h1 : V = T + 5)
  (h2 : 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1) :
  V = 50 ∧ T = 45 :=
by 
  sorry

end vasya_tolya_badges_l296_296175


namespace longer_segment_of_triangle_l296_296075

theorem longer_segment_of_triangle {a b c : ℝ} (h_triangle : a = 40 ∧ b = 90 ∧ c = 100) (h_altitude : ∃ h, h > 0) : 
  ∃ (longer_segment : ℝ), longer_segment = 82.5 :=
by 
  sorry

end longer_segment_of_triangle_l296_296075


namespace distinct_integers_no_perfect_square_product_l296_296344

theorem distinct_integers_no_perfect_square_product
  (k : ℕ) (hk : 0 < k) :
  ∀ a b : ℕ, k^2 < a ∧ a < (k+1)^2 → k^2 < b ∧ b < (k+1)^2 → a ≠ b → ¬∃ m : ℕ, a * b = m^2 :=
by sorry

end distinct_integers_no_perfect_square_product_l296_296344


namespace number_of_rows_l296_296779

-- Definitions of the conditions
def total_students : ℕ := 23
def students_in_restroom : ℕ := 2
def students_absent : ℕ := 3 * students_in_restroom - 1
def students_per_desk : ℕ := 6
def fraction_full (r : ℕ) := (2 * r) / 3

-- The statement we need to prove 
theorem number_of_rows : (total_students - students_in_restroom - students_absent) / (students_per_desk * 2 / 3) = 4 :=
by
  sorry

end number_of_rows_l296_296779


namespace find_t_l296_296219

theorem find_t (t : ℝ) : (∃ y : ℝ, y = -(t - 1) ∧ 2 * y - 4 = 3 * (y - 2)) ↔ t = -1 :=
by sorry

end find_t_l296_296219


namespace find_a_l296_296536

theorem find_a (a : ℝ) (h1 : ∀ (x y : ℝ), ax + 2*y - 2 = 0 → (x + y) = 0)
  (h2 : ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 6 → (∃ A B : ℝ × ℝ, A ≠ B ∧ (A = (x, y) ∧ B = (-x, -y))))
  : a = -2 := 
sorry

end find_a_l296_296536


namespace factor_expression_l296_296379

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) :=
by
  sorry

end factor_expression_l296_296379


namespace hilda_loan_compounding_difference_l296_296966

noncomputable def difference_due_to_compounding (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let A_monthly := P * (1 + r / 12)^(12 * t)
  let A_annually := P * (1 + r)^t
  A_monthly - A_annually

theorem hilda_loan_compounding_difference :
  difference_due_to_compounding 8000 0.10 5 = 376.04 :=
sorry

end hilda_loan_compounding_difference_l296_296966


namespace avg_weight_of_children_is_138_l296_296035

-- Define the average weight of boys and girls
def average_weight_of_boys := 150
def number_of_boys := 6
def average_weight_of_girls := 120
def number_of_girls := 4

-- Calculate total weights and average weight of all children
noncomputable def total_weight_of_boys := number_of_boys * average_weight_of_boys
noncomputable def total_weight_of_girls := number_of_girls * average_weight_of_girls
noncomputable def total_weight_of_children := total_weight_of_boys + total_weight_of_girls
noncomputable def number_of_children := number_of_boys + number_of_girls
noncomputable def average_weight_of_children := total_weight_of_children / number_of_children

-- Lean statement to prove the average weight of all children is 138 pounds
theorem avg_weight_of_children_is_138 : average_weight_of_children = 138 := by
    sorry

end avg_weight_of_children_is_138_l296_296035


namespace cricket_overs_played_initially_l296_296984

variables (x y : ℝ)

theorem cricket_overs_played_initially 
  (h1 : y = 3.2 * x)
  (h2 : 262 - y = 5.75 * 40) : 
  x = 10 := 
sorry

end cricket_overs_played_initially_l296_296984


namespace mean_equals_sum_of_squares_l296_296452

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_equals_sum_of_squares (x y z : ℝ) (h1 : arithmetic_mean x y z = 10)
  (h2 : geometric_mean x y z = 6) (h3 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 :=
  sorry

end mean_equals_sum_of_squares_l296_296452


namespace michael_initial_money_l296_296027

theorem michael_initial_money 
  (M B_initial B_left B_spent : ℕ) 
  (h_split : M / 2 = B_initial - B_left + B_spent): 
  (M / 2 + B_left = 17 + 35) → M = 152 :=
by
  sorry

end michael_initial_money_l296_296027


namespace normal_dist_probability_l296_296218

variable {σ : ℝ} (X : ℝ → ℝ)

theorem normal_dist_probability
  (h1 : ∀ x, X x ∼ Normal 2 (σ^2))
  (h2 : P(X ≤ 4) = 0.84) :
  P(X < 0) = 0.16 :=
by
  sorry  -- Proof outline: P(X < 0) = P(X > 4), and given P(X ≤ 4) = 0.84, thus P(X > 4) = 1 - 0.84 = 0.16.

end normal_dist_probability_l296_296218


namespace divide_payment_correctly_l296_296742

-- Define the number of logs contributed by each person
def logs_troikin : ℕ := 3
def logs_pyaterkin : ℕ := 5
def logs_bestoplivny : ℕ := 0

-- Define the total number of logs
def total_logs : ℕ := logs_troikin + logs_pyaterkin + logs_bestoplivny

-- Define the total number of logs used equally
def logs_per_person : ℚ := total_logs / 3

-- Define the total payment made by Bestoplivny 
def total_payment : ℕ := 80

-- Define the cost per log
def cost_per_log : ℚ := total_payment / logs_per_person

-- Define the contribution of each person to Bestoplivny
def bestoplivny_from_troikin : ℚ := logs_troikin - logs_per_person
def bestoplivny_from_pyaterkin : ℚ := logs_pyaterkin - (logs_per_person - bestoplivny_from_troikin)

-- Define the kopecks received by Troikina and Pyaterkin
def kopecks_troikin : ℚ := bestoplivny_from_troikin * cost_per_log
def kopecks_pyaterkin : ℚ := bestoplivny_from_pyaterkin * cost_per_log

-- Main theorem to prove the correct division of kopecks
theorem divide_payment_correctly : kopecks_troikin = 10 ∧ kopecks_pyaterkin = 70 :=
by
  -- ... Proof goes here
  sorry

end divide_payment_correctly_l296_296742


namespace divide_by_repeating_decimal_l296_296288

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l296_296288


namespace range_of_t_l296_296955

theorem range_of_t (a b c : ℝ) (t : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_inequality : ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → (1 / a^2) + (4 / b^2) + (t / c^2) ≥ 0) :
  t ≥ -9 :=
sorry

end range_of_t_l296_296955


namespace mila_hours_to_match_agnes_monthly_earnings_l296_296461

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end mila_hours_to_match_agnes_monthly_earnings_l296_296461


namespace probability_snow_at_least_once_l296_296439

-- Defining the probability of no snow on the first five days
def no_snow_first_five_days : ℚ := (4 / 5) ^ 5

-- Defining the probability of no snow on the next five days
def no_snow_next_five_days : ℚ := (2 / 3) ^ 5

-- Total probability of no snow during the first ten days
def no_snow_first_ten_days : ℚ := no_snow_first_five_days * no_snow_next_five_days

-- Probability of snow at least once during the first ten days
def snow_at_least_once_first_ten_days : ℚ := 1 - no_snow_first_ten_days

-- Desired proof statement
theorem probability_snow_at_least_once :
  snow_at_least_once_first_ten_days = 726607 / 759375 := by
  sorry

end probability_snow_at_least_once_l296_296439


namespace trip_distance_first_part_l296_296349

theorem trip_distance_first_part (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 70) (h3 : 32 = 70 / ((x / 48) + ((70 - x) / 24))) : x = 35 :=
by
  sorry

end trip_distance_first_part_l296_296349


namespace positive_inequality_l296_296808

open Real

/-- Given positive real numbers x, y, z such that xyz ≥ 1, prove that
    (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0.
-/
theorem positive_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + 
  (y^5 - y^2) / (y^5 + x^2 + z^2) + 
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end positive_inequality_l296_296808


namespace probability_initials_start_with_BCD_l296_296014

theorem probability_initials_start_with_BCD : 
  ∀ (total_students : ℕ) (unique_initials : Finset (Char × Char)), 
  total_students = 30 → 
  (∀ p1 p2 ∈ unique_initials, p1 ≠ p2) →
  (∀ initials, initials ∈ unique_initials → initials.1 ∈ {'B', 'C', 'D'} ∪ consonants) →
  let probability := (unique_initials.filter (λ initials, initials.1 ∈ {'B', 'C', 'D'})).card / (21 * total_students : ℚ) in
  probability = 1 / 21 := 
by sorry

end probability_initials_start_with_BCD_l296_296014


namespace max_cookies_without_ingredients_l296_296140

-- Defining the number of cookies and their composition
def total_cookies : ℕ := 36
def peanuts : ℕ := (2 * total_cookies) / 3
def chocolate_chips : ℕ := total_cookies / 3
def raisins : ℕ := total_cookies / 4
def oats : ℕ := total_cookies / 8

-- Proving the largest number of cookies without any ingredients
theorem max_cookies_without_ingredients : (total_cookies - (max (max peanuts chocolate_chips) raisins)) = 12 := by
    sorry

end max_cookies_without_ingredients_l296_296140


namespace image_of_center_after_transform_l296_296784

structure Point where
  x : ℤ
  y : ℤ

def reflect_across_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def translate_right (p : Point) (units : ℤ) : Point :=
  { x := p.x + units, y := p.y }

def transform_point (p : Point) : Point :=
  translate_right (reflect_across_x p) 5

theorem image_of_center_after_transform :
  transform_point {x := -3, y := 4} = {x := 2, y := -4} := by
  sorry

end image_of_center_after_transform_l296_296784


namespace find_third_number_in_proportion_l296_296693

theorem find_third_number_in_proportion (x : ℝ) (third_number : ℝ) (h1 : x = 0.9) (h2 : 0.75 / 6 = x / third_number) : third_number = 5 := by
  sorry

end find_third_number_in_proportion_l296_296693


namespace solution_set_abs_inequality_l296_296887

theorem solution_set_abs_inequality (x : ℝ) : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end solution_set_abs_inequality_l296_296887


namespace tan_five_pi_over_four_l296_296654

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l296_296654


namespace sum_of_numerator_and_denominator_of_decimal_0_345_l296_296899

def repeating_decimal_to_fraction_sum (x : ℚ) : ℕ :=
if h : x = 115 / 333 then 115 + 333 else 0

theorem sum_of_numerator_and_denominator_of_decimal_0_345 :
  repeating_decimal_to_fraction_sum 345 / 999 = 448 :=
by {
  -- Given: 0.\overline{345} = 345 / 999, simplified to 115 / 333
  -- hence the sum of numerator and denominator = 115 + 333
  -- We don't need the proof steps here, just conclude with the sum
  sorry }

end sum_of_numerator_and_denominator_of_decimal_0_345_l296_296899


namespace largest_band_members_l296_296193

theorem largest_band_members :
  ∃ (r x : ℕ), r * x + 3 = 107 ∧ (r - 3) * (x + 2) = 107 ∧ r * x < 147 :=
sorry

end largest_band_members_l296_296193


namespace complement_of_M_in_U_l296_296227

theorem complement_of_M_in_U :
  let U := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
  let M := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
  ∀ x, x ∉ M → x ∈ U → x ∈ {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  let U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
  let M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
  have h : ∀ x, x ∉ M → x ∈ U → x ∈ {x | 1 < x ∧ x ≤ 3}
  { sorry }
  exact h

end complement_of_M_in_U_l296_296227


namespace log_properties_l296_296933

theorem log_properties:
  log(10) = 1 → (log(5) ^ 2 + log(2) * log(50) = 1) :=
begin
  sorry
end

end log_properties_l296_296933


namespace division_of_repeating_decimal_l296_296322

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l296_296322


namespace sin_subtract_of_obtuse_angle_l296_296401

open Real -- Open the Real namespace for convenience.

theorem sin_subtract_of_obtuse_angle (α : ℝ) 
  (h1 : (π / 2) < α) (h2 : α < π)
  (h3 : sin (π / 4 + α) = 3 / 4)
  : sin (π / 4 - α) = - (sqrt 7) / 4 := 
by 
  sorry -- Proof placeholder.

end sin_subtract_of_obtuse_angle_l296_296401


namespace general_term_formula_sum_formula_and_max_value_l296_296538

-- Definitions for the conditions
def tenth_term : ℕ → ℤ := λ n => 24
def twenty_fifth_term : ℕ → ℤ := λ n => -21

-- Prove the general term formula
theorem general_term_formula (a : ℕ → ℤ) (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) :
  ∀ n : ℕ, a n = -3 * n + 54 := sorry

-- Prove the sum formula and its maximum value
theorem sum_formula_and_max_value (a : ℕ → ℤ) (S : ℕ → ℤ)
  (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) 
  (sum_formula : ∀ n : ℕ, S n = -3 * n^2 / 2 + 51 * n) :
  ∃ max_n : ℕ, S max_n = 578 := sorry

end general_term_formula_sum_formula_and_max_value_l296_296538


namespace calculateDifferentialSavings_l296_296772

/-- 
Assumptions for the tax brackets and deductions/credits.
-/
def taxBracketsCurrent (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 15 / 100
  else if income ≤ 45000 then
    15000 * 15 / 100 + (income - 15000) * 42 / 100
  else
    15000 * 15 / 100 + (45000 - 15000) * 42 / 100 + (income - 45000) * 50 / 100

def taxBracketsProposed (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 12 / 100
  else if income ≤ 45000 then
    15000 * 12 / 100 + (income - 15000) * 28 / 100
  else
    15000 * 12 / 100 + (45000 - 15000) * 28 / 100 + (income - 45000) * 50 / 100

def standardDeduction : ℕ := 3000
def childrenCredit (num_children : ℕ) : ℕ := num_children * 1000

def taxableIncome (income : ℕ) : ℕ :=
  income - standardDeduction

def totalTaxLiabilityCurrent (income num_children : ℕ) : ℕ :=
  (taxBracketsCurrent (taxableIncome income)) - (childrenCredit num_children)

def totalTaxLiabilityProposed (income num_children : ℕ) : ℕ :=
  (taxBracketsProposed (taxableIncome income)) - (childrenCredit num_children)

def differentialSavings (income num_children : ℕ) : ℕ :=
  totalTaxLiabilityCurrent income num_children - totalTaxLiabilityProposed income num_children

/-- 
Statement of the Lean 4 proof problem.
-/
theorem calculateDifferentialSavings : differentialSavings 34500 2 = 2760 :=
by
  sorry

end calculateDifferentialSavings_l296_296772


namespace intersection_M_N_l296_296001

noncomputable def set_M : Set ℚ := {α | ∃ k : ℤ, α = k * 90 - 36}
noncomputable def set_N : Set ℚ := {α | -180 < α ∧ α < 180}

theorem intersection_M_N : set_M ∩ set_N = {-36, 54, 144, -126} := by
  sorry

end intersection_M_N_l296_296001


namespace ab_value_l296_296061

theorem ab_value (a b : ℝ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * (a * b) = 800 :=
by sorry

end ab_value_l296_296061


namespace find_angle_D_l296_296421

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 40) (h4 : B + C = 130) : D = 40 := by
  sorry

end find_angle_D_l296_296421


namespace min_n_A0_An_ge_200_l296_296848

theorem min_n_A0_An_ge_200 :
  (∃ n : ℕ, (n * (n + 1)) / 3 ≥ 200) ∧
  (∀ m < 24, (m * (m + 1)) / 3 < 200) :=
sorry

end min_n_A0_An_ge_200_l296_296848


namespace intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l296_296568

-- Definitions for U, A, B
def U := { x : ℤ | 0 < x ∧ x <= 10 }
def A : Set ℤ := { 1, 2, 4, 5, 9 }
def B : Set ℤ := { 4, 6, 7, 8, 10 }

-- 1. Prove A ∩ B = {4}
theorem intersection_eq : A ∩ B = {4} := by
  sorry

-- 2. Prove A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}
theorem union_eq : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

-- 3. Prove complement_U (A ∪ B) = {3}
def complement_U (s : Set ℤ) : Set ℤ := { x ∈ U | ¬ (x ∈ s) }
theorem complement_union_eq : complement_U (A ∪ B) = {3} := by
  sorry

-- 4. Prove (complement_U A) ∩ (complement_U B) = {3}
theorem intersection_complements_eq : (complement_U A) ∩ (complement_U B) = {3} := by
  sorry

end intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l296_296568


namespace meal_cost_is_25_l296_296500

def total_cost_samosas : ℕ := 3 * 2
def total_cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2
def tip_percentage : ℝ := 0.25

def total_food_cost : ℕ := total_cost_samosas + total_cost_pakoras + cost_mango_lassi
def tip_amount : ℝ := total_food_cost * tip_percentage
def total_meal_cost : ℝ := total_food_cost + tip_amount

theorem meal_cost_is_25 : total_meal_cost = 25 := by
    sorry

end meal_cost_is_25_l296_296500


namespace steven_more_peaches_than_apples_l296_296988

def steven_peaches : Nat := 17
def steven_apples : Nat := 16

theorem steven_more_peaches_than_apples : steven_peaches - steven_apples = 1 := by
  sorry

end steven_more_peaches_than_apples_l296_296988


namespace sin_cos_identity_l296_296261

theorem sin_cos_identity (α β γ : ℝ) (h : α + β + γ = 180) :
    Real.sin α + Real.sin β + Real.sin γ = 
    4 * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) := 
  sorry

end sin_cos_identity_l296_296261


namespace card_giving_ratio_l296_296565

theorem card_giving_ratio (initial_cards cards_to_Bob cards_left : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : cards_to_Bob = 3)
  (h3 : cards_left = 9) : 
  (initial_cards - cards_left - cards_to_Bob) / gcd (initial_cards - cards_left - cards_to_Bob) cards_to_Bob = 2 / 1 :=
by sorry

end card_giving_ratio_l296_296565


namespace total_number_of_trees_l296_296159

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l296_296159


namespace percentage_discount_total_amount_paid_l296_296267

variable (P Q : ℝ)

theorem percentage_discount (h₁ : P > Q) (h₂ : Q > 0) :
  100 * ((P - Q) / P) = 100 * (P - Q) / P :=
sorry

theorem total_amount_paid (h₁ : P > Q) (h₂ : Q > 0) :
  10 * Q = 10 * Q :=
sorry

end percentage_discount_total_amount_paid_l296_296267


namespace distinct_banners_l296_296638

inductive Color
| red
| white
| blue
| green
| yellow

def adjacent_different (a b : Color) : Prop := a ≠ b

theorem distinct_banners : 
  ∃ n : ℕ, n = 320 ∧ ∀ strips : Fin 4 → Color, 
    adjacent_different (strips 0) (strips 1) ∧ 
    adjacent_different (strips 1) (strips 2) ∧ 
    adjacent_different (strips 2) (strips 3) :=
sorry

end distinct_banners_l296_296638


namespace positive_integers_condition_l296_296390

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end positive_integers_condition_l296_296390


namespace find_n_from_A_C_l296_296673

noncomputable def A_n (n : ℕ) : ℕ := n! / (n - 2)!
noncomputable def C_n (n : ℕ) (k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem find_n_from_A_C (n : ℕ) (h : (A_n n)^2 = C_n n (n - 3)) : n = 8 := by
  sorry

end find_n_from_A_C_l296_296673


namespace base4_more_digits_than_base9_l296_296969

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l296_296969


namespace c_geq_one_l296_296856

theorem c_geq_one (a b : ℕ) (c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : c ≥ 1 :=
by sorry

end c_geq_one_l296_296856


namespace sum_is_correct_l296_296993

theorem sum_is_correct (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := 
by 
  sorry

end sum_is_correct_l296_296993


namespace infinite_series_eq_1_div_400_l296_296084

theorem infinite_series_eq_1_div_400 :
  (∑' n:ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 :=
by
  sorry

end infinite_series_eq_1_div_400_l296_296084


namespace gcd_128_144_256_l296_296177

theorem gcd_128_144_256 : Nat.gcd (Nat.gcd 128 144) 256 = 128 :=
  sorry

end gcd_128_144_256_l296_296177


namespace march_1_falls_on_friday_l296_296985

-- Definitions of conditions
def march_days : ℕ := 31
def mondays_in_march : ℕ := 4
def thursdays_in_march : ℕ := 4

-- Lean 4 statement to prove March 1 falls on a Friday
theorem march_1_falls_on_friday 
  (h1 : march_days = 31)
  (h2 : mondays_in_march = 4)
  (h3 : thursdays_in_march = 4)
  : ∃ d : ℕ, d = 5 :=
by sorry

end march_1_falls_on_friday_l296_296985


namespace evaluate_number_l296_296791

theorem evaluate_number (n : ℝ) (h : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24) : n = 1 / 3 :=
by
  sorry

end evaluate_number_l296_296791


namespace limes_left_after_giving_l296_296507

theorem limes_left_after_giving : 
  ∀ (dan_original_limes : ℕ) (limes_given_to_sara : ℕ), 
  dan_original_limes = 9 → limes_given_to_sara = 4 → dan_original_limes - limes_given_to_sara = 5 :=
by 
  intros dan_original_limes limes_given_to_sara H1 H2
  rw [H1, H2]
  exact rfl

end limes_left_after_giving_l296_296507


namespace overall_effect_l296_296743
noncomputable def effect (x : ℚ) : ℚ :=
  ((x * (5 / 6)) * (1 / 10)) + (2 / 3)

theorem overall_effect (x : ℚ) : effect x = (x * (5 / 6) * (1 / 10)) + (2 / 3) :=
  by
  sorry

-- Prove for initial number 1
example : effect 1 = 3 / 4 :=
  by
  sorry

end overall_effect_l296_296743


namespace sufficient_not_necessary_implies_a_lt_1_l296_296005

theorem sufficient_not_necessary_implies_a_lt_1 {x a : ℝ} (h : ∀ x : ℝ, x > 1 → x > a ∧ ¬(x > a → x > 1)) : a < 1 :=
sorry

end sufficient_not_necessary_implies_a_lt_1_l296_296005


namespace intersection_A_B_l296_296002

def set_A (x : ℝ) : Prop := (x + 1 / 2 ≥ 3 / 2) ∨ (x + 1 / 2 ≤ -3 / 2)
def set_B (x : ℝ) : Prop := x^2 + x < 6
def A_cap_B := { x : ℝ | set_A x ∧ set_B x }

theorem intersection_A_B : A_cap_B = { x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2) } :=
sorry

end intersection_A_B_l296_296002


namespace even_function_implies_a_zero_l296_296234

theorem even_function_implies_a_zero (a : ℝ) : 
  (∀ x : ℝ, (λ x, x^2 - |x + a|) (-x) = (λ x, x^2 - |x + a|) (x)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l296_296234


namespace negation_universal_proposition_l296_296884

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x := 
by 
  sorry

end negation_universal_proposition_l296_296884


namespace trig_signs_l296_296044

-- The conditions formulated as hypotheses
theorem trig_signs (h1 : Real.pi / 2 < 2 ∧ 2 < 3 ∧ 3 < Real.pi ∧ Real.pi < 4 ∧ 4 < 3 * Real.pi / 2) : 
  Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := 
sorry

end trig_signs_l296_296044


namespace grapes_average_seeds_l296_296153

def total_seeds_needed : ℕ := 60
def apple_seed_average : ℕ := 6
def pear_seed_average : ℕ := 2
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def extra_seeds_needed : ℕ := 3

-- Calculation of total seeds from apples and pears:
def seeds_from_apples : ℕ := apples_count * apple_seed_average
def seeds_from_pears : ℕ := pears_count * pear_seed_average

def total_seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculation of the remaining seeds needed from grapes:
def seeds_needed_from_grapes : ℕ := total_seeds_needed - total_seeds_from_apples_and_pears - extra_seeds_needed

-- Calculation of the average number of seeds per grape:
def grape_seed_average : ℕ := seeds_needed_from_grapes / grapes_count

-- Prove the correct average number of seeds per grape:
theorem grapes_average_seeds : grape_seed_average = 3 :=
by
  sorry

end grapes_average_seeds_l296_296153


namespace cars_in_group_l296_296986

open Nat

theorem cars_in_group (C : ℕ) : 
  (47 ≤ C) →                  -- At least 47 cars in the group
  (53 ≤ C) →                  -- At least 53 cars in the group
  C ≥ 100 :=                  -- Conclusion: total cars is at least 100
by
  -- Begin the proof
  sorry                       -- Skip proof for now

end cars_in_group_l296_296986


namespace Joan_initial_money_l296_296132

def cost_hummus (containers : ℕ) (price_per_container : ℕ) : ℕ := containers * price_per_container
def cost_apple (quantity : ℕ) (price_per_apple : ℕ) : ℕ := quantity * price_per_apple

theorem Joan_initial_money 
  (containers_of_hummus : ℕ)
  (price_per_hummus : ℕ)
  (cost_chicken : ℕ)
  (cost_bacon : ℕ)
  (cost_vegetables : ℕ)
  (quantity_apple : ℕ)
  (price_per_apple : ℕ)
  (total_cost : ℕ)
  (remaining_money : ℕ):
  containers_of_hummus = 2 →
  price_per_hummus = 5 →
  cost_chicken = 20 →
  cost_bacon = 10 →
  cost_vegetables = 10 →
  quantity_apple = 5 →
  price_per_apple = 2 →
  remaining_money = cost_apple quantity_apple price_per_apple →
  total_cost = cost_hummus containers_of_hummus price_per_hummus + cost_chicken + cost_bacon + cost_vegetables + remaining_money →
  total_cost = 60 :=
by
  intros
  sorry

end Joan_initial_money_l296_296132


namespace total_cost_l296_296134

theorem total_cost 
  (rental_cost: ℝ) 
  (gallons: ℝ) 
  (gas_price: ℝ) 
  (price_per_mile: ℝ) 
  (miles: ℝ)
  (H1: rental_cost = 150)
  (H2: gallons = 8)
  (H3: gas_price = 3.5)
  (H4: price_per_mile = 0.5)
  (H5: miles = 320) :
  rental_cost + (gallons * gas_price) + (miles * price_per_mile) = 338 :=
by {
  have gas_cost : ℝ := gallons * gas_price,
  have mileage_cost : ℝ := miles * price_per_mile,
  rw [←H1, ←H2, ←H3, ←H4, ←H5],
  norm_num,
  sorry
}

end total_cost_l296_296134


namespace remaining_surface_area_correct_l296_296377

open Real

-- Define the original cube and the corner cubes
def orig_cube : ℝ × ℝ × ℝ := (5, 5, 5)
def corner_cube : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define a function to compute the surface area of a cube given dimensions (a, b, c)
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)

-- Original surface area of the cube
def orig_surface_area : ℝ := surface_area 5 5 5

-- Total surface area of the remaining figure after removing 8 corner cubes
def remaining_surface_area : ℝ := 150  -- Calculated directly as 6 * 25

-- Theorem stating that the surface area of the remaining figure is 150 cm^2
theorem remaining_surface_area_correct :
  remaining_surface_area = 150 := sorry

end remaining_surface_area_correct_l296_296377


namespace quadratic_radical_condition_l296_296050

variable (x : ℝ)

theorem quadratic_radical_condition : 
  (∃ (r : ℝ), r = x^2 + 1 ∧ r ≥ 0) ↔ (True) := by
  sorry

end quadratic_radical_condition_l296_296050


namespace find_c_plus_1_over_b_l296_296723

theorem find_c_plus_1_over_b (a b c : ℝ) (h1: a * b * c = 1) 
    (h2: a + 1 / c = 7) (h3: b + 1 / a = 12) : c + 1 / b = 21 / 83 := 
by 
    sorry

end find_c_plus_1_over_b_l296_296723


namespace dice_sum_eight_dice_l296_296469

/--
  Given 8 fair 6-sided dice, prove that the number of ways to obtain
  a sum of 11 on the top faces of these dice, is 120.
-/
theorem dice_sum_eight_dice :
  (∃ n : ℕ, ∀ (dices : List ℕ), (dices.length = 8 ∧ (∀ d ∈ dices, 1 ≤ d ∧ d ≤ 6) 
   ∧ dices.sum = 11) → n = 120) :=
sorry

end dice_sum_eight_dice_l296_296469


namespace karen_packs_cookies_l296_296843

-- Conditions stated as definitions
def school_days := 5
def peanut_butter_days := 2
def ham_sandwich_days := school_days - peanut_butter_days
def cake_days := 1
def probability_ham_and_cake := 0.12

-- Lean theorem statement
theorem karen_packs_cookies : 
  (school_days - cake_days - peanut_butter_days) = 2 :=
by
  sorry

end karen_packs_cookies_l296_296843


namespace smallest_k_value_for_screws_packs_l296_296152

theorem smallest_k_value_for_screws_packs :
  ∃ k : ℕ, k = 60 ∧ (∃ x y : ℕ, (k = 10 * x ∧ k = 12 * y) ∧ x ≠ y) := sorry

end smallest_k_value_for_screws_packs_l296_296152


namespace championship_outcomes_l296_296737

theorem championship_outcomes (students events : ℕ) (h_students : students = 3) (h_events : events = 2) : 
  students ^ events = 9 :=
by
  rw [h_students, h_events]
  have h : 3 ^ 2 = 9 := by norm_num
  exact h

end championship_outcomes_l296_296737


namespace tan_double_angle_l296_296100

theorem tan_double_angle (θ : ℝ) 
  (h1 : Real.sin θ = 4 / 5) 
  (h2 : Real.sin θ - Real.cos θ > 1) : 
  Real.tan (2 * θ) = 24 / 7 := 
sorry

end tan_double_angle_l296_296100


namespace math_problem_l296_296570

theorem math_problem
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) :=
  sorry

end math_problem_l296_296570


namespace middle_number_consecutive_odd_sum_l296_296478

theorem middle_number_consecutive_odd_sum (n : ℤ)
  (h1 : n % 2 = 1) -- n is an odd number
  (h2 : n + (n + 2) + (n + 4) = n + 20) : 
  n + 2 = 9 :=
by
  sorry

end middle_number_consecutive_odd_sum_l296_296478


namespace jill_spent_more_l296_296992

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end jill_spent_more_l296_296992


namespace bag_weight_l296_296210

theorem bag_weight (W : ℕ) 
  (h1 : 2 * W + 82 * (2 * W) = 664) : 
  W = 4 := by
  sorry

end bag_weight_l296_296210


namespace area_of_rectangle_l296_296198

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

end area_of_rectangle_l296_296198


namespace correct_transformation_D_l296_296746

theorem correct_transformation_D : ∀ x, 2 * (x + 1) = x + 7 → x = 5 :=
by
  intro x
  sorry

end correct_transformation_D_l296_296746


namespace amount_of_cocoa_powder_given_by_mayor_l296_296043

def total_cocoa_powder_needed : ℕ := 306
def cocoa_powder_still_needed : ℕ := 47

def cocoa_powder_given_by_mayor : ℕ :=
  total_cocoa_powder_needed - cocoa_powder_still_needed

theorem amount_of_cocoa_powder_given_by_mayor :
  cocoa_powder_given_by_mayor = 259 := by
  sorry

end amount_of_cocoa_powder_given_by_mayor_l296_296043


namespace petals_in_garden_l296_296942

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end petals_in_garden_l296_296942


namespace tetrahedron_volume_l296_296357

theorem tetrahedron_volume (a b c : ℝ)
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  ∃ V : ℝ, 
    V = (1 / (6 * Real.sqrt 2)) * 
        Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
sorry

end tetrahedron_volume_l296_296357


namespace surface_area_change_l296_296490

noncomputable def original_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

noncomputable def new_surface_area (l w h c : ℝ) : ℝ :=
  original_surface_area l w h - 
  (3 * (c * c)) + 
  (2 * c * c)

theorem surface_area_change (l w h c : ℝ) (hl : l = 5) (hw : w = 4) (hh : h = 3) (hc : c = 2) :
  new_surface_area l w h c = original_surface_area l w h - 8 :=
by 
  sorry

end surface_area_change_l296_296490


namespace hyperbola_asymptotes_and_point_l296_296681

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_asymptotes_and_point 
  (x y : ℝ)
  (asymptote1 : ∀ x, y = (1/2) * x)
  (asymptote2 : ∀ x, y = (-1/2) * x)
  (point : (x, y) = (4, Real.sqrt 2))
: hyperbola_equation x y :=
sorry

end hyperbola_asymptotes_and_point_l296_296681


namespace water_segment_length_l296_296905

theorem water_segment_length 
  (total_distance : ℝ)
  (find_probability : ℝ)
  (lose_probability : ℝ)
  (probability_equation : total_distance * lose_probability = 750) :
  total_distance = 2500 → 
  find_probability = 7 / 10 →
  lose_probability = 3 / 10 →
  x = 750 :=
by
  intros h1 h2 h3
  sorry

end water_segment_length_l296_296905


namespace sum_of_ais_l296_296857

theorem sum_of_ais :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℕ), 
    (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (a4 > 0) ∧ (a5 > 0) ∧ (a6 > 0) ∧ (a7 > 0) ∧ (a8 > 0) ∧
    a1^2 + (2*a2)^2 + (3*a3)^2 + (4*a4)^2 + (5*a5)^2 + (6*a6)^2 + (7*a7)^2 + (8*a8)^2 = 204 ∧
    a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 :=
by
  sorry

end sum_of_ais_l296_296857


namespace divide_by_repeating_decimal_l296_296289

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l296_296289


namespace tan_sin_difference_l296_296777

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end tan_sin_difference_l296_296777


namespace w_janous_conjecture_l296_296023

theorem w_janous_conjecture (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 :=
by
  sorry

end w_janous_conjecture_l296_296023


namespace inequality_proof_l296_296343

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end inequality_proof_l296_296343


namespace julia_mile_time_l296_296258

variable (x : ℝ)

theorem julia_mile_time
  (h1 : ∀ x, x > 0)
  (h2 : ∀ x, x <= 13)
  (h3 : 65 = 5 * 13)
  (h4 : 50 = 65 - 15)
  (h5 : 50 = 5 * x) :
  x = 10 := by
  sorry

end julia_mile_time_l296_296258


namespace hcf_462_5_1_l296_296878

theorem hcf_462_5_1 (a b c : ℕ) (h₁ : a = 462) (h₂ : b = 5) (h₃ : c = 2310) (h₄ : Nat.lcm a b = c) : Nat.gcd a b = 1 := by
  sorry

end hcf_462_5_1_l296_296878


namespace remy_sold_110_bottles_l296_296442

theorem remy_sold_110_bottles 
    (price_per_bottle : ℝ)
    (total_evening_sales : ℝ)
    (evening_more_than_morning : ℝ)
    (nick_fewer_than_remy : ℝ)
    (R : ℝ) 
    (total_morning_sales_is : ℝ) :
    price_per_bottle = 0.5 →
    total_evening_sales = 55 →
    evening_more_than_morning = 3 →
    nick_fewer_than_remy = 6 →
    total_morning_sales_is = total_evening_sales - evening_more_than_morning →
    (R * price_per_bottle) + ((R - nick_fewer_than_remy) * price_per_bottle) = total_morning_sales_is →
    R = 110 :=
by
  intros
  sorry

end remy_sold_110_bottles_l296_296442


namespace pyramid_volume_correct_l296_296192

noncomputable def PyramidVolume (base_area : ℝ) (triangle_area_1 : ℝ) (triangle_area_2 : ℝ) : ℝ :=
  let side := Real.sqrt base_area
  let height_1 := (2 * triangle_area_1) / side
  let height_2 := (2 * triangle_area_2) / side
  let h_sq := height_1 ^ 2 - (Real.sqrt (height_1 ^ 2 + height_2 ^ 2 - 512)) ^ 2
  let height := Real.sqrt h_sq
  (1/3) * base_area * height

theorem pyramid_volume_correct :
  PyramidVolume 256 120 112 = 1163 := by
  sorry

end pyramid_volume_correct_l296_296192


namespace min_value_expression_l296_296855

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    4.5 ≤ (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) :=
by
  sorry

end min_value_expression_l296_296855


namespace calculate_f_at_5_l296_296253

noncomputable def g (y : ℝ) : ℝ := (1 / 2) * y^2

noncomputable def f (x y : ℝ) : ℝ := 2 * x^2 + g y

theorem calculate_f_at_5 (y : ℝ) (h1 : f 2 y = 50) (h2 : y = 2*Real.sqrt 21) :
  f 5 y = 92 :=
by
  sorry

end calculate_f_at_5_l296_296253


namespace gcd_18222_24546_66364_eq_2_l296_296201

/-- Definition of three integers a, b, c --/
def a : ℕ := 18222 
def b : ℕ := 24546
def c : ℕ := 66364

/-- Proof of the gcd of the three integers being 2 --/
theorem gcd_18222_24546_66364_eq_2 : Nat.gcd (Nat.gcd a b) c = 2 := by
  sorry

end gcd_18222_24546_66364_eq_2_l296_296201


namespace differentiable_additive_zero_derivative_l296_296249

theorem differentiable_additive_zero_derivative {f : ℝ → ℝ}
  (h1 : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_diff : Differentiable ℝ f) : 
  deriv f 0 = 0 :=
sorry

end differentiable_additive_zero_derivative_l296_296249


namespace shortest_chord_intercepted_by_line_l296_296418

theorem shortest_chord_intercepted_by_line (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 3 = 0 → y = k*x + 1 → (x - y + 1 = 0)) :=
sorry

end shortest_chord_intercepted_by_line_l296_296418


namespace buses_fewer_than_cars_l296_296277

-- Define the conditions
def ratio_buses_to_cars : ℚ := 1 / 3
def number_of_cars : ℕ := 60

-- Define the function to calculate the number of buses from the number of cars
def number_of_buses (cars : ℕ) (ratio : ℚ) : ℕ :=
  (cars * ratio.num : ℚ / (ratio.denom : ℚ)).toNat

-- Define the condition that there are fewer buses than cars
def fewer_buses (cars : ℕ) (buses : ℕ) : ℕ :=
  cars - buses

-- Theorem statement
theorem buses_fewer_than_cars : 
  fewer_buses number_of_cars (number_of_buses number_of_cars ratio_buses_to_cars) = 40 := 
sorry

end buses_fewer_than_cars_l296_296277


namespace total_students_correct_l296_296048

def students_in_general_hall : ℕ := 30
def students_in_biology_hall : ℕ := 2 * students_in_general_hall
def combined_students_general_biology : ℕ := students_in_general_hall + students_in_biology_hall
def students_in_math_hall : ℕ := (3 * combined_students_general_biology) / 5
def total_students_in_all_halls : ℕ := students_in_general_hall + students_in_biology_hall + students_in_math_hall

theorem total_students_correct : total_students_in_all_halls = 144 := by
  -- Proof omitted, it should be
  sorry

end total_students_correct_l296_296048


namespace tangent_line_at_origin_l296_296815

noncomputable def f (x : ℝ) := Real.log (1 + x) + x * Real.exp (-x)

theorem tangent_line_at_origin : 
  ∀ (x : ℝ), (1 : ℝ) * x + (0 : ℝ) = 2 * x := 
sorry

end tangent_line_at_origin_l296_296815


namespace calculate_x_l296_296207

def percentage (p : ℚ) (n : ℚ) := (p / 100) * n

theorem calculate_x : 
  (percentage 47 1442 - percentage 36 1412) + 65 = 234.42 := 
by 
  sorry

end calculate_x_l296_296207


namespace value_of_business_l296_296488

theorem value_of_business (V : ℝ) (h₁ : (3/5) * (1/3) * V = 2000) : V = 10000 :=
by
  sorry

end value_of_business_l296_296488


namespace largest_number_of_positive_consecutive_integers_l296_296613

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end largest_number_of_positive_consecutive_integers_l296_296613


namespace six_digit_multiple_of_nine_l296_296521

theorem six_digit_multiple_of_nine (d : ℕ) (hd : d ≤ 9) (hn : 9 ∣ (30 + d)) : d = 6 := by
  sorry

end six_digit_multiple_of_nine_l296_296521


namespace waxberry_problem_l296_296183

noncomputable def batch_cannot_be_sold : ℚ := 1 - (8 / 9 * 9 / 10)

def probability_distribution (X : ℚ) : ℚ := 
  if X = -3200 then (1 / 5)^4 else
  if X = -2000 then 4 * (1 / 5)^3 * (4 / 5) else
  if X = -800 then 6 * (1 / 5)^2 * (4 / 5)^2 else
  if X = 400 then 4 * (1 / 5) * (4 / 5)^3 else
  if X = 1600 then (4 / 5)^4 else 0

noncomputable def expected_profit : ℚ :=
  -3200 * probability_distribution (-3200) +
  -2000 * probability_distribution (-2000) +
  -800 * probability_distribution (-800) +
  400 * probability_distribution (400) +
  1600 * probability_distribution (1600)

theorem waxberry_problem : 
  batch_cannot_be_sold = 1 / 5 ∧ 
  (probability_distribution (-3200) = 1 / 625 ∧ 
   probability_distribution (-2000) = 16 / 625 ∧ 
   probability_distribution (-800) = 96 / 625 ∧ 
   probability_distribution (400) = 256 / 625 ∧ 
   probability_distribution (1600) = 256 / 625) ∧ 
  expected_profit = 640 :=
by 
  sorry

end waxberry_problem_l296_296183


namespace nabla_eq_37_l296_296648

def nabla (a b : ℤ) : ℤ := a * b + a - b

theorem nabla_eq_37 : nabla (-5) (-7) = 37 := by
  sorry

end nabla_eq_37_l296_296648


namespace Dan_work_hours_l296_296199

theorem Dan_work_hours (x : ℝ) :
  (1 / 15) * x + 3 / 5 = 1 → x = 6 :=
by
  intro h
  sorry

end Dan_work_hours_l296_296199


namespace unique_solution_triple_l296_296089

theorem unique_solution_triple (x y z : ℝ) (h1 : x + y = 3) (h2 : x * y = z^3) : (x = 1.5 ∧ y = 1.5 ∧ z = 0) :=
by
  sorry

end unique_solution_triple_l296_296089


namespace dogs_in_school_l296_296195

theorem dogs_in_school
  (sit: ℕ) (sit_and_stay: ℕ) (stay: ℕ) (stay_and_roll_over: ℕ)
  (roll_over: ℕ) (sit_and_roll_over: ℕ) (all_three: ℕ) (none: ℕ)
  (h1: sit = 50) (h2: sit_and_stay = 17) (h3: stay = 29)
  (h4: stay_and_roll_over = 12) (h5: roll_over = 34)
  (h6: sit_and_roll_over = 18) (h7: all_three = 9) (h8: none = 9) :
  sit + stay + roll_over + sit_and_stay + stay_and_roll_over + sit_and_roll_over - 2 * all_three + none = 84 :=
by sorry

end dogs_in_school_l296_296195


namespace percent_swans_non_ducks_l296_296567

def percent_ducks : ℝ := 35
def percent_swans : ℝ := 30
def percent_herons : ℝ := 20
def percent_geese : ℝ := 15
def percent_non_ducks := 100 - percent_ducks

theorem percent_swans_non_ducks : (percent_swans / percent_non_ducks) * 100 = 46.15 := 
by
  sorry

end percent_swans_non_ducks_l296_296567


namespace find_people_got_off_at_first_stop_l296_296278

def total_seats (rows : ℕ) (seats_per_row : ℕ) : ℕ :=
  rows * seats_per_row

def occupied_seats (total_seats : ℕ) (initial_people : ℕ) : ℕ :=
  total_seats - initial_people

def occupied_seats_after_first_stop (initial_people : ℕ) (boarded_first_stop : ℕ) (got_off_first_stop : ℕ) : ℕ :=
  (initial_people + boarded_first_stop) - got_off_first_stop

def occupied_seats_after_second_stop (occupied_after_first_stop : ℕ) (boarded_second_stop : ℕ) (got_off_second_stop : ℕ) : ℕ :=
  (occupied_after_first_stop + boarded_second_stop) - got_off_second_stop

theorem find_people_got_off_at_first_stop
  (initial_people : ℕ := 16)
  (boarded_first_stop : ℕ := 15)
  (total_rows : ℕ := 23)
  (seats_per_row : ℕ := 4)
  (boarded_second_stop : ℕ := 17)
  (got_off_second_stop : ℕ := 10)
  (empty_seats_after_second_stop : ℕ := 57)
  : ∃ x, (occupied_seats_after_second_stop (occupied_seats_after_first_stop initial_people boarded_first_stop x) boarded_second_stop got_off_second_stop) = total_seats total_rows seats_per_row - empty_seats_after_second_stop :=
by
  sorry

end find_people_got_off_at_first_stop_l296_296278


namespace leo_current_weight_l296_296749

variable (L K : ℝ)

noncomputable def leo_current_weight_predicate :=
  (L + 10 = 1.5 * K) ∧ (L + K = 180)

theorem leo_current_weight : leo_current_weight_predicate L K → L = 104 := by
  sorry

end leo_current_weight_l296_296749


namespace intersection_on_semicircle_l296_296146

theorem intersection_on_semicircle {A B C H D P Q : Point} {ω : Set Point} :
  right_triangle ABC C →
  foot_of_altitude C H →
  D ∈ triangle CBH →
  midpoint (CH) (AD) →
  (BD) ∩ (CH) = P →
  semicircle_with_diameter ω D B →
  tangent_through P ω Q →
  ∃ X, X ∈ ω ∧ X ∈ (CQ) ∧ X ∈ (AD) :=
by
  sorry

end intersection_on_semicircle_l296_296146


namespace F_double_reflection_l296_296890

structure Point where
  x : ℝ
  y : ℝ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := -1, y := -1 }

theorem F_double_reflection :
  reflect_x (reflect_y F) = { x := 1, y := 1 } :=
  sorry

end F_double_reflection_l296_296890


namespace number_of_terriers_groomed_l296_296164

-- Define the initial constants and the conditions from the problem statement
def time_to_groom_poodle := 30
def time_to_groom_terrier := 15
def number_of_poodles := 3
def total_grooming_time := 210

-- Define the problem to prove that the number of terriers groomed is 8
theorem number_of_terriers_groomed (groom_time_poodle groom_time_terrier num_poodles total_time : ℕ) : 
  groom_time_poodle = time_to_groom_poodle → 
  groom_time_terrier = time_to_groom_terrier →
  num_poodles = number_of_poodles →
  total_time = total_grooming_time →
  ∃ n : ℕ, n * groom_time_terrier + num_poodles * groom_time_poodle = total_time ∧ n = 8 := 
by
  intros h1 h2 h3 h4
  sorry

end number_of_terriers_groomed_l296_296164


namespace value_of_a_l296_296000

theorem value_of_a (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^2 - a * x + 4) (h₂ : ∀ x, f (x + 1) = f (1 - x)) :
  a = 2 :=
sorry

end value_of_a_l296_296000


namespace fencing_required_l296_296069

variable (L W : ℝ)
variable (Area : ℝ := 20 * W)

theorem fencing_required (hL : L = 20) (hArea : L * W = 600) : 20 + 2 * W = 80 := by
  sorry

end fencing_required_l296_296069


namespace problem_b_solution_l296_296364

-- Define the expressions as functions
def Sum1 := (∑ i in range 101, i)
def Sum2 := Nat.InfiniteSum (fun n => n)
def Sum3 (n : ℕ) := (∑ i in range (n + 1), i)

-- Define the condition for an algorithm (finite sum)
def can_be_solved_by_algorithm (S : ℕ) := True

theorem problem_b_solution :
  can_be_solved_by_algorithm Sum1 ∧
  ¬ can_be_solved_by_algorithm Sum2 ∧
  ∀ (n : ℕ) (h : n ≥ 1), can_be_solved_by_algorithm (Sum3 n) :=
by sorry

end problem_b_solution_l296_296364


namespace expr_value_l296_296679

variable (a : ℝ)
variable (h : a^2 - 3 * a - 1011 = 0)

theorem expr_value : 2 * a^2 - 6 * a + 1 = 2023 :=
by
  -- insert proof here
  sorry

end expr_value_l296_296679


namespace find_r_divisibility_l296_296391

theorem find_r_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * (x - r)^2 * (x - s) = 10 * x^3 - 5 * x^2 - 52 * x + 56) → r = 4 / 3 :=
by
  sorry

end find_r_divisibility_l296_296391


namespace base_digit_difference_l296_296973

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l296_296973


namespace min_ones_count_in_100_numbers_l296_296529

def sum_eq_product (l : List ℕ) : Prop :=
  l.sum = l.prod

theorem min_ones_count_in_100_numbers : ∀ l : List ℕ, l.length = 100 → sum_eq_product l → l.count 1 ≥ 95 :=
by sorry

end min_ones_count_in_100_numbers_l296_296529


namespace problem1_problem2_l296_296365

theorem problem1 : (40 * Real.sqrt 3 - 18 * Real.sqrt 3 + 8 * Real.sqrt 3) / 6 = 5 * Real.sqrt 3 := 
by sorry

theorem problem2 : (Real.sqrt 3 - 2)^2023 * (Real.sqrt 3 + 2)^2023
                 - Real.sqrt 4 * Real.sqrt (1 / 2)
                 - (Real.pi - 1)^0
                = -2 - Real.sqrt 2 :=
by sorry

end problem1_problem2_l296_296365


namespace mila_needs_48_hours_to_earn_as_much_as_agnes_l296_296460

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end mila_needs_48_hours_to_earn_as_much_as_agnes_l296_296460


namespace problem_1_problem_2_problem_3_l296_296252

theorem problem_1 (x y : ℝ) : x^2 + y^2 + x * y + x + y ≥ -1 / 3 := 
by sorry

theorem problem_2 (x y z : ℝ) : x^2 + y^2 + z^2 + x * y + y * z + z * x + x + y + z ≥ -3 / 8 := 
by sorry

theorem problem_3 (x y z r : ℝ) : x^2 + y^2 + z^2 + r^2 + x * y + x * z + x * r + y * z + y * r + z * r + x + y + z + r ≥ -2 / 5 := 
by sorry

end problem_1_problem_2_problem_3_l296_296252


namespace part1_part2_l296_296427

noncomputable def triangle_area (A B C : ℝ) (a b c : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem part1 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  triangle_area A B C a b c = Real.sqrt 3 :=
sorry

theorem part2 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  ∃ B, 
  (B = 2 * π / 3) ∧ (4 * Real.sin C^2 + 3 * Real.sin A^2 + 2) / (Real.sin B^2) = 5 :=
sorry

end part1_part2_l296_296427


namespace geo_seq_condition_l296_296010

-- Definitions based on conditions
variable (a b c : ℝ)

-- Condition of forming a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, -1 * r = a ∧ a * r = b ∧ b * r = c ∧ c * r = -9

-- Proof problem statement
theorem geo_seq_condition (h : geometric_sequence a b c) : b = -3 ∧ a * c = 9 :=
sorry

end geo_seq_condition_l296_296010


namespace Hazel_shirts_proof_l296_296547

variable (H : ℕ)

def shirts_received_by_Razel (h_shirts : ℕ) : ℕ :=
  2 * h_shirts

def total_shirts (h_shirts : ℕ) (r_shirts : ℕ) : ℕ :=
  h_shirts + r_shirts

theorem Hazel_shirts_proof
  (h_shirts : ℕ)
  (r_shirts : ℕ)
  (total : ℕ)
  (H_nonneg : 0 ≤ h_shirts)
  (R_twice_H : r_shirts = shirts_received_by_Razel h_shirts)
  (T_total : total = total_shirts h_shirts r_shirts)
  (total_is_18 : total = 18) :
  h_shirts = 6 :=
by
  sorry

end Hazel_shirts_proof_l296_296547


namespace part1_part2_l296_296714

theorem part1 (u v w : ℤ) (h_uv : gcd u v = 1) (h_vw : gcd v w = 1) (h_wu : gcd w u = 1) 
: gcd (u * v + v * w + w * u) (u * v * w) = 1 :=
sorry

theorem part2 (u v w : ℤ) (b := u * v + v * w + w * u) (c := u * v * w) (h : gcd b c = 1) 
: gcd u v = 1 ∧ gcd v w = 1 ∧ gcd w u = 1 :=
sorry

end part1_part2_l296_296714


namespace quadratic_root_reciprocal_l296_296533

theorem quadratic_root_reciprocal (p q r s : ℝ) 
    (h1 : ∃ a : ℝ, a^2 + p * a + q = 0 ∧ (1 / a)^2 + r * (1 / a) + s = 0) :
    (p * s - r) * (q * r - p) = (q * s - 1)^2 :=
by
  sorry

end quadratic_root_reciprocal_l296_296533


namespace intersection_eq_one_l296_296686

def M : Set ℕ := {0, 1}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_eq_one : M ∩ N = {1} := 
by
  sorry

end intersection_eq_one_l296_296686


namespace roots_in_interval_l296_296605

theorem roots_in_interval (a b : ℝ) (hb : b > 0) (h_discriminant : a^2 - 4 * b > 0)
  (h_root_interval : ∃ r1 r2 : ℝ, r1 + r2 = -a ∧ r1 * r2 = b ∧ ((-1 ≤ r1 ∧ r1 ≤ 1 ∧ (r2 < -1 ∨ 1 < r2)) ∨ (-1 ≤ r2 ∧ r2 ≤ 1 ∧ (r1 < -1 ∨ 1 < r1)))) : 
  ∃ r : ℝ, (r + a) * r + b = 0 ∧ -b < r ∧ r < b :=
by
  sorry

end roots_in_interval_l296_296605


namespace ParticlePaths128_l296_296917

theorem ParticlePaths128 :
  let moves (p q : ℕ) := (p = 1 ∧ q = 0) ∨ (p = 0 ∧ q = 1) ∨ (p = 1 ∧ q = 1)
  ∧ ∀ x : ℕ × ℕ, x.1 ≤ 6 ∧ x.2 ≤ 6
  → (∃ f : ℕ → ℕ × ℕ, (f 0 = (0, 0)) ∧ (f 12 = (6, 6)) ∧ (∀ n, moves (f (n + 1)).1 (f n).1 ∧ moves (f (n + 1)).2 (f n).2 ∧ f n ≠ f (n - 1))
  → 128 :=
sorry

end ParticlePaths128_l296_296917


namespace M_intersect_P_l296_296141

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }
noncomputable def P : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

theorem M_intersect_P :
  M ∩ P = { y | y ≥ 1 } :=
sorry

end M_intersect_P_l296_296141


namespace mr_slinkums_shipments_l296_296769

theorem mr_slinkums_shipments 
  (T : ℝ) 
  (h : (3 / 4) * T = 150) : 
  T = 200 := 
sorry

end mr_slinkums_shipments_l296_296769


namespace max_integer_value_of_f_l296_296085

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 9 * x + 13) / (3 * x^2 + 9 * x + 5)

theorem max_integer_value_of_f : ∀ x : ℝ, ∃ n : ℤ, f x ≤ n ∧ n = 2 :=
by 
  sorry

end max_integer_value_of_f_l296_296085


namespace digit_for_multiple_of_9_l296_296524

theorem digit_for_multiple_of_9 : 
  -- Condition: Sum of the digits 4, 5, 6, 7, 8, and d must be divisible by 9.
  (∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (4 + 5 + 6 + 7 + 8 + d) % 9 = 0) →
  -- Result: The digit d that makes 45678d a multiple of 9 is 6.
  d = 6 :=
by
  sorry

end digit_for_multiple_of_9_l296_296524


namespace probability_same_gender_probability_same_school_l296_296443

theorem probability_same_gender :
  let school_A := ({m_A1, m_A2, f_A} : Finset (String))
      school_B := ({m_B, f_B1, f_B2} : Finset (String))
      total_outcomes := (school_A × school_B).card
      same_gender_outcomes := 
        (({m_A1, m_A2} × {m_B}) ∪ ({f_A} × {f_B1, f_B2})).card
  in (same_gender_outcomes : ℚ) / total_outcomes = 4 / 9 := 
by
  sorry

theorem probability_same_school :
  let teachers := ({m_A1, m_A2, f_A, m_B, f_B1, f_B2} : Finset (String))
      total_outcomes := (teachers.powerset.filter (λ s, s.card = 2)).card
      same_school_outcomes := 
        (({m_A1, m_A2, f_A}.powerset.filter (λ s, s.card = 2)) ∪ 
         ({m_B, f_B1, f_B2}.powerset.filter (λ s, s.card = 2))).card
  in (same_school_outcomes : ℚ) / total_outcomes = 2 / 5 :=
by
  sorry

end probability_same_gender_probability_same_school_l296_296443


namespace meal_service_count_l296_296034

/-- Define the number of people -/
def people_count : ℕ := 10

/-- Define the number of people that order pasta -/
def pasta_count : ℕ := 5

/-- Define the number of people that order salad -/
def salad_count : ℕ := 5

/-- Combination function to choose 2 people from 10 -/
def choose_2_from_10 : ℕ := Nat.choose 10 2

/-- Number of derangements of 8 people where exactly 2 people receive their correct meals -/
def derangement_8 : ℕ := 21

/-- Number of ways to correctly serve the meals where exactly 2 people receive the correct meal -/
theorem meal_service_count :
  choose_2_from_10 * derangement_8 = 945 :=
  by sorry

end meal_service_count_l296_296034


namespace angie_pretzels_dave_pretzels_l296_296927

theorem angie_pretzels (B S A : ℕ) (hB : B = 12) (hS : S = B / 2) (hA : A = 3 * S) : A = 18 := by
  -- We state the problem using variables B, S, and A for Barry, Shelly, and Angie respectively
  sorry

theorem dave_pretzels (A S D : ℕ) (hA : A = 18) (hS : S = 12 / 2) (hD : D = 25 * (A + S) / 100) : D = 6 := by
  -- We use variables A and S from the first theorem, and introduce D for Dave
  sorry

end angie_pretzels_dave_pretzels_l296_296927


namespace base_digit_difference_l296_296968

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l296_296968


namespace brian_needs_some_cartons_l296_296151

def servings_per_person : ℕ := sorry -- This should be defined with the actual number of servings per person.
def family_members : ℕ := 8
def us_cup_in_ml : ℕ := 250
def ml_per_serving : ℕ := us_cup_in_ml / 2
def ml_per_liter : ℕ := 1000

def total_milk_needed (servings_per_person : ℕ) : ℕ :=
  family_members * servings_per_person * ml_per_serving

def cartons_of_milk_needed (servings_per_person : ℕ) : ℕ :=
  total_milk_needed servings_per_person / ml_per_liter + if total_milk_needed servings_per_person % ml_per_liter = 0 then 0 else 1

theorem brian_needs_some_cartons (servings_per_person : ℕ) : 
  cartons_of_milk_needed servings_per_person = (family_members * servings_per_person * ml_per_serving / ml_per_liter + 
  if (family_members * servings_per_person * ml_per_serving) % ml_per_liter = 0 then 0 else 1) := 
by 
  sorry

end brian_needs_some_cartons_l296_296151


namespace total_fruit_pieces_correct_l296_296238

/-
  Define the quantities of each type of fruit.
-/
def red_apples : Nat := 9
def green_apples : Nat := 4
def purple_grapes : Nat := 3
def yellow_bananas : Nat := 6
def orange_oranges : Nat := 2

/-
  The total number of fruit pieces in the basket.
-/
def total_fruit_pieces : Nat := red_apples + green_apples + purple_grapes + yellow_bananas + orange_oranges

/-
  Prove that the total number of fruit pieces is 24.
-/
theorem total_fruit_pieces_correct : total_fruit_pieces = 24 := by
  sorry

end total_fruit_pieces_correct_l296_296238


namespace zain_coin_total_l296_296335

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l296_296335


namespace intersection_of_A_and_B_l296_296436

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := { x | x^2 + x ≤ 0}

theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0} :=
by
  sorry

end intersection_of_A_and_B_l296_296436


namespace digit_for_multiple_of_9_l296_296523

theorem digit_for_multiple_of_9 : 
  -- Condition: Sum of the digits 4, 5, 6, 7, 8, and d must be divisible by 9.
  (∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (4 + 5 + 6 + 7 + 8 + d) % 9 = 0) →
  -- Result: The digit d that makes 45678d a multiple of 9 is 6.
  d = 6 :=
by
  sorry

end digit_for_multiple_of_9_l296_296523


namespace friends_count_l296_296429

-- Define that Laura has 28 blocks
def blocks := 28

-- Define that each friend gets 7 blocks
def blocks_per_friend := 7

-- The proof statement we want to prove
theorem friends_count : blocks / blocks_per_friend = 4 := by
  sorry

end friends_count_l296_296429


namespace train_length_l296_296922

theorem train_length :
  let speed_kmph := 63
  let time_seconds := 16
  let speed_mps := (speed_kmph * 1000) / 3600
  let length_meters := speed_mps * time_seconds
  length_meters = 280 := 
by
  sorry

end train_length_l296_296922


namespace eight_div_repeat_three_l296_296318

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l296_296318


namespace problem_statement_l296_296097

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := x^n

variable (a : ℝ)
variable (h : a ≠ 1)

theorem problem_statement :
  (f 11 (f 13 a)) ^ 14 = f 2002 a ∧
  f 11 (f 13 (f 14 a)) = f 2002 a :=
by
  sorry

end problem_statement_l296_296097


namespace mean_temperature_l296_296594

def temperatures : List Int := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : Int) = -2 := by
  sorry

end mean_temperature_l296_296594


namespace units_digit_of_product_l296_296055

theorem units_digit_of_product : 
  (4 * 6 * 9) % 10 = 6 := 
by
  sorry

end units_digit_of_product_l296_296055


namespace mod_z_range_l296_296480

noncomputable def z (t : ℝ) : ℂ := Complex.ofReal (1/t) + Complex.I * t

noncomputable def mod_z (t : ℝ) : ℝ := Complex.abs (z t)

theorem mod_z_range : 
  ∀ (t : ℝ), t ≠ 0 → ∃ (r : ℝ), r = mod_z t ∧ r ≥ Real.sqrt 2 :=
  by sorry

end mod_z_range_l296_296480


namespace length_of_equal_sides_l296_296924

-- Definitions based on conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ a = c)

def is_triangle (a b c : ℝ) : Prop :=
(a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def has_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
a + b + c = P

def one_side_length (a : ℝ) : Prop :=
a = 3

-- The proof statement
theorem length_of_equal_sides (a b c : ℝ) :
isosceles_triangle a b c →
is_triangle a b c →
has_perimeter a b c 7 →
one_side_length a ∨ one_side_length b ∨ one_side_length c →
(b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) :=
by
  intros iso tri per side_length
  sorry

end length_of_equal_sides_l296_296924


namespace not_partitionable_1_to_15_l296_296262

theorem not_partitionable_1_to_15 :
  ∀ (A B : Finset ℕ), (∀ x ∈ A, x ∈ Finset.range 16) →
    (∀ x ∈ B, x ∈ Finset.range 16) →
    A.card = 2 → B.card = 13 →
    A ∪ B = Finset.range 16 →
    ¬(A.sum id = B.prod id) :=
by
  -- To be proved
  sorry

end not_partitionable_1_to_15_l296_296262


namespace range_of_m_l296_296214

def P (m : ℝ) : Prop :=
  9 - m > 2 * m ∧ 2 * m > 0

def Q (m : ℝ) : Prop :=
  m > 0 ∧ (Real.sqrt (6) / 2 < Real.sqrt (5 + m) / Real.sqrt (5)) ∧ (Real.sqrt (5 + m) / Real.sqrt (5) < Real.sqrt (2))

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) → (0 < m ∧ m ≤ 5 / 2) ∨ (3 ≤ m ∧ m < 5) :=
sorry

end range_of_m_l296_296214


namespace jan_more_miles_than_ian_l296_296747

noncomputable def distance_diff (d t s : ℝ) : ℝ :=
  let han_distance := (s + 10) * (t + 2)
  let jan_distance := (s + 15) * (t + 3)
  jan_distance - (d + 100)

theorem jan_more_miles_than_ian {d t s : ℝ} (H : d = s * t) (H_han : d + 100 = (s + 10) * (t + 2)) : distance_diff d t s = 165 :=
by {
  sorry
}

end jan_more_miles_than_ian_l296_296747


namespace line_equation_l296_296602

theorem line_equation {k b : ℝ} 
  (h1 : (∀ x : ℝ, k * x + b = -4 * x + 2023 → k = -4))
  (h2 : b = -5) :
  ∀ x : ℝ, k * x + b = -4 * x - 5 := by
sorry

end line_equation_l296_296602


namespace mixed_gender_appointment_schemes_l296_296872

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 
  else n * factorial (n - 1)

noncomputable def P (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

theorem mixed_gender_appointment_schemes : 
  let total_students := 9
  let total_permutations := P total_students 3
  let male_students := 5
  let female_students := 4
  let male_permutations := P male_students 3
  let female_permutations := P female_students 3
  total_permutations - (male_permutations + female_permutations) = 420 :=
by 
  sorry

end mixed_gender_appointment_schemes_l296_296872


namespace tan_sum_pi_over_4_l296_296689

open Real

theorem tan_sum_pi_over_4 {α : ℝ} (h₁ : cos (2 * α) + sin α * (2 * sin α - 1) = 2 / 5) (h₂ : π / 4 < α) (h₃ : α < π) : 
    tan (α + π / 4) = 1 / 7 := sorry

end tan_sum_pi_over_4_l296_296689


namespace parabolic_points_l296_296107

noncomputable def A (x1 : ℝ) (y1 : ℝ) : Prop := y1 = x1^2 - 3
noncomputable def B (x2 : ℝ) (y2 : ℝ) : Prop := y2 = x2^2 - 3

theorem parabolic_points (x1 x2 y1 y2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2)
  (hA : A x1 y1) (hB : B x2 y2) : y1 < y2 :=
by
  sorry

end parabolic_points_l296_296107


namespace new_sample_variance_l296_296105

-- Definitions based on conditions
def sample_size (original : Nat) : Prop := original = 7
def sample_average (original : ℝ) : Prop := original = 5
def sample_variance (original : ℝ) : Prop := original = 2
def new_data_point (point : ℝ) : Prop := point = 5

-- Statement to be proved
theorem new_sample_variance (original_size : Nat) (original_avg : ℝ) (original_var : ℝ) (new_point : ℝ) 
  (h₁ : sample_size original_size) 
  (h₂ : sample_average original_avg) 
  (h₃ : sample_variance original_var) 
  (h₄ : new_data_point new_point) : 
  (8 * original_var + 0) / 8 = 7 / 4 := 
by 
  sorry

end new_sample_variance_l296_296105


namespace bead_game_solution_l296_296494

-- Define the main theorem, stating the solution is valid for r = (b + 1) / b
theorem bead_game_solution {r : ℚ} (h : r > 1) (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 1010) :
  r = (b + 1) / b ∧ (∀ k : ℕ, k ≤ 2021 → True) := by
  sorry

end bead_game_solution_l296_296494


namespace goldfish_growth_solution_l296_296773

def goldfish_growth_problem : Prop :=
  ∃ n : ℕ, 
    (∀ k, (k < n → 3 * (5:ℕ)^k ≠ 243 * (3:ℕ)^k)) ∧
    3 * (5:ℕ)^n = 243 * (3:ℕ)^n

theorem goldfish_growth_solution : goldfish_growth_problem :=
sorry

end goldfish_growth_solution_l296_296773


namespace flagpole_shadow_length_correct_l296_296352

noncomputable def flagpole_shadow_length (flagpole_height building_height building_shadow_length : ℕ) :=
  flagpole_height * building_shadow_length / building_height

theorem flagpole_shadow_length_correct :
  flagpole_shadow_length 18 20 50 = 45 :=
by
  sorry

end flagpole_shadow_length_correct_l296_296352


namespace units_digit_33_219_89_plus_89_19_l296_296895

theorem units_digit_33_219_89_plus_89_19 :
  let units_digit x := x % 10
  units_digit (33 * 219 ^ 89 + 89 ^ 19) = 8 :=
by
  sorry

end units_digit_33_219_89_plus_89_19_l296_296895


namespace hyperbola_h_k_a_b_sum_eq_l296_296982

theorem hyperbola_h_k_a_b_sum_eq :
  ∃ (h k a b : ℝ), 
  h = 0 ∧ 
  k = 0 ∧ 
  a = 4 ∧ 
  (c : ℝ) = 8 ∧ 
  c^2 = a^2 + b^2 ∧ 
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
{ sorry }

end hyperbola_h_k_a_b_sum_eq_l296_296982


namespace meal_cost_with_tip_l296_296498

theorem meal_cost_with_tip 
  (cost_samosas : ℕ := 3 * 2)
  (cost_pakoras : ℕ := 4 * 3)
  (cost_lassi : ℕ := 2)
  (total_cost_before_tip := cost_samosas + cost_pakoras + cost_lassi)
  (tip : ℝ := 0.25 * total_cost_before_tip) :
  (total_cost_before_tip + tip = 25) :=
sorry

end meal_cost_with_tip_l296_296498


namespace edward_cards_l296_296652

noncomputable def num_cards_each_binder : ℝ := (7496.5 + 27.7) / 23
noncomputable def num_cards_fewer_binder : ℝ := num_cards_each_binder - 27.7

theorem edward_cards : 
  (⌊num_cards_each_binder + 0.5⌋ = 327) ∧ (⌊num_cards_fewer_binder + 0.5⌋ = 299) :=
by
  sorry

end edward_cards_l296_296652


namespace coordinates_of_point_A_in_third_quadrant_l296_296834

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := abs y

def distance_to_y_axis (x : ℝ) : ℝ := abs x

theorem coordinates_of_point_A_in_third_quadrant 
  (x y : ℝ)
  (h1 : point_in_third_quadrant x y)
  (h2 : distance_to_x_axis y = 2)
  (h3 : distance_to_y_axis x = 3) :
  (x, y) = (-3, -2) :=
  sorry

end coordinates_of_point_A_in_third_quadrant_l296_296834


namespace division_by_repeating_decimal_l296_296307

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l296_296307


namespace tan_C_l296_296980

theorem tan_C (A B C : ℝ) (hABC : A + B + C = π) (tan_A : Real.tan A = 1 / 2) 
  (cos_B : Real.cos B = 3 * Real.sqrt 10 / 10) : Real.tan C = -1 :=
by
  sorry

end tan_C_l296_296980


namespace eight_div_repeating_three_l296_296294

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l296_296294


namespace man_distance_from_start_l296_296763

noncomputable def distance_from_start (west_distance north_distance : ℝ) : ℝ :=
  Real.sqrt (west_distance^2 + north_distance^2)

theorem man_distance_from_start :
  distance_from_start 10 10 = Real.sqrt 200 :=
by
  sorry

end man_distance_from_start_l296_296763


namespace converse_prop_inverse_prop_contrapositive_prop_l296_296059

-- Given condition: the original proposition is true
axiom original_prop : ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0

-- Converse: If x=0 or y=0, then xy=0 - prove this is true
theorem converse_prop (x y : ℝ) : (x = 0 ∨ y = 0) → x * y = 0 :=
by
  sorry

-- Inverse: If xy ≠ 0, then x ≠ 0 and y ≠ 0 - prove this is true
theorem inverse_prop (x y : ℝ) : x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0 :=
by
  sorry

-- Contrapositive: If x ≠ 0 and y ≠ 0, then xy ≠ 0 - prove this is true
theorem contrapositive_prop (x y : ℝ) : (x ≠ 0 ∧ y ≠ 0) → x * y ≠ 0 :=
by
  sorry

end converse_prop_inverse_prop_contrapositive_prop_l296_296059


namespace people_and_cars_equation_l296_296128

theorem people_and_cars_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end people_and_cars_equation_l296_296128


namespace congruent_triangles_solve_x_l296_296813

theorem congruent_triangles_solve_x (x : ℝ) (h1 : x > 0)
    (h2 : x^2 - 1 = 3) (h3 : x^2 + 1 = 5) (h4 : x^2 + 3 = 7) : x = 2 :=
by
  sorry

end congruent_triangles_solve_x_l296_296813


namespace log_expression_l296_296206

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression :
  log_base 4 16 - (log_base 2 3 * log_base 3 2) = 1 := by
  sorry

end log_expression_l296_296206


namespace tan_5pi_over_4_l296_296661

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l296_296661


namespace total_number_of_trees_l296_296160

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l296_296160


namespace division_by_repeating_decimal_l296_296305

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l296_296305


namespace num_triangles_in_n_gon_l296_296440

-- Definitions for the problem in Lean based on provided conditions
def n_gon (n : ℕ) : Type := sorry  -- Define n-gon as a polygon with n sides
def non_intersecting_diagonals (n : ℕ) : Prop := sorry  -- Define the property of non-intersecting diagonals in an n-gon
def num_triangles (n : ℕ) : ℕ := sorry  -- Define a function to calculate the number of triangles formed by the diagonals in an n-gon

-- Statement of the theorem to prove
theorem num_triangles_in_n_gon (n : ℕ) (h : non_intersecting_diagonals n) : num_triangles n = n - 2 :=
by
  sorry

end num_triangles_in_n_gon_l296_296440


namespace remainder_17_pow_63_mod_7_l296_296617

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l296_296617


namespace original_price_of_dish_l296_296182

-- Define the variables and conditions explicitly
variables (P : ℝ)

-- John's payment after discount and tip over original price
def john_payment : ℝ := 0.9 * P + 0.15 * P

-- Jane's payment after discount and tip over discounted price
def jane_payment : ℝ := 0.9 * P + 0.135 * P

-- Given condition that John's payment is $0.63 more than Jane's
def payment_difference : Prop := john_payment P - jane_payment P = 0.63

theorem original_price_of_dish (h : payment_difference P) : P = 42 :=
by sorry

end original_price_of_dish_l296_296182


namespace not_distributive_add_mul_l296_296489

-- Definition of the addition operation on pairs of real numbers
def pair_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst + b.fst, a.snd + b.snd)

-- Definition of the multiplication operation on pairs of real numbers
def pair_mul (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst)

-- The problem statement: distributive law of addition over multiplication does not hold
theorem not_distributive_add_mul (a b c : ℝ × ℝ) :
  pair_add a (pair_mul b c) ≠ pair_mul (pair_add a b) (pair_add a c) :=
sorry

end not_distributive_add_mul_l296_296489


namespace eight_div_repeating_three_l296_296293

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l296_296293


namespace desiree_age_l296_296373

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end desiree_age_l296_296373


namespace camera_lens_distance_l296_296189

theorem camera_lens_distance (f u : ℝ) (h_fu : f ≠ u) (h_f : f ≠ 0) (h_u : u ≠ 0) :
  (∃ v : ℝ, (1 / f) = (1 / u) + (1 / v) ∧ v = (f * u) / (u - f)) :=
by {
  sorry
}

end camera_lens_distance_l296_296189


namespace smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l296_296744

theorem smallest_prime_divisor_of_3_pow_19_add_11_pow_23 :
  ∀ (n : ℕ), Prime n → n ∣ 3^19 + 11^23 → n = 2 :=
by
  sorry

end smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l296_296744


namespace minimum_amount_l296_296627

-- Define the basic conditions
variables (A O S G : ℕ)
variables (candy_cost : ℝ := 0.1)

-- Hypotheses from the problem
def conditions :=
  A = 2 * O ∧
  S = 2 * G ∧
  A = 2 * S ∧
  A + O + S + G = 90 

-- The question to prove
theorem minimum_amount (h : conditions A O S G) : 
  ∃ (cost : ℝ), cost = 1.9 := 
sorry

end minimum_amount_l296_296627


namespace digit_68th_is_1_l296_296434

noncomputable def largest_n : ℕ :=
  (10^100 - 1) / 14

def digit_at_68th_place (n : ℕ) : ℕ :=
  (n / 10^(68 - 1)) % 10

theorem digit_68th_is_1 : digit_at_68th_place largest_n = 1 :=
sorry

end digit_68th_is_1_l296_296434


namespace problem_l296_296250

noncomputable def k : ℝ := 2.9

theorem problem (k : ℝ) (hₖ : k > 1) 
    (h_sum : ∑' n, (7 * n + 2) / k^n = 20 / 3) : 
    k = 2.9 := 
sorry

end problem_l296_296250


namespace perpendicular_lines_implies_m_values_l296_296225

-- Define the equations of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y - 1 = 0

-- Define the condition of perpendicularity between lines l1 and l2
def perpendicular (m : ℝ) : Prop :=
  let a1 := (m + 2) / (m - 2)
  let a2 := -3 / m
  a1 * a2 = -1

-- The statement to be proved
theorem perpendicular_lines_implies_m_values (m : ℝ) :
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y → perpendicular m) → (m = -1 ∨ m = 6) :=
sorry

end perpendicular_lines_implies_m_values_l296_296225


namespace cost_effective_combination_l296_296563

/--
Jackson wants to impress his girlfriend by filling her hot tub with champagne.
The hot tub holds 400 liters of liquid. He has three types of champagne bottles:
1. Small bottle: Holds 0.75 liters with a price of $70 per bottle.
2. Medium bottle: Holds 1.5 liters with a price of $120 per bottle.
3. Large bottle: Holds 3 liters with a price of $220 per bottle.

If he purchases more than 50 bottles of any type, he will get a 10% discount on 
that type. If he purchases over 100 bottles of any type, he will get 20% off 
on that type of bottles. 

Prove that the most cost-effective combination of bottles for 
Jackson to purchase is 134 large bottles for a total cost of $23,584 after the discount.
-/
theorem cost_effective_combination :
  let volume := 400
  let small_bottle_volume := 0.75
  let small_bottle_cost := 70
  let medium_bottle_volume := 1.5
  let medium_bottle_cost := 120
  let large_bottle_volume := 3
  let large_bottle_cost := 220
  let discount_50 := 0.10
  let discount_100 := 0.20
  let cost_134_large_bottles := (134 * large_bottle_cost) * (1 - discount_100)
  cost_134_large_bottles = 23584 :=
sorry

end cost_effective_combination_l296_296563


namespace shelby_drive_rain_minutes_l296_296444

theorem shelby_drive_rain_minutes
  (total_distance : ℝ)
  (total_time : ℝ)
  (sunny_speed : ℝ)
  (rainy_speed : ℝ)
  (t_sunny : ℝ)
  (t_rainy : ℝ) :
  total_distance = 20 →
  total_time = 50 →
  sunny_speed = 40 →
  rainy_speed = 25 →
  total_time = t_sunny + t_rainy →
  (sunny_speed / 60) * t_sunny + (rainy_speed / 60) * t_rainy = total_distance →
  t_rainy = 30 :=
by
  intros
  sorry

end shelby_drive_rain_minutes_l296_296444


namespace ways_to_sum_2022_using_2s_and_3s_l296_296410

theorem ways_to_sum_2022_using_2s_and_3s : 
  (∃ n : ℕ, n ≤ 337 ∧ 6 * 337 = 2022) →
  (finset.card (finset.Icc 0 337) = 338) :=
by
  intros n h
  rw finset.card_Icc
  sorry

end ways_to_sum_2022_using_2s_and_3s_l296_296410


namespace oranges_now_is_50_l296_296642

def initial_fruits : ℕ := 150
def remaining_fruits : ℕ := initial_fruits / 2
def num_limes (L : ℕ) (O : ℕ) : Prop := O = 2 * L
def total_remaining_fruits (L : ℕ) (O : ℕ) : Prop := O + L = remaining_fruits

theorem oranges_now_is_50 : ∃ O L : ℕ, num_limes L O ∧ total_remaining_fruits L O ∧ O = 50 := by
  sorry

end oranges_now_is_50_l296_296642


namespace tan_five_pi_over_four_l296_296656

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l296_296656


namespace find_z_l296_296683

theorem find_z 
  (m : ℕ)
  (h1 : (1^(m+1) / 5^(m+1)) * (1^18 / z^18) = 1 / (2 * 10^35))
  (hm : m = 34) :
  z = 4 := 
sorry

end find_z_l296_296683


namespace eight_div_repeating_three_l296_296299

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l296_296299


namespace calories_in_250g_of_lemonade_l296_296716

structure Lemonade :=
(lemon_juice_grams : ℕ)
(sugar_grams : ℕ)
(water_grams : ℕ)
(lemon_juice_calories_per_100g : ℕ)
(sugar_calories_per_100g : ℕ)
(water_calories_per_100g : ℕ)

def calorie_count (l : Lemonade) : ℕ :=
(l.lemon_juice_grams * l.lemon_juice_calories_per_100g / 100) +
(l.sugar_grams * l.sugar_calories_per_100g / 100) +
(l.water_grams * l.water_calories_per_100g / 100)

def total_weight (l : Lemonade) : ℕ :=
l.lemon_juice_grams + l.sugar_grams + l.water_grams

def caloric_density (l : Lemonade) : ℚ :=
calorie_count l / total_weight l

theorem calories_in_250g_of_lemonade :
  ∀ (l : Lemonade), 
  l = { lemon_juice_grams := 200, sugar_grams := 300, water_grams := 500,
        lemon_juice_calories_per_100g := 40,
        sugar_calories_per_100g := 390,
        water_calories_per_100g := 0 } →
  (caloric_density l * 250 = 312.5) :=
sorry

end calories_in_250g_of_lemonade_l296_296716


namespace Zain_coins_total_l296_296333

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l296_296333


namespace chess_tournament_winner_l296_296558

theorem chess_tournament_winner :
  ∀ (x : ℕ) (P₉ P₁₀ : ℕ),
  (x > 0) →
  (9 * x) = 4 * P₃ →
  P₉ = (x * (x - 1)) / 2 + 9 * x^2 →
  P₁₀ = (9 * x * (9 * x - 1)) / 2 →
  (9 * x^2 - x) * 2 ≥ 81 * x^2 - 9 * x →
  x = 1 →
  P₃ = 9 :=
by
  sorry

end chess_tournament_winner_l296_296558


namespace maria_workday_ends_at_330_pm_l296_296859

/-- 
Given:
1. Maria's workday is 8 hours long.
2. Her workday does not include her lunch break.
3. Maria starts work at 7:00 A.M.
4. She takes her lunch break at 11:30 A.M., lasting 30 minutes.
Prove that Maria's workday ends at 3:30 P.M.
-/
def maria_end_workday : Prop :=
  let start_time : Nat := 7 * 60 -- in minutes
  let lunch_start_time : Nat := 11 * 60 + 30 -- in minutes
  let lunch_duration : Nat := 30 -- in minutes
  let lunch_end_time : Nat := lunch_start_time + lunch_duration
  let total_work_minutes : Nat := 8 * 60
  let work_before_lunch : Nat := lunch_start_time - start_time
  let remaining_work : Nat := total_work_minutes - work_before_lunch
  let end_time : Nat := lunch_end_time + remaining_work
  end_time = 15 * 60 + 30

theorem maria_workday_ends_at_330_pm : maria_end_workday :=
  by
    sorry

end maria_workday_ends_at_330_pm_l296_296859


namespace total_flower_petals_l296_296940

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end total_flower_petals_l296_296940


namespace book_price_distribution_l296_296754

theorem book_price_distribution :
  ∃ (x y z: ℤ), 
  x + y + z = 109 ∧
  (34 * x + 27.5 * y + 17.5 * z : ℝ) = 2845 ∧
  (x - y : ℤ).natAbs ≤ 2 ∧ (y - z).natAbs ≤ 2 := 
sorry

end book_price_distribution_l296_296754


namespace smallest_value_of_M_l296_296244

theorem smallest_value_of_M :
  ∀ (a b c d e f g M : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 → g > 0 →
  a + b + c + d + e + f + g = 2024 →
  M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g))))) →
  M = 338 :=
by
  intro a b c d e f g M ha hb hc hd he hf hg hsum hmax
  sorry

end smallest_value_of_M_l296_296244


namespace problem_solution_l296_296603

def otimes (a b : ℚ) : ℚ := (a ^ 3) / (b ^ 2)

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (-2016) / 729 := by
  sorry

end problem_solution_l296_296603


namespace tan_5pi_over_4_l296_296664

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l296_296664


namespace expand_polynomial_l296_296086

theorem expand_polynomial (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x ^ 2 - 36 := 
by {
  sorry
}

end expand_polynomial_l296_296086


namespace sum_of_valid_n_l296_296619

theorem sum_of_valid_n :
  (∀ n : ℕ, (nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16) → (n = 14 ∨ n = 16)) →
  14 + 16 = 30 :=
by
  intros h
  sorry

end sum_of_valid_n_l296_296619


namespace subtract_complex_eq_l296_296325

noncomputable def subtract_complex (a b : ℂ) : ℂ := a - b

theorem subtract_complex_eq (i : ℂ) (h_i : i^2 = -1) :
  subtract_complex (5 - 3 * i) (7 - 7 * i) = -2 + 4 * i :=
by
  sorry

end subtract_complex_eq_l296_296325


namespace find_x_l296_296805

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 ≥ 2 ∧ p2 ≥ 2 ∧ p3 ≥ 2 ∧ x = p1 * p2 * p3 ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
    x = 59048 := 
sorry

end find_x_l296_296805


namespace division_of_repeating_decimal_l296_296304

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l296_296304


namespace peanut_total_correct_l296_296841

-- Definitions based on the problem conditions:

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35
def total_peanuts : ℕ := jose_peanuts + kenya_peanuts + malachi_peanuts

-- Statement to be proven:
theorem peanut_total_correct : total_peanuts = 386 :=
by 
  -- The proof would be here, but we skip it according to the instruction
  sorry

end peanut_total_correct_l296_296841


namespace find_inheritance_amount_l296_296138

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end find_inheritance_amount_l296_296138


namespace min_sum_of_ab_l296_296530

theorem min_sum_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = a * b) :
  a + b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_sum_of_ab_l296_296530


namespace ratio_of_80_pencils_l296_296866

theorem ratio_of_80_pencils (C S : ℝ)
  (CP : ℝ := 80 * C)
  (L : ℝ := 30 * S)
  (SP : ℝ := 80 * S)
  (h : CP = SP + L) :
  CP / SP = 11 / 8 :=
by
  -- Start the proof
  sorry

end ratio_of_80_pencils_l296_296866


namespace math_problem_l296_296033

theorem math_problem (x y : ℕ) (h1 : (x + y * I)^3 = 2 + 11 * I) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y * I = 2 + I :=
sorry

end math_problem_l296_296033


namespace division_by_repeating_decimal_l296_296292

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l296_296292


namespace avg_annual_growth_rate_optimal_selling_price_l296_296178

-- Define the conditions and question for the first problem: average annual growth rate.
theorem avg_annual_growth_rate (initial final : ℝ) (years : ℕ) (growth_rate : ℝ) :
  initial = 200 ∧ final = 288 ∧ years = 2 ∧ (final = initial * (1 + growth_rate)^years) →
  growth_rate = 0.2 :=
by
  -- Proof will come here
  sorry

-- Define the conditions and question for the second problem: setting the selling price.
theorem optimal_selling_price (cost initial_volume : ℕ) (initial_price : ℝ) 
(additional_sales_per_dollar : ℕ) (desired_profit : ℝ) (optimal_price : ℝ) :
  cost = 50 ∧ initial_volume = 50 ∧ initial_price = 100 ∧ additional_sales_per_dollar = 5 ∧
  desired_profit = 4000 ∧ 
  (∃ p : ℝ, (p - cost) * (initial_volume + additional_sales_per_dollar * (initial_price - p)) = desired_profit ∧ p = optimal_price) →
  optimal_price = 70 :=
by
  -- Proof will come here
  sorry

end avg_annual_growth_rate_optimal_selling_price_l296_296178


namespace blue_bead_probability_no_adjacent_l296_296096

theorem blue_bead_probability_no_adjacent :
  let total_beads := 9
  let blue_beads := 5
  let green_beads := 3
  let red_bead := 1
  let total_permutations := Nat.factorial total_beads / (Nat.factorial blue_beads * Nat.factorial green_beads * Nat.factorial red_bead)
  let valid_arrangements := (Nat.factorial 4) / (Nat.factorial 3 * Nat.factorial 1)
  let no_adjacent_valid := 4
  let probability_no_adj := (no_adjacent_valid : ℚ) / total_permutations
  probability_no_adj = (1 : ℚ) / 126 := 
by
  sorry

end blue_bead_probability_no_adjacent_l296_296096


namespace rectangle_area_inscribed_circle_l296_296187

theorem rectangle_area_inscribed_circle 
  (radius : ℝ) (width len : ℝ) 
  (h_radius : radius = 5) 
  (h_width : width = 2 * radius) 
  (h_len_ratio : len = 3 * width) 
  : width * len = 300 := 
by
  sorry

end rectangle_area_inscribed_circle_l296_296187


namespace gina_initial_money_l296_296393

variable (M : ℝ)
variable (kept : ℝ := 170)

theorem gina_initial_money (h1 : M * 1 / 4 + M * 1 / 8 + M * 1 / 5 + kept = M) : 
  M = 400 :=
by
  sorry

end gina_initial_money_l296_296393


namespace even_fn_a_eq_zero_l296_296235

def f (x a : ℝ) : ℝ := x^2 - |x + a|

theorem even_fn_a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end even_fn_a_eq_zero_l296_296235


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l296_296901

noncomputable def repeating_decimal_fraction (x : ℚ) : ℚ :=
  if x = 0.345345345... then 115 / 333 else sorry

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.345345345... in 
  let fraction := repeating_decimal_fraction x in
  (fraction.num + fraction.denom) = 448 :=
by {
  sorry
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l296_296901


namespace train_speed_proof_l296_296358

variables (distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse : ℝ)

def question_statement : Prop :=
  distance_to_syracuse = 120 ∧
  total_time_hours = 5.5 ∧
  return_trip_speed = 38.71 →
  average_speed_to_syracuse = 50

theorem train_speed_proof :
  question_statement distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse :=
by
  -- sorry is used to indicate that the proof is omitted
  sorry

end train_speed_proof_l296_296358


namespace arithmetic_sequence_first_term_l296_296734

theorem arithmetic_sequence_first_term
  (a : ℕ) -- First term of the arithmetic sequence
  (d : ℕ := 3) -- Common difference, given as 3
  (n : ℕ := 20) -- Number of terms, given as 20
  (S : ℕ := 650) -- Sum of the sequence, given as 650
  (h : S = (n / 2) * (2 * a + (n - 1) * d)) : a = 4 := 
by
  sorry

end arithmetic_sequence_first_term_l296_296734


namespace minimum_problem_l296_296575

open BigOperators

theorem minimum_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / y) * (x + 1 / y - 2020) + (y + 1 / x) * (y + 1 / x - 2020) ≥ -2040200 := 
sorry

end minimum_problem_l296_296575


namespace total_earnings_l296_296780

theorem total_earnings :
  (15 * 2) + (12 * 1.5) = 48 := by
  sorry

end total_earnings_l296_296780


namespace eight_div_repeat_three_l296_296317

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l296_296317


namespace probability_no_two_adjacent_same_roll_l296_296947

theorem probability_no_two_adjacent_same_roll :
  let total_rolls := 6^5 in
  let valid_rolls := 875 in
  (valid_rolls : ℚ) / total_rolls = 875 / 1296 :=
by
  sorry

end probability_no_two_adjacent_same_roll_l296_296947


namespace leak_empty_time_l296_296644

theorem leak_empty_time
  (pump_fill_time : ℝ)
  (leak_fill_time : ℝ)
  (pump_fill_rate : pump_fill_time = 5)
  (leak_fill_rate : leak_fill_time = 10)
  : (1 / 5 - 1 / leak_fill_time)⁻¹ = 10 :=
by
  -- you can fill in the proof here
  sorry

end leak_empty_time_l296_296644


namespace profit_percentage_correct_l296_296356

noncomputable def overall_profit_percentage : ℚ :=
  let cost_radio := 225
  let overhead_radio := 15
  let price_radio := 300
  let cost_watch := 425
  let overhead_watch := 20
  let price_watch := 525
  let cost_mobile := 650
  let overhead_mobile := 30
  let price_mobile := 800
  
  let total_cost_price := (cost_radio + overhead_radio) + (cost_watch + overhead_watch) + (cost_mobile + overhead_mobile)
  let total_selling_price := price_radio + price_watch + price_mobile
  let total_profit := total_selling_price - total_cost_price
  (total_profit * 100 : ℚ) / total_cost_price
  
theorem profit_percentage_correct :
  overall_profit_percentage = 19.05 := by
  sorry

end profit_percentage_correct_l296_296356


namespace total_band_members_l296_296860

def total_people_in_band (flutes clarinets trumpets pianists : ℕ) 
(number_of_flutes_in band number_of_clarinets_in band number_of_trumpets_in band number_of_pianists_in_band : ℕ) : ℕ :=
number_of_flutes_in_band + number_of_clarinets_in_band + number_of_trumpets_in_band + number_of_pianists_in_band

theorem total_band_members :
  let flutes := 20
  let clarinets := 30
  let trumpets := 60
  let pianists := 20
  let number_of_flutes_in_band := (80 * flutes) / 100
  let number_of_clarinets_in_band := clarinets / 2
  let number_of_trumpets_in_band := trumpets / 3
  let number_of_pianists_in_band := pianists / 10
  total_people_in_band flutes clarinets trumpets pianists 
                          number_of_flutes_in_band 
                          number_of_clarinets_in_band 
                          number_of_trumpets_in_band 
                          number_of_pianists_in_band = 53 :=
by {
  sorry
}

end total_band_members_l296_296860


namespace max_sum_abc_l296_296571

theorem max_sum_abc
  (a b c : ℤ)
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (hA1 : A = (1/7 : ℚ) • ![![(-5 : ℚ), a], ![b, c]])
  (hA2 : A * A = 2 • (1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  a + b + c ≤ 79 :=
by
  sorry

end max_sum_abc_l296_296571


namespace division_of_repeating_decimal_l296_296302

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l296_296302


namespace first_consecutive_odd_number_l296_296607

theorem first_consecutive_odd_number :
  ∃ k : Int, 2 * k - 1 + 2 * k + 1 + 2 * k + 3 = 2 * k - 1 + 128 ∧ 2 * k - 1 = 61 :=
by
  sorry

end first_consecutive_odd_number_l296_296607


namespace minimum_a_l296_296539

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem minimum_a
  (a : ℝ)
  (h : ∀ x : ℤ, (f x)^2 - a * f x ≤ 0 → ∃! x : ℤ, (f x)^2 - a * f x = 0) :
  a = Real.exp 2 + 1 :=
sorry

end minimum_a_l296_296539


namespace girls_joined_school_l296_296698

theorem girls_joined_school
  (initial_girls : ℕ)
  (initial_boys : ℕ)
  (total_pupils_after : ℕ)
  (computed_new_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  total_pupils_after = 1346 →
  computed_new_girls = total_pupils_after - (initial_girls + initial_boys) →
  computed_new_girls = 418 :=
by
  intros h_initial_girls h_initial_boys h_total_pupils_after h_computed_new_girls
  sorry

end girls_joined_school_l296_296698


namespace treaty_signed_on_wednesday_l296_296268

-- This function calculates the weekday after a given number of days since a known weekday.
def weekday_after (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

-- Given the problem conditions:
-- The war started on a Friday: 5th day of the week (considering Sunday as 0)
def war_start_day_of_week : ℕ := 5

-- The number of days after which the treaty was signed
def days_until_treaty : ℕ := 926

-- Expected final day (Wednesday): 3rd day of the week (considering Sunday as 0)
def treaty_day_of_week : ℕ := 3

-- The theorem to be proved:
theorem treaty_signed_on_wednesday :
  weekday_after war_start_day_of_week days_until_treaty = treaty_day_of_week :=
by
  sorry

end treaty_signed_on_wednesday_l296_296268


namespace ratio_of_distance_l296_296150

noncomputable def initial_distance : ℝ := 30 * 20

noncomputable def total_distance : ℝ := 2 * initial_distance

noncomputable def distance_after_storm : ℝ := initial_distance - 200

theorem ratio_of_distance (initial_distance : ℝ) (total_distance : ℝ) (distance_after_storm : ℝ) : 
  distance_after_storm / total_distance = 1 / 3 :=
by
  -- Given conditions
  have h1 : initial_distance = 30 * 20 := by sorry
  have h2 : total_distance = 2 * initial_distance := by sorry
  have h3 : distance_after_storm = initial_distance - 200 := by sorry
  -- Prove the ratio is 1 / 3
  sorry

end ratio_of_distance_l296_296150


namespace zain_coin_total_l296_296334

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l296_296334


namespace find_x_y_l296_296251

theorem find_x_y (A B C : ℝ) (x y : ℝ) (hA : A = 120) (hB : B = 100) (hC : C = 150)
  (hx : A = B + (x / 100) * B) (hy : A = C - (y / 100) * C) : x = 20 ∧ y = 20 :=
by
  sorry

end find_x_y_l296_296251


namespace noon_temperature_l296_296013

variable (a : ℝ)

theorem noon_temperature (h1 : ∀ (x : ℝ), x = a) (h2 : ∀ (y : ℝ), y = a + 10) :
  a + 10 = y :=
by
  sorry

end noon_temperature_l296_296013


namespace find_k_l296_296823

theorem find_k (x y k : ℝ) (h1 : 2 * x - y = 4) (h2 : k * x - 3 * y = 12) : k = 6 := by
  sorry

end find_k_l296_296823


namespace exist_two_numbers_with_GCD_and_LCM_l296_296788

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem exist_two_numbers_with_GCD_and_LCM :
  ∃ A B : ℕ, GCD A B = 21 ∧ LCM A B = 3969 ∧ ((A = 21 ∧ B = 3969) ∨ (A = 147 ∧ B = 567)) :=
by
  sorry

end exist_two_numbers_with_GCD_and_LCM_l296_296788


namespace play_only_one_sport_l296_296560

-- Given conditions
variable (total : ℕ := 150)
variable (B : ℕ := 65)
variable (T : ℕ := 80)
variable (Ba : ℕ := 60)
variable (B_T : ℕ := 20)
variable (B_Ba : ℕ := 15)
variable (T_Ba : ℕ := 25)
variable (B_T_Ba : ℕ := 10)
variable (N : ℕ := 12)

-- Prove the number of members that play only one sport is 115.
theorem play_only_one_sport : 
  (B - (B_T - B_T_Ba) - (B_Ba - B_T_Ba) - B_T_Ba) + 
  (T - (B_T - B_T_Ba) - (T_Ba - B_T_Ba) - B_T_Ba) + 
  (Ba - (B_Ba - B_T_Ba) - (T_Ba - B_T_Ba) - B_T_Ba) = 115 :=
by
  sorry

end play_only_one_sport_l296_296560


namespace comparison_of_abc_l296_296247

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem comparison_of_abc : b < a ∧ a < c :=
by
  sorry

end comparison_of_abc_l296_296247


namespace total_matches_played_l296_296015

-- Definitions
def victories_points := 3
def draws_points := 1
def defeats_points := 0
def points_after_5_games := 8
def games_played := 5
def target_points := 40
def remaining_wins_required := 9

-- Statement to prove
theorem total_matches_played :
  ∃ M : ℕ, points_after_5_games + victories_points * remaining_wins_required < target_points -> M = games_played + remaining_wins_required + 1 :=
sorry

end total_matches_played_l296_296015


namespace fraction_identity_l296_296008

theorem fraction_identity (a b : ℝ) (h : a / b = 3 / 4) : a / (a + b) = 3 / 7 := 
by
  sorry

end fraction_identity_l296_296008


namespace picasso_prints_probability_l296_296583

open Nat

theorem picasso_prints_probability :
  let total_items := 12
  let picasso_prints := 4
  let favorable_outcomes := factorial (total_items - picasso_prints + 1) * factorial picasso_prints
  let total_arrangements := factorial total_items
  let desired_probability := (favorable_outcomes : ℚ) / total_arrangements
  desired_probability = 1 / 55 :=
by
  let total_items := 12
  let picasso_prints := 4
  let favorable_outcomes := factorial (total_items - picasso_prints + 1) * factorial picasso_prints
  let total_arrangements := factorial total_items
  let desired_probability : ℚ := favorable_outcomes / total_arrangements
  show desired_probability = 1 / 55
  sorry

end picasso_prints_probability_l296_296583


namespace sin_cos_inequality_l296_296801

open Real

theorem sin_cos_inequality (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * π) :
  (sin (x - π / 6) > cos x) ↔ (π / 3 < x ∧ x < 4 * π / 3) :=
by sorry

end sin_cos_inequality_l296_296801


namespace employee_pay_l296_296341

theorem employee_pay (y : ℝ) (x : ℝ) (h1 : x = 1.2 * y) (h2 : x + y = 700) : y = 318.18 :=
by
  sorry

end employee_pay_l296_296341


namespace janelle_gave_green_marbles_l296_296020

def initial_green_marbles : ℕ := 26
def bags_blue_marbles : ℕ := 6
def marbles_per_bag : ℕ := 10
def total_blue_marbles : ℕ := bags_blue_marbles * marbles_per_bag
def total_marbles_after_gift : ℕ := 72
def blue_marbles_in_gift : ℕ := 8
def final_blue_marbles : ℕ := total_blue_marbles - blue_marbles_in_gift
def final_green_marbles : ℕ := total_marbles_after_gift - final_blue_marbles
def initial_green_marbles_after_gift : ℕ := final_green_marbles
def green_marbles_given : ℕ := initial_green_marbles - initial_green_marbles_after_gift

theorem janelle_gave_green_marbles :
  green_marbles_given = 6 :=
by {
  sorry
}

end janelle_gave_green_marbles_l296_296020


namespace Evelyn_bottle_caps_problem_l296_296944

theorem Evelyn_bottle_caps_problem (E : ℝ) (H1 : E - 18.0 = 45) : E = 63.0 := 
by
  sorry


end Evelyn_bottle_caps_problem_l296_296944


namespace average_rate_l296_296051

theorem average_rate (distance_run distance_swim : ℝ) (rate_run rate_swim : ℝ) 
  (h1 : distance_run = 2) (h2 : distance_swim = 2) (h3 : rate_run = 10) (h4 : rate_swim = 5) : 
  (distance_run + distance_swim) / ((distance_run / rate_run) * 60 + (distance_swim / rate_swim) * 60) = 0.1111 :=
by
  sorry

end average_rate_l296_296051


namespace tangent_circle_equation_l296_296484

theorem tangent_circle_equation :
  (∃ m : Real, ∃ n : Real,
    (∀ x y : Real, (x - m)^2 + (y - n)^2 = 36) ∧ 
    ((m - 0)^2 + (n - 3)^2 = 25) ∧
    n = 6 ∧ (m = 4 ∨ m = -4)) :=
sorry

end tangent_circle_equation_l296_296484


namespace soak_time_l296_296080

/-- 
Bill needs to soak his clothes for 4 minutes to get rid of each grass stain.
His clothes have 3 grass stains and 1 marinara stain.
The total soaking time is 19 minutes.
Prove that the number of minutes needed to soak for each marinara stain is 7.
-/
theorem soak_time (m : ℕ) (grass_stain_time : ℕ) (num_grass_stains : ℕ) (num_marinara_stains : ℕ) (total_time : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_time = 19) :
  m = 7 :=
by sorry

end soak_time_l296_296080


namespace div_power_n_minus_one_l296_296433

theorem div_power_n_minus_one (n : ℕ) (hn : n > 0) (h : n ∣ (2^n - 1)) : n = 1 := by
  sorry

end div_power_n_minus_one_l296_296433


namespace find_exercise_books_l296_296699

theorem find_exercise_books
  (pencil_ratio pen_ratio exercise_book_ratio eraser_ratio : ℕ)
  (total_pencils total_ratio_units : ℕ)
  (h1 : pencil_ratio = 10)
  (h2 : pen_ratio = 2)
  (h3 : exercise_book_ratio = 3)
  (h4 : eraser_ratio = 4)
  (h5 : total_pencils = 150)
  (h6 : total_ratio_units = pencil_ratio + pen_ratio + exercise_book_ratio + eraser_ratio) :
  (total_pencils / pencil_ratio) * exercise_book_ratio = 45 :=
by
  sorry

end find_exercise_books_l296_296699


namespace find_y_solution_l296_296380

variable (y : ℚ)

theorem find_y_solution (h : (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5) : 
    y = -17/6 :=
by
  sorry

end find_y_solution_l296_296380


namespace last_letter_of_100th_permutation_l296_296450

noncomputable def BRICK := ['B', 'R', 'I', 'C', 'K']

theorem last_letter_of_100th_permutation :
  (Permutations (multiset_finset (multiset.of_finset (set_of_finite (set_univ BRICK))))) 100).last = 'K' := sorry

end last_letter_of_100th_permutation_l296_296450


namespace negation_of_p_is_neg_p_l296_296543

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

-- Define the negation of p
def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p_is_neg_p : ¬p = neg_p := by
  -- The proof is omitted as per the instruction
  sorry

end negation_of_p_is_neg_p_l296_296543


namespace expected_value_of_X_l296_296464

noncomputable def probA : ℝ := 0.7
noncomputable def probB : ℝ := 0.8
noncomputable def probC : ℝ := 0.5

def prob_distribution : MeasureTheory.PMF ℕ :=
{
  support := {0, 1, 2, 3},
  toFun := λ x, match x with
    | 0 => (1 - probA) * (1 - probB) * (1 - probC)
    | 1 => probA * (1 - probB) * (1 - probC) + (1 - probA) * probB * (1 - probC) + (1 - probA) * (1 - probB) * probC
    | 2 => probA * probB * (1 - probC) + probA * (1 - probB) * probC + (1 - probA) * probB * probC
    | 3 => probA * probB * probC
    | _ => 0
}

noncomputable def expected_value : ℝ :=
  MeasureTheory.Integral (λ x, (x : ℝ)) prob_distribution

theorem expected_value_of_X : expected_value = 2 :=
by {
  sorry
}

end expected_value_of_X_l296_296464


namespace gcd_of_324_and_135_l296_296731

theorem gcd_of_324_and_135 : Nat.gcd 324 135 = 27 :=
by
  sorry

end gcd_of_324_and_135_l296_296731


namespace tan_sub_sin_eq_sq3_div2_l296_296774

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end tan_sub_sin_eq_sq3_div2_l296_296774


namespace base4_more_digits_than_base9_l296_296972

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l296_296972


namespace f_at_three_bounds_l296_296217

theorem f_at_three_bounds (a c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^2 - c)
  (h2 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h3 : -1 ≤ f 2 ∧ f 2 ≤ 5) : -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end f_at_three_bounds_l296_296217


namespace sin_product_ge_one_l296_296030

theorem sin_product_ge_one (x : ℝ) (n : ℤ) :
  (∀ α, |Real.sin α| ≤ 1) →
  ∀ x,
  (Real.sin x) * (Real.sin (1755 * x)) * (Real.sin (2011 * x)) ≥ 1 ↔
  ∃ n : ℤ, x = π / 2 + 2 * π * n := by {
    sorry
}

end sin_product_ge_one_l296_296030


namespace one_gallon_fills_one_cubic_foot_l296_296376

theorem one_gallon_fills_one_cubic_foot
  (total_water : ℕ)
  (drinking_cooking : ℕ)
  (shower_water : ℕ)
  (num_showers : ℕ)
  (pool_length : ℕ)
  (pool_width : ℕ)
  (pool_height : ℕ)
  (h_total_water : total_water = 1000)
  (h_drinking_cooking : drinking_cooking = 100)
  (h_shower_water : shower_water = 20)
  (h_num_showers : num_showers = 15)
  (h_pool_length : pool_length = 10)
  (h_pool_width : pool_width = 10)
  (h_pool_height : pool_height = 6) :
  (pool_length * pool_width * pool_height) / 
  (total_water - drinking_cooking - num_showers * shower_water) = 1 := by
  sorry

end one_gallon_fills_one_cubic_foot_l296_296376


namespace total_afternoon_evening_emails_l296_296428

-- Definitions based on conditions
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

-- Statement to be proven
theorem total_afternoon_evening_emails : afternoon_emails + evening_emails = 13 :=
by 
  sorry

end total_afternoon_evening_emails_l296_296428


namespace area_EFCD_l296_296130

-- Defining the geometrical setup and measurements of the trapezoid
variables (AB CD AD BC : ℝ) (h1 : AB = 10) (h2 : CD = 30) (h_altitude : ∃ h : ℝ, h = 18)

-- Defining the midpoints E and F of AD and BC respectively
variables (E F : ℝ) (h_E : E = AD / 2) (h_F : F = BC / 2)

-- Define the intersection of diagonals and the ratio condition
variables (AC BD G : ℝ) (h_ratio : ∃ r : ℝ, r = 1/2)

-- Proving the area of quadrilateral EFCD
theorem area_EFCD : EFCD_area = 225 :=
sorry

end area_EFCD_l296_296130


namespace lambda_equilateral_l296_296520

open Complex

noncomputable def find_lambda (ω : ℂ) (hω : abs ω = 3) : ℝ :=
  let λ := (1 + Real.sqrt 33) / 2
  λ

theorem lambda_equilateral (ω : ℂ) (hω : abs ω = 3) (λ : ℝ) (hλ : λ > 1) :
  ∃ ω, abs ω = 3 ∧ ω^2 ∈ Set.Icc 0 λ ∧ abs (λ * ω) = 3 ∧ (λ = (1 + Real.sqrt 33) / 2) :=
sorry

end lambda_equilateral_l296_296520


namespace total_students_in_halls_l296_296046

theorem total_students_in_halls :
  let S_g := 30
  let S_b := 2 * S_g
  let S_m := 3 / 5 * (S_g + S_b)
  S_g + S_b + S_m = 144 :=
by
  sorry

end total_students_in_halls_l296_296046


namespace blue_tile_probability_l296_296481

theorem blue_tile_probability :
  let total_tiles := 60
  let blue_tiles := (60 / 7).to_int + 1
  (blue_tiles / total_tiles : ℚ) = 3 / 20 :=
by
  let total_tiles := 60
  let blue_tiles := (60 / 7).to_int + 1
  have h1 : blue_tiles = 9 := by sorry
  have h2 : (blue_tiles : ℚ) / total_tiles = 9 / 60 := by sorry
  have h3 : 9 / 60 = 3 / 20 := by norm_num
  exact Eq.trans h2 h3

end blue_tile_probability_l296_296481


namespace ratio_of_surface_areas_of_spheres_l296_296165

theorem ratio_of_surface_areas_of_spheres (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) : 
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 9 := by
  sorry

end ratio_of_surface_areas_of_spheres_l296_296165


namespace number_of_dogs_l296_296557

theorem number_of_dogs
    (total_animals : ℕ)
    (dogs_ratio : ℕ) (bunnies_ratio : ℕ) (birds_ratio : ℕ)
    (h_total : total_animals = 816)
    (h_ratio : dogs_ratio = 3 ∧ bunnies_ratio = 9 ∧ birds_ratio = 11) :
    (total_animals / (dogs_ratio + bunnies_ratio + birds_ratio) * dogs_ratio = 105) :=
by
    sorry

end number_of_dogs_l296_296557


namespace parallel_vectors_l296_296818

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

theorem parallel_vectors (m : ℝ) (h : ∃ k : ℝ, vector_a = (k • vector_b m)) : m = -4 :=
by {
  sorry
}

end parallel_vectors_l296_296818


namespace remainder_3_pow_20_mod_5_l296_296894

theorem remainder_3_pow_20_mod_5 : (3 ^ 20) % 5 = 1 := by
  sorry

end remainder_3_pow_20_mod_5_l296_296894


namespace similar_triangles_y_value_l296_296071

theorem similar_triangles_y_value :
  ∀ (y : ℚ),
    (12 : ℚ) / y = (9 : ℚ) / 6 → 
    y = 8 :=
by
  intros y h
  sorry

end similar_triangles_y_value_l296_296071


namespace find_YW_in_triangle_l296_296702

theorem find_YW_in_triangle
  (X Y Z W : Type)
  (d_XZ d_YZ d_XW d_CW : ℝ)
  (h_XZ : d_XZ = 10)
  (h_YZ : d_YZ = 10)
  (h_XW : d_XW = 12)
  (h_CW : d_CW = 5) : 
  YW = 29 / 12 :=
sorry

end find_YW_in_triangle_l296_296702


namespace limit_Sn_div_Tn_l296_296102

open Set Real

def M_n (n : ℕ) : Set ℝ :=
  {x | ∃ (a : Fin n → Bool), (a (Fin.last n) = true) ∧ (x = Finset.univ.sum (λ k : Fin n, (if a k then 1 else 0) * 10^(-((k:ℕ)+1))))}

def T_n (n : ℕ) : ℕ :=
  2^(n-1)

def S_n (n : ℕ) : ℝ :=
  Finset.univ.sum (λ a : Fin n → Bool, (Finset.univ.sum (λ k : Fin n, (if a k then 1 else 0) * 10^(-((k:ℕ)+1)))))

theorem limit_Sn_div_Tn : tendsto (λ n, S_n n / T_n n) at_top (𝓝 (1/18)) :=
by
  sorry

end limit_Sn_div_Tn_l296_296102


namespace problem_T8_l296_296836

noncomputable def a : Nat → ℚ
| 0     => 1/2
| (n+1) => a n / (1 + 3 * a n)

noncomputable def T (n : Nat) : ℚ :=
  (Finset.range n).sum (λ i => 1 / a (i + 1))

theorem problem_T8 : T 8 = 100 :=
sorry

end problem_T8_l296_296836


namespace matrix_power_problem_l296_296243

def B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![4, 1], ![0, 2]]

theorem matrix_power_problem : B^15 - 3 * B^14 = ![![4, 3], ![0, -2]] :=
  by sorry

end matrix_power_problem_l296_296243


namespace ages_total_l296_296762

-- Define the variables and conditions
variables (A B C : ℕ)

-- State the conditions
def condition1 (B : ℕ) : Prop := B = 14
def condition2 (A B : ℕ) : Prop := A = B + 2
def condition3 (B C : ℕ) : Prop := B = 2 * C

-- The main theorem to prove
theorem ages_total (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 B C) : A + B + C = 37 :=
by
  sorry

end ages_total_l296_296762


namespace cindy_olaf_earnings_l296_296782
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end cindy_olaf_earnings_l296_296782


namespace fraction_of_cracked_pots_is_2_over_5_l296_296197

-- Definitions for the problem conditions
def total_pots : ℕ := 80
def price_per_pot : ℕ := 40
def total_revenue : ℕ := 1920

-- Statement to prove the fraction of cracked pots
theorem fraction_of_cracked_pots_is_2_over_5 
  (C : ℕ) 
  (h1 : (total_pots - C) * price_per_pot = total_revenue) : 
  C / total_pots = 2 / 5 :=
by
  sorry

end fraction_of_cracked_pots_is_2_over_5_l296_296197


namespace length_of_train_is_correct_l296_296921

-- Define the conditions with the provided data and given formulas.
def train_speed_kmh : ℝ := 63
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
def time_to_pass_tree : ℝ := 16
def train_length : ℝ := train_speed_ms * time_to_pass_tree

-- State the problem as a theorem in Lean 4.
theorem length_of_train_is_correct : train_length = 280 := by
  -- conditions are defined, need to calculate the length
  unfold train_length train_speed_ms
  -- specify the conversion calculation manually
  simp
  norm_num
  sorry

end length_of_train_is_correct_l296_296921


namespace hare_race_l296_296454

theorem hare_race :
  ∃ (total_jumps: ℕ) (final_jump_leg: String), total_jumps = 548 ∧ final_jump_leg = "right leg" :=
by
  sorry

end hare_race_l296_296454


namespace rattlesnakes_count_l296_296281

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end rattlesnakes_count_l296_296281


namespace num_tosses_l296_296350

theorem num_tosses (n : ℕ) (h : (1 - (7 / 8 : ℝ)^n) = 0.111328125) : n = 7 :=
by
  sorry

end num_tosses_l296_296350


namespace tank_C_capacity_is_80_percent_of_tank_B_l296_296449

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := 
  Real.pi * r^2 * h

theorem tank_C_capacity_is_80_percent_of_tank_B :
  ∀ (h_C c_C h_B c_B : ℝ), 
    h_C = 10 ∧ c_C = 8 ∧ h_B = 8 ∧ c_B = 10 → 
    (volume_of_cylinder (c_C / (2 * Real.pi)) h_C) / 
    (volume_of_cylinder (c_B / (2 * Real.pi)) h_B) * 100 = 80 := 
by 
  intros h_C c_C h_B c_B h_conditions
  obtain ⟨h_C_10, c_C_8, h_B_8, c_B_10⟩ := h_conditions
  sorry

end tank_C_capacity_is_80_percent_of_tank_B_l296_296449


namespace smallest_diameter_of_tablecloth_l296_296641

theorem smallest_diameter_of_tablecloth (a : ℝ) (h : a = 1) : ∃ d : ℝ, d = Real.sqrt 2 ∧ (∀ (x : ℝ), x < d → ¬(∀ (y : ℝ), (y^2 + y^2 = x^2) → y ≤ a)) :=
by 
  sorry

end smallest_diameter_of_tablecloth_l296_296641


namespace find_divisor_l296_296458

theorem find_divisor (d : ℕ) (N : ℕ) (a b : ℕ)
  (h1 : a = 9) (h2 : b = 79) (h3 : N = 7) :
  (∃ d, (∀ k : ℕ, a ≤ k*d ∧ k*d ≤ b → (k*d) % d = 0) ∧
   ∀ count : ℕ, count = (b / d) - ((a - 1) / d) → count = N) →
  d = 11 :=
by
  sorry

end find_divisor_l296_296458


namespace cost_of_5_pound_bag_is_2_l296_296761

-- Define costs of each type of bag
def cost_10_pound_bag : ℝ := 20.40
def cost_25_pound_bag : ℝ := 32.25
def least_total_cost : ℝ := 98.75

-- Define the total weight constraint
def min_weight : ℕ := 65
def max_weight : ℕ := 80
def weight_25_pound_bags : ℕ := 75

-- Given condition: The total purchase fulfils the condition of minimum cost
def total_cost_3_bags_25 : ℝ := 3 * cost_25_pound_bag
def remaining_cost : ℝ := least_total_cost - total_cost_3_bags_25

-- Prove the cost of the 5-pound bag is $2.00
theorem cost_of_5_pound_bag_is_2 :
  ∃ (cost_5_pound_bag : ℝ), cost_5_pound_bag = remaining_cost :=
by
  sorry

end cost_of_5_pound_bag_is_2_l296_296761


namespace partial_fraction_decomposition_l296_296504

noncomputable def A := 29 / 15
noncomputable def B := 13 / 12
noncomputable def C := 37 / 15

theorem partial_fraction_decomposition :
  let ABC := A * B * C;
  ABC = 13949 / 2700 :=
by
  sorry

end partial_fraction_decomposition_l296_296504


namespace total_coins_Zain_l296_296339

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l296_296339


namespace statues_painted_l296_296789

-- Definitions based on the conditions provided in the problem
def paint_remaining : ℚ := 1/2
def paint_per_statue : ℚ := 1/4

-- The theorem that answers the question
theorem statues_painted (h : paint_remaining = 1/2 ∧ paint_per_statue = 1/4) : 
  (paint_remaining / paint_per_statue) = 2 := 
sorry

end statues_painted_l296_296789


namespace fraction_of_men_married_is_two_thirds_l296_296475

-- Define the total number of faculty members
def total_faculty_members : ℕ := 100

-- Define the number of women as 70% of the faculty members
def women : ℕ := (70 * total_faculty_members) / 100

-- Define the number of men as 30% of the faculty members
def men : ℕ := (30 * total_faculty_members) / 100

-- Define the number of married faculty members as 40% of the faculty members
def married_faculty : ℕ := (40 * total_faculty_members) / 100

-- Define the number of single men as 1/3 of the men
def single_men : ℕ := men / 3

-- Define the number of married men as 2/3 of the men
def married_men : ℕ := (2 * men) / 3

-- Define the fraction of men who are married
def fraction_married_men : ℚ := married_men / men

-- The proof statement
theorem fraction_of_men_married_is_two_thirds : fraction_married_men = 2 / 3 := 
by sorry

end fraction_of_men_married_is_two_thirds_l296_296475


namespace Zain_coins_total_l296_296331

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l296_296331


namespace number_of_members_l296_296486

variable (n : ℕ)

-- Conditions
def each_member_contributes_n_cents : Prop := n * n = 64736

-- Theorem that relates to the number of members being 254
theorem number_of_members (h : each_member_contributes_n_cents n) : n = 254 :=
sorry

end number_of_members_l296_296486


namespace complex_number_purely_imaginary_l296_296101

variable {m : ℝ}

theorem complex_number_purely_imaginary (h1 : 2 * m^2 + m - 1 = 0) (h2 : -m^2 - 3 * m - 2 ≠ 0) : m = 1/2 := by
  sorry

end complex_number_purely_imaginary_l296_296101


namespace seventh_term_in_geometric_sequence_l296_296704

theorem seventh_term_in_geometric_sequence :
  ∃ r, (4 * r^8 = 2097152) ∧ (4 * r^6 = 1048576) :=
by
  sorry

end seventh_term_in_geometric_sequence_l296_296704


namespace natural_numbers_condition_l296_296793

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_numbers_condition (n : ℕ) (p1 p2 : ℕ)
  (hp1_prime : is_prime p1) (hp2_prime : is_prime p2)
  (hn : n = p1 ^ 2) (hn72 : n + 72 = p2 ^ 2) :
  n = 49 ∨ n = 289 :=
  sorry

end natural_numbers_condition_l296_296793


namespace root_not_less_than_a_l296_296850

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^3

theorem root_not_less_than_a (a b c x0 : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a * f b * f c < 0) (hx : f x0 = 0) : ¬ (x0 < a) :=
sorry

end root_not_less_than_a_l296_296850


namespace triangular_array_sum_digits_l296_296076

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2080) : 
  (N.digits 10).sum = 10 :=
sorry

end triangular_array_sum_digits_l296_296076


namespace symmetric_points_on_parabola_l296_296216

theorem symmetric_points_on_parabola
  (x1 x2 : ℝ)
  (m : ℝ)
  (h1 : 2 * x1 * x1 = 2 * x2 * x2)
  (h2 : 2 * x1 * x1 = 2 * x2 * x2 + m)
  (h3 : x1 * x2 = -1 / 2)
  (h4 : x1 + x2 = -1 / 2)
  : m = 3 / 2 :=
sorry

end symmetric_points_on_parabola_l296_296216


namespace louie_monthly_payment_l296_296578

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

noncomputable def monthly_payment (P : ℝ) (r : ℝ) (n t : ℕ) (months : ℕ) : ℝ :=
  (compound_interest P r n t) / months

theorem louie_monthly_payment :
  monthly_payment 1000 0.10 1 3 3 ≈ 444 :=
sorry

end louie_monthly_payment_l296_296578


namespace problem_statement_l296_296037

-- Defining the properties of the function
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def symmetric_about_2 (f : ℝ → ℝ) := ∀ x, f (2 + (2 - x)) = f x

-- Given the function f, even function, and symmetric about line x = 2,
-- and given that f(3) = 3, we need to prove f(-1) = 3.
theorem problem_statement (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : symmetric_about_2 f) 
  (h3 : f 3 = 3) : 
  f (-1) = 3 := 
sorry

end problem_statement_l296_296037


namespace vertex_of_parabola_l296_296269

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l296_296269


namespace game_last_at_most_moves_l296_296608

theorem game_last_at_most_moves
  (n : Nat)
  (positions : Fin n → Fin (n + 1))
  (cards : Fin n → Fin (n + 1))
  (move : (k l : Fin n) → (h1 : k < l) → (h2 : k < cards k) → (positions l = cards k) → Fin n)
  : True :=
sorry

end game_last_at_most_moves_l296_296608


namespace correct_statement_is_A_l296_296903

theorem correct_statement_is_A : 
  (∀ x : ℝ, 0 ≤ x → abs x = x) ∧
  ¬ (∀ x : ℝ, x ≤ 0 → -x = x) ∧
  ¬ (∀ x : ℝ, (x ≠ 0 ∧ x⁻¹ = x) → (x = 1 ∨ x = -1 ∨ x = 0)) ∧
  ¬ (∀ x y : ℝ, x < 0 ∧ y < 0 → abs x < abs y → x < y) :=
by
  sorry

end correct_statement_is_A_l296_296903


namespace fraction_is_five_sixths_l296_296483

-- Define the conditions as given in the problem
def number : ℝ := -72.0
def target_value : ℝ := -60

-- The statement we aim to prove
theorem fraction_is_five_sixths (f : ℝ) (h : f * number = target_value) : f = 5/6 :=
  sorry

end fraction_is_five_sixths_l296_296483


namespace petya_vasya_meet_at_lantern_64_l296_296643

-- Define the total number of lanterns and intervals
def total_lanterns : ℕ := 100
def total_intervals : ℕ := total_lanterns - 1

-- Define the positions of Petya and Vasya at a given time
def petya_initial : ℕ := 1
def vasya_initial : ℕ := 100
def petya_position : ℕ := 22
def vasya_position : ℕ := 88

-- Define the number of intervals covered by Petya and Vasya
def petya_intervals_covered : ℕ := petya_position - petya_initial
def vasya_intervals_covered : ℕ := vasya_initial - vasya_position

-- Define the combined intervals covered
def combined_intervals_covered : ℕ := petya_intervals_covered + vasya_intervals_covered

-- Define the interval after which Petya and Vasya will meet
def meeting_intervals : ℕ := total_intervals - combined_intervals_covered

-- Define the final meeting point according to Petya's travel
def meeting_lantern : ℕ := petya_initial + (meeting_intervals / 2)

theorem petya_vasya_meet_at_lantern_64 : meeting_lantern = 64 := by {
  -- Proof goes here
  sorry
}

end petya_vasya_meet_at_lantern_64_l296_296643


namespace evaluate_expression_l296_296513

theorem evaluate_expression (x : ℝ) (h1 : x^4 + 2 * x + 2 ≠ 0)
    (h2 : x^4 - 2 * x + 2 ≠ 0) :
    ( ( ( (x + 2) ^ 3 * (x^3 - 2 * x + 2) ^ 3 ) / ( ( x^4 + 2 * x + 2) ) ^ 3 ) ^ 3 * 
      ( ( (x - 2) ^ 3 * ( x^3 + 2 * x + 2 ) ^ 3 ) / ( ( x^4 - 2 * x + 2 ) ) ^ 3 ) ^ 3 ) = 1 :=
by
  sorry

end evaluate_expression_l296_296513


namespace proportion_of_boys_correct_l296_296166

noncomputable def proportion_of_boys : ℚ :=
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let expected_children := 3 -- (2 boys and 1 girl)
  let expected_boys := 2 -- Expected number of boys in each family
  
  expected_boys / expected_children

theorem proportion_of_boys_correct : proportion_of_boys = 2 / 3 := by
  sorry

end proportion_of_boys_correct_l296_296166


namespace arrow_in_48th_position_l296_296419

def arrow_sequence : List (String) := ["→", "↑", "↓", "←", "↘"]

theorem arrow_in_48th_position :
  arrow_sequence.get? ((48 % 5) - 1) = some "↓" :=
by
  norm_num
  sorry

end arrow_in_48th_position_l296_296419


namespace contradiction_assumption_l296_296881

theorem contradiction_assumption (a b : ℝ) (h : a ≤ 2 ∧ b ≤ 2) : (a > 2 ∨ b > 2) -> false :=
by
  sorry

end contradiction_assumption_l296_296881


namespace smallest_positive_debt_resolvable_l296_296171

theorem smallest_positive_debt_resolvable :
  ∃ (p g : ℤ), 400 * p + 280 * g = 800 :=
sorry

end smallest_positive_debt_resolvable_l296_296171


namespace inequality_proof_l296_296519

theorem inequality_proof
  (a b c d e f : ℝ)
  (h1 : 1 ≤ a)
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d)
  (h5 : d ≤ e)
  (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := 
by 
  sorry

end inequality_proof_l296_296519


namespace vertex_of_quadratic_l296_296453

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- State the theorem for vertex coordinates
theorem vertex_of_quadratic :
  (∀ x : ℝ, quadratic_function (- (-6) / (2 * -3)) = quadratic_function 1)
  → (1, quadratic_function 1) = (1, 8) :=
by
  intros h
  sorry

end vertex_of_quadratic_l296_296453


namespace range_of_t_in_region_l296_296696

theorem range_of_t_in_region : (t : ℝ) → ((1 - t + 1 > 0) → t < 2) :=
by
  intro t
  intro h
  sorry

end range_of_t_in_region_l296_296696


namespace total_votes_l296_296630

theorem total_votes (V : ℝ) (h1 : 0.70 * V = V - 240) (h2 : 0.30 * V = 240) : V = 800 :=
by
  sorry

end total_votes_l296_296630


namespace division_of_repeating_decimal_l296_296320

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l296_296320


namespace find_reciprocal_sum_l296_296042

theorem find_reciprocal_sum
  (m n : ℕ)
  (h_sum : m + n = 72)
  (h_hcf : Nat.gcd m n = 6)
  (h_lcm : Nat.lcm m n = 210) :
  (1 / (m : ℚ)) + (1 / (n : ℚ)) = 6 / 105 :=
by
  sorry

end find_reciprocal_sum_l296_296042


namespace M_is_range_of_sq_function_l296_296684

noncomputable theory

def M : set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem M_is_range_of_sq_function : M = {y | ∃ x : ℝ, y = x^2} :=
by
  sorry

end M_is_range_of_sq_function_l296_296684


namespace tan_five_pi_over_four_l296_296659

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l296_296659


namespace solve_system_of_equations_l296_296593

variable (a b c : Real)

def K : Real := a * b * c + a^2 * c + c^2 * b + b^2 * a

theorem solve_system_of_equations 
    (h₁ : (a + b) * (a - b) * (b + c) * (b - c) * (c + a) * (c - a) ≠ 0)
    (h₂ : K a b c ≠ 0) :
    ∃ (x y z : Real), 
    x = b^2 - c^2 ∧
    y = c^2 - a^2 ∧
    z = a^2 - b^2 ∧
    (x / (b + c) + y / (c - a) = a + b) ∧
    (y / (c + a) + z / (a - b) = b + c) ∧
    (z / (a + b) + x / (b - c) = c + a) :=
by
  sorry

end solve_system_of_equations_l296_296593


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l296_296902

noncomputable def repeating_decimal_fraction (x : ℚ) : ℚ :=
  if x = 0.345345345... then 115 / 333 else sorry

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.345345345... in 
  let fraction := repeating_decimal_fraction x in
  (fraction.num + fraction.denom) = 448 :=
by {
  sorry
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l296_296902


namespace total_seconds_eq_250200_l296_296360

def bianca_hours : ℝ := 12.5
def celeste_hours : ℝ := 2 * bianca_hours
def mcclain_hours : ℝ := celeste_hours - 8.5
def omar_hours : ℝ := bianca_hours + 3

def total_hours : ℝ := bianca_hours + celeste_hours + mcclain_hours + omar_hours
def hour_to_seconds : ℝ := 3600
def total_seconds : ℝ := total_hours * hour_to_seconds

theorem total_seconds_eq_250200 : total_seconds = 250200 := by
  sorry

end total_seconds_eq_250200_l296_296360


namespace chords_intersecting_theorem_l296_296395

noncomputable def intersecting_chords_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) : ℝ :=
  sorry

theorem chords_intersecting_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) :
  (P - A) * (P - B) = (P - C) * (P - D) :=
by sorry

end chords_intersecting_theorem_l296_296395


namespace thirty_sixty_ninety_triangle_area_l296_296162

theorem thirty_sixty_ninety_triangle_area (hypotenuse : ℝ) (angle : ℝ) (area : ℝ)
  (h_hypotenuse : hypotenuse = 12)
  (h_angle : angle = 30)
  (h_area : area = 18 * Real.sqrt 3) :
  ∃ (base height : ℝ), 
    base = hypotenuse / 2 ∧ 
    height = (hypotenuse / 2) * Real.sqrt 3 ∧ 
    area = (1 / 2) * base * height :=
by {
  sorry
}

end thirty_sixty_ninety_triangle_area_l296_296162


namespace smallest_even_sum_l296_296646

theorem smallest_even_sum :
  ∃ (a b c : Int), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ b ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ c ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ (a + b + c) % 2 = 0 ∧ (a + b + c) = 14 := sorry

end smallest_even_sum_l296_296646


namespace total_flowers_l296_296149

noncomputable def yellow_flowers : ℕ := 10
noncomputable def purple_flowers : ℕ := yellow_flowers + (80 * yellow_flowers) / 100
noncomputable def green_flowers : ℕ := (25 * (yellow_flowers + purple_flowers)) / 100
noncomputable def red_flowers : ℕ := (35 * (yellow_flowers + purple_flowers + green_flowers)) / 100

theorem total_flowers :
  yellow_flowers + purple_flowers + green_flowers + red_flowers = 47 :=
by
  -- Insert proof here
  sorry

end total_flowers_l296_296149


namespace find_ratio_l296_296145

open Real

theorem find_ratio (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / sqrt 7 :=
by
  sorry

end find_ratio_l296_296145


namespace cider_pints_produced_l296_296691

/-- Define the conditions as constants and parameters -/
def golden_apples_per_pint := 20
def pink_apples_per_pint := 40
def farmhands := 6
def apples_per_hour_per_farmhand := 240
def hours_worked := 5
def golden_to_pink_ratio (golden: ℕ) (pink: ℕ) := golden = 2 * pink

/-- Main theorem statement -/
theorem cider_pints_produced : 
  ∀ (golden picked_apples : ℕ),
  let total_apples := picked_apples * farmhands * hours_worked in
  let total_pints := total_apples / (golden_apples_per_pint + pink_apples_per_pint) in
  total_apples = 7200 → 
  total_pints = 120 :=
by
  intros golden picked_apples total_apples total_pints h_tot_apples
  sorry

end cider_pints_produced_l296_296691


namespace kuzya_probability_distance_2h_l296_296208

noncomputable def probability_kuzya_at_distance_2h : ℚ :=
  let h := 1 -- treat each jump length as 1 for simplicity
  let events := finset.range 6 -- number of jumps from 2 to 5
  let prob_at_2h (n : ℕ) : ℚ := 
    if n < 2 then 0
    else if n = 2 then 1/2
    else if n = 3 then 3/8
    else if n = 4 then 3/8
    else if n = 5 then 15/32
    else 0
  (events.sum prob_at_2h) / events.sum (λ n, 1)

theorem kuzya_probability_distance_2h :
  probability_kuzya_at_distance_2h = 5 / 8 :=
sorry

end kuzya_probability_distance_2h_l296_296208


namespace intersection_point_of_lines_l296_296687

theorem intersection_point_of_lines : 
  ∃ x y : ℝ, (3 * x + 4 * y - 2 = 0) ∧ (2 * x + y + 2 = 0) ∧ (x = -2) ∧ (y = 2) := 
by 
  sorry

end intersection_point_of_lines_l296_296687


namespace minimize_S_n_at_7_l296_296397

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 2 - a 1

def conditions (a : ℕ → ℤ) : Prop :=
arithmetic_sequence a ∧ a 2 = -11 ∧ (a 5 + a 9 = -2)

-- Define the sum of first n terms of the sequence
def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Define the minimum S_n and that it occurs at n = 7
theorem minimize_S_n_at_7 (a : ℕ → ℤ) (n : ℕ) (h : conditions a) :
  ∀ m, S a m ≥ S a 7 := sorry

end minimize_S_n_at_7_l296_296397


namespace houses_with_two_car_garage_l296_296829

theorem houses_with_two_car_garage
  (T P GP N G : ℕ)
  (hT : T = 90)
  (hP : P = 40)
  (hGP : GP = 35)
  (hN : N = 35)
  (hFormula : G + P - GP = T - N) :
  G = 50 :=
by
  rw [hT, hP, hGP, hN] at hFormula
  simp at hFormula
  exact hFormula

end houses_with_two_car_garage_l296_296829


namespace total_mission_days_l296_296022

variable (initial_days_first_mission : ℝ := 5)
variable (percentage_longer : ℝ := 0.60)
variable (days_second_mission : ℝ := 3)

theorem total_mission_days : 
  let days_first_mission_extra := initial_days_first_mission * percentage_longer
  let total_days_first_mission := initial_days_first_mission + days_first_mission_extra
  (total_days_first_mission + days_second_mission) = 11 := by
  sorry

end total_mission_days_l296_296022


namespace Eight_div_by_repeating_decimal_0_3_l296_296316

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l296_296316


namespace collinear_SGD_l296_296845

noncomputable def acute_triangle (A B C : ℝ × ℝ) (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A < 90 ∧ angle_B < 90 ∧ angle_C < 90

def intersection_of_medians (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let M1 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let M2 := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let M3 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def foot_of_altitude (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let k := ((B.2 - C.2) * (B.2 * C.1 - B.1 * C.2 - (A.1 * (B.1 - C.1) + A.2 * (B.2 - C.2)))) /
           ((B.1 - C.1) * (B.1 - C.1) + (B.2 - C.2) * (B.2 - C.2))
  (A.1 - (A.1 - B.1) * k, A.2 - (A.2 - B.2) * k)

def parallel_line_pt (A B C : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  let slope := (C.2 - B.2) / (C.1 - B.1)
  (A.1 + k, A.2 + slope * k)

axiom circumcircle_exists (A B C : ℝ × ℝ) : ∃ (O : ℝ × ℝ) (R : ℝ), true

theorem collinear_SGD (A B C : ℝ × ℝ)
  (G := intersection_of_medians A B C)
  (D := foot_of_altitude A B C)
  (circ := circumcircle_exists A B C)
  (S := parallel_line_pt A B C 1):
  acute_triangle A B C
  → collinear [S, G, D] :=
sorry

end collinear_SGD_l296_296845


namespace tan_sin_difference_l296_296776

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end tan_sin_difference_l296_296776


namespace student_calculation_no_error_l296_296074

theorem student_calculation_no_error :
  let correct_result : ℚ := (7 * 4) / (5 / 3)
  let student_result : ℚ := (7 * 4) * (3 / 5)
  correct_result = student_result → 0 = 0 := 
by
  intros correct_result student_result h
  sorry

end student_calculation_no_error_l296_296074


namespace divide_by_repeating_decimal_l296_296310

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l296_296310


namespace quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l296_296527

-- Condition for the quadratic equation having two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)) ↔ (k ≥ 3 / 2) :=
sorry

-- Condition linking the roots of the equation and the properties of the rectangle
theorem roots_form_rectangle_with_diagonal (k : ℝ) 
  (h : k ≥ 3 / 2) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)
  ∧ (x1^2 + x2^2 = 5)) ↔ (k = 2) :=
sorry

end quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l296_296527


namespace y_intercept_probability_l296_296961

theorem y_intercept_probability (b : ℝ) (hb : b ∈ Set.Icc (-2 : ℝ) 3 ) :
  (∃ P : ℚ, P = (2 / 5)) := 
by 
  sorry

end y_intercept_probability_l296_296961


namespace monotone_decreasing_f_find_a_value_l296_296541

-- Condition declarations
variables (a b : ℝ) (h_a_pos : a > 0) (max_val min_val : ℝ)
noncomputable def f (x : ℝ) := x + (a / x) + b

-- Problem 1: Prove that f is monotonically decreasing in (0, sqrt(a)]
theorem monotone_decreasing_f : 
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a → f a b x1 > f a b x2) :=
sorry

-- Conditions for Problem 2
variable (hf_inc : ∀ x1 x2 : ℝ, Real.sqrt a ≤ x1 ∧ x1 < x2 → f a b x1 < f a b x2)
variable (h_max : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≤ 5)
variable (h_min : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≥ 3)

-- Problem 2: Find the value of a
theorem find_a_value : a = 6 :=
sorry

end monotone_decreasing_f_find_a_value_l296_296541


namespace bags_of_cookies_l296_296622

theorem bags_of_cookies (bags : ℕ) (cookies_total candies_total : ℕ) 
    (h1 : bags = 14) (h2 : cookies_total = 28) (h3 : candies_total = 86) :
    bags = 14 :=
by
  exact h1

end bags_of_cookies_l296_296622


namespace point_A_in_fourth_quadrant_l296_296127

-- Defining the coordinates of point A
def x_A : ℝ := 2
def y_A : ℝ := -3

-- Defining the property of the quadrant
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Proposition stating point A is in the fourth quadrant
theorem point_A_in_fourth_quadrant : in_fourth_quadrant x_A y_A :=
by
  sorry

end point_A_in_fourth_quadrant_l296_296127


namespace find_b_l296_296825

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 315 * b) : b = 7 :=
by
  -- The actual proof would go here
  sorry

end find_b_l296_296825


namespace find_k_l296_296552

theorem find_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) (hk : k ≠ 0) : k = 5 :=
sorry

end find_k_l296_296552


namespace roots_are_simplified_sqrt_form_l296_296794

theorem roots_are_simplified_sqrt_form : 
  ∃ m p n : ℕ, gcd m p = 1 ∧ gcd p n = 1 ∧ gcd m n = 1 ∧
    (∀ x : ℝ, (3 * x^2 - 8 * x + 1 = 0) ↔ 
    (x = (m : ℝ) + (Real.sqrt n)/(p : ℝ) ∨ x = (m : ℝ) - (Real.sqrt n)/(p : ℝ))) ∧
    n = 13 :=
by
  sorry

end roots_are_simplified_sqrt_form_l296_296794


namespace side_length_of_square_l296_296040

theorem side_length_of_square (P : ℕ) (h1 : P = 28) (h2 : P = 4 * s) : s = 7 :=
  by sorry

end side_length_of_square_l296_296040


namespace largest_fraction_l296_296017

variable {a b c d e f g h : ℝ}
variable {w x y z : ℝ}

/-- Given real numbers w, x, y, z such that w < x < y < z,
    the fraction z/w represents the largest value among the given fractions. -/
theorem largest_fraction (hwx : w < x) (hxy : x < y) (hyz : y < z) :
  (z / w) > (x / w) ∧ (z / w) > (y / x) ∧ (z / w) > (y / w) ∧ (z / w) > (z / x) :=
by
  sorry

end largest_fraction_l296_296017


namespace A_speed_is_10_l296_296359

noncomputable def A_walking_speed (v t : ℝ) := 
  v * (t + 7) = 140 ∧ v * (t + 7) = 20 * t

theorem A_speed_is_10 (v t : ℝ) 
  (h1 : v * (t + 7) = 140)
  (h2 : v * (t + 7) = 20 * t) :
  v = 10 :=
sorry

end A_speed_is_10_l296_296359


namespace zain_coin_total_l296_296336

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l296_296336


namespace pet_store_initial_house_cats_l296_296918

theorem pet_store_initial_house_cats
    (H : ℕ)
    (h1 : 13 + H - 10 = 8) :
    H = 5 :=
by
  sorry

end pet_store_initial_house_cats_l296_296918


namespace same_bill_at_300_minutes_l296_296154

def monthlyBillA (x : ℕ) : ℝ := 15 + 0.1 * x
def monthlyBillB (x : ℕ) : ℝ := 0.15 * x

theorem same_bill_at_300_minutes : monthlyBillA 300 = monthlyBillB 300 := 
by
  sorry

end same_bill_at_300_minutes_l296_296154


namespace find_x_l296_296672

theorem find_x (x : ℕ) (h₁ : 3 * (Nat.factorial 8) / (Nat.factorial (8 - x)) = 4 * (Nat.factorial 9) / (Nat.factorial (9 - (x - 1)))) : x = 6 :=
sorry

end find_x_l296_296672


namespace find_b_l296_296024

theorem find_b (a b c : ℝ) (k₁ k₂ k₃ : ℤ) :
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  (a + c) / 2 = 44 ∧
  a + b = 5 * k₁ ∧
  b + c = 5 * k₂ ∧
  a + c = 5 * k₃
  → b = 40 :=
by {
  sorry
}

end find_b_l296_296024


namespace neg_p_l296_296954

open Real

variable {f : ℝ → ℝ}

theorem neg_p :
  (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end neg_p_l296_296954


namespace number_of_six_digit_numbers_formable_by_1_2_3_4_l296_296640

theorem number_of_six_digit_numbers_formable_by_1_2_3_4
  (digits : Finset ℕ := {1, 2, 3, 4})
  (pairs_count : ℕ := 2)
  (non_adjacent_pair : ℕ := 1)
  (adjacent_pair : ℕ := 1)
  (six_digit_numbers : ℕ := 432) :
  ∃ (n : ℕ), n = 432 :=
by
  -- Proof will go here
  sorry

end number_of_six_digit_numbers_formable_by_1_2_3_4_l296_296640


namespace area_change_l296_296534

variable (p k : ℝ)
variable {N : ℝ}

theorem area_change (hN : N = 1/2 * (p * p)) (q : ℝ) (hq : q = k * p) :
  q = k * p -> (1/2 * (q * q) = k^2 * N) :=
by
  intros
  sorry

end area_change_l296_296534


namespace initial_cookies_l296_296079

variable (andys_cookies : ℕ)

def total_cookies_andy_ate : ℕ := 3
def total_cookies_brother_ate : ℕ := 5

def arithmetic_sequence_sum (n : ℕ) : ℕ := n * (2 * n - 1)

def total_cookies_team_ate : ℕ := arithmetic_sequence_sum 8

theorem initial_cookies :
  andys_cookies = total_cookies_andy_ate + total_cookies_brother_ate + total_cookies_team_ate :=
  by
    -- Here the missing proof would go
    sorry

end initial_cookies_l296_296079


namespace part1_part2_l296_296228

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a * sin A * sin B + b * cos A^2 = 4 / 3 * a)
variable (h2 : c^2 = a^2 + (1 / 4) * b^2)

theorem part1 : b = 4 / 3 * a := by sorry

theorem part2 : C = π / 3 := by sorry

end part1_part2_l296_296228


namespace min_frac_a_n_over_n_l296_296817

open Nat

def a : ℕ → ℕ
| 0     => 60
| (n+1) => a n + 2 * n

theorem min_frac_a_n_over_n : ∃ n : ℕ, n > 0 ∧ (a n / n = (29 / 2) ∧ ∀ m : ℕ, m > 0 → a m / m ≥ (29 / 2)) :=
by
  sorry

end min_frac_a_n_over_n_l296_296817


namespace positive_integers_condition_l296_296389

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end positive_integers_condition_l296_296389


namespace mean_of_remaining_three_numbers_l296_296451

theorem mean_of_remaining_three_numbers 
    (a b c d : ℝ)
    (h₁ : (a + b + c + d) / 4 = 92)
    (h₂ : d = 120)
    (h₃ : b = 60) : 
    (a + b + c) / 3 = 82.6666666666 := 
by 
    -- This state suggests adding the constraints added so far for the proof:
    sorry

end mean_of_remaining_three_numbers_l296_296451


namespace find_inheritance_amount_l296_296137

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end find_inheritance_amount_l296_296137


namespace equivalent_fraction_power_multiplication_l296_296323

theorem equivalent_fraction_power_multiplication : 
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (2 / 5) = (128 / 3645) := 
by 
  sorry

end equivalent_fraction_power_multiplication_l296_296323


namespace jakes_present_weight_l296_296180

theorem jakes_present_weight:
  ∃ J S : ℕ, J - 15 = 2 * S ∧ J + S = 132 ∧ J = 93 :=
by
  sorry

end jakes_present_weight_l296_296180


namespace expression_exists_l296_296120

theorem expression_exists (a b : ℤ) (h : 5 * a = 3125) (hb : 5 * b = 25) : b = 5 := by
  sorry

end expression_exists_l296_296120


namespace prime_prod_identity_l296_296809

theorem prime_prod_identity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 3 * p + 7 * q = 41) : (p + 1) * (q - 1) = 12 := 
by 
  sorry

end prime_prod_identity_l296_296809


namespace tan_5pi_over_4_l296_296663

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l296_296663


namespace intersection_of_sets_union_of_complement_and_set_l296_296999

def set1 := { x : ℝ | -1 < x ∧ x < 2 }
def set2 := { x : ℝ | x > 0 }
def complement_set2 := { x : ℝ | x ≤ 0 }
def intersection_set := { x : ℝ | 0 < x ∧ x < 2 }
def union_set := { x : ℝ | x < 2 }

theorem intersection_of_sets : 
  { x : ℝ | x ∈ set1 ∧ x ∈ set2 } = intersection_set := 
by 
  sorry

theorem union_of_complement_and_set : 
  { x : ℝ | x ∈ complement_set2 ∨ x ∈ set1 } = union_set := 
by 
  sorry

end intersection_of_sets_union_of_complement_and_set_l296_296999


namespace max_value_of_a_max_value_reached_l296_296531

theorem max_value_of_a (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 :=
by
  sorry

theorem max_value_reached (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  ∃ a, a = Real.sqrt 6 / 3 :=
by
  sorry

end max_value_of_a_max_value_reached_l296_296531


namespace k5_possibility_l296_296479

noncomputable def possible_k5 : Prop :=
  ∃ (intersections : Fin 5 → Fin 5 × Fin 10), 
    ∀ i j : Fin 5, i ≠ j → intersections i ≠ intersections j

theorem k5_possibility : possible_k5 := 
by
  sorry

end k5_possibility_l296_296479


namespace base_digit_difference_l296_296967

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l296_296967


namespace math_problem_l296_296621

theorem math_problem (a : ℝ) (h : a = 1/3) : (3 * a⁻¹ + 2 / 3 * a⁻¹) / a = 33 := by
  sorry

end math_problem_l296_296621


namespace mod_graph_sum_l296_296369

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end mod_graph_sum_l296_296369


namespace calculate_perimeter_l296_296920

noncomputable def length_square := 8
noncomputable def breadth_square := 8 -- since it's a square, length and breadth are the same
noncomputable def length_rectangle := 8
noncomputable def breadth_rectangle := 4

noncomputable def combined_length := length_square + length_rectangle
noncomputable def combined_breadth := breadth_square 

noncomputable def perimeter := 2 * (combined_length + combined_breadth)

theorem calculate_perimeter : 
  length_square = 8 ∧ 
  breadth_square = 8 ∧ 
  length_rectangle = 8 ∧ 
  breadth_rectangle = 4 ∧ 
  perimeter = 48 := 
by 
  sorry

end calculate_perimeter_l296_296920


namespace max_sum_cos_l296_296115

theorem max_sum_cos (a b c : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x) ≥ -1) : a + b + c ≤ 3 := by
  sorry

end max_sum_cos_l296_296115


namespace inverse_proposition_true_l296_296883

-- Define a rectangle and a square
structure Rectangle where
  length : ℝ
  width  : ℝ

def is_square (r : Rectangle) : Prop :=
  r.length = r.width ∧ r.length > 0 ∧ r.width > 0

-- Define the condition that a rectangle with equal adjacent sides is a square
def rectangle_with_equal_adjacent_sides_is_square : Prop :=
  ∀ r : Rectangle, r.length = r.width → is_square r

-- Define the inverse proposition that a square is a rectangle with equal adjacent sides
def square_is_rectangle_with_equal_adjacent_sides : Prop :=
  ∀ r : Rectangle, is_square r → r.length = r.width

-- The proof statement
theorem inverse_proposition_true :
  rectangle_with_equal_adjacent_sides_is_square → square_is_rectangle_with_equal_adjacent_sides :=
by
  sorry

end inverse_proposition_true_l296_296883


namespace distinct_complex_numbers_count_l296_296408

theorem distinct_complex_numbers_count :
  let real_choices := 10
  let imag_choices := 9
  let distinct_complex_numbers := real_choices * imag_choices
  distinct_complex_numbers = 90 :=
by
  sorry

end distinct_complex_numbers_count_l296_296408


namespace expression_greater_than_m_l296_296555

theorem expression_greater_than_m (m : ℚ) : m + 2 > m :=
by sorry

end expression_greater_than_m_l296_296555


namespace variance_is_0_02_l296_296914

def data_points : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_is_0_02 : variance data_points = 0.02 :=
by
  sorry

end variance_is_0_02_l296_296914


namespace alpha_beta_roots_l296_296099

theorem alpha_beta_roots (α β : ℝ) (hαβ1 : α^2 + α - 1 = 0) (hαβ2 : β^2 + β - 1 = 0) (h_sum : α + β = -1) :
  α^4 - 3 * β = 5 :=
by
  sorry

end alpha_beta_roots_l296_296099


namespace min_omega_value_l296_296224

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega_value
  (ω : ℝ) (φ : ℝ)
  (hω : ω > 0)
  (h1 : f (π / 3) ω φ = 0)
  (h2 : f (π / 2) ω φ = 2) :
  ω = 3 :=
sorry

end min_omega_value_l296_296224


namespace find_k_l296_296548

def f (x : ℤ) : ℤ := 3*x^2 - 2*x + 4
def g (x : ℤ) (k : ℤ) : ℤ := x^2 - k * x - 6

theorem find_k : 
  ∃ k : ℤ, f 10 - g 10 k = 10 ∧ k = -18 :=
by 
  sorry

end find_k_l296_296548


namespace carol_twice_as_cathy_l296_296254

-- Define variables for the number of cars each person owns
variables (C L S Ca x : ℕ)

-- Define conditions based on the problem statement
def lindsey_cars := L = C + 4
def susan_cars := S = Ca - 2
def carol_cars := Ca = 2 * x
def total_cars := C + L + S + Ca = 32
def cathy_cars := C = 5

-- State the theorem to prove
theorem carol_twice_as_cathy : 
  lindsey_cars C L ∧ 
  susan_cars S Ca ∧ 
  carol_cars Ca x ∧ 
  total_cars C L S Ca ∧ 
  cathy_cars C
  → x = 5 :=
by
  sorry

end carol_twice_as_cathy_l296_296254


namespace find_m_l296_296651

theorem find_m 
  (m : ℤ) 
  (h1 : ∀ x y : ℤ, -3 * x + y = m → 2 * x + y = 28 → x = -6) : 
  m = 58 :=
by 
  sorry

end find_m_l296_296651


namespace sales_maximized_sales_amount_greater_l296_296912

-- Part 1
theorem sales_maximized (a : ℝ) (ha1 : 1/3 ≤ a) (ha2 : a < 1) :
  ∃ x : ℝ, x = 5 * (1 - a) / a := sorry

-- Part 2
theorem sales_amount_greater (x : ℝ) :
  (2 * x / 3 ≤ 3 * 10) ∧ (20 / 3 - 2 * x / 3 > 0) ↔ 0 < x ∧ x < 5 := sorry

end sales_maximized_sales_amount_greater_l296_296912


namespace tree_height_increase_l296_296327

theorem tree_height_increase
  (initial_height : ℝ)
  (height_increase : ℝ)
  (h6 : ℝ) :
  initial_height = 4 →
  (0 ≤ height_increase) →
  height_increase * 6 + initial_height = (height_increase * 4 + initial_height) + 1 / 7 * (height_increase * 4 + initial_height) →
  height_increase = 2 / 5 :=
by
  intro h_initial h_nonneg h_eq
  sorry

end tree_height_increase_l296_296327


namespace coeff_of_x_square_l296_296123

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem coeff_of_x_square :
  (binom 8 3 = 56) ∧ (8 - 2 * 3 = 2) :=
sorry

end coeff_of_x_square_l296_296123


namespace sqrt7_problem_l296_296556

theorem sqrt7_problem (x y : ℝ) (h1 : 2 < Real.sqrt 7) (h2 : Real.sqrt 7 < 3) (hx : x = 2) (hy : y = Real.sqrt 7 - 2) :
  (x + Real.sqrt 7) * y = 3 :=
by
  sorry

end sqrt7_problem_l296_296556


namespace people_in_room_l296_296516

theorem people_in_room (people chairs : ℕ) (h1 : 5 / 8 * people = 4 / 5 * chairs)
  (h2 : chairs = 5 + 4 / 5 * chairs) : people = 32 :=
by
  sorry

end people_in_room_l296_296516


namespace increasing_iff_range_a_three_distinct_real_roots_l296_296959

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 2 * a then x^2 + (2 - 2 * a) * x else - x^2 + (2 + 2 * a) * x

theorem increasing_iff_range_a (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem three_distinct_real_roots (a t : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2)
  (h_roots : ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
                           f a x₁ = t * f a (2 * a) ∧
                           f a x₂ = t * f a (2 * a) ∧
                           f a x₃ = t * f a (2 * a)) :
  1 < t ∧ t < 9 / 8 :=
sorry

end increasing_iff_range_a_three_distinct_real_roots_l296_296959


namespace greatest_whole_number_solution_l296_296669

theorem greatest_whole_number_solution : ∃ k : ℕ, 5 * k - 4 < 3 - 2 * k ∧ ∀ m : ℕ, (m > k → ¬ (5 * m - 4 < 3 - 2 * m)) :=
by
  exists 0
  split
  {
    rw [nat.cast_zero, mul_zero, sub_zero, zero_sub, neg_lt, neg_zero]
    exact zero_lt_three
  }
  {
    intros m hm
    rw [nat.not_lt]
    exact nat.le_of_lt_succ hm
  }

end greatest_whole_number_solution_l296_296669


namespace count_not_divisible_by_5_or_7_l296_296510

theorem count_not_divisible_by_5_or_7 :
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  total_numbers - count_divisible_by_5_or_7 = 343 :=
by
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  have h : total_numbers - count_divisible_by_5_or_7 = 343 := by sorry
  exact h

end count_not_divisible_by_5_or_7_l296_296510


namespace tan_five_pi_over_four_l296_296667

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l296_296667


namespace sequence_solution_l296_296963

theorem sequence_solution (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2) (h_rec : ∀ n > 0, a (n + 1) = a n ^ 2) : 
  a n = 2 ^ 2 ^ (n - 1) :=
by
  sorry

end sequence_solution_l296_296963


namespace prob_t_prob_vowel_l296_296701

def word := "mathematics"
def total_letters : ℕ := 11
def t_count : ℕ := 2
def vowel_count : ℕ := 4

-- Definition of being a letter "t"
def is_t (c : Char) : Prop := c = 't'

-- Definition of being a vowel
def is_vowel (c : Char) : Prop := c = 'a' ∨ c = 'e' ∨ c = 'i'

theorem prob_t : (t_count : ℚ) / total_letters = 2 / 11 :=
by
  sorry

theorem prob_vowel : (vowel_count : ℚ) / total_letters = 4 / 11 :=
by
  sorry

end prob_t_prob_vowel_l296_296701


namespace non_degenerate_ellipse_l296_296511

theorem non_degenerate_ellipse (k : ℝ) : 
    (∃ x y : ℝ, x^2 + 9 * y^2 - 6 * x + 18 * y = k) ↔ k > -18 :=
sorry

end non_degenerate_ellipse_l296_296511


namespace sally_initial_cards_l296_296028

variable (initial_cards : ℕ)

-- Define the conditions
def cards_given := 41
def cards_lost := 20
def cards_now := 48

-- Define the proof problem
theorem sally_initial_cards :
  initial_cards + cards_given - cards_lost = cards_now → initial_cards = 27 :=
by
  intro h
  sorry

end sally_initial_cards_l296_296028


namespace divide_by_repeating_decimal_l296_296308

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l296_296308


namespace scooter_safety_gear_price_increase_l296_296242

theorem scooter_safety_gear_price_increase :
  let last_year_scooter_price := 200
  let last_year_gear_price := 50
  let scooter_increase_rate := 0.08
  let gear_increase_rate := 0.15
  let total_last_year_price := last_year_scooter_price + last_year_gear_price
  let this_year_scooter_price := last_year_scooter_price * (1 + scooter_increase_rate)
  let this_year_gear_price := last_year_gear_price * (1 + gear_increase_rate)
  let total_this_year_price := this_year_scooter_price + this_year_gear_price
  let total_increase := total_this_year_price - total_last_year_price
  let percent_increase := (total_increase / total_last_year_price) * 100
  percent_increase = 9 :=
by
  -- sorry is added here to skip the proof steps
  sorry

end scooter_safety_gear_price_increase_l296_296242


namespace num_paths_from_E_to_G_pass_through_F_l296_296550

-- Definitions for the positions on the grid.
def E := (0, 4)
def G := (5, 0)
def F := (3, 3)

-- Function to calculate the number of combinations.
def binom (n k: ℕ) : ℕ := Nat.choose n k

-- The mathematical statement to be proven.
theorem num_paths_from_E_to_G_pass_through_F :
  (binom 4 1) * (binom 5 2) = 40 :=
by
  -- Placeholder for the proof.
  sorry

end num_paths_from_E_to_G_pass_through_F_l296_296550


namespace min_n_A0_An_ge_200_l296_296849

theorem min_n_A0_An_ge_200 :
  (∃ n : ℕ, (n * (n + 1)) / 3 ≥ 200) ∧
  (∀ m < 24, (m * (m + 1)) / 3 < 200) :=
sorry

end min_n_A0_An_ge_200_l296_296849


namespace venue_cost_correct_l296_296710

noncomputable def cost_per_guest : ℤ := 500
noncomputable def johns_guests : ℤ := 50
noncomputable def wifes_guests : ℤ := johns_guests + (60 * johns_guests) / 100
noncomputable def total_wedding_cost : ℤ := 50000
noncomputable def guests_cost : ℤ := wifes_guests * cost_per_guest
noncomputable def venue_cost : ℤ := total_wedding_cost - guests_cost

theorem venue_cost_correct : venue_cost = 10000 := 
  by
  -- Proof can be filled in here.
  sorry

end venue_cost_correct_l296_296710


namespace train_length_l296_296750

theorem train_length (speed_faster speed_slower : ℝ) (time_sec : ℝ) (length_each_train : ℝ) :
  speed_faster = 47 ∧ speed_slower = 36 ∧ time_sec = 36 ∧ 
  (length_each_train = 55 ↔ 2 * length_each_train = ((speed_faster - speed_slower) * (1000/3600) * time_sec)) :=
by {
  -- We declare the speeds in km/hr and convert the relative speed to m/s for calculation.
  sorry
}

end train_length_l296_296750


namespace b_days_solve_l296_296473

-- Definitions from the conditions
variable (b_days : ℝ)
variable (a_rate : ℝ) -- work rate of a
variable (b_rate : ℝ) -- work rate of b

-- Condition 1: a is twice as fast as b
def twice_as_fast_as_b : Prop :=
  a_rate = 2 * b_rate

-- Condition 2: a and b together can complete the work in 3.333333333333333 days
def combined_completion_time : Prop :=
  1 / (a_rate + b_rate) = 10 / 3

-- The number of days b alone can complete the work should satisfy this equation
def b_alone_can_complete_in_b_days : Prop :=
  b_rate = 1 / b_days

-- The actual theorem we want to prove:
theorem b_days_solve (b_rate a_rate : ℝ) (h1 : twice_as_fast_as_b a_rate b_rate) (h2 : combined_completion_time a_rate b_rate) : b_days = 10 :=
by
  sorry

end b_days_solve_l296_296473


namespace product_of_integers_between_sqrt_115_l296_296735

theorem product_of_integers_between_sqrt_115 :
  ∃ a b : ℕ, 100 < 115 ∧ 115 < 121 ∧ a = 10 ∧ b = 11 ∧ a * b = 110 := by
  sorry

end product_of_integers_between_sqrt_115_l296_296735


namespace china_junior_1990_problem_l296_296420

theorem china_junior_1990_problem 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1)
  (h1 : a * x = y * z / (y + z))
  (h2 : b * y = x * z / (x + z))
  (h3 : c * z = x * y / (x + y)) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1) :=
sorry

end china_junior_1990_problem_l296_296420


namespace rhind_papyrus_prob_l296_296751

theorem rhind_papyrus_prob (a₁ a₂ a₃ a₄ a₅ : ℝ) (q : ℝ) 
  (h_geom_seq : a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3 ∧ a₅ = a₁ * q^4)
  (h_loaves_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 93)
  (h_condition : a₁ + a₂ = (3/4) * a₃) 
  (q_gt_one : q > 1) :
  a₃ = 12 :=
sorry

end rhind_papyrus_prob_l296_296751


namespace pieces_after_10_cuts_l296_296586

-- Define the number of cuts
def cuts : ℕ := 10

-- Define the function that calculates the number of pieces
def pieces (k : ℕ) : ℕ := k + 1

-- State the theorem to prove the number of pieces given 10 cuts
theorem pieces_after_10_cuts : pieces cuts = 11 :=
by
  -- Proof goes here
  sorry

end pieces_after_10_cuts_l296_296586


namespace solution_set_of_inequality_l296_296851

variable {R : Type} [LinearOrderedField R]

theorem solution_set_of_inequality (f : R -> R) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, 0 < x ∧ x < y → f x < f y) (h3 : f 1 = 0) :
  { x : R | (f x - f (-x)) / x < 0 } = { x : R | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) } :=
sorry

end solution_set_of_inequality_l296_296851


namespace find_n_l296_296976

theorem find_n (n : ℕ) : (1/5)^35 * (1/4)^18 = 1/(n*(10)^35) → n = 2 :=
by
  sorry

end find_n_l296_296976


namespace total_balloons_is_72_l296_296952

-- Definitions for the conditions from the problem
def fred_balloons : Nat := 10
def sam_balloons : Nat := 46
def dan_balloons : Nat := 16

-- The total number of red balloons is the sum of Fred's, Sam's, and Dan's balloons
def total_balloons (f s d : Nat) : Nat := f + s + d

-- The theorem stating the problem to be proved
theorem total_balloons_is_72 : total_balloons fred_balloons sam_balloons dan_balloons = 72 := by
  sorry

end total_balloons_is_72_l296_296952


namespace tan_five_pi_over_four_l296_296655

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l296_296655


namespace six_digit_multiple_of_nine_l296_296522

theorem six_digit_multiple_of_nine (d : ℕ) (hd : d ≤ 9) (hn : 9 ∣ (30 + d)) : d = 6 := by
  sorry

end six_digit_multiple_of_nine_l296_296522


namespace range_of_a_l296_296532

def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = - f x

noncomputable def f (x : ℝ) :=
  if x ≥ 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h_odd : is_odd_function f) 
(hf_pos : ∀ x : ℝ, x ≥ 0 → f x = x^2 + 2*x) : 
  f (2 - a^2) > f a → -2 < a ∧ a < 1 :=
sorry

end range_of_a_l296_296532


namespace sum_geometric_sequence_l296_296226

theorem sum_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : ∀ n, 2 * a n - 2 = S n) : 
  S n = 2^(n+1) - 2 :=
sorry

end sum_geometric_sequence_l296_296226


namespace price_on_friday_is_correct_l296_296717

-- Define initial price on Tuesday
def price_on_tuesday : ℝ := 50

-- Define the percentage increase on Wednesday (20%)
def percentage_increase : ℝ := 0.20

-- Define the percentage discount on Friday (15%)
def percentage_discount : ℝ := 0.15

-- Define the price on Wednesday after the increase
def price_on_wednesday : ℝ := price_on_tuesday * (1 + percentage_increase)

-- Define the price on Friday after the discount
def price_on_friday : ℝ := price_on_wednesday * (1 - percentage_discount)

-- Theorem statement to prove that the price on Friday is 51 dollars
theorem price_on_friday_is_correct : price_on_friday = 51 :=
by
  sorry

end price_on_friday_is_correct_l296_296717


namespace animal_stickers_l296_296054

theorem animal_stickers {flower stickers total_stickers animal_stickers : ℕ} 
  (h_flower_stickers : flower = 8) 
  (h_total_stickers : total_stickers = 14)
  (h_total_eq : total_stickers = flower + animal_stickers) : 
  animal_stickers = 6 :=
by
  sorry

end animal_stickers_l296_296054


namespace both_fifth_and_ninth_terms_are_20_l296_296404

def sequence_a (n : ℕ) : ℕ := n^2 - 14 * n + 65

theorem both_fifth_and_ninth_terms_are_20 : sequence_a 5 = 20 ∧ sequence_a 9 = 20 := 
by
  sorry

end both_fifth_and_ninth_terms_are_20_l296_296404


namespace quadrilateral_angle_difference_l296_296597

theorem quadrilateral_angle_difference (h_ratio : ∀ (a b c d : ℕ), a = 3 * d ∧ b = 4 * d ∧ c = 5 * d ∧ d = 6 * d) 
  (h_sum : ∀ (a b c d : ℕ), a + b + c + d = 360) : 
  ∃ (x : ℕ), 6 * x - 3 * x = 60 := 
by 
  sorry

end quadrilateral_angle_difference_l296_296597


namespace longer_trip_due_to_red_lights_l296_296191

theorem longer_trip_due_to_red_lights :
  ∀ (num_lights : ℕ) (green_time first_route_base_time red_time_per_light second_route_time : ℕ),
  num_lights = 3 →
  first_route_base_time = 10 →
  red_time_per_light = 3 →
  second_route_time = 14 →
  (first_route_base_time + num_lights * red_time_per_light) - second_route_time = 5 :=
by
  intros num_lights green_time first_route_base_time red_time_per_light second_route_time
  sorry

end longer_trip_due_to_red_lights_l296_296191


namespace band_total_l296_296861

theorem band_total (flutes_total clarinets_total trumpets_total pianists_total : ℕ)
                   (flutes_pct clarinets_pct trumpets_pct pianists_pct : ℚ)
                   (h_flutes : flutes_total = 20)
                   (h_clarinets : clarinets_total = 30)
                   (h_trumpets : trumpets_total = 60)
                   (h_pianists : pianists_total = 20)
                   (h_flutes_pct : flutes_pct = 0.8)
                   (h_clarinets_pct : clarinets_pct = 0.5)
                   (h_trumpets_pct : trumpets_pct = 1/3)
                   (h_pianists_pct : pianists_pct = 1/10) :
  flutes_total * flutes_pct + clarinets_total * clarinets_pct + 
  trumpets_total * trumpets_pct + pianists_total * pianists_pct = 53 := by
  sorry

end band_total_l296_296861


namespace find_x_l296_296804

def has_three_distinct_prime_divisors (n : ℕ) : Prop :=
  let x := 9^n - 1
  (Prime 11 ∧ x % 11 = 0)
  ∧ (findDistinctPrimes x).length = 3

theorem find_x (n : ℕ) (h1 : has_three_distinct_prime_divisors n) : 9^n - 1 = 59048 := by
  sorry

end find_x_l296_296804


namespace union_M_N_is_real_l296_296147

def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

theorem union_M_N_is_real : M ∪ N = Set.univ := by
  sorry

end union_M_N_is_real_l296_296147


namespace mean_age_Mendez_children_l296_296879

def Mendez_children_ages : List ℕ := [5, 5, 10, 12, 15]

theorem mean_age_Mendez_children : 
  (5 + 5 + 10 + 12 + 15) / 5 = 9.4 := 
by
  sorry

end mean_age_Mendez_children_l296_296879


namespace stormi_lawns_mowed_l296_296876

def num_lawns_mowed (cars_washed : ℕ) (money_per_car : ℕ) 
                    (lawns_mowed : ℕ) (money_per_lawn : ℕ) 
                    (bike_cost : ℕ) (money_needed : ℕ) : Prop :=
  (cars_washed * money_per_car + lawns_mowed * money_per_lawn) = (bike_cost - money_needed)

theorem stormi_lawns_mowed : num_lawns_mowed 3 10 2 13 80 24 :=
by
  sorry

end stormi_lawns_mowed_l296_296876


namespace typist_current_salary_l296_296167

def original_salary : ℝ := 4000.0000000000005
def increased_salary (os : ℝ) : ℝ := os + (os * 0.1)
def decreased_salary (is : ℝ) : ℝ := is - (is * 0.05)

theorem typist_current_salary : decreased_salary (increased_salary original_salary) = 4180 :=
by
  sorry

end typist_current_salary_l296_296167


namespace eight_div_repeating_three_l296_296296

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l296_296296


namespace smallest_positive_integer_b_l296_296946
-- Import the necessary library

-- Define the conditions and problem statement
def smallest_b_factors (r s : ℤ) := r + s

theorem smallest_positive_integer_b :
  ∃ r s : ℤ, r * s = 1800 ∧ ∀ r' s' : ℤ, r' * s' = 1800 → smallest_b_factors r s ≤ smallest_b_factors r' s' :=
by
  -- Declare that the smallest positive integer b satisfying the conditions is 85
  use 45, 40
  -- Check the core condition
  have rs_eq_1800 := (45 * 40 = 1800)
  sorry

end smallest_positive_integer_b_l296_296946


namespace probability_is_one_third_l296_296506

open Finset Rat

def subset : Finset ℕ := {1, 2, 5, 10, 15, 25, 50}

def isMultipleOf50 (n : ℕ) : Prop := n % 50 = 0

def countValidPairs : ℕ :=
  (subset.product subset).filter (λ p, p.1 ≠ p.2 ∧ isMultipleOf50 (p.1 * p.2)).card

def totalPairs : ℕ :=
  subset.card * (subset.card - 1) / 2

def probability : ℚ :=
  countValidPairs / totalPairs

theorem probability_is_one_third :
  probability = 1 / 3 := by
  sorry

end probability_is_one_third_l296_296506


namespace parabola_point_coord_l296_296807

theorem parabola_point_coord {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : (x - 1)^2 + y^2 = 100) : x = 9 ∧ (y = 6 ∨ y = -6) :=
by 
  sorry

end parabola_point_coord_l296_296807


namespace birdhouse_price_l296_296863

theorem birdhouse_price (S : ℤ) : 
  (2 * 22) + (2 * 16) + (3 * S) = 97 → 
  S = 7 :=
by
  sorry

end birdhouse_price_l296_296863


namespace robinson_family_children_count_l296_296724

theorem robinson_family_children_count 
  (m : ℕ) -- mother's age
  (f : ℕ) (f_age : f = 50) -- father's age is 50
  (x : ℕ) -- number of children
  (y : ℕ) -- average age of children
  (h1 : (m + 50 + x * y) / (2 + x) = 22)
  (h2 : (m + x * y) / (1 + x) = 18) :
  x = 6 := 
sorry

end robinson_family_children_count_l296_296724


namespace new_cases_first_week_l296_296584

theorem new_cases_first_week
  (X : ℕ)
  (second_week_cases : X / 2 = X / 2)
  (third_week_cases : X / 2 + 2000 = (X / 2) + 2000)
  (total_cases : X + X / 2 + (X / 2 + 2000) = 9500) :
  X = 3750 := 
by sorry

end new_cases_first_week_l296_296584


namespace exists_plane_intersecting_in_parallel_lines_l296_296889

variables {Point Line Plane : Type}
variables (a : Line) (S₁ S₂ : Plane)

-- Definitions and assumptions
def intersects_in (a : Line) (P : Plane) : Prop := sorry
def parallel_lines (l₁ l₂ : Line) : Prop := sorry

-- Proof problem statement
theorem exists_plane_intersecting_in_parallel_lines :
  ∃ P : Plane, intersects_in a P ∧
    (∀ l₁ l₂ : Line, (intersects_in l₁ S₁ ∧ intersects_in l₂ S₂ ∧ l₁ = l₂)
                     → parallel_lines l₁ l₂) :=
sorry

end exists_plane_intersecting_in_parallel_lines_l296_296889


namespace second_smallest_packs_of_hot_dogs_l296_296512

theorem second_smallest_packs_of_hot_dogs 
  (n : ℕ) 
  (h1 : ∃ (k : ℕ), n = 2 * k + 2)
  (h2 : 12 * n ≡ 6 [MOD 8]) : 
  n = 4 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l296_296512


namespace count_divisible_by_45_l296_296119

theorem count_divisible_by_45 : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ x % 100 = 45 → x % 45 = 0 → n = 10) :=
by {
  sorry
}

end count_divisible_by_45_l296_296119


namespace arithmetic_mean_pq_is_10_l296_296598

variables {p q r : ℝ}

theorem arithmetic_mean_pq_is_10 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) 
  : (p + q) / 2 = 10 :=
by 
  exact h1

end arithmetic_mean_pq_is_10_l296_296598


namespace division_of_decimal_l296_296313

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l296_296313


namespace sum_integer_solutions_correct_l296_296960

noncomputable def sum_of_integer_solutions (m : ℝ) : ℝ :=
  if (3 ≤ m ∧ m < 6) ∨ (-6 ≤ m ∧ m < -3) then -9 else 0

theorem sum_integer_solutions_correct (m : ℝ) :
  (∀ x : ℝ, (3 * x + m < 0 ∧ x > -5) → (∃ s : ℝ, s = sum_of_integer_solutions m ∧ s = -9)) :=
by
  sorry

end sum_integer_solutions_correct_l296_296960


namespace speed_of_first_train_l296_296892

theorem speed_of_first_train
  (length_train1 length_train2 : ℕ)
  (speed_train2 : ℕ)
  (time_seconds : ℝ)
  (distance_km : ℝ := (length_train1 + length_train2) / 1000)
  (time_hours : ℝ := time_seconds / 3600)
  (relative_speed : ℝ := distance_km / time_hours) :
  length_train1 = 111 →
  length_train2 = 165 →
  speed_train2 = 120 →
  time_seconds = 4.516002356175142 →
  relative_speed = 220 →
  speed_train2 + 100 = relative_speed :=
by
  intros
  sorry

end speed_of_first_train_l296_296892


namespace necessary_but_not_sufficient_for_odd_function_l296_296727

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f (x)

theorem necessary_but_not_sufficient_for_odd_function (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f :=
sorry

end necessary_but_not_sufficient_for_odd_function_l296_296727


namespace greatest_whole_number_satisfying_inequality_l296_296670

-- Define the problem condition
def inequality (x : ℤ) : Prop := 5 * x - 4 < 3 - 2 * x

-- Prove that under this condition, the greatest whole number satisfying it is 0
theorem greatest_whole_number_satisfying_inequality : ∃ (n : ℤ), inequality n ∧ ∀ (m : ℤ), inequality m → m ≤ n :=
begin
  use 0,
  split,
  { -- Proof that 0 satisfies the inequality
    unfold inequality,
    linarith, },
  { -- Proof that 0 is the greatest whole number satisfying the inequality
    intro m,
    unfold inequality,
    intro hm,
    linarith, }
end

#check greatest_whole_number_satisfying_inequality

end greatest_whole_number_satisfying_inequality_l296_296670


namespace square_heptagon_intersection_l296_296733

open EuclideanGeometry

def square_circumscribed_circle_condition
  (k : circle)
  (A B C D O : point)
  (e : line)
  (P Q R : point) : Prop :=
  square A B C D ∧ 
  circumscribed_circle k A B C D ∧ 
  center O k ∧ 
  tangent e k C ∧
  on extension_line A C P ∧
  intersects_line P D e Q ∧
  perpendicular_line P B R

-- Main Theorem
theorem square_heptagon_intersection
  (k : circle)
  (A B C D O : point)
  (e : line)
  (P Q R : point)
  (h : square_circumscribed_circle_condition k A B C D O e P Q R)
  (h1 : dist Q R = dist O A) :
  intersects_perpendicular_bisector_at_heptagon_vertices k A O P :=
begin
  sorry
end

end square_heptagon_intersection_l296_296733


namespace total_number_of_trees_l296_296155

theorem total_number_of_trees (side_length : ℝ) (area_ratio : ℝ) (trees_per_sqm : ℝ) (H : side_length = 100) (R : area_ratio = 3) (T : trees_per_sqm = 4) : 
  let street_area := side_length ^ 2 in 
  let forest_area := area_ratio * street_area in
  let total_trees := forest_area * trees_per_sqm in
  total_trees = 120000 :=
by
  -- proof steps go here
  sorry

end total_number_of_trees_l296_296155


namespace smallest_satisfying_N_is_2520_l296_296094

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end smallest_satisfying_N_is_2520_l296_296094


namespace initial_peanuts_count_l296_296463

def peanuts_initial (P : ℕ) : Prop :=
  P - (1 / 4 : ℝ) * P - 29 = 82

theorem initial_peanuts_count (P : ℕ) (h : peanuts_initial P) : P = 148 :=
by
  -- The complete proof can be constructed here.
  sorry

end initial_peanuts_count_l296_296463


namespace all_n_eq_one_l296_296537

theorem all_n_eq_one (k : ℕ) (n : ℕ → ℕ)
  (h₁ : k ≥ 2)
  (h₂ : ∀ i, 1 ≤ i ∧ i < k → (n (i + 1)) ∣ 2^(n i) - 1)
  (h₃ : (n 1) ∣ 2^(n k) - 1) :
  ∀ i, 1 ≤ i ∧ i ≤ k → n i = 1 := 
sorry

end all_n_eq_one_l296_296537


namespace snow_on_second_day_l296_296361

-- Definition of conditions as variables in Lean
def snow_on_first_day := 6 -- in inches
def snow_melted := 2 -- in inches
def additional_snow_fifth_day := 12 -- in inches
def total_snow := 24 -- in inches

-- The variable for snow on the second day
variable (x : ℕ)

-- Proof goal
theorem snow_on_second_day : snow_on_first_day + x - snow_melted + additional_snow_fifth_day = total_snow → x = 8 :=
by
  intros h
  sorry

end snow_on_second_day_l296_296361


namespace ratio_of_speeds_l296_296628

theorem ratio_of_speeds (L V : ℝ) (R : ℝ) (h1 : L > 0) (h2 : V > 0) (h3 : R ≠ 0)
  (h4 : (1.48 * L) / (R * V) = (1.40 * L) / V) : R = 37 / 35 :=
by
  -- Proof would be inserted here
  sorry

end ratio_of_speeds_l296_296628


namespace total_students_in_halls_l296_296047

theorem total_students_in_halls :
  let S_g := 30
  let S_b := 2 * S_g
  let S_m := 3 / 5 * (S_g + S_b)
  S_g + S_b + S_m = 144 :=
by
  sorry

end total_students_in_halls_l296_296047


namespace product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l296_296441

section TriangularNumbers

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- Statement 1: The product of two consecutive triangular numbers is not a perfect square
theorem product_of_consecutive_triangular_not_square (n : ℕ) (hn : n > 0) :
  ¬ ∃ m : ℕ, triangular (n - 1) * triangular n = m * m := by
  sorry

-- Statement 2: There exist infinitely many larger triangular numbers such that the product with t_n is a perfect square
theorem infinite_larger_triangular_numbers_square_product (n : ℕ) :
  ∃ᶠ m in at_top, ∃ k : ℕ, triangular n * triangular m = k * k := by
  sorry

end TriangularNumbers

end product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l296_296441


namespace election_total_valid_votes_l296_296987

theorem election_total_valid_votes (V B : ℝ) 
    (hA : 0.45 * V = B * V + 250) 
    (hB : 2.5 * B = 62.5) :
    V = 1250 :=
by
  sorry

end election_total_valid_votes_l296_296987


namespace projections_relationship_l296_296770

theorem projections_relationship (a b r : ℝ) (h : r ≠ 0) :
  (∃ α β : ℝ, a = r * Real.cos α ∧ b = r * Real.cos β ∧ (Real.cos α)^2 + (Real.cos β)^2 = 1) → (a^2 + b^2 = r^2) :=
by
  sorry

end projections_relationship_l296_296770


namespace cake_slices_l296_296757

theorem cake_slices (S : ℕ) (h : 347 * S = 6 * 375 + 526) : S = 8 :=
sorry

end cake_slices_l296_296757


namespace total_rattlesnakes_l296_296279

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end total_rattlesnakes_l296_296279


namespace car_rental_total_cost_l296_296133

theorem car_rental_total_cost 
  (rental_cost : ℕ)
  (gallons : ℕ)
  (cost_per_gallon : ℕ)
  (cost_per_mile : ℚ)
  (miles_driven : ℕ)
  (H1 : rental_cost = 150)
  (H2 : gallons = 8)
  (H3 : cost_per_gallon = 350 / 100)
  (H4 : cost_per_mile = 50 / 100)
  (H5 : miles_driven = 320) :
  rental_cost + gallons * cost_per_gallon + miles_driven * cost_per_mile = 338 :=
  sorry

end car_rental_total_cost_l296_296133


namespace seventh_term_of_geometric_sequence_l296_296639

theorem seventh_term_of_geometric_sequence (r : ℝ) 
  (h1 : 3 * r^5 = 729) : 3 * r^6 = 2187 :=
sorry

end seventh_term_of_geometric_sequence_l296_296639


namespace tan_five_pi_over_four_l296_296657

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l296_296657


namespace cube_side_length_ratio_l296_296347

-- Define the conditions and question
variable (s₁ s₂ : ℝ)
variable (weight₁ weight₂ : ℝ)
variable (V₁ V₂ : ℝ)
variable (same_metal : Prop)

-- Conditions
def condition1 (weight₁ : ℝ) : Prop := weight₁ = 4
def condition2 (weight₂ : ℝ) : Prop := weight₂ = 32
def condition3 (V₁ V₂ : ℝ) (s₁ s₂ : ℝ) : Prop := (V₁ = s₁^3) ∧ (V₂ = s₂^3)
def condition4 (same_metal : Prop) : Prop := same_metal

-- Volume definition based on weights and proportion
noncomputable def volume_definition (weight₁ weight₂ V₁ V₂ : ℝ) : Prop :=
(weight₂ / weight₁) = (V₂ / V₁)

-- Define the proof target
theorem cube_side_length_ratio
    (h1 : condition1 weight₁)
    (h2 : condition2 weight₂)
    (h3 : condition3 V₁ V₂ s₁ s₂)
    (h4 : condition4 same_metal)
    (h5 : volume_definition weight₁ weight₂ V₁ V₂) : 
    (s₂ / s₁) = 2 :=
by
  sorry

end cube_side_length_ratio_l296_296347


namespace eight_div_repeating_three_l296_296295

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l296_296295


namespace overall_weighted_defective_shipped_percentage_l296_296764

theorem overall_weighted_defective_shipped_percentage
  (defective_A : ℝ := 0.06) (shipped_A : ℝ := 0.04) (prod_A : ℝ := 0.30)
  (defective_B : ℝ := 0.09) (shipped_B : ℝ := 0.06) (prod_B : ℝ := 0.50)
  (defective_C : ℝ := 0.12) (shipped_C : ℝ := 0.07) (prod_C : ℝ := 0.20) :
  prod_A * defective_A * shipped_A + prod_B * defective_B * shipped_B + prod_C * defective_C * shipped_C = 0.00510 :=
by
  sorry

end overall_weighted_defective_shipped_percentage_l296_296764


namespace find_f_neg_2_l296_296682

theorem find_f_neg_2 (f : ℝ → ℝ) (b x : ℝ) (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 3*x + b) (h3 : f 0 = 0) : f (-2) = 2 := by
sorry

end find_f_neg_2_l296_296682


namespace inverse_function_point_l296_296979

noncomputable def f (a : ℝ) (x : ℝ) := a^(x + 1)

theorem inverse_function_point (a : ℝ) (h_pos : 0 < a) (h_annoylem : f a (-1) = 1) :
  ∃ g : ℝ → ℝ, (∀ y, f a (g y) = y ∧ g (f a y) = y) ∧ g 1 = -1 :=
by
  sorry

end inverse_function_point_l296_296979


namespace problem1_problem2_l296_296081

open Nat

def binomial (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem problem1 : binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006 := by
  sorry

theorem problem2 : binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32 := by
  sorry

end problem1_problem2_l296_296081


namespace compare_abc_l296_296246

theorem compare_abc :
  let a := (4 - Real.log 4) / Real.exp 2
  let b := Real.log 2 / 2
  let c := 1 / Real.exp 1 in
  b < a ∧ a < c := 
sorry

end compare_abc_l296_296246


namespace vector_sum_magnitude_l296_296964

variable (a b : EuclideanSpace ℝ (Fin 3)) -- assuming 3-dimensional Euclidean space for vectors

-- Define the conditions
def mag_a : ℝ := 5
def mag_b : ℝ := 6
def dot_prod_ab : ℝ := -6

-- Prove the required magnitude condition
theorem vector_sum_magnitude (ha : ‖a‖ = mag_a) (hb : ‖b‖ = mag_b) (hab : inner a b = dot_prod_ab) :
  ‖a + b‖ = 7 :=
by
  sorry

end vector_sum_magnitude_l296_296964


namespace number_of_adult_dogs_l296_296705

theorem number_of_adult_dogs (x : ℕ) (h : 2 * 50 + x * 100 + 2 * 150 = 700) : x = 3 :=
by
  -- Definitions from conditions
  have cost_cats := 2 * 50
  have cost_puppies := 2 * 150
  have total_cost := 700
  
  -- Using the provided hypothesis to assert our proof
  sorry

end number_of_adult_dogs_l296_296705


namespace subway_speed_increase_l296_296888

theorem subway_speed_increase (s : ℝ) (h₀ : 0 ≤ s) (h₁ : s ≤ 7) : 
  (s^2 + 2 * s = 63) ↔ (s = 7) :=
by
  sorry 

end subway_speed_increase_l296_296888


namespace moles_of_water_formed_l296_296383

-- Defining the relevant constants
def NH4Cl_moles : ℕ := sorry  -- Some moles of Ammonium chloride (NH4Cl)
def NaOH_moles : ℕ := 3       -- 3 moles of Sodium hydroxide (NaOH)
def H2O_moles : ℕ := 3        -- The total moles of Water (H2O) formed

-- Statement of the problem
theorem moles_of_water_formed :
  NH4Cl_moles ≥ NaOH_moles → H2O_moles = 3 :=
sorry

end moles_of_water_formed_l296_296383


namespace inheritance_amount_l296_296136

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end inheritance_amount_l296_296136


namespace pq_square_identity_l296_296553

theorem pq_square_identity (p q : ℝ) (h1 : p - q = 4) (h2 : p * q = -2) : p^2 + q^2 = 12 :=
by
  sorry

end pq_square_identity_l296_296553


namespace number_of_paths_l296_296118

-- Define the conditions of the problem
def grid_width : ℕ := 7
def grid_height : ℕ := 6
def diagonal_steps : ℕ := 2

-- Define the main proof statement
theorem number_of_paths (width height diag : ℕ) 
  (Nhyp : width = grid_width ∧ height = grid_height ∧ diag = diagonal_steps) : 
  ∃ (paths : ℕ), paths = 6930 := 
sorry

end number_of_paths_l296_296118


namespace fib_ratio_bound_l296_296595

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_ratio_bound {a b n : ℕ} (h1: b > 0) (h2: fib (n-1) > 0)
  (h3: (fib n) * b > (fib (n-1)) * a)
  (h4: (fib (n+1)) * b < (fib n) * a) :
  b ≥ fib (n+1) :=
sorry

end fib_ratio_bound_l296_296595


namespace min_quadratic_expression_l296_296382

theorem min_quadratic_expression:
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 :=
by
  sorry

end min_quadratic_expression_l296_296382


namespace find_a_b_l296_296006

noncomputable def polynomial_factors (a b : ℚ) : Prop :=
  (Polynomial.C 8 * Polynomial.X^3 + Polynomial.C a * Polynomial.X^2 + Polynomial.C 20 * Polynomial.X + Polynomial.C b)
    % (Polynomial.C 3 * Polynomial.X + Polynomial.C 5) = 0

theorem find_a_b (a b : ℚ) (h : polynomial_factors a b) : 
  a = 40 / 3 ∧ b = -25 / 3 :=
by
  sorry

end find_a_b_l296_296006


namespace sum_of_squares_base_case_l296_296172

theorem sum_of_squares_base_case : 1^2 + 2^2 = (1 * 3 * 5) / 3 := by sorry

end sum_of_squares_base_case_l296_296172


namespace university_admission_l296_296913

def students_ratio (x y z : ℕ) : Prop :=
  x * 5 = y * 2 ∧ y * 3 = z * 5

def third_tier_students : ℕ := 1500

theorem university_admission :
  ∀ x y z : ℕ, students_ratio x y z → z = third_tier_students → y - x = 1500 :=
by
  intros x y z hratio hthird
  sorry

end university_admission_l296_296913


namespace min_value_expression_l296_296211

open Real

theorem min_value_expression 
  (a : ℝ) 
  (b : ℝ) 
  (hb : 0 < b) 
  (e : ℝ) 
  (he : e = 2.718281828459045) :
  ∃ x : ℝ, 
  (x = 2 * (1 - log 2)^2) ∧
  ∀ a b, 
    0 < b → 
    ((1 / 2) * exp a - log (2 * b))^2 + (a - b)^2 ≥ x :=
sorry

end min_value_expression_l296_296211


namespace rainfall_ratio_l296_296938

theorem rainfall_ratio (R_1 R_2 : ℕ) (h1 : R_1 + R_2 = 25) (h2 : R_2 = 15) : R_2 / R_1 = 3 / 2 :=
by
  sorry

end rainfall_ratio_l296_296938


namespace circles_internally_tangent_l296_296405

theorem circles_internally_tangent (R r : ℝ) (h1 : R + r = 5) (h2 : R * r = 6) (d : ℝ) (h3 : d = 1) : d = |R - r| :=
by
  -- This allows the logic of the solution to be captured as the theorem we need to prove
  sorry

end circles_internally_tangent_l296_296405


namespace x_squared_plus_y_squared_l296_296551

theorem x_squared_plus_y_squared (x y : ℝ) 
   (h1 : (x + y)^2 = 49) 
   (h2 : x * y = 8) 
   : x^2 + y^2 = 33 := 
by
  sorry

end x_squared_plus_y_squared_l296_296551


namespace natasha_dimes_l296_296257

theorem natasha_dimes (n : ℕ) (h1 : 10 < n) (h2 : n < 100) (h3 : n % 3 = 1) (h4 : n % 4 = 1) (h5 : n % 5 = 1) : n = 61 :=
sorry

end natasha_dimes_l296_296257


namespace greatest_discarded_oranges_l296_296345

theorem greatest_discarded_oranges (n : ℕ) : n % 7 ≤ 6 := 
by 
  sorry

end greatest_discarded_oranges_l296_296345


namespace solve_system_of_equations_l296_296722

theorem solve_system_of_equations 
  (a b c s : ℝ) (x y z : ℝ)
  (h1 : y^2 - z * x = a * (x + y + z)^2)
  (h2 : x^2 - y * z = b * (x + y + z)^2)
  (h3 : z^2 - x * y = c * (x + y + z)^2)
  (h4 : a^2 + b^2 + c^2 - (a * b + b * c + c * a) = a + b + c) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ x + y + z = 0) ∨
  ((x + y + z ≠ 0) ∧
   (x = (2 * c - a - b + 1) * s) ∧
   (y = (2 * a - b - c + 1) * s) ∧
   (z = (2 * b - c - a + 1) * s)) :=
by
  sorry

end solve_system_of_equations_l296_296722


namespace largest_among_options_l296_296820

theorem largest_among_options (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > (1/2) ∧ b > a^2 + b^2 ∧ b > 2*a*b := 
by
  sorry

end largest_among_options_l296_296820


namespace solve_quadratic_equation_l296_296266

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 + 2 * x - 15 = 0) ↔ (x = 3 ∨ x = -5) :=
by
  sorry -- proof omitted

end solve_quadratic_equation_l296_296266


namespace union_A_B_inter_A_B_inter_compA_B_l296_296403

-- Extend the universal set U to be the set of all real numbers ℝ
def U : Set ℝ := Set.univ

-- Define set A as the set of all real numbers x such that -3 ≤ x ≤ 4
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

-- Define set B as the set of all real numbers x such that -1 < x < 5
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Prove that A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5}
theorem union_A_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5} := by
  sorry

-- Prove that A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4}
theorem inter_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by
  sorry

-- Define the complement of A in U
def comp_A : Set ℝ := {x : ℝ | x < -3 ∨ x > 4}

-- Prove that (complement_U A) ∩ B = {x : ℝ | 4 < x ∧ x < 5}
theorem inter_compA_B : comp_A ∩ B = {x : ℝ | 4 < x ∧ x < 5} := by
  sorry

end union_A_B_inter_A_B_inter_compA_B_l296_296403


namespace polar_curve_is_parabola_l296_296601

theorem polar_curve_is_parabola (ρ θ : ℝ) (h : 3 * ρ * Real.sin θ ^ 2 + Real.cos θ = 0) : ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 3 * y ^ 2 + x = 0 :=
by
  sorry

end polar_curve_is_parabola_l296_296601


namespace time_spent_in_park_is_76_19_percent_l296_296139

noncomputable def total_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (t, _, _) => acc + t) 0

noncomputable def total_walking_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (_, w1, w2) => acc + (w1 + w2)) 0

noncomputable def total_trip_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  total_time_in_park trip_times + total_walking_time trip_times

noncomputable def percentage_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℚ :=
  (total_time_in_park trip_times : ℚ) / (total_trip_time trip_times : ℚ) * 100

theorem time_spent_in_park_is_76_19_percent (trip_times : List (ℕ × ℕ × ℕ)) :
  trip_times = [(120, 20, 25), (90, 15, 15), (150, 10, 20), (180, 30, 20), (120, 20, 10), (60, 15, 25)] →
  percentage_time_in_park trip_times = 76.19 :=
by
  intro h
  rw [h]  
  simp
  sorry

end time_spent_in_park_is_76_19_percent_l296_296139


namespace dice_probability_l296_296446

/-- A standard six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

open Die

/-- Calculates the probability that after re-rolling four dice, at least four out of the six total dice show the same number,
given that initially six dice are rolled and there is no three-of-a-kind, and there is a pair of dice showing the same number
which are then set aside before re-rolling the remaining four dice. -/
theorem dice_probability (h1 : ∀ (d1 d2 d3 d4 d5 d6 : Die), 
  ¬ (d1 = d2 ∧ d2 = d3 ∨ d1 = d2 ∧ d2 = d4 ∨ d1 = d2 ∧ d2 = d5 ∨
     d1 = d2 ∧ d2 = d6 ∨ d1 = d3 ∧ d3 = d4 ∨ d1 = d3 ∧ d3 = d5 ∨
     d1 = d3 ∧ d3 = d6 ∨ d1 = d4 ∧ d4 = d5 ∨ d1 = d4 ∧ d4 = d6 ∨
     d1 = d5 ∧ d5 = d6 ∨ d2 = d3 ∧ d3 = d4 ∨ d2 = d3 ∧ d3 = d5 ∨
     d2 = d3 ∧ d3 = d6 ∨ d2 = d4 ∧ d4 = d5 ∨ d2 = d4 ∧ d4 = d6 ∨
     d2 = d5 ∧ d5 = d6 ∨ d3 = d4 ∧ d4 = d5 ∨ d3 = d4 ∧ d4 = d6 ∨ d3 = d5 ∧ d5 = d6 ∨ d4 = d5 ∧ d5 = d6))
    (h2 : ∃ (d1 d2 : Die) (d3 d4 d5 d6 : Die), d1 = d2 ∧ d3 ≠ d1 ∧ d4 ≠ d1 ∧ d5 ≠ d1 ∧ d6 ≠ d1): 
    ℚ := 
11 / 81

end dice_probability_l296_296446


namespace jessica_total_cost_l296_296991

def price_of_cat_toy : ℝ := 10.22
def price_of_cage : ℝ := 11.73
def price_of_cat_food : ℝ := 5.65
def price_of_catnip : ℝ := 2.30
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.07

def discounted_price_of_cat_toy : ℝ := price_of_cat_toy * (1 - discount_rate)
def total_cost_before_tax : ℝ := discounted_price_of_cat_toy + price_of_cage + price_of_cat_food + price_of_catnip
def sales_tax : ℝ := total_cost_before_tax * tax_rate
def total_cost_after_discount_and_tax : ℝ := total_cost_before_tax + sales_tax

theorem jessica_total_cost : total_cost_after_discount_and_tax = 30.90 := by
  sorry

end jessica_total_cost_l296_296991


namespace comparison_l296_296526

noncomputable def a : ℝ := 7 / 9
noncomputable def b : ℝ := 0.7 * Real.exp 0.1
noncomputable def c : ℝ := Real.cos (2 / 3)

theorem comparison : c > a ∧ a > b :=
by
  -- c > a proof
  have h1 : c > a := sorry
  -- a > b proof
  have h2 : a > b := sorry
  exact ⟨h1, h2⟩

end comparison_l296_296526


namespace tangent_line_equation_parallel_to_given_line_l296_296271

theorem tangent_line_equation_parallel_to_given_line :
  ∃ (x y : ℝ),  y = x^3 - 3 * x^2
    ∧ (3 * x^2 - 6 * x = -3)
    ∧ (y = -2)
    ∧ (3 * x + y - 1 = 0) :=
sorry

end tangent_line_equation_parallel_to_given_line_l296_296271


namespace correct_calculation_l296_296329

variable (a b : ℝ)

theorem correct_calculation : ((-a^2)^3 = -a^6) :=
by sorry

end correct_calculation_l296_296329


namespace product_of_de_l296_296738

theorem product_of_de (d e : ℤ) (h1: ∀ (r : ℝ), r^2 - r - 1 = 0 → r^6 - (d : ℝ) * r - (e : ℝ) = 0) : 
  d * e = 40 :=
by
  sorry

end product_of_de_l296_296738


namespace minimum_students_both_l296_296929

variable (U : Type) -- U is the type representing the set of all students
variable (S_P S_C : Set U) -- S_P is the set of students who like physics, S_C is the set of students who like chemistry
variable (total_students : Nat) -- total_students is the total number of students
variable [Fintype U] -- U should be a finite set

-- conditions
variable (h_physics : (Fintype.card S_P).toFloat / total_students * 100 = 68)
variable (h_chemistry : (Fintype.card S_C).toFloat / total_students * 100 = 72)

-- statement of the theorem to be proved
theorem minimum_students_both : (Fintype.card (S_P ∩ S_C)).toFloat / total_students * 100 ≥ 40 := by
  sorry

end minimum_students_both_l296_296929


namespace find_length_of_AL_l296_296131

noncomputable def length_of_AL 
  (A B C L : ℝ) 
  (AB AC AL : ℝ)
  (BC : ℝ)
  (AB_ratio_AC : AB / AC = 5 / 2)
  (BAC_bisector : ∃k, L = k * BC)
  (vector_magnitude : (2 * AB + 5 * AC) = 2016) : Prop :=
  AL = 288

theorem find_length_of_AL 
  (A B C L : ℝ)
  (AB AC AL : ℝ)
  (BC : ℝ)
  (h1 : AB / AC = 5 / 2)
  (h2 : ∃k, L = k * BC)
  (h3 : (2 * AB + 5 * AC) = 2016) : length_of_AL A B C L AB AC AL BC h1 h2 h3 := sorry

end find_length_of_AL_l296_296131


namespace least_n_for_distance_l296_296846

-- Definitions ensuring our points and distances
def A_0 : (ℝ × ℝ) := (0, 0)

-- Assume we have distance function and equilateral triangles on given coordinates
def is_on_x_axis (p : ℕ → ℝ × ℝ) : Prop := ∀ n, (p n).snd = 0
def is_on_parabola (q : ℕ → ℝ × ℝ) : Prop := ∀ n, (q n).snd = (q n).fst^2
def is_equilateral (p : ℕ → ℝ × ℝ) (q : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  let d1 := dist (p (n-1)) (q n)
  let d2 := dist (q n) (p n)
  let d3 := dist (p (n-1)) (p n)
  d1 = d2 ∧ d2 = d3

-- Define the main property we want to prove
def main_property (n : ℕ) (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  A 0 = A_0 ∧ is_on_x_axis A ∧ is_on_parabola B ∧
  (∀ k, is_equilateral A B (k+1)) ∧
  dist A_0 (A n) ≥ 200

-- Final theorem statement
theorem least_n_for_distance (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) :
  (∃ n, main_property n A B ∧ (∀ m, main_property m A B → n ≤ m)) ↔ n = 24 := by
  sorry

end least_n_for_distance_l296_296846


namespace find_digits_sum_l296_296328

theorem find_digits_sum (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : (A = 6) ∧ (B = 6))
  (h4 : (100 * A + 44610 + B) % 72 = 0) : A + B = 12 := 
by
  sorry

end find_digits_sum_l296_296328


namespace compute_fraction_value_l296_296502

theorem compute_fraction_value : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end compute_fraction_value_l296_296502


namespace minimum_value_of_expression_l296_296012

theorem minimum_value_of_expression {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 := 
sorry

end minimum_value_of_expression_l296_296012


namespace find_difference_l296_296248

variable (f : ℝ → ℝ)

-- Conditions
axiom linear_f : ∀ x y a b, f (a * x + b * y) = a * f x + b * f y
axiom f_difference : f 6 - f 2 = 12

theorem find_difference : f 12 - f 2 = 30 :=
by
  sorry

end find_difference_l296_296248


namespace expansion_simplification_l296_296087

variable (x y : ℝ)

theorem expansion_simplification :
  let a := 3 * x + 4
  let b := 2 * x + 6 * y + 7
  a * b = 6 * x ^ 2 + 18 * x * y + 29 * x + 24 * y + 28 :=
by
  sorry

end expansion_simplification_l296_296087


namespace number_of_boundaries_l296_296756

def total_runs : ℕ := 120
def sixes : ℕ := 4
def runs_per_six : ℕ := 6
def percentage_runs_by_running : ℚ := 0.60
def runs_per_boundary : ℕ := 4

theorem number_of_boundaries :
  let runs_by_running := (percentage_runs_by_running * total_runs : ℚ)
  let runs_by_sixes := (sixes * runs_per_six)
  let runs_by_boundaries := (total_runs - runs_by_running - runs_by_sixes : ℚ)
  (runs_by_boundaries / runs_per_boundary) = 6 := by
  sorry

end number_of_boundaries_l296_296756


namespace find_remainder_l296_296997

-- Definitions based on the conditions
variables {R : Type*} [CommRing R] (p : R[X])

-- Conditions
def cond1 : p.eval 1 = 4 :=
by sorry -- Skip the proof for now

def cond2 : p.eval 2 = -3 :=
by sorry -- Skip the proof for now

def cond3 : p.eval (-3) = 1 :=
by sorry -- Skip the proof for now

-- The statement to prove
theorem find_remainder :
  (∀ q : R[X], ∃ r : R[X], r.degree < 3 ∧ p = q * (X - 1) * (X - 2) * (X + 3) + r) →
  (p % ((X - 1) * (X - 2) * (X + 3)) = X^2 - 10 * X + 13) :=
by {
  intros h q,
  have h1 := cond1,
  have h2 := cond2,
  have h3 := cond3,
  sorry -- Skipping the proof
}

end find_remainder_l296_296997


namespace total_games_l296_296707

def joan_games_this_year : ℕ := 4
def joan_games_last_year : ℕ := 9

theorem total_games (this_year_games last_year_games : ℕ) 
    (h1 : this_year_games = joan_games_this_year) 
    (h2 : last_year_games = joan_games_last_year) : 
    this_year_games + last_year_games = 13 := 
by
  rw [h1, h2]
  exact rfl

end total_games_l296_296707


namespace dog_weights_l296_296196

structure DogWeightProgression where
  initial: ℕ   -- initial weight in pounds
  week_9: ℕ    -- weight at 9 weeks in pounds
  month_3: ℕ  -- weight at 3 months in pounds
  month_5: ℕ  -- weight at 5 months in pounds
  year_1: ℕ   -- weight at 1 year in pounds

theorem dog_weights :
  ∃ (golden_retriever labrador poodle : DogWeightProgression),
  golden_retriever.initial = 6 ∧
  golden_retriever.week_9 = 12 ∧
  golden_retriever.month_3 = 24 ∧
  golden_retriever.month_5 = 48 ∧
  golden_retriever.year_1 = 78 ∧
  labrador.initial = 8 ∧
  labrador.week_9 = 24 ∧
  labrador.month_3 = 36 ∧
  labrador.month_5 = 72 ∧
  labrador.year_1 = 102 ∧
  poodle.initial = 4 ∧
  poodle.week_9 = 16 ∧
  poodle.month_3 = 32 ∧
  poodle.month_5 = 32 ∧
  poodle.year_1 = 52 :=
by 
  have golden_retriever : DogWeightProgression := { initial := 6, week_9 := 12, month_3 := 24, month_5 := 48, year_1 := 78 }
  have labrador : DogWeightProgression := { initial := 8, week_9 := 24, month_3 := 36, month_5 := 72, year_1 := 102 }
  have poodle : DogWeightProgression := { initial := 4, week_9 := 16, month_3 := 32, month_5 := 32, year_1 := 52 }
  use golden_retriever, labrador, poodle
  repeat { split };
  { sorry }

end dog_weights_l296_296196


namespace product_of_three_consecutive_integers_l296_296276

theorem product_of_three_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 740)
    (x1 : ℕ := x - 1) (x2 : ℕ := x) (x3 : ℕ := x + 1) :
    x1 * x2 * x3 = 17550 :=
by
  sorry

end product_of_three_consecutive_integers_l296_296276


namespace sum_of_first_five_terms_l296_296194

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_sum_n : ∀ n, S n = n / 2 * (a 1 + a n))
  (h_roots : ∀ x, x^2 - x - 3 = 0 → x = a 2 ∨ x = a 4)
  (h_vieta : a 2 + a 4 = 1) :
  S 5 = 5 / 2 :=
  sorry

end sum_of_first_five_terms_l296_296194


namespace similar_triangle_shortest_side_l296_296070

theorem similar_triangle_shortest_side
  (a₁ : ℕ) (c₁ : ℕ) (c₂ : ℕ)
  (h₁ : a₁ = 15) (h₂ : c₁ = 17) (h₃ : c₂ = 68)
  (right_triangle_1 : a₁^2 + b₁^2 = c₁^2)
  (similar_triangles : ∃ k : ℕ, c₂ = k * c₁) :
  shortest_side = 32 := 
sorry

end similar_triangle_shortest_side_l296_296070


namespace find_functions_l296_296025

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_functions
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_1 : ∀ x : ℝ, |x| ≤ 1 → |f a b c x| ≤ 1)
  (h_2 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ ∀ x : ℝ, |x| ≤ 1 → |f' a b x₀| ≥ |f' a b x| )
  (K : ℝ)
  (h_3 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ |f' a b x₀| = K) :
  (f a b c = fun x ↦ 2 * x^2 - 1) ∨ (f a b c = fun x ↦ -2 * x^2 + 1) ∧ K = 4 := 
sorry

end find_functions_l296_296025


namespace largest_consecutive_sum_is_nine_l296_296615

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end largest_consecutive_sum_is_nine_l296_296615


namespace jinsu_third_attempt_kicks_l296_296549

theorem jinsu_third_attempt_kicks
  (hoseok_kicks : ℕ) (jinsu_first_attempt : ℕ) (jinsu_second_attempt : ℕ) (required_kicks : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first_attempt = 15 →
  jinsu_second_attempt = 15 →
  required_kicks = 19 →
  jinsu_first_attempt + jinsu_second_attempt + required_kicks > hoseok_kicks :=
by
  sorry

end jinsu_third_attempt_kicks_l296_296549


namespace cubic_difference_l296_296110

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : a^3 - b^3 = 353.5 := by
  sorry

end cubic_difference_l296_296110


namespace find_divisor_l296_296631

theorem find_divisor (d : ℕ) (h1 : 109 % d = 1) (h2 : 109 / d = 9) : d = 12 := by
  sorry

end find_divisor_l296_296631


namespace number_of_cows_l296_296476

variable (D C : Nat)

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 30) : C = 15 :=
by
  sorry

end number_of_cows_l296_296476


namespace annual_rate_of_decrease_l296_296885

variable (r : ℝ) (initial_population population_after_2_years : ℝ)

-- Conditions
def initial_population_eq : initial_population = 30000 := sorry
def population_after_2_years_eq : population_after_2_years = 19200 := sorry
def population_formula : population_after_2_years = initial_population * (1 - r)^2 := sorry

-- Goal: Prove that the annual rate of decrease r is 0.2
theorem annual_rate_of_decrease :
  r = 0.2 := sorry

end annual_rate_of_decrease_l296_296885


namespace equivalent_area_CDM_l296_296839

variables A B C D G H E F K L M : Type
variables (trapezoid: Trapezoid A B C D)
variables (G_on_AD : OnBase G A D)
variables (H_on_AD : OnBase H A D)
variables (E_on_BC : OnBase E B C)
variables (F_on_BC : OnBase F B C)
variables (K_intersect_BG_AE: Intersect K BG AE)
variables (L_intersect_EH_GF: Intersect L EH GF)
variables (M_intersect_FD_HC: Intersect M FD HC)
variables (area_ELGK: Area Quadrilateral ELGK = 4)
variables (area_FMHL: Area Quadrilateral FMHL = 8)

theorem equivalent_area_CDM :
  ∃ (CDM_area : ℕ), CDM_area = 5 ∨ CDM_area = 7 :=
  sorry

end equivalent_area_CDM_l296_296839


namespace compare_f_values_l296_296911

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.cos x

theorem compare_f_values :
  f 0 < f 0.5 ∧ f 0.5 < f 0.6 :=
by {
  -- proof would go here
  sorry
}

end compare_f_values_l296_296911


namespace numeral_is_1_11_l296_296983

-- Define the numeral question and condition
def place_value_difference (a b : ℝ) : Prop :=
  10 * b - b = 99.99

-- Now we define the problem statement in Lean
theorem numeral_is_1_11 (a b : ℝ) (h : place_value_difference a b) : 
  a = 100 ∧ b = 11.11 ∧ (a - b = 99.99) :=
  sorry

end numeral_is_1_11_l296_296983


namespace division_by_repeating_decimal_l296_296290

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l296_296290


namespace mary_unanswered_questions_l296_296862

theorem mary_unanswered_questions :
  ∃ (c w u : ℕ), 150 = 6 * c + 3 * u ∧ 118 = 40 + 5 * c - 2 * w ∧ 50 = c + w + u ∧ u = 16 :=
by
  sorry

end mary_unanswered_questions_l296_296862


namespace kenny_total_liquid_l296_296082

def total_liquid (oil_per_recipe water_per_recipe : ℚ) (times : ℕ) : ℚ :=
  (oil_per_recipe + water_per_recipe) * times

theorem kenny_total_liquid :
  total_liquid 0.17 1.17 12 = 16.08 := by
  sorry

end kenny_total_liquid_l296_296082


namespace greatest_value_exprD_l296_296824

-- Conditions
def a : ℚ := 2
def b : ℚ := 5

-- Expressions
def exprA := a / b
def exprB := b / a
def exprC := a - b
def exprD := b - a
def exprE := (1/2 : ℚ) * a

-- Proof problem statement
theorem greatest_value_exprD : exprD = 3 ∧ exprD > exprA ∧ exprD > exprB ∧ exprD > exprC ∧ exprD > exprE := sorry

end greatest_value_exprD_l296_296824


namespace incorrect_statement_for_function_l296_296392

theorem incorrect_statement_for_function (x : ℝ) (h : x > 0) : 
  ¬(∀ x₁ x₂ : ℝ, (x₁ > 0) → (x₂ > 0) → (x₁ < x₂) → (6 / x₁ < 6 / x₂)) := 
sorry

end incorrect_statement_for_function_l296_296392


namespace fraction_equality_l296_296826

variables {a b : ℝ}

theorem fraction_equality (h : ab * (a + b) = 1) (ha : a > 0) (hb : b > 0) : 
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := 
sorry

end fraction_equality_l296_296826


namespace find_K_l296_296241

theorem find_K (K m n : ℝ) (p : ℝ) (hp : p = 0.3333333333333333)
  (eq1 : m = K * n + 5)
  (eq2 : m + 2 = K * (n + p) + 5) : 
  K = 6 := 
by
  sorry

end find_K_l296_296241


namespace regular_seminar_fee_l296_296919

-- Define the main problem statement
theorem regular_seminar_fee 
  (F : ℝ) 
  (discount_per_teacher : ℝ) 
  (number_of_teachers : ℕ)
  (food_allowance_per_teacher : ℝ)
  (total_spent : ℝ) :
  discount_per_teacher = 0.95 * F →
  number_of_teachers = 10 →
  food_allowance_per_teacher = 10 →
  total_spent = 1525 →
  (number_of_teachers * discount_per_teacher + number_of_teachers * food_allowance_per_teacher = total_spent) →
  F = 150 := 
  by sorry

end regular_seminar_fee_l296_296919


namespace pears_for_twenty_apples_l296_296417

-- Definitions based on given conditions
variables (a o p : ℕ) -- represent the number of apples, oranges, and pears respectively
variables (k1 k2 : ℕ) -- scaling factors 

-- Conditions as given
axiom ten_apples_five_oranges : 10 * a = 5 * o
axiom three_oranges_four_pears : 3 * o = 4 * p

-- Proving the number of pears Mia can buy for 20 apples
theorem pears_for_twenty_apples : 13 * p ≤ (20 * a) :=
by
  -- Actual proof would go here
  sorry

end pears_for_twenty_apples_l296_296417


namespace distance_between_city_and_village_l296_296468

variables (S x y : ℝ)

theorem distance_between_city_and_village (h1 : S / 2 - 2 = y * S / (2 * x))
    (h2 : 2 * S / 3 + 2 = x * S / (3 * y)) : S = 6 :=
by
  sorry

end distance_between_city_and_village_l296_296468


namespace log10_two_bounds_l296_296466

theorem log10_two_bounds
  (h1 : 10 ^ 3 = 1000)
  (h2 : 10 ^ 4 = 10000)
  (h3 : 2 ^ 10 = 1024)
  (h4 : 2 ^ 12 = 4096) :
  1 / 4 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 0.4 := 
sorry

end log10_two_bounds_l296_296466


namespace isosceles_triangle_side_length_condition_l296_296398

theorem isosceles_triangle_side_length_condition (x y : ℕ) :
    y = x + 1 ∧ 2 * x + y = 16 → (y = 6 → x = 5) :=
by sorry

end isosceles_triangle_side_length_condition_l296_296398


namespace complex_z_calculation_l296_296858

theorem complex_z_calculation (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z :=
sorry

end complex_z_calculation_l296_296858


namespace sin_exp_intersections_count_l296_296935

open Real

theorem sin_exp_intersections_count : 
  (finset.card
    (finset.filter 
      (λ x, sin x = (1 / 3) ^ x) 
      (finset.Ico 0 (floor (150 * π) + 1)))) = 150 := 
sorry

end sin_exp_intersections_count_l296_296935


namespace find_initial_students_l296_296833

def initial_students (S : ℕ) : Prop :=
  S - 4 + 42 = 48 

theorem find_initial_students (S : ℕ) (h : initial_students S) : S = 10 :=
by {
  -- The proof can be filled out here but we skip it using sorry
  sorry
}

end find_initial_students_l296_296833


namespace double_espresso_cost_l296_296865

-- Define the cost of coffee, days, and total spent as constants
def iced_coffee : ℝ := 2.5
def total_days : ℝ := 20
def total_spent : ℝ := 110

-- Define the cost of double espresso as variable E
variable (E : ℝ)

-- The proposition to prove
theorem double_espresso_cost : (total_days * (E + iced_coffee) = total_spent) → (E = 3) :=
by
  sorry

end double_espresso_cost_l296_296865


namespace solve_for_question_mark_l296_296482

/-- Prove that the number that should replace "?" in the equation 
    300 * 2 + (12 + ?) * (1 / 8) = 602 is equal to 4. -/
theorem solve_for_question_mark : 
  ∃ (x : ℕ), 300 * 2 + (12 + x) * (1 / 8) = 602 ∧ x = 4 := 
by
  sorry

end solve_for_question_mark_l296_296482


namespace equilateral_triangle_stack_impossible_l296_296837

theorem equilateral_triangle_stack_impossible :
  ¬ ∃ n : ℕ, 3 * 55 = 6 * n :=
by
  sorry

end equilateral_triangle_stack_impossible_l296_296837


namespace total_number_of_trees_l296_296156

theorem total_number_of_trees (side_length : ℝ) (area_ratio : ℝ) (trees_per_sqm : ℝ) (H : side_length = 100) (R : area_ratio = 3) (T : trees_per_sqm = 4) : 
  let street_area := side_length ^ 2 in 
  let forest_area := area_ratio * street_area in
  let total_trees := forest_area * trees_per_sqm in
  total_trees = 120000 :=
by
  -- proof steps go here
  sorry

end total_number_of_trees_l296_296156


namespace no_adjacent_same_roll_probability_l296_296950

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l296_296950


namespace icosahedron_inscribed_in_cube_l296_296259

theorem icosahedron_inscribed_in_cube (a m : ℝ) (points_on_faces : Fin 6 → Fin 2 → ℝ × ℝ × ℝ) :
  (∃ points : Fin 12 → ℝ × ℝ × ℝ, 
   (∀ i : Fin 12, ∃ j : Fin 6, (points i).fst = (points_on_faces j 0).fst ∨ (points i).fst = (points_on_faces j 1).fst) ∧
   ∃ segments : Fin 12 → Fin 12 → ℝ, 
   (∀ i j : Fin 12, (segments i j) = m ∨ (segments i j) = a)) →
  a^2 - a*m - m^2 = 0 := sorry

end icosahedron_inscribed_in_cube_l296_296259


namespace smallest_N_divisibility_l296_296092

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end smallest_N_divisibility_l296_296092


namespace gumball_machine_l296_296354

variable (R B G Y O : ℕ)

theorem gumball_machine : 
  (B = (1 / 2) * R) ∧
  (G = 4 * B) ∧
  (Y = (7 / 2) * B) ∧
  (O = (2 / 3) * (R + B)) ∧
  (R = (3 / 2) * Y) ∧
  (Y = 24) →
  (R + B + G + Y + O = 186) :=
sorry

end gumball_machine_l296_296354


namespace zander_stickers_l296_296625

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end zander_stickers_l296_296625


namespace find_k_l296_296604

-- Given: The polynomial x^2 - 3k * x * y - 3y^2 + 6 * x * y - 8
-- We want to prove the value of k such that the polynomial does not contain the term "xy".

theorem find_k (k : ℝ) : 
  (∀ x y : ℝ, (x^2 - 3 * k * x * y - 3 * y^2 + 6 * x * y - 8) = x^2 - 3 * y^2 - 8) → 
  k = 2 := 
by
  intro h
  have h_coeff := h 1 1
  -- We should observe that the polynomial should not contain the xy term
  sorry

end find_k_l296_296604


namespace min_distance_of_complex_numbers_l296_296431

open Complex

theorem min_distance_of_complex_numbers
  (z w : ℂ)
  (h₁ : abs (z + 1 + 3 * Complex.I) = 1)
  (h₂ : abs (w - 7 - 8 * Complex.I) = 3) :
  ∃ d, d = Real.sqrt 185 - 4 ∧ ∀ Z W : ℂ, abs (Z + 1 + 3 * Complex.I) = 1 → abs (W - 7 - 8 * Complex.I) = 3 → abs (Z - W) ≥ d :=
sorry

end min_distance_of_complex_numbers_l296_296431


namespace divide_by_repeating_decimal_l296_296287

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l296_296287


namespace remainder_17_pow_63_div_7_l296_296618

theorem remainder_17_pow_63_div_7 :
  (17^63) % 7 = 6 :=
by
  have h: 17 % 7 = 3 := rfl
  sorry

end remainder_17_pow_63_div_7_l296_296618


namespace curves_intersection_length_of_chord_l296_296019

-- Definitions of C1
def curve_C1_x (t : ℝ) := 2 * t - 1
def curve_C1_y (t : ℝ) := -4 * t + 3

-- Polar form of C2
def curve_C2_polar (ρ θ : ℝ) := ρ = 2 * Real.sqrt 2 * Real.cos (π / 4 - θ)

-- Auxiliary definitions and theorems
def cartesian_equation_C1 (x y : ℝ) := 2 * x + y - 1 = 0
def cartesian_equation_C2 (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 2

-- Distance formula
def distance_center_to_line (cx cy : ℝ) := (abs (2 * cx + cy - 1)) / Real.sqrt (2^2 + 1^2)

-- Length of chord formula
def chord_length (R d : ℝ) := 2 * Real.sqrt (R^2 - d^2)

-- Prove the main problem
theorem curves_intersection_length_of_chord :
  (∀ t : ℝ, cartesian_equation_C1 (curve_C1_x t) (curve_C1_y t)) ∧
  ( ∀ ρ θ : ℝ, curve_C2_polar ρ θ → cartesian_equation_C2 (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ( let d := distance_center_to_line 1 1 in Cartesian_equation_C2 (1:ℝ) (1:ℝ) = (1:ℝ)^2 + (2 - 2)*1^2) ∧
    d < Real.sqrt 2 ∧
    chord_length (Real.sqrt 2) d = 2 * sqrt (2 - 2/5) := 2 * sqrt(2 - 4/5) := sorry

end curves_intersection_length_of_chord_l296_296019


namespace toys_secured_in_25_minutes_l296_296255

def net_toy_gain_per_minute (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  toys_mom_puts - toys_mia_takes

def total_minutes (total_toys : ℕ) (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  (total_toys - 1) / net_toy_gain_per_minute toys_mom_puts toys_mia_takes + 1

theorem toys_secured_in_25_minutes :
  total_minutes 50 5 3 = 25 :=
by
  sorry

end toys_secured_in_25_minutes_l296_296255


namespace remaining_pie_proportion_l296_296083

def carlos_portion : ℝ := 0.6
def maria_share_of_remainder : ℝ := 0.25

theorem remaining_pie_proportion: 
  (1 - carlos_portion) - maria_share_of_remainder * (1 - carlos_portion) = 0.3 := 
by
  -- proof to be implemented here
  sorry

end remaining_pie_proportion_l296_296083


namespace correct_sampling_methods_l296_296840

-- Defining the conditions
def high_income_families : ℕ := 50
def middle_income_families : ℕ := 300
def low_income_families : ℕ := 150
def total_residents : ℕ := 500
def sample_size : ℕ := 100
def worker_group_size : ℕ := 10
def selected_workers : ℕ := 3

-- Definitions of sampling methods
inductive SamplingMethod
| random
| systematic
| stratified

open SamplingMethod

-- Problem statement in Lean 4
theorem correct_sampling_methods :
  (total_residents = high_income_families + middle_income_families + low_income_families) →
  (sample_size = 100) →
  (worker_group_size = 10) →
  (selected_workers = 3) →
  (chosen_method_for_task1 = SamplingMethod.stratified) →
  (chosen_method_for_task2 = SamplingMethod.random) →
  (chosen_method_for_task1, chosen_method_for_task2) = (SamplingMethod.stratified, SamplingMethod.random) :=
by
  intros
  sorry -- Proof to be filled in

end correct_sampling_methods_l296_296840


namespace total_trees_in_forest_l296_296158

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l296_296158


namespace exceeded_by_600_l296_296590

noncomputable def ken_collected : ℕ := 600
noncomputable def mary_collected (ken : ℕ) : ℕ := 5 * ken
noncomputable def scott_collected (mary : ℕ) : ℕ := mary / 3
noncomputable def total_collected (ken mary scott : ℕ) : ℕ := ken + mary + scott
noncomputable def goal : ℕ := 4000
noncomputable def exceeded_goal (total goal : ℕ) : ℕ := total - goal

theorem exceeded_by_600 : exceeded_goal (total_collected ken_collected (mary_collected ken_collected) (scott_collected (mary_collected ken_collected))) goal = 600 := by
  sorry

end exceeded_by_600_l296_296590


namespace possible_values_of_k_l296_296697

theorem possible_values_of_k (k : ℕ) (N : ℕ) (h₁ : (k * (k + 1)) / 2 = N^2) (h₂ : N < 100) :
  k = 1 ∨ k = 8 ∨ k = 49 :=
sorry

end possible_values_of_k_l296_296697


namespace factorization_problem_l296_296273

theorem factorization_problem (a b c : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 7 * x + 12 = (x + a) * (x + b))
  (h2 : ∀ x : ℝ, x^2 - 8 * x - 20 = (x - b) * (x - c)) :
  a - b + c = -9 :=
sorry

end factorization_problem_l296_296273


namespace max_b_of_box_volume_l296_296456

theorem max_b_of_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : Prime c) (h5 : a * b * c = 360) : b = 12 := 
sorry

end max_b_of_box_volume_l296_296456


namespace length_PR_l296_296868

noncomputable def circle_radius : ℝ := 10
noncomputable def distance_PQ : ℝ := 12
noncomputable def midpoint_minor_arc_length_PR : ℝ :=
  let PS : ℝ := distance_PQ / 2
  let OS : ℝ := Real.sqrt (circle_radius^2 - PS^2)
  let RS : ℝ := circle_radius - OS
  Real.sqrt (PS^2 + RS^2)

theorem length_PR :
  midpoint_minor_arc_length_PR = 2 * Real.sqrt 10 :=
by
  sorry

end length_PR_l296_296868


namespace find_boys_l296_296231

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end find_boys_l296_296231


namespace sum_of_roots_l296_296036

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) →
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) →
  a + b + c + d = 8 :=
by
  sorry

end sum_of_roots_l296_296036


namespace probability_reach_target_in_six_or_fewer_steps_l296_296031

theorem probability_reach_target_in_six_or_fewer_steps :
  let q := (21 : ℚ) / 1024
  ∃ x y : ℕ, Nat.Coprime x y ∧ q = x / y ∧ x + y = 1045 :=
by
  let q := (21 : ℚ) / 1024
  use (21, 1024)
  have h_coprime : Nat.Coprime 21 1024 := by
    sorry
  have h_q : q = 21 / 1024 := by
    sorry
  exact ⟨h_coprime, h_q, rfl⟩

end probability_reach_target_in_six_or_fewer_steps_l296_296031


namespace sum_of_integral_c_l296_296204

theorem sum_of_integral_c :
  let discriminant (a b c : ℤ) := b * b - 4 * a * c
  ∃ (valid_c : List ℤ),
    (∀ c ∈ valid_c, c ≤ 30 ∧ ∃ k : ℤ, discriminant 1 (-9) (c) = k * k ∧ k > 0) ∧
    valid_c.sum = 32 := 
by
  sorry

end sum_of_integral_c_l296_296204


namespace modular_inverse_of_35_mod_36_l296_296202

theorem modular_inverse_of_35_mod_36 : 
  ∃ a : ℤ, (35 * a) % 36 = 1 % 36 ∧ a = 35 := 
by 
  sorry

end modular_inverse_of_35_mod_36_l296_296202


namespace f_positive_l296_296112

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

variables (x0 x1 : ℝ)

theorem f_positive (hx0 : f x0 = 0) (hx1 : 0 < x1) (hx0_gt_x1 : x1 < x0) : 0 < f x1 :=
sorry

end f_positive_l296_296112


namespace percent_of_x_is_z_l296_296060

theorem percent_of_x_is_z (x y z : ℝ) (h1 : 0.45 * z = 1.2 * y) (h2 : y = 0.75 * x) : z = 2 * x :=
by
  sorry

end percent_of_x_is_z_l296_296060


namespace inequality_transformation_l296_296554

variable (x y : ℝ)

theorem inequality_transformation (h : x > y) : x - 2 > y - 2 :=
by
  sorry

end inequality_transformation_l296_296554


namespace percentage_income_diff_l296_296414

variable (A B : ℝ)

-- Condition that B's income is 33.33333333333333% greater than A's income
def income_relation (A B : ℝ) : Prop :=
  B = (4 / 3) * A

-- Proof statement to show that A's income is 25% less than B's income
theorem percentage_income_diff : 
  income_relation A B → 
  ((B - A) / B) * 100 = 25 :=
by
  intros h
  rw [income_relation] at h
  sorry

end percentage_income_diff_l296_296414


namespace average_of_data_set_is_five_l296_296675

def data_set : List ℕ := [2, 5, 5, 6, 7]

def sum_of_data_set : ℕ := data_set.sum
def count_of_data_set : ℕ := data_set.length

theorem average_of_data_set_is_five :
  (sum_of_data_set / count_of_data_set) = 5 :=
by
  sorry

end average_of_data_set_is_five_l296_296675


namespace intersection_complement_l296_296148

open Set

noncomputable def U : Set ℝ := univ

def A : Set ℝ := {x | x^2 - 2 * x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_complement (x : ℝ) :
  x ∈ (A ∩ (U \ B)) ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end intersection_complement_l296_296148


namespace ways_to_sum_2022_l296_296411

theorem ways_to_sum_2022 : 
  ∃ n : ℕ, (∀ a b : ℕ, (2022 = 2 * a + 3 * b) ∧ n = (b - a) / 4 ∧ n = 338) := 
sorry

end ways_to_sum_2022_l296_296411


namespace orchard_apples_relation_l296_296066

/-- 
A certain orchard has 10 apple trees, and on average each tree can produce 200 apples. 
Based on experience, for each additional tree planted, the average number of apples produced per tree decreases by 5. 
We are to show that if the orchard has planted x additional apple trees and the total number of apples is y, then the relationship between y and x is:
y = (10 + x) * (200 - 5x)
-/
theorem orchard_apples_relation (x : ℕ) (y : ℕ) 
    (initial_trees : ℕ := 10)
    (initial_apples : ℕ := 200)
    (decrease_per_tree : ℕ := 5)
    (total_trees := initial_trees + x)
    (average_apples := initial_apples - decrease_per_tree * x)
    (total_apples := total_trees * average_apples) :
    y = total_trees * average_apples := 
  by 
    sorry

end orchard_apples_relation_l296_296066


namespace reading_ratio_l296_296564

theorem reading_ratio (x : ℕ) (h1 : 10 * x + 5 * (75 - x) = 500) : 
  (10 * x) / 500 = 1 / 2 :=
by sorry

end reading_ratio_l296_296564


namespace reams_paper_l296_296407

theorem reams_paper (total_reams reams_haley reams_sister : Nat) 
    (h1 : total_reams = 5)
    (h2 : reams_haley = 2)
    (h3 : total_reams = reams_haley + reams_sister) : 
    reams_sister = 3 := by
  sorry

end reams_paper_l296_296407


namespace range_of_y_eq_x_squared_l296_296685

noncomputable def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem range_of_y_eq_x_squared :
  M = { y : ℝ | ∃ x : ℝ, y = x^2 } := by
  sorry

end range_of_y_eq_x_squared_l296_296685


namespace toothpicks_stage_20_l296_296169

-- Definition of the toothpick sequence
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 3 + 3 * (n - 1)

-- Theorem statement
theorem toothpicks_stage_20 : toothpicks 20 = 60 := by
  sorry

end toothpicks_stage_20_l296_296169


namespace rigged_coin_probability_l296_296758

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1 / 2) (h2 : 20 * (p ^ 3) * ((1 - p) ^ 3) = 1 / 12) :
  p = (1 - Real.sqrt 0.86) / 2 :=
by
  sorry

end rigged_coin_probability_l296_296758


namespace volume_at_target_temperature_l296_296098

-- Volume expansion relationship
def volume_change_per_degree_rise (ΔT V_real : ℝ) : Prop :=
  ΔT = 2 ∧ V_real = 3

-- Initial conditions
def initial_conditions (V_initial T_initial : ℝ) : Prop :=
  V_initial = 36 ∧ T_initial = 30

-- Target temperature
def target_temperature (T_target : ℝ) : Prop :=
  T_target = 20

-- Theorem stating the volume at the target temperature
theorem volume_at_target_temperature (ΔT V_real T_initial V_initial T_target V_target : ℝ) 
  (h_rel : volume_change_per_degree_rise ΔT V_real)
  (h_init : initial_conditions V_initial T_initial)
  (h_target : target_temperature T_target) :
  V_target = V_initial + V_real * ((T_target - T_initial) / ΔT) :=
by
  -- Insert proof here
  sorry

end volume_at_target_temperature_l296_296098


namespace sum_of_integral_values_l296_296203

theorem sum_of_integral_values (h1 : ∀ (x y c : ℤ), y = x^2 - 9 * x - c → y = 0 → ∃ r : ℚ, ∃ s : ℚ, r + s = 9 ∧ r * s = c)
    (h2 : ∀ (c : ℤ), (∃ k : ℤ, 81 + 4 * c = k^2 ∧ k^2 ≡ 1 [MOD 4]) ↔ ∃ k : ℤ, 81 + 4 * c = k^2 ∧ k % 2 = 1 ) :
    ∑ c in { c : ℤ | -20 ≤ c ∧ c ≤ 30 ∧ ∃ k : ℤ, 81 + 4 * c = k^2 ∧ k % 2 = 1 }, c = 32 :=
by {
  -- Proof omitted
  sorry
}

end sum_of_integral_values_l296_296203


namespace find_x_and_y_l296_296561

variable {x y : ℝ}

-- Given condition
def angleDCE : ℝ := 58

-- Proof statements
theorem find_x_and_y : x = 180 - angleDCE ∧ y = 180 - angleDCE := by
  sorry

end find_x_and_y_l296_296561


namespace silvia_last_play_without_breach_l296_296126

theorem silvia_last_play_without_breach (N : ℕ) : 
  36 * N < 2000 ∧ 72 * N ≥ 2000 ↔ N = 28 :=
by
  sorry

end silvia_last_play_without_breach_l296_296126


namespace additional_distance_to_achieve_target_average_speed_l296_296371

-- Given conditions
def initial_distance : ℕ := 20
def initial_speed : ℕ := 40
def target_average_speed : ℕ := 55

-- Prove that the additional distance required to average target speed is 90 miles
theorem additional_distance_to_achieve_target_average_speed 
  (total_distance : ℕ) 
  (total_time : ℚ) 
  (additional_distance : ℕ) 
  (additional_speed : ℕ) :
  total_distance = initial_distance + additional_distance →
  total_time = (initial_distance / initial_speed) + (additional_distance / additional_speed) →
  additional_speed = 60 →
  total_distance / total_time = target_average_speed →
  additional_distance = 90 :=
by 
  sorry

end additional_distance_to_achieve_target_average_speed_l296_296371


namespace equivalent_angle_l296_296596

theorem equivalent_angle (theta : ℤ) (k : ℤ) : 
  (∃ k : ℤ, (-525 + k * 360 = 195)) :=
by
  sorry

end equivalent_angle_l296_296596


namespace find_original_price_l296_296904

-- Define the original price and conditions
def original_price (P : ℝ) : Prop :=
  ∃ discount final_price, discount = 0.55 ∧ final_price = 450000 ∧ ((1 - discount) * P = final_price)

-- The theorem to prove the original price before discount
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1000000 :=
by
  sorry

end find_original_price_l296_296904


namespace area_of_park_l296_296886

variable (length breadth speed time perimeter area : ℕ)

axiom ratio_length_breadth : length = breadth / 4
axiom speed_kmh : speed = 12 * 1000 / 60 -- speed in m/min
axiom time_taken : time = 8 -- time in minutes
axiom perimeter_eq : perimeter = speed * time -- perimeter in meters
axiom length_breadth_relation : perimeter = 2 * (length + breadth)

theorem area_of_park : ∃ length breadth, (length = 160 ∧ breadth = 640 ∧ area = length * breadth ∧ area = 102400) :=
by
  sorry

end area_of_park_l296_296886


namespace quadratic_inequalities_l296_296104

variable (c x₁ y₁ y₂ y₃ : ℝ)
noncomputable def quadratic_function := -x₁^2 + 2*x₁ + c

theorem quadratic_inequalities
  (h_c : c < 0)
  (h_y₁ : quadratic_function c x₁ > 0)
  (h_y₂ : y₂ = quadratic_function c (x₁ - 2))
  (h_y₃ : y₃ = quadratic_function c (x₁ + 2)) :
  y₂ < 0 ∧ y₃ < 0 :=
by sorry

end quadratic_inequalities_l296_296104


namespace negation_of_P_l296_296962

variable (x : ℝ)

def P : Prop := ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0

theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 3 < 0 :=
by sorry

end negation_of_P_l296_296962


namespace red_button_probability_l296_296990

/-
Mathematical definitions derived from the problem:
Initial setup:
- Jar A has 6 red buttons and 10 blue buttons.
- Same number of red and blue buttons are removed. Jar A retains 3/4 of original buttons.
- Calculate the final number of red buttons in Jar A and B, and determine the probability both selected buttons are red.
-/
theorem red_button_probability :
  let initial_red := 6
  let initial_blue := 10
  let total_buttons := initial_red + initial_blue
  let removal_fraction := 3 / 4
  let final_buttons := (3 / 4 : ℚ) * total_buttons
  let removed_buttons := total_buttons - final_buttons
  let removed_each_color := removed_buttons / 2
  let final_red_A := initial_red - removed_each_color
  let final_red_B := removed_each_color
  let prob_red_A := final_red_A / final_buttons
  let prob_red_B := final_red_B / removed_buttons
  prob_red_A * prob_red_B = 1 / 6 :=
by
  sorry

end red_button_probability_l296_296990


namespace possible_values_of_a_l296_296827

open Real

noncomputable def f (a : ℕ) (x : ℝ) : ℝ := log x + a / (x + 1)

theorem possible_values_of_a : ∀ a : ℕ,
  (∃ x : ℝ, (1 < x ∧ x < 3) ∧ deriv (f a) x = 0) →
  a = 5 := by
  sorry

end possible_values_of_a_l296_296827


namespace range_of_m_l296_296816

noncomputable def proposition (m : ℝ) : Prop := ∀ x : ℝ, 4^x - 2^(x + 1) + m = 0

theorem range_of_m (m : ℝ) (h : ¬¬proposition m) : m ≤ 1 :=
by
  sorry

end range_of_m_l296_296816


namespace daniel_total_worth_l296_296647

theorem daniel_total_worth
    (sales_tax_paid : ℝ)
    (sales_tax_rate : ℝ)
    (cost_tax_free_items : ℝ)
    (tax_rate_pos : 0 < sales_tax_rate) :
    sales_tax_paid = 0.30 →
    sales_tax_rate = 0.05 →
    cost_tax_free_items = 18.7 →
    ∃ (x : ℝ), 0.05 * x = 0.30 ∧ (x + cost_tax_free_items = 24.7) := by
    sorry

end daniel_total_worth_l296_296647


namespace excircle_tangent_segment_length_l296_296676

theorem excircle_tangent_segment_length (A B C M : ℝ) 
  (h1 : A + B + C = 1) 
  (h2 : M = (1 / 2)) : 
  M = 1 / 2 := 
  by
    -- This is where the proof would go
    sorry

end excircle_tangent_segment_length_l296_296676


namespace compl_union_eq_l296_296004

-- Definitions
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

-- The statement
theorem compl_union_eq : (Aᶜ ∩ U) ∪ B = {2, 4, 5, 6} :=
by sorry

end compl_union_eq_l296_296004


namespace polynomial_roots_power_sum_l296_296412

theorem polynomial_roots_power_sum {a b c : ℝ}
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 6)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 :=
by
  sorry

end polynomial_roots_power_sum_l296_296412


namespace proof_inequality_l296_296998

variable {a b c d : ℝ}

theorem proof_inequality (h1 : a + b + c + d = 6) (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
sorry

end proof_inequality_l296_296998


namespace probability_sum_even_is_five_over_eleven_l296_296052

noncomputable def probability_even_sum : ℚ :=
  let totalBalls := 12
  let totalWays := totalBalls * (totalBalls - 1)
  let evenBalls := 6
  let oddBalls := 6
  let evenWays := evenBalls * (evenBalls - 1)
  let oddWays := oddBalls * (oddBalls - 1)
  let totalEvenWays := evenWays + oddWays
  totalEvenWays / totalWays

theorem probability_sum_even_is_five_over_eleven : probability_even_sum = 5 / 11 := sorry

end probability_sum_even_is_five_over_eleven_l296_296052


namespace Monica_saved_per_week_l296_296718

theorem Monica_saved_per_week(amount_per_cycle : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_saved : ℕ) :
  num_cycles = 5 →
  weeks_per_cycle = 60 →
  (amount_per_cycle * num_cycles) = total_saved →
  total_saved = 4500 →
  total_saved / (weeks_per_cycle * num_cycles) = 75 := 
by
  intros
  sorry

end Monica_saved_per_week_l296_296718


namespace stickers_given_l296_296624

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end stickers_given_l296_296624


namespace generating_sets_Z2_l296_296514

theorem generating_sets_Z2 (a b : ℤ × ℤ) (h : Submodule.span ℤ ({a, b} : Set (ℤ × ℤ)) = ⊤) :
  let a₁ := a.1
  let a₂ := a.2
  let b₁ := b.1
  let b₂ := b.2
  a₁ * b₂ - a₂ * b₁ = 1 ∨ a₁ * b₂ - a₂ * b₁ = -1 := 
by
  sorry

end generating_sets_Z2_l296_296514


namespace trig_identity_l296_296937

open Real

theorem trig_identity : sin (20 * (π / 180)) * cos (10 * (π / 180)) - cos (200 * (π / 180)) * sin (10 * (π / 180)) = 1 / 2 := 
by
  sorry

end trig_identity_l296_296937


namespace total_trees_in_forest_l296_296157

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l296_296157


namespace remainder_theorem_div_l296_296854

noncomputable
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

theorem remainder_theorem_div (A B C : ℝ) (h : p A B C 2 = 13) : p A B C (-2) = 13 :=
by
  -- Proof goes here
  sorry

end remainder_theorem_div_l296_296854


namespace fraction_product_l296_296362

theorem fraction_product : (1 / 2) * (1 / 3) * (1 / 6) * 120 = 10 / 3 :=
by
  sorry

end fraction_product_l296_296362


namespace mows_in_summer_l296_296711

theorem mows_in_summer (S : ℕ) (h1 : 8 - S = 3) : S = 5 :=
sorry

end mows_in_summer_l296_296711


namespace trigonometric_identity_l296_296873

open Real

theorem trigonometric_identity :
  (sin (15 * pi / 180) + sin (25 * pi / 180) + sin (35 * pi / 180) + 
   sin (45 * pi / 180) + sin (55 * pi / 180) + sin (65 * pi / 180) + 
   sin (75 * pi / 180) + sin (85 * pi / 180)) / 
  (cos (10 * pi / 180) * cos (15 * pi / 180) * cos (25 * pi / 180)) = 8 := 
sorry

end trigonometric_identity_l296_296873


namespace sufficient_not_necessary_condition_l296_296009

variables (a b : Line) (α β : Plane)

def Line : Type := sorry
def Plane : Type := sorry

-- Conditions: a and b are different lines, α and β are different planes
axiom diff_lines : a ≠ b
axiom diff_planes : α ≠ β

-- Perpendicular and parallel definitions
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Sufficient but not necessary condition
theorem sufficient_not_necessary_condition
  (h1 : perp a β)
  (h2 : parallel α β) :
  perp a α :=
sorry

end sufficient_not_necessary_condition_l296_296009


namespace max_area_triangle_max_area_quadrilateral_l296_296546

-- Define the terms and conditions

variables {A O : Point}
variables {r d : ℝ}
variables {C D B : Point}

-- Problem (a)
theorem max_area_triangle (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (3 / 4) * d) :=
sorry

-- Problem (b)
theorem max_area_quadrilateral (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (1 / 2) * d) :=
sorry

end max_area_triangle_max_area_quadrilateral_l296_296546


namespace emma_prob_at_least_one_correct_l296_296239

-- Define the probability of getting a question wrong
def prob_wrong : ℚ := 4 / 5

-- Define the probability of getting all five questions wrong
def prob_all_wrong : ℚ := prob_wrong ^ 5

-- Define the probability of getting at least one question correct
def prob_at_least_one_correct : ℚ := 1 - prob_all_wrong

-- Define the main theorem to be proved
theorem emma_prob_at_least_one_correct : prob_at_least_one_correct = 2101 / 3125 := by
  sorry  -- This is where the proof would go

end emma_prob_at_least_one_correct_l296_296239


namespace cuboid_dimensions_sum_l296_296348

theorem cuboid_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 45) 
  (h2 : B * C = 80) 
  (h3 : C * A = 180) : 
  A + B + C = 145 / 9 :=
sorry

end cuboid_dimensions_sum_l296_296348


namespace similar_triangle_shortest_side_l296_296767

theorem similar_triangle_shortest_side (a b c : ℕ) (H1 : a^2 + b^2 = c^2) (H2 : a = 15) (H3 : c = 34) (H4 : b = Int.sqrt 931) : 
  ∃ d : ℝ, d = 3 * Int.sqrt 931 ∧ d = 102  :=
by
  sorry

end similar_triangle_shortest_side_l296_296767


namespace num_valid_n_l296_296387

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end num_valid_n_l296_296387


namespace circle_radius_l296_296161

theorem circle_radius (C : ℝ) (r : ℝ) (h1 : C = 72 * Real.pi) (h2 : C = 2 * Real.pi * r) : r = 36 :=
by
  sorry

end circle_radius_l296_296161


namespace kanul_total_amount_l296_296062

theorem kanul_total_amount (T : ℝ) (h1 : 35000 + 40000 + 0.2 * T = T) : T = 93750 := 
by
  sorry

end kanul_total_amount_l296_296062


namespace polynomial_coefficient_sum_l296_296797

theorem polynomial_coefficient_sum :
  ∀ (a0 a1 a2 a3 a4 a5 : ℤ), 
  (3 - 2 * x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 → 
  a0 + a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 = 233 :=
by
  sorry

end polynomial_coefficient_sum_l296_296797


namespace divide_by_10_result_l296_296256

theorem divide_by_10_result (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end divide_by_10_result_l296_296256


namespace divide_by_repeating_decimal_l296_296309

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l296_296309


namespace min_percentage_both_physics_chemistry_l296_296930

/--
Given:
- A certain school conducted a survey.
- 68% of the students like physics.
- 72% of the students like chemistry.

Prove that the minimum percentage of students who like both physics and chemistry is 40%.
-/
theorem min_percentage_both_physics_chemistry (P C : ℝ)
(hP : P = 0.68) (hC : C = 0.72) :
  ∃ B, B = P + C - 1 ∧ B = 0.40 :=
by
  sorry

end min_percentage_both_physics_chemistry_l296_296930


namespace min_deg_g_correct_l296_296811

open Polynomial

noncomputable def min_deg_g {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  Nat :=
11

theorem min_deg_g_correct {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  (min_deg_g f g h hf hh h_eq = 11) :=
sorry

end min_deg_g_correct_l296_296811


namespace evaluate_f_g_l296_296852

def g (x : ℝ) : ℝ := 3 * x
def f (x : ℝ) : ℝ := x - 6

theorem evaluate_f_g :
  f (g 3) = 3 :=
by
  sorry

end evaluate_f_g_l296_296852


namespace min_value_reciprocal_l296_296573

theorem min_value_reciprocal (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  3 ≤ (1/a) + (1/b) + (1/c) :=
sorry

end min_value_reciprocal_l296_296573


namespace wrong_value_l296_296455

-- Definitions based on the conditions
def initial_mean : ℝ := 32
def corrected_mean : ℝ := 32.5
def num_observations : ℕ := 50
def correct_observation : ℝ := 48

-- We need to prove that the wrong value of the observation was 23
theorem wrong_value (sum_initial : ℝ) (sum_corrected : ℝ) : 
  sum_initial = num_observations * initial_mean ∧ 
  sum_corrected = num_observations * corrected_mean →
  48 - (sum_corrected - sum_initial) = 23 :=
by
  sorry

end wrong_value_l296_296455


namespace translation_vector_coords_l296_296465

-- Definitions according to the given conditions
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def translated_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Statement that we need to prove
theorem translation_vector_coords :
  ∃ (a b : ℝ), 
  (∀ x y : ℝ, original_circle x y ↔ translated_circle (x - a) (y - b)) ∧
  (a, b) = (-1, 2) := 
sorry

end translation_vector_coords_l296_296465


namespace eight_step_paths_board_l296_296760

theorem eight_step_paths_board (P Q : ℕ) (hP : P = 0) (hQ : Q = 7) : 
  ∃ (paths : ℕ), paths = 70 :=
by
  sorry

end eight_step_paths_board_l296_296760


namespace largest_final_number_l296_296906

-- Define the sequence and conditions
def initial_number := List.replicate 40 [3, 1, 1, 2, 3] |> List.join

-- The transformation rule
def valid_transform (a b : ℕ) : ℕ := if a + b <= 9 then a + b else 0

-- Sum of digits of a number
def sum_digits : List ℕ → ℕ := List.foldr (· + ·) 0

-- Define the final valid number pattern
def valid_final_pattern (n : ℕ) : Prop := n = 77

-- The main theorem statement
theorem largest_final_number (seq : List ℕ) (h_seq : seq = initial_number) :
  valid_final_pattern (sum_digits seq) := sorry

end largest_final_number_l296_296906


namespace range_of_m_l296_296688

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) : 
  ∀ m : ℝ, (x + 2 * y > m) ↔ (m < 8) :=
by 
  sorry

end range_of_m_l296_296688


namespace restore_temperature_time_l296_296588

theorem restore_temperature_time :
  let rate_increase := 8 -- degrees per hour
  let duration_increase := 3 -- hours
  let rate_decrease := 4 -- degrees per hour
  let total_increase := rate_increase * duration_increase
  let time := total_increase / rate_decrease
  time = 6 := 
by
  sorry

end restore_temperature_time_l296_296588


namespace four_digit_numbers_divisible_by_17_l296_296819

theorem four_digit_numbers_divisible_by_17 :
  ∃ n, (∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 17 = 0 ↔ ∃ k, x = 17 * k ∧ 59 ≤ k ∧ k ≤ 588) ∧ n = 530 := 
sorry

end four_digit_numbers_divisible_by_17_l296_296819


namespace geometric_sequence_cannot_determine_a3_l296_296113

/--
Suppose we have a geometric sequence {a_n} such that 
the product of the first five terms a_1 * a_2 * a_3 * a_4 * a_5 = 32.
We aim to show that the value of a_3 cannot be determined with the given information.
-/
theorem geometric_sequence_cannot_determine_a3 (a : ℕ → ℝ) (r : ℝ) (h : a 0 * a 1 * a 2 * a 3 * a 4 = 32) : 
  ¬ ∃ x : ℝ, a 2 = x :=
sorry

end geometric_sequence_cannot_determine_a3_l296_296113


namespace tan_five_pi_over_four_l296_296666

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l296_296666


namespace base_digit_difference_l296_296974

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l296_296974


namespace line_equation_through_P_and_intercepts_l296_296487

-- Define the conditions
structure Point (α : Type*) := 
  (x : α) 
  (y : α)

-- Given point P
def P : Point ℝ := ⟨5, 6⟩

-- Equation of a line passing through (x₀, y₀) and 
-- having the intercepts condition: the x-intercept is twice the y-intercept

theorem line_equation_through_P_and_intercepts :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (a * 5 + b * 6 + c = 0) ∧ 
   ((-c / a = 2 * (-c / b)) ∧ (c ≠ 0)) ∧
   (a = 1 ∧ b = 2 ∧ c = -17) ∨
   (a = 6 ∧ b = -5 ∧ c = 0)) :=
sorry

end line_equation_through_P_and_intercepts_l296_296487


namespace solve_inequalities_l296_296875

theorem solve_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ ((x / 2) + ((1 - 3 * x) / 4) ≤ 1) → -3 ≤ x ∧ x < 1 := 
by
  sorry

end solve_inequalities_l296_296875


namespace number_of_outfits_l296_296330

def shirts : ℕ := 5
def hats : ℕ := 3

theorem number_of_outfits : shirts * hats = 15 :=
by 
  -- This part intentionally left blank since no proof required.
  sorry

end number_of_outfits_l296_296330


namespace valid_rearrangements_count_l296_296409

noncomputable def count_valid_rearrangements : ℕ := sorry

theorem valid_rearrangements_count :
  count_valid_rearrangements = 7 :=
sorry

end valid_rearrangements_count_l296_296409


namespace missing_shirts_l296_296720

-- Definition of conditions
def pairs_of_trousers : ℕ := 10
def price_per_trousers : ℕ := 9
def total_cost : ℕ := 140
def price_per_shirt : ℕ := 5
def claimed_number_of_shirts : ℕ := 2

-- The theorem to be proved
theorem missing_shirts : 
  let cost_of_trousers := pairs_of_trousers * price_per_trousers,
      cost_of_shirts := total_cost - cost_of_trousers,
      number_of_shirts := cost_of_shirts / price_per_shirt,
      missing_shirts := number_of_shirts - claimed_number_of_shirts
  in
  missing_shirts = 8 :=
by
  sorry

end missing_shirts_l296_296720


namespace smallest_base_l296_296386

-- Definitions of the conditions
def condition1 (b : ℕ) : Prop := b > 3
def condition2 (b : ℕ) : Prop := b > 7
def condition3 (b : ℕ) : Prop := b > 6
def condition4 (b : ℕ) : Prop := b > 8

-- Main theorem statement
theorem smallest_base : ∀ b : ℕ, condition1 b ∧ condition2 b ∧ condition3 b ∧ condition4 b → b = 9 := by
  sorry

end smallest_base_l296_296386


namespace expression_value_l296_296907

theorem expression_value :
  ∀ (x y : ℚ), (x = -5/4) → (y = -3/2) → -2 * x - y^2 = 1/4 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end expression_value_l296_296907


namespace tan_A_of_triangle_conditions_l296_296124

open Real

def triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ B = π / 4

def form_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b^2 = a^2 + c^2

theorem tan_A_of_triangle_conditions
  (A B C a b c : ℝ)
  (h_angles : triangle_angles A B C)
  (h_seq : form_arithmetic_sequence a b c) :
  tan A = sqrt 2 - 1 :=
by
  sorry

end tan_A_of_triangle_conditions_l296_296124


namespace smaller_cube_size_l296_296759

theorem smaller_cube_size
  (original_cube_side : ℕ)
  (number_of_smaller_cubes : ℕ)
  (painted_cubes : ℕ)
  (unpainted_cubes : ℕ) :
  original_cube_side = 3 → 
  number_of_smaller_cubes = 27 → 
  painted_cubes = 26 → 
  unpainted_cubes = 1 →
  (∃ (side : ℕ), side = original_cube_side / 3 ∧ side = 1) :=
by
  intros h1 h2 h3 h4
  use 1
  have h : 1 = original_cube_side / 3 := sorry
  exact ⟨h, rfl⟩

end smaller_cube_size_l296_296759


namespace badges_before_exchange_l296_296174

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l296_296174


namespace sum_of_roots_quadratic_l296_296385

theorem sum_of_roots_quadratic (a b c : ℝ) (h_eq : a ≠ 0) (h_eqn : -48 * a * (a * 1) + 100 * a + 200 * a^2 = 0) : 
  - b / a = (25 : ℚ) / 12 :=
by
  have h1 : a = -48 := rfl
  have h2 : b = 100 := rfl
  sorry

end sum_of_roots_quadratic_l296_296385


namespace dessert_probability_l296_296632

noncomputable def P (e : Prop) : ℝ := sorry

variables (D C : Prop)

theorem dessert_probability 
  (P_D : P D = 0.6)
  (P_D_and_not_C : P (D ∧ ¬C) = 0.12) :
  P (¬ D) = 0.4 :=
by
  -- Proof is skipped using sorry, as instructed.
  sorry

end dessert_probability_l296_296632


namespace smallest_d_for_inverse_g_l296_296713

def g (x : ℝ) := (x - 3)^2 - 8

theorem smallest_d_for_inverse_g : ∃ d : ℝ, (∀ x y : ℝ, x ≠ y → x ≥ d → y ≥ d → g x ≠ g y) ∧ ∀ d' : ℝ, d' < 3 → ∃ x y : ℝ, x ≠ y ∧ x ≥ d' ∧ y ≥ d' ∧ g x = g y :=
by
  sorry

end smallest_d_for_inverse_g_l296_296713


namespace Dan_speed_must_exceed_45_mph_l296_296272

theorem Dan_speed_must_exceed_45_mph : 
  ∀ (distance speed_Cara time_lag time_required speed_Dan : ℝ),
    distance = 180 →
    speed_Cara = 30 →
    time_lag = 2 →
    time_required = 4 →
    (distance / speed_Cara) = 6 →
    (∀ t, t = distance / speed_Dan → t < time_required) →
    speed_Dan > 45 :=
by
  intro distance speed_Cara time_lag time_required speed_Dan
  intro h1 h2 h3 h4 h5 h6
  sorry

end Dan_speed_must_exceed_45_mph_l296_296272


namespace star_running_back_yardage_l296_296063

-- Definitions
def total_yardage : ℕ := 150
def catching_passes_yardage : ℕ := 60
def running_yardage (total_yardage catching_passes_yardage : ℕ) : ℕ :=
  total_yardage - catching_passes_yardage

-- Statement to prove
theorem star_running_back_yardage :
  running_yardage total_yardage catching_passes_yardage = 90 := 
sorry

end star_running_back_yardage_l296_296063


namespace solve_eq1_solve_eq2_l296_296448

theorem solve_eq1 (x : ℝ):
  (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2 :=
by 
  sorry

theorem solve_eq2 (x : ℝ):
  2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by 
  sorry

end solve_eq1_solve_eq2_l296_296448


namespace meal_cost_is_25_l296_296499

def total_cost_samosas : ℕ := 3 * 2
def total_cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2
def tip_percentage : ℝ := 0.25

def total_food_cost : ℕ := total_cost_samosas + total_cost_pakoras + cost_mango_lassi
def tip_amount : ℝ := total_food_cost * tip_percentage
def total_meal_cost : ℝ := total_food_cost + tip_amount

theorem meal_cost_is_25 : total_meal_cost = 25 := by
    sorry

end meal_cost_is_25_l296_296499


namespace zander_stickers_l296_296626

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end zander_stickers_l296_296626


namespace division_by_repeating_decimal_l296_296306

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l296_296306


namespace greatest_whole_number_inequality_l296_296671

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end greatest_whole_number_inequality_l296_296671


namespace Eight_div_by_repeating_decimal_0_3_l296_296314

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l296_296314


namespace coeff_fourth_term_expansion_l296_296649

theorem coeff_fourth_term_expansion :
  (3 : ℚ) ^ 2 * (-1 : ℚ) / 8 * (Nat.choose 8 3) = -63 :=
by
  sorry

end coeff_fourth_term_expansion_l296_296649


namespace total_peanuts_l296_296842

theorem total_peanuts :
  let jose_peanuts := 85
  let kenya_peanuts := jose_peanuts + 48
  let malachi_peanuts := kenya_peanuts + 35
  jose_peanuts + kenya_peanuts + malachi_peanuts = 386 :=
by
  let jose_peanuts := 85
  let kenya_peanuts := jose_peanuts + 48
  let malachi_peanuts := kenya_peanuts + 35
  calc
    jose_peanuts + kenya_peanuts + malachi_peanuts
      = 85 + (85 + 48) + ((85 + 48) + 35) : sorry
      ... = 386 : sorry

end total_peanuts_l296_296842


namespace zoo_children_count_l296_296353

theorem zoo_children_count:
  ∀ (C : ℕ), 
  (10 * C + 16 * 10 = 220) → 
  C = 6 :=
by
  intro C
  intro h
  sorry

end zoo_children_count_l296_296353


namespace sapling_height_relationship_l296_296073

-- Definition to state the conditions
def initial_height : ℕ := 100
def growth_per_year : ℕ := 50
def height_after_years (years : ℕ) : ℕ := initial_height + growth_per_year * years

-- The theorem statement that should be proved
theorem sapling_height_relationship (x : ℕ) : height_after_years x = 50 * x + 100 := 
by
  sorry

end sapling_height_relationship_l296_296073


namespace pints_of_cider_l296_296690

def pintCider (g : ℕ) (p : ℕ) : ℕ :=
  g / 20 + p / 40

def totalApples (f : ℕ) (h : ℕ) (a : ℕ) : ℕ :=
  f * h * a

theorem pints_of_cider (g p : ℕ) (farmhands : ℕ) (hours : ℕ) (apples_per_hour : ℕ)
  (H1 : g = 1)
  (H2 : p = 2)
  (H3 : farmhands = 6)
  (H4 : hours = 5)
  (H5 : apples_per_hour = 240) :
  pintCider (apples_per_hour * farmhands * hours / 3)
            (apples_per_hour * farmhands * hours * 2 / 3) = 120 :=
by
  sorry

end pints_of_cider_l296_296690


namespace unique_solution_k_l296_296509

theorem unique_solution_k (k : ℕ) (f : ℕ → ℕ) :
  (∀ n : ℕ, (Nat.iterate f n n) = n + k) → k = 0 :=
by
  sorry

end unique_solution_k_l296_296509


namespace original_price_of_books_l296_296472

theorem original_price_of_books (purchase_cost : ℝ) (original_price : ℝ) :
  (purchase_cost = 162) →
  (original_price ≤ 100) ∨ 
  (100 < original_price ∧ original_price ≤ 200 ∧ purchase_cost = original_price * 0.9) ∨ 
  (original_price > 200 ∧ purchase_cost = original_price * 0.8) →
  (original_price = 180 ∨ original_price = 202.5) :=
by
  sorry

end original_price_of_books_l296_296472


namespace MelAge_when_Katherine24_l296_296580

variable (Katherine Mel : ℕ)

-- Conditions
def isYounger (Mel Katherine : ℕ) : Prop :=
  Mel = Katherine - 3

def is24yearsOld (Katherine : ℕ) : Prop :=
  Katherine = 24

-- Statement to Prove
theorem MelAge_when_Katherine24 (Katherine Mel : ℕ) 
  (h1 : isYounger Mel Katherine) 
  (h2 : is24yearsOld Katherine) : 
  Mel = 21 := 
by 
  sorry

end MelAge_when_Katherine24_l296_296580


namespace rational_numbers_include_positives_and_negatives_l296_296077

theorem rational_numbers_include_positives_and_negatives :
  ∃ (r : ℚ), r > 0 ∧ ∃ (r' : ℚ), r' < 0 :=
by
  sorry

end rational_numbers_include_positives_and_negatives_l296_296077


namespace no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l296_296209

noncomputable def is_square_of_prime (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ m = p * p

theorem no_positive_integer_n_ge_2_1001_n_is_square_of_prime :
  ∀ n : ℕ, n ≥ 2 → ¬ is_square_of_prime (n^3 + 1) :=
by
  intro n hn
  sorry

end no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l296_296209


namespace bobs_corn_harvest_l296_296931

theorem bobs_corn_harvest : 
  let row1 := 82 // 8,
      row2 := 94 // 9,
      row3 := 78 // 7,
      row4 := 96 // 12,
      row5 := 85 // 10,
      row6 := 91 // 13,
      row7 := 88 // 11
  in 
  row1 + row2 + row3 + row4 + row5 + row6 + row7 = 62 :=
by 
  have h1 : ((82 : ℕ) // 8) = 10 := by sorry,
  have h2 : ((94 : ℕ) // 9) = 10 := by sorry,
  have h3 : ((78 : ℕ) // 7) = 11 := by sorry,
  have h4 : ((96 : ℕ) // 12) = 8 := by sorry,
  have h5 : ((85 : ℕ) // 10) = 8 := by sorry,
  have h6 : ((91 : ℕ) // 13) = 7 := by sorry,
  have h7 : ((88 : ℕ) // 11) = 8 := by sorry,
  calc
    row1 + row2 + row3 + row4 + row5 + row6 + row7
    _ = 10 + 10 + 11 + 8 + 8 + 7 + 8 := by rw [h1, h2, h3, h4, h5, h6, h7]
    _ = 62 : by norm_num

end bobs_corn_harvest_l296_296931


namespace apples_per_bucket_l296_296582

theorem apples_per_bucket (total_apples buckets : ℕ) (h1 : total_apples = 56) (h2 : buckets = 7) : 
  (total_apples / buckets) = 8 :=
by
  sorry

end apples_per_bucket_l296_296582


namespace meat_cost_per_pound_l296_296867

def total_cost_box : ℝ := 5
def cost_per_bell_pepper : ℝ := 1.5
def num_bell_peppers : ℝ := 4
def num_pounds_meat : ℝ := 2
def total_spent : ℝ := 17

theorem meat_cost_per_pound : total_spent - (total_cost_box + num_bell_peppers * cost_per_bell_pepper) = 6 -> 
                             6 / num_pounds_meat = 3 := by
  sorry

end meat_cost_per_pound_l296_296867


namespace solve_indeterminate_equation_l296_296812

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_indeterminate_equation (x y : ℕ) (hx : is_prime x) (hy : is_prime y) :
  x^2 - y^2 = x * y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
by
  sorry

end solve_indeterminate_equation_l296_296812


namespace total_earnings_l296_296781

theorem total_earnings :
  (15 * 2) + (12 * 1.5) = 48 := by
  sorry

end total_earnings_l296_296781


namespace sum_a_t_l296_296799

theorem sum_a_t (a : ℝ) (t : ℝ) 
  (h₁ : a = 6)
  (h₂ : t = a^2 - 1) : a + t = 41 :=
by
  sorry

end sum_a_t_l296_296799


namespace eight_div_repeating_three_l296_296301

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l296_296301


namespace f_increasing_on_neg_inf_to_one_l296_296814

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 8

theorem f_increasing_on_neg_inf_to_one :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end f_increasing_on_neg_inf_to_one_l296_296814


namespace fourth_angle_of_quadrilateral_l296_296422

theorem fourth_angle_of_quadrilateral (A : ℝ) : 
  (120 + 85 + 90 + A = 360) ↔ A = 65 := 
by
  sorry

end fourth_angle_of_quadrilateral_l296_296422


namespace total_selling_price_correct_l296_296768

-- Define the conditions
def metres_of_cloth : ℕ := 500
def loss_per_metre : ℕ := 5
def cost_price_per_metre : ℕ := 41
def selling_price_per_metre : ℕ := cost_price_per_metre - loss_per_metre
def expected_total_selling_price : ℕ := 18000

-- Define the theorem
theorem total_selling_price_correct : 
  selling_price_per_metre * metres_of_cloth = expected_total_selling_price := 
by
  sorry

end total_selling_price_correct_l296_296768
