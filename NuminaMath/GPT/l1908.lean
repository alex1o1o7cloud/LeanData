import Mathlib

namespace solve_for_x_l1908_190897

theorem solve_for_x (x : ℕ) (h : 2^12 = 64^x) : x = 2 :=
by {
  sorry
}

end solve_for_x_l1908_190897


namespace C_gets_more_than_D_by_500_l1908_190850

-- Definitions based on conditions
def proportionA := 5
def proportionB := 2
def proportionC := 4
def proportionD := 3

def totalProportion := proportionA + proportionB + proportionC + proportionD

def A_share := 2500
def totalMoney := A_share * (totalProportion / proportionA)

def C_share := (proportionC / totalProportion) * totalMoney
def D_share := (proportionD / totalProportion) * totalMoney

-- The theorem stating the final question
theorem C_gets_more_than_D_by_500 : C_share - D_share = 500 := by
  sorry

end C_gets_more_than_D_by_500_l1908_190850


namespace remainder_correct_l1908_190865

def dividend : ℝ := 13787
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89
def remainder : ℝ := dividend - (divisor * quotient)

theorem remainder_correct: remainder = 14 := by
  -- Proof goes here
  sorry

end remainder_correct_l1908_190865


namespace smallest_positive_m_l1908_190880

theorem smallest_positive_m (m : ℕ) :
  (∃ (r s : ℤ), 18 * r * s = 252 ∧ m = 18 * (r + s) ∧ r ≠ s) ∧ m > 0 →
  m = 162 := 
sorry

end smallest_positive_m_l1908_190880


namespace expression_not_equal_l1908_190851

theorem expression_not_equal :
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  e2 ≠ product :=
by
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  sorry

end expression_not_equal_l1908_190851


namespace find_ab_l1908_190872

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l1908_190872


namespace ratio_of_chickens_in_run_to_coop_l1908_190877

def chickens_in_coop : ℕ := 14
def free_ranging_chickens : ℕ := 52
def run_condition (R : ℕ) : Prop := 2 * R - 4 = 52

theorem ratio_of_chickens_in_run_to_coop (R : ℕ) (hR : run_condition R) :
  R / chickens_in_coop = 2 :=
by
  sorry

end ratio_of_chickens_in_run_to_coop_l1908_190877


namespace triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l1908_190881

theorem triangle_angle_tangent_ratio (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c) :
  Real.tan A / Real.tan B = 4 := sorry

theorem triangle_tan_A_minus_B_maximum (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c)
  (h2 : Real.tan A / Real.tan B = 4) : Real.tan (A - B) ≤ 3 / 4 := sorry

end triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l1908_190881


namespace negation_of_prop_original_l1908_190824

-- Definitions and conditions as per the problem
def prop_original : Prop :=
  ∃ x : ℝ, x^2 + x + 1 ≤ 0

def prop_negation : Prop :=
  ∀ x : ℝ, x^2 + x + 1 > 0

-- The theorem states the mathematical equivalence
theorem negation_of_prop_original : ¬ prop_original ↔ prop_negation := 
sorry

end negation_of_prop_original_l1908_190824


namespace find_f_three_l1908_190863

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def f_condition (f : ℝ → ℝ) := ∀ x : ℝ, x < 0 → f x = (1/2)^x

theorem find_f_three (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f_condition f) : f 3 = -8 :=
sorry

end find_f_three_l1908_190863


namespace parabola_focus_distance_l1908_190818

theorem parabola_focus_distance (p m : ℝ) (h1 : p > 0) (h2 : (2 - (-p/2)) = 4) : p = 4 := 
by
  sorry

end parabola_focus_distance_l1908_190818


namespace total_distance_AD_l1908_190874

theorem total_distance_AD :
  let d_AB := 100
  let d_BC := d_AB + 50
  let d_CD := 2 * d_BC
  d_AB + d_BC + d_CD = 550 := by
  sorry

end total_distance_AD_l1908_190874


namespace binomial_expansion_a0_a1_a3_a5_l1908_190859

theorem binomial_expansion_a0_a1_a3_a5 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h : (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_0 + a_1 + a_3 + a_5 = 123 :=
sorry

end binomial_expansion_a0_a1_a3_a5_l1908_190859


namespace distance_between_points_l1908_190864

open Real

theorem distance_between_points : 
  let p1 := (2, 2)
  let p2 := (5, 9)
  dist (p1 : ℝ × ℝ) p2 = sqrt 58 :=
by
  let p1 := (2, 2)
  let p2 := (5, 9)
  have h1 : p1.1 = 2 := rfl
  have h2 : p1.2 = 2 := rfl
  have h3 : p2.1 = 5 := rfl
  have h4 : p2.2 = 9 := rfl
  sorry

end distance_between_points_l1908_190864


namespace find_n_l1908_190840

def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

theorem find_n (n : ℕ) : 3 * n * factorial n + 2 * factorial n = 40320 → n = 8 :=
by
  sorry

end find_n_l1908_190840


namespace whiskers_count_l1908_190853

variable (P C S : ℕ)

theorem whiskers_count :
  P = 14 →
  C = 2 * P - 6 →
  S = P + C + 8 →
  C = 22 ∧ S = 44 :=
by
  intros hP hC hS
  rw [hP] at hC
  rw [hP, hC] at hS
  exact ⟨hC, hS⟩

end whiskers_count_l1908_190853


namespace find_a_and_b_l1908_190803

theorem find_a_and_b (a b m : ℝ) 
  (h1 : (3 * a - 5)^(1 / 3) = -2)
  (h2 : ∀ x, x^2 = b → x = m ∨ x = 1 - 5 * m) : 
  a = -1 ∧ b = 1 / 16 :=
by
  sorry  -- proof to be constructed

end find_a_and_b_l1908_190803


namespace sum_of_coefficients_eq_two_l1908_190822

theorem sum_of_coefficients_eq_two {a b c : ℤ} (h : ∀ x : ℤ, x * (x + 1) = a + b * x + c * x^2) : a + b + c = 2 := 
by
  sorry

end sum_of_coefficients_eq_two_l1908_190822


namespace daniel_practices_total_minutes_in_week_l1908_190883

theorem daniel_practices_total_minutes_in_week :
  let school_minutes_per_day := 15
  let school_days := 5
  let weekend_minutes_per_day := 2 * school_minutes_per_day
  let weekend_days := 2
  let total_school_week_minutes := school_minutes_per_day * school_days
  let total_weekend_minutes := weekend_minutes_per_day * weekend_days
  total_school_week_minutes + total_weekend_minutes = 135 :=
by
  sorry

end daniel_practices_total_minutes_in_week_l1908_190883


namespace arithmetic_sequence_sum_l1908_190817

-- Condition definitions
def a : Int := 3
def d : Int := 2
def a_n : Int := 25
def n : Int := 12

-- Sum formula for an arithmetic sequence proof
theorem arithmetic_sequence_sum :
    let n := 12
    let S_n := (n * (a + a_n)) / 2
    S_n = 168 := by
  sorry

end arithmetic_sequence_sum_l1908_190817


namespace divisor_of_condition_l1908_190879

theorem divisor_of_condition {d z : ℤ} (h1 : ∃ k : ℤ, z = k * d + 6)
  (h2 : ∃ m : ℤ, (z + 3) = d * m) : d = 9 := 
sorry

end divisor_of_condition_l1908_190879


namespace mean_equality_l1908_190896

theorem mean_equality (y z : ℝ)
  (h : (14 + y + z) / 3 = (8 + 15 + 21) / 3)
  (hyz : y = z) :
  y = 15 ∧ z = 15 :=
by sorry

end mean_equality_l1908_190896


namespace Bella_age_l1908_190832

theorem Bella_age (B : ℕ) (h₁ : ∃ n : ℕ, n = B + 9) (h₂ : B + (B + 9) = 19) : B = 5 := 
by
  sorry

end Bella_age_l1908_190832


namespace simplify_expression_l1908_190852

theorem simplify_expression :
  (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 :=
by
  sorry

end simplify_expression_l1908_190852


namespace sum_two_angles_greater_third_l1908_190878

-- Definitions of the angles and the largest angle condition
variables {P A B C} -- Points defining the trihedral angle
variables {α β γ : ℝ} -- Angles α, β, γ
variables (h1 : γ ≥ α) (h2 : γ ≥ β)

-- Statement of the theorem
theorem sum_two_angles_greater_third (P A B C : Type*) (α β γ : ℝ)
  (h1 : γ ≥ α) (h2 : γ ≥ β) : α + β > γ :=
sorry  -- Proof is omitted

end sum_two_angles_greater_third_l1908_190878


namespace athletes_same_color_probability_l1908_190875

theorem athletes_same_color_probability :
  let colors := ["red", "white", "blue"]
  let total_ways := 3 * 3
  let same_color_ways := 3
  total_ways > 0 → 
  (same_color_ways : ℚ) / (total_ways : ℚ) = 1 / 3 :=
by
  sorry

end athletes_same_color_probability_l1908_190875


namespace chameleons_color_change_l1908_190886

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l1908_190886


namespace simplify_expression_l1908_190873

theorem simplify_expression : 
  (1 / (64^(1/3))^9) * 8^6 = 1 := by 
  have h1 : 64 = 2^6 := by rfl
  have h2 : 8 = 2^3 := by rfl
  sorry

end simplify_expression_l1908_190873


namespace birds_in_house_l1908_190810

theorem birds_in_house (B : ℕ) :
  let dogs := 3
  let cats := 18
  let humans := 7
  let total_heads := B + dogs + cats + humans
  let total_feet := 2 * B + 4 * dogs + 4 * cats + 2 * humans
  total_feet = total_heads + 74 → B = 4 :=
by
  intros dogs cats humans total_heads total_feet condition
  -- We assume the condition and work towards the proof.
  sorry

end birds_in_house_l1908_190810


namespace opposite_of_neg_one_fifth_l1908_190802

theorem opposite_of_neg_one_fifth : -(- (1/5)) = (1/5) :=
by
  sorry

end opposite_of_neg_one_fifth_l1908_190802


namespace r_squared_plus_s_squared_l1908_190809

theorem r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end r_squared_plus_s_squared_l1908_190809


namespace angle_in_third_quadrant_l1908_190820

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  180 - α > -90 - k * 360 ∧ 180 - α < -k * 360 := 
by sorry

end angle_in_third_quadrant_l1908_190820


namespace solution_range_l1908_190871

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l1908_190871


namespace unique_real_solution_floor_eq_l1908_190849

theorem unique_real_solution_floor_eq (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ ⌊x⌋ * (x^2 + 1) = x^3 :=
sorry

end unique_real_solution_floor_eq_l1908_190849


namespace eval_derivative_at_one_and_neg_one_l1908_190833

def f (x : ℝ) : ℝ := x^4 + x - 1

theorem eval_derivative_at_one_and_neg_one : 
  (deriv f 1) + (deriv f (-1)) = 2 :=
by 
  -- proof to be filled in
  sorry

end eval_derivative_at_one_and_neg_one_l1908_190833


namespace quadratic_distinct_positive_roots_l1908_190844

theorem quadratic_distinct_positive_roots (a : ℝ) : 
  9 * (a - 2) > 0 → 
  a > 0 → 
  a^2 - 9 * a + 18 > 0 → 
  a ≠ 11 → 
  (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a) := 
by 
  intros h1 h2 h3 h4
  sorry

end quadratic_distinct_positive_roots_l1908_190844


namespace quadratic_inequality_l1908_190836

theorem quadratic_inequality (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := 
by
  sorry

end quadratic_inequality_l1908_190836


namespace area_of_triangle_ADC_l1908_190862

-- Define the constants for the problem
variable (BD DC : ℝ)
variable (abd_area adc_area : ℝ)

-- Given conditions
axiom ratio_condition : BD / DC = 5 / 2
axiom area_abd : abd_area = 35

-- Define the theorem to be proved
theorem area_of_triangle_ADC :
  ∃ adc_area, adc_area = 14 ∧ abd_area / adc_area = BD / DC := 
sorry

end area_of_triangle_ADC_l1908_190862


namespace Mateen_garden_area_l1908_190899

theorem Mateen_garden_area :
  ∀ (L W : ℝ), (50 * L = 2000) ∧ (20 * (2 * L + 2 * W) = 2000) → (L * W = 400) :=
by
  intros L W h
  -- We have two conditions based on the problem:
  -- 1. Mateen must walk the length 50 times to cover 2000 meters.
  -- 2. Mateen must walk the perimeter 20 times to cover 2000 meters.
  have h1 : 50 * L = 2000 := h.1
  have h2 : 20 * (2 * L + 2 * W) = 2000 := h.2
  -- We can use these conditions to derive the area of the garden
  sorry

end Mateen_garden_area_l1908_190899


namespace oxen_count_l1908_190866

theorem oxen_count (B C O : ℕ) (H1 : 3 * B = 4 * C) (H2 : 3 * B = 2 * O) (H3 : 15 * B + 24 * C + O * O = 33 * B + (3 / 2) * O * B) (H4 : 24 * B = 48) (H5 : 60 * C + 30 * B + 18 * (O * (3 / 2) * B) = 108 * B + (3 / 2) * O * B * 18)
: O = 8 :=
by 
  sorry

end oxen_count_l1908_190866


namespace find_y_when_x_is_7_l1908_190812

theorem find_y_when_x_is_7 (x y : ℝ) (h1 : x * y = 200) (h2 : x = 7) : y = 200 / 7 :=
by
  sorry

end find_y_when_x_is_7_l1908_190812


namespace four_distinct_real_solutions_l1908_190887

noncomputable def polynomial (a b c d e x : ℝ) : ℝ :=
  (x - a) * (x - b) * (x - c) * (x - d) * (x - e)

noncomputable def derivative (a b c d e x : ℝ) : ℝ :=
  (x - b) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - b) * (x - d) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - d)

theorem four_distinct_real_solutions (a b c d e : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
    (derivative a b c d e x1 = 0 ∧ derivative a b c d e x2 = 0 ∧ derivative a b c d e x3 = 0 ∧ derivative a b c d e x4 = 0) :=
sorry

end four_distinct_real_solutions_l1908_190887


namespace extra_money_from_customer_l1908_190827

theorem extra_money_from_customer
  (price_per_craft : ℕ)
  (num_crafts_sold : ℕ)
  (deposit_amount : ℕ)
  (remaining_amount : ℕ)
  (total_amount_before_deposit : ℕ)
  (amount_made_from_crafts : ℕ)
  (extra_money : ℕ) :
  price_per_craft = 12 →
  num_crafts_sold = 3 →
  deposit_amount = 18 →
  remaining_amount = 25 →
  total_amount_before_deposit = deposit_amount + remaining_amount →
  amount_made_from_crafts = price_per_craft * num_crafts_sold →
  extra_money = total_amount_before_deposit - amount_made_from_crafts →
  extra_money = 7 :=
by
  intros; sorry

end extra_money_from_customer_l1908_190827


namespace negation_of_prop_p_is_correct_l1908_190800

-- Define the original proposition p
def prop_p (x y : ℝ) : Prop := x > 0 ∧ y > 0 → x * y > 0

-- Define the negation of the proposition p
def neg_prop_p (x y : ℝ) : Prop := x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0

-- The theorem we need to prove
theorem negation_of_prop_p_is_correct : ∀ x y : ℝ, neg_prop_p x y := 
sorry

end negation_of_prop_p_is_correct_l1908_190800


namespace eval_expression_l1908_190861

theorem eval_expression : (8 / 4 - 3 * 2 + 9 - 3^2) = -4 := sorry

end eval_expression_l1908_190861


namespace value_ab_plus_a_plus_b_l1908_190816

noncomputable def polynomial : Polynomial ℝ := Polynomial.C (-1) + Polynomial.X * Polynomial.C (-1) + Polynomial.X^2 * Polynomial.C (-4) + Polynomial.X^4

theorem value_ab_plus_a_plus_b {a b : ℝ} (h : polynomial.eval a = 0 ∧ polynomial.eval b = 0) : a * b + a + b = -1 / 2 :=
sorry

end value_ab_plus_a_plus_b_l1908_190816


namespace polynomial_min_value_l1908_190858

theorem polynomial_min_value (x : ℝ) : x = -3 → x^2 + 6 * x + 10 = 1 :=
by
  intro h
  sorry

end polynomial_min_value_l1908_190858


namespace pier_influence_duration_l1908_190843

noncomputable def distance_affected_by_typhoon (AB AC: ℝ) : ℝ :=
  let AD := 350
  let DC := (AD ^ 2 - AC ^ 2).sqrt
  2 * DC

noncomputable def duration_under_influence (distance speed: ℝ) : ℝ :=
  distance / speed

theorem pier_influence_duration :
  let AB := 400
  let AC := AB * (1 / 2)
  let speed := 40
  duration_under_influence (distance_affected_by_typhoon AB AC) speed = 2.5 :=
by
  -- Proof would go here, but since it's omitted
  sorry

end pier_influence_duration_l1908_190843


namespace more_green_than_yellow_l1908_190823

-- Define constants
def red_peaches : ℕ := 2
def yellow_peaches : ℕ := 6
def green_peaches : ℕ := 14

-- Prove the statement
theorem more_green_than_yellow : green_peaches - yellow_peaches = 8 :=
by
  sorry

end more_green_than_yellow_l1908_190823


namespace average_side_lengths_l1908_190854

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l1908_190854


namespace overall_average_score_l1908_190831

-- Definitions used from conditions
def male_students : Nat := 8
def male_avg_score : Real := 83
def female_students : Nat := 28
def female_avg_score : Real := 92

-- Theorem to prove the overall average score is 90
theorem overall_average_score : 
  (male_students * male_avg_score + female_students * female_avg_score) / (male_students + female_students) = 90 := 
by 
  sorry

end overall_average_score_l1908_190831


namespace angles_of_triangle_l1908_190842

theorem angles_of_triangle (a b c m_a m_b : ℝ) (h1 : m_a ≥ a) (h2 : m_b ≥ b) : 
  ∃ (α β γ : ℝ), ∀ t, 
  (t = 90) ∧ (α = 45) ∧ (β = 45) := 
sorry

end angles_of_triangle_l1908_190842


namespace prime_triplets_l1908_190829

theorem prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p ^ q + q ^ p = r ↔ (p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17) := by
  sorry

end prime_triplets_l1908_190829


namespace tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l1908_190830

variable {α : Real}

theorem tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5 (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l1908_190830


namespace base_b_not_divisible_by_5_l1908_190876

theorem base_b_not_divisible_by_5 (b : ℕ) : b = 4 ∨ b = 7 ∨ b = 8 → ¬ (5 ∣ (2 * b^2 * (b - 1))) :=
by
  sorry

end base_b_not_divisible_by_5_l1908_190876


namespace exists_station_to_complete_loop_l1908_190848

structure CircularHighway where
  fuel_at_stations : List ℝ -- List of fuel amounts at each station
  travel_cost : List ℝ -- List of travel costs between consecutive stations

def total_fuel (hw : CircularHighway) : ℝ :=
  hw.fuel_at_stations.sum

def total_travel_cost (hw : CircularHighway) : ℝ :=
  hw.travel_cost.sum

def sufficient_fuel (hw : CircularHighway) : Prop :=
  total_fuel hw ≥ 2 * total_travel_cost hw

noncomputable def can_return_to_start (hw : CircularHighway) (start_station : ℕ) : Prop :=
  -- Function that checks if starting from a specific station allows for a return
  sorry

theorem exists_station_to_complete_loop (hw : CircularHighway) (h : sufficient_fuel hw) : ∃ start_station, can_return_to_start hw start_station :=
  sorry

end exists_station_to_complete_loop_l1908_190848


namespace farm_section_areas_l1908_190834

theorem farm_section_areas (n : ℕ) (total_area : ℕ) (sections : ℕ) 
  (hn : sections = 5) (ht : total_area = 300) : total_area / sections = 60 :=
by
  sorry

end farm_section_areas_l1908_190834


namespace base7_to_base10_of_645_l1908_190892

theorem base7_to_base10_of_645 :
  (6 * 7^2 + 4 * 7^1 + 5 * 7^0) = 327 := 
by 
  sorry

end base7_to_base10_of_645_l1908_190892


namespace chinese_character_equation_l1908_190835

noncomputable def units_digit (n: ℕ) : ℕ :=
  n % 10

noncomputable def tens_digit (n: ℕ) : ℕ :=
  (n / 10) % 10

noncomputable def hundreds_digit (n: ℕ) : ℕ :=
  (n / 100) % 10

def Math : ℕ := 25
def LoveMath : ℕ := 125
def ILoveMath : ℕ := 3125

theorem chinese_character_equation :
  Math * LoveMath = ILoveMath :=
by
  have h_units_math := units_digit Math
  have h_units_lovemath := units_digit LoveMath
  have h_units_ilovemath := units_digit ILoveMath
  
  have h_tens_math := tens_digit Math
  have h_tens_lovemath := tens_digit LoveMath
  have h_tens_ilovemath := tens_digit ILoveMath

  have h_hundreds_lovemath := hundreds_digit LoveMath
  have h_hundreds_ilovemath := hundreds_digit ILoveMath

  -- Check conditions:
  -- h_units_* should be 0, 1, 5 or 6
  -- h_tens_math == h_tens_lovemath == h_tens_ilovemath
  -- h_hundreds_lovemath == h_hundreds_ilovemath

  sorry -- Proof would go here

end chinese_character_equation_l1908_190835


namespace value_of_x_squared_minus_y_squared_l1908_190867

theorem value_of_x_squared_minus_y_squared (x y : ℝ) 
  (h₁ : x + y = 20) 
  (h₂ : x - y = 6) :
  x^2 - y^2 = 120 := 
by 
  sorry

end value_of_x_squared_minus_y_squared_l1908_190867


namespace completing_square_transformation_l1908_190884

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_l1908_190884


namespace maximum_dn_l1908_190825

-- Definitions of a_n and d_n based on the problem statement
def a (n : ℕ) : ℕ := 150 + (n + 1)^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Statement of the theorem
theorem maximum_dn : ∃ M, M = 2 ∧ ∀ n, d n ≤ M :=
by
  -- proof should be written here
  sorry

end maximum_dn_l1908_190825


namespace relationship_between_x_and_y_l1908_190869

variable (u : ℝ)

theorem relationship_between_x_and_y (h : u > 0) (hx : x = (u + 1)^(1 / u)) (hy : y = (u + 1)^((u + 1) / u)) :
  y^x = x^y :=
by
  sorry

end relationship_between_x_and_y_l1908_190869


namespace number_of_real_b_l1908_190815

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end number_of_real_b_l1908_190815


namespace other_candidate_votes_l1908_190839

-- Define the constants according to the problem
variables (X Y Z : ℝ)
axiom h1 : X = Y + (1 / 2) * Y
axiom h2 : X = 22500
axiom h3 : Y = Z - (2 / 5) * Z

-- Define the goal
theorem other_candidate_votes : Z = 25000 :=
by
  sorry

end other_candidate_votes_l1908_190839


namespace arithmetic_sequence_common_difference_l1908_190882

theorem arithmetic_sequence_common_difference
  (a1 a4 : ℤ) (d : ℤ) 
  (h1 : a1 + (a1 + 4 * d) = 10)
  (h2 : a1 + 3 * d = 7) : 
  d = 2 :=
sorry

end arithmetic_sequence_common_difference_l1908_190882


namespace chord_length_l1908_190826

theorem chord_length
  (x y : ℝ)
  (h_circle : (x-1)^2 + (y-2)^2 = 2)
  (h_line : 3*x - 4*y = 0) :
  ∃ L : ℝ, L = 2 :=
sorry

end chord_length_l1908_190826


namespace sequence_b_n_l1908_190856

theorem sequence_b_n (b : ℕ → ℕ) (h₀ : b 1 = 3) (h₁ : ∀ n, b (n + 1) = b n + 3 * n + 1) :
  b 50 = 3727 :=
sorry

end sequence_b_n_l1908_190856


namespace roots_ratio_quadratic_eq_l1908_190804

theorem roots_ratio_quadratic_eq {k r s : ℝ} 
(h_eq : ∃ a b : ℝ, a * r = b * s) 
(ratio_3_2 : ∃ t : ℝ, r = 3 * t ∧ s = 2 * t) 
(eqn : r + s = -10 ∧ r * s = k) : 
k = 24 := 
sorry

end roots_ratio_quadratic_eq_l1908_190804


namespace min_chord_length_intercepted_line_eq_l1908_190814

theorem min_chord_length_intercepted_line_eq (m : ℝ)
  (hC : ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 16)
  (hL : ∀ (x y : ℝ), (2*m-1)*x + (m-1)*y - 3*m + 1 = 0)
  : ∃ x y : ℝ, x - 2*y - 4 = 0 := sorry

end min_chord_length_intercepted_line_eq_l1908_190814


namespace find_x_l1908_190841

theorem find_x (x : ℕ) (h : x * 6000 = 480 * 10^5) : x = 8000 := 
by
  sorry

end find_x_l1908_190841


namespace solve_for_asterisk_l1908_190808

theorem solve_for_asterisk (asterisk : ℝ) : 
  ((60 / 20) * (60 / asterisk) = 1) → asterisk = 180 :=
by
  sorry

end solve_for_asterisk_l1908_190808


namespace additional_bureaus_needed_correct_l1908_190891

-- The number of bureaus the company has
def total_bureaus : ℕ := 192

-- The number of offices
def total_offices : ℕ := 36

-- The additional bureaus needed to ensure each office gets an equal number
def additional_bureaus_needed (bureaus : ℕ) (offices : ℕ) : ℕ :=
  let bureaus_per_office := bureaus / offices
  let rounded_bureaus_per_office := bureaus_per_office + if bureaus % offices = 0 then 0 else 1
  let total_bureaus_needed := offices * rounded_bureaus_per_office
  total_bureaus_needed - bureaus

-- Problem Statement: Prove that at least 24 more bureaus are needed
theorem additional_bureaus_needed_correct : 
  additional_bureaus_needed total_bureaus total_offices = 24 := 
by
  sorry

end additional_bureaus_needed_correct_l1908_190891


namespace paco_cookies_proof_l1908_190813

-- Define the initial conditions
def initial_cookies : Nat := 40
def cookies_eaten : Nat := 2
def cookies_bought : Nat := 37
def free_cookies_per_bought : Nat := 2

-- Define the total number of cookies after all operations
def total_cookies (initial_cookies cookies_eaten cookies_bought free_cookies_per_bought : Nat) : Nat :=
  let remaining_cookies := initial_cookies - cookies_eaten
  let free_cookies := cookies_bought * free_cookies_per_bought
  let cookies_from_bakery := cookies_bought + free_cookies
  remaining_cookies + cookies_from_bakery

-- The target statement that needs to be proved
theorem paco_cookies_proof : total_cookies initial_cookies cookies_eaten cookies_bought free_cookies_per_bought = 149 :=
by
  sorry

end paco_cookies_proof_l1908_190813


namespace color_of_last_bead_is_white_l1908_190807

-- Defining the pattern of the beads
inductive BeadColor
| White
| Black
| Red

open BeadColor

-- Define the repeating pattern of the beads
def beadPattern : ℕ → BeadColor
| 0 => White
| 1 => Black
| 2 => Black
| 3 => Red
| 4 => Red
| 5 => Red
| (n + 6) => beadPattern n

-- Define the total number of beads
def totalBeads : ℕ := 85

-- Define the position of the last bead
def lastBead : ℕ := totalBeads - 1

-- Proving the color of the last bead
theorem color_of_last_bead_is_white : beadPattern lastBead = White :=
by
  sorry

end color_of_last_bead_is_white_l1908_190807


namespace puzzles_sold_correct_l1908_190885

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end puzzles_sold_correct_l1908_190885


namespace linear_function_no_third_quadrant_l1908_190846

theorem linear_function_no_third_quadrant :
  ∀ x y : ℝ, (y = -5 * x + 2023) → ¬ (x < 0 ∧ y < 0) := 
by
  intros x y h
  sorry

end linear_function_no_third_quadrant_l1908_190846


namespace problem1_problem2_l1908_190870

noncomputable def A : Set ℝ := Set.Icc 1 4
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a

-- Problem 1
theorem problem1 (A := A) (B := B 4) : A ∩ B = Set.Icc 1 4 := by
  sorry 

-- Problem 2
theorem problem2 (A := A) : ∀ a : ℝ, (A ⊆ B a) → (4 ≤ a) := by
  sorry

end problem1_problem2_l1908_190870


namespace only_valid_M_l1908_190889

def digit_sum (n : ℕ) : ℕ :=
  -- definition of digit_sum as a function summing up digits of n
  sorry 

def is_valid_M (M : ℕ) := 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digit_sum (M * k) = digit_sum M

theorem only_valid_M (M : ℕ) :
  is_valid_M M ↔ ∃ n : ℕ, ∀ m : ℕ, M = 10^n - 1 :=
by
  sorry

end only_valid_M_l1908_190889


namespace find_number_l1908_190895

theorem find_number (x : ℝ) (h : 75 = 0.6 * x) : x = 125 :=
sorry

end find_number_l1908_190895


namespace ab_value_l1908_190860

theorem ab_value (a b : ℤ) (h : 48 * a * b = 65 * a * b) : a * b = 0 :=
  sorry

end ab_value_l1908_190860


namespace discount_percentage_l1908_190821

theorem discount_percentage 
  (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) (savings : ℝ) (discounted_food_combo_cost : ℝ) (discounted_total_cost : ℝ) 
  (h1 : evening_ticket_cost = 10) 
  (h2 : food_combo_cost = 10)
  (h3 : discounted_food_combo_cost = 10 * 0.5)
  (h4 : discounted_total_cost = evening_ticket_cost + food_combo_cost - savings)
  (h5 : savings = 7)
: (1 - discounted_total_cost / (evening_ticket_cost + food_combo_cost)) * 100 = 20 :=
by
  sorry

end discount_percentage_l1908_190821


namespace solution_set_Inequality_l1908_190888

theorem solution_set_Inequality : {x : ℝ | abs (1 + x + x^2 / 2) < 1} = {x : ℝ | -2 < x ∧ x < 0} :=
sorry

end solution_set_Inequality_l1908_190888


namespace max_value_ineq_l1908_190819

theorem max_value_ineq (x y : ℝ) (hx1 : -5 ≤ x) (hx2 : x ≤ -3) (hy1 : 1 ≤ y) (hy2 : y ≤ 3) : 
  (x + y) / (x - 1) ≤ 2 / 3 := 
sorry

end max_value_ineq_l1908_190819


namespace sum_products_roots_l1908_190890

theorem sum_products_roots :
  (∃ p q r : ℂ, (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
                  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
                  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) ∧
                  (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r)) →
  ∀ p q r : ℂ, (3 * p) * (q * r) + (3 * q) * (r * p) + (3 * r) * (p * q) =
    (3 * p * q * r) :=
sorry

end sum_products_roots_l1908_190890


namespace complex_solution_l1908_190868

theorem complex_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : (3 - 4 * i) * z = 5 * i) : z = (4 / 5) + (3 / 5) * i :=
by {
  sorry
}

end complex_solution_l1908_190868


namespace problem_statement_l1908_190828

variable {R : Type} [LinearOrderedField R]
variable (f : R → R)

theorem problem_statement
  (hf1 : ∀ x y : R, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x < y → f x < f y)
  (hf2 : ∀ x : R, f (x + 2) = f (- (x + 2))) :
  f (7 / 2) < f 1 ∧ f 1 < f (5 / 2) :=
by
  sorry

end problem_statement_l1908_190828


namespace thickness_of_wall_l1908_190894

theorem thickness_of_wall 
    (brick_length cm : ℝ)
    (brick_width cm : ℝ)
    (brick_height cm : ℝ)
    (num_bricks : ℝ)
    (wall_length cm : ℝ)
    (wall_height cm : ℝ)
    (wall_thickness cm : ℝ) :
    brick_length = 25 → 
    brick_width = 11.25 → 
    brick_height = 6 →
    num_bricks = 7200 → 
    wall_length = 900 → 
    wall_height = 600 →
    wall_length * wall_height * wall_thickness = num_bricks * (brick_length * brick_width * brick_height) →
    wall_thickness = 22.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end thickness_of_wall_l1908_190894


namespace canonical_equations_of_line_l1908_190857

-- Conditions: Two planes given by their equations
def plane1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z + 8 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x + 5 * y - 4 * z + 4 = 0

-- Proving the canonical form of the line
theorem canonical_equations_of_line :
  ∃ x y z, plane1 x y z ∧ plane2 x y z ↔ 
  ∃ t, x = -1 + 5 * t ∧ y = 2 / 5 + 42 * t ∧ z = 60 * t :=
sorry

end canonical_equations_of_line_l1908_190857


namespace solution_y_chemical_A_percentage_l1908_190893

def percent_chemical_A_in_x : ℝ := 0.30
def percent_chemical_A_in_mixture : ℝ := 0.32
def percent_solution_x_in_mixture : ℝ := 0.80
def percent_solution_y_in_mixture : ℝ := 0.20

theorem solution_y_chemical_A_percentage
  (P : ℝ) 
  (h : percent_solution_x_in_mixture * percent_chemical_A_in_x + percent_solution_y_in_mixture * P = percent_chemical_A_in_mixture) :
  P = 0.40 :=
sorry

end solution_y_chemical_A_percentage_l1908_190893


namespace option_D_is_div_by_9_l1908_190805

-- Define the parameters and expressions
def A (k : ℕ) : ℤ := 6 + 6 * 7^k
def B (k : ℕ) : ℤ := 2 + 7^(k - 1)
def C (k : ℕ) : ℤ := 2 * (2 + 7^(k + 1))
def D (k : ℕ) : ℤ := 3 * (2 + 7^k)

-- Define the main theorem to prove that D is divisible by 9
theorem option_D_is_div_by_9 (k : ℕ) (hk : k > 0) : D k % 9 = 0 :=
sorry

end option_D_is_div_by_9_l1908_190805


namespace original_height_of_ball_l1908_190811

theorem original_height_of_ball (h : ℝ) : 
  (h + 2 * (0.5 * h) + 2 * ((0.5)^2 * h) = 200) -> 
  h = 800 / 9 := 
by
  sorry

end original_height_of_ball_l1908_190811


namespace even_increasing_decreasing_l1908_190855

theorem even_increasing_decreasing (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = -x^2) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x < 0 → f x < f (x + 1)) ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1)) :=
by
  sorry

end even_increasing_decreasing_l1908_190855


namespace area_of_sector_l1908_190838

theorem area_of_sector (L θ : ℝ) (hL : L = 4) (hθ : θ = 2) : 
  (1 / 2) * ((L / θ) ^ 2) * θ = 4 := by
  sorry

end area_of_sector_l1908_190838


namespace sum_of_inversion_counts_of_all_permutations_l1908_190837

noncomputable def sum_of_inversion_counts (n : ℕ) (fixed_val : ℕ) (fixed_pos : ℕ) : ℕ :=
  if n = 6 ∧ fixed_val = 4 ∧ fixed_pos = 3 then 120 else 0

theorem sum_of_inversion_counts_of_all_permutations :
  sum_of_inversion_counts 6 4 3 = 120 :=
by
  sorry

end sum_of_inversion_counts_of_all_permutations_l1908_190837


namespace range_of_a_l1908_190806

variable (x a : ℝ)

theorem range_of_a (h1 : ∀ x, x ≤ a → x < 2) (h2 : ∀ x, x < 2) : a ≥ 2 :=
sorry

end range_of_a_l1908_190806


namespace find_a_and_b_l1908_190847

theorem find_a_and_b (a b : ℝ) 
  (h_tangent_slope : (2 * a * 2 + b = 1)) 
  (h_point_on_parabola : (a * 4 + b * 2 + 9 = -1)) : 
  a = 3 ∧ b = -11 :=
by
  sorry

end find_a_and_b_l1908_190847


namespace average_first_20_multiples_of_17_l1908_190898

theorem average_first_20_multiples_of_17 :
  (20 / 2 : ℝ) * (17 + 17 * 20) / 20 = 178.5 := by
  sorry

end average_first_20_multiples_of_17_l1908_190898


namespace first_reduction_percentage_l1908_190801

theorem first_reduction_percentage 
  (P : ℝ)  -- original price
  (x : ℝ)  -- first day reduction percentage
  (h : P > 0) -- price assumption
  (h2 : 0 ≤ x ∧ x ≤ 100) -- percentage assumption
  (cond : P * (1 - x / 100) * 0.86 = 0.774 * P) : 
  x = 10 := 
sorry

end first_reduction_percentage_l1908_190801


namespace value_of_x_squared_plus_reciprocal_l1908_190845

theorem value_of_x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_l1908_190845
