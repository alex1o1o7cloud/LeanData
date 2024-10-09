import Mathlib

namespace Reeta_pencils_l142_14290

-- Let R be the number of pencils Reeta has
variable (R : ℕ)

-- Condition 1: Anika has 4 more than twice the number of pencils as Reeta
def Anika_pencils := 2 * R + 4

-- Condition 2: Together, Anika and Reeta have 64 pencils
def combined_pencils := R + Anika_pencils R

theorem Reeta_pencils (h : combined_pencils R = 64) : R = 20 :=
by
  sorry

end Reeta_pencils_l142_14290


namespace expression_evaluates_to_3_l142_14226

theorem expression_evaluates_to_3 :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 :=
sorry

end expression_evaluates_to_3_l142_14226


namespace total_teachers_correct_l142_14299

noncomputable def total_teachers (x : ℕ) : ℕ := 26 + 104 + x

theorem total_teachers_correct
    (x : ℕ)
    (h : (x : ℝ) / (26 + 104 + x) = 16 / 56) :
  total_teachers x = 182 :=
sorry

end total_teachers_correct_l142_14299


namespace incorrect_statements_l142_14228

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def monotonically_decreasing_in_pos (f : ℝ → ℝ) : Prop :=
∀ x y, 0 < x ∧ x < y → f y ≤ f x

theorem incorrect_statements
  (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_decreasing : monotonically_decreasing_in_pos f) :
  ¬ (∀ a, f (2 * a) < f (-a)) ∧ ¬ (f π > f (-3)) ∧ ¬ (∀ a, f (a^2 + 1) < f 1) :=
by sorry

end incorrect_statements_l142_14228


namespace a7_of_expansion_x10_l142_14227

theorem a7_of_expansion_x10 : 
  (∃ (a : ℕ) (a1 : ℕ) (a2 : ℕ) (a3 : ℕ) 
     (a4 : ℕ) (a5 : ℕ) (a6 : ℕ) 
     (a8 : ℕ) (a9 : ℕ) (a10 : ℕ),
     ((x : ℕ) → x^10 = a + a1*(x-1) + a2*(x-1)^2 + a3*(x-1)^3 + 
                      a4*(x-1)^4 + a5*(x-1)^5 + a6*(x-1)^6 + 
                      120*(x-1)^7 + a8*(x-1)^8 + a9*(x-1)^9 + a10*(x-1)^10)) :=
  sorry

end a7_of_expansion_x10_l142_14227


namespace max_product_of_roots_of_quadratic_l142_14271

theorem max_product_of_roots_of_quadratic :
  ∃ k : ℚ, 6 * k^2 - 8 * k + (4 / 3) = 0 ∧ (64 - 48 * k) ≥ 0 ∧ (∀ k' : ℚ, (64 - 48 * k') ≥ 0 → (k'/3) ≤ (4/9)) :=
by
  sorry

end max_product_of_roots_of_quadratic_l142_14271


namespace perfect_squares_example_l142_14221

def isPerfectSquare (n: ℕ) : Prop := ∃ m: ℕ, m * m = n

theorem perfect_squares_example :
  let a := 10430
  let b := 3970
  let c := 2114
  let d := 386
  isPerfectSquare (a + b) ∧
  isPerfectSquare (a + c) ∧
  isPerfectSquare (a + d) ∧
  isPerfectSquare (b + c) ∧
  isPerfectSquare (b + d) ∧
  isPerfectSquare (c + d) ∧
  isPerfectSquare (a + b + c + d) :=
by
  -- Proof steps go here
  sorry

end perfect_squares_example_l142_14221


namespace liam_annual_income_l142_14295

theorem liam_annual_income (q : ℝ) (I : ℝ) (T : ℝ) 
  (h1 : T = (q + 0.5) * 0.01 * I) 
  (h2 : I > 50000) 
  (h3 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000)) : 
  I = 56000 :=
by
  sorry

end liam_annual_income_l142_14295


namespace Lizzy_money_after_loan_l142_14232

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l142_14232


namespace max_value_of_m_l142_14291

noncomputable def f (x m n : ℝ) : ℝ := x^2 + m*x + n^2
noncomputable def g (x m n : ℝ) : ℝ := x^2 + (m+2)*x + n^2 + m + 1

theorem max_value_of_m (m n t : ℝ) :
  (∀(t : ℝ), f t m n ≥ 0 ∨ g t m n ≥ 0) → m ≤ 1 :=
by
  intro h
  sorry

end max_value_of_m_l142_14291


namespace num_balls_in_box_l142_14282

theorem num_balls_in_box (n : ℕ) (h1: 9 <= n) (h2: (9 : ℝ) / n = 0.30) : n = 30 :=
sorry

end num_balls_in_box_l142_14282


namespace ellipse_focus_m_eq_3_l142_14256

theorem ellipse_focus_m_eq_3 (m : ℝ) (h : m > 0) : 
  (∃ a c : ℝ, a = 5 ∧ c = 4 ∧ c^2 = a^2 - m^2)
  → m = 3 :=
by
  sorry

end ellipse_focus_m_eq_3_l142_14256


namespace parabola_intercepts_l142_14238

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l142_14238


namespace value_of_a_l142_14252

-- Definition of the function and the point
def graph_function (x : ℝ) : ℝ := -x^2
def point_lies_on_graph (a : ℝ) : Prop := (a, -9) ∈ {p : ℝ × ℝ | p.2 = graph_function p.1}

-- The theorem stating that if the point (a, -9) lies on the graph of y = -x^2, then a = ±3
theorem value_of_a (a : ℝ) (h : point_lies_on_graph a) : a = 3 ∨ a = -3 :=
by 
  sorry

end value_of_a_l142_14252


namespace quadrilateral_perimeter_l142_14209

-- Define the basic conditions
variables (a b : ℝ)

-- Let's define what happens when Xiao Ming selected 2 pieces of type A, 7 pieces of type B, and 3 pieces of type C
theorem quadrilateral_perimeter (a b : ℝ) : 2 * (a + 3 * b + 2 * a + b) = 6 * a + 8 * b :=
by sorry

end quadrilateral_perimeter_l142_14209


namespace truncated_cone_resistance_l142_14274

theorem truncated_cone_resistance (a b h : ℝ) (ρ : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_pos : 0 < h) :
  (∫ x in (0:ℝ)..h, ρ / (π * ((a + x * (b - a) / h) / 2) ^ 2)) = 4 * ρ * h / (π * a * b) := 
sorry

end truncated_cone_resistance_l142_14274


namespace sphere_radius_l142_14233

theorem sphere_radius 
  (r h1 h2 : ℝ)
  (A1_eq : 5 * π = π * (r^2 - h1^2))
  (A2_eq : 8 * π = π * (r^2 - h2^2))
  (h1_h2_eq : h1 - h2 = 1) : r = 3 :=
by
  sorry

end sphere_radius_l142_14233


namespace trader_gain_percentage_l142_14245

theorem trader_gain_percentage 
  (C : ℝ) -- cost of each pen
  (h1 : 250 * C ≠ 0) -- ensure the cost of 250 pens is non-zero
  (h2 : 65 * C > 0) -- ensure the gain is positive
  (h3 : 250 * C + 65 * C > 0) -- ensure the selling price is positive
  : (65 / 250) * 100 = 26 := 
sorry

end trader_gain_percentage_l142_14245


namespace snow_fall_time_l142_14258

theorem snow_fall_time :
  (∀ rate_per_six_minutes : ℕ, rate_per_six_minutes = 1 →
    (∀ minute : ℕ, minute = 6 →
      (∀ height_in_m : ℕ, height_in_m = 1 →
        ∃ time_in_hours : ℕ, time_in_hours = 100 ))) :=
sorry

end snow_fall_time_l142_14258


namespace leaves_dropped_on_fifth_day_l142_14247

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end leaves_dropped_on_fifth_day_l142_14247


namespace mean_value_of_interior_angles_of_quadrilateral_l142_14246

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l142_14246


namespace mean_of_jane_scores_l142_14241

theorem mean_of_jane_scores :
  let scores := [96, 95, 90, 87, 91, 75]
  let n := 6
  let sum_scores := 96 + 95 + 90 + 87 + 91 + 75
  let mean := sum_scores / n
  mean = 89 := by
    sorry

end mean_of_jane_scores_l142_14241


namespace repaved_today_l142_14254

theorem repaved_today (total before : ℕ) (h_total : total = 4938) (h_before : before = 4133) : total - before = 805 := by
  sorry

end repaved_today_l142_14254


namespace find_paycheck_l142_14210

variable (P : ℝ) -- P represents the paycheck amount

def initial_balance : ℝ := 800
def rent_payment : ℝ := 450
def electricity_bill : ℝ := 117
def internet_bill : ℝ := 100
def phone_bill : ℝ := 70
def final_balance : ℝ := 1563

theorem find_paycheck :
  initial_balance - rent_payment + P - (electricity_bill + internet_bill) - phone_bill = final_balance → 
    P = 1563 :=
by
  sorry

end find_paycheck_l142_14210


namespace sales_growth_correct_equation_l142_14229

theorem sales_growth_correct_equation (x : ℝ) 
(sales_24th : ℝ) (total_sales_25th_26th : ℝ) 
(h_initial : sales_24th = 5000) (h_total : total_sales_25th_26th = 30000) :
  (5000 * (1 + x)) + (5000 * (1 + x)^2) = 30000 :=
sorry

end sales_growth_correct_equation_l142_14229


namespace volume_relationship_l142_14213

open Real

theorem volume_relationship (r : ℝ) (A M C : ℝ)
  (hA : A = (1/3) * π * r^3)
  (hM : M = π * r^3)
  (hC : C = (4/3) * π * r^3) :
  A + M + (1/2) * C = 2 * π * r^3 :=
by
  sorry

end volume_relationship_l142_14213


namespace smallest_N_exists_l142_14264

def find_smallest_N (N : ℕ) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : ℕ),
  (N ≠ 0) ∧ 
  (c1 = 6 * c2 - 1) ∧ 
  (N + c2 = 6 * c3 - 2) ∧ 
  (2 * N + c3 = 6 * c4 - 3) ∧ 
  (3 * N + c4 = 6 * c5 - 4) ∧ 
  (4 * N + c5 = 6 * c6 - 5) ∧ 
  (5 * N + c6 = 6 * c1)

theorem smallest_N_exists : ∃ (N : ℕ), find_smallest_N N :=
sorry

end smallest_N_exists_l142_14264


namespace fraction_of_300_greater_than_3_fifths_of_125_l142_14207

theorem fraction_of_300_greater_than_3_fifths_of_125 (f : ℚ)
    (h : f * 300 = 3 / 5 * 125 + 45) : 
    f = 2 / 5 :=
sorry

end fraction_of_300_greater_than_3_fifths_of_125_l142_14207


namespace power_modulo_l142_14212

theorem power_modulo {a : ℤ} : a^561 ≡ a [ZMOD 561] :=
sorry

end power_modulo_l142_14212


namespace sum_digits_base8_to_base4_l142_14239

theorem sum_digits_base8_to_base4 :
  ∀ n : ℕ, (n ≥ 512 ∧ n ≤ 4095) →
  (∃ d : ℕ, (4^d > n ∧ n ≥ 4^(d-1))) →
  (d = 6) :=
by {
  sorry
}

end sum_digits_base8_to_base4_l142_14239


namespace solution_set_l142_14244

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

theorem solution_set :
  ∀ x y z : ℝ, system_of_equations x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = -1 ∧ z = -3) ∨ 
  (x = -2 ∧ y = 1 ∧ z = -3) ∨ (x = -2 ∧ y = -1 ∧ z = 3) :=
by
  sorry

end solution_set_l142_14244


namespace mark_cans_l142_14219

variable (r j m : ℕ) -- r for Rachel, j for Jaydon, m for Mark

theorem mark_cans (r j m : ℕ) 
  (h1 : j = 5 + 2 * r)
  (h2 : m = 4 * j)
  (h3 : r + j + m = 135) : 
  m = 100 :=
by
  sorry

end mark_cans_l142_14219


namespace circumcircle_eq_of_triangle_ABC_l142_14251

noncomputable def circumcircle_equation (A B C : ℝ × ℝ) : String := sorry

theorem circumcircle_eq_of_triangle_ABC :
  circumcircle_equation (4, 1) (-6, 3) (3, 0) = "x^2 + y^2 + x - 9y - 12 = 0" :=
sorry

end circumcircle_eq_of_triangle_ABC_l142_14251


namespace front_view_l142_14292

def first_column_heights := [3, 2]
def middle_column_heights := [1, 4, 2]
def third_column_heights := [5]

theorem front_view (h1 : first_column_heights = [3, 2])
                   (h2 : middle_column_heights = [1, 4, 2])
                   (h3 : third_column_heights = [5]) :
    [3, 4, 5] = [
        first_column_heights.foldr max 0,
        middle_column_heights.foldr max 0,
        third_column_heights.foldr max 0
    ] :=
    sorry

end front_view_l142_14292


namespace balloon_count_l142_14279

theorem balloon_count (total_balloons red_balloons blue_balloons black_balloons : ℕ) 
  (h_total : total_balloons = 180)
  (h_red : red_balloons = 3 * blue_balloons)
  (h_black : black_balloons = 2 * blue_balloons) :
  red_balloons = 90 ∧ blue_balloons = 30 ∧ black_balloons = 60 :=
by
  sorry

end balloon_count_l142_14279


namespace total_weight_is_correct_l142_14272

noncomputable def A (B : ℝ) : ℝ := 12 + (1/2) * B
noncomputable def B (C : ℝ) : ℝ := 8 + (1/3) * C
noncomputable def C (A : ℝ) : ℝ := 20 + 2 * A
noncomputable def NewWeightB (A B : ℝ) : ℝ := B + 0.15 * A
noncomputable def NewWeightA (A C : ℝ) : ℝ := A - 0.10 * C

theorem total_weight_is_correct (B C : ℝ) (h1 : A B = (C - 20) / 2)
  (h2 : B = 8 + (1/3) * C) 
  (h3 : C = 20 + 2 * A B) 
  (h4 : NewWeightB (A B) B = 38.35) 
  (h5 : NewWeightA (A B) C = 21.2) :
  NewWeightA (A B) C + NewWeightB (A B) B + C = 139.55 :=
sorry

end total_weight_is_correct_l142_14272


namespace expected_adjacent_black_pairs_proof_l142_14261

-- Define the modified deck conditions.
def modified_deck (n : ℕ) := n = 60
def black_cards (b : ℕ) := b = 30
def red_cards (r : ℕ) := r = 30

-- Define the expected value of pairs of adjacent black cards.
def expected_adjacent_black_pairs (n b : ℕ) : ℚ :=
  b * (b - 1) / (n - 1)

theorem expected_adjacent_black_pairs_proof :
  modified_deck 60 →
  black_cards 30 →
  red_cards 30 →
  expected_adjacent_black_pairs 60 30 = 870 / 59 :=
by intros; sorry

end expected_adjacent_black_pairs_proof_l142_14261


namespace algebraic_expression_value_l142_14237

variables (a b c d m : ℤ)

def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℤ) : Prop := c * d = 1
def abs_eq_2 (m : ℤ) : Prop := |m| = 2

theorem algebraic_expression_value {a b c d m : ℤ} 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : abs_eq_2 m) :
  (2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0) :=
by
  sorry

end algebraic_expression_value_l142_14237


namespace cylinder_radius_unique_l142_14206

theorem cylinder_radius_unique
  (r : ℝ) (h : ℝ) (V : ℝ) (y : ℝ)
  (h_eq : h = 2)
  (V_eq : V = 2 * Real.pi * r ^ 2)
  (y_eq_increase_radius : y = 2 * Real.pi * ((r + 6) ^ 2 - r ^ 2))
  (y_eq_increase_height : y = 6 * Real.pi * r ^ 2) :
  r = 6 :=
by
  sorry

end cylinder_radius_unique_l142_14206


namespace total_percent_sample_candy_l142_14200

theorem total_percent_sample_candy (total_customers : ℕ) (percent_caught : ℝ) (percent_not_caught : ℝ)
  (h1 : percent_caught = 0.22)
  (h2 : percent_not_caught = 0.20)
  (h3 : total_customers = 100) :
  percent_caught + percent_not_caught = 0.28 :=
by
  sorry

end total_percent_sample_candy_l142_14200


namespace find_V_l142_14211

theorem find_V 
  (c : ℝ)
  (R₁ V₁ W₁ R₂ W₂ V₂ : ℝ)
  (h1 : R₁ = c * (V₁ / W₁))
  (h2 : R₁ = 6)
  (h3 : V₁ = 2)
  (h4 : W₁ = 3)
  (h5 : R₂ = 25)
  (h6 : W₂ = 5)
  (h7 : V₂ = R₂ * W₂ / 9) :
  V₂ = 125 / 9 :=
by sorry

end find_V_l142_14211


namespace work_duration_l142_14230

variable (a b c : ℕ)
variable (daysTogether daysA daysB daysC : ℕ)

theorem work_duration (H1 : daysTogether = 4)
                      (H2 : daysA = 12)
                      (H3 : daysB = 18)
                      (H4: a = 1 / 12)
                      (H5: b = 1 / 18)
                      (H6: 1 / daysTogether = 1 / daysA + 1 / daysB + 1 / daysC) :
                      daysC = 9 :=
sorry

end work_duration_l142_14230


namespace youngest_child_age_is_3_l142_14242

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end youngest_child_age_is_3_l142_14242


namespace alcohol_added_amount_l142_14284

theorem alcohol_added_amount :
  ∀ (x : ℝ), (40 * 0.05 + x) = 0.15 * (40 + x + 4.5) -> x = 5.5 :=
by
  intro x
  sorry

end alcohol_added_amount_l142_14284


namespace shortest_altitude_of_right_triangle_l142_14222

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end shortest_altitude_of_right_triangle_l142_14222


namespace derivative_at_neg_one_l142_14208

theorem derivative_at_neg_one (a b c : ℝ) (h : (4*a*(1:ℝ)^3 + 2*b*(1:ℝ)) = 2) :
  (4*a*(-1:ℝ)^3 + 2*b*(-1:ℝ)) = -2 :=
by
  sorry

end derivative_at_neg_one_l142_14208


namespace maximum_height_of_projectile_l142_14235

def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

theorem maximum_height_of_projectile : ∀ t : ℝ, (h t ≤ 116) :=
by sorry

end maximum_height_of_projectile_l142_14235


namespace quadratic_two_distinct_roots_l142_14283

theorem quadratic_two_distinct_roots :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1^2 - 3 = 0 ∧ 2 * x2^2 - 3 = 0) :=
by
  sorry

end quadratic_two_distinct_roots_l142_14283


namespace length_segment_AB_l142_14275

theorem length_segment_AB (A B : ℝ) (hA : A = -5) (hB : B = 2) : |A - B| = 7 :=
by
  sorry

end length_segment_AB_l142_14275


namespace smallest_integer_x_l142_14268

-- Conditions
def condition1 (x : ℤ) : Prop := 7 - 5 * x < 25
def condition2 (x : ℤ) : Prop := ∃ y : ℤ, y = 10 ∧ y - 3 * x > 6

-- Statement
theorem smallest_integer_x : ∃ x : ℤ, condition1 x ∧ condition2 x ∧ ∀ z : ℤ, condition1 z ∧ condition2 z → x ≤ z :=
  sorry

end smallest_integer_x_l142_14268


namespace sin_alpha_neg_point_two_l142_14250

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end sin_alpha_neg_point_two_l142_14250


namespace smallest_sum_divisible_by_3_l142_14248

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def is_consecutive_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4 ∨ p2 = p1 + 6 ∨ p2 = p1 + 2) ∧
  (p3 = p2 + 2 ∨ p3 = p2 + 4) ∧
  (p4 = p3 + 2 ∨ p4 = p3 + 4)

def greater_than_5 (p : ℕ) : Prop := p > 5

theorem smallest_sum_divisible_by_3 :
  ∃ (p1 p2 p3 p4 : ℕ), is_consecutive_prime p1 p2 p3 p4 ∧
                      greater_than_5 p1 ∧
                      (p1 + p2 + p3 + p4) % 3 = 0 ∧
                      (p1 + p2 + p3 + p4) = 48 :=
by sorry

end smallest_sum_divisible_by_3_l142_14248


namespace min_trips_correct_l142_14225

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end min_trips_correct_l142_14225


namespace rocket_travel_time_l142_14220

/-- The rocket's distance formula as an arithmetic series sum.
    We need to prove that the rocket reaches 240 km after 15 seconds
    given the conditions in the problem. -/
theorem rocket_travel_time :
  ∃ n : ℕ, (2 * n + (n * (n - 1))) / 2 = 240 ∧ n = 15 :=
by
  sorry

end rocket_travel_time_l142_14220


namespace cylinder_volume_multiplication_factor_l142_14224

theorem cylinder_volume_multiplication_factor (r h : ℝ) (h_r_positive : r > 0) (h_h_positive : h > 0) :
  let V := π * r^2 * h
  let V' := π * (2.5 * r)^2 * (3 * h)
  let X := V' / V
  X = 18.75 :=
by
  -- Proceed with the proof here
  sorry

end cylinder_volume_multiplication_factor_l142_14224


namespace total_pens_is_50_l142_14255

theorem total_pens_is_50
  (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) (total : ℕ)
  (h1 : red = 8)
  (h2 : black = 3 / 2 * red)
  (h3 : blue = black + 5 ∧ blue = 1 / 5 * total)
  (h4 : green = blue / 2)
  (h5 : purple = 5)
  : total = red + black + blue + green + purple := sorry

end total_pens_is_50_l142_14255


namespace problem_condition_l142_14263

noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry
noncomputable def x : ℤ := sorry
noncomputable def a : ℤ := 0
noncomputable def b : ℤ := -m + n

theorem problem_condition 
  (h1 : m ≠ 0)
  (h2 : n ≠ 0)
  (h3 : m ≠ n)
  (h4 : (x + m)^2 - (x^2 + n^2) = (m - n)^2) :
  x = a * m + b * n :=
sorry

end problem_condition_l142_14263


namespace arc_length_of_sector_l142_14298

theorem arc_length_of_sector (theta : ℝ) (r : ℝ) (h_theta : theta = 90) (h_r : r = 6) : 
  (theta / 360) * 2 * Real.pi * r = 3 * Real.pi :=
by
  sorry

end arc_length_of_sector_l142_14298


namespace oranges_per_tree_correct_l142_14286

-- Definitions for the conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def total_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * total_oranges
def seeds_planted := 2 * frank_oranges
def total_trees := seeds_planted
def total_oranges_picked := 810
def oranges_per_tree := total_oranges_picked / total_trees

-- Theorem statement
theorem oranges_per_tree_correct : oranges_per_tree = 5 :=
by
  -- Proof steps would go here
  sorry

end oranges_per_tree_correct_l142_14286


namespace rhind_papyrus_max_bread_l142_14296

theorem rhind_papyrus_max_bread
  (a1 a2 a3 a4 a5 : ℕ) (d : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 100)
  (h2 : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a4 = a1 + 3 * d)
  (h6 : a5 = a1 + 4 * d)
  (h7 : a3 + a4 + a5 = 3 * (a1 + a2)) :
  a5 = 30 :=
by {
  sorry
}

end rhind_papyrus_max_bread_l142_14296


namespace minimum_votes_for_tall_to_win_l142_14289

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l142_14289


namespace calc_expression_l142_14273

theorem calc_expression : 
  abs (Real.sqrt 3 - 2) + (8:ℝ)^(1/3) - Real.sqrt 16 + (-1)^(2023:ℝ) = -(Real.sqrt 3) - 1 :=
by
  sorry

end calc_expression_l142_14273


namespace smallest_number_of_students_l142_14281

theorem smallest_number_of_students 
  (ninth_to_seventh : ℕ → ℕ → Prop)
  (ninth_to_sixth : ℕ → ℕ → Prop) 
  (r1 : ninth_to_seventh 3 2) 
  (r2 : ninth_to_sixth 7 4) : 
  ∃ n7 n6 n9, 
    ninth_to_seventh n9 n7 ∧ 
    ninth_to_sixth n9 n6 ∧ 
    n9 + n7 + n6 = 47 :=
sorry

end smallest_number_of_students_l142_14281


namespace work_efficiency_ratio_l142_14214

variable (A B : ℝ)
variable (h1 : A = 1 / 2 * B) 
variable (h2 : 1 / (A + B) = 13)
variable (h3 : B = 1 / 19.5)

theorem work_efficiency_ratio : A / B = 1 / 2 := by
  sorry

end work_efficiency_ratio_l142_14214


namespace complement_intersection_l142_14201

-- Defining the universal set U and subsets A and B
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 3, 4}
def B : Finset ℕ := {3, 4, 5}

-- Proving the complement of the intersection of A and B in U
theorem complement_intersection : (U \ (A ∩ B)) = {1, 2, 5} :=
by sorry

end complement_intersection_l142_14201


namespace total_balloons_l142_14240

theorem total_balloons:
  ∀ (R1 R2 G1 G2 B1 B2 Y1 Y2 O1 O2: ℕ),
    R1 = 31 →
    R2 = 24 →
    G1 = 15 →
    G2 = 7 →
    B1 = 12 →
    B2 = 14 →
    Y1 = 18 →
    Y2 = 20 →
    O1 = 10 →
    O2 = 16 →
    (R1 + R2 = 55) ∧
    (G1 + G2 = 22) ∧
    (B1 + B2 = 26) ∧
    (Y1 + Y2 = 38) ∧
    (O1 + O2 = 26) :=
by
  intros
  sorry

end total_balloons_l142_14240


namespace manager_salary_is_3600_l142_14215

noncomputable def manager_salary (M : ℕ) : ℕ :=
  let total_salary_20 := 20 * 1500
  let new_average_salary := 1600
  let total_salary_21 := 21 * new_average_salary
  total_salary_21 - total_salary_20

theorem manager_salary_is_3600 : manager_salary 3600 = 3600 := by
  sorry

end manager_salary_is_3600_l142_14215


namespace function_quadrants_l142_14236

theorem function_quadrants (n : ℝ) (h: ∀ x : ℝ, x ≠ 0 → ((n-1)*x * x > 0)) : n > 1 :=
sorry

end function_quadrants_l142_14236


namespace total_diagonals_in_rectangular_prism_l142_14259

-- We define the rectangular prism with its properties
structure RectangularPrism :=
  (vertices : ℕ)
  (edges : ℕ)
  (distinct_dimensions : ℕ)

-- We specify the conditions for the rectangular prism
def givenPrism : RectangularPrism :=
{
  vertices := 8,
  edges := 12,
  distinct_dimensions := 3
}

-- We assert the total number of diagonals in the rectangular prism
theorem total_diagonals_in_rectangular_prism (P : RectangularPrism) : P = givenPrism → ∃ diag, diag = 16 :=
by
  intro h
  have diag := 16
  use diag
  sorry

end total_diagonals_in_rectangular_prism_l142_14259


namespace problem_x_y_z_l142_14249

theorem problem_x_y_z (x y z : ℕ) (h1 : xy + z = 47) (h2 : yz + x = 47) (h3 : xz + y = 47) : x + y + z = 48 :=
sorry

end problem_x_y_z_l142_14249


namespace arccos_one_over_sqrt_two_l142_14202

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l142_14202


namespace maximum_items_6_yuan_l142_14216

theorem maximum_items_6_yuan :
  ∃ (x : ℕ), (∀ (x' : ℕ), (∃ (y z : ℕ), 6 * x' + 4 * y + 2 * z = 60 ∧ x' + y + z = 16) →
    x' ≤ 7) → x = 7 :=
by
  sorry

end maximum_items_6_yuan_l142_14216


namespace cos_negative_570_equals_negative_sqrt3_div_2_l142_14276

theorem cos_negative_570_equals_negative_sqrt3_div_2 : Real.cos (-570 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_negative_570_equals_negative_sqrt3_div_2_l142_14276


namespace point_on_line_l142_14257

theorem point_on_line : ∀ (t : ℤ), 
  (∃ m : ℤ, (6 - 2) * m = 20 - 8 ∧ (10 - 6) * m = 32 - 20) →
  (∃ b : ℤ, 8 - 2 * m = b) →
  t = m * 35 + b → t = 107 :=
by
  sorry

end point_on_line_l142_14257


namespace files_rem_nat_eq_two_l142_14204

-- Conditions
def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23

-- Correct Answer
def files_remaining : ℕ := initial_music_files + initial_video_files - files_deleted

theorem files_rem_nat_eq_two : files_remaining = 2 := by
  sorry

end files_rem_nat_eq_two_l142_14204


namespace vann_teeth_cleaning_l142_14231

def numDogsCleaned (D : Nat) : Prop :=
  let dogTeethCount := 42
  let catTeethCount := 30
  let pigTeethCount := 28
  let numCats := 10
  let numPigs := 7
  let totalTeeth := 706
  dogTeethCount * D + catTeethCount * numCats + pigTeethCount * numPigs = totalTeeth

theorem vann_teeth_cleaning : numDogsCleaned 5 :=
by
  sorry

end vann_teeth_cleaning_l142_14231


namespace tan_alpha_add_pi_over_4_l142_14287

open Real

theorem tan_alpha_add_pi_over_4 
  (α : ℝ)
  (h1 : tan α = sqrt 3) : 
  tan (α + π / 4) = -2 - sqrt 3 :=
by
  sorry

end tan_alpha_add_pi_over_4_l142_14287


namespace expected_value_die_l142_14285

noncomputable def expected_value (P_Star P_Moon : ℚ) (win_Star lose_Moon : ℚ) : ℚ :=
  P_Star * win_Star + P_Moon * lose_Moon

theorem expected_value_die :
  expected_value (2/5) (3/5) 4 (-3) = -1/5 := by
  sorry

end expected_value_die_l142_14285


namespace binomial_7_4_eq_35_l142_14269

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l142_14269


namespace Megan_popsicles_l142_14288

def minutes_in_hour : ℕ := 60

def total_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * minutes_in_hour + minutes

def popsicle_time : ℕ := 18

def popsicles_consumed (total_minutes : ℕ) (popsicle_time : ℕ) : ℕ :=
  total_minutes / popsicle_time

theorem Megan_popsicles (hours : ℕ) (minutes : ℕ) (popsicle_time : ℕ)
  (total_minutes : ℕ) (h_hours : hours = 5) (h_minutes : minutes = 36) (h_popsicle_time : popsicle_time = 18)
  (h_total_minutes : total_minutes = (5 * 60 + 36)) :
  popsicles_consumed 336 popsicle_time = 18 :=
by 
  sorry

end Megan_popsicles_l142_14288


namespace cost_of_monogramming_each_backpack_l142_14234

def number_of_backpacks : ℕ := 5
def original_price_per_backpack : ℝ := 20.00
def discount_rate : ℝ := 0.20
def total_cost : ℝ := 140.00

theorem cost_of_monogramming_each_backpack : 
  (total_cost - (number_of_backpacks * (original_price_per_backpack * (1 - discount_rate)))) / number_of_backpacks = 12.00 :=
by
  sorry 

end cost_of_monogramming_each_backpack_l142_14234


namespace range_of_a_l142_14278

variable {α : Type*} [LinearOrderedField α]

def setA (a : α) : Set α := {x | abs (x - a) < 1}
def setB : Set α := {x | 1 < x ∧ x < 5}

theorem range_of_a (a : α) (h : setA a ∩ setB = ∅) : a ≤ 0 ∨ a ≥ 6 :=
sorry

end range_of_a_l142_14278


namespace smallest_n_for_divisibility_problem_l142_14266

theorem smallest_n_for_divisibility_problem :
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n + 1) ≠ 0 ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ ¬ (n * (n + 1)) % k = 0) ∧
  ∀ m : ℕ, m > 0 ∧ m < n → (∀ k : ℕ, 1 ≤ k ∧ k ≤ m → (m * (m + 1)) % k ≠ 0)) → n = 4 := sorry

end smallest_n_for_divisibility_problem_l142_14266


namespace yoojung_namjoon_total_flowers_l142_14280

theorem yoojung_namjoon_total_flowers
  (yoojung_flowers : ℕ)
  (namjoon_flowers : ℕ)
  (yoojung_condition : yoojung_flowers = 4 * namjoon_flowers)
  (yoojung_count : yoojung_flowers = 32) :
  yoojung_flowers + namjoon_flowers = 40 :=
by
  sorry

end yoojung_namjoon_total_flowers_l142_14280


namespace min_dot_product_l142_14293

theorem min_dot_product (m n : ℝ) (x1 x2 : ℝ)
    (h1 : m ≠ 0) 
    (h2 : n ≠ 0)
    (h3 : (x1 + 2) * (x2 - x1) + m * x1 * (n - m * x1) = 0) :
    ∃ (x1 : ℝ), (x1 = -2 / (m^2 + 1)) → 
    (x1 + 2) * (x2 + 2) + m * n * x1 = 4 * m^2 / (m^2 + 1) := 
sorry

end min_dot_product_l142_14293


namespace insurance_compensation_l142_14265

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end insurance_compensation_l142_14265


namespace problem_statement_l142_14217

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

theorem problem_statement : 3 * g 3 - g 9 = -48 - 6 * Real.sqrt 3 := by
  sorry

end problem_statement_l142_14217


namespace solve_asterisk_l142_14253

theorem solve_asterisk (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end solve_asterisk_l142_14253


namespace happy_children_count_l142_14294

-- Definitions of the conditions
def total_children : ℕ := 60
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4
def boys_neither_happy_nor_sad : ℕ := 10

-- The theorem we wish to prove
theorem happy_children_count :
  total_children - sad_children - neither_happy_nor_sad_children = 30 :=
by 
  -- Placeholder for the proof
  sorry

end happy_children_count_l142_14294


namespace cost_of_fencing_l142_14218

-- Definitions of ratio and area conditions
def sides_ratio (length width : ℕ) : Prop := length / width = 3 / 2
def area (length width : ℕ) : Prop := length * width = 3750

-- Define the cost per meter in paise
def cost_per_meter : ℕ := 70

-- Convert paise to rupees
def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

-- The main statement we want to prove
theorem cost_of_fencing (length width perimeter : ℕ)
  (H1 : sides_ratio length width)
  (H2 : area length width)
  (H3 : perimeter = 2 * length + 2 * width) :
  paise_to_rupees (perimeter * cost_per_meter) = 175 := by
  sorry

end cost_of_fencing_l142_14218


namespace necessary_but_not_sufficient_l142_14223

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l142_14223


namespace scientific_notation_of_2200_l142_14262

-- Define scientific notation criteria
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ a ∧ a < 10

-- Problem statement
theorem scientific_notation_of_2200 : ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 2200 ∧ a = 2.2 ∧ n = 3 :=
by {
  -- Proof can be added here.
  sorry
}

end scientific_notation_of_2200_l142_14262


namespace juan_speed_l142_14297

-- Statement of given distances and time
def distance : ℕ := 80
def time : ℕ := 8

-- Desired speed in miles per hour
def expected_speed : ℕ := 10

-- Theorem statement: Speed is distance divided by time and should equal 10 miles per hour
theorem juan_speed : distance / time = expected_speed :=
  by
  sorry

end juan_speed_l142_14297


namespace problem_statement_l142_14277

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : 0 < c) (h6 : c < 1) :
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) :=
sorry

end problem_statement_l142_14277


namespace point_in_second_quadrant_l142_14243

def in_second_quadrant (z : Complex) : Prop := 
  z.re < 0 ∧ z.im > 0

theorem point_in_second_quadrant : in_second_quadrant (Complex.ofReal (1) + 2 * Complex.I / (Complex.ofReal (1) - Complex.I)) :=
by sorry

end point_in_second_quadrant_l142_14243


namespace ratio_is_five_thirds_l142_14203

noncomputable def ratio_of_numbers (a b : ℝ) : Prop :=
  (a + b = 4 * (a - b)) → (a = 2 * b) → (a / b = 5 / 3)

theorem ratio_is_five_thirds {a b : ℝ} (h1 : a + b = 4 * (a - b)) (h2 : a = 2 * b) :
  a / b = 5 / 3 :=
  sorry

end ratio_is_five_thirds_l142_14203


namespace value_of_ab_l142_14270

theorem value_of_ab (a b c : ℝ) (C : ℝ) (h1 : (a + b) ^ 2 - c ^ 2 = 4) (h2 : C = Real.pi / 3) : 
  a * b = 4 / 3 :=
by
  sorry

end value_of_ab_l142_14270


namespace division_example_l142_14205

theorem division_example : 72 / (6 / 3) = 36 :=
by sorry

end division_example_l142_14205


namespace retail_price_l142_14260

theorem retail_price (R : ℝ) (wholesale_price : ℝ)
  (discount_rate : ℝ) (profit_rate : ℝ)
  (selling_price : ℝ) :
  wholesale_price = 81 →
  discount_rate = 0.10 →
  profit_rate = 0.20 →
  selling_price = wholesale_price * (1 + profit_rate) →
  selling_price = R * (1 - discount_rate) →
  R = 108 := 
by 
  intros h_wholesale h_discount h_profit h_selling_price h_discounted_selling_price
  sorry

end retail_price_l142_14260


namespace evaluate_product_eq_l142_14267

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product_eq : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 885735 := 
sorry

end evaluate_product_eq_l142_14267
