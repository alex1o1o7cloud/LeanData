import Mathlib

namespace inequality_proof_l159_159011

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  (1 - 2 * x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2 * y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2 * z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) :=
by
  sorry

end inequality_proof_l159_159011


namespace seq_formula_l159_159718

theorem seq_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n - 1) + 1 := 
by 
  sorry

end seq_formula_l159_159718


namespace probability_of_selecting_one_defective_l159_159006

/-- A statement about the probability of selecting exactly 1 defective item from a batch -/
theorem probability_of_selecting_one_defective (good_items : ℕ) (defective_items : ℕ) (selected_items : ℕ) :
  (good_items = 6) → (defective_items = 2) → (selected_items = 3) →
  let X := (C(good_items, 2) * C(defective_items, 1)) / C(good_items + defective_items, 3) in
  X = (15 / 28) := by
  intros h1 h2 h3
  sorry

noncomputable def C (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

end probability_of_selecting_one_defective_l159_159006


namespace first_variety_cost_l159_159541

noncomputable def cost_of_second_variety : ℝ := 8.75
noncomputable def ratio_of_first_variety : ℚ := 5 / 6
noncomputable def ratio_of_second_variety : ℚ := 1 - ratio_of_first_variety
noncomputable def cost_of_mixture : ℝ := 7.50

theorem first_variety_cost :
  ∃ x : ℝ, x * (ratio_of_first_variety : ℝ) + cost_of_second_variety * (ratio_of_second_variety : ℝ) = cost_of_mixture * (ratio_of_first_variety + ratio_of_second_variety : ℝ) 
    ∧ x = 7.25 :=
sorry

end first_variety_cost_l159_159541


namespace probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l159_159978

noncomputable def defect_rate_first_lathe : ℝ := 0.06
noncomputable def defect_rate_second_lathe : ℝ := 0.05
noncomputable def defect_rate_third_lathe : ℝ := 0.05
noncomputable def proportion_first_lathe : ℝ := 0.25
noncomputable def proportion_second_lathe : ℝ := 0.30
noncomputable def proportion_third_lathe : ℝ := 0.45

theorem probability_defective_first_lathe :
  defect_rate_first_lathe * proportion_first_lathe = 0.015 :=
by sorry

theorem overall_probability_defective :
  defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe = 0.0525 :=
by sorry

theorem conditional_probability_second_lathe :
  (defect_rate_second_lathe * proportion_second_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 2 / 7 :=
by sorry

theorem conditional_probability_third_lathe :
  (defect_rate_third_lathe * proportion_third_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 3 / 7 :=
by sorry

end probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l159_159978


namespace sin_cos_power_four_l159_159165

theorem sin_cos_power_four (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := 
sorry

end sin_cos_power_four_l159_159165


namespace point_A_symmetric_to_B_about_l_l159_159400

variables {A B : ℝ × ℝ} {l : ℝ → ℝ → Prop}

-- define point B
def point_B := (1, 2)

-- define the line equation x + y + 3 = 0 as a property
def line_l (x y : ℝ) := x + y + 3 = 0

-- define that A is symmetric to B about line l
def symmetric_about (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  (∀ x y : ℝ, l x y → ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = -(x + y)))
  ∧ ((A.2 - B.2) / (A.1 - B.1) * -1 = -1)

theorem point_A_symmetric_to_B_about_l :
  A = (-5, -4) →
  symmetric_about A B line_l →
  A = (-5, -4) := by
  intros _ sym
  sorry

end point_A_symmetric_to_B_about_l_l159_159400


namespace sum_of_integers_l159_159052

variable (x y : ℕ)

theorem sum_of_integers (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := 
by 
  sorry

end sum_of_integers_l159_159052


namespace percentage_given_to_close_friends_l159_159937

-- Definitions
def total_boxes : ℕ := 20
def pens_per_box : ℕ := 5
def total_pens : ℕ := total_boxes * pens_per_box
def pens_left_after_classmates : ℕ := 45

-- Proposition
theorem percentage_given_to_close_friends (P : ℝ) :
  total_boxes = 20 → pens_per_box = 5 → pens_left_after_classmates = 45 →
  (3 / 4) * (100 - P) = (pens_left_after_classmates : ℝ) →
  P = 40 :=
by
  intros h_total_boxes h_pens_per_box h_pens_left_after h_eq
  sorry

end percentage_given_to_close_friends_l159_159937


namespace kaleb_non_working_games_l159_159443

theorem kaleb_non_working_games (total_games working_game_price earning : ℕ) (h1 : total_games = 10) (h2 : working_game_price = 6) (h3 : earning = 12) :
  total_games - (earning / working_game_price) = 8 :=
by
  sorry

end kaleb_non_working_games_l159_159443


namespace tangent_line_slope_through_origin_l159_159838

theorem tangent_line_slope_through_origin :
  (∃ a : ℝ, (a^3 + a + 16 = (3 * a^2 + 1) * a ∧ a = 2)) →
  (3 * (2 : ℝ)^2 + 1 = 13) :=
by
  intro h
  -- Detailed proof goes here
  sorry

end tangent_line_slope_through_origin_l159_159838


namespace largest_a_has_integer_root_l159_159121

noncomputable theory
open_locale classical

theorem largest_a_has_integer_root :
  ∀ (a : ℤ), (∃ (x : ℤ), (∛ (x^2 - (a + 7)*x + 7*a) + ∛ 3 = 0)) → 
    a ≤ 11 :=
begin
  intro a,
  contrapose,
  push_neg,
  intros not_bound,
  have : 11 < a := by linarith,
  sorry,
end

end largest_a_has_integer_root_l159_159121


namespace fraction_identity_l159_159344

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end fraction_identity_l159_159344


namespace Greatest_Percentage_Difference_l159_159555

def max_percentage_difference (B W P : ℕ) : ℕ :=
  ((max B (max W P) - min B (min W P)) * 100) / (min B (min W P))

def January_B : ℕ := 6
def January_W : ℕ := 4
def January_P : ℕ := 5

def February_B : ℕ := 7
def February_W : ℕ := 5
def February_P : ℕ := 6

def March_B : ℕ := 7
def March_W : ℕ := 7
def March_P : ℕ := 7

def April_B : ℕ := 5
def April_W : ℕ := 6
def April_P : ℕ := 7

def May_B : ℕ := 3
def May_W : ℕ := 4
def May_P : ℕ := 2

theorem Greatest_Percentage_Difference :
  max_percentage_difference May_B May_W May_P >
  max (max_percentage_difference January_B January_W January_P)
      (max (max_percentage_difference February_B February_W February_P)
           (max (max_percentage_difference March_B March_W March_P)
                (max_percentage_difference April_B April_W April_P))) :=
by
  sorry

end Greatest_Percentage_Difference_l159_159555


namespace ratio_mom_pays_to_total_cost_l159_159547

-- Definitions based on the conditions from the problem
def num_shirts := 4
def num_pants := 2
def num_jackets := 2
def cost_per_shirt := 8
def cost_per_pant := 18
def cost_per_jacket := 60
def amount_carrie_pays := 94

-- Calculate total costs based on given definitions
def cost_shirts := num_shirts * cost_per_shirt
def cost_pants := num_pants * cost_per_pant
def cost_jackets := num_jackets * cost_per_jacket
def total_cost := cost_shirts + cost_pants + cost_jackets

-- Amount Carrie's mom pays
def amount_mom_pays := total_cost - amount_carrie_pays

-- The proving statement
theorem ratio_mom_pays_to_total_cost : (amount_mom_pays : ℝ) / (total_cost : ℝ) = 1 / 2 :=
by
  sorry

end ratio_mom_pays_to_total_cost_l159_159547


namespace proof_statements_l159_159075

namespace ProofProblem

-- Definitions for each condition
def is_factor (x y : ℕ) : Prop := ∃ n : ℕ, y = n * x
def is_divisor (x y : ℕ) : Prop := is_factor x y

-- Lean 4 statement for the problem
theorem proof_statements :
  is_factor 4 20 ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (¬ is_divisor 12 75 ∧ ¬ is_divisor 12 29) ∧
  (is_divisor 11 33 ∧ ¬ is_divisor 11 64) ∧
  is_factor 9 180 :=
by
  sorry

end ProofProblem

end proof_statements_l159_159075


namespace f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l159_159360

noncomputable def f (x : ℝ) : ℝ := (1 / 4) ^ x + (1 / 2) ^ x - 1
noncomputable def g (x m : ℝ) : ℝ := (1 - m * 2 ^ x) / (1 + m * 2 ^ x)

theorem f_range_and_boundedness :
  ∀ x : ℝ, x < 0 → 1 < f x ∧ ¬(∃ M : ℝ, ∀ x : ℝ, x < 0 → |f x| ≤ M) :=
by sorry

theorem g_odd_and_bounded (x : ℝ) :
  g x 1 = -g (-x) 1 ∧ |g x 1| < 1 :=
by sorry

theorem g_upper_bound (m : ℝ) (hm : 0 < m ∧ m < 1 / 2) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g x m ≤ (1 - m) / (1 + m) :=
by sorry

end f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l159_159360


namespace jamesons_sword_length_l159_159381

theorem jamesons_sword_length (c j j' : ℕ) (hC: c = 15) 
  (hJ: j = c + 23) (hJJ: j' = j - 5) : 
  j' = 2 * c + 3 := by 
  sorry

end jamesons_sword_length_l159_159381


namespace range_of_m_l159_159577

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4 * x - m < 0 ∧ -1 ≤ x ∧ x ≤ 2) →
  (∃ x : ℝ, x^2 - x - 2 > 0) →
  (∀ x : ℝ, 4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) →
  m > 8 :=
sorry

end range_of_m_l159_159577


namespace mary_number_l159_159291

-- Definitions of the properties and conditions
def is_two_digit_number (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

def switch_digits (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def conditions_met (x : ℕ) : Prop :=
  is_two_digit_number x ∧ 91 ≤ switch_digits (4 * x - 7) ∧ switch_digits (4 * x - 7) ≤ 95

-- The statement to prove
theorem mary_number : ∃ x : ℕ, conditions_met x ∧ x = 14 :=
by {
  sorry
}

end mary_number_l159_159291


namespace daisies_given_l159_159614

theorem daisies_given (S : ℕ) (h : (5 + S) / 2 = 7) : S = 9 := by
  sorry

end daisies_given_l159_159614


namespace min_abs_val_sum_l159_159082

noncomputable
def g (z : ℂ) (β δ : ℂ) : ℂ :=
  (5 + 3 * complex.I) * z^3 + β * z + δ

theorem min_abs_val_sum (β δ : ℂ) (g_real_1 : g 1 β δ ∈ ℝ) (g_real_neg_i : g (-complex.I) β δ ∈ ℝ) :
  ∃ a b c d : ℝ, β = a + b * complex.I ∧ δ = c + d * complex.I ∧
  a = 0 ∧ b = 0 ∧ c = 8 ∧ d = -3 ∧
  abs β + abs δ = real.sqrt 73 :=
begin
  sorry
end

end min_abs_val_sum_l159_159082


namespace roses_cut_l159_159653

def initial_roses : ℕ := 6
def new_roses : ℕ := 16

theorem roses_cut : new_roses - initial_roses = 10 := by
  sorry

end roses_cut_l159_159653


namespace area_of_rectangular_field_l159_159684

-- Definitions from conditions
def L : ℕ := 20
def total_fencing : ℕ := 32

-- Additional variables inferred from the conditions
def W : ℕ := (total_fencing - L) / 2

-- The theorem statement
theorem area_of_rectangular_field : L * W = 120 :=
by
  -- Definitions and substitutions are included in the theorem proof
  sorry

end area_of_rectangular_field_l159_159684


namespace quarters_addition_l159_159465

def original_quarters : ℝ := 783.0
def added_quarters : ℝ := 271.0
def total_quarters : ℝ := 1054.0

theorem quarters_addition :
  original_quarters + added_quarters = total_quarters :=
by
  sorry

end quarters_addition_l159_159465


namespace f_2019_value_l159_159446

noncomputable def B : Set ℚ := {q : ℚ | q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1}

noncomputable def g (x : ℚ) (h : x ∈ B) : ℚ :=
  1 - (2 / x)

noncomputable def f (x : ℚ) (h : x ∈ B) : ℝ :=
  sorry

theorem f_2019_value (h2019 : 2019 ∈ B) :
  f 2019 h2019 = Real.log ((2019 - 0.5) ^ 2 / 2018.5) :=
sorry

end f_2019_value_l159_159446


namespace a_7_is_127_l159_159709

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 0  -- Define a_0 which is not used but useful for indexing
| 1       => 1
| (n + 2) => 2 * (a (n + 1)) + 1

-- Prove that a_7 = 127
theorem a_7_is_127 : a 7 = 127 := 
sorry

end a_7_is_127_l159_159709


namespace rectangle_y_value_l159_159479

theorem rectangle_y_value
  (E : (ℝ × ℝ)) (F : (ℝ × ℝ)) (G : (ℝ × ℝ)) (H : (ℝ × ℝ))
  (hE : E = (0, 0)) (hF : F = (0, 5)) (hG : ∃ y : ℝ, G = (y, 5))
  (hH : ∃ y : ℝ, H = (y, 0)) (area : ℝ) (h_area : area = 35)
  (hy_pos : ∃ y : ℝ, y > 0)
  : ∃ y : ℝ, y = 7 :=
by
  sorry

end rectangle_y_value_l159_159479


namespace train_boxcars_capacity_l159_159769

theorem train_boxcars_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_capacity := 4000
  let blue_capacity := black_capacity * 2
  let red_capacity := blue_capacity * 3
  (black_boxcars * black_capacity) + (blue_boxcars * blue_capacity) + (red_boxcars * red_capacity) = 132000 := by
  sorry

end train_boxcars_capacity_l159_159769


namespace profit_in_may_highest_monthly_profit_and_max_value_l159_159358

def f (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28 else 200 - 14 * x

theorem profit_in_may :
  f 5 = 88 :=
by sorry

theorem highest_monthly_profit_and_max_value :
  ∃ x, 1 ≤ x ∧ x ≤ 12 ∧ f x = 102 :=
by sorry

end profit_in_may_highest_monthly_profit_and_max_value_l159_159358


namespace balloons_difference_l159_159076

theorem balloons_difference (yours friends : ℝ) (hyours : yours = -7) (hfriends : friends = 4.5) :
  friends - yours = 11.5 :=
by
  rw [hyours, hfriends]
  sorry

end balloons_difference_l159_159076


namespace find_f_5pi_div_3_l159_159891

variable (f : ℝ → ℝ)

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem find_f_5pi_div_3
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_function f π)
  (h_def : ∀ x, 0 ≤ x → x ≤ π/2 → f x = Real.sin x) :
  f (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end find_f_5pi_div_3_l159_159891


namespace cube_root_neg_frac_l159_159050

theorem cube_root_neg_frac : (-(1/3 : ℝ))^3 = - 1 / 27 := by
  sorry

end cube_root_neg_frac_l159_159050


namespace part1_part2_l159_159138

def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + a + 1

-- Proof problem 1: Prove that if a = 2, then f(x) ≥ 0 is equivalent to x ≥ 3/2 or x ≤ 1.
theorem part1 (x : ℝ) : f 2 x ≥ 0 ↔ x ≥ (3 / 2 : ℝ) ∨ x ≤ 1 := sorry

-- Proof problem 2: Prove that for a∈[-2,2], if f(x) < 0 always holds, then x ∈ (1, 3/2).
theorem part2 (a x : ℝ) (ha : a ≥ -2 ∧ a ≤ 2) : (∀ x, f a x < 0) ↔ 1 < x ∧ x < (3 / 2 : ℝ) := sorry

end part1_part2_l159_159138


namespace initial_amount_in_cookie_jar_l159_159494

theorem initial_amount_in_cookie_jar (doris_spent : ℕ) (martha_spent : ℕ) (amount_left : ℕ) (spent_eq_martha : martha_spent = doris_spent / 2) (amount_left_eq : amount_left = 12) (doris_spent_eq : doris_spent = 6) : (doris_spent + martha_spent + amount_left = 21) :=
by
  sorry

end initial_amount_in_cookie_jar_l159_159494


namespace hyperbola_asymptote_l159_159140

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 ∧ ∀ (x y : ℝ), (y = 3/5 * x ↔ y = 3 / 5 * x)) → a = 5 :=
by
  sorry

end hyperbola_asymptote_l159_159140


namespace candle_blow_out_l159_159173

-- Definitions related to the problem.
def funnel := true -- Simplified representation of the funnel
def candle_lit := true -- Simplified representation of the lit candle
def airflow_concentration (align: Bool) : Prop :=
if align then true -- Airflow intersects the flame correctly
else false -- Airflow does not intersect the flame correctly

theorem candle_blow_out (align : Bool) : funnel ∧ candle_lit ∧ airflow_concentration align → align := sorry

end candle_blow_out_l159_159173


namespace solve_problem_l159_159232

theorem solve_problem (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
    (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end solve_problem_l159_159232


namespace find_initial_avg_height_l159_159177

noncomputable def initially_calculated_avg_height (A : ℚ) (boys : ℕ) (wrong_height right_height : ℚ) (actual_avg_height : ℚ) :=
  boys = 35 ∧
  wrong_height = 166 ∧
  right_height = 106 ∧
  actual_avg_height = 182 ∧
  35 * A - (wrong_height - right_height) = 35 * actual_avg_height

theorem find_initial_avg_height : ∃ A : ℚ, initially_calculated_avg_height A 35 166 106 182 ∧ A = 183.71 :=
by
  sorry

end find_initial_avg_height_l159_159177


namespace smallest_m_plus_n_l159_159834

theorem smallest_m_plus_n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_lt : m < n)
    (h_eq : 1978^m % 1000 = 1978^n % 1000) : m + n = 26 :=
sorry

end smallest_m_plus_n_l159_159834


namespace sample_size_second_grade_l159_159063

theorem sample_size_second_grade
    (total_students : ℕ)
    (ratio_first : ℕ)
    (ratio_second : ℕ)
    (ratio_third : ℕ)
    (sample_size : ℕ) :
    total_students = 2000 →
    ratio_first = 5 → ratio_second = 3 → ratio_third = 2 →
    sample_size = 20 →
    (20 * (3 / (5 + 3 + 2)) = 6) :=
by
  intros ht hr1 hr2 hr3 hs
  -- The proof would continue from here, but we're finished as the task only requires the statement.
  sorry

end sample_size_second_grade_l159_159063


namespace perpendicular_planes_implies_perpendicular_line_l159_159012

-- Definitions of lines and planes and their properties in space
variable {Space : Type}
variable (m n l : Line Space) -- Lines in space
variable (α β γ : Plane Space) -- Planes in space

-- Conditions: m, n, and l are non-intersecting lines, α, β, and γ are non-intersecting planes
axiom non_intersecting_lines : ¬ (m = n) ∧ ¬ (m = l) ∧ ¬ (n = l)
axiom non_intersecting_planes : ¬ (α = β) ∧ ¬ (α = γ) ∧ ¬ (β = γ)

-- To prove: if α ⊥ γ, β ⊥ γ, and α ∩ β = l, then l ⊥ γ
theorem perpendicular_planes_implies_perpendicular_line
  (h1 : α ⊥ γ) 
  (h2 : β ⊥ γ)
  (h3 : α ∩ β = l) : l ⊥ γ := 
  sorry

end perpendicular_planes_implies_perpendicular_line_l159_159012


namespace greatest_divisor_of_420_and_90_l159_159500

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- Main problem statement
theorem greatest_divisor_of_420_and_90 {d : ℕ} :
  (divides d 420) ∧ (d < 60) ∧ (divides d 90) → d ≤ 30 := 
sorry

end greatest_divisor_of_420_and_90_l159_159500


namespace vasya_birthday_l159_159791

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159791


namespace distance_between_points_eq_l159_159872

noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points_eq :
  dist 1 5 7 2 = 3 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_eq_l159_159872


namespace gcd_of_198_and_286_l159_159330

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end gcd_of_198_and_286_l159_159330


namespace count_multiples_of_7_not_14_lt_500_l159_159262

theorem count_multiples_of_7_not_14_lt_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0}.to_finset.card = 36 := 
by 
sor	 

end count_multiples_of_7_not_14_lt_500_l159_159262


namespace simplify_and_evaluate_l159_159045

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2) :
  ( ( (2 * x - 1) / (x + 1) - x + 1 ) / (x - 2) / (x^2 + 2 * x + 1) ) = -2 - Real.sqrt 2 :=
by sorry

end simplify_and_evaluate_l159_159045


namespace eliminate_all_evil_with_at_most_one_good_l159_159180

-- Defining the problem setting
structure Wizard :=
  (is_good : Bool)

-- The main theorem
theorem eliminate_all_evil_with_at_most_one_good (wizards : List Wizard) (h_wizard_count : wizards.length = 2015) :
  ∃ (banish_sequence : List Wizard), 
    (∀ w ∈ banish_sequence, w.is_good = false) ∨ (∃ (g : Wizard), g.is_good = true ∧ g ∉ banish_sequence) :=
sorry

end eliminate_all_evil_with_at_most_one_good_l159_159180


namespace dacid_average_marks_is_75_l159_159832

/-- Defining the marks obtained in each subject as constants -/
def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

/-- Total marks calculation -/
def total_marks : ℕ :=
  english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks

/-- Number of subjects -/
def number_of_subjects : ℕ := 5

/-- Average marks calculation -/
def average_marks : ℕ :=
  total_marks / number_of_subjects

/-- Theorem proving that Dacid's average marks is 75 -/
theorem dacid_average_marks_is_75 : average_marks = 75 :=
  sorry

end dacid_average_marks_is_75_l159_159832


namespace find_start_number_l159_159271

def count_even_not_divisible_by_3 (start end_ : ℕ) : ℕ :=
  (end_ / 2 + 1) - (end_ / 6 + 1) - (if start = 0 then start / 2 else start / 2 + 1 - (start - 1) / 6 - 1)

theorem find_start_number (start end_ : ℕ) (h1 : end_ = 170) (h2 : count_even_not_divisible_by_3 start end_ = 54) : start = 8 :=
by 
  rw [h1] at h2
  sorry

end find_start_number_l159_159271


namespace center_of_circle_l159_159278

theorem center_of_circle (ρ θ : ℝ) (h : ρ = 2 * Real.cos (θ - π / 4)) : (ρ, θ) = (1, π / 4) :=
sorry

end center_of_circle_l159_159278


namespace eleven_pow_2048_mod_17_l159_159335

theorem eleven_pow_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end eleven_pow_2048_mod_17_l159_159335


namespace vasya_birthday_is_thursday_l159_159805

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159805


namespace solve_linear_system_l159_159941

theorem solve_linear_system :
  ∃ x y : ℤ, x + 9773 = 13200 ∧ 2 * x - 3 * y = 1544 ∧ x = 3427 ∧ y = 1770 := by
  sorry

end solve_linear_system_l159_159941


namespace perfect_square_trinomial_l159_159248

theorem perfect_square_trinomial (k : ℤ) : (∀ x : ℤ, x^2 + 2 * (k + 1) * x + 16 = (x + (k + 1))^2) → (k = 3 ∨ k = -5) :=
by
  sorry

end perfect_square_trinomial_l159_159248


namespace second_candidate_marks_l159_159356

variable (T : ℝ) (pass_mark : ℝ := 160)

-- Conditions
def condition1 : Prop := 0.20 * T + 40 = pass_mark
def condition2 : Prop := 0.30 * T - pass_mark > 0 

-- The statement we want to prove
theorem second_candidate_marks (h1 : condition1 T) (h2 : condition2 T) : 
  (0.30 * T - pass_mark = 20) :=
by 
  -- Skipping proof steps as per the guidelines
  sorry

end second_candidate_marks_l159_159356


namespace function_satisfies_condition_l159_159002

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - x + 4)

theorem function_satisfies_condition :
  ∀ (x : ℝ), 2 * f (1 - x) + 1 = x * f x :=
by
  intro x
  unfold f
  sorry

end function_satisfies_condition_l159_159002


namespace Vasya_birthday_on_Thursday_l159_159800

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159800


namespace martin_total_distance_l159_159945

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l159_159945


namespace find_starting_number_l159_159787

theorem find_starting_number :
  ∃ startnum : ℕ, startnum % 5 = 0 ∧ (∀ k : ℕ, 0 ≤ k ∧ k < 20 → startnum + 5 * k ≤ 100) ∧ startnum = 10 :=
sorry

end find_starting_number_l159_159787


namespace different_gcd_values_count_l159_159824

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159824


namespace expected_value_coin_flip_l159_159544

def probability_heads : ℚ := 2 / 3
def probability_tails : ℚ := 1 / 3
def gain_heads : ℤ := 5
def loss_tails : ℤ := -9

theorem expected_value_coin_flip : (2 / 3 : ℚ) * 5 + (1 / 3 : ℚ) * (-9) = 1 / 3 :=
by sorry

end expected_value_coin_flip_l159_159544


namespace find_distance_between_stations_l159_159365

noncomputable def distance_between_stations (D T : ℝ) : Prop :=
  D = 100 * T ∧
  D = 50 * (T + 15 / 60) ∧
  D = 70 * (T + 7 / 60)

theorem find_distance_between_stations :
  ∃ D T : ℝ, distance_between_stations D T ∧ D = 25 :=
by
  sorry

end find_distance_between_stations_l159_159365


namespace smallest_X_value_l159_159620

noncomputable def T : ℕ := 111000
axiom T_digits_are_0s_and_1s : ∀ d, d ∈ (T.digits 10) → d = 0 ∨ d = 1
axiom T_divisible_by_15 : 15 ∣ T
lemma T_sum_of_digits_mul_3 : (∑ d in (T.digits 10), d) % 3 = 0 := sorry
lemma T_ends_with_0 : T.digits 10 |> List.head = some 0 := sorry

theorem smallest_X_value : ∃ X : ℕ, X = T / 15 ∧ X = 7400 := by
  use 7400
  split
  · calc 7400 = T / 15
    · rw [T]
    · exact div_eq_of_eq_mul_right (show 15 ≠ 0 from by norm_num) rfl
  · exact rfl

end smallest_X_value_l159_159620


namespace burn_down_village_in_1920_seconds_l159_159187

-- Definitions of the initial conditions
def initial_cottages : Nat := 90
def burn_interval_seconds : Nat := 480
def burn_time_per_unit : Nat := 5
def max_burns_per_interval : Nat := burn_interval_seconds / burn_time_per_unit

-- Recurrence relation for the number of cottages after n intervals
def cottages_remaining (n : Nat) : Nat :=
if n = 0 then initial_cottages
else 2 * cottages_remaining (n - 1) - max_burns_per_interval

-- Time taken to burn all cottages is when cottages_remaining(n) becomes 0
def total_burn_time_seconds (intervals : Nat) : Nat :=
intervals * burn_interval_seconds

-- Main theorem statement
theorem burn_down_village_in_1920_seconds :
  ∃ n, cottages_remaining n = 0 ∧ total_burn_time_seconds n = 1920 := by
  sorry

end burn_down_village_in_1920_seconds_l159_159187


namespace smallest_positive_integer_with_eight_factors_l159_159507

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l159_159507


namespace max_price_of_product_l159_159637

theorem max_price_of_product (x : ℝ) 
  (cond1 : (x - 10) * 0.1 = (x - 20) * 0.2) : 
  x = 30 := 
by 
  sorry

end max_price_of_product_l159_159637


namespace must_be_divisor_of_a_l159_159449

theorem must_be_divisor_of_a
    (a b c d : ℕ)
    (h1 : Nat.gcd a b = 40)
    (h2 : Nat.gcd b c = 45)
    (h3 : Nat.gcd c d = 75)
    (h4 : 120 < Nat.gcd d a ∧ Nat.gcd d a < 150) :
    5 ∣ a :=
sorry

end must_be_divisor_of_a_l159_159449


namespace base8_to_base10_problem_l159_159956

theorem base8_to_base10_problem (c d : ℕ) (h : 543 = 3*8^2 + c*8 + d) : (c * d) / 12 = 5 / 4 :=
by 
  sorry

end base8_to_base10_problem_l159_159956


namespace smallest_integer_with_eight_factors_l159_159505

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l159_159505


namespace find_values_of_a_and_b_l159_159137

-- Define the problem
theorem find_values_of_a_and_b (a b : ℚ) (h1 : a + (a / 4) = 3) (h2 : b - 2 * a = 1) :
  a = 12 / 5 ∧ b = 29 / 5 := by
  sorry

end find_values_of_a_and_b_l159_159137


namespace hockey_league_teams_l159_159980

theorem hockey_league_teams (n : ℕ) (h : (n * (n - 1) * 10) / 2 = 1710) : n = 19 :=
by {
  sorry
}

end hockey_league_teams_l159_159980


namespace yellow_yellow_pairs_l159_159689

variable (students_total : ℕ := 150)
variable (blue_students : ℕ := 65)
variable (yellow_students : ℕ := 85)
variable (total_pairs : ℕ := 75)
variable (blue_blue_pairs : ℕ := 30)

theorem yellow_yellow_pairs : 
  (yellow_students - (blue_students - blue_blue_pairs * 2)) / 2 = 40 :=
by 
  -- proof goes here
  sorry

end yellow_yellow_pairs_l159_159689


namespace slices_left_per_person_is_2_l159_159294

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end slices_left_per_person_is_2_l159_159294


namespace slopes_product_l159_159409

variables {a b c x0 y0 alpha beta : ℝ}
variables {P Q : ℝ × ℝ}
variables (M : ℝ × ℝ) (kPQ kOM : ℝ)

-- Conditions: a, b are positive real numbers
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Condition: b^2 = a c
axiom b_squared_eq_a_mul_c : b^2 = a * c

-- Condition: P and Q lie on the hyperbola
axiom P_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1
axiom Q_on_hyperbola : (Q.1^2 / a^2) - (Q.2^2 / b^2) = 1

-- Condition: M is the midpoint of P and Q
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Condition: Slopes kPQ and kOM exist
axiom kOM_def : kOM = y0 / x0
axiom kPQ_def : kPQ = beta / alpha

-- Theorem: Value of the product of the slopes
theorem slopes_product : kPQ * kOM = (1 + Real.sqrt 5) / 2 :=
sorry

end slopes_product_l159_159409


namespace distance_between_centers_l159_159462

-- Declare radii of the circles and the shortest distance between points on the circles
def R := 28
def r := 12
def d := 10

-- Define the problem to prove the distance between the centers
theorem distance_between_centers (R r d : ℝ) (hR : R = 28) (hr : r = 12) (hd : d = 10) : 
  ∀ OO1 : ℝ, OO1 = 6 :=
by sorry

end distance_between_centers_l159_159462


namespace exchange_ways_10_dollar_l159_159419

theorem exchange_ways_10_dollar (p q : ℕ) (H : 2 * p + 5 * q = 200) : 
  ∃ (n : ℕ), n = 20 :=
by {
  sorry
}

end exchange_ways_10_dollar_l159_159419


namespace total_num_novels_receiving_prizes_l159_159205

-- Definitions based on conditions
def total_prize_money : ℕ := 800
def first_place_prize : ℕ := 200
def second_place_prize : ℕ := 150
def third_place_prize : ℕ := 120
def remaining_award_amount : ℕ := 22

-- Total number of novels receiving prizes
theorem total_num_novels_receiving_prizes : 
  (3 + (total_prize_money - (first_place_prize + second_place_prize + third_place_prize)) / remaining_award_amount) = 18 :=
by {
  -- We leave the proof as an exercise (denoted by sorry)
  sorry
}

end total_num_novels_receiving_prizes_l159_159205


namespace limit_proof_l159_159100

noncomputable def limit_at_x_eq_1 : Prop :=
  (filter.tendsto (λ x : ℝ, (1 + cos (real.pi * x)) / (tan (real.pi * x)) ^ 2) (𝓝 1) (𝓝 (1 / 2)))

theorem limit_proof : limit_at_x_eq_1 :=
sorry

end limit_proof_l159_159100


namespace magic_square_d_e_sum_l159_159742

theorem magic_square_d_e_sum 
  (S : ℕ)
  (a b c d e : ℕ)
  (h1 : S = 45 + d)
  (h2 : S = 51 + e) :
  d + e = 57 :=
by
  sorry

end magic_square_d_e_sum_l159_159742


namespace rational_inequality_solution_l159_159046

open Set

theorem rational_inequality_solution (x : ℝ) :
  (x < -1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 5)) ↔ (x - 5) / ((x - 2) * (x^2 - 1)) < 0 := 
sorry

end rational_inequality_solution_l159_159046


namespace abs_sum_neq_3_nor_1_l159_159266

theorem abs_sum_neq_3_nor_1 (a b : ℤ) (h₁ : |a| = 3) (h₂ : |b| = 1) : (|a + b| ≠ 3) ∧ (|a + b| ≠ 1) := sorry

end abs_sum_neq_3_nor_1_l159_159266


namespace correlations_are_1_3_4_l159_159305

def relation1 : Prop := ∃ (age wealth : ℝ), true
def relation2 : Prop := ∀ (point : ℝ × ℝ), ∃ (coords : ℝ × ℝ), coords = point
def relation3 : Prop := ∃ (yield : ℝ) (climate : ℝ), true
def relation4 : Prop := ∃ (diameter height : ℝ), true
def relation5 : Prop := ∃ (student : Type) (school : Type), true

theorem correlations_are_1_3_4 :
  (relation1 ∨ relation3 ∨ relation4) ∧ ¬ (relation2 ∨ relation5) :=
sorry

end correlations_are_1_3_4_l159_159305


namespace tree_last_tree_height_difference_l159_159435

noncomputable def treeHeightDifference : ℝ :=
  let t1 := 1000
  let t2 := 500
  let t3 := 500
  let avgHeight := 800
  let lastTreeHeight := 4 * avgHeight - (t1 + t2 + t3)
  lastTreeHeight - t1

theorem tree_last_tree_height_difference :
  treeHeightDifference = 200 := sorry

end tree_last_tree_height_difference_l159_159435


namespace stuffed_animals_count_l159_159755

theorem stuffed_animals_count
  (total_prizes : ℕ)
  (frisbees : ℕ)
  (yoyos : ℕ)
  (h1 : total_prizes = 50)
  (h2 : frisbees = 18)
  (h3 : yoyos = 18) :
  (total_prizes - (frisbees + yoyos) = 14) :=
by
  sorry

end stuffed_animals_count_l159_159755


namespace congruence_example_l159_159263

theorem congruence_example (x : ℤ) (h : 5 * x + 3 ≡ 1 [ZMOD 18]) : 3 * x + 8 ≡ 14 [ZMOD 18] :=
sorry

end congruence_example_l159_159263


namespace intersection_of_A_and_B_l159_159410

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_of_A_and_B_l159_159410


namespace Richard_remaining_distance_l159_159768

theorem Richard_remaining_distance
  (total_distance : ℕ)
  (day1_distance : ℕ)
  (day2_distance : ℕ)
  (day3_distance : ℕ)
  (half_and_subtract : day2_distance = (day1_distance / 2) - 6)
  (total_distance_to_walk : total_distance = 70)
  (distance_day1 : day1_distance = 20)
  (distance_day3 : day3_distance = 10)
  : total_distance - (day1_distance + day2_distance + day3_distance) = 36 :=
  sorry

end Richard_remaining_distance_l159_159768


namespace value_of_expression_l159_159603

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 9 - 4 * x^2 - 6 * x = 7 := by
  sorry

end value_of_expression_l159_159603


namespace evaluate_expression_evaluate_fraction_l159_159734

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  3 * x^3 + 4 * y^3 = 337 :=
by
  sorry

theorem evaluate_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) 
  (h : 3 * x^3 + 4 * y^3 = 337) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 :=
by
  sorry

end evaluate_expression_evaluate_fraction_l159_159734


namespace compare_negatives_l159_159382

theorem compare_negatives : (- (3 : ℝ) / 5) > (- (5 : ℝ) / 7) :=
by
  sorry

end compare_negatives_l159_159382


namespace equal_clubs_and_students_l159_159426

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l159_159426


namespace olympiad_scores_above_18_l159_159924

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l159_159924


namespace calculate_total_revenue_l159_159096

-- Definitions based on conditions
def apple_pie_slices := 8
def peach_pie_slices := 6
def cherry_pie_slices := 10

def apple_pie_price := 3
def peach_pie_price := 4
def cherry_pie_price := 5

def apple_pie_customers := 88
def peach_pie_customers := 78
def cherry_pie_customers := 45

-- Definition of total revenue
def total_revenue := 
  (apple_pie_customers * apple_pie_price) + 
  (peach_pie_customers * peach_pie_price) + 
  (cherry_pie_customers * cherry_pie_price)

-- Target theorem to prove: total revenue equals 801
theorem calculate_total_revenue : total_revenue = 801 := by
  sorry

end calculate_total_revenue_l159_159096


namespace reciprocal_of_fraction_diff_l159_159702

theorem reciprocal_of_fraction_diff : 
  (∃ (a b : ℚ), a = 1/4 ∧ b = 1/5 ∧ (1 / (a - b)) = 20) :=
sorry

end reciprocal_of_fraction_diff_l159_159702


namespace union_of_A_and_B_l159_159764

open Set

-- Definitions for the conditions
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Statement of the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} :=
by
  sorry

end union_of_A_and_B_l159_159764


namespace different_prime_factors_mn_is_five_l159_159031

theorem different_prime_factors_mn_is_five {m n : ℕ} 
  (m_prime_factors : ∃ (p_1 p_2 p_3 p_4 : ℕ), True)  -- m has 4 different prime factors
  (n_prime_factors : ∃ (q_1 q_2 q_3 : ℕ), True)  -- n has 3 different prime factors
  (gcd_m_n : Nat.gcd m n = 15) : 
  (∃ k : ℕ, k = 5 ∧ (∃ (x_1 x_2 x_3 x_4 x_5 : ℕ), True)) := sorry

end different_prime_factors_mn_is_five_l159_159031


namespace find_a1_l159_159029

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

theorem find_a1 (h_arith : is_arithmetic_sequence a 3) (ha2 : a 2 = -5) : a 1 = -8 :=
sorry

end find_a1_l159_159029


namespace proof_problem_1_proof_problem_2_l159_159017

def S (n : ℕ) : Finset (Fin n → bool) :=
  Finset.univ

def d {n : ℕ} (U V : Fin n → bool) : ℕ :=
  Finset.card ((Finset.filter (fun i => U i ≠ V i) Finset.univ) : Finset (Fin n))

def problem_1 : Prop :=
  let U := fun _ => true
  let S6 := S 6
  ∃! m, m = Finset.card (Finset.filter (fun V => d U V = 2) S6) ∧ m = 15

def problem_2 (n : ℕ) (hn : n ≥ 2) : Prop :=
  let S_n := S n
  ∀ U : Fin n → bool,
  Σ (V : Finset (Fin n → bool)), (V ∈ S_n) → (d U V) = n * 2^(n - 1)

-- Problem statements without proof
theorem proof_problem_1 : problem_1 :=
by sorry

theorem proof_problem_2 (n : ℕ) (hn : n ≥ 2) : problem_2 n hn :=
by sorry

end proof_problem_1_proof_problem_2_l159_159017


namespace gravitational_force_on_space_station_l159_159057

-- Define the problem conditions and gravitational relationship
def gravitational_force_proportionality (f d : ℝ) : Prop :=
  ∃ k : ℝ, f * d^2 = k

-- Given conditions
def earth_surface_distance : ℝ := 6371
def space_station_distance : ℝ := 100000
def surface_gravitational_force : ℝ := 980
def proportionality_constant : ℝ := surface_gravitational_force * earth_surface_distance^2

-- Statement of the proof problem
theorem gravitational_force_on_space_station :
  gravitational_force_proportionality surface_gravitational_force earth_surface_distance →
  ∃ f2 : ℝ, f2 = 3.977 ∧ gravitational_force_proportionality f2 space_station_distance :=
sorry

end gravitational_force_on_space_station_l159_159057


namespace number_of_parakeets_per_cage_l159_159683

def num_cages : ℕ := 9
def parrots_per_cage : ℕ := 2
def total_birds : ℕ := 72

theorem number_of_parakeets_per_cage : (total_birds - (num_cages * parrots_per_cage)) / num_cages = 6 := by
  sorry

end number_of_parakeets_per_cage_l159_159683


namespace max_value_of_expression_eq_two_l159_159208

noncomputable def max_value_of_expression (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) : ℝ :=
  (a^2 + b^2 + c^2) / c^2

theorem max_value_of_expression_eq_two (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) :
  max_value_of_expression a b c h_right_triangle h_a = 2 := by
  sorry

end max_value_of_expression_eq_two_l159_159208


namespace fraction_evaluation_l159_159228

theorem fraction_evaluation : (3 / 8 : ℚ) + 7 / 12 - 2 / 9 = 53 / 72 := by
  sorry

end fraction_evaluation_l159_159228


namespace prove_box_problem_l159_159355

noncomputable def boxProblem : Prop :=
  let height1 := 2
  let width1 := 4
  let length1 := 6
  let clay1 := 48
  let height2 := 3 * height1
  let width2 := 2 * width1
  let length2 := 1.5 * length1
  let volume1 := height1 * width1 * length1
  let volume2 := height2 * width2 * length2
  let n := (volume2 / volume1) * clay1
  n = 432

theorem prove_box_problem : boxProblem := by
  sorry

end prove_box_problem_l159_159355


namespace material_left_eq_l159_159221

theorem material_left_eq :
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  total_bought - used = (51 / 170 : ℚ) :=
by
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  show total_bought - used = (51 / 170)
  sorry

end material_left_eq_l159_159221


namespace factorize_16x2_minus_1_l159_159117

theorem factorize_16x2_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4 * x + 1) * (4 * x - 1) := by
  sorry

end factorize_16x2_minus_1_l159_159117


namespace minimum_distance_focus_to_circle_point_l159_159883

def focus_of_parabola : ℝ × ℝ := (1, 0)
def center_of_circle : ℝ × ℝ := (4, 4)
def radius_of_circle : ℝ := 4
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16

theorem minimum_distance_focus_to_circle_point :
  ∃ P : ℝ × ℝ, circle_equation P.1 P.2 ∧ dist focus_of_parabola P = 5 :=
sorry

end minimum_distance_focus_to_circle_point_l159_159883


namespace sum_of_integers_l159_159051

variable (x y : ℕ)

theorem sum_of_integers (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := 
by 
  sorry

end sum_of_integers_l159_159051


namespace find_x_l159_159343

theorem find_x (x : ℝ) (h : 0.25 * x = 0.10 * 500 - 5) : x = 180 :=
by
  sorry

end find_x_l159_159343


namespace total_painting_area_correct_l159_159841

def barn_width : ℝ := 12
def barn_length : ℝ := 15
def barn_height : ℝ := 6

def area_to_be_painted (width length height : ℝ) : ℝ := 
  2 * (width * height + length * height) + width * length

theorem total_painting_area_correct : area_to_be_painted barn_width barn_length barn_height = 828 := 
  by sorry

end total_painting_area_correct_l159_159841


namespace count_multiples_of_7_not_14_lt_500_l159_159261

theorem count_multiples_of_7_not_14_lt_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0}.to_finset.card = 36 := 
by 
sor	 

end count_multiples_of_7_not_14_lt_500_l159_159261


namespace sum_of_numbers_in_50th_row_l159_159111

-- Defining the array and the row sum
def row_sum (n : ℕ) : ℕ :=
  2^n

-- Proposition stating that the 50th row sum is equal to 2^50
theorem sum_of_numbers_in_50th_row : row_sum 50 = 2^50 :=
by sorry

end sum_of_numbers_in_50th_row_l159_159111


namespace equal_real_roots_of_quadratic_l159_159600

theorem equal_real_roots_of_quadratic (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ (x-4)*(x-4) = 0) ↔ k = 4 ∨ k = -4 :=
by
  sorry

end equal_real_roots_of_quadratic_l159_159600


namespace cans_of_type_B_purchased_l159_159957

variable (T P R : ℕ)

-- Conditions
def cost_per_can_A : ℕ := P / T
def cost_per_can_B : ℕ := 2 * cost_per_can_A T P
def quarters_in_dollar : ℕ := 4

-- Question and proof target
theorem cans_of_type_B_purchased (T P R : ℕ) (hT : T > 0) (hP : P > 0) (hR : R > 0) :
  (4 * R) / (2 * P / T) = 2 * R * T / P :=
by
  sorry

end cans_of_type_B_purchased_l159_159957


namespace quadratic_square_binomial_l159_159865

theorem quadratic_square_binomial (k : ℝ) : 
  (∃ a : ℝ, (x : ℝ) → x^2 - 20 * x + k = (x + a)^2) ↔ k = 100 := 
by
  sorry

end quadratic_square_binomial_l159_159865


namespace volume_increase_is_79_4_percent_l159_159969

noncomputable def original_volume (L B H : ℝ) : ℝ := L * B * H

noncomputable def new_volume (L B H : ℝ) : ℝ :=
  (L * 1.15) * (B * 1.30) * (H * 1.20)

noncomputable def volume_increase (L B H : ℝ) : ℝ :=
  new_volume L B H - original_volume L B H

theorem volume_increase_is_79_4_percent (L B H : ℝ) :
  volume_increase L B H = 0.794 * original_volume L B H := by
  sorry

end volume_increase_is_79_4_percent_l159_159969


namespace find_m_if_f_even_l159_159134

variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem find_m_if_f_even (h : ∀ x, f m x = f m (-x)) : m = -2 :=
by
  sorry

end find_m_if_f_even_l159_159134


namespace length_of_one_side_l159_159060

-- Definitions according to the conditions
def perimeter (nonagon : Type) : ℝ := 171
def sides (nonagon : Type) : ℕ := 9

-- Math proof problem to prove
theorem length_of_one_side (nonagon : Type) : perimeter nonagon / sides nonagon = 19 :=
by
  sorry

end length_of_one_side_l159_159060


namespace route_length_l159_159986

theorem route_length (D : ℝ) (T : ℝ) 
  (hx : T = 400 / D) 
  (hy : 80 = (D / 5) * T) 
  (hz : 80 + (D / 4) * T = D) : 
  D = 180 :=
by
  sorry

end route_length_l159_159986


namespace men_in_first_group_l159_159774

theorem men_in_first_group
  (M : ℕ) -- number of men in the first group
  (h1 : M * 8 * 24 = 12 * 8 * 16) : M = 8 :=
sorry

end men_in_first_group_l159_159774


namespace weight_of_new_person_l159_159347

def total_weight_increase (num_people : ℕ) (weight_increase_per_person : ℝ) : ℝ :=
  num_people * weight_increase_per_person

def new_person_weight (old_person_weight : ℝ) (total_weight_increase : ℝ) : ℝ :=
  old_person_weight + total_weight_increase

theorem weight_of_new_person :
  let old_person_weight := 50
  let num_people := 8
  let weight_increase_per_person := 2.5
  new_person_weight old_person_weight (total_weight_increase num_people weight_increase_per_person) = 70 := 
by
  sorry

end weight_of_new_person_l159_159347


namespace problem1_l159_159349

/-- Problem 1: Given the formula \( S = vt + \frac{1}{2}at^2 \) and the conditions
  when \( t=1, S=4 \) and \( t=2, S=10 \), prove that when \( t=3 \), \( S=18 \). -/
theorem problem1 (v a t S: ℝ) 
  (h₁ : t = 1 → S = 4 → S = v * t + 1 / 2 * a * t^2)
  (h₂ : t = 2 → S = 10 → S = v * t + 1 / 2 * a * t^2):
  t = 3 → S = v * t + 1 / 2 * a * t^2 → S = 18 := by
  sorry

end problem1_l159_159349


namespace find_c_l159_159913

theorem find_c (y c : ℝ) (h : y > 0) (h₂ : (8*y)/20 + (c*y)/10 = 0.7*y) : c = 6 :=
by
  sorry

end find_c_l159_159913


namespace tangent_line_y_intercept_at_P_1_12_is_9_l159_159489

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

noncomputable def tangent_slope_at (x : ℝ) : ℝ := 3 * x^2

noncomputable def tangent_line_y_intercept : ℝ :=
  let P : ℝ × ℝ := (1, curve 1)
  let slope := tangent_slope_at 1
  P.snd - slope * P.fst

theorem tangent_line_y_intercept_at_P_1_12_is_9 :
  tangent_line_y_intercept = 9 :=
sorry

end tangent_line_y_intercept_at_P_1_12_is_9_l159_159489


namespace repeating_decimal_eq_fraction_l159_159229

noncomputable def repeating_decimal_to_fraction (x : ℝ) : Prop :=
  let y := 20.396396396 -- represents 20.\overline{396}
  x = (20376 / 999)

theorem repeating_decimal_eq_fraction : 
  ∃ x : ℝ, repeating_decimal_to_fraction x :=
by
  use 20.396396396 -- represents 20.\overline{396}
  sorry

end repeating_decimal_eq_fraction_l159_159229


namespace vasya_birthday_was_thursday_l159_159807

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159807


namespace students_standing_together_l159_159979

theorem students_standing_together (s : Finset ℕ) (h_size : s.card = 6) (a b : ℕ) (h_ab : a ∈ s ∧ b ∈ s) (h_ab_together : ∃ (l : List ℕ), l.length = 6 ∧ a :: b :: l = l):
  ∃ (arrangements : ℕ), arrangements = 240 := by
  sorry

end students_standing_together_l159_159979


namespace scores_greater_than_18_l159_159928

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l159_159928


namespace find_x_from_percentage_l159_159666

theorem find_x_from_percentage : 
  ∃ x : ℚ, 0.65 * x = 0.20 * 487.50 := 
sorry

end find_x_from_percentage_l159_159666


namespace system_solution_l159_159635

noncomputable def x1 : ℝ := 55 / Real.sqrt 91
noncomputable def y1 : ℝ := 18 / Real.sqrt 91
noncomputable def x2 : ℝ := -55 / Real.sqrt 91
noncomputable def y2 : ℝ := -18 / Real.sqrt 91

theorem system_solution (x y : ℝ) (h1 : x^2 = 4 * y^2 + 19) (h2 : x * y + 2 * y^2 = 18) :
  (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) :=
sorry

end system_solution_l159_159635


namespace find_angle_C_max_area_l159_159422

-- Define the conditions as hypotheses
variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c = 2 * Real.sqrt 3)
variable (h2 : c * Real.cos B + (b - 2 * a) * Real.cos C = 0)

-- Problem (1): Prove that angle C is π/3
theorem find_angle_C : C = Real.pi / 3 :=
by
  sorry

-- Problem (2): Prove that the maximum area of triangle ABC is 3√3
theorem max_area : (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 :=
by
  sorry

end find_angle_C_max_area_l159_159422


namespace scores_greater_than_18_l159_159925

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l159_159925


namespace probability_two_red_faces_eq_three_eighths_l159_159094

def cube_probability : ℚ :=
  let total_cubes := 64 -- Total number of smaller cubes
  let two_red_faces_cubes := 24 -- Number of smaller cubes with exactly two red faces
  two_red_faces_cubes / total_cubes

theorem probability_two_red_faces_eq_three_eighths :
  cube_probability = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_two_red_faces_eq_three_eighths_l159_159094


namespace new_ellipse_standard_equation_l159_159234

theorem new_ellipse_standard_equation :
  let e1 := {x : ℝ × ℝ // (x.1^2 / 9 + x.2^2 / 4 = 1)},
      a := 5,
      c := sqrt 5,
      b := sqrt (a^2 - c^2)
  in
  let e2 := {x : ℝ × ℝ // (x.1^2 / a^2 + x.2^2 / b^2 = 1)}
  in
  e2 = {x : ℝ × ℝ // (x.1^2 / 25 + x.2^2 / 20 = 1)} :=
by
  sorry

end new_ellipse_standard_equation_l159_159234


namespace distinct_prime_factors_bound_l159_159470

open Int

theorem distinct_prime_factors_bound
  (a b : ℕ)
  (h_gcd : Nat.prime_factors (gcd a b).natAbs.card = 8)
  (h_lcm : Nat.prime_factors (natAbs ((a * b) / gcd a b)).card = 32)
  (h_less : Nat.prime_factors a.card < Nat.prime_factors b.card) :
  Nat.prime_factors a.card ≤ 19 :=
by
  sorry

end distinct_prime_factors_bound_l159_159470


namespace sin_pi_over_six_eq_half_l159_159492

theorem sin_pi_over_six_eq_half : Real.sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_pi_over_six_eq_half_l159_159492


namespace linear_function_graph_not_in_second_quadrant_l159_159935

open Real

theorem linear_function_graph_not_in_second_quadrant 
  (k b : ℝ) (h1 : k > 0) (h2 : b < 0) :
  ¬ ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b := 
sorry

end linear_function_graph_not_in_second_quadrant_l159_159935


namespace A_and_C_together_2_hours_l159_159525

theorem A_and_C_together_2_hours (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1 / 5)
  (hBC : B_rate + C_rate = 1 / 3) (hB : B_rate = 1 / 30) : A_rate + C_rate = 1 / 2 := 
by
  sorry

end A_and_C_together_2_hours_l159_159525


namespace cost_of_fixing_clothes_l159_159161

def num_shirts : ℕ := 10
def num_pants : ℕ := 12
def time_per_shirt : ℝ := 1.5
def time_per_pant : ℝ := 3.0
def rate_per_hour : ℝ := 30.0

theorem cost_of_fixing_clothes : 
  let total_time := (num_shirts * time_per_shirt) + (num_pants * time_per_pant)
  let total_cost := total_time * rate_per_hour
  total_cost = 1530 :=
by 
  sorry

end cost_of_fixing_clothes_l159_159161


namespace area_of_square_with_diagonal_40_l159_159835

theorem area_of_square_with_diagonal_40 {d : ℝ} (h : d = 40) : ∃ A : ℝ, A = 800 :=
by
  sorry

end area_of_square_with_diagonal_40_l159_159835


namespace pizza_slices_left_per_person_l159_159296

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

end pizza_slices_left_per_person_l159_159296


namespace olympiad_scores_l159_159917

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l159_159917


namespace scientific_notation_of_number_l159_159784

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000002 = a * 10^n ∧ a = 2 ∧ n = -8 :=
by
  sorry

end scientific_notation_of_number_l159_159784


namespace license_plate_count_l159_159415

-- Formalize the conditions
def is_letter (c : Char) : Prop := 'a' ≤ c ∧ c ≤ 'z'
def is_digit (c : Char) : Prop := '0' ≤ c ∧ c ≤ '9'

-- Define the main proof problem
theorem license_plate_count :
  (26 * (25 + 9) * 26 * 10 = 236600) :=
by sorry

end license_plate_count_l159_159415


namespace sum_of_xy_eq_20_l159_159585

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l159_159585


namespace perpendicular_vectors_l159_159144

/-- Given vectors a and b which are perpendicular, find the value of m -/
theorem perpendicular_vectors (m : ℝ) (a b : ℝ × ℝ)
  (h1 : a = (2 * m, 1))
  (h2 : b = (1, m - 3))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : m = 1 :=
by
  sorry

end perpendicular_vectors_l159_159144


namespace inscribed_sphere_volume_l159_159493

-- Definitions
def edge_length : ℝ := 4
def radius : ℝ := edge_length / 2

-- Statement
theorem inscribed_sphere_volume : (4 / 3) * Real.pi * radius^3 = (32 / 3) * Real.pi :=
by
  sorry

end inscribed_sphere_volume_l159_159493


namespace multiplicative_inverse_of_AB_l159_159763

def A : ℕ := 222222
def B : ℕ := 476190
def N : ℕ := 189
def modulus : ℕ := 1000000

theorem multiplicative_inverse_of_AB :
  (A * B * N) % modulus = 1 % modulus :=
by
  sorry

end multiplicative_inverse_of_AB_l159_159763


namespace magic_island_red_parrots_l159_159171

noncomputable def total_parrots : ℕ := 120

noncomputable def green_parrots : ℕ := (5 * total_parrots) / 8

noncomputable def non_green_parrots : ℕ := total_parrots - green_parrots

noncomputable def red_parrots : ℕ := non_green_parrots / 3

theorem magic_island_red_parrots : red_parrots = 15 :=
by
  sorry

end magic_island_red_parrots_l159_159171


namespace looms_employed_l159_159851

def sales_value := 500000
def manufacturing_expenses := 150000
def establishment_charges := 75000
def profit_decrease := 5000

def profit_per_loom (L : ℕ) : ℕ := (sales_value / L) - (manufacturing_expenses / L)

theorem looms_employed (L : ℕ) (h : profit_per_loom L = profit_decrease) : L = 70 :=
by
  have h_eq : profit_per_loom L = (sales_value - manufacturing_expenses) / L := by
    sorry
  have profit_expression : profit_per_loom L = profit_decrease := by
    sorry
  have L_value : L = (sales_value - manufacturing_expenses) / profit_decrease := by
    sorry
  have L_is_70 : L = 70 := by
    sorry
  exact L_is_70

end looms_employed_l159_159851


namespace intersection_P_Q_l159_159456

def P : Set ℝ := { x : ℝ | 2 ≤ x ∧ x < 4 }
def Q : Set ℝ := { x : ℝ | 3 ≤ x }

theorem intersection_P_Q :
  P ∩ Q = { x : ℝ | 3 ≤ x ∧ x < 4 } :=
by
  sorry  -- Proof step will be provided here

end intersection_P_Q_l159_159456


namespace carol_optimal_strategy_l159_159368

-- Definitions of the random variables
def uniform_A (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def uniform_B (b : ℝ) : Prop := 0.25 ≤ b ∧ b ≤ 0.75
def winning_condition (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Carol's optimal strategy stated as a theorem
theorem carol_optimal_strategy : ∀ (a b c : ℝ), 
  uniform_A a → uniform_B b → (c = 7 / 12) → 
  winning_condition a b c → 
  ∀ (c' : ℝ), uniform_A c' → c' ≠ c → ¬(winning_condition a b c') :=
by
  sorry

end carol_optimal_strategy_l159_159368


namespace binom_18_7_l159_159108

theorem binom_18_7 : Nat.choose 18 7 = 31824 := by sorry

end binom_18_7_l159_159108


namespace probability_two_red_two_blue_l159_159670

theorem probability_two_red_two_blue (total_red total_blue : ℕ) (red_taken blue_taken selected : ℕ)
  (h_red_total : total_red = 12) (h_blue_total : total_blue = 8) (h_selected : selected = 4)
  (h_red_taken : red_taken = 2) (h_blue_taken : blue_taken = 2) :
  (Nat.choose total_red red_taken) * (Nat.choose total_blue blue_taken) /
    (Nat.choose (total_red + total_blue) selected : ℚ) = 1848 / 4845 := 
by 
  sorry

end probability_two_red_two_blue_l159_159670


namespace sum_of_powers_l159_159264

theorem sum_of_powers (n : ℕ) (h : 8 ∣ n) : 
  (∑ k in Finset.range (n + 1), (k + 1) * (-Complex.i)^k) = 1.5 * n + 1 - 0.25 * n * Complex.i :=
by
  sorry

end sum_of_powers_l159_159264


namespace white_patches_count_l159_159090

-- Definitions based on the provided conditions
def total_patches : ℕ := 32
def white_borders_black (x : ℕ) : ℕ := 3 * x
def black_borders_white (x : ℕ) : ℕ := 5 * (total_patches - x)

-- The theorem we need to prove
theorem white_patches_count :
  ∃ x : ℕ, white_borders_black x = black_borders_white x ∧ x = 20 :=
by 
  sorry

end white_patches_count_l159_159090


namespace larger_number_is_17_l159_159069

noncomputable def x : ℤ := 17
noncomputable def y : ℤ := 12

def sum_condition : Prop := x + y = 29
def diff_condition : Prop := x - y = 5

theorem larger_number_is_17 (h_sum : sum_condition) (h_diff : diff_condition) : x = 17 :=
by {
  sorry
}

end larger_number_is_17_l159_159069


namespace units_digit_2_pow_10_l159_159991

theorem units_digit_2_pow_10 : (2 ^ 10) % 10 = 4 := 
sorry

end units_digit_2_pow_10_l159_159991


namespace marble_probability_correct_l159_159672

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l159_159672


namespace max_value_f_l159_159664

-- Definition of the function f(x)
def f (x ϕ : ℝ) : ℝ := sin (x + 2 * ϕ) - 2 * sin ϕ * cos (x + ϕ)

-- Theorem stating the maximum value of f(x) is 1
theorem max_value_f : ∀ (ϕ : ℝ), ∃ x : ℝ, f x ϕ = 1 := by
  sorry

end max_value_f_l159_159664


namespace number_of_multiples_of_7_but_not_14_l159_159257

-- Define the context and conditions
def positive_integers_less_than_500 : set ℕ := {n : ℕ | 0 < n ∧ n < 500 }
def multiples_of_7 : set ℕ := {n : ℕ | n % 7 = 0 }
def multiples_of_14 : set ℕ := {n : ℕ | n % 14 = 0 }
def multiples_of_7_but_not_14 : set ℕ := { n | n ∈ multiples_of_7 ∧ n ∉ multiples_of_14 }

-- Define the theorem to prove
theorem number_of_multiples_of_7_but_not_14 : 
  ∃! n : ℕ, n = 36 ∧ n = finset.card (finset.filter (λ x, x ∈ multiples_of_7_but_not_14) (finset.range 500)) :=
begin
  sorry
end

end number_of_multiples_of_7_but_not_14_l159_159257


namespace problem_theorem_l159_159605

noncomputable def problem_statement : Prop :=
  ∀ (A B C : ℝ) (a b c S : ℝ) (u v : EuclideanSpace ℝ (Fin 3)),
    let AB := u - v in
    let AC := u - v in
    (a > 0) ∧ (b > 0) ∧ (c > 0) 
    ∧ (c = 7) 
    ∧ (cos B = 4/5)
    ∧ (2 * S = (AB ∙ AC))
    ∧ (a = Sqrt(AB ∙ AB))
    ∧ (b = Sqrt(AC ∙ AC))
    → (A = π / 4) ∧ (a = 5)

theorem problem_theorem : problem_statement :=
by
  intros A B C a b c S u v
  let AB := u - v
  let AC := u - v
  assume base_conditions : (a > 0) ∧ (b > 0) ∧ (c = 7) ∧ (cos B = 4/5)
  assume area_condition : (2 * S = (AB ∙ AC))
  assume sides_condition : (a = Sqrt(AB ∙ AB)) ∧ (b = Sqrt(AC ∙ AC))
  sorry

end problem_theorem_l159_159605


namespace max_ab_value_l159_159241

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 4) : ab ≤ 2 :=
sorry

end max_ab_value_l159_159241


namespace min_value_of_quadratic_l159_159813

theorem min_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y^2 - 6 * y + 5 ≥ (x - 3)^2 - 4) ∧ (y^2 - 6 * y + 5 = -4) :=
by sorry

end min_value_of_quadratic_l159_159813


namespace sufficient_not_necessary_l159_159452

theorem sufficient_not_necessary (x : ℝ) :
  (|x - 1| < 2 → x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end sufficient_not_necessary_l159_159452


namespace solution_set_of_inequality_l159_159974

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 2*x + 3 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l159_159974


namespace john_spent_amount_l159_159442

-- Definitions based on the conditions in the problem.
def hours_played : ℕ := 3
def cost_per_6_minutes : ℚ := 0.50
def minutes_per_6_minutes_interval : ℕ := 6
def total_minutes_played (hours : ℕ) : ℕ := hours * 60

-- The theorem statement.
theorem john_spent_amount 
  (h : hours_played = 3) 
  (c : cost_per_6_minutes = 0.50)
  (m_interval : minutes_per_6_minutes_interval = 6) :
  let intervals := (total_minutes_played hours_played) / minutes_per_6_minutes_interval in
  intervals * cost_per_6_minutes = 15 := 
by
  sorry

end john_spent_amount_l159_159442


namespace tulip_gift_count_l159_159461

theorem tulip_gift_count :
  let max_tulips := 11 in
  let odd_tulip_combinations := ∑ k in (Finset.range (max_tulips + 1)), if k % 2 = 1 then Nat.choose max_tulips k else 0 in
  odd_tulip_combinations = 1024 :=
by
  sorry

end tulip_gift_count_l159_159461


namespace time_to_cross_pole_l159_159211

-- Setting up the definitions
def speed_kmh : ℤ := 72
def length_m : ℤ := 180

-- Conversion function from km/hr to m/s
def convert_speed (v : ℤ) : ℚ :=
  v * (1000 : ℚ) / 3600

-- Given conditions in mathematics
def speed_ms : ℚ := convert_speed speed_kmh

-- Desired proposition
theorem time_to_cross_pole : 
  length_m / speed_ms = 9 := 
by
  -- Temporarily skipping the proof
  sorry

end time_to_cross_pole_l159_159211


namespace set_of_possible_values_l159_159035

-- Define the variables and the conditions as a Lean definition
noncomputable def problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : Set ℝ :=
  {x : ℝ | x = (1 / a + 1 / b + 1 / c)}

-- Define the theorem to state that the set of all possible values is [9, ∞)
theorem set_of_possible_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  problem a b c ha hb hc sum_eq_one = {x : ℝ | 9 ≤ x} :=
sorry

end set_of_possible_values_l159_159035


namespace compute_value_l159_159694

theorem compute_value : (142 + 29 + 26 + 14) * 2 = 422 := 
by 
  sorry

end compute_value_l159_159694


namespace vectors_parallel_l159_159899

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end vectors_parallel_l159_159899


namespace march_first_is_tuesday_l159_159595

theorem march_first_is_tuesday (march_15_tuesday : true) :
  true :=
sorry

end march_first_is_tuesday_l159_159595


namespace min_lit_bulbs_l159_159879

theorem min_lit_bulbs (n : ℕ) (h : n ≥ 1) : 
  ∃ rows cols, (rows ⊆ Finset.range n) ∧ (cols ⊆ Finset.range n) ∧ 
  (∀ i j, (i ∈ rows ∧ j ∈ cols) ↔ (i + j) % 2 = 1) ∧ 
  rows.card * (n - cols.card) + cols.card * (n - rows.card) = 2 * n - 2 :=
by sorry

end min_lit_bulbs_l159_159879


namespace infinite_sum_converges_to_3_l159_159383

theorem infinite_sum_converges_to_3 :
  (∑' k : ℕ, (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 3 :=
by
  sorry

end infinite_sum_converges_to_3_l159_159383


namespace tangent_line_at_one_min_value_f_l159_159447

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * |Real.log x - 1|

theorem tangent_line_at_one (a : ℝ) (h1 : a = 1) : 
  ∃ (m b : ℝ), (∀ x : ℝ, f x a = m * x + b) ∧ m = 1 ∧ b = 1 ∧ (x - y + 1 = 0) := 
sorry

theorem min_value_f (a : ℝ) (h1 : 0 < a) : 
  (1 ≤ x ∧ x < e)  →  (x - f x a <= 0) ∨  (∀ (x : ℝ), 
  (f x a = if 0 < a ∧ a ≤ 2 then 1 + a 
          else if 2 < a ∧ a ≤ 2 * Real.exp (2) then 3 * (a / 2)^2 - (a / 2)^2 * Real.log (a / 2) else 
          Real.exp 2) 
   ) := 
sorry

end tangent_line_at_one_min_value_f_l159_159447


namespace train_time_to_pass_bridge_l159_159852

theorem train_time_to_pass_bridge
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ)
  (h1 : length_train = 500) (h2 : length_bridge = 200) (h3 : speed_kmph = 72) :
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_mps
  time = 35 :=
by
  sorry

end train_time_to_pass_bridge_l159_159852


namespace evaluate_expression_l159_159387

theorem evaluate_expression : 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 :=
by
  sorry

end evaluate_expression_l159_159387


namespace total_bills_inserted_l159_159357

theorem total_bills_inserted (x y : ℕ) (h1 : x = 175) (h2 : x + 5 * y = 300) : 
  x + y = 200 :=
by {
  -- Since we focus strictly on the statement per instruction, the proof is omitted
  sorry 
}

end total_bills_inserted_l159_159357


namespace apple_production_l159_159543

variable {S1 S2 S3 : ℝ}

theorem apple_production (h1 : S2 = 0.8 * S1) 
                         (h2 : S3 = 2 * S2) 
                         (h3 : S1 + S2 + S3 = 680) : 
                         S1 = 200 := 
by
  sorry

end apple_production_l159_159543


namespace difference_of_squares_l159_159445

theorem difference_of_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x y : ℤ, a = x^2 - y^2) ∨ 
  (∃ x y : ℤ, b = x^2 - y^2) ∨ 
  (∃ x y : ℤ, a + b = x^2 - y^2) :=
by
  sorry

end difference_of_squares_l159_159445


namespace cheese_left_after_10_customers_l159_159323

theorem cheese_left_after_10_customers :
  ∀ (S : ℕ → ℚ), (∀ n, S n = (20 * n) / (n + 10)) →
  20 - S 10 = 10 := by
  sorry

end cheese_left_after_10_customers_l159_159323


namespace solve_for_y_l159_159732

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159732


namespace shaded_area_size_l159_159157

noncomputable def total_shaded_area : ℝ :=
  let R := 9
  let r := R / 2
  let area_larger_circle := 81 * Real.pi
  let shaded_area_larger_circle := area_larger_circle / 2
  let area_smaller_circle := Real.pi * r^2
  let shaded_area_smaller_circle := area_smaller_circle / 2
  let total_shaded_area := shaded_area_larger_circle + shaded_area_smaller_circle
  total_shaded_area

theorem shaded_area_size:
  total_shaded_area = 50.625 * Real.pi := 
by
  sorry

end shaded_area_size_l159_159157


namespace coat_price_reduction_l159_159272

theorem coat_price_reduction :
  let orig_price := 500
  let first_discount := 0.15 * orig_price
  let price_after_first := orig_price - first_discount
  let second_discount := 0.10 * price_after_first
  let price_after_second := price_after_first - second_discount
  let tax := 0.07 * price_after_second
  let price_with_tax := price_after_second + tax
  let final_price := price_with_tax - 200
  let reduction_amount := orig_price - final_price
  let percent_reduction := (reduction_amount / orig_price) * 100
  percent_reduction = 58.145 :=
by
  sorry

end coat_price_reduction_l159_159272


namespace range_of_m_for_nonnegative_quadratic_l159_159153

-- The statement of the proof problem in Lean
theorem range_of_m_for_nonnegative_quadratic {x m : ℝ} : 
  (∀ x, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_nonnegative_quadratic_l159_159153


namespace gcd_values_360_l159_159825

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159825


namespace remainder_polynomial_division_l159_159334

theorem remainder_polynomial_division :
  ∀ (x : ℝ), (2 * x^2 - 21 * x + 55) % (x + 3) = 136 := 
sorry

end remainder_polynomial_division_l159_159334


namespace completing_square_solution_l159_159818

theorem completing_square_solution (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
sorry

end completing_square_solution_l159_159818


namespace shortest_chord_through_point_l159_159201

theorem shortest_chord_through_point
  (correct_length : ℝ)
  (h1 : correct_length = 2 * Real.sqrt 2)
  (circle_eq : ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 4)
  (passes_point : ∀ (p : ℝ × ℝ), p = (3, 1)) :
  correct_length = 2 * Real.sqrt 2 :=
by {
  -- the proof steps would go here
  sorry
}

end shortest_chord_through_point_l159_159201


namespace total_fireworks_correct_l159_159364

variable (fireworks_num fireworks_reg)
variable (fireworks_H fireworks_E fireworks_L fireworks_O)
variable (fireworks_square fireworks_triangle fireworks_circle)
variable (boxes fireworks_per_box : ℕ)

-- Given Conditions
def fireworks_years_2021_2023 : ℕ := 6 * 4 * 3
def fireworks_HAPPY_NEW_YEAR : ℕ := 5 * 11 + 6
def fireworks_geometric_shapes : ℕ := 4 + 3 + 12
def fireworks_HELLO : ℕ := 8 + 7 + 6 * 2 + 9
def fireworks_additional_boxes : ℕ := 100 * 10

-- Total Fireworks
def total_fireworks : ℕ :=
  fireworks_years_2021_2023 + 
  fireworks_HAPPY_NEW_YEAR + 
  fireworks_geometric_shapes + 
  fireworks_HELLO + 
  fireworks_additional_boxes

theorem total_fireworks_correct : 
  total_fireworks = 1188 :=
  by
  -- The proof is omitted.
  sorry

end total_fireworks_correct_l159_159364


namespace vasya_birthday_l159_159797

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159797


namespace substitution_correct_l159_159124

theorem substitution_correct (x y : ℝ) (h1 : y = x - 1) (h2 : x - 2 * y = 7) :
  x - 2 * x + 2 = 7 :=
by
  sorry

end substitution_correct_l159_159124


namespace Flora_initial_daily_milk_l159_159005

def total_gallons : ℕ := 105
def total_weeks : ℕ := 3
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def extra_gallons_daily : ℕ := 2

theorem Flora_initial_daily_milk : 
  (total_gallons / total_days) = 5 := by
  sorry

end Flora_initial_daily_milk_l159_159005


namespace olympiad_scores_l159_159916

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l159_159916


namespace minimum_milk_candies_l159_159540

/-- A supermarket needs to purchase candies with the following conditions:
 1. The number of watermelon candies is at most 3 times the number of chocolate candies.
 2. The number of milk candies is at least 4 times the number of chocolate candies.
 3. The sum of chocolate candies and watermelon candies is at least 2020.

 Prove that the minimum number of milk candies that need to be purchased is 2020. -/
theorem minimum_milk_candies (x y z : ℕ)
  (h1 : y ≤ 3 * x)
  (h2 : z ≥ 4 * x)
  (h3 : x + y ≥ 2020) :
  z ≥ 2020 :=
sorry

end minimum_milk_candies_l159_159540


namespace range_of_a_l159_159250

open Set

variable {a : ℝ}
def M (a : ℝ) : Set ℝ := { x : ℝ | (2 * a - 1) < x ∧ x < (4 * a) }
def N : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem range_of_a (h : N ⊆ M a) : 1 / 2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l159_159250


namespace alice_total_spending_l159_159095

theorem alice_total_spending :
  let book_price_gbp := 15
  let souvenir_price_eur := 20
  let gbp_to_usd_rate := 1.25
  let eur_to_usd_rate := 1.10
  let book_price_usd := book_price_gbp * gbp_to_usd_rate
  let souvenir_price_usd := souvenir_price_eur * eur_to_usd_rate
  let total_usd := book_price_usd + souvenir_price_usd
  total_usd = 40.75 :=
by
  sorry

end alice_total_spending_l159_159095


namespace expressions_divisible_by_17_l159_159466

theorem expressions_divisible_by_17 (a b : ℤ) : 
  let x := 3 * b - 5 * a
  let y := 9 * a - 2 * b
  (∃ k : ℤ, (2 * x + 3 * y) = 17 * k) ∧ (∃ k : ℤ, (9 * x + 5 * y) = 17 * k) :=
by
  exact ⟨⟨a, by sorry⟩, ⟨b, by sorry⟩⟩

end expressions_divisible_by_17_l159_159466


namespace multiply_polynomials_l159_159765

open Polynomial

variable {R : Type*} [CommRing R]

theorem multiply_polynomials (x : R) :
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 :=
  sorry

end multiply_polynomials_l159_159765


namespace equal_clubs_and_students_l159_159425

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l159_159425


namespace monthly_interest_payment_l159_159371

theorem monthly_interest_payment (principal : ℝ) (annual_rate : ℝ) (months_in_year : ℝ) : 
  principal = 31200 → 
  annual_rate = 0.09 → 
  months_in_year = 12 → 
  (principal * annual_rate) / months_in_year = 234 := 
by 
  intros h_principal h_rate h_months
  rw [h_principal, h_rate, h_months]
  sorry

end monthly_interest_payment_l159_159371


namespace binomial_ratio_l159_159133

theorem binomial_ratio (n : ℕ) (r : ℕ) :
  (Nat.choose n r : ℚ) / (Nat.choose n (r+1) : ℚ) = 1 / 2 →
  (Nat.choose n (r+1) : ℚ) / (Nat.choose n (r+2) : ℚ) = 2 / 3 →
  n = 14 :=
by
  sorry

end binomial_ratio_l159_159133


namespace geometric_sequence_problem_l159_159277

theorem geometric_sequence_problem (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 2 * a 5 = -32)
  (h2 : a 3 + a 4 = 4)
  (hq : ∃ (k : ℤ), q = k) :
  a 9 = -256 := 
sorry

end geometric_sequence_problem_l159_159277


namespace range_of_a_l159_159245

theorem range_of_a (a : ℝ) (hp : a^2 - 2 * a - 2 > 1) (hnq : a <= 0 ∨ a >= 4) : a >= 4 ∨ a < -1 :=
sorry

end range_of_a_l159_159245


namespace proof_a_in_S_l159_159624

def S : Set ℤ := {n : ℤ | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem proof_a_in_S (a : ℤ) (h1 : 3 * a ∈ S) : a ∈ S :=
sorry

end proof_a_in_S_l159_159624


namespace cost_of_bananas_and_cantaloupe_l159_159626

-- Define variables representing the prices
variables (a b c d : ℝ)

-- Define the given conditions as hypotheses
def conditions : Prop :=
  a + b + c + d = 33 ∧
  d = 3 * a ∧
  c = a + 2 * b

-- State the main theorem
theorem cost_of_bananas_and_cantaloupe (h : conditions a b c d) : b + c = 13 :=
by {
  sorry
}

end cost_of_bananas_and_cantaloupe_l159_159626


namespace find_c_find_A_l159_159275

open Real

noncomputable def acute_triangle_sides (A B C a b c : ℝ) : Prop :=
  a = b * cos C + (sqrt 3 / 3) * c * sin B

theorem find_c (A B C a b c : ℝ) (ha : a = 2) (hb : b = sqrt 7) 
  (hab : acute_triangle_sides A B C a b c) : c = 3 := 
sorry

theorem find_A (A B C : ℝ) (h : sqrt 3 * sin (2 * A - π / 6) - 2 * (sin (C - π / 12))^2 = 0)
  (h_range : π / 6 < A ∧ A < π / 2) : A = π / 4 :=
sorry

end find_c_find_A_l159_159275


namespace smallest_integer_with_eight_factors_l159_159506

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l159_159506


namespace average_speed_second_half_l159_159106

theorem average_speed_second_half
  (d : ℕ) (s1 : ℕ) (t : ℕ)
  (h1 : d = 3600)
  (h2 : s1 = 90)
  (h3 : t = 30) :
  (d / 2) / (t - (d / 2 / s1)) = 180 := by
  sorry

end average_speed_second_half_l159_159106


namespace braden_money_box_total_l159_159099

def initial_money : ℕ := 400

def correct_predictions : ℕ := 3

def betting_rules (correct_predictions : ℕ) : ℕ :=
  match correct_predictions with
  | 1 => 25
  | 2 => 50
  | 3 => 75
  | 4 => 200
  | _ => 0

theorem braden_money_box_total:
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  initial_money + winnings = 700 := 
by
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  show initial_money + winnings = 700
  sorry

end braden_money_box_total_l159_159099


namespace proposition_a_is_true_l159_159074

-- Define a quadrilateral
structure Quadrilateral (α : Type*) [Ring α] :=
(a b c d : α)

-- Define properties of a Quadrilateral
def parallel_and_equal_opposite_sides (Q : Quadrilateral ℝ) : Prop := sorry  -- Assumes parallel and equal opposite sides
def is_parallelogram (Q : Quadrilateral ℝ) : Prop := sorry  -- Defines a parallelogram

-- The theorem we need to prove
theorem proposition_a_is_true (Q : Quadrilateral ℝ) (h : parallel_and_equal_opposite_sides Q) : is_parallelogram Q :=
sorry

end proposition_a_is_true_l159_159074


namespace find_smallest_c_l159_159940

/-- Let a₀, a₁, ... and b₀, b₁, ... be geometric sequences with common ratios rₐ and r_b, 
respectively, such that ∑ i=0 ∞ aᵢ = ∑ i=0 ∞ bᵢ = 1 and 
(∑ i=0 ∞ aᵢ²)(∑ i=0 ∞ bᵢ²) = ∑ i=0 ∞ aᵢbᵢ. Prove that a₀ < 4/3 -/
theorem find_smallest_c (r_a r_b : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∑' n, a n = 1)
  (h2 : ∑' n, b n = 1)
  (h3 : (∑' n, (a n)^2) * (∑' n, (b n)^2) = ∑' n, (a n) * (b n)) :
  a 0 < 4 / 3 := by
  sorry

end find_smallest_c_l159_159940


namespace quadratic_roots_expression_l159_159570

theorem quadratic_roots_expression :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 1 = 0) ∧ (x2^2 - 2 * x2 - 1 = 0) →
  (x1 + x2 - x1 * x2 = 3) :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_expression_l159_159570


namespace exact_one_solves_l159_159351

variables (p1 p2 : ℝ)

/-- The probability that exactly one of two persons solves the problem
    when their respective probabilities are p1 and p2. -/
theorem exact_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 + p2 - 2 * p1 * p2) := 
sorry

end exact_one_solves_l159_159351


namespace value_of_a_l159_159265

theorem value_of_a (a : ℝ) (h : (1 : ℝ)^2 - 2 * (1 : ℝ) + a = 0) : a = 1 := 
by 
  sorry

end value_of_a_l159_159265


namespace scores_greater_than_18_l159_159926

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l159_159926


namespace relationship_l159_159129

noncomputable def a : ℝ := Real.log (Real.log Real.pi)
noncomputable def b : ℝ := Real.log Real.pi
noncomputable def c : ℝ := 2^Real.log Real.pi

theorem relationship (a b c : ℝ) (ha : a = Real.log (Real.log Real.pi)) (hb : b = Real.log Real.pi) (hc : c = 2^Real.log Real.pi)
: a < b ∧ b < c := 
by
  sorry

end relationship_l159_159129


namespace least_common_multiple_xyz_l159_159642

theorem least_common_multiple_xyz (x y z : ℕ) 
  (h1 : Nat.lcm x y = 18) 
  (h2 : Nat.lcm y z = 20) : 
  Nat.lcm x z = 90 := 
sorry

end least_common_multiple_xyz_l159_159642


namespace log5_x_l159_159733

theorem log5_x (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ (Real.log 16 / Real.log 2) ^ 2) :
    Real.log x / Real.log 5 = -16 / (Real.log 2 / Real.log 5) := by
  sorry

end log5_x_l159_159733


namespace circle_problem_l159_159499

theorem circle_problem (P : ℝ × ℝ) (QR : ℝ) (S : ℝ × ℝ) (k : ℝ)
  (h1 : P = (5, 12))
  (h2 : QR = 5)
  (h3 : S = (0, k))
  (h4 : dist (0,0) P = 13) -- OP = 13 from the origin to point P
  (h5 : dist (0,0) S = 8) -- OQ = 8 from the origin to point S
: k = 8 ∨ k = -8 :=
by sorry

end circle_problem_l159_159499


namespace least_whole_number_for_ratio_l159_159987

theorem least_whole_number_for_ratio :
  ∃ x : ℕ, (6 - x) * 21 < (7 - x) * 16 ∧ x = 3 :=
by
  sorry

end least_whole_number_for_ratio_l159_159987


namespace fraction_zero_solution_l159_159304

theorem fraction_zero_solution (x : ℝ) (h1 : x - 5 = 0) (h2 : 4 * x^2 - 1 ≠ 0) : x = 5 :=
by {
  sorry -- The proof
}

end fraction_zero_solution_l159_159304


namespace value_of_y_plus_10_l159_159904

theorem value_of_y_plus_10 (x y : ℝ) (h1 : 3 * x = (3 / 4) * y) (h2 : x = 20) : y + 10 = 90 :=
by
  sorry

end value_of_y_plus_10_l159_159904


namespace relationship_not_true_l159_159132

theorem relationship_not_true (a b : ℕ) :
  (b = a + 5 ∨ b = a + 15 ∨ b = a + 29) → ¬(a = b - 9) :=
by
  sorry

end relationship_not_true_l159_159132


namespace math_olympiad_proof_l159_159930

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l159_159930


namespace martin_total_distance_l159_159946

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l159_159946


namespace number_of_girl_students_l159_159324

theorem number_of_girl_students (total_third_graders : ℕ) (boy_students : ℕ) (girl_students : ℕ) 
  (h1 : total_third_graders = 123) (h2 : boy_students = 66) (h3 : total_third_graders = boy_students + girl_students) :
  girl_students = 57 :=
by
  sorry

end number_of_girl_students_l159_159324


namespace total_people_present_l159_159185

/-- This definition encapsulates all the given conditions: 
    The number of parents, pupils, staff members, and performers. -/
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_staff : ℕ := 45
def num_performers : ℕ := 32

/-- Theorem stating that the total number of people present in the program is 880 
    given the stated conditions. -/
theorem total_people_present : num_parents + num_pupils + num_staff + num_performers = 880 :=
by 
  /- We can use Lean's capabilities to verify the arithmetics. -/
  sorry

end total_people_present_l159_159185


namespace journey_time_l159_159996

theorem journey_time
  (speed1 speed2 : ℝ)
  (distance total_time : ℝ)
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance = 240)
  (h4 : total_time = 5) :
  ∃ (t1 t2 : ℝ), (t1 + t2 = total_time) ∧ (speed1 * t1 + speed2 * t2 = distance) ∧ (t1 = 3) := 
by
  use (3 : ℝ), (2 : ℝ)
  simp [h1, h2, h3, h4]
  norm_num
  -- Additional steps to finish the proof would go here, but are omitted as per the requirements
  -- sorry

end journey_time_l159_159996


namespace smallest_constant_l159_159703

theorem smallest_constant (D : ℝ) :
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) → D ≤ Real.sqrt (8 / 17) :=
by
  intros
  sorry

end smallest_constant_l159_159703


namespace det_B_squared_sub_3B_eq_10_l159_159022

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 3], ![2, 2]]

theorem det_B_squared_sub_3B_eq_10 : 
  Matrix.det (B * B - 3 • B) = 10 := by
  sorry

end det_B_squared_sub_3B_eq_10_l159_159022


namespace integer_points_on_segment_l159_159809

open Int

def is_integer_point (x y : ℝ) : Prop := ∃ (a b : ℤ), x = a ∧ y = b

def f (n : ℕ) : ℕ := 
  if 3 ∣ n then 2
  else 0

theorem integer_points_on_segment (n : ℕ) (hn : 0 < n) :
  (f n) = if 3 ∣ n then 2 else 0 := 
  sorry

end integer_points_on_segment_l159_159809


namespace max_value_bx_plus_a_l159_159877

variable (a b : ℝ)

theorem max_value_bx_plus_a (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ |b * x + a| = 2 :=
by
  -- Proof goes here
  sorry

end max_value_bx_plus_a_l159_159877


namespace gaussian_guardians_points_l159_159977

theorem gaussian_guardians_points :
  let Daniel := 7
  let Curtis := 8
  let Sid := 2
  let Emily := 11
  let Kalyn := 6
  let Hyojeong := 12
  let Ty := 1
  let Winston := 7
  Daniel + Curtis + Sid + Emily + Kalyn + Hyojeong + Ty + Winston = 54 :=
by
  sorry

end gaussian_guardians_points_l159_159977


namespace expansion_coefficients_sum_l159_159583

theorem expansion_coefficients_sum : 
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), 
    (x - 2)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 → 
    a_0 + a_2 + a_4 = -122 := 
by 
  intros x a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  sorry

end expansion_coefficients_sum_l159_159583


namespace int_less_than_sqrt_23_l159_159955

theorem int_less_than_sqrt_23 : ∃ (n : ℤ), n < Real.sqrt 23 := by
  use 4
  have h : (4 : ℝ) < Real.sqrt 23 := by
    rw Real.sqrt_lt'_iff
    exact ⟨dec_trivial, dec_trivial⟩
  exact_mod_cast h

end int_less_than_sqrt_23_l159_159955


namespace positive_integer_pair_solution_l159_159559

theorem positive_integer_pair_solution :
  ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ 
    ¬ (7 ∣ (a * b * (a + b))) ∧ 
    (7^7 ∣ ((a + b)^7 - a^7 - b^7)) ∧ 
    (a, b) = (18, 1) :=
by {
  sorry
}

end positive_integer_pair_solution_l159_159559


namespace simplify_fraction_l159_159698

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 :=
by
  sorry

end simplify_fraction_l159_159698


namespace fraction_of_upgraded_sensors_l159_159535

theorem fraction_of_upgraded_sensors (N U : ℕ) (h1 : N = U / 6) :
  (U / (24 * N + U)) = 1 / 5 :=
by
  sorry

end fraction_of_upgraded_sensors_l159_159535


namespace sales_tax_difference_l159_159089

theorem sales_tax_difference (P : ℝ) (d t1 t2 : ℝ) :
  let discounted_price := P * (1 - d)
  let total_cost1 := discounted_price * (1 + t1)
  let total_cost2 := discounted_price * (1 + t2)
  t1 = 0.08 ∧ t2 = 0.075 ∧ P = 50 ∧ d = 0.05 →
  abs ((total_cost1 - total_cost2) - 0.24) < 0.01 :=
by
  sorry

end sales_tax_difference_l159_159089


namespace prime_numbers_satisfying_condition_l159_159388

theorem prime_numbers_satisfying_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℕ, 1 + p * 2^p = x^2) ↔ p = 2 ∨ p = 3 :=
by
  sorry

end prime_numbers_satisfying_condition_l159_159388


namespace henry_geography_math_score_l159_159414

variable (G M : ℕ)

theorem henry_geography_math_score (E : ℕ) (H : ℕ) (total_score : ℕ) 
  (hE : E = 66) 
  (hH : H = (G + M + E) / 3)
  (hTotal : G + M + E + H = total_score) 
  (htotal_score : total_score = 248) :
  G + M = 120 := 
by
  sorry

end henry_geography_math_score_l159_159414


namespace impossible_to_place_numbers_l159_159610

noncomputable def divisible (a b : ℕ) : Prop := ∃ k : ℕ, a * k = b

def connected (G : Finset (ℕ × ℕ)) (u v : ℕ) : Prop := (u, v) ∈ G ∨ (v, u) ∈ G

def valid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, connected G i j → divisible (f i) (f j) ∨ divisible (f j) (f i)

def invalid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, ¬ connected G i j → ¬ divisible (f i) (f j) ∧ ¬ divisible (f j) (f i)

theorem impossible_to_place_numbers (G : Finset (ℕ × ℕ)) :
  (∃ f : ℕ → ℕ, valid_assignment G f ∧ invalid_assignment G f) → False :=
by
  sorry

end impossible_to_place_numbers_l159_159610


namespace gcd_lcm_252_l159_159073

theorem gcd_lcm_252 {a b : ℕ} (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃ S : Finset ℕ, S.card = 8 ∧ ∀ d ∈ S, d = Nat.gcd a b :=
by sorry

end gcd_lcm_252_l159_159073


namespace value_of_n_in_arithmetic_sequence_l159_159751

theorem value_of_n_in_arithmetic_sequence :
  (∃ (a d : ℝ) (n : ℕ), a = 1/3 ∧ (a + d) + (a + 4 * d) = 4 ∧ a + (n - 1) * d = 33) →
  n = 50 := sorry

end value_of_n_in_arithmetic_sequence_l159_159751


namespace exists_int_less_than_sqrt_twenty_three_l159_159954

theorem exists_int_less_than_sqrt_twenty_three : ∃ n : ℤ, n < Real.sqrt 23 := 
  sorry

end exists_int_less_than_sqrt_twenty_three_l159_159954


namespace troll_ratio_l159_159867

theorem troll_ratio 
  (B : ℕ)
  (h1 : 6 + B + (1 / 2 : ℚ) * B = 33) : 
  B / 6 = 3 :=
by
  sorry

end troll_ratio_l159_159867


namespace relationship_log2_2_pow_03_l159_159313

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem relationship_log2_2_pow_03 : 
  log_base_2 0.3 < (0.3)^2 ∧ (0.3)^2 < 2^(0.3) :=
by
  sorry

end relationship_log2_2_pow_03_l159_159313


namespace bridge_length_l159_159997

theorem bridge_length 
  (train_length : ℕ) 
  (speed_km_hr : ℕ) 
  (cross_time_sec : ℕ) 
  (conversion_factor_num : ℕ) 
  (conversion_factor_den : ℕ)
  (expected_length : ℕ) 
  (speed_m_s : ℕ := speed_km_hr * conversion_factor_num / conversion_factor_den)
  (total_distance : ℕ := speed_m_s * cross_time_sec) :
  train_length = 150 →
  speed_km_hr = 45 →
  cross_time_sec = 30 →
  conversion_factor_num = 1000 →
  conversion_factor_den = 3600 →
  expected_length = 225 →
  total_distance - train_length = expected_length :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end bridge_length_l159_159997


namespace white_given_popped_l159_159353

-- Define the conditions
def white_kernels : ℚ := 1 / 2
def yellow_kernels : ℚ := 1 / 3
def blue_kernels : ℚ := 1 / 6

def white_kernels_pop : ℚ := 3 / 4
def yellow_kernels_pop : ℚ := 1 / 2
def blue_kernels_pop : ℚ := 1 / 3

def probability_white_popped : ℚ := white_kernels * white_kernels_pop
def probability_yellow_popped : ℚ := yellow_kernels * yellow_kernels_pop
def probability_blue_popped : ℚ := blue_kernels * blue_kernels_pop

def probability_popped : ℚ := probability_white_popped + probability_yellow_popped + probability_blue_popped

-- The theorem to be proved
theorem white_given_popped : (probability_white_popped / probability_popped) = (27 / 43) := 
by sorry

end white_given_popped_l159_159353


namespace ratio_p_r_l159_159759

     variables (p q r s : ℚ)

     -- Given conditions
     def ratio_p_q := p / q = 3 / 5
     def ratio_r_s := r / s = 5 / 4
     def ratio_s_q := s / q = 1 / 3

     -- Statement to be proved
     theorem ratio_p_r 
       (h1 : ratio_p_q p q)
       (h2 : ratio_r_s r s) 
       (h3 : ratio_s_q s q) : 
       p / r = 36 / 25 :=
     sorry
     
end ratio_p_r_l159_159759


namespace math_olympiad_proof_l159_159934

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l159_159934


namespace solve_for_x_l159_159176

theorem solve_for_x (x : ℝ) (h : 24 - 6 = 3 * x + 3) : x = 5 := by
  sorry

end solve_for_x_l159_159176


namespace points_B_O_I_H_C_cocylic_AH_AO_R_OI_IH_PO_HQ_OH_lengths_incenter_relation_l159_159420

-- Given conditions
variables {A B C : Point}
variables (O I H P Q F : Point) -- points
variables (R : ℝ) -- radii
variables (angle_A_eq_60 : ∠ A = 60)
variables (circumcenter_O : is_circumcenter O A B C)
variables (incenter_I : is_incenter I A B C)
variables (orthocenter_H : is_orthocenter H A B C)
variables (line_OH_intersects_AB_at_P : lies_on (O,H) P)
variables (line_OH_intersects_AC_at_Q : lies_on (O,H) Q)
variables (line_BI_intersects_AC_at_F : lies_on (B,I) F)

-- (1) Prove that points B, O, I, H, C are concyclic
theorem points_B_O_I_H_C_cocylic 
  (circumcenter_cond : is_circumcenter O A B C)
  (incenter_cond : is_incenter I A B C)
  (orthocenter_cond : is_orthocenter H A B C) 
  (angle_A_60 : ∠ A = 60) :
  cocyclic {B, O, I, H, C} := sorry

-- (2) Prove AH = AO = R, OI = IH, and PO = HQ
theorem AH_AO_R_OI_IH_PO_HQ 
  (AH_eq_AO : AH = AO)
  (AO_eq_R : AO = R)
  (OI_eq_IH : OI = IH)
  (PO_eq_HQ : PO = HQ) :
  AH = AO ∧ AO = R ∧ OI = IH ∧ PO = HQ := sorry

-- (3) Prove OH = |AB - AC| and OH = |HB - HC| / sqrt(3)
theorem OH_lengths 
  (AB_neq_AC : AB ≠ AC) :
  OH = abs (AB - AC) ∧ OH = abs (HB - HC) / √(3) := sorry

-- (4) Prove (1/IB) + (1/IC) = (1/IF)
theorem incenter_relation :
  1 / IB + 1 / IC = 1 / IF := sorry

end points_B_O_I_H_C_cocylic_AH_AO_R_OI_IH_PO_HQ_OH_lengths_incenter_relation_l159_159420


namespace cafeteria_orders_green_apples_l159_159487

theorem cafeteria_orders_green_apples (G : ℕ) (h1 : 6 + G = 5 + 16) : G = 15 :=
by
  sorry

end cafeteria_orders_green_apples_l159_159487


namespace sin_105_value_cos_75_value_trigonometric_identity_l159_159104

noncomputable def sin_105_eq : Real := Real.sin (105 * Real.pi / 180)
noncomputable def cos_75_eq : Real := Real.cos (75 * Real.pi / 180)
noncomputable def cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq : Real := 
  Real.cos (Real.pi / 5) * Real.cos (3 * Real.pi / 10) - Real.sin (Real.pi / 5) * Real.sin (3 * Real.pi / 10)

theorem sin_105_value : sin_105_eq = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
  by sorry

theorem cos_75_value : cos_75_eq = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
  by sorry

theorem trigonometric_identity : cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq = 0 := 
  by sorry

end sin_105_value_cos_75_value_trigonometric_identity_l159_159104


namespace benjamin_walks_95_miles_in_a_week_l159_159374

def distance_to_work_one_way : ℕ := 6
def days_to_work : ℕ := 5
def distance_to_dog_walk_one_way : ℕ := 2
def dog_walks_per_day : ℕ := 2
def days_for_dog_walk : ℕ := 7
def distance_to_best_friend_one_way : ℕ := 1
def times_to_best_friend : ℕ := 1
def distance_to_convenience_store_one_way : ℕ := 3
def times_to_convenience_store : ℕ := 2

def total_distance_in_a_week : ℕ :=
  (days_to_work * 2 * distance_to_work_one_way) +
  (days_for_dog_walk * dog_walks_per_day * distance_to_dog_walk_one_way) +
  (times_to_convenience_store * 2 * distance_to_convenience_store_one_way) +
  (times_to_best_friend * 2 * distance_to_best_friend_one_way)

theorem benjamin_walks_95_miles_in_a_week :
  total_distance_in_a_week = 95 :=
by {
  simp [distance_to_work_one_way, days_to_work,
        distance_to_dog_walk_one_way, dog_walks_per_day,
        days_for_dog_walk, distance_to_best_friend_one_way,
        times_to_best_friend, distance_to_convenience_store_one_way,
        times_to_convenience_store, total_distance_in_a_week],
  sorry
}

end benjamin_walks_95_miles_in_a_week_l159_159374


namespace probability_of_event_l159_159675

open ProbabilityTheory

def a_n (n successes: ℕ) (n shots: ℕ) : ℚ := n successes / n shots

def successful_shot_probability : ℚ := 1 / 2

theorem probability_of_event :
  (a_6 = 1 / 2 ∧ ∀ n, (n: ℕ) < 6 → a_n ≤ 1 / 2) ⇔ (ℙ (independent_event (a_n) 6 full.set) = 5 / 64) :=
sorry

end probability_of_event_l159_159675


namespace min_value_c_plus_d_l159_159623

theorem min_value_c_plus_d (c d : ℤ) (h : c * d = 144) : c + d = -145 :=
sorry

end min_value_c_plus_d_l159_159623


namespace polynomial_root_product_l159_159136

theorem polynomial_root_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 1 = 0 → r^6 - b * r - c = 0) → b * c = 40 := 
by
  sorry

end polynomial_root_product_l159_159136


namespace smallest_integer_with_eight_factors_l159_159504

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l159_159504


namespace fair_coin_heads_before_tails_sum_m_n_l159_159938

-- Definition of the states and transitions according to the problem.
def prob_three_heads_before_two_tails : ℚ :=
  let q_1 := (1 : ℚ) / 2 in
  let r_1 := (1 : ℚ) / 4 in
  let q_2 := (1 : ℚ) / 2 + (1 : ℚ) / 2 * r_1 in
  let q_0 := (1 : ℚ) / 2 * q_1 + (1 : ℚ) / 2 * r_1 in
  q_0

-- Main theorem statement
theorem fair_coin_heads_before_tails_sum_m_n : let q := prob_three_heads_before_two_tails in
  ∃ m n : ℕ, (m : ℚ) / (n : ℚ) = q ∧ Nat.gcd m n = 1 ∧ m + n = 11 :=
begin
  sorry
end

end fair_coin_heads_before_tails_sum_m_n_l159_159938


namespace combined_hits_and_misses_total_l159_159745

/-
  Prove that given the conditions for each day regarding the number of misses and
  the ratio of misses to hits, the combined total of hits and misses for the 
  three days is 322.
-/

theorem combined_hits_and_misses_total :
  (∀ (H1 : ℕ) (H2 : ℕ) (H3 : ℕ), 
    (2 * H1 = 60) ∧ (3 * H2 = 84) ∧ (5 * H3 = 100) →
    60 + 84 + 100 + H1 + H2 + H3 = 322) :=
by
  sorry

end combined_hits_and_misses_total_l159_159745


namespace taxi_cost_per_mile_l159_159292

variable (x : ℝ)

-- Mike's total cost
def Mike_total_cost := 2.50 + 36 * x

-- Annie's total cost
def Annie_total_cost := 2.50 + 5.00 + 16 * x

-- The primary theorem to prove
theorem taxi_cost_per_mile : Mike_total_cost x = Annie_total_cost x → x = 0.25 := by
  sorry

end taxi_cost_per_mile_l159_159292


namespace courtyard_width_l159_159679

def width_of_courtyard (w : ℝ) : Prop :=
  28 * 100 * 100 * w = 13788 * 22 * 12

theorem courtyard_width :
  ∃ w : ℝ, width_of_courtyard w ∧ abs (w - 13.012) < 0.001 :=
by
  sorry

end courtyard_width_l159_159679


namespace remainder_2_pow_305_mod_9_l159_159988

theorem remainder_2_pow_305_mod_9 :
  2^305 % 9 = 5 :=
by sorry

end remainder_2_pow_305_mod_9_l159_159988


namespace benjamin_walks_95_miles_in_a_week_l159_159373

def distance_to_work_one_way : ℕ := 6
def days_to_work : ℕ := 5
def distance_to_dog_walk_one_way : ℕ := 2
def dog_walks_per_day : ℕ := 2
def days_for_dog_walk : ℕ := 7
def distance_to_best_friend_one_way : ℕ := 1
def times_to_best_friend : ℕ := 1
def distance_to_convenience_store_one_way : ℕ := 3
def times_to_convenience_store : ℕ := 2

def total_distance_in_a_week : ℕ :=
  (days_to_work * 2 * distance_to_work_one_way) +
  (days_for_dog_walk * dog_walks_per_day * distance_to_dog_walk_one_way) +
  (times_to_convenience_store * 2 * distance_to_convenience_store_one_way) +
  (times_to_best_friend * 2 * distance_to_best_friend_one_way)

theorem benjamin_walks_95_miles_in_a_week :
  total_distance_in_a_week = 95 :=
by {
  simp [distance_to_work_one_way, days_to_work,
        distance_to_dog_walk_one_way, dog_walks_per_day,
        days_for_dog_walk, distance_to_best_friend_one_way,
        times_to_best_friend, distance_to_convenience_store_one_way,
        times_to_convenience_store, total_distance_in_a_week],
  sorry
}

end benjamin_walks_95_miles_in_a_week_l159_159373


namespace part_a_region_part_b_region_part_c_region_l159_159226

-- Definitions for Part (a)
def surface1a (x y z : ℝ) := 2 * y = x ^ 2 + z ^ 2
def surface2a (x y z : ℝ) := x ^ 2 + z ^ 2 = 1
def region_a (x y z : ℝ) := surface1a x y z ∧ surface2a x y z

-- Definitions for Part (b)
def surface1b (x y z : ℝ) := z = 0
def surface2b (x y z : ℝ) := y + z = 2
def surface3b (x y z : ℝ) := y = x ^ 2
def region_b (x y z : ℝ) := surface1b x y z ∧ surface2b x y z ∧ surface3b x y z

-- Definitions for Part (c)
def surface1c (x y z : ℝ) := z = 6 - x ^ 2 - y ^ 2
def surface2c (x y z : ℝ) := x ^ 2 + y ^ 2 = z ^ 2
def region_c (x y z : ℝ) := surface1c x y z ∧ surface2c x y z

-- The formal theorem statements
theorem part_a_region : ∃x y z : ℝ, region_a x y z := by
  sorry

theorem part_b_region : ∃x y z : ℝ, region_b x y z := by
  sorry

theorem part_c_region : ∃x y z : ℝ, region_c x y z := by
  sorry

end part_a_region_part_b_region_part_c_region_l159_159226


namespace sum_n_k_of_binomial_coefficient_ratio_l159_159056

theorem sum_n_k_of_binomial_coefficient_ratio :
  ∃ (n k : ℕ), (n = (7 * k + 5) / 2) ∧ (2 * (n - k) = 5 * (k + 1)) ∧ 
    ((k % 2 = 1) ∧ (n + k = 7 ∨ n + k = 16)) ∧ (23 = 7 + 16) :=
by
  sorry

end sum_n_k_of_binomial_coefficient_ratio_l159_159056


namespace martin_total_distance_l159_159949

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l159_159949


namespace range_of_a_for_inequality_solutions_to_equation_l159_159015

noncomputable def f (x a : ℝ) := x^2 + 2 * a * x + 1
noncomputable def f_prime (x a : ℝ) := 2 * x + 2 * a

theorem range_of_a_for_inequality :
  (∀ x, -2 ≤ x ∧ x ≤ -1 → f x a ≤ f_prime x a) → a ≥ 3 / 2 :=
sorry

theorem solutions_to_equation (a : ℝ) (x : ℝ) :
  f x a = |f_prime x a| ↔ 
  (if a < -1 then x = -1 ∨ x = 1 - 2 * a 
  else if -1 ≤ a ∧ a ≤ 1 then x = 1 ∨ x = -1 ∨ x = 1 - 2 * a ∨ x = -(1 + 2 * a)
  else x = 1 ∨ x = -(1 + 2 * a)) :=
sorry

end range_of_a_for_inequality_solutions_to_equation_l159_159015


namespace identify_false_condition_l159_159705

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
def condition_A (a b c : ℝ) : Prop := quadratic_function a b c (-1) = 0
def condition_B (a b c : ℝ) : Prop := 2 * a + b = 0
def condition_C (a b c : ℝ) : Prop := quadratic_function a b c 1 = 3
def condition_D (a b c : ℝ) : Prop := quadratic_function a b c 2 = 8

-- Main theorem stating which condition is false
theorem identify_false_condition (a b c : ℝ) (ha : a ≠ 0) : ¬ condition_A a b c ∨ ¬ condition_B a b c ∨ ¬ condition_C a b c ∨  ¬ condition_D a b c :=
by
sorry

end identify_false_condition_l159_159705


namespace inequality_no_solution_iff_a_le_neg3_l159_159182

theorem inequality_no_solution_iff_a_le_neg3 (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 1| - |x + 2| < a)) ↔ a ≤ -3 := 
sorry

end inequality_no_solution_iff_a_le_neg3_l159_159182


namespace range_of_a_l159_159873

theorem range_of_a (a : ℝ) :  (5 - a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
by
  intro h
  sorry

end range_of_a_l159_159873


namespace unique_solution_x_y_z_l159_159551

theorem unique_solution_x_y_z (x y z : ℕ) (h1 : Prime y) (h2 : ¬ z % 3 = 0) (h3 : ¬ z % y = 0) :
    x^3 - y^3 = z^2 ↔ (x, y, z) = (8, 7, 13) := by
  sorry

end unique_solution_x_y_z_l159_159551


namespace solve_xy_l159_159564

theorem solve_xy : ∃ (x y : ℝ), x = 1 / 3 ∧ y = 2 / 3 ∧ x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 :=
by
  use 1 / 3, 2 / 3
  sorry

end solve_xy_l159_159564


namespace phil_quarters_l159_159631

def initial_quarters : ℕ := 50

def quarters_after_first_year (initial : ℕ) : ℕ := 2 * initial

def quarters_collected_second_year : ℕ := 3 * 12

def quarters_collected_third_year : ℕ := 12 / 3

def total_quarters_before_loss (initial : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ := 
  quarters_after_first_year initial + second_year + third_year

def lost_quarters (total : ℕ) : ℕ := total / 4

def quarters_left (total : ℕ) (lost : ℕ) : ℕ := total - lost

theorem phil_quarters : 
  quarters_left 
    (total_quarters_before_loss 
      initial_quarters 
      quarters_collected_second_year 
      quarters_collected_third_year)
    (lost_quarters 
      (total_quarters_before_loss 
        initial_quarters 
        quarters_collected_second_year 
        quarters_collected_third_year))
  = 105 :=
by
  sorry

end phil_quarters_l159_159631


namespace digital_earth_correct_purposes_l159_159485

def Purpose : Type := String

def P1 : Purpose := "To deal with natural and social issues of the entire Earth using digital means."
def P2 : Purpose := "To maximize the utilization of natural resources."
def P3 : Purpose := "To conveniently obtain information about the Earth."
def P4 : Purpose := "To provide precise locations, directions of movement, and speeds of moving objects."

def correct_purposes : Set Purpose := {P1, P2, P3}

theorem digital_earth_correct_purposes :
  {P1, P2, P3} = correct_purposes :=
by 
  sorry

end digital_earth_correct_purposes_l159_159485


namespace probability_two_red_two_blue_correct_l159_159673

noncomputable def num_ways_to_choose : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

noncomputable def probability_two_red_two_blue : ℚ :=
  let total_ways := num_ways_to_choose 20 4
  let ways_red := num_ways_to_choose 12 2
  let ways_blue := num_ways_to_choose 8 2
  (ways_red * ways_blue) / total_ways

theorem probability_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 :=
by
  sorry

end probability_two_red_two_blue_correct_l159_159673


namespace vasya_birthday_l159_159793

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159793


namespace vasya_birthday_day_l159_159794

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159794


namespace martin_total_distance_l159_159948

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l159_159948


namespace F_double_prime_coordinates_correct_l159_159498

structure Point where
  x : Int
  y : Int

def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_over_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := 6, y := -4 }

def F' : Point := reflect_over_y_axis F

def F'' : Point := reflect_over_x_axis F'

theorem F_double_prime_coordinates_correct : F'' = { x := -6, y := 4 } :=
  sorry

end F_double_prime_coordinates_correct_l159_159498


namespace range_of_a_l159_159396

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / y = 1) : 
  (x + y + a > 0) ↔ (a > -3 - 2 * Real.sqrt 2) :=
sorry

end range_of_a_l159_159396


namespace a_seq_formula_T_seq_sum_l159_159710

-- Definition of the sequence \( \{a_n\} \)
def a_seq (n : ℕ) (p : ℤ) : ℤ := 2 * n + 5

-- Condition: Sum of the first n terms \( s_n = n^2 + pn \)
def s_seq (n : ℕ) (p : ℤ) : ℤ := n^2 + p * n

-- Condition: \( \{a_2, a_5, a_{10}\} \) form a geometric sequence
def is_geometric (a2 a5 a10 : ℤ) : Prop :=
  a2 * a10 = a5 * a5

-- Definition of the sequence \( \{b_n\} \)
def b_seq (n : ℕ) (p : ℤ) : ℚ := 1 + 5 / (a_seq n p * a_seq (n + 1) p)

-- Function to find the sum of first n terms of \( \{b_n\} \)
def T_seq (n : ℕ) (p : ℤ) : ℚ :=
  n + 5 * (1 / (7 : ℚ) - 1 / (2 * n + 7 : ℚ)) + n / (14 * n + 49 : ℚ)

theorem a_seq_formula (p : ℤ) : ∀ n, a_seq n p = 2 * n + 5 :=
by
  sorry

theorem T_seq_sum (p : ℤ) : ∀ n, T_seq n p = (14 * n^2 + 54 * n) / (14 * n + 49) :=
by
  sorry

end a_seq_formula_T_seq_sum_l159_159710


namespace distance_proof_l159_159213

-- Define the speeds of Alice and Bob
def aliceSpeed : ℚ := 1 / 20 -- Alice's speed in miles per minute
def bobSpeed : ℚ := 3 / 40 -- Bob's speed in miles per minute

-- Define the time they walk/jog
def time : ℚ := 120 -- Time in minutes (2 hours)

-- Calculate the distances
def aliceDistance : ℚ := aliceSpeed * time -- Distance Alice walked
def bobDistance : ℚ := bobSpeed * time -- Distance Bob jogged

-- The total distance between Alice and Bob after 2 hours
def totalDistance : ℚ := aliceDistance + bobDistance

-- Prove that the total distance is 15 miles
theorem distance_proof : totalDistance = 15 := by
  sorry

end distance_proof_l159_159213


namespace necessary_not_sufficient_cond_l159_159735

theorem necessary_not_sufficient_cond (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y < 4 → xy < 4) ∧ ¬(xy < 4 → x + y < 4) :=
  by
    sorry

end necessary_not_sufficient_cond_l159_159735


namespace betty_needs_more_flies_l159_159098

-- Definitions for the number of flies consumed by the frog each day
def fliesMonday : ℕ := 3
def fliesTuesday : ℕ := 2
def fliesWednesday : ℕ := 4
def fliesThursday : ℕ := 5
def fliesFriday : ℕ := 1
def fliesSaturday : ℕ := 2
def fliesSunday : ℕ := 3

-- Definition for the total number of flies eaten by the frog in a week
def totalFliesEaten : ℕ :=
  fliesMonday + fliesTuesday + fliesWednesday + fliesThursday + fliesFriday + fliesSaturday + fliesSunday

-- Definitions for the number of flies caught by Betty
def fliesMorning : ℕ := 5
def fliesAfternoon : ℕ := 6
def fliesEscaped : ℕ := 1

-- Definition for the total number of flies caught by Betty considering the escape
def totalFliesCaught : ℕ := fliesMorning + fliesAfternoon - fliesEscaped

-- Lean 4 statement to prove the number of additional flies Betty needs to catch
theorem betty_needs_more_flies : 
  totalFliesEaten - totalFliesCaught = 10 := 
by
  sorry

end betty_needs_more_flies_l159_159098


namespace first_problem_solution_set_second_problem_a_range_l159_159715

-- Define the function f(x) = |2x - a| + |x - 1|
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 1)

-- First problem: When a = 3, the solution set of the inequality f(x) ≥ 2
theorem first_problem_solution_set (x : ℝ) : (f x 3 ≥ 2) ↔ (x ≤ 2 / 3 ∨ x ≥ 2) :=
by sorry

-- Second problem: If f(x) ≥ 5 - x for ∀ x ∈ ℝ, find the range of the real number a
theorem second_problem_a_range (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ (6 ≤ a) :=
by sorry

end first_problem_solution_set_second_problem_a_range_l159_159715


namespace parallel_vectors_implies_m_eq_neg1_l159_159901

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_implies_m_eq_neg1_l159_159901


namespace triangle_third_side_lengths_l159_159723

theorem triangle_third_side_lengths : 
  ∃ (x : ℕ), (3 < x ∧ x < 11) ∧ (x ≠ 3) ∧ (x ≠ 11) ∧ 
    ((x = 4) ∨ (x = 5) ∨ (x = 6) ∨ (x = 7) ∨ (x = 8) ∨ (x = 9) ∨ (x = 10)) :=
by
  sorry

end triangle_third_side_lengths_l159_159723


namespace clubs_students_equal_l159_159431

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l159_159431


namespace find_m_if_parallel_l159_159720

-- Given vectors
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

-- Parallel condition and the result that m must be -2 or 2
theorem find_m_if_parallel (m : ℝ) (h : ∃ k : ℝ, a m = (k * (b m).fst, k * (b m).snd)) : 
  m = -2 ∨ m = 2 :=
sorry

end find_m_if_parallel_l159_159720


namespace inequality_proof_l159_159174

theorem inequality_proof (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := 
sorry

end inequality_proof_l159_159174


namespace car_balanced_by_cubes_l159_159688

variable (M Ball Cube : ℝ)

-- Conditions from the problem
axiom condition1 : M = Ball + 2 * Cube
axiom condition2 : M + Cube = 2 * Ball

-- Theorem to prove
theorem car_balanced_by_cubes : M = 5 * Cube := sorry

end car_balanced_by_cubes_l159_159688


namespace first_two_digits_of_52x_l159_159526

-- Define the digit values that would make 52x divisible by 6.
def digit_values (x : Nat) : Prop :=
  x = 2 ∨ x = 5 ∨ x = 8

-- The main theorem to prove the first two digits are 52 given the conditions.
theorem first_two_digits_of_52x (x : Nat) (h : digit_values x) : (52 * 10 + x) / 10 = 52 :=
by sorry

end first_two_digits_of_52x_l159_159526


namespace aarti_work_multiple_l159_159367

-- Aarti can do a piece of work in 5 days
def days_per_unit_work := 5

-- It takes her 15 days to complete the certain multiple of work
def days_for_multiple_work := 15

-- Prove the ratio of the days for multiple work to the days per unit work equals 3
theorem aarti_work_multiple :
  days_for_multiple_work / days_per_unit_work = 3 :=
sorry

end aarti_work_multiple_l159_159367


namespace p_q_false_of_not_or_l159_159601

variables (p q : Prop)

theorem p_q_false_of_not_or (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by {
  sorry
}

end p_q_false_of_not_or_l159_159601


namespace samantha_total_cost_l159_159857

-- Defining the conditions in Lean
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℕ := 25
def loads : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Proving the total cost Samantha spends is $11
theorem samantha_total_cost : (loads * washer_cost + num_dryers * (dryer_time / 10 * dryer_cost_per_10_min)) = 1100 :=
by
  sorry

end samantha_total_cost_l159_159857


namespace sum_of_integers_l159_159054

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end sum_of_integers_l159_159054


namespace value_of_y_l159_159310

noncomputable def k : ℝ := 168.75

theorem value_of_y (x y : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x = 3 * y) : y = -16.875 :=
by 
  sorry

end value_of_y_l159_159310


namespace factorial_trailing_zeros_l159_159594

theorem factorial_trailing_zeros (n : ℕ) (h : n = 30) : 
  nat.trailing_zeroes (nat.factorial n) = 7 :=
by
  sorry

end factorial_trailing_zeros_l159_159594


namespace parabola_passes_through_point_l159_159311

theorem parabola_passes_through_point {x y : ℝ} (h_eq : y = (1/2) * x^2 - 2) :
  (x = 2 ∧ y = 0) :=
by
  sorry

end parabola_passes_through_point_l159_159311


namespace abc_value_l159_159598

-- Variables declarations
variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := a + b + c = 1
def condition2 : Prop := a^2 + b^2 + c^2 = 2
def condition3 : Prop := a^3 + b^3 + c^3 = 3

-- Question to prove
theorem abc_value : condition1 a b c → condition2 a b c → condition3 a b c → a * b * c = 1/6 :=
by
  sorry

end abc_value_l159_159598


namespace total_amount_l159_159341

variable (x y z : ℝ)

def condition1 : Prop := y = 0.45 * x
def condition2 : Prop := z = 0.30 * x
def condition3 : Prop := y = 36

theorem total_amount (h1 : condition1 x y)
                     (h2 : condition2 x z)
                     (h3 : condition3 y) :
  x + y + z = 140 :=
by
  sorry

end total_amount_l159_159341


namespace cannot_be_sum_of_two_or_more_consecutive_integers_l159_159337

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem cannot_be_sum_of_two_or_more_consecutive_integers (n : ℕ) :
  (¬∃ k m : ℕ, k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ is_power_of_two n :=
by
  sorry

end cannot_be_sum_of_two_or_more_consecutive_integers_l159_159337


namespace students_with_all_three_pets_l159_159606

theorem students_with_all_three_pets :
  ∀ (total_students : ℕ)
    (dog_fraction cat_fraction : ℚ)
    (other_pets students_no_pets dogs_only cats_only other_pets_only x y z w : ℕ),
    total_students = 40 →
    dog_fraction = 5 / 8 →
    cat_fraction = 1 / 4 →
    other_pets = 8 →
    students_no_pets = 4 →
    dogs_only = 15 →
    cats_only = 3 →
    other_pets_only = 2 →
    dogs_only + x + z + w = total_students * dog_fraction →
    cats_only + x + y + w = total_students * cat_fraction →
    other_pets_only + y + z + w = other_pets →
    dogs_only + cats_only + other_pets_only + x + y + z + w = total_students - students_no_pets →
    w = 4  := 
by
  sorry

end students_with_all_three_pets_l159_159606


namespace fraction_is_one_third_l159_159217

theorem fraction_is_one_third :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 :=
by
  sorry

end fraction_is_one_third_l159_159217


namespace find_max_z_plus_x_l159_159521

theorem find_max_z_plus_x : 
  (∃ (x y z t: ℝ), x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ xt + yz ≥ 6 ∧ z + x = 5) :=
sorry

end find_max_z_plus_x_l159_159521


namespace subcommittee_formation_l159_159523

/-- A Senate committee consists of 10 Republicans and 7 Democrats.
    The number of ways to form a subcommittee with 4 Republicans and 3 Democrats is 7350. -/
theorem subcommittee_formation :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end subcommittee_formation_l159_159523


namespace gcd_of_198_and_286_l159_159331

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end gcd_of_198_and_286_l159_159331


namespace find_x_l159_159897

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for perpendicular vectors
def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem find_x (x : ℝ) (h : perpendicular a (b x)) : x = -3 / 2 :=
by
  sorry

end find_x_l159_159897


namespace vasya_birthday_day_l159_159796

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159796


namespace Roger_years_to_retire_l159_159876

noncomputable def Peter : ℕ := 12
noncomputable def Robert : ℕ := Peter - 4
noncomputable def Mike : ℕ := Robert - 2
noncomputable def Tom : ℕ := 2 * Robert
noncomputable def Roger : ℕ := Peter + Tom + Robert + Mike

theorem Roger_years_to_retire :
  Roger = 42 → 50 - Roger = 8 := by
sorry

end Roger_years_to_retire_l159_159876


namespace smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l159_159405

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem centers_of_symmetry :
  ∀ k : ℤ, ∃ x, x = -Real.pi / 4 + k * Real.pi ∧ f (-x) = f x := sorry

theorem maximum_value :
  ∀ x : ℝ, f x ≤ 2 := sorry

theorem minimum_value :
  ∀ x : ℝ, f x ≥ -1 := sorry

end smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l159_159405


namespace fraction_value_l159_159590

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : x^2 / (x^4 + x^2 + 1) = 1/8 :=
by sorry

end fraction_value_l159_159590


namespace greatest_integer_part_expected_winnings_l159_159994

noncomputable def expected_winnings_one_envelope : ℝ := 500

noncomputable def expected_winnings_two_envelopes : ℝ := 625

noncomputable def expected_winnings_three_envelopes : ℝ := 695.3125

theorem greatest_integer_part_expected_winnings :
  ⌊expected_winnings_three_envelopes⌋ = 695 :=
by 
  sorry

end greatest_integer_part_expected_winnings_l159_159994


namespace find_trajectory_l159_159394

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (y - 1) * (y + 1) / ((x + 1) * (x - 1)) = -1 / 3

theorem find_trajectory (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  trajectory_equation x y → x^2 + 3 * y^2 = 4 :=
by
  sorry

end find_trajectory_l159_159394


namespace part_a_part_b_l159_159469

-- Define sum conditions for consecutive odd integers
def consecutive_odd_sum (N : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ N = n * (2 * k + n)

-- Part (a): Prove 2005 can be written as sum of consecutive odd positive integers
theorem part_a : consecutive_odd_sum 2005 :=
by
  sorry

-- Part (b): Prove 2006 cannot be written as sum of consecutive odd positive integers
theorem part_b : ¬consecutive_odd_sum 2006 :=
by
  sorry

end part_a_part_b_l159_159469


namespace probability_sum_divisible_by_three_l159_159912

open Set
open Finset

-- Defining the set of first nine prime numbers.
def first_nine_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Condition: Two distinct numbers are selected at random.
def distinct_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  (s.product s).filter (λ p => p.1 < p.2)

-- Definition of pairs whose sum is divisible by 3.
def divisible_by_three_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  (distinct_pairs s).filter (λ p => (p.1 + p.2) % 3 = 0)

-- Theorem statement.
theorem probability_sum_divisible_by_three :
  (divisible_by_three_pairs first_nine_primes).card.toRat /
    (distinct_pairs first_nine_primes).card.toRat = 2 / 9 :=
by
  sorry

end probability_sum_divisible_by_three_l159_159912


namespace scientific_notation_1742000_l159_159556

theorem scientific_notation_1742000 : 1742000 = 1.742 * 10^6 := 
by
  sorry

end scientific_notation_1742000_l159_159556


namespace johns_total_payment_l159_159439

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end johns_total_payment_l159_159439


namespace closest_distance_canteen_l159_159083

theorem closest_distance_canteen :
  ∃ (x : ℝ), x = 125 * Real.sqrt 2 ∧
    let A : ℝ := 400 * Real.cos (Real.pi / 6)
    let B : ℝ := 400 * Real.sin (Real.pi / 6)
    let perp_distance_B := Real.sqrt (600^2 - B^2)
    in A^2 + x^2 = B^2 + (perp_distance_B - x)^2 :=
sorry

end closest_distance_canteen_l159_159083


namespace prime_divides_factorial_difference_l159_159571

theorem prime_divides_factorial_difference (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) : 
  p^5 ∣ (Nat.factorial p - p) := by
  sorry

end prime_divides_factorial_difference_l159_159571


namespace fraction_of_capital_subscribed_l159_159686

theorem fraction_of_capital_subscribed (T : ℝ) (x : ℝ) :
  let B_capital := (1 / 4) * T
  let C_capital := (1 / 5) * T
  let Total_profit := 2445
  let A_profit := 815
  A_profit / Total_profit = x → x = 1 / 3 :=
by
  sorry

end fraction_of_capital_subscribed_l159_159686


namespace total_overtime_hours_worked_l159_159158

def gary_wage : ℕ := 12
def mary_wage : ℕ := 14
def john_wage : ℕ := 16
def alice_wage : ℕ := 18
def michael_wage : ℕ := 20

def regular_hours : ℕ := 40
def overtime_rate : ℚ := 1.5

def total_paycheck : ℚ := 3646

theorem total_overtime_hours_worked :
  let gary_overtime := gary_wage * overtime_rate
  let mary_overtime := mary_wage * overtime_rate
  let john_overtime := john_wage * overtime_rate
  let alice_overtime := alice_wage * overtime_rate
  let michael_overtime := michael_wage * overtime_rate
  let regular_total := (gary_wage + mary_wage + john_wage + alice_wage + michael_wage) * regular_hours
  let total_overtime_pay := total_paycheck - regular_total
  let total_overtime_rate := gary_overtime + mary_overtime + john_overtime + alice_overtime + michael_overtime
  let overtime_hours := total_overtime_pay / total_overtime_rate
  overtime_hours.floor = 3 := 
by
  sorry

end total_overtime_hours_worked_l159_159158


namespace truth_probability_of_A_l159_159209

theorem truth_probability_of_A (P_B : ℝ) (P_AB : ℝ) (h : P_AB = 0.45 ∧ P_B = 0.60 ∧ ∀ (P_A : ℝ), P_AB = P_A * P_B) : 
  ∃ (P_A : ℝ), P_A = 0.75 :=
by
  sorry

end truth_probability_of_A_l159_159209


namespace polar_distance_l159_159163

theorem polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hθ : θ1 - θ2 = (Real.pi / 3)) : 
  ∃ d : ℝ, d = real.sqrt (r1 * r1 + r2 * r2 - 2 * r1 * r2 * Real.cos (θ1 - θ2)) := by
  have h1 : r1 = 4 := sorry
  have h2 : r2 = 12 := sorry
  have h3 : θ1 - θ2 = Real.pi / 3 := by assumption
  use 4 * Real.sqrt 7
  sorry

end polar_distance_l159_159163


namespace olympiad_scores_above_18_l159_159921

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l159_159921


namespace painted_cube_probability_l159_159220

-- Define the conditions
def cube_size : Nat := 5
def total_unit_cubes : Nat := cube_size ^ 3
def corner_cubes_with_three_faces : Nat := 1
def edges_with_two_faces : Nat := 3 * (cube_size - 2) -- 3 edges, each (5 - 2) = 3
def faces_with_one_face : Nat := 2 * (cube_size * cube_size - corner_cubes_with_three_faces - edges_with_two_faces)
def no_painted_faces_cubes : Nat := total_unit_cubes - corner_cubes_with_three_faces - faces_with_one_face

-- Compute the probability
def probability := (corner_cubes_with_three_faces * no_painted_faces_cubes) / (total_unit_cubes * (total_unit_cubes - 1) / 2)

-- The theorem statement
theorem painted_cube_probability :
  probability = (2 : ℚ) / 155 := 
by {
  sorry
}

end painted_cube_probability_l159_159220


namespace unique_two_digit_integer_l159_159981

theorem unique_two_digit_integer (t : ℕ) (h : 11 * t % 100 = 36) (ht : 10 ≤ t ∧ t ≤ 99) : t = 76 :=
by
  sorry

end unique_two_digit_integer_l159_159981


namespace determine_h_l159_159230

-- Define the initial quadratic expression
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the form we want to prove
def completed_square_form (x h k : ℝ) : ℝ := 3 * (x - h)^2 + k

-- The proof problem translated to Lean 4
theorem determine_h : ∃ k : ℝ, ∀ x : ℝ, quadratic x = completed_square_form x (-4 / 3) k :=
by
  exists (29 / 3)
  intro x
  sorry

end determine_h_l159_159230


namespace negation_of_exactly_one_even_l159_159325

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

theorem negation_of_exactly_one_even :
  ¬ exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
                                 (is_even a ∧ is_even b) ∨
                                 (is_even a ∧ is_even c) ∨
                                 (is_even b ∧ is_even c) :=
by sorry

end negation_of_exactly_one_even_l159_159325


namespace find_original_number_l159_159078

-- Define the given conditions
def increased_by_twenty_percent (x : ℝ) : ℝ := x * 1.20

-- State the theorem
theorem find_original_number (x : ℝ) (h : increased_by_twenty_percent x = 480) : x = 400 :=
by
  sorry

end find_original_number_l159_159078


namespace rhombus_perimeter_l159_159960

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l159_159960


namespace ajhsme_1989_reappears_at_12_l159_159308

def cycle_length_letters : ℕ := 6
def cycle_length_digits  : ℕ := 4
def target_position : ℕ := Nat.lcm cycle_length_letters cycle_length_digits

theorem ajhsme_1989_reappears_at_12 :
  target_position = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end ajhsme_1989_reappears_at_12_l159_159308


namespace meaningful_expr_implies_x_gt_1_l159_159659

theorem meaningful_expr_implies_x_gt_1 (x : ℝ) : (∃ y : ℝ, y = 1 / real.sqrt (x - 1)) → x > 1 :=
by
  sorry

end meaningful_expr_implies_x_gt_1_l159_159659


namespace problem_part1_problem_part2_l159_159406

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + a

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 3

theorem problem_part1 (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x 2 ≥ 2) :
    ∃ a : ℝ, a = 2 ∧ 
    ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∃ m : ℤ, x = (m * Real.pi / 2 + Real.pi / 12) ∨ x = (m * Real.pi / 2 + Real.pi / 4) := sorry

theorem problem_part2 :
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), g x = 4 → 
    ∃ s : ℝ, s = Real.pi / 3 := sorry

end problem_part1_problem_part2_l159_159406


namespace total_boys_in_class_l159_159423

theorem total_boys_in_class (n : ℕ)
  (h1 : 19 + 19 - 1 = n) :
  n = 37 :=
  sorry

end total_boys_in_class_l159_159423


namespace vec_sub_eq_l159_159143

variables (a b : ℝ × ℝ)
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (-3, 4)

theorem vec_sub_eq : vec_a - vec_b = (5, -3) :=
by 
  -- You can fill in the proof steps here
  sorry

end vec_sub_eq_l159_159143


namespace Adam_total_balls_l159_159542

def number_of_red_balls := 20
def number_of_blue_balls := 10
def number_of_orange_balls := 5
def number_of_pink_balls := 3 * number_of_orange_balls

def total_number_of_balls := 
  number_of_red_balls + number_of_blue_balls + number_of_pink_balls + number_of_orange_balls

theorem Adam_total_balls : total_number_of_balls = 50 := by
  sorry

end Adam_total_balls_l159_159542


namespace largest_a_for_integer_solution_l159_159120

noncomputable def largest_integer_a : ℤ := 11

theorem largest_a_for_integer_solution :
  ∃ (x : ℤ), ∃ (a : ℤ), 
  (∃ (a : ℤ), a ≤ largest_integer_a) ∧
  (a = largest_integer_a → (
    (x^2 - (a + 7) * x + 7 * a)^3 = -3^3)) := 
by 
  sorry

end largest_a_for_integer_solution_l159_159120


namespace multiples_of_7_not_14_l159_159253

theorem multiples_of_7_not_14 :
  { n : ℕ | n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0 }.card = 36 := by
  sorry

end multiples_of_7_not_14_l159_159253


namespace min_value_fraction_l159_159242

theorem min_value_fraction (a b : ℝ) (h₀ : a > b) (h₁ : a * b = 1) :
  ∃ c, c = (2 * Real.sqrt 2) ∧ (a^2 + b^2) / (a - b) ≥ c :=
by sorry

end min_value_fraction_l159_159242


namespace eliminate_y_l159_159773

theorem eliminate_y (x y : ℝ) (h1 : 2 * x + 3 * y = 1) (h2 : 3 * x - 6 * y = 7) :
  (4 * x + 6 * y) + (3 * x - 6 * y) = 9 :=
by
  sorry

end eliminate_y_l159_159773


namespace congruence_solution_count_l159_159013

theorem congruence_solution_count :
  ∃! x : ℕ, x < 50 ∧ x + 20 ≡ 75 [MOD 43] := 
by
  sorry

end congruence_solution_count_l159_159013


namespace sarah_wide_reflections_l159_159386

variables (tall_mirrors_sarah : ℕ) (tall_mirrors_ellie : ℕ) 
          (wide_mirrors_ellie : ℕ) (tall_count : ℕ) (wide_count : ℕ)
          (total_reflections : ℕ) (S : ℕ)

def reflections_in_tall_mirrors_sarah := 10 * tall_count
def reflections_in_tall_mirrors_ellie := 6 * tall_count
def reflections_in_wide_mirrors_ellie := 3 * wide_count
def total_reflections_no_wide_sarah := reflections_in_tall_mirrors_sarah + reflections_in_tall_mirrors_ellie + reflections_in_wide_mirrors_ellie

theorem sarah_wide_reflections :
  reflections_in_tall_mirrors_sarah = 30 →
  reflections_in_tall_mirrors_ellie = 18 →
  reflections_in_wide_mirrors_ellie = 15 →
  tall_count = 3 →
  wide_count = 5 →
  total_reflections = 88 →
  total_reflections = total_reflections_no_wide_sarah + 5 * S →
  S = 5 :=
sorry

end sarah_wide_reflections_l159_159386


namespace no_real_roots_of_quadratic_l159_159317

theorem no_real_roots_of_quadratic 
(a b c : ℝ) 
(h1 : b + c > a)
(h2 : b + a > c)
(h3 : c + a > b) :
(b^2 + c^2 - a^2)^2 - 4 * b^2 * c^2 < 0 :=
by
  sorry

end no_real_roots_of_quadratic_l159_159317


namespace can_form_triangle_l159_159192

-- Define the function to check for the triangle inequality
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Problem statement: Prove that only the set (3, 4, 6) can form a triangle
theorem can_form_triangle :
  (¬ is_triangle 3 4 8) ∧
  (¬ is_triangle 5 6 11) ∧
  (¬ is_triangle 5 8 15) ∧
  (is_triangle 3 4 6) :=
by
  sorry

end can_form_triangle_l159_159192


namespace balloons_initial_count_l159_159178

theorem balloons_initial_count (x : ℕ) (h : x + 13 = 60) : x = 47 :=
by
  -- proof skipped
  sorry

end balloons_initial_count_l159_159178


namespace percentage_increase_sale_l159_159191

theorem percentage_increase_sale (P S : ℝ) (hP : P > 0) (hS : S > 0) 
  (h1 : ∀ P S : ℝ, 0.7 * P * S * (1 + X / 100) = 1.26 * P * S) : 
  X = 80 := 
by
  sorry

end percentage_increase_sale_l159_159191


namespace find_units_digit_l159_159875

theorem find_units_digit : 
  (7^1993 + 5^1993) % 10 = 2 :=
by
  sorry

end find_units_digit_l159_159875


namespace problem_part1_problem_part2_problem_part3_l159_159527

noncomputable def find_ab (a b : ℝ) : Prop :=
  (5 * a + b = 40) ∧ (30 * a + b = 140)

noncomputable def production_cost (x : ℕ) : Prop :=
  (4 * x + 20 + 7 * (100 - x) = 660)

noncomputable def transport_cost (m : ℝ) : Prop :=
  ∃ n : ℝ, 10 ≤ n ∧ n ≤ 20 ∧ (m - 2) * n + 130 = 150

theorem problem_part1 : ∃ (a b : ℝ), find_ab a b ∧ a = 4 ∧ b = 20 := 
  sorry

theorem problem_part2 : ∃ (x : ℕ), production_cost x ∧ x = 20 := 
  sorry

theorem problem_part3 : ∃ (m : ℝ), transport_cost m ∧ m = 4 := 
  sorry

end problem_part1_problem_part2_problem_part3_l159_159527


namespace sofiya_wins_l159_159770

/-- Define the initial configuration and game rules -/
def initial_configuration : Type := { n : Nat // n = 2025 }

/--
  Define the game such that Sofiya starts and follows the strategy of always
  removing a neighbor from the arc with an even number of people.
-/
def winning_strategy (n : initial_configuration) : Prop :=
  n.1 % 2 = 1 ∧ 
  (∀ turn : Nat, turn % 2 = 0 → 
    (∃ arc : initial_configuration, arc.1 % 2 = 0 ∧ arc.1 < n.1) ∧
    (∀ marquis_turn : Nat, marquis_turn % 2 = 1 → 
      (∃ arc : initial_configuration, arc.1 % 2 = 1)))

/-- Sofiya has the winning strategy given the conditions of the game -/
theorem sofiya_wins : winning_strategy ⟨2025, rfl⟩ :=
sorry

end sofiya_wins_l159_159770


namespace cost_of_adult_ticket_l159_159186

def cost_of_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50
def adult_tickets : ℕ := 5

theorem cost_of_adult_ticket
  (A : ℝ)
  (h : 5 * A + 16 * cost_of_child_ticket = total_cost) :
  A = 5.50 :=
by
  sorry

end cost_of_adult_ticket_l159_159186


namespace radius_of_hole_l159_159744

-- Define the dimensions of the rectangular solid
def length1 : ℕ := 3
def length2 : ℕ := 8
def length3 : ℕ := 9

-- Define the radius of the hole
variable (r : ℕ)

-- Condition: The area of the 2 circles removed equals the lateral surface area of the cylinder
axiom area_condition : 2 * Real.pi * r^2 = 2 * Real.pi * r * length1

-- Prove that the radius of the cylindrical hole is 3
theorem radius_of_hole : r = 3 := by
  sorry

end radius_of_hole_l159_159744


namespace everton_college_calculators_l159_159001

theorem everton_college_calculators (total_cost : ℤ) (num_scientific_calculators : ℤ) 
  (cost_per_scientific : ℤ) (cost_per_graphing : ℤ) (total_scientific_cost : ℤ) 
  (num_graphing_calculators : ℤ) (total_graphing_cost : ℤ) (total_calculators : ℤ) :
  total_cost = 1625 ∧
  num_scientific_calculators = 20 ∧
  cost_per_scientific = 10 ∧
  cost_per_graphing = 57 ∧
  total_scientific_cost = num_scientific_calculators * cost_per_scientific ∧
  total_graphing_cost = num_graphing_calculators * cost_per_graphing ∧
  total_cost = total_scientific_cost + total_graphing_cost ∧
  total_calculators = num_scientific_calculators + num_graphing_calculators → 
  total_calculators = 45 :=
by
  intros
  sorry

end everton_college_calculators_l159_159001


namespace gwen_math_problems_l159_159145

-- Problem statement
theorem gwen_math_problems (m : ℕ) (science_problems : ℕ := 11) (problems_finished_at_school : ℕ := 24) (problems_left_for_homework : ℕ := 5) 
  (h1 : m + science_problems = problems_finished_at_school + problems_left_for_homework) : m = 18 := 
by {
  sorry
}

end gwen_math_problems_l159_159145


namespace solution_set_of_inequality_l159_159016

def f : Int → Int
| -1 => -1
| 0 => -1
| 1 => 1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

def g : Int → Int
| -1 => 1
| 0 => 1
| 1 => -1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

theorem solution_set_of_inequality :
  {x | f (g x) > 0} = { -1, 0 } :=
by
  sorry

end solution_set_of_inequality_l159_159016


namespace find_m_l159_159018

def vector (α : Type*) := α × α

def a : vector ℤ := (1, -2)
def b : vector ℤ := (3, 0)

def two_a_plus_b (a b : vector ℤ) : vector ℤ := (2 * a.1 + b.1, 2 * a.2 + b.2)
def m_a_minus_b (m : ℤ) (a b : vector ℤ) : vector ℤ := (m * a.1 - b.1, m * a.2 - b.2)

def parallel (v w : vector ℤ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_m : parallel (two_a_plus_b a b) (m_a_minus_b (-2) a b) :=
by
  sorry -- proof placeholder

end find_m_l159_159018


namespace isabella_most_efficient_jumper_l159_159043

noncomputable def weight_ricciana : ℝ := 120
noncomputable def jump_ricciana : ℝ := 4

noncomputable def weight_margarita : ℝ := 110
noncomputable def jump_margarita : ℝ := 2 * jump_ricciana - 1

noncomputable def weight_isabella : ℝ := 100
noncomputable def jump_isabella : ℝ := jump_ricciana + 3

noncomputable def ratio_ricciana : ℝ := weight_ricciana / jump_ricciana
noncomputable def ratio_margarita : ℝ := weight_margarita / jump_margarita
noncomputable def ratio_isabella : ℝ := weight_isabella / jump_isabella

theorem isabella_most_efficient_jumper :
  ratio_isabella < ratio_margarita ∧ ratio_isabella < ratio_ricciana :=
by
  sorry

end isabella_most_efficient_jumper_l159_159043


namespace conjugate_subgroups_l159_159283

open Group

variables (G : Type*) [Group G] (n : ℕ) (H1 H2 : Subgroup G)
variables (h1 : H1.index = n) (h2 : H2.index = n)
variables (intersectIndex : (H1 ⊓ H2).index = n * (n - 1))

theorem conjugate_subgroups :
  ∃ g : G, g⁻¹ * H1 * g = H2 :=
sorry

end conjugate_subgroups_l159_159283


namespace num_unique_m_values_l159_159760

theorem num_unique_m_values : 
  ∃ (s : Finset Int), 
  (∀ (x1 x2 : Int), x1 * x2 = 36 → x1 + x2 ∈ s) ∧ 
  s.card = 10 := 
sorry

end num_unique_m_values_l159_159760


namespace computation_result_l159_159109

theorem computation_result : 8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end computation_result_l159_159109


namespace soda_ratio_l159_159207

theorem soda_ratio (total_sodas diet_sodas regular_sodas : ℕ) (h1 : total_sodas = 64) (h2 : diet_sodas = 28) (h3 : regular_sodas = total_sodas - diet_sodas) : regular_sodas / Nat.gcd regular_sodas diet_sodas = 9 ∧ diet_sodas / Nat.gcd regular_sodas diet_sodas = 7 :=
by
  sorry

end soda_ratio_l159_159207


namespace MikaWaterLeft_l159_159169

def MikaWaterRemaining (startWater : ℚ) (usedWater : ℚ) : ℚ :=
  startWater - usedWater

theorem MikaWaterLeft :
  MikaWaterRemaining 3 (11 / 8) = 13 / 8 :=
by 
  sorry

end MikaWaterLeft_l159_159169


namespace isosceles_triangle_perimeter_l159_159649

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a^2 - 9 * a + 18 = 0) (h2 : b^2 - 9 * b + 18 = 0) (h3 : a ≠ b) :
  a + 2 * b = 15 :=
by
  -- Proof is omitted.
  sorry

end isosceles_triangle_perimeter_l159_159649


namespace convert_base10_to_base7_l159_159862

-- Definitions for powers and conditions
def n1 : ℕ := 7
def n2 : ℕ := n1 * n1
def n3 : ℕ := n2 * n1
def n4 : ℕ := n3 * n1

theorem convert_base10_to_base7 (n : ℕ) (h₁ : n = 395) : 
  ∃ a b c d : ℕ, 
    a * n3 + b * n2 + c * n1 + d = 395 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 3 :=
by { sorry }

end convert_base10_to_base7_l159_159862


namespace units_digit_base_6_l159_159102

theorem units_digit_base_6 (n m : ℕ) (h₁ : n = 312) (h₂ : m = 67) : (312 * 67) % 6 = 0 :=
by {
  sorry
}

end units_digit_base_6_l159_159102


namespace unique_intersection_value_k_l159_159125

theorem unique_intersection_value_k (k : ℝ) : (∀ x y: ℝ, (y = x^2) ∧ (y = 3*x + k) ↔ k = -9/4) :=
by
  sorry

end unique_intersection_value_k_l159_159125


namespace distance_between_stations_l159_159985

theorem distance_between_stations 
  (distance_P_to_meeting : ℝ)
  (distance_Q_to_meeting : ℝ)
  (h1 : distance_P_to_meeting = 20 * 3)
  (h2 : distance_Q_to_meeting = 25 * 2)
  (h3 : distance_P_to_meeting + distance_Q_to_meeting = D) :
  D = 110 :=
by
  sorry

end distance_between_stations_l159_159985


namespace line_third_quadrant_l159_159249

theorem line_third_quadrant (A B C : ℝ) (h_origin : C = 0)
  (h_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x - B * y = 0) :
  A * B < 0 :=
by
  sorry

end line_third_quadrant_l159_159249


namespace rhombus_perimeter_l159_159964

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l159_159964


namespace measure_of_B_l159_159421

theorem measure_of_B (a b : ℝ) (A B : ℝ) (angleA_nonneg : 0 < A ∧ A < 180) (angleB_nonneg : 0 < B ∧ B < 180)
    (a_eq : a = 1) (b_eq : b = Real.sqrt 3) (A_eq : A = 30) :
    B = 60 :=
by
  sorry

end measure_of_B_l159_159421


namespace age_sum_proof_l159_159168

theorem age_sum_proof : 
  ∀ (Matt Fem Jake : ℕ), 
    Matt = 4 * Fem →
    Fem = 11 →
    Jake = Matt + 5 →
    (Matt + 2) + (Fem + 2) + (Jake + 2) = 110 :=
by
  intros Matt Fem Jake h1 h2 h3
  sorry

end age_sum_proof_l159_159168


namespace olympiad_scores_above_18_l159_159922

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l159_159922


namespace smallest_X_value_l159_159619

noncomputable def T : ℕ := 111000
axiom T_digits_are_0s_and_1s : ∀ d, d ∈ (T.digits 10) → d = 0 ∨ d = 1
axiom T_divisible_by_15 : 15 ∣ T
lemma T_sum_of_digits_mul_3 : (∑ d in (T.digits 10), d) % 3 = 0 := sorry
lemma T_ends_with_0 : T.digits 10 |> List.head = some 0 := sorry

theorem smallest_X_value : ∃ X : ℕ, X = T / 15 ∧ X = 7400 := by
  use 7400
  split
  · calc 7400 = T / 15
    · rw [T]
    · exact div_eq_of_eq_mul_right (show 15 ≠ 0 from by norm_num) rfl
  · exact rfl

end smallest_X_value_l159_159619


namespace right_triangle_hypotenuse_segment_ratio_l159_159155

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ) (AB BC AC BD AD CD : ℝ)
  (h1 : AB = 4 * x) 
  (h2 : BC = 3 * x) 
  (h3 : AC = 5 * x) 
  (h4 : (BD ^ 2) = AD * CD) :
  (CD / AD) = (16 / 9) :=
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l159_159155


namespace xy_value_x2_y2_value_l159_159019

noncomputable def x : ℝ := Real.sqrt 7 + Real.sqrt 3
noncomputable def y : ℝ := Real.sqrt 7 - Real.sqrt 3

theorem xy_value : x * y = 4 := by
  -- proof goes here
  sorry

theorem x2_y2_value : x^2 + y^2 = 20 := by
  -- proof goes here
  sorry

end xy_value_x2_y2_value_l159_159019


namespace no_perfect_square_in_seq_l159_159444

noncomputable def seq : ℕ → ℕ
| 0       => 2
| 1       => 7
| (n + 2) => 4 * seq (n + 1) - seq n

theorem no_perfect_square_in_seq :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), (seq n) = k * k :=
sorry

end no_perfect_square_in_seq_l159_159444


namespace largest_three_digit_multiple_of_8_and_sum_24_is_888_l159_159072

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end largest_three_digit_multiple_of_8_and_sum_24_is_888_l159_159072


namespace smallest_X_divisible_15_l159_159617

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l159_159617


namespace fifth_friend_paid_l159_159562

theorem fifth_friend_paid (a b c d e : ℝ)
  (h1 : a = (1/3) * (b + c + d + e))
  (h2 : b = (1/4) * (a + c + d + e))
  (h3 : c = (1/5) * (a + b + d + e))
  (h4 : a + b + c + d + e = 120) :
  e = 40 :=
sorry

end fifth_friend_paid_l159_159562


namespace word_limit_correct_l159_159326

-- Definition for the conditions
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650
def exceeded_amount : ℕ := 100

-- The total words written
def total_words : ℕ := saturday_words + sunday_words

-- The word limit which we need to prove
def word_limit : ℕ := total_words - exceeded_amount

theorem word_limit_correct : word_limit = 1000 := by
  unfold word_limit total_words saturday_words sunday_words exceeded_amount
  sorry

end word_limit_correct_l159_159326


namespace evaluate_A_minus10_3_l159_159861

def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem evaluate_A_minus10_3 : A (-10) 3 = 1320 := 
  sorry

end evaluate_A_minus10_3_l159_159861


namespace lemon_pie_degrees_l159_159914

def total_students : ℕ := 45
def chocolate_pie_students : ℕ := 15
def apple_pie_students : ℕ := 10
def blueberry_pie_students : ℕ := 7
def cherry_and_lemon_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
def lemon_pie_students := cherry_and_lemon_students / 2

theorem lemon_pie_degrees (students_nonnegative : lemon_pie_students ≥ 0) (students_rounding : lemon_pie_students = 7) :
  (lemon_pie_students * 360 / total_students) = 56 := 
by
  -- Proof to be provided
  sorry

end lemon_pie_degrees_l159_159914


namespace smallest_possible_X_l159_159621

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l159_159621


namespace number_of_students_l159_159315

theorem number_of_students (x : ℕ) (total_cards : ℕ) (h : x * (x - 1) = total_cards) (h_total : total_cards = 182) : x = 14 :=
by
  sorry

end number_of_students_l159_159315


namespace smallest_integer_with_eight_factors_l159_159513

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l159_159513


namespace find_pool_depth_l159_159047

noncomputable def pool_depth (rate volume capacity_percent time length width : ℝ) :=
  volume / (length * width * rate * time / capacity_percent)

theorem find_pool_depth :
  pool_depth 60 75000 0.8 1000 150 50 = 10 := by
  simp [pool_depth] -- Simplifying the complex expression should lead to the solution.
  sorry

end find_pool_depth_l159_159047


namespace pyramid_height_l159_159888

theorem pyramid_height (lateral_edge : ℝ) (h : ℝ) (equilateral_angles : ℝ × ℝ × ℝ) (lateral_edge_length : lateral_edge = 3)
  (lateral_faces_are_equilateral : equilateral_angles = (60, 60, 60)) :
  h = 3 / 4 := by
  sorry

end pyramid_height_l159_159888


namespace problem1_problem2_l159_159009

variables (a b c d e f : ℝ)

-- Define the probabilities and the sum condition
def total_probability (a b c d e f : ℝ) : Prop := a + b + c + d + e + f = 1

-- Define P and Q
def P (a b c d e f : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + e^2 + f^2
def Q (a b c d e f : ℝ) : ℝ := (a + c + e) * (b + d + f)

-- Problem 1
theorem problem1 (h : total_probability a b c d e f) : P a b c d e f ≥ 1/6 := sorry

-- Problem 2
theorem problem2 (h : total_probability a b c d e f) : 
  1/4 ≥ Q a b c d e f ∧ Q a b c d e f ≥ 1/2 - 3/2 * P a b c d e f := sorry

end problem1_problem2_l159_159009


namespace inequality_solution_set_empty_l159_159418

theorem inequality_solution_set_empty (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)) → a ≤ 5 :=
by sorry

end inequality_solution_set_empty_l159_159418


namespace find_abc_l159_159579

theorem find_abc (a b c : ℕ) (k : ℕ) 
  (h1 : a = 2 * k) 
  (h2 : b = 3 * k) 
  (h3 : c = 4 * k) 
  (h4 : k ≠ 0)
  (h5 : 2 * a - b + c = 10) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
sorry

end find_abc_l159_159579


namespace converse_l159_159049

variables {x : ℝ}

def P (x : ℝ) : Prop := x < 0
def Q (x : ℝ) : Prop := x^2 > 0

theorem converse (h : Q x) : P x :=
sorry

end converse_l159_159049


namespace FG_square_l159_159279

def trapezoid_EFGH (EF FG GH EH : ℝ) : Prop :=
  ∃ x y : ℝ, 
  EF = 4 ∧
  EH = 31 ∧
  FG = x ∧
  GH = y ∧
  x^2 + (y - 4)^2 = 961 ∧
  x^2 = 4 * y

theorem FG_square (EF EH FG GH x y : ℝ) (h : trapezoid_EFGH EF FG GH EH) :
  FG^2 = 132 :=
by
  obtain ⟨x, y, h1, h2, h3, h4, h5, h6⟩ := h
  exact sorry

end FG_square_l159_159279


namespace xy_sum_cases_l159_159588

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l159_159588


namespace team_A_champion_probability_l159_159474

/-- Teams A and B are playing a volleyball match.
Team A needs to win one more game to become the champion, while Team B needs to win two more games to become the champion.
The probability of each team winning each game is 0.5. -/
theorem team_A_champion_probability :
  let p_win := (0.5 : ℝ)
  let prob_A_champion := 1 - p_win * p_win
  prob_A_champion = 0.75 := by
  sorry

end team_A_champion_probability_l159_159474


namespace number_of_outcomes_l159_159839

-- Define the conditions
def students : Nat := 4
def events : Nat := 3

-- Define the problem: number of possible outcomes for the champions
theorem number_of_outcomes : students ^ events = 64 :=
by sorry

end number_of_outcomes_l159_159839


namespace slope_of_asymptotes_l159_159115

theorem slope_of_asymptotes (a b : ℝ) (h : a^2 = 144) (k : b^2 = 81) : (b / a = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l159_159115


namespace equivalent_single_discount_l159_159321

variable (x : ℝ)
variable (original_price : ℝ := x)
variable (discount1 : ℝ := 0.15)
variable (discount2 : ℝ := 0.10)
variable (discount3 : ℝ := 0.05)

theorem equivalent_single_discount :
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let equivalent_discount := (1 - final_price / original_price)
  equivalent_discount = 0.27 := 
by 
  sorry

end equivalent_single_discount_l159_159321


namespace necessary_not_sufficient_l159_159753

-- Define the function y = x^2 - 2ax + 1
def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define strict monotonicity on the interval [1, +∞)
def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Define the condition for the function to be strictly increasing on [1, +∞)
def condition_strict_increasing (a : ℝ) : Prop :=
  strictly_increasing_on (quadratic_function a) (Set.Ici 1)

-- The condition to prove
theorem necessary_not_sufficient (a : ℝ) :
  condition_strict_increasing a → (a ≤ 0) := sorry

end necessary_not_sufficient_l159_159753


namespace multiples_of_7_not_14_l159_159254

theorem multiples_of_7_not_14 :
  { n : ℕ | n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0 }.card = 36 := by
  sorry

end multiples_of_7_not_14_l159_159254


namespace parallel_line_dividing_triangle_l159_159853

theorem parallel_line_dividing_triangle (base : ℝ) (length_parallel_line : ℝ) 
    (h_base : base = 24) 
    (h_parallel : (length_parallel_line / base)^2 = 1/2) : 
    length_parallel_line = 12 * Real.sqrt 2 :=
sorry

end parallel_line_dividing_triangle_l159_159853


namespace shifted_graph_sum_l159_159992

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 5

def shift_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)
def shift_up (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f x + k

noncomputable def g (x : ℝ) : ℝ := shift_up (shift_right f 7) 3 x

theorem shifted_graph_sum : (∃ (a b c : ℝ), g x = a * x ^ 2 + b * x + c ∧ (a + b + c = 128)) :=
by
  sorry

end shifted_graph_sum_l159_159992


namespace rattlesnakes_count_l159_159652

theorem rattlesnakes_count (P B R V : ℕ) (h1 : P = 3 * B / 2) (h2 : V = 2 * 420 / 100) (h3 : P + R = 3 * 420 / 4) (h4 : P + B + R + V = 420) : R = 162 :=
by
  sorry

end rattlesnakes_count_l159_159652


namespace vasya_average_not_exceed_4_l159_159020

variable (a b c d e : ℕ) 

-- Total number of grades
def total_grades : ℕ := a + b + c + d + e

-- Initial average condition
def initial_condition : Prop := 
  (a + 2 * b + 3 * c + 4 * d + 5 * e) < 3 * (total_grades a b c d e)

-- New average condition after grade changes
def changed_average (a b c d e : ℕ) : ℚ := 
  ((2 * b + 3 * (a + c) + 4 * d + 5 * e) : ℚ) / (total_grades a b c d e)

-- Proof problem to show the new average grade does not exceed 4
theorem vasya_average_not_exceed_4 (h : initial_condition a b c d e) : 
  (changed_average 0 b (c + a) d e) ≤ 4 := 
sorry

end vasya_average_not_exceed_4_l159_159020


namespace stock_price_no_return_l159_159648

/-- Define the increase and decrease factors. --/
def increase_factor := 117 / 100
def decrease_factor := 83 / 100

/-- Define the proof that the stock price cannot return to its initial value after any number of 
    increases and decreases. --/
theorem stock_price_no_return 
  (P0 : ℝ) (k l : ℕ) : 
  P0 * (increase_factor ^ k) * (decrease_factor ^ l) ≠ P0 :=
by
  sorry

end stock_price_no_return_l159_159648


namespace inverse_passes_through_3_4_l159_159404

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given that f(x) has an inverse
def has_inverse := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Given that y = f(x+1) passes through the point (3,3)
def condition := f (3 + 1) = 3

theorem inverse_passes_through_3_4 
  (h1 : has_inverse f) 
  (h2 : condition f) : 
  f⁻¹ 3 = 4 :=
sorry

end inverse_passes_through_3_4_l159_159404


namespace num_integers_satisfy_inequality_l159_159581

theorem num_integers_satisfy_inequality : ∃ (s : Finset ℤ), (∀ x ∈ s, |7 * x - 5| ≤ 15) ∧ s.card = 5 :=
by
  sorry

end num_integers_satisfy_inequality_l159_159581


namespace rhombus_perimeter_l159_159958

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l159_159958


namespace problem_is_happy_number_512_l159_159909

/-- A number is a "happy number" if it is the square difference of two consecutive odd numbers. -/
def is_happy_number (x : ℕ) : Prop :=
  ∃ n : ℤ, x = 8 * n

/-- The number 512 is a "happy number". -/
theorem problem_is_happy_number_512 : is_happy_number 512 :=
  sorry

end problem_is_happy_number_512_l159_159909


namespace age_of_other_replaced_man_l159_159302

variable (A B C : ℕ)
variable (B_new1 B_new2 : ℕ)
variable (avg_old avg_new : ℕ)

theorem age_of_other_replaced_man (hB : B = 23) 
    (h_avg_new : (B_new1 + B_new2) / 2 = 25)
    (h_avg_inc : (A + B_new1 + B_new2) / 3 > (A + B + C) / 3) : 
    C = 26 := 
  sorry

end age_of_other_replaced_man_l159_159302


namespace hypotenuse_of_right_angle_triangle_l159_159575

theorem hypotenuse_of_right_angle_triangle {a b c : ℕ} (h1 : a^2 + b^2 = c^2) 
  (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b + c = (a * b) / 2): 
  c = 10 ∨ c = 13 :=
sorry

end hypotenuse_of_right_angle_triangle_l159_159575


namespace pyramid_height_eq_cube_volume_l159_159203

theorem pyramid_height_eq_cube_volume (h : ℝ) : 
  let cube_volume := 6^3 in
  let base_area := 10^2 in
  let pyramid_volume := (1 / 3) * base_area * h in
  cube_volume = pyramid_volume → 
  h = 6.48 :=
by
  intros
  sorry

end pyramid_height_eq_cube_volume_l159_159203


namespace porch_length_is_6_l159_159641

-- Define the conditions for the house and porch areas
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_width : ℝ := 4.5
def total_shingle_area : ℝ := 232

-- Define the area calculations
def house_area : ℝ := house_length * house_width
def porch_area : ℝ := total_shingle_area - house_area

-- The theorem to prove
theorem porch_length_is_6 : porch_area / porch_width = 6 := by
  sorry

end porch_length_is_6_l159_159641


namespace basketball_game_l159_159608

theorem basketball_game (a r b d : ℕ) (r_gt_1 : r > 1) (d_gt_0 : d > 0)
  (H1 : a = b)
  (H2 : a * (1 + r) * (1 + r^2) = 4 * b + 6 * d + 2)
  (H3 : a * (1 + r) * (1 + r^2) ≤ 100)
  (H4 : 4 * b + 6 * d ≤ 98) :
  (a + a * r) + (b + (b + d)) = 43 := 
sorry

end basketball_game_l159_159608


namespace smallest_possible_X_l159_159622

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l159_159622


namespace smallest_angle_equilateral_triangle_l159_159756

-- Definitions corresponding to the conditions
structure EquilateralTriangle :=
(vertices : Fin 3 → ℝ × ℝ)
(equilateral : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))

def point_on_line_segment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
((1 - t) * p1.1 + t * p2.1, (1 - t) * p1.2 + t * p2.2)

-- Given an equilateral triangle ABC with vertices A, B, C,
-- and points D on AB, E on AC, D1 on BC, and E1 on BC,
-- such that AB = DB + BD_1 and AC = CE + CE_1,
-- prove the smallest angle between DE_1 and ED_1 is 60 degrees.

theorem smallest_angle_equilateral_triangle
  (ABC : EquilateralTriangle)
  (A B C D E D₁ E₁ : ℝ × ℝ)
  (on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = point_on_line_segment A B t)
  (on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = point_on_line_segment A C t)
  (on_BC : ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ D₁ = point_on_line_segment B C t₁ ∧
                         0 ≤ t₂ ∧ t₂ ≤ 1 ∧ E₁ = point_on_line_segment B C t₂)
  (AB_property : dist A B = dist D B + dist B D₁)
  (AC_property : dist A C = dist E C + dist C E₁) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 60 ∧ θ = 60 :=
sorry

end smallest_angle_equilateral_triangle_l159_159756


namespace who_threw_at_third_child_l159_159628

-- Definitions based on conditions
def children_count : ℕ := 43

def threw_snowball (i j : ℕ) : Prop :=
∃ k, i = (k % children_count).succ ∧ j = ((k + 1) % children_count).succ

-- Conditions
axiom cond_1 : threw_snowball 1 (1 + 1) -- child 1 threw a snowball at the child who threw a snowball at child 2
axiom cond_2 : threw_snowball 2 (2 + 1) -- child 2 threw a snowball at the child who threw a snowball at child 3
axiom cond_3 : threw_snowball 43 1 -- child 43 threw a snowball at the child who threw a snowball at the first child

-- Question to prove
theorem who_threw_at_third_child : threw_snowball 24 3 :=
sorry

end who_threw_at_third_child_l159_159628


namespace cookies_per_batch_l159_159289

theorem cookies_per_batch (students : ℕ) (cookies_per_student : ℕ) (chocolate_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) (cookies_needed : ℕ) (dozens_per_batch : ℕ) :
  (students = 24) →
  (cookies_per_student = 10) →
  (chocolate_batches = 2) →
  (oatmeal_batches = 1) →
  (additional_batches = 2) →
  (cookies_needed = students * cookies_per_student) →
  dozens_per_batch * (12 * (chocolate_batches + oatmeal_batches + additional_batches)) = cookies_needed →
  dozens_per_batch = 4 :=
by
  intros
  sorry

end cookies_per_batch_l159_159289


namespace calculate_expression_l159_159103

theorem calculate_expression : 
  2 - 1 / (2 - 1 / (2 + 2)) = 10 / 7 := 
by sorry

end calculate_expression_l159_159103


namespace least_subset_gcd_l159_159625

variable (S : Set ℕ) (f : ℕ → ℤ)
variable (a : ℕ → ℕ)
variable (k : ℕ)

def conditions (S : Set ℕ) (f : ℕ → ℤ) : Prop :=
  ∃ (a : ℕ → ℕ), 
  (∀ i j, i ≠ j → a i < a j) ∧ 
  (S = {i | ∃ n, i = a n ∧ n < 2004}) ∧ 
  (∀ i, f (a i) < 2003) ∧ 
  (∀ i j, f (a i) = f (a j))

theorem least_subset_gcd (h : conditions S f) : k = 1003 :=
  sorry

end least_subset_gcd_l159_159625


namespace total_rainfall_correct_l159_159546

-- Define the individual rainfall amounts
def rainfall_mon1 : ℝ := 0.17
def rainfall_wed1 : ℝ := 0.42
def rainfall_fri : ℝ := 0.08
def rainfall_mon2 : ℝ := 0.37
def rainfall_wed2 : ℝ := 0.51

-- Define the total rainfall
def total_rainfall : ℝ := rainfall_mon1 + rainfall_wed1 + rainfall_fri + rainfall_mon2 + rainfall_wed2

-- Theorem statement to prove the total rainfall is 1.55 cm
theorem total_rainfall_correct : total_rainfall = 1.55 :=
by
  -- Proof goes here
  sorry

end total_rainfall_correct_l159_159546


namespace total_cows_l159_159156

variable (D C : ℕ)

-- The conditions of the problem translated to Lean definitions
def total_heads := D + C
def total_legs := 2 * D + 4 * C 

-- The main theorem based on the conditions and the result to prove
theorem total_cows (h1 : total_legs D C = 2 * total_heads D C + 40) : C = 20 :=
by
  sorry


end total_cows_l159_159156


namespace group_discount_l159_159198

theorem group_discount (P : ℝ) (D : ℝ) :
  4 * (P - (D / 100) * P) = 3 * P → D = 25 :=
by
  intro h
  sorry

end group_discount_l159_159198


namespace whole_milk_fat_percentage_l159_159676

def fat_in_some_milk : ℝ := 4
def percentage_less : ℝ := 0.5

theorem whole_milk_fat_percentage : ∃ (x : ℝ), fat_in_some_milk = percentage_less * x ∧ x = 8 :=
sorry

end whole_milk_fat_percentage_l159_159676


namespace time_to_fill_cistern_l159_159984

def pipe_p_rate := (1: ℚ) / 10
def pipe_q_rate := (1: ℚ) / 15
def pipe_r_rate := - (1: ℚ) / 30
def combined_rate_p_q := pipe_p_rate + pipe_q_rate
def combined_rate_q_r := pipe_q_rate + pipe_r_rate
def initial_fill := 2 * combined_rate_p_q
def remaining_fill := 1 - initial_fill
def remaining_time := remaining_fill / combined_rate_q_r

theorem time_to_fill_cistern :
  remaining_time = 20 := by sorry

end time_to_fill_cistern_l159_159984


namespace sandbox_area_l159_159848

def length : ℕ := 312
def width : ℕ := 146
def area : ℕ := 45552

theorem sandbox_area : length * width = area := by
  sorry

end sandbox_area_l159_159848


namespace olympiad_scores_above_18_l159_159923

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l159_159923


namespace arithmetic_sequence_solution_l159_159244

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 + a 4 = 4)
  (h2 : a 2 * a 3 = 3)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2):
  (a 1 = -1 ∧ (∀ n, a n = 2 * n - 3) ∧ (∀ n, S n = n^2 - 2 * n)) ∨ 
  (a 1 = 5 ∧ (∀ n, a n = 7 - 2 * n) ∧ (∀ n, S n = 6 * n - n^2)) :=
sorry

end arithmetic_sequence_solution_l159_159244


namespace smallest_integer_with_eight_factors_l159_159502

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l159_159502


namespace original_number_is_500_l159_159087

theorem original_number_is_500 (x : ℝ) (h1 : x * 1.3 = 650) : x = 500 :=
sorry

end original_number_is_500_l159_159087


namespace maximum_possible_savings_is_63_l159_159184

-- Definitions of the conditions
def doughnut_price := 8
def doughnut_discount_2 := 14
def doughnut_discount_4 := 26

def croissant_price := 10
def croissant_discount_3 := 28
def croissant_discount_5 := 45

def muffin_price := 6
def muffin_discount_2 := 11
def muffin_discount_6 := 30

-- Quantities to purchase
def doughnut_qty := 20
def croissant_qty := 15
def muffin_qty := 18

-- Prices calculated from quantities
def total_price_without_discount :=
  doughnut_qty * doughnut_price + croissant_qty * croissant_price + muffin_qty * muffin_price

def total_price_with_discount :=
  5 * doughnut_discount_4 + 3 * croissant_discount_5 + 3 * muffin_discount_6

def maximum_savings := total_price_without_discount - total_price_with_discount

theorem maximum_possible_savings_is_63 : maximum_savings = 63 := by
  -- Proof to be filled in
  sorry

end maximum_possible_savings_is_63_l159_159184


namespace fraction_equivalence_l159_159557

-- Given fractions
def frac1 : ℚ := 3 / 7
def frac2 : ℚ := 4 / 5
def frac3 : ℚ := 5 / 12
def frac4 : ℚ := 2 / 9

-- Expectation
def result : ℚ := 1548 / 805

-- Theorem to prove the equality
theorem fraction_equivalence : ((frac1 + frac2) / (frac3 + frac4)) = result := by
  sorry

end fraction_equivalence_l159_159557


namespace triple_layer_area_l159_159495

theorem triple_layer_area (A B C X Y : ℕ) 
  (h1 : A + B + C = 204) 
  (h2 : 140 = (A + B + C) - X - 2 * Y + X + Y)
  (h3 : X = 24) : 
  Y = 64 := by
  sorry

end triple_layer_area_l159_159495


namespace cosine_of_angle_between_vectors_l159_159701

theorem cosine_of_angle_between_vectors (a1 b1 c1 a2 b2 c2 : ℝ) :
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  dot_product / (magnitude_u * magnitude_v) = 
      (a1 * a2 + b1 * b2 + c1 * c2) / (Real.sqrt (a1^2 + b1^2 + c1^2) * Real.sqrt (a2^2 + b2^2 + c2^2)) :=
by
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  sorry

end cosine_of_angle_between_vectors_l159_159701


namespace sequence_formula_l159_159685

noncomputable def a : ℕ → ℕ
| 0       => 2
| (n + 1) => a n ^ 2 - n * a n + 1

theorem sequence_formula (n : ℕ) : a n = n + 2 :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end sequence_formula_l159_159685


namespace base8_subtraction_correct_l159_159810

def base8_sub (a b : Nat) : Nat := sorry  -- function to perform base 8 subtraction

theorem base8_subtraction_correct :
  base8_sub 0o126 0o45 = 0o41 := sorry

end base8_subtraction_correct_l159_159810


namespace no_solution_for_n_eq_neg2_l159_159236

theorem no_solution_for_n_eq_neg2 : ∀ (x y : ℝ), ¬ (2 * x = 1 + -2 * y ∧ -2 * x = 1 + 2 * y) :=
by sorry

end no_solution_for_n_eq_neg2_l159_159236


namespace fundraiser_brownies_l159_159563

-- Definitions derived from the conditions in the problem statement
def brownie_price := 2
def cookie_price := 2
def donut_price := 2

def students_bringing_brownies (B : Nat) := B
def students_bringing_cookies := 20
def students_bringing_donuts := 15

def brownies_per_student := 12
def cookies_per_student := 24
def donuts_per_student := 12

def total_amount_raised := 2040

theorem fundraiser_brownies (B : Nat) :
  24 * B + 20 * 24 * 2 + 15 * 12 * 2 = total_amount_raised → B = 30 :=
by
  sorry

end fundraiser_brownies_l159_159563


namespace words_to_numbers_l159_159193

def word_to_num (w : String) : Float := sorry

theorem words_to_numbers :
  word_to_num "fifty point zero zero one" = 50.001 ∧
  word_to_num "seventy-five point zero six" = 75.06 :=
by
  sorry

end words_to_numbers_l159_159193


namespace quadrilateral_trapezium_l159_159580

theorem quadrilateral_trapezium (a b c d : ℝ) 
  (h1 : a / 6 = b / 7) 
  (h2 : b / 7 = c / 8) 
  (h3 : c / 8 = d / 9) 
  (h4 : a + b + c + d = 360) : 
  ((a + c = 180) ∨ (b + d = 180)) :=
by
  sorry

end quadrilateral_trapezium_l159_159580


namespace ellipse_non_degenerate_l159_159553

noncomputable def non_degenerate_ellipse_condition (b : ℝ) : Prop := b > -13

theorem ellipse_non_degenerate (b : ℝ) :
  (∃ x y : ℝ, 4*x^2 + 9*y^2 - 16*x + 18*y + 12 = b) → non_degenerate_ellipse_condition b :=
by
  sorry

end ellipse_non_degenerate_l159_159553


namespace quadratic_inequality_solution_non_empty_l159_159025

theorem quadratic_inequality_solution_non_empty
  (a b c : ℝ) (h : a < 0) :
  ∃ x : ℝ, ax^2 + bx + c < 0 :=
sorry

end quadratic_inequality_solution_non_empty_l159_159025


namespace inequality_chain_l159_159568

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem inequality_chain (a b : ℝ) (h1 : b > a) (h2 : a > 3) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f a :=
by
  sorry

end inequality_chain_l159_159568


namespace percentage_of_water_in_mixture_is_17_14_l159_159080

def Liquid_A_water_percentage : ℝ := 0.10
def Liquid_B_water_percentage : ℝ := 0.15
def Liquid_C_water_percentage : ℝ := 0.25
def Liquid_D_water_percentage : ℝ := 0.35

def parts_A : ℝ := 3
def parts_B : ℝ := 2
def parts_C : ℝ := 1
def parts_D : ℝ := 1

def part_unit : ℝ := 100

noncomputable def total_units : ℝ := 
  parts_A * part_unit + parts_B * part_unit + parts_C * part_unit + parts_D * part_unit

noncomputable def total_water_units : ℝ :=
  parts_A * part_unit * Liquid_A_water_percentage +
  parts_B * part_unit * Liquid_B_water_percentage +
  parts_C * part_unit * Liquid_C_water_percentage +
  parts_D * part_unit * Liquid_D_water_percentage

noncomputable def percentage_water : ℝ := (total_water_units / total_units) * 100

theorem percentage_of_water_in_mixture_is_17_14 :
  percentage_water = 17.14 := sorry

end percentage_of_water_in_mixture_is_17_14_l159_159080


namespace geometric_sequence_general_formula_arithmetic_sequence_sum_l159_159569

-- Problem (I)
theorem geometric_sequence_general_formula (a : ℕ → ℝ) (q a1 : ℝ)
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : a 1 + a 2 = 6)
  (h3 : a 1 * a 2 = a 3) :
  a n = 2 ^ n :=
sorry

-- Problem (II)
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (h1 : ∀ n, a n = 2 ^ n)
  (h2 : ∀ n, S n = (n * (b 1 + b n)) / 2)
  (h3 : ∀ n, S (2 * n + 1) = b n * b (n + 1))
  (h4 : ∀ n, b n = 2 * n + 1) :
  T n = 5 - (2 * n + 5) / 2 ^ n :=
sorry

end geometric_sequence_general_formula_arithmetic_sequence_sum_l159_159569


namespace smallest_X_divisible_15_l159_159618

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l159_159618


namespace smallest_integer_with_eight_factors_l159_159503

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l159_159503


namespace cube_pyramid_volume_l159_159202

theorem cube_pyramid_volume (s b h : ℝ) 
  (hcube : s = 6) 
  (hbase : b = 10)
  (eq_volumes : (s ^ 3) = (1 / 3) * (b ^ 2) * h) : 
  h = 162 / 25 := 
by 
  sorry

end cube_pyramid_volume_l159_159202


namespace quadratic_always_positive_l159_159700

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - k + 4 > 0) ↔ -2 * Real.sqrt 3 < k ∧ k < 2 * Real.sqrt 3 := by
  sorry

end quadratic_always_positive_l159_159700


namespace inequality_solution_l159_159122

theorem inequality_solution (x : ℝ) (h : |(x + 4) / 2| < 3) : -10 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l159_159122


namespace rhombus_perimeter_l159_159965

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : 
  let side := (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in 
  4 * side = 8 * real.sqrt 41 := by
  sorry

end rhombus_perimeter_l159_159965


namespace necessary_but_not_sufficient_condition_l159_159881

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  ((a > 2) ∧ (b > 2) → (a + b > 4)) ∧ ¬((a + b > 4) → (a > 2) ∧ (b > 2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l159_159881


namespace hyperbola_focus_distance_l159_159894

theorem hyperbola_focus_distance :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 3 = 1) → ∀ (F₁ F₂ : ℝ × ℝ), ∃ P : ℝ × ℝ, dist P F₁ = 3 → dist P F₂ = 7 :=
by
  sorry

end hyperbola_focus_distance_l159_159894


namespace probability_two_red_two_blue_correct_l159_159674

noncomputable def num_ways_to_choose : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

noncomputable def probability_two_red_two_blue : ℚ :=
  let total_ways := num_ways_to_choose 20 4
  let ways_red := num_ways_to_choose 12 2
  let ways_blue := num_ways_to_choose 8 2
  (ways_red * ways_blue) / total_ways

theorem probability_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 :=
by
  sorry

end probability_two_red_two_blue_correct_l159_159674


namespace smallest_N_is_14_l159_159966

-- Definition of depicted number and cyclic arrangement
def depicted_number : Type := List (Fin 2) -- Depicted numbers are lists of digits (0 corresponds to 1, 1 corresponds to 2)

-- A condition representing the function that checks if a list contains all possible four-digit combinations
def contains_all_four_digit_combinations (arr: List (Fin 2)) : Prop :=
  ∀ (seq: List (Fin 2)), seq.length = 4 → seq ⊆ arr

-- The problem statement: find the smallest N where an arrangement contains all four-digit combinations
def smallest_N (N: Nat) (arr: List (Fin 2)) : Prop :=
  N = arr.length ∧ contains_all_four_digit_combinations arr

theorem smallest_N_is_14 : ∃ (N : Nat) (arr: List (Fin 2)), smallest_N N arr ∧ N = 14 :=
by
  -- Placeholder for the proof
  sorry

end smallest_N_is_14_l159_159966


namespace inequality_proof_l159_159448

open Real

-- Define the conditions
def conditions (a b c : ℝ) := (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b * c = 1)

-- Express the inequality we need to prove
def inequality (a b c : ℝ) :=
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1

-- Statement of the theorem
theorem inequality_proof (a b c : ℝ) (h : conditions a b c) : inequality a b c :=
by {
  sorry
}

end inequality_proof_l159_159448


namespace staircase_steps_l159_159725

theorem staircase_steps (x : ℕ) :
  x % 2 = 1 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 6 = 5 ∧
  x % 7 = 0 → 
  x ≡ 119 [MOD 420] :=
by
  sorry

end staircase_steps_l159_159725


namespace price_of_uniform_l159_159531

-- Definitions based on conditions
def total_salary : ℕ := 600
def months_worked : ℕ := 9
def months_in_year : ℕ := 12
def salary_received : ℕ := 400
def uniform_price (U : ℕ) : Prop := 
    (3/4 * total_salary) - salary_received = U

-- Theorem stating the price of the uniform
theorem price_of_uniform : ∃ U : ℕ, uniform_price U := by
  sorry

end price_of_uniform_l159_159531


namespace sum_of_fractions_l159_159378

theorem sum_of_fractions : 
  (2/100) + (5/1000) + (5/10000) + 3 * (4/1000) = 0.0375 := 
by 
  sorry

end sum_of_fractions_l159_159378


namespace sum_nth_beginning_end_l159_159990

theorem sum_nth_beginning_end (n : ℕ) (F L : ℤ) (M : ℤ) 
  (consecutive : ℤ → ℤ) (median : M = 60) 
  (median_formula : M = (F + L) / 2) :
  n = n → F + L = 120 :=
by
  sorry

end sum_nth_beginning_end_l159_159990


namespace total_distance_traveled_l159_159942

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l159_159942


namespace point_A_coordinates_l159_159306

noncomputable def f (a x : ℝ) : ℝ := a * x - 1

theorem point_A_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 :=
sorry

end point_A_coordinates_l159_159306


namespace smallest_possible_value_l159_159281

theorem smallest_possible_value (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : n ≡ 2 [MOD 9]) (h3 : n ≡ 6 [MOD 7]) :
  n = 116 :=
by
  -- Proof omitted
  sorry

end smallest_possible_value_l159_159281


namespace coin_count_l159_159183

-- Define the conditions and the proof goal
theorem coin_count (total_value : ℕ) (coin_value_20 : ℕ) (coin_value_25 : ℕ) 
    (num_20_paise_coins : ℕ) (total_value_paise : total_value = 7100)
    (value_20_paise : coin_value_20 = 20) (value_25_paise : coin_value_25 = 25)
    (num_20_paise : num_20_paise_coins = 300) : 
    (300 + 44 = 344) :=
by
  -- The proof would go here, currently omitted with sorry
  sorry

end coin_count_l159_159183


namespace number_of_new_bricks_l159_159843

-- Definitions from conditions
def edge_length_original_brick : ℝ := 0.3
def edge_length_new_brick : ℝ := 0.5
def number_original_bricks : ℕ := 600

-- The classroom volume is unchanged, so we set up a proportion problem
-- Assuming the classroom is fully paved
theorem number_of_new_bricks :
  let volume_original_bricks := number_original_bricks * (edge_length_original_brick ^ 2)
  let volume_new_bricks := x * (edge_length_new_brick ^ 2)
  volume_original_bricks = volume_new_bricks → x = 216 := 
by
  sorry

end number_of_new_bricks_l159_159843


namespace problem_a_lt_zero_b_lt_neg_one_l159_159708

theorem problem_a_lt_zero_b_lt_neg_one (a b : ℝ) (ha : a < 0) (hb : b < -1) : 
  ab > a ∧ a > ab^2 := 
by
  sorry

end problem_a_lt_zero_b_lt_neg_one_l159_159708


namespace sequence_to_one_l159_159636

def nextStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n - 1

theorem sequence_to_one (n : ℕ) (h : n > 0) :
  ∃ seq : ℕ → ℕ, seq 0 = n ∧ (∀ i, seq (i + 1) = nextStep (seq i)) ∧ (∃ j, seq j = 1) := by
  sorry

end sequence_to_one_l159_159636


namespace distinct_real_roots_implies_positive_l159_159736

theorem distinct_real_roots_implies_positive (k : ℝ) (x1 x2 : ℝ) (h_distinct : x1 ≠ x2) 
  (h_root1 : x1^2 + 2*x1 - k = 0) 
  (h_root2 : x2^2 + 2*x2 - k = 0) : 
  x1^2 + x2^2 - 2 > 0 := 
sorry

end distinct_real_roots_implies_positive_l159_159736


namespace vasya_birthday_l159_159798

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159798


namespace inequality_implies_bounds_l159_159240

open Real

theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, (exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → (0 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_implies_bounds_l159_159240


namespace area_of_closed_figure_l159_159307

noncomputable def f : ℝ → ℝ := λ x, sin (2 * x) - sqrt 3 * cos (2 * x)

noncomputable def g : ℝ → ℝ := λ x, 2 * sin (2 * x)

theorem area_of_closed_figure : 
  ∫ x in 0..(π / 3), g x = 3 / 2 := by
  sorry

end area_of_closed_figure_l159_159307


namespace num_gcd_values_l159_159827

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159827


namespace probability_of_prime_sum_l159_159065

def is_prime_sum (a b c : ℕ) : Prop :=
  Nat.Prime (a + b + c)

def valid_die_roll (n : ℕ) : Prop :=
  n >= 1 ∧ n <= 6

noncomputable def prime_probability : ℚ :=
  let outcomes := [(a, b, c) | a in Fin 6, b in Fin 6, c in Fin 6, 1 ≤ a + 1, 1 ≤ b + 1, 1 ≤ c + 1]
  let prime_outcomes := outcomes.filter (λ (a, b, c), is_prime_sum (a + 1) (b + 1) (c + 1))
  prime_outcomes.length / outcomes.length

theorem probability_of_prime_sum :
  prime_probability = 37 / 216 := sorry

end probability_of_prime_sum_l159_159065


namespace calculate_expression_l159_159380

theorem calculate_expression :
  |(-1 : ℝ)| + Real.sqrt 9 - (1 - Real.sqrt 3)^0 - (1/2)^(-1 : ℝ) = 1 :=
by
  sorry

end calculate_expression_l159_159380


namespace toothpick_count_l159_159497

theorem toothpick_count (height width : ℕ) (h_height : height = 20) (h_width : width = 10) : 
  (21 * width + 11 * height) = 430 :=
by
  sorry

end toothpick_count_l159_159497


namespace probability_red_then_green_l159_159028

-- Total number of balls and their representation
def total_balls : ℕ := 3
def red_balls : ℕ := 2
def green_balls : ℕ := 1

-- The total number of outcomes when drawing two balls with replacement
def total_outcomes : ℕ := total_balls * total_balls

-- The desired outcomes: drawing a red ball first and a green ball second
def desired_outcomes : ℕ := 2 -- (1,3) and (2,3)

-- Calculating the probability of drawing a red ball first and a green ball second
def probability_drawing_red_then_green : ℚ := desired_outcomes / total_outcomes

-- The theorem we need to prove
theorem probability_red_then_green :
  probability_drawing_red_then_green = 2 / 9 :=
by 
  sorry

end probability_red_then_green_l159_159028


namespace club_positions_l159_159678

def num_ways_to_fill_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

theorem club_positions : num_ways_to_fill_positions 12 = 665280 := by 
  sorry

end club_positions_l159_159678


namespace limit_at_1_eq_one_half_l159_159101

noncomputable def limit_function (x : ℝ) : ℝ := (1 + Real.cos (Real.pi * x)) / (Real.tan (Real.pi * x))^2

theorem limit_at_1_eq_one_half :
  filter.tendsto limit_function (nhds 1) (nhds (1 / 2)) :=
begin
  sorry
end

end limit_at_1_eq_one_half_l159_159101


namespace benjamin_collects_6_dozen_eggs_l159_159372

theorem benjamin_collects_6_dozen_eggs (B : ℕ) (h : B + 3 * B + (B - 4) = 26) : B = 6 :=
by sorry

end benjamin_collects_6_dozen_eggs_l159_159372


namespace gcd_198_286_l159_159328

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end gcd_198_286_l159_159328


namespace problem_statement_l159_159286

noncomputable def r (a b : ℚ) : ℚ := 
  let ab := a * b
  let a_b_recip := a + (1/b)
  let b_a_recip := b + (1/a)
  a_b_recip * b_a_recip

theorem problem_statement (a b : ℚ) (m : ℚ) (h1 : a * b = 3) (h2 : ∃ p, (a + 1 / b) * (b + 1 / a) = (ab + 1 / ab + 2)) :
  r a b = 16 / 3 := by
  sorry

end problem_statement_l159_159286


namespace ezekiel_painted_faces_l159_159868

noncomputable def cuboid_faces_painted (num_cuboids : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
num_cuboids * faces_per_cuboid

theorem ezekiel_painted_faces :
  cuboid_faces_painted 8 6 = 48 := 
by
  sorry

end ezekiel_painted_faces_l159_159868


namespace johny_total_travel_distance_l159_159282

def TravelDistanceSouth : ℕ := 40
def TravelDistanceEast : ℕ := TravelDistanceSouth + 20
def TravelDistanceNorth : ℕ := 2 * TravelDistanceEast
def TravelDistanceWest : ℕ := TravelDistanceNorth / 2

theorem johny_total_travel_distance
    (hSouth : TravelDistanceSouth = 40)
    (hEast  : TravelDistanceEast = 60)
    (hNorth : TravelDistanceNorth = 120)
    (hWest  : TravelDistanceWest = 60)
    (totalDistance : ℕ := TravelDistanceSouth + TravelDistanceEast + TravelDistanceNorth + TravelDistanceWest) :
    totalDistance = 280 := by
  sorry

end johny_total_travel_distance_l159_159282


namespace min_races_needed_l159_159952

noncomputable def minimum_races (total_horses : ℕ) (max_race_horses : ℕ) : ℕ :=
  if total_horses ≤ max_race_horses then 1 else
  if total_horses % max_race_horses = 0 then total_horses / max_race_horses else total_horses / max_race_horses + 1

/-- We need to show that the minimum number of races required to find the top 3 fastest horses
    among 35 horses, where a maximum of 4 horses can race together at a time, is 10. -/
theorem min_races_needed : minimum_races 35 4 = 10 :=
  sorry

end min_races_needed_l159_159952


namespace slope_tangent_line_at_zero_l159_159314

noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) / (x^2 + 1)

theorem slope_tangent_line_at_zero : 
  (deriv f 0) = 2 :=
sorry

end slope_tangent_line_at_zero_l159_159314


namespace count_multiples_of_7_not_14_l159_159256

theorem count_multiples_of_7_not_14 (n : ℕ) : (n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) → ∃ (k : ℕ), k = 36 :=
by
  sorry

end count_multiples_of_7_not_14_l159_159256


namespace min_side_length_l159_159472

noncomputable def side_length_min : ℝ := 30

theorem min_side_length (s r : ℝ) (hs₁ : s^2 ≥ 900) (hr₁ : π * r^2 ≥ 100) (hr₂ : 2 * r ≤ s) :
  s ≥ side_length_min :=
by
  sorry

end min_side_length_l159_159472


namespace total_money_spent_l159_159441

def time_in_minutes_at_arcade : ℕ := 3 * 60
def cost_per_interval : ℕ := 50 -- in cents
def interval_duration : ℕ := 6 -- in minutes
def total_intervals : ℕ := time_in_minutes_at_arcade / interval_duration

theorem total_money_spent :
  ((total_intervals * cost_per_interval) = 1500) := 
by
  sorry

end total_money_spent_l159_159441


namespace polynomial_inequality_l159_159033

theorem polynomial_inequality
  (x1 x2 x3 a b c : ℝ)
  (h1 : x1 > 0) 
  (h2 : x2 > 0) 
  (h3 : x3 > 0)
  (h4 : x1 + x2 + x3 ≤ 1)
  (h5 : x1^3 + a * x1^2 + b * x1 + c = 0)
  (h6 : x2^3 + a * x2^2 + b * x2 + c = 0)
  (h7 : x3^3 + a * x3^2 + b * x3 + c = 0) :
  a^3 * (1 + a + b) - 9 * c * (3 + 3 * a + a^2) ≤ 0 :=
sorry

end polynomial_inequality_l159_159033


namespace arithmetic_sequence_problem_l159_159939

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Use the given specific conditions
theorem arithmetic_sequence_problem 
  (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 2 * a 3 = 21) : 
  a 1 * a 4 = -11 :=
sorry

end arithmetic_sequence_problem_l159_159939


namespace remove_one_to_get_average_of_75_l159_159515

theorem remove_one_to_get_average_of_75 : 
  ∃ l : List ℕ, l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] ∧ 
  (∃ m : ℕ, List.erase l m = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] : List ℕ) ∧ 
  (12 : ℕ) = List.length (List.erase l m) ∧
  7.5 = ((List.sum (List.erase l m) : ℚ) / 12)) :=
sorry

end remove_one_to_get_average_of_75_l159_159515


namespace triangle_inradius_l159_159746

theorem triangle_inradius (A s r : ℝ) (h₁ : A = 3 * s) (h₂ : A = r * s) (h₃ : s ≠ 0) : r = 3 :=
by
  -- Proof omitted
  sorry

end triangle_inradius_l159_159746


namespace area_of_defined_region_eq_14_point_4_l159_159811

def defined_region (x y : ℝ) : Prop :=
  |5 * x - 20| + |3 * y + 9| ≤ 6

def region_area : ℝ :=
  14.4

theorem area_of_defined_region_eq_14_point_4 :
  (∃ (x y : ℝ), defined_region x y) → region_area = 14.4 :=
by
  sorry

end area_of_defined_region_eq_14_point_4_l159_159811


namespace last_three_digits_of_7_pow_120_l159_159114

theorem last_three_digits_of_7_pow_120 :
  7^120 % 1000 = 681 :=
by
  sorry

end last_three_digits_of_7_pow_120_l159_159114


namespace toms_dog_is_12_l159_159066

def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := toms_rabbit_age * 3

theorem toms_dog_is_12 : toms_dog_age = 12 :=
by
  sorry

end toms_dog_is_12_l159_159066


namespace gcd_lcm_product_l159_159822

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159822


namespace distance_A_to_B_is_64_yards_l159_159393

theorem distance_A_to_B_is_64_yards :
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  distance = 64 :=
  by
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  sorry

end distance_A_to_B_is_64_yards_l159_159393


namespace max_earnings_l159_159951

section MaryEarnings

def regular_rate : ℝ := 10
def first_period_hours : ℕ := 40
def second_period_hours : ℕ := 10
def third_period_hours : ℕ := 10
def weekend_days : ℕ := 2
def weekend_bonus_per_day : ℝ := 50
def bonus_threshold_hours : ℕ := 55
def overtime_multiplier_second_period : ℝ := 0.25
def overtime_multiplier_third_period : ℝ := 0.5
def milestone_bonus : ℝ := 100

def regular_pay := regular_rate * first_period_hours
def second_period_pay := (regular_rate * (1 + overtime_multiplier_second_period)) * second_period_hours
def third_period_pay := (regular_rate * (1 + overtime_multiplier_third_period)) * third_period_hours
def weekend_bonus := weekend_days * weekend_bonus_per_day
def milestone_pay := milestone_bonus

def total_earnings := regular_pay + second_period_pay + third_period_pay + weekend_bonus + milestone_pay

theorem max_earnings : total_earnings = 875 := by
  sorry

end MaryEarnings

end max_earnings_l159_159951


namespace carla_order_cost_l159_159216

theorem carla_order_cost (base_cost : ℝ) (coupon : ℝ) (senior_discount_rate : ℝ)
  (additional_charge : ℝ) (tax_rate : ℝ) (conversion_rate : ℝ) :
  base_cost = 7.50 →
  coupon = 2.50 →
  senior_discount_rate = 0.20 →
  additional_charge = 1.00 →
  tax_rate = 0.08 →
  conversion_rate = 0.85 →
  (2 * (base_cost - coupon) * (1 - senior_discount_rate) + additional_charge) * (1 + tax_rate) * conversion_rate = 4.59 :=
by
  sorry

end carla_order_cost_l159_159216


namespace circle_intersection_range_l159_159268

theorem circle_intersection_range (r : ℝ) (H : r > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) → (1 < r ∧ r < 11) := 
by
  sorry

end circle_intersection_range_l159_159268


namespace johns_total_cost_l159_159437

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end johns_total_cost_l159_159437


namespace number_of_floors_l159_159524

-- Definitions
def height_regular_floor : ℝ := 3
def height_last_floor : ℝ := 3.5
def total_height : ℝ := 61

-- Theorem statement
theorem number_of_floors (n : ℕ) : 
  (n ≥ 2) →
  (2 * height_last_floor + (n - 2) * height_regular_floor = total_height) →
  n = 20 :=
sorry

end number_of_floors_l159_159524


namespace Deepak_age_l159_159647

variable (R D : ℕ)

theorem Deepak_age 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26) : D = 15 := 
sorry

end Deepak_age_l159_159647


namespace save_water_negate_l159_159607

/-- If saving 30cm^3 of water is denoted as +30cm^3, then wasting 10cm^3 of water is denoted as -10cm^3. -/
theorem save_water_negate :
  (∀ (save_waste : ℤ → ℤ), save_waste 30 = 30 → save_waste (-10) = -10) :=
by
  sorry

end save_water_negate_l159_159607


namespace sin_cos_sum_l159_159884

theorem sin_cos_sum (α : ℝ) (h1 : Real.sin (2 * α) = 3 / 4) (h2 : π < α) (h3 : α < 3 * π / 2) : 
  Real.sin α + Real.cos α = -Real.sqrt (7 / 4) := 
by
  sorry

end sin_cos_sum_l159_159884


namespace parents_without_fulltime_jobs_l159_159345

theorem parents_without_fulltime_jobs (total : ℕ) (mothers fathers full_time_mothers full_time_fathers : ℕ) 
(h1 : mothers = 2 * fathers / 3)
(h2 : full_time_mothers = 9 * mothers / 10)
(h3 : full_time_fathers = 3 * fathers / 4)
(h4 : mothers + fathers = total) :
(100 * (total - (full_time_mothers + full_time_fathers))) / total = 19 :=
by
  sorry

end parents_without_fulltime_jobs_l159_159345


namespace trapezoid_perimeter_l159_159611

theorem trapezoid_perimeter (x y : ℝ) (h1 : x ≠ 0)
  (h2 : ∀ (AB CD AD BC : ℝ), AB = 2 * x ∧ CD = 4 * x ∧ AD = 2 * y ∧ BC = y) :
  (∀ (P : ℝ), P = AB + BC + CD + AD → P = 6 * x + 3 * y) :=
by sorry

end trapezoid_perimeter_l159_159611


namespace trigonometric_expression_value_l159_159567

variable {α : ℝ}
axiom tan_alpha_eq : Real.tan α = 2

theorem trigonometric_expression_value :
  (1 + 2 * Real.cos (Real.pi / 2 - α) * Real.cos (-10 * Real.pi - α)) /
  (Real.cos (3 * Real.pi / 2 - α) ^ 2 - Real.sin (9 * Real.pi / 2 - α) ^ 2) = 3 :=
by
  have h_tan_alpha : Real.tan α = 2 := tan_alpha_eq
  sorry

end trigonometric_expression_value_l159_159567


namespace monthly_rent_of_shop_l159_159538

theorem monthly_rent_of_shop
  (length width : ℕ) (rent_per_sqft : ℕ)
  (h_length : length = 20) (h_width : width = 18) (h_rent : rent_per_sqft = 48) :
  (length * width * rent_per_sqft) / 12 = 1440 := 
by
  sorry

end monthly_rent_of_shop_l159_159538


namespace olympiad_scores_above_18_l159_159920

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end olympiad_scores_above_18_l159_159920


namespace domain_of_f_l159_159640

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem domain_of_f :
  {x : ℝ | x + 1 > 0} = {x : ℝ | x > -1} :=
by
  sorry

end domain_of_f_l159_159640


namespace smallest_integer_with_eight_factors_l159_159509

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l159_159509


namespace quadratic_solution_l159_159484

-- Definitions come from the conditions of the problem
def satisfies_equation (y : ℝ) : Prop := 6 * y^2 + 2 = 4 * y + 12

-- Statement of the proof
theorem quadratic_solution (y : ℝ) (hy : satisfies_equation y) : (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := 
sorry

end quadratic_solution_l159_159484


namespace power_function_evaluation_l159_159889

theorem power_function_evaluation (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 4 = 2) : f 16 = 4 :=
by
  sorry

end power_function_evaluation_l159_159889


namespace min_int_solution_inequality_l159_159059

theorem min_int_solution_inequality : ∃ x : ℤ, 4 * (x + 1) + 2 > x - 1 ∧ ∀ y : ℤ, 4 * (y + 1) + 2 > y - 1 → y ≥ x := 
by 
  sorry

end min_int_solution_inequality_l159_159059


namespace difference_of_digits_is_three_l159_159119

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem difference_of_digits_is_three :
  ∀ n : ℕ, n = 63 → tens_digit n + ones_digit n = 9 → tens_digit n - ones_digit n = 3 :=
by
  intros n h1 h2
  sorry

end difference_of_digits_is_three_l159_159119


namespace instantaneous_velocity_at_2_l159_159088

noncomputable def S (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

theorem instantaneous_velocity_at_2 :
  (deriv S 2) = 10 :=
by 
  sorry

end instantaneous_velocity_at_2_l159_159088


namespace algebraic_expression_correct_l159_159008

theorem algebraic_expression_correct (a b : ℝ) (h : a = 7 - 3 * b) : a^2 + 6 * a * b + 9 * b^2 = 49 := 
by sorry

end algebraic_expression_correct_l159_159008


namespace toby_breakfast_calories_l159_159322

noncomputable def calories_bread := 100
noncomputable def calories_peanut_butter_per_serving := 200
noncomputable def servings_peanut_butter := 2

theorem toby_breakfast_calories :
  1 * calories_bread + servings_peanut_butter * calories_peanut_butter_per_serving = 500 :=
by
  sorry

end toby_breakfast_calories_l159_159322


namespace ratio_of_two_numbers_l159_159316

variable {a b : ℝ}

theorem ratio_of_two_numbers
  (h1 : a + b = 7 * (a - b))
  (h2 : 0 < b)
  (h3 : a > b) :
  a / b = 4 / 3 := by
  sorry

end ratio_of_two_numbers_l159_159316


namespace range_f_l159_159233

noncomputable def g (x : ℝ) : ℝ := 30 + 14 * Real.cos x - 7 * Real.cos (2 * x)

noncomputable def z (t : ℝ) : ℝ := 40.5 - 14 * (t - 0.5) ^ 2

noncomputable def u (z : ℝ) : ℝ := (Real.pi / 54) * z

noncomputable def f (x : ℝ) : ℝ := Real.sin (u (z (Real.cos x)))

theorem range_f : ∀ x : ℝ, 0.5 ≤ f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end range_f_l159_159233


namespace fencing_required_l159_159206

theorem fencing_required (L W : ℕ) (A : ℕ) (hL : L = 20) (hA : A = 680) (hArea : A = L * W) : 2 * W + L = 88 :=
by
  sorry

end fencing_required_l159_159206


namespace half_sum_of_squares_l159_159197

theorem half_sum_of_squares (n m : ℕ) (h : n ≠ m) :
  ∃ a b : ℕ, ( (2 * n)^2 + (2 * m)^2) / 2 = a^2 + b^2 := by
  sorry

end half_sum_of_squares_l159_159197


namespace temperature_decrease_l159_159490

theorem temperature_decrease (initial : ℤ) (decrease : ℤ) : initial = -3 → decrease = 6 → initial - decrease = -9 :=
by
  intros
  sorry

end temperature_decrease_l159_159490


namespace find_g2_l159_159968

theorem find_g2 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1 / x) = 3 ^ x) : 
  g 2 = (9 - 3 * Real.sqrt 3) / 8 := 
sorry

end find_g2_l159_159968


namespace trout_split_equally_l159_159037

-- Conditions: Nancy and Joan caught 18 trout and split them equally
def total_trout : ℕ := 18
def equal_split (n : ℕ) : ℕ := n / 2

-- Theorem: Prove that if they equally split the trout, each person will get 9 trout.
theorem trout_split_equally : equal_split total_trout = 9 :=
by 
  -- Placeholder for the actual proof
  sorry

end trout_split_equally_l159_159037


namespace zeros_of_f_l159_159064

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x ^ 2 - 2 * x - 3)

theorem zeros_of_f :
  { x : ℝ | f x = 0 } = {1, -1, 3} :=
sorry

end zeros_of_f_l159_159064


namespace number_of_multiples_of_7_but_not_14_l159_159258

-- Define the context and conditions
def positive_integers_less_than_500 : set ℕ := {n : ℕ | 0 < n ∧ n < 500 }
def multiples_of_7 : set ℕ := {n : ℕ | n % 7 = 0 }
def multiples_of_14 : set ℕ := {n : ℕ | n % 14 = 0 }
def multiples_of_7_but_not_14 : set ℕ := { n | n ∈ multiples_of_7 ∧ n ∉ multiples_of_14 }

-- Define the theorem to prove
theorem number_of_multiples_of_7_but_not_14 : 
  ∃! n : ℕ, n = 36 ∧ n = finset.card (finset.filter (λ x, x ∈ multiples_of_7_but_not_14) (finset.range 500)) :=
begin
  sorry
end

end number_of_multiples_of_7_but_not_14_l159_159258


namespace simplified_expression_l159_159663

theorem simplified_expression :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 :=
by
  sorry

end simplified_expression_l159_159663


namespace probability_two_red_two_blue_l159_159669

theorem probability_two_red_two_blue (total_red total_blue : ℕ) (red_taken blue_taken selected : ℕ)
  (h_red_total : total_red = 12) (h_blue_total : total_blue = 8) (h_selected : selected = 4)
  (h_red_taken : red_taken = 2) (h_blue_taken : blue_taken = 2) :
  (Nat.choose total_red red_taken) * (Nat.choose total_blue blue_taken) /
    (Nat.choose (total_red + total_blue) selected : ℚ) = 1848 / 4845 := 
by 
  sorry

end probability_two_red_two_blue_l159_159669


namespace find_simple_annual_rate_l159_159023

-- Conditions from part a).
-- 1. Principal initial amount (P) is $5,000.
-- 2. Annual interest rate for compounded interest (r) is 0.06.
-- 3. Number of times it compounds per year (n) is 2 (semi-annually).
-- 4. Time period (t) is 1 year.
-- 5. The interest earned after one year for simple interest is $6 less than compound interest.

noncomputable def principal : ℝ := 5000
noncomputable def annual_rate_compound : ℝ := 0.06
noncomputable def times_compounded : ℕ := 2
noncomputable def time_years : ℝ := 1
noncomputable def compound_interest : ℝ := principal * (1 + annual_rate_compound / times_compounded) ^ (times_compounded * time_years) - principal
noncomputable def simple_interest : ℝ := compound_interest - 6

-- Question from part a) translated to Lean statement using the condition that simple interest satisfaction
theorem find_simple_annual_rate : 
    ∃ r : ℝ, principal * r * time_years = simple_interest :=
by
  exists (0.0597)
  sorry

end find_simple_annual_rate_l159_159023


namespace purely_imaginary_solution_l159_159402

noncomputable def complex_number_is_purely_imaginary (m : ℝ) : Prop :=
  (m^2 - 2 * m - 3 = 0) ∧ (m + 1 ≠ 0)

theorem purely_imaginary_solution (m : ℝ) (h : complex_number_is_purely_imaginary m) : m = 3 := by
  sorry

end purely_imaginary_solution_l159_159402


namespace solve_for_x_l159_159582

theorem solve_for_x (x : ℤ) (h : -3 * x - 8 = 8 * x + 3) : x = -1 :=
by
  sorry

end solve_for_x_l159_159582


namespace x_pow_12_eq_one_l159_159726

theorem x_pow_12_eq_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 :=
sorry

end x_pow_12_eq_one_l159_159726


namespace xy_product_l159_159887

-- Define the proof problem with the conditions and required statement
theorem xy_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy_distinct : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 := 
  sorry

end xy_product_l159_159887


namespace vasya_birthday_is_thursday_l159_159804

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159804


namespace arithmetic_seq_sum_l159_159712

theorem arithmetic_seq_sum (a_n : ℕ → ℝ) (h_arith_seq : ∃ d, ∀ n, a_n (n + 1) = a_n n + d)
    (h_sum : a_n 5 + a_n 8 = 24) : a_n 6 + a_n 7 = 24 := by
  sorry

end arithmetic_seq_sum_l159_159712


namespace barbi_weight_loss_duration_l159_159859

theorem barbi_weight_loss_duration :
  (∃ x : ℝ, 
    (∃ l_barbi l_luca : ℝ, 
      l_barbi = 1.5 * x ∧ 
      l_luca = 99 ∧ 
      l_luca = l_barbi + 81) ∧
    x = 12) :=
by
  sorry

end barbi_weight_loss_duration_l159_159859


namespace min_bulbs_lit_proof_l159_159878

-- Definitions to capture initial conditions and the problem
def bulb_state (n : ℕ) : Type := fin n → fin n → bool

def initial_state (n : ℕ) : bulb_state n := λ i j, false

-- Function to describe the state change after pressing a bulb
def press_bulb (n : ℕ) (state : bulb_state n) (i j : fin n) : bulb_state n :=
  λ x y, if x = i ∨ y = j then ¬ state x y else state x y

-- Function to calculate the minimum lit bulbs
def min_lit_bulbs (n : ℕ) (initial : bulb_state n) : ℕ :=
  2 * n - 2

-- The theorem statement
theorem min_bulbs_lit_proof (n : ℕ) (initial : bulb_state n) : ∃ seq : list (fin n × fin n), 
  (press_bulb n initial_state seq.head.1 seq.head.2).count true = 2 * n - 2 :=
sorry

end min_bulbs_lit_proof_l159_159878


namespace quadratic_value_at_two_l159_159398

open Real

-- Define the conditions
variables (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 + a * x + b

-- State the proof problem
theorem quadratic_value_at_two (h₀ : f a b (f a b 0) = 0) (h₁ : f a b (f a b 1) = 0) (h₂ : f a b 0 ≠ f a b 1) :
  f a b 2 = 2 := 
sorry

end quadratic_value_at_two_l159_159398


namespace simplify_fraction_l159_159299

noncomputable def sin_15 := Real.sin (15 * Real.pi / 180)
noncomputable def cos_15 := Real.cos (15 * Real.pi / 180)
noncomputable def angle_15 := 15 * Real.pi / 180

theorem simplify_fraction : (1 / sin_15 - 1 / cos_15 = 2 * Real.sqrt 2) :=
by
  sorry

end simplify_fraction_l159_159299


namespace find_number_l159_159093

theorem find_number (x : ℤ) (h : 3 * (x + 8) = 36) : x = 4 :=
by {
  sorry
}

end find_number_l159_159093


namespace math_olympiad_proof_l159_159932

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l159_159932


namespace part1_part2_l159_159395

/-- Given a triangle ABC with sides opposite to angles A, B, C being a, b, c respectively,
and a sin A sin B + b cos^2 A = 5/3 a,
prove that (1) b / a = 5/3. -/
theorem part1 (a b : ℝ) (A B : ℝ) (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a) :
  b / a = 5 / 3 :=
sorry

/-- Given the previous result b / a = 5/3 and the condition c^2 = a^2 + 8/5 b^2,
prove that (2) angle C = 2π / 3. -/
theorem part2 (a b c : ℝ) (A B C : ℝ)
  (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a)
  (h₂ : c^2 = a^2 + (8 / 5) * b^2)
  (h₃ : b / a = 5 / 3) :
  C = 2 * Real.pi / 3 :=
sorry

end part1_part2_l159_159395


namespace reduce_to_one_l159_159463

theorem reduce_to_one (n : ℕ) : ∃ k, (k = 1) :=
by
  sorry

end reduce_to_one_l159_159463


namespace convert_binary_1101_to_decimal_l159_159550

theorem convert_binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by sorry

end convert_binary_1101_to_decimal_l159_159550


namespace sum_of_consecutive_integers_l159_159740

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) = 358800) : 
  n + (n + 1) + (n + 2) + (n + 3) = 98 :=
sorry

end sum_of_consecutive_integers_l159_159740


namespace measure_angle_C_l159_159790

theorem measure_angle_C (A B C : ℝ) (h1 : A = 60) (h2 : B = 60) (h3 : C = 60 - 10) (sum_angles : A + B + C = 180) : C = 53.33 :=
by
  sorry

end measure_angle_C_l159_159790


namespace roots_of_unity_cubic_l159_159864

noncomputable def countRootsOfUnityCubic (c d e : ℤ) : ℕ := sorry

theorem roots_of_unity_cubic :
  ∃ (z : ℂ) (n : ℕ), (z^n = 1) ∧ (∃ (c d e : ℤ), z^3 + c * z^2 + d * z + e = 0)
  ∧ countRootsOfUnityCubic c d e = 12 :=
sorry

end roots_of_unity_cubic_l159_159864


namespace number_of_real_pairs_in_arithmetic_progression_l159_159225

theorem number_of_real_pairs_in_arithmetic_progression : 
  ∃ (pairs : Finset (ℝ × ℝ)), 
  (∀ (a b : ℝ), (a, b) ∈ pairs ↔ 12 + b = 2 * a ∧ b = 2 * a / (ab - 4b + b + 12)) ∧ 
  Finset.card pairs = 2 := sorry

end number_of_real_pairs_in_arithmetic_progression_l159_159225


namespace pizza_slices_left_per_person_l159_159297

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

end pizza_slices_left_per_person_l159_159297


namespace smallest_positive_integer_with_eight_factors_l159_159508

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l159_159508


namespace total_distance_both_l159_159630

-- Define conditions
def speed_onur : ℝ := 35  -- km/h
def speed_hanil : ℝ := 45  -- km/h
def daily_hours_onur : ℝ := 7
def additional_distance_hanil : ℝ := 40
def days_in_week : ℕ := 7

-- Define the daily biking distance for Onur and Hanil
def distance_onur_daily : ℝ := speed_onur * daily_hours_onur
def distance_hanil_daily : ℝ := distance_onur_daily + additional_distance_hanil

-- Define the number of days Onur and Hanil bike in a week
def working_days_onur : ℕ := 5
def working_days_hanil : ℕ := 6

-- Define the total distance covered by Onur and Hanil in a week
def total_distance_onur_week : ℝ := distance_onur_daily * working_days_onur
def total_distance_hanil_week : ℝ := distance_hanil_daily * working_days_hanil

-- Proof statement
theorem total_distance_both : total_distance_onur_week + total_distance_hanil_week = 2935 := by
  sorry

end total_distance_both_l159_159630


namespace no_primes_in_sequence_l159_159519

def P : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

theorem no_primes_in_sequence :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 59 → ¬ Nat.Prime (P + n) :=
by
  sorry

end no_primes_in_sequence_l159_159519


namespace find_remainder_l159_159738

-- Main statement with necessary definitions and conditions
theorem find_remainder (x : ℤ) (h : (x + 11) % 31 = 18) :
  x % 62 = 7 :=
sorry

end find_remainder_l159_159738


namespace model_scale_representation_l159_159536

theorem model_scale_representation :
  let scale_factor := 50
  let model_length_cm := 7.5
  real_length_m = scale_factor * model_length_cm 
  true :=
  by
  let scale_factor := 50
  let model_length_cm := 7.5
  let real_length_m := scale_factor * model_length_cm
  sorry

end model_scale_representation_l159_159536


namespace inequality_f_l159_159243

-- Definitions of the given conditions
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

-- Theorem statement
theorem inequality_f (a b : ℝ) : 
  abs (f 1 a b) + 2 * abs (f 2 a b) + abs (f 3 a b) ≥ 2 :=
by sorry

end inequality_f_l159_159243


namespace marble_probability_l159_159840

theorem marble_probability :
  let redMarbles := 4
  let blueMarbles := 6
  let totalMarbles := redMarbles + blueMarbles
  let firstRedProb := (redMarbles: ℝ) / totalMarbles
  let remainingMarbles := totalMarbles - 1
  let secondBlueProb := (blueMarbles: ℝ) / remainingMarbles
  let combinedProb := firstRedProb * secondBlueProb
  combinedProb = 4 / 15 :=
by
  sorry

end marble_probability_l159_159840


namespace roots_product_eq_l159_159287

theorem roots_product_eq
  (a b m p r : ℚ)
  (h₀ : a * b = 3)
  (h₁ : ∀ x, x^2 - m * x + 3 = 0 → (x = a ∨ x = b))
  (h₂ : ∀ x, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a)) : 
  r = 16 / 3 :=
by
  sorry

end roots_product_eq_l159_159287


namespace solve_for_y_l159_159730

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159730


namespace circle_line_intersect_property_l159_159276
open Real

theorem circle_line_intersect_property :
  let ρ := fun θ : ℝ => 4 * sqrt 2 * sin (3 * π / 4 - θ)
  let cartesian_eq := fun x y : ℝ => (x - 2) ^ 2 + (y - 2) ^ 2 = 8
  let slope := sqrt 3
  let line_param := fun t : ℝ => (1/2 * t, 2 + sqrt 3 / 2 * t)
  let t_roots := {t | ∃ t1 t2 : ℝ, t1 + t2 = 2 ∧ t1 * t2 = -4 ∧ (t = t1 ∨ t = t2)}
  
  (∀ t ∈ t_roots, 
    let (x, y) := line_param t
    cartesian_eq x y)
  → abs ((1 : ℝ) / abs 1 - (1 : ℝ) / abs 2) = 1 / 2 :=
by
  intro ρ cartesian_eq slope line_param t_roots h
  sorry

end circle_line_intersect_property_l159_159276


namespace ellipse_standard_form_l159_159717

theorem ellipse_standard_form (α : ℝ) 
  (x y : ℝ) 
  (hx : x = 5 * Real.cos α) 
  (hy : y = 3 * Real.sin α) : 
  (x^2 / 25) + (y^2 / 9) = 1 := 
by 
  sorry

end ellipse_standard_form_l159_159717


namespace ratio_naomi_to_katherine_l159_159215

theorem ratio_naomi_to_katherine 
  (katherine_time : ℕ) 
  (naomi_total_time : ℕ) 
  (websites_naomi : ℕ)
  (hk : katherine_time = 20)
  (hn : naomi_total_time = 750)
  (wn : websites_naomi = 30) : 
  naomi_total_time / websites_naomi / katherine_time = 5 / 4 := 
by sorry

end ratio_naomi_to_katherine_l159_159215


namespace inverse_function_passing_point_l159_159403

theorem inverse_function_passing_point
  (f : ℝ → ℝ) (hf : Function.Bijective f) (h1 : f(4) = 3) : f⁻¹ 3 = 4 :=
by
  -- to prove
  sorry

end inverse_function_passing_point_l159_159403


namespace product_mn_l159_159482

-- Λet θ1 be the angle L1 makes with the positive x-axis.
-- Λet θ2 be the angle L2 makes with the positive x-axis.
-- Given that θ1 = 3 * θ2 and m = 6 * n.
-- Using the tangent triple angle formula: tan(3θ) = (3 * tan(θ) - tan^3(θ)) / (1 - 3 * tan^2(θ))
-- We need to prove mn = 9/17.

noncomputable def mn_product_condition (θ1 θ2 : ℝ) (m n : ℝ) : Prop :=
θ1 = 3 * θ2 ∧ m = 6 * n ∧ m = Real.tan θ1 ∧ n = Real.tan θ2

theorem product_mn (θ1 θ2 : ℝ) (m n : ℝ) (h : mn_product_condition θ1 θ2 m n) :
  m * n = 9 / 17 :=
sorry

end product_mn_l159_159482


namespace tan_product_identity_l159_159597

theorem tan_product_identity : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 3)) = 4 + 2 * Real.sqrt 3 :=
by
  sorry

end tan_product_identity_l159_159597


namespace quadratic_no_real_roots_l159_159010

-- Given conditions
variables {p q a b c : ℝ}
variables (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variables (hp_neq_q : p ≠ q)

-- p, a, q form a geometric sequence
variables (h_geo : a^2 = p * q)

-- p, b, c, q form an arithmetic sequence
variables (h_arith1 : 2 * b = p + c)
variables (h_arith2 : 2 * c = b + q)

-- Proof statement
theorem quadratic_no_real_roots (hp_pos hq_pos ha_pos hb_pos hc_pos hp_neq_q h_geo h_arith1 h_arith2 : ℝ) :
    (b * (x : ℝ)^2 - 2 * a * x + c = 0) → false :=
sorry

end quadratic_no_real_roots_l159_159010


namespace min_x2_y2_z2_l159_159455

open Real

theorem min_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_x2_y2_z2_l159_159455


namespace train_cross_time_l159_159903

theorem train_cross_time (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ) : 
  length_train = 100 →
  length_bridge = 150 →
  speed_kmph = 63 →
  (length_train + length_bridge) / (speed_kmph * (1000 / 3600)) = 14.29 :=
by
  sorry

end train_cross_time_l159_159903


namespace units_digit_problem_l159_159235

open BigOperators

-- Define relevant constants
def A : ℤ := 21
noncomputable def B : ℤ := 14 -- since B = sqrt(196) = 14

-- Define the terms
noncomputable def term1 : ℤ := (A + B) ^ 20
noncomputable def term2 : ℤ := (A - B) ^ 20

-- Statement of the theorem
theorem units_digit_problem :
  ((term1 - term2) % 10) = 4 := 
sorry

end units_digit_problem_l159_159235


namespace inequality_proof_l159_159520

variable (u v w : ℝ)

theorem inequality_proof (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) (h4 : u + v + w + Real.sqrt (u * v * w) = 4) :
    Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w := 
  sorry

end inequality_proof_l159_159520


namespace solve_n_l159_159148

open Nat

def condition (n : ℕ) : Prop := 2^(n + 1) * 2^3 = 2^10

theorem solve_n (n : ℕ) (hn_pos : 0 < n) (h_cond : condition n) : n = 6 :=
by
  sorry

end solve_n_l159_159148


namespace production_days_l159_159833

theorem production_days (n : ℕ) (h₁ : (50 * n + 95) / (n + 1) = 55) : 
    n = 8 := 
    sorry

end production_days_l159_159833


namespace line_tangent_to_circle_l159_159714

noncomputable def circle_diameter : ℝ := 13
noncomputable def distance_from_center_to_line : ℝ := 6.5

theorem line_tangent_to_circle :
  ∀ (d r : ℝ), d = 13 → r = 6.5 → r = d/2 → distance_from_center_to_line = r → 
  (distance_from_center_to_line = r) := 
by
  intros d r hdiam hdist hradius hdistance
  sorry

end line_tangent_to_circle_l159_159714


namespace M_positive_l159_159149

theorem M_positive (x y : ℝ) : (3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13) > 0 :=
by
  sorry

end M_positive_l159_159149


namespace max_numbers_with_240_product_square_l159_159566

theorem max_numbers_with_240_product_square :
  ∃ (S : Finset ℕ), S.card = 11 ∧ ∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ n m, 240 * k = (n * m) ^ 2 :=
sorry

end max_numbers_with_240_product_square_l159_159566


namespace broadcast_arrangements_l159_159478

/-- 
The TV station plans to select 5 programs from 5 recorded news reports and 4 personality interview programs 
to broadcast one each day from October 1st to October 5th. If the number of news report programs cannot 
be less than 3, there are 9720 different broadcasting arrangements. 
-/
theorem broadcast_arrangements : 
  let news_reports := 5
  let interview_programs := 4
  let total_programs := 5 -∗ find_sequences news_reports interview_programs total_programs = 9720
  sorry

end broadcast_arrangements_l159_159478


namespace white_water_addition_l159_159749

theorem white_water_addition :
  ∃ (W H I T E A R : ℕ), 
  W ≠ H ∧ W ≠ I ∧ W ≠ T ∧ W ≠ E ∧ W ≠ A ∧ W ≠ R ∧
  H ≠ I ∧ H ≠ T ∧ H ≠ E ∧ H ≠ A ∧ H ≠ R ∧
  I ≠ T ∧ I ≠ E ∧ I ≠ A ∧ I ≠ R ∧
  T ≠ E ∧ T ≠ A ∧ T ≠ R ∧
  E ≠ A ∧ E ≠ R ∧
  A ≠ R ∧
  W = 8 ∧ I = 6 ∧ P = 1 ∧ C = 9 ∧ N = 0 ∧
  (10000 * W + 1000 * H + 100 * I + 10 * T + E) + 
  (10000 * W + 1000 * A + 100 * T + 10 * E + R) = 169069 :=
by 
  sorry

end white_water_addition_l159_159749


namespace rhombus_height_l159_159110

theorem rhombus_height (a d1 d2 : ℝ) (h : ℝ)
  (h_a_positive : 0 < a)
  (h_d1_positive : 0 < d1)
  (h_d2_positive : 0 < d2)
  (h_side_geometric_mean : a^2 = d1 * d2) :
  h = a / 2 :=
sorry

end rhombus_height_l159_159110


namespace probability_not_all_same_l159_159320

-- Definitions for the given conditions
inductive Color
| red | yellow | green

def draw (n : ℕ) : list Color := replicate n Color.red ++ replicate n Color.yellow ++ replicate n Color.green

-- Problem statement
theorem probability_not_all_same : 
  let total_ways := 3^3 in
  let same_color_ways := 3 in
  (1 - (same_color_ways / total_ways) = 8 / 9) :=
by
  sorry

end probability_not_all_same_l159_159320


namespace group_sizes_correct_l159_159319

-- Define the number of fruits and groups
def num_bananas : Nat := 527
def num_oranges : Nat := 386
def num_apples : Nat := 319

def groups_bananas : Nat := 11
def groups_oranges : Nat := 103
def groups_apples : Nat := 17

-- Define the expected sizes of each group
def bananas_per_group : Nat := 47
def oranges_per_group : Nat := 3
def apples_per_group : Nat := 18

-- Prove the sizes of the groups are as expected
theorem group_sizes_correct :
  (num_bananas / groups_bananas = bananas_per_group) ∧
  (num_oranges / groups_oranges = oranges_per_group) ∧
  (num_apples / groups_apples = apples_per_group) :=
by
  -- Division in Nat rounds down
  have h1 : num_bananas / groups_bananas = 47 := by sorry
  have h2 : num_oranges / groups_oranges = 3 := by sorry
  have h3 : num_apples / groups_apples = 18 := by sorry
  exact ⟨h1, h2, h3⟩

end group_sizes_correct_l159_159319


namespace polynomial_not_separable_l159_159041

theorem polynomial_not_separable (f g : Polynomial ℂ) :
  (∀ x y : ℂ, f.eval x * g.eval y = x^200 * y^200 + 1) → False :=
sorry

end polynomial_not_separable_l159_159041


namespace trailing_zeros_30_factorial_l159_159593

theorem trailing_zeros_30_factorial : 
  let count_factors (n : ℕ) (p : ℕ) : ℕ := 
    if p <= 1 then 0 else 
    let rec_count (n : ℕ) : ℕ :=
      if n < p then 0 else n / p + rec_count (n / p)
    rec_count n
  in count_factors 30 5 = 7 := 
  sorry

end trailing_zeros_30_factorial_l159_159593


namespace fraction_always_defined_l159_159214

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 := 
by
  -- proof is not required
  sorry

end fraction_always_defined_l159_159214


namespace closest_fraction_to_team_alpha_medals_l159_159545

theorem closest_fraction_to_team_alpha_medals :
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 5) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 6) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 7) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 9) := 
by
  sorry

end closest_fraction_to_team_alpha_medals_l159_159545


namespace multiple_of_four_and_six_prime_sum_even_l159_159775

theorem multiple_of_four_and_six_prime_sum_even {a b : ℤ} 
  (h_a : ∃ m : ℤ, a = 4 * m) 
  (h_b1 : ∃ n : ℤ, b = 6 * n) 
  (h_b2 : Prime b) : 
  Even (a + b) := 
  by sorry

end multiple_of_four_and_six_prime_sum_even_l159_159775


namespace max_min_f_for_a_eq_2_a_sqrt_2_if_diff_is_2_l159_159141

-- Definitions of A and f(x)
def A : Set ℝ := { x | x^2 - 6*x + 8 ≤ 0 }

def f (a x : ℝ) : ℝ := a^x

-- Problem 1: Maximum and Minimum values for a = 2
theorem max_min_f_for_a_eq_2 :
  (∀ x ∈ A, 2 ≤ x ∧ x ≤ 4) → (∀ x ∈ A, f 2 x = 4) ∧ (∀ x ∈ A, f 2 x = 16) :=
by
  sorry

-- Problem 2: Finding x such that the difference is 2
theorem a_sqrt_2_if_diff_is_2 :
  (∀ x ∈ A, 2 ≤ x ∧ x ≤ 4) → (∀ a > 0, a ≠ 1 → ((∃ x y ∈ A, (f a y - f a x = 2)) → a = Real.sqrt 2)) :=
by
  sorry    

end max_min_f_for_a_eq_2_a_sqrt_2_if_diff_is_2_l159_159141


namespace base_8_sum_units_digit_l159_159874

section
  def digit_in_base (n : ℕ) (base : ℕ) (d : ℕ) : Prop :=
  ((n % base) = d)

theorem base_8_sum_units_digit :
  let n1 := 63
  let n2 := 74
  let base := 8
  (digit_in_base n1 base 3) →
  (digit_in_base n2 base 4) →
  digit_in_base (n1 + n2) base 7 :=
by
  intro h1 h2
  -- placeholder for the detailed proof
  sorry
end

end base_8_sum_units_digit_l159_159874


namespace gcd_values_count_l159_159820

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159820


namespace sum_of_integers_sqrt_485_l159_159975

theorem sum_of_integers_sqrt_485 (x y : ℕ) (h1 : x^2 + y^2 = 245) (h2 : x * y = 120) : x + y = Real.sqrt 485 :=
sorry

end sum_of_integers_sqrt_485_l159_159975


namespace integer_solutions_l159_159699

theorem integer_solutions :
  { (x, y) : ℤ × ℤ | x^2 = 1 + 4 * y^3 * (y + 2) } = {(1, 0), (1, -2), (-1, 0), (-1, -2)} :=
by
  sorry

end integer_solutions_l159_159699


namespace intervals_of_monotonicity_range_of_a_for_three_zeros_l159_159893

    -- Define the function f(x) = 1/2 * x^2 - 3 * a * x + 2 * a^2 * ln(x)
    noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * Real.log x

    -- Prove the intervals of monotonicity for f(x)
    theorem intervals_of_monotonicity (a : ℝ) (h : a ≠ 0) :
      (a > 0 → 
         monotone_on (f a) (Ioo 0 a) ∧
         antitone_on (f a) (Ioo a (2 * a)) ∧
         monotone_on (f a) (Ioi (2 * a))) ∧
      (a < 0 → 
         monotone_on (f a) (Ioi 0)) := 
    sorry

    -- Define the function's number of zeros and check the range of 'a' for having 3 zeros
    noncomputable def number_of_zeros (f : ℝ → ℝ) : ℕ := 
      -- A placeholder function representing the number of zeros of f
      sorry 

    theorem range_of_a_for_three_zeros :
      ∃ a : ℝ, (e^(5/4) < a ∧ a < (e^2 / 2)) ∧ number_of_zeros (f a) = 3 :=
    sorry
    
end intervals_of_monotonicity_range_of_a_for_three_zeros_l159_159893


namespace shaded_region_area_l159_159692

noncomputable def line1 (x : ℝ) : ℝ := -(3 / 10) * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -(5 / 7) * x + 47 / 7

noncomputable def intersection_x : ℝ := 17 / 5

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem shaded_region_area : 
  area_under_curve line1 line2 0 intersection_x = 1.91 :=
sorry

end shaded_region_area_l159_159692


namespace find_general_term_find_sum_of_b_l159_159711

variables {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Given conditions
axiom a5 : a 5 = 10
axiom S7 : S 7 = 56

-- Definition of S (Sum of first n terms of an arithmetic sequence)
def S_def (a : ℕ → ℕ) (n : ℕ) : ℕ := n * (a 1 + a n) / 2

-- Definition of the arithmetic sequence
def a_arith_seq (n : ℕ) : ℕ := 2 * n

-- Assuming the axiom for the arithmetic sequence sum
axiom S_is_arith : S 7 = S_def a 7

theorem find_general_term : a = a_arith_seq := 
by sorry

-- Sequence b
def b (n : ℕ) : ℕ := 2 + 9 ^ n

-- Sum of first n terms of sequence b
def T (n : ℕ) : ℕ := (Finset.range n).sum b

-- Prove T_n formula
theorem find_sum_of_b : ∀ n, T n = 2 * n + 9 / 8 * (9 ^ n - 1) :=
by sorry

end find_general_term_find_sum_of_b_l159_159711


namespace sequences_recurrence_relation_l159_159973

theorem sequences_recurrence_relation 
    (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ)
    (h1 : a 1 = 1) (h2 : b 1 = 3) (h3 : c 1 = 2)
    (ha : ∀ i : ℕ, a (i + 1) = a i + c i - b i + 2)
    (hb : ∀ i : ℕ, b (i + 1) = (3 * c i - a i + 5) / 2)
    (hc : ∀ i : ℕ, c (i + 1) = 2 * a i + 2 * b i - 3) : 
    (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2^n + 1) ∧ (∀ n, c n = 3 * 2^(n-1) - 1) := 
sorry

end sequences_recurrence_relation_l159_159973


namespace hundreds_digit_of_8_pow_2048_l159_159863

theorem hundreds_digit_of_8_pow_2048 : 
  (8^2048 % 1000) / 100 = 0 := 
by
  sorry

end hundreds_digit_of_8_pow_2048_l159_159863


namespace emma_troy_wrapping_time_l159_159661

theorem emma_troy_wrapping_time (emma_rate troy_rate total_task_time together_time emma_remaining_time : ℝ) 
  (h1 : emma_rate = 1 / 6) 
  (h2 : troy_rate = 1 / 8) 
  (h3 : total_task_time = 1) 
  (h4 : together_time = 2) 
  (h5 : emma_remaining_time = (total_task_time - (emma_rate + troy_rate) * together_time) / emma_rate) : 
  emma_remaining_time = 2.5 := 
sorry

end emma_troy_wrapping_time_l159_159661


namespace find_n_l159_159483

theorem find_n {x n : ℕ} (h1 : 3 * x - 4 = 8) (h2 : 7 * x - 15 = 13) (h3 : 4 * x + 2 = 18) 
  (h4 : n = 803) : 8 + (n - 1) * 5 = 4018 := by
  sorry

end find_n_l159_159483


namespace quadratic_distinct_real_roots_l159_159154

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^2 + m*x + (m + 3) = 0)) ↔ (m < -2 ∨ m > 6) := 
sorry

end quadratic_distinct_real_roots_l159_159154


namespace number_of_zeros_in_factorial_30_l159_159592

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end number_of_zeros_in_factorial_30_l159_159592


namespace parallel_vectors_implies_m_eq_neg1_l159_159900

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_implies_m_eq_neg1_l159_159900


namespace rational_inequality_solution_l159_159772

variable (x : ℝ)

def inequality_conditions : Prop := (2 * x - 1) / (x + 1) > 1

def inequality_solution : Prop := x < -1 ∨ x > 2

theorem rational_inequality_solution : inequality_conditions x → inequality_solution x :=
by
  sorry

end rational_inequality_solution_l159_159772


namespace fraction_filled_l159_159830

variables (E P p : ℝ)

-- Condition 1: The empty vessel weighs 12% of its total weight when filled.
axiom cond1 : E = 0.12 * (E + P)

-- Condition 2: The weight of the partially filled vessel is one half that of a completely filled vessel.
axiom cond2 : E + p = 1 / 2 * (E + P)

theorem fraction_filled : p / P = 19 / 44 :=
by
  sorry

end fraction_filled_l159_159830


namespace process_can_continue_indefinitely_l159_159118

noncomputable def P (x : ℝ) : ℝ := x^3 - x^2 - x - 1

-- Assume the existence of t > 1 such that P(t) = 0
axiom exists_t : ∃ t : ℝ, t > 1 ∧ P t = 0

def triangle_inequality_fails (a b c : ℝ) : Prop :=
  ¬(a + b > c ∧ b + c > a ∧ c + a > b)

def shorten (a b : ℝ) : ℝ := a + b

def can_continue_indefinitely (a b c : ℝ) : Prop :=
  ∀ t, t > 0 → ∀ a b c, triangle_inequality_fails a b c → 
  (triangle_inequality_fails (shorten b c - shorten a b) b c ∧
   triangle_inequality_fails a (shorten a c - shorten b c) c ∧
   triangle_inequality_fails a b (shorten a b - shorten b c))

theorem process_can_continue_indefinitely (a b c : ℝ) (h : triangle_inequality_fails a b c) :
  can_continue_indefinitely a b c :=
sorry

end process_can_continue_indefinitely_l159_159118


namespace probability_sum_equals_6_l159_159518

theorem probability_sum_equals_6 : 
  let possible_outcomes := 36
  let favorable_outcomes := 5
  (favorable_outcomes / possible_outcomes : ℚ) = 5 / 36 := 
by 
  sorry

end probability_sum_equals_6_l159_159518


namespace marble_probability_correct_l159_159671

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l159_159671


namespace lunch_choices_l159_159651

theorem lunch_choices (chickens drinks : ℕ) (h1 : chickens = 3) (h2 : drinks = 2) : chickens * drinks = 6 :=
by
  sorry

end lunch_choices_l159_159651


namespace possible_sums_of_digits_l159_159086

-- Defining the main theorem
theorem possible_sums_of_digits 
  (A B C : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (hdiv : (A + 6 + 2 + 8 + B + 7 + C + 3) % 9 = 0) :
  A + B + C = 1 ∨ A + B + C = 10 ∨ A + B + C = 19 :=
by
  sorry

end possible_sums_of_digits_l159_159086


namespace cosine_difference_l159_159885

open Real

theorem cosine_difference (a : ℝ)
  (a_pos : 0 < a)
  (a_lt_pi_div_2 : a < π / 2)
  (tan_a : tan a = 2) :
  cos (a - π / 4) = 3 * sqrt 10 / 10 :=
by
  sorry

end cosine_difference_l159_159885


namespace total_situps_l159_159413

theorem total_situps (hani_rate_increase : ℕ) (diana_situps : ℕ) (diana_rate : ℕ) : 
  hani_rate_increase = 3 →
  diana_situps = 40 → 
  diana_rate = 4 →
  let diana_time := diana_situps / diana_rate in
  let hani_rate := diana_rate + hani_rate_increase in
  let hani_situps := hani_rate * diana_time in
  diana_situps + hani_situps = 110 := 
by
  intro hani_rate_increase_is_three diana_situps_is_forty diana_rate_is_four
  let diana_time := diana_situps / diana_rate
  let hani_rate := diana_rate + hani_rate_increase
  let hani_situps := hani_rate * diana_time
  sorry

end total_situps_l159_159413


namespace find_n_l159_159327

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ 100 * n % 103 = 65 % 103 ∧ n = 68 :=
by
  sorry

end find_n_l159_159327


namespace talia_drives_total_distance_l159_159473

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

end talia_drives_total_distance_l159_159473


namespace clubs_equal_students_l159_159428

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l159_159428


namespace remainder_of_special_integers_l159_159615

theorem remainder_of_special_integers :
  let N := { n : ℕ | n ≤ 1050 ∧ (nat.binary_repr n).count (λ b, b = 1) > (nat.binary_repr n).count (λ b, b = 0) }.card in
  N % 1000 = 737 :=
by
  sorry

end remainder_of_special_integers_l159_159615


namespace find_g2_l159_159181

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1)
noncomputable def g (f : ℝ → ℝ) (y : ℝ) : ℝ := f⁻¹ y

variable (a : ℝ)
variable (h_inv : ∀ (x : ℝ), g (f a) (f a x) = x)
variable (h_g4 : g (f a) 4 = 2)

theorem find_g2 : g (f a) 2 = 3 / 2 :=
by sorry

end find_g2_l159_159181


namespace jose_share_of_profit_l159_159196

theorem jose_share_of_profit 
  (profit : ℚ) 
  (tom_investment : ℚ) 
  (tom_months : ℚ)
  (jose_investment : ℚ)
  (jose_months : ℚ)
  (gcd_val : ℚ)
  (total_ratio : ℚ)
  : profit = 6300 → tom_investment = 3000 → tom_months = 12 →
    jose_investment = 4500 → jose_months = 10 →
    gcd_val = 9000 → total_ratio = 9 →
    (jose_investment * jose_months / gcd_val) / total_ratio * profit = 3500 :=
by
  intros h_profit h_tom_investment h_tom_months h_jose_investment h_jose_months h_gcd_val h_total_ratio
  -- the proof would go here
  sorry

end jose_share_of_profit_l159_159196


namespace realNumbersGreaterThan8IsSet_l159_159370

-- Definitions based on conditions:
def verySmallNumbers : Type := {x : ℝ // sorry} -- Need to define what very small numbers would be
def interestingBooks : Type := sorry -- Need to define what interesting books would be
def realNumbersGreaterThan8 : Set ℝ := { x : ℝ | x > 8 }
def tallPeople : Type := sorry -- Need to define what tall people would be

-- Main theorem: Real numbers greater than 8 can form a set
theorem realNumbersGreaterThan8IsSet : Set ℝ :=
  realNumbersGreaterThan8

end realNumbersGreaterThan8IsSet_l159_159370


namespace discount_rate_on_pony_jeans_l159_159342

theorem discount_rate_on_pony_jeans 
  (F P : ℝ) 
  (H1 : F + P = 22) 
  (H2 : 45 * F + 36 * P = 882) : 
  P = 12 :=
by
  sorry

end discount_rate_on_pony_jeans_l159_159342


namespace situps_together_l159_159412

theorem situps_together (hani_rate diana_rate : ℕ) (diana_situps diana_time hani_situps total_situps : ℕ)
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : diana_situps = 40)
  (h4 : diana_time = diana_situps / diana_rate)
  (h5 : hani_situps = hani_rate * diana_time)
  (h6 : total_situps = diana_situps + hani_situps) : 
  total_situps = 110 :=
sorry

end situps_together_l159_159412


namespace vasya_birthday_was_thursday_l159_159806

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159806


namespace chain_of_tangent_circles_exists_iff_integer_angle_multiple_l159_159298

noncomputable def angle_between_tangent_circles (R₁ R₂ : Circle) (line : Line) : ℝ :=
-- the definition should specify how we get the angle between the tangent circles
sorry

def n_tangent_circles_exist (R₁ R₂ : Circle) (n : ℕ) : Prop :=
-- the definition should specify the existence of a chain of n tangent circles
sorry

theorem chain_of_tangent_circles_exists_iff_integer_angle_multiple 
  (R₁ R₂ : Circle) (n : ℕ) (line : Line) : 
  n_tangent_circles_exist R₁ R₂ n ↔ ∃ k : ℤ, angle_between_tangent_circles R₁ R₂ line = k * (360 / n) :=
sorry

end chain_of_tangent_circles_exists_iff_integer_angle_multiple_l159_159298


namespace min_sum_length_perpendicular_chords_l159_159896

variables {p : ℝ} (h : p > 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * p * (x + p)

theorem min_sum_length_perpendicular_chords (h: p > 0) :
  ∃ (AB CD : ℝ), AB * CD = 1 → |AB| + |CD| = 16 * p := sorry

end min_sum_length_perpendicular_chords_l159_159896


namespace new_radius_of_circle_l159_159972

theorem new_radius_of_circle
  (r_1 : ℝ)
  (A_1 : ℝ := π * r_1^2)
  (r_2 : ℝ)
  (A_2 : ℝ := 0.64 * A_1) 
  (h1 : r_1 = 5) 
  (h2 : A_2 = π * r_2^2) : 
  r_2 = 4 :=
by 
  sorry

end new_radius_of_circle_l159_159972


namespace perpendicular_lines_l159_159399

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, ax + 2 * y + 6 = 0) ∧ (∃ x y : ℝ, x + (a - 1) * y + a^2 - 1 = 0) ∧ (∀ m1 m2 : ℝ, m1 * m2 = -1) →
  a = 2/3 :=
by
  sorry

end perpendicular_lines_l159_159399


namespace part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l159_159665

theorem part_a_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2022 → ∃ k : ℕ, k = 65 :=
sorry

theorem part_b_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2023 → ∃ k : ℕ, k = 65 :=
sorry

end part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l159_159665


namespace makenna_garden_larger_by_160_l159_159613

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def karl_length : ℕ := 22
def karl_width : ℕ := 50
def makenna_length : ℕ := 28
def makenna_width : ℕ := 45

def karl_area : ℕ := area karl_length karl_width
def makenna_area : ℕ := area makenna_length makenna_width

theorem makenna_garden_larger_by_160 :
  makenna_area = karl_area + 160 := by
  sorry

end makenna_garden_larger_by_160_l159_159613


namespace good_horse_catches_up_l159_159433

noncomputable def catch_up_days : ℕ := sorry

theorem good_horse_catches_up (x : ℕ) :
  (∀ (good_horse_speed slow_horse_speed head_start_duration : ℕ),
    good_horse_speed = 200 →
    slow_horse_speed = 120 →
    head_start_duration = 10 →
    200 * x = 120 * x + 120 * 10) →
  catch_up_days = x :=
by
  intro h
  have := h 200 120 10 rfl rfl rfl
  sorry

end good_horse_catches_up_l159_159433


namespace culture_medium_preparation_l159_159270

theorem culture_medium_preparation :
  ∀ (V : ℝ), 0 < V → 
  ∃ (nutrient_broth pure_water saline_water : ℝ),
    nutrient_broth = V / 3 ∧
    pure_water = V * 0.3 ∧
    saline_water = V - (nutrient_broth + pure_water) :=
by
  sorry

end culture_medium_preparation_l159_159270


namespace terry_daily_income_l159_159475

theorem terry_daily_income (T : ℕ) (h1 : ∀ j : ℕ, j = 30) (h2 : 7 * 30 = 210) (h3 : 7 * T - 210 = 42) : T = 36 := 
by
  sorry

end terry_daily_income_l159_159475


namespace prove_y_value_l159_159729

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159729


namespace arithmetic_seq_a10_l159_159027

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a10 (h_arith : arithmetic_sequence a) (h2 : a 3 = 5) (h5 : a 6 = 11) : a 10 = 19 := by
  sorry

end arithmetic_seq_a10_l159_159027


namespace rachel_problems_solved_each_minute_l159_159464

-- Definitions and conditions
def problems_solved_each_minute (x : ℕ) : Prop :=
  let problems_before_bed := 12 * x
  let problems_at_lunch := 16
  let total_problems := problems_before_bed + problems_at_lunch
  total_problems = 76

-- Theorem to be proved
theorem rachel_problems_solved_each_minute : ∃ x : ℕ, problems_solved_each_minute x ∧ x = 5 :=
by
  sorry

end rachel_problems_solved_each_minute_l159_159464


namespace square_of_binomial_l159_159385

theorem square_of_binomial (c : ℝ) (h : c = 3600) :
  ∃ a : ℝ, (x : ℝ) → (x + a)^2 = x^2 + 120 * x + c := by
  sorry

end square_of_binomial_l159_159385


namespace cylinder_volume_l159_159786

theorem cylinder_volume (r h : ℝ) (hr : r = 5) (hh : h = 10) :
    π * r^2 * h = 250 * π := by
  -- We leave the actual proof as sorry for now
  sorry

end cylinder_volume_l159_159786


namespace arithmetic_sequence_ratios_l159_159251

noncomputable def a_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {a_n}
noncomputable def b_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {b_n}
noncomputable def S_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {a_n}
noncomputable def T_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {b_n}

theorem arithmetic_sequence_ratios :
  (∀ n : ℕ, 0 < n → S_n n / T_n n = (7 * n + 1) / (4 * n + 27)) →
  (a_n 7 / b_n 7 = 92 / 79) :=
by
  intros h
  sorry

end arithmetic_sequence_ratios_l159_159251


namespace trig_identity_proof_l159_159814

noncomputable def value_expr : ℝ :=
  (2 * Real.cos (10 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.sin (70 * Real.pi / 180)

theorem trig_identity_proof : value_expr = Real.sqrt 3 :=
by
  sorry

end trig_identity_proof_l159_159814


namespace slant_height_of_cone_l159_159573

theorem slant_height_of_cone (r : ℝ) (h : ℝ) (s : ℝ) (unfolds_to_semicircle : s = π) (base_radius : r = 1) : s = 2 :=
by
  sorry

end slant_height_of_cone_l159_159573


namespace remainder_b100_mod_81_l159_159450

def b (n : ℕ) := 7^n + 9^n

theorem remainder_b100_mod_81 : (b 100) % 81 = 38 := by
  sorry

end remainder_b100_mod_81_l159_159450


namespace infinite_seq_condition_l159_159384

theorem infinite_seq_condition (x : ℕ → ℕ) (n m : ℕ) : 
  (∀ i, x i = 0 → x (i + m) = 1) → 
  (∀ i, x i = 1 → x (i + n) = 0) → 
  ∃ d p q : ℕ, n = 2^d * p ∧ m = 2^d * q ∧ p % 2 = 1 ∧ q % 2 = 1  :=
by 
  intros h1 h2 
  sorry

end infinite_seq_condition_l159_159384


namespace gcd_lcm_product_l159_159821

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159821


namespace probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l159_159780

noncomputable def qualification_rate : ℝ := 0.8
def probability_both_qualified (rate : ℝ) : ℝ := rate * rate
def unqualified_rate (rate : ℝ) : ℝ := 1 - rate
def expected_days (n : ℕ) (p : ℝ) : ℝ := n * p

theorem probability_of_both_qualified_bottles : 
  probability_both_qualified qualification_rate = 0.64 :=
by sorry

theorem expected_number_of_days_with_unqualified_milk :
  expected_days 3 (unqualified_rate qualification_rate) = 1.08 :=
by sorry

end probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l159_159780


namespace weight_of_b_l159_159480

variable {A B C : ℤ}

def condition1 (A B C : ℤ) : Prop := (A + B + C) / 3 = 45
def condition2 (A B : ℤ) : Prop := (A + B) / 2 = 42
def condition3 (B C : ℤ) : Prop := (B + C) / 2 = 43

theorem weight_of_b (A B C : ℤ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B) 
  (h3 : condition3 B C) : 
  B = 35 := 
by
  sorry

end weight_of_b_l159_159480


namespace prove_y_value_l159_159727

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159727


namespace max_product_l159_159757

-- Given an integer n and a table of dimension n x n, the maximal value of 
-- the product of n numbers chosen such that no two numbers are from the same row or column 
-- is (n-1)^n * (n+1)!
theorem max_product (n : ℕ) (h : 0 < n) :
  ∃ P : Fin n → ℕ, (∀ i j : Fin n, i ≠ j → P i ≠ P j ∧ P i = n * i.val + (n - i.val - 1)) ∧
    (Finset.univ.prod (λ i : Fin n, P i) = (n - 1)^n * (n + 1)!) := 
sorry

end max_product_l159_159757


namespace inv_proportion_through_point_l159_159026

theorem inv_proportion_through_point (m : ℝ) (x y : ℝ) (h1 : y = m / x) (h2 : x = 2) (h3 : y = -3) : m = -6 := by
  sorry

end inv_proportion_through_point_l159_159026


namespace probability_two_red_two_blue_l159_159668

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (12.choose 2 * 8.choose 2) / (20.choose 4) = 168 / 323 :=
  sorry

end probability_two_red_two_blue_l159_159668


namespace rhombus_perimeter_l159_159962

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l159_159962


namespace clairaut_equation_solution_l159_159280

open Real

noncomputable def clairaut_solution (f : ℝ → ℝ) (C : ℝ) : Prop :=
  (∀ x, f x = C * x + 1/(2 * C)) ∨ (∀ x, (f x)^2 = 2 * x)

theorem clairaut_equation_solution (y : ℝ → ℝ) :
  (∀ x, y x = x * (deriv y x) + 1/(2 * (deriv y x))) →
  ∃ C, clairaut_solution y C :=
sorry

end clairaut_equation_solution_l159_159280


namespace sum_of_triangle_angles_sin_halves_leq_one_l159_159036

theorem sum_of_triangle_angles_sin_halves_leq_one (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC : A + B + C = Real.pi) : 
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := 
sorry 

end sum_of_triangle_angles_sin_halves_leq_one_l159_159036


namespace possible_denominators_of_repeating_decimal_l159_159471

theorem possible_denominators_of_repeating_decimal :
  ∃ (D : Nat), D = 6 ∧ ∀ (a b : Fin 10),
  ¬(a = 0 ∧ b = 0) →
  let frac := (a.val * 10 + b.val) / 99 in 
  let denom := (frac.num.gcd frac.denom) in
  denom.count_divisors = D := 
sorry

end possible_denominators_of_repeating_decimal_l159_159471


namespace James_balloons_l159_159160

theorem James_balloons (A J : ℕ) (h1 : A = 513) (h2 : J = A + 208) : J = 721 :=
by {
  sorry
}

end James_balloons_l159_159160


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l159_159070

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (8 ∣ n) ∧ nat.digits 10 n.sum = 24 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ (8 ∣ m) ∧ nat.digits 10 m.sum = 24 → m ≤ n :=
begin
  sorry
end

end largest_three_digit_multiple_of_8_with_digit_sum_24_l159_159070


namespace vasya_birthday_l159_159799

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159799


namespace calc_subtract_l159_159377

-- Define the repeating decimal
def repeating_decimal := (11 : ℚ) / 9

-- Define the problem statement
theorem calc_subtract : 3 - repeating_decimal = (16 : ℚ) / 9 := by
  sorry

end calc_subtract_l159_159377


namespace part_one_part_two_l159_159522

def discriminant (a b c : ℝ) := b^2 - 4*a*c

theorem part_one (a : ℝ) (h : 0 < a) : 
  (∃ x : ℝ, ax^2 - 3*x + 2 < 0) ↔ 0 < a ∧ a < 9/8 := 
by 
  sorry

theorem part_two (a x : ℝ) : 
  (ax^2 - 3*x + 2 > ax - 1) ↔ 
  (a = 0 ∧ x < 1) ∨ 
  (a < 0 ∧ 3/a < x ∧ x < 1) ∨ 
  (0 < a ∧ (a > 3 ∧ (x < 3/a ∨ x > 1)) ∨ (a = 3 ∧ x ≠ 1) ∨ (0 < a ∧ a < 3 ∧ (x < 1 ∨ x > 3/a))) :=
by 
  sorry

end part_one_part_two_l159_159522


namespace dogsled_course_distance_l159_159189

theorem dogsled_course_distance 
    (t : ℕ)  -- time taken by Team B
    (speed_B : ℕ := 20)  -- average speed of Team B
    (speed_A : ℕ := 25)  -- average speed of Team A
    (tA_eq_tB_minus_3 : t - 3 = tA)  -- Team A’s time relation
    (speedA_eq_speedB_plus_5 : speed_A = speed_B + 5)  -- Team A's average speed in relation to Team B’s average speed
    (distance_eq : speed_B * t = speed_A * (t - 3))  -- Distance equality condition
    (t_eq_15 : t = 15)  -- Time taken by Team B to finish
    :
    (speed_B * t = 300) :=   -- Distance of the course
by
  sorry

end dogsled_course_distance_l159_159189


namespace intervals_of_monotonicity_and_extremum_l159_159576

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extremum :
  (∀ x, x < 1 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x < 0) ∧
  (∃ c, c = 1 ∧ IsLocalMax f c ∧ f c = Real.exp (-1)) :=
by
  sorry

end intervals_of_monotonicity_and_extremum_l159_159576


namespace martin_total_distance_l159_159947

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l159_159947


namespace total_payment_correct_l159_159458

-- Define the conditions for each singer
def firstSingerPayment : ℝ := 2 * 25
def secondSingerPayment : ℝ := 3 * 35
def thirdSingerPayment : ℝ := 4 * 20
def fourthSingerPayment : ℝ := 2.5 * 30

def firstSingerTip : ℝ := 0.15 * firstSingerPayment
def secondSingerTip : ℝ := 0.20 * secondSingerPayment
def thirdSingerTip : ℝ := 0.25 * thirdSingerPayment
def fourthSingerTip : ℝ := 0.18 * fourthSingerPayment

def firstSingerTotal : ℝ := firstSingerPayment + firstSingerTip
def secondSingerTotal : ℝ := secondSingerPayment + secondSingerTip
def thirdSingerTotal : ℝ := thirdSingerPayment + thirdSingerTip
def fourthSingerTotal : ℝ := fourthSingerPayment + fourthSingerTip

-- Define the total amount paid
def totalPayment : ℝ := firstSingerTotal + secondSingerTotal + thirdSingerTotal + fourthSingerTotal

-- The proof problem: Prove the total amount paid
theorem total_payment_correct : totalPayment = 372 := by
  sorry

end total_payment_correct_l159_159458


namespace measure_of_angle_D_in_scalene_triangle_l159_159067

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

end measure_of_angle_D_in_scalene_triangle_l159_159067


namespace savings_example_l159_159539

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

end savings_example_l159_159539


namespace sum_of_digits_of_largest_five_digit_number_with_product_120_l159_159352

theorem sum_of_digits_of_largest_five_digit_number_with_product_120 
  (a b c d e : ℕ)
  (h_digit_a : 0 ≤ a ∧ a ≤ 9)
  (h_digit_b : 0 ≤ b ∧ b ≤ 9)
  (h_digit_c : 0 ≤ c ∧ c ≤ 9)
  (h_digit_d : 0 ≤ d ∧ d ≤ 9)
  (h_digit_e : 0 ≤ e ∧ e ≤ 9)
  (h_product : a * b * c * d * e = 120)
  (h_largest : ∀ f g h i j : ℕ, 
                0 ≤ f ∧ f ≤ 9 → 
                0 ≤ g ∧ g ≤ 9 → 
                0 ≤ h ∧ h ≤ 9 → 
                0 ≤ i ∧ i ≤ 9 → 
                0 ≤ j ∧ j ≤ 9 → 
                f * g * h * i * j = 120 → 
                f * 10000 + g * 1000 + h * 100 + i * 10 + j ≤ a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  a + b + c + d + e = 18 :=
by sorry

end sum_of_digits_of_largest_five_digit_number_with_product_120_l159_159352


namespace calc_root_difference_l159_159691

theorem calc_root_difference :
  ((81: ℝ)^(1/4) + (32: ℝ)^(1/5) - (49: ℝ)^(1/2)) = -2 :=
by
  have h1 : (81: ℝ)^(1/4) = 3 := by sorry
  have h2 : (32: ℝ)^(1/5) = 2 := by sorry
  have h3 : (49: ℝ)^(1/2) = 7 := by sorry
  rw [h1, h2, h3]
  norm_num

end calc_root_difference_l159_159691


namespace part1_part2_l159_159363

-- Part 1: Showing x range for increasing actual processing fee
theorem part1 (x : ℝ) : (x ≤ 99.5) ↔ (∀ y, 0 < y → y ≤ x → (1/2) * Real.log (2 * y + 1) - y / 200 ≤ (1/2) * Real.log (2 * (y + 0.1) + 1) - (y + 0.1) / 200) :=
sorry

-- Part 2: Showing m range for no losses in processing production
theorem part2 (m x : ℝ) (hx : x ∈ Set.Icc 10 20) : 
  (m ≤ (Real.log 41 - 2) / 40) ↔ ((1/2) * Real.log (2 * x + 1) - m * x ≥ (1/20) * x) :=
sorry

end part1_part2_l159_159363


namespace xy_sum_cases_l159_159587

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l159_159587


namespace all_numbers_non_positive_l159_159159

theorem all_numbers_non_positive 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → (a (k - 1) - 2 * a k + a (k + 1) ≥ 0)) : 
  ∀ k, 0 ≤ k → k ≤ n → a k ≤ 0 := 
by 
  sorry

end all_numbers_non_positive_l159_159159


namespace base8_base6_eq_l159_159048

-- Defining the base representations
def base8 (A C : ℕ) := 8 * A + C
def base6 (C A : ℕ) := 6 * C + A

-- The main theorem stating that the integer is 47 in base 10 given the conditions
theorem base8_base6_eq (A C : ℕ) (hAC: base8 A C = base6 C A) (hA: A = 5) (hC: C = 7) : 
  8 * A + C = 47 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end base8_base6_eq_l159_159048


namespace sum_of_ages_l159_159936

-- Definitions based on conditions
variables (J S : ℝ) -- J and S are real numbers

-- First condition: Jane is five years older than Sarah
def jane_older_than_sarah := J = S + 5

-- Second condition: Nine years from now, Jane will be three times as old as Sarah was three years ago
def future_condition := J + 9 = 3 * (S - 3)

-- Conclusion to prove
theorem sum_of_ages (h1 : jane_older_than_sarah J S) (h2 : future_condition J S) : J + S = 28 :=
by
  sorry

end sum_of_ages_l159_159936


namespace yevgeniy_age_2014_l159_159627

theorem yevgeniy_age_2014 (birth_year : ℕ) (h1 : birth_year = 1900 + (birth_year % 100))
  (h2 : 2011 - birth_year = (birth_year / 1000) + ((birth_year % 1000) / 100) + ((birth_year % 100) / 10) + (birth_year % 10)) :
  2014 - birth_year = 23 :=
by
  sorry

end yevgeniy_age_2014_l159_159627


namespace value_of_expression_l159_159596

variables (x y z : ℝ)

axiom eq1 : 3 * x - 4 * y - 2 * z = 0
axiom eq2 : 2 * x + 6 * y - 21 * z = 0
axiom z_ne_zero : z ≠ 0

theorem value_of_expression : (x^2 + 4 * x * y) / (y^2 + z^2) = 7 :=
sorry

end value_of_expression_l159_159596


namespace pinky_pies_count_l159_159632

theorem pinky_pies_count (helen_pies : ℕ) (total_pies : ℕ) (h1 : helen_pies = 56) (h2 : total_pies = 203) : 
  total_pies - helen_pies = 147 := by
  sorry

end pinky_pies_count_l159_159632


namespace goose_survived_first_year_l159_159293

theorem goose_survived_first_year (total_eggs : ℕ) (eggs_hatched_ratio : ℚ) (first_month_survival_ratio : ℚ) 
  (first_year_no_survival_ratio : ℚ) 
  (eggs_hatched_ratio_eq : eggs_hatched_ratio = 2/3) 
  (first_month_survival_ratio_eq : first_month_survival_ratio = 3/4)
  (first_year_no_survival_ratio_eq : first_year_no_survival_ratio = 3/5)
  (total_eggs_eq : total_eggs = 500) :
  ∃ (survived_first_year : ℕ), survived_first_year = 100 :=
by
  sorry

end goose_survived_first_year_l159_159293


namespace number_of_younger_siblings_l159_159766

-- Definitions based on the problem conditions
def Nicole_cards : ℕ := 400
def Cindy_cards : ℕ := 2 * Nicole_cards
def Combined_cards : ℕ := Nicole_cards + Cindy_cards
def Rex_cards : ℕ := Combined_cards / 2
def Rex_remaining_cards : ℕ := 150
def Total_shares : ℕ := Rex_cards / Rex_remaining_cards
def Rex_share : ℕ := 1

-- The theorem to prove how many younger siblings Rex has
theorem number_of_younger_siblings :
  Total_shares - Rex_share = 3 :=
  by
    sorry

end number_of_younger_siblings_l159_159766


namespace smallest_number_with_eight_factors_l159_159512

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l159_159512


namespace compute_expr_l159_159218

theorem compute_expr :
  ((π - 3.14)^0 + (-0.125)^2008 * 8^2008) = 2 := 
by 
  sorry

end compute_expr_l159_159218


namespace parallel_lines_a_l159_159252
-- Import necessary libraries

-- Define the given conditions and the main statement
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 2 = 0 → 3 * x + (a + 2) * y + 1 = 0) →
  (a = -3 ∨ a = 1) :=
by
  -- Place the proof here
  sorry

end parallel_lines_a_l159_159252


namespace clubs_students_equal_l159_159432

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l159_159432


namespace lemons_for_10_gallons_l159_159837

noncomputable def lemon_proportion : Prop :=
  ∃ x : ℝ, (36 / 48) = (x / 10) ∧ x = 7.5

theorem lemons_for_10_gallons : lemon_proportion :=
by
  sorry

end lemons_for_10_gallons_l159_159837


namespace question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l159_159662

theorem question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1
    (a b c d : ℤ)
    (h1 : a + b = 11)
    (h2 : b + c = 9)
    (h3 : c + d = 3)
    : a + d = -1 :=
by
  sorry

end question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l159_159662


namespace adam_earnings_per_lawn_l159_159092

theorem adam_earnings_per_lawn (total_lawns : ℕ) (forgot_lawns : ℕ) (total_earnings : ℕ) :
  total_lawns = 12 →
  forgot_lawns = 8 →
  total_earnings = 36 →
  (total_earnings / (total_lawns - forgot_lawns)) = 9 :=
by
  intros h1 h2 h3
  sorry

end adam_earnings_per_lawn_l159_159092


namespace race_lead_distance_l159_159743

theorem race_lead_distance :
  ∀ (d12 d13 : ℝ) (s1 s2 s3 t : ℝ), 
  d12 = 2 →
  d13 = 4 →
  t > 0 →
  s1 = (d12 / t + s2) →
  s1 = (d13 / t + s3) →
  s2 * t - s3 * t = 2.5 :=
by
  sorry

end race_lead_distance_l159_159743


namespace brick_height_correct_l159_159496

-- Definitions
def wall_length : ℝ := 8
def wall_height : ℝ := 6
def wall_thickness : ℝ := 0.02 -- converted from 2 cm to meters
def brick_length : ℝ := 0.05 -- converted from 5 cm to meters
def brick_width : ℝ := 0.11 -- converted from 11 cm to meters
def brick_height : ℝ := 0.06 -- converted from 6 cm to meters
def number_of_bricks : ℝ := 2909.090909090909

-- Statement to prove
theorem brick_height_correct : brick_height = 0.06 := by
  sorry

end brick_height_correct_l159_159496


namespace correct_sum_l159_159170

theorem correct_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 4) (h3 : x * y = 98) : x + y = 18 := 
by
  sorry

end correct_sum_l159_159170


namespace power_function_at_16_l159_159890

theorem power_function_at_16 :
  ∃ (α : ℝ), (∀ x : ℝ, f x = x ^ α) ∧ (f 4 = 2) → (f 16 = 4) :=
by
  sorry

end power_function_at_16_l159_159890


namespace min_AB_CD_l159_159895

theorem min_AB_CD {p : ℝ} (p_pos : p > 0) :
  ∀ (A B C D : ℝ × ℝ), on_parabola A B C D  &&
  mutually_perpendicular A B C D -> passing_through_origin A B C D -> 
  |AB|  + |CD | = 16 * p.
sorry

end min_AB_CD_l159_159895


namespace sum_of_numbers_odd_probability_l159_159517

namespace ProbabilityProblem

/-- 
  Given a biased die where the probability of rolling an even number is 
  twice the probability of rolling an odd number, and rolling the die three times,
  the probability that the sum of the numbers rolled is odd is 13/27.
-/
theorem sum_of_numbers_odd_probability :
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let prob_all_odd := (p_odd) ^ 3
  let prob_one_odd_two_even := 3 * (p_odd) * (p_even) ^ 2
  prob_all_odd + prob_one_odd_two_even = 13 / 27 :=
by
  sorry

end sum_of_numbers_odd_probability_l159_159517


namespace S_10_value_l159_159488

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (a 1 + a n) / 2

theorem S_10_value (a : ℕ → ℕ) (h1 : a 2 = 3) (h2 : a 9 = 17) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) : 
  S 10 a = 100 := 
by
  sorry

end S_10_value_l159_159488


namespace quadratic_root_neg3_l159_159953

theorem quadratic_root_neg3 : ∃ x : ℝ, x^2 - 9 = 0 ∧ (x = -3) :=
by
  sorry

end quadratic_root_neg3_l159_159953


namespace sale_in_fifth_month_condition_l159_159084

theorem sale_in_fifth_month_condition 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (avg_sale : ℕ)
  (n_months : ℕ)
  (total_sales : ℕ)
  (first_four_sales_and_sixth : ℕ) :
  sale1 = 6435 → 
  sale2 = 6927 → 
  sale3 = 6855 → 
  sale4 = 7230 → 
  sale6 = 6791 → 
  avg_sale = 6800 → 
  n_months = 6 → 
  total_sales = avg_sale * n_months → 
  first_four_sales_and_sixth = sale1 + sale2 + sale3 + sale4 + sale6 → 
  ∃ sale5, sale5 = total_sales - first_four_sales_and_sixth ∧ sale5 = 6562 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end sale_in_fifth_month_condition_l159_159084


namespace find_constants_a_b_l159_159870

variables (x a b : ℝ)

theorem find_constants_a_b (h : (x - a) / (x + b) = (x^2 - 45 * x + 504) / (x^2 + 66 * x - 1080)) :
  a + b = 48 :=
sorry

end find_constants_a_b_l159_159870


namespace permutations_of_six_attractions_is_720_l159_159436

-- Define the number of attractions
def num_attractions : ℕ := 6

-- State the theorem to be proven
theorem permutations_of_six_attractions_is_720 : (num_attractions.factorial = 720) :=
by {
  sorry
}

end permutations_of_six_attractions_is_720_l159_159436


namespace smallest_integer_with_eight_factors_l159_159510

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l159_159510


namespace c_work_time_l159_159339

theorem c_work_time (A B C : ℝ) 
  (h1 : A + B = 1/10) 
  (h2 : B + C = 1/5) 
  (h3 : C + A = 1/15) : 
  C = 1/12 :=
by
  -- Proof will go here
  sorry

end c_work_time_l159_159339


namespace monochromatic_triangle_l159_159460

def R₃ (n : ℕ) : ℕ := sorry

theorem monochromatic_triangle {n : ℕ} (h1 : R₃ 2 = 6)
  (h2 : ∀ n, R₃ (n + 1) ≤ (n + 1) * R₃ n - n + 1) :
  R₃ n ≤ 3 * Nat.factorial n :=
by
  induction n with
  | zero => sorry -- base case proof
  | succ n ih => sorry -- inductive step proof

end monochromatic_triangle_l159_159460


namespace remaining_number_larger_than_4_l159_159477

theorem remaining_number_larger_than_4 (m : ℕ) (h : 2 ≤ m) (a : ℚ) (b : ℚ) (h_sum_inv : (1 : ℚ) - 1 / (2 * m + 1 : ℚ) = 3 / 4 + 1 / b) :
  b > 4 :=
by sorry

end remaining_number_larger_than_4_l159_159477


namespace integral_log_eq_ln2_l159_159000

theorem integral_log_eq_ln2 :
  ∫ x in (0 : ℝ)..(1 : ℝ), (1 / (x + 1)) = Real.log 2 :=
by
  sorry

end integral_log_eq_ln2_l159_159000


namespace clubsuit_commute_l159_159112

-- Define the operation a ♣ b = a^3 * b - a * b^3
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the proposition to prove
theorem clubsuit_commute (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end clubsuit_commute_l159_159112


namespace largest_three_digit_multiple_of_8_with_sum_24_l159_159071

theorem largest_three_digit_multiple_of_8_with_sum_24 :
  ∃ n : ℕ, (n ≥ 100 ∧ n < 1000) ∧ (∃ k, n = 8 * k) ∧ (n.digits.sum = 24) ∧
           ∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (∃ k', m = 8 * k') ∧ (m.digits.sum = 24) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_8_with_sum_24_l159_159071


namespace length_of_each_glass_pane_l159_159212

theorem length_of_each_glass_pane (panes : ℕ) (width : ℕ) (total_area : ℕ) 
    (H_panes : panes = 8) (H_width : width = 8) (H_total_area : total_area = 768) : 
    ∃ length : ℕ, length = 12 := by
  sorry

end length_of_each_glass_pane_l159_159212


namespace math_olympiad_proof_l159_159933

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l159_159933


namespace martin_total_distance_l159_159950

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l159_159950


namespace markers_multiple_of_4_l159_159457

-- Definitions corresponding to conditions
def Lisa_has_12_coloring_books := 12
def Lisa_has_36_crayons := 36
def greatest_number_baskets := 4

-- Theorem statement
theorem markers_multiple_of_4
    (h1 : Lisa_has_12_coloring_books = 12)
    (h2 : Lisa_has_36_crayons = 36)
    (h3 : greatest_number_baskets = 4) :
    ∃ (M : ℕ), M % 4 = 0 :=
by
  sorry

end markers_multiple_of_4_l159_159457


namespace coronavirus_transmission_l159_159778

theorem coronavirus_transmission (x : ℝ) 
  (H: (1 + x) ^ 2 = 225) : (1 + x) ^ 2 = 225 :=
  by
    sorry

end coronavirus_transmission_l159_159778


namespace olympiad_scores_l159_159918

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l159_159918


namespace max_ratio_three_digit_sum_l159_159332

theorem max_ratio_three_digit_sum (N a b c : ℕ) (hN : N = 100 * a + 10 * b + c) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) :
  (∀ (N' a' b' c' : ℕ), N' = 100 * a' + 10 * b' + c' → 1 ≤ a' → b' ≤ 9 → c' ≤ 9 → (N' : ℚ) / (a' + b' + c') ≤ 100) :=
sorry

end max_ratio_three_digit_sum_l159_159332


namespace smallest_integer_with_eight_factors_l159_159514

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l159_159514


namespace canned_food_total_bins_l159_159227

theorem canned_food_total_bins :
  let soup_bins := 0.125
  let vegetable_bins := 0.125
  let pasta_bins := 0.5
  soup_bins + vegetable_bins + pasta_bins = 0.75 := 
by
  sorry

end canned_food_total_bins_l159_159227


namespace average_after_19_innings_is_23_l159_159195

-- Definitions for the conditions given in the problem
variables {A : ℝ} -- Let A be the average score before the 19th inning

-- Conditions: The cricketer scored 95 runs in the 19th inning and his average increased by 4 runs.
def total_runs_after_18_innings (A : ℝ) : ℝ := 18 * A
def total_runs_after_19th_inning (A : ℝ) : ℝ := total_runs_after_18_innings A + 95
def new_average_after_19_innings (A : ℝ) : ℝ := A + 4

-- The statement of the problem as a Lean theorem
theorem average_after_19_innings_is_23 :
  (18 * A + 95) / 19 = A + 4 → A = 19 → (A + 4) = 23 :=
by
  intros hA h_avg_increased
  sorry

end average_after_19_innings_is_23_l159_159195


namespace find_m_l159_159880

def vector_parallel (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem find_m
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (2, -1))
  (h : vector_parallel a (b.1 - a.1, b.2 - a.2)) :
  m = -2 :=
by
  sorry

end find_m_l159_159880


namespace factor_expression_l159_159549

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l159_159549


namespace smallest_integer_with_eight_factors_l159_159501

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l159_159501


namespace Vasya_birthday_on_Thursday_l159_159801

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159801


namespace olympiad_scores_l159_159919

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l159_159919


namespace no_real_solution_arithmetic_progression_l159_159224

theorem no_real_solution_arithmetic_progression :
  ∀ (a b : ℝ), ¬ (12, a, b, a * b) form_arithmetic_progression =
  ∀ (a b : ℝ), ¬ (2 * b = 12 + b + b + a * b) :=
by
  intro a b 
  sorry

end no_real_solution_arithmetic_progression_l159_159224


namespace sum_ap_series_l159_159237

-- Definition of the arithmetic progression sum for given parameters
def ap_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Specific sum calculation for given p
def S_p (p : ℕ) : ℕ :=
  ap_sum p (2 * p - 1) 40

-- Total sum from p = 1 to p = 10
def total_sum : ℕ :=
  (Finset.range 10).sum (λ i => S_p (i + 1))

-- The theorem stating the desired proof
theorem sum_ap_series : total_sum = 80200 := by
  sorry

end sum_ap_series_l159_159237


namespace vectors_parallel_l159_159898

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end vectors_parallel_l159_159898


namespace vasya_birthday_day_l159_159795

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159795


namespace total_selection_methods_l159_159849

def num_courses_group_A := 3
def num_courses_group_B := 4
def total_courses_selected := 3

theorem total_selection_methods 
  (at_least_one_from_each : num_courses_group_A > 0 ∧ num_courses_group_B > 0)
  (total_courses : total_courses_selected = 3) :
  ∃ N, N = 30 :=
sorry

end total_selection_methods_l159_159849


namespace find_z_satisfying_det_eq_l159_159113

namespace MatrixProblem

open Complex -- Use the Complex module

def det (a b c d : ℂ) : ℂ := a * d - b * c

theorem find_z_satisfying_det_eq : 
  ∃ z : ℂ, det 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I := 
by
  have h : det 1 (-1) (3 - Complex.I) ((3 - Complex.I) * Complex.I) = 4 + 2 * Complex.I := sorry
  exact ⟨3 - Complex.I, h, rfl⟩
end MatrixProblem

end find_z_satisfying_det_eq_l159_159113


namespace wholesale_cost_per_bag_l159_159126

theorem wholesale_cost_per_bag (W : ℝ) (h1 : 1.12 * W = 28) : W = 25 :=
sorry

end wholesale_cost_per_bag_l159_159126


namespace least_n_ge_100_divides_sum_of_powers_l159_159333

theorem least_n_ge_100_divides_sum_of_powers (n : ℕ) (h₁ : n ≥ 100) :
    77 ∣ (Finset.sum (Finset.range (n + 1)) (λ k => 2^k) - 1) ↔ n = 119 :=
by
  sorry

end least_n_ge_100_divides_sum_of_powers_l159_159333


namespace distribute_apples_l159_159687

theorem distribute_apples :
  let a, b, c : ℕ in
  (a >= 3) ∧ (b >= 3) ∧ (c >= 3) ∧ (a + b + c = 26) →
  fintype.card { (a', b', c' : ℕ) // a' + b' + c' = 17 } = 171 :=
by
  sorry

end distribute_apples_l159_159687


namespace g_possible_values_l159_159552

noncomputable def g (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((x - 1) / (x + 1)) + Real.arctan (1 / x)

theorem g_possible_values (x : ℝ) (hx₁ : x ≠ 0) (hx₂ : x ≠ -1) (hx₃ : x ≠ 1) :
  g x = (Real.pi / 4) ∨ g x = (5 * Real.pi / 4) :=
sorry

end g_possible_values_l159_159552


namespace reporters_local_politics_percentage_l159_159677

theorem reporters_local_politics_percentage
  (T : ℕ) -- Total number of reporters
  (P : ℝ) -- Percentage of reporters covering politics
  (h1 : 30 / 100 * (P / 100) * T = (P / 100 - 0.7 * (P / 100)) * T)
  (h2 : 92.85714285714286 / 100 * T = (1 - P / 100) * T):
  (0.7 * (P / 100) * T) / T = 5 / 100 :=
by
  sorry

end reporters_local_politics_percentage_l159_159677


namespace clubs_students_equal_l159_159430

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l159_159430


namespace sum_of_xy_eq_20_l159_159586

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l159_159586


namespace hike_distance_l159_159032

theorem hike_distance :
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  stream_to_meadow = 0.4 :=
by
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  show stream_to_meadow = 0.4
  sorry

end hike_distance_l159_159032


namespace largest_awesome_prime_l159_159721

def is_awesome_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ q : ℕ, 1 ≤ q ∧ q < p → Nat.Prime (p + 2 * q)

theorem largest_awesome_prime : ∀ p : ℕ, is_awesome_prime p → p ≤ 3 :=
by
  intro p hp
  sorry

end largest_awesome_prime_l159_159721


namespace sqrt_two_minus_one_pow_zero_l159_159350

theorem sqrt_two_minus_one_pow_zero : (Real.sqrt 2 - 1)^0 = 1 := by
  sorry

end sqrt_two_minus_one_pow_zero_l159_159350


namespace evaluate_expression_l159_159697

theorem evaluate_expression (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l159_159697


namespace smaller_rectangle_perimeter_l159_159847

def problem_conditions (a b : ℝ) : Prop :=
  2 * (a + b) = 96 ∧ 
  8 * b + 11 * a = 342 ∧
  a + b = 48 ∧ 
  (a * (b - 1) <= 0 ∧ b * (a - 1) <= 0 ∧ a > 0 ∧ b > 0)

theorem smaller_rectangle_perimeter (a b : ℝ) (hab : problem_conditions a b) :
  2 * (a / 12 + b / 9) = 9 :=
  sorry

end smaller_rectangle_perimeter_l159_159847


namespace vector_parallel_l159_159719

theorem vector_parallel (x : ℝ) : ∃ x, (1, x) = k * (-2, 3) → x = -3 / 2 :=
by 
  sorry

end vector_parallel_l159_159719


namespace num_students_third_school_l159_159983

variable (x : ℕ)

def num_students_condition := (2 * (x + 40) + (x + 40) + x = 920)

theorem num_students_third_school (h : num_students_condition x) : x = 200 :=
sorry

end num_students_third_school_l159_159983


namespace time_to_fill_tank_l159_159348

-- Define the rates of the pipes
def rate_first_fill : ℚ := 1 / 15
def rate_second_fill : ℚ := 1 / 15
def rate_outlet_empty : ℚ := -1 / 45

-- Define the combined rate
def combined_rate : ℚ := rate_first_fill + rate_second_fill + rate_outlet_empty

-- Define the time to fill the tank
def fill_time (rate : ℚ) : ℚ := 1 / rate

theorem time_to_fill_tank : fill_time combined_rate = 9 := 
by 
  -- Proof omitted
  sorry

end time_to_fill_tank_l159_159348


namespace percent_equality_l159_159829

theorem percent_equality :
  (1 / 4 : ℝ) * 100 = (10 / 100 : ℝ) * 250 :=
by
  sorry

end percent_equality_l159_159829


namespace reciprocals_of_product_one_l159_159024

theorem reciprocals_of_product_one (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x :=
by 
  sorry

end reciprocals_of_product_one_l159_159024


namespace sqrt_expression_eq_l159_159693

theorem sqrt_expression_eq : 
  (Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3)) = -Real.sqrt 3 + 4 := 
by
  sorry

end sqrt_expression_eq_l159_159693


namespace Vasya_birthday_on_Thursday_l159_159802

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159802


namespace is_happy_number_512_l159_159910

def happy_number (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 8 * k

theorem is_happy_number_512 : happy_number 512 := 
  by
  unfold happy_number
  use 64
  sorry

end is_happy_number_512_l159_159910


namespace hikers_rate_l159_159530

-- Define the conditions from the problem
variables (R : ℝ) (time_up time_down : ℝ) (distance_down : ℝ)

-- Conditions given in the problem
axiom condition1 : time_up = 2
axiom condition2 : time_down = 2
axiom condition3 : distance_down = 9
axiom condition4 : (distance_down / time_down) = 1.5 * R

-- The proof goal
theorem hikers_rate (h1 : time_up = 2) 
                    (h2 : time_down = 2) 
                    (h3 : distance_down = 9) 
                    (h4 : distance_down / time_down = 1.5 * R) : R = 3 := 
by 
  sorry

end hikers_rate_l159_159530


namespace exists_rat_not_int_add_pow_int_l159_159706

theorem exists_rat_not_int_add_pow_int (n : ℕ) : 
  (odd n ↔ ∃ a b : ℚ, (0 < a ∧ 0 < b ∧ ¬a ∈ ℤ ∧ ¬b ∈ ℤ) ∧ (a + b ∈ ℤ) ∧ (a^n + b^n ∈ ℤ)) :=
sorry

end exists_rat_not_int_add_pow_int_l159_159706


namespace math_olympiad_proof_l159_159931

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end math_olympiad_proof_l159_159931


namespace cost_relationship_l159_159856

variable {α : Type} [LinearOrderedField α]
variables (bananas_cost apples_cost pears_cost : α)

theorem cost_relationship :
  (5 * bananas_cost = 3 * apples_cost) →
  (10 * apples_cost = 6 * pears_cost) →
  (25 * bananas_cost = 9 * pears_cost) := by
  intros h1 h2
  sorry

end cost_relationship_l159_159856


namespace discount_percentage_l159_159068

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

end discount_percentage_l159_159068


namespace brick_length_l159_159081

theorem brick_length (x : ℝ) (brick_width : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (number_of_bricks : ℕ)
  (h_brick : brick_width = 11.25) (h_brick_height : brick_height = 6)
  (h_wall : wall_length = 800) (h_wall_width : wall_width = 600) 
  (h_wall_height : wall_height = 22.5) (h_bricks_number : number_of_bricks = 1280)
  (h_eq : (wall_length * wall_width * wall_height) = (x * brick_width * brick_height) * number_of_bricks) : 
  x = 125 := by
  sorry

end brick_length_l159_159081


namespace olympiad_scores_l159_159915

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l159_159915


namespace number_534n_divisible_by_12_l159_159554

theorem number_534n_divisible_by_12 (n : ℕ) : (5340 + n) % 12 = 0 ↔ n = 0 := by sorry

end number_534n_divisible_by_12_l159_159554


namespace seeds_sum_l159_159338

def Bom_seeds : ℕ := 300

def Gwi_seeds : ℕ := Bom_seeds + 40

def Yeon_seeds : ℕ := 3 * Gwi_seeds

def total_seeds : ℕ := Bom_seeds + Gwi_seeds + Yeon_seeds

theorem seeds_sum : total_seeds = 1660 := by
  sorry

end seeds_sum_l159_159338


namespace quadratic_sum_l159_159486

theorem quadratic_sum (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) → b + c = -106 :=
by
  intro h
  sorry

end quadratic_sum_l159_159486


namespace quadratic_unique_solution_l159_159782

theorem quadratic_unique_solution (a c : ℕ) (h1 : a + c = 29) (h2 : a < c)
  (h3 : (20^2 - 4 * a * c) = 0) : (a, c) = (4, 25) :=
by
  sorry

end quadratic_unique_solution_l159_159782


namespace modular_inverse_expression_l159_159869

-- Definitions of the inverses as given in the conditions
def inv_7_mod_77 : ℤ := 11
def inv_13_mod_77 : ℤ := 6

-- The main theorem stating the equivalence
theorem modular_inverse_expression :
  (3 * inv_7_mod_77 + 9 * inv_13_mod_77) % 77 = 10 :=
by
  sorry

end modular_inverse_expression_l159_159869


namespace base_r_5555_square_palindrome_l159_159646

theorem base_r_5555_square_palindrome (r : ℕ) (a b c d : ℕ) 
  (h1 : r % 2 = 0) 
  (h2 : r >= 18) 
  (h3 : d - c = 2)
  (h4 : ∀ x, (x = 5 * r^3 + 5 * r^2 + 5 * r + 5) → 
    (x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)) : 
  r = 24 := 
sorry

end base_r_5555_square_palindrome_l159_159646


namespace cubic_inches_in_two_cubic_feet_l159_159223

theorem cubic_inches_in_two_cubic_feet :
  (12 ^ 3) * 2 = 3456 := by
  sorry

end cubic_inches_in_two_cubic_feet_l159_159223


namespace simplify_expression_l159_159467

theorem simplify_expression (x : ℤ) : 120 * x - 55 * x = 65 * x := by
  sorry

end simplify_expression_l159_159467


namespace sum_of_xy_eq_20_l159_159584

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l159_159584


namespace solution_set_a_range_m_l159_159408

theorem solution_set_a (a : ℝ) :
  (∀ x : ℝ, |x - a| ≤ 3 ↔ -6 ≤ x ∧ x ≤ 0) ↔ a = -3 :=
by
  sorry

theorem range_m (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + 8| ≥ 2 * m) ↔ m ≤ 5 / 2 :=
by
  sorry

end solution_set_a_range_m_l159_159408


namespace chocolate_cost_first_store_l159_159097

def cost_first_store (x : ℕ) : ℕ := x
def chocolate_promotion_store : ℕ := 2
def savings_in_three_weeks : ℕ := 6
def number_of_chocolates (weeks : ℕ) : ℕ := 2 * weeks

theorem chocolate_cost_first_store :
  ∀ (weeks : ℕ) (x : ℕ), 
    number_of_chocolates weeks = 6 →
    chocolate_promotion_store * number_of_chocolates weeks + savings_in_three_weeks = cost_first_store x * number_of_chocolates weeks →
    cost_first_store x = 3 :=
by
  intros weeks x h1 h2
  sorry

end chocolate_cost_first_store_l159_159097


namespace mountain_climbing_time_proof_l159_159336

noncomputable def mountain_climbing_time (x : ℝ) : ℝ := (x + 2) / 4

theorem mountain_climbing_time_proof (x : ℝ) (h1 : (x / 3 + (x + 2) / 4 = 4)) : mountain_climbing_time x = 2 := by
  -- assume the given conditions and proof steps explicitly
  sorry

end mountain_climbing_time_proof_l159_159336


namespace find_m_of_lcm_conditions_l159_159713

theorem find_m_of_lcm_conditions (m : ℕ) (h_pos : 0 < m)
  (h1 : Int.lcm 18 m = 54)
  (h2 : Int.lcm m 45 = 180) : m = 36 :=
sorry

end find_m_of_lcm_conditions_l159_159713


namespace tan_double_beta_alpha_value_l159_159131

open Real

-- Conditions
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def beta_in_interval (β : ℝ) : Prop := π / 2 < β ∧ β < π
def cos_beta (β : ℝ) : Prop := cos β = -1 / 3
def sin_alpha_plus_beta (α β : ℝ) : Prop := sin (α + β) = (4 - sqrt 2) / 6

-- Proof problem 1: Prove that tan 2β = 4√2 / 7 given the conditions
theorem tan_double_beta (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  tan (2 * β) = (4 * sqrt 2) / 7 :=
by sorry

-- Proof problem 2: Prove that α = π / 4 given the conditions
theorem alpha_value (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  α = π / 4 :=
by sorry

end tan_double_beta_alpha_value_l159_159131


namespace intersection_M_N_l159_159127

def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l159_159127


namespace pear_weight_l159_159650

theorem pear_weight
  (w_apple : ℕ)
  (p_weight_relation : 12 * w_apple = 8 * P + 5400)
  (apple_weight : w_apple = 530) :
  P = 120 :=
by
  -- sorry, proof is omitted as per instructions
  sorry

end pear_weight_l159_159650


namespace solve_for_y_l159_159731

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159731


namespace eq_op_l159_159696

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 + 2 * x - y

-- State the theorem to be proven
theorem eq_op (k : ℝ) : op k (op k k) = k := sorry

end eq_op_l159_159696


namespace circle_arc_and_circumference_l159_159860

theorem circle_arc_and_circumference (C_X : ℝ) (θ_YOZ : ℝ) (C_D : ℝ) (r_X r_D : ℝ) :
  C_X = 100 ∧ θ_YOZ = 150 ∧ r_X = 50 / π ∧ r_D = 25 / π ∧ C_D = 50 →
  (θ_YOZ / 360) * C_X = 500 / 12 ∧ 2 * π * r_D = C_D :=
by sorry

end circle_arc_and_circumference_l159_159860


namespace multiples_7_not_14_l159_159259

theorem multiples_7_not_14 (n : ℕ) : (n < 500) → (n % 7 = 0) → (n % 14 ≠ 0) → ∃ k, (k = 36) :=
by {
  sorry
}

end multiples_7_not_14_l159_159259


namespace simplify_expression_l159_159077

theorem simplify_expression :
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 :=
by
  sorry

end simplify_expression_l159_159077


namespace total_volume_correct_l159_159105

-- Defining the initial conditions
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Given the above conditions, define the total volume of all cubes.
def total_volume_of_all_cubes : ℕ := (carl_cubes * carl_side_length ^ 3) + (kate_cubes * kate_side_length ^ 3)

-- The statement we need to prove
theorem total_volume_correct :
  total_volume_of_all_cubes = 114 :=
by
  -- Skipping the proof with sorry as per the instruction
  sorry

end total_volume_correct_l159_159105


namespace vasya_birthday_l159_159792

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159792


namespace min_tip_percentage_l159_159361

namespace TipCalculation

def mealCost : Float := 35.50
def totalPaid : Float := 37.275
def maxTipPercent : Float := 0.08

theorem min_tip_percentage : ∃ (P : Float), (P / 100 * mealCost = (totalPaid - mealCost)) ∧ (P < maxTipPercent * 100) ∧ (P = 5) := by
  sorry

end TipCalculation

end min_tip_percentage_l159_159361


namespace slices_left_per_person_is_2_l159_159295

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end slices_left_per_person_is_2_l159_159295


namespace intersection_S_T_l159_159616

def S := {x : ℝ | (x - 2) * (x - 3) ≥ 0}
def T := {x : ℝ | x > 0}

theorem intersection_S_T :
  (S ∩ T) = (Set.Ioc 0 2 ∪ Set.Ici 3) :=
by
  sorry

end intersection_S_T_l159_159616


namespace sum_of_ages_l159_159634

variable (S T : ℕ)

theorem sum_of_ages (h1 : S = T + 7) (h2 : S + 10 = 3 * (T - 3)) : S + T = 33 := by
  sorry

end sum_of_ages_l159_159634


namespace arithmetic_sequence_third_term_l159_159911

theorem arithmetic_sequence_third_term (a d : ℤ) (h : a + (a + 4 * d) = 14) : a + 2 * d = 7 := by
  -- We assume the sum of the first and fifth term is 14 and prove that the third term is 7.
  sorry

end arithmetic_sequence_third_term_l159_159911


namespace total_profit_calculation_l159_159042

-- Define the parameters of the problem
def rajan_investment : ℕ := 20000
def rakesh_investment : ℕ := 25000
def mukesh_investment : ℕ := 15000
def rajan_investment_time : ℕ := 12 -- in months
def rakesh_investment_time : ℕ := 4 -- in months
def mukesh_investment_time : ℕ := 8 -- in months
def rajan_final_share : ℕ := 2400

-- Calculation for total profit
def total_profit (rajan_investment rakesh_investment mukesh_investment
                  rajan_investment_time rakesh_investment_time mukesh_investment_time
                  rajan_final_share : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_investment_time
  let rakesh_share := rakesh_investment * rakesh_investment_time
  let mukesh_share := mukesh_investment * mukesh_investment_time
  let total_investment := rajan_share + rakesh_share + mukesh_share
  (rajan_final_share * total_investment) / rajan_share

-- Proof problem statement
theorem total_profit_calculation :
  total_profit rajan_investment rakesh_investment mukesh_investment
               rajan_investment_time rakesh_investment_time mukesh_investment_time
               rajan_final_share = 4600 :=
by sorry

end total_profit_calculation_l159_159042


namespace total_distance_traveled_l159_159943

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l159_159943


namespace total_feet_l159_159681

theorem total_feet (heads hens : ℕ) (h1 : heads = 46) (h2 : hens = 22) : 
  ∃ feet : ℕ, feet = 140 := 
by 
  sorry

end total_feet_l159_159681


namespace find_pq_of_orthogonal_and_equal_magnitudes_l159_159758

noncomputable def vec_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
noncomputable def vec_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_pq_of_orthogonal_and_equal_magnitudes
    (p q : ℝ)
    (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
    (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
    (p, q) = (-29/12, 43/12) :=
by {
  sorry
}

end find_pq_of_orthogonal_and_equal_magnitudes_l159_159758


namespace total_distance_walked_l159_159376

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l159_159376


namespace total_feet_l159_159682

theorem total_feet (H C : ℕ) (h1 : H + C = 48) (h2 : H = 28) : 2 * H + 4 * C = 136 := 
by
  sorry

end total_feet_l159_159682


namespace sin_of_2000_deg_l159_159905

theorem sin_of_2000_deg (a : ℝ) (h : Real.tan (160 * Real.pi / 180) = a) : 
  Real.sin (2000 * Real.pi / 180) = -a / Real.sqrt (1 + a^2) := 
by
  sorry

end sin_of_2000_deg_l159_159905


namespace calculate_expression_l159_159379

theorem calculate_expression : 
  (1 - Real.sqrt 2)^0 + |(2 - Real.sqrt 5)| + (-1)^2022 - (1/3) * Real.sqrt 45 = 0 :=
by
  sorry

end calculate_expression_l159_159379


namespace total_regular_and_diet_soda_bottles_l159_159204

-- Definitions from the conditions
def regular_soda_bottles := 49
def diet_soda_bottles := 40

-- The statement to prove
theorem total_regular_and_diet_soda_bottles :
  regular_soda_bottles + diet_soda_bottles = 89 :=
by
  sorry

end total_regular_and_diet_soda_bottles_l159_159204


namespace prime_range_for_integer_roots_l159_159417

theorem prime_range_for_integer_roots (p : ℕ) (h_prime : Prime p) 
  (h_int_roots : ∃ (a b : ℤ), a + b = -p ∧ a * b = -300 * p) : 
  1 < p ∧ p ≤ 11 :=
sorry

end prime_range_for_integer_roots_l159_159417


namespace additional_dividend_amount_l159_159200

theorem additional_dividend_amount
  (E : ℝ) (Q : ℝ) (expected_extra_per_earnings : ℝ) (half_of_extra_per_earnings_to_dividend : ℝ) 
  (expected : E = 0.80) (quarterly_earnings : Q = 1.10)
  (extra_per_earnings : expected_extra_per_earnings = 0.30)
  (half_dividend : half_of_extra_per_earnings_to_dividend = 0.15):
  Q - E = expected_extra_per_earnings ∧ 
  expected_extra_per_earnings / 2 = half_of_extra_per_earnings_to_dividend :=
by sorry

end additional_dividend_amount_l159_159200


namespace johns_total_cost_l159_159438

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end johns_total_cost_l159_159438


namespace find_original_mean_l159_159058

noncomputable def original_mean (M : ℝ) : Prop :=
  let num_observations := 50
  let decrement := 47
  let updated_mean := 153
  M * num_observations - (num_observations * decrement) = updated_mean * num_observations

theorem find_original_mean : original_mean 200 :=
by
  unfold original_mean
  simp [*, mul_sub_left_distrib] at *
  sorry

end find_original_mean_l159_159058


namespace final_coordinates_of_F_l159_159655

-- Define the points D, E, F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the initial points D, E, F
def D : Point := ⟨3, -4⟩
def E : Point := ⟨5, -1⟩
def F : Point := ⟨-2, -3⟩

-- Define the reflection over the y-axis
def reflect_over_y (p : Point) : Point := ⟨-p.x, p.y⟩

-- Define the reflection over the x-axis
def reflect_over_x (p : Point) : Point := ⟨p.x, -p.y⟩

-- First reflection over the y-axis
def F' : Point := reflect_over_y F

-- Second reflection over the x-axis
def F'' : Point := reflect_over_x F'

-- The proof problem
theorem final_coordinates_of_F'' :
  F'' = ⟨2, 3⟩ := 
sorry

end final_coordinates_of_F_l159_159655


namespace find_f_neg_a_l159_159397

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 4) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l159_159397


namespace yi_reads_more_than_jia_by_9_pages_l159_159362

-- Define the number of pages in the book
def total_pages : ℕ := 120

-- Define number of pages read per day by Jia and Yi
def pages_per_day_jia : ℕ := 8
def pages_per_day_yi : ℕ := 13

-- Define the number of days in the period
def total_days : ℕ := 7

-- Calculate total pages read by Jia in the given period
def pages_read_by_jia : ℕ := total_days * pages_per_day_jia

-- Calculate the number of reading days by Yi in the given period
def reading_days_yi : ℕ := (total_days / 3) * 2 + (total_days % 3).min 2

-- Calculate total pages read by Yi in the given period
def pages_read_by_yi : ℕ := reading_days_yi * pages_per_day_yi

-- Given all conditions, prove that Yi reads 9 pages more than Jia over the 7-day period
theorem yi_reads_more_than_jia_by_9_pages :
  pages_read_by_yi - pages_read_by_jia = 9 :=
by
  sorry

end yi_reads_more_than_jia_by_9_pages_l159_159362


namespace triangle_side_lengths_l159_159722

theorem triangle_side_lengths (x : ℤ) : 3 < x ∧ x < 11 → ∃ n : ℕ, n = 7 :=
by
  intro h
  use 7
  sorry

end triangle_side_lengths_l159_159722


namespace solve_for_a_l159_159599

theorem solve_for_a (a : ℕ) (h : a^3 = 21 * 25 * 35 * 63) : a = 105 :=
sorry

end solve_for_a_l159_159599


namespace number_of_ways_to_select_book_l159_159309

-- Definitions directly from the problem's conditions
def numMathBooks : Nat := 3
def numChineseBooks : Nat := 5
def numEnglishBooks : Nat := 8

-- The proof problem statement in Lean 4
theorem number_of_ways_to_select_book : numMathBooks + numChineseBooks + numEnglishBooks = 16 := 
by
  show 3 + 5 + 8 = 16
  sorry

end number_of_ways_to_select_book_l159_159309


namespace triangle_area_is_correct_l159_159854

noncomputable def triangle_area_inscribed_circle (r : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ := 
  (1 / 2) * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem triangle_area_is_correct :
  triangle_area_inscribed_circle (18 / Real.pi) (Real.pi / 3) (2 * Real.pi / 3) Real.pi =
  162 * Real.sqrt 3 / (Real.pi^2) :=
by sorry

end triangle_area_is_correct_l159_159854


namespace multiple_of_eight_l159_159269

theorem multiple_of_eight (x y : ℤ) (h : ∀ (k : ℤ), 24 + 16 * k = 8) : ∃ (k : ℤ), x + 16 * y = 8 * k := 
by
  sorry

end multiple_of_eight_l159_159269


namespace mark_speed_l159_159167

theorem mark_speed
  (chris_speed : ℕ)
  (distance_to_school : ℕ)
  (mark_total_distance : ℕ)
  (mark_time_longer : ℕ)
  (chris_speed_eq : chris_speed = 3)
  (distance_to_school_eq : distance_to_school = 9)
  (mark_total_distance_eq : mark_total_distance = 15)
  (mark_time_longer_eq : mark_time_longer = 2) :
  mark_total_distance / (distance_to_school / chris_speed + mark_time_longer) = 3 := 
by
  sorry 

end mark_speed_l159_159167


namespace value_of_abs_h_l159_159695

theorem value_of_abs_h (h : ℝ) : 
  (∃ r s : ℝ, (r + s = -4 * h) ∧ (r * s = -5) ∧ (r^2 + s^2 = 13)) → 
  |h| = (Real.sqrt 3) / 4 :=
by
  sorry

end value_of_abs_h_l159_159695


namespace problem_solution_l159_159151

theorem problem_solution (m n : ℤ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 :=
by
  sorry

end problem_solution_l159_159151


namespace scores_greater_than_18_l159_159929

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l159_159929


namespace compare_two_sqrt_five_five_l159_159107

theorem compare_two_sqrt_five_five : 2 * Real.sqrt 5 < 5 :=
sorry

end compare_two_sqrt_five_five_l159_159107


namespace time_to_cross_bridge_l159_159532

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (time_min : ℝ) :
  speed_km_hr = 5 → length_m = 1250 → time_min = length_m / (speed_km_hr * 1000 / 60) → time_min = 15 :=
by
  intros h_speed h_length h_time
  rw [h_speed, h_length] at h_time
  -- Since 5 km/hr * 1000 / 60 = 83.33 m/min,
  -- substituting into equation gives us 1250 / 83.33 ≈ 15.
  sorry

end time_to_cross_bridge_l159_159532


namespace cube_volume_given_surface_area_l159_159318

theorem cube_volume_given_surface_area (SA : ℝ) (a V : ℝ) (h : SA = 864) (h1 : 6 * a^2 = SA) (h2 : V = a^3) : 
  V = 1728 := 
by 
  sorry

end cube_volume_given_surface_area_l159_159318


namespace polynomial_evaluation_l159_159451

def f (x : ℝ) : ℝ := sorry

theorem polynomial_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 6 * x^2 + 2) :
  f (x^2 - 3) = x^4 - 2 * x^2 - 7 :=
sorry

end polynomial_evaluation_l159_159451


namespace ordered_pair_sqrt_l159_159389

/-- Problem statement: Given positive integers a and b such that a < b, prove that:
sqrt (1 + sqrt (40 + 24 * sqrt 5)) = sqrt a + sqrt b, if (a, b) = (1, 6). -/
theorem ordered_pair_sqrt (a b : ℕ) (h1 : a = 1) (h2 : b = 6) (h3 : a < b) :
  Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b :=
by
  sorry -- The proof is not required in this task.

end ordered_pair_sqrt_l159_159389


namespace trig_expression_evaluation_l159_159401

theorem trig_expression_evaluation
  (α : ℝ)
  (h : Real.tan α = 2) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by 
  sorry

end trig_expression_evaluation_l159_159401


namespace max_students_l159_159644

-- Define the constants for pens and pencils
def pens : ℕ := 1802
def pencils : ℕ := 1203

-- State that the GCD of pens and pencils is 1
theorem max_students : Nat.gcd pens pencils = 1 :=
by sorry

end max_students_l159_159644


namespace find_a_l159_159572

variable {a b c : ℝ}

theorem find_a 
  (h1 : (a + b + c) ^ 2 = 3 * (a ^ 2 + b ^ 2 + c ^ 2))
  (h2 : a + b + c = 12) : 
  a = 4 := 
sorry

end find_a_l159_159572


namespace side_length_of_largest_square_l159_159845

theorem side_length_of_largest_square (A_cross : ℝ) (s : ℝ)
  (h1 : A_cross = 810) : s = 36 :=
  have h_large_squares : 2 * (s / 2)^2 = s^2 / 2 := by sorry
  have h_small_squares : 2 * (s / 4)^2 = s^2 / 8 := by sorry
  have h_combined_area : s^2 / 2 + s^2 / 8 = 810 := by sorry
  have h_final : 5 * s^2 / 8 = 810 := by sorry
  have h_s2 : s^2 = 1296 := by sorry
  have h_s : s = 36 := by sorry
  h_s

end side_length_of_largest_square_l159_159845


namespace algebraic_expression_value_l159_159468

variable (x y A B : ℤ)
variable (x_val : x = -1)
variable (y_val : y = 2)
variable (A_def : A = 2*x + y)
variable (B_def : B = 2*x - y)

theorem algebraic_expression_value : 
  (A^2 - B^2) * (x - 2*y) = 80 := 
by
  rw [x_val, y_val, A_def, B_def]
  sorry

end algebraic_expression_value_l159_159468


namespace thomas_total_training_hours_l159_159789

-- Define the conditions from the problem statement.
def training_hours_first_15_days : ℕ := 15 * 5
def training_hours_next_15_days : ℕ := (15 - 3) * (4 + 3)
def training_hours_next_12_days : ℕ := (12 - 2) * (4 + 3)

-- Prove that the total training hours equals 229.
theorem thomas_total_training_hours : 
  training_hours_first_15_days + training_hours_next_15_days + training_hours_next_12_days = 229 :=
by
  -- conditions as defined
  let t1 := 15 * 5
  let t2 := (15 - 3) * (4 + 3)
  let t3 := (12 - 2) * (4 + 3)
  show t1 + t2 + t3 = 229
  sorry

end thomas_total_training_hours_l159_159789


namespace intervals_of_monotonicity_range_of_a_for_zeros_l159_159892

open Real

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * log x

theorem intervals_of_monotonicity (a : ℝ) (ha : a ≠ 0) :
  (0 < a → ∀ x, (0 < x ∧ x < a → f x a < f (x + 1) a)
            ∧ (a < x ∧ x < 2 * a → f x a > f (x + 1) a)
            ∧ (2 * a < x → f x a < f (x + 1) a))
  ∧ (a < 0 → ∀ x, (0 < x → f x a < f (x + 1) a)) :=
sorry

theorem range_of_a_for_zeros (a x : ℝ) (ha : 0 < a) 
  (h1 : f a a > 0) (h2 : f (2 * a) a < 0) :
  e ^ (5 / 4) < a ∧ a < e ^ 2 / 2 :=
sorry

end intervals_of_monotonicity_range_of_a_for_zeros_l159_159892


namespace original_manufacturing_cost_l159_159831

variable (SP OC : ℝ)
variable (ManuCost : ℝ) -- Declaring manufacturing cost

-- Current conditions
axiom profit_percentage_constant : ∀ SP, 0.5 * SP = SP - 50

-- Problem Statement
theorem original_manufacturing_cost : (∃ OC, 0.5 * SP - OC = 0.5 * SP) ∧ ManuCost = 50 → OC = 50 := by
  sorry

end original_manufacturing_cost_l159_159831


namespace least_hourly_number_l159_159172

def is_clock_equivalent (a b : ℕ) : Prop := (a - b) % 12 = 0

theorem least_hourly_number : ∃ n ≥ 6, is_clock_equivalent n (n * n) ∧ ∀ m ≥ 6, is_clock_equivalent m (m * m) → 9 ≤ m → n = 9 := 
by
  sorry

end least_hourly_number_l159_159172


namespace total_distance_traveled_l159_159944

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l159_159944


namespace sum_of_integers_l159_159053

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end sum_of_integers_l159_159053


namespace find_c_l159_159967

theorem find_c (x c : ℤ) (h1 : 3 * x + 9 = 0) (h2 : c * x - 5 = -11) : c = 2 := by
  have x_eq : x = -3 := by
    linarith
  subst x_eq
  have c_eq : c = 2 := by
    linarith
  exact c_eq

end find_c_l159_159967


namespace final_output_value_of_m_l159_159014

variables (a b m : ℕ)

theorem final_output_value_of_m (h₁ : a = 2) (h₂ : b = 3) (program_logic : (a > b → m = a) ∧ (a ≤ b → m = b)) :
  m = 3 :=
by
  have h₃ : a ≤ b := by
    rw [h₁, h₂]
    exact le_of_lt (by norm_num)
  exact (program_logic.right h₃).trans h₂

end final_output_value_of_m_l159_159014


namespace patty_weighs_more_l159_159044

variable (R : ℝ) (P_0 : ℝ) (L : ℝ) (P : ℝ) (D : ℝ)

theorem patty_weighs_more :
  (R = 100) →
  (P_0 = 4.5 * R) →
  (L = 235) →
  (P = P_0 - L) →
  (D = P - R) →
  D = 115 := by
  sorry

end patty_weighs_more_l159_159044


namespace scores_greater_than_18_l159_159927

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l159_159927


namespace population_after_ten_years_l159_159062

-- Define the initial population and constants
def initial_population : ℕ := 100000
def birth_increase_rate : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

-- Proving the total population at the end of 10 years
theorem population_after_ten_years :
  initial_population + (initial_population * birth_increase_rate).to_nat +
  (immigration_per_year * years - emigration_per_year * years) = 165000 := by
sorry

end population_after_ten_years_l159_159062


namespace necessary_but_not_sufficient_l159_159836

theorem necessary_but_not_sufficient (x: ℝ) :
  (1 < x ∧ x < 4) → (1 < x ∧ x < 3) := by
sorry

end necessary_but_not_sufficient_l159_159836


namespace circles_tangent_parallel_l159_159188

open EuclideanGeometry

theorem circles_tangent_parallel {k₁ k₂ : Circle} 
  {A C : Point} (hA : A ∈ k₁ ∩ k₂) (hC : C ∈ k₁ ∩ k₂)
  {B D : Point} 
  (hB : B ≠ A ∧ B ∈ k₁ ∧ tangent (k₂) (Line.from_points B A))
  (hD : D ≠ C ∧ D ∈ k₂ ∧ tangent (k₁) (Line.from_points D C)) :
  parallel (Line.from_points A D) (Line.from_points B C) :=
by
  sorry

end circles_tangent_parallel_l159_159188


namespace cyclic_quadrilateral_condition_l159_159639

-- Definitions of the points and sides of the triangle
variables (A B C S E F : Type) 

-- Assume S is the centroid of triangle ABC
def is_centroid (A B C S : Type) : Prop := 
  -- actual centralized definition here (omitted)
  sorry

-- Assume E is the midpoint of side AB
def is_midpoint (A B E : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume F is the midpoint of side AC
def is_midpoint_AC (A C F : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume a quadrilateral AESF
def is_cyclic (A E S F : Type) : Prop :=
  -- actual cyclic definition here (omitted)
  sorry 

theorem cyclic_quadrilateral_condition 
  (A B C S E F : Type)
  (a b c : ℝ) 
  (h1 : is_centroid A B C S)
  (h2 : is_midpoint A B E) 
  (h3 : is_midpoint_AC A C F) :
  is_cyclic A E S F ↔ (c^2 + b^2 = 2 * a^2) :=
sorry

end cyclic_quadrilateral_condition_l159_159639


namespace find_number_l159_159816

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 :=
sorry

end find_number_l159_159816


namespace smallest_six_consecutive_number_exists_max_value_N_perfect_square_l159_159359

-- Definition of 'six-consecutive numbers'
def is_six_consecutive (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧
  b ≠ d ∧ c ≠ d ∧ (a + b) * (c + d) = 60

-- Definition of the function F
def F (a b c d : ℕ) : ℤ :=
  let p := (10 * a + c) - (10 * b + d)
  let q := (10 * a + d) - (10 * b + c)
  q - p

-- Exists statement for the smallest six-consecutive number
theorem smallest_six_consecutive_number_exists :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ (1000 * a + 100 * b + 10 * c + d) = 1369 := 
sorry

-- Exists statement for the maximum N such that F(N) is perfect square
theorem max_value_N_perfect_square :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 9613 ∧
  ∃ (k : ℤ), F a b c d = k ^ 2 := 
sorry

end smallest_six_consecutive_number_exists_max_value_N_perfect_square_l159_159359


namespace smallest_number_of_contestants_solving_all_problems_l159_159565

theorem smallest_number_of_contestants_solving_all_problems
    (total_contestants : ℕ)
    (solve_first : ℕ)
    (solve_second : ℕ)
    (solve_third : ℕ)
    (solve_fourth : ℕ)
    (H1 : total_contestants = 100)
    (H2 : solve_first = 90)
    (H3 : solve_second = 85)
    (H4 : solve_third = 80)
    (H5 : solve_fourth = 75)
  : ∃ n, n = 30 := by
  sorry

end smallest_number_of_contestants_solving_all_problems_l159_159565


namespace average_of_xyz_l159_159150

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 :=
by
  sorry

end average_of_xyz_l159_159150


namespace inverse_proportion_value_scientific_notation_l159_159591

-- Statement to prove for Question 1:
theorem inverse_proportion_value (m : ℤ) (x : ℝ) :
  (m - 2) * x ^ (m ^ 2 - 5) = 0 ↔ m = -2 := by
  sorry

-- Statement to prove for Question 2:
theorem scientific_notation : -0.00000032 = -3.2 * 10 ^ (-7) := by
  sorry

end inverse_proportion_value_scientific_notation_l159_159591


namespace abs_eq_abs_iff_eq_frac_l159_159190

theorem abs_eq_abs_iff_eq_frac {x : ℚ} :
  |x - 3| = |x - 4| → x = 7 / 2 :=
by
  intro h
  sorry

end abs_eq_abs_iff_eq_frac_l159_159190


namespace repayment_difference_l159_159855

noncomputable def compounded_repayment (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def simple_repayment (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem repayment_difference {P : ℝ} {r1 r2 : ℝ} {n t : ℕ}
  (hP : P = 12000)
  (hr1 : r1 = 0.08)
  (hr2 : r2 = 0.10)
  (hn : n = 2)
  (ht : t = 12)
  (ht_half : t / 2 = 6) :
  abs ((simple_repayment P r2 t) - (compounded_repayment (compounded_repayment P r1 n (t / 2) / 2) r1 n (t / 2) + (compounded_repayment P r1 n (t / 2) / 2))) = 3901 :=
  sorry

end repayment_difference_l159_159855


namespace range_of_a_l159_159716

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < - (Real.sqrt 3) / 3 ∨ x > (Real.sqrt 3) / 3 →
    a * (3 * x^2 - 1) > 0) →
  a > 0 :=
by
  sorry

end range_of_a_l159_159716


namespace sin_angle_calculation_l159_159247

theorem sin_angle_calculation (α : ℝ) (h : α = 240) : Real.sin (150 - α) = -1 :=
by
  rw [h]
  norm_num
  sorry

end sin_angle_calculation_l159_159247


namespace find_number_of_white_balls_l159_159454

-- Define the conditions
variables (n k : ℕ)
axiom k_ge_2 : k ≥ 2
axiom prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100

-- State the theorem
theorem find_number_of_white_balls (n k : ℕ) (k_ge_2 : k ≥ 2) (prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100) : n = 19 :=
sorry

end find_number_of_white_balls_l159_159454


namespace samantha_laundromat_cost_l159_159858

-- Definitions of given conditions
def washer_cost : ℕ := 4
def dryer_cost_per_10_min : ℝ := 0.25
def num_washes : ℕ := 2
def num_dryers : ℕ := 3
def dryer_time : ℕ := 40

-- Calculate total cost
def washing_cost : ℝ := washer_cost * num_washes
def intervals_10min : ℕ := dryer_time / 10
def single_dryer_cost : ℝ := dryer_cost_per_10_min * intervals_10min
def total_drying_cost : ℝ := single_dryer_cost * num_dryers
def total_cost : ℝ := washing_cost + total_drying_cost

-- The statement to prove
theorem samantha_laundromat_cost : total_cost = 11 :=
by
  unfold washer_cost dryer_cost_per_10_min num_washes num_dryers dryer_time washing_cost intervals_10min single_dryer_cost total_drying_cost total_cost
  norm_num
  done

end samantha_laundromat_cost_l159_159858


namespace right_triangle_longer_leg_l159_159274

theorem right_triangle_longer_leg (a b c : ℕ) (h₀ : a^2 + b^2 = c^2) (h₁ : c = 65) (h₂ : a < b) : b = 60 :=
sorry

end right_triangle_longer_leg_l159_159274


namespace pyramid_volume_l159_159284

noncomputable def volume_of_pyramid 
  (ABCD : Type) 
  (rectangle : ABCD) 
  (DM_perpendicular : Prop) 
  (MA MC MB : ℕ) 
  (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) : ℝ :=
  80 * Real.sqrt 6

theorem pyramid_volume (ABCD : Type) 
    (rectangle : ABCD) 
    (DM_perpendicular : Prop) 
    (MA MC MB DM : ℕ)
    (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) 
  : volume_of_pyramid ABCD rectangle DM_perpendicular MA MC MB lengths = 80 * Real.sqrt 6 :=
  by {
    sorry
  }

end pyramid_volume_l159_159284


namespace intersection_A_complement_B_eq_interval_l159_159411

-- We define universal set U as ℝ
def U := Set ℝ

-- Definitions provided in the problem
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y >= 2 }

-- Complement of B in U
def C_U_B : Set ℝ := { y | y < 2 }

-- Now we state the theorem
theorem intersection_A_complement_B_eq_interval :
  A ∩ C_U_B = { x | 1 < x ∧ x < 2 } :=
by 
  sorry

end intersection_A_complement_B_eq_interval_l159_159411


namespace shortest_time_to_camp_l159_159194

/-- 
Given:
- The width of the river is 1 km.
- The camp is 1 km away from the point directly across the river.
- Swimming speed is 2 km/hr.
- Walking speed is 3 km/hr.

Prove the shortest time required to reach the camp is (2 + √5) / 6 hours.
--/
theorem shortest_time_to_camp :
  ∃ t : ℝ, t = (2 + Real.sqrt 5) / 6 := 
sorry

end shortest_time_to_camp_l159_159194


namespace last_three_digits_of_16_pow_128_l159_159561

theorem last_three_digits_of_16_pow_128 : (16 ^ 128) % 1000 = 721 := 
by
  sorry

end last_three_digits_of_16_pow_128_l159_159561


namespace rhombus_perimeter_l159_159963

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * s = 8 * Real.sqrt 41 := 
by
  sorry

end rhombus_perimeter_l159_159963


namespace blocks_needed_for_wall_l159_159999

theorem blocks_needed_for_wall (length height : ℕ) (block_heights block_lengths : List ℕ)
  (staggered : Bool) (even_ends : Bool)
  (h_length : length = 120)
  (h_height : height = 8)
  (h_block_heights : block_heights = [1])
  (h_block_lengths : block_lengths = [1, 2, 3])
  (h_staggered : staggered = true)
  (h_even_ends : even_ends = true) :
  ∃ (n : ℕ), n = 404 := 
sorry

end blocks_needed_for_wall_l159_159999


namespace diagonal_cells_crossed_l159_159146

theorem diagonal_cells_crossed (m n : ℕ) (h_m : m = 199) (h_n : n = 991) :
  (m + n - Nat.gcd m n) = 1189 := by
  sorry

end diagonal_cells_crossed_l159_159146


namespace prove_y_value_l159_159728

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159728


namespace parkingGarageCharges_l159_159533

variable (W : ℕ)

/-- 
  Conditions:
  1. Weekly rental cost is \( W \) dollars.
  2. Monthly rental cost is $24 per month.
  3. A person saves $232 in a year by renting by the month rather than by the week.
  4. There are 52 weeks in a year.
  5. There are 12 months in a year.
-/
def garageChargesPerWeek : Prop :=
  52 * W = 12 * 24 + 232

theorem parkingGarageCharges
  (h : garageChargesPerWeek W) : W = 10 :=
by
  sorry

end parkingGarageCharges_l159_159533


namespace different_gcd_values_count_l159_159823

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159823


namespace polar_line_equation_l159_159390

/-- A line that passes through a given point in polar coordinates and is parallel to the polar axis
    has a specific polar coordinate equation. -/
theorem polar_line_equation (r : ℝ) (θ : ℝ) (h : r = 6 ∧ θ = π / 6) : θ = π / 6 :=
by
  /- We are given that the line passes through the point \(C(6, \frac{\pi}{6})\) which means
     \(r = 6\) and \(θ = \frac{\pi}{6}\). Since the line is parallel to the polar axis, 
     the angle \(θ\) remains the same. Therefore, the polar coordinate equation of the line 
     is simply \(θ = \frac{\pi}{6}\). -/
  sorry

end polar_line_equation_l159_159390


namespace natural_numbers_between_sqrt_100_and_101_l159_159902

theorem natural_numbers_between_sqrt_100_and_101 :
  ∃ (n : ℕ), n = 200 ∧ (∀ k : ℕ, 100 < Real.sqrt k ∧ Real.sqrt k < 101 -> 10000 < k ∧ k < 10201) := 
by
  sorry

end natural_numbers_between_sqrt_100_and_101_l159_159902


namespace average_lifespan_is_1013_l159_159528

noncomputable def first_factory_lifespan : ℕ := 980
noncomputable def second_factory_lifespan : ℕ := 1020
noncomputable def third_factory_lifespan : ℕ := 1032

noncomputable def total_samples : ℕ := 100

noncomputable def first_samples : ℕ := (1 * total_samples) / 4
noncomputable def second_samples : ℕ := (2 * total_samples) / 4
noncomputable def third_samples : ℕ := (1 * total_samples) / 4

noncomputable def weighted_average_lifespan : ℕ :=
  ((first_factory_lifespan * first_samples) + (second_factory_lifespan * second_samples) + (third_factory_lifespan * third_samples)) / total_samples

theorem average_lifespan_is_1013 : weighted_average_lifespan = 1013 := by
  sorry

end average_lifespan_is_1013_l159_159528


namespace smallest_n_l159_159633

/-- The smallest value of n > 20 that satisfies
    n ≡ 4 [MOD 6]
    n ≡ 3 [MOD 7]
    n ≡ 5 [MOD 8] is 220. -/
theorem smallest_n (n : ℕ) : 
  (n > 20) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (n % 8 = 5) ↔ (n = 220) :=
by 
  sorry

end smallest_n_l159_159633


namespace acute_triangle_l159_159767

theorem acute_triangle (a b c : ℝ) (n : ℕ) (h_n : 2 < n) (h_eq : a^n + b^n = c^n) : a^2 + b^2 > c^2 :=
sorry

end acute_triangle_l159_159767


namespace find_number_l159_159998

-- Define the conditions
def condition (x : ℝ) : Prop := 0.65 * x = (4/5) * x - 21

-- Prove that given the condition, x is 140.
theorem find_number (x : ℝ) (h : condition x) : x = 140 := by
  sorry

end find_number_l159_159998


namespace smallest_number_with_eight_factors_l159_159511

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l159_159511


namespace function_zero_interval_l159_159560

noncomputable def f (x : ℝ) : ℝ := 1 / 4^x - Real.log x / Real.log 4

theorem function_zero_interval :
  ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 := by
  sorry

end function_zero_interval_l159_159560


namespace ab_calculation_l159_159906

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1 / 2) * (4 / a) * (4 / b)

theorem ab_calculation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : triangle_area a b = 4) : a * b = 2 :=
by
  sorry

end ab_calculation_l159_159906


namespace rhombus_perimeter_l159_159961

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end rhombus_perimeter_l159_159961


namespace prove_nat_number_l159_159558

theorem prove_nat_number (p : ℕ) (hp : Nat.Prime p) (n : ℕ) :
  n^2 = p^2 + 3*p + 9 → n = 7 :=
sorry

end prove_nat_number_l159_159558


namespace corridor_perimeter_l159_159301

theorem corridor_perimeter
  (P1 P2 : ℕ)
  (h₁ : P1 = 16)
  (h₂ : P2 = 24) : 
  2 * ((P2 / 4 + (P1 + P2) / 4) + (P2 / 4) - (P1 / 4)) = 40 :=
by {
  -- The proof can be filled here
  sorry
}

end corridor_perimeter_l159_159301


namespace gcd_lcm_sum_l159_159989

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 40 10 = 46 := by
  sorry

end gcd_lcm_sum_l159_159989


namespace find_p_l159_159123

variable (a b p q r1 r2 : ℝ)

-- Given conditions
def roots_eq1 (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) : Prop :=
  -- Using Vieta's Formulas on x^2 + ax + b = 0
  ∀ (r1 r2 : ℝ), r1 + r2 = -a ∧ r1 * r2 = b

def roots_eq2 (r1 r2 : ℝ) (h_3 : r1^2 + r2^2 = -p) (h_4 : r1^2 * r2^2 = q) : Prop :=
  -- Using Vieta's Formulas on x^2 + px + q = 0
  ∀ (r1 r2 : ℝ), r1^2 + r2^2 = -p ∧ r1^2 * r2^2 = q

-- Theorems
theorem find_p (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) (h_3 : r1^2 + r2^2 = -p) :
  p = -a^2 + 2*b := by
  sorry

end find_p_l159_159123


namespace tan_sum_example_l159_159004

theorem tan_sum_example :
  let t1 := Real.tan (17 * Real.pi / 180)
  let t2 := Real.tan (43 * Real.pi / 180)
  t1 + t2 + Real.sqrt 3 * t1 * t2 = Real.sqrt 3 := sorry

end tan_sum_example_l159_159004


namespace count_multiples_of_7_not_14_l159_159255

theorem count_multiples_of_7_not_14 (n : ℕ) : (n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) → ∃ (k : ℕ), k = 36 :=
by
  sorry

end count_multiples_of_7_not_14_l159_159255


namespace maximum_value_of_f_l159_159139

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem maximum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = Real.exp 1 :=
sorry

end maximum_value_of_f_l159_159139


namespace equal_clubs_and_students_l159_159424

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l159_159424


namespace distinct_values_of_b_l159_159704

theorem distinct_values_of_b : ∃ b_list : List ℝ, b_list.length = 8 ∧ ∀ b ∈ b_list, ∃ p q : ℤ, p + q = b ∧ p * q = 8 * b :=
by
  sorry

end distinct_values_of_b_l159_159704


namespace tank_full_capacity_is_72_l159_159817

theorem tank_full_capacity_is_72 (x : ℝ) 
  (h1 : 0.9 * x - 0.4 * x = 36) : 
  x = 72 := 
sorry

end tank_full_capacity_is_72_l159_159817


namespace average_marks_math_chem_l159_159785

-- Definitions to capture the conditions
variables (M P C : ℕ)
variable (cond1 : M + P = 32)
variable (cond2 : C = P + 20)

-- The theorem to prove
theorem average_marks_math_chem (M P C : ℕ) 
  (cond1 : M + P = 32) 
  (cond2 : C = P + 20) : 
  (M + C) / 2 = 26 := 
sorry

end average_marks_math_chem_l159_159785


namespace sum_of_possible_values_of_N_l159_159971

theorem sum_of_possible_values_of_N (N : ℤ) : 
  (N * (N - 8) = 16) -> (∃ a b, N^2 - 8 * N - 16 = 0 ∧ (a + b = 8)) :=
sorry

end sum_of_possible_values_of_N_l159_159971


namespace remainder_of_polynomial_division_l159_159658

theorem remainder_of_polynomial_division :
  Polynomial.eval 2 (8 * X^3 - 22 * X^2 + 30 * X - 45) = -9 :=
by {
  sorry
}

end remainder_of_polynomial_division_l159_159658


namespace sqrt_sum_eq_l159_159116

theorem sqrt_sum_eq :
  (Real.sqrt (9 / 2) + Real.sqrt (2 / 9)) = (11 * Real.sqrt 2 / 6) :=
sorry

end sqrt_sum_eq_l159_159116


namespace compute_expression_l159_159907

noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

theorem compute_expression : 5^(c / d) + 2^(d / c) = 2 * Real.sqrt 2 + 5^(2 / 3) :=
by
  sorry

end compute_expression_l159_159907


namespace track_team_children_l159_159982

/-- There were initially 18 girls and 15 boys on the track team.
    7 more girls joined the team, and 4 boys quit the team.
    The proof shows that the total number of children on the track team after the changes is 36. -/
theorem track_team_children (initial_girls initial_boys girls_joined boys_quit : ℕ)
  (h_initial_girls : initial_girls = 18)
  (h_initial_boys : initial_boys = 15)
  (h_girls_joined : girls_joined = 7)
  (h_boys_quit : boys_quit = 4) :
  initial_girls + girls_joined - boys_quit + initial_boys = 36 :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end track_team_children_l159_159982


namespace october_birth_percentage_l159_159777

theorem october_birth_percentage 
  (jan feb mar apr may jun jul aug sep oct nov dec total : ℕ) 
  (h_total : total = 100)
  (h_jan : jan = 2) (h_feb : feb = 4) (h_mar : mar = 8) (h_apr : apr = 5) 
  (h_may : may = 4) (h_jun : jun = 9) (h_jul : jul = 7) (h_aug : aug = 12) 
  (h_sep : sep = 8) (h_oct : oct = 6) (h_nov : nov = 5) (h_dec : dec = 4) : 
  (oct : ℕ) * 100 / total = 6 := 
by
  sorry

end october_birth_percentage_l159_159777


namespace remainder_count_l159_159021

theorem remainder_count (n : ℕ) (h : n > 5) : 
  ∃ l : List ℕ, l.length = 5 ∧ ∀ x ∈ l, x ∣ 42 ∧ x > 5 := 
sorry

end remainder_count_l159_159021


namespace parity_implies_even_sum_l159_159737

theorem parity_implies_even_sum (n m : ℤ) (h : Even (n^2 + m^2 + n * m)) : ¬Odd (n + m) :=
sorry

end parity_implies_even_sum_l159_159737


namespace line_equation_l159_159680

theorem line_equation
  (t : ℝ)
  (x : ℝ) (y : ℝ)
  (h1 : x = 3 * t + 6)
  (h2 : y = 5 * t - 10) :
  y = (5 / 3) * x - 20 :=
sorry

end line_equation_l159_159680


namespace xy_sum_cases_l159_159589

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l159_159589


namespace find_constants_l159_159003

theorem find_constants (A B C : ℤ) (h1 : 1 = A + B) (h2 : -2 = C) (h3 : 5 = -A) :
  A = -5 ∧ B = 6 ∧ C = -2 :=
by {
  sorry
}

end find_constants_l159_159003


namespace equation_of_line_AC_l159_159030

-- Define the given points A and B
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-3, -5)

-- Define the line m as a predicate
def line_m (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 + 6 = 0

-- Define the condition that line m is the angle bisector of ∠ACB
def is_angle_bisector (A B C : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : Prop := sorry

-- The symmetric point of B with respect to line m
def symmetric_point (B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : (ℝ × ℝ) := sorry

-- Proof statement
theorem equation_of_line_AC :
  ∀ (A B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop),
  A = (1, 1) →
  B = (-3, -5) →
  m = line_m →
  is_angle_bisector A B (symmetric_point B m) m →
  AC = {p : ℝ × ℝ | p.1 = 1} := sorry

end equation_of_line_AC_l159_159030


namespace ABCD_area_is_correct_l159_159609

-- Define rectangle ABCD with the given conditions
def ABCD_perimeter (x : ℝ) : Prop :=
  2 * (4 * x + x) = 160

-- Define the area to be proved
def ABCD_area (x : ℝ) : ℝ :=
  4 * (x ^ 2)

-- The proof problem: given the conditions, the area should be 1024 square centimeters
theorem ABCD_area_is_correct (x : ℝ) (h : ABCD_perimeter x) : 
  ABCD_area x = 1024 := 
by {
  sorry
}

end ABCD_area_is_correct_l159_159609


namespace internet_bill_proof_l159_159290

variable (current_bill : ℕ)
variable (internet_bill_30Mbps : ℕ)
variable (annual_savings : ℕ)
variable (additional_amount_20Mbps : ℕ)

theorem internet_bill_proof
  (h1 : current_bill = 20)
  (h2 : internet_bill_30Mbps = 40)
  (h3 : annual_savings = 120)
  (monthly_savings : ℕ := annual_savings / 12)
  (h4 : monthly_savings = 10)
  (h5 : internet_bill_30Mbps - (current_bill + additional_amount_20Mbps) = 10) :
  additional_amount_20Mbps = 10 :=
by
  sorry

end internet_bill_proof_l159_159290


namespace first_half_day_wednesday_l159_159748

theorem first_half_day_wednesday (h1 : ¬(1 : ℕ) = (4 % 7) ∨ 1 % 7 != 0)
  (h2 : ∀ d : ℕ, d ≤ 31 → d % 7 = ((d + 3) % 7)) : 
  ∃ d : ℕ, d = 25 ∧ ∃ W : ℕ → Prop, W d := sorry

end first_half_day_wednesday_l159_159748


namespace Thabo_books_l159_159476

theorem Thabo_books :
  ∃ (H : ℕ), ∃ (P : ℕ), ∃ (F : ℕ), 
  (H + P + F = 220) ∧ 
  (P = H + 20) ∧ 
  (F = 2 * P) ∧ 
  (H = 40) :=
by
  -- Here will be the formal proof, which is not required for this task.
  sorry

end Thabo_books_l159_159476


namespace fixed_point_of_line_l159_159038

theorem fixed_point_of_line (m : ℝ) : 
  (m - 2) * (-3) - 8 + 3 * m + 2 = 0 :=
by
  sorry

end fixed_point_of_line_l159_159038


namespace det_calculation_l159_159246

-- Given conditions
variables (p q r s : ℤ)
variable (h1 : p * s - q * r = -3)

-- Define the matrix and determinant
def matrix_determinant (a b c d : ℤ) := a * d - b * c

-- Problem statement
theorem det_calculation : matrix_determinant (p + 2 * r) (q + 2 * s) r s = -3 :=
by
  -- Proof goes here
  sorry

end det_calculation_l159_159246


namespace gcd_values_360_l159_159826

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159826


namespace rhombus_perimeter_l159_159959

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l159_159959


namespace johns_total_payment_l159_159440

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end johns_total_payment_l159_159440


namespace wallpaper_expenditure_l159_159273

structure Room :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

def cost_per_square_meter : ℕ := 75

def total_expenditure (room : Room) : ℕ :=
  let perimeter := 2 * (room.length + room.width)
  let area_of_walls := perimeter * room.height
  let area_of_ceiling := room.length * room.width
  let total_area := area_of_walls + area_of_ceiling
  total_area * cost_per_square_meter

theorem wallpaper_expenditure (room : Room) : 
  room = Room.mk 30 25 10 →
  total_expenditure room = 138750 :=
by 
  intros h
  rw [h]
  sorry

end wallpaper_expenditure_l159_159273


namespace least_n_for_distance_l159_159285

theorem least_n_for_distance (n : ℕ) : n = 17 ↔ (100 ≤ n * (n + 1) / 3) := sorry

end least_n_for_distance_l159_159285


namespace intersection_A_B_l159_159142

open Set Real

def A := { x : ℝ | x ^ 2 - 6 * x + 5 ≤ 0 }
def B := { x : ℝ | ∃ y : ℝ, y = log (x - 2) / log 2 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 5 } :=
by
  sorry

end intersection_A_B_l159_159142


namespace difference_in_price_l159_159976

noncomputable def total_cost : ℝ := 70.93
noncomputable def pants_price : ℝ := 34.00

theorem difference_in_price (total_cost pants_price : ℝ) (h_total : total_cost = 70.93) (h_pants : pants_price = 34.00) :
  (total_cost - pants_price) - pants_price = 2.93 :=
by
  sorry

end difference_in_price_l159_159976


namespace units_digit_N_l159_159034

def P (n : ℕ) : ℕ := (n / 10) * (n % 10)
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem units_digit_N (N : ℕ) (h1 : 10 ≤ N ∧ N ≤ 99) (h2 : N = P N + S N) : N % 10 = 9 :=
by
  sorry

end units_digit_N_l159_159034


namespace carpenter_additional_logs_needed_l159_159199

theorem carpenter_additional_logs_needed 
  (total_woodblocks_needed : ℕ) 
  (logs_available : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5)
  (h4 : additional_logs_needed = 8) : 
  (total_woodblocks_needed - (logs_available * woodblocks_per_log)) / woodblocks_per_log = additional_logs_needed :=
by
  sorry

end carpenter_additional_logs_needed_l159_159199


namespace pagoda_lanterns_l159_159434

-- Definitions
def top_layer_lanterns (a₁ : ℕ) : ℕ := a₁
def bottom_layer_lanterns (a₁ : ℕ) : ℕ := a₁ * 2^6
def sum_of_lanterns (a₁ : ℕ) : ℕ := (a₁ * (1 - 2^7)) / (1 - 2)
def total_lanterns : ℕ := 381
def layers : ℕ := 7
def common_ratio : ℕ := 2

-- Problem Statement
theorem pagoda_lanterns (a₁ : ℕ) (h : sum_of_lanterns a₁ = total_lanterns) : 
  top_layer_lanterns a₁ + bottom_layer_lanterns a₁ = 195 := sorry

end pagoda_lanterns_l159_159434


namespace clubs_equal_students_l159_159429

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l159_159429


namespace distinct_m_values_count_l159_159761

theorem distinct_m_values_count :
  ∃ (m_values : Finset ℤ), (∀ x1 x2 : ℤ, x1 * x2 = 36 → m_values ∈ { x1 + x2 }) ∧ m_values.card = 10 :=
by
  sorry

end distinct_m_values_count_l159_159761


namespace gabby_mom_gave_20_l159_159007

theorem gabby_mom_gave_20 (makeup_set_cost saved_money more_needed total_needed mom_money : ℕ)
  (h1 : makeup_set_cost = 65)
  (h2 : saved_money = 35)
  (h3 : more_needed = 10)
  (h4 : total_needed = makeup_set_cost - saved_money)
  (h5 : total_needed - mom_money = more_needed) :
  mom_money = 20 :=
by
  sorry

end gabby_mom_gave_20_l159_159007


namespace cans_left_to_be_loaded_l159_159091

def cartons_total : ℕ := 50
def cartons_loaded : ℕ := 40
def cans_per_carton : ℕ := 20

theorem cans_left_to_be_loaded : (cartons_total - cartons_loaded) * cans_per_carton = 200 := by
  sorry

end cans_left_to_be_loaded_l159_159091


namespace inequality_conditions_l159_159130

variable (a b : ℝ)

theorem inequality_conditions (ha : 1 / a < 1 / b) (hb : 1 / b < 0) : 
  (1 / (a + b) < 1 / (a * b)) ∧ ¬(a * - (1 / a) > b * - (1 / b)) := 
by 
  sorry

end inequality_conditions_l159_159130


namespace rational_solutions_quadratic_l159_159238

theorem rational_solutions_quadratic (k : ℕ) (h_pos : 0 < k) :
  (∃ (x : ℚ), k * x^2 + 24 * x + k = 0) ↔ k = 12 :=
by
  sorry

end rational_solutions_quadratic_l159_159238


namespace diplomats_neither_french_nor_russian_l159_159690

variable (total_diplomats : ℕ)
variable (speak_french : ℕ)
variable (not_speak_russian : ℕ)
variable (speak_both : ℕ)

theorem diplomats_neither_french_nor_russian {total_diplomats speak_french not_speak_russian speak_both : ℕ} 
  (h1 : total_diplomats = 100)
  (h2 : speak_french = 22)
  (h3 : not_speak_russian = 32)
  (h4 : speak_both = 10) :
  ((total_diplomats - (speak_french + (total_diplomats - not_speak_russian) - speak_both)) * 100) / total_diplomats = 20 := 
by
  sorry

end diplomats_neither_french_nor_russian_l159_159690


namespace value_of_T_l159_159629

theorem value_of_T (S : ℝ) (T : ℝ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) : T = 96 := 
by 
  sorry

end value_of_T_l159_159629


namespace jade_handled_84_transactions_l159_159346

def Mabel_transactions : ℕ := 90

def Anthony_transactions (mabel : ℕ) : ℕ := mabel + mabel / 10

def Cal_transactions (anthony : ℕ) : ℕ := (2 * anthony) / 3

def Jade_transactions (cal : ℕ) : ℕ := cal + 18

theorem jade_handled_84_transactions :
  Jade_transactions (Cal_transactions (Anthony_transactions Mabel_transactions)) = 84 := 
sorry

end jade_handled_84_transactions_l159_159346


namespace area_change_factor_l159_159846

theorem area_change_factor (k b : ℝ) (hk : 0 < k) (hb : 0 < b) :
  let S1 := (b * b) / (2 * k)
  let S2 := (b * b) / (16 * k)
  S1 / S2 = 8 :=
by
  sorry

end area_change_factor_l159_159846


namespace model_to_reality_length_l159_159537

-- Defining conditions
def scale_factor := 50 -- one centimeter represents 50 meters
def model_length := 7.5 -- line segment in the model is 7.5 centimeters

-- Statement of the problem
theorem model_to_reality_length (scale_factor model_length : ℝ) 
  (scale_condition : scale_factor = 50) (length_condition : model_length = 7.5) :
  model_length * scale_factor = 375 := 
by
  rw [length_condition, scale_condition]
  norm_num

end model_to_reality_length_l159_159537


namespace money_put_in_by_A_l159_159340

theorem money_put_in_by_A 
  (B_capital : ℕ := 25000)
  (total_profit : ℕ := 9600)
  (A_management_fee : ℕ := 10)
  (A_total_received : ℕ := 4200) 
  (A_puts_in : ℕ) :
  (A_management_fee * total_profit / 100 
    + (A_puts_in / (A_puts_in + B_capital)) * (total_profit - A_management_fee * total_profit / 100) = A_total_received)
  → A_puts_in = 15000 :=
  by
    sorry

end money_put_in_by_A_l159_159340


namespace eight_digit_increasing_numbers_mod_1000_l159_159222

theorem eight_digit_increasing_numbers_mod_1000 : 
  ((Nat.choose 17 8) % 1000) = 310 := 
by 
  sorry -- Proof not required as per instructions

end eight_digit_increasing_numbers_mod_1000_l159_159222


namespace bicycle_cost_price_l159_159079

variable (CP_A SP_B SP_C : ℝ)

theorem bicycle_cost_price 
  (h1 : SP_B = CP_A * 1.20) 
  (h2 : SP_C = SP_B * 1.25) 
  (h3 : SP_C = 225) :
  CP_A = 150 := 
by
  sorry

end bicycle_cost_price_l159_159079


namespace gcd_values_count_l159_159819

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159819


namespace multiples_7_not_14_l159_159260

theorem multiples_7_not_14 (n : ℕ) : (n < 500) → (n % 7 = 0) → (n % 14 ≠ 0) → ∃ k, (k = 36) :=
by {
  sorry
}

end multiples_7_not_14_l159_159260


namespace gcd_fx_x_l159_159886

-- Let x be an instance of ℤ
variable (x : ℤ)

-- Define that x is a multiple of 46200
def is_multiple_of_46200 := ∃ k : ℤ, x = 46200 * k

-- Define the function f(x) = (3x + 5)(5x + 3)(11x + 6)(x + 11)
def f (x : ℤ) := (3 * x + 5) * (5 * x + 3) * (11 * x + 6) * (x + 11)

-- The statement to prove
theorem gcd_fx_x (h : is_multiple_of_46200 x) : Int.gcd (f x) x = 990 := 
by
  -- Placeholder for the proof
  sorry

end gcd_fx_x_l159_159886


namespace ratio_of_x_to_y_l159_159602

-- Defining the given condition
def ratio_condition (x y : ℝ) : Prop :=
  (3 * x - 2 * y) / (2 * x + y) = 3 / 5

-- The theorem to be proven
theorem ratio_of_x_to_y (x y : ℝ) (h : ratio_condition x y) : x / y = 13 / 9 :=
by
  sorry

end ratio_of_x_to_y_l159_159602


namespace geometric_sequence_third_sixth_term_l159_159491

theorem geometric_sequence_third_sixth_term (a r : ℝ) 
  (h3 : a * r^2 = 18) 
  (h6 : a * r^5 = 162) : 
  a = 2 ∧ r = 3 := 
sorry

end geometric_sequence_third_sixth_term_l159_159491


namespace inscribed_square_sum_c_d_eq_200689_l159_159366

theorem inscribed_square_sum_c_d_eq_200689 :
  ∃ (c d : ℕ), Nat.gcd c d = 1 ∧ (∃ x : ℚ, x = (c : ℚ) / (d : ℚ) ∧ 
    let a := 48
    let b := 55
    let longest_side := 73
    let s := (a + b + longest_side) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - longest_side))
    area = 1320 ∧ x = 192720 / 7969 ∧ c + d = 200689) :=
sorry

end inscribed_square_sum_c_d_eq_200689_l159_159366


namespace negation_of_exists_abs_le_two_l159_159645

theorem negation_of_exists_abs_le_two :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) :=
by
  sorry

end negation_of_exists_abs_le_two_l159_159645


namespace population_relation_l159_159303

-- Conditions: average life expectancies
def life_expectancy_gondor : ℝ := 64
def life_expectancy_numenor : ℝ := 92
def combined_life_expectancy (g n : ℕ) : ℝ := 85

-- Proof Problem: Given the conditions, prove the population relation
theorem population_relation (g n : ℕ) (h1 : life_expectancy_gondor * g + life_expectancy_numenor * n = combined_life_expectancy g n * (g + n)) : n = 3 * g :=
by
  sorry

end population_relation_l159_159303


namespace augmented_wedge_volume_proof_l159_159850

open Real

noncomputable def sphere_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * π)

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def wedge_volume (volume_sphere : ℝ) (number_of_wedges : ℕ) : ℝ :=
  volume_sphere / number_of_wedges

noncomputable def augmented_wedge_volume (original_wedge_volume : ℝ) : ℝ :=
  2 * original_wedge_volume

theorem augmented_wedge_volume_proof (circumference : ℝ) (number_of_wedges : ℕ) 
  (volume : ℝ) (augmented_volume : ℝ) :
  circumference = 18 * π →
  number_of_wedges = 6 →
  volume = sphere_volume (sphere_radius circumference) →
  augmented_volume = augmented_wedge_volume (wedge_volume volume number_of_wedges) →
  augmented_volume = 324 * π :=
by
  intros h_circ h_wedges h_vol h_aug_vol
  -- This is where the proof steps would go
  sorry

end augmented_wedge_volume_proof_l159_159850


namespace total_distance_walked_l159_159375

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l159_159375


namespace inequality_proof_l159_159288

variable (a b c d : ℝ)
variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

-- Define conditions
def positive (x : ℝ) := x > 0
def unit_circle (x y : ℝ) := x^2 + y^2 = 1

-- Define the main theorem
theorem inequality_proof
  (ha : positive a)
  (hb : positive b)
  (hc : positive c)
  (hd : positive d)
  (habcd : a * b + c * d = 1)
  (hP1 : unit_circle x1 y1)
  (hP2 : unit_circle x2 y2)
  (hP3 : unit_circle x3 y3)
  (hP4 : unit_circle x4 y4)
  : 
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := sorry

end inequality_proof_l159_159288


namespace xy_system_sol_l159_159416

theorem xy_system_sol (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^3 + y^3 = 416000 / 729 :=
by
  sorry

end xy_system_sol_l159_159416


namespace factor_expression_l159_159548

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l159_159548


namespace average_weight_increase_per_month_l159_159166

theorem average_weight_increase_per_month (w_initial w_final : ℝ) (t : ℝ) 
  (h_initial : w_initial = 3.25) (h_final : w_final = 7) (h_time : t = 3) :
  (w_final - w_initial) / t = 1.25 := 
by 
  sorry

end average_weight_increase_per_month_l159_159166


namespace meaningful_expression_l159_159660

theorem meaningful_expression (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 1))) → x > 1 :=
by sorry

end meaningful_expression_l159_159660


namespace probability_of_yellow_jelly_bean_l159_159085

theorem probability_of_yellow_jelly_bean (P_red P_orange P_green P_yellow : ℝ)
  (h_red : P_red = 0.1)
  (h_orange : P_orange = 0.4)
  (h_green : P_green = 0.25)
  (h_total : P_red + P_orange + P_green + P_yellow = 1) :
  P_yellow = 0.25 :=
by
  sorry

end probability_of_yellow_jelly_bean_l159_159085


namespace fifth_number_in_10th_row_l159_159970

theorem fifth_number_in_10th_row : 
  ∀ (n : ℕ), (∃ (a : ℕ), ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 10 → (m = 10 → a = 67)) :=
by
  sorry

end fifth_number_in_10th_row_l159_159970


namespace four_inv_mod_35_l159_159231

theorem four_inv_mod_35 : ∃ x : ℕ, 4 * x ≡ 1 [MOD 35] ∧ x = 9 := 
by 
  use 9
  sorry

end four_inv_mod_35_l159_159231


namespace sin_theta_fourth_quadrant_l159_159707

-- Given conditions
variables {θ : ℝ} (h1 : Real.cos θ = 1 / 3) (h2 : 3 * pi / 2 < θ ∧ θ < 2 * pi)

-- Proof statement
theorem sin_theta_fourth_quadrant : Real.sin θ = -2 * Real.sqrt 2 / 3 :=
sorry

end sin_theta_fourth_quadrant_l159_159707


namespace ball_bounce_height_lt_one_l159_159354

theorem ball_bounce_height_lt_one :
  ∃ (k : ℕ), 15 * (1/3:ℝ)^k < 1 ∧ k = 3 := 
sorry

end ball_bounce_height_lt_one_l159_159354


namespace simplify_and_evaluate_expr_l159_159175

variables (a b : Int)

theorem simplify_and_evaluate_expr (ha : a = 1) (hb : b = -2) : 
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 :=
by
  sorry

end simplify_and_evaluate_expr_l159_159175


namespace team_total_points_l159_159219

theorem team_total_points :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  team_total = 89 :=
by
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  sorry

end team_total_points_l159_159219


namespace hyperbola_standard_equation_l159_159135

variable {a b c : Real}

def hyperbola_eq : Prop := 
  (∀ x y : Real, (x^2 / a^2 - y^2 / b^2 = 1) → Exists (λ s t : Real, s^2 + t^2 = c^2 ∧ s = t * (3/4) ∧ a^2 = 9 ∧ b^2 = 16))

theorem hyperbola_standard_equation 
  (parabola_directrix : -5)
  (left_focus : ((-5 : Real), 0))
  (asymptote_slope : 4 / 3)
  (sq_sum_eq : a^2 + b^2 = 25) 
: hyperbola_eq := 
by sorry

end hyperbola_standard_equation_l159_159135


namespace problem1_problem2_l159_159578

-- Define A and B as given
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y - 5 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

-- Problem statement 1: Prove 3A + 6B simplifies as expected
theorem problem1 (x y : ℝ) : 3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9 :=
  by
    sorry

-- Problem statement 2: Prove that if 3A + 6B is independent of x, then y = -5
theorem problem2 (y : ℝ) (h : ∀ x : ℝ, 3 * A x y + 6 * B x y = -9) : y = -5 :=
  by
    sorry

end problem1_problem2_l159_159578


namespace subcommittee_count_l159_159781

theorem subcommittee_count :
  let total_members := 12
  let total_teachers := 5
  let subcommittee_size := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_teacher_subcommittees_with_0_teachers := Nat.choose (total_members - total_teachers) subcommittee_size
  let non_teacher_subcommittees_with_1_teacher :=
    Nat.choose total_teachers 1 * Nat.choose (total_members - total_teachers) (subcommittee_size - 1)
  (total_subcommittees
   - (non_teacher_subcommittees_with_0_teachers + non_teacher_subcommittees_with_1_teacher)) = 596 := 
by
  sorry

end subcommittee_count_l159_159781


namespace simplify_expression_l159_159391

theorem simplify_expression (b : ℝ) (h : b ≠ 1 / 2) : 1 - (2 / (1 + (b / (1 - 2 * b)))) = (3 * b - 1) / (1 - b) :=
by
    sorry

end simplify_expression_l159_159391


namespace problem1_l159_159267

theorem problem1 (x y : ℝ) (h : |x + 1| + (2 * x - y)^2 = 0) : x^2 - y = 3 :=
sorry

end problem1_l159_159267


namespace math_score_is_75_l159_159638

def average_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4
def total_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := s1 + s2 + s3 + s4
def average_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := (s1 + s2 + s3 + s4 + s5) / 5
def total_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := s1 + s2 + s3 + s4 + s5

theorem math_score_is_75 (s1 s2 s3 s4 : ℕ) (h1 : average_of_four_subjects s1 s2 s3 s4 = 90)
                            (h2 : average_of_five_subjects s1 s2 s3 s4 s5 = 87) :
  s5 = 75 :=
by
  sorry

end math_score_is_75_l159_159638


namespace population_after_10_years_l159_159061

def initial_population : ℕ := 100000
def birth_increase_percent : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

theorem population_after_10_years :
  let birth_increase := initial_population * birth_increase_percent
  let total_emigration := emigration_per_year * years
  let total_immigration := immigration_per_year * years
  let net_movement := total_immigration - total_emigration
  let final_population := initial_population + birth_increase + net_movement
  final_population = 165000 :=
by
  sorry

end population_after_10_years_l159_159061


namespace distance_AB_polar_l159_159164

open Real

theorem distance_AB_polar (A B : ℝ × ℝ) (θ₁ θ₂ : ℝ) (hA : A = (4, θ₁)) (hB : B = (12, θ₂))
  (hθ : θ₁ - θ₂ = π / 3) : dist (4 * cos θ₁, 4 * sin θ₁) (12 * cos θ₂, 12 * sin θ₂) = 4 * sqrt 13 :=
by
  sorry

end distance_AB_polar_l159_159164


namespace vasya_birthday_was_thursday_l159_159808

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159808


namespace num_chairs_l159_159459

variable (C : Nat)
variable (tables_sticks : Nat := 6 * 9)
variable (stools_sticks : Nat := 4 * 2)
variable (total_sticks_needed : Nat := 34 * 5)
variable (total_sticks_chairs : Nat := 6 * C)

theorem num_chairs (h : total_sticks_chairs + tables_sticks + stools_sticks = total_sticks_needed) : C = 18 := 
by sorry

end num_chairs_l159_159459


namespace binomial_coefficient_third_term_l159_159739

theorem binomial_coefficient_third_term (x a : ℝ) (h : 10 * a^3 * x = 80) : a = 2 :=
by
  sorry

end binomial_coefficient_third_term_l159_159739


namespace sides_of_right_triangle_l159_159776

theorem sides_of_right_triangle (r : ℝ) (a b c : ℝ) 
  (h_area : (2 / (2 / r)) * 2 = 2 * r) 
  (h_right : a^2 + b^2 = c^2) :
  (a = r ∧ b = (4 / 3) * r ∧ c = (5 / 3) * r) ∨
  (b = r ∧ a = (4 / 3) * r ∧ c = (5 / 3) * r) :=
sorry

end sides_of_right_triangle_l159_159776


namespace bryden_collection_value_l159_159844

-- Define the conditions
def face_value_half_dollar : ℝ := 0.5
def face_value_quarter : ℝ := 0.25
def num_half_dollars : ℕ := 5
def num_quarters : ℕ := 3
def multiplier : ℝ := 30

-- Define the problem statement as a theorem
theorem bryden_collection_value : 
  (multiplier * (num_half_dollars * face_value_half_dollar + num_quarters * face_value_quarter)) = 97.5 :=
by
  -- Proof is skipped since it's not required
  sorry

end bryden_collection_value_l159_159844


namespace line_circle_no_intersection_l159_159147

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end line_circle_no_intersection_l159_159147


namespace range_of_a_l159_159152

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l159_159152


namespace measure_angle_R_l159_159747

-- Given conditions
variables {P Q R : Type}
variable {x : ℝ} -- x represents the measure of angles P and Q

-- Setting up the given conditions
def isosceles_triangle (P Q R : Type) (x : ℝ) : Prop :=
  x + x + (x + 40) = 180

-- Statement we need to prove
theorem measure_angle_R (P Q R : Type) (x : ℝ) (h : isosceles_triangle P Q R x) : ∃ r : ℝ, r = 86.67 :=
by {
  sorry
}

end measure_angle_R_l159_159747


namespace math_problem_solution_l159_159312

noncomputable def a_range : Set ℝ := {a : ℝ | (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)}

theorem math_problem_solution (a : ℝ) :
  (1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∨ ((a - 3)^2 - 4 < 0)
  ∧ ¬((1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∧ ((a - 3)^2 - 4 < 0)) →
  a ∈ a_range :=
sorry

end math_problem_solution_l159_159312


namespace remainder_8_pow_310_mod_9_l159_159812

theorem remainder_8_pow_310_mod_9 : (8 ^ 310) % 9 = 8 := 
by
  sorry

end remainder_8_pow_310_mod_9_l159_159812


namespace largest_value_among_given_numbers_l159_159369

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem largest_value_among_given_numbers :
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20 
  b > a ∧ b > c ∧ b > d :=
by
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20
  -- Add the necessary steps to show that b is the largest value
  sorry

end largest_value_among_given_numbers_l159_159369


namespace coefficient_of_x_in_binomial_expansion_coefficient_of_x_in_x_sub_2_exp_5_l159_159481

theorem coefficient_of_x_in_binomial_expansion :
  (binom 5 4 * (-2)^4) = 80 :=
by lint exactly -- This part includes lint checking for matching precisely.

theorem coefficient_of_x_in_x_sub_2_exp_5 :
  coefficient_of_x_in_binomial_expansion := -- Using the previous theorem directly.
begin
  sorry -- Here we skip the actual detailed proof because it's mentioned we don't need to consider solution steps.
end

end coefficient_of_x_in_binomial_expansion_coefficient_of_x_in_x_sub_2_exp_5_l159_159481


namespace derivative_of_f_l159_159128

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : deriv f x = ((x * Real.exp x - Real.exp x) / (x * x)) :=
by
  sorry

end derivative_of_f_l159_159128


namespace total_boys_went_down_slide_l159_159656

-- Definitions according to the conditions given
def boys_went_down_slide1 : ℕ := 22
def boys_went_down_slide2 : ℕ := 13

-- The statement to be proved
theorem total_boys_went_down_slide : boys_went_down_slide1 + boys_went_down_slide2 = 35 := 
by 
  sorry

end total_boys_went_down_slide_l159_159656


namespace sufficient_but_not_necessary_condition_l159_159882

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) : (a - b) * a^2 < 0 → a < b :=
sorry

end sufficient_but_not_necessary_condition_l159_159882


namespace tower_remainder_l159_159529

def num_towers : ℕ := 907200  -- the total number of different towers S for 9 cubes

theorem tower_remainder : num_towers % 1000 = 200 :=
by
  sorry

end tower_remainder_l159_159529


namespace dot_product_AB_BC_l159_159752

variable (AB AC : ℝ × ℝ)

def BC (AB AC : ℝ × ℝ) : ℝ × ℝ := (AC.1 - AB.1, AC.2 - AB.2)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

theorem dot_product_AB_BC :
  ∀ (AB AC : ℝ × ℝ), AB = (2, 3) → AC = (3, 4) →
  dot_product AB (BC AB AC) = 5 :=
by
  intros
  unfold BC
  unfold dot_product
  sorry

end dot_product_AB_BC_l159_159752


namespace line_parabola_one_point_l159_159779

theorem line_parabola_one_point (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 2 ∧ y^2 = 8 * x) 
  → (k = 0 ∨ k = 1) := 
by 
  sorry

end line_parabola_one_point_l159_159779


namespace integer_count_between_cubes_l159_159724

-- Definitions and conditions
def a : ℝ := 10.7
def b : ℝ := 10.8

-- Precomputed values
def a_cubed : ℝ := 1225.043
def b_cubed : ℝ := 1259.712

-- The theorem to prove
theorem integer_count_between_cubes (ha : a ^ 3 = a_cubed) (hb : b ^ 3 = b_cubed) :
  let start := Int.ceil a_cubed
  let end_ := Int.floor b_cubed
  end_ - start + 1 = 34 :=
by
  sorry

end integer_count_between_cubes_l159_159724


namespace quadratic_roots_square_cube_sum_l159_159783

theorem quadratic_roots_square_cube_sum
  (a b c : ℝ) (h : a ≠ 0) (x1 x2 : ℝ)
  (hx : ∀ (x : ℝ), a * x^2 + b * x + c = 0 ↔ x = x1 ∨ x = x2) :
  (x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2) ∧
  (x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3) :=
by
  sorry

end quadratic_roots_square_cube_sum_l159_159783


namespace powderman_distance_approximates_275_yards_l159_159534

noncomputable def distance_run (t : ℝ) : ℝ := 6 * t
noncomputable def sound_distance (t : ℝ) : ℝ := 1080 * (t - 45) / 3

theorem powderman_distance_approximates_275_yards : 
  ∃ t : ℝ, t > 45 ∧ 
  (distance_run t = sound_distance t) → 
  abs (distance_run t - 275) < 1 :=
by
  sorry

end powderman_distance_approximates_275_yards_l159_159534


namespace quadratic_discriminant_l159_159871

def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/2) (-2) = 281/4 := by
  sorry

end quadratic_discriminant_l159_159871


namespace pencils_total_l159_159754

-- Defining the conditions
def packs_to_pencils (packs : ℕ) : ℕ := packs * 12

def jimin_packs : ℕ := 2
def jimin_individual_pencils : ℕ := 7

def yuna_packs : ℕ := 1
def yuna_individual_pencils : ℕ := 9

-- Translating to Lean 4 statement
theorem pencils_total : 
  packs_to_pencils jimin_packs + jimin_individual_pencils + packs_to_pencils yuna_packs + yuna_individual_pencils = 52 := 
by
  sorry

end pencils_total_l159_159754


namespace sushi_father_lollipops_l159_159300

-- Define the conditions
def lollipops_eaten : ℕ := 5
def lollipops_left : ℕ := 7

-- Define the total number of lollipops brought
def total_lollipops := lollipops_eaten + lollipops_left

-- Proof statement
theorem sushi_father_lollipops : total_lollipops = 12 := sorry

end sushi_father_lollipops_l159_159300


namespace least_number_to_subtract_997_l159_159815

theorem least_number_to_subtract_997 (x : ℕ) (h : x = 997) 
  : ∃ y : ℕ, ∀ m (h₁ : m = (997 - y)), 
    m % 5 = 3 ∧ m % 9 = 3 ∧ m % 11 = 3 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end least_number_to_subtract_997_l159_159815


namespace expression_value_zero_l159_159612

theorem expression_value_zero (a b c : ℝ) (h1 : a^2 + b = b^2 + c) (h2 : b^2 + c = c^2 + a) (h3 : c^2 + a = a^2 + b) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 :=
by
  sorry

end expression_value_zero_l159_159612


namespace determine_a_l159_159574

theorem determine_a :
  ∃ a : ℝ, (∀ x : ℝ, y = -((x - a) / (x - a - 1)) ↔ x = (3 - a) / (3 - a - 1)) → a = 2 :=
sorry

end determine_a_l159_159574


namespace mountaineers_arrangement_l159_159866
open BigOperators

-- Definition to state the number of combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The main statement translating our problem
theorem mountaineers_arrangement :
  (choose 4 2) * (choose 6 2) = 120 := by
  sorry

end mountaineers_arrangement_l159_159866


namespace num_gcd_values_l159_159828

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159828


namespace find_removed_number_l159_159516

theorem find_removed_number (numbers : List ℕ) (avg_remain : ℝ) (h_list : numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]) (h_avg : avg_remain = 7.5) :
  ∃ x, x ∈ numbers ∧ 
       (let numbers_removed := numbers.erase x in 
        (numbers.sum - x) / (numbers.length - 1) = avg_remain) := 
by
  sorry

end find_removed_number_l159_159516


namespace solve_for_x_l159_159771

theorem solve_for_x (x : ℤ) (h : 15 * 2 = x - 3 + 5) : x = 28 :=
sorry

end solve_for_x_l159_159771


namespace product_of_largest_two_and_four_digit_primes_l159_159657

theorem product_of_largest_two_and_four_digit_primes :
  let largest_two_digit_prime := 97
  let largest_four_digit_prime := 9973
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end product_of_largest_two_and_four_digit_primes_l159_159657


namespace part1_part2_part3_l159_159654

-- Part 1
theorem part1 (B_count : ℕ) : 
  (1 * 100) + (B_count * 68) + (4 * 20) = 520 → 
  B_count = 5 := 
by sorry

-- Part 2
theorem part2 (A_count B_count : ℕ) : 
  A_count + B_count = 5 → 
  (100 * A_count) + (68 * B_count) = 404 → 
  A_count = 2 ∧ B_count = 3 := 
by sorry

-- Part 3
theorem part3 : 
  ∃ (A_count B_count C_count : ℕ), 
  (A_count <= 16) ∧ (B_count <= 16) ∧ (C_count <= 16) ∧ 
  (A_count + B_count + C_count <= 16) ∧ 
  (100 * A_count + 68 * B_count = 708 ∨ 
   68 * B_count + 20 * C_count = 708 ∨ 
   100 * A_count + 20 * C_count = 708) → 
  ((A_count = 3 ∧ B_count = 6 ∧ C_count = 0) ∨ 
   (A_count = 0 ∧ B_count = 6 ∧ C_count = 15)) := 
by sorry

end part1_part2_part3_l159_159654


namespace bullet_speed_difference_l159_159993

theorem bullet_speed_difference
  (horse_speed : ℝ := 20) 
  (bullet_speed : ℝ := 400) : 
  ((bullet_speed + horse_speed) - (bullet_speed - horse_speed) = 40) := by
  sorry

end bullet_speed_difference_l159_159993


namespace range_of_m_l159_159055

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/(9 - m) + (y^2)/(m - 5) = 1 → 
  (∃ m, (7 < m ∧ m < 9))) := 
sorry

end range_of_m_l159_159055


namespace smallest_positive_root_l159_159392

noncomputable def alpha := Real.arctan (14 / 3)
noncomputable def beta := Real.arctan (13 / 6)

theorem smallest_positive_root:
  ∀ (x : ℝ), (14 * Real.sin(3 * x) - 3 * Real.cos(3 * x) = 13 * Real.sin(2 * x) - 6 * Real.cos(2 * x)) →
  (0 < x) →
  x = (2 * Real.pi - alpha - beta) / 5 :=
by
  sorry

end smallest_positive_root_l159_159392


namespace train_crosses_pole_in_9_seconds_l159_159210

noncomputable def train_crossing_time : ℝ :=
  let speed_km_hr := 72      -- Speed in kilometers per hour
  let speed_m_s := speed_km_hr * (1000 / 3600)  -- Convert speed to meters per second
  let distance_m := 180      -- Length of the train in meters
  distance_m / speed_m_s     -- Time = Distance / Speed

theorem train_crosses_pole_in_9_seconds :
  train_crossing_time = 9 :=
by
  let speed_km_hr := 72
  let speed_m_s := speed_km_hr * (1000 / 3600)
  let distance_m := 180
  have h1 : speed_m_s = 20 := by norm_num [speed_m_s]
  have h2 : train_crossing_time = distance_m / speed_m_s := rfl
  have h3 : distance_m / speed_m_s = 9 := by norm_num [distance_m, h1]
  rwa [←h2, h3]

end train_crosses_pole_in_9_seconds_l159_159210


namespace triangle_problem_l159_159741

noncomputable def length_of_side_c (a : ℝ) (cosB : ℝ) (C : ℝ) : ℝ :=
  a * (Real.sqrt 2 / 2) / (Real.sqrt (1 - cosB^2))

noncomputable def cos_A_minus_pi_over_6 (cosB : ℝ) (cosA : ℝ) (sinA : ℝ) : ℝ :=
  cosA * (Real.sqrt 3 / 2) + sinA * (1 / 2)

theorem triangle_problem (a : ℝ) (cosB : ℝ) (C : ℝ) 
  (ha : a = 6) (hcosB : cosB = 4/5) (hC : C = Real.pi / 4) : 
  (length_of_side_c a cosB C = 5 * Real.sqrt 2) ∧ 
  (cos_A_minus_pi_over_6 cosB (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2)))) (Real.sqrt (1 - (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2))))^2)) = (7 * Real.sqrt 2 - Real.sqrt 6) / 20) :=
by 
  sorry

end triangle_problem_l159_159741


namespace payment_to_C_l159_159995

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℚ := 3360

def work_done (rate : ℚ) (days : ℕ) : ℚ := rate * days

-- Conditions
def person_A_work_rate := work_rate 6
def person_B_work_rate := work_rate 8
def combined_work_rate := person_A_work_rate + person_B_work_rate
def work_by_A_and_B_in_3_days := work_done combined_work_rate 3
def total_work : ℚ := 1
def work_done_by_C := total_work - work_by_A_and_B_in_3_days

-- Proof problem statement
theorem payment_to_C :
  (work_done_by_C / total_work) * total_payment = 420 := 
sorry

end payment_to_C_l159_159995


namespace tangent_line_at_x_2_range_of_m_for_three_roots_l159_159407

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

/-
Part 1: Proving the tangent line equation at x = 2
-/
theorem tangent_line_at_x_2 : ∃ k b, (k = 12) ∧ (b = -17) ∧ 
  (∀ x, 12 * x - f 2 - 17 = 0) :=
by
  sorry

/-
Part 2: Proving the range of m for three distinct real roots
-/
theorem range_of_m_for_three_roots (m : ℝ) :
  (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 + m = 0 ∧ f x2 + m = 0 ∧ f x3 + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
by
  sorry

end tangent_line_at_x_2_range_of_m_for_three_roots_l159_159407


namespace min_value_of_3x_plus_4y_l159_159908

open Real

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

end min_value_of_3x_plus_4y_l159_159908


namespace paint_pyramid_l159_159040

theorem paint_pyramid (colors : Finset ℕ) (n : ℕ) (h : colors.card = 5) :
  let ways_to_paint := 5 * 4 * 3 * 2 * 1
  n = ways_to_paint
:=
sorry

end paint_pyramid_l159_159040


namespace vasya_birthday_is_thursday_l159_159803

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159803


namespace arithmetic_sequence_a7_l159_159750

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)) : a 7 = 9 :=
by
  sorry

end arithmetic_sequence_a7_l159_159750


namespace train_platform_ratio_l159_159643

noncomputable def speed_km_per_hr := 216 -- condition 1
noncomputable def crossing_time_sec := 60 -- condition 2
noncomputable def train_length_m := 1800 -- condition 3

noncomputable def speed_m_per_s := speed_km_per_hr * 1000 / 3600
noncomputable def total_distance_m := speed_m_per_s * crossing_time_sec
noncomputable def platform_length_m := total_distance_m - train_length_m
noncomputable def ratio := train_length_m / platform_length_m

theorem train_platform_ratio : ratio = 1 := by
    sorry

end train_platform_ratio_l159_159643


namespace mutually_exclusive_events_eq_one_l159_159239

-- Define the problem using Lean 4.

def number_of_mutually_exclusive_events : Nat := 1

theorem mutually_exclusive_events_eq_one :
  let E1 := λ (draw: Finset (Sum ℕ ℕ)), (∃ w : ℕ, w ≥ 1) ∧ (draw.card = w + 2)
  let E2 := λ (draw: Finset (Sum ℕ ℕ)), (∃ w : ℕ, w ≥ 1) ∧ (∃ r : ℕ, r ≥ 1)
  let E3 := λ (draw: Finset (Sum ℕ ℕ)), (draw.count(1) = 1) ∧ (draw.count(0) = 2)
  let E4 := λ (draw: Finset (Sum ℕ ℕ)), (∃ w : ℕ, w ≥ 1) ∧ (∀ x : (Sum ℕ ℕ), ¬(x = ℕ))
  (∀ draws: Finset (Sum ℕ ℕ), (E1 draws) ∧ (¬-E2 draws)) -- exclusive check considering draws
  ∧ (number_of_mutually_exclusive_events = 1) :=
sorry -- proof placeholder

end mutually_exclusive_events_eq_one_l159_159239


namespace clubs_equal_students_l159_159427

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l159_159427


namespace constant_function_l159_159762

theorem constant_function {f : ℕ → ℕ} (h : ∀ x y : ℕ, x * f y + y * f x = (x + y) * f (x^2 + y^2)) : ∃ c : ℕ, ∀ x, f x = c := 
sorry

end constant_function_l159_159762


namespace cost_price_percentage_l159_159179

theorem cost_price_percentage (SP CP : ℝ) (hp : SP - CP = (1/3) * CP) : CP = 0.75 * SP :=
by
  sorry

end cost_price_percentage_l159_159179


namespace quad_in_vertex_form_addition_l159_159604

theorem quad_in_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∃ a h k, (4 * x^2 - 8 * x + 3) = a * (x - h) ^ 2 + k) →
  a + h + k = 4 :=
by
  sorry

end quad_in_vertex_form_addition_l159_159604


namespace num_solutions_20_l159_159039

-- Define the number of integer solutions function
def num_solutions (n : ℕ) : ℕ := 4 * n

-- Given conditions
axiom h1 : num_solutions 1 = 4
axiom h2 : num_solutions 2 = 8

-- Theorem to prove the number of solutions for |x| + |y| = 20 is 80
theorem num_solutions_20 : num_solutions 20 = 80 :=
by sorry

end num_solutions_20_l159_159039


namespace fruits_eaten_total_l159_159788

variable (apples blueberries bonnies : ℕ)

noncomputable def total_fruits_eaten : ℕ :=
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 / 4 * third_dog_bonnies
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies

theorem fruits_eaten_total:
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 * third_dog_bonnies / 4
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies = 240 := by
  sorry

end fruits_eaten_total_l159_159788


namespace math_problem_l159_159453

theorem math_problem (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hy_reverse : ∃ a b, x = 10 * a + b ∧ y = 10 * b + a) 
  (h_xy_square_sum : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end math_problem_l159_159453


namespace juniper_bones_proof_l159_159162

-- Define the conditions
def juniper_original_bones : ℕ := 4
def bones_given_by_master : ℕ := juniper_original_bones
def bones_stolen_by_neighbor : ℕ := 2

-- Define the final number of bones Juniper has
def juniper_remaining_bones : ℕ := juniper_original_bones + bones_given_by_master - bones_stolen_by_neighbor

-- State the theorem to prove the given answer
theorem juniper_bones_proof : juniper_remaining_bones = 6 :=
by
  -- Proof omitted
  sorry

end juniper_bones_proof_l159_159162


namespace trip_time_difference_l159_159842

theorem trip_time_difference
  (avg_speed : ℝ)
  (dist1 dist2 : ℝ)
  (h_avg_speed : avg_speed = 60)
  (h_dist1 : dist1 = 540)
  (h_dist2 : dist2 = 570) :
  ((dist2 - dist1) / avg_speed) * 60 = 30 := by
  sorry

end trip_time_difference_l159_159842


namespace probability_two_red_two_blue_l159_159667

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (12.choose 2 * 8.choose 2) / (20.choose 4) = 168 / 323 :=
  sorry

end probability_two_red_two_blue_l159_159667


namespace gcd_198_286_l159_159329

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end gcd_198_286_l159_159329
