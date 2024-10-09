import Mathlib

namespace coefficient_x4_expansion_eq_7_l1060_106018

theorem coefficient_x4_expansion_eq_7 (a : ℝ) : 
  (∀ r : ℕ, 8 - (4 * r) / 3 = 4 → (a ^ r) * (Nat.choose 8 r) = 7) → a = 1 / 2 :=
by
  sorry

end coefficient_x4_expansion_eq_7_l1060_106018


namespace sum_of_digits_l1060_106065

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_of_digits (a b c d : ℕ) (h_distinct : distinct_digits a b c d) (h_eqn : 100*a + 60 + b - (400 + 10*c + d) = 2) :
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
sorry

end sum_of_digits_l1060_106065


namespace range_of_a_l1060_106006

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a > 0

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a) (h2 : q a) : a ≤ -2 :=
by
  sorry

end range_of_a_l1060_106006


namespace problem1_problem2_l1060_106099

-- Problem 1: Prove that the solutions of x^2 + 6x - 7 = 0 are x = -7 and x = 1
theorem problem1 (x : ℝ) : x^2 + 6*x - 7 = 0 ↔ (x = -7 ∨ x = 1) := by
  -- Proof omitted
  sorry

-- Problem 2: Prove that the solutions of 4x(2x+1) = 3(2x+1) are x = -1/2 and x = 3/4
theorem problem2 (x : ℝ) : 4*x*(2*x + 1) = 3*(2*x + 1) ↔ (x = -1/2 ∨ x = 3/4) := by
  -- Proof omitted
  sorry

end problem1_problem2_l1060_106099


namespace max_band_members_l1060_106070

theorem max_band_members (n : ℤ) (h1 : 22 * n % 24 = 2) (h2 : 22 * n < 1000) : 22 * n = 770 :=
  sorry

end max_band_members_l1060_106070


namespace product_value_l1060_106048

noncomputable def product_of_sequence : ℝ :=
  (1/3) * 9 * (1/27) * 81 * (1/243) * 729 * (1/2187) * 6561

theorem product_value : product_of_sequence = 729 := by
  sorry

end product_value_l1060_106048


namespace expand_product_equivalence_l1060_106072

variable (x : ℝ)  -- Assuming x is a real number

theorem expand_product_equivalence : (x + 5) * (x + 7) = x^2 + 12 * x + 35 :=
by
  sorry

end expand_product_equivalence_l1060_106072


namespace evaluate_expression_l1060_106057

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 4 - 2 * g (-2) = 47 :=
by
  sorry

end evaluate_expression_l1060_106057


namespace production_rate_problem_l1060_106014

theorem production_rate_problem :
  ∀ (G T : ℕ), 
  (∀ w t, w * 3 * t = 450 * t / 150) ∧
  (∀ w t, w * 2 * t = 300 * t / 150) ∧
  (∀ w t, w * 2 * t = 360 * t / 90) ∧
  (∀ w t, w * (5/2) * t = 450 * t / 90) ∧
  (75 * 2 * 4 = 300) →
  (75 * 2 * 4 = 600) := sorry

end production_rate_problem_l1060_106014


namespace problem_1_simplification_l1060_106087

theorem problem_1_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) : 
  (x - 2) / (x ^ 2) / (1 - 2 / x) = 1 / x := 
  sorry

end problem_1_simplification_l1060_106087


namespace sum_of_coefficients_of_poly_is_neg_1_l1060_106092

noncomputable def evaluate_poly_sum (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) : ℂ :=
  α^2005 + β^2005

theorem sum_of_coefficients_of_poly_is_neg_1 (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  evaluate_poly_sum α β h1 h2 = -1 := by
  sorry

end sum_of_coefficients_of_poly_is_neg_1_l1060_106092


namespace total_cost_of_fencing_l1060_106059

theorem total_cost_of_fencing (side_count : ℕ) (cost_per_side : ℕ) (h1 : side_count = 4) (h2 : cost_per_side = 79) : side_count * cost_per_side = 316 := by
  sorry

end total_cost_of_fencing_l1060_106059


namespace maximize_distance_l1060_106020

def front_tire_lifespan : ℕ := 20000
def rear_tire_lifespan : ℕ := 30000
def max_distance : ℕ := 24000

theorem maximize_distance : max_distance = 24000 := sorry

end maximize_distance_l1060_106020


namespace max_value_of_y_l1060_106021

noncomputable def max_value_of_function : ℝ := 1 + Real.sqrt 2

theorem max_value_of_y : ∀ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) ≤ max_value_of_function :=
by
  -- Proof goes here
  sorry

example : ∃ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) = max_value_of_function :=
by
  -- Proof goes here
  sorry

end max_value_of_y_l1060_106021


namespace calculate_expression_l1060_106012

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end calculate_expression_l1060_106012


namespace volume_of_pyramid_l1060_106042

theorem volume_of_pyramid (V_cube : ℝ) (h : ℝ) (A : ℝ) (V_pyramid : ℝ) : 
  V_cube = 27 → 
  h = 3 → 
  A = 4.5 → 
  V_pyramid = (1/3) * A * h → 
  V_pyramid = 4.5 := 
by 
  intros V_cube_eq h_eq A_eq V_pyramid_eq 
  sorry

end volume_of_pyramid_l1060_106042


namespace trig_identity_sin_eq_l1060_106071

theorem trig_identity_sin_eq (α : ℝ) (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := 
by 
  sorry

end trig_identity_sin_eq_l1060_106071


namespace intersection_roots_l1060_106084

theorem intersection_roots :
  x^2 - 4*x - 5 = 0 → (x = 5 ∨ x = -1) := by
  sorry

end intersection_roots_l1060_106084


namespace inequality_solution_l1060_106061

theorem inequality_solution (a : ℝ)
  (h : ∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → a * x^2 - 2 * x + 2 > 0) :
  a > 1/2 := 
sorry

end inequality_solution_l1060_106061


namespace total_visit_plans_l1060_106026

def exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition", "Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def painting_exhibitions : List String := ["Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def non_painting_exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition"]

def num_visit_plans (exhibit_list : List String) (paintings : List String) (non_paintings : List String) : Nat :=
  let case1 := paintings.length * non_paintings.length * 2
  let case2 := if paintings.length >= 2 then 2 else 0
  case1 + case2

theorem total_visit_plans : num_visit_plans exhibitions painting_exhibitions non_painting_exhibitions = 10 :=
  sorry

end total_visit_plans_l1060_106026


namespace no_common_points_l1060_106076

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def curve2 (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

theorem no_common_points :
  ¬ ∃ (x y : ℝ), curve1 x y ∧ curve2 x y :=
by sorry

end no_common_points_l1060_106076


namespace area_difference_of_circles_l1060_106050

theorem area_difference_of_circles : 
  let r1 := 30
  let r2 := 15
  let pi := Real.pi
  900 * pi - 225 * pi = 675 * pi := by
  sorry

end area_difference_of_circles_l1060_106050


namespace Zack_traveled_18_countries_l1060_106088

variables (countries_Alex countries_George countries_Joseph countries_Patrick countries_Zack : ℕ)
variables (h1 : countries_Alex = 24)
variables (h2 : countries_George = countries_Alex / 4)
variables (h3 : countries_Joseph = countries_George / 2)
variables (h4 : countries_Patrick = 3 * countries_Joseph)
variables (h5 : countries_Zack = 2 * countries_Patrick)

theorem Zack_traveled_18_countries :
  countries_Zack = 18 :=
by sorry

end Zack_traveled_18_countries_l1060_106088


namespace derivative_at_pi_l1060_106038

noncomputable def f (x : ℝ) : ℝ := (x^2) / (Real.cos x)

theorem derivative_at_pi : deriv f π = -2 * π :=
by
  sorry

end derivative_at_pi_l1060_106038


namespace sum_first_39_natural_numbers_l1060_106046

theorem sum_first_39_natural_numbers :
  (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_first_39_natural_numbers_l1060_106046


namespace a6_is_3_l1060_106032

noncomputable def a4 := 8 / 2 -- Placeholder for positive root
noncomputable def a8 := 8 / 2 -- Placeholder for the second root (we know they are both the same for now)
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 2) = (a (n + 1))^2

theorem a6_is_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a4_a8: a 4 = a4) (h_a4_a8_root : a 8 = a8) : 
  a 6 = 3 :=
by
  sorry

end a6_is_3_l1060_106032


namespace min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l1060_106044

section ProofProblem

theorem min_value_a_cube_plus_b_cube {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

theorem no_exist_2a_plus_3b_eq_6 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  ¬ (2 * a + 3 * b = 6) :=
sorry

end ProofProblem

end min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l1060_106044


namespace polynomial_value_at_3_l1060_106028

-- Definitions based on given conditions
def f (x : ℕ) : ℕ :=
  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def x := 3

-- Proof statement
theorem polynomial_value_at_3 : f x = 1641 := by
  sorry

end polynomial_value_at_3_l1060_106028


namespace noah_uses_36_cups_of_water_l1060_106058

theorem noah_uses_36_cups_of_water
  (O : ℕ) (hO : O = 4)
  (S : ℕ) (hS : S = 3 * O)
  (W : ℕ) (hW : W = 3 * S) :
  W = 36 := 
  by sorry

end noah_uses_36_cups_of_water_l1060_106058


namespace h_j_h_of_3_l1060_106011

def h (x : ℤ) : ℤ := 5 * x + 2
def j (x : ℤ) : ℤ := 3 * x + 4

theorem h_j_h_of_3 : h (j (h 3)) = 277 := by
  sorry

end h_j_h_of_3_l1060_106011


namespace rational_coefficient_exists_in_binomial_expansion_l1060_106095

theorem rational_coefficient_exists_in_binomial_expansion :
  ∃! (n : ℕ), n > 0 ∧ (∀ r, (r % 3 = 0 → (n - r) % 2 = 0 → n = 7)) :=
by
  sorry

end rational_coefficient_exists_in_binomial_expansion_l1060_106095


namespace minimum_value_of_reciprocal_product_l1060_106000

theorem minimum_value_of_reciprocal_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + a * b + 2 * b = 30) : 
  ∃ m : ℝ, m = 1 / (a * b) ∧ m = 1 / 18 :=
sorry

end minimum_value_of_reciprocal_product_l1060_106000


namespace parabola_vertex_is_two_one_l1060_106049

theorem parabola_vertex_is_two_one : 
  ∀ x y : ℝ, (y = (x - 2)^2 + 1) → (2, 1) = (2, 1) :=
by
  intros x y hyp
  sorry

end parabola_vertex_is_two_one_l1060_106049


namespace find_slope_l1060_106097

noncomputable def parabola_equation (x y : ℝ) := y^2 = 8 * x

def point_M : ℝ × ℝ := (-2, 2)

def line_through_focus (k x : ℝ) : ℝ := k * (x - 2)

def focus : ℝ × ℝ := (2, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_slope (k : ℝ) : 
  (∀ x y A B, 
    parabola_equation x y → 
    (x = A ∨ x = B) → 
    line_through_focus k x = y → 
    parabola_equation A (k * (A - 2)) → 
    parabola_equation B (k * (B - 2)) → 
    dot_product (A + 2, (k * (A -2)) - 2) (B + 2, (k * (B - 2)) - 2) = 0) →
  k = 2 :=
sorry

end find_slope_l1060_106097


namespace exists_quadratic_satisfying_conditions_l1060_106022

theorem exists_quadratic_satisfying_conditions :
  ∃ (a b c : ℝ), 
  (a - b + c = 0) ∧
  (∀ x : ℝ, x ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧ 
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
  sorry

end exists_quadratic_satisfying_conditions_l1060_106022


namespace Abby_wins_if_N_2011_Brian_wins_in_31_cases_l1060_106056

-- Definitions and assumptions directly from the problem conditions
inductive Player
| Abby
| Brian

def game_condition (N : ℕ) : Prop :=
  ∀ (p : Player), 
    (p = Player.Abby → (∃ k, N = 2 * k + 1)) ∧ 
    (p = Player.Brian → (∃ k, N = 2 * (2^k - 1))) -- This encodes the winning state conditions for simplicity

-- Part (a)
theorem Abby_wins_if_N_2011 : game_condition 2011 :=
by
  sorry

-- Part (b)
theorem Brian_wins_in_31_cases : 
  (∃ S : Finset ℕ, (∀ N ∈ S, N ≤ 2011 ∧ game_condition N) ∧ S.card = 31) :=
by
  sorry

end Abby_wins_if_N_2011_Brian_wins_in_31_cases_l1060_106056


namespace factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l1060_106010

theorem factorize_polynomial_1 (x y : ℝ) : 
  12 * x ^ 3 * y - 3 * x * y ^ 2 = 3 * x * y * (4 * x ^ 2 - y) := 
by sorry

theorem factorize_polynomial_2 (x : ℝ) : 
  x - 9 * x ^ 3 = x * (1 + 3 * x) * (1 - 3 * x) :=
by sorry

theorem factorize_polynomial_3 (a b : ℝ) : 
  3 * a ^ 2 - 12 * a * b * (a - b) = 3 * (a - 2 * b) ^ 2 := 
by sorry

end factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l1060_106010


namespace additional_cards_l1060_106060

theorem additional_cards (total_cards : ℕ) (num_decks : ℕ) (cards_per_deck : ℕ) 
  (h1 : total_cards = 319) (h2 : num_decks = 6) (h3 : cards_per_deck = 52) : 
  319 - 6 * 52 = 7 := 
by
  sorry

end additional_cards_l1060_106060


namespace find_second_angle_l1060_106017

noncomputable def angle_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem find_second_angle
  (A B C : ℝ)
  (hA : A = 32)
  (hC : C = 2 * A - 12)
  (hB : B = 3 * A)
  (h_sum : angle_in_triangle A B C) :
  B = 96 :=
by sorry

end find_second_angle_l1060_106017


namespace find_g_plus_h_l1060_106069

theorem find_g_plus_h (g h : ℚ) (d : ℚ) 
  (h_prod : (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) :
  g + h = -107 / 24 :=
sorry

end find_g_plus_h_l1060_106069


namespace election_result_l1060_106037

theorem election_result:
  ∀ (Henry_votes India_votes Jenny_votes Ken_votes Lena_votes : ℕ)
    (counted_percentage : ℕ)
    (counted_votes : ℕ), 
    Henry_votes = 14 → 
    India_votes = 11 → 
    Jenny_votes = 10 → 
    Ken_votes = 8 → 
    Lena_votes = 2 → 
    counted_percentage = 90 → 
    counted_votes = 45 → 
    (counted_percentage * Total_votes / 100 = counted_votes) →
    (Total_votes = counted_votes * 100 / counted_percentage) →
    (Remaining_votes = Total_votes - counted_votes) →
    ((Henry_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (India_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (Jenny_votes + Max_remaining_Votes >= Max_votes)) →
    3 = 
    (if Henry_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if India_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if Jenny_votes + Remaining_votes > Max_votes then 1 else 0) := 
  sorry

end election_result_l1060_106037


namespace necessary_but_not_sufficient_condition_not_sufficient_condition_l1060_106029

theorem necessary_but_not_sufficient_condition (x y : ℝ) (h : x > 0) : 
  (x > |y|) → (x > y) :=
by
  sorry

theorem not_sufficient_condition (x y : ℝ) (h : x > 0) :
  ¬ ((x > y) → (x > |y|)) :=
by
  sorry

end necessary_but_not_sufficient_condition_not_sufficient_condition_l1060_106029


namespace men_at_yoga_studio_l1060_106052

open Real

def yoga_men_count (M : ℕ) (avg_weight_men avg_weight_women avg_weight_total : ℝ) (num_women num_total : ℕ) : Prop :=
  avg_weight_men = 190 ∧
  avg_weight_women = 120 ∧
  num_women = 6 ∧
  num_total = 14 ∧
  avg_weight_total = 160 →
  M + num_women = num_total ∧
  (M * avg_weight_men + num_women * avg_weight_women) / num_total = avg_weight_total ∧
  M = 8

theorem men_at_yoga_studio : ∃ M : ℕ, yoga_men_count M 190 120 160 6 14 :=
  by 
  use 8
  sorry

end men_at_yoga_studio_l1060_106052


namespace alice_expected_games_l1060_106005

-- Defining the initial conditions
def skill_levels := Fin 21

def initial_active_player := 0

-- Defining Alice's skill level
def Alice_skill_level := 11

-- Define the tournament structure and conditions
def tournament_round (active: skill_levels) (inactive: Set skill_levels) : skill_levels :=
  sorry

-- Define the expected number of games Alice plays
noncomputable def expected_games_Alice_plays : ℚ :=
  sorry

-- Statement of the problem proving the expected number of games Alice plays
theorem alice_expected_games : expected_games_Alice_plays = 47 / 42 :=
sorry

end alice_expected_games_l1060_106005


namespace trig_expression_equality_l1060_106089

theorem trig_expression_equality :
  (Real.tan (60 * Real.pi / 180) + 2 * Real.sin (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)) 
  = Real.sqrt 2 :=
by
  have h1 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := by sorry
  have h2 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  sorry

end trig_expression_equality_l1060_106089


namespace sum_of_digits_is_10_l1060_106051

def sum_of_digits_of_expression : ℕ :=
  let expression := 2^2010 * 5^2008 * 7
  let simplified := 280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  2 + 8

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2008 * 7 is 10 -/
theorem sum_of_digits_is_10 :
  sum_of_digits_of_expression = 10 :=
by sorry

end sum_of_digits_is_10_l1060_106051


namespace final_price_is_99_l1060_106082

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end final_price_is_99_l1060_106082


namespace cyclist_rejoins_group_time_l1060_106083

noncomputable def travel_time (group_speed cyclist_speed distance : ℝ) : ℝ :=
  distance / (cyclist_speed - group_speed)

theorem cyclist_rejoins_group_time
  (group_speed : ℝ := 35)
  (cyclist_speed : ℝ := 45)
  (distance : ℝ := 10)
  : travel_time group_speed cyclist_speed distance * 2 = 1 / 4 :=
by
  sorry

end cyclist_rejoins_group_time_l1060_106083


namespace cheryl_mms_eaten_l1060_106040

variable (initial_mms : ℕ) (mms_after_dinner : ℕ) (mms_given_to_sister : ℕ) (total_mms_after_lunch : ℕ)

theorem cheryl_mms_eaten (h1 : initial_mms = 25)
                         (h2 : mms_after_dinner = 5)
                         (h3 : mms_given_to_sister = 13)
                         (h4 : total_mms_after_lunch = initial_mms - mms_after_dinner - mms_given_to_sister) :
                         total_mms_after_lunch = 7 :=
by sorry

end cheryl_mms_eaten_l1060_106040


namespace abc_equal_l1060_106094

theorem abc_equal (a b c : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a)
  (h2 : ∀ x : ℝ, b * x^2 + c * x + a ≥ c * x^2 + a * x + b) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l1060_106094


namespace find_p_range_l1060_106019

theorem find_p_range (p : ℝ) (A : ℝ → ℝ) :
  (A = fun x => abs x * x^2 + (p + 2) * x + 1) →
  (∀ x, 0 < x → A x ≠ 0) →
  (-4 < p ∧ p < 0) :=
by
  intro hA h_no_pos_roots
  sorry

end find_p_range_l1060_106019


namespace total_pieces_of_gum_and_candy_l1060_106093

theorem total_pieces_of_gum_and_candy 
  (packages_A : ℕ) (pieces_A : ℕ) (packages_B : ℕ) (pieces_B : ℕ) 
  (packages_C : ℕ) (pieces_C : ℕ) (packages_X : ℕ) (pieces_X : ℕ)
  (packages_Y : ℕ) (pieces_Y : ℕ) 
  (hA : packages_A = 10) (hA_pieces : pieces_A = 4)
  (hB : packages_B = 5) (hB_pieces : pieces_B = 8)
  (hC : packages_C = 13) (hC_pieces : pieces_C = 12)
  (hX : packages_X = 8) (hX_pieces : pieces_X = 6)
  (hY : packages_Y = 6) (hY_pieces : pieces_Y = 10) : 
  packages_A * pieces_A + packages_B * pieces_B + packages_C * pieces_C + 
  packages_X * pieces_X + packages_Y * pieces_Y = 344 := 
by
  sorry

end total_pieces_of_gum_and_candy_l1060_106093


namespace downstream_speed_l1060_106036

def V_u : ℝ := 26
def V_m : ℝ := 28
def V_s : ℝ := V_m - V_u
def V_d : ℝ := V_m + V_s

theorem downstream_speed : V_d = 30 := by
  sorry

end downstream_speed_l1060_106036


namespace bus_ride_cost_l1060_106009

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.85) (h2 : T + B = 9.65) : B = 1.40 :=
sorry

end bus_ride_cost_l1060_106009


namespace onions_left_on_scale_l1060_106034

-- Define the given weights and conditions
def total_weight_of_40_onions : ℝ := 7680 -- in grams
def avg_weight_remaining_onions : ℝ := 190 -- grams
def avg_weight_removed_onions : ℝ := 206 -- grams

-- Converting original weight from kg to grams
def original_weight_kg_to_g (w_kg : ℝ) : ℝ := w_kg * 1000

-- Proof problem
theorem onions_left_on_scale (w_kg : ℝ) (n_total : ℕ) (n_removed : ℕ) 
    (total_weight : ℝ) (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ)
    (h1 : original_weight_kg_to_g w_kg = total_weight)
    (h2 : n_total = 40)
    (h3 : n_removed = 5)
    (h4 : avg_weight_remaining = avg_weight_remaining_onions)
    (h5 : avg_weight_removed = avg_weight_removed_onions) : 
    n_total - n_removed = 35 :=
sorry

end onions_left_on_scale_l1060_106034


namespace imaginary_part_of_z_l1060_106007

open Complex

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : 
  im ((x + I) / (y - I)) = 1 :=
by
  sorry

end imaginary_part_of_z_l1060_106007


namespace geometric_series_common_ratio_l1060_106063

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l1060_106063


namespace sum_of_b_for_quadratic_has_one_solution_l1060_106035

theorem sum_of_b_for_quadratic_has_one_solution :
  (∀ x : ℝ, 3 * x^2 + (b+6) * x + 1 = 0 → 
    ∀ Δ : ℝ, Δ = (b + 6)^2 - 4 * 3 * 1 → 
    Δ = 0 → 
    b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) → 
  (-6 + 2 * Real.sqrt 3 + -6 - 2 * Real.sqrt 3 = -12) := 
by
  sorry

end sum_of_b_for_quadratic_has_one_solution_l1060_106035


namespace angles_of_tangency_triangle_l1060_106078

theorem angles_of_tangency_triangle 
  (A B C : ℝ) 
  (ha : A = 40)
  (hb : B = 80)
  (hc : C = 180 - A - B)
  (a1 b1 c1 : ℝ)
  (ha1 : a1 = (1/2) * (180 - A))
  (hb1 : b1 = (1/2) * (180 - B))
  (hc1 : c1 = 180 - a1 - b1) :
  (a1 = 70 ∧ b1 = 50 ∧ c1 = 60) :=
by sorry

end angles_of_tangency_triangle_l1060_106078


namespace intersection_complement_l1060_106045

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 2}
def complement_U_B : Set ℤ := {x ∈ U | x ∉ B}

theorem intersection_complement :
  A ∩ complement_U_B = {0, 1} :=
by
  sorry

end intersection_complement_l1060_106045


namespace paco_countertop_total_weight_l1060_106075

theorem paco_countertop_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 :=
sorry

end paco_countertop_total_weight_l1060_106075


namespace find_number_l1060_106008

theorem find_number (x : ℤ) (h : 22 * (x - 36) = 748) : x = 70 :=
sorry

end find_number_l1060_106008


namespace sugar_flour_difference_l1060_106030

theorem sugar_flour_difference :
  ∀ (flour_required_kg sugar_required_lb flour_added_kg kg_to_lb),
    flour_required_kg = 2.25 →
    sugar_required_lb = 5.5 →
    flour_added_kg = 1 →
    kg_to_lb = 2.205 →
    (sugar_required_lb / kg_to_lb * 1000) - ((flour_required_kg - flour_added_kg) * 1000) = 1244.8 :=
by
  intros flour_required_kg sugar_required_lb flour_added_kg kg_to_lb
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- sorry is used to skip the actual proof
  sorry

end sugar_flour_difference_l1060_106030


namespace kishore_savings_l1060_106003

noncomputable def total_expenses : ℝ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

def percentage_saved : ℝ := 0.10

theorem kishore_savings (salary : ℝ) :
  (total_expenses + percentage_saved * salary) = salary → 
  (percentage_saved * salary = 2077.78) :=
by
  intros h
  rw [← h]
  sorry

end kishore_savings_l1060_106003


namespace largest_y_value_l1060_106013

theorem largest_y_value (y : ℝ) (h : 3*y^2 + 18*y - 90 = y*(y + 17)) : y ≤ 3 :=
by
  sorry

end largest_y_value_l1060_106013


namespace total_animals_is_200_l1060_106043

-- Definitions for the conditions
def num_cows : Nat := 40
def num_sheep : Nat := 56
def num_goats : Nat := 104

-- The theorem to prove the total number of animals is 200
theorem total_animals_is_200 : num_cows + num_sheep + num_goats = 200 := by
  sorry

end total_animals_is_200_l1060_106043


namespace corn_bag_price_l1060_106085

theorem corn_bag_price
  (cost_seeds: ℕ)
  (cost_fertilizers_pesticides: ℕ)
  (cost_labor: ℕ)
  (total_bags: ℕ)
  (desired_profit_percentage: ℕ)
  (total_cost: ℕ := cost_seeds + cost_fertilizers_pesticides + cost_labor)
  (total_revenue: ℕ := total_cost + (total_cost * desired_profit_percentage / 100))
  (price_per_bag: ℕ := total_revenue / total_bags) :
  cost_seeds = 50 →
  cost_fertilizers_pesticides = 35 →
  cost_labor = 15 →
  total_bags = 10 →
  desired_profit_percentage = 10 →
  price_per_bag = 11 :=
by sorry

end corn_bag_price_l1060_106085


namespace combine_like_terms_1_simplify_expression_2_l1060_106091

-- Problem 1
theorem combine_like_terms_1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by 
  -- Proof goes here 
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by 
  -- Proof goes here 
  sorry

end combine_like_terms_1_simplify_expression_2_l1060_106091


namespace children_count_l1060_106024

theorem children_count (W C n : ℝ) (h1 : 4 * W = 1 / 7) (h2 : n * C = 1 / 14) (h3 : 5 * W + 10 * C = 1 / 4) : n = 10 :=
by
  sorry

end children_count_l1060_106024


namespace numberOfWaysToChoose4Cards_l1060_106004

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l1060_106004


namespace find_set_A_l1060_106031

-- Define the set A based on the condition that its elements satisfy a quadratic equation.
def A (a : ℝ) : Set ℝ := {x | x^2 + 2 * x + a = 0}

-- Assume 1 is an element of set A
axiom one_in_A (a : ℝ) (h : 1 ∈ A a) : a = -3

-- The final theorem to prove: Given 1 ∈ A a, A a should be {-3, 1}
theorem find_set_A (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} :=
by sorry

end find_set_A_l1060_106031


namespace parallel_lines_solution_l1060_106062

theorem parallel_lines_solution (a : ℝ) :
  (∃ (k1 k2 : ℝ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ 
  ∀ x y : ℝ, x + a^2 * y + 6 = 0 → k1*y = x ∧ 
             (a-2) * x + 3 * a * y + 2 * a = 0 → k2*y = x) 
  → (a = -1 ∨ a = 0) :=
by
  sorry

end parallel_lines_solution_l1060_106062


namespace olivia_remaining_usd_l1060_106086

def initial_usd : ℝ := 78
def initial_eur : ℝ := 50
def exchange_rate : ℝ := 1.20
def spent_usd_supermarket : ℝ := 15
def book_eur : ℝ := 10
def spent_usd_lunch : ℝ := 12

theorem olivia_remaining_usd :
  let total_usd := initial_usd + (initial_eur * exchange_rate)
  let remaining_after_supermarket := total_usd - spent_usd_supermarket
  let remaining_after_book := remaining_after_supermarket - (book_eur * exchange_rate)
  let final_remaining := remaining_after_book - spent_usd_lunch
  final_remaining = 99 :=
by
  sorry

end olivia_remaining_usd_l1060_106086


namespace pencils_per_associate_professor_l1060_106033

theorem pencils_per_associate_professor
    (A B P : ℕ) -- the number of associate professors, assistant professors, and pencils per associate professor respectively
    (h1 : A + B = 6) -- there are a total of 6 people
    (h2 : A * P + B = 7) -- total number of pencils is 7
    (h3 : A + 2 * B = 11) -- total number of charts is 11
    : P = 2 :=
by
  -- Placeholder for the proof
  sorry

end pencils_per_associate_professor_l1060_106033


namespace percent_sum_l1060_106067

theorem percent_sum (A B C : ℝ)
  (hA : 0.45 * A = 270)
  (hB : 0.35 * B = 210)
  (hC : 0.25 * C = 150) :
  0.75 * A + 0.65 * B + 0.45 * C = 1110 := by
  sorry

end percent_sum_l1060_106067


namespace average_marks_of_all_candidates_l1060_106096

def n : ℕ := 120
def p : ℕ := 100
def f : ℕ := n - p
def A_p : ℕ := 39
def A_f : ℕ := 15
def total_marks : ℕ := p * A_p + f * A_f
def average_marks : ℚ := total_marks / n

theorem average_marks_of_all_candidates :
  average_marks = 35 := 
sorry

end average_marks_of_all_candidates_l1060_106096


namespace arithmetic_expression_eval_l1060_106041

theorem arithmetic_expression_eval :
  -1 ^ 4 + (4 - ((3 / 8 + 1 / 6 - 3 / 4) * 24)) / 5 = 0.8 := by
  sorry

end arithmetic_expression_eval_l1060_106041


namespace find_a6_plus_a7_plus_a8_l1060_106002

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end find_a6_plus_a7_plus_a8_l1060_106002


namespace calculate_F_2_f_3_l1060_106027

def f (a : ℕ) : ℕ := a ^ 2 - 3 * a + 2

def F (a b : ℕ) : ℕ := b ^ 2 + a + 1

theorem calculate_F_2_f_3 : F 2 (f 3) = 7 :=
by
  show F 2 (f 3) = 7
  sorry

end calculate_F_2_f_3_l1060_106027


namespace simplify_expression_l1060_106001

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end simplify_expression_l1060_106001


namespace largest_eight_digit_with_all_even_digits_l1060_106054

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l1060_106054


namespace math_evening_problem_l1060_106053

theorem math_evening_problem
  (S : ℕ)
  (r : ℕ)
  (fifth_graders_per_row : ℕ := 3)
  (sixth_graders_per_row : ℕ := r - fifth_graders_per_row)
  (total_number_of_students : ℕ := r * r) :
  70 < total_number_of_students ∧ total_number_of_students < 90 → 
  r = 9 ∧ 
  6 * r = 54 ∧
  3 * r = 27 :=
sorry

end math_evening_problem_l1060_106053


namespace polynomial_remainder_distinct_l1060_106023

open Nat

theorem polynomial_remainder_distinct (a b c p : ℕ) (hp : Nat.Prime p) (hp_ge5 : p ≥ 5)
  (ha : Nat.gcd a p = 1) (hb : b^2 ≡ 3 * a * c [MOD p]) (hp_mod3 : p ≡ 2 [MOD 3]) :
  ∀ m1 m2 : ℕ, m1 < p ∧ m2 < p → m1 ≠ m2 → (a * m1^3 + b * m1^2 + c * m1) % p ≠ (a * m2^3 + b * m2^2 + c * m2) % p := 
by
  sorry

end polynomial_remainder_distinct_l1060_106023


namespace find_value_l1060_106079

theorem find_value : (1 / 4 * (5 * 9 * 4) - 7) = 38 := 
by
  sorry

end find_value_l1060_106079


namespace enthalpy_of_formation_C6H6_l1060_106098

theorem enthalpy_of_formation_C6H6 :
  ∀ (enthalpy_C2H2 : ℝ) (enthalpy_C6H6 : ℝ)
  (enthalpy_C6H6_C6H6 : ℝ) (Hess_law : Prop),
  (enthalpy_C2H2 = 226.7) →
  (enthalpy_C6H6 = 631.1) →
  (enthalpy_C6H6_C6H6 = -33.9) →
  Hess_law →
  -- Using the given conditions to accumulate the enthalpy change for the formation of C6H6.
  ∃ Q_formation : ℝ, Q_formation = -82.9 := by
  sorry

end enthalpy_of_formation_C6H6_l1060_106098


namespace number_of_cuboids_painted_l1060_106090

-- Define the problem conditions
def painted_faces (total_faces : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
  total_faces / faces_per_cuboid

-- Define the theorem to prove
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) :
  total_faces = 48 → faces_per_cuboid = 6 → painted_faces total_faces faces_per_cuboid = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end number_of_cuboids_painted_l1060_106090


namespace intersection_complement_M_N_l1060_106016

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }
def complement_M : Set ℝ := { x | x ≤ 1 }

theorem intersection_complement_M_N :
  (complement_M ∩ N) = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_complement_M_N_l1060_106016


namespace find_values_l1060_106073

theorem find_values (a b c : ℤ)
  (h1 : ∀ x, x^2 + 9 * x + 14 = (x + a) * (x + b))
  (h2 : ∀ x, x^2 + 4 * x - 21 = (x + b) * (x - c)) :
  a + b + c = 12 :=
sorry

end find_values_l1060_106073


namespace fraction_checked_by_worker_y_l1060_106066

theorem fraction_checked_by_worker_y
  (f_X f_Y : ℝ)
  (h1 : f_X + f_Y = 1)
  (h2 : 0.005 * f_X + 0.008 * f_Y = 0.0074) :
  f_Y = 0.8 :=
by
  sorry

end fraction_checked_by_worker_y_l1060_106066


namespace sum_of_x_y_is_13_l1060_106068

theorem sum_of_x_y_is_13 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h : x^4 + y^4 = 4721) : x + y = 13 :=
sorry

end sum_of_x_y_is_13_l1060_106068


namespace intersection_A_B_l1060_106055

def A : Set ℝ := {1, 3, 9, 27}
def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = Real.log x / Real.log 3}
theorem intersection_A_B : A ∩ B = {1, 3} := 
by
  sorry

end intersection_A_B_l1060_106055


namespace comprehensiveInvestigation_is_Census_l1060_106064

def comprehensiveInvestigation (s: String) : Prop :=
  s = "Census"

theorem comprehensiveInvestigation_is_Census :
  comprehensiveInvestigation "Census" :=
by
  sorry

end comprehensiveInvestigation_is_Census_l1060_106064


namespace conformal_2z_conformal_z_minus_2_squared_l1060_106081

-- For the function w = 2z
theorem conformal_2z :
  ∀ z : ℂ, true :=
by
  intro z
  sorry

-- For the function w = (z-2)^2
theorem conformal_z_minus_2_squared :
  ∀ z : ℂ, z ≠ 2 → true :=
by
  intro z h
  sorry

end conformal_2z_conformal_z_minus_2_squared_l1060_106081


namespace smallest_pos_int_mod_congruence_l1060_106074

theorem smallest_pos_int_mod_congruence : ∃ n : ℕ, 0 < n ∧ n ≡ 2 [MOD 31] ∧ 5 * n ≡ 409 [MOD 31] :=
by
  sorry

end smallest_pos_int_mod_congruence_l1060_106074


namespace max_value_inequality_l1060_106077

theorem max_value_inequality (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) :
  (kx + y)^2 / (x^2 + y^2) ≤ 2 :=
sorry

end max_value_inequality_l1060_106077


namespace problem_solution_l1060_106039

theorem problem_solution :
  (315^2 - 291^2) / 24 = 606 :=
by
  sorry

end problem_solution_l1060_106039


namespace find_n_from_lcms_l1060_106047

theorem find_n_from_lcms (n : ℕ) (h_pos : n > 0) (h_lcm1 : Nat.lcm 40 n = 200) (h_lcm2 : Nat.lcm n 45 = 180) : n = 100 := 
by
  sorry

end find_n_from_lcms_l1060_106047


namespace perimeter_of_smaller_rectangle_l1060_106080

theorem perimeter_of_smaller_rectangle (s t u : ℝ) (h1 : 4 * s = 160) (h2 : t = s / 2) (h3 : u = t / 3) : 
    2 * (t + u) = 400 / 3 := by
  sorry

end perimeter_of_smaller_rectangle_l1060_106080


namespace johns_raise_percentage_increase_l1060_106025

theorem johns_raise_percentage_increase (original_amount new_amount : ℝ) (h_original : original_amount = 60) (h_new : new_amount = 70) :
  ((new_amount - original_amount) / original_amount) * 100 = 16.67 := 
  sorry

end johns_raise_percentage_increase_l1060_106025


namespace part_a_no_solutions_part_a_infinite_solutions_l1060_106015

theorem part_a_no_solutions (a : ℝ) (x y : ℝ) : 
    a = -1 → ¬(∃ x y : ℝ, a * x + y = a^2 ∧ x + a * y = 1) :=
sorry

theorem part_a_infinite_solutions (a : ℝ) (x y : ℝ) : 
    a = 1 → ∃ x : ℝ, ∃ y : ℝ, a * x + y = a^2 ∧ x + a * y = 1 :=
sorry

end part_a_no_solutions_part_a_infinite_solutions_l1060_106015
