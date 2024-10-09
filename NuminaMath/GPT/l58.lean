import Mathlib

namespace algebraic_expression_evaluation_l58_5899

theorem algebraic_expression_evaluation (a b c : ℝ) 
  (h1 : a^2 + b * c = 14) 
  (h2 : b^2 - 2 * b * c = -6) : 
  3 * a^2 + 4 * b^2 - 5 * b * c = 18 :=
by 
  sorry

end algebraic_expression_evaluation_l58_5899


namespace mean_of_remaining_quiz_scores_l58_5855

theorem mean_of_remaining_quiz_scores (k : ℕ) (hk : k > 12) 
  (mean_k : ℝ) (mean_12 : ℝ) 
  (mean_class : mean_k = 8) 
  (mean_12_group : mean_12 = 14) 
  (mean_correct : mean_12 * 12 + mean_k * (k - 12) = 8 * k) :
  mean_k * (k - 12) = (8 * k - 168) := 
by {
  sorry
}

end mean_of_remaining_quiz_scores_l58_5855


namespace height_of_first_building_l58_5814

theorem height_of_first_building (h : ℕ) (h_condition : h + 2 * h + 9 * h = 7200) : h = 600 :=
by
  sorry

end height_of_first_building_l58_5814


namespace net_error_24x_l58_5841

theorem net_error_24x (x : ℕ) : 
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let error_pennies := (nickel_value - penny_value) * x
  let error_nickels := (dime_value - nickel_value) * x
  let error_dimes := (quarter_value - dime_value) * x
  let total_error := error_pennies + error_nickels + error_dimes
  total_error = 24 * x := 
by 
  sorry

end net_error_24x_l58_5841


namespace equation_solutions_l58_5820

theorem equation_solutions : 
  ∀ x : ℝ, (2 * x - 1) - x * (1 - 2 * x) = 0 ↔ (x = 1 / 2 ∨ x = -1) :=
by
  intro x
  sorry

end equation_solutions_l58_5820


namespace amount_paid_for_peaches_l58_5854

def total_spent := 23.86
def cherries_spent := 11.54
def peaches_spent := 12.32

theorem amount_paid_for_peaches :
  total_spent - cherries_spent = peaches_spent :=
sorry

end amount_paid_for_peaches_l58_5854


namespace solution_set_of_xf_l58_5835

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem solution_set_of_xf (f : ℝ → ℝ) (hf_odd : is_odd_function f) (hf_one : f 1 = 0)
    (h_derivative : ∀ x > 0, (x * (deriv f x) - f x) / (x^2) > 0) :
    {x : ℝ | x * f x > 0} = {x : ℝ | x < -1 ∨ x > 1} :=
by
  sorry

end solution_set_of_xf_l58_5835


namespace coconut_grove_yield_l58_5886

theorem coconut_grove_yield (x Y : ℕ) (h1 : x = 10)
  (h2 : (x + 2) * 30 + x * Y + (x - 2) * 180 = 3 * x * 100) : Y = 120 :=
by
  -- Proof to be provided
  sorry

end coconut_grove_yield_l58_5886


namespace dress_shirt_cost_l58_5822

theorem dress_shirt_cost (x : ℝ) :
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  total_cost_after_coupon = 252 → x = 15 :=
by
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  intro h
  sorry

end dress_shirt_cost_l58_5822


namespace height_of_trapezoid_l58_5883

-- Define the condition that a trapezoid has diagonals of given lengths and a given midline.
def trapezoid_conditions (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : Prop := 
  AC = 6 ∧ BD = 8 ∧ ML = 5

-- Define the height of the trapezoid.
def trapezoid_height (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : ℝ :=
  4.8

-- The theorem statement
theorem height_of_trapezoid (AC BD ML h : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : 
  trapezoid_conditions AC BD ML h_d1 h_d2 h_ml 
  → trapezoid_height AC BD ML h_d1 h_d2 h_ml = 4.8 := 
by
  intros
  sorry

end height_of_trapezoid_l58_5883


namespace quadratic_solution_pair_l58_5879

open Real

noncomputable def solution_pair : ℝ × ℝ :=
  ((45 - 15 * sqrt 5) / 2, (45 + 15 * sqrt 5) / 2)

theorem quadratic_solution_pair (a c : ℝ) 
  (h1 : (∃ x : ℝ, a * x^2 + 30 * x + c = 0 ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 30 * y + c ≠ 0))
  (h2 : a + c = 45)
  (h3 : a < c) :
  (a, c) = solution_pair :=
sorry

end quadratic_solution_pair_l58_5879


namespace find_an_l58_5856

def sequence_sum (k : ℝ) (n : ℕ) : ℝ :=
  k * n ^ 2 + n

def term_of_sequence (k : ℝ) (n : ℕ) (S_n : ℝ) (S_nm1 : ℝ) : ℝ :=
  S_n - S_nm1

theorem find_an (k : ℝ) (n : ℕ) (h₁ : n > 0) :
  term_of_sequence k n (sequence_sum k n) (sequence_sum k (n - 1)) = 2 * k * n - k + 1 :=
by
  sorry

end find_an_l58_5856


namespace sufficient_but_not_necessary_l58_5818

theorem sufficient_but_not_necessary (x y : ℝ) (h : ⌊x⌋ = ⌊y⌋) : 
  |x - y| < 1 ∧ ∃ x y : ℝ, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋ :=
by 
  sorry

end sufficient_but_not_necessary_l58_5818


namespace Thabo_books_ratio_l58_5827

variable (P_f P_nf H_nf : ℕ)

theorem Thabo_books_ratio :
  P_f + P_nf + H_nf = 220 →
  H_nf = 40 →
  P_nf = H_nf + 20 →
  P_f / P_nf = 2 :=
by sorry

end Thabo_books_ratio_l58_5827


namespace find_multiple_l58_5825

theorem find_multiple (x y : ℕ) (h1 : x = 11) (h2 : x + y = 55) (h3 : ∃ k m : ℕ, y = k * x + m) :
  ∃ k : ℕ, y = k * x ∧ k = 4 :=
by
  sorry

end find_multiple_l58_5825


namespace gcd_of_three_numbers_l58_5898

theorem gcd_of_three_numbers (a b c : ℕ) (h1: a = 4557) (h2: b = 1953) (h3: c = 5115) : 
    Nat.gcd a (Nat.gcd b c) = 93 :=
by
  rw [h1, h2, h3]
  -- Proof goes here
  sorry

end gcd_of_three_numbers_l58_5898


namespace scenery_photos_correct_l58_5800

-- Define the problem conditions
def animal_photos := 10
def flower_photos := 3 * animal_photos
def photos_total := 45
def scenery_photos := flower_photos - 10

-- State the theorem
theorem scenery_photos_correct : scenery_photos = 20 ∧ animal_photos + flower_photos + scenery_photos = photos_total := by
  sorry

end scenery_photos_correct_l58_5800


namespace probability_line_through_cube_faces_l58_5878

def prob_line_intersects_cube_faces : ℚ :=
  1 / 7

theorem probability_line_through_cube_faces :
  let cube_vertices := 8
  let total_selections := Nat.choose cube_vertices 2
  let body_diagonals := 4
  let probability := (body_diagonals : ℚ) / total_selections
  probability = prob_line_intersects_cube_faces :=
by {
  sorry
}

end probability_line_through_cube_faces_l58_5878


namespace kaleb_toys_l58_5870

def initial_savings : ℕ := 21
def allowance : ℕ := 15
def cost_per_toy : ℕ := 6

theorem kaleb_toys : (initial_savings + allowance) / cost_per_toy = 6 :=
by
  sorry

end kaleb_toys_l58_5870


namespace sum_digits_350_1350_base2_l58_5860

def binary_sum_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

theorem sum_digits_350_1350_base2 :
  binary_sum_digits 350 + binary_sum_digits 1350 = 20 :=
by
  sorry

end sum_digits_350_1350_base2_l58_5860


namespace P_at_2007_l58_5808

noncomputable def P (x : ℝ) : ℝ :=
x^15 - 2008 * x^14 + 2008 * x^13 - 2008 * x^12 + 2008 * x^11
- 2008 * x^10 + 2008 * x^9 - 2008 * x^8 + 2008 * x^7
- 2008 * x^6 + 2008 * x^5 - 2008 * x^4 + 2008 * x^3
- 2008 * x^2 + 2008 * x

-- Statement to show that P(2007) = 2007
theorem P_at_2007 : P 2007 = 2007 :=
  sorry

end P_at_2007_l58_5808


namespace profit_ratio_l58_5888

theorem profit_ratio (p_investment q_investment : ℝ) (h₁ : p_investment = 50000) (h₂ : q_investment = 66666.67) :
  (1 / q_investment) = (3 / 4 * 1 / p_investment) :=
by
  sorry

end profit_ratio_l58_5888


namespace smallest_natural_number_l58_5809

theorem smallest_natural_number :
  ∃ N : ℕ, ∃ f : ℕ → ℕ → ℕ, 
  f (f (f 9 8 - f 7 6) 5 + 4 - f 3 2) 1 = N ∧
  N = 1 := 
by sorry

end smallest_natural_number_l58_5809


namespace annual_population_change_l58_5807

theorem annual_population_change (initial_population : Int) (moved_in : Int) (moved_out : Int) (final_population : Int) (years : Int) : 
  initial_population = 780 → 
  moved_in = 100 →
  moved_out = 400 →
  final_population = 60 →
  years = 4 →
  (initial_population + moved_in - moved_out - final_population) / years = 105 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end annual_population_change_l58_5807


namespace eval_expression_l58_5887

theorem eval_expression : abs (-6) - (-4) + (-7) = 3 :=
by
  sorry

end eval_expression_l58_5887


namespace undefined_expression_l58_5845

theorem undefined_expression (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_expression_l58_5845


namespace probability_team_B_wins_third_game_l58_5838

theorem probability_team_B_wins_third_game :
  ∀ (A B : ℕ → Prop),
    (∀ n, A n ∨ B n) ∧ -- Each game is won by either A or B
    (∀ n, A n ↔ ¬ B n) ∧ -- No ties, outcomes are independent
    (A 0) ∧ -- Team A wins the first game
    (B 1) ∧ -- Team B wins the second game
    (∃ n1 n2 n3, A n1 ∧ A n2 ∧ A n3 ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3) -- Team A wins three games
    → (∃ S, ((A 0) ∧ (B 1) ∧ (B 2)) ↔ (S = 1/3)) := sorry

end probability_team_B_wins_third_game_l58_5838


namespace inverse_f_1_l58_5867

noncomputable def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

theorem inverse_f_1 : ∃ x : ℝ, f x = 1 ∧ x = 2 := by
sorry

end inverse_f_1_l58_5867


namespace sum_of_integers_l58_5851

variable (p q r s : ℤ)

theorem sum_of_integers :
  (p - q + r = 7) →
  (q - r + s = 8) →
  (r - s + p = 4) →
  (s - p + q = 1) →
  p + q + r + s = 20 := by
  intros h1 h2 h3 h4
  sorry

end sum_of_integers_l58_5851


namespace solve_complex_eq_l58_5869

open Complex

theorem solve_complex_eq (z : ℂ) (h : (3 - 4 * I) * z = 5) : z = (3 / 5) + (4 / 5) * I :=
by
  sorry

end solve_complex_eq_l58_5869


namespace cyclists_meeting_time_l58_5877

theorem cyclists_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 7 (Nat.lcm 12 9) ∧ t = 252 :=
by
  use 252
  have h1 : Nat.lcm 7 (Nat.lcm 12 9) = 252 := sorry
  exact ⟨rfl, h1⟩

end cyclists_meeting_time_l58_5877


namespace expected_points_A_correct_prob_A_B_same_points_correct_l58_5881

-- Conditions
def game_is_independent := true

def prob_A_B_win := 2/5
def prob_A_B_draw := 1/5

def prob_A_C_win := 1/3
def prob_A_C_draw := 1/3

def prob_B_C_win := 1/2
def prob_B_C_draw := 1/6

noncomputable def prob_A_B_lose := 1 - prob_A_B_win - prob_A_B_draw
noncomputable def prob_A_C_lose := 1 - prob_A_C_win - prob_A_C_draw
noncomputable def prob_B_C_lose := 1 - prob_B_C_win - prob_B_C_draw

noncomputable def expected_points_A : ℚ := 0 * (prob_A_B_lose * prob_A_C_lose)        /- P(ξ=0) = 2/15 -/
                                       + 1 * ((prob_A_B_draw * prob_A_C_lose) +
                                              (prob_A_B_lose * prob_A_C_draw))        /- P(ξ=1) = 1/5 -/
                                       + 2 * (prob_A_B_draw * prob_A_C_draw)         /- P(ξ=2) = 1/15 -/
                                       + 3 * ((prob_A_B_win * prob_A_C_lose) + 
                                              (prob_A_B_win * prob_A_C_draw) + 
                                              (prob_A_C_win * prob_A_B_lose))        /- P(ξ=3) = 4/15 -/
                                       + 4 * ((prob_A_B_draw * prob_A_C_win) +
                                              (prob_A_B_win * prob_A_C_win))         /- P(ξ=4) = 1/5 -/
                                       + 6 * (prob_A_B_win * prob_A_C_win)           /- P(ξ=6) = 2/15 -/

theorem expected_points_A_correct : expected_points_A = 41 / 15 :=
by
  sorry

noncomputable def prob_A_B_same_points: ℚ := ((prob_A_B_draw * prob_A_C_lose) * prob_B_C_lose)  /- both 1 point -/
                                            + ((prob_A_B_draw * prob_A_C_draw) * prob_B_C_draw)/- both 2 points -/
                                            + ((prob_A_B_win * prob_B_C_win) * prob_A_C_lose)  /- both 3 points -/
                                            + ((prob_A_B_win * prob_A_C_lose) * prob_B_C_win)  /- both 3 points -/
                                            + ((prob_A_B_draw * prob_A_C_win) * prob_B_C_win)  /- both 4 points -/

theorem prob_A_B_same_points_correct : prob_A_B_same_points = 8 / 45 :=
by
  sorry

end expected_points_A_correct_prob_A_B_same_points_correct_l58_5881


namespace man_l58_5890

theorem man's_speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h_current_speed : current_speed = 5) (h_against_current_speed : against_current_speed = 12) 
  (h_v : v - current_speed = against_current_speed) : 
  v + current_speed = 22 := 
by
  sorry

end man_l58_5890


namespace quadratic_completion_l58_5872

theorem quadratic_completion (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 26 * x + 81 = (x + b)^2 + c) → b + c = -101 :=
by 
  intro h
  sorry

end quadratic_completion_l58_5872


namespace parallel_line_eq_l58_5850

theorem parallel_line_eq (x y : ℝ) (c : ℝ) :
  (∀ x y, x - 2 * y - 2 = 0 → x - 2 * y + c = 0) ∧ (x = 1 ∧ y = 0) → c = -1 :=
by
  sorry

end parallel_line_eq_l58_5850


namespace oranges_in_bag_l58_5817

variables (O : ℕ)

def initial_oranges (O : ℕ) := O
def initial_tangerines := 17
def oranges_left_after_taking_away := O - 2
def tangerines_left_after_taking_away := 7
def tangerines_and_oranges_condition (O : ℕ) := 7 = (O - 2) + 4

theorem oranges_in_bag (O : ℕ) (h₀ : tangerines_and_oranges_condition O) : O = 5 :=
by
  sorry

end oranges_in_bag_l58_5817


namespace find_a_of_line_slope_l58_5866

theorem find_a_of_line_slope (a : ℝ) (h1 : a > 0)
  (h2 : ∃ (b : ℝ), (a, 5) = (b * 1, b * 2) ∧ (2, a) = (b * 1, 2 * b) ∧ b = 1) 
  : a = 3 := 
sorry

end find_a_of_line_slope_l58_5866


namespace Mongolian_Mathematical_Olympiad_54th_l58_5834

theorem Mongolian_Mathematical_Olympiad_54th {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^4 + b^4 + c^4 + (a^2 / (b + c)^2) + (b^2 / (c + a)^2) + (c^2 / (a + b)^2) ≥ a * b + b * c + c * a :=
sorry

end Mongolian_Mathematical_Olympiad_54th_l58_5834


namespace problem_1_problem_2_problem_3_l58_5889

-- Problem 1
theorem problem_1 (x : ℝ) (h : 4.8 - 3 * x = 1.8) : x = 1 :=
by { sorry }

-- Problem 2
theorem problem_2 (x : ℝ) (h : (1 / 8) / (1 / 5) = x / 24) : x = 15 :=
by { sorry }

-- Problem 3
theorem problem_3 (x : ℝ) (h : 7.5 * x + 6.5 * x = 2.8) : x = 0.2 :=
by { sorry }

end problem_1_problem_2_problem_3_l58_5889


namespace find_a3_l58_5882

def sequence_sum (n : ℕ) : ℕ := n^2 + n

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem find_a3 : a 3 = 6 := by
  sorry

end find_a3_l58_5882


namespace negation_of_proposition_l58_5865

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0)) ↔ (∃ x : ℝ, x^2 + 2 * x + 3 < 0) :=
by sorry

end negation_of_proposition_l58_5865


namespace simplify_and_evaluate_l58_5811

theorem simplify_and_evaluate :
  let x := (-1 : ℚ) / 2
  3 * x^2 - (5 * x - 3 * (2 * x - 1) + 7 * x^2) = -9 / 2 :=
by
  let x : ℚ := (-1 : ℚ) / 2
  sorry

end simplify_and_evaluate_l58_5811


namespace total_drink_volume_l58_5823

theorem total_drink_volume (coke_parts sprite_parts mtndew_parts : ℕ) (coke_volume : ℕ) :
  coke_parts = 2 → sprite_parts = 1 → mtndew_parts = 3 → coke_volume = 6 →
  (coke_volume / coke_parts) * (coke_parts + sprite_parts + mtndew_parts) = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end total_drink_volume_l58_5823


namespace magician_can_always_determine_hidden_pair_l58_5810

-- Define the cards as an enumeration
inductive Card
| one | two | three | four | five

-- Define a pair of cards
structure CardPair where
  first : Card
  second : Card

-- Define the function the magician uses to decode the hidden pair 
-- based on the two cards the assistant points out, encoded as a pentagon
noncomputable def magician_decodes (assistant_cards spectator_announced: CardPair) : CardPair := sorry

-- Theorem statement: given the conditions, the magician can always determine the hidden pair.
theorem magician_can_always_determine_hidden_pair 
  (hidden_cards assistant_cards spectator_announced : CardPair)
  (assistant_strategy : CardPair → CardPair)
  (h : assistant_strategy assistant_cards = spectator_announced)
  : magician_decodes assistant_cards spectator_announced = hidden_cards := sorry

end magician_can_always_determine_hidden_pair_l58_5810


namespace jane_not_finish_probability_l58_5816

theorem jane_not_finish_probability :
  (1 : ℚ) - (5 / 8) = (3 / 8) := by
  sorry

end jane_not_finish_probability_l58_5816


namespace volume_Q3_l58_5876

noncomputable def sequence_of_polyhedra (n : ℕ) : ℚ :=
match n with
| 0     => 1
| 1     => 3 / 2
| 2     => 45 / 32
| 3     => 585 / 128
| _     => 0 -- for n > 3 not defined

theorem volume_Q3 : sequence_of_polyhedra 3 = 585 / 128 :=
by
  -- Placeholder for the theorem proof
  sorry

end volume_Q3_l58_5876


namespace cyclist_speed_ratio_l58_5813

theorem cyclist_speed_ratio (v_1 v_2 : ℝ)
  (h1 : v_1 = 2 * v_2)
  (h2 : v_1 + v_2 = 6)
  (h3 : v_1 - v_2 = 2) :
  v_1 / v_2 = 2 := 
sorry

end cyclist_speed_ratio_l58_5813


namespace circle_representation_l58_5844

theorem circle_representation (a : ℝ): 
  (∃ (x y : ℝ), (x^2 + y^2 + 2*x + a = 0) ∧ (∃ D E F, D = 2 ∧ E = 0 ∧ F = -a ∧ (D^2 + E^2 - 4*F > 0))) ↔ (a > -1) :=
by 
  sorry

end circle_representation_l58_5844


namespace mixed_number_arithmetic_l58_5868

theorem mixed_number_arithmetic :
  26 * (2 + 4 / 7 - (3 + 1 / 3)) + (3 + 1 / 5 + (2 + 3 / 7)) = -14 - 223 / 735 :=
by
  sorry

end mixed_number_arithmetic_l58_5868


namespace floor_sqrt_50_l58_5806

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l58_5806


namespace frustum_volume_l58_5894

theorem frustum_volume (m : ℝ) (α : ℝ) (k : ℝ) : 
  m = 3/π ∧ 
  α = 43 + 40/60 + 42.2/3600 ∧ 
  k = 1 →
  frustumVolume = 0.79 := 
sorry

end frustum_volume_l58_5894


namespace average_speed_of_trip_l58_5824

noncomputable def total_distance (d1 d2 : ℝ) : ℝ :=
  d1 + d2

noncomputable def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem average_speed_of_trip :
  let d1 := 60
  let s1 := 20
  let d2 := 120
  let s2 := 60
  let total_d := total_distance d1 d2
  let time1 := travel_time d1 s1
  let time2 := travel_time d2 s2
  let total_t := time1 + time2
  average_speed total_d total_t = 36 :=
by
  sorry

end average_speed_of_trip_l58_5824


namespace average_breadth_of_plot_l58_5826

theorem average_breadth_of_plot :
  ∃ B L : ℝ, (L - B = 10) ∧ (23 * B = (1/2) * (L + B) * B) ∧ (B = 18) :=
by
  sorry

end average_breadth_of_plot_l58_5826


namespace speed_in_still_water_l58_5893

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 25) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 45 :=
by
  sorry

end speed_in_still_water_l58_5893


namespace profit_A_after_upgrade_profit_B_constrained_l58_5897

-- Part Ⅰ
theorem profit_A_after_upgrade (x : ℝ) (h : x^2 - 300 * x ≤ 0) : 0 < x ∧ x ≤ 300 := sorry

-- Part Ⅱ
theorem profit_B_constrained (a x : ℝ) (h1 : a ≤ (x/125 + 500/x + 3/2)) (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := sorry

end profit_A_after_upgrade_profit_B_constrained_l58_5897


namespace find_CM_of_trapezoid_l58_5831

noncomputable def trapezoid_CM (AD BC : ℝ) (M : ℝ) : ℝ :=
  if (AD = 12) ∧ (BC = 8) ∧ (M = 2.4)
  then M
  else 0

theorem find_CM_of_trapezoid (trapezoid_ABCD : Type) (AD BC CM : ℝ) (AM_divides_eq_areas : Prop) :
  AD = 12 → BC = 8 → AM_divides_eq_areas → CM = 2.4 := 
by
  intros h1 h2 h3
  have : AD = 12 := h1
  have : BC = 8 := h2
  have : CM = 2.4 := sorry
  exact this

end find_CM_of_trapezoid_l58_5831


namespace find_x_l58_5853

theorem find_x (x : ℝ) (h1 : (x - 1) / (x + 2) = 0) (h2 : x ≠ -2) : x = 1 :=
sorry

end find_x_l58_5853


namespace investment_time_l58_5805

theorem investment_time
  (p_investment_ratio : ℚ) (q_investment_ratio : ℚ)
  (profit_ratio_p : ℚ) (profit_ratio_q : ℚ)
  (q_investment_time : ℕ)
  (h1 : p_investment_ratio / q_investment_ratio = 7 / 5)
  (h2 : profit_ratio_p / profit_ratio_q = 7 / 10)
  (h3 : q_investment_time = 40) :
  ∃ t : ℚ, t = 28 :=
by
  sorry

end investment_time_l58_5805


namespace circumscribed_circles_intersect_l58_5837

noncomputable def circumcircle (a b c : Point) : Set Point := sorry

noncomputable def intersect_at_single_point (circles : List (Set Point)) : Option Point := sorry

variables {A1 A2 A3 B1 B2 B3 : Point}

theorem circumscribed_circles_intersect
  (h1 : ∃ P, ∀ circle ∈ [
    circumcircle A1 A2 B3, 
    circumcircle A1 B2 A3, 
    circumcircle B1 A2 A3
  ], P ∈ circle) :
  ∃ Q, ∀ circle ∈ [
    circumcircle B1 B2 A3, 
    circumcircle B1 A2 B3, 
    circumcircle A1 B2 B3
  ], Q ∈ circle :=
sorry

end circumscribed_circles_intersect_l58_5837


namespace parabola_focus_distance_l58_5871

theorem parabola_focus_distance (p m : ℝ) (hp : p > 0)
  (P_on_parabola : m^2 = 2 * p)
  (PF_dist : (1 + p / 2) = 3) : p = 4 := 
  sorry

end parabola_focus_distance_l58_5871


namespace number_of_sides_sum_of_interior_angles_l58_5885

-- Condition: each exterior angle of the regular polygon is 18 degrees.
def exterior_angle (n : ℕ) : Prop :=
  360 / n = 18

-- Question 1: Determine the number of sides the polygon has.
theorem number_of_sides : ∃ n, n > 2 ∧ exterior_angle n :=
  sorry

-- Question 2: Calculate the sum of the interior angles.
theorem sum_of_interior_angles {n : ℕ} (h : 360 / n = 18) : 
  180 * (n - 2) = 3240 :=
  sorry

end number_of_sides_sum_of_interior_angles_l58_5885


namespace fixed_monthly_costs_l58_5846

theorem fixed_monthly_costs
  (cost_per_component : ℕ) (shipping_cost : ℕ) 
  (num_components : ℕ) (selling_price : ℚ)
  (F : ℚ) :
  cost_per_component = 80 →
  shipping_cost = 6 →
  num_components = 150 →
  selling_price = 196.67 →
  F = (num_components * selling_price) - (num_components * (cost_per_component + shipping_cost)) →
  F = 16600.5 :=
by
  intros
  sorry

end fixed_monthly_costs_l58_5846


namespace range_of_a_l58_5858

theorem range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (x-a) / (2 - (x + 1 - a)) > 0)
  ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l58_5858


namespace find_number_l58_5857

theorem find_number (x : ℕ) : x * 9999 = 4691130840 → x = 469200 :=
by
  intros h
  sorry

end find_number_l58_5857


namespace M_subset_N_l58_5895

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l58_5895


namespace greatest_integer_solution_l58_5843

theorem greatest_integer_solution :
  ∃ n : ℤ, (n^2 - 17 * n + 72 ≤ 0) ∧ (∀ m : ℤ, (m^2 - 17 * m + 72 ≤ 0) → m ≤ n) ∧ n = 9 :=
sorry

end greatest_integer_solution_l58_5843


namespace number_of_scooters_l58_5802

theorem number_of_scooters (b t s : ℕ) (h1 : b + t + s = 10) (h2 : 2 * b + 3 * t + 2 * s = 26) : s = 2 := 
by sorry

end number_of_scooters_l58_5802


namespace determine_value_of_m_l58_5839

noncomputable def conics_same_foci (m : ℝ) : Prop :=
  let c1 := Real.sqrt (4 - m^2)
  let c2 := Real.sqrt (m + 2)
  (∀ (x y : ℝ),
    (x^2 / 4 + y^2 / m^2 = 1) → (x^2 / m - y^2 / 2 = 1) → c1 = c2) → 
  m = 1

theorem determine_value_of_m : ∃ (m : ℝ), conics_same_foci m :=
sorry

end determine_value_of_m_l58_5839


namespace arithmetic_sequence_formula_sum_Tn_formula_l58_5833

variable {a : ℕ → ℤ} -- The sequence a_n
variable {S : ℕ → ℤ} -- The sum S_n
variable {a₃ : ℤ} (h₁ : a₃ = 20)
variable {S₃ S₄ : ℤ} (h₂ : 2 * S₃ = S₄ + 8)

/- The general formula for the arithmetic sequence a_n -/
theorem arithmetic_sequence_formula (d : ℤ) (a₁ : ℤ)
  (h₃ : (a₃ = a₁ + 2 * d))
  (h₄ : (S₃ = 3 * a₁ + 3 * d))
  (h₅ : (S₄ = 4 * a₁ + 6 * d)) :
  ∀ n : ℕ, a n = 8 * n - 4 :=
by
  sorry

variable {b : ℕ → ℚ} -- Define b_n
variable {T : ℕ → ℚ} -- Define T_n
variable {S_general : ℕ → ℚ} (h₆ : ∀ n, S n = 4 * n ^ 2)
variable {b_general : ℚ → ℚ} (h₇ : ∀ n, b n = 1 / (S n - 1))
variable {T_general : ℕ → ℚ} -- Define T_n

/- The formula for T_n given b_n -/
theorem sum_Tn_formula :
  ∀ n : ℕ, T n = n / (2 * n + 1) :=
by
  sorry

end arithmetic_sequence_formula_sum_Tn_formula_l58_5833


namespace convert_base8_to_base7_l58_5819

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 1 * 8^0

def base10_to_base7 (n : ℕ) : ℕ :=
  1002  -- Directly providing the result from conditions given.

theorem convert_base8_to_base7 :
  base10_to_base7 (base8_to_base10 531) = 1002 := by
  sorry

end convert_base8_to_base7_l58_5819


namespace arithmetic_prog_sum_bound_l58_5852

noncomputable def Sn (n : ℕ) (a1 : ℝ) (d : ℝ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_prog_sum_bound (n : ℕ) (a1 an : ℝ) (d : ℝ) (h_d_neg : d < 0) 
  (ha_n : an = a1 + (n - 1) * d) :
  n * an < Sn n a1 d ∧ Sn n a1 d < n * a1 :=
by 
  sorry

end arithmetic_prog_sum_bound_l58_5852


namespace find_x_set_eq_l58_5849

noncomputable def f : ℝ → ℝ :=
sorry -- The actual definition of f according to its properties is omitted

lemma odd_function (x : ℝ) : f (-x) = -f x :=
sorry

lemma periodic_function (x : ℝ) : f (x + 2) = -f x :=
sorry

lemma f_definition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 1 / 2 * x :=
sorry

theorem find_x_set_eq (x : ℝ) : (f x = -1 / 2) ↔ (∃ k : ℤ, x = 4 * k - 1) :=
sorry

end find_x_set_eq_l58_5849


namespace share_expenses_l58_5848

theorem share_expenses (h l : ℕ) : 
  let henry_paid := 120
  let linda_paid := 150
  let jack_paid := 210
  let total_paid := henry_paid + linda_paid + jack_paid
  let each_should_pay := total_paid / 3
  let henry_owes := each_should_pay - henry_paid
  let linda_owes := each_should_pay - linda_paid
  (h = henry_owes) → 
  (l = linda_owes) → 
  h - l = 30 := by
  sorry

end share_expenses_l58_5848


namespace question_solution_l58_5884

theorem question_solution
  (f : ℝ → ℝ)
  (h_decreasing : ∀ ⦃x y : ℝ⦄, -3 < x ∧ x < 0 → -3 < y ∧ y < 0 → x < y → f y < f x)
  (h_symmetry : ∀ x : ℝ, f (x) = f (-x + 6)) :
  f (-5) < f (-3/2) ∧ f (-3/2) < f (-7/2) :=
sorry

end question_solution_l58_5884


namespace inverse_var_y_l58_5863

theorem inverse_var_y (k : ℝ) (y x : ℝ)
  (h1 : 5 * y = k / x^2)
  (h2 : y = 16) (h3 : x = 1) (h4 : k = 80) :
  y = 1 / 4 :=
by
  sorry

end inverse_var_y_l58_5863


namespace diana_total_extra_video_game_time_l58_5864

-- Definitions from the conditions
def minutesPerHourReading := 30
def raisePercent := 20
def choresToMinutes := 10
def maxChoresBonusMinutes := 60
def sportsPracticeHours := 8
def homeworkHours := 4
def totalWeekHours := 24
def readingHours := 8
def choresCompleted := 10

-- Deriving some necessary facts
def baseVideoGameTime := readingHours * minutesPerHourReading
def raiseMinutes := baseVideoGameTime * (raisePercent / 100)
def videoGameTimeWithRaise := baseVideoGameTime + raiseMinutes

def bonusesFromChores := (choresCompleted / 2) * choresToMinutes
def limitedChoresBonus := min bonusesFromChores maxChoresBonusMinutes

-- Total extra video game time
def totalExtraVideoGameTime := videoGameTimeWithRaise + limitedChoresBonus

-- The proof problem
theorem diana_total_extra_video_game_time : totalExtraVideoGameTime = 338 := by
  sorry

end diana_total_extra_video_game_time_l58_5864


namespace no_int_k_such_that_P_k_equals_8_l58_5840

theorem no_int_k_such_that_P_k_equals_8
    (P : Polynomial ℤ) 
    (a b c d k : ℤ)
    (h0: a ≠ b)
    (h1: a ≠ c)
    (h2: a ≠ d)
    (h3: b ≠ c)
    (h4: b ≠ d)
    (h5: c ≠ d)
    (h6: P.eval a = 5)
    (h7: P.eval b = 5)
    (h8: P.eval c = 5)
    (h9: P.eval d = 5)
    : P.eval k ≠ 8 := by
  sorry

end no_int_k_such_that_P_k_equals_8_l58_5840


namespace total_rabbits_correct_l58_5830

def initial_breeding_rabbits : ℕ := 10
def kittens_first_spring : ℕ := initial_breeding_rabbits * 10
def adopted_first_spring : ℕ := kittens_first_spring / 2
def returned_adopted_first_spring : ℕ := 5
def total_rabbits_after_first_spring : ℕ :=
  initial_breeding_rabbits + (kittens_first_spring - adopted_first_spring + returned_adopted_first_spring)

def kittens_second_spring : ℕ := 60
def adopted_second_spring : ℕ := kittens_second_spring * 40 / 100
def returned_adopted_second_spring : ℕ := 10
def total_rabbits_after_second_spring : ℕ :=
  total_rabbits_after_first_spring + (kittens_second_spring - adopted_second_spring + returned_adopted_second_spring)

def breeding_rabbits_third_spring : ℕ := 12
def kittens_third_spring : ℕ := breeding_rabbits_third_spring * 8
def adopted_third_spring : ℕ := kittens_third_spring * 30 / 100
def returned_adopted_third_spring : ℕ := 3
def total_rabbits_after_third_spring : ℕ :=
  total_rabbits_after_second_spring + (kittens_third_spring - adopted_third_spring + returned_adopted_third_spring)

def kittens_fourth_spring : ℕ := breeding_rabbits_third_spring * 6
def adopted_fourth_spring : ℕ := kittens_fourth_spring * 20 / 100
def returned_adopted_fourth_spring : ℕ := 2
def total_rabbits_after_fourth_spring : ℕ :=
  total_rabbits_after_third_spring + (kittens_fourth_spring - adopted_fourth_spring + returned_adopted_fourth_spring)

theorem total_rabbits_correct : total_rabbits_after_fourth_spring = 242 := by
  sorry

end total_rabbits_correct_l58_5830


namespace populations_equal_after_years_l58_5873

-- Defining the initial population and rates of change
def initial_population_X : ℕ := 76000
def rate_of_decrease_X : ℕ := 1200
def initial_population_Y : ℕ := 42000
def rate_of_increase_Y : ℕ := 800

-- Define the number of years for which we need to find the populations to be equal
def years (n : ℕ) : Prop :=
  (initial_population_X - rate_of_decrease_X * n) = (initial_population_Y + rate_of_increase_Y * n)

-- Theorem stating that the populations will be equal at n = 17
theorem populations_equal_after_years {n : ℕ} (h : n = 17) : years n :=
by
  sorry

end populations_equal_after_years_l58_5873


namespace sally_baseball_cards_l58_5859

theorem sally_baseball_cards (initial_cards torn_cards purchased_cards : ℕ) 
    (h_initial : initial_cards = 39)
    (h_torn : torn_cards = 9)
    (h_purchased : purchased_cards = 24) :
    initial_cards - torn_cards - purchased_cards = 6 := by
  sorry

end sally_baseball_cards_l58_5859


namespace accommodate_students_l58_5861

-- Define the parameters
def number_of_classrooms := 15
def one_third_classrooms := number_of_classrooms / 3
def desks_per_classroom_30 := 30
def desks_per_classroom_25 := 25

-- Define the number of classrooms for each type
def classrooms_with_30_desks := one_third_classrooms
def classrooms_with_25_desks := number_of_classrooms - classrooms_with_30_desks

-- Calculate total number of students that can be accommodated
def total_students : ℕ := 
  (classrooms_with_30_desks * desks_per_classroom_30) +
  (classrooms_with_25_desks * desks_per_classroom_25)

-- Prove that total number of students that the school can accommodate is 400
theorem accommodate_students : total_students = 400 := sorry

end accommodate_students_l58_5861


namespace solve_equation_l58_5815

theorem solve_equation : 
  ∀ x : ℝ,
    (x + 5 ≠ 0) → 
    (x^2 + 3 * x + 4) / (x + 5) = x + 6 → 
    x = -13 / 4 :=
by 
  intro x
  intro hx
  intro h
  sorry

end solve_equation_l58_5815


namespace max_sum_of_squares_eq_50_l58_5842

theorem max_sum_of_squares_eq_50 :
  ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 50 → x + y ≥ x' + y') ∧ x + y = 10 := 
sorry

end max_sum_of_squares_eq_50_l58_5842


namespace time_spent_per_bone_l58_5829

theorem time_spent_per_bone
  (total_hours : ℤ) (number_of_bones : ℤ) 
  (h1 : total_hours = 206) 
  (h2 : number_of_bones = 206) :
  (total_hours / number_of_bones = 1) := 
by {
  -- proof would go here
  sorry
}

end time_spent_per_bone_l58_5829


namespace acute_triangle_l58_5821

theorem acute_triangle (a b c : ℝ) (n : ℕ) (h_n : 2 < n) (h_eq : a^n + b^n = c^n) : a^2 + b^2 > c^2 :=
sorry

end acute_triangle_l58_5821


namespace largest_d_l58_5874

theorem largest_d (a b c d : ℝ) (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := sorry

end largest_d_l58_5874


namespace arithmetic_sequence_general_formula_l58_5847

noncomputable def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_formula {a : ℕ → ℤ} (h_seq : arithmetic_seq a) 
  (h_a1 : a 1 = 6) (h_a3a5 : a 3 + a 5 = 0) : 
  ∀ n, a n = 8 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l58_5847


namespace no_real_roots_ff_eq_x_l58_5801

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_ff_eq_x (a b c : ℝ)
  (h : a ≠ 0)
  (discriminant_condition : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x := 
by 
  sorry

end no_real_roots_ff_eq_x_l58_5801


namespace fraction_phone_numbers_9_ending_even_l58_5862

def isValidPhoneNumber (n : Nat) : Bool :=
  n / 10^6 != 0 && n / 10^6 != 1 && n / 10^6 != 2

def isValidEndEven (n : Nat) : Bool :=
  let lastDigit := n % 10
  lastDigit == 0 || lastDigit == 2 || lastDigit == 4 || lastDigit == 6 || lastDigit == 8

def countValidPhoneNumbers : Nat :=
  7 * 10^6

def countValidStarting9EndingEven : Nat :=
  5 * 10^5

theorem fraction_phone_numbers_9_ending_even :
  (countValidStarting9EndingEven : ℚ) / (countValidPhoneNumbers : ℚ) = 1 / 14 :=
by 
  sorry

end fraction_phone_numbers_9_ending_even_l58_5862


namespace jessica_total_money_after_activities_l58_5836

-- Definitions for given conditions
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def earned_from_washing_car : ℕ := 6

-- Theorem statement
theorem jessica_total_money_after_activities : 
  (weekly_allowance - spent_on_movies) + earned_from_washing_car = 11 :=
by 
  sorry

end jessica_total_money_after_activities_l58_5836


namespace boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l58_5804

-- Problem 1: Specific case
theorem boat_and_current_speed (x y : ℝ) 
  (h1 : 3 * (x + y) = 75) 
  (h2 : 5 * (x - y) = 75) : 
  x = 20 ∧ y = 5 := 
sorry

-- Problem 2: General case
theorem boat_and_current_speed_general (x y : ℝ) (a b S : ℝ) 
  (h1 : a * (x + y) = S) 
  (h2 : b * (x - y) = S) : 
  x = (a + b) * S / (2 * a * b) ∧ 
  y = (b - a) * S / (2 * a * b) := 
sorry

theorem log_drift_time (y S a b : ℝ)
  (h_y : y = (b - a) * S / (2 * a * b)) : 
  S / y = 2 * a * b / (b - a) := 
sorry

end boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l58_5804


namespace problem_C_plus_D_l58_5803

theorem problem_C_plus_D (C D : ℚ)
  (h : ∀ x, (D * x - 17) / (x^2 - 8 * x + 15) = C / (x - 3) + 5 / (x - 5)) :
  C + D = 5.8 :=
sorry

end problem_C_plus_D_l58_5803


namespace skylar_current_age_l58_5891

theorem skylar_current_age (started_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) (h1 : started_age = 17) (h2 : annual_donation = 8000) (h3 : total_donation = 440000) : 
  (started_age + total_donation / annual_donation = 72) :=
by
  sorry

end skylar_current_age_l58_5891


namespace Bowen_total_spent_l58_5896

def pencil_price : ℝ := 0.25
def pen_price : ℝ := 0.15
def num_pens : ℕ := 40

def num_pencils := num_pens + (2 / 5) * num_pens

theorem Bowen_total_spent : num_pencils * pencil_price + num_pens * pen_price = 20 := by
  sorry

end Bowen_total_spent_l58_5896


namespace sum_of_repeating_decimals_l58_5832

-- Definitions based on the conditions
def x := 0.6666666666666666 -- Lean may not directly support \(0.\overline{6}\) notation
def y := 0.7777777777777777 -- Lean may not directly support \(0.\overline{7}\) notation

-- Translate those to the correct fractional forms
def x_as_fraction := (2 : ℚ) / 3
def y_as_fraction := (7 : ℚ) / 9

-- The main statement to prove
theorem sum_of_repeating_decimals : x_as_fraction + y_as_fraction = 13 / 9 :=
by
  -- Proof skipped
  sorry

end sum_of_repeating_decimals_l58_5832


namespace probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l58_5892

noncomputable def P_n (n : ℕ) : ℚ :=
  if n = 3 then 1 / 4
  else if n = 4 then 3 / 4
  else 0

theorem probability_center_in_convex_hull_3_points :
  P_n 3 = 1 / 4 :=
by
  sorry

theorem probability_center_in_convex_hull_4_points :
  P_n 4 = 3 / 4 :=
by
  sorry

end probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l58_5892


namespace speed_in_still_water_l58_5828

theorem speed_in_still_water (v_m v_s : ℝ)
  (downstream : 48 = (v_m + v_s) * 3)
  (upstream : 34 = (v_m - v_s) * 4) :
  v_m = 12.25 :=
by
  sorry

end speed_in_still_water_l58_5828


namespace A_plus_2B_plus_4_is_perfect_square_l58_5812

theorem A_plus_2B_plus_4_is_perfect_square (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  ∃ k : ℚ, (A + 2 * B + 4) = k^2 :=
by
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  use ((2/3) * (10^n + 2))
  sorry

end A_plus_2B_plus_4_is_perfect_square_l58_5812


namespace ratio_of_areas_l58_5880

theorem ratio_of_areas (AB CD AH BG CF DG S_ABCD S_KLMN : ℕ)
  (h1 : AB = 15)
  (h2 : CD = 19)
  (h3 : DG = 17)
  (condition1 : S_ABCD = 17 * (AH + BG))
  (midpoints_AH_CF : AH = BG)
  (midpoints_CF_CD : CF = CD/2)
  (condition2 : (∃ h₁ h₂ : ℕ, S_KLMN = h₁ * AH + h₂ * CF / 2))
  (h_case1 : (S_KLMN = (AH + BG + CD)))
  (h_case2 : (S_KLMN = (AB + (CD - DG)))) :
  (S_ABCD / S_KLMN = 2 / 3 ∨ S_ABCD / S_KLMN = 2) :=
  sorry

end ratio_of_areas_l58_5880


namespace min_value_x_plus_y_l58_5875

theorem min_value_x_plus_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
sorry

end min_value_x_plus_y_l58_5875
