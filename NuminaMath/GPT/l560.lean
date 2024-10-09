import Mathlib

namespace inequality_solution_l560_56028

theorem inequality_solution (x : ℝ) : (3 * x + 4 ≥ 4 * x) ∧ (2 * (x - 1) + x > 7) ↔ (3 < x ∧ x ≤ 4) := 
by 
  sorry

end inequality_solution_l560_56028


namespace cos2alpha_plus_sin2alpha_l560_56019

def point_angle_condition (x y : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  x = -3 ∧ y = 4 ∧ r = 5 ∧ x^2 + y^2 = r^2

theorem cos2alpha_plus_sin2alpha (α : ℝ) (x y r : ℝ)
  (h : point_angle_condition x y r α) : 
  (Real.cos (2 * α) + Real.sin (2 * α)) = -31/25 :=
by
  sorry

end cos2alpha_plus_sin2alpha_l560_56019


namespace three_monotonic_intervals_iff_a_lt_zero_l560_56057

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end three_monotonic_intervals_iff_a_lt_zero_l560_56057


namespace xiao_ming_final_score_l560_56033

theorem xiao_ming_final_score :
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  (speech_image * weight_speech_image +
   content * weight_content +
   effectiveness * weight_effectiveness) = 8.3 :=
by
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  sorry

end xiao_ming_final_score_l560_56033


namespace Heracles_age_l560_56004

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l560_56004


namespace lineup_count_l560_56001

theorem lineup_count (n k : ℕ) (h : n = 13) (k_eq : k = 4) : (n.choose k) = 715 := by
  sorry

end lineup_count_l560_56001


namespace abs_sq_lt_self_iff_l560_56082

theorem abs_sq_lt_self_iff {x : ℝ} : abs x * abs x < x ↔ (0 < x ∧ x < 1) ∨ (x < -1) :=
by
  sorry

end abs_sq_lt_self_iff_l560_56082


namespace cube_surface_area_l560_56011

noncomputable def total_surface_area_of_cube (Q : ℝ) : ℝ :=
  8 * Q * Real.sqrt 3 / 3

theorem cube_surface_area (Q : ℝ) (h : Q > 0) :
  total_surface_area_of_cube Q = 8 * Q * Real.sqrt 3 / 3 :=
sorry

end cube_surface_area_l560_56011


namespace faster_train_speed_l560_56037

theorem faster_train_speed
  (slower_train_speed : ℝ := 60) -- speed of the slower train in km/h
  (length_train1 : ℝ := 1.10) -- length of the slower train in km
  (length_train2 : ℝ := 0.9) -- length of the faster train in km
  (cross_time_sec : ℝ := 47.99999999999999) -- crossing time in seconds
  (cross_time : ℝ := cross_time_sec / 3600) -- crossing time in hours
  (total_distance : ℝ := length_train1 + length_train2) -- total distance covered
  (relative_speed : ℝ := total_distance / cross_time) -- relative speed
  (faster_train_speed : ℝ := relative_speed - slower_train_speed) -- speed of the faster train
  : faster_train_speed = 90 :=
by
  sorry

end faster_train_speed_l560_56037


namespace coefficient_of_y_l560_56047

theorem coefficient_of_y (x y a : ℝ) (h1 : 7 * x + y = 19) (h2 : x + a * y = 1) (h3 : 2 * x + y = 5) : a = 3 :=
sorry

end coefficient_of_y_l560_56047


namespace abs_neg_2023_l560_56068

theorem abs_neg_2023 : abs (-2023) = 2023 := 
by
  sorry

end abs_neg_2023_l560_56068


namespace total_votes_election_l560_56098

theorem total_votes_election 
  (votes_A : ℝ) 
  (valid_votes_percentage : ℝ) 
  (invalid_votes_percentage : ℝ)
  (votes_candidate_A : ℝ) 
  (total_votes : ℝ) 
  (h1 : votes_A = 0.60) 
  (h2 : invalid_votes_percentage = 0.15) 
  (h3 : votes_candidate_A = 285600) 
  (h4 : valid_votes_percentage = 0.85) 
  (h5 : total_votes = 560000) 
  : 
  ((votes_A * valid_votes_percentage * total_votes) = votes_candidate_A) 
  := 
  by sorry

end total_votes_election_l560_56098


namespace problem_trip_l560_56096

noncomputable def validate_trip (a b c : ℕ) (t : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10 ∧ 60 * t = 9 * c - 10 * b

theorem problem_trip (a b c t : ℕ) (h : validate_trip a b c t) : a^2 + b^2 + c^2 = 26 :=
sorry

end problem_trip_l560_56096


namespace tom_buys_oranges_l560_56089

theorem tom_buys_oranges (o a : ℕ) (h₁ : o + a = 7) (h₂ : (90 * o + 60 * a) % 100 = 0) : o = 6 := 
by 
  sorry

end tom_buys_oranges_l560_56089


namespace find_the_number_l560_56085

variable (x : ℕ)

theorem find_the_number (h : 43 + 3 * x = 58) : x = 5 :=
by 
  sorry

end find_the_number_l560_56085


namespace insects_remaining_l560_56035

-- Define the initial counts of spiders, ants, and ladybugs
def spiders : ℕ := 3
def ants : ℕ := 12
def ladybugs : ℕ := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away : ℕ := 2

-- Prove the total number of remaining insects in the playground
theorem insects_remaining : (spiders + ants + ladybugs - ladybugs_flew_away) = 21 := by
  -- Expand the definitions and compute the result
  sorry

end insects_remaining_l560_56035


namespace system_solutions_l560_56050

theorem system_solutions (x a : ℝ) (h1 : a = -3*x^2 + 5*x - 2) (h2 : (x + 2) * a = 4 * (x^2 - 1)) (hx : x ≠ -2) :
  (x = 0 ∧ a = -2) ∨ (x = 1 ∧ a = 0) ∨ (x = -8/3 ∧ a = -110/3) :=
  sorry

end system_solutions_l560_56050


namespace cost_price_of_article_l560_56097

theorem cost_price_of_article :
  ∃ (C : ℝ), 
  (∃ (G : ℝ), C + G = 500 ∧ C + 1.15 * G = 570) ∧ 
  C = (100 / 3) :=
by sorry

end cost_price_of_article_l560_56097


namespace molly_age_l560_56054

theorem molly_age
  (avg_age : ℕ)
  (hakimi_age : ℕ)
  (jared_age : ℕ)
  (molly_age : ℕ)
  (h1 : avg_age = 40)
  (h2 : hakimi_age = 40)
  (h3 : jared_age = hakimi_age + 10)
  (h4 : 3 * avg_age = hakimi_age + jared_age + molly_age) :
  molly_age = 30 :=
by
  sorry

end molly_age_l560_56054


namespace find_x_range_l560_56026

-- Given definition for a decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

-- The main theorem to prove
theorem find_x_range (f : ℝ → ℝ) (h_decreasing : is_decreasing f) :
  {x : ℝ | f (|1 / x|) < f 1} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end find_x_range_l560_56026


namespace points_per_touchdown_l560_56048

theorem points_per_touchdown (P : ℕ) (games : ℕ) (touchdowns_per_game : ℕ) (two_point_conversions : ℕ) (two_point_conversion_value : ℕ) (total_points : ℕ) :
  touchdowns_per_game = 4 →
  games = 15 →
  two_point_conversions = 6 →
  two_point_conversion_value = 2 →
  total_points = (4 * P * 15 + 6 * two_point_conversion_value) →
  total_points = 372 →
  P = 6 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end points_per_touchdown_l560_56048


namespace polynomial_inequality_solution_l560_56009

theorem polynomial_inequality_solution :
  { x : ℝ | x * (x - 5) * (x - 10)^2 > 0 } = { x : ℝ | 0 < x ∧ x < 5 ∨ 10 < x } :=
by
  sorry

end polynomial_inequality_solution_l560_56009


namespace product_of_y_values_l560_56065

theorem product_of_y_values :
  (∀ (x y : ℤ), x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → (y = 1 ∨ y = 2)) →
  (∀ (x y₁ x' y₂ : ℤ), (x, y₁) ≠ (x', y₂) → x = x' ∨ y₁ ≠ y₂) →
  (∀ (x y : ℤ), (x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → y = 1 ∨ y = 2) →
    (∃ (y₁ y₂ : ℤ), y₁ = 1 ∧ y₂ = 2 ∧ y₁ * y₂ = 2)) :=
by {
  sorry
}

end product_of_y_values_l560_56065


namespace xy_maximum_value_l560_56062

theorem xy_maximum_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2 * y) : x - 2 * y ≤ 2 / 3 :=
sorry

end xy_maximum_value_l560_56062


namespace greatest_divisor_form_p_plus_1_l560_56060

theorem greatest_divisor_form_p_plus_1 (n : ℕ) (hn : 0 < n):
  (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → 6 ∣ (p + 1)) ∧
  (∀ d : ℕ, (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → d ∣ (p + 1)) → d ≤ 6) :=
by {
  sorry
}

end greatest_divisor_form_p_plus_1_l560_56060


namespace eval_expression_l560_56029

-- Definitions based on the conditions and problem statement
def x (b : ℕ) : ℕ := b + 9

-- The theorem to prove
theorem eval_expression (b : ℕ) : x b - b + 5 = 14 := by
    sorry

end eval_expression_l560_56029


namespace sum_of_three_numbers_l560_56052

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 54) (h3 : c + a = 58) : 
  a + b + c = 73.5 :=
by
  sorry -- Proof is omitted

end sum_of_three_numbers_l560_56052


namespace atomic_number_R_l560_56043

noncomputable def atomic_number_Pb := 82
def electron_shell_difference := 32

def same_group_atomic_number 
  (atomic_number_Pb : ℕ) 
  (electron_shell_difference : ℕ) : 
  ℕ := 
  atomic_number_Pb + electron_shell_difference

theorem atomic_number_R (R : ℕ) : 
  same_group_atomic_number atomic_number_Pb electron_shell_difference = 114 := 
by
  sorry

end atomic_number_R_l560_56043


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l560_56005

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l560_56005


namespace part1_part2_l560_56025

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b : ℝ × ℝ := (3, -Real.sqrt 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) b

theorem part1 (hx : x ∈ Set.Icc 0 Real.pi) (h_perp : dot_product (a x) b = 0) : x = 5 * Real.pi / 6 :=
sorry

theorem part2 (hx : x ∈ Set.Icc 0 Real.pi) :
  (f x ≤ 2 * Real.sqrt 3) ∧ (f x = 2 * Real.sqrt 3 → x = 0) ∧
  (f x ≥ -2 * Real.sqrt 3) ∧ (f x = -2 * Real.sqrt 3 → x = 5 * Real.pi / 6) :=
sorry

end part1_part2_l560_56025


namespace differentiable_inequality_l560_56081

theorem differentiable_inequality 
  {a b : ℝ} 
  {f g : ℝ → ℝ} 
  (hdiff_f : DifferentiableOn ℝ f (Set.Icc a b))
  (hdiff_g : DifferentiableOn ℝ g (Set.Icc a b))
  (hderiv_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x > deriv g x)) :
  ∀ x ∈ Set.Ioo a b, f x + g a > g x + f a :=
by 
  sorry

end differentiable_inequality_l560_56081


namespace lines_intersect_l560_56022

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
  (2 + 3 * t, 2 - 4 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
  (4 + 5 * u, -6 + 3 * u)

theorem lines_intersect :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = (160 / 29, -160 / 29) :=
by
  sorry

end lines_intersect_l560_56022


namespace max_value_sqrt_abc_expression_l560_56073

theorem max_value_sqrt_abc_expression (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                                       (hb : 0 ≤ b) (hb1 : b ≤ 1)
                                       (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1) :=
sorry

end max_value_sqrt_abc_expression_l560_56073


namespace six_digit_number_unique_solution_l560_56021

theorem six_digit_number_unique_solution
    (a b c d e f : ℕ)
    (hN : (N : ℕ) = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)
    (hM : (M : ℕ) = 100000 * d + 10000 * e + 1000 * f + 100 * a + 10 * b + c)
    (h_eq : 7 * N = 6 * M) :
    N = 461538 :=
by
  sorry

end six_digit_number_unique_solution_l560_56021


namespace quadrilateral_area_l560_56015

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 30) (hh1 : h1 = 10) (hh2 : h2 = 6) :
  (1 / 2 * d * (h1 + h2) = 240) := by
  sorry

end quadrilateral_area_l560_56015


namespace avg_age_of_community_l560_56088

def ratio_of_populations (w m : ℕ) : Prop := w * 2 = m * 3
def avg_age (total_age population : ℚ) : ℚ := total_age / population

theorem avg_age_of_community 
    (k : ℕ)
    (total_women : ℕ := 3 * k) 
    (total_men : ℕ := 2 * k)
    (total_children : ℚ := (2 * k : ℚ) / 3)
    (avg_women_age : ℚ := 40)
    (avg_men_age : ℚ := 36)
    (avg_children_age : ℚ := 10)
    (total_women_age : ℚ := 40 * (3 * k))
    (total_men_age : ℚ := 36 * (2 * k))
    (total_children_age : ℚ := 10 * (total_children)) : 
    avg_age (total_women_age + total_men_age + total_children_age) (total_women + total_men + total_children) = 35 := 
    sorry

end avg_age_of_community_l560_56088


namespace length_of_bridge_is_80_l560_56031

-- Define the given constants
def length_of_train : ℕ := 280
def speed_of_train : ℕ := 18
def time_to_cross : ℕ := 20

-- Define the distance traveled by the train in the given time
def distance_traveled : ℕ := speed_of_train * time_to_cross

-- Define the length of the bridge from the given distance traveled
def length_of_bridge := distance_traveled - length_of_train

-- The theorem to prove the length of the bridge is 80 meters
theorem length_of_bridge_is_80 :
  length_of_bridge = 80 := by
  sorry

end length_of_bridge_is_80_l560_56031


namespace temperature_of_Huangshan_at_night_l560_56074

theorem temperature_of_Huangshan_at_night 
  (T_morning : ℤ) (Rise_noon : ℤ) (Drop_night : ℤ)
  (h1 : T_morning = -12) (h2 : Rise_noon = 8) (h3 : Drop_night = 10) :
  T_morning + Rise_noon - Drop_night = -14 :=
by
  sorry

end temperature_of_Huangshan_at_night_l560_56074


namespace number_of_three_star_reviews_l560_56061

theorem number_of_three_star_reviews:
  ∀ (x : ℕ),
  (6 * 5 + 7 * 4 + 1 * 2 + x * 3) / 18 = 4 →
  x = 4 :=
by
  intros x H
  sorry  -- Placeholder for the proof

end number_of_three_star_reviews_l560_56061


namespace both_fifth_and_ninth_terms_are_20_l560_56058

def sequence_a (n : ℕ) : ℕ := n^2 - 14 * n + 65

theorem both_fifth_and_ninth_terms_are_20 : sequence_a 5 = 20 ∧ sequence_a 9 = 20 := 
by
  sorry

end both_fifth_and_ninth_terms_are_20_l560_56058


namespace first_player_wins_the_game_l560_56017

-- Define the game state with 1992 stones and rules for taking stones
structure GameState where
  stones : Nat

-- Game rule: Each player can take a number of stones that is a divisor of the number of stones the 
-- opponent took on the previous turn
def isValidMove (prevMove: Nat) (currentMove: Nat) : Prop :=
  currentMove > 0 ∧ prevMove % currentMove = 0

-- The first player can take any number of stones but not all at once on their first move
def isFirstMoveValid (move: Nat) : Prop :=
  move > 0 ∧ move < 1992

-- Define the initial state of the game with 1992 stones
def initialGameState : GameState := { stones := 1992 }

-- Definition of optimal play leading to the first player's victory
def firstPlayerWins (s : GameState) : Prop :=
  s.stones = 1992 →
  ∃ move: Nat, isFirstMoveValid move ∧
  ∃ nextState: GameState, nextState.stones = s.stones - move ∧ 
  -- The first player wins with optimal strategy
  sorry

-- Theorem statement in Lean 4 equivalent to the math problem
theorem first_player_wins_the_game :
  firstPlayerWins initialGameState :=
  sorry

end first_player_wins_the_game_l560_56017


namespace questions_answered_second_half_l560_56002

theorem questions_answered_second_half :
  ∀ (q1 q2 p s : ℕ), q1 = 3 → p = 3 → s = 15 → s = (q1 + q2) * p → q2 = 2 :=
by
  intros q1 q2 p s hq1 hp hs h_final_score
  -- proofs go here, but we skip them
  sorry

end questions_answered_second_half_l560_56002


namespace sahil_selling_price_l560_56066

noncomputable def sales_tax : ℝ := 0.10 * 18000
noncomputable def initial_cost_with_tax : ℝ := 18000 + sales_tax

noncomputable def broken_part_cost : ℝ := 3000
noncomputable def software_update_cost : ℝ := 4000
noncomputable def total_repair_cost : ℝ := broken_part_cost + software_update_cost
noncomputable def service_tax_on_repair : ℝ := 0.05 * total_repair_cost
noncomputable def total_repair_cost_with_tax : ℝ := total_repair_cost + service_tax_on_repair

noncomputable def transportation_charges : ℝ := 1500
noncomputable def total_cost_before_depreciation : ℝ := initial_cost_with_tax + total_repair_cost_with_tax + transportation_charges

noncomputable def depreciation_first_year : ℝ := 0.15 * total_cost_before_depreciation
noncomputable def value_after_first_year : ℝ := total_cost_before_depreciation - depreciation_first_year

noncomputable def depreciation_second_year : ℝ := 0.15 * value_after_first_year
noncomputable def value_after_second_year : ℝ := value_after_first_year - depreciation_second_year

noncomputable def profit : ℝ := 0.50 * value_after_second_year
noncomputable def selling_price : ℝ := value_after_second_year + profit

theorem sahil_selling_price : selling_price = 31049.44 := by
  sorry

end sahil_selling_price_l560_56066


namespace min_value_proof_l560_56079

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  4 / (x + 3 * y) + 1 / (x - y)

theorem min_value_proof (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) : 
  min_value_expr x y = 9 / 4 := 
sorry

end min_value_proof_l560_56079


namespace metallic_sheet_dimension_l560_56036

theorem metallic_sheet_dimension :
  ∃ w : ℝ, (∀ (h := 8) (l := 40) (v := 2688),
    v = (w - 2 * h) * (l - 2 * h) * h) → w = 30 :=
by sorry

end metallic_sheet_dimension_l560_56036


namespace probability_red_or_white_l560_56084

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 := 
  sorry

end probability_red_or_white_l560_56084


namespace student_correct_numbers_l560_56063

theorem student_correct_numbers (x y : ℕ) 
  (h1 : (10 * x + 5) * y = 4500)
  (h2 : (10 * x + 3) * y = 4380) : 
  (10 * x + 5 = 75 ∧ y = 60) :=
by 
  sorry

end student_correct_numbers_l560_56063


namespace books_count_l560_56069

theorem books_count (Tim_books Total_books Mike_books : ℕ) (h1 : Tim_books = 22) (h2 : Total_books = 42) : Mike_books = 20 :=
by
  sorry

end books_count_l560_56069


namespace James_selling_percentage_l560_56010

def James_selling_percentage_proof : Prop :=
  ∀ (total_cost original_price return_cost extra_item bought_price out_of_pocket sold_amount : ℝ),
    total_cost = 3000 →
    return_cost = 700 + 500 →
    extra_item = 500 * 1.2 →
    bought_price = 100 →
    out_of_pocket = 2020 →
    sold_amount = out_of_pocket - (total_cost - return_cost + bought_price) →
    sold_amount / extra_item * 100 = 20

theorem James_selling_percentage : James_selling_percentage_proof :=
by
  sorry

end James_selling_percentage_l560_56010


namespace inequality_inequality_holds_l560_56027

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l560_56027


namespace pyramid_surface_area_and_volume_l560_56038

def s := 8
def PF := 15

noncomputable def FM := s / 2
noncomputable def PM := Real.sqrt (PF^2 + FM^2)
noncomputable def baseArea := s^2
noncomputable def lateralAreaTriangle := (1 / 2) * s * PM
noncomputable def totalSurfaceArea := baseArea + 4 * lateralAreaTriangle
noncomputable def volume := (1 / 3) * baseArea * PF

theorem pyramid_surface_area_and_volume :
  totalSurfaceArea = 64 + 16 * Real.sqrt 241 ∧
  volume = 320 :=
by
  sorry

end pyramid_surface_area_and_volume_l560_56038


namespace combine_sum_l560_56087

def A (n m : Nat) : Nat := n.factorial / (n - m).factorial
def C (n m : Nat) : Nat := n.factorial / (m.factorial * (n - m).factorial)

theorem combine_sum (n m : Nat) (hA : A n m = 272) (hC : C n m = 136) : m + n = 19 := by
  sorry

end combine_sum_l560_56087


namespace distinct_pairs_l560_56078

theorem distinct_pairs (x y : ℝ) (h : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧ x^200 - y^200 = 2^199 * (x - y) ↔ (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
by
  sorry

end distinct_pairs_l560_56078


namespace delta_ratio_l560_56071

theorem delta_ratio 
  (Δx : ℝ) (Δy : ℝ) 
  (y_new : ℝ := (1 + Δx)^2 + 1)
  (y_old : ℝ := 1^2 + 1)
  (Δy_def : Δy = y_new - y_old) :
  Δy / Δx = 2 + Δx :=
by
  sorry

end delta_ratio_l560_56071


namespace largest_n_satisfying_inequality_l560_56024

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, n ≥ 1 ∧ n^(6033) < 2011^(2011) ∧ ∀ m : ℕ, m > n → m^(6033) ≥ 2011^(2011) :=
sorry

end largest_n_satisfying_inequality_l560_56024


namespace shadow_taller_pot_length_l560_56023

-- Definitions based on the conditions a)
def height_shorter_pot : ℕ := 20
def shadow_shorter_pot : ℕ := 10
def height_taller_pot : ℕ := 40

-- The proof problem
theorem shadow_taller_pot_length : 
  ∃ (S2 : ℕ), (height_shorter_pot / shadow_shorter_pot = height_taller_pot / S2) ∧ S2 = 20 :=
sorry

end shadow_taller_pot_length_l560_56023


namespace largest_A_divisible_by_8_equal_quotient_remainder_l560_56083

theorem largest_A_divisible_by_8_equal_quotient_remainder :
  ∃ (A B C : ℕ), A = 8 * B + C ∧ B = C ∧ C < 8 ∧ A = 63 := by
  sorry

end largest_A_divisible_by_8_equal_quotient_remainder_l560_56083


namespace system_solution_b_l560_56076

theorem system_solution_b (x y b : ℚ) 
  (h1 : 4 * x + 2 * y = b) 
  (h2 : 3 * x + 7 * y = 3 * b) 
  (hy : y = 3) : 
  b = 22 / 3 := 
by
  sorry

end system_solution_b_l560_56076


namespace tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l560_56000

theorem tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m (m : ℝ) (α : ℝ)
  (h1 : Real.tan α = m / 3)
  (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l560_56000


namespace number_of_girls_in_school_l560_56095

theorem number_of_girls_in_school :
  ∃ G B : ℕ, 
    G + B = 1600 ∧
    (G * 200 / 1600) - 20 = (B * 200 / 1600) ∧
    G = 860 :=
by
  sorry

end number_of_girls_in_school_l560_56095


namespace molecular_weight_of_complex_compound_l560_56041

def molecular_weight (n : ℕ) (N_w : ℝ) (o : ℕ) (O_w : ℝ) (h : ℕ) (H_w : ℝ) (p : ℕ) (P_w : ℝ) : ℝ :=
  (n * N_w) + (o * O_w) + (h * H_w) + (p * P_w)

theorem molecular_weight_of_complex_compound :
  molecular_weight 2 14.01 5 16.00 3 1.01 1 30.97 = 142.02 :=
by
  sorry

end molecular_weight_of_complex_compound_l560_56041


namespace total_amount_received_l560_56094

theorem total_amount_received (B : ℝ) (h1 : (1/3) * B = 36) : (2/3 * B) * 4 = 288 :=
by
  sorry

end total_amount_received_l560_56094


namespace probability_of_8_or_9_ring_l560_56034

theorem probability_of_8_or_9_ring (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  p9 + p8 = 0.5 :=
by
  sorry

end probability_of_8_or_9_ring_l560_56034


namespace interval_of_n_l560_56039

noncomputable def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem interval_of_n (n : ℕ) (hn : 0 < n ∧ n < 2000)
  (h1 : divides n 9999)
  (h2 : divides (n + 4) 999999) :
  801 ≤ n ∧ n ≤ 1200 :=
sorry

end interval_of_n_l560_56039


namespace radius_of_circle_l560_56059

variable {O : Type*} [MetricSpace O]

def distance_near : ℝ := 1
def distance_far : ℝ := 7
def diameter : ℝ := distance_near + distance_far

theorem radius_of_circle (P : O) (r : ℝ) (h1 : distance_near = 1) (h2 : distance_far = 7) :
  r = diameter / 2 :=
by
  -- Proof would go here 
  sorry

end radius_of_circle_l560_56059


namespace mass_percentage_iodine_neq_662_l560_56072

theorem mass_percentage_iodine_neq_662 (atomic_mass_Al : ℝ) (atomic_mass_I : ℝ) (molar_mass_AlI3 : ℝ) :
  atomic_mass_Al = 26.98 ∧ atomic_mass_I = 126.90 ∧ molar_mass_AlI3 = ((1 * atomic_mass_Al) + (3 * atomic_mass_I)) →
  (3 * atomic_mass_I / molar_mass_AlI3 * 100) ≠ 6.62 :=
by
  sorry

end mass_percentage_iodine_neq_662_l560_56072


namespace square_lawn_area_l560_56016

theorem square_lawn_area (map_scale : ℝ) (map_edge_length_cm : ℝ) (actual_edge_length_m : ℝ) (actual_area_m2 : ℝ) 
  (h1 : map_scale = 1 / 5000) 
  (h2 : map_edge_length_cm = 4) 
  (h3 : actual_edge_length_m = (map_edge_length_cm / map_scale) / 100)
  (h4 : actual_area_m2 = actual_edge_length_m^2)
  : actual_area_m2 = 400 := 
by 
  sorry

end square_lawn_area_l560_56016


namespace four_digit_numbers_count_l560_56070

theorem four_digit_numbers_count : (3:ℕ) ^ 4 = 81 := by
  sorry

end four_digit_numbers_count_l560_56070


namespace problem_statement_l560_56013

theorem problem_statement (a b : ℤ) (h : |a + 5| + (b - 2) ^ 2 = 0) : (a + b) ^ 2010 = 3 ^ 2010 :=
by
  sorry

end problem_statement_l560_56013


namespace brendan_fish_caught_afternoon_l560_56007

theorem brendan_fish_caught_afternoon (morning_fish : ℕ) (thrown_fish : ℕ) (dads_fish : ℕ) (total_fish : ℕ) :
  morning_fish = 8 → thrown_fish = 3 → dads_fish = 13 → total_fish = 23 → 
  (morning_fish - thrown_fish) + dads_fish + brendan_afternoon_catch = total_fish → 
  brendan_afternoon_catch = 5 :=
by
  intros morning_fish_eq thrown_fish_eq dads_fish_eq total_fish_eq fish_sum_eq
  sorry

end brendan_fish_caught_afternoon_l560_56007


namespace line_through_midpoint_of_ellipse_l560_56012

theorem line_through_midpoint_of_ellipse:
  (∀ x y : ℝ, (x - 4)^2 + (y - 2)^2 = (1/36) * ((9 * 4) + 36 * (1 / 4)) → (1 + 2 * (y - 2) / (x - 4) = 0)) →
  (x - 8) + 2 * (y - 4) = 0 :=
by
  sorry

end line_through_midpoint_of_ellipse_l560_56012


namespace proportion_correct_l560_56014

theorem proportion_correct (m n : ℤ) (h : 6 * m = 7 * n) (hn : n ≠ 0) : (m : ℚ) / 7 = n / 6 :=
by sorry

end proportion_correct_l560_56014


namespace gcd_360_1260_l560_56090

theorem gcd_360_1260 : gcd 360 1260 = 180 := by
  /- 
  Prime factorization of 360 and 1260 is given:
  360 = 2^3 * 3^2 * 5
  1260 = 2^2 * 3^2 * 5 * 7
  These conditions are implicitly used to deduce the answer.
  -/
  sorry

end gcd_360_1260_l560_56090


namespace mouse_away_from_cheese_l560_56091

theorem mouse_away_from_cheese:
  ∃ a b : ℝ, a = 3 ∧ b = 3 ∧ (a + b = 6) ∧
  ∀ x y : ℝ, (y = -3 * x + 12) → 
  ∀ (a y₀ : ℝ), y₀ = (1/3) * a + 11 →
  (a, b) = (3, 3) :=
by
  sorry

end mouse_away_from_cheese_l560_56091


namespace johnny_distance_walked_l560_56018

theorem johnny_distance_walked
  (dist_q_to_y : ℕ) (matthew_rate : ℕ) (johnny_rate : ℕ) (time_diff : ℕ) (johnny_walked : ℕ):
  dist_q_to_y = 45 →
  matthew_rate = 3 →
  johnny_rate = 4 →
  time_diff = 1 →
  (∃ t: ℕ, johnny_walked = johnny_rate * t 
            ∧ dist_q_to_y = matthew_rate * (t + time_diff) + johnny_walked) →
  johnny_walked = 24 := by
  sorry

end johnny_distance_walked_l560_56018


namespace initial_blocks_l560_56080

variable (x : ℕ)

theorem initial_blocks (h : x + 30 = 65) : x = 35 := by
  sorry

end initial_blocks_l560_56080


namespace how_many_bananas_l560_56099

theorem how_many_bananas (total_fruit apples oranges : ℕ) 
  (h_total : total_fruit = 12) (h_apples : apples = 3) (h_oranges : oranges = 5) :
  total_fruit - apples - oranges = 4 :=
by
  sorry

end how_many_bananas_l560_56099


namespace price_of_70_cans_l560_56008

noncomputable def discounted_price (regular_price : ℝ) (discount_percent : ℝ) : ℝ :=
  regular_price * (1 - discount_percent / 100)

noncomputable def total_price (regular_price : ℝ) (discount_percent : ℝ) (total_cans : ℕ) (cans_per_case : ℕ) : ℝ :=
  let price_per_can := discounted_price regular_price discount_percent
  let full_cases := total_cans / cans_per_case
  let remaining_cans := total_cans % cans_per_case
  full_cases * cans_per_case * price_per_can + remaining_cans * price_per_can

theorem price_of_70_cans :
  total_price 0.55 25 70 24 = 28.875 :=
by
  sorry

end price_of_70_cans_l560_56008


namespace num_female_fox_terriers_l560_56064

def total_dogs : Nat := 2012
def total_female_dogs : Nat := 1110
def total_fox_terriers : Nat := 1506
def male_shih_tzus : Nat := 202

theorem num_female_fox_terriers :
    ∃ (female_fox_terriers: Nat), 
        female_fox_terriers = total_fox_terriers - (total_dogs - total_female_dogs - male_shih_tzus) := by
    sorry

end num_female_fox_terriers_l560_56064


namespace inequality_abc_l560_56030

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_abc_l560_56030


namespace cosine_of_tangent_line_at_e_l560_56003

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem cosine_of_tangent_line_at_e :
  let θ := Real.arctan 2
  Real.cos θ = Real.sqrt (1 / 5) := by
  sorry

end cosine_of_tangent_line_at_e_l560_56003


namespace value_of_A_cos_alpha_plus_beta_l560_56053

noncomputable def f (A x : ℝ) : ℝ := A * Real.cos (x / 4 + Real.pi / 6)

theorem value_of_A {A : ℝ}
  (h1 : f A (Real.pi / 3) = Real.sqrt 2) :
  A = 2 := 
by
  sorry

theorem cos_alpha_plus_beta {α β : ℝ}
  (hαβ1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (hαβ2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h2 : f 2 (4*α + 4*Real.pi/3) = -30 / 17)
  (h3 : f 2 (4*β - 2*Real.pi/3) = 8 / 5) :
  Real.cos (α + β) = -13 / 85 :=
by
  sorry

end value_of_A_cos_alpha_plus_beta_l560_56053


namespace _l560_56093

lemma triangle_inequality_theorem (a b c : ℝ) : 
  a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a > 0 ∧ b > 0 ∧ c > 0) := sorry

lemma no_triangle_1_2_3 : ¬ (1 + 2 > 3 ∧ 1 + 3 > 2 ∧ 2 + 3 > 1) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_3_8_5 : ¬ (3 + 8 > 5 ∧ 3 + 5 > 8 ∧ 8 + 5 > 3) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_4_5_10 : ¬ (4 + 5 > 10 ∧ 4 + 10 > 5 ∧ 5 + 10 > 4) := 
by simp [triangle_inequality_theorem]

lemma triangle_4_5_6 : 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4 := 
by simp [triangle_inequality_theorem]

end _l560_56093


namespace transaction_gain_per_year_l560_56056

noncomputable def principal : ℝ := 9000
noncomputable def time : ℝ := 2
noncomputable def rate_lending : ℝ := 6
noncomputable def rate_borrowing : ℝ := 4

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def total_interest_earned := simple_interest principal rate_lending time
noncomputable def total_interest_paid := simple_interest principal rate_borrowing time

noncomputable def total_gain := total_interest_earned - total_interest_paid
noncomputable def gain_per_year := total_gain / 2

theorem transaction_gain_per_year : gain_per_year = 180 :=
by
  sorry

end transaction_gain_per_year_l560_56056


namespace molecular_weight_neutralization_l560_56086

def molecular_weight_acetic_acid : ℝ := 
  (12.01 * 2) + (1.008 * 4) + (16.00 * 2)

def molecular_weight_sodium_hydroxide : ℝ := 
  22.99 + 16.00 + 1.008

def total_weight_acetic_acid (moles : ℝ) : ℝ := 
  molecular_weight_acetic_acid * moles

def total_weight_sodium_hydroxide (moles : ℝ) : ℝ := 
  molecular_weight_sodium_hydroxide * moles

def total_molecular_weight (moles_ac: ℝ) (moles_naoh : ℝ) : ℝ :=
  total_weight_acetic_acid moles_ac + 
  total_weight_sodium_hydroxide moles_naoh

theorem molecular_weight_neutralization :
  total_molecular_weight 7 10 = 820.344 :=
by
  sorry

end molecular_weight_neutralization_l560_56086


namespace find_fourth_number_l560_56046

theorem find_fourth_number (x y : ℝ) (h1 : 0.25 / x = 2 / y) (h2 : x = 0.75) : y = 6 :=
by
  sorry

end find_fourth_number_l560_56046


namespace circle_radius_l560_56067

theorem circle_radius (x y : ℝ) :
  x^2 + 2 * x + y^2 = 0 → 1 = 1 :=
by sorry

end circle_radius_l560_56067


namespace minimum_x_value_l560_56020

theorem minimum_x_value
  (sales_jan_may june_sales x : ℝ)
  (h_sales_jan_may : sales_jan_may = 38.6)
  (h_june_sales : june_sales = 5)
  (h_total_sales_condition : sales_jan_may + june_sales + 2 * june_sales * (1 + x / 100) + 2 * june_sales * (1 + x / 100)^2 ≥ 70) :
  x = 20 := by
  sorry

end minimum_x_value_l560_56020


namespace radius_of_given_circle_is_eight_l560_56032

noncomputable def radius_of_circle (diameter : ℝ) : ℝ := diameter / 2

theorem radius_of_given_circle_is_eight :
  radius_of_circle 16 = 8 :=
by
  sorry

end radius_of_given_circle_is_eight_l560_56032


namespace repeating_decimal_rational_representation_l560_56044

theorem repeating_decimal_rational_representation :
  (0.12512512512512514 : ℝ) = (125 / 999 : ℝ) :=
sorry

end repeating_decimal_rational_representation_l560_56044


namespace months_to_save_l560_56042

/-- The grandfather saves 530 yuan from his pension every month. -/
def savings_per_month : ℕ := 530

/-- The price of the smartphone is 2000 yuan. -/
def smartphone_price : ℕ := 2000

/-- The number of months needed to save enough money to buy the smartphone. -/
def months_needed : ℕ := smartphone_price / savings_per_month

/-- Proof that the number of months needed is 4. -/
theorem months_to_save : months_needed = 4 :=
by
  sorry

end months_to_save_l560_56042


namespace Travis_spends_on_cereal_l560_56077

theorem Travis_spends_on_cereal (boxes_per_week : ℕ) (cost_per_box : ℝ) (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3.00) 
  (h3 : weeks_per_year = 52) 
: boxes_per_week * weeks_per_year * cost_per_box = 312.00 := 
by
  sorry

end Travis_spends_on_cereal_l560_56077


namespace convert_base_10_to_base_8_l560_56049

theorem convert_base_10_to_base_8 (n : ℕ) (n_eq : n = 3275) : 
  n = 3275 → ∃ (a b c d : ℕ), (a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 = 6323) :=
by 
  sorry

end convert_base_10_to_base_8_l560_56049


namespace smallest_solution_l560_56051

theorem smallest_solution (x : ℕ) (h1 : 6 * x ≡ 17 [MOD 31]) (h2 : x ≡ 3 [MOD 7]) : x = 24 := 
by 
  sorry

end smallest_solution_l560_56051


namespace negation_of_exists_l560_56040

theorem negation_of_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_of_exists_l560_56040


namespace product_power_conjecture_calculate_expression_l560_56055

-- Conjecture Proof
theorem product_power_conjecture (a b : ℂ) (n : ℕ) : (a * b)^n = (a^n) * (b^n) :=
sorry

-- Calculation Proof
theorem calculate_expression : 
  ((-0.125 : ℂ)^2022) * ((2 : ℂ)^2021) * ((4 : ℂ)^2020) = (1 / 32 : ℂ) :=
sorry

end product_power_conjecture_calculate_expression_l560_56055


namespace min_neighbor_pairs_l560_56075

theorem min_neighbor_pairs (n : ℕ) (h : n = 2005) :
  ∃ (pairs : ℕ), pairs = 56430 :=
by
  sorry

end min_neighbor_pairs_l560_56075


namespace tennis_handshakes_l560_56092

theorem tennis_handshakes :
  let num_teams := 4
  let women_per_team := 2
  let total_women := num_teams * women_per_team
  let handshakes_per_woman := total_women - 2
  let total_handshakes_before_division := total_women * handshakes_per_woman
  let actual_handshakes := total_handshakes_before_division / 2
  actual_handshakes = 24 :=
by sorry

end tennis_handshakes_l560_56092


namespace age_of_b_l560_56045

variable (A B C : ℕ)

-- Conditions
def avg_abc : Prop := A + B + C = 78
def avg_ac : Prop := A + C = 58

-- Question: Prove that B = 20
theorem age_of_b (h1 : avg_abc A B C) (h2 : avg_ac A C) : B = 20 := 
by sorry

end age_of_b_l560_56045


namespace jenny_problem_l560_56006

def round_to_nearest_ten (n : ℤ) : ℤ :=
  if n % 10 < 5 then n - (n % 10) else n + (10 - n % 10)

theorem jenny_problem : round_to_nearest_ten (58 + 29) = 90 := 
by
  sorry

end jenny_problem_l560_56006
