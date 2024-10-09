import Mathlib

namespace mh_range_l1766_176615

theorem mh_range (x m : ℝ) (h : 1 / 3 < x ∧ x < 1 / 2) (hx : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 := 
sorry

end mh_range_l1766_176615


namespace price_of_ice_cream_bar_is_correct_l1766_176631

noncomputable def price_ice_cream_bar (n_ice_cream_bars n_sundaes total_price price_of_sundae price_ice_cream_bar : ℝ) : Prop :=
  n_ice_cream_bars = 125 ∧
  n_sundaes = 125 ∧
  total_price = 225 ∧
  price_of_sundae = 1.2 →
  price_ice_cream_bar = 0.6

theorem price_of_ice_cream_bar_is_correct :
  price_ice_cream_bar 125 125 225 1.2 0.6 :=
by
  sorry

end price_of_ice_cream_bar_is_correct_l1766_176631


namespace total_cost_smore_night_l1766_176674

-- Define the costs per item
def cost_graham_cracker : ℝ := 0.10
def cost_marshmallow : ℝ := 0.15
def cost_chocolate : ℝ := 0.25
def cost_caramel_piece : ℝ := 0.20
def cost_toffee_piece : ℝ := 0.05

-- Calculate the cost for each ingredient per S'more
def cost_caramel : ℝ := 2 * cost_caramel_piece
def cost_toffee : ℝ := 4 * cost_toffee_piece

-- Total cost of one S'more
def cost_one_smore : ℝ :=
  cost_graham_cracker + cost_marshmallow + cost_chocolate + cost_caramel + cost_toffee

-- Number of people and S'mores per person
def num_people : ℕ := 8
def smores_per_person : ℕ := 3

-- Total number of S'mores
def total_smores : ℕ := num_people * smores_per_person

-- Total cost of all the S'mores
def total_cost : ℝ := total_smores * cost_one_smore

-- The final statement
theorem total_cost_smore_night : total_cost = 26.40 := 
  sorry

end total_cost_smore_night_l1766_176674


namespace total_cost_of_feeding_pets_for_one_week_l1766_176657

-- Definitions based on conditions
def turtle_food_per_weight : ℚ := 1 / (1 / 2)
def turtle_weight : ℚ := 30
def turtle_food_qty_per_jar : ℚ := 15
def turtle_food_cost_per_jar : ℚ := 3

def bird_food_per_weight : ℚ := 2
def bird_weight : ℚ := 8
def bird_food_qty_per_bag : ℚ := 40
def bird_food_cost_per_bag : ℚ := 5

def hamster_food_per_weight : ℚ := 1.5 / (1 / 2)
def hamster_weight : ℚ := 3
def hamster_food_qty_per_box : ℚ := 20
def hamster_food_cost_per_box : ℚ := 4

-- Theorem stating the equivalent proof problem
theorem total_cost_of_feeding_pets_for_one_week :
  let turtle_food_needed := (turtle_weight * turtle_food_per_weight)
  let turtle_jars_needed := turtle_food_needed / turtle_food_qty_per_jar
  let turtle_cost := turtle_jars_needed * turtle_food_cost_per_jar
  let bird_food_needed := (bird_weight * bird_food_per_weight)
  let bird_bags_needed := bird_food_needed / bird_food_qty_per_bag
  let bird_cost := if bird_bags_needed < 1 then bird_food_cost_per_bag else bird_bags_needed * bird_food_cost_per_bag
  let hamster_food_needed := (hamster_weight * hamster_food_per_weight)
  let hamster_boxes_needed := hamster_food_needed / hamster_food_qty_per_box
  let hamster_cost := if hamster_boxes_needed < 1 then hamster_food_cost_per_box else hamster_boxes_needed * hamster_food_cost_per_box
  turtle_cost + bird_cost + hamster_cost = 21 :=
by
  sorry

end total_cost_of_feeding_pets_for_one_week_l1766_176657


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l1766_176608

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l1766_176608


namespace sum_of_squares_l1766_176618

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l1766_176618


namespace egyptian_method_percentage_error_l1766_176682

theorem egyptian_method_percentage_error :
  let a := 6
  let b := 4
  let c := 20
  let h := Real.sqrt (c^2 - ((a - b) / 2)^2)
  let S := ((a + b) / 2) * h
  let S1 := ((a + b) * c) / 2
  let percentage_error := abs ((20 / Real.sqrt 399) - 1) * 100
  percentage_error = abs ((20 / Real.sqrt 399) - 1) * 100 := by
  sorry

end egyptian_method_percentage_error_l1766_176682


namespace power_root_l1766_176641

noncomputable def x : ℝ := 1024 ^ (1 / 5)

theorem power_root (h : 1024 = 2^10) : x = 4 :=
by
  sorry

end power_root_l1766_176641


namespace sum_A_B_C_l1766_176655

noncomputable def number_B (A : ℕ) : ℕ := (A * 5) / 2
noncomputable def number_C (B : ℕ) : ℕ := (B * 7) / 4

theorem sum_A_B_C (A B C : ℕ) (h1 : A = 16) (h2 : A * 5 = B * 2) (h3 : B * 7 = C * 4) :
  A + B + C = 126 :=
by
  sorry

end sum_A_B_C_l1766_176655


namespace find_c_for_radius_of_circle_l1766_176601

theorem find_c_for_radius_of_circle :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 6 * y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25 - c) ∧
  (∀ x y : ℝ, (x + 4)^2 + (y - 3)^2 = 25 → c = 0) :=
sorry

end find_c_for_radius_of_circle_l1766_176601


namespace diameter_of_circle_given_radius_l1766_176637

theorem diameter_of_circle_given_radius (radius: ℝ) (h: radius = 7): 
  2 * radius = 14 :=
by
  rw [h]
  sorry

end diameter_of_circle_given_radius_l1766_176637


namespace previous_monthly_income_l1766_176679

variable (I : ℝ)

-- Conditions from the problem
def condition1 (I : ℝ) : Prop := 0.40 * I = 0.25 * (I + 600)

theorem previous_monthly_income (h : condition1 I) : I = 1000 := by
  sorry

end previous_monthly_income_l1766_176679


namespace votes_cast_l1766_176670

-- Define the conditions as given in the problem.
def total_votes (V : ℕ) := 35 * V / 100 + (35 * V / 100 + 2400) = V

-- The goal is to prove that the number of total votes V equals 8000.
theorem votes_cast : ∃ V : ℕ, total_votes V ∧ V = 8000 :=
by
  sorry -- The proof is not required, only the statement.

end votes_cast_l1766_176670


namespace circle_tangent_radius_l1766_176683

-- Define the radii of the three given circles
def radius1 : ℝ := 1.0
def radius2 : ℝ := 2.0
def radius3 : ℝ := 3.0

-- Define the problem statement: finding the radius of the fourth circle externally tangent to the given three circles
theorem circle_tangent_radius (r1 r2 r3 : ℝ) (cond1 : r1 = 1) (cond2 : r2 = 2) (cond3 : r3 = 3) : 
  ∃ R : ℝ, R = 6 := by
  sorry

end circle_tangent_radius_l1766_176683


namespace dima_picks_more_berries_l1766_176605

theorem dima_picks_more_berries (N : ℕ) (dima_fastness : ℕ) (sergei_fastness : ℕ) (dima_rate : ℕ) (sergei_rate : ℕ) :
  N = 450 → dima_fastness = 2 * sergei_fastness →
  dima_rate = 1 → sergei_rate = 2 →
  let dima_basket : ℕ := N / 2
  let sergei_basket : ℕ := (2 * N) / 3
  dima_basket > sergei_basket ∧ (dima_basket - sergei_basket) = 50 := 
by {
  sorry
}

end dima_picks_more_berries_l1766_176605


namespace tangent_parallel_x_axis_monotonically_increasing_intervals_l1766_176644

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

theorem tangent_parallel_x_axis (m n : ℝ) (h : m ≠ 0) (h_tangent : 3 * m * (2:ℝ)^2 + 2 * n * (2:ℝ) = 0) :
  n = -3 * m :=
by
  sorry

theorem monotonically_increasing_intervals (m : ℝ) (h : m ≠ 0) : 
  (∀ x : ℝ, 3 * m * x * (x - (2 : ℝ)) > 0 ↔ 
    if m > 0 then x < 0 ∨ 2 < x else 0 < x ∧ x < 2) :=
by
  sorry

end tangent_parallel_x_axis_monotonically_increasing_intervals_l1766_176644


namespace inequality_solution_l1766_176695

theorem inequality_solution (x : ℝ) :
  x + 1 ≥ -3 ∧ -2 * (x + 3) > 0 ↔ -4 ≤ x ∧ x < -3 :=
by sorry

end inequality_solution_l1766_176695


namespace number_of_item_B_l1766_176693

theorem number_of_item_B
    (x y z : ℕ)
    (total_items total_cost : ℕ)
    (hx_price : 1 ≤ x ∧ x ≤ 100)
    (hy_price : 1 ≤ y ∧ y ≤ 100)
    (hz_price : 1 ≤ z ∧ z ≤ 100)
    (h_total_items : total_items = 100)
    (h_total_cost : total_cost = 100)
    (h_price_equation : (x / 8) + 10 * y = z)
    (h_item_equation : x + y + (total_items - (x + y)) = total_items)
    : total_items - (x + y) = 21 :=
sorry

end number_of_item_B_l1766_176693


namespace relationship_among_abc_l1766_176639

noncomputable def a : ℝ := Real.logb 11 10
noncomputable def b : ℝ := (Real.logb 11 9) ^ 2
noncomputable def c : ℝ := Real.logb 10 11

theorem relationship_among_abc : b < a ∧ a < c :=
  sorry

end relationship_among_abc_l1766_176639


namespace Masc_age_difference_l1766_176640

theorem Masc_age_difference (masc_age sam_age : ℕ) (h1 : masc_age + sam_age = 27) (h2 : masc_age = 17) (h3 : sam_age = 10) : masc_age - sam_age = 7 :=
by {
  -- Proof would go here, but it's omitted as per instructions
  sorry
}

end Masc_age_difference_l1766_176640


namespace playerA_winning_moves_l1766_176688

-- Definitions of the game
-- Circles are labeled from 1 to 9
inductive Circle
| A | B | C1 | C2 | C3 | C4 | C5 | C6 | C7

inductive Player
| A | B

def StraightLine (c1 c2 c3 : Circle) : Prop := sorry
-- The straight line property between circles is specified by the game rules

-- Initial conditions
def initial_conditions (playerA_move playerB_move : Circle) : Prop :=
  playerA_move = Circle.A ∧ playerB_move = Circle.B

-- Winning condition
def winning_move (move : Circle) : Prop := sorry
-- This will check if a move leads to a win for Player A

-- Equivalent proof problem
theorem playerA_winning_moves : ∀ (move : Circle), initial_conditions Circle.A Circle.B → 
  (move = Circle.C2 ∨ move = Circle.C3 ∨ move = Circle.C4) → winning_move move :=
by
  sorry

end playerA_winning_moves_l1766_176688


namespace at_least_one_greater_than_one_l1766_176665

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

end at_least_one_greater_than_one_l1766_176665


namespace small_box_dolls_l1766_176622

theorem small_box_dolls (x : ℕ) : 
  (5 * 7 + 9 * x = 71) → x = 4 :=
by
  sorry

end small_box_dolls_l1766_176622


namespace evaluate_log_expression_l1766_176648

noncomputable def evaluate_expression (x y : Real) : Real :=
  (Real.log x / Real.log (y ^ 8)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 7)) * 
  (Real.log (x ^ 7) / Real.log (y ^ 3)) * 
  (Real.log (y ^ 8) / Real.log (x ^ 2))

theorem evaluate_log_expression (x y : Real) : 
  evaluate_expression x y = (1 : Real) := sorry

end evaluate_log_expression_l1766_176648


namespace total_right_handed_players_is_correct_l1766_176619

variable (total_players : ℕ)
variable (throwers : ℕ)
variable (left_handed_non_throwers_ratio : ℕ)
variable (total_right_handed_players : ℕ)

theorem total_right_handed_players_is_correct
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers_ratio = 1 / 3)
  (h4 : total_right_handed_players = 53) :
  total_right_handed_players = throwers + (total_players - throwers) -
    left_handed_non_throwers_ratio * (total_players - throwers) :=
by
  sorry

end total_right_handed_players_is_correct_l1766_176619


namespace num_best_friends_l1766_176626

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end num_best_friends_l1766_176626


namespace divisible_by_9_l1766_176690

theorem divisible_by_9 (n : ℕ) : 9 ∣ (4^n + 15 * n - 1) :=
by
  sorry

end divisible_by_9_l1766_176690


namespace minimum_discount_l1766_176611

theorem minimum_discount (C M : ℝ) (profit_margin : ℝ) (x : ℝ) 
  (hC : C = 800) (hM : M = 1200) (hprofit_margin : profit_margin = 0.2) :
  (M * x - C ≥ C * profit_margin) → (x ≥ 0.8) :=
by
  -- Here, we need to solve the inequality given the conditions
  sorry

end minimum_discount_l1766_176611


namespace gcd_of_terms_l1766_176672

theorem gcd_of_terms (m n : ℕ) : gcd (4 * m^3 * n) (9 * m * n^3) = m * n := 
sorry

end gcd_of_terms_l1766_176672


namespace complete_the_square_l1766_176612

-- Define the initial condition
def initial_eqn (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0

-- Theorem statement for completing the square
theorem complete_the_square (x : ℝ) : initial_eqn x → (x - 3)^2 = 4 :=
by sorry

end complete_the_square_l1766_176612


namespace geo_seq_ratio_l1766_176624

theorem geo_seq_ratio (S : ℕ → ℝ) (r : ℝ) (hS : ∀ n, S n = (1 - r^(n+1)) / (1 - r))
  (hS_ratio : S 10 / S 5 = 1 / 2) : S 15 / S 5 = 3 / 4 := 
by
  sorry

end geo_seq_ratio_l1766_176624


namespace range_cos_2alpha_cos_2beta_l1766_176636

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end range_cos_2alpha_cos_2beta_l1766_176636


namespace total_apples_for_bobbing_l1766_176625

theorem total_apples_for_bobbing (apples_per_bucket : ℕ) (buckets : ℕ) (total_apples : ℕ) : 
  apples_per_bucket = 9 → buckets = 7 → total_apples = apples_per_bucket * buckets → total_apples = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_apples_for_bobbing_l1766_176625


namespace prime_roots_range_l1766_176698

theorem prime_roots_range (p : ℕ) (hp : Prime p) (h : ∃ x₁ x₂ : ℤ, x₁ + x₂ = -p ∧ x₁ * x₂ = -444 * p) : 31 < p ∧ p ≤ 41 :=
by sorry

end prime_roots_range_l1766_176698


namespace problem1_problem2_l1766_176656

-- Definitions based on the given conditions
def p (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < 3 * a
def q (x : ℝ) : Prop := x^2 - 5 * x + 6 < 0

-- Problem (1)
theorem problem1 (a x : ℝ) (h : a = 1) (hp : p a x) (hq : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : ∀ x, q x → p a x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end problem1_problem2_l1766_176656


namespace number_of_mixed_vegetable_plates_l1766_176654

theorem number_of_mixed_vegetable_plates :
  ∃ n : ℕ, n * 70 = 1051 - (16 * 6 + 5 * 45 + 6 * 40) ∧ n = 7 :=
by
  sorry

end number_of_mixed_vegetable_plates_l1766_176654


namespace find_line_AB_l1766_176664

noncomputable def equation_of_line_AB : Prop :=
  ∀ (x y : ℝ), ((x-2)^2 + (y-1)^2 = 10) ∧ ((x+6)^2 + (y+3)^2 = 50) → (2*x + y = 0)

theorem find_line_AB : equation_of_line_AB := by
  sorry

end find_line_AB_l1766_176664


namespace coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l1766_176634

section coexistent_rational_number_pairs

-- Definitions based on the problem conditions:
def coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Proof problem 1
theorem coexistent_pair_example : coexistent_pair 3 (1/2) :=
sorry

-- Proof problem 2
theorem coexistent_pair_neg (m n : ℚ) (h : coexistent_pair m n) :
  coexistent_pair (-n) (-m) :=
sorry

-- Proof problem 3
example : ∃ (p q : ℚ), coexistent_pair p q ∧ (p, q) ≠ (2, 1/3) ∧ (p, q) ≠ (5, 2/3) ∧ (p, q) ≠ (3, 1/2) :=
sorry

-- Proof problem 4
theorem coexistent_pair_find_a (a : ℚ) (h : coexistent_pair a 3) :
  a = -2 :=
sorry

end coexistent_rational_number_pairs

end coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l1766_176634


namespace simplify_and_evaluate_expression_l1766_176651

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 2) : 
  (1 / (x - 3) / (1 / (x^2 - 9)) - x / (x + 1) * ((x^2 + x) / x^2)) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l1766_176651


namespace total_rainfall_l1766_176671

theorem total_rainfall (R1 R2 : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 15) : R1 + R2 = 25 := 
by
  sorry

end total_rainfall_l1766_176671


namespace range_of_a_l1766_176650

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 :=
by
  sorry

end range_of_a_l1766_176650


namespace smallest_integral_k_no_real_roots_l1766_176628

theorem smallest_integral_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧ 
           (∀ j : ℤ, j < k → (∃ x : ℝ, 2 * x * (j * x - 4) - x^2 + 6 = 0)) ∧
           k = 2 :=
by sorry

end smallest_integral_k_no_real_roots_l1766_176628


namespace milk_rate_proof_l1766_176680

theorem milk_rate_proof
  (initial_milk : ℕ := 30000)
  (time_pumped_out : ℕ := 4)
  (rate_pumped_out : ℕ := 2880)
  (time_adding_milk : ℕ := 7)
  (final_milk : ℕ := 28980) :
  ((final_milk - (initial_milk - time_pumped_out * rate_pumped_out)) / time_adding_milk = 1500) :=
by {
  sorry
}

end milk_rate_proof_l1766_176680


namespace necessary_but_not_sufficient_condition_l1766_176645

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((x + 2) * (x - 3) < 0 → |x - 1| < 2) ∧ (¬(|x - 1| < 2 → (x + 2) * (x - 3) < 0)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1766_176645


namespace min_x_plus_2y_l1766_176632

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : x + 2 * y ≥ (1 / 2) + Real.sqrt 3 :=
sorry

end min_x_plus_2y_l1766_176632


namespace at_least_one_nonzero_l1766_176623

theorem at_least_one_nonzero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end at_least_one_nonzero_l1766_176623


namespace smallest_n_correct_l1766_176614

/-- The first term of the geometric sequence. -/
def a₁ : ℚ := 5 / 6

/-- The second term of the geometric sequence. -/
def a₂ : ℚ := 25

/-- The common ratio for the geometric sequence. -/
def r : ℚ := a₂ / a₁

/-- The nth term of the geometric sequence. -/
def a_n (n : ℕ) : ℚ := a₁ * r^(n - 1)

/-- The smallest n such that the nth term is divisible by 10^7. -/
def smallest_n : ℕ := 8

theorem smallest_n_correct :
  ∀ n : ℕ, (a₁ * r^(n - 1)) ∣ (10^7 : ℚ) ↔ n = smallest_n := 
sorry

end smallest_n_correct_l1766_176614


namespace smallest_common_multiple_l1766_176602

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end smallest_common_multiple_l1766_176602


namespace birth_death_rate_interval_l1766_176681

theorem birth_death_rate_interval
  (b_rate : ℕ) (d_rate : ℕ) (population_increase_one_day : ℕ) (seconds_in_one_day : ℕ)
  (net_increase_per_t_seconds : ℕ) (t : ℕ)
  (h1 : b_rate = 5)
  (h2 : d_rate = 3)
  (h3 : population_increase_one_day = 86400)
  (h4 : seconds_in_one_day = 86400)
  (h5 : net_increase_per_t_seconds = b_rate - d_rate)
  (h6 : population_increase_one_day = net_increase_per_t_seconds * (seconds_in_one_day / t)) :
  t = 2 :=
by
  sorry

end birth_death_rate_interval_l1766_176681


namespace number_of_uncertain_events_is_three_l1766_176604

noncomputable def cloudy_day_will_rain : Prop := sorry
noncomputable def fair_coin_heads : Prop := sorry
noncomputable def two_students_same_birth_month : Prop := sorry
noncomputable def olympics_2008_in_beijing : Prop := true

def is_uncertain (event: Prop) : Prop :=
  event ∧ ¬(event = true ∨ event = false)

theorem number_of_uncertain_events_is_three :
  is_uncertain cloudy_day_will_rain ∧
  is_uncertain fair_coin_heads ∧
  is_uncertain two_students_same_birth_month ∧
  ¬is_uncertain olympics_2008_in_beijing →
  3 = 3 :=
by sorry

end number_of_uncertain_events_is_three_l1766_176604


namespace find_interest_rate_of_second_part_l1766_176643

-- Definitions for the problem
def total_sum : ℚ := 2678
def P2 : ℚ := 1648
def P1 : ℚ := total_sum - P2
def r1 : ℚ := 0.03  -- 3% per annum
def t1 : ℚ := 8     -- 8 years
def I1 : ℚ := P1 * r1 * t1
def t2 : ℚ := 3     -- 3 years

-- Statement to prove
theorem find_interest_rate_of_second_part : ∃ r2 : ℚ, I1 = P2 * r2 * t2 ∧ r2 * 100 = 5 := by
  sorry

end find_interest_rate_of_second_part_l1766_176643


namespace max_points_on_four_coplanar_circles_l1766_176661

noncomputable def max_points_on_circles (num_circles : ℕ) (max_intersections : ℕ) : ℕ :=
num_circles * max_intersections

theorem max_points_on_four_coplanar_circles :
  max_points_on_circles 4 2 = 8 := 
sorry

end max_points_on_four_coplanar_circles_l1766_176661


namespace barbara_spent_total_l1766_176617

variables (cost_steaks cost_chicken total_spent per_pound_steak per_pound_chicken : ℝ)
variables (weight_steaks weight_chicken : ℝ)

-- Defining the given conditions
def conditions :=
  per_pound_steak = 15 ∧
  weight_steaks = 4.5 ∧
  cost_steaks = per_pound_steak * weight_steaks ∧

  per_pound_chicken = 8 ∧
  weight_chicken = 1.5 ∧
  cost_chicken = per_pound_chicken * weight_chicken

-- Proving the total spent by Barbara is $79.50
theorem barbara_spent_total 
  (h : conditions per_pound_steak weight_steaks cost_steaks per_pound_chicken weight_chicken cost_chicken) : 
  total_spent = 79.5 :=
sorry

end barbara_spent_total_l1766_176617


namespace largest_prime_factor_13231_l1766_176616

-- Define the conditions
def is_prime (n : ℕ) : Prop := ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

-- State the problem as a theorem in Lean 4
theorem largest_prime_factor_13231 (H1 : 13231 = 121 * 109) 
    (H2 : is_prime 109)
    (H3 : 121 = 11^2) :
    ∃ p, is_prime p ∧ p ∣ 13231 ∧ ∀ q, is_prime q ∧ q ∣ 13231 → q ≤ p :=
by
  sorry

end largest_prime_factor_13231_l1766_176616


namespace problem_proof_l1766_176699

-- Define I, J, and K respectively to be 9^20, 3^41, 3
def I : ℕ := 9^20
def J : ℕ := 3^41
def K : ℕ := 3

theorem problem_proof : I + I + I = J := by
  -- Lean structure placeholder
  sorry

end problem_proof_l1766_176699


namespace total_students_in_class_l1766_176642

-- Definitions of the conditions
def E : ℕ := 55
def T : ℕ := 85
def N : ℕ := 30
def B : ℕ := 20

-- Statement of the theorem to prove the total number of students
theorem total_students_in_class : (E + T - B) + N = 150 := by
  -- Proof is omitted
  sorry

end total_students_in_class_l1766_176642


namespace arithmetic_sequence_properties_l1766_176627

-- Defining the arithmetic sequence and the conditions
variable {a : ℕ → ℤ}
variable {d : ℤ}
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = d

-- Given conditions
variable (h1 : a 5 = 10)
variable (h2 : a 1 + a 2 + a 3 = 3)

-- The theorem to prove
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d → a 1 = -2 ∧ d = 3 :=
sorry

end arithmetic_sequence_properties_l1766_176627


namespace no_maximum_value_l1766_176620

-- Define the conditions and the expression in Lean
def expression (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + a*b + c*d

def condition (a b c d : ℝ) : Prop := a * d - b * c = 1

theorem no_maximum_value : ¬ ∃ M, ∀ a b c d, condition a b c d → expression a b c d ≤ M := by
  sorry

end no_maximum_value_l1766_176620


namespace remainder_2_pow_19_div_7_l1766_176606

theorem remainder_2_pow_19_div_7 :
  2^19 % 7 = 2 := by
  sorry

end remainder_2_pow_19_div_7_l1766_176606


namespace abs_sum_of_first_six_a_sequence_terms_l1766_176600

def a_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -5
  | n+1 => a_sequence n + 2

theorem abs_sum_of_first_six_a_sequence_terms :
  |a_sequence 0| + |a_sequence 1| + |a_sequence 2| + |a_sequence 3| + |a_sequence 4| + |a_sequence 5| = 18 := sorry

end abs_sum_of_first_six_a_sequence_terms_l1766_176600


namespace function_passes_through_fixed_point_l1766_176667

noncomputable def f (a : ℝ) (x : ℝ) := 4 + Real.log (x + 1) / Real.log a

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  f a 0 = 4 := 
by
  sorry

end function_passes_through_fixed_point_l1766_176667


namespace holden_master_bath_size_l1766_176630

theorem holden_master_bath_size (b n m : ℝ) (h_b : b = 309) (h_n : n = 918) (h : 2 * (b + m) = n) : m = 150 := by
  sorry

end holden_master_bath_size_l1766_176630


namespace parakeet_eats_2_grams_per_day_l1766_176660

-- Define the conditions
def parrot_daily : ℕ := 14
def finch_daily (parakeet_daily : ℕ) : ℕ := parakeet_daily / 2
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def num_finches : ℕ := 4
def total_weekly_consumption : ℕ := 266

-- Define the daily consumption equation for all birds
def daily_consumption (parakeet_daily : ℕ) : ℕ :=
  num_parakeets * parakeet_daily + num_parrots * parrot_daily + num_finches * finch_daily parakeet_daily

-- Define the weekly consumption equation
def weekly_consumption (parakeet_daily : ℕ) : ℕ :=
  7 * daily_consumption parakeet_daily

-- State the theorem to prove that each parakeet eats 2 grams per day
theorem parakeet_eats_2_grams_per_day :
  (weekly_consumption 2) = total_weekly_consumption ↔ 2 = 2 :=
by
  sorry

end parakeet_eats_2_grams_per_day_l1766_176660


namespace expression_takes_many_values_l1766_176678

theorem expression_takes_many_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) :
  (∃ y : ℝ, y ≠ 0 ∧ y ≠ (y + 1) ∧ 
    (3 * x ^ 2 + 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 7) / ((x - 3) * (x + 2)) = y) :=
by
  sorry

end expression_takes_many_values_l1766_176678


namespace division_of_polynomial_l1766_176663

theorem division_of_polynomial (a : ℤ) : (-28 * a^3) / (7 * a) = -4 * a^2 := by
  sorry

end division_of_polynomial_l1766_176663


namespace total_is_twenty_l1766_176692

def num_blue := 5
def num_red := 7
def prob_red_or_white : ℚ := 0.75

noncomputable def total_marbles (T : ℕ) (W : ℕ) :=
  5 + 7 + W = T ∧ (7 + W) / T = prob_red_or_white

theorem total_is_twenty : ∃ (T : ℕ) (W : ℕ), total_marbles T W ∧ T = 20 :=
by
  sorry

end total_is_twenty_l1766_176692


namespace interest_rate_of_additional_investment_l1766_176685

section
variable (r : ℝ)

theorem interest_rate_of_additional_investment
  (h : 2800 * 0.05 + 1400 * r = 0.06 * (2800 + 1400)) :
  r = 0.08 := by
  sorry
end

end interest_rate_of_additional_investment_l1766_176685


namespace find_common_difference_l1766_176613

variable {a : ℕ → ℤ}  -- Define the arithmetic sequence as a function from natural numbers to integers
variable (d : ℤ)      -- Define the common difference

-- Assume the conditions given in the problem
axiom h1 : a 2 = 14
axiom h2 : a 5 = 5

theorem find_common_difference (n : ℕ) : d = -3 :=
by {
  -- This part will be filled in by the actual proof
  sorry
}

end find_common_difference_l1766_176613


namespace insured_fraction_l1766_176658

theorem insured_fraction (premium : ℝ) (rate : ℝ) (insured_value : ℝ) (original_value : ℝ)
  (h₁ : premium = 910)
  (h₂ : rate = 0.013)
  (h₃ : insured_value = premium / rate)
  (h₄ : original_value = 87500) :
  insured_value / original_value = 4 / 5 :=
by
  sorry

end insured_fraction_l1766_176658


namespace binary_to_decimal_and_septal_l1766_176646

theorem binary_to_decimal_and_septal :
  let bin : ℕ := 110101
  let dec : ℕ := 53
  let septal : ℕ := 104
  let convert_to_decimal (b : ℕ) : ℕ := 
    (b % 10) * 2^0 + ((b / 10) % 10) * 2^1 + ((b / 100) % 10) * 2^2 + 
    ((b / 1000) % 10) * 2^3 + ((b / 10000) % 10) * 2^4 + ((b / 100000) % 10) * 2^5
  let convert_to_septal (n : ℕ) : ℕ :=
    let rec aux (n : ℕ) (acc : ℕ) (place : ℕ) : ℕ :=
      if n = 0 then acc
      else aux (n / 7) (acc + (n % 7) * place) (place * 10)
    aux n 0 1
  convert_to_decimal bin = dec ∧ convert_to_septal dec = septal :=
by
  sorry

end binary_to_decimal_and_septal_l1766_176646


namespace frank_cookies_l1766_176694

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end frank_cookies_l1766_176694


namespace area_of_smaller_circle_l1766_176684

/-
  Variables and assumptions:
  r: Radius of the smaller circle
  R: Radius of the larger circle which is three times the smaller circle. Hence, R = 3 * r.
  PA = AB = 6: Lengths of the tangent segments
  Area: Calculated area of the smaller circle
-/

theorem area_of_smaller_circle (r : ℝ) (h1 : 6 = r) (h2 : 3 * 6 = R) (h3 : 6 = r) : 
  ∃ (area : ℝ), area = (36 * Real.pi) / 7 :=
by
  sorry 

end area_of_smaller_circle_l1766_176684


namespace ratio_of_blue_to_purple_beads_l1766_176676

theorem ratio_of_blue_to_purple_beads :
  ∃ (B G : ℕ), 
    7 + B + G = 46 ∧ 
    G = B + 11 ∧ 
    B / 7 = 2 :=
by
  sorry

end ratio_of_blue_to_purple_beads_l1766_176676


namespace smallest_prime_number_conditions_l1766_176607

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum -- Summing the digits in base 10

def is_prime (n : ℕ) : Prop := Nat.Prime n

def smallest_prime_number (n : ℕ) : Prop :=
  is_prime n ∧ sum_of_digits n = 17 ∧ n > 200 ∧
  (∀ m : ℕ, is_prime m ∧ sum_of_digits m = 17 ∧ m > 200 → n ≤ m)

theorem smallest_prime_number_conditions (p : ℕ) : 
  smallest_prime_number p ↔ p = 197 :=
by
  sorry

end smallest_prime_number_conditions_l1766_176607


namespace july_savings_l1766_176687

theorem july_savings (january: ℕ := 100) (total_savings: ℕ := 12700) :
  let february := 2 * january
  let march := 2 * february
  let april := 2 * march
  let may := 2 * april
  let june := 2 * may
  let july := 2 * june
  let total := january + february + march + april + may + june + july
  total = total_savings → july = 6400 := 
by
  sorry

end july_savings_l1766_176687


namespace sum_first_15_terms_l1766_176677

noncomputable def sum_of_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

noncomputable def fourth_term (a d : ℝ) : ℝ := a + 3 * d
noncomputable def twelfth_term (a d : ℝ) : ℝ := a + 11 * d

theorem sum_first_15_terms (a d : ℝ) 
  (h : fourth_term a d + twelfth_term a d = 10) : sum_of_terms a d 15 = 75 :=
by
  sorry

end sum_first_15_terms_l1766_176677


namespace second_number_is_correct_l1766_176691

theorem second_number_is_correct (x : Real) (h : 108^2 + x^2 = 19928) : x = Real.sqrt 8264 :=
by
  sorry

end second_number_is_correct_l1766_176691


namespace find_x_l1766_176675

theorem find_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 :=
sorry

end find_x_l1766_176675


namespace correct_options_l1766_176653

-- Definitions for lines l and n
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def line_n (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- The condition for lines to be parallel, equating the slopes
def parallel_lines (a : ℝ) : Prop := -(a + 2) / a = -(a - 2) / 3

-- The condition that line l passes through the point (1, -1)
def passes_through_point (a : ℝ) : Prop := line_l a 1 (-1)

-- The theorem statement
theorem correct_options (a : ℝ) :
  (parallel_lines a → a = 6 ∨ a = -1) ∧ (passes_through_point a) :=
by
  sorry

end correct_options_l1766_176653


namespace sum_of_coefficients_of_expansion_l1766_176666

-- Define a predicate for a term being constant
def is_constant_term (n : ℕ) (term : ℚ) : Prop := 
  term = 0

-- Define the sum of coefficients computation
noncomputable def sum_of_coefficients (n : ℕ) : ℚ := 
  (1 - 3)^n

-- The main statement of the problem in Lean
theorem sum_of_coefficients_of_expansion {n : ℕ} 
  (h : is_constant_term n (2 * n - 10)) : 
  sum_of_coefficients 5 = -32 := 
sorry

end sum_of_coefficients_of_expansion_l1766_176666


namespace max_cards_l1766_176638

def card_cost : ℝ := 0.85
def budget : ℝ := 7.50

theorem max_cards (n : ℕ) : card_cost * n ≤ budget → n ≤ 8 :=
by sorry

end max_cards_l1766_176638


namespace prime_gt_three_square_mod_twelve_l1766_176649

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end prime_gt_three_square_mod_twelve_l1766_176649


namespace solution_exists_l1766_176697

noncomputable def verify_triples (a b c : ℝ) : Prop :=
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ b = -2 * a ∧ c = 4 * a

theorem solution_exists (a b c : ℝ) : verify_triples a b c :=
by
  sorry

end solution_exists_l1766_176697


namespace right_triangle_side_lengths_l1766_176629

theorem right_triangle_side_lengths (a S : ℝ) (b c : ℝ)
  (h1 : S = b + c)
  (h2 : c^2 = a^2 + b^2) :
  b = (S^2 - a^2) / (2 * S) ∧ c = (S^2 + a^2) / (2 * S) :=
by
  sorry

end right_triangle_side_lengths_l1766_176629


namespace part1_part2_l1766_176669

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

def p (m : ℝ) : Prop :=
  let Δ := discriminant 1 m 1
  Δ > 0 ∧ -m / 2 < 0

def q (m : ℝ) : Prop :=
  let Δ := discriminant 4 (4 * (m - 2)) 1
  Δ < 0

theorem part1 (m : ℝ) (hp : p m) : m > 2 := 
sorry

theorem part2 (m : ℝ) (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m ≥ 3) ∨ (1 < m ∧ m ≤ 2) := 
sorry

end part1_part2_l1766_176669


namespace cherie_sparklers_count_l1766_176635

-- Conditions
def koby_boxes : ℕ := 2
def koby_sparklers_per_box : ℕ := 3
def koby_whistlers_per_box : ℕ := 5
def cherie_boxes : ℕ := 1
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := 33

-- Total number of fireworks Koby has
def koby_total_fireworks : ℕ :=
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box)

-- Total number of fireworks Cherie has
def cherie_total_fireworks : ℕ :=
  total_fireworks - koby_total_fireworks

-- Number of sparklers in Cherie's box
def cherie_sparklers : ℕ :=
  cherie_total_fireworks - cherie_whistlers

-- Proof statement
theorem cherie_sparklers_count : cherie_sparklers = 8 := by
  sorry

end cherie_sparklers_count_l1766_176635


namespace complete_the_square_l1766_176621

theorem complete_the_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 :=
by sorry

end complete_the_square_l1766_176621


namespace max_cards_mod3_l1766_176673

theorem max_cards_mod3 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) : 
  ∃ t ⊆ s, t.card = 6 ∧ (t.prod id) % 3 = 1 := sorry

end max_cards_mod3_l1766_176673


namespace hyperbola_eccentricity_l1766_176662

theorem hyperbola_eccentricity (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : b ^ 2 = (5 / 4) * a ^ 2) 
  (h_c : c ^ 2 = a ^ 2 + b ^ 2) : 
  (3 / 2) = c / a :=
by sorry

end hyperbola_eccentricity_l1766_176662


namespace prove_fraction_l1766_176686

noncomputable def michael_brothers_problem (M O Y : ℕ) :=
  Y = 5 ∧
  M + O + Y = 28 ∧
  O = 2 * (M - 1) + 1 →
  Y / O = 1 / 3

theorem prove_fraction (M O Y : ℕ) : michael_brothers_problem M O Y :=
  sorry

end prove_fraction_l1766_176686


namespace cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l1766_176610

-- Define the initial state of the cube vertices
def initial_cube : ℕ → ℕ
| 0 => 1  -- The number at vertex 0 is 1
| _ => 0  -- The numbers at other vertices are 0

-- Define the edge addition operation
def edge_add (v1 v2 : ℕ → ℕ) (edge : ℕ × ℕ) : ℕ → ℕ :=
  λ x => if x = edge.1 ∨ x = edge.2 then v1 x + 1 else v1 x

-- Condition: one can add one to the numbers at the ends of any edge
axiom edge_op : ∀ (v : ℕ → ℕ) (e : ℕ × ℕ), ℕ → ℕ

-- Defining the problem in Lean
theorem cube_numbers_not_all_even :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 2 = 0) :=
by
  -- Proof not required
  sorry

theorem cube_numbers_not_all_divisible_by_3 :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 3 = 0) :=
by
  -- Proof not required
  sorry

end cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l1766_176610


namespace proportional_function_range_l1766_176647

theorem proportional_function_range (m : ℝ) (h : ∀ x : ℝ, (x < 0 → (1 - m) * x > 0) ∧ (x > 0 → (1 - m) * x < 0)) : m > 1 :=
by sorry

end proportional_function_range_l1766_176647


namespace point_on_transformed_graph_l1766_176668

variable (f : ℝ → ℝ)

theorem point_on_transformed_graph :
  (f 12 = 10) →
  3 * (19 / 9) = (f (3 * 4)) / 3 + 3 ∧ (4 + 19 / 9 = 55 / 9) :=
by
  sorry

end point_on_transformed_graph_l1766_176668


namespace max_diff_y_l1766_176609

theorem max_diff_y (x y z : ℕ) (h₁ : 4 < x) (h₂ : x < z) (h₃ : z < y) (h₄ : y < 10) (h₅ : y - x = 5) : y = 9 :=
sorry

end max_diff_y_l1766_176609


namespace range_of_a_l1766_176689

open Set

noncomputable def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1}

theorem range_of_a :
  (∀ x, x ∈ B a → x ∈ A) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l1766_176689


namespace ratio_of_areas_ratio_of_perimeters_l1766_176696

-- Define side lengths
def side_length_A : ℕ := 48
def side_length_B : ℕ := 60

-- Define the area of squares
def area_square (side_length : ℕ) : ℕ := side_length * side_length

-- Define the perimeter of squares
def perimeter_square (side_length : ℕ) : ℕ := 4 * side_length

-- Theorem for the ratio of areas
theorem ratio_of_areas : (area_square side_length_A) / (area_square side_length_B) = 16 / 25 :=
by
  sorry

-- Theorem for the ratio of perimeters
theorem ratio_of_perimeters : (perimeter_square side_length_A) / (perimeter_square side_length_B) = 4 / 5 :=
by
  sorry

end ratio_of_areas_ratio_of_perimeters_l1766_176696


namespace find_a_l1766_176652

-- Definitions for the problem
def quadratic_distinct_roots (a : ℝ) : Prop :=
  let Δ := a^2 - 16
  Δ > 0

def satisfies_root_equation (x1 x2 : ℝ) : Prop :=
  (x1^2 - (20 / (3 * x2^3)) = x2^2 - (20 / (3 * x1^3)))

-- Main statement of the proof problem
theorem find_a (a x1 x2 : ℝ) (h_quadratic_roots : quadratic_distinct_roots a)
               (h_root_equation : satisfies_root_equation x1 x2)
               (h_vieta_sum : x1 + x2 = -a) (h_vieta_product : x1 * x2 = 4) :
  a = -10 :=
by
  sorry

end find_a_l1766_176652


namespace nat_divisibility_l1766_176603

theorem nat_divisibility (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
  sorry

end nat_divisibility_l1766_176603


namespace parabola_directrix_l1766_176633

theorem parabola_directrix (x y : ℝ) (h : y = 8 * x^2) : y = -1 / 32 :=
sorry

end parabola_directrix_l1766_176633


namespace probability_of_winning_pair_l1766_176659

-- Conditions: Define the deck composition and the winning pair.
inductive Color
| Red
| Green
| Blue

inductive Label
| A
| B
| C

structure Card :=
(color : Color)
(label : Label)

def deck : List Card :=
  [ {color := Color.Red, label := Label.A},
    {color := Color.Red, label := Label.B},
    {color := Color.Red, label := Label.C},
    {color := Color.Green, label := Label.A},
    {color := Color.Green, label := Label.B},
    {color := Color.Green, label := Label.C},
    {color := Color.Blue, label := Label.A},
    {color := Color.Blue, label := Label.B},
    {color := Color.Blue, label := Label.C} ]

def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

-- Question: Prove the probability of drawing a winning pair.
theorem probability_of_winning_pair :
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2 ∧ is_winning_pair c1 c2) →
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2) →
  (9 + 9) / 36 = 1 / 2 :=
sorry

end probability_of_winning_pair_l1766_176659
