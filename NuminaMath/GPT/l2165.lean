import Mathlib

namespace NUMINAMATH_GPT_goldie_worked_hours_last_week_l2165_216549

variable (H : ℕ)
variable (money_per_hour : ℕ := 5)
variable (hours_this_week : ℕ := 30)
variable (total_earnings : ℕ := 250)

theorem goldie_worked_hours_last_week :
  H = (total_earnings - hours_this_week * money_per_hour) / money_per_hour :=
sorry

end NUMINAMATH_GPT_goldie_worked_hours_last_week_l2165_216549


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l2165_216577

theorem right_triangle_hypotenuse_length
  (a b : ℝ)
  (ha : a = 12)
  (hb : b = 16) :
  c = 20 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l2165_216577


namespace NUMINAMATH_GPT_lower_percentage_increase_l2165_216541

theorem lower_percentage_increase (E P : ℝ) (h1 : 1.26 * E = 693) (h2 : (1 + P) * E = 660) : P = 0.2 := by
  sorry

end NUMINAMATH_GPT_lower_percentage_increase_l2165_216541


namespace NUMINAMATH_GPT_factorize_quadratic_l2165_216520

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end NUMINAMATH_GPT_factorize_quadratic_l2165_216520


namespace NUMINAMATH_GPT_rectangle_area_l2165_216558

theorem rectangle_area (w l : ℝ) (hw : w = 2) (hl : l = 3) : w * l = 6 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2165_216558


namespace NUMINAMATH_GPT_remainder_twice_sum_first_150_mod_10000_eq_2650_l2165_216551

theorem remainder_twice_sum_first_150_mod_10000_eq_2650 :
  let n := 150
  let S := n * (n + 1) / 2  -- Sum of first 150 numbers
  let result := 2 * S
  result % 10000 = 2650 :=
by
  sorry -- proof not required

end NUMINAMATH_GPT_remainder_twice_sum_first_150_mod_10000_eq_2650_l2165_216551


namespace NUMINAMATH_GPT_xiaomings_mother_money_l2165_216564

-- Definitions for the conditions
def price_A : ℕ := 6
def price_B : ℕ := 9
def units_more_A := 2

-- Main statement to prove
theorem xiaomings_mother_money (x : ℕ) (M : ℕ) :
  M = 6 * x ∧ M = 9 * (x - 2) → M = 36 :=
by
  -- Assuming the conditions are given
  rintro ⟨hA, hB⟩
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_xiaomings_mother_money_l2165_216564


namespace NUMINAMATH_GPT_emerson_row_distance_l2165_216530

theorem emerson_row_distance (d1 d2 total : ℕ) (h1 : d1 = 6) (h2 : d2 = 18) (h3 : total = 39) :
  15 = total - (d1 + d2) :=
by sorry

end NUMINAMATH_GPT_emerson_row_distance_l2165_216530


namespace NUMINAMATH_GPT_max_coefficient_terms_l2165_216543

theorem max_coefficient_terms (x : ℝ) :
  let n := 8
  let T_3 := 7 * x^2
  let T_4 := 7 * x
  true := by
  sorry

end NUMINAMATH_GPT_max_coefficient_terms_l2165_216543


namespace NUMINAMATH_GPT_right_triangle_30_60_90_l2165_216578

theorem right_triangle_30_60_90 (a b : ℝ) (h : a = 15) :
  (b = 30) ∧ (b = 15 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_30_60_90_l2165_216578


namespace NUMINAMATH_GPT_abs_neg_2023_eq_2023_l2165_216566

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_eq_2023_l2165_216566


namespace NUMINAMATH_GPT_solve_for_a_l2165_216596

theorem solve_for_a (a : ℝ) (h : 50 - |a - 2| = |4 - a|) :
  a = -22 ∨ a = 28 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l2165_216596


namespace NUMINAMATH_GPT_log_order_preservation_l2165_216527

theorem log_order_preservation {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (Real.log a > Real.log b) → (a > b) :=
by
  sorry

end NUMINAMATH_GPT_log_order_preservation_l2165_216527


namespace NUMINAMATH_GPT_continuous_stripe_probability_l2165_216536

open ProbabilityTheory

noncomputable def total_stripe_combinations : ℕ := 4 ^ 6

noncomputable def favorable_stripe_outcomes : ℕ := 3 * 4

theorem continuous_stripe_probability :
  (favorable_stripe_outcomes : ℚ) / (total_stripe_combinations : ℚ) = 3 / 1024 := by
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l2165_216536


namespace NUMINAMATH_GPT_find_f_ln2_l2165_216583

variable (f : ℝ → ℝ)

-- Condition: f is an odd function
axiom odd_fn : ∀ x : ℝ, f (-x) = -f x

-- Condition: f(x) = e^(-x) - 2 for x < 0
axiom def_fn : ∀ x : ℝ, x < 0 → f x = Real.exp (-x) - 2

-- Problem: Find f(ln 2)
theorem find_f_ln2 : f (Real.log 2) = 0 := by
  sorry

end NUMINAMATH_GPT_find_f_ln2_l2165_216583


namespace NUMINAMATH_GPT_units_digit_of_n_l2165_216518

-- Definitions
def units_digit (x : ℕ) : ℕ := x % 10

-- Conditions
variables (m n : ℕ)
axiom condition1 : m * n = 23^5
axiom condition2 : units_digit m = 4

-- Theorem statement
theorem units_digit_of_n : units_digit n = 8 :=
sorry

end NUMINAMATH_GPT_units_digit_of_n_l2165_216518


namespace NUMINAMATH_GPT_hyperbola_equation_l2165_216554

noncomputable def sqrt_cubed := Real.sqrt 3

theorem hyperbola_equation
  (P : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (hP : P = (1, sqrt_cubed))
  (hAsymptote : (1 / a)^2 - (sqrt_cubed / b)^2 = 0)
  (hAngle : ∀ F : ℝ × ℝ, ∀ O : ℝ × ℝ, (F.1 - 1)^2 + (F.2 - sqrt_cubed)^2 + F.1^2 + F.2^2 = 16) :
  (a^2 = 4) ∧ (b^2 = 12) ∧ (c = 4) →
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l2165_216554


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l2165_216542

theorem solution_set_of_inequality_system :
  (6 - 2 * x ≥ 0) ∧ (2 * x + 4 > 0) ↔ (-2 < x ∧ x ≤ 3) := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l2165_216542


namespace NUMINAMATH_GPT_intersection_of_A_and_B_is_2_l2165_216586

-- Define the sets A and B based on the given conditions
def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

-- State the theorem that needs to be proved
theorem intersection_of_A_and_B_is_2 : A ∩ B = {2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_is_2_l2165_216586


namespace NUMINAMATH_GPT_find_wall_width_l2165_216509

noncomputable def wall_width (painting_width : ℝ) (painting_height : ℝ) (wall_height : ℝ) (painting_coverage : ℝ) : ℝ :=
  (painting_width * painting_height) / (painting_coverage * wall_height)

-- Given constants
def painting_width : ℝ := 2
def painting_height : ℝ := 4
def wall_height : ℝ := 5
def painting_coverage : ℝ := 0.16
def expected_width : ℝ := 10

theorem find_wall_width : wall_width painting_width painting_height wall_height painting_coverage = expected_width := 
by
  sorry

end NUMINAMATH_GPT_find_wall_width_l2165_216509


namespace NUMINAMATH_GPT_inequality_result_l2165_216513

theorem inequality_result
  (a b : ℝ) 
  (x y : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) :
  x + y ≤ 0 :=
sorry

end NUMINAMATH_GPT_inequality_result_l2165_216513


namespace NUMINAMATH_GPT_prism_volume_l2165_216589

theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l2165_216589


namespace NUMINAMATH_GPT_car_b_speed_l2165_216529

theorem car_b_speed :
  ∀ (v : ℕ),
    (232 - 4 * v = 32) →
    v = 50 :=
  by
  sorry

end NUMINAMATH_GPT_car_b_speed_l2165_216529


namespace NUMINAMATH_GPT_good_jars_l2165_216528

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end NUMINAMATH_GPT_good_jars_l2165_216528


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2165_216523

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 + 2 * k * r1 + (k - 1) = 0 ∧ r2^2 + 2 * k * r2 + (k - 1) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2165_216523


namespace NUMINAMATH_GPT_dot_product_is_one_l2165_216557

variable (a : ℝ × ℝ := (1, 1))
variable (b : ℝ × ℝ := (-1, 2))

theorem dot_product_is_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_GPT_dot_product_is_one_l2165_216557


namespace NUMINAMATH_GPT_balls_per_bag_l2165_216567

theorem balls_per_bag (total_balls : ℕ) (total_bags : ℕ) (h1 : total_balls = 36) (h2 : total_bags = 9) : total_balls / total_bags = 4 :=
by
  sorry

end NUMINAMATH_GPT_balls_per_bag_l2165_216567


namespace NUMINAMATH_GPT_coat_price_proof_l2165_216545

variable (W : ℝ) -- wholesale price
variable (currentPrice : ℝ) -- current price of the coat

-- Condition 1: The retailer marked up the coat by 90%.
def markup_90 : Prop := currentPrice = 1.9 * W

-- Condition 2: Further $4 increase achieves a 100% markup.
def increase_4 : Prop := 2 * W - currentPrice = 4

-- Theorem: The current price of the coat is $76.
theorem coat_price_proof (h1 : markup_90 W currentPrice) (h2 : increase_4 W currentPrice) : currentPrice = 76 :=
sorry

end NUMINAMATH_GPT_coat_price_proof_l2165_216545


namespace NUMINAMATH_GPT_number_of_recipes_needed_l2165_216569

noncomputable def cookies_per_student : ℕ := 3
noncomputable def total_students : ℕ := 150
noncomputable def recipe_yield : ℕ := 20
noncomputable def attendance_drop_rate : ℝ := 0.30

theorem number_of_recipes_needed : 
  ⌈ (total_students * (1 - attendance_drop_rate) * cookies_per_student) / recipe_yield ⌉ = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_recipes_needed_l2165_216569


namespace NUMINAMATH_GPT_valid_integer_pairs_l2165_216579

theorem valid_integer_pairs :
  { (x, y) : ℤ × ℤ |
    (∃ α β : ℝ, α^2 + β^2 < 4 ∧ α + β = (-x : ℝ) ∧ α * β = y ∧ x^2 - 4 * y ≥ 0) } =
  {(-2,1), (-1,-1), (-1,0), (0, -1), (0,0), (1,0), (1,-1), (2,1)} :=
sorry

end NUMINAMATH_GPT_valid_integer_pairs_l2165_216579


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l2165_216511

-- Define the polynomial f(r)
def f (r : ℝ) : ℝ := r ^ 15 + 1

-- Define the polynomial divisor g(r)
def g (r : ℝ) : ℝ := r + 1

-- State the theorem about the remainder when f(r) is divided by g(r)
theorem remainder_of_polynomial_division : 
  (f (-1)) = 0 := by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l2165_216511


namespace NUMINAMATH_GPT_range_of_a_l2165_216516

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a - x > 1 → x < 2 * a - 1)) ∧
  (∀ x : ℝ, (2 * x + 5 > 3 * a → x > (3 * a - 5) / 2)) ∧
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 6 →
    (x < 2 * a - 1 ∧ x > (3 * a - 5) / 2))) →
  7 / 3 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2165_216516


namespace NUMINAMATH_GPT_maximum_teams_tied_for_most_wins_l2165_216533

/-- In a round-robin tournament with 8 teams, each team plays one game
    against each other team, and each game results in one team winning
    and one team losing. -/
theorem maximum_teams_tied_for_most_wins :
  ∀ (teams games wins : ℕ), 
    teams = 8 → 
    games = (teams * (teams - 1)) / 2 →
    wins = 28 →
    ∃ (max_tied_teams : ℕ), max_tied_teams = 5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_teams_tied_for_most_wins_l2165_216533


namespace NUMINAMATH_GPT_rationalize_denominator_l2165_216535

theorem rationalize_denominator : 
  let A := -13 
  let B := -9
  let C := 3
  let D := 2
  let E := 165
  let F := 51
  A + B + C + D + E + F = 199 := by
sorry

end NUMINAMATH_GPT_rationalize_denominator_l2165_216535


namespace NUMINAMATH_GPT_distance_post_office_l2165_216565

theorem distance_post_office 
  (D : ℝ)
  (speed_to_post_office : ℝ := 25)
  (speed_back : ℝ := 4)
  (total_time : ℝ := 5 + (48 / 60)) :
  (D / speed_to_post_office + D / speed_back = total_time) → D = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_post_office_l2165_216565


namespace NUMINAMATH_GPT_negation_of_exists_l2165_216571

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l2165_216571


namespace NUMINAMATH_GPT_quadratic_algebraic_expression_l2165_216546

theorem quadratic_algebraic_expression (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
    a + b - a * b = 2 := by
  sorry

end NUMINAMATH_GPT_quadratic_algebraic_expression_l2165_216546


namespace NUMINAMATH_GPT_geometric_product_l2165_216512

theorem geometric_product (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 10) 
  (h2 : 1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 + 1 / a 6 = 5) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end NUMINAMATH_GPT_geometric_product_l2165_216512


namespace NUMINAMATH_GPT_radius_wheel_l2165_216560

noncomputable def pi : ℝ := 3.14159

theorem radius_wheel (D : ℝ) (N : ℕ) (r : ℝ) (h1 : D = 760.57) (h2 : N = 500) :
  r = (D / N) / (2 * pi) :=
sorry

end NUMINAMATH_GPT_radius_wheel_l2165_216560


namespace NUMINAMATH_GPT_find_x_plus_y_l2165_216524

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 7 * y = 17) (h2 : 3 * x + 5 * y = 11) : x + y = 83 / 23 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l2165_216524


namespace NUMINAMATH_GPT_unknown_card_value_l2165_216587

theorem unknown_card_value (cards_total : ℕ)
  (p1_hand : ℕ) (p1_hand_extra : ℕ) (table_card1 : ℕ) (total_card_values : ℕ)
  (sum_removed_cards_sets : ℕ)
  (n : ℕ) :
  cards_total = 40 ∧ 
  p1_hand = 5 ∧ 
  p1_hand_extra = 3 ∧ 
  table_card1 = 9 ∧ 
  total_card_values = 220 ∧ 
  sum_removed_cards_sets = 15 * n → 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 10 ∧ total_card_values = p1_hand + p1_hand_extra + table_card1 + x + sum_removed_cards_sets → 
  x = 8 := 
sorry

end NUMINAMATH_GPT_unknown_card_value_l2165_216587


namespace NUMINAMATH_GPT_football_goal_average_increase_l2165_216539

theorem football_goal_average_increase :
  ∀ (A : ℝ), 4 * A + 2 = 8 → (8 / 5) - A = 0.1 :=
by
  intro A
  intro h
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_football_goal_average_increase_l2165_216539


namespace NUMINAMATH_GPT_max_value_x2_plus_2xy_l2165_216538

open Real

theorem max_value_x2_plus_2xy (x y : ℝ) (h : x + y = 5) : 
  ∃ (M : ℝ), (M = x^2 + 2 * x * y) ∧ (∀ z w : ℝ, z + w = 5 → z^2 + 2 * z * w ≤ M) :=
by
  sorry

end NUMINAMATH_GPT_max_value_x2_plus_2xy_l2165_216538


namespace NUMINAMATH_GPT_tangent_product_power_l2165_216555

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180))
  * (1 + Real.tan (2 * Real.pi / 180))
  * (1 + Real.tan (3 * Real.pi / 180))
  * (1 + Real.tan (4 * Real.pi / 180))
  * (1 + Real.tan (5 * Real.pi / 180))
  * (1 + Real.tan (6 * Real.pi / 180))
  * (1 + Real.tan (7 * Real.pi / 180))
  * (1 + Real.tan (8 * Real.pi / 180))
  * (1 + Real.tan (9 * Real.pi / 180))
  * (1 + Real.tan (10 * Real.pi / 180))
  * (1 + Real.tan (11 * Real.pi / 180))
  * (1 + Real.tan (12 * Real.pi / 180))
  * (1 + Real.tan (13 * Real.pi / 180))
  * (1 + Real.tan (14 * Real.pi / 180))
  * (1 + Real.tan (15 * Real.pi / 180))
  * (1 + Real.tan (16 * Real.pi / 180))
  * (1 + Real.tan (17 * Real.pi / 180))
  * (1 + Real.tan (18 * Real.pi / 180))
  * (1 + Real.tan (19 * Real.pi / 180))
  * (1 + Real.tan (20 * Real.pi / 180))
  * (1 + Real.tan (21 * Real.pi / 180))
  * (1 + Real.tan (22 * Real.pi / 180))
  * (1 + Real.tan (23 * Real.pi / 180))
  * (1 + Real.tan (24 * Real.pi / 180))
  * (1 + Real.tan (25 * Real.pi / 180))
  * (1 + Real.tan (26 * Real.pi / 180))
  * (1 + Real.tan (27 * Real.pi / 180))
  * (1 + Real.tan (28 * Real.pi / 180))
  * (1 + Real.tan (29 * Real.pi / 180))
  * (1 + Real.tan (30 * Real.pi / 180))
  * (1 + Real.tan (31 * Real.pi / 180))
  * (1 + Real.tan (32 * Real.pi / 180))
  * (1 + Real.tan (33 * Real.pi / 180))
  * (1 + Real.tan (34 * Real.pi / 180))
  * (1 + Real.tan (35 * Real.pi / 180))
  * (1 + Real.tan (36 * Real.pi / 180))
  * (1 + Real.tan (37 * Real.pi / 180))
  * (1 + Real.tan (38 * Real.pi / 180))
  * (1 + Real.tan (39 * Real.pi / 180))
  * (1 + Real.tan (40 * Real.pi / 180))
  * (1 + Real.tan (41 * Real.pi / 180))
  * (1 + Real.tan (42 * Real.pi / 180))
  * (1 + Real.tan (43 * Real.pi / 180))
  * (1 + Real.tan (44 * Real.pi / 180))
  * (1 + Real.tan (45 * Real.pi / 180))
  * (1 + Real.tan (46 * Real.pi / 180))
  * (1 + Real.tan (47 * Real.pi / 180))
  * (1 + Real.tan (48 * Real.pi / 180))
  * (1 + Real.tan (49 * Real.pi / 180))
  * (1 + Real.tan (50 * Real.pi / 180))
  * (1 + Real.tan (51 * Real.pi / 180))
  * (1 + Real.tan (52 * Real.pi / 180))
  * (1 + Real.tan (53 * Real.pi / 180))
  * (1 + Real.tan (54 * Real.pi / 180))
  * (1 + Real.tan (55 * Real.pi / 180))
  * (1 + Real.tan (56 * Real.pi / 180))
  * (1 + Real.tan (57 * Real.pi / 180))
  * (1 + Real.tan (58 * Real.pi / 180))
  * (1 + Real.tan (59 * Real.pi / 180))
  * (1 + Real.tan (60 * Real.pi / 180))

theorem tangent_product_power : tangent_product = 2^30 := by
  sorry

end NUMINAMATH_GPT_tangent_product_power_l2165_216555


namespace NUMINAMATH_GPT_broker_investment_increase_l2165_216521

noncomputable def final_value_stock_A := 
  let initial := 100.0
  let year1 := initial * (1 + 0.80)
  let year2 := year1 * (1 - 0.30)
  year2 * (1 + 0.10)

noncomputable def final_value_stock_B := 
  let initial := 100.0
  let year1 := initial * (1 + 0.50)
  let year2 := year1 * (1 - 0.10)
  year2 * (1 - 0.25)

noncomputable def final_value_stock_C := 
  let initial := 100.0
  let year1 := initial * (1 - 0.30)
  let year2 := year1 * (1 - 0.40)
  year2 * (1 + 0.80)

noncomputable def final_value_stock_D := 
  let initial := 100.0
  let year1 := initial * (1 + 0.40)
  let year2 := year1 * (1 + 0.20)
  year2 * (1 - 0.15)

noncomputable def total_final_value := 
  final_value_stock_A + final_value_stock_B + final_value_stock_C + final_value_stock_D

noncomputable def initial_total_value := 4 * 100.0

noncomputable def net_increase := total_final_value - initial_total_value

noncomputable def net_increase_percentage := (net_increase / initial_total_value) * 100

theorem broker_investment_increase : net_increase_percentage = 14.5625 := 
by
  sorry

end NUMINAMATH_GPT_broker_investment_increase_l2165_216521


namespace NUMINAMATH_GPT_remaining_amount_l2165_216504

def initial_amount : ℕ := 18
def spent_amount : ℕ := 16

theorem remaining_amount : initial_amount - spent_amount = 2 := 
by sorry

end NUMINAMATH_GPT_remaining_amount_l2165_216504


namespace NUMINAMATH_GPT_part1_part2_l2165_216593

theorem part1 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) = 0) → 
  a = -1 ∧ ∀ x : ℝ, (4 * x^2 + 4 * x + 1 = 0 ↔ x = -1/2) :=
sorry

theorem part2 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) > 0) → 
  a > -1 ∧ a ≠ 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2165_216593


namespace NUMINAMATH_GPT_brenda_initial_points_l2165_216500

theorem brenda_initial_points
  (b : ℕ)  -- points scored by Brenda in her play
  (initial_advantage :ℕ := 22)  -- Brenda is initially 22 points ahead
  (david_score : ℕ := 32)  -- David scores 32 points
  (final_advantage : ℕ := 5)  -- Brenda is 5 points ahead after both plays
  (h : initial_advantage + b - david_score = final_advantage) :
  b = 15 :=
by
  sorry

end NUMINAMATH_GPT_brenda_initial_points_l2165_216500


namespace NUMINAMATH_GPT_range_of_m_l2165_216501

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

def tangent_points (m : ℝ) (x₀ : ℝ) : Prop := 
  2 * x₀ ^ 3 - 3 * x₀ ^ 2 + m + 3 = 0

theorem range_of_m (m : ℝ) :
  (∀ x₀, tangent_points m x₀) ∧ m ≠ -2 → (-3 < m ∧ m < -2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2165_216501


namespace NUMINAMATH_GPT_barbara_total_candies_l2165_216563

theorem barbara_total_candies :
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855 := 
by
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  show boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855
  sorry

end NUMINAMATH_GPT_barbara_total_candies_l2165_216563


namespace NUMINAMATH_GPT_total_time_spent_l2165_216522

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end NUMINAMATH_GPT_total_time_spent_l2165_216522


namespace NUMINAMATH_GPT_carrie_weekly_earning_l2165_216572

-- Definitions and conditions
def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weeks_needed : ℕ := 7

-- Calculate the required weekly earning
def weekly_earning : ℕ := (iphone_cost - trade_in_value) / weeks_needed

-- Problem statement: Prove that Carrie makes $80 per week babysitting
theorem carrie_weekly_earning :
  weekly_earning = 80 := by
  sorry

end NUMINAMATH_GPT_carrie_weekly_earning_l2165_216572


namespace NUMINAMATH_GPT_not_right_triangle_condition_C_l2165_216559

theorem not_right_triangle_condition_C :
  ∀ (a b c : ℝ), 
    (a^2 = b^2 + c^2) ∨
    (∀ (angleA angleB angleC : ℝ), angleA = angleB + angleC ∧ angleA + angleB + angleC = 180) ∨
    (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5) ∨
    (a^2 / b^2 = 1 / 2 ∧ b^2 / c^2 = 2 / 3) ->
    ¬ (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 -> angleA = 90 ∨ angleB = 90 ∨ angleC = 90) :=
by
  intro a b c h
  cases h
  case inl h1 =>
    -- Option A: b^2 = a^2 - c^2
    sorry
  case inr h2 =>
    cases h2
    case inl h3 => 
      -- Option B: angleA = angleB + angleC
      sorry
    case inr h4 =>
      cases h4
      case inl h5 =>
        -- Option C: angleA : angleB : angleC = 3 : 4 : 5
        sorry
      case inr h6 =>
        -- Option D: a^2 : b^2 : c^2 = 1 : 2 : 3
        sorry

end NUMINAMATH_GPT_not_right_triangle_condition_C_l2165_216559


namespace NUMINAMATH_GPT_barbara_candies_l2165_216531

theorem barbara_candies :
  ∀ (initial left used : ℝ), initial = 18 ∧ left = 9 → initial - left = used → used = 9 :=
by
  intros initial left used h1 h2
  sorry

end NUMINAMATH_GPT_barbara_candies_l2165_216531


namespace NUMINAMATH_GPT_sum_of_coefficients_l2165_216506

theorem sum_of_coefficients :
  (Nat.choose 50 3 + Nat.choose 50 5) = 2138360 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2165_216506


namespace NUMINAMATH_GPT_union_sets_l2165_216582

noncomputable def setA : Set ℝ := { x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def setB : Set ℝ := { x | 1 < x ∧ x < 5 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l2165_216582


namespace NUMINAMATH_GPT_trigonometric_identity_l2165_216598

theorem trigonometric_identity : (1 / 4) * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2165_216598


namespace NUMINAMATH_GPT_percentage_increase_equiv_l2165_216502

theorem percentage_increase_equiv {P : ℝ} : 
  (P * (1 + 0.08) * (1 + 0.08)) = (P * 1.1664) :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_equiv_l2165_216502


namespace NUMINAMATH_GPT_problem_solution_l2165_216574

noncomputable def f (x a : ℝ) : ℝ :=
  2 * (Real.cos x)^2 - 2 * a * Real.cos x - (2 * a + 1)

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a < 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem problem_solution :
  g a = 1 ∨ g a = (-a^2 / 2 - 2 * a - 1) ∨ g a = 1 - 4 * a →
  (∀ a, g a = 1 / 2 → a = -1) ∧ (f x (-1) ≤ 5) :=
sorry

end NUMINAMATH_GPT_problem_solution_l2165_216574


namespace NUMINAMATH_GPT_number_is_28_l2165_216584

-- Definitions from conditions in part a
def inner_expression := 15 - 15
def middle_expression := 37 - inner_expression
def outer_expression (some_number : ℕ) := 45 - (some_number - middle_expression)

-- Lean 4 statement to state the proof problem
theorem number_is_28 (some_number : ℕ) (h : outer_expression some_number = 54) : some_number = 28 := by
  sorry

end NUMINAMATH_GPT_number_is_28_l2165_216584


namespace NUMINAMATH_GPT_smallest_marble_count_l2165_216552

theorem smallest_marble_count (N : ℕ) (a b c : ℕ) (h1 : N > 1)
  (h2 : N ≡ 2 [MOD 5])
  (h3 : N ≡ 2 [MOD 7])
  (h4 : N ≡ 2 [MOD 9]) : N = 317 :=
sorry

end NUMINAMATH_GPT_smallest_marble_count_l2165_216552


namespace NUMINAMATH_GPT_zeros_of_f_is_pm3_l2165_216526

def f (x : ℝ) : ℝ := x^2 - 9

theorem zeros_of_f_is_pm3 :
  ∃ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by sorry

end NUMINAMATH_GPT_zeros_of_f_is_pm3_l2165_216526


namespace NUMINAMATH_GPT_veronica_loss_more_than_seth_l2165_216510

noncomputable def seth_loss : ℝ := 17.5
noncomputable def jerome_loss : ℝ := 3 * seth_loss
noncomputable def total_loss : ℝ := 89
noncomputable def veronica_loss : ℝ := total_loss - (seth_loss + jerome_loss)

theorem veronica_loss_more_than_seth :
  veronica_loss - seth_loss = 1.5 :=
by
  have h_seth_loss : seth_loss = 17.5 := rfl
  have h_jerome_loss : jerome_loss = 3 * seth_loss := rfl
  have h_total_loss : total_loss = 89 := rfl
  have h_veronica_loss : veronica_loss = total_loss - (seth_loss + jerome_loss) := rfl
  sorry

end NUMINAMATH_GPT_veronica_loss_more_than_seth_l2165_216510


namespace NUMINAMATH_GPT_possible_values_of_x_l2165_216525

theorem possible_values_of_x (x : ℝ) (h : (x^2 - 1) / x = 0) (hx : x ≠ 0) : x = 1 ∨ x = -1 :=
  sorry

end NUMINAMATH_GPT_possible_values_of_x_l2165_216525


namespace NUMINAMATH_GPT_choose_three_consecutive_circles_l2165_216547

theorem choose_three_consecutive_circles (n : ℕ) (hn : n = 33) : 
  ∃ (ways : ℕ), ways = 57 :=
by
  sorry

end NUMINAMATH_GPT_choose_three_consecutive_circles_l2165_216547


namespace NUMINAMATH_GPT_troy_initial_straws_l2165_216575

theorem troy_initial_straws (total_piglets : ℕ) (straws_per_piglet : ℕ)
  (fraction_adult_pigs : ℚ) (fraction_piglets : ℚ) 
  (adult_pigs_straws : ℕ) (piglets_straws : ℕ) 
  (total_straws : ℕ) (initial_straws : ℚ) :
  total_piglets = 20 →
  straws_per_piglet = 6 →
  fraction_adult_pigs = 3 / 5 →
  fraction_piglets = 3 / 5 →
  piglets_straws = total_piglets * straws_per_piglet →
  adult_pigs_straws = piglets_straws →
  total_straws = piglets_straws + adult_pigs_straws →
  (fraction_adult_pigs + fraction_piglets) * initial_straws = total_straws →
  initial_straws = 200 := 
by 
  sorry

end NUMINAMATH_GPT_troy_initial_straws_l2165_216575


namespace NUMINAMATH_GPT_stratified_sampling_sum_l2165_216562

theorem stratified_sampling_sum :
  let grains := 40
  let vegetable_oils := 10
  let animal_foods := 30
  let fruits_and_vegetables := 20
  let sample_size := 20
  let total_food_types := grains + vegetable_oils + animal_foods + fruits_and_vegetables
  let sampling_fraction := sample_size / total_food_types
  let number_drawn := sampling_fraction * (vegetable_oils + fruits_and_vegetables)
  number_drawn = 6 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_sum_l2165_216562


namespace NUMINAMATH_GPT_chosen_number_is_reconstructed_l2165_216505

theorem chosen_number_is_reconstructed (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 26) :
  ∃ (a0 a1 a2 : ℤ), (a0 = 0 ∨ a0 = 1 ∨ a0 = 2) ∧ 
                     (a1 = 0 ∨ a1 = 1 ∨ a1 = 2) ∧ 
                     (a2 = 0 ∨ a2 = 1 ∨ a2 = 2) ∧ 
                     n = a0 * 3^0 + a1 * 3^1 + a2 * 3^2 ∧ 
                     n = (if a0 = 1 then 1 else 0) + (if a0 = 2 then 2 else 0) +
                         (if a1 = 1 then 3 else 0) + (if a1 = 2 then 6 else 0) +
                         (if a2 = 1 then 9 else 0) + (if a2 = 2 then 18 else 0) := 
sorry

end NUMINAMATH_GPT_chosen_number_is_reconstructed_l2165_216505


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l2165_216537

theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  (x ^ 2 / 4 - y ^ 2 / 16 = 1) → (y = 2 * x) ∨ (y = -2 * x) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l2165_216537


namespace NUMINAMATH_GPT_total_tickets_sold_l2165_216550

/-
Problem: Prove that the total number of tickets sold is 65 given the conditions.
Conditions:
1. Senior citizen tickets cost 10 dollars each.
2. Regular tickets cost 15 dollars each.
3. Total sales were 855 dollars.
4. 24 senior citizen tickets were sold.
-/

def senior_tickets_sold : ℕ := 24
def senior_ticket_cost : ℕ := 10
def regular_ticket_cost : ℕ := 15
def total_sales : ℕ := 855

theorem total_tickets_sold (R : ℕ) (H : total_sales = senior_tickets_sold * senior_ticket_cost + R * regular_ticket_cost) :
  senior_tickets_sold + R = 65 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2165_216550


namespace NUMINAMATH_GPT_veranda_width_l2165_216591

theorem veranda_width (w : ℝ) (h_room : 18 * 12 = 216) (h_veranda : 136 = 136) : 
  (18 + 2*w) * (12 + 2*w) = 352 → w = 2 :=
by
  sorry

end NUMINAMATH_GPT_veranda_width_l2165_216591


namespace NUMINAMATH_GPT_percentage_invalid_l2165_216561

theorem percentage_invalid (total_votes valid_votes_A : ℕ) (percent_A : ℝ) (total_valid_votes : ℝ) (percent_invalid : ℝ) :
  total_votes = 560000 →
  valid_votes_A = 333200 →
  percent_A = 0.70 →
  (1 - percent_invalid / 100) * total_votes = total_valid_votes →
  percent_A * total_valid_votes = valid_votes_A →
  percent_invalid = 15 :=
by
  intros h_total_votes h_valid_votes_A h_percent_A h_total_valid_votes h_valid_poll_A
  sorry

end NUMINAMATH_GPT_percentage_invalid_l2165_216561


namespace NUMINAMATH_GPT_jumping_contest_l2165_216544

theorem jumping_contest (grasshopper_jump frog_jump : ℕ) (h_grasshopper : grasshopper_jump = 9) (h_frog : frog_jump = 12) : frog_jump - grasshopper_jump = 3 := by
  ----- h_grasshopper and h_frog are our conditions -----
  ----- The goal is to prove frog_jump - grasshopper_jump = 3 -----
  sorry

end NUMINAMATH_GPT_jumping_contest_l2165_216544


namespace NUMINAMATH_GPT_value_of_m_l2165_216592

theorem value_of_m (m : ℕ) : (5^m = 5 * 25^2 * 125^3) → m = 14 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l2165_216592


namespace NUMINAMATH_GPT_average_of_five_quantities_l2165_216573

theorem average_of_five_quantities (a b c d e : ℝ) 
  (h1 : (a + b + c) / 3 = 4) 
  (h2 : (d + e) / 2 = 33) : 
  ((a + b + c + d + e) / 5) = 15.6 := 
sorry

end NUMINAMATH_GPT_average_of_five_quantities_l2165_216573


namespace NUMINAMATH_GPT_david_age_l2165_216599

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_david_age_l2165_216599


namespace NUMINAMATH_GPT_range_of_a_l2165_216594

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a < 0 ∨ (1/4 < a ∧ a < 4) := 
sorry

end NUMINAMATH_GPT_range_of_a_l2165_216594


namespace NUMINAMATH_GPT_base10_to_base4_addition_l2165_216581

-- Define the base 10 numbers
def n1 : ℕ := 45
def n2 : ℕ := 28

-- Define the base 4 representations
def n1_base4 : ℕ := 2 * 4^2 + 3 * 4^1 + 1 * 4^0
def n2_base4 : ℕ := 1 * 4^2 + 3 * 4^1 + 0 * 4^0

-- The sum of the base 10 numbers
def sum_base10 : ℕ := n1 + n2

-- The expected sum in base 4
def sum_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Prove the equivalence
theorem base10_to_base4_addition :
  (n1 + n2 = n1_base4  + n2_base4) →
  (sum_base10 = sum_base4) :=
by
  sorry

end NUMINAMATH_GPT_base10_to_base4_addition_l2165_216581


namespace NUMINAMATH_GPT_factorize_expression_l2165_216590

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2165_216590


namespace NUMINAMATH_GPT_functional_eq_zero_l2165_216517

theorem functional_eq_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_functional_eq_zero_l2165_216517


namespace NUMINAMATH_GPT_distinct_integers_real_roots_l2165_216540

theorem distinct_integers_real_roots (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > b) (h5 : b > c) :
    (∃ x : ℝ, x^2 + 2 * a * x + 3 * (b + c) = 0) :=
sorry

end NUMINAMATH_GPT_distinct_integers_real_roots_l2165_216540


namespace NUMINAMATH_GPT_find_first_factor_of_lcm_l2165_216588

theorem find_first_factor_of_lcm (hcf : ℕ) (A : ℕ) (X : ℕ) (B : ℕ) (lcm_val : ℕ) 
  (h_hcf : hcf = 59)
  (h_A : A = 944)
  (h_lcm_val : lcm_val = 59 * X * 16)
  (h_A_lcm : A = lcm_val) :
  X = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_first_factor_of_lcm_l2165_216588


namespace NUMINAMATH_GPT_first_three_digits_of_quotient_are_239_l2165_216519

noncomputable def a : ℝ := 0.12345678910114748495051
noncomputable def b_lower_bound : ℝ := 0.515
noncomputable def b_upper_bound : ℝ := 0.516

theorem first_three_digits_of_quotient_are_239 (b : ℝ) (hb : b_lower_bound < b ∧ b < b_upper_bound) :
    0.239 * b < a ∧ a < 0.24 * b := 
sorry

end NUMINAMATH_GPT_first_three_digits_of_quotient_are_239_l2165_216519


namespace NUMINAMATH_GPT_dodecahedron_interior_diagonals_l2165_216548

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_GPT_dodecahedron_interior_diagonals_l2165_216548


namespace NUMINAMATH_GPT_remainder_polynomial_division_l2165_216514

theorem remainder_polynomial_division :
  ∀ (x : ℝ), (2 * x^2 - 21 * x + 55) % (x + 3) = 136 := 
sorry

end NUMINAMATH_GPT_remainder_polynomial_division_l2165_216514


namespace NUMINAMATH_GPT_expected_value_is_minus_one_half_l2165_216595

def prob_heads := 1 / 4
def prob_tails := 2 / 4
def prob_edge := 1 / 4
def win_heads := 4
def win_tails := -3
def win_edge := 0

theorem expected_value_is_minus_one_half :
  (prob_heads * win_heads + prob_tails * win_tails + prob_edge * win_edge) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_is_minus_one_half_l2165_216595


namespace NUMINAMATH_GPT_product_divisible_by_60_l2165_216534

theorem product_divisible_by_60 {a : ℤ} : 
  60 ∣ ((a^2 - 1) * a^2 * (a^2 + 1)) := 
by sorry

end NUMINAMATH_GPT_product_divisible_by_60_l2165_216534


namespace NUMINAMATH_GPT_ryan_correct_percentage_l2165_216503

theorem ryan_correct_percentage :
  let problems1 := 25
  let correct1 := 0.8 * problems1
  let problems2 := 40
  let correct2 := 0.9 * problems2
  let problems3 := 10
  let correct3 := 0.7 * problems3
  let total_problems := problems1 + problems2 + problems3
  let total_correct := correct1 + correct2 + correct3
  (total_correct / total_problems) = 0.84 :=
by 
  sorry

end NUMINAMATH_GPT_ryan_correct_percentage_l2165_216503


namespace NUMINAMATH_GPT_number_of_pencils_l2165_216576

theorem number_of_pencils (P L : ℕ) (h1 : (P : ℚ) / L = 5 / 6) (h2 : L = P + 6) : L = 36 :=
sorry

end NUMINAMATH_GPT_number_of_pencils_l2165_216576


namespace NUMINAMATH_GPT_sum_q_p_eq_zero_l2165_216556

def p (x : Int) : Int := x^2 - 4

def q (x : Int) : Int := 
  if x ≥ 0 then -x
  else x

def q_p (x : Int) : Int := q (p x)

#eval List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0

theorem sum_q_p_eq_zero :
  List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0 :=
sorry

end NUMINAMATH_GPT_sum_q_p_eq_zero_l2165_216556


namespace NUMINAMATH_GPT_generate_sequence_next_three_members_l2165_216568

-- Define the function that generates the sequence
def f (n : ℕ) : ℕ := 2 * (n + 1) ^ 2 * (n + 2) ^ 2

-- Define the predicate that checks if a number can be expressed as the sum of squares of two positive integers
def is_sum_of_squares_of_two_positives (k : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = k

-- The problem statement to prove the equivalence
theorem generate_sequence_next_three_members :
  is_sum_of_squares_of_two_positives (f 1) ∧
  is_sum_of_squares_of_two_positives (f 2) ∧
  is_sum_of_squares_of_two_positives (f 3) ∧
  is_sum_of_squares_of_two_positives (f 4) ∧
  is_sum_of_squares_of_two_positives (f 5) ∧
  is_sum_of_squares_of_two_positives (f 6) ∧
  f 1 = 72 ∧
  f 2 = 288 ∧
  f 3 = 800 ∧
  f 4 = 1800 ∧
  f 5 = 3528 ∧
  f 6 = 6272 :=
sorry

end NUMINAMATH_GPT_generate_sequence_next_three_members_l2165_216568


namespace NUMINAMATH_GPT_romeo_total_profit_is_55_l2165_216508

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end NUMINAMATH_GPT_romeo_total_profit_is_55_l2165_216508


namespace NUMINAMATH_GPT_white_balls_probability_l2165_216570

noncomputable def probability_all_white (total_balls white_balls draw_count : ℕ) : ℚ :=
  if h : total_balls >= draw_count ∧ white_balls >= draw_count then
    (Nat.choose white_balls draw_count : ℚ) / (Nat.choose total_balls draw_count : ℚ)
  else
    0

theorem white_balls_probability :
  probability_all_white 11 5 5 = 1 / 462 :=
by
  sorry

end NUMINAMATH_GPT_white_balls_probability_l2165_216570


namespace NUMINAMATH_GPT_bob_distance_when_meet_l2165_216507

theorem bob_distance_when_meet (total_distance : ℕ) (yolanda_speed : ℕ) (bob_speed : ℕ) 
    (yolanda_additional_distance : ℕ) (t : ℕ) :
    total_distance = 31 ∧ yolanda_speed = 3 ∧ bob_speed = 4 ∧ yolanda_additional_distance = 3 
    ∧ 7 * t = 28 → 4 * t = 16 := by
    sorry

end NUMINAMATH_GPT_bob_distance_when_meet_l2165_216507


namespace NUMINAMATH_GPT_triangle_is_isosceles_right_triangle_l2165_216597

theorem triangle_is_isosceles_right_triangle
  (a b c : ℝ)
  (h1 : (a - b)^2 + (Real.sqrt (2 * a - b - 3)) + (abs (c - 3 * Real.sqrt 2)) = 0) :
  (a = 3) ∧ (b = 3) ∧ (c = 3 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_right_triangle_l2165_216597


namespace NUMINAMATH_GPT_clean_room_to_homework_ratio_l2165_216585

-- Define the conditions
def timeHomework : ℕ := 30
def timeWalkDog : ℕ := timeHomework + 5
def timeTrash : ℕ := timeHomework / 6
def totalTimeAvailable : ℕ := 120
def remainingTime : ℕ := 35

-- Definition to calculate total time spent on other tasks
def totalTimeOnOtherTasks : ℕ := timeHomework + timeWalkDog + timeTrash

-- Definition to calculate the time to clean the room
def timeCleanRoom : ℕ := totalTimeAvailable - remainingTime - totalTimeOnOtherTasks

-- The theorem to prove the ratio
theorem clean_room_to_homework_ratio : (timeCleanRoom : ℚ) / (timeHomework : ℚ) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_clean_room_to_homework_ratio_l2165_216585


namespace NUMINAMATH_GPT_rhombus_area_8_cm2_l2165_216515

open Real

noncomputable def rhombus_area (side : ℝ) (angle : ℝ) : ℝ :=
  (side * side * sin angle) / 2 * 2

theorem rhombus_area_8_cm2 (side : ℝ) (angle : ℝ) (h1 : side = 4) (h2 : angle = π / 4) : rhombus_area side angle = 8 :=
by
  -- Definitions and calculations are omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_GPT_rhombus_area_8_cm2_l2165_216515


namespace NUMINAMATH_GPT_m_zero_sufficient_but_not_necessary_l2165_216532

-- Define the sequence a_n
variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the condition for equal difference of squares sequence
def equal_diff_of_squares_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, (a (n+1))^2 - (a n)^2 = d

-- Define the sequence b_n as an arithmetic sequence with common difference m
variable (b : ℕ → ℝ)
variable (m : ℝ)

def arithmetic_sequence (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n+1) - b n = m

-- Prove "m = 0" is a sufficient but not necessary condition for {b_n} to be an equal difference of squares sequence
theorem m_zero_sufficient_but_not_necessary (a b : ℕ → ℝ) (d m : ℝ) :
  equal_diff_of_squares_sequence a d → arithmetic_sequence b m → (m = 0 → equal_diff_of_squares_sequence b d) ∧ (¬(m ≠ 0) → equal_diff_of_squares_sequence b d) :=
sorry


end NUMINAMATH_GPT_m_zero_sufficient_but_not_necessary_l2165_216532


namespace NUMINAMATH_GPT_length_of_BC_l2165_216580

theorem length_of_BC (a : ℝ) (b_x b_y c_x c_y area : ℝ) 
  (h1 : b_y = b_x ^ 2)
  (h2 : c_y = c_x ^ 2)
  (h3 : b_y = c_y)
  (h4 : area = 64) :
  c_x - b_x = 8 := by
sorry

end NUMINAMATH_GPT_length_of_BC_l2165_216580


namespace NUMINAMATH_GPT_cubic_increasing_l2165_216553

-- The definition of an increasing function
def increasing_function (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- The function y = x^3
def cubic_function (x : ℝ) : ℝ := x^3

-- The statement we want to prove
theorem cubic_increasing : increasing_function cubic_function :=
sorry

end NUMINAMATH_GPT_cubic_increasing_l2165_216553
