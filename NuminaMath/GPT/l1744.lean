import Mathlib

namespace NUMINAMATH_GPT_find_opposite_of_neg_half_l1744_174411

-- Define the given number
def given_num : ℚ := -1/2

-- Define what it means to find the opposite of a number
def opposite (x : ℚ) : ℚ := -x

-- State the theorem
theorem find_opposite_of_neg_half : opposite given_num = 1/2 :=
by
  -- Proof is omitted for now
  sorry

end NUMINAMATH_GPT_find_opposite_of_neg_half_l1744_174411


namespace NUMINAMATH_GPT_add_fractions_l1744_174493

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end NUMINAMATH_GPT_add_fractions_l1744_174493


namespace NUMINAMATH_GPT_SomuAge_l1744_174413

theorem SomuAge (F S : ℕ) (h1 : S = F / 3) (h2 : S - 8 = (F - 8) / 5) : S = 16 :=
by 
  sorry

end NUMINAMATH_GPT_SomuAge_l1744_174413


namespace NUMINAMATH_GPT_geometric_sequence_S6_l1744_174421

-- Assume we have a geometric sequence {a_n} and the sum of the first n terms is denoted as S_n
variable (S : ℕ → ℝ)

-- Conditions given in the problem
axiom S2_eq : S 2 = 2
axiom S4_eq : S 4 = 8

-- The goal is to find the value of S 6
theorem geometric_sequence_S6 : S 6 = 26 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_S6_l1744_174421


namespace NUMINAMATH_GPT_geom_seq_a5_l1744_174477

noncomputable def S3 (a1 q : ℚ) : ℚ := a1 + a1 * q^2
noncomputable def a (a1 q : ℚ) (n : ℕ) : ℚ := a1 * q^(n - 1)

theorem geom_seq_a5 (a1 q : ℚ) (hS3 : S3 a1 q = 5 * a1) (ha7 : a a1 q 7 = 2) :
  a a1 q 5 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_a5_l1744_174477


namespace NUMINAMATH_GPT_henry_wins_l1744_174420

-- Definitions of conditions
def total_games : ℕ := 14
def losses : ℕ := 2
def draws : ℕ := 10

-- Statement of the theorem
theorem henry_wins : (total_games - losses - draws) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_henry_wins_l1744_174420


namespace NUMINAMATH_GPT_winner_lifted_weight_l1744_174496

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end NUMINAMATH_GPT_winner_lifted_weight_l1744_174496


namespace NUMINAMATH_GPT_rectangle_dimensions_l1744_174429

theorem rectangle_dimensions (l w : ℝ) : 
  (∃ x : ℝ, x = l - 3 ∧ x = w - 2 ∧ x^2 = (1 / 2) * l * w) → (l = 9 ∧ w = 8) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1744_174429


namespace NUMINAMATH_GPT_binom_subtract_l1744_174486

theorem binom_subtract :
  (Nat.choose 7 4) - 5 = 30 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_binom_subtract_l1744_174486


namespace NUMINAMATH_GPT_smallest_nonneg_int_mod_15_l1744_174497

theorem smallest_nonneg_int_mod_15 :
  ∃ x : ℕ, x + 7263 ≡ 3507 [MOD 15] ∧ ∀ y : ℕ, y + 7263 ≡ 3507 [MOD 15] → x ≤ y :=
by
  sorry

end NUMINAMATH_GPT_smallest_nonneg_int_mod_15_l1744_174497


namespace NUMINAMATH_GPT_boxes_neither_pens_nor_pencils_l1744_174495

def total_boxes : ℕ := 10
def pencil_boxes : ℕ := 6
def pen_boxes : ℕ := 3
def both_boxes : ℕ := 2

theorem boxes_neither_pens_nor_pencils : (total_boxes - (pencil_boxes + pen_boxes - both_boxes)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_boxes_neither_pens_nor_pencils_l1744_174495


namespace NUMINAMATH_GPT_quadratic_solutions_l1744_174468

theorem quadratic_solutions (x : ℝ) : (2 * x^2 + 5 * x + 3 = 0) → (x = -1 ∨ x = -3 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_solutions_l1744_174468


namespace NUMINAMATH_GPT_range_of_a_l1744_174482

open Set

theorem range_of_a (a x : ℝ) (h : x^2 - 2 * x + 1 - a^2 < 0) (h2 : 0 < x) (h3 : x < 4) :
  a < -3 ∨ a > 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1744_174482


namespace NUMINAMATH_GPT_symmetric_about_y_axis_l1744_174454

theorem symmetric_about_y_axis (m n : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (n + 1, 4))
  (symmetry : A.1 = -B.1)
  : m = 2.5 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_about_y_axis_l1744_174454


namespace NUMINAMATH_GPT_masha_can_generate_all_integers_up_to_1093_l1744_174472

theorem masha_can_generate_all_integers_up_to_1093 :
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n → n ≤ 1093 → f n ∈ {k | ∃ (a b c d e f g : ℤ), a * 1 + b * 3 + c * 9 + d * 27 + e * 81 + f * 243 + g * 729 = k}) :=
sorry

end NUMINAMATH_GPT_masha_can_generate_all_integers_up_to_1093_l1744_174472


namespace NUMINAMATH_GPT_cistern_fill_time_l1744_174430

theorem cistern_fill_time (fillA emptyB : ℕ) (hA : fillA = 8) (hB : emptyB = 12) : (24 : ℕ) = 24 :=
by
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l1744_174430


namespace NUMINAMATH_GPT_exponent_subtraction_l1744_174446

theorem exponent_subtraction (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^n = 2) : a^(m - n) = 3 := by
  sorry

end NUMINAMATH_GPT_exponent_subtraction_l1744_174446


namespace NUMINAMATH_GPT_class_6_1_students_l1744_174441

noncomputable def number_of_students : ℕ :=
  let n := 30
  n

theorem class_6_1_students (n : ℕ) (t : ℕ) (h1 : (n + 1) * t = 527) (h2 : n % 5 = 0) : n = 30 :=
  by
  sorry

end NUMINAMATH_GPT_class_6_1_students_l1744_174441


namespace NUMINAMATH_GPT_seashells_total_l1744_174416

theorem seashells_total :
  let monday := 5
  let tuesday := 7 - 3
  let wednesday := (2 * monday) / 2
  let thursday := 3 * 7
  monday + tuesday + wednesday + thursday = 35 :=
by
  sorry

end NUMINAMATH_GPT_seashells_total_l1744_174416


namespace NUMINAMATH_GPT_find_a_l1744_174494

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1744_174494


namespace NUMINAMATH_GPT_compound_O_atoms_l1744_174431

theorem compound_O_atoms (Cu_weight C_weight O_weight compound_weight : ℝ)
  (Cu_atoms : ℕ) (C_atoms : ℕ) (O_atoms : ℕ)
  (hCu : Cu_weight = 63.55)
  (hC : C_weight = 12.01)
  (hO : O_weight = 16.00)
  (h_compound_weight : compound_weight = 124)
  (h_atoms : Cu_atoms = 1 ∧ C_atoms = 1)
  : O_atoms = 3 :=
sorry

end NUMINAMATH_GPT_compound_O_atoms_l1744_174431


namespace NUMINAMATH_GPT_fruit_baskets_l1744_174444

def apple_choices := 8 -- From 0 to 7 apples
def orange_choices := 13 -- From 0 to 12 oranges

theorem fruit_baskets (a : ℕ) (o : ℕ) (ha : a = 7) (ho : o = 12) :
  (apple_choices * orange_choices) - 1 = 103 := by
  sorry

end NUMINAMATH_GPT_fruit_baskets_l1744_174444


namespace NUMINAMATH_GPT_train_distance_proof_l1744_174455

-- Definitions
def speed_train1 : ℕ := 40
def speed_train2 : ℕ := 48
def time_hours : ℕ := 8
def initial_distance : ℕ := 892

-- Function to calculate distance after given time
def distance (speed time : ℕ) : ℕ := speed * time

-- Increased/Decreased distance after time
def distance_diff : ℕ := distance speed_train2 time_hours - distance speed_train1 time_hours

-- Final distances
def final_distance_same_direction : ℕ := initial_distance + distance_diff
def final_distance_opposite_direction : ℕ := initial_distance - distance_diff

-- Proof statement
theorem train_distance_proof :
  final_distance_same_direction = 956 ∧ final_distance_opposite_direction = 828 :=
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_train_distance_proof_l1744_174455


namespace NUMINAMATH_GPT_identity_problem_l1744_174451

theorem identity_problem
  (a b : ℝ)
  (h₁ : a * b = 2)
  (h₂ : a + b = 3) :
  (a - b)^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_identity_problem_l1744_174451


namespace NUMINAMATH_GPT_solve_x2_plus_4y2_l1744_174480

theorem solve_x2_plus_4y2 (x y : ℝ) (h₁ : x + 2 * y = 6) (h₂ : x * y = -6) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_solve_x2_plus_4y2_l1744_174480


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1744_174473

theorem area_of_triangle_ABC 
  (BD DC : ℕ) 
  (h_ratio : BD / DC = 4 / 3)
  (S_BEC : ℕ) 
  (h_BEC : S_BEC = 105) :
  ∃ (S_ABC : ℕ), S_ABC = 315 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1744_174473


namespace NUMINAMATH_GPT_at_least_two_fail_l1744_174448

theorem at_least_two_fail (p q : ℝ) (n : ℕ) (h_p : p = 0.2) (h_q : q = 1 - p) :
  n ≥ 18 → (1 - ((q^n) * (1 + n * p / 4))) ≥ 0.9 :=
by
  sorry

end NUMINAMATH_GPT_at_least_two_fail_l1744_174448


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_not_necessary_condition_l1744_174423

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : x < -1) : 2 * x ^ 2 + x - 1 > 0 :=
by sorry

theorem not_necessary_condition (h2 : 2 * x ^ 2 + x - 1 > 0) : x > 1/2 ∨ x < -1 :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_not_necessary_condition_l1744_174423


namespace NUMINAMATH_GPT_find_two_numbers_l1744_174478

noncomputable def quadratic_roots (a b : ℝ) : Prop :=
  a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2

theorem find_two_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : 2 * (a * b) / (a + b) = 5 / 2) :
  quadratic_roots a b :=
by
  sorry

end NUMINAMATH_GPT_find_two_numbers_l1744_174478


namespace NUMINAMATH_GPT_main_l1744_174438

theorem main (x y : ℤ) (h1 : abs x = 5) (h2 : abs y = 3) (h3 : x * y > 0) : 
    x - y = 2 ∨ x - y = -2 := sorry

end NUMINAMATH_GPT_main_l1744_174438


namespace NUMINAMATH_GPT_distance_from_mountains_l1744_174412

/-- Given distances and scales from the problem description -/
def distance_between_mountains_map : ℤ := 312 -- in inches
def actual_distance_between_mountains : ℤ := 136 -- in km
def scale_A : ℤ := 1 -- 1 inch represents 1 km
def scale_B : ℤ := 2 -- 1 inch represents 2 km
def distance_from_mountain_A_map : ℤ := 25 -- in inches
def distance_from_mountain_B_map : ℤ := 40 -- in inches

/-- Prove the actual distances from Ram's camp to the mountains -/
theorem distance_from_mountains (dA dB : ℤ) :
  (dA = distance_from_mountain_A_map * scale_A) ∧ 
  (dB = distance_from_mountain_B_map * scale_B) :=
by {
  sorry -- Proof placeholder
}

end NUMINAMATH_GPT_distance_from_mountains_l1744_174412


namespace NUMINAMATH_GPT_find_a5_l1744_174456

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = 2 * n * (n + 1))
  (ha : ∀ n ≥ 2, a n = S n - S (n - 1)) : 
  a 5 = 20 := 
sorry

end NUMINAMATH_GPT_find_a5_l1744_174456


namespace NUMINAMATH_GPT_lemonade_quarts_l1744_174408

theorem lemonade_quarts (total_parts water_parts lemon_juice_parts : ℕ) (total_gallons gallons_to_quarts : ℚ) 
  (h_ratio : water_parts = 4) (h_ratio_lemon : lemon_juice_parts = 1) (h_total_parts : total_parts = water_parts + lemon_juice_parts)
  (h_total_gallons : total_gallons = 1) (h_gallons_to_quarts : gallons_to_quarts = 4) :
  let volume_per_part := total_gallons / total_parts
  let volume_per_part_quarts := volume_per_part * gallons_to_quarts
  let water_volume := water_parts * volume_per_part_quarts
  water_volume = 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_quarts_l1744_174408


namespace NUMINAMATH_GPT_lori_earnings_l1744_174479

theorem lori_earnings
    (red_cars : ℕ)
    (white_cars : ℕ)
    (cost_red_car : ℕ)
    (cost_white_car : ℕ)
    (rental_time_hours : ℕ)
    (rental_time_minutes : ℕ)
    (correct_earnings : ℕ) :
    red_cars = 3 →
    white_cars = 2 →
    cost_red_car = 3 →
    cost_white_car = 2 →
    rental_time_hours = 3 →
    rental_time_minutes = rental_time_hours * 60 →
    correct_earnings = 2340 →
    (red_cars * cost_red_car + white_cars * cost_white_car) * rental_time_minutes = correct_earnings :=
by
  intros
  sorry

end NUMINAMATH_GPT_lori_earnings_l1744_174479


namespace NUMINAMATH_GPT_arithmetic_seq_sin_identity_l1744_174462

theorem arithmetic_seq_sin_identity:
  ∀ (a : ℕ → ℝ), (a 2 + a 6 = (3/2) * Real.pi) → (Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sin_identity_l1744_174462


namespace NUMINAMATH_GPT_number_of_acute_triangles_l1744_174466

def num_triangles : ℕ := 7
def right_triangles : ℕ := 2
def obtuse_triangles : ℕ := 3

theorem number_of_acute_triangles :
  num_triangles - right_triangles - obtuse_triangles = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_acute_triangles_l1744_174466


namespace NUMINAMATH_GPT_difference_in_savings_correct_l1744_174434

def S_last_year : ℝ := 45000
def saved_last_year_pct : ℝ := 0.083
def raise_pct : ℝ := 0.115
def saved_this_year_pct : ℝ := 0.056

noncomputable def saved_last_year_amount : ℝ := saved_last_year_pct * S_last_year
noncomputable def S_this_year : ℝ := S_last_year * (1 + raise_pct)
noncomputable def saved_this_year_amount : ℝ := saved_this_year_pct * S_this_year
noncomputable def difference_in_savings : ℝ := saved_last_year_amount - saved_this_year_amount

theorem difference_in_savings_correct :
  difference_in_savings = 925.20 := by
  sorry

end NUMINAMATH_GPT_difference_in_savings_correct_l1744_174434


namespace NUMINAMATH_GPT_smallest_number_diminished_by_10_l1744_174406

theorem smallest_number_diminished_by_10 (x : ℕ) (h : ∀ n, x - 10 = 24 * n) : x = 34 := 
  sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_10_l1744_174406


namespace NUMINAMATH_GPT_pudding_cost_l1744_174414

theorem pudding_cost (P : ℝ) (h1 : 75 = 5 * P + 65) : P = 2 :=
sorry

end NUMINAMATH_GPT_pudding_cost_l1744_174414


namespace NUMINAMATH_GPT_simplify_expression_l1744_174467

theorem simplify_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x / y) ^ (y - x) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1744_174467


namespace NUMINAMATH_GPT_total_selling_price_correct_l1744_174499

-- Define the cost prices of the three articles
def cost_A : ℕ := 400
def cost_B : ℕ := 600
def cost_C : ℕ := 800

-- Define the desired profit percentages for the three articles
def profit_percent_A : ℚ := 40 / 100
def profit_percent_B : ℚ := 35 / 100
def profit_percent_C : ℚ := 25 / 100

-- Define the selling prices of the three articles
def selling_price_A : ℚ := cost_A * (1 + profit_percent_A)
def selling_price_B : ℚ := cost_B * (1 + profit_percent_B)
def selling_price_C : ℚ := cost_C * (1 + profit_percent_C)

-- Define the total selling price
def total_selling_price : ℚ := selling_price_A + selling_price_B + selling_price_C

-- The proof statement
theorem total_selling_price_correct : total_selling_price = 2370 :=
sorry

end NUMINAMATH_GPT_total_selling_price_correct_l1744_174499


namespace NUMINAMATH_GPT_ab_value_l1744_174476

theorem ab_value (a b c : ℤ) (h1 : a^2 = 16) (h2 : 2 * a * b = -40) : a * b = -20 := 
sorry

end NUMINAMATH_GPT_ab_value_l1744_174476


namespace NUMINAMATH_GPT_short_video_length_l1744_174403

theorem short_video_length 
  (videos_per_day : ℕ) 
  (short_videos_factor : ℕ) 
  (weekly_total_minutes : ℕ) 
  (days_in_week : ℕ) 
  (total_videos : videos_per_day = 3)
  (one_video_longer : short_videos_factor = 6)
  (total_weekly_minutes : weekly_total_minutes = 112)
  (days_a_week : days_in_week = 7) :
  ∃ x : ℕ, (videos_per_day * (short_videos_factor + 2)) * days_in_week = weekly_total_minutes ∧ 
            x = 2 := 
by 
  sorry 

end NUMINAMATH_GPT_short_video_length_l1744_174403


namespace NUMINAMATH_GPT_work_done_by_gas_l1744_174475

def gas_constant : ℝ := 8.31 -- J/(mol·K)
def temperature_change : ℝ := 100 -- K (since 100°C increase is equivalent to 100 K in Kelvin)
def moles_of_gas : ℝ := 1 -- one mole of gas

theorem work_done_by_gas :
  (1/2) * gas_constant * temperature_change = 415.5 :=
by sorry

end NUMINAMATH_GPT_work_done_by_gas_l1744_174475


namespace NUMINAMATH_GPT_consecutive_triples_with_product_divisible_by_1001_l1744_174481

theorem consecutive_triples_with_product_divisible_by_1001 :
  ∃ (a b c : ℕ), 
    (a = 76 ∧ b = 77 ∧ c = 78) ∨ 
    (a = 77 ∧ b = 78 ∧ c = 79) ∧ 
    (a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧ 
    (b = a + 1 ∧ c = b + 1) ∧ 
    (1001 ∣ (a * b * c)) :=
by sorry

end NUMINAMATH_GPT_consecutive_triples_with_product_divisible_by_1001_l1744_174481


namespace NUMINAMATH_GPT_solve_sausage_problem_l1744_174490

def sausage_problem (x y : ℕ) (condition1 : y = x + 300) (condition2 : x = y + 500) : Prop :=
  x + y = 2 * 400

theorem solve_sausage_problem (x y : ℕ) (h1 : y = x + 300) (h2 : x = y + 500) :
  sausage_problem x y h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_solve_sausage_problem_l1744_174490


namespace NUMINAMATH_GPT_scout_troop_profit_l1744_174410

noncomputable def candy_profit (purchase_bars purchase_rate sell_bars sell_rate donation_fraction : ℕ) : ℕ :=
  let cost_price_per_bar := purchase_rate / purchase_bars
  let total_cost := purchase_bars * cost_price_per_bar
  let effective_cost := total_cost * donation_fraction
  let sell_price_per_bar := sell_rate / sell_bars
  let total_revenue := purchase_bars * sell_price_per_bar
  total_revenue - effective_cost

theorem scout_troop_profit :
  candy_profit 1200 3 4 3 1/2 = 700 := by
  sorry

end NUMINAMATH_GPT_scout_troop_profit_l1744_174410


namespace NUMINAMATH_GPT_cosine_product_identity_l1744_174460

open Real

theorem cosine_product_identity (α : ℝ) (n : ℕ) :
  (List.foldr (· * ·) 1 (List.map (λ k => cos (2^k * α)) (List.range (n + 1)))) =
  sin (2^(n + 1) * α) / (2^(n + 1) * sin α) :=
sorry

end NUMINAMATH_GPT_cosine_product_identity_l1744_174460


namespace NUMINAMATH_GPT_teams_points_l1744_174418

-- Definitions of teams and points
inductive Team
| A | B | C | D | E
deriving DecidableEq

def points : Team → ℕ
| Team.A => 6
| Team.B => 5
| Team.C => 4
| Team.D => 3
| Team.E => 2

-- Conditions
axiom no_draws_A : ∀ t : Team, t ≠ Team.A → (points Team.A ≠ points t)
axiom no_loses_B : ∀ t : Team, t ≠ Team.B → (points Team.B > points t) ∨ (points Team.B = points t)
axiom no_wins_D : ∀ t : Team, t ≠ Team.D → (points Team.D < points t)
axiom unique_scores : ∀ (t1 t2 : Team), t1 ≠ t2 → points t1 ≠ points t2

-- Theorem
theorem teams_points :
  points Team.A = 6 ∧
  points Team.B = 5 ∧
  points Team.C = 4 ∧
  points Team.D = 3 ∧
  points Team.E = 2 :=
by
  sorry

end NUMINAMATH_GPT_teams_points_l1744_174418


namespace NUMINAMATH_GPT_percentage_less_than_y_is_70_percent_less_than_z_l1744_174443

variable {x y z : ℝ}

theorem percentage_less_than (h1 : x = 1.20 * y) (h2 : x = 0.36 * z) : y = 0.3 * z :=
by
  sorry

theorem y_is_70_percent_less_than_z (h : y = 0.3 * z) : (1 - y / z) * 100 = 70 :=
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_y_is_70_percent_less_than_z_l1744_174443


namespace NUMINAMATH_GPT_olivia_correct_answers_l1744_174445

theorem olivia_correct_answers (c w : ℕ) 
  (h1 : c + w = 15) 
  (h2 : 6 * c - 3 * w = 45) : 
  c = 10 := 
  sorry

end NUMINAMATH_GPT_olivia_correct_answers_l1744_174445


namespace NUMINAMATH_GPT_quadratic_expression_l1744_174453

theorem quadratic_expression (b c : ℤ) : 
  (∀ x : ℝ, (x^2 - 20*x + 49 = (x + b)^2 + c)) → (b + c = -61) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_expression_l1744_174453


namespace NUMINAMATH_GPT_solve_equation_l1744_174419

theorem solve_equation (x : ℝ) : x * (x + 1) = 12 → (x = -4 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1744_174419


namespace NUMINAMATH_GPT_length_BA_correct_area_ABCDE_correct_l1744_174447

variables {BE CD CE CA : ℝ}
axiom BE_eq : BE = 13
axiom CD_eq : CD = 3
axiom CE_eq : CE = 10
axiom CA_eq : CA = 10

noncomputable def length_BA : ℝ := 3
noncomputable def area_ABCDE : ℝ := 4098 / 61

theorem length_BA_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  length_BA = 3 := 
by { sorry }

theorem area_ABCDE_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  area_ABCDE = 4098 / 61 := 
by { sorry }

end NUMINAMATH_GPT_length_BA_correct_area_ABCDE_correct_l1744_174447


namespace NUMINAMATH_GPT_arithmetic_progression_contains_sixth_power_l1744_174425

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_contains_sixth_power_l1744_174425


namespace NUMINAMATH_GPT_poker_cards_count_l1744_174452

theorem poker_cards_count (total_cards kept_away : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : kept_away = 7) : 
  total_cards - kept_away = 45 :=
by 
  sorry

end NUMINAMATH_GPT_poker_cards_count_l1744_174452


namespace NUMINAMATH_GPT_fraction_people_eating_pizza_l1744_174409

variable (people : ℕ) (initial_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
variable (fraction : ℚ)

theorem fraction_people_eating_pizza (h1 : people = 15)
    (h2 : initial_pizza = 50)
    (h3 : pieces_per_person = 4)
    (h4 : remaining_pizza = 14)
    (h5 : 4 * 15 * fraction = initial_pizza - remaining_pizza) :
    fraction = 3 / 5 := 
  sorry

end NUMINAMATH_GPT_fraction_people_eating_pizza_l1744_174409


namespace NUMINAMATH_GPT_social_logistics_turnover_scientific_notation_l1744_174405

noncomputable def total_social_logistics_turnover_2022 : ℝ := 347.6 * (10 ^ 12)

theorem social_logistics_turnover_scientific_notation :
  total_social_logistics_turnover_2022 = 3.476 * (10 ^ 14) :=
by
  sorry

end NUMINAMATH_GPT_social_logistics_turnover_scientific_notation_l1744_174405


namespace NUMINAMATH_GPT_find_number_l1744_174417

theorem find_number : ∃ (x : ℝ), x + 0.303 + 0.432 = 5.485 ↔ x = 4.750 := 
sorry

end NUMINAMATH_GPT_find_number_l1744_174417


namespace NUMINAMATH_GPT_first_shaded_square_each_column_l1744_174450

/-- A rectangular board with 10 columns, numbered starting from 
    1 to the nth square left-to-right and top-to-bottom. The student shades squares 
    that are perfect squares. Prove that the first shaded square ensuring there's at least 
    one shaded square in each of the 10 columns is 400. -/
theorem first_shaded_square_each_column : 
  (∃ n, (∀ k, 1 ≤ k ∧ k ≤ 10 → ∃ m, m^2 ≡ k [MOD 10] ∧ m^2 ≤ n) ∧ n = 400) :=
sorry

end NUMINAMATH_GPT_first_shaded_square_each_column_l1744_174450


namespace NUMINAMATH_GPT_length_of_each_reel_l1744_174437

theorem length_of_each_reel
  (reels : ℕ)
  (sections : ℕ)
  (length_per_section : ℕ)
  (total_sections : ℕ)
  (h1 : reels = 3)
  (h2 : length_per_section = 10)
  (h3 : total_sections = 30)
  : (total_sections * length_per_section) / reels = 100 := 
by
  sorry

end NUMINAMATH_GPT_length_of_each_reel_l1744_174437


namespace NUMINAMATH_GPT_divisible_by_120_l1744_174436

theorem divisible_by_120 (n : ℤ) : 120 ∣ (n ^ 6 + 2 * n ^ 5 - n ^ 2 - 2 * n) :=
by sorry

end NUMINAMATH_GPT_divisible_by_120_l1744_174436


namespace NUMINAMATH_GPT_students_neither_l1744_174489

-- Define the given conditions
def total_students : Nat := 460
def football_players : Nat := 325
def cricket_players : Nat := 175
def both_players : Nat := 90

-- Define the Lean statement for the proof problem
theorem students_neither (total_students football_players cricket_players both_players : Nat) (h1 : total_students = 460)
  (h2 : football_players = 325) (h3 : cricket_players = 175) (h4 : both_players = 90) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end NUMINAMATH_GPT_students_neither_l1744_174489


namespace NUMINAMATH_GPT_find_a_to_make_f_odd_l1744_174439

noncomputable def f (a : ℝ) (x : ℝ): ℝ := x^3 * (Real.log (Real.exp x + 1) + a * x)

theorem find_a_to_make_f_odd :
  (∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x) ↔ a = -1/2 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_to_make_f_odd_l1744_174439


namespace NUMINAMATH_GPT_cordelia_bleach_time_l1744_174469

theorem cordelia_bleach_time (B D : ℕ) (h1 : B + D = 9) (h2 : D = 2 * B) : B = 3 :=
by
  sorry

end NUMINAMATH_GPT_cordelia_bleach_time_l1744_174469


namespace NUMINAMATH_GPT_max_discount_benefit_l1744_174442

theorem max_discount_benefit {S X : ℕ} (P : ℕ → Prop) :
  S = 1000 →
  X = 99 →
  (∀ s1 s2 s3 s4 : ℕ, s1 ≥ s2 ∧ s2 ≥ s3 ∧ s3 ≥ s4 ∧ s4 ≥ X ∧ s1 + s2 + s3 + s4 = S →
  ∃ N : ℕ, P N ∧ N = 504) := 
by
  intros hS hX
  sorry

end NUMINAMATH_GPT_max_discount_benefit_l1744_174442


namespace NUMINAMATH_GPT_part_I_part_II_l1744_174470

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem part_I (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 0) →
  -1 < a ∧ a ≤ 11/5 :=
sorry

noncomputable def g (x a : ℝ) : ℝ := 
  if abs x ≥ 1 then 2 * x^2 - 2 * a * x + a + 1 
  else -2 * a * x + a + 3

theorem part_II (a : ℝ) :
  (∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 3 ∧ g x1 a = 0 ∧ g x2 a = 0) →
  1 + Real.sqrt 3 < a ∧ a ≤ 19/5 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1744_174470


namespace NUMINAMATH_GPT_find_smaller_number_l1744_174487

theorem find_smaller_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1744_174487


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l1744_174400

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

theorem number_of_zeros_of_f : ∃! x : ℝ, 0 < x ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l1744_174400


namespace NUMINAMATH_GPT_original_employees_229_l1744_174483

noncomputable def original_number_of_employees (reduced_employees : ℕ) (reduction_percentage : ℝ) : ℝ := 
  reduced_employees / (1 - reduction_percentage)

theorem original_employees_229 : original_number_of_employees 195 0.15 = 229 := 
by
  sorry

end NUMINAMATH_GPT_original_employees_229_l1744_174483


namespace NUMINAMATH_GPT_variables_and_unknowns_l1744_174459

theorem variables_and_unknowns (f_1 f_2: ℝ → ℝ → ℝ) (f: ℝ → ℝ → ℝ) :
  (∀ x y, f_1 x y = 0 ∧ f_2 x y = 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (∀ x y, f x y = 0 → (∃ a b, x = a ∧ y = b)) :=
by sorry

end NUMINAMATH_GPT_variables_and_unknowns_l1744_174459


namespace NUMINAMATH_GPT_range_of_a_l1744_174491

/-- Definitions for propositions p and q --/
def p (a : ℝ) : Prop := a > 0 ∧ a < 1
def q (a : ℝ) : Prop := (2 * a - 3) ^ 2 - 4 > 0

/-- Theorem stating the range of possible values for a given conditions --/
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ¬(p a) ∧ ¬(q a) = false) (h4 : p a ∨ q a) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (a ≥ 5 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1744_174491


namespace NUMINAMATH_GPT_value_of_K_l1744_174433

theorem value_of_K (K: ℕ) : 4^5 * 2^3 = 2^K → K = 13 := by
  sorry

end NUMINAMATH_GPT_value_of_K_l1744_174433


namespace NUMINAMATH_GPT_parametric_graph_right_half_circle_l1744_174471

theorem parametric_graph_right_half_circle (θ : ℝ) (x y : ℝ) (hx : x = 3 * Real.cos θ) (hy : y = 3 * Real.sin θ) (hθ : -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2) :
  x^2 + y^2 = 9 ∧ x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_parametric_graph_right_half_circle_l1744_174471


namespace NUMINAMATH_GPT_union_of_sets_l1744_174401

open Set

theorem union_of_sets (A B : Set ℝ) (hA : A = {x | -2 < x ∧ x < 1}) (hB : B = {x | 0 < x ∧ x < 2}) :
  A ∪ B = {x | -2 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_union_of_sets_l1744_174401


namespace NUMINAMATH_GPT_particle_path_count_l1744_174498

def lattice_path_count (n : ℕ) : ℕ :=
sorry -- Placeholder for the actual combinatorial function

theorem particle_path_count : lattice_path_count 7 = sorry :=
sorry -- Placeholder for the actual count

end NUMINAMATH_GPT_particle_path_count_l1744_174498


namespace NUMINAMATH_GPT_greatest_GCD_of_product_7200_l1744_174426

theorem greatest_GCD_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ ∀ d, (d ∣ a ∧ d ∣ b) → d ≤ 60 :=
by
  sorry

end NUMINAMATH_GPT_greatest_GCD_of_product_7200_l1744_174426


namespace NUMINAMATH_GPT_circle_equation_equivalence_l1744_174424

theorem circle_equation_equivalence 
    (x y : ℝ) : 
    x^2 + y^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 :=
sorry

end NUMINAMATH_GPT_circle_equation_equivalence_l1744_174424


namespace NUMINAMATH_GPT_legos_in_box_at_end_l1744_174435

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end NUMINAMATH_GPT_legos_in_box_at_end_l1744_174435


namespace NUMINAMATH_GPT_count_positive_integers_satisfying_properties_l1744_174449

theorem count_positive_integers_satisfying_properties :
  (∃ n : ℕ, ∀ N < 2007,
    (N % 2 = 1) ∧
    (N % 3 = 2) ∧
    (N % 4 = 3) ∧
    (N % 5 = 4) ∧
    (N % 6 = 5) → n = 33) :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_satisfying_properties_l1744_174449


namespace NUMINAMATH_GPT_find_integer_a_l1744_174422

theorem find_integer_a (a : ℤ) : (∃ x : ℕ, a * x = 3) ↔ a = 1 ∨ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_a_l1744_174422


namespace NUMINAMATH_GPT_number_of_rows_l1744_174461

theorem number_of_rows (r : ℕ) (h1 : ∀ bus : ℕ, bus * (4 * r) = 240) : r = 10 :=
sorry

end NUMINAMATH_GPT_number_of_rows_l1744_174461


namespace NUMINAMATH_GPT_b_sequence_is_constant_l1744_174407

noncomputable def b_sequence_formula (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → ∃ d q : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ (∀ n : ℕ, b (n + 1) = b n * q)) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = b n) ∧
  (∀ n : ℕ, n > 0 → b n = 1)

theorem b_sequence_is_constant (a b : ℕ → ℝ) (h : b_sequence_formula a b) : ∀ n : ℕ, n > 0 → b n = 1 :=
  by
    sorry

end NUMINAMATH_GPT_b_sequence_is_constant_l1744_174407


namespace NUMINAMATH_GPT_tangent_parallel_points_l1744_174427

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∃ (x0 y0 : ℝ), (curve x0 = y0) ∧ 
                 (deriv curve x0 = 4) ∧
                 ((x0 = 1 ∧ y0 = 0) ∨ (x0 = -1 ∧ y0 = -4)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_points_l1744_174427


namespace NUMINAMATH_GPT_equal_sum_squares_l1744_174458

open BigOperators

-- Definitions
def n := 10

-- Assuming x and y to be arrays that hold the number of victories and losses for each player respectively.
variables {x y : Fin n → ℝ}

-- Conditions
axiom pair_meet_once : ∀ i : Fin n, x i + y i = (n - 1)

-- Theorem to be proved
theorem equal_sum_squares : ∑ i : Fin n, x i ^ 2 = ∑ i : Fin n, y i ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_equal_sum_squares_l1744_174458


namespace NUMINAMATH_GPT_exponent_fraction_simplification_l1744_174484

theorem exponent_fraction_simplification : 
  (2 ^ 2016 + 2 ^ 2014) / (2 ^ 2016 - 2 ^ 2014) = 5 / 3 := 
by {
  -- proof steps would go here
  sorry
}

end NUMINAMATH_GPT_exponent_fraction_simplification_l1744_174484


namespace NUMINAMATH_GPT_remaining_bottle_caps_l1744_174440

-- Definitions based on conditions
def initial_bottle_caps : ℕ := 65
def eaten_bottle_caps : ℕ := 4

-- Theorem
theorem remaining_bottle_caps : initial_bottle_caps - eaten_bottle_caps = 61 :=
by
  sorry

end NUMINAMATH_GPT_remaining_bottle_caps_l1744_174440


namespace NUMINAMATH_GPT_slopes_and_angles_l1744_174457

theorem slopes_and_angles (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : θ₁ = 3 * θ₂)
  (h2 : m = 5 * n)
  (h3 : m = Real.tan θ₁)
  (h4 : n = Real.tan θ₂)
  (h5 : m ≠ 0) :
  m * n = 5 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_slopes_and_angles_l1744_174457


namespace NUMINAMATH_GPT_Mike_profit_l1744_174492

def total_cost (acres : ℕ) (cost_per_acre : ℕ) : ℕ :=
  acres * cost_per_acre

def revenue (acres_sold : ℕ) (price_per_acre : ℕ) : ℕ :=
  acres_sold * price_per_acre

def profit (revenue : ℕ) (cost : ℕ) : ℕ :=
  revenue - cost

theorem Mike_profit :
  let acres := 200
  let cost_per_acre := 70
  let acres_sold := acres / 2
  let price_per_acre := 200
  let cost := total_cost acres cost_per_acre
  let rev := revenue acres_sold price_per_acre
  profit rev cost = 6000 :=
by
  sorry

end NUMINAMATH_GPT_Mike_profit_l1744_174492


namespace NUMINAMATH_GPT_number_of_people_l1744_174485

theorem number_of_people (x : ℕ) (H : x * (x - 1) = 72) : x = 9 :=
sorry

end NUMINAMATH_GPT_number_of_people_l1744_174485


namespace NUMINAMATH_GPT_least_sum_four_primes_gt_10_l1744_174464

theorem least_sum_four_primes_gt_10 : 
  ∃ (p1 p2 p3 p4 : ℕ), 
    p1 > 10 ∧ p2 > 10 ∧ p3 > 10 ∧ p4 > 10 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1 + p2 + p3 + p4 = 60 ∧
    ∀ (q1 q2 q3 q4 : ℕ), 
      q1 > 10 ∧ q2 > 10 ∧ q3 > 10 ∧ q4 > 10 ∧ 
      Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ Nat.Prime q4 ∧
      q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 →
      q1 + q2 + q3 + q4 ≥ 60 :=
by
  sorry

end NUMINAMATH_GPT_least_sum_four_primes_gt_10_l1744_174464


namespace NUMINAMATH_GPT_triangle_area_l1744_174465

theorem triangle_area (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2 : ℚ) * (a * b) = 84 := 
by
  -- Sorry is used as we are only providing the statement, not the full proof.
  sorry

end NUMINAMATH_GPT_triangle_area_l1744_174465


namespace NUMINAMATH_GPT_age_difference_l1744_174488

theorem age_difference (P M Mo : ℕ)
  (h1 : P = 3 * M / 5)
  (h2 : Mo = 5 * M / 3)
  (h3 : P + M + Mo = 196) :
  Mo - P = 64 := 
sorry

end NUMINAMATH_GPT_age_difference_l1744_174488


namespace NUMINAMATH_GPT_probability_blue_or_green_l1744_174463

def faces : Type := {faces : ℕ // faces = 6}
noncomputable def blue_faces : ℕ := 3
noncomputable def red_faces : ℕ := 2
noncomputable def green_faces : ℕ := 1

theorem probability_blue_or_green :
  (blue_faces + green_faces) / 6 = (2 / 3) := by
  sorry

end NUMINAMATH_GPT_probability_blue_or_green_l1744_174463


namespace NUMINAMATH_GPT_exists_odd_k_l_m_l1744_174428

def odd_nat (n : ℕ) : Prop := n % 2 = 1

theorem exists_odd_k_l_m : 
  ∃ (k l m : ℕ), 
  odd_nat k ∧ odd_nat l ∧ odd_nat m ∧ 
  (k ≠ 0) ∧ (l ≠ 0) ∧ (m ≠ 0) ∧ 
  (1991 * (l * m + k * m + k * l) = k * l * m) :=
by
  sorry

end NUMINAMATH_GPT_exists_odd_k_l_m_l1744_174428


namespace NUMINAMATH_GPT_point_on_x_axis_l1744_174402

theorem point_on_x_axis (m : ℝ) (h : m - 2 = 0) :
  (m + 3, m - 2) = (5, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l1744_174402


namespace NUMINAMATH_GPT_find_g8_l1744_174404

variable (g : ℝ → ℝ)

theorem find_g8 (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) (h2 : g 7 = 8) : g 8 = 64 / 7 :=
sorry

end NUMINAMATH_GPT_find_g8_l1744_174404


namespace NUMINAMATH_GPT_cryptarithm_solution_exists_l1744_174474

theorem cryptarithm_solution_exists :
  ∃ (L E S O : ℕ), L ≠ E ∧ L ≠ S ∧ L ≠ O ∧ E ≠ S ∧ E ≠ O ∧ S ≠ O ∧
  (L < 10) ∧ (E < 10) ∧ (S < 10) ∧ (O < 10) ∧
  (1000 * O + 100 * S + 10 * E + L) +
  (100 * S + 10 * E + L) +
  (10 * E + L) +
  L = 10034 ∧
  ((L = 6 ∧ E = 7 ∧ S = 4 ∧ O = 9) ∨
   (L = 6 ∧ E = 7 ∧ S = 9 ∧ O = 8)) :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_cryptarithm_solution_exists_l1744_174474


namespace NUMINAMATH_GPT_sequence_term_l1744_174415

open Int

-- Define the sequence {S_n} as stated in the problem
def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

-- Define the sequence {a_n} as the finite difference of {S_n}
def a (n : ℕ) : ℤ := if n = 1 then -1 else S n - S (n - 1)

-- The theorem statement
theorem sequence_term (n : ℕ) (hn : n > 0) : a n = 4 * n - 5 :=
by sorry

end NUMINAMATH_GPT_sequence_term_l1744_174415


namespace NUMINAMATH_GPT_find_f_7_l1744_174432

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 4) = f x
axiom piecewise_function (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : f x = 2 * x^3

theorem find_f_7 : f 7 = -2 := by
  sorry

end NUMINAMATH_GPT_find_f_7_l1744_174432
