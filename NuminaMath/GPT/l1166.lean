import Mathlib

namespace part1_part2_l1166_116693

noncomputable def determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Lean statement for Question (1)
theorem part1 :
  determinant 2022 2023 2021 2022 = 1 :=
by sorry

-- Lean statement for Question (2)
theorem part2 (m : ℤ) :
  determinant (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 :=
by sorry

end part1_part2_l1166_116693


namespace sum_of_factors_of_30_l1166_116665

/--
Given the positive integer factors of 30, prove that their sum is 72.
-/
theorem sum_of_factors_of_30 : 
  (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 := 
by 
  sorry

end sum_of_factors_of_30_l1166_116665


namespace basil_plants_count_l1166_116679

-- Define the number of basil plants and the number of oregano plants
variables (B O : ℕ)

-- Define the conditions
def condition1 : Prop := O = 2 * B + 2
def condition2 : Prop := B + O = 17

-- The proof statement
theorem basil_plants_count (h1 : condition1 B O) (h2 : condition2 B O) : B = 5 := by
  sorry

end basil_plants_count_l1166_116679


namespace vote_proportion_inequality_l1166_116649

theorem vote_proportion_inequality
  (a b k : ℕ)
  (hb_odd : b % 2 = 1)
  (hb_min : 3 ≤ b)
  (vote_same : ∀ (i j : ℕ) (hi hj : i ≠ j) (votes : ℕ → ℕ), ∃ (k_max : ℕ), ∀ (cont : ℕ), votes cont ≤ k_max) :
  (k : ℚ) / a ≥ (b - 1) / (2 * b) := sorry

end vote_proportion_inequality_l1166_116649


namespace smaller_number_in_ratio_l1166_116625

noncomputable def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem smaller_number_in_ratio (x : ℕ) (a b : ℕ) (h1 : a = 4 * x) (h2 : b = 5 * x) (h3 : LCM a b = 180) : a = 36 := 
by
  sorry

end smaller_number_in_ratio_l1166_116625


namespace min_value_of_expression_l1166_116666

/-- 
Given α and β are the two real roots of the quadratic equation x^2 - 2a * x + a + 6 = 0,
prove that the minimum value of (α - 1)^2 + (β - 1)^2 is 8.
-/
theorem min_value_of_expression (a α β : ℝ) (h1 : α ^ 2 - 2 * a * α + a + 6 = 0) (h2 : β ^ 2 - 2 * a * β + a + 6 = 0) :
  (α - 1)^2 + (β - 1)^2 ≥ 8 := 
sorry

end min_value_of_expression_l1166_116666


namespace max_colors_for_valid_coloring_l1166_116696

-- Define the 4x4 grid as a type synonym for a set of cells
def Grid4x4 := Fin 4 × Fin 4

-- Condition: Define a valid coloring function for a 4x4 grid
def valid_coloring (colors : ℕ) (f : Grid4x4 → Fin colors) : Prop :=
  ∀ i j : Fin 3, ∃ c : Fin colors, (f (i, j) = c ∨ f (i+1, j) = c) ∧ (f (i+1, j) = c ∨ f (i, j+1) = c)

-- The main theorem to prove
theorem max_colors_for_valid_coloring : 
  ∃ (colors : ℕ), colors = 11 ∧ ∀ f : Grid4x4 → Fin colors, valid_coloring colors f :=
sorry

end max_colors_for_valid_coloring_l1166_116696


namespace Sally_out_of_pocket_payment_l1166_116636

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end Sally_out_of_pocket_payment_l1166_116636


namespace contrapositive_of_square_root_l1166_116697

theorem contrapositive_of_square_root (a b : ℝ) :
  (a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b) ↔ (a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b → a^2 ≥ b) := 
sorry

end contrapositive_of_square_root_l1166_116697


namespace probability_of_F_l1166_116608

theorem probability_of_F (P : String → ℚ) (hD : P "D" = 1/4) (hE : P "E" = 1/3) (hG : P "G" = 1/6) (total : P "D" + P "E" + P "F" + P "G" = 1) :
  P "F" = 1/4 :=
by
  sorry

end probability_of_F_l1166_116608


namespace reflected_ray_equation_l1166_116673

-- Define the initial point
def point_of_emanation : (ℝ × ℝ) := (-1, 3)

-- Define the point after reflection which the ray passes through
def point_after_reflection : (ℝ × ℝ) := (4, 6)

-- Define the expected equation of the line in general form
def expected_line_equation (x y : ℝ) : Prop := 9 * x - 5 * y - 6 = 0

-- The theorem we need to prove
theorem reflected_ray_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) → expected_line_equation x y :=
sorry

end reflected_ray_equation_l1166_116673


namespace max_lessons_l1166_116607

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end max_lessons_l1166_116607


namespace probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l1166_116695

/-- 
There are 30 tiles in box C numbered from 1 to 30 and 30 tiles in box D numbered from 21 to 50. 
We want to prove that the probability of drawing a tile less than 20 from box C and a tile that 
is either odd or greater than 40 from box D is 19/45. 
-/
theorem probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40 :
  (19 / 30) * (2 / 3) = (19 / 45) :=
by sorry

end probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l1166_116695


namespace correct_system_of_equations_l1166_116628

theorem correct_system_of_equations (x y : ℝ) :
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔
  (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by sorry

end correct_system_of_equations_l1166_116628


namespace circle_parametric_solution_l1166_116614

theorem circle_parametric_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (hx : 4 * Real.cos θ = -2) (hy : 4 * Real.sin θ = 2 * Real.sqrt 3) :
    θ = 2 * Real.pi / 3 :=
sorry

end circle_parametric_solution_l1166_116614


namespace linda_babysitting_hours_l1166_116645

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end linda_babysitting_hours_l1166_116645


namespace total_cost_correct_l1166_116610

-- Definitions for the conditions
def num_ladders_1 : ℕ := 10
def rungs_1 : ℕ := 50
def cost_per_rung_1 : ℕ := 2

def num_ladders_2 : ℕ := 20
def rungs_2 : ℕ := 60
def cost_per_rung_2 : ℕ := 3

def num_ladders_3 : ℕ := 30
def rungs_3 : ℕ := 80
def cost_per_rung_3 : ℕ := 4

-- Total cost calculation for the client
def total_cost : ℕ :=
  (num_ladders_1 * rungs_1 * cost_per_rung_1) +
  (num_ladders_2 * rungs_2 * cost_per_rung_2) +
  (num_ladders_3 * rungs_3 * cost_per_rung_3)

-- Statement to be proved
theorem total_cost_correct : total_cost = 14200 :=
by {
  sorry
}

end total_cost_correct_l1166_116610


namespace smallest_clock_equivalent_number_l1166_116689

theorem smallest_clock_equivalent_number :
  ∃ h : ℕ, h > 4 ∧ h^2 % 24 = h % 24 ∧ h = 12 := by
  sorry

end smallest_clock_equivalent_number_l1166_116689


namespace sasha_quarters_l1166_116687

theorem sasha_quarters (h₁ : 2.10 = 0.35 * q) : q = 6 := 
sorry

end sasha_quarters_l1166_116687


namespace number_of_workers_in_each_block_is_200_l1166_116668

-- Conditions
def total_amount : ℕ := 6000
def worth_of_each_gift : ℕ := 2
def number_of_blocks : ℕ := 15

-- Question and answer to be proven
def number_of_workers_in_each_block : ℕ := total_amount / worth_of_each_gift / number_of_blocks

theorem number_of_workers_in_each_block_is_200 :
  number_of_workers_in_each_block = 200 :=
by
  -- Skip the proof with sorry
  sorry

end number_of_workers_in_each_block_is_200_l1166_116668


namespace triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l1166_116660

theorem triangle_angle_ratio_arbitrary (k1 k2 k3 : ℕ) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ (A / B = k1 / k2) ∧ (A / C = k1 / k3) :=
  sorry

theorem convex_quadrilateral_angle_ratio_not_arbitrary (k1 k2 k3 k4 : ℕ) :
  ¬(∃ (A B C D : ℝ), A + B + C + D = 360 ∧
  A < B + C + D ∧
  B < A + C + D ∧
  C < A + B + D ∧
  D < A + B + C) :=
  sorry

theorem convex_pentagon_angle_ratio_not_arbitrary (k1 k2 k3 k4 k5 : ℕ) :
  ¬(∃ (A B C D E : ℝ), A + B + C + D + E = 540 ∧
  A < (B + C + D + E) / 2 ∧
  B < (A + C + D + E) / 2 ∧
  C < (A + B + D + E) / 2 ∧
  D < (A + B + C + E) / 2 ∧
  E < (A + B + C + D) / 2) :=
  sorry

end triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l1166_116660


namespace part_one_part_two_part_three_l1166_116609

theorem part_one : 12 - (-11) - 1 = 22 := 
by
  sorry

theorem part_two : -(1 ^ 4) / ((-3) ^ 2) / (9 / 5) = -5 / 81 := 
by
  sorry

theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := 
by
  sorry

end part_one_part_two_part_three_l1166_116609


namespace speed_of_current_l1166_116674

-- Define the conditions in Lean
theorem speed_of_current (c : ℝ) (r : ℝ) 
  (hu : c - r = 12 / 6) -- upstream speed equation
  (hd : c + r = 12 / 0.75) -- downstream speed equation
  : r = 7 := 
sorry

end speed_of_current_l1166_116674


namespace compute_expression_l1166_116653

theorem compute_expression : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end compute_expression_l1166_116653


namespace leonid_painted_cells_l1166_116632

theorem leonid_painted_cells (k l : ℕ) (hkl : k * l = 74) :
  ∃ (painted_cells : ℕ), painted_cells = ((2 * k + 1) * (2 * l + 1) - 74) ∧ (painted_cells = 373 ∨ painted_cells = 301) :=
by
  sorry

end leonid_painted_cells_l1166_116632


namespace average_bracelets_per_day_l1166_116630

theorem average_bracelets_per_day
  (cost_of_bike : ℕ)
  (price_per_bracelet : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (h1 : cost_of_bike = 112)
  (h2 : price_per_bracelet = 1)
  (h3 : weeks = 2)
  (h4 : days_per_week = 7) :
  (cost_of_bike / price_per_bracelet) / (weeks * days_per_week) = 8 :=
by
  sorry

end average_bracelets_per_day_l1166_116630


namespace find_A_satisfy_3A_multiple_of_8_l1166_116643

theorem find_A_satisfy_3A_multiple_of_8 (A : ℕ) (h : 0 ≤ A ∧ A < 10) : 8 ∣ (30 + A) ↔ A = 2 := 
by
  sorry

end find_A_satisfy_3A_multiple_of_8_l1166_116643


namespace amy_points_per_treasure_l1166_116691

theorem amy_points_per_treasure (treasures_first_level treasures_second_level total_score : ℕ) (h1 : treasures_first_level = 6) (h2 : treasures_second_level = 2) (h3 : total_score = 32) :
  total_score / (treasures_first_level + treasures_second_level) = 4 := by
  sorry

end amy_points_per_treasure_l1166_116691


namespace net_profit_is_correct_l1166_116655

-- Define the purchase price, markup, and overhead percentage
def purchase_price : ℝ := 48
def markup : ℝ := 55
def overhead_percentage : ℝ := 0.30

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the net profit calculation
def net_profit : ℝ := markup - overhead_cost

-- State the theorem
theorem net_profit_is_correct : net_profit = 40.60 :=
by
  sorry

end net_profit_is_correct_l1166_116655


namespace min_contribution_proof_l1166_116624

noncomputable def min_contribution (total_contribution : ℕ) (num_people : ℕ) (max_contribution: ℕ) :=
  ∃ (min_each_person: ℕ), num_people * min_each_person ≤ total_contribution ∧ max_contribution * (num_people - 1) + min_each_person ≥ total_contribution ∧ min_each_person = 2

theorem min_contribution_proof :
  min_contribution 30 15 16 :=
sorry

end min_contribution_proof_l1166_116624


namespace perpendicular_lines_sin_2alpha_l1166_116662

theorem perpendicular_lines_sin_2alpha (α : ℝ) 
  (l1 : ∀ (x y : ℝ), x * (Real.sin α) + y - 1 = 0) 
  (l2 : ∀ (x y : ℝ), x - 3 * y * Real.cos α + 1 = 0) 
  (perp : ∀ (x1 y1 x2 y2 : ℝ), 
        (x1 * (Real.sin α) + y1 - 1 = 0) ∧ 
        (x2 - 3 * y2 * Real.cos α + 1 = 0) → 
        ((-Real.sin α) * (1 / (3 * Real.cos α)) = -1)) :
  Real.sin (2 * α) = (3/5) :=
sorry

end perpendicular_lines_sin_2alpha_l1166_116662


namespace solution_set_of_inequality_l1166_116685

theorem solution_set_of_inequality : {x : ℝ | x^2 - 2 * x ≤ 0} = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l1166_116685


namespace units_digit_47_pow_47_l1166_116612

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end units_digit_47_pow_47_l1166_116612


namespace robert_has_2_more_years_l1166_116667

theorem robert_has_2_more_years (R P T Rb M : ℕ) 
                                 (h1 : R = P + T + Rb + M)
                                 (h2 : R = 42)
                                 (h3 : P = 12)
                                 (h4 : T = 2 * Rb)
                                 (h5 : Rb = P - 4) : Rb - M = 2 := 
by 
-- skipped proof
  sorry

end robert_has_2_more_years_l1166_116667


namespace circle_tangent_to_xaxis_at_origin_l1166_116648

theorem circle_tangent_to_xaxis_at_origin (G E F : ℝ)
  (h : ∀ x y: ℝ, x^2 + y^2 + G*x + E*y + F = 0 → y = 0 ∧ x = 0 ∧ 0 < E) :
  G = 0 ∧ F = 0 ∧ E ≠ 0 :=
by
  sorry

end circle_tangent_to_xaxis_at_origin_l1166_116648


namespace number_without_daughters_l1166_116659

-- Given conditions
def Marilyn_daughters : Nat := 10
def total_women : Nat := 40
def daughters_with_daughters_women_have_each : Nat := 5

-- Helper definition representing the computation of granddaughters
def Marilyn_granddaughters : Nat := total_women - Marilyn_daughters

-- Proving the main statement
theorem number_without_daughters : 
  (Marilyn_daughters - (Marilyn_granddaughters / daughters_with_daughters_women_have_each)) + Marilyn_granddaughters = 34 := by
  sorry

end number_without_daughters_l1166_116659


namespace radius_of_circle_l1166_116684

theorem radius_of_circle
  (r : ℝ) (r_pos : r > 0)
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1^2 + y1^2 = r^2)
  (h2 : x2^2 + y2^2 = r^2)
  (h3 : x1 + y1 = 3)
  (h4 : x2 + y2 = 3)
  (h5 : x1 * x2 + y1 * y2 = -0.5 * r^2) : 
  r = 3 * Real.sqrt 2 :=
by
  sorry

end radius_of_circle_l1166_116684


namespace fraction_august_tips_l1166_116615

variable (A : ℝ) -- Define the average monthly tips A for March, April, May, June, July, and September
variable (august_tips : ℝ) -- Define the tips for August
variable (total_tips : ℝ) -- Define the total tips for all months

-- Define the conditions
def condition_average_tips : Prop := total_tips = 12 * A
def condition_august_tips : Prop := august_tips = 6 * A

-- The theorem we need to prove
theorem fraction_august_tips :
  condition_average_tips A total_tips →
  condition_august_tips A august_tips →
  (august_tips / total_tips) = (1 / 2) :=
by
  intros h_avg h_aug
  rw [condition_average_tips] at h_avg
  rw [condition_august_tips] at h_aug
  rw [h_avg, h_aug]
  simp
  sorry

end fraction_august_tips_l1166_116615


namespace vasya_no_purchase_days_l1166_116664

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l1166_116664


namespace xiao_ming_english_score_l1166_116602

theorem xiao_ming_english_score :
  let a := 92
  let b := 90
  let c := 95
  let w_a := 3
  let w_b := 3
  let w_c := 4
  let total_weight := (w_a + w_b + w_c)
  let score := (a * w_a + b * w_b + c * w_c) / total_weight
  score = 92.6 :=
by
  sorry

end xiao_ming_english_score_l1166_116602


namespace concrete_pillars_l1166_116605

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l1166_116605


namespace correct_product_l1166_116616

theorem correct_product (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)  -- a is a two-digit number
  (h2 : 0 < b)  -- b is a positive integer
  (h3 : (a % 10) * 10 + (a / 10) * b = 161)  -- Reversing the digits of a and multiplying by b yields 161
  : a * b = 224 := 
sorry

end correct_product_l1166_116616


namespace total_garbage_collected_correct_l1166_116606

def Lizzie_group_collected : ℕ := 387
def other_group_collected : ℕ := Lizzie_group_collected - 39
def total_garbage_collected : ℕ := Lizzie_group_collected + other_group_collected

theorem total_garbage_collected_correct :
  total_garbage_collected = 735 :=
sorry

end total_garbage_collected_correct_l1166_116606


namespace most_likely_sitting_people_l1166_116678

theorem most_likely_sitting_people :
  let num_people := 100
  let seats := 100
  let favorite_seats : Fin num_people → Fin seats := sorry
  -- Conditions related to people sitting behavior
  let sits_in_row (i : Fin num_people) : Prop :=
    ∀ j : Fin num_people, j < i → favorite_seats j ≠ favorite_seats i
  let num_sitting_in_row := Finset.card (Finset.filter sits_in_row (Finset.univ : Finset (Fin num_people)))
  -- Prove
  num_sitting_in_row = 10 := 
sorry

end most_likely_sitting_people_l1166_116678


namespace probability_of_two_black_balls_relationship_x_y_l1166_116629

-- Conditions
def initial_black_balls : ℕ := 3
def initial_white_balls : ℕ := 2

variable (x y : ℕ)

-- Given relationship
def total_white_balls := x + 2
def total_black_balls := y + 3
def white_ball_probability := (total_white_balls x) / (total_white_balls x + total_black_balls y + 5)

-- Proof goals
theorem probability_of_two_black_balls :
  (3 / 5) * (2 / 4) = 3 / 10 := by sorry

theorem relationship_x_y :
  white_ball_probability x y = 1 / 3 → y = 2 * x + 1 := by sorry

end probability_of_two_black_balls_relationship_x_y_l1166_116629


namespace number_of_girls_l1166_116631

theorem number_of_girls {total_children boys girls : ℕ} 
  (h_total : total_children = 60) 
  (h_boys : boys = 18) 
  (h_girls : girls = total_children - boys) : 
  girls = 42 := by 
  sorry

end number_of_girls_l1166_116631


namespace false_props_count_is_3_l1166_116604

-- Define the propositions and their inferences

noncomputable def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2
noncomputable def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)
noncomputable def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n
noncomputable def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- The main statement to be proved
theorem false_props_count_is_3 (m n : ℝ) : 
  ¬ (original_prop m n) ∧ ¬ (contrapositive m n) ∧ ¬ (inverse m n) ∧ ¬ (negation m n) →
  (3 = 3) :=
by
  sorry

end false_props_count_is_3_l1166_116604


namespace tom_total_money_l1166_116639

theorem tom_total_money :
  let initial_amount := 74
  let additional_amount := 86
  initial_amount + additional_amount = 160 :=
by
  let initial_amount := 74
  let additional_amount := 86
  show initial_amount + additional_amount = 160
  sorry

end tom_total_money_l1166_116639


namespace mandy_cinnamon_nutmeg_difference_l1166_116618

theorem mandy_cinnamon_nutmeg_difference :
  0.67 - 0.5 = 0.17 :=
by
  sorry

end mandy_cinnamon_nutmeg_difference_l1166_116618


namespace ratio_of_ages_l1166_116694

theorem ratio_of_ages (Sandy_age : ℕ) (Molly_age : ℕ)
  (h1 : Sandy_age = 56)
  (h2 : Molly_age = Sandy_age + 16) :
  (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_ages_l1166_116694


namespace solve_equation_l1166_116699

noncomputable def f (x : ℝ) := (1 / (x^2 + 17 * x + 20)) + (1 / (x^2 + 12 * x + 20)) + (1 / (x^2 - 15 * x + 20))

theorem solve_equation :
  {x : ℝ | f x = 0} = {-1, -4, -5, -20} :=
by
  sorry

end solve_equation_l1166_116699


namespace find_original_number_l1166_116690

theorem find_original_number (n : ℝ) (h : n / 2 = 9) : n = 18 :=
sorry

end find_original_number_l1166_116690


namespace part1_solution_set_part2_range_of_a_l1166_116622

-- Part 1: Prove the solution set of the inequality f(x) < 6 is (-8/3, 4/3)
theorem part1_solution_set (x : ℝ) :
  (|2 * x + 3| + |x - 1| < 6) ↔ (-8 / 3 : ℝ) < x ∧ x < 4 / 3 :=
by sorry

-- Part 2: Prove the range of values for a that makes f(x) + f(-x) ≥ 5 is (-∞, -3/2] ∪ [3/2, +∞)
theorem part2_range_of_a (a : ℝ) (x : ℝ) :
  (|2 * x + a| + |x - 1| + |-2 * x + a| + |-x - 1| ≥ 5) ↔ 
  (a ≤ -3 / 2 ∨ a ≥ 3 / 2) :=
by sorry

end part1_solution_set_part2_range_of_a_l1166_116622


namespace number_of_boys_l1166_116672

theorem number_of_boys 
  (B G : ℕ) 
  (h1 : B + G = 650) 
  (h2 : G = B + 106) :
  B = 272 :=
sorry

end number_of_boys_l1166_116672


namespace cos_double_angle_l1166_116642

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -3/5) : Real.cos (2 * α) = -7/25 :=
by
  sorry

end cos_double_angle_l1166_116642


namespace min_days_equal_shifts_l1166_116626

theorem min_days_equal_shifts (k n : ℕ) (h : 9 * k + 10 * n = 66) : k + n = 7 :=
sorry

end min_days_equal_shifts_l1166_116626


namespace sum_of_digits_ABCED_l1166_116658

theorem sum_of_digits_ABCED {A B C D E : ℕ} (hABCED : 3 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) = 111111) :
  A + B + C + D + E = 20 := 
by
  sorry

end sum_of_digits_ABCED_l1166_116658


namespace greatest_possible_percentage_of_airlines_both_services_l1166_116688

noncomputable def maxPercentageOfAirlinesWithBothServices (percentageInternet percentageSnacks : ℝ) : ℝ :=
  if percentageInternet <= percentageSnacks then percentageInternet else percentageSnacks

theorem greatest_possible_percentage_of_airlines_both_services:
  let p_internet := 0.35
  let p_snacks := 0.70
  maxPercentageOfAirlinesWithBothServices p_internet p_snacks = 0.35 :=
by
  sorry

end greatest_possible_percentage_of_airlines_both_services_l1166_116688


namespace proof_l1166_116652

-- Define proposition p as negated form: ∀ x < 1, log_3 x ≤ 0
def p : Prop := ∀ x : ℝ, x < 1 → Real.log x / Real.log 3 ≤ 0

-- Define proposition q: ∃ x_0 ∈ ℝ, x_0^2 ≥ 2^x_0
def q : Prop := ∃ x_0 : ℝ, x_0^2 ≥ Real.exp (x_0 * Real.log 2)

-- State we need to prove: p ∨ q
theorem proof : p ∨ q := sorry

end proof_l1166_116652


namespace smallest_sum_l1166_116644

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l1166_116644


namespace total_pull_ups_per_week_l1166_116638

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end total_pull_ups_per_week_l1166_116638


namespace tan_theta_cos_double_angle_minus_pi_over_3_l1166_116686

open Real

-- Given conditions
variable (θ : ℝ)
axiom sin_theta : sin θ = 3 / 5
axiom theta_in_second_quadrant : π / 2 < θ ∧ θ < π

-- Questions and answers to prove:
theorem tan_theta : tan θ = - 3 / 4 :=
sorry

theorem cos_double_angle_minus_pi_over_3 : cos (2 * θ - π / 3) = (7 - 24 * Real.sqrt 3) / 50 :=
sorry

end tan_theta_cos_double_angle_minus_pi_over_3_l1166_116686


namespace Donovan_percentage_correct_l1166_116613

-- Definitions based on conditions from part a)
def fullyCorrectAnswers : ℕ := 35
def incorrectAnswers : ℕ := 13
def partiallyCorrectAnswers : ℕ := 7
def pointPerFullAnswer : ℝ := 1
def pointPerPartialAnswer : ℝ := 0.5

-- Lean 4 statement to prove the problem mathematically
theorem Donovan_percentage_correct : 
  (fullyCorrectAnswers * pointPerFullAnswer + partiallyCorrectAnswers * pointPerPartialAnswer) / 
  (fullyCorrectAnswers + incorrectAnswers + partiallyCorrectAnswers) * 100 = 70.00 :=
by
  sorry

end Donovan_percentage_correct_l1166_116613


namespace value_of_expression_l1166_116680

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) : 4 * a + 8 * b - 2 = 4 := 
by 
  sorry

end value_of_expression_l1166_116680


namespace eccentricities_proof_l1166_116669

variable (e1 e2 m n c : ℝ)
variable (h1 : e1 = 2 * c / (m + n))
variable (h2 : e2 = 2 * c / (m - n))
variable (h3 : m ^ 2 + n ^ 2 = 4 * c ^ 2)

theorem eccentricities_proof :
  (e1 * e2) / (Real.sqrt (e1 ^ 2 + e2 ^ 2)) = (Real.sqrt 2) / 2 :=
by sorry

end eccentricities_proof_l1166_116669


namespace rain_stop_time_on_first_day_l1166_116640

-- Define the problem conditions
def raining_time_day1 (x : ℕ) : Prop :=
  let start_time := 7 * 60 -- start time in minutes
  let stop_time := start_time + x * 60 -- stop time in minutes
  stop_time = 17 * 60 -- stop at 17:00 (5:00 PM)

def total_raining_time_46_hours (x : ℕ) : Prop :=
  x + (x + 2) + 2 * (x + 2) = 46

-- Main statement
theorem rain_stop_time_on_first_day (x : ℕ) (h1 : total_raining_time_46_hours x) : raining_time_day1 x :=
  sorry

end rain_stop_time_on_first_day_l1166_116640


namespace number_of_red_balls_l1166_116677

-- Conditions
variables (w r : ℕ)
variable (ratio_condition : 4 * r = 3 * w)
variable (white_balls : w = 8)

-- Prove the number of red balls
theorem number_of_red_balls : r = 6 :=
by
  sorry

end number_of_red_balls_l1166_116677


namespace richmond_population_l1166_116656

theorem richmond_population (R V B : ℕ) (h0 : R = V + 1000) (h1 : V = 4 * B) (h2 : B = 500) : R = 3000 :=
by
  -- skipping proof
  sorry

end richmond_population_l1166_116656


namespace common_fraction_equiv_l1166_116651

noncomputable def decimal_equivalent_frac : Prop :=
  ∃ (x : ℚ), x = 413 / 990 ∧ x = 0.4 + (7/10^2 + 1/10^3) / (1 - 1/10^2)

theorem common_fraction_equiv : decimal_equivalent_frac :=
by
  sorry

end common_fraction_equiv_l1166_116651


namespace average_annual_growth_rate_l1166_116633

-- Define the conditions
def revenue_current_year : ℝ := 280
def revenue_planned_two_years : ℝ := 403.2

-- Define the growth equation
def growth_equation (x : ℝ) : Prop :=
  revenue_current_year * (1 + x)^2 = revenue_planned_two_years

-- State the theorem
theorem average_annual_growth_rate : ∃ x : ℝ, growth_equation x ∧ x = 0.2 := by
  sorry

end average_annual_growth_rate_l1166_116633


namespace ralphStartsWith_l1166_116637

def ralphEndsWith : ℕ := 15
def ralphLoses : ℕ := 59

theorem ralphStartsWith : (ralphEndsWith + ralphLoses = 74) :=
by
  sorry

end ralphStartsWith_l1166_116637


namespace arithmetic_seq_geom_eq_div_l1166_116635

noncomputable def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + n * d

theorem arithmetic_seq_geom_eq_div (a1 d : ℝ) (h1 : d ≠ 0) (h2 : a1 ≠ 0) 
    (h_geom : (a 3 a1 d) ^ 2 = (a 1 a1 d) * (a 7 a1 d)) :
    (a 2 a1 d + a 5 a1 d + a 8 a1 d) / (a 3 a1 d + a 4 a1 d) = 2 := 
by
  sorry

end arithmetic_seq_geom_eq_div_l1166_116635


namespace exists_integers_for_prime_l1166_116698

theorem exists_integers_for_prime (p : ℕ) (hp : Nat.Prime p) : 
  ∃ x y z w : ℤ, x^2 + y^2 + z^2 = w * p ∧ 0 < w ∧ w < p :=
by 
  sorry

end exists_integers_for_prime_l1166_116698


namespace arithmetic_sequence_S12_l1166_116600

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2*a + (n-1)*d) / 2

def a_n (a d n : ℕ) : ℕ :=
  a + (n-1)*d

variable (a d : ℕ)

theorem arithmetic_sequence_S12 (h : a_n a d 4 + a_n a d 9 = 10) :
  arithmetic_sequence_sum a d 12 = 60 :=
by sorry

end arithmetic_sequence_S12_l1166_116600


namespace cupcakes_per_child_l1166_116603

theorem cupcakes_per_child (total_cupcakes children : ℕ) (h1 : total_cupcakes = 96) (h2 : children = 8) : total_cupcakes / children = 12 :=
by
  sorry

end cupcakes_per_child_l1166_116603


namespace values_only_solution_l1166_116620

variables (m n : ℝ) (x a b c : ℝ)

noncomputable def equation := (x + m)^3 - (x + n)^3 = (m + n)^3

theorem values_only_solution (hm : m ≠ 0) (hn : n ≠ 0) (hne : m ≠ n)
  (hx : x = a * m + b * n + c) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end values_only_solution_l1166_116620


namespace remainder_15_plus_3y_l1166_116657

theorem remainder_15_plus_3y (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + 3 * y) % 31 = 11 :=
by
  sorry

end remainder_15_plus_3y_l1166_116657


namespace solve_eq1_solve_eq2_l1166_116650

noncomputable def eq1 (x : ℝ) : Prop := x - 2 = 4 * (x - 2)^2
noncomputable def eq2 (x : ℝ) : Prop := x * (2 * x + 1) = 8 * x - 3

theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 2 ∨ x = 9 / 4 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 1 / 2 ∨ x = 3 :=
by
  sorry

end solve_eq1_solve_eq2_l1166_116650


namespace min_chips_to_A10_l1166_116641

theorem min_chips_to_A10 (n : ℕ) (A : ℕ → ℕ) (hA1 : A 1 = n) :
  (∃ (σ : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i < 10 → (σ i = A i - 2) ∧ (σ (i + 1) = A (i + 1) + 1)) ∨ 
    (∀ i, 1 ≤ i ∧ i < 9 → (σ (i + 1) = A (i + 1) - 2) ∧ (σ (i + 2) = A (i + 2) + 1) ∧ (σ i = A i + 1)) ∧ 
    (∃ (k : ℕ), k = 10 ∧ σ k = 1)) →
  n ≥ 46 := sorry

end min_chips_to_A10_l1166_116641


namespace corrected_mean_l1166_116647

theorem corrected_mean (n : ℕ) (obs_mean : ℝ) (obs_count : ℕ) (wrong_val correct_val : ℝ) :
  obs_count = 40 →
  obs_mean = 100 →
  wrong_val = 75 →
  correct_val = 50 →
  (obs_count * obs_mean - (wrong_val - correct_val)) / obs_count = 3975 / 40 :=
by
  sorry

end corrected_mean_l1166_116647


namespace total_selling_price_l1166_116683

theorem total_selling_price (total_commissions : ℝ) (number_of_appliances : ℕ) (fixed_commission_rate_per_appliance : ℝ) (percentage_commission_rate : ℝ) :
  total_commissions = number_of_appliances * fixed_commission_rate_per_appliance + percentage_commission_rate * S →
  total_commissions = 662 →
  number_of_appliances = 6 →
  fixed_commission_rate_per_appliance = 50 →
  percentage_commission_rate = 0.10 →
  S = 3620 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_selling_price_l1166_116683


namespace problem_inequality_l1166_116621

-- Definitions and conditions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x + f (k - x)

-- The Lean proof problem
theorem problem_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  f a + (a + b) * Real.log 2 ≥ f (a + b) - f b := sorry

end problem_inequality_l1166_116621


namespace factor_theorem_solution_l1166_116681

theorem factor_theorem_solution (t : ℝ) :
  (∃ p q : ℝ, 10 * p * q = 10 * t * t + 21 * t - 10 ∧ (x - q) = (x - t)) →
  t = 2 / 5 ∨ t = -5 / 2 := by
  sorry

end factor_theorem_solution_l1166_116681


namespace total_weight_of_2_meters_l1166_116671

def tape_measure_length : ℚ := 5
def tape_measure_weight : ℚ := 29 / 8
def computer_length : ℚ := 4
def computer_weight : ℚ := 2.8

noncomputable def weight_per_meter_tape_measure : ℚ := tape_measure_weight / tape_measure_length
noncomputable def weight_per_meter_computer : ℚ := computer_weight / computer_length

noncomputable def total_weight : ℚ :=
  2 * weight_per_meter_tape_measure + 2 * weight_per_meter_computer

theorem total_weight_of_2_meters (h1 : tape_measure_length = 5)
    (h2 : tape_measure_weight = 29 / 8) 
    (h3 : computer_length = 4) 
    (h4 : computer_weight = 2.8): 
    total_weight = 57 / 20 := by 
  unfold total_weight
  sorry

end total_weight_of_2_meters_l1166_116671


namespace tangent_line_of_ellipse_l1166_116619

variable {a b x y x₀ y₀ : ℝ}

theorem tangent_line_of_ellipse
    (h1 : 0 < a)
    (h2 : a > b)
    (h3 : b > 0)
    (h4 : (x₀, y₀) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_of_ellipse_l1166_116619


namespace least_number_added_1054_l1166_116675

theorem least_number_added_1054 (x d: ℕ) (h_cond: 1054 + x = 1058) (h_div: d = 2) : 1058 % d = 0 :=
by
  sorry

end least_number_added_1054_l1166_116675


namespace largest_number_l1166_116670

def HCF (a b c d : ℕ) : Prop := d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ 
                                ∀ e, (e ∣ a ∧ e ∣ b ∧ e ∣ c) → e ≤ d
def LCM (a b c m : ℕ) : Prop := m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧ 
                                ∀ n, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem largest_number (a b c : ℕ)
  (hcf: HCF a b c 210)
  (lcm_has_factors: ∃ k1 k2 k3, k1 = 11 ∧ k2 = 17 ∧ k3 = 23 ∧
                                LCM a b c (210 * k1 * k2 * k3)) :
  max a (max b c) = 4830 := 
by
  sorry

end largest_number_l1166_116670


namespace math_dance_residents_l1166_116646

theorem math_dance_residents (p a b : ℕ) (hp : Nat.Prime p) 
    (h1 : b ≥ 1) 
    (h2 : (a + b)^2 = (p + 1) * a + b) :
    b = 1 := by
  sorry

end math_dance_residents_l1166_116646


namespace stream_speed_l1166_116617

variable (D v : ℝ)

/--
The time taken by a man to row his boat upstream is twice the time taken by him to row the same distance downstream.
If the speed of the boat in still water is 63 kmph, prove that the speed of the stream is 21 kmph.
-/
theorem stream_speed (h : D / (63 - v) = 2 * (D / (63 + v))) : v = 21 := 
sorry

end stream_speed_l1166_116617


namespace negation_of_proposition_l1166_116692

open Nat 

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n > 0 ∧ n^2 > 2^n) ↔ ∀ n : ℕ, n > 0 → n^2 ≤ 2^n :=
by
  sorry

end negation_of_proposition_l1166_116692


namespace jose_peanuts_l1166_116627

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end jose_peanuts_l1166_116627


namespace rectangle_ratio_l1166_116676

theorem rectangle_ratio (s x y : ℝ) (h1 : 4 * (x * y) + s * s = 9 * s * s) (h2 : s + 2 * y = 3 * s) (h3 : x + y = 3 * s): x / y = 2 :=
by sorry

end rectangle_ratio_l1166_116676


namespace cos_difference_l1166_116601

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1 / 2) 
  (h2 : Real.cos A + Real.cos B = 3 / 2) : 
  Real.cos (A - B) = 1 / 4 :=
by
  sorry

end cos_difference_l1166_116601


namespace matilda_father_chocolates_left_l1166_116663

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l1166_116663


namespace inequality_for_positive_integers_l1166_116611

theorem inequality_for_positive_integers 
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : 1/a + 1/b = 1)
  (n : ℕ)
  (hn : n > 0) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2^(2*n) - 2^(n + 1) :=
sorry

end inequality_for_positive_integers_l1166_116611


namespace find_original_price_l1166_116682

-- Definitions based on Conditions
def original_price (P : ℝ) : Prop :=
  let increased_price := 1.25 * P
  let final_price := increased_price * 0.75
  final_price = 187.5

theorem find_original_price (P : ℝ) (h : original_price P) : P = 200 :=
  by sorry

end find_original_price_l1166_116682


namespace intersection_points_sum_l1166_116634

theorem intersection_points_sum (x1 x2 x3 y1 y2 y3 A B : ℝ)
(h1 : y1 = x1^3 - 3 * x1 + 2)
(h2 : x1 + 6 * y1 = 6)
(h3 : y2 = x2^3 - 3 * x2 + 2)
(h4 : x2 + 6 * y2 = 6)
(h5 : y3 = x3^3 - 3 * x3 + 2)
(h6 : x3 + 6 * y3 = 6)
(hA : A = x1 + x2 + x3)
(hB : B = y1 + y2 + y3) :
A = 0 ∧ B = 3 := 
by
  sorry

end intersection_points_sum_l1166_116634


namespace red_more_than_yellow_l1166_116661

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l1166_116661


namespace six_digit_palindromes_count_l1166_116623

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l1166_116623


namespace tangent_line_condition_l1166_116654

-- statement only, no proof required
theorem tangent_line_condition {m n u v x y : ℝ}
  (hm : m > 1)
  (curve_eq : x^m + y^m = 1)
  (line_eq : u * x + v * y = 1)
  (u_v_condition : u^n + v^n = 1)
  (mn_condition : 1/m + 1/n = 1)
  : (u * x + v * y = 1) ↔ (u^n + v^n = 1 ∧ 1/m + 1/n = 1) :=
sorry

end tangent_line_condition_l1166_116654
