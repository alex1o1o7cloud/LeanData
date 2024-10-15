import Mathlib

namespace NUMINAMATH_GPT_carl_garden_area_l1553_155397

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end NUMINAMATH_GPT_carl_garden_area_l1553_155397


namespace NUMINAMATH_GPT_zeros_at_end_of_quotient_factorial_l1553_155391

def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem zeros_at_end_of_quotient_factorial :
  count_factors_of_five 2018 - count_factors_of_five 30 - count_factors_of_five 11 = 493 :=
by
  sorry

end NUMINAMATH_GPT_zeros_at_end_of_quotient_factorial_l1553_155391


namespace NUMINAMATH_GPT_evaluate_expression_l1553_155306

def diamond (a b : ℚ) : ℚ := a - (2 / b)

theorem evaluate_expression :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -(11 / 30) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1553_155306


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1553_155385

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 - 7*x + 12 < 0 ↔ 3 < x ∧ x < 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_of_inequality_l1553_155385


namespace NUMINAMATH_GPT_grapes_purchased_l1553_155328

-- Define the given conditions
def price_per_kg_grapes : ℕ := 68
def kg_mangoes : ℕ := 9
def price_per_kg_mangoes : ℕ := 48
def total_paid : ℕ := 908

-- Define the proof problem
theorem grapes_purchased : ∃ (G : ℕ), (price_per_kg_grapes * G + price_per_kg_mangoes * kg_mangoes = total_paid) ∧ (G = 7) :=
by {
  use 7,
  sorry
}

end NUMINAMATH_GPT_grapes_purchased_l1553_155328


namespace NUMINAMATH_GPT_mod_product_eq_15_l1553_155346

theorem mod_product_eq_15 :
  (15 * 24 * 14) % 25 = 15 :=
by
  sorry

end NUMINAMATH_GPT_mod_product_eq_15_l1553_155346


namespace NUMINAMATH_GPT_equation_solution_l1553_155355

theorem equation_solution (x : ℝ) : (3 : ℝ)^(x-1) = 1/9 ↔ x = -1 :=
by sorry

end NUMINAMATH_GPT_equation_solution_l1553_155355


namespace NUMINAMATH_GPT_slower_pipe_time_l1553_155301

/-
One pipe can fill a tank four times as fast as another pipe. 
If together the two pipes can fill the tank in 40 minutes, 
how long will it take for the slower pipe alone to fill the tank?
-/

theorem slower_pipe_time (t : ℕ) (h1 : ∀ t, 1/t + 4/t = 1/40) : t = 200 :=
sorry

end NUMINAMATH_GPT_slower_pipe_time_l1553_155301


namespace NUMINAMATH_GPT_original_cost_of_plants_l1553_155312

theorem original_cost_of_plants
  (discount : ℕ)
  (amount_spent : ℕ)
  (original_cost : ℕ)
  (h_discount : discount = 399)
  (h_amount_spent : amount_spent = 68)
  (h_original_cost : original_cost = discount + amount_spent) :
  original_cost = 467 :=
by
  rw [h_discount, h_amount_spent] at h_original_cost
  exact h_original_cost

end NUMINAMATH_GPT_original_cost_of_plants_l1553_155312


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l1553_155384

theorem parallel_lines_slope_eq (m : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y - 3 = 0 → 6 * x + m * y + 1 = 0) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l1553_155384


namespace NUMINAMATH_GPT_find_g2_l1553_155362

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1)
noncomputable def g (f : ℝ → ℝ) (y : ℝ) : ℝ := f⁻¹ y

variable (a : ℝ)
variable (h_inv : ∀ (x : ℝ), g (f a) (f a x) = x)
variable (h_g4 : g (f a) 4 = 2)

theorem find_g2 : g (f a) 2 = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_find_g2_l1553_155362


namespace NUMINAMATH_GPT_max_value_on_interval_l1553_155395

noncomputable def f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 10

theorem max_value_on_interval :
  (∀ x ∈ Set.Icc (1 : ℝ) 3, f 2 <= f x) → 
  ∃ y ∈ Set.Icc (1 : ℝ) 3, ∀ z ∈ Set.Icc (1 : ℝ) 3, f y >= f z :=
by
  sorry

end NUMINAMATH_GPT_max_value_on_interval_l1553_155395


namespace NUMINAMATH_GPT_find_angle_A_l1553_155372

theorem find_angle_A (A B C a b c : ℝ) 
  (h_triangle: a = Real.sqrt 2)
  (h_sides: b = 2 * Real.sin B + Real.cos B)
  (h_b_eq: b = Real.sqrt 2)
  (h_a_lt_b: a < b)
  : A = Real.pi / 6 := sorry

end NUMINAMATH_GPT_find_angle_A_l1553_155372


namespace NUMINAMATH_GPT_units_digit_5_pow_2023_l1553_155324

theorem units_digit_5_pow_2023 : ∀ n : ℕ, (n > 0) → (5^n % 10 = 5) → (5^2023 % 10 = 5) :=
by
  intros n hn hu
  have h_units_digit : ∀ k : ℕ, (k > 0) → 5^k % 10 = 5 := by
    intro k hk
    sorry -- pattern proof not included
  exact h_units_digit 2023 (by norm_num)

end NUMINAMATH_GPT_units_digit_5_pow_2023_l1553_155324


namespace NUMINAMATH_GPT_train_length_l1553_155371

noncomputable def length_of_train (t : ℝ) (v_train_kmh : ℝ) (v_man_kmh : ℝ) : ℝ :=
  let v_relative_kmh := v_train_kmh - v_man_kmh
  let v_relative_ms := v_relative_kmh * 1000 / 3600
  v_relative_ms * t

theorem train_length : length_of_train 30.99752019838413 80 8 = 619.9504039676826 := 
  by simp [length_of_train]; sorry

end NUMINAMATH_GPT_train_length_l1553_155371


namespace NUMINAMATH_GPT_episodes_relationship_l1553_155304

variable (x y z : ℕ)

theorem episodes_relationship 
  (h1 : x * z = 50) 
  (h2 : y * z = 75) : 
  y = (3 / 2) * x ∧ z = 50 / x := 
by
  sorry

end NUMINAMATH_GPT_episodes_relationship_l1553_155304


namespace NUMINAMATH_GPT_set_intersection_l1553_155334

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def intersection : Set ℕ := {1, 3}

theorem set_intersection : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_set_intersection_l1553_155334


namespace NUMINAMATH_GPT_treasures_on_island_l1553_155322

-- Define the propositions P and K
def P : Prop := ∃ p : Prop, p
def K : Prop := ∃ k : Prop, k

-- Define the claim by A
def A_claim : Prop := K ↔ P

-- Theorem statement as specified part (b)
theorem treasures_on_island (A_is_knight_or_liar : (A_claim ↔ true) ∨ (A_claim ↔ false)) : ∃ P, P :=
by
  sorry

end NUMINAMATH_GPT_treasures_on_island_l1553_155322


namespace NUMINAMATH_GPT_meet_second_time_4_5_minutes_l1553_155369

-- Define the initial conditions
def opposite_ends := true      -- George and Henry start from opposite ends
def pass_in_center := 1.5      -- They pass each other in the center after 1.5 minutes
def no_time_lost := true       -- No time lost in turning
def constant_speeds := true    -- They maintain their respective speeds

-- Prove that they pass each other the second time after 4.5 minutes
theorem meet_second_time_4_5_minutes :
  opposite_ends ∧ pass_in_center = 1.5 ∧ no_time_lost ∧ constant_speeds → 
  ∃ t : ℝ, t = 4.5 := by
  sorry

end NUMINAMATH_GPT_meet_second_time_4_5_minutes_l1553_155369


namespace NUMINAMATH_GPT_problem_correct_answer_l1553_155375

theorem problem_correct_answer (x y : ℕ) (h1 : y > 3) (h2 : x^2 + y^4 = 2 * ((x - 6)^2 + (y + 1)^2)) : x^2 + y^4 = 1994 :=
  sorry

end NUMINAMATH_GPT_problem_correct_answer_l1553_155375


namespace NUMINAMATH_GPT_ratio_of_speeds_l1553_155347

-- Define the speeds V1 and V2
variable {V1 V2 : ℝ}

-- Given the initial conditions
def bike_ride_time_min := 10 -- in minutes
def subway_ride_time_min := 40 -- in minutes
def total_bike_only_time_min := 210 -- 3.5 hours in minutes

-- Prove the ratio of subway speed to bike speed is 5:1
theorem ratio_of_speeds (h : bike_ride_time_min * V1 + subway_ride_time_min * V2 = total_bike_only_time_min * V1) :
  V2 = 5 * V1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1553_155347


namespace NUMINAMATH_GPT_maria_average_speed_l1553_155300

noncomputable def average_speed (total_distance : ℕ) (total_time : ℕ) : ℚ :=
  total_distance / total_time

theorem maria_average_speed :
  average_speed 200 7 = 28 + 4 / 7 :=
sorry

end NUMINAMATH_GPT_maria_average_speed_l1553_155300


namespace NUMINAMATH_GPT_rotated_translated_line_eq_l1553_155319

theorem rotated_translated_line_eq :
  ∀ (x y : ℝ), y = 3 * x → y = - (1 / 3) * x + (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_rotated_translated_line_eq_l1553_155319


namespace NUMINAMATH_GPT_total_cost_is_1_85_times_selling_price_l1553_155341

def total_cost (P : ℝ) : ℝ := 140 * 2 * P + 90 * P

def loss (P : ℝ) : ℝ := 70 * 2 * P + 30 * P

def selling_price (P : ℝ) : ℝ := total_cost P - loss P

theorem total_cost_is_1_85_times_selling_price (P : ℝ) :
  total_cost P = 1.85 * selling_price P := by
  sorry

end NUMINAMATH_GPT_total_cost_is_1_85_times_selling_price_l1553_155341


namespace NUMINAMATH_GPT_intersection_A_B_l1553_155399

-- Define sets A and B based on given conditions
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- Prove the intersection of A and B equals (2,4)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 4} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1553_155399


namespace NUMINAMATH_GPT_total_tour_time_l1553_155315

-- Declare constants for distances
def distance1 : ℝ := 55
def distance2 : ℝ := 40
def distance3 : ℝ := 70
def extra_miles : ℝ := 10

-- Declare constants for speeds
def speed1_part1 : ℝ := 60
def speed1_part2 : ℝ := 40
def speed2 : ℝ := 45
def speed3_part1 : ℝ := 45
def speed3_part2 : ℝ := 35
def speed3_part3 : ℝ := 50
def return_speed : ℝ := 55

-- Declare constants for stop times
def stop1 : ℝ := 1
def stop2 : ℝ := 1.5
def stop3 : ℝ := 2

-- Prove the total time required for the tour
theorem total_tour_time :
  (30 / speed1_part1) + (25 / speed1_part2) + stop1 +
  (distance2 / speed2) + stop2 +
  (20 / speed3_part1) + (30 / speed3_part2) + (20 / speed3_part3) + stop3 +
  ((distance1 + distance2 + distance3 + extra_miles) / return_speed) = 11.40 :=
by
  sorry

end NUMINAMATH_GPT_total_tour_time_l1553_155315


namespace NUMINAMATH_GPT_max_value_of_xy_l1553_155337

theorem max_value_of_xy (x y : ℝ) (h₁ : x + y = 40) (h₂ : x > 0) (h₃ : y > 0) : xy ≤ 400 :=
sorry

end NUMINAMATH_GPT_max_value_of_xy_l1553_155337


namespace NUMINAMATH_GPT_tan_alpha_beta_l1553_155356

theorem tan_alpha_beta (α β : ℝ) (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := 
sorry

end NUMINAMATH_GPT_tan_alpha_beta_l1553_155356


namespace NUMINAMATH_GPT_vertical_complementary_perpendicular_l1553_155378

theorem vertical_complementary_perpendicular (α β : ℝ) (l1 l2 : ℝ) :
  (α = β ∧ α + β = 90) ∧ l1 = l2 -> l1 + l2 = 90 := by
  sorry

end NUMINAMATH_GPT_vertical_complementary_perpendicular_l1553_155378


namespace NUMINAMATH_GPT_integer_part_not_perfect_square_l1553_155336

noncomputable def expr (n : ℕ) : ℝ :=
  2 * Real.sqrt (n + 1) / (Real.sqrt (n + 1) - Real.sqrt n)

theorem integer_part_not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = ⌊expr n⌋ :=
  sorry

end NUMINAMATH_GPT_integer_part_not_perfect_square_l1553_155336


namespace NUMINAMATH_GPT_patriots_won_games_l1553_155326

theorem patriots_won_games (C P M S T E : ℕ) 
  (hC : C > 25)
  (hPC : P > C)
  (hMP : M > P)
  (hSC : S > C)
  (hSP : S < P)
  (hTE : T > E) : 
  P = 35 :=
sorry

end NUMINAMATH_GPT_patriots_won_games_l1553_155326


namespace NUMINAMATH_GPT_percentage_problem_l1553_155381

theorem percentage_problem (p x : ℝ) (h1 : (p / 100) * x = 400) (h2 : (120 / 100) * x = 2400) : p = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1553_155381


namespace NUMINAMATH_GPT_no_nat_solutions_l1553_155357
-- Import the Mathlib library

-- Lean statement for the proof problem
theorem no_nat_solutions (x : ℕ) : ¬ (19 * x^2 + 97 * x = 1997) :=
by {
  -- Solution omitted
  sorry
}

end NUMINAMATH_GPT_no_nat_solutions_l1553_155357


namespace NUMINAMATH_GPT_seating_chart_example_l1553_155383

def seating_chart_representation (a b : ℕ) : String :=
  s!"{a} columns {b} rows"

theorem seating_chart_example :
  seating_chart_representation 4 3 = "4 columns 3 rows" :=
by
  sorry

end NUMINAMATH_GPT_seating_chart_example_l1553_155383


namespace NUMINAMATH_GPT_negation_proposition_correct_l1553_155390

theorem negation_proposition_correct : 
  (∀ x : ℝ, 0 < x → x + 4 / x ≥ 4) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_negation_proposition_correct_l1553_155390


namespace NUMINAMATH_GPT_num_diagonals_29_sides_l1553_155382

-- Define the number of sides
def n : Nat := 29

-- Calculate the combination (binomial coefficient) of selecting 2 vertices from n vertices
def binom (n k : Nat) : Nat := Nat.choose n k

-- Define the number of diagonals in a polygon with n sides
def num_diagonals (n : Nat) : Nat := binom n 2 - n

-- State the theorem to prove the number of diagonals for a polygon with 29 sides is 377
theorem num_diagonals_29_sides : num_diagonals 29 = 377 :=
by
  sorry

end NUMINAMATH_GPT_num_diagonals_29_sides_l1553_155382


namespace NUMINAMATH_GPT_solve_x_sq_plus_y_sq_l1553_155327

theorem solve_x_sq_plus_y_sq (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = 2) : x^2 + y^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_sq_plus_y_sq_l1553_155327


namespace NUMINAMATH_GPT_exercise_l1553_155364

noncomputable def g (x : ℝ) : ℝ := x^3
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem exercise : f (g 3) = 1457 := by
  sorry

end NUMINAMATH_GPT_exercise_l1553_155364


namespace NUMINAMATH_GPT_proportion_solution_l1553_155332

theorem proportion_solution (x : ℚ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by sorry

end NUMINAMATH_GPT_proportion_solution_l1553_155332


namespace NUMINAMATH_GPT_peter_bought_large_glasses_l1553_155387

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end NUMINAMATH_GPT_peter_bought_large_glasses_l1553_155387


namespace NUMINAMATH_GPT_intersection_of_sets_l1553_155320

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1553_155320


namespace NUMINAMATH_GPT_probability_of_two_white_balls_correct_l1553_155338

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_white_balls_correct_l1553_155338


namespace NUMINAMATH_GPT_membership_fee_increase_each_year_l1553_155317

variable (fee_increase : ℕ)

def yearly_membership_fee_increase (first_year_fee sixth_year_fee yearly_increase : ℕ) : Prop :=
  yearly_increase * 5 = sixth_year_fee - first_year_fee

theorem membership_fee_increase_each_year :
  yearly_membership_fee_increase 80 130 10 :=
by
  unfold yearly_membership_fee_increase
  sorry

end NUMINAMATH_GPT_membership_fee_increase_each_year_l1553_155317


namespace NUMINAMATH_GPT_number_of_pipes_l1553_155331

theorem number_of_pipes (d_large d_small: ℝ) (π : ℝ) (h1: d_large = 4) (h2: d_small = 2) : 
  ((π * (d_large / 2)^2) / (π * (d_small / 2)^2) = 4) := 
by
  sorry

end NUMINAMATH_GPT_number_of_pipes_l1553_155331


namespace NUMINAMATH_GPT_dad_use_per_brush_correct_l1553_155310

def toothpaste_total : ℕ := 105
def mom_use_per_brush : ℕ := 2
def anne_brother_use_per_brush : ℕ := 1
def brushing_per_day : ℕ := 3
def days_to_finish : ℕ := 5

-- Defining the daily use function for Anne's Dad
def dad_use_per_brush (D : ℕ) : ℕ := D

theorem dad_use_per_brush_correct (D : ℕ) 
  (h : brushing_per_day * (mom_use_per_brush + anne_brother_use_per_brush * 2 + dad_use_per_brush D) * days_to_finish = toothpaste_total) 
  : dad_use_per_brush D = 3 :=
by sorry

end NUMINAMATH_GPT_dad_use_per_brush_correct_l1553_155310


namespace NUMINAMATH_GPT_intersection_points_vertex_of_function_value_of_m_shift_l1553_155325

noncomputable def quadratic_function (x m : ℝ) : ℝ :=
  (x - m) ^ 2 - 2 * (x - m)

theorem intersection_points (m : ℝ) : 
  ∃ x, quadratic_function x m = 0 ↔ x = m ∨ x = m + 2 := 
by
  sorry

theorem vertex_of_function (m : ℝ) : 
  ∃ x y, y = quadratic_function x m 
  ∧ x = m + 1 ∧ y = -1 := 
by
  sorry

theorem value_of_m_shift (m : ℝ) :
  (m - 2 = 0) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_vertex_of_function_value_of_m_shift_l1553_155325


namespace NUMINAMATH_GPT_contrapositive_statement_l1553_155339

-- Conditions: x and y are real numbers
variables (x y : ℝ)

-- Contrapositive statement: If x ≠ 0 or y ≠ 0, then x^2 + y^2 ≠ 0
theorem contrapositive_statement (hx : x ≠ 0 ∨ y ≠ 0) : x^2 + y^2 ≠ 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_statement_l1553_155339


namespace NUMINAMATH_GPT_unique_k_exists_l1553_155350

theorem unique_k_exists (k : ℕ) (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (a^2 + b^2 = k * a * b) ↔ k = 2 := sorry

end NUMINAMATH_GPT_unique_k_exists_l1553_155350


namespace NUMINAMATH_GPT_area_of_shaded_region_l1553_155354

theorem area_of_shaded_region
  (r_large : ℝ) (r_small : ℝ) (n_small : ℕ) (π : ℝ)
  (A_large : ℝ) (A_small : ℝ) (A_7_small : ℝ) (A_shaded : ℝ)
  (h1 : r_large = 20)
  (h2 : r_small = 10)
  (h3 : n_small = 7)
  (h4 : π = 3.14)
  (h5 : A_large = π * r_large^2)
  (h6 : A_small = π * r_small^2)
  (h7 : A_7_small = n_small * A_small)
  (h8 : A_shaded = A_large - A_7_small) :
  A_shaded = 942 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1553_155354


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1553_155342

open Real

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x + y = 5 * x * y) (hx2y : x = 2 * y) : 
  (1 / x) + (1 / y) = 5 := 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1553_155342


namespace NUMINAMATH_GPT_length_of_bridge_l1553_155353

/-- Prove the length of the bridge -/
theorem length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 70 →
  crossing_time_sec = 13.884603517432893 →
  (70 * (1000 / 3600) * 13.884603517432893 - 120 = 150) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1553_155353


namespace NUMINAMATH_GPT_sqrt_x_minus_2_meaningful_l1553_155329

theorem sqrt_x_minus_2_meaningful (x : ℝ) (h : 0 ≤ x - 2) : 2 ≤ x :=
by sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_meaningful_l1553_155329


namespace NUMINAMATH_GPT_solve_xyz_l1553_155303

theorem solve_xyz (a b c : ℝ) (h1 : a = y + z) (h2 : b = x + z) (h3 : c = x + y) 
                   (h4 : 0 < y) (h5 : 0 < z) (h6 : 0 < x)
                   (hab : b + c > a) (hbc : a + c > b) (hca : a + b > c) :
  x = (b - a + c)/2 ∧ y = (a - b + c)/2 ∧ z = (a + b - c)/2 :=
by
  sorry

end NUMINAMATH_GPT_solve_xyz_l1553_155303


namespace NUMINAMATH_GPT_evaluate_expression_l1553_155389

theorem evaluate_expression (x : ℝ) (h : x = 2) : 4 * x ^ 2 + 1 / 2 = 16.5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1553_155389


namespace NUMINAMATH_GPT_vertical_line_intersect_parabola_ex1_l1553_155316

theorem vertical_line_intersect_parabola_ex1 (m : ℝ) (h : ∀ y : ℝ, (-4 * y^2 + 2*y + 3 = m) → false) :
  m = 13 / 4 :=
sorry

end NUMINAMATH_GPT_vertical_line_intersect_parabola_ex1_l1553_155316


namespace NUMINAMATH_GPT_area_of_shaded_region_l1553_155311

theorem area_of_shaded_region 
  (r R : ℝ)
  (hR : R = 9)
  (h : 2 * r = R) :
  π * R^2 - 3 * (π * r^2) = 20.25 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1553_155311


namespace NUMINAMATH_GPT_minimum_rebate_rate_l1553_155361

open Real

noncomputable def rebate_rate (s p_M p_N p: ℝ) : ℝ := 100 * (p_M + p_N - p) / s

theorem minimum_rebate_rate 
  (s p_M p_N p : ℝ)
  (h_M : 0.19 * 0.4 * s ≤ p_M ∧ p_M ≤ 0.24 * 0.4 * s)
  (h_N : 0.29 * 0.6 * s ≤ p_N ∧ p_N ≤ 0.34 * 0.6 * s)
  (h_total : 0.10 * s ≤ p ∧ p ≤ 0.15 * s) :
  ∃ r : ℝ, r = rebate_rate s p_M p_N p ∧ 0.1 ≤ r ∧ r ≤ 0.2 :=
sorry

end NUMINAMATH_GPT_minimum_rebate_rate_l1553_155361


namespace NUMINAMATH_GPT_girl_scout_cookie_sales_l1553_155388

theorem girl_scout_cookie_sales :
  ∃ C P : ℝ, C + P = 1585 ∧ 1.25 * C + 0.75 * P = 1586.25 ∧ P = 790 :=
by
  sorry

end NUMINAMATH_GPT_girl_scout_cookie_sales_l1553_155388


namespace NUMINAMATH_GPT_maximum_lambda_l1553_155359

theorem maximum_lambda (a b : ℝ) : (27 / 4) * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3 := 
sorry

end NUMINAMATH_GPT_maximum_lambda_l1553_155359


namespace NUMINAMATH_GPT_expression_evaluation_l1553_155377

theorem expression_evaluation : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1553_155377


namespace NUMINAMATH_GPT_ramon_current_age_l1553_155349

variable (R : ℕ) (L : ℕ)

theorem ramon_current_age :
  (L = 23) → (R + 20 = 2 * L) → R = 26 :=
by
  intro hL hR
  rw [hL] at hR
  have : R + 20 = 46 := by linarith
  linarith

end NUMINAMATH_GPT_ramon_current_age_l1553_155349


namespace NUMINAMATH_GPT_percentage_problem_l1553_155392

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 400) : 1.20 * x = 2400 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1553_155392


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1553_155398

theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (y = (1/2) * x) ∨ (y = -(1/2) * x) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1553_155398


namespace NUMINAMATH_GPT_determine_f_4_l1553_155335

theorem determine_f_4 (f g : ℝ → ℝ)
  (h1 : ∀ x y z : ℝ, f (x^2 + y * f z) = x * g x + z * g y)
  (h2 : ∀ x : ℝ, g x = 2 * x) :
  f 4 = 32 :=
sorry

end NUMINAMATH_GPT_determine_f_4_l1553_155335


namespace NUMINAMATH_GPT_girls_count_l1553_155396

-- Define the constants according to the conditions
def boys_on_team : ℕ := 28
def groups : ℕ := 8
def members_per_group : ℕ := 4

-- Calculate the total number of members
def total_members : ℕ := groups * members_per_group

-- Calculate the number of girls by subtracting the number of boys from the total members
def girls_on_team : ℕ := total_members - boys_on_team

-- The proof statement: prove that the number of girls on the team is 4
theorem girls_count : girls_on_team = 4 := by
  -- Skip the proof, completing the statement
  sorry

end NUMINAMATH_GPT_girls_count_l1553_155396


namespace NUMINAMATH_GPT_prove_expression_l1553_155352

theorem prove_expression (a : ℝ) (h : a^2 + a - 1 = 0) : 2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end NUMINAMATH_GPT_prove_expression_l1553_155352


namespace NUMINAMATH_GPT_quadratic_value_l1553_155360

theorem quadratic_value (a b c : ℤ) (a_pos : a > 0) (h_eq : ∀ x : ℝ, (a * x + b)^2 = 49 * x^2 + 70 * x + c) : a + b + c = -134 :=
by
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_quadratic_value_l1553_155360


namespace NUMINAMATH_GPT_find_S_l1553_155393

variable {R S T c : ℝ}

theorem find_S
  (h1 : R = 2)
  (h2 : T = 1/2)
  (h3 : S = 4)
  (h4 : R = c * S / T)
  (h5 : R = 8)
  (h6 : T = 1/3) :
  S = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_S_l1553_155393


namespace NUMINAMATH_GPT_automobile_distance_2_minutes_l1553_155367

theorem automobile_distance_2_minutes (a : ℝ) :
  let acceleration := a / 12
  let time_minutes := 2
  let time_seconds := time_minutes * 60
  let distance_feet := (1 / 2) * acceleration * time_seconds^2
  let distance_yards := distance_feet / 3
  distance_yards = 200 * a := 
by sorry

end NUMINAMATH_GPT_automobile_distance_2_minutes_l1553_155367


namespace NUMINAMATH_GPT_f_three_equals_322_l1553_155313

def f (z : ℝ) : ℝ := (z^2 - 2) * ((z^2 - 2)^2 - 3)

theorem f_three_equals_322 :
  f 3 = 322 :=
by
  -- Proof steps (left out intentionally as per instructions)
  sorry

end NUMINAMATH_GPT_f_three_equals_322_l1553_155313


namespace NUMINAMATH_GPT_pradeep_max_marks_l1553_155323

theorem pradeep_max_marks (M : ℝ) 
  (pass_condition : 0.35 * M = 210) : M = 600 :=
sorry

end NUMINAMATH_GPT_pradeep_max_marks_l1553_155323


namespace NUMINAMATH_GPT_frosting_problem_l1553_155343

-- Define the conditions
def cagney_rate := 1/15  -- Cagney's rate in cupcakes per second
def lacey_rate := 1/45   -- Lacey's rate in cupcakes per second
def total_time := 600  -- Total time in seconds (10 minutes)

-- Function to calculate the combined rate
def combined_rate (r1 r2 : ℝ) : ℝ := r1 + r2

-- Hypothesis combining the conditions
def hypothesis : Prop :=
  combined_rate cagney_rate lacey_rate = 1/11.25

-- Statement to prove: together they can frost 53 cupcakes within 10 minutes 
theorem frosting_problem : ∀ (total_time: ℝ) (hyp : hypothesis),
  total_time / (cagney_rate + lacey_rate) = 53 :=
by
  intro total_time hyp
  sorry

end NUMINAMATH_GPT_frosting_problem_l1553_155343


namespace NUMINAMATH_GPT_total_votes_cast_l1553_155376

variable (total_votes : ℕ)
variable (emily_votes : ℕ)
variable (emily_share : ℚ := 4 / 15)
variable (dexter_share : ℚ := 1 / 3)

theorem total_votes_cast :
  emily_votes = 48 → 
  emily_share * total_votes = emily_votes → 
  total_votes = 180 := by
  intro h_emily_votes
  intro h_emily_share
  sorry

end NUMINAMATH_GPT_total_votes_cast_l1553_155376


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l1553_155345

noncomputable def y (x m : ℝ) : ℝ := x^2 + m / x
noncomputable def y_prime (x m : ℝ) : ℝ := 2 * x - m / x^2

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x ≥ 1, y_prime x m ≥ 0) ↔ m ≤ 2 :=
sorry  -- Proof skipped as instructed

-- Now, state that m < 1 is a sufficient but not necessary condition
theorem m_sufficient_but_not_necessary (m : ℝ) :
  m < 1 → (∀ x ≥ 1, y_prime x m ≥ 0) :=
sorry  -- Proof skipped as instructed

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l1553_155345


namespace NUMINAMATH_GPT_s_l1553_155379

def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def total_money_spent : ℝ := 15.0

theorem s'mores_per_scout :
  (total_money_spent / cost_per_chocolate_bar * sections_per_chocolate_bar) / scouts = 2 :=
by
  sorry

end NUMINAMATH_GPT_s_l1553_155379


namespace NUMINAMATH_GPT_common_ratio_value_l1553_155330

theorem common_ratio_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) → (x + y + z = 0 ∨ x + y + z ≠ 0) → ((x + y) / z = -1 ∨ (x + y) / z = 2) :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_value_l1553_155330


namespace NUMINAMATH_GPT_find_a6_l1553_155305

noncomputable def a_n (n : ℕ) : ℝ := sorry
noncomputable def S_n (n : ℕ) : ℝ := sorry
noncomputable def r : ℝ := sorry

axiom h_pos : ∀ n, a_n n > 0
axiom h_s3 : S_n 3 = 14
axiom h_a3 : a_n 3 = 8

theorem find_a6 : a_n 6 = 64 := by sorry

end NUMINAMATH_GPT_find_a6_l1553_155305


namespace NUMINAMATH_GPT_M_gt_N_l1553_155358

variable (a : ℝ)

def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

theorem M_gt_N : M a > N a := by
  sorry

end NUMINAMATH_GPT_M_gt_N_l1553_155358


namespace NUMINAMATH_GPT_oil_layer_height_l1553_155344

/-- Given a tank with a rectangular bottom measuring 16 cm in length and 12 cm in width, initially containing 6 cm deep water and 6 cm deep oil, and an iron block with dimensions 8 cm in length, 8 cm in width, and 12 cm in height -/

theorem oil_layer_height (volume_water volume_oil volume_iron base_area new_volume_water : ℝ) 
  (base_area_def : base_area = 16 * 12) 
  (volume_water_def : volume_water = base_area * 6) 
  (volume_oil_def : volume_oil = base_area * 6) 
  (volume_iron_def : volume_iron = 8 * 8 * 12) 
  (new_volume_water_def : new_volume_water = volume_water + volume_iron) 
  (new_water_height : new_volume_water / base_area = 10) 
  : (volume_water + volume_oil) / base_area - (new_volume_water / base_area - 6) = 7 :=
by 
  sorry

end NUMINAMATH_GPT_oil_layer_height_l1553_155344


namespace NUMINAMATH_GPT_off_the_rack_suit_cost_l1553_155386

theorem off_the_rack_suit_cost (x : ℝ)
  (h1 : ∀ y, y = 3 * x + 200)
  (h2 : ∀ y, x + y = 1400) :
  x = 300 :=
by
  sorry

end NUMINAMATH_GPT_off_the_rack_suit_cost_l1553_155386


namespace NUMINAMATH_GPT_trapezium_area_l1553_155380

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end NUMINAMATH_GPT_trapezium_area_l1553_155380


namespace NUMINAMATH_GPT_possible_values_of_reciprocal_l1553_155321

theorem possible_values_of_reciprocal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ S, S = { x : ℝ | x >= 9 } ∧ (∃ x, x = (1/a + 1/b) ∧ x ∈ S) :=
sorry

end NUMINAMATH_GPT_possible_values_of_reciprocal_l1553_155321


namespace NUMINAMATH_GPT_inequality_inequality_only_if_k_is_one_half_l1553_155307

theorem inequality_inequality_only_if_k_is_one_half :
  (∀ t : ℝ, -1 < t ∧ t < 1 → (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1) ↔ k = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_inequality_only_if_k_is_one_half_l1553_155307


namespace NUMINAMATH_GPT_sqrt_defined_iff_ge_neg1_l1553_155365

theorem sqrt_defined_iff_ge_neg1 (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x + 1)) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_GPT_sqrt_defined_iff_ge_neg1_l1553_155365


namespace NUMINAMATH_GPT_exponent_equality_l1553_155351

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end NUMINAMATH_GPT_exponent_equality_l1553_155351


namespace NUMINAMATH_GPT_number_of_intersections_l1553_155318

-- Conditions for the problem
def Line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 2
def Line2 (x y : ℝ) : Prop := 5 * x + 3 * y = 6
def Line3 (x y : ℝ) : Prop := x - 4 * y = 8

-- Statement to prove
theorem number_of_intersections : ∃ (p1 p2 p3 : ℝ × ℝ), 
  (Line1 p1.1 p1.2 ∧ Line2 p1.1 p1.2) ∧ 
  (Line1 p2.1 p2.2 ∧ Line3 p2.1 p2.2) ∧ 
  (Line2 p3.1 p3.2 ∧ Line3 p3.1 p3.2) ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end NUMINAMATH_GPT_number_of_intersections_l1553_155318


namespace NUMINAMATH_GPT_customers_not_tipping_l1553_155348

theorem customers_not_tipping (number_of_customers tip_per_customer total_earned_in_tips : ℕ)
  (h_number : number_of_customers = 7)
  (h_tip : tip_per_customer = 3)
  (h_earned : total_earned_in_tips = 6) :
  number_of_customers - (total_earned_in_tips / tip_per_customer) = 5 :=
by
  sorry

end NUMINAMATH_GPT_customers_not_tipping_l1553_155348


namespace NUMINAMATH_GPT_sum_of_six_terms_l1553_155302

variable (a₁ a₂ a₃ a₄ a₅ a₆ q : ℝ)

-- Conditions
def geom_seq := a₂ = q * a₁ ∧ a₃ = q * a₂ ∧ a₄ = q * a₃ ∧ a₅ = q * a₄ ∧ a₆ = q * a₅
def cond₁ : Prop := a₁ + a₃ = 5 / 2
def cond₂ : Prop := a₂ + a₄ = 5 / 4

-- Problem Statement
theorem sum_of_six_terms : geom_seq a₁ a₂ a₃ a₄ a₅ a₆ q → cond₁ a₁ a₃ → cond₂ a₂ a₄ → 
  (a₁ * (1 - q^6) / (1 - q) = 63 / 16) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_six_terms_l1553_155302


namespace NUMINAMATH_GPT_twenty_percent_correct_l1553_155340

def certain_number := 400
def forty_percent (x : ℕ) : ℕ := 40 * x / 100
def twenty_percent_of_certain_number (x : ℕ) : ℕ := 20 * x / 100

theorem twenty_percent_correct : 
  (∃ x : ℕ, forty_percent x = 160) → twenty_percent_of_certain_number certain_number = 80 :=
by
  sorry

end NUMINAMATH_GPT_twenty_percent_correct_l1553_155340


namespace NUMINAMATH_GPT_single_equivalent_discount_l1553_155333

theorem single_equivalent_discount :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  ∃ (k : ℝ), (1 - k) = (1 - discount1) * (1 - discount2) * (1 - discount3) ∧ k = 0.27325 :=
by
  sorry

end NUMINAMATH_GPT_single_equivalent_discount_l1553_155333


namespace NUMINAMATH_GPT_second_hand_travel_distance_l1553_155374

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end NUMINAMATH_GPT_second_hand_travel_distance_l1553_155374


namespace NUMINAMATH_GPT_second_quadrant_necessary_not_sufficient_l1553_155309

variable (α : ℝ)

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → is_second_quadrant α) ∧ ¬ (∀ α, is_second_quadrant α → is_obtuse α) := by
  sorry

end NUMINAMATH_GPT_second_quadrant_necessary_not_sufficient_l1553_155309


namespace NUMINAMATH_GPT_amount_of_bill_is_720_l1553_155373

-- Definitions and conditions
def TD : ℝ := 360
def BD : ℝ := 428.21

-- The relationship between TD, BD, and FV
axiom relationship (FV : ℝ) : BD = TD + (TD * BD) / (FV - TD)

-- The main theorem to prove
theorem amount_of_bill_is_720 : ∃ FV : ℝ, BD = TD + (TD * BD) / (FV - TD) ∧ FV = 720 :=
by
  use 720
  sorry

end NUMINAMATH_GPT_amount_of_bill_is_720_l1553_155373


namespace NUMINAMATH_GPT_veranda_area_l1553_155308

theorem veranda_area (length_room width_room width_veranda : ℕ)
  (h_length : length_room = 20) 
  (h_width : width_room = 12) 
  (h_veranda : width_veranda = 2) : 
  (length_room + 2 * width_veranda) * (width_room + 2 * width_veranda) - (length_room * width_room) = 144 := 
by
  sorry

end NUMINAMATH_GPT_veranda_area_l1553_155308


namespace NUMINAMATH_GPT_correct_calculation_l1553_155314

theorem correct_calculation (a : ℝ) : -3 * a - 2 * a = -5 * a :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1553_155314


namespace NUMINAMATH_GPT_area_of_triangle_PQR_is_correct_l1553_155394

noncomputable def calculate_area_of_triangle_PQR : ℝ := 
  let side_length := 4
  let altitude := 8
  let WO := (side_length * Real.sqrt 2) / 2
  let center_to_vertex_distance := Real.sqrt (WO^2 + altitude^2)
  let WP := (1 / 4) * WO
  let YQ := (1 / 2) * WO
  let XR := (3 / 4) * WO
  let area := (1 / 2) * (YQ - WP) * (XR - YQ)
  area

theorem area_of_triangle_PQR_is_correct :
  calculate_area_of_triangle_PQR = 2.25 := sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_is_correct_l1553_155394


namespace NUMINAMATH_GPT_tangent_line_at_point_is_correct_l1553_155370

theorem tangent_line_at_point_is_correct :
  ∀ (x y : ℝ), (y = x^2 + 2 * x) → (x = 1) → (y = 3) → (4 * x - y - 1 = 0) :=
by
  intros x y h_curve h_x h_y
  -- Here would be the proof
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_is_correct_l1553_155370


namespace NUMINAMATH_GPT_intersection_complement_eq_singleton_l1553_155363

open Set

def U : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}
def CU_A : Set ℤ := U \ A

theorem intersection_complement_eq_singleton : B ∩ CU_A = {0} := 
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_singleton_l1553_155363


namespace NUMINAMATH_GPT_car_return_speed_l1553_155368

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end NUMINAMATH_GPT_car_return_speed_l1553_155368


namespace NUMINAMATH_GPT_sum_of_distinct_integers_eq_zero_l1553_155366

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_eq_zero_l1553_155366
