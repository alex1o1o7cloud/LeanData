import Mathlib

namespace abs_nested_expression_l574_57497

theorem abs_nested_expression (x : ℝ) (h : x = 2023) : 
  abs (abs (abs x - x) - abs x) - x = 0 :=
by
  subst h
  sorry

end abs_nested_expression_l574_57497


namespace problem_statement_l574_57462

-- Define the functions
def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x - 5

-- Define the main theorem statement
theorem problem_statement : f (g (-2)) = 81 := by
  sorry

end problem_statement_l574_57462


namespace greatest_four_digit_multiple_of_17_l574_57442

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l574_57442


namespace overlapping_area_l574_57476

def area_of_overlap (g1 g2 : Grid) : ℝ :=
  -- Dummy implementation to ensure code compiles
  6.0

structure Grid :=
  (size : ℝ) (arrow_direction : Direction)

inductive Direction
| North
| West

theorem overlapping_area (g1 g2 : Grid) 
  (h1 : g1.size = 4) 
  (h2 : g2.size = 4) 
  (d1 : g1.arrow_direction = Direction.North) 
  (d2 : g2.arrow_direction = Direction.West) 
  : area_of_overlap g1 g2 = 6 :=
by
  sorry

end overlapping_area_l574_57476


namespace tetrahedron_cross_section_area_l574_57407

theorem tetrahedron_cross_section_area (a : ℝ) : 
  ∃ (S : ℝ), 
    let AB := a; 
    let AC := a;
    let AD := a;
    S = (3 * a^2) / 8 
    := sorry

end tetrahedron_cross_section_area_l574_57407


namespace Nadia_distance_is_18_l574_57401

-- Variables and conditions
variables (x : ℕ)

-- Definitions based on conditions
def Hannah_walked (x : ℕ) : ℕ := x
def Nadia_walked (x : ℕ) : ℕ := 2 * x
def total_distance (x : ℕ) : ℕ := Hannah_walked x + Nadia_walked x

-- The proof statement
theorem Nadia_distance_is_18 (h : total_distance x = 27) : Nadia_walked x = 18 :=
by
  sorry

end Nadia_distance_is_18_l574_57401


namespace parity_of_expression_l574_57469

theorem parity_of_expression (e m : ℕ) (he : (∃ k : ℕ, e = 2 * k)) : Odd (e ^ 2 + 3 ^ m) :=
  sorry

end parity_of_expression_l574_57469


namespace find_F_16_l574_57483

noncomputable def F : ℝ → ℝ := sorry

lemma F_condition_1 : ∀ x, (x + 4) ≠ 0 ∧ (x + 2) ≠ 0 → (F (4 * x) / F (x + 4) = 16 - (64 * x + 64) / (x^2 + 6 * x + 8)) := sorry

lemma F_condition_2 : F 8 = 33 := sorry

theorem find_F_16 : F 16 = 136 :=
by
  have h1 := F_condition_1
  have h2 := F_condition_2
  sorry

end find_F_16_l574_57483


namespace evaluate_expression_at_2_l574_57492

noncomputable def replace_and_evaluate (x : ℝ) : ℝ :=
  (3 * x - 2) / (-x + 6)

theorem evaluate_expression_at_2 :
  replace_and_evaluate 2 = -2 :=
by
  -- evaluation and computation would go here, skipped with sorry
  sorry

end evaluate_expression_at_2_l574_57492


namespace abs_reciprocal_inequality_l574_57425

theorem abs_reciprocal_inequality (a b : ℝ) (h : 1 / |a| < 1 / |b|) : |a| > |b| :=
sorry

end abs_reciprocal_inequality_l574_57425


namespace children_tickets_l574_57421

-- Definition of the problem
variables (A C t : ℕ) (h_eq_people : A + C = t) (h_eq_money : 9 * A + 5 * C = 190)

-- The main statement we need to prove
theorem children_tickets (h_t : t = 30) : C = 20 :=
by {
  -- Proof will go here eventually
  sorry
}

end children_tickets_l574_57421


namespace part1_part2_part3_l574_57459

-- Define the sequence and conditions
variable {a : ℕ → ℕ}
axiom sequence_def (n : ℕ) : a n = max (a (n + 1)) (a (n + 2)) - min (a (n + 1)) (a (n + 2))

-- Part (1)
axiom a1_def : a 1 = 1
axiom a2_def : a 2 = 2
theorem part1 : a 4 = 1 ∨ a 4 = 3 ∨ a 4 = 5 :=
  sorry

-- Part (2)
axiom has_max (M : ℕ) : ∀ n, a n ≤ M
theorem part2 : ∃ n, a n = 0 :=
  sorry

-- Part (3)
axiom positive_seq : ∀ n, a n > 0
theorem part3 : ¬∃ M : ℝ, ∀ n, a n ≤ M :=
  sorry

end part1_part2_part3_l574_57459


namespace f_at_7_l574_57422

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x + 4) = f x
axiom specific_interval_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_7 : f 7 = -2 := 
  by sorry

end f_at_7_l574_57422


namespace victor_weekly_earnings_l574_57404

def wage_per_hour : ℕ := 12
def hours_monday : ℕ := 5
def hours_tuesday : ℕ := 6
def hours_wednesday : ℕ := 7
def hours_thursday : ℕ := 4
def hours_friday : ℕ := 8

def earnings_monday := hours_monday * wage_per_hour
def earnings_tuesday := hours_tuesday * wage_per_hour
def earnings_wednesday := hours_wednesday * wage_per_hour
def earnings_thursday := hours_thursday * wage_per_hour
def earnings_friday := hours_friday * wage_per_hour

def total_earnings := earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday

theorem victor_weekly_earnings : total_earnings = 360 := by
  sorry

end victor_weekly_earnings_l574_57404


namespace incorrect_statement_B_l574_57418

theorem incorrect_statement_B (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) : ¬ ∀ (x y : ℝ), x * y + A * x + B * y + C = 0 → (x < 0 ∧ y < 0) :=
by
  sorry

end incorrect_statement_B_l574_57418


namespace contradiction_in_triangle_l574_57480

theorem contradiction_in_triangle (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (sum_angles : A + B + C = 180) : false :=
by
  sorry

end contradiction_in_triangle_l574_57480


namespace jacqueline_candy_multiple_l574_57434

theorem jacqueline_candy_multiple :
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  (jackie_candy / total_candy = 10) :=
by
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  show _ = _
  sorry

end jacqueline_candy_multiple_l574_57434


namespace correct_calculation_l574_57485

theorem correct_calculation (x : ℕ) (h : x + 10 = 21) : x * 10 = 110 :=
by
  sorry

end correct_calculation_l574_57485


namespace beetle_speed_l574_57466

theorem beetle_speed
  (distance_ant : ℝ )
  (time_minutes : ℝ)
  (distance_beetle : ℝ) 
  (distance_percent_less : ℝ)
  (time_hours : ℝ)
  (beetle_speed_kmh : ℝ)
  (h1 : distance_ant = 600)
  (h2 : time_minutes = 10)
  (h3 : time_hours = time_minutes / 60)
  (h4 : distance_percent_less = 0.25)
  (h5 : distance_beetle = distance_ant * (1 - distance_percent_less))
  (h6 : beetle_speed_kmh = distance_beetle / time_hours) : 
  beetle_speed_kmh = 2.7 :=
by 
  sorry

end beetle_speed_l574_57466


namespace triangle_formation_ways_l574_57410

-- Given conditions
def parallel_tracks : Prop := true -- The tracks are parallel, implicit condition not affecting calculation
def first_track_checkpoints := 6
def second_track_checkpoints := 10

-- The proof problem
theorem triangle_formation_ways : 
  (first_track_checkpoints * Nat.choose second_track_checkpoints 2) = 270 := by
  sorry

end triangle_formation_ways_l574_57410


namespace min_value_of_y_l574_57427

theorem min_value_of_y (x : ℝ) (hx : x > 0) : (∃ y, y = x + 4 / x^2 ∧ ∀ z, z = x + 4 / x^2 → z ≥ 3) :=
sorry

end min_value_of_y_l574_57427


namespace June_sweets_count_l574_57420

variable (A M J : ℕ)

-- condition: May has three-quarters of the number of sweets that June has
def May_sweets := M = (3/4) * J

-- condition: April has two-thirds of the number of sweets that May has
def April_sweets := A = (2/3) * M

-- condition: April, May, and June have 90 sweets between them
def Total_sweets := A + M + J = 90

-- proof problem: How many sweets does June have?
theorem June_sweets_count : 
  May_sweets M J ∧ April_sweets A M ∧ Total_sweets A M J → J = 40 :=
by
  sorry

end June_sweets_count_l574_57420


namespace shaded_area_correct_l574_57416

noncomputable def grid_width : ℕ := 15
noncomputable def grid_height : ℕ := 5
noncomputable def triangle_base : ℕ := 15
noncomputable def triangle_height : ℕ := 3
noncomputable def total_area : ℝ := (grid_width * grid_height : ℝ)
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
noncomputable def shaded_area : ℝ := total_area - triangle_area

theorem shaded_area_correct : shaded_area = 52.5 := 
by sorry

end shaded_area_correct_l574_57416


namespace solution_set_system_of_inequalities_l574_57400

theorem solution_set_system_of_inequalities :
  { x : ℝ | (2 - x) * (2 * x + 4) ≥ 0 ∧ -3 * x^2 + 2 * x + 1 < 0 } = 
  { x : ℝ | -2 ≤ x ∧ x < -1/3 ∨ 1 < x ∧ x ≤ 2 } := 
by
  sorry

end solution_set_system_of_inequalities_l574_57400


namespace max_gold_coins_l574_57496

theorem max_gold_coins : ∃ n : ℕ, (∃ k : ℕ, n = 7 * k + 2) ∧ 50 < n ∧ n < 150 ∧ n = 149 :=
by
  sorry

end max_gold_coins_l574_57496


namespace circle_placement_in_rectangle_l574_57468

theorem circle_placement_in_rectangle
  (L W : ℝ) (n : ℕ) (side_length diameter : ℝ)
  (h_dim : L = 20) (w_dim : W = 25)
  (h_squares : n = 120) (h_side_length : side_length = 1)
  (h_diameter : diameter = 1) :
  ∃ (x y : ℝ) (circle_radius : ℝ), 
    circle_radius = diameter / 2 ∧
    0 ≤ x ∧ x + diameter / 2 ≤ L ∧ 
    0 ≤ y ∧ y + diameter / 2 ≤ W ∧ 
    ∀ (i : ℕ) (hx : i < n) (sx sy : ℝ),
      0 ≤ sx ∧ sx + side_length ≤ L ∧
      0 ≤ sy ∧ sy + side_length ≤ W ∧
      dist (x, y) (sx + side_length / 2, sy + side_length / 2) ≥ diameter / 2 := 
sorry

end circle_placement_in_rectangle_l574_57468


namespace odd_and_increasing_f1_odd_and_increasing_f2_l574_57458

-- Define the functions
def f1 (x : ℝ) : ℝ := x * |x|
def f2 (x : ℝ) : ℝ := x^3

-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

-- Define the increasing function property
def is_increasing (f : ℝ → ℝ) : Prop := ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → f x1 < f x2

-- Lean statement to prove
theorem odd_and_increasing_f1 : is_odd f1 ∧ is_increasing f1 := by
  sorry

theorem odd_and_increasing_f2 : is_odd f2 ∧ is_increasing f2 := by
  sorry

end odd_and_increasing_f1_odd_and_increasing_f2_l574_57458


namespace correct_average_weight_l574_57456

theorem correct_average_weight 
  (n : ℕ) 
  (w_avg : ℝ) 
  (W_init : ℝ)
  (d1 : ℝ)
  (d2 : ℝ)
  (d3 : ℝ)
  (W_adj : ℝ)
  (w_corr : ℝ)
  (h1 : n = 30)
  (h2 : w_avg = 58.4)
  (h3 : W_init = n * w_avg)
  (h4 : d1 = 62 - 56)
  (h5 : d2 = 59 - 65)
  (h6 : d3 = 54 - 50)
  (h7 : W_adj = W_init + d1 + d2 + d3)
  (h8 : w_corr = W_adj / n) :
  w_corr = 58.5 := 
sorry

end correct_average_weight_l574_57456


namespace segment_area_formula_l574_57488
noncomputable def area_of_segment (r a : ℝ) : ℝ :=
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2)

theorem segment_area_formula (r a : ℝ) : area_of_segment r a =
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2) :=
sorry

end segment_area_formula_l574_57488


namespace geometric_series_sum_l574_57454

open Real

theorem geometric_series_sum :
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  S = -716637955 / 16777216 :=
by
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  have h : S = -716637955 / 16777216 := sorry
  exact h

end geometric_series_sum_l574_57454


namespace milburg_population_l574_57464

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l574_57464


namespace value_of_m_l574_57490

theorem value_of_m
  (m : ℤ)
  (h1 : ∃ p : ℕ → ℝ, p 4 = 1/3 ∧ p 1 = -(m + 4) ∧ p 0 = -11 ∧ (∀ (n : ℕ), (n ≠ 4 ∧ n ≠ 1 ∧ n ≠ 0) → p n = 0) ∧ 1 ≤ p 4 + p 1 + p 0) :
  m = 4 :=
  sorry

end value_of_m_l574_57490


namespace whisky_replacement_l574_57411

variable (x : ℝ) -- Original quantity of whisky in the jar
variable (y : ℝ) -- Quantity of whisky replaced

-- Condition: A jar full of whisky contains 40% alcohol
-- Condition: After replacement, the percentage of alcohol is 24%
theorem whisky_replacement (h : 0 < x) : 
  0.40 * x - 0.40 * y + 0.19 * y = 0.24 * x → y = (16 / 21) * x :=
by
  intro h_eq
  -- Sorry for the proof
  sorry

end whisky_replacement_l574_57411


namespace ball_picking_problem_proof_l574_57429

-- Define the conditions
def red_balls : ℕ := 8
def white_balls : ℕ := 7

-- Define the questions
def num_ways_to_pick_one_ball : ℕ :=
  red_balls + white_balls

def num_ways_to_pick_two_different_color_balls : ℕ :=
  red_balls * white_balls

-- Define the correct answers
def correct_answer_to_pick_one_ball : ℕ := 15
def correct_answer_to_pick_two_different_color_balls : ℕ := 56

-- State the theorem to be proved
theorem ball_picking_problem_proof :
  (num_ways_to_pick_one_ball = correct_answer_to_pick_one_ball) ∧
  (num_ways_to_pick_two_different_color_balls = correct_answer_to_pick_two_different_color_balls) :=
by
  sorry

end ball_picking_problem_proof_l574_57429


namespace ribbon_length_difference_l574_57449

theorem ribbon_length_difference (S : ℝ) : 
  let Seojun_ribbon := S 
  let Siwon_ribbon := S + 8.8 
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3 
  Siwon_new - Seojun_new = 17.4 :=
by
  -- Definition of original ribbon lengths
  let Seojun_ribbon := S
  let Siwon_ribbon := S + 8.8
  -- Seojun cuts and gives 4.3 meters to Siwon
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3
  -- Compute the difference
  have h1 : Siwon_new - Seojun_new = (S + 8.8 + 4.3) - (S - 4.3) := by sorry
  -- Prove the final answer
  have h2 : Siwon_new - Seojun_new = 17.4 := by sorry

  exact h2

end ribbon_length_difference_l574_57449


namespace problem_statement_l574_57408

theorem problem_statement (h: 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := 
by {
  sorry
}

end problem_statement_l574_57408


namespace carrie_mom_money_l574_57479

theorem carrie_mom_money :
  ∀ (sweater_cost t_shirt_cost shoes_cost left_money total_money : ℕ),
  sweater_cost = 24 →
  t_shirt_cost = 6 →
  shoes_cost = 11 →
  left_money = 50 →
  total_money = sweater_cost + t_shirt_cost + shoes_cost + left_money →
  total_money = 91 :=
sorry

end carrie_mom_money_l574_57479


namespace xyz_neg_of_ineq_l574_57495

variables {x y z : ℝ}

theorem xyz_neg_of_ineq
  (h1 : 2 * x - y < 0)
  (h2 : 3 * y - 2 * z < 0)
  (h3 : 4 * z - 3 * x < 0) :
  x < 0 ∧ y < 0 ∧ z < 0 :=
sorry

end xyz_neg_of_ineq_l574_57495


namespace ratio_of_investments_l574_57471

theorem ratio_of_investments (I B_profit total_profit : ℝ) (x : ℝ)
  (h1 : B_profit = 4000) (h2 : total_profit = 28000) (h3 : I * (2 * B_profit / 4000 - 1) = total_profit - B_profit) :
  x = 3 :=
by
  sorry

end ratio_of_investments_l574_57471


namespace birds_flew_up_count_l574_57402

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end birds_flew_up_count_l574_57402


namespace monotonically_increasing_interval_l574_57403

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.log (1/2)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Iio (0 : ℝ) → StrictMono f :=
by
  sorry

end monotonically_increasing_interval_l574_57403


namespace track_champion_races_l574_57439

theorem track_champion_races (total_sprinters : ℕ) (lanes : ℕ) (eliminations_per_race : ℕ)
  (h1 : total_sprinters = 216) (h2 : lanes = 6) (h3 : eliminations_per_race = 5) : 
  (total_sprinters - 1) / eliminations_per_race = 43 :=
by
  -- We acknowledge that a proof is needed here. Placeholder for now.
  sorry

end track_champion_races_l574_57439


namespace negative_remainder_l574_57463

theorem negative_remainder (a : ℤ) (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end negative_remainder_l574_57463


namespace intersection_M_N_eq_M_l574_57489

-- Definitions of M and N
def M : Set ℝ := { x : ℝ | x^2 - x < 0 }
def N : Set ℝ := { x : ℝ | abs x < 2 }

-- Proof statement
theorem intersection_M_N_eq_M : M ∩ N = M := 
  sorry

end intersection_M_N_eq_M_l574_57489


namespace yellow_tint_percent_l574_57432

theorem yellow_tint_percent (total_volume: ℕ) (initial_yellow_percent: ℚ) (yellow_added: ℕ) (answer: ℚ) 
  (h_initial_total: total_volume = 20) 
  (h_initial_yellow: initial_yellow_percent = 0.50) 
  (h_yellow_added: yellow_added = 6) 
  (h_answer: answer = 61.5): 
  (yellow_added + initial_yellow_percent * total_volume) / (total_volume + yellow_added) * 100 = answer := 
by 
  sorry

end yellow_tint_percent_l574_57432


namespace find_x_value_l574_57467

theorem find_x_value (x : ℝ) (h : (7 / (x - 2) + x / (2 - x) = 4)) : x = 3 :=
sorry

end find_x_value_l574_57467


namespace value_of_expression_l574_57487

theorem value_of_expression (x : ℝ) (h : |x| = x + 2) : 19 * x ^ 99 + 3 * x + 27 = 5 :=
by
  have h1: x ≥ -2 := sorry
  have h2: x = -1 := sorry
  sorry

end value_of_expression_l574_57487


namespace multiple_of_age_is_3_l574_57474

def current_age : ℕ := 9
def age_six_years_ago : ℕ := 3
def age_multiple (current : ℕ) (previous : ℕ) : ℕ := current / previous

theorem multiple_of_age_is_3 : age_multiple current_age age_six_years_ago = 3 :=
by
  sorry

end multiple_of_age_is_3_l574_57474


namespace quadratic_one_root_greater_than_two_other_less_than_two_l574_57498

theorem quadratic_one_root_greater_than_two_other_less_than_two (m : ℝ) :
  (∀ x y : ℝ, x^2 + (2 * m - 3) * x + m - 150 = 0 ∧ x > 2 ∧ y < 2) →
  m > 5 :=
by
  sorry

end quadratic_one_root_greater_than_two_other_less_than_two_l574_57498


namespace amount_B_l574_57428

noncomputable def A : ℝ := sorry -- Definition of A
noncomputable def B : ℝ := sorry -- Definition of B

-- Conditions
def condition1 : Prop := A + B = 100
def condition2 : Prop := (3 / 10) * A = (1 / 5) * B

-- Statement to prove
theorem amount_B : condition1 ∧ condition2 → B = 60 :=
by
  intros
  sorry

end amount_B_l574_57428


namespace factorization_identity_l574_57443

variable (a b : ℝ)

theorem factorization_identity : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end factorization_identity_l574_57443


namespace max_value_of_f_l574_57451

noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 6/5 := by
  sorry

end max_value_of_f_l574_57451


namespace solve_for_r_l574_57444

theorem solve_for_r (r s : ℚ) (h : (2 * (r - 45)) / 3 = (3 * s - 2 * r) / 4) (s_val : s = 20) :
  r = 270 / 7 :=
by
  sorry

end solve_for_r_l574_57444


namespace smallest_positive_period_max_min_values_l574_57431

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2 - 1 / 2

-- Theorem 1: Smallest positive period of the function f(x)
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
  sorry

-- Theorem 2: Maximum and minimum values of the function f(x) on [0, π/2]
theorem max_min_values : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x ≤ 1 ∧ f x ≥ -1 / 2 ∧ (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_max = 1) ∧
    (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_min = -1 / 2) :=
  sorry

end smallest_positive_period_max_min_values_l574_57431


namespace jessica_withdrawal_l574_57409

/-- Jessica withdrew some money from her bank account, causing her account balance to decrease by 2/5.
    She then deposited an amount equal to 1/4 of the remaining balance. The final balance in her bank account is $750.
    Prove that Jessica initially withdrew $400. -/
theorem jessica_withdrawal (X W : ℝ) 
  (initial_eq : W = (2 / 5) * X)
  (remaining_eq : X * (3 / 5) + (1 / 4) * (X * (3 / 5)) = 750) :
  W = 400 := 
sorry

end jessica_withdrawal_l574_57409


namespace truncated_cone_radius_l574_57494

theorem truncated_cone_radius (R: ℝ) (l: ℝ) (h: 0 < l)
  (h1 : ∃ (r: ℝ), r = (R + 5) / 2 ∧ (5 + r) = (1 / 2) * (R + r))
  : R = 25 :=
sorry

end truncated_cone_radius_l574_57494


namespace sequence_a19_l574_57419

theorem sequence_a19 :
  ∃ (a : ℕ → ℝ), a 3 = 2 ∧ a 7 = 1 ∧
    (∃ d : ℝ, ∀ n m : ℕ, (1 / (a n + 1) - 1 / (a m + 1)) / (n - m) = d) →
    a 19 = 0 :=
by sorry

end sequence_a19_l574_57419


namespace vermont_clicked_ads_l574_57446

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end vermont_clicked_ads_l574_57446


namespace problem_solution_l574_57441

theorem problem_solution (x y : ℝ) (h₁ : x + Real.cos y = 2010) (h₂ : x + 2010 * Real.sin y = 2011) (h₃ : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := 
sorry

end problem_solution_l574_57441


namespace problem_X_plus_Y_l574_57472

def num_five_digit_even_numbers : Nat := 45000
def num_five_digit_multiples_of_7 : Nat := 12857
def X := num_five_digit_even_numbers
def Y := num_five_digit_multiples_of_7

theorem problem_X_plus_Y : X + Y = 57857 :=
by
  sorry

end problem_X_plus_Y_l574_57472


namespace divisor_and_remainder_correct_l574_57465

theorem divisor_and_remainder_correct:
  ∃ d r : ℕ, d ≠ 0 ∧ 1270 = 74 * d + r ∧ r = 12 ∧ d = 17 :=
by
  sorry

end divisor_and_remainder_correct_l574_57465


namespace part_one_part_two_l574_57440

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 4 - x

-- Problem set (I)
theorem part_one (x : ℝ) : inequality_condition x ↔ (x ≤ -3 ∨ x ≥ 1) :=
sorry

-- Define range conditions for a and b
def range_condition (a b : ℝ) : Prop := a ≥ 3 ∧ b ≥ 3

-- Problem set (II)
theorem part_two (a b : ℝ) (h : range_condition a b) : 2 * (a + b) < a * b + 4 :=
sorry

end part_one_part_two_l574_57440


namespace find_acute_angles_of_alex_triangle_l574_57415

theorem find_acute_angles_of_alex_triangle (α : ℝ) (h1 : α > 0) (h2 : α < 90) :
  let condition1 := «Alex drew a geometric picture by tracing his plastic right triangle four times»
  let condition2 := «Each time aligning the shorter leg with the hypotenuse and matching the vertex of the acute angle with the vertex of the right angle»
  let condition3 := «The "closing" fifth triangle was isosceles»
  α = 90 / 11 :=
sorry

end find_acute_angles_of_alex_triangle_l574_57415


namespace people_joined_group_l574_57445

theorem people_joined_group (x y : ℕ) (h1 : 1430 = 22 * x) (h2 : 1430 = 13 * (x + y)) : y = 45 := 
by 
  -- This is just the statement, so we add sorry to skip the proof
  sorry

end people_joined_group_l574_57445


namespace baseball_league_games_l574_57499

theorem baseball_league_games (n m : ℕ) (h : 3 * n + 4 * m = 76) (h1 : n > 2 * m) (h2 : m > 4) : n = 16 :=
by 
  sorry

end baseball_league_games_l574_57499


namespace boxes_of_apples_l574_57436

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end boxes_of_apples_l574_57436


namespace learning_machine_price_reduction_l574_57473

theorem learning_machine_price_reduction (x : ℝ) (h1 : 2000 * (1 - x) * (1 - x) = 1280) : 2000 * (1 - x)^2 = 1280 :=
by
  sorry

end learning_machine_price_reduction_l574_57473


namespace maximize_garden_area_l574_57453

def optimal_dimensions_area : Prop :=
  let l := 100
  let w := 60
  let area := 6000
  (2 * l) + (2 * w) = 320 ∧ l >= 100 ∧ (l * w) = area

theorem maximize_garden_area : optimal_dimensions_area := by
  sorry

end maximize_garden_area_l574_57453


namespace rectangle_perimeter_l574_57438

theorem rectangle_perimeter (s : ℕ) (h : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end rectangle_perimeter_l574_57438


namespace modulus_product_eq_sqrt_5_l574_57450

open Complex

-- Define the given complex number.
def z : ℂ := 2 + I

-- Declare the product with I.
def z_product := z * I

-- State the theorem that the modulus of the product is sqrt(5).
theorem modulus_product_eq_sqrt_5 : abs z_product = Real.sqrt 5 := 
sorry

end modulus_product_eq_sqrt_5_l574_57450


namespace at_least_one_prob_better_option_l574_57435

-- Definitions based on the conditions in a)

def player_A_prelim := 1 / 2
def player_B_prelim := 1 / 3
def player_C_prelim := 1 / 2

def final_round := 1 / 3

def prelim_prob_A := player_A_prelim * final_round
def prelim_prob_B := player_B_prelim * final_round
def prelim_prob_C := player_C_prelim * final_round

def prob_none := (1 - prelim_prob_A) * (1 - prelim_prob_B) * (1 - prelim_prob_C)

def prob_at_least_one := 1 - prob_none

-- Question 1 statement

theorem at_least_one_prob :
  prob_at_least_one = 31 / 81 :=
sorry

-- Definitions based on the reward options in the conditions

def option_1_lottery_prob := 1 / 3
def option_1_reward := 600
def option_1_expected_value := 600 * 3 * (1 / 3)

def option_2_prelim_reward := 100
def option_2_final_reward := 400

-- Expected values calculation for Option 2

def option_2_expected_value :=
  (300 * (1 / 6) + 600 * (5 / 12) + 900 * (1 / 3) + 1200 * (1 / 12))

-- Question 2 statement

theorem better_option :
  option_1_expected_value < option_2_expected_value :=
sorry

end at_least_one_prob_better_option_l574_57435


namespace pure_imaginary_a_l574_57426

theorem pure_imaginary_a (a : ℝ) :
  (a^2 - 4 = 0) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
by
  sorry

end pure_imaginary_a_l574_57426


namespace number_is_100_l574_57481

theorem number_is_100 (n : ℕ) 
  (hquot : n / 11 = 9) 
  (hrem : n % 11 = 1) : 
  n = 100 := 
by 
  sorry

end number_is_100_l574_57481


namespace find_initial_population_l574_57430

theorem find_initial_population
  (birth_rate : ℕ)
  (death_rate : ℕ)
  (net_growth_rate_percent : ℝ)
  (net_growth_rate_per_person : ℕ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate_percent = 2.1)
  (h4 : net_growth_rate_per_person = birth_rate - death_rate)
  (h5 : (net_growth_rate_per_person : ℝ) / 100 = net_growth_rate_percent / 100) :
  P = 1000 :=
by
  sorry

end find_initial_population_l574_57430


namespace andrew_start_age_l574_57461

-- Define the conditions
def annual_donation : ℕ := 7
def current_age : ℕ := 29
def total_donation : ℕ := 133

-- The theorem to prove
theorem andrew_start_age : (total_donation / annual_donation) = (current_age - 10) :=
by
  sorry

end andrew_start_age_l574_57461


namespace symmetric_point_x_axis_l574_57414

theorem symmetric_point_x_axis (P Q : ℝ × ℝ) (hP : P = (-1, 2)) (hQ : Q = (P.1, -P.2)) : Q = (-1, -2) :=
sorry

end symmetric_point_x_axis_l574_57414


namespace neither_necessary_nor_sufficient_l574_57470

theorem neither_necessary_nor_sufficient (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  ¬(∀ a b, (a > b → (1 / a < 1 / b)) ∧ ((1 / a < 1 / b) → a > b)) := sorry

end neither_necessary_nor_sufficient_l574_57470


namespace value_of_c_l574_57452

theorem value_of_c
    (x y c : ℝ)
    (h1 : 3 * x - 5 * y = 5)
    (h2 : x / (x + y) = c)
    (h3 : x - y = 2.999999999999999) :
    c = 0.7142857142857142 :=
by
    sorry

end value_of_c_l574_57452


namespace Brittany_second_test_grade_is_83_l574_57437

theorem Brittany_second_test_grade_is_83
  (first_test_score : ℝ) (first_test_weight : ℝ) 
  (second_test_weight : ℝ) (final_weighted_average : ℝ) : 
  first_test_score = 78 → 
  first_test_weight = 0.40 →
  second_test_weight = 0.60 →
  final_weighted_average = 81 →
  ∃ G : ℝ, 0.40 * first_test_score + 0.60 * G = final_weighted_average ∧ G = 83 :=
by
  sorry

end Brittany_second_test_grade_is_83_l574_57437


namespace students_like_all_three_l574_57482

variables (N : ℕ) (r : ℚ) (j : ℚ) (o : ℕ) (n : ℕ)

-- Number of students in the class
def num_students := N = 40

-- Fraction of students who like Rock
def fraction_rock := r = 1/4

-- Fraction of students who like Jazz
def fraction_jazz := j = 1/5

-- Number of students who like other genres
def num_other_genres := o = 8

-- Number of students who do not like any of the three genres
def num_no_genres := n = 6

---- Proof theorem
theorem students_like_all_three
  (h1 : num_students N)
  (h2 : fraction_rock r)
  (h3 : fraction_jazz j)
  (h4 : num_other_genres o)
  (h5 : num_no_genres n) :
  ∃ z : ℕ, z = 2 := 
sorry

end students_like_all_three_l574_57482


namespace find_prime_n_l574_57413

theorem find_prime_n (n k m : ℤ) (h1 : n - 6 = k ^ 2) (h2 : n + 10 = m ^ 2) (h3 : m ^ 2 - k ^ 2 = 16) (h4 : Nat.Prime (Int.natAbs n)) : n = 71 := by
  sorry

end find_prime_n_l574_57413


namespace percentage_increase_l574_57460

theorem percentage_increase (x y P : ℚ)
  (h1 : x = 0.9 * y)
  (h2 : x = 123.75)
  (h3 : y = 125 + 1.25 * P) : 
  P = 10 := 
by 
  sorry

end percentage_increase_l574_57460


namespace num_perpendicular_line_plane_pairs_in_cube_l574_57424

-- Definitions based on the problem conditions

def is_perpendicular_line_plane_pair (l : line) (p : plane) : Prop :=
  -- Assume an implementation that defines when a line is perpendicular to a plane
  sorry

-- Define a cube structure with its vertices, edges, and faces
structure Cube :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (faces : Finset (Finset Point))

-- Make assumptions about cube properties
variable (cube : Cube)

-- Define the property of counting perpendicular line-plane pairs
def count_perpendicular_line_plane_pairs (c : Cube) : Nat :=
  -- Assume an implementation that counts the number of such pairs in the cube
  sorry

-- The theorem to prove
theorem num_perpendicular_line_plane_pairs_in_cube (c : Cube) :
  count_perpendicular_line_plane_pairs c = 36 :=
  sorry

end num_perpendicular_line_plane_pairs_in_cube_l574_57424


namespace min_value_of_expression_l574_57455

theorem min_value_of_expression (a : ℝ) (h₀ : a > 0)
  (x₁ x₂ : ℝ)
  (h₁ : x₁ + x₂ = 4 * a)
  (h₂ : x₁ * x₂ = a * a) :
  x₁ + x₂ + a / (x₁ * x₂) = 4 :=
sorry

end min_value_of_expression_l574_57455


namespace consecutive_numbers_square_sum_l574_57478

theorem consecutive_numbers_square_sum (n : ℕ) (a b : ℕ) (h1 : 2 * n + 1 = 144169^2)
  (h2 : a = 72084) (h3 : b = a + 1) : a^2 + b^2 = n + 1 :=
by
  sorry

end consecutive_numbers_square_sum_l574_57478


namespace three_digit_sum_27_l574_57475

theorem three_digit_sum_27 {a b c : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  a + b + c = 27 → (a, b, c) = (9, 9, 9) :=
by
  sorry

end three_digit_sum_27_l574_57475


namespace greatest_divisor_of_product_of_four_consecutive_integers_l574_57423

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l574_57423


namespace part_a_part_b_l574_57457

-- Part (a): Number of ways to distribute 20 identical balls into 6 boxes so that no box is empty
theorem part_a:
  ∃ (n : ℕ), n = Nat.choose 19 5 :=
sorry

-- Part (b): Number of ways to distribute 20 identical balls into 6 boxes if some boxes can be empty
theorem part_b:
  ∃ (n : ℕ), n = Nat.choose 25 5 :=
sorry

end part_a_part_b_l574_57457


namespace sum_of_interior_angles_hexagon_l574_57493

theorem sum_of_interior_angles_hexagon : 
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_hexagon_l574_57493


namespace find_number_subtracted_l574_57486

theorem find_number_subtracted (x : ℕ) (h : 88 - x = 54) : x = 34 := by
  sorry

end find_number_subtracted_l574_57486


namespace no_such_xy_between_988_and_1991_l574_57412

theorem no_such_xy_between_988_and_1991 :
  ¬ ∃ (x y : ℕ), 988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧ 
  (∃ a b : ℕ, xy = x * y ∧ (xy + x = a^2 ∧ xy + y = b^2)) :=
by
  sorry

end no_such_xy_between_988_and_1991_l574_57412


namespace calculate_rows_l574_57484

-- Definitions based on conditions
def totalPecanPies : ℕ := 16
def totalApplePies : ℕ := 14
def piesPerRow : ℕ := 5

-- The goal is to prove the total rows of pies
theorem calculate_rows : (totalPecanPies + totalApplePies) / piesPerRow = 6 := by
  sorry

end calculate_rows_l574_57484


namespace chips_in_bag_l574_57491

theorem chips_in_bag :
  let initial_chips := 5
  let additional_chips := 5
  let daily_chips := 10
  let total_days := 10
  let first_day_chips := initial_chips + additional_chips
  let remaining_days := total_days - 1
  (first_day_chips + remaining_days * daily_chips) = 100 :=
by
  sorry

end chips_in_bag_l574_57491


namespace average_production_n_days_l574_57417

theorem average_production_n_days (n : ℕ) (P : ℕ) 
  (hP : P = 80 * n)
  (h_new_avg : (P + 220) / (n + 1) = 95) : 
  n = 8 := 
by
  sorry -- Proof of the theorem

end average_production_n_days_l574_57417


namespace a_2016_is_1_l574_57447

noncomputable def seq_a (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * b n

theorem a_2016_is_1 (a b : ℕ → ℝ)
  (h1 : a 1 = 1)
  (hb : seq_a a b)
  (h3 : b 1008 = 1) :
  a 2016 = 1 :=
sorry

end a_2016_is_1_l574_57447


namespace ratio_AB_CD_lengths_AB_CD_l574_57448

-- Given conditions as definitions
def ABD_triangle (A B D : Point) : Prop := true  -- In quadrilateral ABCD, a diagonal BD is drawn
def BCD_triangle (B C D : Point) : Prop := true  -- Circles are inscribed in triangles ABD and BCD
def Line_through_B_center_AM_M (A B D M : Point) (AM MD : ℚ) : Prop :=
  (AM = 8/5) ∧ (MD = 12/5)
def Line_through_D_center_BN_N (B C D N : Point) (BN NC : ℚ) : Prop :=
  (BN = 30/11) ∧ (NC = 25/11)

-- Mathematically equivalent proof problems
theorem ratio_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB / CD = 4 / 5 :=
by
  sorry

theorem lengths_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB + CD = 9 ∧
  AB - CD = -1 :=
by 
  sorry

end ratio_AB_CD_lengths_AB_CD_l574_57448


namespace arithmetic_sequence_formula_geometric_sequence_sum_formula_l574_57405

noncomputable def arithmetic_sequence_a_n (n : ℕ) : ℤ :=
  sorry

noncomputable def geometric_sequence_T_n (n : ℕ) : ℤ :=
  sorry

theorem arithmetic_sequence_formula :
  (∃ a₃ : ℤ, a₃ = 5) ∧ (∃ S₃ : ℤ, S₃ = 9) →
  -- Suppose we have an arithmetic sequence $a_n$
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence_a_n n = 2 * n - 1) := 
sorry

theorem geometric_sequence_sum_formula :
  (∃ q : ℤ, q > 0 ∧ q = 3) ∧ (∃ b₃ : ℤ, b₃ = 9) ∧ (∃ T₃ : ℤ, T₃ = 13) →
  -- Suppose we have a geometric sequence $b_n$ where $b_3 = a_5$
  (∀ n : ℕ, n ≥ 1 → geometric_sequence_T_n n = (3 ^ n - 1) / 2) := 
sorry

end arithmetic_sequence_formula_geometric_sequence_sum_formula_l574_57405


namespace calorie_limit_l574_57433

variable (breakfastCalories lunchCalories dinnerCalories extraCalories : ℕ)
variable (plannedCalories : ℕ)

-- Given conditions
axiom breakfast_calories : breakfastCalories = 400
axiom lunch_calories : lunchCalories = 900
axiom dinner_calories : dinnerCalories = 1100
axiom extra_calories : extraCalories = 600

-- To Prove
theorem calorie_limit (h : plannedCalories = (breakfastCalories + lunchCalories + dinnerCalories - extraCalories)) :
  plannedCalories = 1800 := by sorry

end calorie_limit_l574_57433


namespace unique_handshakes_462_l574_57406

theorem unique_handshakes_462 : 
  ∀ (twins triplets : Type) (twin_set : ℕ) (triplet_set : ℕ) (handshakes_among_twins handshakes_among_triplets cross_handshakes_twins cross_handshakes_triplets : ℕ),
  twin_set = 12 ∧
  triplet_set = 4 ∧
  handshakes_among_twins = (24 * 22) / 2 ∧
  handshakes_among_triplets = (12 * 9) / 2 ∧
  cross_handshakes_twins = 24 * (12 / 3) ∧
  cross_handshakes_triplets = 12 * (24 / 3 * 2) →
  (handshakes_among_twins + handshakes_among_triplets + (cross_handshakes_twins + cross_handshakes_triplets) / 2) = 462 := 
by
  sorry

end unique_handshakes_462_l574_57406


namespace octagon_diag_20_algebraic_expr_positive_l574_57477

def octagon_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diag_20 : octagon_diagonals 8 = 20 := by
  -- Formula for diagonals is used here
  sorry

theorem algebraic_expr_positive (x : ℝ) : 2 * x^2 - 2 * x + 1 > 0 := by
  -- Complete the square to show it's always positive
  sorry

end octagon_diag_20_algebraic_expr_positive_l574_57477
