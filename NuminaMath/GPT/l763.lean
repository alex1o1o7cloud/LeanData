import Mathlib

namespace lucy_money_l763_76342

variable (L : ℕ) -- Value for Lucy's original amount of money

theorem lucy_money (h1 : ∀ (L : ℕ), L - 5 = 10 + 5 → L = 20) : L = 20 :=
by sorry

end lucy_money_l763_76342


namespace three_digit_number_second_digit_l763_76352

theorem three_digit_number_second_digit (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (100 * a + 10 * b + c) - (a + b + c) = 261 → b = 7 :=
by sorry

end three_digit_number_second_digit_l763_76352


namespace max_leap_years_in_200_years_l763_76357

theorem max_leap_years_in_200_years (leap_year_interval: ℕ) (span: ℕ) 
  (h1: leap_year_interval = 4) 
  (h2: span = 200) : 
  (span / leap_year_interval) = 50 := 
sorry

end max_leap_years_in_200_years_l763_76357


namespace minimize_expression_l763_76312

theorem minimize_expression (n : ℕ) (h : 0 < n) : 
  (n = 10) ↔ (∀ m : ℕ, 0 < m → ((n / 2) + (50 / n) ≤ (m / 2) + (50 / m))) :=
sorry

end minimize_expression_l763_76312


namespace triangle_angle_sum_l763_76358

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l763_76358


namespace average_height_31_students_l763_76309

theorem average_height_31_students (avg1 avg2 : ℝ) (n1 n2 : ℕ) (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) : ((avg1 * n1 + avg2 * n2) / (n1 + n2)) = 20 :=
by
  sorry

end average_height_31_students_l763_76309


namespace distance_between_A_and_B_is_750_l763_76333

def original_speed := 150 -- derived from the solution

def distance (S D : ℝ) :=
  (D / S) - (D / ((5 / 4) * S)) = 1 ∧
  ((D - 150) / S) - ((5 * (D - 150)) / (6 * S)) = 2 / 3

theorem distance_between_A_and_B_is_750 :
  ∃ D : ℝ, distance original_speed D ∧ D = 750 :=
by
  sorry

end distance_between_A_and_B_is_750_l763_76333


namespace average_speed_is_37_5_l763_76345

-- Define the conditions
def distance_local : ℕ := 60
def speed_local : ℕ := 30
def distance_gravel : ℕ := 10
def speed_gravel : ℕ := 20
def distance_highway : ℕ := 105
def speed_highway : ℕ := 60
def traffic_delay : ℚ := 15 / 60
def obstruction_delay : ℚ := 10 / 60

-- Define the total distance
def total_distance : ℕ := distance_local + distance_gravel + distance_highway

-- Define the total time
def total_time : ℚ :=
  (distance_local / speed_local) +
  (distance_gravel / speed_gravel) +
  (distance_highway / speed_highway) +
  traffic_delay +
  obstruction_delay

-- Define the average speed as distance divided by time
def average_speed : ℚ := total_distance / total_time

theorem average_speed_is_37_5 :
  average_speed = 37.5 := by sorry

end average_speed_is_37_5_l763_76345


namespace total_operations_in_one_hour_l763_76347

theorem total_operations_in_one_hour :
  let additions_per_second := 12000
  let multiplications_per_second := 8000
  (additions_per_second + multiplications_per_second) * 3600 = 72000000 :=
by
  sorry

end total_operations_in_one_hour_l763_76347


namespace compare_A_B_l763_76323

-- Definitions based on conditions from part a)
def A (n : ℕ) : ℕ := 2 * n^2
def B (n : ℕ) : ℕ := 3^n

-- The theorem that needs to be proven
theorem compare_A_B (n : ℕ) (h : n > 0) : A n < B n := 
by sorry

end compare_A_B_l763_76323


namespace number_of_white_balls_l763_76388

theorem number_of_white_balls (x : ℕ) (h : (5 : ℚ) / (5 + x) = 1 / 4) : x = 15 :=
by 
  sorry

end number_of_white_balls_l763_76388


namespace martin_total_distance_l763_76386

theorem martin_total_distance (T S1 S2 t : ℕ) (hT : T = 8) (hS1 : S1 = 70) (hS2 : S2 = 85) (ht : t = T / 2) : S1 * t + S2 * t = 620 := 
by
  sorry

end martin_total_distance_l763_76386


namespace quadratic_inequality_solution_l763_76317

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 3 * x - 18 > 0) ↔ (x < -6 ∨ x > 3) := 
sorry

end quadratic_inequality_solution_l763_76317


namespace sum_of_numbers_l763_76381

theorem sum_of_numbers (x : ℝ) (h1 : x^2 + (2 * x)^2 + (4 * x)^2 = 1701) : x + 2 * x + 4 * x = 63 :=
sorry

end sum_of_numbers_l763_76381


namespace abs_neg_eight_l763_76349

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l763_76349


namespace second_athlete_triple_jump_l763_76367

theorem second_athlete_triple_jump
  (long_jump1 triple_jump1 high_jump1 : ℕ) 
  (long_jump2 high_jump2 : ℕ)
  (average_winner : ℕ) 
  (H1 : long_jump1 = 26) (H2 : triple_jump1 = 30) (H3 : high_jump1 = 7)
  (H4 : long_jump2 = 24) (H5 : high_jump2 = 8) (H6 : average_winner = 22)
  : ∃ x : ℕ, (24 + x + 8) / 3 = 22 ∧ x = 34 := 
by
  sorry

end second_athlete_triple_jump_l763_76367


namespace cabinets_ratio_proof_l763_76362

-- Definitions for the conditions
def initial_cabinets : ℕ := 3
def total_cabinets : ℕ := 26
def additional_cabinets : ℕ := 5
def number_of_counters : ℕ := 3

-- Definition for the unknown cabinets installed per counter
def cabinets_per_counter : ℕ := (total_cabinets - additional_cabinets - initial_cabinets) / number_of_counters

-- The ratio to be proven
theorem cabinets_ratio_proof : (cabinets_per_counter : ℚ) / initial_cabinets = 2 / 1 :=
by
  -- Proof goes here
  sorry

end cabinets_ratio_proof_l763_76362


namespace combined_annual_income_eq_correct_value_l763_76320

theorem combined_annual_income_eq_correct_value :
  let A_income := 5 / 2 * 17000
  let B_income := 1.12 * 17000
  let C_income := 17000
  let D_income := 0.85 * A_income
  (A_income + B_income + C_income + D_income) * 12 = 1375980 :=
by
  sorry

end combined_annual_income_eq_correct_value_l763_76320


namespace part1_part2_l763_76301

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≥ 3 * x + 2) : x ≥ 3 ∨ x ≤ -1 :=
sorry

-- Part (2)
theorem part2 (h : ∀ x, f x a ≤ 0 → x ≤ -1) : a = 2 :=
sorry

end part1_part2_l763_76301


namespace MarthaEndBlocks_l763_76379

theorem MarthaEndBlocks (start_blocks found_blocks total_blocks : ℕ) 
  (h₁ : start_blocks = 11)
  (h₂ : found_blocks = 129) : 
  total_blocks = 140 :=
by
  sorry

end MarthaEndBlocks_l763_76379


namespace total_yellow_balloons_l763_76346

theorem total_yellow_balloons (n_tom : ℕ) (n_sara : ℕ) (h_tom : n_tom = 9) (h_sara : n_sara = 8) : n_tom + n_sara = 17 :=
by
  sorry

end total_yellow_balloons_l763_76346


namespace cos_evaluation_l763_76325

open Real

noncomputable def a (n : ℕ) : ℝ := sorry  -- since it's an arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, a n + a k = 2 * a ((n + k) / 2)

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 6 + a 9 = 3 * a 6 ∧ a 6 = π / 4

theorem cos_evaluation :
  is_arithmetic_sequence a →
  satisfies_condition a →
  cos (a 2 + a 10 + π / 4) = - (sqrt 2 / 2) :=
by
  intros
  sorry

end cos_evaluation_l763_76325


namespace rectangle_dimensions_exist_l763_76334

theorem rectangle_dimensions_exist :
  ∃ (a b c d : ℕ), (a * b + c * d = 81) ∧ (2 * (a + b) = 2 * 2 * (c + d) ∨ 2 * (c + d) = 2 * 2 * (a + b)) :=
by sorry

end rectangle_dimensions_exist_l763_76334


namespace value_of_a_for_positive_root_l763_76370

theorem value_of_a_for_positive_root :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ (3*x - 1)/(x - 3) = a/(3 - x) - 1) → a = -8 :=
by
  sorry

end value_of_a_for_positive_root_l763_76370


namespace smallest_three_digit_integer_l763_76369

theorem smallest_three_digit_integer (n : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧ ¬ (n - 1 ∣ (n!)) ↔ n = 1004 := 
by
  sorry

end smallest_three_digit_integer_l763_76369


namespace quadratic_inequality_ab_l763_76330

theorem quadratic_inequality_ab (a b : ℝ) :
  (∀ x : ℝ, (x > -1 ∧ x < 1 / 3) → a * x^2 + b * x + 1 > 0) →
  a * b = 6 :=
sorry

end quadratic_inequality_ab_l763_76330


namespace sum_of_first_9_terms_arithmetic_sequence_l763_76303

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem sum_of_first_9_terms_arithmetic_sequence
  (h_arith_seq : is_arithmetic_sequence a)
  (h_condition : a 2 + a 8 = 8) :
  (Finset.range 9).sum a = 36 :=
sorry

end sum_of_first_9_terms_arithmetic_sequence_l763_76303


namespace multiple_choice_question_count_l763_76394

theorem multiple_choice_question_count (n : ℕ) : 
  (4 * 224 / (2^4 - 2) = 4^2) → n = 2 := 
by
  sorry

end multiple_choice_question_count_l763_76394


namespace proof_fraction_problem_l763_76382

def fraction_problem :=
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75

theorem proof_fraction_problem : fraction_problem :=
by
  sorry

end proof_fraction_problem_l763_76382


namespace ratio_second_to_first_l763_76372

theorem ratio_second_to_first (F S T : ℕ) 
  (hT : T = 2 * F)
  (havg : (F + S + T) / 3 = 77)
  (hmin : F = 33) :
  S / F = 4 :=
by
  sorry

end ratio_second_to_first_l763_76372


namespace seeds_per_packet_l763_76371

theorem seeds_per_packet (total_seedlings packets : ℕ) (h1 : total_seedlings = 420) (h2 : packets = 60) : total_seedlings / packets = 7 :=
by 
  sorry

end seeds_per_packet_l763_76371


namespace Scruffy_weight_l763_76344

variable {Muffy Puffy Scruffy : ℝ}

def Puffy_weight_condition (Muffy Puffy : ℝ) : Prop := Puffy = Muffy + 5
def Scruffy_weight_condition (Muffy Scruffy : ℝ) : Prop := Scruffy = Muffy + 3
def Combined_weight_condition (Muffy Puffy : ℝ) : Prop := Muffy + Puffy = 23

theorem Scruffy_weight (h1 : Puffy_weight_condition Muffy Puffy) (h2 : Scruffy_weight_condition Muffy Scruffy) (h3 : Combined_weight_condition Muffy Puffy) : Scruffy = 12 := by
  sorry

end Scruffy_weight_l763_76344


namespace largest_number_among_options_l763_76396

def option_a : ℝ := -abs (-4)
def option_b : ℝ := 0
def option_c : ℝ := 1
def option_d : ℝ := -( -3)

theorem largest_number_among_options : 
  max (max option_a (max option_b option_c)) option_d = option_d := by
  sorry

end largest_number_among_options_l763_76396


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l763_76339

variables {m : ℝ}

-- (1) For z to be a real number
theorem real_number_condition : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) :=
by sorry

-- (2) For z to be an imaginary number
theorem imaginary_number_condition : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) :=
by sorry

-- (3) For z to be a purely imaginary number
theorem pure_imaginary_number_condition : (m^2 - 5 * m + 6 = 0 ∧ m^2 - 3 * m ≠ 0) ↔ (m = 2) :=
by sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l763_76339


namespace unique_solution_iff_a_values_l763_76337

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 5 * a

theorem unique_solution_iff_a_values (a : ℝ) :
  (∃! x : ℝ, |f x a| ≤ 3) ↔ (a = 3 / 4 ∨ a = -3 / 4) :=
by
  sorry

end unique_solution_iff_a_values_l763_76337


namespace school_fitness_event_participants_l763_76384

theorem school_fitness_event_participants :
  let p0 := 500 -- initial number of participants in 2000
  let r1 := 0.3 -- increase rate in 2001
  let r2 := 0.4 -- increase rate in 2002
  let r3 := 0.5 -- increase rate in 2003
  let p1 := p0 * (1 + r1) -- participants in 2001
  let p2 := p1 * (1 + r2) -- participants in 2002
  let p3 := p2 * (1 + r3) -- participants in 2003
  p3 = 1365 -- prove that number of participants in 2003 is 1365
:= sorry

end school_fitness_event_participants_l763_76384


namespace express_x_in_terms_of_y_l763_76389

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : x = 7 / 2 + 3 / 2 * y :=
by
  sorry

end express_x_in_terms_of_y_l763_76389


namespace maximum_third_height_l763_76321

theorem maximum_third_height 
  (A B C : Type)
  (h1 h2 : ℕ)
  (h1_pos : h1 = 4) 
  (h2_pos : h2 = 12) 
  (h3_pos : ℕ)
  (triangle_inequality : ∀ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a)
  (scalene : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c)
  : (3 < h3_pos ∧ h3_pos < 6) → h3_pos = 5 := 
sorry

end maximum_third_height_l763_76321


namespace correct_operation_l763_76308

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ (2 * a^3 / a = 2 * a^2) ∧ ¬((a * b)^2 = a * b^2) ∧ ¬((-a^3)^3 = -a^6) :=
by
  sorry

end correct_operation_l763_76308


namespace correct_transformation_l763_76391

-- Definitions of the points and their mapped coordinates
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -3)
def C' : ℝ × ℝ := (1, 2)
def D' : ℝ × ℝ := (-2, 3)

-- Transformation function (as given in the problem)
def skew_reflection_and_vertical_shrink (p : ℝ × ℝ) : ℝ × ℝ :=
  match p with
  | (x, y) => (-y, x)

-- Theorem statement to be proved
theorem correct_transformation :
  skew_reflection_and_vertical_shrink C = C' ∧ skew_reflection_and_vertical_shrink D = D' :=
sorry

end correct_transformation_l763_76391


namespace three_layer_rug_area_l763_76324

theorem three_layer_rug_area 
  (A B C D : ℕ) 
  (hA : A = 350) 
  (hB : B = 250) 
  (hC : C = 45) 
  (h_formula : A = B + C + D) : 
  D = 55 :=
by
  sorry

end three_layer_rug_area_l763_76324


namespace calculate_expression_l763_76315

variable (y : ℝ) (π : ℝ) (Q : ℝ)

theorem calculate_expression (h : 5 * (3 * y - 7 * π) = Q) : 
  10 * (6 * y - 14 * π) = 4 * Q := by
  sorry

end calculate_expression_l763_76315


namespace max_value_y_l763_76331

variable (x : ℝ)
def y : ℝ := -3 * x^2 + 6

theorem max_value_y : ∃ M, ∀ x : ℝ, y x ≤ M ∧ (∀ x : ℝ, y x = M → x = 0) :=
by
  use 6
  sorry

end max_value_y_l763_76331


namespace max_value_min_value_l763_76376

noncomputable def y (x : ℝ) : ℝ := 2 * Real.sin (3 * x + (Real.pi / 3))

theorem max_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 + Real.pi / 18) ↔ y x = 2 :=
sorry

theorem min_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 - 5 * Real.pi / 18) ↔ y x = -2 :=
sorry

end max_value_min_value_l763_76376


namespace sum_of_three_is_odd_implies_one_is_odd_l763_76392

theorem sum_of_three_is_odd_implies_one_is_odd 
  (a b c : ℤ) 
  (h : (a + b + c) % 2 = 1) : 
  a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1 := 
sorry

end sum_of_three_is_odd_implies_one_is_odd_l763_76392


namespace cube_side_length_eq_three_l763_76380

theorem cube_side_length_eq_three (n : ℕ) (h1 : 6 * n^2 = 6 * n^3 / 3) : n = 3 := by
  -- The proof is omitted as per instructions, we use sorry to skip it.
  sorry

end cube_side_length_eq_three_l763_76380


namespace profit_percentage_is_correct_l763_76395

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 65.97
noncomputable def list_price := selling_price / 0.90
noncomputable def profit := selling_price - cost_price
noncomputable def profit_percentage := (profit / cost_price) * 100

theorem profit_percentage_is_correct : profit_percentage = 38.88 := by
  sorry

end profit_percentage_is_correct_l763_76395


namespace candy_sold_tuesday_correct_l763_76397

variable (pieces_sold_monday pieces_left_by_wednesday initial_candy total_pieces_sold : ℕ)
variable (pieces_sold_tuesday : ℕ)

-- Conditions
def initial_candy_amount := 80
def candy_sold_on_monday := 15
def candy_left_by_wednesday := 7

-- Total candy sold by Wednesday
def total_candy_sold_by_wednesday := initial_candy_amount - candy_left_by_wednesday

-- Candy sold on Tuesday
def candy_sold_on_tuesday : ℕ := total_candy_sold_by_wednesday - candy_sold_on_monday

-- Proof statement
theorem candy_sold_tuesday_correct : candy_sold_on_tuesday = 58 := sorry

end candy_sold_tuesday_correct_l763_76397


namespace problem_statement_l763_76338

noncomputable def increase_and_subtract (x p y : ℝ) : ℝ :=
  (x + p * x) - y

theorem problem_statement : increase_and_subtract 75 1.5 40 = 147.5 := by
  sorry

end problem_statement_l763_76338


namespace min_n_satisfies_inequality_l763_76326

theorem min_n_satisfies_inequality :
  ∃ n : ℕ, 0 < n ∧ -3 * (n : ℤ) ^ 4 + 5 * (n : ℤ) ^ 2 - 199 < 0 ∧ (∀ m : ℕ, 0 < m ∧ -3 * (m : ℤ) ^ 4 + 5 * (m : ℤ) ^ 2 - 199 < 0 → 2 ≤ m) := 
  sorry

end min_n_satisfies_inequality_l763_76326


namespace daniel_waist_size_correct_l763_76377

noncomputable def Daniel_waist_size_cm (inches_to_feet : ℝ) (feet_to_cm : ℝ) (waist_size_in_inches : ℝ) : ℝ := 
  (waist_size_in_inches * feet_to_cm) / inches_to_feet

theorem daniel_waist_size_correct :
  Daniel_waist_size_cm 12 30.5 34 = 86.4 :=
by
  -- This skips the proof for now
  sorry

end daniel_waist_size_correct_l763_76377


namespace molecular_weight_N2O5_correct_l763_76329

noncomputable def atomic_weight_N : ℝ := 14.01
noncomputable def atomic_weight_O : ℝ := 16.00
def molecular_formula_N2O5 : (ℕ × ℕ) := (2, 5)

theorem molecular_weight_N2O5_correct :
  let weight := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  weight = 108.02 :=
by
  sorry

end molecular_weight_N2O5_correct_l763_76329


namespace simplify_expression_l763_76385

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := 
by 
  sorry

end simplify_expression_l763_76385


namespace max_area_square_pen_l763_76314

theorem max_area_square_pen (P : ℝ) (h1 : P = 64) : ∃ A : ℝ, A = 256 := 
by
  sorry

end max_area_square_pen_l763_76314


namespace value_of_x_l763_76341

def x : ℚ :=
  (320 / 2) / 3

theorem value_of_x : x = 160 / 3 := 
by
  unfold x
  sorry

end value_of_x_l763_76341


namespace vector_calculation_l763_76368

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (-1, 2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

theorem vector_calculation :
  (dot_product (vector_add (scalar_mult 2 vector_a) vector_b) vector_a) = 1 :=
by
  sorry

end vector_calculation_l763_76368


namespace fermats_little_theorem_l763_76300

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) (hgcd : gcd a p = 1) : (a^(p-1) - 1) % p = 0 := by
  sorry

end fermats_little_theorem_l763_76300


namespace John_l763_76375

theorem John's_net_profit 
  (gross_income : ℕ)
  (car_purchase_cost : ℕ)
  (car_maintenance : ℕ → ℕ → ℕ)
  (car_insurance : ℕ)
  (car_tire_replacement : ℕ)
  (trade_in_value : ℕ)
  (tax_rate : ℚ)
  (total_taxes : ℕ)
  (monthly_maintenance_cost : ℕ)
  (months : ℕ)
  (net_profit : ℕ) :
  gross_income = 30000 →
  car_purchase_cost = 20000 →
  car_maintenance monthly_maintenance_cost months = 3600 →
  car_insurance = 1200 →
  car_tire_replacement = 400 →
  trade_in_value = 6000 →
  tax_rate = 15/100 →
  total_taxes = 4500 →
  monthly_maintenance_cost = 300 →
  months = 12 →
  net_profit = gross_income - (car_purchase_cost + car_maintenance monthly_maintenance_cost months + car_insurance + car_tire_replacement + total_taxes) + trade_in_value →
  net_profit = 6300 := 
by 
  sorry -- Proof to be provided

end John_l763_76375


namespace train_length_180_l763_76348

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_seconds

theorem train_length_180 :
  train_length 6 108 = 180 :=
sorry

end train_length_180_l763_76348


namespace range_of_m_l763_76310

def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 9 ≤ m :=
by
  sorry

end range_of_m_l763_76310


namespace mean_cat_weights_l763_76307

-- Define a list representing the weights of the cats from the stem-and-leaf plot
def cat_weights : List ℕ := [12, 13, 14, 20, 21, 21, 25, 25, 28, 30, 31, 32, 32, 36, 38, 39, 39]

-- Function to calculate the sum of elements in a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Function to calculate the mean of a list of natural numbers
def mean_list (l : List ℕ) : ℚ := (sum_list l : ℚ) / l.length

-- The theorem we need to prove
theorem mean_cat_weights : mean_list cat_weights = 27 := by 
  sorry

end mean_cat_weights_l763_76307


namespace product_of_xyz_is_correct_l763_76335

theorem product_of_xyz_is_correct : 
  ∃ x y z : ℤ, 
    (-3 * x + 4 * y - z = 28) ∧ 
    (3 * x - 2 * y + z = 8) ∧ 
    (x + y - z = 2) ∧ 
    (x * y * z = 2898) :=
by
  sorry

end product_of_xyz_is_correct_l763_76335


namespace no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l763_76354

open Nat

theorem no_odd_prime_pn_plus_1_eq_2m (n p m : ℕ)
  (hn : n > 1) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n + 1 ≠ 2^m := by
  sorry

theorem no_odd_prime_pn_minus_1_eq_2m (n p m : ℕ)
  (hn : n > 2) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n - 1 ≠ 2^m := by
  sorry

end no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l763_76354


namespace no_n_divisible_by_1955_l763_76328

theorem no_n_divisible_by_1955 : ∀ n : ℕ, ¬ (1955 ∣ (n^2 + n + 1)) := by
  sorry

end no_n_divisible_by_1955_l763_76328


namespace terrier_to_poodle_grooming_ratio_l763_76313

-- Definitions and conditions
def time_to_groom_poodle : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_grooming_time : ℕ := 210
def time_to_groom_terrier := total_grooming_time - (num_poodles * time_to_groom_poodle) / num_terriers

-- Theorem statement
theorem terrier_to_poodle_grooming_ratio :
  time_to_groom_terrier / time_to_groom_poodle = 1 / 2 :=
by
  sorry

end terrier_to_poodle_grooming_ratio_l763_76313


namespace problem_statement_l763_76304

def f (x : ℝ) : ℝ := x^5 - x^3 + 1
def g (x : ℝ) : ℝ := x^2 - 2

theorem problem_statement (x1 x2 x3 x4 x5 : ℝ) 
  (h_roots : ∀ x, f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) :
  g x1 * g x2 * g x3 * g x4 * g x5 = -7 := 
sorry

end problem_statement_l763_76304


namespace hyperbola_problem_l763_76383

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : Prop :=
  c / a = 2 * Real.sqrt 3 / 3

def focal_distance (c a : ℝ) : Prop :=
  2 * a^2 = 3 * c

def point_on_hyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b P.1 P.2

def point_satisfies_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 2

noncomputable def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem hyperbola_problem (a b c : ℝ) (P F1 F2 : ℝ × ℝ) :
  (a > 0 ∧ b > 0) →
  eccentricity a c →
  focal_distance c a →
  point_on_hyperbola P a b →
  point_satisfies_condition P F1 F2 →
  distance F1 F2 = 2 * c →
  (distance P F1) * (distance P F2) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hyperbola_problem_l763_76383


namespace remainder_of_2n_div_9_l763_76327

theorem remainder_of_2n_div_9
  (n : ℤ) (h : ∃ k : ℤ, n = 18 * k + 10) : (2 * n) % 9 = 2 := 
by
  sorry

end remainder_of_2n_div_9_l763_76327


namespace bounded_f_l763_76364

theorem bounded_f (f : ℝ → ℝ) (h1 : ∀ x1 x2, |x1 - x2| ≤ 1 → |f x2 - f x1| ≤ 1)
  (h2 : f 0 = 1) : ∀ x, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end bounded_f_l763_76364


namespace ratio_expression_l763_76360

variable (a b c : ℚ)
variable (h1 : a / b = 6 / 5)
variable (h2 : b / c = 8 / 7)

theorem ratio_expression (a b c : ℚ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end ratio_expression_l763_76360


namespace family_ages_sum_today_l763_76363

theorem family_ages_sum_today (A B C D E : ℕ) (h1 : A + B + C + D = 114) (h2 : E = D - 14) :
    (A + 5) + (B + 5) + (C + 5) + (E + 5) = 120 :=
by
  sorry

end family_ages_sum_today_l763_76363


namespace machine_value_correct_l763_76359

-- The present value of the machine
def present_value : ℝ := 1200

-- The depreciation rate function based on the year
def depreciation_rate (year : ℕ) : ℝ :=
  match year with
  | 1 => 0.10
  | 2 => 0.12
  | n => if n > 2 then 0.10 + 0.02 * (n - 1) else 0

-- The repair rate
def repair_rate : ℝ := 0.03

-- Value of the machine after n years
noncomputable def machine_value_after_n_years (initial_value : ℝ) (n : ℕ) : ℝ :=
  let value_first_year := (initial_value - (depreciation_rate 1 * initial_value)) + (repair_rate * initial_value)
  let value_second_year := (value_first_year - (depreciation_rate 2 * value_first_year)) + (repair_rate * value_first_year)
  match n with
  | 1 => value_first_year
  | 2 => value_second_year
  | _ => sorry -- Further generalization would be required for n > 2

-- Theorem statement
theorem machine_value_correct (initial_value : ℝ) :
  machine_value_after_n_years initial_value 2 = 1015.56 := by
  sorry

end machine_value_correct_l763_76359


namespace gummy_vitamins_cost_l763_76366

def bottle_discounted_price (P D_s : ℝ) : ℝ :=
  P * (1 - D_s)

def normal_purchase_discounted_price (discounted_price D_n : ℝ) : ℝ :=
  discounted_price * (1 - D_n)

def bulk_purchase_discounted_price (discounted_price D_b : ℝ) : ℝ :=
  discounted_price * (1 - D_b)

def total_cost (normal_bottles bulk_bottles normal_price bulk_price : ℝ) : ℝ :=
  (normal_bottles * normal_price) + (bulk_bottles * bulk_price)

def apply_coupons (total_cost N_c C : ℝ) : ℝ :=
  total_cost - (N_c * C)

theorem gummy_vitamins_cost 
  (P N_c C D_s D_n D_b : ℝ) 
  (normal_bottles bulk_bottles : ℕ) :
  bottle_discounted_price P D_s = 12.45 → 
  normal_purchase_discounted_price 12.45 D_n = 11.33 → 
  bulk_purchase_discounted_price 12.45 D_b = 11.83 → 
  total_cost 4 3 11.33 11.83 = 80.81 → 
  apply_coupons 80.81 N_c C = 70.81 :=
sorry

end gummy_vitamins_cost_l763_76366


namespace find_sum_of_abc_l763_76390

theorem find_sum_of_abc
  (a b c x y : ℕ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a^2 + b^2 + c^2 = 2011)
  (h3 : Nat.gcd a (Nat.gcd b c) = x)
  (h4 : Nat.lcm a (Nat.lcm b c) = y)
  (h5 : x + y = 388)
  :
  a + b + c = 61 :=
sorry

end find_sum_of_abc_l763_76390


namespace smallest_n_divisibility_problem_l763_76365

theorem smallest_n_divisibility_problem :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → ¬(n^2 + n) % k = 0)) ∧ n = 4 :=
by
  sorry

end smallest_n_divisibility_problem_l763_76365


namespace log_sqrt_defined_l763_76387

open Real

-- Define the conditions for the logarithm and square root arguments
def log_condition (x : ℝ) : Prop := 4 * x - 7 > 0
def sqrt_condition (x : ℝ) : Prop := 2 * x - 3 ≥ 0

-- Define the combined condition
def combined_condition (x : ℝ) : Prop := x > 7 / 4

-- The proof statement
theorem log_sqrt_defined (x : ℝ) : combined_condition x ↔ log_condition x ∧ sqrt_condition x :=
by
  -- Work through the equivalence and proof steps
  sorry

end log_sqrt_defined_l763_76387


namespace set_intersection_A_B_l763_76336

theorem set_intersection_A_B :
  (A : Set ℤ) ∩ (B : Set ℤ) = { -1, 0, 1, 2 } :=
by
  let A := { x : ℤ | x^2 - x - 2 ≤ 0 }
  let B := {x : ℤ | x ∈ Set.univ}
  sorry

end set_intersection_A_B_l763_76336


namespace find_price_of_pastry_l763_76398

-- Define the known values and conditions
variable (P : ℕ)  -- Price of a pastry
variable (usual_pastries : ℕ := 20)
variable (usual_bread : ℕ := 10)
variable (bread_price : ℕ := 4)
variable (today_pastries : ℕ := 14)
variable (today_bread : ℕ := 25)
variable (price_difference : ℕ := 48)

-- Define the usual daily total and today's total
def usual_total := usual_pastries * P + usual_bread * bread_price
def today_total := today_pastries * P + today_bread * bread_price

-- Define the problem statement
theorem find_price_of_pastry (h: usual_total - today_total = price_difference) : P = 18 :=
  by sorry

end find_price_of_pastry_l763_76398


namespace ellipse_equation_l763_76322

theorem ellipse_equation (a b c : ℝ) 
  (h1 : 0 < b) (h2 : b < a) 
  (h3 : c = 3 * Real.sqrt 3) 
  (h4 : a = 6) 
  (h5 : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
by
  sorry

end ellipse_equation_l763_76322


namespace find_largest_C_l763_76378

theorem find_largest_C : 
  ∃ (C : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 10 ≥ C * (x + y + 2)) 
  ∧ (∀ D : ℝ, (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 10 ≥ D * (x + y + 2)) → D ≤ C) 
  ∧ C = Real.sqrt 5 :=
sorry

end find_largest_C_l763_76378


namespace intersection_product_l763_76302

noncomputable def line_l (t : ℝ) := (1 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

def curve_C (x y : ℝ) : Prop := y^2 = 8 * x

theorem intersection_product :
  ∀ (t1 t2 : ℝ), 
  (1 + (1/2) * t1)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t1) →
  (1 + (1/2) * t2)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t2) →
  (1 + (1/2) * t1) * (1 + (1/2) * t2) = 28 := 
  sorry

end intersection_product_l763_76302


namespace find_g_at_75_l763_76399

noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y^2
axiom g_at_50 : g 50 = 25

-- The main result to be proved
theorem find_g_at_75 : g 75 = 100 / 9 :=
by
  sorry

end find_g_at_75_l763_76399


namespace middle_digit_is_3_l763_76318

theorem middle_digit_is_3 (d e f : ℕ) (hd : 0 ≤ d ∧ d ≤ 7) (he : 0 ≤ e ∧ e ≤ 7) (hf : 0 ≤ f ∧ f ≤ 7)
    (h_eq : 64 * d + 8 * e + f = 100 * f + 10 * e + d) : e = 3 :=
sorry

end middle_digit_is_3_l763_76318


namespace equal_expression_exists_l763_76373

-- lean statement for the mathematical problem
theorem equal_expression_exists (a b : ℤ) :
  ∃ (expr : ℤ), expr = 20 * a - 18 * b := by
  sorry

end equal_expression_exists_l763_76373


namespace minimize_function_l763_76306

noncomputable def f (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

theorem minimize_function : 
  (∀ x : ℝ, x > -1 → f x ≥ 1) ∧ (f 2 = 1) :=
by 
  sorry

end minimize_function_l763_76306


namespace infinite_common_divisor_l763_76332

theorem infinite_common_divisor (n : ℕ) : ∃ᶠ n in at_top, Nat.gcd (2 * n - 3) (3 * n - 2) > 1 := 
sorry

end infinite_common_divisor_l763_76332


namespace units_digit_divisible_by_18_l763_76361

theorem units_digit_divisible_by_18 : ∃ n : ℕ, (3150 ≤ 315 * n) ∧ (315 * n < 3160) ∧ (n % 2 = 0) ∧ (315 * n % 18 = 0) ∧ (n = 0) :=
by
  use 0
  sorry

end units_digit_divisible_by_18_l763_76361


namespace unique_parallel_line_in_beta_l763_76316

-- Define the basic geometrical entities.
axiom Plane : Type
axiom Line : Type
axiom Point : Type

-- Definitions relating entities.
def contains (P : Plane) (l : Line) : Prop := sorry
def parallel (A B : Plane) : Prop := sorry
def in_plane (p : Point) (P : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry

-- Statements derived from the conditions in problem.
variables (α β : Plane) (a : Line) (B : Point)
-- Given conditions
axiom plane_parallel : parallel α β
axiom line_in_plane : contains α a
axiom point_in_plane : in_plane B β

-- The ultimate goal derived from the question.
theorem unique_parallel_line_in_beta : 
  ∃! b : Line, (in_plane B β) ∧ (parallel_lines a b) :=
sorry

end unique_parallel_line_in_beta_l763_76316


namespace sum_less_than_addends_then_both_negative_l763_76393

theorem sum_less_than_addends_then_both_negative {a b : ℝ} (h : a + b < a ∧ a + b < b) : a < 0 ∧ b < 0 := 
sorry

end sum_less_than_addends_then_both_negative_l763_76393


namespace complement_of_angle_l763_76311

theorem complement_of_angle (supplement : ℝ) (h_supp : supplement = 130) (original_angle : ℝ) (h_orig : original_angle = 180 - supplement) : 
  (90 - original_angle) = 40 := 
by 
  -- proof goes here
  sorry

end complement_of_angle_l763_76311


namespace remainder_of_num_five_element_subsets_with_two_consecutive_l763_76355

-- Define the set and the problem
noncomputable def num_five_element_subsets_with_two_consecutive (n : ℕ) : ℕ := 
  Nat.choose 14 5 - Nat.choose 10 5

-- Main Lean statement: prove the final condition
theorem remainder_of_num_five_element_subsets_with_two_consecutive :
  (num_five_element_subsets_with_two_consecutive 14) % 1000 = 750 :=
by
  -- Proof goes here
  sorry

end remainder_of_num_five_element_subsets_with_two_consecutive_l763_76355


namespace max_positive_integers_l763_76356

theorem max_positive_integers (a b c d e f : ℤ) (h : (a * b + c * d * e * f) < 0) :
  ∃ n, n ≤ 5 ∧ (∀x ∈ [a, b, c, d, e, f], 0 < x → x ≤ 5) :=
by
  sorry

end max_positive_integers_l763_76356


namespace sqrt_seven_to_six_power_eq_343_l763_76319

theorem sqrt_seven_to_six_power_eq_343 : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_six_power_eq_343_l763_76319


namespace married_fraction_l763_76353

variable (total_people : ℕ) (fraction_women : ℚ) (max_unmarried_women : ℕ)
variable (fraction_married : ℚ)

theorem married_fraction (h1 : total_people = 80)
                         (h2 : fraction_women = 1/4)
                         (h3 : max_unmarried_women = 20)
                         : fraction_married = 3/4 :=
by
  sorry

end married_fraction_l763_76353


namespace expand_product_l763_76340

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := 
by
  sorry

end expand_product_l763_76340


namespace gasoline_needed_l763_76305

theorem gasoline_needed (D : ℕ) 
    (fuel_efficiency : ℕ) 
    (fuel_efficiency_proof : fuel_efficiency = 20)
    (gallons_for_130km : ℕ) 
    (gallons_for_130km_proof : gallons_for_130km = 130 / 20) :
    (D : ℕ) / fuel_efficiency = (D : ℕ) / 20 :=
by
  -- The proof is omitted as per the instruction
  sorry

end gasoline_needed_l763_76305


namespace solve_system_of_equations_l763_76343

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + x * y = 15) (h2 : x^2 + x * y = 10) :
  (x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3) :=
sorry

end solve_system_of_equations_l763_76343


namespace factor_theorem_q_value_l763_76351

theorem factor_theorem_q_value (q : ℤ) (m : ℤ) :
  (∀ m, (m - 8) ∣ (m^2 - q * m - 24)) → q = 5 :=
by
  sorry

end factor_theorem_q_value_l763_76351


namespace combined_water_leak_l763_76374

theorem combined_water_leak
  (largest_rate : ℕ)
  (medium_rate : ℕ)
  (smallest_rate : ℕ)
  (time_minutes : ℕ)
  (h1 : largest_rate = 3)
  (h2 : medium_rate = largest_rate / 2)
  (h3 : smallest_rate = medium_rate / 3)
  (h4 : time_minutes = 120) :
  largest_rate * time_minutes + medium_rate * time_minutes + smallest_rate * time_minutes = 600 := by
  sorry

end combined_water_leak_l763_76374


namespace probability_difference_l763_76350

-- Definitions for probabilities
def P_plane : ℚ := 7 / 10
def P_train : ℚ := 3 / 10
def P_on_time_plane : ℚ := 8 / 10
def P_on_time_train : ℚ := 9 / 10

-- Events definitions
def P_arrive_on_time : ℚ := (7 / 10) * (8 / 10) + (3 / 10) * (9 / 10)
def P_plane_and_on_time : ℚ := (7 / 10) * (8 / 10)
def P_train_and_on_time : ℚ := (3 / 10) * (9 / 10)
def P_conditional_plane_given_on_time : ℚ := P_plane_and_on_time / P_arrive_on_time
def P_conditional_train_given_on_time : ℚ := P_train_and_on_time / P_arrive_on_time

theorem probability_difference :
  P_conditional_plane_given_on_time - P_conditional_train_given_on_time = 29 / 83 :=
by sorry

end probability_difference_l763_76350
