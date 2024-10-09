import Mathlib

namespace students_behind_yoongi_l2103_210354

theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) (behind_students : ℕ)
  (h1 : total_students = 20)
  (h2 : jungkook_position = 3)
  (h3 : yoongi_position = jungkook_position + 1)
  (h4 : behind_students = total_students - yoongi_position) :
  behind_students = 16 :=
by
  sorry

end students_behind_yoongi_l2103_210354


namespace convex_polygon_num_sides_l2103_210351

theorem convex_polygon_num_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 120 + i * 5 < 180) 
  (h2 : (n - 2) * 180 = n * (240 + (n - 1) * 5) / 2) : 
  n = 9 :=
sorry

end convex_polygon_num_sides_l2103_210351


namespace prove_cardinality_l2103_210388

-- Definitions used in Lean 4 Statement adapted from conditions
variable (a b : ℕ)
variable (A B : Finset ℕ)

-- Hypotheses
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_disjoint : Disjoint A B)
variable (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B)

-- The statement to prove
theorem prove_cardinality (a b : ℕ) (A B : Finset ℕ)
  (ha : a > 0) (hb : b > 0) (h_disjoint : Disjoint A B)
  (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B) :
  a * A.card = b * B.card :=
by 
  sorry

end prove_cardinality_l2103_210388


namespace correct_relation_is_identity_l2103_210353

theorem correct_relation_is_identity : 0 = 0 :=
by {
  -- Skipping proof steps as only statement is required
  sorry
}

end correct_relation_is_identity_l2103_210353


namespace folded_segment_square_length_eq_225_div_4_l2103_210376

noncomputable def square_of_fold_length : ℝ :=
  let side_length := 15
  let distance_from_B := 5
  (side_length ^ 2 - distance_from_B * (2 * side_length - distance_from_B)) / 4

theorem folded_segment_square_length_eq_225_div_4 :
  square_of_fold_length = 225 / 4 :=
by
  sorry

end folded_segment_square_length_eq_225_div_4_l2103_210376


namespace locus_of_point_P_l2103_210365

theorem locus_of_point_P (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  (x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2) ↔ 
  ((x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 ∧ x ≠ 2 ∧ x ≠ -2) :=
by
  sorry 

end locus_of_point_P_l2103_210365


namespace num_balls_total_l2103_210310

theorem num_balls_total (m : ℕ) (h1 : 6 < m) (h2 : (6 : ℝ) / (m : ℝ) = 0.3) : m = 20 :=
by
  sorry

end num_balls_total_l2103_210310


namespace preceding_integer_binary_l2103_210383

--- The conditions as definitions in Lean 4

def M := 0b101100 -- M is defined as binary '101100' which is decimal 44
def preceding_binary (n : Nat) : Nat := n - 1 -- Define a function to get the preceding integer in binary

--- The proof problem statement in Lean 4
theorem preceding_integer_binary :
  preceding_binary M = 0b101011 :=
by
  sorry

end preceding_integer_binary_l2103_210383


namespace jackson_entertainment_cost_l2103_210392

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end jackson_entertainment_cost_l2103_210392


namespace box_growth_factor_l2103_210339

/-
Problem: When a large box in the shape of a cuboid measuring 6 centimeters (cm) wide,
4 centimeters (cm) long, and 1 centimeters (cm) high became larger into a volume of
30 centimeters (cm) wide, 20 centimeters (cm) long, and 5 centimeters (cm) high,
find how many times it has grown.
-/

def original_box_volume (w l h : ℕ) : ℕ := w * l * h
def larger_box_volume (w l h : ℕ) : ℕ := w * l * h

theorem box_growth_factor :
  original_box_volume 6 4 1 * 125 = larger_box_volume 30 20 5 :=
by
  -- Proof goes here
  sorry

end box_growth_factor_l2103_210339


namespace range_of_a_l2103_210303

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - a > 0 ∧ 3 * x - 4 < 5) -> False) ↔ (a ≥ 6) :=
by
  sorry

end range_of_a_l2103_210303


namespace sum_of_roots_eq_three_l2103_210314

theorem sum_of_roots_eq_three (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + 2 = 0)
  (h2 : x2^2 - 3*x2 + 2 = 0) 
  (h3 : x1 ≠ x2) : 
  x1 + x2 = 3 := 
sorry

end sum_of_roots_eq_three_l2103_210314


namespace harry_apples_l2103_210355

theorem harry_apples (martha_apples : ℕ) (tim_apples : ℕ) (harry_apples : ℕ)
  (h1 : martha_apples = 68)
  (h2 : tim_apples = martha_apples - 30)
  (h3 : harry_apples = tim_apples / 2) :
  harry_apples = 19 := 
by sorry

end harry_apples_l2103_210355


namespace remaining_credit_l2103_210372

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end remaining_credit_l2103_210372


namespace hyperbola_lattice_points_count_l2103_210396

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - 2 * y^2 = 2000^2

def count_lattice_points (points : List (ℤ × ℤ)) : ℕ :=
  points.length

theorem hyperbola_lattice_points_count : count_lattice_points [(2000, 0), (-2000, 0)] = 2 :=
by
  sorry

end hyperbola_lattice_points_count_l2103_210396


namespace ball_bounce_height_l2103_210342

theorem ball_bounce_height :
  ∃ k : ℕ, k = 4 ∧ 45 * (1 / 3 : ℝ) ^ k < 2 :=
by 
  use 4
  sorry

end ball_bounce_height_l2103_210342


namespace rational_relation_l2103_210325

variable {a b : ℚ}

theorem rational_relation (h1 : a > 0) (h2 : b < 0) (h3 : |a| > |b|) : -a < -b ∧ -b < b ∧ b < a :=
by
  sorry

end rational_relation_l2103_210325


namespace geometric_series_ratio_l2103_210338

theorem geometric_series_ratio (a_1 a_2 S q : ℝ) (hq : |q| < 1)
  (hS : S = a_1 / (1 - q))
  (ha2 : a_2 = a_1 * q) :
  S / (S - a_1) = a_1 / a_2 := 
sorry

end geometric_series_ratio_l2103_210338


namespace tangent_lines_through_point_l2103_210391

theorem tangent_lines_through_point (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) ∧ (x = -2 ∨ (15*x + 8*y - 10 = 0)) ↔ 
  (x = -2 ∨ (15*x + 8*y - 10 = 0)) :=
by
  sorry

end tangent_lines_through_point_l2103_210391


namespace value_of_b_l2103_210366

variable (a b c y1 y2 : ℝ)

def equation1 := (y1 = 4 * a + 2 * b + c)
def equation2 := (y2 = 4 * a - 2 * b + c)
def difference := (y1 - y2 = 8)

theorem value_of_b 
  (h1 : equation1 a b c y1)
  (h2 : equation2 a b c y2)
  (h3 : difference y1 y2) : 
  b = 2 := 
by 
  sorry

end value_of_b_l2103_210366


namespace wall_width_l2103_210384

theorem wall_width (w h l : ℝ)
  (h_eq_6w : h = 6 * w)
  (l_eq_7h : l = 7 * h)
  (V_eq : w * h * l = 86436) :
  w = 7 :=
by
  sorry

end wall_width_l2103_210384


namespace factor_polynomial_l2103_210328

theorem factor_polynomial (x y z : ℝ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) :=
by
  sorry

end factor_polynomial_l2103_210328


namespace example_function_not_power_function_l2103_210386

-- Definition of a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the function y = 2x^(1/2)
def example_function (x : ℝ) : ℝ :=
  2 * x ^ (1 / 2)

-- The statement we want to prove
theorem example_function_not_power_function : ¬ is_power_function example_function := by
  sorry

end example_function_not_power_function_l2103_210386


namespace geometric_series_common_ratio_l2103_210362

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l2103_210362


namespace minimum_square_side_length_l2103_210308

theorem minimum_square_side_length (s : ℝ) (h1 : s^2 ≥ 625) (h2 : ∃ (t : ℝ), t = s / 2) : s = 25 :=
by
  sorry

end minimum_square_side_length_l2103_210308


namespace cosine_identity_l2103_210340

theorem cosine_identity
  (α : ℝ)
  (h : Real.sin (π / 6 + α) = (Real.sqrt 3) / 3) :
  Real.cos (π / 3 - α) = (Real.sqrt 3) / 2 :=
by
  sorry

end cosine_identity_l2103_210340


namespace hourly_wage_increase_l2103_210389

variables (W W' H H' : ℝ)

theorem hourly_wage_increase :
  H' = (2/3) * H →
  W * H = W' * H' →
  W' = (3/2) * W :=
by
  intros h_eq income_eq
  rw [h_eq] at income_eq
  sorry

end hourly_wage_increase_l2103_210389


namespace prob_not_lose_money_proof_min_purchase_price_proof_l2103_210378

noncomputable def prob_not_lose_money : ℚ :=
  let pr_normal_rain := (2 : ℚ) / 3
  let pr_less_rain := (1 : ℚ) / 3
  let pr_price_6_normal := (1 : ℚ) / 4
  let pr_price_6_less := (2 : ℚ) / 3
  pr_normal_rain * pr_price_6_normal + pr_less_rain * pr_price_6_less

theorem prob_not_lose_money_proof : prob_not_lose_money = 7 / 18 := sorry

noncomputable def min_purchase_price : ℚ :=
  let old_exp_income := 500
  let new_yield := 2500
  let cost_increase := 1000
  (7000 + 1500 + cost_increase) / new_yield
  
theorem min_purchase_price_proof : min_purchase_price = 3.4 := sorry

end prob_not_lose_money_proof_min_purchase_price_proof_l2103_210378


namespace prime_p_q_r_condition_l2103_210346

theorem prime_p_q_r_condition (p q r : ℕ) (hp : Nat.Prime p) (hq_pos : 0 < q) (hr_pos : 0 < r)
    (hp_not_dvd_q : ¬ (p ∣ q)) (h3_not_dvd_q : ¬ (3 ∣ q)) (eqn : p^3 = r^3 - q^2) : 
    p = 7 := sorry

end prime_p_q_r_condition_l2103_210346


namespace quadratic_complete_the_square_l2103_210395

theorem quadratic_complete_the_square :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 + 1500 * x + 1500 = (x + b) ^ 2 + c)
      ∧ b = 750
      ∧ c = -748 * 750
      ∧ c / b = -748 := 
by {
  sorry
}

end quadratic_complete_the_square_l2103_210395


namespace sticks_left_in_yard_l2103_210349

def number_of_sticks_picked_up : Nat := 14
def difference_between_picked_and_left : Nat := 10

theorem sticks_left_in_yard 
  (picked_up : Nat := number_of_sticks_picked_up)
  (difference : Nat := difference_between_picked_and_left) 
  : Nat :=
  picked_up - difference

example : sticks_left_in_yard = 4 := by 
  sorry

end sticks_left_in_yard_l2103_210349


namespace product_of_0_5_and_0_8_l2103_210352

theorem product_of_0_5_and_0_8 : (0.5 * 0.8) = 0.4 := by
  sorry

end product_of_0_5_and_0_8_l2103_210352


namespace evaluate_expression_l2103_210302

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 5) :
  3 * x^4 + 2 * y^2 + 10 = 8 * 37 + 7 := 
by
  sorry

end evaluate_expression_l2103_210302


namespace midpoint_trajectory_l2103_210390

theorem midpoint_trajectory (x y : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (8, 0) ∧ (B.1, B.2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } ∧ 
   ∃ P : ℝ × ℝ, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ P = (x, y)) → (x - 4)^2 + y^2 = 1 :=
by sorry

end midpoint_trajectory_l2103_210390


namespace seq_arithmetic_l2103_210359

noncomputable def f (x n : ℝ) : ℝ := (x - 1)^2 + n

def a_n (n : ℝ) : ℝ := n
def b_n (n : ℝ) : ℝ := n + 4
def c_n (n : ℝ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem seq_arithmetic (n : ℕ) (hn : 0 < n) :
  ∃ d, d ≠ 0 ∧ ∀ n, c_n (↑n : ℝ) = c_n (↑n + 1 : ℝ) - d := 
sorry

end seq_arithmetic_l2103_210359


namespace find_f_3_l2103_210368

def f (x : ℝ) : ℝ := x + 3  -- define the function as per the condition

theorem find_f_3 : f (3) = 7 := by
  sorry

end find_f_3_l2103_210368


namespace minimum_soldiers_to_add_l2103_210312

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l2103_210312


namespace triangle_area_l2103_210309

structure Point where
  x : ℝ
  y : ℝ

def area_triangle (A B C : Point) : ℝ := 
  0.5 * (B.x - A.x) * (C.y - A.y)

theorem triangle_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨8, 15⟩
  let C : Point := ⟨8, 0⟩
  area_triangle A B C = 60 :=
by
  sorry

end triangle_area_l2103_210309


namespace document_total_characters_l2103_210344

theorem document_total_characters (T : ℕ) : 
  (∃ (t_1 t_2 t_3 : ℕ) (v_A v_B : ℕ),
      v_A = 100 ∧ v_B = 200 ∧
      t_1 = T / 600 ∧
      v_A * t_1 = T / 6 ∧
      v_B * t_1 = T / 3 ∧
      v_A * 3 * 5 = 1500 ∧
      t_2 = (T / 2 - 1500) / 500 ∧
      (v_A * 3 * t_2 + 1500 + v_A * t_1 = v_B * t_1 + v_B * t_2) ∧
      (v_A * 3 * (T - 3000) / 1000 + 1500 + v_A * T / 6 =
       v_B * 2 * (T - 3000) / 10 + v_B * T / 3)) →
  T = 18000 := by
  sorry

end document_total_characters_l2103_210344


namespace chip_placement_count_l2103_210326

def grid := Fin 4 × Fin 3

def grid_positions (n : Nat) := {s : Finset grid // s.card = n}

def no_direct_adjacency (positions : Finset grid) : Prop :=
  ∀ (x y : grid), x ∈ positions → y ∈ positions →
  (x.fst ≠ y.fst ∨ x.snd ≠ y.snd)

noncomputable def count_valid_placements : Nat :=
  -- Function to count valid placements
  sorry

theorem chip_placement_count :
  count_valid_placements = 4 :=
  sorry

end chip_placement_count_l2103_210326


namespace expected_value_of_winnings_is_4_l2103_210345

noncomputable def expected_value_of_winnings : ℕ := 
  let outcomes := [7, 6, 5, 4, 4, 3, 2, 1]
  let total_winnings := outcomes.sum
  total_winnings / 8

theorem expected_value_of_winnings_is_4 :
  expected_value_of_winnings = 4 :=
by
  sorry

end expected_value_of_winnings_is_4_l2103_210345


namespace sandy_correct_sums_l2103_210332

/-- 
Sandy gets 3 marks for each correct sum and loses 2 marks for each incorrect sum.
Sandy attempts 50 sums and obtains 100 marks within a 45-minute time constraint.
If Sandy receives a 1-mark penalty for each sum not completed within the time limit,
prove that the number of correct sums Sandy got is 25.
-/
theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 50) (h2 : 3 * c - 2 * i - (50 - c) = 100) : c = 25 :=
by
  sorry

end sandy_correct_sums_l2103_210332


namespace value_of_b_l2103_210333

theorem value_of_b (y b : ℝ) (hy : y > 0) (h : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 :=
by
  -- Proof omitted for brevity
  sorry

end value_of_b_l2103_210333


namespace digit_B_identification_l2103_210329

theorem digit_B_identification (B : ℕ) 
  (hB_range : 0 ≤ B ∧ B < 10) 
  (h_units_digit : (5 * B % 10) = 5) 
  (h_product : (10 * B + 5) * (90 + B) = 9045) : 
  B = 9 :=
sorry

end digit_B_identification_l2103_210329


namespace correct_understanding_of_philosophy_l2103_210393

-- Define the conditions based on the problem statement
def philosophy_from_life_and_practice : Prop :=
  -- Philosophy originates from people's lives and practice.
  sorry
  
def philosophy_affects_lives : Prop :=
  -- Philosophy consciously or unconsciously affects people's lives, learning, and work
  sorry

def philosophical_knowledge_requires_learning : Prop :=
  true

def philosophy_not_just_summary : Prop :=
  true

-- Given conditions 1, 2, 3 (as negation of 3 in original problem), and 4 (as negation of 4 in original problem),
-- We need to prove the correct understanding (which is combination ①②) is correct.
theorem correct_understanding_of_philosophy :
  philosophy_from_life_and_practice →
  philosophy_affects_lives →
  philosophical_knowledge_requires_learning →
  philosophy_not_just_summary →
  (philosophy_from_life_and_practice ∧ philosophy_affects_lives) :=
by
  intros
  apply And.intro
  · assumption
  · assumption

end correct_understanding_of_philosophy_l2103_210393


namespace average_age_l2103_210300
open Nat

def age_to_months (years : ℕ) (months : ℕ) : ℕ := years * 12 + months

theorem average_age :
  let age1 := age_to_months 14 9
  let age2 := age_to_months 15 1
  let age3 := age_to_months 14 8
  let total_months := age1 + age2 + age3
  let avg_months := total_months / 3
  let avg_years := avg_months / 12
  let avg_remaining_months := avg_months % 12
  avg_years = 14 ∧ avg_remaining_months = 10 := by
  sorry

end average_age_l2103_210300


namespace min_value_expression_l2103_210379

theorem min_value_expression (x : ℝ) (hx : x > 0) : x + 4/x ≥ 4 :=
sorry

end min_value_expression_l2103_210379


namespace find_middle_number_l2103_210357

theorem find_middle_number (a : Fin 11 → ℝ)
  (h1 : ∀ i : Fin 9, a i + a (⟨i.1 + 1, by linarith [i.2]⟩) + a (⟨i.1 + 2, by linarith [i.2]⟩) = 18)
  (h2 : (Finset.univ.sum a) = 64) :
  a 5 = 8 := 
by
  sorry

end find_middle_number_l2103_210357


namespace quadratic_has_two_distinct_real_roots_l2103_210367

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ p : ℝ, (p = x1 ∨ p = x2) → (p ^ 2 + (4 * m + 1) * p + m = 0)) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l2103_210367


namespace a4_minus_1_divisible_5_l2103_210371

theorem a4_minus_1_divisible_5 (a : ℤ) (h : ¬ (∃ k : ℤ, a = 5 * k)) : 
  (a^4 - 1) % 5 = 0 :=
by
  sorry

end a4_minus_1_divisible_5_l2103_210371


namespace bus_stops_duration_per_hour_l2103_210364

def speed_without_stoppages : ℝ := 90
def speed_with_stoppages : ℝ := 84
def distance_covered_lost := speed_without_stoppages - speed_with_stoppages

theorem bus_stops_duration_per_hour :
  distance_covered_lost / speed_without_stoppages * 60 = 4 :=
by
  sorry

end bus_stops_duration_per_hour_l2103_210364


namespace sqrt_simplify_l2103_210373

theorem sqrt_simplify (a b x : ℝ) (h : a < b) (hx1 : x + b ≥ 0) (hx2 : x + a ≤ 0) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * (Real.sqrt (-(x + a) * (x + b))) :=
by
  sorry

end sqrt_simplify_l2103_210373


namespace find_x_l2103_210320

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : 
  (∀ a b c d : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 :=
sorry

end find_x_l2103_210320


namespace min_frac_sum_l2103_210350

theorem min_frac_sum (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  (3 / b + 2 / a) = 7 + 4 * Real.sqrt 3 := 
sorry

end min_frac_sum_l2103_210350


namespace largest_positive_integer_divisible_l2103_210311

theorem largest_positive_integer_divisible (n : ℕ) :
  (n + 20 ∣ n^3 - 100) ↔ n = 2080 :=
sorry

end largest_positive_integer_divisible_l2103_210311


namespace candy_bar_cost_l2103_210377

-- Define the conditions
def cost_gum_over_candy_bar (C G : ℝ) : Prop :=
  G = (1/2) * C

def total_cost (C G : ℝ) : Prop :=
  2 * G + 3 * C = 6

-- Define the proof problem
theorem candy_bar_cost (C G : ℝ) (h1 : cost_gum_over_candy_bar C G) (h2 : total_cost C G) : C = 1.5 :=
by
  sorry

end candy_bar_cost_l2103_210377


namespace smallest_four_digit_divisible_by_53_l2103_210369

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l2103_210369


namespace mike_investment_l2103_210327

-- Define the given conditions and the conclusion we want to prove
theorem mike_investment (profit : ℝ) (mary_investment : ℝ) (mike_gets_more : ℝ) (total_profit_made : ℝ) :
  profit = 7500 → 
  mary_investment = 600 →
  mike_gets_more = 1000 →
  total_profit_made = 7500 →
  ∃ (mike_investment : ℝ), 
  ((1 / 3) * profit / 2 + (mary_investment / (mary_investment + mike_investment)) * ((2 / 3) * profit) = 
  (1 / 3) * profit / 2 + (mike_investment / (mary_investment + mike_investment)) * ((2 / 3) * profit) + mike_gets_more) →
  mike_investment = 400 :=
sorry

end mike_investment_l2103_210327


namespace ratio_female_to_male_l2103_210336

namespace DeltaSportsClub

variables (f m : ℕ) -- number of female and male members
-- Sum of ages of female and male members respectively
def sum_ages_females := 35 * f
def sum_ages_males := 30 * m
-- Total sum of ages
def total_sum_ages := sum_ages_females f + sum_ages_males m
-- Total number of members
def total_members := f + m

-- Given condition on the average age of all members
def average_age_condition := (total_sum_ages f m) / (total_members f m) = 32

-- The target theorem to prove the ratio of female to male members
theorem ratio_female_to_male (h : average_age_condition f m) : f/m = 2/3 :=
by sorry

end DeltaSportsClub

end ratio_female_to_male_l2103_210336


namespace sum_real_roots_eq_neg4_l2103_210382

-- Define the equation condition
def equation_condition (x : ℝ) : Prop :=
  (2 * x / (x^2 + 5 * x + 3) + 3 * x / (x^2 + x + 3) = 1)

-- Define the statement that sums the real roots
theorem sum_real_roots_eq_neg4 : 
  ∃ S : ℝ, (∀ x : ℝ, equation_condition x → x = -1 ∨ x = -3) ∧ (S = -4) :=
sorry

end sum_real_roots_eq_neg4_l2103_210382


namespace smallest_sum_of_squares_l2103_210356

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l2103_210356


namespace not_pass_first_quadrant_l2103_210337

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  (1/5)^(x + 1) + m

theorem not_pass_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -(1/5) :=
  by
  sorry

end not_pass_first_quadrant_l2103_210337


namespace range_of_m_l2103_210331

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l2103_210331


namespace tenth_term_of_arithmetic_progression_l2103_210399

variable (a d n T_n : ℕ)

def arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_progression :
  arithmetic_progression 8 2 10 = 26 :=
  by
  sorry

end tenth_term_of_arithmetic_progression_l2103_210399


namespace induction_step_l2103_210317

theorem induction_step 
  (k : ℕ) 
  (hk : ∃ m: ℕ, 5^k - 2^k = 3 * m) : 
  ∃ n: ℕ, 5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k :=
by
  sorry

end induction_step_l2103_210317


namespace decrement_from_observation_l2103_210306

theorem decrement_from_observation 
  (n : ℕ) (mean_original mean_updated : ℚ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 194)
  : (mean_original - mean_updated) = 6 :=
by
  sorry

end decrement_from_observation_l2103_210306


namespace calculate_expression_l2103_210385

theorem calculate_expression (a : ℤ) (h : a = -2) : a^3 - a^2 = -12 := 
by
  sorry

end calculate_expression_l2103_210385


namespace prank_combinations_l2103_210323

-- Conditions stated as definitions
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 6
def friday_choices : ℕ := 2

-- Theorem to prove
theorem prank_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 180 :=
by
  sorry

end prank_combinations_l2103_210323


namespace labor_budget_constraint_l2103_210304

-- Define the conditions
def wage_per_carpenter : ℕ := 50
def wage_per_mason : ℕ := 40
def labor_budget : ℕ := 2000
def num_carpenters (x : ℕ) := x
def num_masons (y : ℕ) := y

-- The proof statement
theorem labor_budget_constraint (x y : ℕ) 
    (hx : wage_per_carpenter * num_carpenters x + wage_per_mason * num_masons y ≤ labor_budget) : 
    5 * x + 4 * y ≤ 200 := 
by sorry

end labor_budget_constraint_l2103_210304


namespace finite_decimal_representation_nat_numbers_l2103_210394

theorem finite_decimal_representation_nat_numbers (n : ℕ) : 
  (∀ k : ℕ, k < n → (∃ u v : ℕ, (k + 1 = 2^u ∨ k + 1 = 5^v) ∨ (k - 1 = 2^u ∨ k -1  = 5^v))) ↔ 
  (n = 2 ∨ n = 3 ∨ n = 6) :=
by sorry

end finite_decimal_representation_nat_numbers_l2103_210394


namespace speed_of_man_in_still_water_l2103_210360

def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

theorem speed_of_man_in_still_water :
  (upstream_speed + (downstream_speed - upstream_speed) / 2) = 40 :=
by
  sorry

end speed_of_man_in_still_water_l2103_210360


namespace solve_grape_rate_l2103_210313

noncomputable def grape_rate (G : ℝ) : Prop :=
  11 * G + 7 * 50 = 1428

theorem solve_grape_rate : ∃ G : ℝ, grape_rate G ∧ G = 98 :=
by
  exists 98
  sorry

end solve_grape_rate_l2103_210313


namespace arithmetic_mean_of_18_27_45_l2103_210324

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_of_18_27_45_l2103_210324


namespace equivalent_proposition_l2103_210322

theorem equivalent_proposition (H : Prop) (P : Prop) (Q : Prop) (hpq : H → P → ¬ Q) : (H → ¬ Q → ¬ P) :=
by
  intro h nq np
  sorry

end equivalent_proposition_l2103_210322


namespace range_of_a_l2103_210398

theorem range_of_a (a : ℝ) : 
  (M = {x : ℝ | 2 * x + 1 < 3}) → 
  (N = {x : ℝ | x < a}) → 
  (M ∩ N = N) ↔ a ≤ 1 :=
by
  let M := {x : ℝ | 2 * x + 1 < 3}
  let N := {x : ℝ | x < a}
  simp [Set.subset_def]
  sorry

end range_of_a_l2103_210398


namespace question1_question2_l2103_210343

/-
In ΔABC, the sides opposite to angles A, B, and C are respectively a, b, and c.
It is given that b + c = 2 * a * cos B.

(1) Prove that A = 2B;
(2) If the area of ΔABC is S = a^2 / 4, find the magnitude of angle A.
-/

variables {A B C a b c : ℝ}
variables {S : ℝ}

-- Condition given in the problem
axiom h1 : b + c = 2 * a * Real.cos B
axiom h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4

-- Question 1: Prove that A = 2 * B
theorem question1 (h1 : b + c = 2 * a * Real.cos B) : A = 2 * B := sorry

-- Question 2: Find the magnitude of angle A
theorem question2 (h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4) : A = 90 ∨ A = 45 := sorry

end question1_question2_l2103_210343


namespace tennis_handshakes_l2103_210397

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l2103_210397


namespace f_prime_at_1_l2103_210387

def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

theorem f_prime_at_1 : (deriv f 1) = 11 :=
by
  sorry

end f_prime_at_1_l2103_210387


namespace find_abc_l2103_210305

theorem find_abc (a b c : ℝ) (h1 : a * (b + c) = 198) (h2 : b * (c + a) = 210) (h3 : c * (a + b) = 222) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a * b * c = 1069 :=
by
  sorry

end find_abc_l2103_210305


namespace general_term_of_sequence_l2103_210307

theorem general_term_of_sequence (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) :
  ∀ n, a n = (n^2 + n + 2) / 2 :=
by 
  sorry

end general_term_of_sequence_l2103_210307


namespace Claire_photos_l2103_210316

variable (C : ℕ)

def Lisa_photos := 3 * C
def Robert_photos := C + 28

theorem Claire_photos :
  Lisa_photos C = Robert_photos C → C = 14 :=
by
  sorry

end Claire_photos_l2103_210316


namespace roses_in_vase_l2103_210358

/-- There were initially 16 roses and 3 orchids in the vase.
    Jessica cut 8 roses and 8 orchids from her garden.
    There are now 7 orchids in the vase.
    Prove that the number of roses in the vase now is 24. -/
theorem roses_in_vase
  (initial_roses initial_orchids : ℕ)
  (cut_roses cut_orchids remaining_orchids final_roses : ℕ)
  (h_initial: initial_roses = 16)
  (h_initial_orchids: initial_orchids = 3)
  (h_cut: cut_roses = 8 ∧ cut_orchids = 8)
  (h_remaining_orchids: remaining_orchids = 7)
  (h_orchids_relation: initial_orchids + cut_orchids = remaining_orchids + cut_orchids - 4)
  : final_roses = initial_roses + cut_roses := by
  sorry

end roses_in_vase_l2103_210358


namespace not_right_triangle_angle_ratio_l2103_210380

theorem not_right_triangle_angle_ratio (A B C : ℝ) (h₁ : A / B = 3 / 4) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end not_right_triangle_angle_ratio_l2103_210380


namespace total_cups_l2103_210348

theorem total_cups (t1 t2 : ℕ) (h1 : t2 = 240) (h2 : t2 = t1 - 20) : t1 + t2 = 500 := by
  sorry

end total_cups_l2103_210348


namespace GCF_of_48_180_98_l2103_210361

theorem GCF_of_48_180_98 : Nat.gcd (Nat.gcd 48 180) 98 = 2 :=
by
  sorry

end GCF_of_48_180_98_l2103_210361


namespace point_in_third_quadrant_l2103_210374

open Complex

-- Define that i is the imaginary unit
def imaginary_unit : ℂ := Complex.I

-- Define the condition i * z = 1 - 2i
def condition (z : ℂ) : Prop := imaginary_unit * z = (1 : ℂ) - 2 * imaginary_unit

-- Prove that the point corresponding to the complex number z is located in the third quadrant
theorem point_in_third_quadrant (z : ℂ) (h : condition z) : z.re < 0 ∧ z.im < 0 := sorry

end point_in_third_quadrant_l2103_210374


namespace analytical_expression_of_C3_l2103_210334

def C1 (x : ℝ) : ℝ := x^2 - 2*x + 3
def C2 (x : ℝ) : ℝ := C1 (x + 1)
def C3 (x : ℝ) : ℝ := C2 (-x)

theorem analytical_expression_of_C3 :
  ∀ x, C3 x = x^2 + 2 := by
  sorry

end analytical_expression_of_C3_l2103_210334


namespace unique_fraction_increased_by_20_percent_l2103_210301

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem unique_fraction_increased_by_20_percent (x y : ℕ) (h1 : relatively_prime x y) (h2 : x > 0) (h3 : y > 0) :
  (∃! (x y : ℕ), relatively_prime x y ∧ (x > 0) ∧ (y > 0) ∧ (x + 2) * y = 6 * (y + 2) * x) :=
sorry

end unique_fraction_increased_by_20_percent_l2103_210301


namespace smallest_positive_multiple_of_32_l2103_210318

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end smallest_positive_multiple_of_32_l2103_210318


namespace find_n_l2103_210319

theorem find_n (x : ℝ) (h1 : x = 4.0) (h2 : 3 * x + n = 48) : n = 36 := by
  sorry

end find_n_l2103_210319


namespace area_ratio_correct_l2103_210341

noncomputable def area_ratio_of_ABC_and_GHJ : ℝ :=
  let side_length_ABC := 12
  let BD := 5
  let CE := 5
  let AF := 8
  let area_ABC := (Real.sqrt 3 / 4) * side_length_ABC ^ 2
  (1 / 74338) * area_ABC / area_ABC

theorem area_ratio_correct : area_ratio_of_ABC_and_GHJ = 1 / 74338 := by
  sorry

end area_ratio_correct_l2103_210341


namespace Katya_possible_numbers_l2103_210330

def divisible_by (n m : ℕ) : Prop := m % n = 0

def possible_numbers (n : ℕ) : Prop :=
  let condition1 := divisible_by 7 n  -- Alyona's condition
  let condition2 := divisible_by 5 n  -- Lena's condition
  let condition3 := n < 9             -- Rita's condition
  (condition1 ∨ condition2) ∧ condition3 ∧ 
  ((condition1 ∧ condition3 ∧ ¬condition2) ∨ (condition2 ∧ condition3 ∧ ¬condition1))

theorem Katya_possible_numbers :
  ∀ n : ℕ, 
    (possible_numbers n) ↔ (n = 5 ∨ n = 7) :=
sorry

end Katya_possible_numbers_l2103_210330


namespace ratio_accepted_rejected_l2103_210347

-- Definitions for the conditions given
def eggs_per_day : ℕ := 400
def ratio_accepted_to_rejected : ℕ × ℕ := (96, 4)
def additional_accepted_eggs : ℕ := 12

/-- The ratio of accepted eggs to rejected eggs on that particular day is 99:1. -/
theorem ratio_accepted_rejected (a r : ℕ) (h1 : ratio_accepted_to_rejected = (a, r)) 
  (h2 : (a + r) * (eggs_per_day / (a + r)) = eggs_per_day) 
  (h3 : additional_accepted_eggs = 12) :
  (a + additional_accepted_eggs) / r = 99 :=
  sorry

end ratio_accepted_rejected_l2103_210347


namespace B_contains_only_one_element_l2103_210335

def setA := { x | (x - 1/2) * (x - 3) = 0 }

def setB (a : ℝ) := { x | Real.log (x^2 + a * x + a + 9 / 4) = 0 }

theorem B_contains_only_one_element (a : ℝ) :
  (∃ x, setB a x ∧ ∀ y, setB a y → y = x) →
  (a = 5 ∨ a = -1) :=
by
  intro h
  -- Proof would go here
  sorry

end B_contains_only_one_element_l2103_210335


namespace sock_pairs_proof_l2103_210370

noncomputable def numPairsOfSocks : ℕ :=
  let n : ℕ := sorry
  n

theorem sock_pairs_proof : numPairsOfSocks = 6 := by
  sorry

end sock_pairs_proof_l2103_210370


namespace initial_population_l2103_210375

/-- The population of a town decreases annually at the rate of 20% p.a.
    Given that the population of the town after 2 years is 19200,
    prove that the initial population of the town was 30,000. -/
theorem initial_population (P : ℝ) (h : 0.64 * P = 19200) : P = 30000 :=
sorry

end initial_population_l2103_210375


namespace typing_page_percentage_l2103_210315

/--
Given:
- Original sheet dimensions are 20 cm by 30 cm.
- Margins are 2 cm on each side (left and right), and 3 cm on the top and bottom.
Prove that the percentage of the page used by the typist is 64%.
-/
theorem typing_page_percentage (width height margin_lr margin_tb : ℝ)
  (h1 : width = 20) 
  (h2 : height = 30) 
  (h3 : margin_lr = 2) 
  (h4 : margin_tb = 3) : 
  (width - 2 * margin_lr) * (height - 2 * margin_tb) / (width * height) * 100 = 64 :=
by
  sorry

end typing_page_percentage_l2103_210315


namespace symmetry_about_origin_l2103_210381

def Point : Type := ℝ × ℝ

def A : Point := (2, -1)
def B : Point := (-2, 1)

theorem symmetry_about_origin (A B : Point) : A = (2, -1) ∧ B = (-2, 1) → B = (-A.1, -A.2) :=
by
  sorry

end symmetry_about_origin_l2103_210381


namespace original_number_of_men_l2103_210363

theorem original_number_of_men 
  (x : ℕ)
  (H : 15 * 18 * x = 15 * 18 * (x - 8) + 8 * 15 * 18)
  (h_pos : x > 8) :
  x = 40 :=
sorry

end original_number_of_men_l2103_210363


namespace arithmetic_geometric_sequence_product_l2103_210321

theorem arithmetic_geometric_sequence_product :
  (∀ n : ℕ, ∃ d : ℝ, ∀ m : ℕ, a_n = a_1 + m * d) →
  (∀ n : ℕ, ∃ q : ℝ, ∀ m : ℕ, b_n = b_1 * q ^ m) →
  a_1 = 1 → a_2 = 2 →
  b_1 = 1 → b_2 = 2 →
  a_5 * b_5 = 80 :=
by
  sorry

end arithmetic_geometric_sequence_product_l2103_210321
