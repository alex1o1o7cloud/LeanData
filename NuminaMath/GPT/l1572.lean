import Mathlib

namespace NUMINAMATH_GPT_polygon_sides_l1572_157295

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1572_157295


namespace NUMINAMATH_GPT_parabola_directrix_is_x_eq_1_l1572_157265

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_is_x_eq_1_l1572_157265


namespace NUMINAMATH_GPT_sequence_general_formula_l1572_157299

theorem sequence_general_formula :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (∀ n : ℕ, n > 0 → a n - a (n + 1) = 2 * a n * a (n + 1) / (n * (n + 1))) →
  ∀ n : ℕ, n > 0 → a n = n / (3 * n - 2) :=
by
  intros a h1 h_rec n hn
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1572_157299


namespace NUMINAMATH_GPT_line_intersects_circle_l1572_157227

theorem line_intersects_circle (α : ℝ) (r : ℝ) (hα : true) (hr : r > 0) :
  (∃ x y : ℝ, (x * Real.cos α + y * Real.sin α = 1) ∧ (x^2 + y^2 = r^2)) → r > 1 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l1572_157227


namespace NUMINAMATH_GPT_product_of_two_numbers_l1572_157211

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.lcm a b = 72) (h2 : Nat.gcd a b = 8) :
  a * b = 576 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1572_157211


namespace NUMINAMATH_GPT_fill_cistern_l1572_157233

theorem fill_cistern (p_rate q_rate : ℝ) (total_time first_pipe_time : ℝ) (remaining_fraction : ℝ): 
  p_rate = 1/12 → q_rate = 1/15 → total_time = 2 → remaining_fraction = 7/10 → 
  (remaining_fraction / q_rate) = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_fill_cistern_l1572_157233


namespace NUMINAMATH_GPT_hypotenuse_length_l1572_157270

open Real

-- Definitions corresponding to the conditions
def right_triangle_vertex_length (ADC_length : ℝ) (AEC_length : ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ ADC_length = sqrt 3 * sin x ∧ AEC_length = sin x

def trisect_hypotenuse (BD : ℝ) (DE : ℝ) (EC : ℝ) (c : ℝ) : Prop :=
  BD = c / 3 ∧ DE = c / 3 ∧ EC = c / 3

-- Main theorem definition
theorem hypotenuse_length (x hypotenuse ADC_length AEC_length : ℝ) :
  right_triangle_vertex_length ADC_length AEC_length x →
  trisect_hypotenuse (hypotenuse / 3) (hypotenuse / 3) (hypotenuse / 3) hypotenuse →
  hypotenuse = sqrt 3 * sin x :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1572_157270


namespace NUMINAMATH_GPT_net_rate_25_dollars_per_hour_l1572_157231

noncomputable def net_rate_of_pay (hours : ℕ) (speed : ℕ) (mileage : ℕ) (rate_per_mile : ℚ) (diesel_cost_per_gallon : ℚ) : ℚ :=
  let distance := hours * speed
  let diesel_used := distance / mileage
  let earnings := rate_per_mile * distance
  let diesel_cost := diesel_cost_per_gallon * diesel_used
  let net_earnings := earnings - diesel_cost
  net_earnings / hours

theorem net_rate_25_dollars_per_hour :
  net_rate_of_pay 4 45 15 (0.75 : ℚ) (3.00 : ℚ) = 25 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_net_rate_25_dollars_per_hour_l1572_157231


namespace NUMINAMATH_GPT_vanessa_savings_remaining_l1572_157246

-- Conditions
def initial_investment : ℝ := 50000
def annual_interest_rate : ℝ := 0.035
def investment_duration : ℕ := 3
def conversion_rate : ℝ := 0.85
def cost_per_toy : ℝ := 75

-- Given the above conditions, prove the remaining amount in euros after buying as many toys as possible is 16.9125
theorem vanessa_savings_remaining
  (P : ℝ := initial_investment)
  (r : ℝ := annual_interest_rate)
  (t : ℕ := investment_duration)
  (c : ℝ := conversion_rate)
  (e : ℝ := cost_per_toy) :
  (((P * (1 + r)^t) * c) - (e * (⌊(P * (1 + r)^3 * 0.85) / e⌋))) = 16.9125 :=
sorry

end NUMINAMATH_GPT_vanessa_savings_remaining_l1572_157246


namespace NUMINAMATH_GPT_min_pos_int_k_l1572_157225

noncomputable def minimum_k (x0 : ℝ) : ℝ := (x0 * (Real.log x0 + 1)) / (x0 - 2)

theorem min_pos_int_k : ∃ k : ℝ, (∀ x0 : ℝ, x0 > 2 → k > minimum_k x0) ∧ k = 5 := 
by
  sorry

end NUMINAMATH_GPT_min_pos_int_k_l1572_157225


namespace NUMINAMATH_GPT_time_to_fill_tank_with_leak_l1572_157251

-- Definitions based on the given conditions:
def rate_of_pipe_A := 1 / 6 -- Pipe A fills the tank in 6 hours
def rate_of_leak := 1 / 12 -- The leak empties the tank in 12 hours
def combined_rate := rate_of_pipe_A - rate_of_leak -- Combined rate with leak

-- The proof problem: Prove the time taken to fill the tank with the leak present is 12 hours.
theorem time_to_fill_tank_with_leak : 
  (1 / combined_rate) = 12 := by
    -- Proof goes here...
    sorry

end NUMINAMATH_GPT_time_to_fill_tank_with_leak_l1572_157251


namespace NUMINAMATH_GPT_scientific_notation_correct_l1572_157206

theorem scientific_notation_correct :
  27600 = 2.76 * 10^4 :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1572_157206


namespace NUMINAMATH_GPT_solve_equations_l1572_157230

theorem solve_equations (x : ℝ) :
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l1572_157230


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_minimum_value_of_f_in_interval_l1572_157210

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def f (x m : ℝ) : ℝ := (vec_a x).1 * vec_b.1 + (vec_a x).2 * vec_b.2 + m

theorem smallest_positive_period_of_f :
  ∀ (x : ℝ) (m : ℝ), ∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) m = f x m) → p = Real.pi := 
sorry

theorem minimum_value_of_f_in_interval :
  ∀ (x m : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → ∃ m : ℝ, (∀ x : ℝ, f x m ≥ 5) ∧ m = 5 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_minimum_value_of_f_in_interval_l1572_157210


namespace NUMINAMATH_GPT_parabola_directrix_l1572_157214

theorem parabola_directrix (y : ℝ) : (∃ p : ℝ, x = (1 / (4 * p)) * y^2 ∧ p = 2) → x = -2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1572_157214


namespace NUMINAMATH_GPT_remainder_7_pow_137_mod_11_l1572_157202

theorem remainder_7_pow_137_mod_11 :
    (137 = 13 * 10 + 7) →
    (7^10 ≡ 1 [MOD 11]) →
    (7^137 ≡ 6 [MOD 11]) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_remainder_7_pow_137_mod_11_l1572_157202


namespace NUMINAMATH_GPT_fraction_meaningful_l1572_157242

-- Define the condition about the denominator not being zero.
def denominator_condition (x : ℝ) : Prop := x + 2 ≠ 0

-- The proof problem statement.
theorem fraction_meaningful (x : ℝ) : denominator_condition x ↔ x ≠ -2 :=
by
  -- Ensure that the Lean environment is aware this is a theorem statement.
  sorry -- Proof is omitted as instructed.

end NUMINAMATH_GPT_fraction_meaningful_l1572_157242


namespace NUMINAMATH_GPT_grandson_age_l1572_157245

theorem grandson_age (M S G : ℕ) (h1 : M = 2 * S) (h2 : S = 2 * G) (h3 : M + S + G = 140) : G = 20 :=
by 
  sorry

end NUMINAMATH_GPT_grandson_age_l1572_157245


namespace NUMINAMATH_GPT_calc_g_f_neg_2_l1572_157221

def f (x : ℝ) : ℝ := x^3 - 4 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 1

theorem calc_g_f_neg_2 : g (f (-2)) = 25 := by
  sorry

end NUMINAMATH_GPT_calc_g_f_neg_2_l1572_157221


namespace NUMINAMATH_GPT_positional_relationship_l1572_157257

variables {Point Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Condition 1: Line a is parallel to Plane α
def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry

-- Condition 2: Line b is contained within Plane α
def line_contained_within_plane (b : Line) (α : Plane) : Prop := sorry

-- The positional relationship between line a and line b is either parallel or skew
def lines_parallel_or_skew (a b : Line) : Prop := sorry

theorem positional_relationship (ha : line_parallel_to_plane a α) (hb : line_contained_within_plane b α) :
  lines_parallel_or_skew a b :=
sorry

end NUMINAMATH_GPT_positional_relationship_l1572_157257


namespace NUMINAMATH_GPT_perimeter_F_is_18_l1572_157296

-- Define the dimensions of the rectangles.
def vertical_rectangle : ℤ × ℤ := (3, 5)
def horizontal_rectangle : ℤ × ℤ := (1, 5)

-- Define the perimeter calculation for a single rectangle.
def perimeter (width_height : ℤ × ℤ) : ℤ :=
  2 * width_height.1 + 2 * width_height.2

-- The overlapping width and height.
def overlap_width : ℤ := 5
def overlap_height : ℤ := 1

-- Perimeter of the letter F.
def perimeter_F : ℤ :=
  perimeter vertical_rectangle + perimeter horizontal_rectangle - 2 * overlap_width

-- Statement to prove.
theorem perimeter_F_is_18 : perimeter_F = 18 := by sorry

end NUMINAMATH_GPT_perimeter_F_is_18_l1572_157296


namespace NUMINAMATH_GPT_participants_in_sports_activities_l1572_157232

theorem participants_in_sports_activities:
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = 3 ∧
  let a := 10 * x + 6
  let b := 10 * y + 6
  let c := 10 * z + 6
  a + b + c = 48 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a = 6 ∧ b = 16 ∧ c = 26 ∨ a = 6 ∧ b = 26 ∧ c = 16 ∨ a = 16 ∧ b = 6 ∧ c = 26 ∨ a = 16 ∧ b = 26 ∧ c = 6 ∨ a = 26 ∧ b = 6 ∧ c = 16 ∨ a = 26 ∧ b = 16 ∧ c = 6)
  :=
by {
  sorry
}

end NUMINAMATH_GPT_participants_in_sports_activities_l1572_157232


namespace NUMINAMATH_GPT_dara_jane_age_ratio_l1572_157226

theorem dara_jane_age_ratio :
  ∀ (min_age : ℕ) (jane_current_age : ℕ) (dara_years_til_min_age : ℕ) (d : ℕ) (j : ℕ),
  min_age = 25 →
  jane_current_age = 28 →
  dara_years_til_min_age = 14 →
  d = 17 →
  j = 34 →
  d = dara_years_til_min_age - 14 + 6 →
  j = jane_current_age + 6 →
  (d:ℚ) / j = 1 / 2 := 
by
  intros
  sorry

end NUMINAMATH_GPT_dara_jane_age_ratio_l1572_157226


namespace NUMINAMATH_GPT_min_value_f_l1572_157269

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 3) / (x - 1)

theorem min_value_f : ∀ (x : ℝ), x ≥ 3 → ∃ m : ℝ, m = 9/2 ∧ ∀ y : ℝ, f y ≥ m :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1572_157269


namespace NUMINAMATH_GPT_sufficient_condition_a_gt_1_l1572_157213

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end NUMINAMATH_GPT_sufficient_condition_a_gt_1_l1572_157213


namespace NUMINAMATH_GPT_cody_initial_tickets_l1572_157266

def initial_tickets (lost : ℝ) (spent : ℝ) (left : ℝ) : ℝ :=
  lost + spent + left

theorem cody_initial_tickets : initial_tickets 6.0 25.0 18.0 = 49.0 := by
  sorry

end NUMINAMATH_GPT_cody_initial_tickets_l1572_157266


namespace NUMINAMATH_GPT_third_side_length_not_12_l1572_157283

theorem third_side_length_not_12 (x : ℕ) (h1 : x % 2 = 0) (h2 : 5 < x) (h3 : x < 11) : x ≠ 12 := 
sorry

end NUMINAMATH_GPT_third_side_length_not_12_l1572_157283


namespace NUMINAMATH_GPT_age_of_17th_student_is_75_l1572_157200

variables (T A : ℕ)

def avg_17_students := 17
def avg_5_students := 14
def avg_9_students := 16
def total_17_students := 17 * avg_17_students
def total_5_students := 5 * avg_5_students
def total_9_students := 9 * avg_9_students
def age_17th_student : ℕ := total_17_students - (total_5_students + total_9_students)

theorem age_of_17th_student_is_75 :
  age_17th_student = 75 := by sorry

end NUMINAMATH_GPT_age_of_17th_student_is_75_l1572_157200


namespace NUMINAMATH_GPT_coprime_gcd_l1572_157249

theorem coprime_gcd (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (2 * a + b) (a * (a + b)) = 1 := 
sorry

end NUMINAMATH_GPT_coprime_gcd_l1572_157249


namespace NUMINAMATH_GPT_emily_weight_l1572_157216

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end NUMINAMATH_GPT_emily_weight_l1572_157216


namespace NUMINAMATH_GPT_game_ends_in_65_rounds_l1572_157250

noncomputable def player_tokens_A : Nat := 20
noncomputable def player_tokens_B : Nat := 19
noncomputable def player_tokens_C : Nat := 18
noncomputable def player_tokens_D : Nat := 17

def rounds_until_game_ends (A B C D : Nat) : Nat :=
  -- Implementation to count the rounds will go here, but it is skipped for this statement-only task
  sorry

theorem game_ends_in_65_rounds : rounds_until_game_ends player_tokens_A player_tokens_B player_tokens_C player_tokens_D = 65 :=
  sorry

end NUMINAMATH_GPT_game_ends_in_65_rounds_l1572_157250


namespace NUMINAMATH_GPT_sequence_solution_l1572_157235

theorem sequence_solution :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℝ),
    a 1 = 2 ∧
    (∀ n, b n = (a (n + 1)) / (a n)) ∧
    b 10 * b 11 = 2 →
    a 21 = 2 ^ 11 :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l1572_157235


namespace NUMINAMATH_GPT_total_yardage_progress_l1572_157247

def teamA_moves : List Int := [-5, 8, -3, 6]
def teamB_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress :
  (teamA_moves.sum + teamB_moves.sum) = 10 :=
by
  sorry

end NUMINAMATH_GPT_total_yardage_progress_l1572_157247


namespace NUMINAMATH_GPT_part1_part2_l1572_157260

def setA (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 3 - 2 * a}
def setB := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}

theorem part1 (a : ℝ) : (setA a ∪ setB = setB) ↔ (-(1 / 2) ≤ a) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ∈ setB ↔ x ∈ setA a) ↔ (a ≤ -1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1572_157260


namespace NUMINAMATH_GPT_probability_answered_within_first_four_rings_l1572_157288

theorem probability_answered_within_first_four_rings 
  (P1 P2 P3 P4 : ℝ) (h1 : P1 = 0.1) (h2 : P2 = 0.3) (h3 : P3 = 0.4) (h4 : P4 = 0.1) :
  (1 - ((1 - P1) * (1 - P2) * (1 - P3) * (1 - P4))) = 0.9 := 
sorry

end NUMINAMATH_GPT_probability_answered_within_first_four_rings_l1572_157288


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l1572_157236

theorem sum_of_reciprocals_of_roots (r1 r2 : ℚ) (h_sum : r1 + r2 = 17) (h_prod : r1 * r2 = 6) :
  1 / r1 + 1 / r2 = 17 / 6 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l1572_157236


namespace NUMINAMATH_GPT_sum_of_six_selected_primes_is_even_l1572_157293

noncomputable def prob_sum_even_when_selecting_six_primes : ℚ := 
  let first_twenty_primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
  let num_ways_to_choose_6_without_even_sum := Nat.choose 19 6
  let total_num_ways_to_choose_6 := Nat.choose 20 6
  num_ways_to_choose_6_without_even_sum / total_num_ways_to_choose_6

theorem sum_of_six_selected_primes_is_even : 
  prob_sum_even_when_selecting_six_primes = 354 / 505 := 
sorry

end NUMINAMATH_GPT_sum_of_six_selected_primes_is_even_l1572_157293


namespace NUMINAMATH_GPT_S_shaped_growth_curve_varied_growth_rate_l1572_157263

theorem S_shaped_growth_curve_varied_growth_rate :
  ∀ (population_growth : ℝ → ℝ), 
    (∃ t1 t2 : ℝ, t1 < t2 ∧ 
      (∃ r : ℝ, r = population_growth t1 / t1 ∧ r ≠ population_growth t2 / t2)) 
    → 
    ∀ t3 t4 : ℝ, t3 < t4 → (population_growth t3 / t3) ≠ (population_growth t4 / t4) :=
by
  sorry

end NUMINAMATH_GPT_S_shaped_growth_curve_varied_growth_rate_l1572_157263


namespace NUMINAMATH_GPT_product_of_two_real_numbers_sum_three_times_product_l1572_157244

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem product_of_two_real_numbers_sum_three_times_product
    (h : x + y = 3 * x * y) :
  x * y = (x + y) / 3 :=
sorry

end NUMINAMATH_GPT_product_of_two_real_numbers_sum_three_times_product_l1572_157244


namespace NUMINAMATH_GPT_probability_at_least_one_of_each_color_l1572_157205

theorem probability_at_least_one_of_each_color
  (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h_total : total_balls = 16)
  (h_black : black_balls = 8)
  (h_white : white_balls = 5)
  (h_red : red_balls = 3) :
  ((black_balls.choose 1) * (white_balls.choose 1) * (red_balls.choose 1) : ℚ) / total_balls.choose 3 = 3 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_of_each_color_l1572_157205


namespace NUMINAMATH_GPT_reduced_expression_none_of_these_l1572_157291

theorem reduced_expression_none_of_these (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : b ≠ a^2) (h4 : ab ≠ a^3) :
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 1 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (b^2 + b) / (b - a^2) ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 0 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (a^2 + b) / (a^2 - b) :=
by
  sorry

end NUMINAMATH_GPT_reduced_expression_none_of_these_l1572_157291


namespace NUMINAMATH_GPT_range_of_k_l1572_157207

theorem range_of_k (k : ℝ) : 
  (∀ x, x ∈ {x | -3 ≤ x ∧ x ≤ 2} ∩ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1} ↔ x ∈ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1}) →
   -1 ≤ k ∧ k ≤ 1 / 2 :=
by sorry

end NUMINAMATH_GPT_range_of_k_l1572_157207


namespace NUMINAMATH_GPT_deposits_exceed_10_on_second_Tuesday_l1572_157208

noncomputable def deposits_exceed_10 (n : ℕ) : ℕ :=
2 * (2^n - 1)

theorem deposits_exceed_10_on_second_Tuesday :
  ∃ n, deposits_exceed_10 n > 1000 ∧ 1 + (n - 1) % 7 = 2 ∧ n < 21 :=
sorry

end NUMINAMATH_GPT_deposits_exceed_10_on_second_Tuesday_l1572_157208


namespace NUMINAMATH_GPT_no_possible_arrangement_of_balloons_l1572_157273

/-- 
  There are 10 balloons hanging in a row: blue and green. This statement proves that it is impossible 
  to arrange 10 balloons such that between every two blue balloons, there is an even number of 
  balloons and between every two green balloons, there is an odd number of balloons.
--/

theorem no_possible_arrangement_of_balloons :
  ¬ (∃ (color : Fin 10 → Bool), 
    (∀ i j, i < j ∧ color i = color j ∧ color i = tt → (j - i - 1) % 2 = 0) ∧
    (∀ i j, i < j ∧ color i = color j ∧ color i = ff → (j - i - 1) % 2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_possible_arrangement_of_balloons_l1572_157273


namespace NUMINAMATH_GPT_katie_spending_l1572_157204

theorem katie_spending :
  let price_per_flower : ℕ := 6
  let number_of_roses : ℕ := 5
  let number_of_daisies : ℕ := 5
  let total_number_of_flowers := number_of_roses + number_of_daisies
  let total_spending := total_number_of_flowers * price_per_flower
  total_spending = 60 :=
by
  sorry

end NUMINAMATH_GPT_katie_spending_l1572_157204


namespace NUMINAMATH_GPT_inequality_abc_l1572_157271

theorem inequality_abc (a b c : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) (h5 : 2 ≤ n) :
  (a / (b + c)^(1/(n:ℝ)) + b / (c + a)^(1/(n:ℝ)) + c / (a + b)^(1/(n:ℝ)) ≥ 3 / 2^(1/(n:ℝ))) :=
by sorry

end NUMINAMATH_GPT_inequality_abc_l1572_157271


namespace NUMINAMATH_GPT_sin_cos_value_l1572_157298

variable (α : ℝ) (a b : ℝ × ℝ)
def vectors_parallel : Prop := b = (Real.sin α, Real.cos α) ∧
a = (4, 3) ∧ (∃ k : ℝ, a = (k * (Real.sin α), k * (Real.cos α)))

theorem sin_cos_value (h : vectors_parallel α a b) : ((Real.sin α) * (Real.cos α)) = 12 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_value_l1572_157298


namespace NUMINAMATH_GPT_train_stop_duration_l1572_157254

theorem train_stop_duration (speed_without_stoppages speed_with_stoppages : ℕ) (h1 : speed_without_stoppages = 45) (h2 : speed_with_stoppages = 42) :
  ∃ t : ℕ, t = 4 :=
by
  sorry

end NUMINAMATH_GPT_train_stop_duration_l1572_157254


namespace NUMINAMATH_GPT_SunshinePumpkinsCount_l1572_157217

def MoonglowPumpkins := 14
def SunshinePumpkins := 3 * MoonglowPumpkins + 12

theorem SunshinePumpkinsCount : SunshinePumpkins = 54 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_SunshinePumpkinsCount_l1572_157217


namespace NUMINAMATH_GPT_beautiful_fold_probability_l1572_157280

noncomputable def probability_beautiful_fold (a : ℝ) : ℝ := 1 / 2

theorem beautiful_fold_probability 
  (A B C D F : ℝ × ℝ) 
  (ABCD_square : (A.1 = 0) ∧ (A.2 = 0) ∧ 
                 (B.1 = a) ∧ (B.2 = 0) ∧ 
                 (C.1 = a) ∧ (C.2 = a) ∧ 
                 (D.1 = 0) ∧ (D.2 = a))
  (F_in_square : 0 ≤ F.1 ∧ F.1 ≤ a ∧ 0 ≤ F.2 ∧ F.2 ≤ a):
  probability_beautiful_fold a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_beautiful_fold_probability_l1572_157280


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_l1572_157259

theorem line_intersects_x_axis_at (a b : ℝ) (h1 : a = 12) (h2 : b = 2)
  (c d : ℝ) (h3 : c = 6) (h4 : d = 6) : 
  ∃ x : ℝ, (x, 0) = (15, 0) := 
by
  -- proof needed here
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_l1572_157259


namespace NUMINAMATH_GPT_problem_l1572_157285

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end NUMINAMATH_GPT_problem_l1572_157285


namespace NUMINAMATH_GPT_lines_coplanar_l1572_157262

/-
Given:
- Line 1 parameterized as (2 + s, 4 - k * s, -1 + k * s)
- Line 2 parameterized as (2 * t, 2 + t, 3 - t)
Prove: If these lines are coplanar, then k = -1/2
-/
theorem lines_coplanar (k : ℚ) (s t : ℚ)
  (line1 : ℚ × ℚ × ℚ := (2 + s, 4 - k * s, -1 + k * s))
  (line2 : ℚ × ℚ × ℚ := (2 * t, 2 + t, 3 - t))
  (coplanar : ∃ (s t : ℚ), line1 = line2) :
  k = -1 / 2 := 
sorry

end NUMINAMATH_GPT_lines_coplanar_l1572_157262


namespace NUMINAMATH_GPT_pears_morning_sales_l1572_157294

theorem pears_morning_sales (morning afternoon : ℕ) 
  (h1 : afternoon = 2 * morning)
  (h2 : morning + afternoon = 360) : 
  morning = 120 := 
sorry

end NUMINAMATH_GPT_pears_morning_sales_l1572_157294


namespace NUMINAMATH_GPT_correct_result_l1572_157290

-- Definitions to capture the problem conditions:
def cond1 (a b : ℤ) : Prop := 5 * a^2 * b - 2 * a^2 * b = 3 * a^2 * b
def cond2 (x : ℤ) : Prop := x^6 / x^2 = x^4
def cond3 (a b : ℤ) : Prop := (a - b)^2 = a^2 - b^2

-- Proof statement to verify the correct answer
theorem correct_result (x : ℤ) : (2 * x^2)^3 = 8 * x^6 :=
  by sorry

-- Note that cond1, cond2, and cond3 are intended to capture the erroneous conditions mentioned for completeness.

end NUMINAMATH_GPT_correct_result_l1572_157290


namespace NUMINAMATH_GPT_solve_equation_l1572_157203

theorem solve_equation (x : ℝ) (h : (x - 3) / 2 - (2 * x) / 3 = 1) : x = -15 := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l1572_157203


namespace NUMINAMATH_GPT_books_sum_l1572_157286

theorem books_sum (darryl_books lamont_books loris_books danielle_books : ℕ) 
  (h1 : darryl_books = 20)
  (h2 : lamont_books = 2 * darryl_books)
  (h3 : lamont_books = loris_books + 3)
  (h4 : danielle_books = lamont_books + darryl_books + 10) : 
  darryl_books + lamont_books + loris_books + danielle_books = 167 := 
by
  sorry

end NUMINAMATH_GPT_books_sum_l1572_157286


namespace NUMINAMATH_GPT_resting_time_is_thirty_l1572_157276

-- Defining the conditions as Lean 4 definitions
def speed := 10 -- miles per hour
def time_first_part := 30 -- minutes
def distance_second_part := 15 -- miles
def distance_third_part := 20 -- miles
def total_time := 270 -- minutes

-- Function to convert hours to minutes
def hours_to_minutes (h : ℕ) : ℕ := h * 60

-- Problem statement in Lean 4: Proving the resting time is 30 minutes
theorem resting_time_is_thirty :
  let distance_first := speed * (time_first_part / 60)
  let time_second_part := (distance_second_part / speed) * 60
  let time_third_part := (distance_third_part / speed) * 60
  let times_sum := time_first_part + time_second_part + time_third_part
  total_time = times_sum + 30 := 
  sorry

end NUMINAMATH_GPT_resting_time_is_thirty_l1572_157276


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l1572_157255

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem rectangular_solid_surface_area (l w h : ℕ) (hl : is_prime l) (hw : is_prime w) (hh : is_prime h) (volume_eq_437 : l * w * h = 437) :
  2 * (l * w + w * h + h * l) = 958 :=
sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l1572_157255


namespace NUMINAMATH_GPT_totalMoney_l1572_157240

noncomputable def totalAmount (x : ℝ) : ℝ := 15 * x

theorem totalMoney (x : ℝ) (h : 1.8 * x = 9) : totalAmount x = 75 :=
by sorry

end NUMINAMATH_GPT_totalMoney_l1572_157240


namespace NUMINAMATH_GPT_binary_multiplication_l1572_157228

theorem binary_multiplication :
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  a * b = product :=
by 
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  sorry

end NUMINAMATH_GPT_binary_multiplication_l1572_157228


namespace NUMINAMATH_GPT_percent_university_diploma_no_job_choice_l1572_157238

theorem percent_university_diploma_no_job_choice
    (total_people : ℕ)
    (P1 : 10 * total_people / 100 = total_people / 10)
    (P2 : 20 * total_people / 100 = total_people / 5)
    (P3 : 30 * total_people / 100 = 3 * total_people / 10) :
  25 = (20 * total_people / (80 * total_people / 100)) :=
by
  sorry

end NUMINAMATH_GPT_percent_university_diploma_no_job_choice_l1572_157238


namespace NUMINAMATH_GPT_trig_problem_l1572_157287

variable (α : ℝ)

theorem trig_problem
  (h1 : Real.sin (Real.pi + α) = -1 / 3) :
  Real.cos (α - 3 * Real.pi / 2) = -1 / 3 ∧
  (Real.sin (Real.pi / 2 + α) = 2 * Real.sqrt 2 / 3 ∨ Real.sin (Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3) ∧
  (Real.tan (5 * Real.pi - α) = -Real.sqrt 2 / 4 ∨ Real.tan (5 * Real.pi - α) = Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_GPT_trig_problem_l1572_157287


namespace NUMINAMATH_GPT_Craig_bench_press_percentage_l1572_157292

theorem Craig_bench_press_percentage {Dave_weight : ℕ} (h1 : Dave_weight = 175) (h2 : ∀ w : ℕ, Dave_bench_press = 3 * Dave_weight) 
(Craig_bench_press Mark_bench_press : ℕ) (h3 : Mark_bench_press = 55) (h4 : Mark_bench_press = Craig_bench_press - 50) : 
(Craig_bench_press / (3 * Dave_weight) * 100) = 20 := by
  sorry

end NUMINAMATH_GPT_Craig_bench_press_percentage_l1572_157292


namespace NUMINAMATH_GPT_sum_of_second_and_third_of_four_consecutive_even_integers_l1572_157268

-- Definitions of conditions
variables (n : ℤ)  -- Assume n is an integer

-- Statement of problem
theorem sum_of_second_and_third_of_four_consecutive_even_integers (h : 2 * n + 6 = 160) :
  (n + 2) + (n + 4) = 160 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_second_and_third_of_four_consecutive_even_integers_l1572_157268


namespace NUMINAMATH_GPT_jelly_bean_remaining_l1572_157237

theorem jelly_bean_remaining (J : ℕ) (P : ℕ) (taken_last_4_each : ℕ) (taken_first_each : ℕ) 
 (taken_last_total : ℕ) (taken_first_total : ℕ) (taken_total : ℕ) (remaining : ℕ) :
  J = 8000 →
  P = 10 →
  taken_last_4_each = 400 →
  taken_first_each = 2 * taken_last_4_each →
  taken_last_total = 4 * taken_last_4_each →
  taken_first_total = 6 * taken_first_each →
  taken_total = taken_last_total + taken_first_total →
  remaining = J - taken_total →
  remaining = 1600 :=
by
  intros
  sorry  

end NUMINAMATH_GPT_jelly_bean_remaining_l1572_157237


namespace NUMINAMATH_GPT_square_area_from_isosceles_triangle_l1572_157243

theorem square_area_from_isosceles_triangle:
  ∀ (b h : ℝ) (Side_of_Square : ℝ), b = 2 ∧ h = 3 ∧ Side_of_Square = (6 / 5) 
  → (Side_of_Square ^ 2) = (36 / 25) := 
by
  intro b h Side_of_Square
  rintro ⟨hb, hh, h_side⟩
  sorry

end NUMINAMATH_GPT_square_area_from_isosceles_triangle_l1572_157243


namespace NUMINAMATH_GPT_find_R_value_l1572_157239

noncomputable def x (Q : ℝ) : ℝ := Real.sqrt (Q / 2 + Real.sqrt (Q / 2))
noncomputable def y (Q : ℝ) : ℝ := Real.sqrt (Q / 2 - Real.sqrt (Q / 2))
noncomputable def R (Q : ℝ) : ℝ := (x Q)^6 + (y Q)^6 / 40

theorem find_R_value (Q : ℝ) : R Q = 10 :=
sorry

end NUMINAMATH_GPT_find_R_value_l1572_157239


namespace NUMINAMATH_GPT_find_actual_average_height_l1572_157223

noncomputable def actualAverageHeight (avg_height : ℕ) (num_boys : ℕ) (wrong_height : ℕ) (actual_height : ℕ) : Float :=
  let incorrect_total := avg_height * num_boys
  let difference := wrong_height - actual_height
  let correct_total := incorrect_total - difference
  (Float.ofInt correct_total) / (Float.ofNat num_boys)

theorem find_actual_average_height (avg_height num_boys wrong_height actual_height : ℕ) :
  avg_height = 185 ∧ num_boys = 35 ∧ wrong_height = 166 ∧ actual_height = 106 →
  actualAverageHeight avg_height num_boys wrong_height actual_height = 183.29 := by
  intros h
  have h_avg := h.1
  have h_num := h.2.1
  have h_wrong := h.2.2.1
  have h_actual := h.2.2.2
  rw [h_avg, h_num, h_wrong, h_actual]
  sorry

end NUMINAMATH_GPT_find_actual_average_height_l1572_157223


namespace NUMINAMATH_GPT_angles_of_triangle_l1572_157201

theorem angles_of_triangle 
  (α β γ : ℝ)
  (triangle_ABC : α + β + γ = 180)
  (median_bisector_height : (γ / 4) * 4 = 90) :
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end NUMINAMATH_GPT_angles_of_triangle_l1572_157201


namespace NUMINAMATH_GPT_milk_percentage_after_adding_water_l1572_157219

theorem milk_percentage_after_adding_water
  (initial_total_volume : ℚ) (initial_milk_percentage : ℚ)
  (additional_water_volume : ℚ) :
  initial_total_volume = 60 → initial_milk_percentage = 0.84 → additional_water_volume = 18.75 →
  (50.4 / (initial_total_volume + additional_water_volume) * 100 = 64) :=
by
  intros h1 h2 h3
  rw [h1, h3]
  simp
  sorry

end NUMINAMATH_GPT_milk_percentage_after_adding_water_l1572_157219


namespace NUMINAMATH_GPT_not_even_or_odd_l1572_157277

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem not_even_or_odd : ¬(∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_GPT_not_even_or_odd_l1572_157277


namespace NUMINAMATH_GPT_sequence_property_l1572_157248

noncomputable def seq (n : ℕ) : ℕ := 
if n = 0 then 1 else 
if n = 1 then 3 else 
seq (n-2) + 3 * 2^(n-2)

theorem sequence_property {n : ℕ} (h_pos : n > 0) :
(∀ n : ℕ, n > 0 → seq (n + 2) ≤ seq n + 3 * 2^n) →
(∀ n : ℕ, n > 0 → seq (n + 1) ≥ 2 * seq n + 1) →
seq n = 2^n - 1 := 
sorry

end NUMINAMATH_GPT_sequence_property_l1572_157248


namespace NUMINAMATH_GPT_xyz_values_l1572_157258

theorem xyz_values (x y z : ℝ)
  (h1 : x * y - 5 * y = 20)
  (h2 : y * z - 5 * z = 20)
  (h3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := 
by sorry

end NUMINAMATH_GPT_xyz_values_l1572_157258


namespace NUMINAMATH_GPT_cost_price_percentage_l1572_157224

variable (SP CP : ℝ)

-- Assumption that the profit percent is 25%
axiom profit_percent : 25 = ((SP - CP) / CP) * 100

-- The statement to prove
theorem cost_price_percentage : CP / SP = 0.8 := by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l1572_157224


namespace NUMINAMATH_GPT_initial_weights_of_apples_l1572_157282

variables {A B : ℕ}

theorem initial_weights_of_apples (h₁ : A + B = 75) (h₂ : A - 5 = (B + 5) + 7) :
  A = 46 ∧ B = 29 :=
by
  sorry

end NUMINAMATH_GPT_initial_weights_of_apples_l1572_157282


namespace NUMINAMATH_GPT_find_second_number_l1572_157229

theorem find_second_number (a b c : ℕ) 
  (h1 : a + b + c = 550) 
  (h2 : a = 2 * b) 
  (h3 : c = a / 3) :
  b = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l1572_157229


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1572_157274

theorem isosceles_triangle_base_length (x : ℝ) (h1 : 2 * x + 2 * x + x = 20) : x = 4 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1572_157274


namespace NUMINAMATH_GPT_f_2002_eq_0_l1572_157272

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_2_eq_0 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem f_2002_eq_0 : f 2002 = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_2002_eq_0_l1572_157272


namespace NUMINAMATH_GPT_average_marks_passed_l1572_157279

noncomputable def total_candidates := 120
noncomputable def total_average_marks := 35
noncomputable def passed_candidates := 100
noncomputable def failed_candidates := total_candidates - passed_candidates
noncomputable def average_marks_failed := 15
noncomputable def total_marks := total_average_marks * total_candidates
noncomputable def total_marks_failed := average_marks_failed * failed_candidates

theorem average_marks_passed :
  ∃ P, P * passed_candidates + total_marks_failed = total_marks ∧ P = 39 := by
  sorry

end NUMINAMATH_GPT_average_marks_passed_l1572_157279


namespace NUMINAMATH_GPT_alice_bob_meet_after_six_turns_l1572_157212

/-
Alice and Bob play a game involving a circle whose circumference
is divided by 12 equally-spaced points. The points are numbered
clockwise, from 1 to 12. Both start on point 12. Alice moves clockwise
and Bob, counterclockwise. In a turn of the game, Alice moves 5 points 
clockwise and Bob moves 9 points counterclockwise. The game ends when they stop on
the same point. 
-/
theorem alice_bob_meet_after_six_turns (k : ℕ) :
  (5 * k) % 12 = (12 - (9 * k) % 12) % 12 -> k = 6 :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meet_after_six_turns_l1572_157212


namespace NUMINAMATH_GPT_george_initial_candy_l1572_157275

theorem george_initial_candy (number_of_bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : number_of_bags = 8) (h2 : pieces_per_bag = 81) : 
  number_of_bags * pieces_per_bag = 648 := 
by 
  sorry

end NUMINAMATH_GPT_george_initial_candy_l1572_157275


namespace NUMINAMATH_GPT_find_f2_of_conditions_l1572_157241

theorem find_f2_of_conditions (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
                              (h_g : ∀ x, g x = f x + 9) 
                              (h_g_val : g (-2) = 3) : 
                              f 2 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_f2_of_conditions_l1572_157241


namespace NUMINAMATH_GPT_remainder_x150_l1572_157284

theorem remainder_x150 (x : ℝ) : 
  ∃ r : ℝ, ∃ q : ℝ, x^150 = q * (x - 1)^3 + 11175*x^2 - 22200*x + 11026 := 
by
  sorry

end NUMINAMATH_GPT_remainder_x150_l1572_157284


namespace NUMINAMATH_GPT_volume_sphere_gt_cube_l1572_157234

theorem volume_sphere_gt_cube (a r : ℝ) (h : 6 * a^2 = 4 * π * r^2) : 
  (4 / 3) * π * r^3 > a^3 :=
by sorry

end NUMINAMATH_GPT_volume_sphere_gt_cube_l1572_157234


namespace NUMINAMATH_GPT_paint_faces_l1572_157252

def cuboid_faces : ℕ := 6
def number_of_cuboids : ℕ := 8 
def total_faces_painted : ℕ := cuboid_faces * number_of_cuboids

theorem paint_faces (h1 : cuboid_faces = 6) (h2 : number_of_cuboids = 8) : total_faces_painted = 48 := by
  -- conditions are defined above
  sorry

end NUMINAMATH_GPT_paint_faces_l1572_157252


namespace NUMINAMATH_GPT_max_type_a_workers_l1572_157218

theorem max_type_a_workers (x y : ℕ) (h1 : x + y = 150) (h2 : y ≥ 3 * x) : x ≤ 37 :=
sorry

end NUMINAMATH_GPT_max_type_a_workers_l1572_157218


namespace NUMINAMATH_GPT_positive_difference_between_two_numbers_l1572_157253

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_between_two_numbers_l1572_157253


namespace NUMINAMATH_GPT_find_angle_C_find_side_c_l1572_157209

noncomputable def triangle_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ) : Prop := 
a * Real.cos C = c * Real.sin A

theorem find_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ)
  (h1 : triangle_angle_C a b c C A)
  (h2 : 0 < A) : C = Real.pi / 3 := 
sorry

noncomputable def triangle_side_c (a b c : ℝ) (C : ℝ) : Prop := 
(∃ (area : ℝ), area = 6 ∧ b = 4 ∧ c * c = a * a + b * b - 2 * a * b * Real.cos C)

theorem find_side_c (a b c : ℝ) (C : ℝ) 
  (h1 : triangle_side_c a b c C) : c = 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_side_c_l1572_157209


namespace NUMINAMATH_GPT_ribbon_per_box_l1572_157261

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_per_box_l1572_157261


namespace NUMINAMATH_GPT_find_a_for_perfect_square_trinomial_l1572_157256

theorem find_a_for_perfect_square_trinomial (a : ℝ) :
  (∃ b : ℝ, x^2 - 8*x + a = (x - b)^2) ↔ a = 16 :=
by sorry

end NUMINAMATH_GPT_find_a_for_perfect_square_trinomial_l1572_157256


namespace NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l1572_157215

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l1572_157215


namespace NUMINAMATH_GPT_smallest_b_base_45b_perfect_square_l1572_157297

theorem smallest_b_base_45b_perfect_square : ∃ b : ℕ, b > 3 ∧ (∃ n : ℕ, n^2 = 4 * b + 5) ∧ ∀ b' : ℕ, b' > 3 ∧ (∃ n' : ℕ, n'^2 = 4 * b' + 5) → b ≤ b' := 
sorry

end NUMINAMATH_GPT_smallest_b_base_45b_perfect_square_l1572_157297


namespace NUMINAMATH_GPT_logan_television_hours_l1572_157220

-- Definitions
def minutes_in_an_hour : ℕ := 60
def logan_minutes_watched : ℕ := 300
def logan_hours_watched : ℕ := logan_minutes_watched / minutes_in_an_hour

-- Theorem statement
theorem logan_television_hours : logan_hours_watched = 5 := by
  sorry

end NUMINAMATH_GPT_logan_television_hours_l1572_157220


namespace NUMINAMATH_GPT_simplify_expression_l1572_157278

noncomputable def expression : ℝ :=
  (4 * (Real.sqrt 3 + Real.sqrt 7)) / (5 * Real.sqrt (3 + (1 / 2)))

theorem simplify_expression : expression = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1572_157278


namespace NUMINAMATH_GPT_total_trees_now_l1572_157264

-- Definitions from conditions
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_fallen_trees : ℕ := 5

-- Additional definitions capturing relations
def fell_narra_trees (x : ℕ) : Prop := x + (x + 1) = total_fallen_trees
def new_narra_trees_planted (x : ℕ) : ℕ := 2 * x
def new_mahogany_trees_planted (x : ℕ) : ℕ := 3 * (x + 1)

-- Final goal
theorem total_trees_now (x : ℕ) (h : fell_narra_trees x) :
  initial_mahogany_trees + initial_narra_trees
  - total_fallen_trees
  + new_narra_trees_planted x
  + new_mahogany_trees_planted x = 88 := by
  sorry

end NUMINAMATH_GPT_total_trees_now_l1572_157264


namespace NUMINAMATH_GPT_max_police_officers_needed_l1572_157222

theorem max_police_officers_needed : 
  let streets := 10
  let non_parallel := true
  let curved_streets := 2
  let additional_intersections_per_curved := 3 
  streets = 10 ∧ 
  non_parallel = true ∧ 
  curved_streets = 2 ∧ 
  additional_intersections_per_curved = 3 → 
  ( (streets * (streets - 1) / 2) + (curved_streets * additional_intersections_per_curved) ) = 51 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_police_officers_needed_l1572_157222


namespace NUMINAMATH_GPT_simplify_expression_l1572_157267

theorem simplify_expression :
  (∃ (a b c d : ℝ), 
   a = 14 * Real.sqrt 2 ∧ 
   b = 12 * Real.sqrt 2 ∧ 
   c = 8 * Real.sqrt 2 ∧ 
   d = 12 * Real.sqrt 2 ∧ 
   ((a / b) + (c / d) = 11 / 6)) :=
by 
  use 14 * Real.sqrt 2, 12 * Real.sqrt 2, 8 * Real.sqrt 2, 12 * Real.sqrt 2
  simp
  sorry

end NUMINAMATH_GPT_simplify_expression_l1572_157267


namespace NUMINAMATH_GPT_find_c_plus_1_over_b_l1572_157281

theorem find_c_plus_1_over_b (a b c : ℝ) (h1: a * b * c = 1) 
    (h2: a + 1 / c = 7) (h3: b + 1 / a = 12) : c + 1 / b = 21 / 83 := 
by 
    sorry

end NUMINAMATH_GPT_find_c_plus_1_over_b_l1572_157281


namespace NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l1572_157289

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l1572_157289
