import Mathlib

namespace probability_of_black_ball_l1532_153292

theorem probability_of_black_ball (P_red P_white : ℝ) (h_red : P_red = 0.43) (h_white : P_white = 0.27) : 
  (1 - P_red - P_white) = 0.3 := 
by
  sorry

end probability_of_black_ball_l1532_153292


namespace decorations_cost_l1532_153295

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l1532_153295


namespace min_value_of_f_l1532_153256

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f : ∀ (x : ℝ), x > 2 → f x ≥ 4 := by
  sorry

end min_value_of_f_l1532_153256


namespace square_inscribed_in_right_triangle_side_length_l1532_153207

theorem square_inscribed_in_right_triangle_side_length
  (A B C X Y Z W : ℝ × ℝ)
  (AB BC AC : ℝ)
  (square_side : ℝ)
  (h : 0 < square_side) :
  -- Define the lengths of sides of the triangle.
  AB = 3 ∧ BC = 4 ∧ AC = 5 ∧

  -- Define the square inscribed in the triangle
  (W.1 - A.1)^2 + (W.2 - A.2)^2 = square_side^2 ∧
  (X.1 - W.1)^2 + (X.2 - W.2)^2 = square_side^2 ∧
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = square_side^2 ∧
  (Z.1 - W.1)^2 + (Z.2 - W.2)^2 = square_side^2 ∧
  (Z.1 - C.1)^2 + (Z.2 - C.2)^2 = square_side^2 ∧

  -- Points where square meets triangle sides
  X.1 = A.1 ∧ Z.1 = C.1 ∧ Y.1 = X.1 ∧ W.1 = Z.1 ∧ Z.2 = Y.2 ∧

  -- Right triangle condition
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 ∧
  
  -- Right angle at vertex B
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  →
  -- Prove the side length of the inscribed square
  square_side = 60 / 37 :=
sorry

end square_inscribed_in_right_triangle_side_length_l1532_153207


namespace parallel_vectors_x_value_l1532_153282

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  vectors_parallel (1, -2) (x, 1) → x = -1 / 2 :=
by
  sorry

end parallel_vectors_x_value_l1532_153282


namespace distance_to_y_axis_eq_reflection_across_x_axis_eq_l1532_153287

-- Definitions based on the conditions provided
def point_P : ℝ × ℝ := (4, -2)

-- Statements we need to prove
theorem distance_to_y_axis_eq : (abs (point_P.1) = 4) := 
by
  sorry  -- Proof placeholder

theorem reflection_across_x_axis_eq : (point_P.1 = 4 ∧ -point_P.2 = 2) :=
by
  sorry  -- Proof placeholder

end distance_to_y_axis_eq_reflection_across_x_axis_eq_l1532_153287


namespace unique_solution_of_system_l1532_153293

theorem unique_solution_of_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1 ∧ x = 1 ∧ y = 1 ∧ z = 0 := by
  sorry

end unique_solution_of_system_l1532_153293


namespace minimum_p_plus_q_l1532_153276

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 4 * Real.log x + 1 else 2 * x - 1

theorem minimum_p_plus_q (p q : ℝ) (hpq : p ≠ q) (hf : f p + f q = 2) :
  p + q = 3 - 2 * Real.log 2 := by
  sorry

end minimum_p_plus_q_l1532_153276


namespace proposition_B_proposition_C_l1532_153264

variable (a b c d : ℝ)

-- Proposition B: If |a| > |b|, then a² > b²
theorem proposition_B (h : |a| > |b|) : a^2 > b^2 :=
sorry

-- Proposition C: If (a - b)c² > 0, then a > b
theorem proposition_C (h : (a - b) * c^2 > 0) : a > b :=
sorry

end proposition_B_proposition_C_l1532_153264


namespace geometric_sequence_log_sum_l1532_153210

noncomputable def log_base_three (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∃ r, ∀ n, a (n + 1) = a n * r)
  (h3 : a 6 * a 7 = 9) :
  log_base_three (a 1) + log_base_three (a 2) + log_base_three (a 3) +
  log_base_three (a 4) + log_base_three (a 5) + log_base_three (a 6) +
  log_base_three (a 7) + log_base_three (a 8) + log_base_three (a 9) +
  log_base_three (a 10) + log_base_three (a 11) + log_base_three (a 12) = 12 :=
  sorry

end geometric_sequence_log_sum_l1532_153210


namespace solve_equation_1_solve_equation_2_l1532_153243

theorem solve_equation_1 (x : Real) : 
  (1/3) * (x - 3)^2 = 12 → x = 9 ∨ x = -3 :=
by
  sorry

theorem solve_equation_2 (x : Real) : 
  (2 * x - 1)^2 = (1 - x)^2 → x = 0 ∨ x = 2/3 :=
by
  sorry

end solve_equation_1_solve_equation_2_l1532_153243


namespace perpendicular_lines_l1532_153242

theorem perpendicular_lines (m : ℝ) : 
  (m = -2 → (2-m) * (-(m+3)/(2-m)) + m * (m-3) / (-(m+3)) = 0) → 
  (m = -2 ∨ m = 1) := 
sorry

end perpendicular_lines_l1532_153242


namespace triangle_median_equiv_l1532_153251

-- Assuming necessary non-computable definitions (e.g., α for angles, R for real numbers) and non-computable nature of some geometric properties.

noncomputable def triangle (A B C : ℝ) := 
A + B + C = Real.pi

noncomputable def length_a (R A : ℝ) : ℝ := 2 * R * Real.sin A
noncomputable def length_b (R B : ℝ) : ℝ := 2 * R * Real.sin B
noncomputable def length_c (R C : ℝ) : ℝ := 2 * R * Real.sin C

noncomputable def median_a (b c A : ℝ) : ℝ := (2 * b * c) / (b + c) * Real.cos (A / 2)

theorem triangle_median_equiv (A B C R : ℝ) (hA : triangle A B C) :
  (1 / (length_a R A) + 1 / (length_b R B) = 1 / (median_a (length_b R B) (length_c R C) A)) ↔ (C = 2 * Real.pi / 3) := 
by sorry

end triangle_median_equiv_l1532_153251


namespace poly_div_factor_l1532_153213

theorem poly_div_factor (c : ℚ) : 2 * x + 7 ∣ 8 * x^4 + 27 * x^3 + 6 * x^2 + c * x - 49 ↔
  c = 47.25 :=
  sorry

end poly_div_factor_l1532_153213


namespace percentage_increase_area_l1532_153218

theorem percentage_increase_area (L W : ℝ) (hL : 0 < L) (hW : 0 < W) :
  let A := L * W
  let A' := (1.35 * L) * (1.35 * W)
  let percentage_increase := ((A' - A) / A) * 100
  percentage_increase = 82.25 :=
by
  sorry

end percentage_increase_area_l1532_153218


namespace gcd_of_105_1001_2436_l1532_153267

noncomputable def gcd_problem : ℕ :=
  Nat.gcd (Nat.gcd 105 1001) 2436

theorem gcd_of_105_1001_2436 : gcd_problem = 7 :=
by {
  sorry
}

end gcd_of_105_1001_2436_l1532_153267


namespace integer_coordinates_midpoint_exists_l1532_153281

theorem integer_coordinates_midpoint_exists (P : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧
    ∃ x y : ℤ, (2 * x = (P i).1 + (P j).1) ∧ (2 * y = (P i).2 + (P j).2) := sorry

end integer_coordinates_midpoint_exists_l1532_153281


namespace correct_operation_l1532_153273

theorem correct_operation : 
  (3 - Real.sqrt 2) ^ 2 = 11 - 6 * Real.sqrt 2 :=
sorry

end correct_operation_l1532_153273


namespace proof_problem_l1532_153240

noncomputable def f (a b : ℝ) (x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) : 
  (f a b (11 * π / 12) = 0) ∧
  (|f a b (7 * π / 12)| < |f a b (π / 5)|) ∧
  (¬ (∀ x : ℝ, f a b x = f a b (-x)) ∧ ¬ (∀ x : ℝ, f a b x = -f a b (-x))) := 
sorry

end proof_problem_l1532_153240


namespace number_of_valid_paths_l1532_153249

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_valid_paths (n : ℕ) :
  let valid_paths := binomial (2 * n) n / (n + 1)
  valid_paths = binomial (2 * n) n - binomial (2 * n) (n + 1) := 
sorry

end number_of_valid_paths_l1532_153249


namespace intersection_of_A_and_B_l1532_153227

def set_A : Set ℝ := {x | x^2 ≤ 4 * x}
def set_B : Set ℝ := {x | |x| ≥ 2}

theorem intersection_of_A_and_B :
  {x | x ∈ set_A ∧ x ∈ set_B} = {x | 2 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l1532_153227


namespace linear_equation_conditions_l1532_153261

theorem linear_equation_conditions (m n : ℤ) :
  (∀ x y : ℝ, 4 * x^(m - n) - 5 * y^(m + n) = 6 → 
    m - n = 1 ∧ m + n = 1) →
  m = 1 ∧ n = 0 :=
by
  sorry

end linear_equation_conditions_l1532_153261


namespace find_chemistry_marks_l1532_153269

theorem find_chemistry_marks 
    (marks_english : ℕ := 70)
    (marks_math : ℕ := 63)
    (marks_physics : ℕ := 80)
    (marks_biology : ℕ := 65)
    (average_marks : ℚ := 68.2) :
    ∃ (marks_chemistry : ℕ), 
      (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) = 5 * average_marks 
      → marks_chemistry = 63 :=
by
  sorry

end find_chemistry_marks_l1532_153269


namespace probability_of_three_specific_suits_l1532_153246

noncomputable def probability_at_least_one_from_each_of_three_suits : ℚ :=
  1 - (1 / 4) ^ 5

theorem probability_of_three_specific_suits (hearts clubs diamonds : ℕ) :
  hearts = 0 ∧ clubs = 0 ∧ diamonds = 0 → 
  probability_at_least_one_from_each_of_three_suits = 1023 / 1024 := 
by 
  sorry

end probability_of_three_specific_suits_l1532_153246


namespace B_take_time_4_hours_l1532_153268

theorem B_take_time_4_hours (A_rate B_rate C_rate D_rate : ℚ) :
  (A_rate = 1 / 4) →
  (B_rate + C_rate = 1 / 2) →
  (A_rate + C_rate = 1 / 2) →
  (D_rate = 1 / 8) →
  (A_rate + B_rate + D_rate = 1 / 1.6) →
  (B_rate = 1 / 4) ∧ (1 / B_rate = 4) :=
by
  sorry

end B_take_time_4_hours_l1532_153268


namespace dennis_took_away_l1532_153222

-- Define the initial and remaining number of cards
def initial_cards : ℕ := 67
def remaining_cards : ℕ := 58

-- Define the number of cards taken away
def cards_taken_away (n m : ℕ) : ℕ := n - m

-- Prove that the number of cards taken away is 9
theorem dennis_took_away :
  cards_taken_away initial_cards remaining_cards = 9 :=
by
  -- Placeholder proof
  sorry

end dennis_took_away_l1532_153222


namespace fish_per_bowl_l1532_153278

theorem fish_per_bowl (num_bowls num_fish : ℕ) (h1 : num_bowls = 261) (h2 : num_fish = 6003) :
  num_fish / num_bowls = 23 :=
by {
  sorry
}

end fish_per_bowl_l1532_153278


namespace simple_interest_rate_l1532_153290

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) :
  (T = 20) →
  (SI = P) →
  (SI = P * R * T / 100) →
  R = 5 :=
by
  sorry

end simple_interest_rate_l1532_153290


namespace initial_pokemon_cards_l1532_153272

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end initial_pokemon_cards_l1532_153272


namespace mr_bird_speed_to_be_on_time_l1532_153299

theorem mr_bird_speed_to_be_on_time 
  (d : ℝ) 
  (t : ℝ)
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  (d / t) = 48 :=
by
  sorry

end mr_bird_speed_to_be_on_time_l1532_153299


namespace Prudence_sleep_weeks_l1532_153229

def Prudence_sleep_per_week : Nat := 
  let nights_sleep_weekday := 6
  let nights_sleep_weekend := 9
  let weekday_nights := 5
  let weekend_nights := 2
  let naps := 1
  let naps_days := 2
  weekday_nights * nights_sleep_weekday + weekend_nights * nights_sleep_weekend + naps_days * naps

theorem Prudence_sleep_weeks (w : Nat) (h : w * Prudence_sleep_per_week = 200) : w = 4 :=
by
  sorry

end Prudence_sleep_weeks_l1532_153229


namespace complex_power_sum_eq_self_l1532_153275

theorem complex_power_sum_eq_self (z : ℂ) (h : z^2 + z + 1 = 0) : z^100 + z^101 + z^102 + z^103 = z :=
sorry

end complex_power_sum_eq_self_l1532_153275


namespace total_sessions_l1532_153250

theorem total_sessions (p1 p2 p3 p4 : ℕ) 
(h1 : p1 = 6) 
(h2 : p2 = p1 + 5) 
(h3 : p3 = 8) 
(h4 : p4 = 8) : 
p1 + p2 + p3 + p4 = 33 := 
by
  sorry

end total_sessions_l1532_153250


namespace solve_for_b_l1532_153270

theorem solve_for_b (b : ℝ) : 
  let slope1 := -(3 / 4 : ℝ)
  let slope2 := -(b / 6 : ℝ)
  slope1 * slope2 = -1 → b = -8 :=
by
  intro h
  sorry

end solve_for_b_l1532_153270


namespace problem_I_problem_II_l1532_153283

-- Problem (I)
theorem problem_I (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) : 
  ∀ x, (f x < |x| + 1) → (0 < x ∧ x < 2) :=
by
  intro x hx
  have fx_def : f x = |2 * x - 1| := h x
  sorry

-- Problem (II)
theorem problem_II (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) :
  ∀ x y, (|x - y - 1| ≤ 1 / 3) → (|2 * y + 1| ≤ 1 / 6) → (f x ≤ 5 / 6) :=
by
  intro x y hx hy
  have fx_def : f x = |2 * x - 1| := h x
  sorry

end problem_I_problem_II_l1532_153283


namespace Razorback_shop_total_revenue_l1532_153237

theorem Razorback_shop_total_revenue :
  let Tshirt_price := 62
  let Jersey_price := 99
  let Hat_price := 45
  let Keychain_price := 25
  let Tshirt_sold := 183
  let Jersey_sold := 31
  let Hat_sold := 142
  let Keychain_sold := 215
  let revenue := (Tshirt_price * Tshirt_sold) + (Jersey_price * Jersey_sold) + (Hat_price * Hat_sold) + (Keychain_price * Keychain_sold)
  revenue = 26180 :=
by
  sorry

end Razorback_shop_total_revenue_l1532_153237


namespace imaginary_part_of_product_l1532_153228

def imaginary_unit : ℂ := Complex.I

def z : ℂ := 2 + imaginary_unit

theorem imaginary_part_of_product : (z * imaginary_unit).im = 2 := by
  sorry

end imaginary_part_of_product_l1532_153228


namespace num_two_digit_numbers_with_digit_sum_10_l1532_153265

theorem num_two_digit_numbers_with_digit_sum_10 : 
  ∃ n, n = 9 ∧ ∀ a b, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 10 → ∃ m, 10 * a + b = m :=
sorry

end num_two_digit_numbers_with_digit_sum_10_l1532_153265


namespace area_of_region_AGF_l1532_153203

theorem area_of_region_AGF 
  (ABCD_area : ℝ)
  (hABCD_area : ABCD_area = 160)
  (E F G : ℝ)
  (hE_midpoint : E = (A + B) / 2)
  (hF_midpoint : F = (C + D) / 2)
  (EF_divides : EF_area = ABCD_area / 2)
  (hEF_midpoint : G = (E + F) / 2)
  (AG_divides_upper : AG_area = EF_area / 2) :
  AGF_area = 40 := 
sorry

end area_of_region_AGF_l1532_153203


namespace find_vector_at_t_zero_l1532_153201

def vector_at_t (a d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (a.1 + t*d.1, a.2 + t*d.2)

theorem find_vector_at_t_zero :
  ∃ (a d : ℝ × ℝ),
    vector_at_t a d 1 = (2, 3) ∧
    vector_at_t a d 4 = (8, -5) ∧
    vector_at_t a d 5 = (10, -9) ∧
    vector_at_t a d 0 = (0, 17/3) :=
by
  sorry

end find_vector_at_t_zero_l1532_153201


namespace coefficient_x99_is_zero_l1532_153238

open Polynomial

noncomputable def P (x : ℤ) : Polynomial ℤ := sorry
noncomputable def Q (x : ℤ) : Polynomial ℤ := sorry

theorem coefficient_x99_is_zero : 
    (P 0 = 1) → 
    ((P x)^2 = 1 + x + x^100 * Q x) → 
    (Polynomial.coeff ((P x + 1)^100) 99 = 0) :=
by
    -- Proof omitted
    sorry

end coefficient_x99_is_zero_l1532_153238


namespace red_paint_quarts_l1532_153232

theorem red_paint_quarts (r g w : ℕ) (ratio_rw : r * 5 = w * 4) (w_quarts : w = 15) : r = 12 :=
by 
  -- We provide the skeleton of the proof here: the detailed steps are skipped (as instructed).
  sorry

end red_paint_quarts_l1532_153232


namespace simplify_expression_l1532_153215

theorem simplify_expression (a b : ℂ) (x : ℂ) (hb : b ≠ 0) (ha : a ≠ b) (hx : x = a / b) :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  -- Proof goes here
  sorry

end simplify_expression_l1532_153215


namespace monotonic_decreasing_interval_l1532_153248

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, x < 2 → f' x < 0 :=
by
  intro x hx
  sorry

end monotonic_decreasing_interval_l1532_153248


namespace tile_count_l1532_153214

theorem tile_count (a : ℕ) (h1 : ∃ b : ℕ, b = 2 * a) (h2 : 2 * (Int.floor (a * Real.sqrt 5)) - 1 = 49) :
  2 * a^2 = 50 :=
by
  sorry

end tile_count_l1532_153214


namespace second_term_of_arithmetic_sequence_l1532_153274

-- Define the statement of the problem
theorem second_term_of_arithmetic_sequence 
  (a d : ℝ) 
  (h : a + (a + 2 * d) = 10) : 
  a + d = 5 := 
by 
  sorry

end second_term_of_arithmetic_sequence_l1532_153274


namespace find_n_l1532_153221

theorem find_n (n : ℕ) (h : n > 0) : 
  (3^n + 5^n) % (3^(n-1) + 5^(n-1)) = 0 ↔ n = 1 := 
by sorry

end find_n_l1532_153221


namespace number_of_individuals_left_at_zoo_l1532_153288

theorem number_of_individuals_left_at_zoo 
  (students_class1 students_class2 students_left : ℕ)
  (initial_chaperones remaining_chaperones teachers : ℕ) :
  students_class1 = 10 ∧
  students_class2 = 10 ∧
  initial_chaperones = 5 ∧
  teachers = 2 ∧
  students_left = 10 ∧
  remaining_chaperones = initial_chaperones - 2 →
  (students_class1 + students_class2 - students_left) + remaining_chaperones + teachers = 15 :=
by
  sorry

end number_of_individuals_left_at_zoo_l1532_153288


namespace determine_H_zero_l1532_153285

theorem determine_H_zero (E F G H : ℕ) 
  (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (add_eq : 10 * E + F + 10 * G + E = 10 * H + E)
  (sub_eq : 10 * E + F - (10 * G + E) = E) : 
  H = 0 :=
sorry

end determine_H_zero_l1532_153285


namespace christmas_trees_in_each_box_l1532_153245

theorem christmas_trees_in_each_box
  (T : ℕ)
  (pieces_of_tinsel_in_each_box : ℕ := 4)
  (snow_globes_in_each_box : ℕ := 5)
  (total_boxes : ℕ := 12)
  (total_decorations : ℕ := 120)
  (decorations_per_box : ℕ := pieces_of_tinsel_in_each_box + T + snow_globes_in_each_box)
  (total_decorations_distributed : ℕ := total_boxes * decorations_per_box) :
  total_decorations_distributed = total_decorations → T = 1 := by
  sorry

end christmas_trees_in_each_box_l1532_153245


namespace sum_of_A_and_B_l1532_153286

theorem sum_of_A_and_B (A B : ℕ) (h1 : (1 / 6 : ℚ) * (1 / 3) = 1 / (A * 3))
                       (h2 : (1 / 6 : ℚ) * (1 / 3) = 1 / B) : A + B = 24 :=
by
  sorry

end sum_of_A_and_B_l1532_153286


namespace hypotenuse_length_right_triangle_l1532_153230

theorem hypotenuse_length_right_triangle :
  ∃ (x : ℝ), (x > 7) ∧ ((x - 7)^2 + x^2 = (x + 2)^2) ∧ (x + 2 = 17) :=
by {
  sorry
}

end hypotenuse_length_right_triangle_l1532_153230


namespace inversely_proportional_l1532_153262

theorem inversely_proportional (X Y K : ℝ) (h : X * Y = K - 1) (hK : K > 1) : 
  (∃ c : ℝ, ∀ x y : ℝ, x * y = c) :=
sorry

end inversely_proportional_l1532_153262


namespace area_of_region_l1532_153297

theorem area_of_region (x y : ℝ) : (x^2 + y^2 + 6 * x - 8 * y = 1) → (π * 26) = 26 * π :=
by
  intro h
  sorry

end area_of_region_l1532_153297


namespace sum_difference_even_odd_l1532_153211

-- Define the sum of even integers from 2 to 100
def sum_even (n : ℕ) : ℕ := (n / 2) * (2 + n)

-- Define the sum of odd integers from 1 to 99
def sum_odd (n : ℕ) : ℕ := (n / 2) * (1 + n)

theorem sum_difference_even_odd:
  let a := sum_even 100
  let b := sum_odd 99
  a - b = 50 :=
by
  sorry

end sum_difference_even_odd_l1532_153211


namespace problem1_problem2_l1532_153291

-- Equivalent proof statement for part (1)
theorem problem1 : 2023^2 - 2022 * 2024 = 1 := by
  sorry

-- Equivalent proof statement for part (2)
theorem problem2 (m : ℝ) (h : m ≠ 1) (h1 : m ≠ -1) : 
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by
  sorry

end problem1_problem2_l1532_153291


namespace max_handshakes_l1532_153284

theorem max_handshakes (n m : ℕ) (cond1 : n = 30) (cond2 : m = 5) 
                       (cond3 : ∀ (i : ℕ), i < 30 → ∀ (j : ℕ), j < 30 → i ≠ j → true)
                       (cond4 : ∀ (k : ℕ), k < 5 → ∃ (s : ℕ), s ≤ 10) : 
  ∃ (handshakes : ℕ), handshakes = 325 :=
by
  sorry

end max_handshakes_l1532_153284


namespace Jamie_owns_2_Maine_Coons_l1532_153294

-- Definitions based on conditions
variables (Jamie_MaineCoons Gordon_MaineCoons Hawkeye_MaineCoons Jamie_Persians Gordon_Persians Hawkeye_Persians : ℕ)

-- Conditions
axiom Jamie_owns_4_Persians : Jamie_Persians = 4
axiom Gordon_owns_half_as_many_Persians_as_Jamie : Gordon_Persians = Jamie_Persians / 2
axiom Gordon_owns_one_more_Maine_Coon_than_Jamie : Gordon_MaineCoons = Jamie_MaineCoons + 1
axiom Hawkeye_owns_one_less_Maine_Coon_than_Gordon : Hawkeye_MaineCoons = Gordon_MaineCoons - 1
axiom Hawkeye_owns_no_Persian_cats : Hawkeye_Persians = 0
axiom total_number_of_cats_is_13 : Jamie_Persians + Jamie_MaineCoons + Gordon_Persians + Gordon_MaineCoons + Hawkeye_Persians + Hawkeye_MaineCoons = 13

-- Theorem statement
theorem Jamie_owns_2_Maine_Coons : Jamie_MaineCoons = 2 :=
by {
  -- Ideally, you would provide the proof here, stepping through algebraically as shown in the solution,
  -- but we are skipping the proof as specified in the instructions.
  sorry
}

end Jamie_owns_2_Maine_Coons_l1532_153294


namespace find_num_adults_l1532_153258

-- Define the conditions
def total_eggs : ℕ := 36
def eggs_per_adult : ℕ := 3
def eggs_per_girl : ℕ := 1
def eggs_per_boy := eggs_per_girl + 1
def num_girls : ℕ := 7
def num_boys : ℕ := 10

-- Compute total eggs given to girls
def eggs_given_to_girls : ℕ := num_girls * eggs_per_girl

-- Compute total eggs given to boys
def eggs_given_to_boys : ℕ := num_boys * eggs_per_boy

-- Compute total eggs given to children
def eggs_given_to_children : ℕ := eggs_given_to_girls + eggs_given_to_boys

-- Total number of eggs given to children
def eggs_left_for_adults : ℕ := total_eggs - eggs_given_to_children

-- Calculate the number of adults
def num_adults : ℕ := eggs_left_for_adults / eggs_per_adult

-- Finally, we want to prove that the number of adults is 3
theorem find_num_adults (h1 : total_eggs = 36) 
                        (h2 : eggs_per_adult = 3) 
                        (h3 : eggs_per_girl = 1)
                        (h4 : num_girls = 7) 
                        (h5 : num_boys = 10) : 
                        num_adults = 3 := by
  -- Using the given conditions and computations
  sorry

end find_num_adults_l1532_153258


namespace sum_geometric_series_l1532_153225

theorem sum_geometric_series :
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 195312 / 781250 := by
    sorry

end sum_geometric_series_l1532_153225


namespace triangle_ineq_l1532_153200

theorem triangle_ineq
  (a b c : ℝ)
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_ineq : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  sorry

end triangle_ineq_l1532_153200


namespace symmetric_points_product_l1532_153212

theorem symmetric_points_product (a b : ℝ) 
    (h1 : a + 2 = -4) 
    (h2 : b = 2) : 
    a * b = -12 := 
sorry

end symmetric_points_product_l1532_153212


namespace max_f_eq_4_monotonic_increase_interval_l1532_153263

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_f_eq_4 (x : ℝ) : ∀ x : ℝ, f x ≤ 4 := 
by
  sorry

theorem monotonic_increase_interval (k : ℤ) : ∀ x : ℝ, (k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4) ↔ 
  (0 ≤ Real.sin (2 * x) ∧ Real.sin (2 * x) ≤ 1) :=
by
  sorry

end max_f_eq_4_monotonic_increase_interval_l1532_153263


namespace SallyMcQueenCostCorrect_l1532_153259

def LightningMcQueenCost : ℕ := 140000
def MaterCost : ℕ := (140000 * 10) / 100
def SallyMcQueenCost : ℕ := 3 * MaterCost

theorem SallyMcQueenCostCorrect : SallyMcQueenCost = 42000 := by
  sorry

end SallyMcQueenCostCorrect_l1532_153259


namespace odd_divisibility_l1532_153271

theorem odd_divisibility (n : ℕ) (k : ℕ) (x y : ℤ) (h : n = 2 * k + 1) : (x^n + y^n) % (x + y) = 0 :=
by sorry

end odd_divisibility_l1532_153271


namespace diagonal_length_l1532_153277

noncomputable def length_of_diagonal (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_length
  (a b c : ℝ)
  (h1 : 2 * (a * b + a * c + b * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  length_of_diagonal a b c = 5 := by
  sorry

end diagonal_length_l1532_153277


namespace incorrect_conclusion_symmetry_l1532_153298

/-- Given the function f(x) = sin(1/5 * x + 13/6 * π), we define another function g(x) as the
translated function of f rightward by 10/3 * π units. We need to show that the graph of g(x)
is not symmetrical about the line x = π/4. -/
theorem incorrect_conclusion_symmetry (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = Real.sin (1/5 * x + 13/6 * Real.pi))
  (h₂ : ∀ x, g x = f (x - 10/3 * Real.pi)) :
  ¬ (∀ x, g (2 * (Real.pi / 4) - x) = g x) :=
sorry

end incorrect_conclusion_symmetry_l1532_153298


namespace tetrahedron_faces_equal_l1532_153255

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end tetrahedron_faces_equal_l1532_153255


namespace determine_list_price_l1532_153289

theorem determine_list_price (x : ℝ) :
  0.12 * (x - 15) = 0.15 * (x - 25) → x = 65 :=
by 
  sorry

end determine_list_price_l1532_153289


namespace undefined_expression_real_val_l1532_153226

theorem undefined_expression_real_val (a : ℝ) :
  a = 2 → (a^3 - 8 = 0) :=
by
  intros
  sorry

end undefined_expression_real_val_l1532_153226


namespace left_handed_rock_lovers_l1532_153209

theorem left_handed_rock_lovers (total_people left_handed rock_music right_dislike_rock x : ℕ) :
  total_people = 30 →
  left_handed = 14 →
  rock_music = 20 →
  right_dislike_rock = 5 →
  (x + (left_handed - x) + (rock_music - x) + right_dislike_rock = total_people) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end left_handed_rock_lovers_l1532_153209


namespace calculate_expression_l1532_153208

theorem calculate_expression :
  2 * (-1 / 4) - |1 - Real.sqrt 3| + (-2023)^0 = 3 / 2 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l1532_153208


namespace solve_system_of_equations_l1532_153224

theorem solve_system_of_equations (x y m : ℝ) 
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = m - 3) 
  (h3 : x - y = 2) : m = 8 :=
by
  -- Proof part is replaced with sorry as mentioned
  sorry

end solve_system_of_equations_l1532_153224


namespace hardest_work_diff_l1532_153253

theorem hardest_work_diff 
  (A B C D : ℕ) 
  (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x ∧ D = 4 * x)
  (h_total : A + B + C + D = 240) :
  (D - A) = 72 :=
by
  sorry

end hardest_work_diff_l1532_153253


namespace find_values_of_a_to_make_lines_skew_l1532_153236

noncomputable def lines_are_skew (t u a : ℝ) : Prop :=
  ∀ t u,
    (1 + 2 * t = 4 + 5 * u ∧
     2 + 3 * t = 1 + 2 * u ∧
     a + 4 * t = u) → false

theorem find_values_of_a_to_make_lines_skew :
  ∀ a : ℝ, ¬ a = 3 ↔ lines_are_skew t u a :=
by
  sorry

end find_values_of_a_to_make_lines_skew_l1532_153236


namespace range_of_a_l1532_153205

variable (a : ℝ)

def p : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ (x : ℝ), x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ∈ Set.Iic (-2) ∪ {1} := by
  sorry

end range_of_a_l1532_153205


namespace gcd_2210_145_l1532_153260

-- defining the constants a and b
def a : ℕ := 2210
def b : ℕ := 145

-- theorem stating that gcd(a, b) = 5
theorem gcd_2210_145 : Nat.gcd a b = 5 :=
sorry

end gcd_2210_145_l1532_153260


namespace chord_length_l1532_153233

-- Definitions and conditions for the problem
variables (A D B C G E F : Point)

-- Lengths and radii in the problem
noncomputable def radius : Real := 10
noncomputable def AB : Real := 20
noncomputable def BC : Real := 20
noncomputable def CD : Real := 20

-- Centers of circles
variables (O N P : Circle) (AN ND : Real)

-- Tangent properties and intersection points
variable (tangent_AG : Tangent AG P G)
variable (intersect_AG_N : Intersects AG N E F)

-- Given the geometry setup, prove the length of chord EF.
theorem chord_length (EF_length : Real) :
  EF_length = 2 * Real.sqrt 93.75 := sorry

end chord_length_l1532_153233


namespace smallest_pos_int_b_for_factorization_l1532_153254

theorem smallest_pos_int_b_for_factorization :
  ∃ b : ℤ, 0 < b ∧ ∀ (x : ℤ), ∃ r s : ℤ, r * s = 4032 ∧ r + s = b ∧ x^2 + b * x + 4032 = (x + r) * (x + s) ∧
    (∀ b' : ℤ, 0 < b' → b' ≠ b → ∃ rr ss : ℤ, rr * ss = 4032 ∧ rr + ss = b' ∧ x^2 + b' * x + 4032 = (x + rr) * (x + ss) → b < b') := 
sorry

end smallest_pos_int_b_for_factorization_l1532_153254


namespace binomial_expansion_problem_l1532_153257

noncomputable def binomial_expansion_sum_coefficients (n : ℕ) : ℤ :=
  (1 - 3) ^ n

def general_term_coefficient (n r : ℕ) : ℤ :=
  (-3) ^ r * (Nat.choose n r)

theorem binomial_expansion_problem :
  ∃ (n : ℕ), binomial_expansion_sum_coefficients n = 64 ∧ general_term_coefficient 6 2 = 135 :=
by
  sorry

end binomial_expansion_problem_l1532_153257


namespace crown_distribution_l1532_153234

theorem crown_distribution 
  (A B C D E : ℤ) 
  (h1 : 2 * C = 3 * A)
  (h2 : 4 * D = 3 * B)
  (h3 : 4 * E = 5 * C)
  (h4 : 5 * D = 6 * A)
  (h5 : A + B + C + D + E = 2870) : 
  A = 400 ∧ B = 640 ∧ C = 600 ∧ D = 480 ∧ E = 750 := 
by 
  sorry

end crown_distribution_l1532_153234


namespace total_money_l1532_153239

-- Define the problem with conditions and question transformed into proof statement
theorem total_money (A B : ℕ) (h1 : 2 * A / 3 = B / 2) (h2 : B = 484) : A + B = 847 :=
by
  sorry -- Proof to be filled in

end total_money_l1532_153239


namespace no_real_x_condition_l1532_153266

theorem no_real_x_condition (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 3| + |x - 1| ≤ a) ↔ a < 2 := 
by
  sorry

end no_real_x_condition_l1532_153266


namespace batsman_average_after_12th_innings_l1532_153296

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (h1 : 75 = (A + 12)) 
  (h2 : 11 * A + 75 = 12 * (A + 1)) :
  (A + 1) = 64 :=
by 
  sorry

end batsman_average_after_12th_innings_l1532_153296


namespace total_fencing_cost_l1532_153220

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end total_fencing_cost_l1532_153220


namespace total_cost_of_breakfast_l1532_153244

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l1532_153244


namespace find_AB_l1532_153206

variables {AB CD AD BC AP PD APD PQ Q: ℝ}

def is_rectangle (ABCD : Prop) := ABCD

variables (P_on_BC : Prop)
variable (BP CP: ℝ)
variable (tan_angle_APD: ℝ)

theorem find_AB (ABCD : Prop) (P_on_BC : Prop) (BP CP: ℝ) (tan_angle_APD: ℝ) : 
  is_rectangle ABCD →
  P_on_BC →
  BP = 24 →
  CP = 12 →
  tan_angle_APD = 2 →
  AB = 27 := 
by
  sorry

end find_AB_l1532_153206


namespace f_plus_one_odd_l1532_153217

noncomputable def f : ℝ → ℝ := sorry

theorem f_plus_one_odd (f : ℝ → ℝ)
  (h : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1) :
  ∀ x : ℝ, f x + 1 = -(f (-x) + 1) :=
sorry

end f_plus_one_odd_l1532_153217


namespace difference_blue_yellow_l1532_153247

def total_pebbles : ℕ := 40
def red_pebbles : ℕ := 9
def blue_pebbles : ℕ := 13
def remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
def groups : ℕ := 3
def pebbles_per_group : ℕ := remaining_pebbles / groups
def yellow_pebbles : ℕ := pebbles_per_group

theorem difference_blue_yellow : blue_pebbles - yellow_pebbles = 7 :=
by
  unfold blue_pebbles yellow_pebbles pebbles_per_group remaining_pebbles total_pebbles red_pebbles
  sorry

end difference_blue_yellow_l1532_153247


namespace solve_quadratic_identity_l1532_153235

theorem solve_quadratic_identity (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) :
  (14 * y - 5) ^ 2 = 333 :=
by sorry

end solve_quadratic_identity_l1532_153235


namespace incorrect_conclusion_l1532_153219

def y (x : ℝ) : ℝ := -2 * x + 3

theorem incorrect_conclusion : ∀ (x : ℝ), y x = 0 → x ≠ 0 := 
by
  sorry

end incorrect_conclusion_l1532_153219


namespace phantom_needs_more_money_l1532_153231

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end phantom_needs_more_money_l1532_153231


namespace bottles_per_case_l1532_153202

theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) (h1 : total_bottles = 60000) (h2 : total_cases = 12000) :
  total_bottles / total_cases = 5 :=
by
  -- Using the given problem, so steps from the solution are not required here
  sorry

end bottles_per_case_l1532_153202


namespace largest_whole_number_l1532_153252

theorem largest_whole_number (x : ℕ) (h1 : 9 * x < 150) : x ≤ 16 :=
by sorry

end largest_whole_number_l1532_153252


namespace trigonometric_expression_l1532_153279

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 3) : 
  ((Real.cos (α - π / 2) + Real.cos (α + π)) / (2 * Real.sin α) = 1 / 3) :=
by
  sorry

end trigonometric_expression_l1532_153279


namespace system_has_integer_solution_l1532_153204

theorem system_has_integer_solution (a b : ℤ) : 
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_has_integer_solution_l1532_153204


namespace sum_of_fourth_powers_correct_l1532_153216

noncomputable def sum_of_fourth_powers (x : ℤ) : ℤ :=
  x^4 + (x+1)^4 + (x+2)^4

theorem sum_of_fourth_powers_correct (x : ℤ) (h : x * (x+1) * (x+2) = 36 * x + 12) : 
  sum_of_fourth_powers x = 98 :=
sorry

end sum_of_fourth_powers_correct_l1532_153216


namespace value_of_a_l1532_153241

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem value_of_a (a : ℝ) (h : abs ((a^2) - a) = a / 2) : a = 1 / 2 ∨ a = 3 / 2 := by
  sorry

end value_of_a_l1532_153241


namespace complementary_angle_difference_l1532_153280

theorem complementary_angle_difference (a b : ℝ) (h1 : a = 4 * b) (h2 : a + b = 90) : (a - b) = 54 :=
by
  -- Proof is intentionally omitted
  sorry

end complementary_angle_difference_l1532_153280


namespace minimum_value_of_g_gm_equal_10_implies_m_is_5_l1532_153223

/-- Condition: Definition of the function y in terms of x and m -/
def y (x m : ℝ) : ℝ := x^2 + m * x - 4

/-- Theorem about finding the minimum value of g(m) -/
theorem minimum_value_of_g (m : ℝ) :
  ∃ g : ℝ, g = (if m ≥ -4 then 2 * m
      else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
      else 4 * m + 12) := by
  sorry

/-- Theorem that if the minimum value of g(m) is 10, then m must be 5 -/
theorem gm_equal_10_implies_m_is_5 :
  ∃ m, (if m ≥ -4 then 2 * m
       else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
       else 4 * m + 12) = 10 := by
  use 5
  sorry

end minimum_value_of_g_gm_equal_10_implies_m_is_5_l1532_153223
