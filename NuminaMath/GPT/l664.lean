import Mathlib

namespace NUMINAMATH_GPT_half_lake_covered_day_l664_66455

theorem half_lake_covered_day
  (N : ℕ) -- the total number of flowers needed to cover the entire lake
  (flowers_on_day : ℕ → ℕ) -- a function that gives the number of flowers on a specific day
  (h1 : flowers_on_day 20 = N) -- on the 20th day, the number of flowers is N
  (h2 : ∀ d, flowers_on_day (d + 1) = 2 * flowers_on_day d) -- the number of flowers doubles each day
  : flowers_on_day 19 = N / 2 :=
by
  sorry

end NUMINAMATH_GPT_half_lake_covered_day_l664_66455


namespace NUMINAMATH_GPT_unique_solutions_xy_l664_66457

theorem unique_solutions_xy (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end NUMINAMATH_GPT_unique_solutions_xy_l664_66457


namespace NUMINAMATH_GPT_graph_not_pass_through_second_quadrant_l664_66415

theorem graph_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = 2 * x - 3 ∧ x < 0 ∧ y > 0 :=
by sorry

end NUMINAMATH_GPT_graph_not_pass_through_second_quadrant_l664_66415


namespace NUMINAMATH_GPT_find_r_of_tangential_cones_l664_66461

theorem find_r_of_tangential_cones (r : ℝ) : 
  (∃ (r1 r2 r3 R : ℝ), r1 = 2 * r ∧ r2 = 3 * r ∧ r3 = 10 * r ∧ R = 15 ∧
  -- Additional conditions to ensure the three cones touch and share a slant height
  -- with the truncated cone of radius R
  true) → r = 29 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_r_of_tangential_cones_l664_66461


namespace NUMINAMATH_GPT_original_length_wire_l664_66473

-- Define the conditions.
def length_cut_off_parts : ℕ := 10
def remaining_length_relation (L_remaining : ℕ) : Prop :=
  L_remaining = 4 * (2 * length_cut_off_parts) + 10

-- Define the theorem to prove the original length of the wire.
theorem original_length_wire (L_remaining : ℕ) (H : remaining_length_relation L_remaining) : 
  L_remaining + 2 * length_cut_off_parts = 110 :=
by 
  -- Use the given conditions
  unfold remaining_length_relation at H
  -- The proof would show that the equation holds true.
  sorry

end NUMINAMATH_GPT_original_length_wire_l664_66473


namespace NUMINAMATH_GPT_time_to_finish_furniture_l664_66421

-- Define the problem's conditions
def chairs : ℕ := 7
def tables : ℕ := 3
def minutes_per_piece : ℕ := 4

-- Define total furniture
def total_furniture : ℕ := chairs + tables

-- Define the function to calculate total time
def total_time (pieces : ℕ) (time_per_piece: ℕ) : ℕ :=
  pieces * time_per_piece

-- Theorem statement to be proven
theorem time_to_finish_furniture : total_time total_furniture minutes_per_piece = 40 := 
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_time_to_finish_furniture_l664_66421


namespace NUMINAMATH_GPT_solution_set_of_inequality_l664_66441

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l664_66441


namespace NUMINAMATH_GPT_roots_of_polynomial_l664_66450

theorem roots_of_polynomial :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l664_66450


namespace NUMINAMATH_GPT_max_AC_not_RS_l664_66433

theorem max_AC_not_RS (TotalCars NoACCars MinRS MaxACnotRS : ℕ)
  (h1 : TotalCars = 100)
  (h2 : NoACCars = 49)
  (h3 : MinRS >= 51)
  (h4 : (TotalCars - NoACCars) - MinRS = MaxACnotRS)
  : MaxACnotRS = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_AC_not_RS_l664_66433


namespace NUMINAMATH_GPT_store_discount_difference_l664_66476

theorem store_discount_difference 
  (p : ℝ) -- original price
  (p1 : ℝ := p * 0.60) -- price after initial discount
  (p2 : ℝ := p1 * 0.90) -- price after additional discount
  (claimed_discount : ℝ := 0.55) -- store's claimed discount
  (true_discount : ℝ := (p - p2) / p) -- calculated true discount
  (difference : ℝ := claimed_discount - true_discount)
  : difference = 0.09 :=
sorry

end NUMINAMATH_GPT_store_discount_difference_l664_66476


namespace NUMINAMATH_GPT_range_of_a_l664_66484

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
    sorry

end NUMINAMATH_GPT_range_of_a_l664_66484


namespace NUMINAMATH_GPT_rank_from_right_l664_66483

theorem rank_from_right (n total rank_left : ℕ) (h1 : rank_left = 5) (h2 : total = 21) : n = total - (rank_left - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_rank_from_right_l664_66483


namespace NUMINAMATH_GPT_fraction_of_selected_films_in_color_l664_66425

variables (x y : ℕ)

theorem fraction_of_selected_films_in_color (B C : ℕ) (e : ℚ)
  (h1 : B = 20 * x)
  (h2 : C = 6 * y)
  (h3 : e = (6 * y : ℚ) / (((y / 5 : ℚ) + 6 * y))) :
  e = 30 / 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_of_selected_films_in_color_l664_66425


namespace NUMINAMATH_GPT_length_of_platform_l664_66404

variables (t L T_p T_s : ℝ)
def train_length := 200  -- length of the train in meters
def platform_cross_time := 50  -- time in seconds to cross the platform
def pole_cross_time := 42  -- time in seconds to cross the signal pole

theorem length_of_platform :
  T_p = platform_cross_time ->
  T_s = pole_cross_time ->
  t = train_length ->
  (L = 38) :=
by
  intros hp hsp ht
  sorry  -- proof goes here

end NUMINAMATH_GPT_length_of_platform_l664_66404


namespace NUMINAMATH_GPT_A_finish_time_l664_66465

theorem A_finish_time {A_work B_work C_work : ℝ} 
  (h1 : A_work + B_work + C_work = 1/4)
  (h2 : B_work = 1/24)
  (h3 : C_work = 1/8) :
  1 / A_work = 12 := by
  sorry

end NUMINAMATH_GPT_A_finish_time_l664_66465


namespace NUMINAMATH_GPT_new_person_weight_l664_66474

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person comes 
in place of one of them weighing 65 kg. Prove that the weight of the new person 
is 128 kg.
-/
theorem new_person_weight (w_old : ℝ) (n : ℝ) (delta_w : ℝ) (w_new : ℝ) 
  (h1 : w_old = 65) 
  (h2 : n = 10) 
  (h3 : delta_w = 6.3) 
  (h4 : w_new = w_old + n * delta_w) : 
  w_new = 128 :=
by 
  rw [h1, h2, h3] at h4 
  rw [h4]
  norm_num

end NUMINAMATH_GPT_new_person_weight_l664_66474


namespace NUMINAMATH_GPT_equivalence_statement_l664_66434

open Complex

noncomputable def distinct_complex (a b c d : ℂ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem equivalence_statement (a b c d : ℂ) (h : distinct_complex a b c d) :
  (∀ (z : ℂ), (abs (z - a) + abs (z - b) ≥ abs (z - c) + abs (z - d)))
  ↔ (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ c = t * a + (1 - t) * b ∧ d = (1 - t) * a + t * b) :=
sorry

end NUMINAMATH_GPT_equivalence_statement_l664_66434


namespace NUMINAMATH_GPT_determine_m_for_value_range_l664_66409

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

theorem determine_m_for_value_range :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_for_value_range_l664_66409


namespace NUMINAMATH_GPT_range_of_a_l664_66403

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a < x ∧ x < a + 1) → (-2 ≤ x ∧ x ≤ 2)) ↔ -2 ≤ a ∧ a ≤ 1 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l664_66403


namespace NUMINAMATH_GPT_positive_integers_congruent_to_2_mod_7_lt_500_count_l664_66449

theorem positive_integers_congruent_to_2_mod_7_lt_500_count : 
  ∃ n : ℕ, n = 72 ∧ ∀ k : ℕ, (k < n → (∃ m : ℕ, (m < 500 ∧ m % 7 = 2) ∧ m = 2 + 7 * k)) := 
by
  sorry

end NUMINAMATH_GPT_positive_integers_congruent_to_2_mod_7_lt_500_count_l664_66449


namespace NUMINAMATH_GPT_trains_crossing_time_l664_66482

noncomputable def TrainA_length := 200  -- meters
noncomputable def TrainA_time := 15  -- seconds
noncomputable def TrainB_length := 300  -- meters
noncomputable def TrainB_time := 25  -- seconds

noncomputable def Speed (length : ℕ) (time : ℕ) := (length : ℝ) / (time : ℝ)

noncomputable def TrainA_speed := Speed TrainA_length TrainA_time
noncomputable def TrainB_speed := Speed TrainB_length TrainB_time

noncomputable def relative_speed := TrainA_speed + TrainB_speed
noncomputable def total_distance := (TrainA_length : ℝ) + (TrainB_length : ℝ)

noncomputable def crossing_time := total_distance / relative_speed

theorem trains_crossing_time :
  (crossing_time : ℝ) = 500 / 25.33 :=
sorry

end NUMINAMATH_GPT_trains_crossing_time_l664_66482


namespace NUMINAMATH_GPT_range_of_a_l664_66422

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + 2 * x + a ≤ 0) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l664_66422


namespace NUMINAMATH_GPT_find_initial_pens_l664_66427

-- Conditions in the form of definitions
def initial_pens (P : ℕ) : ℕ := P
def after_mike (P : ℕ) : ℕ := P + 20
def after_cindy (P : ℕ) : ℕ := 2 * after_mike P
def after_sharon (P : ℕ) : ℕ := after_cindy P - 19

-- The final condition
def final_pens (P : ℕ) : ℕ := 31

-- The goal is to prove that the initial number of pens is 5
theorem find_initial_pens : 
  ∃ (P : ℕ), after_sharon P = final_pens P → P = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_initial_pens_l664_66427


namespace NUMINAMATH_GPT_ellipse_standard_equation_l664_66401

theorem ellipse_standard_equation (a c : ℝ) (h1 : a^2 = 13) (h2 : c^2 = 12) :
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ 
    ((∀ x y : ℝ, (x^2 / 13 + y^2 = 1)) ∨ (∀ x y : ℝ, (x^2 + y^2 / 13 = 1)))) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l664_66401


namespace NUMINAMATH_GPT_flour_needed_l664_66405

-- Definitions
def cups_per_loaf := 2.5
def loaves := 2

-- Statement we want to prove
theorem flour_needed {cups_per_loaf loaves : ℝ} (h : cups_per_loaf = 2.5) (l : loaves = 2) : 
  cups_per_loaf * loaves = 5 :=
sorry

end NUMINAMATH_GPT_flour_needed_l664_66405


namespace NUMINAMATH_GPT_son_l664_66408

variable (M S : ℕ)

theorem son's_age (h1 : M = 4 * S) (h2 : (M - 3) + (S - 3) = 49) : S = 11 :=
by
  sorry

end NUMINAMATH_GPT_son_l664_66408


namespace NUMINAMATH_GPT_mixing_paint_l664_66489

theorem mixing_paint (total_parts : ℕ) (blue_parts : ℕ) (red_parts : ℕ) (white_parts : ℕ) (blue_ounces : ℕ) (max_mixture : ℕ) (ounces_per_part : ℕ) :
  total_parts = blue_parts + red_parts + white_parts →
  blue_parts = 7 →
  red_parts = 2 →
  white_parts = 1 →
  blue_ounces = 140 →
  max_mixture = 180 →
  ounces_per_part = blue_ounces / blue_parts →
  max_mixture / ounces_per_part = 9 →
  white_ounces = white_parts * ounces_per_part →
  white_ounces = 20 :=
sorry

end NUMINAMATH_GPT_mixing_paint_l664_66489


namespace NUMINAMATH_GPT_lcm_of_2_4_5_6_l664_66413

theorem lcm_of_2_4_5_6 : Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 6 = 60 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_2_4_5_6_l664_66413


namespace NUMINAMATH_GPT_twice_abs_difference_of_squares_is_4000_l664_66464

theorem twice_abs_difference_of_squares_is_4000 :
  2 * |(105:ℤ)^2 - (95:ℤ)^2| = 4000 :=
by sorry

end NUMINAMATH_GPT_twice_abs_difference_of_squares_is_4000_l664_66464


namespace NUMINAMATH_GPT_next_consecutive_time_l664_66412

theorem next_consecutive_time (current_hour : ℕ) (current_minute : ℕ) 
  (valid_minutes : 0 ≤ current_minute ∧ current_minute < 60) 
  (valid_hours : 0 ≤ current_hour ∧ current_hour < 24) : 
  current_hour = 4 ∧ current_minute = 56 →
  ∃ next_hour next_minute : ℕ, 
    (0 ≤ next_minute ∧ next_minute < 60) ∧ 
    (0 ≤ next_hour ∧ next_hour < 24) ∧
    (next_hour, next_minute) = (12, 34) ∧ 
    (next_hour * 60 + next_minute) - (current_hour * 60 + current_minute) = 458 := 
by sorry

end NUMINAMATH_GPT_next_consecutive_time_l664_66412


namespace NUMINAMATH_GPT_number_of_students_in_class_l664_66411

theorem number_of_students_in_class (S : ℕ) 
  (h1 : ∀ n : ℕ, 4 * n ≠ 0 → S % 4 = 0) -- S is divisible by 4
  (h2 : ∀ G : ℕ, 3 * G ≠ 0 → (S * 3) % 4 = G) -- Number of students who went to the playground (3/4 * S) is integer
  (h3 : ∀ B : ℕ, G - B ≠ 0 → (G * 2) / 3 = 10) -- Number of girls on the playground
  : S = 20 := sorry

end NUMINAMATH_GPT_number_of_students_in_class_l664_66411


namespace NUMINAMATH_GPT_proof_mn_proof_expr_l664_66481

variables (m n : ℚ)
-- Conditions
def condition1 : Prop := (m + n)^2 = 9
def condition2 : Prop := (m - n)^2 = 1

-- Expected results
def expected_mn : ℚ := 2
def expected_expr : ℚ := 3

-- The theorem to be proved
theorem proof_mn : condition1 m n → condition2 m n → m * n = expected_mn :=
by
  sorry

theorem proof_expr : condition1 m n → condition2 m n → m^2 + n^2 - m * n = expected_expr :=
by
  sorry

end NUMINAMATH_GPT_proof_mn_proof_expr_l664_66481


namespace NUMINAMATH_GPT_determine_n_l664_66400

theorem determine_n (n : ℕ) (x : ℤ) (h : x^n + (2 + x)^n + (2 - x)^n = 0) : n = 1 :=
sorry

end NUMINAMATH_GPT_determine_n_l664_66400


namespace NUMINAMATH_GPT_average_A_B_l664_66402

variables (A B C : ℝ)

def conditions (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (B + C) / 2 = 43 ∧
  B = 31

theorem average_A_B (A B C : ℝ) (h : conditions A B C) : (A + B) / 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_A_B_l664_66402


namespace NUMINAMATH_GPT_longest_side_of_rectangle_l664_66423

theorem longest_side_of_rectangle 
    (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 2400) : 
    max l w = 80 :=
by sorry

end NUMINAMATH_GPT_longest_side_of_rectangle_l664_66423


namespace NUMINAMATH_GPT_distinct_positive_integers_criteria_l664_66495

theorem distinct_positive_integers_criteria (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)
  (hxyz_div : x * y * z ∣ (x * y - 1) * (y * z - 1) * (z * x - 1)) :
  (x, y, z) = (2, 3, 5) ∨ (x, y, z) = (2, 5, 3) ∨ (x, y, z) = (3, 2, 5) ∨
  (x, y, z) = (3, 5, 2) ∨ (x, y, z) = (5, 2, 3) ∨ (x, y, z) = (5, 3, 2) :=
by sorry

end NUMINAMATH_GPT_distinct_positive_integers_criteria_l664_66495


namespace NUMINAMATH_GPT_soda_cost_proof_l664_66485

theorem soda_cost_proof (b s : ℤ) (h1 : 4 * b + 3 * s = 440) (h2 : 3 * b + 2 * s = 310) : s = 80 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_proof_l664_66485


namespace NUMINAMATH_GPT_linear_polynomial_divisible_49_l664_66491

theorem linear_polynomial_divisible_49 {P : ℕ → Polynomial ℚ} :
    let Q := Polynomial.C 1 * (Polynomial.X ^ 8) + Polynomial.C 1 * (Polynomial.X ^ 7)
    ∃ a b x, (P x) = Polynomial.C a * Polynomial.X + Polynomial.C b ∧ a ≠ 0 ∧ 
              (∀ i, P (i + 1) = (Polynomial.C 1 * Polynomial.X + Polynomial.C 1) * P i ∨ 
                            P (i + 1) = Polynomial.derivative (P i)) →
              (a - b) % 49 = 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_polynomial_divisible_49_l664_66491


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l664_66417

theorem largest_divisor_of_expression
  (x : ℤ) (h_odd : x % 2 = 1) : 
  ∃ k : ℤ, k = 40 ∧ 40 ∣ (12 * x + 2) * (8 * x + 14) * (10 * x + 10) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l664_66417


namespace NUMINAMATH_GPT_chocolate_bars_in_small_box_l664_66419

-- Given conditions
def num_small_boxes : ℕ := 21
def total_chocolate_bars : ℕ := 525

-- Statement to prove
theorem chocolate_bars_in_small_box : total_chocolate_bars / num_small_boxes = 25 := by
  sorry

end NUMINAMATH_GPT_chocolate_bars_in_small_box_l664_66419


namespace NUMINAMATH_GPT_prob1_prob2_l664_66414

-- Definitions and conditions for Problem 1
def U : Set ℝ := {x | x ≤ 4}
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof Problem 1: Equivalent Lean proof statement
theorem prob1 : (U \ A) ∩ B = {-3, -2, 3} := by
  sorry

-- Definitions and conditions for Problem 2
def tan_alpha_eq_3 (α : ℝ) : Prop := Real.tan α = 3

-- Proof Problem 2: Equivalent Lean proof statement
theorem prob2 (α : ℝ) (h : tan_alpha_eq_3 α) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 ∧
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = -4 / 5 := by
  sorry

end NUMINAMATH_GPT_prob1_prob2_l664_66414


namespace NUMINAMATH_GPT_product_telescope_l664_66451

theorem product_telescope : ((1 + (1 / 1)) * 
                             (1 + (1 / 2)) * 
                             (1 + (1 / 3)) * 
                             (1 + (1 / 4)) * 
                             (1 + (1 / 5)) * 
                             (1 + (1 / 6)) * 
                             (1 + (1 / 7)) * 
                             (1 + (1 / 8)) * 
                             (1 + (1 / 9)) * 
                             (1 + (1 / 10))) = 11 := 
by
  sorry

end NUMINAMATH_GPT_product_telescope_l664_66451


namespace NUMINAMATH_GPT_pencils_ordered_l664_66442

theorem pencils_ordered (pencils_per_student : ℕ) (number_of_students : ℕ) (total_pencils : ℕ) :
  pencils_per_student = 3 →
  number_of_students = 65 →
  total_pencils = pencils_per_student * number_of_students →
  total_pencils = 195 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_pencils_ordered_l664_66442


namespace NUMINAMATH_GPT_problem_statement_l664_66497

def op (x y : ℝ) : ℝ := (x + 3) * (y - 1)

theorem problem_statement (a : ℝ) : (∀ x : ℝ, op (x - a) (x + a) > -16) ↔ -2 < a ∧ a < 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l664_66497


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_l664_66452

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  4 ≤ (1 / a) + (1 / b) :=
sorry

end NUMINAMATH_GPT_minimum_reciprocal_sum_l664_66452


namespace NUMINAMATH_GPT_smallest_value_z_minus_x_l664_66440

theorem smallest_value_z_minus_x 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hmul : x * y * z = 5040) 
  (hxy : x < y) 
  (hyz : y < z) : 
  z - x = 9 := 
  sorry

end NUMINAMATH_GPT_smallest_value_z_minus_x_l664_66440


namespace NUMINAMATH_GPT_prob_at_least_one_l664_66467

-- Defining the probabilities of the alarms going off on time
def prob_A : ℝ := 0.80
def prob_B : ℝ := 0.90

-- Define the complementary event (neither alarm goes off on time)
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

-- The main theorem statement we need to prove
theorem prob_at_least_one : 1 - prob_neither = 0.98 :=
by
  sorry

end NUMINAMATH_GPT_prob_at_least_one_l664_66467


namespace NUMINAMATH_GPT_base8_base6_positive_integer_l664_66447

theorem base8_base6_positive_integer (C D N : ℕ)
  (base8: N = 8 * C + D)
  (base6: N = 6 * D + C)
  (valid_C_base8: C < 8)
  (valid_D_base6: D < 6)
  (valid_C_D: 7 * C = 5 * D)
: N = 43 := by
  sorry

end NUMINAMATH_GPT_base8_base6_positive_integer_l664_66447


namespace NUMINAMATH_GPT_population_total_l664_66463

variable (x y : ℕ)

theorem population_total (h1 : 20 * y = 12 * y * (x + y)) : x + y = 240 :=
  by
  -- Proceed with solving the provided conditions.
  sorry

end NUMINAMATH_GPT_population_total_l664_66463


namespace NUMINAMATH_GPT_calculate_fraction_product_l664_66426

noncomputable def b8 := 2 * (8^2) + 6 * (8^1) + 2 * (8^0) -- 262_8 in base 10
noncomputable def b4 := 1 * (4^1) + 3 * (4^0) -- 13_4 in base 10
noncomputable def b7 := 1 * (7^2) + 4 * (7^1) + 4 * (7^0) -- 144_7 in base 10
noncomputable def b5 := 2 * (5^1) + 4 * (5^0) -- 24_5 in base 10

theorem calculate_fraction_product : 
  ((b8 : ℕ) / (b4 : ℕ)) * ((b7 : ℕ) / (b5 : ℕ)) = 147 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_product_l664_66426


namespace NUMINAMATH_GPT_total_travel_time_is_correct_l664_66445

-- Conditions as definitions
def total_distance : ℕ := 200
def initial_fraction : ℚ := 1 / 4
def initial_time : ℚ := 1 -- in hours
def lunch_time : ℚ := 1 -- in hours
def remaining_fraction : ℚ := 1 / 2
def pit_stop_time : ℚ := 0.5 -- in hours
def speed_increase : ℚ := 10

-- Derived/Calculated values needed for the problem statement
def initial_distance : ℚ := initial_fraction * total_distance
def initial_speed : ℚ := initial_distance / initial_time
def remaining_distance : ℚ := total_distance - initial_distance
def half_remaining_distance : ℚ := remaining_fraction * remaining_distance
def second_drive_time : ℚ := half_remaining_distance / initial_speed
def last_distance : ℚ := remaining_distance - half_remaining_distance
def last_speed : ℚ := initial_speed + speed_increase
def last_drive_time : ℚ := last_distance / last_speed

-- Total time calculation
def total_time : ℚ :=
  initial_time + lunch_time + second_drive_time + pit_stop_time + last_drive_time

-- Lean theorem statement
theorem total_travel_time_is_correct : total_time = 5.25 :=
  sorry

end NUMINAMATH_GPT_total_travel_time_is_correct_l664_66445


namespace NUMINAMATH_GPT_correct_statements_count_l664_66435

-- Definitions
def proper_fraction (x : ℚ) : Prop := (0 < x) ∧ (x < 1)
def improper_fraction (x : ℚ) : Prop := (x ≥ 1)

-- Statements as conditions
def statement1 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a + b)
def statement2 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a * b)
def statement3 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a + b)
def statement4 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a * b)

-- The main theorem stating the correct answer
theorem correct_statements_count : 
  (¬ (∀ a b, statement1 a b)) ∧ 
  (∀ a b, statement2 a b) ∧ 
  (∀ a b, statement3 a b) ∧ 
  (¬ (∀ a b, statement4 a b)) → 
  (2 = 2)
:= by sorry

end NUMINAMATH_GPT_correct_statements_count_l664_66435


namespace NUMINAMATH_GPT_total_games_played_l664_66432

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end NUMINAMATH_GPT_total_games_played_l664_66432


namespace NUMINAMATH_GPT_find_m_l664_66429

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def f' (x : ℝ) : ℝ := -1 / (x^2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * x

theorem find_m (m : ℝ) :
  g 2 m = 1 / (f' 2) →
  m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l664_66429


namespace NUMINAMATH_GPT_volume_of_released_gas_l664_66479

def mol_co2 : ℝ := 2.4
def molar_volume : ℝ := 22.4

theorem volume_of_released_gas : mol_co2 * molar_volume = 53.76 := by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_volume_of_released_gas_l664_66479


namespace NUMINAMATH_GPT_goose_eggs_count_l664_66468

theorem goose_eggs_count (E : ℝ) (h1 : 1 / 4 * E = (1 / 4) * E)
  (h2 : 4 / 5 * (1 / 4) * E = (4 / 5) * (1 / 4) * E)
  (h3 : 3 / 5 * (4 / 5) * (1 / 4) * E = 120)
  (h4 : 120 = 120)
  : E = 800 :=
by
  sorry

end NUMINAMATH_GPT_goose_eggs_count_l664_66468


namespace NUMINAMATH_GPT_area_ratio_l664_66443

-- Definitions corresponding to the conditions
variable {A B C P Q R : Type}
variable (t : ℝ)
variable (h_pos : 0 < t) (h_lt_one : t < 1)

-- Define the areas in terms of provided conditions
noncomputable def area_AP : ℝ := sorry
noncomputable def area_BQ : ℝ := sorry
noncomputable def area_CR : ℝ := sorry
noncomputable def K : ℝ := area_AP * area_BQ * area_CR
noncomputable def L : ℝ := sorry -- Area of triangle ABC

-- The statement to be proved
theorem area_ratio (h_pos : 0 < t) (h_lt_one : t < 1) :
  (K / L) = (1 - t + t^2)^2 :=
sorry

end NUMINAMATH_GPT_area_ratio_l664_66443


namespace NUMINAMATH_GPT_unique_rhombus_property_not_in_rectangle_l664_66487

-- Definitions of properties for a rhombus and a rectangle
def is_rhombus (sides_equal : Prop) (opposite_sides_parallel : Prop) (opposite_angles_equal : Prop)
  (diagonals_perpendicular_and_bisect : Prop) : Prop :=
  sides_equal ∧ opposite_sides_parallel ∧ opposite_angles_equal ∧ diagonals_perpendicular_and_bisect

def is_rectangle (opposite_sides_equal_and_parallel : Prop) (all_angles_right : Prop)
  (diagonals_equal_and_bisect : Prop) : Prop :=
  opposite_sides_equal_and_parallel ∧ all_angles_right ∧ diagonals_equal_and_bisect

-- Proof objective: Prove that the unique property of a rhombus is the perpendicular and bisecting nature of its diagonals
theorem unique_rhombus_property_not_in_rectangle :
  ∀ (sides_equal opposite_sides_parallel opposite_angles_equal
      diagonals_perpendicular_and_bisect opposite_sides_equal_and_parallel
      all_angles_right diagonals_equal_and_bisect : Prop),
  is_rhombus sides_equal opposite_sides_parallel opposite_angles_equal diagonals_perpendicular_and_bisect →
  is_rectangle opposite_sides_equal_and_parallel all_angles_right diagonals_equal_and_bisect →
  diagonals_perpendicular_and_bisect ∧ ¬diagonals_equal_and_bisect :=
by
  sorry

end NUMINAMATH_GPT_unique_rhombus_property_not_in_rectangle_l664_66487


namespace NUMINAMATH_GPT_polynomial_value_at_4_l664_66494

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_4_l664_66494


namespace NUMINAMATH_GPT_eggs_in_nests_l664_66486

theorem eggs_in_nests (x : ℕ) (h1 : 2 * x + 3 + 4 = 17) : x = 5 :=
by
  /- This is where the proof would go, but the problem only requires the statement -/
  sorry

end NUMINAMATH_GPT_eggs_in_nests_l664_66486


namespace NUMINAMATH_GPT_number_of_children_l664_66492

theorem number_of_children (C A : ℕ) (h1 : C = 2 * A) (h2 : C + A = 120) : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l664_66492


namespace NUMINAMATH_GPT_sum_of_possible_values_l664_66454

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) : (∃ x y : ℝ, x * (x - 4) = -21 ∧ y * (y - 4) = -21 ∧ x + y = 4) :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l664_66454


namespace NUMINAMATH_GPT_gcd_2952_1386_l664_66470

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end NUMINAMATH_GPT_gcd_2952_1386_l664_66470


namespace NUMINAMATH_GPT_symmetric_about_y_axis_l664_66488

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem symmetric_about_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  unfold f
  sorry

end NUMINAMATH_GPT_symmetric_about_y_axis_l664_66488


namespace NUMINAMATH_GPT_m_plus_n_eq_47_l664_66499

theorem m_plus_n_eq_47 (m n : ℕ)
  (h1 : m + 8 < n - 1)
  (h2 : (m + m + 3 + m + 8 + n - 1 + n + 3 + 2 * n - 2) / 6 = n)
  (h3 : (m + 8 + (n - 1)) / 2 = n) :
  m + n = 47 :=
sorry

end NUMINAMATH_GPT_m_plus_n_eq_47_l664_66499


namespace NUMINAMATH_GPT_math_problem_l664_66436

theorem math_problem : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end NUMINAMATH_GPT_math_problem_l664_66436


namespace NUMINAMATH_GPT_new_tax_rate_is_correct_l664_66424

noncomputable def new_tax_rate (old_rate : ℝ) (income : ℝ) (savings : ℝ) : ℝ := 
  let old_tax := old_rate * income / 100
  let new_tax := (income - savings) / income * old_tax
  let rate := new_tax / income * 100
  rate

theorem new_tax_rate_is_correct :
  ∀ (income : ℝ) (old_rate : ℝ) (savings : ℝ),
    old_rate = 42 →
    income = 34500 →
    savings = 4830 →
    new_tax_rate old_rate income savings = 28 := 
by
  intros income old_rate savings h1 h2 h3
  sorry

end NUMINAMATH_GPT_new_tax_rate_is_correct_l664_66424


namespace NUMINAMATH_GPT_sequence_diff_ge_abs_m_l664_66475

-- Define the conditions and theorem in Lean

theorem sequence_diff_ge_abs_m
    (m : ℤ) (h_m : |m| ≥ 2)
    (a : ℕ → ℤ)
    (h_seq_not_zero : ¬ (a 1 = 0 ∧ a 2 = 0))
    (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - m * a n)
    (r s : ℕ) (h_r : r > s) (h_s : s ≥ 2)
    (h_equal : a r = a 1 ∧ a s = a 1) :
    r - s ≥ |m| :=
by
  sorry

end NUMINAMATH_GPT_sequence_diff_ge_abs_m_l664_66475


namespace NUMINAMATH_GPT_part1_part2_l664_66431

namespace VectorProblem

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def m := 5 / 9
def n := 8 / 9

def k := -16 / 13

-- Statement 1: Prove vectors satisfy the linear combination
theorem part1 : vector_a = (m * vector_b.1 + n * vector_c.1, m * vector_b.2 + n * vector_c.2) :=
by {
  sorry
}

-- Statement 2: Prove vectors are parallel
theorem part2 : (3 + 4 * k) * 2 + (2 + k) * 5 = 0 :=
by {
  sorry
}

end VectorProblem

end NUMINAMATH_GPT_part1_part2_l664_66431


namespace NUMINAMATH_GPT_largest_prime_factor_2999_l664_66437

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  -- Note: This would require actual computation logic to find the largest prime factor.
  sorry

theorem largest_prime_factor_2999 :
  largest_prime_factor 2999 = 103 :=
by 
  -- Given conditions:
  -- 1. 2999 is an odd number (doesn't need explicit condition in proof).
  -- 2. Sum of digits is 29, thus not divisible by 3.
  -- 3. 2999 is not divisible by 11.
  -- 4. 2999 is not divisible by 7, 13, 17, 19.
  -- 5. Prime factorization of 2999 is 29 * 103.
  admit -- actual proof will need detailed prime factor test results 

end NUMINAMATH_GPT_largest_prime_factor_2999_l664_66437


namespace NUMINAMATH_GPT_find_number_with_21_multiples_of_4_l664_66407

theorem find_number_with_21_multiples_of_4 (n : ℕ) (h₁ : ∀ k : ℕ, n + k * 4 ≤ 92 → k < 21) : n = 80 :=
sorry

end NUMINAMATH_GPT_find_number_with_21_multiples_of_4_l664_66407


namespace NUMINAMATH_GPT_find_num_trumpet_players_l664_66446

namespace OprahWinfreyHighSchoolMarchingBand

def num_trumpet_players (total_weight : ℕ) 
  (num_clarinet : ℕ) (num_trombone : ℕ) 
  (num_tuba : ℕ) (num_drum : ℕ) : ℕ :=
(total_weight - 
  ((num_clarinet * 5) + 
  (num_trombone * 10) + 
  (num_tuba * 20) + 
  (num_drum * 15)))
  / 5

theorem find_num_trumpet_players :
  num_trumpet_players 245 9 8 3 2 = 6 :=
by
  -- calculation and reasoning steps would go here
  sorry

end OprahWinfreyHighSchoolMarchingBand

end NUMINAMATH_GPT_find_num_trumpet_players_l664_66446


namespace NUMINAMATH_GPT_cups_per_girl_l664_66462

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end NUMINAMATH_GPT_cups_per_girl_l664_66462


namespace NUMINAMATH_GPT_area_of_parallelogram_l664_66490

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 18) (h_height : height = 16) : 
  base * height = 288 := 
by
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l664_66490


namespace NUMINAMATH_GPT_problem1_problem2_l664_66410

-- Problem 1
theorem problem1 : (-2) ^ 2 + (Real.sqrt 2 - 1) ^ 0 - 1 = 4 := by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) (A : ℝ) (B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) : a > 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l664_66410


namespace NUMINAMATH_GPT_greatest_num_consecutive_integers_l664_66460

theorem greatest_num_consecutive_integers (N a : ℤ) (h : (N * (2*a + N - 1) = 210)) :
  ∃ N, N = 210 :=
sorry

end NUMINAMATH_GPT_greatest_num_consecutive_integers_l664_66460


namespace NUMINAMATH_GPT_find_distance_between_sides_l664_66469

-- Define the given conditions
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area_trapezium : ℝ := 247

-- Define the distance h between parallel sides
def distance_between_sides (h : ℝ) : Prop :=
  area_trapezium = (1 / 2) * (length_side1 + length_side2) * h

-- Define the theorem we want to prove
theorem find_distance_between_sides : ∃ h : ℝ, distance_between_sides h ∧ h = 13 := by
  sorry

end NUMINAMATH_GPT_find_distance_between_sides_l664_66469


namespace NUMINAMATH_GPT_largest_reciprocal_l664_66456

theorem largest_reciprocal: 
  let A := -(1 / 4)
  let B := 2 / 7
  let C := -2
  let D := 3
  let E := -(3 / 2)
  let reciprocal (x : ℚ) := 1 / x
  reciprocal B > reciprocal A ∧
  reciprocal B > reciprocal C ∧
  reciprocal B > reciprocal D ∧
  reciprocal B > reciprocal E :=
by
  sorry

end NUMINAMATH_GPT_largest_reciprocal_l664_66456


namespace NUMINAMATH_GPT_average_speed_difference_l664_66439

noncomputable def v_R : Float := 56.44102863722254
noncomputable def distance : Float := 750
noncomputable def t_R : Float := distance / v_R
noncomputable def t_P : Float := t_R - 2
noncomputable def v_P : Float := distance / t_P

theorem average_speed_difference : v_P - v_R = 10 := by
  sorry

end NUMINAMATH_GPT_average_speed_difference_l664_66439


namespace NUMINAMATH_GPT_calc_abc_squares_l664_66498

theorem calc_abc_squares :
  ∀ (a b c : ℝ),
  a^2 + 3 * b = 14 →
  b^2 + 5 * c = -13 →
  c^2 + 7 * a = -26 →
  a^2 + b^2 + c^2 = 20.75 :=
by
  intros a b c h1 h2 h3
  -- The proof is omitted; reasoning is provided in the solution.
  sorry

end NUMINAMATH_GPT_calc_abc_squares_l664_66498


namespace NUMINAMATH_GPT_range_of_a_l664_66420

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l664_66420


namespace NUMINAMATH_GPT_find_k_l664_66466

def A (a b : ℤ) : Prop := 3 * a + b - 2 = 0
def B (a b : ℤ) (k : ℤ) : Prop := k * (a^2 - a + 1) - b = 0

theorem find_k (k : ℤ) (h : ∃ a b : ℤ, A a b ∧ B a b k ∧ a > 0) : k = -1 ∨ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l664_66466


namespace NUMINAMATH_GPT_sum_of_fractions_equals_three_l664_66496

-- Definitions according to the conditions
def proper_fraction (a b : ℕ) := 1 ≤ a ∧ a < b
def improper_fraction (a b : ℕ) := a ≥ b
def mixed_number (a b c : ℕ) := a + b / c

-- Constants according to the given problem
def n := 8
def d := 9
def improper_n := 9

-- Values for elements in the conditions
def largest_proper_fraction := n / d
def smallest_improper_fraction := improper_n / d
def smallest_mixed_number := 1 + 1 / d

-- Theorem statement with the correct answer
theorem sum_of_fractions_equals_three :
  largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_fractions_equals_three_l664_66496


namespace NUMINAMATH_GPT_problem_conditions_l664_66477

noncomputable def f (a b c x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem problem_conditions (a b c : ℝ) (h0 : a + b + c = 0)
  (h1 : f a b c 0 > 0) (h2 : f a b c 1 > 0) :
    (a > 0 ∧ -2 < b / a ∧ b / a < -1) ∧
    (∃ z1 z2 : ℝ, 0 < z1 ∧ z1 < 1 ∧ 0 < z2 ∧ z2 < 1 ∧ z1 ≠ z2 ∧ f a b c z1 = 0 ∧ f a b c z2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l664_66477


namespace NUMINAMATH_GPT_collinear_example_l664_66493

structure Vector2D where
  x : ℝ
  y : ℝ

def collinear (u v : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.x = k * u.x ∧ v.y = k * u.y

def a : Vector2D := ⟨1, 2⟩
def b : Vector2D := ⟨2, 4⟩

theorem collinear_example :
  collinear a b :=
by
  sorry

end NUMINAMATH_GPT_collinear_example_l664_66493


namespace NUMINAMATH_GPT_ratio_of_waist_to_hem_l664_66444

theorem ratio_of_waist_to_hem
  (cuffs_length : ℕ)
  (hem_length : ℕ)
  (ruffles_length : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℕ)
  (total_spent : ℕ)
  (waist_length : ℕ) :
  cuffs_length = 50 →
  hem_length = 300 →
  ruffles_length = 20 →
  num_ruffles = 5 →
  lace_cost_per_meter = 6 →
  total_spent = 36 →
  waist_length = (total_spent / lace_cost_per_meter * 100) -
                (2 * cuffs_length + hem_length + num_ruffles * ruffles_length) →
  waist_length / hem_length = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_waist_to_hem_l664_66444


namespace NUMINAMATH_GPT_mod_inverse_17_1200_l664_66458

theorem mod_inverse_17_1200 : ∃ x : ℕ, x < 1200 ∧ 17 * x % 1200 = 1 := 
by
  use 353
  sorry

end NUMINAMATH_GPT_mod_inverse_17_1200_l664_66458


namespace NUMINAMATH_GPT_find_k1_over_k2_plus_k2_over_k1_l664_66448

theorem find_k1_over_k2_plus_k2_over_k1 (p q k k1 k2 : ℚ)
  (h1 : k * (p^2) - (2 * k - 3) * p + 7 = 0)
  (h2 : k * (q^2) - (2 * k - 3) * q + 7 = 0)
  (h3 : p ≠ 0)
  (h4 : q ≠ 0)
  (h5 : k ≠ 0)
  (h6 : k1 ≠ 0)
  (h7 : k2 ≠ 0)
  (h8 : p / q + q / p = 6 / 7)
  (h9 : (p + q) = (2 * k - 3) / k)
  (h10 : p * q = 7 / k)
  (h11 : k1 + k2 = 6)
  (h12 : k1 * k2 = 9 / 4) :
  (k1 / k2 + k2 / k1 = 14) :=
  sorry

end NUMINAMATH_GPT_find_k1_over_k2_plus_k2_over_k1_l664_66448


namespace NUMINAMATH_GPT_first_dig_site_date_difference_l664_66478

-- Definitions for the conditions
def F : Int := sorry  -- The age of the first dig site
def S : Int := sorry  -- The age of the second dig site
def T : Int := sorry  -- The age of the third dig site
def Fo : Int := 8400  -- The age of the fourth dig site
def x : Int := (S - F)

-- The conditions
axiom condition1 : F = S + x
axiom condition2 : T = F + 3700
axiom condition3 : Fo = 2 * T
axiom condition4 : S = 852
axiom condition5 : S > F  -- Ensuring S is older than F for meaningfulness

-- The theorem to prove
theorem first_dig_site_date_difference : x = 352 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_first_dig_site_date_difference_l664_66478


namespace NUMINAMATH_GPT_laura_bought_4_shirts_l664_66480

-- Definitions for the conditions
def pants_price : ℕ := 54
def num_pants : ℕ := 2
def shirt_price : ℕ := 33
def given_money : ℕ := 250
def change_received : ℕ := 10

-- Proving the number of shirts bought is 4
theorem laura_bought_4_shirts :
  (num_pants * pants_price) + (shirt_price * 4) + change_received = given_money :=
by
  sorry

end NUMINAMATH_GPT_laura_bought_4_shirts_l664_66480


namespace NUMINAMATH_GPT_jill_peaches_l664_66430

-- Definitions based on conditions in a
def Steven_has_peaches : ℕ := 19
def Steven_more_than_Jill : ℕ := 13

-- Statement to prove Jill's peaches
theorem jill_peaches : (Steven_has_peaches - Steven_more_than_Jill = 6) :=
by
  sorry

end NUMINAMATH_GPT_jill_peaches_l664_66430


namespace NUMINAMATH_GPT_ellipse_semi_minor_axis_l664_66406

theorem ellipse_semi_minor_axis (b : ℝ) 
    (h1 : 0 < b) 
    (h2 : b < 5)
    (h_ellipse : ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1) 
    (h_eccentricity : 4 / 5 = 4 / 5) : b = 3 := 
sorry

end NUMINAMATH_GPT_ellipse_semi_minor_axis_l664_66406


namespace NUMINAMATH_GPT_trees_in_one_row_l664_66453

variable (total_trees_cleaned : ℕ)
variable (trees_per_row : ℕ)

theorem trees_in_one_row (h1 : total_trees_cleaned = 20) (h2 : trees_per_row = 5) :
  (total_trees_cleaned / trees_per_row) = 4 :=
by
  sorry

end NUMINAMATH_GPT_trees_in_one_row_l664_66453


namespace NUMINAMATH_GPT_find_F_l664_66459

theorem find_F (F C : ℝ) (h1 : C = 30) (h2 : C = (5 / 9) * (F - 30)) : F = 84 := by
  sorry

end NUMINAMATH_GPT_find_F_l664_66459


namespace NUMINAMATH_GPT_initial_number_of_orchids_l664_66428

theorem initial_number_of_orchids 
  (initial_orchids : ℕ)
  (cut_orchids : ℕ)
  (final_orchids : ℕ)
  (h_cut : cut_orchids = 19)
  (h_final : final_orchids = 21) :
  initial_orchids + cut_orchids = final_orchids → initial_orchids = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_orchids_l664_66428


namespace NUMINAMATH_GPT_find_X_in_rectangle_diagram_l664_66418

theorem find_X_in_rectangle_diagram :
  ∀ (X : ℝ),
  (1 + 1 + 1 + 2 + X = 1 + 2 + 1 + 6) → X = 5 :=
by
  intros X h
  sorry

end NUMINAMATH_GPT_find_X_in_rectangle_diagram_l664_66418


namespace NUMINAMATH_GPT_contrapositive_ex_l664_66471

theorem contrapositive_ex (x y : ℝ)
  (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) :
  ¬ (x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_ex_l664_66471


namespace NUMINAMATH_GPT_f_is_32x5_l664_66416

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end NUMINAMATH_GPT_f_is_32x5_l664_66416


namespace NUMINAMATH_GPT_number_of_pizzas_ordered_l664_66438

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 8

-- Define the number of slices each person ate
def slices_per_person : ℕ := 4

-- Define the total number of slices eaten
def total_slices_eaten : ℕ := total_people * slices_per_person

-- Prove that the number of pizzas needed is 3
theorem number_of_pizzas_ordered : total_slices_eaten / slices_per_pizza = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_pizzas_ordered_l664_66438


namespace NUMINAMATH_GPT_evaluate_expression_l664_66472

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l664_66472
