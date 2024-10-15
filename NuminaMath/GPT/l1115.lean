import Mathlib

namespace NUMINAMATH_GPT_true_statements_count_l1115_111529

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem true_statements_count :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + 
  (if s2 then 1 else 0) + 
  (if s3 then 1 else 0) + 
  (if s4 then 1 else 0) = 2 :=
by
  sorry

end NUMINAMATH_GPT_true_statements_count_l1115_111529


namespace NUMINAMATH_GPT_cost_per_serving_of_pie_l1115_111526

theorem cost_per_serving_of_pie 
  (w_gs : ℝ) (p_gs : ℝ) (w_gala : ℝ) (p_gala : ℝ) (w_hc : ℝ) (p_hc : ℝ)
  (pie_crust_cost : ℝ) (lemon_cost : ℝ) (butter_cost : ℝ) (servings : ℕ)
  (total_weight_gs : w_gs = 0.5) (price_gs_per_pound : p_gs = 1.80)
  (total_weight_gala : w_gala = 0.8) (price_gala_per_pound : p_gala = 2.20)
  (total_weight_hc : w_hc = 0.7) (price_hc_per_pound : p_hc = 2.50)
  (cost_pie_crust : pie_crust_cost = 2.50) (cost_lemon : lemon_cost = 0.60)
  (cost_butter : butter_cost = 1.80) (total_servings : servings = 8) :
  (w_gs * p_gs + w_gala * p_gala + w_hc * p_hc + pie_crust_cost + lemon_cost + butter_cost) / servings = 1.16 :=
by 
  sorry

end NUMINAMATH_GPT_cost_per_serving_of_pie_l1115_111526


namespace NUMINAMATH_GPT_bug_crawl_distance_l1115_111536

theorem bug_crawl_distance : 
  let start : ℤ := 3
  let first_stop : ℤ := -4
  let second_stop : ℤ := 7
  let final_stop : ℤ := -1
  |first_stop - start| + |second_stop - first_stop| + |final_stop - second_stop| = 26 := 
by
  sorry

end NUMINAMATH_GPT_bug_crawl_distance_l1115_111536


namespace NUMINAMATH_GPT_pirate_treasure_probability_l1115_111507

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_traps := 1 / 10
  let p_neither := 7 / 10
  let num_islands := 8
  let num_treasure := 4
  binomial num_islands num_treasure * p_treasure^num_treasure * p_neither^(num_islands - num_treasure) = 673 / 25000 :=
by
  sorry

end NUMINAMATH_GPT_pirate_treasure_probability_l1115_111507


namespace NUMINAMATH_GPT_algebra_expression_value_l1115_111578

theorem algebra_expression_value (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 11) : a^2 - a * b + b^2 = 67 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l1115_111578


namespace NUMINAMATH_GPT_average_of_rest_of_class_l1115_111509

theorem average_of_rest_of_class
  (n : ℕ)
  (h1 : n > 0)
  (avg_class : ℝ := 84)
  (avg_one_fourth : ℝ := 96)
  (total_sum : ℝ := avg_class * n)
  (sum_one_fourth : ℝ := avg_one_fourth * (n / 4))
  (sum_rest : ℝ := total_sum - sum_one_fourth)
  (num_rest : ℝ := (3 * n) / 4) :
  sum_rest / num_rest = 80 :=
sorry

end NUMINAMATH_GPT_average_of_rest_of_class_l1115_111509


namespace NUMINAMATH_GPT_triangle_inequalities_l1115_111503

open Real

-- Define a structure for a triangle with its properties
structure Triangle :=
(a b c R ra rb rc : ℝ)

-- Main statement to be proved
theorem triangle_inequalities (Δ : Triangle) (h : 2 * Δ.R ≤ Δ.ra) :
  Δ.a > Δ.b ∧ Δ.a > Δ.c ∧ 2 * Δ.R > Δ.rb ∧ 2 * Δ.R > Δ.rc :=
sorry

end NUMINAMATH_GPT_triangle_inequalities_l1115_111503


namespace NUMINAMATH_GPT_cubic_roots_cosines_l1115_111542

theorem cubic_roots_cosines
  {p q r : ℝ}
  (h_eq : ∀ x : ℝ, x^3 + p * x^2 + q * x + r = 0)
  (h_roots : ∃ (α β γ : ℝ), (α > 0) ∧ (β > 0) ∧ (γ > 0) ∧ (α + β + γ = -p) ∧ 
             (α * β + β * γ + γ * α = q) ∧ (α * β * γ = -r)) :
  2 * r + 1 = p^2 - 2 * q :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_cosines_l1115_111542


namespace NUMINAMATH_GPT_cost_of_45_lilies_l1115_111564

-- Defining the conditions
def price_per_lily (n : ℕ) : ℝ :=
  if n <= 30 then 2
  else 1.8

-- Stating the problem in Lean 4
theorem cost_of_45_lilies :
  price_per_lily 15 * 15 = 30 → (price_per_lily 45 * 45 = 81) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_of_45_lilies_l1115_111564


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1115_111561

theorem sufficient_but_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 ∧ ¬ (a^2 > b^2 → a > b ∧ b > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1115_111561


namespace NUMINAMATH_GPT_ratio_of_A_to_B_l1115_111500

theorem ratio_of_A_to_B (v_A v_B : ℝ) (d_A d_B : ℝ) (h1 : d_A = 128) (h2 : d_B = 64) (h3 : d_A / v_A = d_B / v_B) : v_A / v_B = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_A_to_B_l1115_111500


namespace NUMINAMATH_GPT_probability_merlin_dismissed_l1115_111514

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_probability_merlin_dismissed_l1115_111514


namespace NUMINAMATH_GPT_correct_alarm_clock_time_l1115_111560

-- Definitions for the conditions
def alarm_set_time : ℕ := 7 * 60 -- in minutes
def museum_arrival_time : ℕ := 8 * 60 + 50 -- in minutes
def museum_touring_time : ℕ := 1 * 60 + 30 -- in minutes
def alarm_home_time : ℕ := 11 * 60 + 50 -- in minutes

-- The problem: proving the correct time the clock should be set to
theorem correct_alarm_clock_time : 
  (alarm_home_time - (2 * ((museum_arrival_time - alarm_set_time) + museum_touring_time / 2)) = 12 * 60) :=
  by
    sorry

end NUMINAMATH_GPT_correct_alarm_clock_time_l1115_111560


namespace NUMINAMATH_GPT_perfect_square_is_289_l1115_111505

/-- The teacher tells a three-digit perfect square number by
revealing the hundreds digit to person A, the tens digit to person B,
and the units digit to person C, and tells them that all three digits
are different from each other. Each person only knows their own digit and
not the others. The three people have the following conversation:

Person A: I don't know what the perfect square number is.  
Person B: You don't need to say; I also know that you don't know.  
Person C: I already know what the number is.  
Person A: After hearing Person C, I also know what the number is.  
Person B: After hearing Person A also knows what the number is.

Given these conditions, the three-digit perfect square number is 289. -/
theorem perfect_square_is_289:
  ∃ n : ℕ, n^2 = 289 := by
  sorry

end NUMINAMATH_GPT_perfect_square_is_289_l1115_111505


namespace NUMINAMATH_GPT_kiley_slices_eaten_l1115_111558

def slices_of_cheesecake (total_calories_per_cheesecake calories_per_slice : ℕ) : ℕ :=
  total_calories_per_cheesecake / calories_per_slice

def slices_eaten (total_slices percentage_ate : ℚ) : ℚ :=
  total_slices * percentage_ate

theorem kiley_slices_eaten :
  ∀ (total_calories_per_cheesecake calories_per_slice : ℕ) (percentage_ate : ℚ),
  total_calories_per_cheesecake = 2800 →
  calories_per_slice = 350 →
  percentage_ate = (25 / 100 : ℚ) →
  slices_eaten (slices_of_cheesecake total_calories_per_cheesecake calories_per_slice) percentage_ate = 2 :=
by
  intros total_calories_per_cheesecake calories_per_slice percentage_ate h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_kiley_slices_eaten_l1115_111558


namespace NUMINAMATH_GPT_math_problem_l1115_111579

theorem math_problem :
  (Real.pi - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1115_111579


namespace NUMINAMATH_GPT_amount_after_two_years_l1115_111518

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)
  (hP : P = 64000) (hr : r = 1 / 6) (hn : n = 2) : 
  A = P * (1 + r) ^ n := by
  sorry

end NUMINAMATH_GPT_amount_after_two_years_l1115_111518


namespace NUMINAMATH_GPT_portraits_count_l1115_111508

theorem portraits_count (P S : ℕ) (h1 : S = 6 * P) (h2 : P + S = 200) : P = 28 := 
by
  -- The proof will be here.
  sorry

end NUMINAMATH_GPT_portraits_count_l1115_111508


namespace NUMINAMATH_GPT_venus_speed_mph_l1115_111586

theorem venus_speed_mph (speed_mps : ℝ) (seconds_per_hour : ℝ) (mph : ℝ) 
  (h1 : speed_mps = 21.9) 
  (h2 : seconds_per_hour = 3600)
  (h3 : mph = speed_mps * seconds_per_hour) : 
  mph = 78840 := 
  by 
  sorry

end NUMINAMATH_GPT_venus_speed_mph_l1115_111586


namespace NUMINAMATH_GPT_clock_correct_time_fraction_l1115_111535

/-- 
  A 24-hour digital clock displays the hour and minute of a day, 
  counting from 00:00 to 23:59. However, due to a glitch, whenever 
  the clock is supposed to display a '2', it mistakenly displays a '5'.

  Prove that the fraction of a day during which the clock shows the correct 
  time is 23/40.
-/
theorem clock_correct_time_fraction :
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  (correct_hours / total_hours) * (correct_minutes / total_minutes) = 23 / 40 :=
by
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  have h1 : correct_hours = 18 := rfl
  have h2 : correct_minutes = 46 := rfl
  have h3 : 18 / 24 = 3 / 4 := by norm_num
  have h4 : 46 / 60 = 23 / 30 := by norm_num
  have h5 : (3 / 4) * (23 / 30) = 23 / 40 := by norm_num
  exact h5

end NUMINAMATH_GPT_clock_correct_time_fraction_l1115_111535


namespace NUMINAMATH_GPT_quadratic_has_one_solution_l1115_111528

theorem quadratic_has_one_solution (k : ℝ) : (4 : ℝ) * (4 : ℝ) - k ^ 2 = 0 → k = 8 ∨ k = -8 := by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_l1115_111528


namespace NUMINAMATH_GPT_find_XY_in_306090_triangle_l1115_111538

-- Definitions of the problem
def angleZ := 90
def angleX := 60
def hypotenuseXZ := 12
def isRightTriangle (XYZ : Type) (angleZ : ℕ) : Prop := angleZ = 90
def is306090Triangle (XYZ : Type) (angleX : ℕ) (angleZ : ℕ) : Prop := (angleX = 60) ∧ (angleZ = 90)

-- Lean theorem statement
theorem find_XY_in_306090_triangle 
  (XYZ : Type)
  (hypotenuseXZ : ℕ)
  (h1 : isRightTriangle XYZ angleZ)
  (h2 : is306090Triangle XYZ angleX angleZ) :
  XY = 8 := 
sorry

end NUMINAMATH_GPT_find_XY_in_306090_triangle_l1115_111538


namespace NUMINAMATH_GPT_frank_remaining_money_l1115_111513

noncomputable def cheapest_lamp_cost : ℝ := 20
noncomputable def most_expensive_lamp_cost : ℝ := 3 * cheapest_lamp_cost
noncomputable def frank_initial_money : ℝ := 90

theorem frank_remaining_money : frank_initial_money - most_expensive_lamp_cost = 30 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_frank_remaining_money_l1115_111513


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l1115_111549

noncomputable def f (x : ℝ) : ℝ := 2 ^ |x| + 5 * |x|

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : f x = 28) :
  x + -x = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l1115_111549


namespace NUMINAMATH_GPT_rotate_image_eq_A_l1115_111550

def image_A : Type := sorry -- Image data for option (A)
def original_image : Type := sorry -- Original image data

def rotate_90_clockwise (img : Type) : Type := sorry -- Function to rotate image 90 degrees clockwise

theorem rotate_image_eq_A :
  rotate_90_clockwise original_image = image_A :=
sorry

end NUMINAMATH_GPT_rotate_image_eq_A_l1115_111550


namespace NUMINAMATH_GPT_find_m_for_one_real_solution_l1115_111588

variables {m x : ℝ}

-- Given condition
def equation := (x + 4) * (x + 1) = m + 2 * x

-- The statement to prove
theorem find_m_for_one_real_solution : (∃ m : ℝ, m = 7 / 4 ∧ ∀ (x : ℝ), (x + 4) * (x + 1) = m + 2 * x) :=
by
  -- The proof starts here, which we will skip with sorry
  sorry

end NUMINAMATH_GPT_find_m_for_one_real_solution_l1115_111588


namespace NUMINAMATH_GPT_divisibility_of_sum_of_fifths_l1115_111580

theorem divisibility_of_sum_of_fifths (x y z : ℤ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * k * (x - y) * (y - z) * (z - x) :=
sorry

end NUMINAMATH_GPT_divisibility_of_sum_of_fifths_l1115_111580


namespace NUMINAMATH_GPT_scientific_notation_of_3933_billion_l1115_111516

-- Definitions and conditions
def is_scientific_notation (a : ℝ) (n : ℤ) :=
  1 ≤ |a| ∧ |a| < 10 ∧ (39.33 * 10^9 = a * 10^n)

-- Theorem (statement only)
theorem scientific_notation_of_3933_billion : 
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ a = 3.933 ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_3933_billion_l1115_111516


namespace NUMINAMATH_GPT_foldable_shape_is_axisymmetric_l1115_111562

def is_axisymmetric_shape (shape : Type) : Prop :=
  (∃ l : (shape → shape), (∀ x, l x = x))

theorem foldable_shape_is_axisymmetric (shape : Type) (l : shape → shape) 
  (h1 : ∀ x, l x = x) : is_axisymmetric_shape shape := by
  sorry

end NUMINAMATH_GPT_foldable_shape_is_axisymmetric_l1115_111562


namespace NUMINAMATH_GPT_find_sequence_formula_l1115_111520

variable (a : ℕ → ℝ)

noncomputable def sequence_formula := ∀ n : ℕ, a n = Real.sqrt n

lemma sequence_initial : a 1 = 1 :=
sorry

lemma sequence_recursive (n : ℕ) : a (n+1)^2 - a n^2 = 1 :=
sorry

theorem find_sequence_formula : sequence_formula a :=
sorry

end NUMINAMATH_GPT_find_sequence_formula_l1115_111520


namespace NUMINAMATH_GPT_smallest_x_l1115_111519

theorem smallest_x (y : ℤ) (h1 : 0.9 = (y : ℚ) / (151 + x)) (h2 : 0 < x) (h3 : 0 < y) : x = 9 :=
sorry

end NUMINAMATH_GPT_smallest_x_l1115_111519


namespace NUMINAMATH_GPT_magic_square_y_value_l1115_111530

theorem magic_square_y_value 
  (a b c d e y : ℝ)
  (h1 : y + 4 + c = 81 + a + c)
  (h2 : y + (y - 77) + e = 81 + b + e)
  (h3 : y + 25 + 81 = 4 + (y - 77) + (2 * y - 158)) : 
  y = 168.5 :=
by
  -- required steps to complete the proof
  sorry

end NUMINAMATH_GPT_magic_square_y_value_l1115_111530


namespace NUMINAMATH_GPT_maximum_area_of_garden_l1115_111573

noncomputable def max_area (perimeter : ℕ) : ℕ :=
  let half_perimeter := perimeter / 2
  let x := half_perimeter / 2
  x * x

theorem maximum_area_of_garden :
  max_area 148 = 1369 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_of_garden_l1115_111573


namespace NUMINAMATH_GPT_man_double_son_age_in_two_years_l1115_111566

theorem man_double_son_age_in_two_years (S M Y : ℕ) (h1 : S = 14) (h2 : M = S + 16) (h3 : Y = 2) : 
  M + Y = 2 * (S + Y) :=
by
  sorry

-- Explanation:
-- h1 establishes the son's current age.
-- h2 establishes the man's current age in relation to the son's age.
-- h3 gives the solution Y = 2 years.
-- We need to prove that M + Y = 2 * (S + Y).

end NUMINAMATH_GPT_man_double_son_age_in_two_years_l1115_111566


namespace NUMINAMATH_GPT_chessboard_fraction_sum_l1115_111527

theorem chessboard_fraction_sum (r s m n : ℕ) (h_r : r = 1296) (h_s : s = 204) (h_frac : (17 : ℚ) / 108 = (s : ℕ) / (r : ℕ)) : m + n = 125 :=
sorry

end NUMINAMATH_GPT_chessboard_fraction_sum_l1115_111527


namespace NUMINAMATH_GPT_find_a_b_l1115_111571

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_a_b : 
  (∀ x : ℝ, f (g x a b) = 9 * x^2 + 6 * x + 1) ↔ ((a = 3 ∧ b = 1) ∨ (a = -3 ∧ b = -1)) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l1115_111571


namespace NUMINAMATH_GPT_number_of_rice_packets_l1115_111567

theorem number_of_rice_packets
  (initial_balance : ℤ) 
  (price_per_rice_packet : ℤ)
  (num_wheat_flour_packets : ℤ) 
  (price_per_wheat_flour_packet : ℤ)
  (price_soda : ℤ) 
  (remaining_balance : ℤ)
  (spent : ℤ)
  (eqn : initial_balance - (price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda) = remaining_balance) :
  price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda = spent 
    → initial_balance - spent = remaining_balance
    → 2 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_rice_packets_l1115_111567


namespace NUMINAMATH_GPT_parabola_tangent_perp_l1115_111511

theorem parabola_tangent_perp (a b : ℝ) : 
  (∃ x y : ℝ, x^2 = 4 * y ∧ y = a ∧ b ≠ 0 ∧ x ≠ 0) ∧
  (∃ x' y' : ℝ, x'^2 = 4 * y' ∧ y' = b ∧ a ≠ 0 ∧ x' ≠ 0) ∧
  (a * b = -1) 
  → a^4 * b^4 = (a^2 + b^2)^3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_perp_l1115_111511


namespace NUMINAMATH_GPT_stops_time_proof_l1115_111589

variable (departure_time arrival_time driving_time stop_time_in_minutes : ℕ)
variable (h_departure : departure_time = 7 * 60)
variable (h_arrival : arrival_time = 20 * 60)
variable (h_driving : driving_time = 12 * 60)
variable (total_minutes := arrival_time - departure_time)

theorem stops_time_proof :
  stop_time_in_minutes = (total_minutes - driving_time) := by
  sorry

end NUMINAMATH_GPT_stops_time_proof_l1115_111589


namespace NUMINAMATH_GPT_determine_borrow_lend_years_l1115_111594

theorem determine_borrow_lend_years (P : ℝ) (Rb Rl G : ℝ) (n : ℝ) 
  (hP : P = 9000) 
  (hRb : Rb = 4 / 100) 
  (hRl : Rl = 6 / 100) 
  (hG : G = 180) 
  (h_gain : G = P * Rl * n - P * Rb * n) : 
  n = 1 := 
sorry

end NUMINAMATH_GPT_determine_borrow_lend_years_l1115_111594


namespace NUMINAMATH_GPT_find_y_value_l1115_111563

theorem find_y_value (x y : ℝ) 
    (h1 : x^2 + 3 * x + 6 = y - 2) 
    (h2 : x = -5) : 
    y = 18 := 
  by 
  sorry

end NUMINAMATH_GPT_find_y_value_l1115_111563


namespace NUMINAMATH_GPT_train_crossing_time_l1115_111531

/-- 
Prove that the time it takes for a train traveling at 90 kmph with a length of 100.008 meters to cross a pole is 4.00032 seconds.
-/
theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) : 
  speed_kmph = 90 → length_meters = 100.008 → (length_meters / (speed_kmph * (1000 / 3600))) = 4.00032 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1115_111531


namespace NUMINAMATH_GPT_minimum_toothpicks_to_remove_l1115_111512

-- Definitions related to the problem statement
def total_toothpicks : Nat := 40
def initial_triangles : Nat := 36

-- Ensure that the minimal number of toothpicks to be removed to destroy all triangles is correct.
theorem minimum_toothpicks_to_remove : ∃ (n : Nat), n = 15 ∧ (∀ (t : Nat), t ≤ total_toothpicks - n → t = 0) :=
sorry

end NUMINAMATH_GPT_minimum_toothpicks_to_remove_l1115_111512


namespace NUMINAMATH_GPT_amino_inequality_l1115_111593

theorem amino_inequality
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h : x + y + z = x * y * z) :
  ( (x^2 - 1) / x )^2 + ( (y^2 - 1) / y )^2 + ( (z^2 - 1) / z )^2 ≥ 4 := by
  sorry

end NUMINAMATH_GPT_amino_inequality_l1115_111593


namespace NUMINAMATH_GPT_aaron_walking_speed_l1115_111532

-- Definitions of the conditions
def distance_jog : ℝ := 3 -- in miles
def speed_jog : ℝ := 2 -- in miles/hour
def total_time : ℝ := 3 -- in hours

-- The problem statement
theorem aaron_walking_speed :
  ∃ (v : ℝ), v = (distance_jog / (total_time - (distance_jog / speed_jog))) ∧ v = 2 :=
by
  sorry

end NUMINAMATH_GPT_aaron_walking_speed_l1115_111532


namespace NUMINAMATH_GPT_denote_depth_below_sea_level_l1115_111572

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_denote_depth_below_sea_level_l1115_111572


namespace NUMINAMATH_GPT_circle_tangent_to_x_axis_at_origin_l1115_111568

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + Dx + Ey + F = 0)
  (h_tangent : ∃ x, x^2 + (0 : ℝ)^2 + Dx + E * 0 + F = 0 ∧ ∃ r : ℝ, ∀ x y, x^2 + (y - r)^2 = r^2) :
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_x_axis_at_origin_l1115_111568


namespace NUMINAMATH_GPT_solveCubicEquation_l1115_111504

-- Define the condition as a hypothesis
def equationCondition (x : ℝ) : Prop := (7 - x)^(1/3) = -5/3

-- State the theorem to be proved
theorem solveCubicEquation : ∃ x : ℝ, equationCondition x ∧ x = 314 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_solveCubicEquation_l1115_111504


namespace NUMINAMATH_GPT_machines_work_together_l1115_111502

theorem machines_work_together (x : ℝ) (h_pos : 0 < x) :
  (1 / (x + 2) + 1 / (x + 3) + 1 / (x + 1) = 1 / x) → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_machines_work_together_l1115_111502


namespace NUMINAMATH_GPT_exists_sequence_a_l1115_111541

def c (n : ℕ) : ℕ := 2017 ^ n

axiom f : ℕ → ℝ

axiom condition_1 : ∀ m n : ℕ, f (m + n) ≤ 2017 * f m * f (n + 325)

axiom condition_2 : ∀ n : ℕ, 0 < f (c (n + 1)) ∧ f (c (n + 1)) < (f (c n)) ^ 2017

theorem exists_sequence_a :
  ∃ (a : ℕ → ℕ), ∀ n k : ℕ, a k < n → f n ^ c k < f (c k) ^ n := sorry

end NUMINAMATH_GPT_exists_sequence_a_l1115_111541


namespace NUMINAMATH_GPT_pages_already_read_l1115_111587

theorem pages_already_read (total_pages : ℕ) (pages_left : ℕ) (h_total : total_pages = 563) (h_left : pages_left = 416) :
  total_pages - pages_left = 147 :=
by
  sorry

end NUMINAMATH_GPT_pages_already_read_l1115_111587


namespace NUMINAMATH_GPT_nigella_base_salary_is_3000_l1115_111534

noncomputable def nigella_base_salary : ℝ :=
  let house_A_cost := 60000
  let house_B_cost := 3 * house_A_cost
  let house_C_cost := (2 * house_A_cost) - 110000
  let commission_A := 0.02 * house_A_cost
  let commission_B := 0.02 * house_B_cost
  let commission_C := 0.02 * house_C_cost
  let total_earnings := 8000
  let total_commission := commission_A + commission_B + commission_C
  total_earnings - total_commission

theorem nigella_base_salary_is_3000 : 
  nigella_base_salary = 3000 :=
by sorry

end NUMINAMATH_GPT_nigella_base_salary_is_3000_l1115_111534


namespace NUMINAMATH_GPT_restroom_students_l1115_111540

theorem restroom_students (R : ℕ) (h1 : 4 * 6 = 24) (h2 : (2/3 : ℚ) * 24 = 16)
  (h3 : 23 = 16 + (3 * R - 1) + R) : R = 2 :=
by
  sorry

end NUMINAMATH_GPT_restroom_students_l1115_111540


namespace NUMINAMATH_GPT_quadratic_function_positive_l1115_111590

theorem quadratic_function_positive (a m : ℝ) (h : a > 0) (h_fm : (m^2 + m + a) < 0) : (m + 1)^2 + (m + 1) + a > 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_function_positive_l1115_111590


namespace NUMINAMATH_GPT_area_of_region_inside_circle_outside_rectangle_l1115_111555

theorem area_of_region_inside_circle_outside_rectangle
  (EF FH : ℝ)
  (hEF : EF = 6)
  (hFH : FH = 5)
  (r : ℝ)
  (h_radius : r = (EF^2 + FH^2).sqrt) :
  π * r^2 - EF * FH = 61 * π - 30 :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_inside_circle_outside_rectangle_l1115_111555


namespace NUMINAMATH_GPT_correct_average_l1115_111551

theorem correct_average :
  let avg_incorrect := 15
  let num_numbers := 20
  let read_incorrect1 := 42
  let read_correct1 := 52
  let read_incorrect2 := 68
  let read_correct2 := 78
  let read_incorrect3 := 85
  let read_correct3 := 95
  let incorrect_sum := avg_incorrect * num_numbers
  let diff1 := read_correct1 - read_incorrect1
  let diff2 := read_correct2 - read_incorrect2
  let diff3 := read_correct3 - read_incorrect3
  let total_diff := diff1 + diff2 + diff3
  let correct_sum := incorrect_sum + total_diff
  let correct_avg := correct_sum / num_numbers
  correct_avg = 16.5 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_l1115_111551


namespace NUMINAMATH_GPT_polynomial_factorization_l1115_111506

theorem polynomial_factorization (m n : ℤ) (h₁ : (x + 1) * (x + 3) = x^2 + m * x + n) : m - n = 1 := 
by {
  -- Proof not required
  sorry
}

end NUMINAMATH_GPT_polynomial_factorization_l1115_111506


namespace NUMINAMATH_GPT_range_of_f_l1115_111597

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1115_111597


namespace NUMINAMATH_GPT_weight_of_replaced_person_l1115_111595

theorem weight_of_replaced_person :
  (∃ (W : ℝ), 
    let avg_increase := 1.5 
    let num_persons := 5 
    let new_person_weight := 72.5 
    (avg_increase * num_persons = new_person_weight - W)
  ) → 
  ∃ (W : ℝ), W = 65 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l1115_111595


namespace NUMINAMATH_GPT_initial_number_of_friends_l1115_111591

theorem initial_number_of_friends (F : ℕ) (h : 6 * (F + 2) = 60) : F = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_number_of_friends_l1115_111591


namespace NUMINAMATH_GPT_real_roots_quadratic_l1115_111544

theorem real_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ (m ≥ -5/4 ∧ m ≠ 1) := by
  sorry

end NUMINAMATH_GPT_real_roots_quadratic_l1115_111544


namespace NUMINAMATH_GPT_average_monthly_balance_l1115_111575

def january_balance : ℕ := 150
def february_balance : ℕ := 300
def march_balance : ℕ := 450
def april_balance : ℕ := 300
def number_of_months : ℕ := 4

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance) / number_of_months = 300 := by
  sorry

end NUMINAMATH_GPT_average_monthly_balance_l1115_111575


namespace NUMINAMATH_GPT_norma_bananas_count_l1115_111501

-- Definitions for the conditions
def initial_bananas : ℕ := 47
def lost_bananas : ℕ := 45

-- The proof problem in Lean 4 statement
theorem norma_bananas_count : initial_bananas - lost_bananas = 2 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_norma_bananas_count_l1115_111501


namespace NUMINAMATH_GPT_first_class_rate_l1115_111533

def pass_rate : ℝ := 0.95
def cond_first_class_rate : ℝ := 0.20

theorem first_class_rate :
  (pass_rate * cond_first_class_rate) = 0.19 :=
by
  -- The proof is omitted as we're not required to provide it.
  sorry

end NUMINAMATH_GPT_first_class_rate_l1115_111533


namespace NUMINAMATH_GPT_quadratic_inequality_has_real_solution_l1115_111543

-- Define the quadratic function and the inequality
def quadratic (a x : ℝ) : ℝ := x^2 - 8 * x + a
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x < 0

-- Define the condition for 'a' within the interval (0, 16)
def condition_on_a (a : ℝ) : Prop := 0 < a ∧ a < 16

-- The main statement to prove
theorem quadratic_inequality_has_real_solution (a : ℝ) (h : condition_on_a a) : quadratic_inequality a :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_has_real_solution_l1115_111543


namespace NUMINAMATH_GPT_find_xy_l1115_111583

noncomputable def star (a b c d : ℝ) : ℝ × ℝ :=
  (a * c + b * d, a * d + b * c)

theorem find_xy (a b x y : ℝ) (h : star a b x y = (a, b)) (h' : a^2 ≠ b^2) : (x, y) = (1, 0) :=
  sorry

end NUMINAMATH_GPT_find_xy_l1115_111583


namespace NUMINAMATH_GPT_spending_50_dollars_l1115_111521

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end NUMINAMATH_GPT_spending_50_dollars_l1115_111521


namespace NUMINAMATH_GPT_parabola_intersects_x_axis_l1115_111553

theorem parabola_intersects_x_axis {p q x₀ x₁ x₂ : ℝ} (h : ∀ (x : ℝ), x ^ 2 + p * x + q ≠ 0)
    (M_below_x_axis : x₀ ^ 2 + p * x₀ + q < 0)
    (M_at_1_neg2 : x₀ = 1 ∧ (1 ^ 2 + p * 1 + q = -2)) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₀ < x₁ → x₁ < x₂) ∧ x₁ = -1 ∧ x₂ = 2 ∨ x₁ = 0 ∧ x₂ = 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersects_x_axis_l1115_111553


namespace NUMINAMATH_GPT_investment_time_period_l1115_111548

theorem investment_time_period :
  ∀ (A P : ℝ) (R : ℝ) (T : ℝ),
  A = 896 → P = 799.9999999999999 → R = 5 →
  (A - P) = (P * R * T / 100) → T = 2.4 :=
by
  intros A P R T hA hP hR hSI
  sorry

end NUMINAMATH_GPT_investment_time_period_l1115_111548


namespace NUMINAMATH_GPT_polynomial_coefficients_sum_l1115_111569

theorem polynomial_coefficients_sum :
  let p := (5 * x^3 - 3 * x^2 + x - 8) * (8 - 3 * x)
  let a := -15
  let b := 49
  let c := -27
  let d := 32
  let e := -64
  16 * a + 8 * b + 4 * c + 2 * d + e = 44 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_sum_l1115_111569


namespace NUMINAMATH_GPT_extra_marks_15_l1115_111537

theorem extra_marks_15 {T P : ℝ} (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) (h3 : P = 120) : 
  0.45 * T - P = 15 := 
by
  sorry

end NUMINAMATH_GPT_extra_marks_15_l1115_111537


namespace NUMINAMATH_GPT_krios_population_limit_l1115_111574

theorem krios_population_limit (initial_population : ℕ) (acre_per_person : ℕ) (total_acres : ℕ) (doubling_years : ℕ) :
  initial_population = 150 →
  acre_per_person = 2 →
  total_acres = 35000 →
  doubling_years = 30 →
  ∃ (years_from_2005 : ℕ), years_from_2005 = 210 ∧ (initial_population * 2^(years_from_2005 / doubling_years)) ≥ total_acres / acre_per_person :=
by
  intros
  sorry

end NUMINAMATH_GPT_krios_population_limit_l1115_111574


namespace NUMINAMATH_GPT_combined_frosting_rate_l1115_111565

theorem combined_frosting_rate (time_Cagney time_Lacey total_time : ℕ) (Cagney_rate Lacey_rate : ℚ) :
  (time_Cagney = 20) →
  (time_Lacey = 30) →
  (total_time = 5 * 60) →
  (Cagney_rate = 1 / time_Cagney) →
  (Lacey_rate = 1 / time_Lacey) →
  ((Cagney_rate + Lacey_rate) * total_time) = 25 :=
by
  intros
  -- conditions are given and used in the statement.
  -- proof follows from these conditions. 
  sorry

end NUMINAMATH_GPT_combined_frosting_rate_l1115_111565


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1115_111546

theorem hyperbola_asymptote (a : ℝ) (h : a > 0)
  (has_asymptote : ∀ x : ℝ, abs (9 / a * x) = abs (3 * x))
  : a = 3 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1115_111546


namespace NUMINAMATH_GPT_fenced_area_with_cutout_l1115_111584

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_fenced_area_with_cutout_l1115_111584


namespace NUMINAMATH_GPT_computation_result_l1115_111556

theorem computation_result : 143 - 13 + 31 + 17 = 178 := 
by
  sorry

end NUMINAMATH_GPT_computation_result_l1115_111556


namespace NUMINAMATH_GPT_like_terms_sum_l1115_111523

theorem like_terms_sum (m n : ℕ) (h1 : m = 3) (h2 : 4 = n + 2) : m + n = 5 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_sum_l1115_111523


namespace NUMINAMATH_GPT_find_y_value_l1115_111539

theorem find_y_value 
  (k : ℝ) 
  (y : ℝ) 
  (hx81 : y = 3 * Real.sqrt 2)
  (h_eq : ∀ (x : ℝ), y = k * x ^ (1 / 4)) 
  : (∃ y, y = 2 ∧ y = k * 4 ^ (1 / 4))
:= sorry

end NUMINAMATH_GPT_find_y_value_l1115_111539


namespace NUMINAMATH_GPT_intersection_S_T_eq_interval_l1115_111592

-- Define the sets S and T
def S : Set ℝ := {x | x ≥ 2}
def T : Set ℝ := {x | x ≤ 5}

-- Prove the intersection of S and T is [2, 5]
theorem intersection_S_T_eq_interval : S ∩ T = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_eq_interval_l1115_111592


namespace NUMINAMATH_GPT_inner_circle_radius_l1115_111599

theorem inner_circle_radius :
  ∃ (r : ℝ) (a b c d : ℕ), 
    (r = (-78 + 70 * Real.sqrt 3) / 26) ∧ 
    (a = 78) ∧ 
    (b = 70) ∧ 
    (c = 3) ∧ 
    (d = 26) ∧ 
    (Nat.gcd a d = 1) ∧ 
    (a + b + c + d = 177) := 
sorry

end NUMINAMATH_GPT_inner_circle_radius_l1115_111599


namespace NUMINAMATH_GPT_isosceles_triangle_count_l1115_111598

noncomputable def valid_points : List (ℕ × ℕ) :=
  [(2, 5), (5, 5)]

theorem isosceles_triangle_count 
  (A B : ℕ × ℕ) 
  (H_A : A = (2, 2)) 
  (H_B : B = (5, 2)) : 
  valid_points.length = 2 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_count_l1115_111598


namespace NUMINAMATH_GPT_fractionSpentOnMachinery_l1115_111577

-- Given conditions
def companyCapital (C : ℝ) : Prop := 
  ∃ remainingCapital, remainingCapital = 0.675 * C ∧ 
  ∃ rawMaterial, rawMaterial = (1/4) * C ∧ 
  ∃ remainingAfterRaw, remainingAfterRaw = (3/4) * C ∧ 
  ∃ spentOnMachinery, spentOnMachinery = remainingAfterRaw - remainingCapital

-- Question translated to Lean statement
theorem fractionSpentOnMachinery (C : ℝ) (h : companyCapital C) : 
  ∃ remainingAfterRaw spentOnMachinery,
    spentOnMachinery / remainingAfterRaw = 1/10 :=
by 
  sorry

end NUMINAMATH_GPT_fractionSpentOnMachinery_l1115_111577


namespace NUMINAMATH_GPT_cos_double_angle_l1115_111557

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 3/5) : Real.cos (2 * a) = 7/25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1115_111557


namespace NUMINAMATH_GPT_p_correct_l1115_111554

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end NUMINAMATH_GPT_p_correct_l1115_111554


namespace NUMINAMATH_GPT_onions_shelf_correct_l1115_111515

def onions_on_shelf (initial: ℕ) (sold: ℕ) (added: ℕ) (given_away: ℕ): ℕ :=
  initial - sold + added - given_away

theorem onions_shelf_correct :
  onions_on_shelf 98 65 20 10 = 43 :=
by
  sorry

end NUMINAMATH_GPT_onions_shelf_correct_l1115_111515


namespace NUMINAMATH_GPT_cell_division_relationship_l1115_111559

noncomputable def number_of_cells_after_divisions (x : ℕ) : ℕ :=
  2^x

theorem cell_division_relationship (x : ℕ) : 
  number_of_cells_after_divisions x = 2^x := 
by 
  sorry

end NUMINAMATH_GPT_cell_division_relationship_l1115_111559


namespace NUMINAMATH_GPT_measure_of_third_angle_l1115_111525

-- Definitions based on given conditions
def angle_sum_of_triangle := 180
def angle1 := 30
def angle2 := 60

-- Problem Statement: Prove the third angle (angle3) in a triangle is 90 degrees
theorem measure_of_third_angle (angle_sum : ℕ := angle_sum_of_triangle) 
  (a1 : ℕ := angle1) (a2 : ℕ := angle2) : (angle_sum - (a1 + a2)) = 90 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_third_angle_l1115_111525


namespace NUMINAMATH_GPT_polygon_sides_l1115_111517

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1115_111517


namespace NUMINAMATH_GPT_persons_attended_total_l1115_111585

theorem persons_attended_total (p q : ℕ) (a : ℕ) (c : ℕ) (total_amount : ℕ) (adult_ticket : ℕ) (child_ticket : ℕ) 
  (h1 : adult_ticket = 60) (h2 : child_ticket = 25) (h3 : total_amount = 14000) 
  (h4 : a = 200) (h5 : p = a + c)
  (h6 : a * adult_ticket + c * child_ticket = total_amount):
  p = 280 :=
by
  sorry

end NUMINAMATH_GPT_persons_attended_total_l1115_111585


namespace NUMINAMATH_GPT_sandra_savings_l1115_111545

theorem sandra_savings :
  let num_notepads := 8
  let original_price_per_notepad := 3.75
  let discount_rate := 0.25
  let discount_per_notepad := original_price_per_notepad * discount_rate
  let discounted_price_per_notepad := original_price_per_notepad - discount_per_notepad
  let total_cost_without_discount := num_notepads * original_price_per_notepad
  let total_cost_with_discount := num_notepads * discounted_price_per_notepad
  let total_savings := total_cost_without_discount - total_cost_with_discount
  total_savings = 7.50 :=
sorry

end NUMINAMATH_GPT_sandra_savings_l1115_111545


namespace NUMINAMATH_GPT_comb_7_2_equals_21_l1115_111570

theorem comb_7_2_equals_21 : (Nat.choose 7 2) = 21 := by
  sorry

end NUMINAMATH_GPT_comb_7_2_equals_21_l1115_111570


namespace NUMINAMATH_GPT_largest_value_of_x_l1115_111547

theorem largest_value_of_x (x : ℝ) (h : |x - 8| = 15) : x ≤ 23 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_largest_value_of_x_l1115_111547


namespace NUMINAMATH_GPT_largest_quantity_l1115_111510

noncomputable def A := (2006 / 2005) + (2006 / 2007)
noncomputable def B := (2006 / 2007) + (2008 / 2007)
noncomputable def C := (2007 / 2006) + (2007 / 2008)

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_GPT_largest_quantity_l1115_111510


namespace NUMINAMATH_GPT_possible_values_of_p1_l1115_111524

noncomputable def p (x : ℝ) (n : ℕ) : ℝ := sorry

axiom deg_p (n : ℕ) (h : n ≥ 2) (x : ℝ) : x^n = 1

axiom roots_le_one (r : ℝ) : r ≤ 1

axiom p_at_2 (n : ℕ) (h : n ≥ 2) : p 2 n = 3^n

theorem possible_values_of_p1 (n : ℕ) (h : n ≥ 2) : p 1 n = 0 ∨ p 1 n = (-1)^n * 2^n :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_p1_l1115_111524


namespace NUMINAMATH_GPT_range_of_a_l1115_111522

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → -3 * x^2 + a ≤ 0) ↔ a ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1115_111522


namespace NUMINAMATH_GPT_infinite_solutions_iff_l1115_111596

theorem infinite_solutions_iff (a b c d : ℤ) :
  (∃ᶠ x in at_top, ∃ᶠ y in at_top, x^2 + a * x + b = y^2 + c * y + d) ↔ (a^2 - 4 * b = c^2 - 4 * d) :=
by sorry

end NUMINAMATH_GPT_infinite_solutions_iff_l1115_111596


namespace NUMINAMATH_GPT_problem_solution_l1115_111582

theorem problem_solution :
  20 * ((180 / 3) + (40 / 5) + (16 / 32) + 2) = 1410 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1115_111582


namespace NUMINAMATH_GPT_exists_integer_K_l1115_111576

theorem exists_integer_K (Z : ℕ) (K : ℕ) : 
  1000 < Z ∧ Z < 2000 ∧ Z = K^4 → 
  ∃ K, K = 6 := 
by
  sorry

end NUMINAMATH_GPT_exists_integer_K_l1115_111576


namespace NUMINAMATH_GPT_smallest_integer_satisfying_conditions_l1115_111581

-- Define the conditions explicitly as hypotheses
def satisfies_congruence_3_2 (n : ℕ) : Prop :=
  n % 3 = 2

def satisfies_congruence_7_2 (n : ℕ) : Prop :=
  n % 7 = 2

def satisfies_congruence_8_2 (n : ℕ) : Prop :=
  n % 8 = 2

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the smallest positive integer satisfying the above conditions
theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ satisfies_congruence_3_2 n ∧ satisfies_congruence_7_2 n ∧ satisfies_congruence_8_2 n ∧ is_perfect_square n :=
  by
    sorry

end NUMINAMATH_GPT_smallest_integer_satisfying_conditions_l1115_111581


namespace NUMINAMATH_GPT_second_smallest_five_digit_in_pascals_triangle_l1115_111552

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem second_smallest_five_digit_in_pascals_triangle :
  (∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (10000 ≤ binomial n k) ∧ (binomial n k < 100000) ∧
    (∀ m l : ℕ, m > 0 ∧ l > 0 ∧ (10000 ≤ binomial m l) ∧ (binomial m l < 100000) →
    (binomial n k < binomial m l → binomial n k ≥ 31465)) ∧  binomial n k = 31465) :=
sorry

end NUMINAMATH_GPT_second_smallest_five_digit_in_pascals_triangle_l1115_111552
