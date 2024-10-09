import Mathlib

namespace sum_xyz_is_sqrt_13_l1174_117458

variable (x y z : ℝ)

-- The conditions
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z

axiom eq1 : x^2 + y^2 + x * y = 3
axiom eq2 : y^2 + z^2 + y * z = 4
axiom eq3 : z^2 + x^2 + z * x = 7 

-- The theorem statement: Prove that x + y + z = sqrt(13)
theorem sum_xyz_is_sqrt_13 : x + y + z = Real.sqrt 13 :=
by
  sorry

end sum_xyz_is_sqrt_13_l1174_117458


namespace calculate_amount_l1174_117404

theorem calculate_amount (p1 p2 p3: ℝ) : 
  p1 = 0.15 * 4000 ∧ 
  p2 = p1 - 0.25 * p1 ∧ 
  p3 = 0.07 * p2 -> 
  (p3 + 0.10 * p3) = 34.65 := 
by 
  sorry

end calculate_amount_l1174_117404


namespace find_value_of_pow_function_l1174_117406

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem find_value_of_pow_function :
  (∃ α : ℝ, power_function α 4 = 1/2) →
  ∃ α : ℝ, power_function α (1/4) = 2 :=
by
  sorry

end find_value_of_pow_function_l1174_117406


namespace mod_graph_sum_l1174_117483

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end mod_graph_sum_l1174_117483


namespace hyperbola_params_l1174_117498

theorem hyperbola_params (a b h k : ℝ) (h_positivity : a > 0 ∧ b > 0)
  (asymptote_1 : ∀ x : ℝ, ∃ y : ℝ, y = (3/2) * x + 4)
  (asymptote_2 : ∀ x : ℝ, ∃ y : ℝ, y = -(3/2) * x + 2)
  (passes_through : ∃ x y : ℝ, x = 2 ∧ y = 8 ∧ (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) 
  (standard_form : ∀ x y : ℝ, ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) : 
  a + h = 7/3 := sorry

end hyperbola_params_l1174_117498


namespace smallest_positive_solution_to_congruence_l1174_117444

theorem smallest_positive_solution_to_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 33] ∧ x = 28 := 
by 
  sorry

end smallest_positive_solution_to_congruence_l1174_117444


namespace find_x_plus_2y_squared_l1174_117470

theorem find_x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2 * y) = 48) (h2 : y * (x + 2 * y) = 72) :
  (x + 2 * y) ^ 2 = 96 := 
sorry

end find_x_plus_2y_squared_l1174_117470


namespace largest_angle_in_consecutive_integer_hexagon_l1174_117412

theorem largest_angle_in_consecutive_integer_hexagon : 
  ∀ (x : ℤ), 
  (x - 3) + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 720 → 
  (x + 2 = 122) :=
by intros x h
   sorry

end largest_angle_in_consecutive_integer_hexagon_l1174_117412


namespace monkey_reaches_top_l1174_117407

def monkey_climb_time (tree_height : ℕ) (climb_per_hour : ℕ) (slip_per_hour : ℕ) 
  (rest_hours : ℕ) (cycle_hours : ℕ) : ℕ :=
  if (tree_height % (climb_per_hour - slip_per_hour) > climb_per_hour) 
    then (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours
    else (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours - 1

theorem monkey_reaches_top :
  monkey_climb_time 253 7 4 1 4 = 109 := 
sorry

end monkey_reaches_top_l1174_117407


namespace least_positive_integer_satisfying_congruences_l1174_117488

theorem least_positive_integer_satisfying_congruences :
  ∃ b : ℕ, b > 0 ∧
    (b % 6 = 5) ∧
    (b % 7 = 6) ∧
    (b % 8 = 7) ∧
    (b % 9 = 8) ∧
    ∀ n : ℕ, (n > 0 → (n % 6 = 5) ∧ (n % 7 = 6) ∧ (n % 8 = 7) ∧ (n % 9 = 8) → n ≥ b) ∧
    b = 503 :=
by
  sorry

end least_positive_integer_satisfying_congruences_l1174_117488


namespace problem_remainder_l1174_117450

theorem problem_remainder :
  ((12095 + 12097 + 12099 + 12101 + 12103 + 12105 + 12107) % 10) = 7 := by
  sorry

end problem_remainder_l1174_117450


namespace af_over_cd_is_025_l1174_117426

theorem af_over_cd_is_025
  (a b c d e f X : ℝ)
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end af_over_cd_is_025_l1174_117426


namespace smallest_multiple_of_6_and_15_l1174_117480

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ b = 30 := 
by 
  use 30 
  sorry

end smallest_multiple_of_6_and_15_l1174_117480


namespace increase_in_area_l1174_117421

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ := 2 * (length + width)
noncomputable def radius_of_circle (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)
noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * (radius ^ 2)

theorem increase_in_area :
  let rectangle_length := 60
  let rectangle_width := 20
  let rectangle_area := area_of_rectangle rectangle_length rectangle_width
  let fence_length := perimeter_of_rectangle rectangle_length rectangle_width
  let circle_radius := radius_of_circle fence_length
  let circle_area := area_of_circle circle_radius
  let area_increase := circle_area - rectangle_area
  837.99 ≤ area_increase :=
by
  sorry

end increase_in_area_l1174_117421


namespace eq_abs_piecewise_l1174_117408

theorem eq_abs_piecewise (x : ℝ) : (|x| = if x >= 0 then x else -x) :=
by
  sorry

end eq_abs_piecewise_l1174_117408


namespace length_after_5th_cut_l1174_117495

theorem length_after_5th_cut (initial_length : ℝ) (n : ℕ) (h1 : initial_length = 1) (h2 : n = 5) :
  initial_length / 2^n = 1 / 2^5 := by
  sorry

end length_after_5th_cut_l1174_117495


namespace more_visitors_that_day_l1174_117463

def number_of_visitors_previous_day : ℕ := 100
def number_of_visitors_that_day : ℕ := 666

theorem more_visitors_that_day :
  number_of_visitors_that_day - number_of_visitors_previous_day = 566 :=
by
  sorry

end more_visitors_that_day_l1174_117463


namespace point_in_third_quadrant_l1174_117473

theorem point_in_third_quadrant :
  let sin2018 := Real.sin (2018 * Real.pi / 180)
  let tan117 := Real.tan (117 * Real.pi / 180)
  sin2018 < 0 ∧ tan117 < 0 → 
  (sin2018 < 0 ∧ tan117 < 0) :=
by
  intros
  sorry

end point_in_third_quadrant_l1174_117473


namespace smallest_three_digit_multiple_of_17_l1174_117411

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l1174_117411


namespace part1_part2_l1174_117492

open Nat

variable {a : ℕ → ℝ} -- Defining the arithmetic sequence
variable {S : ℕ → ℝ} -- Defining the sum of the first n terms
variable {m n p q : ℕ} -- Defining the positive integers m, n, p, q
variable {d : ℝ} -- The common difference

-- Conditions
axiom arithmetic_sequence_pos_terms : (∀ k, a k = a 1 + (k - 1) * d) ∧ ∀ k, a k > 0
axiom sum_of_first_n_terms : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2
axiom positive_common_difference : d > 0
axiom constraints_on_mnpq : n < p ∧ q < m ∧ m + n = p + q

-- Parts to prove
theorem part1 : a m * a n < a p * a q :=
by sorry

theorem part2 : S m + S n > S p + S q :=
by sorry

end part1_part2_l1174_117492


namespace roots_inverse_sum_eq_two_thirds_l1174_117467

theorem roots_inverse_sum_eq_two_thirds {x₁ x₂ : ℝ} (h1 : x₁ ^ 2 + 2 * x₁ - 3 = 0) (h2 : x₂ ^ 2 + 2 * x₂ - 3 = 0) : 
  (1 / x₁) + (1 / x₂) = 2 / 3 :=
sorry

end roots_inverse_sum_eq_two_thirds_l1174_117467


namespace snow_total_inches_l1174_117420

theorem snow_total_inches (initial_snow_ft : ℝ) (additional_snow_in : ℝ)
  (melted_snow_in : ℝ) (multiplier : ℝ) (days_after : ℕ) (conversion_rate : ℝ)
  (initial_snow_in : ℝ) (fifth_day_snow_in : ℝ) :
  initial_snow_ft = 0.5 →
  additional_snow_in = 8 →
  melted_snow_in = 2 →
  multiplier = 2 →
  days_after = 5 →
  conversion_rate = 12 →
  initial_snow_in = initial_snow_ft * conversion_rate →
  fifth_day_snow_in = multiplier * initial_snow_in →
  (initial_snow_in + additional_snow_in - melted_snow_in + fifth_day_snow_in) / conversion_rate = 2 :=
by
  sorry

end snow_total_inches_l1174_117420


namespace cost_of_article_l1174_117448

noncomputable def find_cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : Prop :=
  C = 168.57

theorem cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : 
  find_cost_of_article C G h1 h2 :=
by
  sorry

end cost_of_article_l1174_117448


namespace sum_even_integers_between_200_and_600_is_80200_l1174_117485

noncomputable def sum_even_integers_between_200_and_600 (a d n : ℕ) : ℕ :=
  n / 2 * (a + (a + (n - 1) * d))

theorem sum_even_integers_between_200_and_600_is_80200 :
  sum_even_integers_between_200_and_600 202 2 200 = 80200 :=
by
  -- proof would go here
  sorry

end sum_even_integers_between_200_and_600_is_80200_l1174_117485


namespace student_scores_marks_per_correct_answer_l1174_117476

theorem student_scores_marks_per_correct_answer
  (total_questions : ℕ) (total_marks : ℤ) (correct_questions : ℕ)
  (wrong_questions : ℕ) (marks_wrong_answer : ℤ)
  (x : ℤ) (h1 : total_questions = 60) (h2 : total_marks = 110)
  (h3 : correct_questions = 34) (h4 : wrong_questions = total_questions - correct_questions)
  (h5 : marks_wrong_answer = -1) :
  34 * x - 26 = 110 → x = 4 := by
  sorry

end student_scores_marks_per_correct_answer_l1174_117476


namespace total_cost_of_items_l1174_117460

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_items_l1174_117460


namespace find_k_value_l1174_117419

noncomputable def arithmetic_seq (a d : ℤ) : ℕ → ℤ
| n => a + (n - 1) * d

theorem find_k_value (a d : ℤ) (k : ℕ) 
  (h1 : arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 24)
  (h2 : (Finset.range 11).sum (λ i => arithmetic_seq a d (5 + i)) = 110)
  (h3 : arithmetic_seq a d k = 16) : 
  k = 16 :=
sorry

end find_k_value_l1174_117419


namespace inequality_range_of_a_l1174_117487

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2: ℝ) 2 :=
by
  sorry

end inequality_range_of_a_l1174_117487


namespace fractional_sum_equals_015025_l1174_117471

theorem fractional_sum_equals_015025 :
  (2 / 20) + (8 / 200) + (3 / 300) + (5 / 40000) * 2 = 0.15025 := 
by
  sorry

end fractional_sum_equals_015025_l1174_117471


namespace intersection_A_B_subset_A_B_l1174_117415

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
noncomputable def set_B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 22}

theorem intersection_A_B (a : ℝ) (ha : a = 10) : set_A a ∩ set_B = {x : ℝ | 21 ≤ x ∧ x ≤ 22} := by
  sorry

theorem subset_A_B (a : ℝ) : set_A a ⊆ set_B → a ≤ 9 := by
  sorry

end intersection_A_B_subset_A_B_l1174_117415


namespace even_function_m_value_l1174_117481

theorem even_function_m_value {m : ℤ} (h : ∀ (x : ℝ), (m^2 - m - 1) * (-x)^m = (m^2 - m - 1) * x^m) : m = 2 := 
by
  sorry

end even_function_m_value_l1174_117481


namespace distinct_zeros_arithmetic_geometric_sequence_l1174_117494

theorem distinct_zeros_arithmetic_geometric_sequence 
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : a + b = p)
  (h3 : ab = q)
  (h4 : p > 0)
  (h5 : q > 0)
  (h6 : (a = 4 ∧ b = 1) ∨ (a = 1 ∧ b = 4))
  : p + q = 9 := 
sorry

end distinct_zeros_arithmetic_geometric_sequence_l1174_117494


namespace probability_recurrence_relation_l1174_117454

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l1174_117454


namespace slope_parallel_l1174_117425

theorem slope_parallel {x y : ℝ} (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = -1/2 ∧ ( ∀ (x1 x2 : ℝ), 3 * x1 - 6 * y = 15 → ∃ y1 : ℝ, y1 = m * x1) :=
by
  sorry

end slope_parallel_l1174_117425


namespace real_root_exists_l1174_117478

theorem real_root_exists (p1 p2 q1 q2 : ℝ) 
(h : p1 * p2 = 2 * (q1 + q2)) : 
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  sorry

end real_root_exists_l1174_117478


namespace final_remaining_money_l1174_117417

-- Define conditions as given in the problem
def monthly_income : ℕ := 2500
def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50
def expenses_total : ℕ := rent + car_payment + utilities + groceries
def remaining_money : ℕ := monthly_income - expenses_total
def retirement_contribution : ℕ := remaining_money / 2

-- State the theorem to be proven
theorem final_remaining_money : (remaining_money - retirement_contribution) = 650 := by
  sorry

end final_remaining_money_l1174_117417


namespace centroid_quad_area_correct_l1174_117433

noncomputable def centroid_quadrilateral_area (E F G H Q : ℝ × ℝ) (side_length : ℝ) (EQ FQ : ℝ) : ℝ :=
  if h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35 then
    12800 / 9
  else
    sorry

theorem centroid_quad_area_correct (E F G H Q : ℝ × ℝ) (side_length EQ FQ : ℝ) 
  (h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35) :
  centroid_quadrilateral_area E F G H Q side_length EQ FQ = 12800 / 9 :=
sorry

end centroid_quad_area_correct_l1174_117433


namespace find_pairs_eq_l1174_117401

theorem find_pairs_eq : 
  { (m, n) : ℕ × ℕ | 0 < m ∧ 0 < n ∧ m ^ 2 + 2 * n ^ 2 = 3 * (m + 2 * n) } = {(3, 3), (4, 2)} :=
by sorry

end find_pairs_eq_l1174_117401


namespace sum_of_arithmetic_sequence_l1174_117486

noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a1 : ℤ) (d : ℤ)
  (h1 : a1 = -2010)
  (h2 : (S 2011 a1 d) / 2011 - (S 2009 a1 d) / 2009 = 2) :
  S 2010 a1 d = -2010 := 
sorry

end sum_of_arithmetic_sequence_l1174_117486


namespace geometric_series_first_term_l1174_117443

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_l1174_117443


namespace extra_yellow_balls_dispatched_eq_49_l1174_117491

-- Define the given conditions
def ordered_balls : ℕ := 114
def white_balls : ℕ := ordered_balls / 2
def yellow_balls := ordered_balls / 2

-- Define the additional yellow balls dispatched and the ratio condition
def dispatch_error_ratio : ℚ := 8 / 15

-- The statement to prove the number of extra yellow balls dispatched
theorem extra_yellow_balls_dispatched_eq_49
  (ordered_balls_rounded : ordered_balls = 114)
  (white_balls_57 : white_balls = 57)
  (yellow_balls_57 : yellow_balls = 57)
  (ratio_condition : white_balls / (yellow_balls + x) = dispatch_error_ratio) :
  x = 49 :=
  sorry

end extra_yellow_balls_dispatched_eq_49_l1174_117491


namespace prop_A_prop_B_prop_C_prop_D_l1174_117447

variable {a b : ℝ}

-- Proposition A
theorem prop_A (h : a^2 - b^2 = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b < 1 := sorry

-- Proposition B (negation of the original proposition since B is incorrect)
theorem prop_B (h : (1 / b) - (1 / a) = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b ≥ 1 := sorry

-- Proposition C
theorem prop_C (h : a > b + 1) (a_pos : 0 < a) (b_pos : 0 < b) : a^2 > b^2 + 1 := sorry

-- Proposition D (negation of the original proposition since D is incorrect)
theorem prop_D (h1 : a ≤ 1) (h2 : b ≤ 1) (a_pos : 0 < a) (b_pos : 0 < b) : |a - b| < |1 - a * b| := sorry

end prop_A_prop_B_prop_C_prop_D_l1174_117447


namespace irrational_sqrt_10_l1174_117453

theorem irrational_sqrt_10 : Irrational (Real.sqrt 10) :=
sorry

end irrational_sqrt_10_l1174_117453


namespace time_to_see_each_other_again_l1174_117422

variable (t : ℝ) (t_frac : ℚ)
variable (kenny_speed jenny_speed : ℝ)
variable (kenny_initial jenny_initial : ℝ)
variable (building_side distance_between_paths : ℝ)

def kenny_position (t : ℝ) : ℝ := kenny_initial + kenny_speed * t
def jenny_position (t : ℝ) : ℝ := jenny_initial + jenny_speed * t

theorem time_to_see_each_other_again
  (kenny_speed_eq : kenny_speed = 4)
  (jenny_speed_eq : jenny_speed = 2)
  (kenny_initial_eq : kenny_initial = -50)
  (jenny_initial_eq : jenny_initial = -50)
  (building_side_eq : building_side = 100)
  (distance_between_paths_eq : distance_between_paths = 300)
  (t_gt_50 : t > 50)
  (t_frac_eq : t_frac = 50) :
  (t == t_frac) :=
  sorry

end time_to_see_each_other_again_l1174_117422


namespace total_driving_time_is_40_l1174_117413

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end total_driving_time_is_40_l1174_117413


namespace spent_on_music_l1174_117475

variable (total_allowance : ℝ) (fraction_music : ℝ)

-- Assuming the conditions
def conditions : Prop :=
  total_allowance = 50 ∧ fraction_music = 3 / 10

-- The proof problem
theorem spent_on_music (h : conditions total_allowance fraction_music) : 
  total_allowance * fraction_music = 15 := by
  cases h with
  | intro h_total h_fraction =>
  sorry

end spent_on_music_l1174_117475


namespace square_area_and_diagonal_ratio_l1174_117477

theorem square_area_and_diagonal_ratio
    (a b : ℕ)
    (h_perimeter : 4 * a = 16 * b) :
    (a = 4 * b) ∧ ((a^2) / (b^2) = 16) ∧ ((a * Real.sqrt 2) / (b * Real.sqrt 2) = 4) :=
  by
  sorry

end square_area_and_diagonal_ratio_l1174_117477


namespace NewYearSeasonMarkup_theorem_l1174_117497

def NewYearSeasonMarkup (C N : ℝ) : Prop :=
    (0.90 * (1.20 * C * (1 + N)) = 1.35 * C) -> N = 0.25

theorem NewYearSeasonMarkup_theorem (C : ℝ) (h₀ : C > 0) : ∃ (N : ℝ), NewYearSeasonMarkup C N :=
by
  use 0.25
  sorry

end NewYearSeasonMarkup_theorem_l1174_117497


namespace geometric_sum_eqn_l1174_117409

theorem geometric_sum_eqn 
  (a1 q : ℝ) 
  (hne1 : q ≠ 1) 
  (hS2 : a1 * (1 - q^2) / (1 - q) = 1) 
  (hS4 : a1 * (1 - q^4) / (1 - q) = 3) :
  a1 * (1 - q^8) / (1 - q) = 15 :=
by
  sorry

end geometric_sum_eqn_l1174_117409


namespace simplify_expression_l1174_117493

theorem simplify_expression:
  (a = 2) ∧ (b = 1) →
  - (1 / 3 : ℚ) * (a^3 * b - a * b) 
  + a * b^3 
  - (a * b - b) / 2 
  - b / 2 
  + (1 / 3 : ℚ) * (a^3 * b) 
  = (5 / 3 : ℚ) := by 
  intros h
  simp [h.1, h.2]
  sorry

end simplify_expression_l1174_117493


namespace largest_two_digit_number_l1174_117490

-- Define the conditions and the theorem to be proven
theorem largest_two_digit_number (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 4) ∧ (10 ≤ n) ∧ (n < 100) → n = 84 := by
  sorry

end largest_two_digit_number_l1174_117490


namespace jewel_price_reduction_l1174_117466

theorem jewel_price_reduction (P x : ℝ) (P1 : ℝ) (hx : x ≠ 0) 
  (hP1 : P1 = P * (1 - (x / 100) ^ 2))
  (h_final : P1 * (1 - (x / 100) ^ 2) = 2304) : 
  P1 = 2304 / (1 - (x / 100) ^ 2) :=
by
  sorry

end jewel_price_reduction_l1174_117466


namespace longer_piece_length_l1174_117455

-- Conditions
def total_length : ℤ := 69
def is_cuts_into_two_pieces (a b : ℤ) : Prop := a + b = total_length
def is_twice_the_length (a b : ℤ) : Prop := a = 2 * b

-- Question: What is the length of the longer piece?
theorem longer_piece_length
  (a b : ℤ) 
  (H1: is_cuts_into_two_pieces a b)
  (H2: is_twice_the_length a b) :
  a = 46 :=
sorry

end longer_piece_length_l1174_117455


namespace determine_b_l1174_117430

theorem determine_b (a b : ℤ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 :=
by
  sorry

end determine_b_l1174_117430


namespace sum_of_coordinates_of_other_endpoint_l1174_117441

theorem sum_of_coordinates_of_other_endpoint :
  ∀ (x y : ℤ), (7, -15) = ((x + 3) / 2, (y - 5) / 2) → x + y = -14 :=
by
  intros x y h
  sorry

end sum_of_coordinates_of_other_endpoint_l1174_117441


namespace solve_variable_expression_l1174_117438

variable {x y : ℕ}

theorem solve_variable_expression
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (7 * x + 5 * y) / (x - 2 * y) = 26) :
  x = 3 * y :=
sorry

end solve_variable_expression_l1174_117438


namespace axis_of_symmetry_condition_l1174_117435

theorem axis_of_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
    (h_sym : ∀ x y, y = -x → y = (p * x + q) / (r * x + s)) : p = s :=
by
  sorry

end axis_of_symmetry_condition_l1174_117435


namespace books_leftover_l1174_117464

theorem books_leftover :
  (1500 * 45) % 47 = 13 :=
by
  sorry

end books_leftover_l1174_117464


namespace vector_at_t_neg3_l1174_117402

theorem vector_at_t_neg3 :
  let a := (2, 3)
  let b := (12, -37)
  let d := ((b.1 - a.1) / 5, (b.2 - a.2) / 5)
  let line_param (t : ℝ) := (a.1 + t * d.1, a.2 + t * d.2)
  line_param (-3) = (-4, 27) := by
  -- Proof goes here
  sorry

end vector_at_t_neg3_l1174_117402


namespace ice_cream_eaten_on_friday_l1174_117465

theorem ice_cream_eaten_on_friday
  (x : ℝ) -- the amount eaten on Friday night
  (saturday_night : ℝ) -- the amount eaten on Saturday night
  (total : ℝ) -- the total amount eaten
  
  (h1 : saturday_night = 0.25)
  (h2 : total = 3.5)
  (h3 : x + saturday_night = total) : x = 3.25 :=
by
  sorry

end ice_cream_eaten_on_friday_l1174_117465


namespace non_subset_condition_l1174_117482

theorem non_subset_condition (M P : Set α) (non_empty : M ≠ ∅) : 
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := 
sorry

end non_subset_condition_l1174_117482


namespace polynomial_factorization_l1174_117474

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l1174_117474


namespace factorization_correct_l1174_117484

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end factorization_correct_l1174_117484


namespace smallest_nuts_in_bag_l1174_117416

theorem smallest_nuts_in_bag :
  ∃ (N : ℕ), N ≡ 1 [MOD 11] ∧ N ≡ 8 [MOD 13] ∧ N ≡ 3 [MOD 17] ∧
             (∀ M, (M ≡ 1 [MOD 11] ∧ M ≡ 8 [MOD 13] ∧ M ≡ 3 [MOD 17]) → M ≥ N) :=
sorry

end smallest_nuts_in_bag_l1174_117416


namespace algebra_ineq_example_l1174_117418

theorem algebra_ineq_example (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ( ( (xy + yz + zx) / 3 ) ^ (1/3) ) :=
by
  sorry

end algebra_ineq_example_l1174_117418


namespace log_a_properties_l1174_117461

noncomputable def log_a (a x : ℝ) (h : 0 < a ∧ a < 1) : ℝ := Real.log x / Real.log a

theorem log_a_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 1 < x → log_a a x h < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → log_a a x h > 0) ∧
  (¬ ∀ x1 x2 : ℝ, log_a a x1 h > log_a a x2 h → x1 > x2) ∧
  (∀ x y : ℝ, log_a a (x * y) h = log_a a x h + log_a a y h) :=
by
  sorry

end log_a_properties_l1174_117461


namespace man_work_rate_l1174_117405

theorem man_work_rate (W : ℝ) (M S : ℝ)
  (h1 : (M + S) * 3 = W)
  (h2 : S * 5.25 = W) :
  M * 7 = W :=
by 
-- The proof steps will be filled in here.
sorry

end man_work_rate_l1174_117405


namespace rows_of_pies_l1174_117499

theorem rows_of_pies (baked_pecan_pies : ℕ) (baked_apple_pies : ℕ) (pies_per_row : ℕ) : 
  baked_pecan_pies = 16 ∧ baked_apple_pies = 14 ∧ pies_per_row = 5 → 
  (baked_pecan_pies + baked_apple_pies) / pies_per_row = 6 :=
by
  sorry

end rows_of_pies_l1174_117499


namespace question_1_question_2_question_3_question_4_l1174_117479

-- Define each condition as a theorem
theorem question_1 (explanation: String) : explanation = "providing for the living" :=
  sorry

theorem question_2 (usage: String) : usage = "structural auxiliary word, placed between subject and predicate, negating sentence independence" :=
  sorry

theorem question_3 (explanation: String) : explanation = "The Shang dynasty called it 'Xu,' and the Zhou dynasty called it 'Xiang.'" :=
  sorry

theorem question_4 (analysis: String) : analysis = "The statement about the 'ultimate ideal' is incorrect; the original text states that 'enabling people to live and die without regret' is 'the beginning of the King's Way.'" :=
  sorry

end question_1_question_2_question_3_question_4_l1174_117479


namespace min_CD_squared_diff_l1174_117432

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 2)) + (Real.sqrt (z + 2))
noncomputable def f (x y z : ℝ) : ℝ := (C x y z) ^ 2 - (D x y z) ^ 2

theorem min_CD_squared_diff (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  f x y z ≥ 41.4736 :=
sorry

end min_CD_squared_diff_l1174_117432


namespace arithmetic_sequence_1001th_term_l1174_117428

theorem arithmetic_sequence_1001th_term (p q : ℚ)
    (h1 : p + 3 * q = 12)
    (h2 : 12 + 3 * q = 3 * p - q) :
    (p + (1001 - 1) * (3 * q) = 5545) :=
by
  sorry

end arithmetic_sequence_1001th_term_l1174_117428


namespace spherical_to_rectangular_example_l1174_117449

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 5 (3 * Real.pi / 2) (Real.pi / 3) = (0, -5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  simp [spherical_to_rectangular, Real.sin, Real.cos]
  sorry

end spherical_to_rectangular_example_l1174_117449


namespace product_in_base_7_l1174_117459

def base_7_product : ℕ :=
  let b := 7
  Nat.ofDigits b [3, 5, 6] * Nat.ofDigits b [4]

theorem product_in_base_7 :
  base_7_product = Nat.ofDigits 7 [3, 2, 3, 1, 2] :=
by
  -- The proof is formally skipped for this exercise, hence we insert 'sorry'.
  sorry

end product_in_base_7_l1174_117459


namespace midpoint_product_coordinates_l1174_117489

theorem midpoint_product_coordinates :
  ∃ (x y : ℝ), (4 : ℝ) = (-2 + x) / 2 ∧ (-3 : ℝ) = (-7 + y) / 2 ∧ x * y = 10 := by
  sorry

end midpoint_product_coordinates_l1174_117489


namespace total_players_correct_l1174_117403

-- Define the number of players for each type of sport
def cricket_players : Nat := 12
def hockey_players : Nat := 17
def football_players : Nat := 11
def softball_players : Nat := 10

-- The theorem we aim to prove
theorem total_players_correct : 
  cricket_players + hockey_players + football_players + softball_players = 50 := by
  sorry

end total_players_correct_l1174_117403


namespace exercise_l1174_117442

theorem exercise (x y z : ℕ) (h1 : x * y * z = 1) : (7 ^ ((x + y + z) ^ 3) / 7 ^ ((x - y + z) ^ 3)) = 7 ^ 6 := 
by
  sorry

end exercise_l1174_117442


namespace ten_percent_markup_and_markdown_l1174_117472

theorem ten_percent_markup_and_markdown (x : ℝ) (hx : x > 0) : 0.99 * x < x :=
by 
  sorry

end ten_percent_markup_and_markdown_l1174_117472


namespace common_difference_divisible_by_6_l1174_117469

theorem common_difference_divisible_by_6 (p q r d : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp3 : p > 3) (hq3 : q > 3) (hr3 : r > 3) (h1 : q = p + d) (h2 : r = p + 2 * d) : d % 6 = 0 := 
sorry

end common_difference_divisible_by_6_l1174_117469


namespace compute_difference_of_squares_l1174_117437

theorem compute_difference_of_squares :
  let a := 23
  let b := 12
  (a + b) ^ 2 - (a - b) ^ 2 = 1104 := by
sorry

end compute_difference_of_squares_l1174_117437


namespace right_triangle_area_l1174_117451

theorem right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) 
  (h_angle_sum : a = 45) (h_other_angle : b = 45) (h_right_angle : c = 90)
  (h_altitude : ∃ height : ℝ, height = 4) :
  ∃ area : ℝ, area = 8 := 
by
  sorry

end right_triangle_area_l1174_117451


namespace find_value_of_6b_l1174_117457

theorem find_value_of_6b (a b : ℝ) (h1 : 10 * a = 20) (h2 : 120 * a * b = 800) : 6 * b = 20 :=
by
  sorry

end find_value_of_6b_l1174_117457


namespace youngest_person_age_l1174_117496

noncomputable def avg_age_seven_people := 30
noncomputable def avg_age_six_people_when_youngest_born := 25
noncomputable def num_people := 7
noncomputable def num_people_minus_one := 6

theorem youngest_person_age :
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  total_age_seven_people - total_age_six_people = 60 :=
by
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  sorry

end youngest_person_age_l1174_117496


namespace original_number_is_17_l1174_117462

-- Function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  (ones * 10) + tens

-- Problem statement
theorem original_number_is_17 (x : ℕ) (h1 : reverse_digits (2 * x) + 2 = 45) : x = 17 :=
by
  sorry

end original_number_is_17_l1174_117462


namespace total_cost_eq_4800_l1174_117429

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end total_cost_eq_4800_l1174_117429


namespace green_beans_to_onions_ratio_l1174_117446

def cut_conditions
  (potatoes : ℕ)
  (carrots : ℕ)
  (onions : ℕ)
  (green_beans : ℕ) : Prop :=
  carrots = 6 * potatoes ∧ onions = 2 * carrots ∧ potatoes = 2 ∧ green_beans = 8

theorem green_beans_to_onions_ratio (potatoes carrots onions green_beans : ℕ) :
  cut_conditions potatoes carrots onions green_beans →
  green_beans / gcd green_beans onions = 1 ∧ onions / gcd green_beans onions = 3 :=
by
  sorry

end green_beans_to_onions_ratio_l1174_117446


namespace double_people_half_work_l1174_117434

-- Definitions
def initial_person_count (P : ℕ) : Prop := true
def initial_time (T : ℕ) : Prop := T = 16

-- Theorem
theorem double_people_half_work (P T : ℕ) (hP : initial_person_count P) (hT : initial_time T) : P > 0 → (2 * P) * (T / 2) = P * T / 2 := by
  sorry

end double_people_half_work_l1174_117434


namespace sum_of_three_largest_l1174_117439

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l1174_117439


namespace math_problem_l1174_117452

theorem math_problem
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2006)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2007)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2006)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2007)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2006)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2007)
  : (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = -1 / 2006 := by
  sorry

end math_problem_l1174_117452


namespace team_A_days_additional_people_l1174_117445

theorem team_A_days (x : ℕ) (y : ℕ)
  (h1 : 2700 / x = 2 * (1800 / y))
  (h2 : y = x + 1)
  : x = 3 ∧ y = 4 :=
by
  sorry

theorem additional_people (m : ℕ)
  (h1 : (200 : ℝ) * 10 * 3 + 150 * 8 * 4 = 10800)
  (h2 : (170 : ℝ) * (10 + m) * 3 + 150 * 8 * 4 = 1.20 * 10800)
  : m = 6 :=
by
  sorry

end team_A_days_additional_people_l1174_117445


namespace find_a3_plus_a5_l1174_117431

-- Define an arithmetic-geometric sequence
def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, 0 < r ∧ ∃ b : ℝ, a n = b * r ^ n

-- Define the given condition
def given_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

-- Define the target theorem statement
theorem find_a3_plus_a5 (a : ℕ → ℝ) 
  (pos_sequence : is_arithmetic_geometric a) 
  (cond : given_condition a) : 
  a 3 + a 5 = 5 :=
sorry

end find_a3_plus_a5_l1174_117431


namespace charlotte_one_way_journey_time_l1174_117423

def charlotte_distance : ℕ := 60
def charlotte_speed : ℕ := 10

theorem charlotte_one_way_journey_time :
  charlotte_distance / charlotte_speed = 6 :=
by
  sorry

end charlotte_one_way_journey_time_l1174_117423


namespace max_brownies_l1174_117400

-- Definitions for the conditions given in the problem
def is_interior_pieces (m n : ℕ) : ℕ := (m - 2) * (n - 2)
def is_perimeter_pieces (m n : ℕ) : ℕ := 2 * m + 2 * n - 4

-- The assertion that the number of brownies along the perimeter is twice the number in the interior
def condition (m n : ℕ) : Prop := 2 * is_interior_pieces m n = is_perimeter_pieces m n

-- The statement that the maximum number of brownies under the given condition is 84
theorem max_brownies : ∃ (m n : ℕ), condition m n ∧ m * n = 84 := by
  sorry

end max_brownies_l1174_117400


namespace polynomial_division_l1174_117436

noncomputable def polynomial_div_quotient (p q : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.divByMonic p q

theorem polynomial_division 
  (p q : Polynomial ℚ)
  (hq : q = Polynomial.C 3 * Polynomial.X - Polynomial.C 4)
  (hp : p = 10 * Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 8 * Polynomial.X - 9) :
  polynomial_div_quotient p q = (10 / 3) * Polynomial.X ^ 2 - (55 / 9) * Polynomial.X - (172 / 27) :=
by
  sorry

end polynomial_division_l1174_117436


namespace balls_sold_l1174_117468

theorem balls_sold (CP SP_total : ℕ) (loss : ℕ) (n : ℕ) :
  CP = 60 →
  SP_total = 720 →
  loss = 5 * CP →
  loss = n * CP - SP_total →
  n = 17 :=
by
  intros hCP hSP_total hloss htotal
  -- Your proof here
  sorry

end balls_sold_l1174_117468


namespace problem_l1174_117424

variable (x : ℝ)

theorem problem (A B : ℝ) 
  (h : (A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3))): 
  A + B = 15 := by
  sorry

end problem_l1174_117424


namespace market_value_of_stock_l1174_117440

-- Define the given conditions.
def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.09 * face_value
def yield : ℝ := 0.08

-- State the problem: proving the market value of the stock.
theorem market_value_of_stock : (dividend_per_share / yield) * 100 = 112.50 := by
  -- Placeholder for the proof
  sorry

end market_value_of_stock_l1174_117440


namespace cube_root_of_neg_eight_l1174_117414

theorem cube_root_of_neg_eight : ∃ x : ℝ, x ^ 3 = -8 ∧ x = -2 := by 
  sorry

end cube_root_of_neg_eight_l1174_117414


namespace end_same_digit_l1174_117427

theorem end_same_digit
  (a b : ℕ)
  (h : (2 * a + b) % 10 = (2 * b + a) % 10) :
  a % 10 = b % 10 :=
by
  sorry

end end_same_digit_l1174_117427


namespace math_problem_l1174_117410

variable (a b c : ℝ)

variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a ≠ -b) (h5 : b ≠ -c) (h6 : c ≠ -a)

theorem math_problem 
    (h₁ : (a * b) / (a + b) = 4)
    (h₂ : (b * c) / (b + c) = 5)
    (h₃ : (c * a) / (c + a) = 7) :
    (a * b * c) / (a * b + b * c + c * a) = 280 / 83 := 
sorry

end math_problem_l1174_117410


namespace inscribed_square_side_length_l1174_117456

theorem inscribed_square_side_length (a h : ℝ) (ha_pos : 0 < a) (hh_pos : 0 < h) :
    ∃ x : ℝ, x = (h * a) / (a + h) :=
by
  -- Code here demonstrates the existence of x such that x = (h * a) / (a + h)
  sorry

end inscribed_square_side_length_l1174_117456
