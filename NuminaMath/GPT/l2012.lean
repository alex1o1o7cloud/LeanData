import Mathlib

namespace NUMINAMATH_GPT_cannot_use_diff_of_squares_l2012_201244

def diff_of_squares (a b : ℤ) : ℤ := a^2 - b^2

theorem cannot_use_diff_of_squares (x y : ℤ) : 
  ¬ ( ((-x + y) * (x - y)) = diff_of_squares (x - y) (0) ) :=
by {
  sorry
}

end NUMINAMATH_GPT_cannot_use_diff_of_squares_l2012_201244


namespace NUMINAMATH_GPT_income_percent_greater_l2012_201264

variable (A B : ℝ)

-- Condition: A's income is 25% less than B's income
def income_condition (A B : ℝ) : Prop :=
  A = 0.75 * B

-- Statement: B's income is 33.33% greater than A's income
theorem income_percent_greater (A B : ℝ) (h : income_condition A B) :
  B = A * (4 / 3) := by
sorry

end NUMINAMATH_GPT_income_percent_greater_l2012_201264


namespace NUMINAMATH_GPT_divisor_is_five_l2012_201260

theorem divisor_is_five (n d : ℕ) (h1 : ∃ k, n = k * d + 3) (h2 : ∃ l, n^2 = l * d + 4) : d = 5 :=
sorry

end NUMINAMATH_GPT_divisor_is_five_l2012_201260


namespace NUMINAMATH_GPT_sum_ages_l2012_201252

variables (uncle_age eunji_age yuna_age : ℕ)

def EunjiAge (uncle_age : ℕ) := uncle_age - 25
def YunaAge (eunji_age : ℕ) := eunji_age + 3

theorem sum_ages (h_uncle : uncle_age = 41) (h_eunji : EunjiAge uncle_age = eunji_age) (h_yuna : YunaAge eunji_age = yuna_age) :
  eunji_age + yuna_age = 35 :=
sorry

end NUMINAMATH_GPT_sum_ages_l2012_201252


namespace NUMINAMATH_GPT_range_of_m_l2012_201226

noncomputable def set_A (x : ℝ) : ℝ := x^2 - (3 / 2) * x + 1

def A : Set ℝ := {y | ∃ (x : ℝ), x ∈ (Set.Icc (-1/2 : ℝ) 2) ∧ y = set_A x}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m + 1 ∨ x ≤ m - 1}

def sufficient_condition (m : ℝ) : Prop := A ⊆ B m

theorem range_of_m :
  {m : ℝ | sufficient_condition m} = {m | m ≤ -(9 / 16) ∨ m ≥ 3} :=
sorry

end NUMINAMATH_GPT_range_of_m_l2012_201226


namespace NUMINAMATH_GPT_benny_gave_seashells_l2012_201243

theorem benny_gave_seashells (original_seashells : ℕ) (remaining_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 66) 
  (h2 : remaining_seashells = 14) 
  (h3 : original_seashells - remaining_seashells = given_seashells) : 
  given_seashells = 52 := 
by
  sorry

end NUMINAMATH_GPT_benny_gave_seashells_l2012_201243


namespace NUMINAMATH_GPT_percentage_of_alcohol_in_second_vessel_l2012_201257

-- Define the problem conditions
def capacity1 : ℝ := 2
def percentage1 : ℝ := 0.35
def alcohol1 := capacity1 * percentage1

def capacity2 : ℝ := 6 
def percentage2 (x : ℝ) : ℝ := 0.01 * x
def alcohol2 (x : ℝ) := capacity2 * percentage2 x

def total_capacity : ℝ := 8
def final_percentage : ℝ := 0.37
def total_alcohol := total_capacity * final_percentage

theorem percentage_of_alcohol_in_second_vessel (x : ℝ) :
  alcohol1 + alcohol2 x = total_alcohol → x = 37.67 :=
by sorry

end NUMINAMATH_GPT_percentage_of_alcohol_in_second_vessel_l2012_201257


namespace NUMINAMATH_GPT_andrew_worked_days_l2012_201251

-- Definitions per given conditions
def vacation_days_per_work_days (W : ℕ) : ℕ := W / 10
def days_taken_off_in_march := 5
def days_taken_off_in_september := 2 * days_taken_off_in_march
def total_days_off_taken := days_taken_off_in_march + days_taken_off_in_september
def remaining_vacation_days := 15
def total_vacation_days := total_days_off_taken + remaining_vacation_days

theorem andrew_worked_days (W : ℕ) :
  vacation_days_per_work_days W = total_vacation_days → W = 300 := by
  sorry

end NUMINAMATH_GPT_andrew_worked_days_l2012_201251


namespace NUMINAMATH_GPT_probability_entire_grid_black_l2012_201200

-- Definitions of the problem in terms of conditions.
def grid_size : Nat := 4

def prob_black_initial : ℚ := 1 / 2

def middle_squares : List (Nat × Nat) := [(2, 2), (2, 3), (3, 2), (3, 3)]

def edge_squares : List (Nat × Nat) := 
  [ (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3) ]

-- The probability that each of these squares is black independently.
def prob_all_middle_black : ℚ := (1 / 2) ^ 4

def prob_all_edge_black : ℚ := (1 / 2) ^ 12

-- The combined probability that the entire grid is black.
def prob_grid_black := prob_all_middle_black * prob_all_edge_black

-- Statement of the proof problem.
theorem probability_entire_grid_black :
  prob_grid_black = 1 / 65536 := by
  sorry

end NUMINAMATH_GPT_probability_entire_grid_black_l2012_201200


namespace NUMINAMATH_GPT_possible_k_values_l2012_201202

def triangle_right_k_values (AB AC : ℝ × ℝ) (k : ℝ) : Prop :=
  let BC := (AC.1 - AB.1, AC.2 - AB.2)
  let angle_A := AB.1 * AC.1 + AB.2 * AC.2 = 0   -- Condition for ∠A = 90°
  let angle_B := AB.1 * BC.1 + AB.2 * BC.2 = 0   -- Condition for ∠B = 90°
  let angle_C := BC.1 * AC.1 + BC.2 * AC.2 = 0   -- Condition for ∠C = 90°
  (angle_A ∨ angle_B ∨ angle_C)

theorem possible_k_values (k : ℝ) :
  triangle_right_k_values (2, 3) (1, k) k ↔
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13) / 2 :=
by
  sorry

end NUMINAMATH_GPT_possible_k_values_l2012_201202


namespace NUMINAMATH_GPT_train_length_is_95_l2012_201282

noncomputable def train_length (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ := 
  let speed_ms := speed_kmh * 1000 / 3600 
  speed_ms * time_seconds

theorem train_length_is_95 : train_length 1.5980030008814248 214 = 95 := by
  sorry

end NUMINAMATH_GPT_train_length_is_95_l2012_201282


namespace NUMINAMATH_GPT_riley_outside_fraction_l2012_201289

theorem riley_outside_fraction
  (awake_jonsey : ℚ := 2 / 3)
  (jonsey_outside_fraction : ℚ := 1 / 2)
  (awake_riley : ℚ := 3 / 4)
  (total_inside_time : ℚ := 10)
  (hours_per_day : ℕ := 24) :
  let jonsey_inside_time := 1 / 3 * hours_per_day
  let riley_inside_time := (1 - (8 / 9)) * (3 / 4) * hours_per_day
  jonsey_inside_time + riley_inside_time = total_inside_time :=
by
  sorry

end NUMINAMATH_GPT_riley_outside_fraction_l2012_201289


namespace NUMINAMATH_GPT_min_bought_chocolates_l2012_201273

variable (a b : ℕ)

theorem min_bought_chocolates :
    ∃ a : ℕ, 
        ∃ b : ℕ, 
            b = a + 41 
            ∧ (376 - a - b = 3 * a) 
            ∧ a = 67 :=
by
  sorry

end NUMINAMATH_GPT_min_bought_chocolates_l2012_201273


namespace NUMINAMATH_GPT_find_k_l2012_201240

theorem find_k (k : ℝ) (d : ℝ) (h : d = 4) :
  -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - d) → k = -16 :=
by
  intros
  rw [h] at *
  sorry

end NUMINAMATH_GPT_find_k_l2012_201240


namespace NUMINAMATH_GPT_charges_equal_at_x_4_cost_effectiveness_l2012_201245

-- Defining the conditions
def full_price : ℕ := 240

def yA (x : ℕ) : ℕ := 120 * x + 240
def yB (x : ℕ) : ℕ := 144 * x + 144

-- (Ⅰ) Establishing the expressions for the charges is already encapsulated in the definitions.

-- (Ⅱ) Proving the equivalence of the two charges for a specific number of students x.
theorem charges_equal_at_x_4 : ∀ x : ℕ, yA x = yB x ↔ x = 4 := 
by {
  sorry
}

-- (Ⅲ) Discussing which travel agency is more cost-effective based on the number of students x.
theorem cost_effectiveness (x : ℕ) :
  (x < 4 → yA x > yB x) ∧ (x > 4 → yA x < yB x) :=
by {
  sorry
}

end NUMINAMATH_GPT_charges_equal_at_x_4_cost_effectiveness_l2012_201245


namespace NUMINAMATH_GPT_ratio_of_speeds_is_2_l2012_201242

-- Definitions based on conditions
def rate_of_machine_B : ℕ := 100 / 40 -- Rate of Machine B (parts per minute)
def rate_of_machine_A : ℕ := 50 / 10 -- Rate of Machine A (parts per minute)
def ratio_of_speeds (rate_A rate_B : ℕ) : ℕ := rate_A / rate_B -- Ratio of speeds

-- Proof statement
theorem ratio_of_speeds_is_2 : ratio_of_speeds rate_of_machine_A rate_of_machine_B = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_is_2_l2012_201242


namespace NUMINAMATH_GPT_train_average_speed_l2012_201268

theorem train_average_speed (speed : ℕ) (stop_time : ℕ) (running_time : ℕ) (total_time : ℕ)
  (h1 : speed = 60)
  (h2 : stop_time = 24)
  (h3 : running_time = total_time - stop_time)
  (h4 : running_time = 36)
  (h5 : total_time = 60) :
  (speed * running_time / total_time = 36) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end NUMINAMATH_GPT_train_average_speed_l2012_201268


namespace NUMINAMATH_GPT_average_score_remaining_students_l2012_201277

theorem average_score_remaining_students (n : ℕ) (h : n > 15) (avg_all : ℚ) (avg_15 : ℚ) :
  avg_all = 12 → avg_15 = 20 →
  (∃ avg_remaining : ℚ, avg_remaining = (12 * n - 300) / (n - 15)) :=
by
  sorry

end NUMINAMATH_GPT_average_score_remaining_students_l2012_201277


namespace NUMINAMATH_GPT_swapped_digits_greater_by_18_l2012_201265

theorem swapped_digits_greater_by_18 (x : ℕ) : 
  (10 * x + 1) - (10 + x) = 18 :=
  sorry

end NUMINAMATH_GPT_swapped_digits_greater_by_18_l2012_201265


namespace NUMINAMATH_GPT_original_number_of_cards_l2012_201222

-- Declare variables r and b as naturals representing the number of red and black cards, respectively.
variable (r b : ℕ)

-- Assume the probabilities given in the problem.
axiom prob_red : (r : ℝ) / (r + b) = 1 / 3
axiom prob_red_after_add : (r : ℝ) / (r + b + 4) = 1 / 4

-- Define the statement we need to prove.
theorem original_number_of_cards : r + b = 12 :=
by
  -- The proof steps would be here, but we'll use sorry to avoid implementing them.
  sorry

end NUMINAMATH_GPT_original_number_of_cards_l2012_201222


namespace NUMINAMATH_GPT_meet_at_35_l2012_201214

def walking_distance_A (t : ℕ) := 5 * t

def walking_distance_B (t : ℕ) := (t * (7 + t)) / 2

def total_distance (t : ℕ) := walking_distance_A t + walking_distance_B t

theorem meet_at_35 : ∃ (t : ℕ), total_distance t = 100 ∧ walking_distance_A t - walking_distance_B t = 35 := by
  sorry

end NUMINAMATH_GPT_meet_at_35_l2012_201214


namespace NUMINAMATH_GPT_side_length_of_square_IJKL_l2012_201232

theorem side_length_of_square_IJKL 
  (x y : ℝ) (hypotenuse : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 9) 
  (h3 : hypotenuse = Real.sqrt (x^2 + y^2)) : 
  hypotenuse = 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_IJKL_l2012_201232


namespace NUMINAMATH_GPT_percentage_deficit_for_second_side_l2012_201207

-- Defining the given conditions and the problem statement
def side1_excess : ℚ := 0.14
def area_error : ℚ := 0.083
def original_length (L : ℚ) := L
def original_width (W : ℚ) := W
def measured_length_side1 (L : ℚ) := (1 + side1_excess) * L
def measured_width_side2 (W : ℚ) (x : ℚ) := W * (1 - 0.01 * x)
def original_area (L W : ℚ) := L * W
def calculated_area (L W x : ℚ) := 
  measured_length_side1 L * measured_width_side2 W x

theorem percentage_deficit_for_second_side (L W : ℚ) :
  (calculated_area L W 5) / (original_area L W) = 1 + area_error :=
by
  sorry

end NUMINAMATH_GPT_percentage_deficit_for_second_side_l2012_201207


namespace NUMINAMATH_GPT_consecutive_odd_integers_sum_l2012_201241

theorem consecutive_odd_integers_sum (n : ℤ) (h : (n - 2) + (n + 2) = 150) : n = 75 := 
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_integers_sum_l2012_201241


namespace NUMINAMATH_GPT_max_value_of_f_l2012_201209

-- Define the quadratic function
def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

-- Define a proof problem to show that the maximum value of f(x) is 81/16
theorem max_value_of_f : ∃ x : ℝ, f x = 81 / 16 :=
by
  -- The vertex of the quadratic function gives the maximum value since the parabola opens downward
  let x := 9 / (2 * 4)
  use x
  -- sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2012_201209


namespace NUMINAMATH_GPT_francine_leave_time_earlier_l2012_201216

-- Definitions for the conditions in the problem
def leave_time := "noon"  -- Francine and her father leave at noon every day.
def father_meet_time_shorten := 10  -- They arrived home 10 minutes earlier than usual.
def francine_walk_duration := 15  -- Francine walked for 15 minutes.

-- Premises based on the conditions
def usual_meet_time := 12 * 60  -- Meeting time in minutes from midnight (noon = 720 minutes)
def special_day_meet_time := usual_meet_time - father_meet_time_shorten / 2  -- 5 minutes earlier

-- The main theorem to prove: Francine leaves at 11:40 AM (700 minutes from midnight)
theorem francine_leave_time_earlier :
  usual_meet_time - (father_meet_time_shorten / 2 + francine_walk_duration) = (11 * 60 + 40) := by
  sorry

end NUMINAMATH_GPT_francine_leave_time_earlier_l2012_201216


namespace NUMINAMATH_GPT_expression_evaluation_l2012_201269

theorem expression_evaluation (x : ℤ) (hx : x = 4) : 5 * x + 3 - x^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2012_201269


namespace NUMINAMATH_GPT_combined_cost_of_apples_and_strawberries_l2012_201270

theorem combined_cost_of_apples_and_strawberries :
  let cost_of_apples := 15
  let cost_of_strawberries := 26
  cost_of_apples + cost_of_strawberries = 41 :=
by
  sorry

end NUMINAMATH_GPT_combined_cost_of_apples_and_strawberries_l2012_201270


namespace NUMINAMATH_GPT_smaller_number_is_22_l2012_201208

noncomputable def smaller_number (x y : ℕ) : ℕ := 
x

theorem smaller_number_is_22 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_is_22_l2012_201208


namespace NUMINAMATH_GPT_problem_1_problem_2_l2012_201296

-- Problem 1 statement
theorem problem_1 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_cond_a : a = 1/4) (h_cond_q : (1 : ℝ) / 2 < x ∧ x < 1) (h_cond_p : a < x ∧ x < 3 * a): 1 / 2 < x ∧ x < 3 / 4 :=
by sorry

-- Problem 2 statement
theorem problem_2 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_neg_p : ¬(a < x ∧ x < 3 * a)) (h_neg_q : ¬((1 / (2 : ℝ))^(m - 1) < x ∧ x < 1)): 1 / 3 ≤ a ∧ a ≤ 1 / 2 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2012_201296


namespace NUMINAMATH_GPT_equal_opposite_roots_eq_m_l2012_201254

theorem equal_opposite_roots_eq_m (a b c : ℝ) (m : ℝ) (h : (∃ x : ℝ, (a * x - c ≠ 0) ∧ (((x^2 - b * x) / (a * x - c)) = ((m - 1) / (m + 1)))) ∧
(∀ x : ℝ, ((x^2 - b * x) = 0 → x = 0) ∧ (∃ t : ℝ, t > 0 ∧ ((x = t) ∨ (x = -t))))):
  m = (a - b) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_equal_opposite_roots_eq_m_l2012_201254


namespace NUMINAMATH_GPT_sum_of_g_49_l2012_201255

def f (x : ℝ) := 4 * x^2 - 3
def g (y : ℝ) := y^2 + 2 * y + 2

theorem sum_of_g_49 : (g 49) = 30 :=
  sorry

end NUMINAMATH_GPT_sum_of_g_49_l2012_201255


namespace NUMINAMATH_GPT_cutting_wire_random_event_l2012_201262

noncomputable def length : ℝ := sorry

def is_random_event (a : ℝ) : Prop := sorry

theorem cutting_wire_random_event (a : ℝ) (h : a > 0) :
  is_random_event a := 
by
  sorry

end NUMINAMATH_GPT_cutting_wire_random_event_l2012_201262


namespace NUMINAMATH_GPT_find_the_number_l2012_201274

theorem find_the_number :
  ∃ x : ℕ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := 
  sorry

end NUMINAMATH_GPT_find_the_number_l2012_201274


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2012_201285

-- Define the first problem with necessary conditions
theorem simplify_expr1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (b - a)) = (a + b) / (a - b) :=
by
  sorry

-- Define the second problem with necessary conditions
theorem simplify_expr2 (x : ℝ) (hx1 : x ≠ -3) (hx2 : x ≠ 4) (hx3 : x ≠ -4) :
  ((x - 4) / (x + 3)) / (x - 3 - (7 / (x + 3))) = 1 / (x + 4) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2012_201285


namespace NUMINAMATH_GPT_determinant_nonnegative_of_skew_symmetric_matrix_l2012_201261

theorem determinant_nonnegative_of_skew_symmetric_matrix
  (a b c d e f : ℝ)
  (A : Matrix (Fin 4) (Fin 4) ℝ)
  (hA : A = ![
    ![0, a, b, c],
    ![-a, 0, d, e],
    ![-b, -d, 0, f],
    ![-c, -e, -f, 0]]) :
  0 ≤ Matrix.det A := by
  sorry

end NUMINAMATH_GPT_determinant_nonnegative_of_skew_symmetric_matrix_l2012_201261


namespace NUMINAMATH_GPT_find_real_numbers_l2012_201225

theorem find_real_numbers (x1 x2 x3 x4 : ℝ) :
  x1 + x2 * x3 * x4 = 2 →
  x2 + x1 * x3 * x4 = 2 →
  x3 + x1 * x2 * x4 = 2 →
  x4 + x1 * x2 * x3 = 2 →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨ 
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
by sorry

end NUMINAMATH_GPT_find_real_numbers_l2012_201225


namespace NUMINAMATH_GPT_seating_arrangement_correct_l2012_201205

-- Define the number of seating arrangements based on the given conditions

def seatingArrangements : Nat := 
  2 * 4 * 6

theorem seating_arrangement_correct :
  seatingArrangements = 48 := by
  sorry

end NUMINAMATH_GPT_seating_arrangement_correct_l2012_201205


namespace NUMINAMATH_GPT_dave_added_apps_l2012_201212

-- Define the conditions as a set of given facts
def initial_apps : Nat := 10
def deleted_apps : Nat := 17
def remaining_apps : Nat := 4

-- The statement to prove
theorem dave_added_apps : ∃ x : Nat, initial_apps + x - deleted_apps = remaining_apps ∧ x = 11 :=
by
  use 11
  sorry

end NUMINAMATH_GPT_dave_added_apps_l2012_201212


namespace NUMINAMATH_GPT_inequality_solution_l2012_201203

theorem inequality_solution (x : ℝ) (h : ∀ (a b : ℝ) (ha : 0 < a) (hb : 0 < b), x^2 + x < a / b + b / a) : x ∈ Set.Ioo (-2 : ℝ) 1 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l2012_201203


namespace NUMINAMATH_GPT_calculate_expression_l2012_201204

theorem calculate_expression : 200 * 39.96 * 3.996 * 500 = (3996)^2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2012_201204


namespace NUMINAMATH_GPT_time_spent_cleaning_bathroom_l2012_201221

-- Define the times spent on each task
def laundry_time : ℕ := 30
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40
def total_time : ℕ := 120

-- Let b be the time spent cleaning the bathroom
variable (b : ℕ)

-- Total time spent on all tasks is the sum of individual times
def total_task_time := laundry_time + b + room_cleaning_time + homework_time

-- Proof that b = 15 given the total time
theorem time_spent_cleaning_bathroom (h : total_task_time = total_time) : b = 15 :=
by
  sorry

end NUMINAMATH_GPT_time_spent_cleaning_bathroom_l2012_201221


namespace NUMINAMATH_GPT_Pablo_is_70_cm_taller_than_Charlene_l2012_201276

variable (Ruby Pablo Charlene Janet : ℕ)

-- Conditions
axiom h1 : Ruby + 2 = Pablo
axiom h2 : Charlene = 2 * Janet
axiom h3 : Janet = 62
axiom h4 : Ruby = 192

-- The statement to prove
theorem Pablo_is_70_cm_taller_than_Charlene : Pablo - Charlene = 70 :=
by
  -- Formalizing the proof
  sorry

end NUMINAMATH_GPT_Pablo_is_70_cm_taller_than_Charlene_l2012_201276


namespace NUMINAMATH_GPT_second_month_sales_l2012_201292

def sales_first_month : ℝ := 7435
def sales_third_month : ℝ := 7855
def sales_fourth_month : ℝ := 8230
def sales_fifth_month : ℝ := 7562
def sales_sixth_month : ℝ := 5991
def average_sales : ℝ := 7500

theorem second_month_sales : 
  ∃ (second_month_sale : ℝ), 
    (sales_first_month + second_month_sale + sales_third_month + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = average_sales ∧
    second_month_sale = 7927 := by
  sorry

end NUMINAMATH_GPT_second_month_sales_l2012_201292


namespace NUMINAMATH_GPT_triangle_interior_angle_l2012_201234

-- Define the given values and equations
variables (x : ℝ) 
def arc_DE := x + 80
def arc_EF := 2 * x + 30
def arc_FD := 3 * x - 25

-- The main proof statement
theorem triangle_interior_angle :
  arc_DE x + arc_EF x + arc_FD x = 360 →
  0.5 * (arc_EF x) = 60.83 :=
by sorry

end NUMINAMATH_GPT_triangle_interior_angle_l2012_201234


namespace NUMINAMATH_GPT_solve_equation_l2012_201263

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2012_201263


namespace NUMINAMATH_GPT_median_to_hypotenuse_of_right_triangle_l2012_201253

theorem median_to_hypotenuse_of_right_triangle (DE DF : ℝ) (h₁ : DE = 6) (h₂ : DF = 8) :
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  N = 5 :=
by
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  have h : N = 5 :=
    by
      sorry
  exact h

end NUMINAMATH_GPT_median_to_hypotenuse_of_right_triangle_l2012_201253


namespace NUMINAMATH_GPT_pipes_fill_tank_l2012_201220

theorem pipes_fill_tank (T : ℝ) (h1 : T > 0)
  (h2 : (1/4 : ℝ) + 1/T - 1/20 = 1/2.5) : T = 5 := by
  sorry

end NUMINAMATH_GPT_pipes_fill_tank_l2012_201220


namespace NUMINAMATH_GPT_ratio_of_wages_l2012_201299

def hours_per_day_josh : ℕ := 8
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def wage_per_hour_josh : ℕ := 9
def monthly_total_payment : ℚ := 1980

def hours_per_day_carl : ℕ := hours_per_day_josh - 2

def monthly_hours_josh : ℕ := hours_per_day_josh * days_per_week * weeks_per_month
def monthly_hours_carl : ℕ := hours_per_day_carl * days_per_week * weeks_per_month

def monthly_earnings_josh : ℚ := wage_per_hour_josh * monthly_hours_josh
def monthly_earnings_carl : ℚ := monthly_total_payment - monthly_earnings_josh

def hourly_wage_carl : ℚ := monthly_earnings_carl / monthly_hours_carl

theorem ratio_of_wages : hourly_wage_carl / wage_per_hour_josh = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_wages_l2012_201299


namespace NUMINAMATH_GPT_no_integers_satisfy_l2012_201258

def P (x a b c d : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integers_satisfy :
  ∀ a b c d : ℤ, ¬ (P 19 a b c d = 1 ∧ P 62 a b c d = 2) :=
by
  intro a b c d
  sorry

end NUMINAMATH_GPT_no_integers_satisfy_l2012_201258


namespace NUMINAMATH_GPT_a37_b37_sum_l2012_201224

-- Declare the sequences as functions from natural numbers to real numbers
variables {a b : ℕ → ℝ}

-- State the hypotheses based on the conditions
variables (h1 : ∀ n, a (n + 1) = a n + a 2 - a 1)
variables (h2 : ∀ n, b (n + 1) = b n + b 2 - b 1)
variables (h3 : a 1 = 25)
variables (h4 : b 1 = 75)
variables (h5 : a 2 + b 2 = 100)

-- State the theorem to be proved
theorem a37_b37_sum : a 37 + b 37 = 100 := 
by 
  sorry

end NUMINAMATH_GPT_a37_b37_sum_l2012_201224


namespace NUMINAMATH_GPT_sum_infinite_series_l2012_201294

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end NUMINAMATH_GPT_sum_infinite_series_l2012_201294


namespace NUMINAMATH_GPT_set_of_points_l2012_201295

theorem set_of_points : {p : ℝ × ℝ | (2 * p.1 - p.2 = 1) ∧ (p.1 + 4 * p.2 = 5)} = { (1, 1) } :=
by
  sorry

end NUMINAMATH_GPT_set_of_points_l2012_201295


namespace NUMINAMATH_GPT_bathroom_area_is_50_square_feet_l2012_201206

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end NUMINAMATH_GPT_bathroom_area_is_50_square_feet_l2012_201206


namespace NUMINAMATH_GPT_volume_of_sphere_l2012_201288

theorem volume_of_sphere (V : ℝ) (r : ℝ) : r = 1 / 3 → (2 * r) = (16 / 9 * V)^(1/3) → V = 1 / 6 :=
by
  intro h_radius h_diameter
  sorry

end NUMINAMATH_GPT_volume_of_sphere_l2012_201288


namespace NUMINAMATH_GPT_Linda_needs_15_hours_to_cover_fees_l2012_201229

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end NUMINAMATH_GPT_Linda_needs_15_hours_to_cover_fees_l2012_201229


namespace NUMINAMATH_GPT_fraction_equality_l2012_201275

theorem fraction_equality 
  (a b c d : ℝ)
  (h1 : a + c = 2 * b)
  (h2 : 2 * b * d = c * (b + d))
  (hb : b ≠ 0)
  (hd : d ≠ 0) :
  a / b = c / d :=
sorry

end NUMINAMATH_GPT_fraction_equality_l2012_201275


namespace NUMINAMATH_GPT_stones_required_to_pave_hall_l2012_201218

theorem stones_required_to_pave_hall :
  ∀ (hall_length_m hall_breadth_m stone_length_dm stone_breadth_dm: ℕ),
  hall_length_m = 72 →
  hall_breadth_m = 30 →
  stone_length_dm = 6 →
  stone_breadth_dm = 8 →
  (hall_length_m * 10 * hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm) = 4500 := by
  intros _ _ _ _ h_length h_breadth h_slength h_sbreadth
  sorry

end NUMINAMATH_GPT_stones_required_to_pave_hall_l2012_201218


namespace NUMINAMATH_GPT_cone_surface_area_l2012_201281

-- Define the surface area formula for a cone with radius r and slant height l
theorem cone_surface_area (r l : ℝ) : 
  let S := π * r^2 + π * r * l
  S = π * r^2 + π * r * l :=
by sorry

end NUMINAMATH_GPT_cone_surface_area_l2012_201281


namespace NUMINAMATH_GPT_eric_has_more_than_500_paperclips_on_saturday_l2012_201256

theorem eric_has_more_than_500_paperclips_on_saturday :
  ∃ k : ℕ, (4 * 3 ^ k > 500) ∧ (∀ m : ℕ, m < k → 4 * 3 ^ m ≤ 500) ∧ ((k + 1) % 7 = 6) :=
by
  sorry

end NUMINAMATH_GPT_eric_has_more_than_500_paperclips_on_saturday_l2012_201256


namespace NUMINAMATH_GPT_find_speed_of_stream_l2012_201213

def distance : ℝ := 24
def total_time : ℝ := 5
def rowing_speed : ℝ := 10

def speed_of_stream (v : ℝ) : Prop :=
  distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time

theorem find_speed_of_stream : ∃ v : ℝ, speed_of_stream v ∧ v = 2 :=
by
  exists 2
  unfold speed_of_stream
  simp
  sorry -- This would be the proof part which is not required here

end NUMINAMATH_GPT_find_speed_of_stream_l2012_201213


namespace NUMINAMATH_GPT_isosceles_triangle_properties_l2012_201239

/--
  An isosceles triangle has a base of 6 units and legs of 5 units each.
  Prove:
  1. The area of the triangle is 12 square units.
  2. The radius of the inscribed circle is 1.5 units.
-/
theorem isosceles_triangle_properties (base : ℝ) (legs : ℝ) 
  (h_base : base = 6) (h_legs : legs = 5) : 
  ∃ (area : ℝ) (inradius : ℝ), 
  area = 12 ∧ inradius = 1.5 
  :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_properties_l2012_201239


namespace NUMINAMATH_GPT_gray_area_is_50pi_l2012_201272

noncomputable section

-- Define the radii of the inner and outer circles
def R_inner : ℝ := 2.5
def R_outer : ℝ := 3 * R_inner

-- Area of circles
def A_inner : ℝ := Real.pi * R_inner^2
def A_outer : ℝ := Real.pi * R_outer^2

-- Define width of the gray region
def gray_width : ℝ := R_outer - R_inner

-- Gray area calculation
def A_gray : ℝ := A_outer - A_inner

-- The theorem stating the area of the gray region
theorem gray_area_is_50pi :
  gray_width = 5 → A_gray = 50 * Real.pi := by
  -- Here we assume the proof continues
  sorry

end NUMINAMATH_GPT_gray_area_is_50pi_l2012_201272


namespace NUMINAMATH_GPT_four_leaf_area_l2012_201235

theorem four_leaf_area (a : ℝ) : 
  let radius := a / 2
  let semicircle_area := (π * radius ^ 2) / 2
  let triangle_area := (a / 2) * (a / 2) / 2
  let half_leaf_area := semicircle_area - triangle_area
  let leaf_area := 2 * half_leaf_area
  let total_area := 4 * leaf_area
  total_area = a ^ 2 / 2 * (π - 2) := 
by
  sorry

end NUMINAMATH_GPT_four_leaf_area_l2012_201235


namespace NUMINAMATH_GPT_volume_of_cut_out_box_l2012_201238

theorem volume_of_cut_out_box (x : ℝ) : 
  let l := 16
  let w := 12
  let new_l := l - 2 * x
  let new_w := w - 2 * x
  let height := x
  let V := new_l * new_w * height
  V = 4 * x^3 - 56 * x^2 + 192 * x :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cut_out_box_l2012_201238


namespace NUMINAMATH_GPT_find_triples_l2012_201291

theorem find_triples 
  (x y z : ℝ)
  (h1 : x + y * z = 2)
  (h2 : y + z * x = 2)
  (h3 : z + x * y = 2)
 : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end NUMINAMATH_GPT_find_triples_l2012_201291


namespace NUMINAMATH_GPT_jeff_makes_donuts_for_days_l2012_201201

variable (d : ℕ) (boxes donuts_per_box : ℕ) (donuts_per_day eaten_per_day : ℕ) (chris_eaten total_donuts : ℕ)

theorem jeff_makes_donuts_for_days :
  (donuts_per_day = 10) →
  (eaten_per_day = 1) →
  (chris_eaten = 8) →
  (boxes = 10) →
  (donuts_per_box = 10) →
  (total_donuts = boxes * donuts_per_box) →
  (9 * d - chris_eaten = total_donuts) →
  d = 12 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end NUMINAMATH_GPT_jeff_makes_donuts_for_days_l2012_201201


namespace NUMINAMATH_GPT_boys_tried_out_l2012_201284

theorem boys_tried_out (G B C N : ℕ) (hG : G = 9) (hC : C = 2) (hN : N = 21) (h : G + B - C = N) : B = 14 :=
by
  -- The proof is omitted, focusing only on stating the theorem
  sorry

end NUMINAMATH_GPT_boys_tried_out_l2012_201284


namespace NUMINAMATH_GPT_count_positive_integers_m_l2012_201286

theorem count_positive_integers_m :
  ∃ m_values : Finset ℕ, m_values.card = 4 ∧ ∀ m ∈ m_values, 
    ∃ k : ℕ, k > 0 ∧ (7 * m + 2 = m * k + 2 * m) := 
sorry

end NUMINAMATH_GPT_count_positive_integers_m_l2012_201286


namespace NUMINAMATH_GPT_jakes_class_boys_count_l2012_201249

theorem jakes_class_boys_count 
    (ratio_girls_boys : ℕ → ℕ → Prop)
    (students_total : ℕ)
    (ratio_condition : ratio_girls_boys 3 4)
    (total_condition : students_total = 35) :
    ∃ boys : ℕ, boys = 20 :=
by
  sorry

end NUMINAMATH_GPT_jakes_class_boys_count_l2012_201249


namespace NUMINAMATH_GPT_find_a8_l2012_201210

theorem find_a8 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n : ℕ, (1 / (a n + 1)) = (1 / (a 0 + 1)) + n * ((1 / (a 1 + 1 - 1)) / 3)) 
  (h2 : a 2 = 3) 
  (h5 : a 5 = 1) : 
  a 8 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a8_l2012_201210


namespace NUMINAMATH_GPT_ferry_time_difference_l2012_201228

-- Definitions for the given conditions
def speed_p := 8
def time_p := 3
def distance_p := speed_p * time_p
def distance_q := 3 * distance_p
def speed_q := speed_p + 1
def time_q := distance_q / speed_q

-- Theorem to be proven
theorem ferry_time_difference : (time_q - time_p) = 5 := 
by
  let speed_p := 8
  let time_p := 3
  let distance_p := speed_p * time_p
  let distance_q := 3 * distance_p
  let speed_q := speed_p + 1
  let time_q := distance_q / speed_q
  sorry

end NUMINAMATH_GPT_ferry_time_difference_l2012_201228


namespace NUMINAMATH_GPT_copper_sheet_area_l2012_201211

noncomputable def area_of_copper_sheet (l w h : ℝ) (thickness_mm : ℝ) : ℝ :=
  let volume := l * w * h
  let thickness_cm := thickness_mm / 10
  (volume / thickness_cm) / 10000

theorem copper_sheet_area :
  ∀ (l w h thickness_mm : ℝ), 
  l = 80 → w = 20 → h = 5 → thickness_mm = 1 → 
  area_of_copper_sheet l w h thickness_mm = 8 := 
by
  intros l w h thickness_mm hl hw hh hthickness_mm
  rw [hl, hw, hh, hthickness_mm]
  simp [area_of_copper_sheet]
  sorry

end NUMINAMATH_GPT_copper_sheet_area_l2012_201211


namespace NUMINAMATH_GPT_units_digit_product_l2012_201279

theorem units_digit_product (a b : ℕ) (ha : a % 10 = 7) (hb : b % 10 = 4) :
  (a * b) % 10 = 8 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_product_l2012_201279


namespace NUMINAMATH_GPT_sum_final_numbers_l2012_201283

theorem sum_final_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_final_numbers_l2012_201283


namespace NUMINAMATH_GPT_simplify_expression_calculate_difference_of_squares_l2012_201230

section Problem1
variable (a b : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem simplify_expression : ((-2 * a^2) ^ 2 * (-b^2)) / (4 * a^3 * b^2) = -a :=
by sorry
end Problem1

section Problem2

theorem calculate_difference_of_squares : 2023^2 - 2021 * 2025 = 4 :=
by sorry
end Problem2

end NUMINAMATH_GPT_simplify_expression_calculate_difference_of_squares_l2012_201230


namespace NUMINAMATH_GPT_solve_for_q_l2012_201298

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l2012_201298


namespace NUMINAMATH_GPT_simplify_expression_l2012_201223

theorem simplify_expression :
  (↑(Real.sqrt 648) / ↑(Real.sqrt 81) - ↑(Real.sqrt 245) / ↑(Real.sqrt 49)) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_simplify_expression_l2012_201223


namespace NUMINAMATH_GPT_slower_pump_time_l2012_201233

def pool_problem (R : ℝ) :=
  (∀ t : ℝ, (2.5 * R * t = 1) → (t = 5))
  ∧ (∀ R1 R2 : ℝ, (R1 = 1.5 * R) → (R1 + R = 2.5 * R))
  ∧ (∀ t : ℝ, (R * t = 1) → (t = 12.5))

theorem slower_pump_time (R : ℝ) : pool_problem R :=
by
  -- Assume that the combined rates take 5 hours to fill the pool
  sorry

end NUMINAMATH_GPT_slower_pump_time_l2012_201233


namespace NUMINAMATH_GPT_eliminate_denominators_l2012_201278

variable {x : ℝ}

theorem eliminate_denominators (h : 3 / (2 * x) = 1 / (x - 1)) :
  3 * x - 3 = 2 * x := 
by
  sorry

end NUMINAMATH_GPT_eliminate_denominators_l2012_201278


namespace NUMINAMATH_GPT_highest_throw_christine_janice_l2012_201236

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end NUMINAMATH_GPT_highest_throw_christine_janice_l2012_201236


namespace NUMINAMATH_GPT_perfect_square_trinomial_coeff_l2012_201293

theorem perfect_square_trinomial_coeff (m : ℝ) : (∃ a b : ℝ, (a ≠ 0) ∧ ((a * x + b)^2 = x^2 - m * x + 25)) ↔ (m = 10 ∨ m = -10) :=
by sorry

end NUMINAMATH_GPT_perfect_square_trinomial_coeff_l2012_201293


namespace NUMINAMATH_GPT_min_rounds_for_expected_value_l2012_201297

theorem min_rounds_for_expected_value 
  (p1 p2 : ℝ) (h0 : 0 ≤ p1 ∧ p1 ≤ 1) (h1 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h2 : p1 + p2 = 3 / 2)
  (indep : true) -- Assuming independence implicitly
  (X : ℕ → ℕ) (n : ℕ)
  (E_X_eq_24 : (n : ℕ) * (3 * p1 * p2 * (1 - p1 * p2)) = 24) :
  n = 32 := 
sorry

end NUMINAMATH_GPT_min_rounds_for_expected_value_l2012_201297


namespace NUMINAMATH_GPT_initial_geese_count_l2012_201266

-- Define the number of geese that flew away
def geese_flew_away : ℕ := 28

-- Define the number of geese left in the field
def geese_left : ℕ := 23

-- Prove that the initial number of geese in the field was 51
theorem initial_geese_count : geese_left + geese_flew_away = 51 := by
  sorry

end NUMINAMATH_GPT_initial_geese_count_l2012_201266


namespace NUMINAMATH_GPT_either_p_or_q_false_suff_not_p_true_l2012_201271

theorem either_p_or_q_false_suff_not_p_true (p q : Prop) : (p ∨ q = false) → (¬p = true) :=
by
  sorry

end NUMINAMATH_GPT_either_p_or_q_false_suff_not_p_true_l2012_201271


namespace NUMINAMATH_GPT_smallest_pos_int_mult_4410_sq_l2012_201227

noncomputable def smallest_y : ℤ := 10

theorem smallest_pos_int_mult_4410_sq (y : ℕ) (hy : y > 0) :
  (∃ z : ℕ, 4410 * y = z^2) ↔ y = smallest_y :=
sorry

end NUMINAMATH_GPT_smallest_pos_int_mult_4410_sq_l2012_201227


namespace NUMINAMATH_GPT_average_of_seven_starting_with_d_l2012_201259

theorem average_of_seven_starting_with_d (c d : ℕ) (h : d = (c + 3)) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end NUMINAMATH_GPT_average_of_seven_starting_with_d_l2012_201259


namespace NUMINAMATH_GPT_dvds_still_fit_in_book_l2012_201290

def total_capacity : ℕ := 126
def dvds_already_in_book : ℕ := 81

theorem dvds_still_fit_in_book : (total_capacity - dvds_already_in_book = 45) :=
by
  sorry

end NUMINAMATH_GPT_dvds_still_fit_in_book_l2012_201290


namespace NUMINAMATH_GPT_total_balls_l2012_201219

theorem total_balls (colors : ℕ) (balls_per_color : ℕ) (h_colors : colors = 10) (h_balls_per_color : balls_per_color = 35) : 
    colors * balls_per_color = 350 :=
by
  -- Import necessary libraries
  sorry

end NUMINAMATH_GPT_total_balls_l2012_201219


namespace NUMINAMATH_GPT_max_value_g_l2012_201217

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

theorem max_value_g : ∃ (x₁ N : ℝ), (0 ≤ x₁ ∧ x₁ ≤ 4) ∧ (N = 16) ∧ (x₁ = 2) ∧ (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ N) :=
by
  sorry

end NUMINAMATH_GPT_max_value_g_l2012_201217


namespace NUMINAMATH_GPT_rhombus_perimeter_and_radius_l2012_201237

-- Define the rhombus with given diagonals
structure Rhombus where
  d1 : ℝ -- diagonal 1
  d2 : ℝ -- diagonal 2
  h : d1 = 20 ∧ d2 = 16

-- Define the proof problem
theorem rhombus_perimeter_and_radius (r : Rhombus) : 
  let side_length := Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)
  let perimeter := 4 * side_length
  let radius := r.d1 / 2
  perimeter = 16 * Real.sqrt 41 ∧ radius = 10 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_and_radius_l2012_201237


namespace NUMINAMATH_GPT_part1_part2_part3_l2012_201287

-- Part 1
theorem part1 (x : ℝ) (h : abs (x + 2) = abs (x - 4)) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h : abs (x + 2) + abs (x - 4) = 8) : x = -3 ∨ x = 5 :=
by
  sorry

-- Part 3
theorem part3 (t : ℝ) :
  let M := -2 - t
  let N := 4 - 3 * t
  (abs M = abs (M - N) → t = 1/2) ∧ 
  (N = 0 → t = 4/3) ∧
  (abs N = abs (N - M) → t = 2) ∧
  (M = N → t = 3) ∧
  (abs (M - N) = abs (2 * M) → t = 8) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l2012_201287


namespace NUMINAMATH_GPT_find_a_l2012_201267

noncomputable def P (a : ℚ) (k : ℕ) : ℚ := a * (1 / 2)^(k)

theorem find_a (a : ℚ) : (P a 1 + P a 2 + P a 3 = 1) → (a = 8 / 7) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2012_201267


namespace NUMINAMATH_GPT_cindy_marbles_problem_l2012_201246

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end NUMINAMATH_GPT_cindy_marbles_problem_l2012_201246


namespace NUMINAMATH_GPT_geometric_seq_condition_l2012_201250

variable (n : ℕ) (a : ℕ → ℝ)

-- The definition of a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n + 1) = a n * a (n + 2)

-- The main theorem statement
theorem geometric_seq_condition :
  (is_geometric_seq a n → ∀ n, |a n| ≥ 0) →
  ∃ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = a (n + 1) * a (n + 1)) →
  (∀ m, a m = 0 → ¬(is_geometric_seq a n)) :=
sorry

end NUMINAMATH_GPT_geometric_seq_condition_l2012_201250


namespace NUMINAMATH_GPT_complex_norm_wz_l2012_201231

open Complex

theorem complex_norm_wz (w z : ℂ) (h₁ : ‖w + z‖ = 2) (h₂ : ‖w^2 + z^2‖ = 8) : 
  ‖w^4 + z^4‖ = 56 := 
  sorry

end NUMINAMATH_GPT_complex_norm_wz_l2012_201231


namespace NUMINAMATH_GPT_cole_round_trip_time_l2012_201280

-- Define the relevant quantities
def speed_to_work : ℝ := 70 -- km/h
def speed_to_home : ℝ := 105 -- km/h
def time_to_work_mins : ℝ := 72 -- minutes

-- Define the theorem to be proved
theorem cole_round_trip_time : 
  (time_to_work_mins / 60 + (speed_to_work * time_to_work_mins / 60) / speed_to_home) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cole_round_trip_time_l2012_201280


namespace NUMINAMATH_GPT_complement_set_U_A_l2012_201215

-- Definitions of U and A
def U : Set ℝ := { x : ℝ | x^2 ≤ 4 }
def A : Set ℝ := { x : ℝ | |x - 1| ≤ 1 }

-- Theorem statement
theorem complement_set_U_A : (U \ A) = { x : ℝ | -2 ≤ x ∧ x < 0 } := 
by
  sorry

end NUMINAMATH_GPT_complement_set_U_A_l2012_201215


namespace NUMINAMATH_GPT_evaluate_expression_at_3_l2012_201247

-- Define the expression
def expression (x : ℕ) : ℕ := x^2 - 3*x + 2

-- Statement of the problem
theorem evaluate_expression_at_3 : expression 3 = 2 := by
    sorry -- Proof is omitted

end NUMINAMATH_GPT_evaluate_expression_at_3_l2012_201247


namespace NUMINAMATH_GPT_least_number_to_add_l2012_201248

theorem least_number_to_add (n : ℕ) : (3457 + n) % 103 = 0 ↔ n = 45 :=
by sorry

end NUMINAMATH_GPT_least_number_to_add_l2012_201248
