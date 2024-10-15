import Mathlib

namespace NUMINAMATH_GPT_cubic_expression_value_l2289_228981

theorem cubic_expression_value (a b c : ℝ) 
  (h1 : a + b + c = 13) 
  (h2 : ab + ac + bc = 32) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 949 := 
by
  sorry

end NUMINAMATH_GPT_cubic_expression_value_l2289_228981


namespace NUMINAMATH_GPT_roots_cubic_roots_sum_of_squares_l2289_228956

variables {R : Type*} [CommRing R] {p q r s t : R}

theorem roots_cubic_roots_sum_of_squares (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
sorry

end NUMINAMATH_GPT_roots_cubic_roots_sum_of_squares_l2289_228956


namespace NUMINAMATH_GPT_pushups_difference_l2289_228971

theorem pushups_difference :
  let David_pushups := 44
  let Zachary_pushups := 35
  David_pushups - Zachary_pushups = 9 :=
by
  -- Here we define the push-ups counts
  let David_pushups := 44
  let Zachary_pushups := 35
  -- We need to show that David did 9 more push-ups than Zachary.
  show David_pushups - Zachary_pushups = 9
  sorry

end NUMINAMATH_GPT_pushups_difference_l2289_228971


namespace NUMINAMATH_GPT_max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l2289_228991

open Real

theorem max_sum_sqrt_expr (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h_sum : a + b + c = 8) :
  sqrt (3 * a^2 + 1) + sqrt (3 * b^2 + 1) + sqrt (3 * c^2 + 1) ≤ sqrt 201 :=
  sorry

theorem max_sum_sqrt_expr_attained : sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) = sqrt 201 :=
  sorry

end NUMINAMATH_GPT_max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l2289_228991


namespace NUMINAMATH_GPT_expected_volunteers_by_2022_l2289_228982

noncomputable def initial_volunteers : ℕ := 1200
noncomputable def increase_2021 : ℚ := 0.15
noncomputable def increase_2022 : ℚ := 0.30

theorem expected_volunteers_by_2022 :
  (initial_volunteers * (1 + increase_2021) * (1 + increase_2022)) = 1794 := 
by
  sorry

end NUMINAMATH_GPT_expected_volunteers_by_2022_l2289_228982


namespace NUMINAMATH_GPT_solution_set_bf_x2_solution_set_g_l2289_228901

def f (x : ℝ) := x^2 - 5 * x + 6

theorem solution_set_bf_x2 (x : ℝ) : (2 < x ∧ x < 3) ↔ f x < 0 := sorry

noncomputable def g (x : ℝ) := 6 * x^2 - 5 * x + 1

theorem solution_set_g (x : ℝ) : (1 / 3 < x ∧ x < 1 / 2) ↔ g x < 0 := sorry

end NUMINAMATH_GPT_solution_set_bf_x2_solution_set_g_l2289_228901


namespace NUMINAMATH_GPT_flooring_sq_ft_per_box_l2289_228999

/-- The problem statement converted into a Lean theorem -/
theorem flooring_sq_ft_per_box
  (living_room_length : ℕ)
  (living_room_width : ℕ)
  (flooring_installed : ℕ)
  (additional_boxes : ℕ)
  (correct_answer : ℕ) 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_installed = 250)
  (h4 : additional_boxes = 7)
  (h5 : correct_answer = 10) :
  
  (living_room_length * living_room_width - flooring_installed) / additional_boxes = correct_answer :=
by 
  sorry

end NUMINAMATH_GPT_flooring_sq_ft_per_box_l2289_228999


namespace NUMINAMATH_GPT_John_overall_profit_l2289_228919

theorem John_overall_profit :
  let CP_grinder := 15000
  let Loss_percentage_grinder := 0.04
  let CP_mobile_phone := 8000
  let Profit_percentage_mobile_phone := 0.10
  let CP_refrigerator := 24000
  let Profit_percentage_refrigerator := 0.08
  let CP_television := 12000
  let Loss_percentage_television := 0.06
  let SP_grinder := CP_grinder * (1 - Loss_percentage_grinder)
  let SP_mobile_phone := CP_mobile_phone * (1 + Profit_percentage_mobile_phone)
  let SP_refrigerator := CP_refrigerator * (1 + Profit_percentage_refrigerator)
  let SP_television := CP_television * (1 - Loss_percentage_television)
  let Total_CP := CP_grinder + CP_mobile_phone + CP_refrigerator + CP_television
  let Total_SP := SP_grinder + SP_mobile_phone + SP_refrigerator + SP_television
  let Overall_profit := Total_SP - Total_CP
  Overall_profit = 1400 := by
  sorry

end NUMINAMATH_GPT_John_overall_profit_l2289_228919


namespace NUMINAMATH_GPT_find_x_in_sequence_l2289_228942

theorem find_x_in_sequence :
  ∃ x y z : ℤ, 
    (z - 1 = 0) ∧ (y - z = -1) ∧ (x - y = 1) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_sequence_l2289_228942


namespace NUMINAMATH_GPT_explicit_formula_l2289_228908

variable (f : ℝ → ℝ)
variable (is_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
variable (max_value : ∀ x, f x ≤ 13)
variable (value_at_3 : f 3 = 5)
variable (value_at_neg1 : f (-1) = 5)

theorem explicit_formula :
  (∀ x, f x = -2 * x^2 + 4 * x + 11) :=
by
  sorry

end NUMINAMATH_GPT_explicit_formula_l2289_228908


namespace NUMINAMATH_GPT_angle_between_line_and_plane_l2289_228963

-- Define the conditions
def angle_direct_vector_normal_vector (direction_vector_angle : ℝ) := direction_vector_angle = 120

-- Define the goal to prove
theorem angle_between_line_and_plane (direction_vector_angle : ℝ) :
  angle_direct_vector_normal_vector direction_vector_angle → direction_vector_angle = 120 → 90 - (180 - direction_vector_angle) = 30 :=
by
  intros h_angle_eq angle_120
  sorry

end NUMINAMATH_GPT_angle_between_line_and_plane_l2289_228963


namespace NUMINAMATH_GPT_lunch_break_is_48_minutes_l2289_228920

noncomputable def lunch_break_duration (L : ℝ) (p a : ℝ) : Prop :=
  (8 - L) * (p + a) = 0.6 ∧ 
  (9 - L) * p = 0.35 ∧
  (5 - L) * a = 0.1

theorem lunch_break_is_48_minutes :
  ∃ L p a, lunch_break_duration L p a ∧ L * 60 = 48 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_lunch_break_is_48_minutes_l2289_228920


namespace NUMINAMATH_GPT_tailwind_speed_l2289_228905

-- Define the given conditions
def plane_speed_with_wind (P W : ℝ) : Prop := P + W = 460
def plane_speed_against_wind (P W : ℝ) : Prop := P - W = 310

-- Theorem stating the proof problem
theorem tailwind_speed (P W : ℝ) 
  (h1 : plane_speed_with_wind P W) 
  (h2 : plane_speed_against_wind P W) : 
  W = 75 :=
sorry

end NUMINAMATH_GPT_tailwind_speed_l2289_228905


namespace NUMINAMATH_GPT_line_exists_l2289_228914

theorem line_exists (x y x' y' : ℝ)
  (h1 : x' = 3 * x + 2 * y + 1)
  (h2 : y' = x + 4 * y - 3) : 
  (∃ A B C : ℝ, A * x + B * y + C = 0 ∧ A * x' + B * y' + C = 0 ∧ 
  ((A = 1 ∧ B = -1 ∧ C = 4) ∨ (A = 4 ∧ B = -8 ∧ C = -5))) :=
sorry

end NUMINAMATH_GPT_line_exists_l2289_228914


namespace NUMINAMATH_GPT_find_k_value_l2289_228937

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_k_value_l2289_228937


namespace NUMINAMATH_GPT_displacement_representation_l2289_228984

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end NUMINAMATH_GPT_displacement_representation_l2289_228984


namespace NUMINAMATH_GPT_pirate_prob_l2289_228936

def probability_treasure_no_traps := 1 / 3
def probability_traps_no_treasure := 1 / 6
def probability_neither := 1 / 2

theorem pirate_prob : (70 : ℝ) * ((1 / 3)^4 * (1 / 2)^4) = 35 / 648 := by
  sorry

end NUMINAMATH_GPT_pirate_prob_l2289_228936


namespace NUMINAMATH_GPT_correct_option_A_l2289_228938

theorem correct_option_A : 
  (∀ a : ℝ, a^3 * a^4 = a^7) ∧ 
  ¬ (∀ a : ℝ, a^6 / a^2 = a^3) ∧ 
  ¬ (∀ a : ℝ, a^4 - a^2 = a^2) ∧ 
  ¬ (∀ a b : ℝ, (a - b)^2 = a^2 - b^2) :=
by
  /- omitted proofs -/
  sorry

end NUMINAMATH_GPT_correct_option_A_l2289_228938


namespace NUMINAMATH_GPT_final_result_l2289_228979

noncomputable def f : ℝ → ℝ := sorry
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (3 + x) = f x
axiom f_half_periodic : ∀ x : ℝ, f (3 / 2 - x) = f x
axiom f_value_neg2 : f (-2) = -3

axiom a1_value : a 1 = -1
axiom S_n : ∀ n : ℕ, S n = 2 * a n + n

theorem final_result : f (a 5) + f (a 6) = 3 :=
sorry

end NUMINAMATH_GPT_final_result_l2289_228979


namespace NUMINAMATH_GPT_cupcake_difference_l2289_228959

def betty_rate : ℕ := 10
def dora_rate : ℕ := 8
def total_hours : ℕ := 5
def betty_break_hours : ℕ := 2

theorem cupcake_difference :
  (dora_rate * total_hours) - (betty_rate * (total_hours - betty_break_hours)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_cupcake_difference_l2289_228959


namespace NUMINAMATH_GPT_positive_diff_40_x_l2289_228906

theorem positive_diff_40_x
  (x : ℝ)
  (h : (40 + x + 15) / 3 = 35) :
  abs (x - 40) = 10 :=
sorry

end NUMINAMATH_GPT_positive_diff_40_x_l2289_228906


namespace NUMINAMATH_GPT_hexagon_diagonals_l2289_228918

theorem hexagon_diagonals (n : ℕ) (h : n = 6) : (n * (n - 3)) / 2 = 9 := by
  sorry

end NUMINAMATH_GPT_hexagon_diagonals_l2289_228918


namespace NUMINAMATH_GPT_train_passes_man_in_15_seconds_l2289_228995

theorem train_passes_man_in_15_seconds
  (length_of_train : ℝ)
  (speed_of_train : ℝ)
  (speed_of_man : ℝ)
  (direction_opposite : Bool)
  (h1 : length_of_train = 275)
  (h2 : speed_of_train = 60)
  (h3 : speed_of_man = 6)
  (h4 : direction_opposite = true) : 
  ∃ t : ℝ, t = 15 :=
by
  sorry

end NUMINAMATH_GPT_train_passes_man_in_15_seconds_l2289_228995


namespace NUMINAMATH_GPT_trips_per_student_l2289_228977

theorem trips_per_student
  (num_students : ℕ := 5)
  (chairs_per_trip : ℕ := 5)
  (total_chairs : ℕ := 250)
  (T : ℕ) :
  num_students * chairs_per_trip * T = total_chairs → T = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_trips_per_student_l2289_228977


namespace NUMINAMATH_GPT_floor_negative_fraction_l2289_228935

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end NUMINAMATH_GPT_floor_negative_fraction_l2289_228935


namespace NUMINAMATH_GPT_tomatoes_grew_in_absence_l2289_228968

def initial_tomatoes : ℕ := 36
def multiplier : ℕ := 100
def total_tomatoes_after_vacation : ℕ := initial_tomatoes * multiplier

theorem tomatoes_grew_in_absence : 
  total_tomatoes_after_vacation - initial_tomatoes = 3564 :=
by
  -- skipped proof with 'sorry'
  sorry

end NUMINAMATH_GPT_tomatoes_grew_in_absence_l2289_228968


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l2289_228928

theorem triangle_angle_contradiction (a b c : ℝ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : a + b + c = 180) (h₃ : 60 < a ∧ 60 < b ∧ 60 < c) : false :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l2289_228928


namespace NUMINAMATH_GPT_four_digit_perfect_square_is_1156_l2289_228949

theorem four_digit_perfect_square_is_1156 :
  ∃ (N : ℕ), (N ≥ 1000) ∧ (N < 10000) ∧ (∀ a, a ∈ [N / 1000, (N % 1000) / 100, (N % 100) / 10, N % 10] → a < 7) 
              ∧ (∃ n : ℕ, N = n * n) ∧ (∃ m : ℕ, (N + 3333 = m * m)) ∧ (N = 1156) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_perfect_square_is_1156_l2289_228949


namespace NUMINAMATH_GPT_fifth_segment_student_l2289_228954

variable (N : ℕ) (n : ℕ) (second_segment_student : ℕ)

def sampling_interval (N n : ℕ) : ℕ := N / n

def initial_student (second_segment_student interval : ℕ) : ℕ := second_segment_student - interval

def student_number (initial_student interval : ℕ) (segment : ℕ) : ℕ :=
  initial_student + (segment - 1) * interval

theorem fifth_segment_student (N n : ℕ) (second_segment_student : ℕ) (hN : N = 700) (hn : n = 50) (hsecond : second_segment_student = 20) :
  student_number (initial_student second_segment_student (sampling_interval N n)) (sampling_interval N n) 5 = 62 := by
  sorry

end NUMINAMATH_GPT_fifth_segment_student_l2289_228954


namespace NUMINAMATH_GPT_range_of_a_l2289_228985

theorem range_of_a {a : ℝ} (h : (a^2) / 4 + 1 / 2 < 1) : -Real.sqrt 2 < a ∧ a < Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2289_228985


namespace NUMINAMATH_GPT_intersection_points_of_segments_l2289_228961

noncomputable def num_intersection_points (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) : ℕ :=
  3000

theorem intersection_points_of_segments (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) :
  num_intersection_points A B C P Q = 3000 :=
  by sorry

end NUMINAMATH_GPT_intersection_points_of_segments_l2289_228961


namespace NUMINAMATH_GPT_set_of_points_l2289_228917

theorem set_of_points (x y : ℝ) (h : x^2 * y - y ≥ 0) :
  (y ≥ 0 ∧ |x| ≥ 1) ∨ (y ≤ 0 ∧ |x| ≤ 1) :=
sorry

end NUMINAMATH_GPT_set_of_points_l2289_228917


namespace NUMINAMATH_GPT_trajectory_of_P_l2289_228912

def point := ℝ × ℝ

-- Definitions for points A and F, and the circle equation
def A : point := (-1, 0)
def F (x y : ℝ) := (x - 1) ^ 2 + y ^ 2 = 16

-- Main theorem statement: proving the trajectory equation of point P
theorem trajectory_of_P : 
  (∀ (B : point), F B.1 B.2 → 
  (∃ P : point, ∃ (k : ℝ), (P.1 - B.1) * k = -(P.2 - B.2) ∧ (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0)) →
  (∃ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) :=
sorry

end NUMINAMATH_GPT_trajectory_of_P_l2289_228912


namespace NUMINAMATH_GPT_initial_earning_members_l2289_228969

theorem initial_earning_members (n T : ℕ)
  (h₁ : T = n * 782)
  (h₂ : T - 1178 = (n - 1) * 650) :
  n = 14 :=
by sorry

end NUMINAMATH_GPT_initial_earning_members_l2289_228969


namespace NUMINAMATH_GPT_number_of_correct_answers_l2289_228913

def total_questions := 30
def correct_points := 3
def incorrect_points := -1
def total_score := 78

theorem number_of_correct_answers (x : ℕ) :
  3 * x + incorrect_points * (total_questions - x) = total_score → x = 27 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_answers_l2289_228913


namespace NUMINAMATH_GPT_composite_integer_divisors_l2289_228921

theorem composite_integer_divisors (n : ℕ) (k : ℕ) (d : ℕ → ℕ) 
  (h_composite : 1 < n ∧ ¬Prime n)
  (h_divisors : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
  (h_distinct : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j)
  (h_range : d 1 = 1 ∧ d k = n)
  (h_ratio : ∀ i, 1 ≤ i ∧ i < k → (d (i + 1) - d i) = (i * (d 2 - d 1))) : n = 6 :=
by sorry

end NUMINAMATH_GPT_composite_integer_divisors_l2289_228921


namespace NUMINAMATH_GPT_student_B_speed_l2289_228922

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end NUMINAMATH_GPT_student_B_speed_l2289_228922


namespace NUMINAMATH_GPT_jack_received_more_emails_l2289_228970

-- Definitions representing the conditions
def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

-- The theorem statement
theorem jack_received_more_emails : afternoon_emails - morning_emails = 2 := 
by 
  sorry

end NUMINAMATH_GPT_jack_received_more_emails_l2289_228970


namespace NUMINAMATH_GPT_candies_left_l2289_228973

-- Defining the given conditions
def initial_candies : Nat := 30
def eaten_candies : Nat := 23

-- Define the target statement to prove
theorem candies_left : initial_candies - eaten_candies = 7 := by
  sorry

end NUMINAMATH_GPT_candies_left_l2289_228973


namespace NUMINAMATH_GPT_number_of_boys_l2289_228990

-- Definitions from the problem conditions
def trees : ℕ := 29
def trees_per_boy : ℕ := 3

-- Prove the number of boys is 10
theorem number_of_boys : (trees / trees_per_boy) + 1 = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_boys_l2289_228990


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2289_228930

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 + a 6 = 18) :
  S 10 = 90 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2289_228930


namespace NUMINAMATH_GPT_find_k_l2289_228978

theorem find_k
  (k x1 x2 : ℝ)
  (h1 : x1^2 - 3*x1 + k = 0)
  (h2 : x2^2 - 3*x2 + k = 0)
  (h3 : x1 = 2 * x2) :
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l2289_228978


namespace NUMINAMATH_GPT_baseball_cards_l2289_228900

theorem baseball_cards (cards_per_page new_cards pages : ℕ) (h1 : cards_per_page = 8) (h2 : new_cards = 3) (h3 : pages = 2) : 
  (pages * cards_per_page - new_cards = 13) := by
  sorry

end NUMINAMATH_GPT_baseball_cards_l2289_228900


namespace NUMINAMATH_GPT_correct_mean_of_values_l2289_228933

variable (n : ℕ) (mu_incorrect : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mu_correct : ℝ)

theorem correct_mean_of_values
  (h1 : n = 30)
  (h2 : mu_incorrect = 150)
  (h3 : incorrect_value = 135)
  (h4 : correct_value = 165)
  : mu_correct = 151 :=
by
  let S_incorrect := mu_incorrect * n
  let S_correct := S_incorrect - incorrect_value + correct_value
  let mu_correct := S_correct / n
  sorry

end NUMINAMATH_GPT_correct_mean_of_values_l2289_228933


namespace NUMINAMATH_GPT_students_with_two_skills_l2289_228988

theorem students_with_two_skills :
  ∀ (n_students n_chess n_puzzles n_code : ℕ),
  n_students = 120 →
  n_chess = n_students - 50 →
  n_puzzles = n_students - 75 →
  n_code = n_students - 40 →
  (n_chess + n_puzzles + n_code - n_students) = 75 :=
by 
  sorry

end NUMINAMATH_GPT_students_with_two_skills_l2289_228988


namespace NUMINAMATH_GPT_division_of_neg6_by_3_l2289_228926

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end NUMINAMATH_GPT_division_of_neg6_by_3_l2289_228926


namespace NUMINAMATH_GPT_sin_tan_condition_l2289_228962

theorem sin_tan_condition (x : ℝ) (h : Real.sin x = (Real.sqrt 2) / 2) : ¬((∀ x, Real.sin x = (Real.sqrt 2) / 2 → Real.tan x = 1) ∧ (∀ x, Real.tan x = 1 → Real.sin x = (Real.sqrt 2) / 2)) :=
sorry

end NUMINAMATH_GPT_sin_tan_condition_l2289_228962


namespace NUMINAMATH_GPT_inequality_problem_l2289_228955

theorem inequality_problem (x : ℝ) (h_denom : 2 * x^2 + 2 * x + 1 ≠ 0) : 
  -4 ≤ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l2289_228955


namespace NUMINAMATH_GPT_num_ways_4x4_proof_l2289_228975

-- Define a function that represents the number of ways to cut a 2x2 square
noncomputable def num_ways_2x2_cut : ℕ := 4

-- Define a function that represents the number of ways to cut a 3x3 square
noncomputable def num_ways_3x3_cut (ways_2x2 : ℕ) : ℕ :=
  ways_2x2 * 4

-- Define a function that represents the number of ways to cut a 4x4 square
noncomputable def num_ways_4x4_cut (ways_3x3 : ℕ) : ℕ :=
  ways_3x3 * 4

-- Prove the final number of ways to cut the 4x4 square into 3 L-shaped pieces and 1 small square
theorem num_ways_4x4_proof : num_ways_4x4_cut (num_ways_3x3_cut num_ways_2x2_cut) = 64 := by
  sorry

end NUMINAMATH_GPT_num_ways_4x4_proof_l2289_228975


namespace NUMINAMATH_GPT_find_intersection_sums_l2289_228966

noncomputable def cubic_expression (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 2
noncomputable def linear_expression (x : ℝ) : ℝ := -x / 2 + 1

theorem find_intersection_sums :
  (∃ x1 x2 x3 y1 y2 y3,
    cubic_expression x1 = linear_expression x1 ∧
    cubic_expression x2 = linear_expression x2 ∧
    cubic_expression x3 = linear_expression x3 ∧
    (x1 + x2 + x3 = 4) ∧ (y1 + y2 + y3 = 1)) :=
sorry

end NUMINAMATH_GPT_find_intersection_sums_l2289_228966


namespace NUMINAMATH_GPT_cartesian_equation_of_line_l2289_228929

theorem cartesian_equation_of_line (t x y : ℝ)
  (h1 : x = 1 + t / 2)
  (h2 : y = 2 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0 :=
sorry

end NUMINAMATH_GPT_cartesian_equation_of_line_l2289_228929


namespace NUMINAMATH_GPT_time_wandered_l2289_228947

-- Definitions and Hypotheses
def distance : ℝ := 4
def speed : ℝ := 2

-- Proof statement
theorem time_wandered : distance / speed = 2 := by
  sorry

end NUMINAMATH_GPT_time_wandered_l2289_228947


namespace NUMINAMATH_GPT_sum_numbers_l2289_228992

theorem sum_numbers :
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end NUMINAMATH_GPT_sum_numbers_l2289_228992


namespace NUMINAMATH_GPT_four_digit_number_conditions_l2289_228976

-- Define the needed values based on the problem conditions
def first_digit := 1
def second_digit := 3
def third_digit := 4
def last_digit := 9

def number := 1349

-- State the theorem
theorem four_digit_number_conditions :
  (second_digit = 3 * first_digit) ∧ 
  (last_digit = 3 * second_digit) ∧ 
  (number = 1349) :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_four_digit_number_conditions_l2289_228976


namespace NUMINAMATH_GPT_distance_between_x_intercepts_l2289_228952

theorem distance_between_x_intercepts 
  (s1 s2 : ℝ) (P : ℝ × ℝ)
  (h1 : s1 = 2) 
  (h2 : s2 = -4) 
  (hP : P = (8, 20)) :
  let l1_x_intercept := (0 - (20 - P.2)) / s1 + P.1
  let l2_x_intercept := (0 - (20 - P.2)) / s2 + P.1
  abs (l1_x_intercept - l2_x_intercept) = 15 := 
sorry

end NUMINAMATH_GPT_distance_between_x_intercepts_l2289_228952


namespace NUMINAMATH_GPT_degree_not_determined_from_characteristic_l2289_228934

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end NUMINAMATH_GPT_degree_not_determined_from_characteristic_l2289_228934


namespace NUMINAMATH_GPT_Liam_savings_after_trip_and_bills_l2289_228983

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end NUMINAMATH_GPT_Liam_savings_after_trip_and_bills_l2289_228983


namespace NUMINAMATH_GPT_sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l2289_228932

theorem sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0 :
  (9^25 + 11^25) % 100 = 0 := 
sorry

end NUMINAMATH_GPT_sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l2289_228932


namespace NUMINAMATH_GPT_probability_red_second_draw_l2289_228910

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_probability_red_second_draw_l2289_228910


namespace NUMINAMATH_GPT_total_cost_is_correct_l2289_228987

-- Define the costs as constants
def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52

-- Assert that the total cost is correct
theorem total_cost_is_correct : marbles_cost + football_cost + baseball_cost = 20.52 :=
by sorry

end NUMINAMATH_GPT_total_cost_is_correct_l2289_228987


namespace NUMINAMATH_GPT_solve_inequality_l2289_228958

open Set Real

def condition1 (x : ℝ) : Prop := 6 * x + 2 < (x + 2) ^ 2
def condition2 (x : ℝ) : Prop := (x + 2) ^ 2 < 8 * x + 4

theorem solve_inequality (x : ℝ) : condition1 x ∧ condition2 x ↔ x ∈ Ioo (2 + Real.sqrt 2) 4 := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2289_228958


namespace NUMINAMATH_GPT_avg_of_sequence_is_x_l2289_228941

noncomputable def sum_naturals (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem avg_of_sequence_is_x (x : ℝ) :
  let n := 100
  let sum := sum_naturals n
  (sum + x) / (n + 1) = 50 * x → 
  x = 5050 / 5049 :=
by
  intro n sum h
  exact sorry

end NUMINAMATH_GPT_avg_of_sequence_is_x_l2289_228941


namespace NUMINAMATH_GPT_smallest_initial_number_l2289_228943

theorem smallest_initial_number (N : ℕ) (h₁ : N ≤ 999) (h₂ : 27 * N - 240 ≥ 1000) : N = 46 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_initial_number_l2289_228943


namespace NUMINAMATH_GPT_second_percentage_increase_l2289_228909

theorem second_percentage_increase :
  ∀ (P : ℝ) (x : ℝ), (P * 1.30 * (1 + x) = P * 1.5600000000000001) → x = 0.2 :=
by
  intros P x h
  sorry

end NUMINAMATH_GPT_second_percentage_increase_l2289_228909


namespace NUMINAMATH_GPT_f_max_min_l2289_228911

def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom cauchy_f : ∀ x y : ℝ, f (x + y) = f x + f y
axiom less_than_zero : ∀ x : ℝ, x > 0 → f x < 0
axiom f_one : f 1 = -2

theorem f_max_min : (∀ x ∈ [-3, 3], f (-3) = 6 ∧ f 3 = -6) :=
by sorry

end NUMINAMATH_GPT_f_max_min_l2289_228911


namespace NUMINAMATH_GPT_number_of_footballs_l2289_228998

theorem number_of_footballs (x y : ℕ) (h1 : x + y = 20) (h2 : 6 * x + 3 * y = 96) : x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_footballs_l2289_228998


namespace NUMINAMATH_GPT_convert_deg_to_rad_l2289_228957

theorem convert_deg_to_rad (deg : ℝ) (h : deg = -630) : deg * (Real.pi / 180) = -7 * Real.pi / 2 :=
by
  rw [h]
  simp
  sorry

end NUMINAMATH_GPT_convert_deg_to_rad_l2289_228957


namespace NUMINAMATH_GPT_eggs_division_l2289_228994

theorem eggs_division (n_students n_eggs : ℕ) (h_students : n_students = 9) (h_eggs : n_eggs = 73):
  n_eggs / n_students = 8 ∧ n_eggs % n_students = 1 :=
by
  rw [h_students, h_eggs]
  exact ⟨rfl, rfl⟩

end NUMINAMATH_GPT_eggs_division_l2289_228994


namespace NUMINAMATH_GPT_find_particular_number_l2289_228903

def particular_number (x : ℕ) : Prop :=
  (2 * (67 - (x / 23))) = 102

theorem find_particular_number : particular_number 2714 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_particular_number_l2289_228903


namespace NUMINAMATH_GPT_f_one_zero_range_of_a_l2289_228965

variable (f : ℝ → ℝ) (a : ℝ)

-- Conditions
def odd_function : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x
def increasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y
def f_neg_one_zero : Prop := f (-1) = 0
def f_a_minus_half_neg : Prop := f (a - 1/2) < 0

-- Questions
theorem f_one_zero (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) : f 1 = 0 := 
sorry

theorem range_of_a (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) (h4 : f_a_minus_half_neg f a) :
  1/2 < a ∧ a < 3/2 ∨ a < -1/2 :=
sorry

end NUMINAMATH_GPT_f_one_zero_range_of_a_l2289_228965


namespace NUMINAMATH_GPT_quadratic_equation_root_zero_l2289_228946

/-- Given that x = -3 is a root of the quadratic equation x^2 + 3x + k = 0,
    prove that the other root of the equation is 0 and k = 0. -/
theorem quadratic_equation_root_zero (k : ℝ) (h : -3^2 + 3 * -3 + k = 0) :
  (∀ t : ℝ, t^2 + 3 * t + k = 0 → t = 0) ∧ k = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_root_zero_l2289_228946


namespace NUMINAMATH_GPT_city_council_vote_l2289_228960

theorem city_council_vote :
  ∀ (x y x' y' m : ℕ),
    x + y = 350 →
    y > x →
    y - x = m →
    x' - y' = 2 * m →
    x' + y' = 350 →
    x' = (10 * y) / 9 →
    x' - x = 66 :=
by
  intros x y x' y' m h1 h2 h3 h4 h5 h6
  -- proof goes here
  sorry

end NUMINAMATH_GPT_city_council_vote_l2289_228960


namespace NUMINAMATH_GPT_find_x_l2289_228951

theorem find_x (a x : ℤ) (h1 : -6 * a^2 = x * (4 * a + 2)) (h2 : a = 1) : x = -1 :=
sorry

end NUMINAMATH_GPT_find_x_l2289_228951


namespace NUMINAMATH_GPT_trajectory_of_moving_circle_l2289_228974

-- Define the two given circles C1 and C2
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 1}
def C2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 81}

-- Define a moving circle P with center P_center and radius r
structure Circle (α : Type) := 
(center : α × α) 
(radius : ℝ)

def isExternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (P.radius + 1)^2

def isInternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (9 - P.radius)^2

-- Formulate the problem statement
theorem trajectory_of_moving_circle :
  ∀ P : Circle ℝ, 
  isExternallyTangentTo P C1 → 
  isInternallyTangentTo P C2 → 
  (P.center.1^2 / 25 + P.center.2^2 / 21 = 1) := 
sorry

end NUMINAMATH_GPT_trajectory_of_moving_circle_l2289_228974


namespace NUMINAMATH_GPT_FC_value_l2289_228996

variables (DC CB AB AD ED FC CA BD : ℝ)

-- Set the conditions as variables
variable (h_DC : DC = 10)
variable (h_CB : CB = 12)
variable (h_AB : AB = (1/3) * AD)
variable (h_ED : ED = (2/3) * AD)
variable (h_BD : BD = 22)
variable (BD_eq : BD = DC + CB)
variable (CA_eq : CA = CB + AB)

-- Define the relationship for the final result
def find_FC (DC CB AB AD ED FC CA BD : ℝ) := FC = (ED * CA) / AD

-- The main statement to be proven
theorem FC_value : 
  find_FC DC CB AB (33 : ℝ) (22 : ℝ) FC (23 : ℝ) (22 : ℝ) → 
  FC = (506/33) :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_FC_value_l2289_228996


namespace NUMINAMATH_GPT_oranges_in_total_l2289_228927

def number_of_boxes := 3
def oranges_per_box := 8
def total_oranges := 24

theorem oranges_in_total : number_of_boxes * oranges_per_box = total_oranges := 
by {
  -- sorry skips the proof part
  sorry 
}

end NUMINAMATH_GPT_oranges_in_total_l2289_228927


namespace NUMINAMATH_GPT_maria_strawberries_l2289_228944

theorem maria_strawberries (S : ℕ) :
  (21 = 8 + 9 + S) → (S = 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_maria_strawberries_l2289_228944


namespace NUMINAMATH_GPT_population_weight_of_500_students_l2289_228948

-- Definitions
def number_of_students : ℕ := 500
def number_of_selected_students : ℕ := 60

-- Conditions
def condition1 := number_of_students = 500
def condition2 := number_of_selected_students = 60

-- Statement
theorem population_weight_of_500_students : 
  condition1 → condition2 → 
  (∃ p, p = "the weight of the 500 students") := by
  intros _ _
  existsi "the weight of the 500 students"
  rfl

end NUMINAMATH_GPT_population_weight_of_500_students_l2289_228948


namespace NUMINAMATH_GPT_solve_for_a_l2289_228993

theorem solve_for_a (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5 * a * x + a) = x^3 + (1 - 5 * a) * x^2 - 4 * a * x + a) →
  (1 - 5 * a = 0) →
  a = 1 / 5 := 
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_solve_for_a_l2289_228993


namespace NUMINAMATH_GPT_jason_cuts_lawns_l2289_228997

theorem jason_cuts_lawns 
  (time_per_lawn: ℕ)
  (total_cutting_time_hours: ℕ)
  (total_cutting_time_minutes: ℕ)
  (total_yards_cut: ℕ) : 
  time_per_lawn = 30 → 
  total_cutting_time_hours = 8 → 
  total_cutting_time_minutes = total_cutting_time_hours * 60 → 
  total_yards_cut = total_cutting_time_minutes / time_per_lawn → 
  total_yards_cut = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jason_cuts_lawns_l2289_228997


namespace NUMINAMATH_GPT_problem1_problem2_l2289_228972

-- Problem 1: (-3xy)² * 4x² = 36x⁴y²
theorem problem1 (x y : ℝ) : ((-3 * x * y) ^ 2) * (4 * x ^ 2) = 36 * x ^ 4 * y ^ 2 := by
  sorry

-- Problem 2: (x + 2)(2x - 3) = 2x² + x - 6
theorem problem2 (x : ℝ) : (x + 2) * (2 * x - 3) = 2 * x ^ 2 + x - 6 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2289_228972


namespace NUMINAMATH_GPT_M_diff_N_l2289_228931

def A : Set ℝ := sorry
def B : Set ℝ := sorry

def M := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Definition of set subtraction
def set_diff (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∉ B}

-- Given problem statement
theorem M_diff_N : set_diff M N = {x : ℝ | -3 ≤ x ∧ x < 0} := 
by
  sorry

end NUMINAMATH_GPT_M_diff_N_l2289_228931


namespace NUMINAMATH_GPT_num_triangles_with_perimeter_9_l2289_228950

theorem num_triangles_with_perimeter_9 : 
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 6 ∧ 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 9 ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ a ≤ b ∧ b ≤ c) := 
sorry

end NUMINAMATH_GPT_num_triangles_with_perimeter_9_l2289_228950


namespace NUMINAMATH_GPT_unique_solution_abs_eq_l2289_228907

theorem unique_solution_abs_eq : ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_abs_eq_l2289_228907


namespace NUMINAMATH_GPT_dot_product_is_one_l2289_228964

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (-1, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

theorem dot_product_is_one : dot_product vec_a vec_b = 1 :=
by sorry

end NUMINAMATH_GPT_dot_product_is_one_l2289_228964


namespace NUMINAMATH_GPT_length_of_LM_l2289_228986

-- Definitions of the conditions
variable (P Q R L M : Type)
variable (b : Real) (PR_area : Real) (PR_base : Real)
variable (PR_base_eq : PR_base = 15)
variable (crease_parallel : Parallel L M)
variable (projected_area_fraction : Real)
variable (projected_area_fraction_eq : projected_area_fraction = 0.25 * PR_area)

-- Theorem statement to prove the length of LM
theorem length_of_LM : ∀ (LM_length : Real), (LM_length = 7.5) :=
sorry

end NUMINAMATH_GPT_length_of_LM_l2289_228986


namespace NUMINAMATH_GPT_walnuts_amount_l2289_228989

theorem walnuts_amount (w : ℝ) (total_nuts : ℝ) (almonds : ℝ) (h1 : total_nuts = 0.5) (h2 : almonds = 0.25) (h3 : w + almonds = total_nuts) : w = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_walnuts_amount_l2289_228989


namespace NUMINAMATH_GPT_range_of_m_l2289_228923

variable (x y m : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : 2/x + 1/y = 1)
variable (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m)

theorem range_of_m (h1 : 0 < x) (h2 : 0 < y) (h3 : 2/x + 1/y = 1) (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m) : -4 < m ∧ m < 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2289_228923


namespace NUMINAMATH_GPT_z_neq_5_for_every_k_l2289_228940

theorem z_neq_5_for_every_k (z : ℕ) (h₁ : z = 5) :
  ¬ (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (∃ m, n ^ 9 % 10 ^ k = z * (10 ^ m))) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_z_neq_5_for_every_k_l2289_228940


namespace NUMINAMATH_GPT_day_90_N_minus_1_is_Thursday_l2289_228902

/-- 
    Given that the 150th day of year N is a Sunday, 
    and the 220th day of year N+2 is also a Sunday,
    prove that the 90th day of year N-1 is a Thursday.
-/
theorem day_90_N_minus_1_is_Thursday (N : ℕ)
    (h1 : (150 % 7 = 0))  -- 150th day of year N is Sunday
    (h2 : (220 % 7 = 0))  -- 220th day of year N + 2 is Sunday
    : ((90 + 366) % 7 = 4) := -- 366 days in a leap year (N-1), 90th day modulo 7 is Thursday
by
  sorry

end NUMINAMATH_GPT_day_90_N_minus_1_is_Thursday_l2289_228902


namespace NUMINAMATH_GPT_best_model_based_on_R_squared_l2289_228916

theorem best_model_based_on_R_squared:
  ∀ (R2_1 R2_2 R2_3 R2_4: ℝ), 
  R2_1 = 0.98 → R2_2 = 0.80 → R2_3 = 0.54 → R2_4 = 0.35 → 
  R2_1 ≥ R2_2 ∧ R2_1 ≥ R2_3 ∧ R2_1 ≥ R2_4 :=
by
  intros R2_1 R2_2 R2_3 R2_4 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_best_model_based_on_R_squared_l2289_228916


namespace NUMINAMATH_GPT_solve_x_y_l2289_228915

theorem solve_x_y (x y : ℝ) (h1 : x^2 + y^2 = 16 * x - 10 * y + 14) (h2 : x - y = 6) : 
  x + y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_x_y_l2289_228915


namespace NUMINAMATH_GPT_highest_possible_N_l2289_228924

/--
In a football tournament with 15 teams, each team played exactly once against every other team.
A win earns 3 points, a draw earns 1 point, and a loss earns 0 points.
We need to prove that the highest possible integer \( N \) such that there are at least 6 teams with at least \( N \) points is 34.
-/
theorem highest_possible_N : 
  ∃ (N : ℤ) (teams : Fin 15 → ℤ) (successfulTeams : Fin 6 → Fin 15),
    (∀ i j, i ≠ j → teams i + teams j ≤ 207) ∧ 
    (∀ k, k < 6 → teams (successfulTeams k) ≥ 34) ∧ 
    (∀ k, 0 ≤ teams k) ∧ 
    N = 34 := sorry

end NUMINAMATH_GPT_highest_possible_N_l2289_228924


namespace NUMINAMATH_GPT_cricket_player_innings_l2289_228980

theorem cricket_player_innings (n : ℕ) (h1 : 35 * n = 35 * n) (h2 : 35 * n + 79 = 39 * (n + 1)) : n = 10 := by
  sorry

end NUMINAMATH_GPT_cricket_player_innings_l2289_228980


namespace NUMINAMATH_GPT_find_ellipse_parameters_l2289_228904

noncomputable def ellipse_centers_and_axes (F1 F2 : ℝ × ℝ) (d : ℝ) (tangent_slope : ℝ) :=
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  let a := d / 2
  let c := (Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  (h, k, a, b)

theorem find_ellipse_parameters :
  let F1 := (-1, 1)
  let F2 := (5, 1)
  let d := 10
  let tangent_at_x_axis_slope := 1
  let (h, k, a, b) := ellipse_centers_and_axes F1 F2 d tangent_at_x_axis_slope
  h + k + a + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_ellipse_parameters_l2289_228904


namespace NUMINAMATH_GPT_slope_of_line_AB_l2289_228925

-- Define the points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (2, 4)

-- State the proposition that we need to prove
theorem slope_of_line_AB :
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_AB_l2289_228925


namespace NUMINAMATH_GPT_bees_multiple_l2289_228953

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_bees_multiple_l2289_228953


namespace NUMINAMATH_GPT_distinct_triangles_from_tetrahedron_l2289_228945

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_distinct_triangles_from_tetrahedron_l2289_228945


namespace NUMINAMATH_GPT_simplify_expression_simplify_and_evaluate_evaluate_expression_l2289_228939

theorem simplify_expression (a b : ℝ) : 8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := 
by sorry

theorem simplify_and_evaluate (x y : ℝ) (h : x + y = 1/2) : 
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := 
by sorry

theorem evaluate_expression (x y : ℝ) (h : x^2 - 2 * y = 4) : -3 * x^2 + 6 * y + 2 = -10 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_simplify_and_evaluate_evaluate_expression_l2289_228939


namespace NUMINAMATH_GPT_domain_of_f_l2289_228967

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (abs (x + 1) - 5)

theorem domain_of_f :
  {x : ℝ | x - 3 ≥ 0 ∧ abs (x + 1) - 5 ≠ 0} = {x : ℝ | (3 ≤ x ∧ x < 4) ∨ (4 < x)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2289_228967
