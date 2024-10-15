import Mathlib

namespace NUMINAMATH_GPT_problem1_correct_problem2_correct_l1169_116986

noncomputable def problem1 := 5 + (-6) + 3 - 8 - (-4)
noncomputable def problem2 := -2^2 - 3 * (-1)^3 - (-1) / (-1 / 2)^2

theorem problem1_correct : problem1 = -2 := by
  rw [problem1]
  sorry

theorem problem2_correct : problem2 = 3 := by
  rw [problem2]
  sorry

end NUMINAMATH_GPT_problem1_correct_problem2_correct_l1169_116986


namespace NUMINAMATH_GPT_henry_age_l1169_116921

theorem henry_age (H J : ℕ) (h1 : H + J = 43) (h2 : H - 5 = 2 * (J - 5)) : H = 27 :=
by
  -- This is where we would prove the theorem based on the given conditions
  sorry

end NUMINAMATH_GPT_henry_age_l1169_116921


namespace NUMINAMATH_GPT_Johann_oranges_l1169_116991

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end NUMINAMATH_GPT_Johann_oranges_l1169_116991


namespace NUMINAMATH_GPT_find_A_l1169_116941

theorem find_A (A B C D: ℕ) (h1: A ≠ B) (h2: A ≠ C) (h3: A ≠ D) (h4: B ≠ C) (h5: B ≠ D) (h6: C ≠ D)
  (hAB: A * B = 72) (hCD: C * D = 72) (hDiff: A - B = C + D + 2) : A = 6 :=
sorry

end NUMINAMATH_GPT_find_A_l1169_116941


namespace NUMINAMATH_GPT_arun_speed_ratio_l1169_116995

namespace SpeedRatio

variables (V_a V_n V_a' : ℝ)
variable (distance : ℝ := 30)
variable (original_speed_Arun : ℝ := 5)
variable (time_Arun time_Anil time_Arun_new_speed : ℝ)

-- Conditions
theorem arun_speed_ratio :
  V_a = original_speed_Arun →
  time_Arun = distance / V_a →
  time_Anil = distance / V_n →
  time_Arun = time_Anil + 2 →
  time_Arun_new_speed = distance / V_a' →
  time_Arun_new_speed = time_Anil - 1 →
  V_a' / V_a = 2 := 
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1] at *
  sorry

end SpeedRatio

end NUMINAMATH_GPT_arun_speed_ratio_l1169_116995


namespace NUMINAMATH_GPT_average_mark_second_class_l1169_116902

theorem average_mark_second_class
  (avg_mark_class1 : ℝ)
  (num_students_class1 : ℕ)
  (num_students_class2 : ℕ)
  (combined_avg_mark : ℝ) 
  (total_students : ℕ)
  (total_marks_combined : ℝ) :
  avg_mark_class1 * num_students_class1 + x * num_students_class2 = total_marks_combined →
  num_students_class1 + num_students_class2 = total_students →
  combined_avg_mark * total_students = total_marks_combined →
  avg_mark_class1 = 40 →
  num_students_class1 = 30 →
  num_students_class2 = 50 →
  combined_avg_mark = 58.75 →
  total_students = 80 →
  total_marks_combined = 4700 →
  x = 70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_mark_second_class_l1169_116902


namespace NUMINAMATH_GPT_cartons_being_considered_l1169_116972

-- Definitions based on conditions
def packs_per_box : ℕ := 10
def boxes_per_carton : ℕ := 12
def price_per_pack : ℕ := 1
def total_cost : ℕ := 1440

-- Calculate total cost per carton
def cost_per_carton : ℕ := boxes_per_carton * packs_per_box * price_per_pack

-- Formulate the main theorem
theorem cartons_being_considered : (total_cost / cost_per_carton) = 12 :=
by
  -- The relevant steps would go here, but we're only providing the statement
  sorry

end NUMINAMATH_GPT_cartons_being_considered_l1169_116972


namespace NUMINAMATH_GPT_suresh_work_hours_l1169_116957

theorem suresh_work_hours (x : ℝ) (h : x / 15 + 8 / 20 = 1) : x = 9 :=
by 
    sorry

end NUMINAMATH_GPT_suresh_work_hours_l1169_116957


namespace NUMINAMATH_GPT_solution_set_inequality_l1169_116905

theorem solution_set_inequality
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → ax^2 + bx + c > 0) :
  ∃ s : Set ℝ, s = {x | (1/2) < x ∧ x < 1} ∧ ∀ x : ℝ, x ∈ s → cx^2 + bx + a > 0 := by
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1169_116905


namespace NUMINAMATH_GPT_european_postcards_cost_l1169_116979

def price_per_postcard (country : String) : ℝ :=
  if country = "Italy" ∨ country = "Germany" then 0.10
  else if country = "Canada" then 0.07
  else if country = "Mexico" then 0.08
  else 0.0

def num_postcards (decade : Nat) (country : String) : Nat :=
  if decade = 1950 then
    if country = "Italy" then 10
    else if country = "Germany" then 5
    else if country = "Canada" then 8
    else if country = "Mexico" then 12
    else 0
  else if decade = 1960 then
    if country = "Italy" then 16
    else if country = "Germany" then 12
    else if country = "Canada" then 10
    else if country = "Mexico" then 15
    else 0
  else if decade = 1970 then
    if country = "Italy" then 12
    else if country = "Germany" then 18
    else if country = "Canada" then 13
    else if country = "Mexico" then 9
    else 0
  else 0

def total_cost (country : String) : ℝ :=
  (price_per_postcard country) * (num_postcards 1950 country)
  + (price_per_postcard country) * (num_postcards 1960 country)
  + (price_per_postcard country) * (num_postcards 1970 country)

theorem european_postcards_cost : total_cost "Italy" + total_cost "Germany" = 7.30 := by
  sorry

end NUMINAMATH_GPT_european_postcards_cost_l1169_116979


namespace NUMINAMATH_GPT_angle_AOC_is_minus_150_l1169_116900

-- Define the conditions.
def rotate_counterclockwise (angle1 : Int) (angle2 : Int) : Int :=
  angle1 + angle2

-- The initial angle starts at 0°, rotates 120° counterclockwise, and then 270° clockwise
def angle_OA := 0
def angle_OB := rotate_counterclockwise angle_OA 120
def angle_OC := rotate_counterclockwise angle_OB (-270)

-- The theorem stating the resulting angle between OA and OC.
theorem angle_AOC_is_minus_150 : angle_OC = -150 := by
  sorry

end NUMINAMATH_GPT_angle_AOC_is_minus_150_l1169_116900


namespace NUMINAMATH_GPT_process_terminates_with_one_element_in_each_list_final_elements_are_different_l1169_116937

-- Define the initial lists
def List1 := [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
def List2 := [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

-- Predicate to state the termination of the process with exactly one element in each list
theorem process_terminates_with_one_element_in_each_list (List1 List2 : List ℕ):
  ∃ n m, List.length List1 = n ∧ List.length List2 = m ∧ (n = 1 ∧ m = 1) :=
sorry

-- Predicate to state that the final elements in the lists are different
theorem final_elements_are_different (List1 List2 : List ℕ) :
  ∀ a b, a ∈ List1 → b ∈ List2 → (a % 5 = 1 ∧ b % 5 = 4) → a ≠ b :=
sorry

end NUMINAMATH_GPT_process_terminates_with_one_element_in_each_list_final_elements_are_different_l1169_116937


namespace NUMINAMATH_GPT_quadratic_complete_square_l1169_116946

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 + 2 * x + 3) = ((x + 1)^2 + 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l1169_116946


namespace NUMINAMATH_GPT_rect_garden_width_l1169_116976

theorem rect_garden_width (w l : ℝ) (h1 : l = 3 * w) (h2 : l * w = 768) : w = 16 := by
  sorry

end NUMINAMATH_GPT_rect_garden_width_l1169_116976


namespace NUMINAMATH_GPT_min_value_expression_l1169_116950

theorem min_value_expression (x y z : ℝ) (h1 : -1/2 < x ∧ x < 1/2) (h2 : -1/2 < y ∧ y < 1/2) (h3 : -1/2 < z ∧ z < 1/2) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) + 1 / 2) ≥ 2.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_expression_l1169_116950


namespace NUMINAMATH_GPT_subtract_some_number_l1169_116915

theorem subtract_some_number
  (x : ℤ)
  (h : 913 - x = 514) :
  514 - x = 115 :=
by {
  sorry
}

end NUMINAMATH_GPT_subtract_some_number_l1169_116915


namespace NUMINAMATH_GPT_original_number_l1169_116983

theorem original_number (N m a b c : ℕ) (hN : N = 3306) 
  (h_eq : 3306 + m = 222 * (a + b + c)) 
  (hm_digits : m = 100 * a + 10 * b + c) 
  (h1 : a + b + c = 15) 
  (h2 : ∃ (a b c : ℕ), a + b + c = 15 ∧ 100 * a + 10 * b + c = 78): 
  100 * a + 10 * b + c = 753 := 
by sorry

end NUMINAMATH_GPT_original_number_l1169_116983


namespace NUMINAMATH_GPT_factorize_expr_l1169_116975

-- Define the variables a and b as elements of an arbitrary ring
variables {R : Type*} [CommRing R] (a b : R)

-- Prove the factorization identity
theorem factorize_expr : a^2 * b - b = b * (a + 1) * (a - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l1169_116975


namespace NUMINAMATH_GPT_number_of_insects_l1169_116965

-- Conditions
def total_legs : ℕ := 30
def legs_per_insect : ℕ := 6

-- Theorem statement
theorem number_of_insects (total_legs legs_per_insect : ℕ) : 
  total_legs / legs_per_insect = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_insects_l1169_116965


namespace NUMINAMATH_GPT_number_of_solutions_l1169_116924

theorem number_of_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℕ,
    (x < 10^2006) ∧ ((x * (x - 1)) % 10^2006 = 0) → x ≤ n :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l1169_116924


namespace NUMINAMATH_GPT_pie_remaining_portion_l1169_116967

theorem pie_remaining_portion (carlos_portion maria_portion remaining_portion : ℝ)
  (h1 : carlos_portion = 0.6) 
  (h2 : remaining_portion = 1 - carlos_portion)
  (h3 : maria_portion = 0.5 * remaining_portion) :
  remaining_portion - maria_portion = 0.2 := 
by
  sorry

end NUMINAMATH_GPT_pie_remaining_portion_l1169_116967


namespace NUMINAMATH_GPT_suff_but_not_necess_condition_l1169_116971

theorem suff_but_not_necess_condition (a b : ℝ) (h1 : a < 0) (h2 : -1 < b ∧ b < 0) : a + a * b < 0 :=
  sorry

end NUMINAMATH_GPT_suff_but_not_necess_condition_l1169_116971


namespace NUMINAMATH_GPT_spatial_relationship_l1169_116934

variables {a b c : Type}          -- Lines a, b, c
variables {α β γ : Type}          -- Planes α, β, γ

-- Parallel relationship between planes
def plane_parallel (α β : Type) : Prop := sorry
-- Perpendicular relationship between planes
def plane_perpendicular (α β : Type) : Prop := sorry
-- Parallel relationship between lines and planes
def line_parallel_plane (a α : Type) : Prop := sorry
-- Perpendicular relationship between lines and planes
def line_perpendicular_plane (a α : Type) : Prop := sorry
-- Parallel relationship between lines
def line_parallel (a b : Type) : Prop := sorry
-- The angle formed by a line and a plane
def angle (a : Type) (α : Type) : Type := sorry

theorem spatial_relationship :
  (plane_parallel α γ ∧ plane_parallel β γ → plane_parallel α β) ∧
  ¬ (line_parallel_plane a α ∧ line_parallel_plane b α → line_parallel a b) ∧
  ¬ (plane_perpendicular α γ ∧ plane_perpendicular β γ → plane_parallel α β) ∧
  ¬ (line_perpendicular_plane a c ∧ line_perpendicular_plane b c → line_parallel a b) ∧
  (line_parallel a b ∧ plane_parallel α β → angle a α = angle b β) :=
sorry

end NUMINAMATH_GPT_spatial_relationship_l1169_116934


namespace NUMINAMATH_GPT_systematic_sampling_result_l1169_116901

theorem systematic_sampling_result :
  ∀ (total_students sample_size selected1_16 selected33_48 : ℕ),
  total_students = 800 →
  sample_size = 50 →
  selected1_16 = 11 →
  selected33_48 = selected1_16 + 32 →
  selected33_48 = 43 := by
  intros
  sorry

end NUMINAMATH_GPT_systematic_sampling_result_l1169_116901


namespace NUMINAMATH_GPT_entertainment_team_count_l1169_116919

theorem entertainment_team_count 
  (total_members : ℕ)
  (singers : ℕ) 
  (dancers : ℕ) 
  (prob_both_sing_dance_gt_0 : ℚ)
  (sing_count : singers = 2)
  (dance_count : dancers = 5)
  (prob_condition : prob_both_sing_dance_gt_0 = 7/10) :
  total_members = 5 := 
by 
  sorry

end NUMINAMATH_GPT_entertainment_team_count_l1169_116919


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1169_116935

variable {a : ℝ}

/-- Given that the length of the real axis of the hyperbola x^2/a^2 - y^2 = 1 (a > 0) is 1,
    we want to prove that the equation of its asymptotes is y = ± 2x. -/
theorem asymptotes_of_hyperbola (ha : a > 0) (h_len : 2 * a = 1) :
  ∀ x y : ℝ, (y = 2 * x) ∨ (y = -2 * x) :=
by {
  sorry
}

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1169_116935


namespace NUMINAMATH_GPT_unique_solution_set_l1169_116923

theorem unique_solution_set :
  {a : ℝ | ∃ x : ℝ, (x+a)/(x^2-1) = 1 ∧ 
                    (∀ y : ℝ, (y+a)/(y^2-1) = 1 → y = x)} 
  = {-1, 1, -5/4} :=
sorry

end NUMINAMATH_GPT_unique_solution_set_l1169_116923


namespace NUMINAMATH_GPT_pyramid_surface_area_l1169_116964

noncomputable def total_surface_area_of_pyramid (a b : ℝ) (theta : ℝ) (height : ℝ) : ℝ :=
  let base_area := a * b * Real.sin theta
  let slant_height := Real.sqrt (height ^ 2 + (a / 2) ^ 2)
  let lateral_area := 4 * (1 / 2 * a * slant_height)
  base_area + lateral_area

theorem pyramid_surface_area :
  total_surface_area_of_pyramid 12 14 (Real.pi / 3) 15 = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_GPT_pyramid_surface_area_l1169_116964


namespace NUMINAMATH_GPT_Joe_speed_first_part_l1169_116987

theorem Joe_speed_first_part
  (dist1 dist2 : ℕ)
  (speed2 avg_speed total_distance total_time : ℕ)
  (h1 : dist1 = 180)
  (h2 : dist2 = 120)
  (h3 : speed2 = 40)
  (h4 : avg_speed = 50)
  (h5 : total_distance = dist1 + dist2)
  (h6 : total_distance = 300)
  (h7 : total_time = total_distance / avg_speed)
  (h8 : total_time = 6) :
  ∃ v : ℕ, (dist1 / v + dist2 / speed2 = total_time) ∧ v = 60 :=
by
  sorry

end NUMINAMATH_GPT_Joe_speed_first_part_l1169_116987


namespace NUMINAMATH_GPT_shorter_piece_length_l1169_116999

noncomputable def total_length : ℝ := 140
noncomputable def ratio : ℝ := 2 / 5

theorem shorter_piece_length (x : ℝ) (y : ℝ) (h1 : x + y = total_length) (h2 : x = ratio * y) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l1169_116999


namespace NUMINAMATH_GPT_sum_of_primes_l1169_116943

theorem sum_of_primes (a b c : ℕ) (h₁ : Nat.Prime a) (h₂ : Nat.Prime b) (h₃ : Nat.Prime c) (h₄ : b + c = 13) (h₅ : c^2 - a^2 = 72) :
  a + b + c = 20 := 
sorry

end NUMINAMATH_GPT_sum_of_primes_l1169_116943


namespace NUMINAMATH_GPT_lcm_5_7_10_14_l1169_116931

theorem lcm_5_7_10_14 : Nat.lcm (Nat.lcm 5 7) (Nat.lcm 10 14) = 70 := by
  sorry

end NUMINAMATH_GPT_lcm_5_7_10_14_l1169_116931


namespace NUMINAMATH_GPT_number_of_strings_is_multiple_of_3_l1169_116906

theorem number_of_strings_is_multiple_of_3 (N : ℕ) :
  (∀ (avg_total avg_one_third avg_two_third : ℚ), 
    avg_total = 80 ∧ avg_one_third = 70 ∧ avg_two_third = 85 →
    (∃ k : ℕ, N = 3 * k)) :=
by
  intros avg_total avg_one_third avg_two_third h
  sorry

end NUMINAMATH_GPT_number_of_strings_is_multiple_of_3_l1169_116906


namespace NUMINAMATH_GPT_dave_deleted_apps_l1169_116985

theorem dave_deleted_apps : 
  ∀ (a_initial a_left a_deleted : ℕ), a_initial = 16 → a_left = 5 → a_deleted = a_initial - a_left → a_deleted = 11 :=
by
  intros a_initial a_left a_deleted h_initial h_left h_deleted
  rw [h_initial, h_left] at h_deleted
  exact h_deleted

end NUMINAMATH_GPT_dave_deleted_apps_l1169_116985


namespace NUMINAMATH_GPT_fraction_proof_l1169_116994

theorem fraction_proof (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_proof_l1169_116994


namespace NUMINAMATH_GPT_factorize_a_cube_minus_nine_a_l1169_116940

theorem factorize_a_cube_minus_nine_a (a : ℝ) : a^3 - 9 * a = a * (a + 3) * (a - 3) :=
by sorry

end NUMINAMATH_GPT_factorize_a_cube_minus_nine_a_l1169_116940


namespace NUMINAMATH_GPT_det_B_squared_minus_3B_l1169_116955

theorem det_B_squared_minus_3B (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B = ![![2, 4], ![3, 2]]) : 
  Matrix.det (B * B - 3 • B) = 88 := by
  sorry

end NUMINAMATH_GPT_det_B_squared_minus_3B_l1169_116955


namespace NUMINAMATH_GPT_compute_expression_l1169_116933

theorem compute_expression : 2 + ((4 * 3 - 2) / 2 * 3) + 5 = 22 :=
by
  -- Place the solution steps if needed
  sorry

end NUMINAMATH_GPT_compute_expression_l1169_116933


namespace NUMINAMATH_GPT_cos_double_angle_sin_double_angle_l1169_116929

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 :=
by sorry

theorem sin_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.sin (2 * θ) = (Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_GPT_cos_double_angle_sin_double_angle_l1169_116929


namespace NUMINAMATH_GPT_number_of_valid_integers_l1169_116918

def count_valid_numbers : Nat :=
  let one_digit_count : Nat := 6
  let two_digit_count : Nat := 6 * 6
  let three_digit_count : Nat := 6 * 6 * 6
  one_digit_count + two_digit_count + three_digit_count

theorem number_of_valid_integers :
  count_valid_numbers = 258 :=
sorry

end NUMINAMATH_GPT_number_of_valid_integers_l1169_116918


namespace NUMINAMATH_GPT_work_completion_time_equal_l1169_116984

/-- Define the individual work rates of a, b, c, and d --/
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 6
def work_rate_c : ℚ := 1 / 12
def work_rate_d : ℚ := 1 / 10

/-- Define the combined work rate when they work together --/
def combined_work_rate : ℚ := work_rate_a + work_rate_b + work_rate_c + work_rate_d

/-- Define total work as one unit divided by the combined work rate --/
def total_days_to_complete : ℚ := 1 / combined_work_rate

/-- Main theorem to prove: When a, b, c, and d work together, they complete the work in 120/47 days --/
theorem work_completion_time_equal : total_days_to_complete = 120 / 47 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_equal_l1169_116984


namespace NUMINAMATH_GPT_problem_statement_l1169_116930

-- Define operations "※" and "#"
def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

-- Define the proof statement
theorem problem_statement : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1169_116930


namespace NUMINAMATH_GPT_field_width_l1169_116938

theorem field_width (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 288) : W = 60 :=
by
  sorry

end NUMINAMATH_GPT_field_width_l1169_116938


namespace NUMINAMATH_GPT_max_value_of_sum_on_ellipse_l1169_116997

theorem max_value_of_sum_on_ellipse (x y : ℝ) (h : x^2 / 3 + y^2 = 1) : x + y ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_sum_on_ellipse_l1169_116997


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_gt_zero_l1169_116939

theorem necessary_but_not_sufficient_for_gt_zero (x : ℝ) : 
  x ≠ 0 → (¬ (x ≤ 0)) := by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_gt_zero_l1169_116939


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1169_116925

theorem simplify_expr1 : (-4)^2023 * (-0.25)^2024 = -0.25 :=
by 
  sorry

theorem simplify_expr2 : 23 * (-4 / 11) + (-5 / 11) * 23 - 23 * (2 / 11) = -23 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1169_116925


namespace NUMINAMATH_GPT_locus_eqn_l1169_116914

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  ∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)

theorem locus_eqn (a b : ℝ) : 
  locus_of_centers a b ↔ 3 * a^2 + b^2 + 44 * a + 121 = 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_locus_eqn_l1169_116914


namespace NUMINAMATH_GPT_find_monthly_fee_l1169_116990

-- Definitions from conditions
def monthly_fee (total_bill : ℝ) (cost_per_minute : ℝ) (minutes_used : ℝ) : ℝ :=
  total_bill - cost_per_minute * minutes_used

-- Theorem stating the question
theorem find_monthly_fee :
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  total_bill - cost_per_minute * minutes_used = 5.00 :=
by
  -- Definition of variables used in the theorem
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  
  -- The statement of the theorem and leaving the proof as an exercise
  show total_bill - cost_per_minute * minutes_used = 5.00
  sorry

end NUMINAMATH_GPT_find_monthly_fee_l1169_116990


namespace NUMINAMATH_GPT_length_of_second_platform_l1169_116980

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_platform_l1169_116980


namespace NUMINAMATH_GPT_find_cos_A_l1169_116944

noncomputable def cos_A_of_third_quadrant : Real :=
-3 / 5

theorem find_cos_A (A : Real) (h1 : A ∈ Set.Icc (π) (3 * π / 2)) 
  (h2 : Real.sin A = 4 / 5) : Real.cos A = -3 / 5 := 
sorry

end NUMINAMATH_GPT_find_cos_A_l1169_116944


namespace NUMINAMATH_GPT_main_theorem_l1169_116970

def f (m: ℕ) : ℕ := m * (m + 1) / 2

lemma f_1 : f 1 = 1 := by 
  -- placeholder for proof
  sorry

lemma f_functional_eq (m n : ℕ) : f m + f n = f (m + n) - m * n := by
  -- placeholder for proof
  sorry

theorem main_theorem (m : ℕ) : f m = m * (m + 1) / 2 := by
  -- Combining the conditions to conclude the result
  sorry

end NUMINAMATH_GPT_main_theorem_l1169_116970


namespace NUMINAMATH_GPT_seats_on_each_bus_l1169_116977

-- Define the given conditions
def totalStudents : ℕ := 45
def totalBuses : ℕ := 5

-- Define what we need to prove - 
-- that the number of seats on each bus is 9
def seatsPerBus (students : ℕ) (buses : ℕ) : ℕ := students / buses

theorem seats_on_each_bus : seatsPerBus totalStudents totalBuses = 9 := by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_seats_on_each_bus_l1169_116977


namespace NUMINAMATH_GPT_point_in_second_quadrant_condition_l1169_116982

theorem point_in_second_quadrant_condition (a : ℤ)
  (h1 : 3 * a - 9 < 0)
  (h2 : 10 - 2 * a > 0)
  (h3 : |3 * a - 9| = |10 - 2 * a|):
  (a + 2) ^ 2023 - 1 = 0 := 
sorry

end NUMINAMATH_GPT_point_in_second_quadrant_condition_l1169_116982


namespace NUMINAMATH_GPT_frequency_of_group_l1169_116956

-- Definitions based on conditions in the problem
def sampleCapacity : ℕ := 32
def frequencyRate : ℝ := 0.25

-- Lean statement representing the proof
theorem frequency_of_group : (frequencyRate * sampleCapacity : ℝ) = 8 := 
by 
  sorry -- Proof placeholder

end NUMINAMATH_GPT_frequency_of_group_l1169_116956


namespace NUMINAMATH_GPT_B_months_grazing_eq_five_l1169_116947

-- Define the conditions in the problem
def A_oxen : ℕ := 10
def A_months : ℕ := 7
def B_oxen : ℕ := 12
def C_oxen : ℕ := 15
def C_months : ℕ := 3
def total_rent : ℝ := 175
def C_rent_share : ℝ := 45

-- Total ox-units function
def total_ox_units (x : ℕ) : ℕ :=
  A_oxen * A_months + B_oxen * x + C_oxen * C_months

-- Prove that the number of months B's oxen grazed is 5
theorem B_months_grazing_eq_five (x : ℕ) :
  total_ox_units x = 70 + 12 * x + 45 →
  (C_rent_share / total_rent = 45 / total_ox_units x) →
  x = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_B_months_grazing_eq_five_l1169_116947


namespace NUMINAMATH_GPT_base8_to_base10_conversion_l1169_116922

theorem base8_to_base10_conversion : 
  (6 * 8^3 + 3 * 8^2 + 7 * 8^1 + 5 * 8^0) = 3325 := 
by 
  sorry

end NUMINAMATH_GPT_base8_to_base10_conversion_l1169_116922


namespace NUMINAMATH_GPT_scheme2_saves_money_for_80_participants_l1169_116936

-- Define the variables and conditions
def total_charge_scheme1 (x : ℕ) (hx : x > 50) : ℕ :=
  1500 + 240 * x

def total_charge_scheme2 (x : ℕ) (hx : x > 50) : ℕ :=
  270 * (x - 5)

-- Define the theorem
theorem scheme2_saves_money_for_80_participants :
  total_charge_scheme2 80 (by decide) < total_charge_scheme1 80 (by decide) :=
sorry

end NUMINAMATH_GPT_scheme2_saves_money_for_80_participants_l1169_116936


namespace NUMINAMATH_GPT_original_garden_length_l1169_116928

theorem original_garden_length (x : ℝ) (area : ℝ) (reduced_length : ℝ) (width : ℝ) (length_condition : x - reduced_length = width) (area_condition : x * width = area) (given_area : area = 120) (given_reduced_length : reduced_length = 2) : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_original_garden_length_l1169_116928


namespace NUMINAMATH_GPT_max_value_y_eq_neg10_l1169_116948

open Real

theorem max_value_y_eq_neg10 (x : ℝ) (hx : x > 0) : 
  ∃ y, y = 2 - 9 * x - 4 / x ∧ (∀ z, (∃ (x' : ℝ), x' > 0 ∧ z = 2 - 9 * x' - 4 / x') → z ≤ y) ∧ y = -10 :=
by
  sorry

end NUMINAMATH_GPT_max_value_y_eq_neg10_l1169_116948


namespace NUMINAMATH_GPT_scientific_notation_correct_l1169_116963

def big_number : ℕ := 274000000

noncomputable def scientific_notation : ℝ := 2.74 * 10^8

theorem scientific_notation_correct : (big_number : ℝ) = scientific_notation :=
by sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1169_116963


namespace NUMINAMATH_GPT_geometric_sequence_solution_l1169_116969

theorem geometric_sequence_solution:
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ),
    a 2 = 6 → 6 * a1 + a 3 = 30 → q > 2 →
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧
    (∀ n, S n = (3 ^ n - 1) / 2) :=
by
  intros a S q a1 h1 h2 h3
  sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l1169_116969


namespace NUMINAMATH_GPT_problem1_problem2_l1169_116920

variable (x : ℝ)

theorem problem1 : 
  (3 * x + 1) * (3 * x - 1) - (3 * x + 1)^2 = -6 * x - 2 :=
sorry

theorem problem2 : 
  (6 * x^4 - 8 * x^3) / (-2 * x^2) - (3 * x + 2) * (1 - x) = 3 * x - 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1169_116920


namespace NUMINAMATH_GPT_base_six_equals_base_b_l1169_116974

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  6 * 6 + 2

noncomputable def base_b_to_decimal (b : ℕ) : ℕ :=
  b^2 + 2 * b + 4

theorem base_six_equals_base_b (b : ℕ) : b^2 + 2 * b - 34 = 0 → b = 4 := 
by sorry

end NUMINAMATH_GPT_base_six_equals_base_b_l1169_116974


namespace NUMINAMATH_GPT_kitten_weight_l1169_116927

theorem kitten_weight :
  ∃ (x y z : ℝ), x + y + z = 36 ∧ x + z = 3 * y ∧ x + y = 1 / 2 * z ∧ x = 3 := 
by
  sorry

end NUMINAMATH_GPT_kitten_weight_l1169_116927


namespace NUMINAMATH_GPT_tangent_eq_inequality_not_monotonic_l1169_116966

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / (x + a)

theorem tangent_eq (a : ℝ) (h : 0 < a) : 
  ∃ k : ℝ, (k, f 1 a) ∈ {
    p : ℝ × ℝ | p.1 - (a + 1) * p.2 - 1 = 0 
  } :=
  sorry

theorem inequality (x : ℝ) (h : 1 ≤ x) : f x 1 ≤ (x - 1) / 2 := 
  sorry

theorem not_monotonic (a : ℝ) (h : 0 < a) : 
  ¬(∀ x y : ℝ, x < y → f x a ≤ f y a ∨ x < y → f x a ≥ f y a) := 
  sorry

end NUMINAMATH_GPT_tangent_eq_inequality_not_monotonic_l1169_116966


namespace NUMINAMATH_GPT_angle_DNE_l1169_116913

theorem angle_DNE (DE EF FD : ℝ) (EFD END FND : ℝ) 
  (h1 : DE = 2 * EF) 
  (h2 : EF = FD) 
  (h3 : EFD = 34) 
  (h4 : END = 3) 
  (h5 : FND = 18) : 
  ∃ DNE : ℝ, DNE = 104 :=
by 
  sorry

end NUMINAMATH_GPT_angle_DNE_l1169_116913


namespace NUMINAMATH_GPT_isosceles_triangles_perimeter_l1169_116949

theorem isosceles_triangles_perimeter (c d : ℕ) 
  (h1 : ¬(7 = c ∧ 10 = d) ∧ ¬(7 = d ∧ 10 = c))
  (h2 : 2 * c + d = 24) :
  d = 2 :=
sorry

end NUMINAMATH_GPT_isosceles_triangles_perimeter_l1169_116949


namespace NUMINAMATH_GPT_range_of_k_l1169_116908

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*y^2 = 2 ∧ 
  (∀ e : ℝ, (x^2 / 2 + y^2 / (2 / e) = 1 → (2 / e) > 2))) → 
  0 < k ∧ k < 1 :=
by 
sorry

end NUMINAMATH_GPT_range_of_k_l1169_116908


namespace NUMINAMATH_GPT_Sarah_l1169_116959

variable (s g : ℕ)

theorem Sarah's_score_130 (h1 : s = g + 50) (h2 : (s + g) / 2 = 105) : s = 130 :=
by
  sorry

end NUMINAMATH_GPT_Sarah_l1169_116959


namespace NUMINAMATH_GPT_local_value_proof_l1169_116916

-- Definitions based on the conditions
def face_value_7 : ℕ := 7
def local_value_6_in_7098060 : ℕ := 6000
def product_of_face_value_and_local_value : ℕ := face_value_7 * local_value_6_in_7098060
def local_value_6_in_product : ℕ := 6000

-- Theorem statement
theorem local_value_proof : local_value_6_in_product = 6000 :=
by
  -- Direct restatement of the condition in Lean
  sorry

end NUMINAMATH_GPT_local_value_proof_l1169_116916


namespace NUMINAMATH_GPT_greatest_consecutive_integers_sum_36_l1169_116996

theorem greatest_consecutive_integers_sum_36 : ∀ (x : ℤ), (x + (x + 1) + (x + 2) = 36) → (x + 2 = 13) :=
by
  sorry

end NUMINAMATH_GPT_greatest_consecutive_integers_sum_36_l1169_116996


namespace NUMINAMATH_GPT_smallest_number_from_digits_l1169_116958

theorem smallest_number_from_digits : 
  ∀ (d1 d2 d3 d4 : ℕ), (d1 = 2) → (d2 = 0) → (d3 = 1) → (d4 = 6) →
  ∃ n : ℕ, (n = 1026) ∧ 
  ((n = d1 * 1000 + d2 * 100 + d3 * 10 + d4) ∨ 
   (n = d1 * 1000 + d2 * 100 + d4 * 10 + d3) ∨ 
   (n = d1 * 1000 + d3 * 100 + d2 * 10 + d4) ∨ 
   (n = d1 * 1000 + d3 * 100 + d4 * 10 + d2) ∨ 
   (n = d1 * 1000 + d4 * 100 + d2 * 10 + d3) ∨ 
   (n = d1 * 1000 + d4 * 100 + d3 * 10 + d2) ∨ 
   (n = d2 * 1000 + d1 * 100 + d3 * 10 + d4) ∨ 
   (n = d2 * 1000 + d1 * 100 + d4 * 10 + d3) ∨ 
   (n = d2 * 1000 + d3 * 100 + d1 * 10 + d4) ∨ 
   (n = d2 * 1000 + d3 * 100 + d4 * 10 + d1) ∨ 
   (n = d2 * 1000 + d4 * 100 + d1 * 10 + d3) ∨ 
   (n = d2 * 1000 + d4 * 100 + d3 * 10 + d1) ∨ 
   (n = d3 * 1000 + d1 * 100 + d2 * 10 + d4) ∨ 
   (n = d3 * 1000 + d1 * 100 + d4 * 10 + d2) ∨ 
   (n = d3 * 1000 + d2 * 100 + d1 * 10 + d4) ∨ 
   (n = d3 * 1000 + d2 * 100 + d4 * 10 + d1) ∨ 
   (n = d3 * 1000 + d4 * 100 + d1 * 10 + d2) ∨ 
   (n = d3 * 1000 + d4 * 100 + d2 * 10 + d1) ∨ 
   (n = d4 * 1000 + d1 * 100 + d2 * 10 + d3) ∨ 
   (n = d4 * 1000 + d1 * 100 + d3 * 10 + d2) ∨ 
   (n = d4 * 1000 + d2 * 100 + d1 * 10 + d3) ∨ 
   (n = d4 * 1000 + d2 * 100 + d3 * 10 + d1) ∨ 
   (n = d4 * 1000 + d3 * 100 + d1 * 10 + d2) ∨ 
   (n = d4 * 1000 + d3 * 100 + d2 * 10 + d1)) := sorry

end NUMINAMATH_GPT_smallest_number_from_digits_l1169_116958


namespace NUMINAMATH_GPT_sum_not_divisible_by_three_times_any_number_l1169_116960

theorem sum_not_divisible_by_three_times_any_number (n : ℕ) (a : Fin n → ℕ) (h : n ≥ 3) (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k : Fin n, ¬ (a i + a j) ∣ (3 * a k)) :=
sorry

end NUMINAMATH_GPT_sum_not_divisible_by_three_times_any_number_l1169_116960


namespace NUMINAMATH_GPT_children_sit_in_same_row_twice_l1169_116926

theorem children_sit_in_same_row_twice
  (rows : ℕ) (seats_per_row : ℕ) (children : ℕ)
  (h_rows : rows = 7) (h_seats_per_row : seats_per_row = 10) (h_children : children = 50) :
  ∃ (morning_evening_pair : ℕ × ℕ), 
  (morning_evening_pair.1 < rows ∧ morning_evening_pair.2 < rows) ∧ 
  morning_evening_pair.1 = morning_evening_pair.2 :=
by
  sorry

end NUMINAMATH_GPT_children_sit_in_same_row_twice_l1169_116926


namespace NUMINAMATH_GPT_number_of_children_l1169_116992

-- Definitions of the conditions
def crayons_per_child : ℕ := 8
def total_crayons : ℕ := 56

-- Statement of the problem
theorem number_of_children : total_crayons / crayons_per_child = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_children_l1169_116992


namespace NUMINAMATH_GPT_trigonometric_identity_solution_l1169_116988

theorem trigonometric_identity_solution 
  (alpha beta : ℝ)
  (h1 : π / 4 < alpha)
  (h2 : alpha < 3 * π / 4)
  (h3 : 0 < beta)
  (h4 : beta < π / 4)
  (h5 : Real.cos (π / 4 + alpha) = -4 / 5)
  (h6 : Real.sin (3 * π / 4 + beta) = 12 / 13) :
  (Real.sin (alpha + beta) = 63 / 65) ∧
  (Real.cos (alpha - beta) = -33 / 65) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_solution_l1169_116988


namespace NUMINAMATH_GPT_simplify_sqrt_expr_l1169_116932

/-- Simplify the given radical expression and prove its equivalence to the expected result. -/
theorem simplify_sqrt_expr :
  (Real.sqrt (5 * 3) * Real.sqrt ((3 ^ 4) * (5 ^ 2)) = 225 * Real.sqrt 15) := 
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expr_l1169_116932


namespace NUMINAMATH_GPT_min_handshakes_l1169_116989

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end NUMINAMATH_GPT_min_handshakes_l1169_116989


namespace NUMINAMATH_GPT_probability_no_self_draws_l1169_116962

theorem probability_no_self_draws :
  let total_outcomes := 6
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_self_draws_l1169_116962


namespace NUMINAMATH_GPT_find_x_l1169_116917

noncomputable def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_x (x : ℝ) :
  let a := (1, 2*x + 1)
  let b := (2, 3)
  (vector_parallel a b) → x = 1 / 4 :=
by
  intro h
  have h_eq := h
  sorry  -- proof is not needed as per instruction

end NUMINAMATH_GPT_find_x_l1169_116917


namespace NUMINAMATH_GPT_cricket_target_run_rate_cricket_wicket_partnership_score_l1169_116961

noncomputable def remaining_runs_needed (initial_runs : ℕ) (target_runs : ℕ) : ℕ :=
  target_runs - initial_runs

noncomputable def required_run_rate (remaining_runs : ℕ) (remaining_overs : ℕ) : ℚ :=
  (remaining_runs : ℚ) / remaining_overs

theorem cricket_target_run_rate (initial_runs : ℕ) (target_runs : ℕ) (remaining_overs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → remaining_overs = 40 → initial_wickets = 3 →
  required_run_rate (remaining_runs_needed initial_runs target_runs) remaining_overs = 6.25 :=
by
  sorry


theorem cricket_wicket_partnership_score (initial_runs : ℕ) (target_runs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → initial_wickets = 3 →
  remaining_runs_needed initial_runs target_runs = 250 :=
by
  sorry

end NUMINAMATH_GPT_cricket_target_run_rate_cricket_wicket_partnership_score_l1169_116961


namespace NUMINAMATH_GPT_Murtha_pebble_collection_l1169_116952

def sum_of_first_n_natural_numbers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem Murtha_pebble_collection : sum_of_first_n_natural_numbers 20 = 210 := by
  sorry

end NUMINAMATH_GPT_Murtha_pebble_collection_l1169_116952


namespace NUMINAMATH_GPT_remaining_minutes_proof_l1169_116973

def total_series_minutes : ℕ := 360

def first_session_end : ℕ := 17 * 60 + 44  -- in minutes
def first_session_start : ℕ := 15 * 60 + 20  -- in minutes
def second_session_end : ℕ := 20 * 60 + 40  -- in minutes
def second_session_start : ℕ := 19 * 60 + 15  -- in minutes
def third_session_end : ℕ := 22 * 60 + 30  -- in minutes
def third_session_start : ℕ := 21 * 60 + 35  -- in minutes

def first_session_duration : ℕ := first_session_end - first_session_start
def second_session_duration : ℕ := second_session_end - second_session_start
def third_session_duration : ℕ := third_session_end - third_session_start

def total_watched : ℕ := first_session_duration + second_session_duration + third_session_duration

def remaining_time : ℕ := total_series_minutes - total_watched

theorem remaining_minutes_proof : remaining_time = 76 := 
by 
  sorry  -- Proof goes here

end NUMINAMATH_GPT_remaining_minutes_proof_l1169_116973


namespace NUMINAMATH_GPT_intersection_x_val_l1169_116911

theorem intersection_x_val (x y : ℝ) (h1 : y = 3 * x - 24) (h2 : 5 * x + 2 * y = 102) : x = 150 / 11 :=
by
  sorry

end NUMINAMATH_GPT_intersection_x_val_l1169_116911


namespace NUMINAMATH_GPT_evaluate_expression_l1169_116978

theorem evaluate_expression (b : ℚ) (h : b = 4 / 3) :
  (6 * b ^ 2 - 17 * b + 8) * (3 * b - 4) = 0 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1169_116978


namespace NUMINAMATH_GPT_tank_capacity_l1169_116953

variable (c w : ℕ)

-- Conditions
def initial_fraction (w c : ℕ) : Prop := w = c / 7
def final_fraction (w c : ℕ) : Prop := (w + 2) = c / 5

-- The theorem statement
theorem tank_capacity : 
  initial_fraction w c → 
  final_fraction w c → 
  c = 35 := 
by
  sorry  -- indicates that the proof is not provided

end NUMINAMATH_GPT_tank_capacity_l1169_116953


namespace NUMINAMATH_GPT_cos_pi_minus_double_alpha_l1169_116904

theorem cos_pi_minus_double_alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_double_alpha_l1169_116904


namespace NUMINAMATH_GPT_total_kids_played_with_l1169_116968

-- Define the conditions as separate constants
def kidsMonday : Nat := 12
def kidsTuesday : Nat := 7

-- Prove the total number of kids Julia played with
theorem total_kids_played_with : kidsMonday + kidsTuesday = 19 := 
by
  sorry

end NUMINAMATH_GPT_total_kids_played_with_l1169_116968


namespace NUMINAMATH_GPT_contestant_wins_probability_l1169_116951

-- Define the basic parameters: number of questions and number of choices
def num_questions : ℕ := 4
def num_choices : ℕ := 3

-- Define the probability of getting a single question right
def prob_right : ℚ := 1 / num_choices

-- Define the probability of guessing all questions right
def prob_all_right : ℚ := prob_right ^ num_questions

-- Define the probability of guessing exactly three questions right (one wrong)
def prob_one_wrong : ℚ := (prob_right ^ 3) * (2 / num_choices)

-- Calculate the total probability of winning
def total_prob_winning : ℚ := prob_all_right + 4 * prob_one_wrong

-- The final statement to prove
theorem contestant_wins_probability :
  total_prob_winning = 1 / 9 := 
sorry

end NUMINAMATH_GPT_contestant_wins_probability_l1169_116951


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1169_116907

def condition_p (x : ℝ) : Prop := x^2 - 9 > 0
def condition_q (x : ℝ) : Prop := x^2 - (5 / 6) * x + (1 / 6) > 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x, condition_p x → condition_q x) ∧ ¬(∀ x, condition_q x → condition_p x) :=
sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1169_116907


namespace NUMINAMATH_GPT_membership_percentage_change_l1169_116954

-- Definitions required based on conditions
def membersFallChange (initialMembers : ℝ) : ℝ := initialMembers * 1.07
def membersSpringChange (fallMembers : ℝ) : ℝ := fallMembers * 0.81
def membersSummerChange (springMembers : ℝ) : ℝ := springMembers * 1.15

-- Prove the total change in percentage from fall to the end of summer
theorem membership_percentage_change :
  let initialMembers := 100
  let fallMembers := membersFallChange initialMembers
  let springMembers := membersSpringChange fallMembers
  let summerMembers := membersSummerChange springMembers
  ((summerMembers - initialMembers) / initialMembers) * 100 = -0.33 := by
  sorry

end NUMINAMATH_GPT_membership_percentage_change_l1169_116954


namespace NUMINAMATH_GPT_cleaning_cost_l1169_116998

theorem cleaning_cost (num_cleanings : ℕ) (chemical_cost : ℕ) (monthly_cost : ℕ) (tip_percentage : ℚ) 
  (cleaning_sessions_per_month : num_cleanings = 30 / 3)
  (monthly_chemical_cost : chemical_cost = 2 * 200)
  (total_monthly_cost : monthly_cost = 2050)
  (cleaning_cost_with_tip : monthly_cost - chemical_cost =  num_cleanings * (1 + tip_percentage) * x) : 
  x = 150 := 
by
  sorry

end NUMINAMATH_GPT_cleaning_cost_l1169_116998


namespace NUMINAMATH_GPT_evaluate_expression_l1169_116993

theorem evaluate_expression (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9 / 4 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l1169_116993


namespace NUMINAMATH_GPT_tom_total_dimes_l1169_116912

-- Define the original and additional dimes Tom received.
def original_dimes : ℕ := 15
def additional_dimes : ℕ := 33

-- Define the total number of dimes Tom has now.
def total_dimes : ℕ := original_dimes + additional_dimes

-- Statement to prove that the total number of dimes Tom has is 48.
theorem tom_total_dimes : total_dimes = 48 := by
  sorry

end NUMINAMATH_GPT_tom_total_dimes_l1169_116912


namespace NUMINAMATH_GPT_bob_eats_10_apples_l1169_116903

variable (B C : ℕ)
variable (h1 : B + C = 30)
variable (h2 : C = 2 * B)

theorem bob_eats_10_apples : B = 10 :=
by sorry

end NUMINAMATH_GPT_bob_eats_10_apples_l1169_116903


namespace NUMINAMATH_GPT_total_pieces_gum_is_correct_l1169_116909

-- Define the number of packages and pieces per package
def packages : ℕ := 27
def pieces_per_package : ℕ := 18

-- Define the total number of pieces of gum Robin has
def total_pieces_gum : ℕ :=
  packages * pieces_per_package

-- State the theorem and proof obligation
theorem total_pieces_gum_is_correct : total_pieces_gum = 486 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_pieces_gum_is_correct_l1169_116909


namespace NUMINAMATH_GPT_find_B_l1169_116910

variable {A B C D : ℕ}

-- Condition 1: The first dig site (A) was dated 352 years more recent than the second dig site (B)
axiom h1 : A = B + 352

-- Condition 2: The third dig site (C) was dated 3700 years older than the first dig site (A)
axiom h2 : C = A - 3700

-- Condition 3: The fourth dig site (D) was twice as old as the third dig site (C)
axiom h3 : D = 2 * C

-- Condition 4: The age difference between the second dig site (B) and the third dig site (C) was four times the difference between the fourth dig site (D) and the first dig site (A)
axiom h4 : B - C = 4 * (D - A)

-- Condition 5: The fourth dig site is dated 8400 BC.
axiom h5 : D = 8400

-- Prove the question
theorem find_B : B = 7548 :=
by
  sorry

end NUMINAMATH_GPT_find_B_l1169_116910


namespace NUMINAMATH_GPT_intersection_of_complements_l1169_116981

theorem intersection_of_complements 
  (U : Set ℕ) (A B : Set ℕ)
  (hU : U = { x | x ≤ 5 }) 
  (hA : A = {1, 2, 3}) 
  (hB : B = {1, 4}) :
  ((U \ A) ∩ (U \ B)) = {0, 5} :=
by sorry

end NUMINAMATH_GPT_intersection_of_complements_l1169_116981


namespace NUMINAMATH_GPT_find_a_in_triangle_l1169_116945

theorem find_a_in_triangle (b c : ℝ) (cos_B_minus_C : ℝ) (a : ℝ) 
  (hb : b = 7) (hc : c = 6) (hcos : cos_B_minus_C = 15 / 16) :
  a = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_in_triangle_l1169_116945


namespace NUMINAMATH_GPT_distinguishable_balls_boxes_l1169_116942

theorem distinguishable_balls_boxes : (3^6 = 729) :=
by {
  sorry
}

end NUMINAMATH_GPT_distinguishable_balls_boxes_l1169_116942
