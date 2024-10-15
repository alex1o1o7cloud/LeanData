import Mathlib

namespace NUMINAMATH_GPT_largest_possible_A_l816_81653

theorem largest_possible_A : ∃ A B : ℕ, 13 = 4 * A + B ∧ B < A ∧ A = 3 := by
  sorry

end NUMINAMATH_GPT_largest_possible_A_l816_81653


namespace NUMINAMATH_GPT_terry_current_age_l816_81680

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end NUMINAMATH_GPT_terry_current_age_l816_81680


namespace NUMINAMATH_GPT_largest_shaded_area_l816_81679

noncomputable def figureA_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureB_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureC_shaded_area : ℝ := 16 - 4 * Real.sqrt 3

theorem largest_shaded_area : 
  figureC_shaded_area > figureA_shaded_area ∧ figureC_shaded_area > figureB_shaded_area :=
by
  sorry

end NUMINAMATH_GPT_largest_shaded_area_l816_81679


namespace NUMINAMATH_GPT_fraction_computation_l816_81629

theorem fraction_computation :
  (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_computation_l816_81629


namespace NUMINAMATH_GPT_range_of_a_l816_81636

theorem range_of_a (a : ℝ) :
  (a + 1 > 0 ∧ 3 - 2 * a > 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a < 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a > 0)
  → (2 / 3 < a ∧ a < 3 / 2) ∨ (a < -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l816_81636


namespace NUMINAMATH_GPT_smaller_circle_area_l816_81633

theorem smaller_circle_area (r R : ℝ) (hR : R = 3 * r)
  (hTangentLines : ∀ (P A B A' B' : ℝ), P = 5 ∧ A = 5 ∧ PA = 5 ∧ A' = 5 ∧ PA' = 5 ∧ AB = 5 ∧ A'B' = 5 ) :
  π * r^2 = 25 / 3 * π := by
  sorry

end NUMINAMATH_GPT_smaller_circle_area_l816_81633


namespace NUMINAMATH_GPT_geometric_sequence_constant_l816_81692

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ)
    (h1 : ∀ n, a (n+1) = q * a n)
    (h2 : ∀ n, a n > 0)
    (h3 : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4) ^ 2) :
    ∀ n, a n = a 0 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_constant_l816_81692


namespace NUMINAMATH_GPT_intersection_M_N_l816_81609

def M : Set ℝ := {x | x < 1/2}
def N : Set ℝ := {y | y ≥ -4}

theorem intersection_M_N :
  (M ∩ N = {x | -4 ≤ x ∧ x < 1/2}) :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l816_81609


namespace NUMINAMATH_GPT_remainder_div_l816_81643

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 := by
  sorry

end NUMINAMATH_GPT_remainder_div_l816_81643


namespace NUMINAMATH_GPT_simplify_radical_product_l816_81660

theorem simplify_radical_product : 
  (32^(1/5)) * (8^(1/3)) * (4^(1/2)) = 8 := 
by
  sorry

end NUMINAMATH_GPT_simplify_radical_product_l816_81660


namespace NUMINAMATH_GPT_sequence_general_formula_l816_81642

theorem sequence_general_formula (a : ℕ → ℚ) (h₁ : a 1 = 2 / 3)
  (h₂ : ∀ n : ℕ, a (n + 1) = a n + a n * a (n + 1)) : 
  ∀ n : ℕ, a n = 2 / (5 - 2 * n) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l816_81642


namespace NUMINAMATH_GPT_largest_square_area_l816_81686

theorem largest_square_area (total_string_length : ℕ) (h : total_string_length = 32) : ∃ (area : ℕ), area = 64 := 
  by
    sorry

end NUMINAMATH_GPT_largest_square_area_l816_81686


namespace NUMINAMATH_GPT_quadratic_positive_imp_ineq_l816_81635

theorem quadratic_positive_imp_ineq (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c > 0) → b^2 - 4 * c ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_positive_imp_ineq_l816_81635


namespace NUMINAMATH_GPT_arc_length_correct_l816_81687

noncomputable def chord_length := 2
noncomputable def central_angle := 2
noncomputable def half_chord_length := 1
noncomputable def radius := 1 / Real.sin 1
noncomputable def arc_length := 2 * radius

theorem arc_length_correct :
  arc_length = 2 / Real.sin 1 := by
sorry

end NUMINAMATH_GPT_arc_length_correct_l816_81687


namespace NUMINAMATH_GPT_probability_odd_sum_is_correct_l816_81662

-- Define the set of the first twelve prime numbers.
def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Define the problem statement.
noncomputable def probability_odd_sum : ℚ :=
  let even_prime_count := 1
  let odd_prime_count := 11
  let ways_to_pick_1_even_and_4_odd := (Nat.choose odd_prime_count 4)
  let total_ways := Nat.choose 12 5
  (ways_to_pick_1_even_and_4_odd : ℚ) / total_ways

theorem probability_odd_sum_is_correct :
  probability_odd_sum = 55 / 132 :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_sum_is_correct_l816_81662


namespace NUMINAMATH_GPT_proof_cos_2x_cos_2y_l816_81648

variable {θ x y : ℝ}

-- Conditions
def is_arith_seq (a b c : ℝ) := b = (a + c) / 2
def is_geom_seq (a b c : ℝ) := b^2 = a * c

-- Proving the given statement with the provided conditions
theorem proof_cos_2x_cos_2y (h_arith : is_arith_seq (Real.sin θ) (Real.sin x) (Real.cos θ))
                            (h_geom : is_geom_seq (Real.sin θ) (Real.sin y) (Real.cos θ)) :
  2 * Real.cos (2 * x) = Real.cos (2 * y) :=
sorry

end NUMINAMATH_GPT_proof_cos_2x_cos_2y_l816_81648


namespace NUMINAMATH_GPT_am_gm_inequality_for_x_l816_81610

theorem am_gm_inequality_for_x (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by 
  sorry

end NUMINAMATH_GPT_am_gm_inequality_for_x_l816_81610


namespace NUMINAMATH_GPT_xy_sufficient_but_not_necessary_l816_81637

theorem xy_sufficient_but_not_necessary (x y : ℝ) : (x > 0 ∧ y > 0) → (xy > 0) ∧ ¬(xy > 0 → (x > 0 ∧ y > 0)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_xy_sufficient_but_not_necessary_l816_81637


namespace NUMINAMATH_GPT_equilateral_triangle_l816_81650

theorem equilateral_triangle (a b c : ℝ) (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c ∧ a = c := 
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_l816_81650


namespace NUMINAMATH_GPT_bruce_bank_savings_l816_81671

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end NUMINAMATH_GPT_bruce_bank_savings_l816_81671


namespace NUMINAMATH_GPT_function_passes_through_point_l816_81626

noncomputable def special_function (a : ℝ) (x : ℝ) := a^(x - 1) + 1

theorem function_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  special_function a 1 = 2 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_function_passes_through_point_l816_81626


namespace NUMINAMATH_GPT_distance_focus_directrix_l816_81654

theorem distance_focus_directrix (p : ℝ) :
  (∀ (x y : ℝ), y^2 = 2 * p * x ∧ x = 6 ∧ dist (x, y) (p/2, 0) = 10) →
  abs (p) = 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_focus_directrix_l816_81654


namespace NUMINAMATH_GPT_find_a_l816_81655

noncomputable def ab (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a {a : ℝ} : ab a 6 = -3 → a = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l816_81655


namespace NUMINAMATH_GPT_problem_conditions_l816_81619

noncomputable def f (x : ℝ) := x^2 - 2 * x * Real.log x
noncomputable def g (x : ℝ) := Real.exp x - (Real.exp 2 * x^2) / 4

theorem problem_conditions :
  (∀ x > 0, deriv f x > 0) ∧ 
  (∃! x, g x = 0) ∧ 
  (∃ x, f x = g x) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l816_81619


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l816_81605

theorem infinite_geometric_series_sum
  (a : ℚ) (r : ℚ) (h_a : a = 1) (h_r : r = 2 / 3) (h_r_abs_lt_one : |r| < 1) :
  ∑' (n : ℕ), a * r^n = 3 :=
by
  -- Import necessary lemmas and properties for infinite series
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_infinite_geometric_series_sum_l816_81605


namespace NUMINAMATH_GPT_television_screen_horizontal_length_l816_81658

theorem television_screen_horizontal_length :
  ∀ (d : ℝ) (r_l : ℝ) (r_h : ℝ), r_l / r_h = 4 / 3 → d = 27 → 
  let h := (3 / 5) * d
  let l := (4 / 5) * d
  l = 21.6 := by
  sorry

end NUMINAMATH_GPT_television_screen_horizontal_length_l816_81658


namespace NUMINAMATH_GPT_dana_pencils_more_than_jayden_l816_81624

theorem dana_pencils_more_than_jayden :
  ∀ (Jayden_has_pencils : ℕ) (Marcus_has_pencils : ℕ) (Dana_has_pencils : ℕ),
    Jayden_has_pencils = 20 →
    Marcus_has_pencils = Jayden_has_pencils / 2 →
    Dana_has_pencils = Marcus_has_pencils + 25 →
    Dana_has_pencils - Jayden_has_pencils = 15 :=
by
  intros Jayden_has_pencils Marcus_has_pencils Dana_has_pencils
  intro h1
  intro h2
  intro h3
  sorry

end NUMINAMATH_GPT_dana_pencils_more_than_jayden_l816_81624


namespace NUMINAMATH_GPT_tan_A_eq_11_l816_81613

variable (A B C : ℝ)

theorem tan_A_eq_11
  (h1 : Real.sin A = 10 * Real.sin B * Real.sin C)
  (h2 : Real.cos A = 10 * Real.cos B * Real.cos C) :
  Real.tan A = 11 := 
sorry

end NUMINAMATH_GPT_tan_A_eq_11_l816_81613


namespace NUMINAMATH_GPT_bus_length_is_200_l816_81638

def length_of_bus (distance_km distance_secs passing_secs : ℕ) : ℕ :=
  let speed_kms := distance_km / distance_secs
  let speed_ms := speed_kms * 1000
  speed_ms * passing_secs

theorem bus_length_is_200 
  (distance_km : ℕ) (distance_secs : ℕ) (passing_secs : ℕ)
  (h1 : distance_km = 12) (h2 : distance_secs = 300) (h3 : passing_secs = 5) : 
  length_of_bus distance_km distance_secs passing_secs = 200 := 
  by
    sorry

end NUMINAMATH_GPT_bus_length_is_200_l816_81638


namespace NUMINAMATH_GPT_sqrt_eighteen_simplifies_l816_81645

open Real

theorem sqrt_eighteen_simplifies :
  sqrt 18 = 3 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_eighteen_simplifies_l816_81645


namespace NUMINAMATH_GPT_isabel_weekly_distance_l816_81677

def circuit_length : ℕ := 365
def morning_runs : ℕ := 7
def afternoon_runs : ℕ := 3
def days_per_week : ℕ := 7

def morning_distance := morning_runs * circuit_length
def afternoon_distance := afternoon_runs * circuit_length
def daily_distance := morning_distance + afternoon_distance
def weekly_distance := daily_distance * days_per_week

theorem isabel_weekly_distance : weekly_distance = 25550 := by
  sorry

end NUMINAMATH_GPT_isabel_weekly_distance_l816_81677


namespace NUMINAMATH_GPT_complement_union_l816_81661

open Set

variable (U : Set ℕ := {0, 1, 2, 3, 4}) (A : Set ℕ := {1, 2, 3}) (B : Set ℕ := {2, 4})

theorem complement_union (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) : 
  (U \ A ∪ B) = {0, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l816_81661


namespace NUMINAMATH_GPT_smallest_possible_n_l816_81607

theorem smallest_possible_n (n : ℕ) :
  ∃ n, 17 * n - 3 ≡ 0 [MOD 11] ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_n_l816_81607


namespace NUMINAMATH_GPT_exists_triangle_l816_81604

variable (k α m_a : ℝ)

-- Define the main constructibility condition as a noncomputable function.
noncomputable def triangle_constructible (k α m_a : ℝ) : Prop :=
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2))

-- Main theorem statement to prove the existence of the triangle
theorem exists_triangle :
  ∃ (k α m_a : ℝ), triangle_constructible k α m_a := 
sorry

end NUMINAMATH_GPT_exists_triangle_l816_81604


namespace NUMINAMATH_GPT_avg_salary_increase_l816_81673

def initial_avg_salary : ℝ := 1700
def num_employees : ℕ := 20
def manager_salary : ℝ := 3800

theorem avg_salary_increase :
  ((num_employees * initial_avg_salary + manager_salary) / (num_employees + 1)) - initial_avg_salary = 100 :=
by
  sorry

end NUMINAMATH_GPT_avg_salary_increase_l816_81673


namespace NUMINAMATH_GPT_bags_already_made_l816_81672

def bags_per_batch : ℕ := 10
def customer_order : ℕ := 60
def days_to_fulfill : ℕ := 4
def batches_per_day : ℕ := 1

theorem bags_already_made :
  (customer_order - (days_to_fulfill * batches_per_day * bags_per_batch)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_bags_already_made_l816_81672


namespace NUMINAMATH_GPT_dots_not_visible_l816_81631

theorem dots_not_visible (visible_sum : ℕ) (total_faces_sum : ℕ) (num_dice : ℕ) (total_visible_faces : ℕ)
  (h1 : total_faces_sum = 21)
  (h2 : visible_sum = 22) 
  (h3 : num_dice = 3)
  (h4 : total_visible_faces = 7) :
  (num_dice * total_faces_sum - visible_sum) = 41 :=
sorry

end NUMINAMATH_GPT_dots_not_visible_l816_81631


namespace NUMINAMATH_GPT_woods_width_l816_81681

theorem woods_width (Area Length Width : ℝ) (hArea : Area = 24) (hLength : Length = 3) : 
  Width = 8 := 
by
  sorry

end NUMINAMATH_GPT_woods_width_l816_81681


namespace NUMINAMATH_GPT_xiao_hua_correct_answers_l816_81676

theorem xiao_hua_correct_answers :
  ∃ (correct_answers wrong_answers : ℕ), 
    correct_answers + wrong_answers = 15 ∧
    8 * correct_answers - 4 * wrong_answers = 72 ∧
    correct_answers = 11 :=
by
  sorry

end NUMINAMATH_GPT_xiao_hua_correct_answers_l816_81676


namespace NUMINAMATH_GPT_red_balls_unchanged_l816_81603

-- Definitions: 
def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5

def remove_blue_ball (blue_balls : ℕ) : ℕ :=
  if blue_balls > 0 then blue_balls - 1 else blue_balls

-- Condition after one blue ball is removed
def blue_balls_after_removal := remove_blue_ball initial_blue_balls

-- Prove that the number of red balls remain unchanged
theorem red_balls_unchanged : initial_red_balls = 3 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_unchanged_l816_81603


namespace NUMINAMATH_GPT_first_meeting_time_of_boys_l816_81693

theorem first_meeting_time_of_boys 
  (L : ℝ) (v1_kmh : ℝ) (v2_kmh : ℝ) (v1_ms v2_ms : ℝ) (rel_speed : ℝ) (t : ℝ)
  (hv1_km_to_ms : v1_ms = v1_kmh * 1000 / 3600)
  (hv2_km_to_ms : v2_ms = v2_kmh * 1000 / 3600)
  (hrel_speed : rel_speed = v1_ms + v2_ms)
  (hl : L = 4800)
  (hv1 : v1_kmh = 60)
  (hv2 : v2_kmh = 100)
  (ht : t = L / rel_speed) :
  t = 108 := by
  -- we're providing a placeholder for the proof
  sorry

end NUMINAMATH_GPT_first_meeting_time_of_boys_l816_81693


namespace NUMINAMATH_GPT_age_ratio_is_4_over_3_l816_81649

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end NUMINAMATH_GPT_age_ratio_is_4_over_3_l816_81649


namespace NUMINAMATH_GPT_golden_section_length_l816_81634

theorem golden_section_length (MN : ℝ) (MP NP : ℝ) (hMN : MN = 1) (hP : MP + NP = MN) (hgolden : MN / MP = MP / NP) (hMP_gt_NP : MP > NP) : MP = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_GPT_golden_section_length_l816_81634


namespace NUMINAMATH_GPT_inequality_proof_l816_81664

theorem inequality_proof (a b c : ℝ) (hab : a > b) : a * |c| ≥ b * |c| := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l816_81664


namespace NUMINAMATH_GPT_monkeys_and_bananas_l816_81623

theorem monkeys_and_bananas (m1 m2 t b1 b2 : ℕ) (h1 : m1 = 8) (h2 : t = 8) (h3 : b1 = 8) (h4 : b2 = 3) : m2 = 3 :=
by
  -- Here we will include the formal proof steps
  sorry

end NUMINAMATH_GPT_monkeys_and_bananas_l816_81623


namespace NUMINAMATH_GPT_exists_c_gt_zero_l816_81669

theorem exists_c_gt_zero (a b : ℝ) (h : a < b) : ∃ c > 0, a < b + c := 
sorry

end NUMINAMATH_GPT_exists_c_gt_zero_l816_81669


namespace NUMINAMATH_GPT_vector_expression_result_l816_81628

structure Vector2 :=
(x : ℝ)
(y : ℝ)

def vector_dot_product (v1 v2 : Vector2) : ℝ :=
  v1.x * v1.y + v2.x * v2.y

def vector_scalar_mul (c : ℝ) (v : Vector2) : Vector2 :=
  { x := c * v.x, y := c * v.y }

def vector_sub (v1 v2 : Vector2) : Vector2 :=
  { x := v1.x - v2.x, y := v1.y - v2.y }

noncomputable def a : Vector2 := { x := 2, y := -1 }
noncomputable def b : Vector2 := { x := 3, y := -2 }

theorem vector_expression_result :
  vector_dot_product
    (vector_sub (vector_scalar_mul 3 a) b)
    (vector_sub a (vector_scalar_mul 2 b)) = -15 := by
  sorry

end NUMINAMATH_GPT_vector_expression_result_l816_81628


namespace NUMINAMATH_GPT_sequence_value_a_l816_81695

theorem sequence_value_a (a : ℚ) (a_n : ℕ → ℚ)
  (h1 : a_n 1 = a) (h2 : a_n 2 = a)
  (h3 : ∀ n ≥ 3, a_n n = a_n (n - 1) + a_n (n - 2))
  (h4 : a_n 8 = 34) :
  a = 34 / 21 :=
by sorry

end NUMINAMATH_GPT_sequence_value_a_l816_81695


namespace NUMINAMATH_GPT_find_number_of_students_l816_81616

theorem find_number_of_students
  (n : ℕ)
  (average_marks : ℕ → ℚ)
  (wrong_mark_corrected : ℕ → ℕ → ℚ)
  (correct_avg_marks_pred : ℕ → ℚ → Prop)
  (h1 : average_marks n = 60)
  (h2 : wrong_mark_corrected 90 15 = 75)
  (h3 : correct_avg_marks_pred n 57.5) :
  n = 30 :=
sorry

end NUMINAMATH_GPT_find_number_of_students_l816_81616


namespace NUMINAMATH_GPT_initial_clothing_count_l816_81694

theorem initial_clothing_count 
  (donated_first : ℕ) 
  (donated_second : ℕ) 
  (thrown_away : ℕ) 
  (remaining : ℕ) 
  (h1 : donated_first = 5) 
  (h2 : donated_second = 3 * donated_first) 
  (h3 : thrown_away = 15) 
  (h4 : remaining = 65) :
  donated_first + donated_second + thrown_away + remaining = 100 :=
by
  sorry

end NUMINAMATH_GPT_initial_clothing_count_l816_81694


namespace NUMINAMATH_GPT_surface_area_inequality_l816_81667

theorem surface_area_inequality
  (a b c d e f S : ℝ) :
  S ≤ (Real.sqrt 3 / 6) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :=
sorry

end NUMINAMATH_GPT_surface_area_inequality_l816_81667


namespace NUMINAMATH_GPT_relationship_of_rationals_l816_81601

theorem relationship_of_rationals (a b c : ℚ) (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a :=
by {
  sorry
}

end NUMINAMATH_GPT_relationship_of_rationals_l816_81601


namespace NUMINAMATH_GPT_equivalent_single_discount_l816_81675

theorem equivalent_single_discount (x : ℝ) : 
  (1 - 0.15) * (1 - 0.20) * (1 - 0.10) = 1 - 0.388 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l816_81675


namespace NUMINAMATH_GPT_eleven_place_unamed_racer_l816_81656

theorem eleven_place_unamed_racer
  (Rand Hikmet Jack Marta David Todd : ℕ)
  (positions : Fin 15)
  (C_1 : Rand = Hikmet + 6)
  (C_2 : Marta = Jack + 1)
  (C_3 : David = Hikmet + 3)
  (C_4 : Jack = Todd + 3)
  (C_5 : Todd = Rand + 1)
  (C_6 : Marta = 8) :
  ∃ (x : Fin 15), (x ≠ Rand) ∧ (x ≠ Hikmet) ∧ (x ≠ Jack) ∧ (x ≠ Marta) ∧ (x ≠ David) ∧ (x ≠ Todd) ∧ x = 11 := 
sorry

end NUMINAMATH_GPT_eleven_place_unamed_racer_l816_81656


namespace NUMINAMATH_GPT_xy_product_l816_81690

variable {x y : ℝ}

theorem xy_product (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x + 3/x = y + 3/y) : x * y = 3 :=
sorry

end NUMINAMATH_GPT_xy_product_l816_81690


namespace NUMINAMATH_GPT_train_passes_jogger_in_40_seconds_l816_81646

variable (speed_jogger_kmh : ℕ)
variable (speed_train_kmh : ℕ)
variable (head_start : ℕ)
variable (train_length : ℕ)

noncomputable def time_to_pass_jogger (speed_jogger_kmh speed_train_kmh head_start train_length : ℕ) : ℕ :=
  let speed_jogger_ms := (speed_jogger_kmh * 1000) / 3600
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let relative_speed := speed_train_ms - speed_jogger_ms
  let total_distance := head_start + train_length
  total_distance / relative_speed

theorem train_passes_jogger_in_40_seconds : time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end NUMINAMATH_GPT_train_passes_jogger_in_40_seconds_l816_81646


namespace NUMINAMATH_GPT_largest_a_l816_81685

theorem largest_a (a b : ℕ) (x : ℕ) (h_a_range : 2 < a ∧ a < x) (h_b_range : 4 < b ∧ b < 13) (h_fraction_range : 7 * a = 57) : a = 8 :=
sorry

end NUMINAMATH_GPT_largest_a_l816_81685


namespace NUMINAMATH_GPT_expression_value_l816_81665

-- Define the problem statement
theorem expression_value (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : (x + y) / z = (y + z) / x) (h5 : (y + z) / x = (z + x) / y) :
  ∃ k : ℝ, k = 8 ∨ k = -1 := 
sorry

end NUMINAMATH_GPT_expression_value_l816_81665


namespace NUMINAMATH_GPT_largest_n_binomial_l816_81640

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end NUMINAMATH_GPT_largest_n_binomial_l816_81640


namespace NUMINAMATH_GPT_functional_equation_solution_l816_81689

theorem functional_equation_solution (f g : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (x^2 - g y) = g x ^ 2 - y) :
  (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l816_81689


namespace NUMINAMATH_GPT_curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l816_81659

noncomputable def curve_C (a x y : ℝ) := a * x ^ 2 + a * y ^ 2 - 2 * x - 2 * y = 0

theorem curve_C_straight_line (a : ℝ) : a = 0 → ∃ x y : ℝ, curve_C a x y :=
by
  intro ha
  use (-1), 1
  rw [curve_C, ha]
  simp

theorem curve_C_not_tangent (a : ℝ) : a = 1 → ¬ ∀ x y, 3 * x + y = 0 → curve_C a x y :=
by
  sorry

theorem curve_C_fixed_point (x y a : ℝ) : curve_C a 0 0 :=
by
  rw [curve_C]
  simp

theorem curve_C_intersect (a : ℝ) : a = 1 → ∃ x y : ℝ, (x + 2 * y = 0) ∧ curve_C a x y :=
by
  sorry

end NUMINAMATH_GPT_curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l816_81659


namespace NUMINAMATH_GPT_sequence_a_n_l816_81617

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 1 then 1 else (1 : ℚ) / (2 * n - 1)

theorem sequence_a_n (n : ℕ) (hn : n ≥ 1) : 
  (a_n 1 = 1) ∧ 
  (∀ n, a_n n ≠ 0) ∧ 
  (∀ n, n ≥ 2 → a_n n + 2 * a_n n * a_n (n - 1) - a_n (n - 1) = 0) →
  a_n n = 1 / (2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_l816_81617


namespace NUMINAMATH_GPT_find_number_l816_81632

theorem find_number (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l816_81632


namespace NUMINAMATH_GPT_expression_evaluation_l816_81621

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := 
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l816_81621


namespace NUMINAMATH_GPT_original_price_sarees_l816_81691

theorem original_price_sarees (P : ℝ) (h : 0.80 * P * 0.85 = 231.2) : P = 340 := 
by sorry

end NUMINAMATH_GPT_original_price_sarees_l816_81691


namespace NUMINAMATH_GPT_emily_sixth_quiz_score_l816_81688

theorem emily_sixth_quiz_score (q1 q2 q3 q4 q5 target_mean : ℕ) (required_sum : ℕ) (current_sum : ℕ) (s6 : ℕ)
  (h1 : q1 = 94) (h2 : q2 = 97) (h3 : q3 = 88) (h4 : q4 = 91) (h5 : q5 = 102) (h_target_mean : target_mean = 95)
  (h_required_sum : required_sum = 6 * target_mean) (h_current_sum : current_sum = q1 + q2 + q3 + q4 + q5)
  (h6 : s6 = required_sum - current_sum) :
  s6 = 98 :=
by
  sorry

end NUMINAMATH_GPT_emily_sixth_quiz_score_l816_81688


namespace NUMINAMATH_GPT_quadratic_solution_l816_81606

theorem quadratic_solution 
  (x : ℝ)
  (h : x^2 - 2 * x - 1 = 0) : 
  x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_l816_81606


namespace NUMINAMATH_GPT_probability_both_in_picture_l816_81651

-- Define the conditions
def completes_lap (laps_time: ℕ) (time: ℕ) : ℕ := time / laps_time

def position_into_lap (laps_time: ℕ) (time: ℕ) : ℕ := time % laps_time

-- Define the positions of Rachel and Robert
def rachel_position (time: ℕ) : ℚ :=
  let rachel_lap_time := 100
  let laps_completed := completes_lap rachel_lap_time time
  let time_into_lap := position_into_lap rachel_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / rachel_lap_time

def robert_position (time: ℕ) : ℚ :=
  let robert_lap_time := 70
  let laps_completed := completes_lap robert_lap_time time
  let time_into_lap := position_into_lap robert_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / robert_lap_time

-- Define the probability that both are in the picture
theorem probability_both_in_picture :
  let rachel_lap_time := 100
  let robert_lap_time := 70
  let start_time := 720
  let end_time := 780
  ∃ (overlap_time: ℚ) (total_time: ℚ),
    overlap_time / total_time = 1 / 16 :=
sorry

end NUMINAMATH_GPT_probability_both_in_picture_l816_81651


namespace NUMINAMATH_GPT_number_of_keepers_l816_81682

theorem number_of_keepers (k : ℕ)
  (hens : ℕ := 50)
  (goats : ℕ := 45)
  (camels : ℕ := 8)
  (hen_feet : ℕ := 2)
  (goat_feet : ℕ := 4)
  (camel_feet : ℕ := 4)
  (keeper_feet : ℕ := 2)
  (feet_more_than_heads : ℕ := 224)
  (total_heads : ℕ := hens + goats + camels + k)
  (total_feet : ℕ := (hens * hen_feet) + (goats * goat_feet) + (camels * camel_feet) + (k * keeper_feet)):
  total_feet = total_heads + feet_more_than_heads → k = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_keepers_l816_81682


namespace NUMINAMATH_GPT_decagon_diagonals_l816_81647

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_GPT_decagon_diagonals_l816_81647


namespace NUMINAMATH_GPT_gerald_jail_time_l816_81612

theorem gerald_jail_time
    (assault_sentence : ℕ := 3) 
    (poisoning_sentence_years : ℕ := 2) 
    (third_offense_extension : ℕ := 1 / 3) 
    (months_in_year : ℕ := 12)
    : (assault_sentence + poisoning_sentence_years * months_in_year) * (1 + third_offense_extension) = 36 :=
by
  sorry

end NUMINAMATH_GPT_gerald_jail_time_l816_81612


namespace NUMINAMATH_GPT_harry_has_19_apples_l816_81684

def apples_problem := 
  let A_M := 68  -- Martha's apples
  let A_T := A_M - 30  -- Tim's apples (68 - 30)
  let A_H := A_T / 2  -- Harry's apples (38 / 2)
  A_H = 19

theorem harry_has_19_apples : apples_problem :=
by
  -- prove A_H = 19 given the conditions
  sorry

end NUMINAMATH_GPT_harry_has_19_apples_l816_81684


namespace NUMINAMATH_GPT_shenille_points_l816_81668

def shenille_total_points (x y : ℕ) : ℝ :=
  0.6 * x + 0.6 * y

theorem shenille_points (x y : ℕ) (h : x + y = 30) : 
  shenille_total_points x y = 18 := by
  sorry

end NUMINAMATH_GPT_shenille_points_l816_81668


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l816_81620

theorem no_positive_integer_solutions (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2 * n) * y * (y + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l816_81620


namespace NUMINAMATH_GPT_simplify_tan_cot_expr_l816_81627

theorem simplify_tan_cot_expr :
  let tan_45 := 1
  let cot_45 := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 :=
by
  let tan_45 := 1
  let cot_45 := 1
  sorry

end NUMINAMATH_GPT_simplify_tan_cot_expr_l816_81627


namespace NUMINAMATH_GPT_problem_solution_l816_81615

theorem problem_solution (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) 
  (h5 : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l816_81615


namespace NUMINAMATH_GPT_max_sum_x_y_l816_81666

theorem max_sum_x_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x^3 + y^3 + (x + y)^3 + 36 * x * y = 3456) : x + y ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_sum_x_y_l816_81666


namespace NUMINAMATH_GPT_main_problem_l816_81602

def arithmetic_sequence (a : ℕ → ℕ) : Prop := ∃ a₁ d, ∀ n, a (n + 1) = a₁ + n * d

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2

def another_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop := ∀ n, b n = 1 / (a n * a (n + 1))

theorem main_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : a_3 = 5) 
  (h2 : S_3 = 9) 
  (h3 : arithmetic_sequence a)
  (h4 : sequence_sum a S)
  (h5 : another_sequence b a) : 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n = n / (2 * n + 1)) := sorry

end NUMINAMATH_GPT_main_problem_l816_81602


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l816_81600

open Real

theorem trajectory_of_midpoint (A : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A = (-2, 0))
    (hP_on_curve : P.1 = 2 * P.2 ^ 2)
    (hM_midpoint : M = ((A.1 + P.1) / 2, (A.2 + P.2) / 2)) :
    M.1 = 4 * M.2 ^ 2 - 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l816_81600


namespace NUMINAMATH_GPT_avg_adults_proof_l816_81683

variable (n_total : ℕ) (n_girls : ℕ) (n_boys : ℕ) (n_adults : ℕ)
variable (avg_total : ℕ) (avg_girls : ℕ) (avg_boys : ℕ)

def avg_age_adults (n_total n_girls n_boys n_adults avg_total avg_girls avg_boys : ℕ) : ℕ :=
  let sum_total := n_total * avg_total
  let sum_girls := n_girls * avg_girls
  let sum_boys := n_boys * avg_boys
  let sum_adults := sum_total - sum_girls - sum_boys
  sum_adults / n_adults

theorem avg_adults_proof :
  avg_age_adults 50 25 20 5 21 18 20 = 40 := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_avg_adults_proof_l816_81683


namespace NUMINAMATH_GPT_not_chosen_rate_l816_81663

theorem not_chosen_rate (sum : ℝ) (interest_15_percent : ℝ) (extra_interest : ℝ) : 
  sum = 7000 ∧ interest_15_percent = 2100 ∧ extra_interest = 420 →
  ∃ R : ℝ, (sum * 0.15 * 2 = interest_15_percent) ∧ 
           (interest_15_percent - (sum * R / 100 * 2) = extra_interest) ∧ 
           R = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_not_chosen_rate_l816_81663


namespace NUMINAMATH_GPT_population_definition_l816_81611

variable (students : Type) (weights : students → ℝ) (sample : Fin 50 → students)
variable (total_students : Fin 300 → students)
variable (is_selected : students → Prop)

theorem population_definition :
    (∀ s, is_selected s ↔ ∃ i, sample i = s) →
    (population = {w : ℝ | ∃ s, w = weights s}) ↔
    (population = {w : ℝ | ∃ s, w = weights s ∧ ∃ i, total_students i = s}) := by
  sorry

end NUMINAMATH_GPT_population_definition_l816_81611


namespace NUMINAMATH_GPT_company_initial_bureaus_l816_81698

theorem company_initial_bureaus (B : ℕ) (offices : ℕ) (extra_bureaus : ℕ) 
  (h1 : offices = 14) 
  (h2 : extra_bureaus = 10) 
  (h3 : (B + extra_bureaus) % offices = 0) : 
  B = 8 := 
by
  sorry

end NUMINAMATH_GPT_company_initial_bureaus_l816_81698


namespace NUMINAMATH_GPT_find_a3_l816_81678

-- Define the polynomial equality
def polynomial_equality (x : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) :=
  (1 + x) * (2 - x)^6 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7

-- State the main theorem
theorem find_a3 (a0 a1 a2 a4 a5 a6 a7 : ℝ) :
  (∃ (x : ℝ), polynomial_equality x a0 a1 a2 (-25) a4 a5 a6 a7) :=
sorry

end NUMINAMATH_GPT_find_a3_l816_81678


namespace NUMINAMATH_GPT_equation_of_tangent_line_l816_81608

theorem equation_of_tangent_line (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + a * y - 17 = 0) →
   (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 4 * x - 3 * y + 11 = 0) :=
sorry

end NUMINAMATH_GPT_equation_of_tangent_line_l816_81608


namespace NUMINAMATH_GPT_find_a_l816_81697

open Real

def point_in_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x + 4 * y + 4 = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 3 = 0

theorem find_a (a : ℝ) :
  point_in_circle 1 a →
  line_equation 1 a →
  a = -2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_a_l816_81697


namespace NUMINAMATH_GPT_Sam_balloon_count_l816_81644

theorem Sam_balloon_count:
  ∀ (F M S : ℕ), F = 5 → M = 7 → (F + M + S = 18) → S = 6 :=
by
  intros F M S hF hM hTotal
  rw [hF, hM] at hTotal
  linarith

end NUMINAMATH_GPT_Sam_balloon_count_l816_81644


namespace NUMINAMATH_GPT_rationalize_denominator_sum_l816_81639

theorem rationalize_denominator_sum :
  let expr := 1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)
  ∃ (A B C D E F G H I : ℤ), 
    I > 0 ∧
    expr * (Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 11) /
    ((Real.sqrt 5 + Real.sqrt 3)^2 - (Real.sqrt 11)^2) = 
        (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + 
         G * Real.sqrt H) / I ∧
    (A + B + C + D + E + F + G + H + I) = 225 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_sum_l816_81639


namespace NUMINAMATH_GPT_students_on_field_trip_l816_81674

theorem students_on_field_trip (vans: ℕ) (capacity_per_van: ℕ) (adults: ℕ) 
  (H_vans: vans = 3) 
  (H_capacity_per_van: capacity_per_van = 5) 
  (H_adults: adults = 3) : 
  (vans * capacity_per_van - adults = 12) :=
by
  sorry

end NUMINAMATH_GPT_students_on_field_trip_l816_81674


namespace NUMINAMATH_GPT_negation_p_l816_81618

def nonneg_reals := { x : ℝ // 0 ≤ x }

def p := ∀ x : nonneg_reals, Real.exp x.1 ≥ 1

theorem negation_p :
  ¬ p ↔ ∃ x : nonneg_reals, Real.exp x.1 < 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_p_l816_81618


namespace NUMINAMATH_GPT_solve_rational_equation_l816_81670

theorem solve_rational_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ↔ x = -3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_rational_equation_l816_81670


namespace NUMINAMATH_GPT_max_sum_of_distinct_integers_l816_81622

theorem max_sum_of_distinct_integers (A B C : ℕ) (hABC_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (hProduct : A * B * C = 1638) :
  A + B + C ≤ 126 :=
sorry

end NUMINAMATH_GPT_max_sum_of_distinct_integers_l816_81622


namespace NUMINAMATH_GPT_sum_of_solutions_eq_9_l816_81625

theorem sum_of_solutions_eq_9 (x_1 x_2 : ℝ) (h : x^2 - 9 * x + 20 = 0) :
  x_1 + x_2 = 9 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_9_l816_81625


namespace NUMINAMATH_GPT_standard_colony_condition_l816_81652

noncomputable def StandardBacterialColony : Prop := sorry

theorem standard_colony_condition (visible_mass_of_microorganisms : Prop) 
                                   (single_mother_cell : Prop) 
                                   (solid_culture_medium : Prop) 
                                   (not_multiple_types : Prop) 
                                   : StandardBacterialColony :=
sorry

end NUMINAMATH_GPT_standard_colony_condition_l816_81652


namespace NUMINAMATH_GPT_four_digit_even_numbers_divisible_by_4_l816_81614

noncomputable def number_of_4_digit_even_numbers_divisible_by_4 : Nat :=
  500

theorem four_digit_even_numbers_divisible_by_4 : 
  (∃ count : Nat, count = number_of_4_digit_even_numbers_divisible_by_4) :=
sorry

end NUMINAMATH_GPT_four_digit_even_numbers_divisible_by_4_l816_81614


namespace NUMINAMATH_GPT_find_smaller_number_l816_81696

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end NUMINAMATH_GPT_find_smaller_number_l816_81696


namespace NUMINAMATH_GPT_defeated_candidate_percentage_l816_81657

noncomputable def percentage_defeated_candidate (total_votes diff_votes invalid_votes : ℕ) : ℕ :=
  let valid_votes := total_votes - invalid_votes
  let P := 100 * (valid_votes - diff_votes) / (2 * valid_votes)
  P

theorem defeated_candidate_percentage (total_votes : ℕ) (diff_votes : ℕ) (invalid_votes : ℕ) :
  total_votes = 12600 ∧ diff_votes = 5000 ∧ invalid_votes = 100 → percentage_defeated_candidate total_votes diff_votes invalid_votes = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_defeated_candidate_percentage_l816_81657


namespace NUMINAMATH_GPT_system_solution_l816_81641

theorem system_solution (x y z : ℝ) 
  (h1 : x - y ≥ z)
  (h2 : x^2 + 4 * y^2 + 5 = 4 * z) :
  (x = 2 ∧ y = -0.5 ∧ z = 2.5) :=
sorry

end NUMINAMATH_GPT_system_solution_l816_81641


namespace NUMINAMATH_GPT_cylindrical_tank_volume_l816_81699

theorem cylindrical_tank_volume (d h : ℝ) (d_eq_20 : d = 20) (h_eq_10 : h = 10) : 
  π * ((d / 2) ^ 2) * h = 1000 * π :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_tank_volume_l816_81699


namespace NUMINAMATH_GPT_radius_of_circle_with_square_and_chord_l816_81630

theorem radius_of_circle_with_square_and_chord :
  ∃ (r : ℝ), 
    (∀ (chord_length square_side_length : ℝ), chord_length = 6 ∧ square_side_length = 2 → 
    (r = Real.sqrt 10)) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_with_square_and_chord_l816_81630
