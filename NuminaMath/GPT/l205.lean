import Mathlib

namespace correct_choice_is_B_l205_205334

def draw_ray := "Draw ray OP=3cm"
def connect_points := "Connect points A and B"
def draw_midpoint := "Draw the midpoint of points A and B"
def draw_distance := "Draw the distance between points A and B"

-- Mathematical function to identify the correct statement about drawing
def correct_drawing_statement (s : String) : Prop :=
  s = connect_points

theorem correct_choice_is_B :
  correct_drawing_statement connect_points :=
by
  sorry

end correct_choice_is_B_l205_205334


namespace simplify_expression_l205_205571

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := 
by 
sorry

end simplify_expression_l205_205571


namespace geom_prog_identity_l205_205672

-- Define that A, B, C are the n-th, p-th, and k-th terms respectively of the same geometric progression.
variables (a r : ℝ) (n p k : ℕ) (A B C : ℝ)

-- Assume A = ar^(n-1), B = ar^(p-1), C = ar^(k-1)
def isGP (a r : ℝ) (n p k : ℕ) (A B C : ℝ) : Prop :=
  A = a * r^(n-1) ∧ B = a * r^(p-1) ∧ C = a * r^(k-1)

-- Define the statement to be proved
theorem geom_prog_identity (h : isGP a r n p k A B C) : A^(p-k) * B^(k-n) * C^(n-p) = 1 :=
sorry

end geom_prog_identity_l205_205672


namespace problem_statement_l205_205455

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem problem_statement (S : ℝ) (h1 : S = golden_ratio) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 :=
by
  sorry

end problem_statement_l205_205455


namespace height_difference_l205_205220

variable {J L R : ℕ}

theorem height_difference
  (h1 : J = L + 15)
  (h2 : J = 152)
  (h3 : L + R = 295) :
  R - J = 6 :=
sorry

end height_difference_l205_205220


namespace pradeep_maximum_marks_l205_205529

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.20 * M = 185) : M = 925 :=
by
  sorry

end pradeep_maximum_marks_l205_205529


namespace height_of_boxes_l205_205155

-- Conditions
def total_volume : ℝ := 1.08 * 10^6
def cost_per_box : ℝ := 0.2
def total_monthly_cost : ℝ := 120

-- Target height of the boxes
def target_height : ℝ := 12.2

-- Problem: Prove that the height of each box is 12.2 inches
theorem height_of_boxes : 
  (total_monthly_cost / cost_per_box) * ((total_volume / (total_monthly_cost / cost_per_box))^(1/3)) = target_height := 
sorry

end height_of_boxes_l205_205155


namespace bread_slices_remaining_l205_205353

-- Conditions
def total_slices : ℕ := 12
def fraction_eaten_for_breakfast : ℕ := total_slices / 3
def slices_used_for_lunch : ℕ := 2

-- Mathematically Equivalent Proof Problem
theorem bread_slices_remaining : total_slices - fraction_eaten_for_breakfast - slices_used_for_lunch = 6 :=
by
  sorry

end bread_slices_remaining_l205_205353


namespace matrix_addition_l205_205979

variable (A B : Matrix (Fin 2) (Fin 2) ℤ) -- Define matrices with integer entries

-- Define the specific matrices used in the problem
def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![ ![2, 3], ![-1, 4] ]

def matrix_B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![-1, 8], ![-3, 0] ]

-- Define the result matrix
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![3, 14], ![-5, 8] ]

-- The theorem to prove
theorem matrix_addition : 2 • matrix_A + matrix_B = result_matrix := by
  sorry -- Proof omitted

end matrix_addition_l205_205979


namespace simplify_fraction_l205_205915

theorem simplify_fraction (x : ℝ) : (2 * x - 3) / 4 + (4 * x + 5) / 3 = (22 * x + 11) / 12 := by
  sorry

end simplify_fraction_l205_205915


namespace stacked_lego_volume_l205_205782

theorem stacked_lego_volume 
  (lego_volume : ℝ)
  (rows columns layers : ℕ)
  (h1 : lego_volume = 1)
  (h2 : rows = 7)
  (h3 : columns = 5)
  (h4 : layers = 3) :
  rows * columns * layers * lego_volume = 105 :=
by
  sorry

end stacked_lego_volume_l205_205782


namespace number_of_white_stones_is_3600_l205_205908

-- Definitions and conditions
def total_stones : ℕ := 6000
def total_difference_to_4800 : ℕ := 4800
def W : ℕ := 3600

-- Conditions
def condition1 (B : ℕ) : Prop := total_stones - W + B = total_difference_to_4800
def condition2 (B : ℕ) : Prop := W + B = total_stones
def condition3 (B : ℕ) : Prop := W > B

-- Theorem statement
theorem number_of_white_stones_is_3600 :
  ∃ B : ℕ, condition1 B ∧ condition2 B ∧ condition3 B :=
by
  -- TODO: Complete the proof
  sorry

end number_of_white_stones_is_3600_l205_205908


namespace volleyball_practice_start_time_l205_205558

def homework_time := 1 * 60 + 59  -- convert 1:59 p.m. to minutes since 12:00 p.m.
def homework_duration := 96        -- duration in minutes
def buffer_time := 25              -- time between finishing homework and practice
def practice_start_time := 4 * 60  -- convert 4:00 p.m. to minutes since 12:00 p.m.

theorem volleyball_practice_start_time :
  homework_time + homework_duration + buffer_time = practice_start_time := 
by
  sorry

end volleyball_practice_start_time_l205_205558


namespace exists_same_color_points_distance_one_l205_205934

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_distance_one_l205_205934


namespace problem_part1_problem_part2_problem_part3_l205_205214

variable (a b x : ℝ) (p q : ℝ) (n x1 x2 : ℝ)
variable (h1 : x1 = -2) (h2 : x2 = 3)
variable (h3 : x1 < x2)

def equation1 := x + p / x = q
def solution1_p := p = -6
def solution1_q := q = 1

def equation2 := x + 7 / x = 8
def solution2 := x1 = 7

def equation3 := 2 * x + (n^2 - n) / (2 * x - 1) = 2 * n
def solution3 := (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)

theorem problem_part1 : ∀ (x : ℝ), (x + -6 / x = 1) → (p = -6 ∧ q = 1) := by
  sorry

theorem problem_part2 : (max 7 1 = 7) := by
  sorry

theorem problem_part3 : ∀ (n : ℝ), (∃ x1 x2, x1 < x2 ∧ (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)) := by
  sorry

end problem_part1_problem_part2_problem_part3_l205_205214


namespace find_x_l205_205468

-- Conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 8 * x
def area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x / 2

-- Theorem to prove
theorem find_x (x s : ℝ) (h1 : volume_condition x s) (h2 : area_condition x s) : x = 110592 := sorry

end find_x_l205_205468


namespace transactions_proof_l205_205998

def transactions_problem : Prop :=
  let mabel_transactions := 90
  let anthony_transactions := mabel_transactions + (0.10 * mabel_transactions)
  let cal_transactions := (2 / 3) * anthony_transactions
  let jade_transactions := 81
  jade_transactions - cal_transactions = 15

-- The proof is omitted (replace 'sorry' with an actual proof)
theorem transactions_proof : transactions_problem := by
  sorry

end transactions_proof_l205_205998


namespace lines_perpendicular_l205_205188

theorem lines_perpendicular 
  (x y : ℝ)
  (first_angle : ℝ)
  (second_angle : ℝ)
  (h1 : first_angle = 50 + x - y)
  (h2 : second_angle = first_angle - (10 + 2 * x - 2 * y)) :
  first_angle + second_angle = 90 :=
by 
  sorry

end lines_perpendicular_l205_205188


namespace solve_for_a_l205_205308

variable (x y a : ℤ)
variable (hx : x = 1)
variable (hy : y = -3)
variable (eq : a * x - y = 1)
 
theorem solve_for_a : a = -2 := by
  -- Placeholder to satisfy the lean prover, no actual proof steps
  sorry

end solve_for_a_l205_205308


namespace find_ab_l205_205250

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 :=
by
  sorry

end find_ab_l205_205250


namespace horizontal_asymptote_value_l205_205542

theorem horizontal_asymptote_value :
  ∀ (x : ℝ),
  ((8 * x^4 + 6 * x^3 + 7 * x^2 + 2 * x + 4) / 
  (2 * x^4 + 5 * x^3 + 3 * x^2 + x + 6)) = (4 : ℝ) :=
by sorry

end horizontal_asymptote_value_l205_205542


namespace total_points_scored_l205_205726

theorem total_points_scored (points_per_round : ℕ) (rounds : ℕ) (h1 : points_per_round = 42) (h2 : rounds = 2) : 
  points_per_round * rounds = 84 :=
by
  sorry

end total_points_scored_l205_205726


namespace maximize_village_value_l205_205462

theorem maximize_village_value :
  ∃ (x y z : ℕ), 
  x + y + z = 20 ∧ 
  2 * x + 3 * y + 4 * z = 50 ∧ 
  (∀ x' y' z' : ℕ, 
      x' + y' + z' = 20 → 2 * x' + 3 * y' + 4 * z' = 50 → 
      (1.2 * x + 1.5 * y + 1.2 * z : ℝ) ≥ (1.2 * x' + 1.5 * y' + 1.2 * z' : ℝ)) ∧ 
  x = 10 ∧ y = 10 ∧ z = 0 := by 
  sorry

end maximize_village_value_l205_205462


namespace hyperbola_equation_l205_205129

noncomputable def distance_between_vertices : ℝ := 8
noncomputable def eccentricity : ℝ := 5 / 4

theorem hyperbola_equation :
  ∃ a b c : ℝ, 2 * a = distance_between_vertices ∧ 
               c = a * eccentricity ∧ 
               b^2 = c^2 - a^2 ∧ 
               (a = 4 ∧ c = 5 ∧ b^2 = 9) ∧ 
               ∀ x y : ℝ, (x^2 / (a:ℝ)^2) - (y^2 / (b:ℝ)^2) = 1 :=
by 
  sorry

end hyperbola_equation_l205_205129


namespace adam_completes_work_in_10_days_l205_205792

theorem adam_completes_work_in_10_days (W : ℝ) (A : ℝ)
  (h1 : (W / 25) + A = W / 20) :
  W / 10 = (W / 100) * 10 :=
by
  sorry

end adam_completes_work_in_10_days_l205_205792


namespace honey_last_nights_l205_205827

def servings_per_cup : Nat := 1
def cups_per_night : Nat := 2
def container_ounces : Nat := 16
def servings_per_ounce : Nat := 6

theorem honey_last_nights :
  (container_ounces * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 :=
by
  sorry  -- Proof not provided as per requirements

end honey_last_nights_l205_205827


namespace evaluate_expression_at_2_l205_205074

theorem evaluate_expression_at_2 : (3^2 - 2^3) = 1 := 
by
  sorry

end evaluate_expression_at_2_l205_205074


namespace tangent_line_through_B_l205_205925

theorem tangent_line_through_B (x : ℝ) (y : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  (y₀ = x₀^2) →
  (y - y₀ = 2*x₀*(x - x₀)) →
  (3, 5) ∈ ({p : ℝ × ℝ | ∃ t, p.2 - t^2 = 2*t*(p.1 - t)}) →
  (x = 2 * x₀) ∧ (y = y₀) →
  (2*x - y - 1 = 0 ∨ 10*x - y - 25 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end tangent_line_through_B_l205_205925


namespace order_of_abc_l205_205284

noncomputable def a := Real.log 6 / Real.log 0.7
noncomputable def b := Real.rpow 6 0.7
noncomputable def c := Real.rpow 0.7 0.6

theorem order_of_abc : b > c ∧ c > a := by
  sorry

end order_of_abc_l205_205284


namespace find_point_A_coordinates_l205_205306

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end find_point_A_coordinates_l205_205306


namespace speed_of_A_is_3_l205_205868

theorem speed_of_A_is_3:
  (∃ x : ℝ, 3 * x + 3 * (x + 2) = 24) → x = 3 :=
by
  sorry

end speed_of_A_is_3_l205_205868


namespace right_angle_triangle_exists_l205_205052

theorem right_angle_triangle_exists (color : ℤ × ℤ → ℕ) (H1 : ∀ c : ℕ, ∃ p : ℤ × ℤ, color p = c) : 
  ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (color A ≠ color B ∧ color B ≠ color C ∧ color C ≠ color A) ∧
  ((A.1 = B.1 ∧ B.2 = C.2 ∧ A.1 - C.1 = A.2 - B.2) ∨ (A.2 = B.2 ∧ B.1 = C.1 ∧ A.1 - B.1 = A.2 - C.2)) :=
sorry

end right_angle_triangle_exists_l205_205052


namespace selection_methods_count_l205_205725

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem selection_methods_count :
  let females := 8
  let males := 4
  (binomial females 2 * binomial males 1) + (binomial females 1 * binomial males 2) = 112 :=
by
  sorry

end selection_methods_count_l205_205725


namespace washing_machine_capacity_l205_205363

-- Define the problem conditions
def families : Nat := 3
def people_per_family : Nat := 4
def days : Nat := 7
def towels_per_person_per_day : Nat := 1
def loads : Nat := 6

-- Define the statement to prove
theorem washing_machine_capacity :
  (families * people_per_family * days * towels_per_person_per_day) / loads = 14 := by
  sorry

end washing_machine_capacity_l205_205363


namespace johns_age_l205_205507

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l205_205507


namespace arccos_of_one_over_sqrt_two_l205_205601

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l205_205601


namespace camera_lens_distance_l205_205633

theorem camera_lens_distance (f u : ℝ) (h_fu : f ≠ u) (h_f : f ≠ 0) (h_u : u ≠ 0) :
  (∃ v : ℝ, (1 / f) = (1 / u) + (1 / v) ∧ v = (f * u) / (u - f)) :=
by {
  sorry
}

end camera_lens_distance_l205_205633


namespace ratio_of_good_states_l205_205701

theorem ratio_of_good_states (n : ℕ) :
  let total_states := 2^(2*n)
  let good_states := Nat.choose (2 * n) n
  good_states / total_states = (List.range n).foldr (fun i acc => acc * (2*i+1)) 1 / (2^n * Nat.factorial n) := sorry

end ratio_of_good_states_l205_205701


namespace points_calculation_correct_l205_205876

-- Definitions
def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_undestroyed : ℕ := 3
def enemies_destroyed : ℕ := total_enemies - enemies_undestroyed

def points_earned : ℕ := enemies_destroyed * points_per_enemy

-- Theorem statement
theorem points_calculation_correct : points_earned = 72 := by
  sorry

end points_calculation_correct_l205_205876


namespace person_B_processes_components_l205_205142

theorem person_B_processes_components (x : ℕ) (h1 : ∀ x, x > 0 → x + 2 > 0) 
(h2 : ∀ x, x > 0 → (25 / (x + 2)) = (20 / x)) :
  x = 8 := sorry

end person_B_processes_components_l205_205142


namespace no_pairs_xy_perfect_square_l205_205618

theorem no_pairs_xy_perfect_square :
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ ∃ k : ℕ, (xy + 1) * (xy + x + 2) = k^2 := 
by {
  sorry
}

end no_pairs_xy_perfect_square_l205_205618


namespace sum_is_24000_l205_205241

theorem sum_is_24000 (P : ℝ) (R : ℝ) (T : ℝ) : 
  (R = 5) → (T = 2) →
  ((P * (1 + R / 100)^T - P) - (P * R * T / 100) = 60) →
  P = 24000 :=
by
  sorry

end sum_is_24000_l205_205241


namespace find_xyz_l205_205349

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 198) (h5 : y * (z + x) = 216) (h6 : z * (x + y) = 234) :
  x * y * z = 1080 :=
sorry

end find_xyz_l205_205349


namespace time_walking_each_day_l205_205606

variable (days : Finset ℕ) (d1 : ℕ) (d2 : ℕ) (W : ℕ)

def time_spent_parking (days : Finset ℕ) : ℕ :=
  5 * days.card

def time_spent_metal_detector : ℕ :=
  2 * 30 + 3 * 10

def total_timespent (d1 d2 W : ℕ) : ℕ :=
  d1 + d2 + W

theorem time_walking_each_day (total_minutes : ℕ) (total_days : ℕ):
  total_timespent (time_spent_parking days) (time_spent_metal_detector) (total_minutes - time_spent_metal_detector - 5 * total_days)
  = total_minutes → W = 3 := by
  sorry

end time_walking_each_day_l205_205606


namespace melted_mixture_weight_l205_205836

variable (zinc copper total_weight : ℝ)
variable (ratio_zinc ratio_copper : ℝ := 9 / 11)
variable (weight_zinc : ℝ := 31.5)

theorem melted_mixture_weight :
  (zinc / copper = ratio_zinc / ratio_copper) ∧ (zinc = weight_zinc) →
  (total_weight = zinc + copper) →
  total_weight = 70 := 
sorry

end melted_mixture_weight_l205_205836


namespace am_minus_hm_lt_bound_l205_205204

theorem am_minus_hm_lt_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  (x - y)^2 / (2 * (x + y)) < (x - y)^2 / (8 * x) := 
by
  sorry

end am_minus_hm_lt_bound_l205_205204


namespace maria_total_distance_l205_205344

-- Definitions
def total_distance (D : ℝ) : Prop :=
  let d1 := D/2   -- Distance traveled before first stop
  let r1 := D - d1 -- Distance remaining after first stop
  let d2 := r1/4  -- Distance traveled before second stop
  let r2 := r1 - d2 -- Distance remaining after second stop
  let d3 := r2/3  -- Distance traveled before third stop
  let r3 := r2 - d3 -- Distance remaining after third stop
  r3 = 270 -- Remaining distance after third stop equals 270 miles

-- Theorem statement
theorem maria_total_distance : ∃ D : ℝ, total_distance D ∧ D = 1080 :=
sorry

end maria_total_distance_l205_205344


namespace range_of_a_l205_205293

theorem range_of_a 
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x < 0, f x = a^x)
  (h2 : ∀ x ≥ 0, f x = (a - 3) * x + 4 * a)
  (h3 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  0 < a ∧ a ≤ 1 / 4 :=
sorry

end range_of_a_l205_205293


namespace ceil_and_floor_difference_l205_205961

theorem ceil_and_floor_difference (x : ℝ) (ε : ℝ) 
  (h_cond : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) (h_eps : 0 < ε ∧ ε < 1) :
  ⌈x + ε⌉ - (x + ε) = 1 - ε :=
sorry

end ceil_and_floor_difference_l205_205961


namespace geom_series_sum_n_eq_728_div_729_l205_205999

noncomputable def a : ℚ := 1 / 3
noncomputable def r : ℚ := 1 / 3
noncomputable def S_n (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))

theorem geom_series_sum_n_eq_728_div_729 (n : ℕ) (h : S_n n = 728 / 729) : n = 6 :=
by
  sorry

end geom_series_sum_n_eq_728_div_729_l205_205999


namespace intersecting_lines_k_value_l205_205028

theorem intersecting_lines_k_value (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x + 5 ∧ y = -3 * x - 35 ∧ y = 4 * x + k) → k = -7 :=
by
  sorry

end intersecting_lines_k_value_l205_205028


namespace total_blue_balloons_l205_205042

theorem total_blue_balloons (Joan_balloons : ℕ) (Melanie_balloons : ℕ) (Alex_balloons : ℕ) 
  (hJoan : Joan_balloons = 60) (hMelanie : Melanie_balloons = 85) (hAlex : Alex_balloons = 37) :
  Joan_balloons + Melanie_balloons + Alex_balloons = 182 :=
by
  sorry

end total_blue_balloons_l205_205042


namespace simplify_expression_l205_205800
-- Import the entire Mathlib library to ensure all necessary lemmas and theorems are available

-- Define the main problem as a theorem
theorem simplify_expression (t : ℝ) : 
  (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end simplify_expression_l205_205800


namespace breadth_of_rectangular_plot_l205_205516

theorem breadth_of_rectangular_plot :
  ∃ b : ℝ, (∃ l : ℝ, l = 3 * b ∧ l * b = 867) ∧ b = 17 :=
by
  sorry

end breadth_of_rectangular_plot_l205_205516


namespace correct_diagram_is_B_l205_205727

-- Define the diagrams and their respected angles
def sector_angle_A : ℝ := 90
def sector_angle_B : ℝ := 135
def sector_angle_C : ℝ := 180

-- Define the target central angle for one third of the circle
def target_angle : ℝ := 120

-- The proof statement that Diagram B is the correct diagram with the sector angle closest to one third of the circle (120 degrees)
theorem correct_diagram_is_B (A B C : Prop) :
  (B = (sector_angle_A < target_angle ∧ target_angle < sector_angle_B)) := 
sorry

end correct_diagram_is_B_l205_205727


namespace arithmetic_sum_S8_l205_205088

theorem arithmetic_sum_S8 (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, S (n + 1) - S n = S 1 - S 0)
  (h_positive : ∀ n, S n > 0)
  (h_S4 : S 4 = 10)
  (h_S12 : S 12 = 130) : 
  S 8 = 40 :=
sorry

end arithmetic_sum_S8_l205_205088


namespace yards_gained_l205_205054

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end yards_gained_l205_205054


namespace total_students_l205_205091

theorem total_students (m f : ℕ) (h_ratio : 3 * f = 7 * m) (h_males : m = 21) : m + f = 70 :=
by
  sorry

end total_students_l205_205091


namespace slices_ratio_l205_205029

theorem slices_ratio (total_slices : ℕ) (hawaiian_slices : ℕ) (cheese_slices : ℕ) 
  (dean_hawaiian_eaten : ℕ) (frank_hawaiian_eaten : ℕ) (sammy_cheese_eaten : ℕ)
  (total_leftover : ℕ) (hawaiian_leftover : ℕ) (cheese_leftover : ℕ)
  (H1 : total_slices = 12)
  (H2 : hawaiian_slices = 12)
  (H3 : cheese_slices = 12)
  (H4 : dean_hawaiian_eaten = 6)
  (H5 : frank_hawaiian_eaten = 3)
  (H6 : total_leftover = 11)
  (H7 : hawaiian_leftover = hawaiian_slices - dean_hawaiian_eaten - frank_hawaiian_eaten)
  (H8 : cheese_leftover = total_leftover - hawaiian_leftover)
  (H9 : sammy_cheese_eaten = cheese_slices - cheese_leftover)
  : sammy_cheese_eaten / cheese_slices = 1 / 3 :=
by sorry

end slices_ratio_l205_205029


namespace maximum_regular_hours_is_40_l205_205882

-- Definitions based on conditions
def regular_pay_per_hour := 3
def overtime_pay_per_hour := 6
def total_payment_received := 168
def overtime_hours := 8
def overtime_earnings := overtime_hours * overtime_pay_per_hour
def regular_earnings := total_payment_received - overtime_earnings
def maximum_regular_hours := regular_earnings / regular_pay_per_hour

-- Lean theorem statement corresponding to the proof problem
theorem maximum_regular_hours_is_40 : maximum_regular_hours = 40 := by
  sorry

end maximum_regular_hours_is_40_l205_205882


namespace find_BE_l205_205545

-- Definitions from the conditions
variable {A B C D E : Point}
variable (AB BC CA BD BE CE : ℝ)
variable (angleBAE angleCAD : Real.Angle)

-- Given conditions
axiom h1 : AB = 12
axiom h2 : BC = 17
axiom h3 : CA = 15
axiom h4 : BD = 7
axiom h5 : angleBAE = angleCAD

-- Required proof statement
theorem find_BE :
  BE = 1632 / 201 := by
  sorry

end find_BE_l205_205545


namespace frogs_seen_in_pond_l205_205148

-- Definitions from the problem conditions
def initial_frogs_on_lily_pads : ℕ := 5
def frogs_on_logs : ℕ := 3
def baby_frogs_on_rock : ℕ := 2 * 12  -- Two dozen

-- The statement of the proof
theorem frogs_seen_in_pond : initial_frogs_on_lily_pads + frogs_on_logs + baby_frogs_on_rock = 32 :=
by sorry

end frogs_seen_in_pond_l205_205148


namespace base_10_to_base_7_equiv_base_10_to_base_7_678_l205_205416

theorem base_10_to_base_7_equiv : (678 : ℕ) = 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 := 
by
  -- proof steps would go here
  sorry

theorem base_10_to_base_7_678 : "678 in base-7" = "1656" := 
by
  have h1 := base_10_to_base_7_equiv
  -- additional proof steps to show 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 = 1656 in base-7
  sorry

end base_10_to_base_7_equiv_base_10_to_base_7_678_l205_205416


namespace yield_and_fertilization_correlated_l205_205574

-- Define the variables and conditions
def yield_of_crops : Type := sorry
def fertilization : Type := sorry

-- State the condition
def yield_depends_on_fertilization (Y : yield_of_crops) (F : fertilization) : Prop :=
  -- The yield of crops depends entirely on fertilization
  sorry

-- State the theorem with the given condition and the conclusion
theorem yield_and_fertilization_correlated {Y : yield_of_crops} {F : fertilization} :
  yield_depends_on_fertilization Y F → sorry := 
  -- There is a correlation between the yield of crops and fertilization
  sorry

end yield_and_fertilization_correlated_l205_205574


namespace original_sales_tax_percentage_l205_205832

theorem original_sales_tax_percentage
  (current_sales_tax : ℝ := 10 / 3) -- 3 1/3% in decimal
  (difference : ℝ := 10.999999999999991) -- Rs. 10.999999999999991
  (market_price : ℝ := 6600) -- Rs. 6600
  (original_sales_tax : ℝ := 3.5) -- Expected original tax
  :  ((original_sales_tax / 100) * market_price = (current_sales_tax / 100) * market_price + difference) 
  := sorry

end original_sales_tax_percentage_l205_205832


namespace algebraic_expression_l205_205433

def ast (n : ℕ) : ℕ := sorry

axiom condition_1 : ast 1 = 1
axiom condition_2 : ∀ (n : ℕ), ast (n + 1) = 3 * ast n

theorem algebraic_expression (n : ℕ) :
  n > 0 → ast n = 3^(n - 1) :=
by
  -- Proof to be completed
  sorry

end algebraic_expression_l205_205433


namespace total_plums_correct_l205_205117

/-- Each picked number of plums. -/
def melanie_picked := 4
def dan_picked := 9
def sally_picked := 3
def ben_picked := 2 * (melanie_picked + dan_picked)
def sally_ate := 2

/-- The total number of plums picked in the end. -/
def total_plums_picked :=
  melanie_picked + dan_picked + sally_picked + ben_picked - sally_ate

theorem total_plums_correct : total_plums_picked = 40 := by
  sorry

end total_plums_correct_l205_205117


namespace sequence_general_term_l205_205789

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ k ≥ 1, a (k + 1) = 2 * a k) : a n = 2 ^ (n - 1) :=
sorry

end sequence_general_term_l205_205789


namespace expression_evaluation_l205_205598

theorem expression_evaluation : 2^2 - Real.tan (Real.pi / 3) + abs (Real.sqrt 3 - 1) - (3 - Real.pi)^0 = 2 :=
by
  sorry

end expression_evaluation_l205_205598


namespace combined_weight_of_student_and_sister_l205_205621

theorem combined_weight_of_student_and_sister
  (S : ℝ) (R : ℝ)
  (h1 : S = 90)
  (h2 : S - 6 = 2 * R) :
  S + R = 132 :=
by
  sorry

end combined_weight_of_student_and_sister_l205_205621


namespace max_books_borrowed_l205_205326

noncomputable def max_books_per_student : ℕ := 14

theorem max_books_borrowed (students_borrowed_0 : ℕ)
                           (students_borrowed_1 : ℕ)
                           (students_borrowed_2 : ℕ)
                           (total_students : ℕ)
                           (average_books : ℕ)
                           (remaining_students_borrowed_at_least_3 : ℕ)
                           (total_books : ℕ)
                           (max_books : ℕ) 
  (h1 : students_borrowed_0 = 2)
  (h2 : students_borrowed_1 = 10)
  (h3 : students_borrowed_2 = 5)
  (h4 : total_students = 20)
  (h5 : average_books = 2)
  (h6 : remaining_students_borrowed_at_least_3 = total_students - students_borrowed_0 - students_borrowed_1 - students_borrowed_2)
  (h7 : total_books = total_students * average_books)
  (h8 : total_books = (students_borrowed_1 * 1 + students_borrowed_2 * 2) + remaining_students_borrowed_at_least_3 * 3 + (max_books - 6))
  (h_max : max_books = max_books_per_student) :
  max_books ≤ max_books_per_student := 
sorry

end max_books_borrowed_l205_205326


namespace cody_steps_l205_205484

theorem cody_steps (S steps_week1 steps_week2 steps_week3 steps_week4 total_steps_4weeks : ℕ) 
  (h1 : steps_week1 = 7 * S) 
  (h2 : steps_week2 = 7 * (S + 1000)) 
  (h3 : steps_week3 = 7 * (S + 2000)) 
  (h4 : steps_week4 = 7 * (S + 3000)) 
  (h5 : total_steps_4weeks = steps_week1 + steps_week2 + steps_week3 + steps_week4) 
  (h6 : total_steps_4weeks = 70000) : 
  S = 1000 := 
    sorry

end cody_steps_l205_205484


namespace triangle_BC_value_l205_205153

theorem triangle_BC_value (B C A : ℝ) (AB AC BC : ℝ) 
  (hB : B = 45) 
  (hAB : AB = 100)
  (hAC : AC = 100)
  (h_deg : A ≠ 0) :
  BC = 100 * Real.sqrt 2 := 
by 
  sorry

end triangle_BC_value_l205_205153


namespace find_f_2015_l205_205315

variables (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) = f x + f 3

theorem find_f_2015 (h1 : is_even_function f) (h2 : satisfies_condition f) (h3 : f 1 = 2) : f 2015 = 2 :=
by
  sorry

end find_f_2015_l205_205315


namespace least_possible_integer_for_friends_statements_l205_205578

theorem least_possible_integer_for_friends_statements 
    (M : Nat)
    (statement_divisible_by : Nat → Prop)
    (h1 : ∀ n, 1 ≤ n ∧ n ≤ 30 → statement_divisible_by n = (M % n = 0))
    (h2 : ∃ m, 1 ≤ m ∧ m < 30 ∧ (statement_divisible_by m = false ∧ 
                                    statement_divisible_by (m + 1) = false)) :
    M = 12252240 :=
by
  sorry

end least_possible_integer_for_friends_statements_l205_205578


namespace vectors_coplanar_l205_205622

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (1, -3, -7)
def vector_c : ℝ × ℝ × ℝ := (1, 2, 3)

def scalar_triple_product (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product vector_a vector_b vector_c = 0 := 
by
  sorry

end vectors_coplanar_l205_205622


namespace bouquet_cost_l205_205958

theorem bouquet_cost (c₁ : ℕ) (r₁ r₂ : ℕ) (c_discount : ℕ) (discount_percentage: ℕ) :
  (c₁ = 30) → (r₁ = 15) → (r₂ = 45) → (c_discount = 81) → (discount_percentage = 10) → 
  ((c₂ : ℕ) → (c₂ = (c₁ * r₂) / r₁) → (r₂ > 30) → 
  (c_discount = c₂ - (c₂ * discount_percentage / 100))) → 
  c_discount = 81 :=
by
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end bouquet_cost_l205_205958


namespace range_of_f_lt_zero_l205_205960

noncomputable
def f : ℝ → ℝ := sorry

theorem range_of_f_lt_zero 
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_neg2_zero : f (-2) = 0) :
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end range_of_f_lt_zero_l205_205960


namespace average_books_per_month_l205_205556

-- Definitions based on the conditions
def books_sold_january : ℕ := 15
def books_sold_february : ℕ := 16
def books_sold_march : ℕ := 17
def total_books_sold : ℕ := books_sold_january + books_sold_february + books_sold_march
def number_of_months : ℕ := 3

-- The theorem we need to prove
theorem average_books_per_month : total_books_sold / number_of_months = 16 :=
by
  sorry

end average_books_per_month_l205_205556


namespace distances_inequality_l205_205226

theorem distances_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + 
  Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤ 
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + 
  Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 :=
  sorry

end distances_inequality_l205_205226


namespace externally_tangent_internally_tangent_common_chord_and_length_l205_205906

-- Definitions of Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + m = 0

-- Proof problem 1: Externally tangent
theorem externally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 + 10 * Real.sqrt 11 :=
sorry

-- Proof problem 2: Internally tangent
theorem internally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 - 10 * Real.sqrt 11 :=
sorry

-- Proof problem 3: Common chord and length when m = 45
theorem common_chord_and_length :
  (∃ x y : ℝ, circle2 x y 45) →
  (∃ l : ℝ, l = 4 * Real.sqrt 7 ∧ ∀ x y : ℝ, (circle1 x y ∧ circle2 x y 45) → (4*x + 3*y - 23 = 0)) :=
sorry

end externally_tangent_internally_tangent_common_chord_and_length_l205_205906


namespace part1_part2_l205_205816

section

variable (a x : ℝ)

def A : Set ℝ := { x | x ≤ -1 } ∪ { x | x ≥ 5 }
def B (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a + 2 }

-- Part 1
theorem part1 (h : a = -1) :
  B a = { x | -2 ≤ x ∧ x ≤ 1 } ∧
  (A ∩ B a) = { x | -2 ≤ x ∧ x ≤ -1 } ∧
  (A ∪ B a) = { x | x ≤ 1 ∨ x ≥ 5 } := 
sorry

-- Part 2
theorem part2 (h : A ∩ B a = B a) :
  a ≤ -3 ∨ a > 2 := 
sorry

end

end part1_part2_l205_205816


namespace algebraic_expression_interpretation_l205_205933

def donations_interpretation (m n : ℝ) : ℝ := 5 * m + 2 * n
def plazas_area_interpretation (a : ℝ) : ℝ := 6 * a^2

theorem algebraic_expression_interpretation (m n a : ℝ) :
  donations_interpretation m n = 5 * m + 2 * n ∧ plazas_area_interpretation a = 6 * a^2 :=
by
  sorry

end algebraic_expression_interpretation_l205_205933


namespace paint_cost_for_flag_l205_205106

noncomputable def flag_width : ℕ := 12
noncomputable def flag_height : ℕ := 10
noncomputable def paint_cost_per_quart : ℝ := 3.5
noncomputable def coverage_per_quart : ℕ := 4

theorem paint_cost_for_flag : (flag_width * flag_height * 2 / coverage_per_quart : ℝ) * paint_cost_per_quart = 210 := by
  sorry

end paint_cost_for_flag_l205_205106


namespace concentric_but_different_radius_l205_205597

noncomputable def circleF (x y : ℝ) : ℝ :=
  x^2 + y^2 - 1

def pointP (x : ℝ) : ℝ × ℝ :=
  (x, x)

def circleEquation (x y : ℝ) : Prop :=
  circleF x y = 0

def circleEquation' (x y : ℝ) : Prop :=
  circleF x y - circleF x y = 0

theorem concentric_but_different_radius (x : ℝ) (hP : circleF x x ≠ 0) (hCenter : x ≠ 0):
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧
    ∀ x y, (circleEquation x y ↔ x^2 + y^2 = 1) ∧ 
           (circleEquation' x y ↔ x^2 + y^2 = 2) :=
by
  sorry

end concentric_but_different_radius_l205_205597


namespace number_of_triangles_in_lattice_l205_205254

-- Define the triangular lattice structure
def triangular_lattice_rows : List ℕ := [1, 2, 3, 4]

-- Define the main theorem to state the number of triangles
theorem number_of_triangles_in_lattice :
  let number_of_triangles := 1 + 2 + 3 + 6 + 10
  number_of_triangles = 22 :=
by
  -- here goes the proof, which we skip with "sorry"
  sorry

end number_of_triangles_in_lattice_l205_205254


namespace soccer_players_count_l205_205393

theorem soccer_players_count (total_socks : ℕ) (P : ℕ) 
  (h_total_socks : total_socks = 22)
  (h_each_player_contributes : ∀ p : ℕ, p = P → total_socks = 2 * P) :
  P = 11 :=
by
  sorry

end soccer_players_count_l205_205393


namespace log_diff_condition_l205_205156

theorem log_diff_condition (a : ℕ → ℝ) (d e : ℝ) (H1 : ∀ n : ℕ, n > 1 → a n = Real.log n / Real.log 3003)
  (H2 : d = a 2 + a 3 + a 4 + a 5 + a 6) (H3 : e = a 15 + a 16 + a 17 + a 18 + a 19) :
  d - e = -Real.log 1938 / Real.log 3003 := by
  sorry

end log_diff_condition_l205_205156


namespace uncle_money_given_l205_205848

-- Definitions
def lizzy_mother_money : Int := 80
def lizzy_father_money : Int := 40
def candy_expense : Int := 50
def total_money_now : Int := 140

-- Theorem to prove
theorem uncle_money_given : (total_money_now - ((lizzy_mother_money + lizzy_father_money) - candy_expense)) = 70 := 
  by
    sorry

end uncle_money_given_l205_205848


namespace moles_of_C2H6_l205_205723

-- Define the reactive coefficients
def ratio_C := 2
def ratio_H2 := 3
def ratio_C2H6 := 1

-- Given conditions
def moles_C := 6
def moles_H2 := 9

-- Function to calculate moles of C2H6 formed
def moles_C2H6_formed (m_C : ℕ) (m_H2 : ℕ) : ℕ :=
  min (m_C * ratio_C2H6 / ratio_C) (m_H2 * ratio_C2H6 / ratio_H2)

-- Theorem statement: the number of moles of C2H6 formed is 3
theorem moles_of_C2H6 : moles_C2H6_formed moles_C moles_H2 = 3 :=
by {
  -- Sorry is used since we are not providing the proof here
  sorry
}

end moles_of_C2H6_l205_205723


namespace olivia_savings_l205_205811

noncomputable def compound_amount 
  (P : ℝ) -- Initial principal
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem olivia_savings :
  compound_amount 2500 0.045 2 21 = 5077.14 :=
by
  sorry

end olivia_savings_l205_205811


namespace complement_A_in_U_range_of_a_l205_205567

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def f (x : ℝ) : ℝ := (1 / (sqrt (x + 2))) + log (3 - x)
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 3}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < (2 * a - 1)}

theorem complement_A_in_U : compl A = {x | x ≤ -2 ∨ 3 ≤ x} :=
by {
  sorry
}

theorem range_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ Iic 2 :=
by {
  sorry
}

end complement_A_in_U_range_of_a_l205_205567


namespace findC_coordinates_l205_205355

-- Points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Defining points A, B, and stating that point C lies on the positive x-axis
def A : Point := {x := -4, y := -2}
def B : Point := {x := 0, y := -2}
def C (cx : ℝ) : Point := {x := cx, y := 0}

-- The condition that the triangle OBC is similar to triangle ABO
def isSimilar (A B O : Point) (C : Point) : Prop :=
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let OB := (B.x - O.x)^2 + (B.y - O.y)^2
  let OC := (C.x - O.x)^2 + (C.y - O.y)^2
  AB / OB = OB / OC

theorem findC_coordinates :
  ∃ (cx : ℝ), (C cx = {x := 1, y := 0} ∨ C cx = {x := 4, y := 0}) ∧
  isSimilar A B {x := 0, y := 0} (C cx) :=
by
  sorry

end findC_coordinates_l205_205355


namespace yuna_has_most_apples_l205_205417

def apples_count_jungkook : ℕ :=
  6 / 3

def apples_count_yoongi : ℕ :=
  4

def apples_count_yuna : ℕ :=
  5

theorem yuna_has_most_apples : apples_count_yuna > apples_count_yoongi ∧ apples_count_yuna > apples_count_jungkook :=
by
  sorry

end yuna_has_most_apples_l205_205417


namespace p_sufficient_not_necessary_for_q_l205_205346

open Real

def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

theorem p_sufficient_not_necessary_for_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l205_205346


namespace siblings_count_l205_205595

noncomputable def Masud_siblings (M : ℕ) : Prop :=
  (4 * M - 60 = (3 * M) / 4 + 135) → M = 60

theorem siblings_count (M : ℕ) : Masud_siblings M :=
  by
  sorry

end siblings_count_l205_205595


namespace expression_for_f_l205_205098

variable {R : Type*} [CommRing R]

def f (x : R) : R := sorry

theorem expression_for_f (x : R) :
  (f (x-1) = x^2 + 4*x - 5) → (f x = x^2 + 6*x) := by
  sorry

end expression_for_f_l205_205098


namespace find_side_a_in_triangle_l205_205637

noncomputable def triangle_side_a (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  cosA = 4/5 ∧ b = 2 ∧ S = 3 → a = Real.sqrt 13

-- Theorem statement with explicit conditions and proof goal
theorem find_side_a_in_triangle
  (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) :
  cosA = 4 / 5 → b = 2 → S = 3 → a = Real.sqrt 13 :=
by 
  intros 
  sorry

end find_side_a_in_triangle_l205_205637


namespace systematic_sampling_first_group_l205_205089

theorem systematic_sampling_first_group 
  (total_students sample_size group_size group_number drawn_number : ℕ)
  (h1 : total_students = 160)
  (h2 : sample_size = 20)
  (h3 : total_students = sample_size * group_size)
  (h4 : group_number = 16)
  (h5 : drawn_number = 126) 
  : (drawn_lots_first_group : ℕ) 
      = ((drawn_number - ((group_number - 1) * group_size + 1)) + 1) :=
sorry


end systematic_sampling_first_group_l205_205089


namespace John_total_amount_l205_205911

theorem John_total_amount (x : ℝ)
  (h1 : ∃ x : ℝ, (3 * x * 5 * 3 * x) = 300):
  (x + 3 * x + 15 * x) = 380 := by
  sorry

end John_total_amount_l205_205911


namespace kim_total_ounces_l205_205249

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end kim_total_ounces_l205_205249


namespace rabbit_hid_carrots_l205_205056

theorem rabbit_hid_carrots (h_r h_f : ℕ) (x : ℕ)
  (rabbit_holes : 5 * h_r = x) 
  (fox_holes : 7 * h_f = x)
  (holes_relation : h_r = h_f + 6) :
  x = 105 :=
by
  sorry

end rabbit_hid_carrots_l205_205056


namespace find_value_l205_205821

theorem find_value 
    (x y : ℝ) 
    (hx : x = 1 / (Real.sqrt 2 + 1)) 
    (hy : y = 1 / (Real.sqrt 2 - 1)) : 
    x^2 - 3 * x * y + y^2 = 3 := 
by 
    sorry

end find_value_l205_205821


namespace slopes_product_of_tangents_l205_205169

theorem slopes_product_of_tangents 
  (x₀ y₀ : ℝ) 
  (h_hyperbola : (2 * x₀^2) / 3 - y₀^2 / 6 = 1) 
  (h_outside_circle : x₀^2 + y₀^2 > 2) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ * k₂ = 4 ∧ 
    (y₀ - k₁ * x₀)^2 + k₁^2 = 2 ∧ 
    (y₀ - k₂ * x₀)^2 + k₂^2 = 2 :=
by {
  -- this proof will use the properties of tangents to a circle and the constraints given
  -- we don't need to implement it now, but we aim to show the correct relationship
  sorry
}

end slopes_product_of_tangents_l205_205169


namespace running_track_diameter_l205_205370

theorem running_track_diameter 
  (running_track_width : ℕ) 
  (garden_ring_width : ℕ) 
  (play_area_diameter : ℕ) 
  (h1 : running_track_width = 4) 
  (h2 : garden_ring_width = 6) 
  (h3 : play_area_diameter = 14) :
  (2 * ((play_area_diameter / 2) + garden_ring_width + running_track_width)) = 34 := 
by
  sorry

end running_track_diameter_l205_205370


namespace min_value_fraction_l205_205903

theorem min_value_fraction {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ y : ℝ,  y > 0 → (∀ x : ℝ, x > 0 → x + 3 * y = 1 → (1/x + 1/(3*y)) ≥ 4)) :=
sorry

end min_value_fraction_l205_205903


namespace proposition_false_n4_l205_205748

variable {P : ℕ → Prop}

theorem proposition_false_n4
  (h_ind : ∀ (k : ℕ), k ≠ 0 → P k → P (k + 1))
  (h_false_5 : P 5 = False) :
  P 4 = False :=
sorry

end proposition_false_n4_l205_205748


namespace ordered_triples_54000_l205_205660

theorem ordered_triples_54000 : 
  ∃ (count : ℕ), 
  count = 16 ∧ 
  ∀ (a b c : ℕ), 
  0 < a → 0 < b → 0 < c → a^4 * b^2 * c = 54000 → 
  count = 16 := 
sorry

end ordered_triples_54000_l205_205660


namespace k_equals_three_fourths_l205_205900

theorem k_equals_three_fourths : ∀ a b c d : ℝ, a ∈ Set.Ici (-1) → b ∈ Set.Ici (-1) → c ∈ Set.Ici (-1) → d ∈ Set.Ici (-1) →
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) :=
by
  intros
  sorry

end k_equals_three_fourths_l205_205900


namespace right_triangle_conditions_l205_205613

-- Definitions
def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

-- Conditions
def cond1 (A B C : ℝ) : Prop := A + B = C
def cond2 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def cond3 (A B C : ℝ) : Prop := A = B ∧ B = 2 * C
def cond4 (A B C : ℝ) : Prop := A = 2 * B ∧ B = 3 * C

-- Problem statement
theorem right_triangle_conditions (A B C : ℝ) :
  (cond1 A B C → is_right_triangle A B C) ∧
  (cond2 A B C → is_right_triangle A B C) ∧
  ¬(cond3 A B C → is_right_triangle A B C) ∧
  ¬(cond4 A B C → is_right_triangle A B C) :=
by
  sorry

end right_triangle_conditions_l205_205613


namespace system_has_negative_solution_iff_sum_zero_l205_205440

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l205_205440


namespace positive_number_sum_square_l205_205094

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_sum_square_l205_205094


namespace total_children_in_circle_l205_205645

theorem total_children_in_circle 
  (n : ℕ)  -- number of children
  (h_even : Even n)   -- condition: the circle is made up of an even number of children
  (h_pos : n > 0) -- condition: there are some children
  (h_opposite : (15 % n + 15 % n) % n = 0)  -- condition: the 15th child clockwise from Child A is facing Child A (implies opposite)
  : n = 30 := 
sorry

end total_children_in_circle_l205_205645


namespace family_vacation_days_l205_205797

theorem family_vacation_days
  (rained_days : ℕ)
  (total_days : ℕ)
  (clear_mornings : ℕ)
  (H1 : rained_days = 13)
  (H2 : total_days = 18)
  (H3 : clear_mornings = 11) :
  total_days = 18 :=
by
  -- proof to be filled in here
  sorry

end family_vacation_days_l205_205797


namespace smallest_n_transform_l205_205946

open Real

noncomputable def line1_angle : ℝ := π / 30
noncomputable def line2_angle : ℝ := π / 40
noncomputable def line_slope : ℝ := 2 / 45
noncomputable def transform_angle (theta : ℝ) (n : ℕ) : ℝ := theta + n * (7 * π / 120)

theorem smallest_n_transform (theta : ℝ) (n : ℕ) (m : ℕ)
  (h_line1 : line1_angle = π / 30)
  (h_line2 : line2_angle = π / 40)
  (h_slope : tan theta = line_slope)
  (h_transform : transform_angle theta n = theta + m * 2 * π) :
  n = 120 := 
sorry

end smallest_n_transform_l205_205946


namespace regular_polygon_sides_l205_205572

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l205_205572


namespace range_of_a_l205_205415

noncomputable def f (x a : ℝ) := x^2 + 2 * x - a
noncomputable def g (x : ℝ) := 2 * x + 2 * Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2, (1/e) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ e ∧ f x1 a = g x1 ∧ f x2 a = g x2) ↔ 
  1 < a ∧ a ≤ (1/(e^2)) + 2 := 
sorry

end range_of_a_l205_205415


namespace problem1_problem2_l205_205270

theorem problem1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 :=
by
  sorry

theorem problem2 : Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3 :=
by
  sorry

end problem1_problem2_l205_205270


namespace abc_inequality_l205_205365

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) :=
sorry

end abc_inequality_l205_205365


namespace smallest_x_abs_eq_15_l205_205688

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, |5 * x - 3| = 15 ∧ ∀ y : ℝ, |5 * y - 3| = 15 → x ≤ y :=
sorry

end smallest_x_abs_eq_15_l205_205688


namespace bargain_range_l205_205322

theorem bargain_range (cost_price lowest_cp highest_cp : ℝ)
  (h_lowest : lowest_cp = 50)
  (h_highest : highest_cp = 200 / 3)
  (h_marked_at : cost_price = 100)
  (h_lowest_markup : lowest_cp * 2 = cost_price)
  (h_highest_markup : highest_cp * 1.5 = cost_price)
  (profit_margin : ∀ (cp : ℝ), (cp * 1.2 ≥ cp)) : 
  (60 ≤ cost_price * 1.2 ∧ cost_price * 1.2 ≤ 80) :=
by
  sorry

end bargain_range_l205_205322


namespace total_potatoes_brought_home_l205_205768

def number_of_potatoes_each : ℕ := 8

theorem total_potatoes_brought_home (jane_potatoes mom_potatoes dad_potatoes : ℕ) :
  jane_potatoes = number_of_potatoes_each →
  mom_potatoes = number_of_potatoes_each →
  dad_potatoes = number_of_potatoes_each →
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end total_potatoes_brought_home_l205_205768


namespace perimeter_of_large_square_l205_205546

theorem perimeter_of_large_square (squares : List ℕ) (h : squares = [1, 1, 2, 3, 5, 8, 13]) : 2 * (21 + 13) = 68 := by
  sorry

end perimeter_of_large_square_l205_205546


namespace john_speed_above_limit_l205_205588

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem john_speed_above_limit :
  distance / time - speed_limit = 15 :=
by
  sorry

end john_speed_above_limit_l205_205588


namespace find_number_to_add_l205_205073

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l205_205073


namespace expressions_equal_iff_sum_zero_l205_205990

theorem expressions_equal_iff_sum_zero (p q r : ℝ) : (p + qr = (p + q) * (p + r)) ↔ (p + q + r = 0) :=
sorry

end expressions_equal_iff_sum_zero_l205_205990


namespace time_left_to_room_l205_205763

theorem time_left_to_room (total_time minutes_to_gate minutes_to_building : ℕ) 
  (h1 : total_time = 30) 
  (h2 : minutes_to_gate = 15) 
  (h3 : minutes_to_building = 6) : 
  total_time - (minutes_to_gate + minutes_to_building) = 9 :=
by 
  sorry

end time_left_to_room_l205_205763


namespace breadth_of_room_is_6_l205_205423

theorem breadth_of_room_is_6 
(the_room_length : ℝ) 
(the_carpet_width : ℝ) 
(cost_per_meter : ℝ) 
(total_cost : ℝ) 
(h1 : the_room_length = 15) 
(h2 : the_carpet_width = 0.75) 
(h3 : cost_per_meter = 0.30) 
(h4 : total_cost = 36) : 
  ∃ (breadth_of_room : ℝ), breadth_of_room = 6 :=
sorry

end breadth_of_room_is_6_l205_205423


namespace distance_borya_vasya_l205_205247

-- Definitions of the houses and distances on the road
def distance_andrey_gena : ℕ := 2450
def race_length : ℕ := 1000

-- Variables to represent the distances
variables (y b : ℕ)

-- Conditions
def start_position := y
def finish_position := b / 2 + 1225

axiom distance_eq : distance_andrey_gena = 2 * y
axiom race_distance_eq : finish_position - start_position = race_length

-- Proving the distance between Borya's and Vasya's houses
theorem distance_borya_vasya :
  ∃ (d : ℕ), d = 450 :=
by
  sorry

end distance_borya_vasya_l205_205247


namespace minimal_moves_for_7_disks_l205_205103

/-- Mathematical model of the Tower of Hanoi problem with special rules --/
def tower_of_hanoi_moves (n : ℕ) : ℚ :=
  if n = 7 then 23 / 4 else sorry

/-- Proof problem for the minimal number of moves required to transfer all seven disks to rod C --/
theorem minimal_moves_for_7_disks : tower_of_hanoi_moves 7 = 23 / 4 := 
  sorry

end minimal_moves_for_7_disks_l205_205103


namespace dance_problem_l205_205624

theorem dance_problem :
  ∃ (G : ℝ) (B T : ℝ),
    B / G = 3 / 4 ∧
    T = 0.20 * B ∧
    B + G + T = 114 ∧
    G = 60 :=
by
  sorry

end dance_problem_l205_205624


namespace solution_set_of_inequality_l205_205863

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = f x) (h2 : ∀ x, 0 ≤ x → f x = x - 1) :
  { x : ℝ | f (x - 1) > 1 } = { x | x < -1 ∨ x > 3 } :=
by
  sorry

end solution_set_of_inequality_l205_205863


namespace employee_pays_204_l205_205642

-- Definitions based on conditions
def wholesale_cost : ℝ := 200
def markup_percent : ℝ := 0.20
def discount_percent : ℝ := 0.15

def retail_price := wholesale_cost * (1 + markup_percent)
def employee_payment := retail_price * (1 - discount_percent)

-- Theorem with the expected result
theorem employee_pays_204 : employee_payment = 204 := by
  -- Proof not required, we add sorry to avoid the proof details
  sorry

end employee_pays_204_l205_205642


namespace area_of_rectangle_given_conditions_l205_205829

-- Defining the conditions given in the problem
variables (s d r a : ℝ)

-- Given conditions for the problem
def is_square_inscribed_in_circle (s d : ℝ) := 
  d = s * Real.sqrt 2 ∧ 
  d = 4

def is_circle_inscribed_in_rectangle (r : ℝ) :=
  r = 2

def rectangle_dimensions (length width : ℝ) :=
  length = 2 * width ∧ 
  width = 2

-- The theorem we want to prove
theorem area_of_rectangle_given_conditions :
  ∀ (s d r length width : ℝ),
  is_square_inscribed_in_circle s d →
  is_circle_inscribed_in_rectangle r →
  rectangle_dimensions length width →
  a = length * width →
  a = 8 :=
by
  intros s d r length width h1 h2 h3 h4
  sorry

end area_of_rectangle_given_conditions_l205_205829


namespace longer_side_of_rectangle_l205_205287

theorem longer_side_of_rectangle 
  (radius : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) 
  (h1 : radius = 6)
  (h2 : A_rectangle = 3 * (π * radius^2))
  (h3 : shorter_side = 2 * 2 * radius) :
  (A_rectangle / shorter_side) = 4.5 * π :=
by
  sorry

end longer_side_of_rectangle_l205_205287


namespace exists_x_geq_zero_l205_205732

theorem exists_x_geq_zero (h : ∀ x : ℝ, x^2 + x - 1 < 0) : ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
sorry

end exists_x_geq_zero_l205_205732


namespace gcd_positive_ints_l205_205001

theorem gcd_positive_ints (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hdiv : (a^2 + b^2) ∣ (a * c + b * d)) : 
  Nat.gcd (c^2 + d^2) (a^2 + b^2) > 1 := 
sorry

end gcd_positive_ints_l205_205001


namespace percentage_y_less_than_x_l205_205675

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 4 * y) : (x - y) / x * 100 = 75 := by
  sorry

end percentage_y_less_than_x_l205_205675


namespace nadine_hosing_time_l205_205402

theorem nadine_hosing_time (shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) 
  (h1 : shampoos = 3) (h2 : time_per_shampoo = 15) (h3 : total_cleaning_time = 55) : 
  ∃ t : ℕ, t = total_cleaning_time - shampoos * time_per_shampoo ∧ t = 10 := 
by
  sorry

end nadine_hosing_time_l205_205402


namespace ratio_of_integers_l205_205866

theorem ratio_of_integers (a b : ℤ) (h : 1996 * a + b / 96 = a + b) : a / b = 1 / 2016 ∨ b / a = 2016 :=
by
  sorry

end ratio_of_integers_l205_205866


namespace normal_price_of_article_l205_205285

theorem normal_price_of_article (P : ℝ) (h : 0.9 * 0.8 * P = 144) : P = 200 :=
sorry

end normal_price_of_article_l205_205285


namespace emily_annual_holidays_l205_205888

theorem emily_annual_holidays 
    (holidays_per_month : ℕ) 
    (months_in_year : ℕ) 
    (h1: holidays_per_month = 2)
    (h2: months_in_year = 12)
    : holidays_per_month * months_in_year = 24 := 
by
  sorry

end emily_annual_holidays_l205_205888


namespace four_distinct_real_roots_l205_205439

noncomputable def f (x d : ℝ) : ℝ := x^2 + 10*x + d

theorem four_distinct_real_roots (d : ℝ) :
  (∀ r, f r d = 0 → (∃! x, f x d = r)) → d < 25 :=
by
  sorry

end four_distinct_real_roots_l205_205439


namespace cat_can_pass_through_gap_l205_205002

theorem cat_can_pass_through_gap (R : ℝ) (h : ℝ) (π : ℝ) (hπ : π = Real.pi)
  (L₀ : ℝ) (L₁ : ℝ)
  (hL₀ : L₀ = 2 * π * R)
  (hL₁ : L₁ = L₀ + 1)
  (hL₁' : L₁ = 2 * π * (R + h)) :
  h = 1 / (2 * π) :=
by
  sorry

end cat_can_pass_through_gap_l205_205002


namespace isosceles_triangle_perimeter_l205_205110

-- Definitions for the side lengths
def side_a (x : ℝ) := 4 * x - 2
def side_b (x : ℝ) := x + 1
def side_c (x : ℝ) := 15 - 6 * x

-- Main theorem statement
theorem isosceles_triangle_perimeter (x : ℝ) (h1 : side_a x = side_b x ∨ side_a x = side_c x ∨ side_b x = side_c x) :
  (side_a x + side_b x + side_c x = 12.3) :=
  sorry

end isosceles_triangle_perimeter_l205_205110


namespace inequality_positive_numbers_l205_205165

theorem inequality_positive_numbers (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := 
sorry

end inequality_positive_numbers_l205_205165


namespace group_formations_at_fair_l205_205411

theorem group_formations_at_fair : 
  (Nat.choose 7 3) * (Nat.choose 4 4) = 35 := by
  sorry

end group_formations_at_fair_l205_205411


namespace max_vehicles_div_by_100_l205_205018

noncomputable def max_vehicles_passing_sensor (n : ℕ) : ℕ :=
  2 * (20000 * n / (5 + 10 * n))

theorem max_vehicles_div_by_100 : 
  (∀ n : ℕ, (n > 0) → (∃ M : ℕ, M = max_vehicles_passing_sensor n ∧ M / 100 = 40)) :=
sorry

end max_vehicles_div_by_100_l205_205018


namespace even_sum_exactly_one_even_l205_205023

theorem even_sum_exactly_one_even (a b c : ℕ) (h : (a + b + c) % 2 = 0) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  sorry

end even_sum_exactly_one_even_l205_205023


namespace third_grade_parts_in_batch_l205_205982

-- Define conditions
variable (x y s : ℕ) (h_first_grade : 24 = 24) (h_second_grade : 36 = 36)
variable (h_sample_size : 20 = 20) (h_sample_third_grade : 10 = 10)

-- The problem: Prove the total number of third-grade parts in the batch is 60 and the number of second-grade parts sampled is 6
open Nat

theorem third_grade_parts_in_batch
  (h_total_parts : x - y = 60)
  (h_third_grade_proportion : y = (1 / 2) * x)
  (h_second_grade_proportion : s = (36 / 120) * 20) :
  y = 60 ∧ s = 6 := by
  sorry

end third_grade_parts_in_batch_l205_205982


namespace diamonds_in_G15_l205_205031

theorem diamonds_in_G15 (G : ℕ → ℕ) 
  (h₁ : G 1 = 3)
  (h₂ : ∀ n, n ≥ 2 → G (n + 1) = 3 * (2 * (n - 1) + 3) - 3 ) :
  G 15 = 90 := sorry

end diamonds_in_G15_l205_205031


namespace lateral_surface_area_of_pyramid_l205_205493

theorem lateral_surface_area_of_pyramid
  (sin_alpha : ℝ)
  (A_section : ℝ)
  (h1 : sin_alpha = 15 / 17)
  (h2 : A_section = 3 * Real.sqrt 34) :
  ∃ A_lateral : ℝ, A_lateral = 68 :=
sorry

end lateral_surface_area_of_pyramid_l205_205493


namespace sum_of_permutations_of_1234567_l205_205667

theorem sum_of_permutations_of_1234567 : 
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10 ^ 7 - 1) / (10 - 1)
  sum_of_digits * factorial_7 * geometric_series_sum = 22399997760 :=
by
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10^7 - 1) / (10 - 1)
  sorry

end sum_of_permutations_of_1234567_l205_205667


namespace number_in_scientific_notation_l205_205356

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end number_in_scientific_notation_l205_205356


namespace greatest_distance_between_vertices_l205_205938

theorem greatest_distance_between_vertices 
    (inner_perimeter outer_perimeter : ℝ) 
    (inner_square_perimeter_eq : inner_perimeter = 16)
    (outer_square_perimeter_eq : outer_perimeter = 40)
    : ∃ max_distance, max_distance = 2 * Real.sqrt 34 :=
by
  sorry

end greatest_distance_between_vertices_l205_205938


namespace new_average_is_ten_l205_205652

-- Define the initial conditions
def initial_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : Prop :=
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 9 * 7

-- Define the transformation on the nine numbers
def transformed_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : ℝ :=
  (x₁ - 3) + (x₂ - 3) + (x₃ - 3) +
  (x₄ + 5) + (x₅ + 5) + (x₆ + 5) +
  (2 * x₇) + (2 * x₈) + (2 * x₉)

-- The theorem to prove the new average is 10
theorem new_average_is_ten (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h : initial_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉) :
  transformed_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ / 9 = 10 :=
by 
  sorry

end new_average_is_ten_l205_205652


namespace sum_of_solutions_l205_205253

theorem sum_of_solutions :
  (∃ S : Finset ℝ, (∀ x ∈ S, x^2 - 8*x + 21 = abs (x - 5) + 4) ∧ S.sum id = 18) :=
by
  sorry

end sum_of_solutions_l205_205253


namespace find_number_l205_205015

theorem find_number (x : ℝ) (h : 0.3 * x - (1 / 3) * (0.3 * x) = 36) : x = 180 :=
sorry

end find_number_l205_205015


namespace no_solution_k_l205_205917

theorem no_solution_k (k : ℝ) : 
  (∀ t s : ℝ, 
    ∃ (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (d : ℝ × ℝ), 
      (a = (2, 7)) ∧ 
      (b = (5, -9)) ∧ 
      (c = (4, -3)) ∧ 
      (d = (-2, k)) ∧ 
      (a + t • b ≠ c + s • d)) ↔ k = 18 / 5 := 
by
  sorry

end no_solution_k_l205_205917


namespace interval_of_a_l205_205833

theorem interval_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_monotone : ∀ x y, x < y → f y ≤ f x)
  (h_condition : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) : 
  a ∈ Set.Ioo 0 (1/3) ∪ Set.Ioo 1 5 :=
by
  sorry

end interval_of_a_l205_205833


namespace acid_solution_l205_205145

theorem acid_solution (n y : ℝ) (h : n > 30) (h1 : y = 15 * n / (n - 15)) :
  (n / 100) * n = ((n - 15) / 100) * (n + y) :=
by
  sorry

end acid_solution_l205_205145


namespace number_of_fish_disappeared_l205_205902

-- First, define initial amounts of each type of fish
def goldfish_initial := 7
def catfish_initial := 12
def guppies_initial := 8
def angelfish_initial := 5

-- Define the total initial number of fish
def total_fish_initial := goldfish_initial + catfish_initial + guppies_initial + angelfish_initial

-- Define the current number of fish
def fish_current := 27

-- Define the number of fish disappeared
def fish_disappeared := total_fish_initial - fish_current

-- Proof statement
theorem number_of_fish_disappeared:
  fish_disappeared = 5 :=
by
  -- Sorry is a placeholder that indicates the proof is omitted.
  sorry

end number_of_fish_disappeared_l205_205902


namespace find_value_of_a_l205_205806

-- Given conditions
def equation1 (x y : ℝ) : Prop := 4 * y + x + 5 = 0
def equation2 (x y : ℝ) (a : ℝ) : Prop := 3 * y + a * x + 4 = 0

-- The proof problem statement
theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y a → a = -12) :=
sorry

end find_value_of_a_l205_205806


namespace student_error_difference_l205_205380

theorem student_error_difference (num : ℤ) (num_val : num = 480) : 
  (5 / 6 * num - 5 / 16 * num) = 250 := 
by 
  sorry

end student_error_difference_l205_205380


namespace main_theorem_l205_205486

theorem main_theorem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * c * a) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end main_theorem_l205_205486


namespace find_wind_speed_l205_205257

-- Definitions from conditions
def speed_with_wind (j w : ℝ) := (j + w) * 6 = 3000
def speed_against_wind (j w : ℝ) := (j - w) * 9 = 3000

-- Theorem to prove the wind speed is 83.335 mph
theorem find_wind_speed (j w : ℝ) (h1 : speed_with_wind j w) (h2 : speed_against_wind j w) : w = 83.335 :=
by 
  -- Here we would prove the theorem using the given conditions
  sorry

end find_wind_speed_l205_205257


namespace complement_U_A_l205_205741

open Set

def U : Set ℤ := univ
def A : Set ℤ := { x | x^2 - x - 2 ≥ 0 }

theorem complement_U_A :
  (U \ A) = { 0, 1 } := by
  sorry

end complement_U_A_l205_205741


namespace focus_parabola_l205_205041

theorem focus_parabola (x : ℝ) (y : ℝ): (y = 8 * x^2) → (0, 1 / 32) = (0, 1 / 32) :=
by
  intro h
  sorry

end focus_parabola_l205_205041


namespace average_speed_of_train_l205_205198

theorem average_speed_of_train (x : ℝ) (h1 : x > 0): 
  (3 * x) / ((x / 40) + (2 * x / 20)) = 24 :=
by
  sorry

end average_speed_of_train_l205_205198


namespace abs_div_nonzero_l205_205623

theorem abs_div_nonzero (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  ¬ (|a| / a + |b| / b = 1) :=
by
  sorry

end abs_div_nonzero_l205_205623


namespace remainders_identical_l205_205359

theorem remainders_identical (a b : ℕ) (h1 : a > b) :
  ∃ r₁ r₂ q₁ q₂ : ℕ, 
  a = (a - b) * q₁ + r₁ ∧ 
  b = (a - b) * q₂ + r₂ ∧ 
  r₁ = r₂ := by 
sorry

end remainders_identical_l205_205359


namespace minimize_sum_dist_l205_205197

noncomputable section

variables {Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ}

-- Conditions
def clusters (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) :=
  Q3 <= Q1 + Q2 + Q4 / 3 ∧ Q3 = (Q1 + 2 * Q2 + 2 * Q4) / 5 ∧
  Q7 <= Q5 + Q6 + Q8 / 3 ∧ Q7 = (Q5 + 2 * Q6 + 2 * Q8) / 5

-- Sum of distances function
def sum_dist (Q : ℝ) (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) : ℝ :=
  abs (Q - Q1) + abs (Q - Q2) + abs (Q - Q3) + abs (Q - Q4) +
  abs (Q - Q5) + abs (Q - Q6) + abs (Q - Q7) + abs (Q - Q8) + abs (Q - Q9)

-- Theorem
theorem minimize_sum_dist (h : clusters Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) :
  ∃ Q : ℝ, (∀ Q' : ℝ, sum_dist Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 ≤ sum_dist Q' Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) → Q = Q5 :=
sorry

end minimize_sum_dist_l205_205197


namespace max_profit_at_800_l205_205740

open Nat

def P (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 80
  else if h : 100 < x ∧ x ≤ 1000 then 82 - 0.02 * x
  else 0

def f (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 30 * x
  else if h : 100 < x ∧ x ≤ 1000 then 32 * x - 0.02 * x^2
  else 0

theorem max_profit_at_800 :
  ∀ x : ℕ, f x ≤ 12800 ∧ f 800 = 12800 :=
sorry

end max_profit_at_800_l205_205740


namespace find_a7_l205_205112

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l205_205112


namespace provisions_last_days_l205_205655

def num_soldiers_initial : ℕ := 1200
def daily_consumption_initial : ℝ := 3
def initial_duration : ℝ := 30
def extra_soldiers : ℕ := 528
def daily_consumption_new : ℝ := 2.5

noncomputable def total_provisions : ℝ := num_soldiers_initial * daily_consumption_initial * initial_duration
noncomputable def total_soldiers_after_joining : ℕ := num_soldiers_initial + extra_soldiers
noncomputable def new_daily_consumption : ℝ := total_soldiers_after_joining * daily_consumption_new

theorem provisions_last_days : (total_provisions / new_daily_consumption) = 25 := by
  sorry

end provisions_last_days_l205_205655


namespace problem_l205_205465

theorem problem (a b : ℤ) (h : (2 * a + b) ^ 2 + |b - 2| = 0) : (-a - b) ^ 2014 = 1 := 
by
  sorry

end problem_l205_205465


namespace sum_of_integers_with_largest_proper_divisor_55_l205_205862

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def largest_proper_divisor (n d : ℕ) : Prop :=
  (d ∣ n) ∧ (d < n) ∧ ∀ e, (e ∣ n ∧ e < n ∧ e > d) → False

theorem sum_of_integers_with_largest_proper_divisor_55 : 
  (∀ n : ℕ, largest_proper_divisor n 55 → n = 110 ∨ n = 165 ∨ n = 275) →
  110 + 165 + 275 = 550 :=
by
  sorry

end sum_of_integers_with_largest_proper_divisor_55_l205_205862


namespace problem1_div_expr_problem2_div_expr_l205_205179

-- Problem 1
theorem problem1_div_expr : (1 / 30) / ((2 / 3) - (1 / 10) + (1 / 6) - (2 / 5)) = 1 / 10 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

-- Problem 2
theorem problem2_div_expr : (-1 / 20) / (-(1 / 4) - (2 / 5) + (9 / 10) - (3 / 2)) = 1 / 25 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

end problem1_div_expr_problem2_div_expr_l205_205179


namespace sum_primes_between_20_and_40_l205_205201

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l205_205201


namespace domain_of_inverse_l205_205491

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x - 1) + 1

theorem domain_of_inverse :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y = f x) → (y ∈ Set.Icc (3/2) 3) :=
by
  sorry

end domain_of_inverse_l205_205491


namespace shaded_area_l205_205853

-- Definitions and conditions from the problem
def Square1Side := 4 -- in inches
def Square2Side := 12 -- in inches
def Triangle_DGF_similar_to_Triangle_AHF : Prop := (4 / 12) = (3 / 16)

theorem shaded_area
  (h1 : Square1Side = 4)
  (h2 : Square2Side = 12)
  (h3 : Triangle_DGF_similar_to_Triangle_AHF) :
  ∃ shaded_area : ℕ, shaded_area = 10 :=
by
  -- Calculation steps here
  sorry

end shaded_area_l205_205853


namespace parabola_equation_l205_205549

theorem parabola_equation (a : ℝ) :
  (∀ x, (x + 1) * (x - 3) = 0 ↔ x = -1 ∨ x = 3) →
  (∀ y, y = a * (0 + 1) * (0 - 3) → y = 3) →
  a = -1 → 
  (∀ x, y = a * (x + 1) * (x - 3) → y = -x^2 + 2 * x + 3) :=
by
  intros h₁ h₂ ha
  sorry

end parabola_equation_l205_205549


namespace find_f_2_l205_205715

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem find_f_2 (a b : ℝ) (hf_neg2 : f a b (-2) = 7) : f a b 2 = -13 :=
by
  sorry

end find_f_2_l205_205715


namespace total_questions_l205_205535

theorem total_questions (qmc : ℕ) (qtotal : ℕ) (h1 : 10 = qmc) (h2 : qmc = (20 / 100) * qtotal) : qtotal = 50 :=
sorry

end total_questions_l205_205535


namespace proof_problem_l205_205674

-- Conditions
def a : ℤ := 1
def b : ℤ := 0
def c : ℤ := -1 + 3

-- Proof Statement
theorem proof_problem : (2 * a + 3 * c) * b = 0 := by
  sorry

end proof_problem_l205_205674


namespace rectangle_perimeter_l205_205854

theorem rectangle_perimeter (t s : ℝ) (h : t ≥ s) : 2 * (t - s) + 2 * s = 2 * t := 
by 
  sorry

end rectangle_perimeter_l205_205854


namespace least_value_a2000_l205_205530

theorem least_value_a2000 (a : ℕ → ℕ)
  (h1 : ∀ m n, (m ∣ n) → (m < n) → (a m ∣ a n))
  (h2 : ∀ m n, (m ∣ n) → (m < n) → (a m < a n)) :
  a 2000 >= 128 :=
sorry

end least_value_a2000_l205_205530


namespace fewest_people_to_join_CBL_l205_205718

theorem fewest_people_to_join_CBL (initial_people teamsize : ℕ) (even_teams : ℕ → Prop)
  (initial_people_eq : initial_people = 38)
  (teamsize_eq : teamsize = 9)
  (even_teams_def : ∀ n, even_teams n ↔ n % 2 = 0) :
  ∃(p : ℕ), (initial_people + p) % teamsize = 0 ∧ even_teams ((initial_people + p) / teamsize) ∧ p = 16 := by
  sorry

end fewest_people_to_join_CBL_l205_205718


namespace octagon_area_difference_l205_205962

theorem octagon_area_difference (side_length : ℝ) (h : side_length = 1) : 
  let A := 2 * (1 + Real.sqrt 2)
  let triangle_area := (1 / 2) * (1 / 2) * (1 / 2)
  let gray_area := 4 * triangle_area
  let part_with_lines := A - gray_area
  (gray_area - part_with_lines) = 1 / 4 :=
by
  sorry

end octagon_area_difference_l205_205962


namespace sum_of_arithmetic_sequence_l205_205520

variables (a_n : Nat → Int) (S_n : Nat → Int)
variable (n : Nat)

-- Definitions based on given conditions:
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
∀ n, a_n (n + 1) = a_n n + a_n 1 - a_n 0

def a_1 : Int := -2018

def arithmetic_sequence_sum (S_n : Nat → Int) (a_n : Nat → Int) (n : Nat) : Prop :=
S_n n = n * a_n 0 + (n * (n - 1) / 2 * (a_n 1 - a_n 0))

-- Given condition S_12 / 12 - S_10 / 10 = 2
def condition (S_n : Nat → Int) : Prop :=
S_n 12 / 12 - S_n 10 / 10 = 2

-- Goal: Prove S_2018 = -2018
theorem sum_of_arithmetic_sequence (a_n S_n : Nat → Int)
  (h1 : a_n 1 = -2018)
  (h2 : is_arithmetic_sequence a_n)
  (h3 : ∀ n, arithmetic_sequence_sum S_n a_n n)
  (h4 : condition S_n) :
  S_n 2018 = -2018 :=
sorry

end sum_of_arithmetic_sequence_l205_205520


namespace percent_of_x_is_y_l205_205192

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end percent_of_x_is_y_l205_205192


namespace proof_A_proof_C_l205_205230

theorem proof_A (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b ≤ ( (a + b) / 2) ^ 2 := 
sorry

theorem proof_C (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : 
  ∃ y, y = x * (4 - x^2).sqrt ∧ y ≤ 2 := 
sorry

end proof_A_proof_C_l205_205230


namespace eggs_per_group_l205_205238

-- Conditions
def total_eggs : ℕ := 9
def total_groups : ℕ := 3

-- Theorem statement
theorem eggs_per_group : total_eggs / total_groups = 3 :=
sorry

end eggs_per_group_l205_205238


namespace equation_of_circle_given_diameter_l205_205436

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem equation_of_circle_given_diameter :
  ∀ (A B : ℝ × ℝ), A = (-3,0) → B = (1,0) → 
  (∃ (x y : ℝ), is_on_circle (-1, 0) 2 (x, y)) ↔ (x + 1)^2 + y^2 = 4 :=
by
  sorry

end equation_of_circle_given_diameter_l205_205436


namespace inequality_solution_l205_205822

noncomputable def solution_set : Set ℝ :=
  {x : ℝ | x < -2} ∪
  {x : ℝ | -2 < x ∧ x ≤ -1} ∪
  {x : ℝ | 1 ≤ x}

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x + 2)^2 ≥ 0} = solution_set := by
  sorry

end inequality_solution_l205_205822


namespace g_neg_eleven_eq_neg_two_l205_205640

def f (x : ℝ) : ℝ := 2 * x - 7
def g (y : ℝ) : ℝ := 3 * y^2 + 4 * y - 6

theorem g_neg_eleven_eq_neg_two : g (-11) = -2 := by
  sorry

end g_neg_eleven_eq_neg_two_l205_205640


namespace seating_arrangement_fixed_pairs_l205_205945

theorem seating_arrangement_fixed_pairs 
  (total_chairs : ℕ) 
  (total_people : ℕ) 
  (specific_pair_adjacent : Prop)
  (comb : ℕ) 
  (four_factorial : ℕ) 
  (two_factorial : ℕ) 
  : total_chairs = 6 → total_people = 5 → specific_pair_adjacent → comb = Nat.choose 6 4 → 
    four_factorial = Nat.factorial 4 → two_factorial = Nat.factorial 2 → 
    Nat.choose 6 4 * Nat.factorial 4 * Nat.factorial 2 = 720 
  := by
  intros
  sorry

end seating_arrangement_fixed_pairs_l205_205945


namespace scalene_triangle_geometric_progression_l205_205079

theorem scalene_triangle_geometric_progression :
  ∀ (q : ℝ), q ≠ 0 → 
  (∀ b : ℝ, b > 0 → b + q * b > q^2 * b ∧ q * b + q^2 * b > b ∧ b + q^2 * b > q * b) → 
  ¬((0.5 < q ∧ q < 1.7) ∨ q = 2.0) → false :=
by
  intros q hq_ne_zero hq hq_interval
  sorry

end scalene_triangle_geometric_progression_l205_205079


namespace sequence_general_term_l205_205448

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1) :
  (∀ n, a n = if n = 1 then 2 else 2 * n - 1) :=
by
  sorry

end sequence_general_term_l205_205448


namespace greatest_prime_factor_391_l205_205803

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end greatest_prime_factor_391_l205_205803


namespace income_increase_is_60_percent_l205_205401

noncomputable def income_percentage_increase 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : ℝ :=
  (M - T) / T * 100

theorem income_increase_is_60_percent 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : 
  income_percentage_increase J T M h1 h2 = 60 :=
by
  sorry

end income_increase_is_60_percent_l205_205401


namespace circle_equation_and_lines_l205_205200

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, 2)
noncomputable def B : ℝ × ℝ := (4, 4)
noncomputable def C_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 10

structure Line (κ β: ℝ) where
  passes_through : ℝ × ℝ → Prop
  definition : Prop

def line_passes_through_point (κ β : ℝ) (p : ℝ × ℝ) : Prop := p.2 = κ * p.1 + β

theorem circle_equation_and_lines : 
  (∀ p : ℝ × ℝ, p = O ∨ p = A ∨ p = B → C_eq p.1 p.2) ∧
  ((∀ p : ℝ × ℝ, line_passes_through_point 0 2 p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4)) ∧
   (∀ p : ℝ × ℝ, line_passes_through_point (-7 / 3) (32 / 3) p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4))) :=
by 
  sorry

end circle_equation_and_lines_l205_205200


namespace bus_distance_covered_l205_205819

theorem bus_distance_covered (speedTrain speedCar speedBus distanceBus : ℝ) (h1 : speedTrain / speedCar = 16 / 15)
                            (h2 : speedBus = (3 / 4) * speedTrain) (h3 : 450 / 6 = speedCar) (h4 : distanceBus = 8 * speedBus) :
                            distanceBus = 480 :=
by
  sorry

end bus_distance_covered_l205_205819


namespace ellens_initial_legos_l205_205438

-- Define the initial number of Legos as a proof goal
theorem ellens_initial_legos : ∀ (x y : ℕ), (y = x - 17) → (x = 2080) :=
by
  intros x y h
  sorry

end ellens_initial_legos_l205_205438


namespace four_fours_to_seven_l205_205909

theorem four_fours_to_seven :
  (∃ eq1 eq2 : ℕ, eq1 ≠ eq2 ∧
    (eq1 = 4 + 4 - (4 / 4) ∧
     eq2 = 44 / 4 - 4 ∧ eq1 = 7 ∧ eq2 = 7)) :=
by
  existsi (4 + 4 - (4 / 4))
  existsi (44 / 4 - 4)
  sorry

end four_fours_to_seven_l205_205909


namespace length_BC_l205_205109

theorem length_BC {A B C : ℝ} (r1 r2 : ℝ) (AB : ℝ) (h1 : r1 = 8) (h2 : r2 = 5) (h3 : AB = r1 + r2) :
  C = B + (65 : ℝ) / 3 :=
by
  -- Problem set-up and solving comes here if needed
  sorry

end length_BC_l205_205109


namespace range_of_a_l205_205527

noncomputable def f (x a : ℝ) := 2^(2*x) - a * 2^x + 4

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a ≤ 4 :=
by
  sorry

end range_of_a_l205_205527


namespace triangle_inequality_sum_zero_l205_205058

theorem triangle_inequality_sum_zero (a b c p q r : ℝ) (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) (hpqr : p + q + r = 0) : a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
by 
  sorry

end triangle_inequality_sum_zero_l205_205058


namespace relationship_between_m_and_n_l205_205773

variable {X_1 X_2 k m n : ℝ}

-- Given conditions
def inverse_proportional_points (X_1 X_2 k : ℝ) (m n : ℝ) : Prop :=
  m = k / X_1 ∧ n = k / X_2 ∧ k > 0 ∧ X_1 < X_2

theorem relationship_between_m_and_n (h : inverse_proportional_points X_1 X_2 k m n) : m > n :=
by
  -- Insert proof here, skipping with sorry
  sorry

end relationship_between_m_and_n_l205_205773


namespace cost_of_27_pounds_l205_205087

def rate_per_pound : ℝ := 1
def weight_pounds : ℝ := 27

theorem cost_of_27_pounds :
  weight_pounds * rate_per_pound = 27 := 
by 
  -- sorry placeholder indicates that the proof is not provided
  sorry

end cost_of_27_pounds_l205_205087


namespace determine_constant_l205_205501

theorem determine_constant (c : ℝ) :
  (∃ d : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) ↔ c = 16 :=
by
  sorry

end determine_constant_l205_205501


namespace remainder_of_n_mod_5_l205_205116

theorem remainder_of_n_mod_5
  (n : Nat)
  (h1 : n^2 ≡ 4 [MOD 5])
  (h2 : n^3 ≡ 2 [MOD 5]) :
  n ≡ 3 [MOD 5] :=
sorry

end remainder_of_n_mod_5_l205_205116


namespace integrate_differential_eq_l205_205381

theorem integrate_differential_eq {x y C : ℝ} {y' : ℝ → ℝ → ℝ} (h : ∀ x y, (4 * y - 3 * x - 5) * y' x y + 7 * x - 3 * y + 2 = 0) : 
    ∃ C : ℝ, ∀ x y : ℝ, 2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C :=
by
  sorry

end integrate_differential_eq_l205_205381


namespace certain_number_value_l205_205695

theorem certain_number_value (x : ℕ) (p n : ℕ) (hp : Nat.Prime p) (hx : x = 44) (h : x / (n * p) = 2) : n = 2 := 
by
  sorry

end certain_number_value_l205_205695


namespace problem_l205_205994

def setA : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}
def setB : Set ℝ := {x : ℝ | x ≤ 3}

theorem problem : setA ∩ setB = setA := sorry

end problem_l205_205994


namespace volume_of_tetrahedron_PQRS_l205_205751

-- Definitions of the given conditions for the tetrahedron
def PQ := 6
def PR := 4
def PS := 5
def QR := 5
def QS := 6
def RS := 15 / 2  -- RS is (15 / 2), i.e., 7.5
def area_PQR := 12

noncomputable def volume_tetrahedron (PQ PR PS QR QS RS area_PQR : ℝ) : ℝ := 1 / 3 * area_PQR * 4

theorem volume_of_tetrahedron_PQRS :
  volume_tetrahedron PQ PR PS QR QS RS area_PQR = 16 :=
by sorry

end volume_of_tetrahedron_PQRS_l205_205751


namespace combined_percentage_of_students_preferring_tennis_is_39_l205_205157

def total_students_north : ℕ := 1800
def percentage_tennis_north : ℚ := 25 / 100
def total_students_south : ℕ := 3000
def percentage_tennis_south : ℚ := 50 / 100
def total_students_valley : ℕ := 800
def percentage_tennis_valley : ℚ := 30 / 100

def students_prefer_tennis_north : ℚ := total_students_north * percentage_tennis_north
def students_prefer_tennis_south : ℚ := total_students_south * percentage_tennis_south
def students_prefer_tennis_valley : ℚ := total_students_valley * percentage_tennis_valley

def total_students : ℕ := total_students_north + total_students_south + total_students_valley
def total_students_prefer_tennis : ℚ := students_prefer_tennis_north + students_prefer_tennis_south + students_prefer_tennis_valley

def percentage_students_prefer_tennis : ℚ := (total_students_prefer_tennis / total_students) * 100

theorem combined_percentage_of_students_preferring_tennis_is_39 :
  percentage_students_prefer_tennis = 39 := by
  sorry

end combined_percentage_of_students_preferring_tennis_is_39_l205_205157


namespace find_p_l205_205021

-- Lean 4 definitions corresponding to the conditions
variables {p a b x0 y0 : ℝ} (hp : p > 0) (ha : a > 0) (hb : b > 0) (hx0 : x0 ≠ 0)
variables (hA : (y0^2 = 2 * p * x0) ∧ ((x0 / a)^2 - (y0 / b)^2 = 1))
variables (h_dist : x0 + x0 = p^2)
variables (h_ecc : (5^.half) = sqrt 5)

-- The proof problem
theorem find_p :
  p = 1 :=
by
  sorry

end find_p_l205_205021


namespace time_taken_by_x_alone_l205_205772

theorem time_taken_by_x_alone 
  (W : ℝ)
  (Rx Ry Rz : ℝ)
  (h1 : Ry = W / 24)
  (h2 : Ry + Rz = W / 6)
  (h3 : Rx + Rz = W / 4) :
  (W / Rx) = 16 :=
by
  sorry

end time_taken_by_x_alone_l205_205772


namespace transformations_result_l205_205378

theorem transformations_result :
  ∃ (r g : ℕ), r + g = 15 ∧ 
  21 + r - 5 * g = 0 ∧ 
  30 - 2 * r + 2 * g = 24 :=
by
  sorry

end transformations_result_l205_205378


namespace mean_greater_than_median_by_two_l205_205333

theorem mean_greater_than_median_by_two (x : ℕ) (h : x > 0) :
  ((x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5 - (x + 4)) = 2 :=
sorry

end mean_greater_than_median_by_two_l205_205333


namespace inequality_solution_set_l205_205850

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l205_205850


namespace initially_calculated_average_height_l205_205273

theorem initially_calculated_average_height 
    (students : ℕ) (incorrect_height : ℕ) (correct_height : ℕ) (actual_avg_height : ℝ) 
    (A : ℝ) 
    (h_students : students = 30) 
    (h_incorrect_height : incorrect_height = 151) 
    (h_correct_height : correct_height = 136) 
    (h_actual_avg_height : actual_avg_height = 174.5)
    (h_A_definition : (students : ℝ) * A + (incorrect_height - correct_height) = (students : ℝ) * actual_avg_height) : 
    A = 174 := 
by sorry

end initially_calculated_average_height_l205_205273


namespace product_of_fractions_l205_205480

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 :=
by
  sorry

end product_of_fractions_l205_205480


namespace remove_denominators_l205_205081

theorem remove_denominators (x : ℝ) : (1 / 2 - (x - 1) / 3 = 1) → (3 - 2 * (x - 1) = 6) :=
by
  intro h
  sorry

end remove_denominators_l205_205081


namespace find_a2_l205_205403

def S (n : Nat) (a1 d : Int) : Int :=
  n * a1 + (n * (n - 1) * d) / 2

theorem find_a2 (a1 : Int) (d : Int) :
  a1 = -2010 ∧
  (S 2010 a1 d) / 2010 - (S 2008 a1 d) / 2008 = 2 →
  a1 + d = -2008 :=
by
  sorry

end find_a2_l205_205403


namespace Mason_tables_needed_l205_205757

theorem Mason_tables_needed
  (w_silverware_piece : ℕ := 4) 
  (n_silverware_piece_per_setting : ℕ := 3) 
  (w_plate : ℕ := 12) 
  (n_plates_per_setting : ℕ := 2) 
  (n_settings_per_table : ℕ := 8) 
  (n_backup_settings : ℕ := 20) 
  (total_weight : ℕ := 5040) : 
  ∃ (n_tables : ℕ), n_tables = 15 :=
by
  sorry

end Mason_tables_needed_l205_205757


namespace wendy_first_day_miles_l205_205634

-- Define the variables for the problem
def total_miles : ℕ := 493
def miles_day2 : ℕ := 223
def miles_day3 : ℕ := 145

-- Define the proof problem
theorem wendy_first_day_miles :
  total_miles = miles_day2 + miles_day3 + 125 :=
sorry

end wendy_first_day_miles_l205_205634


namespace chairs_left_proof_l205_205952

def red_chairs : ℕ := 4
def yellow_chairs : ℕ := 2 * red_chairs
def blue_chairs : ℕ := 3 * yellow_chairs
def green_chairs : ℕ := blue_chairs / 2
def orange_chairs : ℕ := green_chairs + 2
def total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
def borrowed_chairs : ℕ := 5 + 3
def chairs_left : ℕ := total_chairs - borrowed_chairs

theorem chairs_left_proof : chairs_left = 54 := by
  -- This is where the proof would go
  sorry

end chairs_left_proof_l205_205952


namespace solve_eq1_solve_eq2_solve_eq3_l205_205738

def equation1 (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0
def solution1 (x : ℝ) : Prop := x = 5 ∨ x = 1

theorem solve_eq1 : ∀ x : ℝ, equation1 x ↔ solution1 x := sorry

def equation2 (x : ℝ) : Prop := 3 * x * (2 * x - 1) = 4 * x - 2
def solution2 (x : ℝ) : Prop := x = 1/2 ∨ x = 2/3

theorem solve_eq2 : ∀ x : ℝ, equation2 x ↔ solution2 x := sorry

def equation3 (x : ℝ) : Prop := x^2 - 2 * Real.sqrt 2 * x - 2 = 0
def solution3 (x : ℝ) : Prop := x = Real.sqrt 2 + 2 ∨ x = Real.sqrt 2 - 2

theorem solve_eq3 : ∀ x : ℝ, equation3 x ↔ solution3 x := sorry

end solve_eq1_solve_eq2_solve_eq3_l205_205738


namespace value_of_p_l205_205713

theorem value_of_p (m n p : ℝ) (h1 : m = 6 * n + 5) (h2 : m + 2 = 6 * (n + p) + 5) : p = 1 / 3 :=
by
  sorry

end value_of_p_l205_205713


namespace find_first_train_length_l205_205478

theorem find_first_train_length
  (length_second_train : ℝ)
  (initial_distance : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_minutes : ℝ) :
  length_second_train = 200 →
  initial_distance = 100 →
  speed_first_train_kmph = 54 →
  speed_second_train_kmph = 72 →
  time_minutes = 2.856914303998537 →
  ∃ (L : ℝ), L = 5699.52 :=
by
  sorry

end find_first_train_length_l205_205478


namespace sean_div_julie_eq_two_l205_205071

def sum_n (n : ℕ) := n * (n + 1) / 2

def sean_sum := 2 * sum_n 500

def julie_sum := sum_n 500

theorem sean_div_julie_eq_two : sean_sum / julie_sum = 2 := 
by sorry

end sean_div_julie_eq_two_l205_205071


namespace registration_methods_for_5_students_l205_205654

def number_of_registration_methods (students groups : ℕ) : ℕ :=
  groups ^ students

theorem registration_methods_for_5_students : number_of_registration_methods 5 2 = 32 := by
  sorry

end registration_methods_for_5_students_l205_205654


namespace compute_result_l205_205466

theorem compute_result : (300000 * 200000) / 100000 = 600000 := by
  sorry

end compute_result_l205_205466


namespace text_messages_in_march_l205_205593

/-
Jared sent text messages each month according to the formula:
  T_n = n^3 - n^2 + n
We need to prove that the number of text messages Jared will send in March
(which is the 5th month) is given by T_5 = 105.
-/

def T (n : ℕ) : ℕ := n^3 - n^2 + n

theorem text_messages_in_march : T 5 = 105 :=
by
  -- proof goes here
  sorry

end text_messages_in_march_l205_205593


namespace train_length_is_correct_l205_205904

-- Definitions of speeds and time
def speedTrain_kmph := 100
def speedMotorbike_kmph := 64
def overtakingTime_s := 20

-- Calculate speeds in m/s
def speedTrain_mps := speedTrain_kmph * 1000 / 3600
def speedMotorbike_mps := speedMotorbike_kmph * 1000 / 3600

-- Calculate relative speed
def relativeSpeed_mps := speedTrain_mps - speedMotorbike_mps

-- Calculate the length of the train
def length_of_train := relativeSpeed_mps * overtakingTime_s

-- Theorem: Verifying the length of the train is 200 meters
theorem train_length_is_correct : length_of_train = 200 := by
  -- Sorry placeholder for proof
  sorry

end train_length_is_correct_l205_205904


namespace message_channels_encryption_l205_205180

theorem message_channels_encryption :
  ∃ (assign_key : Fin 105 → Fin 105 → Fin 100),
  ∀ (u v w x : Fin 105), 
  u ≠ v → u ≠ w → u ≠ x → v ≠ w → v ≠ x → w ≠ x →
  (assign_key u v = assign_key u w ∧ assign_key u v = assign_key u x ∧ 
   assign_key u v = assign_key v w ∧ assign_key u v = assign_key v x ∧ 
   assign_key u v = assign_key w x) → False :=
by
  sorry

end message_channels_encryption_l205_205180


namespace sum_of_possible_values_l205_205861

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 5) = 20) : x = -2 ∨ x = 7 :=
sorry

end sum_of_possible_values_l205_205861


namespace num_valid_combinations_l205_205059

-- Definitions based on the conditions
def num_herbs := 4
def num_gems := 6
def num_incompatible_gems := 3
def num_incompatible_herbs := 2

-- Statement to be proved
theorem num_valid_combinations :
  (num_herbs * num_gems) - (num_incompatible_gems * num_incompatible_herbs) = 18 :=
by
  sorry

end num_valid_combinations_l205_205059


namespace alex_money_left_l205_205047

noncomputable def alex_main_income : ℝ := 900
noncomputable def alex_side_income : ℝ := 300
noncomputable def main_job_tax_rate : ℝ := 0.15
noncomputable def side_job_tax_rate : ℝ := 0.20
noncomputable def water_bill : ℝ := 75
noncomputable def main_job_tithe_rate : ℝ := 0.10
noncomputable def side_job_tithe_rate : ℝ := 0.15
noncomputable def grocery_expense : ℝ := 150
noncomputable def transportation_expense : ℝ := 50

theorem alex_money_left :
  let main_income_after_tax := alex_main_income * (1 - main_job_tax_rate)
  let side_income_after_tax := alex_side_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_income_after_tax + side_income_after_tax
  let main_tithe := alex_main_income * main_job_tithe_rate
  let side_tithe := alex_side_income * side_job_tithe_rate
  let total_tithe := main_tithe + side_tithe
  let total_deductions := water_bill + grocery_expense + transportation_expense + total_tithe
  let money_left := total_income_after_tax - total_deductions
  money_left = 595 :=
by
  -- Proof goes here
  sorry

end alex_money_left_l205_205047


namespace packed_oranges_l205_205943

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end packed_oranges_l205_205943


namespace ratio_of_luxury_to_suv_l205_205389

variable (E L S : Nat)

-- Conditions
def condition1 := E * 2 = L * 3
def condition2 := E * 1 = S * 4

-- The statement to prove
theorem ratio_of_luxury_to_suv 
  (h1 : condition1 E L)
  (h2 : condition2 E S) :
  L * 3 = S * 8 :=
by sorry

end ratio_of_luxury_to_suv_l205_205389


namespace sum_of_coefficients_l205_205213

theorem sum_of_coefficients (x y z : ℤ) (h : x = 1 ∧ y = 1 ∧ z = 1) :
    (x - 2 * y + 3 * z) ^ 12 = 4096 :=
by
  sorry

end sum_of_coefficients_l205_205213


namespace fraction_calculation_l205_205292

theorem fraction_calculation : (3/10 : ℚ) + (5/100 : ℚ) - (2/1000 : ℚ) = 348/1000 := 
by 
  sorry

end fraction_calculation_l205_205292


namespace find_y_l205_205097

-- Define the known values and the proportion relation
variable (x y : ℝ)
variable (h1 : 0.75 / x = y / 7)
variable (h2 : x = 1.05)

theorem find_y : y = 5 :=
by
sorry

end find_y_l205_205097


namespace difference_between_shares_l205_205209

def investment_months (amount : ℕ) (months : ℕ) : ℕ :=
  amount * months

def ratio (investment_months : ℕ) (total_investment_months : ℕ) : ℚ :=
  investment_months / total_investment_months

def profit_share (ratio : ℚ) (total_profit : ℝ) : ℝ :=
  ratio * total_profit

theorem difference_between_shares :
  let suresh_investment := 18000
  let rohan_investment := 12000
  let sudhir_investment := 9000
  let suresh_months := 12
  let rohan_months := 9
  let sudhir_months := 8
  let total_profit := 3795
  let suresh_investment_months := investment_months suresh_investment suresh_months
  let rohan_investment_months := investment_months rohan_investment rohan_months
  let sudhir_investment_months := investment_months sudhir_investment sudhir_months
  let total_investment_months := suresh_investment_months + rohan_investment_months + sudhir_investment_months
  let suresh_ratio := ratio suresh_investment_months total_investment_months
  let rohan_ratio := ratio rohan_investment_months total_investment_months
  let sudhir_ratio := ratio sudhir_investment_months total_investment_months
  let rohan_share := profit_share rohan_ratio total_profit
  let sudhir_share := profit_share sudhir_ratio total_profit
  rohan_share - sudhir_share = 345 :=
by
  sorry

end difference_between_shares_l205_205209


namespace ball_color_arrangement_l205_205168

-- Definitions for the conditions
variable (balls_in_red_box balls_in_white_box balls_in_yellow_box : Nat)
variable (red_balls white_balls yellow_balls : Nat)

-- Conditions as assumptions
axiom more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls
axiom different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls
axiom fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box

-- The main theorem to prove
theorem ball_color_arrangement
  (more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls)
  (different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls)
  (fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box) :
  (balls_in_red_box, balls_in_white_box, balls_in_yellow_box) = (yellow_balls, red_balls, white_balls) :=
sorry

end ball_color_arrangement_l205_205168


namespace total_amount_l205_205534

-- Declare the variables
variables (A B C : ℕ)

-- Introduce the conditions as hypotheses
theorem total_amount (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B = 290) : 
  A + B + C = 980 := 
by {
  sorry
}

end total_amount_l205_205534


namespace clark_paid_correct_amount_l205_205968

-- Definitions based on the conditions
def cost_per_part : ℕ := 80
def number_of_parts : ℕ := 7
def total_discount : ℕ := 121

-- Given conditions
def total_cost_without_discount : ℕ := cost_per_part * number_of_parts
def expected_total_cost_after_discount : ℕ := 439

-- Theorem to prove the amount Clark paid after the discount is correct
theorem clark_paid_correct_amount : total_cost_without_discount - total_discount = expected_total_cost_after_discount := by
  sorry

end clark_paid_correct_amount_l205_205968


namespace forgotten_angle_l205_205379

theorem forgotten_angle {n : ℕ} (h₁ : 2070 = (n - 2) * 180 - angle) : angle = 90 :=
by
  sorry

end forgotten_angle_l205_205379


namespace find_square_value_l205_205856

variable (a b : ℝ)
variable (square : ℝ)

-- Conditions: Given the equation square * 3 * a = -3 * a^2 * b
axiom condition : square * 3 * a = -3 * a^2 * b

-- Theorem: Prove that square = -a * b
theorem find_square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) : 
    square = -a * b :=
by
  exact sorry

end find_square_value_l205_205856


namespace geometric_series_evaluation_l205_205341

theorem geometric_series_evaluation (c d : ℝ) (h : (∑' n : ℕ, c / d^(n + 1)) = 3) :
  (∑' n : ℕ, c / (c + 2 * d)^(n + 1)) = (3 * d - 3) / (5 * d - 4) :=
sorry

end geometric_series_evaluation_l205_205341


namespace parabola_constant_term_l205_205077

theorem parabola_constant_term :
  ∃ b c : ℝ, (∀ x : ℝ, (x = 2 → 3 = x^2 + b * x + c) ∧ (x = 4 → 3 = x^2 + b * x + c)) → c = 11 :=
by
  sorry

end parabola_constant_term_l205_205077


namespace math_problem_l205_205075

theorem math_problem (x : ℝ) :
  (x^3 - 8*x^2 + 16*x > 64) ∧ (x^2 - 4*x + 5 > 0) → x > 4 :=
by
  sorry

end math_problem_l205_205075


namespace proof_problem_l205_205384

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1) → (0 ≤ x2) → (x1 ≠ x2) → (x1 - x2) * (f x1 - f x2) > 0

theorem proof_problem (f : ℝ → ℝ) (hf_even : even_function f) (hf_condition : condition f) :
  f 1 < f (-2) ∧ f (-2) < f 3 := sorry

end proof_problem_l205_205384


namespace symmetric_origin_l205_205266

def symmetric_point (p : (Int × Int)) : (Int × Int) :=
  (-p.1, -p.2)

theorem symmetric_origin : symmetric_point (-2, 5) = (2, -5) :=
by
  -- proof goes here
  -- we use sorry to indicate the place where the solution would go
  sorry

end symmetric_origin_l205_205266


namespace smallest_possible_sum_l205_205017

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_ne : x ≠ y) (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 50 :=
sorry

end smallest_possible_sum_l205_205017


namespace min_value_inequality_l205_205931

theorem min_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 := 
by
  sorry

end min_value_inequality_l205_205931


namespace square_free_even_less_than_200_count_l205_205164

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem square_free_even_less_than_200_count : ∃ (count : ℕ), count = 38 ∧ (∀ n : ℕ, n < 200 ∧ is_multiple_of_2 n ∧ is_square_free n → count = 38) :=
by
  sorry

end square_free_even_less_than_200_count_l205_205164


namespace player_A_wins_4_points_game_game_ends_after_5_points_l205_205808

def prob_A_winning_when_serving : ℚ := 2 / 3
def prob_A_winning_when_B_serving : ℚ := 1 / 4
def prob_A_winning_in_4_points : ℚ := 1 / 12
def prob_game_ending_after_5_points : ℚ := 19 / 216

theorem player_A_wins_4_points_game :
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) = prob_A_winning_in_4_points := 
  sorry

theorem game_ends_after_5_points : 
  ((1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (prob_A_winning_when_serving)) + 
  ((prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (1 - prob_A_winning_when_serving)) = 
  prob_game_ending_after_5_points :=
  sorry

end player_A_wins_4_points_game_game_ends_after_5_points_l205_205808


namespace intersection_A_B_l205_205804

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | ∃ y : ℝ, y = x^2 + 2}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ ∃ y : ℝ, y = x^2 + 2} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end intersection_A_B_l205_205804


namespace area_of_triangle_tangent_at_pi_div_two_l205_205385

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem area_of_triangle_tangent_at_pi_div_two :
  let x := Real.pi / 2
  let slope := 1 + Real.cos x
  let point := (x, f x)
  let intercept_y := f x - slope * x
  let x_intercept := -intercept_y / slope
  let y_intercept := intercept_y
  (1 / 2) * x_intercept * y_intercept = 1 / 2 := 
by
  sorry

end area_of_triangle_tangent_at_pi_div_two_l205_205385


namespace part1_part2_l205_205631

section

variable (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_seq : ∀ n, a_seq (n + 1) = (5 * a_seq n - 8) / (a_seq n - 1))
variable (h_initial : a_seq 1 = a)

-- Part 1:
theorem part1 (h_a : a = 3) : 
  ∃ r : ℝ, ∀ n, (a_seq n - 2) / (a_seq n - 4) = r ^ n ∧ a_seq n = (4 * 3 ^ (n - 1) + 2) / (3 ^ (n - 1) + 1) := 
sorry

-- Part 2:
theorem part2 (h_pos : ∀ n, a_seq n > 3) : 3 < a := 
sorry

end

end part1_part2_l205_205631


namespace frequency_of_middle_group_l205_205458

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end frequency_of_middle_group_l205_205458


namespace geometric_sequence_ratio_l205_205483

noncomputable def geometric_sequence_pos (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence_pos a q) (h_q : q^2 = 4) :
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
sorry

end geometric_sequence_ratio_l205_205483


namespace second_flower_shop_groups_l205_205152

theorem second_flower_shop_groups (n : ℕ) (h1 : n ≠ 0) (h2 : n ≠ 9) (h3 : Nat.lcm 9 n = 171) : n = 19 := 
by
  sorry

end second_flower_shop_groups_l205_205152


namespace no_nat_solutions_l205_205739

theorem no_nat_solutions (x y z : ℕ) : x^2 + y^2 + z^2 ≠ 2 * x * y * z :=
sorry

end no_nat_solutions_l205_205739


namespace ellipse_major_axis_length_l205_205756

theorem ellipse_major_axis_length : 
  ∀ (x y : ℝ), x^2 + 2 * y^2 = 2 → 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_major_axis_length_l205_205756


namespace max_value_S_n_l205_205350

theorem max_value_S_n 
  (a : ℕ → ℕ)
  (a1 : a 1 = 2)
  (S : ℕ → ℕ)
  (h : ∀ n, 6 * S n = 3 * a (n + 1) + 4 ^ n - 1) :
  ∃ n, S n = 10 := 
sorry

end max_value_S_n_l205_205350


namespace least_common_duration_l205_205320

theorem least_common_duration 
    (P Q R : ℝ) 
    (x : ℝ)
    (T : ℝ)
    (h1 : P / Q = 7 / 5)
    (h2 : Q / R = 5 / 3)
    (h3 : 8 * P / (6 * Q) = 7 / 10)
    (h4 : (6 * 10) * R / (30 * T) = 1)
    : T = 6 :=
by
  sorry

end least_common_duration_l205_205320


namespace rectangle_breadth_l205_205615

theorem rectangle_breadth (length radius side breadth: ℝ)
  (h1: length = (2/5) * radius)
  (h2: radius = side)
  (h3: side ^ 2 = 1600)
  (h4: length * breadth = 160) :
  breadth = 10 := 
by
  sorry

end rectangle_breadth_l205_205615


namespace ratio_nonupgraded_to_upgraded_l205_205609

-- Define the initial conditions and properties
variable (S : ℝ) (N : ℝ)
variable (h1 : ∀ N, N = S / 32)
variable (h2 : ∀ S, 0.25 * S = 0.25 * S)
variable (h3 : S > 0)

-- Define the theorem to show the required ratio
theorem ratio_nonupgraded_to_upgraded (h3 : 24 * N = 0.75 * S) : (N / (0.25 * S) = 1 / 8) :=
by
  sorry

end ratio_nonupgraded_to_upgraded_l205_205609


namespace fuel_oil_used_l205_205656

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end fuel_oil_used_l205_205656


namespace exist_positive_abc_with_nonzero_integer_roots_l205_205641

theorem exist_positive_abc_with_nonzero_integer_roots :
  ∃ (a b c : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (∀ x y : ℤ, (a * x^2 + b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 + b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) :=
sorry

end exist_positive_abc_with_nonzero_integer_roots_l205_205641


namespace example_problem_l205_205504

variables (a b : ℕ)

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem example_problem (hcf_ab : HCF 385 180 = 30) (a_def: a = 385) (b_def: b = 180) :
  LCM 385 180 = 2310 := 
by
  sorry

end example_problem_l205_205504


namespace Ray_wrote_35_l205_205161

theorem Ray_wrote_35 :
  ∃ (x y : ℕ), (10 * x + y = 35) ∧ (10 * x + y = 4 * (x + y) + 3) ∧ (10 * x + y + 18 = 10 * y + x) :=
by
  sorry

end Ray_wrote_35_l205_205161


namespace inequality_subtraction_l205_205022

theorem inequality_subtraction {a b c : ℝ} (h : a > b) : a - c > b - c := 
sorry

end inequality_subtraction_l205_205022


namespace find_n_l205_205412

theorem find_n (n : ℕ) (h1 : n > 13) (h2 : (12 : ℚ) / (n - 1 : ℚ) = 1 / 3) : n = 37 := by
  sorry

end find_n_l205_205412


namespace lab_techs_share_l205_205318

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end lab_techs_share_l205_205318


namespace num_of_triangles_with_perimeter_10_l205_205852

theorem num_of_triangles_with_perimeter_10 :
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → 
      a + b + c = 10 ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a) ∧ 
    triangles.card = 4 := sorry

end num_of_triangles_with_perimeter_10_l205_205852


namespace orange_sacks_after_95_days_l205_205386

-- Define the conditions as functions or constants
def harvest_per_day : ℕ := 150
def discard_per_day : ℕ := 135
def days_of_harvest : ℕ := 95

-- State the problem formally
theorem orange_sacks_after_95_days :
  (harvest_per_day - discard_per_day) * days_of_harvest = 1425 := 
by 
  sorry

end orange_sacks_after_95_days_l205_205386


namespace find_notebooks_l205_205038

theorem find_notebooks (S N : ℕ) (h1 : N = 4 * S + 3) (h2 : N + 6 = 5 * S) : N = 39 := 
by
  sorry 

end find_notebooks_l205_205038


namespace rotary_club_eggs_needed_l205_205004

theorem rotary_club_eggs_needed 
  (small_children_tickets : ℕ := 53)
  (older_children_tickets : ℕ := 35)
  (adult_tickets : ℕ := 75)
  (senior_tickets : ℕ := 37)
  (waste_percentage : ℝ := 0.03)
  (extra_omelets : ℕ := 25)
  (eggs_per_extra_omelet : ℝ := 2.5) :
  53 * 1 + 35 * 2 + 75 * 3 + 37 * 4 + 
  Nat.ceil (waste_percentage * (53 * 1 + 35 * 2 + 75 * 3 + 37 * 4)) + 
  Nat.ceil (extra_omelets * eggs_per_extra_omelet) = 574 := 
by 
  sorry

end rotary_club_eggs_needed_l205_205004


namespace parabola_translation_l205_205874

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

-- Define the transformation: translate one unit to the right
def translate_right (x : ℝ) : ℝ := initial_parabola (x - 1)

-- Define the transformation: move up three units
def move_up (y : ℝ) : ℝ := y + 3

-- Define the resulting equation after the transformations
def resulting_parabola (x : ℝ) : ℝ := move_up (translate_right x)

-- Define the target equation
def target_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Formalize the proof problem
theorem parabola_translation :
  ∀ x : ℝ, resulting_parabola x = target_parabola x :=
by
  -- Proof steps go here
  sorry

end parabola_translation_l205_205874


namespace solve_for_x_l205_205006

-- Define the operation
def triangle (a b : ℝ) : ℝ := 2 * a - b

-- Define the necessary conditions and the goal
theorem solve_for_x :
  (∀ (a b : ℝ), triangle a b = 2 * a - b) →
  (∃ x : ℝ, triangle x (triangle 1 3) = 2) →
  ∃ x : ℝ, x = 1 / 2 :=
by 
  intros h_main h_eqn
  -- We can skip the proof part as requested.
  sorry

end solve_for_x_l205_205006


namespace evaluate_expression_l205_205828

theorem evaluate_expression : (-1 : ℤ)^(3^3) + (1 : ℤ)^(3^3) = 0 := 
by
  sorry

end evaluate_expression_l205_205828


namespace negation_of_universal_l205_205544

theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x ≥ 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0^2 + x_0 < 0) :=
by
  sorry

end negation_of_universal_l205_205544


namespace elmer_more_than_penelope_l205_205410

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end elmer_more_than_penelope_l205_205410


namespace Jackson_to_Williams_Ratio_l205_205830

-- Define the amounts of money Jackson and Williams have, given the conditions.
def JacksonMoney : ℤ := 125
def TotalMoney : ℤ := 150
-- Define Williams' money based on the given conditions.
def WilliamsMoney : ℤ := TotalMoney - JacksonMoney

-- State the theorem that the ratio of Jackson's money to Williams' money is 5:1
theorem Jackson_to_Williams_Ratio : JacksonMoney / WilliamsMoney = 5 := 
by
  -- Proof steps are omitted as per the instruction.
  sorry

end Jackson_to_Williams_Ratio_l205_205830


namespace geometric_sequence_a7_l205_205531

-- Define the geometric sequence
def geometic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Conditions
def a1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2a4 (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 16

-- The statement to prove
theorem geometric_sequence_a7 (a : ℕ → ℝ) (h1 : a1 a) (h2 : a2a4 a) (gs : geometic_sequence a) :
  a 7 = 64 :=
by
  sorry

end geometric_sequence_a7_l205_205531


namespace find_a_value_l205_205487

-- Definitions of conditions
def eq_has_positive_root (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (x / (x - 5) = 3 - (a / (x - 5)))

-- Statement of the theorem
theorem find_a_value (a : ℝ) (h : eq_has_positive_root a) : a = -5 := 
  sorry

end find_a_value_l205_205487


namespace pure_imaginary_solution_l205_205897

theorem pure_imaginary_solution (a : ℝ) (ha : a + 5 * Complex.I / (1 - 2 * Complex.I) = a + (1 : ℂ) * Complex.I) :
  a = 2 :=
by
  sorry

end pure_imaginary_solution_l205_205897


namespace no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l205_205913

noncomputable def is_square_of_prime (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ m = p * p

theorem no_positive_integer_n_ge_2_1001_n_is_square_of_prime :
  ∀ n : ℕ, n ≥ 2 → ¬ is_square_of_prime (n^3 + 1) :=
by
  intro n hn
  sorry

end no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l205_205913


namespace time_to_cross_pole_correct_l205_205568

-- Definitions of the conditions
def trainSpeed_kmh : ℝ := 120 -- km/hr
def trainLength_m : ℝ := 300 -- meters

-- Assumed conversions
def kmToMeters : ℝ := 1000 -- meters in a km
def hoursToSeconds : ℝ := 3600 -- seconds in an hour

-- Conversion of speed from km/hr to m/s
noncomputable def trainSpeed_ms := (trainSpeed_kmh * kmToMeters) / hoursToSeconds

-- Time to cross the pole
noncomputable def timeToCrossPole := trainLength_m / trainSpeed_ms

-- The theorem stating the proof problem
theorem time_to_cross_pole_correct : timeToCrossPole = 9 := by
  sorry

end time_to_cross_pole_correct_l205_205568


namespace prove_total_bill_is_correct_l205_205569

noncomputable def totalCostAfterDiscounts : ℝ :=
  let adultsMealsCost := 8 * 12
  let teenagersMealsCost := 4 * 10
  let childrenMealsCost := 3 * 7
  let adultsSodasCost := 8 * 3.5
  let teenagersSodasCost := 4 * 3.5
  let childrenSodasCost := 3 * 1.8
  let appetizersCost := 4 * 8
  let dessertsCost := 5 * 5

  let subtotal := adultsMealsCost + teenagersMealsCost + childrenMealsCost +
                  adultsSodasCost + teenagersSodasCost + childrenSodasCost +
                  appetizersCost + dessertsCost

  let discountAdultsMeals := 0.10 * adultsMealsCost
  let discountDesserts := 5
  let discountChildrenMealsAndSodas := 0.15 * (childrenMealsCost + childrenSodasCost)

  let adjustedSubtotal := subtotal - discountAdultsMeals - discountDesserts - discountChildrenMealsAndSodas

  let additionalDiscount := if subtotal > 200 then 0.05 * adjustedSubtotal else 0
  let total := adjustedSubtotal - additionalDiscount

  total

theorem prove_total_bill_is_correct : totalCostAfterDiscounts = 230.70 :=
by sorry

end prove_total_bill_is_correct_l205_205569


namespace find_a_l205_205382

theorem find_a :
  ∃ (a : ℤ), (∀ (x y : ℤ),
    (∃ (m n : ℤ), (x - 8 + m * y) * (x + 3 + n * y) = x^2 + 7 * x * y + a * y^2 - 5 * x - 45 * y - 24) ↔ a = 6) := 
sorry

end find_a_l205_205382


namespace line_equation_k_value_l205_205612

theorem line_equation_k_value (m n k : ℝ) 
    (h1 : m = 2 * n + 5) 
    (h2 : m + 5 = 2 * (n + k) + 5) : 
    k = 2.5 :=
by sorry

end line_equation_k_value_l205_205612


namespace max_area_circle_eq_l205_205051

theorem max_area_circle_eq (m : ℝ) :
  (x y : ℝ) → (x - 1) ^ 2 + (y + m) ^ 2 = -(m - 3) ^ 2 + 1 → 
  (∃ (r : ℝ), (r = (1 : ℝ)) ∧ (m = 3) ∧ ((x - 1) ^ 2 + (y + 3) ^ 2 = 1)) :=
by
  sorry

end max_area_circle_eq_l205_205051


namespace jo_integer_max_l205_205203
noncomputable def jo_integer : Nat :=
  let n := 166
  n

theorem jo_integer_max (n : Nat) (h1 : n < 200) (h2 : ∃ k : Nat, n + 2 = 9 * k) (h3 : ∃ l : Nat, n + 4 = 10 * l) : n ≤ jo_integer := 
by
  unfold jo_integer
  sorry

end jo_integer_max_l205_205203


namespace probability_of_selecting_one_of_each_color_l205_205825

noncomputable def number_of_ways_to_select_4_marbles_from_10 := Nat.choose 10 4
noncomputable def ways_to_select_1_red := Nat.choose 3 1
noncomputable def ways_to_select_1_blue := Nat.choose 3 1
noncomputable def ways_to_select_1_green := Nat.choose 2 1
noncomputable def ways_to_select_1_yellow := Nat.choose 2 1

theorem probability_of_selecting_one_of_each_color :
  (ways_to_select_1_red * ways_to_select_1_blue * ways_to_select_1_green * ways_to_select_1_yellow) / number_of_ways_to_select_4_marbles_from_10 = 6 / 35 :=
by
  sorry

end probability_of_selecting_one_of_each_color_l205_205825


namespace sum_squares_condition_l205_205955

theorem sum_squares_condition
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 75)
  (h2 : ab + bc + ca = 40)
  (h3 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 :=
by sorry

end sum_squares_condition_l205_205955


namespace milk_after_three_operations_l205_205319

-- Define the initial amount of milk and the proportion replaced each step
def initial_milk : ℝ := 100
def proportion_replaced : ℝ := 0.2

-- Define the amount of milk after each replacement operation
noncomputable def milk_after_n_operations (n : ℕ) (milk : ℝ) : ℝ :=
  if n = 0 then milk
  else (1 - proportion_replaced) * milk_after_n_operations (n - 1) milk

-- Define the statement about the amount of milk after three operations
theorem milk_after_three_operations : milk_after_n_operations 3 initial_milk = 51.2 :=
by
  sorry

end milk_after_three_operations_l205_205319


namespace n_must_be_even_l205_205785

open Nat

-- Define the system of equations:
def equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (∀ i, 2 ≤ i ∧ i ≤ n - 1 → (-x (i-1) + 2 * x i - x (i+1) = 1)) ∧
  (2 * x 1 - x 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0)

-- Define the last equation separately due to its unique form:
def last_equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (n ≥ 1 → -x (n-1) + 2 * x n = 1)

-- The theorem to prove that n must be even:
theorem n_must_be_even (n : ℕ) (x : ℕ → ℤ) : 
  equation n x → last_equation n x → Even n :=
by
  intros h₁ h₂
  sorry

end n_must_be_even_l205_205785


namespace simplify_expression_l205_205172

theorem simplify_expression :
  ( (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) ) / ( (2 * 3) * (3 * 4) * (4 * 5) * (5 * 6) ) = 1 / 5 :=
by
  sorry

end simplify_expression_l205_205172


namespace plane_distance_l205_205687

theorem plane_distance (n : ℕ) : n % 45 = 0 ∧ (n / 10) % 100 = 39 ∧ n <= 5000 → n = 1395 := 
by
  sorry

end plane_distance_l205_205687


namespace find_genuine_coin_in_three_weighings_l205_205167

theorem find_genuine_coin_in_three_weighings (coins : Fin 15 → ℝ)
  (even_number_of_counterfeit : ∃ n : ℕ, 2 * n < 15 ∧ (∀ i, coins i = 1) ∨ (∃ j, coins j = 0.5)) : 
  ∃ i, coins i = 1 :=
by sorry

end find_genuine_coin_in_three_weighings_l205_205167


namespace slower_speed_l205_205362

theorem slower_speed (x : ℝ) :
  (50 / x = 70 / 14) → x = 10 := by
  sorry

end slower_speed_l205_205362


namespace cos_identity_l205_205224

theorem cos_identity (α : ℝ) (h : Real.cos (π / 4 - α) = -1 / 3) :
  Real.cos (3 * π / 4 + α) = 1 / 3 :=
sorry

end cos_identity_l205_205224


namespace estimate_production_in_March_l205_205682

theorem estimate_production_in_March 
  (monthly_production : ℕ → ℝ)
  (x y : ℝ)
  (hx : x = 3)
  (hy : y = x + 1) : y = 4 :=
by
  sorry

end estimate_production_in_March_l205_205682


namespace find_m_plus_b_l205_205629

-- Define the given equation
def given_line (x y : ℝ) : Prop := x - 3 * y + 11 = 0

-- Define the reflection of the given line about the x-axis
def reflected_line (x y : ℝ) : Prop := x + 3 * y + 11 = 0

-- Define the slope-intercept form of the reflected line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- State the theorem to prove
theorem find_m_plus_b (m b : ℝ) :
  (∀ x y : ℝ, reflected_line x y ↔ slope_intercept_form m b x y) → m + b = -4 :=
by
  sorry

end find_m_plus_b_l205_205629


namespace tenth_student_solved_six_l205_205170

theorem tenth_student_solved_six : 
  ∀ (n : ℕ), 
    (∀ (i : ℕ) (j : ℕ), 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ n → (∀ k : ℕ, k ≤ n → ∃ s : ℕ, s = 7)) → 
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 9 → ∃ p : ℕ, p = 4) → ∃ m : ℕ, m = 6 := 
by
  sorry

end tenth_student_solved_six_l205_205170


namespace real_condition_proof_l205_205666

noncomputable def real_condition_sufficient_but_not_necessary : Prop := 
∀ x : ℝ, (|x - 2| < 1) → ((x^2 + x - 2) > 0) ∧ (¬ ( ∀ y : ℝ, (y^2 + y - 2) > 0 → |y - 2| < 1))

theorem real_condition_proof : real_condition_sufficient_but_not_necessary :=
by
  sorry

end real_condition_proof_l205_205666


namespace sarah_socks_l205_205406

theorem sarah_socks :
  ∃ (a b c : ℕ), a + b + c = 15 ∧ 2 * a + 4 * b + 5 * c = 45 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ (a = 8 ∨ a = 9) :=
by {
  sorry
}

end sarah_socks_l205_205406


namespace angle_supplement_complement_l205_205471

theorem angle_supplement_complement (a : ℝ) (h : 180 - a = 3 * (90 - a)) : a = 45 :=
by
  sorry

end angle_supplement_complement_l205_205471


namespace arith_seq_a12_value_l205_205787

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ (a₄ : ℝ), a 4 = 1 ∧ a 7 = a 4 + 3 * d ∧ a 9 = a 4 + 5 * d

theorem arith_seq_a12_value
  (h₁ : arithmetic_sequence a (13 / 8))
  (h₂ : a 7 + a 9 = 15)
  (h₃ : a 4 = 1) :
  a 12 = 14 :=
sorry

end arith_seq_a12_value_l205_205787


namespace find_constant_term_l205_205648

theorem find_constant_term (q' : ℝ → ℝ) (c : ℝ) (h1 : ∀ q : ℝ, q' q = 3 * q - c)
  (h2 : q' (q' 7) = 306) : c = 252 :=
by
  sorry

end find_constant_term_l205_205648


namespace quartic_root_sum_l205_205175

theorem quartic_root_sum (a n l : ℝ) (h : ∃ (r1 r2 r3 r4 : ℝ), 
  r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧ 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ 
  r1 + r2 + r3 + r4 = 10 ∧
  r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4 = a ∧
  r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 = n ∧
  r1 * r2 * r3 * r4 = l) : 
  a + n + l = 109 :=
sorry

end quartic_root_sum_l205_205175


namespace find_temp_M_l205_205120

section TemperatureProof

variables (M T W Th F : ℕ)

-- Conditions
def avg_temp_MTWT := (M + T + W + Th) / 4 = 48
def avg_temp_TWThF := (T + W + Th + F) / 4 = 40
def temp_F := F = 10

-- Proof
theorem find_temp_M (h1 : avg_temp_MTWT M T W Th)
                    (h2 : avg_temp_TWThF T W Th F)
                    (h3 : temp_F F)
                    : M = 42 :=
sorry

end TemperatureProof

end find_temp_M_l205_205120


namespace distribute_neg3_l205_205589

theorem distribute_neg3 (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y :=
by sorry

end distribute_neg3_l205_205589


namespace joe_lowest_test_score_dropped_l205_205912

theorem joe_lowest_test_score_dropped 
  (A B C D : ℝ) 
  (h1 : A + B + C + D = 360) 
  (h2 : A + B + C = 255) :
  D = 105 :=
sorry

end joe_lowest_test_score_dropped_l205_205912


namespace largest_side_of_rectangle_l205_205291

theorem largest_side_of_rectangle (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 := 
by
  sorry

end largest_side_of_rectangle_l205_205291


namespace compare_a_b_c_l205_205239

noncomputable def a : ℝ := (1 / 3)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 2)
noncomputable def c : ℝ := Real.logb (1 / 3) (1 / 4)

theorem compare_a_b_c : b < a ∧ a < c := by
  sorry

end compare_a_b_c_l205_205239


namespace average_output_l205_205790

theorem average_output (time1 time2 rate1 rate2 cogs1 cogs2 total_cogs total_time: ℝ) :
  rate1 = 20 → cogs1 = 60 → time1 = cogs1 / rate1 →
  rate2 = 60 → cogs2 = 60 → time2 = cogs2 / rate2 →
  total_cogs = cogs1 + cogs2 → total_time = time1 + time2 →
  (total_cogs / total_time = 30) :=
by
  intros hrate1 hcogs1 htime1 hrate2 hcogs2 htime2 htotalcogs htotaltime
  sorry

end average_output_l205_205790


namespace train_length_l205_205264

theorem train_length (L : ℝ) :
  (∀ t₁ t₂ : ℝ, t₁ = t₂ → L = t₁ / 2) →
  (∀ t : ℝ, t = (8 / 3600) * 36 → L * 2 = t) →
  44 - 36 = 8 →
  L = 40 :=
by
  sorry

end train_length_l205_205264


namespace average_growth_rate_le_max_growth_rate_l205_205758

variable (P : ℝ) (a : ℝ) (b : ℝ) (x : ℝ)

theorem average_growth_rate_le_max_growth_rate (h : (1 + x)^2 = (1 + a) * (1 + b)) :
  x ≤ max a b := 
sorry

end average_growth_rate_le_max_growth_rate_l205_205758


namespace tan_simplification_l205_205473

theorem tan_simplification 
  (θ : ℝ) 
  (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
by 
  sorry

end tan_simplification_l205_205473


namespace true_proposition_among_choices_l205_205974

theorem true_proposition_among_choices (p q : Prop) (hp : p) (hq : ¬ q) :
  p ∧ ¬ q :=
by
  sorry

end true_proposition_among_choices_l205_205974


namespace no_rain_five_days_l205_205859

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end no_rain_five_days_l205_205859


namespace positive_real_as_sum_l205_205721

theorem positive_real_as_sum (k : ℝ) (hk : k > 0) : 
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ n, a n < a (n + 1)) ∧ (∑' n, 1 / 10 ^ a n = k) :=
sorry

end positive_real_as_sum_l205_205721


namespace gcd_divisor_l205_205276

theorem gcd_divisor (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) (hrs : Nat.gcd r s = 60) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) 
  : 13 ∣ p :=
sorry

end gcd_divisor_l205_205276


namespace condition_two_eqn_l205_205683

def line_through_point_and_perpendicular (x1 y1 : ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, (y - y1) = -1/(x - x1) * (x - x1 + c) → x - y + c = 0

theorem condition_two_eqn :
  line_through_point_and_perpendicular 1 (-2) (-3) :=
sorry

end condition_two_eqn_l205_205683


namespace missing_number_l205_205762

theorem missing_number (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 :=
by
sorry

end missing_number_l205_205762


namespace total_coins_l205_205703

-- Define the number of stacks and the number of coins per stack
def stacks : ℕ := 5
def coins_per_stack : ℕ := 3

-- State the theorem to prove the total number of coins
theorem total_coins (s c : ℕ) (hs : s = stacks) (hc : c = coins_per_stack) : s * c = 15 :=
by
  -- Proof is omitted
  sorry

end total_coins_l205_205703


namespace quadratic_radical_type_l205_205065

-- Problem statement: Given that sqrt(2a + 1) is a simplest quadratic radical and the same type as sqrt(48), prove that a = 1.

theorem quadratic_radical_type (a : ℝ) (h1 : ((2 * a) + 1) = 3) : a = 1 :=
by
  sorry

end quadratic_radical_type_l205_205065


namespace solution_set_A_solution_set_B_subset_A_l205_205492

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_A :
  {x : ℝ | f x > 6} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

theorem solution_set_B_subset_A {a : ℝ} :
  (∀ x, f x > |a-1| → x < -1 ∨ x > 2) → a ≤ -5 ∨ a ≥ 7 :=
sorry

end solution_set_A_solution_set_B_subset_A_l205_205492


namespace ajhsme_1989_reappears_at_12_l205_205405

def cycle_length_letters : ℕ := 6
def cycle_length_digits  : ℕ := 4
def target_position : ℕ := Nat.lcm cycle_length_letters cycle_length_digits

theorem ajhsme_1989_reappears_at_12 :
  target_position = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end ajhsme_1989_reappears_at_12_l205_205405


namespace opposite_of_neg_nine_l205_205996

theorem opposite_of_neg_nine : -(-9) = 9 :=
by
  sorry

end opposite_of_neg_nine_l205_205996


namespace correct_sampling_methods_l205_205177

-- Definitions for different sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Conditions from the problem
def situation1 (students_selected_per_class : Nat) : Prop :=
  students_selected_per_class = 2

def situation2 (students_above_110 : Nat) (students_between_90_and_100 : Nat) (students_below_90 : Nat) : Prop :=
  students_above_110 = 10 ∧ students_between_90_and_100 = 40 ∧ students_below_90 = 12

def situation3 (tracks_arranged_for_students : Nat) : Prop :=
  tracks_arranged_for_students = 6

-- Theorem
theorem correct_sampling_methods :
  ∀ (students_selected_per_class students_above_110 students_between_90_and_100 students_below_90 tracks_arranged_for_students: Nat),
  situation1 students_selected_per_class →
  situation2 students_above_110 students_between_90_and_100 students_below_90 →
  situation3 tracks_arranged_for_students →
  (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) = (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
by
  intros
  rfl

end correct_sampling_methods_l205_205177


namespace sum_of_base_8_digits_888_l205_205561

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l205_205561


namespace chip_exits_from_A2_l205_205132

noncomputable def chip_exit_cell (grid_size : ℕ) (initial_cell : ℕ × ℕ) (move_direction : ℕ × ℕ → ℕ × ℕ) : ℕ × ℕ :=
(1, 2) -- A2; we assume the implementation of function movement follows the solution as described

theorem chip_exits_from_A2 :
  chip_exit_cell 4 (3, 2) move_direction = (1, 2) :=
sorry  -- Proof omitted

end chip_exits_from_A2_l205_205132


namespace air_conditioner_consumption_l205_205973

theorem air_conditioner_consumption :
  ∀ (total_consumption_8_hours : ℝ)
    (hours_8 : ℝ)
    (hours_per_day : ℝ)
    (days : ℝ),
    total_consumption_8_hours / hours_8 * hours_per_day * days = 27 :=
by
  intros total_consumption_8_hours hours_8 hours_per_day days
  sorry

end air_conditioner_consumption_l205_205973


namespace find_boys_l205_205215

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end find_boys_l205_205215


namespace minimum_value_of_expression_l205_205302

theorem minimum_value_of_expression (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : 2 * x + 3 * y = 8) : 
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 → (2 / a + 3 / b) ≥ 25 / 8) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 ∧ 2 / a + 3 / b = 25 / 8) :=
sorry

end minimum_value_of_expression_l205_205302


namespace cost_per_minute_of_each_call_l205_205742

theorem cost_per_minute_of_each_call :
  let calls_per_week := 50
  let hours_per_call := 1
  let weeks_per_month := 4
  let total_hours_in_month := calls_per_week * hours_per_call * weeks_per_month
  let total_cost := 600
  let cost_per_hour := total_cost / total_hours_in_month
  let minutes_per_hour := 60
  let cost_per_minute := cost_per_hour / minutes_per_hour
  cost_per_minute = 0.05 := 
by
  sorry

end cost_per_minute_of_each_call_l205_205742


namespace arun_weight_lower_limit_l205_205329

theorem arun_weight_lower_limit :
  ∃ (w : ℝ), w > 60 ∧ w <= 64 ∧ (∀ (a : ℝ), 60 < a ∧ a <= 64 → ((a + 64) / 2 = 63) → a = 62) :=
by
  sorry

end arun_weight_lower_limit_l205_205329


namespace symmetric_points_y_axis_l205_205114

theorem symmetric_points_y_axis (a b : ℤ) 
  (h1 : a + 1 = 2) 
  (h2 : b + 2 = 3) : 
  a + b = 2 :=
by
  sorry

end symmetric_points_y_axis_l205_205114


namespace sequence_term_general_formula_l205_205271

theorem sequence_term_general_formula (S : ℕ → ℚ) (a : ℕ → ℚ) :
  (∀ n, S n = n^2 + (1/2)*n + 5) →
  (∀ n, (n ≥ 2) → a n = S n - S (n - 1)) →
  a 1 = 13/2 →
  (∀ n, a n = if n = 1 then 13/2 else 2*n - 1/2) :=
by
  intros hS ha h1
  sorry

end sequence_term_general_formula_l205_205271


namespace intersection_of_A_and_B_l205_205924

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end intersection_of_A_and_B_l205_205924


namespace pat_peano_maximum_pages_l205_205603

noncomputable def count_fives_in_range : ℕ → ℕ := sorry

theorem pat_peano_maximum_pages (n : ℕ) : 
  (count_fives_in_range 54) = 15 → n ≤ 54 :=
sorry

end pat_peano_maximum_pages_l205_205603


namespace log_expression_evaluation_l205_205307

theorem log_expression_evaluation (log2 log5 : ℝ) (h : log2 + log5 = 1) :
  log2 * (log5 + log10) + 2 * log5 - log5 * log20 = 1 := by
  sorry

end log_expression_evaluation_l205_205307


namespace exponentiation_multiplication_l205_205366

theorem exponentiation_multiplication (a : ℝ) : a^6 * a^2 = a^8 :=
by sorry

end exponentiation_multiplication_l205_205366


namespace vector_parallel_l205_205798

theorem vector_parallel (x y : ℝ) (a b : ℝ × ℝ × ℝ) (h_parallel : a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ k : ℝ, a = k • b) : x + y = 6 :=
by sorry

end vector_parallel_l205_205798


namespace abs_inequality_k_ge_neg3_l205_205610

theorem abs_inequality_k_ge_neg3 (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 :=
sorry

end abs_inequality_k_ge_neg3_l205_205610


namespace minimizing_reciprocal_sum_l205_205476

theorem minimizing_reciprocal_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 30) :
  a = 10 ∧ b = 5 :=
by
  sorry

end minimizing_reciprocal_sum_l205_205476


namespace inequality_proof_l205_205265

theorem inequality_proof 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := 
by {
  sorry
}

end inequality_proof_l205_205265


namespace distribute_ways_l205_205744

/-- There are 5 distinguishable balls and 4 distinguishable boxes.
The total number of ways to distribute these balls into the boxes is 1024. -/
theorem distribute_ways : (4 : ℕ) ^ (5 : ℕ) = 1024 := by
  sorry

end distribute_ways_l205_205744


namespace pyramid_distance_to_larger_cross_section_l205_205505

theorem pyramid_distance_to_larger_cross_section
  (A1 A2 : ℝ) (d : ℝ)
  (h : ℝ)
  (hA1 : A1 = 256 * Real.sqrt 2)
  (hA2 : A2 = 576 * Real.sqrt 2)
  (hd : d = 12)
  (h_ratio : (Real.sqrt (A1 / A2)) = 2 / 3) :
  h = 36 := 
  sorry

end pyramid_distance_to_larger_cross_section_l205_205505


namespace apples_per_basket_l205_205222

theorem apples_per_basket (total_apples : ℕ) (baskets : ℕ) (h1 : total_apples = 629) (h2 : baskets = 37) :
  total_apples / baskets = 17 :=
by
  sorry

end apples_per_basket_l205_205222


namespace rectangle_area_l205_205692

noncomputable def circle_radius := 8
noncomputable def rect_ratio : ℕ × ℕ := (3, 1)
noncomputable def rect_area (width length : ℕ) : ℕ := width * length

theorem rectangle_area (width length : ℕ) 
  (h1 : 2 * circle_radius = width) 
  (h2 : rect_ratio.1 * width = length) : 
  rect_area width length = 768 := 
sorry

end rectangle_area_l205_205692


namespace complement_of_A_in_U_l205_205321

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 2, 4, 5}

-- Proof statement
theorem complement_of_A_in_U : (U \ A) = {3, 6, 7} := by
  sorry

end complement_of_A_in_U_l205_205321


namespace shifted_function_is_correct_l205_205538

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l205_205538


namespace trig_identity_l205_205311

open Real

theorem trig_identity (α : ℝ) (h : 2 * sin α + cos α = 0) : 
  2 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = -12 / 5 :=
sorry

end trig_identity_l205_205311


namespace ratio_dark_blue_to_total_l205_205210

-- Definitions based on the conditions
def total_marbles := 63
def red_marbles := 38
def green_marbles := 4
def dark_blue_marbles := total_marbles - red_marbles - green_marbles

-- The statement to be proven
theorem ratio_dark_blue_to_total : (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end ratio_dark_blue_to_total_l205_205210


namespace identify_conic_section_l205_205604

theorem identify_conic_section (x y : ℝ) :
  (x + 7)^2 = (5 * y - 6)^2 + 125 →
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e * x * y + f = 0 ∧
  (a > 0) ∧ (b < 0) := sorry

end identify_conic_section_l205_205604


namespace monotonic_f_iff_l205_205783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a * x + 5 else 1 + 1 / x

theorem monotonic_f_iff {a : ℝ} :  
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end monotonic_f_iff_l205_205783


namespace find_three_numbers_l205_205499

theorem find_three_numbers (x y z : ℝ)
  (h1 : x - y = (1 / 3) * z)
  (h2 : y - z = (1 / 3) * x)
  (h3 : z - 10 = (1 / 3) * y) :
  x = 45 ∧ y = 37.5 ∧ z = 22.5 :=
by
  sorry

end find_three_numbers_l205_205499


namespace students_left_zoo_l205_205272

theorem students_left_zoo
  (students_first_class students_second_class : ℕ)
  (chaperones teachers : ℕ)
  (initial_individuals remaining_individuals : ℕ)
  (chaperones_left remaining_individuals_after_chaperones_left : ℕ)
  (remaining_students initial_students : ℕ)
  (H1 : students_first_class = 10)
  (H2 : students_second_class = 10)
  (H3 : chaperones = 5)
  (H4 : teachers = 2)
  (H5 : initial_individuals = students_first_class + students_second_class + chaperones + teachers) 
  (H6 : initial_individuals = 27)
  (H7 : remaining_individuals = 15)
  (H8 : chaperones_left = 2)
  (H9 : remaining_individuals_after_chaperones_left = remaining_individuals - chaperones_left)
  (H10 : remaining_individuals_after_chaperones_left = 13)
  (H11 : remaining_students = remaining_individuals_after_chaperones_left - teachers)
  (H12 : remaining_students = 11)
  (H13 : initial_students = students_first_class + students_second_class)
  (H14 : initial_students = 20) :
  20 - 11 = 9 :=
by sorry

end students_left_zoo_l205_205272


namespace solution_to_inequality_l205_205016

theorem solution_to_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : 1 / x > 1 :=
by
  sorry

end solution_to_inequality_l205_205016


namespace find_amount_after_two_years_l205_205537

noncomputable def initial_value : ℝ := 64000
noncomputable def yearly_increase (amount : ℝ) : ℝ := amount / 9
noncomputable def amount_after_year (amount : ℝ) : ℝ := amount + yearly_increase amount
noncomputable def amount_after_two_years : ℝ := amount_after_year (amount_after_year initial_value)

theorem find_amount_after_two_years : amount_after_two_years = 79012.34 :=
by
  sorry

end find_amount_after_two_years_l205_205537


namespace problem_statement_l205_205337

theorem problem_statement
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : ab + ac + bc ≠ 0) :
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
  sorry

end problem_statement_l205_205337


namespace ratio_of_earnings_l205_205899

theorem ratio_of_earnings (jacob_hourly: ℕ) (jake_total: ℕ) (days: ℕ) (hours_per_day: ℕ) (jake_hourly: ℕ) (ratio: ℕ) 
  (h_jacob: jacob_hourly = 6)
  (h_jake_total: jake_total = 720)
  (h_days: days = 5)
  (h_hours_per_day: hours_per_day = 8)
  (h_jake_hourly: jake_hourly = jake_total / (days * hours_per_day))
  (h_ratio: ratio = jake_hourly / jacob_hourly) :
  ratio = 3 := 
sorry

end ratio_of_earnings_l205_205899


namespace circle_area_of_white_cube_l205_205700

/-- 
Marla has a large white cube with an edge length of 12 feet and enough green paint to cover 432 square feet.
Marla paints a white circle centered on each face of the cube, surrounded by a green border.
Prove the area of one of the white circles is 72 square feet.
 -/
theorem circle_area_of_white_cube
  (edge_length : ℝ) (paint_area : ℝ) (faces : ℕ)
  (h_edge_length : edge_length = 12)
  (h_paint_area : paint_area = 432)
  (h_faces : faces = 6) :
  ∃ (circle_area : ℝ), circle_area = 72 :=
by
  sorry

end circle_area_of_white_cube_l205_205700


namespace minimum_n_l205_205579

-- Assume the sequence a_n is defined as part of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

-- Define S_n as the sum of the first n terms in the sequence
def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2 * d

-- Given conditions
def a1 := 2
def d := 1  -- Derived from the condition a1 + a4 = a5

-- Problem Statement
theorem minimum_n (n : ℕ) :
  (sum_arithmetic_sequence a1 d n > 32) ↔ n = 6 :=
sorry

end minimum_n_l205_205579


namespace set_inclusion_l205_205969

def setM : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}

def setN : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}

def setP : Set ℝ := {a | ∃ k : ℤ, a = (k * Real.pi / 2) + (Real.pi / 4)}

theorem set_inclusion : setP ⊆ setN ∧ setN ⊆ setM := by
  sorry

end set_inclusion_l205_205969


namespace arithmetic_sequence_a3_l205_205498

variable (a : ℕ → ℕ)
variable (S5 : ℕ)
variable (arithmetic_seq : Prop)

def is_arithmetic_seq (a : ℕ → ℕ) : Prop := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a3 (h1 : is_arithmetic_seq a) (h2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 25) : a 3 = 5 :=
by
  sorry

end arithmetic_sequence_a3_l205_205498


namespace cynthia_more_miles_l205_205722

open Real

noncomputable def david_speed : ℝ := 55 / 5
noncomputable def cynthia_speed : ℝ := david_speed + 3

theorem cynthia_more_miles (t : ℝ) (ht : t = 5) :
  (cynthia_speed * t) - (david_speed * t) = 15 :=
by
  sorry

end cynthia_more_miles_l205_205722


namespace invisible_dots_48_l205_205693

theorem invisible_dots_48 (visible : Multiset ℕ) (hv : visible = [1, 2, 3, 3, 4, 5, 6, 6, 6]) :
  let total_dots := 4 * (1 + 2 + 3 + 4 + 5 + 6)
  let visible_sum := visible.sum
  total_dots - visible_sum = 48 :=
by
  sorry

end invisible_dots_48_l205_205693


namespace two_digit_integer_one_less_than_lcm_of_3_4_7_l205_205136

theorem two_digit_integer_one_less_than_lcm_of_3_4_7 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n + 1) % (Nat.lcm (Nat.lcm 3 4) 7) = 0 ∧ n = 83 := by
  sorry

end two_digit_integer_one_less_than_lcm_of_3_4_7_l205_205136


namespace octahedron_volume_l205_205234

theorem octahedron_volume (a : ℝ) (h1 : a > 0) :
  (∃ V : ℝ, V = (a^3 * Real.sqrt 2) / 3) :=
sorry

end octahedron_volume_l205_205234


namespace students_playing_both_correct_l205_205371

def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def neither_players : ℕ := 7
def students_playing_both : ℕ := 17

theorem students_playing_both_correct :
  total_students - neither_players = (football_players + long_tennis_players) - students_playing_both :=
by 
  sorry

end students_playing_both_correct_l205_205371


namespace bonus_percentage_is_correct_l205_205140

theorem bonus_percentage_is_correct (kills total_points enemies_points bonus_threshold bonus_percentage : ℕ) 
  (h1 : enemies_points = 10) 
  (h2 : kills = 150) 
  (h3 : total_points = 2250) 
  (h4 : bonus_threshold = 100) 
  (h5 : kills >= bonus_threshold) 
  (h6 : bonus_percentage = (total_points - kills * enemies_points) * 100 / (kills * enemies_points)) : 
  bonus_percentage = 50 := 
by
  sorry

end bonus_percentage_is_correct_l205_205140


namespace find_f_l205_205026

variable (f : ℝ → ℝ)

open Function

theorem find_f (h : ∀ x: ℝ, f (3 * x + 2) = 9 * x + 8) : ∀ x: ℝ, f x = 3 * x + 2 := 
sorry

end find_f_l205_205026


namespace probability_between_C_and_D_l205_205143

theorem probability_between_C_and_D :
  ∀ (A B C D : ℝ) (AB AD BC : ℝ),
    AB = 3 * AD ∧ AB = 6 * BC ∧ D - A = AD ∧ C - A = AD + BC ∧ B - A = AB →
    (C < D) →
    ∃ p : ℝ, p = 1 / 2 := by
  sorry

end probability_between_C_and_D_l205_205143


namespace value_of_b_l205_205950

theorem value_of_b (b : ℝ) (x : ℝ) (h : x = 1) (h_eq : 3 * x^2 - b * x + 3 = 0) : b = 6 :=
by
  sorry

end value_of_b_l205_205950


namespace find_a_minus_c_l205_205231

section
variables (a b c : ℝ)
variables (h₁ : (a + b) / 2 = 110) (h₂ : (b + c) / 2 = 170)

theorem find_a_minus_c : a - c = -120 :=
by
  sorry
end

end find_a_minus_c_l205_205231


namespace find_u_plus_v_l205_205976

theorem find_u_plus_v (u v : ℚ) 
  (h₁ : 3 * u + 7 * v = 17) 
  (h₂ : 5 * u - 3 * v = 9) : 
  u + v = 43 / 11 :=
sorry

end find_u_plus_v_l205_205976


namespace bread_rise_times_l205_205422

-- Defining the conditions
def rise_time : ℕ := 120
def kneading_time : ℕ := 10
def baking_time : ℕ := 30
def total_time : ℕ := 280

-- The proof statement
theorem bread_rise_times (n : ℕ) 
  (h1 : rise_time * n + kneading_time + baking_time = total_time) 
  : n = 2 :=
sorry

end bread_rise_times_l205_205422


namespace angle_proof_l205_205577

-- Variables and assumptions
variable {α : Type} [LinearOrderedField α]    -- using a general type for angles
variable {A B C D E : α}                       -- points of the triangle and extended segment

-- Given conditions
variable (angle_ACB angle_ABC : α)
variable (H1 : angle_ACB = 2 * angle_ABC)      -- angle condition
variable (CD BD AD DE : α)
variable (H2 : CD = 2 * BD)                    -- segment length condition
variable (H3 : AD = DE)                        -- extended segment condition

-- The proof goal in Lean format
theorem angle_proof (H1 : angle_ACB = 2 * angle_ABC) 
  (H2 : CD = 2 * BD) 
  (H3 : AD = DE) :
  angle_ECB + 180 = 2 * angle_EBC := 
sorry  -- proof to be filled in

end angle_proof_l205_205577


namespace no_such_n_exists_l205_205248

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n * sum_of_digits n = 100200300 :=
by
  sorry

end no_such_n_exists_l205_205248


namespace total_children_l205_205710

variable (S C B T : ℕ)

theorem total_children (h1 : T < 19) 
                       (h2 : S = 3 * C) 
                       (h3 : B = S / 2) 
                       (h4 : T = B + S + 1) : 
                       T = 10 := 
  sorry

end total_children_l205_205710


namespace sample_size_l205_205259

theorem sample_size (k n : ℕ) (r : 2 * k + 3 * k + 5 * k = 10 * k) (h : 3 * k = 12) : n = 40 :=
by {
    -- here, we will provide a proof to demonstrate that n = 40 given the conditions
    sorry
}

end sample_size_l205_205259


namespace no_nat_numbers_satisfy_lcm_eq_l205_205012

theorem no_nat_numbers_satisfy_lcm_eq (n m : ℕ) :
  ¬ (Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019) :=
sorry

end no_nat_numbers_satisfy_lcm_eq_l205_205012


namespace find_denominator_l205_205400

theorem find_denominator (y : ℝ) (x : ℝ) (h₀ : y > 0) (h₁ : 9 * y / 20 + 3 * y / x = 0.75 * y) : x = 10 :=
sorry

end find_denominator_l205_205400


namespace tan_angle_addition_l205_205225

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_angle_addition_l205_205225


namespace maximize_probability_sum_is_15_l205_205696

def initial_list : List ℤ := [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]

def valid_pairs (lst : List ℤ) : List (ℤ × ℤ) :=
  (lst.product lst).filter (λ ⟨x, y⟩ => x < y ∧ x + y = 15)

def remove_one_element (lst : List ℤ) (x : ℤ) : List ℤ :=
  lst.erase x

theorem maximize_probability_sum_is_15 :
  (List.length (valid_pairs (remove_one_element initial_list 8))
   = List.maximum (List.map (λ x => List.length (valid_pairs (remove_one_element initial_list x))) initial_list)) :=
sorry

end maximize_probability_sum_is_15_l205_205696


namespace part_one_part_two_l205_205494

variable {x m : ℝ}

theorem part_one (h1 : ∀ x : ℝ, ¬(m * x^2 - (m + 1) * x + (m + 1) ≥ 0)) : m < -1 := sorry

theorem part_two (h2 : ∀ x : ℝ, 1 < x → m * x^2 - (m + 1) * x + (m + 1) ≥ 0) : m ≥ 1 / 3 := sorry

end part_one_part_two_l205_205494


namespace additional_length_of_track_l205_205040

theorem additional_length_of_track (rise : ℝ) (grade1 grade2 : ℝ) (h_rise : rise = 800) (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.02) :
  (rise / grade2) - (rise / grade1) = 20000 :=
by
  sorry

end additional_length_of_track_l205_205040


namespace Sam_bought_cards_l205_205325

theorem Sam_bought_cards (original_cards current_cards : ℕ) 
  (h1 : original_cards = 87) (h2 : current_cards = 74) : 
  original_cards - current_cards = 13 :=
by
  -- The 'sorry' here means the proof is omitted.
  sorry

end Sam_bought_cards_l205_205325


namespace condition_for_equation_l205_205000

theorem condition_for_equation (a b c : ℕ) (ha : 0 < a ∧ a < 20) (hb : 0 < b ∧ b < 20) (hc : 0 < c ∧ c < 20) :
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 :=
by
  sorry

end condition_for_equation_l205_205000


namespace age_difference_is_20_l205_205820

-- Definitions for the ages of the two persons
def elder_age := 35
def younger_age := 15

-- Condition: Difference in ages
def age_difference := elder_age - younger_age

-- Theorem to prove the difference in ages is 20 years
theorem age_difference_is_20 : age_difference = 20 := by
  sorry

end age_difference_is_20_l205_205820


namespace miles_flown_on_thursday_l205_205761
-- Importing the necessary library

-- Defining the problem conditions and the proof goal
theorem miles_flown_on_thursday (x : ℕ) : 
  (∀ y, (3 * (1134 + y) = 7827) → y = x) → x = 1475 :=
by
  intro h
  specialize h 1475
  sorry

end miles_flown_on_thursday_l205_205761


namespace c_geq_one_l205_205464

theorem c_geq_one (a b : ℕ) (c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : c ≥ 1 :=
by sorry

end c_geq_one_l205_205464


namespace fruit_basket_apples_oranges_ratio_l205_205456

theorem fruit_basket_apples_oranges_ratio : 
  ∀ (apples oranges : ℕ), 
  apples = 15 ∧ (2 * apples / 3 + 2 * oranges / 3 = 50) → (apples = 15 ∧ oranges = 60) → apples / gcd apples oranges = 1 ∧ oranges / gcd apples oranges = 4 :=
by 
  intros apples oranges h1 h2
  have h_apples : apples = 15 := by exact h2.1
  have h_oranges : oranges = 60 := by exact h2.2
  rw [h_apples, h_oranges]
  sorry

end fruit_basket_apples_oranges_ratio_l205_205456


namespace arithmetic_sequence_general_term_l205_205734

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ)
  (h1 : ∀ n m, a (n+1) - a n = a (m+1) - a m)
  (h2 : (a 2 + a 6) / 2 = 5)
  (h3 : (a 3 + a 7) / 2 = 7) :
  ∀ n, a n = 2 * n - 3 :=
by 
  sorry

end arithmetic_sequence_general_term_l205_205734


namespace ceiling_lights_l205_205784

variable (S M L : ℕ)

theorem ceiling_lights (hM : M = 12) (hL : L = 2 * M)
    (hBulbs : S + 2 * M + 3 * L = 118) : S - M = 10 :=
by
  sorry

end ceiling_lights_l205_205784


namespace find_f_value_l205_205086

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_value (α : ℝ) (h : f 3 α = Real.sqrt 3) : f (1 / 4) α = 1 / 2 :=
by
  sorry

end find_f_value_l205_205086


namespace hyperbola_asymptote_perpendicular_to_line_l205_205778

variable {a : ℝ}

theorem hyperbola_asymptote_perpendicular_to_line (h : a > 0)
  (C : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1)
  (l : ∀ x y : ℝ, 2 * x - y + 1 = 0) :
  a = 2 :=
by
  sorry

end hyperbola_asymptote_perpendicular_to_line_l205_205778


namespace equal_distribution_of_drawings_l205_205838

theorem equal_distribution_of_drawings (total_drawings : ℕ) (neighbors : ℕ) (drawings_per_neighbor : ℕ)
  (h1 : total_drawings = 54)
  (h2 : neighbors = 6)
  (h3 : total_drawings = neighbors * drawings_per_neighbor) :
  drawings_per_neighbor = 9 :=
by
  rw [h1, h2] at h3
  linarith

end equal_distribution_of_drawings_l205_205838


namespace greatest_possible_x_l205_205011

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end greatest_possible_x_l205_205011


namespace average_excluding_highest_lowest_l205_205942

-- Define the conditions
def batting_average : ℚ := 59
def innings : ℕ := 46
def highest_score : ℕ := 156
def score_difference : ℕ := 150
def lowest_score : ℕ := highest_score - score_difference

-- Prove the average excluding the highest and lowest innings is 58
theorem average_excluding_highest_lowest :
  let total_runs := batting_average * innings
  let runs_excluding := total_runs - highest_score - lowest_score
  let effective_innings := innings - 2
  runs_excluding / effective_innings = 58 := by
  -- Insert proof here
  sorry

end average_excluding_highest_lowest_l205_205942


namespace find_initial_money_l205_205594
 
theorem find_initial_money (x : ℕ) (gift_grandma gift_aunt_uncle gift_parents total_money : ℕ) 
  (h1 : gift_grandma = 25) 
  (h2 : gift_aunt_uncle = 20) 
  (h3 : gift_parents = 75) 
  (h4 : total_money = 279) 
  (h : x + (gift_grandma + gift_aunt_uncle + gift_parents) = total_money) : 
  x = 159 :=
by
  sorry

end find_initial_money_l205_205594


namespace divides_three_and_eleven_l205_205115

theorem divides_three_and_eleven (n : ℕ) (h : n ≥ 1) : (n ∣ 3^n + 1 ∧ n ∣ 11^n + 1) ↔ (n = 1 ∨ n = 2) := by
  sorry

end divides_three_and_eleven_l205_205115


namespace parabola_directrix_l205_205413

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l205_205413


namespace most_convincing_method_l205_205867

-- Defining the survey data
def male_participants : Nat := 4258
def male_believe_doping : Nat := 2360
def female_participants : Nat := 3890
def female_believe_framed : Nat := 2386

-- Defining the question-to-answer equivalence related to the most convincing method
theorem most_convincing_method :
  "Independence Test" = "Independence Test" := 
by
  sorry

end most_convincing_method_l205_205867


namespace min_time_to_cook_noodles_l205_205941

/-- 
Li Ming needs to cook noodles, following these steps: 
① Boil the noodles for 4 minutes; 
② Wash vegetables for 5 minutes; 
③ Prepare the noodles and condiments for 2 minutes; 
④ Boil the water in the pot for 10 minutes; 
⑤ Wash the pot and add water for 2 minutes. 
Apart from step ④, only one step can be performed at a time. 
Prove that the minimum number of minutes needed to complete these tasks is 16.
-/
def total_time : Nat :=
  let t5 := 2 -- Wash the pot and add water
  let t4 := 10 -- Boil the water in the pot
  let t2 := 5 -- Wash vegetables
  let t3 := 2 -- Prepare the noodles and condiments
  let t1 := 4 -- Boil the noodles
  t5 + t4.max (t2 + t3) + t1

theorem min_time_to_cook_noodles : total_time = 16 :=
by
  sorry

end min_time_to_cook_noodles_l205_205941


namespace candidate_lost_by_2460_votes_l205_205158

noncomputable def total_votes : ℝ := 8199.999999999998
noncomputable def candidate_percentage : ℝ := 0.35
noncomputable def rival_percentage : ℝ := 1 - candidate_percentage
noncomputable def candidate_votes := candidate_percentage * total_votes
noncomputable def rival_votes := rival_percentage * total_votes
noncomputable def votes_lost_by := rival_votes - candidate_votes

theorem candidate_lost_by_2460_votes : votes_lost_by = 2460 := by
  sorry

end candidate_lost_by_2460_votes_l205_205158


namespace find_N_l205_205677

theorem find_N (N : ℕ) (h_pos : N > 0) (h_small_factors : 1 + 3 = 4) 
  (h_large_factors : N + N / 3 = 204) : N = 153 :=
  by sorry

end find_N_l205_205677


namespace tangent_line_characterization_l205_205614

theorem tangent_line_characterization 
  (α β m n : ℝ) 
  (h_pos_α : 0 < α) 
  (h_pos_β : 0 < β) 
  (h_alpha_beta : 1/α + 1/β = 1)
  (h_pos_m : 0 < m)
  (h_pos_n : 0 < n) :
  (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ x^α + y^α = 1 → mx + ny = 1) ↔ (m^β + n^β = 1) := 
sorry

end tangent_line_characterization_l205_205614


namespace series_sum_correct_l205_205910

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem series_sum_correct :
  geometric_series_sum (1 / 2) (-1 / 3) 6 = 91 / 243 :=
by
  -- Proof goes here
  sorry

end series_sum_correct_l205_205910


namespace polygon_number_of_sides_l205_205300

-- Definitions based on conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def exterior_angle (angle : ℕ) : ℕ := 30

-- The theorem statement
theorem polygon_number_of_sides (n : ℕ) (angle : ℕ) 
  (h1 : sum_of_exterior_angles n = 360)
  (h2 : exterior_angle angle = 30) : 
  n = 12 := 
by
  sorry

end polygon_number_of_sides_l205_205300


namespace problem1_problem2_l205_205918

noncomputable def f (x : ℝ) : ℝ :=
  if h : 1 ≤ x then x else 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  a * f x - |x - 2|

def problem1_statement (b : ℝ) : Prop :=
  ∀ x, x > 0 → g x 0 ≤ |x - 1| + b

def problem2_statement : Prop :=
  ∃ x, (0 < x) ∧ ∀ y, (0 < y) → g y 1 ≥ g x 1

theorem problem1 : ∀ b : ℝ, problem1_statement b ↔ b ∈ Set.Ici (-1) := sorry

theorem problem2 : ∃ x, problem2_statement ∧ g x 1 = 0 := sorry

end problem1_problem2_l205_205918


namespace expected_value_unfair_die_l205_205352

theorem expected_value_unfair_die :
  let p8 := 3 / 8
  let p1_7 := (1 - p8) / 7
  let E := p1_7 * (1 + 2 + 3 + 4 + 5 + 6 + 7) + p8 * 8
  E = 5.5 := by
  sorry

end expected_value_unfair_die_l205_205352


namespace sum_of_parallelogram_sides_l205_205219

-- Definitions of the given conditions.
def length_one_side : ℕ := 10
def length_other_side : ℕ := 7

-- Theorem stating the sum of the lengths of the four sides of the parallelogram.
theorem sum_of_parallelogram_sides : 
    (length_one_side + length_one_side + length_other_side + length_other_side) = 34 :=
by
    sorry

end sum_of_parallelogram_sides_l205_205219


namespace max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l205_205602

theorem max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h₁ : a ≤ b) (h₂ : b ≤ c) (h₃ : c ≤ 2 * a) :
    b / a + c / b + a / c ≤ 7 / 2 := 
  sorry

end max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l205_205602


namespace physics_marks_l205_205884

theorem physics_marks
  (P C M : ℕ)
  (h1 : P + C + M = 240)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 80 :=
by
  sorry

end physics_marks_l205_205884


namespace fraction_picked_l205_205414

/--
An apple tree has three times as many apples as the number of plums on a plum tree.
Damien picks a certain fraction of the fruits from the trees, and there are 96 plums
and apples remaining on the tree. There were 180 apples on the apple tree before 
Damien picked any of the fruits. Prove that Damien picked 3/5 of the fruits from the trees.
-/
theorem fraction_picked (P F : ℝ) (h1 : 3 * P = 180) (h2 : (1 - F) * (180 + P) = 96) :
  F = 3 / 5 :=
by
  sorry

end fraction_picked_l205_205414


namespace minimal_n_is_40_l205_205508

def sequence_minimal_n (p : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = p ∧
  a 2 = p + 1 ∧
  (∀ n, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
  (∀ n, a n ≥ p) -- Since minimal \(a_n\) implies non-negative with given \(a_1, a_2\)

theorem minimal_n_is_40 (p : ℝ) (a : ℕ → ℝ) (h : sequence_minimal_n p a) : ∃ n, n = 40 ∧ (∀ m, n ≠ m → a n ≤ a m) :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end minimal_n_is_40_l205_205508


namespace tree_height_end_of_2_years_l205_205513

theorem tree_height_end_of_2_years (h4 : ℕ → ℕ)
  (h_tripling : ∀ n, h4 (n + 1) = 3 * h4 n)
  (h4_at_4 : h4 4 = 81) :
  h4 2 = 9 :=
by
  sorry

end tree_height_end_of_2_years_l205_205513


namespace average_power_heater_l205_205843

structure Conditions where
  (M : ℝ)    -- mass of the piston
  (tau : ℝ)  -- time period τ
  (a : ℝ)    -- constant acceleration
  (c : ℝ)    -- specific heat at constant volume
  (R : ℝ)    -- universal gas constant

theorem average_power_heater (cond : Conditions) : 
  let P := cond.M * cond.a^2 * cond.tau / 2 * (1 + cond.c / cond.R)
  P = (cond.M * cond.a^2 * cond.tau / 2) * (1 + cond.c / cond.R) :=
by
  sorry

end average_power_heater_l205_205843


namespace minimum_value_of_expression_l205_205930

theorem minimum_value_of_expression {k x1 x2 : ℝ} 
  (h1 : x1 + x2 = -2 * k)
  (h2 : x1 * x2 = k^2 + k + 3) : 
  (x1 - 1)^2 + (x2 - 1)^2 ≥ 8 :=
sorry

end minimum_value_of_expression_l205_205930


namespace perpendicular_lines_m_l205_205421

theorem perpendicular_lines_m (m : ℝ) :
  (∀ (x y : ℝ), x - 2 * y + 5 = 0 → 2 * x + m * y - 6 = 0) →
  m = 1 :=
by
  sorry

end perpendicular_lines_m_l205_205421


namespace minimum_value_of_y_l205_205502

-- Define the function y
noncomputable def y (x : ℝ) := 2 + 4 * x + 1 / x

-- Prove that the minimum value is 6 for x > 0
theorem minimum_value_of_y : ∃ (x : ℝ), x > 0 ∧ (∀ (y : ℝ), (2 + 4 * x + 1 / x) ≤ y) ∧ (2 + 4 * x + 1 / x) = 6 := 
sorry

end minimum_value_of_y_l205_205502


namespace number_of_scoops_l205_205482

/-- Pierre gets 3 scoops of ice cream given the conditions described -/
theorem number_of_scoops (P : ℕ) (cost_per_scoop total_bill : ℝ) (mom_scoops : ℕ)
  (h1 : cost_per_scoop = 2) 
  (h2 : mom_scoops = 4) 
  (h3 : total_bill = 14) 
  (h4 : cost_per_scoop * P + cost_per_scoop * mom_scoops = total_bill) :
  P = 3 :=
by
  sorry

end number_of_scoops_l205_205482


namespace inequality_holds_l205_205461

theorem inequality_holds (a : ℝ) (h : a ≠ 0) : |a + (1/a)| ≥ 2 :=
by
  sorry

end inequality_holds_l205_205461


namespace strictly_decreasing_interval_l205_205475

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem strictly_decreasing_interval :
  ∀ x, (0 < x) ∧ (x < 2) → (deriv f x < 0) := by
sorry

end strictly_decreasing_interval_l205_205475


namespace mean_of_remaining_four_numbers_l205_205584

theorem mean_of_remaining_four_numbers (a b c d : ℝ) 
  (h_mean_five : (a + b + c + d + 120) / 5 = 100) : 
  (a + b + c + d) / 4 = 95 :=
by
  sorry

end mean_of_remaining_four_numbers_l205_205584


namespace part_I_part_I_correct_interval_part_II_min_value_l205_205547

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem part_I : ∀ x : ℝ, (f x > 2) ↔ ( x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_I_correct_interval : ∀ x : ℝ, (f x > 2) → (x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_II_min_value : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ ∀ x : ℝ, f x ≥ y := 
sorry

end part_I_part_I_correct_interval_part_II_min_value_l205_205547


namespace number_of_roots_eq_seven_l205_205159

noncomputable def problem_function (x : ℝ) : ℝ :=
  (21 * x - 11 + (Real.sin x) / 100) * Real.sin (6 * Real.arcsin x) * Real.sqrt ((Real.pi - 6 * x) * (Real.pi + x))

theorem number_of_roots_eq_seven :
  (∃ xs : List ℝ, (∀ x ∈ xs, problem_function x = 0) ∧ (∀ x ∈ xs, -1 ≤ x ∧ x ≤ 1) ∧ xs.length = 7) :=
sorry

end number_of_roots_eq_seven_l205_205159


namespace dubblefud_red_balls_zero_l205_205314

theorem dubblefud_red_balls_zero
  (R B G : ℕ)
  (H1 : 2^R * 4^B * 5^G = 16000)
  (H2 : B = G) : R = 0 :=
sorry

end dubblefud_red_balls_zero_l205_205314


namespace vlad_taller_than_sister_l205_205246

theorem vlad_taller_than_sister : 
  ∀ (vlad_height sister_height : ℝ), 
  vlad_height = 190.5 → sister_height = 86.36 → vlad_height - sister_height = 104.14 :=
by
  intros vlad_height sister_height vlad_height_eq sister_height_eq
  rw [vlad_height_eq, sister_height_eq]
  sorry

end vlad_taller_than_sister_l205_205246


namespace average_is_five_plus_D_over_two_l205_205791

variable (A B C D : ℝ)

def condition1 := 1001 * C - 2004 * A = 4008
def condition2 := 1001 * B + 3005 * A - 1001 * D = 6010

theorem average_is_five_plus_D_over_two (h1 : condition1 A C) (h2 : condition2 A B D) : 
  (A + B + C + D) / 4 = (5 + D) / 2 := 
by
  sorry

end average_is_five_plus_D_over_two_l205_205791


namespace speed_of_B_l205_205600

theorem speed_of_B 
    (initial_distance : ℕ)
    (speed_of_A : ℕ)
    (time : ℕ)
    (distance_covered_by_A : ℕ)
    (distance_covered_by_B : ℕ)
    : initial_distance = 24 → speed_of_A = 5 → time = 2 → distance_covered_by_A = speed_of_A * time → distance_covered_by_B = initial_distance - distance_covered_by_A → distance_covered_by_B / time = 7 :=
by
  sorry

end speed_of_B_l205_205600


namespace max_jogs_l205_205729

theorem max_jogs (x y z : ℕ) (h1 : 3 * x + 2 * y + 8 * z = 60) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  z ≤ 6 := 
sorry

end max_jogs_l205_205729


namespace julia_gold_watch_percentage_l205_205818

def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def total_watches_before_gold : ℕ := silver_watches + bronze_watches
def total_watches_after_gold : ℕ := 88
def gold_watches : ℕ := total_watches_after_gold - total_watches_before_gold
def percentage_gold_watches : ℚ := (gold_watches : ℚ) / (total_watches_after_gold : ℚ) * 100

theorem julia_gold_watch_percentage :
  percentage_gold_watches = 9.09 := by
  sorry

end julia_gold_watch_percentage_l205_205818


namespace identity_eq_a_minus_b_l205_205661

theorem identity_eq_a_minus_b (a b : ℚ) (x : ℚ) (h : ∀ x, x > 0 → 
  (a / (2^x - 2) + b / (2^x + 3) = (5 * 2^x + 4) / ((2^x - 2) * (2^x + 3)))) : 
  a - b = 3 / 5 := 
by 
  sorry

end identity_eq_a_minus_b_l205_205661


namespace sum_of_vars_l205_205335

variables (a b c d k p : ℝ)

theorem sum_of_vars (h1 : a^2 + b^2 + c^2 + d^2 = 390)
                    (h2 : ab + bc + ca + ad + bd + cd = 5)
                    (h3 : ad + bd + cd = k)
                    (h4 : (a * b * c * d)^2 = p) :
                    a + b + c + d = 20 :=
by
  -- placeholder for the proof
  sorry

end sum_of_vars_l205_205335


namespace at_least_eight_composites_l205_205290

theorem at_least_eight_composites (n : ℕ) (h : n > 1000) :
  ∃ (comps : Finset ℕ), 
    comps.card ≥ 8 ∧ 
    (∀ x ∈ comps, ¬Prime x) ∧ 
    (∀ k, k < 12 → n + k ∈ comps ∨ Prime (n + k)) :=
by
  sorry

end at_least_eight_composites_l205_205290


namespace equation_relating_price_and_tax_and_discount_l205_205651

variable (c t d : ℚ)

theorem equation_relating_price_and_tax_and_discount
  (h1 : 1.30 * c * ((100 + t) / 100) * ((100 - d) / 100) = 351) :
    1.30 * c * (100 + t) * (100 - d) = 3510000 := by
  sorry

end equation_relating_price_and_tax_and_discount_l205_205651


namespace odd_and_increasing_l205_205657

-- Define the function f(x) = e^x - e^{-x}
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- We want to prove that this function is both odd and increasing.
theorem odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
sorry

end odd_and_increasing_l205_205657


namespace question1_question2_l205_205186

noncomputable def f1 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
noncomputable def f2 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2 - 2*x

theorem question1 (a : ℝ) : 
  (∀ x : ℝ, f1 a x = 0 → ∀ y : ℝ, f1 a y = 0 → x = y) ↔ (a = 0 ∨ a < -4 / Real.exp 2) :=
sorry -- Proof of theorem 1

theorem question2 (a m n x0 : ℝ) (h : a ≠ 0) :
  (f2 a x0 = f2 a ((x0 + m) / 2) * (x0 - m) + n ∧ x0 ≠ m) → False :=
sorry -- Proof of theorem 2

end question1_question2_l205_205186


namespace determine_d_value_l205_205244

noncomputable def Q (d : ℚ) (x : ℚ) : ℚ := x^3 + 3 * x^2 + d * x + 8

theorem determine_d_value (d : ℚ) : x - 3 ∣ Q d x → d = -62 / 3 := by
  sorry

end determine_d_value_l205_205244


namespace base_conversion_subtraction_l205_205728

namespace BaseConversion

def base9_to_base10 (n : ℕ) : ℕ :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def base6_to_base10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 6 * 6^0

theorem base_conversion_subtraction : (base9_to_base10 324) - (base6_to_base10 156) = 193 := by
  sorry

end BaseConversion

end base_conversion_subtraction_l205_205728


namespace outer_term_in_proportion_l205_205835

theorem outer_term_in_proportion (a b x : ℝ) (h_ab : a * b = 1) (h_x : x = 0.2) : b = 5 :=
by
  sorry

end outer_term_in_proportion_l205_205835


namespace problem_statement_l205_205126

-- Conditions
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a ^ x > 0
def q (x : ℝ) : Prop := x > 0 ∧ x ≠ 1 ∧ (Real.log 2 / Real.log x + Real.log x / Real.log 2 ≥ 2)

-- Theorem statement
theorem problem_statement (a x : ℝ) : ¬p a ∨ ¬q x :=
by sorry

end problem_statement_l205_205126


namespace solve_for_a_l205_205020

theorem solve_for_a (a x : ℝ) (h₁ : 2 * x - 3 = 5 * x - 2 * a) (h₂ : x = 1) : a = 3 :=
by
  sorry

end solve_for_a_l205_205020


namespace range_of_m_l205_205557

noncomputable def point := (ℝ × ℝ)
noncomputable def P : point := (-1, 1)
noncomputable def Q : point := (2, 2)
noncomputable def M : point := (0, -1)
noncomputable def line_eq (m : ℝ) := ∀ p : point, p.1 + m * p.2 + m = 0

theorem range_of_m (m : ℝ) (l : line_eq m) : -3 < m ∧ m < -2/3 := 
by
  sorry

end range_of_m_l205_205557


namespace bacteria_mass_at_4pm_l205_205995

theorem bacteria_mass_at_4pm 
  (r s t u v w : ℝ)
  (x y z : ℝ)
  (h1 : x = 10.0 * (1 + r))
  (h2 : y = 15.0 * (1 + s))
  (h3 : z = 8.0 * (1 + t))
  (h4 : 28.9 = x * (1 + u))
  (h5 : 35.5 = y * (1 + v))
  (h6 : 20.1 = z * (1 + w)) :
  x = 28.9 / (1 + u) ∧ y = 35.5 / (1 + v) ∧ z = 20.1 / (1 + w) :=
by
  sorry

end bacteria_mass_at_4pm_l205_205995


namespace find_value_a_prove_inequality_l205_205469

noncomputable def arithmetic_sequence (a : ℕ) (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 2 → S n * S n = 3 * n ^ 2 * a_n n + S (n - 1) * S (n - 1) ∧ a_n n ≠ 0

theorem find_value_a {S : ℕ → ℕ} {a_n : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) → a = 3 :=
sorry

noncomputable def sequence_bn (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  ∀ n : ℕ, b_n n = 1 / ((a_n n - 1) * (a_n n + 2))

theorem prove_inequality {S : ℕ → ℕ} {a_n : ℕ → ℕ} {b_n : ℕ → ℕ} {T : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) →
  (sequence_bn a_n b_n) →
  ∀ n : ℕ, T n < 1 / 6 :=
sorry

end find_value_a_prove_inequality_l205_205469


namespace find_m_value_l205_205587

theorem find_m_value (m : ℝ) (h : (m - 4)^2 + 1^2 + 2^2 = 30) : m = 9 ∨ m = -1 :=
by {
  sorry
}

end find_m_value_l205_205587


namespace telephone_charges_equal_l205_205926

theorem telephone_charges_equal (m : ℝ) :
  (9 + 0.25 * m = 12 + 0.20 * m) → m = 60 :=
by
  intro h
  sorry

end telephone_charges_equal_l205_205926


namespace height_of_triangle_is_5_l205_205430

def base : ℝ := 4
def area : ℝ := 10

theorem height_of_triangle_is_5 :
  ∃ (height : ℝ), (base * height) / 2 = area ∧ height = 5 :=
by
  sorry

end height_of_triangle_is_5_l205_205430


namespace value_of_a_l205_205847

theorem value_of_a (a : ℝ) (H1 : A = a) (H2 : B = 1) (H3 : C = a - 3) (H4 : C + B = 0) : a = 2 := by
  sorry

end value_of_a_l205_205847


namespace multiples_of_5_with_units_digit_0_l205_205162

theorem multiples_of_5_with_units_digit_0 (h1 : ∀ n : ℕ, n % 5 = 0 → (n % 10 = 0 ∨ n % 10 = 5))
  (h2 : ∀ m : ℕ, m < 200 → m % 5 = 0) :
  ∃ k : ℕ, k = 19 ∧ (∀ x : ℕ, (x < 200) ∧ (x % 5 = 0) → (x % 10 = 0) → k = (k - 1) + 1) := sorry

end multiples_of_5_with_units_digit_0_l205_205162


namespace allan_initial_balloons_l205_205078

theorem allan_initial_balloons (jake_balloons allan_bought_more allan_total_balloons : ℕ) 
  (h1 : jake_balloons = 4)
  (h2 : allan_bought_more = 3)
  (h3 : allan_total_balloons = 8) :
  ∃ (allan_initial_balloons : ℕ), allan_total_balloons = allan_initial_balloons + allan_bought_more ∧ allan_initial_balloons = 5 := 
by
  sorry

end allan_initial_balloons_l205_205078


namespace sufficient_not_necessary_for_ellipse_l205_205851

-- Define the conditions
def positive_denominator_m (m : ℝ) : Prop := m > 0
def positive_denominator_2m_minus_1 (m : ℝ) : Prop := 2 * m - 1 > 0
def denominators_not_equal (m : ℝ) : Prop := m ≠ 1

-- Define the question
def is_ellipse_condition (m : ℝ) : Prop := m > 1

-- The main theorem
theorem sufficient_not_necessary_for_ellipse (m : ℝ) :
  positive_denominator_m m ∧ positive_denominator_2m_minus_1 m ∧ denominators_not_equal m → is_ellipse_condition m :=
by
  -- Proof omitted
  sorry

end sufficient_not_necessary_for_ellipse_l205_205851


namespace find_a_l205_205035

theorem find_a : 
  ∃ a : ℝ, (a > 0) ∧ (1 / Real.logb 5 a + 1 / Real.logb 6 a + 1 / Real.logb 7 a = 1) ∧ a = 210 :=
by
  sorry

end find_a_l205_205035


namespace shaded_quadrilateral_area_l205_205981

noncomputable def area_of_shaded_quadrilateral : ℝ :=
  let side_lens : List ℝ := [3, 5, 7, 9]
  let total_base: ℝ := side_lens.sum
  let largest_square_height: ℝ := 9
  let height_base_ratio := largest_square_height / total_base
  let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
  let a := heights.get! 0
  let b := heights.get! heights.length - 1
  (largest_square_height * (a + b)) / 2

theorem shaded_quadrilateral_area :
    let side_lens := [3, 5, 7, 9]
    let total_base := side_lens.sum
    let largest_square_height := 9
    let height_base_ratio := largest_square_height / total_base
    let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
    let a := heights.get! 0
    let b := heights.get! heights.length - 1
    (largest_square_height * (a + b)) / 2 = 30.375 :=
by 
  sorry

end shaded_quadrilateral_area_l205_205981


namespace coloring_satisfies_conditions_l205_205694

-- Definitions of point colors
inductive Color
| Red
| White
| Black

def color_point (x y : ℤ) : Color :=
  if (x + y) % 2 = 1 then Color.Red
  else if (x % 2 = 1 ∧ y % 2 = 0) then Color.White
  else Color.Black

-- Problem statement
theorem coloring_satisfies_conditions :
  (∀ y : ℤ, ∃ x1 x2 x3 : ℤ, 
    color_point x1 y = Color.Red ∧ 
    color_point x2 y = Color.White ∧
    color_point x3 y = Color.Black)
  ∧ 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    color_point x1 y1 = Color.White →
    color_point x2 y2 = Color.Red →
    color_point x3 y3 = Color.Black →
    ∃ x4 y4, 
      color_point x4 y4 = Color.Red ∧ 
      x4 = x3 + (x1 - x2) ∧ 
      y4 = y3 + (y1 - y2)) :=
by
  sorry

end coloring_satisfies_conditions_l205_205694


namespace base_conversion_l205_205237

theorem base_conversion (C D : ℕ) (hC : 0 ≤ C) (hC_lt : C < 8) (hD : 0 ≤ D) (hD_lt : D < 5) :
  (8 * C + D = 5 * D + C) → (8 * C + D = 0) :=
by 
  intro h
  sorry

end base_conversion_l205_205237


namespace exists_even_among_pythagorean_triplet_l205_205101

theorem exists_even_among_pythagorean_triplet (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ x, (x = a ∨ x = b ∨ x = c) ∧ x % 2 = 0 :=
sorry

end exists_even_among_pythagorean_triplet_l205_205101


namespace space_left_over_l205_205470

theorem space_left_over (D B : ℕ) (wall_length desk_length bookcase_length : ℝ) (h_wall : wall_length = 15)
  (h_desk : desk_length = 2) (h_bookcase : bookcase_length = 1.5) (h_eq : D = B)
  (h_max : 2 * D + 1.5 * B ≤ wall_length) :
  ∃ w : ℝ, w = wall_length - (D * desk_length + B * bookcase_length) ∧ w = 1 :=
by
  sorry

end space_left_over_l205_205470


namespace parkway_elementary_students_l205_205419

/-- The total number of students in the fifth grade at Parkway Elementary School is 420,
given the following conditions:
1. There are 312 boys.
2. 250 students are playing soccer.
3. 78% of the students that play soccer are boys.
4. There are 53 girl students not playing soccer. -/
theorem parkway_elementary_students (boys : ℕ) (playing_soccer : ℕ) (percent_boys_playing : ℝ) (girls_not_playing_soccer : ℕ)
  (h1 : boys = 312)
  (h2 : playing_soccer = 250)
  (h3 : percent_boys_playing = 0.78)
  (h4 : girls_not_playing_soccer = 53) :
  ∃ total_students : ℕ, total_students = 420 :=
by
  sorry

end parkway_elementary_students_l205_205419


namespace value_of_x_l205_205095

theorem value_of_x (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := 
by 
  sorry

end value_of_x_l205_205095


namespace exists_pythagorean_triple_rational_k_l205_205394

theorem exists_pythagorean_triple_rational_k (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ), (a^2 + b^2 = c^2) ∧ ((a + c : ℚ) / b = k) := by
  sorry

end exists_pythagorean_triple_rational_k_l205_205394


namespace some_employee_not_team_leader_l205_205539

variables (Employee : Type) (isTeamLeader : Employee → Prop) (meetsDeadline : Employee → Prop)

-- Conditions
axiom some_employee_not_meets_deadlines : ∃ e : Employee, ¬ meetsDeadline e
axiom all_team_leaders_meet_deadlines : ∀ e : Employee, isTeamLeader e → meetsDeadline e

-- Theorem to prove
theorem some_employee_not_team_leader : ∃ e : Employee, ¬ isTeamLeader e :=
sorry

end some_employee_not_team_leader_l205_205539


namespace side_length_of_inscribed_square_l205_205147

theorem side_length_of_inscribed_square
  (S1 S2 S3 : ℝ)
  (hS1 : S1 = 1) (hS2 : S2 = 3) (hS3 : S3 = 1) :
  ∃ (x : ℝ), S1 = 1 ∧ S2 = 3 ∧ S3 = 1 ∧ x = 2 := 
by
  sorry

end side_length_of_inscribed_square_l205_205147


namespace find_x_l205_205481

variable (x : ℝ)
variable (y : ℝ := x * 3.5)
variable (z : ℝ := y / 0.00002)

theorem find_x (h : z = 840) : x = 0.0048 :=
sorry

end find_x_l205_205481


namespace find_specified_time_l205_205375

theorem find_specified_time (distance : ℕ) (slow_time fast_time : ℕ → ℕ) (fast_is_double : ∀ x, fast_time x = 2 * slow_time x)
  (distance_value : distance = 900) (slow_time_eq : ∀ x, slow_time x = x + 1) (fast_time_eq : ∀ x, fast_time x = x - 3) :
  2 * (distance / (slow_time x)) = distance / (fast_time x) :=
by
  intros
  rw [distance_value, slow_time_eq, fast_time_eq]
  sorry

end find_specified_time_l205_205375


namespace abs_difference_l205_205055

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end abs_difference_l205_205055


namespace milo_running_distance_l205_205364

theorem milo_running_distance : 
  ∀ (cory_speed milo_skate_speed milo_run_speed time miles_run : ℕ),
  cory_speed = 12 →
  milo_skate_speed = cory_speed / 2 →
  milo_run_speed = milo_skate_speed / 2 →
  time = 2 →
  miles_run = milo_run_speed * time →
  miles_run = 6 :=
by 
  intros cory_speed milo_skate_speed milo_run_speed time miles_run hcory hmilo_skate hmilo_run htime hrun 
  -- Proof steps would go here
  sorry

end milo_running_distance_l205_205364


namespace isosceles_obtuse_triangle_angles_l205_205977

def isosceles (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A
def obtuse (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

noncomputable def sixty_percent_larger_angle : ℝ := 1.6 * 90

theorem isosceles_obtuse_triangle_angles 
  (A B C : ℝ) 
  (h_iso : isosceles A B C) 
  (h_obt : obtuse A B C) 
  (h_large_angle : A = sixty_percent_larger_angle ∨ B = sixty_percent_larger_angle ∨ C = sixty_percent_larger_angle) 
  (h_sum : A + B + C = 180) : 
  (A = 18 ∨ B = 18 ∨ C = 18) := 
sorry

end isosceles_obtuse_triangle_angles_l205_205977


namespace point_divides_segment_in_ratio_l205_205252

theorem point_divides_segment_in_ratio (A B C C1 A1 P : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
  [AddCommGroup C1] [AddCommGroup A1] [AddCommGroup P]
  (h1 : AP / PA1 = 3 / 2)
  (h2 : CP / PC1 = 2 / 1) :
  AC1 / C1B = 2 / 3 :=
sorry

end point_divides_segment_in_ratio_l205_205252


namespace slope_angle_of_perpendicular_line_l205_205673

theorem slope_angle_of_perpendicular_line (l : ℝ → ℝ) (h_perp : ∀ x y : ℝ, l x = y ↔ x - y - 1 = 0) : ∃ α : ℝ, α = 135 :=
by
  sorry

end slope_angle_of_perpendicular_line_l205_205673


namespace ellipse_polar_inverse_sum_l205_205450

noncomputable def ellipse_equation (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

theorem ellipse_polar_inverse_sum (A B : ℝ × ℝ)
  (hA : ∃ α₁, ellipse_equation α₁ = A)
  (hB : ∃ α₂, ellipse_equation α₂ = B)
  (hPerp : A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / (A.1 ^ 2 + A.2 ^ 2) + 1 / (B.1 ^ 2 + B.2 ^ 2)) = 7 / 12 :=
by
  sorry

end ellipse_polar_inverse_sum_l205_205450


namespace two_color_K6_contains_monochromatic_triangle_l205_205628

theorem two_color_K6_contains_monochromatic_triangle (V : Type) [Fintype V] [DecidableEq V]
  (hV : Fintype.card V = 6)
  (color : V → V → Fin 2) :
  ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (color a b = color b c ∧ color b c = color c a) := by
  sorry

end two_color_K6_contains_monochromatic_triangle_l205_205628


namespace calculate_expression_l205_205127

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end calculate_expression_l205_205127


namespace coeff_sum_eq_minus_243_l205_205118

theorem coeff_sum_eq_minus_243 (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x - 2 * y) ^ 5 = a * (x + 2 * y) ^ 5 + a₁ * (x + 2 * y)^4 * y + a₂ * (x + 2 * y)^3 * y^2 
             + a₃ * (x + 2 * y)^2 * y^3 + a₄ * (x + 2 * y) * y^4 + a₅ * y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 :=
by
  intros h
  sorry

end coeff_sum_eq_minus_243_l205_205118


namespace slope_of_tangent_line_at_zero_l205_205034

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 :=
by
  sorry 

end slope_of_tangent_line_at_zero_l205_205034


namespace condition_sufficient_not_necessary_l205_205670

theorem condition_sufficient_not_necessary (x : ℝ) : (0 < x ∧ x < 5) → (|x - 2| < 3) ∧ (¬ ((|x - 2| < 3) → (0 < x ∧ x < 5))) :=
by
  sorry

end condition_sufficient_not_necessary_l205_205670


namespace initial_value_l205_205898

theorem initial_value (x k : ℤ) (h : x + 294 = k * 456) : x = 162 :=
sorry

end initial_value_l205_205898


namespace fraction_of_top10_lists_l205_205636

theorem fraction_of_top10_lists (total_members : ℕ) (min_lists : ℝ) (H1 : total_members = 795) (H2 : min_lists = 198.75) :
  (min_lists / total_members) = 1 / 4 :=
by
  -- The proof is omitted as requested
  sorry

end fraction_of_top10_lists_l205_205636


namespace incorrect_value_of_observation_l205_205061

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end incorrect_value_of_observation_l205_205061


namespace find_y_from_equation_l205_205404

theorem find_y_from_equation :
  ∀ y : ℕ, (12 ^ 3 * 6 ^ 4) / y = 5184 → y = 432 :=
by
  sorry

end find_y_from_equation_l205_205404


namespace maximize_revenue_l205_205891

theorem maximize_revenue (p : ℝ) (h₁ : p ≤ 30) (h₂ : p = 18.75) : 
  ∃(R : ℝ), R = p * (150 - 4 * p) :=
by
  sorry

end maximize_revenue_l205_205891


namespace cubic_expression_l205_205010

theorem cubic_expression {x : ℝ} (h : x + (1/x) = 5) : x^3 + (1/x^3) = 110 := 
by
  sorry

end cubic_expression_l205_205010


namespace range_of_m_l205_205681

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, 0 < x ∧ mx^2 + 2 * x + m > 0) →
  m ≤ -1 := by
  sorry

end range_of_m_l205_205681


namespace walnut_trees_in_park_l205_205895

def num_current_walnut_trees (num_plant : ℕ) (num_total : ℕ) : ℕ :=
  num_total - num_plant

theorem walnut_trees_in_park :
  num_current_walnut_trees 6 10 = 4 :=
by
  -- By the definition of num_current_walnut_trees
  -- We have 10 (total) - 6 (to be planted) = 4 (current)
  sorry

end walnut_trees_in_park_l205_205895


namespace cars_in_section_H_l205_205182

theorem cars_in_section_H
  (rows_G : ℕ) (cars_per_row_G : ℕ) (rows_H : ℕ)
  (cars_per_minute : ℕ) (minutes_spent : ℕ)  
  (total_cars_walked_past : ℕ) :
  rows_G = 15 →
  cars_per_row_G = 10 →
  rows_H = 20 →
  cars_per_minute = 11 →
  minutes_spent = 30 →
  total_cars_walked_past = (rows_G * cars_per_row_G) + ((cars_per_minute * minutes_spent) - (rows_G * cars_per_row_G)) →
  (total_cars_walked_past - (rows_G * cars_per_row_G)) / rows_H = 9 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end cars_in_section_H_l205_205182


namespace boys_girls_rel_l205_205426

theorem boys_girls_rel (b g : ℕ) (h : g = 7 + 2 * (b - 1)) : b = (g - 5) / 2 := 
by sorry

end boys_girls_rel_l205_205426


namespace train_length_l205_205566

theorem train_length (speed : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_km : ℝ) (distance_m : ℝ) 
  (h1 : speed = 60) 
  (h2 : time_seconds = 42) 
  (h3 : time_hours = time_seconds / 3600)
  (h4 : distance_km = speed * time_hours) 
  (h5 : distance_m = distance_km * 1000) :
  distance_m = 700 :=
by 
  sorry

end train_length_l205_205566


namespace box_volume_l205_205268

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end box_volume_l205_205268


namespace expression_equality_l205_205580

theorem expression_equality :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := 
  sorry

end expression_equality_l205_205580


namespace hyperbola_equation_l205_205892

-- Define the conditions
def hyperbola_eq := ∀ (x y a b : ℝ), a > 0 ∧ b > 0 → x^2 / a^2 - y^2 / b^2 = 1
def parabola_eq := ∀ (x y : ℝ), y^2 = (2 / 5) * x
def intersection_point_M := ∃ (x : ℝ), ∀ (y : ℝ), y = 1 → y^2 = (2 / 5) * x
def line_intersect_N := ∀ (F₁ M N : ℝ × ℝ), 
  (N.1 = -1 / 10) ∧ (F₁.1 ≠ M.1) ∧ (N.2 = 0)

-- State the proof problem
theorem hyperbola_equation 
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyp_eq : hyperbola_eq)
  (par_eq : parabola_eq)
  (int_pt_M : intersection_point_M)
  (line_int_N : line_intersect_N) :
  ∀ (x y : ℝ), x^2 / 5 - y^2 / 4 = 1 :=
by sorry

end hyperbola_equation_l205_205892


namespace value_of_x_plus_y_l205_205949

theorem value_of_x_plus_y (x y : ℝ) 
  (h1 : 2 * x - y = -1) 
  (h2 : x + 4 * y = 22) : 
  x + y = 7 :=
sorry

end value_of_x_plus_y_l205_205949


namespace find_fraction_l205_205263

theorem find_fraction
  (N : ℝ)
  (hN : N = 30)
  (h : 0.5 * N = (x / y) * N + 10):
  x / y = 1 / 6 :=
by
  sorry

end find_fraction_l205_205263


namespace number_of_wheels_on_each_bicycle_l205_205258

theorem number_of_wheels_on_each_bicycle 
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (wheels_per_tricycle : ℕ)
  (total_wheels : ℕ)
  (h_bicycles : num_bicycles = 24)
  (h_tricycles : num_tricycles = 14)
  (h_wheels_tricycle : wheels_per_tricycle = 3)
  (h_total_wheels : total_wheels = 90) :
  2 * num_bicycles + 3 * num_tricycles = 90 → 
  num_bicycles = 24 → 
  num_tricycles = 14 → 
  wheels_per_tricycle = 3 → 
  total_wheels = 90 → 
  ∃ b : ℕ, b = 2 :=
by
  sorry

end number_of_wheels_on_each_bicycle_l205_205258


namespace expression_evaluation_l205_205844

theorem expression_evaluation (m : ℝ) (h : m = Real.sqrt 2023 + 2) : m^2 - 4 * m + 5 = 2024 :=
by sorry

end expression_evaluation_l205_205844


namespace sin_range_l205_205750

theorem sin_range (p : Prop) (q : Prop) :
  (¬ ∃ x : ℝ, Real.sin x = 3/2) → (∀ x : ℝ, x^2 - 4 * x + 5 > 0) → (¬p ∧ q) :=
by
  sorry

end sin_range_l205_205750


namespace total_money_shared_l205_205590

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end total_money_shared_l205_205590


namespace bottle_caps_per_box_l205_205774

theorem bottle_caps_per_box (total_bottle_caps boxes : ℕ) (hb : total_bottle_caps = 316) (bn : boxes = 79) :
  total_bottle_caps / boxes = 4 :=
by
  sorry

end bottle_caps_per_box_l205_205774


namespace prove_correct_options_l205_205100

theorem prove_correct_options (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 2) :
  (min (((1 : ℝ) / x) + (1 / y)) = 2) ∧
  (max (x * y) = 1) ∧
  (min (x^2 + y^2) = 2) ∧
  (max (x * (y + 1)) = (9 / 4)) :=
by
  sorry

end prove_correct_options_l205_205100


namespace longest_boat_length_l205_205794

variable (saved money : ℕ) (license_fee docking_multiplier boat_cost : ℕ)

theorem longest_boat_length (h1 : saved = 20000) 
                           (h2 : license_fee = 500) 
                           (h3 : docking_multiplier = 3)
                           (h4 : boat_cost = 1500) : 
                           (saved - license_fee - docking_multiplier * license_fee) / boat_cost = 12 := 
by 
  sorry

end longest_boat_length_l205_205794


namespace factorization_correct_l205_205860

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end factorization_correct_l205_205860


namespace prairie_total_area_l205_205765

theorem prairie_total_area (acres_dust_storm : ℕ) (acres_untouched : ℕ) (h₁ : acres_dust_storm = 64535) (h₂ : acres_untouched = 522) : acres_dust_storm + acres_untouched = 65057 :=
by
  sorry

end prairie_total_area_l205_205765


namespace final_temperature_l205_205870

theorem final_temperature (initial_temp cost_per_tree spent amount temperature_drop : ℝ) 
  (h1 : initial_temp = 80) 
  (h2 : cost_per_tree = 6)
  (h3 : spent = 108) 
  (h4 : temperature_drop = 0.1) 
  (trees_planted : ℝ) 
  (h5 : trees_planted = spent / cost_per_tree) 
  (temp_reduction : ℝ) 
  (h6 : temp_reduction = trees_planted * temperature_drop) 
  (final_temp : ℝ) 
  (h7 : final_temp = initial_temp - temp_reduction) : 
  final_temp = 78.2 := 
by
  sorry

end final_temperature_l205_205870


namespace translate_line_down_l205_205837

theorem translate_line_down (k : ℝ) (b : ℝ) : 
  (∀ x : ℝ, b = 0 → (y = k * x - 3) = (y = k * x - 3)) :=
by
  sorry

end translate_line_down_l205_205837


namespace negation_of_proposition_l205_205724

theorem negation_of_proposition :
  (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) → (∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) :=
sorry

end negation_of_proposition_l205_205724


namespace pear_distribution_problem_l205_205446

-- Defining the given conditions as hypotheses
variables (G P : ℕ)

-- The first condition: P = G + 1
def condition1 : Prop := P = G + 1

-- The second condition: P = 2G - 2
def condition2 : Prop := P = 2 * G - 2

-- The main theorem to prove
theorem pear_distribution_problem (h1 : condition1 G P) (h2 : condition2 G P) :
  G = 3 ∧ P = 4 :=
by
  sorry

end pear_distribution_problem_l205_205446


namespace jims_final_paycheck_l205_205399

noncomputable def final_paycheck (g r t h m b btr : ℝ) := 
  let retirement := g * r
  let gym := m / 2
  let net_before_bonus := g - retirement - t - h - gym
  let after_tax_bonus := b * (1 - btr)
  net_before_bonus + after_tax_bonus

theorem jims_final_paycheck :
  final_paycheck 1120 0.25 100 200 50 500 0.30 = 865 :=
by
  sorry

end jims_final_paycheck_l205_205399


namespace number_four_units_away_from_neg_five_l205_205747

theorem number_four_units_away_from_neg_five (x : ℝ) : 
    abs (x + 5) = 4 ↔ x = -9 ∨ x = -1 :=
by 
  sorry

end number_four_units_away_from_neg_five_l205_205747


namespace smallest_solution_l205_205135

noncomputable def equation (x : ℝ) := x^4 - 40 * x^2 + 400

theorem smallest_solution : ∃ x : ℝ, equation x = 0 ∧ ∀ y : ℝ, equation y = 0 → -2 * Real.sqrt 5 ≤ y :=
by
  sorry

end smallest_solution_l205_205135


namespace bounded_sequence_exists_l205_205206

noncomputable def positive_sequence := ℕ → ℝ

variables {a : positive_sequence}

axiom positive_sequence_pos (n : ℕ) : 0 < a n

axiom sequence_condition (k n m l : ℕ) (h : k + n = m + l) : 
  (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)

theorem bounded_sequence_exists 
  (a : positive_sequence) 
  (h_pos : ∀ n, 0 < a n)
  (h_cond : ∀ (k n m l : ℕ), k + n = m + l → 
              (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ (b c : ℝ), (0 < b) ∧ (0 < c) ∧ (∀ n, b ≤ a n ∧ a n ≤ c) :=
sorry

end bounded_sequence_exists_l205_205206


namespace last_digit_sum_l205_205427

theorem last_digit_sum :
  (2^2 % 10 + 20^20 % 10 + 200^200 % 10 + 2006^2006 % 10) % 10 = 0 := 
by
  sorry

end last_digit_sum_l205_205427


namespace magpies_gather_7_trees_magpies_not_gather_6_trees_l205_205857

-- Define the problem conditions.
def trees (n : ℕ) := (∀ (i : ℕ), i < n → ∃ (m : ℕ), m = i * 10)

-- Define the movement condition for magpies.
def magpie_move (n : ℕ) (d : ℕ) :=
  (∀ (i j : ℕ), i < n ∧ j < n ∧ i ≠ j → ∃ (k : ℕ), k = d ∧ ((i + d < n ∧ j - d < n) ∨ (i - d < n ∧ j + d < n)))

-- Prove that all magpies can gather on one tree for 7 trees.
theorem magpies_gather_7_trees : 
  ∃ (i : ℕ), i < 7 ∧ trees 7 ∧ magpie_move 7 (i * 10) → True :=
by
  -- proof steps here, which are not necessary for the task
  sorry

-- Prove that all magpies cannot gather on one tree for 6 trees.
theorem magpies_not_gather_6_trees : 
  ∀ (i : ℕ), i < 6 ∧ trees 6 ∧ magpie_move 6 (i * 10) → False :=
by
  -- proof steps here, which are not necessary for the task
  sorry

end magpies_gather_7_trees_magpies_not_gather_6_trees_l205_205857


namespace esperanzas_tax_ratio_l205_205007

theorem esperanzas_tax_ratio :
  let rent := 600
  let food_expenses := (3 / 5) * rent
  let mortgage_bill := 3 * food_expenses
  let savings := 2000
  let gross_salary := 4840
  let total_expenses := rent + food_expenses + mortgage_bill + savings
  let taxes := gross_salary - total_expenses
  (taxes / savings) = (2 / 5) := by
  sorry

end esperanzas_tax_ratio_l205_205007


namespace max_cookies_l205_205008

-- Definitions for the conditions
def John_money : ℕ := 2475
def cookie_cost : ℕ := 225

-- Statement of the problem
theorem max_cookies (x : ℕ) : cookie_cost * x ≤ John_money → x ≤ 11 :=
sorry

end max_cookies_l205_205008


namespace bug_paths_from_A_to_B_l205_205989

-- Define the positions A and B and intermediate red and blue points in the lattice
inductive Position
| A
| B
| red1
| red2
| blue1
| blue2

open Position

-- Define the possible directed paths in the lattice
def paths : List (Position × Position) :=
[(A, red1), (A, red2), 
 (red1, blue1), (red1, blue2), 
 (red2, blue1), (red2, blue2), 
 (blue1, B), (blue1, B), (blue1, B), 
 (blue2, B), (blue2, B), (blue2, B)]

-- Define a function that calculates the number of unique paths from A to B without repeating any path
def count_paths : ℕ := sorry

-- The mathematical problem statement
theorem bug_paths_from_A_to_B : count_paths = 24 := sorry

end bug_paths_from_A_to_B_l205_205989


namespace compute_product_sum_l205_205313

theorem compute_product_sum (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  (a * b * c) * ((1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c) = 47 :=
by
  sorry

end compute_product_sum_l205_205313


namespace irreducible_fraction_unique_l205_205367

theorem irreducible_fraction_unique :
  ∃ (a b : ℕ), a = 5 ∧ b = 2 ∧ gcd a b = 1 ∧ (∃ n : ℕ, 10^n = a * b) :=
by
  sorry

end irreducible_fraction_unique_l205_205367


namespace password_decryption_probability_l205_205279

theorem password_decryption_probability :
  let A := (1:ℚ)/5
  let B := (1:ℚ)/3
  let C := (1:ℚ)/4
  let P_decrypt := 1 - (1 - A) * (1 - B) * (1 - C)
  P_decrypt = 3/5 := 
  by
    -- Calculations and logic will be provided here
    sorry

end password_decryption_probability_l205_205279


namespace book_organizing_activity_l205_205746

theorem book_organizing_activity (x : ℕ) (h₁ : x > 0):
  (80 : ℝ) / (x + 5 : ℝ) = (70 : ℝ) / (x : ℝ) :=
sorry

end book_organizing_activity_l205_205746


namespace december_revenue_times_average_l205_205092

def revenue_in_december_is_multiple_of_average_revenue (R_N R_J R_D : ℝ) : Prop :=
  R_N = (3/5) * R_D ∧    -- Condition: November's revenue is 3/5 of December's revenue
  R_J = (1/3) * R_N ∧    -- Condition: January's revenue is 1/3 of November's revenue
  R_D = 2.5 * ((R_N + R_J) / 2)   -- Question: December's revenue is 2.5 times the average of November's and January's revenue

theorem december_revenue_times_average (R_N R_J R_D : ℝ) :
  revenue_in_december_is_multiple_of_average_revenue R_N R_J R_D :=
by
  -- adding sorry to skip the proof
  sorry

end december_revenue_times_average_l205_205092


namespace find_percentage_l205_205935

theorem find_percentage (P : ℕ) (h1 : 0.20 * 650 = 130) (h2 : P * 800 / 100 = 320) : P = 40 := 
by { 
  sorry 
}

end find_percentage_l205_205935


namespace profit_percentage_example_l205_205647

noncomputable def selling_price : ℝ := 100
noncomputable def cost_price (sp : ℝ) : ℝ := 0.75 * sp
noncomputable def profit (sp cp : ℝ) : ℝ := sp - cp
noncomputable def profit_percentage (profit cp : ℝ) : ℝ := (profit / cp) * 100

theorem profit_percentage_example :
  profit_percentage (profit selling_price (cost_price selling_price)) (cost_price selling_price) = 33.33 :=
by
  -- Proof will go here
  sorry

end profit_percentage_example_l205_205647


namespace primes_diff_power_of_two_divisible_by_three_l205_205586

theorem primes_diff_power_of_two_divisible_by_three
  (p q : ℕ) (m n : ℕ)
  (hp : Prime p) (hq : Prime q) (hp_gt : p > 3) (hq_gt : q > 3)
  (diff : q - p = 2^n ∨ p - q = 2^n) :
  3 ∣ (p^(2*m+1) + q^(2*m+1)) := by
  sorry

end primes_diff_power_of_two_divisible_by_three_l205_205586


namespace gcd_2720_1530_l205_205067

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end gcd_2720_1530_l205_205067


namespace number_of_outfits_l205_205845

-- Define the counts of each item
def redShirts : Nat := 6
def greenShirts : Nat := 4
def pants : Nat := 7
def greenHats : Nat := 10
def redHats : Nat := 9

-- Total number of outfits satisfying the conditions
theorem number_of_outfits :
  (redShirts * greenHats * pants) + (greenShirts * redHats * pants) = 672 :=
by
  sorry

end number_of_outfits_l205_205845


namespace range_of_a_l205_205199

noncomputable def tangent_slopes (a x0 : ℝ) : ℝ × ℝ :=
  let k1 := (a * x0 + a - 1) * Real.exp x0
  let k2 := (x0 - 2) * Real.exp (-x0)
  (k1, k2)

theorem range_of_a (a x0 : ℝ) (h : x0 ∈ Set.Icc 0 (3 / 2))
  (h_perpendicular : (tangent_slopes a x0).1 * (tangent_slopes a x0).2 = -1)
  : 1 ≤ a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l205_205199


namespace solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l205_205684

def f (x : ℝ) : ℝ := |3 * x + 1| - |x - 4|

theorem solve_f_lt_zero :
  { x : ℝ | f x < 0 } = { x : ℝ | -5 / 2 < x ∧ x < 3 / 4 } := 
sorry

theorem solve_f_plus_4_abs_x_minus_4_gt_m (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) → m < 15 :=
sorry

end solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l205_205684


namespace sum_eight_smallest_multiples_of_12_l205_205295

theorem sum_eight_smallest_multiples_of_12 :
  let series := (List.range 8).map (λ k => 12 * (k + 1))
  series.sum = 432 :=
by
  sorry

end sum_eight_smallest_multiples_of_12_l205_205295


namespace train_speed_conversion_l205_205009

theorem train_speed_conversion (s_mps : ℝ) (h : s_mps = 30.002399999999998) : 
  s_mps * 3.6 = 108.01 :=
by
  sorry

end train_speed_conversion_l205_205009


namespace problem_1_problem_2_problem_3_l205_205937

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 8 * x^2 + 16 * x - k
noncomputable def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x
noncomputable def h (x : ℝ) (k : ℝ) : ℝ := g x - f x k

theorem problem_1 (k : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x k ≤ g x) → 45 ≤ k := by
  sorry

theorem problem_2 (k : ℝ) : (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ f x k ≤ g x) → -7 ≤ k := by
  sorry

theorem problem_3 (k : ℝ) : (∀ x1 x2 : ℝ, (-3 ≤ x1 ∧ x1 ≤ 3) ∧ (-3 ≤ x2 ∧ x2 ≤ 3) → f x1 k ≤ g x2) → 141 ≤ k := by
  sorry

end problem_1_problem_2_problem_3_l205_205937


namespace calculate_expression_l205_205972

theorem calculate_expression :
  12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 :=
by
  sorry

end calculate_expression_l205_205972


namespace fraction_sum_l205_205528

theorem fraction_sum :
  (7 : ℚ) / 12 + (3 : ℚ) / 8 = 23 / 24 :=
by
  -- Proof is omitted
  sorry

end fraction_sum_l205_205528


namespace jack_needs_more_money_l205_205134

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l205_205134


namespace rectangle_inscribed_area_l205_205714

variables (b h x : ℝ) 

theorem rectangle_inscribed_area (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hx_lt_h : x < h) :
  ∃ A, A = (b * x * (h - x)) / h :=
sorry

end rectangle_inscribed_area_l205_205714


namespace compute_expression_l205_205893

theorem compute_expression : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3 ^ 2 = 28 := by
  sorry

end compute_expression_l205_205893


namespace part1_part2_l205_205445

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x > -1, (x^2 + 3*x + 6) / (x + 1) ≥ a) ↔ (a ≤ 5) := 
  sorry

-- Part 2
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) : 
  2*a + (1/a) + 4*b + (8/b) ≥ 27 :=
  sorry

end part1_part2_l205_205445


namespace log_expression_evaluation_l205_205953

theorem log_expression_evaluation : 
  (4 * Real.log 2 + 3 * Real.log 5 - Real.log (1/5)) = 4 := 
  sorry

end log_expression_evaluation_l205_205953


namespace susan_total_distance_l205_205360

theorem susan_total_distance (a b : ℕ) (r : ℝ) (h1 : a = 15) (h2 : b = 25) (h3 : r = 3) :
  (r * ((a + b) / 60)) = 2 :=
by
  sorry

end susan_total_distance_l205_205360


namespace same_color_difference_perfect_square_l205_205183

theorem same_color_difference_perfect_square :
  (∃ (f : ℤ → ℕ) (a b : ℤ), f a = f b ∧ a ≠ b ∧ ∃ (k : ℤ), a - b = k * k) :=
sorry

end same_color_difference_perfect_square_l205_205183


namespace find_missing_number_l205_205251

theorem find_missing_number (x : ℝ) (h : 0.00375 * x = 153.75) : x = 41000 :=
sorry

end find_missing_number_l205_205251


namespace union_of_A_and_B_l205_205336

section
variable {A B : Set ℝ}
variable (a b : ℝ)

def setA := {x : ℝ | x^2 - 3 * x + a = 0}
def setB := {x : ℝ | x^2 + b = 0}

theorem union_of_A_and_B:
  setA a ∩ setB b = {2} →
  setA a ∪ setB b = ({-2, 1, 2} : Set ℝ) := by
  sorry
end

end union_of_A_and_B_l205_205336


namespace solve_system_of_equations_l205_205085

theorem solve_system_of_equations :
  ∃ x y : ℤ, (2 * x + 7 * y = -6) ∧ (2 * x - 5 * y = 18) ∧ (x = 4) ∧ (y = -2) := 
by
  -- Proof will go here
  sorry

end solve_system_of_equations_l205_205085


namespace larger_triangle_perimeter_l205_205488

def is_similar (a b c : ℕ) (x y z : ℕ) : Prop :=
  x * c = z * a ∧
  x * c = z * b ∧
  y * c = z * a ∧
  y * c = z * c ∧
  a ≠ b ∧ c ≠ b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∧ a ≠ c

theorem larger_triangle_perimeter (a b c x y z : ℕ) 
  (h1 : is_isosceles a b c) 
  (h2 : is_similar a b c x y z) 
  (h3 : c = 12) 
  (h4 : z = 36)
  (h5 : a = 7) 
  (h6 : b = 7) : 
  x + y + z = 78 :=
sorry

end larger_triangle_perimeter_l205_205488


namespace apples_ratio_l205_205111

theorem apples_ratio (bonnie_apples samuel_extra_apples samuel_left_over samuel_total_pies : ℕ) 
  (h_bonnie : bonnie_apples = 8)
  (h_samuel_extra : samuel_extra_apples = 20)
  (h_samuel_left_over : samuel_left_over = 10)
  (h_pie_ratio : samuel_total_pies = (8 + 20) / 7) :
  (28 - samuel_total_pies - 10) / 28 = 1 / 2 := 
by
  sorry

end apples_ratio_l205_205111


namespace mod_equivalence_l205_205685

theorem mod_equivalence (a b : ℤ) (d : ℕ) (hd : d ≠ 0) 
  (a' b' : ℕ) (ha' : a % d = a') (hb' : b % d = b') : (a ≡ b [ZMOD d]) ↔ a' = b' := 
sorry

end mod_equivalence_l205_205685


namespace strictly_increasing_interval_l205_205766

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb (1/3) (x^2 - 4 * x + 3)

theorem strictly_increasing_interval : ∀ x y : ℝ, x < 1 → y < 1 → x < y → f x < f y :=
by
  sorry

end strictly_increasing_interval_l205_205766


namespace correct_calculation_l205_205361

theorem correct_calculation (a : ℝ) : a^3 / a^2 = a := by
  sorry

end correct_calculation_l205_205361


namespace soda_cost_original_l205_205242

theorem soda_cost_original 
  (x : ℚ) -- note: x in rational numbers to capture fractional cost accurately
  (h1 : 3 * (0.90 * x) = 6) :
  x = 20 / 9 :=
by
  sorry

end soda_cost_original_l205_205242


namespace locus_of_point_P_l205_205522

theorem locus_of_point_P (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (hxM : M = (-2, 0))
  (hxN : N = (2, 0))
  (hxPM : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPM)
  (hxPN : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPN)
  : P.fst ^ 2 + P.snd ^ 2 = 4 ∧ P.fst ≠ 2 ∧ P.fst ≠ -2 :=
by
  -- proof omitted
  sorry

end locus_of_point_P_l205_205522


namespace min_value_of_quadratic_l205_205425

theorem min_value_of_quadratic (x : ℝ) : ∃ z : ℝ, z = 2 * x^2 + 16 * x + 40 ∧ z = 8 :=
by {
  sorry
}

end min_value_of_quadratic_l205_205425


namespace average_speed_l205_205420

theorem average_speed 
  (total_distance : ℝ) (total_time : ℝ) 
  (h_distance : total_distance = 26) (h_time : total_time = 4) :
  (total_distance / total_time) = 6.5 :=
by
  rw [h_distance, h_time]
  norm_num

end average_speed_l205_205420


namespace cara_cats_correct_l205_205485

def martha_cats_rats : ℕ := 3
def martha_cats_birds : ℕ := 7
def martha_cats_animals : ℕ := martha_cats_rats + martha_cats_birds

def cara_cats_animals : ℕ := 5 * martha_cats_animals - 3

theorem cara_cats_correct : cara_cats_animals = 47 :=
by
  -- Proof omitted
  -- Here's where the actual calculation steps would go, but we'll just use sorry for now.
  sorry

end cara_cats_correct_l205_205485


namespace negate_exists_implies_forall_l205_205940

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end negate_exists_implies_forall_l205_205940


namespace diophantine_3x_5y_diophantine_3x_5y_indefinite_l205_205639

theorem diophantine_3x_5y (n : ℕ) (h_n_pos : n > 0) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n) ↔ 
    (∃ k : ℕ, (n = 3 * k ∧ n ≥ 15) ∨ 
              (n = 3 * k + 1 ∧ n ≥ 13) ∨ 
              (n = 3 * k + 2 ∧ n ≥ 11) ∨ 
              (n = 8)) :=
sorry

theorem diophantine_3x_5y_indefinite (n m : ℕ) (h_n_large : n > 40 * m):
  ∃ (N : ℕ), ∀ k ≤ N, ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n + k :=
sorry

end diophantine_3x_5y_diophantine_3x_5y_indefinite_l205_205639


namespace cricket_team_members_l205_205133

theorem cricket_team_members (n : ℕ) (captain_age wicket_keeper_age average_whole_age average_remaining_age : ℕ) :
  captain_age = 24 →
  wicket_keeper_age = 31 →
  average_whole_age = 23 →
  average_remaining_age = 22 →
  n * average_whole_age - captain_age - wicket_keeper_age = (n - 2) * average_remaining_age →
  n = 11 :=
by
  intros h_cap_age h_wk_age h_avg_whole h_avg_remain h_eq
  sorry

end cricket_team_members_l205_205133


namespace bisection_second_iteration_value_l205_205712

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_second_iteration_value :
  f 0.25 = -0.234375 :=
by
  -- The proof steps would go here
  sorry

end bisection_second_iteration_value_l205_205712


namespace no_maximal_radius_of_inscribed_cylinder_l205_205817

theorem no_maximal_radius_of_inscribed_cylinder
  (base_radius_cone : ℝ) (height_cone : ℝ)
  (h_base_radius : base_radius_cone = 5) (h_height : height_cone = 10) :
  ¬ ∃ r : ℝ, 0 < r ∧ r < 5 ∧
    ∀ t : ℝ, 0 < t ∧ t < 5 → 2 * Real.pi * (10 * r - r ^ 2) ≥ 2 * Real.pi * (10 * t - t ^ 2) :=
by
  sorry

end no_maximal_radius_of_inscribed_cylinder_l205_205817


namespace min_value_frac_function_l205_205339

theorem min_value_frac_function (x : ℝ) (h : x > -1) : (x^2 / (x + 1)) ≥ 0 :=
sorry

end min_value_frac_function_l205_205339


namespace div_expression_l205_205514

theorem div_expression : (124 : ℝ) / (8 + 14 * 3) = 2.48 := by
  sorry

end div_expression_l205_205514


namespace valve_solution_l205_205564

noncomputable def valve_problem : Prop :=
  ∀ (x y z : ℝ),
  (1 / (x + y + z) = 2) →
  (1 / (x + z) = 4) →
  (1 / (y + z) = 3) →
  (1 / (x + y) = 2.4)

theorem valve_solution : valve_problem :=
by
  -- proof omitted
  intros x y z h1 h2 h3
  sorry

end valve_solution_l205_205564


namespace parallelogram_area_l205_205883

theorem parallelogram_area (b h : ℝ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l205_205883


namespace intersection_point_of_lines_l205_205731

theorem intersection_point_of_lines (x y : ℝ) :
  (2 * x - 3 * y = 3) ∧ (4 * x + 2 * y = 2) ↔ (x = 3/4) ∧ (y = -1/2) :=
by
  sorry

end intersection_point_of_lines_l205_205731


namespace largest_divisor_even_squares_l205_205294

theorem largest_divisor_even_squares (m n : ℕ) (hm : Even m) (hn : Even n) (h : n < m) :
  ∃ k, k = 4 ∧ ∀ a b : ℕ, Even a → Even b → b < a → k ∣ (a^2 - b^2) :=
by
  sorry

end largest_divisor_even_squares_l205_205294


namespace garden_length_is_60_l205_205889

noncomputable def garden_length (w l : ℕ) : Prop :=
  l = 2 * w ∧ 2 * w + 2 * l = 180

theorem garden_length_is_60 (w l : ℕ) (h : garden_length w l) : l = 60 :=
by
  sorry

end garden_length_is_60_l205_205889


namespace vector_addition_correct_dot_product_correct_l205_205398

-- Define the two vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- Define the expected results
def a_plus_b_expected : ℝ × ℝ := (4, 3)
def a_dot_b_expected : ℝ := 5

-- Prove the sum of vectors a and b
theorem vector_addition_correct : a + b = a_plus_b_expected := by
  sorry

-- Prove the dot product of vectors a and b
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = a_dot_b_expected := by
  sorry

end vector_addition_correct_dot_product_correct_l205_205398


namespace sum_of_geometric_sequence_l205_205880

theorem sum_of_geometric_sequence :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 3280 / 6561 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  sorry

end sum_of_geometric_sequence_l205_205880


namespace determine_d_l205_205719

theorem determine_d (m n d : ℝ) (p : ℝ) (hp : p = 0.6666666666666666) 
  (h1 : m = 3 * n + 5) (h2 : m + d = 3 * (n + p) + 5) : d = 2 :=
by {
  sorry
}

end determine_d_l205_205719


namespace find_a_l205_205407

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem find_a (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = -1 ∨ a = (1 / 3) :=
sorry

end find_a_l205_205407


namespace irreducible_fraction_for_any_n_l205_205702

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end irreducible_fraction_for_any_n_l205_205702


namespace problem1_l205_205050

def setA : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def setB (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m - 2}

theorem problem1 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∀ x, x ∈ setA ↔ x ∈ setB m) → 3 ≤ m :=
sorry

end problem1_l205_205050


namespace least_integer_divisors_l205_205288

theorem least_integer_divisors (n m k : ℕ)
  (h_divisors : 3003 = 3 * 7 * 11 * 13)
  (h_form : n = m * 30 ^ k)
  (h_no_div_30 : ¬(30 ∣ m))
  (h_divisor_count : ∀ (p : ℕ) (h : n = p), (p + 1) * (p + 1) * (p + 1) * (p + 1) = 3003)
  : m + k = 104978 :=
sorry

end least_integer_divisors_l205_205288


namespace max_value_a4_b2_c2_d2_l205_205444

theorem max_value_a4_b2_c2_d2
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  a^4 + b^2 + c^2 + d^2 ≤ 100 :=
sorry

end max_value_a4_b2_c2_d2_l205_205444


namespace part_a_part_b_l205_205963

-- Definition based on conditions
def S (n k : ℕ) : ℕ :=
  -- Placeholder: Actual definition would count the coefficients
  -- of (x+1)^n that are not divisible by k.
  sorry

-- Part (a) proof statement
theorem part_a : S 2012 3 = 324 :=
by sorry

-- Part (b) proof statement
theorem part_b : 2012 ∣ S (2012^2011) 2011 :=
by sorry

end part_a_part_b_l205_205963


namespace zoo_visitors_sunday_l205_205812

-- Definitions based on conditions
def friday_visitors : ℕ := 1250
def saturday_multiplier : ℚ := 3
def sunday_decrease_percent : ℚ := 0.15

-- Assert the equivalence
theorem zoo_visitors_sunday : 
  let saturday_visitors := friday_visitors * saturday_multiplier
  let sunday_visitors := saturday_visitors * (1 - sunday_decrease_percent)
  round (sunday_visitors : ℚ) = 3188 :=
by
  sorry

end zoo_visitors_sunday_l205_205812


namespace standard_eq_of_tangent_circle_l205_205057

-- Define the center and tangent condition of the circle
def center : ℝ × ℝ := (1, 2)
def tangent_to_x_axis (r : ℝ) : Prop := r = center.snd

-- The standard equation of the circle given the center and radius
def standard_eq_circle (h k r : ℝ) : Prop := ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement to prove the standard equation of the circle
theorem standard_eq_of_tangent_circle : 
  ∃ r, tangent_to_x_axis r ∧ standard_eq_circle 1 2 r := 
by 
  sorry

end standard_eq_of_tangent_circle_l205_205057


namespace reckha_code_count_l205_205839

theorem reckha_code_count :
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970 :=
by
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  show total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970
  sorry

end reckha_code_count_l205_205839


namespace sara_red_balloons_l205_205737

theorem sara_red_balloons (initial_red : ℕ) (given_red : ℕ) 
  (h_initial : initial_red = 31) (h_given : given_red = 24) : 
  initial_red - given_red = 7 :=
by {
  sorry
}

end sara_red_balloons_l205_205737


namespace jill_bought_5_packs_of_red_bouncy_balls_l205_205986

theorem jill_bought_5_packs_of_red_bouncy_balls
  (r : ℕ) -- number of packs of red bouncy balls
  (yellow_packs : ℕ := 4)
  (bouncy_balls_per_pack : ℕ := 18)
  (extra_red_bouncy_balls : ℕ := 18)
  (total_yellow_bouncy_balls : ℕ := yellow_packs * bouncy_balls_per_pack)
  (total_red_bouncy_balls : ℕ := total_yellow_bouncy_balls + extra_red_bouncy_balls)
  (h : r * bouncy_balls_per_pack = total_red_bouncy_balls) :
  r = 5 :=
by sorry

end jill_bought_5_packs_of_red_bouncy_balls_l205_205986


namespace prob1_prob2_prob3_l205_205208

-- Problem 1
theorem prob1 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k = 2/5 := 
sorry

-- Problem 2
theorem prob2 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  0 < k ∧ k ≤ 2/5 := 
sorry

-- Problem 3
theorem prob3 (k : ℝ) (h₀ : k > 0)
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k ≥ 2/5 := 
sorry

end prob1_prob2_prob3_l205_205208


namespace oppose_estimation_l205_205327

-- Define the conditions
def survey_total : ℕ := 50
def favorable_attitude : ℕ := 15
def total_population : ℕ := 9600

-- Calculate the proportion opposed
def proportion_opposed : ℚ := (survey_total - favorable_attitude) / survey_total

-- Define the statement to be proved
theorem oppose_estimation : 
  proportion_opposed * total_population = 6720 := by
  sorry

end oppose_estimation_l205_205327


namespace li_payment_l205_205372

noncomputable def payment_li (daily_payment_per_unit : ℚ) (days_li_worked : ℕ) : ℚ :=
daily_payment_per_unit * days_li_worked

theorem li_payment (work_per_day : ℚ) (days_li_worked : ℕ) (days_extra_work : ℕ) 
  (difference_payment : ℚ) (daily_payment_per_unit : ℚ) (initial_nanual_workdays : ℕ) :
  work_per_day = 1 →
  days_li_worked = 2 →
  days_extra_work = 3 →
  difference_payment = 2700 →
  daily_payment_per_unit = difference_payment / (initial_nanual_workdays + (3 * 3)) → 
  payment_li daily_payment_per_unit days_li_worked = 450 := 
by 
  intros h_work_per_day h_days_li_worked h_days_extra_work h_diff_payment h_daily_payment 
  sorry

end li_payment_l205_205372


namespace complex_sum_power_l205_205923

noncomputable def z : ℂ := sorry

theorem complex_sum_power (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 :=
sorry

end complex_sum_power_l205_205923


namespace product_of_5_consecutive_integers_divisible_by_60_l205_205795

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l205_205795


namespace find_k_l205_205802

def vector (α : Type) := (α × α)
def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)
def add (v1 v2 : vector ℝ) : vector ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def smul (c : ℝ) (v : vector ℝ) : vector ℝ := (c * v.1, c * v.2)
def cross_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.2 - v1.2 * v2.1

theorem find_k (k : ℝ) (h : cross_product (add a (smul 2 (b k)))
                                          (add (smul 3 a) (smul (-1) (b k))) = 0) : k = -6 :=
sorry

end find_k_l205_205802


namespace buyers_muffin_mix_l205_205967

variable (P C M CM: ℕ)

theorem buyers_muffin_mix
    (h_total: P = 100)
    (h_cake: C = 50)
    (h_both: CM = 17)
    (h_neither: P - (C + M - CM) = 27)
    : M = 73 :=
by sorry

end buyers_muffin_mix_l205_205967


namespace find_a_l205_205374

theorem find_a
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, x = 1 ∨ x = 2 → a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0)
  (h2 : a + b + c = 2) : 
  a = 12 := 
sorry

end find_a_l205_205374


namespace remainder_of_poly_div_l205_205809

theorem remainder_of_poly_div (n : ℕ) (h : n > 2) : (n^3 + 3) % (n + 1) = 2 :=
by 
  sorry

end remainder_of_poly_div_l205_205809


namespace diff_lines_not_parallel_perpendicular_same_plane_l205_205080

-- Variables
variables (m n : Type) (α β : Type)

-- Conditions
-- m and n are different lines, which we can assume as different types (or elements of some type).
-- α and β are different planes, which we can assume as different types (or elements of some type).
-- There exist definitions for parallel and perpendicular relationships between lines and planes.

def areParallel (x y : Type) : Prop := sorry
def arePerpendicularToSamePlane (x y : Type) : Prop := sorry

-- Theorem Statement
theorem diff_lines_not_parallel_perpendicular_same_plane
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : ¬ areParallel m n) :
  ¬ arePerpendicularToSamePlane m n :=
sorry

end diff_lines_not_parallel_perpendicular_same_plane_l205_205080


namespace kids_stay_home_correct_l205_205776

def total_number_of_kids : ℕ := 1363293
def kids_who_go_to_camp : ℕ := 455682
def kids_staying_home : ℕ := total_number_of_kids - kids_who_go_to_camp

theorem kids_stay_home_correct :
  kids_staying_home = 907611 := by 
  sorry

end kids_stay_home_correct_l205_205776


namespace find_x_l205_205024

theorem find_x (n : ℕ) (hn : n % 2 = 1) (hpf : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 * p2 * p3 = 9^n - 1 ∧ [p1, p2, p3].contains 61) :
  9^n - 1 = 59048 :=
by
  sorry

end find_x_l205_205024


namespace vertex_of_parabola_l205_205608

theorem vertex_of_parabola :
  ∀ (x y : ℝ), y = (1 / 3) * (x - 7) ^ 2 + 5 → ∃ h k : ℝ, h = 7 ∧ k = 5 ∧ y = (1 / 3) * (x - h) ^ 2 + k :=
by
  intro x y h
  sorry

end vertex_of_parabola_l205_205608


namespace teal_more_blue_l205_205317

def numSurveyed : ℕ := 150
def numGreen : ℕ := 90
def numBlue : ℕ := 50
def numBoth : ℕ := 40
def numNeither : ℕ := 20

theorem teal_more_blue : 40 + (numSurveyed - (numBoth + (numGreen - numBoth) + numNeither)) = 80 :=
by
  -- Here we simplify numerically until we get the required answer
  -- start with calculating the total accounted and remaining
  sorry

end teal_more_blue_l205_205317


namespace real_roots_exist_l205_205046

noncomputable def cubic_equation (x : ℝ) := x^3 - x^2 - 2*x + 1

theorem real_roots_exist : ∃ (a b : ℝ), 
  cubic_equation a = 0 ∧ cubic_equation b = 0 ∧ a - a * b = 1 := 
by
  sorry

end real_roots_exist_l205_205046


namespace infinite_set_P_l205_205890

-- Define the condition as given in the problem
def has_property_P (P : Set ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∀ p : ℕ, p.Prime → p ∣ k^3 + 6 → p ∈ P)

-- State the proof problem
theorem infinite_set_P (P : Set ℕ) (h : has_property_P P) : ∃ p : ℕ, p ∉ P → false :=
by
  -- The statement asserts that the set P described by has_property_P is infinite.
  sorry

end infinite_set_P_l205_205890


namespace election_majority_l205_205151

theorem election_majority
  (total_votes : ℕ)
  (winning_percent : ℝ)
  (other_percent : ℝ)
  (votes_cast : total_votes = 700)
  (winning_share : winning_percent = 0.84)
  (other_share : other_percent = 0.16) :
  ∃ majority : ℕ, majority = 476 := by
  sorry

end election_majority_l205_205151


namespace souvenirs_total_cost_l205_205928

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end souvenirs_total_cost_l205_205928


namespace option_A_correct_l205_205305

variable (f g : ℝ → ℝ)

-- Given conditions
axiom cond1 : ∀ x : ℝ, f x - g (4 - x) = 2
axiom cond2 : ∀ x : ℝ, deriv g x = deriv f (x - 2)
axiom cond3 : ∀ x : ℝ, f (x + 2) = - f (- x - 2)

theorem option_A_correct : ∀ x : ℝ, f (4 + x) + f (- x) = 0 :=
by
  -- Proving the theorem
  sorry

end option_A_correct_l205_205305


namespace convert_speed_l205_205779

theorem convert_speed (v_m_s : ℚ) (conversion_factor : ℚ) :
  v_m_s = 12 / 43 → conversion_factor = 3.6 → v_m_s * conversion_factor = 1.0046511624 := by
  intros h1 h2
  have h3 : v_m_s = 12 / 43 := h1
  have h4 : conversion_factor = 3.6 := h2
  rw [h3, h4]
  norm_num
  sorry

end convert_speed_l205_205779


namespace parabola_equation_l205_205309

theorem parabola_equation (m : ℝ) (focus : ℝ × ℝ) (M : ℝ × ℝ) 
  (h_vertex : (0, 0) = (0, 0))
  (h_focus : focus = (p, 0))
  (h_point : M = (1, m))
  (h_distance : dist M focus = 2) 
  : (forall x y : ℝ, y^2 = 4*x) :=
sorry

end parabola_equation_l205_205309


namespace pencil_cost_l205_205914

theorem pencil_cost (P : ℕ) (h1 : ∀ p : ℕ, p = 80) (h2 : ∀ p_est, ((16 * P) + (20 * 80)) = p_est → p_est = 2000) (h3 : 36 = 16 + 20) :
    P = 25 :=
  sorry

end pencil_cost_l205_205914


namespace sum_of_roots_of_quadratic_l205_205607

theorem sum_of_roots_of_quadratic : 
  ∀ x1 x2 : ℝ, 
  (3 * x1^2 - 6 * x1 - 7 = 0 ∧ 3 * x2^2 - 6 * x2 - 7 = 0) → 
  (x1 + x2 = 2) := by
  sorry

end sum_of_roots_of_quadratic_l205_205607


namespace max_m_value_l205_205312

theorem max_m_value {m : ℝ} : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0 → x < m)) ∧ ¬(∀ x : ℝ, (x^2 - 2 * x - 8 > 0 ↔ x < m)) → m ≤ -2 :=
sorry

end max_m_value_l205_205312


namespace certain_number_eq_0_08_l205_205786

theorem certain_number_eq_0_08 (x : ℝ) (h : 1 / x = 12.5) : x = 0.08 :=
by
  sorry

end certain_number_eq_0_08_l205_205786


namespace simon_change_l205_205971

def pansy_price : ℝ := 2.50
def pansy_count : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_count : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_count : ℕ := 5
def discount_rate : ℝ := 0.10
def initial_payment : ℝ := 50.00

theorem simon_change : 
  let total_cost := (pansy_count * pansy_price) + (hydrangea_count * hydrangea_price) + (petunia_count * petunia_price)
  let discount := total_cost * discount_rate
  let cost_after_discount := total_cost - discount
  let change := initial_payment - cost_after_discount
  change = 23.00 :=
by
  sorry

end simon_change_l205_205971


namespace arithmetic_sequence_sum_l205_205770

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∀ n, a (n+1) = a n + 3)
  (h_a1_a2 : a 1 + a 2 = 7)
  (h_a3 : a 3 = 8)
  (h_bn : ∀ n, b n = 1 / (a n * a (n+1)))
  :
  (∀ n, a n = 3 * n - 1) ∧ (T n = n / (2 * (3 * n + 2))) :=
by 
  sorry

end arithmetic_sequence_sum_l205_205770


namespace x_value_not_unique_l205_205191

theorem x_value_not_unique (x y : ℝ) (h1 : y = x) (h2 : y = (|x + y - 2|) / (Real.sqrt 2)) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
(∃ y1 y2 : ℝ, (y1 = x1 ∧ y2 = x2 ∧ y1 = (|x1 + y1 - 2|) / Real.sqrt 2 ∧ y2 = (|x2 + y2 - 2|) / Real.sqrt 2)) :=
by
  sorry

end x_value_not_unique_l205_205191


namespace total_marbles_l205_205871

variable (b : ℝ)
variable (r : ℝ) (g : ℝ)
variable (h₁ : r = 1.3 * b)
variable (h₂ : g = 1.5 * b)

theorem total_marbles (b : ℝ) (r : ℝ) (g : ℝ) (h₁ : r = 1.3 * b) (h₂ : g = 1.5 * b) : r + b + g = 3.8 * b :=
by
  sorry

end total_marbles_l205_205871


namespace remainder_when_divided_by_s_minus_2_l205_205799

noncomputable def f (s : ℤ) : ℤ := s^15 + s^2 + 3

theorem remainder_when_divided_by_s_minus_2 : f 2 = 32775 := 
by
  sorry

end remainder_when_divided_by_s_minus_2_l205_205799


namespace no_apples_info_l205_205970

theorem no_apples_info (r d : ℕ) (condition1 : r = 79) (condition2 : d = 53) (condition3 : r = d + 26) : 
  ∀ a : ℕ, (a = a) → false :=
by
  intro a h
  sorry

end no_apples_info_l205_205970


namespace mike_initial_marbles_l205_205377

-- Defining the conditions
def gave_marble (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles
def marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles

-- Using the given conditions
def initial_mike_marbles : ℕ := 8
def given_marbles : ℕ := 4
def remaining_marbles : ℕ := 4

-- Proving the statement
theorem mike_initial_marbles :
  initial_mike_marbles - given_marbles = remaining_marbles :=
by
  -- The proof
  sorry

end mike_initial_marbles_l205_205377


namespace minimum_value_l205_205459

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + 3 * b = 1) : 
  26 ≤ (2 / a + 3 / b) :=
sorry

end minimum_value_l205_205459


namespace ratio_of_number_to_ten_l205_205659

theorem ratio_of_number_to_ten (n : ℕ) (h : n = 200) : n / 10 = 20 :=
by
  sorry

end ratio_of_number_to_ten_l205_205659


namespace sequence_sum_l205_205299

theorem sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) = (1/3) * a n) (h_a4a5 : a 4 + a 5 = 4) :
    a 2 + a 3 = 36 :=
    sorry

end sequence_sum_l205_205299


namespace discount_is_28_l205_205826

-- Definitions
def price_notebook : ℕ := 15
def price_planner : ℕ := 10
def num_notebooks : ℕ := 4
def num_planners : ℕ := 8
def total_cost_with_discount : ℕ := 112

-- The original cost without discount
def original_cost : ℕ := num_notebooks * price_notebook + num_planners * price_planner

-- The discount amount
def discount_amount : ℕ := original_cost - total_cost_with_discount

-- Proof statement
theorem discount_is_28 : discount_amount = 28 := by
  sorry

end discount_is_28_l205_205826


namespace line_through_center_and_perpendicular_l205_205984

def center_of_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 1

def perpendicular_to_line (slope : ℝ) : Prop :=
  slope = 1

theorem line_through_center_and_perpendicular (x y : ℝ) :
  center_of_circle x y →
  perpendicular_to_line 1 →
  (x - y + 1 = 0) :=
by
  intros h_center h_perpendicular
  sorry

end line_through_center_and_perpendicular_l205_205984


namespace find_equidistant_point_l205_205872

theorem find_equidistant_point :
  ∃ (x z : ℝ),
    ((x - 1)^2 + 4^2 + z^2 = (x - 2)^2 + 2^2 + (z - 3)^2) ∧
    ((x - 1)^2 + 4^2 + z^2 = (x - 3)^2 + 9 + (z + 2)^2) ∧
    (x + 2 * z = 5) ∧
    (x = 15 / 8) ∧
    (z = 5 / 8) :=
by
  sorry

end find_equidistant_point_l205_205872


namespace scalene_triangle_minimum_altitude_l205_205173

theorem scalene_triangle_minimum_altitude (a b c : ℕ) (h : ℕ) 
  (h₁ : a ≠ b ∧ b ≠ c ∧ c ≠ a) -- scalene condition
  (h₂ : ∃ k : ℕ, ∃ m : ℕ, k * m = a ∧ m = 6) -- first altitude condition
  (h₃ : ∃ k : ℕ, ∃ n : ℕ, k * n = b ∧ n = 8) -- second altitude condition
  (h₄ : c = (7 : ℕ) * b / (3 : ℕ)) -- third side condition given inequalities and area relations
  : h = 2 := 
sorry

end scalene_triangle_minimum_altitude_l205_205173


namespace sum_f_always_negative_l205_205605

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_always_negative
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
by
  unfold f
  sorry

end sum_f_always_negative_l205_205605


namespace solution_set_of_inequality_l205_205745

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 3*x + 4 > 0 } = { x : ℝ | -1 < x ∧ x < 4 } := 
sorry

end solution_set_of_inequality_l205_205745


namespace rational_solutions_zero_l205_205072

theorem rational_solutions_zero (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end rational_solutions_zero_l205_205072


namespace trajectory_of_center_l205_205807

-- Define the given conditions
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

def tangent_y_axis (x : ℝ) : Prop := x = 0

-- Define the theorem with the given conditions and the desired conclusion
theorem trajectory_of_center (x y : ℝ) (h1 : tangent_circle x y) (h2 : tangent_y_axis x) :
  (y^2 = 8 * x) ∨ (y = 0 ∧ x ≤ 0) :=
sorry

end trajectory_of_center_l205_205807


namespace snow_on_Monday_l205_205526

def snow_on_Tuesday : ℝ := 0.21
def snow_on_Monday_and_Tuesday : ℝ := 0.53

theorem snow_on_Monday : snow_on_Monday_and_Tuesday - snow_on_Tuesday = 0.32 :=
by
  sorry

end snow_on_Monday_l205_205526


namespace initial_invited_people_l205_205521

theorem initial_invited_people (not_showed_up : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) 
  (H1 : not_showed_up = 12) (H2 : table_capacity = 3) (H3 : tables_needed = 2) :
  not_showed_up + (table_capacity * tables_needed) = 18 :=
by
  sorry

end initial_invited_people_l205_205521


namespace mr_william_land_percentage_l205_205875

/--
Given:
1. Farm tax is levied on 90% of the cultivated land.
2. The tax department collected a total of $3840 through the farm tax from the village.
3. Mr. William paid $480 as farm tax.

Prove: The percentage of total land of Mr. William over the total taxable land of the village is 12.5%.
-/
theorem mr_william_land_percentage (T W : ℝ) 
  (h1 : 0.9 * W = 480) 
  (h2 : 0.9 * T = 3840) : 
  (W / T) * 100 = 12.5 :=
by
  sorry

end mr_william_land_percentage_l205_205875


namespace friends_meeting_distance_l205_205343

theorem friends_meeting_distance (R_q : ℝ) (t : ℝ) (D_p D_q trail_length : ℝ) :
  trail_length = 36 ∧ D_p = 1.25 * R_q * t ∧ D_q = R_q * t ∧ D_p + D_q = trail_length → D_p = 20 := by
  sorry

end friends_meeting_distance_l205_205343


namespace increasing_function_solution_l205_205063

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y

theorem increasing_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y)
  ∧ (∀ x y : ℝ, x < y → f x < f y)
  → ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = 1 / (a * x) :=
by {
  sorry
}

end increasing_function_solution_l205_205063


namespace total_weight_of_containers_l205_205873

theorem total_weight_of_containers (x y z : ℕ) :
  x + y = 162 →
  y + z = 168 →
  z + x = 174 →
  x + y + z = 252 :=
by
  intros hxy hyz hzx
  -- proof skipped
  sorry

end total_weight_of_containers_l205_205873


namespace divide_pile_l205_205223

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end divide_pile_l205_205223


namespace find_x_l205_205951

theorem find_x 
  (x : ℝ)
  (h : 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005) : 
  x = 0.225 := 
sorry

end find_x_l205_205951


namespace find_theta_l205_205113

-- Define the angles
variables (VEK KEW EVG θ : ℝ)

-- State the conditions as hypotheses
def conditions (VEK KEW EVG θ : ℝ) := 
  VEK = 70 ∧
  KEW = 40 ∧
  EVG = 110

-- State the theorem
theorem find_theta (VEK KEW EVG θ : ℝ)
  (h : conditions VEK KEW EVG θ) : 
  θ = 40 :=
by {
  sorry
}

end find_theta_l205_205113


namespace sum_of_integers_is_96_l205_205437

theorem sum_of_integers_is_96 (x y : ℤ) (h1 : x = 32) (h2 : y = 2 * x) : x + y = 96 := 
by
  sorry

end sum_of_integers_is_96_l205_205437


namespace mass_percentage_H_in_CaH₂_l205_205099

def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_H : ℝ := 1.008
def molar_mass_CaH₂ : ℝ := atomic_mass_Ca + 2 * atomic_mass_H

theorem mass_percentage_H_in_CaH₂ :
  (2 * atomic_mass_H / molar_mass_CaH₂) * 100 = 4.79 := 
by
  -- Skipping the detailed proof for now
  sorry

end mass_percentage_H_in_CaH₂_l205_205099


namespace range_of_a_l205_205090

-- Define the function f
def f (a x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := -3 * x^2 + 2 * a * x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_prime a x ≤ 0) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l205_205090


namespace false_proposition_l205_205330

-- Definitions based on conditions
def opposite_angles (α β : ℝ) : Prop := α = β
def perpendicular (l m : ℝ → ℝ) : Prop := ∀ x, l x * m x = -1
def parallel (l m : ℝ → ℝ) : Prop := ∃ c, ∀ x, l x = m x + c
def corresponding_angles (α β : ℝ) : Prop := α = β

-- Propositions from the problem
def proposition1 : Prop := ∀ α β, opposite_angles α β → α = β
def proposition2 : Prop := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
def proposition3 : Prop := ∀ α β, α = β → opposite_angles α β
def proposition4 : Prop := ∀ α β, corresponding_angles α β → α = β

-- Statement to prove proposition 3 is false under given conditions
theorem false_proposition : ¬ proposition3 := by
  -- By our analysis, if proposition 3 is false, then it means the given definition for proposition 3 holds under all circumstances.
  sorry

end false_proposition_l205_205330


namespace domain_of_c_is_all_real_l205_205331

theorem domain_of_c_is_all_real (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 3 * x + a ≠ 0) ↔ a < -3 / 4 :=
by
  sorry

end domain_of_c_is_all_real_l205_205331


namespace day_100_days_from_friday_l205_205993

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l205_205993


namespace books_in_special_collection_at_beginning_of_month_l205_205583

theorem books_in_special_collection_at_beginning_of_month
  (loaned_out_real : Real)
  (loaned_out_books : Int)
  (returned_ratio : Real)
  (books_at_end : Int)
  (B : Int)
  (h1 : loaned_out_real = 49.99999999999999)
  (h2 : loaned_out_books = 50)
  (h3 : returned_ratio = 0.70)
  (h4 : books_at_end = 60)
  (h5 : loaned_out_books = Int.floor loaned_out_real)
  (h6 : ∀ (loaned_books : Int), loaned_books ≤ loaned_out_books → returned_ratio * loaned_books + (loaned_books - returned_ratio * loaned_books) = loaned_books)
  : B = 75 :=
by
  sorry

end books_in_special_collection_at_beginning_of_month_l205_205583


namespace domino_trick_l205_205123

theorem domino_trick (x y : ℕ) (h1 : x ≤ 6) (h2 : y ≤ 6)
  (h3 : 10 * x + y + 30 = 62) : x = 3 ∧ y = 2 :=
by
  sorry

end domino_trick_l205_205123


namespace largest_possible_sum_l205_205921

theorem largest_possible_sum (a b : ℤ) (h : a^2 - b^2 = 144) : a + b ≤ 72 :=
sorry

end largest_possible_sum_l205_205921


namespace proof_problem_l205_205575

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_problem_l205_205575


namespace total_earnings_l205_205509

-- Define the constants and conditions.
def regular_hourly_rate : ℕ := 5
def overtime_hourly_rate : ℕ := 6
def regular_hours_per_week : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define the proof problem in Lean 4.
theorem total_earnings : (regular_hours_per_week * 2 * regular_hourly_rate + 
                         ((first_week_hours - regular_hours_per_week) + 
                          (second_week_hours - regular_hours_per_week)) * overtime_hourly_rate) = 472 := 
by 
  exact sorry -- Detailed proof steps would go here.

end total_earnings_l205_205509


namespace race_head_start_l205_205503

-- This statement defines the problem in Lean 4
theorem race_head_start (Va Vb L H : ℝ) 
(h₀ : Va = 51 / 44 * Vb) 
(h₁ : L / Va = (L - H) / Vb) : 
H = 7 / 51 * L := 
sorry

end race_head_start_l205_205503


namespace find_a_l205_205205

variable (x y a : ℝ)

theorem find_a (h1 : (a * x + 8 * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : a = 7 :=
sorry

end find_a_l205_205205


namespace distance_ratio_l205_205463

theorem distance_ratio (x : ℝ) (hx : abs x = 8) : abs (-4) / abs x = 1 / 2 :=
by {
  sorry
}

end distance_ratio_l205_205463


namespace find_ratio_of_square_to_circle_radius_l205_205781

def sector_circle_ratio (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) : Prop :=
  (R = (5 * a * sqrt2) / 2) →
  (r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) →
  (a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2))

theorem find_ratio_of_square_to_circle_radius
  (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) (h1 : R = (5 * a * sqrt2) / 2)
  (h2 : r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) :
  a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2) :=
  sorry

end find_ratio_of_square_to_circle_radius_l205_205781


namespace three_alpha_four_plus_eight_beta_three_eq_876_l205_205576

variable (α β : ℝ)

-- Condition 1: α and β are roots of the equation x^2 - 3x - 4 = 0
def roots_of_quadratic : Prop := α^2 - 3 * α - 4 = 0 ∧ β^2 - 3 * β - 4 = 0

-- Question: 3α^4 + 8β^3 = ?
theorem three_alpha_four_plus_eight_beta_three_eq_876 
  (h : roots_of_quadratic α β) : (3 * α^4 + 8 * β^3 = 876) := sorry

end three_alpha_four_plus_eight_beta_three_eq_876_l205_205576


namespace trajectory_of_circle_center_is_ellipse_l205_205358

theorem trajectory_of_circle_center_is_ellipse 
    (a b : ℝ) (θ : ℝ) 
    (h1 : a ≠ b)
    (h2 : 0 < a)
    (h3 : 0 < b)
    : ∃ (x y : ℝ), 
    (x, y) = (a * Real.cos θ, b * Real.sin θ) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end trajectory_of_circle_center_is_ellipse_l205_205358


namespace rope_length_before_folding_l205_205552

theorem rope_length_before_folding (L : ℝ) (h : L / 4 = 10) : L = 40 :=
by
  sorry

end rope_length_before_folding_l205_205552


namespace range_of_a_l205_205689

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem range_of_a
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f (x^2 + 2) + f (-2 * a * x) ≥ 0) :
  -3/2 ≤ a ∧ a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l205_205689


namespace boarders_joined_l205_205397

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ)
  (final_boarders : ℕ) (x : ℕ)
  (ratio_initial : initial_boarders * 16 = initial_day_scholars * 7)
  (ratio_final : final_boarders * 2 = initial_day_scholars)
  (final_boarders_eq : final_boarders = initial_boarders + x)
  (initial_boarders_val : initial_boarders = 560)
  (initial_day_scholars_val : initial_day_scholars = 1280)
  (final_boarders_val : final_boarders = 640) :
  x = 80 :=
by
  sorry

end boarders_joined_l205_205397


namespace factorization_identity_sum_l205_205680

theorem factorization_identity_sum (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 15 * x + 36 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 20 :=
sorry

end factorization_identity_sum_l205_205680


namespace janet_income_difference_l205_205342

def janet_current_job_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def janet_freelance_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def extra_fica_taxes (weekly_tax : ℝ) (weeks_per_month : ℕ) : ℝ :=
  weekly_tax * weeks_per_month

def healthcare_premiums (monthly_premium : ℝ) : ℝ :=
  monthly_premium

def janet_net_freelance_income (freelance_income : ℝ) (additional_costs : ℝ) : ℝ :=
  freelance_income - additional_costs

theorem janet_income_difference
  (hours_per_week : ℕ)
  (weeks_per_month : ℕ)
  (current_hourly_rate : ℝ)
  (freelance_hourly_rate : ℝ)
  (weekly_tax : ℝ)
  (monthly_premium : ℝ)
  (H_hours : hours_per_week = 40)
  (H_weeks : weeks_per_month = 4)
  (H_current_rate : current_hourly_rate = 30)
  (H_freelance_rate : freelance_hourly_rate = 40)
  (H_weekly_tax : weekly_tax = 25)
  (H_monthly_premium : monthly_premium = 400) :
  janet_net_freelance_income (janet_freelance_income 40 4 40) (extra_fica_taxes 25 4 + healthcare_premiums 400) 
  - janet_current_job_income 40 4 30 = 1100 := 
  by 
    sorry

end janet_income_difference_l205_205342


namespace total_money_is_correct_l205_205842

-- Define the values of different types of coins and the amount of each.
def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4
def cash : ℕ := 45

-- Define the total amount of money.
def total_money : ℕ :=
  (gold_count * gold_value) +
  (silver_count * silver_value) +
  (bronze_count * bronze_value) +
  (titanium_count * titanium_value) + cash

-- The proof statement
theorem total_money_is_correct : total_money = 1055 := by
  sorry

end total_money_is_correct_l205_205842


namespace asymptotes_of_hyperbola_l205_205922

theorem asymptotes_of_hyperbola (k : ℤ) (h1 : (k - 2016) * (k - 2018) < 0) :
  ∀ x y: ℝ, (x ^ 2) - (y ^ 2) = 1 → ∃ a b: ℝ, y = x * a ∨ y = x * b :=
by
  sorry

end asymptotes_of_hyperbola_l205_205922


namespace solve_inequality_l205_205431

def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

theorem solve_inequality : ∀ x : ℝ, |f x| ≤ 4 :=
by
  intro x
  sorry

end solve_inequality_l205_205431


namespace calculation_l205_205489

theorem calculation (a b : ℕ) (h1 : a = 7) (h2 : b = 5) : (a^2 - b^2) ^ 2 = 576 :=
by
  sorry

end calculation_l205_205489


namespace equation_solution_unique_or_not_l205_205146

theorem equation_solution_unique_or_not (a b : ℝ) :
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2) ↔ 
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
by
  sorry

end equation_solution_unique_or_not_l205_205146


namespace polynomial_abs_sum_l205_205555

theorem polynomial_abs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h : (2*X - 1)^5 = a_5 * X^5 + a_4 * X^4 + a_3 * X^3 + a_2 * X^2 + a_1 * X + a_0) :
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 243 :=
by
  sorry

end polynomial_abs_sum_l205_205555


namespace smallest_prime_dividing_sum_l205_205107

theorem smallest_prime_dividing_sum (a b : ℕ) (h₁ : a = 7^15) (h₂ : b = 9^17) (h₃ : a % 2 = 1) (h₄ : b % 2 = 1) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (a + b) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (a + b)) → q ≥ p := by
  sorry

end smallest_prime_dividing_sum_l205_205107


namespace age_is_50_l205_205716

-- Definitions only based on the conditions provided
def future_age (A: ℕ) := A + 5
def past_age (A: ℕ) := A - 5

theorem age_is_50 (A : ℕ) (h : 5 * future_age A - 5 * past_age A = A) : A = 50 := 
by 
  sorry  -- proof should be provided here

end age_is_50_l205_205716


namespace div_pow_eq_l205_205711

theorem div_pow_eq {a : ℝ} (h : a ≠ 0) : a^3 / a^2 = a :=
sorry

end div_pow_eq_l205_205711


namespace rohan_house_rent_percentage_l205_205769

noncomputable def house_rent_percentage (food_percentage entertainment_percentage conveyance_percentage salary savings: ℝ) : ℝ :=
  100 - (food_percentage + entertainment_percentage + conveyance_percentage + (savings / salary * 100))

-- Conditions
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def salary : ℝ := 10000
def savings : ℝ := 2000

-- Theorem
theorem rohan_house_rent_percentage :
  house_rent_percentage food_percentage entertainment_percentage conveyance_percentage salary savings = 20 := 
sorry

end rohan_house_rent_percentage_l205_205769


namespace scientific_notation_4947_66_billion_l205_205671

theorem scientific_notation_4947_66_billion :
  4947.66 * 10^8 = 4.94766 * 10^11 :=
sorry

end scientific_notation_4947_66_billion_l205_205671


namespace smallest_solution_exists_l205_205282

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l205_205282


namespace percentage_increase_l205_205497

theorem percentage_increase (L : ℕ) (h1 : L + 450 = 1350) :
  (450 / L : ℚ) * 100 = 50 := by
  sorry

end percentage_increase_l205_205497


namespace solve_for_x_l205_205243
-- Lean 4 Statement

theorem solve_for_x (x : ℝ) (h : 2^(3 * x) = Real.sqrt 32) : x = 5 / 6 := 
sorry

end solve_for_x_l205_205243


namespace probability_at_least_one_card_each_cousin_correct_l205_205698

noncomputable def probability_at_least_one_card_each_cousin : ℚ :=
  let total_cards := 16
  let cards_per_cousin := 8
  let selections := 3
  let total_ways := Nat.choose total_cards selections
  let ways_all_from_one_cousin := Nat.choose cards_per_cousin selections * 2  -- twice: once for each cousin
  let prob_all_from_one_cousin := (ways_all_from_one_cousin : ℚ) / total_ways
  1 - prob_all_from_one_cousin

theorem probability_at_least_one_card_each_cousin_correct :
  probability_at_least_one_card_each_cousin = 4 / 5 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_card_each_cousin_correct_l205_205698


namespace find_m_value_l205_205193

theorem find_m_value (x y m : ℤ) (h₁ : x = 2) (h₂ : y = -3) (h₃ : 5 * x + m * y + 2 = 0) : m = 4 := 
by 
  sorry

end find_m_value_l205_205193


namespace divide_by_10_result_l205_205658

theorem divide_by_10_result (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end divide_by_10_result_l205_205658


namespace div_by_7_l205_205592

theorem div_by_7 (k : ℕ) : (2^(6*k + 1) + 3^(6*k + 1) + 5^(6*k + 1)) % 7 = 0 := by
  sorry

end div_by_7_l205_205592


namespace distance_point_C_to_line_is_2_inch_l205_205581

/-- 
Four 2-inch squares are aligned in a straight line. The second square from the left is rotated 90 degrees, 
and then shifted vertically downward until it touches the adjacent squares. Prove that the distance from 
point C, the top vertex of the rotated square, to the original line on which the bases of the squares were 
placed is 2 inches.
-/
theorem distance_point_C_to_line_is_2_inch :
  ∀ (squares : Fin 4 → ℝ) (rotation : ℝ) (vertical_shift : ℝ) (C_position : ℝ),
  (∀ n : Fin 4, squares n = 2) →
  rotation = 90 →
  vertical_shift = 0 →
  C_position = 2 →
  C_position = 2 :=
by
  intros squares rotation vertical_shift C_position
  sorry

end distance_point_C_to_line_is_2_inch_l205_205581


namespace hyperbola_eccentricity_cond_l205_205706

def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  let a := Real.sqrt m
  let b := Real.sqrt 3
  let c := Real.sqrt (m + 3)
  let e := 2
  (e * e) = (c * c) / (a * a)

theorem hyperbola_eccentricity_cond (m : ℝ) :
  hyperbola_eccentricity_condition m ↔ m = 1 :=
by
  sorry

end hyperbola_eccentricity_cond_l205_205706


namespace correct_sqrt_evaluation_l205_205767

theorem correct_sqrt_evaluation:
  2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 :=
by 
  sorry

end correct_sqrt_evaluation_l205_205767


namespace calculate_profit_l205_205496

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l205_205496


namespace cuboid_height_l205_205877

/-- Given a cuboid with surface area 2400 cm², length 15 cm, and breadth 10 cm,
    prove that the height is 42 cm. -/
theorem cuboid_height (SA l w : ℝ) (h : ℝ) : 
  SA = 2400 → l = 15 → w = 10 → 2 * (l * w + l * h + w * h) = SA → h = 42 :=
by
  intros hSA hl hw hformula
  sorry

end cuboid_height_l205_205877


namespace bob_wins_even_n_l205_205565

def game_of_islands (n : ℕ) (even_n : n % 2 = 0) : Prop :=
  ∃ strategy : (ℕ → ℕ), -- strategy is a function representing each player's move
    ∀ A B : ℕ → ℕ, -- A and B represent the moves of Alice and Bob respectively
    (A 0 + B 1) = n → (A (A 0 + 1) ≠ B (A 0 + 1)) -- Bob can always mirror Alice’s move.

theorem bob_wins_even_n (n : ℕ) (h : n % 2 = 0) : game_of_islands n h :=
sorry

end bob_wins_even_n_l205_205565


namespace apples_eq_pears_l205_205759

-- Define the conditions
def apples_eq_oranges (a o : ℕ) : Prop := 4 * a = 6 * o
def oranges_eq_pears (o p : ℕ) : Prop := 5 * o = 3 * p

-- The main problem statement
theorem apples_eq_pears (a o p : ℕ) (h1 : apples_eq_oranges a o) (h2 : oranges_eq_pears o p) :
  24 * a = 21 * p :=
sorry

end apples_eq_pears_l205_205759


namespace correct_option_D_l205_205617

theorem correct_option_D (defect_rate_products : ℚ)
                         (rain_probability : ℚ)
                         (cure_rate_hospital : ℚ)
                         (coin_toss_heads_probability : ℚ)
                         (coin_toss_tails_probability : ℚ):
  defect_rate_products = 1/10 →
  rain_probability = 0.9 →
  cure_rate_hospital = 0.1 →
  coin_toss_heads_probability = 0.5 →
  coin_toss_tails_probability = 0.5 →
  coin_toss_tails_probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5
  exact h5

end correct_option_D_l205_205617


namespace actual_order_correct_l205_205185

-- Define the actual order of the students.
def actual_order := ["E", "D", "A", "C", "B"]

-- Define the first person's prediction and conditions.
def first_person_prediction := ["A", "B", "C", "D", "E"]
def first_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  (pos1 ≠ "A") ∧ (pos2 ≠ "B") ∧ (pos3 ≠ "C") ∧ (pos4 ≠ "D") ∧ (pos5 ≠ "E") ∧
  (pos1 ≠ "B") ∧ (pos2 ≠ "A") ∧ (pos2 ≠ "C") ∧ (pos3 ≠ "B") ∧ (pos3 ≠ "D") ∧
  (pos4 ≠ "C") ∧ (pos4 ≠ "E") ∧ (pos5 ≠ "D")

-- Define the second person's prediction and conditions.
def second_person_prediction := ["D", "A", "E", "C", "B"]
def second_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  ((pos1 = "D") ∨ (pos2 = "D") ∨ (pos3 = "D") ∨ (pos4 = "D") ∨ (pos5 = "D")) ∧
  ((pos1 = "A") ∨ (pos2 = "A") ∨ (pos3 = "A") ∨ (pos4 = "A") ∨ (pos5 = "A")) ∧
  (pos1 ≠ "D" ∨ pos2 ≠ "A") ∧ (pos2 ≠ "A" ∨ pos3 ≠ "E") ∧ (pos3 ≠ "E" ∨ pos4 ≠ "C") ∧ (pos4 ≠ "C" ∨ pos5 ≠ "B")

-- The theorem to prove the actual order.
theorem actual_order_correct :
  ∃ (pos1 pos2 pos3 pos4 pos5 : String),
    first_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    second_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    [pos1, pos2, pos3, pos4, pos5] = actual_order :=
by sorry

end actual_order_correct_l205_205185


namespace line_equation_l205_205409

-- Define the points A and M
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 3 1
def M := Point.mk 4 (-3)

def symmetric_point (A M : Point) : Point :=
  Point.mk (2 * M.x - A.x) (2 * M.y - A.y)

def line_through_origin (B : Point) : Prop :=
  7 * B.x + 5 * B.y = 0

theorem line_equation (B : Point) (hB : B = symmetric_point A M) : line_through_origin B :=
  by
  sorry

end line_equation_l205_205409


namespace championship_outcomes_l205_205550

theorem championship_outcomes (students events : ℕ) (h_students : students = 3) (h_events : events = 2) : 
  students ^ events = 9 :=
by
  rw [h_students, h_events]
  have h : 3 ^ 2 = 9 := by norm_num
  exact h

end championship_outcomes_l205_205550


namespace xyz_leq_36_l205_205069

theorem xyz_leq_36 {x y z : ℝ} 
    (hx0 : x > 0) (hy0 : y > 0) (hz0 : z > 0) 
    (hx2 : x ≤ 2) (hy3 : y ≤ 3) 
    (hxyz_sum : x + y + z = 11) : 
    x * y * z ≤ 36 := 
by
  sorry

end xyz_leq_36_l205_205069


namespace problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l205_205991

-- Problem G6.1
theorem problem_G6_1 : (21 ^ 3 - 11 ^ 3) / (21 ^ 2 + 21 * 11 + 11 ^ 2) = 10 := 
  sorry

-- Problem G6.2
theorem problem_G6_2 (p q : ℕ) (h1 : (p : ℚ) * 6 = 4 * (q : ℚ)) : q = 3 * p / 2 := 
  sorry

-- Problem G6.3
theorem problem_G6_3 (q r : ℕ) (h1 : q % 7 = 3) (h2 : r % 7 = 5) (h3 : 18 < r) (h4 : r < 26) : r = 24 := 
  sorry

-- Problem G6.4
def star (a b : ℕ) : ℕ := a * b + 1

theorem problem_G6_4 : star (star 3 4) 2 = 27 := 
  sorry

end problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l205_205991


namespace rectangle_area_l205_205678

theorem rectangle_area (y : ℝ) (w : ℝ) : 
  (3 * w) ^ 2 + w ^ 2 = y ^ 2 → 
  3 * w * w = (3 / 10) * y ^ 2 :=
by
  intro h
  sorry

end rectangle_area_l205_205678


namespace positive_difference_of_b_l205_205929

def g (n : Int) : Int :=
  if n < 0 then n^2 + 3 else 2 * n - 25

theorem positive_difference_of_b :
  let s := g (-3) + g 3
  let t b := g b = -s
  ∃ a b, t a ∧ t b ∧ a ≠ b ∧ |a - b| = 18 :=
by
  sorry

end positive_difference_of_b_l205_205929


namespace average_output_assembly_line_l205_205189

theorem average_output_assembly_line (initial_cogs second_batch_cogs rate1 rate2 : ℕ) (time1 time2 : ℚ)
  (h1 : initial_cogs = 60)
  (h2 : second_batch_cogs = 60)
  (h3 : rate1 = 90)
  (h4 : rate2 = 60)
  (h5 : time1 = 60 / 90)
  (h6 : time2 = 60 / 60)
  (h7 : (120 : ℚ) / (time1 + time2) = (72 : ℚ)) :
  (120 : ℚ) / (time1 + time2) = 72 := by
  sorry

end average_output_assembly_line_l205_205189


namespace second_quadrant_necessary_not_sufficient_l205_205428

variable (α : ℝ) -- Assuming α is a real number for generality.

-- Define what it means for an angle to be in the second quadrant (90° < α < 180°).
def in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

-- Define what it means for an angle to be obtuse (90° < α ≤ 180°).
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α ≤ 180

-- State the theorem to prove: 
-- "The angle α is in the second quadrant" is a necessary but not sufficient condition for "α is an obtuse angle".
theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → in_second_quadrant α) ∧ 
  (∃ α, in_second_quadrant α ∧ ¬is_obtuse α) :=
sorry

end second_quadrant_necessary_not_sufficient_l205_205428


namespace no_prime_degree_measure_l205_205221

theorem no_prime_degree_measure :
  ∀ n, 10 ≤ n ∧ n < 20 → ¬ Nat.Prime (180 * (n - 2) / n) :=
by
  intros n h1 h2 
  sorry

end no_prime_degree_measure_l205_205221


namespace triangle_ABC_two_solutions_l205_205916

theorem triangle_ABC_two_solutions (x : ℝ) (h1 : x > 0) : 
  2 < x ∧ x < 2 * Real.sqrt 2 ↔
  (∃ a b B, a = x ∧ b = 2 ∧ B = Real.pi / 4 ∧ a * Real.sin B < b ∧ b < a) := by
  sorry

end triangle_ABC_two_solutions_l205_205916


namespace jackson_running_increase_l205_205964

theorem jackson_running_increase
    (initial_miles_per_day : ℕ)
    (final_miles_per_day : ℕ)
    (weeks_increasing : ℕ)
    (total_weeks : ℕ)
    (h1 : initial_miles_per_day = 3)
    (h2 : final_miles_per_day = 7)
    (h3 : weeks_increasing = 4)
    (h4 : total_weeks = 5) :
    (final_miles_per_day - initial_miles_per_day) / weeks_increasing = 1 := 
by
  -- provided steps from solution
  sorry

end jackson_running_increase_l205_205964


namespace num_digits_difference_l205_205131

-- Define the two base-10 integers
def n1 : ℕ := 150
def n2 : ℕ := 950

-- Find the number of digits in the base-2 representation of these numbers.
def num_digits_base2 (n : ℕ) : ℕ :=
  Nat.log2 n + 1

-- State the theorem
theorem num_digits_difference :
  num_digits_base2 n2 - num_digits_base2 n1 = 2 :=
by
  sorry

end num_digits_difference_l205_205131


namespace equation_one_solution_equation_two_solution_l205_205823

theorem equation_one_solution (x : ℕ) : 8 * (x + 1)^3 = 64 ↔ x = 1 := by 
  sorry

theorem equation_two_solution (x : ℤ) : (x + 1)^2 = 100 ↔ x = 9 ∨ x = -11 := by 
  sorry

end equation_one_solution_equation_two_solution_l205_205823


namespace value_of_y_l205_205027

theorem value_of_y (x y : ℕ) (h1 : x % y = 6) (h2 : (x : ℝ) / y = 6.12) : y = 50 :=
sorry

end value_of_y_l205_205027


namespace T_description_l205_205855

-- Definitions of conditions
def T (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y ≤ 9) ∨
  (y - 5 = 4 ∧ x ≤ 1) ∨
  (x + 3 = y - 5 ∧ x ≥ 1)

-- The problem statement in Lean: Prove that T describes three rays with a common point (1, 9)
theorem T_description :
  ∀ x y, T x y ↔ 
    ((x = 1 ∧ y ≤ 9) ∨
     (x ≤ 1 ∧ y = 9) ∨
     (x ≥ 1 ∧ y = x + 8)) :=
by sorry

end T_description_l205_205855


namespace money_last_weeks_l205_205849

-- Define the conditions
def dollars_mowing : ℕ := 68
def dollars_weed_eating : ℕ := 13
def dollars_per_week : ℕ := 9

-- Define the total money made
def total_dollars := dollars_mowing + dollars_weed_eating

-- State the theorem to prove the question
theorem money_last_weeks : (total_dollars / dollars_per_week) = 9 :=
by
  sorry

end money_last_weeks_l205_205849


namespace crt_solution_l205_205070

/-- Congruences from the conditions -/
def congruences : Prop :=
  ∃ x : ℤ, 
    (x % 2 = 1) ∧
    (x % 3 = 2) ∧
    (x % 5 = 3) ∧
    (x % 7 = 4)

/-- The target result from the Chinese Remainder Theorem -/
def target_result : Prop :=
  ∃ x : ℤ, 
    (x % 210 = 53)

/-- The proof problem stating that the given conditions imply the target result -/
theorem crt_solution : congruences → target_result :=
by
  sorry

end crt_solution_l205_205070


namespace value_of_expression_l205_205261

theorem value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : (12 * y - 4)^2 = 80 := 
by 
  sorry

end value_of_expression_l205_205261


namespace boy_to_total_ratio_l205_205442

-- Problem Definitions
variables (b g : ℕ) -- number of boys and number of girls

-- Hypothesis: The probability of choosing a boy is (4/5) the probability of choosing a girl
def probability_boy := b / (b + g : ℕ)
def probability_girl := g / (b + g : ℕ)

theorem boy_to_total_ratio (h : probability_boy b g = (4 / 5) * probability_girl b g) : 
  b / (b + g : ℕ) = 4 / 9 :=
sorry

end boy_to_total_ratio_l205_205442


namespace laps_needed_l205_205643

theorem laps_needed (r1 r2 : ℕ) (laps1 : ℕ) (h1 : r1 = 30) (h2 : r2 = 10) (h3 : laps1 = 40) : 
  (r1 * laps1) / r2 = 120 := by
  sorry

end laps_needed_l205_205643


namespace problem_a5_value_l205_205939

def Sn (n : ℕ) : ℕ := 2 * n^2 + 3 * n - 1

theorem problem_a5_value : Sn 5 - Sn 4 = 21 := by
  sorry

end problem_a5_value_l205_205939


namespace rectangle_perimeter_from_square_l205_205611

theorem rectangle_perimeter_from_square (d : ℝ)
  (h : d = 6) :
  ∃ (p : ℝ), p = 12 :=
by
  sorry

end rectangle_perimeter_from_square_l205_205611


namespace percent_increase_first_quarter_l205_205626

theorem percent_increase_first_quarter (S : ℝ) (P : ℝ) :
  (S * 1.75 = (S + (P / 100) * S) * 1.346153846153846) → P = 30 :=
by
  intro h
  sorry

end percent_increase_first_quarter_l205_205626


namespace sin_alpha_plus_beta_eq_33_by_65_l205_205635

theorem sin_alpha_plus_beta_eq_33_by_65 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (hcosα : Real.cos α = 12 / 13) 
  (hcos_2α_β : Real.cos (2 * α + β) = 3 / 5) :
  Real.sin (α + β) = 33 / 65 := 
by 
  sorry

end sin_alpha_plus_beta_eq_33_by_65_l205_205635


namespace combined_stickers_l205_205573

theorem combined_stickers (k j a : ℕ) (h : 7 * j + 5 * a = 54) (hk : k = 42) (hk_ratio : k = 7 * 6) :
  j + a = 54 :=
by
  sorry

end combined_stickers_l205_205573


namespace exists_xy_l205_205396

open Classical

variable (f : ℝ → ℝ)

theorem exists_xy (h : ∃ x₀ y₀ : ℝ, f x₀ ≠ f y₀) : ∃ x y : ℝ, f (x + y) < f (x * y) :=
by
  sorry

end exists_xy_l205_205396


namespace pigeonhole_divisible_l205_205125

theorem pigeonhole_divisible (n : ℕ) (a : Fin (n + 1) → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n) :
  ∃ i j, i ≠ j ∧ a i ∣ a j :=
by
  sorry

end pigeonhole_divisible_l205_205125


namespace amount_spent_per_sibling_l205_205316

-- Definitions and conditions
def total_spent := 150
def amount_per_parent := 30
def num_parents := 2
def num_siblings := 3

-- Claim
theorem amount_spent_per_sibling :
  (total_spent - (amount_per_parent * num_parents)) / num_siblings = 30 :=
by
  sorry

end amount_spent_per_sibling_l205_205316


namespace total_points_l205_205202

theorem total_points (gwen_points_per_4 : ℕ) (lisa_points_per_5 : ℕ) (jack_points_per_7 : ℕ) 
                     (gwen_recycled : ℕ) (lisa_recycled : ℕ) (jack_recycled : ℕ)
                     (gwen_ratio : gwen_points_per_4 = 2) (lisa_ratio : lisa_points_per_5 = 3) 
                     (jack_ratio : jack_points_per_7 = 1) (gwen_pounds : gwen_recycled = 12) 
                     (lisa_pounds : lisa_recycled = 25) (jack_pounds : jack_recycled = 21) 
                     : gwen_points_per_4 * (gwen_recycled / 4) + 
                       lisa_points_per_5 * (lisa_recycled / 5) + 
                       jack_points_per_7 * (jack_recycled / 7) = 24 := by
  sorry

end total_points_l205_205202


namespace problem_mod_1000_l205_205548

noncomputable def M : ℕ := Nat.choose 18 9

theorem problem_mod_1000 : M % 1000 = 620 := by
  sorry

end problem_mod_1000_l205_205548


namespace bales_in_barn_l205_205187

theorem bales_in_barn (stacked today total original : ℕ) (h1 : stacked = 67) (h2 : total = 89) (h3 : total = stacked + original) : original = 22 :=
by
  sorry

end bales_in_barn_l205_205187


namespace ratio_is_one_third_l205_205771

-- Definitions based on given conditions
def total_students : ℕ := 90
def initial_cafeteria_students : ℕ := (2 * total_students) / 3
def initial_outside_students : ℕ := total_students - initial_cafeteria_students
def moved_cafeteria_to_outside : ℕ := 3
def final_cafeteria_students : ℕ := 67
def students_ran_inside : ℕ := final_cafeteria_students - (initial_cafeteria_students - moved_cafeteria_to_outside)

-- Ratio calculation as a proof statement
def ratio_ran_inside_to_outside : ℚ := students_ran_inside / initial_outside_students

-- Proof that the ratio is 1/3
theorem ratio_is_one_third : ratio_ran_inside_to_outside = 1 / 3 :=
by sorry -- Proof omitted

end ratio_is_one_third_l205_205771


namespace molecular_weight_CaO_l205_205084

def atomic_weight_Ca : Float := 40.08
def atomic_weight_O : Float := 16.00

def molecular_weight (atoms : List (String × Float)) : Float :=
  atoms.foldr (fun (_, w) acc => w + acc) 0.0

theorem molecular_weight_CaO :
  molecular_weight [("Ca", atomic_weight_Ca), ("O", atomic_weight_O)] = 56.08 :=
by
  sorry

end molecular_weight_CaO_l205_205084


namespace dividend_calculation_l205_205388

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 18) (h2 : quotient = 9) (h3 : remainder = 5) : 
  (divisor * quotient + remainder = 167) :=
by
  sorry

end dividend_calculation_l205_205388


namespace D_180_equals_43_l205_205217

-- Define D(n) as the number of ways to express the positive integer n
-- as a product of integers strictly greater than 1, where the order of factors matters.
def D (n : Nat) : Nat := sorry  -- The actual implementation is not provided, as per instructions.

theorem D_180_equals_43 : D 180 = 43 :=
by
  sorry  -- The proof is omitted as the task specifies.

end D_180_equals_43_l205_205217


namespace shaded_percentage_l205_205418

-- Definition for the six-by-six grid and total squares
def total_squares : ℕ := 36
def shaded_squares : ℕ := 16

-- Definition of the problem: to prove the percentage of shaded squares
theorem shaded_percentage : (shaded_squares : ℚ) / total_squares * 100 = 44.4 :=
by
  sorry

end shaded_percentage_l205_205418


namespace bike_average_speed_l205_205510

theorem bike_average_speed (distance time : ℕ)
    (h1 : distance = 48)
    (h2 : time = 6) :
    distance / time = 8 := 
  by
    sorry

end bike_average_speed_l205_205510


namespace merchant_profit_percentage_is_35_l205_205691

noncomputable def cost_price : ℝ := 100
noncomputable def markup_percentage : ℝ := 0.80
noncomputable def discount_percentage : ℝ := 0.25

-- Marked price after 80% markup
noncomputable def marked_price (cp : ℝ) (markup_pct : ℝ) : ℝ :=
  cp + (markup_pct * cp)

-- Selling price after 25% discount on marked price
noncomputable def selling_price (mp : ℝ) (discount_pct : ℝ) : ℝ :=
  mp - (discount_pct * mp)

-- Profit as the difference between selling price and cost price
noncomputable def profit (sp cp : ℝ) : ℝ :=
  sp - cp

-- Profit percentage
noncomputable def profit_percentage (profit cp : ℝ) : ℝ :=
  (profit / cp) * 100

theorem merchant_profit_percentage_is_35 :
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  profit_percentage prof cp = 35 :=
by
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  show profit_percentage prof cp = 35
  sorry

end merchant_profit_percentage_is_35_l205_205691


namespace count_valid_ks_l205_205441

theorem count_valid_ks : 
  ∃ (ks : Finset ℕ), (∀ k ∈ ks, k > 0 ∧ k ≤ 50 ∧ 
    ∀ n : ℕ, n > 0 → 7 ∣ (2 * 3^(6 * n) + k * 2^(3 * n + 1) - 1)) ∧ ks.card = 7 :=
sorry

end count_valid_ks_l205_205441


namespace find_n_values_l205_205630

-- Define a function to sum the first n consecutive natural numbers starting from k
def sum_consecutive_numbers (n k : ℕ) : ℕ :=
  n * k + (n * (n - 1)) / 2

-- Define a predicate to check if a number is a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the theorem statement
theorem find_n_values (n : ℕ) (k : ℕ) :
  is_prime (sum_consecutive_numbers n k) →
  n = 1 ∨ n = 2 :=
sorry

end find_n_values_l205_205630


namespace tobias_time_spent_at_pool_l205_205212

-- Define the conditions
def distance_per_interval : ℕ := 100
def time_per_interval : ℕ := 5
def pause_interval : ℕ := 25
def pause_time : ℕ := 5
def total_distance : ℕ := 3000
def total_time_in_hours : ℕ := 3

-- Hypotheses based on the problem conditions
def swimming_time_without_pauses := (total_distance / distance_per_interval) * time_per_interval
def number_of_pauses := (swimming_time_without_pauses / pause_interval)
def total_pause_time := number_of_pauses * pause_time
def total_time := swimming_time_without_pauses + total_pause_time

-- Proof statement
theorem tobias_time_spent_at_pool : total_time / 60 = total_time_in_hours :=
by 
  -- Put proof here
  sorry

end tobias_time_spent_at_pool_l205_205212


namespace difference_of_reciprocals_l205_205451

theorem difference_of_reciprocals (p q : ℝ) (hp : 3 / p = 6) (hq : 3 / q = 15) : p - q = 3 / 10 :=
by
  sorry

end difference_of_reciprocals_l205_205451


namespace rationalize_denominator_l205_205805

theorem rationalize_denominator :
  let A := 5
  let B := 2
  let C := 1
  let D := 4
  A + B + C + D = 12 :=
by
  sorry

end rationalize_denominator_l205_205805


namespace cosine_of_A_l205_205178

theorem cosine_of_A (a b : ℝ) (A B : ℝ) (h1 : b = (5 / 8) * a) (h2 : A = 2 * B) :
  Real.cos A = 7 / 25 :=
by
  sorry

end cosine_of_A_l205_205178


namespace find_b_l205_205449

-- Define the given hyperbola equation and conditions
def hyperbola (x y : ℝ) (b : ℝ) : Prop := x^2 - y^2 / b^2 = 1
def asymptote_line (x y : ℝ) : Prop := 2 * x - y = 0

-- State the theorem to prove
theorem find_b (b : ℝ) (hb : b > 0) :
    (∀ x y : ℝ, hyperbola x y b → asymptote_line x y) → b = 2 :=
by 
  sorry

end find_b_l205_205449


namespace third_root_of_polynomial_l205_205717

variable (a b x : ℝ)
noncomputable def polynomial := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

theorem third_root_of_polynomial (h1 : polynomial a b (-3) = 0) (h2 : polynomial a b 4 = 0) :
  ∃ r : ℝ, r = -17 / 10 ∧ polynomial a b r = 0 :=
by
  sorry

end third_root_of_polynomial_l205_205717


namespace bhanu_income_percentage_l205_205944

variable {I P : ℝ}

theorem bhanu_income_percentage (h₁ : 300 = (P / 100) * I)
                                  (h₂ : 210 = 0.3 * (I - 300)) :
  P = 30 :=
by
  sorry

end bhanu_income_percentage_l205_205944


namespace initial_pipes_num_l205_205066

variable {n : ℕ}

theorem initial_pipes_num (h1 : ∀ t : ℕ, (n * t = 8) → n = 3) (h2 : ∀ t : ℕ, (2 * t = 12) → n = 3) : n = 3 := 
by 
  sorry

end initial_pipes_num_l205_205066


namespace jen_ate_eleven_suckers_l205_205460

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end jen_ate_eleven_suckers_l205_205460


namespace number_of_ordered_pairs_l205_205878

-- Formal statement of the problem in Lean 4
theorem number_of_ordered_pairs : 
  ∃ (n : ℕ), n = 128 ∧ 
  ∀ (a b : ℝ), (∃ (x y : ℤ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 65)) ↔ n = 128 :=
sorry

end number_of_ordered_pairs_l205_205878


namespace rectangles_with_one_gray_cell_l205_205434

-- Define the number of gray cells
def gray_cells : ℕ := 40

-- Define the total rectangles containing exactly one gray cell
def total_rectangles : ℕ := 176

-- The theorem we want to prove
theorem rectangles_with_one_gray_cell (h : gray_cells = 40) : total_rectangles = 176 := 
by 
  sorry

end rectangles_with_one_gray_cell_l205_205434


namespace probability_first_genuine_on_third_test_l205_205121

noncomputable def probability_of_genuine : ℚ := 3 / 4
noncomputable def probability_of_defective : ℚ := 1 / 4
noncomputable def probability_X_eq_3 := probability_of_defective * probability_of_defective * probability_of_genuine

theorem probability_first_genuine_on_third_test :
  probability_X_eq_3 = 3 / 64 :=
by
  sorry

end probability_first_genuine_on_third_test_l205_205121


namespace richard_twice_as_old_as_scott_in_8_years_l205_205841

theorem richard_twice_as_old_as_scott_in_8_years :
  (richard_age - david_age = 6) ∧ (david_age - scott_age = 8) ∧ (david_age = 14) →
  (richard_age + 8 = 2 * (scott_age + 8)) :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end richard_twice_as_old_as_scott_in_8_years_l205_205841


namespace compare_squares_l205_205190

theorem compare_squares (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ (a + b) / 2 * (a + b) / 2 := 
sorry

end compare_squares_l205_205190


namespace fraction_beans_remain_l205_205387

theorem fraction_beans_remain (J B B_remain : ℝ) 
  (h1 : J = 0.10 * (J + B)) 
  (h2 : J + B_remain = 0.60 * (J + B)) : 
  B_remain / B = 5 / 9 := 
by 
  sorry

end fraction_beans_remain_l205_205387


namespace group_size_of_bananas_l205_205707

theorem group_size_of_bananas (totalBananas numberOfGroups : ℕ) (h1 : totalBananas = 203) (h2 : numberOfGroups = 7) :
  totalBananas / numberOfGroups = 29 :=
sorry

end group_size_of_bananas_l205_205707


namespace trajectory_of_midpoint_l205_205699

noncomputable section

open Real

-- Define the points and lines
def C : ℝ × ℝ := (-2, -2)
def A (x : ℝ) : ℝ × ℝ := (x, 0)
def B (y : ℝ) : ℝ × ℝ := (0, y)
def M (x y : ℝ) : ℝ × ℝ := ((x + 0) / 2, (0 + y) / 2)

theorem trajectory_of_midpoint (CA_dot_CB : (C.1 * (A 0).1 + (C.2 - (A 0).2)) * (C.1 * (B 0).1 + (C.2 - (B 0).2)) = 0) :
  ∀ (M : ℝ × ℝ), (M.1 = (A 0).1 / 2) ∧ (M.2 = (B 0).2 / 2) → (M.1 + M.2 + 2 = 0) :=
by
  -- here's where the proof would go
  sorry

end trajectory_of_midpoint_l205_205699


namespace how_many_pairs_of_shoes_l205_205296

theorem how_many_pairs_of_shoes (l k : ℕ) (h_l : l = 52) (h_k : k = 2) : l / k = 26 := by
  sorry

end how_many_pairs_of_shoes_l205_205296


namespace ababab_divisible_by_7_l205_205500

theorem ababab_divisible_by_7 (a b : ℕ) (ha : a < 10) (hb : b < 10) : (101010 * a + 10101 * b) % 7 = 0 :=
by sorry

end ababab_divisible_by_7_l205_205500


namespace Roja_speed_is_8_l205_205788

def Pooja_speed : ℝ := 3
def time_in_hours : ℝ := 4
def distance_between_them : ℝ := 44

theorem Roja_speed_is_8 :
  ∃ R : ℝ, R + Pooja_speed = (distance_between_them / time_in_hours) ∧ R = 8 :=
by
  sorry

end Roja_speed_is_8_l205_205788


namespace factor_polynomial_l205_205429

theorem factor_polynomial (a b m n : ℝ) (h : |m - 4| + (n^2 - 8 * n + 16) = 0) :
  a^2 + 4 * b^2 - m * a * b - n = (a - 2 * b + 2) * (a - 2 * b - 2) :=
by
  sorry

end factor_polynomial_l205_205429


namespace point_inside_circle_l205_205255

theorem point_inside_circle (a : ℝ) (h : 5 * a^2 - 4 * a - 1 < 0) : -1/5 < a ∧ a < 1 :=
    sorry

end point_inside_circle_l205_205255


namespace staffing_correct_l205_205616

-- The number of ways to staff a battle station with constraints.
def staffing_ways (total_applicants unsuitable_fraction: ℕ) (job_openings: ℕ): ℕ :=
  let suitable_candidates := total_applicants * (1 - unsuitable_fraction)
  if suitable_candidates < job_openings then
    0 
  else
    (List.range' (suitable_candidates - job_openings + 1) job_openings).prod

-- Definitions of the problem conditions
def total_applicants := 30
def unsuitable_fraction := 2/3
def job_openings := 5
-- Expected result
def expected_result := 30240

-- The theorem to prove the number of ways to staff the battle station equals the given result.
theorem staffing_correct : staffing_ways total_applicants unsuitable_fraction job_openings = expected_result := by
  sorry

end staffing_correct_l205_205616


namespace assembly_time_constants_l205_205780

theorem assembly_time_constants (a b : ℕ) (f : ℕ → ℝ)
  (h1 : ∀ x, f x = if x < b then a / (Real.sqrt x) else a / (Real.sqrt b))
  (h2 : f 4 = 15)
  (h3 : f b = 10) :
  a = 30 ∧ b = 9 :=
by
  sorry

end assembly_time_constants_l205_205780


namespace employed_population_percentage_l205_205679

theorem employed_population_percentage
  (P : ℝ) -- Total population
  (E : ℝ) -- Fraction of population that is employed
  (employed_males : ℝ) -- Fraction of population that is employed males
  (employed_females_fraction : ℝ)
  (h1 : employed_males = 0.8 * P)
  (h2 : employed_females_fraction = 1 / 3) :
  E = 0.6 :=
by
  -- We don't need the proof here.
  sorry

end employed_population_percentage_l205_205679


namespace num_ordered_triples_l205_205163

-- Given constants
def b : ℕ := 2024
def constant_value : ℕ := 4096576

-- Number of ordered triples (a, b, c) meeting the conditions
theorem num_ordered_triples (h : b = 2024 ∧ constant_value = 2024 * 2024) :
  ∃ (n : ℕ), n = 10 ∧ ∀ (a c : ℕ), a * c = constant_value → a ≤ c → n = 10 :=
by
  -- Translation of the mathematical conditions into the theorem
  sorry

end num_ordered_triples_l205_205163


namespace set_clock_correctly_l205_205554

noncomputable def correct_clock_time
  (T_depart T_arrive T_depart_friend T_return : ℕ) 
  (T_visit := T_depart_friend - T_arrive) 
  (T_return_err := T_return - T_depart) 
  (T_total_travel := T_return_err - T_visit) 
  (T_travel_oneway := T_total_travel / 2) : ℕ :=
  T_depart + T_visit + T_travel_oneway

theorem set_clock_correctly 
  (T_depart T_arrive T_depart_friend T_return : ℕ)
  (h1 : T_depart ≤ T_return) -- The clock runs without accounting for the time away
  (h2 : T_arrive ≤ T_depart_friend) -- The friend's times are correct
  (h3 : T_return ≠ T_depart) -- The man was away for some non-zero duration
: 
  (correct_clock_time T_depart T_arrive T_depart_friend T_return) = 
  (T_depart + (T_depart_friend - T_arrive) + ((T_return - T_depart - (T_depart_friend - T_arrive)) / 2)) :=
sorry

end set_clock_correctly_l205_205554


namespace books_left_l205_205005

variable (initialBooks : ℕ) (soldBooks : ℕ) (remainingBooks : ℕ)

-- Conditions
def initial_conditions := initialBooks = 136 ∧ soldBooks = 109

-- Question: Proving the remaining books after the sale
theorem books_left (initial_conditions : initialBooks = 136 ∧ soldBooks = 109) : remainingBooks = 27 :=
by
  cases initial_conditions
  sorry

end books_left_l205_205005


namespace inheritance_shares_l205_205301

theorem inheritance_shares (A B : ℝ) (h1: A + B = 100) (h2: (1/4) * B - (1/3) * A = 11) : 
  A = 24 ∧ B = 76 := 
by 
  sorry

end inheritance_shares_l205_205301


namespace y_coordinate_of_C_range_l205_205975

noncomputable def A : ℝ × ℝ := (0, 2)

def is_on_parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = P.1 + 4

def is_perpendicular (A B C : ℝ × ℝ) : Prop := 
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

def range_of_y_C (y_C : ℝ) : Prop := y_C ≤ 0 ∨ y_C ≥ 4

theorem y_coordinate_of_C_range (B C : ℝ × ℝ)
  (hB : is_on_parabola B) (hC : is_on_parabola C) (h_perpendicular : is_perpendicular A B C) : 
  range_of_y_C (C.2) :=
sorry

end y_coordinate_of_C_range_l205_205975


namespace circle_center_l205_205813

theorem circle_center (x y : ℝ) : (x^2 - 6 * x + y^2 + 2 * y = 20) → (x,y) = (3,-1) :=
by {
  sorry
}

end circle_center_l205_205813


namespace sum_of_three_distinct_integers_l205_205474

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end sum_of_three_distinct_integers_l205_205474


namespace am_gm_four_vars_l205_205532

theorem am_gm_four_vars {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / a + 1 / b + 1 / c + 1 / d) ≥ 16 :=
by
  sorry

end am_gm_four_vars_l205_205532


namespace existence_of_point_N_l205_205966

-- Given conditions
def is_point_on_ellipse (x y a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ (a^2 = b^2 + (a * (Real.sqrt 2) / 2)^2)

def passes_through_point (x y a b : ℝ) (px py : ℝ) : Prop :=
  (px^2 / a^2) + (py^2 / b^2) = 1

def ellipse_with_eccentricity (a : ℝ) : Prop :=
  (Real.sqrt 2) / 2 = (Real.sqrt (a^2 - (a * (Real.sqrt 2) / 2)^2)) / a

def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

def lines_intersect_ellipse (k a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b

def angle_condition (k t a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b ∧ 
  ((y1 - t) / x1) + ((y2 - t) / x2) = 0

-- Lean 4 statement
theorem existence_of_point_N (a b k t : ℝ) (hx : is_ellipse a b) (hp : passes_through_point 2 (Real.sqrt 2) a b 2 (Real.sqrt 2)) (he : ellipse_with_eccentricity a) (hl : ∀ (x1 y1 x2 y2 : ℝ), lines_intersect_ellipse k a b) :
  ∃ (N : ℝ), N = 4 ∧ angle_condition k N a b :=
sorry

end existence_of_point_N_l205_205966


namespace sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l205_205443

open Real

namespace TriangleProofs

variables 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (BA BC : ℝ) 
  (h1 : sin B = sqrt 7 / 4) 
  (h2 : (cos A / sin A + cos C / sin C = 4 * sqrt 7 / 7)) 
  (h3 : BA * BC = 3 / 2)
  (h4 : a = b ∧ c = b)

-- 1. Prove that sin A * sin C = sin^2 B
theorem sin_a_mul_sin_c_eq_sin_sq_b : sin A * sin C = sin B ^ 2 := 
by sorry

-- 2. Prove that 0 < B ≤ π / 3
theorem zero_lt_B_le_pi_div_3 : 0 < B ∧ B ≤ π / 3 := 
by sorry

-- 3. Find the magnitude of the vector sum.
theorem magnitude_BC_add_BA : abs (BC + BA) = 2 * sqrt 2 := 
by sorry

end TriangleProofs

end sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l205_205443


namespace negation_proposition_l205_205064

theorem negation_proposition (x : ℝ) : ¬(∀ x, x > 0 → x^2 > 0) ↔ ∃ x, x > 0 ∧ x^2 ≤ 0 :=
by
  sorry

end negation_proposition_l205_205064


namespace solution_set_inequality_l205_205638

theorem solution_set_inequality :
  {x : ℝ | (x^2 - 4) * (x - 6)^2 ≤ 0} = {x : ℝ | (-2 ≤ x ∧ x ≤ 2) ∨ x = 6} :=
  sorry

end solution_set_inequality_l205_205638


namespace heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l205_205777

namespace PolygonColoring

/-- Define a regular n-gon and its coloring -/
def regular_ngon (n : ℕ) : Type := sorry

def isosceles_triangle {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

def same_color {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

/-- Part (a) statement -/
theorem heptagon_isosceles_triangle_same_color : 
  ∀ (p : regular_ngon 7), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (b) statement -/
theorem octagon_no_isosceles_triangle_same_color :
  ∃ (p : regular_ngon 8), ¬∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (c) statement -/
theorem general_ngon_isosceles_triangle_same_color :
  ∀ (n : ℕ), (n = 5 ∨ n = 7 ∨ n ≥ 9) → 
  ∀ (p : regular_ngon n), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

end PolygonColoring

end heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l205_205777


namespace kelly_sony_games_solution_l205_205752

def kelly_sony_games_left (n g : Nat) : Nat :=
  n - g

theorem kelly_sony_games_solution (initial : Nat) (given_away : Nat) 
  (h_initial : initial = 132)
  (h_given_away : given_away = 101) :
  kelly_sony_games_left initial given_away = 31 :=
by
  rw [h_initial, h_given_away]
  unfold kelly_sony_games_left
  norm_num

end kelly_sony_games_solution_l205_205752


namespace problem_l205_205815

noncomputable def cubeRoot (x : ℝ) : ℝ :=
  x ^ (1 / 3)

theorem problem (t : ℝ) (h : t = 1 / (1 - cubeRoot 2)) :
  t = (1 + cubeRoot 2) * (1 + cubeRoot 4) :=
by
  sorry

end problem_l205_205815


namespace correct_exponentiation_l205_205749

theorem correct_exponentiation (a : ℝ) :
  (a^2 * a^3 = a^5) ∧
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 + a^2 ≠ a^4) ∧
  (3 * a^3 - a^2 ≠ 2 * a) :=
by
  sorry

end correct_exponentiation_l205_205749


namespace integer_count_of_sqrt_x_l205_205753

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l205_205753


namespace integer_values_of_a_l205_205736

variable (a b c x : ℤ)

theorem integer_values_of_a (h : (x - a) * (x - 12) + 4 = (x + b) * (x + c)) : a = 7 ∨ a = 17 := by
  sorry

end integer_values_of_a_l205_205736


namespace regular_polygon_perimeter_l205_205154

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l205_205154


namespace part1_part2_l205_205184

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → a ≤ 1 := sorry

theorem part2 (a : ℝ) (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  (f x₁ a - f x₂ a) / (x₂ - x₁) < 1 / (x₁ * (x₁ + 1)) := sorry

end part1_part2_l205_205184


namespace reporter_earnings_per_hour_l205_205297

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end reporter_earnings_per_hour_l205_205297


namespace overtime_hours_correct_l205_205625

def regular_pay_rate : ℕ := 3
def max_regular_hours : ℕ := 40
def total_pay_received : ℕ := 192
def overtime_pay_rate : ℕ := 2 * regular_pay_rate
def regular_earnings : ℕ := regular_pay_rate * max_regular_hours
def additional_earnings : ℕ := total_pay_received - regular_earnings
def calculated_overtime_hours : ℕ := additional_earnings / overtime_pay_rate

theorem overtime_hours_correct :
  calculated_overtime_hours = 12 :=
by
  sorry

end overtime_hours_correct_l205_205625


namespace largest_and_smallest_A_l205_205141

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l205_205141


namespace tom_gas_spending_l205_205801

-- Defining the conditions given in the problem
def miles_per_gallon := 50
def miles_per_day := 75
def gas_price := 3
def number_of_days := 10

-- Defining the main theorem to be proven
theorem tom_gas_spending : 
  (miles_per_day * number_of_days) / miles_per_gallon * gas_price = 45 := 
by 
  sorry

end tom_gas_spending_l205_205801


namespace sum_of_first_nine_terms_l205_205304

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a n = a 1 + d * (n - 1)

variables (a : ℕ → ℝ) (h_seq : arithmetic_sequence a)

-- Given condition: a₂ + a₃ + a₇ + a₈ = 20
def condition : Prop := a 2 + a 3 + a 7 + a 8 = 20

-- Statement: Prove that the sum of the first 9 terms is 45
theorem sum_of_first_nine_terms (h : condition a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 45 :=
by sorry

end sum_of_first_nine_terms_l205_205304


namespace range_of_a_l205_205665

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 8

def resolution (a : ℝ) : Prop :=
(p a ∨ q a) ∧ ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1 / 8) ∨ a ≥ 1

theorem range_of_a (a : ℝ) : resolution a := sorry

end range_of_a_l205_205665


namespace total_flowers_sold_l205_205676

-- Definitions for conditions
def roses_per_bouquet : ℕ := 12
def daisies_per_bouquet : ℕ := 12  -- Assuming each daisy bouquet contains the same number of daisies as roses
def total_bouquets : ℕ := 20
def rose_bouquets_sold : ℕ := 10
def daisy_bouquets_sold : ℕ := 10

-- Statement of the equivalent Lean theorem
theorem total_flowers_sold :
  (rose_bouquets_sold * roses_per_bouquet) + (daisy_bouquets_sold * daisies_per_bouquet) = 240 :=
by
  sorry

end total_flowers_sold_l205_205676


namespace interval_1_5_frequency_is_0_70_l205_205560

-- Define the intervals and corresponding frequencies
def intervals : List (ℤ × ℤ) := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

def frequencies : List ℕ := [1, 1, 2, 3, 1, 2]

-- Sample capacity
def sample_capacity : ℕ := 10

-- Calculate the frequency of the sample in the interval [1,5)
noncomputable def frequency_in_interval_1_5 : ℝ := (frequencies.take 4).sum / sample_capacity

-- Prove that the frequency in the interval [1,5) is 0.70
theorem interval_1_5_frequency_is_0_70 : frequency_in_interval_1_5 = 0.70 := by
  sorry

end interval_1_5_frequency_is_0_70_l205_205560


namespace polynomial_divisibility_l205_205885

theorem polynomial_divisibility (a b x y : ℤ) : 
  ∃ k : ℤ, (a * x + b * y)^3 + (b * x + a * y)^3 = k * (a + b) * (x + y) := by
  sorry

end polynomial_divisibility_l205_205885


namespace closest_multiple_of_18_2021_l205_205959

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def closest_multiple_of (n k : ℕ) : ℕ :=
if (n % k) * 2 < k then n - (n % k) else n + (k - n % k)

theorem closest_multiple_of_18_2021 :
  closest_multiple_of 2021 18 = 2016 := by
    sorry

end closest_multiple_of_18_2021_l205_205959


namespace output_value_is_16_l205_205620

def f (x : ℤ) : ℤ :=
  if x < 0 then (x + 1) * (x + 1) else (x - 1) * (x - 1)

theorem output_value_is_16 : f 5 = 16 := by
  sorry

end output_value_is_16_l205_205620


namespace parity_equiv_l205_205286

open Nat

theorem parity_equiv (p q : ℕ) : (Even (p^3 - q^3) ↔ Even (p + q)) :=
by
  sorry

end parity_equiv_l205_205286


namespace employed_males_percent_l205_205014

variable (population : ℝ) (percent_employed : ℝ) (percent_employed_females : ℝ)

theorem employed_males_percent :
  percent_employed = 120 →
  percent_employed_females = 33.33333333333333 →
  2 / 3 * percent_employed = 80 :=
by
  intros h1 h2
  sorry

end employed_males_percent_l205_205014


namespace length_of_PW_l205_205048

-- Given variables
variables (CD WX DP PX : ℝ) (CW : ℝ)

-- Condition 1: CD is parallel to WX
axiom h1 : true -- Parallelism is given as part of the problem

-- Condition 2: CW = 60 units
axiom h2 : CW = 60

-- Condition 3: DP = 18 units
axiom h3 : DP = 18

-- Condition 4: PX = 36 units
axiom h4 : PX = 36

-- Question/Answer: Prove that the length of PW = 40 units
theorem length_of_PW (PW CP : ℝ) (h5 : CP = PW / 2) (h6 : CW = CP + PW) : PW = 40 :=
by sorry

end length_of_PW_l205_205048


namespace probability_X_interval_l205_205869

noncomputable def fx (x c : ℝ) : ℝ :=
  if -c ≤ x ∧ x ≤ c then (1 / c) * (1 - (|x| / c))
  else 0

theorem probability_X_interval (c : ℝ) (hc : 0 < c) :
  (∫ x in (c / 2)..c, fx x c) = 1 / 8 :=
sorry

end probability_X_interval_l205_205869


namespace largest_whole_number_l205_205887

theorem largest_whole_number (n : ℤ) (h : (1 : ℝ) / 4 + n / 8 < 2) : n ≤ 13 :=
by {
  sorry
}

end largest_whole_number_l205_205887


namespace percent_non_sugar_l205_205105

-- Definitions based on the conditions in the problem.
def pie_weight : ℕ := 200
def sugar_weight : ℕ := 50

-- Statement of the proof problem.
theorem percent_non_sugar : ((pie_weight - sugar_weight) * 100) / pie_weight = 75 :=
by
  sorry

end percent_non_sugar_l205_205105


namespace inequality_l205_205553

variable (a b c : ℝ)

noncomputable def condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 / 8

theorem inequality (h : condition a b c) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 :=
sorry

end inequality_l205_205553


namespace smallest_unreachable_integer_l205_205997

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end smallest_unreachable_integer_l205_205997


namespace james_pays_per_episode_l205_205130

-- Conditions
def minor_characters : ℕ := 4
def major_characters : ℕ := 5
def pay_per_minor_character : ℕ := 15000
def multiplier_major_payment : ℕ := 3

-- Theorems and Definitions needed
def pay_per_major_character : ℕ := pay_per_minor_character * multiplier_major_payment
def total_pay_minor : ℕ := minor_characters * pay_per_minor_character
def total_pay_major : ℕ := major_characters * pay_per_major_character
def total_pay_per_episode : ℕ := total_pay_minor + total_pay_major

-- Main statement to prove
theorem james_pays_per_episode : total_pay_per_episode = 285000 := by
  sorry

end james_pays_per_episode_l205_205130


namespace question1_question2_l205_205053

-- Definitions:
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Question 1 Statement:
theorem question1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := by
  sorry

-- Question 2 Statement:
theorem question2 (a : ℝ) (h : A ∪ B a = A) : a > 3 := by
  sorry

end question1_question2_l205_205053


namespace find_angle_B_l205_205664

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {m n : ℝ × ℝ}
variable (h1 : m = (Real.cos A, Real.sin A))
variable (h2 : n = (1, Real.sqrt 3))
variable (h3 : m.1 / n.1 = m.2 / n.2)
variable (h4 : a * Real.cos B + b * Real.cos A = c * Real.sin C)

theorem find_angle_B (h_conditions : a * Real.cos B + b * Real.cos A = c * Real.sin C) : B = Real.pi / 6 :=
sorry

end find_angle_B_l205_205664


namespace triangle_angle_identity_l205_205582

def triangle_angles_arithmetic_sequence (A B C : ℝ) : Prop :=
  A + C = 2 * B

def sum_of_triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = 180

def angle_B_is_60 (B : ℝ) : Prop :=
  B = 60

theorem triangle_angle_identity (A B C a b c : ℝ)
  (h1 : triangle_angles_arithmetic_sequence A B C)
  (h2 : sum_of_triangle_angles A B C)
  (h3 : angle_B_is_60 B) : 
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) :=
by 
  sorry

end triangle_angle_identity_l205_205582


namespace lowest_position_l205_205585

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l205_205585


namespace distance_is_12_l205_205477

def distance_to_Mount_Overlook (D : ℝ) : Prop :=
  let T1 := D / 4
  let T2 := D / 6
  T1 + T2 = 5

theorem distance_is_12 : ∃ D : ℝ, distance_to_Mount_Overlook D ∧ D = 12 :=
by
  use 12
  rw [distance_to_Mount_Overlook]
  sorry

end distance_is_12_l205_205477


namespace exists_pentagon_from_midpoints_l205_205858

noncomputable def pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) : Prop :=
  ∃ (A B C D E : ℝ × ℝ), 
    (A1 = (A + B) / 2) ∧ 
    (B1 = (B + C) / 2) ∧ 
    (C1 = (C + D) / 2) ∧ 
    (D1 = (D + E) / 2) ∧ 
    (E1 = (E + A) / 2)

-- statement of the theorem
theorem exists_pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) :
  pentagon_from_midpoints A1 B1 C1 D1 E1 :=
sorry

end exists_pentagon_from_midpoints_l205_205858


namespace cafeteria_extra_fruit_l205_205119

theorem cafeteria_extra_fruit 
    (red_apples : ℕ)
    (green_apples : ℕ)
    (students : ℕ)
    (total_apples := red_apples + green_apples)
    (apples_taken := students)
    (extra_apples := total_apples - apples_taken)
    (h1 : red_apples = 42)
    (h2 : green_apples = 7)
    (h3 : students = 9) :
    extra_apples = 40 := 
by 
  sorry

end cafeteria_extra_fruit_l205_205119


namespace max_profit_at_300_l205_205128

-- Define the cost and revenue functions and total profit function

noncomputable def cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 400 * x else 90090

noncomputable def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 300 * x - 20000 else -100 * x + 70090

-- The Lean statement for proving maximum profit occurs at x = 300
theorem max_profit_at_300 : ∀ x : ℝ, profit x ≤ profit 300 :=
sorry

end max_profit_at_300_l205_205128


namespace binomial_evaluation_l205_205743

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l205_205743


namespace matt_days_alone_l205_205705

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem matt_days_alone (M P : ℝ) (h1 : work_rate M + work_rate P = work_rate 20) 
  (h2 : 1 - 12 * (work_rate M + work_rate P) = 2 / 5) 
  (h3 : 10 * work_rate M = 2 / 5) : M = 25 :=
by
  sorry

end matt_days_alone_l205_205705


namespace part1_part2_l205_205328

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x m : ℝ) : ℝ := -|x + 3| + m

def solution_set_ineq_1 (a : ℝ) : Set ℝ :=
  if a = 1 then {x | x < 2 ∨ x > 2}
  else if a > 1 then Set.univ
  else {x | x < 1 + a ∨ x > 3 - a}

theorem part1 (a : ℝ) : 
  ∃ S : Set ℝ, S = solution_set_ineq_1 a ∧ ∀ x : ℝ, (f x + a - 1 > 0) ↔ x ∈ S := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := sorry

end part1_part2_l205_205328


namespace base9_to_base10_l205_205383

def num_base9 : ℕ := 521 -- Represents 521_9
def base : ℕ := 9

theorem base9_to_base10 : 
  (1 * base^0 + 2 * base^1 + 5 * base^2) = 424 := 
by
  -- Sorry allows us to skip the proof.
  sorry

end base9_to_base10_l205_205383


namespace triangle_ratio_l205_205704

theorem triangle_ratio
  (D E F X : Type)
  [DecidableEq D] [DecidableEq E] [DecidableEq F] [DecidableEq X]
  (DE DF : ℝ)
  (hDE : DE = 36)
  (hDF : DF = 40)
  (DX_bisects_EDF : ∀ EX FX, (DE * FX = DF * EX)) :
  ∃ (EX FX : ℝ), EX / FX = 9 / 10 :=
sorry

end triangle_ratio_l205_205704


namespace subset_eq_possible_sets_of_B_l205_205864

theorem subset_eq_possible_sets_of_B (B : Set ℕ) 
  (h1 : {1, 2} ⊆ B)
  (h2 : B ⊆ {1, 2, 3, 4}) :
  B = {1, 2} ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end subset_eq_possible_sets_of_B_l205_205864


namespace arctan_sum_of_roots_l205_205395

theorem arctan_sum_of_roots (u v w : ℝ) (h1 : u + v + w = 0) (h2 : u * v + v * w + w * u = -10) (h3 : u * v * w = -11) :
  Real.arctan u + Real.arctan v + Real.arctan w = π / 4 :=
by
  sorry

end arctan_sum_of_roots_l205_205395


namespace square_of_number_l205_205954

theorem square_of_number (x : ℝ) (h : 2 * x = x / 5 + 9) : x^2 = 25 := 
sorry

end square_of_number_l205_205954


namespace left_square_side_length_l205_205096

theorem left_square_side_length 
  (x y z : ℝ)
  (H1 : y = x + 17)
  (H2 : z = x + 11)
  (H3 : x + y + z = 52) : 
  x = 8 := by
  sorry

end left_square_side_length_l205_205096


namespace simplify_and_evaluate_expr_l205_205980

noncomputable def original_expr (x : ℝ) : ℝ := 
  ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))

noncomputable def x_val : ℝ := Real.sqrt 2 - 1

theorem simplify_and_evaluate_expr : original_expr x_val = 1 - (Real.sqrt 2) / 2 :=
  by
    sorry

end simplify_and_evaluate_expr_l205_205980


namespace total_arrangements_l205_205834

-- Question: 
-- Given 6 teachers and 4 schools with specific constraints, 
-- prove that the number of different ways to arrange the teachers is 240.

def teachers : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

def schools : List Nat := [1, 2, 3, 4]

def B_and_D_in_same_school (assignment: Char → Nat) : Prop :=
  assignment 'B' = assignment 'D'

def each_school_has_at_least_one_teacher (assignment: Char → Nat) : Prop :=
  ∀ s ∈ schools, ∃ t ∈ teachers, assignment t = s

noncomputable def num_arrangements : Nat := sorry -- This would actually involve complex combinatorial calculations

theorem total_arrangements : num_arrangements = 240 :=
  sorry

end total_arrangements_l205_205834


namespace intersection_correct_l205_205985

open Set

def M : Set ℤ := {-1, 3, 5}
def N : Set ℤ := {-1, 0, 1, 2, 3}
def MN_intersection : Set ℤ := {-1, 3}

theorem intersection_correct : M ∩ N = MN_intersection := by
  sorry

end intersection_correct_l205_205985


namespace option_A_incorrect_l205_205043

theorem option_A_incorrect {a b m : ℤ} (h : am = bm) : m = 0 ∨ a = b :=
by sorry

end option_A_incorrect_l205_205043


namespace perimeter_of_plot_l205_205524

theorem perimeter_of_plot
  (width : ℝ) 
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : cost_per_meter = 6.5)
  (h2 : total_cost = 1170)
  (h3 : total_cost = (2 * (width + (width + 10))) * cost_per_meter) 
  :
  (2 * ((width + 10) + width)) = 180 :=
by
  sorry

end perimeter_of_plot_l205_205524


namespace inequality_solution_l205_205232

theorem inequality_solution (x : ℝ) : (2 * x - 3 < x + 1) -> (x < 4) :=
by
  intro h
  sorry

end inequality_solution_l205_205232


namespace rotate_right_triangle_along_right_angle_produces_cone_l205_205003

-- Define a right triangle and the conditions for its rotation
structure RightTriangle (α β γ : ℝ) :=
  (zero_angle : α = 0)
  (ninety_angle_1 : β = 90)
  (ninety_angle_2 : γ = 90)
  (sum_180 : α + β + γ = 180)

-- Define the theorem for the resulting shape when rotating the right triangle
theorem rotate_right_triangle_along_right_angle_produces_cone
  (T : RightTriangle α β γ) (line_of_rotation_contains_right_angle : α = 90 ∨ β = 90 ∨ γ = 90) :
  ∃ shape, shape = "cone" :=
sorry

end rotate_right_triangle_along_right_angle_produces_cone_l205_205003


namespace circle_equation_correct_l205_205338

-- Define the given elements: center and radius
def center : (ℝ × ℝ) := (1, -1)
def radius : ℝ := 2

-- Define the equation of the circle with the given center and radius
def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = radius^2

-- Prove that the equation of the circle holds with the given center and radius
theorem circle_equation_correct : 
  ∀ x y : ℝ, circle_eqn x y ↔ (x - 1)^2 + (y + 1)^2 = 4 := 
by
  sorry

end circle_equation_correct_l205_205338


namespace tangent_line_on_x_axis_l205_205176

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1/4

theorem tangent_line_on_x_axis (x0 a : ℝ) (h1: f x0 a = 0) (h2: (3 * x0^2 + a) = 0) : a = -3/4 :=
by sorry

end tangent_line_on_x_axis_l205_205176


namespace larger_of_two_numbers_l205_205733

theorem larger_of_two_numbers (A B : ℕ) (hcf : A.gcd B = 47) (lcm_factors : A.lcm B = 47 * 49 * 11 * 13 * 4913) : max A B = 123800939 :=
sorry

end larger_of_two_numbers_l205_205733


namespace percentage_error_in_area_l205_205886

theorem percentage_error_in_area (s : ℝ) (x : ℝ) (h₁ : s' = 1.08 * s) 
  (h₂ : s^2 = (2 * A)) (h₃ : x^2 = (2 * A)) : 
  (abs ((1.1664 * s^2 - s^2) / s^2 * 100) - 17) ≤ 0.5 := 
sorry

end percentage_error_in_area_l205_205886


namespace number_of_terms_geometric_seq_l205_205760

-- Given conditions
variables (a1 q : ℝ)  -- First term and common ratio of the sequence
variable  (n : ℕ)     -- Number of terms in the sequence

-- The product of the first three terms
axiom condition1 : a1^3 * q^3 = 3

-- The product of the last three terms
axiom condition2 : a1^3 * q^(3 * n - 6) = 9

-- The product of all terms
axiom condition3 : a1^n * q^(n * (n - 1) / 2) = 729

-- Proving the number of terms in the sequence
theorem number_of_terms_geometric_seq : n = 12 := by
  sorry

end number_of_terms_geometric_seq_l205_205760


namespace trajectory_of_point_A_l205_205278

theorem trajectory_of_point_A (m : ℝ) (A B C : ℝ × ℝ) (hBC : B = (-1, 0) ∧ C = (1, 0)) (hBC_dist : dist B C = 2)
  (hRatio : dist A B / dist A C = m) :
  (m = 1 → ∀ x y : ℝ, A = (x, y) → x = 0) ∧
  (m = 0 → ∀ x y : ℝ, A = (x, y) → x^2 + y^2 - 2 * x + 1 = 0) ∧
  (m ≠ 0 ∧ m ≠ 1 → ∀ x y : ℝ, A = (x, y) → (x + (1 + m^2) / (1 - m^2))^2 + y^2 = (2 * m / (1 - m^2))^2) := 
sorry

end trajectory_of_point_A_l205_205278


namespace total_purchase_cost_l205_205764

-- Definitions for the quantities of the items
def quantity_chocolate_bars : ℕ := 10
def quantity_gummy_bears : ℕ := 10
def quantity_chocolate_chips : ℕ := 20

-- Definitions for the costs of the items
def cost_per_chocolate_bar : ℕ := 3
def cost_per_gummy_bear_pack : ℕ := 2
def cost_per_chocolate_chip_bag : ℕ := 5

-- Proof statement to be shown
theorem total_purchase_cost :
  (quantity_chocolate_bars * cost_per_chocolate_bar) + 
  (quantity_gummy_bears * cost_per_gummy_bear_pack) + 
  (quantity_chocolate_chips * cost_per_chocolate_chip_bag) = 150 :=
sorry

end total_purchase_cost_l205_205764


namespace solve_math_problem_l205_205810

theorem solve_math_problem (x : ℕ) (h1 : x > 0) (h2 : x % 3 = 0) (h3 : x % x = 9) : x = 30 := by
  sorry

end solve_math_problem_l205_205810


namespace division_of_fractions_l205_205262

theorem division_of_fractions : (1 / 10) / (1 / 5) = 1 / 2 :=
by
  sorry

end division_of_fractions_l205_205262


namespace bees_on_second_day_l205_205948

-- Define the number of bees on the first day
def bees_first_day : ℕ := 144

-- Define the multiplication factor
def multiplication_factor : ℕ := 3

-- Define the number of bees on the second day
def bees_second_day : ℕ := bees_first_day * multiplication_factor

-- Theorem stating the number of bees on the second day is 432
theorem bees_on_second_day : bees_second_day = 432 := 
by
  sorry

end bees_on_second_day_l205_205948


namespace sum_ai_le_sum_bi_l205_205227

open BigOperators

variable {α : Type*} [LinearOrderedField α]

theorem sum_ai_le_sum_bi {n : ℕ} {a b : Fin n → α}
  (h1 : ∀ i, 0 < a i)
  (h2 : ∀ i, 0 < b i)
  (h3 : ∑ i, (a i)^2 / b i ≤ ∑ i, b i) :
  ∑ i, a i ≤ ∑ i, b i :=
sorry

end sum_ai_le_sum_bi_l205_205227


namespace time_to_cross_signal_post_l205_205137

-- Definition of the conditions
def length_of_train : ℝ := 600  -- in meters
def time_to_cross_bridge : ℝ := 8  -- in minutes
def length_of_bridge : ℝ := 7200  -- in meters

-- Equivalent statement
theorem time_to_cross_signal_post (constant_speed : ℝ) (t : ℝ) 
  (h1 : constant_speed * t = length_of_train) 
  (h2 : constant_speed * time_to_cross_bridge = length_of_train + length_of_bridge) : 
  t * 60 = 36.9 := 
sorry

end time_to_cross_signal_post_l205_205137


namespace k_even_l205_205591

theorem k_even (n a b k : ℕ) (h1 : 2^n - 1 = a * b) (h2 : 2^k ∣ 2^(n-2) + a - b):
  k % 2 = 0 :=
sorry

end k_even_l205_205591


namespace find_g_five_l205_205060

theorem find_g_five 
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : g 0 = 1) : g 5 = Real.exp 5 :=
sorry

end find_g_five_l205_205060


namespace total_apples_eaten_l205_205281

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l205_205281


namespace tangent_line_ellipse_l205_205324

theorem tangent_line_ellipse (x y : ℝ) (h : 2^2 / 8 + 1^2 / 2 = 1) :
    x / 4 + y / 2 = 1 := 
  sorry

end tangent_line_ellipse_l205_205324


namespace scientific_notation_l205_205408

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end scientific_notation_l205_205408


namespace prob_two_red_balls_consecutively_without_replacement_l205_205062

def numOfRedBalls : ℕ := 3
def totalNumOfBalls : ℕ := 8

theorem prob_two_red_balls_consecutively_without_replacement :
  (numOfRedBalls / totalNumOfBalls) * ((numOfRedBalls - 1) / (totalNumOfBalls - 1)) = 3 / 28 :=
by
  sorry

end prob_two_red_balls_consecutively_without_replacement_l205_205062


namespace jenna_weight_lift_l205_205373

theorem jenna_weight_lift:
  ∀ (n : Nat), (2 * 10 * 25 = 500) ∧ (15 * n >= 500) ∧ (n = Nat.ceil (500 / 15 : ℝ))
  → n = 34 := 
by
  intros n h
  have h₀ : 2 * 10 * 25 = 500 := h.1
  have h₁ : 15 * n >= 500 := h.2.1
  have h₂ : n = Nat.ceil (500 / 15 : ℝ) := h.2.2
  sorry

end jenna_weight_lift_l205_205373


namespace burger_meal_cost_l205_205160

-- Define the conditions
variables (B S : ℝ)
axiom cost_of_soda : S = (1 / 3) * B
axiom total_cost : B + S + 2 * (B + S) = 24

-- Prove that the cost of the burger meal is $6
theorem burger_meal_cost : B = 6 :=
by {
  -- We'll use both the axioms provided to show B equals 6
  sorry
}

end burger_meal_cost_l205_205160


namespace set_B_equals_1_4_l205_205653

open Set

def U : Set ℕ := {1, 2, 3, 4}
def C_U_B : Set ℕ := {2, 3}

theorem set_B_equals_1_4 : 
  ∃ B : Set ℕ, B = {1, 4} ∧ U \ B = C_U_B := by
  sorry

end set_B_equals_1_4_l205_205653


namespace range_of_a_proof_l205_205519

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

theorem range_of_a_proof (a : ℝ) : range_of_a a ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_proof_l205_205519


namespace units_digit_of_27_mul_36_l205_205831

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l205_205831


namespace value_of_expression_l205_205351

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem value_of_expression (h : a = Real.log 3 / Real.log 4) : 2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 :=
by
  sorry

end value_of_expression_l205_205351


namespace min_teachers_required_l205_205663

-- Define the conditions
def num_english_teachers : ℕ := 9
def num_history_teachers : ℕ := 7
def num_geography_teachers : ℕ := 6
def max_subjects_per_teacher : ℕ := 2

-- The proposition we want to prove
theorem min_teachers_required :
  ∃ (t : ℕ), t = 13 ∧
    t * max_subjects_per_teacher ≥ num_english_teachers + num_history_teachers + num_geography_teachers :=
sorry

end min_teachers_required_l205_205663


namespace min_b_l205_205082

-- Definitions
def S (n : ℕ) : ℤ := 2^n - 1
def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2^(n-1)
def b (n : ℕ) : ℤ := (a n)^2 - 7 * (a n) + 6

-- Theorem
theorem min_b : ∃ n : ℕ, (b n = -6) :=
sorry

end min_b_l205_205082


namespace smallest_composite_no_prime_factors_less_than_20_l205_205138

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l205_205138


namespace alpha_half_quadrant_l205_205983

open Real

theorem alpha_half_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) :
  (∃ k1 : ℤ, (2 * k1 + 1) * π - π / 4 < α / 2 ∧ α / 2 < (2 * k1 + 1) * π) ∨
  (∃ k2 : ℤ, 2 * k2 * π - π / 4 < α / 2 ∧ α / 2 < 2 * k2 * π) :=
sorry

end alpha_half_quadrant_l205_205983


namespace inequality_solution_solution_set_l205_205102

noncomputable def f (x a : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem inequality_solution (a : ℝ) : 
  f 1 a > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
by sorry

theorem solution_set (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 → f x a > b) ∧ (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x a = b) ↔ 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3 :=
by sorry

end inequality_solution_solution_set_l205_205102


namespace necessary_sufficient_condition_l205_205030

noncomputable def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + (4 - 2 * a)

theorem necessary_sufficient_condition (a : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) : 
  (∀ (x : ℝ), f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end necessary_sufficient_condition_l205_205030


namespace flight_duration_problem_l205_205596

def problem_conditions : Prop :=
  let la_departure_pst := (7, 15) -- 7:15 AM PST
  let ny_arrival_est := (17, 40) -- 5:40 PM EST (17:40 in 24-hour format)
  let time_difference := 3 -- Hours difference (EST is 3 hours ahead of PST)
  let dst_adjustment := 1 -- Daylight saving time adjustment in hours
  ∃ (h m : ℕ), (0 < m ∧ m < 60) ∧ ((h = 7 ∧ m = 25) ∧ (h + m = 32))

theorem flight_duration_problem :
  problem_conditions :=
by
  -- Placeholder for the proof that shows the conditions established above imply h + m = 32
  sorry

end flight_duration_problem_l205_205596


namespace boat_travel_distance_along_stream_l205_205649

theorem boat_travel_distance_along_stream :
  ∀ (v_s : ℝ), (5 - v_s = 2) → (5 + v_s) * 1 = 8 :=
by
  intro v_s
  intro h1
  have vs_value : v_s = 3 := by linarith
  rw [vs_value]
  norm_num

end boat_travel_distance_along_stream_l205_205649


namespace april_rainfall_correct_l205_205894

-- Define the constants for the rainfalls in March and the difference in April
def march_rainfall : ℝ := 0.81
def rain_difference : ℝ := 0.35

-- Define the expected April rainfall based on the conditions
def april_rainfall : ℝ := march_rainfall - rain_difference

-- Theorem to prove that the April rainfall is 0.46 inches
theorem april_rainfall_correct : april_rainfall = 0.46 :=
by
  -- Placeholder for the proof
  sorry

end april_rainfall_correct_l205_205894


namespace petr_receives_1000000_l205_205332

def initial_investment_vp := 200000
def initial_investment_pg := 350000
def third_share_value := 1100000
def total_company_value := 3 * third_share_value

theorem petr_receives_1000000 :
  initial_investment_vp = 200000 →
  initial_investment_pg = 350000 →
  third_share_value = 1100000 →
  total_company_value = 3300000 →
  ∃ (share_pg : ℕ), share_pg = 1000000 :=
by
  intros h_vp h_pg h_as h_total
  let x := initial_investment_vp * 1650000
  let y := initial_investment_pg * 1650000
  -- Skipping calculations
  sorry

end petr_receives_1000000_l205_205332


namespace trapezoid_height_l205_205511

theorem trapezoid_height (A : ℝ) (d1 d2 : ℝ) (h : ℝ) :
  A = 2 ∧ d1 + d2 = 4 → h = Real.sqrt 2 :=
by
  sorry

end trapezoid_height_l205_205511


namespace smaller_number_4582_l205_205256

theorem smaller_number_4582 (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha_b : a < 100) (hb_b : b < 100) (h : a * b = 4582) :
  min a b = 21 :=
sorry

end smaller_number_4582_l205_205256


namespace analytical_expression_of_odd_function_l205_205901

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2 * x + 3 
else if x = 0 then 0 
else -x^2 - 2 * x - 3

theorem analytical_expression_of_odd_function :
  ∀ x : ℝ, f x =
    if x > 0 then x^2 - 2 * x + 3 
    else if x = 0 then 0 
    else -x^2 - 2 * x - 3 :=
by
  sorry

end analytical_expression_of_odd_function_l205_205901


namespace arccos_half_eq_pi_over_three_l205_205033

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l205_205033


namespace find_a_find_m_l205_205668

noncomputable def f (x a : ℝ) : ℝ := Real.exp 1 * x - a * Real.log x

theorem find_a {a : ℝ} (h : ∀ x, f x a = Real.exp 1 - a / x)
  (hx : f (1 / Real.exp 1) a = 0) :
  a = 1 :=
by
  sorry

theorem find_m (a : ℝ) (h_a : a = 1)
  (h_exists : ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) 
    ∧ f x₀ a < x₀ + m) :
  1 + Real.log (Real.exp 1 - 1) < m :=
by
  sorry

end find_a_find_m_l205_205668


namespace scientific_notation_63000_l205_205196

theorem scientific_notation_63000 : 63000 = 6.3 * 10^4 :=
by
  sorry

end scientific_notation_63000_l205_205196


namespace find_fraction_l205_205599

theorem find_fraction (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4)
  (h3 : b = 1) : a / b = (17 + Real.sqrt 269) / 10 :=
by
  sorry

end find_fraction_l205_205599


namespace original_price_l205_205166

theorem original_price (P : ℝ) (profit : ℝ) (profit_percentage : ℝ)
  (h1 : profit = 675) (h2 : profit_percentage = 0.35) :
  P = 1928.57 :=
by
  -- The proof is skipped using sorry
  sorry

end original_price_l205_205166


namespace B_elements_l205_205283

def B : Set ℤ := {x | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} :=
by
  sorry

end B_elements_l205_205283


namespace combined_solid_volume_l205_205533

open Real

noncomputable def volume_truncated_cone (R r h : ℝ) :=
  (1 / 3) * π * h * (R^2 + R * r + r^2)

noncomputable def volume_cylinder (r h : ℝ): ℝ :=
  π * r^2 * h

theorem combined_solid_volume :
  let R := 10
  let r := 3
  let h_cone := 8
  let h_cyl := 10
  volume_truncated_cone R r h_cone + volume_cylinder r h_cyl = (1382 * π) / 3 :=
  by
  sorry

end combined_solid_volume_l205_205533


namespace range_of_f_neg2_l205_205049

def quadratic_fn (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_neg2 (a b : ℝ) (h1 : 1 ≤ quadratic_fn a b (-1) ∧ quadratic_fn a b (-1) ≤ 2)
    (h2 : 2 ≤ quadratic_fn a b 1 ∧ quadratic_fn a b 1 ≤ 4) :
    3 ≤ quadratic_fn a b (-2) ∧ quadratic_fn a b (-2) ≤ 12 :=
sorry

end range_of_f_neg2_l205_205049


namespace arithmetic_sequence_general_term_l205_205233

theorem arithmetic_sequence_general_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 3) :
  ∃ a_n, a_n = a₁ + (n - 1) * d ∧ a_n = 3 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l205_205233


namespace John_next_birthday_age_l205_205390

variable (John Mike Lucas : ℝ)

def John_is_25_percent_older_than_Mike := John = 1.25 * Mike
def Mike_is_30_percent_younger_than_Lucas := Mike = 0.7 * Lucas
def sum_of_ages_is_27_point_3_years := John + Mike + Lucas = 27.3

theorem John_next_birthday_age 
  (h1 : John_is_25_percent_older_than_Mike John Mike) 
  (h2 : Mike_is_30_percent_younger_than_Lucas Mike Lucas) 
  (h3 : sum_of_ages_is_27_point_3_years John Mike Lucas) : 
  John + 1 = 10 := 
sorry

end John_next_birthday_age_l205_205390


namespace find_triplets_l205_205432

noncomputable def phi (t : ℝ) : ℝ := 2 * t^3 + t - 2

theorem find_triplets (x y z : ℝ) (h1 : x^5 = phi y) (h2 : y^5 = phi z) (h3 : z^5 = phi x) :
  ∃ r : ℝ, (x = r ∧ y = r ∧ z = r) ∧ (r^5 = phi r) :=
by
  sorry

end find_triplets_l205_205432


namespace ordered_sum_ways_l205_205454

theorem ordered_sum_ways (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 2) : 
  ∃ (ways : ℕ), ways = 70 :=
by
  sorry

end ordered_sum_ways_l205_205454


namespace price_reduction_l205_205124

theorem price_reduction (x : ℝ) :
  (20 + 2 * x) * (40 - x) = 1200 → x = 20 :=
by
  sorry

end price_reduction_l205_205124


namespace people_on_trolley_l205_205927

-- Given conditions
variable (X : ℕ)

def initial_people : ℕ := 10

def second_stop_people : ℕ := initial_people - 3 + 20

def third_stop_people : ℕ := second_stop_people - 18 + 2

def fourth_stop_people : ℕ := third_stop_people - 5 + X

-- Prove the current number of people on the trolley is 6 + X
theorem people_on_trolley (X : ℕ) : 
  fourth_stop_people X = 6 + X := 
by 
  unfold fourth_stop_people
  unfold third_stop_people
  unfold second_stop_people
  unfold initial_people
  sorry

end people_on_trolley_l205_205927


namespace quadratic_eq_complete_square_l205_205298

theorem quadratic_eq_complete_square (x p q : ℝ) (h : 9 * x^2 - 54 * x + 63 = 0) 
(h_trans : (x + p)^2 = q) : p + q = -1 := sorry

end quadratic_eq_complete_square_l205_205298


namespace smallest_n_satisfies_conditions_l205_205347

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l205_205347


namespace exists_special_function_l205_205541

theorem exists_special_function : ∃ (s : ℚ → ℤ), (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) ∧ (∀ x : ℚ, s x = 1 ∨ s x = -1) :=
by
  sorry

end exists_special_function_l205_205541


namespace max_volume_of_acetic_acid_solution_l205_205669

theorem max_volume_of_acetic_acid_solution :
  (∀ (V : ℝ), 0 ≤ V ∧ (V * 0.09) = (25 * 0.7 + (V - 25) * 0.05)) →
  V = 406.25 :=
by
  sorry

end max_volume_of_acetic_acid_solution_l205_205669


namespace amount_of_money_C_l205_205236

variable (A B C : ℝ)

theorem amount_of_money_C (h1 : A + B + C = 500)
                         (h2 : A + C = 200)
                         (h3 : B + C = 360) :
    C = 60 :=
sorry

end amount_of_money_C_l205_205236


namespace find_angle_C_find_max_area_l205_205690

variable {A B C a b c : ℝ}

-- Given Conditions
def condition1 (c B a b C : ℝ) := c * Real.cos B + (b - 2 * a) * Real.cos C = 0
def condition2 (c : ℝ) := c = 2 * Real.sqrt 3

-- Problem (1): Prove the size of angle C
theorem find_angle_C (h : condition1 c B a b C) (h2 : condition2 c) : C = Real.pi / 3 := 
  sorry

-- Problem (2): Prove the maximum area of ΔABC
theorem find_max_area (h : condition1 c B a b C) (h2 : condition2 c) :
  ∃ (A B : ℝ), B = 2 * Real.pi / 3 - A ∧ 
    (∀ (A B : ℝ), Real.sin (2 * A - Real.pi / 6) = 1 → 
    1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 ∧ 
    a = b ∧ b = c) := 
  sorry

end find_angle_C_find_max_area_l205_205690


namespace geometric_sequence_a6_l205_205228

theorem geometric_sequence_a6 (a : ℕ → ℝ) (r : ℝ)
  (h₁ : a 4 = 7)
  (h₂ : a 8 = 63)
  (h_geom : ∀ n, a n = a 1 * r^(n - 1)) :
  a 6 = 21 :=
sorry

end geometric_sequence_a6_l205_205228


namespace Will_worked_on_Tuesday_l205_205144

variable (HourlyWage MondayHours TotalEarnings : ℝ)

-- Given conditions
def Wage : ℝ := 8
def Monday_worked_hours : ℝ := 8
def Total_two_days_earnings : ℝ := 80

theorem Will_worked_on_Tuesday (HourlyWage_eq : HourlyWage = Wage)
  (MondayHours_eq : MondayHours = Monday_worked_hours)
  (TotalEarnings_eq : TotalEarnings = Total_two_days_earnings) :
  let MondayEarnings := MondayHours * HourlyWage
  let TuesdayEarnings := TotalEarnings - MondayEarnings
  let TuesdayHours := TuesdayEarnings / HourlyWage
  TuesdayHours = 2 :=
by
  sorry

end Will_worked_on_Tuesday_l205_205144


namespace min_balloon_count_l205_205540

theorem min_balloon_count 
(R B : ℕ) (burst_red burst_blue : ℕ) 
(h1 : R = 7 * B) 
(h2 : burst_red = burst_blue / 3) 
(h3 : burst_red ≥ 1) :
R + B = 24 :=
by 
    sorry

end min_balloon_count_l205_205540


namespace q_evaluation_at_3_point_5_l205_205978

def q (x : ℝ) : ℝ :=
  |x - 3|^(1/3) + 2*|x - 3|^(1/5) + |x - 3|^(1/7)

theorem q_evaluation_at_3_point_5 : q 3.5 = 3 :=
by
  sorry

end q_evaluation_at_3_point_5_l205_205978


namespace range_of_m_l205_205479

variable (m : ℝ)
def p := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0
def q := ∃ x : ℝ, x ∈ Set.Icc (1 : ℝ) 2 ∧ Real.log (x^2 - m*x + 1) / Real.log (1/2) < -1

theorem range_of_m (hp : p m) (hq : q m) (hl : (p m) ∨ (q m)) (hf : ¬ ((p m) ∧ (q m))) :
  m < 1/2 ∨ m = 3/2 := sorry

end range_of_m_l205_205479


namespace rectangle_length_fraction_l205_205920

theorem rectangle_length_fraction 
  (s r : ℝ) 
  (A b ℓ : ℝ)
  (area_square : s * s = 1600)
  (radius_eq_side : r = s)
  (area_rectangle : A = ℓ * b)
  (breadth_rect : b = 10)
  (area_rect_val : A = 160) :
  ℓ / r = 2 / 5 := 
by
  sorry

end rectangle_length_fraction_l205_205920


namespace men_seated_count_l205_205340

theorem men_seated_count (total_passengers : ℕ) (two_thirds_women : total_passengers * 2 / 3 = women)
                         (one_eighth_standing : total_passengers / 3 / 8 = standing_men) :
  total_passengers = 48 →
  women = 32 →
  standing_men = 2 →
  men_seated = (total_passengers - women) - standing_men →
  men_seated = 14 :=
by
  intros
  sorry

end men_seated_count_l205_205340


namespace pints_in_vat_l205_205467

-- Conditions
def num_glasses : Nat := 5
def pints_per_glass : Nat := 30

-- Problem statement: prove that the total number of pints in the vat is 150
theorem pints_in_vat : num_glasses * pints_per_glass = 150 :=
by
  -- Proof goes here
  sorry

end pints_in_vat_l205_205467


namespace connie_total_markers_l205_205013

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end connie_total_markers_l205_205013


namespace Aarti_work_days_l205_205518

theorem Aarti_work_days (x : ℕ) : (3 * x = 24) → x = 8 := by
  intro h
  linarith

end Aarti_work_days_l205_205518


namespace janessa_kept_20_cards_l205_205709

-- Definitions based on conditions
def initial_cards : Nat := 4
def father_cards : Nat := 13
def ebay_cards : Nat := 36
def bad_shape_cards : Nat := 4
def cards_given_to_dexter : Nat := 29

-- Prove that Janessa kept 20 cards for herself
theorem janessa_kept_20_cards :
  (initial_cards + father_cards  + ebay_cards - bad_shape_cards) - cards_given_to_dexter = 20 :=
by
  sorry

end janessa_kept_20_cards_l205_205709


namespace find_extrema_l205_205093

noncomputable def y (x : ℝ) := (Real.sin (3 * x))^2

theorem find_extrema : 
  ∃ (x : ℝ), (0 < x ∧ x < 0.6) ∧ (∀ ε > 0, ε < 0.6 - x → y (x + ε) ≤ y x ∧ y (x - ε) ≤ y x) ∧ x = Real.pi / 6 :=
by
  sorry

end find_extrema_l205_205093


namespace alice_weight_l205_205865

theorem alice_weight (a c : ℝ) (h1 : a + c = 200) (h2 : a - c = a / 3) : a = 120 :=
by
  sorry

end alice_weight_l205_205865


namespace factorize_expr_solve_inequality_solve_equation_simplify_expr_l205_205194

-- Problem 1
theorem factorize_expr (x y m n : ℝ) : x^2 * (3 * m - 2 * n) + y^2 * (2 * n - 3 * m) = (3 * m - 2 * n) * (x + y) * (x - y) := 
sorry

-- Problem 2
theorem solve_inequality (x : ℝ) : 
  (∃ x, (x - 3) / 2 + 3 > x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) → -2 < x ∧ x < 1 :=
sorry

-- Problem 3
theorem solve_equation (x : ℝ) : 
  (∃ x, (3 - x) / (x - 4) + 1 / (4 - x) = 1) → x = 3 :=
sorry

-- Problem 4
theorem simplify_expr (a : ℝ) (h : a = 3) : 
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a - 1)) = 3 / 4 :=
sorry

end factorize_expr_solve_inequality_solve_equation_simplify_expr_l205_205194


namespace range_of_a_l205_205391

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end range_of_a_l205_205391


namespace positive_difference_of_numbers_l205_205619

theorem positive_difference_of_numbers :
  ∃ x y : ℕ, x + y = 50 ∧ 3 * y - 4 * x = 10 ∧ y - x = 10 :=
by
  sorry

end positive_difference_of_numbers_l205_205619


namespace amount_after_two_years_l205_205720

noncomputable def annual_increase (initial_amount : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + rate) ^ years

theorem amount_after_two_years :
  annual_increase 32000 (1/8) 2 = 40500 :=
by
  sorry

end amount_after_two_years_l205_205720


namespace rectangle_area_perimeter_l205_205992

theorem rectangle_area_perimeter (a b : ℝ) (h₁ : a * b = 6) (h₂ : a + b = 6) : a^2 + b^2 = 24 := 
by
  sorry

end rectangle_area_perimeter_l205_205992


namespace amy_school_year_hours_l205_205947

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end amy_school_year_hours_l205_205947


namespace fraction_product_simplified_l205_205036

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end fraction_product_simplified_l205_205036


namespace distance_midpoint_AD_to_BC_l205_205108

variable (AC BC BD : ℕ)
variable (perpendicular : Prop)
variable (d : ℝ)

theorem distance_midpoint_AD_to_BC
  (h1 : AC = 6)
  (h2 : BC = 5)
  (h3 : BD = 3)
  (h4 : perpendicular) :
  d = Real.sqrt 5 + 2 := by
  sorry

end distance_midpoint_AD_to_BC_l205_205108


namespace geometric_series_common_ratio_l205_205211

theorem geometric_series_common_ratio (a S r : ℝ)
  (h1 : a = 172)
  (h2 : S = 400)
  (h3 : S = a / (1 - r)) :
  r = 57 / 100 := 
sorry

end geometric_series_common_ratio_l205_205211


namespace floor_sqrt_27_square_l205_205506

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end floor_sqrt_27_square_l205_205506


namespace prime_factors_of_n_l205_205354

def n : ℕ := 400000001

def is_prime (p: ℕ) : Prop := Nat.Prime p

theorem prime_factors_of_n (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : n = p * q) : 
  (p = 19801 ∧ q = 20201) ∨ (p = 20201 ∧ q = 19801) :=
by
  sorry

end prime_factors_of_n_l205_205354


namespace total_worth_of_stock_l205_205171

theorem total_worth_of_stock (x y : ℕ) (cheap_cost expensive_cost : ℝ) 
  (h1 : y = 21) (h2 : x + y = 22)
  (h3 : expensive_cost = 10) (h4 : cheap_cost = 2.5) :
  (x * expensive_cost + y * cheap_cost) = 62.5 :=
by
  sorry

end total_worth_of_stock_l205_205171


namespace factor_x6_minus_64_l205_205025

theorem factor_x6_minus_64 :
  ∀ x : ℝ, (x^6 - 64) = (x-2) * (x+2) * (x^4 + 4*x^2 + 16) :=
by
  sorry

end factor_x6_minus_64_l205_205025


namespace jar_total_value_l205_205495

def total_value_in_jar (p n q : ℕ) (total_coins : ℕ) (value : ℝ) : Prop :=
  p + n + q = total_coins ∧
  n = 3 * p ∧
  q = 4 * n ∧
  value = p * 0.01 + n * 0.05 + q * 0.25

theorem jar_total_value (p : ℕ) (h₁ : 16 * p = 240) : 
  ∃ value, total_value_in_jar p (3 * p) (12 * p) 240 value ∧ value = 47.4 :=
by
  sorry

end jar_total_value_l205_205495


namespace range_of_a_l205_205907

theorem range_of_a (a : ℝ) :
  (∃ A : Finset ℝ, 
    (∀ x, x ∈ A ↔ x^3 - 2 * x^2 + a * x = 0) ∧ A.card = 3) ↔ (a < 0 ∨ (0 < a ∧ a < 1)) :=
by
  sorry

end range_of_a_l205_205907


namespace sum_of_squares_is_42_l205_205735

variables (D T H : ℕ)

theorem sum_of_squares_is_42
  (h1 : 3 * D + T = 2 * H)
  (h2 : 2 * H^3 = 3 * D^3 + T^3)
  (coprime : Nat.gcd (Nat.gcd D T) H = 1) :
  (T^2 + D^2 + H^2 = 42) :=
sorry

end sum_of_squares_is_42_l205_205735


namespace sum_of_x_for_ggg_eq_neg2_l205_205452

noncomputable def g (x : ℝ) := (x^2) / 3 + x - 2

theorem sum_of_x_for_ggg_eq_neg2 : (∃ x1 x2 : ℝ, (g (g (g x1)) = -2 ∧ g (g (g x2)) = -2 ∧ x1 ≠ x2)) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_x_for_ggg_eq_neg2_l205_205452


namespace nine_by_nine_chessboard_dark_light_excess_l205_205195

theorem nine_by_nine_chessboard_dark_light_excess :
  let board_size := 9
  let odd_row_dark := 5
  let odd_row_light := 4
  let even_row_dark := 4
  let even_row_light := 5
  let num_odd_rows := (board_size + 1) / 2
  let num_even_rows := board_size / 2
  let total_dark_squares := (odd_row_dark * num_odd_rows) + (even_row_dark * num_even_rows)
  let total_light_squares := (odd_row_light * num_odd_rows) + (even_row_light * num_even_rows)
  total_dark_squares - total_light_squares = 1 :=
by {
  sorry
}

end nine_by_nine_chessboard_dark_light_excess_l205_205195


namespace new_ratio_cooks_waiters_l205_205181

theorem new_ratio_cooks_waiters
  (initial_ratio : ℕ → ℕ → Prop)
  (cooks waiters : ℕ) :
  initial_ratio 9 24 → 
  12 + waiters = 36 →
  initial_ratio 3 8 →
  9 * 4 = 36 :=
by
  intros h1 h2 h3
  sorry

end new_ratio_cooks_waiters_l205_205181


namespace possible_numbers_tom_l205_205435

theorem possible_numbers_tom (n : ℕ) (h1 : 180 ∣ n) (h2 : 75 ∣ n) (h3 : 500 < n ∧ n < 2500) : n = 900 ∨ n = 1800 :=
sorry

end possible_numbers_tom_l205_205435


namespace erik_orange_juice_count_l205_205824

theorem erik_orange_juice_count (initial_money bread_loaves bread_cost orange_juice_cost remaining_money : ℤ)
  (h₁ : initial_money = 86)
  (h₂ : bread_loaves = 3)
  (h₃ : bread_cost = 3)
  (h₄ : orange_juice_cost = 6)
  (h₅ : remaining_money = 59) :
  (initial_money - remaining_money - (bread_loaves * bread_cost)) / orange_juice_cost = 3 :=
by
  sorry

end erik_orange_juice_count_l205_205824


namespace pondFishEstimate_l205_205987

noncomputable def estimateTotalFish (initialFishMarked : ℕ) (caughtFishTenDaysLater : ℕ) (markedFishCaught : ℕ) : ℕ :=
  initialFishMarked * caughtFishTenDaysLater / markedFishCaught

theorem pondFishEstimate
    (initialFishMarked : ℕ)
    (caughtFishTenDaysLater : ℕ)
    (markedFishCaught : ℕ)
    (h1 : initialFishMarked = 30)
    (h2 : caughtFishTenDaysLater = 50)
    (h3 : markedFishCaught = 2) :
    estimateTotalFish initialFishMarked caughtFishTenDaysLater markedFishCaught = 750 := by
  sorry

end pondFishEstimate_l205_205987


namespace subset_to_union_eq_l205_205644

open Set

variable {α : Type*} (A B : Set α)

theorem subset_to_union_eq (h : A ∩ B = A) : A ∪ B = B :=
by
  sorry

end subset_to_union_eq_l205_205644


namespace difference_highest_lowest_score_l205_205275

-- Declare scores of each player
def Zach_score : ℕ := 42
def Ben_score : ℕ := 21
def Emma_score : ℕ := 35
def Leo_score : ℕ := 28

-- Calculate the highest and lowest scores
def highest_score : ℕ := max (max Zach_score Ben_score) (max Emma_score Leo_score)
def lowest_score : ℕ := min (min Zach_score Ben_score) (min Emma_score Leo_score)

-- Calculate the difference
def score_difference : ℕ := highest_score - lowest_score

theorem difference_highest_lowest_score : score_difference = 21 := 
by
  sorry

end difference_highest_lowest_score_l205_205275


namespace quadrilateral_area_l205_205280

noncomputable def area_of_quadrilateral (a : ℝ) : ℝ :=
  let sqrt3 := Real.sqrt 3
  let num := a^2 * (9 - 5 * sqrt3)
  let denom := 12
  num / denom

theorem quadrilateral_area (a : ℝ) : area_of_quadrilateral a = (a^2 * (9 - 5 * Real.sqrt 3)) / 12 := by
  sorry

end quadrilateral_area_l205_205280


namespace proof_problem_l205_205260

def consistent_system (x y : ℕ) : Prop :=
  x + y = 99 ∧ 3 * x + 1 / 3 * y = 97

theorem proof_problem : ∃ (x y : ℕ), consistent_system x y := sorry

end proof_problem_l205_205260


namespace find_N_l205_205207

theorem find_N : 
  ∀ (a b c N : ℝ), 
  a + b + c = 80 → 
  2 * a = N → 
  b - 10 = N → 
  3 * c = N → 
  N = 38 := 
by sorry

end find_N_l205_205207


namespace count_interesting_quadruples_l205_205076

def interesting_quadruples (a b c d : ℤ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + 2 * d > b + 2 * c 

theorem count_interesting_quadruples : 
  (∃ n : ℤ, n = 582 ∧ ∀ a b c d : ℤ, interesting_quadruples a b c d → n = 582) :=
sorry

end count_interesting_quadruples_l205_205076


namespace division_problem_l205_205570

theorem division_problem (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end division_problem_l205_205570


namespace tangent_line_coordinates_l205_205881

theorem tangent_line_coordinates :
  ∃ x₀ : ℝ, ∃ y₀ : ℝ, (x₀ = 1 ∧ y₀ = Real.exp 1) ∧
  (∀ x : ℝ, ∀ y : ℝ, y = Real.exp x → ∃ m : ℝ, 
    (m = Real.exp 1 ∧ (y - y₀ = m * (x - x₀))) ∧
    (0 - y₀ = m * (0 - x₀))) := sorry

end tangent_line_coordinates_l205_205881


namespace tan_of_angle_in_fourth_quadrant_l205_205896

theorem tan_of_angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5 / 12 :=
sorry

end tan_of_angle_in_fourth_quadrant_l205_205896


namespace increasing_function_unique_root_proof_l205_205490

noncomputable def increasing_function_unique_root (f : ℝ → ℝ) :=
  (∀ x y : ℝ, x < y → f x ≤ f y) -- condition for increasing function
  ∧ ∃! x : ℝ, f x = 0 -- exists exactly one root

theorem increasing_function_unique_root_proof
  (f : ℝ → ℝ)
  (h_inc : ∀ x y : ℝ, x < y → f x ≤ f y)
  (h_ex : ∃ x : ℝ, f x = 0) :
  ∃! x : ℝ, f x = 0 := sorry

end increasing_function_unique_root_proof_l205_205490


namespace min_value_of_f_l205_205289

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end min_value_of_f_l205_205289


namespace distance_between_city_centers_l205_205122

theorem distance_between_city_centers :
  let distance_on_map_cm := 55
  let scale_cm_to_km := 30
  let km_to_m := 1000
  (distance_on_map_cm * scale_cm_to_km * km_to_m) = 1650000 :=
by
  sorry

end distance_between_city_centers_l205_205122


namespace part1_part2_l205_205840

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2 + 1

-- Part 1
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) (hf : f x = 11 / 10) : 
  x = (Real.pi / 6) + Real.arcsin (3 / 5) :=
sorry

-- Part 2
theorem part2 {A B C a b c : ℝ} 
  (hABC : A + B + C = Real.pi) 
  (habc : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) : 
  (0 < B ∧ B ≤ Real.pi / 6) → 
  ∃ y, (0 < y ∧ y ≤ 1 / 2 ∧ f B = y) :=
sorry

end part1_part2_l205_205840


namespace find_special_numbers_l205_205472

theorem find_special_numbers :
  {N : ℕ | ∃ k m a, N = m + 10^k * a ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ k ∧ m < 10^k 
                ∧ ¬(N % 10 = 0) 
                ∧ (N = 6 * (m + 10^(k+1) * (0 : ℕ))) } = {12, 24, 36, 48} := 
by sorry

end find_special_numbers_l205_205472


namespace unique_solution_m_l205_205150

theorem unique_solution_m (m : ℝ) :
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end unique_solution_m_l205_205150


namespace eval_nabla_l205_205174

namespace MathProblem

-- Definition of the operation
def nabla (a b : ℕ) : ℕ :=
  3 + b ^ a

-- Theorem statement
theorem eval_nabla : nabla (nabla 2 3) 4 = 16777219 :=
by
  sorry

end MathProblem

end eval_nabla_l205_205174


namespace factorial_multiple_of_3_l205_205543

theorem factorial_multiple_of_3 (n : ℤ) (h : n ≥ 9) : 3 ∣ (n+1) * (n+3) :=
sorry

end factorial_multiple_of_3_l205_205543


namespace exists_similarity_point_l205_205708

variable {Point : Type} [MetricSpace Point]

noncomputable def similar_triangles (A B A' B' : Point) (O : Point) : Prop :=
  dist A O / dist A' O = dist A B / dist A' B' ∧ dist B O / dist B' O = dist A B / dist A' B'

theorem exists_similarity_point (A B A' B' : Point) (h1 : dist A B ≠ 0) (h2: dist A' B' ≠ 0) :
  ∃ O : Point, similar_triangles A B A' B' O :=
  sorry

end exists_similarity_point_l205_205708


namespace parabola_vertex_l205_205149

theorem parabola_vertex (x y : ℝ) : 
  y^2 + 10 * y + 3 * x + 9 = 0 → 
  (∃ v_x v_y, v_x = 16/3 ∧ v_y = -5 ∧ ∀ (y' : ℝ), (x, y) = (v_x, v_y) ↔ (x, y) = (-1 / 3 * ((y' + 5)^2 - 16), y')) :=
by
  sorry

end parabola_vertex_l205_205149


namespace amount_made_per_jersey_l205_205368

-- Definitions based on conditions
def total_revenue_from_jerseys : ℕ := 25740
def number_of_jerseys_sold : ℕ := 156

-- Theorem statement
theorem amount_made_per_jersey : 
  total_revenue_from_jerseys / number_of_jerseys_sold = 165 := 
by
  sorry

end amount_made_per_jersey_l205_205368


namespace intersection_proof_l205_205755

-- Definitions of sets M and N
def M : Set ℝ := { x | x^2 < 4 }
def N : Set ℝ := { x | x < 1 }

-- The intersection of M and N
def intersection : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Proposition to prove
theorem intersection_proof : M ∩ N = intersection :=
by sorry

end intersection_proof_l205_205755


namespace gwen_walked_time_l205_205104

-- Definition of given conditions
def time_jogged : ℕ := 15
def ratio_jogged_to_walked (j w : ℕ) : Prop := j * 3 = w * 5

-- Definition to state the exact time walked with given ratio
theorem gwen_walked_time (j w : ℕ) (h1 : j = time_jogged) (h2 : ratio_jogged_to_walked j w) : w = 9 :=
by
  sorry

end gwen_walked_time_l205_205104


namespace lucy_cleans_aquariums_l205_205357

theorem lucy_cleans_aquariums :
  (∃ rate : ℕ, rate = 2 / 3) →
  (∃ hours : ℕ, hours = 24) →
  (∃ increments : ℕ, increments = 24 / 3) →
  (∃ aquariums : ℕ, aquariums = (2 * (24 / 3))) →
  aquariums = 16 :=
by
  sorry

end lucy_cleans_aquariums_l205_205357


namespace university_admission_l205_205523

def students_ratio (x y z : ℕ) : Prop :=
  x * 5 = y * 2 ∧ y * 3 = z * 5

def third_tier_students : ℕ := 1500

theorem university_admission :
  ∀ x y z : ℕ, students_ratio x y z → z = third_tier_students → y - x = 1500 :=
by
  intros x y z hratio hthird
  sorry

end university_admission_l205_205523


namespace domain_lg_tan_minus_sqrt3_l205_205345

open Real

theorem domain_lg_tan_minus_sqrt3 :
  {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} =
    {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} :=
by
  sorry

end domain_lg_tan_minus_sqrt3_l205_205345


namespace terminating_decimals_count_l205_205267

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l205_205267


namespace intersection_A_complement_B_l205_205229

-- Definition of the universal set U
def U : Set ℝ := Set.univ

-- Definition of the set A
def A : Set ℝ := {x | x^2 - 2 * x < 0}

-- Definition of the set B
def B : Set ℝ := {x | x > 1}

-- Definition of the complement of B in U
def complement_B : Set ℝ := {x | x ≤ 1}

-- The intersection A ∩ complement_B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- The theorem to prove
theorem intersection_A_complement_B : A ∩ complement_B = intersection :=
by
  -- Proof goes here
  sorry

end intersection_A_complement_B_l205_205229


namespace travis_apples_l205_205245

theorem travis_apples
  (price_per_box : ℕ)
  (num_apples_per_box : ℕ)
  (total_money : ℕ)
  (total_boxes : ℕ)
  (total_apples : ℕ)
  (h1 : price_per_box = 35)
  (h2 : num_apples_per_box = 50)
  (h3 : total_money = 7000)
  (h4 : total_boxes = total_money / price_per_box)
  (h5 : total_apples = total_boxes * num_apples_per_box) :
  total_apples = 10000 :=
sorry

end travis_apples_l205_205245


namespace opposite_of_neg_two_is_two_l205_205562

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l205_205562


namespace primes_count_l205_205037

open Int

theorem primes_count (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ r s : ℤ, ∀ x : ℤ, (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p := 
  by
    sorry

end primes_count_l205_205037


namespace proportion_of_line_segments_l205_205632

theorem proportion_of_line_segments (a b c d : ℕ)
  (h_proportion : a * d = b * c)
  (h_a : a = 2)
  (h_b : b = 4)
  (h_c : c = 3) :
  d = 6 :=
by
  sorry

end proportion_of_line_segments_l205_205632


namespace postcards_initial_count_l205_205988

theorem postcards_initial_count (P : ℕ) 
  (h1 : ∀ n, n = P / 2)
  (h2 : ∀ n, n = (P / 2) * 15 / 5) 
  (h3 : P / 2 + 3 * P / 2 = 36) : 
  P = 18 := 
sorry

end postcards_initial_count_l205_205988


namespace sum_of_tesseract_elements_l205_205240

noncomputable def tesseract_edges : ℕ := 32
noncomputable def tesseract_vertices : ℕ := 16
noncomputable def tesseract_faces : ℕ := 24

theorem sum_of_tesseract_elements : tesseract_edges + tesseract_vertices + tesseract_faces = 72 := by
  -- proof here
  sorry

end sum_of_tesseract_elements_l205_205240


namespace xiaoming_minimum_time_l205_205525

theorem xiaoming_minimum_time :
  let review_time := 30
  let rest_time := 30
  let boil_time := 15
  let homework_time := 25
  (boil_time ≤ rest_time) → 
  (review_time + rest_time + homework_time = 85) :=
by
  intros review_time rest_time boil_time homework_time h_boil_le_rest
  sorry

end xiaoming_minimum_time_l205_205525


namespace values_of_x_that_satisfy_gg_x_eq_g_x_l205_205447

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end values_of_x_that_satisfy_gg_x_eq_g_x_l205_205447


namespace inequality_not_true_l205_205039

variable {x y : ℝ}

theorem inequality_not_true (h : x > y) : ¬(-3 * x + 6 > -3 * y + 6) :=
by
  sorry

end inequality_not_true_l205_205039


namespace total_tickets_spent_l205_205235

def tickets_spent_on_hat : ℕ := 2
def tickets_spent_on_stuffed_animal : ℕ := 10
def tickets_spent_on_yoyo : ℕ := 2

theorem total_tickets_spent :
  tickets_spent_on_hat + tickets_spent_on_stuffed_animal + tickets_spent_on_yoyo = 14 := by
  sorry

end total_tickets_spent_l205_205235


namespace correct_propositions_identification_l205_205814

theorem correct_propositions_identification (x y : ℝ) (h1 : x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)
    (h2 : ¬(x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0))
    (h3 : ¬(¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)))
    (h4 : (¬(x * y ≥ 0) → ¬(x ≥ 0) ∨ ¬(y ≥ 0))) :
  true :=
by
  -- Proof skipped
  sorry

end correct_propositions_identification_l205_205814


namespace minimum_value_of_expression_l205_205348

noncomputable def min_value (a b : ℝ) : ℝ := 1 / a + 3 / b

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : min_value a b ≥ 16 := 
sorry

end minimum_value_of_expression_l205_205348


namespace sum_of_series_l205_205563

theorem sum_of_series :
  (∑' n : ℕ, (3^n) / (3^(3^n) + 1)) = 1 / 2 :=
sorry

end sum_of_series_l205_205563


namespace profit_when_x_is_6_max_profit_l205_205277

noncomputable def design_fee : ℝ := 20000 / 10000
noncomputable def production_cost_per_100 : ℝ := 10000 / 10000

noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 14.7 - 9 / (x - 3)

noncomputable def cost_of_x_sets (x : ℝ) : ℝ :=
  design_fee + x * production_cost_per_100

noncomputable def profit (x : ℝ) : ℝ :=
  P x - cost_of_x_sets x

theorem profit_when_x_is_6 :
  profit 6 = 3.7 := sorry

theorem max_profit :
  ∀ x : ℝ, profit x ≤ 3.7 := sorry

end profit_when_x_is_6_max_profit_l205_205277


namespace fairfield_middle_school_geography_players_l205_205559

/-- At Fairfield Middle School, there are 24 players on the football team.
All players are enrolled in at least one of the subjects: history or geography.
There are 10 players taking history and 6 players taking both subjects.
We need to prove that the number of players taking geography is 20. -/
theorem fairfield_middle_school_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_subjects_players : ℕ)
  (h1 : total_players = 24)
  (h2 : history_players = 10)
  (h3 : both_subjects_players = 6) :
  total_players - (history_players - both_subjects_players) = 20 :=
by {
  sorry
}

end fairfield_middle_school_geography_players_l205_205559


namespace calculate_mean_score_l205_205376

theorem calculate_mean_score (M SD : ℝ) 
  (h1 : M - 2 * SD = 60)
  (h2 : M + 3 * SD = 100) : 
  M = 76 :=
by
  sorry

end calculate_mean_score_l205_205376


namespace inequality_always_holds_l205_205686

variable {a b : ℝ}

theorem inequality_always_holds (ha : a > 0) (hb : b < 0) : 1 / a > 1 / b :=
by
  sorry

end inequality_always_holds_l205_205686


namespace avg_writing_speed_l205_205045

theorem avg_writing_speed 
  (words1 hours1 words2 hours2 : ℕ)
  (h_words1 : words1 = 30000)
  (h_hours1 : hours1 = 60)
  (h_words2 : words2 = 50000)
  (h_hours2 : hours2 = 100) :
  (words1 + words2) / (hours1 + hours2) = 500 :=
by {
  sorry
}

end avg_writing_speed_l205_205045


namespace solve_system1_solve_system2_l205_205957

theorem solve_system1 (x y : ℝ) (h1 : y = x - 4) (h2 : x + y = 6) : x = 5 ∧ y = 1 :=
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x + y = 1) (h2 : 4 * x - y = 5) : x = 1 ∧ y = -1 :=
by sorry

end solve_system1_solve_system2_l205_205957


namespace circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l205_205139

-- Circle 1 with center (8, -3) and passing through point (5, 1)
theorem circle_centered_at_8_neg3_passing_through_5_1 :
  ∃ r : ℝ, (r = 5) ∧ ((x - 8: ℝ)^2 + (y + 3)^2 = r^2) := by
  sorry

-- Circle passing through points A(-1, 5), B(5, 5), and C(6, -2)
theorem circle_passing_through_ABC :
  ∃ D E F : ℝ, (D = -4) ∧ (E = -2) ∧ (F = -20) ∧
    ( ∀ (x : ℝ) (y : ℝ), (x = -1 ∧ y = 5) 
      ∨ (x = 5 ∧ y = 5) 
      ∨ (x = 6 ∧ y = -2) 
      → (x^2 + y^2 + D*x + E*y + F = 0)) := by
  sorry

end circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l205_205139


namespace equal_numbers_possible_l205_205453

noncomputable def circle_operations (n : ℕ) (α : ℝ) : Prop :=
  (n ≥ 3) ∧ (∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n))

-- Statement of the theorem
theorem equal_numbers_possible (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : α > 0) :
  circle_operations n α ↔ ∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n) :=
sorry

end equal_numbers_possible_l205_205453


namespace train_length_l205_205905

open Real

theorem train_length 
  (v : ℝ) -- speed of the train in km/hr
  (t : ℝ) -- time in seconds
  (d : ℝ) -- length of the bridge in meters
  (h_v : v = 36) -- condition 1
  (h_t : t = 50) -- condition 2
  (h_d : d = 140) -- condition 3
  : (v * 1000 / 3600) * t = 360 + 140 := 
sorry

end train_length_l205_205905


namespace roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l205_205512

-- Lean 4 statements to capture the proofs without computation.
theorem roman_created_171 (a b : ℕ) (h_sum : a + b = 17) (h_diff : a - b = 1) : 
  a = 9 ∧ b = 8 ∨ a = 8 ∧ b = 9 := 
  sorry

theorem roman_created_1513_m1 (a b : ℕ) (h_sum : a + b = 15) (h_diff : a - b = 13) : 
  a = 14 ∧ b = 1 ∨ a = 1 ∧ b = 14 := 
  sorry

theorem roman_created_1513_m2 (a b : ℕ) (h_sum : a + b = 151) (h_diff : a - b = 3) : 
  a = 77 ∧ b = 74 ∨ a = 74 ∧ b = 77 := 
  sorry

theorem roman_created_largest (a b : ℕ) (h_sum : a + b = 188) (h_diff : a - b = 10) : 
  a = 99 ∧ b = 89 ∨ a = 89 ∧ b = 99 := 
  sorry

end roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l205_205512


namespace geom_seq_problem_l205_205303

variable {a : ℕ → ℝ}  -- positive geometric sequence

-- Conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a n = a 0 * r^n

theorem geom_seq_problem
  (h_geom : geom_seq a)
  (cond : a 0 * a 4 + 2 * a 2 * a 4 + a 2 * a 6 = 25) :
  a 2 + a 4 = 5 :=
sorry

end geom_seq_problem_l205_205303


namespace problem_statement_l205_205551

theorem problem_statement (x : ℚ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 :=
by
  sorry

end problem_statement_l205_205551


namespace impossible_to_form_11x12x13_parallelepiped_l205_205919

def is_possible_to_form_parallelepiped
  (brick_shapes_form_unit_cubes : Prop)
  (dimensions : ℕ × ℕ × ℕ) : Prop :=
  ∃ bricks : ℕ, 
    (bricks * 4 = dimensions.fst * dimensions.snd * dimensions.snd.fst)

theorem impossible_to_form_11x12x13_parallelepiped 
  (dimensions := (11, 12, 13)) 
  (brick_shapes_form_unit_cubes : Prop) : 
  ¬ is_possible_to_form_parallelepiped brick_shapes_form_unit_cubes dimensions := 
sorry

end impossible_to_form_11x12x13_parallelepiped_l205_205919


namespace f_is_periodic_l205_205932

-- Define the conditions for the function f
def f (x : ℝ) : ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x ≠ 0
axiom f_property : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f (x - a) = 1 / f x

-- Formal problem statement to be proven
theorem f_is_periodic : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = f (x + 2 * a) :=
by {
  sorry
}

end f_is_periodic_l205_205932


namespace koi_fish_after_three_weeks_l205_205793

theorem koi_fish_after_three_weeks
  (f_0 : ℕ := 280) -- initial total number of fish
  (days : ℕ := 21) -- days in 3 weeks
  (koi_added_per_day : ℕ := 2)
  (goldfish_added_per_day : ℕ := 5)
  (goldfish_after_3_weeks : ℕ := 200) :
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  koi_after_3_weeks = 227 :=
by
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  sorry

end koi_fish_after_three_weeks_l205_205793


namespace cos_180_eq_neg1_l205_205754

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l205_205754


namespace bren_age_indeterminate_l205_205646

/-- The problem statement: The ratio of ages of Aman, Bren, and Charlie are in 
the ratio 5:8:7 respectively. A certain number of years ago, the sum of their ages was 76. 
We need to prove that without additional information, it is impossible to uniquely 
determine Bren's age 10 years from now. -/
theorem bren_age_indeterminate
  (x y : ℕ) 
  (h_ratio : true)
  (h_sum : 20 * x - 3 * y = 76) : 
  ∃ x y : ℕ, (20 * x - 3 * y = 76) ∧ ∀ bren_age_future : ℕ, ∃ x' y' : ℕ, (20 * x' - 3 * y' = 76) ∧ (8 * x' + 10) ≠ bren_age_future :=
sorry

end bren_age_indeterminate_l205_205646


namespace bert_kangaroos_equal_to_kameron_in_40_days_l205_205662

theorem bert_kangaroos_equal_to_kameron_in_40_days
  (k_count : ℕ) (b_count : ℕ) (rate : ℕ) (days : ℕ)
  (h1 : k_count = 100)
  (h2 : b_count = 20)
  (h3 : rate = 2)
  (h4 : days = 40) :
  b_count + days * rate = k_count := 
by
  sorry

end bert_kangaroos_equal_to_kameron_in_40_days_l205_205662


namespace range_of_a_l205_205936

open Real

noncomputable def f (x : ℝ) : ℝ := abs (log x)

noncomputable def g (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 1 then 0 
  else abs (x^2 - 4) - 2

noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |h x| = a → has_four_real_roots : Prop) ↔ (1 ≤ a ∧ a < 2 - log 2) := sorry

end range_of_a_l205_205936


namespace cost_price_is_800_l205_205269

theorem cost_price_is_800 (mp sp cp : ℝ) (h1 : mp = 1100) (h2 : sp = 0.8 * mp) (h3 : sp = 1.1 * cp) :
  cp = 800 :=
by
  sorry

end cost_price_is_800_l205_205269


namespace find_a_sq_plus_b_sq_l205_205068

-- Variables and conditions
variables (a b : ℝ)
-- Conditions from the problem
axiom h1 : a - b = 3
axiom h2 : a * b = 9

-- The proof statement
theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 :=
by {
  sorry
}

end find_a_sq_plus_b_sq_l205_205068


namespace range_of_m_l205_205032

noncomputable def f (x m : ℝ) : ℝ := |x^2 - 4| + x^2 + m * x

theorem range_of_m 
  (f_has_two_distinct_zeros : ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 3 ∧ f a m = 0 ∧ f b m = 0) :
  -14 / 3 < m ∧ m < -2 :=
sorry

end range_of_m_l205_205032


namespace initial_length_proof_l205_205796

variables (L : ℕ)

-- Conditions from the problem statement
def condition1 (L : ℕ) : Prop := L - 25 > 118
def condition2 : Prop := 125 - 7 = 118
def initial_length : Prop := L = 143

-- Proof statement
theorem initial_length_proof (L : ℕ) (h1 : condition1 L) (h2 : condition2) : initial_length L :=
sorry

end initial_length_proof_l205_205796


namespace sum_of_coordinates_of_reflected_points_l205_205965

theorem sum_of_coordinates_of_reflected_points (C D : ℝ × ℝ) (hx : C.1 = 3) (hy : C.2 = 8) (hD : D = (-C.1, C.2)) :
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end sum_of_coordinates_of_reflected_points_l205_205965


namespace negation_of_proposition_l205_205216

theorem negation_of_proposition : 
  ¬ (∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 > 0 := by
  sorry

end negation_of_proposition_l205_205216


namespace find_m_value_l205_205730

theorem find_m_value (m : Real) (h : (3 * m + 8) * (m - 3) = 72) : m = (1 + Real.sqrt 1153) / 6 :=
by
  sorry

end find_m_value_l205_205730


namespace book_pages_l205_205218

-- Define the number of pages read each day
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5
def pages_tomorrow : ℕ := 35

-- Total number of pages in the book
def total_pages : ℕ := pages_yesterday + pages_today + pages_tomorrow

-- Proof that the total number of pages is 100
theorem book_pages : total_pages = 100 := by
  -- Skip the detailed proof
  sorry

end book_pages_l205_205218


namespace cos_value_l205_205846

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := 
by 
  sorry

end cos_value_l205_205846


namespace cube_of_prism_volume_l205_205879

theorem cube_of_prism_volume (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by
  sorry

end cube_of_prism_volume_l205_205879


namespace negation_equiv_l205_205083

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop := 
  (is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ is_even c)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop := 
  (is_even a ∧ is_even b) ∨ 
  (is_even a ∧ is_even c) ∨ 
  (is_even b ∧ is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ ¬is_even c)
  
theorem negation_equiv (a b c : ℕ) : 
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c := 
sorry

end negation_equiv_l205_205083


namespace solve_for_x_l205_205369

theorem solve_for_x : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108 / 19 := 
by 
  sorry

end solve_for_x_l205_205369


namespace inequality_am_gm_l205_205019

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (x + z) + (2 * z^2) / (x + y) ≥ x + y + z :=
by
  sorry

end inequality_am_gm_l205_205019


namespace solve_equation_l205_205274

theorem solve_equation :
  ∀ (x : ℝ), (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 8 :=
by
  intros x h
  sorry

end solve_equation_l205_205274


namespace max_value_of_y_l205_205956

theorem max_value_of_y (x : ℝ) (h : 0 < x ∧ x < 1 / 2) : (∃ y, y = x^2 * (1 - 2*x) ∧ y ≤ 1 / 27) :=
sorry

end max_value_of_y_l205_205956


namespace NoahClosetsFit_l205_205517

-- Declare the conditions as Lean variables and proofs
variable (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
variable (H1 : AliClosetCapacity = 200)
variable (H2 : NoahClosetsRatio = 1 / 4)
variable (H3 : NoahClosetsCount = 2)

-- Define the total number of jeans both of Noah's closets can fit
noncomputable def NoahTotalJeans : ℕ := (AliClosetCapacity * NoahClosetsRatio) * NoahClosetsCount

-- Theorem to prove
theorem NoahClosetsFit (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
  (H1 : AliClosetCapacity = 200) 
  (H2 : NoahClosetsRatio = 1 / 4) 
  (H3 : NoahClosetsCount = 2) 
  : NoahTotalJeans AliClosetCapacity NoahClosetsRatio NoahClosetsCount = 100 := 
  by 
    sorry

end NoahClosetsFit_l205_205517


namespace number_of_sides_of_polygon_l205_205627

theorem number_of_sides_of_polygon (n : ℕ) (h1 : (n * (n - 3)) = 340) : n = 20 :=
by
  sorry

end number_of_sides_of_polygon_l205_205627


namespace gold_coins_percentage_is_35_l205_205044

-- Define the conditions: percentage of beads and percentage of silver coins
def percent_beads : ℝ := 0.30
def percent_silver_coins : ℝ := 0.50

-- Definition of the percentage of all objects that are gold coins
def percent_gold_coins (percent_beads percent_silver_coins : ℝ) : ℝ :=
  (1 - percent_beads) * (1 - percent_silver_coins)

-- The statement that we need to prove:
theorem gold_coins_percentage_is_35 :
  percent_gold_coins percent_beads percent_silver_coins = 0.35 :=
  by
    unfold percent_gold_coins percent_beads percent_silver_coins
    sorry

end gold_coins_percentage_is_35_l205_205044


namespace domain_of_f_l205_205392

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 6) / sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : {x : ℝ | ∃ y, y = f x} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_f_l205_205392


namespace value_of_n_l205_205650

-- Definitions of the question and conditions
def is_3_digit_integer (x : ℕ) : Prop := 100 ≤ x ∧ x < 1000
def not_divisible_by (x : ℕ) (d : ℕ) : Prop := ¬ (d ∣ x)

def problem (m n : ℕ) : Prop :=
  lcm m n = 690 ∧ is_3_digit_integer n ∧ not_divisible_by n 3 ∧ not_divisible_by m 2

-- The theorem to prove
theorem value_of_n {m n : ℕ} (h : problem m n) : n = 230 :=
sorry

end value_of_n_l205_205650


namespace tasty_residue_count_2016_l205_205536

def tasty_residue (n : ℕ) (a : ℕ) : Prop :=
  1 < a ∧ a < n ∧ ∃ m : ℕ, m > 1 ∧ a ^ m ≡ a [MOD n]

theorem tasty_residue_count_2016 : 
  (∃ count : ℕ, count = 831 ∧ ∀ a : ℕ, 1 < a ∧ a < 2016 ↔ tasty_residue 2016 a) :=
sorry

end tasty_residue_count_2016_l205_205536


namespace max_possible_ratio_squared_l205_205424

noncomputable def maxRatioSquared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : ℝ :=
  2

theorem max_possible_ratio_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : maxRatioSquared a b h1 h2 h3 h4 = 2 :=
sorry

end max_possible_ratio_squared_l205_205424


namespace arithmetic_sequence_a1_d_l205_205457

theorem arithmetic_sequence_a1_d (a_1 a_2 a_3 a_5 d : ℤ)
  (h1 : a_5 = a_1 + 4 * d)
  (h2 : a_1 + a_2 + a_3 = 3)
  (h3 : a_2 = a_1 + d)
  (h4 : a_3 = a_1 + 2 * d) :
  a_1 = -2 ∧ d = 3 :=
by
  have h_a2 : a_2 = 1 := sorry
  have h_a5 : a_5 = 10 := sorry
  have h_d : d = 3 := sorry
  have h_a1 : a_1 = -2 := sorry
  exact ⟨h_a1, h_d⟩

end arithmetic_sequence_a1_d_l205_205457


namespace capacity_of_each_type_l205_205697

def total_capacity_barrels : ℕ := 7000

def increased_by_first_type : ℕ := 8000

def decreased_by_second_type : ℕ := 3000

theorem capacity_of_each_type 
  (x y : ℕ) 
  (n k : ℕ)
  (h1 : x + y = total_capacity_barrels)
  (h2 : x * (n + k) / n = increased_by_first_type)
  (h3 : y * (n + k) / k = decreased_by_second_type) :
  x = 6400 ∧ y = 600 := sorry

end capacity_of_each_type_l205_205697


namespace first_method_of_exhaustion_l205_205323

-- Define the names
inductive Names where
  | ZuChongzhi
  | LiuHui
  | ZhangHeng
  | YangHui
  deriving DecidableEq

-- Statement of the problem
def method_of_exhaustion_author : Names :=
  Names.LiuHui

-- Main theorem to state the result
theorem first_method_of_exhaustion : method_of_exhaustion_author = Names.LiuHui :=
by 
  sorry

end first_method_of_exhaustion_l205_205323


namespace quadratic_coefficient_nonzero_l205_205775

theorem quadratic_coefficient_nonzero (a : ℝ) (x : ℝ) :
  (a - 3) * x^2 - 3 * x - 4 = 0 → a ≠ 3 :=
sorry

end quadratic_coefficient_nonzero_l205_205775


namespace linear_decreasing_sequence_l205_205515

theorem linear_decreasing_sequence 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_func1 : y1 = -3 * x1 + 1)
  (h_func2 : y2 = -3 * x2 + 1)
  (h_func3 : y3 = -3 * x3 + 1)
  (hx_seq : x1 < x2 ∧ x2 < x3)
  : y3 < y2 ∧ y2 < y1 := 
sorry

end linear_decreasing_sequence_l205_205515


namespace distributor_income_proof_l205_205310

noncomputable def income_2017 (a k x : ℝ) : ℝ :=
  (a + k / (x - 7)) * (x - 5)

theorem distributor_income_proof (a : ℝ) (x : ℝ) (h_range : 10 ≤ x ∧ x ≤ 14) (h_k : k = 3 * a):
  income_2017 a (3 * a) x = 12 * a ↔ x = 13 := by
  sorry

end distributor_income_proof_l205_205310
