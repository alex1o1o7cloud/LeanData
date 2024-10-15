import Mathlib

namespace NUMINAMATH_GPT_ratio_Jane_to_John_l1216_121604

-- Define the conditions as given in the problem.
variable (J N : ℕ) -- total products inspected by John and Jane
variable (rJ rN rT : ℚ) -- rejection rates for John, Jane, and total

-- Setting up the provided conditions
axiom h1 : rJ = 0.005 -- John rejected 0.5% of the products he inspected
axiom h2 : rN = 0.007 -- Jane rejected 0.7% of the products she inspected
axiom h3 : rT = 0.0075 -- 0.75% of the total products were rejected

-- Prove the ratio of products inspected by Jane to products inspected by John is 5
theorem ratio_Jane_to_John : (rJ * J + rN * N) = rT * (J + N) → N = 5 * J :=
by 
  sorry

end NUMINAMATH_GPT_ratio_Jane_to_John_l1216_121604


namespace NUMINAMATH_GPT_total_pictures_l1216_121690

noncomputable def RandyPics : ℕ := 5
noncomputable def PeterPics : ℕ := RandyPics + 3
noncomputable def QuincyPics : ℕ := PeterPics + 20

theorem total_pictures :
  RandyPics + PeterPics + QuincyPics = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_pictures_l1216_121690


namespace NUMINAMATH_GPT_intersection_l1216_121600

noncomputable def M : Set ℝ := { x : ℝ | Real.sqrt (x + 1) ≥ 0 }
noncomputable def N : Set ℝ := { x : ℝ | x^2 + x - 2 < 0 }

theorem intersection (x : ℝ) : x ∈ (M ∩ N) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_GPT_intersection_l1216_121600


namespace NUMINAMATH_GPT_min_cost_to_fence_land_l1216_121654

theorem min_cost_to_fence_land (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * w ^ 2 ≥ 500) : 
  5 * (2 * (l + w)) = 150 * Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_min_cost_to_fence_land_l1216_121654


namespace NUMINAMATH_GPT_max_b_n_occurs_at_n_l1216_121623

def a_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  a1 + (n-1) * d

def S_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  n * a1 + (n * (n-1) / 2) * d

def b_n (n : ℕ) (an : ℚ) : ℚ :=
  (1 + an) / an

theorem max_b_n_occurs_at_n :
  ∀ (n : ℕ) (a1 d : ℚ),
  (a1 = -5/2) →
  (S_n 4 a1 d = 2 * S_n 2 a1 d + 4) →
  n = 4 := sorry

end NUMINAMATH_GPT_max_b_n_occurs_at_n_l1216_121623


namespace NUMINAMATH_GPT_largest_number_of_square_plots_l1216_121647

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_largest_number_of_square_plots_l1216_121647


namespace NUMINAMATH_GPT_problem_l1216_121639

-- Define the variable
variable (x : ℝ)

-- Define the condition
def condition := 3 * x - 1 = 8

-- Define the statement to be proven
theorem problem (h : condition x) : 150 * (1 / x) + 2 = 52 :=
  sorry

end NUMINAMATH_GPT_problem_l1216_121639


namespace NUMINAMATH_GPT_equilateral_triangle_area_decrease_l1216_121683

theorem equilateral_triangle_area_decrease (s : ℝ) (A : ℝ) (s_new : ℝ) (A_new : ℝ)
    (hA : A = 100 * Real.sqrt 3)
    (hs : s^2 = 400)
    (hs_new : s_new = s - 6)
    (hA_new : A_new = (Real.sqrt 3 / 4) * s_new^2) :
    (A - A_new) / A * 100 = 51 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_decrease_l1216_121683


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_divisible_by_3_l1216_121632

theorem sum_of_three_consecutive_integers_divisible_by_3 (a : ℤ) :
  ∃ k : ℤ, k = 3 ∧ (a - 1 + a + (a + 1)) % k = 0 :=
by
  use 3
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_divisible_by_3_l1216_121632


namespace NUMINAMATH_GPT_roots_of_polynomial_l1216_121698

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^2 - 5*x + 6)*(x)*(x-5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1216_121698


namespace NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l1216_121622

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l1216_121622


namespace NUMINAMATH_GPT_swap_values_l1216_121674

theorem swap_values (A B : ℕ) (h₁ : A = 10) (h₂ : B = 20) : 
    let C := A 
    let A := B 
    let B := C
    A = 20 ∧ B = 10 := by
  let C := A
  let A := B
  let B := C
  have h₃ : C = 10 := h₁
  have h₄ : A = 20 := h₂
  have h₅ : B = 10 := h₃
  exact And.intro h₄ h₅

end NUMINAMATH_GPT_swap_values_l1216_121674


namespace NUMINAMATH_GPT_eight_percent_of_fifty_is_four_l1216_121635

theorem eight_percent_of_fifty_is_four : 0.08 * 50 = 4 := by
  sorry

end NUMINAMATH_GPT_eight_percent_of_fifty_is_four_l1216_121635


namespace NUMINAMATH_GPT_stream_speed_l1216_121611

theorem stream_speed (v : ℝ) (t : ℝ) (h1 : t > 0)
  (h2 : ∃ k : ℝ, k = 2 * t)
  (h3 : (9 + v) * t = (9 - v) * (2 * t)) :
  v = 3 := 
sorry

end NUMINAMATH_GPT_stream_speed_l1216_121611


namespace NUMINAMATH_GPT_compose_f_g_f_l1216_121644

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

theorem compose_f_g_f (x : ℝ) : f (g (f 3)) = 79 := by
  sorry

end NUMINAMATH_GPT_compose_f_g_f_l1216_121644


namespace NUMINAMATH_GPT_monomial_2023_l1216_121691

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^n * (n + 1), n)

theorem monomial_2023 :
  monomial 2023 = (-2024, 2023) :=
by
  sorry

end NUMINAMATH_GPT_monomial_2023_l1216_121691


namespace NUMINAMATH_GPT_class_B_has_more_stable_grades_l1216_121656

-- Definitions based on conditions
def avg_score_class_A : ℝ := 85
def avg_score_class_B : ℝ := 85
def var_score_class_A : ℝ := 120
def var_score_class_B : ℝ := 90

-- Proving which class has more stable grades (lower variance indicates more stability)
theorem class_B_has_more_stable_grades :
  var_score_class_B < var_score_class_A :=
by
  -- The proof will need to show the given condition and establish the inequality
  sorry

end NUMINAMATH_GPT_class_B_has_more_stable_grades_l1216_121656


namespace NUMINAMATH_GPT_comb_n_plus_1_2_l1216_121626

theorem comb_n_plus_1_2 (n : ℕ) (h : 0 < n) : 
  (n + 1).choose 2 = (n + 1) * n / 2 :=
by sorry

end NUMINAMATH_GPT_comb_n_plus_1_2_l1216_121626


namespace NUMINAMATH_GPT_last_four_digits_of_7_pow_5000_l1216_121616

theorem last_four_digits_of_7_pow_5000 (h : 7 ^ 250 ≡ 1 [MOD 1250]) : 7 ^ 5000 ≡ 1 [MOD 1250] :=
by
  -- Proof (will be omitted)
  sorry

end NUMINAMATH_GPT_last_four_digits_of_7_pow_5000_l1216_121616


namespace NUMINAMATH_GPT_sequences_properties_l1216_121641

-- Definition of sequences and their properties
variable {n : ℕ}

noncomputable def S (n : ℕ) : ℕ := n^2 - n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else 2 * n - 2
noncomputable def b (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c (n : ℕ) : ℕ := (2 * (n - 1)) / 3^(n - 1)
noncomputable def T (n : ℕ) : ℕ := 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))

-- Main theorem
theorem sequences_properties (n : ℕ) (hn : n > 0) :
  S n = n^2 - n ∧
  (∀ n, a n = if n = 1 then 0 else 2 * n - 2) ∧
  (∀ n, b n = 3^(n-1)) ∧
  (∀ n, T n = 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))) :=
by sorry

end NUMINAMATH_GPT_sequences_properties_l1216_121641


namespace NUMINAMATH_GPT_evaluate_expression_l1216_121627

theorem evaluate_expression (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l1216_121627


namespace NUMINAMATH_GPT_car_A_speed_l1216_121657

theorem car_A_speed (s_A s_B : ℝ) (d_AB d_extra t : ℝ) (h_s_B : s_B = 50) (h_d_AB : d_AB = 40) (h_d_extra : d_extra = 8) (h_time : t = 6) 
(h_distance_traveled_by_car_B : s_B * t = 300) 
(h_distance_difference : d_AB + d_extra = 48) :
  s_A = 58 :=
by
  sorry

end NUMINAMATH_GPT_car_A_speed_l1216_121657


namespace NUMINAMATH_GPT_garden_ratio_l1216_121621

theorem garden_ratio (L W : ℝ) (h1 : 2 * L + 2 * W = 180) (h2 : L = 60) : L / W = 2 :=
by
  -- this is where you would put the proof
  sorry

end NUMINAMATH_GPT_garden_ratio_l1216_121621


namespace NUMINAMATH_GPT_john_total_payment_l1216_121633

theorem john_total_payment :
  let cost_per_appointment := 400
  let total_appointments := 3
  let pet_insurance_cost := 100
  let insurance_coverage := 0.80
  let first_appointment_cost := cost_per_appointment
  let subsequent_appointments := total_appointments - 1
  let subsequent_appointments_cost := subsequent_appointments * cost_per_appointment
  let covered_cost := subsequent_appointments_cost * insurance_coverage
  let uncovered_cost := subsequent_appointments_cost - covered_cost
  let total_cost := first_appointment_cost + pet_insurance_cost + uncovered_cost
  total_cost = 660 :=
by
  sorry

end NUMINAMATH_GPT_john_total_payment_l1216_121633


namespace NUMINAMATH_GPT_not_divisible_l1216_121628

theorem not_divisible (x y : ℕ) (hx : x % 61 ≠ 0) (hy : y % 61 ≠ 0) (h : (7 * x + 34 * y) % 61 = 0) : (5 * x + 16 * y) % 61 ≠ 0 := 
sorry

end NUMINAMATH_GPT_not_divisible_l1216_121628


namespace NUMINAMATH_GPT_average_length_of_strings_l1216_121617

theorem average_length_of_strings {l1 l2 l3 : ℝ} (h1 : l1 = 2) (h2 : l2 = 6) (h3 : l3 = 9) : 
  (l1 + l2 + l3) / 3 = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_length_of_strings_l1216_121617


namespace NUMINAMATH_GPT_parity_of_pq_l1216_121670

theorem parity_of_pq (x y m n p q : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 0)
    (hx : x = p) (hy : y = q) (h1 : x - 1998 * y = n) (h2 : 1999 * x + 3 * y = m) :
    p % 2 = 0 ∧ q % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_parity_of_pq_l1216_121670


namespace NUMINAMATH_GPT_intersection_with_y_axis_l1216_121629

theorem intersection_with_y_axis :
  ∃ (y : ℝ), (y = -x^2 + 3*x - 4) ∧ (x = 0) ∧ (y = -4) := 
by
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l1216_121629


namespace NUMINAMATH_GPT_mental_math_quiz_l1216_121631

theorem mental_math_quiz : ∃ (q_i q_c : ℕ), q_c + q_i = 100 ∧ 10 * q_c - 5 * q_i = 850 ∧ q_i = 10 :=
by
  sorry

end NUMINAMATH_GPT_mental_math_quiz_l1216_121631


namespace NUMINAMATH_GPT_sum_geometric_sequence_first_eight_terms_l1216_121669

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end NUMINAMATH_GPT_sum_geometric_sequence_first_eight_terms_l1216_121669


namespace NUMINAMATH_GPT_stacked_cubes_surface_area_is_945_l1216_121634

def volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

def side_length (v : ℕ) : ℕ := v^(1/3)

def num_visible_faces (i : ℕ) : ℕ :=
  if i == 0 then 5 else 3 -- Bottom cube has 5 faces visible, others have 3 due to rotation

def surface_area (s : ℕ) (faces : ℕ) : ℕ :=
  faces * s^2

def total_surface_area (volumes : List ℕ) : ℕ :=
  (volumes.zipWith surface_area (volumes.enum.map (λ (i, v) => num_visible_faces i))).sum

theorem stacked_cubes_surface_area_is_945 :
  total_surface_area volumes = 945 := 
by 
  sorry

end NUMINAMATH_GPT_stacked_cubes_surface_area_is_945_l1216_121634


namespace NUMINAMATH_GPT_mashed_potatoes_vs_tomatoes_l1216_121650

theorem mashed_potatoes_vs_tomatoes :
  let m := 144
  let t := 79
  m - t = 65 :=
by 
  repeat { sorry }

end NUMINAMATH_GPT_mashed_potatoes_vs_tomatoes_l1216_121650


namespace NUMINAMATH_GPT_fg_of_2_l1216_121646

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_fg_of_2_l1216_121646


namespace NUMINAMATH_GPT_cube_root_expression_l1216_121684

theorem cube_root_expression (x : ℝ) (hx : x ≥ 0) : (x * Real.sqrt (x * x^(1/3)))^(1/3) = x^(5/9) :=
by
  sorry

end NUMINAMATH_GPT_cube_root_expression_l1216_121684


namespace NUMINAMATH_GPT_find_greater_number_l1216_121681

theorem find_greater_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x > y) : x = 25 := 
sorry

end NUMINAMATH_GPT_find_greater_number_l1216_121681


namespace NUMINAMATH_GPT_fraction_meaningful_range_l1216_121636

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_range_l1216_121636


namespace NUMINAMATH_GPT_relationship_abc_l1216_121686

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 15 - Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 11 - Real.sqrt 3

theorem relationship_abc : a > c ∧ c > b := 
by
  unfold a b c
  sorry

end NUMINAMATH_GPT_relationship_abc_l1216_121686


namespace NUMINAMATH_GPT_total_area_of_removed_triangles_l1216_121619

theorem total_area_of_removed_triangles (side_length : ℝ) (half_leg_length : ℝ) :
  side_length = 16 →
  half_leg_length = side_length / 4 →
  4 * (1 / 2) * half_leg_length^2 = 32 :=
by
  intro h_side_length h_half_leg_length
  simp [h_side_length, h_half_leg_length]
  sorry

end NUMINAMATH_GPT_total_area_of_removed_triangles_l1216_121619


namespace NUMINAMATH_GPT_weight_difference_at_end_of_year_l1216_121664

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_at_end_of_year_l1216_121664


namespace NUMINAMATH_GPT_tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l1216_121671

open Real

axiom sin_add_half_pi_div_4_eq_zero (α : ℝ) : 
  sin (α + π / 4) + 2 * sin (α - π / 4) = 0

axiom tan_sub_half_pi_div_4_eq_inv_3 (β : ℝ) : 
  tan (π / 4 - β) = 1 / 3

theorem tan_alpha_eq_inv_3 (α : ℝ) (h : sin (α + π / 4) + 2 * sin (α - π / 4) = 0) : 
  tan α = 1 / 3 := sorry

theorem tan_alpha_add_beta_eq_1 (α β : ℝ) 
  (h1 : tan α = 1 / 3) (h2 : tan (π / 4 - β) = 1 / 3) : 
  tan (α + β) = 1 := sorry

end NUMINAMATH_GPT_tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l1216_121671


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1216_121694

theorem perpendicular_lines_condition (k : ℝ) : 
  (k = 5 → (∃ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 ∧ x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 → (k = 5 ∨ k = -1)) :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1216_121694


namespace NUMINAMATH_GPT_value_of_2_pow_a_l1216_121662

theorem value_of_2_pow_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h1 : (2^a)^b = 2^2) (h2 : 2^a * 2^b = 8): 2^a = 2 := 
by
  sorry

end NUMINAMATH_GPT_value_of_2_pow_a_l1216_121662


namespace NUMINAMATH_GPT_fraction_problem_l1216_121624

def fractions : List (ℚ) := [4/3, 7/5, 12/10, 23/20, 45/40, 89/80]
def subtracted_value : ℚ := -8

theorem fraction_problem :
  (fractions.sum - subtracted_value) = -163 / 240 := by
  sorry

end NUMINAMATH_GPT_fraction_problem_l1216_121624


namespace NUMINAMATH_GPT_number_of_students_l1216_121665

theorem number_of_students (n : ℕ) (A : ℕ) 
  (h1 : A = 10 * n)
  (h2 : (A - 11 + 41) / n = 11) :
  n = 30 := 
sorry

end NUMINAMATH_GPT_number_of_students_l1216_121665


namespace NUMINAMATH_GPT_notebooks_cost_l1216_121652

theorem notebooks_cost 
  (P N : ℝ)
  (h1 : 96 * P + 24 * N = 520)
  (h2 : ∃ x : ℝ, 3 * P + x * N = 60)
  (h3 : P + N = 15.512820512820513) :
  ∃ x : ℕ, x = 4 :=
by
  sorry

end NUMINAMATH_GPT_notebooks_cost_l1216_121652


namespace NUMINAMATH_GPT_radius_of_circle_eq_l1216_121676

-- Define the given quadratic equation representing the circle
noncomputable def circle_eq (x y : ℝ) : ℝ :=
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68

-- State that the radius of the circle given by the equation is 1
theorem radius_of_circle_eq : ∃ r, (∀ x y, circle_eq x y = 0 ↔ (x - 1)^2 + (y - 1.5)^2 = r^2) ∧ r = 1 :=
by 
  use 1
  sorry

end NUMINAMATH_GPT_radius_of_circle_eq_l1216_121676


namespace NUMINAMATH_GPT_proof_statement_l1216_121679

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_statement_l1216_121679


namespace NUMINAMATH_GPT_square_diff_l1216_121688

-- Definitions and conditions from the problem
def three_times_sum_eq (a b : ℝ) : Prop := 3 * (a + b) = 18
def diff_eq (a b : ℝ) : Prop := a - b = 4

-- Goal to prove that a^2 - b^2 = 24 under the given conditions
theorem square_diff (a b : ℝ) (h₁ : three_times_sum_eq a b) (h₂ : diff_eq a b) : a^2 - b^2 = 24 :=
sorry

end NUMINAMATH_GPT_square_diff_l1216_121688


namespace NUMINAMATH_GPT_permutation_probability_l1216_121637

theorem permutation_probability (total_digits: ℕ) (zeros: ℕ) (ones: ℕ) 
  (total_permutations: ℕ) (favorable_permutations: ℕ) (probability: ℚ)
  (h1: total_digits = 6) 
  (h2: zeros = 2) 
  (h3: ones = 4) 
  (h4: total_permutations = 2 ^ total_digits) 
  (h5: favorable_permutations = Nat.choose total_digits zeros) 
  (h6: probability = favorable_permutations / total_permutations) : 
  probability = 15 / 64 := 
sorry

end NUMINAMATH_GPT_permutation_probability_l1216_121637


namespace NUMINAMATH_GPT_tan_sum_identity_l1216_121693

theorem tan_sum_identity (a b : ℝ) (h₁ : Real.tan a = 1/2) (h₂ : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_sum_identity_l1216_121693


namespace NUMINAMATH_GPT_remainder_when_divided_by_29_l1216_121645

theorem remainder_when_divided_by_29 (N : ℤ) (k : ℤ) (h : N = 751 * k + 53) : 
  N % 29 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_29_l1216_121645


namespace NUMINAMATH_GPT_car_bus_washing_inconsistency_l1216_121613

theorem car_bus_washing_inconsistency :
  ∀ (C B : ℕ), 
    C % 2 = 0 →
    B % 2 = 1 →
    7 * C + 18 * B = 309 →
    3 + 8 + 5 + C + B = 15 →
    false :=
by
  sorry

end NUMINAMATH_GPT_car_bus_washing_inconsistency_l1216_121613


namespace NUMINAMATH_GPT_perpendicular_sum_value_of_m_l1216_121630

-- Let a and b be defined as vectors in R^2
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product for vectors in R^2
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors using dot product
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the sum of two vectors
def vector_sum (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- State our proof problem
theorem perpendicular_sum_value_of_m :
  is_perpendicular (vector_sum vector_a (vector_b (-7 / 2))) vector_a :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_perpendicular_sum_value_of_m_l1216_121630


namespace NUMINAMATH_GPT_ones_mult_palindrome_l1216_121607

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 
  digits = digits.reverse

def ones (k : ℕ) : ℕ := (10 ^ k - 1) / 9

theorem ones_mult_palindrome (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_palindrome (ones m * ones n) ↔ (m = n ∧ m ≤ 9 ∧ n ≤ 9) := 
sorry

end NUMINAMATH_GPT_ones_mult_palindrome_l1216_121607


namespace NUMINAMATH_GPT_completing_the_square_l1216_121663

theorem completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 ↔ (x - 2)^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_l1216_121663


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1216_121625

-- Definitions for the convex polyhedron, volume, and surface area
structure ConvexPolyhedron :=
  (volume : ℝ)
  (surface_area : ℝ)

variable {P : ConvexPolyhedron}

-- Statement for Part (a)
theorem part_a (r : ℝ) (h_r : r ≤ P.surface_area) :
  P.volume / P.surface_area ≥ r / 3 := sorry

-- Statement for Part (b)
theorem part_b :
  Exists (fun r : ℝ => r = P.volume / P.surface_area) := sorry

-- Definitions and conditions for the outer and inner polyhedron
structure ConvexPolyhedronPair :=
  (outer_polyhedron : ConvexPolyhedron)
  (inner_polyhedron : ConvexPolyhedron)

variable {CP : ConvexPolyhedronPair}

-- Statement for Part (c)
theorem part_c :
  3 * CP.outer_polyhedron.volume / CP.outer_polyhedron.surface_area ≥
  CP.inner_polyhedron.volume / CP.inner_polyhedron.surface_area := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1216_121625


namespace NUMINAMATH_GPT_loss_percentage_is_17_l1216_121615

noncomputable def loss_percentage (CP SP : ℝ) := ((CP - SP) / CP) * 100

theorem loss_percentage_is_17 :
  let CP : ℝ := 1500
  let SP : ℝ := 1245
  loss_percentage CP SP = 17 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_17_l1216_121615


namespace NUMINAMATH_GPT_M_inter_N_eq_l1216_121673

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem M_inter_N_eq : {x | -2 < x ∧ x < 2} = M ∩ N := by
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_l1216_121673


namespace NUMINAMATH_GPT_value_of_f_neg2011_l1216_121699

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem value_of_f_neg2011 (a b : ℝ) (h : f 2011 a b = 10) : f (-2011) a b = -14 := by
  sorry

end NUMINAMATH_GPT_value_of_f_neg2011_l1216_121699


namespace NUMINAMATH_GPT_neg_sqrt_17_estimate_l1216_121606

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end NUMINAMATH_GPT_neg_sqrt_17_estimate_l1216_121606


namespace NUMINAMATH_GPT_find_value_of_a_plus_b_l1216_121659

noncomputable def A (a b : ℤ) : Set ℤ := {1, a, b}
noncomputable def B (a b : ℤ) : Set ℤ := {a, a^2, a * b}

theorem find_value_of_a_plus_b (a b : ℤ) (h : A a b = B a b) : a + b = -1 :=
by sorry

end NUMINAMATH_GPT_find_value_of_a_plus_b_l1216_121659


namespace NUMINAMATH_GPT_no_bijective_function_l1216_121643

open Set

def is_bijective {α β : Type*} (f : α → β) : Prop :=
  Function.Bijective f

def are_collinear {P : Type*} (A B C : P) : Prop :=
  sorry -- placeholder for the collinearity predicate on points

def are_parallel_or_concurrent {L : Type*} (l₁ l₂ l₃ : L) : Prop :=
  sorry -- placeholder for the condition that lines are parallel or concurrent

theorem no_bijective_function (P : Type*) (D : Type*) :
  ¬ ∃ (f : P → D), is_bijective f ∧
    ∀ A B C : P, are_collinear A B C → are_parallel_or_concurrent (f A) (f B) (f C) :=
by
  sorry

end NUMINAMATH_GPT_no_bijective_function_l1216_121643


namespace NUMINAMATH_GPT_problem_f_sum_zero_l1216_121653

variable (f : ℝ → ℝ)

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def symmetrical (f : ℝ → ℝ) : Prop := ∀ x, f (1 - x) = f x

-- Prove the required sum is zero given the conditions.
theorem problem_f_sum_zero (hf_odd : odd f) (hf_symm : symmetrical f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end NUMINAMATH_GPT_problem_f_sum_zero_l1216_121653


namespace NUMINAMATH_GPT_distinct_elements_in_T_l1216_121689

def sequence1 (k : ℕ) : ℤ := 3 * k - 1
def sequence2 (m : ℕ) : ℤ := 8 * m + 2

def setC : Finset ℤ := Finset.image sequence1 (Finset.range 3000)
def setD : Finset ℤ := Finset.image sequence2 (Finset.range 3000)
def setT : Finset ℤ := setC ∪ setD

theorem distinct_elements_in_T : setT.card = 3000 := by
  sorry

end NUMINAMATH_GPT_distinct_elements_in_T_l1216_121689


namespace NUMINAMATH_GPT_decagon_perimeter_l1216_121620

-- Define the number of sides in a decagon
def num_sides : ℕ := 10

-- Define the length of each side in the decagon
def side_length : ℕ := 3

-- Define the perimeter of a decagon given the number of sides and the side length
def perimeter (n : ℕ) (s : ℕ) : ℕ := n * s

-- State the theorem we want to prove: the perimeter of our given regular decagon
theorem decagon_perimeter : perimeter num_sides side_length = 30 := 
by sorry

end NUMINAMATH_GPT_decagon_perimeter_l1216_121620


namespace NUMINAMATH_GPT_cost_price_per_meter_l1216_121697

theorem cost_price_per_meter
  (S : ℝ) (L : ℝ) (C : ℝ) (total_meters : ℝ) (total_price : ℝ)
  (h1 : total_meters = 400) (h2 : total_price = 18000)
  (h3 : L = 5) (h4 : S = total_price / total_meters) 
  (h5 : C = S + L) :
  C = 50 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l1216_121697


namespace NUMINAMATH_GPT_count_marble_pairs_l1216_121642

-- Define conditions:
structure Marbles :=
(red : ℕ) (green : ℕ) (blue : ℕ) (yellow : ℕ) (white : ℕ)

def tomsMarbles : Marbles :=
  { red := 1, green := 1, blue := 1, yellow := 3, white := 2 }

-- Define a function to count pairs of marbles:
def count_pairs (m : Marbles) : ℕ :=
  -- Count pairs of identical marbles:
  (if m.yellow >= 2 then 1 else 0) + 
  (if m.white >= 2 then 1 else 0) +
  -- Count pairs of different colored marbles:
  (Nat.choose 5 2)

-- Theorem statement:
theorem count_marble_pairs : count_pairs tomsMarbles = 12 :=
  by
    sorry

end NUMINAMATH_GPT_count_marble_pairs_l1216_121642


namespace NUMINAMATH_GPT_sin_alpha_minus_beta_l1216_121678

variables (α β : ℝ)

theorem sin_alpha_minus_beta (h1 : (Real.tan α / Real.tan β) = 7 / 13) 
    (h2 : Real.sin (α + β) = 2 / 3) :
    Real.sin (α - β) = -1 / 5 := 
sorry

end NUMINAMATH_GPT_sin_alpha_minus_beta_l1216_121678


namespace NUMINAMATH_GPT_rancher_total_animals_l1216_121602

theorem rancher_total_animals
  (H C : ℕ) (h1 : C = 5 * H) (h2 : C = 140) :
  C + H = 168 := 
sorry

end NUMINAMATH_GPT_rancher_total_animals_l1216_121602


namespace NUMINAMATH_GPT_total_students_in_college_l1216_121660

theorem total_students_in_college 
  (girls : ℕ) 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (h_ratio : ratio_boys = 8) 
  (h_ratio_girls : ratio_girls = 5) 
  (h_girls : girls = 400) 
  : (ratio_boys * (girls / ratio_girls) + girls = 1040) := 
by 
  sorry

end NUMINAMATH_GPT_total_students_in_college_l1216_121660


namespace NUMINAMATH_GPT_at_most_one_solution_l1216_121610

theorem at_most_one_solution (a b c : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (hcpos : 0 < c) :
  ∃! x : ℝ, a * x + b * ⌊x⌋ - c = 0 :=
sorry

end NUMINAMATH_GPT_at_most_one_solution_l1216_121610


namespace NUMINAMATH_GPT_num_divisors_630_l1216_121661

theorem num_divisors_630 : ∃ d : ℕ, (d = 24) ∧ ∀ n : ℕ, (∃ (a b c d : ℕ), (n = 2^a * 3^b * 5^c * 7^d) ∧ a ≤ 1 ∧ b ≤ 2 ∧ c ≤ 1 ∧ d ≤ 1) ↔ (n ∣ 630) := sorry

end NUMINAMATH_GPT_num_divisors_630_l1216_121661


namespace NUMINAMATH_GPT_candidate_percentage_l1216_121651

variables (M T : ℝ)

theorem candidate_percentage (h1 : (P / 100) * T = M - 30) 
                             (h2 : (45 / 100) * T = M + 15)
                             (h3 : M = 120) : 
                             P = 30 := 
by 
  sorry

end NUMINAMATH_GPT_candidate_percentage_l1216_121651


namespace NUMINAMATH_GPT_find_a3_l1216_121672

-- Define the sequence sum S_n
def S (n : ℕ) : ℚ := (n + 1) / (n + 2)

-- Define the sequence term a_n using S_n
def a (n : ℕ) : ℚ :=
  if h : n = 1 then S 1 else S n - S (n - 1)

-- State the theorem to find the value of a_3
theorem find_a3 : a 3 = 1 / 20 :=
by
  -- The proof is omitted, use sorry to skip it
  sorry

end NUMINAMATH_GPT_find_a3_l1216_121672


namespace NUMINAMATH_GPT_find_B_l1216_121618

structure Point where
  x : Int
  y : Int

def vector_sub (p1 p2 : Point) : Point :=
  ⟨p1.x - p2.x, p1.y - p2.y⟩

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-1, 2⟩
def BA : Point := ⟨3, 3⟩
def B : Point := ⟨-4, -1⟩

theorem find_B :
  vector_sub A BA = B :=
by
  sorry

end NUMINAMATH_GPT_find_B_l1216_121618


namespace NUMINAMATH_GPT_f_23_plus_f_neg14_l1216_121638

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x, f (x + 5) = f x
axiom odd_f : ∀ x, f (-x) = -f x
axiom f_one : f 1 = 1
axiom f_two : f 2 = 2

theorem f_23_plus_f_neg14 : f 23 + f (-14) = -1 := by
  sorry

end NUMINAMATH_GPT_f_23_plus_f_neg14_l1216_121638


namespace NUMINAMATH_GPT_probability_of_drawing_red_ball_l1216_121685

theorem probability_of_drawing_red_ball (total_balls red_balls white_balls: ℕ) 
    (h1 : total_balls = 5) 
    (h2 : red_balls = 2) 
    (h3 : white_balls = 3) : 
    (red_balls : ℚ) / total_balls = 2 / 5 := 
by 
    sorry

end NUMINAMATH_GPT_probability_of_drawing_red_ball_l1216_121685


namespace NUMINAMATH_GPT_term_sequence_10th_l1216_121696

theorem term_sequence_10th :
  let a (n : ℕ) := (-1:ℚ)^(n+1) * (2*n)/(2*n + 1)
  a 10 = -20/21 := 
by
  sorry

end NUMINAMATH_GPT_term_sequence_10th_l1216_121696


namespace NUMINAMATH_GPT_shoe_cost_l1216_121692

def initial_amount : ℕ := 91
def cost_sweater : ℕ := 24
def cost_tshirt : ℕ := 6
def amount_left : ℕ := 50
def cost_shoes : ℕ := 11

theorem shoe_cost :
  initial_amount - (cost_sweater + cost_tshirt) - amount_left = cost_shoes :=
by
  sorry

end NUMINAMATH_GPT_shoe_cost_l1216_121692


namespace NUMINAMATH_GPT_inequality_solution_set_l1216_121612

theorem inequality_solution_set : 
  { x : ℝ | (x + 1) / (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry 

end NUMINAMATH_GPT_inequality_solution_set_l1216_121612


namespace NUMINAMATH_GPT_license_plate_palindrome_probability_l1216_121609

theorem license_plate_palindrome_probability : 
  let p := 775 
  let q := 67600  
  p + q = 776 :=
by
  let p := 775
  let q := 67600
  show p + q = 776
  sorry

end NUMINAMATH_GPT_license_plate_palindrome_probability_l1216_121609


namespace NUMINAMATH_GPT_sum_x_y_m_l1216_121655

theorem sum_x_y_m (a b x y m : ℕ) (ha : a - b = 3) (hx : x = 10 * a + b) (hy : y = 10 * b + a) (hxy : x^2 - y^2 = m^2) : x + y + m = 178 := sorry

end NUMINAMATH_GPT_sum_x_y_m_l1216_121655


namespace NUMINAMATH_GPT_one_fifth_of_five_times_nine_l1216_121667

theorem one_fifth_of_five_times_nine (a b : ℕ) (h1 : a = 5) (h2 : b = 9) : (1 / 5 : ℚ) * (a * b) = 9 := by
  sorry

end NUMINAMATH_GPT_one_fifth_of_five_times_nine_l1216_121667


namespace NUMINAMATH_GPT_YZ_length_l1216_121675

theorem YZ_length : 
  ∀ (X Y Z : Type) 
  (angle_Y angle_Z angle_X : ℝ)
  (XZ YZ : ℝ),
  angle_Y = 45 ∧ angle_Z = 60 ∧ XZ = 6 →
  angle_X = 180 - angle_Y - angle_Z →
  YZ = XZ * (Real.sin angle_X / Real.sin angle_Y) →
  YZ = 3 * (Real.sqrt 6 + Real.sqrt 2) :=
by
  intros X Y Z angle_Y angle_Z angle_X XZ YZ
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_YZ_length_l1216_121675


namespace NUMINAMATH_GPT_mean_of_set_is_12_point_8_l1216_121658

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end NUMINAMATH_GPT_mean_of_set_is_12_point_8_l1216_121658


namespace NUMINAMATH_GPT_oldest_bride_age_l1216_121682

theorem oldest_bride_age (G B : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) : B = 102 :=
by
  sorry

end NUMINAMATH_GPT_oldest_bride_age_l1216_121682


namespace NUMINAMATH_GPT_even_sum_probability_l1216_121649

-- Define the probabilities of even and odd outcomes for each wheel
def probability_even_first_wheel : ℚ := 2 / 3
def probability_odd_first_wheel : ℚ := 1 / 3
def probability_even_second_wheel : ℚ := 3 / 5
def probability_odd_second_wheel : ℚ := 2 / 5

-- Define the probabilities of the scenarios that result in an even sum
def probability_both_even : ℚ := probability_even_first_wheel * probability_even_second_wheel
def probability_both_odd : ℚ := probability_odd_first_wheel * probability_odd_second_wheel

-- Define the total probability of an even sum
def probability_even_sum : ℚ := probability_both_even + probability_both_odd

-- The theorem statement to be proven
theorem even_sum_probability :
  probability_even_sum = 8 / 15 :=
by
  sorry

end NUMINAMATH_GPT_even_sum_probability_l1216_121649


namespace NUMINAMATH_GPT_zero_ending_of_A_l1216_121695

theorem zero_ending_of_A (A : ℕ) (h : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c ∣ A ∧ a + b + c = 8 → a * b * c = 10) : 
  (10 ∣ A) ∧ ¬(100 ∣ A) :=
by
  sorry

end NUMINAMATH_GPT_zero_ending_of_A_l1216_121695


namespace NUMINAMATH_GPT_number_of_sequences_l1216_121680

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end NUMINAMATH_GPT_number_of_sequences_l1216_121680


namespace NUMINAMATH_GPT_domain_f_x_plus_2_l1216_121601

-- Define the function f and its properties
variable (f : ℝ → ℝ)

-- Define the given condition: the domain of y = f(2x - 3) is [-2, 3]
def domain_f_2x_minus_3 : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 3}

-- Express this condition formally
axiom domain_f_2x_minus_3_axiom :
  ∀ (x : ℝ), (x ∈ domain_f_2x_minus_3) → (2 * x - 3 ∈ Set.Icc (-7 : ℝ) 3)

-- Prove the desired result: the domain of y = f(x + 2) is [-9, 1]
theorem domain_f_x_plus_2 :
  ∀ (x : ℝ), (x ∈ Set.Icc (-9 : ℝ) 1) ↔ ((x + 2) ∈ Set.Icc (-7 : ℝ) 3) :=
sorry

end NUMINAMATH_GPT_domain_f_x_plus_2_l1216_121601


namespace NUMINAMATH_GPT_find_13_points_within_radius_one_l1216_121614

theorem find_13_points_within_radius_one (points : Fin 25 → ℝ × ℝ)
  (h : ∀ i j k : Fin 25, min (dist (points i) (points j)) (min (dist (points i) (points k)) (dist (points j) (points k))) < 1) :
  ∃ (subset : Finset (Fin 25)), subset.card = 13 ∧ ∃ (center : ℝ × ℝ), ∀ i ∈ subset, dist (points i) center < 1 :=
  sorry

end NUMINAMATH_GPT_find_13_points_within_radius_one_l1216_121614


namespace NUMINAMATH_GPT_probability_no_adjacent_same_roll_l1216_121608

theorem probability_no_adjacent_same_roll :
  let A := 1 -- rolls a six-sided die
  let B := 2 -- rolls a six-sided die
  let C := 3 -- rolls a six-sided die
  let D := 4 -- rolls a six-sided die
  let E := 5 -- rolls a six-sided die
  let people := [A, B, C, D, E]
  -- A and C are required to roll different numbers
  let prob_A_C_diff := 5 / 6
  -- B must roll different from A and C
  let prob_B_diff := 4 / 6
  -- D must roll different from C and A
  let prob_D_diff := 4 / 6
  -- E must roll different from D and A
  let prob_E_diff := 3 / 6
  (prob_A_C_diff * prob_B_diff * prob_D_diff * prob_E_diff) = 10 / 27 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_adjacent_same_roll_l1216_121608


namespace NUMINAMATH_GPT_find_a_of_normal_vector_l1216_121677

theorem find_a_of_normal_vector (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y + 5 = 0) ∧ (∃ n : ℝ × ℝ, n = (a, a - 2)) → a = 6 := by
  sorry

end NUMINAMATH_GPT_find_a_of_normal_vector_l1216_121677


namespace NUMINAMATH_GPT_Karen_sold_boxes_l1216_121640

theorem Karen_sold_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by
  sorry

end NUMINAMATH_GPT_Karen_sold_boxes_l1216_121640


namespace NUMINAMATH_GPT_max_value_of_cubes_l1216_121687

theorem max_value_of_cubes 
  (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  x^3 + y^3 + z^3 ≤ 27 :=
  sorry

end NUMINAMATH_GPT_max_value_of_cubes_l1216_121687


namespace NUMINAMATH_GPT_find_m_n_l1216_121668

theorem find_m_n (a b : ℝ) (m n : ℤ) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l1216_121668


namespace NUMINAMATH_GPT_num_valid_m_divisors_of_1750_l1216_121603

theorem num_valid_m_divisors_of_1750 : 
  ∃! (m : ℕ) (h1 : m > 0), ∃ (k : ℕ), k > 0 ∧ 1750 = k * (m^2 - 4) :=
sorry

end NUMINAMATH_GPT_num_valid_m_divisors_of_1750_l1216_121603


namespace NUMINAMATH_GPT_v_not_closed_under_operations_l1216_121605

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def v : Set ℕ := {n | ∃ m : ℕ, n = m * m}

def addition_followed_by_multiplication (a b : ℕ) : ℕ :=
  (a + b) * a

def multiplication_followed_by_addition (a b : ℕ) : ℕ :=
  (a * b) + a

def division_followed_by_subtraction (a b : ℕ) : ℕ :=
  if b ≠ 0 then (a / b) - b else 0

def extraction_root_followed_by_multiplication (a b : ℕ) : ℕ :=
  (Nat.sqrt a) * (Nat.sqrt b)

theorem v_not_closed_under_operations : 
  ¬ (∀ a ∈ v, ∀ b ∈ v, addition_followed_by_multiplication a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, multiplication_followed_by_addition a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, division_followed_by_subtraction a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, extraction_root_followed_by_multiplication a b ∈ v) :=
sorry

end NUMINAMATH_GPT_v_not_closed_under_operations_l1216_121605


namespace NUMINAMATH_GPT_Oliver_9th_l1216_121666

def person := ℕ → Prop

axiom Ruby : person
axiom Oliver : person
axiom Quinn : person
axiom Pedro : person
axiom Nina : person
axiom Samuel : person
axiom place : person → ℕ → Prop

-- Conditions given in the problem
axiom Ruby_Oliver : ∀ n, place Ruby n → place Oliver (n + 7)
axiom Quinn_Pedro : ∀ n, place Quinn n → place Pedro (n - 2)
axiom Nina_Oliver : ∀ n, place Nina n → place Oliver (n + 3)
axiom Pedro_Samuel : ∀ n, place Pedro n → place Samuel (n - 3)
axiom Samuel_Ruby : ∀ n, place Samuel n → place Ruby (n + 2)
axiom Quinn_5th : place Quinn 5

-- Question: Prove that Oliver finished in 9th place
theorem Oliver_9th : place Oliver 9 :=
sorry

end NUMINAMATH_GPT_Oliver_9th_l1216_121666


namespace NUMINAMATH_GPT_avg_temp_Brookdale_l1216_121648

noncomputable def avg_temp (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

theorem avg_temp_Brookdale : avg_temp [51, 67, 64, 61, 50, 65, 47] = 57.9 :=
by
  sorry

end NUMINAMATH_GPT_avg_temp_Brookdale_l1216_121648
