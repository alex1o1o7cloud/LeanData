import Mathlib

namespace NUMINAMATH_GPT_tickets_savings_percentage_l2084_208456

theorem tickets_savings_percentage (P S : ℚ) (h : 8 * S = 5 * P) :
  (12 * P - 12 * S) / (12 * P) * 100 = 37.5 :=
by 
  sorry

end NUMINAMATH_GPT_tickets_savings_percentage_l2084_208456


namespace NUMINAMATH_GPT_distance_between_x_intercepts_l2084_208494

theorem distance_between_x_intercepts :
  ∀ (x1 x2 : ℝ),
  (∀ x, x1 = 8 → x2 = 20 → 20 = 4 * (x - 8)) → 
  (∀ x, x1 = 8 → x2 = 20 → 20 = 7 * (x - 8)) → 
  abs ((3 : ℝ) - (36 / 7)) = (15 / 7) :=
by
  intros x1 x2 h1 h2
  sorry

end NUMINAMATH_GPT_distance_between_x_intercepts_l2084_208494


namespace NUMINAMATH_GPT_find_other_root_l2084_208413

theorem find_other_root (a b : ℝ) (h₁ : 3^2 + 3 * a - 2 * a = 0) (h₂ : ∀ x, x^2 + a * x - 2 * a = 0 → (x = 3 ∨ x = b)) :
  b = 6 := 
sorry

end NUMINAMATH_GPT_find_other_root_l2084_208413


namespace NUMINAMATH_GPT_area_under_arccos_cos_l2084_208401

noncomputable def func (x : ℝ) : ℝ := Real.arccos (Real.cos x)

theorem area_under_arccos_cos :
  ∫ x in (0:ℝ)..3 * Real.pi, func x = 3 * Real.pi ^ 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_under_arccos_cos_l2084_208401


namespace NUMINAMATH_GPT_fort_blocks_count_l2084_208478

noncomputable def volume_of_blocks (l w h : ℕ) (wall_thickness floor_thickness top_layer_volume : ℕ) : ℕ :=
  let interior_length := l - 2 * wall_thickness
  let interior_width := w - 2 * wall_thickness
  let interior_height := h - floor_thickness
  let volume_original := l * w * h
  let volume_interior := interior_length * interior_width * interior_height
  volume_original - volume_interior + top_layer_volume

theorem fort_blocks_count : volume_of_blocks 15 12 7 2 1 180 = 912 :=
by
  sorry

end NUMINAMATH_GPT_fort_blocks_count_l2084_208478


namespace NUMINAMATH_GPT_sample_size_proportion_l2084_208451

theorem sample_size_proportion (n : ℕ) (ratio_A B C : ℕ) (A_sample : ℕ) (ratio_A_val : ratio_A = 5) (ratio_B_val : ratio_B = 2) (ratio_C_val : ratio_C = 3) (A_sample_val : A_sample = 15) (total_ratio : ratio_A + ratio_B + ratio_C = 10) : 
  15 / n = 5 / 10 → n = 30 :=
sorry

end NUMINAMATH_GPT_sample_size_proportion_l2084_208451


namespace NUMINAMATH_GPT_arctan_sum_eq_pi_div_two_l2084_208474

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arctan_sum_eq_pi_div_two_l2084_208474


namespace NUMINAMATH_GPT_james_goals_product_l2084_208414

theorem james_goals_product :
  ∃ (g7 g8 : ℕ), g7 < 7 ∧ g8 < 7 ∧ 
  (22 + g7) % 7 = 0 ∧ (22 + g7 + g8) % 8 = 0 ∧ 
  g7 * g8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_james_goals_product_l2084_208414


namespace NUMINAMATH_GPT_lena_nicole_candy_difference_l2084_208473

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end NUMINAMATH_GPT_lena_nicole_candy_difference_l2084_208473


namespace NUMINAMATH_GPT_downstream_speed_l2084_208469

noncomputable def V_b : ℝ := 7
noncomputable def V_up : ℝ := 4
noncomputable def V_s : ℝ := V_b - V_up

theorem downstream_speed :
  V_b + V_s = 10 := sorry

end NUMINAMATH_GPT_downstream_speed_l2084_208469


namespace NUMINAMATH_GPT_math_problem_l2084_208462

-- Conditions
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1
def eccentricity (a c : ℝ) : Prop := c / a = (Real.sqrt 2) / 2
def major_axis_length (a : ℝ) : Prop := 2 * a = 6 * Real.sqrt 2

-- Equations and properties to be proven
def ellipse_equation : Prop := ∃ a b : ℝ, a = 3 * Real.sqrt 2 ∧ b = 3 ∧ ellipse_eq a b
def length_AB (θ : ℝ) : Prop := ∃ AB : ℝ, AB = (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2)
def min_AB_CD : Prop := ∃ θ : ℝ, (Real.sin (2 * θ) = 1) ∧ (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2) + (6 * Real.sqrt 2) / (1 + (Real.cos θ)^2) = 8 * Real.sqrt 2

-- The complete proof problem
theorem math_problem : ellipse_equation ∧
                       (∀ θ : ℝ, length_AB θ) ∧
                       min_AB_CD := by
  sorry

end NUMINAMATH_GPT_math_problem_l2084_208462


namespace NUMINAMATH_GPT_tan_beta_minus_2alpha_l2084_208461

theorem tan_beta_minus_2alpha
  (α β : ℝ)
  (h1 : Real.tan α = 1/2)
  (h2 : Real.tan (α - β) = -1/3) :
  Real.tan (β - 2 * α) = -1/7 := 
sorry

end NUMINAMATH_GPT_tan_beta_minus_2alpha_l2084_208461


namespace NUMINAMATH_GPT_relationship_of_a_b_l2084_208427

theorem relationship_of_a_b
  (a b : Real)
  (h1 : a < 0)
  (h2 : b > 0)
  (h3 : a + b < 0) : 
  -a > b ∧ b > -b ∧ -b > a := 
by
  sorry

end NUMINAMATH_GPT_relationship_of_a_b_l2084_208427


namespace NUMINAMATH_GPT_emily_cleaning_time_l2084_208450

noncomputable def total_time : ℝ := 8 -- total time in hours
noncomputable def lilly_fiona_time : ℝ := 1/4 * total_time -- Lilly and Fiona's combined time in hours
noncomputable def jack_time : ℝ := 1/3 * total_time -- Jack's time in hours
noncomputable def emily_time : ℝ := total_time - lilly_fiona_time - jack_time -- Emily's time in hours
noncomputable def emily_time_minutes : ℝ := emily_time * 60 -- Emily's time in minutes

theorem emily_cleaning_time :
  emily_time_minutes = 200 := by
  sorry

end NUMINAMATH_GPT_emily_cleaning_time_l2084_208450


namespace NUMINAMATH_GPT_length_of_train_l2084_208449

theorem length_of_train
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_mps : ℝ := speed_kmph * (1000 / 3600))
  (total_distance : ℝ := train_speed_mps * crossing_time)
  (train_length : ℝ := total_distance - platform_length)
  (h_speed : speed_kmph = 72)
  (h_platform : platform_length = 260)
  (h_time : crossing_time = 26)
  : train_length = 260 := by
  sorry

end NUMINAMATH_GPT_length_of_train_l2084_208449


namespace NUMINAMATH_GPT_factor_polynomial_l2084_208491

theorem factor_polynomial : 
  (x : ℝ) → (x^2 - 6 * x + 9 - 49 * x^4) = (-7 * x^2 + x - 3) * (7 * x^2 + x - 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l2084_208491


namespace NUMINAMATH_GPT_sum_of_digits_l2084_208407

theorem sum_of_digits (a b c d : ℕ) (h_diff : ∀ x y : ℕ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) (h1 : a + c = 10) (h2 : b + c = 8) (h3 : a + d = 11) : 
  a + b + c + d = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l2084_208407


namespace NUMINAMATH_GPT_correct_multiplication_result_l2084_208418

theorem correct_multiplication_result (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 :=
  by
  sorry

end NUMINAMATH_GPT_correct_multiplication_result_l2084_208418


namespace NUMINAMATH_GPT_pipe_fill_rate_l2084_208497

variable (R_A R_B : ℝ)

theorem pipe_fill_rate :
  R_A = 1 / 32 →
  R_A + R_B = 1 / 6.4 →
  R_B / R_A = 4 :=
by
  intros hRA hSum
  have hRA_pos : R_A ≠ 0 := by linarith
  sorry

end NUMINAMATH_GPT_pipe_fill_rate_l2084_208497


namespace NUMINAMATH_GPT_reflection_correct_l2084_208482

/-- Definition of reflection across the line y = -x -/
def reflection_across_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- Given points C and D, and their images C' and D' respectively, under reflection,
    prove the transformation is correct. -/
theorem reflection_correct :
  (reflection_across_y_eq_neg_x (-3, 2) = (3, -2)) ∧ (reflection_across_y_eq_neg_x (-2, 5) = (2, -5)) :=
  by
    sorry

end NUMINAMATH_GPT_reflection_correct_l2084_208482


namespace NUMINAMATH_GPT_jerrys_age_l2084_208454

theorem jerrys_age (M J : ℕ) (h1 : M = 3 * J - 4) (h2 : M = 14) : J = 6 :=
by 
  sorry

end NUMINAMATH_GPT_jerrys_age_l2084_208454


namespace NUMINAMATH_GPT_train_crossing_time_l2084_208452

theorem train_crossing_time
  (train_length : ℕ)           -- length of the train in meters
  (train_speed_kmh : ℕ)        -- speed of the train in kilometers per hour
  (conversion_factor : ℕ)      -- conversion factor from km/hr to m/s
  (train_speed_ms : ℕ)         -- speed of the train in meters per second
  (time_to_cross : ℚ)          -- time to cross in seconds
  (h1 : train_length = 60)
  (h2 : train_speed_kmh = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : train_speed_ms = train_speed_kmh * conversion_factor)
  (h5 : time_to_cross = train_length / train_speed_ms) :
  time_to_cross = 1.5 :=
by sorry

end NUMINAMATH_GPT_train_crossing_time_l2084_208452


namespace NUMINAMATH_GPT_function_satisfies_equation_l2084_208453

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x - 1)

theorem function_satisfies_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x := by
  sorry

end NUMINAMATH_GPT_function_satisfies_equation_l2084_208453


namespace NUMINAMATH_GPT_sum_of_interior_edges_l2084_208439

-- Conditions
def width_of_frame_piece : ℝ := 1.5
def one_interior_edge : ℝ := 4.5
def total_frame_area : ℝ := 27

-- Statement of the problem as a theorem in Lean
theorem sum_of_interior_edges : 
  (∃ y : ℝ, (width_of_frame_piece * 2 + one_interior_edge) * (width_of_frame_piece * 2 + y) 
    - one_interior_edge * y = total_frame_area) →
  (4 * (one_interior_edge + y) = 12) :=
sorry

end NUMINAMATH_GPT_sum_of_interior_edges_l2084_208439


namespace NUMINAMATH_GPT_unique_a_for_intersection_l2084_208424

def A (a : ℝ) : Set ℝ := {-4, 2 * a - 1, a^2}
def B (a : ℝ) : Set ℝ := {a - 5, 1 - a, 9}

theorem unique_a_for_intersection (a : ℝ) :
  (9 ∈ A a ∩ B a ∧ ∀ x, x ∈ A a ∩ B a → x = 9) ↔ a = -3 := by
  sorry

end NUMINAMATH_GPT_unique_a_for_intersection_l2084_208424


namespace NUMINAMATH_GPT_value_of_x0_l2084_208426

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def f_deriv (x : ℝ) : ℝ := ((x - 1) * Real.exp x) / (x * x)

theorem value_of_x0 (x0 : ℝ) (h : f_deriv x0 = -f x0) : x0 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_value_of_x0_l2084_208426


namespace NUMINAMATH_GPT_least_possible_faces_two_dice_l2084_208470

noncomputable def least_possible_sum_of_faces (a b : ℕ) : ℕ :=
(a + b)

theorem least_possible_faces_two_dice (a b : ℕ) (h1 : 8 ≤ a) (h2 : 8 ≤ b)
  (h3 : ∃ k, 9 * k = 2 * (11 * k)) 
  (h4 : ∃ m, 9 * m = a * b) : 
  least_possible_sum_of_faces a b = 22 :=
sorry

end NUMINAMATH_GPT_least_possible_faces_two_dice_l2084_208470


namespace NUMINAMATH_GPT_equivalent_integer_l2084_208409

theorem equivalent_integer (a b n : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) (hn : 200 ≤ n ∧ n ≤ 251) : 
  a - b ≡ 248 [ZMOD 60] :=
sorry

end NUMINAMATH_GPT_equivalent_integer_l2084_208409


namespace NUMINAMATH_GPT_sum_of_intercepts_l2084_208444

theorem sum_of_intercepts (x₀ y₀ : ℕ) (hx₀ : 4 * x₀ ≡ 2 [MOD 25]) (hy₀ : 5 * y₀ ≡ 23 [MOD 25]) 
  (hx_cond : x₀ < 25) (hy_cond : y₀ < 25) : x₀ + y₀ = 28 :=
  sorry

end NUMINAMATH_GPT_sum_of_intercepts_l2084_208444


namespace NUMINAMATH_GPT_min_distance_parabola_l2084_208479

open Real

theorem min_distance_parabola {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1) : ∃ m : ℝ, m = 2 * sqrt 3 ∧ ∀ Q : ℝ × ℝ, Q = (4, 0) → dist P Q ≥ m :=
by sorry

end NUMINAMATH_GPT_min_distance_parabola_l2084_208479


namespace NUMINAMATH_GPT_min_sum_of_dimensions_l2084_208460

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a * b * c = 1645) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  a + b + c ≥ 129 :=
sorry

end NUMINAMATH_GPT_min_sum_of_dimensions_l2084_208460


namespace NUMINAMATH_GPT_riding_is_four_times_walking_l2084_208400

variable (D : ℝ) -- Total distance of the route
variable (v_r v_w : ℝ) -- Riding speed and walking speed
variable (t_r t_w : ℝ) -- Time spent riding and walking

-- Conditions given in the problem
axiom distance_riding : (2/3) * D = v_r * t_r
axiom distance_walking : (1/3) * D = v_w * t_w
axiom time_relation : t_w = 2 * t_r

-- Desired statement to prove
theorem riding_is_four_times_walking : v_r = 4 * v_w := by
  sorry

end NUMINAMATH_GPT_riding_is_four_times_walking_l2084_208400


namespace NUMINAMATH_GPT_range_of_a_l2084_208434

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) (f_decreasing : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : 
  1/2 ≤ a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2084_208434


namespace NUMINAMATH_GPT_intersection_A_B_l2084_208435

def A : Set Real := { y | ∃ x : Real, y = Real.cos x }
def B : Set Real := { x | x^2 < 9 }

theorem intersection_A_B : A ∩ B = { y | -1 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2084_208435


namespace NUMINAMATH_GPT_tetrahedron_volume_from_cube_l2084_208408

theorem tetrahedron_volume_from_cube {s : ℝ} (h : s = 8) :
  let cube_volume := s^3
  let smaller_tetrahedron_volume := (1/3) * (1/2) * s * s * s
  let total_smaller_tetrahedron_volume := 4 * smaller_tetrahedron_volume
  let tetrahedron_volume := cube_volume - total_smaller_tetrahedron_volume
  tetrahedron_volume = 170.6666 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_from_cube_l2084_208408


namespace NUMINAMATH_GPT_contrapositive_of_quadratic_l2084_208485

theorem contrapositive_of_quadratic (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_quadratic_l2084_208485


namespace NUMINAMATH_GPT_quadratic_radical_same_type_l2084_208402

theorem quadratic_radical_same_type (a : ℝ) (h : (∃ (t : ℝ), t ^ 2 = 3 * a - 4) ∧ (∃ (t : ℝ), t ^ 2 = 8)) : a = 2 :=
by
  -- Extract the properties of the radicals
  sorry

end NUMINAMATH_GPT_quadratic_radical_same_type_l2084_208402


namespace NUMINAMATH_GPT_james_hours_per_day_l2084_208425

theorem james_hours_per_day (h : ℕ) (rental_rate : ℕ) (days_per_week : ℕ) (weekly_income : ℕ)
  (H1 : rental_rate = 20)
  (H2 : days_per_week = 4)
  (H3 : weekly_income = 640)
  (H4 : rental_rate * days_per_week * h = weekly_income) :
  h = 8 :=
sorry

end NUMINAMATH_GPT_james_hours_per_day_l2084_208425


namespace NUMINAMATH_GPT_virus_infection_l2084_208423

theorem virus_infection (x : ℕ) (h : 1 + x + x^2 = 121) : x = 10 := 
sorry

end NUMINAMATH_GPT_virus_infection_l2084_208423


namespace NUMINAMATH_GPT_sum_sequence_up_to_2015_l2084_208446

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end NUMINAMATH_GPT_sum_sequence_up_to_2015_l2084_208446


namespace NUMINAMATH_GPT_new_person_weight_l2084_208464

-- Define the total number of persons and their average weight increase
def num_persons : ℕ := 9
def avg_increase : ℝ := 1.5

-- Define the weight of the person being replaced
def weight_of_replaced_person : ℝ := 65

-- Define the total increase in weight
def total_increase_in_weight : ℝ := num_persons * avg_increase

-- Define the weight of the new person
def weight_of_new_person : ℝ := weight_of_replaced_person + total_increase_in_weight

-- Theorem to prove the weight of the new person is 78.5 kg
theorem new_person_weight : weight_of_new_person = 78.5 := by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_new_person_weight_l2084_208464


namespace NUMINAMATH_GPT_evaluate_f_at_7_l2084_208443

theorem evaluate_f_at_7 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 4) = f x)
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4) :
  f 7 = -3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_7_l2084_208443


namespace NUMINAMATH_GPT_find_x_plus_y_l2084_208458

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 16) (h2 : x + |y| - y = 18) : x + y = 6 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l2084_208458


namespace NUMINAMATH_GPT_scientific_notation_of_number_l2084_208472

theorem scientific_notation_of_number :
  1214000 = 1.214 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_number_l2084_208472


namespace NUMINAMATH_GPT_p_distance_300_l2084_208463

-- Assume q's speed is v meters per second, and the race ends in a tie
variables (v : ℝ) (t : ℝ)
variable (d : ℝ)

-- Conditions
def q_speed : ℝ := v
def p_speed : ℝ := 1.25 * v
def q_distance : ℝ := d
def p_distance : ℝ := d + 60

-- Time equations
def q_time_eq : Prop := d = v * t
def p_time_eq : Prop := d + 60 = (1.25 * v) * t

-- Given the conditions, prove that p ran 300 meters in the race
theorem p_distance_300
  (v_pos : v > 0) 
  (t_pos : t > 0)
  (q_time : q_time_eq v d t)
  (p_time : p_time_eq v d t) :
  p_distance d = 300 :=
by
  sorry

end NUMINAMATH_GPT_p_distance_300_l2084_208463


namespace NUMINAMATH_GPT_third_part_of_division_l2084_208421

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end NUMINAMATH_GPT_third_part_of_division_l2084_208421


namespace NUMINAMATH_GPT_frisbee_price_l2084_208498

theorem frisbee_price 
  (total_frisbees : ℕ)
  (frisbees_at_3 : ℕ)
  (price_x_frisbees : ℕ)
  (total_revenue : ℕ) 
  (min_frisbees_at_x : ℕ)
  (price_at_3 : ℕ) 
  (n_min_at_x : ℕ)
  (h1 : total_frisbees = 60)
  (h2 : price_at_3 = 3)
  (h3 : total_revenue = 200)
  (h4 : n_min_at_x = 20)
  (h5 : min_frisbees_at_x >= n_min_at_x)
  : price_x_frisbees = 4 :=
by
  sorry

end NUMINAMATH_GPT_frisbee_price_l2084_208498


namespace NUMINAMATH_GPT_sandy_age_correct_l2084_208415

def is_age_ratio (S M : ℕ) : Prop := S * 9 = M * 7
def is_age_difference (S M : ℕ) : Prop := M = S + 12

theorem sandy_age_correct (S M : ℕ) (h1 : is_age_ratio S M) (h2 : is_age_difference S M) : S = 42 := by
  sorry

end NUMINAMATH_GPT_sandy_age_correct_l2084_208415


namespace NUMINAMATH_GPT_junior_average_score_l2084_208480

def total_students : ℕ := 20
def proportion_juniors : ℝ := 0.2
def proportion_seniors : ℝ := 0.8
def average_class_score : ℝ := 78
def average_senior_score : ℝ := 75

theorem junior_average_score :
  let num_juniors := total_students * proportion_juniors
  let num_seniors := total_students * proportion_seniors
  let total_score := total_students * average_class_score
  let total_senior_score := num_seniors * average_senior_score
  let total_junior_score := total_score - total_senior_score
  total_junior_score / num_juniors = 90 := 
by
  sorry

end NUMINAMATH_GPT_junior_average_score_l2084_208480


namespace NUMINAMATH_GPT_find_PF2_l2084_208465

open Real

noncomputable def hyperbola_equation (x y : ℝ) := (x^2 / 16) - (y^2 / 20) = 1

noncomputable def distance (P F : ℝ × ℝ) : ℝ := 
  let (px, py) := P
  let (fx, fy) := F
  sqrt ((px - fx)^2 + (py - fy)^2)

theorem find_PF2
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (on_hyperbola : hyperbola_equation P.1 P.2)
  (foci_F1_F2 : F1 = (-6, 0) ∧ F2 = (6, 0))
  (distance_PF1 : distance P F1 = 9) : 
  distance P F2 = 17 := 
by
  sorry

end NUMINAMATH_GPT_find_PF2_l2084_208465


namespace NUMINAMATH_GPT_find_natural_numbers_l2084_208428

theorem find_natural_numbers (x y z : ℕ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_ordered : x < y ∧ y < z)
  (h_reciprocal_sum_nat : ∃ a : ℕ, 1/x + 1/y + 1/z = a) : (x, y, z) = (2, 3, 6) := 
sorry

end NUMINAMATH_GPT_find_natural_numbers_l2084_208428


namespace NUMINAMATH_GPT_perimeter_of_structure_l2084_208445

noncomputable def structure_area : ℝ := 576
noncomputable def num_squares : ℕ := 9
noncomputable def square_area : ℝ := structure_area / num_squares
noncomputable def side_length : ℝ := Real.sqrt square_area
noncomputable def perimeter (side_length : ℝ) : ℝ := 8 * side_length

theorem perimeter_of_structure : perimeter side_length = 64 := by
  -- proof will follow here
  sorry

end NUMINAMATH_GPT_perimeter_of_structure_l2084_208445


namespace NUMINAMATH_GPT_smallest_of_three_consecutive_l2084_208457

theorem smallest_of_three_consecutive (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_three_consecutive_l2084_208457


namespace NUMINAMATH_GPT_four_consecutive_integers_plus_one_is_square_l2084_208476

theorem four_consecutive_integers_plus_one_is_square (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n ^ 2 + n - 1) ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_four_consecutive_integers_plus_one_is_square_l2084_208476


namespace NUMINAMATH_GPT_inequality_solution_set_l2084_208492

theorem inequality_solution_set (x : ℝ) :
  (1 / |x - 1| > 3 / 2) ↔ (1 / 3 < x ∧ x < 5 / 3 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2084_208492


namespace NUMINAMATH_GPT_Spot_dog_reachable_area_l2084_208440

noncomputable def Spot_reachable_area (side_length tether_length : ℝ) : ℝ := 
  -- Note here we compute using the areas described in the problem
  6 * Real.pi * (tether_length^2) / 3 - Real.pi * (side_length^2)

theorem Spot_dog_reachable_area (side_length tether_length : ℝ)
  (H1 : side_length = 2) (H2 : tether_length = 3) :
    Spot_reachable_area side_length tether_length = (22 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_GPT_Spot_dog_reachable_area_l2084_208440


namespace NUMINAMATH_GPT_find_real_m_of_purely_imaginary_z_l2084_208433

theorem find_real_m_of_purely_imaginary_z (m : ℝ) 
  (h1 : m^2 - 8 * m + 15 = 0) 
  (h2 : m^2 - 9 * m + 18 ≠ 0) : 
  m = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_real_m_of_purely_imaginary_z_l2084_208433


namespace NUMINAMATH_GPT_unique_solution_of_system_l2084_208487

theorem unique_solution_of_system (n k m : ℕ) (hnk : n + k = Nat.gcd n k ^ 2) (hkm : k + m = Nat.gcd k m ^ 2) (hmn : m + n = Nat.gcd m n ^ 2) : 
  n = 2 ∧ k = 2 ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_system_l2084_208487


namespace NUMINAMATH_GPT_bulbs_on_perfect_squares_l2084_208471

def is_on (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

theorem bulbs_on_perfect_squares (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  (∀ i : ℕ, 1 ≤ i → i ≤ 100 → ∃ j : ℕ, i = j * j ↔ is_on i) := sorry

end NUMINAMATH_GPT_bulbs_on_perfect_squares_l2084_208471


namespace NUMINAMATH_GPT_sum_of_coefficients_l2084_208459

theorem sum_of_coefficients (a : Fin 7 → ℕ) (x : ℕ) : 
  (1 - x) ^ 6 = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 → 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 := 
by
  intro h
  by_cases hx : x = 1
  · rw [hx] at h
    sorry
  · sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2084_208459


namespace NUMINAMATH_GPT_closest_point_on_line_to_target_l2084_208489

noncomputable def parametricPoint (s : ℝ) : ℝ × ℝ × ℝ :=
  (6 + 3 * s, 2 - 9 * s, 0 + 6 * s)

noncomputable def closestPoint : ℝ × ℝ × ℝ :=
  (249/42, 95/42, -1/7)

theorem closest_point_on_line_to_target :
  ∃ s : ℝ, parametricPoint s = closestPoint :=
by
  sorry

end NUMINAMATH_GPT_closest_point_on_line_to_target_l2084_208489


namespace NUMINAMATH_GPT_hotel_floors_l2084_208499

/-- Given:
  - Each floor has 10 identical rooms.
  - The last floor is unavailable for guests.
  - Hans could be checked into 90 different rooms.
  - There are no other guests.
 - Prove that the total number of floors in the hotel is 10.
--/
theorem hotel_floors :
  (∃ n : ℕ, n ≥ 1 ∧ 10 * (n - 1) = 90) → n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_hotel_floors_l2084_208499


namespace NUMINAMATH_GPT_probability_of_ge_four_is_one_eighth_l2084_208420

noncomputable def probability_ge_four : ℝ :=
sorry

theorem probability_of_ge_four_is_one_eighth :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) →
  (probability_ge_four = 1 / 8) :=
sorry

end NUMINAMATH_GPT_probability_of_ge_four_is_one_eighth_l2084_208420


namespace NUMINAMATH_GPT_find_value_of_s_l2084_208455

variable {r s : ℝ}

theorem find_value_of_s (hr : r > 1) (hs : s > 1) (h1 : 1/r + 1/s = 1) (h2 : r * s = 9) :
  s = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_s_l2084_208455


namespace NUMINAMATH_GPT_percentage_markup_l2084_208411

/--
The owner of a furniture shop charges his customer a certain percentage more than the cost price.
A customer paid Rs. 3000 for a computer table, and the cost price of the computer table was Rs. 2500.
Prove that the percentage markup on the cost price is 20%.
-/
theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 3000) (h₂ : cost_price = 2500) :
  ((selling_price - cost_price) / cost_price) * 100 = 20 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_percentage_markup_l2084_208411


namespace NUMINAMATH_GPT_intersection_points_count_l2084_208477

-- Definition of the two equations as conditions
def eq1 (x y : ℝ) : Prop := y = 3 * x^2
def eq2 (x y : ℝ) : Prop := y^2 - 6 * y + 8 = x^2

-- The theorem stating that the number of intersection points of the two graphs is exactly 4
theorem intersection_points_count : 
  ∃ (points : Finset (ℝ × ℝ)), (∀ p : ℝ × ℝ, p ∈ points ↔ eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_count_l2084_208477


namespace NUMINAMATH_GPT_total_days_on_jury_duty_l2084_208484

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end NUMINAMATH_GPT_total_days_on_jury_duty_l2084_208484


namespace NUMINAMATH_GPT_part1_l2084_208403

theorem part1 (x : ℝ) (hx : x > 0) : 
  (1 / (2 * Real.sqrt (x + 1))) < (Real.sqrt (x + 1) - Real.sqrt x) ∧ (Real.sqrt (x + 1) - Real.sqrt x) < (1 / (2 * Real.sqrt x)) := 
sorry

end NUMINAMATH_GPT_part1_l2084_208403


namespace NUMINAMATH_GPT_all_rationals_on_number_line_l2084_208467

theorem all_rationals_on_number_line :
  ∀ q : ℚ, ∃ p : ℝ, p = ↑q :=
by
  sorry

end NUMINAMATH_GPT_all_rationals_on_number_line_l2084_208467


namespace NUMINAMATH_GPT_volume_of_stone_l2084_208404

def width := 16
def length := 14
def full_height := 9
def initial_water_height := 4
def final_water_height := 9

def volume_before := length * width * initial_water_height
def volume_after := length * width * final_water_height

def volume_stone := volume_after - volume_before

theorem volume_of_stone : volume_stone = 1120 := by
  unfold volume_stone
  unfold volume_after volume_before
  unfold final_water_height initial_water_height width length
  sorry

end NUMINAMATH_GPT_volume_of_stone_l2084_208404


namespace NUMINAMATH_GPT_find_a5_l2084_208475

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (a1 : ℝ)

-- Geometric sequence definition
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a1 * q^n

-- Given conditions
def condition1 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 1 + a 3 = 10

def condition2 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 2 + a 4 = -30

-- Theorem to prove
theorem find_a5 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h1 : geometric_sequence a a1 q)
  (h2 : condition1 a a1 q)
  (h3 : condition2 a a1 q) :
  a 5 = 81 := by
  sorry

end NUMINAMATH_GPT_find_a5_l2084_208475


namespace NUMINAMATH_GPT_minimum_people_who_like_both_l2084_208412

open Nat

theorem minimum_people_who_like_both (total : ℕ) (mozart : ℕ) (bach : ℕ)
  (h_total: total = 100) (h_mozart: mozart = 87) (h_bach: bach = 70) :
  ∃ x, x = mozart + bach - total ∧ x ≥ 57 :=
by
  sorry

end NUMINAMATH_GPT_minimum_people_who_like_both_l2084_208412


namespace NUMINAMATH_GPT_largest_int_less_than_100_rem_5_by_7_l2084_208432

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end NUMINAMATH_GPT_largest_int_less_than_100_rem_5_by_7_l2084_208432


namespace NUMINAMATH_GPT_solve_for_2a_2d_l2084_208483

noncomputable def f (a b c d x : ℝ) : ℝ :=
  (2 * a * x + b) / (c * x + 2 * d)

theorem solve_for_2a_2d (a b c d : ℝ) (habcd_ne_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h : ∀ x, f a b c d (f a b c d x) = x) : 2 * a + 2 * d = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_2a_2d_l2084_208483


namespace NUMINAMATH_GPT_series_sum_eq_half_l2084_208490

theorem series_sum_eq_half : ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_eq_half_l2084_208490


namespace NUMINAMATH_GPT_simplify_expression_l2084_208442

-- Definitions for conditions and parameters
variables {x y : ℝ}

-- The problem statement and proof
theorem simplify_expression : 12 * x^5 * y / (6 * x * y) = 2 * x^4 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2084_208442


namespace NUMINAMATH_GPT_max_square_area_in_rhombus_l2084_208486

noncomputable def side_length_triangle := 10
noncomputable def height_triangle := Real.sqrt (side_length_triangle^2 - (side_length_triangle / 2)^2)
noncomputable def diag_long := 2 * height_triangle
noncomputable def diag_short := side_length_triangle
noncomputable def side_square := diag_short / Real.sqrt 2
noncomputable def area_square := side_square^2

theorem max_square_area_in_rhombus :
  area_square = 50 := by sorry

end NUMINAMATH_GPT_max_square_area_in_rhombus_l2084_208486


namespace NUMINAMATH_GPT_ages_proof_l2084_208495

def hans_now : ℕ := 8

def sum_ages (annika_now emil_now frida_now : ℕ) :=
  hans_now + annika_now + emil_now + frida_now = 58

def annika_age_in_4_years (annika_now : ℕ) : ℕ :=
  3 * (hans_now + 4)

def emil_age_in_4_years (emil_now : ℕ) : ℕ :=
  2 * (hans_now + 4)

def frida_age_in_4_years (frida_now : ℕ) :=
  2 * 12

def annika_frida_age_difference (annika_now frida_now : ℕ) : Prop :=
  annika_now = frida_now + 5

theorem ages_proof :
  ∃ (annika_now emil_now frida_now : ℕ),
    sum_ages annika_now emil_now frida_now ∧
    annika_age_in_4_years annika_now = 36 ∧
    emil_age_in_4_years emil_now = 24 ∧
    frida_age_in_4_years frida_now = 24 ∧
    annika_frida_age_difference annika_now frida_now :=
by
  sorry

end NUMINAMATH_GPT_ages_proof_l2084_208495


namespace NUMINAMATH_GPT_monotonicity_F_range_k_l2084_208436

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x + a * x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x - k * (x^3 - 3 * x)

theorem monotonicity_F (a : ℝ) (ha : a ≠ 0) :
(∀ x : ℝ, (-1 < x ∧ x < 1) → 
    (if (-2 ≤ a ∧ a < 0) ∨ (a > 0) then 0 ≤ (a - a * x^2 + 2) / (1 - x^2)
     else if a < -2 then 
        ((-1 < x ∧ x < -Real.sqrt ((a + 2) / a)) ∨ (Real.sqrt ((a + 2) / a) < x ∧ x < 1)) → 0 ≤ (a - a * x^2 + 2) / (1 - x^2) ∧ 
        (-Real.sqrt ((a + 2) / a) < x ∧ x < Real.sqrt ((a + 2) / a)) → 0 > (a - a * x^2 + 2) / (1 - x^2)
    else false)) :=
sorry

theorem range_k (k : ℝ) (hk : ∀ x : ℝ, (0 < x ∧ x < 1) → f x > k * (x^3 - 3 * x)) :
k ≥ -2 / 3 :=
sorry

end NUMINAMATH_GPT_monotonicity_F_range_k_l2084_208436


namespace NUMINAMATH_GPT_julie_age_end_of_period_is_15_l2084_208431

-- Define necessary constants and variables
def hours_per_day : ℝ := 3
def pay_rate_per_hour_per_year : ℝ := 0.75
def total_days_worked : ℝ := 60
def total_earnings : ℝ := 810

-- Define Julie's age at the end of the four-month period
def julies_age_end_of_period (age: ℝ) : Prop :=
  hours_per_day * pay_rate_per_hour_per_year * age * total_days_worked = total_earnings

-- The final Lean 4 statement that needs proof
theorem julie_age_end_of_period_is_15 : ∃ age : ℝ, julies_age_end_of_period age ∧ age = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_julie_age_end_of_period_is_15_l2084_208431


namespace NUMINAMATH_GPT_middle_pile_cards_l2084_208466

theorem middle_pile_cards (x : Nat) (h : x ≥ 2) : 
    let left := x - 2
    let middle := x + 2
    let right := x
    let middle_after_step3 := middle + 1
    let final_middle := middle_after_step3 - left
    final_middle = 5 := 
by
  sorry

end NUMINAMATH_GPT_middle_pile_cards_l2084_208466


namespace NUMINAMATH_GPT_systematic_sampling_remove_l2084_208410

theorem systematic_sampling_remove (total_people : ℕ) (sample_size : ℕ) (remove_count : ℕ): 
  total_people = 162 → sample_size = 16 → remove_count = 2 → 
  (total_people - 1) % sample_size = sample_size - 1 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_remove_l2084_208410


namespace NUMINAMATH_GPT_Cameron_task_completion_l2084_208481

theorem Cameron_task_completion (C : ℝ) (h1 : ∃ x, x = 9 / C) (h2 : ∃ y, y = 1 / 2) (total_work : ∃ z, z = 1):
  9 - 9 / C + 1/2 = 1 -> C = 18 := by
  sorry

end NUMINAMATH_GPT_Cameron_task_completion_l2084_208481


namespace NUMINAMATH_GPT_sum_of_heights_less_than_perimeter_l2084_208488

theorem sum_of_heights_less_than_perimeter
  (a b c h1 h2 h3 : ℝ) 
  (H1 : h1 ≤ b) 
  (H2 : h2 ≤ c) 
  (H3 : h3 ≤ a) 
  (H4 : h1 < b ∨ h2 < c ∨ h3 < a) : 
  h1 + h2 + h3 < a + b + c :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_heights_less_than_perimeter_l2084_208488


namespace NUMINAMATH_GPT_inequality_proof_l2084_208416

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)

theorem inequality_proof :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) >= 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2084_208416


namespace NUMINAMATH_GPT_extremum_at_one_eq_a_one_l2084_208437

theorem extremum_at_one_eq_a_one 
  (a : ℝ) 
  (h : ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * a * x^2 - 3) ∧ f' 1 = 0) : 
  a = 1 :=
sorry

end NUMINAMATH_GPT_extremum_at_one_eq_a_one_l2084_208437


namespace NUMINAMATH_GPT_TotalLaddersClimbedInCentimeters_l2084_208419

def keaton_ladder_height := 50  -- height of Keaton's ladder in meters
def keaton_climbs := 30  -- number of times Keaton climbs the ladder

def reece_ladder_height := keaton_ladder_height - 6  -- height of Reece's ladder in meters
def reece_climbs := 25  -- number of times Reece climbs the ladder

def total_meters_climbed := (keaton_ladder_height * keaton_climbs) + (reece_ladder_height * reece_climbs)

def total_cm_climbed := total_meters_climbed * 100

theorem TotalLaddersClimbedInCentimeters :
  total_cm_climbed = 260000 :=
by
  sorry

end NUMINAMATH_GPT_TotalLaddersClimbedInCentimeters_l2084_208419


namespace NUMINAMATH_GPT_prove_by_contradiction_l2084_208447

-- Statement: To prove "a > b" by contradiction, assuming the negation "a ≤ b".
theorem prove_by_contradiction (a b : ℝ) (h : a ≤ b) : false := sorry

end NUMINAMATH_GPT_prove_by_contradiction_l2084_208447


namespace NUMINAMATH_GPT_books_from_library_l2084_208441

def initial_books : ℝ := 54.5
def additional_books_1 : ℝ := 23.7
def returned_books_1 : ℝ := 12.3
def additional_books_2 : ℝ := 15.6
def returned_books_2 : ℝ := 9.1
def additional_books_3 : ℝ := 7.2

def total_books : ℝ :=
  initial_books + additional_books_1 - returned_books_1 + additional_books_2 - returned_books_2 + additional_books_3

theorem books_from_library : total_books = 79.6 := by
  sorry

end NUMINAMATH_GPT_books_from_library_l2084_208441


namespace NUMINAMATH_GPT_find_ratio_l2084_208448

open Nat

def sequence_def (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 →
    (a ((n + 2)) / a ((n + 1))) - (a ((n + 1)) / a n) = d

def geometric_difference_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3 ∧ sequence_def a 2

theorem find_ratio (a : ℕ → ℕ) (h : geometric_difference_sequence a) :
  a 12 / a 10 = 399 := sorry

end NUMINAMATH_GPT_find_ratio_l2084_208448


namespace NUMINAMATH_GPT_sqrt_mul_power_expr_l2084_208422

theorem sqrt_mul_power_expr : ( (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 ) = (Real.sqrt 3 + Real.sqrt 2) := 
  sorry

end NUMINAMATH_GPT_sqrt_mul_power_expr_l2084_208422


namespace NUMINAMATH_GPT_average_ABC_l2084_208430

/-- Given three numbers A, B, and C such that 1503C - 3006A = 6012 and 1503B + 4509A = 7509,
their average is 3  -/
theorem average_ABC (A B C : ℚ) 
  (h1 : 1503 * C - 3006 * A = 6012) 
  (h2 : 1503 * B + 4509 * A = 7509) : 
  (A + B + C) / 3 = 3 :=
sorry

end NUMINAMATH_GPT_average_ABC_l2084_208430


namespace NUMINAMATH_GPT_seventy_five_percent_of_number_l2084_208496

variable (N : ℝ)

theorem seventy_five_percent_of_number :
  (1 / 8) * (3 / 5) * (4 / 7) * (5 / 11) * N - (1 / 9) * (2 / 3) * (3 / 4) * (5 / 8) * N = 30 →
  0.75 * N = -1476 :=
by
  sorry

end NUMINAMATH_GPT_seventy_five_percent_of_number_l2084_208496


namespace NUMINAMATH_GPT_no_such_n_exists_l2084_208429

theorem no_such_n_exists : ∀ n : ℕ, n > 1 → ∀ (p1 p2 : ℕ), 
  (Nat.Prime p1) → (Nat.Prime p2) → n = p1^2 → n + 60 = p2^2 → False :=
by
  intro n hn p1 p2 hp1 hp2 h1 h2
  sorry

end NUMINAMATH_GPT_no_such_n_exists_l2084_208429


namespace NUMINAMATH_GPT_four_digit_number_l2084_208468

def digit_constraint (A B C D : ℕ) : Prop :=
  A = B / 3 ∧ C = A + B ∧ D = 3 * B

theorem four_digit_number 
  (A B C D : ℕ) 
  (h₁ : A = B / 3) 
  (h₂ : C = A + B) 
  (h₃ : D = 3 * B)
  (hA_digit : A < 10) 
  (hB_digit : B < 10)
  (hC_digit : C < 10)
  (hD_digit : D < 10) :
  1000 * A + 100 * B + 10 * C + D = 1349 := 
sorry

end NUMINAMATH_GPT_four_digit_number_l2084_208468


namespace NUMINAMATH_GPT_simplify_expression_l2084_208438

noncomputable def algebraic_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) : ℚ :=
(1 - 3 / (a + 2)) / ((a^2 - 2 * a + 1) / (a^2 - 4))

theorem simplify_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) :
  algebraic_expression a h1 h2 h3 = (a - 2) / (a - 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2084_208438


namespace NUMINAMATH_GPT_determine_b_l2084_208405

theorem determine_b (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x) ∧ (∀ x y : ℝ, y - 2 = (b + 9) * x) → 
  b = -6 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l2084_208405


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2084_208406

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 * a 7 = 8)
  (h2 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1)):
  a 2 + a 8 = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2084_208406


namespace NUMINAMATH_GPT_product_of_xy_l2084_208417

theorem product_of_xy : 
  ∃ (x y : ℝ), 3 * x + 4 * y = 60 ∧ 6 * x - 4 * y = 12 ∧ x * y = 72 :=
by
  sorry

end NUMINAMATH_GPT_product_of_xy_l2084_208417


namespace NUMINAMATH_GPT_average_speed_l2084_208493

theorem average_speed (v1 v2 : ℝ) (h1 : v1 = 110) (h2 : v2 = 88) : 
  (2 * v1 * v2) / (v1 + v2) = 97.78 := 
by sorry

end NUMINAMATH_GPT_average_speed_l2084_208493
