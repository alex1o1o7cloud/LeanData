import Mathlib

namespace ratio_of_surface_areas_l1559_155937

theorem ratio_of_surface_areas (s : ℝ) :
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3 :=
by
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  show (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3
  sorry

end ratio_of_surface_areas_l1559_155937


namespace average_selections_correct_l1559_155924

noncomputable def cars := 18
noncomputable def selections_per_client := 3
noncomputable def clients := 18
noncomputable def total_selections := clients * selections_per_client
noncomputable def average_selections_per_car := total_selections / cars

theorem average_selections_correct :
  average_selections_per_car = 3 :=
by
  sorry

end average_selections_correct_l1559_155924


namespace minimum_value_inequality_l1559_155990

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 :=
sorry

end minimum_value_inequality_l1559_155990


namespace vasya_purchase_l1559_155987

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l1559_155987


namespace sum_of_roots_eq_a_plus_b_l1559_155998

theorem sum_of_roots_eq_a_plus_b (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - (a + b) * x + (ab + 1) = 0 → (x = a ∨ x = b)) :
  a + b = a + b :=
by sorry

end sum_of_roots_eq_a_plus_b_l1559_155998


namespace real_number_solution_l1559_155949

theorem real_number_solution : ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 :=
by
  sorry

end real_number_solution_l1559_155949


namespace sqrt_22_gt_4_l1559_155902

theorem sqrt_22_gt_4 : Real.sqrt 22 > 4 := 
sorry

end sqrt_22_gt_4_l1559_155902


namespace william_probability_l1559_155961

def probability_of_correct_answer (p : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  1 - q^n

theorem william_probability :
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  probability_of_correct_answer p q n = 11529 / 15625 :=
by
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  unfold probability_of_correct_answer
  sorry

end william_probability_l1559_155961


namespace log_exp_identity_l1559_155921

theorem log_exp_identity (a : ℝ) (h : a = Real.log 5 / Real.log 4) : 
  (2^a + 2^(-a) = 6 * Real.sqrt 5 / 5) :=
by {
  -- a = log_4 (5) can be rewritten using change-of-base formula: log 5 / log 4
  -- so, it can be used directly in the theorem
  sorry
}

end log_exp_identity_l1559_155921


namespace pow_ge_double_l1559_155974

theorem pow_ge_double (n : ℕ) : 2^n ≥ 2 * n := sorry

end pow_ge_double_l1559_155974


namespace parabola_passes_through_A_C_l1559_155935

theorem parabola_passes_through_A_C : ∃ (a b : ℝ), (2 = a * 1^2 + b * 1 + 1) ∧ (1 = a * 2^2 + b * 2 + 1) :=
by {
  sorry
}

end parabola_passes_through_A_C_l1559_155935


namespace geometric_sequence_ab_product_l1559_155907

theorem geometric_sequence_ab_product (a b : ℝ) (h₁ : 2 ≤ a) (h₂ : a ≤ 16) (h₃ : 2 ≤ b) (h₄ : b ≤ 16)
  (h₅ : ∃ r : ℝ, a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) : a * b = 32 :=
by
  sorry

end geometric_sequence_ab_product_l1559_155907


namespace octal_addition_l1559_155918

theorem octal_addition (x y : ℕ) (h1 : x = 1 * 8^3 + 4 * 8^2 + 6 * 8^1 + 3 * 8^0)
                     (h2 : y = 2 * 8^2 + 7 * 8^1 + 5 * 8^0) :
  x + y = 1 * 8^3 + 7 * 8^2 + 5 * 8^1 + 0 * 8^0 := sorry

end octal_addition_l1559_155918


namespace derivative_sqrt_l1559_155983

/-- The derivative of the function y = sqrt x is 1 / (2 * sqrt x) -/
theorem derivative_sqrt (x : ℝ) (h : 0 < x) : (deriv (fun x => Real.sqrt x) x) = 1 / (2 * Real.sqrt x) :=
sorry

end derivative_sqrt_l1559_155983


namespace solve_fractional_eq_l1559_155928

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -3) : (1 / x = 6 / (x + 3)) → (x = 0.6) :=
by
  sorry

end solve_fractional_eq_l1559_155928


namespace sum_first_11_terms_l1559_155920

-- Define the arithmetic sequence and sum formula
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions given
variables (a1 d : ℤ)
axiom condition : (a1 + d) + (a1 + 9 * d) = 4

-- Proof statement
theorem sum_first_11_terms : sum_arithmetic_sequence a1 d 11 = 22 :=
by
  -- Placeholder for the actual proof
  sorry

end sum_first_11_terms_l1559_155920


namespace line_equation_l1559_155965

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_sq := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_sq) • u

theorem line_equation :
  ∀ (x y : ℝ), projection (4, 3) (x, y) = (-4, -3) → y = (-4 / 3) * x - 25 / 3 :=
by
  intros x y h
  sorry

end line_equation_l1559_155965


namespace find_the_number_l1559_155964

theorem find_the_number (x : ℕ) (h : 18396 * x = 183868020) : x = 9990 :=
by
  sorry

end find_the_number_l1559_155964


namespace james_vacuuming_hours_l1559_155988

/-- James spends some hours vacuuming and 3 times as long on the rest of his chores. 
    He spends 12 hours on his chores in total. -/
theorem james_vacuuming_hours (V : ℝ) (h : V + 3 * V = 12) : V = 3 := 
sorry

end james_vacuuming_hours_l1559_155988


namespace juan_distance_l1559_155905

def running_time : ℝ := 80.0
def speed : ℝ := 10.0
def distance : ℝ := running_time * speed

theorem juan_distance :
  distance = 800.0 :=
by
  sorry

end juan_distance_l1559_155905


namespace percentage_of_failed_candidates_l1559_155968

theorem percentage_of_failed_candidates :
  let total_candidates := 2000
  let girls := 900
  let boys := total_candidates - girls
  let boys_passed := 32 / 100 * boys
  let girls_passed := 32 / 100 * girls
  let total_passed := boys_passed + girls_passed
  let total_failed := total_candidates - total_passed
  let percentage_failed := (total_failed / total_candidates) * 100
  percentage_failed = 68 :=
by
  -- Proof goes here
  sorry

end percentage_of_failed_candidates_l1559_155968


namespace gasoline_added_l1559_155925

noncomputable def initial_amount (capacity: ℕ) : ℝ :=
  (3 / 4) * capacity

noncomputable def final_amount (capacity: ℕ) : ℝ :=
  (9 / 10) * capacity

theorem gasoline_added (capacity: ℕ) (initial_fraction final_fraction: ℝ) (initial_amount final_amount: ℝ) : 
  capacity = 54 ∧ initial_fraction = 3/4 ∧ final_fraction = 9/10 ∧ 
  initial_amount = initial_fraction * capacity ∧ 
  final_amount = final_fraction * capacity →
  final_amount - initial_amount = 8.1 :=
sorry

end gasoline_added_l1559_155925


namespace ceil_square_range_count_l1559_155970

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l1559_155970


namespace no_integer_solutions_l1559_155957

theorem no_integer_solutions (x y z : ℤ) (h : ¬ (x = 0 ∧ y = 0 ∧ z = 0)) : 2 * x^4 + y^4 ≠ 7 * z^4 :=
sorry

end no_integer_solutions_l1559_155957


namespace range_of_a_l1559_155944

theorem range_of_a (a : ℝ) : 
  (∃! x : ℤ, 4 - 2 * x ≥ 0 ∧ (1 / 2 : ℝ) * x - a > 0) ↔ -1 ≤ a ∧ a < -0.5 :=
by
  sorry

end range_of_a_l1559_155944


namespace field_dimensions_l1559_155922

theorem field_dimensions (W L : ℕ) (h1 : L = 2 * W) (h2 : 2 * L + 2 * W = 600) : W = 100 ∧ L = 200 :=
sorry

end field_dimensions_l1559_155922


namespace seeds_in_first_plot_l1559_155986

theorem seeds_in_first_plot (x : ℕ) (h1 : 0 < x)
  (h2 : 200 = 200)
  (h3 : 0.25 * (x : ℝ) = 0.25 * (x : ℝ))
  (h4 : 0.35 * 200 = 70)
  (h5 : (0.25 * (x : ℝ) + 70) / (x + 200) = 0.29) :
  x = 300 :=
by sorry

end seeds_in_first_plot_l1559_155986


namespace angle_triple_supplement_l1559_155982

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l1559_155982


namespace abs_lt_two_nec_but_not_suff_l1559_155979

theorem abs_lt_two_nec_but_not_suff (x : ℝ) :
  (|x - 1| < 2) → (0 < x ∧ x < 3) ∧ ¬((0 < x ∧ x < 3) → (|x - 1| < 2)) := sorry

end abs_lt_two_nec_but_not_suff_l1559_155979


namespace area_of_grey_part_l1559_155972

theorem area_of_grey_part :
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  area2 - area_white = 65 :=
by
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  have : area2 - area_white = 65 := by sorry
  exact this

end area_of_grey_part_l1559_155972


namespace chord_constant_sum_l1559_155923

theorem chord_constant_sum (d : ℝ) (h : d = 1/2) :
  ∀ A B : ℝ × ℝ, (A.2 = A.1^2) → (B.2 = B.1^2) →
  (∃ m : ℝ, A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
  (∃ D : ℝ × ℝ, D = (0, d) ∧ (∃ s : ℝ,
    s = (1 / ((A.1 - D.1)^2 + (A.2 - D.2)^2) + 1 / ((B.1 - D.1)^2 + (B.2 - D.2)^2)) ∧ s = 4)) :=
by 
  sorry

end chord_constant_sum_l1559_155923


namespace find_q_l1559_155945

theorem find_q (q : ℤ) (x : ℤ) (y : ℤ) (h1 : x = 55 + 2 * q) (h2 : y = 4 * q + 41) (h3 : x = y) : q = 7 :=
by
  sorry

end find_q_l1559_155945


namespace line_passes_through_circle_center_l1559_155967

theorem line_passes_through_circle_center
  (a : ℝ)
  (h_line : ∀ (x y : ℝ), 3 * x + y + a = 0 → (x, y) = (-1, 2))
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (-1, 2)) :
  a = 1 :=
by
  sorry

end line_passes_through_circle_center_l1559_155967


namespace smallest_k_l1559_155994

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l1559_155994


namespace exponentiation_rule_l1559_155963

theorem exponentiation_rule (x : ℝ) : (x^5)^2 = x^10 :=
by {
  sorry
}

end exponentiation_rule_l1559_155963


namespace relationship_of_new_stationary_points_l1559_155946

noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def phi (x : ℝ) : ℝ := x^3

noncomputable def g' (x : ℝ) : ℝ := Real.cos x
noncomputable def h' (x : ℝ) : ℝ := 1 / x
noncomputable def phi' (x : ℝ) : ℝ := 3 * x^2

-- Definitions of the new stationary points
noncomputable def new_stationary_point_g (x : ℝ) : Prop := g x = g' x
noncomputable def new_stationary_point_h (x : ℝ) : Prop := h x = h' x
noncomputable def new_stationary_point_phi (x : ℝ) : Prop := phi x = phi' x

theorem relationship_of_new_stationary_points :
  ∃ (a b c : ℝ), (0 < a ∧ a < π) ∧ (1 < b ∧ b < Real.exp 1) ∧ (c ≠ 0) ∧
  new_stationary_point_g a ∧ new_stationary_point_h b ∧ new_stationary_point_phi c ∧
  c > b ∧ b > a :=
by
  sorry

end relationship_of_new_stationary_points_l1559_155946


namespace percentage_of_315_out_of_900_is_35_l1559_155904

theorem percentage_of_315_out_of_900_is_35 :
  (315 : ℝ) / 900 * 100 = 35 := 
by
  sorry

end percentage_of_315_out_of_900_is_35_l1559_155904


namespace binomial_sum_eq_728_l1559_155909

theorem binomial_sum_eq_728 :
  (Nat.choose 6 1) * 2^1 +
  (Nat.choose 6 2) * 2^2 +
  (Nat.choose 6 3) * 2^3 +
  (Nat.choose 6 4) * 2^4 +
  (Nat.choose 6 5) * 2^5 +
  (Nat.choose 6 6) * 2^6 = 728 :=
by
  sorry

end binomial_sum_eq_728_l1559_155909


namespace john_books_per_day_l1559_155930

theorem john_books_per_day (books_per_week := 2) (weeks := 6) (total_books := 48) :
  (total_books / (books_per_week * weeks) = 4) :=
by
  sorry

end john_books_per_day_l1559_155930


namespace green_tiles_in_50th_row_l1559_155940

-- Conditions
def tiles_in_row (n : ℕ) : ℕ := 2 * n - 1

def green_tiles_in_row (n : ℕ) : ℕ := (tiles_in_row n - 1) / 2

-- Prove the number of green tiles in the 50th row
theorem green_tiles_in_50th_row : green_tiles_in_row 50 = 49 :=
by
  -- Placeholder proof
  sorry

end green_tiles_in_50th_row_l1559_155940


namespace temperature_at_night_l1559_155981

theorem temperature_at_night 
  (T_morning : ℝ) 
  (T_rise_noon : ℝ) 
  (T_drop_night : ℝ) 
  (h1 : T_morning = 22) 
  (h2 : T_rise_noon = 6) 
  (h3 : T_drop_night = 10) : 
  (T_morning + T_rise_noon - T_drop_night = 18) :=
by 
  sorry

end temperature_at_night_l1559_155981


namespace perimeter_greater_than_diagonals_l1559_155910

namespace InscribedQuadrilateral

def is_convex_quadrilateral (AB BC CD DA AC BD: ℝ) : Prop :=
  -- Conditions for a convex quadrilateral (simple check)
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ AC > 0 ∧ BD > 0

def is_inscribed_in_circle (AB BC CD DA AC BD: ℝ) (r: ℝ) : Prop :=
  -- Check if quadrilateral is inscribed in a circle of radius 1
  r = 1

theorem perimeter_greater_than_diagonals 
  (AB BC CD DA AC BD: ℝ) 
  (r: ℝ)
  (h1 : is_convex_quadrilateral AB BC CD DA AC BD) 
  (h2 : is_inscribed_in_circle AB BC CD DA AC BD r) :
  0 < (AB + BC + CD + DA) - (AC + BD) ∧ (AB + BC + CD + DA) - (AC + BD) < 2 :=
by
  sorry 

end InscribedQuadrilateral

end perimeter_greater_than_diagonals_l1559_155910


namespace new_volume_is_80_gallons_l1559_155951

-- Define the original volume
def V_original : ℝ := 5

-- Define the factors by which length, width, and height are increased
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 4

-- Define the new volume
def V_new : ℝ := V_original * (length_factor * width_factor * height_factor)

-- Theorem to prove the new volume is 80 gallons
theorem new_volume_is_80_gallons : V_new = 80 := 
by
  -- Proof goes here
  sorry

end new_volume_is_80_gallons_l1559_155951


namespace toms_total_miles_l1559_155913

-- Define the conditions as facts
def days_in_year : ℕ := 365
def first_part_days : ℕ := 183
def second_part_days : ℕ := days_in_year - first_part_days
def miles_per_day_first_part : ℕ := 30
def miles_per_day_second_part : ℕ := 35

-- State the final theorem
theorem toms_total_miles : 
  (first_part_days * miles_per_day_first_part) + (second_part_days * miles_per_day_second_part) = 11860 := by 
  sorry

end toms_total_miles_l1559_155913


namespace range_of_a_l1559_155996

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^3 - a * x^2 - 4 * a * x + 4 * a^2 - 1 = 0 ∧ ∀ y : ℝ, 
  (y ≠ x → y^3 - a * y^2 - 4 * a * y + 4 * a^2 - 1 ≠ 0)) ↔ a < 3 / 4 := 
sorry

end range_of_a_l1559_155996


namespace female_students_in_sample_l1559_155969

/-- In a high school, there are 500 male students and 400 female students in the first grade. 
    If a random sample of size 45 is taken from the students of this grade using stratified sampling by gender, 
    the number of female students in the sample is 20. -/
theorem female_students_in_sample 
  (num_male : ℕ) (num_female : ℕ) (sample_size : ℕ)
  (h_male : num_male = 500)
  (h_female : num_female = 400)
  (h_sample : sample_size = 45)
  (total_students : ℕ := num_male + num_female)
  (sample_ratio : ℚ := sample_size / total_students) :
  num_female * sample_ratio = 20 := 
sorry

end female_students_in_sample_l1559_155969


namespace sum_log_base_5_divisors_l1559_155993

theorem sum_log_base_5_divisors (n : ℕ) (h : n * (n + 1) / 2 = 264) : n = 23 :=
by
  sorry

end sum_log_base_5_divisors_l1559_155993


namespace find_smallest_value_of_sum_of_squares_l1559_155992
noncomputable def smallest_value (x y z : ℚ) := x^2 + y^2 + z^2

theorem find_smallest_value_of_sum_of_squares :
  ∃ (x y z : ℚ), (x + 4) * (y - 4) = 0 ∧ 3 * z - 2 * y = 5 ∧ smallest_value x y z = 457 / 9 :=
by
  sorry

end find_smallest_value_of_sum_of_squares_l1559_155992


namespace t_minus_s_equals_neg_17_25_l1559_155938

noncomputable def t : ℝ := (60 + 30 + 20 + 5 + 5) / 5
noncomputable def s : ℝ := (60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 5 * (5 / 120) + 5 * (5 / 120))
noncomputable def t_minus_s : ℝ := t - s

theorem t_minus_s_equals_neg_17_25 : t_minus_s = -17.25 := by
  sorry

end t_minus_s_equals_neg_17_25_l1559_155938


namespace seating_arrangements_l1559_155960

theorem seating_arrangements (n : ℕ) (max_capacity : ℕ) 
  (h_n : n = 6) (h_max : max_capacity = 4) :
  ∃ k : ℕ, k = 50 :=
by
  sorry

end seating_arrangements_l1559_155960


namespace g_is_even_l1559_155956

noncomputable def g (x : ℝ) := 2 ^ (x ^ 2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  sorry

end g_is_even_l1559_155956


namespace sequence_convergence_l1559_155900

noncomputable def alpha : ℝ := sorry
def bounded (a : ℕ → ℝ) : Prop := ∃ M > 0, ∀ n, ‖a n‖ ≤ M

-- Translation of the math problem
theorem sequence_convergence (a : ℕ → ℝ) (ha : bounded a) (hα : 0 < alpha ∧ alpha ≤ 1) 
  (ineq : ∀ n ≥ 2, a (n+1) ≤ alpha * a n + (1 - alpha) * a (n-1)) : 
  ∃ l, ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖a n - l‖ < ε := 
sorry

end sequence_convergence_l1559_155900


namespace counterexample_to_proposition_l1559_155971

theorem counterexample_to_proposition (x y : ℤ) (h1 : x = -1) (h2 : y = -2) : x > y ∧ ¬ (x^2 > y^2) := by
  sorry

end counterexample_to_proposition_l1559_155971


namespace remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l1559_155911

theorem remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14 
  (a b c d e f g h : ℤ) 
  (h1 : a = 11085)
  (h2 : b = 11087)
  (h3 : c = 11089)
  (h4 : d = 11091)
  (h5 : e = 11093)
  (h6 : f = 11095)
  (h7 : g = 11097)
  (h8 : h = 11099) :
  (2 * (a + b + c + d + e + f + g + h)) % 14 = 2 := 
by
  sorry

end remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l1559_155911


namespace numbers_not_as_difference_of_squares_l1559_155943

theorem numbers_not_as_difference_of_squares :
  {n : ℕ | ¬ ∃ x y : ℕ, x^2 - y^2 = n} = {1, 4} ∪ {4*k + 2 | k : ℕ} :=
by sorry

end numbers_not_as_difference_of_squares_l1559_155943


namespace average_age_before_new_students_joined_l1559_155916

theorem average_age_before_new_students_joined 
  (A : ℝ) 
  (N : ℕ) 
  (new_students_average_age : ℝ) 
  (average_age_drop : ℝ) 
  (original_class_strength : ℕ)
  (hN : N = 17) 
  (h_new_students : new_students_average_age = 32)
  (h_age_drop : average_age_drop = 4)
  (h_strength : original_class_strength = 17)
  (h_equation : 17 * A + 17 * new_students_average_age = (2 * original_class_strength) * (A - average_age_drop)) :
  A = 40 :=
by sorry

end average_age_before_new_students_joined_l1559_155916


namespace domain_v_l1559_155999

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + x - 1)

theorem domain_v :
  {x : ℝ | x >= 0 ∧ Real.sqrt x + x - 1 ≠ 0} = {x : ℝ | x ∈ Set.Ico 0 (Real.sqrt 5 - 1) ∪ Set.Ioi (Real.sqrt 5 - 1)} :=
by
  sorry

end domain_v_l1559_155999


namespace smallest_portion_is_2_l1559_155991

theorem smallest_portion_is_2 (a d : ℝ) (h1 : 5 * a = 120) (h2 : 3 * a + 3 * d = 7 * (2 * a - 3 * d)) : a - 2 * d = 2 :=
by sorry

end smallest_portion_is_2_l1559_155991


namespace andrew_kept_stickers_l1559_155903

theorem andrew_kept_stickers :
  ∃ (b d f e g h : ℕ), b = 2000 ∧ d = (5 * b) / 100 ∧ f = d + 120 ∧ e = (d + f) / 2 ∧ g = 80 ∧ h = (e + g) / 5 ∧ (b - (d + f + e + g + h) = 1392) :=
sorry

end andrew_kept_stickers_l1559_155903


namespace circle_radius_center_l1559_155989

theorem circle_radius_center (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y - 2 = 0) :
  (∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ a = 1 ∧ b = 1 ∧ r = 2) := 
sorry

end circle_radius_center_l1559_155989


namespace rhombus_diagonal_length_l1559_155941

theorem rhombus_diagonal_length
  (d1 d2 A : ℝ)
  (h1 : d1 = 20)
  (h2 : A = 250)
  (h3 : A = (d1 * d2) / 2) :
  d2 = 25 :=
by
  sorry

end rhombus_diagonal_length_l1559_155941


namespace complex_div_eq_i_l1559_155933

open Complex

theorem complex_div_eq_i : (1 + I) / (1 - I) = I := by
  sorry

end complex_div_eq_i_l1559_155933


namespace cone_base_radius_half_l1559_155978

theorem cone_base_radius_half :
  let R : ℝ := sorry
  let semicircle_radius : ℝ := 1
  let unfolded_circumference : ℝ := π
  let base_circumference : ℝ := 2 * π * R
  base_circumference = unfolded_circumference -> R = 1 / 2 :=
by
  sorry

end cone_base_radius_half_l1559_155978


namespace bike_sharing_problem_l1559_155977

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

theorem bike_sharing_problem:
  let total_bikes := 10
  let blue_bikes := 4
  let yellow_bikes := 6
  let inspected_bikes := 4
  let way_two_blue := combinations blue_bikes 2 * combinations yellow_bikes 2
  let way_three_blue := combinations blue_bikes 3 * combinations yellow_bikes 1
  let way_four_blue := combinations blue_bikes 4
  way_two_blue + way_three_blue + way_four_blue = 115 :=
by
  sorry

end bike_sharing_problem_l1559_155977


namespace tan_fraction_identity_l1559_155906

theorem tan_fraction_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
by 
  sorry

end tan_fraction_identity_l1559_155906


namespace perpendicular_lines_a_value_l1559_155919

theorem perpendicular_lines_a_value :
  (∃ (a : ℝ), ∀ (x y : ℝ), (3 * y + x + 5 = 0) ∧ (4 * y + a * x + 3 = 0) → a = -12) :=
by
  sorry

end perpendicular_lines_a_value_l1559_155919


namespace find_x_range_l1559_155942

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 ≥ 0 → x2 ≥ 0 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

theorem find_x_range (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : decreasing_on_nonnegative f)
  (h3 : f (1/3) = 3/4)
  (h4 : ∀ x : ℝ, 4 * f (Real.logb (1/8) x) > 3) :
  ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ True := sorry

end find_x_range_l1559_155942


namespace emily_collected_total_eggs_l1559_155914

def eggs_in_setA : ℕ := (200 * 36) + (250 * 24)
def eggs_in_setB : ℕ := (375 * 42) - 80
def eggs_in_setC : ℕ := (560 / 2 * 50) + (560 / 2 * 32)

def total_eggs_collected : ℕ := eggs_in_setA + eggs_in_setB + eggs_in_setC

theorem emily_collected_total_eggs : total_eggs_collected = 51830 := by
  -- proof goes here
  sorry

end emily_collected_total_eggs_l1559_155914


namespace parabola_point_distance_condition_l1559_155975

theorem parabola_point_distance_condition (k : ℝ) (p : ℝ) (h_p_gt_0 : p > 0) (focus : ℝ × ℝ) (vertex : ℝ × ℝ) :
  vertex = (0, 0) → focus = (0, p/2) → (k^2 = -2 * p * (-2)) → dist (k, -2) focus = 4 → k = 4 ∨ k = -4 :=
by
  sorry

end parabola_point_distance_condition_l1559_155975


namespace sampling_method_selection_l1559_155985

-- Define the sampling methods as data type
inductive SamplingMethod
| SimpleRandomSampling : SamplingMethod
| SystematicSampling : SamplingMethod
| StratifiedSampling : SamplingMethod
| SamplingWithReplacement : SamplingMethod

-- Define our conditions
def basketballs : Nat := 10
def is_random_selection : Bool := true
def no_obvious_stratification : Bool := true

-- The theorem to prove the correct sampling method
theorem sampling_method_selection 
  (b : Nat) 
  (random_selection : Bool) 
  (no_stratification : Bool) : 
  SamplingMethod :=
  if b = 10 ∧ random_selection ∧ no_stratification then SamplingMethod.SimpleRandomSampling 
  else sorry

-- Prove the correct sampling method given our conditions
example : sampling_method_selection basketballs is_random_selection no_obvious_stratification = SamplingMethod.SimpleRandomSampling := 
by
-- skipping the proof here with sorry
sorry

end sampling_method_selection_l1559_155985


namespace magazines_cover_area_l1559_155917

theorem magazines_cover_area (S : ℝ) (n : ℕ) (h_n_15 : n = 15) (h_cover : ∀ m ≤ n, ∃(Sm:ℝ), (Sm ≥ (m : ℝ) / n * S) ) :
  ∃ k : ℕ, k = n - 7 ∧ ∃ (Sk : ℝ), (Sk ≥ 8/15 * S) := 
by
  sorry

end magazines_cover_area_l1559_155917


namespace negation_proof_l1559_155927

open Real

theorem negation_proof :
  (¬ ∃ x : ℕ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, exp x - x - 1 > 0) :=
by
  sorry

end negation_proof_l1559_155927


namespace sum_of_fractions_l1559_155926

theorem sum_of_fractions : 
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (10 / 10) + (11 / 10) + (15 / 10) + (20 / 10) + (25 / 10) + (50 / 10) = 14.1 :=
by sorry

end sum_of_fractions_l1559_155926


namespace simplify_and_evaluate_expression_l1559_155936

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2023) :
  (x + y)^2 + (x + y) * (x - y) - 2 * x^2 = 2023 :=
by
  sorry

end simplify_and_evaluate_expression_l1559_155936


namespace speed_of_man_in_still_water_l1559_155959

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 6.2) (h2 : v_m - v_s = 6) : v_m = 6.1 :=
by
  sorry

end speed_of_man_in_still_water_l1559_155959


namespace four_digit_numbers_no_5s_8s_l1559_155954

def count_valid_four_digit_numbers : Nat :=
  let thousand_place := 7  -- choices: 1, 2, 3, 4, 6, 7, 9
  let other_places := 8  -- choices: 0, 1, 2, 3, 4, 6, 7, 9
  thousand_place * other_places * other_places * other_places

theorem four_digit_numbers_no_5s_8s : count_valid_four_digit_numbers = 3584 :=
by
  rfl

end four_digit_numbers_no_5s_8s_l1559_155954


namespace tire_circumference_l1559_155976

/-- If a tire rotates at 400 revolutions per minute and the car is traveling at 48 km/h, 
    prove that the circumference of the tire in meters is 2. -/
theorem tire_circumference (speed_kmh : ℕ) (revolutions_per_min : ℕ)
  (h1 : speed_kmh = 48) (h2 : revolutions_per_min = 400) : 
  (circumference : ℕ) = 2 := 
sorry

end tire_circumference_l1559_155976


namespace monkey_climbing_distance_l1559_155934

theorem monkey_climbing_distance
  (x : ℝ)
  (h1 : ∀ t : ℕ, t % 2 = 0 → t ≠ 0 → x - 3 > 0) -- condition (2,4)
  (h2 : ∀ t : ℕ, t % 2 = 1 → x > 0) -- condition (5)
  (h3 : 18 * (x - 3) + x = 60) -- condition (6)
  : x = 6 :=
sorry

end monkey_climbing_distance_l1559_155934


namespace second_number_value_l1559_155947

def first_number := ℚ
def second_number := ℚ

variables (x y : ℚ)

/-- Given conditions: 
      (1) \( \frac{1}{5}x = \frac{5}{8}y \)
      (2) \( x + 35 = 4y \)
    Prove that \( y = 40 \) 
-/
theorem second_number_value (h1 : (1/5 : ℚ) * x = (5/8 : ℚ) * y) (h2 : x + 35 = 4 * y) : 
  y = 40 :=
sorry

end second_number_value_l1559_155947


namespace find_percentage_l1559_155955

variable (x p : ℝ)
variable (h1 : 0.25 * x = (p / 100) * 1500 - 20)
variable (h2 : x = 820)

theorem find_percentage : p = 15 :=
by
  sorry

end find_percentage_l1559_155955


namespace solve_for_y_l1559_155980

theorem solve_for_y (y : ℚ) (h : |5 * y - 6| = 0) : y = 6 / 5 :=
by 
  sorry

end solve_for_y_l1559_155980


namespace waiting_time_boarding_l1559_155958

noncomputable def time_taken_uber_to_house : ℕ := 10
noncomputable def time_taken_uber_to_airport : ℕ := 5 * time_taken_uber_to_house
noncomputable def time_taken_bag_check : ℕ := 15
noncomputable def time_taken_security : ℕ := 3 * time_taken_bag_check
noncomputable def total_process_time : ℕ := 180
noncomputable def remaining_time : ℕ := total_process_time - (time_taken_uber_to_house + time_taken_uber_to_airport + time_taken_bag_check + time_taken_security)
noncomputable def time_before_takeoff (B : ℕ) := 2 * B

theorem waiting_time_boarding : ∃ B : ℕ, B + time_before_takeoff B = remaining_time ∧ B = 20 := 
by 
  sorry

end waiting_time_boarding_l1559_155958


namespace smallest_b_value_l1559_155931

variable {a b c d : ℝ}

-- Definitions based on conditions
def is_arithmetic_series (a b c : ℝ) (d : ℝ) : Prop :=
  a = b - d ∧ c = b + d

def abc_product (a b c : ℝ) : Prop :=
  a * b * c = 216

theorem smallest_b_value (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (arith_series : is_arithmetic_series a b c d)
  (abc_216 : abc_product a b c) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_value_l1559_155931


namespace Kevin_ends_with_54_cards_l1559_155984

/-- Kevin starts with 7 cards and finds another 47 cards. 
    This theorem proves that Kevin ends with 54 cards. -/
theorem Kevin_ends_with_54_cards :
  let initial_cards := 7
  let found_cards := 47
  initial_cards + found_cards = 54 := 
by
  let initial_cards := 7
  let found_cards := 47
  sorry

end Kevin_ends_with_54_cards_l1559_155984


namespace problem_statement_l1559_155915

variable {x : Real}
variable {m : Int}
variable {n : Int}

theorem problem_statement (h1 : x^m = 5) (h2 : x^n = 10) : x^(2 * m - n) = 5 / 2 :=
by
  sorry

end problem_statement_l1559_155915


namespace AM_GM_inequality_l1559_155950

theorem AM_GM_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_not_all_eq : x ≠ y ∨ y ≠ z ∨ z ≠ x) :
  (x + y) * (y + z) * (z + x) > 8 * x * y * z :=
by
  sorry

end AM_GM_inequality_l1559_155950


namespace find_math_marks_l1559_155962

theorem find_math_marks
  (e p c b : ℕ)
  (n : ℕ)
  (a : ℚ)
  (M : ℕ) :
  e = 96 →
  p = 82 →
  c = 87 →
  b = 92 →
  n = 5 →
  a = 90.4 →
  (a * n = (e + p + c + b + M)) →
  M = 95 :=
by intros
   sorry

end find_math_marks_l1559_155962


namespace ratio_proof_l1559_155901

variables {d l e : ℕ} -- Define variables representing the number of doctors, lawyers, and engineers
variables (hd : ℕ → ℕ) (hl : ℕ → ℕ) (he : ℕ → ℕ) (ho : ℕ → ℕ)

-- Condition: Average ages
def avg_age_doctors := 40 * d
def avg_age_lawyers := 55 * l
def avg_age_engineers := 35 * e

-- Condition: Overall average age is 45 years
def overall_avg_age := (40 * d + 55 * l + 35 * e) / (d + l + e)

theorem ratio_proof (h1 : 40 * d + 55 * l + 35 * e = 45 * (d + l + e)) : 
  d = l ∧ e = 2 * l :=
by
  sorry

end ratio_proof_l1559_155901


namespace define_interval_l1559_155948

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end define_interval_l1559_155948


namespace train_length_l1559_155966

variable (L V : ℝ)

-- Given conditions
def condition1 : Prop := V = L / 24
def condition2 : Prop := V = (L + 650) / 89

theorem train_length : condition1 L V → condition2 L V → L = 240 := by
  intro h1 h2
  sorry

end train_length_l1559_155966


namespace find_cd_l1559_155953

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) 
  (h1 : g c d 2 = -7) 
  (h2 : g c d (-1) = -25) : 
  (c, d) = (2, 8) := 
by
  sorry

end find_cd_l1559_155953


namespace square_area_is_4802_l1559_155929

-- Condition: the length of the diagonal of the square is 98 meters.
def diagonal (d : ℝ) := d = 98

-- Goal: Prove that the area of the square field is 4802 square meters.
theorem square_area_is_4802 (d : ℝ) (h : diagonal d) : ∃ (A : ℝ), A = 4802 := 
by sorry

end square_area_is_4802_l1559_155929


namespace count_two_digit_integers_congruent_to_2_mod_4_l1559_155939

theorem count_two_digit_integers_congruent_to_2_mod_4 : 
  ∃ n : ℕ, (∀ x : ℕ, 10 ≤ x ∧ x ≤ 99 → x % 4 = 2 → x = 4 * k + 2) ∧ n = 23 := 
by
  sorry

end count_two_digit_integers_congruent_to_2_mod_4_l1559_155939


namespace nancy_rose_bracelets_l1559_155995

-- Definitions based on conditions
def metal_beads_nancy : ℕ := 40
def pearl_beads_nancy : ℕ := metal_beads_nancy + 20
def total_beads_nancy : ℕ := metal_beads_nancy + pearl_beads_nancy

def crystal_beads_rose : ℕ := 20
def stone_beads_rose : ℕ := 2 * crystal_beads_rose
def total_beads_rose : ℕ := crystal_beads_rose + stone_beads_rose

def number_of_bracelets (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

-- Theorem to be proved
theorem nancy_rose_bracelets : number_of_bracelets (total_beads_nancy + total_beads_rose) 8 = 20 := 
by
  -- Definitions will be expanded here
  sorry

end nancy_rose_bracelets_l1559_155995


namespace find_x_if_perpendicular_l1559_155932

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x - 5)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x : ℝ) : Prop :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2 = 0

-- Prove that x = 3 if a and b are perpendicular
theorem find_x_if_perpendicular :
  ∃ x : ℝ, perpendicular x ∧ x = 3 :=
by
  sorry

end find_x_if_perpendicular_l1559_155932


namespace park_area_l1559_155908

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end park_area_l1559_155908


namespace probability_of_shaded_triangle_l1559_155912

theorem probability_of_shaded_triangle 
  (triangles : Finset ℝ) 
  (shaded_triangles : Finset ℝ)
  (h1 : triangles = {1, 2, 3, 4, 5})
  (h2 : shaded_triangles = {1, 4})
  : (shaded_triangles.card / triangles.card) = 2 / 5 := 
  by
  sorry

end probability_of_shaded_triangle_l1559_155912


namespace truncated_pyramid_volume_l1559_155997

theorem truncated_pyramid_volume :
  let unit_cube_vol := 1
  let tetrahedron_base_area := 1 / 2
  let tetrahedron_height := 1 / 2
  let tetrahedron_vol := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let two_tetrahedra_vol := 2 * tetrahedron_vol
  let truncated_pyramid_vol := unit_cube_vol - two_tetrahedra_vol
  truncated_pyramid_vol = 5 / 6 :=
by
  sorry

end truncated_pyramid_volume_l1559_155997


namespace batteries_on_flashlights_l1559_155973

variable (b_flashlights b_toys b_controllers b_total : ℕ)

theorem batteries_on_flashlights :
  b_toys = 15 → 
  b_controllers = 2 → 
  b_total = 19 → 
  b_total = b_flashlights + b_toys + b_controllers → 
  b_flashlights = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end batteries_on_flashlights_l1559_155973


namespace total_savings_percentage_l1559_155952

theorem total_savings_percentage
  (original_coat_price : ℕ) (original_pants_price : ℕ)
  (coat_discount_percent : ℚ) (pants_discount_percent : ℚ)
  (original_total_price : ℕ) (total_savings : ℕ)
  (savings_percentage : ℚ) :
  original_coat_price = 120 →
  original_pants_price = 60 →
  coat_discount_percent = 0.30 →
  pants_discount_percent = 0.60 →
  original_total_price = original_coat_price + original_pants_price →
  total_savings = original_coat_price * coat_discount_percent + original_pants_price * pants_discount_percent →
  savings_percentage = (total_savings / original_total_price) * 100 →
  savings_percentage = 40 := 
by
  intros
  sorry

end total_savings_percentage_l1559_155952
