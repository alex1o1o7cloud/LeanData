import Mathlib

namespace NUMINAMATH_GPT_integer_root_sum_abs_l96_9692

theorem integer_root_sum_abs :
  ∃ a b c m : ℤ, 
    (a + b + c = 0 ∧ ab + bc + ca = -2023 ∧ |a| + |b| + |c| = 94) := sorry

end NUMINAMATH_GPT_integer_root_sum_abs_l96_9692


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l96_9697

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (R : ℝ)
  (h1 : b = 6) (h2 : c = 2) (h3 : A = π / 3) :
  R = (2 * Real.sqrt 21) / 3 :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l96_9697


namespace NUMINAMATH_GPT_intersection_of_S_and_T_l96_9630

open Set

def setS : Set ℝ := { x | (x-2)*(x+3) > 0 }
def setT : Set ℝ := { x | 3 - x ≥ 0 }

theorem intersection_of_S_and_T : setS ∩ setT = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_S_and_T_l96_9630


namespace NUMINAMATH_GPT_cos_diff_trigonometric_identity_l96_9669

-- Problem 1
theorem cos_diff :
  (Real.cos (25 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) - 
   Real.cos (65 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  1/2 :=
sorry

-- Problem 2
theorem trigonometric_identity (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (Real.cos (2 * θ) - Real.sin (2 * θ)) / (1 + (Real.cos θ)^2) = 5/6 :=
sorry

end NUMINAMATH_GPT_cos_diff_trigonometric_identity_l96_9669


namespace NUMINAMATH_GPT_highway_speed_l96_9608

theorem highway_speed 
  (local_distance : ℝ) (local_speed : ℝ)
  (highway_distance : ℝ) (avg_speed : ℝ)
  (h_local : local_distance = 90) 
  (h_local_speed : local_speed = 30)
  (h_highway : highway_distance = 75)
  (h_avg : avg_speed = 38.82) :
  ∃ v : ℝ, v = 60 := 
sorry

end NUMINAMATH_GPT_highway_speed_l96_9608


namespace NUMINAMATH_GPT_no_positive_int_solutions_l96_9602

theorem no_positive_int_solutions
  (x y z t : ℕ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (ht : 0 < t)
  (h1 : x^2 + 2 * y^2 = z^2)
  (h2 : 2 * x^2 + y^2 = t^2) : false :=
by
  sorry

end NUMINAMATH_GPT_no_positive_int_solutions_l96_9602


namespace NUMINAMATH_GPT_father_seven_times_as_old_l96_9631

theorem father_seven_times_as_old (x : ℕ) (father_age : ℕ) (son_age : ℕ) :
  father_age = 38 → son_age = 14 → (father_age - x = 7 * (son_age - x) → x = 10) :=
by
  intros h_father_age h_son_age h_equation
  rw [h_father_age, h_son_age] at h_equation
  sorry

end NUMINAMATH_GPT_father_seven_times_as_old_l96_9631


namespace NUMINAMATH_GPT_number_of_valid_b_l96_9656

theorem number_of_valid_b : ∃ (bs : Finset ℂ), bs.card = 2 ∧ ∀ b ∈ bs, ∃ (x : ℂ), (x + b = b^2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_b_l96_9656


namespace NUMINAMATH_GPT_intersection_of_asymptotes_l96_9699

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem intersection_of_asymptotes : f 3 = 1 :=
by sorry

end NUMINAMATH_GPT_intersection_of_asymptotes_l96_9699


namespace NUMINAMATH_GPT_individual_weights_l96_9609

theorem individual_weights (A P : ℕ) 
    (h1 : 12 * A + 14 * P = 692)
    (h2 : P = A - 10) : 
    A = 32 ∧ P = 22 :=
by
  sorry

end NUMINAMATH_GPT_individual_weights_l96_9609


namespace NUMINAMATH_GPT_union_of_M_and_N_is_correct_l96_9655

def M : Set ℤ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 ≤ n ∧ n ≤ 3 }

theorem union_of_M_and_N_is_correct : M ∪ N = { -2, -1, 0, 1, 2, 3 } := 
by
  sorry

end NUMINAMATH_GPT_union_of_M_and_N_is_correct_l96_9655


namespace NUMINAMATH_GPT_heptagonal_prism_faces_and_vertices_l96_9651

structure HeptagonalPrism where
  heptagonal_basis : ℕ
  lateral_faces : ℕ
  basis_vertices : ℕ

noncomputable def faces (h : HeptagonalPrism) : ℕ :=
  2 + h.lateral_faces

noncomputable def vertices (h : HeptagonalPrism) : ℕ :=
  h.basis_vertices * 2

theorem heptagonal_prism_faces_and_vertices : ∀ h : HeptagonalPrism,
  (h.heptagonal_basis = 2) →
  (h.lateral_faces = 7) →
  (h.basis_vertices = 7) →
  faces h = 9 ∧ vertices h = 14 :=
by
  intros
  simp [faces, vertices]
  sorry

end NUMINAMATH_GPT_heptagonal_prism_faces_and_vertices_l96_9651


namespace NUMINAMATH_GPT_compare_powers_l96_9632

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_powers_l96_9632


namespace NUMINAMATH_GPT_greatest_integer_property_l96_9635

theorem greatest_integer_property :
  ∃ n : ℤ, n < 1000 ∧ (∃ m : ℤ, 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧ 
  (∀ k : ℤ, k < 1000 ∧ (∃ m : ℤ, 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) → k ≤ n) := by
  -- skipped the proof with sorry
  sorry

end NUMINAMATH_GPT_greatest_integer_property_l96_9635


namespace NUMINAMATH_GPT_coeff_z_in_third_eq_l96_9612

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_coeff_z_in_third_eq_l96_9612


namespace NUMINAMATH_GPT_people_in_room_proof_l96_9673

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end NUMINAMATH_GPT_people_in_room_proof_l96_9673


namespace NUMINAMATH_GPT_probability_at_least_one_white_ball_l96_9652

noncomputable def total_combinations : ℕ := (Nat.choose 5 3)
noncomputable def no_white_combinations : ℕ := (Nat.choose 3 3)
noncomputable def prob_no_white_balls : ℚ := no_white_combinations / total_combinations
noncomputable def prob_at_least_one_white_ball : ℚ := 1 - prob_no_white_balls

theorem probability_at_least_one_white_ball :
  prob_at_least_one_white_ball = 9 / 10 :=
by
  have h : total_combinations = 10 := by sorry
  have h1 : no_white_combinations = 1 := by sorry
  have h2 : prob_no_white_balls = 1 / 10 := by sorry
  have h3 : prob_at_least_one_white_ball = 1 - prob_no_white_balls := by sorry
  norm_num [prob_no_white_balls, prob_at_least_one_white_ball, h, h1, h2, h3]

end NUMINAMATH_GPT_probability_at_least_one_white_ball_l96_9652


namespace NUMINAMATH_GPT_geometric_series_sum_eq_l96_9698

-- Given conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 5

-- Define the geometric series sum formula
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The main theorem to prove
theorem geometric_series_sum_eq : geometric_series_sum a r n = 31 / 32 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_eq_l96_9698


namespace NUMINAMATH_GPT_geometric_sequence_a1_cannot_be_2_l96_9607

theorem geometric_sequence_a1_cannot_be_2
  (a : ℕ → ℕ)
  (q : ℕ)
  (h1 : 2 * a 2 + a 3 = a 4)
  (h2 : (a 2 + 1) * (a 3 + 1) = a 5 - 1)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 1 ≠ 2 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_a1_cannot_be_2_l96_9607


namespace NUMINAMATH_GPT_vector_c_solution_l96_9645

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_c_solution
  (a b c : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (2, -3))
  (h3 : vector_parallel (c.1 + 1, c.2 + 2) b)
  (h4 : vector_perpendicular c (3, -1)) :
  c = (-7/9, -7/3) :=
sorry

end NUMINAMATH_GPT_vector_c_solution_l96_9645


namespace NUMINAMATH_GPT_zero_of_function_is_not_intersection_l96_9670

noncomputable def is_function_zero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

theorem zero_of_function_is_not_intersection (f : ℝ → ℝ) :
  ¬ (∀ x : ℝ, is_function_zero f x ↔ (f x = 0 ∧ x ∈ {x | f x = 0})) :=
by
  sorry

end NUMINAMATH_GPT_zero_of_function_is_not_intersection_l96_9670


namespace NUMINAMATH_GPT_inv_sum_mod_l96_9653

theorem inv_sum_mod (x y : ℤ) (h1 : 5 * x ≡ 1 [ZMOD 23]) (h2 : 25 * y ≡ 1 [ZMOD 23]) : (x + y) ≡ 3 [ZMOD 23] := by
  sorry

end NUMINAMATH_GPT_inv_sum_mod_l96_9653


namespace NUMINAMATH_GPT_reduction_in_consumption_l96_9683

def rate_last_month : ℝ := 16
def rate_current : ℝ := 20
def initial_consumption (X : ℝ) : ℝ := X

theorem reduction_in_consumption (X : ℝ) : initial_consumption X - (initial_consumption X * rate_last_month / rate_current) = initial_consumption X * 0.2 :=
by
  sorry

end NUMINAMATH_GPT_reduction_in_consumption_l96_9683


namespace NUMINAMATH_GPT_rate_calculation_l96_9618

def principal : ℝ := 910
def simple_interest : ℝ := 260
def time : ℝ := 4
def rate : ℝ := 7.14

theorem rate_calculation :
  (simple_interest / (principal * time)) * 100 = rate :=
by
  sorry

end NUMINAMATH_GPT_rate_calculation_l96_9618


namespace NUMINAMATH_GPT_total_area_of_combined_figure_l96_9647

noncomputable def combined_area (A_triangle : ℕ) (b : ℕ) : ℕ :=
  let h := (2 * A_triangle) / b
  let A_square := b * b
  A_square + A_triangle

theorem total_area_of_combined_figure :
  combined_area 720 40 = 2320 := by
  sorry

end NUMINAMATH_GPT_total_area_of_combined_figure_l96_9647


namespace NUMINAMATH_GPT_number_of_solutions_sine_exponential_l96_9623

theorem number_of_solutions_sine_exponential :
  let f := λ x => Real.sin x
  let g := λ x => (1 / 3) ^ x
  ∃ n, n = 150 ∧ ∀ k ∈ Set.Icc (0 : ℝ) (150 * Real.pi), f k = g k → (k : ℝ) ∈ {n : ℝ | n ∈ Set.Icc (0 : ℝ) (150 * Real.pi)} :=
sorry

end NUMINAMATH_GPT_number_of_solutions_sine_exponential_l96_9623


namespace NUMINAMATH_GPT_binomial_mod_prime_eq_floor_l96_9659

-- Define the problem's conditions and goal in Lean.
theorem binomial_mod_prime_eq_floor (n p : ℕ) (hp : Nat.Prime p) : (Nat.choose n p) % p = n / p := by
  sorry

end NUMINAMATH_GPT_binomial_mod_prime_eq_floor_l96_9659


namespace NUMINAMATH_GPT_shortest_total_distance_piglet_by_noon_l96_9636

-- Define the distances
def distance_fs : ℕ := 1300  -- Distance through the forest (Piglet to Winnie-the-Pooh)
def distance_pr : ℕ := 600   -- Distance (Piglet to Rabbit)
def distance_rw : ℕ := 500   -- Distance (Rabbit to Winnie-the-Pooh)

-- Define the total distance via Rabbit and via forest
def total_distance_rabbit_path : ℕ := distance_pr + distance_rw + distance_rw
def total_distance_forest_path : ℕ := distance_fs + distance_rw

-- Prove that shortest distance Piglet covers by noon
theorem shortest_total_distance_piglet_by_noon : 
  min (total_distance_forest_path) (total_distance_rabbit_path) = 1600 := by
  sorry

end NUMINAMATH_GPT_shortest_total_distance_piglet_by_noon_l96_9636


namespace NUMINAMATH_GPT_speed_against_current_l96_9689

theorem speed_against_current (V_curr : ℝ) (V_man : ℝ) (V_curr_val : V_curr = 3.2) (V_man_with_curr : V_man = 15) :
    V_man - V_curr = 8.6 := 
by 
  rw [V_curr_val, V_man_with_curr]
  norm_num
  sorry

end NUMINAMATH_GPT_speed_against_current_l96_9689


namespace NUMINAMATH_GPT_inequality_proof_l96_9613

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l96_9613


namespace NUMINAMATH_GPT_laptop_repair_cost_l96_9694

theorem laptop_repair_cost
  (price_phone_repair : ℝ)
  (price_computer_repair : ℝ)
  (price_laptop_repair : ℝ)
  (condition1 : price_phone_repair = 11)
  (condition2 : price_computer_repair = 18)
  (condition3 : 5 * price_phone_repair + 2 * price_laptop_repair + 2 * price_computer_repair = 121) :
  price_laptop_repair = 15 :=
by
  sorry

end NUMINAMATH_GPT_laptop_repair_cost_l96_9694


namespace NUMINAMATH_GPT_vincent_books_l96_9695

theorem vincent_books (x : ℕ) (h1 : 10 + 3 + x = 13 + x)
                      (h2 : 16 * (13 + x) = 224) : x = 1 :=
by sorry

end NUMINAMATH_GPT_vincent_books_l96_9695


namespace NUMINAMATH_GPT_wrapping_paper_per_present_l96_9674

theorem wrapping_paper_per_present :
  let sum_paper := 1 / 2
  let num_presents := 5
  (sum_paper / num_presents) = 1 / 10 := by
  sorry

end NUMINAMATH_GPT_wrapping_paper_per_present_l96_9674


namespace NUMINAMATH_GPT_sams_weight_l96_9678

  theorem sams_weight (j s : ℝ) (h1 : j + s = 240) (h2 : s - j = j / 3) : s = 2880 / 21 :=
  by
    sorry
  
end NUMINAMATH_GPT_sams_weight_l96_9678


namespace NUMINAMATH_GPT_total_paintings_is_correct_l96_9682

-- Definitions for Philip's schedule and starting number of paintings
def philip_paintings_monday_and_tuesday := 3
def philip_paintings_wednesday := 2
def philip_paintings_thursday_and_friday := 5
def philip_initial_paintings := 20

-- Definitions for Amelia's schedule and starting number of paintings
def amelia_paintings_every_day := 2
def amelia_initial_paintings := 45

-- Calculation of total paintings after 5 weeks
def philip_weekly_paintings := 
  (2 * philip_paintings_monday_and_tuesday) + 
  philip_paintings_wednesday + 
  (2 * philip_paintings_thursday_and_friday)

def amelia_weekly_paintings := 
  7 * amelia_paintings_every_day

def total_paintings_after_5_weeks := 5 * philip_weekly_paintings + philip_initial_paintings + 5 * amelia_weekly_paintings + amelia_initial_paintings

-- Proof statement
theorem total_paintings_is_correct :
  total_paintings_after_5_weeks = 225 :=
  by sorry

end NUMINAMATH_GPT_total_paintings_is_correct_l96_9682


namespace NUMINAMATH_GPT_longest_side_obtuse_triangle_l96_9634

theorem longest_side_obtuse_triangle (a b c : ℝ) (h₀ : a = 2) (h₁ : b = 4) 
  (h₂ : a^2 + b^2 < c^2) : 
  2 * Real.sqrt 5 < c ∧ c < 6 :=
by 
  sorry

end NUMINAMATH_GPT_longest_side_obtuse_triangle_l96_9634


namespace NUMINAMATH_GPT_max_value_f_compare_magnitude_l96_9685

open Real

def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- 1. Prove that the maximum value of f(x) is 2.
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
sorry

-- 2. Given the condition, prove 2m + n > 2.
theorem compare_magnitude (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (1 / (2 * n)) = 2) : 
  2 * m + n > 2 :=
sorry

end NUMINAMATH_GPT_max_value_f_compare_magnitude_l96_9685


namespace NUMINAMATH_GPT_triangle_is_obtuse_l96_9654

noncomputable def is_exterior_smaller (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle < interior_angle

noncomputable def sum_of_angles (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle + interior_angle = 180

theorem triangle_is_obtuse (exterior_angle interior_angle : ℝ) (h1 : is_exterior_smaller exterior_angle interior_angle) 
  (h2 : sum_of_angles exterior_angle interior_angle) : ∃ b, 90 < b ∧ b = interior_angle :=
sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l96_9654


namespace NUMINAMATH_GPT_ethel_subtracts_l96_9626

theorem ethel_subtracts (h : 50^2 = 2500) : 2500 - 99 = 49^2 :=
by
  sorry

end NUMINAMATH_GPT_ethel_subtracts_l96_9626


namespace NUMINAMATH_GPT_katie_baked_5_cookies_l96_9638

theorem katie_baked_5_cookies (cupcakes cookies sold left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : sold = 4) 
  (h3 : left = 8) 
  (h4 : cupcakes + cookies = sold + left) : 
  cookies = 5 :=
by sorry

end NUMINAMATH_GPT_katie_baked_5_cookies_l96_9638


namespace NUMINAMATH_GPT_digit_for_divisibility_by_5_l96_9600

theorem digit_for_divisibility_by_5 (B : ℕ) (B_digit_condition : B < 10) :
  (∃ k : ℕ, 6470 + B = 5 * k) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_digit_for_divisibility_by_5_l96_9600


namespace NUMINAMATH_GPT_range_of_a_l96_9641

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ≤ y → f a x ≤ f a y) ∨ (∀ x y, x ≤ y → f a x ≥ f a y) → 
  a ∈ Set.Ico (-2 : ℝ) 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l96_9641


namespace NUMINAMATH_GPT_parallelogram_area_l96_9649

open Real

def line1 (p : ℝ × ℝ) : Prop := p.2 = 2
def line2 (p : ℝ × ℝ) : Prop := p.2 = -2
def line3 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 - 10 = 0
def line4 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 + 20 = 0

theorem parallelogram_area :
  ∃ D : ℝ, D = 30 ∧
  (∀ p : ℝ × ℝ, line1 p ∨ line2 p ∨ line3 p ∨ line4 p) :=
sorry

end NUMINAMATH_GPT_parallelogram_area_l96_9649


namespace NUMINAMATH_GPT_triangular_region_area_l96_9644

theorem triangular_region_area : 
  ∀ (x y : ℝ),  (3 * x + 4 * y = 12) →
  (0 ≤ x ∧ 0 ≤ y) →
  ∃ (A : ℝ), A = 6 := 
by 
  sorry

end NUMINAMATH_GPT_triangular_region_area_l96_9644


namespace NUMINAMATH_GPT_parrot_initial_phrases_l96_9648

theorem parrot_initial_phrases (current_phrases : ℕ) (days_with_parrot : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) :
  current_phrases = 17 →
  days_with_parrot = 49 →
  phrases_per_week = 2 →
  initial_phrases = current_phrases - phrases_per_week * (days_with_parrot / 7) :=
by
  sorry

end NUMINAMATH_GPT_parrot_initial_phrases_l96_9648


namespace NUMINAMATH_GPT_shift_parabola_upwards_l96_9657

theorem shift_parabola_upwards (y x : ℝ) (h : y = x^2) : y + 5 = (x^2 + 5) := by 
  sorry

end NUMINAMATH_GPT_shift_parabola_upwards_l96_9657


namespace NUMINAMATH_GPT_trapezoid_area_difference_l96_9664

def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  0.5 * (base1 + base2) * height

def combined_area (base1 base2 height : ℝ) : ℝ :=
  2 * trapezoid_area base1 base2 height

theorem trapezoid_area_difference :
  let combined_area1 := combined_area 11 19 10
  let combined_area2 := combined_area 9.5 11 8
  combined_area1 - combined_area2 = 136 :=
by
  let combined_area1 := combined_area 11 19 10 
  let combined_area2 := combined_area 9.5 11 8 
  show combined_area1 - combined_area2 = 136
  sorry

end NUMINAMATH_GPT_trapezoid_area_difference_l96_9664


namespace NUMINAMATH_GPT_relationship_y1_y2_l96_9661

theorem relationship_y1_y2 (k b y1 y2 : ℝ) (h₀ : k < 0) (h₁ : y1 = k * (-1) + b) (h₂ : y2 = k * 1 + b) : y1 > y2 := 
by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_l96_9661


namespace NUMINAMATH_GPT_min_am_hm_l96_9672

theorem min_am_hm (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * (1/a + 1/b) ≥ 4 :=
by sorry

end NUMINAMATH_GPT_min_am_hm_l96_9672


namespace NUMINAMATH_GPT_b_41_mod_49_l96_9667

noncomputable def b (n : ℕ) : ℕ :=
  6 ^ n + 8 ^ n

theorem b_41_mod_49 : b 41 % 49 = 35 := by
  sorry

end NUMINAMATH_GPT_b_41_mod_49_l96_9667


namespace NUMINAMATH_GPT_simple_interest_rate_l96_9690

theorem simple_interest_rate
  (A5 A8 : ℝ) (years_between : ℝ := 3) (I3 : ℝ) (annual_interest : ℝ)
  (P : ℝ) (R : ℝ)
  (h1 : A5 = 9800) -- Amount after 5 years is Rs. 9800
  (h2 : A8 = 12005) -- Amount after 8 years is Rs. 12005
  (h3 : I3 = A8 - A5) -- Interest for 3 years
  (h4 : annual_interest = I3 / years_between) -- Annual interest
  (h5 : P = 9800) -- Principal amount after 5 years
  (h6 : R = (annual_interest * 100) / P) -- Rate of interest formula revised
  : R = 7.5 := 
sorry

end NUMINAMATH_GPT_simple_interest_rate_l96_9690


namespace NUMINAMATH_GPT_radian_to_degree_equivalent_l96_9627

theorem radian_to_degree_equivalent : 
  (7 / 12) * (180 : ℝ) = 105 :=
by
  sorry

end NUMINAMATH_GPT_radian_to_degree_equivalent_l96_9627


namespace NUMINAMATH_GPT_cone_sections_equal_surface_area_l96_9676

theorem cone_sections_equal_surface_area {m r : ℝ} (h_r_pos : r > 0) (h_m_pos : m > 0) :
  ∃ (m1 m2 : ℝ), 
  (m1 = m / Real.sqrt 3) ∧ 
  (m2 = m / 3 * Real.sqrt 6) :=
sorry

end NUMINAMATH_GPT_cone_sections_equal_surface_area_l96_9676


namespace NUMINAMATH_GPT_no_integer_solution_for_system_l96_9643

theorem no_integer_solution_for_system :
  (¬ ∃ x y : ℤ, 18 * x + 27 * y = 21 ∧ 27 * x + 18 * y = 69) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_system_l96_9643


namespace NUMINAMATH_GPT_Matt_buys_10_key_chains_l96_9675

theorem Matt_buys_10_key_chains
  (cost_per_keychain_in_pack_of_10 : ℝ)
  (cost_per_keychain_in_pack_of_4 : ℝ)
  (number_of_keychains : ℝ)
  (savings : ℝ)
  (h1 : cost_per_keychain_in_pack_of_10 = 2)
  (h2 : cost_per_keychain_in_pack_of_4 = 3)
  (h3 : savings = 20)
  (h4 : 3 * number_of_keychains - 2 * number_of_keychains = savings) :
  number_of_keychains = 10 := 
by
  sorry

end NUMINAMATH_GPT_Matt_buys_10_key_chains_l96_9675


namespace NUMINAMATH_GPT_age_sum_l96_9611

theorem age_sum (P Q : ℕ) (h1 : P - 12 = (1 / 2 : ℚ) * (Q - 12)) (h2 : (P : ℚ) / Q = (3 / 4 : ℚ)) : P + Q = 42 :=
sorry

end NUMINAMATH_GPT_age_sum_l96_9611


namespace NUMINAMATH_GPT_both_locks_stall_time_l96_9642

-- Definitions of the conditions
def first_lock_time : ℕ := 5
def second_lock_time : ℕ := 3 * first_lock_time - 3
def both_locks_time : ℕ := 5 * second_lock_time

-- The proof statement
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end NUMINAMATH_GPT_both_locks_stall_time_l96_9642


namespace NUMINAMATH_GPT_fraction_of_height_of_head_l96_9605

theorem fraction_of_height_of_head (h_leg: ℝ) (h_total: ℝ) (h_rest: ℝ) (h_head: ℝ):
  h_leg = 1 / 3 ∧ h_total = 60 ∧ h_rest = 25 ∧ h_head = h_total - (h_leg * h_total + h_rest) 
  → h_head / h_total = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_fraction_of_height_of_head_l96_9605


namespace NUMINAMATH_GPT_find_maximum_k_l96_9693

theorem find_maximum_k {k : ℝ} 
  (h_eq : ∀ x, x^2 + k * x + 8 = 0)
  (h_roots_diff : ∀ x₁ x₂, x₁ - x₂ = 10) :
  k = 2 * Real.sqrt 33 := 
sorry

end NUMINAMATH_GPT_find_maximum_k_l96_9693


namespace NUMINAMATH_GPT_shortest_path_from_vertex_to_center_of_non_adjacent_face_l96_9665

noncomputable def shortest_path_on_cube (edge_length : ℝ) : ℝ :=
  edge_length + (edge_length * Real.sqrt 2 / 2)

theorem shortest_path_from_vertex_to_center_of_non_adjacent_face :
  shortest_path_on_cube 1 = 1 + Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_shortest_path_from_vertex_to_center_of_non_adjacent_face_l96_9665


namespace NUMINAMATH_GPT_neg_p_equiv_l96_9614

theorem neg_p_equiv :
  (¬ (∀ x : ℝ, x > 0 → x - Real.log x > 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0 - Real.log x_0 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_equiv_l96_9614


namespace NUMINAMATH_GPT_problem1_l96_9603

theorem problem1 : 1361 + 972 + 693 + 28 = 3000 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l96_9603


namespace NUMINAMATH_GPT_sum_of_factors_l96_9679

theorem sum_of_factors (W F c : ℕ) (hW_gt_20: W > 20) (hF_gt_20: F > 20) (product_eq : W * F = 770) (sum_eq : W + F = c) :
  c = 57 :=
by sorry

end NUMINAMATH_GPT_sum_of_factors_l96_9679


namespace NUMINAMATH_GPT_find_x_l96_9633

theorem find_x
  (x : ℤ)
  (h1 : 71 * x % 9 = 8) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l96_9633


namespace NUMINAMATH_GPT_identify_7_real_coins_l96_9663

theorem identify_7_real_coins (coins : Fin 63 → ℝ) (fakes : Finset (Fin 63)) (h_fakes_count : fakes.card = 7) (real_weight fake_weight : ℝ)
  (h_weights : ∀ i, i ∉ fakes → coins i = real_weight) (h_fake_weights : ∀ i, i ∈ fakes → coins i = fake_weight) (h_lighter : fake_weight < real_weight) :
  ∃ real_coins : Finset (Fin 63), real_coins.card = 7 ∧ (∀ i, i ∈ real_coins → coins i = real_weight) :=
sorry

end NUMINAMATH_GPT_identify_7_real_coins_l96_9663


namespace NUMINAMATH_GPT_sum_is_square_l96_9646

theorem sum_is_square (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : Nat.gcd a b = 1) (h5 : Nat.gcd b c = 1) (h6 : Nat.gcd c a = 1) 
  (h7 : (1:ℚ)/a + (1:ℚ)/b = (1:ℚ)/c) : ∃ k : ℕ, a + b = k ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_is_square_l96_9646


namespace NUMINAMATH_GPT_x_y_sum_cube_proof_l96_9629

noncomputable def x_y_sum_cube (x y : ℝ) : ℝ := x^3 + y^3

theorem x_y_sum_cube_proof (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h_eq : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x_y_sum_cube x y = 307 :=
sorry

end NUMINAMATH_GPT_x_y_sum_cube_proof_l96_9629


namespace NUMINAMATH_GPT_initial_customers_l96_9680

theorem initial_customers (tables : ℕ) (people_per_table : ℕ) (customers_left : ℕ) (h1 : tables = 5) (h2 : people_per_table = 9) (h3 : customers_left = 17) :
  tables * people_per_table + customers_left = 62 :=
by
  sorry

end NUMINAMATH_GPT_initial_customers_l96_9680


namespace NUMINAMATH_GPT_avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l96_9606

-- Average fuel consumption per kilometer
noncomputable def avgFuelConsumption (initial_fuel: ℝ) (final_fuel: ℝ) (distance: ℝ) : ℝ :=
  (initial_fuel - final_fuel) / distance

-- Relationship between remaining fuel Q and distance x
noncomputable def remainingFuel (initial_fuel: ℝ) (consumption_rate: ℝ) (distance: ℝ) : ℝ :=
  initial_fuel - consumption_rate * distance

-- Check if the car can return home without refueling
noncomputable def canReturnHome (initial_fuel: ℝ) (consumption_rate: ℝ) (round_trip_distance: ℝ) (alarm_fuel_level: ℝ) : Bool :=
  initial_fuel - consumption_rate * round_trip_distance ≥ alarm_fuel_level

-- Theorem statements to prove
theorem avg_fuel_consumption_correct :
  avgFuelConsumption 45 27 180 = 0.1 :=
sorry

theorem remaining_fuel_correct :
  ∀ x, remainingFuel 45 0.1 x = 45 - 0.1 * x :=
sorry

theorem cannot_return_home_without_refueling :
  ¬canReturnHome 45 0.1 (220 * 2) 3 :=
sorry

end NUMINAMATH_GPT_avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l96_9606


namespace NUMINAMATH_GPT_Daniella_savings_l96_9615

def initial_savings_of_Daniella (D : ℤ) := D
def initial_savings_of_Ariella (D : ℤ) := D + 200
def interest_rate : ℚ := 0.10
def time_years : ℚ := 2
def total_amount_after_two_years (initial_amount : ℤ) : ℚ :=
  initial_amount + initial_amount * interest_rate * time_years
def final_amount_of_Ariella : ℚ := 720

theorem Daniella_savings :
  ∃ D : ℤ, total_amount_after_two_years (initial_savings_of_Ariella D) = final_amount_of_Ariella ∧ initial_savings_of_Daniella D = 400 :=
by
  sorry

end NUMINAMATH_GPT_Daniella_savings_l96_9615


namespace NUMINAMATH_GPT_combined_time_to_finish_cereal_l96_9610

theorem combined_time_to_finish_cereal : 
  let rate_fat := 1 / 15
  let rate_thin := 1 / 45
  let combined_rate := rate_fat + rate_thin
  let time_needed := 4 / combined_rate
  time_needed = 45 := 
by 
  sorry

end NUMINAMATH_GPT_combined_time_to_finish_cereal_l96_9610


namespace NUMINAMATH_GPT_calculate_f2_f_l96_9666

variable {f : ℝ → ℝ}

-- Definition of the conditions
def tangent_line_at_x2 (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ → ℝ), (∀ x, L x = -x + 1) ∧ (∀ x, f x = L x + (f x - L 2))

theorem calculate_f2_f'2 (h : tangent_line_at_x2 f) :
  f 2 + deriv f 2 = -2 :=
sorry

end NUMINAMATH_GPT_calculate_f2_f_l96_9666


namespace NUMINAMATH_GPT_estimated_number_of_red_balls_l96_9621

theorem estimated_number_of_red_balls (total_balls : ℕ) (red_draws : ℕ) (total_draws : ℕ)
    (h_total_balls : total_balls = 8) (h_red_draws : red_draws = 75) (h_total_draws : total_draws = 100) :
    total_balls * (red_draws / total_draws : ℚ) = 6 := 
by
  sorry

end NUMINAMATH_GPT_estimated_number_of_red_balls_l96_9621


namespace NUMINAMATH_GPT_simplify_expression_l96_9637

theorem simplify_expression(x : ℝ) : 2 * x * (4 * x^2 - 3 * x + 1) - 7 * (2 * x^2 - 3 * x + 4) = 8 * x^3 - 20 * x^2 + 23 * x - 28 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l96_9637


namespace NUMINAMATH_GPT_polygon_has_five_sides_l96_9691

theorem polygon_has_five_sides (angle : ℝ) (h : angle = 108) :
  (∃ n : ℕ, n > 2 ∧ (180 - angle) * n = 360) ↔ n = 5 := 
by
  sorry

end NUMINAMATH_GPT_polygon_has_five_sides_l96_9691


namespace NUMINAMATH_GPT_quadratic_inequality_has_real_solutions_l96_9681

theorem quadratic_inequality_has_real_solutions (c : ℝ) (h : 0 < c) : 
  (∃ x : ℝ, x^2 - 6 * x + c < 0) ↔ (0 < c ∧ c < 9) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_has_real_solutions_l96_9681


namespace NUMINAMATH_GPT_spent_on_puzzle_l96_9688

-- Defining all given conditions
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def final_amount : ℕ := 1

-- Define the total money before spending on the puzzle
def total_before_puzzle := initial_money + saved_money - spent_on_comic

-- Prove that the amount spent on the puzzle is $18
theorem spent_on_puzzle : (total_before_puzzle - final_amount) = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_spent_on_puzzle_l96_9688


namespace NUMINAMATH_GPT_find_tan_half_angle_l96_9671

variable {α : Real} (h₁ : Real.sin α = -24 / 25) (h₂ : α ∈ Set.Ioo (π:ℝ) (3 * π / 2))

theorem find_tan_half_angle : Real.tan (α / 2) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_find_tan_half_angle_l96_9671


namespace NUMINAMATH_GPT_integer_solution_range_l96_9696

theorem integer_solution_range {m : ℝ} : 
  (∀ x : ℤ, -1 ≤ x → x < m → (x = -1 ∨ x = 0)) ↔ (0 < m ∧ m ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_integer_solution_range_l96_9696


namespace NUMINAMATH_GPT_min_value_x_plus_y_l96_9620

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l96_9620


namespace NUMINAMATH_GPT_train_length_at_constant_acceleration_l96_9625

variables (u : ℝ) (t : ℝ) (a : ℝ) (s : ℝ)

theorem train_length_at_constant_acceleration (h₁ : u = 16.67) (h₂ : t = 30) : 
  s = u * t + 0.5 * a * t^2 :=
sorry

end NUMINAMATH_GPT_train_length_at_constant_acceleration_l96_9625


namespace NUMINAMATH_GPT_perimeter_of_first_square_l96_9640

theorem perimeter_of_first_square (p1 p2 p3 : ℕ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24) :
  p1 = 40 := 
  sorry

end NUMINAMATH_GPT_perimeter_of_first_square_l96_9640


namespace NUMINAMATH_GPT_sequence_integers_l96_9628

theorem sequence_integers (a : ℕ → ℤ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, n ≥ 3 → a n = (a (n-1)) ^ 2 + 2 / a (n-2)) : 
  ∀ n, ∃ k : ℤ, a n = k := 
by 
  sorry

end NUMINAMATH_GPT_sequence_integers_l96_9628


namespace NUMINAMATH_GPT_decimals_between_6_1_and_6_4_are_not_two_l96_9686

-- Definitions from the conditions in a)
def is_between (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

-- The main theorem statement
theorem decimals_between_6_1_and_6_4_are_not_two :
  ∀ x, is_between x 6.1 6.4 → false :=
by
  sorry

end NUMINAMATH_GPT_decimals_between_6_1_and_6_4_are_not_two_l96_9686


namespace NUMINAMATH_GPT_new_volume_of_balloon_l96_9650

def initial_volume : ℝ := 2.00  -- Initial volume in liters
def initial_pressure : ℝ := 745  -- Initial pressure in mmHg
def initial_temperature : ℝ := 293.15  -- Initial temperature in Kelvin
def final_pressure : ℝ := 700  -- Final pressure in mmHg
def final_temperature : ℝ := 283.15  -- Final temperature in Kelvin
def final_volume : ℝ := 2.06  -- Expected final volume in liters

theorem new_volume_of_balloon :
  (initial_pressure * initial_volume / initial_temperature) = (final_pressure * final_volume / final_temperature) :=
  sorry  -- Proof to be filled in later

end NUMINAMATH_GPT_new_volume_of_balloon_l96_9650


namespace NUMINAMATH_GPT_crayons_remaining_l96_9624

def initial_crayons : ℕ := 87
def eaten_crayons : ℕ := 7

theorem crayons_remaining : (initial_crayons - eaten_crayons) = 80 := by
  sorry

end NUMINAMATH_GPT_crayons_remaining_l96_9624


namespace NUMINAMATH_GPT_partI_partII_l96_9617

theorem partI (m : ℝ) (h1 : ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2) :
  1 ≤ m ∧ m ≤ 5 :=
sorry

noncomputable def lambda : ℝ := 5

theorem partII (x y z : ℝ) (h2 : 3 * x + 4 * y + 5 * z = lambda) :
  x^2 + y^2 + z^2 ≥ 1/2 :=
sorry

end NUMINAMATH_GPT_partI_partII_l96_9617


namespace NUMINAMATH_GPT_mean_of_all_students_l96_9668

theorem mean_of_all_students (M A : ℕ) (m a : ℕ) (hM : M = 88) (hA : A = 68) (hRatio : m * 5 = 2 * a) : 
  (176 * a + 340 * a) / (7 * a) = 74 :=
by sorry

end NUMINAMATH_GPT_mean_of_all_students_l96_9668


namespace NUMINAMATH_GPT_prime_addition_fraction_equivalence_l96_9662

theorem prime_addition_fraction_equivalence : 
  ∃ n : ℕ, Prime n ∧ (4 + n) * 8 = (7 + n) * 7 ∧ n = 17 := 
sorry

end NUMINAMATH_GPT_prime_addition_fraction_equivalence_l96_9662


namespace NUMINAMATH_GPT_number_of_lines_passing_through_four_points_l96_9687

-- Defining the three-dimensional points and conditions
structure Point3D where
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : 1 ≤ x ∧ x ≤ 5
  h2 : 1 ≤ y ∧ y ≤ 5
  h3 : 1 ≤ z ∧ z ≤ 5

-- Define a valid line passing through four distinct points (Readonly accessors for the conditions)
def valid_line (p1 p2 p3 p4 : Point3D) : Prop := 
  sorry -- Define conditions for points to be collinear and distinct

-- Main theorem statement
theorem number_of_lines_passing_through_four_points : 
  ∃ (lines : ℕ), lines = 150 :=
sorry

end NUMINAMATH_GPT_number_of_lines_passing_through_four_points_l96_9687


namespace NUMINAMATH_GPT_tax_liability_difference_l96_9677

theorem tax_liability_difference : 
  let annual_income := 150000
  let old_tax_rate := 0.45
  let new_tax_rate_1 := 0.30
  let new_tax_rate_2 := 0.35
  let new_tax_rate_3 := 0.40
  let mortgage_interest := 10000
  let old_tax_liability := annual_income * old_tax_rate
  let taxable_income_new := annual_income - mortgage_interest
  let new_tax_liability := 
    if taxable_income_new <= 50000 then 
      taxable_income_new * new_tax_rate_1
    else if taxable_income_new <= 100000 then 
      50000 * new_tax_rate_1 + (taxable_income_new - 50000) * new_tax_rate_2
    else 
      50000 * new_tax_rate_1 + 50000 * new_tax_rate_2 + (taxable_income_new - 100000) * new_tax_rate_3
  let tax_liability_difference := old_tax_liability - new_tax_liability
  tax_liability_difference = 19000 := 
by
  sorry

end NUMINAMATH_GPT_tax_liability_difference_l96_9677


namespace NUMINAMATH_GPT_symmetric_line_equation_l96_9616

theorem symmetric_line_equation (l : ℝ × ℝ → Prop)
  (h1 : ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0)
  (h2 : ∀ p : ℝ × ℝ, l p ↔ p = (0, 2) ∨ p = ⟨-3, 2⟩) :
  ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l96_9616


namespace NUMINAMATH_GPT_two_digit_number_problem_l96_9601

theorem two_digit_number_problem (a b : ℕ) :
  let M := 10 * b + a
  let N := 10 * a + b
  2 * M - N = 19 * b - 8 * a := by
  sorry

end NUMINAMATH_GPT_two_digit_number_problem_l96_9601


namespace NUMINAMATH_GPT_simplify_power_of_product_l96_9604

theorem simplify_power_of_product (x : ℝ) : (5 * x^2)^4 = 625 * x^8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_power_of_product_l96_9604


namespace NUMINAMATH_GPT_logarithmic_inequality_l96_9619

noncomputable def log_a_b (a b : ℝ) := Real.log b / Real.log a

theorem logarithmic_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  log_a_b a b + log_a_b b c + log_a_b a c ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_logarithmic_inequality_l96_9619


namespace NUMINAMATH_GPT_sequence_a113_l96_9660

theorem sequence_a113 {a : ℕ → ℝ} 
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n, (a (n+1))^2 + (a n)^2 = 2 * n * ((a (n+1))^2 - (a n)^2)) :
  a 113 = 15 :=
sorry

end NUMINAMATH_GPT_sequence_a113_l96_9660


namespace NUMINAMATH_GPT_find_x_l96_9684

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) (h₁ : log x 16 = log 4 256) : x = 2 := by
  sorry

end NUMINAMATH_GPT_find_x_l96_9684


namespace NUMINAMATH_GPT_tan_alpha_fraction_value_l96_9639

theorem tan_alpha_fraction_value {α : Real} (h : Real.tan α = 2) : 
  (3 * Real.sin α + Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_fraction_value_l96_9639


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l96_9658

theorem repeating_decimal_to_fraction : (let a := (0.28282828 : ℚ); a = 28/99) := sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l96_9658


namespace NUMINAMATH_GPT_friend_initial_marbles_l96_9622

theorem friend_initial_marbles (total_games : ℕ) (bids_per_game : ℕ) (games_lost : ℕ) (final_marbles : ℕ) 
  (h_games_eq : total_games = 9) (h_bids_eq : bids_per_game = 10) 
  (h_lost_eq : games_lost = 1) (h_final_eq : final_marbles = 90) : 
  ∃ initial_marbles : ℕ, initial_marbles = 20 := by
  sorry

end NUMINAMATH_GPT_friend_initial_marbles_l96_9622
