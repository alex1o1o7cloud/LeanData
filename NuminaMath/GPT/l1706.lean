import Mathlib

namespace parallel_lines_slope_l1706_170616

theorem parallel_lines_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + 2 * y - 1 = 0 → x = -2 * y + 1)
  (h2 : ∀ x y : ℝ, m * x - y = 0 → y = m * x) : 
  m = -1 / 2 :=
by
  sorry

end parallel_lines_slope_l1706_170616


namespace find_x_l1706_170623

theorem find_x (x : ℝ) (h : 6 * x + 7 * x + 3 * x + 2 * x + 4 * x = 360) : 
  x = 180 / 11 := 
by
  sorry

end find_x_l1706_170623


namespace total_fish_count_l1706_170606

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ)
  (h1 : num_fishbowls = 261) (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := 
  by 
    sorry

end total_fish_count_l1706_170606


namespace prob1_prob2_max_area_prob3_circle_diameter_l1706_170658

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end prob1_prob2_max_area_prob3_circle_diameter_l1706_170658


namespace find_number_l1706_170619

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by {
  sorry
}

end find_number_l1706_170619


namespace fraction_div_subtract_l1706_170605

theorem fraction_div_subtract : 
  (5 / 6 : ℚ) / (9 / 10) - (1 / 15) = 116 / 135 := 
by 
  sorry

end fraction_div_subtract_l1706_170605


namespace polygon_interior_sum_polygon_angle_ratio_l1706_170678

-- Part 1: Number of sides based on the sum of interior angles
theorem polygon_interior_sum (n: ℕ) (h: (n - 2) * 180 = 2340) : n = 15 :=
  sorry

-- Part 2: Number of sides based on the ratio of interior to exterior angles
theorem polygon_angle_ratio (n: ℕ) (exterior_angle: ℕ) (ratio: 13 * exterior_angle + 2 * exterior_angle = 180) : n = 15 :=
  sorry

end polygon_interior_sum_polygon_angle_ratio_l1706_170678


namespace fixed_point_of_log_function_l1706_170696

theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (-1, 2) ∧ ∀ x y : ℝ, y = 2 + Real.logb a (x + 2) → y = 2 → x = -1 :=
by
  sorry

end fixed_point_of_log_function_l1706_170696


namespace product_of_roots_l1706_170668

theorem product_of_roots : 
  ∀ (r1 r2 r3 : ℝ), (2 * r1 * r2 * r3 - 3 * (r1 * r2 + r2 * r3 + r3 * r1) - 15 * (r1 + r2 + r3) + 35 = 0) → 
  (r1 * r2 * r3 = -35 / 2) :=
by
  sorry

end product_of_roots_l1706_170668


namespace value_of_expression_l1706_170613

theorem value_of_expression :
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 :=
by
  sorry

end value_of_expression_l1706_170613


namespace red_peppers_weight_l1706_170673

theorem red_peppers_weight (total_weight green_weight : ℝ) (h1 : total_weight = 5.666666667) (h2 : green_weight = 2.8333333333333335) : 
  total_weight - green_weight = 2.8333333336666665 :=
by
  sorry

end red_peppers_weight_l1706_170673


namespace pair_basis_of_plane_l1706_170652

def vector_space := Type
variable (V : Type) [AddCommGroup V] [Module ℝ V]

variables (e1 e2 : V)
variable (h_basis : LinearIndependent ℝ ![e1, e2])
variable (hne : e1 ≠ 0 ∧ e2 ≠ 0)

theorem pair_basis_of_plane
  (v1 v2 : V)
  (hv1 : v1 = e1 + e2)
  (hv2 : v2 = e1 - e2) :
  LinearIndependent ℝ ![v1, v2] :=
sorry

end pair_basis_of_plane_l1706_170652


namespace compressor_stations_distances_l1706_170662

theorem compressor_stations_distances 
    (x y z a : ℝ) 
    (h1 : x + y = 2 * z)
    (h2 : z + y = x + a)
    (h3 : x + z = 75)
    (h4 : 0 ≤ x)
    (h5 : 0 ≤ y)
    (h6 : 0 ≤ z)
    (h7 : 0 < a)
    (h8 : a < 100) :
  (a = 15 → x = 42 ∧ y = 24 ∧ z = 33) :=
by 
  intro ha_eq_15
  sorry

end compressor_stations_distances_l1706_170662


namespace no_prime_divisible_by_77_l1706_170661

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l1706_170661


namespace complex_imag_part_of_z_l1706_170618

theorem complex_imag_part_of_z (z : ℂ) (h : z * (2 + ⅈ) = 3 - 6 * ⅈ) : z.im = -3 := by
  sorry

end complex_imag_part_of_z_l1706_170618


namespace one_eighth_of_2_pow_44_eq_2_pow_x_l1706_170639

theorem one_eighth_of_2_pow_44_eq_2_pow_x (x : ℕ) :
  (2^44 / 8 = 2^x) → x = 41 :=
by
  sorry

end one_eighth_of_2_pow_44_eq_2_pow_x_l1706_170639


namespace least_sum_of_exponents_520_l1706_170686

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l1706_170686


namespace arithmetic_seq_min_S19_l1706_170654

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_seq_min_S19
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_S8 : S a 8 ≤ 6)
  (h_S11 : S a 11 ≥ 27) :
  S a 19 ≥ 133 :=
sorry

end arithmetic_seq_min_S19_l1706_170654


namespace one_fifty_percent_of_eighty_l1706_170693

theorem one_fifty_percent_of_eighty : (150 / 100) * 80 = 120 :=
  by sorry

end one_fifty_percent_of_eighty_l1706_170693


namespace arithmetic_sequence_sum_of_bn_l1706_170690

variable (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ)

theorem arithmetic_sequence (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6) :
  (∀ n, a n = 2 * n) :=
by sorry

theorem sum_of_bn (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6)
                  (h3 : ∀ n, a n = 2 * n)
                  (h4 : ∀ n, b n = 4 / (a n * a (n + 1))) :
  (∀ n, S n = n / (n + 1)) :=
by sorry

end arithmetic_sequence_sum_of_bn_l1706_170690


namespace max_fans_theorem_l1706_170681

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l1706_170681


namespace vasya_correct_l1706_170657

-- Define the condition of a convex quadrilateral
def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a < 180 ∧ b < 180 ∧ c < 180 ∧ d < 180

-- Define the properties of forming two types of triangles from a quadrilateral
def can_form_two_acute_triangles (a b c d : ℝ) : Prop :=
  a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90

def can_form_two_right_triangles (a b c d : ℝ) : Prop :=
  (a = 90 ∧ b = 90) ∨ (b = 90 ∧ c = 90) ∨ (c = 90 ∧ d = 90) ∨ (d = 90 ∧ a = 90)

def can_form_two_obtuse_triangles (a b c d : ℝ) : Prop :=
  ∃ x y z w, (x > 90 ∧ y < 90 ∧ z < 90 ∧ w < 90 ∧ (x + y + z + w = 360)) ∧
             (x > 90 ∨ y > 90 ∨ z > 90 ∨ w > 90)

-- Prove that Vasya's claim is definitively correct
theorem vasya_correct (a b c d : ℝ) (h : convex_quadrilateral a b c d) :
  can_form_two_obtuse_triangles a b c d ∧
  ¬(can_form_two_acute_triangles a b c d) ∧
  ¬(can_form_two_right_triangles a b c d) ∨
  can_form_two_right_triangles a b c d ∧
  can_form_two_obtuse_triangles a b c d := sorry

end vasya_correct_l1706_170657


namespace parabola_axis_l1706_170671

section
variable (x y : ℝ)

-- Condition: Defines the given parabola equation.
def parabola_eq (x y : ℝ) : Prop := x = (1 / 4) * y^2

-- The Proof Problem: Prove that the axis of this parabola is x = -1/2.
theorem parabola_axis (h : parabola_eq x y) : x = - (1 / 2) := 
sorry
end

end parabola_axis_l1706_170671


namespace sum_of_acute_angles_l1706_170647

theorem sum_of_acute_angles (α β : ℝ) (t : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 2 / t) (h_tanβ : Real.tan β = t / 15)
  (h_min : 10 * Real.tan α + 3 * Real.tan β = 4) :
  α + β = π / 4 :=
sorry

end sum_of_acute_angles_l1706_170647


namespace value_of_x_add_y_not_integer_l1706_170637

theorem value_of_x_add_y_not_integer (x y: ℝ) (h1: y = 3 * ⌊x⌋ + 4) (h2: y = 2 * ⌊x - 3⌋ + 7) (h3: ¬ ∃ n: ℤ, x = n): -8 < x + y ∧ x + y < -7 := 
sorry

end value_of_x_add_y_not_integer_l1706_170637


namespace beam_reflection_problem_l1706_170655

theorem beam_reflection_problem
  (A B D C : Point)
  (angle_CDA : ℝ)
  (total_path_length_max : ℝ)
  (equal_angle_reflections : ∀ (k : ℕ), angle_CDA * k ≤ 90)
  (path_length_constraint : ∀ (n : ℕ) (d : ℝ), 2 * n * d ≤ total_path_length_max)
  : angle_CDA = 5 ∧ total_path_length_max = 100 → ∃ (n : ℕ), n = 10 :=
sorry

end beam_reflection_problem_l1706_170655


namespace ribbon_length_ratio_l1706_170641

theorem ribbon_length_ratio (original_length reduced_length : ℕ) (h1 : original_length = 55) (h2 : reduced_length = 35) : 
  (original_length / Nat.gcd original_length reduced_length) = 11 ∧
  (reduced_length / Nat.gcd original_length reduced_length) = 7 := 
  by
    sorry

end ribbon_length_ratio_l1706_170641


namespace group_C_questions_l1706_170672

theorem group_C_questions (a b c : ℕ) (total_questions : ℕ) (h1 : a + b + c = 100)
  (h2 : b = 23)
  (h3 : a ≥ (6 * (a + 2 * b + 3 * c)) / 10)
  (h4 : 2 * b ≤ (25 * (a + 2 * b + 3 * c)) / 100)
  (h5 : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c) :
  c = 1 :=
sorry

end group_C_questions_l1706_170672


namespace aria_analysis_time_l1706_170622

-- Definitions for the number of bones in each section
def skull_bones : ℕ := 29
def spine_bones : ℕ := 33
def thorax_bones : ℕ := 37
def upper_limb_bones : ℕ := 64
def lower_limb_bones : ℕ := 62

-- Definitions for the time spent per bone in each section (in minutes)
def time_per_skull_bone : ℕ := 15
def time_per_spine_bone : ℕ := 10
def time_per_thorax_bone : ℕ := 12
def time_per_upper_limb_bone : ℕ := 8
def time_per_lower_limb_bone : ℕ := 10

-- Definition for the total time needed in minutes
def total_time_in_minutes : ℕ :=
  (skull_bones * time_per_skull_bone) +
  (spine_bones * time_per_spine_bone) +
  (thorax_bones * time_per_thorax_bone) +
  (upper_limb_bones * time_per_upper_limb_bone) +
  (lower_limb_bones * time_per_lower_limb_bone)

-- Definition for the total time needed in hours
def total_time_in_hours : ℚ := total_time_in_minutes / 60

-- Theorem to prove the total time needed in hours is approximately 39.02
theorem aria_analysis_time : abs (total_time_in_hours - 39.02) < 0.01 :=
by
  sorry

end aria_analysis_time_l1706_170622


namespace shanmukham_total_payment_l1706_170632

noncomputable def total_price_shanmukham_pays : Real :=
  let itemA_price : Real := 6650
  let itemA_rebate : Real := 6 -- percentage
  let itemA_tax : Real := 10 -- percentage

  let itemB_price : Real := 8350
  let itemB_rebate : Real := 4 -- percentage
  let itemB_tax : Real := 12 -- percentage

  let itemC_price : Real := 9450
  let itemC_rebate : Real := 8 -- percentage
  let itemC_tax : Real := 15 -- percentage

  let final_price (price : Real) (rebate : Real) (tax : Real) : Real :=
    let rebate_amt := (rebate / 100) * price
    let price_after_rebate := price - rebate_amt
    let tax_amt := (tax / 100) * price_after_rebate
    price_after_rebate + tax_amt

  final_price itemA_price itemA_rebate itemA_tax +
  final_price itemB_price itemB_rebate itemB_tax +
  final_price itemC_price itemC_rebate itemC_tax

theorem shanmukham_total_payment :
  total_price_shanmukham_pays = 25852.12 := by
  sorry

end shanmukham_total_payment_l1706_170632


namespace complement_union_A_B_eq_neg2_0_l1706_170675

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l1706_170675


namespace unique_injective_f_solution_l1706_170610

noncomputable def unique_injective_function (f : ℝ → ℝ) : Prop := 
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))

theorem unique_injective_f_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))
  → (∀ x y : ℝ, f x = f y → x = y) -- injectivity condition
  → ∀ x : ℝ, f x = x :=
sorry

end unique_injective_f_solution_l1706_170610


namespace solution_set_inequality_l1706_170699

theorem solution_set_inequality (m : ℤ) (h₁ : (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2)) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} :=
by
  -- The detailed proof would be added here.
  sorry

end solution_set_inequality_l1706_170699


namespace total_mile_times_l1706_170626

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l1706_170626


namespace num_unique_m_values_l1706_170670

theorem num_unique_m_values : 
  ∃ (s : Finset Int), 
  (∀ (x1 x2 : Int), x1 * x2 = 36 → x1 + x2 ∈ s) ∧ 
  s.card = 10 := 
sorry

end num_unique_m_values_l1706_170670


namespace johns_bakery_fraction_l1706_170608

theorem johns_bakery_fraction :
  ∀ (M : ℝ), 
  (M / 4 + M / 3 + 6 + (24 - (M / 4 + M / 3 + 6)) = 24) →
  (24 : ℝ) = M →
  (4 + 8 + 6 = 18) →
  (24 - 18 = 6) →
  (6 / 24 = (1 / 6 : ℝ)) :=
by
  intros M h1 h2 h3 h4
  sorry

end johns_bakery_fraction_l1706_170608


namespace suzannes_book_pages_l1706_170698

-- Conditions
def pages_read_on_monday : ℕ := 15
def pages_read_on_tuesday : ℕ := 31
def pages_left : ℕ := 18

-- Total number of pages in the book
def total_pages : ℕ := pages_read_on_monday + pages_read_on_tuesday + pages_left

-- Problem statement
theorem suzannes_book_pages : total_pages = 64 :=
by
  -- Proof is not required, only the statement
  sorry

end suzannes_book_pages_l1706_170698


namespace sqrt_six_plus_s_cubed_l1706_170621

theorem sqrt_six_plus_s_cubed (s : ℝ) : 
    Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) :=
sorry

end sqrt_six_plus_s_cubed_l1706_170621


namespace gcd_2197_2209_l1706_170667

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := 
by
  sorry

end gcd_2197_2209_l1706_170667


namespace infinite_pairs_m_n_l1706_170625

theorem infinite_pairs_m_n :
  ∃ (f : ℕ → ℕ × ℕ), (∀ k, (f k).1 > 0 ∧ (f k).2 > 0 ∧ ((f k).1 ∣ (f k).2 ^ 2 + 1) ∧ ((f k).2 ∣ (f k).1 ^ 2 + 1)) :=
sorry

end infinite_pairs_m_n_l1706_170625


namespace line_circle_intersection_l1706_170695

-- Define the line and circle in Lean
def line_eq (x y : ℝ) : Prop := x + y - 6 = 0
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 2

-- Define the proof about the intersection
theorem line_circle_intersection :
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧
  ∀ (x1 y1 x2 y2 : ℝ), (line_eq x1 y1 ∧ circle_eq x1 y1) → (line_eq x2 y2 ∧ circle_eq x2 y2) → (x1 = x2 ∧ y1 = y2) :=
by {
  sorry
}

end line_circle_intersection_l1706_170695


namespace individual_weight_l1706_170602

def total_students : ℕ := 1500
def sampled_students : ℕ := 100

def individual := "the weight of each student"

theorem individual_weight :
  (total_students = 1500) →
  (sampled_students = 100) →
  individual = "the weight of each student" :=
by
  intros h1 h2
  sorry

end individual_weight_l1706_170602


namespace largest_n_for_crates_l1706_170659

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end largest_n_for_crates_l1706_170659


namespace gain_percentage_second_book_l1706_170642

theorem gain_percentage_second_book (C1 C2 SP1 SP2 : ℝ) (H1 : C1 + C2 = 360) (H2 : C1 = 210) (H3 : SP1 = C1 - (15 / 100) * C1) (H4 : SP1 = SP2) (H5 : SP2 = C2 + (19 / 100) * C2) : 
  (19 : ℝ) = 19 := 
by
  sorry

end gain_percentage_second_book_l1706_170642


namespace arithmetic_sequence_n_is_17_l1706_170600

theorem arithmetic_sequence_n_is_17
  (a : ℕ → ℤ)  -- An arithmetic sequence a_n
  (h1 : a 1 = 5)  -- First term is 5
  (h5 : a 5 = -3)  -- Fifth term is -3
  (hn : a n = -27) : n = 17 := sorry

end arithmetic_sequence_n_is_17_l1706_170600


namespace smallest_square_area_l1706_170650

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 4) (h4 : d = 5) :
  ∃ s, s^2 = 81 ∧ (a ≤ s ∧ b ≤ s ∧ c ≤ s ∧ d ≤ s ∧ (a + c) ≤ s ∧ (b + d) ≤ s) :=
sorry

end smallest_square_area_l1706_170650


namespace distance_between_points_l1706_170679

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 2 5 5 1 = 5 :=
by
  sorry

end distance_between_points_l1706_170679


namespace problem_statement_l1706_170689

theorem problem_statement (a b c : ℝ) (h1 : a - b = 2) (h2 : b - c = -3) : a - c = -1 := 
by
  sorry

end problem_statement_l1706_170689


namespace proportional_sets_l1706_170653

/-- Prove that among the sets of line segments, the ones that are proportional are: -/
theorem proportional_sets : 
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  ∃ a b c d, (a, b, c, d) = C ∧ (a * d = b * c) :=
by
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  sorry

end proportional_sets_l1706_170653


namespace distance_to_place_l1706_170636

theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) (D : ℝ) :
  rowing_speed = 5 ∧ current_speed = 1 ∧ total_time = 1 →
  D = 2.4 :=
by
  -- Rowing Parameters
  let V_d := rowing_speed + current_speed
  let V_u := rowing_speed - current_speed
  
  -- Time Variables
  let T_d := total_time / (V_d + V_u)
  let T_u := total_time - T_d

  -- Distance Calculations
  let D1 := V_d * T_d
  let D2 := V_u * T_u

  -- Prove D is the same distance both upstream and downstream
  sorry

end distance_to_place_l1706_170636


namespace original_price_of_coffee_l1706_170666

variable (P : ℝ)

theorem original_price_of_coffee :
  (4 * P - 2 * (1.5 * P) = 2) → P = 2 :=
by
  sorry

end original_price_of_coffee_l1706_170666


namespace woman_completion_days_l1706_170624

variable (M W : ℚ)
variable (work_days_man work_days_total : ℚ)

-- Given conditions
def condition1 : Prop :=
  (10 * M + 15 * W) * 7 = 1

def condition2 : Prop :=
  M * 100 = 1

-- To prove
def one_woman_days : ℚ := 350

theorem woman_completion_days (h1 : condition1 M W) (h2 : condition2 M) :
  1 / W = one_woman_days :=
by
  sorry

end woman_completion_days_l1706_170624


namespace problem_statement_l1706_170688

open Set

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem problem_statement :
  {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} :=
by
  sorry

end problem_statement_l1706_170688


namespace find_angle_A_find_cos2C_minus_pi_over_6_l1706_170607

noncomputable def triangle_area_formula (a b c : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

noncomputable def given_area_formula (b c : ℝ) (S : ℝ) (a : ℝ) (C : ℝ) : Prop :=
  S = (Real.sqrt 3 / 6) * b * (b + c - a * Real.cos C)

noncomputable def angle_A (S b c a C : ℝ) (h : given_area_formula b c S a C) : ℝ :=
  Real.arcsin ((Real.sqrt 3 / 3) * (b + c - a * Real.cos C))

theorem find_angle_A (a b c S C : ℝ) (h : given_area_formula b c S a C) :
  angle_A S b c a C h = π / 3 :=
sorry

-- Part 2 related definitions
noncomputable def cos2C_minus_pi_over_6 (b c a C : ℝ) : ℝ :=
  let cos_C := (b^2 + c^2 - a^2) / (2 * b * c)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let cos_2C := 2 * cos_C^2 - 1
  let sin_2C := 2 * sin_C * cos_C
  cos_2C * (Real.sqrt 3 / 2) + sin_2C * (1 / 2)

theorem find_cos2C_minus_pi_over_6 (b c a C : ℝ) (hb : b = 1) (hc : c = 3) (ha : a = Real.sqrt 7) :
  cos2C_minus_pi_over_6 b c a C = - (4 * Real.sqrt 3 / 7) :=
sorry

end find_angle_A_find_cos2C_minus_pi_over_6_l1706_170607


namespace solve_problem_l1706_170634
noncomputable def is_solution (n : ℕ) : Prop :=
  ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) → (a + b + c ∣ a^2 + b^2 + c^2) → (a + b + c ∣ a^n + b^n + c^n)

theorem solve_problem : {n : ℕ // is_solution (3 * n - 1) ∧ is_solution (3 * n - 2)} :=
sorry

end solve_problem_l1706_170634


namespace blue_eyed_kitten_percentage_is_correct_l1706_170648

def total_blue_eyed_kittens : ℕ := 5 + 6 + 4 + 7 + 3

def total_kittens : ℕ := 12 + 16 + 11 + 19 + 12

def percentage_blue_eyed_kittens (blue : ℕ) (total : ℕ) : ℚ := (blue : ℚ) / (total : ℚ) * 100

theorem blue_eyed_kitten_percentage_is_correct :
  percentage_blue_eyed_kittens total_blue_eyed_kittens total_kittens = 35.71 := sorry

end blue_eyed_kitten_percentage_is_correct_l1706_170648


namespace driver_travel_distance_per_week_l1706_170697

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end driver_travel_distance_per_week_l1706_170697


namespace coefficient_of_x9_in_expansion_l1706_170680

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end coefficient_of_x9_in_expansion_l1706_170680


namespace total_votes_election_l1706_170633

theorem total_votes_election (total_votes fiona_votes elena_votes devin_votes : ℝ) 
  (Fiona_fraction : fiona_votes = (4/15) * total_votes)
  (Elena_fiona : elena_votes = fiona_votes + 15)
  (Devin_elena : devin_votes = 2 * elena_votes)
  (total_eq : total_votes = fiona_votes + elena_votes + devin_votes) :
  total_votes = 675 := 
sorry

end total_votes_election_l1706_170633


namespace simplify_expression_l1706_170635

variable (a b : ℝ)

theorem simplify_expression : 
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := 
  sorry

end simplify_expression_l1706_170635


namespace scientists_from_usa_l1706_170663

theorem scientists_from_usa (total_scientists : ℕ)
  (from_europe : ℕ)
  (from_canada : ℕ)
  (h1 : total_scientists = 70)
  (h2 : from_europe = total_scientists / 2)
  (h3 : from_canada = total_scientists / 5) :
  (total_scientists - from_europe - from_canada) = 21 :=
by
  sorry

end scientists_from_usa_l1706_170663


namespace train_length_l1706_170682

noncomputable def convert_speed (v_kmh : ℝ) : ℝ :=
  v_kmh * (5 / 18)

def length_of_train (speed_mps : ℝ) (time_sec : ℝ) : ℝ :=
  speed_mps * time_sec

theorem train_length (v_kmh : ℝ) (t_sec : ℝ) (length_m : ℝ) :
  v_kmh = 60 →
  t_sec = 45 →
  length_m = 750 →
  length_of_train (convert_speed v_kmh) t_sec = length_m :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_length_l1706_170682


namespace sum_of_two_squares_l1706_170649

theorem sum_of_two_squares (n : ℕ) (k m : ℤ) : 2 * n = k^2 + m^2 → ∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end sum_of_two_squares_l1706_170649


namespace cone_height_is_2_sqrt_15_l1706_170656

noncomputable def height_of_cone (radius : ℝ) (num_sectors : ℕ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let sector_arc_length := circumference / num_sectors
  let base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := radius
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2)

theorem cone_height_is_2_sqrt_15 :
  height_of_cone 8 4 = 2 * Real.sqrt 15 :=
by
  sorry

end cone_height_is_2_sqrt_15_l1706_170656


namespace quadratic_no_real_roots_range_l1706_170609

theorem quadratic_no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 + 2 * x - k = 0)) ↔ k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l1706_170609


namespace factorization_example_l1706_170692

theorem factorization_example (x: ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
sorry

end factorization_example_l1706_170692


namespace cookie_weight_l1706_170629

theorem cookie_weight :
  ∀ (pounds_per_box cookies_per_box ounces_per_pound : ℝ),
    pounds_per_box = 40 →
    cookies_per_box = 320 →
    ounces_per_pound = 16 →
    (pounds_per_box * ounces_per_pound) / cookies_per_box = 2 := 
by 
  intros pounds_per_box cookies_per_box ounces_per_pound hpounds hcookies hounces
  rw [hpounds, hcookies, hounces]
  norm_num

end cookie_weight_l1706_170629


namespace slope_of_line_is_neg_one_l1706_170612

theorem slope_of_line_is_neg_one (y : ℝ) (h : (y - 5) / (5 - (-3)) = -1) : y = -3 :=
by
  sorry

end slope_of_line_is_neg_one_l1706_170612


namespace max_n_value_l1706_170631

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h_ineq : 1/(a - b) + 1/(b - c) ≥ n/(a - c)) : n ≤ 4 := 
sorry

end max_n_value_l1706_170631


namespace speed_ratio_l1706_170628

variable (d_A d_B : ℝ) (t_A t_B : ℝ)

-- Define the conditions
def condition1 : Prop := d_A = (1 + 1/5) * d_B
def condition2 : Prop := t_B = (1 - 1/11) * t_A

-- State the theorem that the speed ratio is 12:11
theorem speed_ratio (h1 : condition1 d_A d_B) (h2 : condition2 t_A t_B) :
  (d_A / t_A) / (d_B / t_B) = 12 / 11 :=
sorry

end speed_ratio_l1706_170628


namespace car_r_speed_l1706_170664

theorem car_r_speed (v : ℝ) (h : 150 / v - 2 = 150 / (v + 10)) : v = 25 :=
sorry

end car_r_speed_l1706_170664


namespace find_k_l1706_170638

noncomputable def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_k
  (k : ℝ)
  (a b c : ℝ × ℝ)
  (ha : a = vec 3 1)
  (hb : b = vec 1 3)
  (hc : c = vec k (-2))
  (h_perp : dot_product (vec (a.1 - c.1) (a.2 - c.2)) (vec (a.1 - b.1) (a.2 - b.2)) = 0) :
  k = 0 :=
sorry

end find_k_l1706_170638


namespace min_expression_value_l1706_170630

theorem min_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ x : ℝ, x = 5 ∧ ∀ y, (y = (b / (3 * a) + 3 / b)) → x ≤ y :=
by
  sorry

end min_expression_value_l1706_170630


namespace customer_can_receive_exact_change_l1706_170644

theorem customer_can_receive_exact_change (k : ℕ) (hk : k ≤ 1000) :
  ∃ change : ℕ, change + k = 1000 ∧ change ≤ 1999 :=
by
  sorry

end customer_can_receive_exact_change_l1706_170644


namespace number_of_candidates_l1706_170615

theorem number_of_candidates
  (n : ℕ)
  (h : n * (n - 1) = 132) : 
  n = 12 :=
sorry

end number_of_candidates_l1706_170615


namespace train_seat_count_l1706_170651

theorem train_seat_count (t : ℝ) (h1 : 0.20 * t = 0.2 * t)
  (h2 : 0.60 * t = 0.6 * t) (h3 : 30 + 0.20 * t + 0.60 * t = t) : t = 150 :=
by
  sorry

end train_seat_count_l1706_170651


namespace necessary_and_sufficient_condition_l1706_170601

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (-16 ≤ a ∧ a ≤ 0) ↔ ∀ x : ℝ, ¬(x^2 + a * x - 4 * a < 0) :=
by
  sorry

end necessary_and_sufficient_condition_l1706_170601


namespace sale_in_second_month_l1706_170614

def sale_first_month : ℕ := 6435
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6191
def average_sale : ℕ := 6700

theorem sale_in_second_month : 
  ∀ (sale_second_month : ℕ), 
    (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month = 6700 * 6) → 
    sale_second_month = 6927 :=
by
  intro sale_second_month h
  sorry

end sale_in_second_month_l1706_170614


namespace smallest_sum_of_three_l1706_170604

open Finset

-- Define the set of numbers
def my_set : Finset ℤ := {10, 2, -4, 15, -7}

-- Statement of the problem: Prove the smallest sum of any three different numbers from the set is -9
theorem smallest_sum_of_three :
  ∃ (a b c : ℤ), a ∈ my_set ∧ b ∈ my_set ∧ c ∈ my_set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
sorry

end smallest_sum_of_three_l1706_170604


namespace smallest_number_l1706_170645

theorem smallest_number (a b c d : ℤ) (h1 : a = -2) (h2 : b = 0) (h3 : c = -3) (h4 : d = 1) : 
  min (min a b) (min c d) = c :=
by
  -- Proof goes here
  sorry

end smallest_number_l1706_170645


namespace interest_amount_eq_750_l1706_170620

-- Definitions
def P : ℕ := 3000
def R : ℕ := 5
def T : ℕ := 5

-- Condition
def interest_less_than_sum := 2250

-- Simple interest formula
def simple_interest (P R T : ℕ) := (P * R * T) / 100

-- Theorem
theorem interest_amount_eq_750 : simple_interest P R T = P - interest_less_than_sum :=
by
  -- We assert that we need to prove the equality holds.
  sorry

end interest_amount_eq_750_l1706_170620


namespace change_is_correct_l1706_170677

-- Define the cost of the pencil in cents
def cost_of_pencil : ℕ := 35

-- Define the amount paid in cents
def amount_paid : ℕ := 100

-- State the theorem for the change
theorem change_is_correct : amount_paid - cost_of_pencil = 65 :=
by sorry

end change_is_correct_l1706_170677


namespace janine_total_pages_l1706_170640

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end janine_total_pages_l1706_170640


namespace notebook_cost_l1706_170627

theorem notebook_cost
  (students : ℕ)
  (majority_students : ℕ)
  (cost : ℕ)
  (notebooks : ℕ)
  (h1 : students = 36)
  (h2 : majority_students > 18)
  (h3 : notebooks > 1)
  (h4 : cost > notebooks)
  (h5 : majority_students * cost * notebooks = 2079) :
  cost = 11 :=
by
  sorry

end notebook_cost_l1706_170627


namespace expected_value_abs_diff_HT_l1706_170611

noncomputable def expected_abs_diff_HT : ℚ :=
  let F : ℕ → ℚ := sorry -- Recurrence relation omitted for brevity
  F 0

theorem expected_value_abs_diff_HT :
  expected_abs_diff_HT = 24 / 7 :=
sorry

end expected_value_abs_diff_HT_l1706_170611


namespace intersection_points_l1706_170646

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 12 * x - 5
def parabola2 (x : ℝ) : ℝ := x ^ 2 - 2 * x + 3

theorem intersection_points :
  { p : ℝ × ℝ | p.snd = parabola1 p.fst ∧ p.snd = parabola2 p.fst } =
  { (1, -14), (4, -5) } :=
by
  sorry

end intersection_points_l1706_170646


namespace length_breadth_difference_l1706_170665

theorem length_breadth_difference (b l : ℕ) (h1 : b = 5) (h2 : l * b = 15 * b) : l - b = 10 :=
by
  sorry

end length_breadth_difference_l1706_170665


namespace find_large_number_l1706_170685

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 50000) 
  (h2 : L = 13 * S + 317) : 
  L = 54140 := 
sorry

end find_large_number_l1706_170685


namespace age_difference_l1706_170674

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a - c = 18 :=
by
  sorry

end age_difference_l1706_170674


namespace area_ratio_of_similar_triangles_l1706_170643

noncomputable def similarity_ratio := 3 / 5

theorem area_ratio_of_similar_triangles (k : ℝ) (h_sim : similarity_ratio = k) : (k^2 = 9 / 25) :=
by
  sorry

end area_ratio_of_similar_triangles_l1706_170643


namespace bruce_age_multiple_of_son_l1706_170669

structure Person :=
  (age : ℕ)

def bruce := Person.mk 36
def son := Person.mk 8
def multiple := 3

theorem bruce_age_multiple_of_son :
  ∃ (x : ℕ), bruce.age + x = multiple * (son.age + x) ∧ x = 6 :=
by
  use 6
  sorry

end bruce_age_multiple_of_son_l1706_170669


namespace elle_practices_hours_l1706_170683

variable (practice_time_weekday : ℕ) (days_weekday : ℕ) (multiplier_saturday : ℕ) (minutes_in_an_hour : ℕ) 
          (total_minutes_weekdays : ℕ) (total_minutes_saturday : ℕ) (total_minutes_week : ℕ) (total_hours : ℕ)

theorem elle_practices_hours :
  practice_time_weekday = 30 ∧
  days_weekday = 5 ∧
  multiplier_saturday = 3 ∧
  minutes_in_an_hour = 60 →
  total_minutes_weekdays = practice_time_weekday * days_weekday →
  total_minutes_saturday = practice_time_weekday * multiplier_saturday →
  total_minutes_week = total_minutes_weekdays + total_minutes_saturday →
  total_hours = total_minutes_week / minutes_in_an_hour →
  total_hours = 4 :=
by
  intros
  sorry

end elle_practices_hours_l1706_170683


namespace at_least_one_half_l1706_170694

theorem at_least_one_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
by
  sorry

end at_least_one_half_l1706_170694


namespace car_travel_distance_l1706_170691

-- Define the conditions: speed and time
def speed : ℝ := 160 -- in km/h
def time : ℝ := 5 -- in hours

-- Define the calculation for distance
def distance (s t : ℝ) : ℝ := s * t

-- Prove that given the conditions, the distance is 800 km
theorem car_travel_distance : distance speed time = 800 := by
  sorry

end car_travel_distance_l1706_170691


namespace peregrines_eat_30_percent_l1706_170676

theorem peregrines_eat_30_percent (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (pigeons_left : ℕ) :
  initial_pigeons = 40 →
  chicks_per_pigeon = 6 →
  pigeons_left = 196 →
  (100 * (initial_pigeons * chicks_per_pigeon + initial_pigeons - pigeons_left)) / 
  (initial_pigeons * chicks_per_pigeon + initial_pigeons) = 30 :=
by
  intros
  sorry

end peregrines_eat_30_percent_l1706_170676


namespace probability_of_neither_red_nor_purple_l1706_170660

theorem probability_of_neither_red_nor_purple :
  let total_balls := 100
  let white_balls := 20
  let green_balls := 30
  let yellow_balls := 10
  let red_balls := 37
  let purple_balls := 3
  let neither_red_nor_purple_balls := white_balls + green_balls + yellow_balls
  (neither_red_nor_purple_balls : ℝ) / (total_balls : ℝ) = 0.6 :=
by
  sorry

end probability_of_neither_red_nor_purple_l1706_170660


namespace units_digit_of_n_l1706_170603

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : m % 10 = 4) : n % 10 = 4 :=
by
  sorry

end units_digit_of_n_l1706_170603


namespace inverse_of_B_cubed_l1706_170684

theorem inverse_of_B_cubed
  (B_inv : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3, -1],
    ![0, 5]
  ]) :
  (B_inv ^ 3) = ![
    ![27, -49],
    ![0, 125]
  ] := 
by
  sorry

end inverse_of_B_cubed_l1706_170684


namespace find_ab_and_m_l1706_170617

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_ab_and_m (a b m : ℝ) (P : ℝ × ℝ)
  (h1 : P = (-1, -2))
  (h2 : ∀ (x : ℝ), (3 * a * x^2 + 2 * b * x) = -1/3 ↔ x = -1)
  (h3 : ∀ (x : ℝ), f a b x = a * x ^ 3 + b * x ^ 2)
  : (a = -13/3 ∧ b = -19/3) ∧ (0 < m ∧ m < 38/39) :=
sorry

end find_ab_and_m_l1706_170617


namespace number_of_kids_l1706_170687

theorem number_of_kids (A K : ℕ) (h1 : A + K = 13) (h2 : 7 * A = 28) : K = 9 :=
by
  sorry

end number_of_kids_l1706_170687
