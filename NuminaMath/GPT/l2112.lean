import Mathlib

namespace NUMINAMATH_GPT_shaded_area_is_one_third_l2112_211236

noncomputable def fractional_shaded_area : ℕ → ℚ
| 0 => 1 / 4
| n + 1 => (1 / 4) * fractional_shaded_area n

theorem shaded_area_is_one_third : (∑' n, fractional_shaded_area n) = 1 / 3 := 
sorry

end NUMINAMATH_GPT_shaded_area_is_one_third_l2112_211236


namespace NUMINAMATH_GPT_cone_height_l2112_211226

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end NUMINAMATH_GPT_cone_height_l2112_211226


namespace NUMINAMATH_GPT_cost_of_each_candy_bar_l2112_211262

theorem cost_of_each_candy_bar
  (p_chips : ℝ)
  (total_cost : ℝ)
  (num_students : ℕ)
  (num_chips_per_student : ℕ)
  (num_candy_bars_per_student : ℕ)
  (h1 : p_chips = 0.50)
  (h2 : total_cost = 15)
  (h3 : num_students = 5)
  (h4 : num_chips_per_student = 2)
  (h5 : num_candy_bars_per_student = 1) :
  ∃ C : ℝ, C = 2 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_candy_bar_l2112_211262


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_a_l2112_211271

-- Define the functions
def f (x a : ℝ) : ℝ := |2 * x - 1| + |2 * x + a|
def g (x : ℝ) : ℝ := x + 3

-- Problem 1: Proving the solution set when a = -2
theorem problem1_solution_set (x : ℝ) : (f x (-2) < g x) ↔ (0 < x ∧ x < 2) :=
  sorry

-- Problem 2: Proving the range of a
theorem problem2_range_of_a (a : ℝ) : 
  (a > -1) ∧ (∀ x, (x ∈ Set.Icc (-a/2) (1/2) → f x a ≤ g x)) ↔ a ∈ Set.Ioo (-1) (4/3) ∨ a = 4/3 :=
  sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_a_l2112_211271


namespace NUMINAMATH_GPT_drum_y_capacity_filled_l2112_211270

-- Definitions of the initial conditions
def capacity_of_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def capacity_of_drum_Y (C : ℝ) (two_c_y : ℝ) := two_c_y = 2 * C
def oil_in_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def oil_in_drum_Y (C : ℝ) (four_fifth_c_y : ℝ) := four_fifth_c_y = 4 / 5 * C

-- Theorem to prove the capacity filled in drum Y after pouring all oil from X
theorem drum_y_capacity_filled {C : ℝ} (hx : 1/2 * C = 1 / 2 * C) (hy : 2 * C = 2 * C) (ox : 1/2 * C = 1 / 2 * C) (oy : 4/5 * 2 * C = 4 / 5 * C) :
  ( (1/2 * C + 4/5 * C) / (2 * C) ) = 13 / 20 :=
by
  sorry

end NUMINAMATH_GPT_drum_y_capacity_filled_l2112_211270


namespace NUMINAMATH_GPT_ratio_of_kids_in_morning_to_total_soccer_l2112_211223

-- Define the known conditions
def total_kids_in_camp : ℕ := 2000
def kids_going_to_soccer_camp : ℕ := total_kids_in_camp / 2
def kids_going_to_soccer_camp_in_afternoon : ℕ := 750
def kids_going_to_soccer_camp_in_morning : ℕ := kids_going_to_soccer_camp - kids_going_to_soccer_camp_in_afternoon

-- Define the conclusion to be proven
theorem ratio_of_kids_in_morning_to_total_soccer :
  (kids_going_to_soccer_camp_in_morning : ℚ) / (kids_going_to_soccer_camp : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_kids_in_morning_to_total_soccer_l2112_211223


namespace NUMINAMATH_GPT_sum_is_zero_l2112_211213

noncomputable def z : ℂ := Complex.cos (3 * Real.pi / 8) + Complex.sin (3 * Real.pi / 8) * Complex.I

theorem sum_is_zero (hz : z^8 = 1) (hz1 : z ≠ 1) :
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^12)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_zero_l2112_211213


namespace NUMINAMATH_GPT_bus_ticket_problem_l2112_211281

variables (x y : ℕ)

theorem bus_ticket_problem (h1 : x + y = 99) (h2 : 2 * x + 3 * y = 280) : x = 17 ∧ y = 82 :=
by
  sorry

end NUMINAMATH_GPT_bus_ticket_problem_l2112_211281


namespace NUMINAMATH_GPT_knives_more_than_forks_l2112_211225

variable (F K S T : ℕ)
variable (x : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  (F = 6) ∧ 
  (K = F + x) ∧ 
  (S = 2 * K) ∧
  (T = F / 2)

-- Total cutlery added
def total_cutlery_added : Prop :=
  (F + 2) + (K + 2) + (S + 2) + (T + 2) = 62

-- Prove that x = 9
theorem knives_more_than_forks :
  initial_conditions F K S T x →
  total_cutlery_added F K S T →
  x = 9 := 
by
  sorry

end NUMINAMATH_GPT_knives_more_than_forks_l2112_211225


namespace NUMINAMATH_GPT_petya_vasya_common_result_l2112_211288

theorem petya_vasya_common_result (a b : ℝ) (h1 : b ≠ 0) (h2 : a/b = (a + b)/(2 * a)) (h3 : a/b ≠ 1) : 
  a/b = -1/2 :=
by 
  sorry

end NUMINAMATH_GPT_petya_vasya_common_result_l2112_211288


namespace NUMINAMATH_GPT_square_diff_problem_l2112_211219

theorem square_diff_problem
  (x : ℤ)
  (h : x^2 = 9801) :
  (x + 3) * (x - 3) = 9792 := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_square_diff_problem_l2112_211219


namespace NUMINAMATH_GPT_sin_B_triangle_area_l2112_211278

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem sin_B (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5) :
  Real.sin B = Real.sqrt 10 / 10 := by
  sorry

theorem triangle_area (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hDiff : c - a = 5 - Real.sqrt 10) (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  1 / 2 * a * c * Real.sin B = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_B_triangle_area_l2112_211278


namespace NUMINAMATH_GPT_calculation_1_calculation_2_calculation_3_calculation_4_l2112_211253

theorem calculation_1 : -3 - (-4) = 1 :=
by sorry

theorem calculation_2 : -1/3 + (-4/3) = -5/3 :=
by sorry

theorem calculation_3 : (-2) * (-3) * (-5) = -30 :=
by sorry

theorem calculation_4 : 15 / 4 * (-1/4) = -15/16 :=
by sorry

end NUMINAMATH_GPT_calculation_1_calculation_2_calculation_3_calculation_4_l2112_211253


namespace NUMINAMATH_GPT_kernel_red_given_popped_l2112_211291

def prob_red_given_popped (P_red : ℚ) (P_green : ℚ) 
                           (P_popped_given_red : ℚ) (P_popped_given_green : ℚ) : ℚ :=
  let P_red_popped := P_red * P_popped_given_red
  let P_green_popped := P_green * P_popped_given_green
  let P_popped := P_red_popped + P_green_popped
  P_red_popped / P_popped

theorem kernel_red_given_popped : prob_red_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end NUMINAMATH_GPT_kernel_red_given_popped_l2112_211291


namespace NUMINAMATH_GPT_min_square_sum_l2112_211264

theorem min_square_sum (a b m n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 15 * a + 16 * b = m * m) (h4 : 16 * a - 15 * b = n * n) : 481 ≤ min (m * m) (n * n) :=
sorry

end NUMINAMATH_GPT_min_square_sum_l2112_211264


namespace NUMINAMATH_GPT_combined_flock_size_after_5_years_l2112_211279

noncomputable def initial_flock_size : ℕ := 100
noncomputable def ducks_killed_per_year : ℕ := 20
noncomputable def ducks_born_per_year : ℕ := 30
noncomputable def years_passed : ℕ := 5
noncomputable def other_flock_size : ℕ := 150

theorem combined_flock_size_after_5_years
  (init_size : ℕ := initial_flock_size)
  (killed_per_year : ℕ := ducks_killed_per_year)
  (born_per_year : ℕ := ducks_born_per_year)
  (years : ℕ := years_passed)
  (other_size : ℕ := other_flock_size) :
  init_size + (years * (born_per_year - killed_per_year)) + other_size = 300 := by
  -- The formal proof would go here.
  sorry

end NUMINAMATH_GPT_combined_flock_size_after_5_years_l2112_211279


namespace NUMINAMATH_GPT_inequality_nonempty_solution_set_l2112_211250

theorem inequality_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x-3| + |x-4| < a) ↔ a > 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_nonempty_solution_set_l2112_211250


namespace NUMINAMATH_GPT_num_pairs_satisfying_inequality_l2112_211231

theorem num_pairs_satisfying_inequality : 
  ∃ (s : Nat), s = 204 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → m^2 + n < 50 → s = 204 :=
by
  sorry

end NUMINAMATH_GPT_num_pairs_satisfying_inequality_l2112_211231


namespace NUMINAMATH_GPT_functional_equation_solution_l2112_211261

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) → (∀ x : ℝ, f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2112_211261


namespace NUMINAMATH_GPT_equation_solution_unique_l2112_211216

theorem equation_solution_unique (x y : ℤ) : 
  x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_unique_l2112_211216


namespace NUMINAMATH_GPT_track_length_l2112_211203

theorem track_length (x : ℝ) 
  (h1 : ∀ {d1 d2 : ℝ}, (d1 + d2 = x / 2) → (d1 = 120) → d2 = x / 2 - 120)
  (h2 : ∀ {d1 d2 : ℝ}, (d1 = x / 2 - 120 + 170) → (d1 = x / 2 + 50))
  (h3 : ∀ {d3 : ℝ}, (d3 = 3 * x / 2 - 170)) :
  x = 418 :=
by
  sorry

end NUMINAMATH_GPT_track_length_l2112_211203


namespace NUMINAMATH_GPT_problem_quadratic_has_real_root_l2112_211292

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end NUMINAMATH_GPT_problem_quadratic_has_real_root_l2112_211292


namespace NUMINAMATH_GPT_puppies_brought_in_correct_l2112_211210

-- Define the initial number of puppies in the shelter
def initial_puppies: Nat := 2

-- Define the number of puppies adopted per day
def puppies_adopted_per_day: Nat := 4

-- Define the number of days over which the puppies are adopted
def adoption_days: Nat := 9

-- Define the total number of puppies adopted after the given days
def total_puppies_adopted: Nat := puppies_adopted_per_day * adoption_days

-- Define the number of puppies brought in
def puppies_brought_in: Nat := total_puppies_adopted - initial_puppies

-- Prove that the number of puppies brought in is 34
theorem puppies_brought_in_correct: puppies_brought_in = 34 := by
  -- proof omitted, filled with sorry to skip the proof
  sorry

end NUMINAMATH_GPT_puppies_brought_in_correct_l2112_211210


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l2112_211211

theorem geometric_sequence_b_value (r b : ℝ) (h1 : 120 * r = b) (h2 : b * r = 27 / 16) (hb_pos : b > 0) : b = 15 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l2112_211211


namespace NUMINAMATH_GPT_burger_cost_proof_l2112_211295

variable {burger_cost fries_cost salad_cost total_cost : ℕ}
variable {quantity_of_fries : ℕ}

theorem burger_cost_proof (h_fries_cost : fries_cost = 2)
    (h_salad_cost : salad_cost = 3 * fries_cost)
    (h_quantity_of_fries : quantity_of_fries = 2)
    (h_total_cost : total_cost = 15)
    (h_equation : burger_cost + (quantity_of_fries * fries_cost) + salad_cost = total_cost) :
    burger_cost = 5 :=
by 
  sorry

end NUMINAMATH_GPT_burger_cost_proof_l2112_211295


namespace NUMINAMATH_GPT_number_of_donuts_correct_l2112_211287

noncomputable def number_of_donuts_in_each_box :=
  let x : ℕ := 12
  let total_boxes : ℕ := 4
  let donuts_given_to_mom : ℕ := x
  let donuts_given_to_sister : ℕ := 6
  let donuts_left : ℕ := 30
  x

theorem number_of_donuts_correct :
  ∀ (x : ℕ),
  (total_boxes * x - donuts_given_to_mom - donuts_given_to_sister = donuts_left) → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_donuts_correct_l2112_211287


namespace NUMINAMATH_GPT_no_complete_divisibility_l2112_211299

-- Definition of non-divisibility
def not_divides (m n : ℕ) := ¬ (m ∣ n)

theorem no_complete_divisibility (a b c d : ℕ) (h : a * d - b * c > 1) : 
  not_divides (a * d - b * c) a ∨ not_divides (a * d - b * c) b ∨ not_divides (a * d - b * c) c ∨ not_divides (a * d - b * c) d :=
by 
  sorry

end NUMINAMATH_GPT_no_complete_divisibility_l2112_211299


namespace NUMINAMATH_GPT_lyle_notebook_cost_l2112_211201

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end NUMINAMATH_GPT_lyle_notebook_cost_l2112_211201


namespace NUMINAMATH_GPT_vector_parallel_example_l2112_211229

theorem vector_parallel_example 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (ha : a = (2, 1)) 
  (hb : b = (4, 2))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  3 • a + 2 • b = (14, 7) := 
by
  sorry

end NUMINAMATH_GPT_vector_parallel_example_l2112_211229


namespace NUMINAMATH_GPT_race_dead_heat_l2112_211240

theorem race_dead_heat (va vb D : ℝ) (hva_vb : va = (15 / 16) * vb) (dist_a : D = D) (dist_b : D = (15 / 16) * D) (race_finish : D / va = (15 / 16) * D / vb) :
  va / vb = 15 / 16 :=
by sorry

end NUMINAMATH_GPT_race_dead_heat_l2112_211240


namespace NUMINAMATH_GPT_count_distribution_schemes_l2112_211232

theorem count_distribution_schemes :
  let total_pieces := 7
  let pieces_A_B := 2 + 2
  let remaining_pieces := total_pieces - pieces_A_B
  let communities := 5

  -- Number of ways to distribute 7 pieces of equipment such that communities A and B receive at least 2 pieces each
  let ways_one_community := 5
  let ways_two_communities := 20  -- 2 * (choose 5 2)
  let ways_three_communities := 10  -- (choose 5 3)

  ways_one_community + ways_two_communities + ways_three_communities = 35 :=
by
  -- The actual proof steps are omitted here.
  sorry

end NUMINAMATH_GPT_count_distribution_schemes_l2112_211232


namespace NUMINAMATH_GPT_intersection_points_relation_l2112_211272

noncomputable def num_intersections (k : ℕ) : ℕ :=
  k * (k - 1) / 2

theorem intersection_points_relation (k : ℕ) :
  num_intersections (k + 1) = num_intersections k + k := by
sorry

end NUMINAMATH_GPT_intersection_points_relation_l2112_211272


namespace NUMINAMATH_GPT_number_of_arrangements_l2112_211277

theorem number_of_arrangements (V T : ℕ) (hV : V = 3) (hT : T = 4) :
  ∃ n : ℕ, n = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l2112_211277


namespace NUMINAMATH_GPT_zack_travel_countries_l2112_211234

theorem zack_travel_countries (G J P Z : ℕ) 
  (hG : G = 6)
  (hJ : J = G / 2)
  (hP : P = 3 * J)
  (hZ : Z = 2 * P) :
  Z = 18 := by
  sorry

end NUMINAMATH_GPT_zack_travel_countries_l2112_211234


namespace NUMINAMATH_GPT_problem_statement_l2112_211247

def op (x y : ℕ) : ℕ := x^2 + 2*y

theorem problem_statement (a : ℕ) : op a (op a a) = 3*a^2 + 4*a := 
by sorry

end NUMINAMATH_GPT_problem_statement_l2112_211247


namespace NUMINAMATH_GPT_jacob_writing_speed_ratio_l2112_211217

theorem jacob_writing_speed_ratio (N : ℕ) (J : ℕ) (hN : N = 25) (h1 : J + N = 75) : J / N = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_jacob_writing_speed_ratio_l2112_211217


namespace NUMINAMATH_GPT_pirate_flag_minimal_pieces_l2112_211214

theorem pirate_flag_minimal_pieces (original_stripes : ℕ) (desired_stripes : ℕ) (cuts_needed : ℕ) : 
  original_stripes = 12 →
  desired_stripes = 10 →
  cuts_needed = 1 →
  ∃ pieces : ℕ, pieces = 2 ∧ 
  (∀ (top_stripes bottom_stripes: ℕ), top_stripes + bottom_stripes = original_stripes → top_stripes = desired_stripes → 
   pieces = 1 + (if bottom_stripes = original_stripes - desired_stripes then 1 else 0)) :=
by intros;
   sorry

end NUMINAMATH_GPT_pirate_flag_minimal_pieces_l2112_211214


namespace NUMINAMATH_GPT_find_angle_B_l2112_211246

noncomputable def angle_B (a b c : ℝ) (B C : ℝ) : Prop :=
b = 2 * Real.sqrt 3 ∧ c = 2 ∧ C = Real.pi / 6 ∧
(Real.sin B = (b * Real.sin C) / c ∧ b > c → (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3))

theorem find_angle_B :
  ∃ (B : ℝ), angle_B 1 (2 * Real.sqrt 3) 2 B (Real.pi / 6) :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_l2112_211246


namespace NUMINAMATH_GPT_min_even_number_for_2015_moves_l2112_211237

theorem min_even_number_for_2015_moves (N : ℕ) (hN : N ≥ 2) :
  ∃ k : ℕ, N = 2 ^ k ∧ 2 ^ k ≥ 2 ∧ k ≥ 4030 :=
sorry

end NUMINAMATH_GPT_min_even_number_for_2015_moves_l2112_211237


namespace NUMINAMATH_GPT_max_lg_value_l2112_211259

noncomputable def max_lg_product (x y : ℝ) (hx: x > 1) (hy: y > 1) (hxy: Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) : ℝ :=
  4

theorem max_lg_value (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  max_lg_product x y hx hy hxy = 4 := 
by
  unfold max_lg_product
  sorry

end NUMINAMATH_GPT_max_lg_value_l2112_211259


namespace NUMINAMATH_GPT_functional_expression_and_range_l2112_211251

-- We define the main problem conditions and prove the required statements based on those conditions
theorem functional_expression_and_range (x y : ℝ) (h1 : ∃ k : ℝ, (y + 2) = k * (4 - x) ∧ k ≠ 0)
                                        (h2 : x = 3 → y = 1) :
                                        (y = -3 * x + 10) ∧ ( -2 < y ∧ y < 1 → 3 < x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_functional_expression_and_range_l2112_211251


namespace NUMINAMATH_GPT_scientific_notation_correct_l2112_211255

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2112_211255


namespace NUMINAMATH_GPT_find_k_x_l2112_211297

-- Define the nonzero polynomial condition
def nonzero_poly (p : Polynomial ℝ) : Prop :=
  ¬ (p = 0)

-- Define the conditions from the problem statement
def conditions (h k : Polynomial ℝ) : Prop :=
  nonzero_poly h ∧ nonzero_poly k ∧ (h.comp k = h * k) ∧ (k.eval 3 = 58)

-- State the main theorem to be proven
theorem find_k_x (h k : Polynomial ℝ) (cond : conditions h k) : 
  k = Polynomial.C 1 + Polynomial.C 49 * Polynomial.X + Polynomial.C (-49) * Polynomial.X^2 :=
sorry

end NUMINAMATH_GPT_find_k_x_l2112_211297


namespace NUMINAMATH_GPT_chemistry_textbook_weight_l2112_211233

theorem chemistry_textbook_weight (G C : ℝ) (h1 : G = 0.62) (h2 : C = G + 6.5) : C = 7.12 :=
by
  sorry

end NUMINAMATH_GPT_chemistry_textbook_weight_l2112_211233


namespace NUMINAMATH_GPT_rectangle_area_l2112_211254

theorem rectangle_area (ABCD : Type*) (small_square : ℕ) (shaded_squares : ℕ) (side_length : ℕ) 
  (shaded_area : ℕ) (width : ℕ) (height : ℕ)
  (H1 : shaded_squares = 3) 
  (H2 : side_length = 2)
  (H3 : shaded_area = side_length * side_length)
  (H4 : width = 6)
  (H5 : height = 4)
  : (width * height) = 24 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2112_211254


namespace NUMINAMATH_GPT_find_d_l2112_211273

theorem find_d (d : ℝ) (h : 3 * (2 - (π / 2)) = 6 + d * π) : d = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l2112_211273


namespace NUMINAMATH_GPT_possible_values_of_a_l2112_211222

variables {a b k : ℤ}

def sum_distances (a : ℤ) (k : ℤ) : ℤ :=
  (a - k).natAbs + (a - (k + 1)).natAbs + (a - (k + 2)).natAbs +
  (a - (k + 3)).natAbs + (a - (k + 4)).natAbs + (a - (k + 5)).natAbs +
  (a - (k + 6)).natAbs + (a - (k + 7)).natAbs + (a - (k + 8)).natAbs +
  (a - (k + 9)).natAbs + (a - (k + 10)).natAbs

theorem possible_values_of_a :
  sum_distances a k = 902 →
  sum_distances b k = 374 →
  a + b = 98 →
  a = 25 ∨ a = 107 ∨ a = -9 :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_l2112_211222


namespace NUMINAMATH_GPT_largest_pack_size_of_markers_l2112_211275

theorem largest_pack_size_of_markers (markers_John markers_Alex : ℕ) (h_John : markers_John = 36) (h_Alex : markers_Alex = 60) : 
  ∃ (n : ℕ), (∀ (x : ℕ), (∀ (y : ℕ), (x * n = markers_John ∧ y * n = markers_Alex) → n ≤ 12) ∧ (12 * x = markers_John ∨ 12 * y = markers_Alex)) :=
by 
  sorry

end NUMINAMATH_GPT_largest_pack_size_of_markers_l2112_211275


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2112_211215

-- Define positive geometric sequence a_n with common ratio q
def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

-- Define the relevant conditions
variable {a q : ℝ}
variable (h1 : a * q^4 + 2 * a * q^2 * q^6 + a * q^4 * q^8 = 16)
variable (h2 : (a * q^4 + a * q^8) / 2 = 4)
variable (pos_q : q > 0)

-- Define the goal: proving the common ratio q is sqrt(2)
theorem common_ratio_of_geometric_sequence : q = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2112_211215


namespace NUMINAMATH_GPT_correct_calculation_l2112_211238

theorem correct_calculation (x : ℝ) (h : 5.46 - x = 3.97) : 5.46 + x = 6.95 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2112_211238


namespace NUMINAMATH_GPT_intersection_A_B_l2112_211285

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2112_211285


namespace NUMINAMATH_GPT_quadratic_polynomial_inequality_l2112_211257

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_inequality_l2112_211257


namespace NUMINAMATH_GPT_ralph_socks_l2112_211205

theorem ralph_socks
  (x y w z : ℕ)
  (h1 : x + y + w + z = 15)
  (h2 : x + 2 * y + 3 * w + 4 * z = 36)
  (hx : x ≥ 1) (hy : y ≥ 1) (hw : w ≥ 1) (hz : z ≥ 1) :
  x = 5 :=
sorry

end NUMINAMATH_GPT_ralph_socks_l2112_211205


namespace NUMINAMATH_GPT_insulation_cost_l2112_211267

def tank_length : ℕ := 4
def tank_width : ℕ := 5
def tank_height : ℕ := 2
def cost_per_sqft : ℕ := 20

def surface_area (L W H : ℕ) : ℕ := 2 * (L * W + L * H + W * H)
def total_cost (SA cost_per_sqft : ℕ) : ℕ := SA * cost_per_sqft

theorem insulation_cost : 
  total_cost (surface_area tank_length tank_width tank_height) cost_per_sqft = 1520 :=
by
  sorry

end NUMINAMATH_GPT_insulation_cost_l2112_211267


namespace NUMINAMATH_GPT_joe_spends_50_per_month_l2112_211252

variable (X : ℕ) -- amount Joe spends per month

theorem joe_spends_50_per_month :
  let initial_amount := 240
  let resale_value := 30
  let months := 12
  let final_amount := 0 -- this means he runs out of money
  (initial_amount = months * X - months * resale_value) →
  X = 50 := 
by
  intros
  sorry

end NUMINAMATH_GPT_joe_spends_50_per_month_l2112_211252


namespace NUMINAMATH_GPT_edward_score_l2112_211274

theorem edward_score (total_points : ℕ) (friend_points : ℕ) 
  (h1 : total_points = 13) (h2 : friend_points = 6) : 
  ∃ edward_points : ℕ, edward_points = 7 :=
by
  sorry

end NUMINAMATH_GPT_edward_score_l2112_211274


namespace NUMINAMATH_GPT_Zlatoust_to_Miass_distance_l2112_211239

theorem Zlatoust_to_Miass_distance
  (x g k m : ℝ)
  (H1 : (x + 18) / k = (x - 18) / m)
  (H2 : (x + 25) / k = (x - 25) / g)
  (H3 : (x + 8) / m = (x - 8) / g) :
  x = 60 :=
sorry

end NUMINAMATH_GPT_Zlatoust_to_Miass_distance_l2112_211239


namespace NUMINAMATH_GPT_number_of_sections_l2112_211218

theorem number_of_sections (pieces_per_section : ℕ) (cost_per_piece : ℕ) (total_cost : ℕ)
  (h1 : pieces_per_section = 30)
  (h2 : cost_per_piece = 2)
  (h3 : total_cost = 480) :
  total_cost / (pieces_per_section * cost_per_piece) = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_sections_l2112_211218


namespace NUMINAMATH_GPT_equation1_sol_equation2_sol_equation3_sol_l2112_211284

theorem equation1_sol (x : ℝ) : 9 * x^2 - (x - 1)^2 = 0 ↔ (x = -0.5 ∨ x = 0.25) :=
sorry

theorem equation2_sol (x : ℝ) : (x * (x - 3) = 10) ↔ (x = 5 ∨ x = -2) :=
sorry

theorem equation3_sol (x : ℝ) : (x + 3)^2 = 2 * x + 5 ↔ (x = -2) :=
sorry

end NUMINAMATH_GPT_equation1_sol_equation2_sol_equation3_sol_l2112_211284


namespace NUMINAMATH_GPT_solve_inverse_function_l2112_211256

-- Define the given functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 1
def g (x : ℝ) : ℝ := x^4 - x^3 + 4*x^2 + 8*x + 8
def h (x : ℝ) : ℝ := x + 1

-- State the mathematical equivalent proof problem
theorem solve_inverse_function (x : ℝ) :
  f ⁻¹' {g x} = {y | h y = x + 1} ↔
  (x = (3 + Real.sqrt 5) / 2) ∨ (x = (3 - Real.sqrt 5) / 2) :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_solve_inverse_function_l2112_211256


namespace NUMINAMATH_GPT_molecular_weight_single_mole_l2112_211200

theorem molecular_weight_single_mole :
  (∀ (w_7m C6H8O7 : ℝ), w_7m = 1344 → (w_7m / 7) = 192) :=
by
  intros w_7m C6H8O7 h
  sorry

end NUMINAMATH_GPT_molecular_weight_single_mole_l2112_211200


namespace NUMINAMATH_GPT_simplify_expression_l2112_211221

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x ^ 2 - 1) / (x ^ 2 + 2 * x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_expression_l2112_211221


namespace NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l2112_211280

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l2112_211280


namespace NUMINAMATH_GPT_sampling_method_is_systematic_l2112_211227

-- Define the conditions
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  required_student_num : Nat

-- Define our specific problem's conditions
def problem_conditions : Grade :=
  { num_classes := 12, students_per_class := 50, required_student_num := 14 }

-- State the theorem
theorem sampling_method_is_systematic (G : Grade) (h1 : G.num_classes = 12) (h2 : G.students_per_class = 50) (h3 : G.required_student_num = 14) : 
  "Systematic sampling" = "Systematic sampling" :=
by
  sorry

end NUMINAMATH_GPT_sampling_method_is_systematic_l2112_211227


namespace NUMINAMATH_GPT_weight_of_square_piece_l2112_211276

open Real

theorem weight_of_square_piece 
  (uniform_density : Prop)
  (side_length_triangle side_length_square : ℝ)
  (weight_triangle : ℝ)
  (ht : side_length_triangle = 6)
  (hs : side_length_square = 6)
  (wt : weight_triangle = 48) :
  ∃ weight_square : ℝ, weight_square = 27.7 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_square_piece_l2112_211276


namespace NUMINAMATH_GPT_increasing_interval_iff_l2112_211286

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 3 * x

def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂

theorem increasing_interval_iff (a : ℝ) (h : a ≠ 0) :
  is_increasing a ↔ a ∈ Set.Ioo (-(5/4)) 0 ∪ Set.Ioi 0 :=
sorry

end NUMINAMATH_GPT_increasing_interval_iff_l2112_211286


namespace NUMINAMATH_GPT_speed_of_slower_train_is_36_l2112_211242

-- Definitions used in the conditions
def length_of_train := 25 -- meters
def combined_length_of_trains := 2 * length_of_train -- meters
def time_to_pass := 18 -- seconds
def speed_of_faster_train := 46 -- km/hr
def conversion_factor := 1000 / 3600 -- to convert from km/hr to m/s

-- Prove that speed of the slower train is 36 km/hr
theorem speed_of_slower_train_is_36 :
  ∃ v : ℕ, v = 36 ∧ ((combined_length_of_trains : ℝ) = ((speed_of_faster_train - v) * conversion_factor * time_to_pass)) :=
sorry

end NUMINAMATH_GPT_speed_of_slower_train_is_36_l2112_211242


namespace NUMINAMATH_GPT_largest_integer_le_zero_l2112_211202

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero (x k : ℝ) (h1 : f x = 0) (h2 : 2 < x) (h3 : x < 3) : k ≤ x ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_le_zero_l2112_211202


namespace NUMINAMATH_GPT_students_sign_up_ways_l2112_211293

theorem students_sign_up_ways :
  let students := 4
  let choices_per_student := 3
  (choices_per_student ^ students) = 3^4 :=
by
  sorry

end NUMINAMATH_GPT_students_sign_up_ways_l2112_211293


namespace NUMINAMATH_GPT_division_number_l2112_211268

-- Definitions from conditions
def D : Nat := 3
def Q : Nat := 4
def R : Nat := 3

-- Theorem statement
theorem division_number : ∃ N : Nat, N = D * Q + R ∧ N = 15 :=
by
  sorry

end NUMINAMATH_GPT_division_number_l2112_211268


namespace NUMINAMATH_GPT_min_shots_to_hit_terrorist_l2112_211209

theorem min_shots_to_hit_terrorist : ∀ terrorist_position : ℕ, (1 ≤ terrorist_position ∧ terrorist_position ≤ 10) →
  ∃ shots : ℕ, shots ≥ 6 ∧ (∀ move : ℕ, (shots - move) ≥ 1 → (terrorist_position + move ≤ 10 → terrorist_position % 2 = move % 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_shots_to_hit_terrorist_l2112_211209


namespace NUMINAMATH_GPT_children_got_off_l2112_211265

theorem children_got_off {x : ℕ} 
  (initial_children : ℕ := 22)
  (children_got_on : ℕ := 40)
  (children_left : ℕ := 2)
  (equation : initial_children + children_got_on - x = children_left) :
  x = 60 :=
sorry

end NUMINAMATH_GPT_children_got_off_l2112_211265


namespace NUMINAMATH_GPT_linear_function_passing_origin_l2112_211258

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_passing_origin_l2112_211258


namespace NUMINAMATH_GPT_max_handshakes_25_people_l2112_211263

-- Define the number of people attending the conference.
def num_people : ℕ := 25

-- Define the combinatorial formula to calculate the maximum number of handshakes.
def max_handshakes (n : ℕ) : ℕ := n.choose 2

-- State the theorem that we need to prove.
theorem max_handshakes_25_people : max_handshakes num_people = 300 :=
by
  -- Proof will be filled in later
  sorry

end NUMINAMATH_GPT_max_handshakes_25_people_l2112_211263


namespace NUMINAMATH_GPT_number_of_books_in_shipment_l2112_211220

theorem number_of_books_in_shipment
  (T : ℕ)                   -- The total number of books
  (displayed_ratio : ℚ)     -- Fraction of books displayed
  (remaining_books : ℕ)     -- Number of books in the storeroom
  (h1 : displayed_ratio = 0.3)
  (h2 : remaining_books = 210)
  (h3 : (1 - displayed_ratio) * T = remaining_books) :
  T = 300 := 
by
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_number_of_books_in_shipment_l2112_211220


namespace NUMINAMATH_GPT_arcsin_double_angle_identity_l2112_211294

open Real

theorem arcsin_double_angle_identity (x θ : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (h₃ : arcsin x = θ) (h₄ : -π / 2 ≤ θ) (h₅ : θ ≤ -π / 4) :
    arcsin (2 * x * sqrt (1 - x^2)) = -(π + 2 * θ) := by
  sorry

end NUMINAMATH_GPT_arcsin_double_angle_identity_l2112_211294


namespace NUMINAMATH_GPT_intersection_point_in_AB_l2112_211269

def A (p : ℝ × ℝ) : Prop := p.snd = 2 * p.fst - 1
def B (p : ℝ × ℝ) : Prop := p.snd = p.fst + 3

theorem intersection_point_in_AB : (4, 7) ∈ {p : ℝ × ℝ | A p} ∩ {p : ℝ × ℝ | B p} :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_in_AB_l2112_211269


namespace NUMINAMATH_GPT_find_c_plus_d_l2112_211235

variables {a b c d : ℝ}

theorem find_c_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : a + d = 10) : c + d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l2112_211235


namespace NUMINAMATH_GPT_total_feet_is_correct_l2112_211245

-- definitions according to conditions
def number_of_heads := 46
def number_of_hens := 24
def number_of_cows := number_of_heads - number_of_hens
def hen_feet := 2
def cow_feet := 4
def total_hen_feet := number_of_hens * hen_feet
def total_cow_feet := number_of_cows * cow_feet
def total_feet := total_hen_feet + total_cow_feet

-- proof statement with sorry
theorem total_feet_is_correct : total_feet = 136 :=
by
  sorry

end NUMINAMATH_GPT_total_feet_is_correct_l2112_211245


namespace NUMINAMATH_GPT_compound_interest_interest_l2112_211228

theorem compound_interest_interest :
  let P := 2000
  let r := 0.05
  let n := 5
  let A := P * (1 + r)^n
  let interest := A - P
  interest = 552.56 := by
  sorry

end NUMINAMATH_GPT_compound_interest_interest_l2112_211228


namespace NUMINAMATH_GPT_union_of_M_and_N_l2112_211266

def M : Set ℝ := {x | x^2 - 6 * x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5 * x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by
  sorry

end NUMINAMATH_GPT_union_of_M_and_N_l2112_211266


namespace NUMINAMATH_GPT_avg_of_six_is_3_9_l2112_211207

noncomputable def avg_of_six_numbers 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : ℝ :=
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6

theorem avg_of_six_is_3_9 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : 
  avg_of_six_numbers avg1 avg2 avg3 h1 h2 h3 = 3.9 := 
by {
  sorry
}

end NUMINAMATH_GPT_avg_of_six_is_3_9_l2112_211207


namespace NUMINAMATH_GPT_negation_exists_to_forall_l2112_211248

theorem negation_exists_to_forall :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_to_forall_l2112_211248


namespace NUMINAMATH_GPT_triangle_inequality_l2112_211204

theorem triangle_inequality 
(a b c : ℝ) (α β γ : ℝ)
(h_t : a + b > c ∧ a + c > b ∧ b + c > a)
(h_opposite : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ α + β + γ = π) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l2112_211204


namespace NUMINAMATH_GPT_unique_ones_digits_divisible_by_8_l2112_211296

/-- Carla likes numbers that are divisible by 8.
    We want to show that there are 5 unique ones digits for such numbers. -/
theorem unique_ones_digits_divisible_by_8 : 
  (Finset.card 
    (Finset.image (fun n => n % 10) 
                  (Finset.filter (fun n => n % 8 = 0) (Finset.range 100)))) = 5 := 
by
  sorry

end NUMINAMATH_GPT_unique_ones_digits_divisible_by_8_l2112_211296


namespace NUMINAMATH_GPT_range_of_m_l2112_211282

noncomputable def A (x : ℝ) : ℝ := x^2 - (3/2) * x + 1

def in_interval (x : ℝ) : Prop := (3/4 ≤ x) ∧ (x ≤ 2)

def B (y : ℝ) (m : ℝ) : Prop := y ≥ 1 - m^2

theorem range_of_m (m : ℝ) :
  (∀ x, in_interval x → B (A x) m) ↔ (m ≤ - (3/4) ∨ m ≥ (3/4)) := 
sorry

end NUMINAMATH_GPT_range_of_m_l2112_211282


namespace NUMINAMATH_GPT_cost_of_first_20_kgs_l2112_211260

theorem cost_of_first_20_kgs 
  (l m n : ℕ) 
  (hl1 : 30 * l +  3 * m = 333) 
  (hl2 : 30 * l +  6 * m = 366) 
  (hl3 : 30 * l + 15 * m = 465) 
  (hl4 : 30 * l + 20 * m = 525) 
  : 20 * l = 200 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_first_20_kgs_l2112_211260


namespace NUMINAMATH_GPT_peanuts_in_box_l2112_211249

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (h1 : initial_peanuts = 4) (h2 : added_peanuts = 2) : initial_peanuts + added_peanuts = 6 := by
  sorry

end NUMINAMATH_GPT_peanuts_in_box_l2112_211249


namespace NUMINAMATH_GPT_interior_edges_sum_l2112_211244

theorem interior_edges_sum (frame_width area outer_length : ℝ) (h1 : frame_width = 2) (h2 : area = 30)
  (h3 : outer_length = 7) : 
  2 * (outer_length - 2 * frame_width) + 2 * ((area / outer_length - 4)) = 7 := 
by
  sorry

end NUMINAMATH_GPT_interior_edges_sum_l2112_211244


namespace NUMINAMATH_GPT_expected_interval_proof_l2112_211283

noncomputable def expected_interval_between_trains : ℝ := 3

theorem expected_interval_proof
  (northern_route_time southern_route_time : ℝ)
  (counter_clockwise_delay : ℝ)
  (home_to_work_less_than_work_to_home : ℝ) :
  northern_route_time = 17 →
  southern_route_time = 11 →
  counter_clockwise_delay = 75 / 60 →
  home_to_work_less_than_work_to_home = 1 →
  expected_interval_between_trains = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_expected_interval_proof_l2112_211283


namespace NUMINAMATH_GPT_find_positive_integer_solutions_l2112_211212

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_solutions_l2112_211212


namespace NUMINAMATH_GPT_schools_participating_l2112_211298

noncomputable def num_schools (students_per_school : ℕ) (total_students : ℕ) : ℕ :=
  total_students / students_per_school

theorem schools_participating (students_per_school : ℕ) (beth_rank : ℕ) 
  (carla_rank : ℕ) (highest_on_team : ℕ) (n : ℕ) :
  students_per_school = 4 ∧ beth_rank = 46 ∧ carla_rank = 79 ∧
  (∀ i, i ≤ 46 → highest_on_team = 40) → 
  num_schools students_per_school ((2 * highest_on_team) - 1) = 19 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_schools_participating_l2112_211298


namespace NUMINAMATH_GPT_travel_time_is_correct_l2112_211206

-- Define the conditions
def speed : ℕ := 60 -- Speed in km/h
def distance : ℕ := 120 -- Distance between points A and B in km

-- Time calculation from A to B 
def time_AB : ℕ := distance / speed

-- Time calculation from B to A (since speed and distance are the same)
def time_BA : ℕ := distance / speed

-- Total time calculation
def total_time : ℕ := time_AB + time_BA

-- The proper statement to prove
theorem travel_time_is_correct : total_time = 4 := by
  -- Additional steps and arguments would go here
  -- skipping proof
  sorry

end NUMINAMATH_GPT_travel_time_is_correct_l2112_211206


namespace NUMINAMATH_GPT_total_number_of_students_l2112_211224

-- Statement translating the problem conditions and conclusion
theorem total_number_of_students (rank_from_right rank_from_left total : ℕ) 
  (h_right : rank_from_right = 13) 
  (h_left : rank_from_left = 8) 
  (total_eq : total = rank_from_right + rank_from_left - 1) : 
  total = 20 := 
by 
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_total_number_of_students_l2112_211224


namespace NUMINAMATH_GPT_probability_of_pulling_blue_ball_l2112_211290

def given_conditions (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ) :=
  total_balls = 15 ∧ initial_blue_balls = 7 ∧ blue_balls_removed = 3

theorem probability_of_pulling_blue_ball
  (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ)
  (hc : given_conditions total_balls initial_blue_balls blue_balls_removed) :
  ((initial_blue_balls - blue_balls_removed) / (total_balls - blue_balls_removed) : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_pulling_blue_ball_l2112_211290


namespace NUMINAMATH_GPT_investigate_local_extrema_l2112_211230

noncomputable def f (x1 x2 : ℝ) : ℝ :=
  3 * x1^2 * x2 - x1^3 - (4 / 3) * x2^3

def is_local_maximum (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∀ (x y : ℝ × ℝ), dist x c < ε → f x.1 x.2 ≤ f c.1 c.2

def is_saddle_point (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∃ (x1 y1 x2 y2 : ℝ × ℝ),
    dist x1 c < ε ∧ dist y1 c < ε ∧ dist x2 c < ε ∧ dist y2 c < ε ∧
    (f x1.1 x1.2 > f c.1 c.2 ∧ f y1.1 y1.2 < f c.1 c.2) ∧
    (f x2.1 x2.2 < f c.1 c.2 ∧ f y2.1 y2.2 > f c.1 c.2)

theorem investigate_local_extrema :
  is_local_maximum f (6, 3) ∧ is_saddle_point f (0, 0) :=
sorry

end NUMINAMATH_GPT_investigate_local_extrema_l2112_211230


namespace NUMINAMATH_GPT_sin_double_angle_l2112_211241

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_sin_double_angle_l2112_211241


namespace NUMINAMATH_GPT_one_belt_one_road_l2112_211208

theorem one_belt_one_road (m n : ℝ) :
  (∀ x y : ℝ, y = x^2 - 2 * x + n ↔ (x, y) ∈ { p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 1 }) →
  (∀ x y : ℝ, y = m * x + 1 ↔ (x, y) ∈ { q : ℝ × ℝ | q.1 = 0 ∧ q.2 = 1 }) →
  (∀ x y : ℝ, y = x^2 - 2 * x + 1 → y = 0) →
  m = -1 ∧ n = 1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_one_belt_one_road_l2112_211208


namespace NUMINAMATH_GPT_problem_maximum_marks_l2112_211243

theorem problem_maximum_marks (M : ℝ) (h : 0.92 * M = 184) : M = 200 :=
sorry

end NUMINAMATH_GPT_problem_maximum_marks_l2112_211243


namespace NUMINAMATH_GPT_min_points_to_guarantee_win_l2112_211289

theorem min_points_to_guarantee_win (P Q R S: ℕ) (bonus: ℕ) :
    (P = 6 ∨ P = 4 ∨ P = 2) ∧ (Q = 6 ∨ Q = 4 ∨ Q = 2) ∧ 
    (R = 6 ∨ R = 4 ∨ R = 2) ∧ (S = 6 ∨ S = 4 ∨ S = 2) →
    (bonus = 3 ↔ ((P = 6 ∧ Q = 4 ∧ R = 2) ∨ (P = 6 ∧ Q = 2 ∧ R = 4) ∨ 
                   (P = 4 ∧ Q = 6 ∧ R = 2) ∨ (P = 4 ∧ Q = 2 ∧ R = 6) ∨ 
                   (P = 2 ∧ Q = 6 ∧ R = 4) ∨ (P = 2 ∧ Q = 4 ∧ R = 6))) →
    (P + Q + R + S + bonus ≥ 24) :=
by sorry

end NUMINAMATH_GPT_min_points_to_guarantee_win_l2112_211289
