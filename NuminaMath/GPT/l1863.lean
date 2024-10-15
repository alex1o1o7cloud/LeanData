import Mathlib

namespace NUMINAMATH_GPT_percent_absent_of_students_l1863_186371

theorem percent_absent_of_students
  (boys girls : ℕ)
  (total_students := boys + girls)
  (boys_absent_fraction girls_absent_fraction : ℚ)
  (boys_absent_fraction_eq : boys_absent_fraction = 1 / 8)
  (girls_absent_fraction_eq : girls_absent_fraction = 1 / 4)
  (total_students_eq : total_students = 160)
  (boys_eq : boys = 80)
  (girls_eq : girls = 80) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_percent_absent_of_students_l1863_186371


namespace NUMINAMATH_GPT_sum_of_numbers_with_lcm_and_ratio_l1863_186395

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 48)
  (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : 
  a + b = 80 := 
by sorry

end NUMINAMATH_GPT_sum_of_numbers_with_lcm_and_ratio_l1863_186395


namespace NUMINAMATH_GPT_xiao_ding_distance_l1863_186337

variable (x y z w : ℕ)

theorem xiao_ding_distance (h1 : x = 4 * y)
                          (h2 : z = x / 2 + 20)
                          (h3 : w = 2 * z - 15)
                          (h4 : x + y + z + w = 705) : 
                          y = 60 := 
sorry

end NUMINAMATH_GPT_xiao_ding_distance_l1863_186337


namespace NUMINAMATH_GPT_domain_of_sqrt_tan_l1863_186393

theorem domain_of_sqrt_tan :
  ∀ x : ℝ, (∃ k : ℤ, k * π ≤ x ∧ x < k * π + π / 2) ↔ 0 ≤ (Real.tan x) :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_tan_l1863_186393


namespace NUMINAMATH_GPT_lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l1863_186326

-- Define a cube with edge length a
structure Cube :=
  (a : ℝ) -- Edge length of the cube

-- Define a pyramid with a given height
structure Pyramid :=
  (h : ℝ) -- Height of the pyramid

-- The main theorem statement for part 4A
theorem lateral_edges_in_same_plane (c : Cube) (p : Pyramid) : p.h = c.a ↔ (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
  O1 = (c.a / 2, c.a / 2, -p.h) ∧
  O2 = (c.a / 2, -p.h, c.a / 2) ∧
  O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

-- The main theorem statement for part 4B
theorem edges_in_planes_for_all_vertices (c : Cube) (p : Pyramid) : p.h = c.a ↔ ∀ (v : ℝ × ℝ × ℝ), -- Iterate over cube vertices
  (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
    O1 = (c.a / 2, c.a / 2, -p.h) ∧
    O2 = (c.a / 2, -p.h, c.a / 2) ∧
    O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

end NUMINAMATH_GPT_lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l1863_186326


namespace NUMINAMATH_GPT_cody_initial_tickets_l1863_186379

theorem cody_initial_tickets (T : ℕ) (h1 : T - 25 + 6 = 30) : T = 49 :=
sorry

end NUMINAMATH_GPT_cody_initial_tickets_l1863_186379


namespace NUMINAMATH_GPT_one_fourths_in_seven_halves_l1863_186341

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end NUMINAMATH_GPT_one_fourths_in_seven_halves_l1863_186341


namespace NUMINAMATH_GPT_translated_parabola_eq_l1863_186308

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Function to translate a parabola equation downward by a units
def translate_downward (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Function to translate a parabola equation rightward by b units
def translate_rightward (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f (x - b)

-- The new parabola equation after translating the given parabola downward by 3 units and rightward by 2 units
def new_parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 9

-- The main theorem stating that translating the original parabola downward by 3 units and rightward by 2 units results in the new parabola equation
theorem translated_parabola_eq :
  ∀ x : ℝ, translate_rightward (translate_downward original_parabola 3) 2 x = new_parabola x :=
by
  sorry

end NUMINAMATH_GPT_translated_parabola_eq_l1863_186308


namespace NUMINAMATH_GPT_volume_of_wall_is_16128_l1863_186394

def wall_width : ℝ := 4
def wall_height : ℝ := 6 * wall_width
def wall_length : ℝ := 7 * wall_height

def wall_volume : ℝ := wall_length * wall_width * wall_height

theorem volume_of_wall_is_16128 :
  wall_volume = 16128 := by
  sorry

end NUMINAMATH_GPT_volume_of_wall_is_16128_l1863_186394


namespace NUMINAMATH_GPT_no_fixed_point_range_of_a_fixed_point_in_interval_l1863_186313

-- Problem (1)
theorem no_fixed_point_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a ≠ x) →
  3 - 2 * Real.sqrt 2 < a ∧ a < 3 + 2 * Real.sqrt 2 :=
by
  sorry

-- Problem (2)
theorem fixed_point_in_interval (f : ℝ → ℝ) (n : ℤ) :
  (∀ x : ℝ, f x = -Real.log x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ n ≤ x₀ ∧ x₀ < n + 1) →
  n = 2 :=
by
  sorry

end NUMINAMATH_GPT_no_fixed_point_range_of_a_fixed_point_in_interval_l1863_186313


namespace NUMINAMATH_GPT_toms_total_profit_l1863_186335

def total_earnings_mowing : ℕ := 4 * 12 + 3 * 15 + 1 * 20
def total_earnings_side_jobs : ℕ := 2 * 10 + 3 * 8 + 1 * 12
def total_earnings : ℕ := total_earnings_mowing + total_earnings_side_jobs
def total_expenses : ℕ := 17 + 5
def total_profit : ℕ := total_earnings - total_expenses

theorem toms_total_profit : total_profit = 147 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_toms_total_profit_l1863_186335


namespace NUMINAMATH_GPT_profit_function_maximize_profit_l1863_186304

def cost_per_item : ℝ := 80
def purchase_quantity : ℝ := 1000
def selling_price_initial : ℝ := 100
def price_increase_per_item : ℝ := 1
def sales_decrease_per_yuan : ℝ := 10
def selling_price (x : ℕ) : ℝ := selling_price_initial + x
def profit (x : ℕ) : ℝ := (selling_price x - cost_per_item) * (purchase_quantity - sales_decrease_per_yuan * x)

theorem profit_function (x : ℕ) (h : 0 ≤ x ∧ x ≤ 100) : 
  profit x = -10 * (x : ℝ)^2 + 800 * (x : ℝ) + 20000 :=
by sorry

theorem maximize_profit :
  ∃ max_x, (0 ≤ max_x ∧ max_x ≤ 100) ∧ 
  (∀ x : ℕ, (0 ≤ x ∧ x ≤ 100) → profit x ≤ profit max_x) ∧ 
  max_x = 40 ∧ 
  profit max_x = 36000 :=
by sorry

end NUMINAMATH_GPT_profit_function_maximize_profit_l1863_186304


namespace NUMINAMATH_GPT_cube_side_length_l1863_186342

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = 1 / 4 * 6 * n^3) : n = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cube_side_length_l1863_186342


namespace NUMINAMATH_GPT_find_intersection_point_l1863_186345

/-- Definition of the parabola -/
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 - 4 * y + 7

/-- Condition for intersection at exactly one point -/
def discriminant (m : ℝ) : ℝ := 4 ^ 2 - 4 * 3 * (m - 7)

/-- Main theorem stating the proof problem -/
theorem find_intersection_point (m : ℝ) :
  (discriminant m = 0) → m = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_point_l1863_186345


namespace NUMINAMATH_GPT_induction_step_l1863_186306

theorem induction_step
  (x y : ℝ)
  (k : ℕ)
  (base : ∀ n, ∃ m, (n = 2 * m - 1) → (x^n + y^n) = (x + y) * m) :
  (x^(2 * k + 1) + y^(2 * k + 1)) = (x + y) * (k + 1) :=
by
  sorry

end NUMINAMATH_GPT_induction_step_l1863_186306


namespace NUMINAMATH_GPT_sum_of_B_coordinates_l1863_186344

theorem sum_of_B_coordinates 
  (x y : ℝ) 
  (A : ℝ × ℝ) 
  (M : ℝ × ℝ)
  (midpoint_x : (A.1 + x) / 2 = M.1) 
  (midpoint_y : (A.2 + y) / 2 = M.2) 
  (A_conds : A = (7, -1))
  (M_conds : M = (4, 3)) :
  x + y = 8 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_B_coordinates_l1863_186344


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1863_186358

theorem simplify_and_evaluate : 
    ∀ (a b : ℤ), a = 1 → b = -1 → 
    ((2 * a^2 * b - 2 * a * b^2 - b^3) / b - (a + b) * (a - b) = 3) := 
by
  intros a b ha hb
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1863_186358


namespace NUMINAMATH_GPT_ending_number_divisible_by_six_l1863_186321

theorem ending_number_divisible_by_six (first_term : ℕ) (n : ℕ) (common_difference : ℕ) (sequence_length : ℕ) 
  (start : first_term = 12) 
  (diff : common_difference = 6)
  (num_terms : sequence_length = 11) :
  first_term + (sequence_length - 1) * common_difference = 72 := by
  sorry

end NUMINAMATH_GPT_ending_number_divisible_by_six_l1863_186321


namespace NUMINAMATH_GPT_remainder_mod7_l1863_186315

theorem remainder_mod7 (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := 
by
  sorry

end NUMINAMATH_GPT_remainder_mod7_l1863_186315


namespace NUMINAMATH_GPT_ranking_of_scores_l1863_186310

-- Let the scores of Ann, Bill, Carol, and Dick be A, B, C, and D respectively.

variables (A B C D : ℝ)

-- Conditions
axiom cond1 : B + D = A + C
axiom cond2 : C + B > D + A
axiom cond3 : C > A + B

-- Statement of the problem
theorem ranking_of_scores : C > D ∧ D > B ∧ B > A :=
by
  -- Placeholder for proof (proof steps aren't required)
  sorry

end NUMINAMATH_GPT_ranking_of_scores_l1863_186310


namespace NUMINAMATH_GPT_picture_area_l1863_186317

theorem picture_area (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (3 * x + 4) * (y + 3) - x * y = 54 → x * y = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_picture_area_l1863_186317


namespace NUMINAMATH_GPT_pyramid_cross_section_distance_l1863_186311

theorem pyramid_cross_section_distance 
  (A1 A2 : ℝ) (d : ℝ) (h : ℝ) 
  (hA1 : A1 = 125 * Real.sqrt 3)
  (hA2 : A2 = 500 * Real.sqrt 3)
  (hd : d = 12) :
  h = 24 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_cross_section_distance_l1863_186311


namespace NUMINAMATH_GPT_max_ab_value_l1863_186338

theorem max_ab_value {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : 6 * a + 8 * b = 72) : ab = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_ab_value_l1863_186338


namespace NUMINAMATH_GPT_sum_of_cubes_l1863_186334

theorem sum_of_cubes {x y : ℝ} (h₁ : x + y = 0) (h₂ : x * y = -1) : x^3 + y^3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1863_186334


namespace NUMINAMATH_GPT_set_intersection_complement_l1863_186332

/-- Definition of the universal set U. -/
def U := ({1, 2, 3, 4, 5} : Set ℕ)

/-- Definition of the set M. -/
def M := ({3, 4, 5} : Set ℕ)

/-- Definition of the set N. -/
def N := ({2, 3} : Set ℕ)

/-- Statement of the problem to be proven. -/
theorem set_intersection_complement :
  ((U \ N) ∩ M) = ({4, 5} : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1863_186332


namespace NUMINAMATH_GPT_smallest_population_multiple_of_3_l1863_186340

theorem smallest_population_multiple_of_3 : 
  ∃ (a : ℕ), ∃ (b c : ℕ), 
  a^2 + 50 = b^2 + 1 ∧ b^2 + 51 = c^2 ∧ 
  (∃ m : ℕ, a * a = 576 ∧ 576 = 3 * m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_population_multiple_of_3_l1863_186340


namespace NUMINAMATH_GPT_equation1_solutions_equation2_solutions_l1863_186333

theorem equation1_solutions (x : ℝ) : 3 * x^2 - 6 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

theorem equation2_solutions (x : ℝ) : x^2 + 4 * x - 1 = 0 ↔ (x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_equation1_solutions_equation2_solutions_l1863_186333


namespace NUMINAMATH_GPT_halfway_fraction_l1863_186302

theorem halfway_fraction (a b : ℚ) (h1 : a = 2 / 9) (h2 : b = 1 / 3) :
  (a + b) / 2 = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_halfway_fraction_l1863_186302


namespace NUMINAMATH_GPT_math_problem_l1863_186331

noncomputable def a : ℝ := 3.67
noncomputable def b : ℝ := 4.83
noncomputable def c : ℝ := 2.57
noncomputable def d : ℝ := -0.12
noncomputable def x : ℝ := 7.25
noncomputable def y : ℝ := -0.55

theorem math_problem :
  (3 * a * (4 * b - 2 * y)^2) / (5 * c * d^3 * 0.5 * x) - (2 * x * y^3) / (a * b^2 * c) = -57.179729 := 
sorry

end NUMINAMATH_GPT_math_problem_l1863_186331


namespace NUMINAMATH_GPT_area_of_intersection_l1863_186365

-- Define the region M
def in_region_M (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

-- Define the region N as it changes with t
def in_region_N (t x : ℝ) : Prop :=
  t ≤ x ∧ x ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1

-- Define the function f(t) which represents the common area of M and N
noncomputable def f (t : ℝ) : ℝ :=
  -t^2 + t + 0.5

-- Prove that f(t) is correct given the above conditions
theorem area_of_intersection (t : ℝ) :
  (∀ x y : ℝ, in_region_M x y → in_region_N t x → y ≤ f t) →
  0 ≤ t ∧ t ≤ 1 →
  f t = -t^2 + t + 0.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_intersection_l1863_186365


namespace NUMINAMATH_GPT_bungee_cord_extension_l1863_186364

variables (m g H k h L₀ T_max : ℝ)
  (mass_nonzero : m ≠ 0)
  (gravity_positive : g > 0)
  (H_positive : H > 0)
  (k_positive : k > 0)
  (L₀_nonnegative : L₀ ≥ 0)
  (T_max_eq : T_max = 4 * m * g)
  (L_eq : L₀ + h = H)
  (hooke_eq : T_max = k * h)

theorem bungee_cord_extension :
  h = H / 2 := sorry

end NUMINAMATH_GPT_bungee_cord_extension_l1863_186364


namespace NUMINAMATH_GPT_three_digit_numbers_divisible_by_13_count_l1863_186388

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_divisible_by_13_count_l1863_186388


namespace NUMINAMATH_GPT_magnitude_of_z_l1863_186351

open Complex

noncomputable def z : ℂ := (1 - I) / (1 + I) + 2 * I

theorem magnitude_of_z : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l1863_186351


namespace NUMINAMATH_GPT_max_additional_pies_l1863_186328

theorem max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) 
  (h₀ : initial_cherries = 500) 
  (h₁ : used_cherries = 350) 
  (h₂ : cherries_per_pie = 35) :
  (initial_cherries - used_cherries) / cherries_per_pie = 4 := 
by
  sorry

end NUMINAMATH_GPT_max_additional_pies_l1863_186328


namespace NUMINAMATH_GPT_Nicki_total_miles_run_l1863_186314

theorem Nicki_total_miles_run:
  ∀ (miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year : ℕ),
  miles_per_week_first_half = 20 →
  miles_per_week_second_half = 30 →
  weeks_in_year = 52 →
  weeks_per_half_year = weeks_in_year / 2 →
  (miles_per_week_first_half * weeks_per_half_year) + (miles_per_week_second_half * weeks_per_half_year) = 1300 :=
by
  intros miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Nicki_total_miles_run_l1863_186314


namespace NUMINAMATH_GPT_number_of_ordered_triples_l1863_186380

theorem number_of_ordered_triples (x y z : ℝ) (hx : x + y = 3) (hy : xy - z^2 = 4)
  (hnn : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) : 
  ∃! (x y z : ℝ), (x + y = 3) ∧ (xy - z^2 = 4) ∧ (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) :=
sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l1863_186380


namespace NUMINAMATH_GPT_convex_polygons_from_fifteen_points_l1863_186392

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end NUMINAMATH_GPT_convex_polygons_from_fifteen_points_l1863_186392


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1863_186319

theorem geometric_sequence_seventh_term :
  let a := 6
  let r := -2
  (a * r^(7 - 1)) = 384 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1863_186319


namespace NUMINAMATH_GPT_beats_per_week_l1863_186391

def beats_per_minute : ℕ := 200
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7

theorem beats_per_week : beats_per_minute * minutes_per_hour * hours_per_day * days_per_week = 168000 := by
  sorry

end NUMINAMATH_GPT_beats_per_week_l1863_186391


namespace NUMINAMATH_GPT_cylinder_base_radius_l1863_186382

theorem cylinder_base_radius (a : ℝ) (h_a_pos : 0 < a) :
  ∃ (R : ℝ), R = 7 * a * Real.sqrt 3 / 24 := 
    sorry

end NUMINAMATH_GPT_cylinder_base_radius_l1863_186382


namespace NUMINAMATH_GPT_fraction_division_result_l1863_186359

theorem fraction_division_result :
  (5/6) / (-9/10) = -25/27 := 
by
  sorry

end NUMINAMATH_GPT_fraction_division_result_l1863_186359


namespace NUMINAMATH_GPT_factor_theorem_l1863_186301

theorem factor_theorem (t : ℝ) : (5 * t^2 + 15 * t - 20 = 0) ↔ (t = 1 ∨ t = -4) :=
by
  sorry

end NUMINAMATH_GPT_factor_theorem_l1863_186301


namespace NUMINAMATH_GPT_balloon_permutations_l1863_186381

theorem balloon_permutations : 
  (Nat.factorial 7 / 
  ((Nat.factorial 1) * 
  (Nat.factorial 1) * 
  (Nat.factorial 2) * 
  (Nat.factorial 2) * 
  (Nat.factorial 1))) = 1260 := by
  sorry

end NUMINAMATH_GPT_balloon_permutations_l1863_186381


namespace NUMINAMATH_GPT_max_possible_value_e_l1863_186339

def b (n : ℕ) : ℕ := (7^n - 1) / 6

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_possible_value_e (n : ℕ) : e n = 1 := by
  sorry

end NUMINAMATH_GPT_max_possible_value_e_l1863_186339


namespace NUMINAMATH_GPT_min_value_x_plus_y_l1863_186385

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 :=
  sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l1863_186385


namespace NUMINAMATH_GPT_stapler_problem_l1863_186357

noncomputable def staplesLeft (initial_staples : ℕ) (dozens : ℕ) (staples_per_report : ℝ) : ℝ :=
  initial_staples - (dozens * 12) * staples_per_report

theorem stapler_problem : staplesLeft 200 7 0.75 = 137 := 
by
  sorry

end NUMINAMATH_GPT_stapler_problem_l1863_186357


namespace NUMINAMATH_GPT_distance_parallel_lines_distance_point_line_l1863_186343

def line1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def point : ℝ × ℝ := (0, 2)

noncomputable def distance_between_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem distance_parallel_lines : distance_between_lines 2 1 (-1) 1 = (2 * Real.sqrt 5) / 5 := by
  sorry

theorem distance_point_line : distance_point_to_line 2 1 (-1) 0 2 = (Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_GPT_distance_parallel_lines_distance_point_line_l1863_186343


namespace NUMINAMATH_GPT_find_alpha_l1863_186347

theorem find_alpha (α : ℝ) :
    7 * α + 8 * α + 45 = 180 →
    α = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_l1863_186347


namespace NUMINAMATH_GPT_solve_fractional_equation_l1863_186361

-- Define the fractional equation as a function
def fractional_equation (x : ℝ) : Prop :=
  (3 / 2) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2)

-- State the theorem we need to prove
theorem solve_fractional_equation : fractional_equation 2 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1863_186361


namespace NUMINAMATH_GPT_inequality_of_sum_l1863_186353

theorem inequality_of_sum 
  (a : ℕ → ℝ)
  (h : ∀ n m, 0 ≤ n → n < m → a n < a m) :
  (0 < a 1 ->
  0 < a 2 ->
  0 < a 3 ->
  0 < a 4 ->
  0 < a 5 ->
  0 < a 6 ->
  0 < a 7 ->
  0 < a 8 ->
  0 < a 9 ->
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / (a 3 + a 6 + a 9) < 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_inequality_of_sum_l1863_186353


namespace NUMINAMATH_GPT_range_of_k_l1863_186397

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l1863_186397


namespace NUMINAMATH_GPT_john_started_5_days_ago_l1863_186336

noncomputable def daily_wage (x : ℕ) : Prop := 250 + 10 * x = 750

theorem john_started_5_days_ago :
  ∃ x : ℕ, daily_wage x ∧ 250 / x = 5 :=
by
  sorry

end NUMINAMATH_GPT_john_started_5_days_ago_l1863_186336


namespace NUMINAMATH_GPT_triangle_base_l1863_186356

noncomputable def side_length_square (p : ℕ) : ℕ := p / 4

noncomputable def area_square (s : ℕ) : ℕ := s * s

noncomputable def area_triangle (h b : ℕ) : ℕ := (h * b) / 2

theorem triangle_base (p h a b : ℕ) (hp : p = 80) (hh : h = 40) (ha : a = (side_length_square p)^2) (eq_areas : area_square (side_length_square p) = area_triangle h b) : b = 20 :=
by {
  -- Here goes the proof which we are omitting
  sorry
}

end NUMINAMATH_GPT_triangle_base_l1863_186356


namespace NUMINAMATH_GPT_fewer_seats_on_right_than_left_l1863_186384

theorem fewer_seats_on_right_than_left : 
  ∀ (left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats : ℕ),
    left_seats = 15 →
    back_seat_capacity = 9 →
    people_per_seat = 3 →
    bus_capacity = 90 →
    right_seats = (bus_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat →
    fewer_seats = left_seats - right_seats →
    fewer_seats = 3 :=
by
  intros left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats
  sorry

end NUMINAMATH_GPT_fewer_seats_on_right_than_left_l1863_186384


namespace NUMINAMATH_GPT_number_added_is_minus_168_l1863_186368

theorem number_added_is_minus_168 (N : ℕ) (X : ℤ) (h1 : N = 180)
  (h2 : N + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = (1/15 : ℚ) * N) : X = -168 :=
by
  sorry

end NUMINAMATH_GPT_number_added_is_minus_168_l1863_186368


namespace NUMINAMATH_GPT_TotalGenuineItems_l1863_186367

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end NUMINAMATH_GPT_TotalGenuineItems_l1863_186367


namespace NUMINAMATH_GPT_fraction_of_juniors_l1863_186318

theorem fraction_of_juniors (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h : 1 / 2 * J = 2 / 3 * S) : J / (J + S) = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_juniors_l1863_186318


namespace NUMINAMATH_GPT_mart_income_percentage_of_juan_l1863_186354

theorem mart_income_percentage_of_juan
  (J T M : ℝ)
  (h1 : T = 0.60 * J)
  (h2 : M = 1.60 * T) :
  M = 0.96 * J :=
by 
  sorry

end NUMINAMATH_GPT_mart_income_percentage_of_juan_l1863_186354


namespace NUMINAMATH_GPT_tom_finishes_in_four_hours_l1863_186305

noncomputable def maryMowingRate := 1 / 3
noncomputable def tomMowingRate := 1 / 6
noncomputable def timeMaryMows := 1
noncomputable def remainingLawn := 1 - (timeMaryMows * maryMowingRate)

theorem tom_finishes_in_four_hours :
  remainingLawn / tomMowingRate = 4 :=
by sorry

end NUMINAMATH_GPT_tom_finishes_in_four_hours_l1863_186305


namespace NUMINAMATH_GPT_mary_starting_weight_l1863_186346

def initial_weight (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ) : ℕ :=
  final_weight + (lost_3 - gained_4) + (gained_2 - lost_1) + lost_1

theorem mary_starting_weight :
  ∀ (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ),
  final_weight = 81 →
  lost_1 = 12 →
  gained_2 = 2 * lost_1 →
  lost_3 = 3 * lost_1 →
  gained_4 = lost_1 / 2 →
  initial_weight final_weight lost_1 gained_2 lost_3 gained_4 = 99 :=
by
  intros final_weight lost_1 gained_2 lost_3 gained_4 h_final_weight h_lost_1 h_gained_2 h_lost_3 h_gained_4
  rw [h_final_weight, h_lost_1] at *
  rw [h_gained_2, h_lost_3, h_gained_4]
  unfold initial_weight
  sorry

end NUMINAMATH_GPT_mary_starting_weight_l1863_186346


namespace NUMINAMATH_GPT_swimming_pool_width_l1863_186324

theorem swimming_pool_width (length width vol depth : ℝ) 
  (H_length : length = 60) 
  (H_depth : depth = 0.5) 
  (H_vol_removal : vol = 2250 / 7.48052) 
  (H_vol_eq : vol = (length * width) * depth) : 
  width = 10.019 :=
by
  -- Assuming the correctness of floating-point arithmetic for the purpose of this example
  sorry

end NUMINAMATH_GPT_swimming_pool_width_l1863_186324


namespace NUMINAMATH_GPT_rick_total_clothes_ironed_l1863_186322

def rick_ironing_pieces
  (shirts_per_hour : ℕ)
  (pants_per_hour : ℕ)
  (hours_shirts : ℕ)
  (hours_pants : ℕ) : ℕ :=
  (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants)

theorem rick_total_clothes_ironed :
  rick_ironing_pieces 4 3 3 5 = 27 :=
by
  sorry

end NUMINAMATH_GPT_rick_total_clothes_ironed_l1863_186322


namespace NUMINAMATH_GPT_pen_distribution_l1863_186330

theorem pen_distribution:
  (∃ (fountain: ℕ) (ballpoint: ℕ), fountain = 2 ∧ ballpoint = 3) ∧
  (∃ (students: ℕ), students = 4) →
  (∀ (s: ℕ), s ≥ 1 → s ≤ 4) →
  ∃ (ways: ℕ), ways = 28 :=
by
  sorry

end NUMINAMATH_GPT_pen_distribution_l1863_186330


namespace NUMINAMATH_GPT_total_potatoes_l1863_186362

theorem total_potatoes (Nancy_potatoes : ℕ) (Sandy_potatoes : ℕ) (Andy_potatoes : ℕ) 
  (h1 : Nancy_potatoes = 6) (h2 : Sandy_potatoes = 7) (h3 : Andy_potatoes = 9) : 
  Nancy_potatoes + Sandy_potatoes + Andy_potatoes = 22 :=
by
  -- The proof can be written here
  sorry

end NUMINAMATH_GPT_total_potatoes_l1863_186362


namespace NUMINAMATH_GPT_general_term_formula_l1863_186360

theorem general_term_formula :
  ∀ n : ℕ, (0 < n) → 
  (-1)^n * (2*n + 1) / (2*n) = ((-1) : ℝ)^n * ((2*n + 1) : ℝ) / (2*n) :=
by {
  sorry
}

end NUMINAMATH_GPT_general_term_formula_l1863_186360


namespace NUMINAMATH_GPT_cost_of_8_dozen_oranges_l1863_186352

noncomputable def cost_per_dozen (cost_5_dozen : ℝ) : ℝ :=
  cost_5_dozen / 5

noncomputable def cost_8_dozen (cost_5_dozen : ℝ) : ℝ :=
  8 * cost_per_dozen cost_5_dozen

theorem cost_of_8_dozen_oranges (cost_5_dozen : ℝ) (h : cost_5_dozen = 39) : cost_8_dozen cost_5_dozen = 62.4 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_8_dozen_oranges_l1863_186352


namespace NUMINAMATH_GPT_solve_equation_l1863_186363

theorem solve_equation (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (x^2 + x + 1 = 1 / (x^2 - x + 1)) ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1863_186363


namespace NUMINAMATH_GPT_find_side_length_of_square_l1863_186329

variable (a : ℝ)

theorem find_side_length_of_square (h1 : a - 3 > 0)
                                   (h2 : 3 * a + 5 * (a - 3) = 57) :
  a = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_side_length_of_square_l1863_186329


namespace NUMINAMATH_GPT_s_neq_t_if_Q_on_DE_l1863_186327

-- Conditions and Definitions
noncomputable def DQ (x : ℝ) := x
noncomputable def QE (x : ℝ) := 10 - x
noncomputable def FQ := 5 * Real.sqrt 3
noncomputable def s (x : ℝ) := (DQ x) ^ 2 + (QE x) ^ 2
noncomputable def t := 2 * FQ ^ 2

-- Lean 4 Statement
theorem s_neq_t_if_Q_on_DE (x : ℝ) : s x ≠ t :=
by
  sorry -- Provided proof step to be filled in

end NUMINAMATH_GPT_s_neq_t_if_Q_on_DE_l1863_186327


namespace NUMINAMATH_GPT_sum_evaluation_l1863_186378

noncomputable def T : ℝ := ∑' k : ℕ, (2*k+1) / 5^(k+1)

theorem sum_evaluation : T = 5 / 16 := sorry

end NUMINAMATH_GPT_sum_evaluation_l1863_186378


namespace NUMINAMATH_GPT_find_coordinates_of_b_l1863_186383

theorem find_coordinates_of_b
  (x y : ℝ)
  (a : ℂ) (b : ℂ)
  (sqrt3 sqrt5 sqrt10 sqrt6 : ℝ)
  (h1 : sqrt3 = Real.sqrt 3)
  (h2 : sqrt5 = Real.sqrt 5)
  (h3 : sqrt10 = Real.sqrt 10)
  (h4 : sqrt6 = Real.sqrt 6)
  (h5 : a = ⟨sqrt3, sqrt5⟩)
  (h6 : ∃ x y : ℝ, b = ⟨x, y⟩ ∧ (sqrt3 * x + sqrt5 * y = 0) ∧ (Real.sqrt (x^2 + y^2) = 2))
  : b = ⟨- sqrt10 / 2, sqrt6 / 2⟩ ∨ b = ⟨sqrt10 / 2, - sqrt6 / 2⟩ := 
  sorry

end NUMINAMATH_GPT_find_coordinates_of_b_l1863_186383


namespace NUMINAMATH_GPT_num_boys_and_girls_l1863_186389

def num_ways_to_select (x : ℕ) := (x * (x - 1) / 2) * (8 - x) * 6

theorem num_boys_and_girls (x : ℕ) (h1 : num_ways_to_select x = 180) :
    x = 5 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_boys_and_girls_l1863_186389


namespace NUMINAMATH_GPT_dog_rabbit_age_ratio_l1863_186377

-- Definitions based on conditions
def cat_age := 8
def rabbit_age := cat_age / 2
def dog_age := 12
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

-- Theorem statement
theorem dog_rabbit_age_ratio : is_multiple dog_age rabbit_age ∧ dog_age / rabbit_age = 3 :=
by
  sorry

end NUMINAMATH_GPT_dog_rabbit_age_ratio_l1863_186377


namespace NUMINAMATH_GPT_Amanda_car_round_trip_time_l1863_186370

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end NUMINAMATH_GPT_Amanda_car_round_trip_time_l1863_186370


namespace NUMINAMATH_GPT_vegetable_options_l1863_186307

open Nat

theorem vegetable_options (V : ℕ) : 
  3 * V + 6 = 57 → V = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_vegetable_options_l1863_186307


namespace NUMINAMATH_GPT_sum_of_first_seven_terms_l1863_186375

variable {a_n : ℕ → ℝ} {d : ℝ}

-- Define the arithmetic progression condition.
def arithmetic_progression (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n n = a_n 0 + n * d

-- We are given that the sequence is an arithmetic progression.
axiom sequence_is_arithmetic_progression : arithmetic_progression a_n d

-- We are also given that the sum of the 3rd, 4th, and 5th terms is 12.
axiom sum_of_terms_is_12 : a_n 2 + a_n 3 + a_n 4 = 12

-- We need to prove that the sum of the first seven terms is 28.
theorem sum_of_first_seven_terms : (a_n 0) + (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) = 28 := 
  sorry

end NUMINAMATH_GPT_sum_of_first_seven_terms_l1863_186375


namespace NUMINAMATH_GPT_minimum_buses_required_l1863_186366

-- Condition definitions
def one_way_trip_time : ℕ := 50
def stop_time : ℕ := 10
def departure_interval : ℕ := 6

-- Total round trip time
def total_round_trip_time : ℕ := 2 * one_way_trip_time + 2 * stop_time

-- The total number of buses needed to ensure the bus departs every departure_interval minutes
-- from both stations A and B.
theorem minimum_buses_required : 
  (total_round_trip_time / departure_interval) = 20 := by
  sorry

end NUMINAMATH_GPT_minimum_buses_required_l1863_186366


namespace NUMINAMATH_GPT_minimum_value_of_A_l1863_186396

open Real

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x * y + y * z + z * x)

theorem minimum_value_of_A (x y z : ℝ) (h : 3 ≤ x) (h2 : 3 ≤ y) (h3 : 3 ≤ z) :
  ∃ v : ℝ, (∀ a b c : ℝ, 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c → A a b c ≥ v) ∧ v = 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_A_l1863_186396


namespace NUMINAMATH_GPT_min_value_expr_l1863_186300

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ k : ℝ, k = 6 ∧ (∃ a b c : ℝ,
                  0 < a ∧
                  0 < b ∧
                  0 < c ∧
                  (k = (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a)) :=
sorry

end NUMINAMATH_GPT_min_value_expr_l1863_186300


namespace NUMINAMATH_GPT_evaluate_expression_l1863_186399

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1863_186399


namespace NUMINAMATH_GPT_largest_angle_l1863_186369

-- Assume the conditions
def angle_a : ℝ := 50
def angle_b : ℝ := 70
def angle_c (y : ℝ) : ℝ := 180 - (angle_a + angle_b)

-- State the proposition
theorem largest_angle (y : ℝ) (h : y = angle_c y) : angle_b = 70 := by
  sorry

end NUMINAMATH_GPT_largest_angle_l1863_186369


namespace NUMINAMATH_GPT_heartsuit_symmetric_solution_l1863_186376

def heartsuit (a b : ℝ) : ℝ :=
  a^3 * b - a^2 * b^2 + a * b^3

theorem heartsuit_symmetric_solution :
  ∀ x y : ℝ, (heartsuit x y = heartsuit y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end NUMINAMATH_GPT_heartsuit_symmetric_solution_l1863_186376


namespace NUMINAMATH_GPT_town_council_original_plan_count_l1863_186325

theorem town_council_original_plan_count (planned_trees current_trees : ℕ) (leaves_per_tree total_leaves : ℕ)
  (h1 : leaves_per_tree = 100)
  (h2 : total_leaves = 1400)
  (h3 : current_trees = total_leaves / leaves_per_tree)
  (h4 : current_trees = 2 * planned_trees) : 
  planned_trees = 7 :=
by
  sorry

end NUMINAMATH_GPT_town_council_original_plan_count_l1863_186325


namespace NUMINAMATH_GPT_circle_equation_l1863_186390

theorem circle_equation (x y : ℝ)
  (h_center : ∀ x y, (x - 3)^2 + (y - 1)^2 = r ^ 2)
  (h_origin : (0 - 3)^2 + (0 - 1)^2 = r ^ 2) :
  (x - 3) ^ 2 + (y - 1) ^ 2 = 10 := by
  sorry

end NUMINAMATH_GPT_circle_equation_l1863_186390


namespace NUMINAMATH_GPT_odd_function_max_to_min_l1863_186303

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_max_to_min (a b : ℝ) (f : ℝ → ℝ)
  (hodd : is_odd_function f)
  (hmax : ∃ x : ℝ, x > 0 ∧ (a * f x + b * x + 1) = 2) :
  ∃ y : ℝ, y < 0 ∧ (a * f y + b * y + 1) = 0 :=
sorry

end NUMINAMATH_GPT_odd_function_max_to_min_l1863_186303


namespace NUMINAMATH_GPT_minimum_value_occurs_at_4_l1863_186320

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_occurs_at_4_l1863_186320


namespace NUMINAMATH_GPT_sin_double_angle_l1863_186323

open Real

theorem sin_double_angle
  {α : ℝ} (h1: tan α = -1/2) (h2: 0 < α ∧ α < π) :
  sin (2 * α) = -4/5 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l1863_186323


namespace NUMINAMATH_GPT_megan_math_problems_l1863_186386

theorem megan_math_problems (num_spelling_problems num_problems_per_hour num_hours total_problems num_math_problems : ℕ) 
  (h1 : num_spelling_problems = 28)
  (h2 : num_problems_per_hour = 8)
  (h3 : num_hours = 8)
  (h4 : total_problems = num_problems_per_hour * num_hours)
  (h5 : total_problems = num_spelling_problems + num_math_problems) :
  num_math_problems = 36 := 
by
  sorry

end NUMINAMATH_GPT_megan_math_problems_l1863_186386


namespace NUMINAMATH_GPT_product_of_k_values_l1863_186373

theorem product_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_eq : a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k) : k = -1 :=
by
  sorry

end NUMINAMATH_GPT_product_of_k_values_l1863_186373


namespace NUMINAMATH_GPT_quadratic_eq_is_general_form_l1863_186312

def quadratic_eq_general_form (x : ℝ) : Prop :=
  x^2 - 2 * (3 * x - 2) + (x + 1) = x^2 - 5 * x + 5

theorem quadratic_eq_is_general_form :
  quadratic_eq_general_form x :=
sorry

end NUMINAMATH_GPT_quadratic_eq_is_general_form_l1863_186312


namespace NUMINAMATH_GPT_dress_total_selling_price_l1863_186309

theorem dress_total_selling_price (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 100) (h2 : discount_rate = 0.30) (h3 : tax_rate = 0.15) : 
  (original_price * (1 - discount_rate) * (1 + tax_rate)) = 80.5 := by
  sorry

end NUMINAMATH_GPT_dress_total_selling_price_l1863_186309


namespace NUMINAMATH_GPT_right_triangle_area_l1863_186349

variable (AB AC : ℝ) (angle_A : ℝ)

def is_right_triangle (AB AC : ℝ) (angle_A : ℝ) : Prop :=
  angle_A = 90

def area_of_triangle (AB AC : ℝ) : ℝ :=
  0.5 * AB * AC

theorem right_triangle_area :
  is_right_triangle AB AC angle_A →
  AB = 35 →
  AC = 15 →
  area_of_triangle AB AC = 262.5 :=
by
  intros
  simp [is_right_triangle, area_of_triangle]
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1863_186349


namespace NUMINAMATH_GPT_total_candies_correct_l1863_186316

-- Define the number of candies each has
def caleb_jellybeans := 3 * 12
def caleb_chocolate_bars := 5
def caleb_gummy_bears := 8
def caleb_total := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears

def sophie_jellybeans := (caleb_jellybeans / 2)
def sophie_chocolate_bars := 3
def sophie_gummy_bears := 12
def sophie_total := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears

def max_jellybeans := (2 * 12) + sophie_jellybeans
def max_chocolate_bars := 6
def max_gummy_bears := 10
def max_total := max_jellybeans + max_chocolate_bars + max_gummy_bears

-- Define the total number of candies
def total_candies := caleb_total + sophie_total + max_total

-- Theorem statement
theorem total_candies_correct : total_candies = 140 := by
  sorry

end NUMINAMATH_GPT_total_candies_correct_l1863_186316


namespace NUMINAMATH_GPT_work_days_in_week_l1863_186372

theorem work_days_in_week (total_toys_per_week : ℕ) (toys_produced_each_day : ℕ) (h1 : total_toys_per_week = 6500) (h2 : toys_produced_each_day = 1300) : 
  total_toys_per_week / toys_produced_each_day = 5 :=
by
  sorry

end NUMINAMATH_GPT_work_days_in_week_l1863_186372


namespace NUMINAMATH_GPT_balls_in_boxes_l1863_186348

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1863_186348


namespace NUMINAMATH_GPT_total_jewelry_pieces_l1863_186350

noncomputable def initial_necklaces : ℕ := 10
noncomputable def initial_earrings : ℕ := 15
noncomputable def bought_necklaces : ℕ := 10
noncomputable def bought_earrings : ℕ := 2 * initial_earrings / 3
noncomputable def extra_earrings_from_mother : ℕ := bought_earrings / 5

theorem total_jewelry_pieces : initial_necklaces + bought_necklaces + initial_earrings + bought_earrings + extra_earrings_from_mother = 47 :=
by
  have total_necklaces : ℕ := initial_necklaces + bought_necklaces
  have total_earrings : ℕ := initial_earrings + bought_earrings + extra_earrings_from_mother
  have total_jewelry : ℕ := total_necklaces + total_earrings
  exact Eq.refl 47
  
#check total_jewelry_pieces -- Check if the type is correct

end NUMINAMATH_GPT_total_jewelry_pieces_l1863_186350


namespace NUMINAMATH_GPT_find_number_l1863_186398

-- Define the number 40 and the percentage 90.
def num : ℝ := 40
def percent : ℝ := 0.9

-- Define the condition that 4/5 of x is smaller than 90% of 40 by 16
def condition (x : ℝ) : Prop := (4/5 : ℝ) * x = percent * num - 16

-- Proof statement in Lean 4
theorem find_number : ∃ x : ℝ, condition x ∧ x = 25 :=
by 
  use 25
  unfold condition
  norm_num
  sorry

end NUMINAMATH_GPT_find_number_l1863_186398


namespace NUMINAMATH_GPT_volume_of_pool_l1863_186355

variable (P T V C : ℝ)

/-- 
The volume of the pool is given as P * T divided by percentage C.
The question is to prove that the volume V of the pool equals 90000 cubic feet given:
  P: The hose can remove 60 cubic feet per minute.
  T: It takes 1200 minutes to drain the pool.
  C: The pool was at 80% capacity when draining started.
-/
theorem volume_of_pool (h1 : P = 60) 
                       (h2 : T = 1200) 
                       (h3 : C = 0.80) 
                       (h4 : P * T / C = V) :
  V = 90000 := 
sorry

end NUMINAMATH_GPT_volume_of_pool_l1863_186355


namespace NUMINAMATH_GPT_correctTechnologyUsedForVolcanicAshMonitoring_l1863_186387

-- Define the choices
inductive Technology
| RemoteSensing : Technology
| GPS : Technology
| GIS : Technology
| DigitalEarth : Technology

-- Define the problem conditions
def primaryTechnologyUsedForVolcanicAshMonitoring := Technology.RemoteSensing

-- The statement to prove
theorem correctTechnologyUsedForVolcanicAshMonitoring : primaryTechnologyUsedForVolcanicAshMonitoring = Technology.RemoteSensing :=
by
  sorry

end NUMINAMATH_GPT_correctTechnologyUsedForVolcanicAshMonitoring_l1863_186387


namespace NUMINAMATH_GPT_standard_deviation_is_one_l1863_186374

def mean : ℝ := 10.5
def value : ℝ := 8.5

theorem standard_deviation_is_one (σ : ℝ) (h : value = mean - 2 * σ) : σ = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_standard_deviation_is_one_l1863_186374
