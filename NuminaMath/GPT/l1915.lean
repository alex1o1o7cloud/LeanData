import Mathlib

namespace NUMINAMATH_GPT_candidate_valid_vote_percentage_l1915_191514

theorem candidate_valid_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_votes : ℕ) 
  (valid_percentage : ℚ)
  (total_votes_eq : total_votes = 560000)
  (invalid_percentage_eq : invalid_percentage = 15 / 100)
  (candidate_votes_eq : candidate_votes = 357000)
  (valid_percentage_eq : valid_percentage = 85 / 100) :
  (candidate_votes / (total_votes * valid_percentage)) * 100 = 75 := 
by
  sorry

end NUMINAMATH_GPT_candidate_valid_vote_percentage_l1915_191514


namespace NUMINAMATH_GPT_difference_in_floors_l1915_191528

-- Given conditions
variable (FA FB FC : ℕ)
variable (h1 : FA = 4)
variable (h2 : FC = 5 * FB - 6)
variable (h3 : FC = 59)

-- The statement to prove
theorem difference_in_floors : FB - FA = 9 :=
by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_difference_in_floors_l1915_191528


namespace NUMINAMATH_GPT_product_of_numbers_larger_than_reciprocal_eq_neg_one_l1915_191523

theorem product_of_numbers_larger_than_reciprocal_eq_neg_one :
  ∃ x y : ℝ, x ≠ y ∧ (x = 1 / x + 2) ∧ (y = 1 / y + 2) ∧ x * y = -1 :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_larger_than_reciprocal_eq_neg_one_l1915_191523


namespace NUMINAMATH_GPT_general_formula_a_n_sum_first_n_b_l1915_191551

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Sequence property
def seq_property (n : ℕ) (S_n : ℕ) : Prop :=
  a_n n ^ 2 + 2 * a_n n = 4 * S_n + 3

-- General formula for {a_n}
theorem general_formula_a_n (n : ℕ) (hpos : ∀ n, a_n n > 0) (S_n : ℕ) (hseq : seq_property n S_n) :
  a_n n = 2 * n + 1 :=
sorry

-- Sum of the first n terms of {b_n}
def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

def sum_b (n : ℕ) (T_n : ℚ) : Prop :=
  T_n = (1 / 2) * ((1 / (2 * n + 1)) - (1 / (2 * n + 3)))

theorem sum_first_n_b (n : ℕ) (hpos : ∀ n, a_n n > 0) (T_n : ℚ) :
  T_n = (n : ℚ) / (3 * (2 * n + 3)) :=
sorry

end NUMINAMATH_GPT_general_formula_a_n_sum_first_n_b_l1915_191551


namespace NUMINAMATH_GPT_cut_half_meter_from_cloth_l1915_191541

theorem cut_half_meter_from_cloth (initial_length : ℝ) (cut_length : ℝ) : 
  initial_length = 8 / 15 → cut_length = 1 / 30 → initial_length - cut_length = 1 / 2 := 
by
  intros h_initial h_cut
  sorry

end NUMINAMATH_GPT_cut_half_meter_from_cloth_l1915_191541


namespace NUMINAMATH_GPT_cos_alpha_plus_pi_over_3_l1915_191554

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (α + π / 3) = -1 / 3 :=
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_pi_over_3_l1915_191554


namespace NUMINAMATH_GPT_alan_total_payment_l1915_191503

-- Define the costs of CDs
def cost_AVN : ℝ := 12
def cost_TheDark : ℝ := 2 * cost_AVN
def cost_TheDark_total : ℝ := 2 * cost_TheDark
def cost_other_CDs : ℝ := cost_AVN + cost_TheDark_total
def cost_90s : ℝ := 0.4 * cost_other_CDs
def total_cost : ℝ := cost_AVN + cost_TheDark_total + cost_90s

-- Formulate the main statement
theorem alan_total_payment :
  total_cost = 84 := by
  sorry

end NUMINAMATH_GPT_alan_total_payment_l1915_191503


namespace NUMINAMATH_GPT_area_correct_l1915_191579

-- Define the conditions provided in the problem
def width (w : ℝ) := True
def length (l : ℝ) := True
def perimeter (p : ℝ) := True

-- Add the conditions about the playground
axiom length_exceeds_width_by : ∃ l w, l = 3 * w + 30
axiom perimeter_is_given : ∃ l w, 2 * (l + w) = 730

-- Define the area of the playground and state the theorem
noncomputable def area_of_playground : ℝ := 83.75 * 281.25

theorem area_correct :
  (∃ l w, l = 3 * w + 30 ∧ 2 * (l + w) = 730) →
  area_of_playground = 23554.6875 :=
by
  sorry

end NUMINAMATH_GPT_area_correct_l1915_191579


namespace NUMINAMATH_GPT_xiao_liang_reaches_museum_l1915_191588

noncomputable def xiao_liang_distance_to_museum : ℝ :=
  let science_museum := (200 * Real.sqrt 2, 200 * Real.sqrt 2)
  let initial_mistake := (-300 * Real.sqrt 2, 300 * Real.sqrt 2)
  let to_supermarket := (-100 * Real.sqrt 2, 500 * Real.sqrt 2)
  Real.sqrt ((science_museum.1 - to_supermarket.1)^2 + (science_museum.2 - to_supermarket.2)^2)

theorem xiao_liang_reaches_museum :
  xiao_liang_distance_to_museum = 600 :=
sorry

end NUMINAMATH_GPT_xiao_liang_reaches_museum_l1915_191588


namespace NUMINAMATH_GPT_suff_but_not_necc_l1915_191593

def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := (x - 2) * (x + 3) = 0

theorem suff_but_not_necc (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_suff_but_not_necc_l1915_191593


namespace NUMINAMATH_GPT_num_ways_to_write_3070_l1915_191583

theorem num_ways_to_write_3070 :
  let valid_digits := {d : ℕ | d ≤ 99}
  ∃ (M : ℕ), 
  M = 6500 ∧
  ∃ (a3 a2 a1 a0 : ℕ) (H : a3 ∈ valid_digits) (H : a2 ∈ valid_digits) (H : a1 ∈ valid_digits) (H : a0 ∈ valid_digits),
  3070 = a3 * 10^3 + a2 * 10^2 + a1 * 10 + a0 := sorry

end NUMINAMATH_GPT_num_ways_to_write_3070_l1915_191583


namespace NUMINAMATH_GPT_ratio_twice_width_to_length_l1915_191550

-- Given conditions:
def length_of_field : ℚ := 24
def width_of_field : ℚ := 13.5

-- The problem is to prove the ratio of twice the width to the length of the field is 9/8
theorem ratio_twice_width_to_length : 2 * width_of_field / length_of_field = 9 / 8 :=
by sorry

end NUMINAMATH_GPT_ratio_twice_width_to_length_l1915_191550


namespace NUMINAMATH_GPT_line_intersects_ellipse_possible_slopes_l1915_191509

theorem line_intersects_ellipse_possible_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔
    (m ≤ -Real.sqrt (1 / 20) ∨ m ≥ Real.sqrt (1 / 20)) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_ellipse_possible_slopes_l1915_191509


namespace NUMINAMATH_GPT_south_120_meters_l1915_191504

-- Define the directions
inductive Direction
| North
| South

-- Define the movement function
def movement (dir : Direction) (distance : Int) : Int :=
  match dir with
  | Direction.North => distance
  | Direction.South => -distance

-- Statement to prove
theorem south_120_meters : movement Direction.South 120 = -120 := 
by
  sorry

end NUMINAMATH_GPT_south_120_meters_l1915_191504


namespace NUMINAMATH_GPT_expression_evaluation_l1915_191508

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end NUMINAMATH_GPT_expression_evaluation_l1915_191508


namespace NUMINAMATH_GPT_sophomores_in_sample_l1915_191594

-- Define the number of freshmen, sophomores, and juniors
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the total number of students in the sample
def sample_size : ℕ := 100

-- Define the expected number of sophomores in the sample
def expected_sophomores : ℕ := (sample_size * sophomores) / total_students

-- Statement of the problem we want to prove
theorem sophomores_in_sample : expected_sophomores = 40 := by
  sorry

end NUMINAMATH_GPT_sophomores_in_sample_l1915_191594


namespace NUMINAMATH_GPT_cube_vertex_numbering_impossible_l1915_191548

-- Definition of the cube problem
def vertex_numbering_possible : Prop :=
  ∃ (v : Fin 8 → ℕ), (∀ i, 1 ≤ v i ∧ v i ≤ 8) ∧
    (∀ (e1 e2 : (Fin 8 × Fin 8)), e1 ≠ e2 → (v e1.1 + v e1.2 ≠ v e2.1 + v e2.2))

theorem cube_vertex_numbering_impossible : ¬ vertex_numbering_possible :=
sorry

end NUMINAMATH_GPT_cube_vertex_numbering_impossible_l1915_191548


namespace NUMINAMATH_GPT_quadratic_solution_range_l1915_191562

theorem quadratic_solution_range :
  ∃ x : ℝ, x^2 + 12 * x - 15 = 0 ∧ 1.1 < x ∧ x < 1.2 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_range_l1915_191562


namespace NUMINAMATH_GPT_merchant_gross_profit_l1915_191555

noncomputable def grossProfit (purchase_price : ℝ) (selling_price : ℝ) (discount : ℝ) : ℝ :=
  (selling_price - discount * selling_price) - purchase_price

theorem merchant_gross_profit :
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  grossProfit P S discount = 8 := 
by
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  unfold grossProfit
  sorry

end NUMINAMATH_GPT_merchant_gross_profit_l1915_191555


namespace NUMINAMATH_GPT_division_remainder_l1915_191552

theorem division_remainder : 4053 % 23 = 5 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1915_191552


namespace NUMINAMATH_GPT_projection_of_a_on_b_l1915_191578

open Real -- Use real numbers for vector operations

variables (a b : ℝ) -- Define a and b to be real numbers

-- Define the conditions as assumptions in Lean 4
def vector_magnitude_a (a : ℝ) : Prop := abs a = 1
def vector_magnitude_b (b : ℝ) : Prop := abs b = 1
def vector_dot_product (a b : ℝ) : Prop := (a + b) * b = 3 / 2

-- Define the goal to prove, using the assumptions
theorem projection_of_a_on_b (ha : vector_magnitude_a a) (hb : vector_magnitude_b b) (h_ab : vector_dot_product a b) : (abs a) * (a / b) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_projection_of_a_on_b_l1915_191578


namespace NUMINAMATH_GPT_minimum_jumps_l1915_191564

theorem minimum_jumps (a b : ℕ) (h : 2 * a + 3 * b = 2016) : a + b = 673 :=
sorry

end NUMINAMATH_GPT_minimum_jumps_l1915_191564


namespace NUMINAMATH_GPT_root_in_interval_iff_a_range_l1915_191527

def f (a x : ℝ) : ℝ := 2 * a * x ^ 2 + 2 * x - 3 - a

theorem root_in_interval_iff_a_range (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0) ↔ (1 ≤ a ∨ a ≤ - (3 + Real.sqrt 7) / 2) :=
sorry

end NUMINAMATH_GPT_root_in_interval_iff_a_range_l1915_191527


namespace NUMINAMATH_GPT_people_per_column_in_second_arrangement_l1915_191595
-- Lean 4 Statement

theorem people_per_column_in_second_arrangement :
  ∀ P X : ℕ, (P = 30 * 16) → (12 * X = P) → X = 40 :=
by
  intros P X h1 h2
  sorry

end NUMINAMATH_GPT_people_per_column_in_second_arrangement_l1915_191595


namespace NUMINAMATH_GPT_equal_angles_count_l1915_191590

-- Definitions corresponding to the problem conditions
def fast_clock_angle (t : ℝ) : ℝ := |30 * t - 5.5 * (t * 60)|
def slow_clock_angle (t : ℝ) : ℝ := |15 * t - 2.75 * (t * 60)|

theorem equal_angles_count :
  ∃ n : ℕ, n = 18 ∧ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 12 →
  fast_clock_angle t = slow_clock_angle t ↔ n = 18 :=
sorry

end NUMINAMATH_GPT_equal_angles_count_l1915_191590


namespace NUMINAMATH_GPT_divides_square_sum_implies_divides_l1915_191591

theorem divides_square_sum_implies_divides (a b : ℤ) (h : 7 ∣ a^2 + b^2) : 7 ∣ a ∧ 7 ∣ b := 
sorry

end NUMINAMATH_GPT_divides_square_sum_implies_divides_l1915_191591


namespace NUMINAMATH_GPT_degree_monomial_equal_four_l1915_191581

def degree_of_monomial (a b : ℝ) := 
  (3 + 1)

theorem degree_monomial_equal_four (a b : ℝ) 
  (h : a^3 * b = (2/3) * a^3 * b) : 
  degree_of_monomial a b = 4 :=
by sorry

end NUMINAMATH_GPT_degree_monomial_equal_four_l1915_191581


namespace NUMINAMATH_GPT_find_number_l1915_191501

theorem find_number (x : ℤ) : (150 - x = x + 68) → x = 41 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l1915_191501


namespace NUMINAMATH_GPT_visited_neither_l1915_191500

theorem visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) 
  (h1 : total = 100) 
  (h2 : iceland = 55) 
  (h3 : norway = 43) 
  (h4 : both = 61) : 
  (total - (iceland + norway - both)) = 63 := 
by 
  sorry

end NUMINAMATH_GPT_visited_neither_l1915_191500


namespace NUMINAMATH_GPT_truncated_pyramid_lateral_surface_area_l1915_191586

noncomputable def lateralSurfaceAreaTruncatedPyramid (s1 s2 h : ℝ) :=
  let l := Real.sqrt (h^2 + ((s1 - s2) / 2)^2)
  let P1 := 4 * s1
  let P2 := 4 * s2
  (1 / 2) * (P1 + P2) * l

theorem truncated_pyramid_lateral_surface_area :
  lateralSurfaceAreaTruncatedPyramid 10 5 7 = 222.9 :=
by
  sorry

end NUMINAMATH_GPT_truncated_pyramid_lateral_surface_area_l1915_191586


namespace NUMINAMATH_GPT_driver_weekly_distance_l1915_191531

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end NUMINAMATH_GPT_driver_weekly_distance_l1915_191531


namespace NUMINAMATH_GPT_additional_stars_needed_l1915_191534

-- Defining the number of stars required per bottle
def stars_per_bottle : Nat := 85

-- Defining the number of bottles Luke needs to fill
def bottles_to_fill : Nat := 4

-- Defining the number of stars Luke has already made
def stars_made : Nat := 33

-- Calculating the number of stars Luke still needs to make
theorem additional_stars_needed : (stars_per_bottle * bottles_to_fill - stars_made) = 307 := by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_additional_stars_needed_l1915_191534


namespace NUMINAMATH_GPT_problem1_problem2_l1915_191563

-- Problem 1: Proving the equation
theorem problem1 (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 → x = 1 :=
sorry

-- Problem 2: Proving the solution for the system of equations
theorem problem2 (x y : ℝ) : (x + 2 * y = 8) ∧ (3 * x - 4 * y = 4) → x = 4 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1915_191563


namespace NUMINAMATH_GPT_base_n_divisible_by_13_l1915_191557

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 7 + 3 * n + 5 * n^2 + 6 * n^3 + 3 * n^4 + 5 * n^5

-- The main theorem stating the result
theorem base_n_divisible_by_13 : 
  (∃ ns : Finset ℕ, ns.card = 16 ∧ ∀ n ∈ ns, 3 ≤ n ∧ n ≤ 200 ∧ f n % 13 = 0) :=
sorry

end NUMINAMATH_GPT_base_n_divisible_by_13_l1915_191557


namespace NUMINAMATH_GPT_complex_expression_equality_l1915_191513

open Complex

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := 
sorry

end NUMINAMATH_GPT_complex_expression_equality_l1915_191513


namespace NUMINAMATH_GPT_peter_twice_as_old_in_years_l1915_191572

def mother_age : ℕ := 60
def harriet_current_age : ℕ := 13
def peter_current_age : ℕ := mother_age / 2
def years_later : ℕ := 4

theorem peter_twice_as_old_in_years : 
  peter_current_age + years_later = 2 * (harriet_current_age + years_later) :=
by
  -- using given conditions 
  -- Peter's current age is 30
  -- Harriet's current age is 13
  -- years_later is 4
  sorry

end NUMINAMATH_GPT_peter_twice_as_old_in_years_l1915_191572


namespace NUMINAMATH_GPT_factor_expression_l1915_191520

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1915_191520


namespace NUMINAMATH_GPT_percent_defective_shipped_l1915_191521

theorem percent_defective_shipped
  (P_d : ℝ) (P_s : ℝ)
  (hP_d : P_d = 0.1)
  (hP_s : P_s = 0.05) :
  P_d * P_s = 0.005 :=
by
  sorry

end NUMINAMATH_GPT_percent_defective_shipped_l1915_191521


namespace NUMINAMATH_GPT_area_ratio_l1915_191574

-- Define the conditions: perimeters relation
def condition (a b : ℝ) := 4 * a = 16 * b

-- Define the theorem to be proved
theorem area_ratio (a b : ℝ) (h : condition a b) : (a * a) = 16 * (b * b) :=
sorry

end NUMINAMATH_GPT_area_ratio_l1915_191574


namespace NUMINAMATH_GPT_common_difference_is_3_l1915_191524

noncomputable def whale_plankton_frenzy (x : ℝ) (y : ℝ) : Prop :=
  (9 * x + 36 * y = 450) ∧
  (x + 5 * y = 53)

theorem common_difference_is_3 :
  ∃ (x y : ℝ), whale_plankton_frenzy x y ∧ y = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_common_difference_is_3_l1915_191524


namespace NUMINAMATH_GPT_Robert_more_than_Claire_l1915_191510

variable (Lisa Claire Robert : ℕ)

theorem Robert_more_than_Claire (h1 : Lisa = 3 * Claire) (h2 : Claire = 10) (h3 : Robert > Claire) :
  Robert > 10 :=
by
  rw [h2] at h3
  assumption

end NUMINAMATH_GPT_Robert_more_than_Claire_l1915_191510


namespace NUMINAMATH_GPT_percent_of_x_is_65_l1915_191598

variable (z y x : ℝ)

theorem percent_of_x_is_65 :
  (0.45 * z = 0.39 * y) → (y = 0.75 * x) → (z / x = 0.65) := by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_65_l1915_191598


namespace NUMINAMATH_GPT_calculate_division_l1915_191516

theorem calculate_division : 
  (- (1 / 28)) / ((1 / 2) - (1 / 4) + (1 / 7) - (1 / 14)) = - (1 / 9) :=
by
  sorry

end NUMINAMATH_GPT_calculate_division_l1915_191516


namespace NUMINAMATH_GPT_ratio_of_roses_l1915_191560

-- Definitions for conditions
def roses_two_days_ago : ℕ := 50
def roses_yesterday : ℕ := roses_two_days_ago + 20
def roses_total : ℕ := 220
def roses_today : ℕ := roses_total - roses_two_days_ago - roses_yesterday

-- Lean statement to prove the ratio of roses planted today to two days ago is 2
theorem ratio_of_roses :
  roses_today / roses_two_days_ago = 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_roses_l1915_191560


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1915_191567

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, -1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1915_191567


namespace NUMINAMATH_GPT_number_of_intersections_l1915_191542

-- Definitions of the given curves.
def curve1 (x y : ℝ) : Prop := x^2 + 4*y^2 = 1
def curve2 (x y : ℝ) : Prop := 4*x^2 + y^2 = 4

-- Statement of the theorem
theorem number_of_intersections : ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2 := sorry

end NUMINAMATH_GPT_number_of_intersections_l1915_191542


namespace NUMINAMATH_GPT_functional_equation_solution_l1915_191568

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧ (∀ x y : ℝ, f (x + y) * f (x + y) = 2 * f x * f y + max (f (x * x) + f (y * y)) (f (x * x + y * y)))

theorem functional_equation_solution (f : ℝ → ℝ) :
  satisfies_conditions f → (∀ x : ℝ, f x = -1 ∨ f x = x - 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1915_191568


namespace NUMINAMATH_GPT_trigonometric_identity_l1915_191536

theorem trigonometric_identity
  (α : ℝ) 
  (h : Real.tan α = -1 / 2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1915_191536


namespace NUMINAMATH_GPT_remainder_of_sum_of_ns_l1915_191584

theorem remainder_of_sum_of_ns (S : ℕ) :
  (∃ (ns : List ℕ), (∀ n ∈ ns, ∃ m : ℕ, n^2 + 12*n - 1997 = m^2) ∧ S = ns.sum) →
  S % 1000 = 154 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_ns_l1915_191584


namespace NUMINAMATH_GPT_bernardo_wins_l1915_191599

/-- 
Bernardo and Silvia play the following game. An integer between 0 and 999 inclusive is selected
and given to Bernardo. Whenever Bernardo receives a number, he doubles it and passes the result 
to Silvia. Whenever Silvia receives a number, she adds 50 to it and passes the result back. 
The winner is the last person who produces a number less than 1000. The smallest initial number 
that results in a win for Bernardo is 16, and the sum of the digits of 16 is 7.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem bernardo_wins (N : ℕ) (h : 16 ≤ N ∧ N ≤ 18) : sum_of_digits 16 = 7 :=
by
  sorry

end NUMINAMATH_GPT_bernardo_wins_l1915_191599


namespace NUMINAMATH_GPT_probability_age_20_to_40_l1915_191502

theorem probability_age_20_to_40 
    (total_people : ℕ) (aged_20_to_30 : ℕ) (aged_30_to_40 : ℕ) 
    (h_total : total_people = 350) 
    (h_aged_20_to_30 : aged_20_to_30 = 105) 
    (h_aged_30_to_40 : aged_30_to_40 = 85) : 
    (190 / 350 : ℚ) = 19 / 35 := 
by 
  sorry

end NUMINAMATH_GPT_probability_age_20_to_40_l1915_191502


namespace NUMINAMATH_GPT_Jon_needs_to_wash_20_pairs_of_pants_l1915_191535

theorem Jon_needs_to_wash_20_pairs_of_pants
  (machine_capacity : ℕ)
  (shirts_per_pound : ℕ)
  (pants_per_pound : ℕ)
  (num_shirts : ℕ)
  (num_loads : ℕ)
  (total_pounds : ℕ)
  (weight_of_shirts : ℕ)
  (remaining_weight : ℕ)
  (num_pairs_of_pants : ℕ) :
  machine_capacity = 5 →
  shirts_per_pound = 4 →
  pants_per_pound = 2 →
  num_shirts = 20 →
  num_loads = 3 →
  total_pounds = num_loads * machine_capacity →
  weight_of_shirts = num_shirts / shirts_per_pound →
  remaining_weight = total_pounds - weight_of_shirts →
  num_pairs_of_pants = remaining_weight * pants_per_pound →
  num_pairs_of_pants = 20 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_Jon_needs_to_wash_20_pairs_of_pants_l1915_191535


namespace NUMINAMATH_GPT_sum_of_roots_l1915_191529

-- States that the sum of the values of x that satisfy the given quadratic equation is 7
theorem sum_of_roots (x : ℝ) :
  (x^2 - 7 * x + 12 = 4) → (∃ a b : ℝ, x^2 - 7 * x + 8 = 0 ∧ a + b = 7) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1915_191529


namespace NUMINAMATH_GPT_range_of_a_for_root_l1915_191587

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_root (a : ℝ) : (∃ x, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 := sorry

end NUMINAMATH_GPT_range_of_a_for_root_l1915_191587


namespace NUMINAMATH_GPT_nine_values_of_x_l1915_191585

theorem nine_values_of_x : ∃! (n : ℕ), ∃! (xs : Finset ℕ), xs.card = n ∧ 
  (∀ x ∈ xs, 3 * x < 100 ∧ 4 * x ≥ 100) ∧ 
  (xs.image (λ x => x)).val = ({25, 26, 27, 28, 29, 30, 31, 32, 33} : Finset ℕ).val :=
sorry

end NUMINAMATH_GPT_nine_values_of_x_l1915_191585


namespace NUMINAMATH_GPT_hexagon_piece_area_l1915_191539

theorem hexagon_piece_area (A : ℝ) (n : ℕ) (h1 : A = 21.12) (h2 : n = 6) : 
  A / n = 3.52 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_hexagon_piece_area_l1915_191539


namespace NUMINAMATH_GPT_transformed_function_zero_l1915_191532

-- Definitions based on conditions
def f : ℝ → ℝ → ℝ := sorry  -- Assume this is the given function f(x, y)

-- Transformed function according to symmetry and reflections
def transformed_f (x y : ℝ) : Prop := f (y + 2) (x - 2) = 0

-- Lean statement to be proved
theorem transformed_function_zero (x y : ℝ) : transformed_f x y := sorry

end NUMINAMATH_GPT_transformed_function_zero_l1915_191532


namespace NUMINAMATH_GPT_probability_other_side_red_l1915_191589

def card_black_black := 4
def card_black_red := 2
def card_red_red := 2

def total_cards := card_black_black + card_black_red + card_red_red

-- Calculate the total number of red faces
def total_red_faces := (card_red_red * 2) + card_black_red

-- Number of red faces that have the other side also red
def red_faces_with_other_red := card_red_red * 2

-- Target probability to prove
theorem probability_other_side_red (h : total_cards = 8) : 
  (red_faces_with_other_red / total_red_faces) = 2 / 3 := 
  sorry

end NUMINAMATH_GPT_probability_other_side_red_l1915_191589


namespace NUMINAMATH_GPT_more_wrappers_than_bottle_caps_at_park_l1915_191522

-- Define the number of bottle caps and wrappers found at the park.
def bottle_caps_found : ℕ := 11
def wrappers_found : ℕ := 28

-- State the theorem to prove the number of more wrappers than bottle caps found at the park is 17.
theorem more_wrappers_than_bottle_caps_at_park : wrappers_found - bottle_caps_found = 17 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_more_wrappers_than_bottle_caps_at_park_l1915_191522


namespace NUMINAMATH_GPT_probability_of_male_selected_l1915_191597

-- Define the total number of students
def num_students : ℕ := 100

-- Define the number of male students
def num_male_students : ℕ := 25

-- Define the number of students selected
def num_students_selected : ℕ := 20

theorem probability_of_male_selected :
  (num_students_selected : ℚ) / num_students = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_male_selected_l1915_191597


namespace NUMINAMATH_GPT_base7_addition_l1915_191526

theorem base7_addition (X Y : ℕ) (h1 : Y + 2 = X) (h2 : X + 5 = 8) : X + Y = 4 :=
by
  sorry

end NUMINAMATH_GPT_base7_addition_l1915_191526


namespace NUMINAMATH_GPT_compute_ζ7_sum_l1915_191592

noncomputable def ζ_power_sum (ζ1 ζ2 ζ3 : ℂ) : Prop :=
  (ζ1 + ζ2 + ζ3 = 2) ∧
  (ζ1^2 + ζ2^2 + ζ3^2 = 6) ∧
  (ζ1^3 + ζ2^3 + ζ3^3 = 8) →
  ζ1^7 + ζ2^7 + ζ3^7 = 58

theorem compute_ζ7_sum (ζ1 ζ2 ζ3 : ℂ) (h : ζ_power_sum ζ1 ζ2 ζ3) : ζ1^7 + ζ2^7 + ζ3^7 = 58 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_compute_ζ7_sum_l1915_191592


namespace NUMINAMATH_GPT_seventh_fisherman_right_neighbor_l1915_191511

theorem seventh_fisherman_right_neighbor (f1 f2 f3 f4 f5 f6 f7 : ℕ) (L1 L2 L3 L4 L5 L6 L7 : ℕ) :
  (L2 * f1 = 12 ∨ L3 * f2 = 12 ∨ L4 * f3 = 12 ∨ L5 * f4 = 12 ∨ L6 * f5 = 12 ∨ L7 * f6 = 12 ∨ L1 * f7 = 12) → 
  (L2 * f1 = 14 ∨ L3 * f2 = 18 ∨ L4 * f3 = 32 ∨ L5 * f4 = 48 ∨ L6 * f5 = 70 ∨ L7 * f6 = x ∨ L1 * f7 = 12) →
  (12 * 12 * 20 * 24 * 32 * 42 * 56) / (12 * 14 * 18 * 32 * 48 * 70) = x :=
by
  sorry

end NUMINAMATH_GPT_seventh_fisherman_right_neighbor_l1915_191511


namespace NUMINAMATH_GPT_planks_needed_l1915_191549

theorem planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (h1 : total_nails = 4) (h2 : nails_per_plank = 2) : total_nails / nails_per_plank = 2 :=
by
  -- Prove that given the conditions, the required result is obtained
  sorry

end NUMINAMATH_GPT_planks_needed_l1915_191549


namespace NUMINAMATH_GPT_grid_3x3_unique_72_l1915_191569

theorem grid_3x3_unique_72 :
  ∃ (f : Fin 3 → Fin 3 → ℕ), 
    (∀ (i j : Fin 3), 1 ≤ f i j ∧ f i j ≤ 9) ∧
    (∀ (i j k : Fin 3), j < k → f i j < f i k) ∧
    (∀ (i j k : Fin 3), i < k → f i j < f k j) ∧
    f 0 0 = 1 ∧ f 1 1 = 5 ∧ f 2 2 = 8 ∧
    (∃! (g : Fin 3 → Fin 3 → ℕ), 
      (∀ (i j : Fin 3), 1 ≤ g i j ∧ g i j ≤ 9) ∧
      (∀ (i j k : Fin 3), j < k → g i j < g i k) ∧
      (∀ (i j k : Fin 3), i < k → g i j < g k j) ∧
      g 0 0 = 1 ∧ g 1 1 = 5 ∧ g 2 2 = 8) :=
sorry

end NUMINAMATH_GPT_grid_3x3_unique_72_l1915_191569


namespace NUMINAMATH_GPT_sandy_marbles_correct_l1915_191544

namespace MarbleProblem

-- Define the number of dozens Jessica has
def jessica_dozens : ℕ := 3

-- Define the conversion from dozens to individual marbles
def dozens_to_marbles (d : ℕ) : ℕ := 12 * d

-- Calculate the number of marbles Jessica has
def jessica_marbles : ℕ := dozens_to_marbles jessica_dozens

-- Define the multiplier for Sandy's marbles
def sandy_multiplier : ℕ := 4

-- Define the number of marbles Sandy has
def sandy_marbles : ℕ := sandy_multiplier * jessica_marbles

theorem sandy_marbles_correct : sandy_marbles = 144 :=
by
  sorry

end MarbleProblem

end NUMINAMATH_GPT_sandy_marbles_correct_l1915_191544


namespace NUMINAMATH_GPT_problem_statement_l1915_191570

theorem problem_statement (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) (hprod : m * n = 5000) 
  (h_m_not_div_10 : ¬ ∃ k, m = 10 * k) (h_n_not_div_10 : ¬ ∃ k, n = 10 * k) :
  m + n = 633 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1915_191570


namespace NUMINAMATH_GPT_age_difference_l1915_191577

variable (x y z : ℝ)

def overall_age_condition (x y z : ℝ) : Prop := (x + y = y + z + 10)

theorem age_difference (x y z : ℝ) (h : overall_age_condition x y z) : (x - z) / 10 = 1 :=
  by
    sorry

end NUMINAMATH_GPT_age_difference_l1915_191577


namespace NUMINAMATH_GPT_rectangle_area_ratio_l1915_191566

theorem rectangle_area_ratio (length width diagonal : ℝ) (h_ratio : length / width = 5 / 2) (h_diagonal : diagonal = 13) :
    ∃ k : ℝ, (length * width) = k * diagonal^2 ∧ k = 10 / 29 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l1915_191566


namespace NUMINAMATH_GPT_difference_between_numbers_l1915_191559

variable (x y : ℕ)

theorem difference_between_numbers (h1 : x + y = 34) (h2 : y = 22) : y - x = 10 := by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l1915_191559


namespace NUMINAMATH_GPT_XiaoKang_min_sets_pushups_pullups_l1915_191573

theorem XiaoKang_min_sets_pushups_pullups (x y : ℕ) (hx : x ≥ 100) (hy : y ≥ 106) (h : 8 * x + 5 * y = 9050) :
  x ≥ 100 ∧ y ≥ 106 :=
by {
  sorry  -- proof not required as per instruction
}

end NUMINAMATH_GPT_XiaoKang_min_sets_pushups_pullups_l1915_191573


namespace NUMINAMATH_GPT_linear_equation_condition_l1915_191540

theorem linear_equation_condition (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x ^ (|a|⁻¹ + 3) = 0) ↔ a = -2 := 
by
  sorry

end NUMINAMATH_GPT_linear_equation_condition_l1915_191540


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l1915_191580

theorem number_of_correct_conclusions
  (a b c : ℕ)
  (h1 : (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) 
  (conclusion1 : (a^b - b^c) % 2 = 1 ∧ (b^c - c^a) % 2 = 1 ∧ (c^a - a^b) % 2 = 1)
  (conclusion4 : ¬ ∃ a b c : ℕ, (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_conclusions_l1915_191580


namespace NUMINAMATH_GPT_find_a4_l1915_191596

-- Define the sequence
noncomputable def a : ℕ → ℝ := sorry

-- Define the initial term a1 and common difference d
noncomputable def a1 : ℝ := sorry
noncomputable def d : ℝ := sorry

-- The conditions from the problem
def condition1 : Prop := a 2 + a 6 = 10 * Real.sqrt 3
def condition2 : Prop := a 3 + a 7 = 14 * Real.sqrt 3

-- Using the conditions to prove a4
theorem find_a4 (h1 : condition1) (h2 : condition2) : a 4 = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_l1915_191596


namespace NUMINAMATH_GPT_noemi_initial_amount_l1915_191556

-- Define the conditions
def lost_on_roulette : Int := 400
def lost_on_blackjack : Int := 500
def still_has : Int := 800
def total_lost : Int := lost_on_roulette + lost_on_blackjack

-- Define the theorem to be proven
theorem noemi_initial_amount : total_lost + still_has = 1700 := by
  -- The proof will be added here
  sorry

end NUMINAMATH_GPT_noemi_initial_amount_l1915_191556


namespace NUMINAMATH_GPT_quadratic_discriminant_l1915_191518

def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/2) (-2) = 281/4 := by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_l1915_191518


namespace NUMINAMATH_GPT_intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l1915_191582

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem intervals_monotonicity_f :
  ∀ k : ℤ,
    (∀ x : ℝ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 → f x = Real.cos (2 * x)) ∧
    (∀ x : ℝ, k * Real.pi + Real.pi / 2 ≤ x ∧ x ≤ k * Real.pi + Real.pi → f x = Real.cos (2 * x)) :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem intervals_monotonicity_g_and_extremum :
  ∀ x : ℝ,
    (-Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi / 3 → (g x ≤ 1 ∧ g x ≥ -1)) :=
sorry

end NUMINAMATH_GPT_intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l1915_191582


namespace NUMINAMATH_GPT_coefficient_x7_in_expansion_l1915_191525

theorem coefficient_x7_in_expansion : 
  let n := 10
  let k := 7
  let binom := Nat.choose n k
  let coeff := 1
  coeff * binom = 120 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x7_in_expansion_l1915_191525


namespace NUMINAMATH_GPT_probability_white_then_black_l1915_191553

-- Definition of conditions
def total_balls := 5
def white_balls := 3
def black_balls := 2

def first_draw_white_probability (total white : ℕ) : ℚ :=
  white / total

def second_draw_black_probability (remaining_white remaining_black : ℕ) : ℚ :=
  remaining_black / (remaining_white + remaining_black)

-- The theorem statement
theorem probability_white_then_black :
  first_draw_white_probability total_balls white_balls *
  second_draw_black_probability (total_balls - 1) black_balls
  = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_white_then_black_l1915_191553


namespace NUMINAMATH_GPT_cubic_equation_solution_bound_l1915_191519

theorem cubic_equation_solution_bound (a : ℝ) :
  a ∈ Set.Ici (-15) → ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ → x₂ ≠ x₃ → x₁ ≠ x₃ →
  (x₁^3 + 6 * x₁^2 + a * x₁ + 8 = 0) →
  (x₂^3 + 6 * x₂^2 + a * x₂ + 8 = 0) →
  (x₃^3 + 6 * x₃^2 + a * x₃ + 8 = 0) →
  False := 
sorry

end NUMINAMATH_GPT_cubic_equation_solution_bound_l1915_191519


namespace NUMINAMATH_GPT_wire_ratio_l1915_191543

theorem wire_ratio (bonnie_pieces : ℕ) (length_per_bonnie_piece : ℕ) (roark_volume : ℕ) 
  (unit_cube_volume : ℕ) (bonnie_cube_volume : ℕ) (roark_pieces_per_unit_cube : ℕ)
  (bonnie_total_wire : ℕ := bonnie_pieces * length_per_bonnie_piece)
  (roark_total_wire : ℕ := (bonnie_cube_volume / unit_cube_volume) * roark_pieces_per_unit_cube) :
  bonnie_pieces = 12 →
  length_per_bonnie_piece = 4 →
  unit_cube_volume = 1 →
  bonnie_cube_volume = 64 →
  roark_pieces_per_unit_cube = 12 →
  (bonnie_total_wire / roark_total_wire : ℚ) = 1 / 16 :=
by sorry

end NUMINAMATH_GPT_wire_ratio_l1915_191543


namespace NUMINAMATH_GPT_square_area_in_ellipse_l1915_191576

theorem square_area_in_ellipse : ∀ (s : ℝ), 
  (s > 0) → 
  (∀ x y, (x = s ∨ x = -s) ∧ (y = s ∨ y = -s) → (x^2) / 4 + (y^2) / 8 = 1) → 
  (2 * s)^2 = 32 / 3 := by
  sorry

end NUMINAMATH_GPT_square_area_in_ellipse_l1915_191576


namespace NUMINAMATH_GPT_abs_diff_l1915_191571

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end NUMINAMATH_GPT_abs_diff_l1915_191571


namespace NUMINAMATH_GPT_petya_catch_bus_l1915_191565

theorem petya_catch_bus 
    (v_p v_b d : ℝ) 
    (h1 : v_b = 5 * v_p)
    (h2 : ∀ t : ℝ, 5 * v_p * t ≤ 0.6) 
    : d = 0.12 := 
sorry

end NUMINAMATH_GPT_petya_catch_bus_l1915_191565


namespace NUMINAMATH_GPT_miles_driven_l1915_191507

def total_miles : ℕ := 1200
def remaining_miles : ℕ := 432

theorem miles_driven : total_miles - remaining_miles = 768 := by
  sorry

end NUMINAMATH_GPT_miles_driven_l1915_191507


namespace NUMINAMATH_GPT_sin_13pi_over_4_eq_neg_sqrt2_over_2_l1915_191547

theorem sin_13pi_over_4_eq_neg_sqrt2_over_2 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_13pi_over_4_eq_neg_sqrt2_over_2_l1915_191547


namespace NUMINAMATH_GPT_total_cubes_proof_l1915_191558

def Grady_initial_red_cubes := 20
def Grady_initial_blue_cubes := 15
def Gage_initial_red_cubes := 10
def Gage_initial_blue_cubes := 12
def Harper_initial_red_cubes := 8
def Harper_initial_blue_cubes := 10

def Gage_red_received := (2 / 5) * Grady_initial_red_cubes
def Gage_blue_received := (1 / 3) * Grady_initial_blue_cubes

def Grady_red_after_Gage := Grady_initial_red_cubes - Gage_red_received
def Grady_blue_after_Gage := Grady_initial_blue_cubes - Gage_blue_received

def Harper_red_received := (1 / 4) * Grady_red_after_Gage
def Harper_blue_received := (1 / 2) * Grady_blue_after_Gage

def Gage_total_red := Gage_initial_red_cubes + Gage_red_received
def Gage_total_blue := Gage_initial_blue_cubes + Gage_blue_received

def Harper_total_red := Harper_initial_red_cubes + Harper_red_received
def Harper_total_blue := Harper_initial_blue_cubes + Harper_blue_received

def Gage_total_cubes := Gage_total_red + Gage_total_blue
def Harper_total_cubes := Harper_total_red + Harper_total_blue

def Gage_Harper_total_cubes := Gage_total_cubes + Harper_total_cubes

theorem total_cubes_proof : Gage_Harper_total_cubes = 61 := by
  sorry

end NUMINAMATH_GPT_total_cubes_proof_l1915_191558


namespace NUMINAMATH_GPT_value_of_expression_l1915_191512

theorem value_of_expression (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : 
  (x - 2 * y + 3 * z) / (x + y + z) = 8 / 9 := 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1915_191512


namespace NUMINAMATH_GPT_adjacent_sum_constant_l1915_191515

theorem adjacent_sum_constant (x y : ℤ) (k : ℤ) (h1 : 2 + x = k) (h2 : x + y = k) (h3 : y + 5 = k) : x - y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_adjacent_sum_constant_l1915_191515


namespace NUMINAMATH_GPT_shopkeeper_discount_l1915_191533

theorem shopkeeper_discount :
  let CP := 100
  let SP_with_discount := 119.7
  let SP_without_discount := 126
  let discount := SP_without_discount - SP_with_discount
  let discount_percentage := (discount / SP_without_discount) * 100
  discount_percentage = 5 := sorry

end NUMINAMATH_GPT_shopkeeper_discount_l1915_191533


namespace NUMINAMATH_GPT_foil_covered_prism_width_l1915_191561

def inner_prism_length (l : ℝ) := l
def inner_prism_width (l : ℝ) := 2 * l
def inner_prism_height (l : ℝ) := l
def inner_prism_volume (l : ℝ) := l * (2 * l) * l

theorem foil_covered_prism_width :
  (∃ l : ℝ, inner_prism_volume l = 128) → (inner_prism_width l + 2 = 8) := by
sorry

end NUMINAMATH_GPT_foil_covered_prism_width_l1915_191561


namespace NUMINAMATH_GPT_min_value_fraction_l1915_191530

theorem min_value_fraction (x : ℝ) (h : x > 9) : (x^2 + 81) / (x - 9) ≥ 27 := 
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1915_191530


namespace NUMINAMATH_GPT_No_response_percentage_l1915_191545

theorem No_response_percentage (total_guests : ℕ) (yes_percentage : ℕ) (non_respondents : ℕ) (yes_guests := total_guests * yes_percentage / 100) (no_guests := total_guests - yes_guests - non_respondents) (no_percentage := no_guests * 100 / total_guests) :
  total_guests = 200 → yes_percentage = 83 → non_respondents = 16 → no_percentage = 9 :=
by
  sorry

end NUMINAMATH_GPT_No_response_percentage_l1915_191545


namespace NUMINAMATH_GPT_geom_seq_increasing_sufficient_necessary_l1915_191575

theorem geom_seq_increasing_sufficient_necessary (a : ℕ → ℝ) (r : ℝ) (h_geo : ∀ n : ℕ, a n = a 0 * r ^ n) 
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) : 
  (a 0 < a 1 ∧ a 1 < a 2) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end NUMINAMATH_GPT_geom_seq_increasing_sufficient_necessary_l1915_191575


namespace NUMINAMATH_GPT_length_of_plot_l1915_191517

theorem length_of_plot 
  (b : ℝ)
  (H1 : 2 * (b + 20) + 2 * b = 5300 / 26.50)
  : (b + 20 = 60) :=
sorry

end NUMINAMATH_GPT_length_of_plot_l1915_191517


namespace NUMINAMATH_GPT_no_solutions_in_naturals_l1915_191506

theorem no_solutions_in_naturals (n k : ℕ) : ¬ (n ≤ n! - k^n ∧ n! - k^n ≤ k * n) :=
sorry

end NUMINAMATH_GPT_no_solutions_in_naturals_l1915_191506


namespace NUMINAMATH_GPT_fourth_term_geom_progression_l1915_191546

theorem fourth_term_geom_progression : 
  ∀ (a b c : ℝ), 
    a = 4^(1/2) → 
    b = 4^(1/3) → 
    c = 4^(1/6) → 
    ∃ d : ℝ, d = 1 ∧ b / a = c / b ∧ c / b = 4^(1/6) / 4^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_fourth_term_geom_progression_l1915_191546


namespace NUMINAMATH_GPT_proposition_false_at_6_l1915_191505

variable (P : ℕ → Prop)

theorem proposition_false_at_6 (h1 : ∀ k : ℕ, 0 < k → P k → P (k + 1)) (h2 : ¬P 7): ¬P 6 :=
by
  sorry

end NUMINAMATH_GPT_proposition_false_at_6_l1915_191505


namespace NUMINAMATH_GPT_probability_4_students_same_vehicle_l1915_191537

-- Define the number of vehicles
def num_vehicles : ℕ := 3

-- Define the probability that 4 students choose the same vehicle
def probability_same_vehicle (n : ℕ) : ℚ :=
  3 / (3^(n : ℤ))

-- Prove that the probability for 4 students is 1/27
theorem probability_4_students_same_vehicle : probability_same_vehicle 4 = 1 / 27 := 
  sorry

end NUMINAMATH_GPT_probability_4_students_same_vehicle_l1915_191537


namespace NUMINAMATH_GPT_intersection_lines_k_l1915_191538

theorem intersection_lines_k (k : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_lines_k_l1915_191538
