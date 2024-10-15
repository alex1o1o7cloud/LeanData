import Mathlib

namespace NUMINAMATH_GPT_odd_function_m_value_l2417_241798

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x - m

theorem odd_function_m_value :
  ∃ m : ℝ, (∀ (x : ℝ), g (-x) m + g x m = 0) ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_m_value_l2417_241798


namespace NUMINAMATH_GPT_expanded_figure_perimeter_l2417_241740

def side_length : ℕ := 2
def bottom_row_squares : ℕ := 3
def total_squares : ℕ := 4

def perimeter (side_length : ℕ) (bottom_row_squares : ℕ) (total_squares: ℕ) : ℕ :=
  2 * side_length * (bottom_row_squares + 1)

theorem expanded_figure_perimeter : perimeter side_length bottom_row_squares total_squares = 20 :=
by
  sorry

end NUMINAMATH_GPT_expanded_figure_perimeter_l2417_241740


namespace NUMINAMATH_GPT_prime_square_minus_five_not_div_by_eight_l2417_241713

theorem prime_square_minus_five_not_div_by_eight (p : ℕ) (prime_p : Prime p) (p_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) :=
sorry

end NUMINAMATH_GPT_prime_square_minus_five_not_div_by_eight_l2417_241713


namespace NUMINAMATH_GPT_pyramid_coloring_methods_l2417_241789

theorem pyramid_coloring_methods : 
  ∀ (P A B C D : ℕ),
    (P ≠ A) ∧ (P ≠ B) ∧ (P ≠ C) ∧ (P ≠ D) ∧
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
    (P < 5) ∧ (A < 5) ∧ (B < 5) ∧ (C < 5) ∧ (D < 5) →
  ∃! (num_methods : ℕ), num_methods = 420 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_coloring_methods_l2417_241789


namespace NUMINAMATH_GPT_smallest_number_of_rectangles_needed_l2417_241700

-- Define the dimensions of the rectangle
def rectangle_area (length width : ℕ) : ℕ := length * width

-- Define the side length of the square
def square_side_length : ℕ := 12

-- Define the number of rectangles needed to cover the square horizontally
def num_rectangles_to_cover_square : ℕ := (square_side_length / 3) * (square_side_length / 4)

-- The theorem must state the total number of rectangles required
theorem smallest_number_of_rectangles_needed : num_rectangles_to_cover_square = 16 := 
by
  -- Proof details are skipped using sorry
  sorry

end NUMINAMATH_GPT_smallest_number_of_rectangles_needed_l2417_241700


namespace NUMINAMATH_GPT_find_positive_integer_solutions_l2417_241794

theorem find_positive_integer_solutions :
  ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ (1 / (a : ℚ)) - (1 / (b : ℚ)) = 1 / 37 ∧ (a, b) = (38, 1332) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_solutions_l2417_241794


namespace NUMINAMATH_GPT_speed_of_first_car_l2417_241768

theorem speed_of_first_car 
  (distance_highway : ℕ)
  (time_to_meet : ℕ)
  (speed_second_car : ℕ)
  (total_distance_covered : distance_highway = time_to_meet * 40 + time_to_meet * speed_second_car): 
  5 * 40 + 5 * 60 = distance_highway := 
by
  /-
    Given:
      - distance_highway : ℕ (The length of the highway, which is 500 miles)
      - time_to_meet : ℕ (The time after which the two cars meet, which is 5 hours)
      - speed_second_car : ℕ (The speed of the second car, which is 60 mph)
      - total_distance_covered : distance_highway = time_to_meet * speed_of_first_car + time_to_meet * speed_second_car

    We need to prove:
      - 5 * 40 + 5 * 60 = distance_highway
  -/

  sorry

end NUMINAMATH_GPT_speed_of_first_car_l2417_241768


namespace NUMINAMATH_GPT_function_properties_l2417_241726

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 + x) + log (2 - x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ ⦃a b : ℝ⦄, 0 < a → a < b → b < 2 → f b < f a) := by
  sorry

end NUMINAMATH_GPT_function_properties_l2417_241726


namespace NUMINAMATH_GPT_quadrilateral_is_square_l2417_241782

-- Define a structure for a quadrilateral with side lengths and diagonal lengths
structure Quadrilateral :=
  (side_a side_b side_c side_d diag_e diag_f : ℝ)

-- Define what it means for a quadrilateral to be a square
def is_square (quad : Quadrilateral) : Prop :=
  quad.side_a = quad.side_b ∧ 
  quad.side_b = quad.side_c ∧ 
  quad.side_c = quad.side_d ∧  
  quad.diag_e = quad.diag_f

-- Define the problem to prove that the given quadrilateral is a square given the conditions
theorem quadrilateral_is_square (quad : Quadrilateral) 
  (h_sides : quad.side_a = quad.side_b ∧ 
             quad.side_b = quad.side_c ∧ 
             quad.side_c = quad.side_d)
  (h_diagonals : quad.diag_e = quad.diag_f) :
  is_square quad := 
  by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_quadrilateral_is_square_l2417_241782


namespace NUMINAMATH_GPT_colorful_triangle_in_complete_graph_l2417_241717

open SimpleGraph

theorem colorful_triangle_in_complete_graph (n : ℕ) (h : n ≥ 3) (colors : Fin n → Fin n → Fin (n - 1)) :
  ∃ (u v w : Fin n), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ colors u v ≠ colors v w ∧ colors v w ≠ colors w u ∧ colors w u ≠ colors u v :=
  sorry

end NUMINAMATH_GPT_colorful_triangle_in_complete_graph_l2417_241717


namespace NUMINAMATH_GPT_least_positive_integer_divisible_by_three_primes_l2417_241718

-- Define the next three distinct primes larger than 5
def prime1 := 7
def prime2 := 11
def prime3 := 13

-- Define the product of these primes
def prod := prime1 * prime2 * prime3

-- Statement of the theorem
theorem least_positive_integer_divisible_by_three_primes : prod = 1001 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_divisible_by_three_primes_l2417_241718


namespace NUMINAMATH_GPT_find_number_of_members_l2417_241751

variable (n : ℕ)

-- We translate the conditions into Lean 4 definitions
def total_collection := 9216
def per_member_contribution := n

-- The goal is to prove that n = 96 given the total collection
theorem find_number_of_members (h : n * n = total_collection) : n = 96 := 
sorry

end NUMINAMATH_GPT_find_number_of_members_l2417_241751


namespace NUMINAMATH_GPT_find_A_l2417_241712

variable (x ω φ b A : ℝ)

-- Given conditions
axiom cos_squared_eq : 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b
axiom A_gt_zero : A > 0

-- Lean 4 statement to prove
theorem find_A : A = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l2417_241712


namespace NUMINAMATH_GPT_max_distance_from_ellipse_to_line_l2417_241763

theorem max_distance_from_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 4) = 1
  let line (x y : ℝ) := x + 2 * y - Real.sqrt 2 = 0
  ∃ (d : ℝ), (∀ (x y : ℝ), ellipse x y → line x y → d = Real.sqrt 10) :=
sorry

end NUMINAMATH_GPT_max_distance_from_ellipse_to_line_l2417_241763


namespace NUMINAMATH_GPT_find_A_l2417_241777

variable {A B C : ℚ}

theorem find_A (h1 : A = 1/2 * B) (h2 : B = 3/4 * C) (h3 : A + C = 55) : A = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l2417_241777


namespace NUMINAMATH_GPT_Iris_total_spent_l2417_241714

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end NUMINAMATH_GPT_Iris_total_spent_l2417_241714


namespace NUMINAMATH_GPT_hours_per_day_l2417_241749

-- Conditions
def days_worked : ℝ := 3
def total_hours_worked : ℝ := 7.5

-- Theorem to prove the number of hours worked each day
theorem hours_per_day : total_hours_worked / days_worked = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_l2417_241749


namespace NUMINAMATH_GPT_mangoes_harvested_l2417_241783

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) (total_mangoes_distributed : ℕ) (total_mangoes : ℕ) :
  neighbors = 8 ∧ mangoes_per_neighbor = 35 ∧ total_mangoes_distributed = neighbors * mangoes_per_neighbor ∧ total_mangoes = 2 * total_mangoes_distributed →
  total_mangoes = 560 :=
by {
  sorry
}

end NUMINAMATH_GPT_mangoes_harvested_l2417_241783


namespace NUMINAMATH_GPT_total_games_in_conference_l2417_241781

-- Definitions based on the conditions
def numTeams := 16
def divisionTeams := 8
def gamesWithinDivisionPerTeam := 21
def gamesAcrossDivisionPerTeam := 16
def totalGamesPerTeam := 37
def totalGameCount := 592
def actualGameCount := 296

-- Proof statement
theorem total_games_in_conference : actualGameCount = (totalGameCount / 2) :=
  by sorry

end NUMINAMATH_GPT_total_games_in_conference_l2417_241781


namespace NUMINAMATH_GPT_overall_average_marks_l2417_241741

theorem overall_average_marks 
  (num_candidates : ℕ) 
  (num_passed : ℕ) 
  (avg_passed : ℕ) 
  (avg_failed : ℕ)
  (h1 : num_candidates = 120) 
  (h2 : num_passed = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (num_passed * avg_passed + (num_candidates - num_passed) * avg_failed) / num_candidates = 35 := 
by
  sorry

end NUMINAMATH_GPT_overall_average_marks_l2417_241741


namespace NUMINAMATH_GPT_not_possible_total_47_l2417_241764

open Nat

theorem not_possible_total_47 (h c : ℕ) : ¬ (13 * h + 5 * c = 47) :=
  sorry

end NUMINAMATH_GPT_not_possible_total_47_l2417_241764


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2417_241787

theorem sufficient_but_not_necessary (x : ℝ) : (x = -1 → x^2 - 5 * x - 6 = 0) ∧ (∃ y : ℝ, y ≠ -1 ∧ y^2 - 5 * y - 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2417_241787


namespace NUMINAMATH_GPT_quadratic_solution_1_l2417_241709

theorem quadratic_solution_1 :
  (∃ x, x^2 - 4 * x + 3 = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

end NUMINAMATH_GPT_quadratic_solution_1_l2417_241709


namespace NUMINAMATH_GPT_system_of_equations_has_integer_solutions_l2417_241706

theorem system_of_equations_has_integer_solutions (a b : ℤ) :
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_has_integer_solutions_l2417_241706


namespace NUMINAMATH_GPT_power_division_l2417_241756

theorem power_division : (19^11 / 19^6 = 247609) := sorry

end NUMINAMATH_GPT_power_division_l2417_241756


namespace NUMINAMATH_GPT_count_valid_x_satisfying_heartsuit_condition_l2417_241766

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem count_valid_x_satisfying_heartsuit_condition :
  (∃ n, ∀ x, 1 ≤ x ∧ x < 1000 → digit_sum (digit_sum x) = 4 → n = 36) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_x_satisfying_heartsuit_condition_l2417_241766


namespace NUMINAMATH_GPT_min_value_of_expression_l2417_241772

theorem min_value_of_expression (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y ≤ 2) :
  ∃ m : ℝ, m = (2 / (x + 3 * y) + 1 / (x - y)) ∧ m = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2417_241772


namespace NUMINAMATH_GPT_sequence_two_cases_l2417_241724

noncomputable def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≥ a (n-1)) ∧  -- nondecreasing
  (∃ n m, n ≠ m ∧ a n ≠ a m) ∧  -- nonconstant
  (∀ n, a n ∣ n^2)  -- a_n | n^2

theorem sequence_two_cases (a : ℕ → ℕ) :
  sequence_property a →
  (∃ n1, ∀ n ≥ n1, a n = n) ∨ (∃ n2, ∀ n ≥ n2, a n = n^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_two_cases_l2417_241724


namespace NUMINAMATH_GPT_percentage_calculation_l2417_241702

variable (x : Real)
variable (hx : x > 0)

theorem percentage_calculation : 
  ∃ p : Real, p = (0.18 * x) / (x + 20) * 100 :=
sorry

end NUMINAMATH_GPT_percentage_calculation_l2417_241702


namespace NUMINAMATH_GPT_general_term_l2417_241775

open Nat

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

theorem general_term (n : ℕ) (hn : n > 0) : (S n - S (n - 1)) = 4 * n - 5 := by
  sorry

end NUMINAMATH_GPT_general_term_l2417_241775


namespace NUMINAMATH_GPT_gcd_g_50_52_l2417_241790

/-- Define the polynomial function g -/
def g (x : ℤ) : ℤ := x^2 - 3 * x + 2023

/-- The theorem stating the gcd of g(50) and g(52) -/
theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_g_50_52_l2417_241790


namespace NUMINAMATH_GPT_weight_difference_l2417_241754

-- Defining the weights of the individuals
variables (a b c d e : ℝ)

-- Given conditions as hypotheses
def conditions :=
  (a = 75) ∧
  ((a + b + c) / 3 = 84) ∧
  ((a + b + c + d) / 4 = 80) ∧
  ((b + c + d + e) / 4 = 79)

-- Theorem statement to prove the desired result
theorem weight_difference (h : conditions a b c d e) : e - d = 3 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_l2417_241754


namespace NUMINAMATH_GPT_exponent_product_l2417_241701

variables {a m n : ℝ}

theorem exponent_product (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 :=
by
  sorry

end NUMINAMATH_GPT_exponent_product_l2417_241701


namespace NUMINAMATH_GPT_proof_problem_l2417_241732

open Real

noncomputable def p : Prop := ∃ x : ℝ, x - 2 > log x / log 10
noncomputable def q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem :
  (p ∧ ¬q) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2417_241732


namespace NUMINAMATH_GPT_hexagon_area_l2417_241758

-- Define the area of a triangle
def triangle_area (base height: ℝ) : ℝ := 0.5 * base * height

-- Given dimensions for each triangle
def base_unit := 1
def original_height := 3
def new_height := 4

-- Calculate areas of each triangle in the new configuration
def single_triangle_area := triangle_area base_unit new_height
def total_triangle_area := 4 * single_triangle_area

-- The area of the rectangular region formed by the hexagon and triangles
def rectangular_region_area := (base_unit + original_height + original_height) * new_height

-- Prove the area of the hexagon
theorem hexagon_area : rectangular_region_area - total_triangle_area = 32 :=
by
  -- We will provide the proof here
  sorry

end NUMINAMATH_GPT_hexagon_area_l2417_241758


namespace NUMINAMATH_GPT_length_ratio_is_correct_width_ratio_is_correct_l2417_241745

-- Definitions based on the conditions
def room_length : ℕ := 25
def room_width : ℕ := 15

-- Calculated perimeter
def room_perimeter : ℕ := 2 * (room_length + room_width)

-- Ratios to be proven
def length_to_perimeter_ratio : ℚ := room_length / room_perimeter
def width_to_perimeter_ratio : ℚ := room_width / room_perimeter

-- Stating the theorems to be proved
theorem length_ratio_is_correct : length_to_perimeter_ratio = 5 / 16 :=
by sorry

theorem width_ratio_is_correct : width_to_perimeter_ratio = 3 / 16 :=
by sorry

end NUMINAMATH_GPT_length_ratio_is_correct_width_ratio_is_correct_l2417_241745


namespace NUMINAMATH_GPT_eleven_power_five_mod_nine_l2417_241773

theorem eleven_power_five_mod_nine : ∃ n : ℕ, (11^5 ≡ n [MOD 9]) ∧ (0 ≤ n ∧ n < 9) ∧ (n = 5) := 
  by 
    sorry

end NUMINAMATH_GPT_eleven_power_five_mod_nine_l2417_241773


namespace NUMINAMATH_GPT_multiples_33_between_1_and_300_l2417_241704

theorem multiples_33_between_1_and_300 : ∃ (x : ℕ), (∀ n : ℕ, n ≤ 300 → n % x = 0 → n / x ≤ 33) ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_multiples_33_between_1_and_300_l2417_241704


namespace NUMINAMATH_GPT_productivity_increase_l2417_241779

/-- 
The original workday is 8 hours. 
During the first 6 hours, productivity is at the planned level (1 unit/hour). 
For the next 2 hours, productivity falls by 25% (0.75 units/hour). 
The workday is extended by 1 hour (now 9 hours). 
During the first 6 hours of the extended shift, productivity remains at the planned level (1 unit/hour). 
For the remaining 3 hours of the extended shift, productivity falls by 30% (0.7 units/hour). 
Prove that the overall productivity for the shift increased by 8% as a result of extending the workday.
-/
theorem productivity_increase
  (planned_productivity : ℝ)
  (initial_work_hours : ℝ)
  (initial_productivity_drop : ℝ)
  (extended_work_hours : ℝ)
  (extended_productivity_drop : ℝ)
  (initial_total_work : ℝ)
  (extended_total_work : ℝ)
  (percentage_increase : ℝ) :
  planned_productivity = 1 →
  initial_work_hours = 8 →
  initial_productivity_drop = 0.25 →
  extended_work_hours = 9 →
  extended_productivity_drop = 0.30 →
  initial_total_work = 7.5 →
  extended_total_work = 8.1 →
  percentage_increase = 8 →
  ((extended_total_work - initial_total_work) / initial_total_work * 100) = percentage_increase :=
sorry

end NUMINAMATH_GPT_productivity_increase_l2417_241779


namespace NUMINAMATH_GPT_alpha_beta_value_l2417_241728

theorem alpha_beta_value :
  ∃ α β : ℝ, (α^2 - 2 * α - 4 = 0) ∧ (β^2 - 2 * β - 4 = 0) ∧ (α + β = 2) ∧ (α^3 + 8 * β + 6 = 30) :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_value_l2417_241728


namespace NUMINAMATH_GPT_reflection_correct_l2417_241797

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def M : point := (3, 2)

theorem reflection_correct : reflect_x_axis M = (3, -2) :=
  sorry

end NUMINAMATH_GPT_reflection_correct_l2417_241797


namespace NUMINAMATH_GPT_solve_diophantine_equation_l2417_241784

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end NUMINAMATH_GPT_solve_diophantine_equation_l2417_241784


namespace NUMINAMATH_GPT_present_age_ratio_l2417_241755

theorem present_age_ratio (D J : ℕ) (h1 : Dan = 24) (h2 : James = 20) : Dan / James = 6 / 5 := by
  sorry

end NUMINAMATH_GPT_present_age_ratio_l2417_241755


namespace NUMINAMATH_GPT_finite_pos_int_set_condition_l2417_241757

theorem finite_pos_int_set_condition (X : Finset ℕ) 
  (hX : ∀ a ∈ X, 0 < a) 
  (h2 : 2 ≤ X.card) 
  (hcond : ∀ {a b : ℕ}, a ∈ X → b ∈ X → a > b → b^2 / (a - b) ∈ X) :
  ∃ a : ℕ, X = {a, 2 * a} :=
by
  sorry

end NUMINAMATH_GPT_finite_pos_int_set_condition_l2417_241757


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2417_241710

theorem geometric_sequence_common_ratio (a1 a2 a3 : ℤ) (r : ℤ)
  (h1 : a1 = 9) (h2 : a2 = -18) (h3 : a3 = 36) (h4 : a2 / a1 = r) (h5 : a3 = a2 * r) :
  r = -2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2417_241710


namespace NUMINAMATH_GPT_g_diff_eq_neg8_l2417_241733

noncomputable def g : ℝ → ℝ := sorry

axiom linear_g : ∀ x y : ℝ, g (x + y) = g x + g y

axiom condition_g : ∀ x : ℝ, g (x + 2) - g x = 4

theorem g_diff_eq_neg8 : g 2 - g 6 = -8 :=
by
  sorry

end NUMINAMATH_GPT_g_diff_eq_neg8_l2417_241733


namespace NUMINAMATH_GPT_forty_percent_of_jacquelines_candy_bars_is_120_l2417_241723

-- Define the number of candy bars Fred has
def fred_candy_bars : ℕ := 12

-- Define the number of candy bars Uncle Bob has
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6

-- Define the total number of candy bars Fred and Uncle Bob have together
def total_candy_bars : ℕ := fred_candy_bars + uncle_bob_candy_bars

-- Define the number of candy bars Jacqueline has
def jacqueline_candy_bars : ℕ := 10 * total_candy_bars

-- Define the number of candy bars that is 40% of Jacqueline's total
def forty_percent_jacqueline_candy_bars : ℕ := (40 * jacqueline_candy_bars) / 100

-- The statement to prove
theorem forty_percent_of_jacquelines_candy_bars_is_120 :
  forty_percent_jacqueline_candy_bars = 120 :=
sorry

end NUMINAMATH_GPT_forty_percent_of_jacquelines_candy_bars_is_120_l2417_241723


namespace NUMINAMATH_GPT_teacups_count_l2417_241774

theorem teacups_count (total_people teacup_capacity : ℕ) (H1 : total_people = 63) (H2 : teacup_capacity = 9) : total_people / teacup_capacity = 7 :=
by
  sorry

end NUMINAMATH_GPT_teacups_count_l2417_241774


namespace NUMINAMATH_GPT_find_x_given_y_l2417_241739

-- Given that x and y are always positive and x^2 and y vary inversely.
-- i.e., we have a relationship x^2 * y = k for a constant k,
-- and given that y = 8 when x = 3, find the value of x when y = 648.

theorem find_x_given_y
  (x y : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_inv : ∀ x y, x^2 * y = 72)
  (h_y : y = 648) : x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_given_y_l2417_241739


namespace NUMINAMATH_GPT_fraction_value_l2417_241720

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 5) (h3 : (∃ m : ℤ, x = m * y)) : x / y = -2 :=
sorry

end NUMINAMATH_GPT_fraction_value_l2417_241720


namespace NUMINAMATH_GPT_square_perimeter_l2417_241750

theorem square_perimeter (area : ℝ) (h : area = 144) : ∃ perimeter : ℝ, perimeter = 48 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_l2417_241750


namespace NUMINAMATH_GPT_smallest_n_l2417_241721

theorem smallest_n (n : ℕ) (h1 : n % 6 = 5) (h2 : n % 7 = 4) (h3 : n > 20) : n = 53 :=
sorry

end NUMINAMATH_GPT_smallest_n_l2417_241721


namespace NUMINAMATH_GPT_money_left_l2417_241715

noncomputable def olivia_money : ℕ := 112
noncomputable def nigel_money : ℕ := 139
noncomputable def ticket_cost : ℕ := 28
noncomputable def num_tickets : ℕ := 6

theorem money_left : (olivia_money + nigel_money - ticket_cost * num_tickets) = 83 :=
by
  sorry

end NUMINAMATH_GPT_money_left_l2417_241715


namespace NUMINAMATH_GPT_average_of_two_numbers_l2417_241791

theorem average_of_two_numbers (A B C : ℝ) (h1 : (A + B + C)/3 = 48) (h2 : C = 32) : (A + B)/2 = 56 := by
  sorry

end NUMINAMATH_GPT_average_of_two_numbers_l2417_241791


namespace NUMINAMATH_GPT_compare_abc_l2417_241765

noncomputable def a := Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def b := Real.cos (Real.pi / 6) ^ 2 - Real.sin (Real.pi / 6) ^ 2
noncomputable def c := Real.tan (30 * Real.pi / 180) / (1 - Real.tan (30 * Real.pi / 180) ^ 2)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l2417_241765


namespace NUMINAMATH_GPT_find_n_l2417_241761

theorem find_n :
  let a := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b := (2 * n : ℕ)
  (a*a - b*b = 0) -> (n = 12) := 
by 
  let a := 24
  let b := 2*n
  sorry

end NUMINAMATH_GPT_find_n_l2417_241761


namespace NUMINAMATH_GPT_max_y_difference_l2417_241748

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference : 
  ∃ x1 x2 : ℝ, 
    f x1 = g x1 ∧ f x2 = g x2 ∧ 
    (∀ x : ℝ, f x = g x → x = x1 ∨ x = x2) ∧ 
    abs ((f x1) - (f x2)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_max_y_difference_l2417_241748


namespace NUMINAMATH_GPT_Derrick_yard_length_l2417_241742

variables (Alex_yard Derrick_yard Brianne_yard Carla_yard Derek_yard : ℝ)

-- Given conditions as hypotheses
theorem Derrick_yard_length :
  (Alex_yard = Derrick_yard / 2) →
  (Brianne_yard = 6 * Alex_yard) →
  (Carla_yard = 3 * Brianne_yard + 5) →
  (Derek_yard = Carla_yard / 2 - 10) →
  (Brianne_yard = 30) →
  Derrick_yard = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Derrick_yard_length_l2417_241742


namespace NUMINAMATH_GPT_sum_real_imaginary_part_l2417_241767

noncomputable def imaginary_unit : ℂ := Complex.I

theorem sum_real_imaginary_part {z : ℂ} (h : z * imaginary_unit = 1 + imaginary_unit) :
  z.re + z.im = 2 := 
sorry

end NUMINAMATH_GPT_sum_real_imaginary_part_l2417_241767


namespace NUMINAMATH_GPT_toy_spending_ratio_l2417_241708

theorem toy_spending_ratio :
  ∃ T : ℝ, 204 - T > 0 ∧ 51 = (204 - T) / 2 ∧ (T / 204) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_toy_spending_ratio_l2417_241708


namespace NUMINAMATH_GPT_animal_legs_l2417_241759

theorem animal_legs (dogs chickens spiders octopus : Nat) (legs_dog legs_chicken legs_spider legs_octopus : Nat)
  (h1 : dogs = 3)
  (h2 : chickens = 4)
  (h3 : spiders = 2)
  (h4 : octopus = 1)
  (h5 : legs_dog = 4)
  (h6 : legs_chicken = 2)
  (h7 : legs_spider = 8)
  (h8 : legs_octopus = 8) :
  dogs * legs_dog + chickens * legs_chicken + spiders * legs_spider + octopus * legs_octopus = 44 := by
    sorry

end NUMINAMATH_GPT_animal_legs_l2417_241759


namespace NUMINAMATH_GPT_units_digit_27_mul_46_l2417_241716

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end NUMINAMATH_GPT_units_digit_27_mul_46_l2417_241716


namespace NUMINAMATH_GPT_Doris_needs_3_weeks_l2417_241788

-- Definitions based on conditions
def hourly_wage : ℕ := 20
def monthly_expenses : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturdays_hours : ℕ := 5
def weekdays_per_week : ℕ := 5

-- Total hours per week
def total_hours_per_week := (weekday_hours_per_day * weekdays_per_week) + saturdays_hours

-- Weekly earnings
def weekly_earnings := hourly_wage * total_hours_per_week

-- Number of weeks needed for monthly expenses
def weeks_needed := monthly_expenses / weekly_earnings

-- Proposition to prove
theorem Doris_needs_3_weeks :
  weeks_needed = 3 := 
by
  sorry

end NUMINAMATH_GPT_Doris_needs_3_weeks_l2417_241788


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2417_241727
-- Lean 4 Proof Statement


theorem arithmetic_sequence_common_difference 
  (a : ℕ) (n : ℕ) (d : ℕ) (S_n : ℕ) (a_n : ℕ) 
  (h1 : a = 2) 
  (h2 : a_n = 29) 
  (h3 : S_n = 155) 
  (h4 : S_n = n * (a + a_n) / 2) 
  (h5 : a_n = a + (n - 1) * d) 
  : d = 3 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2417_241727


namespace NUMINAMATH_GPT_red_balloon_probability_l2417_241753

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end NUMINAMATH_GPT_red_balloon_probability_l2417_241753


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2417_241793

-- Define complex number and evaluate it
noncomputable def z : ℂ := (2 - (1 : ℂ) * Complex.I) / (1 + (1 : ℂ) * Complex.I)

-- Prove that the complex number z lies in the fourth quadrant
theorem point_in_fourth_quadrant (hz: z = (1/2 : ℂ) - (3/2 : ℂ) * Complex.I) : z.im < 0 ∧ z.re > 0 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2417_241793


namespace NUMINAMATH_GPT_min_value_frac_l2417_241707

open Real

theorem min_value_frac (a b : ℝ) (h1 : a + b = 1/2) (h2 : a > 0) (h3 : b > 0) :
    (4 / a + 1 / b) = 18 :=
sorry

end NUMINAMATH_GPT_min_value_frac_l2417_241707


namespace NUMINAMATH_GPT_smallest_x_multiple_of_53_l2417_241738

theorem smallest_x_multiple_of_53 :
  ∃ (x : ℕ), (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
sorry

end NUMINAMATH_GPT_smallest_x_multiple_of_53_l2417_241738


namespace NUMINAMATH_GPT_range_of_a_l2417_241734

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ a > 3 ∨ a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2417_241734


namespace NUMINAMATH_GPT_ducks_at_Lake_Michigan_l2417_241785

variable (D : ℕ)

def ducks_condition := 2 * D + 6 = 206

theorem ducks_at_Lake_Michigan (h : ducks_condition D) : D = 100 :=
by
  sorry

end NUMINAMATH_GPT_ducks_at_Lake_Michigan_l2417_241785


namespace NUMINAMATH_GPT_p_implies_q_l2417_241705

theorem p_implies_q (x : ℝ) (h : |5 * x - 1| > 4) : x^2 - (3/2) * x + (1/2) > 0 := sorry

end NUMINAMATH_GPT_p_implies_q_l2417_241705


namespace NUMINAMATH_GPT_odd_function_condition_l2417_241743

noncomputable def f (x a b : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) ↔ (a = 0 ∧ b = 0) := 
by
  sorry

end NUMINAMATH_GPT_odd_function_condition_l2417_241743


namespace NUMINAMATH_GPT_multiple_of_Roseville_population_l2417_241747

noncomputable def Willowdale_population : ℕ := 2000

noncomputable def Roseville_population : ℕ :=
  (3 * Willowdale_population) - 500

noncomputable def SunCity_population : ℕ := 12000

theorem multiple_of_Roseville_population :
  ∃ m : ℕ, SunCity_population = (m * Roseville_population) + 1000 ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_Roseville_population_l2417_241747


namespace NUMINAMATH_GPT_sum_of_angles_eq_62_l2417_241771

noncomputable def Φ (x : ℝ) : ℝ := Real.sin x
noncomputable def Ψ (x : ℝ) : ℝ := Real.cos x
def θ : List ℝ := [31, 30, 1, 0]

theorem sum_of_angles_eq_62 :
  θ.sum = 62 := by
  sorry

end NUMINAMATH_GPT_sum_of_angles_eq_62_l2417_241771


namespace NUMINAMATH_GPT_soccer_games_total_l2417_241731

variable (wins losses ties total_games : ℕ)

theorem soccer_games_total
    (h1 : losses = 9)
    (h2 : 4 * wins + 3 * losses + ties = 8 * total_games) :
    total_games = 24 :=
by
  sorry

end NUMINAMATH_GPT_soccer_games_total_l2417_241731


namespace NUMINAMATH_GPT_smallest_d_l2417_241719

theorem smallest_d (d : ℝ) : 
  (∃ d, 2 * d = Real.sqrt ((4 * Real.sqrt 3) ^ 2 + (d + 4) ^ 2)) →
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_l2417_241719


namespace NUMINAMATH_GPT_range_of_a_l2417_241737

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l2417_241737


namespace NUMINAMATH_GPT_power_of_two_l2417_241744

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) 
  (hprime_divisors : ∀ p : ℕ, p.Prime → (p ∣ b ^ m - 1 ↔ p ∣ b ^ n - 1)) : 
  ∃ k : ℕ, b + 1 = 2 ^ k :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_l2417_241744


namespace NUMINAMATH_GPT_degree_to_radian_radian_to_degree_l2417_241746

theorem degree_to_radian (d : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (d = 210) → rad = (π / 180) → d * rad = 7 * π / 6 :=
by sorry 

theorem radian_to_degree (r : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (r = -5 * π / 2) → deg = (180 / π) → r * deg = -450 :=
by sorry

end NUMINAMATH_GPT_degree_to_radian_radian_to_degree_l2417_241746


namespace NUMINAMATH_GPT_average_marks_correct_l2417_241796

-- Define the marks obtained in each subject
def english_marks := 86
def mathematics_marks := 85
def physics_marks := 92
def chemistry_marks := 87
def biology_marks := 95

-- Calculate total marks and average marks
def total_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects := 5
def average_marks := total_marks / num_subjects

-- Prove that Dacid's average marks are 89
theorem average_marks_correct : average_marks = 89 := by
  sorry

end NUMINAMATH_GPT_average_marks_correct_l2417_241796


namespace NUMINAMATH_GPT_tan_22_5_expression_l2417_241736

theorem tan_22_5_expression :
  let a := 2
  let b := 1
  let c := 0
  let d := 0
  let t := Real.tan (Real.pi / 8)
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  t = (Real.sqrt a) - (Real.sqrt b) + (Real.sqrt c) - d → 
  a + b + c + d = 3 :=
by
  intros
  exact sorry

end NUMINAMATH_GPT_tan_22_5_expression_l2417_241736


namespace NUMINAMATH_GPT_total_value_is_84_l2417_241769

-- Definitions based on conditions
def number_of_stamps : ℕ := 21
def value_of_7_stamps : ℕ := 28
def stamps_per_7 : ℕ := 7
def stamp_value : ℤ := value_of_7_stamps / stamps_per_7
def total_value_of_collection : ℤ := number_of_stamps * stamp_value

-- Statement to prove the total value of the stamp collection
theorem total_value_is_84 : total_value_of_collection = 84 := by
  sorry

end NUMINAMATH_GPT_total_value_is_84_l2417_241769


namespace NUMINAMATH_GPT_range_k_domain_f_l2417_241725

theorem range_k_domain_f :
  (∀ x : ℝ, x^2 - 6*k*x + k + 8 ≥ 0) ↔ (-8/9 ≤ k ∧ k ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_k_domain_f_l2417_241725


namespace NUMINAMATH_GPT_problem_solution_l2417_241703

variable (a b c d m : ℝ)

-- Conditions
def opposite_numbers (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def absolute_value_eq (m : ℝ) : Prop := |m| = 3

theorem problem_solution
  (h1 : opposite_numbers a b)
  (h2 : reciprocals c d)
  (h3 : absolute_value_eq m) :
  (a + b) / 2023 - 4 * (c * d) + m^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2417_241703


namespace NUMINAMATH_GPT_find_1993_star_1935_l2417_241780

axiom star (x y : ℕ) : ℕ

axiom star_self {x : ℕ} : star x x = 0
axiom star_assoc {x y z : ℕ} : star x (star y z) = star x y + z

theorem find_1993_star_1935 : star 1993 1935 = 58 :=
by
  sorry

end NUMINAMATH_GPT_find_1993_star_1935_l2417_241780


namespace NUMINAMATH_GPT_circle_center_and_radius_l2417_241730

theorem circle_center_and_radius :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ 
    (x - C.1)^2 + (y - C.2)^2 = r^2) ∧ C = (1, -2) ∧ r = Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_l2417_241730


namespace NUMINAMATH_GPT_probability_of_point_in_spheres_l2417_241776

noncomputable def radius_of_inscribed_sphere (R : ℝ) : ℝ := 2 * R / 3
noncomputable def radius_of_tangent_spheres (R : ℝ) : ℝ := 2 * R / 3

theorem probability_of_point_in_spheres
  (R : ℝ)  -- Radius of the circumscribed sphere
  (r : ℝ := radius_of_inscribed_sphere R)  -- Radius of the inscribed sphere
  (r_t : ℝ := radius_of_tangent_spheres R)  -- Radius of each tangent sphere
  (volume : ℝ := 4/3 * Real.pi * r^3)  -- Volume of each smaller sphere
  (total_small_volume : ℝ := 5 * volume)  -- Total volume of smaller spheres
  (circumsphere_volume : ℝ := 4/3 * Real.pi * (2 * R)^3)  -- Volume of the circumscribed sphere
  : 
  total_small_volume / circumsphere_volume = 5 / 27 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_point_in_spheres_l2417_241776


namespace NUMINAMATH_GPT_total_capacity_l2417_241735

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end NUMINAMATH_GPT_total_capacity_l2417_241735


namespace NUMINAMATH_GPT_pencil_eraser_cost_l2417_241786

theorem pencil_eraser_cost (p e : ℕ) (h_eq : 10 * p + 4 * e = 120) (h_gt : p > e) : p + e = 15 :=
by sorry

end NUMINAMATH_GPT_pencil_eraser_cost_l2417_241786


namespace NUMINAMATH_GPT_find_n_from_lcm_gcf_l2417_241722

open scoped Classical

noncomputable def LCM (a b : ℕ) : ℕ := sorry
noncomputable def GCF (a b : ℕ) : ℕ := sorry

theorem find_n_from_lcm_gcf (n m : ℕ) (h1 : LCM n m = 48) (h2 : GCF n m = 18) (h3 : m = 16) : n = 54 :=
by sorry

end NUMINAMATH_GPT_find_n_from_lcm_gcf_l2417_241722


namespace NUMINAMATH_GPT_a_gt_b_l2417_241711

variable (n : ℕ) (a b : ℝ)
variable (n_pos : n > 1) (a_pos : 0 < a) (b_pos : 0 < b)
variable (a_eqn : a^n = a + 1)
variable (b_eqn : b^{2 * n} = b + 3 * a)

theorem a_gt_b : a > b :=
by {
  -- Proof is needed here
  sorry
}

end NUMINAMATH_GPT_a_gt_b_l2417_241711


namespace NUMINAMATH_GPT_find_second_number_l2417_241762

variable (n : ℕ)

theorem find_second_number (h : 8000 * n = 480 * 10^5) : n = 6000 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l2417_241762


namespace NUMINAMATH_GPT_find_second_number_l2417_241778

theorem find_second_number 
  (x y z : ℕ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 * y) / 4)
  (h3 : z = (9 * y) / 7) : 
  y = 40 :=
sorry

end NUMINAMATH_GPT_find_second_number_l2417_241778


namespace NUMINAMATH_GPT_number_of_lines_at_least_two_points_4_by_4_grid_l2417_241795

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end NUMINAMATH_GPT_number_of_lines_at_least_two_points_4_by_4_grid_l2417_241795


namespace NUMINAMATH_GPT_part1_part2_l2417_241760

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 1 < x ∧ x < 5 }
def setC (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 3 }

-- part (1)
theorem part1 (x : ℝ) : (x ∈ setA ∨ x ∈ setB) ↔ (-2 ≤ x ∧ x < 5) :=
sorry

-- part (2)
theorem part2 (a : ℝ) : ((setA ∩ setC a) = setC a) ↔ (a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2417_241760


namespace NUMINAMATH_GPT_area_of_rectangle_l2417_241752

noncomputable def length := 44.4
noncomputable def width := 29.6

theorem area_of_rectangle (h1 : width = 2 / 3 * length) (h2 : 2 * (length + width) = 148) : 
  (length * width) = 1314.24 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2417_241752


namespace NUMINAMATH_GPT_jeans_more_than_scarves_l2417_241792

def num_ties := 34
def num_belts := 40
def num_black_shirts := 63
def num_white_shirts := 42
def num_jeans := (2 / 3) * (num_black_shirts + num_white_shirts)
def num_scarves := (1 / 2) * (num_ties + num_belts)

theorem jeans_more_than_scarves : num_jeans - num_scarves = 33 := by
  sorry

end NUMINAMATH_GPT_jeans_more_than_scarves_l2417_241792


namespace NUMINAMATH_GPT_proposition_4_l2417_241770

theorem proposition_4 (x y ε : ℝ) (h1 : |x - 2| < ε) (h2 : |y - 2| < ε) : |x - y| < 2 * ε :=
by
  sorry

end NUMINAMATH_GPT_proposition_4_l2417_241770


namespace NUMINAMATH_GPT_sum_of_ages_l2417_241799

variables (P M Mo : ℕ)

def age_ratio_PM := 3 * M = 5 * P
def age_ratio_MMo := 3 * Mo = 5 * M
def age_difference := Mo = P + 64

theorem sum_of_ages : age_ratio_PM P M → age_ratio_MMo M Mo → age_difference P Mo → P + M + Mo = 196 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_ages_l2417_241799


namespace NUMINAMATH_GPT_boys_without_calculators_l2417_241729

theorem boys_without_calculators (total_boys total_students students_with_calculators girls_with_calculators : ℕ) 
    (h1 : total_boys = 20) 
    (h2 : total_students = 40) 
    (h3 : students_with_calculators = 30) 
    (h4 : girls_with_calculators = 18) : 
    (total_boys - (students_with_calculators - girls_with_calculators)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_boys_without_calculators_l2417_241729
