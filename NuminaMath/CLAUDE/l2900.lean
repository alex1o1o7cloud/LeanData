import Mathlib

namespace NUMINAMATH_CALUDE_cookies_in_box_graemes_cookies_l2900_290080

/-- Given a box that can hold a certain weight of cookies and cookies of a specific weight,
    calculate the number of cookies that can fit in the box. -/
theorem cookies_in_box (box_capacity : ℕ) (cookie_weight : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  let cookies_per_pound := ounces_per_pound / cookie_weight
  box_capacity * cookies_per_pound

/-- Prove that given a box that can hold 40 pounds of cookies, and each cookie weighing 2 ounces,
    the number of cookies that can fit in the box is equal to 320. -/
theorem graemes_cookies :
  cookies_in_box 40 2 16 = 320 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_box_graemes_cookies_l2900_290080


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l2900_290079

-- Define the sets P and Q
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q : (P.compl) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l2900_290079


namespace NUMINAMATH_CALUDE_square_area_l2900_290087

theorem square_area (side : ℝ) (h : side = 6) : side * side = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l2900_290087


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2900_290034

theorem simplify_and_rationalize : 
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) * (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2900_290034


namespace NUMINAMATH_CALUDE_line_points_relation_l2900_290025

/-- 
Given a line with equation x = 6y + 5, 
if two points (m, n) and (m + Q, n + p) lie on this line, 
and p = 1/3, then Q = 2.
-/
theorem line_points_relation (m n Q p : ℝ) : 
  (m = 6 * n + 5) →
  (m + Q = 6 * (n + p) + 5) →
  (p = 1/3) →
  Q = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_points_relation_l2900_290025


namespace NUMINAMATH_CALUDE_exists_valid_a_l2900_290046

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, a}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem exists_valid_a : ∃ a : ℝ, A a ⊆ B ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_a_l2900_290046


namespace NUMINAMATH_CALUDE_cricket_average_score_l2900_290024

theorem cricket_average_score 
  (total_matches : ℕ) 
  (matches_set1 : ℕ) 
  (matches_set2 : ℕ) 
  (avg_score_set1 : ℝ) 
  (avg_score_set2 : ℝ) 
  (h1 : total_matches = matches_set1 + matches_set2)
  (h2 : matches_set1 = 2)
  (h3 : matches_set2 = 3)
  (h4 : avg_score_set1 = 20)
  (h5 : avg_score_set2 = 30) :
  (matches_set1 * avg_score_set1 + matches_set2 * avg_score_set2) / total_matches = 26 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l2900_290024


namespace NUMINAMATH_CALUDE_calculate_expression_l2900_290070

theorem calculate_expression : (3752 / (39 * 2) + 5030 / (39 * 10) : ℚ) = 61 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2900_290070


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l2900_290071

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l2900_290071


namespace NUMINAMATH_CALUDE_elijah_coffee_pints_l2900_290009

-- Define the conversion rate from cups to pints
def cups_to_pints : ℚ := 1 / 2

-- Define the total amount of liquid consumed in cups
def total_liquid_cups : ℚ := 36

-- Define the amount of water Emilio drank in pints
def emilio_water_pints : ℚ := 9.5

-- Theorem statement
theorem elijah_coffee_pints :
  (total_liquid_cups * cups_to_pints) - emilio_water_pints = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_elijah_coffee_pints_l2900_290009


namespace NUMINAMATH_CALUDE_ram_selection_probability_l2900_290004

theorem ram_selection_probability
  (p_ravi : ℝ)
  (p_both : ℝ)
  (h_ravi : p_ravi = 1 / 5)
  (h_both : p_both = 0.05714285714285714)
  (h_independent : ∀ p_ram : ℝ, p_both = p_ram * p_ravi) :
  ∃ p_ram : ℝ, p_ram = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ram_selection_probability_l2900_290004


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2900_290001

theorem smallest_prime_dividing_sum : ∃ p : Nat, 
  Prime p ∧ p > 7 ∧ p ∣ (2^14 + 7^8) ∧ 
  ∀ q : Nat, Prime q → q ∣ (2^14 + 7^8) → q ≥ p :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2900_290001


namespace NUMINAMATH_CALUDE_acme_profit_calculation_l2900_290002

def initial_outlay : ℝ := 12450
def manufacturing_cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def marketing_expense_rate : ℝ := 0.05
def shipping_cost_rate : ℝ := 0.03
def number_of_sets : ℕ := 950

def revenue : ℝ := selling_price_per_set * number_of_sets
def total_manufacturing_cost : ℝ := initial_outlay + manufacturing_cost_per_set * number_of_sets
def additional_variable_costs : ℝ := (marketing_expense_rate + shipping_cost_rate) * revenue

def profit : ℝ := revenue - total_manufacturing_cost - additional_variable_costs

theorem acme_profit_calculation : profit = 11537.50 := by
  sorry

end NUMINAMATH_CALUDE_acme_profit_calculation_l2900_290002


namespace NUMINAMATH_CALUDE_smallest_difference_of_powers_eleven_is_representable_eleven_is_smallest_l2900_290064

theorem smallest_difference_of_powers : 
  ∀ k l : ℕ, 36^k - 5^l > 0 → 36^k - 5^l ≥ 11 :=
by sorry

theorem eleven_is_representable : 
  ∃ k l : ℕ, 36^k - 5^l = 11 :=
by sorry

theorem eleven_is_smallest :
  (∃ k l : ℕ, 36^k - 5^l = 11) ∧
  (∀ m n : ℕ, 36^m - 5^n > 0 → 36^m - 5^n ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_of_powers_eleven_is_representable_eleven_is_smallest_l2900_290064


namespace NUMINAMATH_CALUDE_construct_angles_l2900_290066

-- Define the given angle
def given_angle : ℝ := 40

-- Define the target angles
def target_angle_a : ℝ := 80
def target_angle_b : ℝ := 160
def target_angle_c : ℝ := 20

-- Theorem to prove the construction of target angles
theorem construct_angles :
  (given_angle + given_angle = target_angle_a) ∧
  (given_angle + given_angle + given_angle + given_angle = target_angle_b) ∧
  (180 - (given_angle + given_angle + given_angle + given_angle) = target_angle_c) :=
by sorry

end NUMINAMATH_CALUDE_construct_angles_l2900_290066


namespace NUMINAMATH_CALUDE_minimal_n_for_square_product_set_l2900_290052

theorem minimal_n_for_square_product_set (m : ℕ+) (p : ℕ) (h1 : p.Prime) (h2 : p ∣ m) 
  (h3 : p > Real.sqrt (2 * m) + 1) :
  ∃ (n : ℕ), n = m + p ∧
  (∀ (k : ℕ), k < n → 
    ¬∃ (S : Finset ℕ), 
      (∀ x ∈ S, m ≤ x ∧ x ≤ k) ∧ 
      (∃ y : ℕ, (S.prod id : ℕ) = y * y)) ∧
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, m ≤ x ∧ x ≤ n) ∧ 
    (∃ y : ℕ, (S.prod id : ℕ) = y * y) :=
by sorry

end NUMINAMATH_CALUDE_minimal_n_for_square_product_set_l2900_290052


namespace NUMINAMATH_CALUDE_compound_interest_principal_l2900_290047

/-- Proves that given specific compound interest conditions, the principal amount is 1500 --/
theorem compound_interest_principal :
  ∀ (CI R T P : ℝ),
    CI = 315 →
    R = 10 →
    T = 2 →
    CI = P * ((1 + R / 100) ^ T - 1) →
    P = 1500 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l2900_290047


namespace NUMINAMATH_CALUDE_M_factors_l2900_290045

/-- The number of positive integer factors of M, where
    M = 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1 -/
def M : ℕ := 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1

/-- The number of positive integer factors of a natural number n -/
def numFactors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem M_factors : numFactors M = 49 := by
  sorry

end NUMINAMATH_CALUDE_M_factors_l2900_290045


namespace NUMINAMATH_CALUDE_village_population_proof_l2900_290019

theorem village_population_proof (P : ℕ) : 
  (0.85 : ℝ) * ((0.90 : ℝ) * P) = 6514 → P = 8518 := by sorry

end NUMINAMATH_CALUDE_village_population_proof_l2900_290019


namespace NUMINAMATH_CALUDE_range_of_b_l2900_290022

theorem range_of_b (b : ℝ) : 
  (∀ a : ℝ, a ≤ -1 → a * 2 * b - b - 3 * a ≥ 0) → 
  b ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l2900_290022


namespace NUMINAMATH_CALUDE_evaluate_expression_l2900_290016

theorem evaluate_expression : (27^24) / (81^12) = 3^24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2900_290016


namespace NUMINAMATH_CALUDE_min_framing_for_picture_l2900_290097

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- Theorem stating the minimum framing needed for the given picture specifications. -/
theorem min_framing_for_picture :
  min_framing_feet 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_picture_l2900_290097


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2900_290098

/-- Represents an ellipse with the given equation and foci on the y-axis -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (4 - m) + y^2 / (m - 3) = 1
  foci_on_y_axis : True

/-- The range of m for a valid ellipse with foci on the y-axis -/
theorem ellipse_m_range (e : Ellipse) : 7/2 < e.m ∧ e.m < 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2900_290098


namespace NUMINAMATH_CALUDE_average_tomatoes_proof_l2900_290043

/-- The number of tomatoes reaped on day 1 -/
def day1_tomatoes : ℕ := 120

/-- The number of tomatoes reaped on day 2 -/
def day2_tomatoes : ℕ := day1_tomatoes + 50

/-- The number of tomatoes reaped on day 3 -/
def day3_tomatoes : ℕ := 2 * day2_tomatoes

/-- The number of tomatoes reaped on day 4 -/
def day4_tomatoes : ℕ := day1_tomatoes / 2

/-- The total number of tomatoes reaped over 4 days -/
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes + day4_tomatoes

/-- The number of days -/
def num_days : ℕ := 4

/-- The average number of tomatoes reaped per day -/
def average_tomatoes : ℚ := total_tomatoes / num_days

theorem average_tomatoes_proof : average_tomatoes = 172.5 := by
  sorry

end NUMINAMATH_CALUDE_average_tomatoes_proof_l2900_290043


namespace NUMINAMATH_CALUDE_homework_points_calculation_l2900_290053

theorem homework_points_calculation (total_points : ℕ) 
  (h1 : total_points = 265)
  (h2 : ∀ (test_points quiz_points : ℕ), test_points = 4 * quiz_points)
  (h3 : ∀ (quiz_points homework_points : ℕ), quiz_points = homework_points + 5) :
  ∃ (homework_points : ℕ), 
    homework_points = 40 ∧ 
    homework_points + (homework_points + 5) + 4 * (homework_points + 5) = total_points :=
by sorry

end NUMINAMATH_CALUDE_homework_points_calculation_l2900_290053


namespace NUMINAMATH_CALUDE_emily_necklaces_l2900_290086

def beads_per_necklace : ℕ := 5
def total_beads_used : ℕ := 20

theorem emily_necklaces :
  total_beads_used / beads_per_necklace = 4 :=
by sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2900_290086


namespace NUMINAMATH_CALUDE_congruent_triangles_x_value_l2900_290039

/-- Given two congruent triangles ABC and DEF, where ABC has sides 3, 4, and 5,
    and DEF has sides 3, 3x-2, and 2x+1, prove that x = 2. -/
theorem congruent_triangles_x_value (x : ℝ) : 
  let a₁ : ℝ := 3
  let b₁ : ℝ := 4
  let c₁ : ℝ := 5
  let a₂ : ℝ := 3
  let b₂ : ℝ := 3 * x - 2
  let c₂ : ℝ := 2 * x + 1
  (a₁ + b₁ + c₁ = a₂ + b₂ + c₂) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_x_value_l2900_290039


namespace NUMINAMATH_CALUDE_log_20_over_27_not_calculable_l2900_290011

-- Define the given logarithms
def log5 : ℝ := 0.6990
def log3 : ℝ := 0.4771

-- Define a function to represent the ability to calculate a logarithm
def can_calculate (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), x = f log5 log3

-- Theorem statement
theorem log_20_over_27_not_calculable :
  ¬(can_calculate (Real.log (20/27))) ∧
  (can_calculate (Real.log 225)) ∧
  (can_calculate (Real.log 750)) ∧
  (can_calculate (Real.log 0.03)) ∧
  (can_calculate (Real.log 9)) :=
sorry

end NUMINAMATH_CALUDE_log_20_over_27_not_calculable_l2900_290011


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2900_290028

def isSolutionSet (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, (x - 1) * f x < 0 ↔ x ∈ S

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 0 < x → x < y → f x < f y)
  (h_f2 : f 2 = 0) :
  isSolutionSet f (Set.Ioo (-2) 0 ∪ Set.Ioo 1 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2900_290028


namespace NUMINAMATH_CALUDE_cylindrical_fortress_pi_l2900_290026

/-- Given a cylindrical fortress with circumference 38 feet and height 11 feet,
    if its volume is calculated as V = (1/12) * (circumference^2 * height),
    then the implied value of π is 3. -/
theorem cylindrical_fortress_pi (circumference height : ℝ) (π : ℝ) : 
  circumference = 38 →
  height = 11 →
  (1/12) * (circumference^2 * height) = π * (circumference / (2 * π))^2 * height →
  π = 3 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_fortress_pi_l2900_290026


namespace NUMINAMATH_CALUDE_cos_sin_sum_zero_l2900_290040

theorem cos_sin_sum_zero (θ a : Real) (h : Real.cos (π / 6 - θ) = a) :
  Real.cos (5 * π / 6 + θ) + Real.sin (2 * π / 3 - θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_zero_l2900_290040


namespace NUMINAMATH_CALUDE_problem_1_l2900_290037

theorem problem_1 (x : ℝ) : 4 * x^2 * (x - 1/4) = 4 * x^3 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2900_290037


namespace NUMINAMATH_CALUDE_vacation_pictures_l2900_290006

theorem vacation_pictures (zoo museum amusement beach deleted : ℕ) 
  (h1 : zoo = 120)
  (h2 : museum = 34)
  (h3 : amusement = 25)
  (h4 : beach = 21)
  (h5 : deleted = 73) :
  zoo + museum + amusement + beach - deleted = 127 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l2900_290006


namespace NUMINAMATH_CALUDE_probability_two_red_one_blue_l2900_290054

/-- Represents a cube composed of smaller unit cubes -/
structure Cube where
  edge_length : ℕ

/-- Represents the painting state of a smaller cube -/
inductive PaintState
  | Unpainted
  | Red
  | Blue
  | RedAndBlue

/-- Represents a painted cube -/
structure PaintedCube where
  cube : Cube
  paint : Cube → PaintState

/-- Calculates the number of cubes with exactly two red faces and one blue face -/
def cubes_with_two_red_one_blue (c : PaintedCube) : ℕ := sorry

/-- Calculates the total number of unit cubes in a larger cube -/
def total_unit_cubes (c : Cube) : ℕ := c.edge_length ^ 3

/-- Theorem stating the probability of selecting a cube with two red faces and one blue face -/
theorem probability_two_red_one_blue (c : PaintedCube) 
  (h1 : c.cube.edge_length = 8)
  (h2 : ∀ (x : Cube), x.edge_length = 1 → c.paint x ≠ PaintState.Unpainted) 
  (h3 : ∃ (layer : ℕ), layer < c.cube.edge_length ∧ 
    ∀ (x : Cube), x.edge_length = 1 → 
      (∃ (i j : ℕ), i < c.cube.edge_length ∧ j < c.cube.edge_length ∧
        (i = layer ∨ i = c.cube.edge_length - 1 - layer ∨
         j = layer ∨ j = c.cube.edge_length - 1 - layer)) →
      c.paint x = PaintState.Blue) :
  (cubes_with_two_red_one_blue c : ℚ) / (total_unit_cubes c.cube : ℚ) = 3 / 32 := by sorry

end NUMINAMATH_CALUDE_probability_two_red_one_blue_l2900_290054


namespace NUMINAMATH_CALUDE_remaining_quarters_count_l2900_290027

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.5
def jeans_cost : ℚ := 11.5

def remaining_money : ℚ := initial_amount - (pizza_cost + soda_cost + jeans_cost)

def quarters_in_dollar : ℕ := 4

theorem remaining_quarters_count : 
  (remaining_money * quarters_in_dollar).floor = 97 := by sorry

end NUMINAMATH_CALUDE_remaining_quarters_count_l2900_290027


namespace NUMINAMATH_CALUDE_total_percentage_solid_colored_sum_solid_color_not_yellow_percentage_l2900_290029

/-- The percentage of marbles that are solid-colored -/
def solid_colored_percentage : ℝ := 0.90

/-- The percentage of marbles that have patterns -/
def patterned_percentage : ℝ := 0.10

/-- The percentage of red marbles among solid-colored marbles -/
def red_percentage : ℝ := 0.40

/-- The percentage of blue marbles among solid-colored marbles -/
def blue_percentage : ℝ := 0.30

/-- The percentage of green marbles among solid-colored marbles -/
def green_percentage : ℝ := 0.20

/-- The percentage of yellow marbles among solid-colored marbles -/
def yellow_percentage : ℝ := 0.10

/-- All marbles are either solid-colored or patterned -/
theorem total_percentage : solid_colored_percentage + patterned_percentage = 1 := by sorry

/-- The sum of percentages for all solid-colored marbles is 100% -/
theorem solid_colored_sum :
  red_percentage + blue_percentage + green_percentage + yellow_percentage = 1 := by sorry

/-- The percentage of marbles that are a solid color other than yellow is 81% -/
theorem solid_color_not_yellow_percentage :
  solid_colored_percentage * (red_percentage + blue_percentage + green_percentage) = 0.81 := by sorry

end NUMINAMATH_CALUDE_total_percentage_solid_colored_sum_solid_color_not_yellow_percentage_l2900_290029


namespace NUMINAMATH_CALUDE_partial_fraction_A_value_l2900_290076

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 20*x^3 + 147*x^2 - 490*x + 588

-- Define the partial fraction decomposition
def partial_fraction (A B C D x : ℝ) : Prop :=
  1 / p x = A / (x + 3) + B / (x - 4) + C / (x - 4)^2 + D / (x - 7)

-- Theorem statement
theorem partial_fraction_A_value :
  ∀ A B C D : ℝ, (∀ x : ℝ, partial_fraction A B C D x) → A = -1/490 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_A_value_l2900_290076


namespace NUMINAMATH_CALUDE_girls_in_college_l2900_290023

theorem girls_in_college (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 600)
  (h2 : boys_ratio = 8)
  (h3 : girls_ratio = 4) :
  (girls_ratio * total_students) / (boys_ratio + girls_ratio) = 200 :=
sorry

end NUMINAMATH_CALUDE_girls_in_college_l2900_290023


namespace NUMINAMATH_CALUDE_min_time_for_all_flashes_l2900_290090

/-- Represents the three possible colors of the lights -/
inductive Color
  | Red
  | Yellow
  | Green

/-- A flash is a sequence of three different colors -/
def Flash := { seq : Fin 3 → Color // ∀ i j, i ≠ j → seq i ≠ seq j }

/-- The number of different possible flashes -/
def numFlashes : Nat := 6

/-- Duration of one flash in seconds -/
def flashDuration : Nat := 3

/-- Interval between consecutive flashes in seconds -/
def intervalDuration : Nat := 3

/-- Theorem: The minimum time required to achieve all different flashes is 33 seconds -/
theorem min_time_for_all_flashes : 
  numFlashes * flashDuration + (numFlashes - 1) * intervalDuration = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_time_for_all_flashes_l2900_290090


namespace NUMINAMATH_CALUDE_heart_king_probability_l2900_290073

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a heart as the first card and a king as the second card -/
def prob_heart_then_king (deck : ℕ) (hearts : ℕ) (kings : ℕ) : ℚ :=
  (hearts / deck) * (kings / (deck - 1))

theorem heart_king_probability :
  prob_heart_then_king StandardDeck NumHearts NumKings = 1 / StandardDeck := by
  sorry

end NUMINAMATH_CALUDE_heart_king_probability_l2900_290073


namespace NUMINAMATH_CALUDE_prob_different_fruits_l2900_290084

/-- The number of fruit types available --/
def num_fruits : ℕ := 5

/-- The number of meals over two days --/
def num_meals : ℕ := 6

/-- The probability of choosing a specific fruit for all meals --/
def prob_same_fruit : ℚ := (1 / num_fruits) ^ num_meals

/-- The probability of eating at least two different kinds of fruit over two days --/
theorem prob_different_fruits : 
  1 - num_fruits * prob_same_fruit = 15620 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_fruits_l2900_290084


namespace NUMINAMATH_CALUDE_intersection_condition_l2900_290036

/-- The set A parameterized by m -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + m * p.1 - p.2 + 2 = 0}

/-- The set B -/
def B : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + 1 = 0}

/-- Theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2900_290036


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l2900_290092

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  /-- The length of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The radius of the circle -/
  r : ℝ
  /-- The circle is tangent to three sides -/
  tangent_to_sides : h = r
  /-- The circle passes through the midpoint of the diagonal -/
  passes_through_midpoint : r^2 = (w/2)^2 + (h/2)^2

/-- The area of the rectangle is √3 * r^2 -/
theorem tangent_circle_rectangle_area (rect : TangentCircleRectangle) : 
  rect.w * rect.h = Real.sqrt 3 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l2900_290092


namespace NUMINAMATH_CALUDE_rectangle_problem_l2900_290095

theorem rectangle_problem (l b : ℝ) : 
  l = 2 * b →
  (l - 5) * (b + 5) - l * b = 75 →
  20 < l ∧ l < 50 →
  10 < b ∧ b < 30 →
  l = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l2900_290095


namespace NUMINAMATH_CALUDE_negative_of_negative_greater_than_negative_of_positive_l2900_290057

theorem negative_of_negative_greater_than_negative_of_positive :
  -(-1) > -(2) := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_greater_than_negative_of_positive_l2900_290057


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l2900_290008

theorem tan_pi_minus_alpha (α : Real) 
  (h1 : α > π / 2) 
  (h2 : α < π) 
  (h3 : 3 * Real.cos (2 * α) - Real.sin α = 2) : 
  Real.tan (π - α) = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l2900_290008


namespace NUMINAMATH_CALUDE_remaining_black_cards_l2900_290065

/-- The number of black cards in a full deck of cards -/
def full_black_cards : ℕ := 26

/-- The number of black cards taken out from the deck -/
def removed_black_cards : ℕ := 5

/-- Theorem: The number of remaining black cards is 21 -/
theorem remaining_black_cards :
  full_black_cards - removed_black_cards = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_cards_l2900_290065


namespace NUMINAMATH_CALUDE_problem_solution_l2900_290091

theorem problem_solution : ∃ n : ℕ, 
  n = (2123 + 1787) * (6 * (2123 - 1787)) + 384 ∧ n = 7884144 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2900_290091


namespace NUMINAMATH_CALUDE_exists_desired_arrangement_l2900_290048

/-- A type representing a 10x10 grid of natural numbers -/
def Grid := Fin 10 → Fin 10 → ℕ

/-- A type representing a domino (1x2 rectangle) in the grid -/
inductive Domino
| horizontal : Fin 10 → Fin 9 → Domino
| vertical : Fin 9 → Fin 10 → Domino

/-- A partition of the grid into dominoes -/
def Partition := List Domino

/-- Function to check if a partition is valid (covers the entire grid without overlaps) -/
def isValidPartition (p : Partition) : Prop := sorry

/-- Function to calculate the sum of numbers in a domino for a given grid -/
def dominoSum (g : Grid) (d : Domino) : ℕ := sorry

/-- Function to count the number of dominoes with even sum in a partition -/
def countEvenSumDominoes (g : Grid) (p : Partition) : ℕ := sorry

/-- The main theorem statement -/
theorem exists_desired_arrangement : 
  ∃ (g : Grid), ∀ (p : Partition), isValidPartition p → countEvenSumDominoes g p = 7 := by sorry

end NUMINAMATH_CALUDE_exists_desired_arrangement_l2900_290048


namespace NUMINAMATH_CALUDE_carries_cake_profit_l2900_290072

/-- Calculates the profit for a cake decorator given their work hours, pay rate, and supply cost. -/
def cake_decorator_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) : ℕ :=
  hours_per_day * days_worked * hourly_rate - supply_cost

/-- Proves that Carrie's profit from decorating a wedding cake is $122. -/
theorem carries_cake_profit :
  cake_decorator_profit 2 4 22 54 = 122 := by
  sorry

end NUMINAMATH_CALUDE_carries_cake_profit_l2900_290072


namespace NUMINAMATH_CALUDE_male_average_height_l2900_290078

/-- Proves that the average height of males in a school is 185 cm given the following conditions:
  - The average height of all students is 180 cm
  - The average height of females is 170 cm
  - The ratio of men to women is 2:1
-/
theorem male_average_height (total_avg : ℝ) (female_avg : ℝ) (male_female_ratio : ℚ) :
  total_avg = 180 →
  female_avg = 170 →
  male_female_ratio = 2 →
  ∃ (male_avg : ℝ), male_avg = 185 := by
  sorry

end NUMINAMATH_CALUDE_male_average_height_l2900_290078


namespace NUMINAMATH_CALUDE_drug_price_reduction_l2900_290093

theorem drug_price_reduction (initial_price final_price : ℝ) (x : ℝ) 
  (h_initial : initial_price = 56)
  (h_final : final_price = 31.5)
  (h_positive : 0 < x ∧ x < 1) :
  initial_price * (1 - x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l2900_290093


namespace NUMINAMATH_CALUDE_power_equation_solution_l2900_290035

theorem power_equation_solution : ∃ y : ℕ, (12 ^ 3 * 6 ^ y) / 432 = 5184 :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2900_290035


namespace NUMINAMATH_CALUDE_min_value_theorem_l2900_290067

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  2 / x + 1 / y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ - 3 = 0 ∧ 2 / x₀ + 1 / y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2900_290067


namespace NUMINAMATH_CALUDE_positive_number_squared_plus_self_l2900_290051

theorem positive_number_squared_plus_self (n : ℝ) : n > 0 ∧ n^2 + n = 210 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_squared_plus_self_l2900_290051


namespace NUMINAMATH_CALUDE_conversation_year_1941_l2900_290003

def is_valid_year (y : ℕ) : Prop := 1900 ≤ y ∧ y ≤ 1999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def swap_digits (n : ℕ) : ℕ :=
  ((n % 10) * 10) + (n / 10)

theorem conversation_year_1941 :
  ∃! (conv_year : ℕ) (elder_birth : ℕ) (younger_birth : ℕ),
    is_valid_year conv_year ∧
    is_valid_year elder_birth ∧
    is_valid_year younger_birth ∧
    elder_birth < younger_birth ∧
    conv_year - elder_birth = digit_sum younger_birth ∧
    conv_year - younger_birth = digit_sum elder_birth ∧
    swap_digits (conv_year - elder_birth) = conv_year - younger_birth ∧
    conv_year = 1941 :=
  sorry

end NUMINAMATH_CALUDE_conversation_year_1941_l2900_290003


namespace NUMINAMATH_CALUDE_min_total_routes_l2900_290088

/-- Represents the number of routes for each airline company -/
structure AirlineRoutes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The minimum number of routes needed to maintain connectivity -/
def min_connectivity : ℕ := 14

/-- The total number of cities in the country -/
def num_cities : ℕ := 15

/-- Predicate to check if the network remains connected after removing any one company's routes -/
def remains_connected (routes : AirlineRoutes) : Prop :=
  routes.a + routes.b ≥ min_connectivity ∧
  routes.b + routes.c ≥ min_connectivity ∧
  routes.c + routes.a ≥ min_connectivity

/-- Theorem stating the minimum number of total routes needed -/
theorem min_total_routes (routes : AirlineRoutes) :
  remains_connected routes → routes.a + routes.b + routes.c ≥ 21 := by
  sorry


end NUMINAMATH_CALUDE_min_total_routes_l2900_290088


namespace NUMINAMATH_CALUDE_perimeter_ABCDE_l2900_290018

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_7 : dist E D = 7
axiom angle_AED_right : (E.1 - A.1) * (E.1 - D.1) + (E.2 - A.2) * (E.2 - D.2) = 0
axiom angle_ABC_right : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

-- Define the theorem
theorem perimeter_ABCDE :
  dist A B + dist B C + dist C D + dist D E + dist E A = 28 :=
sorry

end NUMINAMATH_CALUDE_perimeter_ABCDE_l2900_290018


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2900_290032

theorem min_value_quadratic (x y : ℝ) : 5 * x^2 - 4 * x * y + y^2 + 6 * x + 25 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2900_290032


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_after_school_starts_l2900_290077

/-- Represents a date with a month and day. -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month. -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 10 => 31  -- October
  | 11 => 30  -- November
  | 12 => 31  -- December
  | _ => 30   -- Default for other months (not used in this problem)

/-- Returns the day of the week for a given date. -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry  -- Implementation not needed for the statement

/-- Returns true if the given date is a Terrific Tuesday. -/
def isTerrificTuesday (date : Date) : Prop :=
  dayOfWeek date = DayOfWeek.Tuesday ∧
  (∀ d : Nat, d < date.day → dayOfWeek { month := date.month, day := d } = DayOfWeek.Tuesday →
    (∃ d' : Nat, d' < d ∧ dayOfWeek { month := date.month, day := d' } = DayOfWeek.Tuesday))

/-- The main theorem stating that December 31 is the first Terrific Tuesday after October 3. -/
theorem first_terrific_tuesday_after_school_starts :
  let schoolStart : Date := { month := 10, day := 3 }
  let firstTerrificTuesday : Date := { month := 12, day := 31 }
  dayOfWeek schoolStart = DayOfWeek.Tuesday →
  isTerrificTuesday firstTerrificTuesday ∧
  (∀ date : Date, schoolStart.month ≤ date.month ∧ date.month ≤ firstTerrificTuesday.month →
    (if date.month = schoolStart.month then schoolStart.day ≤ date.day else True) →
    (if date.month = firstTerrificTuesday.month then date.day ≤ firstTerrificTuesday.day else True) →
    date.day ≤ daysInMonth date.month →
    isTerrificTuesday date → date = firstTerrificTuesday) :=
by sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_after_school_starts_l2900_290077


namespace NUMINAMATH_CALUDE_convergence_of_beta_series_l2900_290021

theorem convergence_of_beta_series (α : ℕ → ℝ) (β : ℕ → ℝ) :
  (∀ n : ℕ, α n > 0) →
  (∀ n : ℕ, β n = (α n * n) / (n + 1)) →
  Summable α →
  Summable β := by
sorry

end NUMINAMATH_CALUDE_convergence_of_beta_series_l2900_290021


namespace NUMINAMATH_CALUDE_sodium_hydroxide_moles_l2900_290061

/-- Represents the chemical reaction between Sodium hydroxide and Chlorine to produce Water -/
structure ChemicalReaction where
  naoh : ℝ  -- moles of Sodium hydroxide
  cl2 : ℝ   -- moles of Chlorine
  h2o : ℝ   -- moles of Water produced

/-- The stoichiometric ratio of the reaction -/
def stoichiometricRatio : ℝ := 2

theorem sodium_hydroxide_moles (reaction : ChemicalReaction) 
  (h1 : reaction.cl2 = 2)
  (h2 : reaction.h2o = 2)
  (h3 : reaction.naoh = stoichiometricRatio * reaction.h2o) :
  reaction.naoh = 4 := by
  sorry

end NUMINAMATH_CALUDE_sodium_hydroxide_moles_l2900_290061


namespace NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l2900_290041

/-- A regular polygon where the sum of interior angles is 180° more than three times the sum of exterior angles. -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  h1 : sum_interior_angles = (n - 2) * 180
  h2 : sum_exterior_angles = 360
  h3 : sum_interior_angles = 3 * sum_exterior_angles + 180

/-- The number of sides of the regular polygon is 9. -/
theorem number_of_sides (p : RegularPolygon) : p.n = 9 := by sorry

/-- The measure of each interior angle of the regular polygon is 140°. -/
theorem interior_angle_measure (p : RegularPolygon) : 
  (p.sum_interior_angles / p.n : ℝ) = 140 := by sorry

end NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l2900_290041


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_six_sum_fifteen_l2900_290030

theorem two_digit_divisible_by_six_sum_fifteen (n : ℕ) : 
  10 ≤ n ∧ n < 100 ∧                 -- n is a two-digit number
  n % 6 = 0 ∧                        -- n is divisible by 6
  (n / 10 + n % 10 = 15) →           -- sum of digits is 15
  (n / 10) * (n % 10) = 56 ∨ (n / 10) * (n % 10) = 54 := by
sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_six_sum_fifteen_l2900_290030


namespace NUMINAMATH_CALUDE_profit_function_correct_max_profit_production_break_even_range_l2900_290058

/-- Represents the production and profit model of a company -/
structure CompanyModel where
  fixedCost : ℝ
  variableCost : ℝ
  annualDemand : ℝ
  revenueFunction : ℝ → ℝ

/-- The company's specific model -/
def company : CompanyModel :=
  { fixedCost := 0.5,  -- In ten thousand yuan
    variableCost := 0.025,  -- In ten thousand yuan per hundred units
    annualDemand := 5,  -- In hundreds of units
    revenueFunction := λ x => 5 * x - x^2 }  -- In ten thousand yuan

/-- The profit function for the company -/
def profitFunction (x : ℝ) : ℝ :=
  company.revenueFunction x - (company.variableCost * x + company.fixedCost)

/-- Theorem stating the correctness of the profit function -/
theorem profit_function_correct (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) :
  profitFunction x = 5 * x - x^2 - (0.025 * x + 0.5) := by sorry

/-- Theorem stating the maximum profit production -/
theorem max_profit_production :
  ∃ x, x = 4.75 ∧ ∀ y, 0 ≤ y ∧ y ≤ 5 → profitFunction x ≥ profitFunction y := by sorry

/-- Theorem stating the break-even production range -/
theorem break_even_range :
  ∃ a b, a = 0.1 ∧ b = 48 ∧ 
  ∀ x, (a ≤ x ∧ x ≤ 5) ∨ (5 < x ∧ x < b) → profitFunction x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_profit_function_correct_max_profit_production_break_even_range_l2900_290058


namespace NUMINAMATH_CALUDE_actual_distance_is_1542_l2900_290075

/-- Represents a faulty odometer that skips digits 4 and 7 --/
def FaultyOdometer := ℕ → ℕ

/-- The current reading of the odometer --/
def current_reading : ℕ := 2056

/-- The function that calculates the actual distance traveled --/
def actual_distance (o : FaultyOdometer) (reading : ℕ) : ℕ := sorry

/-- Theorem stating that the actual distance traveled is 1542 miles --/
theorem actual_distance_is_1542 (o : FaultyOdometer) :
  actual_distance o current_reading = 1542 := by sorry

end NUMINAMATH_CALUDE_actual_distance_is_1542_l2900_290075


namespace NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l2900_290007

-- Define the universal set I
def I : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 - 3 = p.1 - 2 ∧ p.1 ≠ 2}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_intersection_equals_singleton :
  (Set.compl M ∩ Set.compl N : Set (ℝ × ℝ)) = {(2, 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l2900_290007


namespace NUMINAMATH_CALUDE_function_property_l2900_290094

-- Define the functions
def f₁ (x : ℝ) : ℝ := |2 * x|
def f₂ (x : ℝ) : ℝ := x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sqrt x
def f₄ (x : ℝ) : ℝ := x - |x|

-- State the theorem
theorem function_property :
  (∀ x, f₁ (2 * x) = 2 * f₁ x) ∧
  (∀ x, f₂ (2 * x) = 2 * f₂ x) ∧
  (∃ x, f₃ (2 * x) ≠ 2 * f₃ x) ∧
  (∀ x, f₄ (2 * x) = 2 * f₄ x) :=
by
  sorry

end NUMINAMATH_CALUDE_function_property_l2900_290094


namespace NUMINAMATH_CALUDE_sets_equality_l2900_290042

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x + 1 = 0}
def N : Set ℝ := {1}

-- Theorem statement
theorem sets_equality : M = N := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_l2900_290042


namespace NUMINAMATH_CALUDE_bob_questions_theorem_l2900_290014

/-- Represents the number of questions Bob creates in each hour -/
def questions_per_hour : Fin 3 → ℕ
  | 0 => 13
  | 1 => 13 * 2
  | 2 => 13 * 2 * 2

/-- The total number of questions Bob creates in three hours -/
def total_questions : ℕ := (questions_per_hour 0) + (questions_per_hour 1) + (questions_per_hour 2)

theorem bob_questions_theorem :
  total_questions = 91 := by
  sorry

end NUMINAMATH_CALUDE_bob_questions_theorem_l2900_290014


namespace NUMINAMATH_CALUDE_max_xy_value_l2900_290059

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  ∃ (max_val : ℝ), max_val = 1/8 ∧ ∀ (z : ℝ), x*y ≤ z ∧ z ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2900_290059


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2900_290033

theorem greatest_divisor_with_remainders : ∃! d : ℕ,
  d > 0 ∧
  (∀ k : ℕ, k > 0 ∧ (∃ q₁ : ℕ, 13976 = k * q₁ + 23) ∧ (∃ q₂ : ℕ, 20868 = k * q₂ + 37) → k ≤ d) ∧
  (∃ q₁ : ℕ, 13976 = d * q₁ + 23) ∧
  (∃ q₂ : ℕ, 20868 = d * q₂ + 37) ∧
  d = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2900_290033


namespace NUMINAMATH_CALUDE_marathon_distance_l2900_290074

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) 
  (yards_per_mile : ℕ) (h1 : marathons = 15) (h2 : miles_per_marathon = 26) 
  (h3 : yards_per_marathon = 395) (h4 : yards_per_mile = 1760) : 
  ∃ (m : ℕ) (y : ℕ), 
    (marathons * miles_per_marathon * yards_per_mile + marathons * yards_per_marathon = 
      m * yards_per_mile + y) ∧ 
    y < yards_per_mile ∧ 
    y = 645 := by
  sorry

end NUMINAMATH_CALUDE_marathon_distance_l2900_290074


namespace NUMINAMATH_CALUDE_first_term_of_specific_series_l2900_290069

/-- Given an infinite geometric series with sum S and sum of squares T,
    this function returns the first term of the series. -/
def first_term_of_geometric_series (S : ℝ) (T : ℝ) : ℝ := 
  sorry

/-- Theorem stating that for an infinite geometric series with sum 27 and 
    sum of squares 108, the first term is 216/31. -/
theorem first_term_of_specific_series : 
  first_term_of_geometric_series 27 108 = 216 / 31 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_specific_series_l2900_290069


namespace NUMINAMATH_CALUDE_function_shift_l2900_290017

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_shift (x : ℝ) :
  (∀ y : ℝ, f (y + 1) = y^2 + 2*y) →
  f (x - 1) = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l2900_290017


namespace NUMINAMATH_CALUDE_students_speaking_both_languages_l2900_290096

theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (japanese : ℕ) (neither : ℕ) :
  total = 50 →
  english = 36 →
  japanese = 20 →
  neither = 8 →
  ∃ x : ℕ, x = 14 ∧ 
    x = english + japanese - (total - neither) :=
by sorry

end NUMINAMATH_CALUDE_students_speaking_both_languages_l2900_290096


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2900_290020

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let vertex1 := (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c)
  let vertex2 := (- e / (2 * d), d * (- e / (2 * d))^2 + e * (- e / (2 * d)) + f)
  let distance := Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2)
  (a = 1 ∧ b = -4 ∧ c = 7 ∧ d = 1 ∧ e = 6 ∧ f = 20) → distance = Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2900_290020


namespace NUMINAMATH_CALUDE_problem_solution_l2900_290005

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b - c - d = 3)
  (h2 : a * b - 3 * b * c + c * d - 3 * d * a = 4)
  (h3 : 3 * a * b - b * c + 3 * c * d - d * a = 5) :
  11 * (a - c)^2 + 17 * (b - d)^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2900_290005


namespace NUMINAMATH_CALUDE_front_wheel_cost_l2900_290031

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def remaining_amount : ℕ := 20

theorem front_wheel_cost : 
  initial_amount - frame_cost - remaining_amount = 25 := by sorry

end NUMINAMATH_CALUDE_front_wheel_cost_l2900_290031


namespace NUMINAMATH_CALUDE_finite_intersection_l2900_290068

def sequence_a : ℕ → ℕ → ℕ
  | a₁, 0 => a₁
  | a₁, n + 1 => n * sequence_a a₁ n + 1

def sequence_b : ℕ → ℕ → ℕ
  | b₁, 0 => b₁
  | b₁, n + 1 => n * sequence_b b₁ n - 1

theorem finite_intersection (a₁ b₁ : ℕ) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → sequence_a a₁ n ≠ sequence_b b₁ n :=
sorry

end NUMINAMATH_CALUDE_finite_intersection_l2900_290068


namespace NUMINAMATH_CALUDE_no_convex_quadrilateral_with_all_acute_triangles_l2900_290060

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Condition for convexity

-- Define an acute-angled triangle
def is_acute_angled_triangle (a b c : ℝ × ℝ) : Prop :=
  sorry -- Condition for all angles being less than 90 degrees

-- Define a diagonal of a quadrilateral
def diagonal (q : ConvexQuadrilateral) (i j : Fin 4) : Prop :=
  sorry -- Condition for i and j being opposite vertices

-- Theorem statement
theorem no_convex_quadrilateral_with_all_acute_triangles :
  ¬ ∃ (q : ConvexQuadrilateral),
    ∀ (i j : Fin 4), diagonal q i j →
      is_acute_angled_triangle (q.vertices i) (q.vertices j) (q.vertices ((i + 1) % 4)) ∧
      is_acute_angled_triangle (q.vertices i) (q.vertices j) (q.vertices ((j + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_no_convex_quadrilateral_with_all_acute_triangles_l2900_290060


namespace NUMINAMATH_CALUDE_valid_pythagorean_grid_exists_l2900_290099

/-- A 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Check if all numbers in the grid are distinct -/
def allDistinct (g : Grid) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → g i j ≠ g k l

/-- Check if all numbers in the grid are less than 100 -/
def allLessThan100 (g : Grid) : Prop :=
  ∀ i j, g i j < 100

/-- Check if all rows in the grid form Pythagorean triples -/
def allRowsPythagorean (g : Grid) : Prop :=
  ∀ i, isPythagoreanTriple (g i 0) (g i 1) (g i 2)

/-- Check if all columns in the grid form Pythagorean triples -/
def allColumnsPythagorean (g : Grid) : Prop :=
  ∀ j, isPythagoreanTriple (g 0 j) (g 1 j) (g 2 j)

/-- The main theorem: there exists a valid grid satisfying all conditions -/
theorem valid_pythagorean_grid_exists : ∃ (g : Grid),
  allDistinct g ∧
  allLessThan100 g ∧
  allRowsPythagorean g ∧
  allColumnsPythagorean g :=
sorry

end NUMINAMATH_CALUDE_valid_pythagorean_grid_exists_l2900_290099


namespace NUMINAMATH_CALUDE_picture_arrangements_l2900_290013

/-- The number of people in the initial group -/
def initial_group_size : ℕ := 4

/-- The number of people combined into one unit -/
def combined_unit_size : ℕ := 2

/-- The effective number of units to arrange -/
def effective_units : ℕ := initial_group_size - combined_unit_size + 1

theorem picture_arrangements :
  (effective_units).factorial = 6 := by
  sorry

end NUMINAMATH_CALUDE_picture_arrangements_l2900_290013


namespace NUMINAMATH_CALUDE_count_special_numbers_l2900_290012

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem count_special_numbers :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_three_digit_number n ∧ is_valid_three_digit_number (n - 297) ∧ 
               n - 297 = reverse_number n) ∧
    S.card = 60 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_l2900_290012


namespace NUMINAMATH_CALUDE_problem_solution_l2900_290049

theorem problem_solution (x y : ℝ) (hx : x = 2 - Real.sqrt 3) (hy : y = 2 + Real.sqrt 3) :
  (x^2 - y^2 = -8 * Real.sqrt 3) ∧ (x^2 + x*y + y^2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2900_290049


namespace NUMINAMATH_CALUDE_fifteenth_term_of_arithmetic_sequence_l2900_290062

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifteenth_term_of_arithmetic_sequence 
  (a : ℕ → ℕ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 15)
  (h_third : a 3 = 27) :
  a 15 = 171 :=
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_arithmetic_sequence_l2900_290062


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_parallel_line_to_parallel_plane_l2900_290050

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (planesParallel : Plane → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- Theorem for proposition ②
theorem perpendicular_to_parallel_plane 
  (m n : Line) (α : Plane)
  (h1 : perpendicularToPlane m α)
  (h2 : parallelToPlane n α) :
  perpendicular m n :=
sorry

-- Theorem for proposition ③
theorem parallel_line_to_parallel_plane 
  (m : Line) (α β : Plane)
  (h1 : planesParallel α β)
  (h2 : lineInPlane m α) :
  parallelToPlane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_parallel_line_to_parallel_plane_l2900_290050


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l2900_290056

/-- The amount of rope in feet bought last week -/
def rope_last_week : ℕ := 6

/-- The difference in feet between last week's and this week's rope purchase -/
def rope_difference : ℕ := 4

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The total amount of rope bought in inches -/
def total_rope_inches : ℕ := 
  (rope_last_week * inches_per_foot) + 
  ((rope_last_week - rope_difference) * inches_per_foot)

theorem sanchez_rope_theorem : total_rope_inches = 96 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l2900_290056


namespace NUMINAMATH_CALUDE_butterfly_cocoon_time_l2900_290089

theorem butterfly_cocoon_time :
  ∀ (L C : ℕ),
    L + C = 120 →
    L = 3 * C →
    C = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_butterfly_cocoon_time_l2900_290089


namespace NUMINAMATH_CALUDE_sum_of_inserted_numbers_l2900_290063

/-- A sequence of five real numbers -/
structure Sequence :=
  (a b c : ℝ)

/-- Check if the first four terms form a harmonic progression -/
def isHarmonicProgression (s : Sequence) : Prop :=
  ∃ (h : ℝ), 1/4 - 1/s.a = 1/s.a - 1/s.b ∧ 1/s.a - 1/s.b = 1/s.b - 1/s.c

/-- Check if the last four terms form a quadratic sequence -/
def isQuadraticSequence (s : Sequence) : Prop :=
  ∃ (p q : ℝ), 
    s.a = 1^2 + p + q ∧
    s.b = 2^2 + 2*p + q ∧
    s.c = 3^2 + 3*p + q ∧
    16 = 4^2 + 4*p + q

/-- The main theorem -/
theorem sum_of_inserted_numbers (s : Sequence) :
  s.a > 0 ∧ s.b > 0 ∧ s.c > 0 →
  isHarmonicProgression s →
  isQuadraticSequence s →
  s.a + s.b + s.c = 33 :=
sorry

end NUMINAMATH_CALUDE_sum_of_inserted_numbers_l2900_290063


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2900_290085

theorem ratio_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2900_290085


namespace NUMINAMATH_CALUDE_real_part_of_z_l2900_290038

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2900_290038


namespace NUMINAMATH_CALUDE_prime_4k_plus_1_properties_l2900_290000

theorem prime_4k_plus_1_properties (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 1) :
  (∃ x : ℤ, (x^2 + 1) % p = 0) ∧
  (∃ r₁ r₂ s₁ s₂ : ℕ,
    r₁ < Real.sqrt p ∧ r₂ < Real.sqrt p ∧ s₁ < Real.sqrt p ∧ s₂ < Real.sqrt p ∧
    (r₁ ≠ r₂ ∨ s₁ ≠ s₂) ∧
    ∃ x : ℤ, (r₁ * x + s₁) % p = (r₂ * x + s₂) % p) ∧
  (∃ r₁ r₂ s₁ s₂ : ℕ,
    r₁ < Real.sqrt p ∧ r₂ < Real.sqrt p ∧ s₁ < Real.sqrt p ∧ s₂ < Real.sqrt p ∧
    p = (r₁ - r₂)^2 + (s₁ - s₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_4k_plus_1_properties_l2900_290000


namespace NUMINAMATH_CALUDE_lisa_speed_equals_eugene_l2900_290082

def eugene_speed : ℚ := 5
def carlos_speed_ratio : ℚ := 3/4
def lisa_speed_ratio : ℚ := 4/3

theorem lisa_speed_equals_eugene (eugene_speed : ℚ) (carlos_speed_ratio : ℚ) (lisa_speed_ratio : ℚ) :
  eugene_speed * carlos_speed_ratio * lisa_speed_ratio = eugene_speed :=
by sorry

end NUMINAMATH_CALUDE_lisa_speed_equals_eugene_l2900_290082


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2900_290015

-- Define the variables and conditions
variable (a b c : ℝ)
variable (h1 : 2 * a + b = c)
variable (h2 : c ≠ 0)

-- Theorem 1
theorem problem_1 : (2 * a + b - c - 1)^2023 = -1 := by sorry

-- Theorem 2
theorem problem_2 : (10 * c) / (4 * a + 2 * b) = 5 := by sorry

-- Theorem 3
theorem problem_3 : (2 * a + b) * 3 = c + 4 * a + 2 * b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2900_290015


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2900_290010

theorem simplify_and_evaluate (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) :
  (1 - 1/a) / ((a^2 - 2*a + 1) / a) = 1 / (a - 1) ∧
  (1 - 1/2) / ((2^2 - 2*2 + 1) / 2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2900_290010


namespace NUMINAMATH_CALUDE_bank_teller_bills_l2900_290044

theorem bank_teller_bills (total_bills : ℕ) (total_value : ℕ) : 
  total_bills = 54 → total_value = 780 → 
  ∃ (five_dollar_bills twenty_dollar_bills : ℕ), 
    five_dollar_bills + twenty_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 20 * twenty_dollar_bills = total_value ∧
    five_dollar_bills = 20 := by
  sorry

end NUMINAMATH_CALUDE_bank_teller_bills_l2900_290044


namespace NUMINAMATH_CALUDE_equation_solution_l2900_290055

theorem equation_solution : ∃ y : ℚ, (3 * (y + 1) / 4 - (1 - y) / 8 = 1) ∧ (y = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2900_290055


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l2900_290081

theorem right_triangle_consecutive_legs (a b c : ℕ) : 
  a + 1 = b →                 -- legs are consecutive whole numbers
  a^2 + b^2 = 41^2 →          -- Pythagorean theorem with hypotenuse 41
  a + b = 57 := by            -- sum of legs is 57
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l2900_290081


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2900_290083

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2900_290083
