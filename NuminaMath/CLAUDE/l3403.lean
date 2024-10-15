import Mathlib

namespace NUMINAMATH_CALUDE_max_digit_sum_2016_l3403_340368

/-- A function that sums the digits of a natural number -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- A function that repeatedly sums the digits until a single digit is obtained -/
def repeatSumDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 2016 digits -/
def has2016Digits (n : ℕ) : Prop := sorry

theorem max_digit_sum_2016 :
  ∀ n : ℕ, has2016Digits n → repeatSumDigits n ≤ 9 ∧ ∃ m : ℕ, has2016Digits m ∧ repeatSumDigits m = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_digit_sum_2016_l3403_340368


namespace NUMINAMATH_CALUDE_choir_average_age_l3403_340343

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (total_people : ℕ) 
  (h1 : num_females = 10) 
  (h2 : num_males = 15) 
  (h3 : avg_age_females = 30) 
  (h4 : avg_age_males = 35) 
  (h5 : total_people = num_females + num_males) :
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 33 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l3403_340343


namespace NUMINAMATH_CALUDE_negative_four_star_two_simplify_a_minus_b_cubed_specific_values_l3403_340366

-- Define the * operation
def star (x y : ℚ) : ℚ := x^2 - 3*y + 3

-- Theorem 1
theorem negative_four_star_two : star (-4) 2 = 13 := by sorry

-- Theorem 2
theorem simplify_a_minus_b_cubed (a b : ℚ) : 
  star (a - b) ((a - b)^2) = -2*a^2 - 2*b^2 + 4*a*b + 3 := by sorry

-- Theorem 3
theorem specific_values : 
  star (-2 - (1/2)) ((-2 - (1/2))^2) = -13/2 := by sorry

end NUMINAMATH_CALUDE_negative_four_star_two_simplify_a_minus_b_cubed_specific_values_l3403_340366


namespace NUMINAMATH_CALUDE_share_sale_value_l3403_340355

/-- The value of the business in rs -/
def business_value : ℚ := 10000

/-- The fraction of the business owned by the man -/
def man_ownership : ℚ := 1/3

/-- The fraction of the man's shares that he sells -/
def sold_fraction : ℚ := 3/5

/-- The amount the man receives for selling his shares -/
def sold_amount : ℚ := 2000

theorem share_sale_value :
  sold_fraction * man_ownership * business_value = sold_amount := by
  sorry

end NUMINAMATH_CALUDE_share_sale_value_l3403_340355


namespace NUMINAMATH_CALUDE_father_daughter_ages_l3403_340387

theorem father_daughter_ages (father_age daughter_age : ℕ) : 
  father_age = 4 * daughter_age ∧ father_age = daughter_age + 30 →
  father_age = 40 ∧ daughter_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_father_daughter_ages_l3403_340387


namespace NUMINAMATH_CALUDE_complement_of_A_l3403_340310

def U : Set ℕ := {x | 0 ≤ x ∧ x < 10}

def A : Set ℕ := {2, 4, 6, 8}

theorem complement_of_A : U \ A = {1, 3, 5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3403_340310


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3403_340322

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 2*x^3 + x + 5) % (x - 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3403_340322


namespace NUMINAMATH_CALUDE_function_properties_l3403_340375

/-- Given a function f(x) = 2x - a/x where f(1) = 3, this theorem proves
    properties about the value of a, the parity of f, and its monotonicity. -/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = 2*x - a/x)
    (h_f1 : f 1 = 3) :
  (a = -1) ∧ 
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, 1 < x₂ ∧ x₂ < x₁ → f x₂ < f x₁) :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l3403_340375


namespace NUMINAMATH_CALUDE_weekly_running_distance_l3403_340384

/-- Calculates the total distance run in a week given the track length, loops per day, and days per week. -/
def total_distance (track_length : ℕ) (loops_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  track_length * loops_per_day * days_per_week

/-- Theorem stating that running 10 loops per day on a 50-meter track for 7 days results in 3500 meters per week. -/
theorem weekly_running_distance :
  total_distance 50 10 7 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_running_distance_l3403_340384


namespace NUMINAMATH_CALUDE_distance_product_sum_bound_l3403_340328

/-- Given an equilateral triangle with side length 1 and a point P inside it,
    let a, b, c be the distances from P to the three sides of the triangle. -/
def DistancesFromPoint (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = Real.sqrt 3 / 2

/-- The sum of products of distances from a point inside an equilateral triangle
    to its sides is bounded. -/
theorem distance_product_sum_bound {a b c : ℝ} (h : DistancesFromPoint a b c) :
  0 < a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_product_sum_bound_l3403_340328


namespace NUMINAMATH_CALUDE_solve_food_bank_problem_l3403_340300

def food_bank_problem (first_week_donation : ℝ) (second_week_multiplier : ℝ) (remaining_food : ℝ) : Prop :=
  let total_donation := first_week_donation + (second_week_multiplier * first_week_donation)
  let food_given_out := total_donation - remaining_food
  let percentage_given_out := (food_given_out / total_donation) * 100
  percentage_given_out = 70

theorem solve_food_bank_problem :
  food_bank_problem 40 2 36 := by
  sorry

end NUMINAMATH_CALUDE_solve_food_bank_problem_l3403_340300


namespace NUMINAMATH_CALUDE_log_5_125_l3403_340350

-- Define the logarithm function
noncomputable def log (a : ℝ) (N : ℝ) : ℝ :=
  Real.log N / Real.log a

-- Theorem statement
theorem log_5_125 : log 5 125 = 3 := by
  sorry


end NUMINAMATH_CALUDE_log_5_125_l3403_340350


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3403_340360

theorem inequality_solution_set (x : ℝ) : 
  (3 * x - 5 > 11 - 2 * x) ↔ (x > 16 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3403_340360


namespace NUMINAMATH_CALUDE_min_value_of_squares_l3403_340312

theorem min_value_of_squares (a b c d : ℝ) (h1 : a * b = 3) (h2 : c + 3 * d = 0) :
  (a - c)^2 + (b - d)^2 ≥ 18/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l3403_340312


namespace NUMINAMATH_CALUDE_lcm_48_180_l3403_340354

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l3403_340354


namespace NUMINAMATH_CALUDE_equal_color_diagonals_l3403_340382

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → Point

/-- A coloring of vertices of a polygon -/
def VertexColoring (n : ℕ) := Fin n → Bool

/-- The number of diagonals with both endpoints of a given color -/
def numSameColorDiagonals (n : ℕ) (coloring : VertexColoring n) (color : Bool) : ℕ := sorry

theorem equal_color_diagonals 
  (polygon : RegularPolygon 20) 
  (coloring : VertexColoring 20)
  (h_black_count : (Finset.filter (fun i => coloring i = true) (Finset.univ : Finset (Fin 20))).card = 10)
  (h_white_count : (Finset.filter (fun i => coloring i = false) (Finset.univ : Finset (Fin 20))).card = 10) :
  numSameColorDiagonals 20 coloring true = numSameColorDiagonals 20 coloring false := by
  sorry


end NUMINAMATH_CALUDE_equal_color_diagonals_l3403_340382


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l3403_340325

/-- The number of perfect square factors of 360 -/
def perfectSquareFactors : ℕ := 4

/-- The prime factorization of 360 -/
def primeFactorization : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

/-- Theorem stating that the number of perfect square factors of 360 is 4 -/
theorem count_perfect_square_factors :
  (List.sum (List.map (fun (p : ℕ × ℕ) => (p.2 / 2 + 1)) primeFactorization)) = perfectSquareFactors := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l3403_340325


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3403_340318

theorem system_solution_ratio (x y c d : ℝ) : 
  (4 * x - 2 * y = c) →
  (5 * y - 10 * x = d) →
  d ≠ 0 →
  c / d = 0 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3403_340318


namespace NUMINAMATH_CALUDE_vehicle_wheels_count_l3403_340353

/-- Proves that each vehicle has 4 wheels given the problem conditions -/
theorem vehicle_wheels_count (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 25)
  (h2 : total_wheels = 100) :
  total_wheels / total_vehicles = 4 := by
  sorry

#check vehicle_wheels_count

end NUMINAMATH_CALUDE_vehicle_wheels_count_l3403_340353


namespace NUMINAMATH_CALUDE_sweets_problem_l3403_340351

/-- The number of sweets initially on the table -/
def initial_sweets : ℕ := 50

/-- The number of sweets Jack took -/
def jack_sweets (total : ℕ) : ℕ := total / 2 + 4

/-- The number of sweets remaining after Jack -/
def after_jack (total : ℕ) : ℕ := total - jack_sweets total

/-- The number of sweets Paul took -/
def paul_sweets (remaining : ℕ) : ℕ := remaining / 3 + 5

/-- The number of sweets remaining after Paul -/
def after_paul (remaining : ℕ) : ℕ := remaining - paul_sweets remaining

/-- Olivia took the last 9 sweets -/
def olivia_sweets : ℕ := 9

theorem sweets_problem :
  after_paul (after_jack initial_sweets) = olivia_sweets :=
sorry

end NUMINAMATH_CALUDE_sweets_problem_l3403_340351


namespace NUMINAMATH_CALUDE_unique_k_satisfying_equation_l3403_340356

theorem unique_k_satisfying_equation : ∃! k : ℕ, 10^k - 1 = 9*k^2 := by sorry

end NUMINAMATH_CALUDE_unique_k_satisfying_equation_l3403_340356


namespace NUMINAMATH_CALUDE_fraction_calculation_l3403_340383

theorem fraction_calculation : 
  (((1 / 2 : ℚ) + (1 / 3)) / ((2 / 7 : ℚ) + (1 / 4))) * (3 / 5) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3403_340383


namespace NUMINAMATH_CALUDE_wallet_total_l3403_340303

theorem wallet_total (nada ali john : ℕ) : 
  ali = nada - 5 →
  john = 4 * nada →
  john = 48 →
  ali + nada + john = 67 := by
sorry

end NUMINAMATH_CALUDE_wallet_total_l3403_340303


namespace NUMINAMATH_CALUDE_james_semesters_paid_l3403_340324

/-- Calculates the number of semesters paid for given the units per semester, cost per unit, and total cost. -/
def semesters_paid (units_per_semester : ℕ) (cost_per_unit : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (units_per_semester * cost_per_unit)

/-- Proves that given 20 units per semester, $50 per unit, and $2000 total cost, the number of semesters paid for is 2. -/
theorem james_semesters_paid :
  semesters_paid 20 50 2000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_semesters_paid_l3403_340324


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3403_340393

theorem polynomial_factorization (x : ℝ) :
  (∃ (a b c d : ℝ), x^2 - 1 = (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x + 1 ≠ (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x^2 + x + 1 ≠ (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x^2 + 4 ≠ (a*x + b) * (c*x + d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3403_340393


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3403_340344

/-- The focal length of the hyperbola y²/9 - x²/7 = 1 is 8 -/
theorem hyperbola_focal_length : ∃ (a b c : ℝ), 
  (a^2 = 9 ∧ b^2 = 7) → 
  (∀ (x y : ℝ), y^2 / 9 - x^2 / 7 = 1 → (x / a)^2 - (y / b)^2 = 1) →
  c^2 = a^2 + b^2 →
  2 * c = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3403_340344


namespace NUMINAMATH_CALUDE_sum_of_cubics_degree_at_most_3_l3403_340302

-- Define a cubic polynomial
def CubicPolynomial (R : Type*) [CommRing R] := {p : Polynomial R // p.degree ≤ 3}

-- Theorem statement
theorem sum_of_cubics_degree_at_most_3 {R : Type*} [CommRing R] 
  (A B : CubicPolynomial R) : 
  (A.val + B.val).degree ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubics_degree_at_most_3_l3403_340302


namespace NUMINAMATH_CALUDE_number_division_remainder_l3403_340399

theorem number_division_remainder (N : ℤ) (D : ℕ) 
  (h1 : N % 125 = 40) 
  (h2 : N % D = 11) : 
  D = 29 := by
sorry

end NUMINAMATH_CALUDE_number_division_remainder_l3403_340399


namespace NUMINAMATH_CALUDE_golf_balls_needed_l3403_340397

def weekend_goal : ℕ := 48
def saturday_balls : ℕ := 16
def sunday_balls : ℕ := 18

theorem golf_balls_needed : weekend_goal - (saturday_balls + sunday_balls) = 14 := by
  sorry

end NUMINAMATH_CALUDE_golf_balls_needed_l3403_340397


namespace NUMINAMATH_CALUDE_smallest_solution_l3403_340331

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation in the problem -/
def equation (x : ℝ) : Prop :=
  floor (x^2) - (floor x)^2 = 17

/-- Theorem stating that 7√2 is the smallest solution -/
theorem smallest_solution :
  ∀ x : ℝ, equation x → x ≥ 7 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l3403_340331


namespace NUMINAMATH_CALUDE_sin_equality_theorem_l3403_340339

theorem sin_equality_theorem (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.sin (720 * π / 180)) ↔ (n = 0 ∨ n = 180) := by
sorry

end NUMINAMATH_CALUDE_sin_equality_theorem_l3403_340339


namespace NUMINAMATH_CALUDE_problem_statement_l3403_340394

theorem problem_statement (n b : ℝ) : n = 2^(1/10) ∧ n^b = 16 → b = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3403_340394


namespace NUMINAMATH_CALUDE_rational_expression_iff_zero_l3403_340301

theorem rational_expression_iff_zero (x : ℝ) : 
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 4) - 1 / (x + Real.sqrt (x^2 + 4)) = q ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_iff_zero_l3403_340301


namespace NUMINAMATH_CALUDE_circumscribed_square_area_l3403_340372

/-- Given a circle with an inscribed square of perimeter p, 
    the area of the square that circumscribes the circle is p²/8 -/
theorem circumscribed_square_area (p : ℝ) (p_pos : p > 0) : 
  let inscribed_square_perimeter := p
  let circumscribed_square_area := p^2 / 8
  inscribed_square_perimeter = p → circumscribed_square_area = p^2 / 8 := by
sorry

end NUMINAMATH_CALUDE_circumscribed_square_area_l3403_340372


namespace NUMINAMATH_CALUDE_water_distribution_l3403_340380

/-- A water distribution problem for four neighborhoods. -/
theorem water_distribution (total : ℕ) (left_for_fourth : ℕ) : 
  total = 1200 → 
  left_for_fourth = 350 → 
  ∃ (first second third fourth : ℕ),
    first + second + third + fourth = total ∧
    second = 2 * first ∧
    third = second + 100 ∧
    fourth = left_for_fourth ∧
    first = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_distribution_l3403_340380


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3403_340381

/-- Two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are different -/
def are_different (l1 l2 : Line) : Prop := sorry

theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) 
  (h1 : are_different m n)
  (h2 : are_parallel m n) 
  (h3 : is_perpendicular_to_plane n β) : 
  is_perpendicular_to_plane m β := by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3403_340381


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3403_340347

theorem sum_of_fractions_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3403_340347


namespace NUMINAMATH_CALUDE_fifth_sample_number_l3403_340313

def isValidNumber (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 700

def findNthValidNumber (sequence : List ℕ) (n : ℕ) : Option ℕ :=
  let validNumbers := sequence.filter isValidNumber
  let uniqueValidNumbers := validNumbers.eraseDups
  uniqueValidNumbers.get? (n - 1)

theorem fifth_sample_number (sequence : List ℕ) : 
  findNthValidNumber sequence 5 = some 328 := by
  sorry

end NUMINAMATH_CALUDE_fifth_sample_number_l3403_340313


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l3403_340377

/-- Represents a cube with painted strips on its faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  stripsPerFace : Nat
  stripWidth : Nat
  stripLength : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedCubes cube
where
  /-- Helper function to calculate the number of painted unit cubes -/
  paintedCubes (cube : PaintedCube) : Nat :=
    let totalPainted := 6 * cube.stripsPerFace * cube.stripLength
    let edgeOverlaps := 12 * cube.stripWidth / 2
    let cornerOverlaps := 8
    totalPainted - edgeOverlaps - cornerOverlaps

/-- Theorem stating that a 6x6x6 cube with specific painted strips has 170 unpainted unit cubes -/
theorem unpainted_cubes_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    stripsPerFace := 2,
    stripWidth := 1,
    stripLength := 6
  }
  unpaintedCubes cube = 170 := by
  sorry


end NUMINAMATH_CALUDE_unpainted_cubes_count_l3403_340377


namespace NUMINAMATH_CALUDE_number_problem_l3403_340389

theorem number_problem (x : ℝ) : 0.20 * x - 4 = 6 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3403_340389


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3403_340364

/-- Given a cafeteria with initial apples, apples handed out, and apples needed per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Theorem stating that with 62 initial apples, 8 apples handed out, and 9 apples per pie,
    the cafeteria can make 6 pies. -/
theorem cafeteria_pies :
  calculate_pies 62 8 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3403_340364


namespace NUMINAMATH_CALUDE_not_proper_subset_of_itself_l3403_340330

def main_set : Set ℕ := {1, 2, 3}

theorem not_proper_subset_of_itself : ¬(main_set ⊂ main_set) := by
  sorry

end NUMINAMATH_CALUDE_not_proper_subset_of_itself_l3403_340330


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3403_340371

/-- Given an initial solution and added water, calculate the new alcohol percentage -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_percentage = 26 →
  added_water = 5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let total_volume := initial_volume + added_water
  let new_percentage := (initial_alcohol / total_volume) * 100
  new_percentage = 19.5 := by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l3403_340371


namespace NUMINAMATH_CALUDE_john_grandpa_money_l3403_340308

theorem john_grandpa_money (x : ℝ) : 
  x > 0 ∧ x + 3 * x = 120 → x = 30 := by sorry

end NUMINAMATH_CALUDE_john_grandpa_money_l3403_340308


namespace NUMINAMATH_CALUDE_farmers_additional_cost_l3403_340395

/-- The additional cost for Farmer Brown's new hay requirements -/
def additional_cost (original_bales : ℕ) (original_price : ℕ) (new_bales : ℕ) (new_price : ℕ) : ℕ :=
  new_bales * new_price - original_bales * original_price

/-- Theorem: The additional cost for Farmer Brown's new requirements is $210 -/
theorem farmers_additional_cost :
  additional_cost 10 15 20 18 = 210 := by
  sorry

end NUMINAMATH_CALUDE_farmers_additional_cost_l3403_340395


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l3403_340342

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is (4/5, 38/5, 59/5) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 4, 6)
  let B : ℝ × ℝ × ℝ := (6, 5, 3)
  let C : ℝ × ℝ × ℝ := (4, 6, 7)
  orthocenter A B C = (4/5, 38/5, 59/5) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l3403_340342


namespace NUMINAMATH_CALUDE_history_homework_time_l3403_340390

/-- Represents the time in minutes for each homework subject and the total available time. -/
structure HomeworkTime where
  total : Nat
  math : Nat
  english : Nat
  science : Nat
  special_project : Nat

/-- Calculates the time remaining for history homework given the times for other subjects. -/
def history_time (hw : HomeworkTime) : Nat :=
  hw.total - (hw.math + hw.english + hw.science + hw.special_project)

/-- Proves that given the specified homework times, the remaining time for history is 25 minutes. -/
theorem history_homework_time :
  let hw : HomeworkTime := {
    total := 180,  -- 3 hours in minutes
    math := 45,
    english := 30,
    science := 50,
    special_project := 30
  }
  history_time hw = 25 := by sorry

end NUMINAMATH_CALUDE_history_homework_time_l3403_340390


namespace NUMINAMATH_CALUDE_root_sum_pow_l3403_340361

theorem root_sum_pow (p q : ℝ) : 
  p^2 - 7*p + 12 = 0 → 
  q^2 - 7*q + 12 = 0 → 
  p^3 + p^4*q^2 + p^2*q^4 + q^3 = 3691 := by
sorry

end NUMINAMATH_CALUDE_root_sum_pow_l3403_340361


namespace NUMINAMATH_CALUDE_concrete_pillars_amount_l3403_340315

/-- Calculates the concrete needed for supporting pillars with environmental factors --/
def concrete_for_pillars (total_concrete : ℝ) (roadway_concrete : ℝ) (anchor_concrete : ℝ) (env_factor : ℝ) : ℝ :=
  let total_anchor_concrete := 2 * anchor_concrete
  let initial_pillar_concrete := total_concrete - roadway_concrete - total_anchor_concrete
  let pillar_increase := initial_pillar_concrete * env_factor
  initial_pillar_concrete + pillar_increase

/-- Theorem stating the amount of concrete needed for supporting pillars --/
theorem concrete_pillars_amount : 
  concrete_for_pillars 4800 1600 700 0.05 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_concrete_pillars_amount_l3403_340315


namespace NUMINAMATH_CALUDE_smallest_angle_for_tan_equation_l3403_340392

theorem smallest_angle_for_tan_equation :
  ∃ x : ℝ, x > 0 ∧ x < 2 * Real.pi ∧
  Real.tan (6 * x) = (Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) ∧
  x = 45 * Real.pi / (7 * 180) ∧
  ∀ y : ℝ, y > 0 → y < 2 * Real.pi →
    Real.tan (6 * y) = (Real.sin y - Real.cos y) / (Real.sin y + Real.cos y) →
    x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_for_tan_equation_l3403_340392


namespace NUMINAMATH_CALUDE_water_conservation_l3403_340385

/-- Represents the amount of water in tons, where negative values indicate waste and positive values indicate savings. -/
def WaterAmount : Type := ℤ

/-- Records the water amount given the number of tons wasted or saved. -/
def recordWaterAmount (tons : ℤ) : WaterAmount := tons

theorem water_conservation (waste : WaterAmount) (save : ℤ) :
  waste = recordWaterAmount (-10) →
  recordWaterAmount save = recordWaterAmount 30 :=
by sorry

end NUMINAMATH_CALUDE_water_conservation_l3403_340385


namespace NUMINAMATH_CALUDE_tomato_price_proof_l3403_340362

/-- The original price per pound of tomatoes -/
def original_price : ℝ := 0.80

/-- The proportion of tomatoes remaining after discarding ruined ones -/
def remaining_proportion : ℝ := 0.90

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.12

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 0.9956

theorem tomato_price_proof :
  selling_price * remaining_proportion = original_price * (1 + profit_percentage) := by
  sorry


end NUMINAMATH_CALUDE_tomato_price_proof_l3403_340362


namespace NUMINAMATH_CALUDE_projection_theorem_l3403_340323

/-- Given vectors a, b, and c in ℝ², prove that the projection of a onto c is 4 -/
theorem projection_theorem (a b c : ℝ × ℝ) : 
  a = (4, 2) → b = (2, 1) → c = (3, 4) → a.1 / b.1 = a.2 / b.2 →
  (a.1 * c.1 + a.2 * c.2) / Real.sqrt (c.1^2 + c.2^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l3403_340323


namespace NUMINAMATH_CALUDE_gnome_ratio_l3403_340333

/-- Represents the properties of garden gnomes -/
structure GnomeProperties where
  total : Nat
  bigNoses : Nat
  blueHatsBigNoses : Nat
  redHatsSmallNoses : Nat

/-- Theorem: The ratio of gnomes with red hats to total gnomes is 3:4 -/
theorem gnome_ratio (g : GnomeProperties) 
  (h1 : g.total = 28)
  (h2 : g.bigNoses = g.total / 2)
  (h3 : g.blueHatsBigNoses = 6)
  (h4 : g.redHatsSmallNoses = 13) :
  (g.redHatsSmallNoses + (g.bigNoses - g.blueHatsBigNoses)) * 4 = g.total * 3 := by
  sorry

#check gnome_ratio

end NUMINAMATH_CALUDE_gnome_ratio_l3403_340333


namespace NUMINAMATH_CALUDE_root_implies_k_value_l3403_340379

theorem root_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l3403_340379


namespace NUMINAMATH_CALUDE_bridget_apples_l3403_340316

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 5 + 4 + 4 = x → x = 22 :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_l3403_340316


namespace NUMINAMATH_CALUDE_average_people_moving_per_hour_l3403_340317

/-- The number of people moving to Texas in 5 days -/
def people_moving : ℕ := 3500

/-- The number of days -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem average_people_moving_per_hour :
  round_to_nearest average_per_hour = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_people_moving_per_hour_l3403_340317


namespace NUMINAMATH_CALUDE_square_product_theorem_l3403_340338

class FiniteSquareRing (R : Type) extends Ring R where
  finite : Finite R
  square_sum_is_square : ∀ a b : R, ∃ c : R, a ^ 2 + b ^ 2 = c ^ 2

theorem square_product_theorem {R : Type} [FiniteSquareRing R] :
  ∀ a b c : R, ∃ d : R, 2 * a * b * c = d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_theorem_l3403_340338


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3403_340348

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 4 = 16)
  (h_sum : a 1 + a 5 = 17) :
  a 3 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3403_340348


namespace NUMINAMATH_CALUDE_fraction_simplification_l3403_340314

theorem fraction_simplification (a x : ℝ) (ha : a > 0) (hx : x > 0) :
  (a * Real.sqrt x - x * Real.sqrt a) / (Real.sqrt a - Real.sqrt x) = Real.sqrt (a * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3403_340314


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l3403_340304

/-- Two natural numbers are consecutive odd numbers -/
def ConsecutiveOddNumbers (p q : ℕ) : Prop :=
  ∃ k : ℕ, p = 2*k + 1 ∧ q = 2*k + 3

/-- A number a is divisible by a number b -/
def IsDivisibleBy (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem consecutive_odd_numbers_divisibility (p q : ℕ) :
  ConsecutiveOddNumbers p q → IsDivisibleBy (p^q + q^p) (p + q) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l3403_340304


namespace NUMINAMATH_CALUDE_power_of_five_sum_equality_l3403_340326

theorem power_of_five_sum_equality (x : ℕ) : 5^6 + 5^6 + 5^6 = 5^x ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_sum_equality_l3403_340326


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l3403_340346

def million : ℝ := 1000000

theorem scientific_notation_of_56_99_million :
  56.99 * million = 5.699 * (10 : ℝ) ^ 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l3403_340346


namespace NUMINAMATH_CALUDE_minimum_framing_for_specific_photo_l3403_340396

/-- Calculates the minimum framing needed for a scaled photograph with a border -/
def minimum_framing (original_width original_height scale_factor border_width : ℕ) : ℕ :=
  let scaled_width := original_width * scale_factor
  let scaled_height := original_height * scale_factor
  let total_width := scaled_width + 2 * border_width
  let total_height := scaled_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + 11) / 12  -- Round up to the nearest foot

theorem minimum_framing_for_specific_photo :
  minimum_framing 5 7 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_minimum_framing_for_specific_photo_l3403_340396


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3403_340376

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3403_340376


namespace NUMINAMATH_CALUDE_puppies_brought_in_puppies_brought_in_solution_l3403_340340

theorem puppies_brought_in (initial_puppies : ℕ) (adoption_rate : ℕ) (adoption_days : ℕ) : ℕ :=
  let total_adopted := adoption_rate * adoption_days
  total_adopted - initial_puppies

theorem puppies_brought_in_solution :
  puppies_brought_in 2 4 9 = 34 := by
  sorry

end NUMINAMATH_CALUDE_puppies_brought_in_puppies_brought_in_solution_l3403_340340


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3403_340352

theorem inequalities_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ ((a - c)^c < (b - c)^c) ∧ (b * Real.exp a > a * Real.exp b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3403_340352


namespace NUMINAMATH_CALUDE_a_n_properties_l3403_340378

def a_n (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2^n + 1 else 2^n - 1

theorem a_n_properties : ∀ n : ℕ,
  (∃ m : ℕ, if n % 2 = 0 then a_n n = 5 * m^2 else a_n n = m^2) :=
by sorry

end NUMINAMATH_CALUDE_a_n_properties_l3403_340378


namespace NUMINAMATH_CALUDE_ivan_purchase_cost_l3403_340341

/-- Calculates the total cost of a discounted purchase -/
def discounted_purchase_cost (original_price discount quantity : ℕ) : ℕ :=
  (original_price - discount) * quantity

/-- Proves that the total cost for Ivan's purchase is $100 -/
theorem ivan_purchase_cost :
  discounted_purchase_cost 12 2 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ivan_purchase_cost_l3403_340341


namespace NUMINAMATH_CALUDE_profit_achieved_min_disks_optimal_l3403_340306

/-- The number of disks Maria buys for $6 -/
def buy_rate : ℕ := 5

/-- The price Maria pays for buy_rate disks -/
def buy_price : ℚ := 6

/-- The number of disks Maria sells for $7 -/
def sell_rate : ℕ := 4

/-- The price Maria receives for sell_rate disks -/
def sell_price : ℚ := 7

/-- The target profit Maria wants to achieve -/
def target_profit : ℚ := 120

/-- The minimum number of disks Maria must sell to make the target profit -/
def min_disks_to_sell : ℕ := 219

theorem profit_achieved (n : ℕ) : 
  n ≥ min_disks_to_sell → 
  n * (sell_price / sell_rate - buy_price / buy_rate) ≥ target_profit :=
by sorry

theorem min_disks_optimal : 
  ∀ m : ℕ, m < min_disks_to_sell → 
  m * (sell_price / sell_rate - buy_price / buy_rate) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_achieved_min_disks_optimal_l3403_340306


namespace NUMINAMATH_CALUDE_vector_magnitude_l3403_340370

theorem vector_magnitude (a b : ℝ × ℝ) : 
  let angle := π / 6
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  dot_product = 3 ∧ 
  magnitude_a = 3 →
  Real.sqrt (b.1^2 + b.2^2) = (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3403_340370


namespace NUMINAMATH_CALUDE_system_solution_l3403_340349

theorem system_solution : ∃! (x y : ℝ), x - y = -5 ∧ 3 * x + 2 * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3403_340349


namespace NUMINAMATH_CALUDE_rectangle_area_l3403_340337

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  length = 4 * width → 
  2 * length + 2 * width = 200 → 
  length * width = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3403_340337


namespace NUMINAMATH_CALUDE_irrational_approximation_l3403_340398

theorem irrational_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| < 1 / q^2 := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l3403_340398


namespace NUMINAMATH_CALUDE_triangle_properties_l3403_340321

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (B = π / 4 → Real.sqrt 3 * b = Real.sqrt 2 * a) ∧
  (a = Real.sqrt 3 ∧ b + c = 3 → b * c = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3403_340321


namespace NUMINAMATH_CALUDE_mn_equation_solutions_l3403_340367

theorem mn_equation_solutions (m n : ℤ) : 
  m^2 * n^2 + m^2 + n^2 + 10*m*n + 16 = 0 ↔ (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_mn_equation_solutions_l3403_340367


namespace NUMINAMATH_CALUDE_smallest_odd_probability_l3403_340374

/-- The probability that the smallest number in a lottery draw is odd -/
theorem smallest_odd_probability (n : ℕ) (k : ℕ) (h1 : n = 90) (h2 : k = 5) :
  let prob := (1 : ℚ) / 2 + (44 : ℚ) * (Nat.choose 45 3 : ℚ) / (2 * (Nat.choose n k : ℚ))
  ∃ (ε : ℚ), abs (prob - 0.5142) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_odd_probability_l3403_340374


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l3403_340357

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → y^2 + 1/y^2 = 13 → x + 1/x ≥ y + 1/y) ∧ x + 1/x = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l3403_340357


namespace NUMINAMATH_CALUDE_expression_equality_l3403_340311

/-- Given two real numbers a and b, prove that the expression
    "the difference between three times the number for A and the number for B
    divided by the sum of the number for A and twice the number for B"
    is equal to (3a - b) / (a + 2b) -/
theorem expression_equality (a b : ℝ) : 
  (3 * a - b) / (a + 2 * b) = 
  (3 * a - b) / (a + 2 * b) := by sorry

end NUMINAMATH_CALUDE_expression_equality_l3403_340311


namespace NUMINAMATH_CALUDE_trays_needed_to_replace_ice_l3403_340305

def ice_cubes_in_glass : ℕ := 8
def ice_cubes_in_pitcher : ℕ := 2 * ice_cubes_in_glass
def spaces_per_tray : ℕ := 12

theorem trays_needed_to_replace_ice : 
  (ice_cubes_in_glass + ice_cubes_in_pitcher) / spaces_per_tray = 2 := by
  sorry

end NUMINAMATH_CALUDE_trays_needed_to_replace_ice_l3403_340305


namespace NUMINAMATH_CALUDE_prob_six_heads_and_return_l3403_340332

/-- The number of nodes in the circular arrangement -/
def num_nodes : ℕ := 5

/-- The total number of coin flips -/
def num_flips : ℕ := 12

/-- The number of heads we're interested in -/
def target_heads : ℕ := 6

/-- Represents the movement on the circular arrangement -/
def net_movement (heads : ℕ) : ℤ :=
  (heads : ℤ) - (num_flips - heads : ℤ)

/-- The condition for returning to the starting node -/
def returns_to_start (heads : ℕ) : Prop :=
  net_movement heads % (num_nodes : ℤ) = 0

/-- The probability of flipping exactly 'heads' number of heads in 'num_flips' flips -/
def prob_heads (heads : ℕ) : ℚ :=
  (Nat.choose num_flips heads : ℚ) / 2^num_flips

/-- The main theorem to prove -/
theorem prob_six_heads_and_return :
  returns_to_start target_heads ∧ prob_heads target_heads = 231 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_prob_six_heads_and_return_l3403_340332


namespace NUMINAMATH_CALUDE_married_men_fraction_l3403_340329

theorem married_men_fraction (total_women : ℕ) (single_women : ℕ) :
  single_women = (3 : ℕ) * total_women / 7 →
  (total_women - single_women) / (total_women + (total_women - single_women)) = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3403_340329


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l3403_340359

theorem modular_congruence_solution : ∃! n : ℕ, 1 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l3403_340359


namespace NUMINAMATH_CALUDE_q_is_false_l3403_340363

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
sorry

end NUMINAMATH_CALUDE_q_is_false_l3403_340363


namespace NUMINAMATH_CALUDE_min_floor_sum_l3403_340369

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  34 ≤ ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l3403_340369


namespace NUMINAMATH_CALUDE_mass_of_compound_l3403_340335

/-- The mass of a compound given its molecular weight and number of moles. -/
def mass (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

/-- Theorem: The mass of 7 moles of a compound with a molecular weight of 588 g/mol is 4116 g. -/
theorem mass_of_compound : mass 588 7 = 4116 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_compound_l3403_340335


namespace NUMINAMATH_CALUDE_evaluate_complex_expression_l3403_340309

theorem evaluate_complex_expression :
  ∃ (x : ℝ), x > 0 ∧ x^2 = (297 - 99*Real.sqrt 5 + 108*Real.sqrt 6 - 36*Real.sqrt 30) / 64 ∧
  x = (3*(Real.sqrt 3 + Real.sqrt 8)) / (4*Real.sqrt (3 + Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_evaluate_complex_expression_l3403_340309


namespace NUMINAMATH_CALUDE_sum_of_roots_is_12_l3403_340358

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (3 + x) = g (3 - x)

-- Define a proposition that g has exactly four distinct real roots
def has_four_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d))

-- The theorem statement
theorem sum_of_roots_is_12 (g : ℝ → ℝ) 
    (h1 : symmetric_about_3 g) 
    (h2 : has_four_distinct_roots g) : 
  ∃ a b c d : ℝ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (a + b + c + d = 12) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_12_l3403_340358


namespace NUMINAMATH_CALUDE_total_supplies_is_1260_l3403_340373

/-- The total number of supplies given the number of rows and items per row -/
def total_supplies (rows : ℕ) (crayons_per_row : ℕ) (colored_pencils_per_row : ℕ) (graphite_pencils_per_row : ℕ) : ℕ :=
  rows * (crayons_per_row + colored_pencils_per_row + graphite_pencils_per_row)

/-- Theorem stating that the total number of supplies is 1260 -/
theorem total_supplies_is_1260 : total_supplies 28 12 15 18 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_total_supplies_is_1260_l3403_340373


namespace NUMINAMATH_CALUDE_north_pond_duck_count_l3403_340365

/-- The number of ducks in Lake Michigan -/
def lake_michigan_ducks : ℕ := 100

/-- The number of ducks in North Pond -/
def north_pond_ducks : ℕ := 2 * lake_michigan_ducks + 6

/-- Theorem stating that North Pond has 206 ducks -/
theorem north_pond_duck_count : north_pond_ducks = 206 := by
  sorry

end NUMINAMATH_CALUDE_north_pond_duck_count_l3403_340365


namespace NUMINAMATH_CALUDE_touching_circle_radius_l3403_340307

/-- A circle touching two semicircles and a line segment --/
structure TouchingCircle where
  /-- Radius of the larger semicircle -/
  R : ℝ
  /-- Radius of the smaller semicircle -/
  r : ℝ
  /-- Radius of the touching circle -/
  x : ℝ
  /-- The smaller semicircle's diameter is half of the larger one -/
  h1 : r = R / 2
  /-- The touching circle is tangent to both semicircles and the line segment -/
  h2 : x > 0 ∧ x < r

/-- The radius of the touching circle is 8 when the larger semicircle has diameter 36 -/
theorem touching_circle_radius (c : TouchingCircle) (h : c.R = 18) : c.x = 8 := by
  sorry

end NUMINAMATH_CALUDE_touching_circle_radius_l3403_340307


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l3403_340327

theorem largest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 7 + Nat.factorial 8) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 7 + Nat.factorial 8) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l3403_340327


namespace NUMINAMATH_CALUDE_quartet_characterization_l3403_340319

def is_valid_quartet (a b c d : ℕ+) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b = c * d ∧ a * b = c + d

def valid_quartets : List (ℕ+ × ℕ+ × ℕ+ × ℕ+) :=
  [(1, 5, 3, 2), (1, 5, 2, 3), (5, 1, 3, 2), (5, 1, 2, 3),
   (2, 3, 1, 5), (3, 2, 1, 5), (2, 3, 5, 1), (3, 2, 5, 1)]

theorem quartet_characterization (a b c d : ℕ+) :
  is_valid_quartet a b c d ↔ (a, b, c, d) ∈ valid_quartets :=
sorry

end NUMINAMATH_CALUDE_quartet_characterization_l3403_340319


namespace NUMINAMATH_CALUDE_copper_part_mass_l3403_340388

/-- Given two parts with equal volume, one made of aluminum and one made of copper,
    prove that the mass of the copper part is approximately 0.086 kg. -/
theorem copper_part_mass
  (ρ_A : Real) -- density of aluminum
  (ρ_M : Real) -- density of copper
  (Δm : Real)  -- mass difference between parts
  (h1 : ρ_A = 2700) -- density of aluminum in kg/m³
  (h2 : ρ_M = 8900) -- density of copper in kg/m³
  (h3 : Δm = 0.06)  -- mass difference in kg
  : ∃ (m_M : Real), abs (m_M - 0.086) < 0.001 ∧ 
    ∃ (V : Real), V > 0 ∧ V = m_M / ρ_M ∧ V = (m_M - Δm) / ρ_A :=
by sorry

end NUMINAMATH_CALUDE_copper_part_mass_l3403_340388


namespace NUMINAMATH_CALUDE_smallest_n_for_factorial_sum_l3403_340320

def lastFourDigits (n : ℕ) : ℕ := n % 10000

def isValidSequence (seq : List ℕ) : Prop :=
  ∀ x ∈ seq, x ≤ 15 ∧ x > 0

theorem smallest_n_for_factorial_sum : 
  (∃ (seq : List ℕ), 
    seq.length = 3 ∧ 
    isValidSequence seq ∧ 
    lastFourDigits (seq.map Nat.factorial).sum = 2001) ∧ 
  (∀ (n : ℕ) (seq : List ℕ), 
    n < 3 → 
    seq.length = n → 
    isValidSequence seq → 
    lastFourDigits (seq.map Nat.factorial).sum ≠ 2001) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorial_sum_l3403_340320


namespace NUMINAMATH_CALUDE_magic_triangle_max_sum_l3403_340345

theorem magic_triangle_max_sum (a b c d e f : ℕ) : 
  a ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  b ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  c ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  d ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  e ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  f ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  a + b + c = c + d + e ∧ c + d + e = e + f + a →
  a + b + c ≤ 48 := by
sorry

end NUMINAMATH_CALUDE_magic_triangle_max_sum_l3403_340345


namespace NUMINAMATH_CALUDE_F_zeros_and_reciprocal_sum_l3403_340391

noncomputable def F (x : ℝ) : ℝ := 1 / (2 * x) + Real.log (x / 2)

theorem F_zeros_and_reciprocal_sum :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ F x₁ = 0 ∧ F x₂ = 0 ∧
    (∀ (x : ℝ), x > 0 ∧ F x = 0 → x = x₁ ∨ x = x₂)) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ F x₁ = 0 ∧ F x₂ = 0 →
    1 / x₁ + 1 / x₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_F_zeros_and_reciprocal_sum_l3403_340391


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l3403_340336

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_22 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 22 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 22 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l3403_340336


namespace NUMINAMATH_CALUDE_expression_evaluation_l3403_340334

theorem expression_evaluation (b x : ℝ) (h : x = b + 9) :
  2*x - b + 5 = b + 23 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3403_340334


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3403_340386

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3403_340386
