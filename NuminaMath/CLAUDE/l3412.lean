import Mathlib

namespace NUMINAMATH_CALUDE_fraction_inequality_l3412_341200

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3412_341200


namespace NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coeff_l3412_341299

theorem quadratic_rational_root_implies_even_coeff 
  (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ) (h_q_nonzero : q ≠ 0), a * (p * p) + b * (p * q) + c * (q * q) = 0) :
  Even a ∨ Even b ∨ Even c := by
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coeff_l3412_341299


namespace NUMINAMATH_CALUDE_puzzle_solution_l3412_341259

/-- Represents the possible values in a cell of the grid -/
inductive CellValue
  | Two
  | Zero
  | One
  | Five
  | Empty

/-- Represents a 5x6 grid -/
def Grid := Matrix (Fin 5) (Fin 6) CellValue

/-- Checks if a given grid satisfies the puzzle constraints -/
def is_valid_grid (g : Grid) : Prop :=
  -- Each row contains each digit exactly once
  (∀ i, ∃! j, g i j = CellValue.Two) ∧
  (∀ i, ∃! j, g i j = CellValue.Zero) ∧
  (∀ i, ∃! j, g i j = CellValue.One) ∧
  (∀ i, ∃! j, g i j = CellValue.Five) ∧
  -- Each column contains each digit exactly once
  (∀ j, ∃! i, g i j = CellValue.Two) ∧
  (∀ j, ∃! i, g i j = CellValue.Zero) ∧
  (∀ j, ∃! i, g i j = CellValue.One) ∧
  (∀ j, ∃! i, g i j = CellValue.Five) ∧
  -- Same digits are not adjacent diagonally
  (∀ i j, i < 4 → j < 5 → g i j ≠ g (i+1) (j+1)) ∧
  (∀ i j, i < 4 → j > 0 → g i j ≠ g (i+1) (j-1))

/-- The theorem stating the solution to the puzzle -/
theorem puzzle_solution (g : Grid) (h : is_valid_grid g) :
  g 4 0 = CellValue.One ∧
  g 4 1 = CellValue.Five ∧
  g 4 2 = CellValue.Empty ∧
  g 4 3 = CellValue.Empty ∧
  g 4 4 = CellValue.Two :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3412_341259


namespace NUMINAMATH_CALUDE_jacks_kids_l3412_341280

theorem jacks_kids (shirts_per_kid : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) : 
  shirts_per_kid = 3 → buttons_per_shirt = 7 → total_buttons = 63 →
  ∃ (num_kids : ℕ), num_kids * shirts_per_kid * buttons_per_shirt = total_buttons ∧ num_kids = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jacks_kids_l3412_341280


namespace NUMINAMATH_CALUDE_distance_difference_l3412_341276

/-- Represents the distance traveled by a biker given their speed and time. -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Camila's constant speed in miles per hour. -/
def camila_speed : ℝ := 15

/-- Represents Daniel's initial speed in miles per hour. -/
def daniel_initial_speed : ℝ := 15

/-- Represents Daniel's reduced speed in miles per hour. -/
def daniel_reduced_speed : ℝ := 10

/-- Represents the total time of the bike ride in hours. -/
def total_time : ℝ := 6

/-- Represents the time at which Daniel's speed changes in hours. -/
def speed_change_time : ℝ := 3

/-- Calculates the distance Camila travels in 6 hours. -/
def camila_distance : ℝ := distance camila_speed total_time

/-- Calculates the distance Daniel travels in 6 hours. -/
def daniel_distance : ℝ := 
  distance daniel_initial_speed speed_change_time + 
  distance daniel_reduced_speed (total_time - speed_change_time)

theorem distance_difference : camila_distance - daniel_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3412_341276


namespace NUMINAMATH_CALUDE_horizontal_axis_independent_l3412_341271

/-- Represents the different types of variables in a graph --/
inductive AxisVariable
  | Dependent
  | Constant
  | Independent
  | Function

/-- Represents a standard graph showing relationships between variables --/
structure StandardGraph where
  horizontalAxis : AxisVariable
  verticalAxis : AxisVariable

/-- Theorem stating that the horizontal axis in a standard graph usually represents the independent variable --/
theorem horizontal_axis_independent (g : StandardGraph) : g.horizontalAxis = AxisVariable.Independent := by
  sorry

end NUMINAMATH_CALUDE_horizontal_axis_independent_l3412_341271


namespace NUMINAMATH_CALUDE_invalid_vote_percentage_l3412_341244

theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_A_share : ℚ)
  (candidate_A_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_A_share = 60 / 100)
  (h3 : candidate_A_votes = 285600) :
  (total_votes - (candidate_A_votes / candidate_A_share : ℚ)) / total_votes = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_invalid_vote_percentage_l3412_341244


namespace NUMINAMATH_CALUDE_complex_inequality_l3412_341203

theorem complex_inequality (z₁ z₂ z₃ z₄ : ℂ) :
  ‖z₁ - z₃‖^2 + ‖z₂ - z₄‖^2 ≤ ‖z₁ - z₂‖^2 + ‖z₂ - z₃‖^2 + ‖z₃ - z₄‖^2 + ‖z₄ - z₁‖^2 ∧
  (‖z₁ - z₃‖^2 + ‖z₂ - z₄‖^2 = ‖z₁ - z₂‖^2 + ‖z₂ - z₃‖^2 + ‖z₃ - z₄‖^2 + ‖z₄ - z₁‖^2 ↔ z₁ + z₃ = z₂ + z₄) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_l3412_341203


namespace NUMINAMATH_CALUDE_art_group_size_l3412_341202

/-- The number of students in the art interest group -/
def num_students : ℕ := 6

/-- The total number of colored papers when each student cuts 10 pieces -/
def total_papers_10 (x : ℕ) : ℕ := 10 * x + 6

/-- The total number of colored papers when each student cuts 12 pieces -/
def total_papers_12 (x : ℕ) : ℕ := 12 * x - 6

/-- Theorem stating that the number of students satisfies the given conditions -/
theorem art_group_size :
  total_papers_10 num_students = total_papers_12 num_students :=
by sorry

end NUMINAMATH_CALUDE_art_group_size_l3412_341202


namespace NUMINAMATH_CALUDE_selection_methods_five_three_two_l3412_341209

/-- The number of ways to select 3 students out of 5 for 3 different language majors,
    where 2 specific students cannot be selected for one particular major -/
def selection_methods (n : ℕ) (k : ℕ) (excluded : ℕ) : ℕ :=
  Nat.choose (n - excluded) 1 * (n - 1).factorial / (n - k).factorial

theorem selection_methods_five_three_two :
  selection_methods 5 3 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_five_three_two_l3412_341209


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3412_341255

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3412_341255


namespace NUMINAMATH_CALUDE_euler_line_equation_l3412_341273

/-- Triangle ABC with vertices A(3,1), B(4,2), and C(2,3) -/
structure Triangle where
  A : ℝ × ℝ := (3, 1)
  B : ℝ × ℝ := (4, 2)
  C : ℝ × ℝ := (2, 3)

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + y - 5 = 0

/-- Theorem: The equation of the Euler line for the given triangle ABC is x + y - 5 = 0 -/
theorem euler_line_equation (t : Triangle) : EulerLine t = fun x y => x + y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_euler_line_equation_l3412_341273


namespace NUMINAMATH_CALUDE_triangle_line_equations_l3412_341265

/-- Triangle with vertices A(4, 0), B(6, 7), and C(0, 3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line passing through the midpoints of sides BC and AB -/
def midpointLine (t : Triangle) : LineEquation :=
  { a := 3, b := 4, c := -29 }

/-- The perpendicular bisector of side BC -/
def perpendicularBisector (t : Triangle) : LineEquation :=
  { a := 3, b := 2, c := -19 }

theorem triangle_line_equations (t : Triangle) :
  (midpointLine t = { a := 3, b := 4, c := -29 }) ∧
  (perpendicularBisector t = { a := 3, b := 2, c := -19 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l3412_341265


namespace NUMINAMATH_CALUDE_constant_expression_l3412_341241

theorem constant_expression (x y : ℝ) (h : x + y = 1) :
  let a := Real.sqrt (1 + x^2)
  let b := Real.sqrt (1 + y^2)
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l3412_341241


namespace NUMINAMATH_CALUDE_next_square_property_l3412_341277

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_square_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_square_property : 
  ∀ n : ℕ, n > 1818 → has_square_property n → n ≥ 1832 :=
sorry

end NUMINAMATH_CALUDE_next_square_property_l3412_341277


namespace NUMINAMATH_CALUDE_two_xy_equals_seven_l3412_341282

theorem two_xy_equals_seven (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (2 : ℝ)^(x+y) = 64)
  (h2 : (9 : ℝ)^(x+y) / (3 : ℝ)^(4*y) = 243) :
  2 * x * y = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_xy_equals_seven_l3412_341282


namespace NUMINAMATH_CALUDE_elephant_weight_l3412_341211

theorem elephant_weight (elephant_weight : ℝ) (donkey_weight : ℝ) : 
  elephant_weight * 2000 + donkey_weight = 6600 →
  donkey_weight = 0.1 * (elephant_weight * 2000) →
  elephant_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_elephant_weight_l3412_341211


namespace NUMINAMATH_CALUDE_factorization_equality_l3412_341285

theorem factorization_equality (x y : ℝ) : 
  x^2 - y^2 + 3*x - y + 2 = (x + y + 2)*(x - y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3412_341285


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l3412_341228

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l3412_341228


namespace NUMINAMATH_CALUDE_student_proportion_is_frequency_rate_l3412_341208

/-- Represents the total number of people in the population -/
def total_population : ℕ := 10

/-- Represents the number of students in the population -/
def number_of_students : ℕ := 4

/-- Represents the proportion of students in the population -/
def student_proportion : ℚ := 2 / 5

/-- Defines what a frequency rate is in this context -/
def is_frequency_rate (proportion : ℚ) (num : ℕ) (total : ℕ) : Prop :=
  proportion = num / total

/-- Theorem stating that the given proportion is a frequency rate -/
theorem student_proportion_is_frequency_rate :
  is_frequency_rate student_proportion number_of_students total_population := by
  sorry

end NUMINAMATH_CALUDE_student_proportion_is_frequency_rate_l3412_341208


namespace NUMINAMATH_CALUDE_different_color_pairings_l3412_341213

/-- The number of distinct colors for bowls and glasses -/
def num_colors : ℕ := 5

/-- The number of pairings where the bowl and glass colors are different -/
def num_different_pairings : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating that the number of different color pairings is 20 -/
theorem different_color_pairings :
  num_different_pairings = 20 := by
  sorry

#eval num_different_pairings -- This should output 20

end NUMINAMATH_CALUDE_different_color_pairings_l3412_341213


namespace NUMINAMATH_CALUDE_expression_equals_95_l3412_341250

theorem expression_equals_95 : 
  let some_number := -5765435
  7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + some_number = 95 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_95_l3412_341250


namespace NUMINAMATH_CALUDE_abs_positive_for_nonzero_l3412_341233

theorem abs_positive_for_nonzero (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_positive_for_nonzero_l3412_341233


namespace NUMINAMATH_CALUDE_coin_selection_probability_l3412_341237

/-- Represents the placement of boxes in drawers -/
inductive BoxPlacement
  | AloneInDrawer
  | WithOneOther
  | Random

/-- Probability of selecting the coin-containing box given a placement -/
def probability (placement : BoxPlacement) : ℚ :=
  match placement with
  | BoxPlacement.AloneInDrawer => 1/2
  | BoxPlacement.WithOneOther => 1/4
  | BoxPlacement.Random => 1/3

theorem coin_selection_probability 
  (boxes : Nat) 
  (drawers : Nat) 
  (coin_box : Nat) 
  (h1 : boxes = 3) 
  (h2 : drawers = 2) 
  (h3 : coin_box = 1) 
  (h4 : ∀ d, d ≤ drawers → d > 0 → ∃ b, b ≤ boxes ∧ b > 0) :
  (probability BoxPlacement.AloneInDrawer = 1/2) ∧
  (probability BoxPlacement.WithOneOther = 1/4) ∧
  (probability BoxPlacement.Random = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_coin_selection_probability_l3412_341237


namespace NUMINAMATH_CALUDE_soda_cost_is_one_l3412_341222

/-- The cost of one can of soda -/
def soda_cost : ℝ := 1

/-- The cost of one soup -/
def soup_cost : ℝ := 3 * soda_cost

/-- The cost of one sandwich -/
def sandwich_cost : ℝ := 3 * soup_cost

/-- The total cost of Sean's purchase -/
def total_cost : ℝ := 3 * soda_cost + 2 * soup_cost + sandwich_cost

theorem soda_cost_is_one :
  soda_cost = 1 ∧ total_cost = 18 := by sorry

end NUMINAMATH_CALUDE_soda_cost_is_one_l3412_341222


namespace NUMINAMATH_CALUDE_tan_sqrt_three_solution_l3412_341248

theorem tan_sqrt_three_solution (x : ℝ) : 
  Real.tan x = Real.sqrt 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sqrt_three_solution_l3412_341248


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3412_341266

theorem inequality_system_solution (m : ℝ) :
  (∀ x : ℝ, (3 * x - 9 > 0 ∧ x > m) ↔ x > 3) →
  m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3412_341266


namespace NUMINAMATH_CALUDE_train_A_speed_l3412_341251

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 38

/-- The time difference between Train A and Train B departures in hours -/
def time_diff : ℝ := 2

/-- The distance from the station where Train B overtakes Train A in miles -/
def overtake_distance : ℝ := 285

/-- Theorem stating that the speed of Train A is 30 miles per hour -/
theorem train_A_speed :
  speed_A = 30 ∧
  speed_A * (overtake_distance / speed_B + time_diff) = overtake_distance :=
sorry

end NUMINAMATH_CALUDE_train_A_speed_l3412_341251


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3412_341264

/-- The discriminant of a quadratic equation ax² + bx + c is equal to b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x² + (5 + 1/5)x + 1/5 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/5
def c : ℚ := 1/5

theorem quadratic_discriminant : discriminant a b c = 576/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3412_341264


namespace NUMINAMATH_CALUDE_interest_calculation_l3412_341206

/-- Calculates the total interest earned after 4 years given an initial investment
    and annual interest rates for each year. -/
def total_interest (initial_investment : ℝ) (rate1 rate2 rate3 rate4 : ℝ) : ℝ :=
  let final_amount := initial_investment * (1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4)
  final_amount - initial_investment

/-- Proves that the total interest earned after 4 years is approximately $572.36416
    given the specified initial investment and interest rates. -/
theorem interest_calculation :
  let initial_investment := 2000
  let rate1 := 0.05
  let rate2 := 0.06
  let rate3 := 0.07
  let rate4 := 0.08
  abs (total_interest initial_investment rate1 rate2 rate3 rate4 - 572.36416) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l3412_341206


namespace NUMINAMATH_CALUDE_max_food_per_guest_l3412_341261

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (h1 : total_food = 323) (h2 : min_guests = 162) :
  (total_food / min_guests : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l3412_341261


namespace NUMINAMATH_CALUDE_cube_edge_color_probability_l3412_341274

theorem cube_edge_color_probability :
  let num_edges : ℕ := 12
  let num_colors : ℕ := 2
  let num_visible_faces : ℕ := 4
  let prob_same_color_face : ℝ := 2 / 2^4

  (1 : ℝ) / 256 = prob_same_color_face^num_visible_faces := by sorry

end NUMINAMATH_CALUDE_cube_edge_color_probability_l3412_341274


namespace NUMINAMATH_CALUDE_cone_volume_contradiction_l3412_341275

theorem cone_volume_contradiction (base_area height volume : ℝ) : 
  base_area = 9 → height = 5 → volume = 45 → (1/3) * base_area * height ≠ volume :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_contradiction_l3412_341275


namespace NUMINAMATH_CALUDE_first_student_guess_l3412_341272

/-- Represents the number of jellybeans guessed by each student -/
structure JellybeanGuesses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the jellybean guessing problem -/
def jellybean_problem (g : JellybeanGuesses) : Prop :=
  g.second = 8 * g.first ∧
  g.third = g.second - 200 ∧
  g.fourth = (g.first + g.second + g.third) / 3 + 25 ∧
  g.fourth = 525

/-- Theorem stating that the first student's guess is 100 jellybeans -/
theorem first_student_guess :
  ∀ g : JellybeanGuesses, jellybean_problem g → g.first = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_first_student_guess_l3412_341272


namespace NUMINAMATH_CALUDE_geometry_book_pages_l3412_341253

theorem geometry_book_pages :
  let new_edition : ℕ := 450
  let old_edition : ℕ := 340
  let deluxe_edition : ℕ := new_edition + old_edition + 125
  (2 * old_edition - 230 = new_edition) ∧
  (deluxe_edition ≥ old_edition + (old_edition / 10)) →
  old_edition = 340 :=
by sorry

end NUMINAMATH_CALUDE_geometry_book_pages_l3412_341253


namespace NUMINAMATH_CALUDE_sinusoidal_symmetric_center_l3412_341231

/-- Given a sinusoidal function with specific properties, prove that its symmetric center is at (-2π/3, 0) -/
theorem sinusoidal_symmetric_center 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h_omega_pos : ω > 0)
  (h_phi_bound : |φ| < π/2)
  (h_f_def : ∀ x, f x = Real.sin (ω * x + φ))
  (h_period : (2 * π) / ω = 4 * π)
  (h_max_at_pi_third : ∀ x, f x ≤ f (π/3)) :
  ∃ (y : ℝ), ∀ (x : ℝ), f (x - (-2*π/3)) = f (-x - (-2*π/3)) ∧ f (-2*π/3) = y :=
sorry

end NUMINAMATH_CALUDE_sinusoidal_symmetric_center_l3412_341231


namespace NUMINAMATH_CALUDE_scenario_one_registration_methods_scenario_two_registration_methods_scenario_three_registration_methods_l3412_341254

/- Define the number of students and events -/
def num_students : ℕ := 6
def num_events : ℕ := 3

/- Theorem for scenario 1 -/
theorem scenario_one_registration_methods :
  (num_events ^ num_students : ℕ) = 729 := by sorry

/- Theorem for scenario 2 -/
theorem scenario_two_registration_methods :
  (num_students * (num_students - 1) * (num_students - 2) : ℕ) = 120 := by sorry

/- Theorem for scenario 3 -/
theorem scenario_three_registration_methods :
  (num_students ^ num_events : ℕ) = 216 := by sorry

end NUMINAMATH_CALUDE_scenario_one_registration_methods_scenario_two_registration_methods_scenario_three_registration_methods_l3412_341254


namespace NUMINAMATH_CALUDE_f_period_three_f_applied_95_times_main_result_l3412_341293

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^3)^(1/3)

theorem f_period_three (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : f (f (f x)) = x :=
sorry

theorem f_applied_95_times (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (f^[95]) x = f (f x) :=
sorry

theorem main_result : (f^[95]) 19 = (1 - 1/19^3)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_f_period_three_f_applied_95_times_main_result_l3412_341293


namespace NUMINAMATH_CALUDE_equal_segments_l3412_341205

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (incenter : Point → Point → Point → Point)
variable (intersect_circle_line : Circle → Point → Point → Point)
variable (circle_through : Point → Point → Point → Circle)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem equal_segments 
  (A B C I X Y : Point) :
  I = incenter A B C →
  X = intersect_circle_line (circle_through A C I) B C →
  Y = intersect_circle_line (circle_through B C I) A C →
  length A Y = length B X :=
sorry

end NUMINAMATH_CALUDE_equal_segments_l3412_341205


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3412_341239

theorem trigonometric_expression_value (α : Real) (h : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3412_341239


namespace NUMINAMATH_CALUDE_line_perpendicular_condition_l3412_341230

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)
variable (linePerpPlane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_condition
  (a b : Line) (α β : Plane)
  (h1 : lineInPlane a α)
  (h2 : linePerpPlane b β)
  (h3 : parallel α β) :
  perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_condition_l3412_341230


namespace NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3412_341240

/-- If the lateral surface of a cone, when unfolded, forms a semicircle with an area of 2π,
    then the height of the cone is √3. -/
theorem cone_height_from_lateral_surface (r h : ℝ) : 
  r > 0 → h > 0 → 2 * π = π * (r^2 + h^2) → h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3412_341240


namespace NUMINAMATH_CALUDE_ratio_equality_l3412_341219

theorem ratio_equality (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x - 2*y + 3*z) / (x + y + z) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3412_341219


namespace NUMINAMATH_CALUDE_haploid_corn_triploid_watermelon_heritable_variation_l3412_341257

-- Define the sources of heritable variations
inductive HeritableVariationSource
  | GeneMutation
  | ChromosomalVariation
  | GeneRecombination

-- Define a structure for crop variations
structure CropVariation where
  name : String
  isChromosomalVariation : Bool

-- Define the property of being a heritable variation
def isHeritableVariation (source : HeritableVariationSource) : Prop :=
  match source with
  | HeritableVariationSource.GeneMutation => True
  | HeritableVariationSource.ChromosomalVariation => True
  | HeritableVariationSource.GeneRecombination => True

-- Theorem statement
theorem haploid_corn_triploid_watermelon_heritable_variation 
  (haploidCorn triploidWatermelon : CropVariation)
  (haploidCornChromosomal : haploidCorn.isChromosomalVariation = true)
  (triploidWatermelonChromosomal : triploidWatermelon.isChromosomalVariation = true) :
  isHeritableVariation HeritableVariationSource.ChromosomalVariation := by
  sorry


end NUMINAMATH_CALUDE_haploid_corn_triploid_watermelon_heritable_variation_l3412_341257


namespace NUMINAMATH_CALUDE_quadratic_equations_root_difference_l3412_341214

theorem quadratic_equations_root_difference (k : ℝ) : 
  (∀ x, x^2 + k*x + 6 = 0 → ∃ y, y^2 - k*y + 6 = 0 ∧ y = x + 5) →
  (∀ y, y^2 - k*y + 6 = 0 → ∃ x, x^2 + k*x + 6 = 0 ∧ y = x + 5) →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_root_difference_l3412_341214


namespace NUMINAMATH_CALUDE_largest_fraction_l3412_341218

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 4/9, 3/8, 9/20]
  ∀ f ∈ fractions, (9:ℚ)/20 ≥ f := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3412_341218


namespace NUMINAMATH_CALUDE_sheet_area_difference_l3412_341234

/-- The combined area (front and back) of a rectangular sheet -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem sheet_area_difference :
  areaDifference 11 19 9.5 11 = 209 := by
  sorry

end NUMINAMATH_CALUDE_sheet_area_difference_l3412_341234


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l3412_341267

/-- Converts a ternary (base 3) number to decimal (base 10) --/
def ternary_to_decimal (a b c : ℕ) : ℕ :=
  a * 3^2 + b * 3^1 + c * 3^0

/-- The ternary number 121₃ is equal to 16 in decimal (base 10) --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l3412_341267


namespace NUMINAMATH_CALUDE_train_speed_proof_l3412_341204

-- Define the given parameters
def train_length : ℝ := 155
def bridge_length : ℝ := 220
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def m_s_to_km_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_proof :
  let total_distance := train_length + bridge_length
  let speed_m_s := total_distance / crossing_time
  let speed_km_hr := speed_m_s * m_s_to_km_hr
  speed_km_hr = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l3412_341204


namespace NUMINAMATH_CALUDE_addition_of_integers_l3412_341287

theorem addition_of_integers : -10 + 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_addition_of_integers_l3412_341287


namespace NUMINAMATH_CALUDE_factor_expression_l3412_341224

theorem factor_expression (y : ℝ) : 5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3412_341224


namespace NUMINAMATH_CALUDE_weight_of_pecans_l3412_341278

/-- Given the total weight of nuts and the weight of almonds, calculate the weight of pecans. -/
theorem weight_of_pecans (total_weight : ℝ) (almond_weight : ℝ) 
  (h1 : total_weight = 0.52) 
  (h2 : almond_weight = 0.14) : 
  total_weight - almond_weight = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_pecans_l3412_341278


namespace NUMINAMATH_CALUDE_percentage_less_l3412_341296

theorem percentage_less (w e y z : ℝ) 
  (hw : w = 0.60 * e) 
  (hz : z = 0.54 * y) 
  (hzw : z = 1.5000000000000002 * w) : 
  e = 0.60 * y := by
sorry

end NUMINAMATH_CALUDE_percentage_less_l3412_341296


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3412_341290

/-- 
Given a quadratic equation 5x^2 - 2x + 2 = 0, 
the coefficient of the linear term is -2 
-/
theorem linear_coefficient_of_quadratic (x : ℝ) : 
  (5 * x^2 - 2 * x + 2 = 0) → 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ b = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3412_341290


namespace NUMINAMATH_CALUDE_max_self_intersections_specific_cases_max_self_intersections_formula_l3412_341236

/-- Maximum number of self-intersection points for a closed polygonal chain -/
def max_self_intersections (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 3) / 2
  else
    n * (n - 4) / 2 + 1

/-- Theorem stating the maximum number of self-intersection points for specific cases -/
theorem max_self_intersections_specific_cases :
  (max_self_intersections 13 = 65) ∧ (max_self_intersections 1950 = 1898851) := by
  sorry

/-- Theorem for the general formula of maximum self-intersection points -/
theorem max_self_intersections_formula (n : ℕ) (h : n > 2) :
  max_self_intersections n = 
    if n % 2 = 1 then
      n * (n - 3) / 2
    else
      n * (n - 4) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_self_intersections_specific_cases_max_self_intersections_formula_l3412_341236


namespace NUMINAMATH_CALUDE_point_b_location_l3412_341297

/-- Represents a point on the number line -/
structure Point where
  value : ℝ

/-- The distance between two points on the number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_location (a b : Point) :
  a.value = -2 ∧ distance a b = 3 → b.value = -5 ∨ b.value = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_b_location_l3412_341297


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l3412_341245

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-1, 3)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (1, 3)

theorem reflection_across_y_axis :
  reflect_y original_point = reflected_point := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l3412_341245


namespace NUMINAMATH_CALUDE_two_new_players_joined_l3412_341270

/-- Given an initial group of players and some new players joining, 
    calculates the number of new players based on the total lives. -/
def new_players (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  (total_lives - initial_players * lives_per_player) / lives_per_player

/-- Proves that 2 new players joined the game given the initial conditions. -/
theorem two_new_players_joined :
  new_players 7 7 63 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_new_players_joined_l3412_341270


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l3412_341268

theorem taxi_ride_cost (uber_cost lyft_cost taxi_cost tip_percentage : ℝ) : 
  uber_cost = lyft_cost + 3 →
  lyft_cost = taxi_cost + 4 →
  uber_cost = 22 →
  tip_percentage = 0.2 →
  taxi_cost + (tip_percentage * taxi_cost) = 18 := by
sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l3412_341268


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3412_341212

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3412_341212


namespace NUMINAMATH_CALUDE_distance_swam_against_current_l3412_341242

/-- Calculates the distance swam against a current given swimming speed, current speed, and time taken. -/
def distance_against_current (swimming_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimming_speed - current_speed) * time

theorem distance_swam_against_current 
  (swimming_speed : ℝ) (current_speed : ℝ) (time : ℝ)
  (h1 : swimming_speed = 4)
  (h2 : current_speed = 2)
  (h3 : time = 5) :
  distance_against_current swimming_speed current_speed time = 10 := by
sorry

end NUMINAMATH_CALUDE_distance_swam_against_current_l3412_341242


namespace NUMINAMATH_CALUDE_count_numbers_with_seven_equals_133_l3412_341201

/-- Returns true if the given natural number contains the digit 7 at least once -/
def contains_seven (n : ℕ) : Bool := sorry

/-- Counts the number of natural numbers from 1 to 700 (inclusive) that contain the digit 7 at least once -/
def count_numbers_with_seven : ℕ := sorry

theorem count_numbers_with_seven_equals_133 : count_numbers_with_seven = 133 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_seven_equals_133_l3412_341201


namespace NUMINAMATH_CALUDE_max_difference_two_digit_numbers_l3412_341216

theorem max_difference_two_digit_numbers :
  ∀ (A B : ℕ),
  (10 ≤ A ∧ A ≤ 99) →
  (10 ≤ B ∧ B ≤ 99) →
  (2 * A = 7 * B / 3) →
  (∀ (C D : ℕ), (10 ≤ C ∧ C ≤ 99) → (10 ≤ D ∧ D ≤ 99) → (2 * C = 7 * D / 3) → (C - D ≤ A - B)) →
  A - B = 56 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_two_digit_numbers_l3412_341216


namespace NUMINAMATH_CALUDE_juice_problem_l3412_341232

theorem juice_problem (J : ℝ) : 
  J > 0 →
  (1/6 : ℝ) * J + (2/5 : ℝ) * (5/6 : ℝ) * J + (2/3 : ℝ) * (1/2 : ℝ) * J + 120 = J →
  J = 720 := by
sorry

end NUMINAMATH_CALUDE_juice_problem_l3412_341232


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l3412_341207

/-- Given a cylinder with original height 3 inches, if increasing the radius by 4 inches
    and the height by 6 inches results in the same new volume, then the original radius
    is 2 + 2√3 inches. -/
theorem cylinder_radius_problem (r : ℝ) : 
  (3 * π * (r + 4)^2 = 9 * π * r^2) → r = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l3412_341207


namespace NUMINAMATH_CALUDE_five_by_five_uncoverable_l3412_341258

/-- Represents a rectangular board -/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Represents a domino -/
structure Domino where
  width : ℕ
  height : ℕ

/-- Checks if a board can be completely covered by a given domino -/
def is_coverable (b : Board) (d : Domino) : Prop :=
  (b.rows * b.cols) % (d.width * d.height) = 0

/-- Theorem stating that a 5x5 board cannot be covered by 1x2 dominoes -/
theorem five_by_five_uncoverable :
  ¬ is_coverable (Board.mk 5 5) (Domino.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_five_by_five_uncoverable_l3412_341258


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l3412_341279

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads : ℝ
  sculptures : ℝ
  coins : ℝ
  silverCoins : ℝ
  goldCoins : ℝ

/-- Theorem stating the percentage of gold coins in the urn -/
theorem gold_coins_percentage (u : UrnComposition) 
  (beads_percent : u.beads = 0.3)
  (sculptures_percent : u.sculptures = 0.1)
  (total_percent : u.beads + u.sculptures + u.coins = 1)
  (silver_coins_percent : u.silverCoins = 0.3 * u.coins)
  (coins_composition : u.silverCoins + u.goldCoins = u.coins) : 
  u.goldCoins = 0.42 := by
  sorry


end NUMINAMATH_CALUDE_gold_coins_percentage_l3412_341279


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3412_341210

theorem complex_fraction_simplification : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324)) = 313 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3412_341210


namespace NUMINAMATH_CALUDE_expression_simplification_l3412_341263

theorem expression_simplification (m : ℝ) (h : m^2 - m - 1 = 0) :
  (m - 1) / (m^2 - 2*m) / (m + 1/(m - 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3412_341263


namespace NUMINAMATH_CALUDE_ladies_walking_distance_l3412_341215

theorem ladies_walking_distance (x y : ℝ) (h1 : x = 2 * y) (h2 : y = 4) :
  x + y = 12 := by sorry

end NUMINAMATH_CALUDE_ladies_walking_distance_l3412_341215


namespace NUMINAMATH_CALUDE_second_term_is_negative_x_cubed_l3412_341221

/-- A line on a two-dimensional coordinate plane defined by a = x^2 - x^3 -/
def line (x : ℝ) : ℝ := x^2 - x^3

/-- The line touches the x-axis in 2 places -/
axiom touches_x_axis_twice : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ line x₁ = 0 ∧ line x₂ = 0

/-- The second term of the equation representing the line is -x^3 -/
theorem second_term_is_negative_x_cubed :
  ∃ f : ℝ → ℝ, (∀ x, line x = f x - x^3) ∧ (∀ x, f x = x^2) :=
sorry

end NUMINAMATH_CALUDE_second_term_is_negative_x_cubed_l3412_341221


namespace NUMINAMATH_CALUDE_mode_of_student_dishes_l3412_341220

def student_dishes : List ℕ := [3, 5, 4, 6, 3, 3, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_student_dishes :
  mode student_dishes = 3 := by sorry

end NUMINAMATH_CALUDE_mode_of_student_dishes_l3412_341220


namespace NUMINAMATH_CALUDE_bernoulli_expectation_and_variance_l3412_341288

/-- A random variable with Bernoulli distribution -/
structure BernoulliRV where
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability mass function for Bernoulli distribution -/
def prob (ξ : BernoulliRV) (k : ℕ) : ℝ :=
  if k = 0 then 1 - ξ.p
  else if k = 1 then ξ.p
  else 0

/-- Expected value of a Bernoulli random variable -/
def expectation (ξ : BernoulliRV) : ℝ := ξ.p

/-- Variance of a Bernoulli random variable -/
def variance (ξ : BernoulliRV) : ℝ := (1 - ξ.p) * ξ.p

/-- Theorem: The expected value and variance of a Bernoulli random variable -/
theorem bernoulli_expectation_and_variance (ξ : BernoulliRV) :
  expectation ξ = ξ.p ∧ variance ξ = (1 - ξ.p) * ξ.p := by sorry

end NUMINAMATH_CALUDE_bernoulli_expectation_and_variance_l3412_341288


namespace NUMINAMATH_CALUDE_unique_value_of_expression_l3412_341243

theorem unique_value_of_expression (x y : ℝ) 
  (h : x * y - 3 * x / (y^2) - 3 * y / (x^2) = 7) : 
  (x - 2) * (y - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_of_expression_l3412_341243


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_A_intersect_B_equals_B_iff_a_less_than_0_l3412_341283

-- Define sets A and B
def A : Set ℝ := {x | x < -3 ∨ x ≥ 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 3}

-- Theorem 1
theorem complement_A_intersect_B_when_a_is_2 :
  (Set.univ \ A) ∩ B 2 = {x | -3 ≤ x ∧ x ≤ -1} := by sorry

-- Theorem 2
theorem A_intersect_B_equals_B_iff_a_less_than_0 (a : ℝ) :
  A ∩ B a = B a ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_A_intersect_B_equals_B_iff_a_less_than_0_l3412_341283


namespace NUMINAMATH_CALUDE_integral_of_improper_rational_function_l3412_341217

noncomputable def F (x : ℝ) : ℝ :=
  x^3 / 3 + x^2 - x + 
  (1 / (4 * Real.sqrt 2)) * Real.log ((x^2 - Real.sqrt 2 * x + 1) / (x^2 + Real.sqrt 2 * x + 1)) + 
  (1 / (2 * Real.sqrt 2)) * (Real.arctan (Real.sqrt 2 * x + 1) + Real.arctan (Real.sqrt 2 * x - 1))

theorem integral_of_improper_rational_function (x : ℝ) :
  deriv F x = (x^6 + 2*x^5 - x^4 + x^2 + 2*x) / (x^4 + 1) := by sorry

end NUMINAMATH_CALUDE_integral_of_improper_rational_function_l3412_341217


namespace NUMINAMATH_CALUDE_stock_percentage_value_l3412_341223

/-- Calculates the percentage value of a stock given its yield and price. -/
def percentageValue (yield : ℝ) (price : ℝ) : ℝ :=
  yield * 100

theorem stock_percentage_value :
  let yield : ℝ := 0.10
  let price : ℝ := 80
  percentageValue yield price = 10 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_value_l3412_341223


namespace NUMINAMATH_CALUDE_desired_depth_calculation_l3412_341247

/-- Calculates the desired depth to be dug given initial and new working conditions -/
theorem desired_depth_calculation
  (initial_men : ℕ)
  (initial_hours : ℕ)
  (initial_depth : ℝ)
  (new_hours : ℕ)
  (extra_men : ℕ)
  (h1 : initial_men = 72)
  (h2 : initial_hours = 8)
  (h3 : initial_depth = 30)
  (h4 : new_hours = 6)
  (h5 : extra_men = 88)
  : ∃ (desired_depth : ℝ), desired_depth = 50 := by
  sorry


end NUMINAMATH_CALUDE_desired_depth_calculation_l3412_341247


namespace NUMINAMATH_CALUDE_expectation_problem_l3412_341295

/-- Given E(X) + E(2X + 1) = 8, prove that E(X) = 7/3 -/
theorem expectation_problem (X : ℝ → ℝ) (E : (ℝ → ℝ) → ℝ) 
  (h : E X + E (λ x => 2 * X x + 1) = 8) :
  E X = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_expectation_problem_l3412_341295


namespace NUMINAMATH_CALUDE_dot_product_zero_nonzero_vectors_l3412_341238

theorem dot_product_zero_nonzero_vectors :
  ∃ (a b : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_zero_nonzero_vectors_l3412_341238


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3412_341269

theorem sqrt_expression_equality : 
  (Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 + Real.sqrt (1/2) = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3412_341269


namespace NUMINAMATH_CALUDE_tangent_line_of_even_cubic_l3412_341289

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a-3)x is an even function,
    then the equation of the tangent line to y = f(x) at (2, f(2)) is 9x - y - 16 = 0 -/
theorem tangent_line_of_even_cubic (a : ℝ) : 
  (∀ x, (x^3 + a*x^2 + (a-3)*x) = ((- x)^3 + a*(- x)^2 + (a-3)*(- x))) →
  ∃ m b, (m * 2 + b = 2^3 + a*2^2 + (a-3)*2) ∧ 
         (∀ x y, y = x^3 + a*x^2 + (a-3)*x → m*x - y - b = 0) ∧
         (m = 9 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_of_even_cubic_l3412_341289


namespace NUMINAMATH_CALUDE_cookie_batches_l3412_341281

/-- The number of batches of cookies made from one bag of chocolate chips -/
def num_batches (chips_per_cookie : ℕ) (chips_per_bag : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  chips_per_bag / (chips_per_cookie * cookies_per_batch)

/-- Theorem: The number of batches of cookies made from one bag of chocolate chips is 3 -/
theorem cookie_batches :
  num_batches 9 81 3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_batches_l3412_341281


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3412_341249

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) : ℝ) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 3 ∧ B = -9 ∧ C = -9 ∧ D = 9 ∧ E = 165 ∧ F = 51 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3412_341249


namespace NUMINAMATH_CALUDE_brenda_baking_days_l3412_341227

/-- Represents the number of cakes Brenda bakes per day -/
def cakes_per_day : ℕ := 20

/-- Represents the number of cakes Brenda has left after selling -/
def cakes_left : ℕ := 90

/-- Theorem: The number of days Brenda baked cakes is 9 -/
theorem brenda_baking_days : 
  ∃ (days : ℕ), 
    (cakes_per_day * days) / 2 = cakes_left ∧ 
    days = 9 := by
  sorry

end NUMINAMATH_CALUDE_brenda_baking_days_l3412_341227


namespace NUMINAMATH_CALUDE_circumcircle_radius_right_triangle_l3412_341235

/-- The radius of the circumcircle of a triangle with side lengths 8, 15, and 17 is 17/2 -/
theorem circumcircle_radius_right_triangle : 
  ∀ (a b c : ℝ), 
  a = 8 → b = 15 → c = 17 →
  a^2 + b^2 = c^2 →
  (∃ (r : ℝ), r = c / 2 ∧ r = 17 / 2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_right_triangle_l3412_341235


namespace NUMINAMATH_CALUDE_batting_average_calculation_l3412_341291

/-- Calculates the batting average given the total innings, highest score, score difference, and average excluding extremes -/
def batting_average (total_innings : ℕ) (highest_score : ℕ) (score_difference : ℕ) (avg_excluding_extremes : ℚ) : ℚ :=
  let lowest_score := highest_score - score_difference
  let runs_excluding_extremes := avg_excluding_extremes * (total_innings - 2)
  let total_runs := runs_excluding_extremes + highest_score + lowest_score
  total_runs / total_innings

theorem batting_average_calculation :
  batting_average 46 179 150 58 = 60 := by
  sorry

end NUMINAMATH_CALUDE_batting_average_calculation_l3412_341291


namespace NUMINAMATH_CALUDE_find_y_l3412_341226

theorem find_y : ∃ y : ℚ, 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3412_341226


namespace NUMINAMATH_CALUDE_washer_cost_l3412_341262

/-- Given a washer-dryer combination costing $1,200, where the washer costs $220 more than the dryer,
    prove that the cost of the washer is $710. -/
theorem washer_cost (total_cost dryer_cost washer_cost : ℕ) : 
  total_cost = 1200 →
  washer_cost = dryer_cost + 220 →
  total_cost = washer_cost + dryer_cost →
  washer_cost = 710 := by
sorry

end NUMINAMATH_CALUDE_washer_cost_l3412_341262


namespace NUMINAMATH_CALUDE_apples_used_correct_l3412_341292

/-- The number of apples used to make lunch in the school cafeteria -/
def apples_used : ℕ := 20

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := 23

/-- The number of apples bought after making lunch -/
def apples_bought : ℕ := 6

/-- The final number of apples in the cafeteria -/
def final_apples : ℕ := 9

/-- Theorem stating that the number of apples used for lunch is correct -/
theorem apples_used_correct : 
  initial_apples - apples_used + apples_bought = final_apples :=
by sorry

end NUMINAMATH_CALUDE_apples_used_correct_l3412_341292


namespace NUMINAMATH_CALUDE_part_one_part_two_l3412_341225

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem part_one : ∀ x : ℝ, f x - f (x + 2) < 1 ↔ x > -1/2 := by sorry

-- Theorem for part II
theorem part_two : (∀ x : ℝ, x ∈ Set.Icc 1 2 → x - f (x + 1 - a) ≤ 1) → (a ≤ 1 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3412_341225


namespace NUMINAMATH_CALUDE_inequality_always_holds_l3412_341286

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, 2 * m * x^2 + m * x - 3/4 < 0) → -6 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l3412_341286


namespace NUMINAMATH_CALUDE_piggy_bank_theorem_l3412_341260

/-- The value of a piggy bank containing dimes and quarters -/
def piggy_bank_value (num_dimes num_quarters : ℕ) (dime_value quarter_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value + (num_quarters : ℚ) * quarter_value

/-- Theorem: The value of a piggy bank with 35 dimes and 65 quarters is $19.75 -/
theorem piggy_bank_theorem :
  piggy_bank_value 35 65 (10 / 100) (25 / 100) = 1975 / 100 := by
  sorry

#eval piggy_bank_value 35 65 (10 / 100) (25 / 100)

end NUMINAMATH_CALUDE_piggy_bank_theorem_l3412_341260


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3412_341298

/-- A cubic function with specific properties -/
def f (a c d : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x + d

theorem cubic_function_properties (a c d : ℝ) (h_a : a ≠ 0) :
  (∀ x, f a c d x = -f a c d (-x)) →  -- f is odd
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a c d 1 ≤ f a c d x) →  -- f(1) is an extreme value
  (f a c d 1 = -2) →  -- f(1) = -2
  (∀ x, f a c d x = x^3 - 3*x) ∧  -- f(x) = x^3 - 3x
  (∀ x, f a c d x ≤ 2)  -- maximum value is 2
  := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3412_341298


namespace NUMINAMATH_CALUDE_walters_money_percentage_l3412_341284

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The total number of cents in Walter's pocket -/
def walters_money : ℕ := penny + 2 * nickel + dime + 2 * quarter

/-- Theorem: Walter's money is 71% of a dollar -/
theorem walters_money_percentage :
  (walters_money : ℚ) / 100 = 71 / 100 := by sorry

end NUMINAMATH_CALUDE_walters_money_percentage_l3412_341284


namespace NUMINAMATH_CALUDE_geometric_sequence_n_l3412_341229

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem geometric_sequence_n (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 → q = 2 → geometric_sequence a₁ q n = 64 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_n_l3412_341229


namespace NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l3412_341246

theorem half_plus_seven_equals_seventeen (n : ℝ) : (1/2 * n + 7 = 17) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l3412_341246


namespace NUMINAMATH_CALUDE_addition_to_reach_91_l3412_341256

theorem addition_to_reach_91 : ∃ x : ℚ, (5 * 12) / (180 / 3) + x = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_addition_to_reach_91_l3412_341256


namespace NUMINAMATH_CALUDE_division_problem_l3412_341252

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 55053 → 
  divisor = 456 → 
  remainder = 333 → 
  dividend = divisor * quotient + remainder → 
  quotient = 120 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3412_341252


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l3412_341294

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The standard form equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A line passing through a point (x₀, y₀) with slope m -/
structure Line where
  x₀ : ℝ
  y₀ : ℝ
  m : ℝ

/-- The equation of a line -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.m * (x - l.x₀) + l.y₀

theorem ellipse_and_line_theorem (e : Ellipse) 
  (h_triangle : e.a = 2 * Real.sqrt ((e.a^2 - e.b^2) / 4))
  (h_minor_axis : e.b = Real.sqrt 3) :
  (∃ (l : Line), l.x₀ = 0 ∧ l.y₀ = 2 ∧ 
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      e.equation x₁ y₁ ∧ 
      e.equation x₂ y₂ ∧
      l.equation x₁ y₁ ∧ 
      l.equation x₂ y₂ ∧
      x₁ ≠ x₂ ∧
      x₁ * x₂ + y₁ * y₂ = 2) ∧
    (l.m = Real.sqrt 2 / 2 ∨ l.m = -Real.sqrt 2 / 2)) ∧
  e.a = 2 ∧
  e.b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l3412_341294
