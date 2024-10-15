import Mathlib

namespace NUMINAMATH_CALUDE_concave_quadrilateral_perimeter_theorem_l3378_337834

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in 2D space -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Check if a point is inside a rectangle -/
def isInsideRectangle (p : Point) (r : Rectangle) : Prop :=
  r.topLeft.x ≤ p.x ∧ p.x ≤ r.bottomRight.x ∧
  r.bottomRight.y ≤ p.y ∧ p.y ≤ r.topLeft.y

/-- Check if a point is inside a triangle formed by three points -/
def isInsideTriangle (p : Point) (a b c : Point) : Prop :=
  sorry  -- Definition of point inside triangle

/-- Calculate the perimeter of a quadrilateral -/
def quadrilateralPerimeter (a b c d : Point) : ℝ :=
  sorry  -- Definition of quadrilateral perimeter

/-- Calculate the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ :=
  sorry  -- Definition of rectangle perimeter

theorem concave_quadrilateral_perimeter_theorem 
  (r : Rectangle) (a x y z : Point) :
  isInsideRectangle a r ∧ 
  isInsideRectangle x r ∧ 
  isInsideRectangle y r ∧
  isInsideTriangle z a x y →
  (quadrilateralPerimeter a x y z < rectanglePerimeter r) ∨
  (quadrilateralPerimeter a x z y < rectanglePerimeter r) ∨
  (quadrilateralPerimeter a y z x < rectanglePerimeter r) :=
by sorry

end NUMINAMATH_CALUDE_concave_quadrilateral_perimeter_theorem_l3378_337834


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3378_337850

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3378_337850


namespace NUMINAMATH_CALUDE_kelly_initial_games_l3378_337823

/-- The number of Nintendo games Kelly gave away -/
def games_given_away : ℕ := 64

/-- The number of Nintendo games Kelly has left -/
def games_left : ℕ := 42

/-- The initial number of Nintendo games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 106 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l3378_337823


namespace NUMINAMATH_CALUDE_mike_picked_seven_apples_l3378_337866

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Nancy ate -/
def nancy_ate : ℝ := 3.0

/-- The number of apples Keith picked -/
def keith_picked : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 10.0

/-- Theorem stating that Mike picked 7.0 apples -/
theorem mike_picked_seven_apples : 
  mike_apples = mike_apples - nancy_ate + keith_picked - apples_left + apples_left :=
by sorry

end NUMINAMATH_CALUDE_mike_picked_seven_apples_l3378_337866


namespace NUMINAMATH_CALUDE_subset_union_inclusion_fraction_inequality_ac_squared_eq_bc_squared_not_sufficient_negation_of_all_positive_real_l3378_337817

-- 1. Set inclusion property
theorem subset_union_inclusion (M N : Set α) : M ⊆ N → M ⊆ (M ∪ N) := by sorry

-- 2. Fraction inequality
theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (b + m) / (a + m) > b / a := by sorry

-- 3. Counterexample for ac² = bc² implying a = b
theorem ac_squared_eq_bc_squared_not_sufficient :
  ∃ (a b c : ℝ), a * c^2 = b * c^2 ∧ a ≠ b := by sorry

-- 4. Negation of universal quantifier
theorem negation_of_all_positive_real :
  ¬(∀ (x : ℝ), x > 0) ≠ (∃ (x : ℝ), x < 0) := by sorry

end NUMINAMATH_CALUDE_subset_union_inclusion_fraction_inequality_ac_squared_eq_bc_squared_not_sufficient_negation_of_all_positive_real_l3378_337817


namespace NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l3378_337882

/-- Represents a monomial in two variables -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  x_exp : ℕ
  y_exp : ℕ

/-- The degree of a monomial is the sum of its exponents -/
def Monomial.degree {α : Type*} [Ring α] (m : Monomial α) : ℕ :=
  m.x_exp + m.y_exp

theorem monomial_coefficient_and_degree :
  let m : Monomial ℤ := { coeff := -2, x_exp := 1, y_exp := 3 }
  (m.coeff = -2) ∧ (m.degree = 4) := by sorry

end NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l3378_337882


namespace NUMINAMATH_CALUDE_certain_number_problem_l3378_337865

theorem certain_number_problem (x y : ℝ) (hx : x = 4) (hy : y + y * x = 48) : y = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3378_337865


namespace NUMINAMATH_CALUDE_adams_average_score_l3378_337864

/-- Given Adam's total score and number of rounds played, calculate the average points per round --/
theorem adams_average_score (total_score : ℕ) (num_rounds : ℕ) 
  (h1 : total_score = 283) (h2 : num_rounds = 4) :
  ∃ (avg : ℚ), avg = (total_score : ℚ) / (num_rounds : ℚ) ∧ 
  ∃ (rounded : ℕ), rounded = round avg ∧ rounded = 71 := by
  sorry

end NUMINAMATH_CALUDE_adams_average_score_l3378_337864


namespace NUMINAMATH_CALUDE_emily_seeds_count_l3378_337826

/-- The number of seeds Emily planted in the big garden -/
def big_garden_seeds : ℕ := 29

/-- The number of small gardens Emily has -/
def num_small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden -/
def seeds_per_small_garden : ℕ := 4

/-- The total number of seeds Emily planted -/
def total_seeds : ℕ := big_garden_seeds + num_small_gardens * seeds_per_small_garden

theorem emily_seeds_count : total_seeds = 41 := by
  sorry

end NUMINAMATH_CALUDE_emily_seeds_count_l3378_337826


namespace NUMINAMATH_CALUDE_paragraphs_per_page_is_twenty_l3378_337884

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 50

/-- Represents the total time taken to read the book in hours -/
def total_reading_time : ℕ := 50

/-- Calculates the number of paragraphs per page in the book -/
def paragraphs_per_page : ℕ :=
  (reading_speed * total_reading_time) / (sentences_per_paragraph * total_pages)

/-- Theorem stating that the number of paragraphs per page is 20 -/
theorem paragraphs_per_page_is_twenty :
  paragraphs_per_page = 20 := by
  sorry

end NUMINAMATH_CALUDE_paragraphs_per_page_is_twenty_l3378_337884


namespace NUMINAMATH_CALUDE_two_circles_distance_formula_l3378_337832

/-- Two circles with radii R and r, whose centers are at distance d apart,
    and whose common internal tangents define four points of tangency
    that form a quadrilateral circumscribed around a circle. -/
structure TwoCirclesConfig where
  R : ℝ
  r : ℝ
  d : ℝ

/-- The theorem stating the relationship between the radii and the distance between centers -/
theorem two_circles_distance_formula (config : TwoCirclesConfig) :
  config.d ^ 2 = (config.R + config.r) ^ 2 + 4 * config.R * config.r :=
sorry

end NUMINAMATH_CALUDE_two_circles_distance_formula_l3378_337832


namespace NUMINAMATH_CALUDE_hyperbola_directrices_distance_l3378_337855

/-- Given a hyperbola with foci at (±√26, 0) and asymptotes y = ±(3/2)x,
    prove that the distance between its two directrices is (8√26)/13 -/
theorem hyperbola_directrices_distance (a b c : ℝ) : 
  (c = Real.sqrt 26) →                  -- focus distance
  (b / a = 3 / 2) →                     -- asymptote slope
  (a^2 + b^2 = 26) →                    -- relation between a, b, and c
  (2 * (a^2 / c)) = (8 * Real.sqrt 26) / 13 := by
  sorry

#check hyperbola_directrices_distance

end NUMINAMATH_CALUDE_hyperbola_directrices_distance_l3378_337855


namespace NUMINAMATH_CALUDE_two_valid_B_values_l3378_337839

/-- Represents a single digit (1 to 9) -/
def SingleDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Converts a single digit B to the two-digit number B1 -/
def toTwoDigit (B : SingleDigit) : ℕ := B.val * 10 + 1

/-- Checks if the equation x^2 - (1B)x + B1 = 0 has positive integer solutions -/
def hasPositiveIntegerSolutions (B : SingleDigit) : Prop :=
  ∃ x : ℕ, x > 0 ∧ x^2 - (10 + B.val) * x + toTwoDigit B = 0

/-- The main theorem stating that exactly two single-digit B values satisfy the condition -/
theorem two_valid_B_values :
  ∃! (S : Finset SingleDigit), S.card = 2 ∧ ∀ B, B ∈ S ↔ hasPositiveIntegerSolutions B :=
sorry

end NUMINAMATH_CALUDE_two_valid_B_values_l3378_337839


namespace NUMINAMATH_CALUDE_three_doors_two_colors_l3378_337846

/-- The number of ways to paint a given number of doors with a given number of colors -/
def paintingWays (doors : ℕ) (colors : ℕ) : ℕ := colors ^ doors

/-- Theorem: The number of ways to paint 3 doors with 2 colors is 8 -/
theorem three_doors_two_colors : paintingWays 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_doors_two_colors_l3378_337846


namespace NUMINAMATH_CALUDE_equation_solution_l3378_337877

theorem equation_solution (x : ℚ) : 
  (5 * x - 3) / (6 * x - 12) = 4 / 3 → x = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3378_337877


namespace NUMINAMATH_CALUDE_thirteen_percent_problem_l3378_337845

theorem thirteen_percent_problem : ∃ x : ℝ, 
  (13 / 100) * x = 85 ∧ 
  Int.floor (x + 0.5) = 654 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_percent_problem_l3378_337845


namespace NUMINAMATH_CALUDE_fourth_vertex_coordinates_l3378_337821

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

/-- The theorem stating the possible coordinates of the fourth vertex of the parallelogram -/
theorem fourth_vertex_coordinates :
  let A : Point := ⟨0, -9⟩
  let B : Point := ⟨2, 6⟩
  let C : Point := ⟨4, 5⟩
  ∃ D : Point, (D = ⟨2, -10⟩ ∨ D = ⟨-2, -8⟩ ∨ D = ⟨6, 20⟩) ∧ 
    is_parallelogram A B C D :=
by
  sorry


end NUMINAMATH_CALUDE_fourth_vertex_coordinates_l3378_337821


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l3378_337849

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l3378_337849


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3378_337830

/-- A function to check if three line segments can form a right triangle -/
def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that among the given sets, only {2, √2, √2} forms a right triangle -/
theorem right_triangle_sets :
  ¬ isRightTriangle 2 3 4 ∧
  ¬ isRightTriangle 1 1 2 ∧
  ¬ isRightTriangle 4 5 6 ∧
  isRightTriangle 2 (Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3378_337830


namespace NUMINAMATH_CALUDE_line_properties_l3378_337887

/-- A line in the 2D plane represented by the equation y = k(x-1) --/
structure Line where
  k : ℝ

/-- The point (1,0) in the 2D plane --/
def point : ℝ × ℝ := (1, 0)

/-- Checks if a given line passes through the point (1,0) --/
def passes_through_point (l : Line) : Prop :=
  0 = l.k * (point.1 - 1)

/-- Checks if a given line is not perpendicular to the x-axis --/
def not_perpendicular_to_x_axis (l : Line) : Prop :=
  l.k ≠ 0

/-- Theorem stating that all lines represented by y = k(x-1) pass through (1,0) and are not perpendicular to the x-axis --/
theorem line_properties (l : Line) : 
  passes_through_point l ∧ not_perpendicular_to_x_axis l :=
sorry

end NUMINAMATH_CALUDE_line_properties_l3378_337887


namespace NUMINAMATH_CALUDE_trillion_scientific_notation_l3378_337804

theorem trillion_scientific_notation :
  (10000 : ℝ) * 10000 * 10000 = 1 * (10 : ℝ)^12 := by sorry

end NUMINAMATH_CALUDE_trillion_scientific_notation_l3378_337804


namespace NUMINAMATH_CALUDE_common_element_exists_l3378_337860

-- Define a type for the index of sets (1 to 2011)
def SetIndex := Fin 2011

-- Define the property of being a set of consecutive integers
def IsConsecutiveSet (S : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ S = Finset.Ico a (b + 1)

-- Define the main theorem
theorem common_element_exists
  (S : SetIndex → Set ℤ)
  (h_nonempty : ∀ i, (S i).Nonempty)
  (h_consecutive : ∀ i, IsConsecutiveSet (S i))
  (h_common : ∀ i j, i ≠ j → (S i ∩ S j).Nonempty) :
  ∃ n : ℤ, n > 0 ∧ ∀ i, n ∈ S i :=
sorry

end NUMINAMATH_CALUDE_common_element_exists_l3378_337860


namespace NUMINAMATH_CALUDE_magical_stack_with_201_fixed_l3378_337853

/-- Definition of a magical stack of cards -/
def is_magical_stack (n : ℕ) : Prop :=
  ∃ (card_from_A card_from_B : ℕ), 
    card_from_A ≤ n ∧ 
    card_from_B > n ∧ 
    card_from_B ≤ 2*n ∧
    (card_from_A = 2 * ((card_from_A + 1) / 2) - 1 ∨
     card_from_B = 2 * (card_from_B / 2))

/-- Theorem stating the number of cards in a magical stack where card 201 retains its position -/
theorem magical_stack_with_201_fixed :
  ∃ (n : ℕ), 
    is_magical_stack n ∧ 
    201 ≤ n ∧
    201 = 2 * ((201 + 1) / 2) - 1 ∧
    n = 201 ∧
    2 * n = 402 := by
  sorry

end NUMINAMATH_CALUDE_magical_stack_with_201_fixed_l3378_337853


namespace NUMINAMATH_CALUDE_area_under_curve_l3378_337890

-- Define the curve
def curve (x : ℝ) : ℝ := 3 * x^2

-- Define the bounds of the region
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- Theorem statement
theorem area_under_curve :
  ∫ x in lower_bound..upper_bound, curve x = 1 := by sorry

end NUMINAMATH_CALUDE_area_under_curve_l3378_337890


namespace NUMINAMATH_CALUDE_exam_attendance_l3378_337836

theorem exam_attendance (passed_percentage : ℝ) (failed_count : ℕ) : 
  passed_percentage = 35 →
  failed_count = 481 →
  (100 - passed_percentage) / 100 * 740 = failed_count :=
by
  sorry

end NUMINAMATH_CALUDE_exam_attendance_l3378_337836


namespace NUMINAMATH_CALUDE_min_value_and_integer_solutions_l3378_337816

theorem min_value_and_integer_solutions (x y : ℝ) : 
  x + y + 2*x*y = 5 →
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → x + y ≥ Real.sqrt 11 - 1) ∧
  (∃ (x y : ℤ), x + y + 2*x*y = 5 ∧ (x + y = 5 ∨ x + y = -7)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_integer_solutions_l3378_337816


namespace NUMINAMATH_CALUDE_legs_in_room_is_40_l3378_337803

/-- Calculates the total number of legs in a room with various furniture items. -/
def total_legs_in_room : ℕ :=
  let four_legged_items := 4 + 1 + 2  -- 4 tables, 1 sofa, 2 chairs
  let three_legged_tables := 3
  let one_legged_table := 1
  let two_legged_rocking_chair := 1
  
  4 * four_legged_items + 
  3 * three_legged_tables + 
  1 * one_legged_table + 
  2 * two_legged_rocking_chair

/-- Theorem stating that the total number of legs in the room is 40. -/
theorem legs_in_room_is_40 : total_legs_in_room = 40 := by
  sorry

end NUMINAMATH_CALUDE_legs_in_room_is_40_l3378_337803


namespace NUMINAMATH_CALUDE_subset_M_proof_l3378_337895

def M : Set ℝ := {x | x ≤ 2 * Real.sqrt 3}

theorem subset_M_proof (b : ℝ) (hb : b ∈ Set.Ioo 0 1) :
  {Real.sqrt (11 + b)} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_M_proof_l3378_337895


namespace NUMINAMATH_CALUDE_sugar_solution_concentration_l3378_337828

theorem sugar_solution_concentration (W : ℝ) (X : ℝ) : 
  W > 0 → -- W is positive (total weight of solution)
  0.08 * W = 0.08 * W - 0.02 * W + X * W / 400 → -- Sugar balance equation
  0.16 * W = 0.06 * W + X * W / 400 → -- Final concentration equation
  X = 40 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_concentration_l3378_337828


namespace NUMINAMATH_CALUDE_set_operations_and_subset_condition_l3378_337808

def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

theorem set_operations_and_subset_condition :
  (∀ x, x ∈ (A ∪ B 1) ↔ -4 < x ∧ x ≤ 3) ∧
  (∀ x, x ∈ (A ∩ (Set.univ \ B 1)) ↔ -4 < x ∧ x < 0) ∧
  (∀ a, B a ⊆ A ↔ -3 < a ∧ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_condition_l3378_337808


namespace NUMINAMATH_CALUDE_function_is_identity_l3378_337837

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (a : ℝ), f (f x - y) = f x + f (f y - f a) + x

theorem function_is_identity (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_is_identity_l3378_337837


namespace NUMINAMATH_CALUDE_cone_surface_area_l3378_337820

/-- 
Given a cone with slant height 2 and lateral surface that unfolds into a semicircle,
prove that its surface area is 3π.
-/
theorem cone_surface_area (h : ℝ) (r : ℝ) : 
  h = 2 ∧ 2 * π * r = 2 * π → π * r * h + π * r^2 = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3378_337820


namespace NUMINAMATH_CALUDE_probability_both_odd_l3378_337893

def m : ℕ := 7
def n : ℕ := 9

def is_odd (k : ℕ) : Prop := k % 2 = 1

def count_odd (k : ℕ) : ℕ := (k + 1) / 2

theorem probability_both_odd : 
  (count_odd m * count_odd n : ℚ) / (m * n : ℚ) = 20 / 63 := by sorry

end NUMINAMATH_CALUDE_probability_both_odd_l3378_337893


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3378_337833

/-- Represents the store's lamp purchasing problem -/
structure LampProblem where
  cost_diff : ℕ  -- Cost difference between type A and B lamps
  budget_A : ℕ   -- Budget for type A lamps
  budget_B : ℕ   -- Budget for type B lamps
  total_lamps : ℕ -- Total number of lamps to purchase
  max_budget : ℕ  -- Maximum total budget
  price_A : ℕ    -- Selling price of type A lamp
  price_B : ℕ    -- Selling price of type B lamp

/-- Calculates the maximum profit for the given LampProblem -/
def max_profit (p : LampProblem) : ℕ :=
  let cost_A := p.budget_A * 2 / 5  -- Cost of type A lamp
  let cost_B := cost_A - p.cost_diff -- Cost of type B lamp
  let max_A := (p.max_budget - cost_B * p.total_lamps) / (cost_A - cost_B)
  let profit := (p.price_A - cost_A) * max_A + (p.price_B - cost_B) * (p.total_lamps - max_A)
  profit

/-- Theorem stating the maximum profit for the given problem -/
theorem max_profit_theorem (p : LampProblem) : 
  p.cost_diff = 40 ∧ 
  p.budget_A = 2000 ∧ 
  p.budget_B = 1600 ∧ 
  p.total_lamps = 80 ∧ 
  p.max_budget = 14550 ∧ 
  p.price_A = 300 ∧ 
  p.price_B = 200 →
  max_profit p = 5780 ∧ 
  (p.max_budget - 160 * p.total_lamps) / 40 = 43 :=
by sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l3378_337833


namespace NUMINAMATH_CALUDE_equation_solutions_l3378_337891

theorem equation_solutions :
  (∃ x : ℚ, (3 - x) / (x + 4) = 1 / 2 ∧ x = 2 / 3) ∧
  (∃ x : ℚ, x / (x - 1) - 2 * x / (3 * x - 3) = 1 ∧ x = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3378_337891


namespace NUMINAMATH_CALUDE_initial_marbles_calculation_l3378_337858

/-- The number of marbles Connie initially had -/
def initial_marbles : ℝ := 972.1

/-- The number of marbles Connie gave to Juan -/
def marbles_to_juan : ℝ := 183.5

/-- The number of marbles Connie gave to Maria -/
def marbles_to_maria : ℝ := 245.7

/-- The number of marbles Connie received from Mike -/
def marbles_from_mike : ℝ := 50.3

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.2

/-- Theorem stating that the initial number of marbles is equal to the sum of
    the current marbles, marbles given away, minus marbles received -/
theorem initial_marbles_calculation :
  initial_marbles = marbles_left + marbles_to_juan + marbles_to_maria - marbles_from_mike :=
by sorry

end NUMINAMATH_CALUDE_initial_marbles_calculation_l3378_337858


namespace NUMINAMATH_CALUDE_investment_rate_proof_l3378_337854

/-- Proves that the unknown investment rate is 0.18 given the problem conditions --/
theorem investment_rate_proof (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (unknown_rate_investment : ℝ) 
  (h1 : total_investment = 22000)
  (h2 : known_rate = 0.14)
  (h3 : total_interest = 3360)
  (h4 : unknown_rate_investment = 7000)
  (h5 : unknown_rate_investment * r + (total_investment - unknown_rate_investment) * known_rate = total_interest) :
  r = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l3378_337854


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l3378_337857

theorem roots_sum_and_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∀ x, x^2 - p*x - 2*q = 0 ↔ x = p ∨ x = q) :
  p + q = p ∧ p * q = -2*q := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l3378_337857


namespace NUMINAMATH_CALUDE_midnight_temperature_l3378_337806

def morning_temp : Int := 7
def noon_rise : Int := 2
def midnight_drop : Int := 10

theorem midnight_temperature : 
  morning_temp + noon_rise - midnight_drop = -1 := by sorry

end NUMINAMATH_CALUDE_midnight_temperature_l3378_337806


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3378_337800

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3 - x) ↔ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3378_337800


namespace NUMINAMATH_CALUDE_unique_positive_root_l3378_337879

/-- The polynomial function f(x) = x^12 + 5x^11 - 3x^10 + 2000x^9 - 1500x^8 -/
def f (x : ℝ) : ℝ := x^12 + 5*x^11 - 3*x^10 + 2000*x^9 - 1500*x^8

/-- The theorem stating that f(x) has exactly one positive real root -/
theorem unique_positive_root : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_root_l3378_337879


namespace NUMINAMATH_CALUDE_line_through_point_and_circle_center_l3378_337899

/-- A line passing through two points on a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that the equation of a line passing through a given point and the center of a given circle is x-2=0. -/
theorem line_through_point_and_circle_center 
  (M : ℝ × ℝ) 
  (C : Circle) 
  (h1 : M.1 = 2 ∧ M.2 = 3) 
  (h2 : C.center = (2, -3)) 
  (h3 : C.radius = 3) : 
  ∃ (l : Line), l.a = 1 ∧ l.b = 0 ∧ l.c = -2 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_and_circle_center_l3378_337899


namespace NUMINAMATH_CALUDE_inequality_proof_l3378_337880

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3378_337880


namespace NUMINAMATH_CALUDE_ratio_problem_l3378_337885

theorem ratio_problem (x y : ℝ) (h : (2*x - 3*y) / (x + 2*y) = 5/4) :
  x / y = 22/3 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3378_337885


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3378_337811

-- Define the function f(x) = x² - mx + 1
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- State the theorem
theorem monotone_increasing_condition (m : ℝ) :
  MonotonicallyIncreasing (f m) 3 8 → m ≤ 6 ∨ m ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3378_337811


namespace NUMINAMATH_CALUDE_defective_from_factory1_l3378_337827

/-- The probability of a product coming from the first factory -/
def p_factory1 : ℝ := 0.20

/-- The probability of a product coming from the second factory -/
def p_factory2 : ℝ := 0.46

/-- The probability of a product coming from the third factory -/
def p_factory3 : ℝ := 0.34

/-- The probability of a defective item from the first factory -/
def p_defective1 : ℝ := 0.03

/-- The probability of a defective item from the second factory -/
def p_defective2 : ℝ := 0.02

/-- The probability of a defective item from the third factory -/
def p_defective3 : ℝ := 0.01

/-- The probability that a randomly selected defective item was produced at the first factory -/
theorem defective_from_factory1 : 
  (p_defective1 * p_factory1) / (p_defective1 * p_factory1 + p_defective2 * p_factory2 + p_defective3 * p_factory3) = 0.322 := by
sorry

end NUMINAMATH_CALUDE_defective_from_factory1_l3378_337827


namespace NUMINAMATH_CALUDE_furniture_reimbursement_l3378_337842

/-- Calculates the reimbursement amount for an overcharged furniture purchase -/
theorem furniture_reimbursement
  (total_paid : ℕ)
  (num_pieces : ℕ)
  (cost_per_piece : ℕ)
  (h1 : total_paid = 20700)
  (h2 : num_pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (num_pieces * cost_per_piece) = 600 := by
sorry

end NUMINAMATH_CALUDE_furniture_reimbursement_l3378_337842


namespace NUMINAMATH_CALUDE_sports_love_distribution_l3378_337831

theorem sports_love_distribution (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (boys_not_love_sports : ℕ) (total_not_love_sports : ℕ) :
  total_students = 50 →
  total_boys = 30 →
  total_girls = 20 →
  boys_not_love_sports = 12 →
  total_not_love_sports = 24 →
  ∃ (boys_love_sports : ℕ) (total_love_sports : ℕ),
    boys_love_sports = total_boys - boys_not_love_sports ∧
    total_love_sports = total_students - total_not_love_sports ∧
    boys_love_sports = 18 ∧
    total_love_sports = 26 := by
  sorry

end NUMINAMATH_CALUDE_sports_love_distribution_l3378_337831


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3378_337878

theorem inscribed_circle_area_ratio (hexagon_side : Real) (hexagon_side_positive : hexagon_side > 0) :
  let hexagon_area := 3 * Real.sqrt 3 * hexagon_side^2 / 2
  let inscribed_circle_radius := hexagon_side * Real.sqrt 3 / 2
  let inscribed_circle_area := Real.pi * inscribed_circle_radius^2
  inscribed_circle_area / hexagon_area > 0.9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3378_337878


namespace NUMINAMATH_CALUDE_retirement_fund_increment_l3378_337812

theorem retirement_fund_increment (y k : ℝ) 
  (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27)
  : k * Real.sqrt y = 810 := by
  sorry

end NUMINAMATH_CALUDE_retirement_fund_increment_l3378_337812


namespace NUMINAMATH_CALUDE_calc_complex_fraction_l3378_337813

theorem calc_complex_fraction : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end NUMINAMATH_CALUDE_calc_complex_fraction_l3378_337813


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3378_337889

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 100 ∧ x₀ * y₀ = 40 ∧ x₀ + y₀ = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3378_337889


namespace NUMINAMATH_CALUDE_lunks_needed_for_bananas_l3378_337810

/-- Exchange rate of lunks to kunks -/
def lunk_to_kunk_rate : ℚ := 2 / 3

/-- Exchange rate of kunks to bananas -/
def kunk_to_banana_rate : ℚ := 5 / 6

/-- Number of bananas to purchase -/
def bananas_to_buy : ℕ := 20

/-- Theorem stating the number of lunks needed to buy the specified number of bananas -/
theorem lunks_needed_for_bananas : 
  ⌈(bananas_to_buy : ℚ) / (kunk_to_banana_rate * lunk_to_kunk_rate)⌉ = 36 := by
  sorry


end NUMINAMATH_CALUDE_lunks_needed_for_bananas_l3378_337810


namespace NUMINAMATH_CALUDE_modern_pentathlon_theorem_l3378_337841

/-- Represents a competitor in the Modern Pentathlon --/
inductive Competitor
| A
| B
| C

/-- Represents an event in the Modern Pentathlon --/
inductive Event
| Shooting
| Fencing
| Swimming
| Equestrian
| CrossCountryRunning

/-- Represents the place a competitor finished in an event --/
inductive Place
| First
| Second
| Third

/-- The scoring system for the Modern Pentathlon --/
structure ScoringSystem where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  a_gt_b : a > b
  b_gt_c : b > c

/-- The results of the Modern Pentathlon --/
def ModernPentathlonResults (s : ScoringSystem) :=
  Competitor → Event → Place

/-- Calculate the total score for a competitor given the results --/
def totalScore (s : ScoringSystem) (results : ModernPentathlonResults s) (competitor : Competitor) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem modern_pentathlon_theorem (s : ScoringSystem) 
  (results : ModernPentathlonResults s)
  (total_A : totalScore s results Competitor.A = 22)
  (total_B : totalScore s results Competitor.B = 9)
  (total_C : totalScore s results Competitor.C = 9)
  (B_first_equestrian : results Competitor.B Event.Equestrian = Place.First) :
  (∃ r : ModernPentathlonResults s, 
    (totalScore s r Competitor.A = 22) ∧ 
    (totalScore s r Competitor.B = 9) ∧ 
    (totalScore s r Competitor.C = 9) ∧
    (r Competitor.B Event.Equestrian = Place.First) ∧
    (r Competitor.B Event.Swimming = Place.Third)) ∧
  (∃ r : ModernPentathlonResults s, 
    (totalScore s r Competitor.A = 22) ∧ 
    (totalScore s r Competitor.B = 9) ∧ 
    (totalScore s r Competitor.C = 9) ∧
    (r Competitor.B Event.Equestrian = Place.First) ∧
    (r Competitor.C Event.Swimming = Place.Third)) :=
  sorry

end NUMINAMATH_CALUDE_modern_pentathlon_theorem_l3378_337841


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_5_range_of_m_for_inequality_l3378_337856

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 6|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Part II
theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, f m x ≥ 7} = {m : ℝ | m ≤ -13 ∨ m ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_5_range_of_m_for_inequality_l3378_337856


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l3378_337863

def M (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_sum_theorem (a b c d : ℝ) :
  ¬(IsUnit (M a b c d).det) →
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c) = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l3378_337863


namespace NUMINAMATH_CALUDE_range_a_characterization_l3378_337888

/-- The range of values for a where "p or q" is true and "p and q" is false -/
def range_a : Set ℝ := Set.union (Set.Ioc 0 0.5) (Set.Ico 1 2)

/-- p is true when 0 < a < 1 -/
def p_true (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- q is true when 0.5 < a < 2 -/
def q_true (a : ℝ) : Prop := 0.5 < a ∧ a < 2

theorem range_a_characterization (a : ℝ) (h : a > 0) :
  a ∈ range_a ↔ (p_true a ∨ q_true a) ∧ ¬(p_true a ∧ q_true a) :=
by sorry

end NUMINAMATH_CALUDE_range_a_characterization_l3378_337888


namespace NUMINAMATH_CALUDE_angle_A_is_90_l3378_337825

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = 180

-- Define the specific conditions of our triangle
def our_triangle (t : Triangle) : Prop :=
  is_valid_triangle t ∧ t.C = 2 * t.B ∧ t.B = 30

-- Theorem statement
theorem angle_A_is_90 (t : Triangle) (h : our_triangle t) : t.A = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_90_l3378_337825


namespace NUMINAMATH_CALUDE_angle_double_supplement_is_120_l3378_337809

-- Define the angle measure
def angle_measure : ℝ → Prop := λ x => 
  -- The angle measure is double its supplement
  x = 2 * (180 - x) ∧ 
  -- The angle measure is positive and less than or equal to 180
  0 < x ∧ x ≤ 180

-- Theorem statement
theorem angle_double_supplement_is_120 : 
  ∃ x : ℝ, angle_measure x ∧ x = 120 :=
sorry

end NUMINAMATH_CALUDE_angle_double_supplement_is_120_l3378_337809


namespace NUMINAMATH_CALUDE_identify_fake_bag_l3378_337862

/-- Represents a bag of coins -/
structure CoinBag where
  id : Nat
  isFake : Bool

/-- Represents the collection of all coin bags -/
def allBags : Finset CoinBag := sorry

/-- The weight of a real coin in grams -/
def realCoinWeight : Nat := 10

/-- The weight of a fake coin in grams -/
def fakeCoinWeight : Nat := 9

/-- The number of bags -/
def numBags : Nat := 10

/-- The expected total weight if all coins were real -/
def expectedTotalWeight : Nat := 550

/-- Calculates the weight of coins taken from a bag -/
def bagWeight (bag : CoinBag) : Nat :=
  if bag.isFake then
    bag.id * fakeCoinWeight
  else
    bag.id * realCoinWeight

/-- Calculates the total weight of all selected coins -/
def totalWeight : Nat := (allBags.sum bagWeight)

/-- Theorem stating that the bag number with fake coins is equal to the difference
    between the expected total weight and the actual total weight -/
theorem identify_fake_bag :
  ∃ (fakeBag : CoinBag), fakeBag ∈ allBags ∧ fakeBag.isFake ∧
    fakeBag.id = expectedTotalWeight - totalWeight := by sorry

end NUMINAMATH_CALUDE_identify_fake_bag_l3378_337862


namespace NUMINAMATH_CALUDE_saree_original_price_l3378_337881

/-- The original price of sarees given successive discounts -/
theorem saree_original_price (final_price : ℝ) 
  (h1 : final_price = 380.16) 
  (h2 : final_price = 0.9 * 0.8 * original_price) : 
  original_price = 528 :=
by
  sorry

#check saree_original_price

end NUMINAMATH_CALUDE_saree_original_price_l3378_337881


namespace NUMINAMATH_CALUDE_complex_sum_real_part_l3378_337851

theorem complex_sum_real_part (a b : ℝ) : 
  (1 + Complex.I) / Complex.I + (1 + Complex.I * Real.sqrt 3) ^ 2 = Complex.mk a b →
  a + b = 2 * Real.sqrt 3 - 2 :=
sorry

end NUMINAMATH_CALUDE_complex_sum_real_part_l3378_337851


namespace NUMINAMATH_CALUDE_rectangle_area_l3378_337892

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3378_337892


namespace NUMINAMATH_CALUDE_tangent_line_distance_l3378_337847

/-- A line with slope 1 is tangent to y = e^x and y^2 = 4x at two different points. -/
theorem tangent_line_distance : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  -- The line is tangent to y = e^x at (x₁, y₁)
  (Real.exp x₁ = y₁) ∧ 
  (Real.exp x₁ = 1) ∧
  -- The line is tangent to y^2 = 4x at (x₂, y₂)
  (y₂^2 = 4 * x₂) ∧
  (y₂ = 2 * Real.sqrt x₂) ∧
  -- Both points lie on a line with slope 1
  (y₂ - y₁ = x₂ - x₁) ∧
  -- The distance between the two points is √2
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_distance_l3378_337847


namespace NUMINAMATH_CALUDE_abs_of_negative_three_l3378_337869

theorem abs_of_negative_three :
  ∀ x : ℝ, x = -3 → |x| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_of_negative_three_l3378_337869


namespace NUMINAMATH_CALUDE_largest_710_triple_l3378_337896

/-- Converts a natural number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 7) :: aux (m / 7)
  aux n |>.reverse

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 7-10 triple -/
def is710Triple (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 3 * n

/-- States that 1422 is the largest 7-10 triple -/
theorem largest_710_triple :
  is710Triple 1422 ∧ ∀ m : ℕ, m > 1422 → ¬is710Triple m :=
sorry

end NUMINAMATH_CALUDE_largest_710_triple_l3378_337896


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3378_337843

/-- The number of distinct ways to arrange n distinct beads on a bracelet, 
    considering rotations and reflections as equivalent -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements 
    for 8 beads is 2520 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3378_337843


namespace NUMINAMATH_CALUDE_total_money_available_l3378_337872

/-- Represents the cost of a single gumdrop in cents -/
def cost_per_gumdrop : ℕ := 4

/-- Represents the number of gumdrops that can be purchased -/
def num_gumdrops : ℕ := 20

/-- Theorem stating that the total amount of money available is 80 cents -/
theorem total_money_available : cost_per_gumdrop * num_gumdrops = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_money_available_l3378_337872


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l3378_337807

/-- Calculates the total cost of decorations for Nathan's wedding reception --/
def total_decoration_cost (num_tables : ℕ) 
  (tablecloth_cost service_charge place_setting_cost place_settings_per_table : ℝ)
  (roses_per_centerpiece rose_cost rose_discount : ℝ)
  (lilies_per_centerpiece lily_cost lily_discount : ℝ)
  (daisies_per_centerpiece daisy_cost : ℝ)
  (sunflowers_per_centerpiece sunflower_cost : ℝ)
  (lighting_cost : ℝ) : ℝ :=
  let tablecloth_total := num_tables * tablecloth_cost * (1 + service_charge)
  let place_settings_total := num_tables * place_settings_per_table * place_setting_cost
  let centerpiece_cost := 
    (roses_per_centerpiece * rose_cost * (1 - rose_discount)) +
    (lilies_per_centerpiece * lily_cost * (1 - lily_discount)) +
    (daisies_per_centerpiece * daisy_cost) +
    (sunflowers_per_centerpiece * sunflower_cost)
  let centerpiece_total := num_tables * centerpiece_cost
  tablecloth_total + place_settings_total + centerpiece_total + lighting_cost

/-- Theorem stating the total cost of decorations for Nathan's wedding reception --/
theorem wedding_decoration_cost : 
  total_decoration_cost 30 25 0.15 12 6 15 6 0.1 20 5 0.05 5 3 3 4 450 = 9562.50 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_l3378_337807


namespace NUMINAMATH_CALUDE_vase_original_price_l3378_337819

/-- Proves that given a vase with an original price P, which is discounted by 25% 
    and then has a 10% sales tax applied, if the total price paid is $165, 
    then the original price P must be $200. -/
theorem vase_original_price (P : ℝ) : 
  (P * (1 - 0.25) * (1 + 0.1) = 165) → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_vase_original_price_l3378_337819


namespace NUMINAMATH_CALUDE_exists_x_in_interval_l3378_337859

theorem exists_x_in_interval (x : ℝ) : 
  ∃ x, x ∈ Set.Icc (-1 : ℝ) (3/10) ∧ x^2 + 3*x - 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_in_interval_l3378_337859


namespace NUMINAMATH_CALUDE_sandwich_cost_l3378_337848

theorem sandwich_cost (N B J : ℕ) (h1 : N > 1) (h2 : B > 0) (h3 : J > 0)
  (h4 : N * (3 * B + 6 * J) = 306) : 6 * N * J = 288 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l3378_337848


namespace NUMINAMATH_CALUDE_system_solution_l3378_337868

theorem system_solution :
  ∃ (x y z : ℝ),
    (x + y + z = 26) ∧
    (3 * x - 2 * y + z = 3) ∧
    (x - 4 * y - 2 * z = -13) ∧
    (x = -32.2) ∧
    (y = -13.8) ∧
    (z = 72) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3378_337868


namespace NUMINAMATH_CALUDE_mean_height_of_players_l3378_337835

def player_heights : List ℝ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

theorem mean_height_of_players : 
  (player_heights.sum / player_heights.length : ℝ) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_of_players_l3378_337835


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l3378_337876

/-- A swimming pool with trapezoidal cross-section -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  shallow_depth : ℝ
  deep_depth : ℝ

/-- Calculate the volume of a swimming pool with trapezoidal cross-section -/
def pool_volume (pool : SwimmingPool) : ℝ :=
  0.5 * (pool.shallow_depth + pool.deep_depth) * pool.width * pool.length

/-- Theorem stating that the volume of the given swimming pool is 270 cubic meters -/
theorem swimming_pool_volume :
  let pool : SwimmingPool := {
    width := 9,
    length := 12,
    shallow_depth := 1,
    deep_depth := 4
  }
  pool_volume pool = 270 := by sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l3378_337876


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3378_337898

theorem square_sum_theorem (x y : ℝ) (h1 : x + 2*y = 4) (h2 : x*y = -8) : 
  x^2 + 4*y^2 = 48 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3378_337898


namespace NUMINAMATH_CALUDE_company_kw_price_l3378_337814

theorem company_kw_price (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1.2 * a = 0.75 * (a + b)) → (1.2 * a = 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_company_kw_price_l3378_337814


namespace NUMINAMATH_CALUDE_least_number_divisible_by_3_4_5_7_8_l3378_337822

theorem least_number_divisible_by_3_4_5_7_8 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) → n ≥ 840 := by
  sorry

#check least_number_divisible_by_3_4_5_7_8

end NUMINAMATH_CALUDE_least_number_divisible_by_3_4_5_7_8_l3378_337822


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l3378_337894

theorem cycle_gain_percent (cost_price selling_price : ℚ) (h1 : cost_price = 900) (h2 : selling_price = 1100) :
  (selling_price - cost_price) / cost_price * 100 = (2 : ℚ) / 9 * 100 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l3378_337894


namespace NUMINAMATH_CALUDE_factorization_implies_k_value_l3378_337824

theorem factorization_implies_k_value (k : ℝ) :
  (∃ (a b c d e f : ℝ), ∀ x y : ℝ,
    x^3 + 3*x^2 - 2*x*y - k*x - 4*y = (a*x + b*y + c) * (d*x^2 + e*x*y + f*y)) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_factorization_implies_k_value_l3378_337824


namespace NUMINAMATH_CALUDE_odot_computation_l3378_337897

def odot (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem odot_computation : odot 2 (odot 3 (odot 4 5)) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_odot_computation_l3378_337897


namespace NUMINAMATH_CALUDE_equation_solution_l3378_337873

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 5 → (x + 60 / (x - 5) = -12 ↔ x = 0 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3378_337873


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l3378_337874

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l3378_337874


namespace NUMINAMATH_CALUDE_fraction_of_trunks_l3378_337883

/-- Given that 38% of garments are bikinis and 63% are either bikinis or trunks,
    prove that 25% of garments are trunks. -/
theorem fraction_of_trunks
  (bikinis : Real)
  (bikinis_or_trunks : Real)
  (h1 : bikinis = 0.38)
  (h2 : bikinis_or_trunks = 0.63) :
  bikinis_or_trunks - bikinis = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_trunks_l3378_337883


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3378_337886

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  1 / x + 4 / y ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3378_337886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3378_337801

/-- 
Given an arithmetic sequence with first term a₁ = k^2 - k + 1 and common difference d = 1,
the sum of the first k + 2 terms is equal to k^3 + 2k^2 + k + 2.
-/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ := k^2 - k + 1
  let d := 1
  let n := k + 2
  let Sn := n * (a₁ + (a₁ + (n - 1) * d)) / 2
  Sn = k^3 + 2*k^2 + k + 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3378_337801


namespace NUMINAMATH_CALUDE_ratio_composition_l3378_337867

theorem ratio_composition (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_composition_l3378_337867


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3378_337840

/-- The value of c for which the line y = 3x + c is tangent to the parabola y² = 12x -/
def tangent_line_c : ℝ := 1

/-- The line equation: y = 3x + c -/
def line_equation (x y c : ℝ) : Prop := y = 3 * x + c

/-- The parabola equation: y² = 12x -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

/-- The line y = 3x + c is tangent to the parabola y² = 12x when c = tangent_line_c -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), line_equation x y tangent_line_c ∧ parabola_equation x y :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3378_337840


namespace NUMINAMATH_CALUDE_ababab_divisible_by_seven_l3378_337844

/-- Given two digits a and b, the function forms the number ababab -/
def formNumber (a b : ℕ) : ℕ := 
  100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

/-- Theorem stating that for any two digits, the number formed as ababab is divisible by 7 -/
theorem ababab_divisible_by_seven (a b : ℕ) (ha : a < 10) (hb : b < 10) : 
  7 ∣ formNumber a b := by
  sorry

#eval formNumber 2 3  -- To check if the function works correctly

end NUMINAMATH_CALUDE_ababab_divisible_by_seven_l3378_337844


namespace NUMINAMATH_CALUDE_v_sum_zero_l3378_337852

noncomputable def v (x : ℝ) : ℝ := -x + (3/2) * Real.sin (x * Real.pi / 2)

theorem v_sum_zero : v (-3.14) + v (-1) + v 1 + v 3.14 = 0 := by sorry

end NUMINAMATH_CALUDE_v_sum_zero_l3378_337852


namespace NUMINAMATH_CALUDE_monomial_condition_and_expression_evaluation_l3378_337818

/-- Given that -2a^2 * b^(y+3) and 4a^x * b^2 form a monomial when added together,
    prove that x = 2 and y = -1, and that under these conditions,
    2(x^2*y - 3*y^3 + 2*x) - 3(x + x^2*y - 2*y^3) - x = 4 -/
theorem monomial_condition_and_expression_evaluation 
  (a b : ℝ) (x y : ℤ) 
  (h : ∃ k, -2 * a^2 * b^(y+3) + 4 * a^x * b^2 = k * a^2 * b^2) :
  x = 2 ∧ y = -1 ∧ 
  2 * (x^2 * y - 3 * y^3 + 2 * x) - 3 * (x + x^2 * y - 2 * y^3) - x = 4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_condition_and_expression_evaluation_l3378_337818


namespace NUMINAMATH_CALUDE_ellipse_slope_bound_l3378_337815

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    and points A(-a, 0), B(a, 0), and P(x, y) on the ellipse such that
    P ≠ A, P ≠ B, and |AP| = |OA|, prove that the absolute value of
    the slope of line OP is greater than √3. -/
theorem ellipse_slope_bound (a b x y : ℝ) :
  a > b ∧ b > 0 ∧
  x^2 / a^2 + y^2 / b^2 = 1 ∧
  (x ≠ -a ∨ y ≠ 0) ∧ (x ≠ a ∨ y ≠ 0) ∧
  (x + a)^2 + y^2 = 4 * a^2 →
  abs (y / x) > Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_bound_l3378_337815


namespace NUMINAMATH_CALUDE_boys_girls_difference_l3378_337838

/-- The number of girls on the playground -/
def num_girls : ℝ := 28.0

/-- The number of boys on the playground -/
def num_boys : ℝ := 35.0

/-- The difference between the number of boys and girls -/
def difference : ℝ := num_boys - num_girls

theorem boys_girls_difference : difference = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_boys_girls_difference_l3378_337838


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l3378_337875

/-- Represents a circle with points and chords -/
structure CircleWithChords where
  numPoints : ℕ
  noTripleIntersection : Bool

/-- Calculates the number of triangles formed inside the circle -/
def trianglesInsideCircle (c : CircleWithChords) : ℕ :=
  sorry

/-- The main theorem stating that for 10 points on a circle with the given conditions,
    the number of triangles formed inside is 105 -/
theorem ten_point_circle_triangles :
  ∀ (c : CircleWithChords),
    c.numPoints = 10 →
    c.noTripleIntersection = true →
    trianglesInsideCircle c = 105 :=
by sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l3378_337875


namespace NUMINAMATH_CALUDE_card_arrangement_probability_l3378_337829

theorem card_arrangement_probability : 
  let n : ℕ := 8  -- total number of cards
  let k : ℕ := 3  -- number of identical cards (О in this case)
  let total_permutations : ℕ := n.factorial
  let favorable_permutations : ℕ := k.factorial
  (favorable_permutations : ℚ) / total_permutations = 1 / 6720 :=
by sorry

end NUMINAMATH_CALUDE_card_arrangement_probability_l3378_337829


namespace NUMINAMATH_CALUDE_mike_bird_feeding_l3378_337802

/-- The number of seeds Mike throws to the birds on the left -/
def seeds_left : ℕ := 20

/-- The total number of seeds Mike starts with -/
def total_seeds : ℕ := 120

/-- The number of additional seeds thrown -/
def additional_seeds : ℕ := 30

/-- The number of seeds left at the end -/
def remaining_seeds : ℕ := 30

theorem mike_bird_feeding :
  seeds_left + 2 * seeds_left + additional_seeds + remaining_seeds = total_seeds :=
by sorry

end NUMINAMATH_CALUDE_mike_bird_feeding_l3378_337802


namespace NUMINAMATH_CALUDE_initial_commission_rate_l3378_337870

theorem initial_commission_rate 
  (unchanged_income : ℝ → ℝ → ℝ → ℝ → Prop)
  (new_rate : ℝ)
  (slump_percentage : ℝ) :
  let initial_rate := 4
  let slump_factor := 1 - slump_percentage / 100
  unchanged_income initial_rate new_rate slump_factor 1 →
  new_rate = 5 →
  slump_percentage = 20.000000000000007 →
  initial_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_commission_rate_l3378_337870


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_diff_main_theorem_l3378_337871

/-- Represents a recurring decimal of the form 0.nnn... where n is a single digit -/
def recurring_decimal (n : ℕ) : ℚ := n / 9

theorem recurring_decimal_sum_diff (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  recurring_decimal a + recurring_decimal b - recurring_decimal c = (a + b - c : ℚ) / 9 := by sorry

theorem main_theorem : 
  recurring_decimal 6 + recurring_decimal 2 - recurring_decimal 4 = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_diff_main_theorem_l3378_337871


namespace NUMINAMATH_CALUDE_flight_chess_starting_position_l3378_337861

theorem flight_chess_starting_position (x : ℤ) :
  x - 5 + 4 + 2 - 3 + 1 = 6 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_flight_chess_starting_position_l3378_337861


namespace NUMINAMATH_CALUDE_butter_for_original_recipe_l3378_337805

/-- Given a recipe where 12 ounces of butter is used for 28 cups of flour
    in a 4x version of the original recipe, prove that the amount of butter
    needed for the original recipe is 3 ounces. -/
theorem butter_for_original_recipe
  (butter_4x : ℝ) -- Amount of butter for 4x recipe
  (flour_4x : ℝ) -- Amount of flour for 4x recipe
  (scale_factor : ℕ) -- Factor by which the original recipe is scaled
  (h1 : butter_4x = 12) -- 12 ounces of butter used in 4x recipe
  (h2 : flour_4x = 28) -- 28 cups of flour used in 4x recipe
  (h3 : scale_factor = 4) -- The recipe is scaled by a factor of 4
  : butter_4x / scale_factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_butter_for_original_recipe_l3378_337805
