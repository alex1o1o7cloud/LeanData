import Mathlib

namespace NUMINAMATH_CALUDE_equal_probabilities_l2002_200275

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red := 100, green := 0 },
    green_box := { red := 0, green := 100 } }

/-- State after the first transfer -/
def first_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red - 8, green := state.red_box.green },
    green_box := { red := state.green_box.red + 8, green := state.green_box.green } }

/-- State after the second transfer -/
def second_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red, green := state.red_box.green + 1 },
    green_box := { red := state.green_box.red - 1, green := state.green_box.green - 7 } }

/-- Calculate the probability of drawing a specific color from a box -/
def draw_probability (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => (box.red : ℚ) / (box.red + box.green : ℚ)
  | "green" => (box.green : ℚ) / (box.red + box.green : ℚ)
  | _ => 0

/-- The main theorem to prove -/
theorem equal_probabilities :
  let final_state := second_transfer (first_transfer initial_state)
  (draw_probability final_state.red_box "green") = (draw_probability final_state.green_box "red") := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_l2002_200275


namespace NUMINAMATH_CALUDE_total_hamburger_configurations_l2002_200277

/-- The number of different condiments available. -/
def num_condiments : ℕ := 10

/-- The number of options for meat patties. -/
def meat_patty_options : ℕ := 4

/-- Theorem: The total number of different hamburger configurations. -/
theorem total_hamburger_configurations :
  (2 ^ num_condiments) * meat_patty_options = 4096 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_configurations_l2002_200277


namespace NUMINAMATH_CALUDE_distance_calculation_l2002_200227

/-- Represents the distance to the destination in kilometers -/
def distance : ℝ := 96

/-- Represents the rowing speed in still water in km/h -/
def rowing_speed : ℝ := 10

/-- Represents the current velocity in km/h -/
def current_velocity : ℝ := 2

/-- Represents the total round trip time in hours -/
def total_time : ℝ := 20

/-- Theorem stating that the given conditions result in the correct distance -/
theorem distance_calculation :
  let speed_with_current := rowing_speed + current_velocity
  let speed_against_current := rowing_speed - current_velocity
  (distance / speed_with_current) + (distance / speed_against_current) = total_time :=
sorry

end NUMINAMATH_CALUDE_distance_calculation_l2002_200227


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2002_200292

theorem arithmetic_calculations : 
  (2 / 5 - 1 / 5 * (-5) + 3 / 5 = 2) ∧ 
  (-2^2 - (-3)^3 / 3 * (1 / 3) = -1) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2002_200292


namespace NUMINAMATH_CALUDE_square_difference_l2002_200291

theorem square_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 2) : a^2 - b^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2002_200291


namespace NUMINAMATH_CALUDE_pencil_cost_l2002_200210

theorem pencil_cost (total_spent notebook_cost ruler_cost num_pencils : ℕ) 
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : ruler_cost = 18)
  (h4 : num_pencils = 3) :
  (total_spent - notebook_cost - ruler_cost) / num_pencils = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2002_200210


namespace NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l2002_200208

/-- Given real functions f and g defined on the entire real line, 
    satisfying certain conditions, prove that |g(y)| ≤ 1 for all y -/
theorem bounded_g_given_bounded_f (f g : ℝ → ℝ) 
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l2002_200208


namespace NUMINAMATH_CALUDE_football_lineup_combinations_l2002_200225

def team_size : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

def lineup_combinations : ℕ := 31680

theorem football_lineup_combinations :
  team_size = 12 ∧ 
  offensive_linemen = 4 ∧ 
  positions = 5 →
  lineup_combinations = offensive_linemen * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) :=
by sorry

end NUMINAMATH_CALUDE_football_lineup_combinations_l2002_200225


namespace NUMINAMATH_CALUDE_evaluate_expression_l2002_200266

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2002_200266


namespace NUMINAMATH_CALUDE_long_sleeve_shirts_to_wash_l2002_200284

theorem long_sleeve_shirts_to_wash :
  ∀ (total_shirts short_sleeve_shirts long_sleeve_shirts shirts_washed shirts_not_washed : ℕ),
    total_shirts = short_sleeve_shirts + long_sleeve_shirts →
    shirts_washed = 29 →
    shirts_not_washed = 1 →
    short_sleeve_shirts = 9 →
    long_sleeve_shirts = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_long_sleeve_shirts_to_wash_l2002_200284


namespace NUMINAMATH_CALUDE_jersey_profit_calculation_l2002_200297

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 185.85

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℝ := 240

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 177

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 23

/-- The difference in cost between a t-shirt and a jersey -/
def tshirt_jersey_diff : ℝ := 30

theorem jersey_profit_calculation :
  jersey_profit = (tshirts_sold * tshirt_profit) / (tshirts_sold + jerseys_sold) - tshirt_jersey_diff :=
by sorry

end NUMINAMATH_CALUDE_jersey_profit_calculation_l2002_200297


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2002_200256

/-- Calculates the mean daily profit for a month given the mean profits of two halves --/
def mean_daily_profit (days : ℕ) (first_half_mean : ℚ) (second_half_mean : ℚ) : ℚ :=
  (first_half_mean * (days / 2) + second_half_mean * (days / 2)) / days

/-- Proves that the mean daily profit for the given scenario is 350 --/
theorem shopkeeper_profit : mean_daily_profit 30 245 455 = 350 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2002_200256


namespace NUMINAMATH_CALUDE_rose_friends_count_l2002_200255

def total_apples : ℕ := 9
def apples_per_friend : ℕ := 3

theorem rose_friends_count : 
  total_apples / apples_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_rose_friends_count_l2002_200255


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2002_200283

/-- 
Given a line L1 with equation 2x + 3y - 6 = 0 and a point P(1, -1),
prove that the line L2 with equation 2x + 3y + 1 = 0 is parallel to L1 and passes through P.
-/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) →  -- Equation of L1
  (2 * 1 + 3 * (-1) + 1 = 0) →  -- L2 passes through P(1, -1)
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 ↔ 
    (∃ (k : ℝ), 2 * x + 3 * y = 2 * 1 + 3 * (-1) + k * (2 * 1 + 3 * (-1) - (2 * 1 + 3 * (-1))))) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2002_200283


namespace NUMINAMATH_CALUDE_total_spider_legs_l2002_200231

/-- The number of spiders in Zoey's room -/
def num_spiders : ℕ := 5

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in Zoey's room is 40 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l2002_200231


namespace NUMINAMATH_CALUDE_volunteer_group_selection_l2002_200269

def class_size : ℕ := 4
def total_classes : ℕ := 4
def group_size : ℕ := class_size * total_classes
def selection_size : ℕ := 3

def select_committee (n k : ℕ) : ℕ := Nat.choose n k

theorem volunteer_group_selection :
  let with_class3 := select_committee class_size 1 * select_committee (group_size - class_size) (selection_size - 1)
  let without_class3 := select_committee (group_size - class_size) selection_size - 
                        (total_classes - 1) * select_committee class_size selection_size
  with_class3 + without_class3 = 472 := by sorry

end NUMINAMATH_CALUDE_volunteer_group_selection_l2002_200269


namespace NUMINAMATH_CALUDE_joe_journey_time_l2002_200214

/-- Represents the scenario of Joe's journey from home to school -/
structure JourneyScenario where
  d : ℝ  -- Total distance from home to school
  walk_speed : ℝ  -- Joe's walking speed
  run_speed : ℝ  -- Joe's running speed
  walk_time : ℝ  -- Time Joe spent walking
  walk_distance : ℝ  -- Distance Joe walked
  run_distance : ℝ  -- Distance Joe ran

/-- The theorem stating the total time of Joe's journey -/
theorem joe_journey_time (scenario : JourneyScenario) :
  scenario.walk_distance = scenario.d / 3 ∧
  scenario.run_distance = 2 * scenario.d / 3 ∧
  scenario.run_speed = 4 * scenario.walk_speed ∧
  scenario.walk_time = 9 ∧
  scenario.walk_distance = scenario.walk_speed * scenario.walk_time ∧
  scenario.run_distance = scenario.run_speed * (13.5 - scenario.walk_time) →
  13.5 = scenario.walk_time + (scenario.run_distance / scenario.run_speed) :=
by sorry


end NUMINAMATH_CALUDE_joe_journey_time_l2002_200214


namespace NUMINAMATH_CALUDE_smallest_x_value_l2002_200285

/-- Given the equation x|x| = 3x + k and the inequality x + 2 ≤ 3,
    the smallest value of x that satisfies these conditions is -2 when k = 2. -/
theorem smallest_x_value (x : ℝ) (k : ℝ) : 
  (x * abs x = 3 * x + k) → 
  (x + 2 ≤ 3) → 
  (k = 2) →
  (∀ y : ℝ, (y * abs y = 3 * y + k) → (y + 2 ≤ 3) → (x ≤ y)) →
  x = -2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2002_200285


namespace NUMINAMATH_CALUDE_quarters_needed_l2002_200207

/-- Represents the cost of items in cents -/
def CandyBarCost : ℕ := 25
def ChocolatePieceCost : ℕ := 75
def JuicePackCost : ℕ := 50

/-- Represents the number of each item to be purchased -/
def CandyBarCount : ℕ := 3
def ChocolatePieceCount : ℕ := 2
def JuicePackCount : ℕ := 1

/-- Represents the value of a quarter in cents -/
def QuarterValue : ℕ := 25

/-- Calculates the total cost in cents -/
def TotalCost : ℕ := 
  CandyBarCost * CandyBarCount + 
  ChocolatePieceCost * ChocolatePieceCount + 
  JuicePackCost * JuicePackCount

/-- Theorem: The number of quarters needed is 11 -/
theorem quarters_needed : TotalCost / QuarterValue = 11 := by
  sorry

end NUMINAMATH_CALUDE_quarters_needed_l2002_200207


namespace NUMINAMATH_CALUDE_floor_ceiling_evaluation_l2002_200270

theorem floor_ceiling_evaluation : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ - ⌊(0.001 : ℝ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_evaluation_l2002_200270


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2002_200212

-- Define the grid size
def gridSize : ℕ := 6

-- Define the jump size
def jumpSize : ℕ := 2

-- Define a position on the grid
structure Position where
  x : ℕ
  y : ℕ

-- Define the starting position
def startPos : Position := ⟨2, 3⟩

-- Define a function to check if a position is on the vertical side
def isOnVerticalSide (p : Position) : Prop :=
  p.x = 0 ∨ p.x = gridSize

-- Define a function to check if a position is on any side
def isOnAnySide (p : Position) : Prop :=
  p.x = 0 ∨ p.x = gridSize ∨ p.y = 0 ∨ p.y = gridSize

-- Define the probability of ending on a vertical side
def probEndVertical (p : Position) : ℝ := sorry

-- State the theorem
theorem frog_jump_probability :
  probEndVertical startPos = 3/4 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2002_200212


namespace NUMINAMATH_CALUDE_prism_with_seven_faces_has_fifteen_edges_l2002_200278

/-- A prism is a polyhedron with two congruent and parallel faces (bases) 
    and all other faces are parallelograms (lateral faces). -/
structure Prism where
  faces : ℕ
  bases : ℕ
  lateral_faces : ℕ
  edges_per_base : ℕ
  lateral_edges : ℕ
  total_edges : ℕ

/-- The number of edges in a prism with 7 faces is 15. -/
theorem prism_with_seven_faces_has_fifteen_edges :
  ∀ (p : Prism), p.faces = 7 → p.total_edges = 15 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_seven_faces_has_fifteen_edges_l2002_200278


namespace NUMINAMATH_CALUDE_isotomic_lines_not_intersect_in_medial_triangle_l2002_200205

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The medial triangle of a given triangle --/
def medialTriangle (t : Triangle) : Triangle := sorry

/-- Checks if a point is inside or on the boundary of a triangle --/
def isInsideOrOnTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Checks if two lines are isotomic with respect to a triangle --/
def areIsotomicLines (l1 l2 : Line) (t : Triangle) : Prop := sorry

/-- The intersection point of two lines, if it exists --/
def lineIntersection (l1 l2 : Line) : Option (ℝ × ℝ) := sorry

theorem isotomic_lines_not_intersect_in_medial_triangle (t : Triangle) (l1 l2 : Line) :
  areIsotomicLines l1 l2 t →
  match lineIntersection l1 l2 with
  | some p => ¬isInsideOrOnTriangle p (medialTriangle t)
  | none => True
  := by sorry

end NUMINAMATH_CALUDE_isotomic_lines_not_intersect_in_medial_triangle_l2002_200205


namespace NUMINAMATH_CALUDE_bus_passenger_count_l2002_200294

/-- Calculates the final number of passengers on a bus after several stops -/
def final_passengers (initial : ℕ) (first_stop : ℕ) (off_other_stops : ℕ) (on_other_stops : ℕ) : ℕ :=
  initial + first_stop - off_other_stops + on_other_stops

/-- Theorem stating that given the specific passenger changes, the final number is 49 -/
theorem bus_passenger_count : final_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_bus_passenger_count_l2002_200294


namespace NUMINAMATH_CALUDE_same_solution_implies_a_and_b_l2002_200226

theorem same_solution_implies_a_and_b (a b : ℝ) :
  (∃ x y : ℝ, x - y = 0 ∧ 2*a*x + b*y = 4 ∧ 2*x + y = 3 ∧ a*x + b*y = 3) →
  a = 1 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_and_b_l2002_200226


namespace NUMINAMATH_CALUDE_max_distance_for_specific_car_l2002_200247

/-- Represents the lifespan of a set of tires in kilometers. -/
structure TireLifespan where
  km : ℕ

/-- Represents a car with front and rear tires. -/
structure Car where
  frontTires : TireLifespan
  rearTires : TireLifespan

/-- Calculates the maximum distance a car can travel with optimal tire swapping. -/
def maxDistance (car : Car) : ℕ :=
  sorry

/-- Theorem stating the maximum distance for a specific car configuration. -/
theorem max_distance_for_specific_car :
  let car := Car.mk (TireLifespan.mk 20000) (TireLifespan.mk 30000)
  maxDistance car = 24000 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_specific_car_l2002_200247


namespace NUMINAMATH_CALUDE_juan_number_puzzle_l2002_200213

theorem juan_number_puzzle (n : ℝ) : ((((n + 2) * 2) - 2) / 2) = 7 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_juan_number_puzzle_l2002_200213


namespace NUMINAMATH_CALUDE_smallest_number_l2002_200232

theorem smallest_number (π : Real) : 
  -π < -3 ∧ -π < -Real.sqrt 2 ∧ -π < -(5/2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2002_200232


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l2002_200204

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  let A : Set ℝ := {0, 1, a^2}
  let B : Set ℝ := {1, 0, 2*a+3}
  A ∩ B = A ∪ B → a = 3 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l2002_200204


namespace NUMINAMATH_CALUDE_inequality_proof_l2002_200260

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a) / (a^2 + b * c) + (2 * b) / (b^2 + c * a) + (2 * c) / (c^2 + a * b) ≤ 
  a / (b * c) + b / (c * a) + c / (a * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2002_200260


namespace NUMINAMATH_CALUDE_set_element_value_l2002_200245

theorem set_element_value (a : ℝ) : 2 ∈ ({0, a, a^2 - 3*a + 2} : Set ℝ) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_element_value_l2002_200245


namespace NUMINAMATH_CALUDE_solve_quadratic_l2002_200262

theorem solve_quadratic (x : ℝ) (h1 : x^2 - 4*x = 0) (h2 : x ≠ 0) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_l2002_200262


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2002_200282

theorem negation_of_proposition (p : Prop) : 
  (¬(∃ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 ∧ x₀^2 + 2*x₀ + 1 ≤ 0)) ↔ 
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → x^2 + 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2002_200282


namespace NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b_l2002_200244

theorem a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b :
  ¬(∀ a b : ℝ, a > b → |a| > |b|) ∧ ¬(∀ a b : ℝ, |a| > |b| → a > b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b_l2002_200244


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2002_200249

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0

def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- State the theorem
theorem p_necessary_not_sufficient :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬q a) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2002_200249


namespace NUMINAMATH_CALUDE_exp_13pi_i_div_2_eq_i_l2002_200290

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem exp_13pi_i_div_2_eq_i : complex_exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_13pi_i_div_2_eq_i_l2002_200290


namespace NUMINAMATH_CALUDE_triangle_ab_length_l2002_200252

/-- Given a triangle ABC with angles B and C both 45 degrees and side BC of length 10,
    prove that the length of side AB is 5√2. -/
theorem triangle_ab_length (A B C : ℝ × ℝ) : 
  let triangle := (A, B, C)
  let angle (X Y Z : ℝ × ℝ) := Real.arccos ((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) / 
    (((X.1 - Y.1)^2 + (X.2 - Y.2)^2).sqrt * ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2).sqrt)
  let distance (X Y : ℝ × ℝ) := ((X.1 - Y.1)^2 + (X.2 - Y.2)^2).sqrt
  angle B A C = π/4 →
  angle C B A = π/4 →
  distance B C = 10 →
  distance A B = 5 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_ab_length_l2002_200252


namespace NUMINAMATH_CALUDE_greatest_number_of_bouquets_l2002_200251

theorem greatest_number_of_bouquets (white_tulips red_tulips : ℕ) 
  (h1 : white_tulips = 21) (h2 : red_tulips = 91) : 
  (Nat.gcd white_tulips red_tulips) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_bouquets_l2002_200251


namespace NUMINAMATH_CALUDE_team_incorrect_answers_contest_result_l2002_200265

theorem team_incorrect_answers 
  (total_questions : Nat) 
  (riley_incorrect : Nat) 
  (ofelia_correct_addition : Nat) : Nat :=
  let riley_correct := total_questions - riley_incorrect
  let ofelia_correct := riley_correct / 2 + ofelia_correct_addition
  let ofelia_incorrect := total_questions - ofelia_correct
  riley_incorrect + ofelia_incorrect

#check @team_incorrect_answers

theorem contest_result : 
  team_incorrect_answers 35 3 5 = 17 := by
  sorry

#check @contest_result

end NUMINAMATH_CALUDE_team_incorrect_answers_contest_result_l2002_200265


namespace NUMINAMATH_CALUDE_cube_painting_probability_l2002_200230

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a painted cube -/
def Cube := Fin 6 → Color

/-- The number of ways to paint a single cube -/
def total_single_cube_paintings : ℕ := 729

/-- The total number of ways to paint two cubes -/
def total_two_cube_paintings : ℕ := 531441

/-- The number of ways to paint two cubes so they are identical after rotation -/
def identical_after_rotation : ℕ := 1178

/-- The probability that two independently painted cubes are identical after rotation -/
def probability_identical_after_rotation : ℚ := 1178 / 531441

theorem cube_painting_probability :
  probability_identical_after_rotation = identical_after_rotation / total_two_cube_paintings :=
by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l2002_200230


namespace NUMINAMATH_CALUDE_smallest_n_squared_existence_of_solution_smallest_n_is_11_l2002_200224

theorem smallest_n_squared (n : ℕ+) : 
  (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) →
  n ≥ 11 :=
by sorry

theorem existence_of_solution : 
  ∃ (x y z : ℕ+), 11^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11 :=
by sorry

theorem smallest_n_is_11 : 
  (∃ (n : ℕ+), ∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) ∧
  (∀ (m : ℕ+), (∃ (x y z : ℕ+), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) → m ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_squared_existence_of_solution_smallest_n_is_11_l2002_200224


namespace NUMINAMATH_CALUDE_marion_additional_points_l2002_200223

/-- Represents the additional points Marion got on the exam -/
def additional_points (total_items : ℕ) (ella_incorrect : ℕ) (marion_score : ℕ) : ℕ :=
  marion_score - (total_items - ella_incorrect) / 2

/-- Proves that Marion got 6 additional points given the exam conditions -/
theorem marion_additional_points :
  additional_points 40 4 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_marion_additional_points_l2002_200223


namespace NUMINAMATH_CALUDE_cyclist_speed_l2002_200280

/-- The cyclist's problem -/
theorem cyclist_speed (initial_time : ℝ) (faster_time : ℝ) (faster_speed : ℝ) :
  initial_time = 6 →
  faster_time = 3 →
  faster_speed = 14 →
  ∃ (distance : ℝ) (initial_speed : ℝ),
    distance = initial_speed * initial_time ∧
    distance = faster_speed * faster_time ∧
    initial_speed = 7 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_l2002_200280


namespace NUMINAMATH_CALUDE_max_value_of_z_l2002_200259

theorem max_value_of_z (x y : ℝ) (h1 : x + 2*y - 5 ≥ 0) (h2 : x - 2*y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∀ x' y', x' + 2*y' - 5 ≥ 0 → x' - 2*y' + 3 ≥ 0 → x' - 5 ≤ 0 → x + y ≥ x' + y' ∧ x + y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2002_200259


namespace NUMINAMATH_CALUDE_binomial_square_exists_l2002_200206

theorem binomial_square_exists : ∃ (b t u : ℝ), ∀ x : ℝ, b * x^2 + 12 * x + 9 = (t * x + u)^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_exists_l2002_200206


namespace NUMINAMATH_CALUDE_some_number_value_l2002_200228

theorem some_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2002_200228


namespace NUMINAMATH_CALUDE_f_g_four_zeros_implies_a_range_l2002_200203

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*a*x - a + 1 else Real.log (-x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 1 - 2*a

def has_four_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧
    ∀ x, f x = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

theorem f_g_four_zeros_implies_a_range (a : ℝ) :
  has_four_zeros (f a ∘ g a) →
  a ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) 1 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_f_g_four_zeros_implies_a_range_l2002_200203


namespace NUMINAMATH_CALUDE_distance_AB_is_70_l2002_200209

/-- The distance between two points A and B, given specific travel conditions of two couriers --/
def distance_AB : ℝ := by sorry

theorem distance_AB_is_70 :
  let t₁ := 14 -- Travel time for first courier in hours
  let d := 10 -- Distance behind A where second courier starts in km
  let x := distance_AB -- Distance from A to B in km
  let v₁ := x / t₁ -- Speed of first courier
  let v₂ := (x + d) / t₁ -- Speed of second courier
  let t₁_20 := 20 / v₁ -- Time for first courier to travel 20 km
  let t₂_20 := 20 / v₂ -- Time for second courier to travel 20 km
  t₁_20 = t₂_20 + 0.5 → -- Second courier is half hour faster over 20 km
  x = 70 := by sorry

end NUMINAMATH_CALUDE_distance_AB_is_70_l2002_200209


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2002_200253

/-- A point lies on the x-axis if and only if its y-coordinate is 0 -/
def lies_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- The coordinates of point A as a function of x -/
def point_A (x : ℝ) : ℝ × ℝ := (2 - x, x + 3)

/-- Theorem stating that if point A lies on the x-axis, its coordinates are (5, 0) -/
theorem point_A_coordinates :
  ∃ x : ℝ, lies_on_x_axis (point_A x) → point_A x = (5, 0) := by
  sorry


end NUMINAMATH_CALUDE_point_A_coordinates_l2002_200253


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2002_200250

theorem triangle_max_perimeter (A B C : ℝ) (b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  (1 - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C →
  1 + b + c ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2002_200250


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2002_200229

theorem cos_alpha_value (α : Real) (h : Real.cos (Real.pi + α) = -1/3) : Real.cos α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2002_200229


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2002_200261

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 1 → b = 2 → c^2 = a^2 + b^2 → c = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2002_200261


namespace NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l2002_200242

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l2002_200242


namespace NUMINAMATH_CALUDE_cubic_root_theorem_l2002_200254

theorem cubic_root_theorem (b c : ℚ) : 
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 3 - Real.sqrt 7) →
  ((-6 : ℝ)^3 + b*(-6) + c = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_root_theorem_l2002_200254


namespace NUMINAMATH_CALUDE_determine_original_prices_l2002_200200

/-- Represents a purchase of products A and B -/
structure Purchase where
  quantityA : ℕ
  quantityB : ℕ
  totalPrice : ℕ

/-- Represents the store's pricing system -/
structure Store where
  priceA : ℕ
  priceB : ℕ

/-- Checks if a purchase is consistent with the store's pricing -/
def isPurchaseConsistent (s : Store) (p : Purchase) : Prop :=
  s.priceA * p.quantityA + s.priceB * p.quantityB = p.totalPrice

/-- The theorem stating that given the purchase data, we can determine the original prices -/
theorem determine_original_prices 
  (p1 p2 : Purchase)
  (h1 : p1.quantityA = 6 ∧ p1.quantityB = 5 ∧ p1.totalPrice = 1140)
  (h2 : p2.quantityA = 3 ∧ p2.quantityB = 7 ∧ p2.totalPrice = 1110) :
  ∃ (s : Store), 
    s.priceA = 90 ∧ 
    s.priceB = 120 ∧ 
    isPurchaseConsistent s p1 ∧ 
    isPurchaseConsistent s p2 :=
  sorry

end NUMINAMATH_CALUDE_determine_original_prices_l2002_200200


namespace NUMINAMATH_CALUDE_min_cuts_for_100_polygons_l2002_200240

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents the state of the paper after cutting -/
structure PaperState where
  pieces : ℕ
  total_vertices : ℕ

/-- Initial state of the square paper -/
def initial_state : PaperState :=
  { pieces := 1, total_vertices := 4 }

/-- Function to model a single cut -/
def cut (state : PaperState) (new_vertices : ℕ) : PaperState :=
  { pieces := state.pieces + 1,
    total_vertices := state.total_vertices + new_vertices }

/-- Predicate to check if the final state is valid -/
def is_valid_final_state (state : PaperState) : Prop :=
  state.pieces = 100 ∧ state.total_vertices = 100 * 20

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_100_polygons :
  ∃ (n : ℕ), n = 1699 ∧
  ∃ (cut_sequence : List ℕ),
    cut_sequence.length = n ∧
    (cut_sequence.all (λ x => x ∈ [2, 3, 4])) ∧
    is_valid_final_state (cut_sequence.foldl cut initial_state) ∧
    ∀ (m : ℕ) (other_sequence : List ℕ),
      m < n →
      other_sequence.length = m →
      (other_sequence.all (λ x => x ∈ [2, 3, 4])) →
      ¬is_valid_final_state (other_sequence.foldl cut initial_state) :=
sorry


end NUMINAMATH_CALUDE_min_cuts_for_100_polygons_l2002_200240


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l2002_200246

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l2002_200246


namespace NUMINAMATH_CALUDE_projection_equality_l2002_200293

def a : Fin 2 → ℚ := ![3, -2]
def b : Fin 2 → ℚ := ![6, -1]
def p : Fin 2 → ℚ := ![9/10, -27/10]

theorem projection_equality (v : Fin 2 → ℚ) (hv : v ≠ 0) :
  (v • a / (v • v)) • v = (v • b / (v • b)) • v → 
  (v • a / (v • v)) • v = p :=
by sorry

end NUMINAMATH_CALUDE_projection_equality_l2002_200293


namespace NUMINAMATH_CALUDE_amber_max_ounces_l2002_200216

def amber_money : ℚ := 7
def candy_price : ℚ := 1
def candy_ounces : ℚ := 12
def chips_price : ℚ := 1.4
def chips_ounces : ℚ := 17

def candy_bags : ℚ := amber_money / candy_price
def chips_bags : ℚ := amber_money / chips_price

def total_candy_ounces : ℚ := candy_bags * candy_ounces
def total_chips_ounces : ℚ := chips_bags * chips_ounces

theorem amber_max_ounces :
  max total_candy_ounces total_chips_ounces = 85 :=
by sorry

end NUMINAMATH_CALUDE_amber_max_ounces_l2002_200216


namespace NUMINAMATH_CALUDE_sum_of_ratios_l2002_200222

theorem sum_of_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x/y + y/z + z/x + y/x + z/y + x/z = 9)
  (h2 : x + y + z = 3) :
  x/y + y/z + z/x = 4.5 ∧ y/x + z/y + x/z = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l2002_200222


namespace NUMINAMATH_CALUDE_third_number_value_l2002_200263

theorem third_number_value (a b c : ℕ) : 
  a + b + c = 500 → 
  a = 200 → 
  b = 2 * c → 
  c = 100 := by
sorry

end NUMINAMATH_CALUDE_third_number_value_l2002_200263


namespace NUMINAMATH_CALUDE_shaded_cubes_is_14_l2002_200248

/-- Represents a 3x3x3 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per edge -/
  edge_length : Nat
  /-- Number of shaded cubes per face -/
  shaded_per_face : Nat
  /-- Total number of faces -/
  total_faces : Nat
  /-- Condition: total cubes is 27 -/
  total_is_27 : total_cubes = 27
  /-- Condition: edge length is 3 -/
  edge_is_3 : edge_length = 3
  /-- Condition: 5 cubes shaded per face (4 corners + 1 center) -/
  shaded_is_5 : shaded_per_face = 5
  /-- Condition: cube has 6 faces -/
  faces_is_6 : total_faces = 6

/-- Function to calculate the number of uniquely shaded cubes -/
def uniquelyShadedCubes (c : ShadedCube) : Nat :=
  8 + 6 -- 8 corner cubes + 6 center face cubes

/-- Theorem: The number of uniquely shaded cubes is 14 -/
theorem shaded_cubes_is_14 (c : ShadedCube) : uniquelyShadedCubes c = 14 := by
  sorry

#check shaded_cubes_is_14

end NUMINAMATH_CALUDE_shaded_cubes_is_14_l2002_200248


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2002_200218

/-- Given a geometric sequence {a_n} with common ratio q, prove that if the sum of the first n terms S_n
    satisfies S_2 = 2a_2 + 3 and S_3 = 2a_3 + 3, then q = 2. -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, a (n + 1) = a n * q)  -- Definition of geometric sequence
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))  -- Sum formula for geometric sequence
  (h3 : S 2 = 2 * a 2 + 3)  -- Given condition for S_2
  (h4 : S 3 = 2 * a 3 + 3)  -- Given condition for S_3
  : q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2002_200218


namespace NUMINAMATH_CALUDE_fourth_number_proof_l2002_200289

theorem fourth_number_proof : ∃ x : ℕ, 9548 + 7314 = 3362 + x ∧ x = 13500 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l2002_200289


namespace NUMINAMATH_CALUDE_min_value_exponential_function_l2002_200258

theorem min_value_exponential_function (x : ℝ) : Real.exp x + 4 * Real.exp (-x) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_function_l2002_200258


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2002_200274

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- arithmetic sequence definition
  q > 1 →  -- common ratio condition
  a 1 + a 4 = 9 →  -- first condition
  a 2 * a 3 = 8 →  -- second condition
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2002_200274


namespace NUMINAMATH_CALUDE_double_root_equations_l2002_200221

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₂ = 2 * x₁

/-- Theorem about double root equations -/
theorem double_root_equations :
  (is_double_root_equation 1 (-3) 2) ∧
  (∀ m n : ℝ, is_double_root_equation 1 (-2-m) (2*n) → 4*m^2 + 5*m*n + n^2 = 0) ∧
  (∀ p q : ℝ, p * q = 2 → is_double_root_equation p 3 q) := by
  sorry


end NUMINAMATH_CALUDE_double_root_equations_l2002_200221


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2002_200237

def fraction_sequence : List ℚ := 
  [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 11/10, 12/10, 13/10]

theorem sum_of_fractions : 
  fraction_sequence.sum = 91/10 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2002_200237


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2002_200267

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - Complex.I) / Complex.I ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2002_200267


namespace NUMINAMATH_CALUDE_smallest_bdf_value_l2002_200220

theorem smallest_bdf_value (a b c d e f : ℕ+) : 
  let expr := (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f
  (expr + 3 = ((a + 1 : ℕ+) : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f) →
  (expr + 4 = (a : ℚ) / b * ((c + 1 : ℕ+) : ℚ) / d * (e : ℚ) / f) →
  (expr + 5 = (a : ℚ) / b * (c : ℚ) / d * ((e + 1 : ℕ+) : ℚ) / f) →
  (∀ k : ℕ+, (b * d * f : ℕ) = k → k ≥ 60) ∧ 
  (∃ b' d' f' : ℕ+, (b' * d' * f' : ℕ) = 60) :=
by sorry

end NUMINAMATH_CALUDE_smallest_bdf_value_l2002_200220


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2002_200215

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2002_200215


namespace NUMINAMATH_CALUDE_pascal_identity_l2002_200288

theorem pascal_identity (n k : ℕ) (h1 : k ≤ n) (h2 : ¬(n = 0 ∧ k = 0)) : 
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_pascal_identity_l2002_200288


namespace NUMINAMATH_CALUDE_no_answer_paradox_correct_answer_is_no_l2002_200264

/-- Represents the possible answers Alice can give to the Black Queen's question -/
inductive Answer
  | Yes
  | No

/-- Represents the possible outcomes of Alice's exam -/
inductive ExamResult
  | Pass
  | Fail

/-- Represents the Black Queen's judgment based on Alice's answer -/
def blackQueenJudgment (answer : Answer) : ExamResult → Prop :=
  match answer with
  | Answer.Yes => fun result => 
      (result = ExamResult.Pass → False) ∧ 
      (result = ExamResult.Fail → False)
  | Answer.No => fun result => 
      (result = ExamResult.Pass → False) ∧ 
      (result = ExamResult.Fail → False)

/-- Theorem stating that answering "No" creates an unresolvable paradox -/
theorem no_answer_paradox : 
  ∀ (result : ExamResult), blackQueenJudgment Answer.No result → False :=
by
  sorry

/-- Theorem stating that "No" is the correct answer to avoid failing the exam -/
theorem correct_answer_is_no : 
  ∀ (answer : Answer), 
    (∀ (result : ExamResult), blackQueenJudgment answer result → False) → 
    answer = Answer.No :=
by
  sorry

end NUMINAMATH_CALUDE_no_answer_paradox_correct_answer_is_no_l2002_200264


namespace NUMINAMATH_CALUDE_min_area_PJ1J2_l2002_200217

/-- Triangle PQR with given side lengths -/
structure Triangle (PQ QR PR : ℝ) where
  side_PQ : PQ = 26
  side_QR : QR = 28
  side_PR : PR = 30

/-- Point Y on side QR -/
def Y (QR : ℝ) := ℝ

/-- Incenter of a triangle -/
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_area_PJ1J2 (t : Triangle 26 28 30) (y : Y 28) :
  ∃ (P Q R : ℝ × ℝ),
    let J1 := incenter P Q (0, y)
    let J2 := incenter P R (0, y)
    ∀ (y' : Y 28),
      let J1' := incenter P Q (0, y')
      let J2' := incenter P R (0, y')
      triangle_area P J1 J2 ≤ triangle_area P J1' J2' ∧
      (∃ (y_min : Y 28), triangle_area P J1 J2 = 51) := by
  sorry

end NUMINAMATH_CALUDE_min_area_PJ1J2_l2002_200217


namespace NUMINAMATH_CALUDE_equation_solution_l2002_200239

theorem equation_solution : 
  ∃ c : ℚ, (c - 35) / 14 = (2 * c + 9) / 49 ∧ c = 1841 / 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2002_200239


namespace NUMINAMATH_CALUDE_audrey_heracles_age_difference_l2002_200235

/-- The age difference between Audrey and Heracles -/
def ageDifference (audreyAge : ℕ) (heraclesAge : ℕ) : ℕ :=
  audreyAge - heraclesAge

theorem audrey_heracles_age_difference :
  ∃ (audreyAge : ℕ),
    heraclesAge = 10 ∧
    audreyAge + 3 = 2 * heraclesAge ∧
    ageDifference audreyAge heraclesAge = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_audrey_heracles_age_difference_l2002_200235


namespace NUMINAMATH_CALUDE_square_conditions_solutions_l2002_200295

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

theorem square_conditions_solutions :
  ∀ a b : ℕ+,
  (is_perfect_square (a.val^2 - 4*b.val) ∧ is_perfect_square (b.val^2 - 4*a.val)) ↔
  ((a, b) = (⟨4, by norm_num⟩, ⟨4, by norm_num⟩) ∨
   (a, b) = (⟨5, by norm_num⟩, ⟨6, by norm_num⟩) ∨
   (a, b) = (⟨6, by norm_num⟩, ⟨5, by norm_num⟩)) :=
by sorry

end NUMINAMATH_CALUDE_square_conditions_solutions_l2002_200295


namespace NUMINAMATH_CALUDE_implicit_second_derivative_l2002_200233

noncomputable def y (x : ℝ) : ℝ := Real.exp x

theorem implicit_second_derivative 
  (h : ∀ x, y x * Real.exp x + Real.exp (y x) = Real.exp 1 + 1) :
  ∀ x, deriv (deriv y) x = 
    (-2 * Real.exp (2*x) * y x * (Real.exp x + Real.exp (y x)) + 
     y x * Real.exp x * (Real.exp x + Real.exp (y x))^2 + 
     (y x)^2 * Real.exp (y x) * Real.exp (2*x)) / 
    (Real.exp x + Real.exp (y x))^3 :=
by
  sorry

end NUMINAMATH_CALUDE_implicit_second_derivative_l2002_200233


namespace NUMINAMATH_CALUDE_square_root_problem_l2002_200296

theorem square_root_problem (x y : ℝ) : 
  (∃ k : ℤ, x + 7 = (3 * k)^2) → 
  ((2*x - y - 13)^(1/3) = -2) → 
  (5*x - 6*y)^(1/2) = 4 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2002_200296


namespace NUMINAMATH_CALUDE_additive_is_odd_l2002_200201

/-- A function satisfying the given additive property -/
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- Theorem stating that an additive function is odd -/
theorem additive_is_odd (f : ℝ → ℝ) (h : is_additive f) : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_additive_is_odd_l2002_200201


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l2002_200241

theorem binomial_equation_solution (x : ℕ) : 
  (Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4) → (x = 5 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l2002_200241


namespace NUMINAMATH_CALUDE_time_difference_for_trips_l2002_200219

/-- Given a truck traveling at a constant speed, this theorem proves the time difference
    between two trips of different distances. -/
theorem time_difference_for_trips
  (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ)
  (h1 : speed = 60)  -- Speed in miles per hour
  (h2 : distance1 = 570)  -- Distance of first trip in miles
  (h3 : distance2 = 540)  -- Distance of second trip in miles
  : (distance1 - distance2) / speed * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_for_trips_l2002_200219


namespace NUMINAMATH_CALUDE_natural_number_equality_l2002_200286

def divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem natural_number_equality (a b : ℕ) 
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, divisible (a^(n+1) + b^(n+1)) (a^n + b^n)) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_natural_number_equality_l2002_200286


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_twelve_l2002_200279

/-- The sum of the tens digit and the ones digit of (1+6)^12 is 1 -/
theorem sum_of_digits_of_seven_to_twelve : 
  (((1 + 6)^12 / 10) % 10 + (1 + 6)^12 % 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_twelve_l2002_200279


namespace NUMINAMATH_CALUDE_trapezoid_circle_radii_l2002_200257

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  -- Base lengths
  BC : ℝ
  AD : ℝ
  -- Inscribed circle exists
  has_inscribed_circle : Bool
  -- Circumscribed circle exists
  has_circumscribed_circle : Bool

/-- The radii of inscribed and circumscribed circles of an isosceles trapezoid -/
def circle_radii (t : IsoscelesTrapezoid) : ℝ × ℝ :=
  sorry

theorem trapezoid_circle_radii (t : IsoscelesTrapezoid) 
  (h1 : t.BC = 4)
  (h2 : t.AD = 16)
  (h3 : t.has_inscribed_circle = true)
  (h4 : t.has_circumscribed_circle = true) :
  circle_radii t = (4, 5 * Real.sqrt 41 / 4) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_circle_radii_l2002_200257


namespace NUMINAMATH_CALUDE_max_value_of_function_l2002_200234

theorem max_value_of_function (x : ℝ) (hx : x < 0) :
  ∃ (max : ℝ), max = -4 * Real.sqrt 3 ∧ ∀ y, y = 3 * x + 4 / x → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2002_200234


namespace NUMINAMATH_CALUDE_pump_operations_proof_l2002_200268

/-- The fraction of air remaining after one pump operation -/
def pump_efficiency : ℝ := 0.5

/-- The target fraction of air remaining -/
def target_fraction : ℝ := 0.001

/-- The minimum number of pump operations needed to reach the target fraction -/
def min_operations : ℕ := 10

theorem pump_operations_proof :
  (∀ n : ℕ, n < min_operations → (pump_efficiency ^ n : ℝ) > target_fraction) ∧
  (pump_efficiency ^ min_operations : ℝ) ≤ target_fraction :=
sorry

end NUMINAMATH_CALUDE_pump_operations_proof_l2002_200268


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2002_200287

theorem min_fraction_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  1 ≤ a ∧ a ≤ 10 →
  1 ≤ b ∧ b ≤ 10 →
  1 ≤ c ∧ c ≤ 10 →
  1 ≤ d ∧ d ≤ 10 →
  (a : ℚ) / b + (c : ℚ) / d ≥ 14 / 45 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2002_200287


namespace NUMINAMATH_CALUDE_stock_profit_percentage_l2002_200276

theorem stock_profit_percentage 
  (stock_worth : ℝ) 
  (profit_portion : ℝ) 
  (loss_portion : ℝ) 
  (loss_percentage : ℝ) 
  (overall_loss : ℝ) 
  (h1 : stock_worth = 12499.99)
  (h2 : profit_portion = 0.2)
  (h3 : loss_portion = 0.8)
  (h4 : loss_percentage = 0.1)
  (h5 : overall_loss = 500) :
  ∃ (P : ℝ), 
    (stock_worth * profit_portion * (1 + P / 100) + 
     stock_worth * loss_portion * (1 - loss_percentage) = 
     stock_worth - overall_loss) ∧ 
    (abs (P - 20.04) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_stock_profit_percentage_l2002_200276


namespace NUMINAMATH_CALUDE_binomial_25_2_l2002_200202

theorem binomial_25_2 : Nat.choose 25 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_binomial_25_2_l2002_200202


namespace NUMINAMATH_CALUDE_farmer_price_l2002_200298

def potato_problem (x : ℝ) : Prop :=
  let andrey_revenue := 60 * (2 * x)
  let boris_revenue := 15 * (1.6 * x) + 45 * (2.24 * x)
  boris_revenue - andrey_revenue = 1200

theorem farmer_price : ∃ x : ℝ, potato_problem x ∧ x = 250 := by
  sorry

end NUMINAMATH_CALUDE_farmer_price_l2002_200298


namespace NUMINAMATH_CALUDE_isabella_marble_problem_l2002_200238

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_marble_problem :
  ∀ k : ℕ, k < 45 → P k ≥ 1 / 2023 ∧ P 45 < 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_isabella_marble_problem_l2002_200238


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2002_200243

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 - 2*x + x^2) = 9 ↔ x = 1 + Real.sqrt 78 ∨ x = 1 - Real.sqrt 78 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2002_200243


namespace NUMINAMATH_CALUDE_vasyas_premium_will_increase_l2002_200272

/-- Represents a car insurance policy -/
structure CarInsurancePolicy where
  premium : ℝ
  hadAccident : Bool

/-- Represents an insurance company -/
class InsuranceCompany where
  renewPolicy : CarInsurancePolicy → CarInsurancePolicy

/-- Axiom: Insurance companies increase premiums for policies with accidents -/
axiom premium_increase_after_accident (company : InsuranceCompany) (policy : CarInsurancePolicy) :
  policy.hadAccident → (company.renewPolicy policy).premium > policy.premium

/-- Theorem: Vasya's insurance premium will increase after his car accident -/
theorem vasyas_premium_will_increase (company : InsuranceCompany) (vasyas_policy : CarInsurancePolicy) 
    (h_accident : vasyas_policy.hadAccident) : 
  (company.renewPolicy vasyas_policy).premium > vasyas_policy.premium :=
by
  sorry


end NUMINAMATH_CALUDE_vasyas_premium_will_increase_l2002_200272


namespace NUMINAMATH_CALUDE_rational_function_sum_l2002_200281

/-- Given rational functions r and s, prove r(x) + s(x) = -x^3 + 3x under specific conditions -/
theorem rational_function_sum (r s : ℝ → ℝ) : 
  (∃ (a b : ℝ), s x = a * (x - 2) * (x + 2) * x) →  -- s(x) is cubic with roots at 2, -2, and 0
  (∃ (b : ℝ), r x = b * x) →  -- r(x) is linear with a root at 0
  r (-1) = 1 →  -- condition on r
  s 1 = 3 →  -- condition on s
  ∀ x, r x + s x = -x^3 + 3*x := by
  sorry

end NUMINAMATH_CALUDE_rational_function_sum_l2002_200281


namespace NUMINAMATH_CALUDE_kelvin_expected_score_l2002_200211

/-- Represents the coin flipping game --/
structure CoinGame where
  /-- The number of coins Kelvin starts with --/
  initialCoins : ℕ
  /-- The probability of getting heads on a single coin flip --/
  headsProbability : ℝ

/-- Calculates the expected score for the game --/
noncomputable def expectedScore (game : CoinGame) : ℝ :=
  sorry

/-- Theorem stating the expected score for Kelvin's specific game --/
theorem kelvin_expected_score :
  let game : CoinGame := { initialCoins := 2, headsProbability := 1/2 }
  expectedScore game = 64/9 := by
  sorry

end NUMINAMATH_CALUDE_kelvin_expected_score_l2002_200211


namespace NUMINAMATH_CALUDE_basic_computer_price_l2002_200299

theorem basic_computer_price
  (total_price : ℝ)
  (price_difference : ℝ)
  (printer_ratio : ℝ)
  (h1 : total_price = 2500)
  (h2 : price_difference = 500)
  (h3 : printer_ratio = 1/4)
  : ∃ (basic_computer_price printer_price : ℝ),
    basic_computer_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_computer_price + price_difference + printer_price) ∧
    basic_computer_price = 1750 :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l2002_200299


namespace NUMINAMATH_CALUDE_count_ways_2024_l2002_200236

/-- The target sum -/
def target_sum : ℕ := 2024

/-- The set of allowed numbers -/
def allowed_numbers : Finset ℕ := {2, 3, 4}

/-- A function that counts the number of ways to express a target sum
    as a sum of non-negative integer multiples of allowed numbers,
    ignoring the order of summands -/
noncomputable def count_ways (target : ℕ) (allowed : Finset ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 57231 ways to express 2024
    as a sum of non-negative integer multiples of 2, 3, and 4,
    ignoring the order of summands -/
theorem count_ways_2024 :
  count_ways target_sum allowed_numbers = 57231 :=
by sorry

end NUMINAMATH_CALUDE_count_ways_2024_l2002_200236


namespace NUMINAMATH_CALUDE_find_b_value_l2002_200273

theorem find_b_value (x y b : ℝ) 
  (eq1 : 7^(3*x - 1) * b^(4*y - 3) = 49^x * 27^y)
  (eq2 : x + y = 4) : 
  b = 3 := by sorry

end NUMINAMATH_CALUDE_find_b_value_l2002_200273


namespace NUMINAMATH_CALUDE_roses_cut_correct_l2002_200271

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

theorem roses_cut_correct (initial_roses final_roses : ℕ) 
  (h : initial_roses ≤ final_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16  -- Should output 10

end NUMINAMATH_CALUDE_roses_cut_correct_l2002_200271
