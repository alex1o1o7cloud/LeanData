import Mathlib

namespace min_coefficient_value_l3574_357447

theorem min_coefficient_value (a b box : ℤ) : 
  (∀ x : ℝ, (a * x + b) * (b * x + a) = 10 * x^2 + box * x + 10) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  box ≥ 29 :=
by sorry

end min_coefficient_value_l3574_357447


namespace divisibility_by_five_l3574_357433

theorem divisibility_by_five (a : ℤ) : 
  5 ∣ (a^3 + 3*a + 1) ↔ a % 5 = 1 ∨ a % 5 = 2 := by
  sorry

end divisibility_by_five_l3574_357433


namespace bertha_descendants_no_daughters_l3574_357401

/-- Represents a person in Bertha's family tree -/
inductive Person
| bertha : Person
| child : Person → Person
| grandchild : Person → Person
| greatgrandchild : Person → Person

/-- Represents the gender of a person -/
inductive Gender
| male
| female

/-- Function to determine the gender of a person -/
def gender : Person → Gender
| Person.bertha => Gender.female
| _ => sorry

/-- Function to count the number of daughters a person has -/
def daughterCount : Person → Nat
| Person.bertha => 7
| _ => sorry

/-- Function to count the number of sons a person has -/
def sonCount : Person → Nat
| Person.bertha => 3
| _ => sorry

/-- Function to count the total number of female descendants of a person -/
def femaleDescendantCount : Person → Nat
| Person.bertha => 40
| _ => sorry

/-- Function to determine if a person has exactly three daughters -/
def hasThreeDaughters : Person → Bool
| _ => sorry

/-- Function to count the number of descendants (including the person) who have no daughters -/
def descendantsWithNoDaughters : Person → Nat
| _ => sorry

/-- Theorem stating that the number of Bertha's descendants with no daughters is 28 -/
theorem bertha_descendants_no_daughters :
  descendantsWithNoDaughters Person.bertha = 28 := by sorry

end bertha_descendants_no_daughters_l3574_357401


namespace apple_cost_is_14_l3574_357464

/-- Represents the cost of groceries in dollars -/
structure GroceryCost where
  total : ℕ
  bananas : ℕ
  bread : ℕ
  milk : ℕ

/-- Calculates the cost of apples given the total cost and costs of other items -/
def appleCost (g : GroceryCost) : ℕ :=
  g.total - (g.bananas + g.bread + g.milk)

/-- Theorem stating that the cost of apples is 14 dollars given the specific grocery costs -/
theorem apple_cost_is_14 (g : GroceryCost) 
    (h1 : g.total = 42)
    (h2 : g.bananas = 12)
    (h3 : g.bread = 9)
    (h4 : g.milk = 7) : 
  appleCost g = 14 := by
  sorry

#eval appleCost { total := 42, bananas := 12, bread := 9, milk := 7 }

end apple_cost_is_14_l3574_357464


namespace only_sample_size_statement_correct_l3574_357489

/-- Represents a statistical study with a population and a sample. -/
structure StatisticalStudy where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a statement about the statistical study. -/
inductive Statement
  | sample_is_population
  | sample_average_is_population_average
  | examinees_are_population
  | sample_size_is_1000

/-- Checks if a statement is correct for the given statistical study. -/
def is_correct_statement (study : StatisticalStudy) (stmt : Statement) : Prop :=
  match stmt with
  | Statement.sample_is_population => False
  | Statement.sample_average_is_population_average => False
  | Statement.examinees_are_population => False
  | Statement.sample_size_is_1000 => study.sample_size = 1000

/-- The main theorem stating that only the sample size statement is correct. -/
theorem only_sample_size_statement_correct (study : StatisticalStudy) 
    (h1 : study.population_size = 70000)
    (h2 : study.sample_size = 1000) :
    ∀ (stmt : Statement), is_correct_statement study stmt ↔ stmt = Statement.sample_size_is_1000 := by
  sorry

end only_sample_size_statement_correct_l3574_357489


namespace time_to_fill_leaking_basin_l3574_357462

/-- Calculates the time to fill a leaking basin from a waterfall -/
theorem time_to_fill_leaking_basin 
  (waterfall_flow : ℝ) 
  (basin_capacity : ℝ) 
  (leak_rate : ℝ) 
  (h1 : waterfall_flow = 24)
  (h2 : basin_capacity = 260)
  (h3 : leak_rate = 4) : 
  basin_capacity / (waterfall_flow - leak_rate) = 13 := by
sorry


end time_to_fill_leaking_basin_l3574_357462


namespace P_equals_complement_union_l3574_357431

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 ≠ p.1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ -p.1}

-- Define set P
def P : Set (ℝ × ℝ) := {p | p.2^2 ≠ p.1^2}

-- Theorem statement
theorem P_equals_complement_union :
  P = (U \ M) ∪ (U \ N) := by sorry

end P_equals_complement_union_l3574_357431


namespace surface_area_after_corner_removal_l3574_357408

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the problem setup -/
structure ProblemSetup where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions
  numCorners : ℕ

/-- Theorem stating that the surface area remains unchanged after removing corner cubes -/
theorem surface_area_after_corner_removal (p : ProblemSetup) 
  (h1 : p.originalCube.length = 4)
  (h2 : p.originalCube.width = 4)
  (h3 : p.originalCube.height = 4)
  (h4 : p.cornerCube.length = 2)
  (h5 : p.cornerCube.width = 2)
  (h6 : p.cornerCube.height = 2)
  (h7 : p.numCorners = 8) :
  surfaceArea p.originalCube = 96 := by
  sorry

#eval surfaceArea { length := 4, width := 4, height := 4 }

end surface_area_after_corner_removal_l3574_357408


namespace unique_solution_iff_a_eq_plus_minus_two_l3574_357471

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + y^2 + z^2 + 4*y = 0
def equation2 (a x y z : ℝ) : Prop := x + a*y + a*z - a = 0

-- Define what it means for the system to have a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y z : ℝ), equation1 x y z ∧ equation2 a x y z

-- State the theorem
theorem unique_solution_iff_a_eq_plus_minus_two :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 2 ∨ a = -2) := by sorry

end unique_solution_iff_a_eq_plus_minus_two_l3574_357471


namespace blue_marbles_most_numerous_l3574_357461

/-- Given a set of marbles with specific conditions, prove that blue marbles are the most numerous -/
theorem blue_marbles_most_numerous (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - red - blue) :
  blue > red ∧ blue > yellow := by
  sorry

end blue_marbles_most_numerous_l3574_357461


namespace arrangement_count_proof_l3574_357417

/-- The number of ways to arrange 4 distinct digits in a 2 × 3 grid with 2 empty cells -/
def arrangement_count : ℕ := 360

/-- The size of the grid -/
def grid_size : ℕ × ℕ := (2, 3)

/-- The number of available digits -/
def digit_count : ℕ := 4

/-- The number of empty cells -/
def empty_cell_count : ℕ := 2

/-- The total number of cells in the grid -/
def total_cells : ℕ := grid_size.1 * grid_size.2

theorem arrangement_count_proof :
  arrangement_count = (Nat.choose total_cells empty_cell_count) * (Nat.factorial digit_count) :=
sorry

end arrangement_count_proof_l3574_357417


namespace fraction_simplification_l3574_357476

theorem fraction_simplification : 
  (20 : ℚ) / 19 * 15 / 28 * 76 / 45 = 95 / 84 := by
  sorry

end fraction_simplification_l3574_357476


namespace office_population_l3574_357445

theorem office_population (men women : ℕ) : 
  men = women →
  6 = women / 5 →
  men + women = 60 := by
sorry

end office_population_l3574_357445


namespace consecutive_integer_fraction_minimum_l3574_357451

theorem consecutive_integer_fraction_minimum (a b : ℤ) (h1 : a = b + 1) (h2 : a > b) :
  ∀ ε > 0, ∃ a b : ℤ, a = b + 1 ∧ a > b ∧ (a + b : ℚ) / (a - b) + (a - b : ℚ) / (a + b) < 2 + ε ∧
  ∀ a' b' : ℤ, a' = b' + 1 → a' > b' → 2 ≤ (a' + b' : ℚ) / (a' - b') + (a' - b' : ℚ) / (a' + b') :=
sorry

end consecutive_integer_fraction_minimum_l3574_357451


namespace route_length_is_200_l3574_357437

/-- Two trains traveling on a route --/
structure TrainRoute where
  length : ℝ
  train_a_time : ℝ
  train_b_time : ℝ
  meeting_distance : ℝ

/-- The specific train route from the problem --/
def problem_route : TrainRoute where
  length := 200
  train_a_time := 10
  train_b_time := 6
  meeting_distance := 75

/-- Theorem stating that the given conditions imply the route length is 200 miles --/
theorem route_length_is_200 (route : TrainRoute) :
  route.train_a_time = 10 ∧
  route.train_b_time = 6 ∧
  route.meeting_distance = 75 →
  route.length = 200 := by
  sorry

#check route_length_is_200

end route_length_is_200_l3574_357437


namespace lcm_equation_solution_l3574_357453

theorem lcm_equation_solution :
  ∀ x y : ℕ, 
    x > 0 ∧ y > 0 → 
    Nat.lcm x y = 1 + 2*x + 3*y ↔ (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) := by
  sorry

end lcm_equation_solution_l3574_357453


namespace triangle_max_area_l3574_357439

/-- Given a triangle ABC where the sum of two sides is 4 and one angle is 30°, 
    the maximum area of the triangle is 1 -/
theorem triangle_max_area (a b : ℝ) (C : ℝ) (h1 : a + b = 4) (h2 : C = 30 * π / 180) :
  ∀ S : ℝ, S = 1/2 * a * b * Real.sin C → S ≤ 1 :=
by sorry

end triangle_max_area_l3574_357439


namespace circular_track_circumference_l3574_357483

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference (speed1 speed2 : ℝ) (time : ℝ) (h1 : speed1 = 7)
    (h2 : speed2 = 8) (h3 : time = 42) :
    speed1 * time + speed2 * time = 630 := by
  sorry

end circular_track_circumference_l3574_357483


namespace hyperbola_a_value_l3574_357400

/-- A hyperbola with equation x²/(a-3) + y²/(2-a) = 1, foci on the y-axis, and focal distance 4 -/
structure Hyperbola where
  a : ℝ
  equation : ∀ x y : ℝ, x^2 / (a - 3) + y^2 / (2 - a) = 1
  foci_on_y_axis : True  -- This is a placeholder for the foci condition
  focal_distance : ℝ
  focal_distance_value : focal_distance = 4

/-- The value of 'a' for the given hyperbola is 1/2 -/
theorem hyperbola_a_value (h : Hyperbola) : h.a = 1/2 :=
sorry

end hyperbola_a_value_l3574_357400


namespace tyler_remaining_money_l3574_357487

/-- Calculates the remaining money after Tyler's purchases -/
def remaining_money (initial_amount : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
                    (eraser_count : ℕ) (eraser_price : ℕ) : ℕ :=
  initial_amount - (scissors_count * scissors_price + eraser_count * eraser_price)

/-- Theorem stating that Tyler will have $20 remaining after his purchases -/
theorem tyler_remaining_money : 
  remaining_money 100 8 5 10 4 = 20 := by
  sorry

end tyler_remaining_money_l3574_357487


namespace matthews_walking_rate_l3574_357486

/-- Proves that Matthew's walking rate is 3 km per hour given the problem conditions -/
theorem matthews_walking_rate (total_distance : ℝ) (johnny_start_delay : ℝ) (johnny_rate : ℝ) (johnny_distance : ℝ) :
  total_distance = 45 →
  johnny_start_delay = 1 →
  johnny_rate = 4 →
  johnny_distance = 24 →
  ∃ (matthews_rate : ℝ),
    matthews_rate = 3 ∧
    matthews_rate * (johnny_distance / johnny_rate + johnny_start_delay) = total_distance - johnny_distance :=
by sorry

end matthews_walking_rate_l3574_357486


namespace three_cell_shapes_count_l3574_357402

/-- Represents the number of cells in a shape -/
inductive ShapeSize
| Three : ShapeSize
| Four : ShapeSize

/-- Represents a configuration of shapes -/
structure Configuration :=
  (threeCell : ℕ)
  (fourCell : ℕ)

/-- Checks if a configuration is valid -/
def isValidConfiguration (config : Configuration) : Prop :=
  3 * config.threeCell + 4 * config.fourCell = 22

/-- Checks if a configuration matches the desired solution -/
def isDesiredSolution (config : Configuration) : Prop :=
  config.threeCell = 6 ∧ config.fourCell = 1

/-- The main theorem to prove -/
theorem three_cell_shapes_count :
  ∃ (config : Configuration),
    isValidConfiguration config ∧ isDesiredSolution config :=
sorry

end three_cell_shapes_count_l3574_357402


namespace kendra_change_l3574_357442

/-- Calculates the change received after a purchase -/
def calculate_change (toy_price hat_price : ℕ) (num_toys num_hats : ℕ) (paid : ℕ) : ℕ :=
  paid - (toy_price * num_toys + hat_price * num_hats)

/-- Proves that Kendra received $30 in change -/
theorem kendra_change : calculate_change 20 10 2 3 100 = 30 := by
  sorry

end kendra_change_l3574_357442


namespace shaded_square_area_ratio_l3574_357419

theorem shaded_square_area_ratio (n : ℕ) (shaded_area : ℕ) : 
  n = 5 → shaded_area = 5 → (shaded_area : ℚ) / (n^2 : ℚ) = 1/5 := by
  sorry

end shaded_square_area_ratio_l3574_357419


namespace f_positive_implies_a_range_a_range_implies_f_positive_f_positive_iff_a_range_l3574_357499

/-- The function f(x) = ax^2 - 2x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

/-- Theorem: If f(x) > 0 for all x in (1, 4), then a > 1/2 -/
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f a x > 0) → a > 1/2 := by
  sorry

/-- Theorem: If a > 1/2, then f(x) > 0 for all x in (1, 4) -/
theorem a_range_implies_f_positive (a : ℝ) :
  a > 1/2 → (∀ x, 1 < x ∧ x < 4 → f a x > 0) := by
  sorry

/-- The main theorem combining both directions -/
theorem f_positive_iff_a_range (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f a x > 0) ↔ a > 1/2 := by
  sorry

end f_positive_implies_a_range_a_range_implies_f_positive_f_positive_iff_a_range_l3574_357499


namespace unique_base_sum_l3574_357405

def sum_single_digits (b : ℕ) : ℕ := 
  if b % 2 = 0 then
    b * (b - 1) / 2
  else
    (b^2 - 1) / 2

theorem unique_base_sum : 
  ∃! b : ℕ, b > 0 ∧ sum_single_digits b = 2 * b + 8 :=
sorry

end unique_base_sum_l3574_357405


namespace equal_area_dividing_line_slope_l3574_357458

/-- Given two circles with radius 4 units centered at (0, 20) and (7, 13),
    and a line passing through (4, 0) that divides the total area of both circles equally,
    prove that the absolute value of the slope of this line is 33/15 -/
theorem equal_area_dividing_line_slope (r : ℝ) (c₁ c₂ : ℝ × ℝ) (p : ℝ × ℝ) (m : ℝ) :
  r = 4 →
  c₁ = (0, 20) →
  c₂ = (7, 13) →
  p = (4, 0) →
  (∀ x y, y = m * (x - p.1) + p.2) →
  (∀ x y, (x - c₁.1)^2 + (y - c₁.2)^2 = r^2 → 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1) = 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1)) →
  (∀ x y, (x - c₂.1)^2 + (y - c₂.2)^2 = r^2 → 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1) = 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1)) →
  abs m = 33 / 15 := by
sorry


end equal_area_dividing_line_slope_l3574_357458


namespace correct_guess_probability_l3574_357438

/-- The number of possible choices for the last digit -/
def last_digit_choices : ℕ := 4

/-- The number of possible choices for the second-to-last digit -/
def second_last_digit_choices : ℕ := 3

/-- The probability of correctly guessing the two-digit code -/
def guess_probability : ℚ := 1 / (last_digit_choices * second_last_digit_choices)

theorem correct_guess_probability : guess_probability = 1 / 12 := by
  sorry

end correct_guess_probability_l3574_357438


namespace sunzi_wood_measurement_problem_l3574_357425

theorem sunzi_wood_measurement_problem (x y : ℝ) : 
  (x - y = 4.5 ∧ (x / 2) + 1 = y) ↔ (x - y = 4.5 ∧ y - x / 2 = 1) :=
by sorry

end sunzi_wood_measurement_problem_l3574_357425


namespace people_arrangement_l3574_357450

/-- Given a total of 1600 people and columns of 85 people each, prove:
    1. The number of complete columns
    2. The number of people in the incomplete column
    3. The total number of rows
    4. The row in which the last person stands -/
theorem people_arrangement (total_people : ℕ) (people_per_column : ℕ) 
    (h1 : total_people = 1600)
    (h2 : people_per_column = 85) :
    let complete_columns := total_people / people_per_column
    let remaining_people := total_people % people_per_column
    (complete_columns = 18) ∧ 
    (remaining_people = 70) ∧
    (remaining_people = 70) ∧
    (remaining_people = 70) := by
  sorry

end people_arrangement_l3574_357450


namespace distance_minus_nine_to_nine_l3574_357410

-- Define the distance function for points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem distance_minus_nine_to_nine : distance (-9) 9 = 18 := by
  sorry

end distance_minus_nine_to_nine_l3574_357410


namespace arithmetic_sequence_common_difference_l3574_357443

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (a5_eq_6 : a 5 = 6)
  (a3_eq_2 : a 3 = 2) :
  ∀ n, a (n + 1) - a n = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l3574_357443


namespace negative_integer_equation_solution_l3574_357479

theorem negative_integer_equation_solution :
  ∃ (M : ℤ), (M < 0) ∧ (2 * M^2 + M = 12) → M = -4 := by
  sorry

end negative_integer_equation_solution_l3574_357479


namespace units_digit_of_17_pow_2045_l3574_357435

theorem units_digit_of_17_pow_2045 : (17^2045 : ℕ) % 10 = 7 := by sorry

end units_digit_of_17_pow_2045_l3574_357435


namespace line_slope_is_one_l3574_357436

theorem line_slope_is_one : 
  let line_eq := fun (x y : ℝ) => x - y + 1 = 0
  ∃ m : ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧ x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) ∧ m = 1 :=
by sorry

end line_slope_is_one_l3574_357436


namespace principal_calculation_l3574_357456

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time

/-- Problem statement -/
theorem principal_calculation (interest rate time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 0.13)
  (h3 : time = 5)
  : ∃ (principal : ℝ), simple_interest principal rate time = interest ∧ principal = 6180 := by
  sorry

end principal_calculation_l3574_357456


namespace only_setB_forms_triangle_l3574_357441

/-- Represents a set of three line segments --/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a set of line segments can form a triangle --/
def canFormTriangle (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The given sets of line segments --/
def setA : LineSegmentSet := ⟨1, 2, 4⟩
def setB : LineSegmentSet := ⟨4, 6, 8⟩
def setC : LineSegmentSet := ⟨5, 6, 12⟩
def setD : LineSegmentSet := ⟨2, 3, 5⟩

/-- Theorem: Only set B can form a triangle --/
theorem only_setB_forms_triangle :
  ¬(canFormTriangle setA) ∧
  canFormTriangle setB ∧
  ¬(canFormTriangle setC) ∧
  ¬(canFormTriangle setD) :=
by sorry

end only_setB_forms_triangle_l3574_357441


namespace ac_squared_gt_bc_squared_l3574_357466

theorem ac_squared_gt_bc_squared (a b c : ℝ) : a > b → a * c^2 > b * c^2 := by
  sorry

end ac_squared_gt_bc_squared_l3574_357466


namespace tim_nickels_l3574_357449

/-- The number of nickels Tim got for shining shoes -/
def nickels : ℕ := sorry

/-- The number of dimes Tim got for shining shoes -/
def dimes_shining : ℕ := 13

/-- The number of dimes Tim found in his tip jar -/
def dimes_tip : ℕ := 7

/-- The number of half-dollars Tim found in his tip jar -/
def half_dollars : ℕ := 9

/-- The total amount Tim got in dollars -/
def total_amount : ℚ := 665 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 50 / 100

theorem tim_nickels :
  nickels * nickel_value + 
  dimes_shining * dime_value + 
  dimes_tip * dime_value + 
  half_dollars * half_dollar_value = total_amount ∧
  nickels = 3 := by sorry

end tim_nickels_l3574_357449


namespace cube_sum_of_conjugate_fractions_l3574_357496

theorem cube_sum_of_conjugate_fractions :
  let x := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
  let y := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)
  x^3 + y^3 = 2702 := by
sorry

end cube_sum_of_conjugate_fractions_l3574_357496


namespace root_power_sums_equal_l3574_357434

-- Define the polynomial
def p (x : ℂ) : ℂ := x^3 + 2*x^2 + 3*x + 4

-- Define the sum of nth powers of roots
def S (n : ℕ) : ℂ := sorry

theorem root_power_sums_equal :
  S 1 = -2 ∧ S 2 = -2 ∧ S 3 = -2 := by sorry

end root_power_sums_equal_l3574_357434


namespace book_cost_l3574_357459

theorem book_cost : ∃ (x : ℝ), x = 1 + (1/2) * x ∧ x = 2 := by sorry

end book_cost_l3574_357459


namespace at_least_one_positive_l3574_357454

theorem at_least_one_positive (x y z : ℝ) : 
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/4
  let c := z^2 - 2*x + π/4
  max a (max b c) > 0 := by
sorry

end at_least_one_positive_l3574_357454


namespace newspaper_conference_attendees_l3574_357465

/-- The minimum number of people attending the newspaper conference -/
def min_attendees : ℕ := 126

/-- The number of writers at the conference -/
def writers : ℕ := 35

/-- The minimum number of editors at the conference -/
def min_editors : ℕ := 39

/-- The maximum number of people who are both writers and editors -/
def max_both : ℕ := 26

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * max_both

theorem newspaper_conference_attendees :
  ∀ N : ℕ,
  (N ≥ writers + min_editors - max_both + neither) →
  (N ≥ min_attendees) :=
by sorry

end newspaper_conference_attendees_l3574_357465


namespace inverse_mod_53_l3574_357478

theorem inverse_mod_53 (h : (21⁻¹ : ZMod 53) = 17) : (32⁻¹ : ZMod 53) = 36 := by
  sorry

end inverse_mod_53_l3574_357478


namespace min_distance_sum_l3574_357409

open Complex

theorem min_distance_sum (z₁ z₂ : ℂ) (h₁ : z₁ = -Real.sqrt 3 - I) (h₂ : z₂ = 3 + Real.sqrt 3 * I) :
  (∃ (θ : ℝ), ∀ (z : ℂ), z = (2 + Real.cos θ) + I * Real.sin θ →
    ∀ (w : ℂ), abs (w - z₁) + abs (w - z₂) ≥ abs (z - z₁) + abs (z - z₂)) ∧
  (∃ (z : ℂ), abs (z - z₁) + abs (z - z₂) = 2 + 2 * Real.sqrt 3) :=
by sorry

end min_distance_sum_l3574_357409


namespace square_to_rectangle_area_ratio_l3574_357452

/-- A rectangle with a square inside it -/
structure RectangleWithSquare where
  square_side : ℝ
  rect_width : ℝ
  rect_length : ℝ
  width_to_side_ratio : rect_width = 3 * square_side
  length_to_width_ratio : rect_length = 2 * rect_width

/-- The theorem stating that the area of the square is 1/18 of the area of the rectangle -/
theorem square_to_rectangle_area_ratio (r : RectangleWithSquare) :
  (r.square_side ^ 2) / (r.rect_width * r.rect_length) = 1 / 18 := by
  sorry

end square_to_rectangle_area_ratio_l3574_357452


namespace square_difference_dollar_l3574_357404

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem square_difference_dollar (x y : ℝ) :
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2*x^2*y^2 + y^4) := by
  sorry

end square_difference_dollar_l3574_357404


namespace smallest_mu_inequality_l3574_357469

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  ∃ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) ∧
  (∀ μ' : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ'*b*c + 2*c*d) → μ' ≥ μ) ∧
  μ = 2 :=
sorry

end smallest_mu_inequality_l3574_357469


namespace square_sum_reciprocal_l3574_357406

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l3574_357406


namespace unique_solution_quadratic_inequality_l3574_357429

theorem unique_solution_quadratic_inequality (p : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 + p*x + 5 ∧ x^2 + p*x + 5 ≤ 1) → (p = 4 ∨ p = -4) :=
by sorry

end unique_solution_quadratic_inequality_l3574_357429


namespace max_initial_happy_citizens_l3574_357497

/-- Represents the state of happiness for a citizen --/
inductive MoodState
| Happy
| Unhappy

/-- Represents a citizen in Happy City --/
structure Citizen where
  id : Nat
  mood : MoodState

/-- Represents the state of Happy City --/
structure HappyCity where
  citizens : List Citizen
  day : Nat

/-- Function to simulate a day of smiling in Happy City --/
def smileDay (city : HappyCity) : HappyCity :=
  sorry

/-- Function to count happy citizens --/
def countHappy (city : HappyCity) : Nat :=
  sorry

/-- Theorem stating the maximum initial number of happy citizens --/
theorem max_initial_happy_citizens :
  ∀ (initialCity : HappyCity),
    initialCity.citizens.length = 2014 →
    (∃ (finalCity : HappyCity),
      finalCity = (smileDay ∘ smileDay ∘ smileDay ∘ smileDay) initialCity ∧
      countHappy finalCity = 2000) →
    countHappy initialCity ≤ 32 :=
  sorry

end max_initial_happy_citizens_l3574_357497


namespace calculate_expression_l3574_357430

theorem calculate_expression : 
  let tan_60 : ℝ := Real.sqrt 3
  |2 - tan_60| - 1 + 4 + Real.sqrt 3 = 5 := by sorry

end calculate_expression_l3574_357430


namespace max_different_ages_l3574_357498

/-- The maximum number of different integer ages within one standard deviation of the average -/
theorem max_different_ages (average_age : ℝ) (std_dev : ℝ) : average_age = 31 → std_dev = 9 →
  (Set.Icc (average_age - std_dev) (average_age + std_dev) ∩ Set.range (Int.cast : ℤ → ℝ)).ncard = 19 := by
  sorry

end max_different_ages_l3574_357498


namespace min_sum_position_l3574_357468

/-- An arithmetic sequence {a_n} with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The theorem stating when S_n reaches its minimum value -/
theorem min_sum_position (seq : ArithmeticSequence) 
  (h1 : seq.a 2 = -2)
  (h2 : seq.S 4 = -4) :
  ∃ n : ℕ, (n = 2 ∨ n = 3) ∧ 
    (∀ m : ℕ, m ≥ 1 → seq.S n ≤ seq.S m) :=
sorry

end min_sum_position_l3574_357468


namespace binomial_coefficient_16_15_l3574_357414

theorem binomial_coefficient_16_15 : Nat.choose 16 15 = 16 := by
  sorry

end binomial_coefficient_16_15_l3574_357414


namespace ellipse_dot_product_bounds_l3574_357440

/-- Given an ellipse with specific properties, prove that the dot product of vectors AP and FP is bounded. -/
theorem ellipse_dot_product_bounds (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0)
  (h_top_focus : 2 = Real.sqrt (a^2 - b^2))
  (h_eccentricity : (Real.sqrt (a^2 - b^2)) / a = 1/2) :
  ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 →
  0 ≤ (x + 2) * (x + 1) + y^2 ∧ (x + 2) * (x + 1) + y^2 ≤ 12 := by
  sorry

end ellipse_dot_product_bounds_l3574_357440


namespace slope_of_line_l3574_357493

theorem slope_of_line (x y : ℝ) : 3 * y + 2 * x = 12 → (y - 4) / x = -2 / 3 := by
  sorry

end slope_of_line_l3574_357493


namespace inverse_variation_sqrt_l3574_357472

/-- Given that z varies inversely as √w, prove that w = 64 when z = 2, 
    given that z = 8 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w : ℝ, w > 0 → z * Real.sqrt w = k) 
    (h1 : 8 * Real.sqrt 4 = z * Real.sqrt w) : 
    z = 2 → w = 64 := by
  sorry

end inverse_variation_sqrt_l3574_357472


namespace sum_seven_multiples_of_12_l3574_357428

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The sum of the first seven multiples of 12 -/
theorem sum_seven_multiples_of_12 :
  arithmetic_sum 12 12 7 = 336 := by
  sorry

end sum_seven_multiples_of_12_l3574_357428


namespace no_integer_solutions_l3574_357492

theorem no_integer_solutions : 
  ¬ ∃ (x : ℤ), (x^2 - 3*x + 2)^2 - 3*(x^2 - 3*x) - 4 = 0 := by
  sorry

end no_integer_solutions_l3574_357492


namespace complex_equation_solution_l3574_357481

theorem complex_equation_solution (z : ℂ) : (3 - 4*I)*z = 25 → z = 3 + 4*I := by
  sorry

end complex_equation_solution_l3574_357481


namespace imaginary_part_of_complex_expression_l3574_357490

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (1 + I) / (1 - I) + (1 - I)
  Complex.im z = 0 := by
  sorry

end imaginary_part_of_complex_expression_l3574_357490


namespace six_lines_intersection_possibilities_l3574_357467

/-- Represents a line in a plane -/
structure Line

/-- Represents an intersection point of two lines -/
structure IntersectionPoint

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset Line
  intersections : Finset IntersectionPoint
  no_triple_intersections : ∀ p : IntersectionPoint, p ∈ intersections → 
    (∃! l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ p ∈ intersections)

theorem six_lines_intersection_possibilities 
  (config : LineConfiguration) 
  (h_six_lines : config.lines.card = 6) :
  (∃ config' : LineConfiguration, config'.lines = config.lines ∧ config'.intersections.card = 12) ∧
  ¬(∃ config' : LineConfiguration, config'.lines = config.lines ∧ config'.intersections.card = 16) :=
sorry

end six_lines_intersection_possibilities_l3574_357467


namespace remainder_after_adding_2030_l3574_357411

theorem remainder_after_adding_2030 (m : ℤ) (h : m % 7 = 2) : (m + 2030) % 7 = 2 := by
  sorry

end remainder_after_adding_2030_l3574_357411


namespace initial_scissors_l3574_357473

theorem initial_scissors (added : ℕ) (total : ℕ) (h1 : added = 22) (h2 : total = 76) :
  total - added = 54 := by
  sorry

end initial_scissors_l3574_357473


namespace mod_seven_equivalence_l3574_357463

theorem mod_seven_equivalence : (41^1723 - 18^1723) % 7 = 2 := by
  sorry

end mod_seven_equivalence_l3574_357463


namespace waiter_tips_fraction_l3574_357448

theorem waiter_tips_fraction (salary tips : ℝ) 
  (h : tips = 0.625 * (salary + tips)) : 
  tips / salary = 5 / 3 := by
sorry

end waiter_tips_fraction_l3574_357448


namespace goat_max_distance_l3574_357420

theorem goat_max_distance (center : ℝ × ℝ) (radius : ℝ) :
  center = (6, 8) →
  radius = 15 →
  let dist_to_center := Real.sqrt ((center.1 - 0)^2 + (center.2 - 0)^2)
  let max_distance := dist_to_center + radius
  max_distance = 25 := by sorry

end goat_max_distance_l3574_357420


namespace exists_bound_for_digit_sum_of_factorial_l3574_357426

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The statement to be proved -/
theorem exists_bound_for_digit_sum_of_factorial :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (n.factorial) ≥ 10^100 := by
  sorry

end exists_bound_for_digit_sum_of_factorial_l3574_357426


namespace triangle_angle_property_l3574_357488

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is inside a triangle -/
def isInsideTriangle (P : Point3D) (T : Triangle3D) : Prop := sorry

/-- Check if a point is outside the plane of a triangle -/
def isOutsidePlane (D : Point3D) (T : Triangle3D) : Prop := sorry

/-- Angle between three points in 3D space -/
def angle (A B C : Point3D) : ℝ := sorry

/-- An angle is acute if it's less than 90 degrees -/
def isAcute (θ : ℝ) : Prop := θ < Real.pi / 2

/-- An angle is obtuse if it's greater than 90 degrees -/
def isObtuse (θ : ℝ) : Prop := θ > Real.pi / 2

theorem triangle_angle_property (T : Triangle3D) (P D : Point3D) :
  isInsideTriangle P T →
  isOutsidePlane D T →
  (isAcute (angle T.A P D) ∨ isAcute (angle T.B P D) ∨ isAcute (angle T.C P D)) →
  (isObtuse (angle T.A P D) ∨ isObtuse (angle T.B P D) ∨ isObtuse (angle T.C P D)) := by
  sorry

end triangle_angle_property_l3574_357488


namespace fourth_term_of_geometric_sequence_l3574_357477

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 1 / a 0

theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_first : a 0 = 2^(1/2 : ℝ))
  (h_second : a 1 = 2^(1/4 : ℝ))
  (h_third : a 2 = 2^(1/8 : ℝ)) :
  a 3 = 1 := by
  sorry

end fourth_term_of_geometric_sequence_l3574_357477


namespace connect_four_games_l3574_357403

/-- 
Given:
- The ratio of games Kaleb won to games he lost is 3:2
- Kaleb won 18 games

Prove: The total number of games played is 30
-/
theorem connect_four_games (games_won : ℕ) (games_lost : ℕ) : 
  (games_won : ℚ) / games_lost = 3 / 2 → 
  games_won = 18 → 
  games_won + games_lost = 30 := by
sorry

end connect_four_games_l3574_357403


namespace game_result_l3574_357485

def f (n : ℕ) : ℕ :=
  if n ^ 2 = n then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 1
  else 0

def allie_rolls : List ℕ := [3, 4, 6, 1]
def betty_rolls : List ℕ := [4, 2, 5, 1]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : total_points allie_rolls * total_points betty_rolls = 117 := by
  sorry

end game_result_l3574_357485


namespace marble_probability_l3574_357423

/-- The number of green marbles -/
def green_marbles : ℕ := 7

/-- The number of purple marbles -/
def purple_marbles : ℕ := 5

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 8

/-- The number of successful outcomes (choosing green marbles) -/
def num_success : ℕ := 3

/-- The probability of choosing a green marble in a single trial -/
def p : ℚ := green_marbles / total_marbles

/-- The probability of choosing a purple marble in a single trial -/
def q : ℚ := purple_marbles / total_marbles

/-- The binomial probability of choosing exactly 3 green marbles in 8 trials -/
def binomial_prob : ℚ := (Nat.choose num_trials num_success : ℚ) * p ^ num_success * q ^ (num_trials - num_success)

theorem marble_probability : binomial_prob = 9378906 / 67184015 := by
  sorry

end marble_probability_l3574_357423


namespace product_arrangement_count_l3574_357480

/-- The number of products to arrange -/
def n : ℕ := 5

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of arrangements with A and B together -/
def arrangementsWithABTogether : ℕ := 2 * factorial (n - 1)

/-- The number of arrangements with C and D together -/
def arrangementsWithCDTogether : ℕ := 2 * 2 * factorial (n - 2)

/-- The total number of valid arrangements -/
def validArrangements : ℕ := arrangementsWithABTogether - arrangementsWithCDTogether

theorem product_arrangement_count : validArrangements = 24 := by
  sorry

end product_arrangement_count_l3574_357480


namespace chef_cherries_l3574_357482

theorem chef_cherries (used_for_pie : ℕ) (left_over : ℕ) (initial : ℕ) : 
  used_for_pie = 60 → left_over = 17 → initial = used_for_pie + left_over → initial = 77 :=
by sorry

end chef_cherries_l3574_357482


namespace prob_odd_die_roll_l3574_357432

/-- The number of possible outcomes when rolling a die -/
def total_outcomes : ℕ := 6

/-- The number of favorable outcomes (odd numbers) when rolling a die -/
def favorable_outcomes : ℕ := 3

/-- The probability of an event in a finite sample space -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem: The probability of rolling an odd number on a standard six-sided die is 1/2 -/
theorem prob_odd_die_roll : probability favorable_outcomes total_outcomes = 1 / 2 := by
  sorry

end prob_odd_die_roll_l3574_357432


namespace circle_equation_with_diameter_l3574_357415

/-- Given points A and B, prove the equation of the circle with AB as diameter -/
theorem circle_equation_with_diameter (A B : ℝ × ℝ) (h : A = (-4, 0) ∧ B = (0, 2)) :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 5 ↔ 
  (x - A.1)^2 + (y - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4 ∧
  (x - B.1)^2 + (y - B.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4 := by
  sorry

end circle_equation_with_diameter_l3574_357415


namespace count_non_negative_rationals_l3574_357455

def rational_list : List ℚ := [-8, 0, -1.04, -(-3), 1/3, -|-2|]

theorem count_non_negative_rationals :
  (rational_list.filter (λ x => x ≥ 0)).length = 3 := by sorry

end count_non_negative_rationals_l3574_357455


namespace modified_cube_edge_count_l3574_357457

/-- Represents a cube with smaller cubes removed from alternate corners -/
structure ModifiedCube where
  side_length : ℕ
  removed_cube_side_length : ℕ
  removed_corners : ℕ

/-- Calculates the number of edges in a modified cube -/
def edge_count (c : ModifiedCube) : ℕ :=
  12 + 3 * c.removed_corners

/-- Theorem stating that a cube of side length 4 with unit cubes removed from 4 corners has 24 edges -/
theorem modified_cube_edge_count :
  ∀ (c : ModifiedCube), 
    c.side_length = 4 ∧ 
    c.removed_cube_side_length = 1 ∧ 
    c.removed_corners = 4 → 
    edge_count c = 24 := by
  sorry

#check modified_cube_edge_count

end modified_cube_edge_count_l3574_357457


namespace complex_base_representation_exists_unique_integer_representable_in_base_neg4_plus_i_l3574_357421

/-- Representation of a complex number in a complex base -n+i --/
structure ComplexBaseRepresentation (n : ℕ+) where
  coeffs : Fin 4 → Fin 257
  nonzero_lead : coeffs 3 ≠ 0

/-- The value represented by a ComplexBaseRepresentation --/
def value (n : ℕ+) (rep : ComplexBaseRepresentation n) : ℂ :=
  (rep.coeffs 3 : ℂ) * (-n + Complex.I)^3 +
  (rep.coeffs 2 : ℂ) * (-n + Complex.I)^2 +
  (rep.coeffs 1 : ℂ) * (-n + Complex.I) +
  (rep.coeffs 0 : ℂ)

/-- Theorem stating the existence and uniqueness of the representation --/
theorem complex_base_representation_exists_unique (n : ℕ+) (z : ℂ) 
  (h : ∃ (r s : ℤ), z = r + s * Complex.I) :
  ∃! (rep : ComplexBaseRepresentation n), value n rep = z :=
sorry

/-- Theorem stating that for base -4+i, there exist integers representable in four digits --/
theorem integer_representable_in_base_neg4_plus_i :
  ∃ (k : ℤ) (rep : ComplexBaseRepresentation 4),
    value 4 rep = k ∧ k = (value 4 rep).re :=
sorry

end complex_base_representation_exists_unique_integer_representable_in_base_neg4_plus_i_l3574_357421


namespace ratio_problem_l3574_357474

theorem ratio_problem (a b c : ℝ) 
  (hab : a / b = 11 / 3) 
  (hac : a / c = 11 / 15) : 
  b / c = 1 / 5 := by sorry

end ratio_problem_l3574_357474


namespace original_plus_increase_equals_current_l3574_357470

/-- The number of bacteria originally in the petri dish -/
def original_bacteria : ℕ := 600

/-- The current number of bacteria in the petri dish -/
def current_bacteria : ℕ := 8917

/-- The increase in the number of bacteria -/
def bacteria_increase : ℕ := 8317

/-- Theorem stating that the original number of bacteria plus the increase
    equals the current number of bacteria -/
theorem original_plus_increase_equals_current :
  original_bacteria + bacteria_increase = current_bacteria := by
  sorry

end original_plus_increase_equals_current_l3574_357470


namespace high_low_game_combinations_l3574_357491

/-- Represents the types of cards in the high-low game -/
inductive CardType
| High
| Low

/-- The high-low card game -/
structure HighLowGame where
  totalCards : Nat
  highCards : Nat
  lowCards : Nat
  highCardPoints : Nat
  lowCardPoints : Nat
  targetPoints : Nat

/-- Calculates the total points for a given combination of high and low cards -/
def calculatePoints (game : HighLowGame) (highCount : Nat) (lowCount : Nat) : Nat :=
  highCount * game.highCardPoints + lowCount * game.lowCardPoints

/-- Checks if a given combination of high and low cards achieves the target points -/
def isValidCombination (game : HighLowGame) (highCount : Nat) (lowCount : Nat) : Prop :=
  calculatePoints game highCount lowCount = game.targetPoints

/-- Theorem: In the high-low game, to earn exactly 5 points, 
    the number of low cards drawn must be either 1, 3, or 5 -/
theorem high_low_game_combinations (game : HighLowGame) 
    (h1 : game.totalCards = 52)
    (h2 : game.highCards = game.lowCards)
    (h3 : game.highCards + game.lowCards = game.totalCards)
    (h4 : game.highCardPoints = 2)
    (h5 : game.lowCardPoints = 1)
    (h6 : game.targetPoints = 5) :
    ∀ (highCount lowCount : Nat), 
      isValidCombination game highCount lowCount → 
      lowCount = 1 ∨ lowCount = 3 ∨ lowCount = 5 :=
  sorry


end high_low_game_combinations_l3574_357491


namespace animal_arrangement_count_l3574_357494

def num_pigs : ℕ := 5
def num_rabbits : ℕ := 3
def num_dogs : ℕ := 2
def num_chickens : ℕ := 6

def total_animals : ℕ := num_pigs + num_rabbits + num_dogs + num_chickens

def num_animal_types : ℕ := 4

theorem animal_arrangement_count :
  (Nat.factorial num_animal_types) *
  (Nat.factorial num_pigs) *
  (Nat.factorial num_rabbits) *
  (Nat.factorial num_dogs) *
  (Nat.factorial num_chickens) = 12441600 :=
by sorry

end animal_arrangement_count_l3574_357494


namespace ratio_problem_l3574_357407

theorem ratio_problem (a b c d : ℝ) (h1 : b / a = 4) (h2 : d / c = 2) : (a + b) / (c + d) = 5 / 12 := by
  sorry

end ratio_problem_l3574_357407


namespace number_count_with_incorrect_average_l3574_357446

theorem number_count_with_incorrect_average (n : ℕ) : 
  (n : ℝ) * 40.2 - (n : ℝ) * 40.1 = 35 → n = 350 := by
  sorry

end number_count_with_incorrect_average_l3574_357446


namespace class_size_theorem_l3574_357475

theorem class_size_theorem :
  ∀ (m d : ℕ),
  (∃ (r : ℕ), r = 3 * m ∧ r = 5 * d) →
  30 < m + d →
  m + d < 40 →
  m + d = 32 :=
by
  sorry

end class_size_theorem_l3574_357475


namespace diplomat_languages_l3574_357444

theorem diplomat_languages (total : ℕ) (french : ℕ) (not_russian : ℕ) (both_percent : ℚ) 
  (h_total : total = 180)
  (h_french : french = 14)
  (h_not_russian : not_russian = 32)
  (h_both_percent : both_percent = 1/10) : 
  (total - (french + (total - not_russian) - (both_percent * total))) / total = 1/5 := by
  sorry

end diplomat_languages_l3574_357444


namespace volume_change_with_pressure_increase_l3574_357416

theorem volume_change_with_pressure_increase {P V P' V' : ℝ} (h1 : P > 0) (h2 : V > 0) :
  (P * V = P' * V') → -- inverse proportionality
  (P' = 1.2 * P) → -- 20% increase in pressure
  (V' = V * (5/6)) -- 16.67% decrease in volume
  := by sorry

end volume_change_with_pressure_increase_l3574_357416


namespace triangle_689_is_acute_l3574_357460

-- Define a triangle with sides in the ratio 6:8:9
def Triangle (t : ℝ) : Fin 3 → ℝ
| 0 => 6 * t
| 1 => 8 * t
| 2 => 9 * t

-- Define what it means for a triangle to be acute
def IsAcute (triangle : Fin 3 → ℝ) : Prop :=
  (triangle 0)^2 + (triangle 1)^2 > (triangle 2)^2 ∧
  (triangle 0)^2 + (triangle 2)^2 > (triangle 1)^2 ∧
  (triangle 1)^2 + (triangle 2)^2 > (triangle 0)^2

-- Theorem statement
theorem triangle_689_is_acute (t : ℝ) (h : t > 0) : IsAcute (Triangle t) := by
  sorry

end triangle_689_is_acute_l3574_357460


namespace inequality_proof_l3574_357413

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (a + b + c) / 2 := by
  sorry

end inequality_proof_l3574_357413


namespace max_a_value_l3574_357427

theorem max_a_value (a b : ℕ) (ha : 1 < a) (hb : a < b) :
  (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a| + |x - b|) →
  (∀ c : ℕ, (1 < c ∧ c < b ∧
    (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + c| + |x - b|)) →
    c ≤ a) →
  a = 4031 :=
by sorry

end max_a_value_l3574_357427


namespace fraction_sum_simplification_l3574_357418

theorem fraction_sum_simplification :
  3 / 462 + 17 / 42 = 95 / 231 := by sorry

end fraction_sum_simplification_l3574_357418


namespace recipe_flour_amount_l3574_357412

/-- The amount of flour Mary has already put in the recipe -/
def flour_already_added : ℕ := 3

/-- The amount of flour Mary still needs to add to the recipe -/
def flour_to_be_added : ℕ := 6

/-- The total amount of flour required for the recipe -/
def total_flour_required : ℕ := flour_already_added + flour_to_be_added

theorem recipe_flour_amount : total_flour_required = 9 := by
  sorry

end recipe_flour_amount_l3574_357412


namespace intersection_characterization_l3574_357495

-- Define set A
def A : Set ℝ := {x | Real.log (2 * x) < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 3 * x + 2}

-- Define the intersection of A and B
def A_intersect_B : Set (ℝ × ℝ) := {p | p.1 ∈ A ∧ p.2 ∈ B ∧ p.2 = 3 * p.1 + 2}

-- Theorem statement
theorem intersection_characterization : 
  A_intersect_B = {p : ℝ × ℝ | 2 < p.2 ∧ p.2 < 14} := by
  sorry

end intersection_characterization_l3574_357495


namespace min_score_to_tie_record_l3574_357424

/-- Proves that the minimum average score per player in the final round to tie the league record
    is 12.5833 points less than the current league record. -/
theorem min_score_to_tie_record (
  league_record : ℝ)
  (team_size : ℕ)
  (season_length : ℕ)
  (current_score : ℝ)
  (bonus_points : ℕ)
  (h1 : league_record = 287.5)
  (h2 : team_size = 6)
  (h3 : season_length = 12)
  (h4 : current_score = 19350.5)
  (h5 : bonus_points = 300)
  : ∃ (min_score : ℝ), 
    league_record - min_score = 12.5833 ∧ 
    min_score * team_size + current_score + bonus_points = league_record * (team_size * season_length) :=
by
  sorry

end min_score_to_tie_record_l3574_357424


namespace z_in_first_quadrant_l3574_357422

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_in_first_quadrant :
  (2 - Complex.I) * z = 1 + Complex.I →
  0 < z.re ∧ 0 < z.im :=
by sorry

end z_in_first_quadrant_l3574_357422


namespace smaller_square_area_half_larger_l3574_357484

/-- A circle with an inscribed square and a smaller square -/
structure SquaresInCircle where
  /-- The radius of the circle -/
  R : ℝ
  /-- The side length of the larger square -/
  a : ℝ
  /-- The side length of the smaller square -/
  b : ℝ
  /-- The larger square is inscribed in the circle -/
  h1 : R = a * Real.sqrt 2 / 2
  /-- The smaller square has one side coinciding with a side of the larger square -/
  h2 : b ≤ a
  /-- The smaller square has two vertices on the circle -/
  h3 : R^2 = (a/2 - b/2)^2 + b^2
  /-- The side length of the larger square is 4 units -/
  h4 : a = 4

/-- The area of the smaller square is half the area of the larger square -/
theorem smaller_square_area_half_larger (sq : SquaresInCircle) : 
  sq.b^2 = sq.a^2 / 2 := by
  sorry

end smaller_square_area_half_larger_l3574_357484
