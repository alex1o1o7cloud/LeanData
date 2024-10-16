import Mathlib

namespace NUMINAMATH_CALUDE_hannah_strawberries_l2335_233534

/-- The number of strawberries Hannah has at the end of April -/
def strawberries_at_end_of_april (daily_harvest : ℕ) (days_in_april : ℕ) (given_away : ℕ) (stolen : ℕ) : ℕ :=
  daily_harvest * days_in_april - (given_away + stolen)

theorem hannah_strawberries :
  strawberries_at_end_of_april 5 30 20 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_hannah_strawberries_l2335_233534


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2335_233508

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 80)
  (h_badminton : badminton = 48)
  (h_tennis : tennis = 46)
  (h_neither : neither = 7)
  : badminton + tennis - (total - neither) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2335_233508


namespace NUMINAMATH_CALUDE_commercial_break_duration_l2335_233591

theorem commercial_break_duration :
  let five_minute_commercials : ℕ := 3
  let two_minute_commercials : ℕ := 11
  let five_minute_duration : ℕ := 5
  let two_minute_duration : ℕ := 2
  (five_minute_commercials * five_minute_duration + two_minute_commercials * two_minute_duration : ℕ) = 37 :=
by sorry

end NUMINAMATH_CALUDE_commercial_break_duration_l2335_233591


namespace NUMINAMATH_CALUDE_notebooks_in_class2_l2335_233523

/-- The number of notebooks that do not belong to Class 1 -/
def not_class1 : ℕ := 162

/-- The number of notebooks that do not belong to Class 2 -/
def not_class2 : ℕ := 143

/-- The number of notebooks that belong to both Class 1 and Class 2 -/
def both_classes : ℕ := 87

/-- The total number of notebooks -/
def total_notebooks : ℕ := not_class1 + not_class2 - both_classes

theorem notebooks_in_class2 : 
  total_notebooks - (total_notebooks - not_class2) = 53 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_in_class2_l2335_233523


namespace NUMINAMATH_CALUDE_decimal_67_to_binary_l2335_233525

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : Nat) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2 = 1) :: acc)
    aux n []

-- Theorem statement
theorem decimal_67_to_binary :
  decimalToBinary 67 = [true, false, false, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_decimal_67_to_binary_l2335_233525


namespace NUMINAMATH_CALUDE_proportion_problem_l2335_233563

theorem proportion_problem (hours_per_day : ℕ) (h : hours_per_day = 24) :
  ∃ x : ℕ, (36 : ℚ) / 3 = x / (24 * hours_per_day) ∧ x = 6912 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l2335_233563


namespace NUMINAMATH_CALUDE_machines_in_first_group_l2335_233558

/-- The number of machines in the first group -/
def num_machines : ℕ := 8

/-- The time taken by the first group to complete a job lot (in hours) -/
def time_first_group : ℕ := 6

/-- The number of machines in the second group -/
def num_machines_second : ℕ := 12

/-- The time taken by the second group to complete a job lot (in hours) -/
def time_second_group : ℕ := 4

/-- The work rate of a single machine (job lots per hour) -/
def work_rate : ℚ := 1 / (num_machines_second * time_second_group)

theorem machines_in_first_group :
  num_machines * work_rate * time_first_group = 1 :=
sorry

end NUMINAMATH_CALUDE_machines_in_first_group_l2335_233558


namespace NUMINAMATH_CALUDE_intersection_M_N_l2335_233520

def M : Set ℝ := { x | -3 < x ∧ x < 1 }
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2335_233520


namespace NUMINAMATH_CALUDE_num_intersection_points_is_correct_l2335_233526

/-- The number of distinct intersection points of two equations -/
def num_intersection_points : ℕ := 3

/-- First equation -/
def equation1 (x y : ℝ) : Prop :=
  (x - y + 3) * (2*x + 3*y - 9) = 0

/-- Second equation -/
def equation2 (x y : ℝ) : Prop :=
  (2*x - y + 2) * (x + 3*y - 6) = 0

/-- A point satisfies both equations -/
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

theorem num_intersection_points_is_correct :
  ∃ (points : Finset (ℝ × ℝ)),
    points.card = num_intersection_points ∧
    (∀ p ∈ points, is_intersection_point p) ∧
    (∀ p : ℝ × ℝ, is_intersection_point p → p ∈ points) :=
  sorry

end NUMINAMATH_CALUDE_num_intersection_points_is_correct_l2335_233526


namespace NUMINAMATH_CALUDE_black_beads_count_l2335_233514

theorem black_beads_count (white_beads : ℕ) (total_pulled : ℕ) :
  white_beads = 51 →
  total_pulled = 32 →
  ∃ (black_beads : ℕ),
    (1 : ℚ) / 6 * black_beads + (1 : ℚ) / 3 * white_beads = total_pulled ∧
    black_beads = 90 :=
by sorry

end NUMINAMATH_CALUDE_black_beads_count_l2335_233514


namespace NUMINAMATH_CALUDE_arithmetic_sequence_d_value_l2335_233578

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem arithmetic_sequence_d_value
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 1)
  (h3 : a 3 = 11) :
  d = 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_d_value_l2335_233578


namespace NUMINAMATH_CALUDE_uneven_gender_probability_l2335_233567

/-- The number of children in the family -/
def num_children : ℕ := 8

/-- The probability of a child being male (or female) -/
def gender_prob : ℚ := 1/2

/-- The total number of possible gender combinations -/
def total_combinations : ℕ := 2^num_children

/-- The number of combinations with an even split of genders -/
def even_split_combinations : ℕ := Nat.choose num_children (num_children / 2)

/-- The probability of having an uneven number of sons and daughters -/
def prob_uneven : ℚ := 1 - (even_split_combinations : ℚ) / total_combinations

theorem uneven_gender_probability :
  prob_uneven = 93/128 :=
sorry

end NUMINAMATH_CALUDE_uneven_gender_probability_l2335_233567


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2335_233585

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2335_233585


namespace NUMINAMATH_CALUDE_boat_cost_correct_l2335_233576

/-- The cost of taking a boat to the Island of Mysteries -/
def boat_cost : ℚ := 254

/-- The cost of taking a plane to the Island of Mysteries -/
def plane_cost : ℚ := 600

/-- The amount saved by taking the boat instead of the plane -/
def savings : ℚ := 346

/-- Theorem stating that the boat cost is correct given the plane cost and savings -/
theorem boat_cost_correct : boat_cost = plane_cost - savings := by sorry

end NUMINAMATH_CALUDE_boat_cost_correct_l2335_233576


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2335_233596

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 7, where the remainder when divided by x - 2 is 21,
    the remainder when divided by x + 2 is 21 - 2F -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x ↦ D * x^4 + E * x^2 + F * x + 7
  (q 2 = 21) → 
  ∃ r : ℝ, ∀ x : ℝ, ∃ k : ℝ, q x = (x + 2) * k + r ∧ r = 21 - 2 * F :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2335_233596


namespace NUMINAMATH_CALUDE_bryan_pushups_l2335_233583

theorem bryan_pushups (planned_sets : ℕ) (pushups_per_set : ℕ) (actual_total : ℕ)
  (h1 : planned_sets = 3)
  (h2 : pushups_per_set = 15)
  (h3 : actual_total = 40) :
  planned_sets * pushups_per_set - actual_total = 5 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l2335_233583


namespace NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l2335_233539

theorem empty_quadratic_inequality_solution_set
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l2335_233539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l2335_233568

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_7 : a 7 = 8)
  (h_23 : a 23 = 22) :
  a 15 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l2335_233568


namespace NUMINAMATH_CALUDE_population_increase_birth_rate_l2335_233599

/-- Calculates the percentage increase in population due to birth over a given time period. -/
def population_increase_percentage (initial_population : ℕ) (final_population : ℕ) 
  (years : ℕ) (emigration_rate : ℕ) (immigration_rate : ℕ) : ℚ :=
  let net_migration := (immigration_rate - emigration_rate) * years
  let total_increase := final_population - initial_population - net_migration
  (total_increase : ℚ) / (initial_population : ℚ) * 100

/-- The percentage increase in population due to birth over 10 years is 55%. -/
theorem population_increase_birth_rate : 
  population_increase_percentage 100000 165000 10 2000 2500 = 55 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_birth_rate_l2335_233599


namespace NUMINAMATH_CALUDE_video_game_price_l2335_233562

theorem video_game_price (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) : 
  total_games = 16 → 
  non_working_games = 8 → 
  total_earnings = 56 → 
  total_earnings / (total_games - non_working_games) = 7 := by
sorry

end NUMINAMATH_CALUDE_video_game_price_l2335_233562


namespace NUMINAMATH_CALUDE_closest_integer_to_expression_l2335_233561

theorem closest_integer_to_expression : 
  let expr := (8^1500 + 8^1502) / (8^1501 + 8^1501)
  expr = 65/16 ∧ 
  ∀ n : ℤ, |expr - 4| ≤ |expr - n| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_expression_l2335_233561


namespace NUMINAMATH_CALUDE_combination_equality_implies_n_18_l2335_233564

theorem combination_equality_implies_n_18 (n : ℕ) :
  (Nat.choose n 14 = Nat.choose n 4) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_n_18_l2335_233564


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2335_233531

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 3 * x^2 + 6 * x - |-21 + 5|
  ∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 19 ∧ x₂ = -1 - Real.sqrt 19 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2335_233531


namespace NUMINAMATH_CALUDE_ourNumber_decimal_l2335_233571

/-- Represents a number in millions, thousands, and ones -/
structure LargeNumber where
  millions : Nat
  thousands : Nat
  ones : Nat

/-- Converts a LargeNumber to its decimal representation -/
def toDecimal (n : LargeNumber) : Nat :=
  n.millions * 1000000 + n.thousands * 1000 + n.ones

/-- The specific large number we're working with -/
def ourNumber : LargeNumber :=
  { millions := 10
  , thousands := 300
  , ones := 50 }

theorem ourNumber_decimal : toDecimal ourNumber = 10300050 := by
  sorry

end NUMINAMATH_CALUDE_ourNumber_decimal_l2335_233571


namespace NUMINAMATH_CALUDE_solution_set_l2335_233524

theorem solution_set (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ∧ x + 2 > 0 → x ∈ Set.Ioo (-2) (-1) ∪ {-1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l2335_233524


namespace NUMINAMATH_CALUDE_minimum_value_problem_l2335_233546

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^x * 4^y)) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.sqrt 2 = Real.sqrt (2^a * 4^b) → 1/x + x/y ≤ 1/a + a/b) ∧
  (1/x + x/y = 2 * Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l2335_233546


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l2335_233501

def hours_before_midterms (
  chess_hours_per_week : ℕ)
  (drama_hours_per_week : ℕ)
  (glee_hours_odd_week : ℕ)
  (robotics_hours_even_week : ℕ)
  (soccer_hours_odd_week : ℕ)
  (soccer_hours_even_week : ℕ)
  (weeks_in_semester : ℕ)
  (sick_weeks : ℕ)
  (midterm_week : ℕ)
  (drama_cancel_week : ℕ)
  (holiday_week : ℕ)
  (holiday_soccer_hours : ℕ) : ℕ :=
  -- Function body
  sorry

theorem annie_extracurricular_hours :
  hours_before_midterms 2 8 3 4 1 2 12 2 8 5 7 1 = 81 :=
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l2335_233501


namespace NUMINAMATH_CALUDE_total_power_cost_l2335_233560

/-- Represents the cost of power for each appliance in Joseph's house --/
structure ApplianceCosts where
  waterHeater : ℝ
  refrigerator : ℝ
  electricOven : ℝ
  airConditioner : ℝ
  washingMachine : ℝ

/-- Calculates the total cost of power for all appliances --/
def totalCost (costs : ApplianceCosts) : ℝ :=
  costs.waterHeater + costs.refrigerator + costs.electricOven + costs.airConditioner + costs.washingMachine

/-- Theorem stating the total cost of power for all appliances --/
theorem total_power_cost (costs : ApplianceCosts) 
  (h1 : costs.refrigerator = 3 * costs.waterHeater)
  (h2 : costs.electricOven = 500)
  (h3 : costs.electricOven = 2.5 * costs.waterHeater)
  (h4 : costs.airConditioner = 300)
  (h5 : costs.washingMachine = 100) :
  totalCost costs = 1700 := by
  sorry


end NUMINAMATH_CALUDE_total_power_cost_l2335_233560


namespace NUMINAMATH_CALUDE_triangle_right_angled_l2335_233577

theorem triangle_right_angled (A B C : Real) : 
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 + (Real.sin C) ^ 2 = 2 * ((Real.cos A) ^ 2 + (Real.cos B) ^ 2 + (Real.cos C) ^ 2) → 
  A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l2335_233577


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2335_233574

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x > 1 ∧ x < 2 → x^2 + m*x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2335_233574


namespace NUMINAMATH_CALUDE_no_perfect_squares_l2335_233565

/-- Represents a 100-digit number with a repeating pattern -/
def RepeatingNumber (pattern : ℕ) : ℕ :=
  -- Implementation details omitted for simplicity
  sorry

/-- N₁ is a 100-digit number consisting of all 3's -/
def N1 : ℕ := RepeatingNumber 3

/-- N₂ is a 100-digit number consisting of all 6's -/
def N2 : ℕ := RepeatingNumber 6

/-- N₃ is a 100-digit number with repeating pattern 15 -/
def N3 : ℕ := RepeatingNumber 15

/-- N₄ is a 100-digit number with repeating pattern 21 -/
def N4 : ℕ := RepeatingNumber 21

/-- N₅ is a 100-digit number with repeating pattern 27 -/
def N5 : ℕ := RepeatingNumber 27

/-- A number is a perfect square if there exists an integer whose square equals the number -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_perfect_squares : ¬(is_perfect_square N1 ∨ is_perfect_square N2 ∨ 
                               is_perfect_square N3 ∨ is_perfect_square N4 ∨ 
                               is_perfect_square N5) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l2335_233565


namespace NUMINAMATH_CALUDE_evening_ticket_price_is_seven_l2335_233542

/-- Represents the earnings of a movie theater on a single day. -/
structure TheaterEarnings where
  matineePrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  openingNightCustomers : ℕ
  totalEarnings : ℕ

/-- Calculates the evening ticket price based on the theater's earnings. -/
def eveningTicketPrice (e : TheaterEarnings) : ℕ :=
  let totalCustomers := e.matineeCustomers + e.eveningCustomers + e.openingNightCustomers
  let popcornEarnings := (totalCustomers / 2) * e.popcornPrice
  let knownEarnings := e.matineeCustomers * e.matineePrice + 
                       e.openingNightCustomers * e.openingNightPrice + 
                       popcornEarnings
  (e.totalEarnings - knownEarnings) / e.eveningCustomers

/-- Theorem stating that the evening ticket price is 7 dollars given the specific conditions. -/
theorem evening_ticket_price_is_seven :
  let e : TheaterEarnings := {
    matineePrice := 5,
    openingNightPrice := 10,
    popcornPrice := 10,
    matineeCustomers := 32,
    eveningCustomers := 40,
    openingNightCustomers := 58,
    totalEarnings := 1670
  }
  eveningTicketPrice e = 7 := by sorry

end NUMINAMATH_CALUDE_evening_ticket_price_is_seven_l2335_233542


namespace NUMINAMATH_CALUDE_marked_nodes_on_circle_l2335_233538

/-- Represents a node in the hexagon grid -/
structure Node where
  x : ℤ
  y : ℤ

/-- Represents a circle in the hexagon grid -/
structure Circle where
  center : Node
  radius : ℕ

/-- The side length of the regular hexagon -/
def hexagon_side_length : ℕ := 5

/-- The side length of the equilateral triangles -/
def triangle_side_length : ℕ := 1

/-- The total number of nodes in the hexagon -/
def total_nodes : ℕ := 91

/-- A function that determines if a node is marked -/
def is_marked : Node → Prop := sorry

/-- A function that determines if a node lies on a given circle -/
def on_circle : Node → Circle → Prop := sorry

/-- The main theorem to be proved -/
theorem marked_nodes_on_circle :
  (∃ (marked_nodes : Finset Node), 
    (∀ n ∈ marked_nodes, is_marked n) ∧ 
    (marked_nodes.card > total_nodes / 2)) →
  (∃ (c : Circle) (five_nodes : Finset Node),
    five_nodes.card = 5 ∧
    (∀ n ∈ five_nodes, is_marked n ∧ on_circle n c)) :=
by sorry

end NUMINAMATH_CALUDE_marked_nodes_on_circle_l2335_233538


namespace NUMINAMATH_CALUDE_sam_sandwich_count_l2335_233500

/-- The number of sandwiches Sam eats per day -/
def sandwiches_per_day : ℕ := sorry

/-- The ratio of apples to sandwiches -/
def apples_per_sandwich : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of apples eaten in a week -/
def total_apples : ℕ := 280

theorem sam_sandwich_count :
  sandwiches_per_day = 10 ∧
  sandwiches_per_day * apples_per_sandwich * days_in_week = total_apples :=
sorry

end NUMINAMATH_CALUDE_sam_sandwich_count_l2335_233500


namespace NUMINAMATH_CALUDE_correct_dispersion_measure_l2335_233521

-- Define a type for measures of data dispersion
structure DisperesionMeasure where
  makeFullUseOfData : Bool
  useMultipleNumericalValues : Bool
  smallerValueForLargerDispersion : Bool

-- Define a function to check if a dispersion measure is correct
def isCorrectMeasure (m : DisperesionMeasure) : Prop :=
  m.makeFullUseOfData ∧ m.useMultipleNumericalValues ∧ ¬m.smallerValueForLargerDispersion

-- Theorem: The correct dispersion measure makes full use of data and uses multiple numerical values
theorem correct_dispersion_measure :
  ∃ (m : DisperesionMeasure), isCorrectMeasure m ∧
    m.makeFullUseOfData = true ∧
    m.useMultipleNumericalValues = true :=
  sorry


end NUMINAMATH_CALUDE_correct_dispersion_measure_l2335_233521


namespace NUMINAMATH_CALUDE_xiaolin_mean_calculation_l2335_233551

theorem xiaolin_mean_calculation 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (x : ℝ) 
  (hx : x = (a + b) / 2) 
  (y : ℝ) 
  (hy : y = (x + c) / 2) : 
  y < (a + b + c) / 3 := by
sorry

end NUMINAMATH_CALUDE_xiaolin_mean_calculation_l2335_233551


namespace NUMINAMATH_CALUDE_specific_l_shape_perimeter_l2335_233586

/-- Represents an L-shaped region formed by congruent squares -/
structure LShapedRegion where
  squareCount : Nat
  topRowCount : Nat
  bottomRowCount : Nat
  totalArea : ℝ

/-- Calculates the perimeter of an L-shaped region -/
def calculatePerimeter (region : LShapedRegion) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific L-shaped region is 91 cm -/
theorem specific_l_shape_perimeter :
  let region : LShapedRegion := {
    squareCount := 8,
    topRowCount := 3,
    bottomRowCount := 5,
    totalArea := 392
  }
  calculatePerimeter region = 91 := by
  sorry

end NUMINAMATH_CALUDE_specific_l_shape_perimeter_l2335_233586


namespace NUMINAMATH_CALUDE_division_problem_l2335_233557

theorem division_problem (n : ℕ) : 
  (n / 15 = 6) ∧ (n % 15 = 5) → n = 95 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2335_233557


namespace NUMINAMATH_CALUDE_vladimir_digits_puzzle_l2335_233504

/-- Represents a three-digit number formed by digits a, b, c in that order -/
def form_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem vladimir_digits_puzzle :
  ∀ a b c : ℕ,
  a > b → b > c → c > 0 →
  form_number a b c = form_number c b a + form_number c a b →
  a = 9 ∧ b = 5 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_vladimir_digits_puzzle_l2335_233504


namespace NUMINAMATH_CALUDE_symmetry_x_axis_of_point_A_l2335_233505

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetry_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_x_axis_of_point_A :
  let A : Point3D := { x := 1, y := 2, z := 1 }
  symmetry_x_axis A = { x := 1, y := 2, z := -1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_of_point_A_l2335_233505


namespace NUMINAMATH_CALUDE_root_product_equality_l2335_233503

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_equality_l2335_233503


namespace NUMINAMATH_CALUDE_sugar_mixture_profit_l2335_233512

/-- 
Proves that mixing 41.724 kg of sugar costing Rs. 9 per kg with 21.276 kg of sugar costing Rs. 7 per kg 
results in a 10% gain when selling the mixture at Rs. 9.24 per kg, given that the total weight of the mixture is 63 kg.
-/
theorem sugar_mixture_profit (
  total_weight : ℝ) 
  (sugar_a_cost sugar_b_cost selling_price : ℝ)
  (sugar_a_weight sugar_b_weight : ℝ) :
  total_weight = 63 →
  sugar_a_cost = 9 →
  sugar_b_cost = 7 →
  selling_price = 9.24 →
  sugar_a_weight = 41.724 →
  sugar_b_weight = 21.276 →
  sugar_a_weight + sugar_b_weight = total_weight →
  let total_cost := sugar_a_cost * sugar_a_weight + sugar_b_cost * sugar_b_weight
  let total_revenue := selling_price * total_weight
  total_revenue = 1.1 * total_cost :=
by sorry

end NUMINAMATH_CALUDE_sugar_mixture_profit_l2335_233512


namespace NUMINAMATH_CALUDE_cube_root_3a_5b_square_root_4x_y_l2335_233587

-- Part 1
theorem cube_root_3a_5b (a b : ℝ) (h : b = 4 * Real.sqrt (3 * a - 2) + 2 * Real.sqrt (2 - 3 * a) + 5) :
  (3 * a + 5 * b) ^ (1/3 : ℝ) = 3 := by sorry

-- Part 2
theorem square_root_4x_y (x y : ℝ) (h : (x - 3)^2 + Real.sqrt (y - 4) = 0) :
  (4 * x + y) ^ (1/2 : ℝ) = 4 ∨ (4 * x + y) ^ (1/2 : ℝ) = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_3a_5b_square_root_4x_y_l2335_233587


namespace NUMINAMATH_CALUDE_male_to_female_ratio_l2335_233550

/-- Represents the Math club with its member composition -/
structure MathClub where
  total_members : ℕ
  female_members : ℕ
  male_members : ℕ
  total_is_sum : total_members = female_members + male_members

/-- The specific Math club instance from the problem -/
def problem_club : MathClub :=
  { total_members := 18
    female_members := 6
    male_members := 12
    total_is_sum := by rfl }

/-- The ratio of male to female members is 2:1 -/
theorem male_to_female_ratio (club : MathClub) 
  (h1 : club.total_members = 18) 
  (h2 : club.female_members = 6) : 
  club.male_members / club.female_members = 2 := by
  sorry

#check male_to_female_ratio problem_club rfl rfl

end NUMINAMATH_CALUDE_male_to_female_ratio_l2335_233550


namespace NUMINAMATH_CALUDE_jim_savings_rate_l2335_233598

/-- 
Given:
- Sara has already saved 4100 dollars
- Sara saves 10 dollars per week
- Jim saves x dollars per week
- After 820 weeks, Sara and Jim have saved the same amount

Prove: x = 15
-/

theorem jim_savings_rate (x : ℚ) : 
  (4100 + 820 * 10 = 820 * x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_jim_savings_rate_l2335_233598


namespace NUMINAMATH_CALUDE_hoseok_calculation_l2335_233590

theorem hoseok_calculation (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_calculation_l2335_233590


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2335_233582

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h1 : principal = 10000) 
  (h2 : time = 1) (h3 : interest = 500) : 
  (interest / (principal * time)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2335_233582


namespace NUMINAMATH_CALUDE_fifth_friend_contribution_l2335_233513

def friend_contribution (total : ℝ) (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = total ∧
  a = (1/2) * (b + c + d + e) ∧
  b = (1/3) * (a + c + d + e) ∧
  c = (1/4) * (a + b + d + e) ∧
  d = (1/5) * (a + b + c + e)

theorem fifth_friend_contribution :
  ∃ a b c d : ℝ, friend_contribution 120 a b c d 52.55 := by
  sorry

end NUMINAMATH_CALUDE_fifth_friend_contribution_l2335_233513


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2335_233540

theorem x_minus_y_values (x y : ℝ) 
  (hx : |x| = 4) 
  (hy : |y| = 2) 
  (hxy : |x + y| = x + y) : 
  x - y = 2 ∨ x - y = 6 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2335_233540


namespace NUMINAMATH_CALUDE_minimal_difference_factors_l2335_233549

theorem minimal_difference_factors : ∃ (a b : ℤ),
  a * b = 1234567890 ∧
  ∀ (x y : ℤ), x * y = 1234567890 → |x - y| ≥ |a - b| ∧
  a = 36070 ∧ b = 34227 := by sorry

end NUMINAMATH_CALUDE_minimal_difference_factors_l2335_233549


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l2335_233581

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 6 = 0

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  ∃ (chord_length : ℝ), 
    (∀ (x y : ℝ), line_eq x y → circle_eq x y → chord_length = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l2335_233581


namespace NUMINAMATH_CALUDE_simplify_sqrt_squared_l2335_233517

theorem simplify_sqrt_squared (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_squared_l2335_233517


namespace NUMINAMATH_CALUDE_spherical_distance_60N_l2335_233544

/-- The spherical distance between two points on a latitude circle --/
def spherical_distance (R : ℝ) (latitude : ℝ) (arc_length : ℝ) : ℝ :=
  sorry

/-- Theorem: Spherical distance between two points on 60°N latitude --/
theorem spherical_distance_60N (R : ℝ) (h : R > 0) :
  spherical_distance R (π / 3) (π * R / 2) = π * R / 3 :=
sorry

end NUMINAMATH_CALUDE_spherical_distance_60N_l2335_233544


namespace NUMINAMATH_CALUDE_convex_polygon_equal_division_l2335_233589

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  is_convex : Bool

/-- A straight line that divides a polygon -/
structure DividingLine where
  -- Add necessary fields for a dividing line

/-- A smaller polygon resulting from division -/
structure SmallerPolygon where
  perimeter : ℝ
  longest_side : ℝ

/-- Function to divide a convex polygon with a dividing line -/
def divide_polygon (p : ConvexPolygon) (l : DividingLine) : (SmallerPolygon × SmallerPolygon) :=
  sorry

/-- Theorem stating that any convex polygon can be divided into two smaller polygons
    with equal perimeters and equal longest sides -/
theorem convex_polygon_equal_division (p : ConvexPolygon) :
  ∃ (l : DividingLine),
    let (p1, p2) := divide_polygon p l
    p1.perimeter = p2.perimeter ∧ p1.longest_side = p2.longest_side :=
  sorry

end NUMINAMATH_CALUDE_convex_polygon_equal_division_l2335_233589


namespace NUMINAMATH_CALUDE_rectangle_variability_l2335_233570

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the perimeter, area, and one side length
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)
def area (r : Rectangle) : ℝ := r.length * r.width
def oneSideLength (r : Rectangle) : ℝ := r.length

-- State the theorem
theorem rectangle_variability (fixedPerimeter : ℝ) (r : Rectangle) 
  (h : perimeter r = fixedPerimeter) :
  ∃ (r' : Rectangle), 
    perimeter r' = fixedPerimeter ∧ 
    area r' ≠ area r ∧
    oneSideLength r' ≠ oneSideLength r :=
sorry

end NUMINAMATH_CALUDE_rectangle_variability_l2335_233570


namespace NUMINAMATH_CALUDE_factorization_equality_l2335_233559

theorem factorization_equality (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2335_233559


namespace NUMINAMATH_CALUDE_trisection_intersection_l2335_233516

noncomputable section

def f (x : ℝ) : ℝ := Real.log (x^2)

theorem trisection_intersection (A B C D E F : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := E
  let (x₄, y₄) := F
  0 < x₁ → x₁ < x₂ →
  x₁ = 1 → x₂ = 8 →
  y₁ = f x₁ → y₂ = f x₂ →
  C.2 = (2/3 : ℝ) * y₁ + (1/3 : ℝ) * y₂ →
  D.2 = (1/3 : ℝ) * y₁ + (2/3 : ℝ) * y₂ →
  y₃ = f x₃ → y₃ = C.2 →
  y₄ = f x₄ → y₄ = D.2 →
  x₃ = 4 ∧ x₄ = 16 := by
  sorry

end

end NUMINAMATH_CALUDE_trisection_intersection_l2335_233516


namespace NUMINAMATH_CALUDE_x_squared_eq_one_necessary_not_sufficient_l2335_233506

theorem x_squared_eq_one_necessary_not_sufficient (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_necessary_not_sufficient_l2335_233506


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2335_233592

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2335_233592


namespace NUMINAMATH_CALUDE_finite_solutions_hyperbolic_diophantine_equation_l2335_233556

theorem finite_solutions_hyperbolic_diophantine_equation
  (a b c d : ℤ) (ha : a ≠ 0) (hbcad : b * c - a * d ≠ 0) :
  Set.Finite {p : ℤ × ℤ | a * p.1 * p.2 + b * p.1 + c * p.2 + d = 0} :=
by sorry

end NUMINAMATH_CALUDE_finite_solutions_hyperbolic_diophantine_equation_l2335_233556


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2335_233580

theorem cubic_equation_solutions :
  (¬ ∃ (x y : ℕ), x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2335_233580


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2335_233566

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = 8*y
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points
def intersection_points (M N : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ line M.1 M.2 ∧
  parabola N.1 N.2 ∧ line N.1 N.2 ∧
  M ≠ N

-- Theorem statement
theorem intersection_chord_length :
  ∀ M N : ℝ × ℝ, intersection_points M N →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 16 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2335_233566


namespace NUMINAMATH_CALUDE_num_divisors_23232_l2335_233555

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- 23232 as a positive integer -/
def n : ℕ+ := 23232

/-- Theorem stating that the number of positive divisors of 23232 is 42 -/
theorem num_divisors_23232 : num_divisors n = 42 := by sorry

end NUMINAMATH_CALUDE_num_divisors_23232_l2335_233555


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l2335_233532

/-- Given positive integers c and d where 58 in base c equals 85 in base d,
    the least possible sum of c and d is 15. -/
theorem least_sum_of_bases (c d : ℕ) (hc : c > 0) (hd : d > 0)
  (h_eq : 5 * c + 8 = 8 * d + 5) : 
  (∀ c' d' : ℕ, c' > 0 → d' > 0 → 5 * c' + 8 = 8 * d' + 5 → c' + d' ≥ c + d) ∧ c + d = 15 := by
  sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l2335_233532


namespace NUMINAMATH_CALUDE_cut_piece_weight_for_equal_copper_percent_l2335_233588

/-- Represents an alloy with a given weight and copper percentage -/
structure Alloy where
  weight : ℝ
  copper_percent : ℝ

/-- Theorem stating the weight of the cut piece that equalizes copper percentages -/
theorem cut_piece_weight_for_equal_copper_percent 
  (alloy1 alloy2 : Alloy) 
  (h1 : alloy1.weight = 10)
  (h2 : alloy2.weight = 15)
  (h3 : alloy1.copper_percent ≠ alloy2.copper_percent) :
  ∃ x : ℝ, 
    x > 0 ∧ 
    x < min alloy1.weight alloy2.weight ∧
    ((alloy1.weight - x) * alloy1.copper_percent + x * alloy2.copper_percent) / alloy1.weight = 
    ((alloy2.weight - x) * alloy2.copper_percent + x * alloy1.copper_percent) / alloy2.weight → 
    x = 6 := by
  sorry

#check cut_piece_weight_for_equal_copper_percent

end NUMINAMATH_CALUDE_cut_piece_weight_for_equal_copper_percent_l2335_233588


namespace NUMINAMATH_CALUDE_andrew_kept_490_stickers_l2335_233593

/-- The number of stickers Andrew bought -/
def total_stickers : ℕ := 1500

/-- The number of stickers Daniel received -/
def daniel_stickers : ℕ := 250

/-- The number of stickers Fred received -/
def fred_stickers : ℕ := daniel_stickers + 120

/-- The number of stickers Emily received -/
def emily_stickers : ℕ := (daniel_stickers + fred_stickers) / 2

/-- The number of stickers Gina received -/
def gina_stickers : ℕ := 80

/-- The number of stickers Andrew kept -/
def andrew_stickers : ℕ := total_stickers - (daniel_stickers + fred_stickers + emily_stickers + gina_stickers)

theorem andrew_kept_490_stickers : andrew_stickers = 490 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_490_stickers_l2335_233593


namespace NUMINAMATH_CALUDE_m_range_l2335_233535

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the proposition p
def p (m : ℝ) : Prop := M.1 - M.2 + m < 0

-- Define the proposition q
def q (m : ℝ) : Prop := m ≠ -2

-- Define the theorem
theorem m_range (m : ℝ) : p m ∧ q m ↔ m ∈ Set.Ioo (-Real.pi) (-2) ∪ Set.Ioo (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2335_233535


namespace NUMINAMATH_CALUDE_ceiling_examples_ceiling_equals_two_m_range_equation_solutions_l2335_233529

-- Definition of the ceiling function for rational numbers
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Theorem 1: Calculating specific ceiling values
theorem ceiling_examples : ceiling (4.7) = 5 ∧ ceiling (-5.3) = -5 := by sorry

-- Theorem 2: Relationship when ceiling equals 2
theorem ceiling_equals_two (a : ℚ) : ceiling a = 2 ↔ 1 < a ∧ a ≤ 2 := by sorry

-- Theorem 3: Range of m satisfying the given condition
theorem m_range (m : ℚ) : ceiling (-2*m + 7) = -3 ↔ 5 ≤ m ∧ m < 5.5 := by sorry

-- Theorem 4: Solutions to the equation
theorem equation_solutions (n : ℚ) : ceiling (4.5*n - 2.5) = 3*n + 1 ↔ n = 2 ∨ n = 7/3 := by sorry

end NUMINAMATH_CALUDE_ceiling_examples_ceiling_equals_two_m_range_equation_solutions_l2335_233529


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2335_233536

theorem opposite_of_negative_two :
  -((-2 : ℤ)) = (2 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2335_233536


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2335_233547

theorem arithmetic_mean_problem (x : ℝ) : 
  (x + 5 + 17 + 3*x + 11 + 3*x + 6) / 5 = 19 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2335_233547


namespace NUMINAMATH_CALUDE_bus_wheel_radius_l2335_233554

/-- The radius of a bus wheel given its speed and revolutions per minute -/
theorem bus_wheel_radius 
  (speed_kmh : ℝ) 
  (rpm : ℝ) 
  (h1 : speed_kmh = 66) 
  (h2 : rpm = 70.06369426751593) : 
  ∃ (r : ℝ), abs (r - 2500.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bus_wheel_radius_l2335_233554


namespace NUMINAMATH_CALUDE_initial_cats_proof_l2335_233579

def total_cats : ℕ := 7
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2

def initial_cats : ℕ := total_cats - (female_kittens + male_kittens)

theorem initial_cats_proof : initial_cats = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_cats_proof_l2335_233579


namespace NUMINAMATH_CALUDE_product_mod_800_l2335_233541

theorem product_mod_800 : (2431 * 1587) % 800 = 397 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_800_l2335_233541


namespace NUMINAMATH_CALUDE_weight_of_four_moles_l2335_233594

/-- Given a compound with a molecular weight of 260, prove that 4 moles of this compound weighs 1040 grams. -/
theorem weight_of_four_moles (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 260 → moles = 4 → moles * molecular_weight = 1040 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_four_moles_l2335_233594


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l2335_233518

theorem complex_magnitude_proof : Complex.abs (3/4 + 3*I) = (Real.sqrt 153)/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l2335_233518


namespace NUMINAMATH_CALUDE_workshop_efficiency_l2335_233522

theorem workshop_efficiency (x : ℝ) : 
  (1500 / x - 1500 / (2.5 * x) = 18) → x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_efficiency_l2335_233522


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2335_233575

-- Define the coefficients of the original quadratic equation
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 2

-- Define the roots of the original quadratic equation
def α : ℝ := sorry
def β : ℝ := sorry

-- Define the coefficients of the new quadratic equation
def a' : ℝ := 4
def p : ℝ := sorry
def q : ℝ := sorry

-- State the theorem
theorem quadratic_roots_relation :
  (3 * α^2 + 4 * α + 2 = 0) ∧
  (3 * β^2 + 4 * β + 2 = 0) ∧
  (4 * (2*α + 1)^2 + p * (2*α + 1) + q = 0) ∧
  (4 * (2*β + 1)^2 + p * (2*β + 1) + q = 0) →
  p = 8/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2335_233575


namespace NUMINAMATH_CALUDE_num_sandwiches_al_can_order_l2335_233507

/-- Represents the number of different types of bread offered at the deli. -/
def num_breads : Nat := 5

/-- Represents the number of different types of meat offered at the deli. -/
def num_meats : Nat := 7

/-- Represents the number of different types of cheese offered at the deli. -/
def num_cheeses : Nat := 6

/-- Represents the number of restricted sandwich combinations. -/
def num_restricted : Nat := 16

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem num_sandwiches_al_can_order :
  (num_breads * num_meats * num_cheeses) - num_restricted = 194 := by
  sorry

end NUMINAMATH_CALUDE_num_sandwiches_al_can_order_l2335_233507


namespace NUMINAMATH_CALUDE_go_game_probability_l2335_233543

theorem go_game_probability (P : ℝ) (h1 : P > 1/2) 
  (h2 : P^2 + (1-P)^2 = 5/8) : P = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_go_game_probability_l2335_233543


namespace NUMINAMATH_CALUDE_money_needed_proof_l2335_233552

def car_wash_count : ℕ := 5
def car_wash_price : ℚ := 8.5
def dog_walk_count : ℕ := 4
def dog_walk_price : ℚ := 6.75
def lawn_mow_count : ℕ := 3
def lawn_mow_price : ℚ := 12.25
def bicycle_price : ℚ := 150.25
def helmet_price : ℚ := 35.75
def lock_price : ℚ := 24.5

def total_money_made : ℚ := 
  car_wash_count * car_wash_price + 
  dog_walk_count * dog_walk_price + 
  lawn_mow_count * lawn_mow_price

def total_cost : ℚ := 
  bicycle_price + helmet_price + lock_price

theorem money_needed_proof : 
  total_cost - total_money_made = 104.25 := by sorry

end NUMINAMATH_CALUDE_money_needed_proof_l2335_233552


namespace NUMINAMATH_CALUDE_sum_is_odd_l2335_233528

theorem sum_is_odd : Odd (2^1990 + 3^1990 + 7^1990 + 9^1990) := by
  sorry

end NUMINAMATH_CALUDE_sum_is_odd_l2335_233528


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2335_233595

theorem right_triangle_third_side
  (m n : ℝ)
  (h1 : |m - 3| + Real.sqrt (n - 4) = 0)
  (h2 : m > 0 ∧ n > 0)
  (h3 : ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ ((a = m ∧ b = n) ∨ (a = m ∧ c = n) ∨ (b = m ∧ c = n)))
  : ∃ (x : ℝ), (x = 5 ∨ x = Real.sqrt 7) ∧
    ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ ((a = m ∧ b = n ∧ c = x) ∨ (a = m ∧ c = n ∧ b = x) ∨ (b = m ∧ c = n ∧ a = x)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2335_233595


namespace NUMINAMATH_CALUDE_rectangular_box_width_l2335_233553

/-- Proves that the width of rectangular boxes is 5 cm given the conditions of the problem -/
theorem rectangular_box_width (wooden_length wooden_width wooden_height : ℕ)
                               (box_length box_height : ℕ)
                               (max_boxes : ℕ) :
  wooden_length = 800 →
  wooden_width = 1000 →
  wooden_height = 600 →
  box_length = 4 →
  box_height = 6 →
  max_boxes = 4000000 →
  ∃ (box_width : ℕ),
    box_width = 5 ∧
    wooden_length * wooden_width * wooden_height =
    max_boxes * (box_length * box_width * box_height) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_width_l2335_233553


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2335_233572

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x ∈ U | x^2 - 4*x + 3 = 0}

theorem complement_of_M_in_U : (U \ M) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2335_233572


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solution_l2335_233515

theorem cubic_equation_integer_solution :
  ∃! (x : ℤ), 2 * x^3 + 5 * x^2 - 9 * x - 18 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solution_l2335_233515


namespace NUMINAMATH_CALUDE_work_completion_time_l2335_233509

theorem work_completion_time (x : ℝ) (h1 : x > 0) : 
  (6 * (1/x + 1/20) = 0.7) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2335_233509


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2335_233519

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  3 * a 1 - (1/2) * a 3 = (1/2) * a 3 - 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a → ArithmeticCondition a →
  ∀ n : ℕ, (a (n + 3) + a (n + 2)) / (a (n + 1) + a n) = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2335_233519


namespace NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l2335_233533

/-- Given a sequence of seven consecutive even integers with a sum of 700,
    the smallest number in the sequence is 94. -/
theorem smallest_of_seven_consecutive_evens (seq : List ℤ) : 
  seq.length = 7 ∧ 
  (∀ i ∈ seq, ∃ k : ℤ, i = 2 * k) ∧ 
  (∀ i j, i ∈ seq → j ∈ seq → i ≠ j → (i - j).natAbs = 2) ∧
  seq.sum = 700 →
  seq.minimum? = some 94 := by
sorry

end NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l2335_233533


namespace NUMINAMATH_CALUDE_max_xy_value_l2335_233502

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 5 * y = 200) : x * y ≤ 285 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2335_233502


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2335_233569

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 + a₁ * x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2335_233569


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2335_233527

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (0 : ℤ) ≤ x → x^2 < 2*x + 1 → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2335_233527


namespace NUMINAMATH_CALUDE_power_equation_solution_l2335_233548

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^42 → n = 42 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2335_233548


namespace NUMINAMATH_CALUDE_cos_double_angle_when_tan_is_one_l2335_233537

theorem cos_double_angle_when_tan_is_one (θ : Real) (h : Real.tan θ = 1) : 
  Real.cos (2 * θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_when_tan_is_one_l2335_233537


namespace NUMINAMATH_CALUDE_book_price_increase_l2335_233530

theorem book_price_increase (new_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) :
  new_price = 420 ∧ 
  increase_percentage = 40 ∧ 
  new_price = original_price * (1 + increase_percentage / 100) → 
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l2335_233530


namespace NUMINAMATH_CALUDE_minimum_students_l2335_233584

theorem minimum_students (boys girls : ℕ) : 
  boys > 0 → 
  girls > 0 → 
  (3 * boys) / 4 = (2 * girls) / 3 → 
  ∃ (total : ℕ), total = boys + girls ∧ total ≥ 17 ∧ 
    ∀ (b g : ℕ), b > 0 → g > 0 → (3 * b) / 4 = (2 * g) / 3 → b + g ≥ total :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_students_l2335_233584


namespace NUMINAMATH_CALUDE_correct_statements_l2335_233511

-- Define the types of relationships
inductive Relationship
| Function
| Correlation

-- Define the types of analysis methods
inductive AnalysisMethod
| Regression

-- Define the properties of relationships
def isDeterministic (r : Relationship) : Prop :=
  match r with
  | Relationship.Function => True
  | Relationship.Correlation => False

-- Define the properties of analysis methods
def isCommonlyUsedFor (m : AnalysisMethod) (r : Relationship) : Prop :=
  match m, r with
  | AnalysisMethod.Regression, Relationship.Correlation => True
  | _, _ => False

-- Theorem to prove
theorem correct_statements :
  isDeterministic Relationship.Function ∧
  ¬isDeterministic Relationship.Correlation ∧
  isCommonlyUsedFor AnalysisMethod.Regression Relationship.Correlation :=
by sorry


end NUMINAMATH_CALUDE_correct_statements_l2335_233511


namespace NUMINAMATH_CALUDE_fraction_equality_l2335_233545

theorem fraction_equality : (1 : ℚ) / 4 - (1 : ℚ) / 6 + (1 : ℚ) / 8 = (5 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2335_233545


namespace NUMINAMATH_CALUDE_finance_to_manufacturing_ratio_l2335_233510

theorem finance_to_manufacturing_ratio 
  (finance_angle : ℝ) 
  (manufacturing_angle : ℝ) 
  (h1 : finance_angle = 72) 
  (h2 : manufacturing_angle = 108) : 
  (finance_angle / manufacturing_angle) = (2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_finance_to_manufacturing_ratio_l2335_233510


namespace NUMINAMATH_CALUDE_three_distinct_zeros_range_l2335_233573

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- State the theorem
theorem three_distinct_zeros_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_zeros_range_l2335_233573


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l2335_233597

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  z₁ = 2 + I →
  (z₁.re = -z₂.re ∧ z₁.im = z₂.im) →
  z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l2335_233597
