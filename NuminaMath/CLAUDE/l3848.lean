import Mathlib

namespace NUMINAMATH_CALUDE_park_area_l3848_384871

theorem park_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 3 * width → 
  perimeter = 2 * (width + length) → 
  perimeter = 72 → 
  area = width * length → 
  area = 243 := by sorry

end NUMINAMATH_CALUDE_park_area_l3848_384871


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3848_384848

theorem hyperbola_equation (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  (∃ e : ℝ, e = k * Real.sqrt 5 ∧ 
   (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = k * x)) →
  (∃ x y : ℝ, x^2 / (4 * b^2) - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3848_384848


namespace NUMINAMATH_CALUDE_variance_linear_transform_l3848_384820

variable {α : Type*} [LinearOrderedField α]
variable (x : Finset ℕ → α)
variable (n : ℕ)

def variance (x : Finset ℕ → α) (n : ℕ) : α := sorry

theorem variance_linear_transform 
  (h : variance x n = 2) : 
  variance (fun i => 3 * x i + 2) n = 18 := by
  sorry

end NUMINAMATH_CALUDE_variance_linear_transform_l3848_384820


namespace NUMINAMATH_CALUDE_workshop_day_probability_l3848_384817

/-- The probability of a student being absent on a normal day -/
def normal_absence_rate : ℚ := 1/20

/-- The probability of a student being absent on the workshop day -/
def workshop_absence_rate : ℚ := min (2 * normal_absence_rate) 1

/-- The probability of a student being present on the workshop day -/
def workshop_presence_rate : ℚ := 1 - workshop_absence_rate

/-- The probability of one student being absent and one being present on the workshop day -/
def one_absent_one_present : ℚ := 
  workshop_absence_rate * workshop_presence_rate * 2

theorem workshop_day_probability : one_absent_one_present = 18/100 := by
  sorry

end NUMINAMATH_CALUDE_workshop_day_probability_l3848_384817


namespace NUMINAMATH_CALUDE_problem_solution_l3848_384886

theorem problem_solution (a : ℚ) : a + a / 3 + a / 4 = 4 → a = 48 / 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3848_384886


namespace NUMINAMATH_CALUDE_inequality_theorem_l3848_384879

theorem inequality_theorem (x y : ℝ) 
  (h1 : y ≥ 0) 
  (h2 : y * (y + 1) ≤ (x + 1)^2) 
  (h3 : y * (y - 1) ≤ x^2) : 
  y * (y - 1) ≤ x^2 ∧ y * (y + 1) ≤ (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3848_384879


namespace NUMINAMATH_CALUDE_hunter_frog_count_l3848_384897

/-- The total number of frogs Hunter saw in the pond -/
def total_frogs (initial : ℕ) (on_logs : ℕ) (babies : ℕ) : ℕ :=
  initial + on_logs + babies

/-- Theorem stating the total number of frogs Hunter saw -/
theorem hunter_frog_count :
  total_frogs 5 3 24 = 32 := by
  sorry

end NUMINAMATH_CALUDE_hunter_frog_count_l3848_384897


namespace NUMINAMATH_CALUDE_count_eight_to_thousand_l3848_384836

/-- Count of digit 8 in a single integer -/
def count_eight (n : ℕ) : ℕ := sorry

/-- Sum of count_eight for integers from 1 to n -/
def sum_count_eight (n : ℕ) : ℕ := sorry

/-- The count of digit 8 in integers from 1 to 1000 is 300 -/
theorem count_eight_to_thousand : sum_count_eight 1000 = 300 := by sorry

end NUMINAMATH_CALUDE_count_eight_to_thousand_l3848_384836


namespace NUMINAMATH_CALUDE_equation_solution_l3848_384816

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^10 = (10 * x)^5 ↔ x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3848_384816


namespace NUMINAMATH_CALUDE_team_a_games_played_l3848_384849

theorem team_a_games_played (team_a_win_ratio : ℚ) (team_b_win_ratio : ℚ) 
  (team_b_extra_wins : ℕ) (team_b_extra_losses : ℕ) :
  team_a_win_ratio = 3/4 →
  team_b_win_ratio = 2/3 →
  team_b_extra_wins = 5 →
  team_b_extra_losses = 3 →
  ∃ (a : ℕ), 
    a = 4 ∧
    team_b_win_ratio * (a + team_b_extra_wins + team_b_extra_losses) = 
      team_a_win_ratio * a + team_b_extra_wins :=
by sorry

end NUMINAMATH_CALUDE_team_a_games_played_l3848_384849


namespace NUMINAMATH_CALUDE_game_theorem_l3848_384878

/-- Represents the outcome of a single round -/
inductive RoundOutcome
| OddDifference
| EvenDifference

/-- Represents the game state -/
structure GameState :=
  (playerAPoints : ℤ)
  (playerBPoints : ℤ)

/-- The game rules -/
def gameRules (n : ℕ+) (outcome : RoundOutcome) (state : GameState) : GameState :=
  match outcome with
  | RoundOutcome.OddDifference  => ⟨state.playerAPoints - 2, state.playerBPoints + 2⟩
  | RoundOutcome.EvenDifference => ⟨state.playerAPoints + n, state.playerBPoints - n⟩

/-- The probability of an odd difference in a single round -/
def probOddDifference : ℚ := 3/5

/-- The probability of an even difference in a single round -/
def probEvenDifference : ℚ := 2/5

/-- The expected value of player A's points after the game -/
def expectedValue (n : ℕ+) : ℚ := (6 * n - 18) / 5

/-- The theorem to be proved -/
theorem game_theorem (n : ℕ+) :
  (∀ m : ℕ+, m < n → expectedValue m ≤ 0) ∧
  expectedValue n > 0 ∧
  n = 4 →
  (probOddDifference^3 + 3 * probOddDifference^2 * probEvenDifference) *
  (3 * probOddDifference * probEvenDifference^2) / 
  (1 - probEvenDifference^3) = 4/13 := by
  sorry


end NUMINAMATH_CALUDE_game_theorem_l3848_384878


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3848_384862

theorem trigonometric_identity : 
  Real.cos (π / 12) * Real.cos (5 * π / 12) + Real.cos (π / 8)^2 - 1/2 = (Real.sqrt 2 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3848_384862


namespace NUMINAMATH_CALUDE_roberto_outfits_l3848_384838

/-- Calculates the number of different outfits given the number of options for each clothing item. -/
def calculate_outfits (trousers shirts jackets ties : ℕ) : ℕ :=
  trousers * shirts * jackets * ties

/-- Theorem stating that Roberto can create 240 different outfits. -/
theorem roberto_outfits :
  calculate_outfits 5 6 4 2 = 240 := by
  sorry

#eval calculate_outfits 5 6 4 2

end NUMINAMATH_CALUDE_roberto_outfits_l3848_384838


namespace NUMINAMATH_CALUDE_problem_solution_l3848_384866

theorem problem_solution : 48 / (7 - 3/4 + 1/8) = 128/17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3848_384866


namespace NUMINAMATH_CALUDE_shortest_distance_is_zero_l3848_384828

/-- Define a 3D vector -/
def Vector3D := Fin 3 → ℝ

/-- Define the first line -/
def line1 (t : ℝ) : Vector3D := fun i => 
  match i with
  | 0 => 4 + 3*t
  | 1 => 1 - t
  | 2 => 3 + 2*t

/-- Define the second line -/
def line2 (s : ℝ) : Vector3D := fun i =>
  match i with
  | 0 => 1 + 2*s
  | 1 => 2 + 3*s
  | 2 => 5 - 2*s

/-- Calculate the square of the distance between two points -/
def distanceSquared (v w : Vector3D) : ℝ :=
  (v 0 - w 0)^2 + (v 1 - w 1)^2 + (v 2 - w 2)^2

/-- Theorem: The shortest distance between the two lines is 0 -/
theorem shortest_distance_is_zero :
  ∃ (t s : ℝ), distanceSquared (line1 t) (line2 s) = 0 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_is_zero_l3848_384828


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_l3848_384823

theorem sum_of_four_cubes (k : ℤ) : ∃ (a b c d : ℤ), 24 * k = a^3 + b^3 + c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_l3848_384823


namespace NUMINAMATH_CALUDE_min_cost_theorem_l3848_384855

/-- Represents the voting system in country Y -/
structure VotingSystem where
  total_voters : Nat
  sellable_voters : Nat
  preference_voters : Nat
  initial_votes : Nat
  votes_to_win : Nat

/-- Calculates the number of votes a candidate can secure based on the price offered -/
def supply_function (system : VotingSystem) (price : Nat) : Nat :=
  if price = 0 then system.initial_votes
  else if price ≤ system.sellable_voters then min (system.initial_votes + price) system.total_voters
  else min (system.initial_votes + system.sellable_voters) system.total_voters

/-- Calculates the minimum cost to win the election -/
def min_cost_to_win (system : VotingSystem) : Nat :=
  let required_additional_votes := system.votes_to_win - system.initial_votes
  required_additional_votes * (required_additional_votes + 1)

/-- The main theorem stating the minimum cost to win the election -/
theorem min_cost_theorem (system : VotingSystem) 
    (h1 : system.total_voters = 35)
    (h2 : system.sellable_voters = 14)
    (h3 : system.preference_voters = 21)
    (h4 : system.initial_votes = 10)
    (h5 : system.votes_to_win = 18) :
    min_cost_to_win system = 162 := by
  sorry

#eval min_cost_to_win { total_voters := 35, sellable_voters := 14, preference_voters := 21, initial_votes := 10, votes_to_win := 18 }

end NUMINAMATH_CALUDE_min_cost_theorem_l3848_384855


namespace NUMINAMATH_CALUDE_stadium_sections_theorem_l3848_384858

theorem stadium_sections_theorem : 
  ∃ (N : ℕ), N > 0 ∧ 
  (∃ (A C : ℕ), 7 * A = 11 * C ∧ N = A + C) ∧ 
  (∀ (M : ℕ), M > 0 → 
    (∃ (A C : ℕ), 7 * A = 11 * C ∧ M = A + C) → M ≥ N) ∧
  N = 18 :=
sorry

end NUMINAMATH_CALUDE_stadium_sections_theorem_l3848_384858


namespace NUMINAMATH_CALUDE_ticket_probability_problem_l3848_384892

theorem ticket_probability_problem : ∃! n : ℕ, 
  1 ≤ n ∧ n ≤ 20 ∧ 
  (↑(Finset.filter (λ x => x % n = 0) (Finset.range 20)).card / 20 : ℚ) = 3/10 ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ticket_probability_problem_l3848_384892


namespace NUMINAMATH_CALUDE_laura_workout_speed_l3848_384893

theorem laura_workout_speed :
  ∃! x : ℝ, x > 0 ∧ (30 / (3 * x + 2) + 3 / x = (230 - 10) / 60) := by
  sorry

end NUMINAMATH_CALUDE_laura_workout_speed_l3848_384893


namespace NUMINAMATH_CALUDE_marble_difference_l3848_384826

theorem marble_difference (connie_marbles juan_marbles : ℕ) 
  (h1 : connie_marbles = 323)
  (h2 : juan_marbles = 498)
  (h3 : juan_marbles > connie_marbles) : 
  juan_marbles - connie_marbles = 175 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3848_384826


namespace NUMINAMATH_CALUDE_square_equals_four_digit_l3848_384819

theorem square_equals_four_digit : ∃ (M N : ℕ), 
  10 ≤ M ∧ M < 100 ∧ 
  1000 ≤ N ∧ N < 10000 ∧ 
  M^2 = N :=
sorry

end NUMINAMATH_CALUDE_square_equals_four_digit_l3848_384819


namespace NUMINAMATH_CALUDE_christine_needs_32_tablespoons_l3848_384811

/-- Represents the number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- Represents the number of cakes Christine is making -/
def num_cakes : ℕ := 2

/-- Represents the number of egg whites required for each cake -/
def egg_whites_per_cake : ℕ := 8

/-- Calculates the total number of tablespoons of aquafaba needed -/
def aquafaba_needed : ℕ := aquafaba_per_egg * num_cakes * egg_whites_per_cake

/-- Proves that Christine needs 32 tablespoons of aquafaba -/
theorem christine_needs_32_tablespoons : aquafaba_needed = 32 := by
  sorry

end NUMINAMATH_CALUDE_christine_needs_32_tablespoons_l3848_384811


namespace NUMINAMATH_CALUDE_range_of_m_l3848_384854

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - m ≥ 0) → m ∈ Set.Icc (-4) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3848_384854


namespace NUMINAMATH_CALUDE_product_equals_square_l3848_384824

theorem product_equals_square : 100 * 19.98 * 1.998 * 1000 = (1998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3848_384824


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3848_384884

theorem arithmetic_simplification : 2 - (-3) - 4 - (-5) * 2 - 6 - (-7) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3848_384884


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l3848_384835

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original current : ℕ) : ℕ :=
  current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l3848_384835


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3848_384847

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 169 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3848_384847


namespace NUMINAMATH_CALUDE_circle_common_chord_l3848_384898

variables (a b x y : ℝ)

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*b*y = 0

-- Define the resulting circle
def resultCircle (x y : ℝ) : Prop := (a^2 + b^2)*(x^2 + y^2) - 2*a*b*(b*x + a*y) = 0

-- Theorem statement
theorem circle_common_chord (hb : b ≠ 0) :
  ∃ (x y : ℝ), circle1 a x y ∧ circle2 b x y →
  resultCircle a b x y ∧
  ∀ (x' y' : ℝ), resultCircle a b x' y' →
    ∃ (t : ℝ), x' = x + t*(y - x) ∧ y' = y + t*(x - y) :=
sorry

end NUMINAMATH_CALUDE_circle_common_chord_l3848_384898


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3848_384808

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h1 : a 5 * a 6 = 3) 
  (h2 : a 9 * a 10 = 9) : 
  a 7 * a 8 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3848_384808


namespace NUMINAMATH_CALUDE_function_composition_equality_l3848_384876

theorem function_composition_equality (a b c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = c * x + d)
  (ha : a = 2 * c) :
  (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ c = 1/2) := by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3848_384876


namespace NUMINAMATH_CALUDE_employee_pay_solution_exists_and_unique_l3848_384880

/-- Represents the weekly pay of employees X, Y, and Z -/
structure EmployeePay where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Conditions for the employee pay problem -/
def satisfiesConditions (pay : EmployeePay) : Prop :=
  pay.x = 1.2 * pay.y ∧
  pay.z = 0.75 * pay.x ∧
  pay.x + pay.y + pay.z = 1540

/-- Theorem stating the existence and uniqueness of the solution -/
theorem employee_pay_solution_exists_and_unique :
  ∃! pay : EmployeePay, satisfiesConditions pay :=
sorry

end NUMINAMATH_CALUDE_employee_pay_solution_exists_and_unique_l3848_384880


namespace NUMINAMATH_CALUDE_construction_rate_calculation_l3848_384843

/-- Represents the hourly rate for construction work -/
def construction_rate : ℝ := 14.67

/-- Represents the total weekly earnings -/
def total_earnings : ℝ := 300

/-- Represents the hourly rate for library work -/
def library_rate : ℝ := 8

/-- Represents the total weekly work hours -/
def total_hours : ℝ := 25

/-- Represents the weekly hours worked at the library -/
def library_hours : ℝ := 10

theorem construction_rate_calculation :
  construction_rate = (total_earnings - library_rate * library_hours) / (total_hours - library_hours) :=
by sorry

#check construction_rate_calculation

end NUMINAMATH_CALUDE_construction_rate_calculation_l3848_384843


namespace NUMINAMATH_CALUDE_line_general_form_l3848_384883

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralForm where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with slope -3 passing through the point (1, 2),
    its general form equation is 3x + y - 5 = 0 -/
theorem line_general_form (l : Line) 
    (h1 : l.slope = -3)
    (h2 : l.point = (1, 2)) :
    ∃ (g : GeneralForm), g.a = 3 ∧ g.b = 1 ∧ g.c = -5 :=
by sorry

end NUMINAMATH_CALUDE_line_general_form_l3848_384883


namespace NUMINAMATH_CALUDE_city_outgoing_roads_l3848_384888

/-- Represents a city with squares and roads -/
structure City where
  /-- Number of squares in the city -/
  squares : ℕ
  /-- Number of streets going out of the city -/
  outgoing_streets : ℕ
  /-- Number of avenues going out of the city -/
  outgoing_avenues : ℕ
  /-- Number of crescents going out of the city -/
  outgoing_crescents : ℕ
  /-- Total number of outgoing roads is 3 -/
  outgoing_total : outgoing_streets + outgoing_avenues + outgoing_crescents = 3

/-- Theorem: In a city where exactly three roads meet at every square (one street, one avenue, and one crescent),
    and three roads go outside of the city, there must be exactly one street, one avenue, and one crescent going out of the city -/
theorem city_outgoing_roads (c : City) : 
  c.outgoing_streets = 1 ∧ c.outgoing_avenues = 1 ∧ c.outgoing_crescents = 1 := by
  sorry

end NUMINAMATH_CALUDE_city_outgoing_roads_l3848_384888


namespace NUMINAMATH_CALUDE_sum_odd_integers_13_to_41_l3848_384833

/-- The sum of odd integers from 13 to 41, inclusive -/
def sumOddIntegers : ℕ :=
  let first := 13
  let last := 41
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem sum_odd_integers_13_to_41 :
  sumOddIntegers = 405 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_13_to_41_l3848_384833


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l3848_384844

/-- Represents the number of wheels on a vehicle -/
inductive VehicleType
  | twoWheeler
  | fourWheeler

/-- Calculates the number of wheels for a given vehicle type -/
def wheelCount (v : VehicleType) : Nat :=
  match v with
  | .twoWheeler => 2
  | .fourWheeler => 4

/-- Represents a parking configuration -/
structure ParkingConfig where
  twoWheelers : Nat
  fourWheelers : Nat

/-- Calculates the total number of wheels for a given parking configuration -/
def totalWheels (config : ParkingConfig) : Nat :=
  config.twoWheelers * wheelCount VehicleType.twoWheeler +
  config.fourWheelers * wheelCount VehicleType.fourWheeler

/-- Theorem stating that multiple solutions exist for the parking problem -/
theorem multiple_solutions_exist :
  ∃ (config1 config2 : ParkingConfig),
    totalWheels config1 = 70 ∧
    totalWheels config2 = 70 ∧
    config1.fourWheelers ≠ config2.fourWheelers :=
by
  sorry

#check multiple_solutions_exist

end NUMINAMATH_CALUDE_multiple_solutions_exist_l3848_384844


namespace NUMINAMATH_CALUDE_workshop_workers_count_l3848_384857

/-- Proves that the total number of workers in a workshop is 28 given the salary conditions --/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  W = N + 7 →  -- Total workers = Non-technicians + Technicians
  W * 8000 = 7 * 14000 + N * 6000 →  -- Total salary equation
  W = 28 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l3848_384857


namespace NUMINAMATH_CALUDE_roper_lawn_cutting_l3848_384832

/-- The number of times Mr. Roper cuts his lawn per month from April to September -/
def summer_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn per month from October to March -/
def winter_cuts : ℕ := 3

/-- The number of months in each season (summer and winter) -/
def months_per_season : ℕ := 6

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

/-- The average number of times Mr. Roper cuts his lawn per month -/
def average_cuts : ℚ := (summer_cuts * months_per_season + winter_cuts * months_per_season) / months_in_year

theorem roper_lawn_cutting :
  average_cuts = 9 := by sorry

end NUMINAMATH_CALUDE_roper_lawn_cutting_l3848_384832


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l3848_384822

theorem complex_fraction_equals_neg_i : (1 + 2*Complex.I) / (Complex.I - 2) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l3848_384822


namespace NUMINAMATH_CALUDE_largest_special_number_l3848_384891

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem largest_special_number : 
  ∀ n : ℕ, n < 200 → is_perfect_square n → n % 3 = 0 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_l3848_384891


namespace NUMINAMATH_CALUDE_inequality_proof_l3848_384860

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp : 0 < p) 
  (hpa : p ≤ a) (hpb : p ≤ b) (hpc : p ≤ c) (hpd : p ≤ d) (hpe : p ≤ e)
  (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 
    25 + 6 * (Real.sqrt (q/p) - Real.sqrt (p/q))^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3848_384860


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3848_384873

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence with first term 2, last term 29, and common difference 3 is 155 -/
theorem arithmetic_sequence_sum : arithmetic_sum 2 29 3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3848_384873


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_six_l3848_384827

theorem unique_square_divisible_by_six : ∃! x : ℕ,
  (∃ n : ℕ, x = n^2) ∧ 
  (∃ k : ℕ, x = 6 * k) ∧
  50 ≤ x ∧
  x ≤ 150 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_six_l3848_384827


namespace NUMINAMATH_CALUDE_parking_lot_length_l3848_384829

/-- Proves that given the conditions of the parking lot problem, the length is 500 feet -/
theorem parking_lot_length
  (width : ℝ)
  (usable_percentage : ℝ)
  (area_per_car : ℝ)
  (total_cars : ℝ)
  (h1 : width = 400)
  (h2 : usable_percentage = 0.8)
  (h3 : area_per_car = 10)
  (h4 : total_cars = 16000)
  : ∃ (length : ℝ), length = 500 ∧ width * length * usable_percentage = total_cars * area_per_car :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_length_l3848_384829


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3848_384899

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence definition
  r > 1 →  -- increasing sequence
  a 1 + a 3 + a 5 = 21 →  -- given condition
  a 3 = 6 →  -- given condition
  a 5 + a 7 + a 9 = 84 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3848_384899


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3848_384818

/-- Given a triangle ABC where a + b = 10 and cos C is a root of 2x^2 - 3x - 2 = 0,
    prove that the minimum perimeter of the triangle is 10 + 5√3 -/
theorem min_perimeter_triangle (a b c : ℝ) (C : ℝ) :
  a + b = 10 →
  2 * (Real.cos C)^2 - 3 * (Real.cos C) - 2 = 0 →
  ∃ (p : ℝ), p = a + b + c ∧ p ≥ 10 + 5 * Real.sqrt 3 ∧
  ∀ (a' b' c' : ℝ), a' + b' = 10 →
    2 * (Real.cos C)^2 - 3 * (Real.cos C) - 2 = 0 →
    a' + b' + c' ≥ p :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3848_384818


namespace NUMINAMATH_CALUDE_kho_kho_problem_l3848_384896

/-- Represents the number of students who left to play kho-kho -/
def students_who_left (initial_boys initial_girls remaining_girls : ℕ) : ℕ :=
  initial_girls - remaining_girls

/-- Proves that 8 girls left to play kho-kho given the problem conditions -/
theorem kho_kho_problem (initial_boys initial_girls remaining_girls : ℕ) :
  initial_boys = initial_girls →
  initial_boys + initial_girls = 32 →
  initial_boys = 2 * remaining_girls →
  students_who_left initial_boys initial_girls remaining_girls = 8 :=
by
  sorry

#check kho_kho_problem

end NUMINAMATH_CALUDE_kho_kho_problem_l3848_384896


namespace NUMINAMATH_CALUDE_abs_2x_minus_5_l3848_384872

theorem abs_2x_minus_5 (x : ℝ) (h : |2*x - 3| - 3 + 2*x = 0) : |2*x - 5| = 5 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_minus_5_l3848_384872


namespace NUMINAMATH_CALUDE_power_difference_mod_six_l3848_384887

theorem power_difference_mod_six :
  (47^2045 - 18^2045) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_mod_six_l3848_384887


namespace NUMINAMATH_CALUDE_max_notebooks_is_14_l3848_384805

/-- Represents the pricing options for notebooks -/
structure NotebookPricing where
  single_price : ℕ
  pack3_price : ℕ
  pack7_price : ℕ

/-- Calculates the maximum number of notebooks that can be bought with a given budget and pricing -/
def max_notebooks (budget : ℕ) (pricing : NotebookPricing) : ℕ :=
  sorry

/-- The specific pricing and budget from the problem -/
def problem_pricing : NotebookPricing :=
  { single_price := 2
  , pack3_price := 5
  , pack7_price := 10 }

def problem_budget : ℕ := 20

/-- Theorem stating that the maximum number of notebooks that can be bought is 14 -/
theorem max_notebooks_is_14 : 
  max_notebooks problem_budget problem_pricing = 14 := by sorry

end NUMINAMATH_CALUDE_max_notebooks_is_14_l3848_384805


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l3848_384802

theorem imaginary_part_of_1_minus_2i :
  Complex.im (1 - 2 * Complex.I) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l3848_384802


namespace NUMINAMATH_CALUDE_function_lower_bound_l3848_384894

theorem function_lower_bound (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, |x - 1/a| + |x + a| ≥ 2 := by sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3848_384894


namespace NUMINAMATH_CALUDE_scaled_model_height_l3848_384865

/-- Represents a cylindrical monument --/
structure CylindricalMonument where
  height : ℝ
  baseRadius : ℝ
  volume : ℝ

/-- Represents a scaled model of the monument --/
structure ScaledModel where
  volume : ℝ
  height : ℝ

/-- Theorem stating the relationship between the original monument and its scaled model --/
theorem scaled_model_height 
  (monument : CylindricalMonument) 
  (model : ScaledModel) : 
  monument.height = 100 ∧ 
  monument.baseRadius = 20 ∧ 
  monument.volume = 125600 ∧ 
  model.volume = 1.256 → 
  model.height = 1 := by
  sorry


end NUMINAMATH_CALUDE_scaled_model_height_l3848_384865


namespace NUMINAMATH_CALUDE_conditional_prob_is_two_thirds_l3848_384821

/-- The sample space for two coin flips -/
def S : Finset (Fin 2 × Fin 2) := Finset.univ

/-- Event A: at least one tail shows up -/
def A : Finset (Fin 2 × Fin 2) := {(0, 1), (1, 0), (1, 1)}

/-- Event B: exactly one head shows up -/
def B : Finset (Fin 2 × Fin 2) := {(0, 1), (1, 0)}

/-- The probability measure for the sample space -/
def P (E : Finset (Fin 2 × Fin 2)) : ℚ := (E.card : ℚ) / (S.card : ℚ)

/-- The conditional probability of B given A -/
def conditional_prob : ℚ := P (A ∩ B) / P A

theorem conditional_prob_is_two_thirds : conditional_prob = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_prob_is_two_thirds_l3848_384821


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3848_384810

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3848_384810


namespace NUMINAMATH_CALUDE_problem_statement_l3848_384813

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (3 * x + 1)) :
  (x - y + 3 * x * y) / (x * y) = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3848_384813


namespace NUMINAMATH_CALUDE_unique_solution_implies_coefficients_l3848_384882

theorem unique_solution_implies_coefficients
  (a b : ℚ)
  (h1 : ∀ x y : ℚ, a * x + y = 2 ∧ x + b * y = 2 ↔ x = 2 ∧ y = 1) :
  a = 1/2 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_implies_coefficients_l3848_384882


namespace NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l3848_384869

theorem log_50_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < Real.log 50 / Real.log 10) ∧ (Real.log 50 / Real.log 10 < b) ∧ (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l3848_384869


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3848_384831

theorem sqrt_fraction_simplification :
  Real.sqrt ((25 : ℝ) / 49 + (16 : ℝ) / 81) = (53 : ℝ) / 63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3848_384831


namespace NUMINAMATH_CALUDE_apps_files_difference_l3848_384853

/-- Represents the contents of Dave's phone -/
structure PhoneContents where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone -/
def initial : PhoneContents := { apps := 24, files := 9 }

/-- The final state of Dave's phone -/
def final : PhoneContents := { apps := 12, files := 5 }

/-- The theorem stating the difference between apps and files in the final state -/
theorem apps_files_difference : final.apps - final.files = 7 := by
  sorry

end NUMINAMATH_CALUDE_apps_files_difference_l3848_384853


namespace NUMINAMATH_CALUDE_circumcircle_equation_correct_l3848_384809

/-- The circumcircle of a triangle AOB, where O is the origin (0, 0), A is at (4, 0), and B is at (0, 3) --/
def CircumcircleAOB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 3*p.2 = 0}

/-- Point O is the origin --/
def O : ℝ × ℝ := (0, 0)

/-- Point A has coordinates (4, 0) --/
def A : ℝ × ℝ := (4, 0)

/-- Point B has coordinates (0, 3) --/
def B : ℝ × ℝ := (0, 3)

/-- The circumcircle equation is correct for the given triangle AOB --/
theorem circumcircle_equation_correct :
  O ∈ CircumcircleAOB ∧ A ∈ CircumcircleAOB ∧ B ∈ CircumcircleAOB :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_correct_l3848_384809


namespace NUMINAMATH_CALUDE_age_difference_is_ten_l3848_384840

/-- The age difference between Declan's elder son and younger son -/
def age_difference : ℕ → ℕ → ℕ
  | elder_age, younger_age => elder_age - younger_age

/-- The current age of Declan's elder son -/
def elder_son_age : ℕ := 40

/-- The age of Declan's younger son 30 years from now -/
def younger_son_future_age : ℕ := 60

/-- The number of years in the future when the younger son's age is known -/
def years_in_future : ℕ := 30

theorem age_difference_is_ten :
  age_difference elder_son_age (younger_son_future_age - years_in_future) = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_ten_l3848_384840


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l3848_384881

theorem intersection_empty_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - x - 6 > 0}
  let B : Set ℝ := {x | (x - m) * (x - 2*m) ≤ 0}
  A ∩ B = ∅ → m ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l3848_384881


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3848_384804

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given conditions for the geometric sequence -/
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 = 10 ∧ a 2 + a 4 = 5

theorem geometric_sequence_fifth_term (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_cond : sequence_conditions a) : 
  a 5 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3848_384804


namespace NUMINAMATH_CALUDE_sin_equality_l3848_384875

theorem sin_equality (x : ℝ) (h : Real.sin (x + π/4) = 1/3) :
  Real.sin (4*x) - 2 * Real.cos (3*x) * Real.sin x = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_l3848_384875


namespace NUMINAMATH_CALUDE_allen_pizza_payment_l3848_384803

theorem allen_pizza_payment (num_boxes : ℕ) (cost_per_box : ℚ) (tip_fraction : ℚ) (change_received : ℚ) :
  num_boxes = 5 →
  cost_per_box = 7 →
  tip_fraction = 1 / 7 →
  change_received = 60 →
  let total_cost := num_boxes * cost_per_box
  let tip := tip_fraction * total_cost
  let total_paid := total_cost + tip
  let money_given := total_paid + change_received
  money_given = 100 := by
  sorry

end NUMINAMATH_CALUDE_allen_pizza_payment_l3848_384803


namespace NUMINAMATH_CALUDE_cot_150_degrees_l3848_384806

theorem cot_150_degrees : Real.cos (150 * π / 180) / Real.sin (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_150_degrees_l3848_384806


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3848_384845

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3848_384845


namespace NUMINAMATH_CALUDE_mothers_age_twice_lucys_l3848_384859

/-- Given Lucy's age and her mother's age in 2012, find the year when the mother's age will be twice Lucy's age -/
theorem mothers_age_twice_lucys (lucy_age_2012 : ℕ) (mother_age_multiplier : ℕ) : 
  lucy_age_2012 = 10 →
  mother_age_multiplier = 5 →
  ∃ (years_after_2012 : ℕ),
    (lucy_age_2012 + years_after_2012) * 2 = (lucy_age_2012 * mother_age_multiplier + years_after_2012) ∧
    2012 + years_after_2012 = 2042 :=
by sorry

end NUMINAMATH_CALUDE_mothers_age_twice_lucys_l3848_384859


namespace NUMINAMATH_CALUDE_equipment_theorem_l3848_384864

/-- Represents the sales data for equipment A and B -/
structure SalesData where
  a : ℕ  -- quantity of A
  b : ℕ  -- quantity of B
  total : ℕ  -- total amount in yuan

/-- Represents the problem setup -/
structure EquipmentProblem where
  sale1 : SalesData
  sale2 : SalesData
  totalPieces : ℕ
  maxRatio : ℕ  -- max ratio of A to B
  maxCost : ℕ

/-- The main theorem to prove -/
theorem equipment_theorem (p : EquipmentProblem) 
  (h1 : p.sale1 = ⟨20, 10, 1100⟩)
  (h2 : p.sale2 = ⟨25, 20, 1750⟩)
  (h3 : p.totalPieces = 50)
  (h4 : p.maxRatio = 2)
  (h5 : p.maxCost = 2000) :
  ∃ (priceA priceB : ℕ),
    priceA = 30 ∧ 
    priceB = 50 ∧ 
    (∃ (validPlans : Finset (ℕ × ℕ)),
      validPlans.card = 9 ∧
      ∀ (plan : ℕ × ℕ), plan ∈ validPlans ↔ 
        (plan.1 + plan.2 = p.totalPieces ∧
         plan.1 ≤ p.maxRatio * plan.2 ∧
         plan.1 * priceA + plan.2 * priceB ≤ p.maxCost)) :=
by sorry

end NUMINAMATH_CALUDE_equipment_theorem_l3848_384864


namespace NUMINAMATH_CALUDE_madeline_and_brother_total_l3848_384889

/-- Given Madeline has $48 and her brother has half as much, prove that they have $72 together. -/
theorem madeline_and_brother_total (madeline_amount : ℕ) (brother_amount : ℕ) : 
  madeline_amount = 48 → 
  brother_amount = madeline_amount / 2 → 
  madeline_amount + brother_amount = 72 := by
sorry

end NUMINAMATH_CALUDE_madeline_and_brother_total_l3848_384889


namespace NUMINAMATH_CALUDE_box_height_proof_l3848_384814

/-- Proves that a box with given dimensions and cube requirements has a specific height -/
theorem box_height_proof (length width : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) (height : ℝ) : 
  length = 10 →
  width = 13 →
  cube_volume = 5 →
  min_cubes = 130 →
  height = (min_cubes : ℝ) * cube_volume / (length * width) →
  height = 5 := by
sorry

end NUMINAMATH_CALUDE_box_height_proof_l3848_384814


namespace NUMINAMATH_CALUDE_train_length_l3848_384842

/-- Given a train traveling at 72 km/hr that crosses a pole in 9 seconds, prove that its length is 180 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 72 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = (speed * 1000 / 3600) * time →
  length = 180 := by sorry

end NUMINAMATH_CALUDE_train_length_l3848_384842


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_for_f_geq_a_squared_minus_a_l3848_384846

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for part I
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_geq_a_squared_minus_a :
  {a : ℝ | ∀ x : ℝ, f x ≥ a^2 - a} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_for_f_geq_a_squared_minus_a_l3848_384846


namespace NUMINAMATH_CALUDE_x_value_after_z_doubled_l3848_384870

theorem x_value_after_z_doubled (x y z_original z_doubled : ℚ) : 
  x = (1 / 3) * y →
  y = (1 / 4) * z_doubled →
  z_original = 48 →
  z_doubled = 2 * z_original →
  x = 8 := by sorry

end NUMINAMATH_CALUDE_x_value_after_z_doubled_l3848_384870


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3848_384830

/-- A hyperbola with a focus on the y-axis and asymptotic lines y = ± (√5/2)x has eccentricity 3√5/5 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2 * a = Real.sqrt 5 * b) →
  (Real.sqrt ((a^2 + b^2) / a^2) = 3 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3848_384830


namespace NUMINAMATH_CALUDE_count_eight_digit_numbers_seven_different_is_correct_l3848_384890

/-- The number of 8-digit numbers where exactly 7 digits are all different -/
def count_eight_digit_numbers_seven_different : ℕ := 5080320

/-- Theorem stating that the count of 8-digit numbers where exactly 7 digits are all different is 5080320 -/
theorem count_eight_digit_numbers_seven_different_is_correct :
  count_eight_digit_numbers_seven_different = 5080320 := by sorry

end NUMINAMATH_CALUDE_count_eight_digit_numbers_seven_different_is_correct_l3848_384890


namespace NUMINAMATH_CALUDE_equivalent_discount_l3848_384815

/-- Proves that a single discount of 32.5% before taxes is equivalent to a series of discounts
    (25% followed by 10%) and a 5% sales tax, given an original price of $50. -/
theorem equivalent_discount (original_price : ℝ) (first_discount second_discount tax : ℝ)
  (single_discount : ℝ) :
  original_price = 50 →
  first_discount = 0.25 →
  second_discount = 0.10 →
  tax = 0.05 →
  single_discount = 0.325 →
  original_price * (1 - single_discount) * (1 + tax) =
  original_price * (1 - first_discount) * (1 - second_discount) * (1 + tax) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l3848_384815


namespace NUMINAMATH_CALUDE_sufficient_condition_l3848_384863

theorem sufficient_condition (x y : ℝ) : x > 3 ∧ y > 3 → x + y > 6 ∧ x * y > 9 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_l3848_384863


namespace NUMINAMATH_CALUDE_marina_extra_parks_l3848_384825

/-- The number of theme parks in Jamestown -/
def jamestown_parks : ℕ := 20

/-- The number of additional theme parks Venice has compared to Jamestown -/
def venice_extra_parks : ℕ := 25

/-- The total number of theme parks in all three towns -/
def total_parks : ℕ := 135

/-- The number of theme parks in Venice -/
def venice_parks : ℕ := jamestown_parks + venice_extra_parks

/-- The number of theme parks in Marina Del Ray -/
def marina_parks : ℕ := total_parks - (jamestown_parks + venice_parks)

/-- The difference in theme parks between Marina Del Ray and Jamestown -/
def marina_jamestown_difference : ℕ := marina_parks - jamestown_parks

theorem marina_extra_parks :
  marina_jamestown_difference = 50 := by sorry

end NUMINAMATH_CALUDE_marina_extra_parks_l3848_384825


namespace NUMINAMATH_CALUDE_original_number_proof_l3848_384861

theorem original_number_proof (h1 : 213 * 16 = 3408) 
  (h2 : ∃ x, x * 21.3 = 34.080000000000005) : 
  ∃ x, x * 21.3 = 34.080000000000005 ∧ x = 1.6 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l3848_384861


namespace NUMINAMATH_CALUDE_magnification_factor_l3848_384885

theorem magnification_factor (magnified_diameter actual_diameter : ℝ) 
  (h1 : magnified_diameter = 0.2)
  (h2 : actual_diameter = 0.0002) :
  magnified_diameter / actual_diameter = 1000 := by
sorry

end NUMINAMATH_CALUDE_magnification_factor_l3848_384885


namespace NUMINAMATH_CALUDE_sodium_chloride_percentage_l3848_384807

theorem sodium_chloride_percentage
  (tank_capacity : ℝ)
  (fill_ratio : ℝ)
  (evaporation_rate : ℝ)
  (time : ℝ)
  (final_water_concentration : ℝ)
  (h1 : tank_capacity = 24)
  (h2 : fill_ratio = 1/4)
  (h3 : evaporation_rate = 0.4)
  (h4 : time = 6)
  (h5 : final_water_concentration = 1/2) :
  let initial_volume := tank_capacity * fill_ratio
  let evaporated_water := evaporation_rate * time
  let final_volume := initial_volume - evaporated_water
  let initial_sodium_chloride_percentage := 
    100 * (initial_volume - (final_volume * final_water_concentration)) / initial_volume
  initial_sodium_chloride_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_sodium_chloride_percentage_l3848_384807


namespace NUMINAMATH_CALUDE_quadratic_root_sum_squares_l3848_384834

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 + 4 * h * x + 6 = 0 ∧ 
               2 * y^2 + 4 * h * y + 6 = 0 ∧ 
               x^2 + y^2 = 34) → 
  |h| = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_squares_l3848_384834


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3848_384837

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (p : ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, a (n + 1) = p * a n + 2^n) →
  geometric_sequence a →
  ∀ n : ℕ, a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3848_384837


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3848_384851

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

-- Define the conditions of the problem
def problem_ellipse : Ellipse :=
  { center := (5, 2)
  , a := 5
  , b := 2 }

-- Define the point that the ellipse passes through
def point_on_ellipse : ℝ × ℝ := (3, 1)

-- Theorem statement
theorem ellipse_foci_distance :
  let e := problem_ellipse
  let (x, y) := point_on_ellipse
  let (cx, cy) := e.center
  (((x - cx) / e.a) ^ 2 + ((y - cy) / e.b) ^ 2 ≤ 1) →
  (2 * Real.sqrt (e.a ^ 2 - e.b ^ 2) = 2 * Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3848_384851


namespace NUMINAMATH_CALUDE_binary_1101110_equals_3131_base4_l3848_384867

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The binary representation of 1101110 -/
def binary_1101110 : List Bool := [true, true, false, true, true, true, false]

theorem binary_1101110_equals_3131_base4 :
  decimal_to_base4 (binary_to_decimal binary_1101110) = [3, 1, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101110_equals_3131_base4_l3848_384867


namespace NUMINAMATH_CALUDE_original_car_cost_l3848_384868

/-- Proves that the original cost of a car is 39200 given the specified conditions -/
theorem original_car_cost (C : ℝ) : 
  C > 0 →  -- Ensure the cost is positive
  (68400 - (C + 8000)) / C * 100 = 54.054054054054056 →
  C = 39200 := by
  sorry

end NUMINAMATH_CALUDE_original_car_cost_l3848_384868


namespace NUMINAMATH_CALUDE_sum_of_prime_divisors_of_N_l3848_384841

/-- The number of ways to choose a committee from 11 men and 12 women,
    where the number of women is always one more than the number of men. -/
def N : ℕ := (Finset.range 12).sum (λ k => Nat.choose 11 k * Nat.choose 12 (k + 1))

/-- The sum of prime numbers that divide N -/
def sum_of_prime_divisors (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).sum (λ p => if p ∣ n then p else 0)

theorem sum_of_prime_divisors_of_N : sum_of_prime_divisors N = 79 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_divisors_of_N_l3848_384841


namespace NUMINAMATH_CALUDE_heptagon_exterior_angle_sum_l3848_384812

/-- The exterior angle sum of a heptagon is 360 degrees. -/
theorem heptagon_exterior_angle_sum : ℝ :=
  360

#check heptagon_exterior_angle_sum

end NUMINAMATH_CALUDE_heptagon_exterior_angle_sum_l3848_384812


namespace NUMINAMATH_CALUDE_max_point_condition_l3848_384874

/-- The function f(x) defined as (x-a)^2 * (x-1) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - a)^2 * (x - 1)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x - a) * (3*x - a - 2)

theorem max_point_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_max_point_condition_l3848_384874


namespace NUMINAMATH_CALUDE_big_dig_nickel_output_l3848_384839

/-- Represents the daily mining output of Big Dig Mining Company -/
structure MiningOutput where
  copper : ℝ
  iron : ℝ
  nickel : ℝ

/-- Calculates the total daily output -/
def totalOutput (output : MiningOutput) : ℝ :=
  output.copper + output.iron + output.nickel

theorem big_dig_nickel_output :
  ∀ output : MiningOutput,
  output.copper = 360 ∧
  output.iron = 0.6 * totalOutput output ∧
  output.nickel = 0.1 * totalOutput output →
  output.nickel = 120 := by
sorry


end NUMINAMATH_CALUDE_big_dig_nickel_output_l3848_384839


namespace NUMINAMATH_CALUDE_alex_pictures_l3848_384801

/-- The number of pictures Alex has, given processing time per picture and total processing time. -/
def number_of_pictures (minutes_per_picture : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours * 60) / minutes_per_picture

/-- Theorem stating that Alex has 960 pictures. -/
theorem alex_pictures : number_of_pictures 2 32 = 960 := by
  sorry

end NUMINAMATH_CALUDE_alex_pictures_l3848_384801


namespace NUMINAMATH_CALUDE_anniversary_sale_cost_l3848_384800

def original_ice_cream_price : ℚ := 12
def ice_cream_discount : ℚ := 2
def juice_price_per_5_cans : ℚ := 2
def ice_cream_tubs : ℕ := 2
def juice_cans : ℕ := 10

theorem anniversary_sale_cost : 
  (ice_cream_tubs * (original_ice_cream_price - ice_cream_discount)) + 
  (juice_cans / 5 * juice_price_per_5_cans) = 24 := by
  sorry

end NUMINAMATH_CALUDE_anniversary_sale_cost_l3848_384800


namespace NUMINAMATH_CALUDE_quadratic_solution_for_b_l3848_384852

theorem quadratic_solution_for_b (a b c m : ℝ) (h1 : m = c * a * (b - 1) / (a - b^2)) 
  (h2 : c * a ≠ 0) : m * b^2 + c * a * b - m * a - c * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_for_b_l3848_384852


namespace NUMINAMATH_CALUDE_randy_initial_amount_l3848_384850

def initial_amount (spend_per_visit : ℕ) (visits_per_month : ℕ) (months : ℕ) (remaining : ℕ) : ℕ :=
  spend_per_visit * visits_per_month * months + remaining

theorem randy_initial_amount :
  initial_amount 2 4 12 104 = 200 :=
by sorry

end NUMINAMATH_CALUDE_randy_initial_amount_l3848_384850


namespace NUMINAMATH_CALUDE_infinite_primes_4n_plus_3_l3848_384877

theorem infinite_primes_4n_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ p % 4 = 3) →
  ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_4n_plus_3_l3848_384877


namespace NUMINAMATH_CALUDE_det_A_equals_l3848_384856

-- Define the matrix as a function of y
def A (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![y^2 + 1, 2*y, 2*y;
     2*y, y^2 + 3, 2*y;
     2*y, 2*y, y^2 + 5]

-- State the theorem
theorem det_A_equals (y : ℝ) : 
  Matrix.det (A y) = y^6 + y^4 + 35*y^2 + 15 - 32*y := by
  sorry

end NUMINAMATH_CALUDE_det_A_equals_l3848_384856


namespace NUMINAMATH_CALUDE_wage_cut_and_raise_l3848_384895

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let cut_wage := 0.7 * original_wage
  let required_raise := (original_wage / cut_wage) - 1
  ∃ ε > 0, abs (required_raise - 0.4286) < ε :=
by sorry

end NUMINAMATH_CALUDE_wage_cut_and_raise_l3848_384895
