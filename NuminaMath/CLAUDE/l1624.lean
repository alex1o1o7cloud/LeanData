import Mathlib

namespace NUMINAMATH_CALUDE_initial_distance_is_54km_l1624_162412

/-- Represents the cycling scenario described in the problem -/
structure CyclingScenario where
  v : ℝ  -- Initial speed in km/h
  t : ℝ  -- Time shown on cycle computer in hours
  d : ℝ  -- Initial distance from home in km

/-- The conditions of the cycling scenario -/
def scenario_conditions (s : CyclingScenario) : Prop :=
  s.d = s.v * s.t ∧  -- Initial condition
  s.d = (2/3 * s.v) + (s.v - 1) * s.t ∧  -- After first speed change
  s.d = (2/3 * s.v) + (3/4 * (s.v - 1)) + (s.v - 2) * s.t  -- After second speed change

/-- The theorem stating that the initial distance is 54 km -/
theorem initial_distance_is_54km (s : CyclingScenario) 
  (h : scenario_conditions s) : s.d = 54 := by
  sorry

#check initial_distance_is_54km

end NUMINAMATH_CALUDE_initial_distance_is_54km_l1624_162412


namespace NUMINAMATH_CALUDE_vector_relations_l1624_162482

def a : Fin 3 → ℝ
| 0 => 2
| 1 => -1
| 2 => 3
| _ => 0

def b (x : ℝ) : Fin 3 → ℝ
| 0 => -4
| 1 => 2
| 2 => x
| _ => 0

theorem vector_relations (x : ℝ) :
  (∀ i : Fin 3, (a i) * (b x i) = 0 → x = 10/3) ∧
  (∃ k : ℝ, ∀ i : Fin 3, (a i) = k * (b x i) → x = -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l1624_162482


namespace NUMINAMATH_CALUDE_strictly_increasing_derivative_properties_l1624_162470

/-- A function with a strictly increasing derivative -/
structure StrictlyIncreasingDerivative (f : ℝ → ℝ) : Prop where
  deriv_increasing : ∀ x y, x < y → (deriv f x) < (deriv f y)

/-- The main theorem -/
theorem strictly_increasing_derivative_properties
  (f : ℝ → ℝ) (hf : StrictlyIncreasingDerivative f) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ↔ f (x₁ + 1) + f x₂ > f x₁ + f (x₂ + 1)) ∧
  (StrictMono f ↔ ∀ x : ℝ, x < 0 → f x < f 0) :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_derivative_properties_l1624_162470


namespace NUMINAMATH_CALUDE_transmission_time_is_6_4_minutes_l1624_162474

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 80

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 768

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- Represents the total number of chunks to be sent -/
def total_chunks : ℕ := num_blocks * chunks_per_block

/-- Represents the time in seconds to send all chunks -/
def transmission_time_seconds : ℚ := total_chunks / transmission_rate

/-- Represents the time in minutes to send all chunks -/
def transmission_time_minutes : ℚ := transmission_time_seconds / 60

/-- Theorem stating that the transmission time is 6.4 minutes -/
theorem transmission_time_is_6_4_minutes : transmission_time_minutes = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_6_4_minutes_l1624_162474


namespace NUMINAMATH_CALUDE_problem_statement_l1624_162466

theorem problem_statement (x : ℝ) (y : ℝ) (h_y_pos : y > 0) : 
  let A : Set ℝ := {x^2 + x + 1, -x, -x - 1}
  let B : Set ℝ := {-y, -y/2, y + 1}
  A = B → x^2 + y^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1624_162466


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1624_162409

theorem binomial_coefficient_n_minus_two (n : ℕ) (h : n ≥ 2) :
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1624_162409


namespace NUMINAMATH_CALUDE_max_value_inequality_l1624_162495

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b^2 * c^2 * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1624_162495


namespace NUMINAMATH_CALUDE_fifth_number_in_specific_pascal_row_l1624_162463

/-- Represents a row in Pascal's triangle -/
def PascalRow (n : ℕ) := Fin (n + 1) → ℕ

/-- The nth row of Pascal's triangle -/
def nthPascalRow (n : ℕ) : PascalRow n := 
  fun k => Nat.choose n k.val

/-- The condition that a row starts with 1 and then 15 -/
def startsWithOneAndFifteen (row : PascalRow 15) : Prop :=
  row 0 = 1 ∧ row 1 = 15

theorem fifth_number_in_specific_pascal_row : 
  ∀ (row : PascalRow 15), 
    startsWithOneAndFifteen row → 
    row 4 = Nat.choose 15 4 ∧ 
    Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_specific_pascal_row_l1624_162463


namespace NUMINAMATH_CALUDE_minimum_orange_chips_l1624_162485

theorem minimum_orange_chips 
  (purple green orange : ℕ) 
  (h1 : green ≥ purple / 3)
  (h2 : green ≤ orange / 4)
  (h3 : purple + green ≥ 75) :
  orange ≥ 76 := by
  sorry

end NUMINAMATH_CALUDE_minimum_orange_chips_l1624_162485


namespace NUMINAMATH_CALUDE_red_exhausted_first_l1624_162417

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability that red marbles are the first to be exhausted -/
def probability_red_exhausted (bag : MarbleBag) : ℚ :=
  sorry

/-- The theorem stating the probability of red marbles being exhausted first -/
theorem red_exhausted_first (bag : MarbleBag) 
  (h1 : bag.red = 3) 
  (h2 : bag.blue = 5) 
  (h3 : bag.green = 7) : 
  probability_red_exhausted bag = 21 / 40 := by
  sorry

end NUMINAMATH_CALUDE_red_exhausted_first_l1624_162417


namespace NUMINAMATH_CALUDE_no_solution_iff_a_eq_one_or_neg_two_l1624_162456

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- Theorem statement
theorem no_solution_iff_a_eq_one_or_neg_two (a : ℝ) :
  (∀ x : ℝ, ¬ equation a x) ↔ (a = 1 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_eq_one_or_neg_two_l1624_162456


namespace NUMINAMATH_CALUDE_jewelry_ratio_l1624_162476

def jewelry_problem (initial_necklaces initial_earrings bought_necklaces total_jewelry : ℕ) : Prop :=
  let bought_earrings := (2 * initial_earrings) / 3
  let total_before_mother := initial_necklaces + initial_earrings + bought_necklaces + bought_earrings
  let mother_earrings := total_jewelry - total_before_mother
  (mother_earrings : ℚ) / bought_earrings = 6 / 5

theorem jewelry_ratio :
  jewelry_problem 10 15 10 57 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_ratio_l1624_162476


namespace NUMINAMATH_CALUDE_savings_calculation_l1624_162414

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) 
  (h1 : income = 16000)
  (h2 : ratio_income = 5)
  (h3 : ratio_expenditure = 4) :
  income - (income * ratio_expenditure / ratio_income) = 3200 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1624_162414


namespace NUMINAMATH_CALUDE_grade12_selection_l1624_162411

/-- Represents the number of students selected from each grade -/
structure GradeSelection where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the ratio of students in grades 10, 11, and 12 -/
structure GradeRatio where
  k : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Theorem: Given the conditions, prove that 360 students were selected from grade 12 -/
theorem grade12_selection
  (total_sample : ℕ)
  (ratio : GradeRatio)
  (selection : GradeSelection)
  (h1 : total_sample = 1200)
  (h2 : ratio = { k := 2, grade11 := 5, grade12 := 3 })
  (h3 : selection.grade10 = 240)
  (h4 : selection.grade10 + selection.grade11 + selection.grade12 = total_sample)
  (h5 : selection.grade10 * (ratio.k + ratio.grade11 + ratio.grade12) = 
        total_sample * ratio.k) :
  selection.grade12 = 360 := by
  sorry


end NUMINAMATH_CALUDE_grade12_selection_l1624_162411


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1624_162489

/-- A parabola with its focus on the line 3x - 4y - 12 = 0 -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_line : 3 * focus.1 - 4 * focus.2 - 12 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation
  | VerticalAxis (p : ℝ) : StandardEquation  -- y² = 4px
  | HorizontalAxis (p : ℝ) : StandardEquation  -- x² = 4py

theorem parabola_standard_equation (p : Parabola) :
  (∃ (eq : StandardEquation), eq = StandardEquation.VerticalAxis 4 ∨ eq = StandardEquation.HorizontalAxis (-3)) :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1624_162489


namespace NUMINAMATH_CALUDE_milk_cost_l1624_162449

/-- The cost of a gallon of milk given the following conditions:
  * 4 pounds of coffee beans and 2 gallons of milk were bought
  * A pound of coffee beans costs $2.50
  * The total cost is $17
-/
theorem milk_cost (coffee_pounds : ℕ) (milk_gallons : ℕ) 
  (coffee_price : ℚ) (total_cost : ℚ) :
  coffee_pounds = 4 →
  milk_gallons = 2 →
  coffee_price = 5/2 →
  total_cost = 17 →
  ∃ (milk_price : ℚ), 
    milk_price * milk_gallons + coffee_price * coffee_pounds = total_cost ∧
    milk_price = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_milk_cost_l1624_162449


namespace NUMINAMATH_CALUDE_min_value_a_l1624_162497

theorem min_value_a (p : Prop) (h : ¬∀ x > 0, a < x + 1/x) : 
  ∃ a : ℝ, (∀ b : ℝ, (∃ x > 0, b ≥ x + 1/x) → a ≤ b) ∧ (∃ x > 0, a ≥ x + 1/x) ∧ a = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1624_162497


namespace NUMINAMATH_CALUDE_unique_solution_l1624_162436

theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ 5^29 * x^15 = 2 * 10^29 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1624_162436


namespace NUMINAMATH_CALUDE_pythagorean_triple_odd_l1624_162402

theorem pythagorean_triple_odd (a : ℕ) (h1 : a ≥ 3) (h2 : Odd a) :
  a^2 + ((a^2 - 1) / 2)^2 = ((a^2 + 1) / 2)^2 := by
  sorry

#check pythagorean_triple_odd

end NUMINAMATH_CALUDE_pythagorean_triple_odd_l1624_162402


namespace NUMINAMATH_CALUDE_tangency_point_satisfies_equations_unique_tangency_point_l1624_162475

/-- The point of tangency for two parabolas -/
def point_of_tangency : ℝ × ℝ := (-7, -24)

/-- The first parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32

/-- The second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  parabola1 point_of_tangency.1 point_of_tangency.2 ∧
  parabola2 point_of_tangency.1 point_of_tangency.2 := by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem unique_tangency_point :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency := by sorry

end NUMINAMATH_CALUDE_tangency_point_satisfies_equations_unique_tangency_point_l1624_162475


namespace NUMINAMATH_CALUDE_dan_catches_cate_l1624_162499

/-- The time it takes for Dan to catch Cate given their initial distance and speeds -/
theorem dan_catches_cate (initial_distance : ℝ) (dan_speed : ℝ) (cate_speed : ℝ)
  (h1 : initial_distance = 50)
  (h2 : dan_speed = 8)
  (h3 : cate_speed = 6)
  (h4 : dan_speed > cate_speed) :
  (initial_distance / (dan_speed - cate_speed)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_dan_catches_cate_l1624_162499


namespace NUMINAMATH_CALUDE_first_player_wins_first_player_wins_modified_l1624_162408

/-- Represents the state of the game with two piles of stones -/
structure GameState :=
  (pile1 : Nat)
  (pile2 : Nat)

/-- Represents a move in the game -/
inductive Move
  | TakeFromFirst
  | TakeFromSecond
  | TakeFromBoth
  | TransferToSecond

/-- Defines if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.TakeFromFirst => state.pile1 > 0
  | Move.TakeFromSecond => state.pile2 > 0
  | Move.TakeFromBoth => state.pile1 > 0 && state.pile2 > 0
  | Move.TransferToSecond => state.pile1 > 0

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeFromFirst => ⟨state.pile1 - 1, state.pile2⟩
  | Move.TakeFromSecond => ⟨state.pile1, state.pile2 - 1⟩
  | Move.TakeFromBoth => ⟨state.pile1 - 1, state.pile2 - 1⟩
  | Move.TransferToSecond => ⟨state.pile1 - 1, state.pile2 + 1⟩

/-- Determines if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Bool :=
  state.pile1 = 0 && state.pile2 = 0

/-- Theorem: The first player has a winning strategy in the two-pile stone game -/
theorem first_player_wins (initialState : GameState) 
  (h : initialState = ⟨7, 7⟩) : 
  ∃ (strategy : GameState → Move), 
    ∀ (opponentMove : Move), 
      isValidMove initialState (strategy initialState) ∧ 
      ¬isGameOver (applyMove initialState (strategy initialState)) ∧
      isGameOver (applyMove (applyMove initialState (strategy initialState)) opponentMove) :=
sorry

/-- Theorem: The first player has a winning strategy in the modified two-pile stone game -/
theorem first_player_wins_modified (initialState : GameState) 
  (h : initialState = ⟨7, 7⟩) : 
  ∃ (strategy : GameState → Move), 
    ∀ (opponentMove : Move), 
      isValidMove initialState (strategy initialState) ∧ 
      ¬isGameOver (applyMove initialState (strategy initialState)) ∧
      isGameOver (applyMove (applyMove initialState (strategy initialState)) opponentMove) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_first_player_wins_modified_l1624_162408


namespace NUMINAMATH_CALUDE_total_timeout_time_is_185_l1624_162491

/-- Calculates the total time spent in time-out given the number of running time-outs and the duration of each time-out. -/
def total_timeout_time (running_timeouts : ℕ) (timeout_duration : ℕ) : ℕ :=
  let food_throwing_timeouts := 5 * running_timeouts - 1
  let swearing_timeouts := food_throwing_timeouts / 3
  let total_timeouts := running_timeouts + food_throwing_timeouts + swearing_timeouts
  total_timeouts * timeout_duration

/-- Proves that the total time spent in time-out is 185 minutes given the specified conditions. -/
theorem total_timeout_time_is_185 : 
  total_timeout_time 5 5 = 185 := by
  sorry

#eval total_timeout_time 5 5

end NUMINAMATH_CALUDE_total_timeout_time_is_185_l1624_162491


namespace NUMINAMATH_CALUDE_replacement_concentration_l1624_162492

/-- Proves that the concentration of the replacing solution is 25% given the initial and final conditions --/
theorem replacement_concentration (initial_concentration : ℝ) (final_concentration : ℝ) (replaced_fraction : ℝ) :
  initial_concentration = 0.40 →
  final_concentration = 0.35 →
  replaced_fraction = 1/3 →
  (1 - replaced_fraction) * initial_concentration + replaced_fraction * final_concentration = final_concentration →
  replaced_fraction * (final_concentration - initial_concentration) / replaced_fraction = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_replacement_concentration_l1624_162492


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1624_162416

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1624_162416


namespace NUMINAMATH_CALUDE_expansion_sum_l1624_162432

-- Define the sum of coefficients of the expansion
def P (n : ℕ) : ℕ := 4^n

-- Define the sum of all binomial coefficients
def S (n : ℕ) : ℕ := 2^n

-- Theorem statement
theorem expansion_sum (n : ℕ) : P n + S n = 272 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_sum_l1624_162432


namespace NUMINAMATH_CALUDE_construct_equilateral_triangle_l1624_162459

/-- A triangle with angles 80°, 50°, and 50° -/
structure DraftingTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_angles : angle1 + angle2 + angle3 = 180
  angle_values : angle1 = 80 ∧ angle2 = 50 ∧ angle3 = 50

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  all_sides_equal : side1 = side2 ∧ side2 = side3

/-- Theorem stating that an equilateral triangle can be constructed using the drafting triangle -/
theorem construct_equilateral_triangle (d : DraftingTriangle) : 
  ∃ (e : EquilateralTriangle), True := by sorry

end NUMINAMATH_CALUDE_construct_equilateral_triangle_l1624_162459


namespace NUMINAMATH_CALUDE_max_k_for_even_quadratic_min_one_l1624_162434

/-- A quadratic function f(x) = x^2 + mx + n -/
def f (m n x : ℝ) : ℝ := x^2 + m*x + n

/-- The absolute value function h(x) = |f(x)| -/
def h (m n x : ℝ) : ℝ := |f m n x|

/-- Theorem: Maximum value of k for even quadratic function with minimum 1 -/
theorem max_k_for_even_quadratic_min_one :
  ∃ (k : ℝ), k = 1/2 ∧
  ∀ (m n : ℝ),
    (∀ x, f m n (-x) = f m n x) →  -- f is even
    (∀ x, f m n x ≥ 1) →           -- minimum of f is 1
    (∃ M, M ≥ k ∧
      ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → h m n x ≤ M) →  -- max of h in [-1,1] is M ≥ k
    k ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_even_quadratic_min_one_l1624_162434


namespace NUMINAMATH_CALUDE_four_thirteenths_cycle_sum_l1624_162400

/-- Represents a repeating decimal with a two-digit cycle -/
structure RepeatingDecimal where
  whole : ℕ
  cycle : ℕ × ℕ

/-- Converts a fraction to a repeating decimal -/
def fractionToRepeatingDecimal (n d : ℕ) : RepeatingDecimal :=
  sorry

/-- Extracts the cycle digits from a repeating decimal -/
def getCycleDigits (r : RepeatingDecimal) : ℕ × ℕ :=
  r.cycle

theorem four_thirteenths_cycle_sum :
  let r := fractionToRepeatingDecimal 4 13
  let (c, d) := getCycleDigits r
  c + d = 3 := by
    sorry

end NUMINAMATH_CALUDE_four_thirteenths_cycle_sum_l1624_162400


namespace NUMINAMATH_CALUDE_group_size_is_21_l1624_162429

/-- Represents the Pinterest group --/
structure PinterestGroup where
  /-- Number of people in the group --/
  people : ℕ
  /-- Average number of pins contributed per person per day --/
  pinsPerDay : ℕ
  /-- Number of pins deleted per person per week --/
  pinsDeletedPerWeek : ℕ
  /-- Initial number of pins --/
  initialPins : ℕ
  /-- Number of pins after 4 weeks --/
  pinsAfterMonth : ℕ

/-- Calculates the number of people in the Pinterest group --/
def calculateGroupSize (group : PinterestGroup) : ℕ :=
  let netPinsPerWeek := group.pinsPerDay * 7 - group.pinsDeletedPerWeek
  let pinsAddedPerPerson := netPinsPerWeek * 4
  let totalPinsAdded := group.pinsAfterMonth - group.initialPins
  totalPinsAdded / pinsAddedPerPerson

/-- Theorem stating that the number of people in the group is 21 --/
theorem group_size_is_21 (group : PinterestGroup) 
  (h1 : group.pinsPerDay = 10)
  (h2 : group.pinsDeletedPerWeek = 5)
  (h3 : group.initialPins = 1000)
  (h4 : group.pinsAfterMonth = 6600) :
  calculateGroupSize group = 21 := by
  sorry

#eval calculateGroupSize { 
  people := 0,  -- This value doesn't matter for the calculation
  pinsPerDay := 10, 
  pinsDeletedPerWeek := 5, 
  initialPins := 1000, 
  pinsAfterMonth := 6600 
}

end NUMINAMATH_CALUDE_group_size_is_21_l1624_162429


namespace NUMINAMATH_CALUDE_trip_duration_proof_l1624_162405

/-- Calculates the total time spent on a trip visiting three countries. -/
def total_trip_time (first_country_stay : ℕ) : ℕ :=
  first_country_stay + 2 * first_country_stay * 2

/-- Proves that the total trip time is 10 weeks given the specified conditions. -/
theorem trip_duration_proof :
  let first_country_stay := 2
  total_trip_time first_country_stay = 10 := by
  sorry

#eval total_trip_time 2

end NUMINAMATH_CALUDE_trip_duration_proof_l1624_162405


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l1624_162451

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8) :
  ∃ m M : ℝ, (∀ x y z : ℝ, x + y + z = 5 → x^2 + y^2 + z^2 = 8 → m ≤ x ∧ x ≤ M) ∧
            (∃ x y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8 ∧ x = m) ∧
            (∃ x y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8 ∧ x = M) ∧
            m + M = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l1624_162451


namespace NUMINAMATH_CALUDE_culture_medium_composition_l1624_162472

/-- Represents the composition of a culture medium --/
structure CultureMedium where
  salineWater : ℝ
  nutrientBroth : ℝ
  pureWater : ℝ

/-- The initial mixture ratio --/
def initialMixture : CultureMedium := {
  salineWater := 0.1
  nutrientBroth := 0.05
  pureWater := 0
}

/-- The required total volume of the culture medium in liters --/
def totalVolume : ℝ := 1

/-- The required percentage of pure water in the final mixture --/
def pureWaterPercentage : ℝ := 0.3

theorem culture_medium_composition :
  ∃ (final : CultureMedium),
    final.salineWater + final.nutrientBroth + final.pureWater = totalVolume ∧
    final.nutrientBroth / (final.salineWater + final.nutrientBroth) = initialMixture.nutrientBroth / (initialMixture.salineWater + initialMixture.nutrientBroth) ∧
    final.pureWater = totalVolume * pureWaterPercentage ∧
    final.nutrientBroth = 1/3 ∧
    final.pureWater = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_culture_medium_composition_l1624_162472


namespace NUMINAMATH_CALUDE_triangle_angle_not_greater_than_60_l1624_162462

theorem triangle_angle_not_greater_than_60 (a b c : ℝ) (h_triangle : a + b + c = 180) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_greater_than_60_l1624_162462


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l1624_162486

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (4 * x = x^2 - 8) ↔ (x^2 - 4*x - 8 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l1624_162486


namespace NUMINAMATH_CALUDE_courtyard_length_l1624_162423

/-- Proves that a courtyard with given width and number of bricks of specific dimensions has a certain length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) :
  width = 14 →
  brick_length = 0.25 →
  brick_width = 0.15 →
  num_bricks = 8960 →
  (width * (num_bricks * brick_length * brick_width / width)) = 24 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l1624_162423


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1624_162484

theorem quadratic_factorization (C D : ℤ) :
  (∀ x, 15 * x^2 - 56 * x + 48 = (C * x - 8) * (D * x - 6)) →
  C * D + C = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1624_162484


namespace NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l1624_162406

/-- A decreasing function on ℝ -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem solution_set_of_decreasing_function
  (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) (h_f_1 : f 1 = 0) :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l1624_162406


namespace NUMINAMATH_CALUDE_expression_evaluation_l1624_162431

theorem expression_evaluation :
  let x : ℝ := 16
  let expr := (2 + x * (2 + Real.sqrt x) - 4^2) / (Real.sqrt x - 4 + x^2)
  expr = 41 / 128 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1624_162431


namespace NUMINAMATH_CALUDE_circle_equation_l1624_162477

theorem circle_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ), c = (0, -2) ∧ r = 4 ∧ (x - c.1)^2 + (y - c.2)^2 = r^2) ↔
  x^2 + (y + 2)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l1624_162477


namespace NUMINAMATH_CALUDE_pythons_for_fifteen_alligators_l1624_162441

/-- The number of Burmese pythons required to eat a given number of alligators in a specified time period. -/
def pythons_required (alligators : ℕ) (weeks : ℕ) : ℕ :=
  (alligators + weeks - 1) / weeks

/-- The theorem stating that 5 Burmese pythons are required to eat 15 alligators in 3 weeks. -/
theorem pythons_for_fifteen_alligators : pythons_required 15 3 = 5 := by
  sorry

#eval pythons_required 15 3

end NUMINAMATH_CALUDE_pythons_for_fifteen_alligators_l1624_162441


namespace NUMINAMATH_CALUDE_greatest_integer_radius_of_semicircle_l1624_162452

theorem greatest_integer_radius_of_semicircle (A : ℝ) (h : A < 45 * Real.pi) :
  ∃ (r : ℕ), r = 9 ∧ (∀ (n : ℕ), (↑n : ℝ)^2 * Real.pi / 2 ≤ A → n ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_of_semicircle_l1624_162452


namespace NUMINAMATH_CALUDE_lawn_width_proof_l1624_162428

theorem lawn_width_proof (length width : ℝ) (road_width cost_per_sqm total_cost : ℝ) : 
  length = 80 →
  road_width = 15 →
  cost_per_sqm = 3 →
  total_cost = 5625 →
  (road_width * width + road_width * length - road_width * road_width) * cost_per_sqm = total_cost →
  width = 60 := by
sorry

end NUMINAMATH_CALUDE_lawn_width_proof_l1624_162428


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l1624_162427

theorem multiplication_subtraction_difference : ∃ (x : ℤ), x = 22 ∧ 3 * x - (62 - x) = 26 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l1624_162427


namespace NUMINAMATH_CALUDE_intersection_point_count_l1624_162426

theorem intersection_point_count :
  ∃! p : ℝ × ℝ, 
    (p.1 + p.2 - 5) * (2 * p.1 - 3 * p.2 + 5) = 0 ∧ 
    (p.1 - p.2 + 1) * (3 * p.1 + 2 * p.2 - 12) = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_count_l1624_162426


namespace NUMINAMATH_CALUDE_matt_work_time_l1624_162493

/-- The number of minutes Matt worked on Monday -/
def monday_minutes : ℕ := 450

/-- The number of minutes Matt worked on Tuesday -/
def tuesday_minutes : ℕ := monday_minutes / 2

/-- The additional minutes Matt worked on the certain day compared to Tuesday -/
def additional_minutes : ℕ := 75

/-- The number of minutes Matt worked on the certain day -/
def certain_day_minutes : ℕ := tuesday_minutes + additional_minutes

theorem matt_work_time : certain_day_minutes = 300 := by
  sorry

end NUMINAMATH_CALUDE_matt_work_time_l1624_162493


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l1624_162403

-- Define the sets A and B as functions of a
def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a+3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

-- State the theorem
theorem set_intersection_and_union :
  ∃ (a : ℝ), (A a ∩ B a = {2, 5}) ∧ (a = 2) ∧ (A a ∪ B a = {-4, 2, 4, 5, 25}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l1624_162403


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l1624_162478

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l1624_162478


namespace NUMINAMATH_CALUDE_function_equivalence_l1624_162419

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + 2*x + 2

-- State the theorem
theorem function_equivalence :
  (∀ x ≥ 0, f (Real.sqrt x - 1) = x + 1) →
  (∀ x ≥ -1, f x = x^2 + 2*x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l1624_162419


namespace NUMINAMATH_CALUDE_doghouse_area_doghouse_area_value_l1624_162425

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem doghouse_area (side_length : Real) (rope_length : Real) 
  (h1 : side_length = 2)
  (h2 : rope_length = 3) : 
  Real := by
  sorry

#check doghouse_area

theorem doghouse_area_value : 
  doghouse_area 2 3 rfl rfl = (22 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_doghouse_area_doghouse_area_value_l1624_162425


namespace NUMINAMATH_CALUDE_NH4I_molecular_weight_l1624_162467

/-- The molecular weight of NH4I in grams per mole -/
def molecular_weight_NH4I : ℝ := 145

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 9

/-- The total weight in grams for the given number of moles -/
def given_total_weight : ℝ := 1305

theorem NH4I_molecular_weight :
  molecular_weight_NH4I = given_total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_NH4I_molecular_weight_l1624_162467


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l1624_162413

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerMod10 (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) % 10

theorem units_digit_of_sum : unitsDigit ((33 : ℕ)^43 + (43 : ℕ)^32) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l1624_162413


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1624_162465

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1624_162465


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slope_product_l1624_162433

/-- Given a parabola y^2 = 2px (p > 0) and a line y = x - p intersecting the parabola at points A and B,
    the product of the slopes of lines OA and OB is -2, where O is the coordinate origin. -/
theorem parabola_line_intersection_slope_product (p : ℝ) (h : p > 0) : 
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 2*p*A.1) ∧ 
    (B.2^2 = 2*p*B.1) ∧
    (A.2 = A.1 - p) ∧ 
    (B.2 = B.1 - p) ∧
    ((A.2 / A.1) * (B.2 / B.1) = -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slope_product_l1624_162433


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1624_162442

theorem hyperbola_focal_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 10 - y^2 / 2 = 1
  ∃ (f : ℝ), f = 4 * Real.sqrt 3 ∧ 
    ∀ (x y : ℝ), h x y → 
      f = 2 * Real.sqrt ((Real.sqrt 10)^2 + (Real.sqrt 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1624_162442


namespace NUMINAMATH_CALUDE_sum_of_ratios_l1624_162443

theorem sum_of_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x/y + y/z + z/x = Real.sqrt ((x/y)^2 + (y/z)^2 + (z/x)^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l1624_162443


namespace NUMINAMATH_CALUDE_smallest_multiple_105_with_105_divisors_l1624_162469

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest positive integer that is a multiple of 105 and has exactly 105 positive integral divisors -/
def n : ℕ := sorry

theorem smallest_multiple_105_with_105_divisors :
  n > 0 ∧ 
  105 ∣ n ∧ 
  num_divisors n = 105 ∧ 
  ∀ m : ℕ, m > 0 → 105 ∣ m → num_divisors m = 105 → m ≥ n ∧
  n / 105 = 6289125 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_105_with_105_divisors_l1624_162469


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1624_162481

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_sum_24 : a 2 + a 4 = 20)
  (h_sum_35 : a 3 + a 5 = 40) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1624_162481


namespace NUMINAMATH_CALUDE_nh4i_equilibrium_constant_l1624_162401

/-- Equilibrium constant for a chemical reaction --/
def equilibrium_constant (c_nh3 c_hi : ℝ) : ℝ := c_nh3 * c_hi

/-- Concentration of HI produced from NH₄I decomposition --/
def c_hi_from_nh4i (c_hi c_h2 : ℝ) : ℝ := c_hi + 2 * c_h2

theorem nh4i_equilibrium_constant (c_h2 c_hi : ℝ) 
  (h1 : c_h2 = 1) 
  (h2 : c_hi = 4) :
  equilibrium_constant (c_hi_from_nh4i c_hi c_h2) c_hi = 24 := by
  sorry

end NUMINAMATH_CALUDE_nh4i_equilibrium_constant_l1624_162401


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1624_162488

theorem tank_capacity_proof (T : ℚ) 
  (h1 : (5/8 : ℚ) * T + 15 = (4/5 : ℚ) * T) : 
  T = 86 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1624_162488


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l1624_162461

theorem sum_of_specific_numbers : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l1624_162461


namespace NUMINAMATH_CALUDE_apple_count_equality_l1624_162490

/-- The number of apples Marin has -/
def marins_apples : ℕ := 3

/-- The number of apples David has -/
def davids_apples : ℕ := 3

/-- The difference between Marin's and David's apple counts -/
def apple_difference : ℤ := marins_apples - davids_apples

theorem apple_count_equality : apple_difference = 0 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_equality_l1624_162490


namespace NUMINAMATH_CALUDE_alina_twist_result_l1624_162440

/-- Alina's twisting method for periodic decimal fractions -/
def twist (n : ℚ) : ℚ :=
  sorry

/-- The period length of the decimal representation of 503/2022 -/
def period_length : ℕ := 336

theorem alina_twist_result :
  twist (503 / 2022) = 9248267898383371824480369515011881956675900099900099900099 / (10^period_length - 1) :=
sorry

end NUMINAMATH_CALUDE_alina_twist_result_l1624_162440


namespace NUMINAMATH_CALUDE_floor_of_2_99_l1624_162424

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the properties of the floor function
axiom floor_le (x : ℝ) : (floor x : ℝ) ≤ x
axiom floor_lt (x : ℝ) : x < (floor x : ℝ) + 1

-- Theorem statement
theorem floor_of_2_99 : floor 2.99 = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_2_99_l1624_162424


namespace NUMINAMATH_CALUDE_product_equality_l1624_162444

theorem product_equality : 2.5 * 8.5 * (5.2 - 0.2) = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1624_162444


namespace NUMINAMATH_CALUDE_factorial_ratio_l1624_162454

theorem factorial_ratio : (11 : ℕ).factorial / ((7 : ℕ).factorial * (4 : ℕ).factorial) = 330 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1624_162454


namespace NUMINAMATH_CALUDE_flour_calculation_l1624_162421

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := sorry

/-- The total number of cups of flour required by the recipe -/
def total_flour_required : ℕ := 10

/-- The number of cups of flour Mary still needs to add -/
def flour_to_be_added : ℕ := 4

/-- Theorem: The number of cups of flour Mary has already put in is equal to
    the difference between the total cups of flour required and the cups of flour
    she still needs to add -/
theorem flour_calculation :
  flour_already_added = total_flour_required - flour_to_be_added :=
sorry

end NUMINAMATH_CALUDE_flour_calculation_l1624_162421


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1624_162455

theorem smallest_n_congruence : ∃ n : ℕ+, (∀ m : ℕ+, 19 * m ≡ 1453 [MOD 8] → n ≤ m) ∧ 19 * n ≡ 1453 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1624_162455


namespace NUMINAMATH_CALUDE_music_purchase_total_spent_l1624_162430

/-- Represents the purchase of music albums -/
structure MusicPurchase where
  country_albums : ℕ
  pop_albums : ℕ
  country_price : ℕ
  pop_price : ℕ
  songs_per_album : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ

/-- Calculates the total cost before discounts -/
def total_cost_before_discounts (purchase : MusicPurchase) : ℕ :=
  purchase.country_albums * purchase.country_price + purchase.pop_albums * purchase.pop_price

/-- Calculates the number of discounts -/
def number_of_discounts (purchase : MusicPurchase) : ℕ :=
  (purchase.country_albums + purchase.pop_albums) / purchase.discount_threshold

/-- Calculates the total amount spent after discounts -/
def total_amount_spent (purchase : MusicPurchase) : ℕ :=
  total_cost_before_discounts purchase - number_of_discounts purchase * purchase.discount_amount

/-- Theorem: The total amount spent on music albums after applying discounts is $108 -/
theorem music_purchase_total_spent (purchase : MusicPurchase) 
  (h1 : purchase.country_albums = 4)
  (h2 : purchase.pop_albums = 5)
  (h3 : purchase.country_price = 12)
  (h4 : purchase.pop_price = 15)
  (h5 : purchase.songs_per_album = 8)
  (h6 : purchase.discount_threshold = 3)
  (h7 : purchase.discount_amount = 5) :
  total_amount_spent purchase = 108 := by
  sorry

#eval total_amount_spent {
  country_albums := 4,
  pop_albums := 5,
  country_price := 12,
  pop_price := 15,
  songs_per_album := 8,
  discount_threshold := 3,
  discount_amount := 5
}

end NUMINAMATH_CALUDE_music_purchase_total_spent_l1624_162430


namespace NUMINAMATH_CALUDE_chinese_character_equation_l1624_162458

theorem chinese_character_equation : ∃! (math love i : ℕ),
  (math ≠ love ∧ math ≠ i ∧ love ≠ i) ∧
  (math > 0 ∧ love > 0 ∧ i > 0) ∧
  (math * (love * 1000 + math) = i * 1000 + love * 100 + math) ∧
  (math = 25 ∧ love = 125 ∧ i = 3) := by
  sorry

end NUMINAMATH_CALUDE_chinese_character_equation_l1624_162458


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1624_162446

/-- The atomic weight of Copper (Cu) in g/mol -/
def atomic_weight_Cu : ℝ := 63.546

/-- The atomic weight of Carbon (C) in g/mol -/
def atomic_weight_C : ℝ := 12.011

/-- The atomic weight of Oxygen (O) in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Cu atoms in the compound -/
def num_Cu : ℕ := 1

/-- The number of C atoms in the compound -/
def num_C : ℕ := 1

/-- The number of O atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  (num_Cu : ℝ) * atomic_weight_Cu +
  (num_C : ℝ) * atomic_weight_C +
  (num_O : ℝ) * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 123.554 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1624_162446


namespace NUMINAMATH_CALUDE_accounting_equation_l1624_162450

def p : ℂ := 7
def z : ℂ := 7 + 175 * Complex.I

theorem accounting_equation (h : 3 * p - z = 15000) : 
  p = 5002 + (175 / 3) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_accounting_equation_l1624_162450


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_one_l1624_162483

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- State the theorem
theorem subset_implies_a_geq_one (a : ℝ) : A ⊆ B a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_one_l1624_162483


namespace NUMINAMATH_CALUDE_movie_theater_seats_l1624_162498

theorem movie_theater_seats (total_seats : ℕ) (num_sections : ℕ) (seats_per_section : ℕ) :
  total_seats = 270 → num_sections = 9 → total_seats = num_sections * seats_per_section →
  seats_per_section = 30 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_seats_l1624_162498


namespace NUMINAMATH_CALUDE_max_ratio_on_unit_circle_l1624_162439

theorem max_ratio_on_unit_circle :
  let a : ℂ := Real.sqrt 17
  let b : ℂ := Complex.I * Real.sqrt 19
  (∃ (k : ℝ), k = 4/3 ∧
    ∀ (z : ℂ), Complex.abs z = 1 →
      Complex.abs (a - z) / Complex.abs (b - z) ≤ k) ∧
    ∀ (k' : ℝ), (∀ (z : ℂ), Complex.abs z = 1 →
      Complex.abs (a - z) / Complex.abs (b - z) ≤ k') →
      k' ≥ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_on_unit_circle_l1624_162439


namespace NUMINAMATH_CALUDE_odd_integers_equality_l1624_162457

theorem odd_integers_equality (a b c d k m : ℤ) :
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2 * k →
  b + c = 2 * m →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_integers_equality_l1624_162457


namespace NUMINAMATH_CALUDE_total_songs_two_days_l1624_162496

-- Define the number of songs listened to yesterday
def songs_yesterday : ℕ := 9

-- Define the relationship between yesterday's and today's songs
def song_relationship (x : ℕ) : Prop :=
  songs_yesterday = 2 * (x.sqrt : ℕ) - 5

-- Theorem to prove
theorem total_songs_two_days (x : ℕ) 
  (h : song_relationship x) : songs_yesterday + x = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_songs_two_days_l1624_162496


namespace NUMINAMATH_CALUDE_farm_tax_total_l1624_162420

/-- Represents the farm tax collected from a village -/
structure FarmTax where
  /-- Total amount collected from the village -/
  total : ℝ
  /-- Amount paid by Mr. William -/
  william_paid : ℝ
  /-- Percentage of total taxable land owned by Mr. William -/
  william_percentage : ℝ
  /-- Assertion that Mr. William's percentage is 50% -/
  h_percentage : william_percentage = 50
  /-- Assertion that Mr. William paid $480 -/
  h_william_paid : william_paid = 480
  /-- The total tax is twice what Mr. William paid -/
  h_total : total = 2 * william_paid

/-- Theorem stating that the total farm tax collected is $960 -/
theorem farm_tax_total (ft : FarmTax) : ft.total = 960 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_total_l1624_162420


namespace NUMINAMATH_CALUDE_dodecagon_triangle_count_l1624_162453

/-- A regular dodecagon -/
structure RegularDodecagon where
  vertices : Finset ℕ
  regular : vertices.card = 12

/-- Count of triangles with specific properties in a regular dodecagon -/
def triangle_count (d : RegularDodecagon) : ℕ × ℕ :=
  let equilateral := 4  -- Number of equilateral triangles
  let scalene := 168    -- Number of scalene triangles
  (equilateral, scalene)

/-- Theorem stating the correct count of equilateral and scalene triangles in a regular dodecagon -/
theorem dodecagon_triangle_count (d : RegularDodecagon) :
  triangle_count d = (4, 168) := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_triangle_count_l1624_162453


namespace NUMINAMATH_CALUDE_square_pyramid_dihedral_angle_cosine_l1624_162448

/-- A pyramid with a square base and specific properties -/
structure SquarePyramid where
  -- The length of the congruent edges
  edge_length : ℝ
  -- The measure of the dihedral angle between faces PQR and PRS
  dihedral_angle : ℝ
  -- Angle QPR is 45°
  angle_QPR_is_45 : angle_QPR = Real.pi / 4
  -- The base is square (implied by the problem setup)
  base_is_square : True

/-- The theorem statement -/
theorem square_pyramid_dihedral_angle_cosine 
  (P : SquarePyramid) 
  (a b : ℝ) 
  (h : Real.cos P.dihedral_angle = a + Real.sqrt b) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_dihedral_angle_cosine_l1624_162448


namespace NUMINAMATH_CALUDE_ratio_a5_a7_l1624_162447

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  decreasing : ∀ n, a (n + 1) < a n
  geometric : ∀ n k, a (n + k) = a n * (a 2 / a 1) ^ k
  prop1 : a 2 * a 8 = 6
  prop2 : a 4 + a 6 = 5

/-- The main theorem about the ratio of a_5 to a_7 -/
theorem ratio_a5_a7 (seq : SpecialGeometricSequence) : seq.a 5 / seq.a 7 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a5_a7_l1624_162447


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1624_162464

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  ((a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a)) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1624_162464


namespace NUMINAMATH_CALUDE_unique_integer_triangle_with_unit_incircle_l1624_162471

/-- A triangle with integer side lengths and an inscribed circle of radius 1 -/
structure IntegerTriangleWithUnitIncircle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  h_incircle : (a + b + c) * 2 = (a + b - c) * (b + c - a) * (c + a - b)

/-- The only triangle with integer side lengths and an inscribed circle of radius 1 has sides 5, 4, and 3 -/
theorem unique_integer_triangle_with_unit_incircle :
  ∀ t : IntegerTriangleWithUnitIncircle, t.a = 5 ∧ t.b = 4 ∧ t.c = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_triangle_with_unit_incircle_l1624_162471


namespace NUMINAMATH_CALUDE_simplify_expression_l1624_162435

theorem simplify_expression (x : ℝ) : 2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1624_162435


namespace NUMINAMATH_CALUDE_fifth_polygon_exterior_angles_sum_l1624_162418

/-- Represents a polygon in the sequence -/
structure Polygon where
  sides : ℕ

/-- Generates the next polygon in the sequence -/
def nextPolygon (p : Polygon) : Polygon :=
  { sides := p.sides + 2 }

/-- The sequence of polygons -/
def polygonSequence : ℕ → Polygon
  | 0 => { sides := 4 }  -- Square
  | n + 1 => nextPolygon (polygonSequence n)

/-- Sum of exterior angles of a polygon -/
def sumExteriorAngles (p : Polygon) : ℝ := 360

theorem fifth_polygon_exterior_angles_sum :
  sumExteriorAngles (polygonSequence 4) = 360 := by
  sorry

end NUMINAMATH_CALUDE_fifth_polygon_exterior_angles_sum_l1624_162418


namespace NUMINAMATH_CALUDE_certain_number_is_negative_eleven_l1624_162437

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem certain_number_is_negative_eleven :
  ∃ (certain_number : ℤ),
    (binary_op 3 < certain_number) ∧
    (certain_number ≤ binary_op 4) ∧
    (∀ m : ℤ, (binary_op 3 < m) ∧ (m ≤ binary_op 4) → certain_number ≤ m) ∧
    certain_number = -11 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_negative_eleven_l1624_162437


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_15_sided_l1624_162438

/-- The number of sides in the regular polygon -/
def n : ℕ := 15

/-- The total number of diagonals in a regular n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular n-sided polygon -/
def shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in a regular n-sided polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  shortest_diagonals n / total_diagonals n

theorem prob_shortest_diagonal_15_sided :
  prob_shortest_diagonal n = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_15_sided_l1624_162438


namespace NUMINAMATH_CALUDE_farm_legs_count_l1624_162404

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of legs a buffalo has -/
def buffalo_legs : ℕ := 4

/-- The total number of animals in the farm -/
def total_animals : ℕ := 13

/-- The number of chickens in the farm -/
def num_chickens : ℕ := 4

/-- The number of buffalos in the farm -/
def num_buffalos : ℕ := total_animals - num_chickens

theorem farm_legs_count : 
  num_chickens * chicken_legs + num_buffalos * buffalo_legs = 44 := by
sorry

end NUMINAMATH_CALUDE_farm_legs_count_l1624_162404


namespace NUMINAMATH_CALUDE_quadratic_roots_for_positive_discriminant_l1624_162460

theorem quadratic_roots_for_positive_discriminant
  (a b c : ℝ) (h_a : a ≠ 0) (h_disc : b^2 - 4*a*c > 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = 0 ∧
    a * x₂^2 + b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_for_positive_discriminant_l1624_162460


namespace NUMINAMATH_CALUDE_residue_of_9_pow_2010_mod_17_l1624_162422

theorem residue_of_9_pow_2010_mod_17 : 9^2010 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_9_pow_2010_mod_17_l1624_162422


namespace NUMINAMATH_CALUDE_min_value_expression_l1624_162473

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((2*a + 2*a*b - b*(b + 1))^2 + (b - 4*a^2 + 2*a*(b + 1))^2) / (4*a^2 + b^2) ≥ 1 ∧
  ((2*1 + 2*1*1 - 1*(1 + 1))^2 + (1 - 4*1^2 + 2*1*(1 + 1))^2) / (4*1^2 + 1^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1624_162473


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1624_162494

-- Define the polynomial
def P (x : ℝ) : ℝ := 4*x^4 + 4*x^3 - 11*x^2 - 6*x + 9

-- Define the divisor
def D (x : ℝ) : ℝ := (x - 1)^2

-- Define the quotient
def Q (x : ℝ) : ℝ := 4*x^2 + 12*x + 9

-- Theorem statement
theorem polynomial_divisibility :
  ∀ x : ℝ, P x = D x * Q x :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1624_162494


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1624_162479

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = 15 ∧ h_large / h_small = Real.sqrt area_ratio :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1624_162479


namespace NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l1624_162415

theorem multiply_whole_and_mixed_number :
  7 * (9 + 2 / 5) = 65 + 4 / 5 := by sorry

end NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l1624_162415


namespace NUMINAMATH_CALUDE_number_comparison_l1624_162445

def A : ℕ := 888888888888888888888  -- 19 eights
def B : ℕ := 333333333333333333333333333333333333333333333333333333333333333333333  -- 68 threes
def C : ℕ := 444444444444444444444  -- 19 fours
def D : ℕ := 666666666666666666666666666666666666666666666666666666666666666666667  -- 67 sixes and one seven

theorem number_comparison : C * D - A * B = 444444444444444444444 := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l1624_162445


namespace NUMINAMATH_CALUDE_defective_units_percentage_l1624_162407

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 4

/-- The percentage of all units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.2

/-- The percentage of all units that are defective -/
def defective_percent : ℝ := 5

theorem defective_units_percentage : 
  defective_shipped_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l1624_162407


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1624_162487

theorem sin_alpha_value (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.sin (30 * π / 180) ∧ 
                   y = 2 * Real.cos (30 * π / 180) ∧ 
                   x = 2 * Real.sin α ∧ 
                   y = 2 * Real.cos α) → 
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1624_162487


namespace NUMINAMATH_CALUDE_base_conversion_185_to_113_l1624_162480

/-- Converts a base 13 number to base 10 --/
def base13ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 13^2 + tens * 13^1 + ones * 13^0

/-- Checks if a number is a valid base 13 digit --/
def isValidBase13Digit (d : Nat) : Prop :=
  d < 13

theorem base_conversion_185_to_113 :
  (∀ d, isValidBase13Digit d → d < 13) →
  base13ToBase10 1 1 3 = 185 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_185_to_113_l1624_162480


namespace NUMINAMATH_CALUDE_complex_number_location_l1624_162410

theorem complex_number_location (z : ℂ) (h : z + z * Complex.I = 3 + 2 * Complex.I) : 
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1624_162410


namespace NUMINAMATH_CALUDE_waste_bread_price_is_correct_l1624_162468

/-- Calculates the price per pound of wasted bread products given the following conditions:
  * Minimum wage is $8/hour
  * 20 pounds of meat wasted at $5/pound
  * 15 pounds of fruits and vegetables wasted at $4/pound
  * 60 pounds of bread products wasted (price unknown)
  * 10 hours of time-and-a-half pay for janitorial staff (normal pay $10/hour)
  * Total work hours to pay for everything is 50 hours
-/
def wasteBreadPrice (
  minWage : ℚ)
  (meatWeight : ℚ)
  (meatPrice : ℚ)
  (fruitVegWeight : ℚ)
  (fruitVegPrice : ℚ)
  (breadWeight : ℚ)
  (janitorHours : ℚ)
  (janitorWage : ℚ)
  (totalWorkHours : ℚ) : ℚ :=
  let meatCost := meatWeight * meatPrice
  let fruitVegCost := fruitVegWeight * fruitVegPrice
  let janitorCost := janitorHours * (janitorWage * 1.5)
  let totalEarnings := totalWorkHours * minWage
  let breadCost := totalEarnings - (meatCost + fruitVegCost + janitorCost)
  breadCost / breadWeight

theorem waste_bread_price_is_correct :
  wasteBreadPrice 8 20 5 15 4 60 10 10 50 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_waste_bread_price_is_correct_l1624_162468
