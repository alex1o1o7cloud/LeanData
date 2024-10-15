import Mathlib

namespace NUMINAMATH_CALUDE_fourth_term_is_negative_24_l3063_306352

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n-1)

-- Define the conditions of our specific sequence
def sequence_conditions (x : ℝ) : Prop :=
  ∃ (r : ℝ), 
    geometric_sequence x r 2 = 3*x + 3 ∧
    geometric_sequence x r 3 = 6*x + 6

-- Theorem statement
theorem fourth_term_is_negative_24 :
  ∀ x : ℝ, sequence_conditions x → geometric_sequence x 2 4 = -24 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_24_l3063_306352


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3063_306329

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes, accounting for time zone difference -/
def timeDifference (departure : Time) (arrival : Time) (timeZoneDiff : ℤ) : ℕ :=
  let totalMinutes := (arrival.hours - departure.hours) * 60 + arrival.minutes - departure.minutes
  (totalMinutes + timeZoneDiff * 60).toNat

/-- Theorem stating the flight duration property -/
theorem flight_duration_sum (departureTime : Time) (arrivalTime : Time) 
    (h : ℕ) (m : ℕ) (mPos : 0 < m) (mLt60 : m < 60) :
    departureTime.hours = 15 ∧ departureTime.minutes = 15 →
    arrivalTime.hours = 16 ∧ arrivalTime.minutes = 50 →
    timeDifference departureTime arrivalTime (-1) = h * 60 + m →
    h + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l3063_306329


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l3063_306378

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 27) ∧ k ∣ (7 * m + 4))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 27) ∧ k ∣ (7 * n + 4)) ∧
  n = 220 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l3063_306378


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l3063_306393

/-- Given a two-digit number, constructs a six-digit number by repeating it three times -/
def repeat_twice (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + n

/-- Theorem: For any two-digit number, the six-digit number formed by repeating it three times is divisible by 10101 -/
theorem six_digit_divisibility (n : ℕ) (h : n ≥ 10 ∧ n ≤ 99) : 
  (repeat_twice n) % 10101 = 0 := by
  sorry


end NUMINAMATH_CALUDE_six_digit_divisibility_l3063_306393


namespace NUMINAMATH_CALUDE_c_share_l3063_306358

def total_amount : ℕ := 880

def share_ratio (a b c : ℕ) : Prop :=
  4 * a = 5 * b ∧ 5 * b = 10 * c

theorem c_share (a b c : ℕ) (h1 : share_ratio a b c) (h2 : a + b + c = total_amount) :
  c = 160 := by
  sorry

end NUMINAMATH_CALUDE_c_share_l3063_306358


namespace NUMINAMATH_CALUDE_regular_tetrahedron_vertices_and_edges_l3063_306372

/-- A regular tetrahedron is a regular triangular pyramid -/
structure RegularTetrahedron where
  is_regular_triangular_pyramid : Bool

/-- The number of vertices in a regular tetrahedron -/
def num_vertices (t : RegularTetrahedron) : ℕ := 4

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- Theorem stating that a regular tetrahedron has 4 vertices and 6 edges -/
theorem regular_tetrahedron_vertices_and_edges (t : RegularTetrahedron) :
  num_vertices t = 4 ∧ num_edges t = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_vertices_and_edges_l3063_306372


namespace NUMINAMATH_CALUDE_football_players_l3063_306314

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 250)
  (h2 : cricket = 90)
  (h3 : neither = 50)
  (h4 : both = 50) :
  total - neither - (cricket - both) = 160 :=
by
  sorry

#check football_players

end NUMINAMATH_CALUDE_football_players_l3063_306314


namespace NUMINAMATH_CALUDE_second_catch_up_l3063_306395

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the state of the race -/
structure RaceState where
  runner1 : Runner
  runner2 : Runner
  laps_completed : ℝ

/-- Defines the initial state of the race -/
def initial_state : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 1 },
    laps_completed := 0 }

/-- Defines the state after the second runner doubles their speed -/
def intermediate_state : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 2 },
    laps_completed := 0.5 }

/-- Theorem stating that the first runner will catch up again when the second runner has completed 2.5 laps -/
theorem second_catch_up (state : RaceState) :
  state.runner1.speed > state.runner2.speed →
  ∃ t : ℝ, t > 0 ∧ 
    state.runner1.speed * t = (state.laps_completed + 2.5) ∧
    state.runner2.speed * t = 2 :=
  sorry

end NUMINAMATH_CALUDE_second_catch_up_l3063_306395


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3063_306382

theorem fraction_inequality_solution_set (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3063_306382


namespace NUMINAMATH_CALUDE_range_of_k_l3063_306333

/-- An odd function that is strictly decreasing on [0, +∞) -/
def OddDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f y < f x)

theorem range_of_k (f : ℝ → ℝ) (h_odd_dec : OddDecreasingFunction f) :
  (∀ k x : ℝ, f (k * x^2 + 2) + f (k * x + k) ≤ 0) ↔ 
  (∀ k : ℝ, 0 ≤ k) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3063_306333


namespace NUMINAMATH_CALUDE_function_transformation_l3063_306300

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_transformation (h : f 1 = -2) : f (-(-1)) + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3063_306300


namespace NUMINAMATH_CALUDE_unique_lcm_solution_l3063_306365

theorem unique_lcm_solution : ∃! (n : ℕ), n > 0 ∧ Nat.lcm n (n - 30) = n + 1320 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_lcm_solution_l3063_306365


namespace NUMINAMATH_CALUDE_unique_solution_l3063_306310

theorem unique_solution : ∃! x : ℝ, x^29 * 4^15 = 2 * 10^29 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3063_306310


namespace NUMINAMATH_CALUDE_modified_number_wall_m_value_l3063_306379

/-- Represents a modified Number Wall with given values -/
structure ModifiedNumberWall where
  m : ℕ
  row1 : Vector ℕ 4
  row2 : Vector ℕ 3
  row3 : Vector ℕ 2
  row4 : ℕ

/-- The modified Number Wall satisfies the sum property -/
def is_valid_wall (wall : ModifiedNumberWall) : Prop :=
  wall.row1.get 0 = wall.m ∧
  wall.row1.get 1 = 5 ∧
  wall.row1.get 2 = 10 ∧
  wall.row1.get 3 = 6 ∧
  wall.row2.get 1 = 18 ∧
  wall.row4 = 56 ∧
  wall.row2.get 0 = wall.row1.get 0 + wall.row1.get 1 ∧
  wall.row2.get 1 = wall.row1.get 1 + wall.row1.get 2 ∧
  wall.row2.get 2 = wall.row1.get 2 + wall.row1.get 3 ∧
  wall.row3.get 0 = wall.row2.get 0 + wall.row2.get 1 ∧
  wall.row3.get 1 = wall.row2.get 1 + wall.row2.get 2 ∧
  wall.row4 = wall.row3.get 0 + wall.row3.get 1

/-- The value of 'm' in a valid modified Number Wall is 17 -/
theorem modified_number_wall_m_value (wall : ModifiedNumberWall) :
  is_valid_wall wall → wall.m = 17 := by sorry

end NUMINAMATH_CALUDE_modified_number_wall_m_value_l3063_306379


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l3063_306367

/-- Given a gas station that sold 2 boxes of chocolate candy, 5 boxes of sugar candy,
    and some boxes of gum, with a total of 9 boxes sold, prove that 2 boxes of gum were sold. -/
theorem gas_station_candy_boxes : 
  let chocolate_boxes : ℕ := 2
  let sugar_boxes : ℕ := 5
  let total_boxes : ℕ := 9
  let gum_boxes : ℕ := total_boxes - chocolate_boxes - sugar_boxes
  gum_boxes = 2 := by sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l3063_306367


namespace NUMINAMATH_CALUDE_max_cutlery_sets_l3063_306321

theorem max_cutlery_sets (dinner_forks knives soup_spoons teaspoons dessert_forks butter_knives : ℕ) 
  (max_capacity : ℕ) (dinner_fork_weight knife_weight soup_spoon_weight teaspoon_weight dessert_fork_weight butter_knife_weight : ℕ) : 
  dinner_forks = 6 →
  knives = dinner_forks + 9 →
  soup_spoons = 2 * knives →
  teaspoons = dinner_forks / 2 →
  dessert_forks = teaspoons / 3 →
  butter_knives = 2 * dessert_forks →
  max_capacity = 20000 →
  dinner_fork_weight = 80 →
  knife_weight = 100 →
  soup_spoon_weight = 85 →
  teaspoon_weight = 50 →
  dessert_fork_weight = 70 →
  butter_knife_weight = 65 →
  (max_capacity - (dinner_forks * dinner_fork_weight + knives * knife_weight + 
    soup_spoons * soup_spoon_weight + teaspoons * teaspoon_weight + 
    dessert_forks * dessert_fork_weight + butter_knives * butter_knife_weight)) / 
    (dinner_fork_weight + knife_weight) = 84 := by
  sorry

end NUMINAMATH_CALUDE_max_cutlery_sets_l3063_306321


namespace NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l3063_306315

theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → |x| > 2) ∧ 
  (∃ x : ℝ, |x| > 2 ∧ x ≥ a) →
  a ≤ -2 :=
by sorry

theorem max_value_is_negative_two :
  ∃ a : ℝ, 
    (∀ x : ℝ, x < a → |x| > 2) ∧
    (∃ x : ℝ, |x| > 2 ∧ x ≥ a) ∧
    a = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l3063_306315


namespace NUMINAMATH_CALUDE_power_product_l3063_306302

theorem power_product (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by sorry

end NUMINAMATH_CALUDE_power_product_l3063_306302


namespace NUMINAMATH_CALUDE_min_value_circle_line_l3063_306324

/-- The minimum value of 1/a + 4/b for a circle and a line passing through its center --/
theorem min_value_circle_line (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y - 1 = 0 → a*x - 2*b*y + 2 = 0) →
  (1/a + 4/b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_circle_line_l3063_306324


namespace NUMINAMATH_CALUDE_fib_F15_units_digit_l3063_306343

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The period of the units digit in the Fibonacci sequence -/
def fib_units_period : ℕ := 60

/-- Theorem: The units digit of F_{F_15} is 5 -/
theorem fib_F15_units_digit : fib (fib 15) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fib_F15_units_digit_l3063_306343


namespace NUMINAMATH_CALUDE_f_range_l3063_306370

def f (x : ℕ) : ℤ := Int.floor ((x + 1) / 2 : ℚ) - Int.floor (x / 2 : ℚ)

theorem f_range : ∀ x : ℕ, f x = 0 ∨ f x = 1 ∧ ∃ a b : ℕ, f a = 0 ∧ f b = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l3063_306370


namespace NUMINAMATH_CALUDE_factor_expression_l3063_306326

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3063_306326


namespace NUMINAMATH_CALUDE_equation_solutions_l3063_306339

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2)/(x + 1) * (x + (15 - x)/(x + 1))
  ∀ x : ℝ, f x = 60 ↔ x = 5 ∨ x = 6 ∨ x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3063_306339


namespace NUMINAMATH_CALUDE_bromine_only_liquid_l3063_306344

-- Define the set of elements
inductive Element : Type
| Bromine : Element
| Krypton : Element
| Phosphorus : Element
| Xenon : Element

-- Define the state of matter
inductive State : Type
| Solid : State
| Liquid : State
| Gas : State

-- Define the function to determine the state of an element at given temperature and pressure
def stateAtConditions (e : Element) (temp : ℝ) (pressure : ℝ) : State := sorry

-- Define the temperature and pressure conditions
def roomTemp : ℝ := 25
def atmPressure : ℝ := 1.0

-- Theorem statement
theorem bromine_only_liquid :
  ∀ e : Element, 
    stateAtConditions e roomTemp atmPressure = State.Liquid ↔ e = Element.Bromine :=
sorry

end NUMINAMATH_CALUDE_bromine_only_liquid_l3063_306344


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3063_306309

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {2, 4, 5, 8, 10}
  A ∩ B = {2, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3063_306309


namespace NUMINAMATH_CALUDE_number_sum_problem_l3063_306394

theorem number_sum_problem (x : ℝ) (h : 20 + x = 30) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_problem_l3063_306394


namespace NUMINAMATH_CALUDE_probability_of_at_least_one_of_each_color_l3063_306350

-- Define the number of marbles of each color
def red_marbles : Nat := 3
def blue_marbles : Nat := 3
def green_marbles : Nat := 3

-- Define the total number of marbles
def total_marbles : Nat := red_marbles + blue_marbles + green_marbles

-- Define the number of marbles to be selected
def selected_marbles : Nat := 4

-- Define the probability of selecting at least one marble of each color
def prob_at_least_one_of_each : Rat := 9/14

-- Theorem statement
theorem probability_of_at_least_one_of_each_color :
  prob_at_least_one_of_each = 
    (Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2 +
     Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 +
     Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
    Nat.choose total_marbles selected_marbles := by
  sorry

end NUMINAMATH_CALUDE_probability_of_at_least_one_of_each_color_l3063_306350


namespace NUMINAMATH_CALUDE_boating_group_size_l3063_306318

theorem boating_group_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 10 ∧ 
  n % 5 = 1 ∧ 
  n = 46 := by
  sorry

end NUMINAMATH_CALUDE_boating_group_size_l3063_306318


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3063_306376

theorem movie_ticket_cost (x : ℝ) : 
  (2 * x + 3 * (x - 2) = 39) →
  x = 9 := by sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l3063_306376


namespace NUMINAMATH_CALUDE_salary_comparison_l3063_306359

/-- Given salaries of A, B, and C with specified relationships, prove the percentage differences -/
theorem salary_comparison (a b c : ℝ) 
  (h1 : a = b * 0.8)  -- A's salary is 20% less than B's
  (h2 : c = a * 1.3)  -- C's salary is 30% more than A's
  : (b - a) / a = 0.25 ∧ (c - b) / b = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l3063_306359


namespace NUMINAMATH_CALUDE_sqrt_fraction_difference_l3063_306399

theorem sqrt_fraction_difference : Real.sqrt (9/4) - Real.sqrt (4/9) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_difference_l3063_306399


namespace NUMINAMATH_CALUDE_calculation_proof_l3063_306375

theorem calculation_proof : 5^2 * 7 + 9 * 4 - 35 / 5 = 204 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3063_306375


namespace NUMINAMATH_CALUDE_lemonade_stand_cost_l3063_306336

-- Define the given conditions
def total_profit : ℝ := 44
def lemonade_revenue : ℝ := 47
def lemonades_sold : ℕ := 50
def babysitting_income : ℝ := 31
def lemon_cost : ℝ := 0.20
def sugar_cost : ℝ := 0.15
def ice_cost : ℝ := 0.05
def sunhat_cost : ℝ := 10

-- Define the theorem
theorem lemonade_stand_cost :
  let variable_cost_per_lemonade := lemon_cost + sugar_cost + ice_cost
  let total_variable_cost := variable_cost_per_lemonade * lemonades_sold
  let total_cost := total_variable_cost + sunhat_cost
  total_cost = 30 := by sorry

end NUMINAMATH_CALUDE_lemonade_stand_cost_l3063_306336


namespace NUMINAMATH_CALUDE_circle_passes_800_squares_l3063_306353

/-- A circle on a unit square grid -/
structure GridCircle where
  radius : ℕ
  -- The circle does not touch any grid lines or pass through any lattice points
  no_grid_touch : True

/-- The number of squares a circle passes through on a unit square grid -/
def squares_passed (c : GridCircle) : ℕ :=
  4 * (2 * c.radius)

/-- Theorem: A circle with radius 100 passes through 800 squares -/
theorem circle_passes_800_squares (c : GridCircle) (h : c.radius = 100) :
  squares_passed c = 800 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_800_squares_l3063_306353


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3063_306331

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3063_306331


namespace NUMINAMATH_CALUDE_ben_joe_shirt_difference_l3063_306391

/-- The number of new shirts Alex has -/
def alex_shirts : ℕ := 4

/-- The number of additional shirts Joe has compared to Alex -/
def joe_extra_shirts : ℕ := 3

/-- The number of new shirts Ben has -/
def ben_shirts : ℕ := 15

/-- The number of new shirts Joe has -/
def joe_shirts : ℕ := alex_shirts + joe_extra_shirts

theorem ben_joe_shirt_difference : ben_shirts - joe_shirts = 8 := by
  sorry

end NUMINAMATH_CALUDE_ben_joe_shirt_difference_l3063_306391


namespace NUMINAMATH_CALUDE_one_box_can_be_emptied_l3063_306387

/-- Represents a state of three boxes with balls -/
structure BoxState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents an operation of doubling balls in one box by transferring from another -/
inductive DoubleOperation
  | DoubleAFromB
  | DoubleAFromC
  | DoubleBFromA
  | DoubleBFromC
  | DoubleCFromA
  | DoubleCFromB

/-- Applies a single doubling operation to a BoxState -/
def applyOperation (state : BoxState) (op : DoubleOperation) : BoxState :=
  match op with
  | DoubleOperation.DoubleAFromB => ⟨state.a * 2, state.b - state.a, state.c⟩
  | DoubleOperation.DoubleAFromC => ⟨state.a * 2, state.b, state.c - state.a⟩
  | DoubleOperation.DoubleBFromA => ⟨state.a - state.b, state.b * 2, state.c⟩
  | DoubleOperation.DoubleBFromC => ⟨state.a, state.b * 2, state.c - state.b⟩
  | DoubleOperation.DoubleCFromA => ⟨state.a - state.c, state.b, state.c * 2⟩
  | DoubleOperation.DoubleCFromB => ⟨state.a, state.b - state.c, state.c * 2⟩

/-- Applies a sequence of doubling operations to a BoxState -/
def applyOperations (state : BoxState) (ops : List DoubleOperation) : BoxState :=
  ops.foldl applyOperation state

/-- Predicate to check if any box is empty -/
def isAnyBoxEmpty (state : BoxState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem stating that one box can be emptied -/
theorem one_box_can_be_emptied (initial : BoxState) :
  ∃ (ops : List DoubleOperation), isAnyBoxEmpty (applyOperations initial ops) :=
sorry

end NUMINAMATH_CALUDE_one_box_can_be_emptied_l3063_306387


namespace NUMINAMATH_CALUDE_correct_dice_configuration_l3063_306301

/-- Represents a die face with a number of dots -/
structure DieFace where
  dots : Nat
  h : dots ≥ 1 ∧ dots ≤ 6

/-- Represents the configuration of four dice -/
structure DiceConfiguration where
  faceA : DieFace
  faceB : DieFace
  faceC : DieFace
  faceD : DieFace

/-- Theorem stating the correct number of dots on each face -/
theorem correct_dice_configuration :
  ∃ (config : DiceConfiguration),
    config.faceA.dots = 3 ∧
    config.faceB.dots = 5 ∧
    config.faceC.dots = 6 ∧
    config.faceD.dots = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_dice_configuration_l3063_306301


namespace NUMINAMATH_CALUDE_binomial_10_3_l3063_306332

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3063_306332


namespace NUMINAMATH_CALUDE_exactly_one_statement_implies_negation_l3063_306398

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∧ ¬q
def statement3 (p q : Prop) : Prop := ¬p ∧ q
def statement4 (p q : Prop) : Prop := ¬p ∧ ¬q

def negation_of_or (p q : Prop) : Prop := ¬(p ∨ q)

theorem exactly_one_statement_implies_negation (p q : Prop) :
  (∃! i : Fin 4, match i with
    | 0 => statement1 p q → negation_of_or p q
    | 1 => statement2 p q → negation_of_or p q
    | 2 => statement3 p q → negation_of_or p q
    | 3 => statement4 p q → negation_of_or p q) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_statement_implies_negation_l3063_306398


namespace NUMINAMATH_CALUDE_profit_percentage_l3063_306377

/-- If selling an article at 2/3 of a certain price results in a 15% loss,
    then selling at the full certain price results in a 27.5% profit. -/
theorem profit_percentage (certain_price : ℝ) (cost_price : ℝ) :
  certain_price > 0 →
  cost_price > 0 →
  (2 / 3 : ℝ) * certain_price = 0.85 * cost_price →
  (certain_price - cost_price) / cost_price = 0.275 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_l3063_306377


namespace NUMINAMATH_CALUDE_compare_expressions_inequality_proof_l3063_306304

-- Part 1
theorem compare_expressions (x : ℝ) : (x + 7) * (x + 8) > (x + 6) * (x + 9) := by
  sorry

-- Part 2
theorem inequality_proof (a b c d : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) (h4 : c < d) :
  a * d + c < b * c + d := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_inequality_proof_l3063_306304


namespace NUMINAMATH_CALUDE_circle_equation_and_slope_range_l3063_306303

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the line y = 2x
def line_center (x y : ℝ) : Prop := y = 2 * x

-- Define the line x + y - 3 = 0
def line_intersect (x y : ℝ) : Prop := x + y - 3 = 0

-- Define points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define point M
def point_M : ℝ × ℝ := (0, 5)

-- Define the dot product of OA and OB
def OA_dot_OB_zero : Prop :=
  (point_A.1 - origin.1) * (point_B.1 - origin.1) + 
  (point_A.2 - origin.2) * (point_B.2 - origin.2) = 0

-- Define the slope range for line MP
def slope_range (k : ℝ) : Prop := k ≤ -1/2 ∨ k ≥ 2

theorem circle_equation_and_slope_range :
  (∀ x y, circle_C x y → ((x, y) = origin ∨ line_center x y)) ∧
  (∀ x y, line_intersect x y → circle_C x y → ((x, y) = point_A ∨ (x, y) = point_B)) ∧
  OA_dot_OB_zero →
  (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧
  (∀ k, (∃ x y, circle_C x y ∧ y - 5 = k * x) ↔ slope_range k) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_slope_range_l3063_306303


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3063_306308

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + Complex.I) / (1 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3063_306308


namespace NUMINAMATH_CALUDE_least_common_multiple_5_to_10_l3063_306357

theorem least_common_multiple_5_to_10 : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, 5 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 5 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_5_to_10_l3063_306357


namespace NUMINAMATH_CALUDE_probability_of_sum_magnitude_at_least_sqrt2_l3063_306330

/-- The roots of z^12 - 1 = 0 -/
def twelfthRootsOfUnity : Finset ℂ := sorry

/-- The condition that v and w are distinct -/
def areDistinct (v w : ℂ) : Prop := v ≠ w

/-- The condition that v and w are roots of z^12 - 1 = 0 -/
def areRoots (v w : ℂ) : Prop := v ∈ twelfthRootsOfUnity ∧ w ∈ twelfthRootsOfUnity

/-- The number of pairs (v, w) satisfying |v + w| ≥ √2 -/
def satisfyingPairs : ℕ := sorry

/-- The total number of distinct pairs (v, w) -/
def totalPairs : ℕ := sorry

theorem probability_of_sum_magnitude_at_least_sqrt2 :
  satisfyingPairs / totalPairs = 10 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_magnitude_at_least_sqrt2_l3063_306330


namespace NUMINAMATH_CALUDE_ma_xiaotiao_rank_l3063_306380

theorem ma_xiaotiao_rank (total_participants : ℕ) (ma_rank : ℕ) : 
  total_participants = 34 →
  ma_rank > 0 →
  ma_rank ≤ total_participants →
  total_participants - ma_rank = 2 * (ma_rank - 1) →
  ma_rank = 12 := by
  sorry

end NUMINAMATH_CALUDE_ma_xiaotiao_rank_l3063_306380


namespace NUMINAMATH_CALUDE_special_linear_function_at_two_l3063_306337

/-- A linear function satisfying specific conditions -/
structure SpecialLinearFunction where
  f : ℝ → ℝ
  linear : ∀ x y c : ℝ, f (x + y) = f x + f y ∧ f (c * x) = c * f x
  inverse_relation : ∀ x : ℝ, f x = 3 * f⁻¹ x + 5
  f_one : f 1 = 5

/-- The main theorem stating the value of f(2) for the special linear function -/
theorem special_linear_function_at_two (slf : SpecialLinearFunction) :
  slf.f 2 = 2 * Real.sqrt 3 + (5 * Real.sqrt 3) / (Real.sqrt 3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_special_linear_function_at_two_l3063_306337


namespace NUMINAMATH_CALUDE_lighthouse_ship_position_l3063_306338

/-- Represents cardinal directions --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a relative position with a direction and angle --/
structure RelativePosition where
  primaryDirection : Direction
  secondaryDirection : Direction
  angle : ℝ

/-- Returns the opposite direction --/
def oppositeDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East

/-- Returns the opposite relative position --/
def oppositePosition (pos : RelativePosition) : RelativePosition :=
  { primaryDirection := oppositeDirection pos.primaryDirection,
    secondaryDirection := oppositeDirection pos.secondaryDirection,
    angle := pos.angle }

theorem lighthouse_ship_position 
  (lighthousePos : RelativePosition) 
  (h1 : lighthousePos.primaryDirection = Direction.North)
  (h2 : lighthousePos.secondaryDirection = Direction.East)
  (h3 : lighthousePos.angle = 38) :
  oppositePosition lighthousePos = 
    { primaryDirection := Direction.South,
      secondaryDirection := Direction.West,
      angle := 38 } := by
  sorry

end NUMINAMATH_CALUDE_lighthouse_ship_position_l3063_306338


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3063_306385

/-- Given a quadratic equation x^2 + kx - 2 = 0 where x = 1 is one root,
    prove that x = -2 is the other root. -/
theorem other_root_of_quadratic (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 2 = 0 → -2^2 + k * (-2) - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3063_306385


namespace NUMINAMATH_CALUDE_min_occupied_seats_l3063_306360

/-- Represents a row of seats -/
structure SeatRow :=
  (total : ℕ)
  (occupied : Finset ℕ)
  (h_occupied : occupied.card ≤ total)

/-- Predicts if a new person must sit next to someone -/
def mustSitNext (row : SeatRow) : Prop :=
  ∀ n : ℕ, n ≤ row.total → n ∉ row.occupied →
    (n > 1 ∧ n - 1 ∈ row.occupied) ∨ (n < row.total ∧ n + 1 ∈ row.occupied)

/-- The theorem to be proved -/
theorem min_occupied_seats :
  ∃ (row : SeatRow),
    row.total = 120 ∧
    row.occupied.card = 40 ∧
    mustSitNext row ∧
    ∀ (row' : SeatRow),
      row'.total = 120 →
      row'.occupied.card < 40 →
      ¬mustSitNext row' :=
by sorry

end NUMINAMATH_CALUDE_min_occupied_seats_l3063_306360


namespace NUMINAMATH_CALUDE_cheapest_caterer_l3063_306373

def first_caterer_cost (people : ℕ) : ℚ := 120 + 18 * people
def second_caterer_cost (people : ℕ) : ℚ := 250 + 15 * people

theorem cheapest_caterer (people : ℕ) :
  (people ≥ 44 → second_caterer_cost people ≤ first_caterer_cost people) ∧
  (people < 44 → second_caterer_cost people > first_caterer_cost people) :=
sorry

end NUMINAMATH_CALUDE_cheapest_caterer_l3063_306373


namespace NUMINAMATH_CALUDE_painting_survey_l3063_306319

theorem painting_survey (total : ℕ) (not_enjoy_not_understand : ℕ) (enjoy : ℕ) (understand : ℕ) :
  total = 440 →
  not_enjoy_not_understand = 110 →
  enjoy = understand →
  (enjoy : ℚ) / total = 3 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_painting_survey_l3063_306319


namespace NUMINAMATH_CALUDE_max_difference_l3063_306335

theorem max_difference (a b : ℝ) (ha : -5 ≤ a ∧ a ≤ 10) (hb : -5 ≤ b ∧ b ≤ 10) :
  ∃ (x y : ℝ), -5 ≤ x ∧ x ≤ 10 ∧ -5 ≤ y ∧ y ≤ 10 ∧ x - y = 15 ∧ ∀ (c d : ℝ), -5 ≤ c ∧ c ≤ 10 ∧ -5 ≤ d ∧ d ≤ 10 → c - d ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_l3063_306335


namespace NUMINAMATH_CALUDE_sets_equality_implies_x_minus_y_l3063_306361

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {1, x, y}
def B (x y : ℝ) : Set ℝ := {1, x^2, 2*y}

-- State the theorem
theorem sets_equality_implies_x_minus_y (x y : ℝ) : 
  A x y = B x y → x - y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_x_minus_y_l3063_306361


namespace NUMINAMATH_CALUDE_taehun_shortest_hair_l3063_306364

-- Define the hair lengths
def junseop_hair_cm : ℝ := 9
def junseop_hair_mm : ℝ := 8
def taehun_hair : ℝ := 8.9
def hayul_hair : ℝ := 9.3

-- Define the conversion factor from mm to cm
def mm_to_cm : ℝ := 0.1

-- Theorem statement
theorem taehun_shortest_hair :
  let junseop_total := junseop_hair_cm + junseop_hair_mm * mm_to_cm
  taehun_hair < junseop_total ∧ taehun_hair < hayul_hair := by sorry

end NUMINAMATH_CALUDE_taehun_shortest_hair_l3063_306364


namespace NUMINAMATH_CALUDE_inverse_division_identity_l3063_306323

theorem inverse_division_identity (x : ℝ) (hx : x ≠ 0) : 1 / x⁻¹ = x := by
  sorry

end NUMINAMATH_CALUDE_inverse_division_identity_l3063_306323


namespace NUMINAMATH_CALUDE_library_biography_increase_l3063_306334

theorem library_biography_increase (B : ℝ) (h1 : B > 0) : 
  let original_biographies := 0.20 * B
  let new_biographies := (7 / 9) * B
  let percentage_increase := (new_biographies / original_biographies - 1) * 100
  percentage_increase = 3500 / 9 := by sorry

end NUMINAMATH_CALUDE_library_biography_increase_l3063_306334


namespace NUMINAMATH_CALUDE_sum_of_products_l3063_306396

theorem sum_of_products (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 9)
  (h2 : y^2 + y*z + z^2 = 16)
  (h3 : z^2 + z*x + x^2 = 25) :
  x*y + y*z + z*x = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3063_306396


namespace NUMINAMATH_CALUDE_fiftieth_central_ring_number_l3063_306392

/-- Returns the number of digits in a positive integer -/
def numDigits (n : ℕ+) : ℕ :=
  (Nat.log 10 n.val) + 1

/-- Defines a Central Ring Number -/
def isCentralRingNumber (x : ℕ+) : Prop :=
  numDigits (3 * x) > numDigits x

/-- Returns the nth Central Ring Number -/
def nthCentralRingNumber (n : ℕ+) : ℕ+ :=
  sorry

theorem fiftieth_central_ring_number :
  nthCentralRingNumber 50 = 81 :=
sorry

end NUMINAMATH_CALUDE_fiftieth_central_ring_number_l3063_306392


namespace NUMINAMATH_CALUDE_carver_school_earnings_l3063_306316

/-- Represents a school with its student count and work days -/
structure School where
  name : String
  students : ℕ
  days : ℕ

/-- Calculates the total payment for all schools -/
def totalPayment (schools : List School) (basePayment dailyWage : ℚ) : ℚ :=
  (schools.map (fun s => s.students * s.days) |>.sum : ℕ) * dailyWage + 
  (schools.length : ℕ) * basePayment

/-- Calculates the earnings for a specific school -/
def schoolEarnings (school : School) (dailyWage : ℚ) : ℚ :=
  (school.students * school.days : ℕ) * dailyWage

theorem carver_school_earnings :
  let allen := School.mk "Allen" 7 3
  let balboa := School.mk "Balboa" 5 6
  let carver := School.mk "Carver" 4 10
  let schools := [allen, balboa, carver]
  let basePayment := 20
  ∃ dailyWage : ℚ,
    totalPayment schools basePayment dailyWage = 900 ∧
    schoolEarnings carver dailyWage = 369.60 := by
  sorry

end NUMINAMATH_CALUDE_carver_school_earnings_l3063_306316


namespace NUMINAMATH_CALUDE_fifteenth_prime_l3063_306340

/-- Given that 5 is the third prime number, prove that the fifteenth prime number is 59. -/
theorem fifteenth_prime : 
  (∃ (f : ℕ → ℕ), f 3 = 5 ∧ (∀ n, n ≥ 1 → Prime (f n)) ∧ (∀ n m, n < m → f n < f m)) → 
  (∃ (g : ℕ → ℕ), g 15 = 59 ∧ (∀ n, n ≥ 1 → Prime (g n)) ∧ (∀ n m, n < m → g n < g m)) :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l3063_306340


namespace NUMINAMATH_CALUDE_evaluate_expression_l3063_306397

theorem evaluate_expression : 
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3063_306397


namespace NUMINAMATH_CALUDE_sum_of_symmetric_points_coords_l3063_306384

/-- Two points P₁ and P₂ are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are the same. -/
def symmetric_wrt_y_axis (P₁ P₂ : ℝ × ℝ) : Prop :=
  P₁.1 = -P₂.1 ∧ P₁.2 = P₂.2

/-- Given two points P₁(a,-5) and P₂(3,b) that are symmetric with respect to the y-axis,
    prove that a + b = -8. -/
theorem sum_of_symmetric_points_coords (a b : ℝ) 
    (h : symmetric_wrt_y_axis (a, -5) (3, b)) : 
  a + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_points_coords_l3063_306384


namespace NUMINAMATH_CALUDE_max_weight_theorem_l3063_306389

def weight_set : Set ℕ := {2, 5, 10}

def is_measurable (w : ℕ) : Prop :=
  ∃ (a b c : ℕ), w = 2*a + 5*b + 10*c

def max_measurable : ℕ := 17

theorem max_weight_theorem :
  (∀ w : ℕ, is_measurable w → w ≤ max_measurable) ∧
  is_measurable max_measurable :=
sorry

end NUMINAMATH_CALUDE_max_weight_theorem_l3063_306389


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l3063_306354

/-- The path length of a vertex of an equilateral triangle rotating around a square --/
theorem triangle_rotation_path_length 
  (square_side : ℝ) 
  (triangle_side : ℝ) 
  (h_square : square_side = 6) 
  (h_triangle : triangle_side = 3) : 
  let path_length := 4 * 3 * (2 * π * triangle_side / 3)
  path_length = 24 * π := by sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l3063_306354


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3063_306355

/-- Given four distinct points P, A, B, C on a plane, prove that if 
    PA + PB + PC = 0 and AB + AC + m * AP = 0, then m = -3 -/
theorem vector_equation_solution (P A B C : EuclideanSpace ℝ (Fin 2)) 
    (h1 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
    (h2 : (A - P) + (B - P) + (C - P) = 0)
    (h3 : ∃ m : ℝ, (B - A) + (C - A) + m • (P - A) = 0) : 
  ∃ m : ℝ, (B - A) + (C - A) + m • (P - A) = 0 ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3063_306355


namespace NUMINAMATH_CALUDE_max_stores_visited_l3063_306368

theorem max_stores_visited (
  num_stores : ℕ) 
  (total_visits : ℕ) 
  (num_shoppers : ℕ) 
  (num_two_store_visitors : ℕ) 
  (h1 : num_stores = 8)
  (h2 : total_visits = 21)
  (h3 : num_shoppers = 12)
  (h4 : num_two_store_visitors = 8)
  (h5 : num_two_store_visitors ≤ num_shoppers)
  (h6 : ∀ n : ℕ, n ≤ num_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits = 3 ∧ 
  ∀ n : ℕ, n ≤ num_shoppers → ∃ k : ℕ, k ≤ max_visits ∧ 
  (num_two_store_visitors * 2 + (num_shoppers - num_two_store_visitors) * k = total_visits) :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l3063_306368


namespace NUMINAMATH_CALUDE_probability_two_from_ten_with_two_defective_l3063_306327

/-- The probability of drawing at least one defective product -/
def probability_at_least_one_defective (total : ℕ) (defective : ℕ) (draw : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) draw : ℚ) / (Nat.choose total draw : ℚ)

/-- Theorem stating the probability of drawing at least one defective product -/
theorem probability_two_from_ten_with_two_defective :
  probability_at_least_one_defective 10 2 2 = 17 / 45 := by
sorry

end NUMINAMATH_CALUDE_probability_two_from_ten_with_two_defective_l3063_306327


namespace NUMINAMATH_CALUDE_separation_sister_chromatids_not_in_first_division_l3063_306381

-- Define the events
inductive MeioticEvent
| PairingHomologousChromosomes
| CrossingOver
| SeparationSisterChromatids
| SeparationHomologousChromosomes

-- Define the property of occurring during the first meiotic division
def occursInFirstMeioticDivision : MeioticEvent → Prop :=
  fun event =>
    match event with
    | MeioticEvent.PairingHomologousChromosomes => True
    | MeioticEvent.CrossingOver => True
    | MeioticEvent.SeparationSisterChromatids => False
    | MeioticEvent.SeparationHomologousChromosomes => True

-- Theorem stating that separation of sister chromatids is the only event
-- that does not occur during the first meiotic division
theorem separation_sister_chromatids_not_in_first_division :
  ∀ (e : MeioticEvent),
    ¬occursInFirstMeioticDivision e ↔ e = MeioticEvent.SeparationSisterChromatids :=
by sorry

end NUMINAMATH_CALUDE_separation_sister_chromatids_not_in_first_division_l3063_306381


namespace NUMINAMATH_CALUDE_male_average_score_l3063_306345

theorem male_average_score 
  (female_count : ℕ) 
  (male_count : ℕ) 
  (total_count : ℕ) 
  (female_avg : ℚ) 
  (total_avg : ℚ) 
  (h1 : female_count = 20)
  (h2 : male_count = 30)
  (h3 : total_count = female_count + male_count)
  (h4 : female_avg = 75)
  (h5 : total_avg = 72) :
  (total_count * total_avg - female_count * female_avg) / male_count = 70 := by
sorry

end NUMINAMATH_CALUDE_male_average_score_l3063_306345


namespace NUMINAMATH_CALUDE_coord_relationship_l3063_306306

/-- The relationship between x and y coordinates on lines y = x or y = -x --/
theorem coord_relationship (x y : ℝ) : (y = x ∨ y = -x) → |x| - |y| = 0 := by
  sorry

end NUMINAMATH_CALUDE_coord_relationship_l3063_306306


namespace NUMINAMATH_CALUDE_intersection_condition_l3063_306351

def A : Set (ℕ × ℝ) := {p | 3 * p.1 + p.2 - 2 = 0}

def B (k : ℤ) : Set (ℕ × ℝ) := {p | k * (p.1^2 - p.1 + 1) - p.2 = 0}

theorem intersection_condition (k : ℤ) : 
  k ≠ 0 → (∃ p : ℕ × ℝ, p ∈ A ∩ B k) → k = -1 ∨ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3063_306351


namespace NUMINAMATH_CALUDE_inequality_solution_l3063_306349

def inequality (x : ℝ) : Prop :=
  1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24

theorem inequality_solution (x : ℝ) :
  inequality x → (x > -7 ∧ x < 1) ∨ (x > 3 ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3063_306349


namespace NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l3063_306347

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_n_with_specific_digit_sums :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_specific_digit_sums_l3063_306347


namespace NUMINAMATH_CALUDE_tangent_length_is_three_l3063_306386

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

/-- The point P -/
def P : ℝ × ℝ := (-1, 4)

/-- The length of the tangent line from a point to a circle -/
noncomputable def tangentLength (p : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The length of the tangent line from P to the circle is 3 -/
theorem tangent_length_is_three :
  tangentLength P = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_length_is_three_l3063_306386


namespace NUMINAMATH_CALUDE_modular_inverse_7_mod_29_l3063_306390

theorem modular_inverse_7_mod_29 :
  ∃ x : ℕ, x < 29 ∧ (7 * x) % 29 = 1 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_7_mod_29_l3063_306390


namespace NUMINAMATH_CALUDE_yoongi_multiplication_l3063_306388

theorem yoongi_multiplication (n : ℚ) : n * 15 = 45 → n - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_multiplication_l3063_306388


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_radii_l3063_306312

theorem tetrahedron_sphere_radii (r : ℝ) (R : ℝ) :
  r = Real.sqrt 2 - 1 →
  R = Real.sqrt 6 + 1 →
  ∃ (a : ℝ),
    r = (a * Real.sqrt 2) / 4 ∧
    R = (a * Real.sqrt 6) / 4 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_radii_l3063_306312


namespace NUMINAMATH_CALUDE_sara_golf_balls_l3063_306322

theorem sara_golf_balls (x : ℕ) : x = 16 * (3 * 4) → x / 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l3063_306322


namespace NUMINAMATH_CALUDE_measure_10_liters_l3063_306362

/-- Represents the state of water in two containers -/
structure WaterState :=
  (container1 : ℕ)  -- Amount of water in container 1 (11-liter container)
  (container2 : ℕ)  -- Amount of water in container 2 (9-liter container)

/-- Defines the possible operations on the water containers -/
inductive WaterOperation
  | Fill1      -- Fill container 1
  | Fill2      -- Fill container 2
  | Empty1     -- Empty container 1
  | Empty2     -- Empty container 2
  | Pour1to2   -- Pour from container 1 to container 2
  | Pour2to1   -- Pour from container 2 to container 1

/-- Applies a single operation to a water state -/
def applyOperation (state : WaterState) (op : WaterOperation) : WaterState :=
  match op with
  | WaterOperation.Fill1    => { container1 := 11, container2 := state.container2 }
  | WaterOperation.Fill2    => { container1 := state.container1, container2 := 9 }
  | WaterOperation.Empty1   => { container1 := 0,  container2 := state.container2 }
  | WaterOperation.Empty2   => { container1 := state.container1, container2 := 0 }
  | WaterOperation.Pour1to2 => 
      let amount := min state.container1 (9 - state.container2)
      { container1 := state.container1 - amount, container2 := state.container2 + amount }
  | WaterOperation.Pour2to1 => 
      let amount := min state.container2 (11 - state.container1)
      { container1 := state.container1 + amount, container2 := state.container2 - amount }

/-- Theorem: It is possible to measure out exactly 10 liters of water -/
theorem measure_10_liters : ∃ (ops : List WaterOperation), 
  (ops.foldl applyOperation { container1 := 0, container2 := 0 }).container1 = 10 ∨
  (ops.foldl applyOperation { container1 := 0, container2 := 0 }).container2 = 10 :=
sorry

end NUMINAMATH_CALUDE_measure_10_liters_l3063_306362


namespace NUMINAMATH_CALUDE_inequality_proof_l3063_306305

theorem inequality_proof (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3063_306305


namespace NUMINAMATH_CALUDE_min_cost_tree_purchase_l3063_306356

/-- Represents the cost and quantity of trees --/
structure TreePurchase where
  cypress_price : ℕ
  pine_price : ℕ
  cypress_count : ℕ
  pine_count : ℕ

/-- The conditions of the tree purchasing problem --/
def tree_problem (p : TreePurchase) : Prop :=
  2 * p.cypress_price + 3 * p.pine_price = 850 ∧
  3 * p.cypress_price + 2 * p.pine_price = 900 ∧
  p.cypress_count + p.pine_count = 80 ∧
  p.cypress_count ≥ 2 * p.pine_count

/-- The total cost of a tree purchase --/
def total_cost (p : TreePurchase) : ℕ :=
  p.cypress_price * p.cypress_count + p.pine_price * p.pine_count

/-- The theorem stating the minimum cost and optimal purchase --/
theorem min_cost_tree_purchase :
  ∃ (p : TreePurchase), tree_problem p ∧
    total_cost p = 14700 ∧
    p.cypress_count = 54 ∧
    p.pine_count = 26 ∧
    (∀ (q : TreePurchase), tree_problem q → total_cost q ≥ total_cost p) :=
by sorry

end NUMINAMATH_CALUDE_min_cost_tree_purchase_l3063_306356


namespace NUMINAMATH_CALUDE_last_digit_padic_fermat_l3063_306320

/-- Represents a p-adic integer with a non-zero last digit -/
structure PAdic (p : ℕ) where
  digits : ℕ → ℕ
  last_nonzero : digits 0 ≠ 0
  bound : ∀ n, digits n < p

/-- The last digit of a p-adic number -/
def last_digit {p : ℕ} (a : PAdic p) : ℕ := a.digits 0

/-- Exponentiation for p-adic numbers -/
def padic_pow {p : ℕ} (a : PAdic p) (n : ℕ) : PAdic p :=
  sorry

/-- Subtraction for p-adic numbers -/
def padic_sub {p : ℕ} (a b : PAdic p) : PAdic p :=
  sorry

theorem last_digit_padic_fermat (p : ℕ) (hp : Prime p) (a : PAdic p) :
  last_digit (padic_sub (padic_pow a (p - 1)) (PAdic.mk (λ _ => 1) sorry sorry)) = 0 :=
sorry

end NUMINAMATH_CALUDE_last_digit_padic_fermat_l3063_306320


namespace NUMINAMATH_CALUDE_spinner_probability_l3063_306348

theorem spinner_probability : 
  ∀ (p_A p_B p_C p_D p_E : ℚ),
  p_A = 2/7 →
  p_B = 3/14 →
  p_C = p_E →
  p_D = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3063_306348


namespace NUMINAMATH_CALUDE_sum_divisible_by_101_iff_digits_congruent_l3063_306374

/-- Represents a four-digit positive integer with different non-zero digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The value of the reverse of a four-digit number -/
def reverse_value (n : FourDigitNumber) : Nat :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- The theorem stating the condition for the sum of a number and its reverse to be divisible by 101 -/
theorem sum_divisible_by_101_iff_digits_congruent (n : FourDigitNumber) :
  (value n + reverse_value n) % 101 = 0 ↔ (n.a + n.d) % 101 = (n.b + n.c) % 101 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_101_iff_digits_congruent_l3063_306374


namespace NUMINAMATH_CALUDE_major_premise_incorrect_l3063_306311

theorem major_premise_incorrect : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_major_premise_incorrect_l3063_306311


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3063_306325

theorem triangle_perimeter_range (a b c A B C : ℝ) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  c = 2 →
  a * Real.cos B + b * Real.cos A = (Real.sqrt 3 * c) / (2 * Real.sin C) →
  A + B + C = π →
  let P := a + b + c
  4 < P ∧ P ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3063_306325


namespace NUMINAMATH_CALUDE_helen_oranges_l3063_306369

/-- Given that Helen starts with 9 oranges and receives 29 more from Ann, 
    prove that she ends up with 38 oranges in total. -/
theorem helen_oranges : 
  let initial_oranges : ℕ := 9
  let oranges_from_ann : ℕ := 29
  initial_oranges + oranges_from_ann = 38 := by sorry

end NUMINAMATH_CALUDE_helen_oranges_l3063_306369


namespace NUMINAMATH_CALUDE_sum_of_interchanged_digits_divisible_by_11_l3063_306371

theorem sum_of_interchanged_digits_divisible_by_11 (a b : ℕ) 
  (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  ∃ k : ℕ, (10 * a + b) + (10 * b + a) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interchanged_digits_divisible_by_11_l3063_306371


namespace NUMINAMATH_CALUDE_constant_term_zero_implies_m_zero_l3063_306366

theorem constant_term_zero_implies_m_zero :
  ∀ m : ℝ, (m^2 - m = 0) → (m = 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_zero_implies_m_zero_l3063_306366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_of_d_l3063_306346

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_range_of_d (d : ℝ) :
  (arithmetic_sequence 24 d 9 ≥ 0 ∧ arithmetic_sequence 24 d 10 < 0) →
  -3 ≤ d ∧ d < -8/3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_of_d_l3063_306346


namespace NUMINAMATH_CALUDE_travelers_meeting_l3063_306307

/-- The problem of two travelers meeting --/
theorem travelers_meeting
  (total_distance : ℝ)
  (travel_time : ℝ)
  (shook_speed : ℝ)
  (h_total_distance : total_distance = 490)
  (h_travel_time : travel_time = 7)
  (h_shook_speed : shook_speed = 37)
  : ∃ (beta_speed : ℝ),
    beta_speed = 33 ∧
    total_distance = shook_speed * travel_time + beta_speed * travel_time :=
by
  sorry

#check travelers_meeting

end NUMINAMATH_CALUDE_travelers_meeting_l3063_306307


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3063_306328

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3063_306328


namespace NUMINAMATH_CALUDE_fraction_simplification_l3063_306313

theorem fraction_simplification : (5 * 6 - 4) / 8 = 13 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3063_306313


namespace NUMINAMATH_CALUDE_vasily_salary_higher_l3063_306341

/-- Represents the salary distribution for graduates --/
structure GraduateSalary where
  high : ℝ  -- Salary for 1/5 of graduates
  very_high : ℝ  -- Salary for 1/10 of graduates
  low : ℝ  -- Salary for 1/20 of graduates
  medium : ℝ  -- Salary for remaining graduates

/-- Calculates the expected salary for a student --/
def expected_salary (
  total_students : ℕ
  ) (graduating_students : ℕ
  ) (non_graduate_salary : ℝ
  ) (graduate_salary : GraduateSalary
  ) : ℝ :=
  sorry

/-- Calculates the salary after a number of years with annual increase --/
def salary_after_years (
  initial_salary : ℝ
  ) (annual_increase : ℝ
  ) (years : ℕ
  ) : ℝ :=
  sorry

theorem vasily_salary_higher (
  total_students : ℕ
  ) (graduating_students : ℕ
  ) (non_graduate_salary : ℝ
  ) (graduate_salary : GraduateSalary
  ) (fyodor_initial_salary : ℝ
  ) (fyodor_annual_increase : ℝ
  ) (years : ℕ
  ) : 
  total_students = 300 →
  graduating_students = 270 →
  non_graduate_salary = 25000 →
  graduate_salary.high = 60000 →
  graduate_salary.very_high = 80000 →
  graduate_salary.low = 25000 →
  graduate_salary.medium = 40000 →
  fyodor_initial_salary = 25000 →
  fyodor_annual_increase = 3000 →
  years = 4 →
  expected_salary total_students graduating_students non_graduate_salary graduate_salary = 39625 ∧
  expected_salary total_students graduating_students non_graduate_salary graduate_salary - 
    salary_after_years fyodor_initial_salary fyodor_annual_increase years = 2625 :=
by sorry

end NUMINAMATH_CALUDE_vasily_salary_higher_l3063_306341


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3063_306317

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_second_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_first : a 1 = 2)
  (h_relation : 16 * a 3 * a 5 = 8 * a 4 - 1) :
  a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3063_306317


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3063_306363

/-- The volume of a right circular cone formed by rolling up a five-sixth sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 6
  let base_radius : ℝ := sector_fraction * r
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * height
  volume = (25 / 3) * Real.pi * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3063_306363


namespace NUMINAMATH_CALUDE_max_m_value_min_weighted_sum_of_squares_l3063_306342

-- Part 1
theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2*m) → m ≤ 1 :=
sorry

-- Part 2
theorem min_weighted_sum_of_squares :
  let f (a b c : ℝ) := 4*a^2 + 9*b^2 + c^2
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    f a b c ≥ 36/49 ∧
    (f a b c = 36/49 ↔ a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_min_weighted_sum_of_squares_l3063_306342


namespace NUMINAMATH_CALUDE_image_of_one_three_preimage_of_one_three_l3063_306383

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Define the set A (which is the same as B)
def A : Set (ℝ × ℝ) := Set.univ

-- Theorem for the image of (1,3)
theorem image_of_one_three : f (1, 3) = (4, -2) := by sorry

-- Theorem for the preimage of (1,3)
theorem preimage_of_one_three : f (2, -1) = (1, 3) := by sorry

end NUMINAMATH_CALUDE_image_of_one_three_preimage_of_one_three_l3063_306383
