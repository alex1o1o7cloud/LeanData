import Mathlib

namespace NUMINAMATH_CALUDE_max_profit_l2007_200724

/-- Represents the shopping mall's helmet purchasing problem --/
structure HelmetProblem where
  costA : ℕ → ℕ  -- Cost function for type A helmets
  costB : ℕ → ℕ  -- Cost function for type B helmets
  sellA : ℕ      -- Selling price of type A helmet
  sellB : ℕ      -- Selling price of type B helmet
  totalHelmets : ℕ  -- Total number of helmets to purchase
  maxCost : ℕ    -- Maximum total cost
  minProfit : ℕ  -- Minimum required profit

/-- Calculates the profit for a given number of type A helmets --/
def profit (p : HelmetProblem) (numA : ℕ) : ℤ :=
  let numB := p.totalHelmets - numA
  (p.sellA - p.costA 1) * numA + (p.sellB - p.costB 1) * numB

/-- Theorem stating the maximum profit configuration --/
theorem max_profit (p : HelmetProblem) : 
  p.costA 8 + p.costB 6 = 630 →
  p.costA 6 + p.costB 8 = 700 →
  p.sellA = 58 →
  p.sellB = 98 →
  p.totalHelmets = 200 →
  p.maxCost = 10200 →
  p.minProfit = 6180 →
  p.costA 1 = 30 →
  p.costB 1 = 65 →
  (∀ n : ℕ, n ≤ p.totalHelmets → 
    p.costA n + p.costB (p.totalHelmets - n) ≤ p.maxCost →
    profit p n ≥ p.minProfit →
    profit p n ≤ profit p 80) ∧
  profit p 80 = 6200 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_l2007_200724


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2007_200745

/-- Proves that the retail price of a machine is $120 given specific conditions --/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2007_200745


namespace NUMINAMATH_CALUDE_initial_group_size_l2007_200773

theorem initial_group_size (average_increase : ℝ) (old_weight new_weight : ℝ) :
  average_increase = 3.5 ∧ old_weight = 47 ∧ new_weight = 68 →
  (new_weight - old_weight) / average_increase = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l2007_200773


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2007_200717

/-- A quadratic equation x^2 + kx + 1 = 0 has two equal real roots if and only if k = ±2 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ (∀ y : ℝ, y^2 + k*y + 1 = 0 → y = x)) ↔ 
  k = 2 ∨ k = -2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2007_200717


namespace NUMINAMATH_CALUDE_tired_painting_time_l2007_200728

/-- Represents the time needed to paint houses -/
def paint_time (people : ℕ) (houses : ℕ) (hours : ℝ) (efficiency : ℝ) : Prop :=
  people * hours * efficiency = houses * 32

theorem tired_painting_time :
  paint_time 8 2 4 1 →
  paint_time 5 2 8 0.8 :=
by
  sorry

end NUMINAMATH_CALUDE_tired_painting_time_l2007_200728


namespace NUMINAMATH_CALUDE_cucumber_salad_problem_l2007_200751

theorem cucumber_salad_problem (total : ℕ) (cucumber : ℕ) (tomato : ℕ) : 
  total = 280 →
  tomato = 3 * cucumber →
  total = cucumber + tomato →
  cucumber = 70 := by
sorry

end NUMINAMATH_CALUDE_cucumber_salad_problem_l2007_200751


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2007_200788

theorem x_minus_y_values (x y : ℝ) 
  (h1 : |x| = 5)
  (h2 : y^2 = 16)
  (h3 : x + y > 0) :
  x - y = 1 ∨ x - y = 9 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2007_200788


namespace NUMINAMATH_CALUDE_paper_width_calculation_l2007_200748

theorem paper_width_calculation (length : Real) (comparison_width : Real) (area_difference : Real) :
  length = 11 →
  comparison_width = 4.5 →
  area_difference = 100 →
  ∃ width : Real,
    2 * length * width = 2 * comparison_width * length + area_difference ∧
    width = 199 / 22 := by
  sorry

end NUMINAMATH_CALUDE_paper_width_calculation_l2007_200748


namespace NUMINAMATH_CALUDE_x_equals_y_l2007_200737

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l2007_200737


namespace NUMINAMATH_CALUDE_train_speed_l2007_200726

/-- Proves that a train with given length crossing a bridge with given length in a given time has a specific speed -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 160 →
  bridge_length = 215 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2007_200726


namespace NUMINAMATH_CALUDE_infinite_symmetric_subset_exists_l2007_200718

/-- A color type representing black and white --/
inductive Color
| Black
| White

/-- A point in the plane with integer coordinates --/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point with integer coordinates --/
def Coloring := Point → Color

/-- A set of points is symmetric about a center point if for every point in the set,
    its reflection about the center is also in the set --/
def IsSymmetric (S : Set Point) (center : Point) : Prop :=
  ∀ p ∈ S, Point.mk (2 * center.x - p.x) (2 * center.y - p.y) ∈ S

/-- The main theorem stating the existence of an infinite symmetric subset --/
theorem infinite_symmetric_subset_exists (coloring : Coloring) :
  ∃ (S : Set Point) (c : Color) (center : Point),
    Set.Infinite S ∧ (∀ p ∈ S, coloring p = c) ∧ IsSymmetric S center :=
  sorry

end NUMINAMATH_CALUDE_infinite_symmetric_subset_exists_l2007_200718


namespace NUMINAMATH_CALUDE_percentage_and_subtraction_l2007_200771

theorem percentage_and_subtraction (y : ℝ) : 
  (20 : ℝ) / y = (80 : ℝ) / 100 → y = 25 ∧ y - 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_and_subtraction_l2007_200771


namespace NUMINAMATH_CALUDE_hospital_staff_remaining_l2007_200753

/-- Given a hospital with an initial count of doctors and nurses,
    calculate the remaining staff after some quit. -/
def remaining_staff (initial_doctors : ℕ) (initial_nurses : ℕ)
                    (doctors_quit : ℕ) (nurses_quit : ℕ) : ℕ :=
  (initial_doctors - doctors_quit) + (initial_nurses - nurses_quit)

/-- Theorem stating that with 11 doctors and 18 nurses initially,
    if 5 doctors and 2 nurses quit, 22 staff members remain. -/
theorem hospital_staff_remaining :
  remaining_staff 11 18 5 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_remaining_l2007_200753


namespace NUMINAMATH_CALUDE_max_abs_z_l2007_200754

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 5 - 12*I) = 3) : 
  ∃ (max_abs : ℝ), max_abs = 16 ∧ Complex.abs z ≤ max_abs ∧ 
  ∀ (w : ℂ), Complex.abs (w + 5 - 12*I) = 3 → Complex.abs w ≤ max_abs :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_l2007_200754


namespace NUMINAMATH_CALUDE_complex_sum_parts_zero_l2007_200704

theorem complex_sum_parts_zero (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 1 / (i * (1 - i))
  a + b = 0 ∧ z = Complex.mk a b :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_parts_zero_l2007_200704


namespace NUMINAMATH_CALUDE_piece_exits_at_A2_l2007_200757

/-- Represents the directions a piece can move --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a cell on the 4x4 board --/
structure Cell where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the board --/
structure BoardState where
  piece_position : Cell
  arrows : Fin 4 → Fin 4 → Direction

/-- Defines the initial state of the board --/
def initial_state : BoardState := sorry

/-- Defines a single move on the board --/
def move (state : BoardState) : BoardState := sorry

/-- Checks if a cell is on the edge of the board --/
def is_edge_cell (cell : Cell) : Bool := sorry

/-- Simulates the movement of the piece until it exits the board --/
def simulate_until_exit (state : BoardState) : Cell := sorry

/-- The main theorem to prove --/
theorem piece_exits_at_A2 :
  let final_cell := simulate_until_exit initial_state
  final_cell.row = 0 ∧ final_cell.col = 1 := by sorry

end NUMINAMATH_CALUDE_piece_exits_at_A2_l2007_200757


namespace NUMINAMATH_CALUDE_count_monomials_is_four_l2007_200794

/-- An algebraic expression is a monomial if it consists of a single term. -/
def is_monomial (expr : String) : Bool :=
  match expr with
  | "-1" => true
  | "-2/3*a^2" => true
  | "1/6*x^2*y" => true
  | "3a+b" => false
  | "0" => true
  | "(x-1)/2" => false
  | _ => false

/-- The list of algebraic expressions to be checked. -/
def expressions : List String := ["-1", "-2/3*a^2", "1/6*x^2*y", "3a+b", "0", "(x-1)/2"]

/-- Theorem stating that the number of monomials in the given list of expressions is 4. -/
theorem count_monomials_is_four :
  (expressions.filter is_monomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_monomials_is_four_l2007_200794


namespace NUMINAMATH_CALUDE_eight_div_repeating_third_l2007_200779

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by 0.3333... --/
theorem eight_div_repeating_third : 8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_div_repeating_third_l2007_200779


namespace NUMINAMATH_CALUDE_ball_placement_methods_l2007_200752

/-- The number of different balls -/
def n : ℕ := 4

/-- The number of different boxes -/
def m : ℕ := 4

/-- The number of ways to place n different balls into m different boxes without any empty boxes -/
def noEmptyBoxes (n m : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to place n different balls into m different boxes allowing empty boxes -/
def allowEmptyBoxes (n m : ℕ) : ℕ := m ^ n

/-- The number of ways to place n different balls into m different boxes with exactly one box left empty -/
def oneEmptyBox (n m : ℕ) : ℕ := 
  Nat.choose m 1 * Nat.choose n (m - 1) * Nat.factorial (m - 1)

theorem ball_placement_methods :
  noEmptyBoxes n m = 24 ∧
  allowEmptyBoxes n m = 256 ∧
  oneEmptyBox n m = 144 := by sorry

end NUMINAMATH_CALUDE_ball_placement_methods_l2007_200752


namespace NUMINAMATH_CALUDE_car_speed_equality_l2007_200765

/-- Proves that given the conditions of the car problem, the average speed of Car Y equals that of Car X -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_start : ℝ) : 
  speed_x = 35 →
  start_delay = 72 / 60 →
  distance_after_y_start = 210 →
  ∃ (time_after_y_start : ℝ), 
    time_after_y_start > 0 ∧ 
    speed_x * time_after_y_start = distance_after_y_start ∧
    distance_after_y_start / time_after_y_start = speed_x := by
  sorry

#check car_speed_equality

end NUMINAMATH_CALUDE_car_speed_equality_l2007_200765


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2007_200783

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2007_200783


namespace NUMINAMATH_CALUDE_population_change_approx_19_58_percent_l2007_200705

/-- Represents the population change over three years given specific growth and decrease rates -/
def population_change (natural_growth : ℝ) (migration_year1 : ℝ) (migration_year2 : ℝ) (migration_year3 : ℝ) (disaster_decrease : ℝ) : ℝ :=
  let year1 := (1 + natural_growth) * (1 + migration_year1)
  let year2 := (1 + natural_growth) * (1 + migration_year2)
  let year3 := (1 + natural_growth) * (1 + migration_year3)
  let three_year_change := year1 * year2 * year3
  three_year_change * (1 - disaster_decrease)

/-- Theorem stating that the population change over three years is approximately 19.58% -/
theorem population_change_approx_19_58_percent :
  let natural_growth := 0.09
  let migration_year1 := -0.01
  let migration_year2 := -0.015
  let migration_year3 := -0.02
  let disaster_decrease := 0.03
  abs (population_change natural_growth migration_year1 migration_year2 migration_year3 disaster_decrease - 1.1958) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_population_change_approx_19_58_percent_l2007_200705


namespace NUMINAMATH_CALUDE_min_value_of_function_l2007_200766

open Real

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let f := fun (x : ℝ) => x - 1 - (log x) / x
  (∀ y > 0, f y ≥ 0) ∧ (∃ z > 0, f z = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2007_200766


namespace NUMINAMATH_CALUDE_hiker_journey_distance_l2007_200727

/-- Represents the hiker's journey --/
structure HikerJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the hiker's journey --/
def journey_conditions (j : HikerJourney) : Prop :=
  j.distance = j.speed * j.time ∧
  j.distance = (j.speed + 1) * (3/4 * j.time) ∧
  j.distance = (j.speed - 1) * (j.time + 3)

/-- The theorem statement --/
theorem hiker_journey_distance :
  ∀ j : HikerJourney, journey_conditions j → j.distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_hiker_journey_distance_l2007_200727


namespace NUMINAMATH_CALUDE_corrected_mean_l2007_200793

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 100 ∧ original_mean = 45 ∧ incorrect_value = 32 ∧ correct_value = 87 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (45 + 55 / 100) :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l2007_200793


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2007_200701

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ (Set.univ \ N) = {x : ℝ | -2 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2007_200701


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l2007_200749

theorem sum_geq_sqrt_three (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (h : a * b + b * c + c * a = 1) : a + b + c ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l2007_200749


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_parallel_implies_perpendicular_planes_l2007_200742

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_implies_parallel 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

-- Theorem 2
theorem perpendicular_parallel_implies_perpendicular_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_parallel_implies_perpendicular_planes_l2007_200742


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2007_200744

/-- Calculates the total distance travelled by a car with increasing speed over a given number of hours -/
def totalDistance (initialDistance : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  let distanceList := List.range hours |>.map (fun h => initialDistance + h * speedIncrease)
  distanceList.sum

/-- Theorem stating that a car with given initial speed and speed increase travels 546 km in 12 hours -/
theorem car_distance_theorem :
  totalDistance 35 2 12 = 546 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2007_200744


namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l2007_200716

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a + 2*b) + b / (a + b) ≥ (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a + 2*b) + b / (a + b) = (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) ↔ a = Real.sqrt 2 * b :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l2007_200716


namespace NUMINAMATH_CALUDE_cube_root_equation_sum_l2007_200782

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = Real.rpow x.val (1/3) + Real.rpow y.val (1/3) - Real.rpow z.val (1/3)) →
  x.val + y.val + z.val = 79 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_sum_l2007_200782


namespace NUMINAMATH_CALUDE_age_of_b_l2007_200776

theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 29 → 
  (a + c) / 2 = 32 → 
  b = 23 := by
sorry

end NUMINAMATH_CALUDE_age_of_b_l2007_200776


namespace NUMINAMATH_CALUDE_range_of_f_l2007_200731

def f (x : ℕ) : ℕ := 2 * x + 1

def domain : Set ℕ := {1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2007_200731


namespace NUMINAMATH_CALUDE_exercise_minimum_sets_l2007_200750

/-- Represents the exercise routine over 100 days -/
structure ExerciseRoutine where
  pushups_per_set : ℕ
  pullups_per_set : ℕ
  initial_reps : ℕ
  days : ℕ

/-- Calculates the total number of repetitions over the given days -/
def total_reps (routine : ExerciseRoutine) : ℕ :=
  routine.days * (2 * routine.initial_reps + routine.days - 1) / 2

/-- Represents the solution to the exercise problem -/
structure ExerciseSolution where
  pushup_sets : ℕ
  pullup_sets : ℕ

/-- Theorem stating the minimum number of sets for push-ups and pull-ups -/
theorem exercise_minimum_sets (routine : ExerciseRoutine) 
  (h1 : routine.pushups_per_set = 8)
  (h2 : routine.pullups_per_set = 5)
  (h3 : routine.initial_reps = 41)
  (h4 : routine.days = 100) :
  ∃ (solution : ExerciseSolution), 
    solution.pushup_sets ≥ 100 ∧ 
    solution.pullup_sets ≥ 106 ∧
    solution.pushup_sets * routine.pushups_per_set + 
    solution.pullup_sets * routine.pullups_per_set = 
    total_reps routine :=
  sorry

end NUMINAMATH_CALUDE_exercise_minimum_sets_l2007_200750


namespace NUMINAMATH_CALUDE_cube_surface_area_l2007_200702

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1728 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 864 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2007_200702


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l2007_200714

theorem linear_function_decreasing (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -3 * x₁ - 7 →
  y₂ = -3 * x₂ - 7 →
  x₁ > x₂ →
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l2007_200714


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l2007_200719

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
by sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l2007_200719


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l2007_200795

theorem smallest_base_for_fourth_power (b : ℕ) (N : ℕ) : b = 18 ↔ 
  (∃ (x : ℕ), N = x^4) ∧ 
  (11 * 30 * N).digits b = [7, 7, 7] ∧ 
  ∀ (b' : ℕ), b' < b → 
    ¬(∃ (N' : ℕ) (x' : ℕ), 
      N' = x'^4 ∧ 
      (11 * 30 * N').digits b' = [7, 7, 7]) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l2007_200795


namespace NUMINAMATH_CALUDE_gcd_47_power_plus_one_l2007_200720

theorem gcd_47_power_plus_one : Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_47_power_plus_one_l2007_200720


namespace NUMINAMATH_CALUDE_fraction_problem_l2007_200792

theorem fraction_problem (f : ℚ) : f * (-72 : ℚ) = -60 → f = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2007_200792


namespace NUMINAMATH_CALUDE_game_ends_after_17_rounds_l2007_200777

/-- Represents a player in the token game -/
structure Player where
  name : String
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Determines if the game has ended -/
def gameEnded (state : GameState) : Bool :=
  state.players.any (fun p => p.tokens = 0)

/-- Updates the game state for one round -/
def updateGameState (state : GameState) : GameState :=
  sorry -- Implementation details omitted

/-- Runs the game until it ends -/
def runGame (initialState : GameState) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the game ends after 17 rounds -/
theorem game_ends_after_17_rounds :
  let initialState := GameState.mk
    [Player.mk "A" 20, Player.mk "B" 18, Player.mk "C" 16]
    0
  runGame initialState = 17 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_after_17_rounds_l2007_200777


namespace NUMINAMATH_CALUDE_conjunction_implies_disjunction_l2007_200725

theorem conjunction_implies_disjunction (p q : Prop) : (p ∧ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_conjunction_implies_disjunction_l2007_200725


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2007_200798

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2007_200798


namespace NUMINAMATH_CALUDE_octal_is_smallest_l2007_200723

-- Define the base conversion function
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

-- Define the given numbers
def binary : Nat := toDecimal [1, 0, 1, 0, 1, 0] 2
def quinary : Nat := toDecimal [1, 1, 1] 5
def octal : Nat := toDecimal [3, 2] 8
def senary : Nat := toDecimal [5, 4] 6

-- Theorem statement
theorem octal_is_smallest : 
  octal ≤ binary ∧ octal ≤ quinary ∧ octal ≤ senary :=
sorry

end NUMINAMATH_CALUDE_octal_is_smallest_l2007_200723


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l2007_200709

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l2007_200709


namespace NUMINAMATH_CALUDE_paint_usage_l2007_200778

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) :
  initial_paint = 360 ∧
  first_week_fraction = 1/4 ∧
  second_week_fraction = 1/2 →
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 225 :=
by sorry

end NUMINAMATH_CALUDE_paint_usage_l2007_200778


namespace NUMINAMATH_CALUDE_value_of_x_l2007_200787

theorem value_of_x : ∃ x : ℝ, 3 * x + 15 = (1/3) * (7 * x + 45) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2007_200787


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2007_200730

theorem shaded_area_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h1 : total_squares = 5) 
  (h2 : shaded_squares = 2) : 
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2007_200730


namespace NUMINAMATH_CALUDE_expression_equals_three_l2007_200791

theorem expression_equals_three (x : ℝ) (h : x^2 - 4*x = 5) : 
  ∃ (f : ℝ → ℝ), f x = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_three_l2007_200791


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l2007_200736

theorem recurring_decimal_sum : 
  (1 : ℚ) / 3 + 4 / 99 + 5 / 999 = 42 / 111 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l2007_200736


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l2007_200758

theorem shirt_cost_problem (total_cost : ℕ) (num_shirts : ℕ) (known_shirt_cost : ℕ) (num_known_shirts : ℕ) :
  total_cost = 85 →
  num_shirts = 5 →
  known_shirt_cost = 15 →
  num_known_shirts = 3 →
  ∃ (remaining_shirt_cost : ℕ),
    remaining_shirt_cost * (num_shirts - num_known_shirts) + known_shirt_cost * num_known_shirts = total_cost ∧
    remaining_shirt_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l2007_200758


namespace NUMINAMATH_CALUDE_distance_between_points_l2007_200741

/-- The distance between two points A(4,-3) and B(4,5) is 8. -/
theorem distance_between_points : Real.sqrt ((4 - 4)^2 + (5 - (-3))^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2007_200741


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_54000_perfect_cube_l2007_200796

/-- 
A number is a perfect cube if it can be expressed as the cube of an integer.
-/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/--
The smallest positive integer that, when multiplied by 54000, results in a perfect cube is 1.
-/
theorem smallest_multiplier_for_54000_perfect_cube :
  ∀ n : ℕ+, is_perfect_cube (54000 * n) → 1 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_54000_perfect_cube_l2007_200796


namespace NUMINAMATH_CALUDE_complex_sum_square_l2007_200772

variable (a b c : ℂ)

theorem complex_sum_square (h1 : a^2 + a*b + b^2 = 1 + I)
                           (h2 : b^2 + b*c + c^2 = -2)
                           (h3 : c^2 + c*a + a^2 = 1) :
  (a*b + b*c + c*a)^2 = (-11 - 4*I) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_square_l2007_200772


namespace NUMINAMATH_CALUDE_x_abs_x_is_k_function_l2007_200786

/-- Definition of a K function -/
def is_k_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0)

/-- The function f(x) = x|x| -/
def f (x : ℝ) : ℝ := x * |x|

/-- Theorem: f(x) = x|x| is a K function -/
theorem x_abs_x_is_k_function : is_k_function f := by sorry

end NUMINAMATH_CALUDE_x_abs_x_is_k_function_l2007_200786


namespace NUMINAMATH_CALUDE_siblings_water_consumption_l2007_200706

def cups_per_week (daily_cups : ℕ) : ℕ := daily_cups * 7

theorem siblings_water_consumption :
  let theo_daily := 8
  let mason_daily := 7
  let roxy_daily := 9
  let zara_daily := 10
  let lily_daily := 6
  cups_per_week theo_daily +
  cups_per_week mason_daily +
  cups_per_week roxy_daily +
  cups_per_week zara_daily +
  cups_per_week lily_daily = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_siblings_water_consumption_l2007_200706


namespace NUMINAMATH_CALUDE_min_value_expression_l2007_200761

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ m : ℝ, m = -500000 ∧ ∀ x y : ℝ, x > 0 → y > 0 →
    (x + 1/y) * (x + 1/y - 1000) + (y + 1/x) * (y + 1/x - 1000) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2007_200761


namespace NUMINAMATH_CALUDE_lexie_paintings_l2007_200740

theorem lexie_paintings (num_rooms : ℕ) (paintings_per_room : ℕ) 
  (h1 : num_rooms = 4) 
  (h2 : paintings_per_room = 8) : 
  num_rooms * paintings_per_room = 32 := by
sorry

end NUMINAMATH_CALUDE_lexie_paintings_l2007_200740


namespace NUMINAMATH_CALUDE_equation_solution_l2007_200710

theorem equation_solution : 
  ∃ x : ℚ, (x - 50) / 3 = (5 - 3 * x) / 4 + 2 ∧ x = 287 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2007_200710


namespace NUMINAMATH_CALUDE_squares_in_figure_100_l2007_200703

/-- The number of nonoverlapping unit squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^2 + 2 * n + 1

/-- The sequence of nonoverlapping unit squares follows the pattern -/
axiom sequence_pattern :
  f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25

/-- The number of nonoverlapping unit squares in figure 100 is 20201 -/
theorem squares_in_figure_100 : f 100 = 20201 := by sorry

end NUMINAMATH_CALUDE_squares_in_figure_100_l2007_200703


namespace NUMINAMATH_CALUDE_coffee_consumption_l2007_200763

theorem coffee_consumption (people : ℕ) (coffee_per_cup : ℚ) (coffee_cost : ℚ) (weekly_spend : ℚ) :
  people = 4 →
  coffee_per_cup = 1/2 →
  coffee_cost = 5/4 →
  weekly_spend = 35 →
  (weekly_spend / coffee_cost / coffee_per_cup / people / 7 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_l2007_200763


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2007_200739

theorem rectangle_area_increase (L W : ℝ) (h1 : L * W = 500) : 
  (1.2 * L) * (1.2 * W) = 720 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2007_200739


namespace NUMINAMATH_CALUDE_negation_absolute_value_inequality_l2007_200781

theorem negation_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_absolute_value_inequality_l2007_200781


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2007_200797

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) :
  x / y = 23 / 24 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2007_200797


namespace NUMINAMATH_CALUDE_small_gardens_and_pepper_seeds_l2007_200743

/-- Represents the number of small gardens for each vegetable type -/
structure SmallGardens where
  tomatoes : ℕ
  lettuce : ℕ
  peppers : ℕ

/-- Represents the seed requirements for each vegetable type -/
structure SeedRequirements where
  tomatoes : ℕ
  lettuce : ℕ
  peppers : ℕ

def total_seeds : ℕ := 42
def big_garden_seeds : ℕ := 36

def small_gardens : SmallGardens :=
  { tomatoes := 3
  , lettuce := 2
  , peppers := 0 }

def seed_requirements : SeedRequirements :=
  { tomatoes := 4
  , lettuce := 3
  , peppers := 2 }

def remaining_seeds : ℕ := total_seeds - big_garden_seeds

theorem small_gardens_and_pepper_seeds :
  (small_gardens.tomatoes + small_gardens.lettuce + small_gardens.peppers = 5) ∧
  (small_gardens.peppers * seed_requirements.peppers = 0) :=
by sorry

end NUMINAMATH_CALUDE_small_gardens_and_pepper_seeds_l2007_200743


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l2007_200780

theorem largest_integer_inequality : ∀ y : ℤ, y ≤ 7 ↔ (y : ℚ) / 4 + 3 / 7 < 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l2007_200780


namespace NUMINAMATH_CALUDE_jenna_earnings_l2007_200700

def calculate_earnings (distance : ℕ) : ℚ :=
  let first_100 := min distance 100
  let next_200 := min (distance - 100) 200
  let beyond_300 := max (distance - 300) 0
  0.4 * first_100 + 0.5 * next_200 + 0.6 * beyond_300

def round_trip_distance : ℕ := 800

def base_earnings : ℚ := 2 * calculate_earnings (round_trip_distance / 2)

def bonus : ℚ := 100 * (round_trip_distance / 500)

def weather_reduction : ℚ := 0.1

def rest_stop_reduction : ℚ := 0.05

def performance_incentive : ℚ := 0.05

def maintenance_cost : ℚ := 50 * (round_trip_distance / 500)

def fuel_cost_rate : ℚ := 0.15

theorem jenna_earnings :
  let reduced_bonus := bonus * (1 - weather_reduction)
  let earnings_with_bonus := base_earnings + reduced_bonus
  let earnings_with_incentive := earnings_with_bonus * (1 + performance_incentive)
  let earnings_after_rest_stop := earnings_with_incentive * (1 - rest_stop_reduction)
  let fuel_cost := earnings_after_rest_stop * fuel_cost_rate
  let net_earnings := earnings_after_rest_stop - maintenance_cost - fuel_cost
  net_earnings = 380 := by sorry

end NUMINAMATH_CALUDE_jenna_earnings_l2007_200700


namespace NUMINAMATH_CALUDE_stack_probability_l2007_200713

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the probability of a stack of crates being a certain height -/
def probabilityOfStackHeight (crate : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of a stack of 10 crates being 41 ft tall -/
theorem stack_probability :
  let crate : CrateDimensions := { length := 3, width := 4, height := 6 }
  probabilityOfStackHeight crate 10 41 = 190 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_stack_probability_l2007_200713


namespace NUMINAMATH_CALUDE_paint_remaining_l2007_200721

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint > 0 → 
  (initial_paint - (initial_paint / 2) - ((initial_paint - (initial_paint / 2)) / 2)) / initial_paint = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_paint_remaining_l2007_200721


namespace NUMINAMATH_CALUDE_x_intercepts_count_l2007_200708

theorem x_intercepts_count :
  let f : ℝ → ℝ := λ x => (x - 5) * (x^2 + 5*x + 6) * (x - 1)
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l2007_200708


namespace NUMINAMATH_CALUDE_solve_equation_l2007_200784

theorem solve_equation :
  ∃ x : ℚ, 5 * x + 9 * x = 350 - 10 * (x - 5) ∧ x = 50 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2007_200784


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2007_200732

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2007_200732


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l2007_200770

/-- The probability of picking two red balls from a bag containing red, blue, and green balls -/
theorem prob_two_red_balls (red blue green : ℕ) (h : red = 5 ∧ blue = 6 ∧ green = 4) :
  let total := red + blue + green
  (red : ℚ) / total * ((red - 1) : ℚ) / (total - 1) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l2007_200770


namespace NUMINAMATH_CALUDE_bull_work_hours_equality_l2007_200715

/-- Represents the work rate of bulls ploughing fields -/
structure BullWork where
  bulls : ℕ
  fields : ℕ
  days : ℕ
  hours_per_day : ℝ

/-- Calculates the total bull-hours for a given BullWork -/
def total_bull_hours (work : BullWork) : ℝ :=
  work.bulls * work.fields * work.days * work.hours_per_day

theorem bull_work_hours_equality (work1 work2 : BullWork) 
  (h1 : work1.bulls = 10)
  (h2 : work1.fields = 20)
  (h3 : work1.days = 3)
  (h4 : work2.bulls = 30)
  (h5 : work2.fields = 32)
  (h6 : work2.days = 2)
  (h7 : work2.hours_per_day = 8)
  (h8 : total_bull_hours work1 = total_bull_hours work2) :
  work1.hours_per_day = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_bull_work_hours_equality_l2007_200715


namespace NUMINAMATH_CALUDE_log_cube_difference_l2007_200760

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_cube_difference 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_one : a ≠ 1) 
  (h_diff : f a x₁ - f a x₂ = 2) : 
  f a (x₁^3) - f a (x₂^3) = 6 := by
sorry

end NUMINAMATH_CALUDE_log_cube_difference_l2007_200760


namespace NUMINAMATH_CALUDE_books_added_to_shelf_l2007_200769

theorem books_added_to_shelf (initial_action_figures initial_books added_books : ℕ) :
  initial_action_figures = 7 →
  initial_books = 2 →
  initial_action_figures = (initial_books + added_books) + 1 →
  added_books = 4 :=
by sorry

end NUMINAMATH_CALUDE_books_added_to_shelf_l2007_200769


namespace NUMINAMATH_CALUDE_lotto_winnings_theorem_l2007_200707

/-- The amount of money won by each boy in the "Russian Lotto" draw. -/
structure LottoWinnings where
  kolya : ℕ
  misha : ℕ
  vitya : ℕ

/-- The conditions of the "Russian Lotto" draw. -/
def lotto_conditions (w : LottoWinnings) : Prop :=
  w.misha = w.kolya + 943 ∧
  w.vitya = w.misha + 127 ∧
  w.misha + w.kolya = w.vitya + 479

/-- The theorem stating the correct winnings for each boy. -/
theorem lotto_winnings_theorem :
  ∃ (w : LottoWinnings), lotto_conditions w ∧ w.kolya = 606 ∧ w.misha = 1549 ∧ w.vitya = 1676 :=
by
  sorry

end NUMINAMATH_CALUDE_lotto_winnings_theorem_l2007_200707


namespace NUMINAMATH_CALUDE_probability_of_X_selection_l2007_200790

theorem probability_of_X_selection (p_Y p_both : ℝ) 
  (h1 : p_Y = 2/7)
  (h2 : p_both = 0.09523809523809523)
  (h3 : p_both = p_Y * p_X)
  : p_X = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_X_selection_l2007_200790


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l2007_200756

/-- The point of intersection for two lines defined by linear equations -/
def intersection_point : ℚ × ℚ := (24/25, 34/25)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = 7x - 4 -/
def line2 (x y : ℚ) : Prop := 2 * y = 7 * x - 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l2007_200756


namespace NUMINAMATH_CALUDE_store_sales_total_l2007_200733

/-- The total money made from selling DVD players and a washing machine -/
def total_money (dvd_price : ℕ) (dvd_quantity : ℕ) (washing_machine_price : ℕ) : ℕ :=
  dvd_price * dvd_quantity + washing_machine_price

/-- Theorem: The total money made from selling 8 DVD players at 240 yuan each
    and one washing machine at 898 yuan is equal to 240 * 8 + 898 yuan -/
theorem store_sales_total :
  total_money 240 8 898 = 240 * 8 + 898 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_total_l2007_200733


namespace NUMINAMATH_CALUDE_sally_quarters_l2007_200722

/-- Given that Sally had 760 quarters initially and spent 418 quarters,
    prove that she now has 342 quarters. -/
theorem sally_quarters : 
  ∀ (initial spent remaining : ℕ), 
  initial = 760 → 
  spent = 418 → 
  remaining = initial - spent → 
  remaining = 342 := by sorry

end NUMINAMATH_CALUDE_sally_quarters_l2007_200722


namespace NUMINAMATH_CALUDE_solve_equation_l2007_200759

theorem solve_equation (x : ℝ) : 2*x + 5 - 3*x + 7 = 8 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2007_200759


namespace NUMINAMATH_CALUDE_sum_of_roots_l2007_200747

theorem sum_of_roots (a b c d : ℝ) : 
  (∀ x : ℝ, x^2 - 2*c*x - 5*d = 0 ↔ x = a ∨ x = b) →
  (∀ x : ℝ, x^2 - 2*a*x - 5*b = 0 ↔ x = c ∨ x = d) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b + c + d = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2007_200747


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l2007_200735

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cylinder_height = 6 →
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cylinder_height = 216 - 54 * π :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l2007_200735


namespace NUMINAMATH_CALUDE_polynomial_factor_l2007_200762

variables {F : Type*} [Field F]
variables (P Q R S : F → F)

theorem polynomial_factor 
  (h : ∀ x, P (x^3) + x * Q (x^3) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S x) : 
  P 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2007_200762


namespace NUMINAMATH_CALUDE_tan_difference_absolute_value_l2007_200768

theorem tan_difference_absolute_value (α β : Real) : 
  (∃ x y : Real, x^2 - 2*x - 4 = 0 ∧ y^2 - 2*y - 4 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  |Real.tan (α - β)| = 2 * Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_absolute_value_l2007_200768


namespace NUMINAMATH_CALUDE_fraction_equality_l2007_200712

theorem fraction_equality (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2007_200712


namespace NUMINAMATH_CALUDE_max_non_managers_l2007_200734

/-- The maximum number of non-managers in a department with 9 managers,
    given that the ratio of managers to non-managers must be greater than 7:32 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 41 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l2007_200734


namespace NUMINAMATH_CALUDE_widget_count_l2007_200738

theorem widget_count : ∃ (a b c d e f : ℕ),
  3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255 ∧
  3^a * 11^b * 5^c * 7^d * 13^e * 17^f = 351125648000 ∧
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_widget_count_l2007_200738


namespace NUMINAMATH_CALUDE_max_x_plus_z_l2007_200755

theorem max_x_plus_z (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x*t + y*z = 6) :
  x + z ≤ Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_x_plus_z_l2007_200755


namespace NUMINAMATH_CALUDE_no_solution_exists_l2007_200746

theorem no_solution_exists : ¬∃ (x y z : ℝ), 
  (x^2 ≠ y^2) ∧ (y^2 ≠ z^2) ∧ (z^2 ≠ x^2) ∧
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2007_200746


namespace NUMINAMATH_CALUDE_exists_polyhedron_with_properties_l2007_200711

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  volumeBelowWater : ℝ
  surfaceAreaAboveWater : ℝ

/-- Theorem stating the existence of a convex polyhedron with the given properties -/
theorem exists_polyhedron_with_properties :
  ∃ (p : ConvexPolyhedron),
    p.volumeBelowWater = 0.9 * p.volume ∧
    p.surfaceAreaAboveWater > 0.5 * p.surfaceArea :=
sorry

end NUMINAMATH_CALUDE_exists_polyhedron_with_properties_l2007_200711


namespace NUMINAMATH_CALUDE_tan_two_alpha_l2007_200764

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_two_alpha (α : ℝ) (h : (deriv f) α = 3 * f α) : Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l2007_200764


namespace NUMINAMATH_CALUDE_complement_of_union_in_U_l2007_200785

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union_in_U : (U \ (A ∪ B)) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_in_U_l2007_200785


namespace NUMINAMATH_CALUDE_admission_probability_l2007_200767

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of universities -/
def num_universities : ℕ := 3

/-- The total number of possible admission arrangements -/
def total_arrangements : ℕ := num_universities ^ num_students

/-- The number of arrangements where each university admits at least one student -/
def favorable_arrangements : ℕ := 36

/-- The probability that each university admits at least one student -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem admission_probability : probability = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_admission_probability_l2007_200767


namespace NUMINAMATH_CALUDE_line_through_three_points_l2007_200775

/-- Given a line passing through points (0, 4), (5, k), and (15, 1), prove that k = 3 -/
theorem line_through_three_points (k : ℝ) : 
  (∃ (m b : ℝ), 4 = b ∧ k = 5*m + b ∧ 1 = 15*m + b) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_three_points_l2007_200775


namespace NUMINAMATH_CALUDE_difference_of_squares_l2007_200789

theorem difference_of_squares (a b : ℝ) : (-a + b) * (-a - b) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2007_200789


namespace NUMINAMATH_CALUDE_sequence_is_geometric_l2007_200774

theorem sequence_is_geometric (a : ℝ) (h : a ≠ 0) :
  (∃ S : ℕ → ℝ, ∀ n : ℕ, S n = a^n - 1) →
  (∃ r : ℝ, ∀ n : ℕ, ∃ u : ℕ → ℝ, u (n+1) = r * u n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_is_geometric_l2007_200774


namespace NUMINAMATH_CALUDE_teacher_wang_pen_purchase_l2007_200729

/-- Given that Teacher Wang has enough money to buy 72 pens at 5 yuan each,
    prove that he can buy 60 pens when the price increases to 6 yuan each. -/
theorem teacher_wang_pen_purchase (initial_pens : ℕ) (initial_price : ℕ) (new_price : ℕ) :
  initial_pens = 72 → initial_price = 5 → new_price = 6 →
  (initial_pens * initial_price) / new_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_teacher_wang_pen_purchase_l2007_200729


namespace NUMINAMATH_CALUDE_journey_length_l2007_200799

theorem journey_length : 
  ∀ (L : ℝ) (T : ℝ),
  L = 60 * T →
  L = 50 * (T + 3/4) →
  L = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_length_l2007_200799
