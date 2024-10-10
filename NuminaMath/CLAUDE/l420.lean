import Mathlib

namespace inequality_proof_l420_42039

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 := by
  sorry

end inequality_proof_l420_42039


namespace no_rain_probability_l420_42071

theorem no_rain_probability (rain_prob : ℚ) (days : ℕ) : 
  rain_prob = 2/3 → days = 5 → (1 - rain_prob)^days = 1/243 := by
  sorry

end no_rain_probability_l420_42071


namespace soldier_difference_l420_42068

/-- Calculates the difference in the number of soldiers between two sides in a war scenario --/
theorem soldier_difference (
  daily_food : ℕ)  -- Daily food requirement per soldier on the first side
  (food_difference : ℕ)  -- Difference in food given to soldiers on the second side
  (first_side_soldiers : ℕ)  -- Number of soldiers on the first side
  (total_food : ℕ)  -- Total amount of food for both sides
  (h1 : daily_food = 10)  -- Each soldier needs 10 pounds of food per day
  (h2 : food_difference = 2)  -- Soldiers on the second side get 2 pounds less food
  (h3 : first_side_soldiers = 4000)  -- The first side has 4000 soldiers
  (h4 : total_food = 68000)  -- The total amount of food for both sides is 68000 pounds
  : (first_side_soldiers - (total_food - first_side_soldiers * daily_food) / (daily_food - food_difference) = 500) :=
by sorry

end soldier_difference_l420_42068


namespace fraction_count_l420_42051

/-- A fraction is an expression of the form A/B where A and B are polynomials and B contains letters -/
def IsFraction (expr : String) : Prop := sorry

/-- The set of given expressions -/
def ExpressionSet : Set String := {"1/m", "b/3", "(x-1)/π", "2/(x+y)", "a+1/a"}

/-- Counts the number of fractions in a set of expressions -/
def CountFractions (s : Set String) : ℕ := sorry

theorem fraction_count : CountFractions ExpressionSet = 3 := by sorry

end fraction_count_l420_42051


namespace notebook_problem_l420_42017

theorem notebook_problem (x : ℕ) (h : x^2 + 20 = (x + 1)^2 - 9) : x^2 + 20 = 216 := by
  sorry

end notebook_problem_l420_42017


namespace fuel_tank_capacities_solve_problem_l420_42063

/-- Represents the fuel tank capacities and prices for two cars -/
structure CarFuelData where
  small_capacity : ℝ
  large_capacity : ℝ
  small_fill_cost : ℝ
  large_fill_cost : ℝ
  price_difference : ℝ

/-- The theorem to be proved -/
theorem fuel_tank_capacities (data : CarFuelData) : 
  data.small_capacity = 30 ∧ data.large_capacity = 40 :=
by
  have total_capacity : data.small_capacity + data.large_capacity = 70 := by sorry
  have small_fill_equation : data.small_capacity * (data.large_fill_cost / data.large_capacity - data.price_difference) = data.small_fill_cost := by sorry
  have large_fill_equation : data.large_capacity * (data.large_fill_cost / data.large_capacity) = data.large_fill_cost := by sorry
  have price_relation : data.large_fill_cost / data.large_capacity = data.small_fill_cost / data.small_capacity + data.price_difference := by sorry
  
  sorry -- The proof would go here

/-- The specific instance of CarFuelData for our problem -/
def problem_data : CarFuelData := {
  small_capacity := 30,  -- to be proved
  large_capacity := 40,  -- to be proved
  small_fill_cost := 45,
  large_fill_cost := 68,
  price_difference := 0.29
}

/-- The main theorem applied to our specific problem -/
theorem solve_problem : 
  problem_data.small_capacity = 30 ∧ problem_data.large_capacity = 40 :=
fuel_tank_capacities problem_data

end fuel_tank_capacities_solve_problem_l420_42063


namespace intersection_empty_union_equals_B_l420_42013

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem for the first part
theorem intersection_empty (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≥ 2 ∨ a ≤ -1/2 :=
sorry

-- Theorem for the second part
theorem union_equals_B (a : ℝ) :
  A a ∪ B = B ↔ a ≤ -2 :=
sorry

end intersection_empty_union_equals_B_l420_42013


namespace corrected_mean_l420_42054

theorem corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  initial_mean = 36 →
  incorrect_value = 23 →
  correct_value = 48 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = 36.5 := by
  sorry

end corrected_mean_l420_42054


namespace smallest_prime_factor_in_C_l420_42069

def C : Set Nat := {62, 64, 65, 69, 71}

theorem smallest_prime_factor_in_C :
  ∃ x ∈ C, ∀ y ∈ C, ∀ p q : Nat,
    Prime p → Prime q → p ∣ x → q ∣ y → p ≤ q :=
by sorry

end smallest_prime_factor_in_C_l420_42069


namespace walts_investment_rate_l420_42026

/-- Given Walt's investment scenario, prove that the unknown interest rate is 9% -/
theorem walts_investment_rate : 
  ∀ (total_extra : ℝ) (total_interest : ℝ) (known_amount : ℝ) (known_rate : ℝ),
  total_extra = 9000 →
  total_interest = 770 →
  known_amount = 4000 →
  known_rate = 0.08 →
  ∃ (unknown_rate : ℝ),
    unknown_rate = 0.09 ∧
    total_interest = known_amount * known_rate + (total_extra - known_amount) * unknown_rate :=
by
  sorry

#check walts_investment_rate

end walts_investment_rate_l420_42026


namespace pizza_toppings_combinations_l420_42053

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem pizza_toppings_combinations :
  choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l420_42053


namespace point_on_right_branch_l420_42076

/-- 
Given a point P(a, b) on the hyperbola x² - 4y² = m (m ≠ 0),
if a - 2b > 0 and a + 2b > 0, then a > 0.
-/
theorem point_on_right_branch 
  (m : ℝ) (hm : m ≠ 0)
  (a b : ℝ) 
  (h_hyperbola : a^2 - 4*b^2 = m)
  (h_diff : a - 2*b > 0)
  (h_sum : a + 2*b > 0) : 
  a > 0 := by sorry

end point_on_right_branch_l420_42076


namespace estimate_value_l420_42092

theorem estimate_value : 5 < (3 * Real.sqrt 15 - Real.sqrt 3) * Real.sqrt (1/3) ∧
                         (3 * Real.sqrt 15 - Real.sqrt 3) * Real.sqrt (1/3) < 6 := by
  sorry

end estimate_value_l420_42092


namespace invalid_domain_l420_42074

def f (x : ℝ) : ℝ := x^2

def N : Set ℝ := {1, 2}

theorem invalid_domain : ¬(∀ x ∈ ({1, Real.sqrt 2, 2} : Set ℝ), f x ∈ N) := by
  sorry

end invalid_domain_l420_42074


namespace valid_seating_arrangements_l420_42052

-- Define the type for people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person
| Fiona : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 6 → Person

-- Define the conditions for a valid seating arrangement
def IsValidArrangement (arrangement : SeatingArrangement) : Prop :=
  -- Alice is not next to Bob or Carla
  (∀ i : Fin 5, arrangement i = Person.Alice → 
    arrangement (i + 1) ≠ Person.Bob ∧ arrangement (i + 1) ≠ Person.Carla) ∧
  (∀ i : Fin 5, arrangement (i + 1) = Person.Alice → 
    arrangement i ≠ Person.Bob ∧ arrangement i ≠ Person.Carla) ∧
  -- Derek is not next to Eric
  (∀ i : Fin 5, arrangement i = Person.Derek → arrangement (i + 1) ≠ Person.Eric) ∧
  (∀ i : Fin 5, arrangement (i + 1) = Person.Derek → arrangement i ≠ Person.Eric) ∧
  -- Fiona is not at either end
  (arrangement 0 ≠ Person.Fiona) ∧ (arrangement 5 ≠ Person.Fiona) ∧
  -- All people are seated and each seat has exactly one person
  (∀ p : Person, ∃! i : Fin 6, arrangement i = p)

-- The theorem to be proved
theorem valid_seating_arrangements :
  (∃ (arrangements : Finset SeatingArrangement), 
    (∀ arr ∈ arrangements, IsValidArrangement arr) ∧ 
    arrangements.card = 16) :=
sorry

end valid_seating_arrangements_l420_42052


namespace line_through_point_parallel_to_vector_l420_42023

/-- A line in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point lies on a line given by its parametric equation -/
def lies_on_line (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

theorem line_through_point_parallel_to_vector 
  (P : ℝ × ℝ) (d : ℝ × ℝ) :
  let l : Line2D := ⟨P, d⟩
  (∀ x y : ℝ, (x - P.1) / d.1 = (y - P.2) / d.2 ↔ lies_on_line (x, y) l) :=
by sorry

end line_through_point_parallel_to_vector_l420_42023


namespace angles_on_y_axis_l420_42093

def terminal_side_on_y_axis (θ : Real) : Prop :=
  ∃ n : Int, θ = n * Real.pi + Real.pi / 2

theorem angles_on_y_axis :
  {θ : Real | terminal_side_on_y_axis θ} = {θ : Real | ∃ n : Int, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end angles_on_y_axis_l420_42093


namespace pentagon_triangle_side_ratio_l420_42022

theorem pentagon_triangle_side_ratio :
  ∀ (p t s : ℝ),
  p > 0 ∧ t > 0 ∧ s > 0 →
  5 * p = 3 * t →
  5 * p = 4 * s →
  p / t = 3 / 5 := by
sorry

end pentagon_triangle_side_ratio_l420_42022


namespace three_numbers_sum_l420_42080

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  (a + b + c) / 3 = a + 8 ∧  -- mean is 8 more than least
  (a + b + c) / 3 = c - 20 ∧  -- mean is 20 less than greatest
  b = 10  -- median is 10
  → a + b + c = 66 := by sorry

end three_numbers_sum_l420_42080


namespace division_problem_l420_42033

theorem division_problem (n : ℕ) : 
  n % 7 = 5 ∧ n / 7 = 12 → n / 8 = 11 := by sorry

end division_problem_l420_42033


namespace wall_width_calculation_l420_42015

/-- The width of a wall given a string length and a relation to that length -/
def wall_width (string_length_m : ℕ) (string_length_cm : ℕ) : ℕ :=
  let string_length_total_cm := string_length_m * 100 + string_length_cm
  5 * string_length_total_cm + 80

theorem wall_width_calculation :
  wall_width 1 70 = 930 := by sorry

end wall_width_calculation_l420_42015


namespace profit_maximized_optimal_selling_price_l420_42062

/-- Profit function given the increase in selling price -/
def profit (x : ℝ) : ℝ := (2 + x) * (200 - 20 * x)

/-- The optimal price increase that maximizes profit -/
def optimal_price_increase : ℝ := 4

/-- The maximum profit achievable -/
def max_profit : ℝ := 720

/-- Theorem stating that the profit function reaches its maximum at the optimal price increase -/
theorem profit_maximized :
  (∀ x : ℝ, profit x ≤ profit optimal_price_increase) ∧
  profit optimal_price_increase = max_profit :=
sorry

/-- The initial selling price -/
def initial_price : ℝ := 10

/-- Theorem stating the optimal selling price -/
theorem optimal_selling_price :
  initial_price + optimal_price_increase = 14 :=
sorry

end profit_maximized_optimal_selling_price_l420_42062


namespace smaller_number_value_l420_42084

theorem smaller_number_value (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : 
  min x y = 28.5 := by
sorry

end smaller_number_value_l420_42084


namespace sundae_price_l420_42078

/-- Proves that the price of each sundae is $1.20 given the specified conditions -/
theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) : 
  ice_cream_bars = 125 →
  sundaes = 125 →
  total_price = 225 →
  ice_cream_price = 0.60 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 1.20 := by
sorry

end sundae_price_l420_42078


namespace correct_initial_driving_time_l420_42065

/-- Represents the driving scenario with given conditions -/
structure DrivingScenario where
  totalDistance : ℝ
  initialSpeed : ℝ
  finalSpeed : ℝ
  lateTime : ℝ
  earlyTime : ℝ

/-- Calculates the time driven at the initial speed -/
def initialDrivingTime (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating the correct initial driving time for the given scenario -/
theorem correct_initial_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.totalDistance = 45)
  (h2 : scenario.initialSpeed = 15)
  (h3 : scenario.finalSpeed = 60)
  (h4 : scenario.lateTime = 1)
  (h5 : scenario.earlyTime = 0.5) :
  initialDrivingTime scenario = 7/3 := by
  sorry

end correct_initial_driving_time_l420_42065


namespace rectangular_solid_surface_area_l420_42035

/-- A rectangular solid with prime edge lengths and volume 455 has surface area 382 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 455 →
  2 * (a * b + b * c + c * a) = 382 := by
sorry

end rectangular_solid_surface_area_l420_42035


namespace mary_james_seating_probability_l420_42090

/-- The number of chairs in the row -/
def totalChairs : ℕ := 10

/-- The set of broken chair numbers -/
def brokenChairs : Finset ℕ := {4, 7}

/-- The set of available chairs -/
def availableChairs : Finset ℕ := Finset.range totalChairs \ brokenChairs

/-- The probability that Mary and James do not sit next to each other -/
def probabilityNotAdjacent : ℚ := 3/4

theorem mary_james_seating_probability :
  let totalWays := Nat.choose availableChairs.card 2
  let adjacentWays := (availableChairs.filter (fun n => n + 1 ∈ availableChairs)).card
  (totalWays - adjacentWays : ℚ) / totalWays = probabilityNotAdjacent :=
sorry

end mary_james_seating_probability_l420_42090


namespace inequality_solution_set_l420_42021

theorem inequality_solution_set (x : ℝ) : 
  |x - 4| - |x + 1| < 3 ↔ x ∈ Set.Ioo (-1/2 : ℝ) 4 ∪ Set.Ici 4 :=
sorry

end inequality_solution_set_l420_42021


namespace sam_distance_theorem_l420_42008

/-- Calculates the distance traveled given an average speed and time -/
def distanceTraveled (avgSpeed : ℝ) (time : ℝ) : ℝ := avgSpeed * time

theorem sam_distance_theorem (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
    (h1 : marguerite_distance = 100)
    (h2 : marguerite_time = 2.4)
    (h3 : sam_time = 3) :
  distanceTraveled (marguerite_distance / marguerite_time) sam_time = 125 := by
  sorry

#check sam_distance_theorem

end sam_distance_theorem_l420_42008


namespace class_average_calculation_l420_42024

theorem class_average_calculation (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 20 → 
  excluded_students = 5 → 
  excluded_avg = 50 → 
  remaining_avg = 90 → 
  (total_students * (total_students * remaining_avg - excluded_students * remaining_avg + 
   excluded_students * excluded_avg)) / (total_students * total_students) = 80 := by
  sorry

end class_average_calculation_l420_42024


namespace hank_aaron_home_runs_l420_42037

/-- The number of home runs hit by Dave Winfield -/
def dave_winfield_hr : ℕ := 465

/-- The number of home runs hit by Hank Aaron -/
def hank_aaron_hr : ℕ := 2 * dave_winfield_hr - 175

/-- Theorem stating that Hank Aaron hit 755 home runs -/
theorem hank_aaron_home_runs : hank_aaron_hr = 755 := by sorry

end hank_aaron_home_runs_l420_42037


namespace simplify_sqrt_expression_l420_42047

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l420_42047


namespace divide_polynomials_expand_and_simplify_l420_42045

-- Part 1
theorem divide_polynomials (x : ℝ) (h : x ≠ 0) : 
  6 * x^3 / (-3 * x^2) = -2 * x := by sorry

-- Part 2
theorem expand_and_simplify (x : ℝ) : 
  (2*x + 3) * (2*x - 3) - 4 * (x - 2)^2 = 16*x - 25 := by sorry

end divide_polynomials_expand_and_simplify_l420_42045


namespace journey_time_calculation_l420_42038

/-- Calculates the time spent on the road given start time, end time, and total stop time. -/
def timeOnRoad (startTime endTime stopTime : ℕ) : ℕ :=
  (endTime - startTime) - stopTime

/-- Proves that for a journey from 7:00 AM to 8:00 PM with 60 minutes of stops, the time on the road is 12 hours. -/
theorem journey_time_calculation :
  let startTime : ℕ := 7  -- 7:00 AM
  let endTime : ℕ := 20   -- 8:00 PM (20:00 in 24-hour format)
  let stopTime : ℕ := 1   -- 60 minutes = 1 hour
  timeOnRoad startTime endTime stopTime = 12 := by
  sorry

end journey_time_calculation_l420_42038


namespace folded_paper_thickness_l420_42011

/-- The thickness of a folded paper stack -/
def folded_thickness (initial_thickness : ℝ) : ℝ := 2 * initial_thickness

/-- Theorem: Folding a 0.2 cm thick paper stack once results in a 0.4 cm thick stack -/
theorem folded_paper_thickness :
  folded_thickness 0.2 = 0.4 := by
  sorry

end folded_paper_thickness_l420_42011


namespace negation_equivalence_l420_42040

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 3*x + 3 < 0) ↔ (∀ x : ℝ, x^2 - 3*x + 3 ≥ 0) := by
  sorry

end negation_equivalence_l420_42040


namespace total_notes_count_l420_42010

/-- Proves that given a total amount of 480 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes, the total number of notes is 90. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 480 →
  note_count * 1 + note_count * 5 + note_count * 10 = total_amount →
  3 * note_count = 90 :=
by
  sorry

#check total_notes_count

end total_notes_count_l420_42010


namespace function_form_l420_42095

theorem function_form (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = Real.sin x ^ 2 - 1/4 := by
  sorry

end function_form_l420_42095


namespace cylinder_surface_area_l420_42082

/-- The surface area of a cylinder given its unfolded lateral surface dimensions -/
theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) 
  (h_eq : h = 6 * Real.pi) (w_eq : w = 4 * Real.pi) :
  ∃ (r : ℝ), (r = 3 ∨ r = 2) ∧ 
    (2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
     2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi) :=
by sorry

end cylinder_surface_area_l420_42082


namespace minibus_capacity_insufficient_l420_42073

theorem minibus_capacity_insufficient (students : ℕ) (bus_capacity : ℕ) (num_buses : ℕ) : 
  students = 300 → 
  bus_capacity = 23 → 
  num_buses = 13 → 
  num_buses * bus_capacity < students := by
sorry

end minibus_capacity_insufficient_l420_42073


namespace simplify_complex_root_expression_l420_42019

theorem simplify_complex_root_expression (a : ℝ) :
  (((a^16)^(1/8))^(1/4) + ((a^16)^(1/4))^(1/8))^2 = 4*a := by sorry

end simplify_complex_root_expression_l420_42019


namespace a2_value_l420_42029

theorem a2_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₂ = 42 := by
sorry

end a2_value_l420_42029


namespace amanda_earnings_l420_42058

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Hours worked on Monday -/
def monday_hours : ℝ := 5 * 1.5

/-- Hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Hours worked on Thursday -/
def thursday_hours : ℝ := 2 * 2

/-- Hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total hours worked in the week -/
def total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours

/-- Amanda's earnings for the week -/
def weekly_earnings : ℝ := total_hours * hourly_rate

theorem amanda_earnings : weekly_earnings = 410 := by
  sorry

end amanda_earnings_l420_42058


namespace faster_train_speed_l420_42087

/-- Proves that the speed of the faster train is 50 km/hr given the problem conditions -/
theorem faster_train_speed 
  (speed_diff : ℝ) 
  (faster_train_length : ℝ) 
  (passing_time : ℝ) :
  speed_diff = 32 →
  faster_train_length = 75 →
  passing_time = 15 →
  ∃ (slower_speed faster_speed : ℝ),
    faster_speed - slower_speed = speed_diff ∧
    faster_train_length / passing_time * 3.6 = speed_diff ∧
    faster_speed = 50 := by
  sorry

#check faster_train_speed

end faster_train_speed_l420_42087


namespace salon_non_clients_l420_42000

theorem salon_non_clients (manicure_cost : ℝ) (total_earnings : ℝ) (total_fingers : ℕ) (fingers_per_person : ℕ) :
  manicure_cost = 20 →
  total_earnings = 200 →
  total_fingers = 210 →
  fingers_per_person = 10 →
  (total_fingers / fingers_per_person : ℝ) - (total_earnings / manicure_cost) = 11 :=
by sorry

end salon_non_clients_l420_42000


namespace production_equation_holds_l420_42088

/-- Represents the production rate of a factory -/
structure FactoryProduction where
  current_rate : ℝ
  original_rate : ℝ
  h_rate_increase : current_rate = original_rate + 50

/-- The equation representing the production scenario -/
def production_equation (fp : FactoryProduction) : Prop :=
  (450 / fp.original_rate) - (400 / fp.current_rate) = 1

/-- Theorem stating that the production equation holds for the given scenario -/
theorem production_equation_holds (fp : FactoryProduction) :
  production_equation fp := by
  sorry

#check production_equation_holds

end production_equation_holds_l420_42088


namespace range_of_a_l420_42056

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → 
  a ∈ Set.Iic 1 ∪ Set.Ici 3 :=
by sorry

end range_of_a_l420_42056


namespace rectangle_area_l420_42001

theorem rectangle_area (a b : ℕ) : 
  a ≠ b →                  -- rectangle is not a square
  a % 2 = 0 →              -- one side is even
  a * b = 3 * (2 * a + 2 * b) →  -- area is three times perimeter
  a * b = 162              -- area is 162
:= by sorry

end rectangle_area_l420_42001


namespace pipe_speed_ratio_l420_42034

-- Define the rates of pipes A, B, and C
def rate_A : ℚ := 1 / 21
def rate_B : ℚ := 2 / 21
def rate_C : ℚ := 4 / 21

-- State the theorem
theorem pipe_speed_ratio :
  -- Conditions
  (rate_A + rate_B + rate_C = 1 / 3) →  -- All pipes fill the tank in 3 hours
  (rate_C = 2 * rate_B) →               -- Pipe C is twice as fast as B
  (rate_A = 1 / 21) →                   -- Pipe A alone takes 21 hours
  -- Conclusion
  (rate_B / rate_A = 2) :=
by sorry

end pipe_speed_ratio_l420_42034


namespace shaded_area_half_circle_l420_42072

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the intersecting point
def intersectionPoint : ℝ × ℝ := sorry

-- Define the four lines
def lines : List (ℝ × ℝ → ℝ × ℝ → Prop) := sorry

-- Define the condition that the point is inside the circle
def pointInsideCircle (c : Circle) (p : ℝ × ℝ) : Prop := sorry

-- Define the condition that the lines form eight 45° angles
def formsEightFortyFiveAngles (p : ℝ × ℝ) (ls : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Prop := sorry

-- Define the shaded sectors
def shadedSectors (c : Circle) (p : ℝ × ℝ) (ls : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Set (ℝ × ℝ) := sorry

-- Define the area of a set in ℝ²
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- The theorem to be proved
theorem shaded_area_half_circle (c : Circle) :
  pointInsideCircle c intersectionPoint →
  formsEightFortyFiveAngles intersectionPoint lines →
  area (shadedSectors c intersectionPoint lines) = π * c.radius^2 / 2 := by
  sorry

end shaded_area_half_circle_l420_42072


namespace cookie_problem_l420_42018

theorem cookie_problem : ∃! C : ℕ, 0 < C ∧ C < 80 ∧ C % 6 = 5 ∧ C % 9 = 7 ∧ C = 29 := by
  sorry

end cookie_problem_l420_42018


namespace trailing_zeros_30_factorial_l420_42044

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial : trailingZeros 30 = 7 := by
  sorry

end trailing_zeros_30_factorial_l420_42044


namespace sale_price_calculation_l420_42091

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  let sellingPrice := costPrice * (1 + profitRate)
  sellingPrice * (1 + taxRate)

/-- Theorem stating that the sale price including tax is approximately 677.60 -/
theorem sale_price_calculation :
  let costPrice : ℝ := 545.13
  let profitRate : ℝ := 0.13
  let taxRate : ℝ := 0.10
  abs (salePriceWithTax costPrice profitRate taxRate - 677.60) < 0.01 := by
  sorry

end sale_price_calculation_l420_42091


namespace conference_handshakes_l420_42066

/-- The number of handshakes in a conference where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 35 people where each person shakes hands with every other person exactly once, the total number of handshakes is 595. -/
theorem conference_handshakes :
  handshakes 35 = 595 := by
  sorry

end conference_handshakes_l420_42066


namespace debbys_store_inventory_l420_42007

/-- Represents a DVD rental store inventory --/
structure DVDStore where
  initial_count : ℕ
  rental_rate : ℚ
  sold_count : ℕ

/-- Calculates the remaining DVD count after sales --/
def remaining_dvds (store : DVDStore) : ℕ :=
  store.initial_count - store.sold_count

/-- Theorem stating the remaining DVD count for Debby's store --/
theorem debbys_store_inventory :
  let store : DVDStore := {
    initial_count := 150,
    rental_rate := 35 / 100,
    sold_count := 20
  }
  remaining_dvds store = 130 := by
  sorry

end debbys_store_inventory_l420_42007


namespace simplify_and_evaluate_expression_l420_42014

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 3 + 1
  let y : ℝ := Real.sqrt 3
  ((3 * x + y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2)) / (2 / (x^2 * y - x * y^2)) = (3 + Real.sqrt 3) / 2 :=
by sorry

end simplify_and_evaluate_expression_l420_42014


namespace degree_to_radian_conversion_l420_42025

theorem degree_to_radian_conversion (deg : ℝ) (rad : ℝ) : 
  (180 : ℝ) = π → 240 = (4 / 3 : ℝ) * π := by
  sorry

end degree_to_radian_conversion_l420_42025


namespace factorization_equality_l420_42083

theorem factorization_equality (c : ℤ) : 
  (∀ x : ℤ, x^2 - x + c = (x + 2) * (x - 3)) → c = -6 := by
sorry

end factorization_equality_l420_42083


namespace function_composition_problem_l420_42009

theorem function_composition_problem (a b : ℝ) : 
  (∀ x, (3 * ((a * x) + b) - 4) = 4 * x + 5) → 
  a + b = 13 / 3 := by
  sorry

end function_composition_problem_l420_42009


namespace cubic_inequality_l420_42081

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end cubic_inequality_l420_42081


namespace cos_squared_30_minus_2_minus_pi_to_0_l420_42050

theorem cos_squared_30_minus_2_minus_pi_to_0 :
  Real.cos (30 * π / 180) ^ 2 - (2 - π) ^ 0 = -(1/4) := by sorry

end cos_squared_30_minus_2_minus_pi_to_0_l420_42050


namespace smallest_n_cube_plus_2square_eq_odd_square_l420_42049

theorem smallest_n_cube_plus_2square_eq_odd_square : 
  (∀ n : ℕ, 0 < n → n < 7 → ¬∃ k : ℕ, k % 2 = 1 ∧ n^3 + 2*n^2 = k^2) ∧
  (∃ k : ℕ, k % 2 = 1 ∧ 7^3 + 2*7^2 = k^2) :=
by sorry

end smallest_n_cube_plus_2square_eq_odd_square_l420_42049


namespace triangle_perimeter_l420_42057

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) : 
  c = 2 → b = 2 * a → C = π / 3 → a + b + c = 2 + 2 * Real.sqrt 3 := by
  sorry

end triangle_perimeter_l420_42057


namespace constant_ratio_locus_l420_42027

/-- The locus of points with a constant ratio of distances -/
theorem constant_ratio_locus (x y : ℝ) :
  (((x - 4)^2 + y^2) / (x - 3)^2 = 4) →
  (3 * x^2 - y^2 - 16 * x + 20 = 0) :=
by sorry

end constant_ratio_locus_l420_42027


namespace min_voters_for_tall_win_l420_42070

/-- Structure representing the giraffe beauty contest voting system -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Definition of the specific contest configuration -/
def contest : GiraffeContest :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

/-- Theorem stating the minimum number of voters needed for Tall to win -/
theorem min_voters_for_tall_win (c : GiraffeContest) 
  (h1 : c.total_voters = c.num_districts * c.precincts_per_district * c.voters_per_precinct)
  (h2 : c = contest) : 
  ∃ (min_voters : Nat), 
    min_voters = 30 ∧ 
    min_voters ≤ c.total_voters ∧
    min_voters = (c.num_districts / 2 + 1) * (c.precincts_per_district / 2 + 1) * (c.voters_per_precinct / 2 + 1) :=
by sorry


end min_voters_for_tall_win_l420_42070


namespace bd_length_is_15_l420_42028

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a kite
def is_kite (q : Quadrilateral) : Prop :=
  let AB := dist q.A q.B
  let BC := dist q.B q.C
  let CD := dist q.C q.D
  let DA := dist q.D q.A
  AB = CD ∧ BC = DA

-- Define the specific quadrilateral from the problem
def problem_quadrilateral : Quadrilateral :=
  { A := (0, 0),  -- Arbitrary placement
    B := (7, 0),  -- AB = 7
    C := (7, 19), -- BC = 19
    D := (0, 11)  -- DA = 11
  }

-- Theorem statement
theorem bd_length_is_15 (q : Quadrilateral) :
  is_kite q →
  dist q.A q.B = 7 →
  dist q.B q.C = 19 →
  dist q.C q.D = 7 →
  dist q.D q.A = 11 →
  dist q.B q.D = 15 :=
by sorry

#check bd_length_is_15

end bd_length_is_15_l420_42028


namespace john_january_savings_l420_42032

theorem john_january_savings :
  let base_income : ℝ := 2000
  let bonus_rate : ℝ := 0.15
  let transport_rate : ℝ := 0.05
  let rent : ℝ := 500
  let utilities : ℝ := 100
  let food : ℝ := 300
  let misc_rate : ℝ := 0.10

  let total_income : ℝ := base_income * (1 + bonus_rate)
  let transport_expense : ℝ := total_income * transport_rate
  let misc_expense : ℝ := total_income * misc_rate
  let total_expenses : ℝ := transport_expense + rent + utilities + food + misc_expense
  let savings : ℝ := total_income - total_expenses

  savings = 1055 := by sorry

end john_january_savings_l420_42032


namespace babysitting_time_calculation_l420_42003

/-- Calculates the time spent babysitting given the hourly rate and total earnings -/
def time_spent (hourly_rate : ℚ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / hourly_rate) * 60

/-- Proves that given an hourly rate of $12 and total earnings of $10, the time spent babysitting is 50 minutes -/
theorem babysitting_time_calculation (hourly_rate : ℚ) (total_earnings : ℚ) 
  (h1 : hourly_rate = 12)
  (h2 : total_earnings = 10) :
  time_spent hourly_rate total_earnings = 50 := by
  sorry

end babysitting_time_calculation_l420_42003


namespace inequality_proof_l420_42016

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := by
  sorry

end inequality_proof_l420_42016


namespace negation_of_all_greater_than_sin_l420_42005

theorem negation_of_all_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x₀ : ℝ, x₀ ≤ Real.sin x₀) := by
  sorry

end negation_of_all_greater_than_sin_l420_42005


namespace division_problem_l420_42059

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 15 ∧ divisor = 3 ∧ remainder = 3 →
  dividend = divisor * quotient + remainder →
  quotient = 4 := by
sorry

end division_problem_l420_42059


namespace hyperbola_equation_from_properties_l420_42085

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  focal_length : ℝ
  directrix : ℝ

/-- The equation of a hyperbola given its properties -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => 2 * x^2 - 2 * y^2 = 1

/-- Theorem: Given a hyperbola with center at the origin, focal length 2, 
    and one directrix at x = -1/2, its equation is 2x^2 - 2y^2 = 1 -/
theorem hyperbola_equation_from_properties 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focal_length : h.focal_length = 2)
  (h_directrix : h.directrix = -1/2) :
  ∀ x y, hyperbola_equation h x y ↔ 2 * x^2 - 2 * y^2 = 1 := by
  sorry

end hyperbola_equation_from_properties_l420_42085


namespace no_equal_consecutive_digit_sums_l420_42079

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

def S (n : ℕ) : ℕ :=
  sum_of_digits (2^n)

theorem no_equal_consecutive_digit_sums :
  ∀ n : ℕ, n > 0 → S (n + 1) ≠ S n :=
sorry

end no_equal_consecutive_digit_sums_l420_42079


namespace a1_range_for_three_greater_terms_l420_42099

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a1_range_for_three_greater_terms
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a (1/2))
  (b : ℕ → ℝ)
  (h_b : ∀ n, b n = n / 2)
  (h_three : ∃! (s : Finset ℕ),
    s.card = 3 ∧ (∀ n ∈ s, a n > b n) ∧ (∀ n ∉ s, a n ≤ b n)) :
  6 < a 1 ∧ a 1 ≤ 16 :=
sorry

end a1_range_for_three_greater_terms_l420_42099


namespace ginos_popsicle_sticks_l420_42002

/-- Gino's popsicle stick problem -/
theorem ginos_popsicle_sticks (initial : Real) (given : Real) (remaining : Real) :
  initial = 63.0 →
  given = 50.0 →
  remaining = initial - given →
  remaining = 13.0 := by sorry

end ginos_popsicle_sticks_l420_42002


namespace probability_not_snowing_l420_42004

theorem probability_not_snowing (p_snowing : ℚ) (h : p_snowing = 5/8) : 
  1 - p_snowing = 3/8 := by
sorry

end probability_not_snowing_l420_42004


namespace simplify_fraction_l420_42098

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end simplify_fraction_l420_42098


namespace miran_has_least_paper_l420_42086

def miran_paper : ℕ := 6
def junga_paper : ℕ := 13
def minsu_paper : ℕ := 10

theorem miran_has_least_paper : 
  miran_paper ≤ junga_paper ∧ miran_paper ≤ minsu_paper :=
sorry

end miran_has_least_paper_l420_42086


namespace solve_inequality_find_a_range_l420_42089

-- Define the function f
def f (x : ℝ) := |x + 2|

-- Part 1: Solve the inequality
theorem solve_inequality :
  {x : ℝ | 2 * f x < 4 - |x - 1|} = {x : ℝ | -7/3 < x ∧ x < -1} := by sorry

-- Part 2: Find the range of a
theorem find_a_range (m n : ℝ) (h1 : m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 := by sorry

end solve_inequality_find_a_range_l420_42089


namespace samara_tire_expense_l420_42030

/-- Calculates Samara's spending on tires given the other expenses -/
def samaras_tire_spending (alberto_total : ℕ) (samara_oil : ℕ) (samara_detailing : ℕ) (difference : ℕ) : ℕ :=
  alberto_total - (samara_oil + samara_detailing + difference)

theorem samara_tire_expense :
  samaras_tire_spending 2457 25 79 1886 = 467 := by
  sorry

end samara_tire_expense_l420_42030


namespace shares_owned_shares_owned_example_l420_42043

/-- Calculates the number of shares owned based on dividend payment and earnings -/
theorem shares_owned (expected_earnings dividend_ratio additional_dividend_rate actual_earnings total_dividend : ℚ) : ℚ :=
  let base_dividend := expected_earnings * dividend_ratio
  let additional_earnings := actual_earnings - expected_earnings
  let additional_dividend := (additional_earnings / 0.1) * additional_dividend_rate
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend / total_dividend_per_share

/-- Proves that the number of shares owned is 600 given the specific conditions -/
theorem shares_owned_example : shares_owned 0.8 0.5 0.04 1.1 312 = 600 := by
  sorry

end shares_owned_shares_owned_example_l420_42043


namespace sherry_age_l420_42060

theorem sherry_age (randolph_age sydney_age sherry_age : ℕ) : 
  randolph_age = 55 →
  randolph_age = sydney_age + 5 →
  sydney_age = 2 * sherry_age →
  sherry_age = 25 := by sorry

end sherry_age_l420_42060


namespace constant_term_expansion_l420_42036

def p (x : ℝ) : ℝ := x^3 + 2*x + 3
def q (x : ℝ) : ℝ := 2*x^4 + x^2 + 7

theorem constant_term_expansion : 
  (p 0) * (q 0) = 21 := by sorry

end constant_term_expansion_l420_42036


namespace shoe_repair_time_l420_42006

theorem shoe_repair_time (heel_time shoe_count total_time : ℕ) (h1 : heel_time = 10) (h2 : shoe_count = 2) (h3 : total_time = 30) :
  (total_time - heel_time * shoe_count) / shoe_count = 5 :=
by sorry

end shoe_repair_time_l420_42006


namespace negation_of_exists_greater_than_one_l420_42041

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by
  sorry

end negation_of_exists_greater_than_one_l420_42041


namespace max_product_constraint_l420_42055

theorem max_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : 6 * a + 8 * b = 72) :
  a * b ≤ 27 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 6 * a₀ + 8 * b₀ = 72 ∧ a₀ * b₀ = 27 := by
  sorry

end max_product_constraint_l420_42055


namespace initial_jasmine_percentage_l420_42031

/-- Proof of initial jasmine percentage in a solution --/
theorem initial_jasmine_percentage
  (initial_volume : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (final_jasmine_percentage : ℝ)
  (h1 : initial_volume = 100)
  (h2 : added_jasmine = 5)
  (h3 : added_water = 10)
  (h4 : final_jasmine_percentage = 8.695652173913043)
  : (100 * (initial_volume * (final_jasmine_percentage / 100) - added_jasmine) / initial_volume) = 5 := by
  sorry

end initial_jasmine_percentage_l420_42031


namespace min_profit_is_128_l420_42075

/-- The profit function for a stationery item -/
def profit (x : ℝ) : ℝ :=
  let y := -2 * x + 60
  y * (x - 10)

/-- The theorem stating the minimum profit -/
theorem min_profit_is_128 :
  ∃ (x_min : ℝ), 15 ≤ x_min ∧ x_min ≤ 26 ∧
  ∀ (x : ℝ), 15 ≤ x → x ≤ 26 → profit x_min ≤ profit x ∧
  profit x_min = 128 :=
sorry

end min_profit_is_128_l420_42075


namespace fourth_side_length_l420_42046

/-- A quadrilateral inscribed in a circle with three equal sides -/
structure InscribedQuadrilateral where
  -- The radius of the circumscribed circle
  r : ℝ
  -- The length of three equal sides
  s : ℝ
  -- Assumption that the radius is 150√2
  h1 : r = 150 * Real.sqrt 2
  -- Assumption that the three equal sides have length 150
  h2 : s = 150

/-- The length of the fourth side of the quadrilateral -/
def fourthSide (q : InscribedQuadrilateral) : ℝ := 375

/-- Theorem stating that the fourth side has length 375 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : 
  fourthSide q = 375 := by sorry

end fourth_side_length_l420_42046


namespace number_puzzle_l420_42067

theorem number_puzzle (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end number_puzzle_l420_42067


namespace arun_weight_estimation_l420_42094

/-- Arun's weight estimation problem -/
theorem arun_weight_estimation (x : ℝ) 
  (h1 : 65 < x)  -- Arun's lower bound
  (h2 : 60 < x ∧ x < 70)  -- Brother's estimation
  (h3 : x ≤ 68)  -- Mother's estimation
  (h4 : (65 + x) / 2 = 67)  -- Average of probable weights
  : x = 68 := by
  sorry

end arun_weight_estimation_l420_42094


namespace min_value_inequality_l420_42077

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 := by
  sorry

end min_value_inequality_l420_42077


namespace line_passes_through_fixed_point_l420_42096

/-- The line equation ax + y + a + 1 = 0 always passes through the point (-1, -1) for all values of a. -/
theorem line_passes_through_fixed_point (a : ℝ) : a * (-1) + (-1) + a + 1 = 0 := by
  sorry

end line_passes_through_fixed_point_l420_42096


namespace four_tuple_solution_l420_42012

theorem four_tuple_solution (x y z w : ℝ) 
  (h1 : x^2 + y^2 + z^2 + w^2 = 4)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 + 1/w^2 = 5 - 1/(x*y*z*w)^2) :
  (x = 1 ∨ x = -1) ∧ 
  (y = 1 ∨ y = -1) ∧ 
  (z = 1 ∨ z = -1) ∧ 
  (w = 1 ∨ w = -1) ∧
  (x*y*z*w = 1 ∨ x*y*z*w = -1) :=
by sorry

end four_tuple_solution_l420_42012


namespace first_donor_coins_l420_42048

theorem first_donor_coins (d1 d2 d3 d4 : ℕ) : 
  d2 = 2 * d1 →
  d3 = 3 * d2 →
  d4 = 4 * d3 →
  d1 + d2 + d3 + d4 = 132 →
  d1 = 4 := by
sorry

end first_donor_coins_l420_42048


namespace line_points_count_l420_42020

theorem line_points_count (n : ℕ) 
  (point1 : ∃ (a b : ℕ), a * b = 80 ∧ a + b + 1 = n)
  (point2 : ∃ (c d : ℕ), c * d = 90 ∧ c + d + 1 = n) :
  n = 22 := by
sorry

end line_points_count_l420_42020


namespace perpendicular_vectors_l420_42097

/-- Given vectors a and b in ℝ³, if a is perpendicular to b, then x = -2 -/
theorem perpendicular_vectors (a b : ℝ × ℝ × ℝ) (h : a.1 = -1 ∧ a.2.1 = 2 ∧ a.2.2 = 1/2) 
  (k : b.1 = -3 ∧ b.2.2 = 2) (perp : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) :
  b.2.1 = -2 := by
  sorry

end perpendicular_vectors_l420_42097


namespace shortest_player_height_l420_42064

/-- Given the height of the tallest player and the difference in height between
    the tallest and shortest players, calculate the height of the shortest player. -/
theorem shortest_player_height
  (tallest_height : ℝ)
  (height_difference : ℝ)
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5) :
  tallest_height - height_difference = 68.25 := by
  sorry

#check shortest_player_height

end shortest_player_height_l420_42064


namespace stock_investment_net_increase_l420_42042

theorem stock_investment_net_increase (x : ℝ) (x_pos : x > 0) :
  x * 1.5 * 0.7 = 1.05 * x := by
  sorry

end stock_investment_net_increase_l420_42042


namespace staircase_perimeter_l420_42061

/-- Represents a staircase-shaped region with specific properties -/
structure StaircaseRegion where
  right_angles : Bool
  congruent_segments : ℕ
  segment_length : ℝ
  area : ℝ
  bottom_width : ℝ

/-- Calculates the perimeter of a staircase-shaped region -/
def perimeter (s : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific staircase region -/
theorem staircase_perimeter :
  ∀ (s : StaircaseRegion),
    s.right_angles = true →
    s.congruent_segments = 8 →
    s.segment_length = 1 →
    s.area = 41 →
    s.bottom_width = 7 →
    perimeter s = 128 / 7 :=
by
  sorry

end staircase_perimeter_l420_42061
