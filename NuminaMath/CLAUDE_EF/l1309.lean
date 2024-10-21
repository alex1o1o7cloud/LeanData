import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_pairs_dividing_sum_of_squares_plus_one_l1309_130930

/-- A sequence of pairs of positive integers -/
def sequence_ab : ℕ → ℕ × ℕ := sorry

/-- The property that a and b are coprime -/
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The property that ab divides a² + b² + 1 -/
def divides_sum_of_squares_plus_one (a b : ℕ) : Prop :=
  ∃ k : ℕ, k * (a * b) = a^2 + b^2 + 1

theorem infinitely_many_coprime_pairs_dividing_sum_of_squares_plus_one :
  ∃ f : ℕ → ℕ × ℕ,
    (∀ n : ℕ, is_coprime (f n).1 (f n).2) ∧
    (∀ n : ℕ, divides_sum_of_squares_plus_one (f n).1 (f n).2) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_pairs_dividing_sum_of_squares_plus_one_l1309_130930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_a_range_l1309_130965

open Real

theorem log_inequality_implies_a_range :
  (∀ x ∈ Set.Ioc 1 2, logb a x > (x - 1)^2) → a ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_a_range_l1309_130965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_derivative_equality_sum_bound_l1309_130901

/-- Given a function f(x) = x^2(ln x - a) where a is a real number,
    prove that if f(x) has an extremum at x = e, f'(x₁) = f'(x₂), and x₁ < x₂,
    then 2 < x₁ + x₂ < e -/
theorem extremum_derivative_equality_sum_bound
  (f : ℝ → ℝ)
  (a : ℝ)
  (h_f : ∀ x, f x = x^2 * (Real.log x - a))
  (h_extremum : HasDerivAt f 0 (Real.exp 1))
  (x₁ x₂ : ℝ)
  (h_deriv_eq : deriv f x₁ = deriv f x₂)
  (h_lt : x₁ < x₂) :
  2 < x₁ + x₂ ∧ x₁ + x₂ < Real.exp 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_derivative_equality_sum_bound_l1309_130901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_24_hours_l1309_130949

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water, the stream's speed, and the distance to the destination. -/
noncomputable def total_time_round_trip (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  time_downstream + time_upstream

/-- Theorem stating that the total time for the given round trip is 24 hours. -/
theorem round_trip_time_is_24_hours :
  total_time_round_trip 9 1.5 105 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_24_hours_l1309_130949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_odd_numbers_without_three_in_ten_thousands_count_numbers_greater_than_21345_l1309_130958

/-- The number of permutations of n elements taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := 
  if r > n then 0
  else Nat.factorial n / Nat.factorial (n - r)

/-- The set of digits used -/
def digits : Finset ℕ := {1, 2, 3, 4, 5}

/-- A five-digit number formed from the given digits -/
structure FiveDigitNumber where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ
  d4 : ℕ
  d5 : ℕ
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d5 ∈ digits
  distinct : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
             d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧
             d3 ≠ d4 ∧ d3 ≠ d5 ∧
             d4 ≠ d5

theorem count_odd_numbers_without_three_in_ten_thousands (n : Finset FiveDigitNumber) :
  (n.filter (λ x => x.d5 % 2 = 1 ∧ x.d1 ≠ 3)).card = 60 := by
  sorry

theorem count_numbers_greater_than_21345 (n : Finset FiveDigitNumber) :
  (n.filter (λ x => x.d1 * 10000 + x.d2 * 1000 + x.d3 * 100 + x.d4 * 10 + x.d5 > 21345)).card = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_odd_numbers_without_three_in_ten_thousands_count_numbers_greater_than_21345_l1309_130958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1309_130986

open Set
open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (tan (2 * x)) / (sqrt (x - x^2))

-- Define the domain of f
def domain_f : Set ℝ := Ioo 0 (π/4) ∪ Ioo (π/4) 1

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1309_130986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_contains_10_5_ounces_l1309_130961

/-- Represents the coffee consumption and costs for Maddie's mom --/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  bag_cost : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  total_weekly_spend : ℚ

/-- Calculates the number of ounces of beans in a bag of coffee --/
noncomputable def ounces_in_bag (c : CoffeeConsumption) : ℚ :=
  let daily_ounces := c.cups_per_day * c.ounces_per_cup
  let weekly_ounces := daily_ounces * 7
  let weekly_milk_cost := c.milk_gallons_per_week * c.milk_cost_per_gallon
  let weekly_bean_cost := c.total_weekly_spend - weekly_milk_cost
  let cost_per_ounce := weekly_bean_cost / weekly_ounces
  c.bag_cost / cost_per_ounce

/-- Theorem stating that given Maddie's mom's coffee consumption, a bag contains 10.5 ounces of beans --/
theorem bag_contains_10_5_ounces :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    ounces_per_cup := 3/2,
    bag_cost := 8,
    milk_gallons_per_week := 1/2,
    milk_cost_per_gallon := 4,
    total_weekly_spend := 18
  }
  ounces_in_bag c = 21/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_contains_10_5_ounces_l1309_130961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_maze_probability_l1309_130947

structure Maze where
  junctions : Set Char
  paths : Set (Char × Char)
  start : Char
  goal : Char

noncomputable def prob_reach (m : Maze) (j : Char) : ℚ :=
  sorry

theorem harry_maze_probability (m : Maze) :
  m.junctions = {'S', 'V', 'W', 'X', 'Y', 'Z', 'B'} →
  m.paths = {('S', 'V'), ('V', 'W'), ('V', 'X'), ('V', 'Y'), ('W', 'X'), ('W', 'Z'), ('Y', 'X'), ('Y', 'Z'), ('X', 'B'), ('Z', 'B')} →
  m.start = 'S' →
  m.goal = 'B' →
  prob_reach m 'B' = 11 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_maze_probability_l1309_130947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_men_is_30_l1309_130999

/-- Represents the road construction project --/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  completedLength : ℚ
  completedDays : ℚ
  extraMen : ℚ

/-- Calculates the initial number of men employed in the project --/
def initialMen (project : RoadProject) : ℚ :=
  let remainingLength := project.totalLength - project.completedLength
  let remainingDays := project.totalDays - project.completedDays
  let initialRate := project.completedLength / project.completedDays
  let requiredRate := remainingLength / remainingDays
  (requiredRate * project.extraMen) / (requiredRate - initialRate)

/-- Theorem stating that the initial number of men employed is 30 --/
theorem initial_men_is_30 (project : RoadProject) 
    (h1 : project.totalLength = 10)
    (h2 : project.totalDays = 300)
    (h3 : project.completedLength = 2)
    (h4 : project.completedDays = 100)
    (h5 : project.extraMen = 30) : 
  initialMen project = 30 := by
  sorry

def main : IO Unit := do
  let result := initialMen { totalLength := 10, totalDays := 300, completedLength := 2, completedDays := 100, extraMen := 30 }
  IO.println s!"The initial number of men employed: {result}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_men_is_30_l1309_130999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_6_7_8_before_less_than_5_l1309_130924

/-- Represents a fair 8-sided die -/
def Die := Fin 8

/-- The probability of rolling a specific number on the die -/
noncomputable def prob_roll (n : Die) : ℝ := 1 / 8

/-- The probability of rolling a number less than 5 -/
noncomputable def prob_less_than_5 : ℝ := 4 / 8

/-- The probability of rolling 6, 7, or 8 -/
noncomputable def prob_6_7_8 : ℝ := 3 / 8

/-- The probability of rolling 6, 7, and 8 in order, given that we only consider rolls of 6, 7, or 8 -/
noncomputable def prob_6_7_8_in_order : ℝ := 1 / 6 * 1 / 5 * 1 / 4

/-- The probability of not rolling a number less than 5 in three consecutive rolls -/
noncomputable def prob_no_less_than_5_in_three_rolls : ℝ := (1 / 2) ^ 3

theorem prob_6_7_8_before_less_than_5 :
  prob_no_less_than_5_in_three_rolls * prob_6_7_8_in_order = 1 / 960 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_6_7_8_before_less_than_5_l1309_130924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1309_130945

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 
  7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

-- State the theorem
theorem f_max_min :
  (∀ x : ℝ, f x ≤ 10) ∧ (∃ x : ℝ, f x = 10) ∧
  (∀ x : ℝ, f x ≥ 6) ∧ (∃ x : ℝ, f x = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1309_130945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_common_difference_is_3_min_value_achieved_l1309_130937

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d > 0
  h2 : ∀ n, a n = a 1 + (n - 1) * d
  h3 : a 1 = 5
  h4 : (a 5 - 1) ^ 2 = a 2 * a 10

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The expression to be minimized -/
noncomputable def F (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (2 * S seq n + n + 32) / (seq.a n + 1)

/-- The main theorem stating the minimum value of F -/
theorem min_value_F (seq : ArithmeticSequence) :
  ∀ n : ℕ, n > 0 → F seq n ≥ 20/3 := by
  sorry

/-- Proof that the common difference d is 3 -/
theorem common_difference_is_3 (seq : ArithmeticSequence) : seq.d = 3 := by
  sorry

/-- Proof that the minimum value is achieved -/
theorem min_value_achieved (seq : ArithmeticSequence) :
  ∃ n : ℕ, n > 0 ∧ F seq n = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_common_difference_is_3_min_value_achieved_l1309_130937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_properties_l1309_130908

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - 2 * Real.log x

-- State the theorem
theorem extreme_points_properties (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x > 0, ∃ y, f a x = y) →  -- f is defined for all x > 0
  (x₁ < x₂) →  -- x₁ is less than x₂
  (∃ ε > 0, ∀ x, x₁ - ε < x ∧ x < x₁ + ε → f a x₁ ≥ f a x) →  -- x₁ is a local maximum
  (∃ ε > 0, ∀ x, x₂ - ε < x ∧ x < x₂ + ε → f a x₂ ≤ f a x) →  -- x₂ is a local minimum
  (0 < a ∧ a < 1) ∧ f a x₂ < x₂ - 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_properties_l1309_130908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nikola_completed_four_jobs_l1309_130987

/- Define the problem parameters -/
def num_ants : ℕ := 400
def food_per_ant : ℕ := 2
def food_cost_per_ounce : ℚ := 1 / 10
def job_start_fee : ℕ := 5
def cost_per_leaf : ℚ := 1 / 100
def leaves_raked : ℕ := 6000

/- Define the total cost of food -/
def total_food_cost : ℚ := ↑num_ants * ↑food_per_ant * food_cost_per_ounce

/- Define the money earned from raking leaves -/
def leaf_raking_earnings : ℚ := ↑leaves_raked * cost_per_leaf

/- Define the money earned from starting jobs -/
def job_start_earnings : ℚ := total_food_cost - leaf_raking_earnings

/- Define the number of jobs completed -/
def jobs_completed : ℕ := (job_start_earnings / ↑job_start_fee).floor.toNat

/- Theorem statement -/
theorem nikola_completed_four_jobs : jobs_completed = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nikola_completed_four_jobs_l1309_130987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_stopping_distance_l1309_130904

/-- Represents the distance traveled by a car in a given second -/
def distance_in_second (n : ℕ) : ℝ :=
  35 - 5 * (n - 1)

/-- Calculates the total distance traveled by the car until it stops -/
def total_distance : ℝ :=
  List.sum (List.map distance_in_second (List.range 8))

/-- Theorem stating that the total distance traveled by the car is 140 meters -/
theorem car_stopping_distance : total_distance = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_stopping_distance_l1309_130904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_divides_l1309_130953

theorem exists_n_divides (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ n : ℕ, n > 0 ∧ a ∣ b^n - n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_divides_l1309_130953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_percentage_l1309_130984

noncomputable def initial_job_income : ℝ := 60
noncomputable def initial_freelance_income : ℝ := 40
noncomputable def initial_online_income : ℝ := 20

noncomputable def new_job_income : ℝ := 120
noncomputable def new_freelance_income : ℝ := 60
noncomputable def new_online_income : ℝ := 35

def weeks_per_month : ℕ := 4

noncomputable def total_initial_weekly_income : ℝ := initial_job_income + initial_freelance_income + initial_online_income
noncomputable def total_new_weekly_income : ℝ := new_job_income + new_freelance_income + new_online_income

noncomputable def total_initial_monthly_income : ℝ := total_initial_weekly_income * weeks_per_month
noncomputable def total_new_monthly_income : ℝ := total_new_weekly_income * weeks_per_month

noncomputable def percentage_increase : ℝ := (total_new_monthly_income - total_initial_monthly_income) / total_initial_monthly_income * 100

theorem income_increase_percentage :
  abs (percentage_increase - 79.17) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_percentage_l1309_130984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6006_as_difference_of_powers_l1309_130970

theorem multiples_of_6006_as_difference_of_powers (n : ℕ) : 
  (n = 150) →
  (∃ k : ℕ, k = 1825) →
  (∃ f : ℕ → ℕ → Bool, 
    (∀ i j, f i j = true ↔ (i < j ∧ j ≤ n ∧ (6006 ∣ (12^j - 12^i)))) →
    (k = (Finset.sum (Finset.range (n + 1)) (λ i => 
      (Finset.sum (Finset.range (n + 1)) (λ j => 
        if f i j then 1 else 0))))))
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6006_as_difference_of_powers_l1309_130970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l1309_130985

noncomputable section

/-- The area of a circular sector with perimeter 3cm and central angle 1 radian -/
def sector_area (perimeter : ℝ) (angle : ℝ) : ℝ :=
  (perimeter^2 / (4 * Real.pi + 2 * angle)) * angle / 2

theorem sector_area_specific : 
  sector_area 3 1 = (1/2) * Real.sin 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l1309_130985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l1309_130994

theorem max_value_x_plus_2y (x y : ℝ) (h : (2 : ℝ)^x + (4 : ℝ)^y = 1) : 
  ∀ z : ℝ, x + 2*y ≤ z → z ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l1309_130994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cube_volume_solution_correctness_l1309_130943

def cube_volume (edge : ℝ) : ℝ := edge ^ 3

theorem remaining_cube_volume (original_edge small_edge : ℝ) 
  (h1 : original_edge = 3)
  (h2 : small_edge = 1) :
  cube_volume original_edge - 8 * cube_volume small_edge = 19 := by
  -- Replace the cube_volume with its definition
  unfold cube_volume
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the expression
  norm_num

#eval cube_volume 3 - 8 * cube_volume 1

theorem solution_correctness :
  ∃ (original_edge small_edge : ℝ),
    original_edge = 3 ∧
    small_edge = 1 ∧
    cube_volume original_edge - 8 * cube_volume small_edge = 19 := by
  use 3, 1
  constructor
  · rfl
  constructor
  · rfl
  · exact remaining_cube_volume 3 1 rfl rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cube_volume_solution_correctness_l1309_130943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_sum_theorem_l1309_130933

theorem sign_sum_theorem (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (x : ℝ), x ∈ ({5, 1, -1, -5} : Set ℝ) ∧
  (a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d) = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_sum_theorem_l1309_130933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1309_130995

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- The intersection point of the circle and asymptote -/
noncomputable def intersection_point (h : Hyperbola a b) : ℝ × ℝ :=
  sorry

/-- The area of the triangle formed by origin, right focus, and intersection point -/
noncomputable def triangle_area (h : Hyperbola a b) : ℝ :=
  sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  triangle_area h = b^2 → eccentricity h = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1309_130995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_function_correct_l1309_130905

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | (0 ≤ p.1 ∧ p.1 ≤ 1 ∧ p.2 = 0) ∨
       (p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨
       (0 ≤ p.1 ∧ p.1 ≤ 1 ∧ p.2 = 1) ∨
       (p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1)}

-- Define the distance function
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x
  else if 1 < x ∧ x ≤ 2 then Real.sqrt (1 + (x - 1)^2)
  else if 2 < x ∧ x ≤ 3 then Real.sqrt (1 + (3 - x)^2)
  else if 3 < x ∧ x ≤ 4 then 4 - x
  else 0  -- Default case for x outside [0, 4]

-- Theorem statement
theorem distance_function_correct :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 →
  ∃ P : ℝ × ℝ, P ∈ square ∧
  f x = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_function_correct_l1309_130905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_beam_path_length_l1309_130948

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The cube side length -/
def cubeSideLength : ℝ := 10

/-- The reflection point on the cube face -/
def reflectionPoint : Point3D := ⟨6, 10, 3⟩

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: The light beam's path length is 10√145 -/
theorem light_beam_path_length :
  let startPoint : Point3D := ⟨0, 10, 0⟩
  let pathLength := distance startPoint reflectionPoint * 2
  pathLength = 10 * Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_beam_path_length_l1309_130948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_points_l1309_130978

/-- The slope angle of a line passing through two points -/
noncomputable def slope_angle (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.arctan ((y2 - y1) / (x2 - x1))

/-- Theorem: The slope angle of the line passing through (-3, 2) and (-2, 3) is π/4 -/
theorem slope_angle_specific_points :
  slope_angle (-3) 2 (-2) 3 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_points_l1309_130978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l1309_130993

theorem coefficient_x4_in_expansion : ∃ (c : ℕ), 
  c = (Nat.choose 5 2) * 2^2 ∧ 
  c = (Finset.range 6).sum (λ k ↦ Nat.choose 5 k * (2^k) * 
    (if 10 - 3*k = 4 then 1 else 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l1309_130993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_angles_l1309_130935

/-- A triangle with special height properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Side lengths
  a : ℝ -- length of BC
  b : ℝ -- length of AC
  c : ℝ -- length of AB
  -- Heights
  h_a : ℝ -- height relative to BC
  h_c : ℝ -- height relative to AB
  -- Properties
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  height_condition_a : h_a ≥ a
  height_condition_c : h_c ≥ c

/-- The angles of a special triangle are 45°, 45°, and 90° -/
theorem special_triangle_angles (t : SpecialTriangle) :
  ∃ (α β γ : ℝ), α = 45 ∧ β = 45 ∧ γ = 90 ∧
  (∃ (angleABC angleBCA angleCAB : ℝ), 
    angleABC = α ∧ angleBCA = β ∧ angleCAB = γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_angles_l1309_130935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_supply_duration_l1309_130968

/-- Represents the number of days a water tank can supply a village -/
noncomputable def days_of_supply (capacity : ℝ) (daily_consumption : ℝ) (leak : ℝ) : ℝ :=
  capacity / (daily_consumption + leak)

/-- Theorem: A tank that lasts 60 days with a 10 L/day leak and 48 days with a 20 L/day leak will last 80 days without a leak -/
theorem tank_supply_duration (capacity : ℝ) (daily_consumption : ℝ) 
  (h1 : days_of_supply capacity daily_consumption 10 = 60)
  (h2 : days_of_supply capacity daily_consumption 20 = 48) :
  days_of_supply capacity daily_consumption 0 = 80 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_supply_duration_l1309_130968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_condition_l1309_130922

-- Define the basic types
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the types for lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := V × V
def Plane (V : Type*) [NormedAddCommGroup V] := V × V

-- Define the perpendicular relation
def perpendicular (l m : Line V) : Prop := sorry

-- Define the perpendicular relation between a line and a plane
def perpendicular_plane (l : Line V) (π : Plane V) : Prop := sorry

-- Define the subset relation for a line in a plane
def line_in_plane (m : Line V) (π : Plane V) : Prop := sorry

-- State the theorem
theorem perpendicular_line_plane_condition 
  (m : Line V) (π : Plane V) (h : line_in_plane m π) : 
  (∀ l : Line V, perpendicular_plane l π → perpendicular l m) ∧ 
  (∃ l : Line V, perpendicular l m ∧ ¬perpendicular_plane l π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_condition_l1309_130922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l1309_130963

/-- Proves that the speed of the stream is 20 kmph given the conditions -/
theorem stream_speed (boat_speed stream_speed : ℝ) : 
  boat_speed = 60 →
  (fun d ↦ d / (boat_speed - stream_speed)) = (fun d ↦ 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l1309_130963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_quadratic_inequality_l1309_130939

theorem range_of_m_for_quadratic_inequality :
  ∀ m : ℝ, (∀ x > 0, m * x^2 + 2 * x + m ≤ 0) ↔ m ∈ Set.Iic (-1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_quadratic_inequality_l1309_130939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l1309_130917

theorem seed_germination_problem (seeds_second_plot : ℚ) : 
  (300 * (15 / 100) + seeds_second_plot * (35 / 100) = (300 + seeds_second_plot) * (23 / 100)) →
  seeds_second_plot = 200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l1309_130917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inequality_1_tangent_inequality_2_l1309_130942

open Set Real

-- Define the set of x values satisfying the first inequality
def S₁ : Set ℝ := {x | ∃ k : ℤ, k * π - π / 6 ≤ x ∧ x < k * π + π / 2}

-- Define the set of x values satisfying the second inequality
def S₂ : Set ℝ := {x | ∃ k : ℤ, k * π - π / 2 < x ∧ x ≤ k * π + π / 3}

-- Theorem for the first inequality
theorem tangent_inequality_1 :
  {x : ℝ | Real.sqrt 3 / 3 + Real.tan x ≥ 0} = S₁ :=
sorry

-- Theorem for the second inequality
theorem tangent_inequality_2 :
  {x : ℝ | Real.tan x - Real.sqrt 3 ≤ 0} = S₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inequality_1_tangent_inequality_2_l1309_130942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missouri_to_new_york_distance_l1309_130926

/-- The distance between two locations by plane -/
noncomputable def plane_distance : ℝ := 2000

/-- The increase factor for car travel compared to plane travel -/
noncomputable def car_increase_factor : ℝ := 1.4

/-- The distance between two locations by car -/
noncomputable def car_distance : ℝ := plane_distance * car_increase_factor

/-- The distance from the midpoint to either end by car -/
noncomputable def midpoint_to_end_distance : ℝ := car_distance / 2

theorem missouri_to_new_york_distance :
  midpoint_to_end_distance = 1400 := by
  -- Unfold the definitions
  unfold midpoint_to_end_distance car_distance plane_distance car_increase_factor
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missouri_to_new_york_distance_l1309_130926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_tangent_to_circle_l1309_130955

/-- Rotation of a point (x, y) around (a, b) by angle θ --/
noncomputable def rotate_point (x y a b θ : ℝ) : ℝ × ℝ :=
  (a + (x - a) * Real.cos θ - (y - b) * Real.sin θ,
   b + (x - a) * Real.sin θ + (y - b) * Real.cos θ)

/-- Line equation after rotation --/
noncomputable def rotated_line (x y : ℝ) : Prop :=
  let (x', y') := rotate_point x y 1 0 (15 * Real.pi / 180)
  x' + y' - 1 = 0

/-- Circle equation --/
def circle_eq (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4

/-- Theorem stating that the rotated line is tangent to the circle --/
theorem rotated_line_tangent_to_circle :
  ∃! (x y : ℝ), rotated_line x y ∧ circle_eq x y :=
sorry

#check rotated_line_tangent_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_tangent_to_circle_l1309_130955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_l1309_130900

-- Define the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (0, -2)
def E : ℝ × ℝ := (2, 2)

-- Define the angles in radians
noncomputable def angle_BAD : ℝ := 50 * Real.pi / 180
noncomputable def angle_ABD : ℝ := 60 * Real.pi / 180
noncomputable def angle_BDE : ℝ := 70 * Real.pi / 180
noncomputable def angle_EDB : ℝ := 80 * Real.pi / 180
noncomputable def angle_DEC : ℝ := 90 * Real.pi / 180
noncomputable def angle_ECD : ℝ := 30 * Real.pi / 180

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_segment :
  distance D E > distance A B ∧
  distance D E > distance B D ∧
  distance D E > distance E C ∧
  distance D E > distance C A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_l1309_130900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equation_l1309_130911

theorem power_of_three_equation (x : ℝ) : (3 : ℝ)^2 * (3 : ℝ)^x = 81 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equation_l1309_130911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warning_is_correct_l1309_130969

def warning : String := "warning"

theorem warning_is_correct : warning = "warning" := by
  rfl

#check warning_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warning_is_correct_l1309_130969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_moves_equal_l1309_130918

/-- The probability of a meteor hitting a square -/
noncomputable def p : ℝ := sorry

/-- The expected number of valid moves for a knight -/
noncomputable def knight_moves : ℝ := 8 * (1 - p)

/-- The expected number of valid moves for a bishop -/
noncomputable def bishop_moves : ℝ := 4 * ((1 - p) / p)

/-- Theorem stating that the expected number of moves are equal when p = 1/2 -/
theorem expected_moves_equal : 
  knight_moves = bishop_moves ↔ p = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_moves_equal_l1309_130918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_payment_distribution_l1309_130990

/-- Represents the time in weeks for a team to complete the renovation alone -/
structure TeamTime where
  weeks : ℚ
  weeks_positive : weeks > 0

/-- Represents the renovation project -/
structure RenovationProject where
  team_a_time : TeamTime
  team_b_time : TeamTime
  team_a_solo_weeks : ℚ
  team_a_solo_weeks_positive : team_a_solo_weeks > 0
  total_cost : ℚ
  total_cost_positive : total_cost > 0

/-- Calculates the work done by each team -/
noncomputable def work_done (project : RenovationProject) : ℚ × ℚ :=
  let remaining_weeks := 
    (1 / project.team_a_time.weeks + 1 / project.team_b_time.weeks)⁻¹ *
    (1 - project.team_a_solo_weeks / project.team_a_time.weeks)
  let team_a_work := project.team_a_solo_weeks / project.team_a_time.weeks + 
    remaining_weeks / project.team_a_time.weeks
  let team_b_work := remaining_weeks / project.team_b_time.weeks
  (team_a_work, team_b_work)

/-- Theorem stating that the payment should be equally distributed -/
theorem equal_payment_distribution (project : RenovationProject) 
  (h1 : project.team_a_time.weeks = 18)
  (h2 : project.team_b_time.weeks = 12)
  (h3 : project.team_a_solo_weeks = 3)
  (h4 : project.total_cost = 4000) :
  let (work_a, work_b) := work_done project
  work_a = work_b ∧ work_a = 1/2 ∧ work_b = 1/2 := by
  sorry

#check equal_payment_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_payment_distribution_l1309_130990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_degree_l1309_130925

noncomputable def min_degree (c : ℝ) : ℕ := ⌈(Real.pi / Real.arccos (c / 2) : ℝ)⌉.toNat

theorem polynomial_equation_degree (c : ℝ) :
  0 < c →
  (c < 2 →
    ∃ (f g : Polynomial ℝ) (n : ℕ),
      (∀ i, 0 ≤ f.coeff i) ∧
      (∀ i, 0 ≤ g.coeff i) ∧
      (X^2 - c • X + 1 : Polynomial ℝ) * g = f ∧
      f.degree = some n ∧
      n = min_degree c ∧
      ∀ m < n, ¬∃ (f' g' : Polynomial ℝ),
        (∀ i, 0 ≤ f'.coeff i) ∧
        (∀ i, 0 ≤ g'.coeff i) ∧
        (X^2 - c • X + 1 : Polynomial ℝ) * g' = f' ∧
        f'.degree = some m) ∧
  (c ≥ 2 →
    ¬∃ (f g : Polynomial ℝ),
      (∀ i, 0 ≤ f.coeff i) ∧
      (∀ i, 0 ≤ g.coeff i) ∧
      (X^2 - c • X + 1 : Polynomial ℝ) * g = f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_degree_l1309_130925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1309_130944

theorem quadratic_equation_solution : ∃ (a b : ℕ), 
  (∀ x : ℝ, x^2 + 10*x = 40 → x = Real.sqrt (a : ℝ) - b ∨ x ≠ Real.sqrt (a : ℝ) - b) ∧
  (Real.sqrt (a : ℝ) - b > 0) ∧
  a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1309_130944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l1309_130976

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between the focus and the asymptote of a hyperbola -/
def focus_asymptote_distance (h : Hyperbola) : ℝ :=
  h.b

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ :=
  2 * h.a

/-- Theorem: If a hyperbola has eccentricity 5/4 and focus-asymptote distance 3,
    then its real axis length is 8 -/
theorem hyperbola_real_axis_length 
    (h : Hyperbola) 
    (h_eccentricity : eccentricity h = 5/4)
    (h_distance : focus_asymptote_distance h = 3) : 
    real_axis_length h = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l1309_130976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lois_book_purchase_l1309_130906

/-- The number of books Lois purchased from the book store --/
def books_purchased (initial_books : ℕ) (nephew_fraction : ℚ) (library_fraction : ℚ) (final_books : ℕ) : ℕ :=
  (final_books : ℤ) - ((initial_books : ℤ) - (initial_books * nephew_fraction).floor - ((initial_books - (initial_books * nephew_fraction).floor) * library_fraction).floor) |>.toNat

/-- Theorem stating that Lois purchased 3 books --/
theorem lois_book_purchase :
  books_purchased 40 (1/4) (1/3) 23 = 3 := by
  rw [books_purchased]
  norm_num
  rfl

#eval books_purchased 40 (1/4) (1/3) 23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lois_book_purchase_l1309_130906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l1309_130991

/-- A predicate to check if six points form a cube -/
def is_cube (P Q R S T U : ℝ × ℝ × ℝ) : Prop := sorry

/-- Function to calculate the volume of a cube given its vertices -/
def cube_volume (P Q R S T U : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Function to calculate the volume of a pyramid given its vertices -/
def pyramid_volume (P Q R S : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Given a cube PQRSTU with volume 8, prove that the volume of pyramid PQRS is 4/3 -/
theorem pyramid_volume_in_cube (P Q R S T U : ℝ × ℝ × ℝ) : 
  is_cube P Q R S T U → 
  cube_volume P Q R S T U = 8 → 
  pyramid_volume P Q R S = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_cube_l1309_130991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l1309_130954

theorem line_segment_endpoint (x : ℝ) : 
  let start : ℝ × ℝ := (3, -2)
  let end_point : ℝ × ℝ := (x, 10)
  let length : ℝ := 15
  (end_point.1 - start.1)^2 + (end_point.2 - start.2)^2 = length^2 → x = 12 ∨ x = -6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l1309_130954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_set_pairs_with_union_l1309_130981

theorem count_set_pairs_with_union : 
  ∃ (pairs : List (Set (Fin 2) × Set (Fin 2))), 
    (∀ (p : Set (Fin 2) × Set (Fin 2)), p ∈ pairs ↔ p.1 ∪ p.2 = Set.univ) ∧ 
    pairs.length = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_set_pairs_with_union_l1309_130981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_length_is_410_l1309_130921

/-- Calculate the total fence length for three rectangular yards -/
noncomputable def total_fence_length (yard_a_side : ℝ) (yard_a_area : ℝ) 
                       (yard_b_side : ℝ) (yard_b_area : ℝ) 
                       (yard_c_side : ℝ) (yard_c_area : ℝ) : ℝ :=
  let yard_a_other_side := yard_a_area / yard_a_side
  let yard_b_other_side := yard_b_area / yard_b_side
  let yard_c_other_side := yard_c_area / yard_c_side
  let yard_a_fence := yard_a_side + 2 * yard_a_other_side
  let yard_b_fence := yard_b_side + 2 * yard_b_other_side
  let yard_c_fence := yard_c_side + 2 * yard_c_other_side
  yard_a_fence + yard_b_fence + yard_c_fence

theorem fence_length_is_410 : 
  total_fence_length 40 320 60 480 80 720 = 410 := by
  -- Unfold the definition of total_fence_length
  unfold total_fence_length
  -- Simplify the arithmetic expressions
  simp [add_assoc, mul_add, add_mul]
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_length_is_410_l1309_130921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sixes_l1309_130907

/-- A larger cube constructed from 27 smaller dice. -/
structure LargeCube where
  small_cubes : Fin 27 → Die

/-- A standard six-sided die. -/
inductive Die
  | one | two | three | four | five | six

/-- The position of a small cube in the large cube. -/
inductive Position
  | face_center
  | edge_center
  | corner

/-- The probability of showing a six for each position. -/
def prob_six (pos : Position) : ℚ :=
  match pos with
  | Position.face_center => 1/6
  | Position.edge_center => 1/3
  | Position.corner => 1/2

/-- The number of small cubes in each position. -/
def num_cubes (pos : Position) : ℕ :=
  match pos with
  | Position.face_center => 6
  | Position.edge_center => 12
  | Position.corner => 8

/-- The expected number of sixes showing on the outer surface of the larger cube. -/
theorem expected_sixes (cube : LargeCube) :
  (prob_six Position.face_center * num_cubes Position.face_center) +
  (prob_six Position.edge_center * num_cubes Position.edge_center) +
  (prob_six Position.corner * num_cubes Position.corner) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sixes_l1309_130907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_part_i_part_ii_part_iii_l1309_130957

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- We start from index 1, so we can define a(0) arbitrarily
  | 1 => 2  -- a_1 = 2 (given in the problem)
  | 2 => 1  -- a_2 = 1 (given in the problem)
  | (n + 3) => if a (n + 2) / a (n + 1) > 1 then a (n + 2) / a (n + 1) else a (n + 1) / a (n + 2)

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := max (a (2*n - 1)) (a (2*n))

-- The main theorem
theorem sequence_properties :
  (∃ k, a k = 1 → ∀ m, ∃ n ≥ m, a n = 1) ∧
  ((∀ n, a n ≠ 1) → ∀ n, b n > b (n + 1)) :=
by
  sorry

-- Additional theorems for specific parts of the problem

-- Part I: Values of a_4 and a_5
theorem part_i : a 4 = 2 ∧ a 5 = 1 :=
by
  sorry

-- Part II: If a_k = 1, then there are infinitely many terms equal to 1
theorem part_ii : ∀ k, a k = 1 → ∀ m, ∃ n ≥ m, a n = 1 :=
by
  sorry

-- Part III: If no term equals 1, then b_n is monotonically decreasing
theorem part_iii : (∀ n, a n ≠ 1) → ∀ n, b n > b (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_part_i_part_ii_part_iii_l1309_130957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_tangent_to_circle_l1309_130946

open Real

-- Define the given point P and circle C with center O and radius r1
variable (P O : ℝ × ℝ)
variable (r1 r2 : ℝ)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem circle_through_point_tangent_to_circle :
  ∃ Q : ℝ × ℝ, 
    (distance Q O = |r1 - r2| ∨ distance Q O = r1 + r2) ∧ 
    distance Q P = r2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_tangent_to_circle_l1309_130946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_problem_l1309_130929

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t/2, Real.sqrt 2/2 + Real.sqrt 3 * t/2)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ - Real.pi/4)

theorem intersection_problem :
  -- The slope angle of line l is π/3
  (∃ α : ℝ, α = Real.pi/3 ∧ ∀ t : ℝ, (line_l t).2 - (line_l 0).2 = Real.tan α * ((line_l t).1 - (line_l 0).1)) ∧
  -- The length of AB is √10/2
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ t₁ t₂ θ₁ θ₂ : ℝ, 
      A = line_l t₁ ∧ B = line_l t₂ ∧
      (curve_C θ₁ * Real.cos θ₁, curve_C θ₁ * Real.sin θ₁) = A ∧
      (curve_C θ₂ * Real.cos θ₂, curve_C θ₂ * Real.sin θ₂) = B ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_problem_l1309_130929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1309_130989

-- Define the series
noncomputable def series_term (n : ℕ) : ℚ := 
  if n % 2 = 0 then 1 / 7^(n+1) else 2 / 7^(n+1)

noncomputable def infinite_series : ℚ := 10 * 79 * (∑' n, series_term n)

-- Theorem statement
theorem series_sum : infinite_series = 3/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1309_130989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_t_is_one_max_t_for_inequality_l1309_130998

-- Define the function f(x) with parameter t
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^2 - (2*t + 1)*x + t * Real.log x

-- Define the function g(x) with parameter t
def g (t : ℝ) (x : ℝ) : ℝ := (1 - t) * x

-- Theorem for part (1)
theorem extreme_values_when_t_is_one :
  let f₁ := f 1
  ∃ (x_max x_min : ℝ), 
    (∀ x > 0, f₁ x ≤ f₁ x_max) ∧
    (∀ x > 0, f₁ x ≥ f₁ x_min) ∧
    f₁ x_max = -5/4 - Real.log 2 ∧
    f₁ x_min = -2 := by sorry

-- Theorem for part (2)
theorem max_t_for_inequality :
  ∃ (t_max : ℝ),
    t_max = (Real.exp 1) * (Real.exp 1 - 2) / (Real.exp 1 - 1) ∧
    (∀ t > t_max, ¬∃ x ∈ Set.Icc 1 (Real.exp 1), f t x ≥ g t x) ∧
    (∀ t ≤ t_max, ∃ x ∈ Set.Icc 1 (Real.exp 1), f t x ≥ g t x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_t_is_one_max_t_for_inequality_l1309_130998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1309_130988

noncomputable section

def vector := ℝ × ℝ

def dot_product (v w : vector) : ℝ := v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector) : Prop := dot_product v w = 0

def parallel (v w : vector) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

noncomputable def magnitude (v : vector) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_sum_magnitude (x y : ℝ) :
  let a : vector := (x, 1)
  let b : vector := (1, y)
  let c : vector := (2, -4)
  perpendicular a c → parallel b c →
  magnitude (a.1 + b.1, a.2 + b.2) = Real.sqrt 10 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1309_130988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l1309_130980

noncomputable def f (x a : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≥ -4) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = -4) →
  a = -4 := by
  sorry

#check min_value_implies_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l1309_130980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_middle_divisor_implies_large_prime_divisor_l1309_130967

theorem no_middle_divisor_implies_large_prime_divisor (n : ℕ) (hn : n > 0) :
  (∀ d : ℕ, d > 0 → d ∣ n → ¬(n^2 ≤ d^4 ∧ d^4 ≤ n^3)) →
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ p^4 > n^3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_middle_divisor_implies_large_prime_divisor_l1309_130967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1309_130936

/-- The function f(x) = 2sin(x)cos(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

/-- The function g(x) obtained by translating f(x) π/12 units left and 1 unit up -/
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/12) + 1

/-- Theorem stating that the minimum value of |2x₁ + x₂| is π/3 given the conditions -/
theorem min_value_theorem (x₁ x₂ : ℝ) (h : f x₁ * g x₂ = 2) :
  (|2 * x₁ + x₂|) = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1309_130936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_interval_difference_l1309_130962

/-- Definition of the function P(x) --/
noncomputable def P (t : ℝ) (x : ℝ) : ℝ := ((t^2 + t) * x - 1) / (t^2 * x)

/-- Definition of a "good interval" --/
def isGoodInterval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧ 
  (∀ x, m ≤ x ∧ x ≤ n → m ≤ f x ∧ f x ≤ n) ∧
  (∀ x y, m ≤ x ∧ x < y ∧ y ≤ n → f x < f y)

/-- Main theorem statement --/
theorem max_good_interval_difference (t : ℝ) (m n : ℝ) 
  (ht : t ≠ 0) (h_good : isGoodInterval (P t) m n) :
  n - m ≤ 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_interval_difference_l1309_130962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l1309_130932

/-- The shaded area in a square with circles -/
theorem shaded_area_square_with_circles (square_side : ℝ) (num_circles : ℕ) : 
  square_side = 20 → num_circles = 9 → 
  (square_side^2 - num_circles * Real.pi * (square_side / (2 * Real.sqrt (num_circles : ℝ)))^2) = 400 - 100 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l1309_130932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_ratio_sum_l1309_130979

-- Define the prism structure
structure Prism where
  base_sides : Nat
  height : ℝ
  edge_length : ℝ

-- Define the point type
def Point := ℝ × ℝ × ℝ

-- Define the midpoint type
def Midpoint := Point

-- Define the intersection point type
def IntersectionPoint := Point

-- Define the prism with octagonal base and unit edge length
def octagonal_prism : Prism :=
  { base_sides := 8,
    height := 1,
    edge_length := 1 }

-- Define the set of midpoints
noncomputable def midpoints : Finset Midpoint := sorry

-- Define a point inside the prism
def interior_point (p : Point) : Prop := sorry

-- Define the intersection points
noncomputable def intersection_points (p : Point) : Finset IntersectionPoint := sorry

-- Define the condition that intersection points are not on edges
def not_on_edges (p : Point) : Prop := sorry

-- Define the condition that each face has exactly one intersection point
def one_per_face (p : Point) : Prop := sorry

-- Define the ratio sum
noncomputable def ratio_sum (p : Point) : ℝ := sorry

-- Theorem statement
theorem prism_ratio_sum 
  (p : Point) 
  (h_interior : interior_point p) 
  (h_not_on_edges : not_on_edges p) 
  (h_one_per_face : one_per_face p) : 
  ratio_sum p = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_ratio_sum_l1309_130979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_l1309_130902

-- Define the given conditions
noncomputable def distance : ℝ := 1.5
noncomputable def speed_day1 : ℝ := 3
noncomputable def speed_day2 : ℝ := 6
noncomputable def late_minutes : ℝ := 7

-- Define the function to calculate travel time in minutes
noncomputable def travel_time (d : ℝ) (s : ℝ) : ℝ := (d / s) * 60

-- Theorem statement
theorem early_arrival : 
  (travel_time distance speed_day1 + late_minutes) - (travel_time distance speed_day2) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_l1309_130902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_inequality_l1309_130972

theorem log_base_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) (hb1 : b < 1) :
  Real.log 3 / Real.log a > Real.log 3 / Real.log b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_inequality_l1309_130972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l1309_130903

/-- The circle C with center (a, a+2) and radius 1 -/
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a - 2)^2 = 1

/-- The point A(0,3) -/
def point_A : ℝ × ℝ := (0, 3)

/-- The origin O(0,0) -/
def origin : ℝ × ℝ := (0, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The existence of point M on circle C satisfying the condition -/
def exists_point_M (a : ℝ) : Prop :=
  ∃ M : ℝ × ℝ, circle_C a M.1 M.2 ∧ distance M point_A = 2 * distance M origin

theorem circle_intersection_range (a : ℝ) :
  exists_point_M a → -3 ≤ a ∧ a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l1309_130903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_is_one_fifth_l1309_130915

/-- Two congruent right circular cones with given properties containing a sphere -/
structure ConePair where
  base_radius : ℝ
  height : ℝ
  intersection_distance : ℝ

/-- The maximum squared radius of a sphere that can fit inside the cone pair -/
noncomputable def max_sphere_radius_squared (c : ConePair) : ℝ :=
  1 / 5

/-- Theorem stating that for the given cone configuration, the maximum squared radius of an inscribed sphere is 1/5 -/
theorem max_sphere_radius_squared_is_one_fifth (c : ConePair) 
  (h1 : c.base_radius = 5)
  (h2 : c.height = 10)
  (h3 : c.intersection_distance = 5) :
  max_sphere_radius_squared c = 1 / 5 := by
  sorry

#check max_sphere_radius_squared_is_one_fifth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_is_one_fifth_l1309_130915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_lower_bound_l1309_130923

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

theorem extremum_and_lower_bound (a : ℝ) (h : a > 0) :
  (∀ x ≥ 0, HasDerivAt (f a) ((a / (a + 1)) - 1/2) 1) ∧ a = 1 ∧
  (∀ x ≥ 0, f a x ≥ Real.log 2) ∧ a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_lower_bound_l1309_130923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_planes_l1309_130997

/-- The cosine of the angle between two planes --/
noncomputable def cos_angle_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  let n₁ := (a₁, b₁, c₁)
  let n₂ := (a₂, b₂, c₂)
  let dot_product := a₁ * a₂ + b₁ * b₂ + c₁ * c₂
  let magnitude₁ := Real.sqrt (a₁^2 + b₁^2 + c₁^2)
  let magnitude₂ := Real.sqrt (a₂^2 + b₂^2 + c₂^2)
  dot_product / (magnitude₁ * magnitude₂)

/-- Theorem: The cosine of the angle between the planes x - 2y + 3z - 4 = 0 and 4x + y - 2z + 6 = 0 is -4 / (17√2) --/
theorem cos_angle_specific_planes :
  cos_angle_between_planes 1 (-2) 3 (-4) 4 1 (-2) 6 = -4 / (17 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_planes_l1309_130997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_eight_l1309_130909

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  XY : ℝ
  YZ : ℝ
  isIsosceles : XY > 0 ∧ YZ > 0

/-- The length of a median in an isosceles triangle -/
noncomputable def medianLength (t : IsoscelesTriangle) : ℝ :=
  Real.sqrt (t.XY ^ 2 - (t.YZ / 2) ^ 2)

/-- Theorem: In the given isosceles triangle, the median length is 8 -/
theorem median_length_is_eight :
  let t : IsoscelesTriangle := ⟨10, 12, by norm_num⟩
  medianLength t = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_eight_l1309_130909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l1309_130934

/-- Given vectors a and b in ℝ³, if (a + l*b) is perpendicular to a, then l = -2 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ × ℝ) (l : ℝ) 
  (h1 : a = (0, 1, -1))
  (h2 : b = (1, 1, 0))
  (h3 : (a.1 + l * b.1, a.2.1 + l * b.2.1, a.2.2 + l * b.2.2) • a = 0) :
  l = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l1309_130934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1309_130920

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1/4
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 9/4

-- Define the points
def P (t : ℝ) : ℝ × ℝ := (t, t - 1)
def E : {p : ℝ × ℝ // circle1 p.1 p.2} := sorry
def F : {p : ℝ × ℝ // circle2 p.1 p.2} := sorry

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem max_distance_difference :
  ∃ t : ℝ, ∀ t' : ℝ,
    distance (P t) F.val - distance (P t) E.val ≥
    distance (P t') F.val - distance (P t') E.val ∧
    distance (P t) F.val - distance (P t) E.val = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1309_130920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_bags_probabilities_l1309_130910

/-- Represents the locations where items can be: source, intermediate, or destination -/
inductive Location where
  | Source
  | Intermediate
  | Destination

/-- Represents the state of the system at any point -/
structure SystemState where
  source : List ℕ
  intermediate : List ℕ
  destination : List ℕ

/-- Represents a move in the system -/
inductive Move where
  | SourceToIntermediate
  | IntermediateToDestination

/-- The probability of choosing to move from source to intermediate when there's a choice -/
def moveProb : ℚ := 1/2

/-- The number of items in the system -/
def numItems : ℕ := 4

/-- A function that simulates the random choice of moves -/
noncomputable def randomMove : SystemState → Move :=
  sorry

/-- A function that applies a move to a system state -/
def applyMove : SystemState → Move → SystemState :=
  sorry

/-- A function that checks if all items are in the destination -/
def isComplete : SystemState → Bool :=
  sorry

/-- A function that checks if the items in the destination are in reverse order compared to the initial order -/
def isReverseOrder : SystemState → Bool :=
  sorry

/-- A function that checks if the second item from the bottom in the source is at the bottom of the destination -/
def isSecondAtBottom : SystemState → Bool :=
  sorry

/-- A function that represents the probability of an event -/
noncomputable def Probability (event : SystemState → Bool) : ℚ :=
  sorry

theorem cement_bags_probabilities :
  let initialState : SystemState := ⟨[1,2,3,4], [], []⟩
  (Probability (fun (finalState : SystemState) => isComplete finalState ∧ isReverseOrder finalState) = 1/8) ∧
  (Probability (fun (finalState : SystemState) => isComplete finalState ∧ isSecondAtBottom finalState) = 1/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_bags_probabilities_l1309_130910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_inequality_l1309_130913

theorem sine_double_angle_inequality (α β γ : ℝ) 
  (acute_triangle : α + β + γ = Real.pi)
  (acute_angles : 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2)
  (angle_order : α < β ∧ β < γ) :
  Real.sin (2*α) > Real.sin (2*β) ∧ Real.sin (2*β) > Real.sin (2*γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_inequality_l1309_130913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covering_ways_eq_fib_l1309_130952

/-- The number of ways to cover a 2 × n grid with 1 × 2 dominoes -/
def coveringWays : ℕ → ℕ 
  | 0 => 1
  | 1 => 1
  | n + 2 => coveringWays (n + 1) + coveringWays n

/-- The Fibonacci sequence with F₁ = 1 and F₂ = 2 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem stating that the number of covering ways is equal to the Fibonacci number -/
theorem covering_ways_eq_fib : ∀ n : ℕ, coveringWays n = fib n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_covering_ways_eq_fib_l1309_130952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_money_ratio_l1309_130982

/-- Proves that the ratio of money spent to money made is 1:2 --/
theorem survey_money_ratio
  (hours_worked : ℕ)
  (hourly_rate : ℚ)
  (amount_left : ℚ)
  (h1 : hours_worked = 8)
  (h2 : hourly_rate = 18)
  (h3 : amount_left = 72) :
  let total_made := hours_worked * hourly_rate
  let amount_spent := total_made - amount_left
  amount_spent / total_made = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_money_ratio_l1309_130982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_integer_ratios_l1309_130940

/-- Given two arithmetic sequences {a_n} and {b_n} with sums A_n and B_n respectively -/
def A_n : ℕ+ → ℚ := sorry

def B_n : ℕ+ → ℚ := sorry

/-- The ratio of the sums of the first n terms -/
axiom sum_ratio (n : ℕ+) : A_n n / B_n n = (6 * n + 54) / (n + 5)

/-- The ratio of the nth terms of the sequences -/
def term_ratio (n : ℕ+) : ℚ := (6 * n + 24) / (n + 2)

/-- Main theorem: There are exactly 4 positive integers n for which term_ratio n is an integer -/
theorem exactly_four_integer_ratios :
  ∃ (S : Finset ℕ+), S.card = 4 ∧ (∀ n : ℕ+, n ∈ S ↔ ∃ (k : ℤ), term_ratio n = k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_integer_ratios_l1309_130940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1309_130927

-- Define the function type
def FunctionQ2R := ℚ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : FunctionQ2R) : Prop :=
  ∀ x y : ℚ, f x ^ 2 - f y ^ 2 = f (x + y) * f (x - y)

-- Define the solution form
def SolutionForm (f : FunctionQ2R) : Prop :=
  (f = λ _ => 0) ∨
  (∃ k a : ℝ, k > 0 ∧ (∀ x : ℚ, f x = k * (a ^ (x : ℝ) - a ^ (-(x : ℝ)))))

-- State the theorem
theorem functional_equation_solution :
  ∀ f : FunctionQ2R, SatisfiesEquation f → SolutionForm f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1309_130927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1309_130966

/-- The time taken by person A to complete a work -/
noncomputable def time_A : ℝ → ℝ := λ D => D

/-- The time taken by person B to complete the same work -/
noncomputable def time_B : ℝ → ℝ := λ D => D / 2

/-- The fraction of work completed by both A and B together in one day -/
noncomputable def work_per_day : ℝ → ℝ := λ D => 1 / time_A D + 1 / time_B D

theorem work_completion_time (D : ℝ) 
  (h1 : time_A D = D)
  (h2 : time_B D = D / 2)
  (h3 : work_per_day D = 0.3) :
  D = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1309_130966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l1309_130912

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  (x - 1)^2 / 7^2 - (y + 6)^2 / 3^2 = 1

-- Define the focus coordinates
noncomputable def focus : ℝ × ℝ := (1 + Real.sqrt 58, -6)

-- Theorem statement
theorem hyperbola_focus :
  ∀ x y : ℝ, hyperbola x y →
  (∀ x' y' : ℝ, hyperbola x' y' → x' ≤ focus.1) →
  focus.1 = 1 + Real.sqrt 58 ∧ focus.2 = -6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l1309_130912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_tunnel_time_l1309_130941

/-- Time for a train to cross a tunnel -/
theorem train_crossing_tunnel_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (tunnel_length : ℝ) 
  (h1 : train_length = 597) 
  (h2 : train_speed_kmh = 87) 
  (h3 : tunnel_length = 475) : 
  ∃ (time : ℝ), abs (time - 44.34) < 0.01 ∧ 
  time = (train_length + tunnel_length) / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_tunnel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_tunnel_time_l1309_130941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l1309_130971

open Real

-- Define the triangle and points
variable (A B C P Q : EuclideanSpace ℝ (Fin 2))

-- Define the equilateral property
def is_equilateral (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define the point positions
def P_on_CB (A B C P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • C + t • B

def Q_on_AB (A B C Q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • A + s • B

-- Define the distance relationships
def distance_relations (A B C P Q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A C = dist A P ∧ dist A P = 1.5 * dist P Q ∧ dist P Q = dist Q B

-- Define the angle measure in degrees
noncomputable def angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry -- Definition of angle measure in degrees

-- State the theorem
theorem angle_B_is_60_degrees
  (h_equilateral : is_equilateral A B C)
  (h_P_on_CB : P_on_CB A B C P)
  (h_Q_on_AB : Q_on_AB A B C Q)
  (h_distances : distance_relations A B C P Q) :
  angle_measure A B C = 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l1309_130971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_exponents_l1309_130992

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ k : ℚ, a x y = k * b x y ∨ b x y = k * a x y

/-- The first monomial 3xy^m -/
def monomial1 (x y m : ℕ) : ℚ := 3 * (x : ℚ) * (y^m : ℚ)

/-- The second monomial -x^ny -/
def monomial2 (x y n : ℕ) : ℚ := -1 * ((x^n : ℕ) : ℚ) * (y : ℚ)

theorem monomial_exponents (m n : ℕ) :
  like_terms (λ x y ↦ monomial1 x y m) (λ x y ↦ monomial2 x y n) → m - n = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_exponents_l1309_130992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1309_130950

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := (a*x - b) / (x^2 - 4)

-- Define the interval (-2, 2)
def interval : Set ℝ := { x | -2 < x ∧ x < 2 }

-- Main theorem
theorem main_theorem (a b : ℝ) : 
  (∀ x, x ∈ interval → f a b x = -f a b (-x)) → -- f is odd
  (f a b 1 = -1/3) → -- f(1) = -1/3
  (a = 1 ∧ b = 0) ∧ -- Part 1
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f 1 0 x > f 1 0 y) ∧ -- Part 2: f is monotonically decreasing
  (∀ t, t ∈ interval → (f 1 0 (t-1) + f 1 0 t < 0 ↔ 1/2 < t ∧ t < 2)) -- Part 3
  := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1309_130950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1309_130977

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality (x : ℝ) : f x + f (x - 1/2) > 1 ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1309_130977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_polynomial_implies_all_integer_l1309_130975

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem consecutive_integer_polynomial_implies_all_integer 
  (x y : ℝ) (h_distinct : x ≠ y) :
  (∃ k : ℕ, ∀ n : ℕ, n ∈ Finset.range 4 → 
    is_integer ((x^(n+k) - y^(n+k)) / (x - y))) →
  (∀ n : ℕ, is_integer ((x^n - y^n) / (x - y))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_polynomial_implies_all_integer_l1309_130975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_example_l1309_130983

/-- The edge length of a cube that can be covered with paper for a given cost. -/
noncomputable def cube_edge_length (paper_charge : ℝ) (area_per_kg : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := total_cost / paper_charge * area_per_kg
  Real.sqrt (surface_area / 6)

/-- Theorem: The edge length of a cube that can be covered with paper for 1800 Rs,
    given that paper costs 60 Rs per kg and 1 kg covers 20 sq. m., is 10 meters. -/
theorem cube_edge_length_example : cube_edge_length 60 20 1800 = 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cube_edge_length 60 20 1800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_example_l1309_130983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equal_consecutive_terms_l1309_130916

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) = floor (2 * (a (n + 1) : ℝ) / a n) + floor (2 * (a n : ℝ) / a (n + 1))

theorem sequence_equal_consecutive_terms (a : ℕ → ℕ) (h : sequence_property a) :
  ∃ m : ℕ, m ≥ 3 ∧ a m = a (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equal_consecutive_terms_l1309_130916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_right_triangle_l1309_130996

-- Define the circle ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points A and B
variable (A B : ℝ × ℝ)

-- Define the circle ω
variable (ω : Circle)

-- A and B are inside ω
def inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define circle S with diameter AB
noncomputable def circleS (A B : ℝ × ℝ) : Circle :=
  { center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2),
    radius := ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) / 2 }

-- Point C is on the circumference of ω
def onCircumference (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Angle ACB is 90 degrees (right angle)
def isRightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- The main theorem
theorem inscribed_right_triangle 
  (h1 : inside A ω) (h2 : inside B ω) : 
  ∃ C : ℝ × ℝ, onCircumference C ω ∧ 
               onCircumference C (circleS A B) ∧ 
               isRightAngle A B C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_right_triangle_l1309_130996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_six_l1309_130959

/-- σ(n) is the sum of all positive divisors of n -/
def sigma (n : ℕ) : ℕ := sorry

/-- p(n) is the largest prime divisor of n -/
def largest_prime_divisor (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem unique_solution_is_six :
  ∀ n : ℕ, n ≥ 2 → (sigma n / (largest_prime_divisor n - 1) = n ↔ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_six_l1309_130959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_alpha_beta_beta_value_l1309_130974

-- Problem 1
theorem cos_sum_alpha_beta (α β : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : Real.cos β = 4/5) 
  (h3 : π/2 < α ∧ α < π) 
  (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -1 := by sorry

-- Problem 2
theorem beta_value (α β : Real) 
  (h1 : Real.cos α = 1/7) 
  (h2 : Real.cos (α - β) = 13/14) 
  (h3 : 0 < β ∧ β < α ∧ α < π/2) : 
  β = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_alpha_beta_beta_value_l1309_130974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_statement1_true_l1309_130973

-- Define the statements as functions
def statement1 (a b : ℝ) : Prop := 2 * (a + b) = 2 * a + 2 * b
def statement2 (a b : ℝ) : Prop := (5 : ℝ)^(a + b) = (5 : ℝ)^a + (5 : ℝ)^b
def statement3 (x y : ℝ) : Prop := x > 0 ∧ y > 0 → Real.log (x + y) = Real.log x + Real.log y
def statement4 (a b : ℝ) : Prop := Real.sqrt (a^2 + b^2) = a + b
def statement5 (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2

-- Theorem stating that only statement1 is true for all real numbers
theorem only_statement1_true :
  (∀ a b : ℝ, statement1 a b) ∧
  (∃ a b : ℝ, ¬statement2 a b) ∧
  (∃ x y : ℝ, ¬statement3 x y) ∧
  (∃ a b : ℝ, ¬statement4 a b) ∧
  (∃ x y : ℝ, ¬statement5 x y) := by
  sorry

#check only_statement1_true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_statement1_true_l1309_130973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_S_l1309_130960

open Real

/-- The sum S as defined in the problem -/
noncomputable def S (a b c d : ℝ) : ℝ := a/(a+b+d) + b/(a+b+c) + c/(b+c+d) + d/(a+c+d)

/-- The theorem stating the range of S -/
theorem range_of_S :
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
  (∀ x : ℝ, 1 < x ∧ x < 2 → ∃ (a' b' c' d' : ℝ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ S a' b' c' d' = x) ∧
  (1 < S a b c d ∧ S a b c d < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_S_l1309_130960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_l1309_130938

def a : ℕ → ℕ
| 0 => 2
| n + 1 => 2^(a n)

theorem sequence_eventually_constant (n : ℕ) (hn : n ≥ 1) : 
  ∃ s : ℕ, ∀ k ≥ s, (a k) % n = (a (k + 1)) % n := by
  sorry

#check sequence_eventually_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_l1309_130938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_seven_l1309_130964

-- Define the sales volume function
noncomputable def sales_volume (a : ℝ) (x : ℝ) : ℝ :=
  if 6 < x ∧ x < 9 then 150 / (x - 6) + a * (x - 9) ^ 2
  else if 9 ≤ x ∧ x ≤ 15 then 177 / (x - 6) - x
  else 0

-- Define the profit function
noncomputable def profit (a : ℝ) (x : ℝ) : ℝ := (x - 6) * sales_volume a x

-- Theorem statement
theorem max_profit_at_seven (a : ℝ) :
  sales_volume a 8 = 80 →
  ∃ (max_profit : ℝ), 
    (∀ x, 6 < x ∧ x ≤ 15 → profit a x ≤ max_profit) ∧
    profit a 7 = max_profit ∧
    max_profit = 170 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_seven_l1309_130964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_on_AC_length_l1309_130956

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the lengths of the sides
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the median length
noncomputable def median_length (P Q R : ℝ × ℝ) : ℝ :=
  side_length P ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

-- Theorem statement
theorem median_on_AC_length (t : Triangle) : 
  side_length t.B t.C = 2 →
  side_length t.A t.B = 2 * Real.sqrt 3 →
  (∃ b : ℝ, side_length t.A t.C = b ∧ 
    ∃ x : ℝ, x^2 - 4*x + b = 0 ∧ 
    ∀ y : ℝ, y^2 - 4*y + b = 0 → y = x) →
  median_length t.B t.A t.C = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_on_AC_length_l1309_130956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hexagon_dots_l1309_130919

/-- The number of triangles in the nth hexagon -/
def triangles_in_hexagon (n : ℕ) : ℕ := 
  if n = 1 then 1 else 6 * (n - 1)

/-- The total number of triangles up to and including the nth hexagon -/
def total_triangles (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (fun i => triangles_in_hexagon (i + 1))

/-- The number of dots in the nth hexagon -/
def dots_in_hexagon (n : ℕ) : ℕ := 3 * total_triangles n

theorem fourth_hexagon_dots : 
  dots_in_hexagon 4 = 111 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hexagon_dots_l1309_130919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_integer_root_l1309_130951

theorem cubic_polynomial_integer_root 
  (p q : ℚ) 
  (h1 : (fun x : ℝ => x^3 + p*x + q) (3 - Real.sqrt 5) = 0)
  (h2 : ∃ (r : ℤ), (fun x : ℝ => x^3 + p*x + q) r = 0) :
  ∃ (r : ℤ), (fun x : ℝ => x^3 + p*x + q) r = 0 ∧ r = -6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_integer_root_l1309_130951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_octagon_l1309_130914

/-- The area between the inscribed and circumscribed circles of a regular octagon with side length 2 is equal to π. -/
theorem area_between_circles_octagon : 
  let side_length : ℝ := 2
  let apothem : ℝ := side_length / 2 * Real.tan (π / 8)⁻¹
  let circumradius : ℝ := side_length / 2 / Real.sin (π / 8)
  (circumradius^2 - apothem^2) * π = π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_octagon_l1309_130914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_min_values_sin_2alpha_value_l1309_130931

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (x + Real.pi / 2)

-- Statement 1: Smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  T = 2 * Real.pi := by
sorry

-- Statement 2: Maximum and minimum values
theorem max_min_values :
  (∃ x, f x = Real.sqrt 2) ∧
  (∃ y, f y = -Real.sqrt 2) ∧
  (∀ z, f z ≤ Real.sqrt 2 ∧ f z ≥ -Real.sqrt 2) := by
sorry

-- Statement 3: Value of sin(2α) when f(α) = 3/4
theorem sin_2alpha_value :
  ∀ α, f α = 3/4 → Real.sin (2 * α) = -7/16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_min_values_sin_2alpha_value_l1309_130931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leonardo_nap_duration_l1309_130928

theorem leonardo_nap_duration : 
  let minutes_per_hour : ℚ := 60
  let first_nap : ℚ := minutes_per_hour * (1 / 5)
  let second_nap : ℚ := minutes_per_hour * (1 / 4)
  let third_nap : ℚ := minutes_per_hour * (1 / 6)
  first_nap + second_nap + third_nap = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leonardo_nap_duration_l1309_130928
