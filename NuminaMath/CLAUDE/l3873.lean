import Mathlib

namespace NUMINAMATH_CALUDE_bus_schedule_theorem_l3873_387367

def is_valid_interval (T : ℚ) : Prop :=
  T < 30 ∧
  T > 0 ∧
  ∀ k : ℤ, ∀ t₀ : ℚ, 0 ≤ t₀ ∧ t₀ < T →
    (¬ (0 ≤ (t₀ + k * T) % 60 ∧ (t₀ + k * T) % 60 < 5)) ∧
    (¬ (38 ≤ (t₀ + k * T) % 60 ∧ (t₀ + k * T) % 60 < 43))

def valid_intervals : Set ℚ := {20, 15, 12, 10, 7.5, 5 + 5/11}

theorem bus_schedule_theorem :
  ∀ T : ℚ, is_valid_interval T ↔ T ∈ valid_intervals :=
sorry

end NUMINAMATH_CALUDE_bus_schedule_theorem_l3873_387367


namespace NUMINAMATH_CALUDE_election_votes_proof_l3873_387387

theorem election_votes_proof (total_votes : ℕ) (second_candidate_votes : ℕ) : 
  -- Given conditions
  total_votes = 27500 ∧ 
  (20000 : ℚ) / total_votes = 8011 / 11000 ∧
  total_votes = 2500 + second_candidate_votes + 20000 →
  -- Conclusion
  second_candidate_votes = 5000 := by
sorry


end NUMINAMATH_CALUDE_election_votes_proof_l3873_387387


namespace NUMINAMATH_CALUDE_journey_distance_l3873_387359

/-- Proves that a journey with given conditions has a total distance of 126 km -/
theorem journey_distance (total_time : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_time = 12 ∧
  speed1 = 21 ∧
  speed2 = 14 ∧
  speed3 = 6 →
  (1 / speed1 + 1 / speed2 + 1 / speed3) * (total_time / 3) = 126 := by
  sorry


end NUMINAMATH_CALUDE_journey_distance_l3873_387359


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3873_387314

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation 
  (l1 l2 l : Line2D) (P : Point2D) : 
  l1 = Line2D.mk 1 2 (-11) →
  l2 = Line2D.mk 2 1 (-10) →
  pointOnLine P l1 →
  pointOnLine P l2 →
  pointOnLine P l →
  perpendicularLines l l2 →
  l = Line2D.mk 1 (-2) 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3873_387314


namespace NUMINAMATH_CALUDE_class_average_problem_l3873_387377

theorem class_average_problem (first_group_percentage : Real) 
                               (second_group_percentage : Real) 
                               (first_group_average : Real) 
                               (second_group_average : Real) 
                               (overall_average : Real) :
  first_group_percentage = 0.25 →
  second_group_percentage = 0.50 →
  first_group_average = 0.80 →
  second_group_average = 0.65 →
  overall_average = 0.75 →
  let remainder_percentage := 1 - first_group_percentage - second_group_percentage
  let remainder_average := (overall_average - first_group_percentage * first_group_average - 
                            second_group_percentage * second_group_average) / remainder_percentage
  remainder_average = 0.90 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l3873_387377


namespace NUMINAMATH_CALUDE_garrison_provisions_duration_l3873_387360

/-- The number of days provisions last for a garrison with reinforcements --/
def provisions_duration (initial_men : ℕ) (reinforcement_men : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_men * days_before_reinforcement + (initial_men + reinforcement_men) * days_after_reinforcement) / initial_men

/-- Theorem stating that given the problem conditions, the provisions were supposed to last 54 days initially --/
theorem garrison_provisions_duration :
  provisions_duration 2000 1600 18 20 = 54 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provisions_duration_l3873_387360


namespace NUMINAMATH_CALUDE_athlete_arrangement_count_l3873_387354

theorem athlete_arrangement_count : ℕ := by
  -- Define the number of athletes and tracks
  let num_athletes : ℕ := 6
  let num_tracks : ℕ := 6

  -- Define the restrictions for athletes A and B
  let a_possible_tracks : ℕ := 4  -- A can't be on 1st or 2nd track
  let b_possible_tracks : ℕ := 2  -- B must be on 5th or 6th track

  -- Define the number of remaining athletes to be arranged
  let remaining_athletes : ℕ := num_athletes - 2  -- excluding A and B

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := b_possible_tracks * a_possible_tracks * (Nat.factorial remaining_athletes)

  -- Prove that the total number of arrangements is 144
  sorry

end NUMINAMATH_CALUDE_athlete_arrangement_count_l3873_387354


namespace NUMINAMATH_CALUDE_find_two_fake_coins_l3873_387385

/-- Represents the state of our coin testing process -/
structure CoinState where
  total : Nat
  fake : Nat
  deriving Repr

/-- Represents the result of a test -/
inductive TestResult
  | Signal
  | NoSignal
  deriving Repr

/-- A function that simulates a test -/
def test (coins : Nat) (state : CoinState) : TestResult := sorry

/-- A function that updates the state based on a test result -/
def updateState (coins : Nat) (state : CoinState) (result : TestResult) : CoinState := sorry

/-- A function that represents a single step in our testing strategy -/
def testStep (state : CoinState) : CoinState := sorry

/-- The main theorem stating that we can find two fake coins in five steps -/
theorem find_two_fake_coins 
  (initial_state : CoinState) 
  (h1 : initial_state.total = 49) 
  (h2 : initial_state.fake = 24) : 
  ∃ (final_state : CoinState), 
    (final_state.total = 2 ∧ final_state.fake = 2) ∧ 
    (∃ (s1 s2 s3 s4 : CoinState), 
      s1 = testStep initial_state ∧ 
      s2 = testStep s1 ∧ 
      s3 = testStep s2 ∧ 
      s4 = testStep s3 ∧ 
      final_state = testStep s4) :=
sorry

end NUMINAMATH_CALUDE_find_two_fake_coins_l3873_387385


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l3873_387327

def symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem product_of_symmetric_complex_numbers :
  ∀ z₁ z₂ : ℂ, 
    symmetric_about_imaginary_axis z₁ z₂ → 
    z₁ = 1 + 2*I → 
    z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l3873_387327


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3873_387340

theorem trigonometric_identity (α : ℝ) : 
  (Real.sin (7 * α) / Real.sin α) - 2 * (Real.cos (2 * α) + Real.cos (4 * α) + Real.cos (6 * α)) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3873_387340


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt35_l3873_387361

theorem closest_integer_to_sqrt35 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 35| ≤ |m - Real.sqrt 35| ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt35_l3873_387361


namespace NUMINAMATH_CALUDE_wall_completion_time_proof_l3873_387345

/-- Represents the wall dimensions -/
structure WallDimensions where
  thickness : ℝ
  length : ℝ
  height : ℝ

/-- Represents the working conditions -/
structure WorkingConditions where
  normal_pace : ℝ
  break_duration : ℝ
  break_count : ℕ
  faster_rate : ℝ
  faster_duration : ℝ
  min_work_between_breaks : ℝ

/-- Calculates the shortest possible time to complete the wall -/
def shortest_completion_time (wall : WallDimensions) (conditions : WorkingConditions) : ℝ :=
  sorry

theorem wall_completion_time_proof (wall : WallDimensions) (conditions : WorkingConditions) :
  wall.thickness = 0.25 ∧
  wall.length = 50 ∧
  wall.height = 2 ∧
  conditions.normal_pace = 26 ∧
  conditions.break_duration = 0.5 ∧
  conditions.break_count = 6 ∧
  conditions.faster_rate = 1.25 ∧
  conditions.faster_duration = 1 ∧
  conditions.min_work_between_breaks = 0.75 →
  shortest_completion_time wall conditions = 27.25 :=
by sorry

end NUMINAMATH_CALUDE_wall_completion_time_proof_l3873_387345


namespace NUMINAMATH_CALUDE_function_non_negative_range_l3873_387369

theorem function_non_negative_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = x^2 - 4*x + a) →
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) →
  a ∈ Set.Ici 3 := by
sorry

end NUMINAMATH_CALUDE_function_non_negative_range_l3873_387369


namespace NUMINAMATH_CALUDE_expressions_equality_l3873_387320

variable (a b c : ℝ)

theorem expressions_equality :
  (a - (b + c) = a - b - c) ∧
  (a + (-b - c) = a - b - c) ∧
  (a - (b - c) ≠ a - b - c) ∧
  ((-c) + (a - b) = a - b - c) := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l3873_387320


namespace NUMINAMATH_CALUDE_jakes_weight_l3873_387376

/-- Proves Jake's present weight given the conditions of the problem -/
theorem jakes_weight (jake kendra : ℕ) 
  (h1 : jake - 8 = 2 * kendra)
  (h2 : jake + kendra = 290) : 
  jake = 196 := by
  sorry

end NUMINAMATH_CALUDE_jakes_weight_l3873_387376


namespace NUMINAMATH_CALUDE_truck_distance_7_gallons_l3873_387338

/-- Represents the distance a truck can travel given an amount of diesel fuel. -/
def truckDistance (gallons : ℚ) : ℚ :=
  150 * (gallons / 5)

/-- Theorem: The truck travels 210 miles on 7 gallons of diesel. -/
theorem truck_distance_7_gallons : truckDistance 7 = 210 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_7_gallons_l3873_387338


namespace NUMINAMATH_CALUDE_sector_area_sexagesimal_l3873_387392

/-- The area of a sector with radius 4 and central angle 625/6000 of a full circle is 5π/3 -/
theorem sector_area_sexagesimal (π : ℝ) (h : π > 0) : 
  let r : ℝ := 4
  let angle_fraction : ℝ := 625 / 6000
  let sector_area := (1/2) * (angle_fraction * 2 * π) * r^2
  sector_area = 5 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sector_area_sexagesimal_l3873_387392


namespace NUMINAMATH_CALUDE_evolute_parabola_evolute_ellipse_l3873_387319

-- Part 1: Parabola
theorem evolute_parabola (x y X Y : ℝ) :
  x^2 = 2 * (1 - y) →
  27 * X^2 = -8 * Y^3 :=
sorry

-- Part 2: Ellipse
theorem evolute_ellipse (a b c t X Y : ℝ) :
  c^2 = a^2 - b^2 →
  X = -(c^2 / a) * (Real.cos t)^3 ∧
  Y = -(c^2 / b) * (Real.sin t)^3 :=
sorry

end NUMINAMATH_CALUDE_evolute_parabola_evolute_ellipse_l3873_387319


namespace NUMINAMATH_CALUDE_percentage_of_part_to_whole_l3873_387342

theorem percentage_of_part_to_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 25 →
  part = 70 ∧ whole = 280 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_part_to_whole_l3873_387342


namespace NUMINAMATH_CALUDE_system_solution_l3873_387362

theorem system_solution : ∃ (x y : ℝ), 
  (x^2 - 6 * Real.sqrt (3 - 2*x) - y + 11 = 0) ∧ 
  (y^2 - 4 * Real.sqrt (3*y - 2) + 4*x + 16 = 0) ∧
  (x = -3) ∧ (y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3873_387362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3873_387370

/-- Proves that in an arithmetic sequence with a₁ = 2 and a₃ = 8, the common difference is 3 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 2)  -- First term is 2
  (h3 : a 3 = 8)  -- Third term is 8
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3873_387370


namespace NUMINAMATH_CALUDE_symmetric_f_inequality_solution_l3873_387353

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - a|

-- State the theorem
theorem symmetric_f_inequality_solution (a : ℝ) :
  (∀ x : ℝ, f a x = f a (2 - x)) →
  {x : ℝ | f a (x^2 - 3) < f a (x - 1)} = {x : ℝ | -3 < x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_f_inequality_solution_l3873_387353


namespace NUMINAMATH_CALUDE_red_faces_cube_l3873_387396

theorem red_faces_cube (n : ℕ) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/4 ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_red_faces_cube_l3873_387396


namespace NUMINAMATH_CALUDE_largest_number_theorem_l3873_387397

theorem largest_number_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_products_eq : p * q + p * r + q * r = 1)
  (product_eq : p * q * r = 2) :
  max p (max q r) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_theorem_l3873_387397


namespace NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l3873_387339

theorem find_integers_with_sum_and_lcm : ∃ (a b : ℕ+), 
  (a + b : ℕ) = 3972 ∧ 
  Nat.lcm a b = 985928 ∧ 
  a = 1964 ∧ 
  b = 2008 := by
sorry

end NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l3873_387339


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3873_387332

theorem quadratic_inequality_empty_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3873_387332


namespace NUMINAMATH_CALUDE_square_equation_solution_l3873_387384

theorem square_equation_solution : 
  ∀ x y : ℕ+, x^2 = y^2 + 7*y + 6 ↔ x = 6 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3873_387384


namespace NUMINAMATH_CALUDE_square_root_equation_l3873_387324

theorem square_root_equation (x : ℝ) : 
  Real.sqrt (x - 3) = 5 → x = 28 := by sorry

end NUMINAMATH_CALUDE_square_root_equation_l3873_387324


namespace NUMINAMATH_CALUDE_square_cube_remainder_l3873_387315

theorem square_cube_remainder (a n : ℕ) 
  (h1 : a^2 % n = 8)
  (h2 : a^3 % n = 25)
  (h3 : n > 25) :
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_square_cube_remainder_l3873_387315


namespace NUMINAMATH_CALUDE_final_balance_calculation_l3873_387399

def football_club_balance (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) (players_bought : ℕ) (buying_price : ℝ) : ℝ :=
  initial_balance + players_sold * selling_price - players_bought * buying_price

theorem final_balance_calculation (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) (players_bought : ℕ) (buying_price : ℝ) :
  initial_balance = 100 ∧ players_sold = 2 ∧ selling_price = 10 ∧ players_bought = 4 ∧ buying_price = 15 →
  football_club_balance initial_balance players_sold selling_price players_bought buying_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_final_balance_calculation_l3873_387399


namespace NUMINAMATH_CALUDE_optimal_solution_l3873_387341

-- Define the normal distribution parameters
def μ : ℝ := 800
def σ : ℝ := 50

-- Define the probability p₀
def p₀ : ℝ := 0.9772

-- Define vehicle capacities and costs
def capacity_A : ℕ := 36
def capacity_B : ℕ := 60
def cost_A : ℕ := 1600
def cost_B : ℕ := 2400

-- Define the optimization problem
def optimal_fleet (a b : ℕ) : Prop :=
  -- Total vehicles constraint
  a + b ≤ 21 ∧
  -- Type B vehicles constraint
  b ≤ a + 7 ∧
  -- Probability constraint (simplified)
  (a * capacity_A + b * capacity_B : ℝ) ≥ μ + σ * 2 ∧
  -- Minimizes cost
  ∀ a' b' : ℕ,
    (a' * capacity_A + b' * capacity_B : ℝ) ≥ μ + σ * 2 →
    a' + b' ≤ 21 →
    b' ≤ a' + 7 →
    a * cost_A + b * cost_B ≤ a' * cost_A + b' * cost_B

-- Theorem statement
theorem optimal_solution :
  optimal_fleet 5 12 :=
sorry

end NUMINAMATH_CALUDE_optimal_solution_l3873_387341


namespace NUMINAMATH_CALUDE_x_fourth_coefficient_is_20th_term_l3873_387358

def binomial_sum (n : ℕ) : ℕ := (n.choose 4) + ((n + 1).choose 4) + ((n + 2).choose 4)

def arithmetic_sequence (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem x_fourth_coefficient_is_20th_term :
  ∃ n : ℕ, n = 5 ∧ 
  binomial_sum n = arithmetic_sequence (-2) 3 20 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_coefficient_is_20th_term_l3873_387358


namespace NUMINAMATH_CALUDE_empty_solution_set_inequality_l3873_387311

theorem empty_solution_set_inequality (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_inequality_l3873_387311


namespace NUMINAMATH_CALUDE_max_y_over_x_l3873_387331

theorem max_y_over_x (x y : ℝ) (h : x^2 + y^2 - 6*x - 6*y + 12 = 0) :
  ∃ (max : ℝ), max = 3 + 2 * Real.sqrt 2 ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 6*x' - 6*y' + 12 = 0 → y' / x' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3873_387331


namespace NUMINAMATH_CALUDE_interest_group_signups_l3873_387318

theorem interest_group_signups :
  let num_students : ℕ := 5
  let num_groups : ℕ := 3
  num_groups ^ num_students = 243 :=
by sorry

end NUMINAMATH_CALUDE_interest_group_signups_l3873_387318


namespace NUMINAMATH_CALUDE_distance_is_600_l3873_387351

/-- The distance between two points A and B, given specific train travel conditions. -/
def distance_between_points : ℝ :=
  let forward_speed : ℝ := 200
  let return_speed : ℝ := 100
  let time_difference : ℝ := 3
  600

/-- Theorem stating that the distance between points A and B is 600 km under given conditions. -/
theorem distance_is_600 (forward_speed return_speed time_difference : ℝ)
  (h1 : forward_speed = 200)
  (h2 : return_speed = 100)
  (h3 : time_difference = 3)
  : distance_between_points = 600 :=
by sorry

end NUMINAMATH_CALUDE_distance_is_600_l3873_387351


namespace NUMINAMATH_CALUDE_inequality_and_sum_theorem_l3873_387302

def f (x : ℝ) : ℝ := |3*x - 1|

theorem inequality_and_sum_theorem :
  (∀ x : ℝ, f x - f (2 - x) > x ↔ x ∈ Set.Ioo (6/5) 4) ∧
  (∀ a b : ℝ, a + b = 2 → f (a^2) + f (b^2) ≥ 4) := by sorry

end NUMINAMATH_CALUDE_inequality_and_sum_theorem_l3873_387302


namespace NUMINAMATH_CALUDE_at_most_one_solution_l3873_387365

-- Define a monotonic function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- Theorem statement
theorem at_most_one_solution
  (f : ℝ → ℝ) (c : ℝ) (h : Monotonic f) :
  ∃! x, f x = c ∨ ∀ x, f x ≠ c :=
sorry

end NUMINAMATH_CALUDE_at_most_one_solution_l3873_387365


namespace NUMINAMATH_CALUDE_least_pencils_l3873_387313

theorem least_pencils (p : ℕ) : p > 0 ∧ 
  p % 5 = 4 ∧ 
  p % 6 = 3 ∧ 
  p % 8 = 5 ∧ 
  (∀ q : ℕ, q > 0 ∧ q % 5 = 4 ∧ q % 6 = 3 ∧ q % 8 = 5 → p ≤ q) → 
  p = 69 := by
sorry

end NUMINAMATH_CALUDE_least_pencils_l3873_387313


namespace NUMINAMATH_CALUDE_binomial_identity_solutions_l3873_387390

theorem binomial_identity_solutions (n : ℕ) :
  ∀ x y : ℝ, (x + y)^n = x^n + y^n ↔
    (n = 1 ∧ True) ∨
    (∃ k : ℕ, n = 2 * k ∧ (x = 0 ∨ y = 0)) ∨
    (∃ k : ℕ, n = 2 * k + 1 ∧ (x = 0 ∨ y = 0 ∨ x = -y)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_identity_solutions_l3873_387390


namespace NUMINAMATH_CALUDE_gcd_one_powers_of_two_l3873_387344

def sequence_a : ℕ → ℕ
  | 0 => 3
  | (n + 1) => sequence_a n + n * (sequence_a n - 1)

theorem gcd_one_powers_of_two (m : ℕ) :
  (∀ n, Nat.gcd m (sequence_a n) = 1) ↔ ∃ t : ℕ, m = 2^t :=
sorry

end NUMINAMATH_CALUDE_gcd_one_powers_of_two_l3873_387344


namespace NUMINAMATH_CALUDE_mitchell_gum_chewing_l3873_387352

theorem mitchell_gum_chewing (packets : ℕ) (pieces_per_packet : ℕ) (unchewed_pieces : ℕ) :
  packets = 8 →
  pieces_per_packet = 7 →
  unchewed_pieces = 2 →
  packets * pieces_per_packet - unchewed_pieces = 54 :=
by sorry

end NUMINAMATH_CALUDE_mitchell_gum_chewing_l3873_387352


namespace NUMINAMATH_CALUDE_polygon_exterior_interior_sum_equal_l3873_387303

theorem polygon_exterior_interior_sum_equal (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_interior_sum_equal_l3873_387303


namespace NUMINAMATH_CALUDE_factorization_equality_l3873_387322

theorem factorization_equality (a : ℝ) : 
  (2 / 9) * a^2 - (4 / 3) * a + 2 = (2 / 9) * (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3873_387322


namespace NUMINAMATH_CALUDE_special_number_in_list_l3873_387386

theorem special_number_in_list (l : List ℝ) (n : ℝ) (h1 : l.length = 21) 
  (h2 : n ∈ l) (h3 : n = 4 * ((l.sum - n) / 20)) : 
  n = (1 / 6 : ℝ) * l.sum :=
by
  sorry

end NUMINAMATH_CALUDE_special_number_in_list_l3873_387386


namespace NUMINAMATH_CALUDE_unique_symmetric_matrix_condition_l3873_387355

/-- A symmetric 2x2 matrix with real entries -/
structure SymmetricMatrix2x2 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The trace of a symmetric 2x2 matrix -/
def trace (M : SymmetricMatrix2x2) : ℝ := M.x + M.z

/-- The determinant of a symmetric 2x2 matrix -/
def det (M : SymmetricMatrix2x2) : ℝ := M.x * M.z - M.y * M.y

/-- The main theorem -/
theorem unique_symmetric_matrix_condition (a b : ℝ) :
  (∃! M : SymmetricMatrix2x2, trace M = a ∧ det M = b) ↔ ∃ t : ℝ, a = 2 * t ∧ b = t ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_symmetric_matrix_condition_l3873_387355


namespace NUMINAMATH_CALUDE_total_sum_lent_is_2769_l3873_387346

/-- Calculates the total sum lent given the conditions of the problem -/
def totalSumLent (secondPart : ℕ) : ℕ :=
  let firstPart := (secondPart * 5) / 8
  firstPart + secondPart

/-- Proves that the total sum lent is 2769 given the problem conditions -/
theorem total_sum_lent_is_2769 :
  totalSumLent 1704 = 2769 := by
  sorry

#eval totalSumLent 1704

end NUMINAMATH_CALUDE_total_sum_lent_is_2769_l3873_387346


namespace NUMINAMATH_CALUDE_fan_weight_l3873_387350

/-- Given a box with fans, calculate the weight of a single fan. -/
theorem fan_weight (total_weight : ℝ) (num_fans : ℕ) (empty_box_weight : ℝ) 
  (h1 : total_weight = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight - empty_box_weight) / num_fans = 0.76 := by
  sorry

#check fan_weight

end NUMINAMATH_CALUDE_fan_weight_l3873_387350


namespace NUMINAMATH_CALUDE_houses_with_pool_count_l3873_387326

/-- Represents the number of houses in a development with various features -/
structure Development where
  total : ℕ
  with_garage : ℕ
  with_both : ℕ
  with_neither : ℕ

/-- The number of houses with an in-the-ground swimming pool in the development -/
def houses_with_pool (d : Development) : ℕ :=
  d.total - d.with_garage + d.with_both - d.with_neither

/-- Theorem stating that in the given development, 40 houses have an in-the-ground swimming pool -/
theorem houses_with_pool_count (d : Development) 
  (h1 : d.total = 65)
  (h2 : d.with_garage = 50)
  (h3 : d.with_both = 35)
  (h4 : d.with_neither = 10) : 
  houses_with_pool d = 40 := by
  sorry

end NUMINAMATH_CALUDE_houses_with_pool_count_l3873_387326


namespace NUMINAMATH_CALUDE_eight_brown_boxes_contain_480_sticks_l3873_387394

/-- Calculates the number of sticks of gum in a given number of brown boxes. -/
def sticksInBrownBoxes (numBoxes : ℕ) : ℕ :=
  let packsPerCarton : ℕ := 5
  let sticksPerPack : ℕ := 3
  let cartonsPerBox : ℕ := 4
  numBoxes * cartonsPerBox * packsPerCarton * sticksPerPack

/-- Theorem stating that 8 brown boxes contain 480 sticks of gum. -/
theorem eight_brown_boxes_contain_480_sticks :
  sticksInBrownBoxes 8 = 480 := by
  sorry


end NUMINAMATH_CALUDE_eight_brown_boxes_contain_480_sticks_l3873_387394


namespace NUMINAMATH_CALUDE_prob_at_least_7_heads_in_9_flips_is_correct_l3873_387379

/-- The probability of getting at least 7 heads in 9 flips of a fair coin -/
def prob_at_least_7_heads_in_9_flips : ℚ :=
  46 / 512

/-- Theorem stating that the probability of getting at least 7 heads in 9 flips of a fair coin is 46/512 -/
theorem prob_at_least_7_heads_in_9_flips_is_correct :
  prob_at_least_7_heads_in_9_flips = 46 / 512 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_7_heads_in_9_flips_is_correct_l3873_387379


namespace NUMINAMATH_CALUDE_independence_test_checks_categorical_variables_l3873_387371

/-- An independence test is a statistical method used to check relationships between variables. -/
def independence_test : Type := sorry

/-- Categorical variables are a type of variable in statistics. -/
def categorical_variable : Type := sorry

/-- The relationship between variables that an independence test checks. -/
def relationship_checked_by_independence_test : Type := sorry

/-- Theorem stating that independence tests check relationships between categorical variables. -/
theorem independence_test_checks_categorical_variables :
  relationship_checked_by_independence_test = categorical_variable := by sorry

end NUMINAMATH_CALUDE_independence_test_checks_categorical_variables_l3873_387371


namespace NUMINAMATH_CALUDE_glass_bowls_problem_l3873_387389

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 2393

/-- The buying price per bowl in rupees -/
def buying_price : ℚ := 18

/-- The selling price per bowl in rupees -/
def selling_price : ℚ := 20

/-- The number of bowls sold -/
def bowls_sold : ℕ := 104

/-- The percentage gain -/
def percentage_gain : ℚ := 0.4830917874396135

theorem glass_bowls_problem :
  let total_cost : ℚ := initial_bowls * buying_price
  let revenue : ℚ := bowls_sold * selling_price
  let gain : ℚ := revenue - (bowls_sold * buying_price)
  percentage_gain = (gain / total_cost) * 100 :=
by sorry

end NUMINAMATH_CALUDE_glass_bowls_problem_l3873_387389


namespace NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l3873_387301

theorem no_three_digit_perfect_square_sum : 
  ∀ (a b c : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 1 ≤ a → 
  ¬∃ (m : ℕ), m^2 = 111 * (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l3873_387301


namespace NUMINAMATH_CALUDE_family_of_lines_fixed_point_l3873_387316

/-- The point that all lines in the family kx+y+2k+1=0 pass through -/
theorem family_of_lines_fixed_point (k : ℝ) : 
  k * (-2) + (-1) + 2 * k + 1 = 0 := by
  sorry

#check family_of_lines_fixed_point

end NUMINAMATH_CALUDE_family_of_lines_fixed_point_l3873_387316


namespace NUMINAMATH_CALUDE_bank_teller_bills_count_l3873_387363

theorem bank_teller_bills_count :
  ∀ (num_20_dollar_bills : ℕ),
  (20 * 5 + num_20_dollar_bills * 20 = 780) →
  (20 + num_20_dollar_bills = 54) :=
by
  sorry

end NUMINAMATH_CALUDE_bank_teller_bills_count_l3873_387363


namespace NUMINAMATH_CALUDE_courtyard_paving_cost_l3873_387323

/-- Calculates the cost of paving a rectangular courtyard -/
theorem courtyard_paving_cost 
  (ratio_long : ℝ) 
  (ratio_short : ℝ) 
  (diagonal : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_ratio : ratio_long / ratio_short = 4 / 3) 
  (h_diagonal : diagonal = 45) 
  (h_cost : cost_per_sqm = 0.5) : 
  ⌊(ratio_long * ratio_short * (diagonal^2 / (ratio_long^2 + ratio_short^2)) * cost_per_sqm * 100) / 100⌋ = 486 := by
sorry

end NUMINAMATH_CALUDE_courtyard_paving_cost_l3873_387323


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3873_387388

theorem ellipse_hyperbola_foci (a b : ℝ) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = Real.sqrt 444 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3873_387388


namespace NUMINAMATH_CALUDE_sum_and_reverse_contradiction_l3873_387317

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem sum_and_reverse_contradiction :
  let sum := 137 + 276
  sum = 413 ∧ reverse_digits sum ≠ 534 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reverse_contradiction_l3873_387317


namespace NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_specific_l3873_387382

/-- The distance at which a dog catches a fox given initial conditions -/
theorem dog_catches_fox (initial_distance : ℝ) (dog_leap : ℝ) (fox_leap : ℝ) 
  (dog_leaps_per_unit : ℕ) (fox_leaps_per_unit : ℕ) : ℝ :=
  let dog_distance_per_unit := dog_leap * dog_leaps_per_unit
  let fox_distance_per_unit := fox_leap * fox_leaps_per_unit
  let relative_distance_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let time_units_to_catch := initial_distance / relative_distance_per_unit
  time_units_to_catch * dog_distance_per_unit

/-- The specific case of the dog catching the fox problem -/
theorem dog_catches_fox_specific : 
  dog_catches_fox 30 2 1 2 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_specific_l3873_387382


namespace NUMINAMATH_CALUDE_sum_and_count_equals_431_l3873_387328

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_equals_431 : 
  sum_of_integers 10 30 + count_even_integers 10 30 = 431 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_equals_431_l3873_387328


namespace NUMINAMATH_CALUDE_line_l_equation_and_symmetric_points_l3873_387373

/-- Parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Line l intersecting the parabola -/
def Line_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (m b : ℝ), p.2 = m * p.1 + b}

/-- Point P that bisects segment AB -/
def P : ℝ × ℝ := (2, 2)

/-- A and B are points where line l intersects the parabola -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

theorem line_l_equation_and_symmetric_points :
  (∀ p ∈ Line_l, 2 * p.1 - p.2 - 2 = 0) ∧
  (∃ (C D : ℝ × ℝ), C ∈ Parabola ∧ D ∈ Parabola ∧
    (∀ p ∈ Line_l, (C.1 + D.1) * p.2 = (C.2 + D.2) * p.1 + C.1 * D.2 - C.2 * D.1) ∧
    (∀ p ∈ {p : ℝ × ℝ | p.1 + 2 * p.2 - 19 = 0}, p = C ∨ p = D)) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_and_symmetric_points_l3873_387373


namespace NUMINAMATH_CALUDE_solve_nested_equation_l3873_387304

theorem solve_nested_equation : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 58 ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_nested_equation_l3873_387304


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3873_387325

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x = 1 → x^2 ≠ 1) ∧ 
  ¬(∀ x, x^2 ≠ 1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3873_387325


namespace NUMINAMATH_CALUDE_square_area_comparison_l3873_387378

theorem square_area_comparison (a b : ℝ) (h : b = 4 * a) :
  b ^ 2 = 16 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_comparison_l3873_387378


namespace NUMINAMATH_CALUDE_existence_of_p_n_l3873_387337

/-- The polynomial a(x, y) -/
def a (x y : ℝ) : ℝ := x^2 * y + x * y^2

/-- The polynomial b(x, y) -/
def b (x y : ℝ) : ℝ := x^2 + x * y + y^2

/-- The existence of polynomial p_n for all natural numbers n -/
theorem existence_of_p_n :
  ∀ n : ℕ, ∃ p_n : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → ℝ,
    ∀ x y : ℝ, p_n a b = (x + y)^n + (-1)^n * (x^n + y^n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_p_n_l3873_387337


namespace NUMINAMATH_CALUDE_equal_savings_l3873_387300

theorem equal_savings (your_initial : ℕ) (friend_initial : ℕ) (your_rate : ℕ) (friend_rate : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  friend_initial = 210 →
  your_rate = 7 →
  friend_rate = 5 →
  weeks = 25 →
  your_initial + your_rate * weeks = friend_initial + friend_rate * weeks :=
by sorry

end NUMINAMATH_CALUDE_equal_savings_l3873_387300


namespace NUMINAMATH_CALUDE_hannah_dolls_multiplier_l3873_387309

theorem hannah_dolls_multiplier (x : ℝ) : 
  x > 0 → -- Hannah has a positive number of times as many dolls
  8 * x + 8 = 48 → -- Total dolls equation
  x = 5 := by sorry

end NUMINAMATH_CALUDE_hannah_dolls_multiplier_l3873_387309


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_two_l3873_387380

def A : Set ℝ := {x | x < 2}
def B (m : ℝ) : Set ℝ := {x | x < m}

theorem subset_implies_m_leq_two (m : ℝ) : B m ⊆ A → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_two_l3873_387380


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3873_387321

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - 5*x + b > 0 ↔ x < -1/3 ∨ x > 1/2) →
  (∀ x : ℝ, bx^2 - 5*x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3873_387321


namespace NUMINAMATH_CALUDE_max_diagonal_area_ratio_l3873_387343

/-- A triangle with an inscribed rectangle -/
structure TriangleWithInscribedRectangle where
  /-- The area of the triangle -/
  area : ℝ
  /-- The length of the shortest diagonal of any inscribed rectangle -/
  shortest_diagonal : ℝ
  /-- The area is positive -/
  area_pos : 0 < area

/-- The theorem statement -/
theorem max_diagonal_area_ratio (T : TriangleWithInscribedRectangle) :
  T.shortest_diagonal ^ 2 / T.area ≤ 4 * Real.sqrt 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_max_diagonal_area_ratio_l3873_387343


namespace NUMINAMATH_CALUDE_factor_count_l3873_387312

def has_factors (m : ℕ) (k : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = k

theorem factor_count (n : ℕ) :
  n > 0 →
  has_factors (2 * n) 28 →
  has_factors (3 * n) 30 →
  has_factors (6 * n) 35 := by
  sorry

end NUMINAMATH_CALUDE_factor_count_l3873_387312


namespace NUMINAMATH_CALUDE_half_power_decreasing_l3873_387307

theorem half_power_decreasing (a b : ℝ) (h : a > b) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_half_power_decreasing_l3873_387307


namespace NUMINAMATH_CALUDE_operation_2011_result_l3873_387364

def operation_result (n : ℕ) : ℕ :=
  match n % 3 with
  | 1 => 133
  | 2 => 55
  | 0 => 250
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

theorem operation_2011_result :
  operation_result 2011 = 133 :=
sorry

end NUMINAMATH_CALUDE_operation_2011_result_l3873_387364


namespace NUMINAMATH_CALUDE_no_valid_n_l3873_387368

theorem no_valid_n : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (1000 ≤ n / 4) ∧ (n / 4 ≤ 9999) ∧ 
  (1000 ≤ 4 * n) ∧ (4 * n ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l3873_387368


namespace NUMINAMATH_CALUDE_min_value_a_l3873_387335

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 25) : 
  a ≥ 16 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1/x + (16 - ε)/y) < 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3873_387335


namespace NUMINAMATH_CALUDE_game_terminates_l3873_387333

/-- Represents the state of knowledge for each player -/
structure PlayerKnowledge where
  lower : Nat
  upper : Nat

/-- Represents the game state -/
structure GameState where
  player1 : PlayerKnowledge
  player2 : PlayerKnowledge
  turn : Nat

/-- Updates the game state based on a negative response -/
def updateGameState (state : GameState) : GameState :=
  sorry

/-- Checks if a player knows the other's number -/
def knowsNumber (knowledge : PlayerKnowledge) : Bool :=
  sorry

/-- Simulates the game for a given initial state -/
def playGame (initialState : GameState) : Nat :=
  sorry

/-- Theorem stating that the game will terminate -/
theorem game_terminates (n : Nat) :
  ∃ (k : Nat), ∀ (m : Nat),
    let initialState : GameState := {
      player1 := { lower := 1, upper := n + 1 },
      player2 := { lower := 1, upper := n + 1 },
      turn := 0
    }
    playGame initialState ≤ k :=
  sorry

end NUMINAMATH_CALUDE_game_terminates_l3873_387333


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_sum_of_digits_l3873_387395

/-- Function to create a number with n ones -/
def ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem: For all natural numbers n, the number formed by 3^n ones is divisible by the sum of its digits -/
theorem infinitely_many_divisible_by_sum_of_digits (n : ℕ) :
  ∃ (k : ℕ), k > 0 ∧ (ones (3^n) % sumOfDigits (ones (3^n)) = 0) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_sum_of_digits_l3873_387395


namespace NUMINAMATH_CALUDE_incorrect_operation_l3873_387330

theorem incorrect_operation (a b : ℝ) : (-a^3)^2 * (-b^2)^3 = -a^6 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_l3873_387330


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3873_387398

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = x + m ∧ x^2 / 2 + y^2 = 1 → 
    ∃! p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1 ∧ p.2 = p.1 + m) ↔ 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3873_387398


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_l3873_387308

theorem complex_fourth_quadrant (a : ℝ) : 
  let z₁ : ℂ := 3 - a * Complex.I
  let z₂ : ℂ := 1 + 2 * Complex.I
  (z₁ / z₂).re > 0 ∧ (z₁ / z₂).im < 0 ↔ -6 < a ∧ a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_l3873_387308


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3873_387334

theorem ratio_x_to_y (x y : ℚ) (h : (8 * x + 5 * y) / (10 * x + 3 * y) = 4 / 7) :
  x / y = -23 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3873_387334


namespace NUMINAMATH_CALUDE_athlete_seating_arrangements_l3873_387393

def number_of_arrangements (n : ℕ) (team_sizes : List ℕ) : ℕ :=
  (Nat.factorial n) * (team_sizes.map Nat.factorial).prod

theorem athlete_seating_arrangements :
  number_of_arrangements 4 [4, 3, 2, 3] = 20736 := by
  sorry

end NUMINAMATH_CALUDE_athlete_seating_arrangements_l3873_387393


namespace NUMINAMATH_CALUDE_cosine_value_in_special_triangle_l3873_387374

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem cosine_value_in_special_triangle (t : Triangle) 
  (h1 : t.c = 2 * t.a)  -- Given condition: c = 2a
  (h2 : Real.sin t.B ^ 2 = Real.sin t.A * Real.sin t.C)  -- Given condition: sin²B = sin A * sin C
  : Real.cos t.B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_in_special_triangle_l3873_387374


namespace NUMINAMATH_CALUDE_office_employee_count_l3873_387336

/-- Proves that the total number of employees in an office is 100 given specific salary and employee count conditions. -/
theorem office_employee_count :
  let avg_salary : ℚ := 720
  let officer_salary : ℚ := 1320
  let manager_salary : ℚ := 840
  let worker_salary : ℚ := 600
  let officer_count : ℕ := 10
  let manager_count : ℕ := 20
  ∃ (worker_count : ℕ),
    (officer_count : ℚ) * officer_salary + (manager_count : ℚ) * manager_salary + (worker_count : ℚ) * worker_salary =
    ((officer_count + manager_count + worker_count) : ℚ) * avg_salary ∧
    officer_count + manager_count + worker_count = 100 :=
by sorry

end NUMINAMATH_CALUDE_office_employee_count_l3873_387336


namespace NUMINAMATH_CALUDE_next_simultaneous_ring_l3873_387356

def library_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def hospital_interval : ℕ := 30

theorem next_simultaneous_ring : 
  Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval = 360 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_ring_l3873_387356


namespace NUMINAMATH_CALUDE_max_difference_three_digit_mean_505_l3873_387348

/-- The maximum difference between two three-digit integers with a mean of 505 -/
theorem max_difference_three_digit_mean_505 :
  ∃ (x y : ℕ),
    100 ≤ x ∧ x < 1000 ∧
    100 ≤ y ∧ y < 1000 ∧
    (x + y) / 2 = 505 ∧
    ∀ (a b : ℕ),
      100 ≤ a ∧ a < 1000 ∧
      100 ≤ b ∧ b < 1000 ∧
      (a + b) / 2 = 505 →
      x - y ≥ a - b ∧
    x - y = 810 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_three_digit_mean_505_l3873_387348


namespace NUMINAMATH_CALUDE_diamond_computation_l3873_387357

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.five
  | Element.one, Element.five => Element.four
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.five
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.three
  | Element.two, Element.five => Element.two
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.two
  | Element.three, Element.four => Element.one
  | Element.three, Element.five => Element.five
  | Element.four, Element.one => Element.five
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.four
  | Element.five, Element.two => Element.three
  | Element.five, Element.three => Element.five
  | Element.five, Element.four => Element.two
  | Element.five, Element.five => Element.one

theorem diamond_computation :
  diamond (diamond Element.four Element.five) (diamond Element.one Element.three) = Element.two :=
by sorry

end NUMINAMATH_CALUDE_diamond_computation_l3873_387357


namespace NUMINAMATH_CALUDE_correct_assignment_count_l3873_387306

/-- The number of ways to assign volunteers to pavilions. -/
def assign_volunteers (total_volunteers : ℕ) (female_volunteers : ℕ) (male_volunteers : ℕ) (pavilions : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating the correct number of ways to assign volunteers. -/
theorem correct_assignment_count :
  assign_volunteers 8 3 5 3 = 180 :=
sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l3873_387306


namespace NUMINAMATH_CALUDE_f_has_min_value_neg_ten_l3873_387383

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x - 1

-- Theorem statement
theorem f_has_min_value_neg_ten :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -10 :=
by sorry

end NUMINAMATH_CALUDE_f_has_min_value_neg_ten_l3873_387383


namespace NUMINAMATH_CALUDE_distance_traveled_l3873_387329

theorem distance_traveled (speed1 speed2 distance_diff : ℝ) (h1 : speed1 = 10)
  (h2 : speed2 = 20) (h3 : distance_diff = 40) :
  let time := distance_diff / (speed2 - speed1)
  let actual_distance := speed1 * time
  actual_distance = 40 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l3873_387329


namespace NUMINAMATH_CALUDE_zacks_countries_l3873_387366

theorem zacks_countries (alex george joseph patrick zack : ℚ) 
  (h_alex : alex = 30)
  (h_george : george = 3/5 * alex)
  (h_joseph : joseph = 1/3 * george)
  (h_patrick : patrick = 4/3 * joseph)
  (h_zack : zack = 5/2 * patrick) :
  zack = 20 := by
sorry

end NUMINAMATH_CALUDE_zacks_countries_l3873_387366


namespace NUMINAMATH_CALUDE_trading_card_boxes_l3873_387372

/-- Calculates the number of fully filled boxes for a card type -/
def fullBoxes (cards : ℕ) (capacity : ℕ) : ℕ := cards / capacity

/-- Represents the trading card sorting problem -/
theorem trading_card_boxes 
  (total_cards : ℕ) 
  (magic_cards : ℕ) 
  (rare_cards : ℕ) 
  (common_cards : ℕ) 
  (magic_capacity : ℕ) 
  (rare_capacity : ℕ) 
  (common_capacity : ℕ) 
  (h1 : total_cards = 94)
  (h2 : magic_cards = 33)
  (h3 : rare_cards = 28)
  (h4 : common_cards = 33)
  (h5 : magic_capacity = 8)
  (h6 : rare_capacity = 10)
  (h7 : common_capacity = 12)
  (h8 : total_cards = magic_cards + rare_cards + common_cards) :
  fullBoxes magic_cards magic_capacity + 
  fullBoxes rare_cards rare_capacity + 
  fullBoxes common_cards common_capacity = 8 := by
sorry

end NUMINAMATH_CALUDE_trading_card_boxes_l3873_387372


namespace NUMINAMATH_CALUDE_seeking_cause_is_sufficient_condition_l3873_387347

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  inequality : Prop
  condition : Prop

/-- Definition of a sufficient condition -/
def is_sufficient_condition (am : AnalysisMethod) : Prop :=
  am.condition → am.inequality

/-- "Seeking the cause from the result" in the analysis method -/
def seeking_cause_from_result (am : AnalysisMethod) : Prop :=
  ∃ (condition : Prop), is_sufficient_condition { inequality := am.inequality, condition := condition }

theorem seeking_cause_is_sufficient_condition (am : AnalysisMethod) :
  seeking_cause_from_result am ↔ ∃ (condition : Prop), is_sufficient_condition { inequality := am.inequality, condition := condition } :=
sorry

end NUMINAMATH_CALUDE_seeking_cause_is_sufficient_condition_l3873_387347


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3873_387375

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 90 →
  a = 18 →
  a^2 + b^2 = c^2 →
  a + b + c = 28 + 2 * Real.sqrt 106 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3873_387375


namespace NUMINAMATH_CALUDE_beam_max_strength_l3873_387305

/-- The strength of a rectangular beam cut from a circular log is maximized when its width is 2R/√3 and its height is 2R√2/√3, where R is the radius of the log. -/
theorem beam_max_strength (R : ℝ) (R_pos : R > 0) :
  let strength (x y : ℝ) := x * y^2
  let constraint (x y : ℝ) := x^2 + y^2 = 4 * R^2
  ∃ (k : ℝ), k > 0 ∧
    ∀ (x y : ℝ), constraint x y →
      strength x y ≤ k * strength (2*R/Real.sqrt 3) (2*R*Real.sqrt 2/Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_beam_max_strength_l3873_387305


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3873_387391

/-- The number of lines in a 4x4 grid (both horizontal and vertical) -/
def gridLines : ℕ := 5

/-- The number of lines needed to form a rectangle (both horizontal and vertical) -/
def linesNeeded : ℕ := 2

/-- The number of ways to choose horizontal lines for a rectangle -/
def horizontalChoices : ℕ := Nat.choose gridLines linesNeeded

/-- The number of ways to choose vertical lines for a rectangle -/
def verticalChoices : ℕ := Nat.choose gridLines linesNeeded

/-- Theorem: The number of rectangles on a 4x4 grid is 100 -/
theorem rectangles_on_4x4_grid : horizontalChoices * verticalChoices = 100 := by
  sorry


end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3873_387391


namespace NUMINAMATH_CALUDE_sum_equation_solution_l3873_387310

theorem sum_equation_solution (x : ℤ) : 
  (1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_solution_l3873_387310


namespace NUMINAMATH_CALUDE_kevin_ran_17_miles_l3873_387381

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Kevin's total running distance -/
def kevin_total_distance : ℝ :=
  let segment1 := distance 10 0.5
  let segment2 := distance 20 0.5
  let segment3 := distance 8 0.25
  segment1 + segment2 + segment3

/-- Theorem stating that Kevin's total running distance is 17 miles -/
theorem kevin_ran_17_miles : kevin_total_distance = 17 := by
  sorry

end NUMINAMATH_CALUDE_kevin_ran_17_miles_l3873_387381


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3873_387349

theorem polynomial_divisibility (a b c d : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ ka kb kc kd : ℤ, a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3873_387349
