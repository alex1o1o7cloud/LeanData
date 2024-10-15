import Mathlib

namespace NUMINAMATH_CALUDE_joint_probability_female_literate_l3288_328812

/-- Represents the total number of employees -/
def total_employees : ℕ := 1400

/-- Represents the proportion of female employees -/
def female_ratio : ℚ := 3/5

/-- Represents the proportion of male employees -/
def male_ratio : ℚ := 2/5

/-- Represents the proportion of engineers in the workforce -/
def engineer_ratio : ℚ := 7/20

/-- Represents the proportion of managers in the workforce -/
def manager_ratio : ℚ := 1/4

/-- Represents the proportion of support staff in the workforce -/
def support_ratio : ℚ := 2/5

/-- Represents the overall computer literacy rate -/
def overall_literacy_rate : ℚ := 31/50

/-- Represents the computer literacy rate for male engineers -/
def male_engineer_literacy : ℚ := 4/5

/-- Represents the computer literacy rate for female engineers -/
def female_engineer_literacy : ℚ := 3/4

/-- Represents the computer literacy rate for male managers -/
def male_manager_literacy : ℚ := 11/20

/-- Represents the computer literacy rate for female managers -/
def female_manager_literacy : ℚ := 3/5

/-- Represents the computer literacy rate for male support staff -/
def male_support_literacy : ℚ := 2/5

/-- Represents the computer literacy rate for female support staff -/
def female_support_literacy : ℚ := 1/2

/-- Theorem stating that the joint probability of a randomly selected employee being both female and computer literate is equal to 36.75% -/
theorem joint_probability_female_literate : 
  (female_ratio * engineer_ratio * female_engineer_literacy + 
   female_ratio * manager_ratio * female_manager_literacy + 
   female_ratio * support_ratio * female_support_literacy) = 147/400 := by
  sorry

end NUMINAMATH_CALUDE_joint_probability_female_literate_l3288_328812


namespace NUMINAMATH_CALUDE_students_like_both_sports_l3288_328862

/-- The number of students who like basketball -/
def B : ℕ := 10

/-- The number of students who like cricket -/
def C : ℕ := 8

/-- The number of students who like either basketball or cricket or both -/
def B_union_C : ℕ := 14

/-- The number of students who like both basketball and cricket -/
def B_intersect_C : ℕ := B + C - B_union_C

theorem students_like_both_sports : B_intersect_C = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_like_both_sports_l3288_328862


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l3288_328835

/-- Represents the revenue from full-price tickets in a charity event -/
def revenue_full_price (full_price : ℝ) (num_full_price : ℝ) : ℝ :=
  full_price * num_full_price

/-- Represents the revenue from discounted tickets in a charity event -/
def revenue_discounted (full_price : ℝ) (num_discounted : ℝ) : ℝ :=
  0.75 * full_price * num_discounted

/-- Theorem stating that the revenue from full-price tickets can be determined -/
theorem charity_ticket_revenue 
  (full_price : ℝ) 
  (num_full_price num_discounted : ℝ) 
  (h1 : num_full_price + num_discounted = 150)
  (h2 : revenue_full_price full_price num_full_price + 
        revenue_discounted full_price num_discounted = 2250)
  : ∃ (r : ℝ), revenue_full_price full_price num_full_price = r :=
by
  sorry


end NUMINAMATH_CALUDE_charity_ticket_revenue_l3288_328835


namespace NUMINAMATH_CALUDE_domain_of_f_l3288_328873

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 1)
def domain_f_squared (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3}

-- Theorem statement
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_squared f → f (x^2 - 1) ≠ 0) →
  (∀ y, f y ≠ 0 → -1 ≤ y ∧ y ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l3288_328873


namespace NUMINAMATH_CALUDE_average_of_sequence_l3288_328886

theorem average_of_sequence (z : ℝ) : 
  let sequence := [0, 3*z, 6*z, 12*z, 24*z]
  (sequence.sum / sequence.length : ℝ) = 9*z := by
sorry

end NUMINAMATH_CALUDE_average_of_sequence_l3288_328886


namespace NUMINAMATH_CALUDE_brown_ball_weight_calculation_l3288_328828

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 6

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := 9.12

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := total_weight - blue_ball_weight

theorem brown_ball_weight_calculation :
  brown_ball_weight = 3.12 := by sorry

end NUMINAMATH_CALUDE_brown_ball_weight_calculation_l3288_328828


namespace NUMINAMATH_CALUDE_football_players_l3288_328836

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 410)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 140) :
  total - cricket + both - neither = 375 := by
  sorry

end NUMINAMATH_CALUDE_football_players_l3288_328836


namespace NUMINAMATH_CALUDE_polar_sin_is_circle_l3288_328895

-- Define the polar coordinate equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = Real.sin θ

-- Define the transformation from polar to Cartesian coordinates
def to_cartesian (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define a circle in Cartesian coordinates
def is_circle (x y : ℝ) (h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_sin_is_circle :
  ∃ (h k r : ℝ), ∀ (x y ρ θ : ℝ),
    polar_equation ρ θ → to_cartesian x y ρ θ →
    is_circle x y h k r :=
sorry

end NUMINAMATH_CALUDE_polar_sin_is_circle_l3288_328895


namespace NUMINAMATH_CALUDE_min_contribution_problem_l3288_328800

/-- Proves that given 12 people contributing a total of $20.00, with a maximum individual contribution of $9.00, the minimum amount each person must have contributed is $1.00. -/
theorem min_contribution_problem (n : ℕ) (total : ℚ) (max_contrib : ℚ) (h1 : n = 12) (h2 : total = 20) (h3 : max_contrib = 9) : 
  ∃ (min_contrib : ℚ), 
    min_contrib = 1 ∧ 
    n * min_contrib ≤ total ∧ 
    ∀ (individual_contrib : ℚ), individual_contrib ≤ max_contrib → 
      (n - 1) * min_contrib + individual_contrib ≤ total :=
by sorry

end NUMINAMATH_CALUDE_min_contribution_problem_l3288_328800


namespace NUMINAMATH_CALUDE_min_value_of_f_l3288_328823

/-- The function f(x) = (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 
  ((2*x + a) * Real.exp (x - 1)) + ((x^2 + a*x - 1) * Real.exp (x - 1))

theorem min_value_of_f (a : ℝ) :
  (f_deriv a (-2) = 0) →  -- x = -2 is an extremum point
  (∃ x, f a x = -1) ∧ (∀ y, f a y ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3288_328823


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3288_328888

/-- An arithmetic sequence with sum of first n terms Sₙ -/
structure ArithmeticSequence where
  S : ℕ → ℝ
  is_arithmetic : ∃ (a₁ d : ℝ), ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Properties of a specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 10 = 0 ∧ seq.S 15 = 25

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  (∃ a₅ : ℝ, a₅ = -1/3 ∧ ∀ n : ℕ, seq.S n = n * a₅ + (n * (n - 1) / 2) * (2/3)) ∧ 
  (∀ n : ℕ, seq.S n ≥ seq.S 5) ∧
  (∃ min_value : ℝ, min_value = -49 ∧ ∀ n : ℕ, n * seq.S n ≥ min_value) ∧
  (¬∃ max_value : ℝ, ∀ n : ℕ, seq.S n / n ≤ max_value) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3288_328888


namespace NUMINAMATH_CALUDE_toy_problem_solution_l3288_328811

/-- Represents the problem of Mia and her mom putting toys in a box -/
def ToyProblem (totalToys : ℕ) (putIn : ℕ) (takeOut : ℕ) (cycleTime : ℚ) : Prop :=
  let netIncrease := putIn - takeOut
  let cycles := (totalToys - 1) / netIncrease + 1
  cycles * cycleTime / 60 = 12.5

/-- The theorem statement for the toy problem -/
theorem toy_problem_solution :
  ToyProblem 50 5 3 (30 / 60) :=
sorry

end NUMINAMATH_CALUDE_toy_problem_solution_l3288_328811


namespace NUMINAMATH_CALUDE_all_lamps_on_iff_even_l3288_328896

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a grid of lamps -/
def LampGrid (n : ℕ) := Fin n → Fin n → LampState

/-- Function to toggle a lamp state -/
def toggleLamp : LampState → LampState
| LampState.On => LampState.Off
| LampState.Off => LampState.On

/-- Function to press a switch at position (i, j) -/
def pressSwitch (grid : LampGrid n) (i j : Fin n) : LampGrid n :=
  fun x y => if x = i ∨ y = j then toggleLamp (grid x y) else grid x y

/-- Predicate to check if all lamps are on -/
def allLampsOn (grid : LampGrid n) : Prop :=
  ∀ i j, grid i j = LampState.On

/-- Main theorem: It's possible to achieve all lamps on iff n is even -/
theorem all_lamps_on_iff_even (n : ℕ) :
  (∀ (initialGrid : LampGrid n), ∃ (switches : List (Fin n × Fin n)),
    allLampsOn (switches.foldl (fun g (i, j) => pressSwitch g i j) initialGrid)) ↔
  Even n :=
sorry

end NUMINAMATH_CALUDE_all_lamps_on_iff_even_l3288_328896


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l3288_328867

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ+, n ∣ (2^n.val - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l3288_328867


namespace NUMINAMATH_CALUDE_student_selection_probability_l3288_328854

theorem student_selection_probability : 
  let total_students : ℕ := 4
  let selected_students : ℕ := 2
  let target_group : ℕ := 2
  let favorable_outcomes : ℕ := target_group * (total_students - target_group)
  let total_outcomes : ℕ := Nat.choose total_students selected_students
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3288_328854


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3288_328871

theorem exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3288_328871


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l3288_328884

theorem min_value_exponential_sum (x y : ℝ) (h : 2 * x + 3 * y = 6) :
  ∃ (m : ℝ), m = 16 ∧ ∀ a b, 2 * a + 3 * b = 6 → 4^a + 8^b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l3288_328884


namespace NUMINAMATH_CALUDE_digit_multiplication_theorem_l3288_328876

/-- A function that checks if a number is a digit (0-9) -/
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A function that converts a three-digit number to its decimal representation -/
def three_digit_to_decimal (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- A function that converts a four-digit number to its decimal representation -/
def four_digit_to_decimal (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem digit_multiplication_theorem (A B C D : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D)
  (h_multiplication : three_digit_to_decimal A B C * D = four_digit_to_decimal A B C D) :
  C + D = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_theorem_l3288_328876


namespace NUMINAMATH_CALUDE_milk_distribution_l3288_328804

theorem milk_distribution (boxes : Nat) (bottles_per_box : Nat) (eaten : Nat) (people : Nat) :
  boxes = 7 →
  bottles_per_box = 9 →
  eaten = 7 →
  people = 8 →
  (boxes * bottles_per_box - eaten) / people = 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_distribution_l3288_328804


namespace NUMINAMATH_CALUDE_large_puzzle_cost_l3288_328850

theorem large_puzzle_cost (small large : ℝ) 
  (h1 : small + large = 23)
  (h2 : large + 3 * small = 39) : 
  large = 15 := by
  sorry

end NUMINAMATH_CALUDE_large_puzzle_cost_l3288_328850


namespace NUMINAMATH_CALUDE_degree_of_5x_cubed_plus_9_to_10_l3288_328885

/-- The degree of a polynomial of the form (ax³ + b)ⁿ where a and b are constants and n is a positive integer -/
def degree_of_cubic_plus_constant_to_power (a b : ℝ) (n : ℕ+) : ℕ :=
  3 * n

/-- Theorem stating that the degree of (5x³ + 9)¹⁰ is 30 -/
theorem degree_of_5x_cubed_plus_9_to_10 :
  degree_of_cubic_plus_constant_to_power 5 9 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_5x_cubed_plus_9_to_10_l3288_328885


namespace NUMINAMATH_CALUDE_boat_license_count_l3288_328861

/-- Represents the set of possible letters for a boat license -/
def BoatLicenseLetter : Finset Char := {'A', 'M'}

/-- Represents the set of possible digits for a boat license -/
def BoatLicenseDigit : Finset Nat := Finset.range 10

/-- The number of digits in a boat license -/
def BoatLicenseDigitCount : Nat := 5

/-- Calculates the total number of possible boat licenses -/
def TotalBoatLicenses : Nat :=
  BoatLicenseLetter.card * (BoatLicenseDigit.card ^ BoatLicenseDigitCount)

/-- Theorem stating that the total number of boat licenses is 200,000 -/
theorem boat_license_count :
  TotalBoatLicenses = 200000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l3288_328861


namespace NUMINAMATH_CALUDE_initial_cards_calculation_l3288_328872

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem initial_cards_calculation :
  initial_cards = cards_given_away + cards_left :=
by sorry

end NUMINAMATH_CALUDE_initial_cards_calculation_l3288_328872


namespace NUMINAMATH_CALUDE_johns_pool_depth_l3288_328887

theorem johns_pool_depth (sarah_depth john_depth : ℕ) : 
  sarah_depth = 5 →
  john_depth = 2 * sarah_depth + 5 →
  john_depth = 15 := by
  sorry

end NUMINAMATH_CALUDE_johns_pool_depth_l3288_328887


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l3288_328802

/-- Given a parabola y = 4x and two points on it with perpendicular tangents,
    prove that the x-coordinate of the intersection of these tangents is -1. -/
theorem intersection_of_perpendicular_tangents_on_parabola
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = 4 * x₁)
  (h₂ : y₂ = 4 * x₂)
  (h_perp : (4 / y₁) * (4 / y₂) = -1) :
  ∃ (x y : ℝ), x = -1 ∧ 
    y = (4 / y₁) * x + y₁ / 2 ∧
    y = (4 / y₂) * x + y₂ / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l3288_328802


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l3288_328879

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun n => start + (n - 1) * (total / sampleSize)

theorem systematic_sampling_fourth_element :
  let total := 800
  let sampleSize := 50
  let interval := total / sampleSize
  let start := 7
  (systematicSample total sampleSize start 4 = 55) ∧ 
  (49 ≤ systematicSample total sampleSize start 4) ∧ 
  (systematicSample total sampleSize start 4 ≤ 64) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l3288_328879


namespace NUMINAMATH_CALUDE_taxi_cost_proof_l3288_328897

/-- The cost per mile for a taxi ride to the airport -/
def cost_per_mile : ℚ := 5 / 14

/-- Mike's distance in miles -/
def mike_distance : ℚ := 28

theorem taxi_cost_proof :
  ∀ (x : ℚ),
  (2.5 + x * mike_distance = 2.5 + 5 + x * 14) →
  x = cost_per_mile := by
  sorry

end NUMINAMATH_CALUDE_taxi_cost_proof_l3288_328897


namespace NUMINAMATH_CALUDE__l3288_328865

def smallest_angle_theorem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 2) (hc : ‖c‖ = 5) (habc : a + b + c = 0) :
  Real.arccos (inner a c / (‖a‖ * ‖c‖)) = π := by sorry

end NUMINAMATH_CALUDE__l3288_328865


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3288_328826

theorem quadratic_inequality (x : ℝ) : 
  3 * x^2 + 2 * x - 3 > 10 - 2 * x ↔ x < (-2 - Real.sqrt 43) / 3 ∨ x > (-2 + Real.sqrt 43) / 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3288_328826


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3288_328840

theorem polynomial_simplification (x : ℝ) : (x^2 - 4) * (x - 2) * (x + 2) = x^4 - 8*x^2 + 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3288_328840


namespace NUMINAMATH_CALUDE_hour_hand_path_l3288_328868

/-- The number of times the hour hand covers its path in a day -/
def coverages_per_day : ℕ := 2

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours for one full rotation of the hour hand -/
def hours_per_rotation : ℕ := 12

/-- The path covered by the hour hand in one rotation, in degrees -/
def path_per_rotation : ℝ := 360

theorem hour_hand_path :
  path_per_rotation = 360 :=
sorry

end NUMINAMATH_CALUDE_hour_hand_path_l3288_328868


namespace NUMINAMATH_CALUDE_daniel_correct_answers_l3288_328833

/-- Represents a mathematics competition --/
structure MathCompetition where
  total_problems : ℕ
  points_correct : ℕ
  points_incorrect : ℤ

/-- Represents a contestant's performance --/
structure ContestantPerformance where
  correct_answers : ℕ
  incorrect_answers : ℕ
  total_score : ℤ

/-- The specific competition Daniel participated in --/
def danielCompetition : MathCompetition :=
  { total_problems := 12
  , points_correct := 4
  , points_incorrect := -3 }

/-- Calculates the score based on correct and incorrect answers --/
def calculateScore (comp : MathCompetition) (perf : ContestantPerformance) : ℤ :=
  (comp.points_correct : ℤ) * perf.correct_answers + comp.points_incorrect * perf.incorrect_answers

/-- Theorem stating that Daniel must have answered 9 questions correctly --/
theorem daniel_correct_answers (comp : MathCompetition) (perf : ContestantPerformance) :
    comp = danielCompetition →
    perf.correct_answers + perf.incorrect_answers = comp.total_problems →
    calculateScore comp perf = 21 →
    perf.correct_answers = 9 := by
  sorry

end NUMINAMATH_CALUDE_daniel_correct_answers_l3288_328833


namespace NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l3288_328815

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (145 - 7 * p)

/-- The optimal price that maximizes revenue --/
def optimal_price : ℕ := 10

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℕ, p ≤ 30 → R p ≤ R optimal_price := by
  sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l3288_328815


namespace NUMINAMATH_CALUDE_containers_per_truck_is_160_l3288_328857

/-- The number of trucks with 20 boxes each -/
def trucks_with_20_boxes : ℕ := 7

/-- The number of trucks with 12 boxes each -/
def trucks_with_12_boxes : ℕ := 5

/-- The number of boxes on trucks with 20 boxes -/
def boxes_on_20_box_trucks : ℕ := 20

/-- The number of boxes on trucks with 12 boxes -/
def boxes_on_12_box_trucks : ℕ := 12

/-- The number of containers of oil in each box -/
def containers_per_box : ℕ := 8

/-- The number of trucks for redistribution -/
def redistribution_trucks : ℕ := 10

/-- The total number of containers of oil -/
def total_containers : ℕ := 
  (trucks_with_20_boxes * boxes_on_20_box_trucks + 
   trucks_with_12_boxes * boxes_on_12_box_trucks) * containers_per_box

/-- The number of containers per truck after redistribution -/
def containers_per_truck : ℕ := total_containers / redistribution_trucks

theorem containers_per_truck_is_160 : containers_per_truck = 160 := by
  sorry

end NUMINAMATH_CALUDE_containers_per_truck_is_160_l3288_328857


namespace NUMINAMATH_CALUDE_root_implies_inequality_l3288_328859

theorem root_implies_inequality (a b : ℝ) 
  (h : (a + b + a) * (a + b + b) = 9) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_inequality_l3288_328859


namespace NUMINAMATH_CALUDE_green_mandm_probability_l3288_328892

structure MandMJar :=
  (green : ℕ) (red : ℕ) (blue : ℕ) (orange : ℕ) (yellow : ℕ) (purple : ℕ) (brown : ℕ) (pink : ℕ)

def initial_jar : MandMJar :=
  ⟨35, 25, 10, 15, 0, 0, 0, 0⟩

def carter_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green - 20, jar.red - 8, jar.blue, jar.orange, jar.yellow, jar.purple, jar.brown, jar.pink⟩

def sister_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red / 2, jar.blue, jar.orange, jar.yellow + 14, jar.purple, jar.brown, jar.pink⟩

def alex_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, jar.blue, 0, jar.yellow - 3, jar.purple + 8, jar.brown, jar.pink⟩

def cousin_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, 0, jar.orange, jar.yellow, jar.purple, jar.brown + 10, jar.pink⟩

def sister_adds_pink (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, jar.blue, jar.orange, jar.yellow, jar.purple, jar.brown, jar.pink + 10⟩

def total_mandms (jar : MandMJar) : ℕ :=
  jar.green + jar.red + jar.blue + jar.orange + jar.yellow + jar.purple + jar.brown + jar.pink

theorem green_mandm_probability :
  let final_jar := sister_adds_pink (cousin_eats (alex_eats (sister_eats (carter_eats initial_jar))))
  (final_jar.green : ℚ) / ((total_mandms final_jar - 1) : ℚ) = 15 / 61 := by sorry

end NUMINAMATH_CALUDE_green_mandm_probability_l3288_328892


namespace NUMINAMATH_CALUDE_ratio_of_system_l3288_328890

theorem ratio_of_system (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_system_l3288_328890


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3288_328870

theorem dining_bill_calculation (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h_total : total = 184.80)
  (h_tax : tax_rate = 0.10)
  (h_tip : tip_rate = 0.20) : 
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total ∧ 
    food_price = 140 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3288_328870


namespace NUMINAMATH_CALUDE_joeys_swimming_time_l3288_328874

theorem joeys_swimming_time (ethan_time : ℝ) 
  (h1 : ethan_time > 0)
  (h2 : 3/4 * ethan_time = 9)
  (h3 : ethan_time = 12) :
  1/2 * ethan_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_joeys_swimming_time_l3288_328874


namespace NUMINAMATH_CALUDE_average_extra_chores_l3288_328866

/-- Proves that given the specified conditions, the average number of extra chores per week is 15 -/
theorem average_extra_chores
  (fixed_allowance : ℝ)
  (extra_chore_pay : ℝ)
  (total_weeks : ℕ)
  (total_earned : ℝ)
  (h1 : fixed_allowance = 20)
  (h2 : extra_chore_pay = 1.5)
  (h3 : total_weeks = 10)
  (h4 : total_earned = 425) :
  (total_earned / total_weeks - fixed_allowance) / extra_chore_pay = 15 := by
  sorry

#check average_extra_chores

end NUMINAMATH_CALUDE_average_extra_chores_l3288_328866


namespace NUMINAMATH_CALUDE_intersection_when_m_is_2_subset_iff_m_leq_1_l3288_328824

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (3 - 2*x) ∧ x ∈ Set.Icc (-13/2) (3/2)}
def B (m : ℝ) : Set ℝ := Set.Icc (1 - m) (m + 1)

-- Statement 1: When m = 2, A ∩ B = [0, 3]
theorem intersection_when_m_is_2 : A ∩ B 2 = Set.Icc 0 3 := by sorry

-- Statement 2: B ⊆ A if and only if m ≤ 1
theorem subset_iff_m_leq_1 : ∀ m, B m ⊆ A ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_2_subset_iff_m_leq_1_l3288_328824


namespace NUMINAMATH_CALUDE_heart_ratio_theorem_l3288_328844

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_theorem : (heart 3 5 : ℚ) / (heart 5 3 : ℚ) = 26 / 67 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_theorem_l3288_328844


namespace NUMINAMATH_CALUDE_oblique_projection_preserves_parallelogram_l3288_328883

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  a : Point2D
  b : Point2D
  c : Point2D
  d : Point2D

/-- Represents an oblique projection transformation -/
def ObliqueProjection := Point2D → Point2D

/-- Checks if four points form a parallelogram -/
def isParallelogram (a b c d : Point2D) : Prop := sorry

/-- The theorem stating that oblique projection preserves parallelograms -/
theorem oblique_projection_preserves_parallelogram 
  (p : Parallelogram) (proj : ObliqueProjection) :
  let p' := Parallelogram.mk 
    (proj p.a) (proj p.b) (proj p.c) (proj p.d)
  isParallelogram p'.a p'.b p'.c p'.d := by sorry

end NUMINAMATH_CALUDE_oblique_projection_preserves_parallelogram_l3288_328883


namespace NUMINAMATH_CALUDE_parabola_translation_l3288_328820

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original_parabola : Parabola := { a := 1, b := -2, c := 4 }

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * d, c := p.c + p.a * d^2 - p.b * d }

/-- The resulting parabola after translation -/
def translated_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 1

theorem parabola_translation :
  translated_parabola = { a := 1, b := -4, c := 10 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3288_328820


namespace NUMINAMATH_CALUDE_pattern_symmetries_l3288_328891

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : Point
  direction : Point

/-- Represents the pattern on the line -/
structure Pattern where
  line : Line
  unit_length : ℝ  -- Length of one repeating unit

/-- Represents a rigid motion transformation -/
inductive RigidMotion
  | Rotation (center : Point) (angle : ℝ)
  | Translation (direction : Point) (distance : ℝ)
  | Reflection (line : Line)

/-- Checks if a transformation preserves the pattern -/
def preserves_pattern (p : Pattern) (t : RigidMotion) : Prop :=
  sorry

theorem pattern_symmetries (p : Pattern) :
  (∃ (center : Point), preserves_pattern p (RigidMotion.Rotation center (2 * π / 3))) ∧
  (∃ (center : Point), preserves_pattern p (RigidMotion.Rotation center (4 * π / 3))) ∧
  (preserves_pattern p (RigidMotion.Translation p.line.direction p.unit_length)) ∧
  (preserves_pattern p (RigidMotion.Reflection p.line)) ∧
  (∃ (perp_line : Line), 
    (perp_line.direction.x * p.line.direction.x + perp_line.direction.y * p.line.direction.y = 0) ∧
    preserves_pattern p (RigidMotion.Reflection perp_line)) :=
by sorry

end NUMINAMATH_CALUDE_pattern_symmetries_l3288_328891


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3288_328810

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + b) ∧
    (m * 1 - f 1 + b = 0) ∧
    (m = 4 ∧ b = -2) := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l3288_328810


namespace NUMINAMATH_CALUDE_triangle_side_length_l3288_328899

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Conditions
  b = 3 * Real.sqrt 3 →
  B = Real.pi / 3 →
  Real.sin A = 1 / 3 →
  -- Law of Sines (given as an additional condition since it's a fundamental property)
  a / Real.sin B = b / Real.sin A →
  -- Conclusion
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3288_328899


namespace NUMINAMATH_CALUDE_box_balls_count_l3288_328808

theorem box_balls_count : ∃ x : ℕ, (x > 20 ∧ x < 30 ∧ x - 20 = 30 - x) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_box_balls_count_l3288_328808


namespace NUMINAMATH_CALUDE_yella_computer_usage_l3288_328827

/-- Yella's computer usage problem -/
theorem yella_computer_usage 
  (last_week_usage : ℕ) 
  (this_week_first_4_days : ℕ) 
  (this_week_last_3_days : ℕ) 
  (next_week_weekday_classes : ℕ) 
  (next_week_weekday_gaming : ℕ) 
  (next_week_weekend_usage : ℕ) : 
  last_week_usage = 91 →
  this_week_first_4_days = 8 →
  this_week_last_3_days = 10 →
  next_week_weekday_classes = 5 →
  next_week_weekday_gaming = 3 →
  next_week_weekend_usage = 12 →
  (last_week_usage - (4 * this_week_first_4_days + 3 * this_week_last_3_days) = 29) ∧
  (last_week_usage - (5 * (next_week_weekday_classes + next_week_weekday_gaming) + 2 * next_week_weekend_usage) = 27) :=
by sorry

end NUMINAMATH_CALUDE_yella_computer_usage_l3288_328827


namespace NUMINAMATH_CALUDE_max_n_theorem_l3288_328878

/-- Represents a convex polygon with interior points -/
structure ConvexPolygonWithInteriorPoints where
  n : ℕ  -- number of vertices in the polygon
  interior_points : ℕ  -- number of interior points
  no_collinear : Bool  -- no three points are collinear

/-- Calculates the number of triangles formed in a polygon with interior points -/
def num_triangles (p : ConvexPolygonWithInteriorPoints) : ℕ :=
  p.n + p.interior_points + 198

/-- The maximum value of n for which no more than 300 triangles are formed -/
def max_n_for_300_triangles : ℕ := 102

/-- Theorem stating the maximum value of n for which no more than 300 triangles are formed -/
theorem max_n_theorem (p : ConvexPolygonWithInteriorPoints) 
    (h1 : p.interior_points = 100)
    (h2 : p.no_collinear = true) :
    (∀ m : ℕ, m > max_n_for_300_triangles → num_triangles { n := m, interior_points := 100, no_collinear := true } > 300) ∧
    num_triangles { n := max_n_for_300_triangles, interior_points := 100, no_collinear := true } ≤ 300 :=
  sorry

end NUMINAMATH_CALUDE_max_n_theorem_l3288_328878


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3288_328852

-- Define the two given lines
def l₁ (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := y = 1 - x

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define the line passing through the intersection point and origin
def target_line (x y : ℝ) : Prop := 3 * x + 2 * y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧
  ∀ (x y : ℝ), (x = 0 ∧ y = 0) ∨ (x = x₀ ∧ y = y₀) → target_line x y :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l3288_328852


namespace NUMINAMATH_CALUDE_parrot_female_fraction_l3288_328848

theorem parrot_female_fraction (total_birds : ℝ) (female_parrot_fraction : ℝ) : 
  (3 / 5 : ℝ) * total_birds +                   -- number of parrots
  (2 / 5 : ℝ) * total_birds =                   -- number of toucans
  total_birds ∧                                 -- total number of birds
  (3 / 4 : ℝ) * ((2 / 5 : ℝ) * total_birds) +   -- number of female toucans
  female_parrot_fraction * ((3 / 5 : ℝ) * total_birds) = -- number of female parrots
  (1 / 2 : ℝ) * total_birds →                   -- total number of female birds
  female_parrot_fraction = (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_parrot_female_fraction_l3288_328848


namespace NUMINAMATH_CALUDE_f_sum_equals_negative_two_l3288_328834

def f (x : ℝ) : ℝ := x^3 - x - 1

theorem f_sum_equals_negative_two : 
  f 2023 + (deriv f) 2023 + f (-2023) - (deriv f) (-2023) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_negative_two_l3288_328834


namespace NUMINAMATH_CALUDE_candies_in_packet_l3288_328831

/-- The number of candies in a packet of candy -/
def candies_per_packet : ℕ := 18

/-- The number of packets Bobby buys -/
def num_packets : ℕ := 2

/-- The number of days per week Bobby eats 2 candies -/
def days_eating_two : ℕ := 5

/-- The number of days per week Bobby eats 1 candy -/
def days_eating_one : ℕ := 2

/-- The number of weeks it takes to finish the packets -/
def weeks_to_finish : ℕ := 3

/-- Theorem stating the number of candies in a packet -/
theorem candies_in_packet :
  candies_per_packet * num_packets = 
  (days_eating_two * 2 + days_eating_one * 1) * weeks_to_finish :=
by sorry

end NUMINAMATH_CALUDE_candies_in_packet_l3288_328831


namespace NUMINAMATH_CALUDE_weaving_sum_l3288_328816

/-- The sum of an arithmetic sequence with first term 5, common difference 16/29, and 30 terms -/
theorem weaving_sum : 
  let a₁ : ℚ := 5
  let d : ℚ := 16 / 29
  let n : ℕ := 30
  (n : ℚ) * a₁ + (n * (n - 1) : ℚ) / 2 * d = 390 := by
  sorry

end NUMINAMATH_CALUDE_weaving_sum_l3288_328816


namespace NUMINAMATH_CALUDE_expression_value_l3288_328847

theorem expression_value : 
  (121^2 - 19^2) / (91^2 - 13^2) * ((91 - 13)*(91 + 13)) / ((121 - 19)*(121 + 19)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3288_328847


namespace NUMINAMATH_CALUDE_oil_volume_in_tank_l3288_328864

/-- The volume of oil in a cylindrical tank with given dimensions and mixture ratio -/
theorem oil_volume_in_tank (tank_height : ℝ) (tank_diameter : ℝ) (fill_percentage : ℝ) 
  (oil_ratio : ℝ) (water_ratio : ℝ) (h_height : tank_height = 8) 
  (h_diameter : tank_diameter = 3) (h_fill : fill_percentage = 0.75) 
  (h_ratio : oil_ratio / (oil_ratio + water_ratio) = 3 / 10) : 
  (fill_percentage * π * (tank_diameter / 2)^2 * tank_height) * (oil_ratio / (oil_ratio + water_ratio)) = 4.05 * π := by
sorry

end NUMINAMATH_CALUDE_oil_volume_in_tank_l3288_328864


namespace NUMINAMATH_CALUDE_find_number_l3288_328841

theorem find_number : ∃ x : ℝ, (38 + 2 * x = 124) ∧ (x = 43) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3288_328841


namespace NUMINAMATH_CALUDE_equation_solution_l3288_328821

theorem equation_solution (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3288_328821


namespace NUMINAMATH_CALUDE_money_division_l3288_328856

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h1 : total = 527)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  b = 93 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3288_328856


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l3288_328843

/-- An isosceles triangle with given leg and base lengths -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.leg + t.base

/-- The perimeter of a quadrilateral formed by joining two isosceles triangles along their legs -/
def perimeterQuadLeg (t : IsoscelesTriangle) : ℝ := 2 * t.base + 2 * t.leg

/-- The perimeter of a quadrilateral formed by joining two isosceles triangles along their bases -/
def perimeterQuadBase (t : IsoscelesTriangle) : ℝ := 4 * t.leg

theorem isosceles_triangle_sides (t : IsoscelesTriangle) :
  perimeter t = 100 ∧
  perimeterQuadLeg t + 4 = perimeterQuadBase t →
  t.leg = 34 ∧ t.base = 32 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l3288_328843


namespace NUMINAMATH_CALUDE_cosine_values_l3288_328894

def terminalPoint : ℝ × ℝ := (-3, 4)

theorem cosine_values (α : ℝ) (h : terminalPoint ∈ {p : ℝ × ℝ | ∃ t, p.1 = t * Real.cos α ∧ p.2 = t * Real.sin α}) :
  Real.cos α = -3/5 ∧ Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_values_l3288_328894


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l3288_328845

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 + x^2 - 6*x - 6
  (∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 3 ∨ x = -2) ∧
  (p (-1) = 0) ∧ (p 3 = 0) ∧ (p (-2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l3288_328845


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3288_328851

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, is_three_digit m → is_7_heavy m → 104 ≤ m) ∧ 
  is_three_digit 104 ∧ 
  is_7_heavy 104 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3288_328851


namespace NUMINAMATH_CALUDE_rectangle_area_l3288_328858

/-- The area of a rectangle with perimeter 40 feet and length twice its width is 800/9 square feet. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : perimeter = 40)
  (h2 : length = 2 * width)
  (h3 : perimeter = 2 * length + 2 * width)
  (h4 : area = length * width) :
  area = 800 / 9 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l3288_328858


namespace NUMINAMATH_CALUDE_function_properties_l3288_328837

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem function_properties :
  -- Part 1: When a = 0, the zero of the function is x = 2
  (∃ x : ℝ, f 0 x = 0 ∧ x = 2) ∧
  
  -- Part 2: When a = 1, the range of m for solutions in [1,3] is [-1/4, 2]
  (∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ f 1 x = m) ↔ m ∈ Set.Icc (-1/4) 2) ∧
  
  -- Part 3: When a > 0, the solution set of f(x) > 0
  (∀ a : ℝ, a > 0 →
    (a = 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x ≠ 2}) ∧
    (0 < a ∧ a < 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x > 1/a ∨ x < 2}) ∧
    (a > 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x < 1/a ∨ x > 2}))
  := by sorry


end NUMINAMATH_CALUDE_function_properties_l3288_328837


namespace NUMINAMATH_CALUDE_second_square_size_l3288_328855

/-- Represents a square on the board -/
structure Square :=
  (size : Nat)
  (position : Nat × Nat)

/-- Represents the board configuration -/
def BoardConfiguration := List Square

/-- Checks if a given configuration covers the entire 10x10 board -/
def covers_board (config : BoardConfiguration) : Prop := sorry

/-- Checks if all squares in the configuration have different sizes -/
def all_different_sizes (config : BoardConfiguration) : Prop := sorry

/-- Checks if the last two squares in the configuration are 5x5 and 4x4 -/
def last_two_squares_correct (config : BoardConfiguration) : Prop := sorry

/-- Checks if the second square in the configuration is 8x8 -/
def second_square_is_8x8 (config : BoardConfiguration) : Prop := sorry

theorem second_square_size (config : BoardConfiguration) :
  config.length = 6 →
  covers_board config →
  all_different_sizes config →
  last_two_squares_correct config →
  second_square_is_8x8 config :=
sorry

end NUMINAMATH_CALUDE_second_square_size_l3288_328855


namespace NUMINAMATH_CALUDE_double_reflection_of_F_l3288_328805

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem double_reflection_of_F :
  let F : ℝ × ℝ := (5, 2)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_y_equals_x F'
  F'' = (2, -5) := by sorry

end NUMINAMATH_CALUDE_double_reflection_of_F_l3288_328805


namespace NUMINAMATH_CALUDE_zero_in_interval_l3288_328806

-- Define the function f(x) = x^5 + x - 3
def f (x : ℝ) : ℝ := x^5 + x - 3

-- Theorem statement
theorem zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3288_328806


namespace NUMINAMATH_CALUDE_line_equation_proof_l3288_328830

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The projection point P(-2,1) -/
def projection_point : Point := ⟨-2, 1⟩

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The line passing through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  ⟨a, b, c⟩

/-- The line perpendicular to a given line passing through a point -/
def perpendicular_line_through_point (l : Line) (p : Point) : Line :=
  ⟨l.b, -l.a, l.a * p.y - l.b * p.x⟩

theorem line_equation_proof (L : Line) : 
  (point_on_line projection_point L) ∧ 
  (perpendicular L (line_through_points origin projection_point)) →
  L = ⟨2, -1, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_line_equation_proof_l3288_328830


namespace NUMINAMATH_CALUDE_base_conversion_2200_to_base9_l3288_328838

-- Define a function to convert a base 9 number to base 10
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

-- Theorem statement
theorem base_conversion_2200_to_base9 :
  base9ToBase10 [4, 1, 0, 3] = 2200 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_2200_to_base9_l3288_328838


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3288_328817

/-- 
Given an equilateral triangle where one of its sides is also a side of an isosceles triangle,
this theorem proves that if the isosceles triangle has a perimeter of 65 and a base of 25,
then the perimeter of the equilateral triangle is 60.
-/
theorem equilateral_triangle_perimeter 
  (s : ℝ) 
  (h_isosceles_perimeter : s + s + 25 = 65) 
  (h_equilateral_side : s > 0) : 
  3 * s = 60 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3288_328817


namespace NUMINAMATH_CALUDE_min_value_sequence_l3288_328849

theorem min_value_sequence (a : ℕ → ℝ) (h1 : a 1 = 25) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ 9 ∧ ∃ m : ℕ, m ≥ 1 ∧ a m / m = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_sequence_l3288_328849


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l3288_328882

theorem set_membership_implies_value (m : ℤ) : 3 ∈ ({1, m + 2} : Set ℤ) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l3288_328882


namespace NUMINAMATH_CALUDE_coin_count_l3288_328853

theorem coin_count (num_25_cent num_10_cent : ℕ) : 
  num_25_cent = 17 → num_10_cent = 17 → num_25_cent + num_10_cent = 34 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_l3288_328853


namespace NUMINAMATH_CALUDE_small_birdhouse_price_is_seven_l3288_328842

/-- Represents the price of birdhouses and sales information. -/
structure BirdhouseSales where
  large_price : ℕ
  medium_price : ℕ
  large_sold : ℕ
  medium_sold : ℕ
  small_sold : ℕ
  total_sales : ℕ

/-- Calculates the price of small birdhouses given the sales information. -/
def small_birdhouse_price (sales : BirdhouseSales) : ℕ :=
  (sales.total_sales - (sales.large_price * sales.large_sold + sales.medium_price * sales.medium_sold)) / sales.small_sold

/-- Theorem stating that the price of small birdhouses is $7 given the specific sales information. -/
theorem small_birdhouse_price_is_seven :
  let sales := BirdhouseSales.mk 22 16 2 2 3 97
  small_birdhouse_price sales = 7 := by
  sorry

#eval small_birdhouse_price (BirdhouseSales.mk 22 16 2 2 3 97)

end NUMINAMATH_CALUDE_small_birdhouse_price_is_seven_l3288_328842


namespace NUMINAMATH_CALUDE_max_abs_sum_quadratic_coeff_l3288_328832

theorem max_abs_sum_quadratic_coeff (a b c : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) → 
  |a| + |b| + |c| ≤ 3 ∧ ∃ a' b' c' : ℝ, 
    (∀ x : ℝ, |x| ≤ 1 → |a'*x^2 + b'*x + c'| ≤ 1) ∧ 
    |a'| + |b'| + |c'| = 3 :=
sorry

end NUMINAMATH_CALUDE_max_abs_sum_quadratic_coeff_l3288_328832


namespace NUMINAMATH_CALUDE_complex_multiplication_l3288_328860

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication :
  (1 + i) * i = -1 + i :=
sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3288_328860


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3288_328839

/-- Given a quadratic function f(x) = x^2 - 2x + 3, if f(m) = f(n) where m ≠ n, 
    then f(m + n) = 3 -/
theorem quadratic_function_property (m n : ℝ) (h : m ≠ n) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  f m = f n → f (m + n) = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3288_328839


namespace NUMINAMATH_CALUDE_f_at_2_equals_neg_26_l3288_328893

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_at_2_equals_neg_26 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_f_at_2_equals_neg_26_l3288_328893


namespace NUMINAMATH_CALUDE_zero_point_existence_l3288_328813

theorem zero_point_existence (a : ℝ) :
  a < -2 → 
  (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
  (¬ ∀ a : ℝ, (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) → a < -2) :=
sorry

end NUMINAMATH_CALUDE_zero_point_existence_l3288_328813


namespace NUMINAMATH_CALUDE_candidate_a_republican_voters_l3288_328889

theorem candidate_a_republican_voters (total : ℝ) (h_total_pos : total > 0) : 
  let dem_percent : ℝ := 0.7
  let rep_percent : ℝ := 1 - dem_percent
  let dem_for_a_percent : ℝ := 0.8
  let total_for_a_percent : ℝ := 0.65
  let rep_for_a_percent : ℝ := 
    (total_for_a_percent - dem_percent * dem_for_a_percent) / rep_percent
  rep_for_a_percent = 0.3 := by
sorry

end NUMINAMATH_CALUDE_candidate_a_republican_voters_l3288_328889


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l3288_328809

/-- Represents the state of the game machine -/
structure GameState :=
  (score : ℕ)
  (rubles_spent : ℕ)

/-- Defines the possible moves in the game -/
inductive Move
| insert_one : Move
| insert_two : Move

/-- Applies a move to the current game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.insert_one => ⟨state.score + 1, state.rubles_spent + 1⟩
  | Move.insert_two => ⟨state.score * 2, state.rubles_spent + 2⟩

/-- Checks if the game state is valid (score ≤ 50) -/
def is_valid_state (state : GameState) : Prop :=
  state.score ≤ 50

/-- Checks if the game is won (score = 50) -/
def is_winning_state (state : GameState) : Prop :=
  state.score = 50

/-- The main theorem to prove -/
theorem min_rubles_to_win :
  ∃ (moves : List Move),
    let final_state := moves.foldl apply_move ⟨0, 0⟩
    is_valid_state final_state ∧
    is_winning_state final_state ∧
    final_state.rubles_spent = 11 ∧
    (∀ (other_moves : List Move),
      let other_final_state := other_moves.foldl apply_move ⟨0, 0⟩
      is_valid_state other_final_state →
      is_winning_state other_final_state →
      other_final_state.rubles_spent ≥ 11) :=
sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l3288_328809


namespace NUMINAMATH_CALUDE_minimize_sum_of_distances_l3288_328818

/-- Given points P, Q, and R in ℝ², if R is chosen to minimize the sum of distances |PR| + |RQ|, then R lies on the line segment PQ. -/
theorem minimize_sum_of_distances (P Q R : ℝ × ℝ) :
  P = (-2, -2) →
  Q = (0, -1) →
  R.1 = 2 →
  (∀ S : ℝ × ℝ, dist P R + dist R Q ≤ dist P S + dist S Q) →
  R.2 = 0 := by sorry


end NUMINAMATH_CALUDE_minimize_sum_of_distances_l3288_328818


namespace NUMINAMATH_CALUDE_max_university_students_l3288_328875

theorem max_university_students (j m : ℕ) : 
  m = 2 * j + 100 →  -- Max's university has twice Julie's students plus 100
  m + j = 5400 →     -- Total students in both universities
  m = 3632           -- Number of students in Max's university
  := by sorry

end NUMINAMATH_CALUDE_max_university_students_l3288_328875


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3288_328829

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares : a^2 + b^2 + c^2 = 0.1) : 
  a^4 + b^4 + c^4 = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3288_328829


namespace NUMINAMATH_CALUDE_power_product_equality_l3288_328822

theorem power_product_equality : (-0.25)^2014 * (-4)^2015 = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3288_328822


namespace NUMINAMATH_CALUDE_motel_pricing_l3288_328801

/-- Represents the motel's pricing structure and guest stays. -/
structure MotelStay where
  flatFee : ℝ
  regularRate : ℝ
  discountRate : ℝ
  markStay : ℕ
  markCost : ℝ
  lucyStay : ℕ
  lucyCost : ℝ

/-- The motel's pricing satisfies the given conditions. -/
def validPricing (m : MotelStay) : Prop :=
  m.discountRate = 0.8 * m.regularRate ∧
  m.markStay = 5 ∧
  m.lucyStay = 7 ∧
  m.markCost = m.flatFee + 3 * m.regularRate + 2 * m.discountRate ∧
  m.lucyCost = m.flatFee + 3 * m.regularRate + 4 * m.discountRate ∧
  m.markCost = 310 ∧
  m.lucyCost = 410

/-- The theorem stating the correct flat fee and regular rate. -/
theorem motel_pricing (m : MotelStay) (h : validPricing m) :
  m.flatFee = 22.5 ∧ m.regularRate = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_motel_pricing_l3288_328801


namespace NUMINAMATH_CALUDE_multiply_1307_by_1307_l3288_328814

theorem multiply_1307_by_1307 : 1307 * 1307 = 1709249 := by
  sorry

end NUMINAMATH_CALUDE_multiply_1307_by_1307_l3288_328814


namespace NUMINAMATH_CALUDE_f_minus_g_greater_than_two_l3288_328880

noncomputable def f (x : ℝ) : ℝ := (2 - x^3) * Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem f_minus_g_greater_than_two (x : ℝ) (h : x ∈ Set.Ioo 0 1) : f x - g x > 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_g_greater_than_two_l3288_328880


namespace NUMINAMATH_CALUDE_pennies_indeterminate_l3288_328846

/-- Represents the number of coins Sandy has -/
structure SandyCoins where
  pennies : ℕ
  nickels : ℕ

/-- Represents the state of Sandy's coins before and after her dad's borrowing -/
structure SandyState where
  initial : SandyCoins
  borrowed_nickels : ℕ
  remaining : SandyCoins

/-- Defines the conditions of the problem -/
def valid_state (s : SandyState) : Prop :=
  s.initial.nickels = 31 ∧
  s.borrowed_nickels = 20 ∧
  s.remaining.nickels = 11 ∧
  s.initial.nickels = s.remaining.nickels + s.borrowed_nickels

/-- Theorem stating that the initial number of pennies cannot be determined -/
theorem pennies_indeterminate (s1 s2 : SandyState) :
  valid_state s1 → valid_state s2 → s1.initial.pennies ≠ s2.initial.pennies → True := by
  sorry

end NUMINAMATH_CALUDE_pennies_indeterminate_l3288_328846


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l3288_328863

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The mean of two numbers is 50 -/
def MeanIs50 (x y : ℕ) : Prop := (x + y) / 2 = 50

theorem max_ratio_two_digit_mean_50 :
  ∃ (x y : ℕ), TwoDigitPositiveInt x ∧ TwoDigitPositiveInt y ∧ MeanIs50 x y ∧
    ∀ (a b : ℕ), TwoDigitPositiveInt a → TwoDigitPositiveInt b → MeanIs50 a b →
      (a : ℚ) / b ≤ (x : ℚ) / y ∧ (x : ℚ) / y = 99 := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l3288_328863


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l3288_328898

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l3288_328898


namespace NUMINAMATH_CALUDE_total_tabs_is_300_l3288_328819

/-- Calculates the total number of tabs opened across all browsers --/
def totalTabs (numBrowsers : ℕ) (windowsPerBrowser : ℕ) (initialTabsPerWindow : ℕ) (additionalTabsPerTwelve : ℕ) : ℕ :=
  let tabsPerWindow := initialTabsPerWindow + additionalTabsPerTwelve
  let tabsPerBrowser := tabsPerWindow * windowsPerBrowser
  tabsPerBrowser * numBrowsers

/-- Proves that the total number of tabs is 300 given the specified conditions --/
theorem total_tabs_is_300 :
  totalTabs 4 5 12 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_tabs_is_300_l3288_328819


namespace NUMINAMATH_CALUDE_marys_age_l3288_328877

/-- Given that Suzy is 20 years old now and in four years she will be twice Mary's age,
    prove that Mary is currently 8 years old. -/
theorem marys_age (suzy_age : ℕ) (mary_age : ℕ) : 
  suzy_age = 20 → 
  (suzy_age + 4 = 2 * (mary_age + 4)) → 
  mary_age = 8 := by
sorry

end NUMINAMATH_CALUDE_marys_age_l3288_328877


namespace NUMINAMATH_CALUDE_curve_L_properties_l3288_328825

/-- Definition of the curve L -/
def L (p : ℕ) (x y : ℤ) : Prop := 4 * y^2 = (x - p) * p

/-- A prime number is odd if it's not equal to 2 -/
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p ≠ 2

theorem curve_L_properties (p : ℕ) (hp : is_odd_prime p) :
  (∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ ∀ (x y : ℤ), (x, y) ∈ S → y ≠ 0 ∧ L p x y) ∧
  (∀ (x y : ℤ), L p x y → ¬ ∃ (d : ℤ), d^2 = x^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_curve_L_properties_l3288_328825


namespace NUMINAMATH_CALUDE_circle_center_l3288_328869

/-- Given a circle with equation (x-2)^2 + (y+1)^2 = 3, prove that its center is at (2, -1) -/
theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3288_328869


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3288_328881

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 52) (h3 : x = 3 * y) :
  ∃ y_new : ℝ, ((-10 : ℝ) * y_new = k) ∧ (y_new = -50.7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3288_328881


namespace NUMINAMATH_CALUDE_odd_digits_base4_523_l3288_328807

/-- Represents a digit in base 4 --/
def Base4Digit := Fin 4

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : ℕ) : List Base4Digit := sorry

/-- Checks if a Base4Digit is odd --/
def isOddBase4Digit (d : Base4Digit) : Bool := sorry

/-- Counts the number of odd digits in a list of Base4Digits --/
def countOddDigits (digits : List Base4Digit) : ℕ := sorry

theorem odd_digits_base4_523 :
  countOddDigits (toBase4 523) = 2 := by sorry

end NUMINAMATH_CALUDE_odd_digits_base4_523_l3288_328807


namespace NUMINAMATH_CALUDE_can_guess_number_l3288_328803

theorem can_guess_number (n : Nat) (q : Nat) : n ≤ 1000 → q = 10 → 2^q ≥ n → True := by
  sorry

end NUMINAMATH_CALUDE_can_guess_number_l3288_328803
