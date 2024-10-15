import Mathlib

namespace NUMINAMATH_CALUDE_min_teams_in_championship_l4014_401493

/-- Represents a soccer championship with the given rules --/
structure SoccerChampionship where
  numTeams : ℕ
  /-- Each team plays one match against every other team --/
  totalMatches : ℕ := numTeams * (numTeams - 1) / 2
  /-- Winning team gets 2 points, tie gives 1 point to each team, losing team gets 0 points --/
  pointSystem : List ℕ := [2, 1, 0]

/-- Represents the points of a team --/
structure TeamPoints where
  wins : ℕ
  draws : ℕ
  points : ℕ := 2 * wins + draws

/-- The condition that one team has the most points but fewer wins than any other team --/
def hasUniqueLeader (c : SoccerChampionship) (leader : TeamPoints) (others : List TeamPoints) : Prop :=
  ∀ team ∈ others, leader.points > team.points ∧ leader.wins < team.wins

/-- The main theorem stating the minimum number of teams --/
theorem min_teams_in_championship : 
  ∀ c : SoccerChampionship, 
  ∀ leader : TeamPoints,
  ∀ others : List TeamPoints,
  hasUniqueLeader c leader others →
  c.numTeams ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_teams_in_championship_l4014_401493


namespace NUMINAMATH_CALUDE_trig_problem_l4014_401476

theorem trig_problem (θ : Real) 
  (h1 : θ > 0) 
  (h2 : θ < Real.pi / 2) 
  (h3 : Real.cos (θ + Real.pi / 6) = 1 / 3) : 
  Real.sin θ = (2 * Real.sqrt 6 - 1) / 6 ∧ 
  Real.sin (2 * θ + Real.pi / 6) = (4 * Real.sqrt 6 + 7) / 18 := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l4014_401476


namespace NUMINAMATH_CALUDE_base5_23104_equals_1654_l4014_401448

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (d₄ d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₄ * 5^4 + d₃ * 5^3 + d₂ * 5^2 + d₁ * 5^1 + d₀ * 5^0

/-- The base 5 number 23104 is equal to 1654 in base 10 --/
theorem base5_23104_equals_1654 :
  base5ToBase10 2 3 1 0 4 = 1654 := by
  sorry

end NUMINAMATH_CALUDE_base5_23104_equals_1654_l4014_401448


namespace NUMINAMATH_CALUDE_shorter_base_length_for_specific_trapezoid_l4014_401409

/-- Represents a trapezoid with a median line divided by a diagonal -/
structure TrapezoidWithDiagonal where
  median_length : ℝ
  segment_difference : ℝ

/-- Calculates the length of the shorter base of the trapezoid -/
def shorter_base_length (t : TrapezoidWithDiagonal) : ℝ :=
  t.median_length - t.segment_difference

/-- Theorem stating the length of the shorter base given specific measurements -/
theorem shorter_base_length_for_specific_trapezoid :
  let t : TrapezoidWithDiagonal := { median_length := 16, segment_difference := 4 }
  shorter_base_length t = 12 := by
  sorry

end NUMINAMATH_CALUDE_shorter_base_length_for_specific_trapezoid_l4014_401409


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4014_401464

theorem inequality_solution_set :
  let S := {x : ℝ | (x + 5) * (3 - 2*x) ≤ 6}
  S = {x : ℝ | -9 ≤ x ∧ x ≤ 1/2} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4014_401464


namespace NUMINAMATH_CALUDE_exactly_one_real_solution_l4014_401443

theorem exactly_one_real_solution :
  ∃! x : ℝ, ((-4 * (x - 3)^2 : ℝ) ≥ 0) := by sorry

end NUMINAMATH_CALUDE_exactly_one_real_solution_l4014_401443


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l4014_401408

theorem negative_fraction_comparison : -3/5 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l4014_401408


namespace NUMINAMATH_CALUDE_congruence_solution_unique_solution_in_range_l4014_401491

theorem congruence_solution (m : ℤ) : 
  (13 * m ≡ 9 [ZMOD 47]) ↔ (m ≡ 26 [ZMOD 47]) :=
by sorry

theorem unique_solution_in_range : 
  ∃! x : ℕ, x < 47 ∧ (13 * x ≡ 9 [ZMOD 47]) :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_unique_solution_in_range_l4014_401491


namespace NUMINAMATH_CALUDE_necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l4014_401478

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (6 - m) = 1 ∧ m ≠ 4

/-- The condition 2 < m < 6 is necessary for the equation to represent an ellipse -/
theorem necessary_condition (m : ℝ) :
  is_ellipse m → 2 < m ∧ m < 6 := by sorry

/-- The condition 2 < m < 6 is not sufficient for the equation to represent an ellipse -/
theorem not_sufficient_condition :
  ∃ m : ℝ, 2 < m ∧ m < 6 ∧ ¬(is_ellipse m) := by sorry

/-- The main theorem stating that 2 < m < 6 is necessary but not sufficient -/
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_ellipse m → 2 < m ∧ m < 6) ∧
  (∃ m : ℝ, 2 < m ∧ m < 6 ∧ ¬(is_ellipse m)) := by sorry

end NUMINAMATH_CALUDE_necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l4014_401478


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l4014_401434

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (n ≤ 6 ∧ ((1 : ℚ) / 5 + (n : ℚ) / 8 + 1 < 2)) ∧
  ∀ (m : ℕ), m > 6 → ((1 : ℚ) / 5 + (m : ℚ) / 8 + 1 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l4014_401434


namespace NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l4014_401449

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: In any sequence of 39 consecutive natural numbers, 
    there exists at least one number whose digit sum is divisible by 11 -/
theorem digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digitSum (N + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l4014_401449


namespace NUMINAMATH_CALUDE_margo_walk_distance_l4014_401435

/-- Calculates the total distance of a round trip given the times for each leg and the average speed -/
def round_trip_distance (outbound_time inbound_time : ℚ) (average_speed : ℚ) : ℚ :=
  let total_time := outbound_time + inbound_time
  average_speed * (total_time / 60)

/-- Proves that given the specific conditions of Margo's walk, the total distance is 2 miles -/
theorem margo_walk_distance :
  round_trip_distance (15 : ℚ) (25 : ℚ) (3 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_margo_walk_distance_l4014_401435


namespace NUMINAMATH_CALUDE_basket_weight_l4014_401444

/-- Given a basket of persimmons, prove the weight of the empty basket. -/
theorem basket_weight (total_weight half_weight : ℝ) 
  (h1 : total_weight = 62)
  (h2 : half_weight = 34) : 
  ∃ (basket_weight persimmons_weight : ℝ),
    basket_weight + persimmons_weight = total_weight ∧ 
    basket_weight + persimmons_weight / 2 = half_weight ∧
    basket_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_basket_weight_l4014_401444


namespace NUMINAMATH_CALUDE_problem_solution_l4014_401487

-- Definition of the relation (x, y) = n
def relation (x y n : ℝ) : Prop := x^n = y

theorem problem_solution :
  -- Part 1
  relation 10 1000 3 ∧
  relation (-5) 25 2 ∧
  -- Part 2
  (∀ x, relation x 16 2 → (x = 4 ∨ x = -4)) ∧
  -- Part 3
  (∀ a b, relation 4 a 2 → relation b 8 3 → relation b a 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4014_401487


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l4014_401442

def systematic_sampling (total : ℕ) (sample_size : ℕ) (interval_start : ℕ) (interval_end : ℕ) : ℕ :=
  let sampling_interval := total / sample_size
  let interval_size := interval_end - interval_start + 1
  interval_size / sampling_interval

theorem systematic_sampling_result :
  systematic_sampling 420 21 281 420 = 7 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l4014_401442


namespace NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l4014_401416

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- Two angles are adjacent if they share a common vertex and a common side,
    but have no common interior points -/
def Adjacent (α β : ℝ) : Prop := True  -- We simplify this for the statement

theorem adjacent_complementary_angles_are_complementary 
  (α β : ℝ) (h1 : Adjacent α β) (h2 : Complementary α β) : Complementary α β := by
  sorry

end NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l4014_401416


namespace NUMINAMATH_CALUDE_monthly_expenses_calculation_l4014_401413

/-- Calculates monthly expenses given initial investment, monthly revenue, and payback period. -/
def calculate_monthly_expenses (initial_investment : ℕ) (monthly_revenue : ℕ) (payback_months : ℕ) : ℕ :=
  (monthly_revenue * payback_months - initial_investment) / payback_months

theorem monthly_expenses_calculation (initial_investment monthly_revenue payback_months : ℕ) 
  (h1 : initial_investment = 25000)
  (h2 : monthly_revenue = 4000)
  (h3 : payback_months = 10) :
  calculate_monthly_expenses initial_investment monthly_revenue payback_months = 1500 := by
  sorry

#eval calculate_monthly_expenses 25000 4000 10

end NUMINAMATH_CALUDE_monthly_expenses_calculation_l4014_401413


namespace NUMINAMATH_CALUDE_specific_building_occupancy_l4014_401463

/-- Represents the building structure and occupancy --/
structure Building where
  floors : Nat
  first_floor_apartments : Nat
  common_difference : Nat
  one_bedroom_occupancy : Nat
  two_bedroom_occupancy : Nat
  three_bedroom_occupancy : Nat

/-- Calculates the total number of people in the building --/
def total_occupancy (b : Building) : Nat :=
  let last_floor_apartments := b.first_floor_apartments + (b.floors - 1) * b.common_difference
  let total_apartments := (b.floors * (b.first_floor_apartments + last_floor_apartments)) / 2
  let apartments_per_type := total_apartments / 3
  apartments_per_type * (b.one_bedroom_occupancy + b.two_bedroom_occupancy + b.three_bedroom_occupancy)

/-- Theorem stating the total occupancy of the specific building --/
theorem specific_building_occupancy :
  let b : Building := {
    floors := 25,
    first_floor_apartments := 3,
    common_difference := 2,
    one_bedroom_occupancy := 2,
    two_bedroom_occupancy := 4,
    three_bedroom_occupancy := 5
  }
  total_occupancy b = 2475 := by
  sorry

end NUMINAMATH_CALUDE_specific_building_occupancy_l4014_401463


namespace NUMINAMATH_CALUDE_divisor_exists_l4014_401472

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem divisor_exists : ∃ d : ℕ, 
  d > 0 ∧ 
  is_prime (9453 / d) ∧ 
  is_perfect_square (9453 % d) ∧ 
  d = 61 := by
sorry

end NUMINAMATH_CALUDE_divisor_exists_l4014_401472


namespace NUMINAMATH_CALUDE_animal_survival_probability_l4014_401400

theorem animal_survival_probability (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.7) 
  (h2 : p_25 = 0.56) : 
  p_25 / p_20 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_animal_survival_probability_l4014_401400


namespace NUMINAMATH_CALUDE_system_solution_l4014_401458

theorem system_solution :
  ∃! (x y : ℚ), (4 * x - 3 * y = -2) ∧ (8 * x + 5 * y = 7) ∧ x = 1/4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4014_401458


namespace NUMINAMATH_CALUDE_sqrt_tan_domain_l4014_401465

theorem sqrt_tan_domain (x : ℝ) :
  ∃ (y : ℝ), y = Real.sqrt (Real.tan x) ↔ ∃ (k : ℤ), k * Real.pi ≤ x ∧ x < k * Real.pi + Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_tan_domain_l4014_401465


namespace NUMINAMATH_CALUDE_sequence_ratio_l4014_401423

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-1 : ℝ) - a₁ = a₁ - a₂) →  -- arithmetic sequence condition
  (a₂ - (-4 : ℝ) = a₁ - a₂) →  -- arithmetic sequence condition
  (b₁ / (-1 : ℝ) = b₂ / b₁) →  -- geometric sequence condition
  (b₂ / b₁ = b₃ / b₂) →        -- geometric sequence condition
  (b₃ / b₂ = (-4 : ℝ) / b₃) →  -- geometric sequence condition
  (a₂ - a₁) / b₂ = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l4014_401423


namespace NUMINAMATH_CALUDE_multiple_of_ab_l4014_401438

theorem multiple_of_ab (a b : ℕ+) : 
  (∃ k : ℕ, a.val ^ 2017 + b.val = k * a.val * b.val) ↔ 
  ((a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2^2017)) := by
sorry

end NUMINAMATH_CALUDE_multiple_of_ab_l4014_401438


namespace NUMINAMATH_CALUDE_calvins_roaches_l4014_401418

theorem calvins_roaches (total insects : ℕ) (scorpions : ℕ) (roaches crickets caterpillars : ℕ) : 
  insects = 27 →
  scorpions = 3 →
  crickets = roaches / 2 →
  caterpillars = 2 * scorpions →
  insects = roaches + scorpions + crickets + caterpillars →
  roaches = 12 := by
sorry

end NUMINAMATH_CALUDE_calvins_roaches_l4014_401418


namespace NUMINAMATH_CALUDE_circle_properties_l4014_401410

/-- Given a circle with circumference 36 cm, prove its diameter and area. -/
theorem circle_properties (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  let d := 2 * r
  let A := Real.pi * r^2
  d = 36 / Real.pi ∧ A = 324 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l4014_401410


namespace NUMINAMATH_CALUDE_rectangle_formations_with_restrictions_l4014_401447

/-- The number of ways to choose 4 lines to form a rectangle -/
def rectangleFormations (h v : ℕ) (hRestricted vRestricted : Fin 2 → ℕ) : ℕ :=
  let hChoices := (Nat.choose h 2) - 1
  let vChoices := (Nat.choose v 2) - 1
  hChoices * vChoices

/-- Theorem stating the number of ways to form a rectangle with given conditions -/
theorem rectangle_formations_with_restrictions :
  rectangleFormations 6 7 ![2, 5] ![3, 6] = 280 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_with_restrictions_l4014_401447


namespace NUMINAMATH_CALUDE_circular_paper_pieces_for_square_border_l4014_401439

theorem circular_paper_pieces_for_square_border (side_length : ℝ) (pieces_per_circle : ℕ) : 
  side_length = 10 → pieces_per_circle = 20 → (4 * side_length) / (2 * π) * pieces_per_circle = 40 := by
  sorry

#check circular_paper_pieces_for_square_border

end NUMINAMATH_CALUDE_circular_paper_pieces_for_square_border_l4014_401439


namespace NUMINAMATH_CALUDE_displacement_increment_from_2_to_2_plus_d_l4014_401446

/-- Represents the displacement of an object at time t -/
def displacement (t : ℝ) : ℝ := 2 * t^2

/-- Represents the increment in displacement between two time points -/
def displacementIncrement (t₁ t₂ : ℝ) : ℝ := displacement t₂ - displacement t₁

theorem displacement_increment_from_2_to_2_plus_d (d : ℝ) :
  displacementIncrement 2 (2 + d) = 8 * d + 2 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_displacement_increment_from_2_to_2_plus_d_l4014_401446


namespace NUMINAMATH_CALUDE_min_a_value_l4014_401490

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2*a) → a ≥ 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_a_value_l4014_401490


namespace NUMINAMATH_CALUDE_volume_maximized_at_one_l4014_401474

/-- The volume function of the lidless square box -/
def V (x : ℝ) : ℝ := x * (6 - 2*x)^2

/-- The derivative of the volume function -/
def V' (x : ℝ) : ℝ := 12*x^2 - 48*x + 36

theorem volume_maximized_at_one :
  ∀ x ∈ Set.Ioo 0 3, V x ≤ V 1 :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_one_l4014_401474


namespace NUMINAMATH_CALUDE_ball_trajectory_5x5_table_l4014_401414

/-- Represents a square pool table --/
structure PoolTable :=
  (size : Nat)

/-- Represents a ball's trajectory on the pool table --/
structure BallTrajectory :=
  (table : PoolTable)
  (start_corner : Nat × Nat)
  (angle : Real)

/-- Represents the final state of the ball --/
structure FinalState :=
  (end_pocket : String)
  (edge_hits : Nat)
  (diagonal_squares : Nat)

/-- Main theorem about the ball's trajectory on a 5x5 pool table --/
theorem ball_trajectory_5x5_table :
  ∀ (t : PoolTable) (b : BallTrajectory),
    t.size = 5 →
    b.table = t →
    b.start_corner = (0, 0) →
    b.angle = 45 →
    ∃ (f : FinalState),
      f.end_pocket = "upper-left" ∧
      f.edge_hits = 5 ∧
      f.diagonal_squares = 23 :=
sorry

end NUMINAMATH_CALUDE_ball_trajectory_5x5_table_l4014_401414


namespace NUMINAMATH_CALUDE_wage_increase_for_unit_productivity_increase_l4014_401436

/-- Regression line equation for workers' wages as a function of labor productivity -/
def regression_line (x : ℝ) : ℝ := 80 * x + 50

/-- Theorem: The average increase in wage when labor productivity increases by 1 unit -/
theorem wage_increase_for_unit_productivity_increase :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 80 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_for_unit_productivity_increase_l4014_401436


namespace NUMINAMATH_CALUDE_congruent_triangles_equal_perimeter_l4014_401424

/-- Represents a triangle -/
structure Triangle where
  perimeter : ℝ

/-- Two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

theorem congruent_triangles_equal_perimeter (t1 t2 : Triangle) 
  (h1 : Congruent t1 t2) (h2 : t1.perimeter = 5) : t2.perimeter = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_equal_perimeter_l4014_401424


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l4014_401433

/-- Given that a is a real number and i is the imaginary unit, 
    if (a+3i)/(1-2i) is a pure imaginary number, then a = 6 -/
theorem complex_fraction_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ b : ℝ, (a + 3 * Complex.I) / (1 - 2 * Complex.I) = b * Complex.I) →
  a = 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l4014_401433


namespace NUMINAMATH_CALUDE_income_calculation_l4014_401455

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 8 = expenditure * 15 →
  income - expenditure = savings →
  savings = 7000 →
  income = 15000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l4014_401455


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4014_401422

theorem sqrt_equation_solution :
  ∃! (x : ℝ), Real.sqrt x + Real.sqrt (x + 8) = 8 ∧ x = 49/4 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4014_401422


namespace NUMINAMATH_CALUDE_intersection_A_B_l4014_401484

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4014_401484


namespace NUMINAMATH_CALUDE_quadratic_root_power_sums_l4014_401480

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots x₁ and x₂,
    s_n denotes the sum of the n-th powers of the roots. -/
def s (n : ℕ) (x₁ x₂ : ℝ) : ℝ := x₁^n + x₂^n

/-- Theorem stating the relations between sums of powers of roots of a quadratic equation -/
theorem quadratic_root_power_sums 
  (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h : a ≠ 0)
  (hroot : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  (∀ n : ℕ, n ≥ 2 → a * s n x₁ x₂ + b * s (n-1) x₁ x₂ + c * s (n-2) x₁ x₂ = 0) ∧
  (a * s 2 x₁ x₂ + b * s 1 x₁ x₂ + 2 * c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_power_sums_l4014_401480


namespace NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l4014_401469

-- Equation (1)
theorem equation_one_solution (x : ℝ) : 
  2 * (x - 2) - 3 * (4 * x - 1) = 9 * (1 - x) ↔ x = -10 := by sorry

-- System of Equations (2)
theorem system_of_equations_solution (x y : ℝ) :
  (4 * (x - y - 1) = 3 * (1 - y) - 2 ∧ x / 2 + y / 3 = 2) ↔ (x = 2 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l4014_401469


namespace NUMINAMATH_CALUDE_cost_price_calculation_l4014_401430

/-- Proves that the cost price of an article is 480, given the selling price and profit percentage -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 595.2 → 
  profit_percentage = 24 → 
  ∃ (cost_price : ℝ), 
    cost_price = 480 ∧ 
    selling_price = cost_price * (1 + profit_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l4014_401430


namespace NUMINAMATH_CALUDE_miley_bought_two_cellphones_l4014_401454

/-- The number of cellphones Miley bought -/
def num_cellphones : ℕ := 2

/-- The cost of each cellphone in dollars -/
def cost_per_cellphone : ℝ := 800

/-- The discount rate for buying more than one cellphone -/
def discount_rate : ℝ := 0.05

/-- The total amount Miley paid in dollars -/
def total_paid : ℝ := 1520

/-- Theorem stating that the number of cellphones Miley bought is 2 -/
theorem miley_bought_two_cellphones :
  num_cellphones = 2 ∧
  num_cellphones > 1 ∧
  (1 - discount_rate) * (num_cellphones : ℝ) * cost_per_cellphone = total_paid :=
by sorry

end NUMINAMATH_CALUDE_miley_bought_two_cellphones_l4014_401454


namespace NUMINAMATH_CALUDE_log_ratio_squared_l4014_401425

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (Real.log a)^2 - 4 * (Real.log a) + 1 = 0) →
  (2 * (Real.log b)^2 - 4 * (Real.log b) + 1 = 0) →
  ((Real.log (a / b))^2 = 2) := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l4014_401425


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt13_l4014_401432

/-- A circle with two given points on its circumference and its center on the y-axis -/
structure CircleWithPoints where
  center : ℝ × ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  point1_on_circle : (point1.1 - center.1)^2 + (point1.2 - center.2)^2 = (point2.1 - center.1)^2 + (point2.2 - center.2)^2

/-- The radius of the circle is √13 -/
theorem circle_radius_is_sqrt13 (c : CircleWithPoints) 
  (h1 : c.point1 = (2, 5)) 
  (h2 : c.point2 = (3, 6)) : 
  Real.sqrt ((c.point1.1 - c.center.1)^2 + (c.point1.2 - c.center.2)^2) = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_is_sqrt13_l4014_401432


namespace NUMINAMATH_CALUDE_C₁_cartesian_polar_equiv_l4014_401421

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + 16 = 0

/-- The curve C₁ in polar coordinates -/
def C₁_polar (ρ θ : ℝ) : Prop := ρ^2 - 8*ρ*Real.cos θ - 10*ρ*Real.sin θ + 16 = 0

/-- Theorem stating the equivalence of Cartesian and polar representations of C₁ -/
theorem C₁_cartesian_polar_equiv :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    (C₁ x y ↔ C₁_polar ρ θ) :=
by
  sorry

end NUMINAMATH_CALUDE_C₁_cartesian_polar_equiv_l4014_401421


namespace NUMINAMATH_CALUDE_license_plate_count_l4014_401486

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of alphanumeric characters (letters + digits) -/
def num_alphanumeric : ℕ := num_letters + num_digits

/-- The number of different license plates that can be formed -/
def num_license_plates : ℕ := num_letters * num_digits * num_alphanumeric

theorem license_plate_count : num_license_plates = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l4014_401486


namespace NUMINAMATH_CALUDE_determine_set_B_l4014_401405

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the theorem
theorem determine_set_B (A B : Set Nat) 
  (h1 : (A ∪ B)ᶜ = {1}) 
  (h2 : A ∩ Bᶜ = {3}) 
  (h3 : A ⊆ U) 
  (h4 : B ⊆ U) : 
  B = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_determine_set_B_l4014_401405


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l4014_401466

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + a - 2024 = 0) → 
  (b^2 + b - 2024 = 0) → 
  (a^2 + 2*a + b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l4014_401466


namespace NUMINAMATH_CALUDE_factorization_equality_l4014_401420

theorem factorization_equality (x y : ℝ) :
  3 * x^2 - x * y - y^2 = ((Real.sqrt 13 + 1) / 2 * x + y) * ((Real.sqrt 13 - 1) / 2 * x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4014_401420


namespace NUMINAMATH_CALUDE_correct_pizza_dough_amounts_l4014_401473

/-- Calculates the required amounts of milk and water for a given amount of flour in Luca's pizza dough recipe. -/
def pizzaDoughCalculation (flourAmount : ℚ) : ℚ × ℚ :=
  let milkToFlourRatio : ℚ := 80 / 400
  let waterToMilkRatio : ℚ := 1 / 2
  let milkAmount : ℚ := flourAmount * milkToFlourRatio
  let waterAmount : ℚ := milkAmount * waterToMilkRatio
  (milkAmount, waterAmount)

/-- Theorem stating the correct amounts of milk and water for 1200 mL of flour. -/
theorem correct_pizza_dough_amounts :
  pizzaDoughCalculation 1200 = (240, 120) := by
  sorry

#eval pizzaDoughCalculation 1200

end NUMINAMATH_CALUDE_correct_pizza_dough_amounts_l4014_401473


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l4014_401419

theorem candy_sampling_percentage
  (caught_sampling : Real)
  (total_sampling : Real)
  (h1 : caught_sampling = 22)
  (h2 : total_sampling = 27.5)
  : total_sampling - caught_sampling = 5.5 := by
sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l4014_401419


namespace NUMINAMATH_CALUDE_line_intersects_circle_l4014_401496

theorem line_intersects_circle (m : ℝ) (h_m : 0 < m ∧ m < 4/3) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + m*x₁ + m^2 - m = 0) ∧ 
  (x₂^2 + m*x₂ + m^2 - m = 0) ∧
  ∃ (x y : ℝ), 
    (m*x + y + m^2 - m = 0) ∧ 
    ((x - 1)^2 + (y + 1)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l4014_401496


namespace NUMINAMATH_CALUDE_tracy_balloons_l4014_401470

theorem tracy_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (total_after : ℕ) :
  brooke_initial = 12 →
  brooke_added = 8 →
  tracy_initial = 6 →
  total_after = 35 →
  ∃ (tracy_added : ℕ),
    brooke_initial + brooke_added + (tracy_initial + tracy_added) / 2 = total_after ∧
    tracy_added = 24 :=
by sorry

end NUMINAMATH_CALUDE_tracy_balloons_l4014_401470


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4014_401402

/-- The quadratic function y = x^2 - ax + a + 3 -/
def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3

theorem quadratic_function_properties (a : ℝ) :
  (∃ x, f a x = 0) ↔ (a ≤ -2 ∨ a ≥ 6) ∧
  (∀ x, f a x ≥ 4 ↔ 
    (a > 2 ∧ (x ≤ 1 ∨ x ≥ a - 1)) ∨
    (a = 2 ∧ true) ∨
    (a < 2 ∧ (x ≤ a - 1 ∨ x ≥ 1))) ∧
  ((∃ x ∈ Set.Icc 2 4, f a x = 0) → a ∈ Set.Icc 6 7) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l4014_401402


namespace NUMINAMATH_CALUDE_simplify_fraction_l4014_401411

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4014_401411


namespace NUMINAMATH_CALUDE_inequalities_solution_l4014_401407

theorem inequalities_solution :
  (∀ x : ℝ, (x - 2) * (1 - 3 * x) > 2 ↔ 1 < x ∧ x < 4/3) ∧
  (∀ x : ℝ, |((x + 1) / (x - 1))| > 2 ↔ (1/3 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l4014_401407


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4014_401481

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ + a₈ = 10, prove that a₅ = 5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 10) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4014_401481


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l4014_401499

theorem unique_solution_diophantine_equation :
  ∀ a b c d : ℕ+,
    4^(a:ℕ) * 5^(b:ℕ) - 3^(c:ℕ) * 11^(d:ℕ) = 1 →
    a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l4014_401499


namespace NUMINAMATH_CALUDE_solve_amusement_park_problem_l4014_401440

def amusement_park_problem (ticket_price : ℕ) (weekday_visitors : ℕ) (saturday_visitors : ℕ) (total_revenue : ℕ) : Prop :=
  let weekday_total := weekday_visitors * 5
  let sunday_visitors := (total_revenue - ticket_price * (weekday_total + saturday_visitors)) / ticket_price
  sunday_visitors = 300

theorem solve_amusement_park_problem :
  amusement_park_problem 3 100 200 3000 := by
  sorry

end NUMINAMATH_CALUDE_solve_amusement_park_problem_l4014_401440


namespace NUMINAMATH_CALUDE_four_girls_wins_l4014_401428

theorem four_girls_wins (a b c d : ℕ) : 
  a + b = 8 ∧ 
  a + c = 10 ∧ 
  b + c = 12 ∧ 
  a + d = 12 ∧ 
  b + d = 14 ∧ 
  c + d = 16 → 
  ({a, b, c, d} : Finset ℕ) = {3, 5, 7, 9} := by
sorry

end NUMINAMATH_CALUDE_four_girls_wins_l4014_401428


namespace NUMINAMATH_CALUDE_parabola_through_points_l4014_401431

/-- A parabola passing through three specific points -/
def parabola (x y : ℝ) : Prop :=
  y = -x^2 + 2*x + 3

theorem parabola_through_points :
  parabola (-1) 0 ∧ parabola 3 0 ∧ parabola 0 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_points_l4014_401431


namespace NUMINAMATH_CALUDE_cube_greater_iff_l4014_401452

theorem cube_greater_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_greater_iff_l4014_401452


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l4014_401427

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l4014_401427


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l4014_401403

theorem same_solution_implies_c_value (x : ℝ) (c : ℝ) :
  (3 * x + 9 = 6) ∧ (c * x - 15 = -5) → c = -10 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l4014_401403


namespace NUMINAMATH_CALUDE_polygon_sides_count_l4014_401441

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (2 * 360 : ℝ) = (n - 2 : ℝ) * 180 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l4014_401441


namespace NUMINAMATH_CALUDE_time_equation_l4014_401477

/-- Given the equations V = 2gt + V₀ and S = (1/3)gt² + V₀t + Ct³, where C is a constant,
    prove that the time t can be expressed as t = (V - V₀) / (2g). -/
theorem time_equation (g V V₀ S t : ℝ) (C : ℝ) :
  V = 2 * g * t + V₀ ∧ S = (1/3) * g * t^2 + V₀ * t + C * t^3 →
  t = (V - V₀) / (2 * g) := by
  sorry

end NUMINAMATH_CALUDE_time_equation_l4014_401477


namespace NUMINAMATH_CALUDE_hockey_players_count_l4014_401479

/-- The number of hockey players in a games hour -/
def hockey_players (total players cricket football softball : ℕ) : ℕ :=
  total - (cricket + football + softball)

/-- Theorem: There are 17 hockey players in the ground -/
theorem hockey_players_count : hockey_players 50 12 11 10 = 17 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l4014_401479


namespace NUMINAMATH_CALUDE_a_profit_calculation_l4014_401456

def total_subscription : ℕ := 50000
def total_profit : ℕ := 36000

def subscription_difference_a_b : ℕ := 4000
def subscription_difference_b_c : ℕ := 5000

def c_subscription (x : ℕ) : ℕ := x
def b_subscription (x : ℕ) : ℕ := x + subscription_difference_b_c
def a_subscription (x : ℕ) : ℕ := x + subscription_difference_b_c + subscription_difference_a_b

theorem a_profit_calculation :
  ∃ x : ℕ,
    c_subscription x + b_subscription x + a_subscription x = total_subscription ∧
    (a_subscription x : ℚ) / (total_subscription : ℚ) * (total_profit : ℚ) = 15120 :=
  sorry

end NUMINAMATH_CALUDE_a_profit_calculation_l4014_401456


namespace NUMINAMATH_CALUDE_katies_miles_l4014_401417

/-- Proves that Katie's miles run is 10, given Adam's miles and the difference between their runs -/
theorem katies_miles (adam_miles : ℕ) (difference : ℕ) (h1 : adam_miles = 35) (h2 : difference = 25) :
  adam_miles - difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_katies_miles_l4014_401417


namespace NUMINAMATH_CALUDE_square_to_parallelogram_l4014_401453

/-- Represents a plane figure --/
structure PlaneFigure where
  -- Add necessary fields

/-- Represents the oblique side drawing method --/
def obliqueSideDrawing (figure : PlaneFigure) : PlaneFigure :=
  sorry

/-- Predicate to check if a figure is a square --/
def isSquare (figure : PlaneFigure) : Prop :=
  sorry

/-- Predicate to check if a figure is a parallelogram --/
def isParallelogram (figure : PlaneFigure) : Prop :=
  sorry

/-- Theorem: The intuitive diagram of a square using oblique side drawing is a parallelogram --/
theorem square_to_parallelogram (figure : PlaneFigure) :
  isSquare figure → isParallelogram (obliqueSideDrawing figure) :=
by sorry

end NUMINAMATH_CALUDE_square_to_parallelogram_l4014_401453


namespace NUMINAMATH_CALUDE_complex_real_condition_l4014_401401

theorem complex_real_condition (a : ℝ) :
  (((a - Complex.I) / (2 + Complex.I)).im = 0) → a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l4014_401401


namespace NUMINAMATH_CALUDE_li_fang_outfits_l4014_401467

/-- The number of unique outfit combinations given a set of shirts, skirts, and dresses -/
def outfit_combinations (num_shirts num_skirts num_dresses : ℕ) : ℕ :=
  num_shirts * num_skirts + num_dresses

/-- Theorem: Given 4 shirts, 3 skirts, and 2 dresses, the total number of unique outfit combinations is 14 -/
theorem li_fang_outfits : outfit_combinations 4 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_li_fang_outfits_l4014_401467


namespace NUMINAMATH_CALUDE_sandy_parentheses_problem_l4014_401461

theorem sandy_parentheses_problem (p q r s : ℤ) (h1 : p = 2) (h2 : q = 4) (h3 : r = 6) (h4 : s = 8) :
  ∃ t : ℤ, p + (q - (r + (s - t))) = p + q - r + s - 10 ∧ t = 8 := by
sorry

end NUMINAMATH_CALUDE_sandy_parentheses_problem_l4014_401461


namespace NUMINAMATH_CALUDE_fourth_student_guess_is_525_l4014_401445

/-- Represents the number of jellybeans guessed by each student -/
def jellybean_guess : Fin 4 → ℕ
  | 0 => 100  -- First student's guess
  | 1 => 8 * jellybean_guess 0  -- Second student's guess
  | 2 => jellybean_guess 1 - 200  -- Third student's guess
  | 3 => (jellybean_guess 0 + jellybean_guess 1 + jellybean_guess 2) / 3 + 25  -- Fourth student's guess

/-- Theorem stating that the fourth student's guess is 525 -/
theorem fourth_student_guess_is_525 : jellybean_guess 3 = 525 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_guess_is_525_l4014_401445


namespace NUMINAMATH_CALUDE_square_area_error_l4014_401483

def error_in_area (excess_error : Real) (deficit_error : Real) : Real :=
  let correct_factor := (1 + excess_error) * (1 - deficit_error)
  (1 - correct_factor) * 100

theorem square_area_error :
  error_in_area 0.03 0.04 = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l4014_401483


namespace NUMINAMATH_CALUDE_will_chocolate_pieces_l4014_401471

/-- Calculates the number of chocolate pieces Will has left after giving some boxes away. -/
def chocolate_pieces_left (total_boxes : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (total_boxes - boxes_given) * pieces_per_box

/-- Proves that Will has 16 pieces of chocolate left after giving some boxes to his brother. -/
theorem will_chocolate_pieces : chocolate_pieces_left 7 3 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_will_chocolate_pieces_l4014_401471


namespace NUMINAMATH_CALUDE_mhsc_unanswered_questions_l4014_401429

/-- Represents the scoring system for the Math High School Contest -/
structure ScoringSystem where
  initial : ℤ
  correct : ℤ
  wrong : ℤ
  unanswered : ℤ

/-- Calculates the score based on a given scoring system and number of questions -/
def calculateScore (system : ScoringSystem) (correct wrong unanswered : ℕ) : ℤ :=
  system.initial + system.correct * correct + system.wrong * wrong + system.unanswered * unanswered

theorem mhsc_unanswered_questions (newSystem oldSystem : ScoringSystem)
    (totalQuestions newScore oldScore : ℕ) :
    newSystem = ScoringSystem.mk 0 6 0 1 →
    oldSystem = ScoringSystem.mk 25 5 (-2) 0 →
    totalQuestions = 30 →
    newScore = 110 →
    oldScore = 95 →
    ∃ (correct wrong unanswered : ℕ),
      correct + wrong + unanswered = totalQuestions ∧
      calculateScore newSystem correct wrong unanswered = newScore ∧
      calculateScore oldSystem correct wrong unanswered = oldScore ∧
      unanswered = 10 :=
  sorry


end NUMINAMATH_CALUDE_mhsc_unanswered_questions_l4014_401429


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l4014_401406

theorem waitress_income_fraction (salary : ℚ) (salary_pos : salary > 0) :
  let first_week_tips := (11 / 4) * salary
  let second_week_tips := (7 / 3) * salary
  let total_salary := 2 * salary
  let total_tips := first_week_tips + second_week_tips
  let total_income := total_salary + total_tips
  (total_tips / total_income) = 61 / 85 := by
  sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l4014_401406


namespace NUMINAMATH_CALUDE_product_of_extremes_is_cube_l4014_401457

theorem product_of_extremes_is_cube (a : Fin 2022 → ℕ)
  (h : ∀ i : Fin 2021, ∃ k : ℕ, a i * a (i.succ) = k^3) :
  ∃ m : ℕ, a 0 * a 2021 = m^3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_extremes_is_cube_l4014_401457


namespace NUMINAMATH_CALUDE_quadratic_roots_in_fourth_quadrant_l4014_401495

/-- A point in the fourth quadrant -/
structure FourthQuadrantPoint where
  x : ℝ
  y : ℝ
  x_pos : 0 < x
  y_neg : y < 0

/-- Quadratic equation coefficients -/
structure QuadraticCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation has two distinct real roots -/
def has_two_distinct_real_roots (q : QuadraticCoeffs) : Prop :=
  0 < q.b ^ 2 - 4 * q.a * q.c

theorem quadratic_roots_in_fourth_quadrant 
  (p : FourthQuadrantPoint) (q : QuadraticCoeffs) 
  (h : p.x = q.a ∧ p.y = q.c) : 
  has_two_distinct_real_roots q := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_in_fourth_quadrant_l4014_401495


namespace NUMINAMATH_CALUDE_five_digit_number_probability_l4014_401498

/-- The set of digits to choose from -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- The number of digits to select -/
def num_selected : Nat := 3

/-- The length of the number to form -/
def num_length : Nat := 5

/-- The number of digits that should be used twice -/
def num_twice_used : Nat := 2

/-- The probability of forming a number with two digits each used twice -/
def probability : Rat := 3/5

theorem five_digit_number_probability :
  (Finset.card digits = 5) →
  (num_selected = 3) →
  (num_length = 5) →
  (num_twice_used = 2) →
  (probability = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_number_probability_l4014_401498


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l4014_401489

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem line_perp_parallel_implies_planes_perp 
  (m : Line3D) (α β : Plane3D) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l4014_401489


namespace NUMINAMATH_CALUDE_solution_count_l4014_401450

-- Define the equation
def equation (x a : ℝ) : Prop :=
  Real.log (2 - x^2) / Real.log (x - a) = 2

-- Theorem statement
theorem solution_count (a : ℝ) :
  (∀ x, ¬ equation x a) ∨
  (∃! x, equation x a) ∨
  (∃ x y, x ≠ y ∧ equation x a ∧ equation y a) :=
by
  -- Case 1: No solution
  have h1 : a ≤ -2 ∨ a = 0 ∨ a ≥ Real.sqrt 2 → ∀ x, ¬ equation x a := by sorry
  -- Case 2: One solution
  have h2 : (-Real.sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < Real.sqrt 2) → ∃! x, equation x a := by sorry
  -- Case 3: Two solutions
  have h3 : -2 < a ∧ a < -Real.sqrt 2 → ∃ x y, x ≠ y ∧ equation x a ∧ equation y a := by sorry
  sorry -- Complete the proof using h1, h2, and h3


end NUMINAMATH_CALUDE_solution_count_l4014_401450


namespace NUMINAMATH_CALUDE_sum_of_xyz_l4014_401415

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 160) : 
  x + y + z = 14 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l4014_401415


namespace NUMINAMATH_CALUDE_new_female_percentage_new_female_percentage_proof_l4014_401492

theorem new_female_percentage (initial_female_percentage : ℝ) 
                               (additional_male_hires : ℕ) 
                               (total_employees_after : ℕ) : ℝ :=
  let initial_employees := total_employees_after - additional_male_hires
  let initial_female_employees := (initial_female_percentage / 100) * initial_employees
  (initial_female_employees / total_employees_after) * 100

#check 
  @new_female_percentage 60 20 240 = 55

theorem new_female_percentage_proof :
  new_female_percentage 60 20 240 = 55 := by
  sorry

end NUMINAMATH_CALUDE_new_female_percentage_new_female_percentage_proof_l4014_401492


namespace NUMINAMATH_CALUDE_photo_arrangements_l4014_401488

/-- Represents the number of people in the photo arrangement --/
def total_people : ℕ := 7

/-- Represents the number of students in the photo arrangement --/
def num_students : ℕ := 6

/-- Represents the position of the teacher in the row --/
def teacher_position : ℕ := 4

/-- Represents the number of positions to the left of the teacher --/
def left_positions : ℕ := 3

/-- Represents the number of positions to the right of the teacher --/
def right_positions : ℕ := 3

/-- Represents the number of positions available for Student A --/
def positions_for_A : ℕ := 5

/-- Represents the number of positions available for Student B --/
def positions_for_B : ℕ := 5

/-- Represents the number of remaining students after placing A and B --/
def remaining_students : ℕ := 4

/-- Theorem stating the number of different arrangements --/
theorem photo_arrangements :
  (positions_for_A * (positions_for_B - 1) * (remaining_students!)) * 2 = 960 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l4014_401488


namespace NUMINAMATH_CALUDE_simplify_expression_l4014_401482

theorem simplify_expression (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 2 * a * b - a^2 ≠ 0) :
  (a^2 - 2*a*b + b^2) / (a*b) - (2*a*b - b^2) / (2*a*b - a^2) = (a^2 - 2*a*b + 2*b^2) / (a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4014_401482


namespace NUMINAMATH_CALUDE_prob_three_even_in_five_rolls_l4014_401475

/-- A fair 10-sided die -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1 / 2

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice we want to show even numbers -/
def numEven : ℕ := 3

/-- The probability of rolling exactly three even numbers when five fair 10-sided dice are rolled -/
theorem prob_three_even_in_five_rolls : 
  (numDice.choose numEven : ℚ) * probEven ^ numEven * (1 - probEven) ^ (numDice - numEven) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_in_five_rolls_l4014_401475


namespace NUMINAMATH_CALUDE_foldable_rectangle_short_side_l4014_401412

/-- A rectangle with the property that when folded along its diagonal,
    it forms a trapezoid with three equal sides. -/
structure FoldableRectangle where
  long_side : ℝ
  short_side : ℝ
  long_side_positive : 0 < long_side
  short_side_positive : 0 < short_side
  long_side_longer : short_side ≤ long_side
  forms_equal_sided_trapezoid : True  -- This is a placeholder for the folding property

/-- The theorem stating that a rectangle with longer side 12 cm, when folded to form
    a trapezoid with three equal sides, has a shorter side of 4√3 cm. -/
theorem foldable_rectangle_short_side
  (rect : FoldableRectangle)
  (h_long : rect.long_side = 12) :
  rect.short_side = 4 * Real.sqrt 3 := by
  sorry

#check foldable_rectangle_short_side

end NUMINAMATH_CALUDE_foldable_rectangle_short_side_l4014_401412


namespace NUMINAMATH_CALUDE_loan_division_l4014_401459

theorem loan_division (total : ℝ) (rate1 rate2 years1 years2 : ℝ) : 
  total = 2665 ∧ rate1 = 3/100 ∧ rate2 = 5/100 ∧ years1 = 5 ∧ years2 = 3 →
  ∃ (part1 part2 : ℝ), 
    part1 + part2 = total ∧
    part1 * rate1 * years1 = part2 * rate2 * years2 ∧
    part2 = 1332.5 := by
  sorry

end NUMINAMATH_CALUDE_loan_division_l4014_401459


namespace NUMINAMATH_CALUDE_precious_stone_cost_l4014_401485

theorem precious_stone_cost (num_stones : ℕ) (total_amount : ℕ) (h1 : num_stones = 8) (h2 : total_amount = 14280) :
  total_amount / num_stones = 1785 := by
sorry

end NUMINAMATH_CALUDE_precious_stone_cost_l4014_401485


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_relationships_l4014_401468

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_and_parallel_relationships 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : contained_in m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_relationships_l4014_401468


namespace NUMINAMATH_CALUDE_almond_distribution_l4014_401497

/-- The number of almonds Elaine received -/
def elaine_almonds : ℕ := 12

/-- The number of almonds Daniel received -/
def daniel_almonds : ℕ := elaine_almonds - 8

theorem almond_distribution :
  (elaine_almonds = daniel_almonds + 8) ∧
  (daniel_almonds = elaine_almonds / 3) →
  elaine_almonds = 12 := by
  sorry

end NUMINAMATH_CALUDE_almond_distribution_l4014_401497


namespace NUMINAMATH_CALUDE_trick_or_treat_distribution_l4014_401426

/-- The number of blocks in the village -/
def num_blocks : ℕ := 9

/-- The total number of children going trick or treating -/
def total_children : ℕ := 54

/-- There are some children on each block -/
axiom children_on_each_block : ∀ b : ℕ, b < num_blocks → ∃ c : ℕ, c > 0

/-- The number of children on each block -/
def children_per_block : ℕ := total_children / num_blocks

theorem trick_or_treat_distribution :
  children_per_block = 6 :=
sorry

end NUMINAMATH_CALUDE_trick_or_treat_distribution_l4014_401426


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_138_18_is_6_exists_no_greater_main_result_l4014_401462

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem gcd_138_18_is_6 : Nat.gcd 138 18 = 6 :=
by sorry

theorem exists_no_greater : ¬∃ m : ℕ, 138 < m ∧ m < 150 ∧ Nat.gcd m 18 = 6 :=
by sorry

theorem main_result : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_138_18_is_6_exists_no_greater_main_result_l4014_401462


namespace NUMINAMATH_CALUDE_people_remaining_on_bus_l4014_401404

/-- The number of people remaining on a bus after a field trip with multiple stops -/
theorem people_remaining_on_bus 
  (left_side : ℕ) 
  (right_side : ℕ) 
  (back_section : ℕ) 
  (standing : ℕ) 
  (teachers : ℕ) 
  (driver : ℕ) 
  (first_stop : ℕ) 
  (second_stop : ℕ) 
  (third_stop : ℕ) 
  (h1 : left_side = 42)
  (h2 : right_side = 38)
  (h3 : back_section = 5)
  (h4 : standing = 15)
  (h5 : teachers = 2)
  (h6 : driver = 1)
  (h7 : first_stop = 15)
  (h8 : second_stop = 19)
  (h9 : third_stop = 5) :
  left_side + right_side + back_section + standing + teachers + driver - 
  (first_stop + second_stop + third_stop) = 64 :=
by sorry

end NUMINAMATH_CALUDE_people_remaining_on_bus_l4014_401404


namespace NUMINAMATH_CALUDE_mark_soup_donation_l4014_401437

theorem mark_soup_donation (shelters : ℕ) (people_per_shelter : ℕ) (cans_per_person : ℕ)
  (h1 : shelters = 6)
  (h2 : people_per_shelter = 30)
  (h3 : cans_per_person = 10) :
  shelters * people_per_shelter * cans_per_person = 1800 :=
by sorry

end NUMINAMATH_CALUDE_mark_soup_donation_l4014_401437


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l4014_401451

theorem abs_m_minus_n_equals_five (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l4014_401451


namespace NUMINAMATH_CALUDE_projected_strings_intersection_criterion_l4014_401494

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral2D where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the ratio of two line segments -/
def segmentRatio (P Q R : Point2D) : ℝ := sorry

/-- Determines if two projected strings intersect in 3D space -/
def stringsIntersect (quad : Quadrilateral2D) (P Q R S : Point2D) : Prop :=
  let ratio1 := segmentRatio quad.A P quad.B
  let ratio2 := segmentRatio quad.B Q quad.C
  let ratio3 := segmentRatio quad.C R quad.D
  let ratio4 := segmentRatio quad.D S quad.A
  ratio1 * ratio2 * ratio3 * ratio4 = 1

/-- Theorem: Projected strings intersect in 3D iff their segment ratio product is 1 -/
theorem projected_strings_intersection_criterion 
  (quad : Quadrilateral2D) (P Q R S : Point2D) : 
  stringsIntersect quad P Q R S ↔ 
  segmentRatio quad.A P quad.B * 
  segmentRatio quad.B Q quad.C * 
  segmentRatio quad.C R quad.D * 
  segmentRatio quad.D S quad.A = 1 := by sorry

#check projected_strings_intersection_criterion

end NUMINAMATH_CALUDE_projected_strings_intersection_criterion_l4014_401494


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l4014_401460

theorem arithmetic_sequence_sum_divisibility :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x c : ℕ), x > 0 → c ≥ 0 → 
    n ∣ (10 * x + 45 * c)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (x c : ℕ), x > 0 ∧ c ≥ 0 ∧ 
      ¬(m ∣ (10 * x + 45 * c))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l4014_401460
