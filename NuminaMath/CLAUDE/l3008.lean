import Mathlib

namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_two_l3008_300815

theorem product_xyz_equals_negative_two 
  (x y z : ℝ) 
  (h1 : x + 2 / y = 2) 
  (h2 : y + 2 / z = 2) : 
  x * y * z = -2 := by sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_two_l3008_300815


namespace NUMINAMATH_CALUDE_max_ratio_three_digit_number_to_digit_sum_l3008_300820

theorem max_ratio_three_digit_number_to_digit_sum :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c : ℚ) / (a + b + c) ≤ 100 ∧
    ∃ (a₀ b₀ c₀ : ℕ),
      1 ≤ a₀ ∧ a₀ ≤ 9 ∧
      0 ≤ b₀ ∧ b₀ ≤ 9 ∧
      0 ≤ c₀ ∧ c₀ ≤ 9 ∧
      (100 * a₀ + 10 * b₀ + c₀ : ℚ) / (a₀ + b₀ + c₀) = 100 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_three_digit_number_to_digit_sum_l3008_300820


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_example_l3008_300819

/-- Sum of an arithmetic series -/
def arithmetic_series_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℚ :=
  let n : ℚ := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

/-- Theorem: The sum of the arithmetic series with first term -35, last term 1, and common difference 2 is -323 -/
theorem arithmetic_series_sum_example : 
  arithmetic_series_sum (-35) 1 2 = -323 := by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_example_l3008_300819


namespace NUMINAMATH_CALUDE_sqrt_13_between_3_and_4_l3008_300893

theorem sqrt_13_between_3_and_4 (a : ℝ) (h : a = Real.sqrt 13) : 3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_between_3_and_4_l3008_300893


namespace NUMINAMATH_CALUDE_rectangle_side_length_l3008_300836

theorem rectangle_side_length (a b d : ℝ) : 
  a = 4 →
  a / b = 2 * (b / d) →
  d^2 = a^2 + b^2 →
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l3008_300836


namespace NUMINAMATH_CALUDE_objective_function_range_l3008_300834

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y > 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x + y

-- Theorem statement
theorem objective_function_range :
  ∃ (min max : ℝ), min = 1 ∧ max = 6 ∧
  (∀ x y : ℝ, FeasibleRegion x y →
    min ≤ ObjectiveFunction x y ∧ ObjectiveFunction x y ≤ max) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    FeasibleRegion x1 y1 ∧ FeasibleRegion x2 y2 ∧
    ObjectiveFunction x1 y1 = min ∧ ObjectiveFunction x2 y2 = max) :=
by
  sorry

end NUMINAMATH_CALUDE_objective_function_range_l3008_300834


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3008_300853

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3008_300853


namespace NUMINAMATH_CALUDE_range_of_m_l3008_300803

/-- Proposition p: The equation x²/(2m) - y²/(m-1) = 1 represents an ellipse with foci on the y-axis -/
def prop_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

/-- Proposition q: The eccentricity e of the hyperbola y²/5 - x²/m = 1 is in the interval (1,2) -/
def prop_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) →
  1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3008_300803


namespace NUMINAMATH_CALUDE_total_eggs_collected_l3008_300844

/-- The number of dozen eggs Benjamin collects per day -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects per day -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects per day -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem total_eggs_collected :
  total_eggs = 26 := by sorry

end NUMINAMATH_CALUDE_total_eggs_collected_l3008_300844


namespace NUMINAMATH_CALUDE_max_product_of_roots_l3008_300823

theorem max_product_of_roots (m : ℝ) : 
  let product_of_roots := m / 5
  let discriminant := 100 - 20 * m
  (discriminant ≥ 0) →  -- Condition for real roots
  product_of_roots ≤ 1 ∧ 
  (product_of_roots = 1 ↔ m = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_roots_l3008_300823


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3008_300862

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*x + y^2 + 10*y = -75

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 3 + Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3008_300862


namespace NUMINAMATH_CALUDE_trent_total_distance_l3008_300812

/-- Represents the distance Trent traveled throughout his day -/
def trent_travel (block_length : ℕ) (walk_blocks : ℕ) (bus_blocks : ℕ) (bike_blocks : ℕ) : ℕ :=
  2 * (walk_blocks + bus_blocks + bike_blocks) * block_length

/-- Theorem stating the total distance Trent traveled -/
theorem trent_total_distance :
  trent_travel 50 4 7 5 = 1600 := by sorry

end NUMINAMATH_CALUDE_trent_total_distance_l3008_300812


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3008_300870

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 1}

theorem set_intersection_theorem : M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3008_300870


namespace NUMINAMATH_CALUDE_solution_range_l3008_300833

def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_range (f : ℝ → ℝ) (h_monotone : monotone_increasing f) (h_zero : f 1 = 0) :
  {x : ℝ | f (x^2 + 3*x - 3) < 0} = Set.Ioo (-4) 1 := by sorry

end NUMINAMATH_CALUDE_solution_range_l3008_300833


namespace NUMINAMATH_CALUDE_average_speed_last_hour_l3008_300847

theorem average_speed_last_hour (total_distance : ℝ) (total_time : ℝ) 
  (first_30_speed : ℝ) (next_30_speed : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_30_speed = 50 →
  next_30_speed = 70 →
  let first_30_distance := first_30_speed * (30 / 60)
  let next_30_distance := next_30_speed * (30 / 60)
  let last_60_distance := total_distance - (first_30_distance + next_30_distance)
  let last_60_time := 60 / 60
  last_60_distance / last_60_time = 60 := by
  sorry

#check average_speed_last_hour

end NUMINAMATH_CALUDE_average_speed_last_hour_l3008_300847


namespace NUMINAMATH_CALUDE_not_sum_to_seven_l3008_300821

def pairs : List (Int × Int) := [(4, 3), (-1, 8), (10, -2), (2, 5), (3, 5)]

def sum_to_seven (pair : Int × Int) : Bool :=
  pair.1 + pair.2 = 7

theorem not_sum_to_seven : 
  ∀ (pair : Int × Int), 
    pair ∈ pairs → 
      (¬(sum_to_seven pair) ↔ (pair = (10, -2) ∨ pair = (3, 5))) := by
  sorry

#eval pairs.filter (λ pair => ¬(sum_to_seven pair))

end NUMINAMATH_CALUDE_not_sum_to_seven_l3008_300821


namespace NUMINAMATH_CALUDE_power_equation_solution_l3008_300857

theorem power_equation_solution :
  (∃ x : ℤ, (10 : ℝ)^655 * (10 : ℝ)^x = 1000) ∧
  (∀ x : ℤ, (10 : ℝ)^655 * (10 : ℝ)^x = 1000 → x = -652) :=
by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3008_300857


namespace NUMINAMATH_CALUDE_parabola_point_distance_l3008_300827

/-- Given a parabola y² = x with focus at (1/4, 0), prove that a point on the parabola
    with distance 1 from the focus has x-coordinate 3/4 -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = x →                                           -- Point (x, y) is on the parabola
  (x - 1/4)^2 + y^2 = 1 →                             -- Distance from (x, y) to focus (1/4, 0) is 1
  x = 3/4 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l3008_300827


namespace NUMINAMATH_CALUDE_triangle_angle_impossibility_l3008_300869

theorem triangle_angle_impossibility : ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- all angles are positive
  a + b + c = 180 ∧        -- sum of angles is 180 degrees
  a = 60 ∧                 -- one angle is 60 degrees
  b = 2 * a ∧              -- another angle is twice the first
  c ≠ 0                    -- the third angle is non-zero
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_impossibility_l3008_300869


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l3008_300877

/-- A convex hexagon with interior angles as consecutive integers has its largest angle equal to 122° -/
theorem hexagon_largest_angle : ∀ (a b c d e f : ℕ),
  -- The angles are natural numbers
  -- The angles are consecutive integers
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 →
  -- The sum of interior angles of a hexagon is 720°
  a + b + c + d + e + f = 720 →
  -- The largest angle is 122°
  f = 122 := by
sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l3008_300877


namespace NUMINAMATH_CALUDE_turkey_cost_l3008_300874

/-- The cost of turkeys given their weights and price per kilogram -/
theorem turkey_cost (w1 w2 w3 w4 : ℝ) (price_per_kg : ℝ) : 
  w1 = 6 →
  w2 = 9 →
  w3 = 2 * w2 →
  w4 = (w1 + w2 + w3) / 2 →
  price_per_kg = 2 →
  (w1 + w2 + w3 + w4) * price_per_kg = 99 :=
by
  sorry

#check turkey_cost

end NUMINAMATH_CALUDE_turkey_cost_l3008_300874


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l3008_300854

theorem smallest_satisfying_number : ∃ (n : ℕ), n = 1806 ∧ 
  (∀ (m : ℕ), m < n → 
    ∃ (p : ℕ), Prime p ∧ (m % (p - 1) = 0 → m % p ≠ 0)) ∧
  (∀ (p : ℕ), Prime p → (n % (p - 1) = 0 → n % p = 0)) := by
  sorry

#check smallest_satisfying_number

end NUMINAMATH_CALUDE_smallest_satisfying_number_l3008_300854


namespace NUMINAMATH_CALUDE_S_is_two_rays_with_common_endpoint_l3008_300891

/-- The set S of points (x, y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≥ 5) ∨
               (5 = y - 2 ∧ x + 3 ≥ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≥ x + 3)}

/-- Two rays with a common endpoint -/
def TwoRaysWithCommonEndpoint : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x = 2 ∧ y ≥ 7) ∨
               (y = 7 ∧ x ≥ 2)}

/-- Theorem stating that S is equivalent to two rays with a common endpoint -/
theorem S_is_two_rays_with_common_endpoint : S = TwoRaysWithCommonEndpoint := by
  sorry

end NUMINAMATH_CALUDE_S_is_two_rays_with_common_endpoint_l3008_300891


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l3008_300861

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h_red : prob_red = 0.42)
  (h_white : prob_white = 0.28)
  (h_sum : prob_red + prob_white + (1 - prob_red - prob_white) = 1) : 
  1 - prob_red - prob_white = 0.3 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l3008_300861


namespace NUMINAMATH_CALUDE_smallest_subset_size_for_divisibility_l3008_300841

theorem smallest_subset_size_for_divisibility : ∃ (n : ℕ),
  n = 337 ∧
  (∀ (S : Finset ℕ), S ⊆ Finset.range 2005 → S.card = n →
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 2004 ∣ (a^2 - b^2)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T ⊆ Finset.range 2005 ∧ T.card = m ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬(2004 ∣ (a^2 - b^2))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_subset_size_for_divisibility_l3008_300841


namespace NUMINAMATH_CALUDE_diane_gambling_problem_l3008_300846

theorem diane_gambling_problem (initial_amount : ℝ) : 
  (initial_amount + 65 + 50 = 215) → initial_amount = 100 := by
sorry

end NUMINAMATH_CALUDE_diane_gambling_problem_l3008_300846


namespace NUMINAMATH_CALUDE_valid_C_characterization_l3008_300809

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- A sequence is bounded below -/
def BoundedBelow (a : IntegerSequence) : Prop :=
  ∃ M : ℤ, ∀ n : ℕ, M ≤ a n

/-- A sequence satisfies the given inequality for a given C -/
def SatisfiesInequality (a : IntegerSequence) (C : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → (0 : ℝ) ≤ a (n - 1) + C * a n + a (n + 1) ∧ 
                   a (n - 1) + C * a n + a (n + 1) < 1

/-- A sequence is periodic -/
def Periodic (a : IntegerSequence) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a (n + p) = a n

/-- The set of all C that satisfy the conditions -/
def ValidC : Set ℝ :=
  {C : ℝ | ∀ a : IntegerSequence, BoundedBelow a → SatisfiesInequality a C → Periodic a}

theorem valid_C_characterization : ValidC = Set.Ici (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_valid_C_characterization_l3008_300809


namespace NUMINAMATH_CALUDE_power_division_rule_l3008_300830

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3008_300830


namespace NUMINAMATH_CALUDE_sqrt_square_789256_l3008_300865

theorem sqrt_square_789256 : (Real.sqrt 789256)^2 = 789256 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_789256_l3008_300865


namespace NUMINAMATH_CALUDE_number_puzzle_l3008_300825

theorem number_puzzle (x : ℝ) : (x / 8) - 160 = 12 → x = 1376 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3008_300825


namespace NUMINAMATH_CALUDE_max_container_weight_l3008_300875

def can_transport (k : ℕ) : Prop :=
  ∀ (distribution : List ℕ),
    (distribution.sum = 1500) →
    (∀ x ∈ distribution, x ≤ k ∧ x > 0) →
    ∃ (platform_loads : List ℕ),
      (platform_loads.length = 25) ∧
      (∀ load ∈ platform_loads, load ≤ 80) ∧
      (platform_loads.sum = 1500)

theorem max_container_weight :
  (can_transport 26) ∧ ¬(can_transport 27) := by sorry

end NUMINAMATH_CALUDE_max_container_weight_l3008_300875


namespace NUMINAMATH_CALUDE_sons_age_l3008_300894

theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3008_300894


namespace NUMINAMATH_CALUDE_selection_problem_l3008_300813

theorem selection_problem (n r : ℕ) (h : r < n) :
  -- Number of ways to select r people from 2n people in a row with no adjacent selections
  (Nat.choose (2*n - r + 1) r) = 
    (Nat.choose (2*n - r + 1) r) ∧
  -- Number of ways to select r people from 2n people in a circle with no adjacent selections
  ((2*n : ℚ) / (2*n - r : ℚ)) * (Nat.choose (2*n - r) r) = 
    ((2*n : ℚ) / (2*n - r : ℚ)) * (Nat.choose (2*n - r) r) := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l3008_300813


namespace NUMINAMATH_CALUDE_train_length_problem_l3008_300878

theorem train_length_problem (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 144) :
  let rel_speed := (v_fast - v_slow) * (5 / 18)
  let train_length := rel_speed * t / 2
  train_length = 200 := by
sorry

end NUMINAMATH_CALUDE_train_length_problem_l3008_300878


namespace NUMINAMATH_CALUDE_first_sample_in_systematic_sampling_l3008_300889

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => firstSample + (n - 1) * (total / sampleSize)

theorem first_sample_in_systematic_sampling
  (total : ℕ) (sampleSize : ℕ) (fourthSample : ℕ) 
  (h1 : total = 800)
  (h2 : sampleSize = 80)
  (h3 : fourthSample = 39) :
  ∃ firstSample : ℕ, 
    firstSample ∈ Finset.range 10 ∧ 
    systematicSample total sampleSize firstSample 4 = fourthSample ∧
    firstSample = 9 :=
by sorry

end NUMINAMATH_CALUDE_first_sample_in_systematic_sampling_l3008_300889


namespace NUMINAMATH_CALUDE_correct_arrangements_l3008_300888

/-- Represents a student with a grade -/
structure Student where
  grade : Nat

/-- Represents a car with students -/
structure Car where
  students : Finset Student

/-- The total number of students -/
def totalStudents : Nat := 8

/-- The number of grades -/
def numGrades : Nat := 4

/-- The number of students per grade -/
def studentsPerGrade : Nat := 2

/-- The number of students per car -/
def studentsPerCar : Nat := 4

/-- Twin sisters from first grade -/
def twinSisters : Finset Student := sorry

/-- All students -/
def allStudents : Finset Student := sorry

/-- Checks if a car has exactly two students from the same grade -/
def hasTwoSameGrade (car : Car) : Prop := sorry

/-- The number of ways to arrange students in car A -/
def numArrangements : Nat := sorry

/-- Main theorem -/
theorem correct_arrangements :
  numArrangements = 24 := by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l3008_300888


namespace NUMINAMATH_CALUDE_prob_ace_then_diamond_standard_deck_l3008_300858

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (diamonds : Nat)
  (ace_of_diamonds : Nat)

/-- Probability of drawing an Ace first and a diamond second from a standard deck -/
def prob_ace_then_diamond (d : Deck) : ℚ :=
  let prob_ace_of_diamonds := d.ace_of_diamonds / d.cards
  let prob_other_ace := (d.aces - d.ace_of_diamonds) / d.cards
  let prob_diamond_after_ace_of_diamonds := (d.diamonds - 1) / (d.cards - 1)
  let prob_diamond_after_other_ace := d.diamonds / (d.cards - 1)
  prob_ace_of_diamonds * prob_diamond_after_ace_of_diamonds +
  prob_other_ace * prob_diamond_after_other_ace

theorem prob_ace_then_diamond_standard_deck :
  prob_ace_then_diamond { cards := 52, aces := 4, diamonds := 13, ace_of_diamonds := 1 } = 119 / 3571 :=
sorry

end NUMINAMATH_CALUDE_prob_ace_then_diamond_standard_deck_l3008_300858


namespace NUMINAMATH_CALUDE_expression_factorization_l3008_300801

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x^2 - 3) - (-3 * x^3 + 6 * x^2 - 3) = 3 * x^2 * (5 * x + 13) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3008_300801


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3008_300859

theorem arithmetic_expression_evaluation :
  65 + (126 / 14) + (35 * 11) - 250 - (500 / 5)^2 = -9791 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3008_300859


namespace NUMINAMATH_CALUDE_work_time_ratio_l3008_300802

theorem work_time_ratio (a b : ℝ) (h1 : b = 18) (h2 : 1/a + 1/b = 1/3) :
  a / b = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_work_time_ratio_l3008_300802


namespace NUMINAMATH_CALUDE_min_sum_distances_l3008_300806

theorem min_sum_distances (a b : ℝ) :
  Real.sqrt ((a - 1)^2 + (b - 1)^2) + Real.sqrt ((a + 1)^2 + (b + 1)^2) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_distances_l3008_300806


namespace NUMINAMATH_CALUDE_distance_from_center_to_chords_l3008_300849

/-- A circle with two chords drawn through the ends of a diameter -/
structure CircleWithChords where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first chord -/
  chord1_length : ℝ
  /-- The length of the second chord -/
  chord2_length : ℝ
  /-- The chords intersect on the circumference -/
  chords_intersect_on_circumference : True
  /-- The chords are drawn through the ends of a diameter -/
  chords_through_diameter_ends : True
  /-- The first chord has length 12 -/
  chord1_is_12 : chord1_length = 12
  /-- The second chord has length 16 -/
  chord2_is_16 : chord2_length = 16

/-- The theorem stating the distances from the center to the chords -/
theorem distance_from_center_to_chords (c : CircleWithChords) :
  ∃ (d1 d2 : ℝ), d1 = 8 ∧ d2 = 6 ∧
  d1 = c.chord2_length / 2 ∧
  d2 = c.chord1_length / 2 :=
sorry

end NUMINAMATH_CALUDE_distance_from_center_to_chords_l3008_300849


namespace NUMINAMATH_CALUDE_smallest_w_l3008_300867

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^4) (1452 * w) → 
  is_factor (3^3) (1452 * w) → 
  is_factor (13^3) (1452 * w) → 
  w ≥ 79132 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l3008_300867


namespace NUMINAMATH_CALUDE_set_equation_solution_l3008_300895

theorem set_equation_solution (p q : ℝ) : 
  let M := {x : ℝ | x^2 + p*x - 2 = 0}
  let N := {x : ℝ | x^2 - 2*x + q = 0}
  (M ∪ N = {-1, 0, 2}) → (p = -1 ∧ q = 0) := by
sorry

end NUMINAMATH_CALUDE_set_equation_solution_l3008_300895


namespace NUMINAMATH_CALUDE_carlos_pesos_sum_of_digits_l3008_300884

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the exchange rate from dollars to pesos -/
def exchangeRate : ℚ := 12 / 8

theorem carlos_pesos_sum_of_digits :
  ∀ d : ℕ,
  (exchangeRate * d - 72 : ℚ) = d →
  sumOfDigits d = 9 := by sorry

end NUMINAMATH_CALUDE_carlos_pesos_sum_of_digits_l3008_300884


namespace NUMINAMATH_CALUDE_evaluate_expression_l3008_300898

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3008_300898


namespace NUMINAMATH_CALUDE_percent_equality_l3008_300835

theorem percent_equality (x : ℝ) : (60 / 100 * 600 = 50 / 100 * x) → x = 720 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3008_300835


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_20_l3008_300883

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n : ℕ | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers with sum 20 -/
theorem unique_consecutive_sum_20 : 
  ∃! p : ℕ × ℕ, 2 ≤ p.2 ∧ ConsecutiveSum p.1 p.2 = 20 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_20_l3008_300883


namespace NUMINAMATH_CALUDE_festival_lineup_theorem_l3008_300866

/-- The minimum number of Gennadys required for the festival lineup -/
def min_gennadys (num_alexanders num_borises num_vasilys : ℕ) : ℕ :=
  max 0 (num_borises - 1 - (num_alexanders + num_vasilys))

/-- Theorem stating the minimum number of Gennadys required for the festival lineup -/
theorem festival_lineup_theorem (num_alexanders num_borises num_vasilys : ℕ) 
  (h1 : num_alexanders = 45)
  (h2 : num_borises = 122)
  (h3 : num_vasilys = 27) :
  min_gennadys num_alexanders num_borises num_vasilys = 49 := by
  sorry

#eval min_gennadys 45 122 27

end NUMINAMATH_CALUDE_festival_lineup_theorem_l3008_300866


namespace NUMINAMATH_CALUDE_kiran_work_completion_l3008_300805

/-- Given that Kiran completes 1/3 of the work in 6 days, prove that he will finish the remaining work in 12 days. -/
theorem kiran_work_completion (work_rate : ℝ) (h1 : work_rate * 6 = 1/3) : 
  work_rate * 12 = 2/3 := by sorry

end NUMINAMATH_CALUDE_kiran_work_completion_l3008_300805


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l3008_300831

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l3008_300831


namespace NUMINAMATH_CALUDE_polynomial_sum_l3008_300808

-- Define the polynomials
def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3008_300808


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3008_300814

-- Define the triangle
structure ObtuseIsoscelesTriangle where
  -- The largest angle in degrees
  largest_angle : ℝ
  -- One of the two equal angles in degrees
  equal_angle : ℝ
  -- Conditions
  is_obtuse : largest_angle > 90
  is_isosceles : equal_angle = equal_angle
  angle_sum : largest_angle + 2 * equal_angle = 180

-- Theorem statement
theorem smallest_angle_measure (t : ObtuseIsoscelesTriangle) 
  (h : t.largest_angle = 108) : t.equal_angle = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3008_300814


namespace NUMINAMATH_CALUDE_radius_B_is_three_fifths_l3008_300822

/-- A structure representing the configuration of circles A, B, C, and D. -/
structure CircleConfiguration where
  /-- Radius of circle A -/
  radius_A : ℝ
  /-- Radius of circle B -/
  radius_B : ℝ
  /-- Radius of circle D -/
  radius_D : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externally_tangent : Prop
  /-- Circles A, B, and C are internally tangent to circle D -/
  internally_tangent : Prop
  /-- Circles B and C are congruent -/
  B_C_congruent : Prop
  /-- The center of D is tangent to circle A at one point -/
  D_center_tangent_A : Prop

/-- Theorem stating that given the specific configuration of circles, the radius of circle B is 3/5. -/
theorem radius_B_is_three_fifths (config : CircleConfiguration)
  (h1 : config.radius_A = 2)
  (h2 : config.radius_D = 3) :
  config.radius_B = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_radius_B_is_three_fifths_l3008_300822


namespace NUMINAMATH_CALUDE_complement_of_57_13_l3008_300871

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

/-- The main theorem stating that the complement of 57°13' is 32°47' -/
theorem complement_of_57_13 :
  complement { degrees := 57, minutes := 13, valid := by sorry } =
  { degrees := 32, minutes := 47, valid := by sorry } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_57_13_l3008_300871


namespace NUMINAMATH_CALUDE_inequality_proof_l3008_300811

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_min : min (a * b) (min (b * c) (c * a)) ≥ 1) :
  (((a^2 + 1) * (b^2 + 1) * (c^2 + 1))^(1/3) : ℝ) ≤ ((a + b + c) / 3)^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3008_300811


namespace NUMINAMATH_CALUDE_triple_equation_solution_l3008_300817

theorem triple_equation_solution :
  ∀ (a b c : ℝ), 
    ((2*a+1)^2 - 4*b = 5 ∧ 
     (2*b+1)^2 - 4*c = 5 ∧ 
     (2*c+1)^2 - 4*a = 5) ↔ 
    ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = -1 ∧ b = -1 ∧ c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_triple_equation_solution_l3008_300817


namespace NUMINAMATH_CALUDE_cafeteria_earnings_l3008_300872

/-- Calculates the total earnings from selling apples and oranges in a cafeteria. -/
theorem cafeteria_earnings (initial_apples initial_oranges : ℕ)
                           (apple_price orange_price : ℚ)
                           (remaining_apples remaining_oranges : ℕ)
                           (h1 : initial_apples = 50)
                           (h2 : initial_oranges = 40)
                           (h3 : apple_price = 0.80)
                           (h4 : orange_price = 0.50)
                           (h5 : remaining_apples = 10)
                           (h6 : remaining_oranges = 6) :
  (initial_apples - remaining_apples) * apple_price +
  (initial_oranges - remaining_oranges) * orange_price = 49 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_earnings_l3008_300872


namespace NUMINAMATH_CALUDE_j_mod_2_not_zero_l3008_300868

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_j_mod_2_not_zero_l3008_300868


namespace NUMINAMATH_CALUDE_product_remainder_mod_ten_l3008_300845

theorem product_remainder_mod_ten (a b c : ℕ) : 
  a % 10 = 7 → b % 10 = 1 → c % 10 = 3 → (a * b * c) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_ten_l3008_300845


namespace NUMINAMATH_CALUDE_sock_selection_with_red_l3008_300840

def total_socks : ℕ := 7
def socks_to_select : ℕ := 3

theorem sock_selection_with_red (total_socks : ℕ) (socks_to_select : ℕ) : 
  total_socks = 7 → socks_to_select = 3 → 
  (Nat.choose total_socks socks_to_select) - (Nat.choose (total_socks - 1) socks_to_select) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_with_red_l3008_300840


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3008_300880

theorem binomial_coefficient_congruence (p n : ℕ) (hp : Prime p) :
  (Nat.choose n p) ≡ (n / p : ℕ) [MOD p] := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3008_300880


namespace NUMINAMATH_CALUDE_soccer_league_games_l3008_300881

/-- The number of games played in a soccer league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 10 teams, where each team plays every other team once, 
    the total number of games played is 45 -/
theorem soccer_league_games : games_played 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3008_300881


namespace NUMINAMATH_CALUDE_dividend_division_theorem_l3008_300832

theorem dividend_division_theorem : ∃ (q r : ℕ), 
  220030 = (555 + 445) * q + r ∧ 
  r < (555 + 445) ∧ 
  r = 30 ∧ 
  q = 2 * (555 - 445) :=
by sorry

end NUMINAMATH_CALUDE_dividend_division_theorem_l3008_300832


namespace NUMINAMATH_CALUDE_multiplication_mistake_difference_l3008_300863

theorem multiplication_mistake_difference : 
  let correct_multiplication := 137 * 43
  let mistaken_multiplication := 137 * 34
  correct_multiplication - mistaken_multiplication = 1233 := by
sorry

end NUMINAMATH_CALUDE_multiplication_mistake_difference_l3008_300863


namespace NUMINAMATH_CALUDE_third_angle_is_70_l3008_300828

-- Define a triangle type
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the sum of angles in a triangle
def sum_of_angles (t : Triangle) : Real :=
  t.angle1 + t.angle2 + t.angle3

-- Theorem statement
theorem third_angle_is_70 (t : Triangle) 
  (h1 : t.angle1 = 50)
  (h2 : t.angle2 = 60)
  (h3 : sum_of_angles t = 180) : 
  t.angle3 = 70 := by
sorry


end NUMINAMATH_CALUDE_third_angle_is_70_l3008_300828


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3008_300896

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80*x + 1551) / (x^2 + 57*x - 2970)) →
  α + β = 137 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3008_300896


namespace NUMINAMATH_CALUDE_parallel_vector_implies_zero_y_coordinate_l3008_300818

/-- Given vectors a and b in R², if b - a is parallel to a, then the y-coordinate of b is 0 -/
theorem parallel_vector_implies_zero_y_coordinate (m n : ℝ) :
  let a : Fin 2 → ℝ := ![1, 0]
  let b : Fin 2 → ℝ := ![m, n]
  (∃ (k : ℝ), (b - a) = k • a) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_implies_zero_y_coordinate_l3008_300818


namespace NUMINAMATH_CALUDE_amount_after_two_years_l3008_300855

theorem amount_after_two_years
  (initial_amount : ℝ)
  (annual_rate : ℝ)
  (years : ℕ)
  (h1 : initial_amount = 51200)
  (h2 : annual_rate = 1 / 8)
  (h3 : years = 2) :
  initial_amount * (1 + annual_rate) ^ years = 64800 :=
by sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3008_300855


namespace NUMINAMATH_CALUDE_factory_shutdown_probabilities_l3008_300839

/-- The number of factories -/
def num_factories : ℕ := 5

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The probability of all factories choosing Sunday to shut down -/
def prob_all_sunday : ℚ := 1 / 7^num_factories

/-- The probability of at least two factories choosing the same day to shut down -/
def prob_at_least_two_same : ℚ := 1 - (num_days.factorial / (num_days - num_factories).factorial) / 7^num_factories

theorem factory_shutdown_probabilities :
  (prob_all_sunday = 1 / 16807) ∧
  (prob_at_least_two_same = 2041 / 2401) := by
  sorry


end NUMINAMATH_CALUDE_factory_shutdown_probabilities_l3008_300839


namespace NUMINAMATH_CALUDE_equation_solution_l3008_300864

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ x ≠ -6 ∧
  (3*x - 6) / (x^2 + 5*x - 6) = (x + 3) / (x - 1) ∧
  x = 9/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3008_300864


namespace NUMINAMATH_CALUDE_gwens_birthday_money_l3008_300851

theorem gwens_birthday_money (mom_gift dad_gift : ℕ) 
  (h1 : mom_gift = 8)
  (h2 : dad_gift = 5) :
  mom_gift - dad_gift = 3 := by
  sorry

end NUMINAMATH_CALUDE_gwens_birthday_money_l3008_300851


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3008_300876

theorem power_of_three_plus_five_mod_eight :
  (3^100 + 5) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3008_300876


namespace NUMINAMATH_CALUDE_quadratic_always_positive_range_l3008_300890

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_range_l3008_300890


namespace NUMINAMATH_CALUDE_unique_nonzero_solution_sum_of_squares_l3008_300873

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * y - 2 * y - 3 * x = 0
def equation2 (y z : ℝ) : Prop := y * z - 3 * z - 5 * y = 0
def equation3 (x z : ℝ) : Prop := x * z - 5 * x - 2 * z = 0

-- Define the theorem
theorem unique_nonzero_solution_sum_of_squares :
  ∃! (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    equation1 a b ∧ equation2 b c ∧ equation3 a c →
    a^2 + b^2 + c^2 = 152 :=
by sorry

end NUMINAMATH_CALUDE_unique_nonzero_solution_sum_of_squares_l3008_300873


namespace NUMINAMATH_CALUDE_part_one_part_two_l3008_300838

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Part 1
theorem part_one (a : ℝ) : (∀ y ≥ 0, ∃ x, f a x = y) ∧ (∀ x, f a x ≥ 0) → a = 3/2 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x, f a x ≥ 0) → 
  (∀ y ∈ Set.Icc (-19/4) (-2), ∃ a ∈ Set.Icc (-1) (3/2), g a = y) ∧ 
  (∀ a ∈ Set.Icc (-1) (3/2), g a ∈ Set.Icc (-19/4) (-2)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3008_300838


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l3008_300807

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 10 →
  num_slices = 10 →
  (π * (diameter/2)^2 * thickness) / num_slices = 5*π/4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l3008_300807


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l3008_300810

/-- 
Given two lines in the xy-plane defined by their equations,
this theorem states that if these lines are perpendicular,
then the parameter m must equal 1/2.
-/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - m * y + 2 * m = 0 → x + 2 * y - m = 0 → 
    (1 : ℝ) / m * (-1 / 2 : ℝ) = -1) → 
  m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l3008_300810


namespace NUMINAMATH_CALUDE_optimal_purchase_is_cheapest_l3008_300824

/-- Park admission fee per person -/
def individual_fee : ℕ := 5

/-- Group ticket fee -/
def group_fee : ℕ := 40

/-- Maximum number of people allowed per group ticket -/
def group_max : ℕ := 10

/-- Cost function for purchasing tickets -/
def ticket_cost (group_tickets : ℕ) (individual_tickets : ℕ) : ℕ :=
  group_tickets * group_fee + individual_tickets * individual_fee

/-- The most economical way to purchase tickets -/
def optimal_purchase (x : ℕ) : ℕ × ℕ :=
  let a := x / group_max
  let b := x % group_max
  if b < 8 then (a, b)
  else if b = 8 then (a, 8)  -- or (a + 1, 0), both are optimal
  else (a + 1, 0)

theorem optimal_purchase_is_cheapest (x : ℕ) :
  let (g, i) := optimal_purchase x
  ∀ (g' i' : ℕ), g' * group_max + i' ≥ x →
    ticket_cost g i ≤ ticket_cost g' i' :=
sorry

end NUMINAMATH_CALUDE_optimal_purchase_is_cheapest_l3008_300824


namespace NUMINAMATH_CALUDE_tree_rings_l3008_300842

theorem tree_rings (thin_rings : ℕ) : 
  (∀ (fat_rings : ℕ), fat_rings = 2) →
  (70 * (fat_rings + thin_rings) = 40 * (fat_rings + thin_rings) + 180) →
  thin_rings = 4 := by
sorry

end NUMINAMATH_CALUDE_tree_rings_l3008_300842


namespace NUMINAMATH_CALUDE_pages_to_read_tonight_l3008_300816

def pages_three_nights_ago : ℕ := 20

def pages_two_nights_ago (x : ℕ) : ℕ := x^2 + 5

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + sum_of_digits (n / 10))

def pages_last_night (x : ℕ) : ℕ := 3 * sum_of_digits x

def total_pages : ℕ := 500

theorem pages_to_read_tonight : 
  total_pages - (pages_three_nights_ago + 
                 pages_two_nights_ago pages_three_nights_ago + 
                 pages_last_night (pages_two_nights_ago pages_three_nights_ago)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_tonight_l3008_300816


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_4_l3008_300887

theorem binomial_coefficient_16_4 : Nat.choose 16 4 = 1820 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_4_l3008_300887


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l3008_300848

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = 2*x + 3

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = (3/4) * x
def asymptote2 (x y : ℝ) : Prop := y = -(3/4) * x

-- Theorem statement
theorem hyperbola_asymptote_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    asymptote1 x1 y1 ∧ line x1 y1 ∧ 
    asymptote2 x2 y2 ∧ line x2 y2 ∧
    x1 = -12/5 ∧ y1 = -9/5 ∧
    x2 = -12/11 ∧ y2 = 9/11 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l3008_300848


namespace NUMINAMATH_CALUDE_sqrt_90000_equals_300_l3008_300879

theorem sqrt_90000_equals_300 : Real.sqrt 90000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_90000_equals_300_l3008_300879


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3008_300850

theorem polynomial_simplification (x : ℝ) : 
  (x^5 + x^4 + x + 10) - (x^5 + 2*x^4 - x^3 + 12) = -x^4 + x^3 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3008_300850


namespace NUMINAMATH_CALUDE_overlap_percentage_l3008_300856

theorem overlap_percentage (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 10 →
  rect_length = 18 →
  rect_width = 10 →
  (2 * square_side - rect_length) * rect_width / (rect_length * rect_width) * 100 = 11.11 := by
sorry

end NUMINAMATH_CALUDE_overlap_percentage_l3008_300856


namespace NUMINAMATH_CALUDE_algebraic_expression_values_l3008_300892

theorem algebraic_expression_values (p q : ℝ) :
  (p * 1^3 + q * 1 + 1 = 2023) →
  (p * (-1)^3 + q * (-1) + 1 = -2021) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_values_l3008_300892


namespace NUMINAMATH_CALUDE_room_dimension_is_15_l3008_300886

/-- Represents the dimensions and properties of a room to be whitewashed -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  doorArea : ℝ
  windowArea : ℝ
  windowCount : ℕ
  whitewashCost : ℝ
  totalCost : ℝ

/-- Calculates the total area to be whitewashed in the room -/
def areaToWhitewash (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - (r.doorArea + r.windowCount * r.windowArea)

/-- Theorem stating that the unknown dimension of the room is 15 feet -/
theorem room_dimension_is_15 (r : Room) 
  (h1 : r.length = 25)
  (h2 : r.height = 12)
  (h3 : r.doorArea = 18)
  (h4 : r.windowArea = 12)
  (h5 : r.windowCount = 3)
  (h6 : r.whitewashCost = 5)
  (h7 : r.totalCost = 4530)
  (h8 : r.totalCost = r.whitewashCost * areaToWhitewash r) :
  r.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_is_15_l3008_300886


namespace NUMINAMATH_CALUDE_changhyeon_money_problem_l3008_300860

theorem changhyeon_money_problem (initial_money : ℕ) : 
  (initial_money / 2 - 300) / 2 - 400 = 0 → initial_money = 2200 := by
  sorry

end NUMINAMATH_CALUDE_changhyeon_money_problem_l3008_300860


namespace NUMINAMATH_CALUDE_expression_simplification_l3008_300885

theorem expression_simplification (a b : ℚ) (ha : a = -2) (hb : b = 3) :
  (((a - b) / (a^2 - 2*a*b + b^2) - a / (a^2 - 2*a*b)) / (b / (a - 2*b))) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3008_300885


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_necessary_not_sufficient_l3008_300843

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields

structure Plane3D where
  -- Add necessary fields

-- Define perpendicularity relations
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def perpendicular_plane_plane (p1 p2 : Plane3D) : Prop :=
  sorry

def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_plane_necessary_not_sufficient 
  (l : Line3D) (α : Plane3D) :
  (perpendicular_line_plane l α → 
    ∃ (p : Plane3D), plane_contains_line p l ∧ perpendicular_plane_plane p α) ∧
  ¬(∀ (l : Line3D) (α : Plane3D), 
    (∃ (p : Plane3D), plane_contains_line p l ∧ perpendicular_plane_plane p α) → 
    perpendicular_line_plane l α) :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_necessary_not_sufficient_l3008_300843


namespace NUMINAMATH_CALUDE_shaded_area_approx_l3008_300882

-- Define the circle and rectangle
def circle_radius : ℝ := 3
def rectangle_side_OA : ℝ := 2
def rectangle_side_AB : ℝ := 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (rectangle_side_OA, 0)
def B : ℝ × ℝ := (rectangle_side_OA, rectangle_side_AB)
def C : ℝ × ℝ := (0, rectangle_side_AB)

-- Define the function to calculate the area of the shaded region
def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_approx :
  abs (shaded_area - 6.23) < 0.01 := by sorry

end NUMINAMATH_CALUDE_shaded_area_approx_l3008_300882


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l3008_300899

theorem ratio_sum_theorem (a b c : ℝ) 
  (h : ∃ k : ℝ, a = 2*k ∧ b = 3*k ∧ c = 5*k) : (a + b) / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l3008_300899


namespace NUMINAMATH_CALUDE_num_non_congruent_triangles_l3008_300826

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The set of points in the 3x3 grid -/
def gridPoints : List Point := [
  ⟨0, 0⟩, ⟨0.5, 0⟩, ⟨1, 0⟩,
  ⟨0, 0.5⟩, ⟨0.5, 0.5⟩, ⟨1, 0.5⟩,
  ⟨0, 1⟩, ⟨0.5, 1⟩, ⟨1, 1⟩
]

/-- Predicate to check if two triangles are congruent -/
def areCongruent (t1 t2 : Triangle) : Prop := sorry

/-- The set of all possible triangles formed from the grid points -/
def allTriangles : List Triangle := sorry

/-- The set of non-congruent triangles -/
def nonCongruentTriangles : List Triangle := sorry

/-- Theorem: The number of non-congruent triangles is 3 -/
theorem num_non_congruent_triangles : 
  nonCongruentTriangles.length = 3 := by sorry

end NUMINAMATH_CALUDE_num_non_congruent_triangles_l3008_300826


namespace NUMINAMATH_CALUDE_area_transformation_l3008_300804

-- Define a function representing the area under a curve
noncomputable def area_under_curve (f : ℝ → ℝ) : ℝ := sorry

-- Define the original function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem area_transformation (h : area_under_curve g = 15) :
  area_under_curve (fun x ↦ 4 * g (2 * x - 4)) = 30 := by sorry

end NUMINAMATH_CALUDE_area_transformation_l3008_300804


namespace NUMINAMATH_CALUDE_percentage_problem_l3008_300897

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 24 + 0.1 * 40 = 5.92 ↔ P = 8 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3008_300897


namespace NUMINAMATH_CALUDE_volume_of_specific_open_box_l3008_300852

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem stating that the volume of the specific open box is 5120 m³. -/
theorem volume_of_specific_open_box :
  openBoxVolume 48 36 8 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_open_box_l3008_300852


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3008_300829

/-- Given a quadratic equation x^2 - 4x = 5, prove that its standard form coefficients are 1, -4, and -5 -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 - 4*x = 5 ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -4 ∧ c = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3008_300829


namespace NUMINAMATH_CALUDE_grape_sales_properties_l3008_300800

/-- Represents the properties of the grape sales scenario -/
structure GrapeSales where
  initial_price : ℝ
  initial_volume : ℝ
  cost_price : ℝ
  price_reduction_effect : ℝ

/-- Calculates the daily sales profit for a given price reduction -/
def daily_profit (g : GrapeSales) (price_reduction : ℝ) : ℝ :=
  let new_price := g.initial_price - price_reduction
  let new_volume := g.initial_volume + price_reduction * g.price_reduction_effect
  (new_price - g.cost_price) * new_volume

/-- Calculates the profit as a function of selling price -/
def profit_function (g : GrapeSales) (x : ℝ) : ℝ :=
  (x - g.cost_price) * (g.initial_volume + (g.initial_price - x) * g.price_reduction_effect)

/-- Theorem stating the properties of the grape sales scenario -/
theorem grape_sales_properties (g : GrapeSales) 
  (h1 : g.initial_price = 30)
  (h2 : g.initial_volume = 60)
  (h3 : g.cost_price = 15)
  (h4 : g.price_reduction_effect = 10) :
  daily_profit g 2 = 1040 ∧ 
  (∃ (x : ℝ), x = 51/2 ∧ ∀ (y : ℝ), profit_function g y ≤ profit_function g x) ∧
  (∃ (max_profit : ℝ), max_profit = 1102.5 ∧ 
    ∀ (y : ℝ), profit_function g y ≤ max_profit) := by
  sorry

end NUMINAMATH_CALUDE_grape_sales_properties_l3008_300800


namespace NUMINAMATH_CALUDE_min_sum_of_distinct_integers_with_odd_square_sums_l3008_300837

theorem min_sum_of_distinct_integers_with_odd_square_sums (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (m n p : ℕ), 
    (a + b = 2 * m + 1) ∧ (a + c = 2 * n + 1) ∧ (a + d = 2 * p + 1) ∧
    (∃ (x y z : ℕ), (2 * m + 1 = x^2) ∧ (2 * n + 1 = y^2) ∧ (2 * p + 1 = z^2)) →
  10 * (a + b + c + d) ≥ 670 :=
by sorry

#check min_sum_of_distinct_integers_with_odd_square_sums

end NUMINAMATH_CALUDE_min_sum_of_distinct_integers_with_odd_square_sums_l3008_300837
