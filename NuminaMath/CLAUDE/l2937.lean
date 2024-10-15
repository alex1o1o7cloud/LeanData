import Mathlib

namespace NUMINAMATH_CALUDE_donation_scientific_correct_l2937_293700

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The donation amount in yuan -/
def donation_amount : ℝ := 2175000000

/-- The scientific notation of the donation amount -/
def donation_scientific : ScientificNotation := {
  coefficient := 2.175,
  exponent := 9,
  valid := by sorry
}

/-- Theorem stating that the donation amount is correctly represented in scientific notation -/
theorem donation_scientific_correct : 
  donation_amount = donation_scientific.coefficient * (10 : ℝ) ^ donation_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_donation_scientific_correct_l2937_293700


namespace NUMINAMATH_CALUDE_average_cost_per_meter_l2937_293701

def silk_length : Real := 9.25
def silk_cost : Real := 416.25
def cotton_length : Real := 7.5
def cotton_cost : Real := 337.50
def wool_length : Real := 6
def wool_cost : Real := 378

def total_length : Real := silk_length + cotton_length + wool_length
def total_cost : Real := silk_cost + cotton_cost + wool_cost

theorem average_cost_per_meter : total_cost / total_length = 49.75 := by sorry

end NUMINAMATH_CALUDE_average_cost_per_meter_l2937_293701


namespace NUMINAMATH_CALUDE_square_octagon_tessellation_l2937_293708

-- Define the internal angles of regular polygons
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def hexagon_angle : ℝ := 120
def octagon_angle : ℝ := 135

-- Define a predicate for seamless tessellation
def can_tessellate (angle1 angle2 : ℝ) : Prop :=
  ∃ (n m : ℕ), n * angle1 + m * angle2 = 360

-- Theorem statement
theorem square_octagon_tessellation :
  can_tessellate square_angle octagon_angle ∧
  ¬can_tessellate square_angle hexagon_angle ∧
  ¬can_tessellate square_angle pentagon_angle ∧
  ¬can_tessellate hexagon_angle octagon_angle ∧
  ¬can_tessellate pentagon_angle octagon_angle :=
sorry

end NUMINAMATH_CALUDE_square_octagon_tessellation_l2937_293708


namespace NUMINAMATH_CALUDE_driver_stops_theorem_l2937_293788

/-- Calculates the number of stops a delivery driver needs to make -/
def num_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) : ℕ :=
  total_boxes / boxes_per_stop

theorem driver_stops_theorem :
  let total_boxes : ℕ := 27
  let boxes_per_stop : ℕ := 9
  num_stops total_boxes boxes_per_stop = 3 := by
sorry

end NUMINAMATH_CALUDE_driver_stops_theorem_l2937_293788


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l2937_293739

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define the condition that y decreases as x increases
def decreasing_y (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

-- Define what it means for a point to be in Quadrant III
def in_quadrant_III (x : ℝ) (y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- Theorem statement
theorem linear_function_not_in_quadrant_III (k : ℝ) :
  decreasing_y (linear_function k) →
  ¬∃ x, in_quadrant_III x (linear_function k x) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l2937_293739


namespace NUMINAMATH_CALUDE_half_of_number_l2937_293768

theorem half_of_number (N : ℚ) (h : (4/15 : ℚ) * (5/7 : ℚ) * N = (4/9 : ℚ) * (2/5 : ℚ) * N + 8) : 
  N / 2 = 315 := by
  sorry

end NUMINAMATH_CALUDE_half_of_number_l2937_293768


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2937_293757

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := x = k*y - 1

-- Define the intersection points
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the reflection point
def reflection_point (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, -y₁)

-- Theorem statement
theorem ellipse_intersection_fixed_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points k x₁ y₁ x₂ y₂ →
  let (x₁', y₁') := reflection_point x₁ y₁
  (x₁' ≠ x₂ ∨ y₁' ≠ y₂) →
  ∃ (t : ℝ), t * (x₂ - x₁') + x₁' = -4 ∧ t * (y₂ - y₁') + y₁' = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2937_293757


namespace NUMINAMATH_CALUDE_problem_statement_l2937_293784

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) : 
  a + b^2 + c^3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2937_293784


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2937_293718

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line with slope m and y-intercept -3
def line (m x : ℝ) : ℝ := m * x - 3

-- Define the set of valid slopes
def valid_slopes : Set ℝ := {m : ℝ | m ≤ -Real.sqrt (4/55) ∨ m ≥ Real.sqrt (4/55)}

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x : ℝ, ellipse x (line m x)) ↔ m ∈ valid_slopes :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2937_293718


namespace NUMINAMATH_CALUDE_min_trips_for_elevator_l2937_293744

def weights : List ℕ := [50, 51, 55, 57, 58, 59, 60, 63, 75, 140]
def max_capacity : ℕ := 180

def is_valid_trip (trip : List ℕ) : Prop :=
  trip.sum ≤ max_capacity

def covers_all_weights (trips : List (List ℕ)) : Prop :=
  weights.all (λ w => ∃ t ∈ trips, w ∈ t)

theorem min_trips_for_elevator : 
  ∃ (trips : List (List ℕ)), 
    trips.length = 4 ∧ 
    (∀ t ∈ trips, is_valid_trip t) ∧
    covers_all_weights trips ∧
    (∀ (other_trips : List (List ℕ)), 
      (∀ t ∈ other_trips, is_valid_trip t) → 
      covers_all_weights other_trips → 
      other_trips.length ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_min_trips_for_elevator_l2937_293744


namespace NUMINAMATH_CALUDE_extremal_values_sum_l2937_293743

/-- Given real numbers x and y satisfying 4x^2 - 5xy + 4y^2 = 5, 
    S_max and S_min are the maximum and minimum values of x^2 + y^2 respectively. -/
theorem extremal_values_sum (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let S := x^2 + y^2
  let S_max := (10 : ℝ) / 3
  let S_min := (10 : ℝ) / 13
  (1 / S_max) + (1 / S_min) = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_extremal_values_sum_l2937_293743


namespace NUMINAMATH_CALUDE_ones_divisible_by_27_l2937_293740

def ones_number (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem ones_divisible_by_27 :
  ∃ k : ℕ, ones_number 27 = 27 * k :=
sorry

end NUMINAMATH_CALUDE_ones_divisible_by_27_l2937_293740


namespace NUMINAMATH_CALUDE_train_length_l2937_293747

/-- The length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : Real) (platform_length : Real) (crossing_time : Real) :
  train_speed = 72 / 3.6 →
  platform_length = 50.024 →
  crossing_time = 15 →
  train_speed * crossing_time - platform_length = 249.976 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2937_293747


namespace NUMINAMATH_CALUDE_cubic_difference_l2937_293782

theorem cubic_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2937_293782


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2937_293704

-- Problem 1
theorem factorization_problem_1 (a b x y : ℝ) :
  a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  a^3 + 2*a^2*b + a*b^2 = a * (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2937_293704


namespace NUMINAMATH_CALUDE_not_divisible_by_67_l2937_293781

theorem not_divisible_by_67 (x y : ℕ) 
  (h1 : ¬(67 ∣ x))
  (h2 : ¬(67 ∣ y))
  (h3 : 67 ∣ (7*x + 32*y)) :
  ¬(67 ∣ (10*x + 17*y + 1)) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_67_l2937_293781


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2937_293786

def A : Set ℝ := {y | ∃ x, y = Real.cos x}
def B : Set ℝ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2937_293786


namespace NUMINAMATH_CALUDE_function_comparison_l2937_293735

theorem function_comparison (x₁ x₂ : ℝ) (h1 : x₁ < x₂) (h2 : x₁ + x₂ = 0) :
  let f := fun x => x^2 + 2*x + 4
  f x₁ < f x₂ := by
sorry

end NUMINAMATH_CALUDE_function_comparison_l2937_293735


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2937_293750

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3/2) + 3 * (1/6) + q = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2937_293750


namespace NUMINAMATH_CALUDE_perfect_score_l2937_293734

theorem perfect_score (perfect_score : ℕ) (h : 3 * perfect_score = 63) : perfect_score = 21 := by
  sorry

end NUMINAMATH_CALUDE_perfect_score_l2937_293734


namespace NUMINAMATH_CALUDE_loop_condition_proof_l2937_293793

theorem loop_condition_proof (i₀ S₀ : ℕ) (result : ℕ) : 
  i₀ = 12 → S₀ = 1 → result = 11880 →
  (∃ n : ℕ, result = Nat.factorial n - Nat.factorial (i₀ - 1)) →
  (∀ i S : ℕ, i > 9 ↔ result = S ∧ S = Nat.factorial i - Nat.factorial (i₀ - 1)) :=
by sorry

end NUMINAMATH_CALUDE_loop_condition_proof_l2937_293793


namespace NUMINAMATH_CALUDE_fourth_term_is_2016_l2937_293787

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  second_term : a 2 = 606
  sum_first_four : a 1 + a 2 + a 3 + a 4 = 3834

/-- The fourth term of the arithmetic sequence is 2016 -/
theorem fourth_term_is_2016 (seq : ArithmeticSequence) : seq.a 4 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_2016_l2937_293787


namespace NUMINAMATH_CALUDE_f_at_one_equals_neg_7007_l2937_293790

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 10
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 100*x + c

-- State the theorem
theorem f_at_one_equals_neg_7007 (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c 1 = -7007 :=
by sorry


end NUMINAMATH_CALUDE_f_at_one_equals_neg_7007_l2937_293790


namespace NUMINAMATH_CALUDE_aaron_remaining_erasers_l2937_293713

def initial_erasers : ℕ := 225
def given_to_doris : ℕ := 75
def given_to_ethan : ℕ := 40
def given_to_fiona : ℕ := 50

theorem aaron_remaining_erasers :
  initial_erasers - (given_to_doris + given_to_ethan + given_to_fiona) = 60 := by
  sorry

end NUMINAMATH_CALUDE_aaron_remaining_erasers_l2937_293713


namespace NUMINAMATH_CALUDE_linda_total_coins_l2937_293799

/-- Represents the number of coins Linda has -/
structure Coins where
  dimes : Nat
  quarters : Nat
  nickels : Nat

/-- Calculates the total number of coins -/
def totalCoins (c : Coins) : Nat :=
  c.dimes + c.quarters + c.nickels

/-- Linda's initial coins -/
def initialCoins : Coins :=
  { dimes := 2, quarters := 6, nickels := 5 }

/-- Coins given by Linda's mother -/
def givenCoins (initial : Coins) : Coins :=
  { dimes := 2, quarters := 10, nickels := 2 * initial.nickels }

/-- Linda's final coins after receiving coins from her mother -/
def finalCoins (initial : Coins) : Coins :=
  { dimes := initial.dimes + (givenCoins initial).dimes,
    quarters := initial.quarters + (givenCoins initial).quarters,
    nickels := initial.nickels + (givenCoins initial).nickels }

theorem linda_total_coins :
  totalCoins (finalCoins initialCoins) = 35 := by
  sorry

end NUMINAMATH_CALUDE_linda_total_coins_l2937_293799


namespace NUMINAMATH_CALUDE_sum_and_multiply_base8_l2937_293765

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 -/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Sums numbers from 1 to n in base 8 -/
def sumBase8 (n : ℕ) : ℕ := sorry

theorem sum_and_multiply_base8 :
  base10ToBase8 (3 * (sumBase8 (base8ToBase10 30))) = 1604 := by sorry

end NUMINAMATH_CALUDE_sum_and_multiply_base8_l2937_293765


namespace NUMINAMATH_CALUDE_rectangle_area_l2937_293773

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2937_293773


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2937_293778

/-- Represents a trapezoid ABCD with midline MN -/
structure Trapezoid where
  AD : ℝ  -- Length of side AD
  BC : ℝ  -- Length of side BC
  MN : ℝ  -- Length of midline MN
  is_trapezoid : AD ≠ BC  -- Ensures it's actually a trapezoid
  midline_property : MN = (AD + BC) / 2  -- Property of the midline

/-- Theorem: In a trapezoid with AD = 2 and MN = 6, BC must equal 10 -/
theorem trapezoid_side_length (T : Trapezoid) (h1 : T.AD = 2) (h2 : T.MN = 6) : T.BC = 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2937_293778


namespace NUMINAMATH_CALUDE_y_axis_reflection_l2937_293779

/-- Given a point P with coordinates (-5, 3), its reflection across the y-axis has coordinates (5, 3). -/
theorem y_axis_reflection :
  let P : ℝ × ℝ := (-5, 3)
  let P_reflected : ℝ × ℝ := (5, 3)
  P_reflected = (- P.1, P.2) :=
by sorry

end NUMINAMATH_CALUDE_y_axis_reflection_l2937_293779


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2937_293731

theorem simplify_square_roots : 
  Real.sqrt (10 + 6 * Real.sqrt 2) + Real.sqrt (10 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2937_293731


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l2937_293705

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l2937_293705


namespace NUMINAMATH_CALUDE_min_bananas_theorem_l2937_293766

/-- Represents the number of bananas a monkey takes from the pile -/
structure MonkeyTake where
  amount : ℕ

/-- Represents the final distribution of bananas among the monkeys -/
structure FinalDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of bananas in the pile -/
def totalBananas (t1 t2 t3 : MonkeyTake) : ℕ :=
  t1.amount + t2.amount + t3.amount

/-- Calculates the final distribution of bananas -/
def calculateDistribution (t1 t2 t3 : MonkeyTake) : FinalDistribution :=
  { first := 2 * t1.amount / 3 + t2.amount / 3 + 5 * t3.amount / 12
  , second := t1.amount / 6 + t2.amount / 3 + 5 * t3.amount / 12
  , third := t1.amount / 6 + t2.amount / 3 + t3.amount / 6 }

/-- Checks if the distribution satisfies the 4:3:2 ratio -/
def isValidRatio (d : FinalDistribution) : Prop :=
  3 * d.first = 4 * d.second ∧ 2 * d.second = 3 * d.third

/-- The main theorem stating the minimum number of bananas -/
theorem min_bananas_theorem (t1 t2 t3 : MonkeyTake) :
  (∀ d : FinalDistribution, d = calculateDistribution t1 t2 t3 → isValidRatio d) →
  totalBananas t1 t2 t3 ≥ 558 :=
sorry

end NUMINAMATH_CALUDE_min_bananas_theorem_l2937_293766


namespace NUMINAMATH_CALUDE_min_value_expression_l2937_293772

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 3 + 2*Real.sqrt 2 ∧ 
  ∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' + 2*y' + z' = 1 → 
    (1 / (x' + y')) + (2 / (y' + z')) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2937_293772


namespace NUMINAMATH_CALUDE_c_monthly_income_l2937_293783

/-- The monthly income ratio between A and B -/
def income_ratio : ℚ := 5 / 2

/-- The percentage increase of B's income over C's income -/
def income_increase_percentage : ℚ := 12 / 100

/-- A's annual income in rupees -/
def a_annual_income : ℕ := 470400

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating C's monthly income -/
theorem c_monthly_income :
  let a_monthly_income : ℚ := a_annual_income / months_per_year
  let b_monthly_income : ℚ := a_monthly_income / income_ratio
  let c_monthly_income : ℚ := b_monthly_income / (1 + income_increase_percentage)
  c_monthly_income = 14000 := by sorry

end NUMINAMATH_CALUDE_c_monthly_income_l2937_293783


namespace NUMINAMATH_CALUDE_stack_map_views_l2937_293725

def StackMap : Type := List (List Nat)

def frontView (sm : StackMap) : List Nat :=
  sm.map (List.foldl max 0)

def rightSideView (sm : StackMap) : List Nat :=
  List.map (List.foldl max 0) (List.transpose sm)

theorem stack_map_views (sm : StackMap) 
  (h1 : sm = [[3, 1, 2], [2, 4, 3], [1, 1, 3]]) : 
  frontView sm = [3, 4, 3] ∧ rightSideView sm = [3, 4, 3] := by
  sorry

end NUMINAMATH_CALUDE_stack_map_views_l2937_293725


namespace NUMINAMATH_CALUDE_dog_walking_distance_l2937_293764

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog2_daily_miles : ℝ) :
  total_weekly_miles = 70 →
  dog2_daily_miles = 8 →
  ∃ dog1_daily_miles : ℝ, 
    dog1_daily_miles * 7 + dog2_daily_miles * 7 = total_weekly_miles ∧
    dog1_daily_miles = 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_walking_distance_l2937_293764


namespace NUMINAMATH_CALUDE_complex_root_equation_l2937_293777

theorem complex_root_equation (p : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (3 - Complex.I : ℂ)^2 + p * (3 - Complex.I) + 10 = 0 →
  p = -6 := by
sorry

end NUMINAMATH_CALUDE_complex_root_equation_l2937_293777


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2937_293769

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 13 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 13

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2937_293769


namespace NUMINAMATH_CALUDE_coloring_books_total_l2937_293792

theorem coloring_books_total (initial : ℕ) (given_away : ℕ) (bought : ℕ) : 
  initial = 34 → given_away = 3 → bought = 48 → 
  initial - given_away + bought = 79 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_total_l2937_293792


namespace NUMINAMATH_CALUDE_sum_of_powers_l2937_293791

theorem sum_of_powers (n : ℕ) :
  n^5 + n^5 + n^5 + n^5 + n^5 = 5 * n^5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2937_293791


namespace NUMINAMATH_CALUDE_certain_number_minus_one_l2937_293730

theorem certain_number_minus_one (x : ℝ) (h : 15 * x = 45) : x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_one_l2937_293730


namespace NUMINAMATH_CALUDE_solve_grocery_problem_l2937_293755

def grocery_problem (total_budget : ℝ) (chicken_cost bacon_cost vegetable_cost : ℝ)
  (apple_cost : ℝ) (apple_count : ℕ) (hummus_count : ℕ) : Prop :=
  let remaining_after_meat_and_veg := total_budget - (chicken_cost + bacon_cost + vegetable_cost)
  let remaining_after_apples := remaining_after_meat_and_veg - (apple_cost * apple_count)
  let hummus_total_cost := remaining_after_apples
  let hummus_unit_cost := hummus_total_cost / hummus_count
  hummus_unit_cost = 5

theorem solve_grocery_problem :
  grocery_problem 60 20 10 10 2 5 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_grocery_problem_l2937_293755


namespace NUMINAMATH_CALUDE_expand_polynomial_product_l2937_293746

theorem expand_polynomial_product : ∀ t : ℝ,
  (3 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) = 
  -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_product_l2937_293746


namespace NUMINAMATH_CALUDE_subletter_monthly_rent_subletter_rent_is_400_l2937_293761

/-- Calculates the monthly rent for each subletter given the number of subletters,
    John's monthly rent, and John's annual profit. -/
theorem subletter_monthly_rent 
  (num_subletters : ℕ) 
  (john_monthly_rent : ℕ) 
  (john_annual_profit : ℕ) : ℕ :=
  let total_annual_rent := john_monthly_rent * 12 + john_annual_profit
  total_annual_rent / (num_subletters * 12)

/-- Proves that each subletter pays $400 per month given the specific conditions. -/
theorem subletter_rent_is_400 :
  subletter_monthly_rent 3 900 3600 = 400 := by
  sorry

end NUMINAMATH_CALUDE_subletter_monthly_rent_subletter_rent_is_400_l2937_293761


namespace NUMINAMATH_CALUDE_prob_three_same_color_standard_deck_l2937_293702

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing three cards of the same color from a standard deck -/
def prob_three_same_color (d : Deck) : ℚ :=
  let red_cards := d.red_suits * d.cards_per_suit
  let total_ways := d.total_cards * (d.total_cards - 1) * (d.total_cards - 2)
  let same_color_ways := 2 * (red_cards * (red_cards - 1) * (red_cards - 2))
  same_color_ways / total_ways

/-- Theorem: The probability of drawing three cards of the same color from a standard 52-card deck is 40/85 -/
theorem prob_three_same_color_standard_deck :
  prob_three_same_color standard_deck = 40 / 85 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_color_standard_deck_l2937_293702


namespace NUMINAMATH_CALUDE_class_size_from_mark_error_l2937_293767

theorem class_size_from_mark_error (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 40 → average_increase = 1/2 → 
  (mark_increase : ℚ) / average_increase = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_mark_error_l2937_293767


namespace NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l2937_293733

-- Define a function type
def Function := ℝ → ℝ

-- Define what it means for a function to have an inverse
def HasInverse (f : Function) : Prop :=
  ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the horizontal line test
def PassesHorizontalLineTest (f : Function) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- Theorem statement
theorem inverse_iff_horizontal_line_test (f : Function) :
  HasInverse f ↔ PassesHorizontalLineTest f :=
sorry

end NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l2937_293733


namespace NUMINAMATH_CALUDE_carnation_dozen_cost_carnation_dozen_cost_proof_l2937_293745

theorem carnation_dozen_cost (single_cost : ℚ) (teacher_dozens : ℕ) (friend_singles : ℕ) (total_spent : ℚ) : ℚ :=
  let dozen_cost := (total_spent - single_cost * friend_singles) / teacher_dozens
  dozen_cost

#check carnation_dozen_cost (1/2) 5 14 25 = 18/5

-- The proof is omitted
theorem carnation_dozen_cost_proof :
  carnation_dozen_cost (1/2) 5 14 25 = 18/5 := by sorry

end NUMINAMATH_CALUDE_carnation_dozen_cost_carnation_dozen_cost_proof_l2937_293745


namespace NUMINAMATH_CALUDE_equation_solution_l2937_293737

theorem equation_solution : ∃ x : ℤ, 45 - (5 * 3) = x + 7 ∧ x = 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2937_293737


namespace NUMINAMATH_CALUDE_sector_central_angle_l2937_293789

/-- Given a sector with radius 2 cm and area 4 cm², 
    prove that its central angle measures 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r^2 * α → α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2937_293789


namespace NUMINAMATH_CALUDE_sequence_sum_equality_l2937_293774

/-- Given two integer sequences satisfying a specific condition, 
    there exists a positive integer k such that the sum of the k-th terms 
    equals the sum of the (k+2018)-th terms. -/
theorem sequence_sum_equality 
  (a b : ℕ → ℤ) 
  (h : ∀ n ≥ 3, (a n - a (n-1)) * (a n - a (n-2)) + 
                (b n - b (n-1)) * (b n - b (n-2)) = 0) : 
  ∃ k : ℕ+, a k + b k = a (k + 2018) + b (k + 2018) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_equality_l2937_293774


namespace NUMINAMATH_CALUDE_product_of_primes_l2937_293727

theorem product_of_primes (p q : ℕ) : 
  Prime p → Prime q → 
  2 < p → p < 6 → 
  8 < q → q < 24 → 
  15 < p * q → p * q < 36 → 
  p * q = 33 := by sorry

end NUMINAMATH_CALUDE_product_of_primes_l2937_293727


namespace NUMINAMATH_CALUDE_sum_of_powers_l2937_293722

theorem sum_of_powers (k n : ℕ) : 
  (∀ x y : ℝ, 2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n) → 
  k + n = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2937_293722


namespace NUMINAMATH_CALUDE_power_comparison_l2937_293798

theorem power_comparison : 2^100 < 3^75 := by sorry

end NUMINAMATH_CALUDE_power_comparison_l2937_293798


namespace NUMINAMATH_CALUDE_simplify_expression_l2937_293748

theorem simplify_expression : (7^3 * (2^5)^3) / ((7^2) * 2^(3*3)) = 448 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2937_293748


namespace NUMINAMATH_CALUDE_fenced_rectangle_fence_length_l2937_293754

/-- A rectangular region with a fence on three sides and a wall on the fourth -/
structure FencedRectangle where
  short_side : ℝ
  long_side : ℝ
  area : ℝ
  fence_length : ℝ

/-- Properties of the fenced rectangle -/
def is_valid_fenced_rectangle (r : FencedRectangle) : Prop :=
  r.long_side = 2 * r.short_side ∧
  r.area = r.short_side * r.long_side ∧
  r.fence_length = 2 * r.short_side + r.long_side

theorem fenced_rectangle_fence_length 
  (r : FencedRectangle) 
  (h : is_valid_fenced_rectangle r) 
  (area_eq : r.area = 200) : 
  r.fence_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_fenced_rectangle_fence_length_l2937_293754


namespace NUMINAMATH_CALUDE_corn_acreage_l2937_293794

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l2937_293794


namespace NUMINAMATH_CALUDE_sin_cos_15_deg_l2937_293724

theorem sin_cos_15_deg : 4 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_deg_l2937_293724


namespace NUMINAMATH_CALUDE_satisfaction_theorem_l2937_293736

/-- Represents the setup of people around a round table -/
structure TableSetup :=
  (num_men : ℕ)
  (num_women : ℕ)

/-- Defines what it means for a man to be satisfied -/
def is_satisfied (setup : TableSetup) (p : ℝ) : Prop :=
  p = 1 - (setup.num_men - 1) / (setup.num_men + setup.num_women - 1) *
    (setup.num_men - 2) / (setup.num_men + setup.num_women - 2)

/-- The main theorem about the probability of satisfaction and expected number of satisfied men -/
theorem satisfaction_theorem (setup : TableSetup) 
  (h1 : setup.num_men = 50) (h2 : setup.num_women = 50) :
  ∃ (p : ℝ), 
    is_satisfied setup p ∧ 
    p = 25 / 33 ∧
    setup.num_men * p = 1250 / 33 := by
  sorry


end NUMINAMATH_CALUDE_satisfaction_theorem_l2937_293736


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l2937_293753

theorem lcm_gcf_relation (n : ℕ) (h1 : Nat.lcm n 16 = 52) (h2 : Nat.gcd n 16 = 8) : n = 26 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l2937_293753


namespace NUMINAMATH_CALUDE_construct_axes_l2937_293715

/-- A parabola in a 2D plane -/
structure Parabola where
  f : ℝ → ℝ
  is_parabola : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  is_line : a ≠ 0 ∨ b ≠ 0

/-- Compass and straightedge construction operations -/
inductive Construction
  | point : Point → Construction
  | line : Point → Point → Construction
  | circle : Point → Point → Construction
  | intersect_lines : Line → Line → Construction
  | intersect_line_circle : Line → Point → Point → Construction
  | intersect_circles : Point → Point → Point → Point → Construction

/-- The theorem stating that coordinate axes can be constructed given a parabola -/
theorem construct_axes (p : Parabola) : 
  ∃ (origin : Point) (x_axis y_axis : Line) (constructions : List Construction),
    (∀ x : ℝ, p.f x = x^2) →
    (origin.x = 0 ∧ origin.y = 0) ∧
    (∀ x : ℝ, x_axis.a * x + x_axis.b * 0 + x_axis.c = 0) ∧
    (∀ y : ℝ, y_axis.a * 0 + y_axis.b * y + y_axis.c = 0) :=
sorry

end NUMINAMATH_CALUDE_construct_axes_l2937_293715


namespace NUMINAMATH_CALUDE_coat_cost_after_discount_l2937_293732

/-- The cost of Mr. Zubir's purchases --/
structure Purchase where
  pants : ℝ
  shirt : ℝ
  coat : ℝ

/-- The conditions of Mr. Zubir's purchase --/
def purchase_conditions (p : Purchase) : Prop :=
  p.pants + p.shirt = 100 ∧
  p.pants + p.coat = 244 ∧
  p.coat = 5 * p.shirt

/-- The discount rate applied to the purchase --/
def discount_rate : ℝ := 0.1

/-- Theorem stating the cost of the coat after discount --/
theorem coat_cost_after_discount (p : Purchase) 
  (h : purchase_conditions p) : 
  p.coat * (1 - discount_rate) = 162 := by
  sorry

end NUMINAMATH_CALUDE_coat_cost_after_discount_l2937_293732


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2937_293726

/-- The measure of angle P in a pentagon PQRST where ∠P = 2∠Q = 4∠R = 3∠S = 6∠T is 240° -/
theorem pentagon_angle_measure (P Q R S T : ℝ) : 
  P + Q + R + S + T = 540 → -- sum of angles in a pentagon
  P = 2 * Q →              -- ∠P = 2∠Q
  P = 4 * R →              -- ∠P = 4∠R
  P = 3 * S →              -- ∠P = 3∠S
  P = 6 * T →              -- ∠P = 6∠T
  P = 240 := by            -- ∠P = 240°
sorry


end NUMINAMATH_CALUDE_pentagon_angle_measure_l2937_293726


namespace NUMINAMATH_CALUDE_machine_production_time_difference_l2937_293712

/-- Given two machines X and Y that produce widgets, this theorem proves
    that machine X takes 2 days longer than machine Y to produce W widgets. -/
theorem machine_production_time_difference
  (W : ℝ) -- W represents the number of widgets
  (h1 : (W / 6 + W / 4) * 3 = 5 * W / 4) -- Combined production in 3 days
  (h2 : W / 6 * 18 = 3 * W) -- Machine X production in 18 days
  : (W / (W / 6)) - (W / (W / 4)) = 2 :=
sorry

end NUMINAMATH_CALUDE_machine_production_time_difference_l2937_293712


namespace NUMINAMATH_CALUDE_inequality_solution_l2937_293760

theorem inequality_solution (x : ℝ) : (10 * x^2 + 1 < 7 * x) ∧ ((2 * x - 7) / (-3 * x + 1) > 0) ↔ x > 1/3 ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2937_293760


namespace NUMINAMATH_CALUDE_digit_fraction_statement_l2937_293776

theorem digit_fraction_statement : 
  ∃ (a b c : ℕ) (f g h : ℚ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    f + 2 * g + h = 1 ∧
    f = 1/2 ∧
    g = 1/5 ∧
    h = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_digit_fraction_statement_l2937_293776


namespace NUMINAMATH_CALUDE_james_travel_distance_l2937_293717

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James' travel distance -/
theorem james_travel_distance :
  let speed : ℝ := 80.0
  let time : ℝ := 16.0
  distance speed time = 1280.0 := by
  sorry

end NUMINAMATH_CALUDE_james_travel_distance_l2937_293717


namespace NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l2937_293707

theorem unit_digit_of_3_to_58 : 3^58 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l2937_293707


namespace NUMINAMATH_CALUDE_problem_solution_l2937_293723

theorem problem_solution (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^4 = s^3) 
  (h3 : r - p = 25) : 
  s - q = 73 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2937_293723


namespace NUMINAMATH_CALUDE_water_margin_price_l2937_293752

theorem water_margin_price :
  ∀ (x : ℝ),
    (x > 0) →
    (3600 / (x + 60) = (1 / 2) * (4800 / x)) →
    x = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_water_margin_price_l2937_293752


namespace NUMINAMATH_CALUDE_eleven_percent_greater_than_80_l2937_293711

theorem eleven_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 11 / 100) → x = 88.8 := by
sorry

end NUMINAMATH_CALUDE_eleven_percent_greater_than_80_l2937_293711


namespace NUMINAMATH_CALUDE_polynomial_C_value_l2937_293785

def polynomial (A B C D : ℤ) (x : ℝ) : ℝ := x^6 - 12*x^5 + A*x^4 + B*x^3 + C*x^2 + D*x + 36

theorem polynomial_C_value (A B C D : ℤ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ x : ℝ, polynomial A B C D x = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12)) →
  C = -171 := by
sorry

end NUMINAMATH_CALUDE_polynomial_C_value_l2937_293785


namespace NUMINAMATH_CALUDE_sin_cos_sum_14_16_l2937_293759

theorem sin_cos_sum_14_16 : 
  Real.sin (14 * π / 180) * Real.cos (16 * π / 180) + 
  Real.cos (14 * π / 180) * Real.sin (16 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_14_16_l2937_293759


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2937_293714

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (a b : Line) (α β : Plane)
  (h1 : perp_line_plane a α)
  (h2 : perp_line_plane b β)
  (h3 : perp_plane α β) :
  perp_line a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2937_293714


namespace NUMINAMATH_CALUDE_remainder_conversion_l2937_293763

theorem remainder_conversion (N : ℕ) : 
  N % 72 = 68 → N % 24 = 20 := by
sorry

end NUMINAMATH_CALUDE_remainder_conversion_l2937_293763


namespace NUMINAMATH_CALUDE_product_of_group_at_least_72_l2937_293795

theorem product_of_group_at_least_72 (group1 group2 group3 : List Nat) : 
  (group1 ++ group2 ++ group3).toFinset = Finset.range 9 →
  (group1.prod ≥ 72) ∨ (group2.prod ≥ 72) ∨ (group3.prod ≥ 72) := by
  sorry

end NUMINAMATH_CALUDE_product_of_group_at_least_72_l2937_293795


namespace NUMINAMATH_CALUDE_tangent_addition_subtraction_l2937_293741

theorem tangent_addition_subtraction (γ β : Real) (h1 : Real.tan γ = 5) (h2 : Real.tan β = 3) :
  Real.tan (γ + β) = -4/7 ∧ Real.tan (γ - β) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_addition_subtraction_l2937_293741


namespace NUMINAMATH_CALUDE_marks_trees_l2937_293729

theorem marks_trees (initial_trees planted_trees : ℕ) :
  initial_trees = 13 →
  planted_trees = 12 →
  initial_trees + planted_trees = 25 :=
by sorry

end NUMINAMATH_CALUDE_marks_trees_l2937_293729


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2937_293738

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 10*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 45 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2937_293738


namespace NUMINAMATH_CALUDE_animal_rescue_donation_l2937_293780

/-- Represents the daily earnings and costs for a 5-day bake sale --/
structure BakeSaleData :=
  (earnings : Fin 5 → ℕ)
  (costs : Fin 5 → ℕ)

/-- Represents the distribution percentages for charities --/
structure CharityDistribution :=
  (homeless_shelter : ℚ)
  (food_bank : ℚ)
  (park_restoration : ℚ)
  (animal_rescue : ℚ)

/-- Calculates the total donation to the Animal Rescue Center --/
def calculateAnimalRescueDonation (data : BakeSaleData) (dist : CharityDistribution) (personal_contribution : ℕ) : ℚ :=
  sorry

/-- Theorem stating the total donation to the Animal Rescue Center --/
theorem animal_rescue_donation
  (data : BakeSaleData)
  (dist : CharityDistribution)
  (h1 : data.earnings = ![450, 550, 400, 600, 500])
  (h2 : data.costs = ![80, 100, 70, 120, 90])
  (h3 : dist.homeless_shelter = 30 / 100)
  (h4 : dist.food_bank = 25 / 100)
  (h5 : dist.park_restoration = 20 / 100)
  (h6 : dist.animal_rescue = 25 / 100)
  (h7 : personal_contribution = 20) :
  calculateAnimalRescueDonation data dist personal_contribution = 535 := by
  sorry

end NUMINAMATH_CALUDE_animal_rescue_donation_l2937_293780


namespace NUMINAMATH_CALUDE_min_acquaintances_in_village_l2937_293751

/-- Represents a village with residents and their acquaintances. -/
structure Village where
  residents : Finset ℕ
  acquaintances : Finset (ℕ × ℕ)

/-- Checks if a given set of residents can be seated according to the problem's conditions. -/
def canBeSeatedCircularly (v : Village) (group : Finset ℕ) : Prop :=
  group.card = 6 ∧ ∃ (seating : Fin 6 → ℕ), 
    (∀ i, seating i ∈ group) ∧
    (∀ i, (seating i, seating ((i + 1) % 6)) ∈ v.acquaintances ∧
          (seating i, seating ((i + 5) % 6)) ∈ v.acquaintances)

/-- The main theorem statement. -/
theorem min_acquaintances_in_village (v : Village) :
  v.residents.card = 200 ∧ 
  (∀ group : Finset ℕ, group ⊆ v.residents → canBeSeatedCircularly v group) →
  v.acquaintances.card = 19600 :=
sorry

end NUMINAMATH_CALUDE_min_acquaintances_in_village_l2937_293751


namespace NUMINAMATH_CALUDE_garland_arrangement_count_l2937_293719

/-- The number of ways to arrange blue, red, and white bulbs in a garland with no adjacent white bulbs -/
def garland_arrangements (blue red white : ℕ) : ℕ :=
  (Nat.choose (blue + red) blue) * (Nat.choose (blue + red + 1) white)

/-- Theorem stating the number of arrangements for 8 blue, 6 red, and 12 white bulbs -/
theorem garland_arrangement_count : garland_arrangements 8 6 12 = 1366365 := by
  sorry

end NUMINAMATH_CALUDE_garland_arrangement_count_l2937_293719


namespace NUMINAMATH_CALUDE_equidistant_planes_count_l2937_293728

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if 4 points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Function to check if a plane is equidistant from 4 points
def is_equidistant_plane (plane : Plane3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Theorem statement
theorem equidistant_planes_count 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ are_coplanar p1 p2 p3 p4) : 
  ∃! (planes : Finset Plane3D), 
    (planes.card = 7) ∧ 
    (∀ plane ∈ planes, is_equidistant_plane plane p1 p2 p3 p4) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_planes_count_l2937_293728


namespace NUMINAMATH_CALUDE_beads_per_package_l2937_293758

theorem beads_per_package (total_packages : Nat) (total_beads : Nat) : 
  total_packages = 8 → total_beads = 320 → (total_beads / total_packages : Nat) = 40 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_package_l2937_293758


namespace NUMINAMATH_CALUDE_unfair_die_theorem_l2937_293775

def unfair_die_expected_value (p_eight : ℚ) (p_other : ℚ) : ℚ :=
  (1 * p_other + 2 * p_other + 3 * p_other + 4 * p_other + 
   5 * p_other + 6 * p_other + 7 * p_other) + (8 * p_eight)

theorem unfair_die_theorem :
  let p_eight : ℚ := 3/8
  let p_other : ℚ := 1/14
  unfair_die_expected_value p_eight p_other = 5 := by
sorry

#eval unfair_die_expected_value (3/8 : ℚ) (1/14 : ℚ)

end NUMINAMATH_CALUDE_unfair_die_theorem_l2937_293775


namespace NUMINAMATH_CALUDE_max_sum_after_erasing_l2937_293706

-- Define the initial set of numbers
def initial_numbers : List ℕ := List.range 13 |>.map (· + 4)

-- Define a function to check if a list can be divided into groups with equal sums
def can_be_divided_equally (numbers : List ℕ) : Prop :=
  ∃ (k : ℕ) (groups : List (List ℕ)),
    k > 1 ∧
    groups.length = k ∧
    groups.all (λ group ↦ group.sum = (numbers.sum / k)) ∧
    groups.join.toFinset = numbers.toFinset

-- Define the theorem
theorem max_sum_after_erasing (numbers : List ℕ) :
  numbers.sum = 121 →
  numbers ⊆ initial_numbers →
  ¬ can_be_divided_equally numbers →
  ∀ (other_numbers : List ℕ),
    other_numbers ⊆ initial_numbers →
    other_numbers.sum > 121 →
    can_be_divided_equally other_numbers :=
sorry

end NUMINAMATH_CALUDE_max_sum_after_erasing_l2937_293706


namespace NUMINAMATH_CALUDE_total_cookies_l2937_293703

def cookie_problem (chris kenny glenn dan anne : ℕ) : Prop :=
  chris = kenny / 3 ∧
  glenn = 4 * chris ∧
  glenn = 24 ∧
  dan = 2 * (chris + kenny) ∧
  anne = kenny / 2

theorem total_cookies :
  ∀ chris kenny glenn dan anne : ℕ,
  cookie_problem chris kenny glenn dan anne →
  chris + kenny + glenn + dan + anne = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l2937_293703


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l2937_293756

theorem unique_three_digit_divisible_by_seven : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 4 ∧          -- units digit is 4
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 7 = 0 ∧           -- divisible by 7
    n = 658               -- the number is 658
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l2937_293756


namespace NUMINAMATH_CALUDE_no_real_solutions_l2937_293742

theorem no_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 5) (h2 : y + 1 / x = 1 / 6) : False :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2937_293742


namespace NUMINAMATH_CALUDE_function_domain_condition_l2937_293716

/-- A function f(x) = (kx + 5) / (kx^2 + 4kx + 3) is defined for all real x if and only if 0 ≤ k < 3/4 -/
theorem function_domain_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (k * x + 5) / (k * x^2 + 4 * k * x + 3)) ↔ 
  (0 ≤ k ∧ k < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_function_domain_condition_l2937_293716


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l2937_293709

theorem at_least_one_real_root (p₁ p₂ q₁ q₂ : ℝ) (h : p₁ * p₂ = 2 * (q₁ + q₂)) :
  (∃ x : ℝ, x^2 + p₁*x + q₁ = 0) ∨ (∃ x : ℝ, x^2 + p₂*x + q₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l2937_293709


namespace NUMINAMATH_CALUDE_invalid_period_pair_l2937_293720

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 48) ≡ a n [ZMOD 35]

def least_period_mod5 (a : ℕ → ℤ) (i : ℕ) : Prop :=
  (∀ n, n ≥ 1 → a (n + i) ≡ a n [ZMOD 5]) ∧
  (∀ k, k < i → ∃ n, n ≥ 1 ∧ ¬(a (n + k) ≡ a n [ZMOD 5]))

def least_period_mod7 (a : ℕ → ℤ) (j : ℕ) : Prop :=
  (∀ n, n ≥ 1 → a (n + j) ≡ a n [ZMOD 7]) ∧
  (∀ k, k < j → ∃ n, n ≥ 1 ∧ ¬(a (n + k) ≡ a n [ZMOD 7]))

theorem invalid_period_pair :
  ∀ a : ℕ → ℤ,
  is_valid_sequence a →
  ∀ i j : ℕ,
  least_period_mod5 a i →
  least_period_mod7 a j →
  (i, j) ≠ (16, 4) :=
by sorry

end NUMINAMATH_CALUDE_invalid_period_pair_l2937_293720


namespace NUMINAMATH_CALUDE_sqrt_a_plus_one_real_iff_a_geq_neg_one_l2937_293710

theorem sqrt_a_plus_one_real_iff_a_geq_neg_one (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 1) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_one_real_iff_a_geq_neg_one_l2937_293710


namespace NUMINAMATH_CALUDE_c_range_l2937_293749

def P (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x y : ℝ, 1/2 < x ∧ x < y → (x^2 - 2*c*x + 1) < (y^2 - 2*c*y + 1)

theorem c_range (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) :
  (P c ∨ q c) ∧ ¬(P c ∧ q c) → 1/2 < c ∧ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_c_range_l2937_293749


namespace NUMINAMATH_CALUDE_expression_simplification_l2937_293762

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ((2*x + y) * (2*x - y) - (2*x - y)^2 - y*(x - 2*y)) / (2*x) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2937_293762


namespace NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l2937_293721

/-- Represents the possible outcomes of a fair die toss -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Event A: an odd number is facing up -/
def event_A (face : DieFace) : Prop :=
  face = DieFace.one ∨ face = DieFace.three ∨ face = DieFace.five

/-- Event D: either 2 or 4 is facing up -/
def event_D (face : DieFace) : Prop :=
  face = DieFace.two ∨ face = DieFace.four

/-- The sample space of a fair die toss -/
def sample_space : Set DieFace :=
  {DieFace.one, DieFace.two, DieFace.three, DieFace.four, DieFace.five, DieFace.six}

theorem A_D_mutually_exclusive_not_complementary :
  (∀ (face : DieFace), ¬(event_A face ∧ event_D face)) ∧
  (∃ (face : DieFace), ¬event_A face ∧ ¬event_D face) :=
by sorry

end NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l2937_293721


namespace NUMINAMATH_CALUDE_number_equation_l2937_293771

theorem number_equation : ∃ n : ℚ, n = (n - 5) * 4 := by
  use 20 / 3
  sorry

end NUMINAMATH_CALUDE_number_equation_l2937_293771


namespace NUMINAMATH_CALUDE_investment_growth_l2937_293796

def calculate_final_value (initial_investment : ℝ) : ℝ :=
  let year1_increase := 0.75
  let year2_decrease := 0.30
  let year3_increase := 0.45
  let year4_decrease := 0.15
  let tax_rate := 0.20
  let fee_rate := 0.02

  let year1_value := initial_investment * (1 + year1_increase)
  let year1_after_fees := year1_value * (1 - fee_rate)

  let year2_value := year1_after_fees * (1 - year2_decrease)
  let year2_after_fees := year2_value * (1 - fee_rate)

  let year3_value := year2_after_fees * (1 + year3_increase)
  let year3_after_fees := year3_value * (1 - fee_rate)

  let year4_value := year3_after_fees * (1 - year4_decrease)
  let year4_after_fees := year4_value * (1 - fee_rate)

  let capital_gains := year4_after_fees - initial_investment
  let taxes := capital_gains * tax_rate
  year4_after_fees - taxes

theorem investment_growth (initial_investment : ℝ) :
  initial_investment = 100 →
  calculate_final_value initial_investment = 131.408238206 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l2937_293796


namespace NUMINAMATH_CALUDE_bisection_second_iteration_l2937_293770

def f (x : ℝ) := x^3 + 3*x - 1

theorem bisection_second_iteration
  (h1 : f 0 < 0)
  (h2 : f (1/2) > 0) :
  let second_iteration := (0 + 1/2) / 2
  second_iteration = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_bisection_second_iteration_l2937_293770


namespace NUMINAMATH_CALUDE_james_downhill_speed_l2937_293797

/-- Proves that James' speed on the downhill trail is 5 miles per hour given the problem conditions. -/
theorem james_downhill_speed :
  ∀ (v : ℝ),
    v > 0 →
    (20 : ℝ) / v = (12 : ℝ) / 3 + 1 - 1 →
    v = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_downhill_speed_l2937_293797
