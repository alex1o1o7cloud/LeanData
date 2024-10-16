import Mathlib

namespace NUMINAMATH_CALUDE_experiment_arrangements_l2062_206206

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 6

/-- Represents whether procedure A is at the beginning or end -/
inductive A_position
| beginning
| end

/-- Calculates the number of arrangements for a given A position -/
def arrangements_for_A_position (pos : A_position) : ℕ := 
  (Nat.factorial (num_procedures - 3)) * 2

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ :=
  arrangements_for_A_position A_position.beginning + 
  arrangements_for_A_position A_position.end

/-- Theorem stating that the total number of arrangements is 96 -/
theorem experiment_arrangements :
  total_arrangements = 96 := by sorry

end NUMINAMATH_CALUDE_experiment_arrangements_l2062_206206


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l2062_206279

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a line passes through the first quadrant -/
def passesFirstQuadrant (l : Line) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ l.a * x + l.b * y + l.c = 0

theorem line_not_in_first_quadrant (l : Line) 
  (h1 : ¬passesFirstQuadrant l) 
  (h2 : l.a * l.b > 0) : 
  l.a * l.c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l2062_206279


namespace NUMINAMATH_CALUDE_f_two_expression_l2062_206263

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (f 1 = 2) ∧ 
  (∀ x y : ℝ, f (x * y + f x + 1) = x * f y + f x)

/-- The main theorem stating that f(2) can be expressed as c + 2 -/
theorem f_two_expression 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) :
  ∃ c : ℝ, f 2 = c + 2 :=
sorry

end NUMINAMATH_CALUDE_f_two_expression_l2062_206263


namespace NUMINAMATH_CALUDE_vector_at_t_6_l2062_206226

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem vector_at_t_6 (h0 : line_vector 0 = (2, -1, 3))
                      (h4 : line_vector 4 = (6, 7, -1)) :
  line_vector 6 = (8, 11, -3) := by sorry

end NUMINAMATH_CALUDE_vector_at_t_6_l2062_206226


namespace NUMINAMATH_CALUDE_expected_value_of_sum_is_twelve_l2062_206268

def marbles : Finset ℕ := Finset.range 7

def choose_three (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ subset => subset.card = 3)

def sum_of_subset (subset : Finset ℕ) : ℕ :=
  subset.sum id

theorem expected_value_of_sum_is_twelve :
  let all_choices := choose_three marbles
  let sum_of_sums := all_choices.sum sum_of_subset
  let num_choices := all_choices.card
  (sum_of_sums : ℚ) / num_choices = 12 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_is_twelve_l2062_206268


namespace NUMINAMATH_CALUDE_minimum_framing_feet_l2062_206289

def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

def enlarged_width : ℕ := original_width * enlargement_factor
def enlarged_height : ℕ := original_height * enlargement_factor

def final_width : ℕ := enlarged_width + 2 * border_width
def final_height : ℕ := enlarged_height + 2 * border_width

def perimeter_inches : ℕ := 2 * (final_width + final_height)

def inches_per_foot : ℕ := 12

theorem minimum_framing_feet :
  (perimeter_inches + inches_per_foot - 1) / inches_per_foot = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_framing_feet_l2062_206289


namespace NUMINAMATH_CALUDE_distance_between_foci_l2062_206256

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) + Real.sqrt ((x + 4)^2 + (y - 5)^2) = 18

-- Define the foci
def focus1 : ℝ × ℝ := (2, 3)
def focus2 : ℝ × ℝ := (-4, 5)

-- Theorem statement
theorem distance_between_foci :
  let d := Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2)
  d = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2062_206256


namespace NUMINAMATH_CALUDE_probability_black_white_balls_l2062_206282

theorem probability_black_white_balls (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) :
  total_balls = black_balls + white_balls + green_balls →
  black_balls = 3 →
  white_balls = 3 →
  green_balls = 1 →
  (black_balls * white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_black_white_balls_l2062_206282


namespace NUMINAMATH_CALUDE_same_club_probability_l2062_206217

theorem same_club_probability (n : ℕ) (h : n = 8) :
  let p := 1 / n
  (n : ℝ) * p * p = 1 / n :=
by
  sorry

end NUMINAMATH_CALUDE_same_club_probability_l2062_206217


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2062_206270

theorem system_of_equations_solution :
  ∃ (x y z : ℝ),
    (4*x - 3*y + z = -9) ∧
    (2*x + 5*y - 3*z = 8) ∧
    (x + y + 2*z = 5) ∧
    (x = 1 ∧ y = -1 ∧ z = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2062_206270


namespace NUMINAMATH_CALUDE_greatest_base6_digit_sum_l2062_206208

/-- Represents a base-6 digit -/
def Base6Digit := Fin 6

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : ℕ) : List Base6Digit :=
  sorry

/-- Calculates the sum of digits in a list -/
def digitSum (digits : List Base6Digit) : ℕ :=
  sorry

/-- Theorem: The greatest possible sum of digits in the base-6 representation
    of a positive integer less than 1728 is 20 -/
theorem greatest_base6_digit_sum :
  ∃ (n : ℕ), n > 0 ∧ n < 1728 ∧
  digitSum (toBase6 n) = 20 ∧
  ∀ (m : ℕ), m > 0 → m < 1728 → digitSum (toBase6 m) ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_greatest_base6_digit_sum_l2062_206208


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2062_206291

theorem imaginary_part_of_complex_number (z : ℂ) (h : z = -1 + Complex.I) :
  z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2062_206291


namespace NUMINAMATH_CALUDE_quadratic_sum_of_d_and_e_l2062_206299

/-- Given a quadratic polynomial x^2 - 16x + 15, when written in the form (x+d)^2 + e,
    the sum of d and e is -57. -/
theorem quadratic_sum_of_d_and_e : ∃ d e : ℝ, 
  (∀ x, x^2 - 16*x + 15 = (x+d)^2 + e) ∧ d + e = -57 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_d_and_e_l2062_206299


namespace NUMINAMATH_CALUDE_decagon_partition_impossible_l2062_206248

/-- A partition of a polygon into triangles -/
structure TrianglePartition (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  is_valid : black_sides - white_sides = n

/-- Property that the number of sides in a valid triangle partition is divisible by 3 -/
def sides_divisible_by_three (partition : TrianglePartition n) : Prop :=
  partition.black_sides % 3 = 0 ∧ partition.white_sides % 3 = 0

theorem decagon_partition_impossible :
  ¬ ∃ (partition : TrianglePartition 10), sides_divisible_by_three partition :=
sorry

end NUMINAMATH_CALUDE_decagon_partition_impossible_l2062_206248


namespace NUMINAMATH_CALUDE_wall_width_calculation_l2062_206250

theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) :
  mirror_side = 18 →
  wall_length = 20.25 →
  (mirror_side * mirror_side) * 2 = wall_length * (648 / wall_length) :=
by
  sorry

#check wall_width_calculation

end NUMINAMATH_CALUDE_wall_width_calculation_l2062_206250


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l2062_206290

theorem hyperbola_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let c := Real.sqrt (a^2 + b^2)  -- focal distance
  (c / a = e) →
  (∀ a' b', a' > 0 → b' > 0 → c / a' = e → (b'^2 + 1) / (3 * a') ≥ (b^2 + 1) / (3 * a)) →
  (b^2 + 1) / (3 * a) = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l2062_206290


namespace NUMINAMATH_CALUDE_rearrange_pegs_l2062_206233

/-- Represents a position on the board --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the board state --/
def BoardState := List Position

/-- Checks if a given arrangement of pegs satisfies the condition of 5 rows with 4 pegs each --/
def isValidArrangement (arrangement : BoardState) : Bool :=
  sorry

/-- Counts the number of pegs that need to be moved to transform one arrangement into another --/
def pegsMoved (initial : BoardState) (final : BoardState) : Nat :=
  sorry

/-- The main theorem stating that it's possible to achieve the desired arrangement by moving exactly 3 pegs --/
theorem rearrange_pegs (initial : BoardState) :
  (initial.length = 10) →
  ∃ (final : BoardState), 
    isValidArrangement final ∧ 
    pegsMoved initial final = 3 :=
  sorry

end NUMINAMATH_CALUDE_rearrange_pegs_l2062_206233


namespace NUMINAMATH_CALUDE_fold_point_sum_l2062_206278

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Determines if a line folds one point onto another -/
def folds (l : Line) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  midpoint.y = l.slope * midpoint.x + l.intercept

/-- The main theorem -/
theorem fold_point_sum (l : Line) :
  folds l ⟨1, 3⟩ ⟨5, 1⟩ →
  folds l ⟨8, 4⟩ ⟨m, n⟩ →
  m + n = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fold_point_sum_l2062_206278


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2062_206251

theorem cube_root_simplification :
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2062_206251


namespace NUMINAMATH_CALUDE_unique_solution_implies_k_equals_one_l2062_206273

/-- The set of real solutions to the quadratic equation kx^2 + 4x + 4 = 0 -/
def A (k : ℝ) : Set ℝ := {x : ℝ | k * x^2 + 4 * x + 4 = 0}

/-- Theorem: If the set A contains only one element, then k = 1 -/
theorem unique_solution_implies_k_equals_one (k : ℝ) : (∃! x, x ∈ A k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_k_equals_one_l2062_206273


namespace NUMINAMATH_CALUDE_tangent_line_at_y_axis_l2062_206239

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4) / (x - 2)

theorem tangent_line_at_y_axis (x y : ℝ) :
  (f 0 = -2) →
  (∀ x, deriv f x = (x^2 - 4*x - 4) / (x - 2)^2) →
  (y = -x - 2) ↔ (y - f 0 = deriv f 0 * (x - 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_y_axis_l2062_206239


namespace NUMINAMATH_CALUDE_distance_AB_is_correct_l2062_206258

/-- The distance between two points A and B, given the conditions of the problem. -/
def distance_AB : ℝ :=
  let first_meeting_distance : ℝ := 700
  let second_meeting_distance : ℝ := 400
  -- Define the distance as a variable to be solved
  let d : ℝ := 1700
  d

theorem distance_AB_is_correct : distance_AB = 1700 := by
  -- Unfold the definition of distance_AB
  unfold distance_AB
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_distance_AB_is_correct_l2062_206258


namespace NUMINAMATH_CALUDE_sum_equals_5000_minus_N_l2062_206229

theorem sum_equals_5000_minus_N (N : ℕ) : 988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_5000_minus_N_l2062_206229


namespace NUMINAMATH_CALUDE_travel_statements_correct_l2062_206245

/-- Represents a traveler (cyclist or motorcyclist) --/
structure Traveler where
  startTime : ℝ
  arrivalTime : ℝ
  distanceTraveled : ℝ → ℝ
  speed : ℝ → ℝ

/-- The travel scenario between two towns --/
structure TravelScenario where
  cyclist : Traveler
  motorcyclist : Traveler
  totalDistance : ℝ

/-- Properties of the travel scenario --/
def TravelScenario.properties (scenario : TravelScenario) : Prop :=
  -- The total distance is 80km
  scenario.totalDistance = 80 ∧
  -- The cyclist starts 3 hours before the motorcyclist
  scenario.cyclist.startTime + 3 = scenario.motorcyclist.startTime ∧
  -- The cyclist arrives 1 hour before the motorcyclist
  scenario.cyclist.arrivalTime + 1 = scenario.motorcyclist.arrivalTime ∧
  -- The cyclist's speed pattern (acceleration then constant)
  (∃ t₀ : ℝ, ∀ t, t ≥ scenario.cyclist.startTime → 
    (t ≤ t₀ → scenario.cyclist.speed t < scenario.cyclist.speed (t + 1)) ∧
    (t > t₀ → scenario.cyclist.speed t = scenario.cyclist.speed t₀)) ∧
  -- The motorcyclist's constant speed
  (∀ t₁ t₂, scenario.motorcyclist.speed t₁ = scenario.motorcyclist.speed t₂) ∧
  -- The catch-up time
  (∃ t : ℝ, t = scenario.motorcyclist.startTime + 1.5 ∧
    scenario.cyclist.distanceTraveled t = scenario.motorcyclist.distanceTraveled t)

/-- The main theorem stating the correctness of all statements --/
theorem travel_statements_correct (scenario : TravelScenario) 
  (h : scenario.properties) : 
  -- Statement 1: Timing difference
  (scenario.cyclist.startTime + 3 = scenario.motorcyclist.startTime ∧
   scenario.cyclist.arrivalTime + 1 = scenario.motorcyclist.arrivalTime) ∧
  -- Statement 2: Speed patterns
  (∃ t₀ : ℝ, ∀ t, t ≥ scenario.cyclist.startTime → 
    (t ≤ t₀ → scenario.cyclist.speed t < scenario.cyclist.speed (t + 1)) ∧
    (t > t₀ → scenario.cyclist.speed t = scenario.cyclist.speed t₀)) ∧
  (∀ t₁ t₂, scenario.motorcyclist.speed t₁ = scenario.motorcyclist.speed t₂) ∧
  -- Statement 3: Catch-up time
  (∃ t : ℝ, t = scenario.motorcyclist.startTime + 1.5 ∧
    scenario.cyclist.distanceTraveled t = scenario.motorcyclist.distanceTraveled t) :=
by sorry

end NUMINAMATH_CALUDE_travel_statements_correct_l2062_206245


namespace NUMINAMATH_CALUDE_car_speed_problem_l2062_206283

theorem car_speed_problem (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (distance : ℝ) :
  speed1 = 45 →
  time = 14/3 →
  distance = 490 →
  (speed1 + speed2) * time = distance →
  speed2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2062_206283


namespace NUMINAMATH_CALUDE_max_quotient_value_l2062_206204

theorem max_quotient_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 300)
  (hb : 800 ≤ b ∧ b ≤ 1600)
  (hab : a + b ≤ 1800) :
  ∃ (a' b' : ℝ), 
    100 ≤ a' ∧ a' ≤ 300 ∧
    800 ≤ b' ∧ b' ≤ 1600 ∧
    a' + b' ≤ 1800 ∧
    b' / a' = 5 ∧
    ∀ (x y : ℝ), 
      100 ≤ x ∧ x ≤ 300 → 
      800 ≤ y ∧ y ≤ 1600 → 
      x + y ≤ 1800 → 
      y / x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2062_206204


namespace NUMINAMATH_CALUDE_unique_solution_condition_smallest_divisor_double_factorial_divides_sum_double_factorial_l2062_206246

-- Definition of double factorial
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

-- Theorem for the first part of the problem
theorem unique_solution_condition (a : ℝ) :
  (∃! x y : ℝ, x + y - 144 = 0 ∧ x * y - 5184 - 0.1 * a^2 = 0) ↔ a = 0 := by sorry

-- Theorem for the second part of the problem
theorem smallest_divisor_double_factorial :
  ∀ n : ℕ, n > 2022 → n ∣ (double_factorial 2021 + double_factorial 2022) → n ≥ 2023 := by sorry

-- Theorem that 2023 divides the sum of double factorials
theorem divides_sum_double_factorial :
  2023 ∣ (double_factorial 2021 + double_factorial 2022) := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_smallest_divisor_double_factorial_divides_sum_double_factorial_l2062_206246


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2062_206218

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 ≥ m^2 - 3*m) → 
  m < 1 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2062_206218


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_neg_one_three_l2062_206216

/-- Given an angle α whose terminal side passes through the point (-1, 3),
    prove that sin α = (3 * √10) / 10 -/
theorem sin_alpha_for_point_neg_one_three (α : Real) :
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 3) →
  Real.sin α = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_neg_one_three_l2062_206216


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2062_206202

def set_A : Set ℝ := {x | (x - 1) / (x + 3) < 0}
def set_B : Set ℝ := {x | abs x < 2}

theorem intersection_of_A_and_B : 
  set_A ∩ set_B = {x | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2062_206202


namespace NUMINAMATH_CALUDE_bernardo_wins_l2062_206265

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2 * N < 1000 ∧
  2 * N + 100 < 1000 ∧
  4 * N + 200 < 1000 ∧
  4 * N + 300 < 1000 ∧
  8 * N + 600 < 1000 ∧
  8 * N + 700 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  ∃ N, game_winner N ∧ 
    (∀ M, M < N → ¬game_winner M) ∧
    N = 38 ∧
    sum_of_digits N = 11 :=
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l2062_206265


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2062_206219

/-- Calculates the simple interest rate given the principal, time, and interest amount. -/
def calculate_interest_rate (principal : ℕ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / ((principal : ℚ) * (time : ℚ))

theorem interest_rate_problem (principal time interest_difference : ℕ) 
  (h1 : principal = 3000)
  (h2 : time = 5)
  (h3 : interest_difference = 2400)
  (h4 : principal - interest_difference > 0) :
  calculate_interest_rate principal time (principal - interest_difference) = 4 := by
  sorry

#eval calculate_interest_rate 3000 5 600

end NUMINAMATH_CALUDE_interest_rate_problem_l2062_206219


namespace NUMINAMATH_CALUDE_quartic_root_ratio_l2062_206235

theorem quartic_root_ratio (a b c d e : ℝ) (h : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = 3) : 
  d / e = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_ratio_l2062_206235


namespace NUMINAMATH_CALUDE_triangle_properties_l2062_206287

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Define the triangle
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  -- Given conditions
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A →
  b + c = Real.sqrt 10 →
  a = 2 →
  -- Prove
  Real.cos A = 1/2 ∧
  (1/2 : ℝ) * b * c * Real.sin A = (7 * Real.sqrt 3) / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2062_206287


namespace NUMINAMATH_CALUDE_composite_number_quotient_l2062_206264

def composite_numbers : List ℕ := [4, 6, 8, 9, 10, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39]

def product_6th_to_15th : ℕ := (List.take 10 (List.drop 5 composite_numbers)).prod

def product_16th_to_25th : ℕ := (List.take 10 (List.drop 15 composite_numbers)).prod

theorem composite_number_quotient :
  (product_6th_to_15th : ℚ) / product_16th_to_25th =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_composite_number_quotient_l2062_206264


namespace NUMINAMATH_CALUDE_tournament_participants_l2062_206247

theorem tournament_participants :
  ∃ n : ℕ,
    n > 0 ∧
    (n - 2) * (n - 3) / 2 + 7 = 62 ∧
    n = 13 :=
by sorry

end NUMINAMATH_CALUDE_tournament_participants_l2062_206247


namespace NUMINAMATH_CALUDE_ten_player_tournament_decided_in_seven_rounds_l2062_206227

/-- Represents a chess tournament -/
structure ChessTournament where
  num_players : ℕ
  rounds : ℕ

/-- The scoring system for the tournament -/
def score_system : ℕ → ℚ
  | 0 => 0     -- Loss
  | 1 => 1/2   -- Draw
  | _ => 1     -- Win

/-- The maximum possible score for a player after a given number of rounds -/
def max_score (t : ChessTournament) : ℚ := t.rounds

/-- The total points distributed after a given number of rounds -/
def total_points (t : ChessTournament) : ℚ := (t.num_players * t.rounds) / 2

/-- A tournament is decided if the maximum score is greater than the average of the remaining points -/
def is_decided (t : ChessTournament) : Prop :=
  max_score t > (total_points t - max_score t) / (t.num_players - 1)

/-- The main theorem: A 10-player tournament is decided after 7 rounds -/
theorem ten_player_tournament_decided_in_seven_rounds :
  let t : ChessTournament := ⟨10, 7⟩
  is_decided t ∧ ∀ r < 7, ¬is_decided ⟨10, r⟩ := by sorry

end NUMINAMATH_CALUDE_ten_player_tournament_decided_in_seven_rounds_l2062_206227


namespace NUMINAMATH_CALUDE_leonardo_sleep_fraction_l2062_206213

-- Define the number of minutes in an hour
def minutes_in_hour : ℕ := 60

-- Define Leonardo's sleep duration in minutes
def leonardo_sleep_minutes : ℕ := 12

-- Theorem to prove
theorem leonardo_sleep_fraction :
  (leonardo_sleep_minutes : ℚ) / minutes_in_hour = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_leonardo_sleep_fraction_l2062_206213


namespace NUMINAMATH_CALUDE_point_on_circle_l2062_206259

/-- Given a circle C with maximum radius 2 containing points (2,y) and (-2,0),
    prove that the y-coordinate of (2,y) is 0 -/
theorem point_on_circle (y : ℝ) : 
  (∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    radius ≤ 2 ∧
    (2, y) ∈ C ∧
    (-2, 0) ∈ C ∧
    C = {p : ℝ × ℝ | dist p center = radius}) →
  y = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_l2062_206259


namespace NUMINAMATH_CALUDE_theater_sales_total_cost_l2062_206205

/-- Represents the theater ticket sales problem --/
structure TheaterSales where
  total_tickets : ℕ
  balcony_surplus : ℕ
  orchestra_price : ℕ
  balcony_price : ℕ

/-- Calculate the total cost of tickets sold --/
def total_cost (sales : TheaterSales) : ℕ :=
  let orchestra_tickets := (sales.total_tickets - sales.balcony_surplus) / 2
  let balcony_tickets := sales.total_tickets - orchestra_tickets
  orchestra_tickets * sales.orchestra_price + balcony_tickets * sales.balcony_price

/-- Theorem stating that the total cost for the given conditions is $3320 --/
theorem theater_sales_total_cost :
  let sales : TheaterSales := {
    total_tickets := 370,
    balcony_surplus := 190,
    orchestra_price := 12,
    balcony_price := 8
  }
  total_cost sales = 3320 := by
  sorry


end NUMINAMATH_CALUDE_theater_sales_total_cost_l2062_206205


namespace NUMINAMATH_CALUDE_hex_to_decimal_conversion_l2062_206297

/-- Given a hexadecimal number m02₍₆₎ that is equivalent to 146 in decimal, 
    prove that m = 4. -/
theorem hex_to_decimal_conversion (m : ℕ) : 
  (2 + m * 6^2 = 146) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_decimal_conversion_l2062_206297


namespace NUMINAMATH_CALUDE_chicken_food_consumption_l2062_206296

/-- Given Dany's farm animals and their food consumption, calculate the amount of food each chicken eats per day. -/
theorem chicken_food_consumption 
  (num_cows : ℕ) 
  (num_sheep : ℕ) 
  (num_chickens : ℕ) 
  (cow_sheep_consumption : ℕ) 
  (total_consumption : ℕ) 
  (h1 : num_cows = 4) 
  (h2 : num_sheep = 3) 
  (h3 : num_chickens = 7) 
  (h4 : cow_sheep_consumption = 2) 
  (h5 : total_consumption = 35) : 
  ((total_consumption - (num_cows + num_sheep) * cow_sheep_consumption) / num_chickens : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_food_consumption_l2062_206296


namespace NUMINAMATH_CALUDE_peach_difference_l2062_206266

/-- Given a basket of peaches with specific counts for each color, 
    prove the difference between green and red peaches. -/
theorem peach_difference (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h_red : red = 7)
  (h_yellow : yellow = 71)
  (h_green : green = 8) :
  green - red = 1 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2062_206266


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l2062_206254

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_after_15th_innings 
  (b : Batsman)
  (h1 : b.innings = 14)
  (h2 : newAverage b 85 = b.average + 3) :
  newAverage b 85 = 43 := by
  sorry

#check batsman_average_after_15th_innings

end NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l2062_206254


namespace NUMINAMATH_CALUDE_two_real_roots_iff_nonneg_discriminant_quadratic_always_two_real_roots_l2062_206280

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has two real roots if and only if its discriminant is non-negative -/
theorem two_real_roots_iff_nonneg_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x y : ℝ, a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 ∧ x ≠ y ↔ discriminant a b c ≥ 0 :=
sorry

theorem quadratic_always_two_real_roots (k : ℝ) :
  discriminant 1 (-(k+4)) (4*k) ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_two_real_roots_iff_nonneg_discriminant_quadratic_always_two_real_roots_l2062_206280


namespace NUMINAMATH_CALUDE_bridget_apples_proof_l2062_206294

/-- Represents the number of apples Bridget initially bought -/
def initial_apples : ℕ := 20

/-- Represents the number of apples Bridget ate -/
def apples_eaten : ℕ := 2

/-- Represents the number of apples Bridget gave to Cassie -/
def apples_to_cassie : ℕ := 5

/-- Represents the number of apples Bridget kept for herself -/
def apples_kept : ℕ := 6

theorem bridget_apples_proof :
  let remaining_after_eating := initial_apples - apples_eaten
  let remaining_after_ann := remaining_after_eating - (remaining_after_eating / 3)
  let final_remaining := remaining_after_ann - apples_to_cassie
  final_remaining = apples_kept :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_proof_l2062_206294


namespace NUMINAMATH_CALUDE_triangle_BC_length_l2062_206207

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.A.1 = 1 ∧ t.A.2 = 1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1 ∧
  t.B.2 = t.C.2 ∧
  (1/2 * (t.C.1 - t.B.1) * (t.B.2 - t.A.2) = 32)

-- Theorem statement
theorem triangle_BC_length (t : Triangle) :
  triangle_conditions t → (t.C.1 - t.B.1 = 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_BC_length_l2062_206207


namespace NUMINAMATH_CALUDE_vector_q_in_terms_of_c_and_d_l2062_206241

/-- Given a line segment CD and points P and Q, where P divides CD internally
    in the ratio 3:5 and Q divides DP externally in the ratio 1:2,
    prove that vector Q can be expressed in terms of vectors C and D. -/
theorem vector_q_in_terms_of_c_and_d
  (C D P Q : EuclideanSpace ℝ (Fin 3))
  (h_P_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • C + t • D)
  (h_CP_PD : ∃ k : ℝ, k > 0 ∧ dist C P = k * (3 / 8) ∧ dist P D = k * (5 / 8))
  (h_Q_external : ∃ s : ℝ, s < 0 ∧ Q = (1 - s) • D + s • P ∧ abs s = 2) :
  Q = (5 / 8) • C + (-13 / 8) • D :=
by sorry

end NUMINAMATH_CALUDE_vector_q_in_terms_of_c_and_d_l2062_206241


namespace NUMINAMATH_CALUDE_total_distance_is_20_l2062_206232

/-- Represents the walking scenario with given speeds and total time -/
structure WalkingScenario where
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ
  total_time : ℝ

/-- Calculates the total distance walked given a WalkingScenario -/
def total_distance (s : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating that the total distance walked is 20 km -/
theorem total_distance_is_20 (s : WalkingScenario) 
  (h1 : s.flat_speed = 4)
  (h2 : s.uphill_speed = 3)
  (h3 : s.downhill_speed = 6)
  (h4 : s.total_time = 5) :
  total_distance s = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_20_l2062_206232


namespace NUMINAMATH_CALUDE_p_minus_q_equals_two_l2062_206221

-- Define an invertible function g
variable (g : ℝ → ℝ)
variable (hg : Function.Injective g)

-- Define p and q based on the given conditions
variable (p q : ℝ)
variable (hp : g p = 3)
variable (hq : g q = 5)

-- State the theorem
theorem p_minus_q_equals_two : p - q = 2 := by
  sorry

end NUMINAMATH_CALUDE_p_minus_q_equals_two_l2062_206221


namespace NUMINAMATH_CALUDE_x_fifth_plus_inverse_l2062_206293

theorem x_fifth_plus_inverse (x : ℝ) (h_pos : x > 0) (h_eq : x^2 + 1/x^2 = 7) :
  x^5 + 1/x^5 = 123 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_plus_inverse_l2062_206293


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2062_206261

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2062_206261


namespace NUMINAMATH_CALUDE_love_betty_jane_l2062_206253

variable (A B : Prop)

theorem love_betty_jane : ((A → B) → A) → (A ∧ (B ∨ ¬B)) :=
  sorry

end NUMINAMATH_CALUDE_love_betty_jane_l2062_206253


namespace NUMINAMATH_CALUDE_line_intercepts_sum_zero_l2062_206237

/-- Given a line l with equation 2x+(k-3)y-2k+6=0 where k ≠ 3,
    if the sum of its x-intercept and y-intercept is 0, then k = 1 -/
theorem line_intercepts_sum_zero (k : ℝ) (h1 : k ≠ 3) :
  (∃ x y : ℝ, 2*x + (k-3)*y - 2*k + 6 = 0) →
  (∃ x_int y_int : ℝ,
    (2*x_int - 2*k + 6 = 0) ∧
    ((k-3)*y_int - 2*k + 6 = 0) ∧
    (x_int + y_int = 0)) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_zero_l2062_206237


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l2062_206272

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) → 
  (b^3 - 15*b^2 + 22*b - 8 = 0) → 
  (c^3 - 15*c^2 + 22*c - 8 = 0) → 
  (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l2062_206272


namespace NUMINAMATH_CALUDE_smallest_common_divisor_l2062_206214

theorem smallest_common_divisor : ∃ (x : ℕ), 
  x - 16 = 136 ∧ 
  (∀ d : ℕ, d > 0 ∧ d ∣ 136 ∧ d ∣ 6 ∧ d ∣ 8 ∧ d ∣ 10 → d ≥ 2) ∧
  2 ∣ 136 ∧ 2 ∣ 6 ∧ 2 ∣ 8 ∧ 2 ∣ 10 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_common_divisor_l2062_206214


namespace NUMINAMATH_CALUDE_product_equals_eight_l2062_206286

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem product_equals_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_nonzero : ∀ n, a n ≠ 0)
  (h_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (h_equal : b 6 = a 6) :
  b 1 * b 7 * b 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l2062_206286


namespace NUMINAMATH_CALUDE_negation_of_conditional_l2062_206295

theorem negation_of_conditional (a b : ℝ) :
  ¬(a > b → 2*a > 2*b) ↔ (a ≤ b → 2*a ≤ 2*b) := by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l2062_206295


namespace NUMINAMATH_CALUDE_stadium_length_conversion_l2062_206292

/-- Converts yards to feet given a conversion factor -/
def yards_to_feet (yards : ℕ) (conversion_factor : ℕ) : ℕ :=
  yards * conversion_factor

/-- Theorem: The stadium length of 62 yards is equal to 186 feet -/
theorem stadium_length_conversion :
  yards_to_feet 62 3 = 186 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_conversion_l2062_206292


namespace NUMINAMATH_CALUDE_sugar_price_increase_l2062_206234

theorem sugar_price_increase (initial_price : ℝ) (initial_quantity : ℝ) : 
  initial_quantity > 0 →
  initial_price > 0 →
  initial_price * initial_quantity = 5 * (0.4 * initial_quantity) →
  initial_price = 2 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l2062_206234


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2062_206224

theorem nested_fraction_evaluation :
  1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2062_206224


namespace NUMINAMATH_CALUDE_problem_solution_l2062_206285

theorem problem_solution (a b c : ℚ) : 
  8 = (2 / 100) * a → 
  2 = (8 / 100) * b → 
  c = b / a → 
  c = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2062_206285


namespace NUMINAMATH_CALUDE_last_four_average_l2062_206267

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / list.length) = 65 →
  ((list.take 3).sum / 3) = 60 →
  ((list.drop 3).sum / 4) = 68.75 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l2062_206267


namespace NUMINAMATH_CALUDE_raisins_cranberries_fraction_l2062_206215

/-- Represents the quantities (in pounds) of each ingredient in the mixture -/
structure Quantities where
  raisins : ℕ
  almonds : ℕ
  cashews : ℕ
  walnuts : ℕ
  dried_apricots : ℕ
  dried_cranberries : ℕ

/-- Represents the prices (in dollars per pound) of each ingredient -/
structure Prices where
  raisins : ℕ
  almonds : ℕ
  cashews : ℕ
  walnuts : ℕ
  dried_apricots : ℕ
  dried_cranberries : ℕ

/-- Calculates the total cost of the mixture -/
def total_cost (q : Quantities) (p : Prices) : ℕ :=
  q.raisins * p.raisins + q.almonds * p.almonds + q.cashews * p.cashews +
  q.walnuts * p.walnuts + q.dried_apricots * p.dried_apricots + q.dried_cranberries * p.dried_cranberries

/-- Calculates the cost of raisins and dried cranberries combined -/
def raisins_cranberries_cost (q : Quantities) (p : Prices) : ℕ :=
  q.raisins * p.raisins + q.dried_cranberries * p.dried_cranberries

/-- Theorem stating that the fraction of the total cost that is the cost of raisins and dried cranberries is 19/107 -/
theorem raisins_cranberries_fraction (q : Quantities) (p : Prices)
  (h_quantities : q = { raisins := 5, almonds := 4, cashews := 3, walnuts := 2, dried_apricots := 4, dried_cranberries := 3 })
  (h_prices : p = { raisins := 2, almonds := 6, cashews := 8, walnuts := 10, dried_apricots := 5, dried_cranberries := 3 }) :
  (raisins_cranberries_cost q p : ℚ) / (total_cost q p) = 19 / 107 := by
  sorry

end NUMINAMATH_CALUDE_raisins_cranberries_fraction_l2062_206215


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_500_l2062_206209

def isPowerOfTwo (n : ℕ) : Prop := ∃ k, n = 2^k

def isDistinctPowersOfTwoSum (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (λ e => 2^e)).sum ∧ 
  exponents.length ≥ 2 ∧
  exponents.Nodup

theorem least_exponent_sum_for_500 :
  ∃ (exponents : List ℕ),
    isDistinctPowersOfTwoSum 500 exponents ∧
    exponents.sum = 32 ∧
    ∀ (other_exponents : List ℕ),
      isDistinctPowersOfTwoSum 500 other_exponents →
      other_exponents.sum ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_500_l2062_206209


namespace NUMINAMATH_CALUDE_function_range_l2062_206255

theorem function_range (f : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = Real.sin x - Real.sqrt 3 * Real.cos x) →
  Set.range f = Set.Icc (-Real.sqrt 3) 1 := by
sorry

end NUMINAMATH_CALUDE_function_range_l2062_206255


namespace NUMINAMATH_CALUDE_inequality_proof_l2062_206242

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y) / (y + z) + (y^2 * z) / (z + x) + (z^2 * x) / (x + y) ≥ (1/2) * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2062_206242


namespace NUMINAMATH_CALUDE_smallest_multiple_with_remainder_l2062_206260

theorem smallest_multiple_with_remainder : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → 
    (m % 5 = 0 ∧ m % 7 = 0 ∧ m % 3 = 1) → False) ∧ 
  n % 5 = 0 ∧ n % 7 = 0 ∧ n % 3 = 1 :=
by
  use 70
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_remainder_l2062_206260


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_value_l2062_206284

theorem absolute_sum_zero_implies_value (a b : ℝ) :
  |3*a + b + 5| + |2*a - 2*b - 2| = 0 →
  2*a^2 - 3*a*b = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_value_l2062_206284


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2062_206240

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 20)
  (h2 : c.left_handed = 8)
  (h3 : c.jazz_lovers = 15)
  (h4 : c.right_handed_non_jazz = 2)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  c.left_handed + c.jazz_lovers - c.total_members + c.right_handed_non_jazz = 5 := by
  sorry

#check left_handed_jazz_lovers

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2062_206240


namespace NUMINAMATH_CALUDE_quadratic_coefficient_not_one_l2062_206200

/-- A quadratic equation in x is of the form px^2 + qx + r = 0 where p ≠ 0 -/
def is_quadratic_equation (p q r : ℝ) : Prop := p ≠ 0

theorem quadratic_coefficient_not_one (a : ℝ) :
  is_quadratic_equation (a - 1) (-1) 7 → a ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_not_one_l2062_206200


namespace NUMINAMATH_CALUDE_smallest_N_proof_l2062_206277

/-- The smallest natural number N such that N × 999 consists entirely of the digit seven in its decimal representation -/
def smallest_N : ℕ := 778556334111889667445223

/-- Predicate to check if a natural number consists entirely of the digit seven -/
def all_sevens (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

theorem smallest_N_proof :
  (smallest_N * 999).digits 10 = List.replicate 27 7 ∧
  ∀ m : ℕ, m < smallest_N → ¬(all_sevens (m * 999)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l2062_206277


namespace NUMINAMATH_CALUDE_new_ratio_second_term_l2062_206252

theorem new_ratio_second_term 
  (a b x : ℤ) 
  (h1 : a = 7)
  (h2 : b = 11)
  (h3 : x = 5)
  (h4 : a + x = 3) :
  ∃ y : ℤ, (a + x) * y = 3 * (b + x) ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_second_term_l2062_206252


namespace NUMINAMATH_CALUDE_digit_sum_possibilities_l2062_206211

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Predicate to check if four digits are all different -/
def all_different (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- The theorem stating the possible sums of four different digits -/
theorem digit_sum_possibilities (a b c d : Digit) 
  (h : all_different a b c d) :
  (a.val + b.val + c.val + d.val = 10) ∨ 
  (a.val + b.val + c.val + d.val = 18) ∨ 
  (a.val + b.val + c.val + d.val = 19) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_possibilities_l2062_206211


namespace NUMINAMATH_CALUDE_parabola_properties_l2062_206271

-- Define the parabolas
def parabola_G (x y : ℝ) : Prop := x^2 = y
def parabola_M (x y : ℝ) : Prop := y^2 = 4*x

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the focus of parabola M
def focus_M : ℝ × ℝ := (1, 0)

-- Define the property of being inscribed in a parabola
def inscribed_in_parabola (t : Triangle) (p : ℝ → ℝ → Prop) : Prop :=
  p t.A.1 t.A.2 ∧ p t.B.1 t.B.2 ∧ p t.C.1 t.C.2

-- Define the property of a line being tangent to a parabola
def line_tangent_to_parabola (p q : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (t : ℝ), parabola ((1-t)*p.1 + t*q.1) ((1-t)*p.2 + t*q.2)

-- Define the property of points being concyclic
def concyclic (p q r s : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧
    (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2 ∧
    (r.1 - center.1)^2 + (r.2 - center.2)^2 = radius^2 ∧
    (s.1 - center.1)^2 + (s.2 - center.2)^2 = radius^2

theorem parabola_properties (t : Triangle) 
  (h1 : inscribed_in_parabola t parabola_G)
  (h2 : line_tangent_to_parabola t.A t.B parabola_M)
  (h3 : line_tangent_to_parabola t.A t.C parabola_M) :
  line_tangent_to_parabola t.B t.C parabola_M ∧
  concyclic t.A t.C t.B focus_M := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2062_206271


namespace NUMINAMATH_CALUDE_find_k_l2062_206244

theorem find_k : ∃ k : ℝ, (5 * 2 - k * 3 - 7 = 0) ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2062_206244


namespace NUMINAMATH_CALUDE_brittany_age_is_32_l2062_206220

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

/-- Theorem: Brittany's age when she returns from vacation is 32 -/
theorem brittany_age_is_32 : brittany_age_after_vacation 25 3 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_brittany_age_is_32_l2062_206220


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2062_206231

/-- An isosceles triangle with altitude 10 to its base and perimeter 40 has area 75 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 →  -- positive base and side lengths
  2 * s + 2 * b = 40 →  -- perimeter condition
  s^2 = b^2 + 100 →  -- Pythagorean theorem with altitude 10
  b * 10 = 75 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2062_206231


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2062_206298

/-- In an isosceles triangle XYZ, where angle X is congruent to angle Z,
    and angle Z is five times angle Y, the measure of angle X is 900/11 degrees. -/
theorem isosceles_triangle_angle_measure (X Y Z : ℝ) : 
  X = Z →                   -- Angle X is congruent to angle Z
  Z = 5 * Y →               -- Angle Z is five times angle Y
  X + Y + Z = 180 →         -- Sum of angles in a triangle is 180 degrees
  X = 900 / 11 :=           -- Measure of angle X is 900/11 degrees
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2062_206298


namespace NUMINAMATH_CALUDE_tina_july_savings_l2062_206238

/-- Represents Tina's savings and spending --/
structure TinaSavings where
  june : ℕ
  july : ℕ
  august : ℕ
  books_spent : ℕ
  shoes_spent : ℕ
  remaining : ℕ

/-- Theorem stating that Tina saved $14 in July --/
theorem tina_july_savings (s : TinaSavings) 
  (h1 : s.june = 27)
  (h2 : s.august = 21)
  (h3 : s.books_spent = 5)
  (h4 : s.shoes_spent = 17)
  (h5 : s.remaining = 40)
  (h6 : s.june + s.july + s.august = s.books_spent + s.shoes_spent + s.remaining) :
  s.july = 14 := by
  sorry


end NUMINAMATH_CALUDE_tina_july_savings_l2062_206238


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2062_206274

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem quadratic_one_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2062_206274


namespace NUMINAMATH_CALUDE_points_in_quadrant_I_l2062_206276

theorem points_in_quadrant_I (x y : ℝ) : y > 3*x ∧ y > 5 - 2*x → x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrant_I_l2062_206276


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2062_206201

/-- Conversion from polar to rectangular coordinates -/
theorem polar_to_rectangular (r : ℝ) (θ : ℝ) :
  let (x, y) := (r * Real.cos θ, r * Real.sin θ)
  (x, y) = (5 / 2, -5 * Real.sqrt 3 / 2) ↔ r = 5 ∧ θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2062_206201


namespace NUMINAMATH_CALUDE_min_modulus_of_complex_l2062_206222

theorem min_modulus_of_complex (t : ℝ) : 
  let z : ℂ := (t - 1) + (t + 1) * I
  ∃ (m : ℝ), (∀ t : ℝ, Complex.abs z ≥ m) ∧ (∃ t₀ : ℝ, Complex.abs (((t₀ - 1) : ℂ) + (t₀ + 1) * I) = m) ∧ m = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_of_complex_l2062_206222


namespace NUMINAMATH_CALUDE_x_range_when_p_and_not_q_x_in_range_implies_p_and_not_q_l2062_206212

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x : ℝ) : Prop := 1/(3 - x) > 1

-- Define the set representing the range of x
def range_x : Set ℝ := {x | x < -3 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3}

-- Theorem statement
theorem x_range_when_p_and_not_q (x : ℝ) :
  p x ∧ ¬(q x) → x ∈ range_x :=
by
  sorry

-- Theorem for the converse (to show equivalence)
theorem x_in_range_implies_p_and_not_q (x : ℝ) :
  x ∈ range_x → p x ∧ ¬(q x) :=
by
  sorry

end NUMINAMATH_CALUDE_x_range_when_p_and_not_q_x_in_range_implies_p_and_not_q_l2062_206212


namespace NUMINAMATH_CALUDE_cube_root_of_zero_l2062_206210

theorem cube_root_of_zero (x : ℝ) : x^3 = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_zero_l2062_206210


namespace NUMINAMATH_CALUDE_book_distribution_l2062_206281

theorem book_distribution (x : ℕ) : 
  (∃ n : ℕ, x = 5 * n + 6) ∧ 
  (1 ≤ x - 7 * ((x - 6) / 5 - 1) ∧ x - 7 * ((x - 6) / 5 - 1) < 7) ↔ 
  (1 ≤ x - 7 * ((x - 6) / 5 - 1) ∧ x - 7 * ((x - 6) / 5 - 1) < 7) :=
sorry

end NUMINAMATH_CALUDE_book_distribution_l2062_206281


namespace NUMINAMATH_CALUDE_expression_evaluation_l2062_206230

theorem expression_evaluation : (-1)^2 + (1/2 - 7/12 + 5/6) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2062_206230


namespace NUMINAMATH_CALUDE_barbi_weight_loss_duration_l2062_206269

/-- Proves that Barbi lost weight for 12 months given the conditions -/
theorem barbi_weight_loss_duration :
  let barbi_monthly_loss : ℝ := 1.5
  let luca_yearly_loss : ℝ := 9
  let luca_years : ℕ := 11
  let weight_loss_difference : ℝ := 81

  let barbi_months : ℝ := (luca_yearly_loss * luca_years - weight_loss_difference) / barbi_monthly_loss

  barbi_months = 12 := by sorry

end NUMINAMATH_CALUDE_barbi_weight_loss_duration_l2062_206269


namespace NUMINAMATH_CALUDE_smallest_n_for_equal_cost_l2062_206228

theorem smallest_n_for_equal_cost : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬∃ (r g b : ℕ+), 18 * r = 21 * g ∧ 21 * g = 25 * b ∧ 25 * b = 24 * m) ∧
  (∃ (r g b : ℕ+), 18 * r = 21 * g ∧ 21 * g = 25 * b ∧ 25 * b = 24 * n) ∧
  n = 132 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_equal_cost_l2062_206228


namespace NUMINAMATH_CALUDE_angle_C_is_pi_over_three_max_area_equilateral_l2062_206223

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0 ∧ t.c = 2 * Real.sqrt 3

-- Theorem 1: Angle C is π/3
theorem angle_C_is_pi_over_three (t : Triangle) (h : satisfiesConditions t) : 
  t.C = π / 3 := by sorry

-- Theorem 2: Maximum area is 3√3 and occurs when the triangle is equilateral
theorem max_area_equilateral (t : Triangle) (h : satisfiesConditions t) :
  (∃ (area : ℝ), area = 3 * Real.sqrt 3 ∧ 
    area = (1/2) * t.a * t.b * Real.sin t.C ∧
    t.a = t.b ∧ t.b = t.c) := by sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_over_three_max_area_equilateral_l2062_206223


namespace NUMINAMATH_CALUDE_at_most_12_moves_for_9_l2062_206262

/-- A move is defined as reversing the order of any block of consecutive increasing or decreasing numbers -/
def is_valid_move (perm : List Nat) (start finish : Nat) : Prop :=
  start < finish ∧ finish ≤ perm.length ∧
  (∀ i, start < i ∧ i < finish → perm[i-1]! < perm[i]! ∨ perm[i-1]! > perm[i]!)

/-- The function that counts the minimum number of moves needed to sort a permutation -/
def min_moves_to_sort (perm : List Nat) : Nat :=
  sorry

/-- Theorem stating that at most 12 moves are needed to sort any permutation of numbers from 1 to 9 -/
theorem at_most_12_moves_for_9 :
  ∀ perm : List Nat, perm.Nodup → perm.length = 9 → (∀ n, n ∈ perm ↔ 1 ≤ n ∧ n ≤ 9) →
  min_moves_to_sort perm ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_at_most_12_moves_for_9_l2062_206262


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l2062_206275

/-- The value of p for which the focus of the parabola y² = 2px (p > 0) 
    coincides with the right focus of the hyperbola x² - y² = 2 -/
theorem parabola_hyperbola_focus_coincidence (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 - y^2 = 2 ∧ x = p/2 ∧ x = 2) → 
  p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l2062_206275


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2062_206288

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- Theorem about a specific arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 1 + seq.a 3 = 16)
    (h2 : seq.S 4 = 28) :
  (∀ n : ℕ, seq.a n = 12 - 2 * n) ∧
  (∀ n : ℕ, n ≤ 5 → seq.S n ≤ seq.S 5) ∧
  seq.S 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2062_206288


namespace NUMINAMATH_CALUDE_problem_solution_l2062_206249

theorem problem_solution : (((Real.sqrt 25 - 1) / 2) ^ 2 + 3) ⁻¹ * 10 = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2062_206249


namespace NUMINAMATH_CALUDE_quadratic_root_l2062_206257

theorem quadratic_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : 9*a - 3*b + c = 0) : 
  a*(-3)^2 + b*(-3) + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_l2062_206257


namespace NUMINAMATH_CALUDE_largest_choir_size_l2062_206243

theorem largest_choir_size : 
  ∃ (n s : ℕ), 
    n * s < 150 ∧ 
    n * s + 3 = (n + 2) * (s - 3) ∧ 
    ∀ (m n' s' : ℕ), 
      m < 150 → 
      m + 3 = n' * s' → 
      m = (n' + 2) * (s' - 3) → 
      m ≤ n * s :=
by sorry

end NUMINAMATH_CALUDE_largest_choir_size_l2062_206243


namespace NUMINAMATH_CALUDE_lake_radius_l2062_206236

/-- Given a circular lake with a diameter of 26 meters, its radius is 13 meters. -/
theorem lake_radius (lake_diameter : ℝ) (h : lake_diameter = 26) : 
  lake_diameter / 2 = 13 := by sorry

end NUMINAMATH_CALUDE_lake_radius_l2062_206236


namespace NUMINAMATH_CALUDE_circle_graph_proportion_l2062_206225

theorem circle_graph_proportion (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) :
  (y = 3.6 * x) ↔ (y / 360 = x / 100) :=
by sorry

end NUMINAMATH_CALUDE_circle_graph_proportion_l2062_206225


namespace NUMINAMATH_CALUDE_all_X_composite_except_101_l2062_206203

def X (n : ℕ) : ℕ := 
  (10^(2*n + 1) - 1) / 9

theorem all_X_composite_except_101 (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ X n = a * b :=
sorry

end NUMINAMATH_CALUDE_all_X_composite_except_101_l2062_206203
