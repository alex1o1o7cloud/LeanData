import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_specific_values_l1233_123335

theorem difference_of_squares_specific_values :
  let x : ℤ := 10
  let y : ℤ := 15
  (x - y) * (x + y) = -125 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_specific_values_l1233_123335


namespace NUMINAMATH_CALUDE_mini_marshmallows_count_l1233_123370

/-- Calculates the number of mini marshmallows used in a recipe --/
def mini_marshmallows_used (total_marshmallows : ℕ) (large_marshmallows : ℕ) : ℕ :=
  total_marshmallows - large_marshmallows

/-- Proves that the number of mini marshmallows used is correct --/
theorem mini_marshmallows_count 
  (total_marshmallows : ℕ) 
  (large_marshmallows : ℕ) 
  (h : large_marshmallows ≤ total_marshmallows) :
  mini_marshmallows_used total_marshmallows large_marshmallows = 
    total_marshmallows - large_marshmallows :=
by
  sorry

#eval mini_marshmallows_used 18 8  -- Should output 10

end NUMINAMATH_CALUDE_mini_marshmallows_count_l1233_123370


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1233_123388

/-- If the quadratic equation x^2 - 3x + 2k = 0 has a root of 1, then k = 1 -/
theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*k = 0) ∧ (1^2 - 3*1 + 2*k = 0) → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1233_123388


namespace NUMINAMATH_CALUDE_distinct_values_count_l1233_123332

def original_expression : ℕ → ℕ := λ n => 3^(3^(3^3))

def parenthesization1 : ℕ → ℕ := λ n => 3^((3^3)^3)
def parenthesization2 : ℕ → ℕ := λ n => 3^(3^(3^3 + 1))
def parenthesization3 : ℕ → ℕ := λ n => (3^(3^3))^3

theorem distinct_values_count :
  ∃ (S : Finset ℕ),
    S.card = 4 ∧
    (∀ x ∈ S, x ≠ original_expression 0) ∧
    (∀ x ∈ S, (x = parenthesization1 0) ∨ (x = parenthesization2 0) ∨ (x = parenthesization3 0) ∨
              (∃ y, x = 3^y ∧ y ≠ 3^(3^3))) :=
by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l1233_123332


namespace NUMINAMATH_CALUDE_k_set_characterization_l1233_123391

theorem k_set_characterization (r : ℕ) :
  let h := 2^r
  let k_set := {k : ℕ | ∃ (m n : ℕ), 
    Odd m ∧ m > 1 ∧
    k ∣ m^k - 1 ∧
    m ∣ n^((m^k - 1)/k) + 1}
  k_set = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ ¬ Even t} :=
by sorry

end NUMINAMATH_CALUDE_k_set_characterization_l1233_123391


namespace NUMINAMATH_CALUDE_pig_feed_per_day_l1233_123344

/-- Given that Randy has 2 pigs and they are fed 140 pounds of pig feed per week,
    prove that each pig is fed 10 pounds of feed per day. -/
theorem pig_feed_per_day (num_pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ) :
  num_pigs = 2 →
  total_feed_per_week = 140 →
  days_per_week = 7 →
  (total_feed_per_week / num_pigs) / days_per_week = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_pig_feed_per_day_l1233_123344


namespace NUMINAMATH_CALUDE_existence_implies_lower_bound_l1233_123360

theorem existence_implies_lower_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_lower_bound_l1233_123360


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l1233_123379

-- Define a quadrilateral in a plane
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define vector equality
def vector_equal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 = v2.1 ∧ v1.2 = v2.2

-- Define vector scaling
def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

-- Define condition p
def condition_p (q : Quadrilateral) : Prop :=
  vector_equal (q.B.1 - q.A.1, q.B.2 - q.A.2) (vector_scale 2 (q.C.1 - q.D.1, q.C.2 - q.D.2))

-- Define a trapezoid
def is_trapezoid (q : Quadrilateral) : Prop :=
  (q.A.2 - q.B.2) / (q.A.1 - q.B.1) = (q.D.2 - q.C.2) / (q.D.1 - q.C.1) ∨
  (q.A.2 - q.D.2) / (q.A.1 - q.D.1) = (q.B.2 - q.C.2) / (q.B.1 - q.C.1)

-- Theorem statement
theorem condition_p_sufficient_not_necessary (q : Quadrilateral) :
  (condition_p q → is_trapezoid q) ∧ ¬(is_trapezoid q → condition_p q) :=
sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l1233_123379


namespace NUMINAMATH_CALUDE_syllogism_cos_periodic_l1233_123351

-- Define the properties
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the cosine function
def cos : ℝ → ℝ := sorry

-- Theorem to prove
theorem syllogism_cos_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  (IsTrigonometric cos) →
  (IsPeriodic cos) := by sorry

end NUMINAMATH_CALUDE_syllogism_cos_periodic_l1233_123351


namespace NUMINAMATH_CALUDE_equation_transformation_l1233_123346

theorem equation_transformation (x y : ℝ) : x - 3 = y - 3 → x - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1233_123346


namespace NUMINAMATH_CALUDE_multiple_of_nine_square_greater_than_144_less_than_30_l1233_123316

theorem multiple_of_nine_square_greater_than_144_less_than_30 (x : ℕ) :
  (∃ k : ℕ, x = 9 * k) →
  x^2 > 144 →
  x < 30 →
  x = 18 ∨ x = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_square_greater_than_144_less_than_30_l1233_123316


namespace NUMINAMATH_CALUDE_strictly_increasing_interval_l1233_123340

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem strictly_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 3) (k * π + π / 6)) :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_interval_l1233_123340


namespace NUMINAMATH_CALUDE_satisfaction_survey_stats_l1233_123302

def data : List ℝ := [34, 35, 35, 36]

theorem satisfaction_survey_stats (median mode mean variance : ℝ) :
  median = 35 ∧
  mode = 35 ∧
  mean = 35 ∧
  variance = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_satisfaction_survey_stats_l1233_123302


namespace NUMINAMATH_CALUDE_division_remainder_zero_l1233_123313

theorem division_remainder_zero (dividend : ℝ) (divisor : ℝ) (quotient : ℝ) 
  (h1 : dividend = 57843.67)
  (h2 : divisor = 1242.51)
  (h3 : quotient = 46.53) :
  dividend - divisor * quotient = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_zero_l1233_123313


namespace NUMINAMATH_CALUDE_plane_division_theorem_l1233_123318

/-- The number of regions formed by h horizontal lines and s non-horizontal lines -/
def num_regions (h s : ℕ) : ℕ := h * (s + 1) + 1 + s * (s + 1) / 2

/-- The set of valid solutions for (h, s) -/
def valid_solutions : Set (ℕ × ℕ) :=
  {(995, 1), (176, 10), (80, 21)}

theorem plane_division_theorem :
  ∀ h s : ℕ, h > 0 ∧ s > 0 →
    (num_regions h s = 1992 ↔ (h, s) ∈ valid_solutions) := by
  sorry

#check plane_division_theorem

end NUMINAMATH_CALUDE_plane_division_theorem_l1233_123318


namespace NUMINAMATH_CALUDE_enclosed_area_circular_arcs_l1233_123312

/-- The area enclosed by a curve composed of 9 congruent circular arcs -/
theorem enclosed_area_circular_arcs (n : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) 
  (h1 : n = 9)
  (h2 : arc_length = 5 * π / 6)
  (h3 : hexagon_side = 3) :
  ∃ (area : ℝ), area = 13.5 * Real.sqrt 3 + 375 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_circular_arcs_l1233_123312


namespace NUMINAMATH_CALUDE_choose_three_from_fifteen_l1233_123371

theorem choose_three_from_fifteen (n k : ℕ) : n = 15 ∧ k = 3 → Nat.choose n k = 455 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_fifteen_l1233_123371


namespace NUMINAMATH_CALUDE_sum_of_digits_of_five_to_23_l1233_123390

/-- The sum of the tens digit and the ones digit of (2+3)^23 is 7 -/
theorem sum_of_digits_of_five_to_23 :
  let n : ℕ := (2 + 3)^23
  (n / 10 % 10) + (n % 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_five_to_23_l1233_123390


namespace NUMINAMATH_CALUDE_project_completion_time_l1233_123305

/-- Represents the completion of an engineering project --/
structure Project where
  initialWorkers : ℕ
  initialWorkCompleted : ℚ
  initialDuration : ℕ
  additionalWorkers : ℕ

/-- Calculates the total days required to complete the project --/
def totalDays (p : Project) : ℕ :=
  let totalWorkers := p.initialWorkers + p.additionalWorkers
  let remainingWork := 1 - p.initialWorkCompleted
  let initialWorkRate := p.initialWorkCompleted / p.initialDuration
  let totalWorkRate := initialWorkRate * totalWorkers / p.initialWorkers
  p.initialDuration + (remainingWork / totalWorkRate).ceil.toNat

/-- Theorem stating that for the given project parameters, the total days to complete is 70 --/
theorem project_completion_time (p : Project) 
  (h1 : p.initialWorkers = 6)
  (h2 : p.initialWorkCompleted = 1/3)
  (h3 : p.initialDuration = 35)
  (h4 : p.additionalWorkers = 6) :
  totalDays p = 70 := by
  sorry

#eval totalDays { initialWorkers := 6, initialWorkCompleted := 1/3, initialDuration := 35, additionalWorkers := 6 }

end NUMINAMATH_CALUDE_project_completion_time_l1233_123305


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1233_123333

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 3 →
  a 6 = 1 / 9 →
  a 4 * a 5 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1233_123333


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1233_123300

theorem algebraic_simplification (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1233_123300


namespace NUMINAMATH_CALUDE_sweets_distribution_l1233_123301

theorem sweets_distribution (total_sweets : ℕ) (num_children : ℕ) (remaining_fraction : ℚ) 
  (h1 : total_sweets = 288)
  (h2 : num_children = 48)
  (h3 : remaining_fraction = 1 / 3)
  : (total_sweets * (1 - remaining_fraction)) / num_children = 4 := by
  sorry

end NUMINAMATH_CALUDE_sweets_distribution_l1233_123301


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l1233_123306

theorem arctan_sum_equals_pi_half (n : ℕ+) :
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/n) = π/2 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l1233_123306


namespace NUMINAMATH_CALUDE_brendan_weekly_taxes_l1233_123380

/-- Calculates Brendan's weekly taxes paid after deduction -/
def weekly_taxes_paid (wage1 wage2 wage3 : ℚ) 
                      (hours1 hours2 hours3 : ℚ)
                      (tips1 tips2 tips3 : ℚ)
                      (reported_tips1 reported_tips2 reported_tips3 : ℚ)
                      (tax_rate1 tax_rate2 tax_rate3 : ℚ)
                      (deduction : ℚ) : ℚ :=
  let income1 := wage1 * hours1 + reported_tips1 * tips1 * hours1
  let income2 := wage2 * hours2 + reported_tips2 * tips2 * hours2
  let income3 := wage3 * hours3 + reported_tips3 * tips3 * hours3
  let taxes1 := income1 * tax_rate1
  let taxes2 := income2 * tax_rate2
  let taxes3 := income3 * tax_rate3
  taxes1 + taxes2 + taxes3 - deduction

theorem brendan_weekly_taxes :
  weekly_taxes_paid 12 15 10    -- wages
                    12 8 10     -- hours
                    20 15 5     -- tips
                    (1/2) (1/4) (3/5)  -- reported tips percentages
                    (22/100) (18/100) (16/100)  -- tax rates
                    50  -- deduction
  = 5588 / 100 := by
  sorry

end NUMINAMATH_CALUDE_brendan_weekly_taxes_l1233_123380


namespace NUMINAMATH_CALUDE_same_color_probability_l1233_123323

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def drawn_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

theorem same_color_probability : 
  (Nat.choose red_marbles drawn_marbles + 
   Nat.choose white_marbles drawn_marbles + 
   Nat.choose blue_marbles drawn_marbles : ℚ) / 
  (Nat.choose total_marbles drawn_marbles : ℚ) = 11 / 612 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1233_123323


namespace NUMINAMATH_CALUDE_kyler_won_zero_games_l1233_123363

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kyler : Player

/-- Represents the number of games won by each player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Peter => 5
  | Player.Emma => 4
  | Player.Kyler => 0  -- This is what we want to prove

/-- Represents the number of games lost by each player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Peter => 3
  | Player.Emma => 4
  | Player.Kyler => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := (games_won Player.Peter + games_won Player.Emma + games_won Player.Kyler +
                        games_lost Player.Peter + games_lost Player.Emma + games_lost Player.Kyler) / 2

theorem kyler_won_zero_games :
  games_won Player.Kyler = 0 ∧
  2 * total_games = games_won Player.Peter + games_won Player.Emma + games_won Player.Kyler +
                    games_lost Player.Peter + games_lost Player.Emma + games_lost Player.Kyler :=
by sorry

end NUMINAMATH_CALUDE_kyler_won_zero_games_l1233_123363


namespace NUMINAMATH_CALUDE_city_mpg_calculation_l1233_123393

/-- The average miles per gallon (mpg) for an SUV in the city -/
def city_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 25 gallons of gasoline -/
def max_distance : ℝ := 305

/-- The amount of gasoline in gallons used for the maximum distance -/
def gasoline_amount : ℝ := 25

/-- Theorem stating that the average mpg in the city is 12.2 -/
theorem city_mpg_calculation : city_mpg = max_distance / gasoline_amount := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_calculation_l1233_123393


namespace NUMINAMATH_CALUDE_find_first_number_l1233_123329

theorem find_first_number (x : ℝ) (y : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (y + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 128 := by
sorry

end NUMINAMATH_CALUDE_find_first_number_l1233_123329


namespace NUMINAMATH_CALUDE_daisy_toys_theorem_l1233_123398

/-- The number of dog toys Daisy's owner bought on Wednesday -/
def wednesday_toys (monday_toys tuesday_left tuesday_bought total_if_found : ℕ) : ℕ :=
  total_if_found - (tuesday_left + tuesday_bought)

theorem daisy_toys_theorem (monday_toys tuesday_left tuesday_bought total_if_found : ℕ) 
  (h1 : monday_toys = 5)
  (h2 : tuesday_left = 3)
  (h3 : tuesday_bought = 3)
  (h4 : total_if_found = 13) :
  wednesday_toys monday_toys tuesday_left tuesday_bought total_if_found = 7 := by
  sorry

#eval wednesday_toys 5 3 3 13

end NUMINAMATH_CALUDE_daisy_toys_theorem_l1233_123398


namespace NUMINAMATH_CALUDE_line_slope_l1233_123339

/-- A straight line in the xy-plane with y-intercept 10 and passing through (100, 1000) has slope 9.9 -/
theorem line_slope (f : ℝ → ℝ) (h1 : f 0 = 10) (h2 : f 100 = 1000) :
  (f 100 - f 0) / (100 - 0) = 9.9 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1233_123339


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1233_123392

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), r > 0 ∧ a₁ > 0 ∧ ∀ n, a n = a₁ * r ^ (n - 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 3 + a 2 * a 6 + 2 * a 3 ^ 2 = 36) →
  (a 2 + a 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1233_123392


namespace NUMINAMATH_CALUDE_special_polyhedron_volume_l1233_123315

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  -- The polyhedron is convex
  isConvex : Bool
  -- Number of square faces
  numSquareFaces : Nat
  -- Number of hexagonal faces
  numHexagonalFaces : Nat
  -- No two square faces share a vertex
  noSharedSquareVertices : Bool
  -- All edges have unit length
  unitEdgeLength : Bool

/-- The volume of the special polyhedron -/
noncomputable def specialPolyhedronVolume (p : SpecialPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the special polyhedron -/
theorem special_polyhedron_volume :
  ∀ (p : SpecialPolyhedron),
    p.isConvex = true ∧
    p.numSquareFaces = 6 ∧
    p.numHexagonalFaces = 8 ∧
    p.noSharedSquareVertices = true ∧
    p.unitEdgeLength = true →
    specialPolyhedronVolume p = 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_volume_l1233_123315


namespace NUMINAMATH_CALUDE_sqrt_sum_upper_bound_l1233_123308

theorem sqrt_sum_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_upper_bound_l1233_123308


namespace NUMINAMATH_CALUDE_range_of_a_l1233_123367

open Set

def p (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2
def q (a : ℝ) : Prop := a ≥ -10

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ Iio (-10) ∪ Ioo (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1233_123367


namespace NUMINAMATH_CALUDE_semicircle_radius_l1233_123368

theorem semicircle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 2 = 12.5 * π) (h_xz_arc : π * y = 9 * π) :
  z / 2 = Real.sqrt 424 / 2 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l1233_123368


namespace NUMINAMATH_CALUDE_fifteen_apples_solution_l1233_123343

/-- The number of friends sharing the apples -/
def num_friends : ℕ := 5

/-- The function representing the number of apples remaining after each friend takes their share -/
def apples_remaining (initial_apples : ℚ) (friend : ℕ) : ℚ :=
  match friend with
  | 0 => initial_apples
  | n + 1 => (apples_remaining initial_apples n / 2) - (1 / 2)

/-- The theorem stating that 15 is the correct initial number of apples -/
theorem fifteen_apples_solution :
  ∃ (initial_apples : ℚ),
    initial_apples = 15 ∧
    apples_remaining initial_apples num_friends = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_apples_solution_l1233_123343


namespace NUMINAMATH_CALUDE_x_value_l1233_123389

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1233_123389


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1233_123373

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 6*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1233_123373


namespace NUMINAMATH_CALUDE_marks_age_relation_l1233_123321

/-- Proves that Mark's age will be 2 years more than twice Aaron's age in 4 years -/
theorem marks_age_relation (mark_current_age aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age - 3 = 3 * (aaron_current_age - 3) + 1 →
  (mark_current_age + 4) = 2 * (aaron_current_age + 4) + 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_age_relation_l1233_123321


namespace NUMINAMATH_CALUDE_polynomial_equation_l1233_123317

-- Define polynomials over real numbers
variable (x : ℝ)

-- Define f(x) and h(x) as polynomials
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 - 4*x + 1
def h (x : ℝ) : ℝ := -x^4 - 2*x^3 + 4*x^2 + 9*x - 5

-- State the theorem
theorem polynomial_equation :
  f x + h x = 3*x^2 + 5*x - 4 := by sorry

end NUMINAMATH_CALUDE_polynomial_equation_l1233_123317


namespace NUMINAMATH_CALUDE_inequality_proof_l1233_123395

theorem inequality_proof (a b c : ℝ) 
  (ha : -1 < a ∧ a < -2/3) 
  (hb : -1/3 < b ∧ b < 0) 
  (hc : c > 1) : 
  1/c < 1/(b-a) ∧ 1/(b-a) < 1/(a*b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1233_123395


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1233_123381

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 23 ∧ x - y = 7 → x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1233_123381


namespace NUMINAMATH_CALUDE_original_number_proof_l1233_123362

theorem original_number_proof (t : ℝ) : 
  t * (1 + 0.125) - t * (1 - 0.25) = 30 → t = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1233_123362


namespace NUMINAMATH_CALUDE_nicks_age_l1233_123334

theorem nicks_age (N : ℝ) : 
  (N + (N + 6)) / 2 + 5 = 21 → N = 13 := by
  sorry

end NUMINAMATH_CALUDE_nicks_age_l1233_123334


namespace NUMINAMATH_CALUDE_ratio_a_to_d_l1233_123319

theorem ratio_a_to_d (a b c d : ℚ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (hcd : c / d = 5 / 7) :
  a / d = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_a_to_d_l1233_123319


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l1233_123341

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 4) :
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l1233_123341


namespace NUMINAMATH_CALUDE_part_one_part_two_l1233_123354

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m-9)*x + m^2 - 9*m ≥ 0}

-- Define the complement of B in ℝ
def C_R_B (m : ℝ) : Set ℝ := (Set.univ : Set ℝ) \ B m

-- Part 1: If A ∩ B = [-3, 3], then m = 12
theorem part_one (m : ℝ) : A ∩ B m = Set.Icc (-3) 3 → m = 12 := by sorry

-- Part 2: If A ⊆ C_ℝB, then 5 < m < 6
theorem part_two (m : ℝ) : A ⊆ C_R_B m → 5 < m ∧ m < 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1233_123354


namespace NUMINAMATH_CALUDE_max_inverse_sum_14th_power_l1233_123322

/-- A quadratic polynomial x^2 - tx + q with roots r1 and r2 -/
structure QuadraticPolynomial where
  t : ℝ
  q : ℝ
  r1 : ℝ
  r2 : ℝ
  is_root : r1^2 - t*r1 + q = 0 ∧ r2^2 - t*r2 + q = 0

/-- The condition that the sum of powers of roots are equal up to 13th power -/
def equal_sum_powers (p : QuadraticPolynomial) : Prop :=
  ∀ n : ℕ, n ≤ 13 → p.r1^n + p.r2^n = p.r1 + p.r2

/-- The theorem statement -/
theorem max_inverse_sum_14th_power (p : QuadraticPolynomial) 
  (h : equal_sum_powers p) : 
  (∀ p' : QuadraticPolynomial, equal_sum_powers p' → 
    1 / p'.r1^14 + 1 / p'.r2^14 ≤ 1 / p.r1^14 + 1 / p.r2^14) →
  1 / p.r1^14 + 1 / p.r2^14 = 2 :=
sorry

end NUMINAMATH_CALUDE_max_inverse_sum_14th_power_l1233_123322


namespace NUMINAMATH_CALUDE_heart_shape_area_l1233_123348

/-- The area of a heart shape composed of specific geometric elements -/
theorem heart_shape_area : 
  let π : ℝ := 3.14
  let semicircle_diameter : ℝ := 10
  let sector_radius : ℝ := 10
  let sector_angle : ℝ := 45
  let square_side : ℝ := 10
  let semicircle_area : ℝ := 2 * (1/2 * π * (semicircle_diameter/2)^2)
  let sector_area : ℝ := 2 * ((sector_angle/360) * π * sector_radius^2)
  let square_area : ℝ := square_side^2
  semicircle_area + sector_area + square_area = 257 := by
  sorry

end NUMINAMATH_CALUDE_heart_shape_area_l1233_123348


namespace NUMINAMATH_CALUDE_calculation_proof_l1233_123310

theorem calculation_proof : (27 * 0.92 * 0.85) / (23 * 1.7 * 1.8) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1233_123310


namespace NUMINAMATH_CALUDE_largest_number_with_property_l1233_123378

/-- A function that returns true if a natural number has all distinct digits --/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a number is not divisible by 11 --/
def not_divisible_by_11 (n : ℕ) : Prop := sorry

/-- A function that returns true if all subsequences of digits in a number are not divisible by 11 --/
def all_subsequences_not_divisible_by_11 (n : ℕ) : Prop := sorry

/-- The main theorem stating that 987654321 is the largest natural number 
    with all distinct digits and all subsequences not divisible by 11 --/
theorem largest_number_with_property : 
  ∀ n : ℕ, n > 987654321 → 
  ¬(has_distinct_digits n ∧ all_subsequences_not_divisible_by_11 n) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_property_l1233_123378


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1233_123374

theorem angle_measure_proof :
  ∃ (x : ℝ), x + (3 * x - 8) = 90 ∧ x = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1233_123374


namespace NUMINAMATH_CALUDE_jogger_distance_l1233_123338

theorem jogger_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  actual_speed = 12 →
  faster_speed = 16 →
  extra_distance = 10 →
  (∃ time : ℝ, time > 0 ∧ faster_speed * time = actual_speed * time + extra_distance) →
  actual_speed * (extra_distance / (faster_speed - actual_speed)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_jogger_distance_l1233_123338


namespace NUMINAMATH_CALUDE_number_ratio_problem_l1233_123376

theorem number_ratio_problem (N : ℝ) (x : ℝ) 
  (h1 : N = 280) 
  (h2 : (1/5) * N + 4 = x * N - 10) : 
  x = 1/4 := by sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l1233_123376


namespace NUMINAMATH_CALUDE_newspaper_photos_l1233_123342

/-- The total number of photos in a newspaper -/
def total_photos (pages_with_4 : ℕ) (pages_with_6 : ℕ) : ℕ :=
  pages_with_4 * 4 + pages_with_6 * 6

/-- Theorem stating that the total number of photos is 208 -/
theorem newspaper_photos : total_photos 25 18 = 208 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_l1233_123342


namespace NUMINAMATH_CALUDE_equation_solution_l1233_123358

theorem equation_solution : 
  ∃! x : ℝ, 45 - (28 - (37 - (15 - x))) = 58 ∧ x = 19 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1233_123358


namespace NUMINAMATH_CALUDE_new_vessel_capacity_l1233_123365

/-- Given two vessels with different alcohol concentrations, prove the capacity of a new vessel that contains their combined contents plus water to achieve a specific concentration. -/
theorem new_vessel_capacity
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percent : ℝ)
  (total_liquid : ℝ)
  (new_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percent = 0.25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percent = 0.40)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 0.29000000000000004) :
  (vessel1_capacity * vessel1_alcohol_percent + vessel2_capacity * vessel2_alcohol_percent) / new_concentration = 10 := by
  sorry

#eval (2 * 0.25 + 6 * 0.40) / 0.29000000000000004

end NUMINAMATH_CALUDE_new_vessel_capacity_l1233_123365


namespace NUMINAMATH_CALUDE_circplus_commutative_l1233_123337

/-- The ⊕ operation -/
def circplus (a b : ℝ) : ℝ := a^2 + a*b + b^2

/-- Theorem: x ⊕ y = y ⊕ x for all real x and y -/
theorem circplus_commutative : ∀ x y : ℝ, circplus x y = circplus y x := by
  sorry

end NUMINAMATH_CALUDE_circplus_commutative_l1233_123337


namespace NUMINAMATH_CALUDE_simplify_expression_l1233_123349

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1233_123349


namespace NUMINAMATH_CALUDE_group_ratio_theorem_l1233_123355

/-- Represents the group composition and average ages -/
structure GroupComposition where
  avg_age : ℝ
  doc_age : ℝ
  law_age : ℝ
  eng_age : ℝ
  doc_count : ℝ
  law_count : ℝ
  eng_count : ℝ

/-- Theorem stating the ratios of group members based on given average ages -/
theorem group_ratio_theorem (g : GroupComposition) 
  (h1 : g.avg_age = 45)
  (h2 : g.doc_age = 40)
  (h3 : g.law_age = 55)
  (h4 : g.eng_age = 35)
  (h5 : g.avg_age * (g.doc_count + g.law_count + g.eng_count) = 
        g.doc_age * g.doc_count + g.law_age * g.law_count + g.eng_age * g.eng_count) :
  g.doc_count / g.law_count = 1 ∧ g.eng_count / g.law_count = 2 := by
  sorry

#check group_ratio_theorem

end NUMINAMATH_CALUDE_group_ratio_theorem_l1233_123355


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_product_l1233_123336

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_of (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_three_digit_factorial_product :
  ∃! n : ℕ, is_three_digit n ∧
    let (a, b, c) := digits_of n
    2 * n = 3 * (factorial a * factorial b * factorial c) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_product_l1233_123336


namespace NUMINAMATH_CALUDE_largest_square_area_l1233_123399

theorem largest_square_area (side_length : ℝ) (corner_size : ℝ) : 
  side_length = 5 → 
  corner_size = 1 → 
  (side_length - 2 * corner_size)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l1233_123399


namespace NUMINAMATH_CALUDE_temp_increase_proof_l1233_123330

-- Define the temperatures
def last_night_temp : Int := -5
def current_temp : Int := 3

-- Define the temperature difference function
def temp_difference (t1 t2 : Int) : Int := t2 - t1

-- Theorem to prove
theorem temp_increase_proof : 
  temp_difference last_night_temp current_temp = 8 := by
  sorry

end NUMINAMATH_CALUDE_temp_increase_proof_l1233_123330


namespace NUMINAMATH_CALUDE_triangle_shape_l1233_123304

theorem triangle_shape (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a * Real.cos A = b * Real.cos B) →
  (A = B ∨ A + B = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l1233_123304


namespace NUMINAMATH_CALUDE_quadratic_equation_two_roots_l1233_123385

-- Define the geometric progression
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = a * q^2

-- Define the quadratic equation
def has_two_distinct_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * Real.sqrt 2 * b * x₁ + c = 0 ∧
                        a * x₂^2 + 2 * Real.sqrt 2 * b * x₂ + c = 0

-- Theorem statement
theorem quadratic_equation_two_roots
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_geom : is_geometric_progression a b c) :
  has_two_distinct_roots a b c :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_roots_l1233_123385


namespace NUMINAMATH_CALUDE_cats_owned_by_olly_l1233_123397

def shoes_per_animal : ℕ := 4

def num_dogs : ℕ := 3

def num_ferrets : ℕ := 1

def total_shoes : ℕ := 24

def num_cats : ℕ := (total_shoes - (num_dogs + num_ferrets) * shoes_per_animal) / shoes_per_animal

theorem cats_owned_by_olly :
  num_cats = 2 := by sorry

end NUMINAMATH_CALUDE_cats_owned_by_olly_l1233_123397


namespace NUMINAMATH_CALUDE_quadratic_equation_k_l1233_123327

/-- Given a quadratic equation x^2 - 3x + k = 0 with two real roots a and b,
    if ab + 2a + 2b = 1, then k = -5 -/
theorem quadratic_equation_k (a b k : ℝ) :
  (∀ x, x^2 - 3*x + k = 0 ↔ x = a ∨ x = b) →
  (a*b + 2*a + 2*b = 1) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_l1233_123327


namespace NUMINAMATH_CALUDE_solution_relationship_l1233_123359

theorem solution_relationship (c c' d d' : ℝ) 
  (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : -d / (2 * c) = 2 * (-d' / (3 * c'))) :
  d / (2 * c) = 2 * d' / (3 * c') := by
  sorry

end NUMINAMATH_CALUDE_solution_relationship_l1233_123359


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l1233_123353

theorem partial_fraction_sum (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 20*x^2 + 125*x - 500 = (x - p)*(x - q)*(x - r)) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20*s^2 + 125*s - 500) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 720 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l1233_123353


namespace NUMINAMATH_CALUDE_correct_result_largest_negative_integer_result_l1233_123387

/-- Given polynomial A -/
def A (x : ℝ) : ℝ := 3 * x^2 - x + 1

/-- Given polynomial B -/
def B (x : ℝ) : ℝ := -x^2 - 2*x - 3

/-- Theorem stating the correct result of A - B -/
theorem correct_result (x : ℝ) : A x - B x = 4 * x^2 + x + 4 := by sorry

/-- Theorem stating the value of A - B when x is the largest negative integer -/
theorem largest_negative_integer_result : A (-1) - B (-1) = 7 := by sorry

end NUMINAMATH_CALUDE_correct_result_largest_negative_integer_result_l1233_123387


namespace NUMINAMATH_CALUDE_tangerines_per_day_l1233_123314

theorem tangerines_per_day 
  (initial : ℕ) 
  (days : ℕ) 
  (remaining : ℕ) 
  (h1 : initial > remaining) 
  (h2 : days > 0) : 
  (initial - remaining) / days = (initial - remaining) / days :=
by sorry

end NUMINAMATH_CALUDE_tangerines_per_day_l1233_123314


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l1233_123311

/-- The number of legs a centipede has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid orders for a centipede to put on socks and shoes -/
def valid_orders : ℕ := Nat.factorial total_items / (2 ^ num_legs)

/-- Theorem stating the number of valid orders for a centipede to put on socks and shoes -/
theorem centipede_sock_shoe_orders :
  valid_orders = Nat.factorial total_items / (2 ^ num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l1233_123311


namespace NUMINAMATH_CALUDE_sports_league_games_l1233_123352

/-- Calculates the number of games in a sports league with two divisions -/
theorem sports_league_games (n₁ n₂ : ℕ) (intra_games : ℕ) (inter_games : ℕ) :
  n₁ = 5 →
  n₂ = 6 →
  intra_games = 3 →
  inter_games = 2 →
  (n₁ * (n₁ - 1) * intra_games / 2) +
  (n₂ * (n₂ - 1) * intra_games / 2) +
  (n₁ * n₂ * inter_games) = 135 :=
by
  sorry

#check sports_league_games

end NUMINAMATH_CALUDE_sports_league_games_l1233_123352


namespace NUMINAMATH_CALUDE_total_crayons_l1233_123382

theorem total_crayons (billy_crayons jane_crayons : Float) 
  (h1 : billy_crayons = 62.0) 
  (h2 : jane_crayons = 52.0) : 
  billy_crayons + jane_crayons = 114.0 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1233_123382


namespace NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l1233_123347

-- Define the geometric bodies
inductive GeometricBody
| Pyramid
| Prism
| Frustum
| Cuboid

-- Define a function to count parallel surfaces
def parallelSurfaces : GeometricBody → ℕ
| GeometricBody.Pyramid => 0
| GeometricBody.Prism => 6
| GeometricBody.Frustum => 2
| GeometricBody.Cuboid => 6

-- Theorem: Only the frustum has exactly two parallel surfaces
theorem only_frustum_has_two_parallel_surfaces :
  ∀ b : GeometricBody, parallelSurfaces b = 2 ↔ b = GeometricBody.Frustum :=
by sorry

end NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l1233_123347


namespace NUMINAMATH_CALUDE_bug_position_after_3000_jumps_l1233_123369

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point
| seven : Point

/-- Determines if a point is odd-numbered -/
def isOdd : Point → Bool
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true
  | Point.six => false
  | Point.seven => true

/-- Performs a single jump based on the current point -/
def jump : Point → Point
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.seven
  | Point.five => Point.seven
  | Point.six => Point.two
  | Point.seven => Point.two

/-- Performs multiple jumps -/
def multiJump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (multiJump start n)

theorem bug_position_after_3000_jumps :
  multiJump Point.seven 3000 = Point.two :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_3000_jumps_l1233_123369


namespace NUMINAMATH_CALUDE_division_problem_l1233_123372

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = 172 →
  divisor = 17 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  quotient = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1233_123372


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1233_123377

theorem balls_in_boxes (x y z : ℕ) : 
  x + y + z = 320 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  ∃ (a b c : ℕ), a ≤ x ∧ b ≤ y ∧ c ≤ z ∧ 6*a + 11*b + 15*c = 1001 :=
by sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1233_123377


namespace NUMINAMATH_CALUDE_boundary_length_special_square_l1233_123375

/-- The length of the boundary of a special figure constructed from a square --/
theorem boundary_length_special_square : 
  ∀ (s : Real) (a : Real),
    s * s = 64 →  -- area of the square is 64
    a = s / 4 →   -- length of each arc segment
    (16 : Real) + 14 * Real.pi = 
      4 * s +     -- sum of straight segments
      12 * (a * Real.pi / 2) +  -- sum of side arcs
      4 * (a * Real.pi / 2)     -- sum of corner arcs
    := by sorry

end NUMINAMATH_CALUDE_boundary_length_special_square_l1233_123375


namespace NUMINAMATH_CALUDE_inequality_relations_l1233_123328

theorem inequality_relations (r p q : ℝ) 
  (hr : r > 0) (hp : p > 0) (hq : q > 0) (hpq : p^2 * r > q^2 * r) : 
  p > q ∧ |p| > |q| ∧ 1/p < 1/q := by sorry

end NUMINAMATH_CALUDE_inequality_relations_l1233_123328


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1233_123383

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

/-- The theorem stating that the two given circles are externally tangent -/
theorem circles_externally_tangent :
  let c1 : ℝ × ℝ := (0, 8)
  let c2 : ℝ × ℝ := (-6, 0)
  let r1 : ℝ := 6
  let r2 : ℝ := 2
  externally_tangent c1 c2 r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1233_123383


namespace NUMINAMATH_CALUDE_natural_number_equation_solutions_l1233_123303

theorem natural_number_equation_solutions (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_equation_solutions_l1233_123303


namespace NUMINAMATH_CALUDE_fifth_store_cars_l1233_123386

def store_count : Nat := 5
def car_counts : Vector Nat 4 := ⟨[30, 14, 14, 21], rfl⟩
def mean : Rat := 104/5

theorem fifth_store_cars : 
  ∃ x : Nat, (car_counts.toList.sum + x) / store_count = mean :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_store_cars_l1233_123386


namespace NUMINAMATH_CALUDE_sprint_no_wind_time_l1233_123345

/-- A sprinter's performance under different wind conditions -/
structure SprintPerformance where
  with_wind_distance : ℝ
  against_wind_distance : ℝ
  time_with_wind : ℝ
  time_against_wind : ℝ
  wind_speed : ℝ
  no_wind_speed : ℝ

/-- Theorem stating the time taken to run 100 meters in no wind condition -/
theorem sprint_no_wind_time (perf : SprintPerformance) 
  (h1 : perf.with_wind_distance = 90)
  (h2 : perf.against_wind_distance = 70)
  (h3 : perf.time_with_wind = 10)
  (h4 : perf.time_against_wind = 10)
  (h5 : perf.with_wind_distance / (perf.no_wind_speed + perf.wind_speed) = perf.time_with_wind)
  (h6 : perf.against_wind_distance / (perf.no_wind_speed - perf.wind_speed) = perf.time_against_wind)
  : (100 : ℝ) / perf.no_wind_speed = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_sprint_no_wind_time_l1233_123345


namespace NUMINAMATH_CALUDE_triangle_area_10_24_26_l1233_123331

/-- The area of a triangle with side lengths 10, 24, and 26 is 120 -/
theorem triangle_area_10_24_26 : 
  ∀ (a b c area : ℝ), 
    a = 10 → b = 24 → c = 26 →
    (a * a + b * b = c * c) →  -- Pythagorean theorem condition
    area = (1/2) * a * b →
    area = 120 := by sorry

end NUMINAMATH_CALUDE_triangle_area_10_24_26_l1233_123331


namespace NUMINAMATH_CALUDE_problem_sample_is_valid_problem_sample_sequence_correct_l1233_123356

/-- Represents a systematic sample -/
structure SystematicSample where
  first : ℕ
  interval : ℕ
  size : ℕ
  population : ℕ

/-- Checks if a systematic sample is valid -/
def isValidSystematicSample (s : SystematicSample) : Prop :=
  s.first > 0 ∧
  s.first ≤ s.population ∧
  s.interval > 0 ∧
  s.size > 0 ∧
  s.population ≥ s.size ∧
  ∀ i : ℕ, i < s.size → s.first + i * s.interval ≤ s.population

/-- The specific systematic sample from the problem -/
def problemSample : SystematicSample :=
  { first := 3
    interval := 10
    size := 6
    population := 60 }

/-- Theorem stating that the problem's sample is valid -/
theorem problem_sample_is_valid : isValidSystematicSample problemSample := by
  sorry

/-- The sequence of numbers in the systematic sample -/
def sampleSequence (s : SystematicSample) : List ℕ :=
  List.range s.size |>.map (λ i => s.first + i * s.interval)

/-- Theorem stating that the sample sequence matches the given answer -/
theorem problem_sample_sequence_correct :
  sampleSequence problemSample = [3, 13, 23, 33, 43, 53] := by
  sorry

end NUMINAMATH_CALUDE_problem_sample_is_valid_problem_sample_sequence_correct_l1233_123356


namespace NUMINAMATH_CALUDE_samuel_has_twelve_apples_left_l1233_123320

/-- The number of apples Samuel has left after buying, eating, and making pie -/
def samuels_remaining_apples (bonnies_apples : ℕ) (samuels_extra_apples : ℕ) : ℕ :=
  let samuels_apples := bonnies_apples + samuels_extra_apples
  let after_eating := samuels_apples / 2
  let used_for_pie := after_eating / 7
  after_eating - used_for_pie

/-- Theorem stating that Samuel has 12 apples left -/
theorem samuel_has_twelve_apples_left :
  samuels_remaining_apples 8 20 = 12 := by
  sorry

#eval samuels_remaining_apples 8 20

end NUMINAMATH_CALUDE_samuel_has_twelve_apples_left_l1233_123320


namespace NUMINAMATH_CALUDE_number_problem_l1233_123357

theorem number_problem (x : ℝ) : 35 + 3 * x = 56 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1233_123357


namespace NUMINAMATH_CALUDE_condition_relation_l1233_123364

theorem condition_relation (p q : Prop) 
  (h : (p → ¬q) ∧ ¬(¬q → p)) : 
  (q → ¬p) ∧ ¬(¬p → q) := by
  sorry

end NUMINAMATH_CALUDE_condition_relation_l1233_123364


namespace NUMINAMATH_CALUDE_total_quantities_l1233_123366

theorem total_quantities (total_avg : ℝ) (subset1_count : ℕ) (subset1_avg : ℝ) (subset2_count : ℕ) (subset2_avg : ℝ) :
  total_avg = 6 →
  subset1_count = 3 →
  subset1_avg = 4 →
  subset2_count = 2 →
  subset2_avg = 33 →
  ∃ (n : ℕ), n = subset1_count + subset2_count ∧ n = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_total_quantities_l1233_123366


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l1233_123326

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the sample sizes for each grade --/
structure SampleSizes where
  grade10 : ℕ
  grade11 : ℕ

/-- Checks if the sampling is proportional across grades --/
def isProportionalSampling (dist : GradeDistribution) (sample : SampleSizes) : Prop :=
  (dist.grade10 : ℚ) / sample.grade10 = (dist.grade11 : ℚ) / sample.grade11

theorem stratified_sampling_proportion 
  (dist : GradeDistribution)
  (sample : SampleSizes)
  (h1 : dist.grade10 = 50)
  (h2 : dist.grade11 = 40)
  (h3 : dist.grade12 = 40)
  (h4 : sample.grade11 = 8)
  (h5 : isProportionalSampling dist sample) :
  sample.grade10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l1233_123326


namespace NUMINAMATH_CALUDE_cindy_marbles_l1233_123384

/-- Proves that Cindy initially had 500 marbles given the conditions -/
theorem cindy_marbles : 
  ∀ (initial_marbles : ℕ),
  (initial_marbles - 4 * 80 > 0) →
  (4 * (initial_marbles - 4 * 80) = 720) →
  initial_marbles = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l1233_123384


namespace NUMINAMATH_CALUDE_managers_salary_l1233_123396

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 600 →
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 14100 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l1233_123396


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1233_123350

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 2 ∧ y > 2 → x + y > 4) ∧
  (∃ x y : ℝ, x + y > 4 ∧ ¬(x > 2 ∧ y > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1233_123350


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1233_123324

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (6, y)
  parallel a b → y = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1233_123324


namespace NUMINAMATH_CALUDE_pencil_cartons_l1233_123307

/-- Given an order of pencils and erasers, prove the number of cartons of pencils -/
theorem pencil_cartons (pencil_cost eraser_cost total_cartons total_cost : ℕ) 
  (h1 : pencil_cost = 6)
  (h2 : eraser_cost = 3)
  (h3 : total_cartons = 100)
  (h4 : total_cost = 360) :
  ∃ (pencil_cartons eraser_cartons : ℕ),
    pencil_cartons + eraser_cartons = total_cartons ∧
    pencil_cost * pencil_cartons + eraser_cost * eraser_cartons = total_cost ∧
    pencil_cartons = 20 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cartons_l1233_123307


namespace NUMINAMATH_CALUDE_sum_equals_270_l1233_123309

/-- The sum of the arithmetic sequence with first term a, common difference d, and n terms -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

/-- The sum of two arithmetic sequences, each with 5 terms and common difference 10 -/
def two_sequence_sum (a₁ a₂ : ℕ) : ℕ := arithmetic_sum a₁ 10 5 + arithmetic_sum a₂ 10 5

theorem sum_equals_270 : two_sequence_sum 3 11 = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_270_l1233_123309


namespace NUMINAMATH_CALUDE_max_a_value_l1233_123325

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- The condition for m -/
def MCondition (m a : ℚ) : Prop := 1/2 < m ∧ m < a

/-- The main theorem -/
theorem max_a_value :
  ∃ (a : ℚ), a = 75/149 ∧
  (∀ (m : ℚ), MCondition m a →
    ∀ (x y : ℤ), 0 < x → x ≤ 150 → LatticePoint x y → ¬LineEquation m x y) ∧
  (∀ (a' : ℚ), a < a' →
    ∃ (m : ℚ), MCondition m a' ∧
    ∃ (x y : ℤ), 0 < x ∧ x ≤ 150 ∧ LatticePoint x y ∧ LineEquation m x y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1233_123325


namespace NUMINAMATH_CALUDE_identity_proof_l1233_123361

theorem identity_proof (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l1233_123361


namespace NUMINAMATH_CALUDE_function_identity_l1233_123394

theorem function_identity (f : ℝ → ℝ) 
  (h_bounded : ∃ a b : ℝ, ∃ M : ℝ, ∀ x ∈ Set.Icc a b, |f x| ≤ M)
  (h_additive : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂)
  (h_one : f 1 = 1) :
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l1233_123394
