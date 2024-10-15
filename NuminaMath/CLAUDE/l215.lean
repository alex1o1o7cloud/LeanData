import Mathlib

namespace NUMINAMATH_CALUDE_brad_siblings_product_l215_21521

/-- A family structure with a focus on two siblings -/
structure Family :=
  (total_sisters : ℕ)
  (total_brothers : ℕ)
  (sarah_sisters : ℕ)
  (sarah_brothers : ℕ)

/-- The number of sisters and brothers that Brad has -/
def brad_siblings (f : Family) : ℕ × ℕ :=
  (f.total_sisters, f.total_brothers - 1)

/-- The theorem stating the product of Brad's siblings -/
theorem brad_siblings_product (f : Family) 
  (h1 : f.sarah_sisters = 4)
  (h2 : f.sarah_brothers = 7)
  (h3 : f.total_sisters = f.sarah_sisters + 1)
  (h4 : f.total_brothers = f.sarah_brothers + 1) :
  (brad_siblings f).1 * (brad_siblings f).2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_brad_siblings_product_l215_21521


namespace NUMINAMATH_CALUDE_radians_to_degrees_l215_21551

theorem radians_to_degrees (π : ℝ) (h : π > 0) :
  (8 * π / 5) * (180 / π) = 288 := by
  sorry

end NUMINAMATH_CALUDE_radians_to_degrees_l215_21551


namespace NUMINAMATH_CALUDE_lower_profit_percentage_l215_21519

/-- Proves that given an article with a cost price of $800, if the profit at 18% is $72 more than the profit at another percentage, then that other percentage is 9%. -/
theorem lower_profit_percentage (cost_price : ℝ) (higher_percentage lower_percentage : ℝ) : 
  cost_price = 800 →
  higher_percentage = 18 →
  (higher_percentage / 100) * cost_price = (lower_percentage / 100) * cost_price + 72 →
  lower_percentage = 9 := by
  sorry

end NUMINAMATH_CALUDE_lower_profit_percentage_l215_21519


namespace NUMINAMATH_CALUDE_prob_B_given_A_value_l215_21593

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 10

/-- Represents the number of black balls initially in the box -/
def black_balls : ℕ := 8

/-- Represents the number of red balls initially in the box -/
def red_balls : ℕ := 2

/-- Represents the number of balls each player draws -/
def balls_drawn : ℕ := 2

/-- Calculates the probability of player B drawing 2 black balls given that player A has drawn 2 black balls -/
def prob_B_given_A : ℚ :=
  (Nat.choose (black_balls - balls_drawn) balls_drawn) / (Nat.choose total_balls balls_drawn)

theorem prob_B_given_A_value : prob_B_given_A = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_given_A_value_l215_21593


namespace NUMINAMATH_CALUDE_inequality_proof_l215_21581

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l215_21581


namespace NUMINAMATH_CALUDE_positive_real_inequality_l215_21597

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l215_21597


namespace NUMINAMATH_CALUDE_shortest_distance_moving_points_l215_21560

/-- The shortest distance between two points moving along perpendicular edges of a square -/
theorem shortest_distance_moving_points (side_length : ℝ) (v1 v2 : ℝ) 
  (h1 : side_length = 10)
  (h2 : v1 = 30 / 100)
  (h3 : v2 = 40 / 100) :
  ∃ t : ℝ, ∃ x y : ℝ,
    x = v1 * t ∧
    y = v2 * t ∧
    ∀ s : ℝ, (v1 * s - side_length)^2 + (v2 * s)^2 ≥ x^2 + y^2 ∧
    Real.sqrt (x^2 + y^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_moving_points_l215_21560


namespace NUMINAMATH_CALUDE_chord_length_of_perpendicular_bisector_l215_21514

/-- 
Given a circle with radius 15 units and a chord that is the perpendicular bisector of a radius,
prove that the length of this chord is 26 units.
-/
theorem chord_length_of_perpendicular_bisector (r : ℝ) (chord_length : ℝ) : 
  r = 15 → 
  chord_length = 2 * Real.sqrt (r^2 - (r/2)^2) → 
  chord_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_of_perpendicular_bisector_l215_21514


namespace NUMINAMATH_CALUDE_fifteenth_thirty_seventh_215th_digit_l215_21540

def decimal_representation (n d : ℕ) : List ℕ := sorry

def nth_digit (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem fifteenth_thirty_seventh_215th_digit :
  let rep := decimal_representation 15 37
  nth_digit 215 rep = 0 := by sorry

end NUMINAMATH_CALUDE_fifteenth_thirty_seventh_215th_digit_l215_21540


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l215_21500

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 
  let width := 2 * r
  let length := ratio * width
  width * length = 588 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l215_21500


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l215_21591

theorem power_fraction_simplification :
  (3^2023 + 3^2021) / (3^2023 - 3^2021) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l215_21591


namespace NUMINAMATH_CALUDE_age_ratio_dan_james_l215_21510

theorem age_ratio_dan_james : 
  ∀ (dan_future_age james_age : ℕ),
    dan_future_age = 28 →
    james_age = 20 →
    ∃ (dan_age : ℕ),
      dan_age + 4 = dan_future_age ∧
      dan_age * 5 = james_age * 6 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_dan_james_l215_21510


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_three_a_greater_than_four_when_A_subset_B_l215_21588

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- Theorem for part 1
theorem intersection_complement_when_a_three :
  A ∩ (Set.univ \ B 3) = Set.Icc 3 4 := by sorry

-- Theorem for part 2
theorem a_greater_than_four_when_A_subset_B (a : ℝ) :
  A ⊆ B a → a > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_three_a_greater_than_four_when_A_subset_B_l215_21588


namespace NUMINAMATH_CALUDE_system_implication_l215_21502

variable {X : Type*} [LinearOrder X]
variable (f g : X → ℝ)

theorem system_implication :
  (∀ x, f x > 0 ∧ g x > 0) →
  (∀ x, f x > 0 ∧ f x + g x > 0) :=
by sorry

example : ∃ f g : ℝ → ℝ, 
  (∃ x, f x > 0 ∧ f x + g x > 0) ∧
  ¬(∀ x, f x > 0 ∧ g x > 0) :=
by sorry

end NUMINAMATH_CALUDE_system_implication_l215_21502


namespace NUMINAMATH_CALUDE_skipping_odometer_theorem_l215_21582

/-- Represents an odometer that skips the digit 6 -/
def SkippingOdometer : Type := ℕ

/-- Converts a regular odometer reading to a skipping odometer reading -/
def toSkippingReading (n : ℕ) : SkippingOdometer :=
  sorry

/-- Converts a skipping odometer reading back to the actual distance -/
def toActualDistance (s : SkippingOdometer) : ℕ :=
  sorry

theorem skipping_odometer_theorem :
  toActualDistance (toSkippingReading 1464) = 2005 :=
sorry

end NUMINAMATH_CALUDE_skipping_odometer_theorem_l215_21582


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l215_21538

theorem solve_system_of_equations (a b m : ℤ) 
  (eq1 : a - b = 6)
  (eq2 : 2 * a + b = m)
  (opposite : a + b = 0) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l215_21538


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l215_21541

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 61 ways to put 5 distinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : ballsInBoxes 5 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l215_21541


namespace NUMINAMATH_CALUDE_katie_game_difference_l215_21528

theorem katie_game_difference : 
  ∀ (katie_new_games katie_old_games friends_new_games : ℕ),
  katie_new_games = 57 →
  katie_old_games = 39 →
  friends_new_games = 34 →
  katie_new_games + katie_old_games - friends_new_games = 62 := by
sorry

end NUMINAMATH_CALUDE_katie_game_difference_l215_21528


namespace NUMINAMATH_CALUDE_cosine_symmetry_and_monotonicity_l215_21507

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem cosine_symmetry_and_monotonicity (ω : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω x = f ω (3 * Real.pi / 2 - x)) →
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → f ω x ≥ f ω y) →
  ω = 2/3 ∨ ω = 2 := by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_and_monotonicity_l215_21507


namespace NUMINAMATH_CALUDE_intersection_M_N_l215_21517

def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l215_21517


namespace NUMINAMATH_CALUDE_path_count_theorem_l215_21509

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the number of paths in a grid with the given constraints -/
def count_paths (g : Grid) : ℕ :=
  Nat.choose (g.width + g.height - 1) g.height -
  Nat.choose (g.width + g.height - 2) g.height +
  Nat.choose (g.width + g.height - 3) g.height

/-- The problem statement -/
theorem path_count_theorem (g : Grid) (h1 : g.width = 7) (h2 : g.height = 6) :
  count_paths g = 1254 := by
  sorry

end NUMINAMATH_CALUDE_path_count_theorem_l215_21509


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l215_21548

theorem ice_cream_consumption (friday_amount saturday_amount : Real) 
  (h1 : friday_amount = 3.25) 
  (h2 : saturday_amount = 0.25) : 
  friday_amount + saturday_amount = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l215_21548


namespace NUMINAMATH_CALUDE_jamies_mothers_age_twice_l215_21511

/-- 
Given:
- Jamie's age in 2010 is 10 years
- Jamie's mother's age in 2010 is 5 times Jamie's age
Prove that the year when Jamie's mother's age will be twice Jamie's age is 2040
-/
theorem jamies_mothers_age_twice (jamie_age_2010 : ℕ) (mother_age_multiplier : ℕ) : 
  jamie_age_2010 = 10 →
  mother_age_multiplier = 5 →
  ∃ (years_passed : ℕ),
    (jamie_age_2010 + years_passed) * 2 = (jamie_age_2010 * mother_age_multiplier + years_passed) ∧
    2010 + years_passed = 2040 := by
  sorry

#check jamies_mothers_age_twice

end NUMINAMATH_CALUDE_jamies_mothers_age_twice_l215_21511


namespace NUMINAMATH_CALUDE_marathon_end_time_l215_21532

-- Define the start time of the marathon
def start_time : Nat := 15  -- 3:00 p.m. in 24-hour format

-- Define the duration of the marathon in minutes
def duration : Nat := 780

-- Define a function to calculate the end time
def calculate_end_time (start : Nat) (duration_minutes : Nat) : Nat :=
  (start + duration_minutes / 60) % 24

-- Theorem to prove
theorem marathon_end_time :
  calculate_end_time start_time duration = 4 := by
  sorry


end NUMINAMATH_CALUDE_marathon_end_time_l215_21532


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l215_21564

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l215_21564


namespace NUMINAMATH_CALUDE_path_area_and_cost_calculation_l215_21550

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost_calculation 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_unit : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.8)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 759.36 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1518.72 := by
  sorry

#eval path_area 75 55 2.8
#eval construction_cost (path_area 75 55 2.8) 2

end NUMINAMATH_CALUDE_path_area_and_cost_calculation_l215_21550


namespace NUMINAMATH_CALUDE_climb_10_stairs_l215_21590

/-- Function representing the number of ways to climb n stairs -/
def climb_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | n + 4 => climb_ways (n + 3) + climb_ways (n + 2) + climb_ways n

/-- Theorem stating that there are 151 ways to climb 10 stairs -/
theorem climb_10_stairs : climb_ways 10 = 151 := by
  sorry

end NUMINAMATH_CALUDE_climb_10_stairs_l215_21590


namespace NUMINAMATH_CALUDE_trailingZeros_100_factorial_l215_21585

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailingZeros_100_factorial :
  trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeros_100_factorial_l215_21585


namespace NUMINAMATH_CALUDE_total_flowers_planted_l215_21508

theorem total_flowers_planted (num_people : ℕ) (num_days : ℕ) (flowers_per_day : ℕ) : 
  num_people = 5 → num_days = 2 → flowers_per_day = 20 → 
  num_people * num_days * flowers_per_day = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_planted_l215_21508


namespace NUMINAMATH_CALUDE_total_marks_calculation_l215_21589

theorem total_marks_calculation (obtained_marks : ℝ) (percentage : ℝ) (total_marks : ℝ) : 
  obtained_marks = 450 → percentage = 90 → obtained_marks = (percentage / 100) * total_marks → 
  total_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l215_21589


namespace NUMINAMATH_CALUDE_binomial_12_6_l215_21559

theorem binomial_12_6 : Nat.choose 12 6 = 1848 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_6_l215_21559


namespace NUMINAMATH_CALUDE_class_composition_l215_21522

theorem class_composition (boys_score girls_score class_average : ℝ) 
  (boys_score_val : boys_score = 80)
  (girls_score_val : girls_score = 90)
  (class_average_val : class_average = 86) :
  let boys_percentage : ℝ := 40
  let girls_percentage : ℝ := 100 - boys_percentage
  class_average = (boys_percentage * boys_score + girls_percentage * girls_score) / 100 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l215_21522


namespace NUMINAMATH_CALUDE_vector_collinearity_l215_21576

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- The problem statement -/
theorem vector_collinearity (m : ℝ) :
  collinear (m + 3, 2) (m, 1) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l215_21576


namespace NUMINAMATH_CALUDE_birds_on_fence_l215_21501

/-- Given an initial number of birds and a final total number of birds,
    calculate the number of birds that joined. -/
def birds_joined (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 2 initial birds and 6 final birds,
    the number of birds that joined is 4. -/
theorem birds_on_fence : birds_joined 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l215_21501


namespace NUMINAMATH_CALUDE_hemisphere_base_area_l215_21594

/-- If the surface area of a hemisphere is 9, then the area of its base is 3 -/
theorem hemisphere_base_area (r : ℝ) (h : 3 * Real.pi * r^2 = 9) : Real.pi * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_base_area_l215_21594


namespace NUMINAMATH_CALUDE_square_ratio_proof_l215_21512

theorem square_ratio_proof : 
  ∀ (s₁ s₂ : ℝ), s₁ > 0 ∧ s₂ > 0 →
  (s₁^2 / s₂^2 = 45 / 64) →
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (s₁ / s₂ = (a : ℝ) * Real.sqrt b / c) ∧
  (a + b + c = 16) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l215_21512


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l215_21555

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 34*x^2 + 225 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 34*y^2 + 225 = 0 → x ≤ y) ∧
  x = -5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l215_21555


namespace NUMINAMATH_CALUDE_trailingZerosOfSquareMinusFactorial_l215_21569

-- Define the number we're working with
def n : ℕ := 999999

-- Define the factorial function
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

-- Define a function to count trailing zeros
def countTrailingZeros (x : ℕ) : ℕ :=
  if x = 0 then 0
  else if x % 10 = 0 then 1 + countTrailingZeros (x / 10)
  else 0

-- Theorem statement
theorem trailingZerosOfSquareMinusFactorial :
  countTrailingZeros (n^2 - factorial 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trailingZerosOfSquareMinusFactorial_l215_21569


namespace NUMINAMATH_CALUDE_golden_ratio_bounds_l215_21535

theorem golden_ratio_bounds : ∃ x : ℝ, x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_bounds_l215_21535


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l215_21568

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l215_21568


namespace NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l215_21537

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point where a line intersects the x-axis -/
def XAxisIntersection : ℝ → ℝ × ℝ := λ x ↦ (x, 0)

/-- Theorem: Tangent line intersection for two specific circles -/
theorem tangent_intersection_x_coordinate :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (12, 0), radius := 5 }
  ∃ (x : ℝ), x > 0 ∧ 
    ∃ (l : Set (ℝ × ℝ)), 
      (XAxisIntersection x ∈ l) ∧ 
      (∃ p1 ∈ l, (p1.1 - c1.center.1)^2 + (p1.2 - c1.center.2)^2 = c1.radius^2) ∧
      (∃ p2 ∈ l, (p2.1 - c2.center.1)^2 + (p2.2 - c2.center.2)^2 = c2.radius^2) ∧
      x = 9/2 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l215_21537


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l215_21533

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) :
  t1.isIsosceles ∧
  t1.a = 30 ∧ t1.b = 30 ∧ t1.c = 15 ∧
  t2.isSimilar t1 ∧
  min t2.a (min t2.b t2.c) = 45 →
  t2.perimeter = 225 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l215_21533


namespace NUMINAMATH_CALUDE_distance_to_directrix_l215_21542

/-- The distance from a point on a parabola to its directrix -/
theorem distance_to_directrix (p : ℝ) (x y : ℝ) (h : y^2 = 2*p*x) :
  x + p/2 = 9/4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l215_21542


namespace NUMINAMATH_CALUDE_vidyas_age_l215_21567

theorem vidyas_age (vidya_age : ℕ) (mother_age : ℕ) : 
  mother_age = 3 * vidya_age + 5 →
  mother_age = 44 →
  vidya_age = 13 := by
sorry

end NUMINAMATH_CALUDE_vidyas_age_l215_21567


namespace NUMINAMATH_CALUDE_product_equals_one_l215_21505

theorem product_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 4)
  (eq2 : y + 1/z = 1)
  (eq3 : z + 1/x = 7/3) :
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l215_21505


namespace NUMINAMATH_CALUDE_max_b_minus_a_l215_21562

theorem max_b_minus_a (a b : ℝ) : 
  a < 0 → 
  (∀ x : ℝ, (x^2 + 2017*a)*(x + 2016*b) ≥ 0) → 
  b - a ≤ 2017 :=
by sorry

end NUMINAMATH_CALUDE_max_b_minus_a_l215_21562


namespace NUMINAMATH_CALUDE_race_head_start_l215_21504

/-- Given two runners A and B, where A's speed is 21/19 times B's speed,
    the head start fraction that A should give B for a dead heat is 2/21 of the race length. -/
theorem race_head_start (speed_a speed_b length head_start : ℝ) :
  speed_a = (21 / 19) * speed_b →
  length > 0 →
  head_start > 0 →
  length / speed_a = (length - head_start) / speed_b →
  head_start / length = 2 / 21 := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l215_21504


namespace NUMINAMATH_CALUDE_vector_angle_in_circle_l215_21574

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the theorem
theorem vector_angle_in_circle (O A B C : ℝ × ℝ) (r : ℝ) :
  A ∈ Circle O r →
  B ∈ Circle O r →
  C ∈ Circle O r →
  (A.1 - O.1, A.2 - O.2) = (1/2) * ((B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2)) →
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_angle_in_circle_l215_21574


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l215_21518

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (d : ℚ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  ∃ (q : ℚ), q = 1/2 ∧ ∀ (n : ℕ), a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l215_21518


namespace NUMINAMATH_CALUDE_book_weight_l215_21572

theorem book_weight (total_weight : ℝ) (num_books : ℕ) (h1 : total_weight = 42) (h2 : num_books = 14) :
  total_weight / num_books = 3 := by
sorry

end NUMINAMATH_CALUDE_book_weight_l215_21572


namespace NUMINAMATH_CALUDE_bell_rings_theorem_l215_21558

/-- Represents the number of times a bell rings for a single class -/
def bell_rings_per_class : ℕ := 2

/-- Represents the total number of classes in a day -/
def total_classes : ℕ := 5

/-- Represents the current class number (1-indexed) -/
def current_class : ℕ := 5

/-- Calculates the total number of bell rings up to and including the current class -/
def total_bell_rings (completed_classes : ℕ) (current_class : ℕ) : ℕ :=
  completed_classes * bell_rings_per_class + 1

/-- Theorem: Given 5 classes where the bell rings twice for each completed class 
    and once for the current class (Music), the total number of bell rings is 9 -/
theorem bell_rings_theorem : 
  total_bell_rings (current_class - 1) current_class = 9 := by
  sorry

end NUMINAMATH_CALUDE_bell_rings_theorem_l215_21558


namespace NUMINAMATH_CALUDE_positive_difference_of_roots_l215_21515

theorem positive_difference_of_roots : ∃ (r₁ r₂ : ℝ),
  (r₁^2 - 5*r₁ - 26) / (r₁ + 5) = 3*r₁ + 8 ∧
  (r₂^2 - 5*r₂ - 26) / (r₂ + 5) = 3*r₂ + 8 ∧
  r₁ ≠ r₂ ∧
  |r₁ - r₂| = 8 :=
by sorry

end NUMINAMATH_CALUDE_positive_difference_of_roots_l215_21515


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l215_21598

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : ∀ x, f a b c (x + 1) - f a b c x = 2 * x)
  (h2 : f a b c 0 = 1) :
  (∀ x, f a b c x = x^2 - x + 1) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b c x > 2 * x + m) ↔ m ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l215_21598


namespace NUMINAMATH_CALUDE_contrapositive_isosceles_equal_angles_l215_21543

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties of a triangle
def Triangle.isIsosceles (t : Triangle) : Prop := sorry
def Triangle.hasEqualInteriorAngles (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_equal_angles (t : Triangle) :
  (¬(t.isIsosceles) → ¬(t.hasEqualInteriorAngles)) ↔
  (t.hasEqualInteriorAngles → t.isIsosceles) := by sorry

end NUMINAMATH_CALUDE_contrapositive_isosceles_equal_angles_l215_21543


namespace NUMINAMATH_CALUDE_laura_age_l215_21536

theorem laura_age :
  ∃ (L : ℕ), 
    L > 0 ∧
    L < 100 ∧
    (L - 1) % 8 = 0 ∧
    (L + 1) % 7 = 0 ∧
    (∃ (A : ℕ), 
      A > L ∧
      A < 100 ∧
      (A - 1) % 8 = 0 ∧
      (A + 1) % 7 = 0) →
    L = 41 := by
  sorry

end NUMINAMATH_CALUDE_laura_age_l215_21536


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l215_21520

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_perfect_square (n + reverse_digits n)

theorem two_digit_reverse_sum_square :
  {n : ℕ | satisfies_condition n} = {29, 38, 47, 56, 65, 74, 83, 92} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l215_21520


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l215_21561

/-- The cost of one blue pill in dollars -/
def blue_pill_cost : ℚ := 11

/-- The number of days in the treatment period -/
def days : ℕ := 21

/-- The daily discount in dollars after the first week -/
def daily_discount : ℚ := 2

/-- The number of days with discount -/
def discount_days : ℕ := 14

/-- The total cost without discount for the entire period -/
def total_cost_without_discount : ℚ := 735

/-- The number of blue pills taken daily -/
def daily_blue_pills : ℕ := 2

/-- The number of orange pills taken daily -/
def daily_orange_pills : ℕ := 1

/-- The cost difference between orange and blue pills in dollars -/
def orange_blue_cost_difference : ℚ := 2

theorem blue_pill_cost_proof :
  blue_pill_cost * (daily_blue_pills * days + daily_orange_pills * days) +
  orange_blue_cost_difference * (daily_orange_pills * days) -
  daily_discount * discount_days = total_cost_without_discount - daily_discount * discount_days :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l215_21561


namespace NUMINAMATH_CALUDE_consecutive_cube_diff_square_l215_21539

theorem consecutive_cube_diff_square (x : ℤ) :
  ∃ y : ℤ, (x + 1)^3 - x^3 = y^2 →
  ∃ a b : ℤ, y = a^2 + b^2 ∧ b = a + 1 := by
sorry

end NUMINAMATH_CALUDE_consecutive_cube_diff_square_l215_21539


namespace NUMINAMATH_CALUDE_parallelogram_properties_l215_21547

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  opposite : v1 = v2

/-- The fourth vertex and diagonal intersection of a specific parallelogram -/
theorem parallelogram_properties (p : Parallelogram) 
  (h1 : p.v1 = (2, -3))
  (h2 : p.v2 = (8, 5))
  (h3 : p.v3 = (5, 0)) :
  p.v4 = (5, 2) ∧ 
  (((p.v1.1 + p.v2.1) / 2, (p.v1.2 + p.v2.2) / 2) : ℝ × ℝ) = (5, 1) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l215_21547


namespace NUMINAMATH_CALUDE_money_distribution_l215_21575

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 400)
  (ac_sum : a + c = 300)
  (bc_sum : b + c = 150) :
  c = 50 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l215_21575


namespace NUMINAMATH_CALUDE_equation_solution_l215_21530

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ 9 - 3 / (1 / x) + 3 = 3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l215_21530


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l215_21527

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 16 →
  num_slices = 16 →
  (π * (diameter/2)^2 * thickness) / num_slices = 2 * π := by
  sorry

#check pizza_slice_volume

end NUMINAMATH_CALUDE_pizza_slice_volume_l215_21527


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l215_21579

def num_boys : ℕ := 3
def num_girls : ℕ := 2

def arrange_chess_team (boys : ℕ) (girls : ℕ) : ℕ :=
  (girls.factorial) * (boys.factorial)

theorem chess_team_arrangements :
  arrange_chess_team num_boys num_girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l215_21579


namespace NUMINAMATH_CALUDE_square_root_sum_l215_21531

theorem square_root_sum (x : ℝ) :
  (Real.sqrt (64 - x^2) - Real.sqrt (16 - x^2) = 4) →
  (Real.sqrt (64 - x^2) + Real.sqrt (16 - x^2) = 12) :=
by sorry

end NUMINAMATH_CALUDE_square_root_sum_l215_21531


namespace NUMINAMATH_CALUDE_area_of_ring_l215_21544

/-- The area of a ring formed between two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 7) :
  π * r₁^2 - π * r₂^2 = 95 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_ring_l215_21544


namespace NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l215_21596

/-- Represents a cube made of unit cubes --/
structure Cube where
  size : ℕ

/-- Calculates the number of visible unit cubes from a corner of the cube --/
def visibleUnitCubes (c : Cube) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem: The maximum number of visible unit cubes from a single point in a 9x9x9 cube is 220 --/
theorem max_visible_cubes_9x9x9 :
  ∀ (c : Cube), c.size = 9 → visibleUnitCubes c = 220 := by
  sorry

#eval visibleUnitCubes { size := 9 }

end NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l215_21596


namespace NUMINAMATH_CALUDE_agnes_twice_jane_age_l215_21580

/-- The number of years until Agnes is twice as old as Jane -/
def years_until_double_age (agnes_age : ℕ) (jane_age : ℕ) : ℕ :=
  (agnes_age - 2 * jane_age) / (2 - 1)

/-- Theorem stating that it will take 13 years for Agnes to be twice as old as Jane -/
theorem agnes_twice_jane_age (agnes_current_age jane_current_age : ℕ) 
  (h1 : agnes_current_age = 25) 
  (h2 : jane_current_age = 6) : 
  years_until_double_age agnes_current_age jane_current_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_agnes_twice_jane_age_l215_21580


namespace NUMINAMATH_CALUDE_problem_solution_l215_21545

theorem problem_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : 8 * x^2 + 16 * x * y = x^3 + 3 * x^2 * y) (h4 : y = 2 * x) :
  x = 40 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l215_21545


namespace NUMINAMATH_CALUDE_problem_solution_l215_21552

theorem problem_solution :
  ∀ (A B C : ℝ) (a n b c : ℕ) (d : ℕ+),
    A^2 + B^2 + C^2 = 3 →
    A * B + B * C + C * A = 3 →
    a = A^2 →
    29 * n + 42 * b = a →
    5 < b →
    b < 10 →
    (Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) = 
      (c * Real.sqrt 21 - 18 * Real.sqrt 15 - 2 * Real.sqrt 35 + b) / 59 →
    d = (Nat.factors c).length →
    a = 1 ∧ b = 9 ∧ c = 20 ∧ d = 6 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l215_21552


namespace NUMINAMATH_CALUDE_egg_difference_l215_21586

/-- Given that Megan bought 2 dozen eggs, 3 eggs broke, and twice as many cracked,
    prove that the difference between eggs in perfect condition and cracked eggs is 9. -/
theorem egg_difference (total : ℕ) (broken : ℕ) (cracked : ℕ) :
  total = 2 * 12 →
  broken = 3 →
  cracked = 2 * broken →
  total - (broken + cracked) - cracked = 9 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_l215_21586


namespace NUMINAMATH_CALUDE_digital_earth_function_l215_21526

/-- Represents the functions of a system --/
inductive SystemFunction
| InformationProcessing
| GeographicInformationManagement
| InformationIntegrationAndDisplay
| SpatialPositioning

/-- Represents different systems --/
inductive System
| DigitalEarth
| RemoteSensing
| GeographicInformationSystem
| GlobalPositioningSystem

/-- Defines the function of a given system --/
def system_function : System → SystemFunction
| System.DigitalEarth => SystemFunction.InformationIntegrationAndDisplay
| System.RemoteSensing => SystemFunction.InformationProcessing
| System.GeographicInformationSystem => SystemFunction.GeographicInformationManagement
| System.GlobalPositioningSystem => SystemFunction.SpatialPositioning

/-- Theorem: The function of Digital Earth is Information Integration and Display --/
theorem digital_earth_function :
  system_function System.DigitalEarth = SystemFunction.InformationIntegrationAndDisplay := by
  sorry

end NUMINAMATH_CALUDE_digital_earth_function_l215_21526


namespace NUMINAMATH_CALUDE_equality_check_l215_21570

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  (-(3 * 2)^2 ≠ -3 * 2^2) ∧ 
  (-|2^3| ≠ |-2^3|) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l215_21570


namespace NUMINAMATH_CALUDE_platform_length_calculation_l215_21563

/-- Calculates the length of a platform given train specifications and crossing time -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmph = 72 →
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length = 380 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l215_21563


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l215_21503

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0,
    if one of its asymptotes is y = 3x, then b = 3. -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 - y^2/b^2 = 1) → 
  (∃ x y : ℝ, y = 3*x ∧ x^2 - y^2/b^2 = 1) → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l215_21503


namespace NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l215_21549

def total_items : ℕ := 10
def genuine_items : ℕ := 6
def defective_items : ℕ := 4

theorem probability_second_genuine_given_first_genuine :
  let first_genuine : ℝ := genuine_items / total_items
  let second_genuine : ℝ := (genuine_items - 1) / (total_items - 1)
  let both_genuine : ℝ := first_genuine * second_genuine
  both_genuine / first_genuine = 5 / 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l215_21549


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l215_21534

/-- A quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (parallel : (D.1 - C.1) * (B.2 - A.2) = (D.2 - C.2) * (B.1 - A.1))
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 7)
  (DC_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 6)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 10)

/-- The perimeter of the quadrilateral ABCD is 35.2 cm -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2) +
  Real.sqrt ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2) +
  Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2) +
  Real.sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2) = 35.2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l215_21534


namespace NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l215_21553

/-- Represents the number of free ends after k iterations of extending segments -/
def num_free_ends (k : ℕ) : ℕ := 1 + 4 * k

/-- Theorem stating that there exists a number of iterations that results in 1001 free ends -/
theorem exists_k_for_1001_free_ends : ∃ k : ℕ, num_free_ends k = 1001 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l215_21553


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l215_21513

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 2) / (x - 1) > 0} = {x : ℝ | x > 1 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l215_21513


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l215_21583

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {1, 3, 4, 6}

theorem complement_intersection_M_N : (M ∩ N)ᶜ = {2, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l215_21583


namespace NUMINAMATH_CALUDE_office_age_problem_l215_21577

theorem office_age_problem (total_persons : Nat) (group1_persons : Nat) (group2_persons : Nat)
  (total_avg_age : ℝ) (group1_avg_age : ℝ) (group2_avg_age : ℝ)
  (h1 : total_persons = 16)
  (h2 : group1_persons = 5)
  (h3 : group2_persons = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16)
  (h7 : group1_persons + group2_persons + 2 = total_persons) :
  ∃ (person15_age : ℝ),
    person15_age = total_persons * total_avg_age -
      (group1_persons * group1_avg_age + group2_persons * group2_avg_age) ∧
    person15_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_office_age_problem_l215_21577


namespace NUMINAMATH_CALUDE_gumball_problem_solution_l215_21573

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- 
Given a gumball machine with the specified number of gumballs for each color,
this function returns the minimum number of gumballs one must buy to guarantee
getting four of the same color.
-/
def minGumballsToBuy (machine : GumballMachine) : Nat :=
  sorry

/-- The theorem stating the correct answer for the given problem -/
theorem gumball_problem_solution :
  let machine : GumballMachine := { red := 10, white := 6, blue := 8, green := 9 }
  minGumballsToBuy machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_gumball_problem_solution_l215_21573


namespace NUMINAMATH_CALUDE_root_difference_zero_l215_21529

theorem root_difference_zero : ∃ (r : ℝ), 
  (∀ x : ℝ, x^2 + 20*x + 75 = -25 ↔ x = r) ∧ 
  (abs (r - r) = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_difference_zero_l215_21529


namespace NUMINAMATH_CALUDE_train_length_calculation_l215_21565

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (time_to_cross : ℝ) : 
  train_speed = 63 →
  man_speed = 3 →
  time_to_cross = 29.997600191984642 →
  (train_speed - man_speed) * time_to_cross * (1000 / 3600) = 500 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l215_21565


namespace NUMINAMATH_CALUDE_intersection_complement_equal_set_l215_21571

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_equal_set : M ∩ (Set.univ \ N) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_set_l215_21571


namespace NUMINAMATH_CALUDE_production_normality_l215_21516

-- Define the parameters of the normal distribution
def μ : ℝ := 8.0
def σ : ℝ := 0.15

-- Define the 3-sigma range
def lower_bound : ℝ := μ - 3 * σ
def upper_bound : ℝ := μ + 3 * σ

-- Define the observed diameters
def morning_diameter : ℝ := 7.9
def afternoon_diameter : ℝ := 7.5

-- Define what it means for a production to be normal
def is_normal (x : ℝ) : Prop := lower_bound ≤ x ∧ x ≤ upper_bound

-- Theorem statement
theorem production_normality :
  is_normal morning_diameter ∧ ¬is_normal afternoon_diameter :=
sorry

end NUMINAMATH_CALUDE_production_normality_l215_21516


namespace NUMINAMATH_CALUDE_constant_variance_properties_l215_21524

/-- A sequence is constant variance if the sequence of its squares is arithmetic -/
def ConstantVariance (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2)^2 - a (n + 1)^2 = a (n + 1)^2 - a n^2

/-- A sequence is constant if all its terms are equal -/
def ConstantSequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_variance_properties (a : ℕ → ℝ) :
  (ConstantSequence a → ConstantVariance a) ∧
  (ConstantVariance a → ArithmeticSequence (λ n => (a n)^2)) ∧
  (ConstantVariance a → ConstantVariance (λ n => a (2*n))) :=
sorry

end NUMINAMATH_CALUDE_constant_variance_properties_l215_21524


namespace NUMINAMATH_CALUDE_consecutive_color_draws_probability_l215_21525

def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5
def total_chips : ℕ := blue_chips + green_chips + red_chips

def probability_consecutive_color_draws : ℚ :=
  (Nat.factorial 3 * Nat.factorial blue_chips * Nat.factorial green_chips * Nat.factorial red_chips) /
  Nat.factorial total_chips

theorem consecutive_color_draws_probability :
  probability_consecutive_color_draws = 1 / 4620 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_color_draws_probability_l215_21525


namespace NUMINAMATH_CALUDE_bike_sharing_growth_specific_bike_sharing_case_l215_21592

/-- Represents the growth of shared bicycles over three months -/
theorem bike_sharing_growth 
  (initial_bikes : ℕ) 
  (planned_increase : ℕ) 
  (growth_rate : ℝ) : 
  initial_bikes * (1 + growth_rate)^2 = initial_bikes + planned_increase :=
by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_bike_sharing_case (x : ℝ) : 
  1000 * (1 + x)^2 = 1000 + 440 :=
by
  sorry

end NUMINAMATH_CALUDE_bike_sharing_growth_specific_bike_sharing_case_l215_21592


namespace NUMINAMATH_CALUDE_min_abs_w_l215_21599

theorem min_abs_w (w : ℂ) (h : Complex.abs (w - 2*I) + Complex.abs (w + 3) = 6) :
  ∃ (min_abs : ℝ), min_abs = 1 ∧ ∀ (z : ℂ), Complex.abs (w - 2*I) + Complex.abs (w + 3) = 6 → Complex.abs z ≥ min_abs :=
sorry

end NUMINAMATH_CALUDE_min_abs_w_l215_21599


namespace NUMINAMATH_CALUDE_two_objects_ten_recipients_l215_21554

/-- The number of ways to distribute two distinct objects among a given number of recipients. -/
def distributionWays (recipients : ℕ) : ℕ := recipients * recipients

/-- Theorem: The number of ways to distribute two distinct objects among ten recipients is 100. -/
theorem two_objects_ten_recipients :
  distributionWays 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_two_objects_ten_recipients_l215_21554


namespace NUMINAMATH_CALUDE_range_of_x_l215_21523

theorem range_of_x (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a ≠ 0 → b ≠ 0 → |2*a - b| + |a + b| ≥ |a| * (|x - 1| + |x + 1|)) →
  x ∈ Set.Icc (-3/2) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l215_21523


namespace NUMINAMATH_CALUDE_marks_remaining_money_l215_21587

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount : ℕ) (num_books : ℕ) (price_per_book : ℕ) : ℕ :=
  initial_amount - (num_books * price_per_book)

/-- Proves that Mark is left with $35 after his purchase --/
theorem marks_remaining_money :
  remaining_money 85 10 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l215_21587


namespace NUMINAMATH_CALUDE_new_person_weight_l215_21546

theorem new_person_weight (initial_count : ℕ) (initial_weight : ℝ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  weight_increase = 6 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l215_21546


namespace NUMINAMATH_CALUDE_inequality_proof_l215_21506

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) : 
  (π/2) + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l215_21506


namespace NUMINAMATH_CALUDE_triangle_side_values_l215_21584

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+,
    (triangle_exists 8 11 (y.val ^ 2 - 1) ↔ y.val = 3 ∨ y.val = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l215_21584


namespace NUMINAMATH_CALUDE_set_intersection_equality_l215_21595

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | ∃ k : ℤ, x = k}

theorem set_intersection_equality :
  (Aᶜ ∩ B ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 0}) = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l215_21595


namespace NUMINAMATH_CALUDE_ferry_speed_difference_l215_21578

/-- Represents the speed and time of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a ferry -/
def distance (journey : FerryJourney) : ℝ :=
  journey.speed * journey.time

theorem ferry_speed_difference :
  let ferryP : FerryJourney := { speed := 8, time := 3 }
  let ferryQ : FerryJourney := { speed := (3 * distance ferryP) / (ferryP.time + 5), time := ferryP.time + 5 }
  ferryQ.speed - ferryP.speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l215_21578


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l215_21556

/-- Given a geometric sequence {a_n} with a₃ = 6 and S₃ = 18,
    prove that the common ratio q is either 1 or -1/2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = 6)
  (h_S3 : a 1 + a 2 + a 3 = 18) :
  q = 1 ∨ q = -1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l215_21556


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_l215_21566

-- Define the characteristics of the data
structure DataCharacteristics where
  partsOfWhole : Bool
  categorical : Bool
  compareProportions : Bool

-- Define the types of statistical graphs
inductive StatisticalGraph
  | PieChart
  | BarGraph
  | LineGraph
  | Histogram

-- Define the suitability of a graph for given data characteristics
def isSuitable (graph : StatisticalGraph) (data : DataCharacteristics) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => data.partsOfWhole ∧ data.categorical ∧ data.compareProportions
  | _ => False

-- Theorem statement
theorem pie_chart_most_suitable (data : DataCharacteristics) 
  (h1 : data.partsOfWhole = true) 
  (h2 : data.categorical = true) 
  (h3 : data.compareProportions = true) :
  ∀ (graph : StatisticalGraph), 
    isSuitable graph data → graph = StatisticalGraph.PieChart := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_l215_21566


namespace NUMINAMATH_CALUDE_P_superset_Q_l215_21557

-- Define the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Theorem statement
theorem P_superset_Q : Q ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_P_superset_Q_l215_21557
