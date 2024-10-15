import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_expression_l2003_200365

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) :
  ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 3 ∧ 2*x*y + 3*z = 21/4 ∧ ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 3 → 2*a*b + 3*c ≤ 21/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2003_200365


namespace NUMINAMATH_CALUDE_initial_number_proof_l2003_200397

theorem initial_number_proof (x : ℝ) : 
  x + 3889 - 47.80600000000004 = 3854.002 → x = 12.808000000000158 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2003_200397


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2003_200376

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The set of all possible outcomes when drawing two balls from the bag -/
def allOutcomes : Finset Outcome := sorry

/-- The event "Exactly one black ball" -/
def exactlyOneBlack (outcome : Outcome) : Prop :=
  (outcome.first = Color.Black ∧ outcome.second = Color.Red) ∨
  (outcome.first = Color.Red ∧ outcome.second = Color.Black)

/-- The event "Exactly two black balls" -/
def exactlyTwoBlack (outcome : Outcome) : Prop :=
  outcome.first = Color.Black ∧ outcome.second = Color.Black

theorem mutually_exclusive_not_contradictory :
  (∀ o : Outcome, ¬(exactlyOneBlack o ∧ exactlyTwoBlack o)) ∧
  (∃ o : Outcome, ¬exactlyOneBlack o ∧ ¬exactlyTwoBlack o) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2003_200376


namespace NUMINAMATH_CALUDE_sum_yz_zero_percent_of_x_l2003_200357

theorem sum_yz_zero_percent_of_x (x y z : ℚ) 
  (h1 : (3/5) * (x - y) = (3/10) * (x + y))
  (h2 : (2/5) * (x + z) = (1/5) * (y + z))
  (h3 : (1/2) * (x - z) = (1/4) * (x + y + z)) :
  y + z = 0 * x :=
by sorry

end NUMINAMATH_CALUDE_sum_yz_zero_percent_of_x_l2003_200357


namespace NUMINAMATH_CALUDE_a_6_equals_11_l2003_200337

/-- Given a sequence {aₙ} where Sₙ is the sum of its first n terms -/
def S (n : ℕ) : ℕ := n^2 + 1

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Proof that the 6th term of the sequence is 11 -/
theorem a_6_equals_11 : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_a_6_equals_11_l2003_200337


namespace NUMINAMATH_CALUDE_range_of_m_l2003_200333

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2*x - 3 > 0) → (x < m - 1 ∨ x > m + 1)) ∧ 
  (∃ x : ℝ, (x < m - 1 ∨ x > m + 1) ∧ ¬(x^2 - 2*x - 3 > 0)) →
  0 ≤ m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2003_200333


namespace NUMINAMATH_CALUDE_simplify_expression_l2003_200303

theorem simplify_expression (a b : ℝ) :
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2003_200303


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2003_200327

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/hr -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) :
  train_length = 320 →
  crossing_time = 7.999360051195905 →
  ∃ (speed_kmh : Real), 
    abs (speed_kmh - (train_length / crossing_time * 3.6)) < 0.001 ∧ 
    abs (speed_kmh - 144.018) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2003_200327


namespace NUMINAMATH_CALUDE_p_adic_valuation_properties_l2003_200341

/-- The p-adic valuation of an integer -/
noncomputable def v_p (p : ℕ) (n : ℤ) : ℕ := sorry

/-- Properties of p-adic valuation for prime p and integers m, n -/
theorem p_adic_valuation_properties (p : ℕ) (m n : ℤ) (hp : Nat.Prime p) :
  (v_p p (m * n) = v_p p m + v_p p n) ∧
  (v_p p (m + n) ≥ min (v_p p m) (v_p p n)) ∧
  (v_p p (Int.gcd m n) = min (v_p p m) (v_p p n)) ∧
  (v_p p (Int.lcm m n) = max (v_p p m) (v_p p n)) :=
by sorry

end NUMINAMATH_CALUDE_p_adic_valuation_properties_l2003_200341


namespace NUMINAMATH_CALUDE_order_of_x_l2003_200374

theorem order_of_x (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (eq1 : x₁ + x₂ + x₃ = a₁)
  (eq2 : x₂ + x₃ + x₄ = a₂)
  (eq3 : x₃ + x₄ + x₅ = a₃)
  (eq4 : x₄ + x₅ + x₁ = a₄)
  (eq5 : x₅ + x₁ + x₂ = a₅)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅) :
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ :=
by sorry


end NUMINAMATH_CALUDE_order_of_x_l2003_200374


namespace NUMINAMATH_CALUDE_line_translations_l2003_200380

/-- Represents a line in the form y = mx + b --/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Translates a line vertically --/
def translateVertical (l : Line) (dy : ℝ) : Line :=
  { m := l.m, b := l.b + dy }

/-- Translates a line horizontally --/
def translateHorizontal (l : Line) (dx : ℝ) : Line :=
  { m := l.m, b := l.b - l.m * dx }

theorem line_translations (original : Line) :
  (original.m = 2 ∧ original.b = -4) →
  (translateVertical original 3 = { m := 2, b := -1 } ∧
   translateHorizontal original 3 = { m := 2, b := -10 }) :=
by sorry

end NUMINAMATH_CALUDE_line_translations_l2003_200380


namespace NUMINAMATH_CALUDE_man_double_son_age_l2003_200353

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2 -/
theorem man_double_son_age 
  (son_age : ℕ) 
  (age_difference : ℕ) 
  (h1 : son_age = 18) 
  (h2 : age_difference = 20) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_double_son_age_l2003_200353


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2003_200336

theorem sum_of_x_and_y (x y : ℝ) 
  (hx : |x| = 1) 
  (hy : |y| = 2) 
  (hxy : x * y > 0) : 
  x + y = 3 ∨ x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2003_200336


namespace NUMINAMATH_CALUDE_function_divisibility_property_l2003_200329

theorem function_divisibility_property (f : ℤ → ℤ) : 
  (∀ m n : ℤ, (Int.gcd m n : ℤ) ∣ (f m + f n)) → 
  ∃ k : ℤ, ∀ n : ℤ, f n = k * n :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l2003_200329


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2003_200306

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ - 6 = 0 → 
  x₂^2 - 5*x₂ - 6 = 0 → 
  x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = -5/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2003_200306


namespace NUMINAMATH_CALUDE_lamp_distance_in_specific_classroom_l2003_200314

/-- Represents a classroom with two lamps -/
structure Classroom where
  length : ℝ
  ceiling_height : ℝ
  lamp1_position : ℝ
  lamp2_position : ℝ
  lamp1_circle_diameter : ℝ
  lamp2_illumination_length : ℝ

/-- The distance between two lamps in the classroom -/
def lamp_distance (c : Classroom) : ℝ :=
  |c.lamp1_position - c.lamp2_position|

/-- Theorem stating the distance between lamps in the specific classroom setup -/
theorem lamp_distance_in_specific_classroom :
  ∀ (c : Classroom),
    c.length = 10 ∧
    c.lamp1_circle_diameter = 6 ∧
    c.lamp2_illumination_length = 10 ∧
    c.lamp1_position = c.length / 2 ∧
    c.lamp2_position = 1 →
    lamp_distance c = 4 := by
  sorry

#check lamp_distance_in_specific_classroom

end NUMINAMATH_CALUDE_lamp_distance_in_specific_classroom_l2003_200314


namespace NUMINAMATH_CALUDE_number_problem_l2003_200360

theorem number_problem (x : ℝ) : 0.1 * x = 0.2 * 650 + 190 → x = 3200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2003_200360


namespace NUMINAMATH_CALUDE_largest_inscribed_semicircle_area_l2003_200346

theorem largest_inscribed_semicircle_area (r : ℝ) (h : r = 1) : 
  let A := π * (1 / Real.sqrt 3)^2 / 2
  120 * A / π = 20 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_semicircle_area_l2003_200346


namespace NUMINAMATH_CALUDE_exists_row_or_column_with_sqrt_n_distinct_l2003_200313

/-- Represents a grid with n rows and n columns -/
structure Grid (n : ℕ) where
  entries : Fin n → Fin n → Fin n

/-- A grid is valid if each number from 1 to n appears exactly n times -/
def isValidGrid {n : ℕ} (g : Grid n) : Prop :=
  ∀ k : Fin n, (Finset.sum Finset.univ (λ i => Finset.sum Finset.univ (λ j => if g.entries i j = k then 1 else 0))) = n

/-- The number of distinct elements in a row -/
def distinctInRow {n : ℕ} (g : Grid n) (i : Fin n) : ℕ :=
  Finset.card (Finset.image (g.entries i) Finset.univ)

/-- The number of distinct elements in a column -/
def distinctInColumn {n : ℕ} (g : Grid n) (j : Fin n) : ℕ :=
  Finset.card (Finset.image (λ i => g.entries i j) Finset.univ)

/-- The main theorem -/
theorem exists_row_or_column_with_sqrt_n_distinct {n : ℕ} (g : Grid n) (h : isValidGrid g) :
  (∃ i : Fin n, distinctInRow g i ≥ Int.ceil (Real.sqrt n)) ∨
  (∃ j : Fin n, distinctInColumn g j ≥ Int.ceil (Real.sqrt n)) := by
  sorry

end NUMINAMATH_CALUDE_exists_row_or_column_with_sqrt_n_distinct_l2003_200313


namespace NUMINAMATH_CALUDE_ellipse_properties_l2003_200399

/-- Given an ellipse with equation x²/25 + y²/9 = 1, prove its semi-major axis length and eccentricity -/
theorem ellipse_properties : ∃ (a b c : ℝ), 
  (∀ x y : ℝ, x^2/25 + y^2/9 = 1 → 
    a = 5 ∧ 
    b = 3 ∧ 
    c^2 = a^2 - b^2 ∧ 
    a = 5 ∧ 
    c/a = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2003_200399


namespace NUMINAMATH_CALUDE_return_flight_is_98_minutes_l2003_200300

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  outbound_time : ℝ
  total_time : ℝ
  still_air_difference : ℝ

/-- Calculates the return flight time given a flight scenario --/
def return_flight_time (scenario : FlightScenario) : ℝ :=
  scenario.total_time - scenario.outbound_time

/-- Theorem stating that the return flight time is 98 minutes --/
theorem return_flight_is_98_minutes (scenario : FlightScenario) 
  (h1 : scenario.outbound_time = 120)
  (h2 : scenario.total_time = 222)
  (h3 : scenario.still_air_difference = 6) :
  return_flight_time scenario = 98 := by
  sorry

#eval return_flight_time { outbound_time := 120, total_time := 222, still_air_difference := 6 }

end NUMINAMATH_CALUDE_return_flight_is_98_minutes_l2003_200300


namespace NUMINAMATH_CALUDE_base_conversion_difference_l2003_200323

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100000) * 7^5 +
  ((n / 10000) % 10) * 7^4 +
  ((n / 1000) % 10) * 7^3 +
  ((n / 100) % 10) * 7^2 +
  ((n / 10) % 10) * 7^1 +
  (n % 10) * 7^0

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 10000) * 8^4 +
  ((n / 1000) % 10) * 8^3 +
  ((n / 100) % 10) * 8^2 +
  ((n / 10) % 10) * 8^1 +
  (n % 10) * 8^0

theorem base_conversion_difference :
  base7ToBase10 543210 - base8ToBase10 43210 = 76717 := by
  sorry

#eval base7ToBase10 543210 - base8ToBase10 43210

end NUMINAMATH_CALUDE_base_conversion_difference_l2003_200323


namespace NUMINAMATH_CALUDE_f_sum_equals_half_l2003_200345

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f (x - 2)

def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > -2 ∧ x < 0 → f x = -2^x

theorem f_sum_equals_half (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period_4 f)
  (h_condition : f_condition f) :
  f 1 + f 4 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_f_sum_equals_half_l2003_200345


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2003_200311

def polynomial (z : ℂ) : ℂ := z^4 + z^3 + z^2 + z + 1

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

theorem smallest_n_for_roots_of_unity : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), polynomial z = 0 → is_nth_root_of_unity z n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (w : ℂ), polynomial w = 0 ∧ ¬is_nth_root_of_unity w m) ∧
  n = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2003_200311


namespace NUMINAMATH_CALUDE_smallest_positive_angle_correct_largest_negative_angle_correct_equivalent_angles_in_range_correct_l2003_200342

-- Define the original angle
def original_angle : Int := -2010

-- Define a function to find the smallest positive equivalent angle
def smallest_positive_equivalent (angle : Int) : Int :=
  angle % 360

-- Define a function to find the largest negative equivalent angle
def largest_negative_equivalent (angle : Int) : Int :=
  (angle % 360) - 360

-- Define a function to find equivalent angles within a range
def equivalent_angles_in_range (angle : Int) (lower : Int) (upper : Int) : List Int :=
  let base_angle := angle % 360
  List.filter (fun x => lower ≤ x ∧ x < upper)
    [base_angle - 720, base_angle - 360, base_angle, base_angle + 360]

-- Theorem statements
theorem smallest_positive_angle_correct :
  smallest_positive_equivalent original_angle = 150 := by sorry

theorem largest_negative_angle_correct :
  largest_negative_equivalent original_angle = -210 := by sorry

theorem equivalent_angles_in_range_correct :
  equivalent_angles_in_range original_angle (-720) 720 = [-570, -210, 150, 510] := by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_correct_largest_negative_angle_correct_equivalent_angles_in_range_correct_l2003_200342


namespace NUMINAMATH_CALUDE_bear_food_consumption_l2003_200383

/-- The weight of Victor in pounds -/
def victor_weight : ℝ := 126

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_in_three_weeks : ℝ := 15

/-- The number of weeks in the given condition -/
def given_weeks : ℝ := 3

/-- Theorem: For any number of weeks, the bear eats 5 times that many "Victors" worth of food -/
theorem bear_food_consumption (x : ℝ) : 
  (victors_in_three_weeks / given_weeks) * x = 5 * x := by
sorry

end NUMINAMATH_CALUDE_bear_food_consumption_l2003_200383


namespace NUMINAMATH_CALUDE_square_difference_l2003_200338

theorem square_difference (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2003_200338


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2003_200318

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  a 1 / a 0 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2003_200318


namespace NUMINAMATH_CALUDE_circle_equation_l2003_200302

-- Define the center of the circle
def center : ℝ × ℝ := (3, 1)

-- Define a point on the circle (the origin)
def origin : ℝ × ℝ := (0, 0)

-- Define the equation of a circle
def is_on_circle (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = (origin.1 - center.1)^2 + (origin.2 - center.2)^2

-- Theorem statement
theorem circle_equation : 
  ∀ x y : ℝ, is_on_circle x y ↔ (x - 3)^2 + (y - 1)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2003_200302


namespace NUMINAMATH_CALUDE_multiples_of_four_between_70_and_300_l2003_200334

theorem multiples_of_four_between_70_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 70 ∧ n < 300) (Finset.range 300)).card = 57 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_70_and_300_l2003_200334


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2003_200328

theorem largest_solution_of_equation (x : ℝ) : 
  (3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45)) →
  x ≤ (-39 + Real.sqrt 801) / 20 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2003_200328


namespace NUMINAMATH_CALUDE_range_of_f_l2003_200362

/-- The function f(x) = |x+3| - |x-5| -/
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

/-- The range of f is [-8, 18] -/
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2003_200362


namespace NUMINAMATH_CALUDE_undetermined_sum_l2003_200366

/-- Operation # defined for non-negative integers a and b, and positive integer c -/
def sharp (a b c : ℕ) : ℕ := 4 * a^3 + 4 * b^3 + 8 * a^2 * b + c

/-- Operation * defined for non-negative integers a and b, and positive integer d -/
def star (a b d : ℕ) : ℕ := 2 * a^2 - 3 * b^2 + d^3

/-- Theorem stating that the value of (a + b) + 6 cannot be determined -/
theorem undetermined_sum (a b x c d : ℕ) (hc : c > 0) (hd : d > 0) 
  (h1 : sharp a x c = 250) (h2 : star a b d + x = 50) : 
  ∃ (a' b' x' c' d' : ℕ), 
    c' > 0 ∧ d' > 0 ∧
    sharp a' x' c' = 250 ∧ 
    star a' b' d' + x' = 50 ∧
    a + b + 6 ≠ a' + b' + 6 :=
sorry

end NUMINAMATH_CALUDE_undetermined_sum_l2003_200366


namespace NUMINAMATH_CALUDE_solution_value_l2003_200389

/-- A system of two linear equations in two variables -/
structure LinearSystem :=
  (a b c : ℝ)
  (d e f : ℝ)

/-- The condition for a linear system to not have a unique solution -/
def noUniqueSolution (sys : LinearSystem) : Prop :=
  sys.a * sys.e = sys.b * sys.d ∧ sys.a * sys.f = sys.c * sys.d

/-- The theorem stating that if the given system doesn't have a unique solution, then d = 40 -/
theorem solution_value (k : ℝ) :
  let sys : LinearSystem := ⟨12, 16, d, k, 12, 30⟩
  noUniqueSolution sys → d = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_solution_value_l2003_200389


namespace NUMINAMATH_CALUDE_circle_radius_l2003_200326

theorem circle_radius (x y : ℝ) : (x - 1)^2 + y^2 = 9 → 3 = Real.sqrt ((x - 1)^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2003_200326


namespace NUMINAMATH_CALUDE_nancy_gardens_l2003_200310

theorem nancy_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : big_garden_seeds = 28)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 6 :=
by sorry

end NUMINAMATH_CALUDE_nancy_gardens_l2003_200310


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l2003_200322

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem fourth_term_of_sequence (a : ℕ → ℝ) :
  a 1 = 1 → (∀ n, a (n + 1) = 2 * a n) → a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l2003_200322


namespace NUMINAMATH_CALUDE_breakfast_cost_is_30_25_l2003_200309

/-- Represents the menu prices and orders for a breakfast at a cafe. -/
structure BreakfastOrder where
  toast_price : ℝ
  egg_price : ℝ
  coffee_price : ℝ
  juice_price : ℝ
  dale_toast : ℕ
  dale_eggs : ℕ
  dale_coffee : ℕ
  andrew_toast : ℕ
  andrew_eggs : ℕ
  andrew_juice : ℕ
  melanie_toast : ℕ
  melanie_eggs : ℕ
  melanie_juice : ℕ
  service_charge_rate : ℝ

/-- Calculates the total cost of a breakfast order including service charge. -/
def totalCost (order : BreakfastOrder) : ℝ :=
  let subtotal := 
    order.toast_price * (order.dale_toast + order.andrew_toast + order.melanie_toast : ℝ) +
    order.egg_price * (order.dale_eggs + order.andrew_eggs + order.melanie_eggs : ℝ) +
    order.coffee_price * (order.dale_coffee : ℝ) +
    order.juice_price * (order.andrew_juice + order.melanie_juice : ℝ)
  subtotal * (1 + order.service_charge_rate)

/-- Theorem stating that the total cost of the given breakfast order is £30.25. -/
theorem breakfast_cost_is_30_25 : 
  let order : BreakfastOrder := {
    toast_price := 1,
    egg_price := 3,
    coffee_price := 2,
    juice_price := 1.5,
    dale_toast := 2,
    dale_eggs := 2,
    dale_coffee := 1,
    andrew_toast := 1,
    andrew_eggs := 2,
    andrew_juice := 1,
    melanie_toast := 3,
    melanie_eggs := 1,
    melanie_juice := 2,
    service_charge_rate := 0.1
  }
  totalCost order = 30.25 := by sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_30_25_l2003_200309


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l2003_200379

/-- Given two vectors a and b that are parallel, prove that sin²α + 2sinα*cosα = 3/2 -/
theorem parallel_vectors_trig_identity (α : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (Real.sin (α - π/3), Real.cos α + π/3)
  (∃ k : ℝ, b = k • a) →
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l2003_200379


namespace NUMINAMATH_CALUDE_jennifer_museum_trips_l2003_200398

/-- Calculates the total miles traveled for round trips to two museums -/
def total_miles_traveled (distance1 distance2 : ℕ) : ℕ :=
  2 * distance1 + 2 * distance2

/-- Theorem: Jennifer travels 40 miles in total to visit both museums -/
theorem jennifer_museum_trips : total_miles_traveled 5 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_museum_trips_l2003_200398


namespace NUMINAMATH_CALUDE_necessary_condition_l2003_200395

theorem necessary_condition (a b x y : ℤ) 
  (ha : 0 < a) (hb : 0 < b) 
  (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  x > a ∧ y > b := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_l2003_200395


namespace NUMINAMATH_CALUDE_max_value_2sin_l2003_200354

theorem max_value_2sin (x : ℝ) : ∃ (M : ℝ), M = 2 ∧ ∀ y : ℝ, 2 * Real.sin x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_2sin_l2003_200354


namespace NUMINAMATH_CALUDE_sin_150_minus_sin_30_equals_zero_l2003_200307

theorem sin_150_minus_sin_30_equals_zero :
  Real.sin (150 * π / 180) - Real.sin (30 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_minus_sin_30_equals_zero_l2003_200307


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l2003_200350

/-- A sequence where a_n + 2 forms a geometric sequence -/
def IsGeometricPlus2 (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, (a (n + 1) + 2) = (a n + 2) * q

theorem geometric_sequence_a8 (a : ℕ → ℝ) 
  (h_geom : IsGeometricPlus2 a) 
  (h_a2 : a 2 = -1) 
  (h_a4 : a 4 = 2) : 
  a 8 = 62 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l2003_200350


namespace NUMINAMATH_CALUDE_midpoint_property_l2003_200349

/-- Given two points P and Q in the plane, their midpoint R satisfies 3x - 2y = -15 --/
theorem midpoint_property (P Q R : ℝ × ℝ) : 
  P = (-8, 15) → 
  Q = (6, -3) → 
  R.1 = (P.1 + Q.1) / 2 → 
  R.2 = (P.2 + Q.2) / 2 → 
  3 * R.1 - 2 * R.2 = -15 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l2003_200349


namespace NUMINAMATH_CALUDE_cube_surface_area_l2003_200394

theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 5 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 150 * (a ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2003_200394


namespace NUMINAMATH_CALUDE_exists_periodic_functions_with_nonperiodic_difference_l2003_200351

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

/-- The period of a function. -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- Theorem stating that there exist periodic functions g and h with periods 6 and 2π respectively,
    such that their difference is not periodic. -/
theorem exists_periodic_functions_with_nonperiodic_difference :
  ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ Period g 6 ∧
    IsPeriodic h ∧ Period h (2 * Real.pi) ∧
    ¬IsPeriodic (g - h) := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_functions_with_nonperiodic_difference_l2003_200351


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2003_200304

open Complex

theorem smallest_absolute_value_of_z (z : ℂ) (h : abs (z - 12) + abs (z - 5*I) = 13) :
  ∃ (w : ℂ), abs (z - 12) + abs (z - 5*I) = 13 ∧ abs w ≤ abs z ∧ abs w = 60 / 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2003_200304


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l2003_200305

theorem smallest_advantageous_discount : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    (1 - m / 100 : ℝ) ≥ (1 - 0.15)^2 ∨ 
    (1 - m / 100 : ℝ) ≥ (1 - 0.10)^3 ∨ 
    (1 - m / 100 : ℝ) ≥ (1 - 0.25) * (1 - 0.05)) ∧
  (1 - n / 100 : ℝ) < (1 - 0.15)^2 ∧
  (1 - n / 100 : ℝ) < (1 - 0.10)^3 ∧
  (1 - n / 100 : ℝ) < (1 - 0.25) * (1 - 0.05) ∧
  n = 29 :=
by sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l2003_200305


namespace NUMINAMATH_CALUDE_fish_tagging_theorem_l2003_200340

/-- The number of fish in the pond -/
def total_fish : ℕ := 3200

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 80

/-- The number of tagged fish found in the second catch -/
def tagged_in_second : ℕ := 2

/-- The number of fish initially caught, tagged, and returned -/
def initially_tagged : ℕ := 80

theorem fish_tagging_theorem :
  (tagged_in_second : ℚ) / second_catch = initially_tagged / total_fish →
  initially_tagged = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_tagging_theorem_l2003_200340


namespace NUMINAMATH_CALUDE_parallelogram_exists_l2003_200320

/-- Represents a cell in the grid -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents the grid and its blue cells -/
structure Grid where
  n : Nat
  blue_cells : Finset Cell

/-- Predicate to check if four cells form a parallelogram -/
def is_parallelogram (c1 c2 c3 c4 : Cell) : Prop :=
  (c2.x - c1.x = c4.x - c3.x) ∧ (c2.y - c1.y = c4.y - c3.y)

/-- Main theorem: In an n x n grid with 2n blue cells, there exist 4 blue cells forming a parallelogram -/
theorem parallelogram_exists (g : Grid) (h1 : g.blue_cells.card = 2 * g.n) :
  ∃ c1 c2 c3 c4 : Cell, c1 ∈ g.blue_cells ∧ c2 ∈ g.blue_cells ∧ c3 ∈ g.blue_cells ∧ c4 ∈ g.blue_cells ∧
    is_parallelogram c1 c2 c3 c4 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_exists_l2003_200320


namespace NUMINAMATH_CALUDE_relative_error_comparison_l2003_200301

theorem relative_error_comparison :
  let line1_length : ℚ := 15
  let line1_error : ℚ := 3 / 100
  let line2_length : ℚ := 125
  let line2_error : ℚ := 1 / 4
  let relative_error1 : ℚ := line1_error / line1_length
  let relative_error2 : ℚ := line2_error / line2_length
  relative_error1 = relative_error2 :=
by sorry

end NUMINAMATH_CALUDE_relative_error_comparison_l2003_200301


namespace NUMINAMATH_CALUDE_prayer_difference_l2003_200359

/-- Represents the number of prayers for a pastor in a week -/
structure WeeklyPrayers where
  weekday : ℕ  -- Number of prayers on a weekday
  sunday : ℕ   -- Number of prayers on Sunday

/-- Calculates the total number of prayers in a week -/
def totalPrayers (wp : WeeklyPrayers) : ℕ :=
  6 * wp.weekday + wp.sunday

/-- Pastor Paul's prayer schedule -/
def paulPrayers : WeeklyPrayers where
  weekday := 20
  sunday := 40

/-- Pastor Bruce's prayer schedule -/
def brucePrayers : WeeklyPrayers where
  weekday := paulPrayers.weekday / 2
  sunday := 2 * paulPrayers.sunday

theorem prayer_difference :
  totalPrayers paulPrayers - totalPrayers brucePrayers = 20 := by
  sorry

end NUMINAMATH_CALUDE_prayer_difference_l2003_200359


namespace NUMINAMATH_CALUDE_december_burger_expenditure_l2003_200330

/-- The daily expenditure on burgers given the total monthly expenditure and number of days -/
def daily_burger_expenditure (total_expenditure : ℚ) (days : ℕ) : ℚ :=
  total_expenditure / days

theorem december_burger_expenditure :
  let total_expenditure : ℚ := 465
  let days : ℕ := 31
  daily_burger_expenditure total_expenditure days = 15 := by
sorry

end NUMINAMATH_CALUDE_december_burger_expenditure_l2003_200330


namespace NUMINAMATH_CALUDE_toys_in_box_time_l2003_200385

/-- The time it takes to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_minute : ℕ) (toys_out_per_minute : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 25 minutes to put all toys in the box under given conditions -/
theorem toys_in_box_time : time_to_put_toys_in_box 50 5 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_toys_in_box_time_l2003_200385


namespace NUMINAMATH_CALUDE_stadium_length_l2003_200393

theorem stadium_length (w h p : ℝ) (hw : w = 18) (hh : h = 16) (hp : p = 34) :
  ∃ l : ℝ, l = 24 ∧ p^2 = l^2 + w^2 + h^2 :=
by sorry

end NUMINAMATH_CALUDE_stadium_length_l2003_200393


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2003_200377

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (a * Real.cos x - 1) * (a * x^2 - x + 16 * a) < 0) ↔ 
  (a < -1 ∨ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2003_200377


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2003_200396

theorem geometric_sequence_fourth_term 
  (x : ℝ) 
  (h1 : ∃ r : ℝ, (3*x + 3) = x * r) 
  (h2 : ∃ r : ℝ, (6*x + 6) = (3*x + 3) * r) :
  ∃ r : ℝ, -24 = (6*x + 6) * r :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2003_200396


namespace NUMINAMATH_CALUDE_ballpoint_pen_price_l2003_200386

-- Define the problem parameters
def total_pens : Nat := 30
def total_pencils : Nat := 75
def total_cost : ℝ := 690

def gel_pens : Nat := 20
def ballpoint_pens : Nat := 10
def standard_pencils : Nat := 50
def mechanical_pencils : Nat := 25

def avg_price_gel : ℝ := 1.5
def avg_price_mechanical : ℝ := 3
def avg_price_standard : ℝ := 2

-- Theorem to prove
theorem ballpoint_pen_price :
  ∃ (avg_price_ballpoint : ℝ),
    avg_price_ballpoint = 48.5 ∧
    total_cost = 
      gel_pens * avg_price_gel +
      mechanical_pencils * avg_price_mechanical +
      standard_pencils * avg_price_standard +
      ballpoint_pens * avg_price_ballpoint :=
by sorry

end NUMINAMATH_CALUDE_ballpoint_pen_price_l2003_200386


namespace NUMINAMATH_CALUDE_right_triangle_rotation_volumes_l2003_200325

theorem right_triangle_rotation_volumes 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (K₁ K₂ K₃ : ℝ),
    K₁ = (2/3) * a * b^2 * Real.pi ∧
    K₂ = (2/3) * a^2 * b * Real.pi ∧
    K₃ = (2/3) * (a^2 * b^2) / c * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_volumes_l2003_200325


namespace NUMINAMATH_CALUDE_decreasing_interval_of_even_f_l2003_200355

def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + (k - 1) * x + 3

theorem decreasing_interval_of_even_f (k : ℝ) :
  (∀ x, f k x = f k (-x)) →
  ∀ x > 0, ∀ y > x, f k y < f k x :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_even_f_l2003_200355


namespace NUMINAMATH_CALUDE_quadrilateral_diagonals_l2003_200332

-- Define a convex quadrilateral
structure ConvexQuadrilateral :=
  (perimeter : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (is_convex : perimeter > 0 ∧ diagonal1 > 0 ∧ diagonal2 > 0)

-- Theorem statement
theorem quadrilateral_diagonals 
  (q : ConvexQuadrilateral) 
  (h1 : q.perimeter = 2004) 
  (h2 : q.diagonal1 = 1001) : 
  (q.diagonal2 ≠ 1) ∧ 
  (∃ q' : ConvexQuadrilateral, q'.perimeter = 2004 ∧ q'.diagonal1 = 1001 ∧ q'.diagonal2 = 2) ∧
  (∃ q'' : ConvexQuadrilateral, q''.perimeter = 2004 ∧ q''.diagonal1 = 1001 ∧ q''.diagonal2 = 1001) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonals_l2003_200332


namespace NUMINAMATH_CALUDE_cyclist_round_time_l2003_200388

/-- Proves that a cyclist completes one round of a rectangular park in 8 minutes
    given the specified conditions. -/
theorem cyclist_round_time (length width : ℝ) (area perimeter : ℝ) (speed : ℝ) : 
  length / width = 4 →
  area = length * width →
  area = 102400 →
  perimeter = 2 * (length + width) →
  speed = 12 * 1000 / 3600 →
  (perimeter / speed) / 60 = 8 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_round_time_l2003_200388


namespace NUMINAMATH_CALUDE_license_plate_difference_l2003_200358

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of digits available --/
def num_digits : ℕ := 10

/-- The number of possible Florida license plates --/
def florida_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible North Dakota license plates --/
def north_dakota_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of possible license plates between Florida and North Dakota --/
def plate_difference : ℕ := florida_plates - north_dakota_plates

theorem license_plate_difference : plate_difference = 28121600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2003_200358


namespace NUMINAMATH_CALUDE_train_length_l2003_200381

theorem train_length (t_platform : ℝ) (t_pole : ℝ) (l_platform : ℝ)
  (h1 : t_platform = 33)
  (h2 : t_pole = 18)
  (h3 : l_platform = 250) :
  ∃ l_train : ℝ, l_train = 300 ∧ (l_train + l_platform) / t_platform = l_train / t_pole :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2003_200381


namespace NUMINAMATH_CALUDE_album_collection_problem_l2003_200317

/-- The number of albums in either Andrew's or John's collection, but not both -/
def exclusive_albums (shared : ℕ) (andrew_total : ℕ) (john_exclusive : ℕ) : ℕ :=
  (andrew_total - shared) + john_exclusive

theorem album_collection_problem (shared : ℕ) (andrew_total : ℕ) (john_exclusive : ℕ)
  (h1 : shared = 12)
  (h2 : andrew_total = 20)
  (h3 : john_exclusive = 8) :
  exclusive_albums shared andrew_total john_exclusive = 16 := by
  sorry

end NUMINAMATH_CALUDE_album_collection_problem_l2003_200317


namespace NUMINAMATH_CALUDE_binary_sum_to_hex_l2003_200363

/-- The sum of 11111111111₂ and 11111111₂ in base 16 is 8FE₁₆ -/
theorem binary_sum_to_hex : 
  (fun (n : ℕ) => (2^11 - 1) + (2^8 - 1)) 0 = 
  (fun (n : ℕ) => 8 * 16^2 + 15 * 16^1 + 14 * 16^0) 0 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_to_hex_l2003_200363


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2003_200352

theorem sum_of_squares_of_roots : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 2*x₁ + 4)^(x₁^2 - 2*x₁ + 3) = 625 ∧
  (x₂^2 - 2*x₂ + 4)^(x₂^2 - 2*x₂ + 3) = 625 ∧
  x₁ ≠ x₂ ∧
  x₁^2 + x₂^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2003_200352


namespace NUMINAMATH_CALUDE_firecrackers_confiscated_l2003_200315

theorem firecrackers_confiscated (initial : ℕ) (remaining : ℕ) : 
  initial = 48 →
  remaining < initial →
  (1 : ℚ) / 6 * remaining = remaining - (2 * 15) →
  initial - remaining = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_firecrackers_confiscated_l2003_200315


namespace NUMINAMATH_CALUDE_black_squares_in_45th_row_l2003_200324

/-- Represents the number of squares in the nth row of the stair-step pattern -/
def squares_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of black squares in the nth row of the stair-step pattern -/
def black_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

/-- Theorem stating that the number of black squares in the 45th row is 45 -/
theorem black_squares_in_45th_row :
  black_squares_in_row 45 = 45 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_in_45th_row_l2003_200324


namespace NUMINAMATH_CALUDE_quadratic_with_irrational_root_l2003_200348

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_with_irrational_root_l2003_200348


namespace NUMINAMATH_CALUDE_subset_removal_distinctness_l2003_200387

theorem subset_removal_distinctness (n : ℕ) :
  ∀ (S : Finset ℕ) (A : Fin n → Finset ℕ),
    S = Finset.range n →
    (∀ i j, i ≠ j → A i ≠ A j) →
    (∀ i, A i ⊆ S) →
    ∃ x ∈ S, ∀ i j, i ≠ j → A i \ {x} ≠ A j \ {x} :=
by sorry

end NUMINAMATH_CALUDE_subset_removal_distinctness_l2003_200387


namespace NUMINAMATH_CALUDE_lucky_coin_steps_l2003_200392

/-- Represents the state of a coin on the number line -/
inductive CoinState
| HeadsUp
| TailsUp
| NoCoin

/-- Represents the direction Lucky is facing -/
inductive Direction
| Positive
| Negative

/-- Represents Lucky's position and the state of the number line -/
structure GameState where
  position : Int
  direction : Direction
  coins : Int → CoinState

/-- Represents the procedure Lucky follows -/
def step (state : GameState) : GameState :=
  sorry

/-- Counts the number of tails-up coins -/
def countTailsUp (coins : Int → CoinState) : Nat :=
  sorry

/-- Theorem stating that the process stops after 6098 steps -/
theorem lucky_coin_steps :
  ∀ (initial : GameState),
    initial.position = 0 ∧
    initial.direction = Direction.Positive ∧
    (∀ n : Int, initial.coins n = CoinState.HeadsUp) →
    ∃ (final : GameState) (steps : Nat),
      steps = 6098 ∧
      countTailsUp final.coins = 20 ∧
      (∀ k : Nat, k < steps → countTailsUp (step^[k] initial).coins < 20) :=
  sorry

end NUMINAMATH_CALUDE_lucky_coin_steps_l2003_200392


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l2003_200375

/-- The percentage of motorists who exceed the speed limit -/
def speeding_percentage : ℝ := 25

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 60

/-- The percentage of motorists who receive speeding tickets -/
def ticket_percentage : ℝ := 10

theorem speeding_ticket_percentage :
  ticket_percentage = speeding_percentage * (1 - no_ticket_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l2003_200375


namespace NUMINAMATH_CALUDE_exist_triangle_area_le_two_l2003_200319

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define the condition for points within the square region
def WithinSquare (p : LatticePoint) : Prop :=
  |p.1| ≤ 2 ∧ |p.2| ≤ 2

-- Define a function to calculate the area of a triangle given three points
def TriangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Define the property of three points not being collinear
def NotCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  TriangleArea p1 p2 p3 ≠ 0

-- Main theorem
theorem exist_triangle_area_le_two 
  (points : Fin 6 → LatticePoint)
  (h1 : ∀ i, WithinSquare (points i))
  (h2 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → NotCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ TriangleArea (points i) (points j) (points k) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_exist_triangle_area_le_two_l2003_200319


namespace NUMINAMATH_CALUDE_ammonia_formed_l2003_200390

-- Define the chemical species
structure ChemicalSpecies where
  name : String
  coefficient : ℕ

-- Define the chemical equation
structure ChemicalEquation where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

-- Define the reaction conditions
structure ReactionConditions where
  li3n_amount : ℚ
  h2o_amount : ℚ
  lioh_amount : ℚ

-- Define the balanced equation
def balanced_equation : ChemicalEquation :=
  { reactants := [
      { name := "Li3N", coefficient := 1 },
      { name := "H2O", coefficient := 3 }
    ],
    products := [
      { name := "LiOH", coefficient := 3 },
      { name := "NH3", coefficient := 1 }
    ]
  }

-- Define the reaction conditions
def reaction_conditions : ReactionConditions :=
  { li3n_amount := 1,
    h2o_amount := 54,
    lioh_amount := 3 }

-- Theorem statement
theorem ammonia_formed (eq : ChemicalEquation) (conditions : ReactionConditions) :
  eq = balanced_equation ∧
  conditions = reaction_conditions →
  ∃ (nh3_amount : ℚ), nh3_amount = 1 :=
sorry

end NUMINAMATH_CALUDE_ammonia_formed_l2003_200390


namespace NUMINAMATH_CALUDE_lottery_probability_l2003_200308

/-- The probability of winning in a lottery with 10 balls labeled 1 to 10, 
    where winning occurs if the selected number is not less than 6. -/
theorem lottery_probability : 
  let total_balls : ℕ := 10
  let winning_balls : ℕ := 5
  let probability := winning_balls / total_balls
  probability = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_lottery_probability_l2003_200308


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l2003_200378

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when two dice are rolled -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (sum > 7) -/
def favorableOutcomes : ℕ := totalOutcomes - 21

/-- The probability of the sum being greater than 7 when two fair dice are rolled -/
def probSumGreaterThanSeven : ℚ := favorableOutcomes / totalOutcomes

theorem prob_sum_greater_than_seven :
  probSumGreaterThanSeven = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l2003_200378


namespace NUMINAMATH_CALUDE_nancy_antacid_consumption_l2003_200316

/-- Calculates the number of antacids Nancy takes per month based on her eating habits. -/
def antacids_per_month (indian_antacids : ℕ) (mexican_antacids : ℕ) (other_antacids : ℕ)
  (indian_freq : ℕ) (mexican_freq : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let other_days := days_per_week - indian_freq - mexican_freq
  let weekly_antacids := indian_antacids * indian_freq + mexican_antacids * mexican_freq + other_antacids * other_days
  weekly_antacids * weeks_per_month

/-- Theorem stating that Nancy takes 60 antacids per month given her eating habits. -/
theorem nancy_antacid_consumption :
  antacids_per_month 3 2 1 3 2 7 4 = 60 := by
  sorry

#eval antacids_per_month 3 2 1 3 2 7 4

end NUMINAMATH_CALUDE_nancy_antacid_consumption_l2003_200316


namespace NUMINAMATH_CALUDE_prob_two_boys_one_girl_l2003_200371

/-- A hobby group with boys and girls -/
structure HobbyGroup where
  boys : Nat
  girls : Nat

/-- The probability of selecting exactly one boy and one girl -/
def prob_one_boy_one_girl (group : HobbyGroup) : Rat :=
  if group.boys ≥ 1 ∧ group.girls ≥ 1 then
    (group.boys * group.girls : Rat) / (group.boys + group.girls).choose 2
  else
    0

/-- Theorem: The probability of selecting exactly one boy and one girl
    from a group of 2 boys and 1 girl is 2/3 -/
theorem prob_two_boys_one_girl :
  prob_one_boy_one_girl ⟨2, 1⟩ = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_boys_one_girl_l2003_200371


namespace NUMINAMATH_CALUDE_borrowed_amount_correct_l2003_200344

/-- The amount of money Yoque borrowed -/
def borrowed_amount : ℝ := 150

/-- The number of months for repayment -/
def repayment_months : ℕ := 11

/-- The additional percentage added to the repayment -/
def additional_percentage : ℝ := 0.1

/-- The monthly payment amount -/
def monthly_payment : ℝ := 15

/-- Theorem stating that the borrowed amount satisfies the given conditions -/
theorem borrowed_amount_correct : 
  borrowed_amount * (1 + additional_percentage) = repayment_months * monthly_payment := by
  sorry


end NUMINAMATH_CALUDE_borrowed_amount_correct_l2003_200344


namespace NUMINAMATH_CALUDE_typing_speed_difference_l2003_200339

theorem typing_speed_difference (before_speed after_speed : ℕ) 
  (h1 : before_speed = 10) 
  (h2 : after_speed = 8) 
  (difference : ℕ) 
  (h3 : difference = 10) : 
  ∃ (minutes : ℕ), minutes * before_speed - minutes * after_speed = difference ∧ minutes = 5 :=
sorry

end NUMINAMATH_CALUDE_typing_speed_difference_l2003_200339


namespace NUMINAMATH_CALUDE_colored_balls_permutations_l2003_200335

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  factorial n / (counts.map factorial).prod

theorem colored_balls_permutations :
  let total_balls : ℕ := 5
  let color_counts : List ℕ := [1, 1, 2, 1]  -- red, blue, yellow, white
  multiset_permutations total_balls color_counts = 60 := by
  sorry

end NUMINAMATH_CALUDE_colored_balls_permutations_l2003_200335


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l2003_200331

def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 8 = 0

theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → given_line x y → (x + 2) * (y - 3) = 0) ∧
  (∀ x y : ℝ, given_line x y → perpendicular_line x y → (x + 2) * (2 * x + y) + (y - 3) * (x + 2 * y) = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l2003_200331


namespace NUMINAMATH_CALUDE_max_distinct_values_l2003_200367

-- Define a 4x4 grid of non-negative integers
def Grid := Matrix (Fin 4) (Fin 4) ℕ

-- Define a function to check if a set of 5 cells sums to 5
def SumToFive (g : Grid) (cells : Finset (Fin 4 × Fin 4)) : Prop :=
  cells.card = 5 ∧ (cells.sum (fun c => g c.1 c.2) = 5)

-- Define the property that all valid 5-cell configurations sum to 5
def AllConfigsSumToFive (g : Grid) : Prop :=
  ∀ cells : Finset (Fin 4 × Fin 4), SumToFive g cells

-- Define the number of distinct values in the grid
def DistinctValues (g : Grid) : ℕ :=
  (Finset.univ.image (fun i => Finset.univ.image (g i))).card

-- State the theorem
theorem max_distinct_values (g : Grid) (h : AllConfigsSumToFive g) :
  DistinctValues g ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_values_l2003_200367


namespace NUMINAMATH_CALUDE_power_multiplication_l2003_200364

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2003_200364


namespace NUMINAMATH_CALUDE_water_capacity_equals_volume_l2003_200368

/-- A cylindrical bucket -/
structure CylindricalBucket where
  volume : ℝ
  lateral_area : ℝ
  surface_area : ℝ

/-- The amount of water a cylindrical bucket can hold -/
def water_capacity (bucket : CylindricalBucket) : ℝ := sorry

/-- Theorem: The amount of water a cylindrical bucket can hold is equal to its volume -/
theorem water_capacity_equals_volume (bucket : CylindricalBucket) :
  water_capacity bucket = bucket.volume := sorry

end NUMINAMATH_CALUDE_water_capacity_equals_volume_l2003_200368


namespace NUMINAMATH_CALUDE_either_equal_or_irrational_l2003_200372

theorem either_equal_or_irrational (m : ℤ) (n : ℝ) 
  (h : m^2 + 1/n = n^2 + 1/m) : n = m ∨ ¬(∃ (p q : ℤ), n = p / q) :=
by sorry

end NUMINAMATH_CALUDE_either_equal_or_irrational_l2003_200372


namespace NUMINAMATH_CALUDE_four_integer_average_l2003_200321

theorem four_integer_average (a b c d : ℕ+) : 
  (a + b : ℚ) / 2 = 35 →
  c ≤ 130 →
  d ≤ 130 →
  (a + b + c + d : ℚ) / 4 = 50.25 :=
by sorry

end NUMINAMATH_CALUDE_four_integer_average_l2003_200321


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2003_200369

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_difference (a₁_A a₁_B d_A d_B : ℝ) (n : ℕ) :
  a₁_A = 20 ∧ a₁_B = 40 ∧ d_A = 12 ∧ d_B = -12 ∧ n = 51 →
  |arithmetic_sequence a₁_A d_A n - arithmetic_sequence a₁_B d_B n| = 1180 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2003_200369


namespace NUMINAMATH_CALUDE_annual_salary_is_20_l2003_200391

/-- Represents the total annual cash salary in rupees -/
def annual_salary : ℕ := sorry

/-- Represents the number of months the servant worked -/
def months_worked : ℕ := 9

/-- Represents the total amount received by the servant after 9 months in rupees -/
def amount_received : ℕ := 55

/-- Represents the price of the turban in rupees -/
def turban_price : ℕ := 50

/-- Theorem stating that the annual salary is 20 rupees -/
theorem annual_salary_is_20 :
  annual_salary = 20 :=
by sorry

end NUMINAMATH_CALUDE_annual_salary_is_20_l2003_200391


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2003_200347

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x / a + y / b = 1
  let foci_distance_sum := 4 * c / 5
  let eccentricity := c / a
  (∀ x y, hyperbola x y → line x y) → 
  (foci_distance_sum = 2 * b) →
  eccentricity = 5 * Real.sqrt 21 / 21 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2003_200347


namespace NUMINAMATH_CALUDE_g_geq_neg_two_solution_set_f_minus_g_geq_m_plus_two_iff_l2003_200384

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x - 1| + 2
def g (x : ℝ) : ℝ := -|x + 2| + 3

-- Theorem for the first part of the problem
theorem g_geq_neg_two_solution_set :
  {x : ℝ | g x ≥ -2} = {x : ℝ | -7 ≤ x ∧ x ≤ 3} :=
sorry

-- Theorem for the second part of the problem
theorem f_minus_g_geq_m_plus_two_iff (m : ℝ) :
  (∀ x : ℝ, f x - g x ≥ m + 2) ↔ m ≤ -1/2 :=
sorry

end NUMINAMATH_CALUDE_g_geq_neg_two_solution_set_f_minus_g_geq_m_plus_two_iff_l2003_200384


namespace NUMINAMATH_CALUDE_share_difference_for_given_distribution_l2003_200312

/-- Represents the distribution of money among three people -/
structure Distribution where
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ
  share2 : ℕ

/-- Calculates the difference between the first and third person's shares -/
def shareDifference (d : Distribution) : ℕ :=
  let part := d.share2 / d.ratio2
  let share1 := part * d.ratio1
  let share3 := part * d.ratio3
  share3 - share1

/-- Theorem stating the difference between shares for the given distribution -/
theorem share_difference_for_given_distribution :
  ∀ d : Distribution,
    d.ratio1 = 3 ∧ d.ratio2 = 5 ∧ d.ratio3 = 9 ∧ d.share2 = 1500 →
    shareDifference d = 1800 := by
  sorry

#check share_difference_for_given_distribution

end NUMINAMATH_CALUDE_share_difference_for_given_distribution_l2003_200312


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2003_200361

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 1/2| < 1/2 → x^3 < 1) ∧
  ∃ y : ℝ, y^3 < 1 ∧ |y - 1/2| ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2003_200361


namespace NUMINAMATH_CALUDE_captain_america_awakening_year_l2003_200356

theorem captain_america_awakening_year : 2019 * 0.313 + 2.019 * 687 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_captain_america_awakening_year_l2003_200356


namespace NUMINAMATH_CALUDE_maria_carrots_thrown_out_l2003_200373

/-- The number of carrots Maria initially picked -/
def initial_carrots : ℕ := 48

/-- The number of additional carrots Maria picked the next day -/
def additional_carrots : ℕ := 15

/-- The total number of carrots Maria had after picking additional carrots -/
def total_carrots : ℕ := 52

/-- The number of carrots Maria threw out -/
def carrots_thrown_out : ℕ := 11

theorem maria_carrots_thrown_out : 
  initial_carrots - carrots_thrown_out + additional_carrots = total_carrots :=
sorry

end NUMINAMATH_CALUDE_maria_carrots_thrown_out_l2003_200373


namespace NUMINAMATH_CALUDE_acid_solution_concentration_l2003_200382

/-- Proves that replacing half of a 50% acid solution with a solution of unknown concentration to obtain a 40% solution implies the unknown concentration is 30% -/
theorem acid_solution_concentration (original_concentration : ℝ) 
  (final_concentration : ℝ) (replaced_fraction : ℝ) (replacement_concentration : ℝ) :
  original_concentration = 50 →
  final_concentration = 40 →
  replaced_fraction = 0.5 →
  (1 - replaced_fraction) * original_concentration + replaced_fraction * replacement_concentration = 100 * final_concentration →
  replacement_concentration = 30 := by
sorry

end NUMINAMATH_CALUDE_acid_solution_concentration_l2003_200382


namespace NUMINAMATH_CALUDE_team_a_games_l2003_200370

theorem team_a_games (a : ℕ) : 
  (2 : ℚ) / 3 * a + (1 : ℚ) / 3 * a = a → -- Team A's wins + losses = total games
  (3 : ℚ) / 5 * (a + 12) = (2 : ℚ) / 3 * a + 6 → -- Team B's wins = Team A's wins + 6
  (2 : ℚ) / 5 * (a + 12) = (1 : ℚ) / 3 * a + 6 → -- Team B's losses = Team A's losses + 6
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_team_a_games_l2003_200370


namespace NUMINAMATH_CALUDE_joan_payment_amount_l2003_200343

-- Define the costs and change as constants
def cat_toy_cost : ℚ := 877 / 100
def cage_cost : ℚ := 1097 / 100
def change_received : ℚ := 26 / 100

-- Define the theorem
theorem joan_payment_amount :
  cat_toy_cost + cage_cost + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_joan_payment_amount_l2003_200343
