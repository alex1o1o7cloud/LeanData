import Mathlib

namespace NUMINAMATH_CALUDE_unique_six_digit_number_l765_76528

def is_valid_number (n : ℕ) : Prop :=
  (100000 ≤ n) ∧ (n < 1000000) ∧ (n / 100000 = 1) ∧
  ((n % 100000) * 10 + 1 = 3 * n)

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l765_76528


namespace NUMINAMATH_CALUDE_expand_product_l765_76513

theorem expand_product (x : ℝ) : (7*x + 5) * (5*x^2 - 2*x + 4) = 35*x^3 + 11*x^2 + 18*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l765_76513


namespace NUMINAMATH_CALUDE_tangent_and_cosine_identities_l765_76578

theorem tangent_and_cosine_identities 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : 0 < β ∧ β < π) 
  (h3 : (Real.tan α)^2 - 5*(Real.tan α) + 6 = 0) 
  (h4 : (Real.tan β)^2 - 5*(Real.tan β) + 6 = 0) : 
  Real.tan (α + β) = -1 ∧ Real.cos (α - β) = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_cosine_identities_l765_76578


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l765_76591

theorem smallest_n_divisibility : ∃ (n : ℕ), n = 4058209 ∧
  (∃ (m : ℤ), n + 2015 = 2016 * m) ∧
  (∃ (k : ℤ), n + 2016 = 2015 * k) ∧
  (∀ (n' : ℕ), n' < n →
    (∃ (m : ℤ), n' + 2015 = 2016 * m) →
    (∃ (k : ℤ), n' + 2016 = 2015 * k) → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l765_76591


namespace NUMINAMATH_CALUDE_second_visit_date_l765_76538

/-- Represents the bill amount for a single person --/
structure Bill :=
  (base : ℕ)
  (date : ℕ)

/-- The restaurant scenario --/
structure RestaurantScenario :=
  (first_visit : Bill)
  (second_visit : Bill)
  (num_friends : ℕ)
  (days_between : ℕ)

/-- The conditions of the problem --/
def problem_conditions (scenario : RestaurantScenario) : Prop :=
  scenario.num_friends = 3 ∧
  scenario.days_between = 4 ∧
  scenario.first_visit.base + scenario.first_visit.date = 168 ∧
  scenario.num_friends * scenario.second_visit.base + scenario.second_visit.date = 486 ∧
  scenario.first_visit.base = scenario.second_visit.base ∧
  scenario.second_visit.date = scenario.first_visit.date + scenario.days_between

/-- The theorem to prove --/
theorem second_visit_date (scenario : RestaurantScenario) :
  problem_conditions scenario → scenario.second_visit.date = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_second_visit_date_l765_76538


namespace NUMINAMATH_CALUDE_car_speed_in_kmph_l765_76555

/-- Proves that a car covering 375 meters in 15 seconds has a speed of 90 kmph -/
theorem car_speed_in_kmph : 
  let distance : ℝ := 375 -- distance in meters
  let time : ℝ := 15 -- time in seconds
  let conversion_factor : ℝ := 3.6 -- conversion factor from m/s to kmph
  (distance / time) * conversion_factor = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_in_kmph_l765_76555


namespace NUMINAMATH_CALUDE_tangent_line_sin_plus_one_l765_76573

/-- The equation of the tangent line to y = sin x + 1 at (0, 1) is x - y + 1 = 0 -/
theorem tangent_line_sin_plus_one (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.sin t + 1
  let df : ℝ → ℝ := λ t => Real.cos t
  let tangent_point : ℝ × ℝ := (0, 1)
  let tangent_slope : ℝ := df tangent_point.1
  x - y + 1 = 0 ↔ y = tangent_slope * (x - tangent_point.1) + tangent_point.2 :=
by
  sorry

#check tangent_line_sin_plus_one

end NUMINAMATH_CALUDE_tangent_line_sin_plus_one_l765_76573


namespace NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l765_76572

theorem odd_prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) :
  (∃ (a b : ℕ+), a.val^2 + b.val^2 = p) ↔ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l765_76572


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l765_76563

/-- For any positive integer n, there exists a positive integer m such that
    for all k in the range 0 ≤ k < n, m + k is not an integer power of a prime number. -/
theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ m : ℕ+, ∀ k : ℕ, k < n → ¬∃ (p : ℕ) (e : ℕ), Prime p ∧ (m + k : ℕ) = p ^ e :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l765_76563


namespace NUMINAMATH_CALUDE_transistor_count_2005_l765_76507

/-- Calculates the number of transistors in a CPU after applying Moore's law and an additional growth law over a specified time period. -/
def transistor_count (initial_count : ℕ) (years : ℕ) : ℕ :=
  let doubling_cycles := years / 2
  let tripling_cycles := years / 6
  initial_count * 2^doubling_cycles + initial_count * 3^tripling_cycles

/-- Theorem stating that the number of transistors in a CPU in 2005 is 68,500,000,
    given an initial count of 500,000 in 1990 and the application of Moore's law
    and an additional growth law. -/
theorem transistor_count_2005 :
  transistor_count 500000 15 = 68500000 := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2005_l765_76507


namespace NUMINAMATH_CALUDE_equation_identity_l765_76544

theorem equation_identity (a b c x : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a^2 * ((x - b) / (a - b)) * ((x - c) / (a - c)) +
  b^2 * ((x - a) / (b - a)) * ((x - c) / (b - c)) +
  c^2 * ((x - a) / (c - a)) * ((x - b) / (c - b)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l765_76544


namespace NUMINAMATH_CALUDE_activity_ratio_theorem_l765_76590

/-- Represents the ratio of time spent on two activities -/
structure TimeRatio where
  activity1 : ℝ
  activity2 : ℝ

/-- Calculates the score based on time spent on an activity -/
def calculateScore (pointsPerHour : ℝ) (hours : ℝ) : ℝ :=
  pointsPerHour * hours

/-- Theorem stating the relationship between activities and score -/
theorem activity_ratio_theorem (timeActivity1 : ℝ) (pointsPerHour : ℝ) (finalScore : ℝ) :
  timeActivity1 = 9 →
  pointsPerHour = 15 →
  finalScore = 45 →
  ∃ (ratio : TimeRatio),
    ratio.activity1 = timeActivity1 ∧
    ratio.activity2 = finalScore / pointsPerHour ∧
    ratio.activity2 / ratio.activity1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_activity_ratio_theorem_l765_76590


namespace NUMINAMATH_CALUDE_profit_conditions_l765_76596

/-- Represents the profit function given the price increase -/
def profit_function (x : ℝ) : ℝ := (50 - 40 + x) * (500 - 10 * x)

/-- Represents the selling price given the price increase -/
def selling_price (x : ℝ) : ℝ := x + 50

/-- Represents the number of units sold given the price increase -/
def units_sold (x : ℝ) : ℝ := 500 - 10 * x

/-- Theorem stating the conditions for achieving a profit of 8000 yuan -/
theorem profit_conditions :
  (∃ x : ℝ, profit_function x = 8000 ∧
    ((selling_price x = 60 ∧ units_sold x = 400) ∨
     (selling_price x = 80 ∧ units_sold x = 200))) :=
by sorry

end NUMINAMATH_CALUDE_profit_conditions_l765_76596


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l765_76549

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l765_76549


namespace NUMINAMATH_CALUDE_trajectory_equation_MN_range_l765_76552

-- Define the circle P
structure CircleP where
  center : ℝ × ℝ
  passes_through_F : center.1^2 + center.2^2 = (center.1 - 1)^2 + center.2^2
  tangent_to_l : center.1 + 1 = abs center.2

-- Define the circle F
def circleF (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory C
def trajectoryC (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points M and N
structure Intersection (p : CircleP) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  on_circle_P : (M.1 - p.center.1)^2 + (M.2 - p.center.2)^2 = (p.center.1 - 1)^2 + p.center.2^2
              ∧ (N.1 - p.center.1)^2 + (N.2 - p.center.2)^2 = (p.center.1 - 1)^2 + p.center.2^2
  on_circle_F : circleF M.1 M.2 ∧ circleF N.1 N.2

-- Theorem statements
theorem trajectory_equation (p : CircleP) : trajectoryC p.center.1 p.center.2 := by sorry

theorem MN_range (p : CircleP) (i : Intersection p) : 
  Real.sqrt 3 ≤ Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) ∧ 
  Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) < 2 := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_MN_range_l765_76552


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l765_76582

theorem complex_magnitude_equation (z : ℂ) : 
  (z + Complex.I) * (1 - Complex.I) = 1 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l765_76582


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l765_76545

/-- A quadratic equation x^2 - 101x + k = 0 where k is an integer -/
def quadratic_equation (k : ℤ) (x : ℝ) : Prop :=
  x^2 - 101*x + k = 0

/-- Definition of a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The main theorem stating that no integer k exists such that both roots of the quadratic equation are prime -/
theorem no_prime_roots_for_quadratic :
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    is_prime p ∧ is_prime q ∧
    quadratic_equation k (p : ℝ) ∧ quadratic_equation k (q : ℝ) ∧
    p ≠ q :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l765_76545


namespace NUMINAMATH_CALUDE_undefined_expression_l765_76501

theorem undefined_expression (a : ℝ) : 
  ¬ (∃ x : ℝ, x = (a + 3) / (a^2 - 9)) ↔ a = -3 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_expression_l765_76501


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l765_76511

/-- Given a hyperbola with semi-major axis a and semi-minor axis b, 
    and a point P on its right branch satisfying |PF₁| = 4|PF₂|, 
    prove that the eccentricity e is in the range (1, 5/3] -/
theorem hyperbola_eccentricity_range (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) 
  (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)  -- P is on the hyperbola
  (h₄ : P.1 > 0)  -- P is on the right branch
  (h₅ : ‖P - F₁‖ = 4 * ‖P - F₂‖)  -- |PF₁| = 4|PF₂|
  (h₆ : F₁.1 < 0 ∧ F₂.1 > 0)  -- F₁ is left focus, F₂ is right focus
  (h₇ : ‖F₁ - F₂‖ = 2 * (a^2 + b^2).sqrt)  -- distance between foci
  : 1 < (a^2 + b^2).sqrt / a ∧ (a^2 + b^2).sqrt / a ≤ 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l765_76511


namespace NUMINAMATH_CALUDE_no_y_intercepts_l765_76505

/-- A parabola defined by x = 2y^2 - 3y + 7 -/
def parabola (y : ℝ) : ℝ := 2 * y^2 - 3 * y + 7

/-- A y-intercept occurs when x = 0 -/
def is_y_intercept (y : ℝ) : Prop := parabola y = 0

/-- The parabola has no y-intercepts -/
theorem no_y_intercepts : ¬∃ y : ℝ, is_y_intercept y := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l765_76505


namespace NUMINAMATH_CALUDE_prove_weekly_earnings_l765_76564

def total_earnings : ℕ := 133
def num_weeks : ℕ := 19
def weekly_earnings : ℚ := total_earnings / num_weeks

theorem prove_weekly_earnings : weekly_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_prove_weekly_earnings_l765_76564


namespace NUMINAMATH_CALUDE_prob_no_adjacent_same_five_people_l765_76554

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of faces on the standard die -/
def d : ℕ := 6

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ :=
  (d - 1)^(n - 1) * (d - 2) / d^n

theorem prob_no_adjacent_same_five_people (h : n = 5) :
  prob_no_adjacent_same = 625 / 1944 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_same_five_people_l765_76554


namespace NUMINAMATH_CALUDE_validMSetIs0And8_l765_76553

/-- The function f(x) = x^2 + mx - 2m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2*m - 1

/-- Predicate to check if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Predicate to check if all roots of f are integers -/
def hasOnlyIntegerRoots (m : ℝ) : Prop :=
  ∀ x : ℝ, f m x = 0 → isInteger x

/-- The set of m values for which f has only integer roots -/
def validMSet : Set ℝ := {m | hasOnlyIntegerRoots m}

/-- Theorem stating that the set of valid m values is {0, -8} -/
theorem validMSetIs0And8 : validMSet = {0, -8} := by sorry

end NUMINAMATH_CALUDE_validMSetIs0And8_l765_76553


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l765_76579

theorem banana_orange_equivalence (banana_value orange_value : ℚ) : 
  (3 / 4 : ℚ) * 12 * banana_value = 6 * orange_value →
  (2 / 3 : ℚ) * 9 * banana_value = 4 * orange_value := by
sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l765_76579


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l765_76537

theorem complex_root_magnitude (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a) 
  (h3 : a < (n + 1 : ℝ) / (n - 1 : ℝ)) 
  (h4 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l765_76537


namespace NUMINAMATH_CALUDE_claire_age_l765_76551

def age_problem (gabriel fiona ethan claire : ℕ) : Prop :=
  (gabriel = fiona - 2) ∧
  (fiona = ethan + 5) ∧
  (ethan = claire + 6) ∧
  (gabriel = 21)

theorem claire_age :
  ∀ gabriel fiona ethan claire : ℕ,
  age_problem gabriel fiona ethan claire →
  claire = 12 := by
sorry

end NUMINAMATH_CALUDE_claire_age_l765_76551


namespace NUMINAMATH_CALUDE_n_divided_by_six_l765_76527

theorem n_divided_by_six (n : ℕ) (h : n = 6^2024) : n / 6 = 6^2023 := by
  sorry

end NUMINAMATH_CALUDE_n_divided_by_six_l765_76527


namespace NUMINAMATH_CALUDE_triangle_foldable_to_2020_layers_l765_76581

/-- A triangle in a plane --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A folding method that transforms a triangle into a uniformly thick object --/
structure FoldingMethod where
  apply : Triangle → ℕ

/-- The theorem stating that any triangle can be folded into 2020 layers --/
theorem triangle_foldable_to_2020_layers :
  ∀ (t : Triangle), ∃ (f : FoldingMethod), f.apply t = 2020 :=
sorry

end NUMINAMATH_CALUDE_triangle_foldable_to_2020_layers_l765_76581


namespace NUMINAMATH_CALUDE_election_winner_percentage_l765_76594

theorem election_winner_percentage (total_votes winner_majority : ℕ) 
  (h_total : total_votes = 500) 
  (h_majority : winner_majority = 200) : 
  (((total_votes + winner_majority) / 2) / total_votes : ℚ) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l765_76594


namespace NUMINAMATH_CALUDE_negative_inequality_l765_76514

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l765_76514


namespace NUMINAMATH_CALUDE_apple_basket_problem_l765_76586

theorem apple_basket_problem (x : ℕ) : 
  (x / 2 - 2) - ((x / 2 - 2) / 2 - 3) = 24 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l765_76586


namespace NUMINAMATH_CALUDE_mode_is_97_l765_76506

/-- Represents a test score with its frequency -/
structure ScoreFrequency where
  score : Nat
  frequency : Nat

/-- Definition of the dataset from the stem-and-leaf plot -/
def testScores : List ScoreFrequency := [
  ⟨75, 2⟩, ⟨81, 2⟩, ⟨82, 3⟩, ⟨89, 2⟩, ⟨93, 1⟩, ⟨94, 2⟩, ⟨97, 4⟩,
  ⟨106, 1⟩, ⟨112, 2⟩, ⟨114, 3⟩, ⟨120, 1⟩
]

/-- Definition of mode: the score with the highest frequency -/
def isMode (s : ScoreFrequency) (scores : List ScoreFrequency) : Prop :=
  ∀ t ∈ scores, s.frequency ≥ t.frequency

/-- Theorem stating that 97 is the mode of the test scores -/
theorem mode_is_97 : ∃ s ∈ testScores, s.score = 97 ∧ isMode s testScores := by
  sorry

end NUMINAMATH_CALUDE_mode_is_97_l765_76506


namespace NUMINAMATH_CALUDE_sum_of_digits_18_to_21_l765_76539

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def sum_of_digits_range (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => sum_of_digits (a + i))

theorem sum_of_digits_18_to_21 :
  sum_of_digits_range 18 21 = 24 :=
by
  sorry

-- The following definition is provided as a condition from the problem
axiom sum_of_digits_0_to_99 : sum_of_digits_range 0 99 = 900

end NUMINAMATH_CALUDE_sum_of_digits_18_to_21_l765_76539


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l765_76569

theorem cone_vertex_angle (r l : ℝ) (h : r > 0) (h2 : l > 0) : 
  (π * r * l) / (π * r^2) = 2 → 2 * Real.arcsin (r / l) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l765_76569


namespace NUMINAMATH_CALUDE_convex_quadrilateral_area_is_120_l765_76588

def convex_quadrilateral_area (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧  -- areas are positive
  a < d ∧ b < d ∧ c < d ∧          -- fourth triangle has largest area
  a = 10 ∧ b = 20 ∧ c = 30 →       -- given areas
  a + b + c + d = 120              -- total area

theorem convex_quadrilateral_area_is_120 :
  ∀ a b c d : ℝ, convex_quadrilateral_area a b c d :=
by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_area_is_120_l765_76588


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l765_76566

theorem min_sum_of_squares (x y : ℝ) (h : x + y = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (a b : ℝ), a + b = 2 → x^2 + y^2 ≤ a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l765_76566


namespace NUMINAMATH_CALUDE_simple_interest_rate_l765_76547

/-- Given a principal amount and a simple interest rate, if the sum of money
becomes 7/6 of itself in 2 years, then the rate is 1/12 -/
theorem simple_interest_rate (P : ℝ) (R : ℝ) (P_pos : P > 0) :
  P * (1 + 2 * R) = (7 / 6) * P → R = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l765_76547


namespace NUMINAMATH_CALUDE_intersection_point_l765_76521

-- Define the system of equations
def system (x y m n : ℝ) : Prop :=
  2 * x + y = m ∧ x - y = n

-- Define the solution to the system
def solution : ℝ × ℝ := (-1, 3)

-- Define the lines
def line1 (x y m : ℝ) : Prop := y = -2 * x + m
def line2 (x y n : ℝ) : Prop := y = x - n

-- Theorem statement
theorem intersection_point :
  ∀ (m n : ℝ),
  system (solution.1) (solution.2) m n →
  ∃ (x y : ℝ), 
    line1 x y m ∧ 
    line2 x y n ∧ 
    x = solution.1 ∧ 
    y = solution.2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l765_76521


namespace NUMINAMATH_CALUDE_well_diameter_l765_76568

/-- The diameter of a circular well given its depth and volume -/
theorem well_diameter (depth : ℝ) (volume : ℝ) (h1 : depth = 14) (h2 : volume = 43.982297150257104) :
  let radius := Real.sqrt (volume / (Real.pi * depth))
  2 * radius = 2 := by sorry

end NUMINAMATH_CALUDE_well_diameter_l765_76568


namespace NUMINAMATH_CALUDE_inequality_solution_set_l765_76531

theorem inequality_solution_set (x : ℝ) : 
  (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l765_76531


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l765_76585

theorem geometric_sequence_properties (a₁ q : ℝ) (h_q : -1 < q ∧ q < 0) :
  let a : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  (∀ n : ℕ, a n * a (n + 1) < 0) ∧
  (∀ n : ℕ, |a n| > |a (n + 1)|) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l765_76585


namespace NUMINAMATH_CALUDE_complex_power_sum_l765_76515

/-- Given a complex number z such that z + 1/z = 2cos(5°), 
    prove that z^1000 + 1/z^1000 = -2cos(40°) -/
theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1000 + 1/z^1000 = -2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l765_76515


namespace NUMINAMATH_CALUDE_original_number_proof_l765_76500

theorem original_number_proof (x : ℝ) (h : 1 - 1/x = 5/2) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l765_76500


namespace NUMINAMATH_CALUDE_cone_surface_area_l765_76583

/-- A cone with base radius 1 and lateral surface that unfolds into a semicircle has a total surface area of 3π. -/
theorem cone_surface_area (cone : Real → Real → Real) 
  (h1 : cone 1 2 = 2 * Real.pi) -- Lateral surface area
  (h2 : cone 0 1 = Real.pi) -- Base area
  : cone 0 1 + cone 1 2 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l765_76583


namespace NUMINAMATH_CALUDE_handshake_problem_l765_76593

theorem handshake_problem (n : ℕ) (s : ℕ) : 
  n * (n - 1) / 2 + s = 159 → n = 18 ∧ s = 6 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l765_76593


namespace NUMINAMATH_CALUDE_savings_after_purchase_l765_76504

/-- Calculates the amount left in savings after buying sweaters, scarves, and mittens for a family --/
theorem savings_after_purchase (sweater_price scarf_price mitten_price : ℕ) 
  (family_members total_savings : ℕ) : 
  sweater_price = 35 →
  scarf_price = 25 →
  mitten_price = 15 →
  family_members = 10 →
  total_savings = 800 →
  total_savings - (sweater_price + scarf_price + mitten_price) * family_members = 50 := by
  sorry

end NUMINAMATH_CALUDE_savings_after_purchase_l765_76504


namespace NUMINAMATH_CALUDE_school_council_composition_l765_76529

theorem school_council_composition :
  -- Total number of classes
  ∀ (total_classes : ℕ),
  -- Number of students per council
  ∀ (students_per_council : ℕ),
  -- Number of classes with more girls than boys
  ∀ (classes_more_girls : ℕ),
  -- Number of boys and girls in Petya's class
  ∀ (petyas_class_boys petyas_class_girls : ℕ),
  -- Total number of boys and girls across all councils
  ∀ (total_boys total_girls : ℕ),

  total_classes = 20 →
  students_per_council = 5 →
  classes_more_girls = 15 →
  petyas_class_boys = 1 →
  petyas_class_girls = 4 →
  total_boys = total_girls →
  total_boys + total_girls = total_classes * students_per_council →

  -- Conclusion: In the remaining 4 classes, there are 19 boys and 1 girl
  ∃ (remaining_boys remaining_girls : ℕ),
    remaining_boys = 19 ∧
    remaining_girls = 1 ∧
    remaining_boys + remaining_girls = (total_classes - classes_more_girls - 1) * students_per_council :=
by sorry

end NUMINAMATH_CALUDE_school_council_composition_l765_76529


namespace NUMINAMATH_CALUDE_at_least_one_even_difference_l765_76536

theorem at_least_one_even_difference (n : ℕ) (a b : Fin (2*n+1) → ℤ) 
  (h : ∃ σ : Equiv.Perm (Fin (2*n+1)), ∀ i, b i = a (σ i)) :
  ∃ k : Fin (2*n+1), Even (a k - b k) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_even_difference_l765_76536


namespace NUMINAMATH_CALUDE_english_failure_percentage_l765_76556

/-- The percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 25

/-- The percentage of students who failed in both Hindi and English -/
def failed_both : ℝ := 27

/-- The percentage of students who passed in both subjects -/
def passed_both : ℝ := 54

/-- The percentage of students who failed in English -/
def failed_english : ℝ := 100 - passed_both - failed_hindi + failed_both

theorem english_failure_percentage :
  failed_english = 48 :=
sorry

end NUMINAMATH_CALUDE_english_failure_percentage_l765_76556


namespace NUMINAMATH_CALUDE_complement_union_A_B_l765_76567

def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}
def U : Set Int := Set.univ

theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : Int | ∃ k : Int, x = 3 * k} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l765_76567


namespace NUMINAMATH_CALUDE_correction_is_subtract_30x_l765_76509

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "half-dollar" => 50
  | "dollar" => 100
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculates the correction needed for mistaken coin counts -/
def correction_amount (x : ℕ) : ℤ :=
  (coin_value "dollar" - coin_value "half-dollar") * x -
  (coin_value "quarter" - coin_value "nickel") * x

theorem correction_is_subtract_30x (x : ℕ) :
  correction_amount x = -30 * x :=
sorry

end NUMINAMATH_CALUDE_correction_is_subtract_30x_l765_76509


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l765_76562

/-- Represents the number of students in a grade -/
def total_students : ℕ := 246

/-- Theorem: The system of equations {x + y = 246, y = 2x + 2} correctly represents
    the scenario where the total number of students is 246, and the number of boys (y)
    is 2 more than twice the number of girls (x). -/
theorem correct_system_of_equations (x y : ℕ) :
  x + y = total_students ∧ y = 2 * x + 2 →
  x + y = total_students ∧ y = 2 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l765_76562


namespace NUMINAMATH_CALUDE_equation_solution_l765_76534

theorem equation_solution : ∃! x : ℝ, (2 / 3) * x - 2 = 4 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l765_76534


namespace NUMINAMATH_CALUDE_odd_power_of_seven_plus_one_divisible_by_eight_l765_76546

theorem odd_power_of_seven_plus_one_divisible_by_eight (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, (7^n : ℤ) + 1 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_odd_power_of_seven_plus_one_divisible_by_eight_l765_76546


namespace NUMINAMATH_CALUDE_quadratic_function_conditions_l765_76512

/-- A quadratic function passing through (1, -4) with vertex at (-1, 0) -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 1

/-- Theorem stating that f(x) satisfies the required conditions -/
theorem quadratic_function_conditions :
  (f 1 = -4) ∧ (∀ x : ℝ, f x ≥ f (-1)) ∧ (f (-1) = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_conditions_l765_76512


namespace NUMINAMATH_CALUDE_age_multiplier_problem_l765_76592

theorem age_multiplier_problem (A : ℕ) (N : ℚ) : 
  A = 50 → (A + 5) * N - 5 * (A - 5) = A → N = 5 := by
sorry

end NUMINAMATH_CALUDE_age_multiplier_problem_l765_76592


namespace NUMINAMATH_CALUDE_rosys_age_l765_76550

theorem rosys_age (rosy_age : ℕ) : 
  (rosy_age + 12 + 4 = 2 * (rosy_age + 4)) → rosy_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_rosys_age_l765_76550


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l765_76532

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: For an arithmetic sequence, a_1 < a_3 iff the sequence is monotonically increasing -/
theorem arithmetic_sequence_increasing_iff_a1_lt_a3 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 < a 3 ↔ MonotonicallyIncreasing a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l765_76532


namespace NUMINAMATH_CALUDE_expression_equals_x_plus_one_l765_76599

theorem expression_equals_x_plus_one (x : ℝ) (h : x ≠ -1) :
  (x^2 + 2*x + 1) / (x + 1) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_x_plus_one_l765_76599


namespace NUMINAMATH_CALUDE_max_value_sqrt7_plus_2xy_l765_76580

theorem max_value_sqrt7_plus_2xy (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) 
  (h3 : x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32) : 
  ∃ (M : ℝ), M = 16 ∧ ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
  x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32 → Real.sqrt 7*(x + 2*y) + 2*x*y ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt7_plus_2xy_l765_76580


namespace NUMINAMATH_CALUDE_password_20_combinations_l765_76589

def password_combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem password_20_combinations :
  ∃ (k : ℕ), k ≤ 5 ∧ password_combinations 5 k = 20 ↔ k = 3 :=
sorry

end NUMINAMATH_CALUDE_password_20_combinations_l765_76589


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l765_76576

theorem sarahs_bowling_score (s g : ℕ) : 
  s = g + 50 ∧ (s + g) / 2 = 105 → s = 130 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l765_76576


namespace NUMINAMATH_CALUDE_slope_condition_l765_76530

/-- Given two points A(-3, 10) and B(5, y) in a coordinate plane, 
    if the slope of the line through A and B is -4/3, then y = -2/3. -/
theorem slope_condition (y : ℚ) : 
  let A : ℚ × ℚ := (-3, 10)
  let B : ℚ × ℚ := (5, y)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -4/3 → y = -2/3 := by
sorry

end NUMINAMATH_CALUDE_slope_condition_l765_76530


namespace NUMINAMATH_CALUDE_investment_time_period_l765_76517

/-- 
Given:
- principal: The sum invested (in rupees)
- rate_difference: The difference in interest rates (as a decimal)
- interest_difference: The additional interest earned due to the higher rate (in rupees)

Proves that the time period for which the sum is invested is 2 years.
-/
theorem investment_time_period 
  (principal : ℝ) 
  (rate_difference : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 14000)
  (h2 : rate_difference = 0.03)
  (h3 : interest_difference = 840) :
  principal * rate_difference * 2 = interest_difference := by
  sorry

end NUMINAMATH_CALUDE_investment_time_period_l765_76517


namespace NUMINAMATH_CALUDE_no_real_roots_l765_76575

theorem no_real_roots : ∀ x : ℝ, x^2 + 3*x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l765_76575


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l765_76540

-- Define the function f(x) = (x - 1)^2 - 2
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y := by
  sorry

-- Note: Set.Ici 1 represents the interval [1, +∞)

end NUMINAMATH_CALUDE_f_increasing_on_interval_l765_76540


namespace NUMINAMATH_CALUDE_hexagon_area_sum_l765_76516

/-- A regular hexagon with side length 3 -/
structure RegularHexagon where
  side_length : ℝ
  side_length_eq : side_length = 3

/-- The area of a regular hexagon can be expressed as √p + √q where p and q are positive integers -/
def hexagon_area (h : RegularHexagon) : ∃ (p q : ℕ+), Real.sqrt p.val + Real.sqrt q.val = (3 * Real.sqrt 3 / 2) * h.side_length ^ 2 :=
  sorry

/-- The sum of p and q is 297 -/
theorem hexagon_area_sum (h : RegularHexagon) : 
  ∃ (p q : ℕ+), (Real.sqrt p.val + Real.sqrt q.val = (3 * Real.sqrt 3 / 2) * h.side_length ^ 2) ∧ (p.val + q.val = 297) :=
  sorry

end NUMINAMATH_CALUDE_hexagon_area_sum_l765_76516


namespace NUMINAMATH_CALUDE_equation_representation_l765_76525

/-- Given that a number is 5 more than three times a and equals 9, 
    prove that the equation is 3a + 5 = 9 -/
theorem equation_representation (a : ℝ) : 
  (3 * a + 5 = 9) ↔ (∃ x, x = 3 * a + 5 ∧ x = 9) := by sorry

end NUMINAMATH_CALUDE_equation_representation_l765_76525


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l765_76557

/-- The minimum number of additional coins needed for distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_coins_needed := (num_friends * (num_friends + 1)) / 2
  if total_coins_needed > initial_coins then
    total_coins_needed - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed for Alex's distribution. -/
theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 80) :
  min_additional_coins num_friends initial_coins = 40 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l765_76557


namespace NUMINAMATH_CALUDE_candle_burning_l765_76574

/-- Candle burning problem -/
theorem candle_burning (h₀ : ℕ) (burn_rate : ℕ → ℝ) (T : ℝ) : 
  (h₀ = 150) →
  (∀ k, burn_rate k = 15 * k) →
  (T = (15 : ℝ) * (h₀ * (h₀ + 1) / 2)) →
  (∃ m : ℕ, 
    (7.5 * m * (m + 1) ≤ T / 2) ∧ 
    (T / 2 < 7.5 * (m + 1) * (m + 2)) ∧
    (h₀ - m = 45)) :=
by sorry

end NUMINAMATH_CALUDE_candle_burning_l765_76574


namespace NUMINAMATH_CALUDE_pattern_3_7_verify_other_pairs_l765_76541

/-- The pattern function that transforms two numbers according to the given rule -/
def pattern (a b : ℕ) : ℕ := (a + b) * a - a

/-- The theorem stating that the pattern applied to (3, 7) results in 27 -/
theorem pattern_3_7 : pattern 3 7 = 27 := by
  sorry

/-- Verification of other given pairs -/
theorem verify_other_pairs :
  pattern 2 3 = 8 ∧
  pattern 4 5 = 32 ∧
  pattern 5 8 = 60 ∧
  pattern 6 7 = 72 ∧
  pattern 7 8 = 98 := by
  sorry

end NUMINAMATH_CALUDE_pattern_3_7_verify_other_pairs_l765_76541


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l765_76597

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The condition that a = 3 is sufficient for the lines to be parallel -/
theorem sufficient_condition (a : ℝ) :
  a = 3 → are_parallel 2 a 1 (a - 1) 3 (-2) :=
by sorry

/-- The condition that a = 3 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ 3 ∧ are_parallel 2 a 1 (a - 1) 3 (-2) :=
by sorry

/-- The main theorem stating that a = 3 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = 3 → are_parallel 2 a 1 (a - 1) 3 (-2)) ∧
  (∃ a : ℝ, a ≠ 3 ∧ are_parallel 2 a 1 (a - 1) 3 (-2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l765_76597


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l765_76595

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel α β → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l765_76595


namespace NUMINAMATH_CALUDE_robin_cupcakes_l765_76587

/-- Calculates the total number of cupcakes Robin has after baking and selling. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (additional : ℕ) : ℕ :=
  initial - sold + additional

/-- Theorem stating that Robin has 59 cupcakes in total. -/
theorem robin_cupcakes : total_cupcakes 42 22 39 = 59 := by
  sorry

end NUMINAMATH_CALUDE_robin_cupcakes_l765_76587


namespace NUMINAMATH_CALUDE_painted_cube_probability_l765_76519

/-- Represents a rectangular prism with painted faces -/
structure PaintedPrism where
  length : ℕ
  width : ℕ
  height : ℕ
  painted_face1 : ℕ × ℕ
  painted_face2 : ℕ × ℕ

/-- Calculates the total number of unit cubes in the prism -/
def total_cubes (p : PaintedPrism) : ℕ :=
  p.length * p.width * p.height

/-- Calculates the number of cubes with exactly one painted face -/
def cubes_with_one_painted_face (p : PaintedPrism) : ℕ :=
  (p.painted_face1.1 - 2) * (p.painted_face1.2 - 2) +
  (p.painted_face2.1 - 2) * (p.painted_face2.2 - 2) + 2

/-- Calculates the number of cubes with no painted faces -/
def cubes_with_no_painted_faces (p : PaintedPrism) : ℕ :=
  total_cubes p - (p.painted_face1.1 * p.painted_face1.2 +
                   p.painted_face2.1 * p.painted_face2.2 -
                   (p.painted_face1.1 + p.painted_face2.1))

/-- The main theorem to be proved -/
theorem painted_cube_probability (p : PaintedPrism)
  (h1 : p.length = 4)
  (h2 : p.width = 3)
  (h3 : p.height = 3)
  (h4 : p.painted_face1 = (4, 3))
  (h5 : p.painted_face2 = (3, 3)) :
  (cubes_with_one_painted_face p * cubes_with_no_painted_faces p : ℚ) /
  (total_cubes p * (total_cubes p - 1) / 2) = 221 / 630 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l765_76519


namespace NUMINAMATH_CALUDE_least_number_divisibility_l765_76571

theorem least_number_divisibility (n : ℕ) : n = 215988 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 48 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 64 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 72 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 108 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 12) = 125 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    (n + 12) = 48 * k₁ ∧
    (n + 12) = 64 * k₂ ∧
    (n + 12) = 72 * k₃ ∧
    (n + 12) = 108 * k₄ ∧
    (n + 12) = 125 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l765_76571


namespace NUMINAMATH_CALUDE_function_equality_implies_m_zero_l765_76510

/-- Given two functions f and g, prove that m = 0 when 3f(3) = 2g(3) -/
theorem function_equality_implies_m_zero (m : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 3*x + 2*m
  let g := fun (x : ℝ) => 2*x^2 - 6*x + 5*m
  3 * f 3 = 2 * g 3 → m = 0 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_zero_l765_76510


namespace NUMINAMATH_CALUDE_expected_sixes_two_dice_l765_76598

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling a 6 on a single die -/
def prob_six : ℚ := 1 / num_sides

/-- The expected number of 6's when rolling two dice -/
def expected_sixes : ℚ := 1 / 4

/-- Theorem stating that the expected number of 6's when rolling two eight-sided dice is 1/4 -/
theorem expected_sixes_two_dice : 
  expected_sixes = 2 * prob_six := by sorry

end NUMINAMATH_CALUDE_expected_sixes_two_dice_l765_76598


namespace NUMINAMATH_CALUDE_pablo_share_fraction_l765_76522

/-- Represents the number of eggs each person has -/
structure EggDistribution :=
  (mia : ℕ)
  (sofia : ℕ)
  (pablo : ℕ)
  (juan : ℕ)

/-- The initial distribution of eggs -/
def initial_distribution (m : ℕ) : EggDistribution :=
  { mia := m
  , sofia := 3 * m
  , pablo := 12 * m
  , juan := 5 }

/-- The fraction of eggs Pablo gives to Sofia -/
def pablo_to_sofia_fraction (m : ℕ) : ℚ :=
  (4 * m + 5 : ℚ) / (48 * m : ℚ)

theorem pablo_share_fraction (m : ℕ) :
  let init := initial_distribution m
  let total := init.mia + init.sofia + init.pablo + init.juan
  let equal_share := total / 4
  let sofia_needs := equal_share - init.sofia
  sofia_needs / init.pablo = pablo_to_sofia_fraction m := by
  sorry

end NUMINAMATH_CALUDE_pablo_share_fraction_l765_76522


namespace NUMINAMATH_CALUDE_can_display_properties_l765_76503

/-- Represents a triangular display of cans. -/
structure CanDisplay where
  totalCans : ℕ
  canWeight : ℕ

/-- Calculates the number of rows in the display. -/
def numberOfRows (d : CanDisplay) : ℕ :=
  Nat.sqrt d.totalCans

/-- Calculates the total weight of the display in kg. -/
def totalWeight (d : CanDisplay) : ℕ :=
  d.totalCans * d.canWeight

/-- Theorem stating the properties of the specific can display. -/
theorem can_display_properties (d : CanDisplay) 
  (h1 : d.totalCans = 225)
  (h2 : d.canWeight = 5) :
  numberOfRows d = 15 ∧ totalWeight d = 1125 := by
  sorry

end NUMINAMATH_CALUDE_can_display_properties_l765_76503


namespace NUMINAMATH_CALUDE_parabola_shift_down_2_l765_76533

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Shifts a parabola vertically -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

theorem parabola_shift_down_2 :
  let original := Parabola.mk 2 4
  let shifted := shift_parabola original (-2)
  shifted = Parabola.mk 2 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_down_2_l765_76533


namespace NUMINAMATH_CALUDE_special_function_property_l765_76508

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 ≠ 0) :
  (f 3 - f 1) / f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l765_76508


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l765_76526

theorem inverse_variation_cube (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → 3 * (y x) = k / (x^3)) →  -- 3y varies inversely as the cube of x
  y 3 = 27 →                              -- y = 27 when x = 3
  y 9 = 1 :=                              -- y = 1 when x = 9
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l765_76526


namespace NUMINAMATH_CALUDE_erased_number_l765_76548

theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) 
  (h2 : 8 * a - b = 1703) : a + b = 214 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_l765_76548


namespace NUMINAMATH_CALUDE_total_revenue_over_three_days_l765_76543

-- Define pie types
inductive PieType
  | Apple
  | Blueberry
  | Cherry

-- Define a structure for daily sales data
structure DailySales where
  apple_price : ℕ
  blueberry_price : ℕ
  cherry_price : ℕ
  apple_sold : ℕ
  blueberry_sold : ℕ
  cherry_sold : ℕ

def slices_per_pie : ℕ := 6

def day1_sales : DailySales := {
  apple_price := 5,
  blueberry_price := 6,
  cherry_price := 7,
  apple_sold := 12,
  blueberry_sold := 8,
  cherry_sold := 10
}

def day2_sales : DailySales := {
  apple_price := 6,
  blueberry_price := 7,
  cherry_price := 8,
  apple_sold := 15,
  blueberry_sold := 10,
  cherry_sold := 14
}

def day3_sales : DailySales := {
  apple_price := 4,
  blueberry_price := 7,
  cherry_price := 9,
  apple_sold := 18,
  blueberry_sold := 7,
  cherry_sold := 13
}

def calculate_daily_revenue (sales : DailySales) : ℕ :=
  sales.apple_price * slices_per_pie * sales.apple_sold +
  sales.blueberry_price * slices_per_pie * sales.blueberry_sold +
  sales.cherry_price * slices_per_pie * sales.cherry_sold

theorem total_revenue_over_three_days :
  calculate_daily_revenue day1_sales +
  calculate_daily_revenue day2_sales +
  calculate_daily_revenue day3_sales = 4128 := by
  sorry


end NUMINAMATH_CALUDE_total_revenue_over_three_days_l765_76543


namespace NUMINAMATH_CALUDE_line_contains_diameter_l765_76559

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 8 = 0, 
    prove that the line 2x + y + 1 = 0 contains a diameter of the circle -/
theorem line_contains_diameter (x y : ℝ) :
  (x^2 + y^2 - 2*x + 6*y + 8 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 2*x₁ + 6*y₁ + 8 = 0) ∧
    (x₂^2 + y₂^2 - 2*x₂ + 6*y₂ + 8 = 0) ∧
    (2*x₁ + y₁ + 1 = 0) ∧
    (2*x₂ + y₂ + 1 = 0) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = (2*1)^2 + (2*(-3))^2)) :=
by sorry

end NUMINAMATH_CALUDE_line_contains_diameter_l765_76559


namespace NUMINAMATH_CALUDE_remainder_of_a_l765_76558

theorem remainder_of_a (a : ℤ) :
  (a^100 % 73 = 2) → (a^101 % 73 = 69) → (a % 73 = 71) := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_a_l765_76558


namespace NUMINAMATH_CALUDE_probability_of_three_in_three_eighths_l765_76523

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

theorem probability_of_three_in_three_eighths :
  let digits := decimal_representation 3 8
  (digits.count 3) / (digits.length : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_in_three_eighths_l765_76523


namespace NUMINAMATH_CALUDE_point_symmetry_y_axis_l765_76524

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetric_y_axis (A B : Point) : Prop :=
  A.x = -B.x ∧ A.y = B.y

theorem point_symmetry_y_axis : 
  let A : Point := ⟨-1, 8⟩
  ∀ B : Point, symmetric_y_axis A B → B = ⟨1, 8⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_y_axis_l765_76524


namespace NUMINAMATH_CALUDE_fair_entrance_fee_l765_76584

/-- Represents the entrance fee structure and ride costs at a fair --/
structure FairPrices where
  under18Fee : ℝ
  over18Fee : ℝ
  rideCost : ℝ
  under18Fee_pos : 0 < under18Fee
  over18Fee_eq : over18Fee = 1.2 * under18Fee
  rideCost_eq : rideCost = 0.5

/-- Calculates the total cost for a group at the fair --/
def totalCost (prices : FairPrices) (numUnder18 : ℕ) (numOver18 : ℕ) (totalRides : ℕ) : ℝ :=
  numUnder18 * prices.under18Fee + numOver18 * prices.over18Fee + totalRides * prices.rideCost

/-- The main theorem stating the entrance fee for persons under 18 --/
theorem fair_entrance_fee :
  ∃ (prices : FairPrices), totalCost prices 2 1 9 = 20.5 ∧ prices.under18Fee = 5 := by
  sorry

end NUMINAMATH_CALUDE_fair_entrance_fee_l765_76584


namespace NUMINAMATH_CALUDE_existence_of_k_values_l765_76518

/-- Represents a triple of numbers -/
structure Triple :=
  (a b c : ℤ)

/-- Checks if the sums of powers with exponents 1, 2, and 3 are equal for two triples -/
def sumPowersEqual (t1 t2 : Triple) : Prop :=
  ∀ m : ℕ, m ≤ 3 → t1.a^m + t1.b^m + t1.c^m = t2.a^m + t2.b^m + t2.c^m

/-- Represents the 6-member group formed from two triples -/
def sixMemberGroup (t1 t2 : Triple) (k : ℤ) : Finset ℤ :=
  {t1.a, t1.b, t1.c, t2.a + k, t2.b + k, t2.c + k}

/-- Checks if a 6-member group can be simplified to a 4-member group -/
def simplifiesToFour (t1 t2 : Triple) (k : ℤ) : Prop :=
  (sixMemberGroup t1 t2 k).card = 4

/-- Checks if a 6-member group can be simplified to a 5-member group but not further -/
def simplifiesToFiveOnly (t1 t2 : Triple) (k : ℤ) : Prop :=
  (sixMemberGroup t1 t2 k).card = 5

/-- The main theorem to be proved -/
theorem existence_of_k_values 
  (I II III IV : Triple)
  (h1 : sumPowersEqual I II)
  (h2 : sumPowersEqual III IV) :
  ∃ k : ℤ, 
    (simplifiesToFour I II k ∨ simplifiesToFour II I k) ∧
    (simplifiesToFiveOnly III IV k ∨ simplifiesToFiveOnly IV III k) :=
sorry

end NUMINAMATH_CALUDE_existence_of_k_values_l765_76518


namespace NUMINAMATH_CALUDE_bills_theorem_l765_76542

/-- Represents the water and electricity bills for DingDing's family -/
structure Bills where
  may_water : ℝ
  may_total : ℝ
  june_water_increase : ℝ
  june_electricity_increase : ℝ

/-- Calculates the total bill for June -/
def june_total (b : Bills) : ℝ :=
  b.may_water * (1 + b.june_water_increase) + 
  (b.may_total - b.may_water) * (1 + b.june_electricity_increase)

/-- Calculates the total bill for May and June -/
def may_june_total (b : Bills) : ℝ :=
  b.may_total + june_total b

/-- Theorem stating the properties of the bills -/
theorem bills_theorem (b : Bills) 
  (h1 : b.may_total = 140)
  (h2 : b.june_water_increase = 0.1)
  (h3 : b.june_electricity_increase = 0.2) :
  june_total b = -0.1 * b.may_water + 168 ∧
  may_june_total b = 304 ↔ b.may_water = 40 := by
  sorry

end NUMINAMATH_CALUDE_bills_theorem_l765_76542


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l765_76561

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_pattern :
  ∃ (n : ℕ) (a : ℕ),
    n > 0 ∧
    a ≥ 1 ∧ a < 30 ∧
    consecutive_pair a (a + 1) ∧
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 30 ∧ i ≠ a ∧ i ≠ (a + 1) → is_divisible n i) ∧
    ¬(is_divisible n a) ∧
    ¬(is_divisible n (a + 1)) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ (b : ℕ),
        b ≥ 1 ∧ b < 30 ∧
        consecutive_pair b (b + 1) ∧
        (∀ i : ℕ, 1 ≤ i ∧ i ≤ 30 ∧ i ≠ b ∧ i ≠ (b + 1) → is_divisible m i) ∧
        ¬(is_divisible m b) ∧
        ¬(is_divisible m (b + 1)))) ∧
    n = 12252240 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l765_76561


namespace NUMINAMATH_CALUDE_train_length_calculation_l765_76560

/-- The length of a train given its speed, the speed of a man it's passing, and the time it takes to cross the man completely. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 53.99568034557235 →
  ∃ (train_length : ℝ), abs (train_length - 900) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l765_76560


namespace NUMINAMATH_CALUDE_best_estimate_and_error_prob_l765_76535

/-- Represents a measurement with an error margin and probability --/
structure Measurement where
  value : ℝ
  error_margin : ℝ
  error_prob : ℝ

/-- The problem setup --/
def river_length_problem (gsa awra : Measurement) : Prop :=
  gsa.value = 402 ∧
  gsa.error_margin = 0.5 ∧
  gsa.error_prob = 0.04 ∧
  awra.value = 403 ∧
  awra.error_margin = 0.5 ∧
  awra.error_prob = 0.04

/-- The theorem to prove --/
theorem best_estimate_and_error_prob
  (gsa awra : Measurement)
  (h : river_length_problem gsa awra) :
  ∃ (estimate error_prob : ℝ),
    estimate = 402.5 ∧
    error_prob = 0.04 :=
  sorry

end NUMINAMATH_CALUDE_best_estimate_and_error_prob_l765_76535


namespace NUMINAMATH_CALUDE_range_of_m_l765_76502

def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m : 
  ∀ m : ℝ, (m > 0 ∧ 
    (∀ x : ℝ, p x → q x m) ∧ 
    (∃ x : ℝ, ¬(p x) ∧ q x m)) ↔ 
  m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l765_76502


namespace NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l765_76570

/-- Calculates the time needed to read a book given reading speed and book properties -/
theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : ℕ :=
  let sentences_per_page := paragraphs_per_page * sentences_per_paragraph
  let total_sentences := sentences_per_page * total_pages
  total_sentences / reading_speed

/-- Proves that Gwendolyn will take 225 hours to read the book -/
theorem gwendolyn_reading_time : 
  reading_time_calculation 200 30 15 100 = 225 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l765_76570


namespace NUMINAMATH_CALUDE_greg_read_more_than_brad_l765_76577

/-- Calculates the difference in pages read between Greg and Brad --/
def pages_difference : ℕ :=
  let greg_week1 := 7 * 18
  let greg_week2_3 := 14 * 22
  let greg_total := greg_week1 + greg_week2_3
  let brad_days1_5 := 5 * 26
  let brad_days6_17 := 12 * 20
  let brad_total := brad_days1_5 + brad_days6_17
  greg_total - brad_total

/-- The total number of pages both Greg and Brad need to read --/
def total_pages : ℕ := 800

/-- Theorem stating the difference in pages read between Greg and Brad --/
theorem greg_read_more_than_brad : pages_difference = 64 ∧ greg_total + brad_total = total_pages :=
  sorry

end NUMINAMATH_CALUDE_greg_read_more_than_brad_l765_76577


namespace NUMINAMATH_CALUDE_f_extrema_l765_76520

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + 1

theorem f_extrema :
  let I : Set ℝ := Set.Icc 0 (Real.pi / 2)
  (∀ x ∈ I, f x ≥ 1) ∧
  (∀ x ∈ I, f x ≤ 3 + Real.sqrt 3) ∧
  (∃ x ∈ I, f x = 1) ∧
  (∃ x ∈ I, f x = 3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l765_76520


namespace NUMINAMATH_CALUDE_dress_pocket_ratio_l765_76565

/-- Proves that the ratio of dresses with pockets to the total number of dresses is 1:2 --/
theorem dress_pocket_ratio :
  ∀ (total_dresses : ℕ) (dresses_with_pockets : ℕ) (total_pockets : ℕ),
    total_dresses = 24 →
    total_pockets = 32 →
    dresses_with_pockets * 2 = total_dresses * 1 →
    dresses_with_pockets * 2 + dresses_with_pockets * 3 = total_pockets * 3 →
    (dresses_with_pockets : ℚ) / total_dresses = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dress_pocket_ratio_l765_76565
