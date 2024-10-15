import Mathlib

namespace NUMINAMATH_CALUDE_three_card_draw_probability_l1379_137996

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of diamonds in a standard deck -/
def NumDiamonds : ℕ := 13

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Probability of drawing an Ace as the first card, a diamond as the second card, 
    and a Jack as the third card from a standard 52-card deck -/
theorem three_card_draw_probability : 
  (NumAces / StandardDeck) * (NumDiamonds / (StandardDeck - 1)) * (NumJacks / (StandardDeck - 2)) = 1 / 650 :=
by sorry

end NUMINAMATH_CALUDE_three_card_draw_probability_l1379_137996


namespace NUMINAMATH_CALUDE_f_intersects_axes_twice_l1379_137965

/-- The quadratic function f(x) = x^2 + 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The number of intersection points between f(x) and the coordinate axes -/
def num_intersections : ℕ := 2

/-- Theorem stating that f(x) intersects the coordinate axes at exactly two points -/
theorem f_intersects_axes_twice :
  (∃! x : ℝ, f x = 0) ∧ (∃! y : ℝ, f 0 = y) ∧ num_intersections = 2 :=
sorry

end NUMINAMATH_CALUDE_f_intersects_axes_twice_l1379_137965


namespace NUMINAMATH_CALUDE_slow_clock_theorem_l1379_137972

/-- Represents a clock with a specific overlap time between its minute and hour hands. -/
structure Clock where
  overlap_time : ℕ  -- Time in minutes between each overlap of minute and hour hands

/-- The number of overlaps in a 24-hour period for any clock -/
def num_overlaps : ℕ := 22

/-- Calculates the length of a 24-hour period for a given clock in minutes -/
def period_length (c : Clock) : ℕ :=
  num_overlaps * c.overlap_time

/-- The length of a standard 24-hour period in minutes -/
def standard_period : ℕ := 24 * 60

/-- Theorem stating that a clock with 66-minute overlaps is 12 minutes slower over 24 hours -/
theorem slow_clock_theorem (c : Clock) (h : c.overlap_time = 66) :
  period_length c - standard_period = 12 := by
  sorry


end NUMINAMATH_CALUDE_slow_clock_theorem_l1379_137972


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1379_137903

/-- Given that x is inversely proportional to y, prove that y₁/y₂ = 5/3 when x₁/x₂ = 3/5 -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_prop : ∃ (k : ℝ), ∀ (x y : ℝ), x * y = k)
  (h_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1379_137903


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l1379_137936

/-- The number of distinct arrangements of the letters in "BALLOON" -/
def balloon_arrangements : ℕ := 1260

/-- The total number of letters in "BALLOON" -/
def total_letters : ℕ := 7

/-- The number of times 'L' appears in "BALLOON" -/
def l_count : ℕ := 2

/-- The number of times 'O' appears in "BALLOON" -/
def o_count : ℕ := 2

/-- Theorem stating that the number of distinct arrangements of the letters in "BALLOON" is 1260 -/
theorem balloon_arrangements_count :
  balloon_arrangements = (Nat.factorial total_letters) / (Nat.factorial l_count * Nat.factorial o_count) :=
sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l1379_137936


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1379_137992

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  commonDiff : ℕ
  numTerms : ℕ
  sumOddTerms : ℕ
  sumEvenTerms : ℕ
  isEven : Even numTerms
  diffIs2 : commonDiff = 2

/-- Theorem stating the number of terms in the sequence with given conditions -/
theorem arithmetic_sequence_terms
  (seq : ArithmeticSequence)
  (h1 : seq.sumOddTerms = 15)
  (h2 : seq.sumEvenTerms = 35) :
  seq.numTerms = 20 := by
  sorry

#check arithmetic_sequence_terms

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1379_137992


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1379_137927

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) :
  x^2 + (1 / x)^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1379_137927


namespace NUMINAMATH_CALUDE_solve_for_a_l1379_137934

theorem solve_for_a : ∃ a : ℝ, (2 : ℝ) - a * (1 : ℝ) = 3 ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1379_137934


namespace NUMINAMATH_CALUDE_quadratic_roots_same_sign_a_range_l1379_137953

theorem quadratic_roots_same_sign_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) →
  0 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_same_sign_a_range_l1379_137953


namespace NUMINAMATH_CALUDE_min_k_value_l1379_137909

def is_valid (n k : ℕ) : Prop :=
  ∀ i ∈ Finset.range (k - 1), n % (i + 2) = i + 1

theorem min_k_value :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (n : ℕ), n > 2000 ∧ n < 3000 ∧ is_valid n k ∧
    ∀ (m : ℕ), m < n → ¬(is_valid m k)) ∧
  ∀ (j : ℕ), j < k →
    ¬(∃ (n : ℕ), n > 2000 ∧ n < 3000 ∧ is_valid n j ∧
      ∀ (m : ℕ), m < n → ¬(is_valid m j)) ∧
  k = 9 :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l1379_137909


namespace NUMINAMATH_CALUDE_bread_loaves_from_flour_l1379_137985

/-- Given 5 cups of flour and requiring 2.5 cups per loaf, prove that 2 loaves can be baked. -/
theorem bread_loaves_from_flour (total_flour : ℝ) (flour_per_loaf : ℝ) (h1 : total_flour = 5) (h2 : flour_per_loaf = 2.5) :
  total_flour / flour_per_loaf = 2 := by
sorry

end NUMINAMATH_CALUDE_bread_loaves_from_flour_l1379_137985


namespace NUMINAMATH_CALUDE_ferry_travel_time_difference_l1379_137987

/-- Represents the properties of a ferry --/
structure Ferry where
  baseSpeed : ℝ  -- Speed without current in km/h
  currentEffect : ℝ  -- Speed reduction due to current in km/h
  travelTime : ℝ  -- Travel time in hours
  routeLength : ℝ  -- Route length in km

/-- The problem setup --/
def ferryProblem : Prop := ∃ (p q : Ferry),
  -- Ferry p properties
  p.baseSpeed = 6 ∧
  p.currentEffect = 1 ∧
  p.travelTime = 3 ∧
  
  -- Ferry q properties
  q.baseSpeed = p.baseSpeed + 3 ∧
  q.currentEffect = p.currentEffect / 2 ∧
  q.routeLength = 2 * p.routeLength ∧
  
  -- Calculate effective speeds
  let pEffectiveSpeed := p.baseSpeed - p.currentEffect
  let qEffectiveSpeed := q.baseSpeed - q.currentEffect
  
  -- Calculate route lengths
  p.routeLength = pEffectiveSpeed * p.travelTime ∧
  
  -- Calculate q's travel time
  q.travelTime = q.routeLength / qEffectiveSpeed ∧
  
  -- The difference in travel time is approximately 0.5294 hours
  abs (q.travelTime - p.travelTime - 0.5294) < 0.0001

/-- The theorem to be proved --/
theorem ferry_travel_time_difference : ferryProblem := by
  sorry

end NUMINAMATH_CALUDE_ferry_travel_time_difference_l1379_137987


namespace NUMINAMATH_CALUDE_star_property_l1379_137919

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) => (a + d, b - c)

theorem star_property : 
  ∃ (y : ℤ), star (3, y) (4, 2) = star (4, 5) (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_star_property_l1379_137919


namespace NUMINAMATH_CALUDE_factor_4t_squared_minus_100_l1379_137926

theorem factor_4t_squared_minus_100 (t : ℝ) : 4 * t^2 - 100 = (2*t - 10) * (2*t + 10) := by
  sorry

end NUMINAMATH_CALUDE_factor_4t_squared_minus_100_l1379_137926


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1379_137905

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (1 + x) + 2 * f (1 - x) = 6 - 1 / x) :
  f (Real.sqrt 2) = 3 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ x > 0, g x = (m^2 - 2*m - 2) * x^(m^2 + 3*m + 2))
  (h2 : StrictMono g)
  (h3 : ∀ x, g (2*x - 1) ≥ 1) :
  ∀ x, x ≤ 0 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1379_137905


namespace NUMINAMATH_CALUDE_complex_cube_root_of_unity_sum_l1379_137981

theorem complex_cube_root_of_unity_sum (ω : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω^2 + ω + 1 = 0 →
  ((-1 + Complex.I * Real.sqrt 3) / 2)^4 + ((-1 - Complex.I * Real.sqrt 3) / 2)^4 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_of_unity_sum_l1379_137981


namespace NUMINAMATH_CALUDE_coffee_package_size_l1379_137940

/-- Proves that the size of the larger coffee package is 10 ounces given the conditions -/
theorem coffee_package_size (total_coffee : ℕ) (larger_package_count : ℕ) 
  (small_package_size : ℕ) (small_package_count : ℕ) (larger_package_size : ℕ) :
  total_coffee = 115 ∧ 
  larger_package_count = 7 ∧
  small_package_size = 5 ∧
  small_package_count = larger_package_count + 2 ∧
  total_coffee = larger_package_count * larger_package_size + small_package_count * small_package_size →
  larger_package_size = 10 := by
  sorry

#check coffee_package_size

end NUMINAMATH_CALUDE_coffee_package_size_l1379_137940


namespace NUMINAMATH_CALUDE_ripe_oranges_harvested_per_day_l1379_137913

/-- The number of days of harvest -/
def harvest_days : ℕ := 25

/-- The total number of sacks of ripe oranges after the harvest period -/
def total_ripe_oranges : ℕ := 2050

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := total_ripe_oranges / harvest_days

theorem ripe_oranges_harvested_per_day :
  ripe_oranges_per_day = 82 := by
  sorry

end NUMINAMATH_CALUDE_ripe_oranges_harvested_per_day_l1379_137913


namespace NUMINAMATH_CALUDE_second_group_size_l1379_137952

theorem second_group_size (n : ℕ) : 
  (30 : ℝ) * 20 + n * 30 = (30 + n) * 24 → n = 20 := by sorry

end NUMINAMATH_CALUDE_second_group_size_l1379_137952


namespace NUMINAMATH_CALUDE_least_multiple_72_112_l1379_137907

theorem least_multiple_72_112 : 
  (∀ k : ℕ, k > 0 ∧ k < 14 → ¬(112 ∣ 72 * k)) ∧ (112 ∣ 72 * 14) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_72_112_l1379_137907


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1379_137933

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1379_137933


namespace NUMINAMATH_CALUDE_three_similar_points_l1379_137906

/-- Right trapezoid ABCD with given side lengths -/
structure RightTrapezoid where
  AB : ℝ
  AD : ℝ
  BC : ℝ
  ab_positive : AB > 0
  ad_positive : AD > 0
  bc_positive : BC > 0

/-- Point P on side AB of the trapezoid -/
def PointP (t : RightTrapezoid) := { x : ℝ // 0 ≤ x ∧ x ≤ t.AB }

/-- Condition for triangle PAD to be similar to triangle PBC -/
def IsSimilar (t : RightTrapezoid) (p : PointP t) : Prop :=
  p.val / (t.AB - p.val) = t.AD / t.BC ∨ p.val / t.BC = t.AD / (t.AB - p.val)

/-- The main theorem stating that there are exactly 3 points P satisfying the similarity condition -/
theorem three_similar_points (t : RightTrapezoid) 
  (h1 : t.AB = 7) (h2 : t.AD = 2) (h3 : t.BC = 3) : 
  ∃! (s : Finset (PointP t)), s.card = 3 ∧ ∀ p ∈ s, IsSimilar t p := by
  sorry

end NUMINAMATH_CALUDE_three_similar_points_l1379_137906


namespace NUMINAMATH_CALUDE_max_sum_distance_from_line_max_sum_distance_from_line_tight_l1379_137978

theorem max_sum_distance_from_line (x₁ y₁ x₂ y₂ : ℝ) :
  x₁^2 + y₁^2 = 1 →
  x₂^2 + y₂^2 = 1 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 →
  |x₁ + y₁ - 1| + |x₂ + y₂ - 1| ≤ 2 + Real.sqrt 6 :=
by sorry

theorem max_sum_distance_from_line_tight :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁^2 + y₁^2 = 1 ∧
    x₂^2 + y₂^2 = 1 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 ∧
    |x₁ + y₁ - 1| + |x₂ + y₂ - 1| = 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_distance_from_line_max_sum_distance_from_line_tight_l1379_137978


namespace NUMINAMATH_CALUDE_julian_needs_more_legos_l1379_137964

/-- The number of legos Julian has -/
def julianLegos : ℕ := 400

/-- The number of legos required for one airplane model -/
def legosPerModel : ℕ := 240

/-- The number of airplane models Julian wants to make -/
def numModels : ℕ := 2

/-- The number of additional legos Julian needs -/
def additionalLegosNeeded : ℕ := 80

theorem julian_needs_more_legos : 
  julianLegos + additionalLegosNeeded = legosPerModel * numModels := by
  sorry

end NUMINAMATH_CALUDE_julian_needs_more_legos_l1379_137964


namespace NUMINAMATH_CALUDE_factor_sum_l1379_137968

theorem factor_sum (P Q : ℚ) : 
  (∃ b c : ℚ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^3 + Q*X^2 + 45*X - 14) →
  P + Q = 260/7 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l1379_137968


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_plus_abs_inequality_l1379_137966

def f (x : ℝ) := |3 * x + 1| - |x - 4|

theorem f_inequality_solution (x : ℝ) :
  f x < 0 ↔ -5/2 < x ∧ x < 3/4 := by sorry

theorem f_plus_abs_inequality (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) ↔ m < 15 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_plus_abs_inequality_l1379_137966


namespace NUMINAMATH_CALUDE_watch_payment_l1379_137954

theorem watch_payment (original_price : ℚ) (discount_rate : ℚ) (dime_value : ℚ) (quarter_value : ℚ) :
  original_price = 15 →
  discount_rate = 1/5 →
  dime_value = 1/10 →
  quarter_value = 1/4 →
  ∃ (num_dimes num_quarters : ℕ),
    (num_dimes : ℚ) = 2 * (num_quarters : ℚ) ∧
    (original_price * (1 - discount_rate) = dime_value * num_dimes + quarter_value * num_quarters) ∧
    num_dimes = 52 :=
by sorry

end NUMINAMATH_CALUDE_watch_payment_l1379_137954


namespace NUMINAMATH_CALUDE_rational_sqrt_two_sum_l1379_137950

theorem rational_sqrt_two_sum (n : ℕ) : n ≥ 2 →
  (∃ a : ℝ, (∃ q : ℚ, a + Real.sqrt 2 = q) ∧ (∃ r : ℚ, a^n + Real.sqrt 2 = r)) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_two_sum_l1379_137950


namespace NUMINAMATH_CALUDE_sequence_max_term_l1379_137941

def a (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem sequence_max_term :
  ∃ (k : ℕ), k = 7 ∧ a k = 108 ∧ ∀ (n : ℕ), a n ≤ a k :=
sorry

end NUMINAMATH_CALUDE_sequence_max_term_l1379_137941


namespace NUMINAMATH_CALUDE_population_growth_rate_l1379_137963

theorem population_growth_rate (initial_population : ℝ) (final_population : ℝ) (second_year_decrease : ℝ) :
  initial_population = 20000 →
  final_population = 18750 →
  second_year_decrease = 0.25 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 0.25 ∧
    final_population = initial_population * (1 + first_year_increase) * (1 - second_year_decrease) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_rate_l1379_137963


namespace NUMINAMATH_CALUDE_wheel_probabilities_l1379_137995

theorem wheel_probabilities :
  ∀ (p_C p_D : ℚ),
    (1 : ℚ)/3 + (1 : ℚ)/4 + p_C + p_D = 1 →
    p_C = 2 * p_D →
    p_C = (5 : ℚ)/18 ∧ p_D = (5 : ℚ)/36 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_probabilities_l1379_137995


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1379_137912

/-- 
Given an angle α with vertex at the origin, initial side on the positive x-axis,
and terminal side on the ray 3x + 4y = 0 with x > 0, prove that sin α = -3/5.
-/
theorem sin_alpha_value (α : Real) : 
  (∃ (x y : Real), x > 0 ∧ 3 * x + 4 * y = 0 ∧ 
   x = Real.cos α ∧ y = Real.sin α) → 
  Real.sin α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1379_137912


namespace NUMINAMATH_CALUDE_stadium_length_l1379_137971

/-- Given a rectangular stadium with perimeter 800 meters and breadth 300 meters, its length is 100 meters. -/
theorem stadium_length (perimeter breadth : ℝ) (h1 : perimeter = 800) (h2 : breadth = 300) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter → perimeter / 2 - breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_l1379_137971


namespace NUMINAMATH_CALUDE_only_D_is_symmetric_l1379_137980

-- Define the type for shapes
inductive Shape
| A
| B
| C
| D
| E

-- Define a function to check if a shape is horizontally symmetric
def isHorizontallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.D => True
  | _ => False

-- Theorem statement
theorem only_D_is_symmetric :
  ∀ s : Shape, isHorizontallySymmetric s ↔ s = Shape.D :=
by
  sorry

end NUMINAMATH_CALUDE_only_D_is_symmetric_l1379_137980


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l1379_137984

theorem simplify_and_evaluate (a : ℝ) (h : a ≠ 1) :
  (1 - 2 / (a + 1)) / ((a^2 - 2*a + 1) / (a + 1)) = 1 / (a - 1) :=
sorry

theorem evaluate_at_two :
  (1 - 2 / (2 + 1)) / ((2^2 - 2*2 + 1) / (2 + 1)) = 1 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l1379_137984


namespace NUMINAMATH_CALUDE_simplify_expression_l1379_137986

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (144 : ℝ) ^ (1/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1379_137986


namespace NUMINAMATH_CALUDE_prob_pass_exactly_once_l1379_137922

/-- The probability of passing a single computer test -/
def p : ℚ := 1 / 3

/-- The number of times the test is taken -/
def n : ℕ := 3

/-- The number of times we want the event to occur -/
def k : ℕ := 1

/-- The probability of passing exactly k times in n independent trials -/
def prob_exactly_k (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem prob_pass_exactly_once :
  prob_exactly_k p n k = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_pass_exactly_once_l1379_137922


namespace NUMINAMATH_CALUDE_used_computer_lifespan_l1379_137929

/-- Proves the lifespan of used computers given certain conditions -/
theorem used_computer_lifespan 
  (new_computer_cost : ℕ)
  (new_computer_lifespan : ℕ)
  (used_computer_cost : ℕ)
  (num_used_computers : ℕ)
  (savings : ℕ)
  (h1 : new_computer_cost = 600)
  (h2 : new_computer_lifespan = 6)
  (h3 : used_computer_cost = 200)
  (h4 : num_used_computers = 2)
  (h5 : savings = 200)
  (h6 : new_computer_cost - savings = num_used_computers * used_computer_cost) :
  ∃ (used_computer_lifespan : ℕ), 
    used_computer_lifespan * num_used_computers = new_computer_lifespan ∧ 
    used_computer_lifespan = 3 := by
  sorry


end NUMINAMATH_CALUDE_used_computer_lifespan_l1379_137929


namespace NUMINAMATH_CALUDE_new_players_count_l1379_137973

theorem new_players_count (returning_players : ℕ) (groups : ℕ) (players_per_group : ℕ) :
  returning_players = 6 →
  groups = 9 →
  players_per_group = 6 →
  groups * players_per_group - returning_players = 48 := by
sorry

end NUMINAMATH_CALUDE_new_players_count_l1379_137973


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_real_roots_l1379_137999

theorem quadratic_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 - 5*x₁ - 1 = 0) ∧ (x₂^2 - 5*x₂ - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_real_roots_l1379_137999


namespace NUMINAMATH_CALUDE_vector_inequality_l1379_137943

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors
variable (A B C D : V)

-- Define the theorem
theorem vector_inequality (h : C - B = -(B - C)) :
  C - B + (A - D) - (B - C) ≠ A - D :=
sorry

end NUMINAMATH_CALUDE_vector_inequality_l1379_137943


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1379_137900

theorem lcm_gcf_problem (n : ℕ+) :
  Nat.lcm n 16 = 52 →
  Nat.gcd n 16 = 8 →
  n = 26 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1379_137900


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1379_137991

theorem absolute_value_inequality (x y z : ℝ) :
  (|x + y + z| + |x*y + y*z + z*x| + |x*y*z| ≤ 1) →
  (max (|x|) (max (|y|) (|z|)) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1379_137991


namespace NUMINAMATH_CALUDE_solution_to_equation_l1379_137939

theorem solution_to_equation : ∃ (x y : ℤ), x + 3 * y = 7 ∧ x = -2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1379_137939


namespace NUMINAMATH_CALUDE_handmade_sweater_cost_l1379_137902

/-- The cost of a handmade sweater given Maria's shopping scenario -/
theorem handmade_sweater_cost 
  (num_sweaters num_scarves : ℕ)
  (scarf_cost : ℚ)
  (initial_savings remaining_savings : ℚ)
  (h1 : num_sweaters = 6)
  (h2 : num_scarves = 6)
  (h3 : scarf_cost = 20)
  (h4 : initial_savings = 500)
  (h5 : remaining_savings = 200) :
  (initial_savings - remaining_savings - num_scarves * scarf_cost) / num_sweaters = 30 := by
  sorry

end NUMINAMATH_CALUDE_handmade_sweater_cost_l1379_137902


namespace NUMINAMATH_CALUDE_solution_difference_l1379_137942

theorem solution_difference (p q : ℝ) : 
  (p - 2) * (p + 4) = 26 * p - 100 →
  (q - 2) * (q + 4) = 26 * q - 100 →
  p ≠ q →
  p > q →
  p - q = 4 * Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l1379_137942


namespace NUMINAMATH_CALUDE_speed_upstream_l1379_137998

theorem speed_upstream (boat_speed : ℝ) (current_speed : ℝ) (h1 : boat_speed = 60) (h2 : current_speed = 17) :
  boat_speed - current_speed = 43 := by
  sorry

end NUMINAMATH_CALUDE_speed_upstream_l1379_137998


namespace NUMINAMATH_CALUDE_gluten_free_pasta_cost_l1379_137957

theorem gluten_free_pasta_cost 
  (mustard_oil_quantity : ℕ)
  (mustard_oil_price : ℚ)
  (pasta_quantity : ℕ)
  (pasta_sauce_quantity : ℕ)
  (pasta_sauce_price : ℚ)
  (initial_money : ℚ)
  (remaining_money : ℚ)
  (h1 : mustard_oil_quantity = 2)
  (h2 : mustard_oil_price = 13)
  (h3 : pasta_quantity = 3)
  (h4 : pasta_sauce_quantity = 1)
  (h5 : pasta_sauce_price = 5)
  (h6 : initial_money = 50)
  (h7 : remaining_money = 7) :
  (initial_money - remaining_money - 
   (mustard_oil_quantity * mustard_oil_price + pasta_sauce_quantity * pasta_sauce_price)) / pasta_quantity = 4 := by
  sorry

end NUMINAMATH_CALUDE_gluten_free_pasta_cost_l1379_137957


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l1379_137925

theorem unique_solution_to_equation :
  ∀ x y z : ℝ,
  x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14 →
  x = -1 ∧ y = -2 ∧ z = -3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l1379_137925


namespace NUMINAMATH_CALUDE_box_side_area_l1379_137990

theorem box_side_area (L W H : ℕ) : 
  L * H = (L * W) / 2 →  -- front area is half of top area
  L * W = (3/2) * (H * W) →  -- top area is 1.5 times side area
  3 * H = 2 * L →  -- length to height ratio is 3:2
  L * W * H = 3000 →  -- volume is 3000
  H * W = 200 :=  -- side area is 200
by sorry

end NUMINAMATH_CALUDE_box_side_area_l1379_137990


namespace NUMINAMATH_CALUDE_anna_savings_account_l1379_137937

def geometricSeriesSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem anna_savings_account (a : ℕ) (r : ℕ) (target : ℕ) :
  a = 2 → r = 2 → target = 500 →
  (∀ k < 8, geometricSeriesSum a r k < target) ∧
  geometricSeriesSum a r 8 ≥ target :=
by sorry

end NUMINAMATH_CALUDE_anna_savings_account_l1379_137937


namespace NUMINAMATH_CALUDE_height_weight_only_correlation_l1379_137988

-- Define the types of relationships
inductive Relationship
  | HeightWeight
  | DistanceTime
  | HeightVision
  | VolumeEdge

-- Define a property for correlation
def is_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.HeightWeight => True
  | _ => False

-- Define a property for functional relationships
def is_functional (r : Relationship) : Prop :=
  match r with
  | Relationship.DistanceTime => True
  | Relationship.VolumeEdge => True
  | _ => False

-- Theorem statement
theorem height_weight_only_correlation :
  ∀ r : Relationship, is_correlated r ↔ r = Relationship.HeightWeight ∧ ¬is_functional r :=
sorry

end NUMINAMATH_CALUDE_height_weight_only_correlation_l1379_137988


namespace NUMINAMATH_CALUDE_point_translation_l1379_137948

def translate_point (x y dx dy : Int) : (Int × Int) := (x + dx, y + dy)

theorem point_translation :
  let P : (Int × Int) := (-5, 1)
  let P1 := translate_point P.1 P.2 2 0
  let P2 := translate_point P1.1 P1.2 0 (-4)
  P2 = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l1379_137948


namespace NUMINAMATH_CALUDE_problem_solution_l1379_137938

theorem problem_solution : 
  ((-3 : ℝ)^0 + (1/3)^2 + (-2)^3 = -62/9) ∧ 
  (∀ x : ℝ, (x + 1)^2 - (1 - 2*x)*(1 + 2*x) = 5*x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1379_137938


namespace NUMINAMATH_CALUDE_book_pages_introduction_l1379_137924

theorem book_pages_introduction (total_pages : ℕ) (text_pages : ℕ) : 
  total_pages = 98 →
  text_pages = 19 →
  (total_pages - total_pages / 2 - text_pages * 2 = 11) :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_introduction_l1379_137924


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1379_137979

theorem rope_cutting_problem (initial_length : ℝ) : 
  initial_length / 2 / 2 / 5 = 5 → initial_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1379_137979


namespace NUMINAMATH_CALUDE_min_c_value_l1379_137958

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b = c)
  (h_unique_solution : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1501 ∧ ∃ (a' b' c' : ℕ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' < b' ∧ b' < c' ∧ a' + b' = c' ∧ c' = 1501 ∧
    ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - c'| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1379_137958


namespace NUMINAMATH_CALUDE_eye_color_hair_color_proportions_l1379_137921

/-- Represents the population characteristics of a kingdom -/
structure Kingdom where
  total : ℕ
  blondes : ℕ
  blue_eyes : ℕ
  blonde_blue_eyes : ℕ
  blonde_blue_eyes_le_blondes : blonde_blue_eyes ≤ blondes
  blonde_blue_eyes_le_blue_eyes : blonde_blue_eyes ≤ blue_eyes
  blondes_le_total : blondes ≤ total
  blue_eyes_le_total : blue_eyes ≤ total

/-- The main theorem about eye color and hair color proportions in the kingdom -/
theorem eye_color_hair_color_proportions (k : Kingdom) :
  (k.blonde_blue_eyes : ℚ) / k.blue_eyes > (k.blondes : ℚ) / k.total →
  (k.blonde_blue_eyes : ℚ) / k.blondes > (k.blue_eyes : ℚ) / k.total :=
by
  sorry

end NUMINAMATH_CALUDE_eye_color_hair_color_proportions_l1379_137921


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1379_137955

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (2 : ℚ) / 3 → q ≤ q') →
  q - p = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1379_137955


namespace NUMINAMATH_CALUDE_ticket_sales_total_l1379_137951

/-- Calculates the total amount collected from ticket sales -/
def totalAmountCollected (adultPrice studentPrice : ℚ) (totalTickets studentTickets : ℕ) : ℚ :=
  let adultTickets := totalTickets - studentTickets
  adultPrice * adultTickets + studentPrice * studentTickets

/-- Proves that the total amount collected from ticket sales is 222.50 -/
theorem ticket_sales_total : 
  totalAmountCollected 4 (5/2) 59 9 = 445/2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l1379_137951


namespace NUMINAMATH_CALUDE_special_sequence_representation_l1379_137944

/-- A sequence of natural numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, a n < 2 * n)

/-- The main theorem statement -/
theorem special_sequence_representation (a : ℕ → ℕ) (h : SpecialSequence a) :
  ∀ m : ℕ, (∃ n, a n = m) ∨ (∃ k l, a k - a l = m) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_representation_l1379_137944


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1379_137923

theorem sine_cosine_inequality (b a : ℝ) (hb : 0 < b ∧ b < 1) (ha : 0 < a ∧ a < Real.pi / 2) :
  Real.rpow b (Real.sin a) < Real.rpow b (Real.sin a) ∧ Real.rpow b (Real.sin a) < Real.rpow b (Real.cos a) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1379_137923


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l1379_137928

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  (reciprocals.sum / 4 : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l1379_137928


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l1379_137969

def S : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_empty :
  (S \ M) ∩ (S \ N) = ∅ := by sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l1379_137969


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1379_137901

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32)
  (h2 : failed_english = 56)
  (h3 : failed_both = 12) :
  100 - (failed_hindi + failed_english - failed_both) = 24 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1379_137901


namespace NUMINAMATH_CALUDE_area_of_three_sectors_l1379_137910

/-- The area of a figure formed by three sectors of a circle,
    where each sector subtends an angle of 40° at the center
    and the circle has a radius of 15. -/
theorem area_of_three_sectors (r : ℝ) (angle : ℝ) (n : ℕ) :
  r = 15 →
  angle = 40 * π / 180 →
  n = 3 →
  n * (angle / (2 * π) * π * r^2) = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_three_sectors_l1379_137910


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1379_137956

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1379_137956


namespace NUMINAMATH_CALUDE_evelyn_skittles_l1379_137989

def skittles_problem (starting_skittles shared_skittles : ℕ) : ℕ :=
  starting_skittles - shared_skittles

theorem evelyn_skittles :
  skittles_problem 76 72 = 4 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_skittles_l1379_137989


namespace NUMINAMATH_CALUDE_mr_c_net_loss_l1379_137962

/-- Represents the value of a house and its transactions -/
structure HouseTransaction where
  initial_value : ℝ
  first_sale_loss_percent : ℝ
  second_sale_gain_percent : ℝ
  additional_tax : ℝ

/-- Calculates the net loss for Mr. C after two transactions -/
def net_loss (t : HouseTransaction) : ℝ :=
  let first_sale_price := t.initial_value * (1 - t.first_sale_loss_percent)
  let second_sale_price := first_sale_price * (1 + t.second_sale_gain_percent) + t.additional_tax
  second_sale_price - t.initial_value

/-- Theorem stating that Mr. C's net loss is $1560 -/
theorem mr_c_net_loss :
  let t : HouseTransaction := {
    initial_value := 8000,
    first_sale_loss_percent := 0.15,
    second_sale_gain_percent := 0.2,
    additional_tax := 200
  }
  net_loss t = 1560 := by
  sorry

end NUMINAMATH_CALUDE_mr_c_net_loss_l1379_137962


namespace NUMINAMATH_CALUDE_problem_statement_l1379_137994

theorem problem_statement :
  (∀ x : ℝ, x^4 - x^3 - x + 1 ≥ 0) ∧
  (1 + 1 + 1 = 3 ∧ 1^3 + 1^3 + 1^3 = 1^4 + 1^4 + 1^4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1379_137994


namespace NUMINAMATH_CALUDE_gwen_zoo_pictures_l1379_137982

/-- The number of pictures Gwen took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The number of pictures Gwen took at the museum -/
def museum_pictures : ℕ := 29

/-- The number of pictures Gwen deleted -/
def deleted_pictures : ℕ := 15

/-- The number of pictures Gwen had after deleting -/
def remaining_pictures : ℕ := 55

/-- Theorem stating that the number of pictures Gwen took at the zoo is 41 -/
theorem gwen_zoo_pictures :
  zoo_pictures = 41 :=
by
  have h1 : zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures :=
    sorry
  sorry

end NUMINAMATH_CALUDE_gwen_zoo_pictures_l1379_137982


namespace NUMINAMATH_CALUDE_min_data_for_plan_y_effectiveness_l1379_137974

/-- Represents the cost in cents for Plan X given the data usage in MB -/
def cost_plan_x (data : ℕ) : ℕ := 20 * data

/-- Represents the cost in cents for Plan Y given the data usage in MB -/
def cost_plan_y (data : ℕ) : ℕ := 1500 + 10 * data

/-- Proves that 151 MB is the minimum amount of data that makes Plan Y more cost-effective -/
theorem min_data_for_plan_y_effectiveness : 
  ∀ d : ℕ, d ≥ 151 ↔ cost_plan_y d < cost_plan_x d :=
by sorry

end NUMINAMATH_CALUDE_min_data_for_plan_y_effectiveness_l1379_137974


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l1379_137945

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l1379_137945


namespace NUMINAMATH_CALUDE_razorback_tshirt_profit_l1379_137959

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_profit :
  let profit_per_shirt : ℕ := 9
  let shirts_sold : ℕ := 245
  let total_profit : ℕ := profit_per_shirt * shirts_sold
  total_profit = 2205 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_profit_l1379_137959


namespace NUMINAMATH_CALUDE_lcm_18_60_l1379_137983

theorem lcm_18_60 : Nat.lcm 18 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_60_l1379_137983


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1379_137947

/-- The maximum y-coordinate of a point on the graph of r = sin 3θ is 9/8 -/
theorem max_y_coordinate_sin_3theta (θ : Real) :
  let r := Real.sin (3 * θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  ∀ y', y' = r * Real.sin θ → y' ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1379_137947


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_128_l1379_137930

/-- Represents a rectangle with an inscribed ellipse -/
structure RectangleWithEllipse where
  rect_area : ℝ
  ellipse_area : ℝ
  major_axis : ℝ

/-- The perimeter of the rectangle given the specified conditions -/
def rectangle_perimeter (r : RectangleWithEllipse) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the rectangle under given conditions -/
theorem rectangle_perimeter_is_128 (r : RectangleWithEllipse) 
  (h1 : r.rect_area = 4032)
  (h2 : r.ellipse_area = 4032 * Real.pi)
  (h3 : r.major_axis = 2 * rectangle_perimeter r / 2) : 
  rectangle_perimeter r = 128 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_128_l1379_137930


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1379_137949

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n * 17 = 85 ∧ 
  (∀ m : ℕ, m * 17 ≤ 99 → m * 17 ≤ 85) := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1379_137949


namespace NUMINAMATH_CALUDE_albert_cabbage_count_l1379_137915

-- Define the number of rows in Albert's cabbage patch
def num_rows : ℕ := 12

-- Define the number of cabbage heads in each row
def heads_per_row : ℕ := 15

-- Define the total number of cabbage heads
def total_heads : ℕ := num_rows * heads_per_row

-- Theorem statement
theorem albert_cabbage_count : total_heads = 180 := by
  sorry

end NUMINAMATH_CALUDE_albert_cabbage_count_l1379_137915


namespace NUMINAMATH_CALUDE_remainder_91_92_mod_100_l1379_137911

theorem remainder_91_92_mod_100 : 91^92 % 100 = 81 := by
  sorry

end NUMINAMATH_CALUDE_remainder_91_92_mod_100_l1379_137911


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l1379_137977

/-- If the cost price of 60 articles is equal to the selling price of 40 articles,
    then the percent profit is 50%. -/
theorem percent_profit_calculation (C S : ℝ) 
  (h : C > 0) 
  (eq : 60 * C = 40 * S) : 
  (S - C) / C * 100 = 50 :=
by sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l1379_137977


namespace NUMINAMATH_CALUDE_set_a_constraint_l1379_137975

theorem set_a_constraint (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a ≥ 0}
  1 ∉ A → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_set_a_constraint_l1379_137975


namespace NUMINAMATH_CALUDE_cherry_pies_count_l1379_137970

/-- Given a total number of pies and a ratio of three types of pies,
    calculate the number of pies of the third type. -/
def calculate_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ) : ℕ :=
  let total_ratio := ratio_apple + ratio_blueberry + ratio_cherry
  let pies_per_part := total_pies / total_ratio
  ratio_cherry * pies_per_part

/-- Theorem stating that given 36 total pies and a ratio of 2:3:4 for apple, blueberry, and cherry pies,
    the number of cherry pies is 16. -/
theorem cherry_pies_count :
  calculate_cherry_pies 36 2 3 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l1379_137970


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1379_137932

theorem train_speed_calculation (train_length platform_length crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : platform_length = 380.04)
  (h3 : crossing_time = 25) :
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * 3.6
  ∃ ε > 0, abs (speed_kmh - 72.01) < ε :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1379_137932


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l1379_137917

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l1379_137917


namespace NUMINAMATH_CALUDE_parrot_initial_phrases_l1379_137931

/-- The number of phrases a parrot initially knew, given the current number of phrases,
    the rate of learning, and the duration of ownership. -/
theorem parrot_initial_phrases (current_phrases : ℕ) (phrases_per_week : ℕ) (days_owned : ℕ) 
    (h1 : current_phrases = 17)
    (h2 : phrases_per_week = 2)
    (h3 : days_owned = 49) :
  current_phrases - (days_owned / 7 * phrases_per_week) = 3 := by
  sorry

#check parrot_initial_phrases

end NUMINAMATH_CALUDE_parrot_initial_phrases_l1379_137931


namespace NUMINAMATH_CALUDE_candy_distribution_proof_l1379_137904

def distribute_candies (n : ℕ) (k : ℕ) (min_counts : List ℕ) : ℕ :=
  sorry

theorem candy_distribution_proof :
  distribute_candies 10 4 [1, 1, 1, 0] = 3176 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_proof_l1379_137904


namespace NUMINAMATH_CALUDE_total_cost_of_suits_l1379_137997

/-- The total cost of two suits, given the cost of an off-the-rack suit and the pricing rule for a tailored suit. -/
theorem total_cost_of_suits (off_the_rack_cost : ℕ) : 
  off_the_rack_cost = 300 →
  off_the_rack_cost + (3 * off_the_rack_cost + 200) = 1400 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_suits_l1379_137997


namespace NUMINAMATH_CALUDE_remaining_milk_calculation_l1379_137946

/-- The amount of milk arranged by the shop owner in liters -/
def total_milk : ℝ := 21.52

/-- The amount of milk sold in liters -/
def sold_milk : ℝ := 12.64

/-- The amount of remaining milk in liters -/
def remaining_milk : ℝ := total_milk - sold_milk

theorem remaining_milk_calculation : remaining_milk = 8.88 := by
  sorry

end NUMINAMATH_CALUDE_remaining_milk_calculation_l1379_137946


namespace NUMINAMATH_CALUDE_original_sweets_per_child_l1379_137908

/-- Proves that the original number of sweets per child is 15 --/
theorem original_sweets_per_child (total_children : ℕ) (absent_children : ℕ) (extra_sweets : ℕ) : 
  total_children = 112 → 
  absent_children = 32 → 
  extra_sweets = 6 → 
  ∃ (total_sweets : ℕ), 
    total_sweets = total_children * 15 ∧ 
    total_sweets = (total_children - absent_children) * (15 + extra_sweets) := by
  sorry


end NUMINAMATH_CALUDE_original_sweets_per_child_l1379_137908


namespace NUMINAMATH_CALUDE_divisibility_and_ratio_theorem_l1379_137976

theorem divisibility_and_ratio_theorem (k : ℕ) (h : k > 1) :
  ∃ a b : ℕ, 1 < a ∧ a < b ∧ (a^2 + b^2 - 1) / (a * b) = k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_ratio_theorem_l1379_137976


namespace NUMINAMATH_CALUDE_average_age_decrease_l1379_137918

theorem average_age_decrease (initial_avg : ℝ) : 
  let initial_total := 10 * initial_avg
  let new_total := initial_total - 45 + 15
  let new_avg := new_total / 10
  initial_avg - new_avg = 3 := by sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1379_137918


namespace NUMINAMATH_CALUDE_average_daily_high_temp_l1379_137967

def daily_highs : List ℝ := [51, 63, 59, 56, 47, 64, 52]

theorem average_daily_high_temp : 
  (daily_highs.sum / daily_highs.length : ℝ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_high_temp_l1379_137967


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l1379_137920

-- Define the first equation
def equation1 (x : ℝ) : Prop := 2 / x = 3 / (x + 2)

-- Define the second equation
def equation2 (x : ℝ) : Prop := 5 / (x - 2) + 1 = (x - 7) / (2 - x)

-- Theorem for the first equation
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x ≠ 0 ∧ x + 2 ≠ 0 := by sorry

-- Theorem for the second equation
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x ∧ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l1379_137920


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l1379_137960

theorem sqrt_expression_simplification :
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 90 / Real.sqrt 2 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l1379_137960


namespace NUMINAMATH_CALUDE_inequality_always_true_l1379_137935

theorem inequality_always_true (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l1379_137935


namespace NUMINAMATH_CALUDE_inequality_proof_l1379_137916

theorem inequality_proof (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1379_137916


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l1379_137914

theorem quadratic_polynomial_property (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let p := fun x => (a^2 + a*b + b^2 + a*c + b*c + c^2) * x^2 - 
                    (a + b) * (b + c) * (a + c) * x + 
                    a * b * c * (a + b + c)
  p a = a^4 ∧ p b = b^4 ∧ p c = c^4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l1379_137914


namespace NUMINAMATH_CALUDE_tunnel_length_l1379_137993

theorem tunnel_length : 
  ∀ (initial_speed : ℝ),
  initial_speed > 0 →
  (400 + 8600) / initial_speed = 10 →
  (400 + 8600) / (initial_speed + 0.1 * 60) = 9 :=
by sorry

end NUMINAMATH_CALUDE_tunnel_length_l1379_137993


namespace NUMINAMATH_CALUDE_ratio_problem_l1379_137961

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : 
  y / x = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1379_137961
