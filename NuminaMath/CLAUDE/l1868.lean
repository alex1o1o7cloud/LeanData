import Mathlib

namespace NUMINAMATH_CALUDE_merck_hourly_rate_l1868_186832

/-- Represents the babysitting data for Layla --/
structure BabysittingData where
  donaldson_hours : ℕ
  merck_hours : ℕ
  hille_hours : ℕ
  total_earnings : ℚ

/-- Calculates the hourly rate for babysitting --/
def hourly_rate (data : BabysittingData) : ℚ :=
  data.total_earnings / (data.donaldson_hours + data.merck_hours + data.hille_hours)

/-- Theorem stating that the hourly rate for the Merck family is $17.0625 --/
theorem merck_hourly_rate (data : BabysittingData) 
  (h1 : data.donaldson_hours = 7)
  (h2 : data.merck_hours = 6)
  (h3 : data.hille_hours = 3)
  (h4 : data.total_earnings = 273) :
  hourly_rate data = 17.0625 := by
  sorry

end NUMINAMATH_CALUDE_merck_hourly_rate_l1868_186832


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_for_given_intersection_l1868_186895

-- Define set A
def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem m_value_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_for_given_intersection_l1868_186895


namespace NUMINAMATH_CALUDE_square_binomial_constant_l1868_186801

theorem square_binomial_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 50*x + c = (x + a)^2 + b) → c = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l1868_186801


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l1868_186857

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem triangle_area_comparison : 
  triangleArea 30 30 45 > triangleArea 30 30 55 := by sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l1868_186857


namespace NUMINAMATH_CALUDE_steve_reading_time_l1868_186886

/-- Calculates the number of weeks needed to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_day : ℕ) (reading_days_per_week : ℕ) : ℕ :=
  total_pages / (pages_per_day * reading_days_per_week)

/-- Proves that it takes 7 weeks to read a 2100-page book when reading 100 pages on 3 days per week. -/
theorem steve_reading_time : weeks_to_read 2100 100 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_steve_reading_time_l1868_186886


namespace NUMINAMATH_CALUDE_max_regions_five_lines_l1868_186824

/-- The maximum number of regions created by n line segments in a rectangle -/
def max_regions (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of regions created by 5 line segments in a rectangle is 16 -/
theorem max_regions_five_lines : max_regions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_five_lines_l1868_186824


namespace NUMINAMATH_CALUDE_section_formula_vector_form_l1868_186834

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q⃗ = (5/8)C⃗ + (3/8)D⃗ --/
theorem section_formula_vector_form (C D Q : EuclideanSpace ℝ (Fin 3)) :
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) →  -- Q is on line segment CD
  (∃ s : ℝ, s > 0 ∧ dist C Q = (3 / (3 + 5)) * s ∧ dist Q D = (5 / (3 + 5)) * s) →  -- CQ:QD = 3:5
  Q = (5 / 8) • C + (3 / 8) • D :=
by sorry

end NUMINAMATH_CALUDE_section_formula_vector_form_l1868_186834


namespace NUMINAMATH_CALUDE_binomial_expansion_5_plus_4_cubed_l1868_186897

theorem binomial_expansion_5_plus_4_cubed : (5 + 4)^3 = 729 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_5_plus_4_cubed_l1868_186897


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l1868_186804

/-- The sum of the series ∑(n=1 to ∞) (4n-3)/3^n is equal to 1 -/
theorem series_sum_equals_one :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / 3^n) = 1 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l1868_186804


namespace NUMINAMATH_CALUDE_remaining_grain_l1868_186831

theorem remaining_grain (original : ℕ) (spilled : ℕ) (remaining : ℕ) : 
  original = 50870 → spilled = 49952 → remaining = original - spilled → remaining = 918 := by
  sorry

end NUMINAMATH_CALUDE_remaining_grain_l1868_186831


namespace NUMINAMATH_CALUDE_gumball_probability_l1868_186871

theorem gumball_probability (blue_twice_prob : ℚ) 
  (h1 : blue_twice_prob = 16/49) : 
  let blue_prob : ℚ := (blue_twice_prob.sqrt)
  let pink_prob : ℚ := 1 - blue_prob
  pink_prob = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l1868_186871


namespace NUMINAMATH_CALUDE_expression_value_l1868_186836

theorem expression_value : (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1868_186836


namespace NUMINAMATH_CALUDE_simple_random_for_small_population_systematic_for_large_uniform_population_stratified_for_population_with_strata_l1868_186812

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define a structure for a sampling scenario
structure SamplingScenario where
  populationSize : ℕ
  sampleSize : ℕ
  hasStrata : Bool
  uniformDistribution : Bool

-- Define the function to determine the most appropriate sampling method
def mostAppropriateSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

-- Theorem for the first scenario
theorem simple_random_for_small_population :
  mostAppropriateSamplingMethod { populationSize := 20, sampleSize := 4, hasStrata := false, uniformDistribution := true } = SamplingMethod.SimpleRandom :=
  sorry

-- Theorem for the second scenario
theorem systematic_for_large_uniform_population :
  mostAppropriateSamplingMethod { populationSize := 1280, sampleSize := 32, hasStrata := false, uniformDistribution := true } = SamplingMethod.Systematic :=
  sorry

-- Theorem for the third scenario
theorem stratified_for_population_with_strata :
  mostAppropriateSamplingMethod { populationSize := 180, sampleSize := 15, hasStrata := true, uniformDistribution := false } = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_simple_random_for_small_population_systematic_for_large_uniform_population_stratified_for_population_with_strata_l1868_186812


namespace NUMINAMATH_CALUDE_fraction_problem_l1868_186898

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * F * (2/5) * N = 35)
  (h2 : (40/100) * N = 420) : F = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1868_186898


namespace NUMINAMATH_CALUDE_gcd_1237_1849_l1868_186830

theorem gcd_1237_1849 : Nat.gcd 1237 1849 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1237_1849_l1868_186830


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1868_186847

theorem min_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 18*x + 8*y + 10 → (∀ a b : ℝ, a^2 + b^2 = 18*a + 8*b + 10 → 4*x + 3*y ≤ 4*a + 3*b) → 4*x + 3*y = -40 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1868_186847


namespace NUMINAMATH_CALUDE_courtyard_length_l1868_186825

/-- Proves that the length of a courtyard is 25 meters given specific conditions -/
theorem courtyard_length : 
  ∀ (width : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℕ),
  width = 15 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 18750 →
  (width * (total_bricks : ℝ) * brick_length * brick_width) / width = 25 := by
sorry

end NUMINAMATH_CALUDE_courtyard_length_l1868_186825


namespace NUMINAMATH_CALUDE_total_shaded_area_is_72_l1868_186800

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a parallelogram with base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := p.base * p.height

/-- Represents the overlap between shapes -/
structure Overlap where
  width : ℝ
  height : ℝ

/-- Calculates the area of overlap -/
def overlapArea (o : Overlap) : ℝ := o.width * o.height

/-- Theorem: The total shaded area of intersection between the given rectangle and parallelogram is 72 square units -/
theorem total_shaded_area_is_72 (r : Rectangle) (p : Parallelogram) (o : Overlap) : 
  r.width = 4 ∧ r.height = 12 ∧ p.base = 10 ∧ p.height = 4 ∧ o.width = 4 ∧ o.height = 4 →
  rectangleArea r + parallelogramArea p - overlapArea o = 72 := by
  sorry


end NUMINAMATH_CALUDE_total_shaded_area_is_72_l1868_186800


namespace NUMINAMATH_CALUDE_smallest_difference_vovochka_sum_l1868_186829

/-- Vovochka's sum method for three-digit numbers -/
def vovochkaSum (a b c d e f : ℕ) : ℕ := 
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- Correct sum method for three-digit numbers -/
def correctSum (a b c d e f : ℕ) : ℕ := 
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f)

/-- Theorem: The smallest positive difference between Vovochka's sum and the correct sum is 1800 -/
theorem smallest_difference_vovochka_sum : 
  ∀ a b c d e f : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
  vovochkaSum a b c d e f - correctSum a b c d e f ≥ 1800 ∧
  ∃ a b c d e f : ℕ, 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    vovochkaSum a b c d e f - correctSum a b c d e f = 1800 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_vovochka_sum_l1868_186829


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_l1868_186849

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 2 * x₁ - 1 = 0) → 
  (2 * x₂^2 - 2 * x₂ - 1 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_l1868_186849


namespace NUMINAMATH_CALUDE_survey_respondents_l1868_186878

/-- Prove that the number of customers who responded to a survey is 50, given the following conditions:
  1. The average income of all customers is $45,000
  2. There are 10 wealthiest customers
  3. The average income of the 10 wealthiest customers is $55,000
  4. The average income of the remaining customers is $42,500
-/
theorem survey_respondents (N : ℕ) : 
  (10 * 55000 + (N - 10) * 42500 = N * 45000) → N = 50 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l1868_186878


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_l1868_186870

theorem parallelogram_smaller_angle (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ x + (x + 90) = 180 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_l1868_186870


namespace NUMINAMATH_CALUDE_map_coloring_theorem_l1868_186841

/-- Represents a map with regions -/
structure Map where
  regions : Nat
  adjacency : List (Nat × Nat)

/-- The minimum number of colors needed to color a map -/
def minColors (m : Map) : Nat :=
  sorry

/-- Theorem: The minimum number of colors needed for a 26-region map is 4 -/
theorem map_coloring_theorem (m : Map) (h1 : m.regions = 26) 
  (h2 : ∀ (i j : Nat), (i, j) ∈ m.adjacency → i ≠ j) 
  (h3 : minColors m > 3) : 
  minColors m = 4 :=
sorry

end NUMINAMATH_CALUDE_map_coloring_theorem_l1868_186841


namespace NUMINAMATH_CALUDE_neither_alive_probability_l1868_186873

/-- The probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1/4

/-- The probability that a woman will be alive for 10 more years -/
def prob_woman_alive : ℚ := 1/3

/-- The probability that neither the man nor the woman will be alive for 10 more years -/
def prob_neither_alive : ℚ := (1 - prob_man_alive) * (1 - prob_woman_alive)

theorem neither_alive_probability : prob_neither_alive = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_neither_alive_probability_l1868_186873


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1868_186863

/-- Given a quadratic function f(x) = ax² + bx where a > 0 and b > 0,
    and the slope of the tangent line at x = 1 is 2,
    prove that the minimum value of (8a + b) / (ab) is 9 -/
theorem min_value_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : 2*a + b = 2) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (8*x + y) / (x*y) ≥ 9) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (8*x + y) / (x*y) = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1868_186863


namespace NUMINAMATH_CALUDE_percentage_calculation_l1868_186840

theorem percentage_calculation (P : ℝ) : 
  P * (0.3 * (0.5 * 4000)) = 90 → P = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1868_186840


namespace NUMINAMATH_CALUDE_car_average_speed_l1868_186842

theorem car_average_speed 
  (total_time : ℝ) 
  (first_interval : ℝ) 
  (first_speed : ℝ) 
  (second_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : first_interval = 4) 
  (h3 : first_speed = 70) 
  (h4 : second_speed = 60) : 
  (first_speed * first_interval + second_speed * (total_time - first_interval)) / total_time = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l1868_186842


namespace NUMINAMATH_CALUDE_window_width_calculation_l1868_186805

/-- Calculates the width of each window in a room given the room dimensions,
    door dimensions, number of windows, window height, cost per square foot,
    and total cost of whitewashing. -/
theorem window_width_calculation (room_length room_width room_height : ℝ)
                                 (door_height door_width : ℝ)
                                 (num_windows : ℕ)
                                 (window_height : ℝ)
                                 (cost_per_sqft total_cost : ℝ) :
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_height = 6 ∧ door_width = 3 ∧
  num_windows = 3 ∧
  window_height = 3 ∧
  cost_per_sqft = 9 ∧
  total_cost = 8154 →
  ∃ (window_width : ℝ),
    window_width = 4 ∧
    total_cost = (2 * (room_length + room_width) * room_height -
                  door_height * door_width -
                  num_windows * window_height * window_width) * cost_per_sqft :=
by sorry

end NUMINAMATH_CALUDE_window_width_calculation_l1868_186805


namespace NUMINAMATH_CALUDE_max_ab_value_l1868_186820

/-- The maximum value of ab given the conditions -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a * 2 - b * (-1) = 2) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → (x - 2)^2 + (y + 1)^2 = 4) :
  a * b ≤ 1/2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * 2 - b * (-1) = 2 ∧ a * b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1868_186820


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l1868_186843

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the line
structure Line where
  m : ℝ
  c : ℝ

-- Define the problem
theorem hyperbola_line_intersection
  (h : Hyperbola)
  (l : Line)
  (P Q R : Point)
  (h_eccentricity : h.e = Real.sqrt 3)
  (l_slope : l.m = 1)
  (intersect_y_axis : R.x = 0)
  (dot_product : P.x * Q.x + P.y * Q.y = -3)
  (segment_ratio : P.x - R.x = 3 * (R.x - Q.x))
  (on_line_P : P.y = l.m * P.x + l.c)
  (on_line_Q : Q.y = l.m * Q.x + l.c)
  (on_line_R : R.y = l.m * R.x + l.c)
  (on_hyperbola_P : 2 * P.x^2 - P.y^2 = 2 * h.a^2)
  (on_hyperbola_Q : 2 * Q.x^2 - Q.y^2 = 2 * h.a^2) :
  (l.c = 1 ∨ l.c = -1) ∧ h.a^2 = 1 ∧ h.b^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l1868_186843


namespace NUMINAMATH_CALUDE_trains_meeting_time_l1868_186891

/-- Two trains meeting problem -/
theorem trains_meeting_time
  (distance : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (start_time_diff : ℝ)
  (h_distance : distance = 200)
  (h_speed_A : speed_A = 20)
  (h_speed_B : speed_B = 25)
  (h_start_time_diff : start_time_diff = 1) :
  let initial_distance_A := speed_A * start_time_diff
  let remaining_distance := distance - initial_distance_A
  let relative_speed := speed_A + speed_B
  let meeting_time := remaining_distance / relative_speed
  meeting_time + start_time_diff = 5 := by sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l1868_186891


namespace NUMINAMATH_CALUDE_circle_radius_l1868_186884

/-- Given a circle with equation x^2 + y^2 - 2ax + 2 = 0 and center (2, 0), its radius is √2 -/
theorem circle_radius (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2 = 0 ↔ (x - 2)^2 + y^2 = 2) → 
  (∃ r : ℝ, r > 0 ∧ r^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1868_186884


namespace NUMINAMATH_CALUDE_no_abcd_2012_l1868_186867

theorem no_abcd_2012 (a b c d : ℤ) 
  (h : (a - b) * (c + d) = (a + b) * (c - d)) : 
  a * b * c * d ≠ 2012 := by
  sorry

end NUMINAMATH_CALUDE_no_abcd_2012_l1868_186867


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_bound_l1868_186806

def is_permutation (σ : Fin 100 → ℕ) : Prop :=
  Function.Bijective σ ∧ ∀ i, σ i ∈ Finset.range 101

def consecutive_sum (σ : Fin 100 → ℕ) (start : Fin 91) : ℕ :=
  (Finset.range 10).sum (λ i => σ (start + i))

theorem largest_consecutive_sum_bound :
  (∃ A : ℕ, A = 505 ∧
    (∀ σ : Fin 100 → ℕ, is_permutation σ →
      ∃ start : Fin 91, consecutive_sum σ start ≥ A) ∧
    ∀ B : ℕ, B > A →
      ∃ σ : Fin 100 → ℕ, is_permutation σ ∧
        ∀ start : Fin 91, consecutive_sum σ start < B) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_bound_l1868_186806


namespace NUMINAMATH_CALUDE_potato_peeling_result_l1868_186814

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ := 60
  homer_rate : ℕ := 4
  christen_rate : ℕ := 6
  homer_solo_time : ℕ := 5

/-- Calculates the number of potatoes Christen peeled and the total time taken -/
def peel_potatoes (scenario : PotatoPeeling) : ℕ × ℕ := by
  sorry

/-- Theorem stating the correct result of the potato peeling scenario -/
theorem potato_peeling_result (scenario : PotatoPeeling) :
  peel_potatoes scenario = (24, 9) := by
  sorry

end NUMINAMATH_CALUDE_potato_peeling_result_l1868_186814


namespace NUMINAMATH_CALUDE_boys_pass_percentage_l1868_186826

theorem boys_pass_percentage (total_candidates : ℕ) (girls : ℕ) (girls_pass_rate : ℚ) (total_fail_rate : ℚ) :
  total_candidates = 2000 →
  girls = 900 →
  girls_pass_rate = 32 / 100 →
  total_fail_rate = 647 / 1000 →
  let boys := total_candidates - girls
  let total_pass_rate := 1 - total_fail_rate
  let total_pass := total_pass_rate * total_candidates
  let girls_pass := girls_pass_rate * girls
  let boys_pass := total_pass - girls_pass
  let boys_pass_rate := boys_pass / boys
  boys_pass_rate = 38 / 100 := by
sorry

end NUMINAMATH_CALUDE_boys_pass_percentage_l1868_186826


namespace NUMINAMATH_CALUDE_correct_calculation_l1868_186845

theorem correct_calculation (a : ℝ) : -2*a + (2*a - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1868_186845


namespace NUMINAMATH_CALUDE_football_purchase_theorem_l1868_186858

/-- Represents the cost and quantity of footballs purchased by a school --/
structure FootballPurchase where
  type_a_cost : ℕ
  type_b_cost : ℕ
  type_a_quantity : ℕ
  type_b_quantity : ℕ
  total_cost : ℕ
  cost_difference : ℕ

/-- Represents the second purchase with budget constraints --/
structure SecondPurchase where
  budget : ℕ
  total_quantity : ℕ

/-- Theorem stating the costs of footballs and minimum quantity of type A footballs in second purchase --/
theorem football_purchase_theorem (fp : FootballPurchase) (sp : SecondPurchase) :
  fp.type_a_quantity = 50 ∧ 
  fp.type_b_quantity = 25 ∧ 
  fp.total_cost = 7500 ∧ 
  fp.cost_difference = 30 ∧ 
  fp.type_b_cost = fp.type_a_cost + fp.cost_difference ∧
  sp.budget = 4800 ∧
  sp.total_quantity = 50 →
  fp.type_a_cost = 90 ∧ 
  fp.type_b_cost = 120 ∧ 
  (∃ m : ℕ, m ≥ 40 ∧ m * fp.type_a_cost + (sp.total_quantity - m) * fp.type_b_cost ≤ sp.budget) :=
by sorry

end NUMINAMATH_CALUDE_football_purchase_theorem_l1868_186858


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1868_186896

theorem cubic_equation_solution (x : ℝ) (h : 9 / x^2 = x / 25) : x = (225 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1868_186896


namespace NUMINAMATH_CALUDE_floor_abs_sum_l1868_186848

theorem floor_abs_sum : ⌊|(-7.6 : ℝ)|⌋ + |⌊(-7.6 : ℝ)⌋| = 15 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l1868_186848


namespace NUMINAMATH_CALUDE_correct_prices_l1868_186868

def chair_desk_prices (total_price : ℕ) (price_difference : ℕ) : ℕ × ℕ :=
  let chair_price := (total_price - price_difference) / 2
  let desk_price := total_price - chair_price
  (chair_price, desk_price)

theorem correct_prices : chair_desk_prices 115 45 = (35, 80) := by
  sorry

end NUMINAMATH_CALUDE_correct_prices_l1868_186868


namespace NUMINAMATH_CALUDE_candle_height_shadow_relation_l1868_186880

/-- Given two positions of a gnomon and the shadows cast, we can relate the height of the object to the shadow lengths and distance between positions. -/
theorem candle_height_shadow_relation 
  (h : ℝ) -- height of the candle
  (d : ℝ) -- distance between gnomon positions
  (a : ℝ) -- length of shadow in first position
  (b : ℝ) -- length of shadow in second position
  (x : ℝ) -- length from base of candle at first position to end of shadow in second position plus d
  (h_pos : h > 0)
  (d_pos : d > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (x_def : x = d + b) -- definition of x
  : x = h * (1 + d / (a + b)) := by
  sorry


end NUMINAMATH_CALUDE_candle_height_shadow_relation_l1868_186880


namespace NUMINAMATH_CALUDE_triangle_inequality_l1868_186817

/-- Given a triangle with sides a, b, c and angle γ opposite side c,
    prove that c ≥ (a + b) * sin(γ/2) --/
theorem triangle_inequality (a b c γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angle : 0 < γ ∧ γ < π)
  (h_opposite : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) :
  c ≥ (a + b) * Real.sin (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1868_186817


namespace NUMINAMATH_CALUDE_triangle_properties_l1868_186846

open Real

structure Triangle (A B C : ℝ) where
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_properties (A B C : ℝ) (h : Triangle A B C) 
  (h1 : A + B = 3 * C) (h2 : 2 * sin (A - C) = sin B) (h3 : ∃ (AB : ℝ), AB = 5) :
  sin A = (3 * sqrt 10) / 10 ∧ 
  ∃ (height : ℝ), height = 6 ∧ 
    height * 5 / 2 = (sqrt 10 * 3 * sqrt 5 * sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1868_186846


namespace NUMINAMATH_CALUDE_square_of_triple_l1868_186861

theorem square_of_triple (a : ℝ) : (3 * a)^2 = 9 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_triple_l1868_186861


namespace NUMINAMATH_CALUDE_max_value_h_exists_m_for_inequality_l1868_186819

open Real

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := log x

/-- The square function -/
def g (x : ℝ) : ℝ := x^2

/-- The function h(x) = ln x - x + 1 -/
noncomputable def h (x : ℝ) : ℝ := f x - x + 1

theorem max_value_h :
  ∀ x > 0, h x ≤ 0 ∧ ∃ x₀ > 0, h x₀ = 0 :=
sorry

theorem exists_m_for_inequality (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hlt : x₁ < x₂) :
  ∃ m ≤ (-1/2), m * (g x₂ - g x₁) - x₂ * f x₂ + x₁ * f x₁ > 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_h_exists_m_for_inequality_l1868_186819


namespace NUMINAMATH_CALUDE_range_of_a_l1868_186813

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀ + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1868_186813


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l1868_186881

-- (1)
theorem simplify_expression_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = (14 * Real.sqrt 5) / 5 := by sorry

-- (2)
theorem simplify_expression_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 := by sorry

-- (3)
theorem simplify_expression_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 := by sorry

-- (4)
theorem simplify_expression_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3)^2 = 2 * Real.sqrt 15 - 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l1868_186881


namespace NUMINAMATH_CALUDE_exam_mode_l1868_186869

def scores : List Nat := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (fun acc x =>
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem exam_mode :
  mode scores = some 9 := by
  sorry

end NUMINAMATH_CALUDE_exam_mode_l1868_186869


namespace NUMINAMATH_CALUDE_lcm_of_three_numbers_l1868_186807

theorem lcm_of_three_numbers (A B C : ℕ+) 
  (h_product : A * B * C = 185771616)
  (h_hcf_abc : Nat.gcd A (Nat.gcd B C) = 121)
  (h_hcf_ab : Nat.gcd A B = 363) :
  Nat.lcm A (Nat.lcm B C) = 61919307 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_three_numbers_l1868_186807


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l1868_186808

/-- Given plane vectors OA, OB, OC satisfying certain conditions, 
    the maximum value of x + y is √2. -/
theorem max_value_x_plus_y (OA OB OC : ℝ × ℝ) (x y : ℝ) : 
  (norm OA = 1) → 
  (norm OB = 1) → 
  (norm OC = 1) → 
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) → 
  (OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2)) → 
  (∃ (x y : ℝ), x + y ≤ Real.sqrt 2 ∧ 
    ∀ (x' y' : ℝ), (OC = (x' * OA.1 + y' * OB.1, x' * OA.2 + y' * OB.2)) → 
      x' + y' ≤ x + y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l1868_186808


namespace NUMINAMATH_CALUDE_base4_calculation_l1868_186818

/-- Convert a number from base 4 to base 10 -/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to base 4 -/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- Multiplication in base 4 -/
def mul_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a * base4_to_base10 b)

/-- Division in base 4 -/
def div_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a / base4_to_base10 b)

theorem base4_calculation : 
  div_base4 (mul_base4 231 24) 3 = 2310 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l1868_186818


namespace NUMINAMATH_CALUDE_descending_order_abcd_l1868_186810

theorem descending_order_abcd (a b c d : ℚ) 
  (h1 : 2006 = 9 * a) 
  (h2 : 2006 = 15 * b) 
  (h3 : 2006 = 32 * c) 
  (h4 : 2006 = 68 * d) : 
  a > b ∧ b > c ∧ c > d := by
  sorry

end NUMINAMATH_CALUDE_descending_order_abcd_l1868_186810


namespace NUMINAMATH_CALUDE_m_range_l1868_186855

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, (|x - m| < 1 ↔ 1/3 < x ∧ x < 1/2)) ↔ 
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1868_186855


namespace NUMINAMATH_CALUDE_max_positive_cyclic_sequence_l1868_186852

theorem max_positive_cyclic_sequence (x : Fin 2022 → ℝ) 
  (h_nonzero : ∀ i, x i ≠ 0)
  (h_inequality : ∀ i : Fin 2022, x i + 1 / x (Fin.succ i) < 0)
  (h_cyclic : x 0 = x (Fin.last 2021)) : 
  (Finset.filter (fun i => x i > 0) Finset.univ).card ≤ 1010 := by
  sorry

end NUMINAMATH_CALUDE_max_positive_cyclic_sequence_l1868_186852


namespace NUMINAMATH_CALUDE_train_length_calculation_l1868_186892

/-- Calculates the length of a train given its speed, time to cross a platform, and the platform length. -/
def train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : ℝ :=
  speed * time - platform_length

/-- Proves that a train with speed 35 m/s crossing a 250.056 m platform in 20 seconds has a length of 449.944 m. -/
theorem train_length_calculation :
  train_length 35 20 250.056 = 449.944 := by
  sorry

#eval train_length 35 20 250.056

end NUMINAMATH_CALUDE_train_length_calculation_l1868_186892


namespace NUMINAMATH_CALUDE_blue_box_lightest_l1868_186823

/-- Represents the color of balls -/
inductive BallColor
  | Yellow
  | White
  | Blue

/-- Represents a box of balls -/
structure Box where
  color : BallColor
  ballCount : Nat
  ballWeight : Nat

/-- Calculate the total weight of balls in a box -/
def boxWeight (box : Box) : Nat :=
  box.ballCount * box.ballWeight

/-- Theorem: The box with blue balls has the lightest weight -/
theorem blue_box_lightest (yellowBox whiteBox blueBox : Box)
  (h_yellow : yellowBox.color = BallColor.Yellow ∧ yellowBox.ballCount = 50 ∧ yellowBox.ballWeight = 50)
  (h_white : whiteBox.color = BallColor.White ∧ whiteBox.ballCount = 60 ∧ whiteBox.ballWeight = 45)
  (h_blue : blueBox.color = BallColor.Blue ∧ blueBox.ballCount = 40 ∧ blueBox.ballWeight = 55) :
  boxWeight blueBox < boxWeight yellowBox ∧ boxWeight blueBox < boxWeight whiteBox :=
by sorry

end NUMINAMATH_CALUDE_blue_box_lightest_l1868_186823


namespace NUMINAMATH_CALUDE_expression_simplification_l1868_186893

theorem expression_simplification (x y : ℝ) (hx : x = 2) (hy : y = 1/2) :
  (x + y) * (x - y) + (x - y)^2 - (x^2 - 3*x*y) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1868_186893


namespace NUMINAMATH_CALUDE_tournament_max_points_l1868_186890

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : Nat)
  (games_per_pair : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (points_for_loss : Nat)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : Nat :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Represents the maximum points achievable by top teams -/
def max_points_for_top_teams (t : Tournament) : Nat :=
  let games_against_lower := (t.num_teams - 3) * t.games_per_pair
  let points_from_lower := games_against_lower * t.points_for_win
  let games_among_top := 2 * t.games_per_pair
  let points_from_top := games_among_top * t.points_for_win / 2
  points_from_lower + points_from_top

/-- The main theorem to be proved -/
theorem tournament_max_points :
  ∀ t : Tournament,
    t.num_teams = 8 ∧
    t.games_per_pair = 2 ∧
    t.points_for_win = 3 ∧
    t.points_for_draw = 1 ∧
    t.points_for_loss = 0 →
    max_points_for_top_teams t = 36 := by
  sorry

end NUMINAMATH_CALUDE_tournament_max_points_l1868_186890


namespace NUMINAMATH_CALUDE_student_average_grade_previous_year_l1868_186875

/-- Represents the average grade of a student for a given year -/
structure YearlyAverage where
  courses : ℕ
  average : ℝ

/-- Calculates the total points for a year -/
def totalPoints (ya : YearlyAverage) : ℝ := ya.courses * ya.average

theorem student_average_grade_previous_year 
  (last_year : YearlyAverage)
  (prev_year : YearlyAverage)
  (h1 : last_year.courses = 6)
  (h2 : last_year.average = 100)
  (h3 : prev_year.courses = 5)
  (h4 : (totalPoints last_year + totalPoints prev_year) / (last_year.courses + prev_year.courses) = 81) :
  prev_year.average = 58.2 := by
  sorry


end NUMINAMATH_CALUDE_student_average_grade_previous_year_l1868_186875


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l1868_186876

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x ≤ 22 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l1868_186876


namespace NUMINAMATH_CALUDE_more_squirrels_than_nuts_l1868_186865

def num_squirrels : ℕ := 4
def num_nuts : ℕ := 2

theorem more_squirrels_than_nuts : num_squirrels - num_nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_more_squirrels_than_nuts_l1868_186865


namespace NUMINAMATH_CALUDE_trapezoid_median_equilateral_triangles_l1868_186899

/-- The median of a trapezoid formed by sides of two equilateral triangles -/
theorem trapezoid_median_equilateral_triangles 
  (large_side : ℝ) 
  (small_side : ℝ) 
  (h1 : large_side = 4) 
  (h2 : small_side = large_side / 2) : 
  (large_side + small_side) / 2 = 3 := by
  sorry

#check trapezoid_median_equilateral_triangles

end NUMINAMATH_CALUDE_trapezoid_median_equilateral_triangles_l1868_186899


namespace NUMINAMATH_CALUDE_original_number_problem_l1868_186862

theorem original_number_problem : ∃ x : ℕ, x / 3 = 42 ∧ x = 126 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l1868_186862


namespace NUMINAMATH_CALUDE_triangle_regions_l1868_186864

theorem triangle_regions (p : ℕ) (h_prime : Nat.Prime p) (h_ge_3 : p ≥ 3) :
  let num_lines := 3 * p
  (num_lines * (num_lines + 1)) / 2 + 1 = 3 * p^2 - 3 * p + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_regions_l1868_186864


namespace NUMINAMATH_CALUDE_micheal_licks_proof_l1868_186822

/-- The number of licks it takes for Dan to reach the center of a lollipop. -/
def dan_licks : ℕ := 58

/-- The number of licks it takes for Sam to reach the center of a lollipop. -/
def sam_licks : ℕ := 70

/-- The number of licks it takes for David to reach the center of a lollipop. -/
def david_licks : ℕ := 70

/-- The number of licks it takes for Lance to reach the center of a lollipop. -/
def lance_licks : ℕ := 39

/-- The average number of licks it takes for all 5 people to reach the center of a lollipop. -/
def average_licks : ℕ := 60

/-- The number of licks it takes for Micheal to reach the center of a lollipop. -/
def micheal_licks : ℕ := 63

/-- Theorem stating that Micheal takes 63 licks to reach the center of a lollipop,
    given the number of licks for Dan, Sam, David, Lance, and the average. -/
theorem micheal_licks_proof :
  (dan_licks + sam_licks + david_licks + lance_licks + micheal_licks) / 5 = average_licks :=
by sorry

end NUMINAMATH_CALUDE_micheal_licks_proof_l1868_186822


namespace NUMINAMATH_CALUDE_student_arrangement_l1868_186853

theorem student_arrangement (n m k : ℕ) (hn : n = 5) (hm : m = 4) (hk : k = 2) : 
  (Nat.choose m k / 2) * (Nat.factorial n / Nat.factorial (n - k)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_l1868_186853


namespace NUMINAMATH_CALUDE_shooter_probabilities_l1868_186883

/-- A shooter has a probability of hitting the target in a single shot -/
def hit_probability : ℝ := 0.5

/-- The number of shots taken -/
def num_shots : ℕ := 4

/-- The probability of hitting the target exactly k times in n shots -/
def prob_exact_hits (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

/-- The probability of hitting the target at least once in n shots -/
def prob_at_least_one_hit (n : ℕ) : ℝ :=
  1 - (1 - hit_probability) ^ n

theorem shooter_probabilities :
  (prob_exact_hits num_shots 3 = 1/4) ∧
  (prob_at_least_one_hit num_shots = 15/16) := by
  sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l1868_186883


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1868_186837

theorem diophantine_equation_solutions (n : ℕ) : n ∈ ({1, 2, 3} : Set ℕ) ↔ 
  ∃ (a b c : ℤ), a^n + b^n = c^n + n ∧ n ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1868_186837


namespace NUMINAMATH_CALUDE_min_value_expression_l1868_186828

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), 
    x^2 + y^2 + 1/x^2 + y/x ≥ m) ∧
  (∃ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), 
    x^2 + y^2 + 1/x^2 + y/x = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1868_186828


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_iff_rational_l1868_186882

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ := fun n ↦ a + n * d

/-- A geometric progression is a sequence where each term after the first
    is found by multiplying the previous term by a fixed, non-zero number. -/
def GeometricProgression (a r : ℚ) : ℕ → ℚ := fun n ↦ a * r^n

/-- A subsequence of a sequence is a sequence that can be derived from the original
    sequence by deleting some or no elements without changing the order of the
    remaining elements. -/
def Subsequence (f g : ℕ → ℚ) : Prop :=
  ∃ h : ℕ → ℕ, Monotone h ∧ ∀ n, f (h n) = g n

theorem arithmetic_to_geometric_iff_rational (a d : ℚ) (hd : d ≠ 0) :
  (∃ (b r : ℚ) (hr : r ≠ 1), Subsequence (ArithmeticProgression a d) (GeometricProgression b r)) ↔
  ∃ q : ℚ, a = q * d := by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_iff_rational_l1868_186882


namespace NUMINAMATH_CALUDE_stating_crabapple_sequence_count_l1868_186811

/-- Represents the number of students in the class -/
def num_students : ℕ := 11

/-- Represents the number of days the class meets -/
def num_days : ℕ := 3

/-- 
  Calculates the number of possible sequences for selecting students to receive a crabapple,
  given that no student can be selected on consecutive days.
-/
def crabapple_sequences (n : ℕ) (d : ℕ) : ℕ :=
  if d = 1 then n
  else if d = 2 then n * (n - 1)
  else n * (n - 1) * (n - 1)

/-- 
  Theorem stating that the number of possible sequences for selecting students
  to receive a crabapple over three days in a class of 11 students,
  where no student can be selected on consecutive days, is 1100.
-/
theorem crabapple_sequence_count :
  crabapple_sequences num_students num_days = 1100 := by
  sorry

end NUMINAMATH_CALUDE_stating_crabapple_sequence_count_l1868_186811


namespace NUMINAMATH_CALUDE_sheep_in_pen_l1868_186887

theorem sheep_in_pen (total : ℕ) (rounded_up : ℕ) (wandered_off : ℕ) : 
  wandered_off = 9 →
  wandered_off = total / 10 →
  rounded_up = total * 9 / 10 →
  rounded_up = 81 := by
sorry

end NUMINAMATH_CALUDE_sheep_in_pen_l1868_186887


namespace NUMINAMATH_CALUDE_maria_towels_result_l1868_186816

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def towels_maria_kept (green_towels white_towels towels_given : ℕ) : ℕ :=
  green_towels + white_towels - towels_given

/-- Theorem stating that Maria ended up with 22 towels. -/
theorem maria_towels_result :
  towels_maria_kept 35 21 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_maria_towels_result_l1868_186816


namespace NUMINAMATH_CALUDE_max_profit_scheme_l1868_186866

-- Define the variables
def bean_sprout_price : ℚ := 60
def dried_tofu_price : ℚ := 40
def bean_sprout_sell : ℚ := 80
def dried_tofu_sell : ℚ := 55
def total_units : ℕ := 200
def max_cost : ℚ := 10440

-- Define the profit function
def profit (bean_sprouts : ℕ) : ℚ :=
  (bean_sprout_sell - bean_sprout_price) * bean_sprouts + 
  (dried_tofu_sell - dried_tofu_price) * (total_units - bean_sprouts)

-- Theorem statement
theorem max_profit_scheme :
  ∀ bean_sprouts : ℕ,
  (2 * bean_sprout_price + 3 * dried_tofu_price = 240) →
  (3 * bean_sprout_price + 4 * dried_tofu_price = 340) →
  (bean_sprouts + (total_units - bean_sprouts) = total_units) →
  (bean_sprout_price * bean_sprouts + dried_tofu_price * (total_units - bean_sprouts) ≤ max_cost) →
  (bean_sprouts ≥ (3/2) * (total_units - bean_sprouts)) →
  profit bean_sprouts ≤ profit 122 ∧ profit 122 = 3610 := by
sorry

end NUMINAMATH_CALUDE_max_profit_scheme_l1868_186866


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l1868_186851

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) 
  (h2 : Real.cos (θ - Real.pi / 4) = 3 / 5) : 
  Real.tan (θ + Real.pi / 4) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l1868_186851


namespace NUMINAMATH_CALUDE_firstYearStudents2012_is_set_l1868_186803

/-- A type representing a student -/
structure Student :=
  (name : String)
  (year : Nat)
  (school : String)
  (enrollmentYear : Nat)

/-- Definition of a well-defined criterion for set membership -/
def hasWellDefinedCriterion (s : Set Student) : Prop :=
  ∀ x : Student, (x ∈ s) ∨ (x ∉ s)

/-- The set of all first-year high school students at a certain school in 2012 -/
def firstYearStudents2012 (school : String) : Set Student :=
  {s : Student | s.year = 1 ∧ s.school = school ∧ s.enrollmentYear = 2012}

/-- Theorem stating that the collection of first-year students in 2012 forms a set -/
theorem firstYearStudents2012_is_set (school : String) :
  hasWellDefinedCriterion (firstYearStudents2012 school) :=
sorry

end NUMINAMATH_CALUDE_firstYearStudents2012_is_set_l1868_186803


namespace NUMINAMATH_CALUDE_expression_value_l1868_186839

theorem expression_value : 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1868_186839


namespace NUMINAMATH_CALUDE_initial_crayons_count_l1868_186827

/-- 
Given a person who:
1. Has an initial number of crayons
2. Loses half of their crayons
3. Buys 20 new crayons
4. Ends up with 29 crayons total
This theorem proves that the initial number of crayons was 18.
-/
theorem initial_crayons_count (initial : ℕ) 
  (h1 : initial / 2 + 20 = 29) : initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l1868_186827


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l1868_186874

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_check : 
  ¬ is_pythagorean_triple 12 15 18 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 6 9 15 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l1868_186874


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1868_186859

theorem gcd_of_three_numbers : Nat.gcd 12222 (Nat.gcd 18333 36666) = 6111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1868_186859


namespace NUMINAMATH_CALUDE_inequality_solution_set_minimum_m_value_minimum_fraction_value_l1868_186821

def f (x : ℝ) := |x + 1| + |x - 1|

theorem inequality_solution_set :
  {x : ℝ | f x < 2*x + 3} = {x : ℝ | x > -1/2} := by sorry

theorem minimum_m_value :
  (∃ (x₀ : ℝ), f x₀ ≤ 2) ∧ 
  (∀ (m : ℝ), (∃ (x : ℝ), f x ≤ m) → m ≥ 2) := by sorry

theorem minimum_fraction_value :
  ∀ (a b : ℝ), a > 0 → b > 0 → 3*a + b = 2 →
  1/(2*a) + 1/(a+b) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_minimum_m_value_minimum_fraction_value_l1868_186821


namespace NUMINAMATH_CALUDE_binomial_10_3_l1868_186877

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l1868_186877


namespace NUMINAMATH_CALUDE_pasta_cost_is_one_dollar_l1868_186850

/-- The cost of pasta per box for Sam's spaghetti and meatballs dinner -/
def pasta_cost (total_cost sauce_cost meatballs_cost : ℚ) : ℚ :=
  total_cost - (sauce_cost + meatballs_cost)

/-- Theorem: The cost of pasta per box is $1.00 -/
theorem pasta_cost_is_one_dollar :
  pasta_cost 8 2 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pasta_cost_is_one_dollar_l1868_186850


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1868_186885

theorem smallest_multiplier_for_perfect_square (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 2 → ¬(∃ k : ℕ, 1152 * m = k * k)) ∧ 
  (∃ k : ℕ, 1152 * 2 = k * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1868_186885


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1868_186879

/-- Given a hyperbola and a parabola with specific properties, prove that the parameter p of the parabola equals 4. -/
theorem parabola_hyperbola_intersection (a b p k : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  a = 2*Real.sqrt 2 →  -- Real axis length
  b = p/2 →  -- Imaginary axis endpoint coincides with parabola focus
  (∀ x y, x^2 = 2*p*y) →  -- Parabola equation
  (∃ x y, y = k*x - 1 ∧ x^2 = 2*p*y) →  -- Line is tangent to parabola
  k = p/(4*Real.sqrt 2) →  -- Line is parallel to hyperbola asymptote
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1868_186879


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1868_186860

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal [true, false, true, true, true, true, false, false]) = [2, 3, 3, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1868_186860


namespace NUMINAMATH_CALUDE_lucy_flour_problem_l1868_186802

/-- The amount of flour Lucy had at the start of the week -/
def initial_flour : ℝ := 500

/-- The amount of flour Lucy used for baking cookies -/
def used_flour : ℝ := 240

/-- The amount of flour Lucy needs to buy to have a full bag -/
def flour_to_buy : ℝ := 370

theorem lucy_flour_problem :
  (initial_flour - used_flour) / 2 + flour_to_buy = initial_flour :=
by sorry

end NUMINAMATH_CALUDE_lucy_flour_problem_l1868_186802


namespace NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l1868_186872

theorem abs_x_less_than_2_sufficient_not_necessary :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  ¬(∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) := by
sorry

end NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l1868_186872


namespace NUMINAMATH_CALUDE_goals_scored_over_two_days_l1868_186844

/-- The total number of goals scored by Gina and Tom over two days -/
def total_goals (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ) : ℕ :=
  gina_day1 + gina_day2 + tom_day1 + tom_day2

/-- Theorem stating the total number of goals scored by Gina and Tom -/
theorem goals_scored_over_two_days :
  ∃ (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ),
    gina_day1 = 2 ∧
    tom_day1 = gina_day1 + 3 ∧
    tom_day2 = 6 ∧
    gina_day2 = tom_day2 - 2 ∧
    total_goals gina_day1 gina_day2 tom_day1 tom_day2 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_goals_scored_over_two_days_l1868_186844


namespace NUMINAMATH_CALUDE_probability_AR55_l1868_186835

/-- Represents the set of possible symbols for each position in the license plate -/
def LicensePlateSymbols : Fin 4 → Type
  | 0 => Fin 5  -- Vowels (A, E, I, O, U)
  | 1 => Fin 21 -- Non-vowels (consonants)
  | 2 => Fin 10 -- Digits (0-9)
  | 3 => Fin 10 -- Digits (0-9)

/-- The total number of possible license plates -/
def totalLicensePlates : ℕ := 5 * 21 * 10 * 10

/-- Represents a specific license plate -/
def SpecificPlate : Fin 4 → ℕ
  | 0 => 0  -- 'A' (first vowel)
  | 1 => 17 -- 'R' (18th consonant, 0-indexed)
  | 2 => 5  -- '5'
  | 3 => 5  -- '5'

/-- The probability of randomly selecting the license plate "AR55" -/
theorem probability_AR55 : 
  (1 : ℚ) / totalLicensePlates = 1 / 10500 :=
sorry

end NUMINAMATH_CALUDE_probability_AR55_l1868_186835


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1868_186889

theorem inequality_system_solution (x : ℝ) : 
  ((3*x - 2) / (x - 6) ≤ 1 ∧ 2*x^2 - x - 1 > 0) ↔ 
  ((-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6)) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1868_186889


namespace NUMINAMATH_CALUDE_systematic_sample_interval_count_l1868_186854

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  totalPopulation : ℕ
  sampleSize : ℕ
  intervalStart : ℕ
  intervalEnd : ℕ

/-- Calculates the number of selected items within a given interval in a systematic sample -/
def selectedInInterval (s : SystematicSample) : ℕ :=
  let stepSize := s.totalPopulation / s.sampleSize
  let intervalSize := s.intervalEnd - s.intervalStart + 1
  intervalSize / stepSize

/-- The main theorem statement -/
theorem systematic_sample_interval_count :
  let s : SystematicSample := {
    totalPopulation := 840,
    sampleSize := 21,
    intervalStart := 481,
    intervalEnd := 720
  }
  selectedInInterval s = 6 := by sorry

end NUMINAMATH_CALUDE_systematic_sample_interval_count_l1868_186854


namespace NUMINAMATH_CALUDE_triangle_property_l1868_186888

theorem triangle_property (A B C : Real) (a b c : Real) :
  2 * Real.sin (2 * A) * Real.cos A - Real.sin (3 * A) + Real.sqrt 3 * Real.cos A = Real.sqrt 3 →
  a = 1 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sin (2 * C) →
  A = π / 3 ∧ (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 6 :=
by sorry


end NUMINAMATH_CALUDE_triangle_property_l1868_186888


namespace NUMINAMATH_CALUDE_sum_of_squares_l1868_186815

theorem sum_of_squares (x y z : ℤ) : 
  x + y + 57 = 0 → y - z + 17 = 0 → x - z + 44 = 0 → x^2 + y^2 + z^2 = 1993 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1868_186815


namespace NUMINAMATH_CALUDE_associative_property_only_l1868_186894

theorem associative_property_only (a b c : ℕ) : 
  (a + b) + c = a + (b + c) ↔ 
  ∃ (x y z : ℕ), x + y + z = x + (y + z) ∧ x = 57 ∧ y = 24 ∧ z = 76 :=
by sorry

end NUMINAMATH_CALUDE_associative_property_only_l1868_186894


namespace NUMINAMATH_CALUDE_smallest_twin_prime_pair_mean_l1868_186833

/-- Twin prime pair -/
def is_twin_prime_pair (p q : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ q = p + 2

/-- Smallest twin prime pair -/
def smallest_twin_prime_pair (p q : Nat) : Prop :=
  is_twin_prime_pair p q ∧ ∀ (r s : Nat), is_twin_prime_pair r s → p ≤ r

/-- Arithmetic mean of two numbers -/
def arithmetic_mean (a b : Nat) : Rat :=
  (a + b : Rat) / 2

theorem smallest_twin_prime_pair_mean :
  ∃ (p q : Nat), smallest_twin_prime_pair p q ∧ arithmetic_mean p q = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_twin_prime_pair_mean_l1868_186833


namespace NUMINAMATH_CALUDE_intersection_union_when_a_2_complement_intersection_condition_l1868_186809

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | |x| < a}

theorem intersection_union_when_a_2 :
  A ∩ B 2 = {x | 1/2 ≤ x ∧ x < 2} ∧
  A ∪ B 2 = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_intersection_condition (a : ℝ) :
  (Aᶜ ∩ B a = B a) ↔ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_union_when_a_2_complement_intersection_condition_l1868_186809


namespace NUMINAMATH_CALUDE_students_with_A_or_B_l1868_186856

theorem students_with_A_or_B (fraction_A fraction_B : ℝ) 
  (h1 : fraction_A = 0.7)
  (h2 : fraction_B = 0.2) : 
  fraction_A + fraction_B = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_students_with_A_or_B_l1868_186856


namespace NUMINAMATH_CALUDE_protective_clothing_production_l1868_186838

/-- Represents the situation of a factory producing protective clothing --/
theorem protective_clothing_production 
  (total_production : ℕ) 
  (overtime_increase : ℚ) 
  (days_ahead : ℕ) 
  (x : ℚ) 
  (h1 : total_production = 1000) 
  (h2 : overtime_increase = 1/5) 
  (h3 : days_ahead = 2) 
  (h4 : x > 0) :
  (total_production / x) - (total_production / ((1 + overtime_increase) * x)) = days_ahead :=
sorry

end NUMINAMATH_CALUDE_protective_clothing_production_l1868_186838
