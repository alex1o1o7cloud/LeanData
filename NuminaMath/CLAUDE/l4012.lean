import Mathlib

namespace NUMINAMATH_CALUDE_medium_supermarkets_sample_l4012_401244

/-- Represents the number of supermarkets to be sampled in a stratified sampling method. -/
def stratified_sample (total_large : ℕ) (total_medium : ℕ) (total_small : ℕ) (sample_size : ℕ) : ℕ :=
  let total := total_large + total_medium + total_small
  (sample_size * total_medium) / total

/-- Theorem stating that the number of medium-sized supermarkets to be sampled is 20. -/
theorem medium_supermarkets_sample :
  stratified_sample 200 400 1400 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_medium_supermarkets_sample_l4012_401244


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l4012_401220

/-- A line passing through three points in a rectangular coordinate system -/
structure Line where
  p1 : Prod ℝ ℝ
  p2 : Prod ℝ ℝ
  p3 : Prod ℝ ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- Theorem: The x-intercept of the line passing through (10, 3), (-10, -7), and (5, 1) is 4 -/
theorem x_intercept_of_specific_line :
  let l : Line := { p1 := (10, 3), p2 := (-10, -7), p3 := (5, 1) }
  x_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l4012_401220


namespace NUMINAMATH_CALUDE_pie_slices_sold_today_l4012_401298

theorem pie_slices_sold_today (total : ℕ) (yesterday : ℕ) (today : ℕ) 
  (h1 : total = 7) 
  (h2 : yesterday = 5) 
  (h3 : total = yesterday + today) : 
  today = 2 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_sold_today_l4012_401298


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l4012_401280

/-- Given a line with slope 5 passing through (-2, 3), prove m + b = 18 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 5 → 
  3 = 5 * (-2) + b → 
  m + b = 18 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l4012_401280


namespace NUMINAMATH_CALUDE_quadratic_roots_l4012_401285

theorem quadratic_roots : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + x₁ = 0 ∧ x₂^2 + x₂ = 0) ∧ 
  x₁ = 0 ∧ x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l4012_401285


namespace NUMINAMATH_CALUDE_triangle_properties_l4012_401209

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a + b = 13 →
  c = 7 →
  4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7/2 →
  C = π/3 ∧ 
  π * (2 * (1/2 * a * b * Real.sin C) / (a + b + c))^2 = 3*π :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4012_401209


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l4012_401296

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 2 ↔ ∀ y : ℝ, 0 < y ∧ y < x → f y > f x :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l4012_401296


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l4012_401223

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total_products : ℕ
  sample_size : ℕ
  known_sample_number : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSampling) : ℕ :=
  let interval := s.total_products / s.sample_size
  let first_number := s.known_sample_number % interval
  first_number + (s.sample_size - 1) * interval

/-- Theorem stating the largest number in the sample for the given scenario -/
theorem largest_number_in_sample :
  let s : SystematicSampling := {
    total_products := 90,
    sample_size := 9,
    known_sample_number := 36
  }
  largest_sample_number s = 86 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_sample_l4012_401223


namespace NUMINAMATH_CALUDE_solutions_of_fourth_power_equation_l4012_401278

theorem solutions_of_fourth_power_equation :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_of_fourth_power_equation_l4012_401278


namespace NUMINAMATH_CALUDE_factory_workers_count_l4012_401261

/-- Proves the number of workers in a factory given certain salary information --/
theorem factory_workers_count :
  let initial_average : ℚ := 430
  let initial_supervisor_salary : ℚ := 870
  let new_average : ℚ := 390
  let new_supervisor_salary : ℚ := 510
  let total_people : ℕ := 9
  ∃ (workers : ℕ),
    (workers : ℚ) + 1 = (total_people : ℚ) ∧
    (workers + 1) * initial_average = workers * initial_average + initial_supervisor_salary ∧
    total_people * new_average = workers * initial_average + new_supervisor_salary ∧
    workers = 8 := by
  sorry

end NUMINAMATH_CALUDE_factory_workers_count_l4012_401261


namespace NUMINAMATH_CALUDE_exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary_l4012_401284

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure TwoDrawOutcome :=
  (first second : BallColor)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Finset TwoDrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : TwoDrawOutcome) : Prop := sorry

/-- The event of drawing exactly two red balls -/
def exactlyTwoRed (outcome : TwoDrawOutcome) : Prop := sorry

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (event1 event2 : TwoDrawOutcome → Prop) : Prop := sorry

/-- Two events are complementary if their union is the entire sample space -/
def complementary (event1 event2 : TwoDrawOutcome → Prop) : Prop := sorry

theorem exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoRed ∧
  ¬complementary exactlyOneBlack exactlyTwoRed := by sorry

end NUMINAMATH_CALUDE_exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary_l4012_401284


namespace NUMINAMATH_CALUDE_max_product_sum_300_l4012_401279

theorem max_product_sum_300 (a b : ℤ) (h : a + b = 300) :
  a * b ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l4012_401279


namespace NUMINAMATH_CALUDE_smallest_prime_after_four_nonprimes_l4012_401269

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if four consecutive natural numbers are all nonprime -/
def fourConsecutiveNonPrime (n : ℕ) : Prop :=
  ¬(isPrime n) ∧ ¬(isPrime (n + 1)) ∧ ¬(isPrime (n + 2)) ∧ ¬(isPrime (n + 3))

/-- The theorem stating that 29 is the smallest prime after four consecutive nonprimes -/
theorem smallest_prime_after_four_nonprimes :
  ∃ n : ℕ, fourConsecutiveNonPrime n ∧ isPrime (n + 4) ∧
  ∀ m : ℕ, m < n → ¬(fourConsecutiveNonPrime m ∧ isPrime (m + 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_four_nonprimes_l4012_401269


namespace NUMINAMATH_CALUDE_degrees_to_minutes_03_negative_comparison_l4012_401287

-- Define the conversion factor from degrees to minutes
def degrees_to_minutes (d : ℝ) : ℝ := d * 60

-- Theorem 1: 0.3 degrees is equal to 18 minutes
theorem degrees_to_minutes_03 : degrees_to_minutes 0.3 = 18 := by sorry

-- Theorem 2: -2 is greater than -3
theorem negative_comparison : -2 > -3 := by sorry

end NUMINAMATH_CALUDE_degrees_to_minutes_03_negative_comparison_l4012_401287


namespace NUMINAMATH_CALUDE_ladybug_count_l4012_401229

theorem ladybug_count (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) 
  (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_count_l4012_401229


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4012_401265

/-- Hyperbola M with equation x^2 - y^2/b^2 = 1 -/
def hyperbola_M (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 1

/-- Line l with slope 1 passing through the left vertex (-1, 0) -/
def line_l (x y : ℝ) : Prop :=
  y = x + 1

/-- Asymptotes of hyperbola M -/
def asymptotes_M (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 0

/-- Point A is the left vertex of hyperbola M -/
def point_A : ℝ × ℝ :=
  (-1, 0)

/-- Point B is the intersection of line l and one asymptote -/
def point_B (b : ℝ) : ℝ × ℝ :=
  sorry

/-- Point C is the intersection of line l and the other asymptote -/
def point_C (b : ℝ) : ℝ × ℝ :=
  sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The eccentricity of hyperbola M -/
def eccentricity (b : ℝ) : ℝ :=
  sorry

theorem hyperbola_eccentricity (b : ℝ) :
  hyperbola_M b (point_A.1) (point_A.2) →
  line_l (point_B b).1 (point_B b).2 →
  line_l (point_C b).1 (point_C b).2 →
  asymptotes_M b (point_B b).1 (point_B b).2 →
  asymptotes_M b (point_C b).1 (point_C b).2 →
  distance point_A (point_B b) = distance (point_B b) (point_C b) →
  eccentricity b = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4012_401265


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l4012_401294

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5. -/
theorem arithmetic_sequence_fourth_term (b y : ℝ) 
  (h : b + (b + 2*y) = 10) : b + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l4012_401294


namespace NUMINAMATH_CALUDE_min_relevant_number_l4012_401236

def A (n : ℕ) := Finset.range (2*n + 1) \ {0}

def is_relevant_number (n m : ℕ) : Prop :=
  n ≥ 2 ∧ m ≥ 4 ∧
  ∀ (P : Finset ℕ), P ⊆ A n → P.card = m →
    ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem min_relevant_number (n : ℕ) :
  n ≥ 2 → (∃ (m : ℕ), is_relevant_number n m) →
  ∃ (m : ℕ), is_relevant_number n m ∧ ∀ (k : ℕ), is_relevant_number n k → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_relevant_number_l4012_401236


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l4012_401295

theorem apple_pear_equivalence : 
  ∀ (apple_value pear_value : ℚ),
  (3/4 * 16 : ℚ) * apple_value = 10 * pear_value →
  (2/5 * 20 : ℚ) * apple_value = (20/3 : ℚ) * pear_value := by
sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l4012_401295


namespace NUMINAMATH_CALUDE_power_of_product_l4012_401211

theorem power_of_product (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4012_401211


namespace NUMINAMATH_CALUDE_symmetry_of_graphs_l4012_401292

theorem symmetry_of_graphs (f : ℝ → ℝ) (a : ℝ) :
  ∀ x y : ℝ, f (a - x) = y ↔ f (x - a) = y :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_graphs_l4012_401292


namespace NUMINAMATH_CALUDE_problem_solution_l4012_401273

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 2|

-- Define the solution set M
def M : Set ℝ := {x : ℝ | 2/3 ≤ x ∧ x ≤ 6}

-- Theorem statement
theorem problem_solution :
  (∀ x ∈ M, f x ≥ -1) ∧
  (∀ x ∉ M, f x < -1) ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 4*a + b + c = 6 →
    1/(2*a + b) + 1/(2*a + c) ≥ 2/3) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 4*a + b + c = 6 ∧
    1/(2*a + b) + 1/(2*a + c) = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l4012_401273


namespace NUMINAMATH_CALUDE_exactly_one_common_course_l4012_401202

/-- The number of ways two people can choose 2 courses each from 4 courses with exactly one course in common -/
def common_course_choices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose n k - Nat.choose n k - Nat.choose n k

theorem exactly_one_common_course :
  common_course_choices 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_common_course_l4012_401202


namespace NUMINAMATH_CALUDE_nunzio_pizza_consumption_l4012_401238

/-- Represents the number of pieces in a whole pizza -/
def pieces_per_pizza : ℕ := 8

/-- Represents the number of pizzas Nunzio eats in the given period -/
def total_pizzas : ℕ := 27

/-- Represents the number of days in the given period -/
def total_days : ℕ := 72

/-- Calculates the number of pizza pieces Nunzio eats per day -/
def pieces_per_day : ℕ := (total_pizzas * pieces_per_pizza) / total_days

/-- Theorem stating that Nunzio eats 3 pieces of pizza per day -/
theorem nunzio_pizza_consumption : pieces_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_nunzio_pizza_consumption_l4012_401238


namespace NUMINAMATH_CALUDE_daisy_seeds_count_l4012_401254

/-- The number of daisy seeds planted by Hortense -/
def daisy_seeds : ℕ := sorry

/-- The number of sunflower seeds planted by Hortense -/
def sunflower_seeds : ℕ := 25

/-- The percentage of daisy seeds that germinate -/
def daisy_germination_rate : ℚ := 60 / 100

/-- The percentage of sunflower seeds that germinate -/
def sunflower_germination_rate : ℚ := 80 / 100

/-- The percentage of germinated plants that produce flowers -/
def flower_production_rate : ℚ := 80 / 100

/-- The total number of plants that produce flowers -/
def total_flowering_plants : ℕ := 28

theorem daisy_seeds_count :
  (↑daisy_seeds * daisy_germination_rate * flower_production_rate +
   ↑sunflower_seeds * sunflower_germination_rate * flower_production_rate : ℚ) = total_flowering_plants ∧
  daisy_seeds = 25 :=
sorry

end NUMINAMATH_CALUDE_daisy_seeds_count_l4012_401254


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l4012_401201

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => x < 150 ∧ (x + 17) % 46 = 75 % 46) (Finset.range 150)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l4012_401201


namespace NUMINAMATH_CALUDE_rationalize_denominator_example_l4012_401266

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem rationalize_denominator_example : (7 : ℝ) / cubeRoot 343 = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_example_l4012_401266


namespace NUMINAMATH_CALUDE_tan_5040_degrees_equals_zero_l4012_401227

theorem tan_5040_degrees_equals_zero : Real.tan (5040 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_5040_degrees_equals_zero_l4012_401227


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l4012_401243

/-- The coordinates of the foci of an ellipse with equation x^2/10 + y^2 = 1 are (3,0) and (-3,0) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/10 + y^2 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ f₁ = (3, 0) ∧ f₂ = (-3, 0) ∧
    ∀ (f : ℝ × ℝ), f ∈ ellipse → f = f₁ ∨ f = f₂ :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l4012_401243


namespace NUMINAMATH_CALUDE_number_of_subsets_l4012_401270

theorem number_of_subsets (S : Set ℕ) : 
  (∃ (B : Set ℕ), {1, 2} ⊆ B ∧ B ⊆ {1, 2, 3}) ∧ 
  (∀ (B : Set ℕ), {1, 2} ⊆ B ∧ B ⊆ {1, 2, 3} → B = {1, 2} ∨ B = {1, 2, 3}) :=
by sorry

end NUMINAMATH_CALUDE_number_of_subsets_l4012_401270


namespace NUMINAMATH_CALUDE_ball_probability_l4012_401245

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 2)
  (h5 : red = 15)
  (h6 : purple = 3)
  (h7 : total = white + green + yellow + red + purple) :
  (white + green + yellow : ℚ) / total = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l4012_401245


namespace NUMINAMATH_CALUDE_two_digit_number_with_remainders_l4012_401233

theorem two_digit_number_with_remainders : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n % 9 = 7 ∧ 
  n % 7 = 5 ∧ 
  n % 3 = 1 ∧ 
  n = 61 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_with_remainders_l4012_401233


namespace NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l4012_401281

theorem larger_number_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 45) 
  (diff_eq : x - y = 7) : 
  max x y = 26 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l4012_401281


namespace NUMINAMATH_CALUDE_hyperbola_properties_l4012_401203

/-- Given a hyperbola C with the equation (x²/a²) - (y²/b²) = 1, where a > 0 and b > 0,
    real axis length 4√2, and eccentricity √6/2, prove the following statements. -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_real_axis : 2 * a = 4 * Real.sqrt 2)
  (h_eccentricity : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 6 / 2) :
  /- 1. The standard equation is x²/8 - y²/4 = 1 -/
  (a^2 = 8 ∧ b^2 = 4) ∧ 
  /- 2. The locus equation of the midpoint Q of AP, where A(3,0) and P is any point on C,
        is ((2x - 3)²/8) - y² = 1 -/
  (∀ x y : ℝ, ((2*x - 3)^2 / 8) - y^2 = 1 ↔ 
    ∃ px py : ℝ, (px^2 / a^2) - (py^2 / b^2) = 1 ∧ x = (px + 3) / 2 ∧ y = py / 2) ∧
  /- 3. The minimum value of |AP| is 3 - 2√2 -/
  (∀ px py : ℝ, (px^2 / a^2) - (py^2 / b^2) = 1 → 
    Real.sqrt ((px - 3)^2 + py^2) ≥ 3 - 2 * Real.sqrt 2) ∧
  (∃ px py : ℝ, (px^2 / a^2) - (py^2 / b^2) = 1 ∧ 
    Real.sqrt ((px - 3)^2 + py^2) = 3 - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l4012_401203


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l4012_401288

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (2 - |x|) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l4012_401288


namespace NUMINAMATH_CALUDE_work_increase_percentage_l4012_401283

theorem work_increase_percentage (p : ℕ) (W : ℝ) (h : p > 0) :
  let original_work_per_person := W / p
  let remaining_persons := (2 : ℝ) / 3 * p
  let new_work_per_person := W / remaining_persons
  (new_work_per_person - original_work_per_person) / original_work_per_person * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_work_increase_percentage_l4012_401283


namespace NUMINAMATH_CALUDE_sandys_earnings_l4012_401224

/-- Calculates the total earnings for a given hourly rate and working hours over three days -/
def total_earnings (hourly_rate : ℕ) (hours_day1 hours_day2 hours_day3 : ℕ) : ℕ :=
  hourly_rate * (hours_day1 + hours_day2 + hours_day3)

/-- Proves that Sandy's total earnings for the three days are $450 -/
theorem sandys_earnings :
  total_earnings 15 10 6 14 = 450 := by
  sorry

#eval total_earnings 15 10 6 14

end NUMINAMATH_CALUDE_sandys_earnings_l4012_401224


namespace NUMINAMATH_CALUDE_usual_bus_time_l4012_401239

/-- Proves that the usual time to catch the bus is 12 minutes, given that walking
    at 4/5 of the usual speed results in missing the bus by 3 minutes. -/
theorem usual_bus_time (T : ℝ) (h : (5 / 4) * T = T + 3) : T = 12 := by
  sorry

end NUMINAMATH_CALUDE_usual_bus_time_l4012_401239


namespace NUMINAMATH_CALUDE_factor_expression_l4012_401255

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4012_401255


namespace NUMINAMATH_CALUDE_square_fits_in_unit_cube_l4012_401290

theorem square_fits_in_unit_cube : ∃ (s : ℝ), s ≥ 1.05 ∧ 
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 
    s = min (Real.sqrt (2 * (1 - x)^2)) (Real.sqrt (1 + 2 * x^2)) :=
sorry

end NUMINAMATH_CALUDE_square_fits_in_unit_cube_l4012_401290


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4012_401250

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - 4| ≥ a) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4012_401250


namespace NUMINAMATH_CALUDE_ellipse_properties_l4012_401246

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eccentricity : (a^2 - b^2) / a^2 = 3/4
  h_point_on_ellipse : 2/a^2 + 1/(2*b^2) = 1

/-- The theorem statement -/
theorem ellipse_properties (C : Ellipse) :
  C.a^2 = 4 ∧ C.b^2 = 2 ∧
  (∀ (P Q : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    P ∈ l ∧ Q ∈ l ∧
    P.1^2/4 + P.2^2 = 1 ∧
    Q.1^2/4 + Q.2^2 = 1 ∧
    P.1 * Q.1 + P.2 * Q.2 = 0 →
    1/2 * abs (P.1 * Q.2 - P.2 * Q.1) ≥ 4/5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4012_401246


namespace NUMINAMATH_CALUDE_perimeter_of_non_shaded_region_l4012_401228

/-- A structure representing the geometrical figure described in the problem -/
structure Figure where
  outer_rectangle_length : ℝ
  outer_rectangle_width : ℝ
  small_rectangle_side : ℝ
  shaded_square_side : ℝ
  shaded_rectangle_length : ℝ
  shaded_rectangle_width : ℝ
  shaded_area : ℝ

/-- The theorem statement based on the problem -/
theorem perimeter_of_non_shaded_region
  (fig : Figure)
  (h1 : fig.outer_rectangle_length = 12)
  (h2 : fig.outer_rectangle_width = 9)
  (h3 : fig.small_rectangle_side = 3)
  (h4 : fig.shaded_square_side = 3)
  (h5 : fig.shaded_rectangle_length = 3)
  (h6 : fig.shaded_rectangle_width = 2)
  (h7 : fig.shaded_area = 65)
  : ∃ (p : ℝ), p = 30 ∧ p = 2 * (12 + 3) :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_non_shaded_region_l4012_401228


namespace NUMINAMATH_CALUDE_community_service_arrangements_l4012_401206

def arrange_people (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem community_service_arrangements : 
  arrange_people 6 4 + arrange_people 6 3 + arrange_people 6 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_community_service_arrangements_l4012_401206


namespace NUMINAMATH_CALUDE_solve_system_1_solve_system_2_solve_inequality_solve_inequality_system_l4012_401282

-- System of linear equations 1
theorem solve_system_1 (x y : ℝ) : 
  x = 7 * y ∧ 2 * x + y = 30 → x = 14 ∧ y = 2 := by sorry

-- System of linear equations 2
theorem solve_system_2 (x y : ℝ) : 
  x / 2 + y / 3 = 7 ∧ x / 3 - y / 4 = -1 → x = 6 ∧ y = 12 := by sorry

-- Linear inequality
theorem solve_inequality (x : ℝ) :
  4 + 3 * (x - 1) > -5 ↔ x > -2 := by sorry

-- System of linear inequalities
theorem solve_inequality_system (x : ℝ) :
  (1 / 2 * (x - 2) + 3 > 7 ∧ -1 / 3 * (x + 3) - 4 > -10) ↔ (x > 10 ∧ x < 15) := by sorry

end NUMINAMATH_CALUDE_solve_system_1_solve_system_2_solve_inequality_solve_inequality_system_l4012_401282


namespace NUMINAMATH_CALUDE_part_one_part_two_l4012_401249

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - (a + 1/a)*x + 1 < 0

def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Theorem for part (1)
theorem part_one (a x : ℝ) (h1 : a = 2) (h2 : a > 1) (h3 : p x a ∧ q x) :
  1 ≤ x ∧ x < 2 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) (h : a > 1)
  (h_necessary : ∀ x, q x → p x a)
  (h_not_sufficient : ∃ x, p x a ∧ ¬q x) :
  3 < a := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4012_401249


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4012_401259

/-- Given a geometric sequence where the first term is 3 and the second term is 6,
    prove that the seventh term is 192. -/
theorem seventh_term_of_geometric_sequence :
  ∀ (a : ℕ → ℝ), 
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →
    a 7 = 192 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4012_401259


namespace NUMINAMATH_CALUDE_complex_division_problem_l4012_401214

theorem complex_division_problem (a : ℝ) (h : (a^2 - 9 : ℂ) + (a + 3 : ℂ) * I = (0 : ℂ) + b * I) :
  (a + I^19) / (1 + I) = 1 - 2*I :=
sorry

end NUMINAMATH_CALUDE_complex_division_problem_l4012_401214


namespace NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_9000_l4012_401213

theorem count_primes_with_squares_between_5000_and_9000 :
  ∃ (S : Finset Nat),
    (∀ p ∈ S, Nat.Prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧
    (∀ p : Nat, Nat.Prime p → 5000 ≤ p^2 → p^2 ≤ 9000 → p ∈ S) ∧
    Finset.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_9000_l4012_401213


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l4012_401234

/-- An arithmetic sequence with first term 2 and last term 3 -/
def is_arithmetic_sequence (x y : ℝ) : Prop :=
  x - 2 = 3 - y ∧ y - x = 3 - y

/-- A geometric sequence with first term 2 and last term 3 -/
def is_geometric_sequence (m n : ℝ) : Prop :=
  m / 2 = 3 / n ∧ n / m = 3 / n

theorem arithmetic_geometric_sum (x y m n : ℝ) 
  (h1 : is_arithmetic_sequence x y) 
  (h2 : is_geometric_sequence m n) : 
  x + y + m * n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l4012_401234


namespace NUMINAMATH_CALUDE_chunks_for_two_dozen_bananas_l4012_401263

/-- The number of chunks needed to purchase a given number of bananas -/
def chunks_needed (bananas : ℚ) : ℚ :=
  (bananas * 3 * 8) / (7 * 5)

theorem chunks_for_two_dozen_bananas :
  chunks_needed 24 = 576 / 35 := by
  sorry

end NUMINAMATH_CALUDE_chunks_for_two_dozen_bananas_l4012_401263


namespace NUMINAMATH_CALUDE_fraction_subtraction_result_l4012_401251

theorem fraction_subtraction_result : 
  (3 * 5 + 5 * 7 + 7 * 9) / (2 * 4 + 4 * 6 + 6 * 8) - 
  (2 * 4 + 4 * 6 + 6 * 8) / (3 * 5 + 5 * 7 + 7 * 9) = 74 / 119 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_result_l4012_401251


namespace NUMINAMATH_CALUDE_pear_mango_weight_equivalence_l4012_401276

/-- Given that 9 pears weigh the same as 6 mangoes, 
    prove that 36 pears weigh the same as 24 mangoes. -/
theorem pear_mango_weight_equivalence 
  (pear_weight mango_weight : ℝ) 
  (h : 9 * pear_weight = 6 * mango_weight) : 
  36 * pear_weight = 24 * mango_weight := by
  sorry

end NUMINAMATH_CALUDE_pear_mango_weight_equivalence_l4012_401276


namespace NUMINAMATH_CALUDE_candy_probability_l4012_401222

/-- The number of red candies initially in the jar -/
def red_candies : ℕ := 15

/-- The number of blue candies initially in the jar -/
def blue_candies : ℕ := 15

/-- The total number of candies initially in the jar -/
def total_candies : ℕ := red_candies + blue_candies

/-- The number of candies Terry picks -/
def terry_picks : ℕ := 3

/-- The number of candies Mary picks -/
def mary_picks : ℕ := 2

/-- The probability that Terry and Mary pick candies of the same color -/
def same_color_probability : ℚ := 8008 / 142221

theorem candy_probability : 
  same_color_probability = 
    (Nat.choose red_candies terry_picks * Nat.choose (red_candies - terry_picks) mary_picks + 
     Nat.choose blue_candies terry_picks * Nat.choose (blue_candies - terry_picks) mary_picks) / 
    (Nat.choose total_candies terry_picks * Nat.choose (total_candies - terry_picks) mary_picks) :=
sorry

end NUMINAMATH_CALUDE_candy_probability_l4012_401222


namespace NUMINAMATH_CALUDE_indigo_restaurant_rating_l4012_401231

/-- The average star rating for a restaurant given the number of reviews for each star rating. -/
def averageStarRating (fiveStars fourStars threeStars twoStars : ℕ) : ℚ :=
  let totalStars := 5 * fiveStars + 4 * fourStars + 3 * threeStars + 2 * twoStars
  let totalReviews := fiveStars + fourStars + threeStars + twoStars
  (totalStars : ℚ) / totalReviews

/-- Theorem stating that the average star rating for Indigo Restaurant is 4 stars. -/
theorem indigo_restaurant_rating :
  averageStarRating 6 7 4 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_indigo_restaurant_rating_l4012_401231


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l4012_401291

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x + 1
def line2 (x y : ℝ) : Prop := 5 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 99 / 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l4012_401291


namespace NUMINAMATH_CALUDE_cards_after_exchange_and_giveaway_l4012_401262

/-- Represents the number of cards in a box for each sport --/
structure CardCounts where
  basketball : ℕ
  baseball : ℕ
  football : ℕ
  hockey : ℕ
  soccer : ℕ

/-- Represents the number of boxes for each sport --/
structure BoxCounts where
  basketball : ℕ
  baseball : ℕ
  football : ℕ
  hockey : ℕ
  soccer : ℕ

/-- Calculate the total number of cards --/
def totalCards (cards : CardCounts) (boxes : BoxCounts) : ℕ :=
  cards.basketball * boxes.basketball +
  cards.baseball * boxes.baseball +
  cards.football * boxes.football +
  cards.hockey * boxes.hockey +
  cards.soccer * boxes.soccer

/-- The number of cards exchanged between Ben and Alex --/
def exchangedCards (cards : CardCounts) (boxes : BoxCounts) : ℕ :=
  (cards.basketball / 2) * boxes.basketball +
  (cards.baseball / 2) * boxes.baseball

theorem cards_after_exchange_and_giveaway 
  (ben_cards : CardCounts)
  (ben_boxes : BoxCounts)
  (alex_cards : CardCounts)
  (alex_boxes : BoxCounts)
  (h1 : ben_cards.basketball = 20)
  (h2 : ben_cards.baseball = 15)
  (h3 : ben_cards.football = 12)
  (h4 : ben_boxes.basketball = 8)
  (h5 : ben_boxes.baseball = 10)
  (h6 : ben_boxes.football = 12)
  (h7 : alex_cards.hockey = 15)
  (h8 : alex_cards.soccer = 18)
  (h9 : alex_boxes.hockey = 6)
  (h10 : alex_boxes.soccer = 9)
  (cards_given_away : ℕ)
  (h11 : cards_given_away = 175) :
  totalCards ben_cards ben_boxes + totalCards alex_cards alex_boxes - cards_given_away = 531 := by
  sorry


end NUMINAMATH_CALUDE_cards_after_exchange_and_giveaway_l4012_401262


namespace NUMINAMATH_CALUDE_A_intersect_B_l4012_401207

def A : Set ℤ := {1, 2, 3, 4}

def B : Set ℤ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem A_intersect_B : A ∩ B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l4012_401207


namespace NUMINAMATH_CALUDE_x_sixth_power_is_one_l4012_401216

theorem x_sixth_power_is_one (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_power_is_one_l4012_401216


namespace NUMINAMATH_CALUDE_triangle_angle_sine_equivalence_l4012_401217

theorem triangle_angle_sine_equivalence (A B C : Real) (h : A > 0 ∧ B > 0 ∧ C > 0) :
  (A > B ↔ Real.sin A > Real.sin B) :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_equivalence_l4012_401217


namespace NUMINAMATH_CALUDE_range_of_a_l4012_401297

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + a^2 > 0

-- Define the range of a
def range_a : Set ℝ := Set.Icc (-2) (-1) ∪ Set.Ioo 2 6

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4012_401297


namespace NUMINAMATH_CALUDE_toms_out_of_pocket_cost_l4012_401242

theorem toms_out_of_pocket_cost 
  (visit_cost : ℝ) 
  (cast_cost : ℝ) 
  (insurance_coverage_percentage : ℝ) 
  (h1 : visit_cost = 300)
  (h2 : cast_cost = 200)
  (h3 : insurance_coverage_percentage = 60) :
  let total_cost := visit_cost + cast_cost
  let insurance_coverage := (insurance_coverage_percentage / 100) * total_cost
  let out_of_pocket_cost := total_cost - insurance_coverage
  out_of_pocket_cost = 200 := by
sorry

end NUMINAMATH_CALUDE_toms_out_of_pocket_cost_l4012_401242


namespace NUMINAMATH_CALUDE_equation_solution_l4012_401218

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ * (x₁ + 2) = -3 * (x₁ + 2)) ∧ 
  (x₂ * (x₂ + 2) = -3 * (x₂ + 2)) ∧ 
  x₁ = -2 ∧ x₂ = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4012_401218


namespace NUMINAMATH_CALUDE_surface_area_of_cuboid_from_cubes_l4012_401230

/-- The surface area of a cuboid formed by three cubes in a row -/
theorem surface_area_of_cuboid_from_cubes (cube_side_length : ℝ) (h : cube_side_length = 8) : 
  let cuboid_length : ℝ := 3 * cube_side_length
  let cuboid_width : ℝ := cube_side_length
  let cuboid_height : ℝ := cube_side_length
  2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 896 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_cuboid_from_cubes_l4012_401230


namespace NUMINAMATH_CALUDE_expression_value_l4012_401267

theorem expression_value : 
  (20-19+18-17+16-15+14-13+12-11+10-9+8-7+6-5+4-3+2-1) / 
  (1-2+3-4+5-6+7-8+9-10+11-12+13-14+15-16+17-18+19-20) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4012_401267


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l4012_401277

/-- Given a circle ρ = 2cos θ, and a point Q on the extension of a chord OP such that OP/PQ = 2/3,
    prove that the trajectory of Q is a circle with equation ρ = 5cos θ. -/
theorem trajectory_of_Q (θ : Real) (ρ ρ_0 : Real → Real) :
  (∀ θ, ρ_0 θ = 2 * Real.cos θ) →  -- Given circle equation
  (∀ θ, ρ_0 θ / (ρ θ - ρ_0 θ) = 2 / 3) →  -- Ratio condition
  (∀ θ, ρ θ = 5 * Real.cos θ) :=  -- Trajectory equation to prove
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l4012_401277


namespace NUMINAMATH_CALUDE_stock_percent_change_l4012_401257

theorem stock_percent_change (initial_value : ℝ) : 
  let day1_value := initial_value * (1 - 0.1)
  let day2_value := day1_value * (1 + 0.2)
  (day2_value - initial_value) / initial_value = 0.08 := by
sorry

end NUMINAMATH_CALUDE_stock_percent_change_l4012_401257


namespace NUMINAMATH_CALUDE_arrange_four_on_eight_l4012_401215

/-- The number of ways to arrange n people on m chairs in a row,
    such that no two people sit next to each other -/
def arrangePeople (n m : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 120 ways to arrange 4 people on 8 chairs in a row,
    such that no two people sit next to each other -/
theorem arrange_four_on_eight :
  arrangePeople 4 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_four_on_eight_l4012_401215


namespace NUMINAMATH_CALUDE_construction_materials_l4012_401235

theorem construction_materials (concrete stone total : Real) 
  (h1 : concrete = 0.17)
  (h2 : stone = 0.5)
  (h3 : total = 0.83) :
  total - (concrete + stone) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_l4012_401235


namespace NUMINAMATH_CALUDE_simplify_expression_l4012_401208

theorem simplify_expression (x : ℝ) : (3*x)^4 - (2*x)*(x^3) = 79*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4012_401208


namespace NUMINAMATH_CALUDE_degrees_120_45_equals_120_75_l4012_401237

/-- Converts degrees and minutes to decimal degrees -/
def degreesMinutesToDecimal (degrees : ℕ) (minutes : ℕ) : ℚ :=
  degrees + (minutes : ℚ) / 60

/-- Theorem stating that 120°45' is equal to 120.75° -/
theorem degrees_120_45_equals_120_75 :
  degreesMinutesToDecimal 120 45 = 120.75 := by
  sorry

end NUMINAMATH_CALUDE_degrees_120_45_equals_120_75_l4012_401237


namespace NUMINAMATH_CALUDE_shortest_side_length_l4012_401241

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radius of inscribed circle
  r : ℝ
  -- Segments of side 'a' divided by tangent point
  a1 : ℝ
  a2 : ℝ
  -- Conditions
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < a1 ∧ 0 < a2
  tangent_point : a = a1 + a2
  radius : r = 5
  side_sum : b + c = 36
  segments : a1 = 7 ∧ a2 = 9

/-- The length of the shortest side in the triangle is 14 units -/
theorem shortest_side_length (t : TriangleWithInscribedCircle) : 
  min t.a (min t.b t.c) = 14 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l4012_401241


namespace NUMINAMATH_CALUDE_no_five_coprime_two_digit_composites_l4012_401204

theorem no_five_coprime_two_digit_composites : 
  ¬ ∃ (a b c d e : ℕ), 
    (10 ≤ a ∧ a < 100 ∧ ¬ Nat.Prime a) ∧
    (10 ≤ b ∧ b < 100 ∧ ¬ Nat.Prime b) ∧
    (10 ≤ c ∧ c < 100 ∧ ¬ Nat.Prime c) ∧
    (10 ≤ d ∧ d < 100 ∧ ¬ Nat.Prime d) ∧
    (10 ≤ e ∧ e < 100 ∧ ¬ Nat.Prime e) ∧
    (Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ Nat.Coprime a e ∧
     Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime b e ∧
     Nat.Coprime c d ∧ Nat.Coprime c e ∧
     Nat.Coprime d e) :=
by
  sorry


end NUMINAMATH_CALUDE_no_five_coprime_two_digit_composites_l4012_401204


namespace NUMINAMATH_CALUDE_two_integers_problem_l4012_401289

theorem two_integers_problem (x y : ℕ+) :
  (x / Nat.gcd x y + y / Nat.gcd x y : ℚ) = 18 →
  Nat.lcm x y = 975 →
  (x = 75 ∧ y = 195) ∨ (x = 195 ∧ y = 75) := by
  sorry

end NUMINAMATH_CALUDE_two_integers_problem_l4012_401289


namespace NUMINAMATH_CALUDE_complete_square_sum_l4012_401252

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l4012_401252


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l4012_401212

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice.choose 1 * prob_one * (prob_not_one ^ 2)) +
  2 * (num_dice.choose 2 * (prob_one ^ 2) * prob_not_one) +
  3 * (prob_one ^ num_dice)

theorem expected_ones_is_half : expected_ones = 1/2 := by sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l4012_401212


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l4012_401240

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio 5:2,
    prove that its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l4012_401240


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_forty_l4012_401264

theorem prime_square_minus_one_divisible_by_forty (p : ℕ) 
  (h_prime : Nat.Prime p) (h_geq_seven : p ≥ 7) : 
  40 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_forty_l4012_401264


namespace NUMINAMATH_CALUDE_apple_pile_count_l4012_401247

theorem apple_pile_count (initial_apples added_apples : ℕ) 
  (h1 : initial_apples = 8)
  (h2 : added_apples = 5) : 
  initial_apples + added_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_apple_pile_count_l4012_401247


namespace NUMINAMATH_CALUDE_largest_value_l4012_401299

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 15372 + 2/3074)
  (hb : b = 15372 - 2/3074)
  (hc : c = 15372 / (2/3074))
  (hd : d = 15372 * (2/3074))
  (he : e = 15372.3074) :
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l4012_401299


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l4012_401225

/-- An arithmetic sequence with a_5 = 15 -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 5 = 15

/-- Theorem: In an arithmetic sequence where a_5 = 15, the sum of a_2, a_4, a_6, and a_8 is 60 -/
theorem sum_of_specific_terms (a : ℕ → ℝ) (h : arithmeticSequence a) :
  a 2 + a 4 + a 6 + a 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l4012_401225


namespace NUMINAMATH_CALUDE_sum_of_exterior_angles_constant_l4012_401260

/-- A convex polygon with n sides, where n ≥ 3 -/
structure ConvexPolygon where
  n : ℕ
  sides_ge_three : n ≥ 3

/-- The sum of exterior angles of a convex polygon -/
def sum_of_exterior_angles (p : ConvexPolygon) : ℝ := sorry

/-- Theorem: The sum of exterior angles of any convex polygon is 360° -/
theorem sum_of_exterior_angles_constant (p : ConvexPolygon) :
  sum_of_exterior_angles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_of_exterior_angles_constant_l4012_401260


namespace NUMINAMATH_CALUDE_inequality_proof_l4012_401200

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (h : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4012_401200


namespace NUMINAMATH_CALUDE_negative_terms_min_value_at_min_value_l4012_401258

/-- The sequence a_n defined as n^2 - 5n + 4 -/
def a_n (n : ℝ) : ℝ := n^2 - 5*n + 4

/-- There are exactly two integer values of n for which a_n < 0 -/
theorem negative_terms : ∃! (s : Finset ℤ), (∀ n ∈ s, a_n n < 0) ∧ s.card = 2 :=
sorry

/-- The minimum value of a_n occurs when n = 5/2 -/
theorem min_value_at : ∀ n : ℝ, a_n n ≥ a_n (5/2) :=
sorry

/-- The minimum value of a_n is -1/4 -/
theorem min_value : a_n (5/2) = -1/4 :=
sorry

end NUMINAMATH_CALUDE_negative_terms_min_value_at_min_value_l4012_401258


namespace NUMINAMATH_CALUDE_list_property_l4012_401293

theorem list_property (list : List ℝ) (n : ℝ) : 
  list.Nodup →
  n ∈ list →
  n = 4 * ((list.sum - n) / (list.length - 1)) →
  n = (1 / 6) * list.sum →
  list.length = 21 := by
  sorry

end NUMINAMATH_CALUDE_list_property_l4012_401293


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l4012_401253

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l4012_401253


namespace NUMINAMATH_CALUDE_max_y_coordinate_l4012_401286

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l4012_401286


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l4012_401248

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k, n! + 1 < k ∧ k < n! + n → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l4012_401248


namespace NUMINAMATH_CALUDE_age_difference_l4012_401221

theorem age_difference (father_age : ℕ) (son_age_5_years_ago : ℕ) :
  father_age = 38 →
  son_age_5_years_ago = 14 →
  father_age - (son_age_5_years_ago + 5) = 19 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l4012_401221


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l4012_401272

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 :=
by sorry

theorem min_value_achieved (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (p₀ q₀ r₀ : ℝ), p₀ > 0 ∧ q₀ > 0 ∧ r₀ > 0 ∧
    4 * p₀^3 + 6 * q₀^3 + 24 * r₀^3 + 8 / (3 * p₀ * q₀ * r₀) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l4012_401272


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l4012_401271

/-- Given 5 consecutive even numbers whose sum is 240, prove that the smallest of these numbers is 44 -/
theorem smallest_of_five_consecutive_even_numbers (n : ℕ) : 
  (∃ (a b c d : ℕ), 
    a = n + 2 ∧ 
    b = n + 4 ∧ 
    c = n + 6 ∧ 
    d = n + 8 ∧ 
    n + a + b + c + d = 240 ∧ 
    Even n ∧ Even a ∧ Even b ∧ Even c ∧ Even d) → 
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l4012_401271


namespace NUMINAMATH_CALUDE_not_a_implies_condition_l4012_401256

/-- Represents a student in the course -/
structure Student :=
  (name : String)

/-- Represents the exam result for a student -/
structure ExamResult :=
  (student : Student)
  (allMultipleChoiceCorrect : Bool)
  (essayScore : ℝ)
  (receivedA : Bool)

/-- The professor's grading policy -/
axiom grading_policy : 
  ∀ (result : ExamResult), 
    result.allMultipleChoiceCorrect ∧ result.essayScore ≥ 80 → result.receivedA

/-- The theorem to be proved -/
theorem not_a_implies_condition (result : ExamResult) : 
  ¬result.receivedA → ¬result.allMultipleChoiceCorrect ∨ result.essayScore < 80 :=
sorry

end NUMINAMATH_CALUDE_not_a_implies_condition_l4012_401256


namespace NUMINAMATH_CALUDE_tinas_pens_l4012_401268

theorem tinas_pens (pink green blue : ℕ) : 
  pink = 12 ∧ 
  green = pink - 9 ∧ 
  blue = green + 3 → 
  pink + green + blue = 21 := by
  sorry

end NUMINAMATH_CALUDE_tinas_pens_l4012_401268


namespace NUMINAMATH_CALUDE_shelby_driving_time_l4012_401232

/-- Represents the driving scenario for Shelby --/
structure DrivingScenario where
  sunnySpeed : ℝ  -- Speed in miles per hour when not raining
  rainySpeed : ℝ  -- Speed in miles per hour when raining
  totalDistance : ℝ  -- Total distance traveled in miles
  totalTime : ℝ  -- Total time traveled in hours

/-- Calculates the time spent driving in the rain --/
def timeInRain (scenario : DrivingScenario) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the time spent driving in the rain is 40 minutes --/
theorem shelby_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.sunnySpeed = 40)
  (h2 : scenario.rainySpeed = 25)
  (h3 : scenario.totalDistance = 20)
  (h4 : scenario.totalTime = 0.75)  -- 45 minutes in hours
  : timeInRain scenario = 40 / 60 := by
  sorry


end NUMINAMATH_CALUDE_shelby_driving_time_l4012_401232


namespace NUMINAMATH_CALUDE_round_robin_tournament_participants_l4012_401205

/-- The number of games played in a round-robin tournament with n participants -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin chess tournament where 190 games were played, there were 20 participants -/
theorem round_robin_tournament_participants : ∃ n : ℕ, gamesPlayed n = 190 ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_participants_l4012_401205


namespace NUMINAMATH_CALUDE_machine_year_production_l4012_401275

/-- A machine that produces items at a constant rate. -/
structure Machine where
  production_rate : ℕ  -- Items produced per hour

/-- Represents a year with a fixed number of days. -/
structure Year where
  days : ℕ

/-- Calculates the total number of units produced by a machine in a year. -/
def units_produced (m : Machine) (y : Year) : ℕ :=
  m.production_rate * y.days * 24

/-- Theorem stating that a machine producing one item per hour will make 8760 units in a year of 365 days. -/
theorem machine_year_production :
  ∀ (m : Machine) (y : Year),
    m.production_rate = 1 →
    y.days = 365 →
    units_produced m y = 8760 :=
by
  sorry


end NUMINAMATH_CALUDE_machine_year_production_l4012_401275


namespace NUMINAMATH_CALUDE_base_b_sum_product_l4012_401226

/-- Given a base b, this function converts a number from base b to decimal --/
def toDecimal (b : ℕ) (n : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a decimal number to base b --/
def fromDecimal (b : ℕ) (n : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the given product and sum in base b --/
theorem base_b_sum_product (b : ℕ) : 
  (toDecimal b 14) * (toDecimal b 17) * (toDecimal b 18) = toDecimal b 6274 →
  (toDecimal b 14) + (toDecimal b 17) + (toDecimal b 18) = 49 := by
  sorry

end NUMINAMATH_CALUDE_base_b_sum_product_l4012_401226


namespace NUMINAMATH_CALUDE_inverse_function_value_l4012_401274

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State the theorem
theorem inverse_function_value 
  (h1 : f 3 = 8) 
  (h2 : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f) :
  f_inv 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_value_l4012_401274


namespace NUMINAMATH_CALUDE_symmetric_even_function_implies_odd_l4012_401210

/-- A function satisfying certain symmetry and evenness properties -/
def SymmetricEvenFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1) + f (1 - x) = 0) ∧ 
  (∀ x, f (x + 1) = f (-x + 1)) ∧
  (f (-3/2) = 1)

/-- The main theorem stating that f(x-2) is an odd function -/
theorem symmetric_even_function_implies_odd (f : ℝ → ℝ) 
  (h : SymmetricEvenFunction f) : 
  ∀ x, f (-(x - 2)) = -f (x - 2) := by
sorry

end NUMINAMATH_CALUDE_symmetric_even_function_implies_odd_l4012_401210


namespace NUMINAMATH_CALUDE_exponent_operations_l4012_401219

theorem exponent_operations (a : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3*a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_exponent_operations_l4012_401219
