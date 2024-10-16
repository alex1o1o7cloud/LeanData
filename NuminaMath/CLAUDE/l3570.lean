import Mathlib

namespace NUMINAMATH_CALUDE_partnership_profit_l3570_357081

/-- Given the investment ratio and B's share of profit, calculate the total profit --/
theorem partnership_profit (a b c : ℕ) (b_share : ℕ) (h1 : a = 6) (h2 : b = 2) (h3 : c = 3) (h4 : b_share = 800) :
  (b_share / b) * (a + b + c) = 4400 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l3570_357081


namespace NUMINAMATH_CALUDE_business_value_l3570_357018

/-- Given a man who owns 2/3 of a business and sells 3/4 of his shares for 75,000 Rs,
    the total value of the business is 150,000 Rs. -/
theorem business_value (owned_fraction : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  owned_fraction = 2/3 →
  sold_fraction = 3/4 →
  sale_price = 75000 →
  (owned_fraction * sold_fraction * (sale_price : ℚ) / (owned_fraction * sold_fraction)) = 150000 :=
by sorry

end NUMINAMATH_CALUDE_business_value_l3570_357018


namespace NUMINAMATH_CALUDE_domain_subset_iff_a_range_l3570_357099

theorem domain_subset_iff_a_range (a : ℝ) (h : a < 1) :
  (∀ x, (x - a - 1) * (2 * a - x) > 0 → x ∈ (Set.Iic (-1) ∪ Set.Ici 1)) ↔
  a ∈ (Set.Iic (-2) ∪ Set.Icc (1/2) 1) :=
sorry

end NUMINAMATH_CALUDE_domain_subset_iff_a_range_l3570_357099


namespace NUMINAMATH_CALUDE_absolute_value_sum_range_l3570_357036

theorem absolute_value_sum_range : 
  ∃ (min_value : ℝ), 
    (∀ x : ℝ, |x - 1| + |x - 2| ≥ min_value) ∧ 
    (∃ x : ℝ, |x - 1| + |x - 2| = min_value) ∧
    (∀ a : ℝ, (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ min_value) ∧
    min_value = 1 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_range_l3570_357036


namespace NUMINAMATH_CALUDE_percentage_without_full_time_jobs_l3570_357020

theorem percentage_without_full_time_jobs :
  let total_parents : ℝ := 100
  let mothers : ℝ := 0.6 * total_parents
  let fathers : ℝ := 0.4 * total_parents
  let mothers_with_jobs : ℝ := (7/8) * mothers
  let fathers_with_jobs : ℝ := (3/4) * fathers
  let parents_with_jobs : ℝ := mothers_with_jobs + fathers_with_jobs
  let parents_without_jobs : ℝ := total_parents - parents_with_jobs
  (parents_without_jobs / total_parents) * 100 = 18 :=
by sorry

end NUMINAMATH_CALUDE_percentage_without_full_time_jobs_l3570_357020


namespace NUMINAMATH_CALUDE_diamond_composition_l3570_357002

/-- Define the diamond operation -/
def diamond (k : ℝ) (x y : ℝ) : ℝ := x^2 - k*y

/-- Theorem stating the result of h ◇ (h ◇ h) -/
theorem diamond_composition (h : ℝ) : diamond 3 h (diamond 3 h h) = -2*h^2 + 9*h := by
  sorry

end NUMINAMATH_CALUDE_diamond_composition_l3570_357002


namespace NUMINAMATH_CALUDE_gcd_459_357_l3570_357072

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3570_357072


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_30_l3570_357046

theorem least_product_of_two_primes_above_30 (p q : ℕ) : 
  Prime p → Prime q → p ≠ q → p > 30 → q > 30 → 
  ∀ r s : ℕ, Prime r → Prime s → r ≠ s → r > 30 → s > 30 → 
  p * q ≤ r * s := by
  sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_30_l3570_357046


namespace NUMINAMATH_CALUDE_competition_results_l3570_357051

structure GradeData where
  boys_rate : ℝ
  girls_rate : ℝ

def seventh_grade : GradeData :=
  { boys_rate := 0.4, girls_rate := 0.6 }

def eighth_grade : GradeData :=
  { boys_rate := 0.5, girls_rate := 0.7 }

theorem competition_results :
  (seventh_grade.boys_rate < eighth_grade.boys_rate) ∧
  ((seventh_grade.boys_rate + eighth_grade.boys_rate) / 2 < (seventh_grade.girls_rate + eighth_grade.girls_rate) / 2) :=
by sorry

end NUMINAMATH_CALUDE_competition_results_l3570_357051


namespace NUMINAMATH_CALUDE_volunteers_distribution_count_l3570_357031

def number_of_volunteers : ℕ := 5
def number_of_schools : ℕ := 3

/-- The number of ways to distribute volunteers to schools -/
def distribute_volunteers : ℕ := sorry

theorem volunteers_distribution_count :
  distribute_volunteers = 150 := by sorry

end NUMINAMATH_CALUDE_volunteers_distribution_count_l3570_357031


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3570_357016

theorem imaginary_power_sum : Complex.I ^ 21 + Complex.I ^ 103 + Complex.I ^ 50 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3570_357016


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3570_357061

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- State the theorem
theorem vertex_of_quadratic :
  ∃ (a h k : ℝ), (∀ x, f x = a * (x - h)^2 + k) ∧ (f h = k) ∧ (∀ x, f x ≤ k) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3570_357061


namespace NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l3570_357076

theorem odd_multiple_of_nine_is_multiple_of_three (S : ℤ) :
  Odd S → (∃ k : ℤ, S = 9 * k) → (∃ m : ℤ, S = 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l3570_357076


namespace NUMINAMATH_CALUDE_system_solution_l3570_357060

theorem system_solution :
  ∃ (x y : ℚ), 
    (5 * x - 3 * y = -7) ∧ 
    (4 * x + 6 * y = 34) ∧ 
    (x = 10 / 7) ∧ 
    (y = 33 / 7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3570_357060


namespace NUMINAMATH_CALUDE_complex_magnitude_l3570_357012

theorem complex_magnitude (z : ℂ) (h : 1 + z * Complex.I = 2 * Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3570_357012


namespace NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l3570_357064

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) +
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l3570_357064


namespace NUMINAMATH_CALUDE_magic_deck_price_is_two_l3570_357028

/-- The price of a magic card deck given initial and final quantities and total earnings -/
def magic_deck_price (initial : ℕ) (final : ℕ) (earnings : ℕ) : ℚ :=
  earnings / (initial - final)

/-- Theorem: The price of each magic card deck is 2 dollars -/
theorem magic_deck_price_is_two :
  magic_deck_price 5 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_magic_deck_price_is_two_l3570_357028


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3570_357074

theorem largest_solution_of_equation :
  let f : ℝ → ℝ := λ x => (3*x)/7 + 2/(7*x)
  ∃ x : ℝ, x > 0 ∧ f x = 3/4 ∧ ∀ y : ℝ, y > 0 → f y = 3/4 → y ≤ x ∧ x = (21 + Real.sqrt 345) / 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3570_357074


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3570_357027

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 + Complex.I) = -1 / (2 * Complex.I)) :
  z.im = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3570_357027


namespace NUMINAMATH_CALUDE_amount_decreased_l3570_357008

theorem amount_decreased (x y : ℝ) (h1 : x = 50.0) (h2 : 0.20 * x - y = 6) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_amount_decreased_l3570_357008


namespace NUMINAMATH_CALUDE_regression_estimate_l3570_357026

theorem regression_estimate :
  let regression_equation (x : ℝ) := 4.75 * x + 2.57
  regression_equation 28 = 135.57 := by sorry

end NUMINAMATH_CALUDE_regression_estimate_l3570_357026


namespace NUMINAMATH_CALUDE_coins_divisible_by_six_l3570_357059

theorem coins_divisible_by_six (n : ℕ) : 
  (∃ (a b c : ℕ), n = 2*a + 2*b + 2*c) ∧ 
  (∃ (x y : ℕ), n = 3*x + 3*y) → 
  ∃ (z : ℕ), n = 6*z :=
sorry

end NUMINAMATH_CALUDE_coins_divisible_by_six_l3570_357059


namespace NUMINAMATH_CALUDE_flooring_rate_calculation_l3570_357066

/-- Given a rectangular room with specified dimensions and total flooring cost,
    calculate the rate per square meter. -/
theorem flooring_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 10)
    (h_width : width = 4.75)
    (h_total_cost : total_cost = 42750) : 
    total_cost / (length * width) = 900 := by
  sorry

#check flooring_rate_calculation

end NUMINAMATH_CALUDE_flooring_rate_calculation_l3570_357066


namespace NUMINAMATH_CALUDE_num_colorings_is_162_l3570_357073

/-- Represents the three colors available for coloring --/
inductive Color
| Red
| White
| Blue

/-- Represents a coloring of a single triangle --/
structure TriangleColoring :=
  (a b c : Color)
  (different_colors : a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- Represents a coloring of the entire figure (four triangles) --/
structure FigureColoring :=
  (t1 t2 t3 t4 : TriangleColoring)
  (connected_different : t1.c = t2.a ∧ t2.c = t3.a ∧ t3.c = t4.a)

/-- The number of valid colorings for the figure --/
def num_colorings : ℕ := sorry

/-- Theorem stating that the number of valid colorings is 162 --/
theorem num_colorings_is_162 : num_colorings = 162 := by sorry

end NUMINAMATH_CALUDE_num_colorings_is_162_l3570_357073


namespace NUMINAMATH_CALUDE_largest_tile_size_l3570_357079

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℕ := 378
def courtyard_width : ℕ := 525

-- Define the tile size in centimeters
def tile_size : ℕ := 21

-- Theorem statement
theorem largest_tile_size :
  (courtyard_length % tile_size = 0) ∧
  (courtyard_width % tile_size = 0) ∧
  (∀ s : ℕ, s > tile_size →
    (courtyard_length % s ≠ 0) ∨ (courtyard_width % s ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_largest_tile_size_l3570_357079


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l3570_357049

theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
  (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
  (a = 2 ∧ b = -1 ∧ p = -1 ∧ q = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l3570_357049


namespace NUMINAMATH_CALUDE_unique_fixed_point_for_rotationally_invariant_function_l3570_357071

-- Define a function that remains unchanged when its graph is rotated by π/2
def RotationallyInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (-y) = x

-- Theorem statement
theorem unique_fixed_point_for_rotationally_invariant_function
  (f : ℝ → ℝ) (h : RotationallyInvariant f) :
  ∃! x : ℝ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_unique_fixed_point_for_rotationally_invariant_function_l3570_357071


namespace NUMINAMATH_CALUDE_circle_C_equation_tangent_lines_through_M_l3570_357082

-- Define the points A, B, and M
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (3, 1)

-- Define circle C with diameter AB
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 4}

-- Theorem for the equation of circle C
theorem circle_C_equation : C = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4} := by sorry

-- Theorem for the tangent lines passing through M
theorem tangent_lines_through_M : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∃ (k : ℝ), l = {p : ℝ × ℝ | p.1 = 3} ∨ l = {p : ℝ × ℝ | 3*p.1 - 4*p.2 - 5 = 0}) ↔
    (M ∈ l ∧ ∃! (p : ℝ × ℝ), p ∈ l ∩ C) := by sorry

end NUMINAMATH_CALUDE_circle_C_equation_tangent_lines_through_M_l3570_357082


namespace NUMINAMATH_CALUDE_magnitude_of_linear_combination_l3570_357077

/-- Given two plane vectors α and β, prove that |2α + β| = √10 -/
theorem magnitude_of_linear_combination (α β : ℝ × ℝ) 
  (h1 : ‖α‖ = 1) 
  (h2 : ‖β‖ = 2) 
  (h3 : α • (α - 2 • β) = 0) : 
  ‖2 • α + β‖ = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_linear_combination_l3570_357077


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3570_357075

/-- The first term of the geometric series -/
def a₁ : ℚ := 4 / 7

/-- The second term of the geometric series -/
def a₂ : ℚ := -16 / 21

/-- The third term of the geometric series -/
def a₃ : ℚ := -64 / 63

/-- The common ratio of the geometric series -/
def r : ℚ := -4 / 3

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3570_357075


namespace NUMINAMATH_CALUDE_variance_best_for_stability_l3570_357096

/-- Represents a statistical measure -/
inductive StatMeasure
  | Mode
  | Variance
  | Mean
  | Frequency

/-- Represents an athlete's performance data -/
structure AthleteData where
  results : List Float
  len : Nat
  h_len : len = 10

/-- Assesses the stability of performance based on a statistical measure -/
def assessStability (measure : StatMeasure) (data : AthleteData) : Bool :=
  sorry

/-- Theorem stating that variance is the most suitable measure for assessing stability -/
theorem variance_best_for_stability (data : AthleteData) :
  ∀ (m : StatMeasure), m ≠ StatMeasure.Variance →
    assessStability StatMeasure.Variance data = true ∧
    assessStability m data = false :=
  sorry

end NUMINAMATH_CALUDE_variance_best_for_stability_l3570_357096


namespace NUMINAMATH_CALUDE_max_xy_value_l3570_357015

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  ∀ z w : ℝ, z > 0 → w > 0 → z + w = 18 → x * y ≥ z * w ∧ x * y ≤ 81 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l3570_357015


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3570_357032

/-- Given the quadratic equation x^2 + 4x + 4 = 0, which can be transformed
    into the form (x + h)^2 = k, prove that h + k = 2. -/
theorem quadratic_transformation (h k : ℝ) : 
  (∀ x, x^2 + 4*x + 4 = 0 ↔ (x + h)^2 = k) → h + k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3570_357032


namespace NUMINAMATH_CALUDE_parts_cost_calculation_l3570_357013

theorem parts_cost_calculation (total_amount : ℝ) (total_parts : ℕ) 
  (expensive_parts : ℕ) (expensive_cost : ℝ) :
  total_amount = 2380 →
  total_parts = 59 →
  expensive_parts = 40 →
  expensive_cost = 50 →
  ∃ (other_cost : ℝ),
    other_cost = 20 ∧
    total_amount = expensive_parts * expensive_cost + (total_parts - expensive_parts) * other_cost :=
by sorry

end NUMINAMATH_CALUDE_parts_cost_calculation_l3570_357013


namespace NUMINAMATH_CALUDE_expression_bounds_l3570_357070

theorem expression_bounds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m M : ℝ),
    (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → m ≤ (3 * |x + y|) / (|x| + |y|) ∧ (3 * |x + y|) / (|x| + |y|) ≤ M) ∧
    m = 0 ∧ M = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3570_357070


namespace NUMINAMATH_CALUDE_prob_girl_from_E_expected_value_X_l3570_357009

/-- Represents a family with a number of boys and girls -/
structure Family :=
  (boys : Nat)
  (girls : Nat)

/-- The set of all families -/
def Families : Finset (Fin 5) := Finset.univ

/-- The number of boys and girls in each family -/
def familyData : Fin 5 → Family
  | ⟨0, _⟩ => ⟨0, 0⟩  -- Family A
  | ⟨1, _⟩ => ⟨1, 0⟩  -- Family B
  | ⟨2, _⟩ => ⟨0, 1⟩  -- Family C
  | ⟨3, _⟩ => ⟨1, 1⟩  -- Family D
  | ⟨4, _⟩ => ⟨1, 2⟩  -- Family E

/-- The total number of children -/
def totalChildren : Nat := Finset.sum Families (λ i => (familyData i).boys + (familyData i).girls)

/-- The total number of girls -/
def totalGirls : Nat := Finset.sum Families (λ i => (familyData i).girls)

/-- Probability of selecting a girl from family E given that a girl is selected -/
theorem prob_girl_from_E : 
  (familyData 4).girls / totalGirls = 1 / 2 := by sorry

/-- Probability distribution of X when selecting 3 families -/
def probDistX (x : Fin 3) : Rat :=
  match x with
  | ⟨0, _⟩ => 1 / 10
  | ⟨1, _⟩ => 3 / 5
  | ⟨2, _⟩ => 3 / 10

/-- Expected value of X -/
def expectedX : Rat := Finset.sum (Finset.range 3) (λ i => i * probDistX i)

theorem expected_value_X : expectedX = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_girl_from_E_expected_value_X_l3570_357009


namespace NUMINAMATH_CALUDE_soccer_team_boys_l3570_357004

/-- Proves the number of boys on a soccer team given certain conditions -/
theorem soccer_team_boys (total : ℕ) (attendees : ℕ) : 
  total = 30 → 
  attendees = 20 → 
  ∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys + (girls / 3) = attendees ∧
    boys = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_boys_l3570_357004


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l3570_357054

def chocolate_problem (cost_per_bar total_bars unsold_bars : ℕ) : Prop :=
  let sold_bars := total_bars - unsold_bars
  let money_made := sold_bars * cost_per_bar
  money_made = 9

theorem olivia_chocolate_sales : chocolate_problem 3 7 4 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l3570_357054


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3570_357014

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3570_357014


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3570_357003

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  855 * (π / 180) = 59 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3570_357003


namespace NUMINAMATH_CALUDE_min_value_zero_l3570_357048

theorem min_value_zero (x y : ℕ) (hx : x ≤ 2) (hy : y ≤ 3) :
  (x^2 * y^2 : ℝ) / (x^2 + y^2)^2 ≥ 0 ∧
  ∃ (a b : ℕ), a ≤ 2 ∧ b ≤ 3 ∧ (a^2 * b^2 : ℝ) / (a^2 + b^2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_zero_l3570_357048


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l3570_357017

theorem lottery_winning_probability :
  let megaball_count : ℕ := 30
  let winnerball_count : ℕ := 50
  let chosen_winnerball_count : ℕ := 6

  let megaball_prob : ℚ := 1 / megaball_count
  let winnerball_prob : ℚ := 1 / (winnerball_count.choose chosen_winnerball_count)

  megaball_prob * winnerball_prob = 1 / 477621000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l3570_357017


namespace NUMINAMATH_CALUDE_table_tennis_players_l3570_357040

theorem table_tennis_players (singles_tables doubles_tables : ℕ) : 
  singles_tables + doubles_tables = 13 → 
  4 * doubles_tables = 2 * singles_tables + 4 → 
  4 * doubles_tables = 20 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_players_l3570_357040


namespace NUMINAMATH_CALUDE_copper_content_bounds_l3570_357006

/-- Represents the composition of an alloy --/
structure Alloy where
  nickel : ℝ
  copper : ℝ
  manganese : ℝ
  sum_to_one : nickel + copper + manganese = 1

/-- The three initial alloys --/
def alloy1 : Alloy := ⟨0.3, 0.7, 0, by norm_num⟩
def alloy2 : Alloy := ⟨0, 0.1, 0.9, by norm_num⟩
def alloy3 : Alloy := ⟨0.15, 0.25, 0.6, by norm_num⟩

/-- The fraction of each initial alloy in the new alloy --/
structure Fractions where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ
  sum_to_one : x1 + x2 + x3 = 1
  manganese_constraint : 0.9 * x2 + 0.6 * x3 = 0.4

/-- The copper content in the new alloy --/
def copper_content (f : Fractions) : ℝ :=
  0.7 * f.x1 + 0.1 * f.x2 + 0.25 * f.x3

/-- The main theorem stating the bounds of copper content --/
theorem copper_content_bounds (f : Fractions) : 
  0.4 ≤ copper_content f ∧ copper_content f ≤ 13/30 := by sorry

end NUMINAMATH_CALUDE_copper_content_bounds_l3570_357006


namespace NUMINAMATH_CALUDE_peach_basket_problem_l3570_357033

theorem peach_basket_problem (x : ℕ) : 
  (x > 0) →
  (x - (x / 2 + 1) > 0) →
  (x - (x / 2 + 1) - ((x - (x / 2 + 1)) / 2 - 1) = 4) →
  (x = 14) :=
by
  sorry

#check peach_basket_problem

end NUMINAMATH_CALUDE_peach_basket_problem_l3570_357033


namespace NUMINAMATH_CALUDE_pencil_count_l3570_357030

theorem pencil_count (rows : ℕ) (pencils_per_row : ℕ) (h1 : rows = 2) (h2 : pencils_per_row = 3) :
  rows * pencils_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3570_357030


namespace NUMINAMATH_CALUDE_least_months_to_triple_l3570_357024

def interest_rate : ℝ := 1.06

def amount_owed (t : ℕ) : ℝ := interest_rate ^ t

def exceeds_triple (t : ℕ) : Prop := amount_owed t > 3

theorem least_months_to_triple : 
  (∀ n < 20, ¬(exceeds_triple n)) ∧ (exceeds_triple 20) :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l3570_357024


namespace NUMINAMATH_CALUDE_positive_real_power_difference_integer_l3570_357085

theorem positive_real_power_difference_integer (x : ℝ) (h1 : x > 0) 
  (h2 : ∃ (a b : ℤ), x^2012 - x^2001 = a ∧ x^2001 - x^1990 = b) : 
  ∃ (n : ℤ), x = n :=
sorry

end NUMINAMATH_CALUDE_positive_real_power_difference_integer_l3570_357085


namespace NUMINAMATH_CALUDE_expression_always_zero_l3570_357098

theorem expression_always_zero (x y : ℝ) : 
  5 * (x^3 - 3*x^2*y - 2*x*y^2) - 3 * (x^3 - 5*x^2*y + 2*y^3) + 2 * (-x^3 + 5*x*y^2 + 3*y^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_zero_l3570_357098


namespace NUMINAMATH_CALUDE_max_bw_edges_grid_l3570_357052

/-- Represents a square grid with corners removed and colored squares. -/
structure ColoredGrid :=
  (size : ℕ)
  (corner_size : ℕ)
  (coloring : ℕ → ℕ → Bool)

/-- Checks if a 2x2 square forms a checkerboard pattern. -/
def is_checkerboard (g : ColoredGrid) (x y : ℕ) : Prop :=
  g.coloring x y ≠ g.coloring (x+1) y ∧
  g.coloring x y ≠ g.coloring x (y+1) ∧
  g.coloring x y = g.coloring (x+1) (y+1)

/-- Counts the number of black-white edges in the grid. -/
def count_bw_edges (g : ColoredGrid) : ℕ := sorry

/-- The main theorem statement. -/
theorem max_bw_edges_grid (g : ColoredGrid) :
  g.size = 300 →
  g.corner_size = 100 →
  (∀ x y, x < g.size - g.corner_size ∧ y < g.size - g.corner_size →
    ¬is_checkerboard g x y) →
  count_bw_edges g ≤ 49998 :=
sorry

end NUMINAMATH_CALUDE_max_bw_edges_grid_l3570_357052


namespace NUMINAMATH_CALUDE_club_membership_increase_l3570_357023

theorem club_membership_increase (current_members additional_members : ℕ) 
  (h1 : current_members = 10)
  (h2 : additional_members = 15) :
  let new_total := current_members + additional_members
  new_total - current_members = 15 ∧ new_total > 2 * current_members :=
by sorry

end NUMINAMATH_CALUDE_club_membership_increase_l3570_357023


namespace NUMINAMATH_CALUDE_problem_solution_l3570_357087

def f (a x : ℝ) := |2*x + a| - |2*x + 3|
def g (x : ℝ) := |x - 1| - 3

theorem problem_solution :
  (∀ x : ℝ, |g x| < 2 ↔ (2 < x ∧ x < 6) ∨ (-4 < x ∧ x < 0)) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ (0 ≤ a ∧ a ≤ 6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3570_357087


namespace NUMINAMATH_CALUDE_shirt_cost_l3570_357067

def flat_rate_shipping : ℝ := 5
def shipping_rate : ℝ := 0.2
def shipping_threshold : ℝ := 50
def socks_price : ℝ := 5
def shorts_price : ℝ := 15
def swim_trunks_price : ℝ := 14
def total_bill : ℝ := 102
def shorts_quantity : ℕ := 2

theorem shirt_cost (shirt_price : ℝ) : 
  (shirt_price + socks_price + shorts_quantity * shorts_price + swim_trunks_price > shipping_threshold) →
  (shirt_price + socks_price + shorts_quantity * shorts_price + swim_trunks_price) * 
    (1 + shipping_rate) = total_bill →
  shirt_price = 36 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l3570_357067


namespace NUMINAMATH_CALUDE_quadratic_sum_l3570_357035

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_sum (a b c : ℝ) :
  f a b c 1 = 0 →
  f a b c 5 = 0 →
  (∀ x, f a b c x ≥ 36) →
  (∃ x₀, f a b c x₀ = 36) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3570_357035


namespace NUMINAMATH_CALUDE_investment_loss_l3570_357068

/-- Given two investors with capitals in ratio 1:9 and proportional loss distribution,
    if one investor's loss is 603, then the total loss is 670. -/
theorem investment_loss (capital_ratio : ℚ) (investor1_loss : ℚ) (total_loss : ℚ) :
  capital_ratio = 1 / 9 →
  investor1_loss = 603 →
  total_loss = investor1_loss / (capital_ratio / (capital_ratio + 1)) →
  total_loss = 670 := by
  sorry

end NUMINAMATH_CALUDE_investment_loss_l3570_357068


namespace NUMINAMATH_CALUDE_cube_skew_lines_theorem_l3570_357094

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  edge_length : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point1 : Point3D
  point2 : Point3D

/-- Calculate the distance between two skew lines in 3D space -/
def distance_between_skew_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Check if a line segment is perpendicular to two other lines -/
def is_perpendicular_to_lines (segment : Line3D) (l1 l2 : Line3D) : Prop := sorry

/-- Main theorem about the distance between skew lines in a cube and their common perpendicular -/
theorem cube_skew_lines_theorem (cube : Cube) : 
  let A₁D := Line3D.mk cube.A₁ cube.D
  let D₁C := Line3D.mk cube.D₁ cube.C
  let X := Point3D.mk ((2 * cube.D.x + cube.A₁.x) / 3) ((2 * cube.D.y + cube.A₁.y) / 3) ((2 * cube.D.z + cube.A₁.z) / 3)
  let Y := Point3D.mk ((2 * cube.D₁.x + cube.C.x) / 3) ((2 * cube.D₁.y + cube.C.y) / 3) ((2 * cube.D₁.z + cube.C.z) / 3)
  let XY := Line3D.mk X Y
  distance_between_skew_lines A₁D D₁C = cube.edge_length * Real.sqrt 3 / 3 ∧
  is_perpendicular_to_lines XY A₁D D₁C := by
  sorry

end NUMINAMATH_CALUDE_cube_skew_lines_theorem_l3570_357094


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3570_357029

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 + 3*x^2*Real.sqrt 2 + 6*x + 2*Real.sqrt 2) + (x + Real.sqrt 2)
  ∀ x : ℂ, f x = 0 ↔ x = -Real.sqrt 2 ∨ x = -Real.sqrt 2 + Complex.I ∨ x = -Real.sqrt 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3570_357029


namespace NUMINAMATH_CALUDE_difference_of_same_prime_divisors_l3570_357050

/-- For any natural number, there exist two natural numbers with the same number of distinct prime divisors whose difference is the original number. -/
theorem difference_of_same_prime_divisors (n : ℕ) : 
  ∃ a b : ℕ, n = a - b ∧ (Finset.card (Nat.factorization a).support = Finset.card (Nat.factorization b).support) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_same_prime_divisors_l3570_357050


namespace NUMINAMATH_CALUDE_calculate_expression_l3570_357044

theorem calculate_expression : (42 / (3^2 + 2 * 3 - 1)) * 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3570_357044


namespace NUMINAMATH_CALUDE_suzanna_distance_l3570_357055

/-- Represents the distance in miles Suzanna cycles in a given time -/
def distance_cycled (time_minutes : ℕ) : ℚ :=
  (time_minutes / 10 : ℚ) * 2

/-- Proves that Suzanna cycles 8 miles in 40 minutes given her steady speed -/
theorem suzanna_distance : distance_cycled 40 = 8 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_distance_l3570_357055


namespace NUMINAMATH_CALUDE_sum_of_roots_l3570_357042

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 18*c^2 + 25*c - 75 = 0) 
  (hd : 9*d^3 - 72*d^2 - 345*d + 3060 = 0) : 
  c + d = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3570_357042


namespace NUMINAMATH_CALUDE_bake_sale_girls_l3570_357037

theorem bake_sale_girls (initial_total : ℕ) : 
  -- Initial conditions
  (3 * initial_total / 5 : ℚ) = initial_total * (60 : ℚ) / 100 →
  -- Changes in group composition
  let new_total := initial_total - 1 + 3
  let new_girls := (3 * initial_total / 5 : ℚ) - 3
  -- Final condition
  new_girls / new_total = (1 : ℚ) / 2 →
  -- Conclusion
  (3 * initial_total / 5 : ℚ) = 24 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_girls_l3570_357037


namespace NUMINAMATH_CALUDE_simplify_expression_l3570_357045

theorem simplify_expression :
  (Real.sqrt 10 + Real.sqrt 15) / (Real.sqrt 3 + Real.sqrt 5 - Real.sqrt 2) =
  (2 * Real.sqrt 30 + 5 * Real.sqrt 2 + 11 * Real.sqrt 5 + 5 * Real.sqrt 3) / 6 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3570_357045


namespace NUMINAMATH_CALUDE_sum_of_inscribed_squares_l3570_357078

/-- The sum of areas of an infinite series of inscribed squares -/
theorem sum_of_inscribed_squares (a : ℝ) (h : a > 0) :
  ∃ S : ℝ, S = (4 * a^2) / 3 ∧ 
  S = a^2 + ∑' n, (a^2 / 4^n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_squares_l3570_357078


namespace NUMINAMATH_CALUDE_apple_juice_production_l3570_357043

theorem apple_juice_production (total_production : ℝ) 
  (mixed_percentage : ℝ) (juice_percentage : ℝ) :
  total_production = 5.5 →
  mixed_percentage = 0.2 →
  juice_percentage = 0.5 →
  (1 - mixed_percentage) * juice_percentage * total_production = 2.2 := by
sorry

end NUMINAMATH_CALUDE_apple_juice_production_l3570_357043


namespace NUMINAMATH_CALUDE_range_of_a_l3570_357011

def set_A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

def set_B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : ℝ) : set_A a ∩ set_B = ∅ → 2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3570_357011


namespace NUMINAMATH_CALUDE_quartic_root_sum_l3570_357021

theorem quartic_root_sum (p q : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 4 + p * (Complex.I + 2 : ℂ) ^ 2 + q * (Complex.I + 2 : ℂ) + 1 = 0 →
  p + q = 10 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_sum_l3570_357021


namespace NUMINAMATH_CALUDE_max_value_sin_squared_minus_two_sin_minus_two_l3570_357097

theorem max_value_sin_squared_minus_two_sin_minus_two :
  ∀ x : ℝ, 
    -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 →
    ∀ y : ℝ, 
      y = Real.sin x ^ 2 - 2 * Real.sin x - 2 →
      y ≤ 1 ∧ ∃ x₀ : ℝ, Real.sin x₀ ^ 2 - 2 * Real.sin x₀ - 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_sin_squared_minus_two_sin_minus_two_l3570_357097


namespace NUMINAMATH_CALUDE_parabola_properties_l3570_357092

/-- Parabola properties -/
theorem parabola_properties (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - (a + 1) * x
  (f 2 = 0) →
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = -4 ∧ f x₁ = f x₂ → a = -1/5) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₂ ≥ -2 ∧ f x₁ < f x₂ → -1/5 ≤ a ∧ a < 0) ∧
  (∃ (x : ℝ), x = 1 ∧ ∀ (y : ℝ), f (x + y) = f (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3570_357092


namespace NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l3570_357063

theorem difference_of_odd_squares_divisible_by_eight (n p : ℤ) :
  ∃ k : ℤ, (2 * n + 1)^2 - (2 * p + 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l3570_357063


namespace NUMINAMATH_CALUDE_sum_divisible_by_ten_l3570_357084

theorem sum_divisible_by_ten (n : ℕ) : 
  10 ∣ (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) ↔ n % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_divisible_by_ten_l3570_357084


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l3570_357007

theorem power_equality_implies_exponent (p : ℕ) : 16^10 = 4^p → p = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l3570_357007


namespace NUMINAMATH_CALUDE_opposite_of_three_l3570_357053

theorem opposite_of_three : (-(3 : ℝ)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3570_357053


namespace NUMINAMATH_CALUDE_parabola_vertex_position_l3570_357090

/-- A parabola with points P, Q, and M satisfying specific conditions -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  m : ℝ
  h1 : y₁ = a * (-2)^2 + b * (-2) + c
  h2 : y₂ = a * 4^2 + b * 4 + c
  h3 : y₃ = a * m^2 + b * m + c
  h4 : 2 * a * m + b = 0
  h5 : y₃ ≥ y₂
  h6 : y₂ > y₁

theorem parabola_vertex_position (p : Parabola) : p.m > 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_position_l3570_357090


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l3570_357000

-- Part 1
theorem simplify_expression (x y : ℝ) :
  (3 * x^2 - 2 * x * y + 5 * y^2) - 2 * (x^2 - x * y - 2 * y^2) = x^2 + 9 * y^2 := by
  sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (A B : ℝ) 
  (h1 : A = -x - 2*y - 1)
  (h2 : B = x + 2*y + 2)
  (h3 : x + 2*y = 6) :
  A + 3*B = 17 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l3570_357000


namespace NUMINAMATH_CALUDE_uniform_pickup_ways_l3570_357065

def number_of_students : ℕ := 5
def correct_picks : ℕ := 2

theorem uniform_pickup_ways :
  (number_of_students.choose correct_picks) * 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_uniform_pickup_ways_l3570_357065


namespace NUMINAMATH_CALUDE_circle_area_ratio_false_l3570_357058

theorem circle_area_ratio_false : 
  ¬ (∀ (r : ℝ), r > 0 → (π * r^2) / (π * (2*r)^2) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_false_l3570_357058


namespace NUMINAMATH_CALUDE_larger_number_problem_l3570_357005

theorem larger_number_problem (x y : ℕ) 
  (h1 : y - x = 1500)
  (h2 : y = 6 * x + 15) : 
  y = 1797 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3570_357005


namespace NUMINAMATH_CALUDE_cube_plane_intersection_theorem_l3570_357041

/-- A regular polygon that can be formed by the intersection of a cube and a plane -/
inductive CubeIntersectionPolygon
  | Triangle
  | Quadrilateral
  | Hexagon

/-- The set of all possible regular polygons that can be formed by the intersection of a cube and a plane -/
def possibleIntersectionPolygons : Set CubeIntersectionPolygon :=
  {CubeIntersectionPolygon.Triangle, CubeIntersectionPolygon.Quadrilateral, CubeIntersectionPolygon.Hexagon}

/-- A function that determines if a given regular polygon can be formed by the intersection of a cube and a plane -/
def isValidIntersectionPolygon (p : CubeIntersectionPolygon) : Prop :=
  p ∈ possibleIntersectionPolygons

theorem cube_plane_intersection_theorem :
  ∀ (p : CubeIntersectionPolygon), isValidIntersectionPolygon p ↔
    (p = CubeIntersectionPolygon.Triangle ∨
     p = CubeIntersectionPolygon.Quadrilateral ∨
     p = CubeIntersectionPolygon.Hexagon) :=
by sorry


end NUMINAMATH_CALUDE_cube_plane_intersection_theorem_l3570_357041


namespace NUMINAMATH_CALUDE_one_third_of_nine_times_seven_l3570_357088

theorem one_third_of_nine_times_seven : (1 / 3 : ℚ) * (9 * 7) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_nine_times_seven_l3570_357088


namespace NUMINAMATH_CALUDE_problem_solution_l3570_357039

theorem problem_solution (x y : ℝ) 
  (h1 : x^2 + y^2 - x*y = 2) 
  (h2 : x^4 + y^4 + x^2*y^2 = 8) : 
  x^8 + y^8 + x^2014*y^2014 = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3570_357039


namespace NUMINAMATH_CALUDE_second_wing_rooms_per_hall_l3570_357069

/-- Represents a hotel wing -/
structure Wing where
  floors : Nat
  hallsPerFloor : Nat
  roomsPerHall : Nat

/-- Represents a hotel with two wings -/
structure Hotel where
  wing1 : Wing
  wing2 : Wing
  totalRooms : Nat

def Hotel.secondWingRoomsPerHall (h : Hotel) : Nat :=
  (h.totalRooms - h.wing1.floors * h.wing1.hallsPerFloor * h.wing1.roomsPerHall) / 
  (h.wing2.floors * h.wing2.hallsPerFloor)

theorem second_wing_rooms_per_hall :
  let h : Hotel := {
    wing1 := { floors := 9, hallsPerFloor := 6, roomsPerHall := 32 },
    wing2 := { floors := 7, hallsPerFloor := 9, roomsPerHall := 0 }, -- roomsPerHall is unknown
    totalRooms := 4248
  }
  h.secondWingRoomsPerHall = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_wing_rooms_per_hall_l3570_357069


namespace NUMINAMATH_CALUDE_phase_shift_sine_specific_phase_shift_l3570_357047

/-- The phase shift of a sine function of the form y = A sin(Bx + C) is -C/B -/
theorem phase_shift_sine (A B C : ℝ) (h : B ≠ 0) :
  let f := λ x : ℝ => A * Real.sin (B * x + C)
  let phase_shift := -C / B
  ∀ x : ℝ, f (x + phase_shift) = A * Real.sin (B * x) := by
  sorry

/-- The phase shift of y = 3 sin(3x + π/4) is -π/12 -/
theorem specific_phase_shift :
  let f := λ x : ℝ => 3 * Real.sin (3 * x + π/4)
  let phase_shift := -π/12
  ∀ x : ℝ, f (x + phase_shift) = 3 * Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_phase_shift_sine_specific_phase_shift_l3570_357047


namespace NUMINAMATH_CALUDE_pen_cost_theorem_l3570_357062

/-- The average cost per pen in cents, rounded to the nearest whole number,
    given the number of pens, cost of pens, and shipping cost. -/
def average_cost_per_pen (num_pens : ℕ) (pen_cost shipping_cost : ℚ) : ℕ :=
  let total_cost_cents := (pen_cost + shipping_cost) * 100
  let average_cost := total_cost_cents / num_pens
  (average_cost + 1/2).floor.toNat

/-- Theorem stating that for 300 pens costing $29.85 with $8.10 shipping,
    the average cost per pen is 13 cents when rounded to the nearest whole number. -/
theorem pen_cost_theorem :
  average_cost_per_pen 300 (29.85) (8.10) = 13 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_theorem_l3570_357062


namespace NUMINAMATH_CALUDE_sum_of_bases_l3570_357001

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The value of digit C in base 13 -/
def C : Nat := 12

theorem sum_of_bases :
  to_base_10 [7, 5, 3] 9 + to_base_10 [2, C, 4] 13 = 1129 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_bases_l3570_357001


namespace NUMINAMATH_CALUDE_adamek_marbles_l3570_357022

theorem adamek_marbles :
  ∀ (n : ℕ), 
    (∃ (a b : ℕ), n = 3 * a ∧ n = 4 * b) →
    (∃ (k : ℕ), 3 * (k + 8) = 4 * k) →
    n = 96 := by
  sorry

end NUMINAMATH_CALUDE_adamek_marbles_l3570_357022


namespace NUMINAMATH_CALUDE_triangle_on_hyperbola_l3570_357093

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y = 1 / x

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define parallel lines
def parallel (p1 p2 p3 p4 : PointOnHyperbola) : Prop :=
  (p2.y - p1.y) / (p2.x - p1.x) = (p4.y - p3.y) / (p4.x - p3.x)

-- Define the theorem
theorem triangle_on_hyperbola
  (A B C A₁ B₁ C₁ : PointOnHyperbola)
  (h1 : parallel A B A₁ B₁)
  (h2 : parallel B C B₁ C₁) :
  parallel A C₁ A₁ C := by
  sorry

end NUMINAMATH_CALUDE_triangle_on_hyperbola_l3570_357093


namespace NUMINAMATH_CALUDE_red_packet_probability_l3570_357091

def red_packet_amounts : List ℝ := [1.49, 1.31, 2.19, 3.40, 0.61]

def total_amount : ℝ := 9

def num_people : ℕ := 5

def threshold : ℝ := 4

def probability_ab_sum_ge_threshold (amounts : List ℝ) (total : ℝ) (n : ℕ) (t : ℝ) : ℚ :=
  sorry

theorem red_packet_probability :
  probability_ab_sum_ge_threshold red_packet_amounts total_amount num_people threshold = 2/5 :=
sorry

end NUMINAMATH_CALUDE_red_packet_probability_l3570_357091


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3570_357038

theorem tan_alpha_value (α : Real) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3570_357038


namespace NUMINAMATH_CALUDE_gecko_lizard_insect_ratio_l3570_357080

theorem gecko_lizard_insect_ratio :
  let num_geckos : ℕ := 5
  let insects_per_gecko : ℕ := 6
  let num_lizards : ℕ := 3
  let total_insects : ℕ := 66
  let geckos_total := num_geckos * insects_per_gecko
  let lizards_total := total_insects - geckos_total
  let insects_per_lizard := lizards_total / num_lizards
  insects_per_lizard / insects_per_gecko = 2 :=
by sorry

end NUMINAMATH_CALUDE_gecko_lizard_insect_ratio_l3570_357080


namespace NUMINAMATH_CALUDE_union_complement_equality_l3570_357057

def I : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_equality : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l3570_357057


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3570_357034

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 70 →
  percentage = 50 →
  final = initial * (1 + percentage / 100) →
  final = 105 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3570_357034


namespace NUMINAMATH_CALUDE_product_of_roots_l3570_357095

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (x - 1) * (x + 4) = 22 ∧ (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3570_357095


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3570_357086

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 36, and 30, the distance between two adjacent parallel lines is 2√10. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 36 ∧ 
    chord3 = 30 ∧ 
    chord1^2 = 4 * (r^2 - (d/2)^2) ∧
    chord2^2 = 4 * (r^2 - d^2) ∧
    chord3^2 = 4 * (r^2 - (3*d/2)^2)) →
  d = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3570_357086


namespace NUMINAMATH_CALUDE_fraction_of_108_l3570_357025

theorem fraction_of_108 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 108 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_108_l3570_357025


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l3570_357056

theorem square_sum_equals_one (x y : ℝ) :
  (x^2 + y^2 + 1)^2 - 4 = 0 → x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l3570_357056


namespace NUMINAMATH_CALUDE_evaluate_nested_root_l3570_357010

theorem evaluate_nested_root : (((4 ^ (1/3)) ^ 4) ^ (1/2)) ^ 6 = 256 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_root_l3570_357010


namespace NUMINAMATH_CALUDE_square_measurement_error_l3570_357089

theorem square_measurement_error (S : ℝ) (S' : ℝ) (h : S' > 0) :
  S'^2 = S^2 * (1 + 0.0404) → (S' - S) / S * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_measurement_error_l3570_357089


namespace NUMINAMATH_CALUDE_cube_equal_angle_planes_l3570_357083

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Calculates the angle between a plane and a line -/
def angle_plane_line (p : Plane) (l : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Checks if a plane passes through a given point -/
def plane_through_point (p : Plane) (point : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Theorem: There are exactly 4 planes through vertex A of a cube such that 
    the angles between each plane and the lines AB, AD, and AA₁ are all equal -/
theorem cube_equal_angle_planes (c : Cube) : 
  ∃! (planes : Finset Plane), 
    planes.card = 4 ∧ 
    ∀ p ∈ planes, 
      plane_through_point p (c.vertices 0) ∧
      ∃ θ : ℝ, 
        angle_plane_line p (c.vertices 0, c.vertices 1) = θ ∧
        angle_plane_line p (c.vertices 0, c.vertices 3) = θ ∧
        angle_plane_line p (c.vertices 0, c.vertices 4) = θ :=
  sorry

end NUMINAMATH_CALUDE_cube_equal_angle_planes_l3570_357083


namespace NUMINAMATH_CALUDE_percentage_of_120_to_80_l3570_357019

theorem percentage_of_120_to_80 : ∃ (p : ℝ), (120 : ℝ) / 80 * 100 = p ∧ p = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_80_l3570_357019
