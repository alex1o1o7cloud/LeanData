import Mathlib

namespace NUMINAMATH_CALUDE_sum_in_arithmetic_sequence_l1168_116820

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_in_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 7 = 37 →
  a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_arithmetic_sequence_l1168_116820


namespace NUMINAMATH_CALUDE_least_possible_xy_l1168_116821

theorem least_possible_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 128 ∧ ∃ (a b : ℕ+), (a : ℕ) * (b : ℕ) = 128 ∧ (1 : ℚ) / a + (1 : ℚ) / (2 * b) = (1 : ℚ) / 8 :=
sorry

end NUMINAMATH_CALUDE_least_possible_xy_l1168_116821


namespace NUMINAMATH_CALUDE_production_days_calculation_l1168_116819

theorem production_days_calculation (n : ℕ) : 
  (∀ (k : ℕ), k ≤ n → (60 * k : ℝ) = (60 : ℝ) * k) → 
  ((60 * n + 90 : ℝ) / (n + 1) = 62) → 
  n = 14 :=
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l1168_116819


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1168_116855

theorem quadratic_roots_range (h1 : ∃ x : ℝ, Real.log x < 0)
  (h2 : ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0))
  (m : ℝ) :
  -2 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1168_116855


namespace NUMINAMATH_CALUDE_row_sum_equals_2013_squared_l1168_116804

theorem row_sum_equals_2013_squared :
  let n : ℕ := 1007
  let row_sum (k : ℕ) : ℕ := k * (2 * k - 1)
  row_sum n = 2013^2 := by
  sorry

end NUMINAMATH_CALUDE_row_sum_equals_2013_squared_l1168_116804


namespace NUMINAMATH_CALUDE_value_of_a_l1168_116800

-- Define the operation *
def star (x y : ℝ) : ℝ := x + y - x * y

-- Define a
def a : ℝ := star 1 (star 0 1)

-- Theorem statement
theorem value_of_a : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1168_116800


namespace NUMINAMATH_CALUDE_floor_abs_sum_l1168_116844

theorem floor_abs_sum : ⌊|(-3.7 : ℝ)|⌋ + |⌊(-3.7 : ℝ)⌋| = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l1168_116844


namespace NUMINAMATH_CALUDE_duke_three_pointers_l1168_116818

/-- The number of additional three-pointers Duke scored in the final game compared to his normal amount -/
def additional_three_pointers (
  points_to_tie : ℕ
  ) (points_over_record : ℕ
  ) (old_record : ℕ
  ) (free_throws : ℕ
  ) (regular_baskets : ℕ
  ) (normal_three_pointers : ℕ
  ) : ℕ :=
  let total_points := points_to_tie + points_over_record
  let points_from_free_throws := free_throws * 1
  let points_from_regular_baskets := regular_baskets * 2
  let points_from_three_pointers := total_points - (points_from_free_throws + points_from_regular_baskets)
  let three_pointers_scored := points_from_three_pointers / 3
  three_pointers_scored - normal_three_pointers

theorem duke_three_pointers :
  additional_three_pointers 17 5 257 5 4 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_duke_three_pointers_l1168_116818


namespace NUMINAMATH_CALUDE_ship_journey_day1_distance_l1168_116831

/-- Represents the distance traveled by a ship over three days -/
structure ShipJourney where
  day1_north : ℝ
  day2_east : ℝ
  day3_east : ℝ

/-- Calculates the total distance traveled by the ship -/
def total_distance (journey : ShipJourney) : ℝ :=
  journey.day1_north + journey.day2_east + journey.day3_east

/-- Theorem stating the distance traveled north on the first day -/
theorem ship_journey_day1_distance :
  ∀ (journey : ShipJourney),
    journey.day2_east = 3 * journey.day1_north →
    journey.day3_east = journey.day2_east + 110 →
    total_distance journey = 810 →
    journey.day1_north = 100 := by
  sorry

end NUMINAMATH_CALUDE_ship_journey_day1_distance_l1168_116831


namespace NUMINAMATH_CALUDE_fourier_series_sum_l1168_116898

open Real

noncomputable def y (x : ℝ) : ℝ := x * cos x

theorem fourier_series_sum : 
  ∃ (S : ℝ), S = ∑' k, (4 * k^2 + 1) / (4 * k^2 - 1)^2 ∧ S = π^2 / 8 + 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fourier_series_sum_l1168_116898


namespace NUMINAMATH_CALUDE_bee_count_l1168_116899

/-- The total number of bees in a hive after additional bees fly in -/
def total_bees (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 16 initial bees and 9 additional bees, the total is 25 -/
theorem bee_count : total_bees 16 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1168_116899


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1168_116896

theorem trigonometric_identities :
  (∀ x y : ℝ, x = 20 * π / 180 ∧ y = 40 * π / 180 →
    Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3) ∧
  (∀ z w : ℝ, z = 50 * π / 180 ∧ w = 10 * π / 180 →
    Real.sin z * (1 + Real.sqrt 3 * Real.tan w) = 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1168_116896


namespace NUMINAMATH_CALUDE_largest_number_l1168_116856

theorem largest_number (a b c d e : ℝ) : 
  a = 15467 + 3 / 5791 → 
  b = 15467 - 3 / 5791 → 
  c = 15467 * 3 / 5791 → 
  d = 15467 / (3 / 5791) → 
  e = 15467.5791 → 
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1168_116856


namespace NUMINAMATH_CALUDE_inequality_proof_l1168_116853

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha1 : a ≥ 1) (hb1 : b ≥ 1) (hc1 : c ≥ 1)
  (habcd : a * b * c * d = 1) :
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 
  1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1168_116853


namespace NUMINAMATH_CALUDE_a_plus_b_equals_two_l1168_116802

theorem a_plus_b_equals_two (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / x) →
  (2 = a + b / 1) →
  (4 = a + b / 4) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_two_l1168_116802


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l1168_116866

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2*a| + |x - 3*a|

-- Theorem 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 2 ∨ a = -2 :=
sorry

-- Theorem 2
theorem inequality_implies_m_range (m : ℝ) :
  (∀ x, ∃ a ∈ Set.Icc (-2) 2, m^2 - |m| - f x a < 0) →
  -1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l1168_116866


namespace NUMINAMATH_CALUDE_penelope_greta_ratio_l1168_116837

/-- The amount of food animals eat per day in pounds -/
structure AnimalFood where
  penelope : ℝ
  greta : ℝ
  milton : ℝ
  elmer : ℝ

/-- The conditions given in the problem -/
def problem_conditions (food : AnimalFood) : Prop :=
  food.penelope = 20 ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.elmer = food.penelope + 60

/-- The theorem to be proved -/
theorem penelope_greta_ratio (food : AnimalFood) :
  problem_conditions food → food.penelope / food.greta = 10 := by
  sorry

end NUMINAMATH_CALUDE_penelope_greta_ratio_l1168_116837


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1168_116865

/-- Theorem: For a parabola x^2 = 2py (p > 0) with a point A(m, 4) on it,
    if the distance from A to its focus is 17/4, then p = 1/2 and m = ±2. -/
theorem parabola_focus_distance (p m : ℝ) : 
  p > 0 →  -- p is positive
  m^2 = 2*p*4 →  -- A(m, 4) is on the parabola
  (m^2 + (4 - p/2)^2)^(1/2) = 17/4 →  -- Distance from A to focus is 17/4
  p = 1/2 ∧ (m = 2 ∨ m = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1168_116865


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1168_116803

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x, x > a → (x - 1) / x > 0) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1168_116803


namespace NUMINAMATH_CALUDE_books_for_girls_l1168_116806

theorem books_for_girls (num_girls num_boys total_books : ℕ) 
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375)
  (h_equal_division : ∃ (books_per_student : ℕ), 
    total_books = books_per_student * (num_girls + num_boys)) :
  ∃ (books_for_girls : ℕ), books_for_girls = 225 ∧ 
    books_for_girls = num_girls * (total_books / (num_girls + num_boys)) := by
  sorry

end NUMINAMATH_CALUDE_books_for_girls_l1168_116806


namespace NUMINAMATH_CALUDE_min_mn_tangent_line_circle_l1168_116801

/-- Given positive real numbers m and n, if the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1, then the minimum value of mn is 3 + 2√2. -/
theorem min_mn_tangent_line_circle (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_tangent : ∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 ≥ 1) :
  ∃ (min_mn : ℝ), min_mn = 3 + 2 * Real.sqrt 2 ∧ m * n ≥ min_mn := by
  sorry

end NUMINAMATH_CALUDE_min_mn_tangent_line_circle_l1168_116801


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1168_116851

-- Define the equations of the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1168_116851


namespace NUMINAMATH_CALUDE_attend_both_reunions_l1168_116873

/-- The number of people attending both reunions at the Taj Hotel -/
def both_reunions (total guests : ℕ) (oates hall : ℕ) : ℕ :=
  oates + hall - total

/-- Theorem stating the number of people attending both reunions -/
theorem attend_both_reunions : 
  both_reunions 150 70 52 = 28 := by sorry

end NUMINAMATH_CALUDE_attend_both_reunions_l1168_116873


namespace NUMINAMATH_CALUDE_unique_prime_value_l1168_116864

def f (n : ℕ) : ℤ := n^3 - 9*n^2 + 23*n - 15

theorem unique_prime_value : ∃! (n : ℕ), n > 0 ∧ Nat.Prime (Int.natAbs (f n)) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_value_l1168_116864


namespace NUMINAMATH_CALUDE_third_candidate_votes_l1168_116833

theorem third_candidate_votes : 
  let total_votes : ℕ := 23400
  let candidate1_votes : ℕ := 7636
  let candidate2_votes : ℕ := 11628
  let winning_percentage : ℚ := 49.69230769230769 / 100
  ∀ (third_candidate_votes : ℕ),
    (candidate1_votes + candidate2_votes + third_candidate_votes = total_votes) ∧
    (candidate2_votes = (winning_percentage * total_votes).floor) →
    third_candidate_votes = 4136 := by
sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l1168_116833


namespace NUMINAMATH_CALUDE_sum_of_series_equals_three_l1168_116863

theorem sum_of_series_equals_three : 
  ∑' k : ℕ+, (k : ℝ)^2 / 2^(k : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_three_l1168_116863


namespace NUMINAMATH_CALUDE_power_sum_equals_zero_l1168_116814

theorem power_sum_equals_zero : 1^2009 + (-1)^2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_zero_l1168_116814


namespace NUMINAMATH_CALUDE_constant_term_of_binomial_expansion_l1168_116807

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the constant term in the expansion of (x - 1/x)^8
def constantTerm : ℕ := binomial 8 4

-- Theorem statement
theorem constant_term_of_binomial_expansion :
  constantTerm = 70 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_binomial_expansion_l1168_116807


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1168_116858

theorem inequality_system_solution_set :
  {x : ℝ | (6 - 2*x ≥ 0) ∧ (2*x + 4 > 0)} = {x : ℝ | -2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1168_116858


namespace NUMINAMATH_CALUDE_binomial_18_10_l1168_116843

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1168_116843


namespace NUMINAMATH_CALUDE_information_spread_l1168_116836

theorem information_spread (population : ℕ) (h : population = 1000000) : 
  ∃ (n : ℕ), (2^(n+1) - 1 ≥ population) ∧ (∀ m : ℕ, m < n → 2^(m+1) - 1 < population) :=
sorry

end NUMINAMATH_CALUDE_information_spread_l1168_116836


namespace NUMINAMATH_CALUDE_lollipop_distribution_l1168_116881

theorem lollipop_distribution (num_kids : ℕ) (additional_lollipops : ℕ) (initial_lollipops : ℕ) : 
  num_kids = 42 → 
  additional_lollipops = 22 → 
  (initial_lollipops + additional_lollipops) % num_kids = 0 → 
  initial_lollipops < num_kids → 
  initial_lollipops = 62 := by
sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l1168_116881


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1168_116848

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) ↔ b = -6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1168_116848


namespace NUMINAMATH_CALUDE_milk_tea_price_proof_l1168_116867

/-- The cost of a cup of milk tea in dollars -/
def milk_tea_cost : ℝ := 2.4

/-- The cost of a slice of cake in dollars -/
def cake_slice_cost : ℝ := 0.75 * milk_tea_cost

theorem milk_tea_price_proof :
  (cake_slice_cost = 0.75 * milk_tea_cost) ∧
  (2 * cake_slice_cost + milk_tea_cost = 6) →
  milk_tea_cost = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_milk_tea_price_proof_l1168_116867


namespace NUMINAMATH_CALUDE_stating_chameleon_change_theorem_l1168_116828

/-- Represents the change in the number of chameleons of a specific color. -/
structure ChameleonChange where
  green : ℤ
  yellow : ℤ

/-- Represents the weather conditions for a month. -/
structure WeatherConditions where
  sunny_days : ℕ
  cloudy_days : ℕ

/-- 
Theorem stating that the increase in green chameleons is equal to 
the increase in yellow chameleons plus the difference between sunny and cloudy days.
-/
theorem chameleon_change_theorem (weather : WeatherConditions) (change : ChameleonChange) :
  change.yellow = 5 →
  weather.sunny_days = 18 →
  weather.cloudy_days = 12 →
  change.green = change.yellow + (weather.sunny_days - weather.cloudy_days) :=
by sorry

end NUMINAMATH_CALUDE_stating_chameleon_change_theorem_l1168_116828


namespace NUMINAMATH_CALUDE_california_new_york_ratio_l1168_116838

/-- Proves that the ratio of Coronavirus cases in California to New York is 1:2 --/
theorem california_new_york_ratio : 
  ∀ (california texas : ℕ), 
  california = texas + 400 →
  2000 + california + texas = 3600 →
  california * 2 = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_california_new_york_ratio_l1168_116838


namespace NUMINAMATH_CALUDE_triangle_segment_length_l1168_116876

theorem triangle_segment_length (a b c h x : ℝ) : 
  a = 24 → b = 45 → c = 51 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l1168_116876


namespace NUMINAMATH_CALUDE_serena_age_proof_l1168_116805

/-- Serena's current age -/
def serena_age : ℕ := 9

/-- Serena's mother's current age -/
def mother_age : ℕ := 39

/-- Years into the future when the age comparison is made -/
def years_later : ℕ := 6

theorem serena_age_proof :
  serena_age = 9 ∧
  mother_age = 39 ∧
  mother_age + years_later = 3 * (serena_age + years_later) :=
by sorry

end NUMINAMATH_CALUDE_serena_age_proof_l1168_116805


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1168_116852

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x > 0 → 3 * x + 1 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1168_116852


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1168_116817

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ x, (-1 < x ∧ x < 2) → x < 3) ∧
  ¬(∀ x, x < 3 → (-1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1168_116817


namespace NUMINAMATH_CALUDE_solve_cake_problem_l1168_116809

def cake_problem (cost_per_cake : ℕ) (john_payment : ℕ) : Prop :=
  ∃ (num_cakes : ℕ),
    cost_per_cake = 12 ∧
    john_payment = 18 ∧
    num_cakes * cost_per_cake = 2 * john_payment ∧
    num_cakes = 3

theorem solve_cake_problem :
  ∀ (cost_per_cake : ℕ) (john_payment : ℕ),
    cake_problem cost_per_cake john_payment :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cake_problem_l1168_116809


namespace NUMINAMATH_CALUDE_equation_has_real_root_l1168_116845

theorem equation_has_real_root :
  ∃ x : ℝ, (Real.sqrt (x + 16) + 4 / Real.sqrt (x + 16) = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l1168_116845


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1168_116862

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1168_116862


namespace NUMINAMATH_CALUDE_total_cakes_served_l1168_116886

theorem total_cakes_served (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
sorry

end NUMINAMATH_CALUDE_total_cakes_served_l1168_116886


namespace NUMINAMATH_CALUDE_log_inequality_specific_inequality_l1168_116897

-- Part 1
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 := by sorry

-- Part 2
theorem specific_inequality :
  6 + Real.sqrt 10 > 2 * Real.sqrt 3 + 2 := by sorry

end NUMINAMATH_CALUDE_log_inequality_specific_inequality_l1168_116897


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l1168_116893

theorem max_value_on_ellipse (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ellipse : 4 * x^2 + 9 * y^2 = 36) : 
  ∀ u v : ℝ, u > 0 → v > 0 → 4 * u^2 + 9 * v^2 = 36 → x + 2*y ≤ 5 ∧ x + 2*y = 5 → u + 2*v ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l1168_116893


namespace NUMINAMATH_CALUDE_quadratic_inequality_implication_l1168_116892

theorem quadratic_inequality_implication (x : ℝ) :
  x^2 - 5*x + 6 < 0 → 20 < x^2 + 5*x + 6 ∧ x^2 + 5*x + 6 < 30 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implication_l1168_116892


namespace NUMINAMATH_CALUDE_last_remaining_is_125_l1168_116859

/-- Represents the marking process on a list of numbers -/
def markingProcess (n : ℕ) : ℕ → Prop :=
  sorry

/-- The last remaining number after the marking process -/
def lastRemaining (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the last remaining number is 125 when starting with 150 numbers -/
theorem last_remaining_is_125 : lastRemaining 150 = 125 :=
  sorry

end NUMINAMATH_CALUDE_last_remaining_is_125_l1168_116859


namespace NUMINAMATH_CALUDE_perimeter_is_64_l1168_116887

/-- A structure formed by nine congruent squares -/
structure SquareStructure where
  /-- The side length of each square in the structure -/
  side_length : ℝ
  /-- The total area of the structure is 576 square centimeters -/
  total_area_eq : side_length ^ 2 * 9 = 576

/-- The perimeter of the square structure -/
def perimeter (s : SquareStructure) : ℝ :=
  8 * s.side_length

/-- Theorem stating that the perimeter of the structure is 64 centimeters -/
theorem perimeter_is_64 (s : SquareStructure) : perimeter s = 64 := by
  sorry

#check perimeter_is_64

end NUMINAMATH_CALUDE_perimeter_is_64_l1168_116887


namespace NUMINAMATH_CALUDE_sallys_pears_l1168_116830

theorem sallys_pears (sara_pears : ℕ) (total_pears : ℕ) (sally_pears : ℕ) :
  sara_pears = 45 →
  total_pears = 56 →
  sally_pears = total_pears - sara_pears →
  sally_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_sallys_pears_l1168_116830


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1168_116808

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d →
  c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1168_116808


namespace NUMINAMATH_CALUDE_total_marbles_l1168_116825

/-- Represents the colors of marbles in the bag -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents the bag of marbles -/
structure Bag where
  marbles : Color → ℕ
  total : ℕ
  ratio_sum : ℕ

/-- The theorem stating the total number of marbles in the bag -/
theorem total_marbles (b : Bag) 
  (h1 : b.marbles Color.Red + b.marbles Color.Blue + b.marbles Color.Green + b.marbles Color.Yellow = b.total)
  (h2 : b.marbles Color.Red = 1 * b.ratio_sum)
  (h3 : b.marbles Color.Blue = 3 * b.ratio_sum)
  (h4 : b.marbles Color.Green = 4 * b.ratio_sum)
  (h5 : b.marbles Color.Yellow = 2 * b.ratio_sum)
  (h6 : b.marbles Color.Green = 40)
  : b.total = 100 := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l1168_116825


namespace NUMINAMATH_CALUDE_scientific_notation_of_190_million_l1168_116883

theorem scientific_notation_of_190_million :
  (190000000 : ℝ) = 1.9 * (10 : ℝ)^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_190_million_l1168_116883


namespace NUMINAMATH_CALUDE_max_inscribed_circle_radius_l1168_116857

/-- The maximum radius of an inscribed circle centered at (0,0) in the curve |y| = 1 - a x^2 where |x| ≤ 1/√a -/
noncomputable def f (a : ℝ) : ℝ :=
  if a ≤ 1/2 then 1 else Real.sqrt (4*a - 1) / (2*a)

/-- The curve C defined by |y| = 1 - a x^2 where |x| ≤ 1/√a -/
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = 1 - a * p.1^2 ∧ |p.1| ≤ 1/Real.sqrt a}

theorem max_inscribed_circle_radius (a : ℝ) (ha : a > 0) :
  ∀ r : ℝ, r > 0 → (∀ p : ℝ × ℝ, p ∈ C a → (p.1^2 + p.2^2 ≥ r^2)) → r ≤ f a :=
sorry

end NUMINAMATH_CALUDE_max_inscribed_circle_radius_l1168_116857


namespace NUMINAMATH_CALUDE_newOp_examples_l1168_116826

-- Define the new operation
def newOp (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- Theorem statement
theorem newOp_examples : 
  (newOp (-5) 4 = -42) ∧ (newOp (-3) (-6) = 39) := by
  sorry

end NUMINAMATH_CALUDE_newOp_examples_l1168_116826


namespace NUMINAMATH_CALUDE_tree_height_problem_l1168_116869

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 20 →  -- One tree is 20 feet taller than the other
  h₁ / h₂ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 50  -- The shorter tree is 50 feet tall
:= by sorry

end NUMINAMATH_CALUDE_tree_height_problem_l1168_116869


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l1168_116890

theorem rectangle_area_preservation (original_length original_width : ℝ) 
  (h_length : original_length = 140)
  (h_width : original_width = 40)
  (length_increase_percent : ℝ) 
  (h_increase : length_increase_percent = 30) :
  let new_length := original_length * (1 + length_increase_percent / 100)
  let new_width := (original_length * original_width) / new_length
  let width_decrease_percent := (original_width - new_width) / original_width * 100
  ∃ ε > 0, abs (width_decrease_percent - 23.08) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l1168_116890


namespace NUMINAMATH_CALUDE_project_payment_l1168_116842

/-- Represents the hourly wage of candidate q -/
def q_wage : ℝ := 14

/-- Represents the hourly wage of candidate p -/
def p_wage : ℝ := 21

/-- Represents the number of hours candidate p needs to complete the job -/
def p_hours : ℝ := 20

/-- Represents the number of hours candidate q needs to complete the job -/
def q_hours : ℝ := p_hours + 10

/-- The total payment for the project -/
def total_payment : ℝ := 420

theorem project_payment :
  (p_wage = q_wage * 1.5) ∧
  (p_wage = q_wage + 7) ∧
  (q_hours = p_hours + 10) ∧
  (p_wage * p_hours = q_wage * q_hours) →
  total_payment = 420 := by sorry

end NUMINAMATH_CALUDE_project_payment_l1168_116842


namespace NUMINAMATH_CALUDE_triangle_count_bound_l1168_116884

/-- A structure representing a configuration of points and equilateral triangles on a plane. -/
structure PointTriangleConfig where
  n : ℕ  -- number of points
  k : ℕ  -- number of equilateral triangles
  n_gt_3 : n > 3
  convex_n_gon : Bool  -- represents that the n points form a convex n-gon
  unit_triangles : Bool  -- represents that the k triangles are equilateral with side length 1

/-- Theorem stating that the number of equilateral triangles is less than 2/3 of the number of points. -/
theorem triangle_count_bound (config : PointTriangleConfig) : config.k < 2 * config.n / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_bound_l1168_116884


namespace NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l1168_116815

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_first_four_composites_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l1168_116815


namespace NUMINAMATH_CALUDE_true_discount_example_l1168_116891

/-- Given a banker's discount and face value, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (face_value : ℚ) : ℚ :=
  (face_value * bankers_discount) / (face_value + bankers_discount)

/-- Theorem stating that for given values, the true discount is 480 -/
theorem true_discount_example : true_discount 576 2880 = 480 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_example_l1168_116891


namespace NUMINAMATH_CALUDE_scheme2_more_cost_effective_l1168_116895

/-- Represents the cost of purchasing x pens and (x+10) notebooks under scheme 1 -/
def y₁ (x : ℝ) : ℝ := 15 * x + 40

/-- Represents the cost of purchasing x pens and (x+10) notebooks under scheme 2 -/
def y₂ (x : ℝ) : ℝ := 15.2 * x + 32

/-- Theorem stating that scheme 2 is always more cost-effective than scheme 1 -/
theorem scheme2_more_cost_effective (x : ℝ) (h : x ≥ 0) : y₂ x ≤ y₁ x := by
  sorry

end NUMINAMATH_CALUDE_scheme2_more_cost_effective_l1168_116895


namespace NUMINAMATH_CALUDE_matts_climbing_speed_l1168_116894

/-- Prove Matt's climbing speed given Jason's speed and their height difference after 7 minutes -/
theorem matts_climbing_speed 
  (jason_speed : ℝ) 
  (time : ℝ) 
  (height_diff : ℝ) 
  (h1 : jason_speed = 12)
  (h2 : time = 7)
  (h3 : height_diff = 42) :
  ∃ (matt_speed : ℝ), 
    matt_speed = 6 ∧ 
    jason_speed * time = matt_speed * time + height_diff :=
by sorry

end NUMINAMATH_CALUDE_matts_climbing_speed_l1168_116894


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1168_116827

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1168_116827


namespace NUMINAMATH_CALUDE_unique_prime_triple_solution_l1168_116824

theorem unique_prime_triple_solution :
  ∀ (p q r k : ℕ),
    Nat.Prime p → Nat.Prime q → Nat.Prime r →
    p ≠ q → p ≠ r → q ≠ r →
    (p * q - k) % r = 0 →
    (q * r - k) % p = 0 →
    (r * p - k) % q = 0 →
    p * q > k →
    (p = 2 ∧ q = 3 ∧ r = 5 ∧ k = 1) ∨
    (p = 2 ∧ q = 5 ∧ r = 3 ∧ k = 1) ∨
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ k = 1) ∨
    (p = 3 ∧ q = 5 ∧ r = 2 ∧ k = 1) ∨
    (p = 5 ∧ q = 2 ∧ r = 3 ∧ k = 1) ∨
    (p = 5 ∧ q = 3 ∧ r = 2 ∧ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triple_solution_l1168_116824


namespace NUMINAMATH_CALUDE_boy_late_to_school_l1168_116868

/-- Proves that a boy traveling to school was 1 hour late on the first day given specific conditions -/
theorem boy_late_to_school (distance : ℝ) (speed_day1 speed_day2 : ℝ) (early_time : ℝ) : 
  distance = 60 ∧ 
  speed_day1 = 10 ∧ 
  speed_day2 = 20 ∧ 
  early_time = 1 ∧
  distance / speed_day2 + early_time = distance / speed_day1 - 1 →
  distance / speed_day1 - (distance / speed_day2 + early_time) = 1 :=
by
  sorry

#check boy_late_to_school

end NUMINAMATH_CALUDE_boy_late_to_school_l1168_116868


namespace NUMINAMATH_CALUDE_simplify_expression_l1168_116888

theorem simplify_expression (y : ℝ) : 8*y - 3 + 2*y + 15 = 10*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1168_116888


namespace NUMINAMATH_CALUDE_vector_parallelism_l1168_116861

theorem vector_parallelism (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, x]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1168_116861


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_subset_l1168_116829

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |2*x + 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 1} = {x : ℝ | -1 ≤ x ∧ x ≤ -1/3} :=
sorry

-- Part 2
def P (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ -2*x + 1}

theorem range_of_a_given_subset :
  (∀ a : ℝ, Set.Icc (-1 : ℝ) (-1/4) ⊆ P a) →
  {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) (-1/4), f a x ≤ -2*x + 1} = Set.Icc (-3/4 : ℝ) (5/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_subset_l1168_116829


namespace NUMINAMATH_CALUDE_larry_wins_probability_l1168_116823

theorem larry_wins_probability (larry_prob julius_prob : ℚ) : 
  larry_prob = 3/5 →
  julius_prob = 2/5 →
  let win_prob := larry_prob / (1 - (1 - larry_prob) * (1 - julius_prob))
  win_prob = 11/15 := by
sorry

end NUMINAMATH_CALUDE_larry_wins_probability_l1168_116823


namespace NUMINAMATH_CALUDE_max_candy_leftover_l1168_116849

theorem max_candy_leftover (x : ℕ) (h1 : x > 120) : 
  ∃ (q : ℕ), x = 12 * (10 + q) + 11 ∧ 
  ∀ (r : ℕ), r < 11 → ∃ (q' : ℕ), x ≠ 12 * (10 + q') + r :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l1168_116849


namespace NUMINAMATH_CALUDE_remainder_3m_mod_5_l1168_116810

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3m_mod_5_l1168_116810


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1168_116889

/-- Proves that the speed of a boat in still water is 22 km/hr, given that it travels 54 km downstream in 2 hours with a stream speed of 5 km/hr. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 54)
  (h3 : downstream_time = 2)
  : ∃ (still_water_speed : ℝ),
    still_water_speed = 22 ∧
    downstream_distance = (still_water_speed + stream_speed) * downstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1168_116889


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1168_116885

-- Define the curve C1
def C1 (t : ℝ) : ℝ × ℝ := (8 * t^2, 8 * t)

-- Define the circle C2
def C2 (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the focus of C1
def focus_C1 : ℝ × ℝ := (2, 0)

-- Define the line with slope 1 passing through the focus of C1
def tangent_line (x y : ℝ) : Prop := y - x = -2

-- State the theorem
theorem tangent_circle_radius 
  (r : ℝ) 
  (hr : r > 0) 
  (h_tangent : ∃ p : ℝ × ℝ, p ∈ C2 r ∧ tangent_line p.1 p.2) : 
  r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1168_116885


namespace NUMINAMATH_CALUDE_max_value_of_expression_achievable_max_value_l1168_116877

theorem max_value_of_expression (n : ℕ) : 
  10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ 870 := by
  sorry

theorem achievable_max_value : 
  ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - n) = 870 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_achievable_max_value_l1168_116877


namespace NUMINAMATH_CALUDE_company_workforce_l1168_116839

theorem company_workforce (total_workers : ℕ) 
  (h1 : total_workers % 3 = 0)  -- Ensures total_workers is divisible by 3
  (h2 : total_workers / 3 * 2 % 5 = 0)  -- Ensures (2/3) of total_workers is divisible by 5
  (h3 : 120 * 30 / 8 = total_workers)  -- Relation between total workers and male workers
  : (total_workers - 120 : ℕ) = 330 := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l1168_116839


namespace NUMINAMATH_CALUDE_movie_children_count_prove_movie_children_count_l1168_116840

theorem movie_children_count : ℕ → Prop :=
  fun num_children =>
    let total_cost : ℕ := 76
    let num_adults : ℕ := 5
    let adult_ticket_cost : ℕ := 10
    let child_ticket_cost : ℕ := 7
    let concession_cost : ℕ := 12
    
    (num_adults * adult_ticket_cost + num_children * child_ticket_cost + concession_cost = total_cost) →
    num_children = 2

theorem prove_movie_children_count : ∃ (n : ℕ), movie_children_count n :=
  sorry

end NUMINAMATH_CALUDE_movie_children_count_prove_movie_children_count_l1168_116840


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1168_116834

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 2 →
  B = π / 3 ∧
  (∃ (S : ℝ), S = Real.sqrt 3 ∧ ∀ (S' : ℝ), S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1168_116834


namespace NUMINAMATH_CALUDE_chantel_bracelet_count_l1168_116832

/-- The number of bracelets Chantel has at the end of the process --/
def final_bracelet_count (initial_daily_production : ℕ) (initial_days : ℕ) 
  (first_giveaway : ℕ) (second_daily_production : ℕ) (second_days : ℕ) 
  (second_giveaway : ℕ) : ℕ :=
  initial_daily_production * initial_days - first_giveaway + 
  second_daily_production * second_days - second_giveaway

/-- Theorem stating that Chantel ends up with 13 bracelets --/
theorem chantel_bracelet_count : 
  final_bracelet_count 2 5 3 3 4 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_chantel_bracelet_count_l1168_116832


namespace NUMINAMATH_CALUDE_parking_lot_vehicles_l1168_116816

-- Define the initial number of cars and trucks
def initial_cars : ℝ := 14
def initial_trucks : ℝ := 49

-- Define the changes in the parking lot
def cars_left : ℕ := 3
def trucks_arrived : ℕ := 6

-- Define the ratios
def initial_ratio : ℝ := 3.5
def final_ratio : ℝ := 2.3

-- Theorem statement
theorem parking_lot_vehicles :
  -- Initial condition
  initial_cars = initial_ratio * initial_trucks ∧
  -- Final condition after changes
  (initial_cars - cars_left) = final_ratio * (initial_trucks + trucks_arrived) →
  -- Conclusion: Total number of vehicles originally parked
  initial_cars + initial_trucks = 63 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_parking_lot_vehicles_l1168_116816


namespace NUMINAMATH_CALUDE_max_coefficient_of_expansion_l1168_116882

theorem max_coefficient_of_expansion : 
  ∃ (a b c d : ℕ+), 
    (∀ x : ℝ, (6 * x + 3)^3 = a * x^3 + b * x^2 + c * x + d) ∧ 
    (max a (max b (max c d)) = 324) := by
  sorry

end NUMINAMATH_CALUDE_max_coefficient_of_expansion_l1168_116882


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1168_116811

/-- 
Given a geometric sequence where the fifth term is 64 and the sixth term is 128,
prove that the first term of the sequence is 4.
-/
theorem geometric_sequence_first_term (a b c d : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    b = a * r ∧ 
    c = b * r ∧ 
    d = c * r ∧ 
    64 = d * r ∧ 
    128 = 64 * r) → 
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1168_116811


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l1168_116822

theorem consecutive_numbers_product (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + d = 109) → (b * c = 2970) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l1168_116822


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1168_116872

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1168_116872


namespace NUMINAMATH_CALUDE_grid_broken_lines_theorem_l1168_116841

/-- Represents a grid of cells -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a broken line in the grid -/
structure BrokenLine :=
  (length : ℕ)

/-- Checks if it's possible to construct a set of broken lines in a grid -/
def canConstructBrokenLines (g : Grid) (lines : List BrokenLine) : Prop :=
  -- The actual implementation would be complex and is omitted
  sorry

theorem grid_broken_lines_theorem (g : Grid) :
  g.width = 11 ∧ g.height = 1 →
  (canConstructBrokenLines g (List.replicate 8 ⟨5⟩)) ∧
  ¬(canConstructBrokenLines g (List.replicate 5 ⟨8⟩)) :=
by sorry

end NUMINAMATH_CALUDE_grid_broken_lines_theorem_l1168_116841


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l1168_116879

/-- Represents a pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {17, 23, 26, 29, 35}

/-- The area of a CornerCutPentagon is 895 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  895

/-- The area of a CornerCutPentagon is correct -/
theorem corner_cut_pentagon_area_is_correct (p : CornerCutPentagon) :
  corner_cut_pentagon_area p = 895 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l1168_116879


namespace NUMINAMATH_CALUDE_solve_for_c_l1168_116860

theorem solve_for_c (x y c : ℝ) (h1 : x / (2 * y) = 5 / 2) (h2 : (7 * x + 4 * y) / c = 13) :
  c = 3 * y := by
sorry

end NUMINAMATH_CALUDE_solve_for_c_l1168_116860


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1168_116813

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let a : ℝ := 3
  let b : ℝ := -12
  let c : ℝ := 9
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1168_116813


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l1168_116880

/-- Two sequences of positive integers satisfying the given conditions -/
def SequencePair : Type :=
  { pair : (ℕ → ℕ) × (ℕ → ℕ) //
    (∀ n, pair.1 n > 0 ∧ pair.2 n > 0) ∧
    pair.1 0 ≥ 2 ∧ pair.2 0 ≥ 2 ∧
    (∀ n, pair.1 (n + 1) = Nat.gcd (pair.1 n) (pair.2 n) + 1) ∧
    (∀ n, pair.2 (n + 1) = Nat.lcm (pair.1 n) (pair.2 n) - 1) }

/-- The sequence a_n is eventually periodic -/
theorem sequence_eventually_periodic (seq : SequencePair) :
  ∃ (N t : ℕ), t > 0 ∧ ∀ n ≥ N, seq.1.1 (n + t) = seq.1.1 n :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l1168_116880


namespace NUMINAMATH_CALUDE_distance_between_locations_l1168_116878

theorem distance_between_locations (speed_A speed_B : ℝ) (time : ℝ) (remaining_distance : ℝ) :
  speed_B = (4/5) * speed_A →
  time = 3 →
  remaining_distance = 3 →
  ∃ (distance_AB : ℝ),
    distance_AB = speed_A * time + speed_B * time + remaining_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l1168_116878


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_squared_l1168_116870

theorem arithmetic_geometric_mean_squared (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_squared_l1168_116870


namespace NUMINAMATH_CALUDE_saree_stripes_l1168_116874

theorem saree_stripes (brown_stripes : ℕ) (gold_stripes : ℕ) (blue_stripes : ℕ) 
  (h1 : gold_stripes = 3 * brown_stripes)
  (h2 : blue_stripes = 5 * gold_stripes)
  (h3 : brown_stripes = 4) : 
  blue_stripes = 60 := by
  sorry

end NUMINAMATH_CALUDE_saree_stripes_l1168_116874


namespace NUMINAMATH_CALUDE_complex_number_on_line_l1168_116875

theorem complex_number_on_line (a : ℝ) : 
  (Complex.I * (Complex.I⁻¹ * a + (1 - Complex.I) / 2)).re + 
  (Complex.I * (Complex.I⁻¹ * a + (1 - Complex.I) / 2)).im = 0 → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_on_line_l1168_116875


namespace NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l1168_116835

theorem square_area_ratio_when_tripled (s : ℝ) (h : s > 0) :
  (s^2) / ((3*s)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l1168_116835


namespace NUMINAMATH_CALUDE_undergrad_sample_count_l1168_116850

/-- Represents the number of undergraduate students in a stratified sample -/
def undergrad_sample_size (total_population : ℕ) (undergrad_population : ℕ) (sample_size : ℕ) : ℕ :=
  (undergrad_population * sample_size) / total_population

/-- Theorem stating the number of undergraduate students in the stratified sample -/
theorem undergrad_sample_count :
  undergrad_sample_size 5600 3000 280 = 150 := by
  sorry

end NUMINAMATH_CALUDE_undergrad_sample_count_l1168_116850


namespace NUMINAMATH_CALUDE_sachin_age_l1168_116846

/-- Represents the ages of Sachin, Rahul, and Praveen -/
structure Ages where
  sachin : ℝ
  rahul : ℝ
  praveen : ℝ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.rahul = ages.sachin + 7 ∧
  ages.sachin / ages.rahul = 7 / 9 ∧
  ages.praveen = 2 * ages.rahul ∧
  ages.sachin / ages.rahul = 7 / 9 ∧
  ages.rahul / ages.praveen = 9 / 18

/-- Theorem stating that if the ages satisfy the conditions, then Sachin's age is 24.5 -/
theorem sachin_age (ages : Ages) : 
  satisfiesConditions ages → ages.sachin = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l1168_116846


namespace NUMINAMATH_CALUDE_floor_sum_example_l1168_116871

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1168_116871


namespace NUMINAMATH_CALUDE_shoe_factory_production_l1168_116812

/-- The monthly production plan of a shoe factory. -/
def monthly_plan : ℝ := 5000

/-- The production in the first week as a fraction of the monthly plan. -/
def first_week : ℝ := 0.2

/-- The production in the second week as a fraction of the first week's production. -/
def second_week : ℝ := 1.2

/-- The production in the third week as a fraction of the first two weeks' combined production. -/
def third_week : ℝ := 0.6

/-- The production in the fourth week in pairs of shoes. -/
def fourth_week : ℝ := 1480

/-- Theorem stating that the given production schedule results in the monthly plan. -/
theorem shoe_factory_production :
  first_week * monthly_plan +
  second_week * first_week * monthly_plan +
  third_week * (first_week * monthly_plan + second_week * first_week * monthly_plan) +
  fourth_week = monthly_plan := by sorry

end NUMINAMATH_CALUDE_shoe_factory_production_l1168_116812


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1168_116847

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n > 2 → 
  interior_angle = 160 →
  (n : ℝ) * interior_angle = 180 * (n - 2) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1168_116847


namespace NUMINAMATH_CALUDE_brand_z_fraction_fraction_to_percentage_l1168_116854

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of brand Z gasoline
  y : ℚ  -- Amount of brand Y gasoline

/-- Fills the tank with brand Z gasoline when empty -/
def initial_fill : TankState :=
  { z := 1, y := 0 }

/-- Fills the tank with brand Y when 1/4 empty -/
def first_refill (s : TankState) : TankState :=
  { z := 3/4 * s.z, y := 1/4 + 3/4 * s.y }

/-- Fills the tank with brand Z when half empty -/
def second_refill (s : TankState) : TankState :=
  { z := 1/2 + 1/2 * s.z, y := 1/2 * s.y }

/-- Fills the tank with brand Y when half empty -/
def third_refill (s : TankState) : TankState :=
  { z := 1/2 * s.z, y := 1/2 + 1/2 * s.y }

/-- The final state of the tank after all refills -/
def final_state : TankState :=
  third_refill (second_refill (first_refill initial_fill))

/-- Theorem stating that the fraction of brand Z gasoline in the final state is 7/16 -/
theorem brand_z_fraction :
  final_state.z / (final_state.z + final_state.y) = 7/16 := by
  sorry

/-- Theorem stating that 7/16 is equivalent to 43.75% -/
theorem fraction_to_percentage :
  (7/16 : ℚ) = 43.75/100 := by
  sorry

end NUMINAMATH_CALUDE_brand_z_fraction_fraction_to_percentage_l1168_116854
