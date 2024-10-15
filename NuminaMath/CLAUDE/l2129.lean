import Mathlib

namespace NUMINAMATH_CALUDE_max_weight_is_6250_l2129_212933

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 1250

/-- The maximum weight of crates on a single trip -/
def max_total_weight : ℕ := max_crates * min_crate_weight

/-- Theorem stating that the maximum weight of crates on a single trip is 6250 kg -/
theorem max_weight_is_6250 : max_total_weight = 6250 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_is_6250_l2129_212933


namespace NUMINAMATH_CALUDE_permutation_absolute_difference_equality_l2129_212947

theorem permutation_absolute_difference_equality :
  ∀ (a : Fin 2011 → Fin 2011), Function.Bijective a →
  ∃ j k : Fin 2011, j < k ∧ |a j - j| = |a k - k| :=
by
  sorry

end NUMINAMATH_CALUDE_permutation_absolute_difference_equality_l2129_212947


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2129_212952

theorem fraction_sum_equality (m n p : ℝ) 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2129_212952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2129_212987

/-- Given an arithmetic sequence with first three terms 2x - 3, 3x + 1, and 5x + k,
    prove that k = 5 - x makes these terms form an arithmetic sequence. -/
theorem arithmetic_sequence_condition (x k : ℝ) : 
  let a₁ := 2*x - 3
  let a₂ := 3*x + 1
  let a₃ := 5*x + k
  (a₂ - a₁ = a₃ - a₂) → k = 5 - x := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2129_212987


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l2129_212981

/-- Represents the average birth rate in people per two seconds -/
def average_birth_rate : ℝ := 4

/-- Represents the death rate in people per two seconds -/
def death_rate : ℝ := 2

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the net population increase in one day -/
def net_increase_per_day : ℕ := 86400

theorem birth_rate_calculation :
  average_birth_rate = 4 :=
by
  sorry

#check birth_rate_calculation

end NUMINAMATH_CALUDE_birth_rate_calculation_l2129_212981


namespace NUMINAMATH_CALUDE_decimal_29_to_binary_l2129_212990

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_29_to_binary :
  decimal_to_binary 29 = [1, 1, 1, 0, 1] := by sorry

end NUMINAMATH_CALUDE_decimal_29_to_binary_l2129_212990


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l2129_212957

theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 113 →
  jake_weight - 33 = 2 * sister_weight →
  jake_weight + sister_weight = 153 := by
sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l2129_212957


namespace NUMINAMATH_CALUDE_intersection_solution_l2129_212937

/-- Given two linear functions that intersect at x = 2, prove that the solution
    to their system of equations is (2, 2) -/
theorem intersection_solution (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ 2 * x - 2
  let g : ℝ → ℝ := fun x ↦ a * x + b
  (f 2 = g 2) →
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 2 ∧ 2 * p.1 - p.2 = 2 ∧ p.2 = a * p.1 + b) :=
by sorry

end NUMINAMATH_CALUDE_intersection_solution_l2129_212937


namespace NUMINAMATH_CALUDE_toy_store_revenue_l2129_212997

theorem toy_store_revenue (december : ℝ) (november january : ℝ) 
  (h1 : november = (3/5) * december) 
  (h2 : january = (1/3) * november) : 
  december = (5/2) * ((november + january) / 2) := by
  sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l2129_212997


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l2129_212904

theorem max_pairs_after_loss (initial_pairs : ℕ) (lost_shoes : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 24 →
  lost_shoes = 9 →
  max_pairs = initial_pairs - (lost_shoes / 2) →
  max_pairs = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_after_loss_l2129_212904


namespace NUMINAMATH_CALUDE_inequality_range_l2129_212986

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(a*x - 1) < (1/3 : ℝ)^(a*x^2)) ↔ -4 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2129_212986


namespace NUMINAMATH_CALUDE_joker_probability_l2129_212946

/-- A deck of cards with Jokers -/
structure DeckWithJokers where
  total_cards : ℕ
  joker_cards : ℕ
  unique_cards : Prop
  shuffled : Prop

/-- The probability of drawing a specific card type from a deck -/
def probability_of_draw (deck : DeckWithJokers) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / deck.total_cards

/-- Our specific deck configuration -/
def our_deck : DeckWithJokers := {
  total_cards := 54
  joker_cards := 2
  unique_cards := True
  shuffled := True
}

/-- Theorem: The probability of drawing a Joker from our deck is 1/27 -/
theorem joker_probability :
  probability_of_draw our_deck our_deck.joker_cards = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_joker_probability_l2129_212946


namespace NUMINAMATH_CALUDE_ali_money_problem_l2129_212923

theorem ali_money_problem (initial_money : ℝ) : 
  (initial_money / 2 - (initial_money / 2) / 3 = 160) → initial_money = 480 := by
  sorry

end NUMINAMATH_CALUDE_ali_money_problem_l2129_212923


namespace NUMINAMATH_CALUDE_triangle_sin_c_equals_one_l2129_212991

theorem triangle_sin_c_equals_one (a b c A B C : ℝ) : 
  a = 1 → 
  b = Real.sqrt 3 → 
  A + C = 2 * B → 
  0 < a ∧ 0 < b ∧ 0 < c → 
  0 < A ∧ 0 < B ∧ 0 < C → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A → 
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B → 
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C → 
  Real.sin C = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_sin_c_equals_one_l2129_212991


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l2129_212934

/-- Proves that given a meal with a 12% sales tax, an 18% tip on the original price,
    and a total cost of $33.00, the original cost of the meal before tax and tip is $25.5. -/
theorem meal_cost_calculation (original_cost : ℝ) : 
  let tax_rate : ℝ := 0.12
  let tip_rate : ℝ := 0.18
  let total_cost : ℝ := 33.00
  (1 + tax_rate + tip_rate) * original_cost = total_cost → original_cost = 25.5 := by
sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l2129_212934


namespace NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l2129_212912

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := x^2 + a*x + a - 1/2 = 0

-- Define the set of possible values for a
def valid_a_set : Set ℝ := {a | a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2}

-- Define the inequality
def inequality (m t a x₁ x₂ : ℝ) : Prop :=
  m^2 + t*m + 4*Real.sqrt 2 + 6 ≥ (x₁ - 3*x₂)*(x₂ - 3*x₁)

theorem quadratic_roots_and_inequality :
  ∀ a ∈ valid_a_set,
  ∀ x₁ x₂ : ℝ,
  quadratic_equation a x₁ ∧ quadratic_equation a x₂ →
  (∀ t ∈ Set.Icc (-1 : ℝ) 1,
    ∃ m : ℝ, inequality m t a x₁ x₂) ↔
  ∃ m : ℝ, m ≤ -1 ∨ m = 0 ∨ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l2129_212912


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2129_212975

theorem cube_equation_solution (a d : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * d) : d = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2129_212975


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2129_212910

theorem roots_sum_of_squares (m n a b : ℝ) : 
  (∀ x, x^2 - m*x + n = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = m^2 - 2*n := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2129_212910


namespace NUMINAMATH_CALUDE_f_of_two_eq_two_fifths_l2129_212993

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.sin x * Real.cos x

theorem f_of_two_eq_two_fifths : f (Real.arctan 2) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_eq_two_fifths_l2129_212993


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l2129_212956

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ ∀ x, f x ≠ 0 := by sorry

theorem cubic_equation_negation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  apply negation_of_existence

end NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l2129_212956


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2129_212906

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2129_212906


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l2129_212989

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 4 * x - 19 ∧ 2 * x + y = 95 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l2129_212989


namespace NUMINAMATH_CALUDE_lexi_run_distance_l2129_212970

/-- Proves that running 13 laps on a quarter-mile track equals 3.25 miles -/
theorem lexi_run_distance (lap_length : ℚ) (num_laps : ℕ) : 
  lap_length = 1/4 → num_laps = 13 → lap_length * num_laps = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_lexi_run_distance_l2129_212970


namespace NUMINAMATH_CALUDE_sugar_needed_proof_l2129_212979

/-- Given a recipe requiring a total amount of flour, with some flour already added,
    and the remaining flour needed being 2 cups more than the sugar needed,
    prove that the amount of sugar needed is correct. -/
theorem sugar_needed_proof 
  (total_flour : ℕ)  -- Total flour needed
  (added_flour : ℕ)  -- Flour already added
  (h1 : total_flour = 11)  -- Total flour is 11 cups
  (h2 : added_flour = 2)   -- 2 cups of flour already added
  : 
  total_flour - added_flour - 2 = 7  -- Sugar needed is 7 cups
  := by sorry

end NUMINAMATH_CALUDE_sugar_needed_proof_l2129_212979


namespace NUMINAMATH_CALUDE_derivative_at_point_is_constant_l2129_212967

/-- The derivative of a function at a point is a constant value. -/
theorem derivative_at_point_is_constant (f : ℝ → ℝ) (a : ℝ) : 
  ∃ (c : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |x - a| ≠ 0 → 
    |(f x - f a) / (x - a) - c| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_at_point_is_constant_l2129_212967


namespace NUMINAMATH_CALUDE_selection_theorem_l2129_212966

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of workers -/
def total_workers : ℕ := 11

/-- The number of workers who can do typesetting -/
def typesetting_workers : ℕ := 7

/-- The number of workers who can do printing -/
def printing_workers : ℕ := 6

/-- The number of workers to be selected for each task -/
def workers_per_task : ℕ := 4

/-- The number of ways to select workers for typesetting and printing -/
def selection_ways : ℕ := 
  choose typesetting_workers workers_per_task * 
  choose (total_workers - workers_per_task) workers_per_task +
  choose (printing_workers - workers_per_task + 1) (printing_workers - workers_per_task) * 
  choose 2 1 * 
  choose (typesetting_workers - 1) workers_per_task +
  choose (printing_workers - workers_per_task + 2) (printing_workers - workers_per_task) * 
  choose (typesetting_workers - 2) workers_per_task * 
  choose 2 2

theorem selection_theorem : selection_ways = 185 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l2129_212966


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2129_212927

/-- An isosceles right triangle with perimeter 3p has area (153 - 108√2) / 2 * p^2 -/
theorem isosceles_right_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let leg := (9 * p - 6 * p * Real.sqrt 2)
  let area := (1 / 2) * leg ^ 2
  area = (153 - 108 * Real.sqrt 2) / 2 * p ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2129_212927


namespace NUMINAMATH_CALUDE_joannas_family_money_ratio_l2129_212913

/-- Prove that given the conditions of Joanna's family's money, the ratio of her brother's money to Joanna's money is 3:1 -/
theorem joannas_family_money_ratio :
  ∀ (brother_multiple : ℚ),
  (8 : ℚ) + 8 * brother_multiple + 4 = 36 →
  brother_multiple = 3 :=
by sorry

end NUMINAMATH_CALUDE_joannas_family_money_ratio_l2129_212913


namespace NUMINAMATH_CALUDE_min_sum_intercepts_l2129_212903

/-- A line passing through (1, 1) with positive intercepts -/
structure LineThroughOneOne where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 1 / b = 1

/-- The sum of intercepts of a line -/
def sumOfIntercepts (l : LineThroughOneOne) : ℝ := l.a + l.b

/-- The equation x + y - 2 = 0 minimizes the sum of intercepts -/
theorem min_sum_intercepts :
  ∀ l : LineThroughOneOne, sumOfIntercepts l ≥ 4 ∧
  (sumOfIntercepts l = 4 ↔ l.a = 2 ∧ l.b = 2) :=
sorry

end NUMINAMATH_CALUDE_min_sum_intercepts_l2129_212903


namespace NUMINAMATH_CALUDE_original_number_proof_l2129_212907

theorem original_number_proof :
  ∃! x : ℕ, (x + 2) % 17 = 0 ∧ x < 17 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2129_212907


namespace NUMINAMATH_CALUDE_not_divisible_by_1955_l2129_212958

theorem not_divisible_by_1955 : ∀ n : ℕ, ¬(1955 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_1955_l2129_212958


namespace NUMINAMATH_CALUDE_bicycle_cost_is_150_l2129_212973

/-- The cost of the bicycle Patrick wants to buy. -/
def bicycle_cost : ℕ := 150

/-- The amount Patrick saved, which is half the price of the bicycle. -/
def patricks_savings : ℕ := bicycle_cost / 2

/-- The amount Patrick lent to his friend. -/
def lent_amount : ℕ := 50

/-- The amount Patrick has left after lending money to his friend. -/
def remaining_amount : ℕ := 25

/-- Theorem stating that the bicycle cost is 150, given the conditions. -/
theorem bicycle_cost_is_150 :
  patricks_savings - lent_amount = remaining_amount →
  bicycle_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_is_150_l2129_212973


namespace NUMINAMATH_CALUDE_f_monotonic_iff_a_in_range_l2129_212996

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x - 7

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem f_monotonic_iff_a_in_range (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_iff_a_in_range_l2129_212996


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2129_212926

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a-2, 2*a-2}
  A ⊆ B → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2129_212926


namespace NUMINAMATH_CALUDE_students_per_group_l2129_212995

theorem students_per_group 
  (total_students : ℕ) 
  (num_teachers : ℕ) 
  (h1 : total_students = 256) 
  (h2 : num_teachers = 8) 
  (h3 : num_teachers > 0) :
  total_students / num_teachers = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l2129_212995


namespace NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_percentages_l2129_212908

/-- Represents different types of charts --/
inductive ChartType
| PieChart
| LineChart
| BarChart

/-- Represents characteristics of data --/
structure DataCharacteristics where
  is_percentage : Bool
  total_is_100_percent : Bool
  part_whole_relationship_important : Bool

/-- Determines the most appropriate chart type based on data characteristics --/
def most_appropriate_chart (data : DataCharacteristics) : ChartType :=
  if data.is_percentage ∧ data.total_is_100_percent ∧ data.part_whole_relationship_important then
    ChartType.PieChart
  else
    ChartType.BarChart

/-- Theorem stating that a pie chart is most appropriate for percentage data summing to 100% 
    where the part-whole relationship is important --/
theorem pie_chart_most_appropriate_for_percentages 
  (data : DataCharacteristics) 
  (h1 : data.is_percentage = true) 
  (h2 : data.total_is_100_percent = true)
  (h3 : data.part_whole_relationship_important = true) : 
  most_appropriate_chart data = ChartType.PieChart :=
by
  sorry

#check pie_chart_most_appropriate_for_percentages

end NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_percentages_l2129_212908


namespace NUMINAMATH_CALUDE_consecutive_nonprime_integers_l2129_212955

theorem consecutive_nonprime_integers : ∃ (a : ℕ),
  (25 < a) ∧
  (a + 4 < 50) ∧
  (¬ Nat.Prime a) ∧
  (¬ Nat.Prime (a + 1)) ∧
  (¬ Nat.Prime (a + 2)) ∧
  (¬ Nat.Prime (a + 3)) ∧
  (¬ Nat.Prime (a + 4)) ∧
  ((a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) % 10 = 0) ∧
  (a + 4 = 36) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_nonprime_integers_l2129_212955


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2129_212961

theorem arithmetic_sequence_sum : ∃ (n : ℕ), 
  let a := 71  -- first term
  let d := 2   -- common difference
  let l := 99  -- last term
  n = (l - a) / d + 1 ∧ 
  3 * (n * (a + l) / 2) = 3825 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2129_212961


namespace NUMINAMATH_CALUDE_smallest_number_l2129_212930

theorem smallest_number : ∀ (a b c d : ℝ), a = 0 ∧ b = -1 ∧ c = -Real.sqrt 2 ∧ d = 2 → 
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2129_212930


namespace NUMINAMATH_CALUDE_route_down_length_for_given_conditions_l2129_212935

/-- Represents a hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time : ℝ
  rate_down_factor : ℝ

/-- Calculates the length of the route down the mountain -/
def route_down_length (hike : MountainHike) : ℝ :=
  hike.rate_up * hike.rate_down_factor * hike.time

/-- Theorem stating the length of the route down the mountain for the given conditions -/
theorem route_down_length_for_given_conditions :
  let hike : MountainHike := {
    rate_up := 3,
    time := 2,
    rate_down_factor := 1.5
  }
  route_down_length hike = 9 := by sorry

end NUMINAMATH_CALUDE_route_down_length_for_given_conditions_l2129_212935


namespace NUMINAMATH_CALUDE_fair_queue_l2129_212951

def queue_problem (initial_queue : ℕ) (net_change : ℕ) (interval : ℕ) (total_time : ℕ) : Prop :=
  let intervals := total_time / interval
  let final_queue := initial_queue + intervals * net_change
  final_queue = 24

theorem fair_queue : queue_problem 12 1 5 60 := by
  sorry

end NUMINAMATH_CALUDE_fair_queue_l2129_212951


namespace NUMINAMATH_CALUDE_g_of_one_eq_neg_three_l2129_212950

-- Define the function f
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

-- State the theorem
theorem g_of_one_eq_neg_three
  (g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0)
  (h2 : g (-1) = 1) :
  g 1 = -3 :=
by sorry

end NUMINAMATH_CALUDE_g_of_one_eq_neg_three_l2129_212950


namespace NUMINAMATH_CALUDE_ellipse_chord_properties_l2129_212911

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse mx² + ny² = 1 -/
structure Ellipse where
  m : ℝ
  n : ℝ
  h_positive : m > 0 ∧ n > 0
  h_distinct : m ≠ n

/-- Theorem about properties of chords in an ellipse -/
theorem ellipse_chord_properties (e : Ellipse) (a b c d : Point) (e_mid f_mid : Point) : 
  -- AB is a chord with slope 1
  (b.y - a.y) / (b.x - a.x) = 1 →
  -- CD is perpendicular to AB
  (d.y - c.y) / (d.x - c.x) = -1 →
  -- E is midpoint of AB
  e_mid.x = (a.x + b.x) / 2 ∧ e_mid.y = (a.y + b.y) / 2 →
  -- F is midpoint of CD
  f_mid.x = (c.x + d.x) / 2 ∧ f_mid.y = (c.y + d.y) / 2 →
  -- A, B, C, D are on the ellipse
  e.m * a.x^2 + e.n * a.y^2 = 1 ∧
  e.m * b.x^2 + e.n * b.y^2 = 1 ∧
  e.m * c.x^2 + e.n * c.y^2 = 1 ∧
  e.m * d.x^2 + e.n * d.y^2 = 1 →
  -- Conclusion 1: |CD|² - |AB|² = 4|EF|²
  ((c.x - d.x)^2 + (c.y - d.y)^2) - ((a.x - b.x)^2 + (a.y - b.y)^2) = 
    4 * ((e_mid.x - f_mid.x)^2 + (e_mid.y - f_mid.y)^2) ∧
  -- Conclusion 2: A, B, C, D are concyclic
  ∃ (center : Point) (r : ℝ),
    (a.x - center.x)^2 + (a.y - center.y)^2 = r^2 ∧
    (b.x - center.x)^2 + (b.y - center.y)^2 = r^2 ∧
    (c.x - center.x)^2 + (c.y - center.y)^2 = r^2 ∧
    (d.x - center.x)^2 + (d.y - center.y)^2 = r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_properties_l2129_212911


namespace NUMINAMATH_CALUDE_locus_of_G_l2129_212941

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point F
def F : ℝ × ℝ := (2, 0)

-- Define the locus W
def W (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Theorem statement
theorem locus_of_G (x y : ℝ) :
  (∃ (h : ℝ × ℝ), unit_circle h.1 h.2 ∧
    (∃ (c : ℝ × ℝ), (c.1 - F.1)^2 + (c.2 - F.2)^2 = (c.1 - x)^2 + (c.2 - y)^2 ∧
      (c.1 - h.1)^2 + (c.2 - h.2)^2 = ((c.1 - F.1)^2 + (c.2 - F.2)^2) / 4)) →
  W x y :=
sorry

end NUMINAMATH_CALUDE_locus_of_G_l2129_212941


namespace NUMINAMATH_CALUDE_multiply_monomials_l2129_212963

theorem multiply_monomials (a : ℝ) : 3 * a^3 * (-4 * a^2) = -12 * a^5 := by sorry

end NUMINAMATH_CALUDE_multiply_monomials_l2129_212963


namespace NUMINAMATH_CALUDE_solve_equation_l2129_212960

theorem solve_equation (m : ℝ) : (m - 6) ^ 4 = (1 / 16)⁻¹ ↔ m = 8 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2129_212960


namespace NUMINAMATH_CALUDE_no_real_solutions_l2129_212999

theorem no_real_solutions :
  ∀ x : ℝ, (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2129_212999


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2129_212988

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + 8x - 6 -/
def a : ℝ := 5
def b : ℝ := 8
def c : ℝ := -6

/-- Theorem: The discriminant of 5x^2 + 8x - 6 is 184 -/
theorem quadratic_discriminant : discriminant a b c = 184 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2129_212988


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_a_equals_two_l2129_212942

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The dot product of two Vector2D -/
def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Perpendicularity of two Vector2D -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_imply_a_equals_two (a : ℝ) :
  let m : Vector2D := ⟨a, 2⟩
  let n : Vector2D := ⟨1, 1 - a⟩
  perpendicular m n → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_a_equals_two_l2129_212942


namespace NUMINAMATH_CALUDE_function_property_l2129_212978

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h2 : f (-3) = a) : 
  f 12 = -4 * a := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2129_212978


namespace NUMINAMATH_CALUDE_proposition_conditions_l2129_212964

theorem proposition_conditions (p q : Prop) : 
  (p ∨ q) → ¬(p ∧ q) → ¬p → (¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_proposition_conditions_l2129_212964


namespace NUMINAMATH_CALUDE_jovanas_shells_l2129_212949

theorem jovanas_shells (initial_shells : ℕ) : 
  initial_shells + 23 = 28 → initial_shells = 5 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l2129_212949


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l2129_212938

/-- A quadratic function f(x) = kx^2 - 7x - 7 intersects the x-axis if and only if
    k ≥ -7/4 and k ≠ 0 -/
theorem quadratic_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l2129_212938


namespace NUMINAMATH_CALUDE_centroid_coincides_with_inscribed_sphere_center_l2129_212928

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the centroids of faces opposite to vertices -/
structure FaceCentroids where
  SA : Point3D
  SB : Point3D
  SC : Point3D
  SD : Point3D

/-- Calculates the centroid of a system of homogeneous thin plates -/
def systemCentroid (t : Tetrahedron) (fc : FaceCentroids) : Point3D :=
  sorry

/-- Calculates the center of the inscribed sphere of a tetrahedron -/
def inscribedSphereCenter (t : Tetrahedron) : Point3D :=
  sorry

/-- Main theorem: The centroid of the system coincides with the center of the inscribed sphere -/
theorem centroid_coincides_with_inscribed_sphere_center 
  (t : Tetrahedron) (fc : FaceCentroids) :
  systemCentroid t fc = inscribedSphereCenter (Tetrahedron.mk fc.SA fc.SB fc.SC fc.SD) :=
by
  sorry

end NUMINAMATH_CALUDE_centroid_coincides_with_inscribed_sphere_center_l2129_212928


namespace NUMINAMATH_CALUDE_peach_pies_count_l2129_212965

/-- Given a total of 30 pies distributed among apple, blueberry, and peach flavors
    in the ratio 3:2:5, prove that the number of peach pies is 15. -/
theorem peach_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 3 →
  blueberry_ratio = 2 →
  peach_ratio = 5 →
  peach_ratio * (total_pies / (apple_ratio + blueberry_ratio + peach_ratio)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_peach_pies_count_l2129_212965


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2129_212943

/-- Represents the outcome of selecting an item -/
inductive ItemSelection
  | Qualified
  | Defective

/-- Represents a batch of products -/
structure Batch where
  qualified : ℕ
  defective : ℕ
  qualified_exceeds_two : qualified > 2
  defective_exceeds_two : defective > 2

/-- Represents the selection of two items from a batch -/
def TwoItemSelection := Prod ItemSelection ItemSelection

/-- Event: At least one defective item -/
def AtLeastOneDefective (selection : TwoItemSelection) : Prop :=
  selection.1 = ItemSelection.Defective ∨ selection.2 = ItemSelection.Defective

/-- Event: All qualified items -/
def AllQualified (selection : TwoItemSelection) : Prop :=
  selection.1 = ItemSelection.Qualified ∧ selection.2 = ItemSelection.Qualified

/-- Theorem: AtLeastOneDefective and AllQualified are mutually exclusive -/
theorem mutually_exclusive_events (batch : Batch) (selection : TwoItemSelection) :
  ¬(AtLeastOneDefective selection ∧ AllQualified selection) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2129_212943


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2129_212939

theorem quadratic_coefficient (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2129_212939


namespace NUMINAMATH_CALUDE_sin_greater_cos_range_l2129_212919

theorem sin_greater_cos_range (x : ℝ) : 
  x ∈ Set.Ioo (0 : ℝ) (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sin_greater_cos_range_l2129_212919


namespace NUMINAMATH_CALUDE_bread_slices_calculation_l2129_212945

/-- Represents the number of pieces a single slice of bread is torn into -/
def pieces_per_slice : ℕ := 4

/-- Represents the total number of bread pieces -/
def total_pieces : ℕ := 8

/-- Calculates the number of original bread slices -/
def original_slices : ℕ := total_pieces / pieces_per_slice

theorem bread_slices_calculation :
  original_slices = 2 := by sorry

end NUMINAMATH_CALUDE_bread_slices_calculation_l2129_212945


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l2129_212931

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) :
  ∃ n : ℕ, x * y + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_is_square_l2129_212931


namespace NUMINAMATH_CALUDE_engineering_majors_consecutive_probability_l2129_212936

/-- The number of people sitting at the round table -/
def total_people : ℕ := 11

/-- The number of engineering majors -/
def engineering_majors : ℕ := 5

/-- The number of ways to arrange engineering majors consecutively after fixing one position -/
def consecutive_arrangements : ℕ := 7

/-- The number of ways to choose seats for engineering majors without restriction -/
def total_arrangements : ℕ := Nat.choose (total_people - 1) (engineering_majors - 1)

/-- The probability of engineering majors sitting consecutively -/
def probability : ℚ := consecutive_arrangements / total_arrangements

theorem engineering_majors_consecutive_probability :
  probability = 1 / 30 :=
sorry

end NUMINAMATH_CALUDE_engineering_majors_consecutive_probability_l2129_212936


namespace NUMINAMATH_CALUDE_inequality_solution_part1_inequality_solution_part2_l2129_212944

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the inequality function
def inequality (x a : ℝ) : Prop := lg (|x + 3| + |x - 7|) > a

theorem inequality_solution_part1 :
  ∀ x : ℝ, inequality x 1 ↔ (x < -3 ∨ x > 7) := by sorry

theorem inequality_solution_part2 :
  ∀ a : ℝ, (∀ x : ℝ, inequality x a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_part1_inequality_solution_part2_l2129_212944


namespace NUMINAMATH_CALUDE_three_digit_palindrome_average_l2129_212932

/-- Reverses the digits of a three-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a three-digit number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n % 10 = n / 100

theorem three_digit_palindrome_average (m n : ℕ) : 
  100 ≤ m ∧ m < 1000 ∧
  100 ≤ n ∧ n < 1000 ∧
  is_palindrome m ∧
  (m + n) / 2 = reverse_digits m ∧
  m = 161 ∧ n = 161 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_palindrome_average_l2129_212932


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2129_212914

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let line := fun (x : ℝ) => (3/2) * x
  let intersection_projection_is_focus := 
    ∃ (x y : ℝ), hyperbola x y ∧ y = line x ∧ 
    (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ x = c)
  2

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2129_212914


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2129_212909

/-- Given a data set (2, 4, 6, 8) with median m and variance n, 
    and the equation ma + nb = 1 where a > 0 and b > 0,
    prove that the minimum value of 1/a + 1/b is 20. -/
theorem min_value_reciprocal_sum (m n a b : ℝ) : 
  m = 5 → 
  n = 5 → 
  m * a + n * b = 1 → 
  a > 0 → 
  b > 0 → 
  (1 / a + 1 / b) ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2129_212909


namespace NUMINAMATH_CALUDE_integer_fraction_sum_equals_three_l2129_212968

theorem integer_fraction_sum_equals_three (a b : ℕ+) :
  let A := (a + 1 : ℝ) / b + b / a
  (∃ k : ℤ, A = k) → A = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_sum_equals_three_l2129_212968


namespace NUMINAMATH_CALUDE_probability_of_prime_is_two_fifths_l2129_212915

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of numbers from 1 to 10 -/
def numberSet : Finset ℕ := sorry

/-- The set of prime numbers in the numberSet -/
def primeSet : Finset ℕ := sorry

/-- The probability of selecting a prime number from the numberSet -/
def probabilityOfPrime : ℚ := sorry

theorem probability_of_prime_is_two_fifths : 
  probabilityOfPrime = 2 / 5 := sorry

end NUMINAMATH_CALUDE_probability_of_prime_is_two_fifths_l2129_212915


namespace NUMINAMATH_CALUDE_smoothie_combinations_l2129_212900

theorem smoothie_combinations (n_smoothies : ℕ) (n_supplements : ℕ) : 
  n_smoothies = 7 → n_supplements = 8 → n_smoothies * (n_supplements.choose 3) = 392 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_combinations_l2129_212900


namespace NUMINAMATH_CALUDE_chess_game_duration_l2129_212969

theorem chess_game_duration (game_hours : ℕ) (game_minutes : ℕ) (analysis_minutes : ℕ) : 
  game_hours = 20 → game_minutes = 15 → analysis_minutes = 22 →
  game_hours * 60 + game_minutes + analysis_minutes = 1237 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_duration_l2129_212969


namespace NUMINAMATH_CALUDE_second_fraction_greater_l2129_212992

/-- Define the first fraction -/
def fraction1 : ℚ := (77 * 10^2009 + 7) / (77.77 * 10^2010)

/-- Define the second fraction -/
def fraction2 : ℚ := (33 * (10^2010 - 1) / 9) / (33 * (10^2011 - 1) / 99)

/-- Theorem stating that the second fraction is greater than the first -/
theorem second_fraction_greater : fraction2 > fraction1 := by
  sorry

end NUMINAMATH_CALUDE_second_fraction_greater_l2129_212992


namespace NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_neg_two_l2129_212980

/-- A polynomial in x and y with a parameter m -/
def polynomial (x y m : ℝ) : ℝ := 8 * x^2 + (m + 1) * x * y - 5 * y + x * y - 8

theorem no_xy_term_implies_m_eq_neg_two (m : ℝ) :
  (∀ x y : ℝ, polynomial x y m = 8 * x^2 - 5 * y - 8) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_neg_two_l2129_212980


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2129_212924

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ∈ Set.Ico 1 2 → (x^2 - a ≤ 0 → a > 4)) ∧
  ¬(∀ x : ℝ, x ∈ Set.Ico 1 2 → (a > 4 → x^2 - a ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2129_212924


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l2129_212983

theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ x₀ : ℝ, (2 * x₀ - a = Real.log (2 * x₀ + b)) ∧ 
    (∀ x : ℝ, 2 * x - a ≤ Real.log (2 * x + b))) :
  (4 / a + 1 / b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l2129_212983


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2129_212962

theorem gcd_of_specific_numbers :
  let m : ℕ := 33333333
  let n : ℕ := 666666666
  Nat.gcd m n = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2129_212962


namespace NUMINAMATH_CALUDE_problem_solution_l2129_212901

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0

-- Theorem statement
theorem problem_solution : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2129_212901


namespace NUMINAMATH_CALUDE_point_relationship_l2129_212929

/-- Given points A(-2,a), B(-1,b), C(3,c) on the graph of y = 4/x, prove that b < a < c -/
theorem point_relationship (a b c : ℝ) : 
  (a = 4 / (-2)) → (b = 4 / (-1)) → (c = 4 / 3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_point_relationship_l2129_212929


namespace NUMINAMATH_CALUDE_picnic_attendance_l2129_212920

theorem picnic_attendance (total_students : ℕ) (picnic_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1500 →
  picnic_attendees = 975 →
  total_students = girls + boys →
  picnic_attendees = (3 * girls / 4) + (3 * boys / 5) →
  (3 * girls / 4 : ℕ) = 375 :=
by sorry

end NUMINAMATH_CALUDE_picnic_attendance_l2129_212920


namespace NUMINAMATH_CALUDE_equivalence_condition_l2129_212953

/-- Hyperbola C with equation x² - y²/3 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Left vertex of the hyperbola -/
def A₁ : ℝ × ℝ := (-1, 0)

/-- Right vertex of the hyperbola -/
def A₂ : ℝ × ℝ := (1, 0)

/-- Moving line l with equation x = my + n -/
def line_l (m n y : ℝ) : ℝ := m * y + n

/-- Intersection point T of A₁M and A₂N -/
structure Point_T (m n : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  on_A₁M : ∃ (x₁ y₁ : ℝ), hyperbola_C x₁ y₁ ∧ y₀ = (y₁ / (x₁ + 1)) * (x₀ + 1)
  on_A₂N : ∃ (x₂ y₂ : ℝ), hyperbola_C x₂ y₂ ∧ y₀ = (y₂ / (x₂ - 1)) * (x₀ - 1)
  on_line_l : x₀ = line_l m n y₀

/-- The main theorem to prove -/
theorem equivalence_condition (m : ℝ) :
  ∀ (n : ℝ), (∃ (T : Point_T m n), n = 2 ↔ T.x₀ = 1/2) := by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l2129_212953


namespace NUMINAMATH_CALUDE_max_value_constraint_l2129_212985

/-- Given a point (3,1) lying on the line mx + ny + 1 = 0 where mn > 0, 
    the maximum value of 3/m + 1/n is -16. -/
theorem max_value_constraint (m n : ℝ) : 
  m * n > 0 → 
  3 * m + n = -1 → 
  (3 / m + 1 / n) ≤ -16 ∧ 
  ∃ m₀ n₀ : ℝ, m₀ * n₀ > 0 ∧ 3 * m₀ + n₀ = -1 ∧ 3 / m₀ + 1 / n₀ = -16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2129_212985


namespace NUMINAMATH_CALUDE_blueberry_zucchini_trade_l2129_212905

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_traded : ℕ := 6

/-- The number of zucchinis received in trade for containers_traded -/
def zucchinis_received : ℕ := 3

/-- The total number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 12

theorem blueberry_zucchini_trade :
  bushes_needed * containers_per_bush * zucchinis_received = 
  target_zucchinis * containers_traded := by
  sorry

end NUMINAMATH_CALUDE_blueberry_zucchini_trade_l2129_212905


namespace NUMINAMATH_CALUDE_total_laces_is_6x_l2129_212918

/-- Given a number of shoe pairs, calculate the total number of laces needed -/
def total_laces (x : ℕ) : ℕ :=
  let lace_sets_per_pair := 2
  let color_options := 3
  x * lace_sets_per_pair * color_options

/-- Theorem stating that the total number of laces is 6x -/
theorem total_laces_is_6x (x : ℕ) : total_laces x = 6 * x := by
  sorry

end NUMINAMATH_CALUDE_total_laces_is_6x_l2129_212918


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_for_solutions_l2129_212954

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + a|

-- Theorem for part I
theorem solution_set_for_a_equals_2 :
  {x : ℝ | f x 2 > 6} = {x : ℝ | x < -3 ∨ x > 3} := by sorry

-- Theorem for part II
theorem range_of_a_for_solutions :
  {a : ℝ | ∃ x, f x a < a^2 - 1} = {a : ℝ | a < -1 - Real.sqrt 2 ∨ a > 1 + Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_for_solutions_l2129_212954


namespace NUMINAMATH_CALUDE_digit2015_is_8_l2129_212902

/-- The function that generates the list of digits of positive even numbers -/
def evenNumberDigits : ℕ → List ℕ := sorry

/-- The 2015th digit in the list of digits of positive even numbers -/
def digit2015 : ℕ := (evenNumberDigits 0).nthLe 2014 sorry

/-- Theorem stating that the 2015th digit is 8 -/
theorem digit2015_is_8 : digit2015 = 8 := by sorry

end NUMINAMATH_CALUDE_digit2015_is_8_l2129_212902


namespace NUMINAMATH_CALUDE_least_sum_with_constraint_l2129_212982

theorem least_sum_with_constraint (x y z : ℕ+) : 
  (∀ a b c : ℕ+, x + y + z ≤ a + b + c) → 
  (x + y + z = 37) → 
  (5 * y = 6 * z) → 
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_constraint_l2129_212982


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2129_212977

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3*x - 2 ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2129_212977


namespace NUMINAMATH_CALUDE_expand_product_l2129_212998

theorem expand_product (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2129_212998


namespace NUMINAMATH_CALUDE_expression_value_l2129_212940

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 3) 
  (eq2 : x * w + y * z = 6) : 
  (2 * x + y) * (2 * z + w) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2129_212940


namespace NUMINAMATH_CALUDE_function_characterization_l2129_212916

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ m n : ℝ, f (m + n) = f m + f n - 6) ∧
  (∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ f (-1) = k) ∧
  (∀ x : ℝ, x > -1 → f x > 0)

theorem function_characterization (f : ℝ → ℝ) (h : is_valid_function f) :
  ∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ ∀ x : ℝ, f x = k * x + 6 :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2129_212916


namespace NUMINAMATH_CALUDE_expression_evaluation_l2129_212984

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 5 + 1/2
  ((x^2 / (x - 1) - x + 1) / ((4*x^2 - 4*x + 1) / (1 - x))) = -Real.sqrt 5 / 10 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2129_212984


namespace NUMINAMATH_CALUDE_equal_probability_sums_l2129_212959

def num_dice : ℕ := 8
def min_face_value : ℕ := 1
def max_face_value : ℕ := 6

def min_sum : ℕ := num_dice * min_face_value
def max_sum : ℕ := num_dice * max_face_value

def symmetric_sum (s : ℕ) : ℕ := 2 * ((min_sum + max_sum) / 2) - s

theorem equal_probability_sums :
  symmetric_sum 11 = 45 :=
sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l2129_212959


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2129_212994

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 6 = 0) ∧ (x₂^2 - 2*x₂ - 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2129_212994


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l2129_212976

/-- The time it takes for pipe A to fill the cistern -/
def fill_time_A : ℝ := 10

/-- The time it takes for pipe B to empty the cistern -/
def empty_time_B : ℝ := 12

/-- The time it takes to fill the cistern with both pipes open -/
def fill_time_both : ℝ := 60

/-- Theorem stating that the fill time for pipe A is correct -/
theorem pipe_A_fill_time :
  fill_time_A = 10 ∧
  (1 / fill_time_A - 1 / empty_time_B = 1 / fill_time_both) :=
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l2129_212976


namespace NUMINAMATH_CALUDE_work_day_ends_at_430pm_l2129_212948

-- Define the structure for time
structure Time where
  hours : Nat
  minutes : Nat

-- Define the work schedule
def workStartTime : Time := { hours := 8, minutes := 0 }
def lunchStartTime : Time := { hours := 13, minutes := 0 }
def lunchDuration : Nat := 30
def totalWorkHours : Nat := 8

-- Function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Function to calculate time difference in hours
def timeDifferenceInHours (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes - (t1.hours * 60 + t1.minutes)) / 60

-- Theorem stating that Maria's work day ends at 4:30 P.M.
theorem work_day_ends_at_430pm :
  let lunchEndTime := addMinutes lunchStartTime lunchDuration
  let workBeforeLunch := timeDifferenceInHours workStartTime lunchStartTime
  let remainingWorkHours := totalWorkHours - workBeforeLunch
  let endTime := addMinutes lunchEndTime (remainingWorkHours * 60)
  endTime = { hours := 16, minutes := 30 } :=
by sorry

end NUMINAMATH_CALUDE_work_day_ends_at_430pm_l2129_212948


namespace NUMINAMATH_CALUDE_height_percentage_difference_l2129_212971

theorem height_percentage_difference (P Q : ℝ) (h : Q = P * (1 + 66.67 / 100)) :
  P = Q * (1 - 40 / 100) :=
sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l2129_212971


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l2129_212925

/-- Given a man's rowing speed against the stream and his speed in still water,
    calculate his speed with the stream. -/
theorem mans_speed_with_stream
  (speed_against_stream : ℝ)
  (speed_still_water : ℝ)
  (h1 : speed_against_stream = 4)
  (h2 : speed_still_water = 5) :
  speed_still_water + (speed_still_water - speed_against_stream) = 6 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l2129_212925


namespace NUMINAMATH_CALUDE_haley_sunday_tv_hours_l2129_212917

/-- Represents the number of hours Haley watched TV -/
structure TVWatchingHours where
  saturday : ℕ
  total : ℕ

/-- Calculates the number of hours Haley watched TV on Sunday -/
def sunday_hours (h : TVWatchingHours) : ℕ :=
  h.total - h.saturday

/-- Theorem stating that Haley watched TV for 3 hours on Sunday -/
theorem haley_sunday_tv_hours :
  ∀ h : TVWatchingHours, h.saturday = 6 → h.total = 9 → sunday_hours h = 3 := by
  sorry

end NUMINAMATH_CALUDE_haley_sunday_tv_hours_l2129_212917


namespace NUMINAMATH_CALUDE_sequence_product_l2129_212972

/-- An arithmetic sequence where no term is zero -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m ∧ a n ≠ 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_product (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 4 - 2 * (a 7)^2 + 3 * a 8 = 0 →
  b 7 = a 7 →
  b 3 * b 7 * b 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l2129_212972


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2129_212974

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 1000)
  (h2 : rate = 10)
  (h3 : time = 4)
  : interest = (2500 * rate * time) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2129_212974


namespace NUMINAMATH_CALUDE_quarter_circles_sum_approaches_diameter_l2129_212921

/-- The sum of the lengths of the arcs of quarter circles approaches the diameter as n approaches infinity -/
theorem quarter_circles_sum_approaches_diameter (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * D / (4 * n)) - D| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_approaches_diameter_l2129_212921


namespace NUMINAMATH_CALUDE_no_real_solutions_l2129_212922

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 3) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2129_212922
