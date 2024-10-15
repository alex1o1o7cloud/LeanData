import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_power_of_two_l3040_304070

theorem unique_solution_power_of_two (a b m : ℕ) : 
  a > 0 → b > 0 → (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_of_two_l3040_304070


namespace NUMINAMATH_CALUDE_min_shift_sinusoidal_graphs_l3040_304040

open Real

theorem min_shift_sinusoidal_graphs : 
  let f (x : ℝ) := 2 * sin (x + π/6)
  let g (x : ℝ) := 2 * sin (x - π/3)
  ∃ φ : ℝ, φ > 0 ∧ (∀ x : ℝ, f (x - φ) = g x) ∧
    (∀ ψ : ℝ, ψ > 0 ∧ (∀ x : ℝ, f (x - ψ) = g x) → φ ≤ ψ) ∧
    φ = π/2 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_sinusoidal_graphs_l3040_304040


namespace NUMINAMATH_CALUDE_yogurt_topping_combinations_l3040_304032

/-- The number of yogurt flavors --/
def yogurt_flavors : ℕ := 6

/-- The number of available toppings --/
def toppings : ℕ := 8

/-- The number of toppings to choose --/
def choose_toppings : ℕ := 2

/-- Theorem stating the number of unique combinations --/
theorem yogurt_topping_combinations : 
  yogurt_flavors * Nat.choose toppings choose_toppings = 168 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_topping_combinations_l3040_304032


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l3040_304031

theorem least_possible_smallest_integer
  (a b c d : ℤ)
  (different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (average : (a + b + c + d) / 4 = 74)
  (largest : d = 90)
  (ordered : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  a ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l3040_304031


namespace NUMINAMATH_CALUDE_next_square_number_proof_l3040_304092

/-- The next square number after 4356 composed of four consecutive digits -/
def next_square_number : ℕ := 5476

/-- The square root of the next square number -/
def square_root : ℕ := 74

/-- Predicate to check if a number is composed of four consecutive digits -/
def is_composed_of_four_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a : ℕ), a > 0 ∧ a < 7 ∧
  (n = a * 1000 + (a + 1) * 100 + (a + 2) * 10 + (a + 3) ∨
   n = a * 1000 + (a + 1) * 100 + (a + 3) * 10 + (a + 2) ∨
   n = a * 1000 + (a + 2) * 100 + (a + 1) * 10 + (a + 3) ∨
   n = a * 1000 + (a + 2) * 100 + (a + 3) * 10 + (a + 1) ∨
   n = a * 1000 + (a + 3) * 100 + (a + 1) * 10 + (a + 2) ∨
   n = a * 1000 + (a + 3) * 100 + (a + 2) * 10 + (a + 1))

theorem next_square_number_proof :
  next_square_number = square_root ^ 2 ∧
  is_composed_of_four_consecutive_digits next_square_number ∧
  ∀ (n : ℕ), 4356 < n ∧ n < next_square_number →
    ¬(∃ (m : ℕ), n = m ^ 2 ∧ is_composed_of_four_consecutive_digits n) :=
by sorry

end NUMINAMATH_CALUDE_next_square_number_proof_l3040_304092


namespace NUMINAMATH_CALUDE_cubic_function_property_l3040_304020

/-- Given a cubic function f(x) = ax³ + bx + 8, if f(-2) = 10, then f(2) = 6 -/
theorem cubic_function_property (a b : ℝ) : 
  let f := λ x : ℝ => a * x^3 + b * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3040_304020


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l3040_304087

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (diff_eq : x - y = 10) : 
  x^2 - y^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l3040_304087


namespace NUMINAMATH_CALUDE_inverse_proportion_k_negative_l3040_304071

/-- Given two points A(-2, y₁) and B(5, y₂) on the graph of y = k/x (k ≠ 0),
    if y₁ > y₂, then k < 0. -/
theorem inverse_proportion_k_negative
  (k : ℝ) (y₁ y₂ : ℝ)
  (hk : k ≠ 0)
  (hA : y₁ = k / (-2))
  (hB : y₂ = k / 5)
  (hy : y₁ > y₂) :
  k < 0 :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_negative_l3040_304071


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3040_304034

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.474747... -/
def x : ℚ := RepeatingDecimal 4 7

theorem repeating_decimal_sum : x = 47 / 99 ∧ 47 + 99 = 146 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3040_304034


namespace NUMINAMATH_CALUDE_election_results_l3040_304025

/-- Election results theorem -/
theorem election_results 
  (vote_percentage_A : ℝ) 
  (vote_percentage_B : ℝ) 
  (vote_percentage_C : ℝ) 
  (vote_percentage_D : ℝ) 
  (majority_difference : ℕ) 
  (h1 : vote_percentage_A = 0.45) 
  (h2 : vote_percentage_B = 0.30) 
  (h3 : vote_percentage_C = 0.20) 
  (h4 : vote_percentage_D = 0.05) 
  (h5 : vote_percentage_A + vote_percentage_B + vote_percentage_C + vote_percentage_D = 1) 
  (h6 : majority_difference = 1620) : 
  ∃ (total_votes : ℕ), 
    total_votes = 10800 ∧ 
    (vote_percentage_A * total_votes : ℝ) = 4860 ∧ 
    (vote_percentage_B * total_votes : ℝ) = 3240 ∧ 
    (vote_percentage_C * total_votes : ℝ) = 2160 ∧ 
    (vote_percentage_D * total_votes : ℝ) = 540 ∧ 
    (vote_percentage_A * total_votes - vote_percentage_B * total_votes : ℝ) = majority_difference :=
by sorry


end NUMINAMATH_CALUDE_election_results_l3040_304025


namespace NUMINAMATH_CALUDE_quadratic_diophantine_equation_solution_l3040_304056

theorem quadratic_diophantine_equation_solution 
  (a b c : ℕ+) 
  (h : (a * c : ℕ) = b^2 + b + 1) : 
  ∃ (x y : ℤ), (a : ℤ) * x^2 - (2 * (b : ℤ) + 1) * x * y + (c : ℤ) * y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_diophantine_equation_solution_l3040_304056


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_equation_l3040_304001

theorem no_rational_solutions_for_equation :
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2*y^5 + 5*z^5 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_equation_l3040_304001


namespace NUMINAMATH_CALUDE_cosine_equality_l3040_304049

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (980 * π / 180) → n = 100 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l3040_304049


namespace NUMINAMATH_CALUDE_perception_arrangements_l3040_304050

/-- The number of distinct arrangements of letters in a word with specific letter frequencies -/
def word_arrangements (total : ℕ) (double_count : ℕ) (single_count : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial 2 ^ double_count)

/-- Theorem stating the number of arrangements for the given word structure -/
theorem perception_arrangements :
  word_arrangements 10 3 4 = 453600 := by
  sorry

#eval word_arrangements 10 3 4

end NUMINAMATH_CALUDE_perception_arrangements_l3040_304050


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3040_304029

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3040_304029


namespace NUMINAMATH_CALUDE_modular_congruence_l3040_304088

theorem modular_congruence (x : ℤ) : 
  (5 * x + 9) % 18 = 4 → (3 * x + 15) % 18 = 12 := by sorry

end NUMINAMATH_CALUDE_modular_congruence_l3040_304088


namespace NUMINAMATH_CALUDE_unique_intersection_l3040_304045

/-- The value of a for which the graphs of y = ax² + 5x + 2 and y = -2x - 2 intersect at exactly one point -/
def intersection_value : ℚ := 49 / 16

/-- The first graph equation -/
def graph1 (a x : ℚ) : ℚ := a * x^2 + 5 * x + 2

/-- The second graph equation -/
def graph2 (x : ℚ) : ℚ := -2 * x - 2

/-- Theorem stating that the graphs intersect at exactly one point when a = 49/16 -/
theorem unique_intersection :
  ∃! x : ℚ, graph1 intersection_value x = graph2 x :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l3040_304045


namespace NUMINAMATH_CALUDE_count_integers_with_2_and_3_l3040_304067

def count_integers_with_digits (lower_bound upper_bound : ℕ) (digit1 digit2 : ℕ) : ℕ :=
  sorry

theorem count_integers_with_2_and_3 :
  count_integers_with_digits 1000 2000 2 3 = 108 :=
sorry

end NUMINAMATH_CALUDE_count_integers_with_2_and_3_l3040_304067


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3040_304036

/-- The distance between the foci of a hyperbola defined by x^2 - y^2 = 1 is 2√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x^2 - y^2 = 1 → (x - f₁.1)^2 + (y - f₁.2)^2 = (x - f₂.1)^2 + (y - f₂.2)^2) ∧
    dist f₁ f₂ = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3040_304036


namespace NUMINAMATH_CALUDE_production_equation_l3040_304084

/-- Represents the equation for a production scenario where increasing the daily rate by 20%
    completes the task 4 days earlier. -/
theorem production_equation (x : ℝ) (h : x > 0) :
  (3000 : ℝ) / x = 4 + (3000 : ℝ) / (x * (1 + 20 / 100)) :=
sorry

end NUMINAMATH_CALUDE_production_equation_l3040_304084


namespace NUMINAMATH_CALUDE_first_digit_389_base4_is_1_l3040_304058

-- Define a function to convert a number to its base-4 representation
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

-- Theorem statement
theorem first_digit_389_base4_is_1 :
  (toBase4 389).reverse.head? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_389_base4_is_1_l3040_304058


namespace NUMINAMATH_CALUDE_smallest_multiple_sixty_four_satisfies_smallest_satisfying_number_l3040_304078

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 450 * x % 640 = 0 → x ≥ 64 := by
  sorry

theorem sixty_four_satisfies : 450 * 64 % 640 = 0 := by
  sorry

theorem smallest_satisfying_number : ∃ x : ℕ, x > 0 ∧ 450 * x % 640 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ 450 * y % 640 = 0) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_sixty_four_satisfies_smallest_satisfying_number_l3040_304078


namespace NUMINAMATH_CALUDE_geometry_problem_l3040_304037

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line_AB (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the circle equation
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 25/2

theorem geometry_problem :
  -- The equation of line AB
  (∀ x y : ℝ, (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ↔ line_AB x y) ∧
  -- The circle with center C is tangent to line AB
  (∃ x y : ℝ, line_AB x y ∧ circle_C x y ∧
    ∀ x' y' : ℝ, line_AB x' y' → ((x' - C.1)^2 + (y' - C.2)^2 ≥ 25/2)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_problem_l3040_304037


namespace NUMINAMATH_CALUDE_negation_equivalence_l3040_304047

theorem negation_equivalence :
  (¬ ∃ x : ℤ, x^2 + 2*x + 1 ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3040_304047


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3040_304060

theorem cubic_root_sum (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3040_304060


namespace NUMINAMATH_CALUDE_ten_men_and_boys_complete_in_ten_days_l3040_304096

/-- The number of days it takes for a group of men and boys to complete a work -/
def daysToComplete (numMen numBoys : ℕ) : ℚ :=
  10 / ((2 * numMen : ℚ) / 3 + (numBoys : ℚ) / 3)

/-- Theorem stating that 10 men and 10 boys will complete the work in 10 days -/
theorem ten_men_and_boys_complete_in_ten_days :
  daysToComplete 10 10 = 10 := by sorry

end NUMINAMATH_CALUDE_ten_men_and_boys_complete_in_ten_days_l3040_304096


namespace NUMINAMATH_CALUDE_fraction_relation_l3040_304010

theorem fraction_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l3040_304010


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l3040_304018

theorem discount_percentage_calculation (washing_machine_cost dryer_cost total_paid : ℚ) : 
  washing_machine_cost = 100 →
  dryer_cost = washing_machine_cost - 30 →
  total_paid = 153 →
  (washing_machine_cost + dryer_cost - total_paid) / (washing_machine_cost + dryer_cost) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l3040_304018


namespace NUMINAMATH_CALUDE_worker_hours_per_day_l3040_304016

/-- Represents a factory worker's productivity and work schedule -/
structure Worker where
  widgets_per_hour : ℕ
  days_per_week : ℕ
  widgets_per_week : ℕ

/-- Calculates the number of hours a worker works per day -/
def hours_per_day (w : Worker) : ℚ :=
  (w.widgets_per_week : ℚ) / (w.widgets_per_hour : ℚ) / (w.days_per_week : ℚ)

/-- Theorem stating that a worker with given productivity and output works 8 hours per day -/
theorem worker_hours_per_day (w : Worker)
    (h1 : w.widgets_per_hour = 20)
    (h2 : w.days_per_week = 5)
    (h3 : w.widgets_per_week = 800) :
    hours_per_day w = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_hours_per_day_l3040_304016


namespace NUMINAMATH_CALUDE_batsman_average_is_60_l3040_304090

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  highest_score : ℕ
  lowest_score : ℕ
  average_excluding_extremes : ℚ

/-- Calculate the batting average given the batsman's statistics -/
def batting_average (stats : BatsmanStats) : ℚ :=
  let total_runs := (stats.total_innings - 2) * stats.average_excluding_extremes + stats.highest_score + stats.lowest_score
  total_runs / stats.total_innings

theorem batsman_average_is_60 (stats : BatsmanStats) :
  stats.total_innings = 46 ∧
  stats.highest_score - stats.lowest_score = 140 ∧
  stats.average_excluding_extremes = 58 ∧
  stats.highest_score = 174 →
  batting_average stats = 60 := by
  sorry

#eval batting_average {
  total_innings := 46,
  highest_score := 174,
  lowest_score := 34,
  average_excluding_extremes := 58
}

end NUMINAMATH_CALUDE_batsman_average_is_60_l3040_304090


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3040_304044

theorem no_infinite_prime_sequence :
  ¬∃ (p : ℕ → ℕ), (∀ k, p (k + 1) = 5 * p k + 4) ∧ (∀ k, Nat.Prime (p k)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3040_304044


namespace NUMINAMATH_CALUDE_sequence_sum_property_l3040_304055

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem sequence_sum_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → (sequence_sum a n - 1)^2 = a n * sequence_sum a n) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l3040_304055


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3040_304073

/-- A parabola passing through two points with the same y-coordinate -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  n : ℝ

/-- The x-coordinate of the axis of symmetry of a parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := 2

/-- Theorem: The axis of symmetry of a parabola passing through (1,n) and (3,n) is x = 2 -/
theorem parabola_axis_of_symmetry (p : Parabola) : 
  p.n = p.a * 1^2 + p.b * 1 + p.c ∧ 
  p.n = p.a * 3^2 + p.b * 3 + p.c → 
  axisOfSymmetry p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3040_304073


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3040_304075

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: f is defined on {x ∈ ℝ | x ≠ 0}
  (∀ x : ℝ, x ≠ 0 → f x = Real.log (abs x)) ∧
  -- Condition 2: f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- Condition 3: f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- Condition 4: For any non-zero real numbers x and y, f(xy) = f(x) + f(y)
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → f (x * y) = f x + f y) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3040_304075


namespace NUMINAMATH_CALUDE_ludwig_daily_salary_l3040_304011

def weekly_salary : ℚ := 55
def full_days : ℕ := 4
def half_days : ℕ := 3

theorem ludwig_daily_salary : 
  ∃ (daily_salary : ℚ), 
    (daily_salary * full_days + daily_salary * half_days / 2 = weekly_salary) ∧
    daily_salary = 10 := by
sorry

end NUMINAMATH_CALUDE_ludwig_daily_salary_l3040_304011


namespace NUMINAMATH_CALUDE_counterexample_exists_l3040_304013

theorem counterexample_exists : ∃ a : ℝ, a^2 > 0 ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3040_304013


namespace NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l3040_304064

/-- Given two similar quadrilaterals with sides (a, b, c, d) and (a', b', c', d') respectively,
    and similarity ratio k, prove that the areas of rectangles formed by opposite sides
    are proportional to k^2 -/
theorem similar_quadrilaterals_rectangle_areas
  (a b c d a' b' c' d' k : ℝ)
  (h_similar : a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' ∧ d / d' = k) :
  a * c / (a' * c') = k^2 ∧ b * d / (b' * d') = k^2 := by
  sorry

end NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l3040_304064


namespace NUMINAMATH_CALUDE_catalog_arrangements_l3040_304065

theorem catalog_arrangements : 
  let n : ℕ := 7  -- number of letters in "catalog"
  Nat.factorial n = 5040 := by sorry

end NUMINAMATH_CALUDE_catalog_arrangements_l3040_304065


namespace NUMINAMATH_CALUDE_show_revenue_l3040_304028

/-- Calculates the total revenue for two shows given the attendance of the first show,
    the multiplier for the second show's attendance, and the ticket price. -/
def totalRevenue (firstShowAttendance : ℕ) (secondShowMultiplier : ℕ) (ticketPrice : ℕ) : ℕ :=
  (firstShowAttendance + secondShowMultiplier * firstShowAttendance) * ticketPrice

/-- Theorem stating that the total revenue for both shows is $20,000 -/
theorem show_revenue : totalRevenue 200 3 25 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_show_revenue_l3040_304028


namespace NUMINAMATH_CALUDE_sand_heap_radius_l3040_304026

/-- Given a cylindrical bucket of sand and a conical heap formed from it, 
    prove that the radius of the heap's base is 63 cm. -/
theorem sand_heap_radius : 
  ∀ (h_cylinder r_cylinder h_cone r_cone : ℝ),
  h_cylinder = 36 ∧ 
  r_cylinder = 21 ∧ 
  h_cone = 12 ∧
  π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone →
  r_cone = 63 := by
  sorry

end NUMINAMATH_CALUDE_sand_heap_radius_l3040_304026


namespace NUMINAMATH_CALUDE_only_one_is_ultra_prime_l3040_304082

-- Define f(n) as the sum of all divisors of n
def f (n : ℕ) : ℕ := sorry

-- Define g(n) = n + f(n)
def g (n : ℕ) : ℕ := n + f n

-- Define ultra-prime
def is_ultra_prime (n : ℕ) : Prop := f (g n) = 2 * n + 3

-- Theorem statement
theorem only_one_is_ultra_prime :
  ∃! (n : ℕ), n < 100 ∧ is_ultra_prime n :=
sorry

end NUMINAMATH_CALUDE_only_one_is_ultra_prime_l3040_304082


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3040_304095

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the second term of the sequence is 3. -/
theorem second_term_of_geometric_series (a : ℝ) :
  (∑' n, a * (1/4)^n : ℝ) = 16 →
  a * (1/4) = 3 := by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3040_304095


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3040_304053

theorem complex_fraction_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (2 + i) / (1 + i) = a + b * i → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3040_304053


namespace NUMINAMATH_CALUDE_initial_money_equals_spent_plus_left_l3040_304043

/-- The amount of money Trisha spent on meat -/
def meat_cost : ℕ := 17

/-- The amount of money Trisha spent on chicken -/
def chicken_cost : ℕ := 22

/-- The amount of money Trisha spent on veggies -/
def veggies_cost : ℕ := 43

/-- The amount of money Trisha spent on eggs -/
def eggs_cost : ℕ := 5

/-- The amount of money Trisha spent on dog food -/
def dog_food_cost : ℕ := 45

/-- The amount of money Trisha had left after shopping -/
def money_left : ℕ := 35

/-- The total amount of money Trisha spent -/
def total_spent : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost

/-- The theorem stating that the initial amount of money Trisha had
    is equal to the sum of all her expenses plus the amount left after shopping -/
theorem initial_money_equals_spent_plus_left :
  total_spent + money_left = 167 := by sorry

end NUMINAMATH_CALUDE_initial_money_equals_spent_plus_left_l3040_304043


namespace NUMINAMATH_CALUDE_specialPrimes_eq_l3040_304063

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if a number has all digits 0 to b-1 exactly once in base b -/
def hasAllDigitsOnce (n : ℕ) (b : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    (digits.length = b) ∧
    (∀ d, d ∈ digits → d < b) ∧
    (digits.toFinset = Finset.range b) ∧
    (n = digits.foldr (λ d acc => acc * b + d) 0)

/-- The set of prime numbers with the special digit property -/
def specialPrimes : Set ℕ :=
  {p | ∃ b : ℕ, isPrime p ∧ b > 1 ∧ hasAllDigitsOnce p b}

/-- The theorem stating that the set of special primes is equal to {2, 5, 7, 11, 19} -/
theorem specialPrimes_eq : specialPrimes = {2, 5, 7, 11, 19} := by sorry

end NUMINAMATH_CALUDE_specialPrimes_eq_l3040_304063


namespace NUMINAMATH_CALUDE_modulo_13_residue_l3040_304030

theorem modulo_13_residue : (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_modulo_13_residue_l3040_304030


namespace NUMINAMATH_CALUDE_festival_fruit_prices_l3040_304059

/-- Proves that given the conditions from the problem, the cost per kg of oranges is 2.2 yuan and the cost per kg of bananas is 5.4 yuan -/
theorem festival_fruit_prices :
  let orange_price : ℚ := x
  let pear_price : ℚ := x
  let apple_price : ℚ := y
  let banana_price : ℚ := y
  ∀ x y : ℚ,
  (9 * x + 10 * y = 73.8) →
  (17 * x + 6 * y = 69.8) →
  (x = 2.2 ∧ y = 5.4) :=
by
  sorry

end NUMINAMATH_CALUDE_festival_fruit_prices_l3040_304059


namespace NUMINAMATH_CALUDE_smallest_coin_count_l3040_304027

def is_valid_coin_combination (dimes quarters : ℕ) : Prop :=
  dimes * 10 + quarters * 25 = 265 ∧ dimes > quarters

def coin_count (dimes quarters : ℕ) : ℕ :=
  dimes + quarters

theorem smallest_coin_count : 
  (∃ d q : ℕ, is_valid_coin_combination d q) ∧ 
  (∀ d q : ℕ, is_valid_coin_combination d q → coin_count d q ≥ 16) ∧
  (∃ d q : ℕ, is_valid_coin_combination d q ∧ coin_count d q = 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_coin_count_l3040_304027


namespace NUMINAMATH_CALUDE_total_scoops_needed_l3040_304099

/-- Calculates the total number of scoops needed for baking ingredients --/
theorem total_scoops_needed
  (flour_cups : ℚ)
  (sugar_cups : ℚ)
  (milk_cups : ℚ)
  (flour_scoop : ℚ)
  (sugar_scoop : ℚ)
  (milk_scoop : ℚ)
  (h_flour : flour_cups = 4)
  (h_sugar : sugar_cups = 3)
  (h_milk : milk_cups = 2)
  (h_flour_scoop : flour_scoop = 1/4)
  (h_sugar_scoop : sugar_scoop = 1/3)
  (h_milk_scoop : milk_scoop = 1/2) :
  ⌈flour_cups / flour_scoop⌉ + ⌈sugar_cups / sugar_scoop⌉ + ⌈milk_cups / milk_scoop⌉ = 29 :=
by sorry

end NUMINAMATH_CALUDE_total_scoops_needed_l3040_304099


namespace NUMINAMATH_CALUDE_function_coefficient_sum_l3040_304012

theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_function_coefficient_sum_l3040_304012


namespace NUMINAMATH_CALUDE_gcf_of_450_and_144_l3040_304062

theorem gcf_of_450_and_144 : Nat.gcd 450 144 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_450_and_144_l3040_304062


namespace NUMINAMATH_CALUDE_number_calculation_l3040_304052

theorem number_calculation (x : Float) (h : x = 0.08999999999999998) :
  let number := x * 0.1
  number = 0.008999999999999999 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3040_304052


namespace NUMINAMATH_CALUDE_coin_probability_l3040_304005

theorem coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) : 
  (Nat.choose 6 3 : ℝ) * p^3 * (1-p)^3 = 1/20 → p = 1/400 := by
  sorry

end NUMINAMATH_CALUDE_coin_probability_l3040_304005


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3040_304022

theorem imaginary_part_of_z (z : ℂ) (h : z / (2 - I) = I) : z.im = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3040_304022


namespace NUMINAMATH_CALUDE_f_bound_l3040_304072

/-- The function f(x) = (e^x - 1) / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem f_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, 0 < |x| ∧ |x| < Real.log (1 + a) → |f x - 1| < a :=
sorry

end NUMINAMATH_CALUDE_f_bound_l3040_304072


namespace NUMINAMATH_CALUDE_cost_per_dozen_is_240_l3040_304061

/-- Calculates the cost per dozen donuts given the total number of donuts,
    selling price per donut, desired profit, and total number of dozens. -/
def cost_per_dozen (total_donuts : ℕ) (price_per_donut : ℚ) (desired_profit : ℚ) (total_dozens : ℕ) : ℚ :=
  let total_sales := total_donuts * price_per_donut
  let total_cost := total_sales - desired_profit
  total_cost / total_dozens

/-- Proves that the cost per dozen donuts is $2.40 given the specified conditions. -/
theorem cost_per_dozen_is_240 :
  cost_per_dozen 120 1 96 10 = 240 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_dozen_is_240_l3040_304061


namespace NUMINAMATH_CALUDE_eugene_pencils_eugene_final_pencils_l3040_304017

theorem eugene_pencils (initial_pencils : ℕ) (received_pencils : ℕ) 
  (pack_size : ℕ) (num_friends : ℕ) (given_away : ℕ) : ℕ :=
  let total_after_receiving := initial_pencils + received_pencils
  let total_in_packs := pack_size * (num_friends + 1)
  let total_before_giving := total_after_receiving + total_in_packs
  total_before_giving - given_away

theorem eugene_final_pencils :
  eugene_pencils 51 6 12 3 8 = 97 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_eugene_final_pencils_l3040_304017


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l3040_304086

theorem batsman_running_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : 
  total_runs = 120 →
  boundaries = 6 →
  sixes = 4 →
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l3040_304086


namespace NUMINAMATH_CALUDE_line_division_theorem_l3040_304076

/-- A line in the plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three lines divide the plane into six parts -/
def divides_into_six_parts (l₁ l₂ l₃ : Line) : Prop :=
  sorry

/-- The set of k values that satisfy the condition -/
def k_values : Set ℝ :=
  {0, -1, -2}

theorem line_division_theorem (k : ℝ) :
  let l₁ : Line := ⟨1, -2, 1⟩  -- x - 2y + 1 = 0
  let l₂ : Line := ⟨1, 0, -1⟩ -- x - 1 = 0
  let l₃ : Line := ⟨1, k, 0⟩  -- x + ky = 0
  divides_into_six_parts l₁ l₂ l₃ → k ∈ k_values := by
  sorry

end NUMINAMATH_CALUDE_line_division_theorem_l3040_304076


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l3040_304066

theorem scientific_notation_proof : 
  ∃ (a : ℝ) (n : ℤ), 680000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.8 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l3040_304066


namespace NUMINAMATH_CALUDE_tank_full_time_l3040_304089

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := system.capacity / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the tank will be full after 54 minutes -/
theorem tank_full_time (system : TankSystem) 
  (h1 : system.capacity = 900)
  (h2 : system.fill_rate_a = 40)
  (h3 : system.fill_rate_b = 30)
  (h4 : system.drain_rate = 20) :
  time_to_fill system = 54 := by
  sorry

#eval time_to_fill { capacity := 900, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l3040_304089


namespace NUMINAMATH_CALUDE_classroom_sum_problem_l3040_304008

theorem classroom_sum_problem (a b : ℤ) : 
  3 * a + 4 * b = 161 → (a = 17 ∨ b = 17) → (a = 31 ∨ b = 31) := by
  sorry

end NUMINAMATH_CALUDE_classroom_sum_problem_l3040_304008


namespace NUMINAMATH_CALUDE_dog_tail_length_l3040_304069

theorem dog_tail_length (body_length : ℝ) (head_length : ℝ) (tail_length : ℝ) 
  (overall_length : ℝ) (width : ℝ) (height : ℝ) :
  tail_length = body_length / 2 →
  head_length = body_length / 6 →
  height = 1.5 * width →
  overall_length = 30 →
  width = 12 →
  overall_length = body_length + head_length + tail_length →
  tail_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_tail_length_l3040_304069


namespace NUMINAMATH_CALUDE_difference_of_squares_l3040_304019

theorem difference_of_squares (m : ℝ) : m^2 - 16 = (m + 4) * (m - 4) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3040_304019


namespace NUMINAMATH_CALUDE_a2b2_value_l3040_304003

def is_arithmetic_progression (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_progression (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem a2b2_value (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h_arith : is_arithmetic_progression 1 a₁ a₂ 4)
  (h_geom : is_geometric_progression 1 b₁ b₂ b₃ 4) : 
  a₂ * b₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_a2b2_value_l3040_304003


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3040_304079

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 150)
  (h2 : throwers = 90)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 5 = 0)
  : total_players - (total_players - throwers) / 5 = 138 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3040_304079


namespace NUMINAMATH_CALUDE_largest_equal_sum_digits_l3040_304068

/-- The sum of decimal digits of a natural number -/
def sumDecimalDigits (n : ℕ) : ℕ := sorry

/-- The sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 503 is the largest number less than 1000 
    with equal sum of decimal and binary digits -/
theorem largest_equal_sum_digits : 
  ∀ n : ℕ, n < 1000 → n > 503 → 
    sumDecimalDigits n ≠ sumBinaryDigits n :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_sum_digits_l3040_304068


namespace NUMINAMATH_CALUDE_square_plus_one_divides_l3040_304006

theorem square_plus_one_divides (n : ℕ) : (n^2 + 1) ∣ n ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_one_divides_l3040_304006


namespace NUMINAMATH_CALUDE_shopkeeper_total_amount_l3040_304035

/-- Represents the total amount a shopkeeper receives for selling cloth. -/
def totalAmount (totalMetres : ℕ) (costPrice : ℕ) (lossPerMetre : ℕ) : ℕ :=
  totalMetres * (costPrice - lossPerMetre)

/-- Proves that the shopkeeper's total amount is 18000 for the given conditions. -/
theorem shopkeeper_total_amount :
  totalAmount 600 35 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_amount_l3040_304035


namespace NUMINAMATH_CALUDE_chicken_vaccine_probabilities_l3040_304041

theorem chicken_vaccine_probabilities :
  let n : ℕ := 5  -- number of chickens
  let p : ℝ := 0.8  -- probability of not being infected
  let q : ℝ := 1 - p  -- probability of being infected
  
  -- Probability of no chicken being infected
  (p ^ n : ℝ) = 1024 / 3125 ∧
  
  -- Probability of exactly one chicken being infected
  (n : ℝ) * (p ^ (n - 1)) * q = 256 / 625 :=
by sorry

end NUMINAMATH_CALUDE_chicken_vaccine_probabilities_l3040_304041


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l3040_304042

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def joe_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sarah_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => round_to_nearest_five (i + 1))

theorem sum_difference_theorem :
  joe_sum 60 - sarah_sum 60 = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l3040_304042


namespace NUMINAMATH_CALUDE_exam_probabilities_l3040_304057

/-- Represents the probability of passing the exam for each attempt -/
structure PassProbability where
  male : ℚ
  female : ℚ

/-- Represents the exam conditions -/
structure ExamConditions where
  pass_prob : PassProbability
  max_attempts : ℕ
  free_attempts : ℕ

/-- Calculates the probability of both passing within first two attempts -/
def prob_both_pass_free (conditions : ExamConditions) : ℚ :=
  sorry

/-- Calculates the probability of passing with one person requiring a third attempt -/
def prob_one_third_attempt (conditions : ExamConditions) : ℚ :=
  sorry

theorem exam_probabilities (conditions : ExamConditions) 
  (h1 : conditions.pass_prob.male = 3/4)
  (h2 : conditions.pass_prob.female = 2/3)
  (h3 : conditions.max_attempts = 5)
  (h4 : conditions.free_attempts = 2) :
  prob_both_pass_free conditions = 5/6 ∧ 
  prob_one_third_attempt conditions = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_exam_probabilities_l3040_304057


namespace NUMINAMATH_CALUDE_four_digit_number_counts_l3040_304002

def digits : Finset ℕ := {1, 2, 3, 4, 5}

def four_digit_numbers_no_repetition : ℕ := sorry

def four_digit_numbers_with_repetition : ℕ := sorry

def odd_four_digit_numbers_no_repetition : ℕ := sorry

theorem four_digit_number_counts :
  four_digit_numbers_no_repetition = 120 ∧
  four_digit_numbers_with_repetition = 625 ∧
  odd_four_digit_numbers_no_repetition = 72 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_counts_l3040_304002


namespace NUMINAMATH_CALUDE_caging_theorem_l3040_304015

/-- The number of ways to cage 6 animals in 6 cages, where 4 cages are too small for 6 animals -/
def caging_arrangements : ℕ := 24

/-- The total number of animals -/
def total_animals : ℕ := 6

/-- The total number of cages -/
def total_cages : ℕ := 6

/-- The number of cages that are too small for most animals -/
def small_cages : ℕ := 4

/-- The number of animals that can't fit in the small cages -/
def large_animals : ℕ := 6

theorem caging_theorem : 
  caging_arrangements = 24 ∧ 
  total_animals = 6 ∧ 
  total_cages = 6 ∧ 
  small_cages = 4 ∧ 
  large_animals = 6 :=
sorry

end NUMINAMATH_CALUDE_caging_theorem_l3040_304015


namespace NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l3040_304000

theorem complex_exp_210_deg_60th_power : 
  (Complex.exp (210 * π / 180 * I)) ^ 60 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l3040_304000


namespace NUMINAMATH_CALUDE_digit_150_is_zero_l3040_304039

/-- The decimal representation of 16/81 -/
def decimal_rep : ℚ := 16 / 81

/-- The repeating cycle in the decimal representation of 16/81 -/
def cycle : List ℕ := [1, 9, 7, 5, 3, 0, 8, 6, 4]

/-- The length of the repeating cycle -/
def cycle_length : ℕ := 9

/-- The position of the 150th digit within the cycle -/
def position_in_cycle : ℕ := 150 % cycle_length

/-- The 150th digit after the decimal point in the decimal representation of 16/81 -/
def digit_150 : ℕ := cycle[position_in_cycle]

theorem digit_150_is_zero : digit_150 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_zero_l3040_304039


namespace NUMINAMATH_CALUDE_product_of_three_primes_l3040_304081

theorem product_of_three_primes : 
  ∃ (p q r : ℕ), 
    989 * 1001 * 1007 + 320 = p * q * r ∧ 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p < q ∧ q < r ∧
    p = 991 ∧ q = 997 ∧ r = 1009 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_primes_l3040_304081


namespace NUMINAMATH_CALUDE_find_g_of_x_l3040_304074

theorem find_g_of_x (x : ℝ) (g : ℝ → ℝ) 
  (h : ∀ x, 4 * x^4 + 2 * x^2 - x + 7 + g x = x^3 - 4 * x^2 + 6) : 
  g = λ x => -4 * x^4 + x^3 - 6 * x^2 + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_find_g_of_x_l3040_304074


namespace NUMINAMATH_CALUDE_marble_count_l3040_304091

theorem marble_count (total : ℕ) (blue : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 50)
  (h2 : blue = 5)
  (h3 : prob_red_or_white = 9/10) :
  total - blue = 45 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l3040_304091


namespace NUMINAMATH_CALUDE_coordinate_change_l3040_304094

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors a, b, c
variable (a b c : V)

-- Define that {a, b, c} is a basis
variable (h₁ : LinearIndependent ℝ ![a, b, c])
variable (h₂ : Submodule.span ℝ {a, b, c} = ⊤)

-- Define that {a+b, a-b, c} is also a basis
variable (h₃ : LinearIndependent ℝ ![a + b, a - b, c])
variable (h₄ : Submodule.span ℝ {a + b, a - b, c} = ⊤)

-- Define the vector p
variable (p : V)

-- State the theorem
theorem coordinate_change (hp : p = a - 2 • b + 3 • c) :
  p = (-1/2 : ℝ) • (a + b) + (3/2 : ℝ) • (a - b) + 3 • c := by sorry

end NUMINAMATH_CALUDE_coordinate_change_l3040_304094


namespace NUMINAMATH_CALUDE_faster_walking_speed_l3040_304038

/-- Proves that given a person walks 50 km at 10 km/hr, if they walked at a faster speed
    for the same time and covered 70 km, the faster speed is 14 km/hr -/
theorem faster_walking_speed (actual_distance : ℝ) (original_speed : ℝ) (extra_distance : ℝ)
    (h1 : actual_distance = 50)
    (h2 : original_speed = 10)
    (h3 : extra_distance = 20) :
    let time := actual_distance / original_speed
    let total_distance := actual_distance + extra_distance
    let faster_speed := total_distance / time
    faster_speed = 14 := by sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l3040_304038


namespace NUMINAMATH_CALUDE_circle_properties_l3040_304048

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: Given the circle equation x^2 + y^2 + 4x - 6y - 3 = 0,
    prove that its center is (-2, 3) and its radius is 4 -/
theorem circle_properties :
  let eq : CircleEquation := ⟨4, -6, -3⟩
  let props : CircleProperties := ⟨(-2, 3), 4⟩
  (∀ x y : ℝ, x^2 + y^2 + eq.a * x + eq.b * y + eq.c = 0 ↔ 
    (x - props.center.1)^2 + (y - props.center.2)^2 = props.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3040_304048


namespace NUMINAMATH_CALUDE_log_relation_l3040_304054

theorem log_relation (a b : ℝ) : 
  a = Real.log 400 / Real.log 16 → b = Real.log 20 / Real.log 2 → a = b / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l3040_304054


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3040_304085

theorem complex_fraction_evaluation : (1 - I) / (2 + I) = 1/5 - 3/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3040_304085


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3040_304093

/-- Represents the roots of the quadratic equation x^2 - 8x + 15 = 0 --/
def roots : Set ℝ := {x : ℝ | x^2 - 8*x + 15 = 0}

/-- Represents an isosceles triangle with side lengths from the roots --/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  h_roots : side1 ∈ roots ∧ side2 ∈ roots
  h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base

/-- The perimeter of an isosceles triangle --/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

/-- Theorem stating that the perimeter of the isosceles triangle is either 11 or 13 --/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  perimeter t = 11 ∨ perimeter t = 13 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3040_304093


namespace NUMINAMATH_CALUDE_cement_bags_calculation_l3040_304097

theorem cement_bags_calculation (cement_cost : ℕ) (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ)
  (sand_cost_per_ton : ℕ) (total_payment : ℕ) :
  cement_cost = 10 →
  sand_lorries = 20 →
  sand_tons_per_lorry = 10 →
  sand_cost_per_ton = 40 →
  total_payment = 13000 →
  (total_payment - sand_lorries * sand_tons_per_lorry * sand_cost_per_ton) / cement_cost = 500 := by
  sorry

end NUMINAMATH_CALUDE_cement_bags_calculation_l3040_304097


namespace NUMINAMATH_CALUDE_equal_winning_chance_l3040_304004

/-- Represents a lottery ticket -/
structure LotteryTicket where
  id : ℕ

/-- Represents a lottery -/
structure Lottery where
  winningProbability : ℝ
  totalTickets : ℕ

/-- The probability of a ticket winning is equal to the lottery's winning probability -/
def ticketWinningProbability (lottery : Lottery) (ticket : LotteryTicket) : ℝ :=
  lottery.winningProbability

theorem equal_winning_chance (lottery : Lottery) 
    (h1 : lottery.winningProbability = 0.002)
    (h2 : lottery.totalTickets = 1000) :
    ∀ (t1 t2 : LotteryTicket), ticketWinningProbability lottery t1 = ticketWinningProbability lottery t2 :=
  sorry


end NUMINAMATH_CALUDE_equal_winning_chance_l3040_304004


namespace NUMINAMATH_CALUDE_angle_terminal_side_l3040_304009

theorem angle_terminal_side (θ : Real) (a : Real) : 
  (2 * Real.sin (π / 8) ^ 2 - 1, a) ∈ Set.range (λ t : Real × Real => (t.1 * Real.cos θ - t.2 * Real.sin θ, t.1 * Real.sin θ + t.2 * Real.cos θ)) ∧ 
  Real.sin θ = 2 * Real.sqrt 3 * Real.sin (13 * π / 12) * Real.cos (π / 12) →
  a = - Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l3040_304009


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l3040_304024

theorem units_digit_of_sum_of_cubes : 
  (42^3 + 24^3) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l3040_304024


namespace NUMINAMATH_CALUDE_exists_a_min_value_3_l3040_304077

open Real

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - log x

theorem exists_a_min_value_3 :
  ∃ a : ℝ, ∀ x : ℝ, 0 < x → x ≤ exp 1 → g a x ≥ 3 ∧
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ exp 1 ∧ g a x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_exists_a_min_value_3_l3040_304077


namespace NUMINAMATH_CALUDE_star_3_5_l3040_304051

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 3*a*b + b^2

-- Theorem statement
theorem star_3_5 : star 3 5 = 79 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l3040_304051


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3040_304021

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3040_304021


namespace NUMINAMATH_CALUDE_cylinder_volume_l3040_304023

/-- The volume of a cylinder with base radius 2 cm and height h cm is 4πh cm³ -/
theorem cylinder_volume (h : ℝ) : 
  let r : ℝ := 2
  let V : ℝ := π * r^2 * h
  V = 4 * π * h := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3040_304023


namespace NUMINAMATH_CALUDE_number_pattern_l3040_304080

/-- Represents a number as a string of consecutive '1' digits -/
def ones (n : ℕ) : ℕ :=
  (10 ^ n - 1) / 9

/-- The main theorem to be proved -/
theorem number_pattern (n : ℕ) (h : n ≤ 123456) :
  n * 9 + (n + 1) = ones (n + 1) :=
sorry

end NUMINAMATH_CALUDE_number_pattern_l3040_304080


namespace NUMINAMATH_CALUDE_bread_in_pond_l3040_304083

/-- Proves that the total number of bread pieces thrown in a pond is 100 given the specified conditions --/
theorem bread_in_pond (duck1_half : ℕ → ℕ) (duck2_pieces duck3_pieces left_in_water : ℕ) : 
  duck1_half = (λ x => x / 2) ∧ 
  duck2_pieces = 13 ∧ 
  duck3_pieces = 7 ∧ 
  left_in_water = 30 → 
  ∃ total : ℕ, 
    total = 100 ∧ 
    duck1_half total + duck2_pieces + duck3_pieces + left_in_water = total :=
by
  sorry

end NUMINAMATH_CALUDE_bread_in_pond_l3040_304083


namespace NUMINAMATH_CALUDE_percent_only_cat_owners_l3040_304046

/-- Given a school with the following student statistics:
  * total_students: The total number of students
  * cat_owners: The number of students who own cats
  * dog_owners: The number of students who own dogs
  * both_owners: The number of students who own both cats and dogs

  This theorem proves that the percentage of students who own only cats is 8%.
-/
theorem percent_only_cat_owners
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 80)
  (h3 : dog_owners = 150)
  (h4 : both_owners = 40) :
  (((cat_owners - both_owners : ℚ) / total_students) * 100 = 8) := by
  sorry

end NUMINAMATH_CALUDE_percent_only_cat_owners_l3040_304046


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3040_304007

theorem no_solution_absolute_value_equation :
  ¬∃ (x : ℝ), |2*x - 5| = 3*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3040_304007


namespace NUMINAMATH_CALUDE_inequality_solution_l3040_304033

theorem inequality_solution (x : ℝ) :
  (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ↔ 
  (x > -9/2 ∧ x < -2) ∨ (x > (1 - Real.sqrt 5) / 2 ∧ x < (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3040_304033


namespace NUMINAMATH_CALUDE_special_rectangle_difference_l3040_304098

/-- A rectangle with perimeter 4r and diagonal k times the length of one side -/
structure SpecialRectangle (r k : ℝ) where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 2 * r
  diagonal_eq : length ^ 2 + width ^ 2 = (k * length) ^ 2

/-- The absolute difference between length and width is k times the length -/
theorem special_rectangle_difference (r k : ℝ) (rect : SpecialRectangle r k) :
  |rect.length - rect.width| = k * rect.length :=
sorry

end NUMINAMATH_CALUDE_special_rectangle_difference_l3040_304098


namespace NUMINAMATH_CALUDE_complement_of_union_l3040_304014

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {1,3,6}
def N : Finset Nat := {2,3,4}

theorem complement_of_union : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3040_304014
