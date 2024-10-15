import Mathlib

namespace NUMINAMATH_CALUDE_brandys_trail_mix_raisins_l2486_248612

/-- The weight of raisins in a trail mix -/
def weight_of_raisins (weight_of_peanuts weight_of_chips total_weight : Real) : Real :=
  total_weight - (weight_of_peanuts + weight_of_chips)

/-- Theorem stating the weight of raisins in Brandy's trail mix -/
theorem brandys_trail_mix_raisins : 
  weight_of_raisins 0.17 0.17 0.42 = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_brandys_trail_mix_raisins_l2486_248612


namespace NUMINAMATH_CALUDE_no_solution_when_m_is_seven_l2486_248640

theorem no_solution_when_m_is_seven :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) ≠ (x - 7) / (x - 8) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_m_is_seven_l2486_248640


namespace NUMINAMATH_CALUDE_three_intersection_points_l2486_248654

-- Define the three lines
def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line2 x y ∧ line3 x y) ∨ (line1 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
sorry

end NUMINAMATH_CALUDE_three_intersection_points_l2486_248654


namespace NUMINAMATH_CALUDE_alpha_sufficient_not_necessary_for_beta_l2486_248696

theorem alpha_sufficient_not_necessary_for_beta :
  (∀ x : ℝ, x = -1 → x ≤ 0) ∧ 
  (∃ x : ℝ, x ≤ 0 ∧ x ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_alpha_sufficient_not_necessary_for_beta_l2486_248696


namespace NUMINAMATH_CALUDE_min_value_theorem_l2486_248672

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 2) (h4 : m * n > 0) :
  2 / m + 1 / n ≥ 9 / 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 2 ∧ m₀ * n₀ > 0 ∧ 2 / m₀ + 1 / n₀ = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2486_248672


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l2486_248623

theorem consecutive_even_numbers_divisible_by_eight (n : ℤ) : 
  ∃ k : ℤ, 4 * n * (n + 1) = 8 * k := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l2486_248623


namespace NUMINAMATH_CALUDE_max_stores_visited_l2486_248677

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  stores : Nat
  total_visits : Nat
  unique_visitors : Nat
  double_visitors : Nat

/-- Theorem stating the maximum number of stores visited by any single person -/
theorem max_stores_visited (s : ShoppingScenario) 
  (h1 : s.stores = 7)
  (h2 : s.total_visits = 21)
  (h3 : s.unique_visitors = 11)
  (h4 : s.double_visitors = 7)
  (h5 : s.double_visitors ≤ s.unique_visitors)
  (h6 : s.double_visitors * 2 ≤ s.total_visits) :
  ∃ (max_visits : Nat), max_visits = 4 ∧ 
  ∀ (individual_visits : Nat), individual_visits ≤ max_visits :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l2486_248677


namespace NUMINAMATH_CALUDE_sugar_recipes_l2486_248639

/-- The number of full recipes that can be made with a given amount of sugar -/
def full_recipes (total_sugar : ℚ) (sugar_per_recipe : ℚ) : ℚ :=
  total_sugar / sugar_per_recipe

/-- Theorem: Given 47 2/3 cups of sugar and a recipe requiring 1 1/2 cups of sugar,
    the number of full recipes that can be made is 31 7/9 -/
theorem sugar_recipes :
  let total_sugar : ℚ := 47 + 2/3
  let sugar_per_recipe : ℚ := 1 + 1/2
  full_recipes total_sugar sugar_per_recipe = 31 + 7/9 := by
sorry

end NUMINAMATH_CALUDE_sugar_recipes_l2486_248639


namespace NUMINAMATH_CALUDE_mike_yard_sale_books_l2486_248628

/-- Calculates the number of books bought at a yard sale -/
def books_bought_at_yard_sale (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem: The number of books Mike bought at the yard sale is 21 -/
theorem mike_yard_sale_books :
  let initial_books : ℕ := 35
  let final_books : ℕ := 56
  books_bought_at_yard_sale initial_books final_books = 21 := by
  sorry

end NUMINAMATH_CALUDE_mike_yard_sale_books_l2486_248628


namespace NUMINAMATH_CALUDE_modular_equivalence_in_range_l2486_248675

theorem modular_equivalence_in_range (a b : ℤ) (h1 : a ≡ 54 [ZMOD 53]) (h2 : b ≡ 98 [ZMOD 53]) :
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 200 ∧ (a - b) ≡ n [ZMOD 53] ∧ n = 168 := by
  sorry

end NUMINAMATH_CALUDE_modular_equivalence_in_range_l2486_248675


namespace NUMINAMATH_CALUDE_function_property_l2486_248693

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2486_248693


namespace NUMINAMATH_CALUDE_complement_subset_l2486_248651

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the set N
def N : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Theorem statement
theorem complement_subset : Set.compl N ⊆ Set.compl M := by
  sorry

end NUMINAMATH_CALUDE_complement_subset_l2486_248651


namespace NUMINAMATH_CALUDE_center_cell_value_l2486_248621

theorem center_cell_value (a b c d e f g h i : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0) →
  (a * b * c = 1) →
  (d * e * f = 1) →
  (g * h * i = 1) →
  (a * d * g = 1) →
  (b * e * h = 1) →
  (c * f * i = 1) →
  (a * b * d * e = 2) →
  (b * c * e * f = 2) →
  (d * e * g * h = 2) →
  (e * f * h * i = 2) →
  e = 1 :=
by sorry

end NUMINAMATH_CALUDE_center_cell_value_l2486_248621


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_max_value_f_min_value_ab_l2486_248633

-- Define the function f
def f (x : ℝ) : ℝ := |x + 5| - |x - 1|

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | -6 ≤ x ∧ x ≤ -4 ∨ x ≥ 6} :=
sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : 
  ∀ x : ℝ, f x ≤ 6 :=
sorry

-- Theorem for the minimum value of ab
theorem min_value_ab (a b : ℝ) (h : Real.log a + Real.log (2 * b) = Real.log (a + 4 * b + 6)) :
  a * b ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_max_value_f_min_value_ab_l2486_248633


namespace NUMINAMATH_CALUDE_computer_price_calculation_l2486_248627

theorem computer_price_calculation (P : ℝ) : 
  (P * 1.2 * 0.9 * 1.3 = 351) → P = 250 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_calculation_l2486_248627


namespace NUMINAMATH_CALUDE_sphere_radius_l2486_248689

theorem sphere_radius (V : Real) (r : Real) : V = (4 / 3) * Real.pi * r^3 → V = 36 * Real.pi → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l2486_248689


namespace NUMINAMATH_CALUDE_problem_statement_l2486_248606

theorem problem_statement (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 3) 
  (h3 : x = 1) : 
  y = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2486_248606


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2486_248601

theorem system_of_equations_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (solutions.card = 8) ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      (x = 2 * y^2 - 1 ∧ y = 2 * z^2 - 1 ∧ z = 2 * x^2 - 1)) :=
by sorry


end NUMINAMATH_CALUDE_system_of_equations_solutions_l2486_248601


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2486_248660

def sheep_horse_ratio : ℚ := 2 / 7
def horse_food_per_day : ℕ := 230
def total_horse_food : ℕ := 12880

theorem stewart_farm_sheep_count :
  ∃ (sheep horses : ℕ),
    sheep / horses = sheep_horse_ratio ∧
    horses * horse_food_per_day = total_horse_food ∧
    sheep = 16 := by sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2486_248660


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l2486_248695

def num_people : ℕ := 10

-- Function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => valid_arrangements (n + 1) + valid_arrangements n

def total_outcomes : ℕ := 2^num_people

theorem no_adjacent_standing_probability :
  (valid_arrangements num_people : ℚ) / total_outcomes = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l2486_248695


namespace NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_in_sequence_l2486_248652

/-- Given a natural number, return the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by the sum of its digits -/
def is_divisible_by_digit_sum (n : ℕ) : Prop :=
  n % digit_sum n = 0

/-- Theorem: In any sequence of 18 consecutive three-digit numbers, 
    at least one number is divisible by the sum of its digits -/
theorem exists_divisible_by_digit_sum_in_sequence :
  ∀ (start : ℕ), 100 ≤ start → start + 17 < 1000 →
  ∃ (k : ℕ), k ∈ Finset.range 18 ∧ is_divisible_by_digit_sum (start + k) :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_in_sequence_l2486_248652


namespace NUMINAMATH_CALUDE_complex_simplification_l2486_248699

theorem complex_simplification (i : ℂ) (h : i * i = -1) : 
  (1 + i) / i = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2486_248699


namespace NUMINAMATH_CALUDE_joey_work_hours_l2486_248610

/-- Calculates the number of hours Joey needs to work to buy sneakers -/
def hours_needed (sneaker_cost lawn_count lawn_pay figure_count figure_pay hourly_wage : ℕ) : ℕ :=
  let lawn_income := lawn_count * lawn_pay
  let figure_income := figure_count * figure_pay
  let total_income := lawn_income + figure_income
  let remaining_cost := sneaker_cost - total_income
  remaining_cost / hourly_wage

/-- Proves that Joey needs to work 10 hours to buy the sneakers -/
theorem joey_work_hours : 
  hours_needed 92 3 8 2 9 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_joey_work_hours_l2486_248610


namespace NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l2486_248622

-- Define the cost per 3 pages in cents
def cost_per_3_pages : ℚ := 7

-- Define the budget in dollars
def budget : ℚ := 35

-- Define the function to calculate the number of pages
def pages_copied (cost_per_3_pages budget : ℚ) : ℚ :=
  (budget * 100) * (3 / cost_per_3_pages)

-- Theorem statement
theorem pages_copied_for_35_dollars :
  pages_copied cost_per_3_pages budget = 1500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l2486_248622


namespace NUMINAMATH_CALUDE_triangle_area_implies_x_value_l2486_248666

theorem triangle_area_implies_x_value (x : ℝ) (h1 : x > 0) :
  (1/2 : ℝ) * x * (3*x) = 54 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_x_value_l2486_248666


namespace NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l2486_248669

/-- The series sum from n=1 to infinity of (3^n) / (1 + 3^n + 3^(n+1) + 3^(2n+1)) equals 1/4 -/
theorem series_sum_equals_one_fourth :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l2486_248669


namespace NUMINAMATH_CALUDE_part1_part2_l2486_248682

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

-- Part 1
theorem part1 (a b c : ℝ) (h : |a - b| > c) : ∀ x : ℝ, f x a b > c := by sorry

-- Part 2
theorem part2 (a : ℝ) :
  (∃ x : ℝ, f x a 1 < 2 - |a - 2|) ↔ (1/2 < a ∧ a < 5/2) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2486_248682


namespace NUMINAMATH_CALUDE_tan_cube_identity_l2486_248655

theorem tan_cube_identity (x y : ℝ) (φ : ℝ) (h : Real.tan φ ^ 3 = x / y) :
  x / Real.sin φ + y / Real.cos φ = (x ^ (2/3) + y ^ (2/3)) ^ (3/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_cube_identity_l2486_248655


namespace NUMINAMATH_CALUDE_chess_match_draw_probability_l2486_248608

theorem chess_match_draw_probability (john_win_prob mike_win_prob : ℚ) 
  (h1 : john_win_prob = 4/9)
  (h2 : mike_win_prob = 5/18) : 
  1 - (john_win_prob + mike_win_prob) = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_draw_probability_l2486_248608


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2486_248650

open Real

theorem trigonometric_identities (α : ℝ) :
  (tan α = 1 / 3) →
  (1 / (2 * sin α * cos α + cos α ^ 2) = 2 / 3) ∧
  (tan (π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) / (cos (-α - π) * sin (-π - α)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2486_248650


namespace NUMINAMATH_CALUDE_special_function_property_l2486_248671

/-- A differentiable function f satisfying f(x) - f''(x) > 0 for all x --/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x : ℝ, Differentiable ℝ (deriv f)) ∧
  (∀ x : ℝ, f x - (deriv (deriv f)) x > 0)

/-- Theorem stating that for a special function f, ef(2015) > f(2016) --/
theorem special_function_property (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  Real.exp 1 * f 2015 > f 2016 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l2486_248671


namespace NUMINAMATH_CALUDE_fence_area_calculation_l2486_248670

/-- The time (in hours) it takes the first painter to paint the entire fence alone -/
def painter1_time : ℝ := 12

/-- The time (in hours) it takes the second painter to paint the entire fence alone -/
def painter2_time : ℝ := 15

/-- The reduction in combined painting speed (in square feet per hour) when the painters work together -/
def speed_reduction : ℝ := 5

/-- The time (in hours) it takes both painters to paint the fence together -/
def combined_time : ℝ := 7

/-- The total area of the fence in square feet -/
def fence_area : ℝ := 700

theorem fence_area_calculation :
  (combined_time * (fence_area / painter1_time + fence_area / painter2_time - speed_reduction) = fence_area) :=
sorry

end NUMINAMATH_CALUDE_fence_area_calculation_l2486_248670


namespace NUMINAMATH_CALUDE_distinct_primes_not_dividing_l2486_248683

/-- A function that pairs the positive divisors of a number -/
def divisor_pairing (n : ℕ+) : Set (ℕ × ℕ) := sorry

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem -/
theorem distinct_primes_not_dividing (n : ℕ+) 
  (h : ∀ (pair : ℕ × ℕ), pair ∈ divisor_pairing n → is_prime (pair.1 + pair.2)) :
  (∀ (p q : ℕ), 
    (∃ (pair1 pair2 : ℕ × ℕ), pair1 ∈ divisor_pairing n ∧ pair2 ∈ divisor_pairing n ∧ 
      p = pair1.1 + pair1.2 ∧ q = pair2.1 + pair2.2 ∧ p ≠ q) →
    (∀ (r : ℕ), (∃ (pair : ℕ × ℕ), pair ∈ divisor_pairing n ∧ r = pair.1 + pair.2) → 
      ¬(r ∣ n))) :=
sorry

end NUMINAMATH_CALUDE_distinct_primes_not_dividing_l2486_248683


namespace NUMINAMATH_CALUDE_parabola_equation_l2486_248604

/-- A parabola with focus (0,1) and vertex at (0,0) has the standard equation x^2 = 4y -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (0, 1)
  let vertex : ℝ × ℝ := (0, 0)
  let p : ℝ := focus.2 - vertex.2
  (x^2 = 4*y) ↔ (
    (∀ (x' y' : ℝ), (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - (focus.2 - p))^2) ∧
    vertex = (0, 0) ∧
    focus = (0, 1)
  ) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2486_248604


namespace NUMINAMATH_CALUDE_monotonicity_intervals_two_zeros_condition_l2486_248676

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((1-a)/2) * x^2 - a*x - a

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) (h : a > 0) :
  (∀ x < -1, (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo (-1) a, (f' a x < 0)) ∧
  (∀ x > a, (f' a x > 0)) :=
sorry

-- Theorem for the range of a when f has exactly two zeros in (-2, 0)
theorem two_zeros_condition (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Ioo (-2) 0 ∧ y ∈ Set.Ioo (-2) 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   ∀ z ∈ Set.Ioo (-2) 0, f a z = 0 → (z = x ∨ z = y)) ↔
  (a > 0 ∧ a < 1/3) :=
sorry

end

end NUMINAMATH_CALUDE_monotonicity_intervals_two_zeros_condition_l2486_248676


namespace NUMINAMATH_CALUDE_fred_found_43_seashells_l2486_248625

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The total number of seashells found by Tom and Fred -/
def total_seashells : ℕ := 58

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := total_seashells - tom_seashells

theorem fred_found_43_seashells : fred_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_fred_found_43_seashells_l2486_248625


namespace NUMINAMATH_CALUDE_students_in_biology_or_chemistry_l2486_248609

theorem students_in_biology_or_chemistry (both : ℕ) (biology : ℕ) (chemistry_only : ℕ) : 
  both = 15 → biology = 35 → chemistry_only = 18 → 
  (biology - both) + chemistry_only = 38 := by
sorry

end NUMINAMATH_CALUDE_students_in_biology_or_chemistry_l2486_248609


namespace NUMINAMATH_CALUDE_lattice_points_limit_l2486_248688

/-- The number of lattice points inside a circle of radius r centered at the origin -/
noncomputable def f (r : ℝ) : ℝ := sorry

/-- The difference between f(r) and πr^2 -/
noncomputable def g (r : ℝ) : ℝ := f r - Real.pi * r^2

theorem lattice_points_limit :
  (∀ ε > 0, ∃ R, ∀ r ≥ R, |f r / r^2 - Real.pi| < ε) ∧
  (∀ h < 2, ∀ ε > 0, ∃ R, ∀ r ≥ R, |g r / r^h| < ε) := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_limit_l2486_248688


namespace NUMINAMATH_CALUDE_circles_common_chord_l2486_248681

-- Define the circles
def circle1 (x y a : ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4
def circle2 (x y b : ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1

-- Define the condition for intersection
def intersect (a b : ℝ) : Prop := 1 < |a + b| ∧ |a + b| < Real.sqrt 3

-- Define the equation of the common chord
def common_chord (x a b : ℝ) : Prop := (2*a + 2*b)*x + 3 + b^2 - a^2 = 0

-- Theorem statement
theorem circles_common_chord (a b : ℝ) (h : intersect a b) :
  ∀ x y : ℝ, circle1 x y a ∧ circle2 x y b → common_chord x a b :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l2486_248681


namespace NUMINAMATH_CALUDE_unique_denomination_l2486_248647

/-- Given unlimited supply of stamps of denominations 4, n, and n+1 cents,
    57 cents is the greatest postage that cannot be formed -/
def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 57 → ∃ a b c : ℕ, k = 4*a + n*b + (n+1)*c

/-- 21 is the only positive integer satisfying the condition -/
theorem unique_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n :=
sorry

end NUMINAMATH_CALUDE_unique_denomination_l2486_248647


namespace NUMINAMATH_CALUDE_inequality_solution_l2486_248637

def solution_set (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 2 then {x | 1 < x ∧ x ≤ 2/a}
  else if a = 2 then ∅
  else if a > 2 then {x | 2/a ≤ x ∧ x < 1}
  else ∅

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} = solution_set a := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2486_248637


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2486_248690

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The fourth term of a geometric sequence with first term 3 and second term 1/3 is 1/243 -/
theorem fourth_term_of_geometric_sequence :
  let a := 3
  let a₂ := 1/3
  let r := a₂ / a
  geometric_term a r 4 = 1/243 := by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2486_248690


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l2486_248620

theorem probability_of_black_ball
  (p_red : ℝ) (p_white : ℝ) (p_black : ℝ)
  (h1 : p_red = 0.52)
  (h2 : p_white = 0.28)
  (h3 : p_red + p_white + p_black = 1) :
  p_black = 0.2 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l2486_248620


namespace NUMINAMATH_CALUDE_quartic_roots_sum_l2486_248680

theorem quartic_roots_sum (p q r s : ℂ) : 
  (p^4 = p^2 + p + 2) → 
  (q^4 = q^2 + q + 2) → 
  (r^4 = r^2 + r + 2) → 
  (s^4 = s^2 + s + 2) → 
  p * (q - r)^2 + q * (r - s)^2 + r * (s - p)^2 + s * (p - q)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_quartic_roots_sum_l2486_248680


namespace NUMINAMATH_CALUDE_positive_sum_product_iff_l2486_248605

theorem positive_sum_product_iff (a b : ℝ) : (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_product_iff_l2486_248605


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2486_248630

def selling_price : ℝ := 24000
def cost_price : ℝ := 20000
def potential_profit_percentage : ℝ := 8

theorem discount_percentage_calculation :
  let potential_profit := (potential_profit_percentage / 100) * cost_price
  let selling_price_with_potential_profit := cost_price + potential_profit
  let discount_amount := selling_price - selling_price_with_potential_profit
  let discount_percentage := (discount_amount / selling_price) * 100
  discount_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2486_248630


namespace NUMINAMATH_CALUDE_average_age_increase_l2486_248664

theorem average_age_increase (n : ℕ) (A : ℝ) : 
  n = 10 → 
  ((n * A + 21 + 21 - 10 - 12) / n) - A = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l2486_248664


namespace NUMINAMATH_CALUDE_james_water_storage_l2486_248684

def cask_capacity : ℕ := 20

def barrel_capacity (cask_cap : ℕ) : ℕ := 2 * cask_cap + 3

def total_storage (cask_cap barrel_cap num_barrels : ℕ) : ℕ :=
  cask_cap + num_barrels * barrel_cap

theorem james_water_storage :
  total_storage cask_capacity (barrel_capacity cask_capacity) 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_james_water_storage_l2486_248684


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l2486_248646

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l2486_248646


namespace NUMINAMATH_CALUDE_circular_tablecloth_radius_increase_l2486_248636

theorem circular_tablecloth_radius_increase :
  let initial_circumference : ℝ := 50
  let final_circumference : ℝ := 64
  let initial_radius : ℝ := initial_circumference / (2 * Real.pi)
  let final_radius : ℝ := final_circumference / (2 * Real.pi)
  final_radius - initial_radius = 7 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circular_tablecloth_radius_increase_l2486_248636


namespace NUMINAMATH_CALUDE_annual_loss_is_14400_l2486_248635

/-- The number of yellow balls in the box -/
def yellow_balls : ℕ := 3

/-- The number of white balls in the box -/
def white_balls : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := yellow_balls + white_balls

/-- The number of balls drawn in each attempt -/
def drawn_balls : ℕ := 3

/-- The reward for drawing 3 balls of the same color (in yuan) -/
def same_color_reward : ℚ := 5

/-- The payment for drawing 3 balls of different colors (in yuan) -/
def diff_color_payment : ℚ := 1

/-- The number of people drawing balls per day -/
def people_per_day : ℕ := 100

/-- The number of days in a year for this calculation -/
def days_per_year : ℕ := 360

/-- The probability of drawing 3 balls of the same color -/
def prob_same_color : ℚ := 1 / 10

/-- The probability of drawing 3 balls of different colors -/
def prob_diff_color : ℚ := 9 / 10

/-- The expected earnings per person (in yuan) -/
def expected_earnings_per_person : ℚ := 
  prob_same_color * same_color_reward - prob_diff_color * diff_color_payment

/-- The daily earnings (in yuan) -/
def daily_earnings : ℚ := expected_earnings_per_person * people_per_day

/-- Theorem: The annual loss is 14400 yuan -/
theorem annual_loss_is_14400 : 
  -daily_earnings * days_per_year = 14400 := by sorry

end NUMINAMATH_CALUDE_annual_loss_is_14400_l2486_248635


namespace NUMINAMATH_CALUDE_not_all_rationals_repeating_l2486_248679

-- Define rational numbers
def Rational : Type := ℚ

-- Define integers
def Integer : Type := ℤ

-- Define repeating decimal
def RepeatingDecimal (x : ℚ) : Prop := sorry

-- Statement that integers are rational numbers
axiom integer_is_rational : Integer → Rational

-- Statement that not all integers are repeating decimals
axiom not_all_integers_repeating : ∃ (n : Integer), ¬(RepeatingDecimal (integer_is_rational n))

-- Theorem to prove
theorem not_all_rationals_repeating : ¬(∀ (q : Rational), RepeatingDecimal q) := by
  sorry

end NUMINAMATH_CALUDE_not_all_rationals_repeating_l2486_248679


namespace NUMINAMATH_CALUDE_rectangle_area_l2486_248632

theorem rectangle_area (length width area : ℝ) : 
  length = 24 →
  width = 0.875 * length →
  area = length * width →
  area = 504 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2486_248632


namespace NUMINAMATH_CALUDE_painting_price_change_l2486_248638

theorem painting_price_change (P : ℝ) (h : P > 0) : 
  let first_year_price := 1.30 * P
  let final_price := 1.105 * P
  let second_year_decrease := (first_year_price - final_price) / first_year_price
  second_year_decrease = 0.15 := by sorry

end NUMINAMATH_CALUDE_painting_price_change_l2486_248638


namespace NUMINAMATH_CALUDE_hiking_campers_l2486_248698

theorem hiking_campers (morning_rowing : ℕ) (afternoon_rowing : ℕ) (total_campers : ℕ)
  (h1 : morning_rowing = 41)
  (h2 : afternoon_rowing = 26)
  (h3 : total_campers = 71)
  : total_campers - (morning_rowing + afternoon_rowing) = 4 := by
  sorry

end NUMINAMATH_CALUDE_hiking_campers_l2486_248698


namespace NUMINAMATH_CALUDE_bernold_can_win_l2486_248667

/-- Represents the game board -/
structure GameBoard :=
  (size : Nat)
  (arnold_moves : Nat → Nat → Bool)
  (bernold_moves : Nat → Nat → Bool)

/-- Defines the game rules -/
def game_rules (board : GameBoard) : Prop :=
  board.size = 2007 ∧
  (∀ x y, board.arnold_moves x y ↔ 
    x + 1 < board.size ∧ y + 1 < board.size ∧ 
    ¬board.bernold_moves x y ∧ ¬board.bernold_moves (x+1) y ∧ 
    ¬board.bernold_moves x (y+1) ∧ ¬board.bernold_moves (x+1) (y+1)) ∧
  (∀ x y, board.bernold_moves x y → x < board.size ∧ y < board.size)

/-- Theorem: Bernold can always win -/
theorem bernold_can_win (board : GameBoard) (h : game_rules board) :
  ∃ (strategy : Nat → Nat → Bool), 
    (∀ x y, strategy x y → board.bernold_moves x y) ∧
    (∀ (arnold_strategy : Nat → Nat → Bool), 
      (∀ x y, arnold_strategy x y → board.arnold_moves x y) →
      (Finset.sum (Finset.product (Finset.range board.size) (Finset.range board.size))
        (fun (x, y) => if arnold_strategy x y then 4 else 0) ≤ 
          (1003 * 1004) / 2)) :=
sorry

end NUMINAMATH_CALUDE_bernold_can_win_l2486_248667


namespace NUMINAMATH_CALUDE_inverse_difference_l2486_248614

-- Define a real-valued function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the condition that f(x+2) is the inverse of f⁻¹(x-1)
axiom inverse_condition : ∀ x, f (x + 2) = f_inv (x - 1)

-- Define the theorem
theorem inverse_difference :
  f_inv 2010 - f_inv 1 = 4018 :=
sorry

end NUMINAMATH_CALUDE_inverse_difference_l2486_248614


namespace NUMINAMATH_CALUDE_d11d_divisible_by_5_l2486_248645

/-- Represents a base-7 digit -/
def Base7Digit := {d : ℕ // d < 7}

/-- Converts a base-7 number of the form d11d to its decimal equivalent -/
def toDecimal (d : Base7Digit) : ℕ := 344 * d.val + 56

/-- A base-7 number d11d_7 is divisible by 5 if and only if d = 1 -/
theorem d11d_divisible_by_5 (d : Base7Digit) : 
  5 ∣ toDecimal d ↔ d.val = 1 := by sorry

end NUMINAMATH_CALUDE_d11d_divisible_by_5_l2486_248645


namespace NUMINAMATH_CALUDE_dice_probability_l2486_248665

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of outcomes where all dice show the same number -/
def allSameOutcomes : ℕ := numSides

/-- The number of possible sequences (e.g., 1-2-3-4-5, 2-3-4-5-6) -/
def numSequences : ℕ := 2

/-- The number of ways to arrange each sequence -/
def sequenceArrangements : ℕ := Nat.factorial numDice

/-- The probability of rolling five fair 6-sided dice where they don't all show
    the same number and the numbers do not form a sequence -/
theorem dice_probability : 
  (totalOutcomes - allSameOutcomes - numSequences * sequenceArrangements) / totalOutcomes = 7530 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l2486_248665


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l2486_248678

/-- The number of green balls in the first urn -/
def green1 : ℕ := 6

/-- The number of blue balls in the first urn -/
def blue1 : ℕ := 4

/-- The number of green balls in the second urn -/
def green2 : ℕ := 20

/-- The probability that both drawn balls are of the same color -/
def same_color_prob : ℚ := 65/100

/-- The number of blue balls in the second urn -/
def N : ℕ := 4

/-- The total number of balls in the first urn -/
def total1 : ℕ := green1 + blue1

/-- The total number of balls in the second urn -/
def total2 : ℕ := green2 + N

theorem urn_probability_theorem :
  (green1 : ℚ) / total1 * green2 / total2 + (blue1 : ℚ) / total1 * N / total2 = same_color_prob :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l2486_248678


namespace NUMINAMATH_CALUDE_thursday_seeds_count_l2486_248644

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := 20

/-- The total number of seeds planted -/
def total_seeds : ℕ := 22

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := total_seeds - seeds_wednesday

theorem thursday_seeds_count : seeds_thursday = 2 := by
  sorry

end NUMINAMATH_CALUDE_thursday_seeds_count_l2486_248644


namespace NUMINAMATH_CALUDE_complement_intersection_MN_l2486_248617

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_MN : 
  (M ∩ N)ᶜ = {1, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_MN_l2486_248617


namespace NUMINAMATH_CALUDE_eight_power_y_equals_one_eighth_of_two_power_36_l2486_248600

theorem eight_power_y_equals_one_eighth_of_two_power_36 :
  ∀ y : ℝ, (1/8 : ℝ) * (2^36) = 8^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_y_equals_one_eighth_of_two_power_36_l2486_248600


namespace NUMINAMATH_CALUDE_angle_properties_l2486_248662

/-- Given a point P on the unit circle, determine the quadrant and smallest positive angle -/
theorem angle_properties (α : Real) : 
  (∃ P : Real × Real, P.1 = Real.sin (5 * Real.pi / 6) ∧ P.2 = Real.cos (5 * Real.pi / 6) ∧ 
   P.1 = Real.sin α ∧ P.2 = Real.cos α) →
  (α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) ∧
  (∃ β : Real, β = 5 * Real.pi / 3 ∧ 
   Real.sin β = Real.sin α ∧ Real.cos β = Real.cos α ∧
   ∀ γ : Real, 0 < γ ∧ γ < β → 
   Real.sin γ ≠ Real.sin α ∨ Real.cos γ ≠ Real.cos α) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l2486_248662


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2486_248692

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ ≠ r₂) ∧ 
    (∀ x : ℝ, x^2 + p*x + m = 0 ↔ x = r₁ ∨ x = r₂) ∧
    (∀ x : ℝ, x^2 + m*x + n = 0 ↔ x = r₁/2 ∨ x = r₂/2)) →
  n / p = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2486_248692


namespace NUMINAMATH_CALUDE_sharks_score_l2486_248616

theorem sharks_score (total_points eagles_points sharks_points : ℕ) : 
  total_points = 60 → 
  eagles_points = sharks_points + 18 → 
  eagles_points + sharks_points = total_points → 
  sharks_points = 21 := by
sorry

end NUMINAMATH_CALUDE_sharks_score_l2486_248616


namespace NUMINAMATH_CALUDE_leahs_birdseed_supply_l2486_248686

/-- Represents the number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feed (boxes_bought : ℕ) (boxes_in_pantry : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) (grams_per_box : ℕ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let weekly_consumption := parrot_consumption + cockatiel_consumption
  total_grams / weekly_consumption

/-- Theorem stating that Leah can feed her birds for 12 weeks without going back to the store -/
theorem leahs_birdseed_supply : weeks_of_feed 3 5 100 50 225 = 12 := by
  sorry

end NUMINAMATH_CALUDE_leahs_birdseed_supply_l2486_248686


namespace NUMINAMATH_CALUDE_section_A_average_weight_l2486_248687

/-- Proves that the average weight of section A is 40 kg given the conditions of the problem -/
theorem section_A_average_weight
  (students_A : ℕ)
  (students_B : ℕ)
  (avg_weight_B : ℝ)
  (avg_weight_total : ℝ)
  (h1 : students_A = 30)
  (h2 : students_B = 20)
  (h3 : avg_weight_B = 35)
  (h4 : avg_weight_total = 38) :
  let total_students := students_A + students_B
  let avg_weight_A := (avg_weight_total * total_students - avg_weight_B * students_B) / students_A
  avg_weight_A = 40 := by
sorry


end NUMINAMATH_CALUDE_section_A_average_weight_l2486_248687


namespace NUMINAMATH_CALUDE_average_speed_first_part_l2486_248673

def total_distance : ℝ := 250
def total_time : ℝ := 5.4
def distance_at_v : ℝ := 148
def speed_known : ℝ := 60

theorem average_speed_first_part (v : ℝ) : 
  (distance_at_v / v) + ((total_distance - distance_at_v) / speed_known) = total_time →
  v = 40 := by
sorry

end NUMINAMATH_CALUDE_average_speed_first_part_l2486_248673


namespace NUMINAMATH_CALUDE_B_equals_C_l2486_248603

def A : Set Int := {-1, 1}

def B : Set Int := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x + y}

def C : Set Int := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x - y}

theorem B_equals_C : B = C := by sorry

end NUMINAMATH_CALUDE_B_equals_C_l2486_248603


namespace NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l2486_248607

/-- A natural number has exactly three distinct divisors if and only if it is the square of a prime number. -/
theorem three_divisors_iff_prime_square (n : ℕ) : (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ d ∈ s, d ∣ n) ↔ ∃ p, Nat.Prime p ∧ n = p^2 := by
  sorry

end NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l2486_248607


namespace NUMINAMATH_CALUDE_expression_simplification_l2486_248694

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3) :
  (3 / (a - 1) + 1) / ((a^2 + 2*a) / (a^2 - 1)) = (3 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2486_248694


namespace NUMINAMATH_CALUDE_clothing_store_profit_l2486_248657

/-- Represents the daily profit function for a clothing store -/
def daily_profit (cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_price - price_reduction - cost) * (initial_sales + 2 * price_reduction)

theorem clothing_store_profit 
  (cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ)
  (h_cost : cost = 50)
  (h_initial_price : initial_price = 90)
  (h_initial_sales : initial_sales = 20) :
  (∃ (x : ℝ), daily_profit cost initial_price initial_sales x = 1200) ∧
  (¬ ∃ (y : ℝ), daily_profit cost initial_price initial_sales y = 2000) := by
  sorry

#check clothing_store_profit

end NUMINAMATH_CALUDE_clothing_store_profit_l2486_248657


namespace NUMINAMATH_CALUDE_pizza_group_composition_l2486_248613

theorem pizza_group_composition :
  ∀ (boys girls : ℕ),
  (∀ (b : ℕ), b ≤ boys → 6 ≤ b ∧ b ≤ 7) →
  (∀ (g : ℕ), g ≤ girls → 2 ≤ g ∧ g ≤ 3) →
  49 ≤ 6 * boys + 2 * girls →
  7 * boys + 3 * girls ≤ 59 →
  boys = 8 ∧ girls = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_group_composition_l2486_248613


namespace NUMINAMATH_CALUDE_tile_difference_l2486_248641

theorem tile_difference (initial_blue : ℕ) (initial_green : ℕ) (border_tiles : ℕ) :
  initial_blue = 15 →
  initial_green = 8 →
  border_tiles = 12 →
  (initial_blue + border_tiles / 2) - (initial_green + border_tiles / 2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_tile_difference_l2486_248641


namespace NUMINAMATH_CALUDE_f_of_f_of_one_eq_31_l2486_248642

def f (x : ℝ) : ℝ := 4 * x^3 + 2 * x^2 - 5 * x + 1

theorem f_of_f_of_one_eq_31 : f (f 1) = 31 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_one_eq_31_l2486_248642


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l2486_248668

def total_missiles : ℕ := 70
def selected_missiles : ℕ := 7

def systematic_sample (start : ℕ) (interval : ℕ) : List ℕ :=
  List.range selected_missiles |>.map (fun i => start + i * interval)

theorem correct_systematic_sample :
  ∃ (start : ℕ), start ≤ total_missiles ∧
  systematic_sample start (total_missiles / selected_missiles) =
    [3, 13, 23, 33, 43, 53, 63] :=
by sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l2486_248668


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2486_248656

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 + Complex.I)) :
  z.im = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2486_248656


namespace NUMINAMATH_CALUDE_millet_majority_on_wednesday_l2486_248611

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Rat
  other_seeds : Rat

/-- Calculates the next day's feeder state -/
def next_day (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet * (4/5),
    other_seeds := 0 }

/-- Adds new seeds to the feeder (every other day) -/
def add_seeds (state : FeederState) : FeederState :=
  { day := state.day,
    millet := state.millet + 2/5,
    other_seeds := state.other_seeds + 3/5 }

/-- Initial state of the feeder on Monday -/
def initial_state : FeederState :=
  { day := 1, millet := 2/5, other_seeds := 3/5 }

/-- Theorem: On Wednesday (day 3), millet is more than half of total seeds -/
theorem millet_majority_on_wednesday :
  let state_wednesday := add_seeds (next_day (next_day initial_state))
  state_wednesday.millet > (state_wednesday.millet + state_wednesday.other_seeds) / 2 := by
  sorry


end NUMINAMATH_CALUDE_millet_majority_on_wednesday_l2486_248611


namespace NUMINAMATH_CALUDE_figure_404_has_2022_squares_l2486_248624

/-- The number of squares in the nth figure of the sequence -/
def squares_in_figure (n : ℕ) : ℕ := 7 + (n - 1) * 5

theorem figure_404_has_2022_squares :
  squares_in_figure 404 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_figure_404_has_2022_squares_l2486_248624


namespace NUMINAMATH_CALUDE_oscar_christina_age_ratio_l2486_248602

def christina_age : ℕ := sorry
def oscar_age : ℕ := 6

theorem oscar_christina_age_ratio :
  (oscar_age + 15) / christina_age = 3 / 5 :=
by
  have h1 : christina_age + 5 = 80 / 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_oscar_christina_age_ratio_l2486_248602


namespace NUMINAMATH_CALUDE_piano_lesson_cost_l2486_248685

/-- Calculate the total cost of piano lessons -/
theorem piano_lesson_cost (lesson_cost : ℝ) (lesson_duration : ℝ) (total_hours : ℝ) : 
  lesson_cost = 30 ∧ lesson_duration = 1.5 ∧ total_hours = 18 →
  (total_hours / lesson_duration) * lesson_cost = 360 := by
  sorry

end NUMINAMATH_CALUDE_piano_lesson_cost_l2486_248685


namespace NUMINAMATH_CALUDE_points_on_parabola_l2486_248658

-- Define the sequence of points
def SequencePoints (x y : ℕ → ℝ) : Prop :=
  ∀ n, Real.sqrt ((x n)^2 + (y n)^2) - y n = 6

-- Define the parabola
def OnParabola (x y : ℝ) : Prop :=
  y = (x^2 / 12) - 3

-- Theorem statement
theorem points_on_parabola 
  (x y : ℕ → ℝ) 
  (h : SequencePoints x y) :
  ∀ n, OnParabola (x n) (y n) := by
sorry

end NUMINAMATH_CALUDE_points_on_parabola_l2486_248658


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2486_248691

theorem cricket_team_average_age
  (n : ℕ) -- Total number of players
  (a : ℝ) -- Average age of the whole team
  (h1 : n = 11)
  (h2 : a = 28)
  (h3 : ((n * a) - (a + (a + 3))) / (n - 2) = a - 1) :
  a = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2486_248691


namespace NUMINAMATH_CALUDE_original_price_correct_l2486_248649

/-- The original price of a shirt before discount -/
def original_price : ℝ := 975

/-- The discount percentage applied to the shirt -/
def discount_percentage : ℝ := 0.20

/-- The discounted price of the shirt -/
def discounted_price : ℝ := 780

/-- Theorem stating that the original price is correct given the discount and discounted price -/
theorem original_price_correct : 
  original_price * (1 - discount_percentage) = discounted_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_correct_l2486_248649


namespace NUMINAMATH_CALUDE_incorrect_calculation_correction_l2486_248629

theorem incorrect_calculation_correction (x : ℝ) : 
  25 * x = 812 → x / 4 = 8.12 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_correction_l2486_248629


namespace NUMINAMATH_CALUDE_total_family_members_eq_243_l2486_248663

/-- The total number of grandchildren and extended family members for Grandma Olga -/
def total_family_members : ℕ :=
  let daughters := 6
  let sons := 5
  let children_per_daughter := 10 + 9  -- 10 sons + 9 daughters
  let stepchildren_per_daughter := 4
  let children_per_son := 8 + 7  -- 8 daughters + 7 sons
  let inlaws_per_son := 3
  let children_per_inlaw := 2

  daughters * children_per_daughter +
  daughters * stepchildren_per_daughter +
  sons * children_per_son +
  sons * inlaws_per_son * children_per_inlaw

theorem total_family_members_eq_243 :
  total_family_members = 243 := by
  sorry

end NUMINAMATH_CALUDE_total_family_members_eq_243_l2486_248663


namespace NUMINAMATH_CALUDE_star_composition_l2486_248643

/-- Define the binary operation ★ -/
def star (x y : ℝ) : ℝ := x^2 - 2*y + 1

/-- Theorem: For any real number k, k ★ (k ★ k) = -k^2 + 4k - 1 -/
theorem star_composition (k : ℝ) : star k (star k k) = -k^2 + 4*k - 1 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l2486_248643


namespace NUMINAMATH_CALUDE_interest_calculation_period_l2486_248619

theorem interest_calculation_period (P n : ℝ) 
  (h1 : P * n / 20 = 40)
  (h2 : P * ((1 + 0.05)^n - 1) = 41) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_calculation_period_l2486_248619


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2486_248648

theorem binomial_divisibility (p k : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ m : ℤ, (p : ℤ)^3 * m = (Nat.choose (k * p) p : ℤ) - k := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2486_248648


namespace NUMINAMATH_CALUDE_min_sum_of_product_2004_l2486_248626

theorem min_sum_of_product_2004 (x y z : ℕ+) (h : x * y * z = 2004) :
  ∃ (a b c : ℕ+), a * b * c = 2004 ∧ a + b + c ≤ x + y + z ∧ a + b + c = 174 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2004_l2486_248626


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2486_248661

/-- The side length of an equilateral triangle with perimeter 2 meters is 2/3 meters. -/
theorem equilateral_triangle_side_length : 
  ∀ (side_length : ℝ), 
    (side_length > 0) →
    (3 * side_length = 2) →
    side_length = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2486_248661


namespace NUMINAMATH_CALUDE_credit_card_balance_transfer_l2486_248697

theorem credit_card_balance_transfer (G : ℝ) : 
  let gold_limit : ℝ := G
  let platinum_limit : ℝ := 2 * G
  let gold_balance : ℝ := G / 3
  let platinum_balance : ℝ := platinum_limit / 4
  let new_platinum_balance : ℝ := platinum_balance + gold_balance
  let unspent_portion : ℝ := (platinum_limit - new_platinum_balance) / platinum_limit
  unspent_portion = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_credit_card_balance_transfer_l2486_248697


namespace NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l2486_248653

/-- Given two jars of alcohol-water mixtures with volumes V and 2V, and ratios p:1 and q:1 respectively,
    the ratio of alcohol to water in the resulting mixture is (p(q+1) + 2p + 2q) : (q+1 + 2p + 2) -/
theorem alcohol_water_mixture_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let first_jar_alcohol := (p / (p + 1)) * V
  let first_jar_water := (1 / (p + 1)) * V
  let second_jar_alcohol := (2 * q / (q + 1)) * V
  let second_jar_water := (2 / (q + 1)) * V
  let total_alcohol := first_jar_alcohol + second_jar_alcohol
  let total_water := first_jar_water + second_jar_water
  total_alcohol / total_water = (p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l2486_248653


namespace NUMINAMATH_CALUDE_involutive_function_property_l2486_248615

/-- A function f that is its own inverse -/
def InvolutiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The main theorem -/
theorem involutive_function_property
  (a b c d : ℝ)
  (hb : b ≠ 0)
  (hd : d ≠ 0)
  (h_c_a : 3 * c^2 = 2 * a^2)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (2*a*x + b) / (3*c*x + d))
  (h_involutive : InvolutiveFunction f) :
  2*a + 3*d = -4*a := by
sorry

end NUMINAMATH_CALUDE_involutive_function_property_l2486_248615


namespace NUMINAMATH_CALUDE_candy_theorem_l2486_248659

/-- The total number of candy pieces caught by four friends -/
def total_candy (tabitha stan julie carlos : ℕ) : ℕ :=
  tabitha + stan + julie + carlos

/-- Theorem: Given the conditions, the friends caught 72 pieces of candy in total -/
theorem candy_theorem (tabitha stan julie carlos : ℕ) 
  (h1 : tabitha = 22)
  (h2 : stan = 13)
  (h3 : julie = tabitha / 2)
  (h4 : carlos = 2 * stan) :
  total_candy tabitha stan julie carlos = 72 := by
  sorry

end NUMINAMATH_CALUDE_candy_theorem_l2486_248659


namespace NUMINAMATH_CALUDE_fruit_bowl_oranges_l2486_248618

theorem fruit_bowl_oranges (bananas apples oranges : ℕ) : 
  bananas = 2 → 
  apples = 2 * bananas → 
  bananas + apples + oranges = 12 → 
  oranges = 6 := by
sorry

end NUMINAMATH_CALUDE_fruit_bowl_oranges_l2486_248618


namespace NUMINAMATH_CALUDE_trapezoid_area_is_15_l2486_248631

/-- A trapezoid bounded by y = 2x, y = 8, y = 2, and the y-axis -/
structure Trapezoid where
  /-- The line y = 2x -/
  line_1 : ℝ → ℝ := λ x => 2 * x
  /-- The line y = 8 -/
  line_2 : ℝ → ℝ := λ _ => 8
  /-- The line y = 2 -/
  line_3 : ℝ → ℝ := λ _ => 2
  /-- The y-axis (x = 0) -/
  y_axis : ℝ → ℝ := λ y => 0

/-- The area of the trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  15

/-- Theorem stating that the area of the given trapezoid is 15 square units -/
theorem trapezoid_area_is_15 (t : Trapezoid) : trapezoidArea t = 15 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_15_l2486_248631


namespace NUMINAMATH_CALUDE_sum_simplification_l2486_248674

theorem sum_simplification : -1^2022 + (-1)^2023 + 1^2024 - 1^2025 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l2486_248674


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2486_248634

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2486_248634
