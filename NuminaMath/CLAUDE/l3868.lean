import Mathlib

namespace NUMINAMATH_CALUDE_min_groups_for_athletes_l3868_386896

theorem min_groups_for_athletes (total_athletes : ℕ) (max_group_size : ℕ) (h1 : total_athletes = 30) (h2 : max_group_size = 12) : 
  ∃ (num_groups : ℕ), 
    num_groups ≥ 1 ∧ 
    num_groups ≤ total_athletes ∧
    ∃ (group_size : ℕ), 
      group_size > 0 ∧
      group_size ≤ max_group_size ∧
      total_athletes = num_groups * group_size ∧
      ∀ (n : ℕ), n < num_groups → 
        ¬∃ (g : ℕ), g > 0 ∧ g ≤ max_group_size ∧ total_athletes = n * g :=
by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_athletes_l3868_386896


namespace NUMINAMATH_CALUDE_zoe_gre_exam_month_l3868_386898

-- Define the months as an enumeration
inductive Month
  | January | February | March | April | May | June | July | August | September | October | November | December

-- Define a function to add months
def addMonths (start : Month) (n : Nat) : Month :=
  match n with
  | 0 => start
  | Nat.succ m => addMonths (match start with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) m

-- Theorem statement
theorem zoe_gre_exam_month :
  addMonths Month.April 2 = Month.June :=
by sorry

end NUMINAMATH_CALUDE_zoe_gre_exam_month_l3868_386898


namespace NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l3868_386854

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_36_pow_12 : tens_digit (36^12) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l3868_386854


namespace NUMINAMATH_CALUDE_equation_solution_l3868_386809

theorem equation_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : 2*x + y ≠ 0) 
  (h3 : (x + y) / x = y / (2*x + y)) : x = -y/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3868_386809


namespace NUMINAMATH_CALUDE_boys_age_l3868_386826

theorem boys_age (current_age : ℕ) : 
  (current_age = 2 * (current_age - 5)) → current_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_l3868_386826


namespace NUMINAMATH_CALUDE_parabola_vertex_l3868_386847

/-- Given a parabola y = -x^2 + ax + b ≤ 0 with roots at x = -4 and x = 6,
    prove that its vertex is at (1, 25). -/
theorem parabola_vertex (a b : ℝ) :
  (∀ x, -x^2 + a*x + b ≤ 0 ↔ x ∈ Set.Ici 6 ∪ Set.Iic (-4)) →
  ∃ k, -1^2 + a*1 + b = k ∧ ∀ x, -x^2 + a*x + b ≤ k :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3868_386847


namespace NUMINAMATH_CALUDE_problem_statement_l3868_386840

theorem problem_statement : (2 * Real.sqrt 2 - 1)^2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5) = 5 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3868_386840


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3868_386831

theorem sqrt_equation_solution (x : ℝ) (h : x > 0) : 18 / Real.sqrt x = 2 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3868_386831


namespace NUMINAMATH_CALUDE_ellipse_minimum_area_l3868_386841

/-- An ellipse containing two specific circles has a minimum area -/
theorem ellipse_minimum_area (a b : ℝ) (h_positive_a : a > 0) (h_positive_b : b > 0) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) →
  a * b ≥ 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_minimum_area_l3868_386841


namespace NUMINAMATH_CALUDE_distance_between_trees_problem_l3868_386810

/-- The distance between consecutive trees in a yard -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  (yard_length : ℚ) / (num_trees - 1 : ℚ)

/-- Theorem: The distance between consecutive trees in a 400-meter yard with 26 trees is 16 meters -/
theorem distance_between_trees_problem :
  distance_between_trees 400 26 = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_problem_l3868_386810


namespace NUMINAMATH_CALUDE_max_y_over_x_l3868_386886

theorem max_y_over_x (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 ∧ ∃ (x₀ y₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ 1 / x₀ + 2 * y₀ = 3 ∧ y₀ / x₀ = 9 / 8 :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3868_386886


namespace NUMINAMATH_CALUDE_jane_max_tickets_l3868_386872

/-- Calculates the maximum number of tickets that can be bought given a budget and pricing structure -/
def maxTickets (budget : ℕ) (normalPrice discountPrice : ℕ) (discountThreshold : ℕ) : ℕ :=
  let fullPriceTickets := min discountThreshold (budget / normalPrice)
  let remainingBudget := budget - fullPriceTickets * normalPrice
  fullPriceTickets + remainingBudget / discountPrice

/-- The maximum number of tickets Jane can buy is 11 -/
theorem jane_max_tickets :
  maxTickets 150 15 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l3868_386872


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l3868_386819

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x + 1

theorem tangent_line_parallel_point (P₀ : ℝ × ℝ) : 
  P₀.1 = 1 ∧ P₀.2 = f P₀.1 ∧ f' P₀.1 = 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l3868_386819


namespace NUMINAMATH_CALUDE_cubic_polynomial_r_value_l3868_386858

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  p : Int
  q : Int
  r : Int

/-- The property that all roots of a cubic polynomial are negative integers -/
def hasAllNegativeIntegerRoots (g : CubicPolynomial) : Prop := sorry

/-- Theorem: For a cubic polynomial g(x) = x^3 + px^2 + qx + r with all roots being negative integers
    and p + q + r = 100, the value of r must be 0 -/
theorem cubic_polynomial_r_value (g : CubicPolynomial)
    (h1 : hasAllNegativeIntegerRoots g)
    (h2 : g.p + g.q + g.r = 100) :
    g.r = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_r_value_l3868_386858


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3868_386822

def n : ℕ := 81 * 83 * 85 * 87 + 89

theorem distinct_prime_factors_count : Nat.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3868_386822


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1140_l3868_386850

theorem sum_of_largest_and_smallest_prime_factors_of_1140 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1140 ∧ largest ∣ 1140 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1140 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1140 → p ≥ smallest) ∧
    smallest + largest = 21 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1140_l3868_386850


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l3868_386867

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 = (1 - a^3) / (1 - a) := by sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l3868_386867


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l3868_386895

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 4

/-- Theorem: The number of intersection points between the parabola y = -x^2 + 4x - 4 
    and the coordinate axes is equal to 2 -/
theorem parabola_intersection_points : 
  (∃! x : ℝ, f x = 0) ∧ (∃! y : ℝ, f 0 = y) ∧ 
  (∀ x y : ℝ, (x = 0 ∨ y = 0) → (y = f x) → (x = 0 ∧ y = f 0) ∨ (y = 0 ∧ f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l3868_386895


namespace NUMINAMATH_CALUDE_non_matching_pairings_eq_twenty_l3868_386873

/-- The number of colors available for bowls and glasses -/
def num_colors : ℕ := 5

/-- The number of non-matching pairings between bowls and glasses -/
def non_matching_pairings : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating that the number of non-matching pairings is 20 -/
theorem non_matching_pairings_eq_twenty : non_matching_pairings = 20 := by
  sorry

end NUMINAMATH_CALUDE_non_matching_pairings_eq_twenty_l3868_386873


namespace NUMINAMATH_CALUDE_bag_draw_comparison_l3868_386838

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- Random variable for drawing with replacement -/
def xi₁ (b : Bag) : ℕ → ℝ := sorry

/-- Random variable for drawing without replacement -/
def xi₂ (b : Bag) : ℕ → ℝ := sorry

/-- Expected value of a random variable -/
def expectation (X : ℕ → ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℕ → ℝ) : ℝ := sorry

/-- Theorem about expected values and variances of xi₁ and xi₂ -/
theorem bag_draw_comparison (b : Bag) (h : b.red = 1 ∧ b.black = 2) : 
  expectation (xi₁ b) = expectation (xi₂ b) ∧ 
  variance (xi₁ b) > variance (xi₂ b) := by sorry

end NUMINAMATH_CALUDE_bag_draw_comparison_l3868_386838


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_greater_than_one_l3868_386835

/-- Given functions f and g, prove that if for any x₁ in [-1, 2], 
    there exists an x₂ in [0, 2] such that f(x₁) > g(x₂), then a > 1 -/
theorem function_inequality_implies_a_greater_than_one (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (0 : ℝ) 2, x₁^2 > 2^x₂ - a) → 
  a > 1 := by
  sorry

#check function_inequality_implies_a_greater_than_one

end NUMINAMATH_CALUDE_function_inequality_implies_a_greater_than_one_l3868_386835


namespace NUMINAMATH_CALUDE_sum_of_cubes_square_not_prime_product_l3868_386803

theorem sum_of_cubes_square_not_prime_product (a b : ℕ+) (n : ℕ) :
  a^3 + b^3 = n^2 →
  ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ a + b = p * q :=
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_square_not_prime_product_l3868_386803


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l3868_386813

/-- The area of the shaded region in a square with quarter circles at each corner -/
theorem shaded_area_square_with_quarter_circles 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 15) 
  (h2 : circle_radius = 5) : 
  square_side ^ 2 - 4 * (π / 4 * circle_radius ^ 2) = 225 - 25 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l3868_386813


namespace NUMINAMATH_CALUDE_largest_visible_sum_l3868_386891

/-- Represents a standard die with opposite faces summing to 7 -/
structure Die where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents a 3x3x3 cube assembled from 27 dice -/
structure Cube where
  dice : Fin 3 → Fin 3 → Fin 3 → Die

/-- Calculates the sum of visible values on the 6 faces of the cube -/
def visibleSum (c : Cube) : Nat :=
  sorry

/-- States that the largest possible sum of visible values is 288 -/
theorem largest_visible_sum (c : Cube) : 
  visibleSum c ≤ 288 ∧ ∃ c' : Cube, visibleSum c' = 288 :=
sorry

end NUMINAMATH_CALUDE_largest_visible_sum_l3868_386891


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_with_equal_digit_pairs_l3868_386856

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_two_pairs_of_equal_digits (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * a + 10 * b + b

theorem four_digit_perfect_square_with_equal_digit_pairs :
  is_four_digit 7744 ∧ is_perfect_square 7744 ∧ has_two_pairs_of_equal_digits 7744 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_with_equal_digit_pairs_l3868_386856


namespace NUMINAMATH_CALUDE_no_solution_condition_l3868_386830

theorem no_solution_condition (m : ℚ) : 
  (∀ x : ℚ, x ≠ 5 ∧ x ≠ -5 → 1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25)) ↔ 
  m = -1 ∨ m = 5 ∨ m = -5/11 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3868_386830


namespace NUMINAMATH_CALUDE_negation_of_zero_product_l3868_386865

theorem negation_of_zero_product (a b : ℝ) :
  ¬(a * b = 0 → a = 0 ∨ b = 0) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_zero_product_l3868_386865


namespace NUMINAMATH_CALUDE_notebooks_left_l3868_386857

theorem notebooks_left (total : ℕ) (h1 : total = 28) : 
  total - (total / 4 + total * 3 / 7) = 9 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_left_l3868_386857


namespace NUMINAMATH_CALUDE_least_prime_factor_of_9_5_plus_9_4_l3868_386897

theorem least_prime_factor_of_9_5_plus_9_4 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (9^5 + 9^4) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (9^5 + 9^4) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_9_5_plus_9_4_l3868_386897


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3868_386874

-- Define a random variable following a normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
def P (ξ : normal_distribution 3 σ) (pred : ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_symmetry (σ : ℝ) (c : ℝ) :
  (∀ (ξ : normal_distribution 3 σ), P ξ (λ x => x > c + 1) = P ξ (λ x => x < c - 1)) →
  c = 3 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3868_386874


namespace NUMINAMATH_CALUDE_unique_perfect_square_divisor_l3868_386846

theorem unique_perfect_square_divisor : ∃! (n : ℕ), n > 0 ∧ ∃ (k : ℕ), (n^3 - 1989) / n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_divisor_l3868_386846


namespace NUMINAMATH_CALUDE_internally_tangent_circles_distance_l3868_386899

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

theorem internally_tangent_circles_distance
  (r₁ r₂ d : ℝ)
  (h₁ : r₁ = 3)
  (h₂ : r₂ = 6)
  (h₃ : internally_tangent r₁ r₂ d) :
  d = 3 :=
sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_distance_l3868_386899


namespace NUMINAMATH_CALUDE_no_solution_for_equal_ratios_l3868_386862

theorem no_solution_for_equal_ratios :
  ¬∃ (x : ℝ), (4 + x) / (5 + x) = (1 + x) / (2 + x) := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_equal_ratios_l3868_386862


namespace NUMINAMATH_CALUDE_two_solutions_l3868_386825

/-- The quadratic equation with absolute value term -/
def quadratic_abs_equation (x : ℝ) : Prop :=
  x^2 - |x| - 6 = 0

/-- The number of distinct real solutions to the equation -/
def num_solutions : ℕ := 2

/-- Theorem stating that the equation has exactly two distinct real solutions -/
theorem two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ 
  quadratic_abs_equation a ∧ 
  quadratic_abs_equation b ∧
  (∀ x : ℝ, quadratic_abs_equation x → x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l3868_386825


namespace NUMINAMATH_CALUDE_three_circles_middle_radius_l3868_386864

/-- Configuration of three circles with two common tangent lines -/
structure ThreeCirclesConfig where
  r_large : ℝ  -- radius of the largest circle
  r_small : ℝ  -- radius of the smallest circle
  r_middle : ℝ  -- radius of the middle circle
  tangent_lines : ℕ  -- number of common tangent lines

/-- Theorem: In a configuration of three circles with two common tangent lines,
    if the radius of the largest circle is 18 and the radius of the smallest circle is 8,
    then the radius of the middle circle is 12. -/
theorem three_circles_middle_radius 
  (config : ThreeCirclesConfig) 
  (h1 : config.r_large = 18) 
  (h2 : config.r_small = 8) 
  (h3 : config.tangent_lines = 2) : 
  config.r_middle = 12 := by
  sorry

end NUMINAMATH_CALUDE_three_circles_middle_radius_l3868_386864


namespace NUMINAMATH_CALUDE_smallest_n_terminating_with_2_l3868_386889

def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_2 (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 2 + 10 * m

theorem smallest_n_terminating_with_2 :
  ∃ n : ℕ+, 
    is_terminating_decimal n ∧ 
    contains_digit_2 n.val ∧ 
    (∀ m : ℕ+, m < n → ¬(is_terminating_decimal m ∧ contains_digit_2 m.val)) ∧
    n = 2 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_terminating_with_2_l3868_386889


namespace NUMINAMATH_CALUDE_min_cars_theorem_l3868_386844

/-- Calculates the minimum number of cars needed for a family where each car must rest one day a week and all adults want to drive daily. -/
def min_cars_needed (num_adults : ℕ) : ℕ :=
  if num_adults ≤ 6 then
    num_adults + 1
  else
    (num_adults * 7 + 5) / 6

theorem min_cars_theorem (num_adults : ℕ) :
  (num_adults = 5 → min_cars_needed num_adults = 6) ∧
  (num_adults = 8 → min_cars_needed num_adults = 10) :=
by sorry

#eval min_cars_needed 5  -- Should output 6
#eval min_cars_needed 8  -- Should output 10

end NUMINAMATH_CALUDE_min_cars_theorem_l3868_386844


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l3868_386804

theorem sum_of_reciprocals_of_roots (p₁ p₂ : ℝ) : 
  p₁^2 - 17*p₁ + 8 = 0 → 
  p₂^2 - 17*p₂ + 8 = 0 → 
  p₁ ≠ p₂ →
  1/p₁ + 1/p₂ = 17/8 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l3868_386804


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l3868_386882

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 :=
sorry

theorem min_value_attained (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) = 1 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_attained_l3868_386882


namespace NUMINAMATH_CALUDE_vegetables_used_l3868_386880

def initial_beef : ℝ := 4
def unused_beef : ℝ := 1
def veg_to_beef_ratio : ℝ := 2

theorem vegetables_used : 
  let beef_used := initial_beef - unused_beef
  let vegetables_used := beef_used * veg_to_beef_ratio
  vegetables_used = 6 := by sorry

end NUMINAMATH_CALUDE_vegetables_used_l3868_386880


namespace NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l3868_386861

theorem quadratic_coefficient_of_equation : ∃ (a b c d e f : ℝ),
  (∀ x, a * x^2 + b * x + c = d * x^2 + e * x + f) →
  (a = 5 ∧ b = -1 ∧ c = -3 ∧ d = 1 ∧ e = 1 ∧ f = -3) →
  (a - d = 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l3868_386861


namespace NUMINAMATH_CALUDE_work_hours_per_day_l3868_386863

theorem work_hours_per_day 
  (total_hours : ℝ) 
  (total_days : ℝ) 
  (h1 : total_hours = 8.0) 
  (h2 : total_days = 4.0) 
  (h3 : total_days > 0) : 
  total_hours / total_days = 2.0 := by
sorry

end NUMINAMATH_CALUDE_work_hours_per_day_l3868_386863


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l3868_386869

theorem mean_equality_implies_z (z : ℚ) : 
  (4 + 16 + 20) / 3 = (2 * 4 + z) / 2 → z = 56 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l3868_386869


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_l3868_386834

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def distinct_perfect_square_sum (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 100

theorem unique_perfect_square_sum : 
  ∃! (abc : ℕ × ℕ × ℕ), distinct_perfect_square_sum abc.1 abc.2.1 abc.2.2 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_l3868_386834


namespace NUMINAMATH_CALUDE_ellipse_existence_and_uniqueness_l3868_386817

/-- A structure representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A structure representing an ellipse in a 2D plane -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ
  rotation : ℝ

/-- Function to check if two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Function to check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  sorry

/-- Function to check if an ellipse has its axes on given lines -/
def ellipseAxesOnLines (e : Ellipse) (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating the existence and uniqueness of ellipses -/
theorem ellipse_existence_and_uniqueness 
  (l1 l2 : Line) (p1 p2 : Point) 
  (h_perp : arePerpendicular l1 l2) :
  (p1 ≠ p2 → ∃! e : Ellipse, pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ ellipseAxesOnLines e l1 l2) ∧
  (p1 = p2 → ∃ e : Ellipse, pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ ellipseAxesOnLines e l1 l2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_existence_and_uniqueness_l3868_386817


namespace NUMINAMATH_CALUDE_fraction_positivity_l3868_386808

theorem fraction_positivity (x : ℝ) : (x + 2) / ((x - 3)^3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_positivity_l3868_386808


namespace NUMINAMATH_CALUDE_shopkeeper_loss_l3868_386885

theorem shopkeeper_loss (X : ℝ) (h : X > 0) : 
  let intended_sale_price := 1.1 * X
  let remaining_goods_value := 0.4 * X
  let actual_sale_price := 1.1 * remaining_goods_value
  let loss := X - actual_sale_price
  let percentage_loss := (loss / X) * 100
  percentage_loss = 56 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_l3868_386885


namespace NUMINAMATH_CALUDE_pension_program_participation_rate_l3868_386892

structure Shift where
  members : ℕ
  participation_rate : ℚ

def company_x : List Shift := [
  { members := 60, participation_rate := 1/5 },
  { members := 50, participation_rate := 2/5 },
  { members := 40, participation_rate := 1/10 }
]

theorem pension_program_participation_rate :
  let total_workers := (company_x.map (λ s => s.members)).sum
  let participating_workers := (company_x.map (λ s => (s.members : ℚ) * s.participation_rate)).sum
  participating_workers / total_workers = 6/25 := by
sorry

end NUMINAMATH_CALUDE_pension_program_participation_rate_l3868_386892


namespace NUMINAMATH_CALUDE_rain_period_end_time_l3868_386845

def start_time : ℕ := 8  -- 8 am
def rain_duration : ℕ := 4
def no_rain_duration : ℕ := 5

def total_duration : ℕ := rain_duration + no_rain_duration

def end_time : ℕ := start_time + total_duration

theorem rain_period_end_time :
  end_time = 17  -- 5 pm in 24-hour format
:= by sorry

end NUMINAMATH_CALUDE_rain_period_end_time_l3868_386845


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3868_386871

theorem nested_fraction_equality : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3868_386871


namespace NUMINAMATH_CALUDE_rational_times_sqrt_two_rational_implies_zero_l3868_386884

theorem rational_times_sqrt_two_rational_implies_zero (x : ℚ) :
  (∃ (y : ℚ), y = x * Real.sqrt 2) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_times_sqrt_two_rational_implies_zero_l3868_386884


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3868_386852

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (a ≠ 5 ∧ b ≠ -5) ↔ (a + b ≠ 0) → False :=
by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3868_386852


namespace NUMINAMATH_CALUDE_triangle_side_length_l3868_386881

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = Real.sqrt 3) 
  (h2 : B = π / 4) 
  (h3 : A = π / 3) 
  (h4 : C = π - A - B) 
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h6 : 0 < A ∧ A < π) 
  (h7 : 0 < B ∧ B < π) 
  (h8 : 0 < C ∧ C < π) 
  (h9 : a / Real.sin A = b / Real.sin B) 
  (h10 : a / Real.sin A = c / Real.sin C) 
  (h11 : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) : 
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3868_386881


namespace NUMINAMATH_CALUDE_smallest_multiple_of_90_with_128_divisors_l3868_386842

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := sorry

-- Define the property of being a multiple of 90
def is_multiple_of_90 (n : ℕ) : Prop := ∃ k : ℕ, n = 90 * k

-- Define the main theorem
theorem smallest_multiple_of_90_with_128_divisors :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(is_multiple_of_90 m ∧ num_divisors m = 128)) ∧
    is_multiple_of_90 n ∧
    num_divisors n = 128 ∧
    n / 90 = 1728 := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_90_with_128_divisors_l3868_386842


namespace NUMINAMATH_CALUDE_total_students_is_184_l3868_386811

/-- Represents the number of students that can be transported in one car for a school --/
structure CarCapacity where
  capacity : ℕ

/-- Represents a school participating in the competition --/
structure School where
  students : ℕ
  carCapacity : CarCapacity

/-- Represents the state of both schools at a given point --/
structure CompetitionState where
  school1 : School
  school2 : School

/-- Checks if the given state satisfies the initial conditions --/
def initialConditionsSatisfied (state : CompetitionState) : Prop :=
  state.school1.students = state.school2.students ∧
  state.school1.carCapacity.capacity = 15 ∧
  state.school2.carCapacity.capacity = 13 ∧
  (state.school2.students + state.school2.carCapacity.capacity - 1) / state.school2.carCapacity.capacity =
    (state.school1.students / state.school1.carCapacity.capacity) + 1

/-- Checks if the given state satisfies the conditions after adding one student to each school --/
def middleConditionsSatisfied (state : CompetitionState) : Prop :=
  (state.school1.students + 1) / state.school1.carCapacity.capacity =
  (state.school2.students + 1) / state.school2.carCapacity.capacity

/-- Checks if the given state satisfies the final conditions --/
def finalConditionsSatisfied (state : CompetitionState) : Prop :=
  ((state.school1.students + 2) / state.school1.carCapacity.capacity) + 1 =
  (state.school2.students + 2) / state.school2.carCapacity.capacity

/-- The main theorem stating that under the given conditions, the total number of students is 184 --/
theorem total_students_is_184 (state : CompetitionState) :
  initialConditionsSatisfied state →
  middleConditionsSatisfied state →
  finalConditionsSatisfied state →
  state.school1.students + state.school2.students + 4 = 184 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_is_184_l3868_386811


namespace NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l3868_386853

theorem triangle_angle_sine_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  Real.sin (α/2 + β) + Real.sin (β/2 + γ) + Real.sin (γ/2 + α) > 
  Real.sin α + Real.sin β + Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l3868_386853


namespace NUMINAMATH_CALUDE_unique_solution_l3868_386843

/-- Represents the ages of the grandchildren --/
structure GrandchildrenAges where
  martinka : ℕ
  tomasek : ℕ
  jaromir : ℕ
  kacka : ℕ
  ida : ℕ
  verka : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.martinka = ages.tomasek + 8 ∧
  ages.verka = ages.ida + 7 ∧
  ages.martinka = ages.jaromir + 1 ∧
  ages.kacka = ages.tomasek + 11 ∧
  ages.jaromir = ages.ida + 4 ∧
  ages.tomasek + ages.jaromir = 13

/-- The theorem stating that there is a unique solution satisfying all conditions --/
theorem unique_solution : ∃! ages : GrandchildrenAges, satisfiesConditions ages ∧
  ages.martinka = 11 ∧
  ages.tomasek = 3 ∧
  ages.jaromir = 10 ∧
  ages.kacka = 14 ∧
  ages.ida = 6 ∧
  ages.verka = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3868_386843


namespace NUMINAMATH_CALUDE_furniture_assembly_time_l3868_386800

/-- Given the number of chairs and tables, and the time spent on each piece,
    calculate the total time taken to assemble all furniture. -/
theorem furniture_assembly_time 
  (num_chairs : ℕ) 
  (num_tables : ℕ) 
  (time_per_piece : ℕ) 
  (h1 : num_chairs = 4) 
  (h2 : num_tables = 2) 
  (h3 : time_per_piece = 8) : 
  (num_chairs + num_tables) * time_per_piece = 48 := by
  sorry

end NUMINAMATH_CALUDE_furniture_assembly_time_l3868_386800


namespace NUMINAMATH_CALUDE_shift_repeating_segment_2011th_digit_6_l3868_386877

/-- Represents a repeating decimal with an initial non-repeating part and a repeating segment. -/
structure RepeatingDecimal where
  initial : ℚ
  repeating : List ℕ

/-- Shifts the repeating segment of a repeating decimal. -/
def shiftRepeatingSegment (d : RepeatingDecimal) (n : ℕ) : RepeatingDecimal :=
  sorry

/-- Gets the nth digit after the decimal point in a repeating decimal. -/
def nthDigitAfterDecimal (d : RepeatingDecimal) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem about shifting the repeating segment. -/
theorem shift_repeating_segment_2011th_digit_6 (d : RepeatingDecimal) :
  d.initial = 0.1 ∧ d.repeating = [2, 3, 4, 5, 6, 7, 8] →
  ∃ (k : ℕ), 
    let d' := shiftRepeatingSegment d k
    nthDigitAfterDecimal d' 2011 = 6 ∧
    d'.initial = 0.1 ∧ d'.repeating = [2, 3, 4, 5, 6, 7, 8] :=
  sorry

end NUMINAMATH_CALUDE_shift_repeating_segment_2011th_digit_6_l3868_386877


namespace NUMINAMATH_CALUDE_smallest_with_eight_odd_sixteen_even_divisors_l3868_386806

/-- Count of positive odd integer divisors of a number -/
def countOddDivisors (n : ℕ) : ℕ := sorry

/-- Count of positive even integer divisors of a number -/
def countEvenDivisors (n : ℕ) : ℕ := sorry

/-- Proposition: 3000 is the smallest positive integer with 8 odd and 16 even divisors -/
theorem smallest_with_eight_odd_sixteen_even_divisors :
  (∀ m : ℕ, m > 0 ∧ m < 3000 → 
    countOddDivisors m ≠ 8 ∨ countEvenDivisors m ≠ 16) ∧
  countOddDivisors 3000 = 8 ∧ 
  countEvenDivisors 3000 = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_eight_odd_sixteen_even_divisors_l3868_386806


namespace NUMINAMATH_CALUDE_inequality_region_is_triangle_l3868_386870

/-- The region described by a system of inequalities -/
def InequalityRegion (x y : ℝ) : Prop :=
  x + y - 1 ≤ 0 ∧ -x + y - 1 ≤ 0 ∧ y ≥ -1

/-- The triangle with vertices (0, 1), (2, -1), and (-2, -1) -/
def Triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = -1) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    ((x = 2*t - 2 ∧ y = -1) ∨
     (x = 2*t ∧ y = -t) ∨
     (x = -2*t ∧ y = t)))

theorem inequality_region_is_triangle :
  ∀ x y : ℝ, InequalityRegion x y ↔ Triangle x y :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_is_triangle_l3868_386870


namespace NUMINAMATH_CALUDE_honey_production_l3868_386807

theorem honey_production (bees : ℕ) (days : ℕ) (honey_per_bee : ℝ) :
  bees = 70 → days = 70 → honey_per_bee = 1 →
  bees * honey_per_bee = 70 := by
sorry

end NUMINAMATH_CALUDE_honey_production_l3868_386807


namespace NUMINAMATH_CALUDE_sqrt_4_4_times_9_2_l3868_386839

theorem sqrt_4_4_times_9_2 : Real.sqrt (4^4 * 9^2) = 144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_4_times_9_2_l3868_386839


namespace NUMINAMATH_CALUDE_sum_divisors_2_3_power_l3868_386805

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j is 360, then i + j = 6 -/
theorem sum_divisors_2_3_power (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 360 → i + j = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisors_2_3_power_l3868_386805


namespace NUMINAMATH_CALUDE_nice_set_property_l3868_386814

def nice (P : Set (ℤ × ℤ)) : Prop :=
  (∀ a b, (a, b) ∈ P → (b, a) ∈ P) ∧
  (∀ a b c d, (a, b) ∈ P → (c, d) ∈ P → (a + c, b - d) ∈ P)

theorem nice_set_property (p q : ℤ) (h1 : Nat.gcd p.natAbs q.natAbs = 1) 
  (h2 : p % 2 ≠ q % 2) :
  ∀ (P : Set (ℤ × ℤ)), nice P → (p, q) ∈ P → P = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_nice_set_property_l3868_386814


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l3868_386837

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β := by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l3868_386837


namespace NUMINAMATH_CALUDE_max_sum_of_four_digits_l3868_386893

def is_valid_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem max_sum_of_four_digits :
  ∀ A B C D : ℕ,
    is_valid_digit A → is_valid_digit B → is_valid_digit C → is_valid_digit D →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B) + (C + D) ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_four_digits_l3868_386893


namespace NUMINAMATH_CALUDE_average_speed_last_segment_l3868_386868

theorem average_speed_last_segment (total_distance : ℝ) (total_time : ℝ) 
  (speed_first : ℝ) (speed_second : ℝ) :
  total_distance = 108 ∧ 
  total_time = 1.5 ∧ 
  speed_first = 70 ∧ 
  speed_second = 60 → 
  ∃ speed_last : ℝ, 
    speed_last = 86 ∧ 
    (speed_first + speed_second + speed_last) / 3 = total_distance / total_time :=
by sorry

end NUMINAMATH_CALUDE_average_speed_last_segment_l3868_386868


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3868_386816

theorem circle_area_ratio (R S : Real) (hR : R > 0) (hS : S > 0) 
  (h_diameter : R = 0.4 * S) : 
  (π * R^2) / (π * S^2) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3868_386816


namespace NUMINAMATH_CALUDE_puppies_given_away_l3868_386821

def initial_puppies : ℕ := 7
def remaining_puppies : ℕ := 2

theorem puppies_given_away : initial_puppies - remaining_puppies = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_away_l3868_386821


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_m_eq_6_l3868_386851

/-- A function f is monotonically decreasing on an interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

/-- The function f(x) = x^3 - mx^2 + 2m^2 - 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - m*x^2 + 2*m^2 - 5

theorem monotone_decreasing_implies_m_eq_6 :
  ∀ m : ℝ, MonotonicallyDecreasing (f m) (-9) 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_m_eq_6_l3868_386851


namespace NUMINAMATH_CALUDE_short_sleeve_shirts_count_l3868_386876

theorem short_sleeve_shirts_count :
  ∀ (total_shirts long_sleeve_shirts washed_shirts unwashed_shirts : ℕ),
    long_sleeve_shirts = 47 →
    washed_shirts = 20 →
    unwashed_shirts = 66 →
    total_shirts = washed_shirts + unwashed_shirts →
    total_shirts = (total_shirts - long_sleeve_shirts) + long_sleeve_shirts →
    (total_shirts - long_sleeve_shirts) = 39 :=
by sorry

end NUMINAMATH_CALUDE_short_sleeve_shirts_count_l3868_386876


namespace NUMINAMATH_CALUDE_soccer_ball_contribution_l3868_386824

theorem soccer_ball_contribution (k l m : ℝ) : 
  k ≥ 0 → l ≥ 0 → m ≥ 0 →
  k + l + m = 6 →
  2 * k ≤ l + m →
  2 * l ≤ k + m →
  2 * m ≤ k + l →
  k = 2 ∧ l = 2 ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_contribution_l3868_386824


namespace NUMINAMATH_CALUDE_is_circle_center_l3868_386828

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, 2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l3868_386828


namespace NUMINAMATH_CALUDE_trapezoid_shorter_lateral_l3868_386801

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  longer_lateral : ℝ
  base_difference : ℝ
  right_angle_intersection : Bool

/-- 
  Theorem: In a trapezoid where the lines containing the lateral sides intersect at a right angle,
  if the longer lateral side is 8 and the difference between the bases is 10,
  then the shorter lateral side is 6.
-/
theorem trapezoid_shorter_lateral 
  (t : Trapezoid) 
  (h1 : t.longer_lateral = 8) 
  (h2 : t.base_difference = 10) 
  (h3 : t.right_angle_intersection = true) : 
  ∃ (shorter_lateral : ℝ), shorter_lateral = 6 := by
  sorry

#check trapezoid_shorter_lateral

end NUMINAMATH_CALUDE_trapezoid_shorter_lateral_l3868_386801


namespace NUMINAMATH_CALUDE_ball_bearing_sale_price_l3868_386866

/-- The sale price of ball bearings that satisfies the given conditions -/
def sale_price : ℝ := 0.75

theorem ball_bearing_sale_price :
  let num_machines : ℕ := 10
  let bearings_per_machine : ℕ := 30
  let normal_price : ℝ := 1
  let bulk_discount : ℝ := 0.2
  let total_savings : ℝ := 120
  
  let total_bearings : ℕ := num_machines * bearings_per_machine
  let normal_total_cost : ℝ := total_bearings * normal_price
  let sale_total_cost : ℝ := total_bearings * sale_price * (1 - bulk_discount)
  
  normal_total_cost - sale_total_cost = total_savings :=
by sorry

end NUMINAMATH_CALUDE_ball_bearing_sale_price_l3868_386866


namespace NUMINAMATH_CALUDE_train_length_is_300_l3868_386833

/-- The length of the train in meters -/
def train_length : ℝ := 300

/-- The time (in seconds) it takes for the train to cross the platform -/
def platform_crossing_time : ℝ := 39

/-- The time (in seconds) it takes for the train to cross a signal pole -/
def pole_crossing_time : ℝ := 12

/-- The length of the platform in meters -/
def platform_length : ℝ := 675

/-- Theorem stating that the train length is 300 meters given the conditions -/
theorem train_length_is_300 :
  train_length = 300 ∧
  train_length + platform_length = (train_length / pole_crossing_time) * platform_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_is_300_l3868_386833


namespace NUMINAMATH_CALUDE_detergent_volume_in_new_solution_l3868_386883

/-- Represents the components of a cleaning solution -/
inductive Component
| Bleach
| Detergent
| Water

/-- Represents the ratio of components in a solution -/
def Ratio := Component → ℚ

def original_ratio : Ratio :=
  fun c => match c with
  | Component.Bleach => 4
  | Component.Detergent => 40
  | Component.Water => 100

def new_ratio : Ratio :=
  fun c => match c with
  | Component.Bleach => 3 * (original_ratio Component.Bleach)
  | Component.Detergent => (1/2) * (original_ratio Component.Detergent)
  | Component.Water => original_ratio Component.Water

def water_volume : ℚ := 300

theorem detergent_volume_in_new_solution :
  (new_ratio Component.Detergent / new_ratio Component.Water) * water_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_detergent_volume_in_new_solution_l3868_386883


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3868_386875

open Real

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → a = 3 → c = Real.sqrt 6 → 
  (sin C = sin (π/4) ∨ sin C = sin (3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3868_386875


namespace NUMINAMATH_CALUDE_negative_seven_minus_seven_l3868_386894

theorem negative_seven_minus_seven : (-7) - 7 = -14 := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_minus_seven_l3868_386894


namespace NUMINAMATH_CALUDE_sum_of_prime_factor_exponents_l3868_386890

/-- The sum of exponents in the given expression of prime factors -/
def sum_of_exponents : ℕ :=
  9 + 5 + 7 + 4 + 6 + 3 + 5 + 2

/-- The theorem states that the sum of exponents in the given expression equals 41 -/
theorem sum_of_prime_factor_exponents : sum_of_exponents = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factor_exponents_l3868_386890


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3868_386836

theorem sqrt_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * Real.sqrt (50 * x) = 60 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3868_386836


namespace NUMINAMATH_CALUDE_four_position_assignments_l3868_386878

def number_of_assignments (n : ℕ) : ℕ := n.factorial

theorem four_position_assignments :
  number_of_assignments 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_position_assignments_l3868_386878


namespace NUMINAMATH_CALUDE_rainfall_problem_l3868_386879

/-- Rainfall problem --/
theorem rainfall_problem (monday_rain tuesday_rain wednesday_rain thursday_rain friday_rain : ℝ)
  (h_monday : monday_rain = 3)
  (h_tuesday : tuesday_rain = 2 * monday_rain)
  (h_wednesday : wednesday_rain = 0)
  (h_friday : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h_average : (monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain) / 7 = 4) :
  thursday_rain = 5 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l3868_386879


namespace NUMINAMATH_CALUDE_ball_problem_proof_l3868_386829

/-- Represents the arrangement of 8 balls with specific conditions -/
def arrangement_count : ℕ := 576

/-- Represents the number of ways to take out 4 balls ensuring each color is taken -/
def takeout_count : ℕ := 40

/-- Represents the number of ways to divide 8 balls into three groups, each with at least 2 balls -/
def division_count : ℕ := 490

/-- Total number of balls -/
def total_balls : ℕ := 8

/-- Number of black balls -/
def black_balls : ℕ := 4

/-- Number of red balls -/
def red_balls : ℕ := 2

/-- Number of yellow balls -/
def yellow_balls : ℕ := 2

theorem ball_problem_proof :
  (total_balls = black_balls + red_balls + yellow_balls) ∧
  (arrangement_count = 576) ∧
  (takeout_count = 40) ∧
  (division_count = 490) := by
  sorry

end NUMINAMATH_CALUDE_ball_problem_proof_l3868_386829


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l3868_386855

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem smallest_fourth_number (a b c d : ℕ) 
  (ha : is_two_digit a) (hb : is_two_digit b) (hc : is_two_digit c) (hd : is_two_digit d)
  (h1 : a = 45) (h2 : b = 26) (h3 : c = 63)
  (h4 : sum_of_digits a + sum_of_digits b + sum_of_digits c + sum_of_digits d = (a + b + c + d) / 3)
  (h5 : (a + b + c + d) % 7 = 0) :
  d ≥ 37 := by
sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l3868_386855


namespace NUMINAMATH_CALUDE_birds_on_fence_l3868_386859

theorem birds_on_fence (initial_birds : ℕ) : 
  initial_birds + 8 = 20 → initial_birds = 12 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3868_386859


namespace NUMINAMATH_CALUDE_deepak_age_l3868_386888

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 2 = 26 →
  deepak_age = 18 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3868_386888


namespace NUMINAMATH_CALUDE_selection_methods_count_l3868_386818

def n : ℕ := 10  -- Total number of college student village officials
def k : ℕ := 3   -- Number of individuals to be selected

def total_without_b : ℕ := Nat.choose (n - 1) k
def without_a_and_c : ℕ := Nat.choose (n - 3) k

theorem selection_methods_count : 
  total_without_b - without_a_and_c = 49 := by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l3868_386818


namespace NUMINAMATH_CALUDE_expression_simplification_l3868_386815

theorem expression_simplification (α : Real) (h : π < α ∧ α < (3*π)/2) :
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) + Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) = -2 / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3868_386815


namespace NUMINAMATH_CALUDE_n_to_b_equals_eight_l3868_386802

theorem n_to_b_equals_eight :
  let n : ℝ := 2 ^ (1/4)
  let b : ℝ := 12.000000000000002
  n ^ b = 8 := by
sorry

end NUMINAMATH_CALUDE_n_to_b_equals_eight_l3868_386802


namespace NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l3868_386849

theorem positive_integer_solutions_inequality (x : ℕ+) : 
  (2 * x.val - 3 ≤ 5) ↔ x ∈ ({1, 2, 3, 4} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l3868_386849


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3868_386860

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let P : Point2D := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3868_386860


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l3868_386812

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the sides of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The side length of the given isosceles trapezoid is 5 -/
theorem isosceles_trapezoid_side_length :
  let t : IsoscelesTrapezoid := { base1 := 10, base2 := 16, area := 52 }
  side_length t = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l3868_386812


namespace NUMINAMATH_CALUDE_marked_percentage_above_cost_price_l3868_386827

/-- Proves that for an article with given cost price, selling price, and discount percentage,
    the marked percentage above the cost price is correct. -/
theorem marked_percentage_above_cost_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 540)
  (h2 : selling_price = 496.80)
  (h3 : discount_percentage = 19.999999999999996)
  : (((selling_price / (1 - discount_percentage / 100) - cost_price) / cost_price) * 100 = 15) := by
  sorry

end NUMINAMATH_CALUDE_marked_percentage_above_cost_price_l3868_386827


namespace NUMINAMATH_CALUDE_g_at_negative_two_l3868_386848

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 28*x - 84

theorem g_at_negative_two : g (-2) = 320 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l3868_386848


namespace NUMINAMATH_CALUDE_dress_price_problem_l3868_386823

/-- 
Given a dress with an original price x, if Barb buys it for (x/2 - 10) dollars 
and saves 80 dollars, then x = 140.
-/
theorem dress_price_problem (x : ℝ) 
  (h1 : x - (x / 2 - 10) = 80) : x = 140 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_problem_l3868_386823


namespace NUMINAMATH_CALUDE_bunny_burrow_exits_l3868_386887

-- Define the rate at which a bunny comes out of its burrow
def bunny_rate : ℕ := 3

-- Define the number of bunnies
def num_bunnies : ℕ := 20

-- Define the time period in hours
def time_period : ℕ := 10

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem statement
theorem bunny_burrow_exits :
  bunny_rate * minutes_per_hour * time_period * num_bunnies = 36000 := by
  sorry

end NUMINAMATH_CALUDE_bunny_burrow_exits_l3868_386887


namespace NUMINAMATH_CALUDE_gym_membership_ratio_l3868_386832

theorem gym_membership_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (35 * f + 30 * m) / (f + m) = 32 → f / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_gym_membership_ratio_l3868_386832


namespace NUMINAMATH_CALUDE_haley_carrots_count_l3868_386820

/-- The number of carrots Haley picked -/
def haley_carrots : ℕ := 39

/-- The number of carrots Haley's mom picked -/
def mom_carrots : ℕ := 38

/-- The number of good carrots -/
def good_carrots : ℕ := 64

/-- The number of bad carrots -/
def bad_carrots : ℕ := 13

theorem haley_carrots_count : haley_carrots = 39 := by
  have total_carrots : ℕ := good_carrots + bad_carrots
  have total_carrots_alt : ℕ := haley_carrots + mom_carrots
  have h1 : total_carrots = total_carrots_alt := by sorry
  sorry

end NUMINAMATH_CALUDE_haley_carrots_count_l3868_386820
