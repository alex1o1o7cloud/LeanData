import Mathlib

namespace NUMINAMATH_CALUDE_fruit_cost_percentage_increase_l2360_236081

theorem fruit_cost_percentage_increase (max_cost min_cost : ℝ) 
  (h_max : max_cost = 45)
  (h_min : min_cost = 30) :
  (max_cost - min_cost) / min_cost * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_fruit_cost_percentage_increase_l2360_236081


namespace NUMINAMATH_CALUDE_quadratic_value_at_negative_two_l2360_236013

theorem quadratic_value_at_negative_two (a b : ℝ) :
  (2 * a * 1^2 + b * 1 = 3) → (a * (-2)^2 - b * (-2) = 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_at_negative_two_l2360_236013


namespace NUMINAMATH_CALUDE_team_division_probabilities_l2360_236042

def totalTeams : ℕ := 8
def weakTeams : ℕ := 3
def groupSize : ℕ := 4

/-- The probability that one of the groups has exactly 2 weak teams -/
def prob_exactly_two_weak : ℚ := 6/7

/-- The probability that group A has at least 2 weak teams -/
def prob_at_least_two_weak : ℚ := 1/2

theorem team_division_probabilities :
  (totalTeams = 8 ∧ weakTeams = 3 ∧ groupSize = 4) →
  (prob_exactly_two_weak = 6/7 ∧ prob_at_least_two_weak = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_team_division_probabilities_l2360_236042


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2360_236001

theorem quadratic_no_real_roots (c : ℤ) : 
  c < 3 → 
  (∀ x : ℝ, x^2 + 2*x + c ≠ 0) → 
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2360_236001


namespace NUMINAMATH_CALUDE_bread_pieces_in_pond_l2360_236072

theorem bread_pieces_in_pond :
  ∀ (total : ℕ),
    (∃ (duck1 duck2 duck3 : ℕ),
      duck1 = total / 2 ∧
      duck2 = 13 ∧
      duck3 = 7 ∧
      duck1 + duck2 + duck3 + 30 = total) →
    total = 100 := by
sorry

end NUMINAMATH_CALUDE_bread_pieces_in_pond_l2360_236072


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2360_236012

/-- Represents the dimensions of a rectangle -/
structure RectDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the given dimensions satisfy the problem conditions -/
def satisfiesConditions (dim : RectDimensions) : Prop :=
  dim.length + dim.width = 11 ∧
  (dim.length = 5 ∧ dim.width = 6) ∨
  (dim.length = 8 ∧ dim.width = 3) ∨
  (dim.length = 4 ∧ dim.width = 7)

theorem rectangle_dimensions :
  ∀ (dim : RectDimensions),
    (2 * (dim.length + dim.width) = 22) →
    (∃ (subRect : RectDimensions),
      subRect.length = 2 ∧ subRect.width = 6 ∧
      subRect.length ≤ dim.length ∧ subRect.width ≤ dim.width) →
    satisfiesConditions dim :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2360_236012


namespace NUMINAMATH_CALUDE_smallest_covering_circle_l2360_236057

-- Define the plane region
def plane_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle equation
def circle_equation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_covering_circle :
  ∃ (a b r : ℝ), 
    (∀ x y, plane_region x y → circle_equation x y a b r) ∧
    (∀ a' b' r', (∀ x y, plane_region x y → circle_equation x y a' b' r') → r' ≥ r) ∧
    a = 2 ∧ b = 1 ∧ r^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_circle_l2360_236057


namespace NUMINAMATH_CALUDE_binomial_identity_l2360_236050

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identity_l2360_236050


namespace NUMINAMATH_CALUDE_solve_square_root_equation_l2360_236095

theorem solve_square_root_equation (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 12 → x = 27 := by sorry

end NUMINAMATH_CALUDE_solve_square_root_equation_l2360_236095


namespace NUMINAMATH_CALUDE_kaleb_shirts_l2360_236064

theorem kaleb_shirts (initial_shirts : ℕ) (removed_shirts : ℕ) :
  initial_shirts = 17 →
  removed_shirts = 7 →
  initial_shirts - removed_shirts = 10 :=
by sorry

end NUMINAMATH_CALUDE_kaleb_shirts_l2360_236064


namespace NUMINAMATH_CALUDE_pool_width_calculation_l2360_236051

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ

/-- Calculates the total area of the pool and deck -/
def totalArea (p : PoolWithDeck) : ℝ :=
  (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth)

/-- Theorem stating the width of the pool given specific conditions -/
theorem pool_width_calculation (p : PoolWithDeck) 
    (h1 : p.poolLength = 20)
    (h2 : p.deckWidth = 3)
    (h3 : totalArea p = 728) :
    p.poolWidth = 572 / 46 := by
  sorry

end NUMINAMATH_CALUDE_pool_width_calculation_l2360_236051


namespace NUMINAMATH_CALUDE_min_sum_squares_l2360_236040

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0)
  (sum_cond : x₁ + 2*x₂ + 3*x₃ = 60) :
  x₁^2 + x₂^2 + x₃^2 ≥ 1800/7 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 2*y₂ + 3*y₃ = 60 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 1800/7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2360_236040


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l2360_236045

theorem flu_transmission_rate (initial_infected : ℕ) (total_infected : ℕ) (transmission_rate : ℝ) : 
  initial_infected = 1 →
  total_infected = 100 →
  initial_infected + transmission_rate + transmission_rate * (initial_infected + transmission_rate) = total_infected →
  transmission_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l2360_236045


namespace NUMINAMATH_CALUDE_inequality_system_solution_expression_factorization_l2360_236060

-- Part 1: System of inequalities
theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 ≤ 4 - x ∧ x - 1 < 3 * x / 2) ↔ (-2 < x ∧ x ≤ 1) := by sorry

-- Part 2: Expression factorization
theorem expression_factorization (a x y : ℝ) :
  a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_expression_factorization_l2360_236060


namespace NUMINAMATH_CALUDE_xiao_ming_exam_probabilities_l2360_236030

/-- Represents the probabilities of scoring in different ranges in a math exam -/
structure ExamProbabilities where
  above90 : ℝ
  between80and89 : ℝ
  between70and79 : ℝ
  between60and69 : ℝ

/-- Calculates the probability of scoring above 80 -/
def probAbove80 (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89

/-- Calculates the probability of passing the exam (scoring above 60) -/
def probPassing (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89 + p.between70and79 + p.between60and69

/-- Theorem stating the probabilities for Xiao Ming's math exam -/
theorem xiao_ming_exam_probabilities (p : ExamProbabilities)
    (h1 : p.above90 = 0.18)
    (h2 : p.between80and89 = 0.51)
    (h3 : p.between70and79 = 0.15)
    (h4 : p.between60and69 = 0.09) :
    probAbove80 p = 0.69 ∧ probPassing p = 0.93 := by
  sorry


end NUMINAMATH_CALUDE_xiao_ming_exam_probabilities_l2360_236030


namespace NUMINAMATH_CALUDE_triangle_properties_l2360_236041

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a * Real.sin t.C = Real.sqrt 3 * t.c * Real.cos t.A ∧
  t.b = 2 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) :
  satisfies_conditions t → t.A = π / 3 ∧ t.a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2360_236041


namespace NUMINAMATH_CALUDE_sum_greater_product_iff_one_l2360_236074

theorem sum_greater_product_iff_one (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ a = 1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_product_iff_one_l2360_236074


namespace NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l2360_236039

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Theorem statement
theorem consecutive_numbers_digit_sum_exists :
  ∃ n : ℕ, sumOfDigits n = 52 ∧ sumOfDigits (n + 4) = 20 :=
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l2360_236039


namespace NUMINAMATH_CALUDE_optimal_rectangular_enclosure_area_l2360_236094

theorem optimal_rectangular_enclosure_area
  (perimeter : ℝ)
  (min_length : ℝ)
  (min_width : ℝ)
  (h_perimeter : perimeter = 400)
  (h_min_length : min_length = 100)
  (h_min_width : min_width = 50) :
  ∃ (length width : ℝ),
    length ≥ min_length ∧
    width ≥ min_width ∧
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℝ),
      l ≥ min_length →
      w ≥ min_width →
      2 * (l + w) = perimeter →
      l * w ≤ length * width ∧
      length * width = 10000 :=
by sorry

end NUMINAMATH_CALUDE_optimal_rectangular_enclosure_area_l2360_236094


namespace NUMINAMATH_CALUDE_solve_for_m_l2360_236058

-- Define the functions f and g
def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m
def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

-- State the theorem
theorem solve_for_m : 
  ∀ m : ℚ, 3 * (f m 5) = 2 * (g m 5) → m = 10/7 := by sorry

end NUMINAMATH_CALUDE_solve_for_m_l2360_236058


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2360_236047

theorem fraction_subtraction : 
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2360_236047


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l2360_236065

theorem unique_two_digit_multiple : ∃! t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l2360_236065


namespace NUMINAMATH_CALUDE_ursula_initial_money_l2360_236054

/-- Calculates the initial amount of money Ursula had given the purchase details --/
def initial_money (num_hotdogs : ℕ) (price_hotdog : ℚ) (num_salads : ℕ) (price_salad : ℚ) (change : ℚ) : ℚ :=
  num_hotdogs * price_hotdog + num_salads * price_salad + change

/-- Proves that Ursula's initial money was $20.00 given the purchase details --/
theorem ursula_initial_money :
  initial_money 5 (3/2) 3 (5/2) 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ursula_initial_money_l2360_236054


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2360_236055

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, (1202 + m) % 4 = 0 → n ≤ m) ∧ (1202 + n) % 4 = 0 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2360_236055


namespace NUMINAMATH_CALUDE_vasya_figure_cells_l2360_236088

/-- A figure that can be cut into both 2x2 squares and zigzags of 4 cells -/
structure VasyaFigure where
  cells : ℕ
  divisible_by_4 : 4 ∣ cells
  can_cut_into_2x2 : ∃ n : ℕ, cells = 4 * n
  can_cut_into_zigzags : ∃ m : ℕ, cells = 4 * m

/-- The number of cells in Vasya's figure is a multiple of 8 and is at least 16 -/
theorem vasya_figure_cells (fig : VasyaFigure) : 
  ∃ k : ℕ, fig.cells = 8 * k ∧ fig.cells ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_vasya_figure_cells_l2360_236088


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2360_236031

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {x | |x - 2| ≥ 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = A_intersect_B := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2360_236031


namespace NUMINAMATH_CALUDE_available_sandwich_kinds_l2360_236076

/-- The number of sandwich kinds initially available on the menu. -/
def initial_sandwich_kinds : ℕ := 9

/-- The number of sandwich kinds that were sold out. -/
def sold_out_sandwich_kinds : ℕ := 5

/-- Theorem stating that the number of currently available sandwich kinds is 4. -/
theorem available_sandwich_kinds : 
  initial_sandwich_kinds - sold_out_sandwich_kinds = 4 := by
  sorry

end NUMINAMATH_CALUDE_available_sandwich_kinds_l2360_236076


namespace NUMINAMATH_CALUDE_final_center_coordinates_l2360_236037

-- Define the initial center coordinates
def initial_center : ℝ × ℝ := (6, -5)

-- Define the reflection about y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the reflection over y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the composition of the two reflections
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_axis (reflect_y_eq_x p)

-- Theorem statement
theorem final_center_coordinates :
  double_reflection initial_center = (5, 6) := by sorry

end NUMINAMATH_CALUDE_final_center_coordinates_l2360_236037


namespace NUMINAMATH_CALUDE_problem_solution_l2360_236067

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ∧
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2360_236067


namespace NUMINAMATH_CALUDE_equal_projections_implies_a_equals_one_l2360_236029

-- Define the points and vectors
def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 4)

def OA (a : ℝ) : ℝ × ℝ := A a
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem equal_projections_implies_a_equals_one (a : ℝ) :
  dot_product (OA a) OC = dot_product OB OC → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_projections_implies_a_equals_one_l2360_236029


namespace NUMINAMATH_CALUDE_bread_slices_proof_l2360_236028

/-- The number of slices Andy ate at each time -/
def slices_eaten_per_time : ℕ := 3

/-- The number of times Andy ate slices -/
def times_andy_ate : ℕ := 2

/-- The number of slices needed to make one piece of toast bread -/
def slices_per_toast : ℕ := 2

/-- The number of pieces of toast bread made -/
def toast_pieces_made : ℕ := 10

/-- The number of slices left after making toast -/
def slices_left : ℕ := 1

/-- The total number of slices in the original loaf of bread -/
def total_slices : ℕ := 27

theorem bread_slices_proof :
  total_slices = 
    slices_eaten_per_time * times_andy_ate + 
    slices_per_toast * toast_pieces_made + 
    slices_left :=
by
  sorry

end NUMINAMATH_CALUDE_bread_slices_proof_l2360_236028


namespace NUMINAMATH_CALUDE_smallest_exponent_of_ten_l2360_236005

theorem smallest_exponent_of_ten (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 2012 → 
  a.factorial * b.factorial * c.factorial = m * 10^n → 
  ¬(10 ∣ m) → 
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), a.factorial * b.factorial * c.factorial = m' * 10^k ∧ 10 ∣ m') →
  n = 501 := by
sorry

end NUMINAMATH_CALUDE_smallest_exponent_of_ten_l2360_236005


namespace NUMINAMATH_CALUDE_set_equality_l2360_236033

open Set Real

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | Real.log (x - 2) ≤ 0}

theorem set_equality : (Aᶜ ∪ B) = Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_set_equality_l2360_236033


namespace NUMINAMATH_CALUDE_abs_diff_one_if_sum_one_l2360_236004

theorem abs_diff_one_if_sum_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_one_if_sum_one_l2360_236004


namespace NUMINAMATH_CALUDE_mobile_phone_costs_and_schemes_l2360_236018

/-- Given the cost equations for mobile phones, this theorem proves the costs of each type
    and the number of valid purchasing schemes. -/
theorem mobile_phone_costs_and_schemes :
  ∃ (cost_A cost_B : ℕ) (num_schemes : ℕ),
    -- Cost equations
    (2 * cost_A + 3 * cost_B = 7400) ∧
    (3 * cost_A + 5 * cost_B = 11700) ∧
    -- Costs of phones
    (cost_A = 1900) ∧
    (cost_B = 1200) ∧
    -- Number of valid purchasing schemes
    (num_schemes = 9) ∧
    -- Definition of valid purchasing schemes
    (∀ m : ℕ, 
      (12 ≤ m ∧ m ≤ 20) ↔ 
      (44400 ≤ 1900*m + 1200*(30-m) ∧ 1900*m + 1200*(30-m) ≤ 50000)) := by
  sorry


end NUMINAMATH_CALUDE_mobile_phone_costs_and_schemes_l2360_236018


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2360_236003

/-- A quadratic function of the form f(x) = x^2 + c*x + d -/
def f (c d : ℝ) (x : ℝ) : ℝ := x^2 + c*x + d

/-- The theorem stating the uniqueness of c and d for the given condition -/
theorem quadratic_function_uniqueness :
  ∀ c d : ℝ,
  (∀ x : ℝ, (f c d (f c d x + 2*x)) / (f c d x) = 2*x^2 + 1984*x + 2024) →
  c = 1982 ∧ d = 21 := by
  sorry

#check quadratic_function_uniqueness

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2360_236003


namespace NUMINAMATH_CALUDE_polly_cooking_time_l2360_236046

/-- Represents the cooking time for a week -/
structure CookingTime where
  breakfast_daily : ℕ
  lunch_daily : ℕ
  dinner_four_days : ℕ
  total_week : ℕ

/-- Calculates the time spent cooking dinner on the remaining days -/
def remaining_dinner_time (c : CookingTime) : ℕ :=
  c.total_week - (7 * (c.breakfast_daily + c.lunch_daily) + 4 * c.dinner_four_days)

/-- Theorem stating that given the conditions, Polly spends 90 minutes cooking dinner on the remaining days -/
theorem polly_cooking_time :
  let c : CookingTime := {
    breakfast_daily := 20,
    lunch_daily := 5,
    dinner_four_days := 10,
    total_week := 305
  }
  remaining_dinner_time c = 90 := by sorry

end NUMINAMATH_CALUDE_polly_cooking_time_l2360_236046


namespace NUMINAMATH_CALUDE_chenny_cups_bought_l2360_236026

def plate_cost : ℝ := 2
def spoon_cost : ℝ := 1.5
def fork_cost : ℝ := 1.25
def cup_cost : ℝ := 3
def num_plates : ℕ := 9

def total_spoons_forks_cost : ℝ := 13.5
def total_plates_cups_cost : ℝ := 25.5

theorem chenny_cups_bought :
  ∃ (num_spoons num_forks num_cups : ℕ),
    num_spoons = num_forks ∧
    num_spoons * spoon_cost + num_forks * fork_cost = total_spoons_forks_cost ∧
    num_plates * plate_cost + num_cups * cup_cost = total_plates_cups_cost ∧
    num_cups = 2 :=
by sorry

end NUMINAMATH_CALUDE_chenny_cups_bought_l2360_236026


namespace NUMINAMATH_CALUDE_unique_solution_5a_7b_plus_4_eq_3c_l2360_236025

theorem unique_solution_5a_7b_plus_4_eq_3c :
  ∀ a b c : ℕ, 5^a * 7^b + 4 = 3^c → a = 1 ∧ b = 0 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_5a_7b_plus_4_eq_3c_l2360_236025


namespace NUMINAMATH_CALUDE_problem_statement_l2360_236022

theorem problem_statement (x y P Q : ℝ) 
  (h1 : x + y = P)
  (h2 : x^2 + y^2 = Q)
  (h3 : x^3 + y^3 = P^2) :
  Q = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2360_236022


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2360_236079

theorem binomial_divisibility (n k : ℕ) (h_k : k > 1) :
  (∀ m : ℕ, 1 ≤ m ∧ m < n → k ∣ Nat.choose n m) ↔
  ∃ (p : ℕ) (t : ℕ+), Nat.Prime p ∧ n = p ^ (t : ℕ) ∧ k = p :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2360_236079


namespace NUMINAMATH_CALUDE_jimmy_lost_points_l2360_236011

def jimmy_problem (points_to_pass : ℕ) (points_per_exam : ℕ) (num_exams : ℕ) (extra_points : ℕ) : Prop :=
  let total_exam_points := points_per_exam * num_exams
  let current_points := points_to_pass + extra_points
  let lost_points := total_exam_points - current_points
  lost_points = 5

theorem jimmy_lost_points :
  jimmy_problem 50 20 3 5 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_lost_points_l2360_236011


namespace NUMINAMATH_CALUDE_ratio_problem_l2360_236071

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2360_236071


namespace NUMINAMATH_CALUDE_new_alloy_aluminum_bounds_l2360_236099

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  aluminum : ℝ
  copper : ℝ
  magnesium : ℝ

/-- Given three alloys and their compositions, proves that a new alloy with 20% copper
    made from these alloys will have an aluminum percentage between 15% and 40% -/
theorem new_alloy_aluminum_bounds 
  (alloy1 : AlloyComposition)
  (alloy2 : AlloyComposition)
  (alloy3 : AlloyComposition)
  (h1 : alloy1.aluminum = 0.6 ∧ alloy1.copper = 0.15 ∧ alloy1.magnesium = 0.25)
  (h2 : alloy2.aluminum = 0 ∧ alloy2.copper = 0.3 ∧ alloy2.magnesium = 0.7)
  (h3 : alloy3.aluminum = 0.45 ∧ alloy3.copper = 0 ∧ alloy3.magnesium = 0.55)
  : ∃ (x1 x2 x3 : ℝ), 
    x1 + x2 + x3 = 1 ∧
    0.15 * x1 + 0.3 * x2 = 0.2 ∧
    0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧
    0.6 * x1 + 0.45 * x3 ≤ 0.4 :=
by sorry

end NUMINAMATH_CALUDE_new_alloy_aluminum_bounds_l2360_236099


namespace NUMINAMATH_CALUDE_backyard_area_l2360_236024

/-- The area of a rectangular backyard given specific walking conditions -/
theorem backyard_area (length width : ℝ) : 
  (20 * length = 800) →
  (8 * (2 * length + 2 * width) = 800) →
  (length * width = 400) := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l2360_236024


namespace NUMINAMATH_CALUDE_flat_tax_calculation_l2360_236009

/-- Calculate the flat tax on a property with given characteristics -/
def calculate_flat_tax (condo_price condo_size barn_price barn_size detached_price detached_size
                        townhouse_price townhouse_size garage_price garage_size pool_price pool_size
                        tax_rate : ℝ) : ℝ :=
  let condo_value := condo_price * condo_size
  let barn_value := barn_price * barn_size
  let detached_value := detached_price * detached_size
  let townhouse_value := townhouse_price * townhouse_size
  let garage_value := garage_price * garage_size
  let pool_value := pool_price * pool_size
  let total_value := condo_value + barn_value + detached_value + townhouse_value + garage_value + pool_value
  total_value * tax_rate

theorem flat_tax_calculation :
  calculate_flat_tax 98 2400 84 1200 102 3500 96 2750 60 480 50 600 0.0125 = 12697.50 := by
  sorry

end NUMINAMATH_CALUDE_flat_tax_calculation_l2360_236009


namespace NUMINAMATH_CALUDE_town_population_l2360_236059

theorem town_population (P : ℕ) : 
  (P + 100 : ℕ) ≥ 400 →
  (((P + 100 - 400) / 2) / 2 / 2 / 2 : ℕ) = 60 →
  P = 1260 := by
  sorry

end NUMINAMATH_CALUDE_town_population_l2360_236059


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2360_236092

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2360_236092


namespace NUMINAMATH_CALUDE_contractor_absent_days_l2360_236043

/-- Proves that given the specified contract conditions, the number of absent days is 8 -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_amount : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_amount = 490) :
  ∃ (days_absent : ℕ), 
    days_absent = 8 ∧ 
    (daily_pay * (total_days - days_absent) - daily_fine * days_absent = total_amount) :=
by sorry


end NUMINAMATH_CALUDE_contractor_absent_days_l2360_236043


namespace NUMINAMATH_CALUDE_cats_remaining_l2360_236089

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l2360_236089


namespace NUMINAMATH_CALUDE_constant_term_proof_l2360_236090

theorem constant_term_proof (a k n : ℤ) :
  (∀ x, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) →
  a - n + k = 7 →
  (3 * 0 + 2) * (2 * 0 - 3) = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l2360_236090


namespace NUMINAMATH_CALUDE_cosine_sum_upper_bound_l2360_236017

theorem cosine_sum_upper_bound (α β γ : Real) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 2) : 
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_upper_bound_l2360_236017


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l2360_236061

theorem pirate_treasure_probability :
  let n : ℕ := 8
  let p_treasure : ℚ := 1/3
  let p_trap : ℚ := 1/6
  let p_empty : ℚ := 1/2
  let k : ℕ := 4
  p_treasure + p_trap + p_empty = 1 →
  (n.choose k : ℚ) * p_treasure^k * p_empty^(n-k) = 35/648 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l2360_236061


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_zero_l2360_236068

theorem complex_arithmetic_expression_equals_zero :
  -6 * (1/3 - 1/2) - 3^2 / (-12) - |-7/4| = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_zero_l2360_236068


namespace NUMINAMATH_CALUDE_parabola_vertices_distance_l2360_236019

/-- Given an equation representing portions of two parabolas, 
    this theorem states the distance between their vertices. -/
theorem parabola_vertices_distance : 
  ∃ (f g : ℝ → ℝ),
    (∀ x y : ℝ, (Real.sqrt (x^2 + y^2) + |y + 2| = 4) ↔ 
      ((y ≥ -2 ∧ y = f x) ∨ (y < -2 ∧ y = g x))) →
    ∃ (v1 v2 : ℝ × ℝ),
      (v1.1 = 0 ∧ v1.2 = f 0) ∧
      (v2.1 = 0 ∧ v2.2 = g 0) ∧
      Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 58 / 11 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertices_distance_l2360_236019


namespace NUMINAMATH_CALUDE_some_number_value_l2360_236016

theorem some_number_value (x : ℝ) : 40 + 5 * 12 / (x / 3) = 41 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2360_236016


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2360_236052

theorem sum_of_fractions : (1 : ℚ) / 3 + (2 : ℚ) / 7 = (13 : ℚ) / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2360_236052


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2360_236034

theorem solve_exponential_equation :
  ∃ x : ℝ, 3^(3*x) = Real.sqrt 81 ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2360_236034


namespace NUMINAMATH_CALUDE_vectors_form_basis_l2360_236048

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (2, 6) → b = (-1, 3) → LinearIndependent ℝ ![a, b] := by sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l2360_236048


namespace NUMINAMATH_CALUDE_circle_radius_from_spherical_coords_l2360_236023

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) is √2 -/
theorem circle_radius_from_spherical_coords : 
  ∀ θ : Real, 
  let ρ : Real := 2
  let φ : Real := π / 4
  let x : Real := ρ * Real.sin φ * Real.cos θ
  let y : Real := ρ * Real.sin φ * Real.sin θ
  let z : Real := ρ * Real.cos φ
  let r : Real := Real.sqrt (x^2 + y^2)
  r = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_from_spherical_coords_l2360_236023


namespace NUMINAMATH_CALUDE_coin_array_problem_l2360_236086

/-- The number of coins in a triangular array -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating the problem -/
theorem coin_array_problem :
  ∃ (n : ℕ), triangular_sum n = 2211 ∧ sum_of_digits n = 12 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l2360_236086


namespace NUMINAMATH_CALUDE_remainder_theorem_l2360_236032

theorem remainder_theorem : ∃ q : ℕ, 2^300 + 300 = (2^150 + 2^75 + 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2360_236032


namespace NUMINAMATH_CALUDE_sequence_existence_and_boundedness_l2360_236066

theorem sequence_existence_and_boundedness (a : ℝ) (n : ℕ) (hn : n > 0) :
  ∃! x : Fin (n + 2) → ℝ,
    (x 0 = 0 ∧ x (Fin.last n) = 0) ∧
    (∀ i : Fin (n + 1), i.val > 0 →
      (x i + x (i + 1)) / 2 = x i + (x i)^3 - a^3) ∧
    (∀ i : Fin (n + 2), |x i| ≤ |a|) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_and_boundedness_l2360_236066


namespace NUMINAMATH_CALUDE_polygon_30_sides_diagonals_l2360_236093

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_polygon_30_sides_diagonals_l2360_236093


namespace NUMINAMATH_CALUDE_molecular_weight_of_BaSO4_l2360_236008

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of moles of BaSO4 -/
def moles_BaSO4 : ℝ := 3

/-- The molecular weight of BaSO4 in g/mol -/
def molecular_weight_BaSO4 : ℝ := atomic_weight_Ba + atomic_weight_S + 4 * atomic_weight_O

/-- The total weight of the given moles of BaSO4 in grams -/
def total_weight_BaSO4 : ℝ := moles_BaSO4 * molecular_weight_BaSO4

theorem molecular_weight_of_BaSO4 :
  total_weight_BaSO4 = 700.164 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_of_BaSO4_l2360_236008


namespace NUMINAMATH_CALUDE_transform_sine_to_cosine_l2360_236038

/-- Given a function f(x) = √3 * sin(2x), prove that translating it right by π/4 
    and then compressing its x-coordinates by half results in g(x) = -√3 * cos(4x) -/
theorem transform_sine_to_cosine (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (2 * x)
  let g : ℝ → ℝ := λ x => -Real.sqrt 3 * Real.cos (4 * x)
  let h : ℝ → ℝ := λ x => f (x / 2 + π / 4)
  h x = g x := by
  sorry

end NUMINAMATH_CALUDE_transform_sine_to_cosine_l2360_236038


namespace NUMINAMATH_CALUDE_monster_hunt_proof_l2360_236035

/-- The sum of a geometric sequence with initial term 2, common ratio 2, and 5 terms -/
def monster_sum : ℕ := 
  List.range 5
  |> List.map (fun n => 2 * 2^n)
  |> List.sum

theorem monster_hunt_proof : monster_sum = 62 := by
  sorry

end NUMINAMATH_CALUDE_monster_hunt_proof_l2360_236035


namespace NUMINAMATH_CALUDE_dans_remaining_potatoes_l2360_236091

/-- Given an initial number of potatoes and a number of eaten potatoes,
    calculate the remaining number of potatoes. -/
def remaining_potatoes (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that Dan's remaining potatoes is 3 given the initial conditions. -/
theorem dans_remaining_potatoes :
  remaining_potatoes 7 4 = 3 := by sorry

end NUMINAMATH_CALUDE_dans_remaining_potatoes_l2360_236091


namespace NUMINAMATH_CALUDE_overlap_percentage_l2360_236069

theorem overlap_percentage (square_side : ℝ) (rect_length rect_width : ℝ) :
  square_side = 18 →
  rect_length = 20 →
  rect_width = 18 →
  (rect_length * rect_width - 2 * square_side * square_side + (2 * square_side - rect_length) * rect_width) / (rect_length * rect_width) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_percentage_l2360_236069


namespace NUMINAMATH_CALUDE_power_product_simplification_l2360_236082

theorem power_product_simplification : (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l2360_236082


namespace NUMINAMATH_CALUDE_exists_non_increasing_log_l2360_236073

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem exists_non_increasing_log :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ¬(∀ (x y : ℝ), x > y → log a x > log a y) :=
by sorry

end NUMINAMATH_CALUDE_exists_non_increasing_log_l2360_236073


namespace NUMINAMATH_CALUDE_total_stuffed_animals_l2360_236015

/-- 
Given:
- x: initial number of stuffed animals
- y: additional stuffed animals from mom
- z: factor of increase from dad's gift

Prove: The total number of stuffed animals is (x + y) * (1 + z)
-/
theorem total_stuffed_animals (x y : ℕ) (z : ℝ) :
  (x + y : ℝ) * (1 + z) = x + y + z * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_total_stuffed_animals_l2360_236015


namespace NUMINAMATH_CALUDE_seven_twelfths_decimal_l2360_236096

theorem seven_twelfths_decimal : 
  (7 : ℚ) / 12 = 0.5833333333333333 := by sorry

end NUMINAMATH_CALUDE_seven_twelfths_decimal_l2360_236096


namespace NUMINAMATH_CALUDE_quadratic_max_l2360_236036

/-- The quadratic function f(x) = -2x^2 + 8x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem quadratic_max :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_max : ℝ), f x_max = 2) ∧
  (∀ (x : ℝ), f x = 2 → x = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_l2360_236036


namespace NUMINAMATH_CALUDE_total_pamphlets_is_10700_l2360_236077

-- Define the printing rates and durations
def mike_initial_rate : ℕ := 600
def mike_initial_duration : ℕ := 9
def mike_final_duration : ℕ := 2

def leo_initial_rate : ℕ := 2 * mike_initial_rate
def leo_initial_duration : ℕ := mike_initial_duration / 3

def sally_initial_rate : ℕ := 3 * mike_initial_rate
def sally_initial_duration : ℕ := leo_initial_duration / 2
def sally_final_duration : ℕ := 1

-- Define the function to calculate total pamphlets
def calculate_total_pamphlets : ℕ :=
  -- Mike's pamphlets
  let mike_pamphlets := mike_initial_rate * mike_initial_duration + 
                        (mike_initial_rate / 3) * mike_final_duration

  -- Leo's pamphlets
  let leo_pamphlets := leo_initial_rate * 1 + 
                       (leo_initial_rate / 2) * 1 + 
                       (leo_initial_rate / 4) * 1

  -- Sally's pamphlets
  let sally_pamphlets := sally_initial_rate * sally_initial_duration + 
                         (leo_initial_rate / 2) * sally_final_duration

  mike_pamphlets + leo_pamphlets + sally_pamphlets

-- Theorem statement
theorem total_pamphlets_is_10700 :
  calculate_total_pamphlets = 10700 := by
  sorry

end NUMINAMATH_CALUDE_total_pamphlets_is_10700_l2360_236077


namespace NUMINAMATH_CALUDE_candy_distribution_l2360_236083

def total_candy : ℕ := 648
def sister_candy : ℕ := 48
def num_friends : ℕ := 3
def num_bags : ℕ := 8

theorem candy_distribution :
  let remaining_candy := total_candy - sister_candy
  let people := num_friends + 1
  let candy_per_person := remaining_candy / people
  let candy_per_bag := candy_per_person / num_bags
  candy_per_bag = 18 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2360_236083


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2360_236020

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i^3 * (i + 1)) / (i - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2360_236020


namespace NUMINAMATH_CALUDE_x_zero_value_l2360_236087

noncomputable def f (x : ℝ) : ℝ := x * (2014 + Real.log x)

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2015) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l2360_236087


namespace NUMINAMATH_CALUDE_average_marks_of_combined_classes_l2360_236085

theorem average_marks_of_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) :
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 40 →
  avg₂ = 60 →
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℝ) = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_of_combined_classes_l2360_236085


namespace NUMINAMATH_CALUDE_expression_value_l2360_236007

theorem expression_value : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2360_236007


namespace NUMINAMATH_CALUDE_chord_relations_l2360_236070

/-- Represents a chord in a unit circle -/
structure Chord where
  length : ℝ

/-- Represents the configuration of chords in the unit circle -/
structure CircleChords where
  MP : Chord
  PQ : Chord
  NR : Chord
  MN : Chord

/-- The given configuration of chords satisfying the problem conditions -/
def given_chords : CircleChords :=
  { MP := ⟨1⟩
  , PQ := ⟨1⟩
  , NR := ⟨2⟩
  , MN := ⟨3⟩ }

theorem chord_relations (c : CircleChords) (h : c = given_chords) :
  (c.MN.length - c.NR.length = 1) ∧
  (c.MN.length * c.NR.length = 6) ∧
  (c.MN.length ^ 2 - c.NR.length ^ 2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_chord_relations_l2360_236070


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l2360_236063

/-- A parabola passing through (2, 3) and (0, 7) has coefficients a, b, c such that a + b + c = 4 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 2)^2 + 3) → -- Vertex form condition
  (a * 0^2 + b * 0 + c = 7) →                      -- Passes through (0, 7)
  (a + b + c = 4) := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l2360_236063


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l2360_236078

theorem square_perimeter_relation (perimeter_A : ℝ) (area_ratio : ℝ) : 
  perimeter_A = 36 →
  area_ratio = 1/3 →
  let side_A := perimeter_A / 4
  let area_A := side_A ^ 2
  let area_B := area_ratio * area_A
  let side_B := Real.sqrt area_B
  let perimeter_B := 4 * side_B
  perimeter_B = 12 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l2360_236078


namespace NUMINAMATH_CALUDE_volume_of_specific_open_box_l2360_236002

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutLength : ℝ) : ℝ :=
  (sheetLength - 2 * cutLength) * (sheetWidth - 2 * cutLength) * cutLength

/-- Theorem stating that the volume of the open box is 5440 m³ given the specified dimensions. -/
theorem volume_of_specific_open_box :
  openBoxVolume 50 36 8 = 5440 := by
  sorry

#eval openBoxVolume 50 36 8

end NUMINAMATH_CALUDE_volume_of_specific_open_box_l2360_236002


namespace NUMINAMATH_CALUDE_unique_solution_l2360_236049

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 81

-- State the theorem
theorem unique_solution :
  ∃! x, f x = 1/4 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2360_236049


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2360_236010

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + m = 0 → y = x) → 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2360_236010


namespace NUMINAMATH_CALUDE_coin_toss_frequency_l2360_236053

/-- Represents the frequency of an event -/
def frequency (occurrences : ℕ) (totalTrials : ℕ) : ℚ :=
  occurrences / totalTrials

/-- Given a coin tossed 10 times with 6 heads, prove the frequency of heads is 3/5 -/
theorem coin_toss_frequency :
  let totalTosses : ℕ := 10
  let headCount : ℕ := 6
  frequency headCount totalTosses = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_frequency_l2360_236053


namespace NUMINAMATH_CALUDE_sine_function_omega_l2360_236006

/-- Given a function f(x) = 2sin(ωx + π/6) with ω > 0, if it intersects the y-axis at (0, 1) 
    and has two adjacent x-intercepts A and B such that the area of triangle PAB is π, 
    then ω = 1/2 -/
theorem sine_function_omega (ω : ℝ) (f : ℝ → ℝ) (A B : ℝ) : 
  ω > 0 →
  (∀ x, f x = 2 * Real.sin (ω * x + π / 6)) →
  f 0 = 1 →
  f A = 0 →
  f B = 0 →
  A < B →
  (B - A) * 1 / 2 = π →
  ω = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_omega_l2360_236006


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2360_236056

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the tangent condition
def is_tangent (x y : ℝ) : Prop := ∃ (t : ℝ), circle_M (x + t) (y + t) ∧ 
  ∀ (s : ℝ), s ≠ t → ¬(circle_M (x + s) (y + s))

-- Define the minimization condition
def is_minimized (P M A B : ℝ × ℝ) : Prop := 
  ∀ (Q : ℝ × ℝ), point_P Q.1 Q.2 → 
    (Q.1 - M.1)^2 + (Q.2 - M.2)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥
    (P.1 - M.1)^2 + (P.2 - M.2)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem tangent_line_equation :
  ∀ (P M A B : ℝ × ℝ),
    circle_M M.1 M.2 →
    point_P P.1 P.2 →
    is_tangent (P.1 - A.1) (P.2 - A.2) →
    is_tangent (P.1 - B.1) (P.2 - B.2) →
    is_minimized P M A B →
    2 * A.1 + A.2 + 1 = 0 ∧ 2 * B.1 + B.2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2360_236056


namespace NUMINAMATH_CALUDE_kindergarten_boys_count_l2360_236097

/-- Given a kindergarten with a 2:3 ratio of boys to girls and 18 girls, prove there are 12 boys -/
theorem kindergarten_boys_count (total_girls : ℕ) (boys_to_girls_ratio : ℚ) : 
  total_girls = 18 → boys_to_girls_ratio = 2/3 → 
  (total_girls : ℚ) * boys_to_girls_ratio = 12 := by
sorry

end NUMINAMATH_CALUDE_kindergarten_boys_count_l2360_236097


namespace NUMINAMATH_CALUDE_triangle_side_values_l2360_236098

theorem triangle_side_values (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * π / 180 →  -- Convert 30° to radians
  a = 1 →
  c = Real.sqrt 3 →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →  -- Law of Cosines
  (b = 1 ∨ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_values_l2360_236098


namespace NUMINAMATH_CALUDE_jane_max_tickets_l2360_236080

/-- The cost of a single ticket -/
def ticket_cost : ℕ := 18

/-- Jane's available money -/
def jane_money : ℕ := 150

/-- The number of tickets required for a discount -/
def discount_threshold : ℕ := 5

/-- The discount rate as a fraction -/
def discount_rate : ℚ := 1 / 10

/-- Calculate the cost of n tickets with possible discount -/
def cost_with_discount (n : ℕ) : ℚ :=
  if n ≤ discount_threshold then
    n * ticket_cost
  else
    discount_threshold * ticket_cost + (n - discount_threshold) * ticket_cost * (1 - discount_rate)

/-- The maximum number of tickets Jane can buy -/
def max_tickets : ℕ := 8

/-- Theorem stating the maximum number of tickets Jane can buy -/
theorem jane_max_tickets :
  ∀ n : ℕ, cost_with_discount n ≤ jane_money ↔ n ≤ max_tickets :=
by sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l2360_236080


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l2360_236062

/-- Given a quadratic equation with specific roots, prove the intersection points of a related parabola with the x-axis -/
theorem parabola_intersection_points 
  (a m : ℝ) 
  (h1 : a * (-1 + m)^2 = 3) 
  (h2 : a * (3 + m)^2 = 3) :
  let f (x : ℝ) := a * (x + m - 2)^2 - 3
  ∃ (x1 x2 : ℝ), x1 = 5 ∧ x2 = 1 ∧ f x1 = 0 ∧ f x2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l2360_236062


namespace NUMINAMATH_CALUDE_sheep_count_l2360_236021

theorem sheep_count : ∀ (num_sheep : ℕ), 
  (∀ (sheep : ℕ), sheep ≤ num_sheep → (sheep * 1 = sheep)) →  -- One sheep eats one bag in 40 days
  (num_sheep * 1 = 40) →  -- Total bags eaten by all sheep is 40
  num_sheep = 40 := by
sorry

end NUMINAMATH_CALUDE_sheep_count_l2360_236021


namespace NUMINAMATH_CALUDE_no_valid_cube_labeling_l2360_236075

/-- A labeling of a cube's edges with 0s and 1s -/
def CubeLabeling := Fin 12 → Fin 2

/-- The set of edges for each face of a cube -/
def cube_faces : Fin 6 → Finset (Fin 12) := sorry

/-- The sum of labels on a face's edges -/
def face_sum (l : CubeLabeling) (face : Fin 6) : Nat :=
  (cube_faces face).sum (λ e => l e)

/-- A labeling is valid if the sum of labels on each face's edges equals 3 -/
def is_valid_labeling (l : CubeLabeling) : Prop :=
  ∀ face : Fin 6, face_sum l face = 3

theorem no_valid_cube_labeling :
  ¬ ∃ l : CubeLabeling, is_valid_labeling l := sorry

end NUMINAMATH_CALUDE_no_valid_cube_labeling_l2360_236075


namespace NUMINAMATH_CALUDE_triangle_existence_theorem_l2360_236044

def triangle_exists (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_x_values : Set ℕ :=
  {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem triangle_existence_theorem :
  ∀ x : ℕ, x > 0 → (triangle_exists 6 15 x ↔ x ∈ valid_x_values) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_theorem_l2360_236044


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l2360_236084

def team_size : ℕ := 12
def offensive_linemen : ℕ := 5

theorem starting_lineup_combinations : 
  (offensive_linemen) *
  (team_size - 1) *
  (team_size - 2) *
  ((team_size - 3) * (team_size - 4) / 2) = 19800 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l2360_236084


namespace NUMINAMATH_CALUDE_rachel_book_count_l2360_236014

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_book_count : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_rachel_book_count_l2360_236014


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l2360_236000

-- Define the operation
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0

-- State the theorem
theorem complex_in_first_quadrant :
  ∃ z : ℂ, determinant z (1 + Complex.I) 2 1 = 0 ∧ is_in_first_quadrant z :=
sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l2360_236000


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_one_negation_of_inequality_l2360_236027

theorem negation_of_forall_gt_one (P : ℝ → Prop) :
  (¬ ∀ x > 1, P x) ↔ (∃ x > 1, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_one_negation_of_inequality_l2360_236027
