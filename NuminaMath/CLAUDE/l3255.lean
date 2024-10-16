import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_linear_if_all_powers_l3255_325503

/-- A sequence defined by a polynomial recurrence -/
def PolynomialSequence (P : ℕ → ℕ) (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => P (PolynomialSequence P n k)

/-- Predicate to check if a number is a perfect power greater than 1 -/
def IsPerfectPower (m : ℕ) : Prop :=
  ∃ (b : ℕ) (k : ℕ), k > 1 ∧ m = k^b

theorem polynomial_linear_if_all_powers (P : ℕ → ℕ) (n : ℕ) :
  (∀ (x y : ℕ), ∃ (a b c : ℤ), P x - P y = a * (x - y) + b * x + c) →
  (∀ (b : ℕ), ∃ (k : ℕ), IsPerfectPower (PolynomialSequence P n k)) →
  ∃ (m q : ℤ), ∀ (x : ℕ), P x = m * x + q :=
sorry

end NUMINAMATH_CALUDE_polynomial_linear_if_all_powers_l3255_325503


namespace NUMINAMATH_CALUDE_f_at_negative_three_l3255_325509

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem f_at_negative_three : f (-3) = 110 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_three_l3255_325509


namespace NUMINAMATH_CALUDE_employees_using_public_transportation_l3255_325596

theorem employees_using_public_transportation
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h1 : total_employees = 100)
  (h2 : drive_percentage = 60 / 100)
  (h3 : public_transport_fraction = 1 / 2) :
  (total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction = 20 := by
  sorry

end NUMINAMATH_CALUDE_employees_using_public_transportation_l3255_325596


namespace NUMINAMATH_CALUDE_same_grade_percentage_l3255_325556

/-- Represents the grade distribution table -/
def gradeDistribution : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![4, 3, 2, 1],
    ![1, 6, 2, 0],
    ![3, 1, 3, 2],
    ![0, 1, 2, 2]]

/-- Total number of students -/
def totalStudents : ℕ := 36

/-- Sum of diagonal elements in the grade distribution table -/
def sameGradeCount : ℕ := (gradeDistribution 0 0) + (gradeDistribution 1 1) + (gradeDistribution 2 2) + (gradeDistribution 3 3)

/-- Theorem stating the percentage of students who received the same grade on both tests -/
theorem same_grade_percentage :
  (sameGradeCount : ℚ) / totalStudents = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l3255_325556


namespace NUMINAMATH_CALUDE_johns_mean_score_l3255_325518

def johns_scores : List ℝ := [89, 92, 95, 88, 90]

theorem johns_mean_score :
  (johns_scores.sum / johns_scores.length : ℝ) = 90.8 := by
  sorry

end NUMINAMATH_CALUDE_johns_mean_score_l3255_325518


namespace NUMINAMATH_CALUDE_complex_pairs_sum_l3255_325525

theorem complex_pairs_sum : ∃ (a₁ b₁ a₂ b₂ : ℕ+), 
  a₁ < b₁ ∧ 
  a₂ < b₂ ∧ 
  (a₁ + Complex.I * b₁) * (b₁ - Complex.I * a₁) = 2020 ∧
  (a₂ + Complex.I * b₂) * (b₂ - Complex.I * a₂) = 2020 ∧
  (a₁ : ℕ) + (b₁ : ℕ) + (a₂ : ℕ) + (b₂ : ℕ) = 714 ∧
  (a₁, b₁) ≠ (a₂, b₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_pairs_sum_l3255_325525


namespace NUMINAMATH_CALUDE_store_promotion_probabilities_l3255_325589

/-- A store promotion event with three prizes -/
structure StorePromotion where
  p_first : ℝ  -- Probability of winning first prize
  p_second : ℝ  -- Probability of winning second prize
  p_third : ℝ  -- Probability of winning third prize
  h_first : 0 ≤ p_first ∧ p_first ≤ 1
  h_second : 0 ≤ p_second ∧ p_second ≤ 1
  h_third : 0 ≤ p_third ∧ p_third ≤ 1

/-- The probability of winning a prize in the store promotion -/
def prob_win_prize (sp : StorePromotion) : ℝ :=
  sp.p_first + sp.p_second + sp.p_third

/-- The probability of not winning any prize in the store promotion -/
def prob_no_prize (sp : StorePromotion) : ℝ :=
  1 - prob_win_prize sp

/-- Theorem stating the probabilities for a specific store promotion -/
theorem store_promotion_probabilities (sp : StorePromotion) 
  (h1 : sp.p_first = 0.1) (h2 : sp.p_second = 0.2) (h3 : sp.p_third = 0.4) : 
  prob_win_prize sp = 0.7 ∧ prob_no_prize sp = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_store_promotion_probabilities_l3255_325589


namespace NUMINAMATH_CALUDE_wetland_area_conversion_l3255_325563

/-- Proves that 20.26 thousand hectares is equal to 2.026 × 10^9 square meters. -/
theorem wetland_area_conversion :
  (20.26 * 1000 : ℝ) * (10^4 : ℝ) = 2.026 * (10^9 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wetland_area_conversion_l3255_325563


namespace NUMINAMATH_CALUDE_prob_first_success_third_trial_l3255_325570

/-- Probability of first success on third trial in a geometric distribution -/
theorem prob_first_success_third_trial (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  let q := 1 - p
  (q ^ 2) * p = p * (1 - p) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_first_success_third_trial_l3255_325570


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3255_325501

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3255_325501


namespace NUMINAMATH_CALUDE_min_cost_theorem_l3255_325577

/-- Represents the prices and quantities of garbage bins -/
structure GarbageBinProblem where
  priceA : ℕ  -- Price of type A garbage bin
  priceB : ℕ  -- Price of type B garbage bin
  totalBins : ℕ  -- Total number of bins to purchase
  maxTypeA : ℕ  -- Maximum number of type A bins

/-- Conditions of the garbage bin problem -/
def problemConditions (p : GarbageBinProblem) : Prop :=
  p.priceA + 2 * p.priceB = 340 ∧
  3 * p.priceA + p.priceB = 420 ∧
  p.totalBins = 30 ∧
  p.maxTypeA = 16

/-- Total cost function -/
def totalCost (p : GarbageBinProblem) (x : ℕ) : ℕ :=
  p.priceA * x + p.priceB * (p.totalBins - x)

/-- Theorem stating the minimum cost -/
theorem min_cost_theorem (p : GarbageBinProblem) 
  (h : problemConditions p) : 
  (∀ x, x ≤ p.maxTypeA → totalCost p p.maxTypeA ≤ totalCost p x) ∧ 
  totalCost p p.maxTypeA = 3280 := by
  sorry

#check min_cost_theorem

end NUMINAMATH_CALUDE_min_cost_theorem_l3255_325577


namespace NUMINAMATH_CALUDE_product_of_sum_of_four_squares_l3255_325532

theorem product_of_sum_of_four_squares (a b : ℤ)
  (ha : ∃ x₁ x₂ x₃ x₄ : ℤ, a = x₁^2 + x₂^2 + x₃^2 + x₄^2)
  (hb : ∃ y₁ y₂ y₃ y₄ : ℤ, b = y₁^2 + y₂^2 + y₃^2 + y₄^2) :
  ∃ z₁ z₂ z₃ z₄ : ℤ, a * b = z₁^2 + z₂^2 + z₃^2 + z₄^2 :=
by sorry

end NUMINAMATH_CALUDE_product_of_sum_of_four_squares_l3255_325532


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_A_l3255_325566

-- Define sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}

-- Theorem statement
theorem A_intersect_B_eq_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_A_l3255_325566


namespace NUMINAMATH_CALUDE_exactly_two_out_of_four_probability_l3255_325543

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability mass function -/
def binomialPMF (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_out_of_four_probability :
  binomialPMF n k p = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_four_probability_l3255_325543


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l3255_325590

/-- Number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- n has exactly 18 positive integer divisors -/
def has_18_divisors (n : ℕ) : Prop := num_divisors n = 18

theorem smallest_with_18_divisors :
  ∃ (n : ℕ), has_18_divisors n ∧ ∀ m : ℕ, has_18_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l3255_325590


namespace NUMINAMATH_CALUDE_intersection_M_N_l3255_325547

open Set

def M : Set ℝ := {x | x < 2017}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3255_325547


namespace NUMINAMATH_CALUDE_marbles_exceed_200_l3255_325530

def marbles (n : ℕ) : ℕ := 3 * 2^(n - 1)

theorem marbles_exceed_200 :
  ∀ k : ℕ, k < 9 → marbles k ≤ 200 ∧ marbles 9 > 200 :=
by sorry

end NUMINAMATH_CALUDE_marbles_exceed_200_l3255_325530


namespace NUMINAMATH_CALUDE_weekly_running_distance_l3255_325527

/-- Calculates the total distance run in a week given the number of days, hours per day, and speed. -/
def total_distance_run (days_per_week : ℕ) (hours_per_day : ℝ) (speed_mph : ℝ) : ℝ :=
  days_per_week * hours_per_day * speed_mph

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem weekly_running_distance :
  total_distance_run 5 1.5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_weekly_running_distance_l3255_325527


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3255_325593

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2

-- Define set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem intersection_of_A_and_B 
  (A : Set ℝ) 
  (h : f '' A ⊆ B) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3255_325593


namespace NUMINAMATH_CALUDE_line_parameterization_l3255_325506

/-- Given a line y = 2x - 40 parameterized by (x, y) = (f(t), 20t - 14),
    prove that f(t) = 10t + 13 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20 * t - 14 = 2 * (f t) - 40) → 
  (∀ t : ℝ, f t = 10 * t + 13) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3255_325506


namespace NUMINAMATH_CALUDE_rectangle_area_l3255_325581

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 2500 →
  rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 200 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3255_325581


namespace NUMINAMATH_CALUDE_symmetric_quadratic_l3255_325538

/-- A quadratic function f(x) = x² + (a-2)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-2)*x + 3

/-- The interval [a, b] -/
def interval (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

/-- The line of symmetry x = 1 -/
def symmetry_line : ℝ := 1

/-- The statement that the graph of f is symmetric about x = 1 on [a, b] -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x ∈ interval a b, f a x = f a (2*symmetry_line - x)

theorem symmetric_quadratic (a b : ℝ) :
  is_symmetric a b → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_l3255_325538


namespace NUMINAMATH_CALUDE_share_price_increase_l3255_325554

theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter := P * 1.25
  let second_quarter := first_quarter * 1.24
  (second_quarter - P) / P * 100 = 55 := by
sorry

end NUMINAMATH_CALUDE_share_price_increase_l3255_325554


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_a_range_l3255_325567

theorem quadratic_two_zeros_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 4 * x₁ - 2 = 0 ∧ a * x₂^2 + 4 * x₂ - 2 = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_a_range_l3255_325567


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l3255_325516

theorem cafeteria_red_apples :
  ∀ (red_apples green_apples students_wanting_fruit extra_apples : ℕ),
    green_apples = 15 →
    students_wanting_fruit = 5 →
    extra_apples = 16 →
    red_apples + green_apples = students_wanting_fruit + extra_apples →
    red_apples = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l3255_325516


namespace NUMINAMATH_CALUDE_probability_equals_three_fourths_l3255_325569

/-- The set S in R^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | -2 ≤ p.2 ∧ p.2 ≤ |p.1| ∧ -2 ≤ p.1 ∧ p.1 ≤ 2}

/-- The subset of S where |x| + |y| < 2 -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ S ∧ |p.1| + |p.2| < 2}

/-- The area of a set in R^2 -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem probability_equals_three_fourths :
  area T / area S = 3/4 := by sorry

end NUMINAMATH_CALUDE_probability_equals_three_fourths_l3255_325569


namespace NUMINAMATH_CALUDE_total_shaded_area_is_71_l3255_325573

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  r.width * r.height

theorem total_shaded_area_is_71 (rect1 rect2 overlap : Rectangle)
    (h1 : rect1.width = 4 ∧ rect1.height = 12)
    (h2 : rect2.width = 5 ∧ rect2.height = 7)
    (h3 : overlap.width = 3 ∧ overlap.height = 4) :
    area rect1 + area rect2 - area overlap = 71 := by
  sorry

#check total_shaded_area_is_71

end NUMINAMATH_CALUDE_total_shaded_area_is_71_l3255_325573


namespace NUMINAMATH_CALUDE_solution_set_abs_equation_l3255_325520

theorem solution_set_abs_equation (x : ℝ) :
  |x - 2| + |2*x - 3| = |3*x - 5| ↔ x ≤ 3/2 ∨ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_equation_l3255_325520


namespace NUMINAMATH_CALUDE_jellybean_problem_l3255_325568

theorem jellybean_problem (initial_quantity : ℕ) : 
  (initial_quantity : ℝ) * (0.75^3) = 27 → initial_quantity = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3255_325568


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3255_325544

theorem trigonometric_problem (α : Real) 
  (h1 : 3 * Real.pi / 4 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan α + 1 / Real.tan α = -10/3) : 
  Real.tan α = -1/3 ∧ 
  (5 * Real.sin (α/2)^2 + 8 * Real.sin (α/2) * Real.cos (α/2) + 11 * Real.cos (α/2)^2 - 8) / 
  (Real.sqrt 2 * Real.sin (α - Real.pi/4)) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3255_325544


namespace NUMINAMATH_CALUDE_male_gerbil_fraction_l3255_325553

theorem male_gerbil_fraction (total_pets : ℕ) (total_gerbils : ℕ) (total_males : ℕ) :
  total_pets = 90 →
  total_gerbils = 66 →
  total_males = 25 →
  (total_pets - total_gerbils) / 3 + (total_males - (total_pets - total_gerbils) / 3) = total_males →
  (total_males - (total_pets - total_gerbils) / 3) / total_gerbils = 17 / 66 := by
  sorry

end NUMINAMATH_CALUDE_male_gerbil_fraction_l3255_325553


namespace NUMINAMATH_CALUDE_product_grade_probabilities_l3255_325571

theorem product_grade_probabilities :
  ∀ (p_quality p_second : ℝ),
  p_quality = 0.98 →
  p_second = 0.21 →
  0 ≤ p_quality ∧ p_quality ≤ 1 →
  0 ≤ p_second ∧ p_second ≤ 1 →
  ∃ (p_first p_third : ℝ),
    p_first = p_quality - p_second ∧
    p_third = 1 - p_quality ∧
    p_first = 0.77 ∧
    p_third = 0.02 :=
by
  sorry

end NUMINAMATH_CALUDE_product_grade_probabilities_l3255_325571


namespace NUMINAMATH_CALUDE_is_circle_center_l3255_325548

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l3255_325548


namespace NUMINAMATH_CALUDE_square_calculation_identity_l3255_325533

theorem square_calculation_identity (x : ℝ) : ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_calculation_identity_l3255_325533


namespace NUMINAMATH_CALUDE_intersection_with_y_axis_l3255_325550

/-- The intersection point of the line y = 5x + 1 with the y-axis is (0, 1) -/
theorem intersection_with_y_axis :
  let f : ℝ → ℝ := λ x ↦ 5 * x + 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_with_y_axis_l3255_325550


namespace NUMINAMATH_CALUDE_salesman_profit_l3255_325502

/-- Calculates the salesman's profit from backpack sales --/
theorem salesman_profit : 
  let initial_cost : ℚ := 1500
  let import_tax_rate : ℚ := 5 / 100
  let total_cost : ℚ := initial_cost * (1 + import_tax_rate)
  let swap_meet_sales : ℚ := 30 * 22
  let department_store_sales : ℚ := 25 * 35
  let online_sales_regular : ℚ := 10 * 28
  let online_sales_discounted : ℚ := 5 * 28 * (1 - 10 / 100)
  let local_market_sales_1 : ℚ := 10 * 33
  let local_market_sales_2 : ℚ := 5 * 40
  let local_market_sales_3 : ℚ := 15 * 25
  let shipping_expenses : ℚ := 60
  let total_revenue : ℚ := swap_meet_sales + department_store_sales + 
    online_sales_regular + online_sales_discounted + 
    local_market_sales_1 + local_market_sales_2 + local_market_sales_3
  let profit : ℚ := total_revenue - total_cost - shipping_expenses
  profit = 1211 := by sorry

end NUMINAMATH_CALUDE_salesman_profit_l3255_325502


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3255_325500

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 1 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3255_325500


namespace NUMINAMATH_CALUDE_linear_equation_root_range_l3255_325505

theorem linear_equation_root_range (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) ↔ (k < 1 ∨ k > 3) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_root_range_l3255_325505


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3255_325574

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 - 2*x₀ - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3255_325574


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_3x_l3255_325565

theorem factorization_of_x_squared_minus_3x (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_3x_l3255_325565


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l3255_325579

theorem geometric_mean_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^x * 4^y)) :
  x^2 + 2*y^2 ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l3255_325579


namespace NUMINAMATH_CALUDE_square_diff_minus_diff_squares_l3255_325514

theorem square_diff_minus_diff_squares (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_square_diff_minus_diff_squares_l3255_325514


namespace NUMINAMATH_CALUDE_trapezium_circle_radius_l3255_325511

/-- Represents a trapezium PQRS with a circle tangent to all sides -/
structure TrapeziumWithCircle where
  -- Length of PQ and SR
  side_length : ℝ
  -- Area of the trapezium
  area : ℝ
  -- Assertion that SP is parallel to RQ
  sp_parallel_rq : Prop
  -- Assertion that all sides are tangent to the circle
  all_sides_tangent : Prop

/-- The radius of the circle in a trapezium with given properties -/
def circle_radius (t : TrapeziumWithCircle) : ℝ :=
  12

/-- Theorem stating that for a trapezium with given properties, the radius of the inscribed circle is 12 -/
theorem trapezium_circle_radius 
  (t : TrapeziumWithCircle) 
  (h1 : t.side_length = 25)
  (h2 : t.area = 600) :
  circle_radius t = 12 := by sorry

end NUMINAMATH_CALUDE_trapezium_circle_radius_l3255_325511


namespace NUMINAMATH_CALUDE_mary_nickels_count_l3255_325599

/-- The number of nickels Mary has after receiving some from her dad and sister -/
def total_nickels (initial : ℕ) (from_dad : ℕ) (from_sister : ℕ) : ℕ :=
  initial + from_dad + from_sister

/-- Theorem stating that Mary's total nickels is the sum of her initial amount and what she received -/
theorem mary_nickels_count : total_nickels 7 12 9 = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_count_l3255_325599


namespace NUMINAMATH_CALUDE_intersection_distance_l3255_325542

/-- The distance between the intersection points of the line x - y + 1 = 0 and the circle x² + y² = 2 is equal to √6. -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.1 - A.2 + 1 = 0) ∧ (A.1^2 + A.2^2 = 2) ∧
    (B.1 - B.2 + 1 = 0) ∧ (B.1^2 + B.2^2 = 2) ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3255_325542


namespace NUMINAMATH_CALUDE_count_false_propositions_l3255_325504

-- Define the original proposition
def original_prop (a : ℝ) : Prop := a > 1 → a > 2

-- Define the inverse proposition
def inverse_prop (a : ℝ) : Prop := ¬(a > 1) → ¬(a > 2)

-- Define the negation proposition
def negation_prop (a : ℝ) : Prop := ¬(a > 1 → a > 2)

-- Define the converse proposition
def converse_prop (a : ℝ) : Prop := a > 2 → a > 1

-- Count the number of false propositions
def count_false_props : ℕ := 2

-- Theorem statement
theorem count_false_propositions :
  count_false_props = 2 :=
sorry

end NUMINAMATH_CALUDE_count_false_propositions_l3255_325504


namespace NUMINAMATH_CALUDE_orange_problem_l3255_325545

theorem orange_problem (total : ℕ) (ripe_fraction : ℚ) (eaten_ripe_fraction : ℚ) (eaten_unripe_fraction : ℚ) :
  total = 96 →
  ripe_fraction = 1/2 →
  eaten_ripe_fraction = 1/4 →
  eaten_unripe_fraction = 1/8 →
  (total : ℚ) * (1 - ripe_fraction * eaten_ripe_fraction - (1 - ripe_fraction) * eaten_unripe_fraction) = 78 := by
  sorry

end NUMINAMATH_CALUDE_orange_problem_l3255_325545


namespace NUMINAMATH_CALUDE_subtract_fractions_l3255_325507

theorem subtract_fractions : (2 : ℚ) / 3 - 5 / 12 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l3255_325507


namespace NUMINAMATH_CALUDE_white_square_area_l3255_325523

theorem white_square_area (cube_edge : ℝ) (green_paint_area : ℝ) (white_square_area : ℝ) : 
  cube_edge = 12 →
  green_paint_area = 432 →
  white_square_area = (cube_edge ^ 2) - (green_paint_area / 6) →
  white_square_area = 72 := by
sorry

end NUMINAMATH_CALUDE_white_square_area_l3255_325523


namespace NUMINAMATH_CALUDE_pet_shop_legs_l3255_325515

/-- The number of legs for each animal type --/
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def snake_legs : ℕ := 0
def spider_legs : ℕ := 8

/-- The number of each animal type --/
def num_birds : ℕ := 3
def num_dogs : ℕ := 5
def num_snakes : ℕ := 4
def num_spiders : ℕ := 1

/-- The total number of legs in the pet shop --/
def total_legs : ℕ := 
  num_birds * bird_legs + 
  num_dogs * dog_legs + 
  num_snakes * snake_legs + 
  num_spiders * spider_legs

theorem pet_shop_legs : total_legs = 34 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_legs_l3255_325515


namespace NUMINAMATH_CALUDE_three_in_A_even_not_in_A_l3255_325587

-- Define the set A
def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- Theorem statements
theorem three_in_A : 3 ∈ A := by sorry

theorem even_not_in_A : ∀ k : ℤ, (4*k - 2) ∉ A := by sorry

end NUMINAMATH_CALUDE_three_in_A_even_not_in_A_l3255_325587


namespace NUMINAMATH_CALUDE_unique_zero_point_condition_l3255_325536

def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

theorem unique_zero_point_condition (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ f a x = 0) ↔ (1 < a ∧ a < 5) ∨ a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_condition_l3255_325536


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3255_325557

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x < 0 → x + 1 / x ≤ -2)) ↔ (∃ x : ℝ, x < 0 ∧ x + 1 / x > -2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3255_325557


namespace NUMINAMATH_CALUDE_n_minus_m_equals_six_l3255_325585

-- Define the sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Define the set difference operation
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem n_minus_m_equals_six : set_difference N M = {6} := by
  sorry

end NUMINAMATH_CALUDE_n_minus_m_equals_six_l3255_325585


namespace NUMINAMATH_CALUDE_gcd_and_prime_check_l3255_325510

theorem gcd_and_prime_check : 
  (Nat.gcd 7854 15246 = 6) ∧ ¬(Nat.Prime 6) := by sorry

end NUMINAMATH_CALUDE_gcd_and_prime_check_l3255_325510


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l3255_325562

theorem decimal_arithmetic : 0.5 - 0.03 + 0.007 = 0.477 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l3255_325562


namespace NUMINAMATH_CALUDE_mersenne_fermat_prime_composite_l3255_325524

theorem mersenne_fermat_prime_composite (n : ℕ) (h : n > 2) :
  (Nat.Prime (2^n - 1) → ¬Nat.Prime (2^n + 1)) ∧
  (Nat.Prime (2^n + 1) → ¬Nat.Prime (2^n - 1)) :=
sorry

end NUMINAMATH_CALUDE_mersenne_fermat_prime_composite_l3255_325524


namespace NUMINAMATH_CALUDE_motorcyclist_wait_time_l3255_325586

/-- Given a hiker and a motorcyclist with specified speeds, prove the time it takes for the
    motorcyclist to cover the distance the hiker walks in 48 minutes. -/
theorem motorcyclist_wait_time (hiker_speed : ℝ) (motorcyclist_speed : ℝ) 
    (hiker_walk_time : ℝ) (h1 : hiker_speed = 6) (h2 : motorcyclist_speed = 30) 
    (h3 : hiker_walk_time = 48) :
    (hiker_speed * hiker_walk_time) / motorcyclist_speed = 9.6 := by
  sorry

#check motorcyclist_wait_time

end NUMINAMATH_CALUDE_motorcyclist_wait_time_l3255_325586


namespace NUMINAMATH_CALUDE_special_line_equation_l3255_325564

/-- A line passing through point M(3, -4) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The line passes through point M(3, -4)
  passes_through_M : slope * 3 + y_intercept = -4
  -- The intercepts on the coordinate axes are opposite numbers
  opposite_intercepts : (y_intercept = 0 ∧ -y_intercept / slope = 0) ∨ 
                        (y_intercept ≠ 0 ∧ -y_intercept / slope = -y_intercept)

/-- The equation of the special line is either x + y = -1 or 4x + 3y = 0 -/
theorem special_line_equation (l : SpecialLine) : 
  (l.slope = -1 ∧ l.y_intercept = -1) ∨ (l.slope = -4/3 ∧ l.y_intercept = 0) := by
  sorry

#check special_line_equation

end NUMINAMATH_CALUDE_special_line_equation_l3255_325564


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3255_325583

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -2 → b ≠ -2 → c ≠ -2 → d ≠ -2 →
  ω^4 = 1 →
  ω ≠ 1 →
  1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2 →
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) + 1 / (d + 2) = 2 := by
sorry


end NUMINAMATH_CALUDE_complex_sum_theorem_l3255_325583


namespace NUMINAMATH_CALUDE_kindergarten_card_problem_l3255_325597

/-- Represents the distribution of cards among children in a kindergarten. -/
structure CardDistribution where
  ma_three : ℕ  -- Number of children with three "MA" cards
  ma_two : ℕ    -- Number of children with two "MA" cards and one "NY" card
  ny_two : ℕ    -- Number of children with two "NY" cards and one "MA" card
  ny_three : ℕ  -- Number of children with three "NY" cards

/-- The conditions given in the problem. -/
def problem_conditions (d : CardDistribution) : Prop :=
  d.ma_three + d.ma_two = 20 ∧
  d.ny_two + d.ny_three = 30 ∧
  d.ma_two + d.ny_two = 40

/-- The theorem stating that given the problem conditions, 
    the number of children with all three cards the same is 10. -/
theorem kindergarten_card_problem (d : CardDistribution) :
  problem_conditions d → d.ma_three + d.ny_three = 10 := by
  sorry


end NUMINAMATH_CALUDE_kindergarten_card_problem_l3255_325597


namespace NUMINAMATH_CALUDE_perfume_tax_rate_l3255_325572

theorem perfume_tax_rate (price_before_tax : ℝ) (total_price : ℝ) : price_before_tax = 92 → total_price = 98.90 → (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_perfume_tax_rate_l3255_325572


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_ratio_range_l3255_325537

theorem triangle_geometric_sequence_ratio_range (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b^2 = a*c) : 2 ≤ (b/a + a/b) ∧ (b/a + a/b) < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_ratio_range_l3255_325537


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_even_numbers_l3255_325555

theorem smallest_of_three_consecutive_even_numbers (a b c : ℕ) : 
  (∃ n : ℕ, a = 2 * n ∧ b = 2 * n + 2 ∧ c = 2 * n + 4) →  -- consecutive even numbers
  a + b + c = 162 →                                      -- sum is 162
  a = 52 :=                                              -- smallest number is 52
by sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_even_numbers_l3255_325555


namespace NUMINAMATH_CALUDE_january_oil_bill_l3255_325528

theorem january_oil_bill (january_bill february_bill : ℚ) : 
  (february_bill / january_bill = 3 / 2) →
  ((february_bill + 20) / january_bill = 5 / 3) →
  january_bill = 120 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l3255_325528


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3255_325534

/-- Calculates the final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

/-- Theorem stating that given the specific conditions, the final stock price is $151.2 -/
theorem stock_price_calculation :
  final_stock_price 120 0.8 0.3 = 151.2 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3255_325534


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_13_l3255_325552

def numbers : List Nat := [45, 63, 98, 121, 169]

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def prime_factors (n : Nat) : Set Nat :=
  {p : Nat | is_prime p ∧ n % p = 0}

theorem largest_prime_factor_is_13 :
  ∃ (n : Nat), n ∈ numbers ∧ 13 ∈ prime_factors n ∧
  ∀ (m : Nat), m ∈ numbers → ∀ (p : Nat), p ∈ prime_factors m → p ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_is_13_l3255_325552


namespace NUMINAMATH_CALUDE_emily_numbers_l3255_325588

theorem emily_numbers (n : ℕ) : 
  (n % 5 = 0 ∧ n % 10 = 0) → 
  (∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ n / 10 % 10 = d) →
  (∃ count : ℕ, count = 9 ∧ 
    ∀ d : ℕ, d < 10 ∧ d ≠ 0 → 
    ∃ m : ℕ, m % 5 = 0 ∧ m % 10 = 0 ∧ m / 10 % 10 = d) :=
by
  sorry

#check emily_numbers

end NUMINAMATH_CALUDE_emily_numbers_l3255_325588


namespace NUMINAMATH_CALUDE_supply_lasts_18_months_l3255_325541

/-- Represents the number of pills in a supply -/
def supply : ℕ := 60

/-- Represents the fraction of a pill taken per dose -/
def dose : ℚ := 1 / 3

/-- Represents the number of days between doses -/
def days_between_doses : ℕ := 3

/-- Represents the average number of days in a month -/
def days_per_month : ℕ := 30

/-- Calculates the number of months a supply of medicine will last -/
def months_supply_lasts : ℚ :=
  (supply : ℚ) * (days_between_doses : ℚ) / dose / days_per_month

theorem supply_lasts_18_months :
  months_supply_lasts = 18 := by sorry

end NUMINAMATH_CALUDE_supply_lasts_18_months_l3255_325541


namespace NUMINAMATH_CALUDE_charging_bull_rounds_in_hour_l3255_325546

/-- The time in seconds for the racing magic to circle the track once -/
def racing_magic_time : ℕ := 60

/-- The time in minutes when they meet at the starting point for the second time -/
def meeting_time : ℕ := 6

/-- The number of rounds the charging bull makes in an hour -/
def charging_bull_rounds : ℕ := 70

/-- Theorem stating that the charging bull makes 70 rounds in an hour -/
theorem charging_bull_rounds_in_hour :
  charging_bull_rounds = 70 := by sorry

end NUMINAMATH_CALUDE_charging_bull_rounds_in_hour_l3255_325546


namespace NUMINAMATH_CALUDE_polygon_sides_l3255_325521

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → ∃ n : ℕ, n = 8 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3255_325521


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_negative_i_l3255_325584

theorem imaginary_sum_equals_negative_i (i : ℂ) (hi : i^2 = -1) :
  i^11 + i^16 + i^21 + i^26 + i^31 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_negative_i_l3255_325584


namespace NUMINAMATH_CALUDE_probability_even_balls_correct_l3255_325508

def probability_even_balls (n : ℕ) : ℚ :=
  1/2 - 1/(2*(2^n - 1))

theorem probability_even_balls_correct (n : ℕ) :
  probability_even_balls n = 1/2 - 1/(2*(2^n - 1)) :=
sorry

end NUMINAMATH_CALUDE_probability_even_balls_correct_l3255_325508


namespace NUMINAMATH_CALUDE_stating_min_connections_for_given_problem_l3255_325522

/-- Represents the number of cities -/
def num_cities : Nat := 100

/-- Represents the number of different routes -/
def num_routes : Nat := 1000

/-- 
Given a number of cities and a number of routes, 
calculates the minimum number of flight connections per city 
that allows for the specified number of routes.
-/
def min_connections (cities : Nat) (routes : Nat) : Nat :=
  sorry

/-- 
Theorem stating that given 100 cities and 1000 routes, 
the minimum number of connections per city is 4.
-/
theorem min_connections_for_given_problem : 
  min_connections num_cities num_routes = 4 := by sorry

end NUMINAMATH_CALUDE_stating_min_connections_for_given_problem_l3255_325522


namespace NUMINAMATH_CALUDE_twentieth_digit_of_half_power_twenty_l3255_325512

theorem twentieth_digit_of_half_power_twenty (n : ℕ) : n = 20 → 
  ∃ (x : ℚ), x = (1/2)^20 ∧ 
  (∃ (a b : ℕ), x = a / (10^n) ∧ x < (a + 1) / (10^n) ∧ a % 10 = 1) :=
sorry

end NUMINAMATH_CALUDE_twentieth_digit_of_half_power_twenty_l3255_325512


namespace NUMINAMATH_CALUDE_existence_of_close_ratios_l3255_325582

theorem existence_of_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  |((a - b) / (c - d)) - 1| < (1 : ℝ) / 100000 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_close_ratios_l3255_325582


namespace NUMINAMATH_CALUDE_max_blue_points_max_blue_points_2016_l3255_325540

/-- Given a set of spheres, some red and some green, with blue points at each red-green contact,
    the maximum number of blue points is achieved when there are equal numbers of red and green spheres. -/
theorem max_blue_points (total_spheres : ℕ) (h_total : total_spheres = 2016) :
  ∃ (red_spheres green_spheres : ℕ),
    red_spheres + green_spheres = total_spheres ∧
    red_spheres * green_spheres ≤ (total_spheres / 2) ^ 2 :=
by sorry

/-- The maximum number of blue points for 2016 spheres is 1008^2. -/
theorem max_blue_points_2016 :
  ∃ (red_spheres green_spheres : ℕ),
    red_spheres + green_spheres = 2016 ∧
    red_spheres * green_spheres = 1008 ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_blue_points_max_blue_points_2016_l3255_325540


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3255_325594

theorem smallest_integer_solution : 
  ∃ x : ℤ, (x ≥ 0) ∧ 
    (⌊x / 8⌋ - ⌊x / 40⌋ + ⌊x / 240⌋ = 210) ∧ 
    (∀ y : ℤ, y ≥ 0 → ⌊y / 8⌋ - ⌊y / 40⌋ + ⌊y / 240⌋ = 210 → y ≥ x) ∧
    x = 2016 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3255_325594


namespace NUMINAMATH_CALUDE_chocolate_candies_cost_l3255_325595

/-- The cost of buying a specific number of chocolate candies -/
theorem chocolate_candies_cost
  (candies_per_box : ℕ)
  (cost_per_box : ℚ)
  (total_candies : ℕ)
  (h1 : candies_per_box = 30)
  (h2 : cost_per_box = 7.5)
  (h3 : total_candies = 450) :
  (total_candies / candies_per_box : ℚ) * cost_per_box = 112.5 :=
sorry

end NUMINAMATH_CALUDE_chocolate_candies_cost_l3255_325595


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3255_325526

/-- A point in a 2D Cartesian coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis. -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

/-- The theorem stating that the symmetric point of (2, -8) with respect to the y-axis is (-2, -8). -/
theorem symmetric_point_theorem :
  let A : Point := ⟨2, -8⟩
  let B : Point := ⟨-2, -8⟩
  symmetricAboutYAxis A B := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3255_325526


namespace NUMINAMATH_CALUDE_james_heavy_lifting_days_l3255_325592

/-- Calculates the number of days until James can lift heavy again after an injury. -/
def daysUntilHeavyLifting (painSubsideDays : ℕ) (healingMultiplier : ℕ) (waitAfterHealingDays : ℕ) (waitBeforeHeavyLiftingWeeks : ℕ) : ℕ :=
  let healingDays := painSubsideDays * healingMultiplier
  let totalDaysBeforeWorkout := healingDays + waitAfterHealingDays
  let waitBeforeHeavyLiftingDays := waitBeforeHeavyLiftingWeeks * 7
  totalDaysBeforeWorkout + waitBeforeHeavyLiftingDays

/-- Theorem stating that James can lift heavy again after 39 days given the specific conditions. -/
theorem james_heavy_lifting_days :
  daysUntilHeavyLifting 3 5 3 3 = 39 := by
  sorry

#eval daysUntilHeavyLifting 3 5 3 3

end NUMINAMATH_CALUDE_james_heavy_lifting_days_l3255_325592


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3255_325549

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (2 * a^2 - 7 * a + 3 = 0) →
  (2 * b^2 - 7 * b + 3 = 0) →
  (a - b)^2 = 25 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3255_325549


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3255_325519

def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem quadratic_minimum_value :
  ∀ x : ℝ, f x ≥ 1 ∧ ∃ x₀ : ℝ, f x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3255_325519


namespace NUMINAMATH_CALUDE_cosine_identity_proof_l3255_325535

theorem cosine_identity_proof : 2 * (Real.cos (15 * π / 180))^2 - Real.cos (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_proof_l3255_325535


namespace NUMINAMATH_CALUDE_cubic_sum_equals_nine_l3255_325580

theorem cubic_sum_equals_nine (a b : ℝ) 
  (h1 : a^5 - a^4*b - a^4 + a - b - 1 = 0)
  (h2 : 2*a - 3*b = 1) : 
  a^3 + b^3 = 9 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_nine_l3255_325580


namespace NUMINAMATH_CALUDE_shortest_distance_theorem_l3255_325560

theorem shortest_distance_theorem (a b c : ℝ) :
  a = 8 ∧ b = 6 ∧ c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_theorem_l3255_325560


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3255_325531

theorem trigonometric_expression_equality : 
  (Real.sin (7 * π / 180) + Real.sin (8 * π / 180) * Real.cos (15 * π / 180)) / 
  (Real.cos (7 * π / 180) - Real.sin (8 * π / 180) * Real.sin (15 * π / 180)) = 
  2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3255_325531


namespace NUMINAMATH_CALUDE_f_positive_iff_l3255_325551

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff (x : ℝ) : f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_l3255_325551


namespace NUMINAMATH_CALUDE_total_amount_after_stock_sale_l3255_325591

def initial_wallet_amount : ℝ := 300
def initial_investment : ℝ := 2000
def stock_price_increase_percentage : ℝ := 0.30

theorem total_amount_after_stock_sale :
  initial_wallet_amount +
  initial_investment * (1 + stock_price_increase_percentage) =
  2900 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_after_stock_sale_l3255_325591


namespace NUMINAMATH_CALUDE_mira_total_distance_l3255_325578

/-- Mira's jogging schedule for five days -/
structure JoggingSchedule where
  monday_speed : ℝ
  monday_time : ℝ
  tuesday_speed : ℝ
  tuesday_time : ℝ
  wednesday_speed : ℝ
  wednesday_time : ℝ
  thursday_speed : ℝ
  thursday_time : ℝ
  friday_speed : ℝ
  friday_time : ℝ

/-- Calculate the total distance jogged given a schedule -/
def total_distance (schedule : JoggingSchedule) : ℝ :=
  schedule.monday_speed * schedule.monday_time +
  schedule.tuesday_speed * schedule.tuesday_time +
  schedule.wednesday_speed * schedule.wednesday_time +
  schedule.thursday_speed * schedule.thursday_time +
  schedule.friday_speed * schedule.friday_time

/-- Mira's actual jogging schedule -/
def mira_schedule : JoggingSchedule := {
  monday_speed := 4
  monday_time := 2
  tuesday_speed := 5
  tuesday_time := 1.5
  wednesday_speed := 6
  wednesday_time := 2
  thursday_speed := 5
  thursday_time := 2.5
  friday_speed := 3
  friday_time := 1
}

/-- Theorem stating that Mira jogs a total of 43 miles in five days -/
theorem mira_total_distance : total_distance mira_schedule = 43 := by
  sorry

end NUMINAMATH_CALUDE_mira_total_distance_l3255_325578


namespace NUMINAMATH_CALUDE_symmetric_points_l3255_325558

/-- Given a point M with coordinates (x, y), this theorem proves the coordinates
    of points symmetric to M with respect to x-axis, y-axis, and origin. -/
theorem symmetric_points (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  let M_x_sym : ℝ × ℝ := (x, -y)  -- Symmetric to x-axis
  let M_y_sym : ℝ × ℝ := (-x, y)  -- Symmetric to y-axis
  let M_origin_sym : ℝ × ℝ := (-x, -y)  -- Symmetric to origin
  (M_x_sym = (x, -y)) ∧
  (M_y_sym = (-x, y)) ∧
  (M_origin_sym = (-x, -y)) := by
sorry


end NUMINAMATH_CALUDE_symmetric_points_l3255_325558


namespace NUMINAMATH_CALUDE_ratio_to_thirteen_l3255_325561

theorem ratio_to_thirteen : ∃ x : ℚ, (5 : ℚ) / 1 = x / 13 ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_thirteen_l3255_325561


namespace NUMINAMATH_CALUDE_parallelogram_height_l3255_325517

/-- Proves that the height of a parallelogram is 18 cm given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (h1 : area = 648) (h2 : base = 36) :
  area / base = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3255_325517


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3255_325575

theorem sqrt_product_plus_one : 
  Real.sqrt ((21:ℝ) * 20 * 19 * 18 + 1) = 379 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3255_325575


namespace NUMINAMATH_CALUDE_money_distribution_l3255_325559

theorem money_distribution (a b c d : ℤ) : 
  a + b + c + d = 600 →
  a + c = 200 →
  b + c = 350 →
  a + d = 300 →
  a ≥ 2 * b →
  c = 150 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l3255_325559


namespace NUMINAMATH_CALUDE_pentagon_c_y_coordinate_l3255_325529

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculates the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Checks if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- The y-coordinate of vertex C in the given pentagon is 21 -/
theorem pentagon_c_y_coordinate :
  ∀ (p : Pentagon),
    p.A = (0, 0) →
    p.B = (0, 5) →
    p.D = (5, 5) →
    p.E = (5, 0) →
    hasVerticalSymmetry p →
    pentagonArea p = 65 →
    p.C.2 = 21 := by sorry

end NUMINAMATH_CALUDE_pentagon_c_y_coordinate_l3255_325529


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3255_325539

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 3 - 2*I) : 
  z.im = -8/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3255_325539


namespace NUMINAMATH_CALUDE_library_boards_equal_l3255_325576

/-- Represents a library event (entry or exit) --/
inductive LibraryEvent
| entry : Nat → LibraryEvent
| exit : Nat → LibraryEvent

/-- The state of the library boards --/
structure LibraryState :=
  (entry_board : List Nat)
  (exit_board : List Nat)
  (current_count : Nat)

/-- Processes a single library event --/
def process_event (state : LibraryState) (event : LibraryEvent) : LibraryState :=
  match event with
  | LibraryEvent.entry n => 
      { entry_board := n :: state.entry_board,
        exit_board := state.exit_board,
        current_count := state.current_count + 1 }
  | LibraryEvent.exit n =>
      { entry_board := state.entry_board,
        exit_board := n :: state.exit_board,
        current_count := state.current_count - 1 }

/-- Processes a sequence of library events --/
def process_events (initial_state : LibraryState) (events : List LibraryEvent) : LibraryState :=
  events.foldl process_event initial_state

/-- Theorem: The entry and exit boards contain the same numbers at the end of the day --/
theorem library_boards_equal (events : List LibraryEvent) :
  let final_state := process_events { entry_board := [], exit_board := [], current_count := 0 } events
  (final_state.entry_board.toFinset = final_state.exit_board.toFinset) ∧ 
  (final_state.current_count = 0) := by
  sorry

end NUMINAMATH_CALUDE_library_boards_equal_l3255_325576


namespace NUMINAMATH_CALUDE_encoded_CDE_is_174_l3255_325598

/-- Represents the encoding of a base-6 digit --/
inductive Digit
| A | B | C | D | E | F

/-- Converts a Digit to its corresponding base-6 value --/
def digit_to_base6 : Digit → Nat
| Digit.A => 5
| Digit.B => 0
| Digit.C => 4
| Digit.D => 5
| Digit.E => 0
| Digit.F => 1

/-- Converts a base-6 number represented as a list of Digits to base-10 --/
def base6_to_base10 (digits : List Digit) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + (digit_to_base6 d) * (6^i)) 0

/-- The main theorem to prove --/
theorem encoded_CDE_is_174 :
  base6_to_base10 [Digit.C, Digit.D, Digit.E] = 174 :=
by sorry

end NUMINAMATH_CALUDE_encoded_CDE_is_174_l3255_325598


namespace NUMINAMATH_CALUDE_opposite_numbers_l3255_325513

-- Define the concept of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem opposite_numbers : are_opposite (-|-(1/100)|) (-(-1/100)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l3255_325513
