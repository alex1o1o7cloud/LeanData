import Mathlib

namespace NUMINAMATH_CALUDE_complex_power_abs_l1588_158811

theorem complex_power_abs : Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 3) ^ 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_abs_l1588_158811


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1588_158841

theorem cryptarithm_solution : 
  ∀ y : ℕ, 
    100000 ≤ y ∧ y < 1000000 →
    y * 3 = (y % 100000) * 10 + y / 100000 →
    y = 142857 ∨ y = 285714 :=
by
  sorry

#check cryptarithm_solution

end NUMINAMATH_CALUDE_cryptarithm_solution_l1588_158841


namespace NUMINAMATH_CALUDE_energy_bar_difference_l1588_158800

theorem energy_bar_difference (older younger : ℕ) 
  (h1 : older = younger + 17) : 
  (older - 3) = (younger + 3) + 11 := by
  sorry

end NUMINAMATH_CALUDE_energy_bar_difference_l1588_158800


namespace NUMINAMATH_CALUDE_monomial_properties_l1588_158847

/-- Represents a monomial of the form ax²y -/
structure Monomial where
  a : ℝ
  x : ℝ
  y : ℝ

/-- Checks if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  (m1.x ^ 2 * m1.y = m2.x ^ 2 * m2.y)

/-- Returns the coefficient of a monomial -/
def coefficient (m : Monomial) : ℝ := m.a

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := 3

theorem monomial_properties (m : Monomial) (h : m.a ≠ 0) :
  same_type m { a := -2, x := m.x, y := m.y } ∧
  coefficient m = m.a ∧
  degree m = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l1588_158847


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l1588_158833

theorem unique_solution_for_exponential_equation :
  ∀ a b p : ℕ+,
  p.val.Prime →
  (2 : ℕ)^(a : ℕ) + (p : ℕ)^(b : ℕ) = 19^(a : ℕ) →
  a = 1 ∧ b = 1 ∧ p = 17 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l1588_158833


namespace NUMINAMATH_CALUDE_selection_theorem_l1588_158883

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of books on the shelf -/
def total_books : ℕ := 10

/-- The number of books to be selected -/
def books_to_select : ℕ := 5

/-- The number of specific books that must be included -/
def specific_books : ℕ := 2

/-- The number of ways to select 5 books from 10 books, given that 2 specific books must always be included -/
def selection_ways : ℕ := binomial (total_books - specific_books) (books_to_select - specific_books)

theorem selection_theorem : selection_ways = 56 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l1588_158883


namespace NUMINAMATH_CALUDE_walter_gets_49_bananas_l1588_158851

/-- Calculates the number of bananas Walter gets when sharing with Jefferson -/
def walters_bananas (jeffersons_bananas : ℕ) : ℕ :=
  let walters_fewer := jeffersons_bananas / 4
  let walters_original := jeffersons_bananas - walters_fewer
  let total_bananas := jeffersons_bananas + walters_original
  total_bananas / 2

/-- Proves that Walter gets 49 bananas when sharing with Jefferson -/
theorem walter_gets_49_bananas :
  walters_bananas 56 = 49 := by
  sorry

end NUMINAMATH_CALUDE_walter_gets_49_bananas_l1588_158851


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1588_158888

theorem quadratic_factorization :
  ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1588_158888


namespace NUMINAMATH_CALUDE_y_minimum_value_l1588_158829

/-- The function y in terms of x, a, b, and k -/
def y (x a b k : ℝ) : ℝ := 3 * (x - a)^2 + (x - b)^2 + k * x

/-- The derivative of y with respect to x -/
def y_deriv (x a b k : ℝ) : ℝ := 8 * x - 6 * a - 2 * b + k

/-- The second derivative of y with respect to x -/
def y_second_deriv : ℝ := 8

theorem y_minimum_value (a b k : ℝ) :
  ∃ x : ℝ, y_deriv x a b k = 0 ∧
           y_second_deriv > 0 ∧
           x = (6 * a + 2 * b - k) / 8 :=
sorry

end NUMINAMATH_CALUDE_y_minimum_value_l1588_158829


namespace NUMINAMATH_CALUDE_remainder_theorem_l1588_158890

-- Define the polynomial
def p (x : ℝ) : ℝ := x^5 - x^3 + 3*x^2 + 2

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, p = (λ x => (x + 2) * q x + (-10)) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1588_158890


namespace NUMINAMATH_CALUDE_joan_seashells_l1588_158801

/-- The number of seashells Joan has after receiving some from Sam -/
def total_seashells (original : ℕ) (received : ℕ) : ℕ :=
  original + received

/-- Theorem: If Joan found 70 seashells and Sam gave her 27 seashells, 
    then Joan now has 97 seashells -/
theorem joan_seashells : total_seashells 70 27 = 97 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l1588_158801


namespace NUMINAMATH_CALUDE_sum_of_squares_and_minimum_l1588_158863

/-- Given an equation and Vieta's formulas, prove the sum of squares and its minimum value -/
theorem sum_of_squares_and_minimum (m : ℝ) (x₁ x₂ : ℝ) 
  (eq : x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂)
  (vieta1 : x₁ + x₂ = -(m + 1))
  (vieta2 : x₁ * x₂ = 2*m - 2)
  (D_nonneg : (m + 3)^2 ≥ 0) :
  (x₁^2 + x₂^2 = (m - 1)^2 + 4) ∧ 
  (∀ m', (m' - 1)^2 + 4 ≥ 4) ∧
  (∃ m₀, (m₀ - 1)^2 + 4 = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_minimum_l1588_158863


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1588_158832

/-- Parabola struct representing y^2 = ax --/
structure Parabola where
  a : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- Line struct representing y = m(x - h) + k --/
structure Line where
  m : ℝ
  h : ℝ
  k : ℝ
  eq : ∀ x y : ℝ, y = m * (x - h) + k

/-- The length of the chord AB formed by intersecting a parabola with a line --/
def chordLength (p : Parabola) (l : Line) : ℝ := sorry

theorem parabola_chord_length :
  let p : Parabola := { a := 3, eq := sorry }
  let f : ℝ × ℝ := (3/4, 0)
  let l : Line := { m := Real.sqrt 3 / 3, h := 3/4, k := 0, eq := sorry }
  chordLength p l = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1588_158832


namespace NUMINAMATH_CALUDE_carla_bug_collection_l1588_158813

theorem carla_bug_collection (leaves : ℕ) (days : ℕ) (items_per_day : ℕ) 
  (h1 : leaves = 30)
  (h2 : days = 10)
  (h3 : items_per_day = 5) :
  let total_items := days * items_per_day
  let bugs := total_items - leaves
  bugs = 20 := by sorry

end NUMINAMATH_CALUDE_carla_bug_collection_l1588_158813


namespace NUMINAMATH_CALUDE_excluded_angle_measure_l1588_158853

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: In a polygon where the sum of all interior angles except one is 1680°,
    the measure of the excluded interior angle is 120°. -/
theorem excluded_angle_measure (n : ℕ) (h : sum_interior_angles n - 120 = 1680) :
  120 = sum_interior_angles n - 1680 := by
  sorry

end NUMINAMATH_CALUDE_excluded_angle_measure_l1588_158853


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1588_158849

theorem inscribed_squares_ratio (r : ℝ) (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * a)^2 + (2 * b)^2 = r^2 ∧ 
  (a + 2*b)^2 + b^2 = r^2 → 
  a / b = 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1588_158849


namespace NUMINAMATH_CALUDE_three_digit_sum_property_divisibility_condition_l1588_158886

def three_digit_num (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem three_digit_sum_property (a b c : ℕ) 
  (ha : a ≥ 1 ∧ a ≤ 9) (hb : b ≥ 1 ∧ b ≤ 9) (hc : c ≥ 1 ∧ c ≤ 9) :
  three_digit_num a b c + three_digit_num b c a + three_digit_num c a b = 111 * (a + b + c) :=
sorry

theorem divisibility_condition (a b c : ℕ) 
  (ha : a ≥ 1 ∧ a ≤ 9) (hb : b ≥ 1 ∧ b ≤ 9) (hc : c ≥ 1 ∧ c ≤ 9) :
  (∃ k : ℕ, three_digit_num a b c + three_digit_num b c a + three_digit_num c a b = 7 * k) →
  (a + b + c = 7 ∨ a + b + c = 14 ∨ a + b + c = 21) :=
sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_divisibility_condition_l1588_158886


namespace NUMINAMATH_CALUDE_percentage_problem_l1588_158899

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 50 → 
  (0.6 * N) = ((P / 100) * 10 + 27) → 
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1588_158899


namespace NUMINAMATH_CALUDE_work_completion_proof_l1588_158819

/-- The number of days it takes a to complete the work alone -/
def a_days : ℕ := 45

/-- The number of days it takes b to complete the work alone -/
def b_days : ℕ := 40

/-- The number of days b worked alone to complete the remaining work -/
def b_remaining_days : ℕ := 23

/-- The number of days a worked before leaving -/
def days_a_worked : ℕ := 9

theorem work_completion_proof :
  let total_work := 1
  let a_rate := total_work / a_days
  let b_rate := total_work / b_days
  let combined_rate := a_rate + b_rate
  combined_rate * days_a_worked + b_rate * b_remaining_days = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l1588_158819


namespace NUMINAMATH_CALUDE_max_savings_63_l1588_158820

/-- Represents the price of a pastry package -/
structure PastryPrice where
  quantity : Nat
  price : Nat

/-- Represents the discount options for a type of pastry -/
structure PastryDiscount where
  regular_price : Nat
  discounts : List PastryPrice

/-- Calculates the minimum cost for a given quantity using available discounts -/
def min_cost (discount : PastryDiscount) (quantity : Nat) : Nat :=
  sorry

/-- Calculates the cost without any discounts -/
def regular_cost (discount : PastryDiscount) (quantity : Nat) : Nat :=
  sorry

/-- Doughnut discount options -/
def doughnut_discount : PastryDiscount :=
  { regular_price := 8,
    discounts := [
      { quantity := 12, price := 8 },
      { quantity := 24, price := 14 },
      { quantity := 48, price := 26 }
    ] }

/-- Croissant discount options -/
def croissant_discount : PastryDiscount :=
  { regular_price := 10,
    discounts := [
      { quantity := 12, price := 10 },
      { quantity := 36, price := 28 },
      { quantity := 60, price := 45 }
    ] }

/-- Muffin discount options -/
def muffin_discount : PastryDiscount :=
  { regular_price := 6,
    discounts := [
      { quantity := 12, price := 6 },
      { quantity := 24, price := 11 },
      { quantity := 72, price := 30 }
    ] }

theorem max_savings_63 :
  let doughnut_qty := 20 * 12
  let croissant_qty := 15 * 12
  let muffin_qty := 18 * 12
  let total_discounted := min_cost doughnut_discount doughnut_qty +
                          min_cost croissant_discount croissant_qty +
                          min_cost muffin_discount muffin_qty
  let total_regular := regular_cost doughnut_discount doughnut_qty +
                       regular_cost croissant_discount croissant_qty +
                       regular_cost muffin_discount muffin_qty
  total_regular - total_discounted = 63 :=
by sorry

end NUMINAMATH_CALUDE_max_savings_63_l1588_158820


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_y_values_l1588_158836

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 3), v i = k * w i

/-- Vector a defined as (1, 2, -y) -/
def a (y : ℝ) : Fin 3 → ℝ
  | 0 => 1
  | 1 => 2
  | 2 => -y
  | _ => 0

/-- Vector b defined as (x, 1, 2) -/
def b (x : ℝ) : Fin 3 → ℝ
  | 0 => x
  | 1 => 1
  | 2 => 2
  | _ => 0

/-- The sum of two vectors -/
def vec_add (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => v i + w i

/-- The scalar multiplication of a vector -/
def vec_smul (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => k * v i

theorem vector_parallel_implies_x_y_values (x y : ℝ) :
  parallel (vec_add (a y) (vec_smul 2 (b x))) (vec_add (vec_smul 2 (a y)) (vec_smul (-1) (b x))) →
  x = 1/2 ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_y_values_l1588_158836


namespace NUMINAMATH_CALUDE_min_value_expression_l1588_158816

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x) / (x + 2 * y) + y / x ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1588_158816


namespace NUMINAMATH_CALUDE_total_students_l1588_158824

/-- The number of students in different study halls --/
structure StudyHalls where
  general : ℕ
  biology : ℕ
  chemistry : ℕ
  math : ℕ
  arts : ℕ

/-- Conditions for the study halls problem --/
def study_halls_conditions (halls : StudyHalls) : Prop :=
  halls.general = 30 ∧
  halls.biology = 2 * halls.general ∧
  halls.chemistry = halls.general + 10 ∧
  halls.math = (3 * (halls.general + halls.biology + halls.chemistry)) / 5 ∧
  halls.arts * 20 / 100 = halls.general

/-- Theorem stating that the total number of students is 358 --/
theorem total_students (halls : StudyHalls) 
  (h : study_halls_conditions halls) : 
  halls.general + halls.biology + halls.chemistry + halls.math + halls.arts = 358 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l1588_158824


namespace NUMINAMATH_CALUDE_spade_equation_solution_l1588_158858

def spade (A B : ℝ) : ℝ := A^2 + 2*A*B + 3*B + 7

theorem spade_equation_solution :
  ∃ A : ℝ, spade A 5 = 97 ∧ (A = 5 ∨ A = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l1588_158858


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1588_158803

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m - 1) * (-2) - 1 + (2 * m - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1588_158803


namespace NUMINAMATH_CALUDE_sandys_money_l1588_158892

theorem sandys_money (pie_cost sandwich_cost book_cost remaining_money : ℕ) : 
  pie_cost = 6 →
  sandwich_cost = 3 →
  book_cost = 10 →
  remaining_money = 38 →
  pie_cost + sandwich_cost + book_cost + remaining_money = 57 := by
sorry

end NUMINAMATH_CALUDE_sandys_money_l1588_158892


namespace NUMINAMATH_CALUDE_three_circles_tangency_theorem_l1588_158843

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to check if two circles are tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the function to get the tangency point of two circles
def tangency_point (c1 c2 : Circle) : Point :=
  sorry

-- Define the function to check if a point is on a circle
def point_on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := c.center
  (p.x - x)^2 + (p.y - y)^2 = c.radius^2

-- Define the function to check if two points form a diameter of a circle
def is_diameter (p1 p2 : Point) (c : Circle) : Prop :=
  let (x, y) := c.center
  (p1.x + p2.x) / 2 = x ∧ (p1.y + p2.y) / 2 = y

-- Theorem statement
theorem three_circles_tangency_theorem (S1 S2 S3 : Circle) :
  are_tangent S1 S2 ∧ are_tangent S2 S3 ∧ are_tangent S3 S1 →
  let C := tangency_point S1 S2
  let A := tangency_point S2 S3
  let B := tangency_point S3 S1
  let A1 := sorry -- Intersection of line CA with S3
  let B1 := sorry -- Intersection of line CB with S3
  point_on_circle A1 S3 ∧ point_on_circle B1 S3 ∧ is_diameter A1 B1 S3 :=
sorry

end NUMINAMATH_CALUDE_three_circles_tangency_theorem_l1588_158843


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l1588_158857

/-- The general equation of a line passing through two given points. -/
theorem line_equation_through_two_points :
  ∀ (x y : ℝ), 
  (∃ (t : ℝ), x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 1 * t) ↔ 
  x - 2*y + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l1588_158857


namespace NUMINAMATH_CALUDE_min_value_problem_l1588_158844

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1/a + 2) * (1/b + 2) ≥ 16 ∧
  ((1/a + 2) * (1/b + 2) = 16 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1588_158844


namespace NUMINAMATH_CALUDE_zero_necessary_for_odd_zero_not_sufficient_for_odd_l1588_158882

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- f(0) = 0 is a necessary condition for f to be odd -/
theorem zero_necessary_for_odd (f : ℝ → ℝ) :
  IsOdd f → f 0 = 0 :=
sorry

/-- f(0) = 0 is not a sufficient condition for f to be odd -/
theorem zero_not_sufficient_for_odd :
  ∃ f : ℝ → ℝ, f 0 = 0 ∧ ¬IsOdd f :=
sorry

end NUMINAMATH_CALUDE_zero_necessary_for_odd_zero_not_sufficient_for_odd_l1588_158882


namespace NUMINAMATH_CALUDE_radius_scientific_notation_l1588_158834

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- The given radius in centimeters -/
def radius : ℝ := 0.000012

/-- The scientific notation representation of the radius -/
def radiusScientific : ScientificNotation :=
  { coefficient := 1.2
    exponent := -5
    h1 := by sorry
    h2 := by sorry }

theorem radius_scientific_notation :
  radius = radiusScientific.coefficient * (10 : ℝ) ^ radiusScientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_radius_scientific_notation_l1588_158834


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1588_158817

/-- The standard equation of a hyperbola with the same asymptotes as x²/9 - y²/16 = 1
    and passing through the point (-√3, 2√3) -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ m : ℝ, x^2 / 9 - y^2 / 16 = m) ∧
  ((-Real.sqrt 3)^2 / 9 - (2 * Real.sqrt 3)^2 / 16 = -5/12) →
  y^2 / 5 - x^2 / (15/4) = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1588_158817


namespace NUMINAMATH_CALUDE_distance_from_origin_l1588_158891

theorem distance_from_origin (x y : ℝ) (n : ℝ) : 
  y = 15 →
  (x - 2)^2 + (y - 8)^2 = 13^2 →
  x > 2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1588_158891


namespace NUMINAMATH_CALUDE_maximize_expected_score_l1588_158866

structure QuestionType where
  correct_prob : ℝ
  points : ℕ

def expected_score (first second : QuestionType) : ℝ :=
  first.correct_prob * (first.points + second.correct_prob * second.points) +
  first.correct_prob * (1 - second.correct_prob) * first.points

theorem maximize_expected_score (type_a type_b : QuestionType)
  (ha : type_a.correct_prob = 0.8)
  (hb : type_b.correct_prob = 0.6)
  (pa : type_a.points = 20)
  (pb : type_b.points = 80) :
  expected_score type_b type_a > expected_score type_a type_b :=
sorry

end NUMINAMATH_CALUDE_maximize_expected_score_l1588_158866


namespace NUMINAMATH_CALUDE_set_operations_l1588_158867

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 5}
def B : Set Nat := {3, 5, 6}

theorem set_operations :
  (A ∩ B = {3, 5}) ∧ ((U \ A) ∪ B = {3, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1588_158867


namespace NUMINAMATH_CALUDE_sallys_pens_l1588_158859

theorem sallys_pens (students : ℕ) (pens_per_student : ℕ) (pens_taken_home : ℕ) :
  students = 44 →
  pens_per_student = 7 →
  pens_taken_home = 17 →
  ∃ (initial_pens : ℕ),
    initial_pens = 342 ∧
    pens_taken_home = (initial_pens - students * pens_per_student) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sallys_pens_l1588_158859


namespace NUMINAMATH_CALUDE_square_root_of_1708249_l1588_158881

theorem square_root_of_1708249 :
  Real.sqrt 1708249 = 1307 := by sorry

end NUMINAMATH_CALUDE_square_root_of_1708249_l1588_158881


namespace NUMINAMATH_CALUDE_both_languages_students_l1588_158814

/-- The number of students taking both French and Spanish classes -/
def students_taking_both (french_class : ℕ) (spanish_class : ℕ) (total_students : ℕ) (students_one_language : ℕ) : ℕ :=
  french_class + spanish_class - total_students

theorem both_languages_students :
  let french_class : ℕ := 21
  let spanish_class : ℕ := 21
  let students_one_language : ℕ := 30
  let total_students : ℕ := students_one_language + students_taking_both french_class spanish_class total_students students_one_language
  students_taking_both french_class spanish_class total_students students_one_language = 6 := by
  sorry

end NUMINAMATH_CALUDE_both_languages_students_l1588_158814


namespace NUMINAMATH_CALUDE_removed_triangles_area_l1588_158840

theorem removed_triangles_area (r s : ℝ) : 
  (r + s)^2 + (r - s)^2 = 16^2 → 
  2 * (r^2 + s^2) = 256 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l1588_158840


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l1588_158862

/-- Given a tank that can be filled by either 18 buckets of 60 liters each or 216 buckets of unknown capacity, 
    prove that the capacity of each bucket in the second case is 5 liters. -/
theorem bucket_capacity_problem (tank_capacity : ℝ) (bucket_count_1 bucket_count_2 : ℕ) 
  (bucket_capacity_1 : ℝ) (bucket_capacity_2 : ℝ) 
  (h1 : tank_capacity = bucket_count_1 * bucket_capacity_1)
  (h2 : tank_capacity = bucket_count_2 * bucket_capacity_2)
  (h3 : bucket_count_1 = 18)
  (h4 : bucket_capacity_1 = 60)
  (h5 : bucket_count_2 = 216) :
  bucket_capacity_2 = 5 := by
  sorry

#check bucket_capacity_problem

end NUMINAMATH_CALUDE_bucket_capacity_problem_l1588_158862


namespace NUMINAMATH_CALUDE_acme_cheaper_than_beta_l1588_158876

/-- Acme's pricing function -/
def acme_price (n : ℕ) : ℕ := 45 + 10 * n

/-- Beta's pricing function -/
def beta_price (n : ℕ) : ℕ := 15 * n

/-- Beta's minimum order quantity -/
def beta_min_order : ℕ := 5

/-- The minimum number of shirts above Beta's minimum order for which Acme is cheaper -/
def min_shirts_above_min : ℕ := 5

theorem acme_cheaper_than_beta :
  ∀ n : ℕ, n ≥ beta_min_order + min_shirts_above_min →
    acme_price (beta_min_order + min_shirts_above_min) < beta_price (beta_min_order + min_shirts_above_min) ∧
    ∀ m : ℕ, m < beta_min_order + min_shirts_above_min → acme_price m ≥ beta_price m :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_than_beta_l1588_158876


namespace NUMINAMATH_CALUDE_triangle_properties_l1588_158835

open Real

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a > 0 ∧ b > 0 ∧ c > 0)
  (h6 : a * cos C + Real.sqrt 3 * a * sin C - b - c = 0)
  (h7 : b^2 + c^2 = 2 * a^2)

def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_properties (t : Triangle) :
  t.A = π / 3 ∧
  isEquilateral t ∧
  ∃ (D : ℝ × ℝ), 
    let B := (0, 0)
    let C := (t.c, 0)
    let A := (t.b * cos t.C, t.b * sin t.C)
    let AC := (A.1 - C.1, A.2 - C.2)
    let AD := (D.1 - A.1, D.2 - A.2)
    2 * (D.1 - B.1, D.2 - B.2) = (C.1 - D.1, C.2 - D.2) ∧
    (AD.1 * AC.1 + AD.2 * AC.2) / Real.sqrt (AC.1^2 + AC.2^2) = 2/3 * Real.sqrt (AC.1^2 + AC.2^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1588_158835


namespace NUMINAMATH_CALUDE_remainder_theorem_l1588_158897

-- Define the polynomial q(x)
def q (x D : ℝ) : ℝ := 2 * x^4 - 3 * x^2 + D * x + 6

-- Theorem statement
theorem remainder_theorem (D : ℝ) :
  (q 2 D = 6) → (q (-2) D = 52) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1588_158897


namespace NUMINAMATH_CALUDE_num_ways_to_sum_correct_l1588_158823

/-- The number of ways to choose k natural numbers that sum to n -/
def num_ways_to_sum (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: The number of ways to choose k natural numbers that sum to n
    is equal to (n+k-1) choose (k-1) -/
theorem num_ways_to_sum_correct (n k : ℕ) :
  num_ways_to_sum n k = Nat.choose (n + k - 1) (k - 1) := by
  sorry

#check num_ways_to_sum_correct

end NUMINAMATH_CALUDE_num_ways_to_sum_correct_l1588_158823


namespace NUMINAMATH_CALUDE_work_time_ratio_l1588_158872

/-- The time it takes for Dev and Tina to complete the task together -/
def T : ℝ := 10

/-- The time it takes for Dev to complete the task alone -/
def dev_time : ℝ := T + 20

/-- The time it takes for Tina to complete the task alone -/
def tina_time : ℝ := T + 5

/-- The time it takes for Alex to complete the task alone -/
def alex_time : ℝ := T + 10

/-- The ratio of time taken by Dev, Tina, and Alex working alone -/
def time_ratio : Prop :=
  ∃ (k : ℝ), k > 0 ∧ dev_time = 6 * k ∧ tina_time = 3 * k ∧ alex_time = 4 * k

theorem work_time_ratio : time_ratio := by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l1588_158872


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1588_158893

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (z.re = 0) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1588_158893


namespace NUMINAMATH_CALUDE_weight_of_other_new_member_l1588_158810

/-- Given the initial and final average weights of a group, the number of initial members,
    and the weight of one new member, calculate the weight of the other new member. -/
theorem weight_of_other_new_member
  (initial_average : ℝ)
  (final_average : ℝ)
  (initial_members : ℕ)
  (weight_of_one_new_member : ℝ)
  (h1 : initial_average = 48)
  (h2 : final_average = 51)
  (h3 : initial_members = 23)
  (h4 : weight_of_one_new_member = 78) :
  (initial_members + 2) * final_average - initial_members * initial_average - weight_of_one_new_member = 93 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_other_new_member_l1588_158810


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1588_158855

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A decagon (10-sided polygon) has 35 diagonals -/
theorem decagon_diagonals :
  num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1588_158855


namespace NUMINAMATH_CALUDE_star_equation_solution_l1588_158848

-- Define the ☆ operation
def star (a b : ℝ) : ℝ := a + b - 1

-- Theorem statement
theorem star_equation_solution :
  ∃ x : ℝ, star 2 x = 1 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1588_158848


namespace NUMINAMATH_CALUDE_second_number_in_sequence_l1588_158870

theorem second_number_in_sequence (x y z : ℝ) : 
  z = 4 * y →
  y = 2 * x →
  (x + y + z) / 3 = 165 →
  y = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_in_sequence_l1588_158870


namespace NUMINAMATH_CALUDE_ninety_mile_fare_l1588_158807

/-- Represents the fare structure for a taxi ride -/
structure TaxiFare where
  baseFare : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.baseFare + tf.ratePerMile * distance

theorem ninety_mile_fare :
  ∃ (tf : TaxiFare),
    tf.baseFare = 30 ∧
    totalFare tf 60 = 150 ∧
    totalFare tf 90 = 210 := by
  sorry

end NUMINAMATH_CALUDE_ninety_mile_fare_l1588_158807


namespace NUMINAMATH_CALUDE_common_ratio_is_three_l1588_158822

/-- An arithmetic-geometric sequence with its properties -/
structure ArithGeomSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  q : ℝ      -- Common ratio
  h1 : a 3 = 2 * S 2 + 1
  h2 : a 4 = 2 * S 3 + 1
  h3 : ∀ n : ℕ, n ≥ 2 → a (n+1) = q * a n

/-- The common ratio of the arithmetic-geometric sequence is 3 -/
theorem common_ratio_is_three (seq : ArithGeomSeq) : seq.q = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_three_l1588_158822


namespace NUMINAMATH_CALUDE_tangent_iff_k_eq_zero_l1588_158869

/-- A line with equation x - ky - 1 = 0 -/
structure Line (k : ℝ) where
  equation : ∀ x y : ℝ, x - k * y - 1 = 0

/-- A circle with center (2,1) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}

/-- The line is tangent to the circle -/
def IsTangent (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ Circle ∧ p.1 - k * p.2 - 1 = 0

/-- The main theorem: k = 0 is necessary and sufficient for the line to be tangent to the circle -/
theorem tangent_iff_k_eq_zero (k : ℝ) : IsTangent k ↔ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_iff_k_eq_zero_l1588_158869


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l1588_158871

/-- Given two points (4,3) and (2,-3) on a coordinate plane, this theorem proves:
    1. The direct distance between them is 2√10
    2. The horizontal distance between them is 2
    3. The ratio of the horizontal distance to the direct distance is not an integer -/
theorem distance_ratio_theorem :
  let p1 : ℝ × ℝ := (4, 3)
  let p2 : ℝ × ℝ := (2, -3)
  let direct_distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let horizontal_distance := |p1.1 - p2.1|
  let ratio := horizontal_distance / direct_distance
  (direct_distance = 2 * Real.sqrt 10) ∧
  (horizontal_distance = 2) ∧
  ¬(∃ n : ℤ, ratio = n) :=
by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l1588_158871


namespace NUMINAMATH_CALUDE_survey_order_correct_l1588_158854

-- Define the steps of the survey process
inductive SurveyStep
  | CollectData
  | OrganizeData
  | DrawPieChart
  | AnalyzeData

-- Define a function to represent the correct order of steps
def correctOrder : List SurveyStep :=
  [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeData]

-- Define a function to check if a given order is correct
def isCorrectOrder (order : List SurveyStep) : Prop :=
  order = correctOrder

-- Theorem stating that the given order is correct
theorem survey_order_correct :
  isCorrectOrder [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeData] :=
by sorry

end NUMINAMATH_CALUDE_survey_order_correct_l1588_158854


namespace NUMINAMATH_CALUDE_washer_dryer_cost_difference_l1588_158830

theorem washer_dryer_cost_difference :
  ∀ (washer_cost dryer_cost : ℝ),
    dryer_cost = 490 →
    washer_cost > dryer_cost →
    washer_cost + dryer_cost = 1200 →
    washer_cost - dryer_cost = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_difference_l1588_158830


namespace NUMINAMATH_CALUDE_prince_total_spent_prince_total_spent_proof_l1588_158809

-- Define the total number of CDs
def total_cds : ℕ := 200

-- Define the percentage of CDs that cost $10
def percentage_expensive : ℚ := 40 / 100

-- Define the cost of expensive CDs
def cost_expensive : ℕ := 10

-- Define the cost of cheap CDs
def cost_cheap : ℕ := 5

-- Define the fraction of expensive CDs Prince bought
def fraction_bought : ℚ := 1 / 2

-- Theorem to prove
theorem prince_total_spent (total_cds : ℕ) (percentage_expensive : ℚ) 
  (cost_expensive cost_cheap : ℕ) (fraction_bought : ℚ) : ℕ :=
  -- The total amount Prince spent on CDs
  1000

-- Proof of the theorem
theorem prince_total_spent_proof :
  prince_total_spent total_cds percentage_expensive cost_expensive cost_cheap fraction_bought = 1000 := by
  sorry

end NUMINAMATH_CALUDE_prince_total_spent_prince_total_spent_proof_l1588_158809


namespace NUMINAMATH_CALUDE_series_sum_210_l1588_158818

def series_sum (n : ℕ) : ℤ :=
  let groups := n / 3
  let last_term := 3 * (groups - 1)
  (groups : ℤ) * last_term / 2

theorem series_sum_210 :
  series_sum 210 = 7245 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_210_l1588_158818


namespace NUMINAMATH_CALUDE_number_problem_l1588_158868

theorem number_problem (x : ℝ) : 0.4 * x = 0.2 * 650 + 190 ↔ x = 800 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1588_158868


namespace NUMINAMATH_CALUDE_abs_a_pow_b_eq_one_l1588_158865

/-- Given that (2a+b-1)^2 + |a-b+4| = 0, prove that |a^b| = 1 -/
theorem abs_a_pow_b_eq_one (a b : ℝ) 
  (h : (2*a + b - 1)^2 + |a - b + 4| = 0) : 
  |a^b| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_pow_b_eq_one_l1588_158865


namespace NUMINAMATH_CALUDE_badminton_tournament_l1588_158828

theorem badminton_tournament (n : ℕ) : 
  (∃ (x : ℕ), 
    (5 * n * (5 * n - 1)) / 2 = 7 * x ∧ 
    4 * x = (2 * n * (2 * n - 1)) / 2 + 2 * n * 3 * n) → 
  n = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_badminton_tournament_l1588_158828


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1588_158898

theorem weight_of_replaced_person 
  (n : ℕ) 
  (average_increase : ℝ) 
  (new_person_weight : ℝ) : 
  n = 10 → 
  average_increase = 3.2 → 
  new_person_weight = 97 → 
  (n : ℝ) * average_increase = new_person_weight - 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1588_158898


namespace NUMINAMATH_CALUDE_equation_solution_l1588_158894

theorem equation_solution : 
  ∃ (x : ℝ), 
    x ≠ (3/2) ∧ 
    (5 - 3*x = 1) ∧
    ((1 + 1/(1 + 1/(1 + 1/(2*x - 3)))) = 1/(x - 1)) ∧
    x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1588_158894


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1588_158856

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n-1)

theorem geometric_sequence_sum (a r : ℝ) (h1 : a < 0) :
  let seq := geometric_sequence a r
  (seq 2 * seq 4 + 2 * seq 3 * seq 5 + seq 4 * seq 6 = 36) →
  (seq 3 + seq 5 = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1588_158856


namespace NUMINAMATH_CALUDE_max_monthly_profit_l1588_158805

/-- Represents the monthly profit function for Xiao Ming's eye-protecting desk lamp business. -/
def monthly_profit (x : ℝ) : ℝ := -10 * x^2 + 700 * x - 10000

/-- Represents the monthly sales volume function. -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 500

/-- The cost price of each lamp. -/
def cost_price : ℝ := 20

/-- The maximum allowed profit percentage. -/
def max_profit_percentage : ℝ := 0.6

/-- Theorem stating the maximum monthly profit and the corresponding selling price. -/
theorem max_monthly_profit :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    max_profit = 2160 ∧
    optimal_price = 32 ∧
    (∀ x : ℝ, cost_price ≤ x ∧ x ≤ cost_price * (1 + max_profit_percentage) →
      monthly_profit x ≤ max_profit) ∧
    monthly_profit optimal_price = max_profit :=
  sorry

/-- Lemma: The monthly profit function is correctly defined based on the given conditions. -/
lemma profit_function_correct :
  ∀ x : ℝ, monthly_profit x = (x - cost_price) * sales_volume x :=
  sorry

/-- Lemma: The selling price is within the specified range. -/
lemma selling_price_range :
  ∀ x : ℝ, monthly_profit x > 0 → cost_price ≤ x ∧ x ≤ cost_price * (1 + max_profit_percentage) :=
  sorry

end NUMINAMATH_CALUDE_max_monthly_profit_l1588_158805


namespace NUMINAMATH_CALUDE_largest_angle_cosine_l1588_158896

theorem largest_angle_cosine (t : ℝ) (h : t > 1) :
  let a := t^2 + t + 1
  let b := t^2 - 1
  let c := 2*t + 1
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a^2 + b^2 - c^2) / (2*a*b) = -1/2 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_l1588_158896


namespace NUMINAMATH_CALUDE_expression_factorization_l1588_158861

theorem expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1588_158861


namespace NUMINAMATH_CALUDE_total_ways_eq_19_l1588_158875

/-- The number of direct bus services from place A to place B -/
def direct_services : ℕ := 4

/-- The number of bus services from place A to place C -/
def services_A_to_C : ℕ := 5

/-- The number of bus services from place C to place B -/
def services_C_to_B : ℕ := 3

/-- The total number of ways to travel from place A to place B -/
def total_ways : ℕ := direct_services + services_A_to_C * services_C_to_B

theorem total_ways_eq_19 : total_ways = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_eq_19_l1588_158875


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_l1588_158825

/-- A function f(x) with an extremum at x = 1 and f(1) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2 (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 2 := by
  sorry

#check extremum_implies_f_2

end NUMINAMATH_CALUDE_extremum_implies_f_2_l1588_158825


namespace NUMINAMATH_CALUDE_circle_diameter_problem_l1588_158806

/-- Given two circles A and B where A is inside B, proves that the diameter of A
    satisfies the given conditions. -/
theorem circle_diameter_problem (center_distance : ℝ) (diameter_B : ℝ) :
  center_distance = 5 →
  diameter_B = 20 →
  let radius_B := diameter_B / 2
  let area_B := π * radius_B ^ 2
  ∃ (radius_A : ℝ),
    π * radius_A ^ 2 * 6 = area_B ∧
    (2 * radius_A : ℝ) = 2 * Real.sqrt (50 / 3) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_problem_l1588_158806


namespace NUMINAMATH_CALUDE_total_dividend_income_l1588_158838

-- Define the investments for each stock
def investment_A : ℕ := 2000
def investment_B : ℕ := 2500
def investment_C : ℕ := 1500
def investment_D : ℕ := 2000
def investment_E : ℕ := 2000

-- Define the dividend yields for each stock for each year
def yield_A : Fin 3 → ℚ
  | 0 => 5/100
  | 1 => 4/100
  | 2 => 3/100

def yield_B : Fin 3 → ℚ
  | 0 => 3/100
  | 1 => 5/100
  | 2 => 4/100

def yield_C : Fin 3 → ℚ
  | 0 => 4/100
  | 1 => 6/100
  | 2 => 4/100

def yield_D : Fin 3 → ℚ
  | 0 => 6/100
  | 1 => 3/100
  | 2 => 5/100

def yield_E : Fin 3 → ℚ
  | 0 => 2/100
  | 1 => 7/100
  | 2 => 6/100

-- Calculate the total dividend income for a single stock over 3 years
def total_dividend (investment : ℕ) (yield : Fin 3 → ℚ) : ℚ :=
  (yield 0 * investment) + (yield 1 * investment) + (yield 2 * investment)

-- Theorem: The total dividend income from all stocks over 3 years is 1330
theorem total_dividend_income :
  total_dividend investment_A yield_A +
  total_dividend investment_B yield_B +
  total_dividend investment_C yield_C +
  total_dividend investment_D yield_D +
  total_dividend investment_E yield_E = 1330 := by
  sorry


end NUMINAMATH_CALUDE_total_dividend_income_l1588_158838


namespace NUMINAMATH_CALUDE_solve_for_R_l1588_158877

theorem solve_for_R (R : ℝ) : (R^3)^(1/4) = 64 * 4^(1/16) → R = 256 * 2^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_R_l1588_158877


namespace NUMINAMATH_CALUDE_smallest_X_value_l1588_158860

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting only of 0s and 1s that is divisible by 15 -/
def T : ℕ := 1110

/-- X is defined as T divided by 15 -/
def X : ℕ := T / 15

theorem smallest_X_value :
  (onlyZerosAndOnes T) ∧ 
  (T % 15 = 0) ∧
  (∀ n : ℕ, n < T → ¬(onlyZerosAndOnes n ∧ n % 15 = 0)) →
  X = 74 := by sorry

end NUMINAMATH_CALUDE_smallest_X_value_l1588_158860


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1588_158850

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + k * x₁ - 2 = 0) → 
  (2 * x₂^2 + k * x₂ - 2 = 0) → 
  ((x₁ - 2) * (x₂ - 2) = 10) → 
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1588_158850


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1588_158815

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : ∃ (k : ℤ), n^4 - n^2 = 12 * k ∧ ∀ (m : ℤ), (∀ (n : ℤ), ∃ (l : ℤ), n^4 - n^2 = m * l) → m ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1588_158815


namespace NUMINAMATH_CALUDE_sector_area_l1588_158808

/-- The area of a circular sector with radius 2 cm and central angle 120° is 4π/3 cm² -/
theorem sector_area (r : ℝ) (θ_deg : ℝ) (A : ℝ) : 
  r = 2 → θ_deg = 120 → A = (1/2) * r^2 * (θ_deg * π / 180) → A = (4/3) * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1588_158808


namespace NUMINAMATH_CALUDE_project_speedup_l1588_158873

/-- Calculates the number of days saved when additional workers join a project -/
def days_saved (original_workers : ℕ) (original_days : ℕ) (additional_workers : ℕ) : ℕ :=
  original_days - (original_workers * original_days) / (original_workers + additional_workers)

/-- Theorem stating that 10 additional workers save 6 days on a 12-day project with 10 original workers -/
theorem project_speedup :
  days_saved 10 12 10 = 6 := by sorry

end NUMINAMATH_CALUDE_project_speedup_l1588_158873


namespace NUMINAMATH_CALUDE_odd_function_value_l1588_158887

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = -x^3 + (a-2)x^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  -x^3 + (a-2)*x^2 + x

theorem odd_function_value (a : ℝ) :
  IsOdd (f a) → f a a = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l1588_158887


namespace NUMINAMATH_CALUDE_min_union_cardinality_l1588_158804

theorem min_union_cardinality (A B : Finset ℕ) (hA : A.card = 30) (hB : B.card = 20) :
  35 ≤ (A ∪ B).card := by sorry

end NUMINAMATH_CALUDE_min_union_cardinality_l1588_158804


namespace NUMINAMATH_CALUDE_max_take_home_pay_l1588_158831

/-- The take-home pay function for a given income x (in thousands of dollars) -/
def takehomePay (x : ℝ) : ℝ := 1000 * x - 20 * x^2

/-- The income that maximizes take-home pay -/
def maxTakeHomeIncome : ℝ := 25

theorem max_take_home_pay :
  ∀ x : ℝ, takehomePay x ≤ takehomePay maxTakeHomeIncome :=
sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l1588_158831


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1588_158837

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1588_158837


namespace NUMINAMATH_CALUDE_daltons_uncle_gift_l1588_158880

/-- The amount of money Dalton's uncle gave him -/
def uncles_gift (jump_rope_cost board_game_cost playground_ball_cost savings needed : ℕ) : ℕ :=
  jump_rope_cost + board_game_cost + playground_ball_cost - savings - needed

theorem daltons_uncle_gift :
  uncles_gift 7 12 4 6 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_daltons_uncle_gift_l1588_158880


namespace NUMINAMATH_CALUDE_unique_f_zero_unique_solution_l1588_158878

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2

/-- The theorem stating that f(0) = 1 is the only valid solution -/
theorem unique_f_zero (f : ℝ → ℝ) (h : FunctionalEq f) : f 0 = 1 := by
  sorry

/-- The theorem stating that f(x) = x² + 1 is the unique solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEq f) : 
  ∀ x : ℝ, f x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_zero_unique_solution_l1588_158878


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1588_158812

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

/-- The property that 2a_2 + a_3 = a_4 for a geometric sequence -/
def property1 (a q : ℝ) : Prop :=
  2 * (geometric_sequence a q 2) + (geometric_sequence a q 3) = geometric_sequence a q 4

/-- The property that (a_2 + 1)(a_3 + 1) = a_5 - 1 for a geometric sequence -/
def property2 (a q : ℝ) : Prop :=
  (geometric_sequence a q 2 + 1) * (geometric_sequence a q 3 + 1) = geometric_sequence a q 5 - 1

/-- Theorem stating that for a geometric sequence satisfying both properties, a_1 ≠ 2 -/
theorem geometric_sequence_property (a q : ℝ) (h1 : property1 a q) (h2 : property2 a q) : a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1588_158812


namespace NUMINAMATH_CALUDE_coastal_village_population_l1588_158827

theorem coastal_village_population (total_population : ℕ) 
  (h1 : total_population = 540) 
  (h2 : ∃ (part_size : ℕ), 4 * part_size = total_population) 
  (h3 : ∃ (male_population : ℕ), male_population = 2 * (total_population / 4)) :
  ∃ (male_population : ℕ), male_population = 270 := by
sorry

end NUMINAMATH_CALUDE_coastal_village_population_l1588_158827


namespace NUMINAMATH_CALUDE_fraction_comparison_l1588_158839

theorem fraction_comparison : 
  (111110 : ℚ) / 111111 < (333331 : ℚ) / 333334 ∧ (333331 : ℚ) / 333334 < (222221 : ℚ) / 222223 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1588_158839


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l1588_158895

theorem polynomial_subtraction :
  let p₁ : Polynomial ℝ := X^5 - 3*X^4 + X^2 + 15
  let p₂ : Polynomial ℝ := 2*X^5 - 3*X^3 + 2*X^2 + 18
  p₁ - p₂ = -X^5 - 3*X^4 + 3*X^3 - X^2 - 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l1588_158895


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l1588_158885

theorem factorial_ratio_simplification (N : ℕ) :
  (Nat.factorial (N + 1) * N) / Nat.factorial (N + 2) = N / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l1588_158885


namespace NUMINAMATH_CALUDE_total_amount_l1588_158802

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℚ
  y : ℚ
  z : ℚ

/-- The problem setup -/
def problem_setup (s : ShareDistribution) : Prop :=
  s.y = 18 ∧ s.y = 0.45 * s.x ∧ s.z = 0.3 * s.x

/-- The theorem statement -/
theorem total_amount (s : ShareDistribution) : 
  problem_setup s → s.x + s.y + s.z = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_l1588_158802


namespace NUMINAMATH_CALUDE_monomial_degree_6_l1588_158846

def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

theorem monomial_degree_6 (a : ℕ) : 
  monomial_degree 2 a = 6 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_degree_6_l1588_158846


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1588_158889

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  (3 + 2*i) / i = Complex.mk 2 (-3) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1588_158889


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1588_158852

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x^2 - 2*x < 0 → 0 < x ∧ x < 4) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < 4 ∧ x^2 - 2*x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1588_158852


namespace NUMINAMATH_CALUDE_union_when_m_neg_one_subset_condition_l1588_158884

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for part 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Theorem for part 2
theorem subset_condition (m : ℝ) :
  A ⊆ B m ↔ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_union_when_m_neg_one_subset_condition_l1588_158884


namespace NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l1588_158842

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 45

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the percentage of work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_37_5 : meeting_percentage = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l1588_158842


namespace NUMINAMATH_CALUDE_paint_color_combinations_l1588_158864

theorem paint_color_combinations (n : ℕ) (h : n = 9) : 
  (n - 1 : ℕ) = 8 := by sorry

end NUMINAMATH_CALUDE_paint_color_combinations_l1588_158864


namespace NUMINAMATH_CALUDE_penguin_count_l1588_158874

theorem penguin_count (zebras tigers zookeepers : ℕ) 
  (h1 : zebras = 22)
  (h2 : tigers = 8)
  (h3 : zookeepers = 12)
  (h4 : ∀ (penguins : ℕ), 
    (penguins + zebras + tigers + zookeepers) + 132 = 
    4 * penguins + 4 * zebras + 4 * tigers + 2 * zookeepers) :
  ∃ (penguins : ℕ), penguins = 10 := by
sorry

end NUMINAMATH_CALUDE_penguin_count_l1588_158874


namespace NUMINAMATH_CALUDE_visit_neither_country_l1588_158845

theorem visit_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 50 →
  iceland = 25 →
  norway = 23 →
  both = 21 →
  total - (iceland + norway - both) = 23 := by
  sorry

end NUMINAMATH_CALUDE_visit_neither_country_l1588_158845


namespace NUMINAMATH_CALUDE_exam_max_marks_calculation_l1588_158826

/-- Represents the maximum marks and passing criteria for a subject -/
structure Subject where
  max_marks : ℕ
  passing_percentage : ℚ

/-- Represents a student's performance in a subject -/
structure Performance where
  score : ℕ
  failed_by : ℕ

/-- Calculates the maximum marks for a subject given the performance and passing criteria -/
def calculate_max_marks (perf : Performance) (pass_percentage : ℚ) : ℕ :=
  ((perf.score + perf.failed_by : ℚ) / pass_percentage).ceil.toNat

theorem exam_max_marks_calculation (math science english : Subject) 
    (math_perf science_perf english_perf : Performance) : 
    math.max_marks = 275 ∧ science.max_marks = 414 ∧ english.max_marks = 300 :=
  by
    have h_math : math.passing_percentage = 2/5 := by sorry
    have h_science : science.passing_percentage = 7/20 := by sorry
    have h_english : english.passing_percentage = 3/10 := by sorry
    
    have h_math_perf : math_perf = ⟨90, 20⟩ := by sorry
    have h_science_perf : science_perf = ⟨110, 35⟩ := by sorry
    have h_english_perf : english_perf = ⟨80, 10⟩ := by sorry
    
    have h_math_max : math.max_marks = calculate_max_marks math_perf math.passing_percentage := by sorry
    have h_science_max : science.max_marks = calculate_max_marks science_perf science.passing_percentage := by sorry
    have h_english_max : english.max_marks = calculate_max_marks english_perf english.passing_percentage := by sorry
    
    sorry

end NUMINAMATH_CALUDE_exam_max_marks_calculation_l1588_158826


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1588_158879

theorem rational_equation_solution :
  let x : ℚ := -26/9
  (2*x + 18) / (x - 6) = (2*x - 4) / (x + 10) := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1588_158879


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1588_158821

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) * z = 1 + I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1588_158821
