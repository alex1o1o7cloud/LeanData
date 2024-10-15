import Mathlib

namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l4038_403895

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -2
  let y : ℝ := 3
  second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l4038_403895


namespace NUMINAMATH_CALUDE_remainder_of_P_div_Q_l4038_403847

/-- P(x) is a polynomial defined as x^(6n) + x^(5n) + x^(4n) + x^(3n) + x^(2n) + x^n + 1 -/
def P (x n : ℕ) : ℕ := x^(6*n) + x^(5*n) + x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1

/-- Q(x) is a polynomial defined as x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 -/
def Q (x : ℕ) : ℕ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- Theorem stating that the remainder of P(x) divided by Q(x) is 7 when n is a multiple of 7 -/
theorem remainder_of_P_div_Q (x n : ℕ) (h : ∃ k, n = 7 * k) :
  P x n % Q x = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_of_P_div_Q_l4038_403847


namespace NUMINAMATH_CALUDE_fraction_simplification_l4038_403877

theorem fraction_simplification : (210 : ℚ) / 21 * 7 / 98 * 6 / 4 = 15 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4038_403877


namespace NUMINAMATH_CALUDE_brothers_selection_probability_l4038_403800

theorem brothers_selection_probability
  (prob_X_initial : ℚ) (prob_Y_initial : ℚ)
  (prob_X_interview : ℚ) (prob_X_test : ℚ)
  (prob_Y_interview : ℚ) (prob_Y_test : ℚ)
  (h1 : prob_X_initial = 1 / 7)
  (h2 : prob_Y_initial = 2 / 5)
  (h3 : prob_X_interview = 3 / 4)
  (h4 : prob_X_test = 4 / 9)
  (h5 : prob_Y_interview = 5 / 8)
  (h6 : prob_Y_test = 7 / 10) :
  prob_X_initial * prob_X_interview * prob_X_test *
  prob_Y_initial * prob_Y_interview * prob_Y_test = 7 / 840 := by
  sorry

end NUMINAMATH_CALUDE_brothers_selection_probability_l4038_403800


namespace NUMINAMATH_CALUDE_omega_roots_quadratic_equation_l4038_403840

theorem omega_roots_quadratic_equation :
  ∀ (ω : ℂ) (α β : ℂ),
    ω^5 = 1 →
    ω ≠ 1 →
    α = ω + ω^2 →
    β = ω^3 + ω^4 →
    ∃ (a b : ℝ), ∀ (x : ℂ), x = α ∨ x = β → x^2 + a*x + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_omega_roots_quadratic_equation_l4038_403840


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l4038_403874

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 + 1 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l4038_403874


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4038_403890

theorem trigonometric_identity (α : Real) 
  (h : (1 + Real.sin α) * (1 - Real.cos α) = 1) : 
  (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4038_403890


namespace NUMINAMATH_CALUDE_parallelogram_slant_height_l4038_403813

/-- Given a rectangle and a shape composed of an isosceles triangle and a parallelogram,
    prove that the slant height of the parallelogram is approximately 8.969 inches
    when the areas are equal. -/
theorem parallelogram_slant_height (rectangle_length rectangle_width triangle_base triangle_height parallelogram_base parallelogram_height : ℝ) 
  (h_rectangle_length : rectangle_length = 5)
  (h_rectangle_width : rectangle_width = 24)
  (h_triangle_base : triangle_base = 12)
  (h_parallelogram_base : parallelogram_base = 12)
  (h_equal_heights : triangle_height = parallelogram_height)
  (h_equal_areas : rectangle_length * rectangle_width = 
    (1/2 * triangle_base * triangle_height) + (parallelogram_base * parallelogram_height)) :
  ∃ (slant_height : ℝ), abs (slant_height - 8.969) < 0.001 ∧ 
    slant_height^2 = parallelogram_height^2 + (parallelogram_base/2)^2 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_slant_height_l4038_403813


namespace NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_smallest_n_is_two_l4038_403856

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 721 * n ≡ 1137 * n [ZMOD 30] → n ≥ 2 :=
sorry

theorem two_satisfies_congruence : 721 * 2 ≡ 1137 * 2 [ZMOD 30] :=
sorry

theorem smallest_n_is_two : 
  ∃ (n : ℕ), n > 0 ∧ 721 * n ≡ 1137 * n [ZMOD 30] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 721 * m ≡ 1137 * m [ZMOD 30] → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_smallest_n_is_two_l4038_403856


namespace NUMINAMATH_CALUDE_final_milk_composition_l4038_403887

/-- The percentage of milk remaining after each replacement operation -/
def replacement_factor : ℝ := 0.7

/-- The number of replacement operations performed -/
def num_operations : ℕ := 5

/-- The final percentage of milk in the container after all operations -/
def final_milk_percentage : ℝ := replacement_factor ^ num_operations * 100

/-- Theorem stating the final percentage of milk after the operations -/
theorem final_milk_composition :
  ∃ ε > 0, |final_milk_percentage - 16.807| < ε :=
sorry

end NUMINAMATH_CALUDE_final_milk_composition_l4038_403887


namespace NUMINAMATH_CALUDE_overlap_ratio_l4038_403873

theorem overlap_ratio (circle_area square_area overlap_area : ℝ) 
  (h1 : overlap_area = 0.5 * circle_area)
  (h2 : overlap_area = 0.25 * square_area) :
  (square_area - overlap_area) / (circle_area + square_area - overlap_area) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_overlap_ratio_l4038_403873


namespace NUMINAMATH_CALUDE_complex_real_roots_relationship_l4038_403892

theorem complex_real_roots_relationship (a : ℝ) : 
  ¬(∀ x : ℂ, x^2 + a*x - a = 0 → (∀ y : ℝ, y^2 - a*y + a ≠ 0)) ∧
  ¬(∀ y : ℝ, y^2 - a*y + a = 0 → (∀ x : ℂ, x^2 + a*x - a ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_complex_real_roots_relationship_l4038_403892


namespace NUMINAMATH_CALUDE_log_equation_solution_l4038_403833

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 2 + Real.log x / Real.log 8 = 5 → x = 2^(15/4) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4038_403833


namespace NUMINAMATH_CALUDE_num_tilings_div_by_eight_l4038_403803

/-- A tromino is an L-shaped tile covering exactly three cells -/
structure Tromino :=
  (shape : List (Int × Int))
  (shape_size : shape.length = 3)

/-- A tiling of a square grid using trominos -/
def Tiling (n : Nat) := List (List (Option Tromino))

/-- The size of the square grid -/
def gridSize : Nat := 999

/-- The number of distinct tilings of an n x n grid using trominos -/
def numDistinctTilings (n : Nat) : Nat :=
  sorry

/-- Theorem: The number of distinct tilings of a 999x999 grid using trominos is divisible by 8 -/
theorem num_tilings_div_by_eight :
  ∃ k : Nat, numDistinctTilings gridSize = 8 * k :=
sorry

end NUMINAMATH_CALUDE_num_tilings_div_by_eight_l4038_403803


namespace NUMINAMATH_CALUDE_largest_prime_diff_126_l4038_403858

/-- Two natural numbers are different if they are not equal -/
def different (a b : ℕ) : Prop := a ≠ b

/-- A natural number is even if it's divisible by 2 -/
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- The largest prime difference for 126 -/
theorem largest_prime_diff_126 : 
  ∃ (p q : ℕ), 
    Prime p ∧ 
    Prime q ∧ 
    different p q ∧
    p + q = 126 ∧
    even 126 ∧ 
    126 > 7 ∧
    ∀ (r s : ℕ), Prime r → Prime s → different r s → r + s = 126 → s - r ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_diff_126_l4038_403858


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l4038_403832

/-- A quadratic equation is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 3x - 2 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l4038_403832


namespace NUMINAMATH_CALUDE_elongation_rate_improved_l4038_403860

def elongation_rate_comparison (x y : Fin 10 → ℝ) : Prop :=
  let z : Fin 10 → ℝ := fun i => x i - y i
  let z_mean : ℝ := (Finset.sum Finset.univ (fun i => z i)) / 10
  let z_variance : ℝ := (Finset.sum Finset.univ (fun i => (z i - z_mean)^2)) / 10
  z_mean = 11 ∧ 
  z_variance = 61 ∧ 
  z_mean ≥ 2 * Real.sqrt (z_variance / 10)

theorem elongation_rate_improved (x y : Fin 10 → ℝ) 
  (h : elongation_rate_comparison x y) : 
  ∃ (z_mean z_variance : ℝ), 
    z_mean = 11 ∧ 
    z_variance = 61 ∧ 
    z_mean ≥ 2 * Real.sqrt (z_variance / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_elongation_rate_improved_l4038_403860


namespace NUMINAMATH_CALUDE_virus_spread_l4038_403809

/-- The average number of computers infected by one computer in each round -/
def average_infection_rate : ℝ := 8

/-- The number of infected computers after two rounds -/
def infected_after_two_rounds : ℕ := 81

/-- The number of infected computers after three rounds -/
def infected_after_three_rounds : ℕ := 729

theorem virus_spread :
  (1 + average_infection_rate + average_infection_rate ^ 2 = infected_after_two_rounds) ∧
  ((1 + average_infection_rate) ^ 3 > 700) := by
  sorry

end NUMINAMATH_CALUDE_virus_spread_l4038_403809


namespace NUMINAMATH_CALUDE_eulerian_circuit_iff_even_degree_l4038_403876

/-- A graph is a pair of a type of vertices and an edge relation -/
structure Graph (V : Type) :=
  (adj : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- An Eulerian circuit in a graph is a path that traverses every edge exactly once and returns to the starting vertex -/
def has_eulerian_circuit {V : Type} (G : Graph V) : Prop := sorry

/-- Theorem: A graph has an Eulerian circuit if and only if every vertex has even degree -/
theorem eulerian_circuit_iff_even_degree {V : Type} (G : Graph V) :
  has_eulerian_circuit G ↔ ∀ v : V, Even (degree G v) := by sorry

end NUMINAMATH_CALUDE_eulerian_circuit_iff_even_degree_l4038_403876


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l4038_403871

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    if the rectangle's area is three times the circle's area,
    then the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (circle_radius : ℝ) (rectangle_shorter_side rectangle_longer_side : ℝ) :
  circle_radius = 6 →
  rectangle_shorter_side = 2 * circle_radius →
  rectangle_shorter_side * rectangle_longer_side = 3 * Real.pi * circle_radius^2 →
  rectangle_longer_side = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l4038_403871


namespace NUMINAMATH_CALUDE_number_of_siblings_l4038_403854

def total_spent : ℕ := 150
def cost_per_sibling : ℕ := 30
def cost_per_parent : ℕ := 30
def num_parents : ℕ := 2

theorem number_of_siblings :
  (total_spent - num_parents * cost_per_parent) / cost_per_sibling = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_siblings_l4038_403854


namespace NUMINAMATH_CALUDE_max_number_in_sample_l4038_403879

/-- Represents a systematic sample from a range of products -/
structure SystematicSample where
  total_products : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Creates a systematic sample given total products and sample size -/
def create_systematic_sample (total_products sample_size : ℕ) : SystematicSample :=
  { total_products := total_products
  , sample_size := sample_size
  , start := 0  -- Assuming start is 0 for simplicity
  , interval := total_products / sample_size
  }

/-- Checks if a number is in the systematic sample -/
def is_in_sample (sample : SystematicSample) (n : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k < sample.sample_size ∧ n = sample.start + k * sample.interval

/-- Gets the maximum number in the systematic sample -/
def max_in_sample (sample : SystematicSample) : ℕ :=
  sample.start + (sample.sample_size - 1) * sample.interval

/-- Theorem: If 58 is in a systematic sample of size 10 from 80 products, 
    then the maximum number in the sample is 74 -/
theorem max_number_in_sample :
  let sample := create_systematic_sample 80 10
  is_in_sample sample 58 → max_in_sample sample = 74 := by
  sorry


end NUMINAMATH_CALUDE_max_number_in_sample_l4038_403879


namespace NUMINAMATH_CALUDE_carla_earnings_l4038_403891

/-- Carla's earnings over two weeks in June --/
theorem carla_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 28 →
  extra_earnings = 63 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1 : ℚ) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2 : ℚ) = 289.80 := by
  sorry

#check carla_earnings

end NUMINAMATH_CALUDE_carla_earnings_l4038_403891


namespace NUMINAMATH_CALUDE_valid_combination_exists_l4038_403815

/-- Represents a combination of cards -/
structure CardCombination where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Checks if a card combination is valid according to the given conditions -/
def isValidCombination (c : CardCombination) : Prop :=
  c.red + c.blue + c.green = 20 ∧
  c.red ≥ 2 ∧
  c.blue ≥ 3 ∧
  c.green ≥ 1 ∧
  3 * c.red + 5 * c.blue + 7 * c.green = 84

/-- There exists a valid card combination that satisfies all conditions -/
theorem valid_combination_exists : ∃ c : CardCombination, isValidCombination c := by
  sorry

#check valid_combination_exists

end NUMINAMATH_CALUDE_valid_combination_exists_l4038_403815


namespace NUMINAMATH_CALUDE_max_original_points_l4038_403827

/-- Represents a rectangular matrix of points on a grid -/
structure RectMatrix where
  rows : ℕ
  cols : ℕ

/-- The maximum grid size -/
def maxGridSize : ℕ := 19

/-- The number of additional points -/
def additionalPoints : ℕ := 45

/-- Checks if a rectangular matrix fits within the maximum grid size -/
def fitsInGrid (rect : RectMatrix) : Prop :=
  rect.rows ≤ maxGridSize ∧ rect.cols ≤ maxGridSize

/-- Checks if a rectangular matrix can be expanded by adding the additional points -/
def canBeExpanded (small rect : RectMatrix) : Prop :=
  (rect.rows - small.rows) * (rect.cols - small.cols) = additionalPoints

/-- The theorem stating the maximum number of points in the original matrix -/
theorem max_original_points : 
  ∃ (small large : RectMatrix), 
    fitsInGrid small ∧ 
    fitsInGrid large ∧
    canBeExpanded small large ∧
    (small.rows = large.rows ∨ small.cols = large.cols) ∧
    small.rows * small.cols = 285 ∧
    ∀ (other : RectMatrix), 
      fitsInGrid other → 
      (∃ (expanded : RectMatrix), 
        fitsInGrid expanded ∧ 
        canBeExpanded other expanded ∧ 
        (other.rows = expanded.rows ∨ other.cols = expanded.cols)) →
      other.rows * other.cols ≤ 285 :=
sorry

end NUMINAMATH_CALUDE_max_original_points_l4038_403827


namespace NUMINAMATH_CALUDE_sphere_identical_views_l4038_403820

/-- A geometric body in 3D space -/
inductive GeometricBody
  | Sphere
  | Cylinder
  | TriangularPrism
  | Cone

/-- Represents a 2D view of a geometric body -/
structure View where
  shape : Type
  size : ℝ

/-- Returns true if all views are identical -/
def identicalViews (front side top : View) : Prop :=
  front = side ∧ side = top

/-- Returns the front, side, and top views of a geometric body -/
def getViews (body : GeometricBody) : (View × View × View) :=
  sorry

theorem sphere_identical_views :
  ∀ (body : GeometricBody),
    (∃ (front side top : View), 
      getViews body = (front, side, top) ∧ 
      identicalViews front side top) 
    ↔ 
    body = GeometricBody.Sphere :=
  sorry

end NUMINAMATH_CALUDE_sphere_identical_views_l4038_403820


namespace NUMINAMATH_CALUDE_paintings_distribution_l4038_403898

theorem paintings_distribution (total_paintings : ℕ) (num_rooms : ℕ) (paintings_per_room : ℕ) :
  total_paintings = 32 →
  num_rooms = 4 →
  total_paintings = num_rooms * paintings_per_room →
  paintings_per_room = 8 := by
  sorry

end NUMINAMATH_CALUDE_paintings_distribution_l4038_403898


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l4038_403878

/-- A quadratic equation ax² + bx + c = 0 -/
structure QuadraticEq where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic equation -/
def discriminant (q : QuadraticEq) : ℝ := q.b^2 - 4*q.a*q.c

/-- A quadratic equation has real roots iff its discriminant is non-negative -/
def has_real_roots (q : QuadraticEq) : Prop := discriminant q ≥ 0

theorem sufficient_but_not_necessary_condition 
  (q1 q2 : QuadraticEq) 
  (h1 : has_real_roots q1)
  (h2 : has_real_roots q2)
  (h3 : q1.a ≠ q2.a) :
  (∀ w c, w * c > 0 → 
    has_real_roots ⟨q2.a, q1.b, q1.c⟩ ∨ has_real_roots ⟨q1.a, q2.b, q2.c⟩) ∧ 
  (∃ w c, w * c ≤ 0 ∧ 
    (has_real_roots ⟨q2.a, q1.b, q1.c⟩ ∨ has_real_roots ⟨q1.a, q2.b, q2.c⟩)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l4038_403878


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l4038_403831

/-- The x-coordinate of the intersection point of y = 9 / (x^2 + 3) and x + y = 3 is 0 -/
theorem intersection_x_coordinate : ∃ y : ℝ, 
  y = 9 / (0^2 + 3) ∧ 0 + y = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l4038_403831


namespace NUMINAMATH_CALUDE_line_equation_proof_l4038_403888

/-- Given two points (1, 0.5) and (1.5, 2), and the fact that the line passing through these points
    splits 8 circles such that the total circle area to the left of the line is 4π,
    prove that the equation of this line is 6x - 2y = 5. -/
theorem line_equation_proof (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (num_circles : ℕ) (left_area : ℝ) :
  p1 = (1, 0.5) →
  p2 = (1.5, 2) →
  num_circles = 8 →
  left_area = 4 * Real.pi →
  ∃ (f : ℝ → ℝ), (∀ x y, f x = y ↔ 6 * x - 2 * y = 5) ∧
                 (∀ x, f x = (x - p1.1) * ((p2.2 - p1.2) / (p2.1 - p1.1)) + p1.2) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l4038_403888


namespace NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l4038_403837

theorem prime_square_remainders_mod_180 :
  ∃! (s : Finset Nat), 
    (∀ r ∈ s, r < 180) ∧ 
    (∀ p : Nat, Prime p → p > 5 → (p^2 % 180) ∈ s) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l4038_403837


namespace NUMINAMATH_CALUDE_four_integer_b_values_l4038_403842

/-- A function that checks if a given integer b results in integer roots for the quadratic equation x^2 + bx + 7b = 0 -/
def has_integer_roots (b : ℤ) : Prop :=
  ∃ p q : ℤ, p + q = -b ∧ p * q = 7 * b

/-- The theorem stating that there are exactly 4 integer values of b for which the quadratic equation x^2 + bx + 7b = 0 always has integer roots -/
theorem four_integer_b_values :
  ∃! (s : Finset ℤ), s.card = 4 ∧ ∀ b : ℤ, has_integer_roots b ↔ b ∈ s :=
sorry

end NUMINAMATH_CALUDE_four_integer_b_values_l4038_403842


namespace NUMINAMATH_CALUDE_fraction_equality_l4038_403805

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4038_403805


namespace NUMINAMATH_CALUDE_cos_fourth_power_identity_l4038_403875

theorem cos_fourth_power_identity (θ : ℝ) : 
  (Real.cos θ)^4 = (1/8) * Real.cos (4*θ) + (1/2) * Real.cos (2*θ) + 0 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_cos_fourth_power_identity_l4038_403875


namespace NUMINAMATH_CALUDE_sequence_property_l4038_403806

/-- The sum of the first n terms of the sequence {a_n} -/
def S (a : ℕ+ → ℕ) (n : ℕ+) : ℕ := (Finset.range n.val).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem stating that if S_n = 2a_n - 2 for all n, then a_n = 2^n for all n -/
theorem sequence_property (a : ℕ+ → ℕ) 
    (h : ∀ n : ℕ+, S a n = 2 * a n - 2) : 
    ∀ n : ℕ+, a n = 2^n.val := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l4038_403806


namespace NUMINAMATH_CALUDE_sum_of_ages_is_100_l4038_403810

/-- Given the conditions about Alice, Ben, and Charlie's ages, prove that the sum of their ages is 100. -/
theorem sum_of_ages_is_100 (A B C : ℕ) 
  (h1 : A = 20 + B + C) 
  (h2 : A^2 = 2000 + (B + C)^2) : 
  A + B + C = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_100_l4038_403810


namespace NUMINAMATH_CALUDE_range_of_a_l4038_403824

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1) ^ x₁ > (2 * a - 1) ^ x₂) →
  1/2 < a ∧ a ≤ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4038_403824


namespace NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l4038_403880

-- Define the tile sides
inductive Side
| Top
| Right
| Bottom
| Left

-- Define the tiles
structure Tile :=
  (id : Nat)
  (top : Nat)
  (right : Nat)
  (bottom : Nat)
  (left : Nat)

-- Define the rectangles
inductive Rectangle
| A
| B
| C
| D

-- Define the placement of tiles
def Placement := Tile → Rectangle

-- Define the adjacency relation between rectangles
def Adjacent : Rectangle → Rectangle → Prop := sorry

-- Define the matching condition for adjacent tiles
def MatchingSides (t1 t2 : Tile) (s1 s2 : Side) : Prop := sorry

-- Define the validity of a placement
def ValidPlacement (p : Placement) : Prop := sorry

-- Define the tiles from the problem
def tileI : Tile := ⟨1, 6, 8, 3, 7⟩
def tileII : Tile := ⟨2, 7, 6, 2, 9⟩
def tileIII : Tile := ⟨3, 5, 1, 9, 0⟩
def tileIV : Tile := ⟨4, 0, 9, 4, 5⟩

-- Theorem statement
theorem tileIV_in_rectangle_C :
  ∀ (p : Placement), ValidPlacement p → p tileIV = Rectangle.C := by
  sorry

end NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l4038_403880


namespace NUMINAMATH_CALUDE_mandatory_work_effect_l4038_403850

/-- Represents the labor market for doctors -/
structure DoctorLaborMarket where
  state_supply : ℝ → ℝ  -- Supply function for state sector
  state_demand : ℝ → ℝ  -- Demand function for state sector
  private_supply : ℝ → ℝ  -- Supply function for private sector
  private_demand : ℝ → ℝ  -- Demand function for private sector

/-- Represents the policy of mandatory work in public healthcare -/
structure MandatoryWorkPolicy where
  years_required : ℕ  -- Number of years required in public healthcare

/-- The equilibrium wage in the state sector -/
def state_equilibrium_wage (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The equilibrium price in the private healthcare sector -/
def private_equilibrium_price (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The effect of the mandatory work policy on the labor market -/
def apply_policy (market : DoctorLaborMarket) (policy : MandatoryWorkPolicy) : DoctorLaborMarket :=
  sorry

theorem mandatory_work_effect (initial_market : DoctorLaborMarket) (policy : MandatoryWorkPolicy) :
  let final_market := apply_policy initial_market policy
  state_equilibrium_wage final_market > state_equilibrium_wage initial_market ∧
  private_equilibrium_price final_market < private_equilibrium_price initial_market :=
sorry

end NUMINAMATH_CALUDE_mandatory_work_effect_l4038_403850


namespace NUMINAMATH_CALUDE_freshman_sample_size_l4038_403834

/-- Calculates the number of students to be sampled from a specific stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation sampleSize stratumSize : ℕ) : ℕ :=
  (stratumSize * sampleSize) / totalPopulation

/-- The number of students to be sampled from the freshman year in a stratified sampling -/
theorem freshman_sample_size :
  let totalPopulation : ℕ := 4500
  let sampleSize : ℕ := 150
  let freshmanSize : ℕ := 1200
  stratifiedSampleSize totalPopulation sampleSize freshmanSize = 40 := by
sorry

#eval stratifiedSampleSize 4500 150 1200

end NUMINAMATH_CALUDE_freshman_sample_size_l4038_403834


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l4038_403894

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 * Nat.factorial (n - 3)) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l4038_403894


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l4038_403839

theorem power_tower_mod_1000 : 5^(5^(5^5)) ≡ 125 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l4038_403839


namespace NUMINAMATH_CALUDE_log_stack_theorem_l4038_403893

/-- Represents a stack of logs -/
structure LogStack where
  bottom_row : ℕ
  top_row : ℕ
  row_difference : ℕ

/-- Calculates the number of rows in the log stack -/
def num_rows (stack : LogStack) : ℕ :=
  (stack.bottom_row - stack.top_row) / stack.row_difference + 1

/-- Calculates the total number of logs in the stack -/
def total_logs (stack : LogStack) : ℕ :=
  (num_rows stack * (stack.bottom_row + stack.top_row)) / 2

/-- The main theorem about the log stack -/
theorem log_stack_theorem (stack : LogStack) 
  (h1 : stack.bottom_row = 15)
  (h2 : stack.top_row = 5)
  (h3 : stack.row_difference = 2) :
  total_logs stack = 60 ∧ stack.top_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_theorem_l4038_403893


namespace NUMINAMATH_CALUDE_probability_three_fives_out_of_five_dice_probability_exactly_three_fives_l4038_403870

/-- The probability of exactly 3 out of 5 fair 10-sided dice showing the number 5 -/
theorem probability_three_fives_out_of_five_dice : ℚ :=
  81 / 10000

/-- A fair 10-sided die -/
def fair_10_sided_die : Finset ℕ := Finset.range 10

/-- The probability of rolling a 5 on a fair 10-sided die -/
def prob_roll_5 : ℚ := 1 / 10

/-- The probability of not rolling a 5 on a fair 10-sided die -/
def prob_not_roll_5 : ℚ := 9 / 10

/-- The number of ways to choose 3 dice out of 5 -/
def ways_to_choose_3_out_of_5 : ℕ := 10

theorem probability_exactly_three_fives (n : ℕ) (k : ℕ) 
  (h1 : n = 5) (h2 : k = 3) : 
  probability_three_fives_out_of_five_dice = 
    (ways_to_choose_3_out_of_5 : ℚ) * (prob_roll_5 ^ k) * (prob_not_roll_5 ^ (n - k)) :=
sorry

end NUMINAMATH_CALUDE_probability_three_fives_out_of_five_dice_probability_exactly_three_fives_l4038_403870


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l4038_403869

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the tangent at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop :=
  tangent_slope a * tangent_slope b = -1

-- Define the y-coordinate of the intersection point
def intersection_y (a b : ℝ) : ℝ := 2 * a * b

-- Theorem statement
theorem intersection_point_y_coordinate 
  (a b : ℝ) 
  (ha : parabola a = 2 * a^2) 
  (hb : parabola b = 2 * b^2) 
  (hperp : perpendicular_tangents a b) :
  intersection_y a b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l4038_403869


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l4038_403841

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l4038_403841


namespace NUMINAMATH_CALUDE_sam_speed_calculation_l4038_403872

def alex_speed : ℚ := 6
def jamie_relative_speed : ℚ := 4/5
def sam_relative_speed : ℚ := 3/4

theorem sam_speed_calculation :
  alex_speed * jamie_relative_speed * sam_relative_speed = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_sam_speed_calculation_l4038_403872


namespace NUMINAMATH_CALUDE_max_value_fraction_l4038_403816

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^2 / (x^2 + y^2 + x*y) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l4038_403816


namespace NUMINAMATH_CALUDE_special_function_value_at_neg_three_l4038_403859

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) ∧ (f 1 = 2)

theorem special_function_value_at_neg_three 
  (f : ℝ → ℝ) (h : special_function f) : f (-3) = -12 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_at_neg_three_l4038_403859


namespace NUMINAMATH_CALUDE_distance_AB_is_750_l4038_403814

/-- The distance between two points A and B -/
def distance_AB : ℝ := 750

/-- The speed of person B in meters per minute -/
def speed_B : ℝ := 50

/-- The time it takes for A to catch up with B when moving in the same direction (in minutes) -/
def time_same_direction : ℝ := 30

/-- The time it takes for A and B to meet when moving towards each other (in minutes) -/
def time_towards_each_other : ℝ := 6

/-- The theorem stating that the distance between A and B is 750 meters -/
theorem distance_AB_is_750 : distance_AB = 750 :=
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_750_l4038_403814


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4038_403821

theorem quadratic_inequality (a b c : ℝ) : a^2 + a*b + a*c < 0 → b^2 > 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4038_403821


namespace NUMINAMATH_CALUDE_fran_speed_to_match_joann_l4038_403822

/-- Proves that Fran needs to ride at 30 mph to cover the same distance as Joann -/
theorem fran_speed_to_match_joann (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) :
  joann_speed = 15 →
  joann_time = 4 →
  fran_time = 2 →
  (fran_time * (joann_speed * joann_time / fran_time) = joann_speed * joann_time) ∧
  (joann_speed * joann_time / fran_time = 30) := by
  sorry

end NUMINAMATH_CALUDE_fran_speed_to_match_joann_l4038_403822


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l4038_403882

/-- Proves the time taken for a train to cross a bridge given its length, speed, and time to pass a fixed point on the bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (signal_post_time : ℝ) 
  (bridge_fixed_point_time : ℝ) 
  (h1 : train_length = 600) 
  (h2 : signal_post_time = 40) 
  (h3 : bridge_fixed_point_time = 1200) :
  let train_speed := train_length / signal_post_time
  let bridge_length := train_speed * bridge_fixed_point_time - train_length
  let total_distance := bridge_length + train_length
  total_distance / train_speed = 1240 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l4038_403882


namespace NUMINAMATH_CALUDE_work_completion_days_l4038_403864

/-- The number of days required for the second group to complete the work -/
def days_for_second_group : ℕ := 4

/-- The daily work output of a boy -/
def boy_work : ℝ := 1

/-- The daily work output of a man -/
def man_work : ℝ := 2 * boy_work

/-- The total amount of work to be done -/
def total_work : ℝ := (12 * man_work + 16 * boy_work) * 5

theorem work_completion_days :
  (13 * man_work + 24 * boy_work) * days_for_second_group = total_work := by sorry

end NUMINAMATH_CALUDE_work_completion_days_l4038_403864


namespace NUMINAMATH_CALUDE_congruence_problem_l4038_403804

theorem congruence_problem (x : ℤ) : 
  (3 * x + 7) % 16 = 5 → (4 * x + 3) % 16 = 11 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4038_403804


namespace NUMINAMATH_CALUDE_equation_solution_l4038_403865

theorem equation_solution (k : ℝ) : (∃ x : ℝ, 2 * x + k - 3 = 6 ∧ x = 3) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4038_403865


namespace NUMINAMATH_CALUDE_airport_distance_l4038_403863

/-- Proves that the distance to the airport is 315 miles given the problem conditions -/
theorem airport_distance : 
  ∀ (d : ℝ) (t : ℝ),
  (d = 45 * (t + 1.5)) →  -- If continued at initial speed, arriving on time
  (d - 45 = 60 * (t - 1)) →  -- Adjusted speed for remaining journey, arriving 1 hour early
  d = 315 := by sorry

end NUMINAMATH_CALUDE_airport_distance_l4038_403863


namespace NUMINAMATH_CALUDE_park_outer_diameter_l4038_403857

/-- Given a park layout with a central statue, lawn, and jogging path, 
    this theorem proves the diameter of the outer boundary. -/
theorem park_outer_diameter 
  (statue_diameter : ℝ) 
  (lawn_width : ℝ) 
  (jogging_path_width : ℝ) 
  (h1 : statue_diameter = 8) 
  (h2 : lawn_width = 10) 
  (h3 : jogging_path_width = 5) : 
  statue_diameter / 2 + lawn_width + jogging_path_width = 19 ∧ 
  2 * (statue_diameter / 2 + lawn_width + jogging_path_width) = 38 :=
sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l4038_403857


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4038_403844

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  b * Real.sin (2 * A) = Real.sqrt 3 * a * Real.sin B →
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 →
  b / c = 3 * Real.sqrt 3 / 4 →
  A = π / 6 ∧ a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4038_403844


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4038_403835

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f (x + y))^2 = (f x)^2 + (f y)^2) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4038_403835


namespace NUMINAMATH_CALUDE_sales_increase_l4038_403896

theorem sales_increase (P : ℝ) (N : ℝ) (h1 : P > 0) (h2 : N > 0) :
  let discount_rate : ℝ := 0.1
  let income_increase_rate : ℝ := 0.08
  let new_price : ℝ := P * (1 - discount_rate)
  let N' : ℝ := N * (1 + income_increase_rate) / (1 - discount_rate)
  (N' - N) / N = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_sales_increase_l4038_403896


namespace NUMINAMATH_CALUDE_complete_collection_probability_l4038_403801

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def uncollected_stickers : ℕ := 6
def collected_stickers : ℕ := 12

theorem complete_collection_probability :
  (Nat.choose uncollected_stickers uncollected_stickers * Nat.choose collected_stickers (selected_stickers - uncollected_stickers)) /
  (Nat.choose total_stickers selected_stickers) = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_complete_collection_probability_l4038_403801


namespace NUMINAMATH_CALUDE_derivative_of_one_plus_cos_l4038_403897

theorem derivative_of_one_plus_cos (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 1 + Real.cos x
  HasDerivAt f (-Real.sin x) x := by sorry

end NUMINAMATH_CALUDE_derivative_of_one_plus_cos_l4038_403897


namespace NUMINAMATH_CALUDE_inequality_proof_l4038_403852

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4038_403852


namespace NUMINAMATH_CALUDE_our_equation_is_linear_l4038_403849

/-- Definition of a linear equation in two variables -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The specific equation we want to prove is linear -/
def our_equation (x y : ℝ) : ℝ := x + y - 5

theorem our_equation_is_linear :
  is_linear_equation our_equation :=
sorry

end NUMINAMATH_CALUDE_our_equation_is_linear_l4038_403849


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l4038_403812

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

theorem f_derivative_at_zero : 
  (deriv f) 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l4038_403812


namespace NUMINAMATH_CALUDE_indeterminate_divisor_l4038_403836

theorem indeterminate_divisor (x y : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) →
  (∃ m : ℤ, x + 7 = y * m + 12) →
  ¬ (∃! y : ℤ, ∃ m : ℤ, x + 7 = y * m + 12) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_divisor_l4038_403836


namespace NUMINAMATH_CALUDE_unique_invariant_quadratic_l4038_403828

/-- A quadratic equation that remains unchanged when its roots are used as coefficients. -/
def InvariantQuadratic (p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧
  ∃ (x y : ℝ),
    x^2 + p*x + q = 0 ∧
    y^2 + p*y + q = 0 ∧
    x ≠ y ∧
    x^2 + y*x + (x*y) = 0 ∧
    p = -(x + y) ∧
    q = x * y

theorem unique_invariant_quadratic :
  ∀ (p q : ℤ), InvariantQuadratic p q → p = 1 ∧ q = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_invariant_quadratic_l4038_403828


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_defined_l4038_403825

theorem sqrt_x_minus_one_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_defined_l4038_403825


namespace NUMINAMATH_CALUDE_fibonacci_triangle_isosceles_l4038_403829

def fibonacci_set : Set ℕ := {2, 3, 5, 8, 13, 21, 34, 55, 89, 144}

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem fibonacci_triangle_isosceles :
  ∀ a b c : ℕ,
    a ∈ fibonacci_set →
    b ∈ fibonacci_set →
    c ∈ fibonacci_set →
    is_triangle a b c →
    is_isosceles a b c :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_triangle_isosceles_l4038_403829


namespace NUMINAMATH_CALUDE_f_form_l4038_403866

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_continuous : Continuous f
axiom f_functional_equation : ∀ x y : ℝ, f (Real.sqrt (x^2 + y^2)) = f x * f y

-- State the theorem to be proved
theorem f_form : ∀ x : ℝ, f x = (f 1) ^ (x^2) := by sorry

end NUMINAMATH_CALUDE_f_form_l4038_403866


namespace NUMINAMATH_CALUDE_min_abs_diff_sum_l4038_403899

theorem min_abs_diff_sum (x a b : ℚ) : 
  x ≠ a ∧ x ≠ b ∧ a ≠ b → 
  a > b → 
  (∀ y : ℚ, |y - a| + |y - b| ≥ 2) ∧ (∃ z : ℚ, |z - a| + |z - b| = 2) →
  2022 + a - b = 2024 := by
sorry

end NUMINAMATH_CALUDE_min_abs_diff_sum_l4038_403899


namespace NUMINAMATH_CALUDE_product_and_quotient_cube_square_l4038_403823

theorem product_and_quotient_cube_square (a b k : ℕ) : 
  100 ≤ a * b ∧ a * b < 1000 →  -- three-digit number condition
  a * b = k^3 →                 -- product is cube of k
  (a : ℚ) / b = k^2 →           -- quotient is square of k
  a = 243 ∧ b = 3 ∧ k = 9 := by
sorry

end NUMINAMATH_CALUDE_product_and_quotient_cube_square_l4038_403823


namespace NUMINAMATH_CALUDE_supermarket_profit_analysis_l4038_403886

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℤ) : ℝ := -5 * x + 150

/-- Represents the daily profit as a function of selling price -/
def profit (x : ℤ) : ℝ := (sales_volume x) * (x - 10)

theorem supermarket_profit_analysis 
  (x : ℤ) 
  (h_range : 10 ≤ x ∧ x ≤ 15) 
  (h_sales_12 : sales_volume 12 = 90) 
  (h_sales_14 : sales_volume 14 = 80) :
  (∃ (k b : ℝ), ∀ (x : ℤ), sales_volume x = k * x + b) ∧ 
  (profit 14 = 320) ∧
  (∀ (y : ℤ), 10 ≤ y ∧ y ≤ 15 → profit y ≤ profit 15) ∧
  (profit 15 = 375) :=
sorry

end NUMINAMATH_CALUDE_supermarket_profit_analysis_l4038_403886


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l4038_403862

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The period of the Fibonacci sequence modulo 9 -/
def fib_mod_9_period : ℕ := 24

theorem fib_150_mod_9 :
  fib 149 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l4038_403862


namespace NUMINAMATH_CALUDE_square_divisibility_l4038_403802

theorem square_divisibility (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l4038_403802


namespace NUMINAMATH_CALUDE_ellipse_m_value_l4038_403868

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis along y-axis, and focal length 4 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1)
  (major_axis : m - 2 > 10 - m)
  (focal_length : ℝ)
  (focal_length_eq : focal_length = 4)

/-- The value of m for the given ellipse is 8 -/
theorem ellipse_m_value (e : Ellipse m) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l4038_403868


namespace NUMINAMATH_CALUDE_emily_fishing_total_weight_l4038_403818

theorem emily_fishing_total_weight :
  let trout_count : ℕ := 4
  let catfish_count : ℕ := 3
  let bluegill_count : ℕ := 5
  let trout_weight : ℚ := 2
  let catfish_weight : ℚ := 1.5
  let bluegill_weight : ℚ := 2.5
  let total_weight : ℚ := trout_count * trout_weight + catfish_count * catfish_weight + bluegill_count * bluegill_weight
  total_weight = 25 := by sorry

end NUMINAMATH_CALUDE_emily_fishing_total_weight_l4038_403818


namespace NUMINAMATH_CALUDE_max_apartments_five_by_five_l4038_403817

/-- Represents a building with a given number of floors and windows per floor. -/
structure Building where
  floors : ℕ
  windowsPerFloor : ℕ

/-- Calculates the maximum number of apartments in a building. -/
def maxApartments (b : Building) : ℕ :=
  b.floors * b.windowsPerFloor

/-- Theorem stating that for a 5-story building with 5 windows per floor,
    the maximum number of apartments is 25. -/
theorem max_apartments_five_by_five :
  ∀ (b : Building),
    b.floors = 5 →
    b.windowsPerFloor = 5 →
    maxApartments b = 25 := by
  sorry

#check max_apartments_five_by_five

end NUMINAMATH_CALUDE_max_apartments_five_by_five_l4038_403817


namespace NUMINAMATH_CALUDE_complex_magnitude_l4038_403851

-- Define complex numbers w and z
variable (w z : ℂ)

-- Define the given conditions
theorem complex_magnitude (h1 : w * z = 20 - 15 * I) (h2 : Complex.abs w = 5) :
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l4038_403851


namespace NUMINAMATH_CALUDE_fraction_undefined_l4038_403883

theorem fraction_undefined (x : ℚ) : (2 * x + 1 = 0) ↔ (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_l4038_403883


namespace NUMINAMATH_CALUDE_total_worksheets_l4038_403819

theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (problems_left : ℕ) :
  problems_per_worksheet = 4 →
  graded_worksheets = 8 →
  problems_left = 32 →
  graded_worksheets + (problems_left / problems_per_worksheet) = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_worksheets_l4038_403819


namespace NUMINAMATH_CALUDE_johns_final_elevation_l4038_403867

/-- Calculates the final elevation after descending for a given time. -/
def finalElevation (startElevation : ℝ) (descentRate : ℝ) (time : ℝ) : ℝ :=
  startElevation - descentRate * time

/-- Proves that John's final elevation is 350 feet. -/
theorem johns_final_elevation :
  let startElevation : ℝ := 400
  let descentRate : ℝ := 10
  let time : ℝ := 5
  finalElevation startElevation descentRate time = 350 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_elevation_l4038_403867


namespace NUMINAMATH_CALUDE_janet_crayons_l4038_403889

/-- The number of crayons Michelle has initially -/
def michelle_initial : ℕ := 2

/-- The number of crayons Michelle has after Janet gives her all of her crayons -/
def michelle_final : ℕ := 4

/-- The number of crayons Janet has initially -/
def janet_initial : ℕ := michelle_final - michelle_initial

theorem janet_crayons : janet_initial = 2 := by sorry

end NUMINAMATH_CALUDE_janet_crayons_l4038_403889


namespace NUMINAMATH_CALUDE_exponential_inequality_l4038_403811

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l4038_403811


namespace NUMINAMATH_CALUDE_sequence_difference_theorem_l4038_403861

theorem sequence_difference_theorem (a : Fin 29 → ℤ) 
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_bound : ∀ k, k ≤ 22 → a (k + 7) - a k ≤ 13) :
  ∃ i j, a i - a j = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_theorem_l4038_403861


namespace NUMINAMATH_CALUDE_truncated_pyramid_diagonal_l4038_403846

/-- Regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  height : ℝ
  lower_base_side : ℝ
  upper_base_side : ℝ

/-- Diagonal of a truncated pyramid -/
def diagonal (p : TruncatedPyramid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specified truncated pyramid is 6 -/
theorem truncated_pyramid_diagonal :
  let p : TruncatedPyramid := ⟨2, 5, 3⟩
  diagonal p = 6 := by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_diagonal_l4038_403846


namespace NUMINAMATH_CALUDE_problem_solution_l4038_403830

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x * y = 2 * a) 
  (h2 : x * z = 3 * b) 
  (h3 : y * z = 4 * c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (3*a*b)/(2*c) + (8*a*c)/(3*b) + (6*b*c)/a ∧ 
  x*y*z = 2 * Real.sqrt (6*a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4038_403830


namespace NUMINAMATH_CALUDE_parallelogram_properties_l4038_403808

-- Define the points
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the vector operation
def vectorOp : ℝ × ℝ := (3 * AB.1 - 2 * AC.1 + BC.1, 3 * AB.2 - 2 * AC.2 + BC.2)

-- Define point D
def D : ℝ × ℝ := (A.1 + BC.1, A.2 + BC.2)

theorem parallelogram_properties :
  vectorOp = (0, 2) ∧ D = (2, -1) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_properties_l4038_403808


namespace NUMINAMATH_CALUDE_thursday_dogs_l4038_403855

/-- The number of dogs Harry walks on Monday, Wednesday, and Friday -/
def dogs_mon_wed_fri : ℕ := 7

/-- The number of dogs Harry walks on Tuesday -/
def dogs_tuesday : ℕ := 12

/-- The amount Harry is paid per dog -/
def pay_per_dog : ℕ := 5

/-- Harry's total earnings for the week -/
def total_earnings : ℕ := 210

/-- The number of days Harry walks 7 dogs -/
def days_with_seven_dogs : ℕ := 3

theorem thursday_dogs :
  ∃ (dogs_thursday : ℕ),
    dogs_thursday * pay_per_dog =
      total_earnings -
      (days_with_seven_dogs * dogs_mon_wed_fri + dogs_tuesday) * pay_per_dog ∧
    dogs_thursday = 9 :=
sorry

end NUMINAMATH_CALUDE_thursday_dogs_l4038_403855


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_60_feet_fencing_l4038_403884

/-- The maximum area of a rectangular pen given a fixed perimeter -/
theorem max_area_rectangular_pen (perimeter : ℝ) :
  perimeter > 0 →
  ∃ (area : ℝ), area = (perimeter / 4) ^ 2 ∧
  ∀ (width height : ℝ), width > 0 → height > 0 → width * 2 + height * 2 = perimeter →
  width * height ≤ area := by
  sorry

/-- The maximum area of a rectangular pen with 60 feet of fencing is 225 square feet -/
theorem max_area_60_feet_fencing :
  ∃ (area : ℝ), area = 225 ∧
  ∀ (width height : ℝ), width > 0 → height > 0 → width * 2 + height * 2 = 60 →
  width * height ≤ area := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_60_feet_fencing_l4038_403884


namespace NUMINAMATH_CALUDE_inequality_and_maximum_l4038_403845

theorem inequality_and_maximum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 3) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3) ∧
  (c = a * b → ∀ c', c' = a * b → c' ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_maximum_l4038_403845


namespace NUMINAMATH_CALUDE_graph_translation_up_one_unit_l4038_403838

/-- Represents a vertical translation of a function --/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- The original quadratic function --/
def originalFunction : ℝ → ℝ := fun x ↦ x^2

theorem graph_translation_up_one_unit :
  verticalTranslation originalFunction 1 = fun x ↦ x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_graph_translation_up_one_unit_l4038_403838


namespace NUMINAMATH_CALUDE_dvd_cost_l4038_403848

/-- Given that two identical DVDs cost $40, prove that five DVDs cost $100. -/
theorem dvd_cost (cost_of_two : ℝ) (h : cost_of_two = 40) :
  5 / 2 * cost_of_two = 100 := by
  sorry

end NUMINAMATH_CALUDE_dvd_cost_l4038_403848


namespace NUMINAMATH_CALUDE_product_prs_is_96_l4038_403885

theorem product_prs_is_96 (p r s : ℕ) 
  (hp : 4^p - 4^3 = 192)
  (hr : 3^r + 81 = 162)
  (hs : 7^s - 7^2 = 3994) :
  p * r * s = 96 := by
  sorry

end NUMINAMATH_CALUDE_product_prs_is_96_l4038_403885


namespace NUMINAMATH_CALUDE_hearing_aid_cost_proof_l4038_403853

/-- The cost of a single hearing aid -/
def hearing_aid_cost : ℝ := 2500

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.80

/-- The amount John pays for both hearing aids -/
def john_payment : ℝ := 1000

/-- Theorem stating that the cost of each hearing aid is $2500 -/
theorem hearing_aid_cost_proof : 
  (1 - insurance_coverage) * (2 * hearing_aid_cost) = john_payment := by
  sorry

end NUMINAMATH_CALUDE_hearing_aid_cost_proof_l4038_403853


namespace NUMINAMATH_CALUDE_triangle_side_value_l4038_403826

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 - c^2 = 2*b →
  Real.sin A * Real.cos C = 3 * Real.cos A * Real.sin A →
  b = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l4038_403826


namespace NUMINAMATH_CALUDE_cans_collected_l4038_403843

/-- The total number of cans collected by six people -/
def total_cans (solomon juwan levi gaby michelle sarah : ℕ) : ℕ :=
  solomon + juwan + levi + gaby + michelle + sarah

/-- Theorem stating the total number of cans collected by six people -/
theorem cans_collected :
  ∀ (solomon juwan levi gaby michelle sarah : ℕ),
    solomon = 66 →
    solomon = 3 * juwan →
    levi = juwan / 2 →
    gaby = (5 * solomon) / 2 →
    michelle = gaby / 3 →
    sarah = gaby - levi - 6 →
    total_cans solomon juwan levi gaby michelle sarah = 467 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l4038_403843


namespace NUMINAMATH_CALUDE_building_floors_l4038_403807

theorem building_floors (total_height : ℝ) (regular_floor_height : ℝ) (extra_height : ℝ) :
  total_height = 61 ∧
  regular_floor_height = 3 ∧
  extra_height = 0.5 →
  ∃ (n : ℕ), n = 20 ∧
    total_height = regular_floor_height * (n - 2 : ℝ) + (regular_floor_height + extra_height) * 2 :=
by
  sorry


end NUMINAMATH_CALUDE_building_floors_l4038_403807


namespace NUMINAMATH_CALUDE_one_child_truthful_l4038_403881

structure Child where
  name : String
  truthful : Bool

def grisha_claim (masha sasha natasha : Child) : Prop :=
  masha.truthful ∧ sasha.truthful ∧ natasha.truthful

def contradictions_exist (masha sasha natasha : Child) : Prop :=
  ¬(masha.truthful ∧ sasha.truthful ∧ natasha.truthful)

theorem one_child_truthful (masha sasha natasha : Child) :
  grisha_claim masha sasha natasha →
  contradictions_exist masha sasha natasha →
  ∃! c : Child, c ∈ [masha, sasha, natasha] ∧ c.truthful :=
by
  sorry

#check one_child_truthful

end NUMINAMATH_CALUDE_one_child_truthful_l4038_403881
