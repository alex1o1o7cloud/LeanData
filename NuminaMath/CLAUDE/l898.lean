import Mathlib

namespace NUMINAMATH_CALUDE_triangle_theorem_l898_89805

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.cos t.C + Real.cos t.A * Real.cos t.B = Real.sqrt 3 * Real.sin t.A * Real.cos t.B)
  (h2 : t.a + t.c = 1)
  (h3 : 0 < t.B)
  (h4 : t.B < Real.pi)
  (h5 : 0 < t.a)
  (h6 : t.a < 1) :
  Real.cos t.B = 1/2 ∧ 1/2 ≤ t.b ∧ t.b < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l898_89805


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l898_89818

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three natural numbers form a scalene triangle -/
def isScalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A function that checks if three natural numbers satisfy the triangle inequality -/
def satisfiesTriangleInequality (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), 
    a ≥ 11 ∧ b ≥ 11 ∧ c ≥ 11 ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    isScalene a b c ∧
    satisfiesTriangleInequality a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 41) ∧
    (∀ (x y z : ℕ), 
      x ≥ 11 ∧ y ≥ 11 ∧ z ≥ 11 →
      isPrime x ∧ isPrime y ∧ isPrime z →
      isScalene x y z →
      satisfiesTriangleInequality x y z →
      isPrime (x + y + z) →
      x + y + z ≥ 41) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l898_89818


namespace NUMINAMATH_CALUDE_sum_of_roots_is_18_l898_89883

/-- A function f that satisfies f(3 + x) = f(3 - x) for all real x -/
def symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

/-- The proposition that f has exactly 6 distinct real roots -/
def has_6_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
     r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
     r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
     r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
     r₅ ≠ r₆)

theorem sum_of_roots_is_18 (f : ℝ → ℝ) 
  (h₁ : symmetric_about_3 f) (h₂ : has_6_distinct_roots f) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_18_l898_89883


namespace NUMINAMATH_CALUDE_cos_a3_plus_a5_l898_89871

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_a3_plus_a5 (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 1 + a 4 + a 7 = 5 * Real.pi / 4) : 
  Real.cos (a 3 + a 5) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_a3_plus_a5_l898_89871


namespace NUMINAMATH_CALUDE_stating_equation_is_quadratic_l898_89819

/-- 
Theorem stating that when a = 3, the equation 3x^(a-1) - x = 5 is quadratic in x.
-/
theorem equation_is_quadratic (x : ℝ) : 
  let a : ℝ := 3
  let f : ℝ → ℝ := λ x => 3 * x^(a - 1) - x - 5
  ∃ (p q r : ℝ), f x = p * x^2 + q * x + r := by
  sorry

end NUMINAMATH_CALUDE_stating_equation_is_quadratic_l898_89819


namespace NUMINAMATH_CALUDE_pablo_book_pages_l898_89804

/-- The number of books Pablo reads -/
def num_books : ℕ := 12

/-- The total amount of money Pablo earned in cents -/
def total_earned : ℕ := 1800

/-- The number of pages in each book -/
def pages_per_book : ℕ := total_earned / num_books

theorem pablo_book_pages :
  pages_per_book = 150 :=
by sorry

end NUMINAMATH_CALUDE_pablo_book_pages_l898_89804


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l898_89816

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def in_band (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem quadratic_function_max_value
  (a b c : ℝ)  -- Coefficients of f(x) = ax^2 + bx + c
  (h1 : in_band (f a b c (-2) + 2) 0 4)
  (h2 : in_band (f a b c 0 + 2) 0 4)
  (h3 : in_band (f a b c 2 + 2) 0 4)
  (h4 : ∀ t : ℝ, in_band (t + 1) (-1) 3 → in_band (f a b c t) (-5/2) (5/2)) :
  ∃ t : ℝ, |f a b c t| = 5/2 ∧ ∀ s : ℝ, |f a b c s| ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l898_89816


namespace NUMINAMATH_CALUDE_taehyung_age_l898_89856

theorem taehyung_age : 
  ∀ (taehyung_age uncle_age : ℕ),
  uncle_age = taehyung_age + 17 →
  (taehyung_age + 4) + (uncle_age + 4) = 43 →
  taehyung_age = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_taehyung_age_l898_89856


namespace NUMINAMATH_CALUDE_car_speed_problem_l898_89802

/-- Given a car traveling for two hours with an average speed of 95 km/h
    and a second hour speed of 70 km/h, prove that the speed in the first hour is 120 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 70 ∧ average_speed = 95 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l898_89802


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l898_89849

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + ((a + 1) / a) ^ 2)
  Real.sqrt 2 < e ∧ e < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l898_89849


namespace NUMINAMATH_CALUDE_meadowbrook_impossibility_l898_89860

theorem meadowbrook_impossibility : ¬ ∃ (h c : ℕ), 21 * h + 6 * c = 74 := by
  sorry

end NUMINAMATH_CALUDE_meadowbrook_impossibility_l898_89860


namespace NUMINAMATH_CALUDE_perpendicular_lengths_determine_side_length_l898_89896

/-- An equilateral triangle with a point inside and perpendiculars to the sides -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The lengths of the perpendiculars from the interior point to the sides
  perp_length_1 : ℝ
  perp_length_2 : ℝ
  perp_length_3 : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  side_length_pos : 0 < side_length
  perp_lengths_pos : 0 < perp_length_1 ∧ 0 < perp_length_2 ∧ 0 < perp_length_3
  perp_sum_bound : perp_length_1 + perp_length_2 + perp_length_3 < side_length * (3 / 2)

/-- The theorem stating the relationship between the perpendicular lengths and the side length -/
theorem perpendicular_lengths_determine_side_length 
  (t : EquilateralTriangleWithPoint) 
  (h1 : t.perp_length_1 = 2) 
  (h2 : t.perp_length_2 = 3) 
  (h3 : t.perp_length_3 = 4) : 
  t.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lengths_determine_side_length_l898_89896


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l898_89821

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₄ = 5 and a₉ = 17, then a₁₄ = 29. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : IsArithmeticSequence a)
    (h_a4 : a 4 = 5)
    (h_a9 : a 9 = 17) : 
  a 14 = 29 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l898_89821


namespace NUMINAMATH_CALUDE_speed_gain_per_week_l898_89838

def initial_speed : ℝ := 80
def training_weeks : ℕ := 16
def speed_increase_percentage : ℝ := 0.20

theorem speed_gain_per_week :
  let final_speed := initial_speed * (1 + speed_increase_percentage)
  let total_speed_gain := final_speed - initial_speed
  let speed_gain_per_week := total_speed_gain / training_weeks
  speed_gain_per_week = 1 := by
  sorry

end NUMINAMATH_CALUDE_speed_gain_per_week_l898_89838


namespace NUMINAMATH_CALUDE_quadrilateral_area_l898_89862

-- Define the radius of the large circle
def R : ℝ := 6

-- Define the centers of the smaller circles
structure Center where
  x : ℝ
  y : ℝ

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : Center
  B : Center
  C : Center
  D : Center

-- Define the condition that the circles touch
def circles_touch (q : Quadrilateral) : Prop := sorry

-- Define the condition that A and C touch at the center of the large circle
def AC_touch_center (q : Quadrilateral) : Prop := sorry

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) :
  circles_touch q →
  AC_touch_center q →
  area q = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l898_89862


namespace NUMINAMATH_CALUDE_inequality_system_solution_l898_89809

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l898_89809


namespace NUMINAMATH_CALUDE_exists_valid_square_forming_strategy_l898_89870

/-- Represents a geometric shape on a graph paper --/
structure Shape :=
  (area : ℝ)
  (is_square : Bool)

/-- Represents a cutting strategy for a shape --/
structure CuttingStrategy :=
  (num_parts : ℕ)
  (all_triangles : Bool)

/-- The original figure given in the problem --/
def original_figure : Shape :=
  { area := 1, is_square := false }

/-- Checks if a cutting strategy is valid for the given conditions --/
def is_valid_strategy (s : CuttingStrategy) : Bool :=
  (s.num_parts ≤ 4) ∨ (s.num_parts ≤ 5 ∧ s.all_triangles)

/-- Theorem stating that there exists a valid cutting strategy to form a square --/
theorem exists_valid_square_forming_strategy :
  ∃ (s : CuttingStrategy) (result : Shape),
    is_valid_strategy s ∧
    result.is_square ∧
    result.area = original_figure.area :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_square_forming_strategy_l898_89870


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l898_89878

/-- The equations of an ellipse and a hyperbola with shared foci -/
theorem ellipse_hyperbola_equations :
  ∀ (a b m n : ℝ),
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/m^2 - y^2/n^2 = 1) →
  (a^2 - b^2 = 4 ∧ m^2 + n^2 = 4) →
  (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, y = k*x → x^2/m^2 - y^2/n^2 = 1) →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    (x₁ - 2)^2 + y₁^2 = (x₁ + 2)^2 + y₁^2 ∧
    x₁^2/m^2 - y₁^2/n^2 = 1 ∧
    x₂^2/a^2 + y₂^2/b^2 = 1 ∧
    x₂^2/m^2 - y₂^2/n^2 = 1) →
  (∀ x y : ℝ, 11*x^2/60 + 11*y^2/16 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
  (∀ x y : ℝ, 5*x^2/4 - 5*y^2/16 = 1 ↔ x^2/m^2 - y^2/n^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l898_89878


namespace NUMINAMATH_CALUDE_supermarket_profit_analysis_l898_89832

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

end NUMINAMATH_CALUDE_supermarket_profit_analysis_l898_89832


namespace NUMINAMATH_CALUDE_unique_ambiguous_product_l898_89859

def numbers : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

def is_valid_product (p : Nat) : Prop :=
  ∃ (s : Finset Nat), s ⊆ numbers ∧ s.card = 5 ∧ s.prod id = p

def parity_ambiguous (p : Nat) : Prop :=
  ∃ (s1 s2 : Finset Nat), s1 ≠ s2 ∧
    s1 ⊆ numbers ∧ s2 ⊆ numbers ∧
    s1.card = 5 ∧ s2.card = 5 ∧
    s1.prod id = p ∧ s2.prod id = p ∧
    s1.sum id % 2 ≠ s2.sum id % 2

theorem unique_ambiguous_product :
  ∃! p, is_valid_product p ∧ parity_ambiguous p ∧ p = 420 := by sorry

end NUMINAMATH_CALUDE_unique_ambiguous_product_l898_89859


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l898_89815

/-- The polynomial function defined by x^3(x-3)^2(2+x) -/
def f (x : ℝ) : ℝ := x^3 * (x-3)^2 * (2+x)

/-- The set of roots of the polynomial equation x^3(x-3)^2(2+x) = 0 -/
def roots : Set ℝ := {x : ℝ | f x = 0}

/-- Theorem: The set of roots of the polynomial equation x^3(x-3)^2(2+x) = 0 is {0, 3, -2} -/
theorem roots_of_polynomial : roots = {0, 3, -2} := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l898_89815


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l898_89836

theorem gcd_lcm_product (a b : ℕ) (ha : a = 150) (hb : b = 225) :
  (Nat.gcd a b) * (Nat.lcm a b) = 33750 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l898_89836


namespace NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l898_89840

def spiderman_comics : ℕ := 6
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 4
def batman_comics : ℕ := 3

def total_comics : ℕ := spiderman_comics + archie_comics + garfield_comics + batman_comics

def comic_groups : ℕ := 4

theorem comic_book_stacking_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * batman_comics.factorial) *
  comic_groups.factorial = 1244160 :=
by sorry

end NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l898_89840


namespace NUMINAMATH_CALUDE_min_value_theorem_l898_89851

theorem min_value_theorem (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq_10 : p + q + r + s + t + u = 10) :
  1/p + 9/q + 4/r + 16/s + 25/t + 36/u ≥ 44.1 ∧
  ∃ (p' q' r' s' t' u' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 10 ∧
    1/p' + 9/q' + 4/r' + 16/s' + 25/t' + 36/u' = 44.1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l898_89851


namespace NUMINAMATH_CALUDE_solution_set_a_eq_1_min_value_range_l898_89884

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 3

-- Theorem 1: Solution set for a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x ≤ 4} = Set.Icc 0 (1/2) := by sorry

-- Theorem 2: Range of a for which f(x) has a minimum value
theorem min_value_range :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) ↔ a ∈ Set.Icc (-3) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_1_min_value_range_l898_89884


namespace NUMINAMATH_CALUDE_determinant_inequality_l898_89899

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (a : ℝ) : 
  det2 (a^2) 1 3 2 < det2 a 0 4 1 ↔ -1 < a ∧ a < 3/2 := by sorry

end NUMINAMATH_CALUDE_determinant_inequality_l898_89899


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l898_89861

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l898_89861


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l898_89892

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 27 = (x + n)^2 + 3) → 
  b = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l898_89892


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l898_89876

/-- Prove that for an ellipse with the given properties, its eccentricity is 2/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 - b^2), 0)
  let l := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * (x - F.1)}
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧
    (A.2 < 0 ∧ B.2 > 0) ∧ 
    (-A.2 = 2 * B.2) →
  (Real.sqrt (a^2 - b^2)) / a = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l898_89876


namespace NUMINAMATH_CALUDE_square_value_l898_89863

theorem square_value : ∃ (square : ℚ), 
  16.2 * ((4 + 1/7 - square * 700) / (1 + 2/7)) = 8.1 ∧ square = 0.005 := by sorry

end NUMINAMATH_CALUDE_square_value_l898_89863


namespace NUMINAMATH_CALUDE_d_range_l898_89846

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The condition that a₃a₄ + 1 = 0 for an arithmetic sequence -/
def sequence_condition (a₁ d : ℝ) : Prop :=
  (arithmetic_sequence a₁ d 3) * (arithmetic_sequence a₁ d 4) + 1 = 0

/-- The theorem stating the range of possible values for d -/
theorem d_range (a₁ d : ℝ) :
  sequence_condition a₁ d → d ≤ -2 ∨ d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_d_range_l898_89846


namespace NUMINAMATH_CALUDE_sum_18_probability_l898_89880

/-- A fair coin with sides labeled 5 and 15 -/
inductive Coin
| Five : Coin
| Fifteen : Coin

/-- A standard six-sided die -/
inductive Die
| One : Die
| Two : Die
| Three : Die
| Four : Die
| Five : Die
| Six : Die

/-- The probability of getting a sum of 18 when flipping the coin and rolling the die -/
def prob_sum_18 : ℚ :=
  1 / 12

/-- Theorem stating that the probability of getting a sum of 18 is 1/12 -/
theorem sum_18_probability : prob_sum_18 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_18_probability_l898_89880


namespace NUMINAMATH_CALUDE_max_value_fraction_l898_89858

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y - x*y = 0) :
  (4 / (x + y)) ≤ 4/9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 4*y - x*y = 0 ∧ 4 / (x + y) = 4/9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l898_89858


namespace NUMINAMATH_CALUDE_coin_exchange_optimization_l898_89886

/-- Represents the denominations of US currency --/
inductive USDenomination
| FiveDollarBill
| Quarter
| Dime
| Nickel
| Penny

/-- Represents a collection of coins and bills --/
def CoinCollection := List USDenomination

def value_of_denomination (d : USDenomination) : ℚ :=
  match d with
  | USDenomination.FiveDollarBill => 5
  | USDenomination.Quarter => 1/4
  | USDenomination.Dime => 1/10
  | USDenomination.Nickel => 1/20
  | USDenomination.Penny => 1/100

def total_value (collection : CoinCollection) : ℚ :=
  collection.map value_of_denomination |>.sum

def initial_collection : CoinCollection :=
  (List.replicate 12 USDenomination.Quarter) ++
  (List.replicate 7 USDenomination.Penny) ++
  (List.replicate 20 USDenomination.Nickel) ++
  (List.replicate 15 USDenomination.Dime)

def optimal_collection : CoinCollection :=
  [USDenomination.FiveDollarBill] ++
  (List.replicate 2 USDenomination.Quarter) ++
  [USDenomination.Nickel] ++
  (List.replicate 2 USDenomination.Penny)

theorem coin_exchange_optimization :
  (total_value initial_collection = 557/100) ∧
  (total_value optimal_collection = 557/100) ∧
  (optimal_collection.length = 6) ∧
  (∀ (c : CoinCollection),
    total_value c = 557/100 → c.length ≥ optimal_collection.length) :=
sorry

end NUMINAMATH_CALUDE_coin_exchange_optimization_l898_89886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l898_89843

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) :
  (∀ k : ℕ, a (k + 2) - a (k + 1) = a (k + 1) - a k) →  -- arithmetic sequence condition
  a 1 = 1/3 →
  a 2 + a 5 = 4 →
  a n = 33 →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l898_89843


namespace NUMINAMATH_CALUDE_sales_increase_l898_89813

theorem sales_increase (P : ℝ) (N : ℝ) (h1 : P > 0) (h2 : N > 0) :
  let discount_rate : ℝ := 0.1
  let income_increase_rate : ℝ := 0.08
  let new_price : ℝ := P * (1 - discount_rate)
  let N' : ℝ := N * (1 + income_increase_rate) / (1 - discount_rate)
  (N' - N) / N = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_sales_increase_l898_89813


namespace NUMINAMATH_CALUDE_prob_odd_diagonals_eq_1_126_l898_89847

/-- Represents a 3x3 grid arrangement of numbers 1 to 9 -/
def Grid := Fin 9 → Fin 9

/-- Checks if a given grid has odd sums on both diagonals -/
def has_odd_diagonal_sums (g : Grid) : Prop :=
  (g 0 + g 4 + g 8).val % 2 = 1 ∧ (g 2 + g 4 + g 6).val % 2 = 1

/-- The set of all valid grid arrangements -/
def all_grids : Finset Grid :=
  sorry

/-- The set of grid arrangements with odd diagonal sums -/
def odd_diagonal_grids : Finset Grid :=
  sorry

/-- The probability of a random grid having odd diagonal sums -/
def prob_odd_diagonals : ℚ :=
  (odd_diagonal_grids.card : ℚ) / (all_grids.card : ℚ)

theorem prob_odd_diagonals_eq_1_126 : prob_odd_diagonals = 1 / 126 :=
  sorry

end NUMINAMATH_CALUDE_prob_odd_diagonals_eq_1_126_l898_89847


namespace NUMINAMATH_CALUDE_fraction_example_l898_89830

/-- A fraction is defined as an expression with a variable in the denominator. -/
def is_fraction (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), ∀ x, f x = g x / h x ∧ h x ≠ 0 ∧ ∃ y, h y ≠ h 0

/-- The expression 1 / (1 - x) is a fraction. -/
theorem fraction_example : is_fraction (λ x => 1 / (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_example_l898_89830


namespace NUMINAMATH_CALUDE_croissant_price_is_three_l898_89826

def discount_rate : ℝ := 0.1
def discount_threshold : ℝ := 50
def num_quiches : ℕ := 2
def price_per_quiche : ℝ := 15
def num_croissants : ℕ := 6
def num_biscuits : ℕ := 6
def price_per_biscuit : ℝ := 2
def final_price : ℝ := 54

def total_cost (price_per_croissant : ℝ) : ℝ :=
  num_quiches * price_per_quiche + 
  num_croissants * price_per_croissant + 
  num_biscuits * price_per_biscuit

theorem croissant_price_is_three :
  ∃ (price_per_croissant : ℝ),
    price_per_croissant = 3 ∧
    total_cost price_per_croissant > discount_threshold ∧
    (1 - discount_rate) * total_cost price_per_croissant = final_price :=
sorry

end NUMINAMATH_CALUDE_croissant_price_is_three_l898_89826


namespace NUMINAMATH_CALUDE_apples_removed_by_ricki_l898_89898

/-- The number of apples Ricki removed -/
def rickis_apples : ℕ := 14

/-- The initial number of apples in the basket -/
def initial_apples : ℕ := 74

/-- The final number of apples in the basket -/
def final_apples : ℕ := 32

/-- Samson removed twice as many apples as Ricki -/
def samsons_apples : ℕ := 2 * rickis_apples

theorem apples_removed_by_ricki :
  rickis_apples = 14 ∧
  initial_apples = final_apples + rickis_apples + samsons_apples :=
by sorry

end NUMINAMATH_CALUDE_apples_removed_by_ricki_l898_89898


namespace NUMINAMATH_CALUDE_sprinkle_cans_remaining_l898_89801

theorem sprinkle_cans_remaining (initial : ℕ) (final : ℕ) 
  (h1 : initial = 12) 
  (h2 : final = initial / 2 - 3) : 
  final = 3 := by
  sorry

end NUMINAMATH_CALUDE_sprinkle_cans_remaining_l898_89801


namespace NUMINAMATH_CALUDE_not_all_positive_k_real_roots_not_all_negative_k_nonzero_im_not_all_real_k_not_pure_imaginary_l898_89850

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z k : ℂ) : Prop := 10 * z^2 - 7 * i * z - k = 0

-- Statement A is false
theorem not_all_positive_k_real_roots :
  ¬ ∀ (k : ℝ), k > 0 → ∀ (z : ℂ), equation z k → z.im = 0 :=
sorry

-- Statement B is false
theorem not_all_negative_k_nonzero_im :
  ¬ ∀ (k : ℝ), k < 0 → ∀ (z : ℂ), equation z k → z.im ≠ 0 :=
sorry

-- Statement C is false
theorem not_all_real_k_not_pure_imaginary :
  ¬ ∀ (k : ℝ), ∀ (z : ℂ), equation z k → z.re ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_not_all_positive_k_real_roots_not_all_negative_k_nonzero_im_not_all_real_k_not_pure_imaginary_l898_89850


namespace NUMINAMATH_CALUDE_number_of_walls_l898_89807

/-- Given the following conditions:
  - Each wall has 30 bricks in a single row
  - There are 50 rows in each wall
  - 3000 bricks will be used to make all the walls
  Prove that the number of walls that can be built is 2. -/
theorem number_of_walls (bricks_per_row : ℕ) (rows_per_wall : ℕ) (total_bricks : ℕ) :
  bricks_per_row = 30 →
  rows_per_wall = 50 →
  total_bricks = 3000 →
  total_bricks / (bricks_per_row * rows_per_wall) = 2 :=
by sorry

end NUMINAMATH_CALUDE_number_of_walls_l898_89807


namespace NUMINAMATH_CALUDE_john_soup_vegetables_l898_89828

/-- Represents the weights of vegetables used in John's soup recipe --/
structure SoupVegetables where
  carrots : ℝ
  potatoes : ℝ
  bell_peppers : ℝ

/-- Calculates the total weight of vegetables used in the soup --/
def total_vegetable_weight (v : SoupVegetables) : ℝ :=
  v.carrots + v.potatoes + v.bell_peppers

/-- Represents John's soup recipe --/
structure SoupRecipe where
  beef_bought : ℝ
  beef_unused : ℝ
  vegetables : SoupVegetables

/-- Theorem stating the correct weights of vegetables in John's soup --/
theorem john_soup_vegetables (recipe : SoupRecipe) : 
  recipe.beef_bought = 4 ∧ 
  recipe.beef_unused = 1 ∧ 
  total_vegetable_weight recipe.vegetables = 2 * (recipe.beef_bought - recipe.beef_unused) ∧
  recipe.vegetables.carrots = recipe.vegetables.potatoes ∧
  recipe.vegetables.bell_peppers = 2 * recipe.vegetables.carrots →
  recipe.vegetables = SoupVegetables.mk 1.5 1.5 3 := by
  sorry


end NUMINAMATH_CALUDE_john_soup_vegetables_l898_89828


namespace NUMINAMATH_CALUDE_circle_fixed_point_l898_89864

/-- A circle with center (a, b) on the parabola y^2 = 4x and tangent to x = -1 passes through (1, 0) -/
theorem circle_fixed_point (a b : ℝ) : 
  b^2 = 4*a →  -- Center (a, b) lies on the parabola y^2 = 4x
  (a + 1)^2 = (1 - a)^2 + b^2 -- Circle is tangent to x = -1
  → (1 - a)^2 + 0^2 = (a + 1)^2 -- Point (1, 0) lies on the circle
  := by sorry

end NUMINAMATH_CALUDE_circle_fixed_point_l898_89864


namespace NUMINAMATH_CALUDE_time_to_make_one_toy_l898_89865

/-- Given that a worker makes 40 toys in 80 hours, prove that it takes 2 hours to make one toy. -/
theorem time_to_make_one_toy (total_hours : ℝ) (total_toys : ℝ) 
  (h1 : total_hours = 80) (h2 : total_toys = 40) : 
  total_hours / total_toys = 2 := by
sorry

end NUMINAMATH_CALUDE_time_to_make_one_toy_l898_89865


namespace NUMINAMATH_CALUDE_derivative_of_one_plus_cos_l898_89814

theorem derivative_of_one_plus_cos (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 1 + Real.cos x
  HasDerivAt f (-Real.sin x) x := by sorry

end NUMINAMATH_CALUDE_derivative_of_one_plus_cos_l898_89814


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l898_89889

theorem modulus_of_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l898_89889


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l898_89810

/-- The rowing speed of a man in still water, given his speeds with and against the stream -/
theorem mans_rowing_speed (with_stream against_stream : ℝ) (h1 : with_stream = 10) (h2 : against_stream = 6) :
  (with_stream + against_stream) / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l898_89810


namespace NUMINAMATH_CALUDE_square_sum_equals_four_l898_89844

theorem square_sum_equals_four (x y : ℝ) (h : x^2 + y^2 + x^2*y^2 - 4*x*y + 1 = 0) : 
  (x + y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_four_l898_89844


namespace NUMINAMATH_CALUDE_prob_one_two_given_different_l898_89873

/-- The probability of at least one die showing 2, given that two fair dice show different numbers -/
theorem prob_one_two_given_different : ℝ := by
  -- Define the sample space of outcomes where the two dice show different numbers
  let different_outcomes : Finset (ℕ × ℕ) := sorry

  -- Define the event of at least one die showing 2, given different numbers
  let event_one_two : Finset (ℕ × ℕ) := sorry

  -- Define the probability measure
  let prob : Finset (ℕ × ℕ) → ℝ := sorry

  -- The probability is the measure of the event divided by the measure of the sample space
  have h : prob event_one_two / prob different_outcomes = 1 / 3 := by sorry

  exact 1 / 3

end NUMINAMATH_CALUDE_prob_one_two_given_different_l898_89873


namespace NUMINAMATH_CALUDE_product_prs_is_96_l898_89831

theorem product_prs_is_96 (p r s : ℕ) 
  (hp : 4^p - 4^3 = 192)
  (hr : 3^r + 81 = 162)
  (hs : 7^s - 7^2 = 3994) :
  p * r * s = 96 := by
  sorry

end NUMINAMATH_CALUDE_product_prs_is_96_l898_89831


namespace NUMINAMATH_CALUDE_opponent_total_score_l898_89806

theorem opponent_total_score (team_scores : List Nat) 
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : ∃ lost_games : List Nat, lost_games.length = 6 ∧ 
    ∀ score ∈ lost_games, score + 1 ∈ team_scores)
  (h3 : ∃ double_score_games : List Nat, double_score_games.length = 5 ∧ 
    ∀ score ∈ double_score_games, 2 * (score / 2) ∈ team_scores)
  (h4 : ∃ tie_score : Nat, tie_score ∈ team_scores ∧ 
    tie_score ∉ (Classical.choose h2) ∧ tie_score ∉ (Classical.choose h3)) :
  (team_scores.sum + 6 - (Classical.choose h3).sum / 2 - (Classical.choose h4)) = 69 := by
sorry

end NUMINAMATH_CALUDE_opponent_total_score_l898_89806


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l898_89881

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

theorem parabola_point_coordinates (p : Parabola) (F : ℝ × ℝ) (A : PointOnParabola p) :
  p.equation = (fun x y => y^2 = 4*x) →
  F = (1, 0) →
  (A.x, A.y) • (1 - A.x, -A.y) = -4 →
  (A.x = 1 ∧ A.y = 2) ∨ (A.x = 1 ∧ A.y = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l898_89881


namespace NUMINAMATH_CALUDE_add_and_round_to_hundredth_l898_89853

-- Define the two numbers to be added
def a : Float := 123.456
def b : Float := 78.9102

-- Define the sum of the two numbers
def sum : Float := a + b

-- Define a function to round to the nearest hundredth
def roundToHundredth (x : Float) : Float :=
  (x * 100).round / 100

-- Theorem statement
theorem add_and_round_to_hundredth :
  roundToHundredth sum = 202.37 := by
  sorry

end NUMINAMATH_CALUDE_add_and_round_to_hundredth_l898_89853


namespace NUMINAMATH_CALUDE_clothing_discount_l898_89869

theorem clothing_discount (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : f > 0) (h3 : f < 1) :
  (f * P - (1/2) * P = 0.4 * (f * P)) → f = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_discount_l898_89869


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l898_89842

/-- The polynomial h(x) -/
def h (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2*x + 15

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 - x^3 + b*x^2 + 120*x + c

/-- The theorem statement -/
theorem polynomial_root_problem (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0) →
  f b c (-1) = -1995.25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l898_89842


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l898_89833

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 2 * |x - 1| - 1 < 0} = {x : ℝ | 1/2 < x ∧ x < 3/2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l898_89833


namespace NUMINAMATH_CALUDE_election_votes_proof_l898_89854

theorem election_votes_proof (total_votes : ℕ) 
  (winner_percent : ℚ) (second_percent : ℚ) (third_percent : ℚ)
  (winner_second_diff : ℕ) (winner_third_diff : ℕ) (winner_fourth_diff : ℕ) :
  winner_percent = 2/5 ∧ 
  second_percent = 7/25 ∧ 
  third_percent = 1/5 ∧
  winner_second_diff = 1536 ∧
  winner_third_diff = 3840 ∧
  winner_fourth_diff = 5632 →
  total_votes = 12800 ∧
  (winner_percent * total_votes).num = 5120 ∧
  (second_percent * total_votes).num = 3584 ∧
  (third_percent * total_votes).num = 2560 ∧
  total_votes - (winner_percent * total_votes).num - 
    (second_percent * total_votes).num - 
    (third_percent * total_votes).num = 1536 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_proof_l898_89854


namespace NUMINAMATH_CALUDE_cost_price_calculation_l898_89835

theorem cost_price_calculation (profit_difference : ℝ) 
  (h1 : profit_difference = 72) 
  (h2 : (0.18 - 0.09) * cost_price = profit_difference) : 
  cost_price = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l898_89835


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_l898_89855

def has_exactly_four_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 4

theorem divisors_of_n_squared (n : ℕ) (h : has_exactly_four_divisors n) :
  let d := Finset.filter (· ∣ n^2) (Finset.range (n^2 + 1))
  d.card = 7 ∨ d.card = 9 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_l898_89855


namespace NUMINAMATH_CALUDE_min_even_integers_l898_89817

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 28 →
  a + b + c + d = 46 →
  a + b + c + d + e + f = 64 →
  ∃ (x y z w u v : ℤ), 
    x + y = 28 ∧
    x + y + z + w = 46 ∧
    x + y + z + w + u + v = 64 ∧
    Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧ Odd u ∧ Odd v :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l898_89817


namespace NUMINAMATH_CALUDE_fifth_root_equation_solution_l898_89845

theorem fifth_root_equation_solution :
  ∃ x : ℝ, (x^(1/2) : ℝ) = 3 ∧ x^(1/2) = (x * (x^3)^(1/2))^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_equation_solution_l898_89845


namespace NUMINAMATH_CALUDE_correct_selection_count_l898_89891

/-- The number of ways to select 4 students from 7, including both boys and girls -/
def select_students (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose boys to_select

/-- Theorem stating the correct number of selections -/
theorem correct_selection_count :
  select_students 7 4 3 4 = 34 := by
  sorry

#eval select_students 7 4 3 4

end NUMINAMATH_CALUDE_correct_selection_count_l898_89891


namespace NUMINAMATH_CALUDE_solution_values_l898_89879

def has_twenty_solutions (n : ℕ+) : Prop :=
  (Finset.filter (fun (x, y, z) => 3 * x + 4 * y + z = n) 
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 20

theorem solution_values (n : ℕ+) (h : has_twenty_solutions n) : n = 21 ∨ n = 22 :=
sorry

end NUMINAMATH_CALUDE_solution_values_l898_89879


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_60_feet_fencing_l898_89822

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

end NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_60_feet_fencing_l898_89822


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l898_89841

def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 1, 3]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

theorem x_eq_2_sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 2 ∧ parallel (a x) (b x)) ∧
  (∀ x : ℝ, x = 2 → parallel (a x) (b x)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l898_89841


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l898_89874

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((2 - i) ^ 2) = -4 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l898_89874


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l898_89888

theorem quadratic_equation_roots (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x + 10 = 0 ↔ x = 2 ∨ x = -5/3) → k = 24 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l898_89888


namespace NUMINAMATH_CALUDE_painting_class_combinations_l898_89877

theorem painting_class_combinations : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_painting_class_combinations_l898_89877


namespace NUMINAMATH_CALUDE_locus_of_point_P_l898_89823

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- A line with slope 1 -/
def line_slope_1 (x y x' y' : ℝ) : Prop := y = x + (y' - x')

/-- The locus equation -/
def locus (x y : ℝ) : Prop := 148*x^2 + 13*y^2 + 64*x*y - 20 = 0

/-- The theorem statement -/
theorem locus_of_point_P :
  ∀ (x' y' x1 y1 x2 y2 : ℝ),
  ellipse x1 y1 ∧ ellipse x2 y2 ∧  -- A and B are on the ellipse
  line_slope_1 x1 y1 x' y' ∧ line_slope_1 x2 y2 x' y' ∧  -- A, B, and P are on a line with slope 1
  x' = (x1 + 2*x2) / 3 ∧  -- AP = 2PB condition
  x1 < x2 →  -- Ensure A and B are distinct points
  locus x' y' :=
by sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l898_89823


namespace NUMINAMATH_CALUDE_eighth_prime_is_19_l898_89897

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Theorem: The 8th prime number is 19 -/
theorem eighth_prime_is_19 : nthPrime 8 = 19 := by sorry

end NUMINAMATH_CALUDE_eighth_prime_is_19_l898_89897


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_quadratic_inequality_condition_l898_89894

-- Statement 1
theorem necessary_not_sufficient_condition (x : ℝ) :
  (x + |x| > 0 → x ≠ 0) ∧ ¬(x ≠ 0 → x + |x| > 0) := by sorry

-- Statement 2
theorem quadratic_inequality_condition (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c ≤ 0) ↔ 
  (∀ x : ℝ, a*x^2 + b*x + c ≥ 0) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_quadratic_inequality_condition_l898_89894


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l898_89812

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular tank -/
def surfaceArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the cost of insulation for a given surface area and cost per square foot -/
def insulationCost (area : ℝ) (costPerSqFt : ℝ) : ℝ :=
  area * costPerSqFt

/-- Theorem: The cost to insulate a tank with given dimensions is $1640 -/
theorem tank_insulation_cost :
  let tankDim : TankDimensions := { length := 7, width := 3, height := 2 }
  let costPerSqFt : ℝ := 20
  insulationCost (surfaceArea tankDim) costPerSqFt = 1640 := by
  sorry


end NUMINAMATH_CALUDE_tank_insulation_cost_l898_89812


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l898_89866

theorem cos_20_minus_cos_40 : Real.cos (20 * Real.pi / 180) - Real.cos (40 * Real.pi / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l898_89866


namespace NUMINAMATH_CALUDE_reading_time_proof_l898_89803

/-- The number of days it takes to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Theorem: It takes 12 days to read a 240-page book at 20 pages per day -/
theorem reading_time_proof : days_to_read 240 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_proof_l898_89803


namespace NUMINAMATH_CALUDE_union_equality_iff_m_range_l898_89893

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ 2*m + 1}

theorem union_equality_iff_m_range (m : ℝ) : 
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_m_range_l898_89893


namespace NUMINAMATH_CALUDE_total_time_is_110_l898_89887

def driving_time_one_way : ℕ := 20
def parent_teacher_night_time : ℕ := 70

def total_time : ℕ := 2 * driving_time_one_way + parent_teacher_night_time

theorem total_time_is_110 : total_time = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_110_l898_89887


namespace NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l898_89868

theorem temperature_difference (B D N : ℝ) : 
  B = D - N →
  (∃ k, k = (D - N + 10) - (D - 4) ∧ (k = 1 ∨ k = -1)) →
  (N = 13 ∨ N = 15) :=
sorry

theorem product_of_N_values : 
  (∃ N₁ N₂ : ℝ, (N₁ = 13 ∧ N₂ = 15) ∧ N₁ * N₂ = 195) :=
sorry

end NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l898_89868


namespace NUMINAMATH_CALUDE_line_points_product_l898_89890

/-- Given a line k passing through the origin with slope √7 / 3,
    if points (x, 8) and (20, y) lie on this line, then x * y = 160. -/
theorem line_points_product (x y : ℝ) : 
  (∃ k : ℝ → ℝ, k 0 = 0 ∧ 
   (∀ x₁ x₂, x₁ ≠ x₂ → (k x₂ - k x₁) / (x₂ - x₁) = Real.sqrt 7 / 3) ∧
   k x = 8 ∧ k 20 = y) →
  x * y = 160 := by
sorry


end NUMINAMATH_CALUDE_line_points_product_l898_89890


namespace NUMINAMATH_CALUDE_daughter_child_weight_l898_89811

/-- Represents the weights of family members in kilograms -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  child : ℝ

/-- The conditions of the problem -/
def weightConditions (w : FamilyWeights) : Prop :=
  w.mother + w.daughter + w.child = 160 ∧
  w.child = (1 / 5) * w.mother ∧
  w.daughter = 40

/-- The theorem to be proved -/
theorem daughter_child_weight (w : FamilyWeights) 
  (h : weightConditions w) : w.daughter + w.child = 60 := by
  sorry


end NUMINAMATH_CALUDE_daughter_child_weight_l898_89811


namespace NUMINAMATH_CALUDE_homes_cleaned_l898_89895

theorem homes_cleaned (earning_per_home : ℝ) (total_earned : ℝ) (h1 : earning_per_home = 46.0) (h2 : total_earned = 12696) :
  total_earned / earning_per_home = 276 := by
  sorry

end NUMINAMATH_CALUDE_homes_cleaned_l898_89895


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l898_89827

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 = -1 →
  a 3 = 4 →
  a 4 + a 5 = 17 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l898_89827


namespace NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l898_89885

theorem smallest_denominator_between_fractions :
  ∀ (a b : ℕ), 
    a > 0 → b > 0 →
    (a : ℚ) / b > 6 / 17 →
    (a : ℚ) / b < 9 / 25 →
    b ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l898_89885


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l898_89824

theorem max_value_sum_of_roots (a b c : ℝ) 
  (sum_eq_two : a + b + c = 2)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -1)
  (c_ge : c ≥ -2) :
  (∃ x y z : ℝ, x + y + z = 2 ∧ x ≥ -1/2 ∧ y ≥ -1 ∧ z ≥ -2 ∧
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 8) = Real.sqrt 66) ∧
  (∀ x y z : ℝ, x + y + z = 2 → x ≥ -1/2 → y ≥ -1 → z ≥ -2 →
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 8) ≤ Real.sqrt 66) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l898_89824


namespace NUMINAMATH_CALUDE_smoothie_price_l898_89820

theorem smoothie_price (cake_price : ℚ) (smoothies_sold : ℕ) (cakes_sold : ℕ) (total_revenue : ℚ) :
  cake_price = 2 →
  smoothies_sold = 40 →
  cakes_sold = 18 →
  total_revenue = 156 →
  ∃ (smoothie_price : ℚ), smoothie_price * smoothies_sold + cake_price * cakes_sold = total_revenue ∧ smoothie_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_smoothie_price_l898_89820


namespace NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l898_89852

theorem triangle_angle_sine_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  2 * (Real.sin α / α + Real.sin β / β + Real.sin γ / γ) ≤ 
    (1 / β + 1 / γ) * Real.sin α + 
    (1 / γ + 1 / α) * Real.sin β + 
    (1 / α + 1 / β) * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l898_89852


namespace NUMINAMATH_CALUDE_side_significant_digits_l898_89857

/-- The area of the square in square meters -/
def area : ℝ := 2.7509

/-- The precision of the area measurement in square meters -/
def precision : ℝ := 0.0001

/-- The number of significant digits in the measurement of the side of the square -/
def significant_digits : ℕ := 5

/-- Theorem stating that the number of significant digits in the measurement of the side of the square is 5 -/
theorem side_significant_digits : 
  ∀ (side : ℝ), side^2 = area → significant_digits = 5 := by
  sorry

end NUMINAMATH_CALUDE_side_significant_digits_l898_89857


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l898_89882

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 6 meters, width 5 meters, and depth 2 meters is 104 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 6 5 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l898_89882


namespace NUMINAMATH_CALUDE_dozen_chocolates_cost_l898_89839

/-- The cost of a magazine in dollars -/
def magazine_cost : ℝ := 1

/-- The cost of a chocolate bar in dollars -/
def chocolate_cost : ℝ := 2

/-- The number of magazines that cost the same as 4 chocolate bars -/
def magazines_equal_to_4_chocolates : ℕ := 8

theorem dozen_chocolates_cost (h : 4 * chocolate_cost = magazines_equal_to_4_chocolates * magazine_cost) :
  12 * chocolate_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_dozen_chocolates_cost_l898_89839


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_thirteen_thirds_l898_89837

theorem floor_plus_x_eq_thirteen_thirds (x : ℚ) : 
  (Int.floor x : ℚ) + x = 13/3 → x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_thirteen_thirds_l898_89837


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l898_89867

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

-- Define the number 2102012 in base 3
def number_base_3 : List Nat := [2, 1, 0, 2, 0, 1, 2]

-- Convert the base 3 number to decimal
def number_decimal : Nat := base_3_to_decimal number_base_3

-- Statement to prove
theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ number_decimal ∧ 
  ∀ (q : Nat), Nat.Prime q → q ∣ number_decimal → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l898_89867


namespace NUMINAMATH_CALUDE_max_income_at_11_l898_89848

def bicycle_rental (x : ℕ) : ℝ :=
  if x ≤ 6 then 50 * x - 115
  else -3 * x^2 + 68 * x - 115

theorem max_income_at_11 :
  ∀ x : ℕ, 3 ≤ x → x ≤ 20 →
    bicycle_rental x ≤ bicycle_rental 11 := by
  sorry

end NUMINAMATH_CALUDE_max_income_at_11_l898_89848


namespace NUMINAMATH_CALUDE_distance_sum_to_axes_l898_89800

/-- The sum of distances from point P(-1, -2) to x-axis and y-axis is 3 -/
theorem distance_sum_to_axes : 
  let P : ℝ × ℝ := (-1, -2)
  abs P.2 + abs P.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_to_axes_l898_89800


namespace NUMINAMATH_CALUDE_reflect_point_coords_l898_89808

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to reflect a point across the xz-plane
def reflectAcrossXZPlane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Theorem statement
theorem reflect_point_coords :
  let original := Point3D.mk (-4) 3 5
  let reflected := reflectAcrossXZPlane original
  reflected.x = 4 ∧ reflected.y = -3 ∧ reflected.z = 5 := by
  sorry


end NUMINAMATH_CALUDE_reflect_point_coords_l898_89808


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_724_l898_89875

theorem quadratic_expression_equals_724 
  (x y : ℝ) 
  (h1 : 4 * x + y = 18) 
  (h2 : x + 4 * y = 20) : 
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_724_l898_89875


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l898_89829

def total_players : ℕ := 14
def triplets : ℕ := 3
def starters : ℕ := 6

theorem volleyball_team_selection :
  (Nat.choose total_players starters) -
  ((Nat.choose triplets 2) * (Nat.choose (total_players - triplets) (starters - 2)) +
   (Nat.choose triplets 3) * (Nat.choose (total_players - triplets) (starters - 3))) = 1848 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l898_89829


namespace NUMINAMATH_CALUDE_greatest_a_value_l898_89834

theorem greatest_a_value (a : ℝ) : 
  (9 * Real.sqrt ((3 * a)^2 + 1^2) - 9 * a^2 - 1) / (Real.sqrt (1 + 3 * a^2) + 2) = 3 →
  a ≤ Real.sqrt (13/3) :=
by sorry

end NUMINAMATH_CALUDE_greatest_a_value_l898_89834


namespace NUMINAMATH_CALUDE_samantha_bus_time_l898_89825

/-- Calculates the time Samantha spends on the bus given her schedule --/
theorem samantha_bus_time :
  let leave_time : Nat := 7 * 60 + 15  -- 7:15 AM in minutes
  let return_time : Nat := 17 * 60 + 15  -- 5:15 PM in minutes
  let total_away_time : Nat := return_time - leave_time
  let class_time : Nat := 8 * 45  -- 8 classes of 45 minutes each
  let lunch_time : Nat := 40
  let extracurricular_time : Nat := 90
  let total_school_time : Nat := class_time + lunch_time + extracurricular_time
  let bus_time : Nat := total_away_time - total_school_time
  bus_time = 110 := by sorry

end NUMINAMATH_CALUDE_samantha_bus_time_l898_89825


namespace NUMINAMATH_CALUDE_semicircle_arc_length_l898_89872

-- Define the right triangle with inscribed semicircle
structure RightTriangleWithSemicircle where
  -- Hypotenuse segments
  a : ℝ
  b : ℝ
  -- Assumption: a and b are positive
  ha : a > 0
  hb : b > 0
  -- Assumption: The semicircle is inscribed in the right triangle
  -- with its diameter on the hypotenuse

-- Define the theorem
theorem semicircle_arc_length 
  (triangle : RightTriangleWithSemicircle) 
  (h_a : triangle.a = 30) 
  (h_b : triangle.b = 40) : 
  ∃ (arc_length : ℝ), arc_length = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_semicircle_arc_length_l898_89872
