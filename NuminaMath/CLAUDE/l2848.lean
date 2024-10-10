import Mathlib

namespace cubic_equation_root_l2848_284845

theorem cubic_equation_root (h : ℚ) : 
  (3 : ℚ)^3 + h * 3 - 14 = 0 → h = -13/3 := by
  sorry

end cubic_equation_root_l2848_284845


namespace no_n_exists_for_combination_equality_l2848_284857

theorem no_n_exists_for_combination_equality :
  ¬ ∃ (n : ℕ), n > 0 ∧ (Nat.choose n 3 = Nat.choose (n-1) 3 + Nat.choose (n-1) 4) := by
  sorry

end no_n_exists_for_combination_equality_l2848_284857


namespace parabola_point_ordering_l2848_284897

/-- Theorem: For a parabola y = (x-2)² + k and two points on it, prove y₁ > y₂ > k -/
theorem parabola_point_ordering (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  y₁ = (x₁ - 2)^2 + k →
  y₂ = (x₂ - 2)^2 + k →
  x₂ > 2 →
  2 > x₁ →
  x₁ + x₂ < 4 →
  y₁ > y₂ ∧ y₂ > k :=
by sorry

end parabola_point_ordering_l2848_284897


namespace conference_duration_l2848_284836

def minutes_in_hour : ℕ := 60

def day1_hours : ℕ := 7
def day1_minutes : ℕ := 15

def day2_hours : ℕ := 8
def day2_minutes : ℕ := 45

def total_conference_minutes : ℕ := 
  (day1_hours * minutes_in_hour + day1_minutes) +
  (day2_hours * minutes_in_hour + day2_minutes)

theorem conference_duration :
  total_conference_minutes = 960 := by
  sorry

end conference_duration_l2848_284836


namespace f_monotonicity_l2848_284814

def f (m n : ℕ) (x : ℝ) : ℝ := x^(m/n)

theorem f_monotonicity (m n : ℕ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m n x₁ < f m n x₂) ∧
  (n % 2 = 1 ∧ m % 2 = 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f m n x₁ > f m n x₂) ∧
  (n % 2 = 1 ∧ m % 2 = 1 → ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f m n x₁ < f m n x₂) :=
by sorry

end f_monotonicity_l2848_284814


namespace ratio_fourth_term_l2848_284825

theorem ratio_fourth_term (x y : ℝ) (hx : x = 0.8571428571428571) :
  (0.75 : ℝ) / x = 7 / y → y = 8 := by
sorry

end ratio_fourth_term_l2848_284825


namespace female_employees_count_l2848_284849

/-- Represents the number of employees in a company -/
structure Company where
  total : ℕ
  female : ℕ
  male : ℕ
  female_managers : ℕ
  male_managers : ℕ

/-- The conditions of the company -/
def company_conditions (c : Company) : Prop :=
  c.female_managers = 300 ∧
  c.female_managers + c.male_managers = (2 : ℚ) / 5 * c.total ∧
  c.male_managers = (2 : ℚ) / 5 * c.male ∧
  c.total = c.female + c.male

/-- The theorem stating that the number of female employees is 750 -/
theorem female_employees_count (c : Company) : 
  company_conditions c → c.female = 750 := by
  sorry

end female_employees_count_l2848_284849


namespace max_value_polynomial_l2848_284851

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  ∃ M : ℝ, M = (6084 : ℝ) / 17 ∧
  ∀ z w : ℝ, z + w = 5 →
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≤ M ∧
    ∃ a b : ℝ, a + b = 5 ∧
      a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 = M :=
by sorry

end max_value_polynomial_l2848_284851


namespace polynomial_factorization_l2848_284826

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 2*x) * (x^2 + 2*x + 2) + 1 = (x + 1)^4 ∧
  (x^2 - 4*x) * (x^2 - 4*x + 8) + 16 = (x - 2)^4 := by
  sorry

end polynomial_factorization_l2848_284826


namespace hexagonal_prism_vertices_l2848_284885

/-- A hexagonal prism is a three-dimensional geometric shape with hexagonal bases -/
structure HexagonalPrism :=
  (base : Nat)
  (height : Nat)

/-- The number of vertices in a hexagonal prism -/
def num_vertices (prism : HexagonalPrism) : Nat :=
  12

/-- Theorem: A hexagonal prism has 12 vertices -/
theorem hexagonal_prism_vertices (prism : HexagonalPrism) :
  num_vertices prism = 12 := by
  sorry

end hexagonal_prism_vertices_l2848_284885


namespace point_e_satisfies_conditions_l2848_284818

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Theorem: Point E(0, 0) satisfies the area ratio conditions in quadrilateral ABCD -/
theorem point_e_satisfies_conditions (A B C D E : Point) 
  (hA : A = ⟨-2, -4⟩) (hB : B = ⟨-2, 3⟩) (hC : C = ⟨4, 6⟩) (hD : D = ⟨4, -1⟩) (hE : E = ⟨0, 0⟩) :
  triangleArea E A B / triangleArea E C D = 1 / 2 ∧
  triangleArea E A D / triangleArea E B C = 3 / 4 :=
sorry

end point_e_satisfies_conditions_l2848_284818


namespace neither_necessary_nor_sufficient_l2848_284844

theorem neither_necessary_nor_sufficient : 
  ¬(∀ x : ℝ, -1/2 < x ∧ x < 1 → 0 < x ∧ x < 2) ∧ 
  ¬(∀ x : ℝ, 0 < x ∧ x < 2 → -1/2 < x ∧ x < 1) := by
  sorry

end neither_necessary_nor_sufficient_l2848_284844


namespace inequality_solution_l2848_284853

theorem inequality_solution (x : ℝ) : 
  (5 - 1 / (3 * x + 4) < 7) ↔ (x < -11/6 ∨ x > -4/3) :=
by sorry

end inequality_solution_l2848_284853


namespace one_minus_repeating_third_equals_two_thirds_l2848_284875

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1/3

-- Theorem statement
theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_third = 2/3 := by
  sorry

end one_minus_repeating_third_equals_two_thirds_l2848_284875


namespace roof_area_l2848_284874

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 4 * width →
  length - width = 45 →
  width * length = 900 := by
sorry

end roof_area_l2848_284874


namespace complex_circle_extrema_l2848_284869

theorem complex_circle_extrema (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 1) :
  (∃ w : ℂ, Complex.abs (w - (1 + 2*I)) = 1 ∧ Complex.abs (w - (3 + I)) = Real.sqrt 5 + 1) ∧
  (∃ v : ℂ, Complex.abs (v - (1 + 2*I)) = 1 ∧ Complex.abs (v - (3 + I)) = Real.sqrt 5 - 1) ∧
  (∀ u : ℂ, Complex.abs (u - (1 + 2*I)) = 1 →
    Real.sqrt 5 - 1 ≤ Complex.abs (u - (3 + I)) ∧ Complex.abs (u - (3 + I)) ≤ Real.sqrt 5 + 1) :=
by sorry

end complex_circle_extrema_l2848_284869


namespace quadratic_inequality_range_l2848_284871

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end quadratic_inequality_range_l2848_284871


namespace polynomial_equality_l2848_284850

theorem polynomial_equality : 99^5 - 5*99^4 + 10*99^3 - 10*99^2 + 5*99 - 1 = 98^5 := by
  sorry

end polynomial_equality_l2848_284850


namespace binomial_max_remainder_l2848_284899

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem binomial_max_remainder (k : ℕ) (h1 : 30 ≤ k) (h2 : k ≤ 70) :
  ∃ M : ℕ, 
    (∀ j : ℕ, 30 ≤ j → j ≤ 70 → 
      (binomial 100 j) / Nat.gcd (binomial 100 j) (binomial 100 (j+3)) ≤ M) ∧
    M % 1000 = 664 := by
  sorry

end binomial_max_remainder_l2848_284899


namespace floor_sum_of_squares_and_product_l2848_284891

theorem floor_sum_of_squares_and_product (p q r s : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s →
  p^2 + q^2 = 2500 →
  r^2 + s^2 = 2500 →
  p * q = 1152 →
  r * s = 1152 →
  ⌊p + q + r + s⌋ = 138 := by
sorry

end floor_sum_of_squares_and_product_l2848_284891


namespace equal_sharing_contribution_l2848_284803

def earnings : List ℕ := [10, 30, 50, 40, 70]

theorem equal_sharing_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  match max_earner with
  | some max => max - equal_share = 30
  | none => False
  := by sorry

end equal_sharing_contribution_l2848_284803


namespace manager_wage_l2848_284822

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.2 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.manager - 3.4

/-- The theorem stating that under the given conditions, the manager's hourly wage is $8.50 -/
theorem manager_wage (w : Wages) (h : wage_conditions w) : w.manager = 8.5 := by
  sorry

end manager_wage_l2848_284822


namespace point_M_coordinates_midpoint_E_points_P₁_P₂_l2848_284809

noncomputable section

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define vertices
def A (b : ℝ) : ℝ × ℝ := (0, b)
def B (b : ℝ) : ℝ × ℝ := (0, -b)
def Q (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Theorem statements
theorem point_M_coordinates (a b : ℝ) (h : 0 < b ∧ b < a) :
  ∃ M : ℝ × ℝ, vec_add (A b) M = vec_scale (1/2) (vec_add (vec_add (A b) (Q a)) (vec_add (A b) (B b))) →
  M = (a/2, -b/2) := sorry

theorem midpoint_E (a b k₁ k₂ : ℝ) (h : k₁ * k₂ = -b^2 / a^2) :
  ∃ C D E : ℝ × ℝ, ellipse a b C.1 C.2 ∧ ellipse a b D.1 D.2 ∧
  C.2 = k₁ * C.1 + p ∧ D.2 = k₁ * D.1 + p ∧ E.2 = k₂ * E.1 ∧ E.2 = k₁ * E.1 + p →
  E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) := sorry

theorem points_P₁_P₂ (a b : ℝ) (P P₁ P₂ : ℝ × ℝ) (h₁ : a = 10 ∧ b = 5) (h₂ : P = (-8, -1)) :
  ellipse a b P₁.1 P₁.2 ∧ ellipse a b P₂.1 P₂.2 ∧
  vec_add (vec_add P P₁) (vec_add P P₂) = vec_add P (Q a) →
  (P₁ = (-6, -4) ∧ P₂ = (8, 3)) ∨ (P₁ = (8, 3) ∧ P₂ = (-6, -4)) := sorry

end point_M_coordinates_midpoint_E_points_P₁_P₂_l2848_284809


namespace expression_evaluation_l2848_284827

theorem expression_evaluation (b : ℚ) (h : b = 4/3) :
  (3 * b^2 - 14 * b + 5) * (3 * b - 4) = 0 := by
  sorry

end expression_evaluation_l2848_284827


namespace range_of_m_l2848_284865

theorem range_of_m (x m : ℝ) : 
  (∀ x, -1 < x ∧ x < 4 → x > 2*m^2 - 3) ∧ 
  (∃ x, x > 2*m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) → 
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l2848_284865


namespace f_lower_bound_a_range_l2848_284866

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: f(x) ≥ 2 for all x and a
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by sorry

-- Theorem 2: If f(-3/2) < 3, then -1 < a < 0
theorem a_range (a : ℝ) : f (-3/2) a < 3 → -1 < a ∧ a < 0 := by sorry

end f_lower_bound_a_range_l2848_284866


namespace midpoint_fraction_l2848_284859

theorem midpoint_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  (a + b) / 2 = (19 : ℚ) / 24 := by
sorry

end midpoint_fraction_l2848_284859


namespace solution_difference_l2848_284868

theorem solution_difference (p q : ℝ) : 
  ((6 * p - 18) / (p^2 + 3*p - 18) = p + 3) →
  ((6 * q - 18) / (q^2 + 3*q - 18) = q + 3) →
  p ≠ q →
  p > q →
  p - q = 9 := by
sorry

end solution_difference_l2848_284868


namespace merchant_profit_percentage_l2848_284862

/-- The profit percentage for a merchant who marks up goods by 75% and then offers a 10% discount -/
theorem merchant_profit_percentage : 
  let markup_percentage : ℝ := 75
  let discount_percentage : ℝ := 10
  let cost_price : ℝ := 100
  let marked_price : ℝ := cost_price * (1 + markup_percentage / 100)
  let selling_price : ℝ := marked_price * (1 - discount_percentage / 100)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := (profit / cost_price) * 100
  profit_percentage = 57.5 := by sorry

end merchant_profit_percentage_l2848_284862


namespace tenth_term_of_sequence_l2848_284810

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a : ℚ) (d : ℚ) (h1 : a = 1/2) (h2 : d = 2/3) :
  arithmetic_sequence a d 10 = 13/2 := by
  sorry

end tenth_term_of_sequence_l2848_284810


namespace quadratic_equation_determination_l2848_284893

theorem quadratic_equation_determination (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 → x = 5 ∨ x = 3 ∨ x = -6 ∨ x = -4) →
  (5 + 3 = -b) →
  ((-6) * (-4) = c) →
  (b = -8 ∧ c = 24) := by
  sorry

end quadratic_equation_determination_l2848_284893


namespace tan_sum_equals_three_l2848_284819

theorem tan_sum_equals_three (α β : Real) 
  (h1 : α + β = π/3)
  (h2 : Real.sin α * Real.sin β = (Real.sqrt 3 - 3)/6) :
  Real.tan α + Real.tan β = 3 := by
  sorry

end tan_sum_equals_three_l2848_284819


namespace ellipse_intersection_theorem_l2848_284880

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ

/-- The area of a triangle formed by two points on an ellipse and a fixed point -/
def triangleArea (e : Ellipse) (l : Line) (A : Point) : ℝ := sorry

/-- The main theorem -/
theorem ellipse_intersection_theorem (e : Ellipse) (l : Line) (A : Point) :
  e.a = 2 ∧ e.b = Real.sqrt 2 ∧ 
  A.x = 2 ∧ A.y = 0 ∧
  triangleArea e l A = Real.sqrt 10 / 3 →
  l.k = 1 ∨ l.k = -1 := by sorry

end ellipse_intersection_theorem_l2848_284880


namespace sector_central_angle_l2848_284815

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 2/5 * Real.pi) :
  (2 * area) / (r^2) = π / 5 := by
sorry

end sector_central_angle_l2848_284815


namespace pancake_problem_l2848_284873

theorem pancake_problem (pancakes_made : ℕ) (family_size : ℕ) : pancakes_made = 12 → family_size = 8 → 
  (pancakes_made - family_size) + (family_size - (pancakes_made - family_size)) = 4 := by
  sorry

end pancake_problem_l2848_284873


namespace quadratic_equation_m_value_l2848_284837

theorem quadratic_equation_m_value : ∃! m : ℤ, |m| = 2 ∧ m + 2 ≠ 0 := by sorry

end quadratic_equation_m_value_l2848_284837


namespace unique_solution_l2848_284870

/-- Represents a four-digit number as individual digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Converts a four-digit number to its numerical value -/
def to_nat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Converts a two-digit number to its numerical value -/
def two_digit_to_nat (a b : Nat) : Nat :=
  10 * a + b

/-- States that A̅B² = A̅CDB -/
def condition1 (n : FourDigitNumber) : Prop :=
  (two_digit_to_nat n.a n.b)^2 = to_nat n

/-- States that C̅D³ = A̅CBD -/
def condition2 (n : FourDigitNumber) : Prop :=
  (two_digit_to_nat n.c n.d)^3 = 1000 * n.a + 100 * n.c + 10 * n.b + n.d

/-- The main theorem stating that the only solution is A = 9, B = 6, C = 2, D = 1 -/
theorem unique_solution :
  ∀ n : FourDigitNumber, condition1 n ∧ condition2 n →
  n.a = 9 ∧ n.b = 6 ∧ n.c = 2 ∧ n.d = 1 := by sorry

end unique_solution_l2848_284870


namespace min_value_expression_l2848_284860

theorem min_value_expression (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  1 / (2*x + y)^2 + 4 / (x - 2*y)^2 ≥ 3/5 :=
sorry

end min_value_expression_l2848_284860


namespace boys_camp_total_l2848_284840

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 21 →
  total = 150 := by
  sorry

end boys_camp_total_l2848_284840


namespace inequality_proof_l2848_284861

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1/2) : 
  (1 - a^2 + c^2) / (c * (a + 2*b)) + 
  (1 - b^2 + a^2) / (a * (b + 2*c)) + 
  (1 - c^2 + b^2) / (b * (c + 2*a)) ≥ 6 := by
sorry


end inequality_proof_l2848_284861


namespace inscribed_circle_radius_in_square_midpoint_triangle_l2848_284887

/-- Given a square with side length 12, this theorem proves that the radius of the circle
    inscribed in the triangle formed by connecting midpoints of adjacent sides to each other
    and to the opposite side is 2√5 - √2. -/
theorem inscribed_circle_radius_in_square_midpoint_triangle :
  let square_side : ℝ := 12
  let midpoint_triangle_area : ℝ := 54
  let midpoint_triangle_semiperimeter : ℝ := 6 * Real.sqrt 5 + 3 * Real.sqrt 2
  let inscribed_circle_radius : ℝ := midpoint_triangle_area / midpoint_triangle_semiperimeter
  inscribed_circle_radius = 2 * Real.sqrt 5 - Real.sqrt 2 := by
  sorry


end inscribed_circle_radius_in_square_midpoint_triangle_l2848_284887


namespace rectangle_area_l2848_284895

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 120,
    prove that its area is 675. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 120 → l * b = 675 := by sorry

end rectangle_area_l2848_284895


namespace arithmetic_sequence_length_l2848_284896

/-- An arithmetic sequence with given first term, second term, and last term -/
structure ArithmeticSequence where
  first_term : ℕ
  second_term : ℕ
  last_term : ℕ

/-- The number of terms in an arithmetic sequence -/
def num_terms (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℕ :=
  seq.second_term - seq.first_term

theorem arithmetic_sequence_length :
  let seq := ArithmeticSequence.mk 13 19 127
  num_terms seq = 20 := by sorry

end arithmetic_sequence_length_l2848_284896


namespace total_days_1999_to_2005_l2848_284838

def is_leap_year (year : ℕ) : Bool :=
  year = 2000 || year = 2004

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).map (fun i => days_in_year (start_year + i))
    |>.sum

theorem total_days_1999_to_2005 :
  total_days 1999 2005 = 2557 := by
  sorry

end total_days_1999_to_2005_l2848_284838


namespace constant_term_zero_implies_m_negative_one_l2848_284842

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x - m^2 + 1

/-- The constant term of the quadratic equation -/
def constant_term (m : ℝ) : ℝ := quadratic_equation m 0

theorem constant_term_zero_implies_m_negative_one :
  constant_term (-1) = 0 ∧ (∀ m : ℝ, constant_term m = 0 → m = -1) :=
sorry

end constant_term_zero_implies_m_negative_one_l2848_284842


namespace simple_random_for_ten_basketballs_l2848_284894

/-- Enumeration of sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | WithReplacement

/-- Definition of a sampling scenario --/
structure SamplingScenario where
  population_size : ℕ
  sample_size : ℕ
  for_quality_testing : Bool

/-- Function to determine the appropriate sampling method --/
def appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- Theorem stating that Simple Random Sampling is appropriate for the given scenario --/
theorem simple_random_for_ten_basketballs :
  let scenario : SamplingScenario := {
    population_size := 10,
    sample_size := 1,
    for_quality_testing := true
  }
  appropriate_sampling_method scenario = SamplingMethod.SimpleRandom :=
sorry

end simple_random_for_ten_basketballs_l2848_284894


namespace hostel_expenditure_hostel_expenditure_result_l2848_284889

/-- Calculates the new total expenditure of a hostel after accommodating more students -/
theorem hostel_expenditure (initial_students : ℕ) (additional_students : ℕ) 
  (average_decrease : ℕ) (total_increase : ℕ) : ℕ :=
  let new_students := initial_students + additional_students
  let original_average := (total_increase + new_students * average_decrease) / (new_students - initial_students)
  new_students * (original_average - average_decrease)

/-- The total expenditure of the hostel after accommodating more students is 7500 rupees -/
theorem hostel_expenditure_result : 
  hostel_expenditure 100 25 10 500 = 7500 := by
  sorry

end hostel_expenditure_hostel_expenditure_result_l2848_284889


namespace absolute_value_equation_unique_solution_l2848_284890

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x - 3| :=
by sorry

end absolute_value_equation_unique_solution_l2848_284890


namespace fish_count_after_transfer_l2848_284854

/-- The total number of fish after Lilly gives some to Jack -/
def total_fish (lilly_initial : ℕ) (rosy : ℕ) (jack_initial : ℕ) (transfer : ℕ) : ℕ :=
  (lilly_initial - transfer) + rosy + (jack_initial + transfer)

/-- Theorem stating the total number of fish after the transfer -/
theorem fish_count_after_transfer :
  total_fish 10 9 15 2 = 34 := by
  sorry

end fish_count_after_transfer_l2848_284854


namespace quadratic_polynomial_discriminant_l2848_284846

/-- Given a quadratic polynomial P(x) = ax² + bx + c where a ≠ 0,
    if P(x) = x - 2 has exactly one root and
    P(x) = 1 - x/2 has exactly one root,
    then the discriminant of P(x) is -1/2 -/
theorem quadratic_polynomial_discriminant
  (a b c : ℝ) (ha : a ≠ 0)
  (h1 : ∃! x, a * x^2 + b * x + c = x - 2)
  (h2 : ∃! x, a * x^2 + b * x + c = 1 - x / 2) :
  b^2 - 4*a*c = -1/2 := by
sorry

end quadratic_polynomial_discriminant_l2848_284846


namespace a_gt_b_iff_f_a_gt_f_b_l2848_284864

theorem a_gt_b_iff_f_a_gt_f_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a + Real.log a > b + Real.log b :=
sorry

end a_gt_b_iff_f_a_gt_f_b_l2848_284864


namespace value_of_A_l2848_284806

/-- Given the value assignments for letters and words, prove the value of A -/
theorem value_of_A (H M A T E : ℤ)
  (h1 : H = 10)
  (h2 : M + A + T + H = 35)
  (h3 : T + E + A + M = 42)
  (h4 : M + E + E + T = 38) :
  A = 21 := by
  sorry

end value_of_A_l2848_284806


namespace quadratic_intersection_point_l2848_284847

theorem quadratic_intersection_point 
  (a b c d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : b ≠ 0) : 
  let f1 := fun x : ℝ => a * x^2 + b * x + c
  let f2 := fun x : ℝ => a * x^2 - b * x + c + d
  ∃! p : ℝ × ℝ, 
    f1 p.1 = f2 p.1 ∧ 
    p = (d / (2 * b), a * (d^2 / (4 * b^2)) + d / 2 + c) :=
by sorry

end quadratic_intersection_point_l2848_284847


namespace max_value_implies_a_range_l2848_284888

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (a + 2) * x

theorem max_value_implies_a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, f a x ≤ f a (1/2)) : 0 < a ∧ a < 2 := by
  sorry

end max_value_implies_a_range_l2848_284888


namespace quadrilateral_count_l2848_284821

/-- The number of points on the circumference of the circle -/
def n : ℕ := 15

/-- The number of vertices required from the circumference -/
def k : ℕ := 3

/-- The number of different convex quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose n k

theorem quadrilateral_count :
  num_quadrilaterals = 455 :=
sorry

end quadrilateral_count_l2848_284821


namespace trigonometric_inequality_l2848_284808

theorem trigonometric_inequality (α : ℝ) : 4 * Real.sin (3 * α) + 5 ≥ 4 * Real.cos (2 * α) + 5 * Real.sin α := by
  sorry

end trigonometric_inequality_l2848_284808


namespace same_commission_list_price_is_65_l2848_284863

-- Define the list price
def list_price : ℝ := 65

-- Define Alice's selling price
def alice_selling_price : ℝ := list_price - 15

-- Define Bob's selling price
def bob_selling_price : ℝ := list_price - 25

-- Define Alice's commission rate
def alice_commission_rate : ℝ := 0.12

-- Define Bob's commission rate
def bob_commission_rate : ℝ := 0.15

-- Theorem stating that Alice and Bob get the same commission
theorem same_commission :
  alice_commission_rate * alice_selling_price = bob_commission_rate * bob_selling_price :=
by sorry

-- Main theorem proving that the list price is 65
theorem list_price_is_65 : list_price = 65 :=
by sorry

end same_commission_list_price_is_65_l2848_284863


namespace K_idempotent_l2848_284867

/-- The set of all 2013 × 2013 arrays with entries 0 and 1 -/
def F : Type := Fin 2013 → Fin 2013 → Fin 2

/-- The sum of all entries sharing a row or column with a[i,j] -/
def S (A : F) (i j : Fin 2013) : ℕ :=
  (Finset.sum (Finset.range 2013) (fun k => A i k)) +
  (Finset.sum (Finset.range 2013) (fun k => A k j)) -
  A i j

/-- The transformation K -/
def K (A : F) : F :=
  fun i j => (S A i j) % 2

/-- The main theorem: K(K(A)) = K(A) for all A in F -/
theorem K_idempotent (A : F) : K (K A) = K A := by sorry

end K_idempotent_l2848_284867


namespace parallel_transitivity_l2848_284898

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  parallel a c → parallel b c → parallel a b := by sorry

end parallel_transitivity_l2848_284898


namespace function_graphs_common_point_l2848_284882

/-- Given real numbers a, b, c, and d, if the graphs of y = 2a + 1/(x-b) and y = 2c + 1/(x-d) 
    have exactly one common point, then the graphs of y = 2b + 1/(x-a) and y = 2d + 1/(x-c) 
    also have exactly one common point. -/
theorem function_graphs_common_point (a b c d : ℝ) :
  (∃! x : ℝ, 2 * a + 1 / (x - b) = 2 * c + 1 / (x - d)) →
  (∃! x : ℝ, 2 * b + 1 / (x - a) = 2 * d + 1 / (x - c)) :=
by sorry

end function_graphs_common_point_l2848_284882


namespace average_income_Q_R_l2848_284823

theorem average_income_Q_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 := by
sorry

end average_income_Q_R_l2848_284823


namespace binary_to_octal_conversion_l2848_284830

/-- Converts a binary number represented as a list of bits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The binary representation of 1010 101₂ -/
def binary_num : List Bool := [true, false, true, false, true, false, true]

/-- The octal representation of 125₈ -/
def octal_num : List ℕ := [1, 2, 5]

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry

#eval binary_to_decimal binary_num
#eval decimal_to_octal (binary_to_decimal binary_num)

end binary_to_octal_conversion_l2848_284830


namespace m_values_l2848_284855

def A : Set ℝ := {x | x^2 - 3*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

theorem m_values : ∀ m : ℝ, (A ∪ B m = A) ↔ (m = 0 ∨ m = -1/2 ∨ m = 1/5) := by sorry

end m_values_l2848_284855


namespace least_three_digit_7_heavy_l2848_284834

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The set of three-digit numbers -/
def three_digit_numbers : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

theorem least_three_digit_7_heavy : 
  ∃ (n : ℕ), n ∈ three_digit_numbers ∧ is_7_heavy n ∧ 
  ∀ (m : ℕ), m ∈ three_digit_numbers → is_7_heavy m → n ≤ m :=
by sorry

end least_three_digit_7_heavy_l2848_284834


namespace isosceles_right_triangle_perimeter_twice_area_l2848_284841

theorem isosceles_right_triangle_perimeter_twice_area :
  ∃! a : ℝ, a > 0 ∧ (2 * a + a * Real.sqrt 2 = 2 * (1 / 2 * a^2)) := by sorry

end isosceles_right_triangle_perimeter_twice_area_l2848_284841


namespace ratatouille_cost_per_quart_l2848_284892

/-- Calculates the cost per quart of ratatouille given ingredient quantities and prices -/
theorem ratatouille_cost_per_quart :
  let eggplant_oz : Real := 88
  let eggplant_price : Real := 0.22
  let zucchini_oz : Real := 60.8
  let zucchini_price : Real := 0.15
  let tomato_oz : Real := 73.6
  let tomato_price : Real := 0.25
  let onion_oz : Real := 43.2
  let onion_price : Real := 0.07
  let basil_oz : Real := 16
  let basil_price : Real := 2.70 / 4
  let bell_pepper_oz : Real := 12
  let bell_pepper_price : Real := 0.20
  let total_yield_quarts : Real := 4.5
  let total_cost : Real := 
    eggplant_oz * eggplant_price +
    zucchini_oz * zucchini_price +
    tomato_oz * tomato_price +
    onion_oz * onion_price +
    basil_oz * basil_price +
    bell_pepper_oz * bell_pepper_price
  let cost_per_quart : Real := total_cost / total_yield_quarts
  cost_per_quart = 14.02 := by
  sorry

end ratatouille_cost_per_quart_l2848_284892


namespace intersection_abscissas_l2848_284813

-- Define the parabola and line equations
def parabola (x : ℝ) : ℝ := x^2 - 4*x
def line : ℝ := 5

-- Define the intersection points
def intersection_points : Set ℝ := {x | parabola x = line}

-- Theorem statement
theorem intersection_abscissas :
  intersection_points = {-1, 5} := by sorry

end intersection_abscissas_l2848_284813


namespace train_length_l2848_284886

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (cross_time : ℝ) : 
  speed_kmph = 72 → cross_time = 7 → speed_kmph * (1000 / 3600) * cross_time = 140 := by
  sorry

end train_length_l2848_284886


namespace binomial_factorial_product_l2848_284872

theorem binomial_factorial_product : (Nat.choose 60 3) * (Nat.factorial 10) = 124467072000 := by
  sorry

end binomial_factorial_product_l2848_284872


namespace division_of_fractions_l2848_284829

theorem division_of_fractions : (5 : ℚ) / 6 / (7 / 4) = 10 / 21 := by sorry

end division_of_fractions_l2848_284829


namespace circle_area_difference_l2848_284824

theorem circle_area_difference : 
  let r1 : ℝ := 20
  let d2 : ℝ := 20
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 300 * π := by sorry

end circle_area_difference_l2848_284824


namespace induction_contrapositive_l2848_284858

theorem induction_contrapositive (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  (¬ P 4) →
  (¬ P 3) :=
by sorry

end induction_contrapositive_l2848_284858


namespace no_simultaneous_squares_l2848_284828

theorem no_simultaneous_squares : ¬∃ (m n : ℕ), ∃ (k l : ℕ), m^2 + n = k^2 ∧ n^2 + m = l^2 := by
  sorry

end no_simultaneous_squares_l2848_284828


namespace average_marks_l2848_284832

theorem average_marks (total_subjects : ℕ) (avg_five_subjects : ℝ) (sixth_subject_marks : ℝ) :
  total_subjects = 6 →
  avg_five_subjects = 74 →
  sixth_subject_marks = 104 →
  (avg_five_subjects * 5 + sixth_subject_marks) / total_subjects = 79 :=
by
  sorry

end average_marks_l2848_284832


namespace smallest_divisible_by_nine_l2848_284804

/-- The smallest digit d such that 528,d46 is divisible by 9 -/
def smallest_digit : ℕ := 2

/-- A function that constructs the number 528,d46 given a digit d -/
def construct_number (d : ℕ) : ℕ := 528000 + d * 100 + 46

theorem smallest_divisible_by_nine :
  (∀ d : ℕ, d < smallest_digit → ¬(9 ∣ construct_number d)) ∧
  (9 ∣ construct_number smallest_digit) :=
sorry

end smallest_divisible_by_nine_l2848_284804


namespace point_on_y_axis_l2848_284811

theorem point_on_y_axis (m : ℝ) :
  (m + 1 = 0) → ((m + 1, m + 4) : ℝ × ℝ) = (0, 3) := by
  sorry

end point_on_y_axis_l2848_284811


namespace cleaning_time_with_help_l2848_284879

-- Define the grove dimensions
def trees_width : ℕ := 4
def trees_height : ℕ := 5

-- Define the initial cleaning time per tree
def initial_cleaning_time : ℕ := 6

-- Define the helper effect (halves the cleaning time)
def helper_effect : ℚ := 1/2

-- Theorem to prove
theorem cleaning_time_with_help :
  let total_trees := trees_width * trees_height
  let cleaning_time_with_help := initial_cleaning_time * helper_effect
  let total_cleaning_time := (total_trees : ℚ) * cleaning_time_with_help
  total_cleaning_time / 60 = 1 := by sorry

end cleaning_time_with_help_l2848_284879


namespace firefly_group_size_l2848_284876

theorem firefly_group_size (butterfly_group_size : ℕ) (min_butterflies : ℕ) 
  (h1 : butterfly_group_size = 44)
  (h2 : min_butterflies = 748) :
  ∃ (firefly_group_size : ℕ),
    firefly_group_size = 
      (((min_butterflies + butterfly_group_size - 1) / butterfly_group_size) * butterfly_group_size) :=
by
  sorry

end firefly_group_size_l2848_284876


namespace smaller_number_of_sum_and_product_l2848_284801

theorem smaller_number_of_sum_and_product (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  min x y = 3 := by
sorry

end smaller_number_of_sum_and_product_l2848_284801


namespace selection_methods_equality_l2848_284835

def num_male_students : ℕ := 20
def num_female_students : ℕ := 30
def total_students : ℕ := num_male_students + num_female_students
def num_selected : ℕ := 4

theorem selection_methods_equality :
  (Nat.choose total_students num_selected - Nat.choose num_male_students num_selected - Nat.choose num_female_students num_selected) =
  (Nat.choose num_male_students 1 * Nat.choose num_female_students 3 +
   Nat.choose num_male_students 2 * Nat.choose num_female_students 2 +
   Nat.choose num_male_students 3 * Nat.choose num_female_students 1) :=
by sorry

end selection_methods_equality_l2848_284835


namespace product_347_6_base9_l2848_284802

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Theorem: The product of 347₉ and 6₉ in base 9 is 2316₉ --/
theorem product_347_6_base9 :
  base10ToBase9 (base9ToBase10 [7, 4, 3] * base9ToBase10 [6]) = [6, 1, 3, 2] := by
  sorry

end product_347_6_base9_l2848_284802


namespace crust_vs_bread_expenditure_l2848_284877

/-- Represents the percentage increase in expenditure when buying crust instead of bread -/
def expenditure_increase : ℝ := 36

/-- The ratio of crust weight to bread weight -/
def crust_weight_ratio : ℝ := 0.75

/-- The ratio of crust price to bread price -/
def crust_price_ratio : ℝ := 1.2

/-- The ratio of bread that is actually consumed -/
def bread_consumption_ratio : ℝ := 0.85

/-- The ratio of crust that is actually consumed -/
def crust_consumption_ratio : ℝ := 1

theorem crust_vs_bread_expenditure :
  expenditure_increase = 
    ((crust_consumption_ratio / crust_weight_ratio) / 
     bread_consumption_ratio * crust_price_ratio - 1) * 100 := by
  sorry

#eval expenditure_increase

end crust_vs_bread_expenditure_l2848_284877


namespace equation_solution_l2848_284817

theorem equation_solution : ∃! x : ℚ, (3 / 4 : ℚ) + 1 / x = 7 / 8 :=
  sorry

end equation_solution_l2848_284817


namespace two_people_two_rooms_probability_prove_two_people_two_rooms_probability_l2848_284839

/-- The probability of two individuals randomly choosing different rooms out of two available rooms -/
theorem two_people_two_rooms_probability : ℝ :=
  1 / 2

/-- Prove that the probability of two individuals randomly choosing different rooms out of two available rooms is 1/2 -/
theorem prove_two_people_two_rooms_probability :
  two_people_two_rooms_probability = 1 / 2 := by
  sorry

end two_people_two_rooms_probability_prove_two_people_two_rooms_probability_l2848_284839


namespace distance_to_x_axis_on_ellipse_l2848_284852

/-- The distance from a point on an ellipse to the x-axis, given specific conditions -/
theorem distance_to_x_axis_on_ellipse (x y : ℝ) : 
  (x^2 / 2 + y^2 / 6 = 1) →  -- Point (x, y) is on the ellipse
  (x * x + (y + 2) * (y - 2) = 0) →  -- Dot product condition
  |y| = Real.sqrt 3 := by
  sorry

end distance_to_x_axis_on_ellipse_l2848_284852


namespace quadratic_equation_solution_l2848_284843

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -9 ∧ x₂ = -1 ∧ 
  (x₁^2 + 10*x₁ + 9 = 0) ∧ 
  (x₂^2 + 10*x₂ + 9 = 0) :=
by sorry

end quadratic_equation_solution_l2848_284843


namespace partition_natural_numbers_l2848_284878

theorem partition_natural_numbers : 
  ∃ (partition : ℕ → Fin 100), 
    (∀ i : Fin 100, ∃ n : ℕ, partition n = i) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → 
      partition a = partition c ∨ 
      partition a = partition b ∨ 
      partition b = partition c) :=
sorry

end partition_natural_numbers_l2848_284878


namespace solve_for_B_l2848_284856

theorem solve_for_B : ∃ B : ℝ, (4 * B + 4 - 3 = 33) ∧ (B = 8) := by sorry

end solve_for_B_l2848_284856


namespace sine_unit_implies_on_y_axis_l2848_284831

-- Define the type for angles
def Angle : Type := ℝ

-- Define the sine function
noncomputable def sine (α : Angle) : ℝ := Real.sin α

-- Define a predicate for a directed line segment of unit length
def is_unit_directed_segment (x : ℝ) : Prop := x = 1 ∨ x = -1

-- Define a predicate for a point being on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Theorem statement
theorem sine_unit_implies_on_y_axis (α : Angle) :
  is_unit_directed_segment (sine α) →
  ∃ (y : ℝ), on_y_axis 0 y ∧ (0, y) = (Real.cos α, Real.sin α) :=
sorry

end sine_unit_implies_on_y_axis_l2848_284831


namespace milk_for_six_cookies_l2848_284807

/-- Represents the number of cups of milk required for a given number of cookies -/
def milkRequired (cookies : ℕ) : ℚ :=
  sorry

theorem milk_for_six_cookies :
  let cookies_per_quart : ℕ := 24 / 4
  let pints_per_quart : ℕ := 2
  let cups_per_pint : ℕ := 2
  milkRequired 6 = 4 := by
  sorry

end milk_for_six_cookies_l2848_284807


namespace isosceles_triangle_side_lengths_l2848_284883

theorem isosceles_triangle_side_lengths 
  (perimeter : ℝ) 
  (height : ℝ) 
  (is_isosceles : Bool) 
  (h1 : perimeter = 16) 
  (h2 : height = 4) 
  (h3 : is_isosceles = true) : 
  ∃ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 ∧ a + b + c = perimeter := by
  sorry

end isosceles_triangle_side_lengths_l2848_284883


namespace marnie_chips_consumption_l2848_284833

/-- Given a bag of chips and Marnie's eating pattern, calculate the number of days to finish the bag -/
def days_to_finish_chips (total_chips : ℕ) (first_day_consumption : ℕ) (daily_consumption : ℕ) : ℕ :=
  1 + ((total_chips - first_day_consumption) + daily_consumption - 1) / daily_consumption

/-- Theorem: It takes Marnie 10 days to eat the whole bag of chips -/
theorem marnie_chips_consumption :
  days_to_finish_chips 100 10 10 = 10 := by
  sorry

#eval days_to_finish_chips 100 10 10

end marnie_chips_consumption_l2848_284833


namespace stool_sticks_calculation_l2848_284812

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a table makes -/
def table_sticks : ℕ := 9

/-- The number of sticks Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped up -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

theorem stool_sticks_calculation :
  stool_sticks * stools_chopped = 
    hours_warm * sticks_per_hour - 
    (chair_sticks * chairs_chopped + table_sticks * tables_chopped) :=
by sorry

end stool_sticks_calculation_l2848_284812


namespace existence_of_integers_l2848_284848

theorem existence_of_integers : ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 := by
  sorry

end existence_of_integers_l2848_284848


namespace undefined_expression_l2848_284881

theorem undefined_expression (x : ℝ) : 
  (x^2 - 22*x + 121 = 0) ↔ (x = 11) := by sorry

#check undefined_expression

end undefined_expression_l2848_284881


namespace average_weight_section_A_l2848_284884

theorem average_weight_section_A (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_B : ℝ) (avg_weight_total : ℝ) :
  students_A = 24 →
  students_B = 16 →
  avg_weight_B = 35 →
  avg_weight_total = 38 →
  (students_A * avg_weight_section_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total →
  avg_weight_section_A = 40 := by
  sorry

#check average_weight_section_A

end average_weight_section_A_l2848_284884


namespace locus_of_centers_l2848_284816

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 3)² + y² = 25 -/
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C₁ if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and
    internally tangent to C₂ satisfies the equation 4a² + 4b² - 52a - 169 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) →
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end locus_of_centers_l2848_284816


namespace digit_arrangement_count_l2848_284820

theorem digit_arrangement_count : 
  let digits : List ℕ := [4, 7, 5, 2, 0]
  let n : ℕ := digits.length
  let non_zero_digits : List ℕ := digits.filter (· ≠ 0)
  96 = (n - 1) * Nat.factorial (non_zero_digits.length) := by
  sorry

end digit_arrangement_count_l2848_284820


namespace largest_divisible_n_l2848_284805

theorem largest_divisible_n : ∃ (n : ℕ), n = 180 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 20) ∣ (m^3 + 1000))) ∧ 
  ((n + 20) ∣ (n^3 + 1000)) := by
  sorry

end largest_divisible_n_l2848_284805


namespace complex_division_by_i_l2848_284800

theorem complex_division_by_i (z : ℂ) : z.re = -2 ∧ z.im = -1 → z / Complex.I = -1 + 2 * Complex.I := by
  sorry

end complex_division_by_i_l2848_284800
