import Mathlib

namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1659_165929

-- Define the equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (6 - m) = 1 ∧ 
  m - 2 > 0 ∧ 6 - m > 0 ∧ m - 2 ≠ 6 - m

-- Define the condition
def condition (m : ℝ) : Prop :=
  2 < m ∧ m < 6

-- Theorem stating that the condition is necessary but not sufficient
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → condition m) ∧
  ¬(∀ m : ℝ, condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1659_165929


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1659_165919

theorem complex_sum_problem (a b c d g h : ℂ) : 
  b = 4 →
  g = -a - c →
  a + b * I + c + d * I + g + h * I = 3 * I →
  d + h = -1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1659_165919


namespace NUMINAMATH_CALUDE_fire_chief_hats_l1659_165983

theorem fire_chief_hats (o_brien_current : ℕ) (h1 : o_brien_current = 34) : ∃ (simpson : ℕ),
  simpson = 15 ∧ o_brien_current + 1 = 2 * simpson + 5 := by
  sorry

end NUMINAMATH_CALUDE_fire_chief_hats_l1659_165983


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1659_165939

/-- An arithmetic sequence with common difference d ≠ 0 -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def FormGeometricSequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ArithmeticSequence a d)
  (h_geom : FormGeometricSequence (a 2) (a 3) (a 6)) :
  ∃ q : ℝ, q = 3 ∧ FormGeometricSequence (a 2) (a 3) (a 6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1659_165939


namespace NUMINAMATH_CALUDE_has_18_divisors_smallest_with_18_divisors_smallest_integer_with_18_divisors_l1659_165977

/-- A function that counts the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- 131072 has exactly 18 positive divisors -/
theorem has_18_divisors : countDivisors 131072 = 18 := sorry

/-- For any positive integer smaller than 131072, 
    the number of its positive divisors is not 18 -/
theorem smallest_with_18_divisors (n : ℕ) : 
  0 < n → n < 131072 → countDivisors n ≠ 18 := sorry

/-- 131072 is the smallest positive integer with exactly 18 positive divisors -/
theorem smallest_integer_with_18_divisors : 
  ∀ n : ℕ, 0 < n → countDivisors n = 18 → n ≥ 131072 := by
  sorry

end NUMINAMATH_CALUDE_has_18_divisors_smallest_with_18_divisors_smallest_integer_with_18_divisors_l1659_165977


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_of_two_unbounded_l1659_165952

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: For any positive real number M, there exists a positive integer n 
    such that the sum of the digits of 2^n is greater than M -/
theorem sum_of_digits_of_power_of_two_unbounded :
  ∀ M : ℝ, M > 0 → ∃ n : ℕ, (sumOfDigits (2^n : ℕ)) > M := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_of_two_unbounded_l1659_165952


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1659_165968

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 → area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio ∧ h_large = 15 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1659_165968


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1659_165936

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a1 : a 1 = 3)
  (h_a3 : a 3 = 12) :
  a 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1659_165936


namespace NUMINAMATH_CALUDE_hexagon_toothpicks_l1659_165962

/-- Represents a hexagonal pattern of small equilateral triangles -/
structure HexagonalPattern :=
  (max_row_triangles : ℕ)

/-- Calculates the total number of small triangles in the hexagonal pattern -/
def total_triangles (h : HexagonalPattern) : ℕ :=
  let half_triangles := (h.max_row_triangles * (h.max_row_triangles + 1)) / 2
  2 * half_triangles + h.max_row_triangles

/-- Calculates the number of boundary toothpicks in the hexagonal pattern -/
def boundary_toothpicks (h : HexagonalPattern) : ℕ :=
  6 * h.max_row_triangles

/-- Calculates the total number of toothpicks required to construct the hexagonal pattern -/
def total_toothpicks (h : HexagonalPattern) : ℕ :=
  3 * total_triangles h - boundary_toothpicks h

/-- Theorem stating that a hexagonal pattern with 1001 triangles in its largest row requires 3006003 toothpicks -/
theorem hexagon_toothpicks :
  let h : HexagonalPattern := ⟨1001⟩
  total_toothpicks h = 3006003 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_toothpicks_l1659_165962


namespace NUMINAMATH_CALUDE_complex_point_to_number_l1659_165959

theorem complex_point_to_number (z : ℂ) : (z / Complex.I).re = 3 ∧ (z / Complex.I).im = -1 → z = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_point_to_number_l1659_165959


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1659_165993

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 10 ∧ x₂ = 2 - Real.sqrt 10) ∧ 
  (x₁^2 - 4*x₁ - 6 = 0 ∧ x₂^2 - 4*x₂ - 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1659_165993


namespace NUMINAMATH_CALUDE_ab_equals_zero_l1659_165946

theorem ab_equals_zero (a b : ℤ) (h : |a - b| + |a * b| = 2) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l1659_165946


namespace NUMINAMATH_CALUDE_exists_n_plus_S_n_eq_1980_l1659_165900

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of n such that n + S(n) = 1980 -/
theorem exists_n_plus_S_n_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

end NUMINAMATH_CALUDE_exists_n_plus_S_n_eq_1980_l1659_165900


namespace NUMINAMATH_CALUDE_tetrahedron_vector_equality_l1659_165974

-- Define the tetrahedron O-ABC
variable (O A B C : EuclideanSpace ℝ (Fin 3))

-- Define vectors a, b, c
variable (a b c : EuclideanSpace ℝ (Fin 3))

-- Define points M and N
variable (M N : EuclideanSpace ℝ (Fin 3))

-- State the theorem
theorem tetrahedron_vector_equality 
  (h1 : A - O = a) 
  (h2 : B - O = b) 
  (h3 : C - O = c) 
  (h4 : M - O = (2/3) • (A - O)) 
  (h5 : N - O = (1/2) • (B - O) + (1/2) • (C - O)) :
  M - N = (1/2) • b + (1/2) • c - (2/3) • a := by sorry

end NUMINAMATH_CALUDE_tetrahedron_vector_equality_l1659_165974


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1659_165980

/-- Represents a cylindrical water tank with a given capacity and initial water level. -/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ

/-- Proves that a water tank with the given properties has a capacity of 30 liters. -/
theorem water_tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 6)
  (h2 : (tank.initialWater + 5) / tank.capacity = 1 / 3) :
  tank.capacity = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1659_165980


namespace NUMINAMATH_CALUDE_parabola_focus_l1659_165956

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k) -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -2x^2 - 4x + 1, its focus is at (-1, 23/8) -/
theorem parabola_focus (p : Parabola) (f : Focus) :
  p.a = -2 ∧ p.b = -4 ∧ p.c = 1 →
  f.h = -1 ∧ f.k = 23/8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1659_165956


namespace NUMINAMATH_CALUDE_warriors_height_order_l1659_165905

theorem warriors_height_order (heights : Set ℝ) (h : Set.Infinite heights) :
  ∃ (subseq : ℕ → ℝ), (∀ n, subseq n ∈ heights) ∧ 
    (Set.Infinite (Set.range subseq)) ∧ 
    (∀ n m, n < m → subseq n < subseq m) :=
sorry

end NUMINAMATH_CALUDE_warriors_height_order_l1659_165905


namespace NUMINAMATH_CALUDE_modulus_of_x_is_sqrt_10_l1659_165963

-- Define the complex number x
def x : ℂ := sorry

-- State the theorem
theorem modulus_of_x_is_sqrt_10 :
  x + Complex.I = (2 - Complex.I) / Complex.I →
  Complex.abs x = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_x_is_sqrt_10_l1659_165963


namespace NUMINAMATH_CALUDE_adam_shopping_cost_l1659_165921

/-- The total cost of Adam's shopping given the number of sandwiches, 
    price per sandwich, and price of water. -/
def total_cost (num_sandwiches : ℕ) (price_per_sandwich : ℕ) (price_of_water : ℕ) : ℕ :=
  num_sandwiches * price_per_sandwich + price_of_water

/-- Theorem stating that Adam's total shopping cost is $11 -/
theorem adam_shopping_cost : 
  total_cost 3 3 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_adam_shopping_cost_l1659_165921


namespace NUMINAMATH_CALUDE_factors_of_60_l1659_165994

-- Define the number of positive factors function
def num_positive_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- State the theorem
theorem factors_of_60 : num_positive_factors 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_60_l1659_165994


namespace NUMINAMATH_CALUDE_company_plants_1500_trees_l1659_165931

/-- Represents the number of trees chopped down in the first half of the year -/
def trees_chopped_first_half : ℕ := 200

/-- Represents the number of trees chopped down in the second half of the year -/
def trees_chopped_second_half : ℕ := 300

/-- Represents the number of trees to be planted for each tree chopped down -/
def trees_planted_per_chopped : ℕ := 3

/-- Calculates the total number of trees that need to be planted -/
def trees_to_plant : ℕ := (trees_chopped_first_half + trees_chopped_second_half) * trees_planted_per_chopped

/-- Theorem stating that the company needs to plant 1500 trees -/
theorem company_plants_1500_trees : trees_to_plant = 1500 := by
  sorry

end NUMINAMATH_CALUDE_company_plants_1500_trees_l1659_165931


namespace NUMINAMATH_CALUDE_average_weight_problem_l1659_165924

/-- Given the average weight of three people and two subsets of them, prove the average weight of the remaining subset. -/
theorem average_weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 60)  -- Average weight of a, b, and c is 60 kg
  (h2 : (b + c) / 2 = 50)      -- Average weight of b and c is 50 kg
  (h3 : b = 60)                -- Weight of b is 60 kg
  : (a + b) / 2 = 70 :=        -- Average weight of a and b is 70 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1659_165924


namespace NUMINAMATH_CALUDE_largest_prime_with_2023_digits_p_is_prime_p_has_2023_digits_smallest_k_divisible_by_30_l1659_165907

/-- The largest prime with 2023 digits -/
def p : ℕ := sorry

theorem largest_prime_with_2023_digits (q : ℕ) (h : q > p) : ¬ Prime q := sorry

theorem p_is_prime : Prime p := sorry

theorem p_has_2023_digits : (Nat.digits 10 p).length = 2023 := sorry

theorem smallest_k_divisible_by_30 : 
  ∃ k : ℕ, k > 0 ∧ 30 ∣ (p^3 - k) ∧ ∀ m : ℕ, 0 < m ∧ m < k → ¬(30 ∣ (p^3 - m)) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_with_2023_digits_p_is_prime_p_has_2023_digits_smallest_k_divisible_by_30_l1659_165907


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l1659_165922

theorem divisibility_of_expression (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (ha_gt_7 : a > 7) (hb_gt_7 : b > 7) : 
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l1659_165922


namespace NUMINAMATH_CALUDE_parallel_lines_condition_not_sufficient_nor_necessary_l1659_165991

theorem parallel_lines_condition_not_sufficient_nor_necessary (a : ℝ) :
  ¬(((a = 1) ↔ (∃ k : ℝ, k ≠ 0 ∧ a^2 = 1 ∧ k = 1))) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_not_sufficient_nor_necessary_l1659_165991


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1659_165930

theorem triangle_side_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / b^2 ≥ 1 ∧ ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧ (a'^2 + c'^2) / b'^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1659_165930


namespace NUMINAMATH_CALUDE_power_function_fixed_point_l1659_165901

theorem power_function_fixed_point (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^α
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_fixed_point_l1659_165901


namespace NUMINAMATH_CALUDE_log_inequality_l1659_165984

def number_of_distinct_prime_divisors (n : ℕ) : ℕ := sorry

theorem log_inequality (n : ℕ) (k : ℕ) (h : k = number_of_distinct_prime_divisors n) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1659_165984


namespace NUMINAMATH_CALUDE_pauls_cousin_score_l1659_165915

theorem pauls_cousin_score (paul_score : ℕ) (total_score : ℕ) 
  (h1 : paul_score = 3103)
  (h2 : total_score = 5816) :
  total_score - paul_score = 2713 :=
by sorry

end NUMINAMATH_CALUDE_pauls_cousin_score_l1659_165915


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1659_165988

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1659_165988


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1659_165943

/-- Represents a square quilt -/
structure Quilt :=
  (size : ℕ)
  (shaded_diagonal_squares : ℕ)
  (shaded_full_squares : ℕ)

/-- Calculates the fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  (q.shaded_diagonal_squares / 2 + q.shaded_full_squares : ℚ) / (q.size * q.size)

/-- Theorem stating the fraction of the quilt that is shaded -/
theorem quilt_shaded_fraction :
  ∀ q : Quilt, 
    q.size = 4 → 
    q.shaded_diagonal_squares = 4 → 
    q.shaded_full_squares = 1 → 
    shaded_fraction q = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1659_165943


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_l1659_165914

/-- Given an ellipse with equation x²/24 + y²/49 = 1, 
    prove that the equation of the hyperbola whose vertices are the foci of this ellipse 
    and whose foci are the vertices of this ellipse is y²/25 - x²/24 = 1 -/
theorem hyperbola_from_ellipse (x y : ℝ) :
  (x^2 / 24 + y^2 / 49 = 1) →
  ∃ (x' y' : ℝ), (y'^2 / 25 - x'^2 / 24 = 1 ∧ 
    (∀ (a b c : ℝ), (a^2 = 49 ∧ b^2 = 24 ∧ c^2 = a^2 - b^2) →
      ((x' = 0 ∧ y' = c) ∨ (x' = 0 ∧ y' = -c)) ∧
      ((x' = 0 ∧ y' = a) ∨ (x' = 0 ∧ y' = -a)))) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_from_ellipse_l1659_165914


namespace NUMINAMATH_CALUDE_max_a_part_1_range_a_part_2_l1659_165932

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem max_a_part_1 : 
  (∃ a_max : ℝ, ∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) → a ≤ a_max) ∧
  (∀ x : ℝ, g x ≤ 5 → f 1 x ≤ 6) :=
sorry

-- Part II
theorem range_a_part_2 : 
  {a : ℝ | ∀ x : ℝ, f a x + g x ≥ 3} = {a : ℝ | a ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_max_a_part_1_range_a_part_2_l1659_165932


namespace NUMINAMATH_CALUDE_P_below_line_l1659_165926

/-- A line in 2D space represented by the equation 2x - y + 3 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point P with coordinates (1, -1) -/
def P : Point := ⟨1, -1⟩

/-- A point is below the line if 2x - y + 3 > 0 -/
def is_below (p : Point) : Prop := 2 * p.x - p.y + 3 > 0

theorem P_below_line : is_below P := by
  sorry

end NUMINAMATH_CALUDE_P_below_line_l1659_165926


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1659_165928

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ,
  x₁^2 - 4*x₁ + 2 = 0 →
  x₂^2 - 4*x₂ + 2 = 0 →
  x₁ + x₂ - x₁*x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1659_165928


namespace NUMINAMATH_CALUDE_sales_tax_difference_l1659_165975

-- Define the item price
def item_price : ℝ := 20

-- Define the two tax rates
def tax_rate_1 : ℝ := 0.065
def tax_rate_2 : ℝ := 0.06

-- State the theorem
theorem sales_tax_difference : 
  (tax_rate_1 - tax_rate_2) * item_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l1659_165975


namespace NUMINAMATH_CALUDE_min_value_and_y_l1659_165965

theorem min_value_and_y (x y z : ℝ) (h : 2*x - 3*y + z = 3) :
  ∃ (min_val : ℝ), 
    (∀ x' y' z' : ℝ, 2*x' - 3*y' + z' = 3 → x'^2 + (y'-1)^2 + z'^2 ≥ min_val) ∧
    (x^2 + (y-1)^2 + z^2 = min_val) ∧
    (min_val = 18/7) ∧
    (y = -2/7) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_y_l1659_165965


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l1659_165937

def original_price : ℝ := 103.5
def sale_price : ℝ := 78.2
def price_increase_percentage : ℝ := 25
def price_difference : ℝ := 5.75

theorem discount_percentage_proof :
  ∃ (discount_percentage : ℝ),
    sale_price = original_price - (discount_percentage / 100) * original_price ∧
    original_price - (sale_price + price_increase_percentage / 100 * sale_price) = price_difference ∧
    (discount_percentage ≥ 24.43 ∧ discount_percentage ≤ 24.45) := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l1659_165937


namespace NUMINAMATH_CALUDE_inverse_implies_negation_angle_60_iff_arithmetic_sequence_not_necessary_condition_xy_not_necessary_condition_ab_l1659_165964

-- Proposition 1
theorem inverse_implies_negation (P : Prop) : 
  (¬P → P) → (P → ¬P) := by sorry

-- Proposition 2
theorem angle_60_iff_arithmetic_sequence (A B C : ℝ) : 
  (B = 60 ∧ A + B + C = 180) ↔ (∃ d : ℝ, A = B - d ∧ C = B + d) := by sorry

-- Proposition 3 (counterexample)
theorem not_necessary_condition_xy : 
  ∃ x y : ℝ, x + y > 3 ∧ x * y > 2 ∧ ¬(x > 1 ∧ y > 2) := by sorry

-- Proposition 4 (counterexample)
theorem not_necessary_condition_ab : 
  ∃ a b m : ℝ, a < b ∧ ¬(a * m^2 < b * m^2) := by sorry

end NUMINAMATH_CALUDE_inverse_implies_negation_angle_60_iff_arithmetic_sequence_not_necessary_condition_xy_not_necessary_condition_ab_l1659_165964


namespace NUMINAMATH_CALUDE_ant_height_proof_l1659_165904

theorem ant_height_proof (rope_length : ℝ) (base_distance : ℝ) (shadow_rate : ℝ) (time : ℝ) 
  (h_rope : rope_length = 10)
  (h_base : base_distance = 6)
  (h_rate : shadow_rate = 0.3)
  (h_time : time = 5)
  (h_right_triangle : rope_length ^ 2 = base_distance ^ 2 + (rope_length ^ 2 - base_distance ^ 2))
  : ∃ (height : ℝ), 
    height = 2 ∧ 
    (shadow_rate * time) / base_distance = height / (rope_length ^ 2 - base_distance ^ 2).sqrt :=
by sorry

end NUMINAMATH_CALUDE_ant_height_proof_l1659_165904


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1659_165982

theorem rectangular_to_polar_conversion :
  ∃ (r : ℝ) (θ : ℝ), 
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = 2 ∧
    r * Real.sin θ = -2 ∧
    r = 2 * Real.sqrt 2 ∧
    θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1659_165982


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_nine_l1659_165916

theorem sum_of_cubes_divisible_by_nine (x : ℤ) : 
  ∃ k : ℤ, (x - 1)^3 + x^3 + (x + 1)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_nine_l1659_165916


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l1659_165941

/-- Given a line y = a intersecting the parabola y = x^2 at points A and B, 
    and a point C on the parabola such that angle ACB is a right angle, 
    the range of possible values for a is [1, +∞) -/
theorem parabola_intersection_range (a : ℝ) : 
  (∃ A B C : ℝ × ℝ, 
    (A.2 = a ∧ A.2 = A.1^2) ∧ 
    (B.2 = a ∧ B.2 = B.1^2) ∧ 
    (C.2 = C.1^2) ∧ 
    ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0)) 
  ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l1659_165941


namespace NUMINAMATH_CALUDE_beam_width_calculation_beam_width_250_pounds_l1659_165951

/-- The maximum load a beam can support is directly proportional to its width -/
def load_proportional_to_width (load width : ℝ) : Prop :=
  ∃ k : ℝ, load = k * width

/-- Theorem: Given the proportionality between load and width, and a reference beam,
    calculate the width of a beam supporting a specific load -/
theorem beam_width_calculation
  (reference_width reference_load target_load : ℝ)
  (h_positive : reference_width > 0 ∧ reference_load > 0 ∧ target_load > 0)
  (h_prop : load_proportional_to_width reference_load reference_width)
  (h_prop_target : load_proportional_to_width target_load (target_load * reference_width / reference_load)) :
  load_proportional_to_width target_load ((target_load * reference_width) / reference_load) :=
by sorry

/-- The width of a beam supporting 250 pounds, given a reference beam of 3.5 inches
    supporting 583.3333 pounds, is 1.5 inches -/
theorem beam_width_250_pounds :
  (250 : ℝ) * (3.5 : ℝ) / (583.3333 : ℝ) = (1.5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_beam_width_calculation_beam_width_250_pounds_l1659_165951


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1659_165979

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition "a = 0" is necessary but not sufficient for a complex number z = a + bi to be purely imaginary. -/
theorem a_zero_necessary_not_sufficient :
  (∀ z : ℂ, is_purely_imaginary z → z.re = 0) ∧
  ¬(∀ z : ℂ, z.re = 0 → is_purely_imaginary z) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1659_165979


namespace NUMINAMATH_CALUDE_function_inequality_and_ratio_l1659_165960

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Define the maximum value T
def T : ℝ := 3

-- Theorem statement
theorem function_inequality_and_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = T) :
  (∀ x : ℝ, f x ≥ T) ∧ (2 / (1/a + 1/b) ≤ Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_and_ratio_l1659_165960


namespace NUMINAMATH_CALUDE_accident_insurance_probability_l1659_165955

theorem accident_insurance_probability (p1 p2 : ℝ) 
  (h1 : p1 = 1 / 20)
  (h2 : p2 = 1 / 21)
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1)
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) :
  1 - (1 - p1) * (1 - p2) = 2 / 21 := by
sorry


end NUMINAMATH_CALUDE_accident_insurance_probability_l1659_165955


namespace NUMINAMATH_CALUDE_second_to_first_ratio_l1659_165923

theorem second_to_first_ratio (x y z : ℝ) : 
  y = 90 →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  y / x = 2 := by
sorry

end NUMINAMATH_CALUDE_second_to_first_ratio_l1659_165923


namespace NUMINAMATH_CALUDE_fraction_of_girls_l1659_165986

theorem fraction_of_girls (total : ℕ) (boys : ℕ) (h1 : total = 45) (h2 : boys = 30) :
  (total - boys : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_l1659_165986


namespace NUMINAMATH_CALUDE_triangle_fence_problem_l1659_165944

theorem triangle_fence_problem (a b c : ℕ) : 
  a ≤ b → b ≤ c → 
  a + b + c = 2022 → 
  c - b = 1 → 
  b - a = 2 → 
  b = 674 := by
sorry

end NUMINAMATH_CALUDE_triangle_fence_problem_l1659_165944


namespace NUMINAMATH_CALUDE_homework_problem_distribution_l1659_165970

theorem homework_problem_distribution (total : ℕ) (true_false : ℕ) : 
  total = 45 → true_false = 6 → ∃ (multiple_choice free_response : ℕ),
    multiple_choice = 2 * free_response ∧
    free_response > true_false ∧
    multiple_choice + free_response + true_false = total ∧
    free_response - true_false = 7 :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_distribution_l1659_165970


namespace NUMINAMATH_CALUDE_lemon_ratio_l1659_165999

-- Define the number of lemons for each person
def levi_lemons : ℕ := 5
def jayden_lemons : ℕ := levi_lemons + 6
def ian_lemons : ℕ := 66  -- This is derived from the total, not given directly
def eli_lemons : ℕ := ian_lemons / 2
def total_lemons : ℕ := 115

-- Theorem statement
theorem lemon_ratio : 
  levi_lemons = 5 ∧
  jayden_lemons = levi_lemons + 6 ∧
  eli_lemons = ian_lemons / 2 ∧
  levi_lemons + jayden_lemons + eli_lemons + ian_lemons = total_lemons ∧
  total_lemons = 115 →
  jayden_lemons * 3 = eli_lemons := by
  sorry

end NUMINAMATH_CALUDE_lemon_ratio_l1659_165999


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1659_165957

/-- Given a man who is 24 years older than his son, and the son's present age is 22,
    prove that it will take 2 years for the man's age to be twice the age of his son. -/
theorem mans_age_twice_sons (man_age son_age future_years : ℕ) : 
  man_age = son_age + 24 →
  son_age = 22 →
  future_years = 2 →
  (man_age + future_years) = 2 * (son_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1659_165957


namespace NUMINAMATH_CALUDE_power_multiplication_l1659_165953

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1659_165953


namespace NUMINAMATH_CALUDE_julia_tag_game_l1659_165910

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 16

/-- The difference in the number of kids Julia played with on Monday compared to Tuesday -/
def difference : ℕ := 12

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tag_game :
  tuesday_kids = 4 :=
sorry

end NUMINAMATH_CALUDE_julia_tag_game_l1659_165910


namespace NUMINAMATH_CALUDE_simplify_expression_l1659_165950

theorem simplify_expression (x y : ℝ) :
  let P := 2 * x + 3 * y
  let Q := x - 2 * y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (8 * x^2 - 4 * x * y - 24 * y^2) / (3 * x^2 + 16 * x * y + 5 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1659_165950


namespace NUMINAMATH_CALUDE_cans_restocked_day2_is_1500_l1659_165903

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  total_cans_given : ℕ

/-- Calculates the number of cans restocked after the second day --/
def cans_restocked_day2 (fb : FoodBank) : ℕ :=
  fb.total_cans_given - (fb.initial_stock - fb.day1_people * fb.day1_cans_per_person + fb.day1_restock - fb.day2_people * fb.day2_cans_per_person)

/-- Theorem stating that for the given scenario, the number of cans restocked after the second day is 1500 --/
theorem cans_restocked_day2_is_1500 (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_cans_per_person = 2)
  (h7 : fb.total_cans_given = 2500) :
  cans_restocked_day2 fb = 1500 := by
  sorry

end NUMINAMATH_CALUDE_cans_restocked_day2_is_1500_l1659_165903


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1659_165996

theorem min_value_quadratic (x y : ℝ) (h1 : |y| ≤ 1) (h2 : 2 * x + y = 1) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x' y' : ℝ), |y'| ≤ 1 → 2 * x' + y' = 1 →
    2 * x'^2 + 16 * x' + 3 * y'^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1659_165996


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1659_165942

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1659_165942


namespace NUMINAMATH_CALUDE_average_increase_theorem_l1659_165992

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (stats : CricketStats) (newInningsRuns : ℕ) : ℚ :=
  (stats.totalRuns + newInningsRuns) / (stats.innings + 1)

theorem average_increase_theorem (initialStats : CricketStats) :
  initialStats.innings = 9 →
  newAverage initialStats 200 = initialStats.average + 8 →
  newAverage initialStats 200 = 128 := by
  sorry


end NUMINAMATH_CALUDE_average_increase_theorem_l1659_165992


namespace NUMINAMATH_CALUDE_smallest_angle_in_characteristic_triangle_l1659_165925

/-- A characteristic triangle is a triangle where one interior angle is twice another. -/
structure CharacteristicTriangle where
  α : ℝ  -- The larger angle (characteristic angle)
  β : ℝ  -- The smaller angle
  γ : ℝ  -- The third angle
  angle_sum : α + β + γ = 180
  characteristic : α = 2 * β

/-- The smallest angle in a characteristic triangle with characteristic angle 100° is 30°. -/
theorem smallest_angle_in_characteristic_triangle :
  ∀ (t : CharacteristicTriangle), t.α = 100 → min t.α (min t.β t.γ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_characteristic_triangle_l1659_165925


namespace NUMINAMATH_CALUDE_negation_equivalence_l1659_165909

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1659_165909


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l1659_165934

theorem hyperbola_midpoint_existence :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - y₁^2/9 = 1) ∧
    (x₂^2 - y₂^2/9 = 1) ∧
    ((x₁ + x₂)/2 = -1) ∧
    ((y₁ + y₂)/2 = -4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l1659_165934


namespace NUMINAMATH_CALUDE_m_range_l1659_165998

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Ici (3/2), f (x/m) - 4*m^2 * f x ≤ f (x-1) + 4 * f m) →
  m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1659_165998


namespace NUMINAMATH_CALUDE_max_rooks_300x300_l1659_165969

/-- Represents a chessboard with side length n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a valid rook placement on a chessboard -/
structure RookPlacement (n : ℕ) where
  board : Chessboard n
  rooks : Finset (ℕ × ℕ)
  valid : ∀ (r1 r2 : ℕ × ℕ), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 →
    (r1.1 = r2.1 ∨ r1.2 = r2.2) → 
    (∀ r3 ∈ rooks, r3 ≠ r1 ∧ r3 ≠ r2 → r3.1 ≠ r1.1 ∧ r3.2 ≠ r1.2 ∧ r3.1 ≠ r2.1 ∧ r3.2 ≠ r2.2)

/-- The maximum number of rooks that can be placed on a 300x300 chessboard
    such that each rook attacks no more than one other rook is 400 -/
theorem max_rooks_300x300 : 
  ∀ (p : RookPlacement 300), Finset.card p.rooks ≤ 400 ∧ 
  ∃ (p' : RookPlacement 300), Finset.card p'.rooks = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_rooks_300x300_l1659_165969


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1659_165917

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem fifth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_third : a 3 = -4) 
  (h_seventh : a 7 = -16) : 
  a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1659_165917


namespace NUMINAMATH_CALUDE_count_three_digit_multiples_of_15_l1659_165958

theorem count_three_digit_multiples_of_15 : 
  (Finset.filter (λ x => x % 15 = 0) (Finset.range 900)).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_count_three_digit_multiples_of_15_l1659_165958


namespace NUMINAMATH_CALUDE_sibling_difference_l1659_165940

/-- Given the number of siblings for Masud, calculate the number of siblings for Janet -/
def janet_siblings (masud_siblings : ℕ) : ℕ :=
  4 * masud_siblings - 60

/-- Given the number of siblings for Masud, calculate the number of siblings for Carlos -/
def carlos_siblings (masud_siblings : ℕ) : ℕ :=
  (3 * masud_siblings) / 4

/-- Theorem stating the difference in siblings between Janet and Carlos -/
theorem sibling_difference (masud_siblings : ℕ) (h : masud_siblings = 60) :
  janet_siblings masud_siblings - carlos_siblings masud_siblings = 135 := by
  sorry


end NUMINAMATH_CALUDE_sibling_difference_l1659_165940


namespace NUMINAMATH_CALUDE_cos_seventh_power_decomposition_l1659_165945

theorem cos_seventh_power_decomposition :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (Real.cos θ)^7 = b₁ * Real.cos θ + b₂ * Real.cos (2*θ) + b₃ * Real.cos (3*θ) + 
                               b₄ * Real.cos (4*θ) + b₅ * Real.cos (5*θ) + b₆ * Real.cos (6*θ) + 
                               b₇ * Real.cos (7*θ)) ∧
    (b₁ = 35/64 ∧ b₂ = 0 ∧ b₃ = 21/64 ∧ b₄ = 0 ∧ b₅ = 7/64 ∧ b₆ = 0 ∧ b₇ = 1/64) ∧
    (b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 1785/4096) := by
  sorry

end NUMINAMATH_CALUDE_cos_seventh_power_decomposition_l1659_165945


namespace NUMINAMATH_CALUDE_triangle_base_increase_l1659_165920

theorem triangle_base_increase (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) :
  let new_height := 0.95 * h
  let new_area := 1.045 * (0.5 * b * h)
  ∃ x : ℝ, x > 0 ∧ new_area = 0.5 * (b * (1 + x / 100)) * new_height ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_base_increase_l1659_165920


namespace NUMINAMATH_CALUDE_john_running_speed_equation_l1659_165911

theorem john_running_speed_equation :
  ∃ (x : ℝ), x > 0 ∧ 6.6 * x^2 - 31.6 * x - 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_john_running_speed_equation_l1659_165911


namespace NUMINAMATH_CALUDE_max_value_xy_expression_l1659_165978

theorem max_value_xy_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x + 5*y < 90) :
  xy*(90 - 4*x - 5*y) ≤ 1350 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4*x₀ + 5*y₀ < 90 ∧ x₀*y₀*(90 - 4*x₀ - 5*y₀) = 1350 :=
by sorry

end NUMINAMATH_CALUDE_max_value_xy_expression_l1659_165978


namespace NUMINAMATH_CALUDE_mode_is_highest_rectangle_middle_l1659_165949

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  -- Add necessary fields here

/-- The mode of a frequency distribution --/
def mode (h : FrequencyHistogram) : ℝ :=
  sorry

/-- The middle position of the highest rectangle in a frequency histogram --/
def highestRectangleMiddle (h : FrequencyHistogram) : ℝ :=
  sorry

/-- Theorem stating that the mode corresponds to the middle of the highest rectangle --/
theorem mode_is_highest_rectangle_middle (h : FrequencyHistogram) :
  mode h = highestRectangleMiddle h :=
sorry

end NUMINAMATH_CALUDE_mode_is_highest_rectangle_middle_l1659_165949


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1659_165948

theorem solve_linear_equation (y : ℚ) (h : -3 * y - 8 = 10 * y + 5) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1659_165948


namespace NUMINAMATH_CALUDE_yangtze_length_scientific_notation_l1659_165902

/-- The length of the Yangtze River in meters -/
def yangtze_length : ℕ := 6300000

/-- The scientific notation representation of the Yangtze River's length -/
def yangtze_scientific : ℝ := 6.3 * (10 ^ 6)

theorem yangtze_length_scientific_notation : 
  (yangtze_length : ℝ) = yangtze_scientific := by sorry

end NUMINAMATH_CALUDE_yangtze_length_scientific_notation_l1659_165902


namespace NUMINAMATH_CALUDE_highlighters_count_l1659_165967

/-- The number of pink highlighters in the desk -/
def pink_highlighters : ℕ := 7

/-- The number of yellow highlighters in the desk -/
def yellow_highlighters : ℕ := 4

/-- The number of blue highlighters in the desk -/
def blue_highlighters : ℕ := 5

/-- The number of green highlighters in the desk -/
def green_highlighters : ℕ := 3

/-- The number of orange highlighters in the desk -/
def orange_highlighters : ℕ := 6

/-- The total number of highlighters in the desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + orange_highlighters

theorem highlighters_count : total_highlighters = 25 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l1659_165967


namespace NUMINAMATH_CALUDE_gadget_production_proof_l1659_165938

/-- Represents the production rate of gadgets per worker per hour -/
def gadget_rate (workers : ℕ) (hours : ℕ) (gadgets : ℕ) : ℚ :=
  (gadgets : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Calculates the number of gadgets produced given workers, hours, and rate -/
def gadgets_produced (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours : ℚ) * rate

theorem gadget_production_proof :
  let rate1 := gadget_rate 150 3 600
  let rate2 := gadget_rate 100 4 800
  let final_rate := max rate1 rate2
  gadgets_produced 75 5 final_rate = 750 := by
  sorry

end NUMINAMATH_CALUDE_gadget_production_proof_l1659_165938


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l1659_165906

theorem triangle_is_right_angle (u : ℝ) 
  (h1 : 0 < 3*u - 2) 
  (h2 : 0 < 3*u + 2) 
  (h3 : 0 < 6*u) : 
  (3*u - 2) + (3*u + 2) = 6*u := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l1659_165906


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l1659_165971

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 4)
  (h2 : parrots_per_cage = 8)
  (h3 : total_birds = 40)
  : (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l1659_165971


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l1659_165972

theorem students_liking_both_desserts
  (total_students : ℕ)
  (like_apple_pie : ℕ)
  (like_chocolate_cake : ℕ)
  (like_neither : ℕ)
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 25)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 10) :
  (like_apple_pie + like_chocolate_cake) - (total_students - like_neither) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l1659_165972


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1659_165954

theorem fraction_sum_equals_decimal : (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1659_165954


namespace NUMINAMATH_CALUDE_projection_result_l1659_165912

def a : ℝ × ℝ := (-4, 2)
def b : ℝ × ℝ := (3, 4)

theorem projection_result (v : ℝ × ℝ) (p : ℝ × ℝ) 
  (h1 : v ≠ (0, 0)) 
  (h2 : p = (v.1 * (a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2), 
             v.2 * (a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2)))
  (h3 : p = (v.1 * (b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2), 
             v.2 * (b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2))) :
  p = (-44/53, 154/53) := by
sorry

end NUMINAMATH_CALUDE_projection_result_l1659_165912


namespace NUMINAMATH_CALUDE_indefinite_stick_shortening_process_l1659_165985

theorem indefinite_stick_shortening_process : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a > b ∧ a > c ∧
  a > b + c ∧
  ∃ (a' b' c' : ℝ), 
    a' = b + c ∧
    b' = b ∧
    c' = c ∧
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' > b' ∧ a' > c' ∧
    a' > b' + c' :=
by sorry


end NUMINAMATH_CALUDE_indefinite_stick_shortening_process_l1659_165985


namespace NUMINAMATH_CALUDE_negate_all_guitarists_proficient_l1659_165987

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Musician : U → Prop)
variable (Guitarist : U → Prop)
variable (ProficientViolinist : U → Prop)

-- Theorem statement
theorem negate_all_guitarists_proficient :
  (∃ x, Guitarist x ∧ ¬ProficientViolinist x) ↔ 
  ¬(∀ x, Guitarist x → ProficientViolinist x) :=
by sorry

end NUMINAMATH_CALUDE_negate_all_guitarists_proficient_l1659_165987


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1659_165947

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x + 4| ≥ m^2 - 5*m) ↔ -1 ≤ m ∧ m ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1659_165947


namespace NUMINAMATH_CALUDE_ticket_problem_l1659_165933

theorem ticket_problem (T : ℚ) : 
  (1/2 : ℚ) * T + (1/4 : ℚ) * ((1/2 : ℚ) * T) = 3600 → T = 5760 := by
  sorry

#check ticket_problem

end NUMINAMATH_CALUDE_ticket_problem_l1659_165933


namespace NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l1659_165997

theorem multiple_of_six_is_multiple_of_three (m : ℤ) :
  (∀ k : ℤ, ∃ n : ℤ, k * 6 = n * 3) →
  (∃ l : ℤ, m = l * 6) →
  (∃ j : ℤ, m = j * 3) :=
sorry

end NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l1659_165997


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1659_165966

theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, x^2 + p*x + q ≥ 1) → q = 1 + p^2/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1659_165966


namespace NUMINAMATH_CALUDE_train_speed_problem_l1659_165990

/-- Given a train that covers a distance in 360 minutes and the same distance in 90 minutes at 20 kmph,
    prove that the average speed of the train when it takes 360 minutes is 5 kmph. -/
theorem train_speed_problem (distance : ℝ) (speed_90min : ℝ) :
  distance = speed_90min * (90 / 60) →
  speed_90min = 20 →
  (distance / (360 / 60)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1659_165990


namespace NUMINAMATH_CALUDE_no_sum_of_squared_digits_greater_than_2008_l1659_165995

def sum_of_squared_digits (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  List.sum (List.map (λ d => d * d) digits)

theorem no_sum_of_squared_digits_greater_than_2008 :
  ∀ n : ℕ, n > 2008 → n ≠ sum_of_squared_digits n :=
sorry

end NUMINAMATH_CALUDE_no_sum_of_squared_digits_greater_than_2008_l1659_165995


namespace NUMINAMATH_CALUDE_intersection_M_N_l1659_165973

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1659_165973


namespace NUMINAMATH_CALUDE_average_annual_growth_rate_l1659_165913

/-- Given growth rates a and b for two consecutive years, 
    the average annual growth rate over these two years 
    is equal to √((a+1)(b+1)) - 1 -/
theorem average_annual_growth_rate 
  (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a + 1) * (b + 1)) - 1 ∧ 
  (1 + x)^2 = (1 + a) * (1 + b) :=
by sorry

end NUMINAMATH_CALUDE_average_annual_growth_rate_l1659_165913


namespace NUMINAMATH_CALUDE_unique_special_number_l1659_165908

/-- A three-digit number is represented by its hundreds, tens, and ones digits. -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- The value of a three-digit number. -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The product of the digits of a three-digit number. -/
def ThreeDigitNumber.digitProduct (n : ThreeDigitNumber) : Nat :=
  n.hundreds * n.tens * n.ones

/-- A three-digit number is special if it equals five times the product of its digits. -/
def ThreeDigitNumber.isSpecial (n : ThreeDigitNumber) : Prop :=
  n.value = 5 * n.digitProduct

/-- There exists a unique three-digit number that is special, and it is 175. -/
theorem unique_special_number :
  ∃! n : ThreeDigitNumber, n.isSpecial ∧ n.value = 175 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l1659_165908


namespace NUMINAMATH_CALUDE_interest_rate_is_five_percent_l1659_165935

-- Define the principal amount and interest rate
variable (P : ℝ) -- Principal amount
variable (r : ℝ) -- Interest rate (as a decimal)

-- Define the conditions
def condition1 : Prop := P * (1 + 3 * r) = 460
def condition2 : Prop := P * (1 + 8 * r) = 560

-- Theorem to prove
theorem interest_rate_is_five_percent 
  (h1 : condition1 P r) 
  (h2 : condition2 P r) : 
  r = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_is_five_percent_l1659_165935


namespace NUMINAMATH_CALUDE_building_height_l1659_165981

/-- Given a flagpole and a building casting shadows under similar conditions,
    prove that the height of the building is 22 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 55)
  : (flagpole_height / flagpole_shadow) * building_shadow = 22 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l1659_165981


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1659_165961

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 1, 1]

theorem inverse_of_B_cubed :
  let B_inv := !![3, -1; 1, 1]
  (B_inv^3)⁻¹ = !![20, -12; 12, -4] := by sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1659_165961


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l1659_165918

/-- The time when a ball hits the ground -/
theorem ball_hit_ground_time : ∃ (t : ℚ),
  t = 2313 / 1000 ∧
  0 = -4.9 * t^2 + 7 * t + 10 :=
by sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l1659_165918


namespace NUMINAMATH_CALUDE_tens_digit_of_nine_to_1010_l1659_165927

theorem tens_digit_of_nine_to_1010 :
  (9 : ℕ) ^ 1010 % 100 = 1 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_to_1010_l1659_165927


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1659_165976

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1659_165976


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l1659_165989

-- Define the number of balls
def num_balls : ℕ := 10

-- Define the probability of a ball being in its original position after two transpositions
def prob_original_position : ℚ := 18 / 25

-- Theorem statement
theorem expected_balls_in_original_position :
  (num_balls : ℚ) * prob_original_position = 72 / 10 := by
sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l1659_165989
