import Mathlib

namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l1299_129959

theorem mod_equivalence_problem : ∃! m : ℤ, 0 ≤ m ∧ m ≤ 8 ∧ m ≡ 500000 [ZMOD 9] ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l1299_129959


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1100_for_23_divisibility_least_addition_is_4_l1299_129950

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_to_1100_for_23_divisibility :
  ∃ (x : ℕ), x < 23 ∧ (1100 + x) % 23 = 0 ∧ ∀ (y : ℕ), y < x → (1100 + y) % 23 ≠ 0 :=
by
  apply least_addition_for_divisibility 1100 23
  norm_num

#eval (1100 + 4) % 23  -- This should evaluate to 0

theorem least_addition_is_4 :
  4 < 23 ∧ (1100 + 4) % 23 = 0 ∧ ∀ (y : ℕ), y < 4 → (1100 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1100_for_23_divisibility_least_addition_is_4_l1299_129950


namespace NUMINAMATH_CALUDE_residue_of_12_pow_2040_mod_19_l1299_129931

theorem residue_of_12_pow_2040_mod_19 :
  (12 : ℤ) ^ 2040 ≡ 7 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_12_pow_2040_mod_19_l1299_129931


namespace NUMINAMATH_CALUDE_range_of_a_when_not_p_range_of_m_when_p_necessary_not_sufficient_l1299_129955

-- Define the propositions
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + a + 3 = 0
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Theorem 1: When ¬p is true, a ∈ (-2, 6)
theorem range_of_a_when_not_p :
  ∀ a : ℝ, ¬(p a) → -2 < a ∧ a < 6 :=
sorry

-- Theorem 2: When p is necessary but not sufficient for q, m ∈ (-∞, -3] ∪ [7, +∞)
theorem range_of_m_when_p_necessary_not_sufficient :
  ∀ m : ℝ, (∀ a : ℝ, q m a → p a) ∧ (∃ a : ℝ, p a ∧ ¬(q m a)) →
  m ≤ -3 ∨ m ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_not_p_range_of_m_when_p_necessary_not_sufficient_l1299_129955


namespace NUMINAMATH_CALUDE_ten_tall_flags_made_l1299_129992

/-- Calculates the number of tall flags made given the total fabric area,
    dimensions of each flag type, number of square and wide flags made,
    and remaining fabric area. -/
def tall_flags_made (total_fabric : ℕ) (square_side : ℕ) (wide_length wide_width : ℕ)
  (tall_length tall_width : ℕ) (square_flags_made wide_flags_made : ℕ)
  (fabric_left : ℕ) : ℕ :=
  let square_area := square_side * square_side
  let wide_area := wide_length * wide_width
  let tall_area := tall_length * tall_width
  let used_area := square_area * square_flags_made + wide_area * wide_flags_made
  let tall_flags_area := total_fabric - used_area - fabric_left
  tall_flags_area / tall_area

/-- Theorem stating that given the problem conditions, 10 tall flags were made. -/
theorem ten_tall_flags_made :
  tall_flags_made 1000 4 5 3 3 5 16 20 294 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ten_tall_flags_made_l1299_129992


namespace NUMINAMATH_CALUDE_factoring_expression_l1299_129908

theorem factoring_expression (x y : ℝ) :
  5 * x * (x + 4) + 2 * (x + 4) * (y + 2) = (x + 4) * (5 * x + 2 * y + 4) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1299_129908


namespace NUMINAMATH_CALUDE_correct_calculation_l1299_129930

theorem correct_calculation (x : ℝ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1299_129930


namespace NUMINAMATH_CALUDE_max_intersection_points_quadrilateral_circle_l1299_129970

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a plane -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a line segment and a circle -/
def intersectionPointsLineSegmentCircle (segment : (ℝ × ℝ) × (ℝ × ℝ)) (circle : Circle) : ℕ :=
  sorry

/-- The number of intersection points between a quadrilateral and a circle -/
def intersectionPointsQuadrilateralCircle (quad : Quadrilateral) (circle : Circle) : ℕ :=
  sorry

/-- Theorem: The maximum number of intersection points between a quadrilateral and a circle is 8 -/
theorem max_intersection_points_quadrilateral_circle :
  ∀ (quad : Quadrilateral) (circle : Circle),
    intersectionPointsQuadrilateralCircle quad circle ≤ 8 ∧
    ∃ (quad' : Quadrilateral) (circle' : Circle),
      intersectionPointsQuadrilateralCircle quad' circle' = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_quadrilateral_circle_l1299_129970


namespace NUMINAMATH_CALUDE_moving_circle_theorem_l1299_129976

-- Define the circles F1 and F2
def F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the locus of the center of E
def E_locus (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the slope range
def slope_range (k : ℝ) : Prop := 
  (k ≥ -Real.sqrt 6 / 4 ∧ k < 0) ∨ (k > 0 ∧ k ≤ Real.sqrt 6 / 4)

-- State the theorem
theorem moving_circle_theorem 
  (E : ℝ → ℝ → Prop) -- The moving circle E
  (A B M H : ℝ × ℝ) -- Points A, B, M, H
  (l : ℝ → ℝ) -- Line l
  (h1 : ∀ x y, E x y → (∃ r > 0, ∀ u v, F1 u v → ((x - u)^2 + (y - v)^2 = (r + 1)^2))) -- E externally tangent to F1
  (h2 : ∀ x y, E x y → (∃ r > 0, ∀ u v, F2 u v → ((x - u)^2 + (y - v)^2 = (3 - r)^2))) -- E internally tangent to F2
  (h3 : A.1 > 0 ∧ A.2 = 0 ∧ E A.1 A.2) -- A on positive x-axis and on E
  (h4 : E B.1 B.2 ∧ B.2 ≠ 0) -- B on E and not on x-axis
  (h5 : ∀ x, l x = (B.2 / (B.1 - A.1)) * (x - A.1)) -- l passes through A and B
  (h6 : M.2 = l M.1 ∧ H.1 = 0) -- M on l, H on y-axis
  (h7 : (B.1 - 1) * (H.1 - 1) + B.2 * H.2 = 0) -- BF2 ⊥ HF2
  (h8 : (M.1 - A.1)^2 + (M.2 - A.2)^2 ≥ M.1^2 + M.2^2) -- ∠MOA ≥ ∠MAO
  : (∀ x y, E x y ↔ E_locus x y) ∧ 
    (∀ k, (∃ x, l x = k * (x - A.1)) → slope_range k) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_theorem_l1299_129976


namespace NUMINAMATH_CALUDE_kg_conversion_hour_conversion_l1299_129993

-- Define conversion factors
def grams_per_kg : ℝ := 1000
def minutes_per_hour : ℝ := 60

-- Theorem 1: Convert 70 kg 50 g to kg
theorem kg_conversion (mass_kg : ℝ) (mass_g : ℝ) :
  mass_kg + mass_g / grams_per_kg = 70.05 :=
by sorry

-- Theorem 2: Convert 3.7 hours to hours and minutes
theorem hour_conversion (hours : ℝ) :
  ∃ (whole_hours : ℕ) (minutes : ℕ),
    hours = whole_hours + minutes / minutes_per_hour ∧
    whole_hours = 3 ∧
    minutes = 42 :=
by sorry

end NUMINAMATH_CALUDE_kg_conversion_hour_conversion_l1299_129993


namespace NUMINAMATH_CALUDE_incorrect_operator_is_second_l1299_129906

def original_expression : List Int := [3, 5, -7, 9, -11, 13, -15, 17]

def calculate (expr : List Int) : Int :=
  expr.foldl (· + ·) 0

def flip_operator (expr : List Int) (index : Nat) : List Int :=
  expr.mapIdx (fun i x => if i == index then -x else x)

theorem incorrect_operator_is_second :
  ∃ (i : Nat), i < original_expression.length ∧
    calculate (flip_operator original_expression i) = -4 ∧
    i = 1 ∧
    ∀ (j : Nat), j < original_expression.length → j ≠ i →
      calculate (flip_operator original_expression j) ≠ -4 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operator_is_second_l1299_129906


namespace NUMINAMATH_CALUDE_circplus_two_three_one_l1299_129988

/-- Definition of the ⊕ operation -/
def circplus (a b c : ℝ) : ℝ := b^2 - 4*a*c + c^2

/-- Theorem: The value of ⊕(2, 3, 1) is 2 -/
theorem circplus_two_three_one : circplus 2 3 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_circplus_two_three_one_l1299_129988


namespace NUMINAMATH_CALUDE_retest_probability_l1299_129940

theorem retest_probability (total : ℕ) (p_physics : ℚ) (p_chemistry : ℚ) (p_biology : ℚ) :
  total = 50 →
  p_physics = 9 / 50 →
  p_chemistry = 1 / 5 →
  p_biology = 11 / 50 →
  let p_one_subject := p_physics + p_chemistry + p_biology
  let p_more_than_one := 1 - p_one_subject
  p_more_than_one = 2 / 5 := by
  sorry

#eval (2 : ℚ) / 5 -- This should output 0.4

end NUMINAMATH_CALUDE_retest_probability_l1299_129940


namespace NUMINAMATH_CALUDE_equation_solution_l1299_129903

theorem equation_solution (p : ℝ) (hp : p > 0) :
  ∃ x : ℝ, Real.sqrt (x^2 + 2*p*x - p^2) - Real.sqrt (x^2 - 2*p*x - p^2) = 1 ↔
  (|p| < 1/2 ∧ (x = Real.sqrt ((p^2 + 1/4) / (1 - 4*p^2)) ∨
               x = -Real.sqrt ((p^2 + 1/4) / (1 - 4*p^2)))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1299_129903


namespace NUMINAMATH_CALUDE_probability_of_graduate_degree_l1299_129990

theorem probability_of_graduate_degree (G C N : ℕ) : 
  G * 8 = N →
  C * 3 = N * 2 →
  (G : ℚ) / (G + C) = 3 / 19 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_graduate_degree_l1299_129990


namespace NUMINAMATH_CALUDE_inequality_preservation_l1299_129913

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1299_129913


namespace NUMINAMATH_CALUDE_interest_rate_is_nine_percent_l1299_129904

/-- Calculates the simple interest rate given two loans and the total interest received. -/
def calculate_interest_rate (principal1 : ℚ) (time1 : ℚ) (principal2 : ℚ) (time2 : ℚ) (total_interest : ℚ) : ℚ :=
  (100 * total_interest) / (principal1 * time1 + principal2 * time2)

/-- Theorem stating that the interest rate is 9% for the given loan conditions. -/
theorem interest_rate_is_nine_percent :
  let principal1 : ℚ := 5000
  let time1 : ℚ := 2
  let principal2 : ℚ := 3000
  let time2 : ℚ := 4
  let total_interest : ℚ := 1980
  calculate_interest_rate principal1 time1 principal2 time2 total_interest = 9 := by
  sorry

#eval calculate_interest_rate 5000 2 3000 4 1980

end NUMINAMATH_CALUDE_interest_rate_is_nine_percent_l1299_129904


namespace NUMINAMATH_CALUDE_fraction_equality_l1299_129978

theorem fraction_equality (x y : ℚ) (h : x / y = 7 / 2) : (x - 2 * y) / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1299_129978


namespace NUMINAMATH_CALUDE_simplify_expression_l1299_129948

theorem simplify_expression : 
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1299_129948


namespace NUMINAMATH_CALUDE_multiplication_solution_l1299_129986

def possible_digits : Set Nat := {2, 4, 5, 6, 7, 8, 9}

def valid_multiplication (A B C D E : Nat) : Prop :=
  A ∈ possible_digits ∧ B ∈ possible_digits ∧ C ∈ possible_digits ∧ 
  D ∈ possible_digits ∧ E ∈ possible_digits ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E ∧
  E = 7 ∧
  (3 * (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1) = 
   100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1)

theorem multiplication_solution : 
  ∃ (A B C D E : Nat), valid_multiplication A B C D E ∧ A = 4 ∧ B = 2 ∧ C = 8 ∧ D = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_solution_l1299_129986


namespace NUMINAMATH_CALUDE_odd_divisor_of_3n_plus_1_l1299_129957

theorem odd_divisor_of_3n_plus_1 (n : ℕ) :
  n ≥ 1 ∧ Odd n ∧ n ∣ (3^n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisor_of_3n_plus_1_l1299_129957


namespace NUMINAMATH_CALUDE_congruence_solution_l1299_129984

theorem congruence_solution : ∃ n : ℕ, n ≤ 4 ∧ n ≡ -2323 [ZMOD 5] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1299_129984


namespace NUMINAMATH_CALUDE_dividend_calculation_l1299_129914

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 13)
  (h_quotient : quotient = 17)
  (h_remainder : remainder = 1) :
  divisor * quotient + remainder = 222 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1299_129914


namespace NUMINAMATH_CALUDE_complex_product_equals_one_l1299_129918

theorem complex_product_equals_one (x : ℂ) (h : x = Complex.exp (Complex.I * Real.pi / 7)) :
  (x^2 + x^4) * (x^4 + x^8) * (x^6 + x^12) * (x^8 + x^16) * (x^10 + x^20) * (x^12 + x^24) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_one_l1299_129918


namespace NUMINAMATH_CALUDE_remaining_money_proof_l1299_129901

def salary : ℚ := 190000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_amount : ℚ := salary * (1 - (food_fraction + rent_fraction + clothes_fraction))

theorem remaining_money_proof :
  remaining_amount = 19000 := by sorry

end NUMINAMATH_CALUDE_remaining_money_proof_l1299_129901


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l1299_129951

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 5}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1^2 + 6*p.1 - 8}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define the line y = x - 1
def center_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the lines l
def line_l1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 12*p.1 - 5*p.2 = 0}

theorem circle_and_line_theorem :
  -- The center of circle_C lies on center_line
  (∃ c : ℝ × ℝ, c ∈ center_line ∧ ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = 5) ∧
  -- circle_C passes through the intersection of parabola and x_axis
  (∀ p : ℝ × ℝ, p ∈ parabola ∩ x_axis → p ∈ circle_C) ∧
  -- For any line through origin intersecting circle_C at M and N with ON = 2OM,
  -- the line is either line_l1 or line_l2
  (∀ l : Set (ℝ × ℝ), origin ∈ l →
    (∃ M N : ℝ × ℝ, M ∈ l ∩ circle_C ∧ N ∈ l ∩ circle_C ∧ 
      N.1 = 2*M.1 ∧ N.2 = 2*M.2) →
    l = line_l1 ∨ l = line_l2) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l1299_129951


namespace NUMINAMATH_CALUDE_remainder_of_sum_divided_by_eight_l1299_129961

theorem remainder_of_sum_divided_by_eight :
  (2356789 + 211) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_sum_divided_by_eight_l1299_129961


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1299_129902

theorem inequality_solution_set (x : ℝ) : (3 + x) * (2 - x) < 0 ↔ x > 2 ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1299_129902


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l1299_129967

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (b + c) + 1 / (c + a) + 1 / (a + b) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) = 1 / (b + c) + 1 / (c + a) + 1 / (a + b)) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l1299_129967


namespace NUMINAMATH_CALUDE_order_of_numbers_l1299_129945

theorem order_of_numbers (m n : ℝ) (hm : m < 0) (hn : n > 0) (hmn : m + n < 0) :
  -m > n ∧ n > -n ∧ -n > m := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1299_129945


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1299_129962

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides --/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1299_129962


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1299_129958

/-- Given that the line y = x - 1 is tangent to the curve y = e^(x+a), prove that a = -2 --/
theorem tangent_line_to_exponential_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ - 1 = Real.exp (x₀ + a) ∧ 1 = Real.exp (x₀ + a)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1299_129958


namespace NUMINAMATH_CALUDE_initial_pears_eq_sum_l1299_129985

/-- The number of pears Sara picked initially -/
def initial_pears : ℕ := sorry

/-- The number of apples Sara picked -/
def apples : ℕ := 27

/-- The number of pears Sara gave to Dan -/
def pears_given : ℕ := 28

/-- The number of pears Sara has left -/
def pears_left : ℕ := 7

/-- Theorem stating that the initial number of pears is equal to the sum of pears given and pears left -/
theorem initial_pears_eq_sum : initial_pears = pears_given + pears_left := by sorry

end NUMINAMATH_CALUDE_initial_pears_eq_sum_l1299_129985


namespace NUMINAMATH_CALUDE_package_volume_calculation_l1299_129929

/-- Calculates the total volume needed to package a collection given box dimensions and cost constraints. -/
theorem package_volume_calculation 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (box_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : box_length = 20) 
  (h2 : box_width = 20) 
  (h3 : box_height = 15) 
  (h4 : box_cost = 0.9) 
  (h5 : total_cost = 459) :
  (total_cost / box_cost) * (box_length * box_width * box_height) = 3060000 := by
  sorry

end NUMINAMATH_CALUDE_package_volume_calculation_l1299_129929


namespace NUMINAMATH_CALUDE_triangle_properties_l1299_129956

open Real

theorem triangle_properties (A B C : ℝ) (R : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  R = 1 →
  (sin A - sin B + sin C) / sin C = sin B / (sin A + sin B - sin C) →
  ∃ (S : ℝ),
    A = π / 3 ∧
    S ≤ 3 * sqrt 3 / 4 ∧
    (∀ (S' : ℝ), S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1299_129956


namespace NUMINAMATH_CALUDE_cone_height_ratio_l1299_129966

/-- Proves the ratio of heights for a cone with reduced height --/
theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) :
  base_circumference = 18 * Real.pi →
  original_height = 36 →
  shorter_volume = 270 * Real.pi →
  ∃ (shorter_height : ℝ),
    shorter_height / original_height = 5 / 18 ∧
    shorter_volume = (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height :=
by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l1299_129966


namespace NUMINAMATH_CALUDE_strawberry_calories_is_4_l1299_129916

/-- The number of strawberries Zoe ate -/
def num_strawberries : ℕ := 12

/-- The amount of yogurt Zoe ate in ounces -/
def yogurt_ounces : ℕ := 6

/-- The number of calories per ounce of yogurt -/
def yogurt_calories_per_ounce : ℕ := 17

/-- The total calories Zoe ate -/
def total_calories : ℕ := 150

/-- The number of calories in each strawberry -/
def strawberry_calories : ℕ := (total_calories - yogurt_ounces * yogurt_calories_per_ounce) / num_strawberries

theorem strawberry_calories_is_4 : strawberry_calories = 4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_calories_is_4_l1299_129916


namespace NUMINAMATH_CALUDE_student_distribution_l1299_129911

theorem student_distribution (total : ℕ) (schemes : ℕ) : 
  total = 7 → 
  schemes = 108 → 
  (∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys * Nat.choose girls 2 * 6 = schemes ∧
    boys = 3 ∧ 
    girls = 4) :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_l1299_129911


namespace NUMINAMATH_CALUDE_cosine_angle_C_l1299_129927

/-- Given a triangle ABC with side lengths and angle relation, prove the cosine of angle C -/
theorem cosine_angle_C (A B C : ℝ) (BC AC : ℝ) (h1 : BC = 5) (h2 : AC = 4) 
  (h3 : Real.cos (A - B) = 7/8) : Real.cos C = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_C_l1299_129927


namespace NUMINAMATH_CALUDE_binomSum_not_div_five_l1299_129975

def binomSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose (2 * n + 1) (2 * k + 1) * 2^(3 * k))

theorem binomSum_not_div_five (n : ℕ) : ¬(5 ∣ binomSum n) := by
  sorry

end NUMINAMATH_CALUDE_binomSum_not_div_five_l1299_129975


namespace NUMINAMATH_CALUDE_smallest_angle_in_4_5_7_ratio_triangle_l1299_129994

theorem smallest_angle_in_4_5_7_ratio_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  b = (5/4) * a →
  c = (7/4) * a →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_4_5_7_ratio_triangle_l1299_129994


namespace NUMINAMATH_CALUDE_polynomial_property_l1299_129942

def P (x d e : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + 9

theorem polynomial_property (d e : ℝ) :
  (∃ k : ℝ, 
    -- Mean of zeros
    (- d / (3 * 3)) = k ∧ 
    -- Twice the product of zeros
    2 * (- 9 / 3) = k ∧ 
    -- Sum of coefficients
    (3 + d + e + 9) = k) →
  e = -72 := by
sorry

end NUMINAMATH_CALUDE_polynomial_property_l1299_129942


namespace NUMINAMATH_CALUDE_max_visible_cubes_eq_400_l1299_129943

/-- The size of the cube's edge -/
def cube_size : ℕ := 12

/-- The number of unit cubes visible on a single face -/
def face_count : ℕ := cube_size ^ 2

/-- The number of unit cubes overcounted on each edge -/
def edge_overcount : ℕ := cube_size - 1

/-- The maximum number of visible unit cubes from a single point -/
def max_visible_cubes : ℕ := 3 * face_count - 3 * edge_overcount + 1

theorem max_visible_cubes_eq_400 : max_visible_cubes = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_eq_400_l1299_129943


namespace NUMINAMATH_CALUDE_xyz_product_abs_l1299_129989

theorem xyz_product_abs (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_eq : x + 2/y = y + 2/z ∧ y + 2/z = z + 2/x) :
  |x * y * z| = 2 := by sorry

end NUMINAMATH_CALUDE_xyz_product_abs_l1299_129989


namespace NUMINAMATH_CALUDE_two_digit_times_nine_equals_1068_l1299_129977

theorem two_digit_times_nine_equals_1068 : ∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ x * 9 = 1068 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_times_nine_equals_1068_l1299_129977


namespace NUMINAMATH_CALUDE_egg_price_decrease_impact_l1299_129974

-- Define the types
structure CakeShop where
  eggDemand : ℝ
  productionScale : ℝ
  cakeOutput : ℝ
  marketSupply : ℝ

-- Define the egg price change
def eggPriceDecrease : ℝ := 0.05

-- Define the impact of egg price decrease on cake shops
def impactOnCakeShop (shop : CakeShop) (priceDecrease : ℝ) : CakeShop :=
  { eggDemand := shop.eggDemand * (1 + priceDecrease),
    productionScale := shop.productionScale * (1 + priceDecrease),
    cakeOutput := shop.cakeOutput * (1 + priceDecrease),
    marketSupply := shop.marketSupply * (1 + priceDecrease) }

-- Theorem statement
theorem egg_price_decrease_impact (shop : CakeShop) :
  let newShop := impactOnCakeShop shop eggPriceDecrease
  newShop.eggDemand > shop.eggDemand ∧
  newShop.productionScale > shop.productionScale ∧
  newShop.cakeOutput > shop.cakeOutput ∧
  newShop.marketSupply > shop.marketSupply :=
by sorry

end NUMINAMATH_CALUDE_egg_price_decrease_impact_l1299_129974


namespace NUMINAMATH_CALUDE_calculate_expression_l1299_129999

theorem calculate_expression : (-2)^3 + Real.sqrt 12 + (1/3)⁻¹ = 2 * Real.sqrt 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1299_129999


namespace NUMINAMATH_CALUDE_a_in_range_and_negative_one_in_A_l1299_129946

def A : Set ℝ := {x | x^2 - 2 < 0}

theorem a_in_range_and_negative_one_in_A (a : ℝ) (h : a ∈ A) :
  -Real.sqrt 2 < a ∧ a < Real.sqrt 2 ∧ -1 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_in_range_and_negative_one_in_A_l1299_129946


namespace NUMINAMATH_CALUDE_fraction_calculation_l1299_129991

theorem fraction_calculation : (500^2 : ℝ) / (152^2 - 148^2) = 208.333 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1299_129991


namespace NUMINAMATH_CALUDE_donna_marcia_total_pencils_l1299_129928

/-- The number of pencils bought by Cindi -/
def cindi_pencils : ℕ := 60

/-- The number of pencils bought by Marcia -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils bought by Donna -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- The total number of pencils bought by Donna and Marcia -/
def total_pencils : ℕ := donna_pencils + marcia_pencils

theorem donna_marcia_total_pencils :
  total_pencils = 480 := by
  sorry

end NUMINAMATH_CALUDE_donna_marcia_total_pencils_l1299_129928


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1299_129926

theorem quadratic_perfect_square (x : ℝ) : ∃ (a : ℝ), x^2 - 20*x + 100 = (x + a)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1299_129926


namespace NUMINAMATH_CALUDE_equation_solutions_l1299_129968

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 1 = x + 7 ∧ x = 4) ∧
  (∃ x : ℚ, (x + 1) / 2 - 1 = (1 - 2 * x) / 3 ∧ x = 5 / 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1299_129968


namespace NUMINAMATH_CALUDE_hyperbola_distance_property_l1299_129939

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_distance_property (P : ℝ × ℝ) :
  is_on_hyperbola P.1 P.2 →
  distance P F1 = 12 →
  (distance P F2 = 2 ∨ distance P F2 = 22) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_distance_property_l1299_129939


namespace NUMINAMATH_CALUDE_polynomial_division_l1299_129964

theorem polynomial_division (x : ℝ) : 
  x^6 - 14*x^4 + 8*x^3 - 26*x^2 + 14*x - 3 = 
  (x - 3) * (x^5 + 3*x^4 - 5*x^3 - 7*x^2 - 47*x - 7) + (-24) := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_l1299_129964


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_810_l1299_129949

theorem sin_n_equals_cos_810 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * Real.pi / 180) = Real.cos (810 * Real.pi / 180) →
  n = -180 ∨ n = 0 ∨ n = 180 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_810_l1299_129949


namespace NUMINAMATH_CALUDE_algae_coverage_on_day_17_algae_doubles_daily_full_coverage_on_day_20_l1299_129947

/-- Represents the coverage of algae on the lake on a given day -/
def algae_coverage (day : ℕ) : ℝ :=
  2^(day - 17)

/-- The day when the lake is completely covered with algae -/
def full_coverage_day : ℕ := 20

theorem algae_coverage_on_day_17 :
  algae_coverage 17 = 0.125 ∧ 1 - algae_coverage 17 = 0.875 := by sorry

theorem algae_doubles_daily (d : ℕ) (h : d < full_coverage_day) :
  algae_coverage (d + 1) = 2 * algae_coverage d := by sorry

theorem full_coverage_on_day_20 :
  algae_coverage full_coverage_day = 1 := by sorry

end NUMINAMATH_CALUDE_algae_coverage_on_day_17_algae_doubles_daily_full_coverage_on_day_20_l1299_129947


namespace NUMINAMATH_CALUDE_count_valid_selections_32_card_deck_l1299_129925

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Calculates the number of ways to choose 6 cards from a deck
    such that all suits are represented -/
def count_valid_selections (d : Deck) : Nat :=
  let s1 := Nat.choose 4 2 * (Nat.choose 8 2)^2 * 8^2
  let s2 := Nat.choose 4 1 * Nat.choose 8 3 * 8^3
  s1 + s2

/-- The main theorem to be proved -/
theorem count_valid_selections_32_card_deck :
  ∃ (d : Deck), d.total_cards = 32 ∧ d.num_suits = 4 ∧ d.cards_per_suit = 8 ∧
  count_valid_selections d = 415744 :=
by
  sorry

end NUMINAMATH_CALUDE_count_valid_selections_32_card_deck_l1299_129925


namespace NUMINAMATH_CALUDE_ani_winning_strategy_l1299_129933

/-- Represents the state of the game with three buckets -/
structure GameState :=
  (bucket1 bucket2 bucket3 : ℕ)

/-- Defines a valid game state where each bucket has at least one marble -/
def ValidGameState (state : GameState) : Prop :=
  state.bucket1 > 0 ∧ state.bucket2 > 0 ∧ state.bucket3 > 0

/-- Defines the total number of marbles in the game -/
def TotalMarbles (state : GameState) : ℕ :=
  state.bucket1 + state.bucket2 + state.bucket3

/-- Defines a valid move in the game -/
def ValidMove (marbles : ℕ) : Prop :=
  marbles = 1 ∨ marbles = 2 ∨ marbles = 3

/-- Defines whether a game state is a winning position for the current player -/
def IsWinningPosition (state : GameState) : Prop :=
  sorry

/-- Theorem: Ani has a winning strategy if and only if n is even and n ≥ 6 -/
theorem ani_winning_strategy (n : ℕ) :
  (∃ (initialState : GameState),
    ValidGameState initialState ∧
    TotalMarbles initialState = n ∧
    IsWinningPosition initialState) ↔
  (Even n ∧ n ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_ani_winning_strategy_l1299_129933


namespace NUMINAMATH_CALUDE_minimum_team_size_l1299_129954

theorem minimum_team_size : ∃ n : ℕ, n > 0 ∧ n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m : ℕ, m > 0 → m % 8 = 0 → m % 9 = 0 → m % 10 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_minimum_team_size_l1299_129954


namespace NUMINAMATH_CALUDE_tom_payment_l1299_129963

/-- The total amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1235 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 75 = 1235 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l1299_129963


namespace NUMINAMATH_CALUDE_lead_in_mixture_l1299_129920

theorem lead_in_mixture (total : ℝ) (copper_weight : ℝ) (lead_percent : ℝ) (copper_percent : ℝ)
  (h1 : copper_weight = 12)
  (h2 : copper_percent = 0.60)
  (h3 : lead_percent = 0.25)
  (h4 : copper_weight = copper_percent * total) :
  lead_percent * total = 5 := by
sorry

end NUMINAMATH_CALUDE_lead_in_mixture_l1299_129920


namespace NUMINAMATH_CALUDE_johns_allowance_l1299_129921

theorem johns_allowance (allowance : ℝ) : 
  (allowance > 0) →
  (2 / 3 * (2 / 5 * allowance) = 1.28) →
  allowance = 4.80 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l1299_129921


namespace NUMINAMATH_CALUDE_statement_A_statement_B_statement_C_statement_D_l1299_129980

/-- Given polynomials M and N -/
def M (a x : ℝ) : ℝ := a * x^2 - 2*x + 3
def N (b x : ℝ) : ℝ := x^2 - b*x - 1

/-- Statement A -/
theorem statement_A : ∃ x : ℝ, M 1 x - N 2 x ≠ -4*x + 2 := by sorry

/-- Statement B -/
theorem statement_B : ∀ x : ℝ, M (-1) x + N 2 x = -4*x + 2 := by sorry

/-- Statement C -/
theorem statement_C : ∃ x : ℝ, x ≠ 1 ∧ |M 1 x - N 4 x| = 6 := by sorry

/-- Statement D -/
theorem statement_D : (∀ x : ℝ, ∃ c : ℝ, 2 * M a x + N b x = c) → a = -1/2 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_statement_A_statement_B_statement_C_statement_D_l1299_129980


namespace NUMINAMATH_CALUDE_friday_to_thursday_ratio_l1299_129998

/-- Represents the study time for each day of the week -/
structure StudyTime where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  weekend : ℝ

/-- The study time satisfies the given conditions -/
def valid_study_time (st : StudyTime) : Prop :=
  st.wednesday = 2 ∧
  st.thursday = 3 * st.wednesday ∧
  st.weekend = st.wednesday + st.thursday + st.friday ∧
  st.wednesday + st.thursday + st.friday + st.weekend = 22

/-- The theorem to be proved -/
theorem friday_to_thursday_ratio (st : StudyTime) 
  (h : valid_study_time st) : st.friday / st.thursday = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_friday_to_thursday_ratio_l1299_129998


namespace NUMINAMATH_CALUDE_factorization_equality_l1299_129969

theorem factorization_equality (m n : ℝ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1299_129969


namespace NUMINAMATH_CALUDE_negation_of_existence_statement_l1299_129972

theorem negation_of_existence_statement :
  ¬(∃ x : ℝ, x ≤ 0) ↔ (∀ x : ℝ, x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_statement_l1299_129972


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1299_129960

theorem smallest_max_sum (p q r s t : ℕ+) (h : p + q + r + s + t = 2025) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∀ m : ℕ, (∃ p' q' r' s' t' : ℕ+, p' + q' + r' + s' + t' = 2025 ∧ 
    max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) < m) → m > 676 := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1299_129960


namespace NUMINAMATH_CALUDE_lines_skew_iff_a_neq_4_l1299_129917

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ (t u : ℝ), l1.point + t • l1.direction = l2.point + u • l2.direction)

/-- The main theorem -/
theorem lines_skew_iff_a_neq_4 (a : ℝ) :
  let l1 : Line3D := ⟨(2, 3, a), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(5, 2, 1), (6, 3, 2)⟩
  are_skew l1 l2 ↔ a ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_lines_skew_iff_a_neq_4_l1299_129917


namespace NUMINAMATH_CALUDE_garden_area_l1299_129997

theorem garden_area (total_posts : ℕ) (post_distance : ℕ) (longer_side_posts : ℕ) (shorter_side_posts : ℕ) :
  total_posts = 24 →
  post_distance = 4 →
  longer_side_posts = 2 * shorter_side_posts →
  longer_side_posts + shorter_side_posts = total_posts + 4 →
  (shorter_side_posts - 1) * post_distance * (longer_side_posts - 1) * post_distance = 576 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l1299_129997


namespace NUMINAMATH_CALUDE_count_scalene_triangles_l1299_129973

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ 
  a + b + c < 15 ∧
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_scalene_triangles : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    S.card = 6 ∧ 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2) :=
sorry

end NUMINAMATH_CALUDE_count_scalene_triangles_l1299_129973


namespace NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l1299_129983

/-- The value of Adam's coin collection -/
def adams_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) : ℕ :=
  total_coins * (sample_value / sample_coins)

/-- Theorem: Adam's coin collection is worth 80 dollars -/
theorem adams_collection_worth_80_dollars :
  adams_collection_value 20 5 20 = 80 :=
by sorry

end NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l1299_129983


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1299_129932

theorem max_value_trig_expression (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x + 1 ≤ Real.sqrt 13 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1299_129932


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l1299_129910

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l1299_129910


namespace NUMINAMATH_CALUDE_license_plate_count_l1299_129965

/-- The number of digits in a license plate -/
def num_digits : ℕ := 5

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 21

/-- The number of positions where the consonant pair can be placed -/
def consonant_pair_positions : ℕ := 6

/-- The number of distinct license plates -/
def num_license_plates : ℕ := 
  consonant_pair_positions * digit_choices ^ num_digits * (num_consonants * (num_consonants - 1))

theorem license_plate_count : num_license_plates = 2520000000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1299_129965


namespace NUMINAMATH_CALUDE_no_lattice_equilateral_triangle_l1299_129923

-- Define a lattice point as a point with integer coordinates
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define an equilateral triangle
def Equilateral (a b c : ℝ × ℝ) : Prop :=
  let d := (fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d a b = d b c ∧ d b c = d c a

-- Theorem statement
theorem no_lattice_equilateral_triangle :
  ¬ ∃ (a b c : ℝ × ℝ), LatticePoint a ∧ LatticePoint b ∧ LatticePoint c ∧ Equilateral a b c :=
sorry

end NUMINAMATH_CALUDE_no_lattice_equilateral_triangle_l1299_129923


namespace NUMINAMATH_CALUDE_equal_digit_probability_l1299_129996

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := 9 / 20

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := 11 / 20

/-- The number of ways to choose half the dice -/
def num_combinations : ℕ := (num_dice.choose (num_dice / 2))

/-- The probability of getting an equal number of one-digit and two-digit numbers when rolling 6 20-sided dice -/
theorem equal_digit_probability : 
  (num_combinations : ℚ) * (prob_one_digit ^ (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2)) = 485264 / 1600000 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l1299_129996


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1299_129979

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (m, 6)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1299_129979


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1299_129941

/-- Q is a cubic polynomial with the given coefficients -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x + 8

/-- The theorem states that if x-3 is a factor of Q(x), then d = -62/3 -/
theorem factor_implies_d_value (d : ℝ) : 
  (∀ x, Q d x = 0 ↔ x = 3) → d = -62/3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1299_129941


namespace NUMINAMATH_CALUDE_sale_total_cost_l1299_129938

/-- Calculates the total cost of ice cream and juice during a sale. -/
def calculate_total_cost (original_ice_cream_price : ℚ) 
                         (ice_cream_discount : ℚ) 
                         (juice_price : ℚ) 
                         (juice_cans_per_price : ℕ) 
                         (ice_cream_tubs : ℕ) 
                         (juice_cans : ℕ) : ℚ :=
  let sale_ice_cream_price := original_ice_cream_price - ice_cream_discount
  let ice_cream_cost := sale_ice_cream_price * ice_cream_tubs
  let juice_cost := (juice_price / juice_cans_per_price) * juice_cans
  ice_cream_cost + juice_cost

/-- Theorem stating that the total cost is $24 for the given conditions. -/
theorem sale_total_cost : 
  calculate_total_cost 12 2 2 5 2 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sale_total_cost_l1299_129938


namespace NUMINAMATH_CALUDE_strip_coloring_problem_l1299_129971

/-- The number of valid colorings for a strip of length n -/
def validColorings : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validColorings (n + 1) + validColorings n

/-- The problem statement -/
theorem strip_coloring_problem :
  validColorings 9 = 89 := by sorry

end NUMINAMATH_CALUDE_strip_coloring_problem_l1299_129971


namespace NUMINAMATH_CALUDE_quinn_caught_four_frogs_l1299_129952

-- Define the number of frogs caught by each person
def alster_frogs : ℕ := 2
def bret_frogs : ℕ := 12

-- Define Quinn's frogs in terms of Alster's
def quinn_frogs : ℕ := alster_frogs

-- Define the relationship between Bret's and Quinn's frogs
axiom bret_quinn_relation : bret_frogs = 3 * quinn_frogs

theorem quinn_caught_four_frogs : quinn_frogs = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_caught_four_frogs_l1299_129952


namespace NUMINAMATH_CALUDE_complex_div_i_coords_l1299_129905

/-- The complex number (3+4i)/i corresponds to the point (4, -3) in the complex plane -/
theorem complex_div_i_coords : 
  let z : ℂ := (3 + 4*I) / I
  (z.re = 4 ∧ z.im = -3) :=
by sorry

end NUMINAMATH_CALUDE_complex_div_i_coords_l1299_129905


namespace NUMINAMATH_CALUDE_crayons_per_box_l1299_129924

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) 
  (h1 : total_crayons = 80) (h2 : num_boxes = 10) :
  total_crayons / num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_box_l1299_129924


namespace NUMINAMATH_CALUDE_correct_selection_ways_l1299_129936

/-- Represents the selection of athletes for a commendation meeting. -/
structure AthletesSelection where
  totalMales : Nat
  totalFemales : Nat
  maleCaptain : Nat
  femaleCaptain : Nat
  selectionSize : Nat

/-- Calculates the number of ways to select athletes under different conditions. -/
def selectionWays (s : AthletesSelection) : Nat × Nat × Nat × Nat :=
  let totalAthletes := s.totalMales + s.totalFemales
  let totalCaptains := s.maleCaptain + s.femaleCaptain
  let nonCaptains := totalAthletes - totalCaptains
  (
    Nat.choose s.totalMales 3 * Nat.choose s.totalFemales 2,
    Nat.choose totalCaptains 1 * Nat.choose nonCaptains 4 + Nat.choose totalCaptains 2 * Nat.choose nonCaptains 3,
    Nat.choose totalAthletes s.selectionSize - Nat.choose s.totalMales s.selectionSize,
    Nat.choose totalAthletes s.selectionSize - Nat.choose nonCaptains s.selectionSize - Nat.choose (s.totalMales - 1) (s.selectionSize - 1)
  )

/-- Theorem stating the correct number of ways to select athletes under different conditions. -/
theorem correct_selection_ways (s : AthletesSelection) 
  (h1 : s.totalMales = 6)
  (h2 : s.totalFemales = 4)
  (h3 : s.maleCaptain = 1)
  (h4 : s.femaleCaptain = 1)
  (h5 : s.selectionSize = 5) :
  selectionWays s = (120, 196, 246, 191) := by
  sorry

end NUMINAMATH_CALUDE_correct_selection_ways_l1299_129936


namespace NUMINAMATH_CALUDE_predict_sales_at_34_degrees_l1299_129944

/-- Represents the linear regression model for cold drink sales -/
structure ColdDrinkSalesModel where
  /-- Calculates the predicted sales volume based on temperature -/
  predict : ℝ → ℝ

/-- Theorem: Given the linear regression model ŷ = 2x + 60, 
    the predicted sales volume for a day with highest temperature 34°C is 128 cups -/
theorem predict_sales_at_34_degrees 
  (model : ColdDrinkSalesModel)
  (h_model : ∀ x, model.predict x = 2 * x + 60) :
  model.predict 34 = 128 := by
  sorry

end NUMINAMATH_CALUDE_predict_sales_at_34_degrees_l1299_129944


namespace NUMINAMATH_CALUDE_complex_norm_squared_l1299_129981

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 6*I) : 
  Complex.normSq z = 13/2 := by
sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l1299_129981


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l1299_129995

/-- Given that Alice takes 40 minutes to clean her room and Bob spends 3/8 of Alice's time,
    prove that Bob's cleaning time is 15 minutes. -/
theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) :
  alice_time = 40 →
  bob_fraction = 3 / 8 →
  (bob_fraction * alice_time : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l1299_129995


namespace NUMINAMATH_CALUDE_part_one_part_two_l1299_129900

noncomputable section

open Real

-- Define the function f
def f (a b x : ℝ) : ℝ := a * log x - b * x^2

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := -3 * x + 2 * log 2 + 2

-- Part 1: Prove that a = 2 and b = 1
theorem part_one : 
  ∀ a b : ℝ, (∀ x : ℝ, f a b x = f a b 2 + (x - 2) * (-3)) → a = 2 ∧ b = 1 := 
by sorry

-- Define the function h for part 2
def h (x m : ℝ) : ℝ := 2 * log x - x^2 + m

-- Part 2: Prove the range of m
theorem part_two : 
  ∀ m : ℝ, (∃ x y : ℝ, 1/exp 1 ≤ x ∧ x < y ∧ y ≤ exp 1 ∧ h x m = 0 ∧ h y m = 0) 
  → 1 < m ∧ m ≤ 1/(exp 1)^2 + 2 := 
by sorry

end

end NUMINAMATH_CALUDE_part_one_part_two_l1299_129900


namespace NUMINAMATH_CALUDE_a_range_l1299_129953

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) → 
  a > -1 ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_a_range_l1299_129953


namespace NUMINAMATH_CALUDE_pythagorean_orthogonal_sum_zero_l1299_129934

theorem pythagorean_orthogonal_sum_zero
  (a b c d : ℝ)
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : a*c + b*d = 0) :
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_orthogonal_sum_zero_l1299_129934


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1299_129937

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) :
  initial = 240 →
  percentage = 20 →
  result = initial * (1 + percentage / 100) →
  result = 288 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1299_129937


namespace NUMINAMATH_CALUDE_initial_amount_theorem_l1299_129907

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def num_posters : ℕ := 2
def num_notebooks : ℕ := 3
def num_bookmarks : ℕ := 2

def leftover : ℕ := 14

def total_cost : ℕ := num_posters * poster_cost + num_notebooks * notebook_cost + num_bookmarks * bookmark_cost

theorem initial_amount_theorem : total_cost + leftover = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_theorem_l1299_129907


namespace NUMINAMATH_CALUDE_mixture_composition_l1299_129909

theorem mixture_composition 
  (x_percent_a : Real) 
  (y_percent_a : Real) 
  (mixture_percent_a : Real) 
  (h1 : x_percent_a = 0.3) 
  (h2 : y_percent_a = 0.4) 
  (h3 : mixture_percent_a = 0.32) :
  ∃ (x_proportion : Real),
    x_proportion * x_percent_a + (1 - x_proportion) * y_percent_a = mixture_percent_a ∧ 
    x_proportion = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l1299_129909


namespace NUMINAMATH_CALUDE_total_eggs_l1299_129987

def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_town_hall : ℕ := 3

theorem total_eggs : eggs_club_house + eggs_park + eggs_town_hall = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_l1299_129987


namespace NUMINAMATH_CALUDE_zoo_trip_admission_cost_l1299_129982

theorem zoo_trip_admission_cost 
  (total_budget : ℕ) 
  (bus_rental_cost : ℕ) 
  (num_students : ℕ) 
  (h1 : total_budget = 350) 
  (h2 : bus_rental_cost = 100) 
  (h3 : num_students = 25) :
  (total_budget - bus_rental_cost) / num_students = 10 :=
by sorry

end NUMINAMATH_CALUDE_zoo_trip_admission_cost_l1299_129982


namespace NUMINAMATH_CALUDE_number_and_percentage_problem_l1299_129935

theorem number_and_percentage_problem (N P : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 25 ∧ 
  (P/100 : ℝ) * N = 300 →
  N = 750 ∧ P = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_and_percentage_problem_l1299_129935


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1299_129919

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 19 ↔ 
  (x = 20 ∧ y = 380) ∨ (x = 380 ∧ y = 20) ∨ 
  (x = 18 ∧ y = -342) ∨ (x = -342 ∧ y = 18) ∨ 
  (x = 38 ∧ y = 38) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1299_129919


namespace NUMINAMATH_CALUDE_average_candy_count_l1299_129922

def candy_counts : List Nat := [5, 7, 9, 12, 12, 15, 15, 18, 25]

theorem average_candy_count (num_bags : Nat) (counts : List Nat) 
  (h1 : num_bags = 9)
  (h2 : counts = candy_counts)
  (h3 : counts.length = num_bags) :
  Int.floor ((counts.sum : ℝ) / num_bags + 0.5) = 13 :=
by sorry

end NUMINAMATH_CALUDE_average_candy_count_l1299_129922


namespace NUMINAMATH_CALUDE_tournament_battles_one_team_remains_l1299_129912

/-- The number of battles needed to determine a champion in a tournament --/
def battles_to_champion (initial_teams : ℕ) : ℕ :=
  if initial_teams ≤ 1 then 0
  else if initial_teams = 2 then 1
  else (initial_teams - 1) / 2

/-- Theorem: In a tournament with 2017 teams, 1008 battles are needed to determine a champion --/
theorem tournament_battles :
  battles_to_champion 2017 = 1008 := by
  sorry

/-- Lemma: The number of teams remaining after n battles --/
lemma teams_remaining (initial_teams n : ℕ) : ℕ :=
  if n ≥ (initial_teams - 1) / 2 then 1
  else initial_teams - 2 * n

/-- Theorem: After 1008 battles, only one team remains in a tournament of 2017 teams --/
theorem one_team_remains :
  teams_remaining 2017 1008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tournament_battles_one_team_remains_l1299_129912


namespace NUMINAMATH_CALUDE_alloy_chromium_percentage_l1299_129915

/-- The percentage of chromium in an alloy mixture -/
def chromium_percentage (m1 m2 p1 p2 p3 : ℝ) : Prop :=
  m1 * p1 / 100 + m2 * p2 / 100 = (m1 + m2) * p3 / 100

/-- The problem statement -/
theorem alloy_chromium_percentage :
  ∃ (x : ℝ),
    chromium_percentage 15 30 12 x 9.333333333333334 ∧
    x = 8 := by sorry

end NUMINAMATH_CALUDE_alloy_chromium_percentage_l1299_129915
