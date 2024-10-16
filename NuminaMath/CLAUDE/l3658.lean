import Mathlib

namespace NUMINAMATH_CALUDE_average_weight_BCDE_l3658_365834

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight of B, C, D, and E is 51 kg. -/
theorem average_weight_BCDE (w_A w_B w_C w_D w_E : ℝ) : 
  (w_A + w_B + w_C) / 3 = 50 →
  (w_A + w_B + w_C + w_D) / 4 = 53 →
  w_E = w_D + 3 →
  w_A = 73 →
  (w_B + w_C + w_D + w_E) / 4 = 51 := by
sorry

end NUMINAMATH_CALUDE_average_weight_BCDE_l3658_365834


namespace NUMINAMATH_CALUDE_watch_gain_percentage_l3658_365877

/-- Calculates the gain percentage when a watch is sold at a different price -/
def gainPercentage (costPrice sellPrice : ℚ) : ℚ :=
  (sellPrice - costPrice) / costPrice * 100

/-- Theorem: The gain percentage is 5% under the given conditions -/
theorem watch_gain_percentage :
  let costPrice : ℚ := 933.33
  let initialLossPercentage : ℚ := 10
  let initialSellPrice : ℚ := costPrice * (1 - initialLossPercentage / 100)
  let newSellPrice : ℚ := initialSellPrice + 140
  gainPercentage costPrice newSellPrice = 5 := by
  sorry

end NUMINAMATH_CALUDE_watch_gain_percentage_l3658_365877


namespace NUMINAMATH_CALUDE_divisors_of_sum_of_primes_l3658_365838

-- Define a prime number p ≥ 5
def p : ℕ := sorry

-- Define q as the smallest prime number greater than p
def q : ℕ := sorry

-- Define n as the number of positive divisors of p + q
def n : ℕ := sorry

-- Axioms based on the problem conditions
axiom p_prime : Nat.Prime p
axiom p_ge_5 : p ≥ 5
axiom q_prime : Nat.Prime q
axiom q_gt_p : q > p
axiom q_smallest : ∀ r, Nat.Prime r → r > p → r ≥ q

-- Theorem to prove
theorem divisors_of_sum_of_primes :
  n ≥ 4 ∧ (∀ m, m ≥ 6 → n ≤ m) := by sorry

end NUMINAMATH_CALUDE_divisors_of_sum_of_primes_l3658_365838


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3658_365882

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3658_365882


namespace NUMINAMATH_CALUDE_short_sleeves_not_pants_count_l3658_365829

def students : Finset ℕ := Finset.range 30

def short_sleeves : Finset ℕ := {1, 3, 7, 10, 23, 27}

def short_pants : Finset ℕ := {1, 9, 11, 20, 23}

theorem short_sleeves_not_pants_count :
  (short_sleeves \ short_pants).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_short_sleeves_not_pants_count_l3658_365829


namespace NUMINAMATH_CALUDE_sum_of_ages_is_105_l3658_365856

/-- Calculates the sum of Riza's and her son's ages given their initial conditions -/
def sumOfAges (rizaAgeAtBirth : ℕ) (sonCurrentAge : ℕ) : ℕ :=
  rizaAgeAtBirth + 2 * sonCurrentAge

/-- Proves that the sum of Riza's and her son's ages is 105 years -/
theorem sum_of_ages_is_105 :
  sumOfAges 25 40 = 105 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_105_l3658_365856


namespace NUMINAMATH_CALUDE_license_plate_count_is_9360_l3658_365801

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possibilities for the second character (letter or digit) -/
def num_second_char : ℕ := num_letters + num_digits

/-- The number of ways to design a 4-character license plate with the given conditions -/
def license_plate_count : ℕ := num_letters * num_second_char * 1 * num_digits

theorem license_plate_count_is_9360 : license_plate_count = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_is_9360_l3658_365801


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3658_365831

theorem imaginary_part_of_complex_number (z : ℂ) : z = (1 - Complex.I) * Complex.I → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3658_365831


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3658_365800

/-- The perimeter of a rhombus inscribed in a rectangle --/
theorem rhombus_perimeter (w l : ℝ) (hw : w = 20) (hl : l = 25) :
  let s := Real.sqrt (w^2 / 4 + l^2 / 4)
  let perimeter := 4 * s
  ∃ ε > 0, abs (perimeter - 64.04) < ε := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3658_365800


namespace NUMINAMATH_CALUDE_annes_total_distance_l3658_365861

/-- Anne's hiking journey -/
def annes_hike (flat_speed flat_time uphill_speed uphill_time downhill_speed downhill_time : ℝ) : ℝ :=
  flat_speed * flat_time + uphill_speed * uphill_time + downhill_speed * downhill_time

/-- Theorem: Anne's total distance traveled is 14 miles -/
theorem annes_total_distance :
  annes_hike 3 2 2 2 4 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_annes_total_distance_l3658_365861


namespace NUMINAMATH_CALUDE_floretta_balloon_count_l3658_365859

/-- The number of water balloons Floretta is left with after Milly takes extra -/
def florettas_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (extra_taken : ℕ) : ℕ :=
  (total_packs * balloons_per_pack) / 2 - extra_taken

/-- Theorem stating the number of balloons Floretta is left with -/
theorem floretta_balloon_count :
  florettas_balloons 5 6 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_floretta_balloon_count_l3658_365859


namespace NUMINAMATH_CALUDE_limit_of_a_is_three_fourths_l3658_365879

def a (n : ℕ) : ℚ := (3 * n^2 + 2) / (4 * n^2 - 1)

theorem limit_of_a_is_three_fourths :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/4| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_is_three_fourths_l3658_365879


namespace NUMINAMATH_CALUDE_number_equation_solution_l3658_365842

theorem number_equation_solution : ∃ x : ℝ, 35 + 3 * x = 50 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3658_365842


namespace NUMINAMATH_CALUDE_cantaloupes_sum_l3658_365857

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_sum : total_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_sum_l3658_365857


namespace NUMINAMATH_CALUDE_expression_simplification_l3658_365865

theorem expression_simplification (x : ℝ) : (36 + 12*x)^2 - (12^2*x^2 + 36^2) = 864*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3658_365865


namespace NUMINAMATH_CALUDE_place_value_ratio_l3658_365833

-- Define the number
def number : ℝ := 58624.0791

-- Define the place value of 6 (thousands)
def place_value_6 : ℝ := 1000

-- Define the place value of 7 (tenths)
def place_value_7 : ℝ := 0.1

-- Theorem statement
theorem place_value_ratio :
  place_value_6 / place_value_7 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3658_365833


namespace NUMINAMATH_CALUDE_three_lines_intersect_once_l3658_365839

/-- A parabola defined by y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is outside a parabola -/
def is_outside (pt : Point) (par : Parabola) : Prop :=
  pt.y^2 > 2 * par.p * pt.x

/-- Predicate to check if a line intersects a parabola at exactly one point -/
def intersects_once (l : Line) (par : Parabola) : Prop :=
  sorry -- Definition of intersection at exactly one point

/-- The main theorem -/
theorem three_lines_intersect_once (par : Parabola) (M : Point) 
  (h_outside : is_outside M par) : 
  ∃ (l₁ l₂ l₃ : Line), 
    (∀ l : Line, (intersects_once l par ∧ l.a * M.x + l.b * M.y + l.c = 0) ↔ 
      (l = l₁ ∨ l = l₂ ∨ l = l₃)) :=
sorry

end NUMINAMATH_CALUDE_three_lines_intersect_once_l3658_365839


namespace NUMINAMATH_CALUDE_chess_draw_probability_l3658_365845

theorem chess_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.6)
  (h2 : prob_A_not_lose = 0.8) :
  prob_A_not_lose - prob_A_win = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l3658_365845


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l3658_365886

def num_books : ℕ := 6
def num_people : ℕ := 3

/-- The number of ways to divide 6 books into three parts of 2 books each -/
def divide_equal_parts : ℕ := 15

/-- The number of ways to distribute 6 books to three people, each receiving 2 books -/
def distribute_equal : ℕ := 90

/-- The number of ways to distribute 6 books to three people without restrictions -/
def distribute_unrestricted : ℕ := 729

/-- The number of ways to distribute 6 books to three people, with each person receiving at least 1 book -/
def distribute_at_least_one : ℕ := 481

theorem book_distribution_theorem :
  divide_equal_parts = 15 ∧
  distribute_equal = 90 ∧
  distribute_unrestricted = 729 ∧
  distribute_at_least_one = 481 :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l3658_365886


namespace NUMINAMATH_CALUDE_max_area_PAB_l3658_365849

-- Define the fixed points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

-- Define the lines passing through A and B
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

-- Define the intersection point P
def P (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of triangle PAB
def area_PAB (m : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_PAB :
  ∃ (max_area : ℝ), 
    (∀ m : ℝ, area_PAB m ≤ max_area) ∧ 
    (∃ m : ℝ, area_PAB m = max_area) ∧
    max_area = 5/2 :=
sorry

end NUMINAMATH_CALUDE_max_area_PAB_l3658_365849


namespace NUMINAMATH_CALUDE_quadratic_polynomial_sufficiency_necessity_l3658_365872

/-- A second-degree polynomial with distinct roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  x₁ : ℝ
  x₂ : ℝ
  distinct_roots : x₁ ≠ x₂
  is_root_x₁ : a * x₁^2 + b * x₁ + c = 0
  is_root_x₂ : a * x₂^2 + b * x₂ + c = 0

/-- The value of the polynomial at a given x -/
def QuadraticPolynomial.value (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem quadratic_polynomial_sufficiency_necessity 
    (p : QuadraticPolynomial) : 
    (p.a^2 + 3*p.a*p.c - p.b^2 = 0 → p.value (p.x₁^3) = p.value (p.x₂^3)) ∧
    (∃ p : QuadraticPolynomial, p.value (p.x₁^3) = p.value (p.x₂^3) ∧ p.a^2 + 3*p.a*p.c - p.b^2 ≠ 0) :=
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_sufficiency_necessity_l3658_365872


namespace NUMINAMATH_CALUDE_cubic_inequality_with_equality_l3658_365880

theorem cubic_inequality_with_equality (a b : ℝ) :
  a < b → a^3 - 3*a ≤ b^3 - 3*b + 4 ∧
  (a = -1 ∧ b = 1 → a^3 - 3*a = b^3 - 3*b + 4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_with_equality_l3658_365880


namespace NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l3658_365852

-- Define the equation of the circle
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 + (Real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∃ x y, circle_equation x y t) → t > -3 * Real.sqrt 3 / 2 :=
sorry

-- Theorem for the value of t when diameter is 6
theorem t_value_for_diameter_6 (t : ℝ) :
  (∃ x y, circle_equation x y t) →
  (∃ x₁ y₁ x₂ y₂, circle_equation x₁ y₁ t ∧ circle_equation x₂ y₂ t ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6) →
  t = 9 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l3658_365852


namespace NUMINAMATH_CALUDE_ratio_x_y_is_four_to_one_l3658_365808

theorem ratio_x_y_is_four_to_one 
  (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : 2 * Real.log (x - 2*y) = Real.log x + Real.log y) : 
  x / y = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_y_is_four_to_one_l3658_365808


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3658_365890

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define g(x) in terms of f(x)
def g (x : ℝ) : ℝ := f x + (-2)*x

-- Theorem statement
theorem quadratic_function_properties :
  -- Vertex of f(x) is at (1, 16)
  (f 1 = 16) ∧
  -- Roots of f(x) are 8 units apart
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₂ - r₁ = 8) →
  -- Conclusion 1: f(x) = -x^2 + 2x + 15
  (∀ x : ℝ, f x = -x^2 + 2*x + 15) ∧
  -- Conclusion 2: Maximum value of g(x) on [0, 2] is 7
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ 2 → g x ≤ 7) ∧ (∃ x : ℝ, x ≥ 0 ∧ x ≤ 2 ∧ g x = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3658_365890


namespace NUMINAMATH_CALUDE_binomial_510_510_l3658_365888

theorem binomial_510_510 : Nat.choose 510 510 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_510_510_l3658_365888


namespace NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l3658_365889

/-- The largest base-5 number with five digits -/
def largest_base5_5digit : ℕ := 44444

/-- Convert a base-5 number to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10000) * 5^4 + ((n / 1000) % 5) * 5^3 + ((n / 100) % 5) * 5^2 + ((n / 10) % 5) * 5^1 + (n % 5) * 5^0

theorem largest_base5_5digit_in_base10 :
  base5_to_base10 largest_base5_5digit = 3124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l3658_365889


namespace NUMINAMATH_CALUDE_average_multiples_of_6_up_to_100_l3658_365812

def multiples_of_6 (n : ℕ) : Finset ℕ :=
  Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))

theorem average_multiples_of_6_up_to_100 :
  let S := multiples_of_6 100
  (S.sum id) / S.card = 51 := by
  sorry

end NUMINAMATH_CALUDE_average_multiples_of_6_up_to_100_l3658_365812


namespace NUMINAMATH_CALUDE_chord_length_specific_case_l3658_365887

/-- The length of the chord cut by a circle on a line -/
def chord_length (a b c d e f : ℝ) : ℝ :=
  let circle := fun (x y : ℝ) => x^2 + y^2 + a*x + b*y + c
  let line := fun (x y : ℝ) => d*x + e*y + f
  -- The actual calculation of the chord length would go here
  0  -- Placeholder

theorem chord_length_specific_case :
  chord_length 0 (-2) (-1) 2 (-1) (-1) = 2 * Real.sqrt 30 / 5 := by
  sorry

#check chord_length_specific_case

end NUMINAMATH_CALUDE_chord_length_specific_case_l3658_365887


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l3658_365846

theorem lcm_gcd_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l3658_365846


namespace NUMINAMATH_CALUDE_triangle_exists_l3658_365805

/-- A triangle with vertices in ℝ² --/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- The area of a triangle --/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The altitudes of a triangle --/
def Triangle.altitudes (t : Triangle) : List ℝ := sorry

/-- Theorem: There exists a triangle with all altitudes less than 1 and area greater than or equal to 10 --/
theorem triangle_exists : ∃ t : Triangle, (∀ h ∈ t.altitudes, h < 1) ∧ t.area ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exists_l3658_365805


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3658_365837

/-- A quadratic function f(x) = ax^2 + bx + c with a ≠ 0 and satisfying 5a + b + 2c = 0 has two distinct real roots. -/
theorem quadratic_two_roots (a b c : ℝ) (ha : a ≠ 0) (h_cond : 5 * a + b + 2 * c = 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3658_365837


namespace NUMINAMATH_CALUDE_inverse_variation_result_l3658_365847

/-- Represents the inverse relationship between y^2 and √⁴z -/
def inverse_relationship (y z : ℝ) : Prop :=
  ∃ k : ℝ, y^2 * z^(1/4) = k

theorem inverse_variation_result :
  ∀ y z : ℝ,
  inverse_relationship y z →
  inverse_relationship 3 16 →
  y = 6 →
  z = 1/16 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_result_l3658_365847


namespace NUMINAMATH_CALUDE_two_fifths_divided_by_three_l3658_365898

theorem two_fifths_divided_by_three : (2 : ℚ) / 5 / 3 = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_divided_by_three_l3658_365898


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3658_365802

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = I - 3 → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3658_365802


namespace NUMINAMATH_CALUDE_exists_synchronous_exp_sin_synchronous_log_square_implies_a_gt_2e_l3658_365807

/-- Definition of synchronous functions -/
def Synchronous (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  f m = g m ∧ f n = g n

/-- Statement for option B -/
theorem exists_synchronous_exp_sin :
  ∃ n : ℝ, 1/2 < n ∧ n < 1 ∧
  Synchronous (fun x ↦ Real.exp x - 1) (fun x ↦ Real.sin (π * x)) 0 n :=
sorry

/-- Statement for option C -/
theorem synchronous_log_square_implies_a_gt_2e (a : ℝ) :
  (∃ m n : ℝ, Synchronous (fun x ↦ a * Real.log x) (fun x ↦ x^2) m n) →
  a > 2 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_exists_synchronous_exp_sin_synchronous_log_square_implies_a_gt_2e_l3658_365807


namespace NUMINAMATH_CALUDE_triangle_max_area_l3658_365835

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) : 
  A = (2 * π) / 3 →
  b + 2 * c = 8 →
  0 < a ∧ 0 < b ∧ 0 < c →
  (∀ b' c' : ℝ, b' + 2 * c' = 8 → 
    b' * c' * Real.sin A ≤ b * c * Real.sin A) →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  a = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3658_365835


namespace NUMINAMATH_CALUDE_k_value_in_set_union_l3658_365814

theorem k_value_in_set_union (A B : Set ℕ) (k : ℕ) :
  A = {1, 2, k} →
  B = {1, 2, 3, 5} →
  A ∪ B = {1, 2, 3, 5} →
  k = 3 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_k_value_in_set_union_l3658_365814


namespace NUMINAMATH_CALUDE_no_integer_solution_for_2006_l3658_365892

theorem no_integer_solution_for_2006 : ¬∃ (x y : ℤ), x^2 - y^2 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_2006_l3658_365892


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3658_365806

def age_problem (A_current B_current : ℕ) : Prop :=
  B_current = 37 ∧
  A_current = B_current + 7 ∧
  (A_current + 10) / (B_current - 10) = 2

theorem age_ratio_proof :
  ∃ A_current B_current : ℕ, age_problem A_current B_current :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3658_365806


namespace NUMINAMATH_CALUDE_part_one_part_two_l3658_365840

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 < 0}

-- Define the specific set A as given in the problem
def A_specific : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Theorem for part (1)
theorem part_one (a b : ℝ) (h : A a b = A_specific) : a + b = -7 := by
  sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) (h : A (-3) (-4) = A_specific) :
  (∀ x, x ∈ A (-3) (-4) → x ∉ B m) → m ≤ -3 ∨ m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3658_365840


namespace NUMINAMATH_CALUDE_marias_flower_bed_area_l3658_365809

/-- Represents a rectangular flower bed with fence posts --/
structure FlowerBed where
  total_posts : ℕ
  post_spacing : ℕ
  longer_side_posts : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the flower bed --/
def flower_bed_area (fb : FlowerBed) : ℕ :=
  (fb.shorter_side_posts - 1) * fb.post_spacing * ((fb.longer_side_posts - 1) * fb.post_spacing)

/-- Theorem stating that Maria's flower bed has an area of 350 square yards --/
theorem marias_flower_bed_area :
  ∃ fb : FlowerBed,
    fb.total_posts = 24 ∧
    fb.post_spacing = 5 ∧
    fb.longer_side_posts = 3 * fb.shorter_side_posts - 1 ∧
    fb.total_posts = fb.longer_side_posts + fb.shorter_side_posts + 2 ∧
    flower_bed_area fb = 350 :=
by sorry

end NUMINAMATH_CALUDE_marias_flower_bed_area_l3658_365809


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3658_365862

/-- The constant term in the binomial expansion of (x - 2/x)^8 -/
def constant_term : ℤ := 1120

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The general term of the binomial expansion (x - 2/x)^8 -/
def general_term (r : ℕ) : ℤ := (-2)^r * binomial 8 r

theorem constant_term_binomial_expansion :
  constant_term = general_term 4 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3658_365862


namespace NUMINAMATH_CALUDE_kim_class_hours_l3658_365854

def class_hours (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

theorem kim_class_hours : 
  class_hours 4 2 1 = 6 := by sorry

end NUMINAMATH_CALUDE_kim_class_hours_l3658_365854


namespace NUMINAMATH_CALUDE_problem_statement_l3658_365860

theorem problem_statement (x y : ℝ) (h : |x - 3| + (y + 4)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3658_365860


namespace NUMINAMATH_CALUDE_ratio_b_to_c_l3658_365819

theorem ratio_b_to_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_b_to_c_l3658_365819


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3658_365896

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) : 
  a + b = 5 → a^3 + b^3 = 125 → a * b = 0 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3658_365896


namespace NUMINAMATH_CALUDE_angle_D_value_l3658_365868

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the given conditions
axiom sum_A_B : A + B = 180
axiom C_eq_D : C = D
axiom A_value : A = 50

-- State the theorem to be proven
theorem angle_D_value : D = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l3658_365868


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l3658_365813

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop := a ≠ 0

/-- The equation 3x² + 1 = 0 -/
def equation_D : ℝ → Prop := fun x ↦ 3 * x^2 + 1 = 0

theorem equation_D_is_quadratic :
  is_quadratic_equation 3 0 1 := by sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l3658_365813


namespace NUMINAMATH_CALUDE_linear_function_shift_l3658_365873

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shifts a linear function horizontally -/
def shiftHorizontal (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + f.slope * units }

/-- Shifts a linear function vertically -/
def shiftVertical (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - units }

/-- The theorem to be proved -/
theorem linear_function_shift :
  let f := LinearFunction.mk 3 2
  let f_shifted_left := shiftHorizontal f 3
  let f_final := shiftVertical f_shifted_left 1
  f_final = LinearFunction.mk 3 10 := by sorry

end NUMINAMATH_CALUDE_linear_function_shift_l3658_365873


namespace NUMINAMATH_CALUDE_inequality_solution_l3658_365822

theorem inequality_solution (x : ℝ) : x - 1 / x > 0 ↔ (-1 < x ∧ x < 0) ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3658_365822


namespace NUMINAMATH_CALUDE_farm_animals_after_transaction_l3658_365844

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the ratio of horses to cows -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def initial_ratio : Ratio := { numerator := 3, denominator := 1 }
def final_ratio : Ratio := { numerator := 5, denominator := 3 }

def transaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

theorem farm_animals_after_transaction (farm : FarmAnimals) :
  farm.horses / farm.cows = initial_ratio.numerator / initial_ratio.denominator →
  (transaction farm).horses / (transaction farm).cows = final_ratio.numerator / final_ratio.denominator →
  (transaction farm).horses - (transaction farm).cows = 30 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_after_transaction_l3658_365844


namespace NUMINAMATH_CALUDE_gym_visitors_l3658_365883

theorem gym_visitors (initial_count : ℕ) (left_count : ℕ) (final_count : ℕ) :
  final_count ≥ initial_count - left_count →
  (final_count - (initial_count - left_count)) = 
  (final_count + left_count - initial_count) :=
by sorry

end NUMINAMATH_CALUDE_gym_visitors_l3658_365883


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3658_365875

theorem unique_positive_integer_solution :
  ∀ x y : ℕ+, 2 * x^2 + 5 * y^2 = 11 * (x * y - 11) → x = 14 ∧ y = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3658_365875


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3658_365825

theorem quadratic_roots_sum (a b : ℝ) : 
  (∃ (p q r : ℝ), p * (Complex.I ^ 2) + q * Complex.I + r = 0 ∧ 
   (3 + a * Complex.I) * ((3 + a * Complex.I) - (b - 2 * Complex.I)) = 0) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3658_365825


namespace NUMINAMATH_CALUDE_lampshade_container_volume_l3658_365893

/-- The volume of the smallest cylindrical container that can fit a conical lampshade -/
theorem lampshade_container_volume
  (h : ℝ) -- height of the lampshade
  (d : ℝ) -- diameter of the lampshade base
  (h_pos : h > 0)
  (d_pos : d > 0)
  (h_val : h = 15)
  (d_val : d = 8) :
  let r := d / 2 -- radius of the container
  let v := π * r^2 * h -- volume of the container
  v = 240 * π :=
sorry

end NUMINAMATH_CALUDE_lampshade_container_volume_l3658_365893


namespace NUMINAMATH_CALUDE_weight_of_A_l3658_365804

theorem weight_of_A (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 7 →
  (b + c + d + e) / 4 = 79 →
  a = 79 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_A_l3658_365804


namespace NUMINAMATH_CALUDE_william_bottle_caps_l3658_365811

/-- The number of bottle caps William has in total -/
def total_bottle_caps (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that William has 43 bottle caps in total -/
theorem william_bottle_caps : 
  total_bottle_caps 2 41 = 43 := by
  sorry

end NUMINAMATH_CALUDE_william_bottle_caps_l3658_365811


namespace NUMINAMATH_CALUDE_empty_set_subset_subset_transitive_l3658_365876

-- Define the empty set
def emptySet : Set α := ∅

-- Define subset relation
def isSubset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Theorem 1: The empty set is a subset of any set
theorem empty_set_subset (S : Set α) : isSubset emptySet S := by sorry

-- Theorem 2: Transitivity of subset relation
theorem subset_transitive (A B C : Set α) 
  (h1 : isSubset A B) (h2 : isSubset B C) : isSubset A C := by sorry

end NUMINAMATH_CALUDE_empty_set_subset_subset_transitive_l3658_365876


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3658_365818

/-- Given a quadratic inequality ax^2 + bx - 2 > 0 with solution set (1,4), prove a + b = 2 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x - 2 > 0 ↔ 1 < x ∧ x < 4) → 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3658_365818


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l3658_365885

theorem fraction_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (x + y) + y / (y + z) + z / (z + x) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l3658_365885


namespace NUMINAMATH_CALUDE_sum_of_integers_l3658_365826

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 14) 
  (h2 : x.val * y.val = 180) : 
  x.val + y.val = 2 * Real.sqrt 229 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3658_365826


namespace NUMINAMATH_CALUDE_grid_value_l3658_365891

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1 : ℝ)

theorem grid_value (row : ArithmeticSequence) (col : ArithmeticSequence) : 
  row.first = 25 ∧ 
  row.nthTerm 4 = 11 ∧ 
  col.nthTerm 2 = 11 ∧ 
  col.nthTerm 3 = 11 ∧
  row.nthTerm 7 = col.nthTerm 1 →
  row.nthTerm 7 = -3 := by
  sorry


end NUMINAMATH_CALUDE_grid_value_l3658_365891


namespace NUMINAMATH_CALUDE_probability_A_hits_twice_B_hits_thrice_l3658_365899

def probability_A_hits : ℚ := 2/3
def probability_B_hits : ℚ := 3/4
def num_shots : ℕ := 4
def num_A_hits : ℕ := 2
def num_B_hits : ℕ := 3

theorem probability_A_hits_twice_B_hits_thrice : 
  (Nat.choose num_shots num_A_hits * probability_A_hits ^ num_A_hits * (1 - probability_A_hits) ^ (num_shots - num_A_hits)) *
  (Nat.choose num_shots num_B_hits * probability_B_hits ^ num_B_hits * (1 - probability_B_hits) ^ (num_shots - num_B_hits)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_hits_twice_B_hits_thrice_l3658_365899


namespace NUMINAMATH_CALUDE_circle_position_l3658_365824

def circle_center : ℝ × ℝ := (1, 2)
def circle_radius : ℝ := 1

def distance_to_y_axis (center : ℝ × ℝ) : ℝ := |center.1|
def distance_to_x_axis (center : ℝ × ℝ) : ℝ := |center.2|

def is_tangent_to_y_axis (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  distance_to_y_axis center = radius

def is_disjoint_from_x_axis (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  distance_to_x_axis center > radius

theorem circle_position :
  is_tangent_to_y_axis circle_center circle_radius ∧
  is_disjoint_from_x_axis circle_center circle_radius :=
by sorry

end NUMINAMATH_CALUDE_circle_position_l3658_365824


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l3658_365810

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 2

/-- The total number of peanuts after Mary adds some -/
def total_peanuts : ℕ := 6

/-- Theorem stating that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 :=
  by sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l3658_365810


namespace NUMINAMATH_CALUDE_equation_solution_l3658_365820

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^14 = (14 * x)^7 ↔ x = 2/7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3658_365820


namespace NUMINAMATH_CALUDE_power_sum_of_i_l3658_365836

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i :
  i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l3658_365836


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3658_365843

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3658_365843


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3658_365823

theorem quadratic_inequality_solution_set 
  (a : ℝ) (ha : a < 0) :
  {x : ℝ | 42 * x^2 + a * x - a^2 < 0} = {x : ℝ | a / 7 < x ∧ x < -a / 6} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3658_365823


namespace NUMINAMATH_CALUDE_complex_division_by_i_l3658_365895

theorem complex_division_by_i (z : ℂ) : z.re = -2 ∧ z.im = -1 → z / Complex.I = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_by_i_l3658_365895


namespace NUMINAMATH_CALUDE_diagonal_length_16_12_rectangle_l3658_365850

/-- The length of a diagonal in a 16 cm by 12 cm rectangle is 20 cm -/
theorem diagonal_length_16_12_rectangle : 
  ∀ (a b : ℝ), a = 16 ∧ b = 12 → Real.sqrt (a^2 + b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_16_12_rectangle_l3658_365850


namespace NUMINAMATH_CALUDE_number_division_remainder_l3658_365894

theorem number_division_remainder (N : ℤ) (D : ℤ) : 
  N % 281 = 160 → N % D = 21 → D = 139 := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainder_l3658_365894


namespace NUMINAMATH_CALUDE_no_real_roots_implications_l3658_365881

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_real_roots_implications
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_implications_l3658_365881


namespace NUMINAMATH_CALUDE_stone_falling_in_water_l3658_365817

/-- Stone falling in water problem -/
theorem stone_falling_in_water
  (stone_density : ℝ)
  (lake_depth : ℝ)
  (gravity : ℝ)
  (water_density : ℝ)
  (h_stone_density : stone_density = 2.1)
  (h_lake_depth : lake_depth = 8.5)
  (h_gravity : gravity = 980.8)
  (h_water_density : water_density = 1.0) :
  ∃ (time velocity : ℝ),
    (abs (time - 1.82) < 0.01) ∧
    (abs (velocity - 935) < 1) ∧
    time = Real.sqrt ((2 * lake_depth * 100) / ((stone_density - water_density) / stone_density * gravity)) ∧
    velocity = ((stone_density - water_density) / stone_density * gravity) * time :=
  sorry


end NUMINAMATH_CALUDE_stone_falling_in_water_l3658_365817


namespace NUMINAMATH_CALUDE_expression_simplification_l3658_365866

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -a - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3658_365866


namespace NUMINAMATH_CALUDE_vector_calculation_l3658_365830

theorem vector_calculation (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (1, -1) → (1/3 : ℝ) • a - (4/3 : ℝ) • b = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l3658_365830


namespace NUMINAMATH_CALUDE_water_flow_restrictor_problem_l3658_365863

/-- Proves that given a reduced flow rate of 2 gallons per minute, which is 1 gallon per minute less than 0.6 times the original flow rate, the original flow rate is 5 gallons per minute. -/
theorem water_flow_restrictor_problem (original_rate : ℝ) : 
  (2 : ℝ) = 0.6 * original_rate - 1 → original_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_restrictor_problem_l3658_365863


namespace NUMINAMATH_CALUDE_one_color_triangle_l3658_365832

/-- Represents a stick with a color and length -/
structure Stick where
  color : Bool  -- True for blue, False for yellow
  length : ℝ
  positive : length > 0

/-- Represents a hexagon formed by 6 sticks -/
structure Hexagon where
  sticks : Fin 6 → Stick
  alternating : ∀ i, (sticks i).color ≠ (sticks (i + 1)).color
  three_yellow : ∃ a b c, (sticks a).color = false ∧ (sticks b).color = false ∧ (sticks c).color = false ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c
  three_blue : ∃ a b c, (sticks a).color = true ∧ (sticks b).color = true ∧ (sticks c).color = true ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (a b c : Stick) : Prop :=
  a.length + b.length > c.length ∧
  b.length + c.length > a.length ∧
  c.length + a.length > b.length

/-- Main theorem: In a hexagon with alternating colored sticks where any three consecutive sticks can form a triangle, 
    it's possible to form a triangle using sticks of only one color -/
theorem one_color_triangle (h : Hexagon)
  (consecutive_triangle : ∀ i, canFormTriangle (h.sticks i) (h.sticks (i + 1)) (h.sticks (i + 2))) :
  (∃ a b c, (h.sticks a).color = (h.sticks b).color ∧ 
            (h.sticks b).color = (h.sticks c).color ∧ 
            canFormTriangle (h.sticks a) (h.sticks b) (h.sticks c)) :=
by sorry

end NUMINAMATH_CALUDE_one_color_triangle_l3658_365832


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l3658_365828

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence 
  (a : ℕ → ℕ) 
  (h_seq : fibonacci_like_sequence a) 
  (h_7 : a 7 = 42) 
  (h_9 : a 9 = 110) : 
  a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l3658_365828


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_1998_l3658_365803

theorem sum_of_four_cubes_1998 : ∃ (a b c d : ℤ), 1998 = a^3 + b^3 + c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_1998_l3658_365803


namespace NUMINAMATH_CALUDE_dan_remaining_marbles_l3658_365870

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem dan_remaining_marbles :
  initial_marbles - marbles_given = 50 :=
sorry

end NUMINAMATH_CALUDE_dan_remaining_marbles_l3658_365870


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l3658_365827

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_club : ℕ) 
  (science_club : ℕ) 
  (either_club : ℕ) 
  (h1 : total_students = 300)
  (h2 : drama_club = 120)
  (h3 : science_club = 180)
  (h4 : either_club = 250) :
  drama_club + science_club - either_club = 50 := by
  sorry

#check students_in_both_clubs

end NUMINAMATH_CALUDE_students_in_both_clubs_l3658_365827


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l3658_365878

/-- Given a point W with four angles around it, where one angle is 90°, 
    another is y°, a third is 3y°, and the sum of all angles is 360°, 
    prove that y = 67.5° -/
theorem angle_sum_around_point (y : ℝ) : 
  90 + y + 3*y = 360 → y = 67.5 := by sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l3658_365878


namespace NUMINAMATH_CALUDE_min_value_theorem_l3658_365815

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 1000) + (y + 1/x) * (y + 1/x - 1000) ≥ -500000 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3658_365815


namespace NUMINAMATH_CALUDE_valid_a_values_l3658_365884

def A (a : ℝ) : Set ℝ := {2, 1 - a, a^2 - a + 2}

theorem valid_a_values : ∀ a : ℝ, 4 ∈ A a ↔ a = -3 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_valid_a_values_l3658_365884


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3658_365855

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Constructs the first line l1 given parameter a -/
def line1 (a : ℝ) : Line :=
  { a := a, b := a + 2, c := 1 }

/-- Constructs the second line l2 given parameter a -/
def line2 (a : ℝ) : Line :=
  { a := 1, b := a, c := 2 }

/-- States that a = -3 is a sufficient but not necessary condition for perpendicularity -/
theorem perpendicular_condition :
  (∀ a : ℝ, a = -3 → are_perpendicular (line1 a) (line2 a)) ∧
  (∃ a : ℝ, a ≠ -3 ∧ are_perpendicular (line1 a) (line2 a)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3658_365855


namespace NUMINAMATH_CALUDE_point_movement_specific_point_movement_l3658_365867

/-- Given a point P in a 2D Cartesian coordinate system, moving it right and up results in a new point P'. -/
theorem point_movement (x y dx dy : ℝ) :
  let P : ℝ × ℝ := (x, y)
  let P' : ℝ × ℝ := (x + dx, y + dy)
  P' = (x + dx, y + dy) :=
by sorry

/-- The specific case of moving point P(2, -3) right by 2 units and up by 4 units results in P'(4, 1). -/
theorem specific_point_movement :
  let P : ℝ × ℝ := (2, -3)
  let P' : ℝ × ℝ := (2 + 2, -3 + 4)
  P' = (4, 1) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_specific_point_movement_l3658_365867


namespace NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l3658_365816

theorem base_10_to_base_8_conversion : 
  (2 * 8^2 + 3 * 8^1 + 5 * 8^0 : ℕ) = 157 := by sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l3658_365816


namespace NUMINAMATH_CALUDE_card_game_draw_probability_l3658_365851

theorem card_game_draw_probability (ben_win : ℚ) (sara_win : ℚ) (h1 : ben_win = 5 / 12) (h2 : sara_win = 1 / 4) :
  1 - (ben_win + sara_win) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_card_game_draw_probability_l3658_365851


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3658_365853

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Checks if a number is in the range 1 to 16 -/
def validNumber (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 16

/-- Checks if all numbers in the grid are unique and in the range 1 to 16 -/
def validGrid (g : Grid) : Prop :=
  ∀ i j, validNumber (g i j) ∧
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Checks if the sum of numbers in a list is odd -/
def isOddSum (l : List ℕ) : Prop :=
  l.sum % 2 = 1

/-- Gets a row from the grid -/
def getRow (g : Grid) (i : Fin 4) : List ℕ :=
  [g i 0, g i 1, g i 2, g i 3]

/-- Gets a column from the grid -/
def getCol (g : Grid) (j : Fin 4) : List ℕ :=
  [g 0 j, g 1 j, g 2 j, g 3 j]

/-- Gets the main diagonal from top-left to bottom-right -/
def getDiag1 (g : Grid) : List ℕ :=
  [g 0 0, g 1 1, g 2 2, g 3 3]

/-- Gets the main diagonal from top-right to bottom-left -/
def getDiag2 (g : Grid) : List ℕ :=
  [g 0 3, g 1 2, g 2 1, g 3 0]

/-- Checks if all rows, columns, and main diagonals have odd sums -/
def allOddSums (g : Grid) : Prop :=
  (∀ i : Fin 4, isOddSum (getRow g i)) ∧
  (∀ j : Fin 4, isOddSum (getCol g j)) ∧
  isOddSum (getDiag1 g) ∧
  isOddSum (getDiag2 g)

theorem no_valid_arrangement :
  ¬∃ g : Grid, validGrid g ∧ allOddSums g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3658_365853


namespace NUMINAMATH_CALUDE_cn_tower_height_is_553_l3658_365841

/-- The height of the Space Needle in meters -/
def space_needle_height : ℕ := 184

/-- The difference in height between the CN Tower and the Space Needle in meters -/
def height_difference : ℕ := 369

/-- The height of the CN Tower in meters -/
def cn_tower_height : ℕ := space_needle_height + height_difference

theorem cn_tower_height_is_553 : cn_tower_height = 553 := by
  sorry

end NUMINAMATH_CALUDE_cn_tower_height_is_553_l3658_365841


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3658_365848

/-- Given an arithmetic sequence, if A is the sum of the first n terms
    and B is the sum of the first 2n terms, then the sum of the first 3n terms
    is equal to 3(B - A) -/
theorem arithmetic_sequence_sum (n : ℕ) (A B : ℝ) :
  (∃ a d : ℝ, A = (n : ℝ) / 2 * (2 * a + (n - 1) * d) ∧
               B = (2 * n : ℝ) / 2 * (2 * a + (2 * n - 1) * d)) →
  (3 * n : ℝ) / 2 * (2 * a + (3 * n - 1) * d) = 3 * (B - A) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3658_365848


namespace NUMINAMATH_CALUDE_problem_solution_l3658_365858

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3658_365858


namespace NUMINAMATH_CALUDE_max_pieces_8x8_grid_l3658_365864

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)

/-- Represents the number of pieces after cutting -/
def num_pieces (g : Grid) (num_cuts : Nat) : Nat :=
  sorry

/-- The maximum number of pieces that can be obtained from an 8x8 grid -/
theorem max_pieces_8x8_grid :
  ∃ (max_pieces : Nat), 
    (∀ (g : Grid) (num_cuts : Nat), 
      g.size = 8 → num_pieces g num_cuts ≤ max_pieces) ∧ 
    (∃ (g : Grid) (num_cuts : Nat), 
      g.size = 8 ∧ num_pieces g num_cuts = max_pieces) ∧
    max_pieces = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_8x8_grid_l3658_365864


namespace NUMINAMATH_CALUDE_cereal_box_total_price_l3658_365897

/-- Calculates the total price paid for discounted cereal boxes -/
theorem cereal_box_total_price 
  (initial_price : ℕ) 
  (price_reduction : ℕ) 
  (num_boxes : ℕ) 
  (h1 : initial_price = 104)
  (h2 : price_reduction = 24)
  (h3 : num_boxes = 20) : 
  (initial_price - price_reduction) * num_boxes = 1600 := by
  sorry

#check cereal_box_total_price

end NUMINAMATH_CALUDE_cereal_box_total_price_l3658_365897


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3658_365874

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3658_365874


namespace NUMINAMATH_CALUDE_wicket_keeper_age_l3658_365821

theorem wicket_keeper_age (team_size : ℕ) (team_avg_age : ℝ) (remaining_avg_age : ℝ) : 
  team_size = 11 →
  team_avg_age = 24 →
  remaining_avg_age = team_avg_age - 1 →
  ∃ (wicket_keeper_age : ℝ),
    wicket_keeper_age = team_avg_age + 9 ∧
    (team_size - 2) * remaining_avg_age + wicket_keeper_age + team_avg_age = team_size * team_avg_age :=
by sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_l3658_365821


namespace NUMINAMATH_CALUDE_flour_for_two_loaves_l3658_365869

/-- The amount of flour needed for one loaf of bread in cups -/
def flour_per_loaf : ℝ := 2.5

/-- The number of loaves of bread to be baked -/
def num_loaves : ℕ := 2

/-- Theorem: The amount of flour needed for two loaves of bread is 5 cups -/
theorem flour_for_two_loaves : flour_per_loaf * num_loaves = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_two_loaves_l3658_365869


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l3658_365871

theorem circus_ticket_cost (total_spent : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℕ) : 
  total_spent = 308 → num_tickets = 7 → cost_per_ticket = total_spent / num_tickets → cost_per_ticket = 44 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l3658_365871
