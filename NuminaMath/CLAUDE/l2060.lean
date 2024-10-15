import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_special_series_l2060_206038

def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_special_series :
  let a₁ := 1
  let d := 2
  let secondToLast := 99
  let last := 100
  let n := (secondToLast - a₁) / d + 1
  arithmeticSequenceSum a₁ d n + last = 2600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_series_l2060_206038


namespace NUMINAMATH_CALUDE_distinct_arrangements_eq_factorial_l2060_206057

/-- The number of ways to arrange n distinct objects in n positions -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of boxes -/
def num_boxes : ℕ := 5

/-- The number of digits to place -/
def num_digits : ℕ := 4

/-- Theorem: The number of ways to arrange 4 distinct digits and 1 blank in 5 boxes
    is equal to 5! -/
theorem distinct_arrangements_eq_factorial :
  factorial num_boxes = 120 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_eq_factorial_l2060_206057


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2060_206040

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (m n : ℝ) 
  (hmn : m * n = 2 / 9) :
  (((m + n) * c)^2 / a^2 - ((m - n) * b * c / a)^2 / b^2 = 1) →
  (c^2 / a^2 - 1 = (3 * Real.sqrt 2 / 4)^2) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2060_206040


namespace NUMINAMATH_CALUDE_cos_36_cos_24_minus_sin_36_sin_24_l2060_206084

theorem cos_36_cos_24_minus_sin_36_sin_24 :
  Real.cos (36 * π / 180) * Real.cos (24 * π / 180) -
  Real.sin (36 * π / 180) * Real.sin (24 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_cos_24_minus_sin_36_sin_24_l2060_206084


namespace NUMINAMATH_CALUDE_difference_of_squares_701_697_l2060_206088

theorem difference_of_squares_701_697 : 701^2 - 697^2 = 5592 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_701_697_l2060_206088


namespace NUMINAMATH_CALUDE_physics_marks_l2060_206062

theorem physics_marks (total_average : ℝ) (phys_math_avg : ℝ) (phys_chem_avg : ℝ)
  (h1 : total_average = 65)
  (h2 : phys_math_avg = 90)
  (h3 : phys_chem_avg = 70) :
  ∃ (physics chemistry mathematics : ℝ),
    physics + chemistry + mathematics = 3 * total_average ∧
    physics + mathematics = 2 * phys_math_avg ∧
    physics + chemistry = 2 * phys_chem_avg ∧
    physics = 125 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_l2060_206062


namespace NUMINAMATH_CALUDE_digit_difference_l2060_206050

theorem digit_difference (e : ℕ) (X Y : ℕ) : 
  e > 8 →
  X < e →
  Y < e →
  (e * X + Y) + (e * X + X) = 2 * e^2 + 4 * e + 3 →
  X - Y = (2 * e^2 + 4 * e - 726) / 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_l2060_206050


namespace NUMINAMATH_CALUDE_toy_cost_price_l2060_206037

/-- Given a man sold 18 toys for Rs. 25200 and gained the cost price of 3 toys,
    prove that the cost price of a single toy is Rs. 1200. -/
theorem toy_cost_price (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) 
    (h1 : total_selling_price = 25200)
    (h2 : num_toys_sold = 18)
    (h3 : num_toys_gain = 3) :
  ∃ (cost_price : ℕ), cost_price = 1200 ∧ 
    total_selling_price = num_toys_sold * (cost_price + (num_toys_gain * cost_price) / num_toys_sold) :=
by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2060_206037


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2060_206082

theorem sum_of_three_numbers : 300 + 2020 + 10001 = 12321 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2060_206082


namespace NUMINAMATH_CALUDE_equation_solution_l2060_206055

theorem equation_solution : 
  ∃ x : ℚ, (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5) ∧ x = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2060_206055


namespace NUMINAMATH_CALUDE_gcd_of_specific_squares_l2060_206081

theorem gcd_of_specific_squares : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_squares_l2060_206081


namespace NUMINAMATH_CALUDE_square_sum_geq_linear_l2060_206096

theorem square_sum_geq_linear (a b : ℝ) : a^2 + b^2 ≥ 2*(a - b - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_linear_l2060_206096


namespace NUMINAMATH_CALUDE_trapezoid_area_difference_l2060_206077

/-- A trapezoid with specific properties -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  right_angles : ℕ
  
/-- The area difference between largest and smallest regions -/
def area_difference (t : Trapezoid) : ℝ := sorry

theorem trapezoid_area_difference :
  ∀ t : Trapezoid,
    t.side1 = 4 ∧ 
    t.side2 = 4 ∧ 
    t.side3 = 5 ∧ 
    t.side4 = Real.sqrt 17 ∧
    t.right_angles = 2 →
    240 * (area_difference t) = 240 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_difference_l2060_206077


namespace NUMINAMATH_CALUDE_resistance_of_single_rod_l2060_206016

/-- The resistance of the entire construction between points A and B -/
def R : ℝ := 8

/-- The number of identical metallic rods in the network -/
def num_rods : ℕ := 13

/-- The resistance of one rod -/
def R₀ : ℝ := 20

/-- The relation between the total resistance and the resistance of one rod -/
axiom resistance_relation : R = (4/10) * R₀

theorem resistance_of_single_rod : R₀ = 20 :=
  sorry

end NUMINAMATH_CALUDE_resistance_of_single_rod_l2060_206016


namespace NUMINAMATH_CALUDE_equal_sum_number_properties_l2060_206095

def is_equal_sum_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 + (n / 10) % 10 = (n / 100) % 10 + n % 10)

def transform (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

def F (n : ℕ) : ℚ := (n + transform n : ℚ) / 101

def G (n : ℕ) : ℚ := (n - transform n : ℚ) / 99

theorem equal_sum_number_properties :
  (∀ n : ℕ, is_equal_sum_number n → 
    (F n - G n = 72 → n = 5236)) ∧
  (∃ n : ℕ, is_equal_sum_number n ∧ 
    (F n / 13).isInt ∧ (G n / 7).isInt ∧
    (∀ m : ℕ, is_equal_sum_number m ∧ 
      (F m / 13).isInt ∧ (G m / 7).isInt → m ≤ n) ∧
    n = 9647) := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_number_properties_l2060_206095


namespace NUMINAMATH_CALUDE_family_ages_l2060_206048

theorem family_ages (oleg_age : ℕ) (father_age : ℕ) (grandfather_age : ℕ) :
  father_age = oleg_age + 32 →
  grandfather_age = father_age + 32 →
  (oleg_age - 3) + (father_age - 3) + (grandfather_age - 3) < 100 →
  oleg_age > 0 →
  oleg_age = 4 ∧ father_age = 36 ∧ grandfather_age = 68 := by
  sorry

#check family_ages

end NUMINAMATH_CALUDE_family_ages_l2060_206048


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l2060_206060

theorem arccos_one_equals_zero :
  Real.arccos 1 = 0 := by
  sorry

#check arccos_one_equals_zero

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l2060_206060


namespace NUMINAMATH_CALUDE_sunzi_remainder_problem_l2060_206054

theorem sunzi_remainder_problem :
  let S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3}
  (∃ (min max : ℕ), min ∈ S ∧ max ∈ S ∧
    (∀ x ∈ S, min ≤ x) ∧
    (∀ x ∈ S, x ≤ max) ∧
    min + max = 196) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_remainder_problem_l2060_206054


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l2060_206029

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan 3) = 13 * Real.sqrt 10 / 50 := by sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l2060_206029


namespace NUMINAMATH_CALUDE_intersection_M_N_l2060_206003

def M : Set ℝ := {-1, 0, 1}

def N : Set ℝ := {x : ℝ | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2060_206003


namespace NUMINAMATH_CALUDE_employees_with_advanced_degrees_l2060_206067

/-- Proves that the number of employees with advanced degrees is 78 given the conditions in the problem -/
theorem employees_with_advanced_degrees :
  ∀ (total_employees : ℕ) 
    (female_employees : ℕ) 
    (male_college_only : ℕ) 
    (female_advanced : ℕ),
  total_employees = 148 →
  female_employees = 92 →
  male_college_only = 31 →
  female_advanced = 53 →
  ∃ (male_advanced : ℕ),
    male_advanced + female_advanced + male_college_only + (female_employees - female_advanced) = total_employees ∧
    male_advanced + female_advanced = 78 :=
by sorry

end NUMINAMATH_CALUDE_employees_with_advanced_degrees_l2060_206067


namespace NUMINAMATH_CALUDE_ellipse_constants_l2060_206068

/-- An ellipse with foci at (1, 1) and (1, 5) passing through (12, -4) -/
structure Ellipse where
  foci1 : ℝ × ℝ := (1, 1)
  foci2 : ℝ × ℝ := (1, 5)
  point : ℝ × ℝ := (12, -4)

/-- The standard form of an ellipse equation -/
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating the constants of the ellipse -/
theorem ellipse_constants (e : Ellipse) :
  ∃ (a b h k : ℝ),
    a > 0 ∧ b > 0 ∧
    a = 13 ∧ b = Real.sqrt 153 ∧ h = 1 ∧ k = 3 ∧
    standard_form a b h k e.point.1 e.point.2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_constants_l2060_206068


namespace NUMINAMATH_CALUDE_print_shop_charge_l2060_206078

/-- The charge per color copy at print shop X -/
def charge_x : ℝ := 1.20

/-- The charge per color copy at print shop Y -/
def charge_y : ℝ := 1.70

/-- The number of copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℝ := 20

theorem print_shop_charge : 
  charge_x * num_copies + additional_charge = charge_y * num_copies :=
by sorry

end NUMINAMATH_CALUDE_print_shop_charge_l2060_206078


namespace NUMINAMATH_CALUDE_order_of_expressions_l2060_206010

theorem order_of_expressions :
  let a := (1/3 : ℝ) ^ Real.pi
  let b := (1/3 : ℝ) ^ (1/2)
  let c := Real.pi ^ (1/2)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2060_206010


namespace NUMINAMATH_CALUDE_find_k_l2060_206099

theorem find_k (d m n : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, (3*x^2 - 4*x + 2)*(d*x^3 + k*x^2 + m*x + n) = 6*x^5 - 11*x^4 + 14*x^3 - 4*x^2 + 8*x - 3) → 
  (∃ k : ℝ, k = -11/3 ∧ ∀ x : ℝ, (3*x^2 - 4*x + 2)*(d*x^3 + k*x^2 + m*x + n) = 6*x^5 - 11*x^4 + 14*x^3 - 4*x^2 + 8*x - 3) :=
by sorry

end NUMINAMATH_CALUDE_find_k_l2060_206099


namespace NUMINAMATH_CALUDE_player_a_wins_iff_perfect_square_l2060_206039

/-- The divisor erasing game on a positive integer N -/
def DivisorGame (N : ℕ+) :=
  ∀ (d : ℕ+), d ∣ N → (∃ (m : ℕ+), m ∣ N ∧ (d ∣ m ∨ m ∣ d))

/-- Player A's winning condition -/
def PlayerAWins (N : ℕ+) :=
  ∀ (strategy : ℕ+ → ℕ+),
    (∀ (d : ℕ+), d ∣ N → strategy d ∣ N ∧ (d ∣ strategy d ∨ strategy d ∣ d)) →
    ∃ (move : ℕ+ → ℕ+), 
      (∀ (d : ℕ+), d ∣ N → move d ∣ N ∧ (d ∣ move d ∨ move d ∣ d)) ∧
      (∀ (d : ℕ+), d ∣ N → move (strategy (move d)) ≠ d)

/-- The main theorem: Player A wins if and only if N is a perfect square -/
theorem player_a_wins_iff_perfect_square (N : ℕ+) :
  PlayerAWins N ↔ ∃ (n : ℕ+), N = n * n :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_iff_perfect_square_l2060_206039


namespace NUMINAMATH_CALUDE_field_resizing_problem_l2060_206018

theorem field_resizing_problem : ∃ m : ℝ, 
  m > 0 ∧ (3 * m + 14) * (m + 1) = 240 ∧ abs (m - 6.3) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_field_resizing_problem_l2060_206018


namespace NUMINAMATH_CALUDE_smallest_marble_collection_l2060_206030

theorem smallest_marble_collection : ∀ n : ℕ,
  (n % 4 = 0) →  -- one fourth are red
  (n % 5 = 0) →  -- one fifth are blue
  (n ≥ 8 + 5) →  -- at least 8 white and 5 green
  (∃ r b w g : ℕ, 
    r + b + w + g = n ∧
    r = n / 4 ∧
    b = n / 5 ∧
    w = 8 ∧
    g = 5) →
  n ≥ 220 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_collection_l2060_206030


namespace NUMINAMATH_CALUDE_eraser_distribution_l2060_206097

theorem eraser_distribution (total_erasers : ℕ) (num_friends : ℕ) (erasers_per_friend : ℕ) :
  total_erasers = 3840 →
  num_friends = 48 →
  erasers_per_friend = total_erasers / num_friends →
  erasers_per_friend = 80 :=
by sorry

end NUMINAMATH_CALUDE_eraser_distribution_l2060_206097


namespace NUMINAMATH_CALUDE_rational_division_and_linear_combination_l2060_206007

theorem rational_division_and_linear_combination (m a b c d k : ℤ) : 
  (∀ (x : ℤ), (x ∣ 5*m + 6 ∧ x ∣ 8*m + 7) ↔ (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13)) ∧
  ((k ∣ a*m + b ∧ k ∣ c*m + d) → k ∣ a*d - b*c) := by
  sorry

end NUMINAMATH_CALUDE_rational_division_and_linear_combination_l2060_206007


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l2060_206092

theorem unique_solution_for_system :
  ∃! (x y z : ℕ+), 
    (x.val : ℤ)^2 + y.val - z.val = 100 ∧ 
    (x.val : ℤ) + y.val^2 - z.val = 124 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l2060_206092


namespace NUMINAMATH_CALUDE_function_min_value_l2060_206063

/-- Given constants a and b, and a function f with specific properties, 
    prove that its minimum value on (0, +∞) is -4 -/
theorem function_min_value 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^3 + b * Real.log (x + Real.sqrt (1 + x^2)) + 3)
  (h_max : ∀ x < 0, f x ≤ 10)
  (h_exists_max : ∃ x < 0, f x = 10) :
  ∃ y > 0, ∀ x > 0, f x ≥ f y ∧ f y = -4 := by
sorry

end NUMINAMATH_CALUDE_function_min_value_l2060_206063


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l2060_206006

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l2060_206006


namespace NUMINAMATH_CALUDE_negation_of_union_l2060_206066

theorem negation_of_union (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end NUMINAMATH_CALUDE_negation_of_union_l2060_206066


namespace NUMINAMATH_CALUDE_range_of_b_l2060_206026

theorem range_of_b (a b c : ℝ) (sum_cond : a + b + c = 9) (prod_cond : a * b + b * c + c * a = 24) :
  1 ≤ b ∧ b ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l2060_206026


namespace NUMINAMATH_CALUDE_equal_pairs_comparison_l2060_206042

theorem equal_pairs_comparison : 
  (-3^5 = (-3)^5) ∧ 
  (-2^2 ≠ (-2)^2) ∧ 
  (-4 * 2^3 ≠ -4^2 * 3) ∧ 
  (-(-3)^2 ≠ -(-2)^3) :=
by sorry

end NUMINAMATH_CALUDE_equal_pairs_comparison_l2060_206042


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2060_206012

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  h1 : a 6 = 12
  h2 : S 3 = 12
  h3 : ∀ n : ℕ, S n = (n / 2) * (a 1 + a n)  -- Sum formula for arithmetic sequence
  h4 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Common difference property

/-- The general term of the arithmetic sequence is 2n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2060_206012


namespace NUMINAMATH_CALUDE_log_expression_equals_three_l2060_206071

theorem log_expression_equals_three :
  (Real.log 243 / Real.log 3) / (Real.log 27 / Real.log 3) -
  (Real.log 729 / Real.log 3) / (Real.log 9 / Real.log 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_three_l2060_206071


namespace NUMINAMATH_CALUDE_halving_period_correct_l2060_206093

/-- The number of years it takes for the cost of a ticket to Mars to be halved. -/
def halving_period : ℕ := 10

/-- The initial cost of a ticket to Mars in dollars. -/
def initial_cost : ℕ := 1000000

/-- The cost of a ticket to Mars after 30 years in dollars. -/
def cost_after_30_years : ℕ := 125000

/-- The number of years passed. -/
def years_passed : ℕ := 30

/-- Theorem stating that the halving period is correct given the initial conditions. -/
theorem halving_period_correct : 
  initial_cost / (2 ^ (years_passed / halving_period)) = cost_after_30_years :=
sorry

end NUMINAMATH_CALUDE_halving_period_correct_l2060_206093


namespace NUMINAMATH_CALUDE_incorrect_expression_l2060_206004

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) :
  ¬ ((x - y) / y = -1 / 6) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l2060_206004


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l2060_206061

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs (z - 3 + 3 * Complex.I) = 3) :
  ∃ (min : ℝ), min = 59 ∧ ∀ (w : ℂ), Complex.abs (w - 3 + 3 * Complex.I) = 3 →
    Complex.abs (w - 2 - Complex.I) ^ 2 + Complex.abs (w - 6 + 2 * Complex.I) ^ 2 ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l2060_206061


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l2060_206036

/-- Given a triangle with sides a, b, c, this theorem proves bounds on its area S. -/
theorem triangle_area_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := S / p
  3 * Real.sqrt 3 * r^2 ≤ S ∧ S ≤ p^2 / (3 * Real.sqrt 3) ∧
  S ≤ (a^2 + b^2 + c^2) / (4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l2060_206036


namespace NUMINAMATH_CALUDE_tan_sum_of_quadratic_roots_l2060_206024

theorem tan_sum_of_quadratic_roots (α β : Real) (h : ∀ x, x^2 + 6*x + 7 = 0 ↔ x = Real.tan α ∨ x = Real.tan β) :
  Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_of_quadratic_roots_l2060_206024


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l2060_206083

/-- The distance of a fly from the ceiling in a room -/
theorem fly_distance_from_ceiling :
  ∀ (z : ℝ),
  (2 : ℝ)^2 + 5^2 + z^2 = 7^2 →
  z = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l2060_206083


namespace NUMINAMATH_CALUDE_min_sum_squares_y_coords_l2060_206032

/-- 
Given a line passing through (4, 0) and intersecting the parabola y^2 = 4x at two points,
the minimum value of the sum of the squares of the y-coordinates of these two points is 32.
-/
theorem min_sum_squares_y_coords : 
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
  y₁^2 = 4 * (m * y₁ + 4) →
  y₂^2 = 4 * (m * y₂ + 4) →
  y₁ ≠ y₂ →
  ∀ (z₁ z₂ : ℝ),
  z₁^2 = 4 * (m * z₁ + 4) →
  z₂^2 = 4 * (m * z₂ + 4) →
  z₁ ≠ z₂ →
  y₁^2 + y₂^2 ≤ z₁^2 + z₂^2 →
  y₁^2 + y₂^2 = 32 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_squares_y_coords_l2060_206032


namespace NUMINAMATH_CALUDE_percentage_problem_l2060_206051

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : P / 100 * N = 160)
  (h2 : 60 / 100 * N = 240) : 
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2060_206051


namespace NUMINAMATH_CALUDE_carl_driving_hours_l2060_206043

theorem carl_driving_hours :
  let daily_hours : ℕ := 2
  let additional_weekly_hours : ℕ := 6
  let days_in_two_weeks : ℕ := 14
  let weeks : ℕ := 2
  (daily_hours * days_in_two_weeks) + (additional_weekly_hours * weeks) = 40 :=
by sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l2060_206043


namespace NUMINAMATH_CALUDE_equation_solution_l2060_206019

theorem equation_solution : ∃! x : ℝ, (81 : ℝ) ^ (x - 1) / (9 : ℝ) ^ (x - 1) = (729 : ℝ) ^ x ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2060_206019


namespace NUMINAMATH_CALUDE_divisibility_implication_l2060_206027

theorem divisibility_implication (n m : ℤ) : 
  (31 ∣ (6 * n + 11 * m)) → (31 ∣ (n + 7 * m)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2060_206027


namespace NUMINAMATH_CALUDE_total_amount_is_2150_6_l2060_206087

/-- Calculates the total amount paid for fruits with discounts and taxes -/
def total_amount_paid (
  grapes_kg : ℝ)   (grapes_price : ℝ)
  (mangoes_kg : ℝ)  (mangoes_price : ℝ)
  (apples_kg : ℝ)   (apples_price : ℝ)
  (strawberries_kg : ℝ) (strawberries_price : ℝ)
  (oranges_kg : ℝ)  (oranges_price : ℝ)
  (kiwis_kg : ℝ)    (kiwis_price : ℝ)
  (grapes_apples_discount : ℝ)
  (oranges_kiwis_discount : ℝ)
  (mangoes_strawberries_tax : ℝ) : ℝ :=
  let grapes_total := grapes_kg * grapes_price
  let mangoes_total := mangoes_kg * mangoes_price
  let apples_total := apples_kg * apples_price
  let strawberries_total := strawberries_kg * strawberries_price
  let oranges_total := oranges_kg * oranges_price
  let kiwis_total := kiwis_kg * kiwis_price
  
  let total_before_discounts_taxes := grapes_total + mangoes_total + apples_total + 
                                      strawberries_total + oranges_total + kiwis_total
  
  let grapes_apples_discount_amount := (grapes_total + apples_total) * grapes_apples_discount
  let oranges_kiwis_discount_amount := (oranges_total + kiwis_total) * oranges_kiwis_discount
  let mangoes_strawberries_tax_amount := (mangoes_total + strawberries_total) * mangoes_strawberries_tax
  
  total_before_discounts_taxes - grapes_apples_discount_amount - oranges_kiwis_discount_amount + mangoes_strawberries_tax_amount

/-- Theorem stating that the total amount paid for fruits is 2150.6 -/
theorem total_amount_is_2150_6 :
  total_amount_paid 8 70 9 45 5 30 3 100 10 40 6 60 0.1 0.05 0.12 = 2150.6 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_2150_6_l2060_206087


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l2060_206013

/-- The trajectory of point Q given point P on a circle -/
theorem trajectory_of_Q (m n : ℝ) : 
  m^2 + n^2 = 2 →   -- P is on the circle x^2 + y^2 = 2
  ∃ x y : ℝ,
    x = m + n ∧     -- x-coordinate of Q
    y = 2 * m * n ∧ -- y-coordinate of Q
    y = x^2 - 2 ∧   -- trajectory equation
    -2 ≤ x ∧ x ≤ 2  -- domain constraint
  := by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l2060_206013


namespace NUMINAMATH_CALUDE_vanya_cookies_l2060_206047

theorem vanya_cookies (total : ℚ) (vanya_before : ℚ) (shared : ℚ) :
  total > 0 ∧ vanya_before ≥ 0 ∧ shared ≥ 0 ∧
  total = vanya_before + shared ∧
  vanya_before + shared / 2 = 5 * (shared / 2) →
  vanya_before / total = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_vanya_cookies_l2060_206047


namespace NUMINAMATH_CALUDE_remainder_71_73_mod_8_l2060_206001

theorem remainder_71_73_mod_8 : (71 * 73) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_73_mod_8_l2060_206001


namespace NUMINAMATH_CALUDE_contact_lenses_sold_l2060_206056

/-- Represents the number of pairs of hard contact lenses sold -/
def hard_lenses : ℕ := sorry

/-- Represents the number of pairs of soft contact lenses sold -/
def soft_lenses : ℕ := sorry

/-- The price of a pair of hard contact lenses in cents -/
def hard_price : ℕ := 8500

/-- The price of a pair of soft contact lenses in cents -/
def soft_price : ℕ := 15000

/-- The total sales in cents -/
def total_sales : ℕ := 145500

theorem contact_lenses_sold :
  (soft_lenses = hard_lenses + 5) →
  (hard_price * hard_lenses + soft_price * soft_lenses = total_sales) →
  (hard_lenses + soft_lenses = 11) := by
  sorry

end NUMINAMATH_CALUDE_contact_lenses_sold_l2060_206056


namespace NUMINAMATH_CALUDE_gcd_product_is_square_l2060_206075

theorem gcd_product_is_square (x y z : ℕ+) 
  (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x.val (Nat.gcd y.val z.val)) * x.val * y.val * z.val = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_is_square_l2060_206075


namespace NUMINAMATH_CALUDE_shekars_social_studies_score_l2060_206035

theorem shekars_social_studies_score 
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (total_subjects : ℕ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 85)
  (h5 : average_score = 75)
  (h6 : total_subjects = 5) :
  ∃ (social_studies_score : ℕ),
    social_studies_score = 82 ∧
    (math_score + science_score + english_score + biology_score + social_studies_score : ℚ) / total_subjects = average_score :=
by
  sorry

end NUMINAMATH_CALUDE_shekars_social_studies_score_l2060_206035


namespace NUMINAMATH_CALUDE_cube_difference_l2060_206074

theorem cube_difference (c d : ℝ) (h1 : c - d = 7) (h2 : c^2 + d^2 = 85) : c^3 - d^3 = 721 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l2060_206074


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l2060_206023

-- Define the exponent
def n : ℕ := 2008

-- Define the function to count terms
def countTerms (n : ℕ) : ℕ :=
  (n / 2 + 1) * (n + 1)

-- Theorem statement
theorem simplified_expression_terms :
  countTerms n = 2018045 :=
sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l2060_206023


namespace NUMINAMATH_CALUDE_divisor_problem_l2060_206094

theorem divisor_problem (D : ℕ) (hD : D > 0) 
  (h1 : 242 % D = 6)
  (h2 : 698 % D = 13)
  (h3 : 940 % D = 5) : 
  D = 14 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2060_206094


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l2060_206015

-- Define the total amount spent excluding taxes
variable (T : ℝ)

-- Define the tax rate on clothing as a percentage
variable (x : ℝ)

-- Define the spending percentages
def clothing_percent : ℝ := 0.45
def food_percent : ℝ := 0.45
def other_percent : ℝ := 0.10

-- Define the tax rates
def other_tax_rate : ℝ := 0.10
def total_tax_rate : ℝ := 0.0325

-- Theorem statement
theorem clothing_tax_rate :
  clothing_percent * T * (x / 100) + other_percent * T * other_tax_rate = total_tax_rate * T →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l2060_206015


namespace NUMINAMATH_CALUDE_father_ate_eight_brownies_l2060_206079

/-- The number of brownies Father ate -/
def fatherAte (initialBrownies : ℕ) (mooneyAte : ℕ) (additionalBrownies : ℕ) (finalBrownies : ℕ) : ℕ :=
  initialBrownies + additionalBrownies - mooneyAte - finalBrownies

/-- Proves that Father ate 8 brownies given the problem conditions -/
theorem father_ate_eight_brownies :
  fatherAte (2 * 12) 4 (2 * 12) 36 = 8 := by
  sorry

#eval fatherAte (2 * 12) 4 (2 * 12) 36

end NUMINAMATH_CALUDE_father_ate_eight_brownies_l2060_206079


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l2060_206069

theorem rectangle_area_problem :
  ∃ (length width : ℕ+), 
    (length : ℝ) * (width : ℝ) = ((length : ℝ) + 3) * ((width : ℝ) - 1) ∧
    (length : ℝ) * (width : ℝ) = ((length : ℝ) - 3) * ((width : ℝ) + 2) ∧
    (length : ℝ) * (width : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l2060_206069


namespace NUMINAMATH_CALUDE_greater_number_problem_l2060_206002

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 16) : max x y = 33 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l2060_206002


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l2060_206045

theorem salt_mixture_proof :
  let initial_amount : ℝ := 150
  let initial_concentration : ℝ := 0.35
  let added_amount : ℝ := 120
  let added_concentration : ℝ := 0.80
  let final_concentration : ℝ := 0.55
  
  (initial_amount * initial_concentration + added_amount * added_concentration) / (initial_amount + added_amount) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_salt_mixture_proof_l2060_206045


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2060_206005

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 - m * x + n = 0 ∧ 2 * y^2 - m * y + n = 0 ∧ x + y = 6 ∧ x * y = 10) →
  m + n = 32 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2060_206005


namespace NUMINAMATH_CALUDE_dans_egg_purchase_l2060_206014

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Dan bought -/
def total_eggs : ℕ := 108

/-- The number of dozens of eggs Dan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem dans_egg_purchase : dozens_bought = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_egg_purchase_l2060_206014


namespace NUMINAMATH_CALUDE_otimes_h_otimes_h_l2060_206028

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 - x*y + y^2

/-- Theorem stating the result of h ⊗ (h ⊗ h) -/
theorem otimes_h_otimes_h (h : ℝ) : otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_h_otimes_h_l2060_206028


namespace NUMINAMATH_CALUDE_students_who_got_on_correct_l2060_206046

/-- The number of students who got on the bus at the first stop -/
def students_who_got_on (initial_students final_students : ℝ) : ℝ :=
  final_students - initial_students

theorem students_who_got_on_correct (initial_students final_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : final_students = 13) :
  students_who_got_on initial_students final_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_who_got_on_correct_l2060_206046


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l2060_206070

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem units_digit_47_power_47 : unitsDigit (47^47) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l2060_206070


namespace NUMINAMATH_CALUDE_sin_cos_power_inequality_l2060_206044

theorem sin_cos_power_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  (Real.sin x) ^ (Real.sin x) < (Real.cos x) ^ (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_inequality_l2060_206044


namespace NUMINAMATH_CALUDE_no_finite_maximum_for_expression_l2060_206090

open Real

theorem no_finite_maximum_for_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_constraint : x + y + z = 9) : 
  ¬ ∃ (M : ℝ), ∀ (x y z : ℝ), 
    0 < x → 0 < y → 0 < z → x + y + z = 9 →
    (x^2 + 2*y^2)/(x + y) + (2*x^2 + z^2)/(x + z) + (y^2 + 2*z^2)/(y + z) ≤ M :=
sorry

end NUMINAMATH_CALUDE_no_finite_maximum_for_expression_l2060_206090


namespace NUMINAMATH_CALUDE_total_musicians_l2060_206065

/-- Represents a musical group with a specific number of male and female musicians. -/
structure MusicGroup where
  males : Nat
  females : Nat

/-- The total number of musicians in a group is the sum of males and females. -/
def MusicGroup.total (g : MusicGroup) : Nat :=
  g.males + g.females

/-- The orchestra has 11 males and 12 females. -/
def orchestra : MusicGroup :=
  { males := 11, females := 12 }

/-- The band has twice the number of musicians as the orchestra. -/
def band : MusicGroup :=
  { males := 2 * orchestra.males, females := 2 * orchestra.females }

/-- The choir has 12 males and 17 females. -/
def choir : MusicGroup :=
  { males := 12, females := 17 }

/-- Theorem: The total number of musicians in the orchestra, band, and choir is 98. -/
theorem total_musicians :
  orchestra.total + band.total + choir.total = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_musicians_l2060_206065


namespace NUMINAMATH_CALUDE_triangle_tangent_identity_l2060_206053

theorem triangle_tangent_identity (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  Real.tan (A/2) * Real.tan (B/2) + Real.tan (B/2) * Real.tan (C/2) + Real.tan (C/2) * Real.tan (A/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_identity_l2060_206053


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_is_18_min_value_exists_l2060_206009

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8/x + 1/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 8/a + 1/b = 1 → x + 2*y ≤ a + 2*b :=
by sorry

theorem min_value_is_18 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8/x + 1/y = 1) :
  x + 2*y ≥ 18 :=
by sorry

theorem min_value_exists :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 8/x + 1/y = 1 ∧ x + 2*y = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_is_18_min_value_exists_l2060_206009


namespace NUMINAMATH_CALUDE_subtracted_number_l2060_206025

theorem subtracted_number (m n x : ℕ) : 
  m > 0 → n > 0 → m = 15 * n - x → m % 5 = 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2060_206025


namespace NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2060_206098

/-- The monomial type -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  vars : List (α × Nat)

/-- Definition of the coefficient of a monomial -/
def coefficient (m : Monomial ℚ) : ℚ := m.coeff

/-- Definition of the degree of a monomial -/
def degree (m : Monomial ℚ) : Nat := m.vars.foldr (λ (_, exp) acc => acc + exp) 0

/-- The given monomial -/
def m : Monomial ℚ := ⟨-3/5, [(1, 1), (1, 2)]⟩

theorem monomial_coefficient_and_degree :
  coefficient m = -3/5 ∧ degree m = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2060_206098


namespace NUMINAMATH_CALUDE_gcd_98_63_l2060_206059

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l2060_206059


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2060_206033

theorem trigonometric_identity : 
  (Real.sin (110 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2060_206033


namespace NUMINAMATH_CALUDE_cost_per_side_of_square_park_l2060_206076

/-- Represents the cost of fencing a square park -/
def CostOfFencing : Type :=
  { total : ℕ // total > 0 }

/-- Calculates the cost of fencing each side of a square park -/
def costPerSide (c : CostOfFencing) : ℕ :=
  c.val / 4

/-- Theorem: The cost of fencing each side of a square park is 43 dollars,
    given that the total cost of fencing is 172 dollars -/
theorem cost_per_side_of_square_park :
  ∀ (c : CostOfFencing), c.val = 172 → costPerSide c = 43 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_side_of_square_park_l2060_206076


namespace NUMINAMATH_CALUDE_prob_more_heads_than_tails_is_correct_l2060_206085

/-- The probability of getting more heads than tails when flipping 10 coins -/
def prob_more_heads_than_tails : ℚ := 193 / 512

/-- The number of coins being flipped -/
def num_coins : ℕ := 10

/-- The total number of possible outcomes when flipping 10 coins -/
def total_outcomes : ℕ := 2^num_coins

/-- The probability of getting exactly 5 heads (and 5 tails) when flipping 10 coins -/
def prob_equal_heads_tails : ℚ := 63 / 256

theorem prob_more_heads_than_tails_is_correct :
  prob_more_heads_than_tails = (1 - prob_equal_heads_tails) / 2 :=
sorry

end NUMINAMATH_CALUDE_prob_more_heads_than_tails_is_correct_l2060_206085


namespace NUMINAMATH_CALUDE_rectangles_not_always_similar_l2060_206031

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define similarity for rectangles
def are_similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Theorem statement
theorem rectangles_not_always_similar :
  ∃ (r1 r2 : Rectangle), ¬(are_similar r1 r2) :=
sorry

end NUMINAMATH_CALUDE_rectangles_not_always_similar_l2060_206031


namespace NUMINAMATH_CALUDE_min_value_of_a_l2060_206064

theorem min_value_of_a (a b : ℤ) : 
  a < 26 → 
  b > 14 → 
  b < 31 → 
  (a : ℚ) / (b : ℚ) ≥ 4/3 → 
  a ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2060_206064


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2060_206058

/-- The coordinates of a point symmetric to P(2,3) with respect to the x-axis are (2,-3) -/
theorem symmetric_point_x_axis : 
  let P : ℝ × ℝ := (2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2060_206058


namespace NUMINAMATH_CALUDE_john_candies_proof_l2060_206091

/-- The number of candies John has -/
def john_candies : ℕ := 35

/-- The number of candies Mark has -/
def mark_candies : ℕ := 30

/-- The number of candies Peter has -/
def peter_candies : ℕ := 25

/-- The number of candies each person has after sharing equally -/
def shared_candies : ℕ := 30

/-- The number of people sharing the candies -/
def num_people : ℕ := 3

theorem john_candies_proof :
  john_candies = shared_candies * num_people - mark_candies - peter_candies :=
by sorry

end NUMINAMATH_CALUDE_john_candies_proof_l2060_206091


namespace NUMINAMATH_CALUDE_plane_division_l2060_206049

/-- The maximum number of parts a plane can be divided into by n lines -/
def f (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of parts a plane can be divided into by n lines is (n^2 + n + 2) / 2 -/
theorem plane_division (n : ℕ) : f n = (n^2 + n + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l2060_206049


namespace NUMINAMATH_CALUDE_money_value_difference_l2060_206073

theorem money_value_difference (exchange_rate : ℝ) (marco_dollars : ℝ) (juliette_euros : ℝ) :
  exchange_rate = 1.5 →
  marco_dollars = 600 →
  juliette_euros = 350 →
  let juliette_dollars := juliette_euros * exchange_rate
  (marco_dollars - juliette_dollars) / marco_dollars * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_money_value_difference_l2060_206073


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2060_206080

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_product : a 1 * a 7 = 3/4) :
  a 4 = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2060_206080


namespace NUMINAMATH_CALUDE_min_value_theorem_l2060_206022

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 5) :
  (9 / a) + (16 / b) + (25 / c) ≥ 144 / 5 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 5 ∧
    (9 / a') + (16 / b') + (25 / c') = 144 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2060_206022


namespace NUMINAMATH_CALUDE_number_square_equation_l2060_206072

theorem number_square_equation : ∃ x : ℝ, x^2 + 145 = (x - 19)^2 ∧ x = 108/19 := by
  sorry

end NUMINAMATH_CALUDE_number_square_equation_l2060_206072


namespace NUMINAMATH_CALUDE_escalator_walking_speed_l2060_206021

/-- Proves that a person walks at 5 ft/sec on an escalator given specific conditions -/
theorem escalator_walking_speed 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15) 
  (h2 : escalator_length = 200) 
  (h3 : time_taken = 10) : 
  ∃ (walking_speed : ℝ), 
    walking_speed = 5 ∧ 
    escalator_length = (walking_speed + escalator_speed) * time_taken :=
by
  sorry

end NUMINAMATH_CALUDE_escalator_walking_speed_l2060_206021


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2060_206008

def polynomial (x : ℝ) : ℝ := 10 * x^4 - 22 * x^3 + 5 * x^2 - 8 * x - 45

def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2060_206008


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l2060_206041

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 →
    1 / (x^3 + 2*x^2 - 17*x - 30) = A / (x - 5) + B / (x + 2) + C / (x + 2)^2) →
  A = 1 / 49 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l2060_206041


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2060_206034

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / ((x - 4)^2 + 8) ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2060_206034


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2060_206086

/-- Proves that the number of crayons lost or given away is equal to the sum of crayons given away and crayons lost -/
theorem crayons_lost_or_given_away 
  (initial : ℕ) 
  (given_away : ℕ) 
  (lost : ℕ) 
  (left : ℕ) 
  (h1 : initial = given_away + lost + left) : 
  given_away + lost = initial - left :=
by sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2060_206086


namespace NUMINAMATH_CALUDE_problem_solution_l2060_206017

theorem problem_solution (x : ℝ) (h : x^2 - 3*x - 1 = 0) : -3*x^2 + 9*x + 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2060_206017


namespace NUMINAMATH_CALUDE_first_group_number_from_sixteenth_l2060_206020

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_group_number : ℕ
  sixteenth_group_number : ℕ

/-- Theorem stating the relationship between the 1st and 16th group numbers in the given systematic sampling -/
theorem first_group_number_from_sixteenth
  (s : SystematicSampling)
  (h1 : s.population_size = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = 8)
  (h4 : s.sixteenth_group_number = 126) :
  s.first_group_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_group_number_from_sixteenth_l2060_206020


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l2060_206052

/-- Represents the dimensions and frame properties of a painting --/
structure FramedPainting where
  width : ℝ
  height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting --/
def framedDimensions (p : FramedPainting) : ℝ × ℝ :=
  (p.width + 2 * p.side_frame_width, p.height + 6 * p.side_frame_width)

/-- Calculates the area of the framed painting --/
def framedArea (p : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions p
  w * h

/-- Calculates the area of the original painting --/
def paintingArea (p : FramedPainting) : ℝ :=
  p.width * p.height

/-- Theorem statement for the framed painting problem --/
theorem framed_painting_ratio (p : FramedPainting)
  (h1 : p.width = 20)
  (h2 : p.height = 30)
  (h3 : framedArea p = 2 * paintingArea p) :
  let (w, h) := framedDimensions p
  w / h = 1 / 2 := by
  sorry

#check framed_painting_ratio

end NUMINAMATH_CALUDE_framed_painting_ratio_l2060_206052


namespace NUMINAMATH_CALUDE_simplify_expression_l2060_206089

theorem simplify_expression (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  12 * x^3 * y^4 / (9 * x^2 * y^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2060_206089


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_l2060_206000

theorem smallest_perfect_cube (Z K : ℤ) : 
  (2000 < Z) → (Z < 3000) → (K > 1) → (Z = K * K^2) → 
  (∃ n : ℤ, Z = n^3) → 
  (∀ K' : ℤ, K' < K → ¬(2000 < K'^3 ∧ K'^3 < 3000)) →
  K = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_l2060_206000


namespace NUMINAMATH_CALUDE_apps_deleted_minus_added_l2060_206011

theorem apps_deleted_minus_added (initial_apps added_apps final_apps : ℕ) : 
  initial_apps + added_apps - final_apps - added_apps = 3 :=
by
  sorry

#check apps_deleted_minus_added 32 125 29

end NUMINAMATH_CALUDE_apps_deleted_minus_added_l2060_206011
