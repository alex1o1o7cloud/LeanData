import Mathlib

namespace NUMINAMATH_CALUDE_solve_race_problem_l1690_169078

def race_problem (patrick_time manu_extra_time : ℕ) (amy_speed_ratio : ℚ) : Prop :=
  let manu_time := patrick_time + manu_extra_time
  let amy_time := manu_time / amy_speed_ratio
  amy_time = 36

theorem solve_race_problem :
  race_problem 60 12 2 := by sorry

end NUMINAMATH_CALUDE_solve_race_problem_l1690_169078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1690_169042

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 1 + a 3 = 2 * a 2) ∧
  (∃ a : ℕ → ℝ, a 1 + a 3 = 2 * a 2 ∧ ¬is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1690_169042


namespace NUMINAMATH_CALUDE_expression_simplification_l1690_169043

theorem expression_simplification :
  ((3 + 4 + 5 + 7) / 3) + ((3 * 6 + 9) / 4) = 157 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1690_169043


namespace NUMINAMATH_CALUDE_double_box_11_l1690_169011

def box_sum (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem double_box_11 : box_sum (box_sum 11) = 28 := by
  sorry

end NUMINAMATH_CALUDE_double_box_11_l1690_169011


namespace NUMINAMATH_CALUDE_sin_105_cos_105_l1690_169090

theorem sin_105_cos_105 :
  Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_105_cos_105_l1690_169090


namespace NUMINAMATH_CALUDE_planes_parallel_if_perp_to_same_line_l1690_169072

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- The main theorem
theorem planes_parallel_if_perp_to_same_line 
  (a : Line) (α β : Plane) 
  (h_diff : α ≠ β) 
  (h_perp_α : perp a α) 
  (h_perp_β : perp a β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perp_to_same_line_l1690_169072


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1690_169096

theorem largest_prime_factors_difference (n : Nat) (h : n = 195195) :
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧
    p ∣ n ∧
    q ∣ n ∧
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
    p - q = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1690_169096


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l1690_169086

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 2 ∧ ∀ x ∈ interval, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l1690_169086


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1690_169037

theorem complex_magnitude_example : Complex.abs (11 + 18 * Complex.I + 4 - 3 * Complex.I) = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1690_169037


namespace NUMINAMATH_CALUDE_min_cost_1001_grid_square_l1690_169062

/-- Represents a grid square with side length n -/
def GridSquare (n : ℕ) := {m : ℕ × ℕ | m.1 ≤ n ∧ m.2 ≤ n}

/-- The cost of coloring a single cell -/
def colorCost : ℕ := 1

/-- The minimum number of cells that need to be colored to create a complete grid square -/
def minColoredCells (n : ℕ) : ℕ := n * n + 2 * n * (n - 1)

theorem min_cost_1001_grid_square :
  minColoredCells 1001 * colorCost = 503000 :=
sorry

end NUMINAMATH_CALUDE_min_cost_1001_grid_square_l1690_169062


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l1690_169031

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  a + 4 * b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l1690_169031


namespace NUMINAMATH_CALUDE_optimal_newspaper_sales_l1690_169070

/-- Represents the newsstand's daily newspaper sales and profit calculation. -/
structure NewspaperSales where
  buyPrice : ℚ
  sellPrice : ℚ
  returnPrice : ℚ
  highDemandDays : ℕ
  lowDemandDays : ℕ
  highDemandAmount : ℕ
  lowDemandAmount : ℕ

/-- Calculates the monthly profit for a given number of daily purchases. -/
def monthlyProfit (sales : NewspaperSales) (dailyPurchase : ℕ) : ℚ :=
  sorry

/-- Theorem stating the optimal daily purchase and maximum monthly profit. -/
theorem optimal_newspaper_sales :
  ∃ (sales : NewspaperSales),
    sales.buyPrice = 24/100 ∧
    sales.sellPrice = 40/100 ∧
    sales.returnPrice = 8/100 ∧
    sales.highDemandDays = 20 ∧
    sales.lowDemandDays = 10 ∧
    sales.highDemandAmount = 300 ∧
    sales.lowDemandAmount = 200 ∧
    (∀ x : ℕ, monthlyProfit sales x ≤ monthlyProfit sales 300) ∧
    monthlyProfit sales 300 = 1120 := by
  sorry

end NUMINAMATH_CALUDE_optimal_newspaper_sales_l1690_169070


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1690_169044

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/2 and common ratio 1/3 -/
theorem geometric_sequence_sum : 
  geometric_sum (1/2) (1/3) 8 = 4920/6561 := by
  sorry

#eval geometric_sum (1/2) (1/3) 8

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1690_169044


namespace NUMINAMATH_CALUDE_simplify_fraction_l1690_169021

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1690_169021


namespace NUMINAMATH_CALUDE_greatest_integer_jo_l1690_169035

theorem greatest_integer_jo (n : ℕ) : 
  n > 0 ∧ 
  n < 150 ∧ 
  ∃ k : ℕ, n + 2 = 9 * k ∧ 
  ∃ l : ℕ, n + 4 = 8 * l →
  n ≤ 146 ∧ 
  ∃ m : ℕ, 146 > 0 ∧ 
  146 < 150 ∧ 
  146 + 2 = 9 * m ∧ 
  ∃ p : ℕ, 146 + 4 = 8 * p :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_jo_l1690_169035


namespace NUMINAMATH_CALUDE_equation_solution_l1690_169059

theorem equation_solution : ∃! x : ℝ, 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1690_169059


namespace NUMINAMATH_CALUDE_garden_trees_l1690_169055

/-- The number of trees in a garden with given specifications -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem stating that the number of trees in the garden is 26 -/
theorem garden_trees : number_of_trees 400 16 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l1690_169055


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l1690_169084

-- Define the function f
def f (x : ℝ) : ℝ := (x + 3)^2 - 7

-- State the theorem
theorem smallest_invertible_domain : 
  ∃ (c : ℝ), c = -3 ∧ 
  (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ∧ 
  (∀ c' < c, ∃ x y, x ≥ c' → y ≥ c' → f x = f y ∧ x ≠ y) :=
sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l1690_169084


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1690_169097

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
if the distance between one of its foci and an asymptote is one-fourth of its focal distance,
then the eccentricity of the hyperbola is 2√3/3.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focal_distance := 2 * c
  let asymptote_distance := b
  focal_distance = 4 * asymptote_distance →
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1690_169097


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1690_169052

theorem modular_arithmetic_problem :
  ∃ (x y : ℤ), 
    (7 * x ≡ 1 [ZMOD 56]) ∧ 
    (13 * y ≡ 1 [ZMOD 56]) ∧ 
    (3 * x + 9 * y ≡ 39 [ZMOD 56]) ∧ 
    (0 ≤ (3 * x + 9 * y) % 56) ∧ 
    ((3 * x + 9 * y) % 56 < 56) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1690_169052


namespace NUMINAMATH_CALUDE_harmonic_geometric_log_ratio_l1690_169016

/-- Given distinct positive real numbers a, b, c forming a harmonic sequence,
    and their logarithms forming a geometric sequence, 
    prove that the common ratio of the geometric sequence is a non-1 cube root of unity. -/
theorem harmonic_geometric_log_ratio 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hharmseq : (1 / b) = (1 / (2 : ℝ)) * ((1 / a) + (1 / c)))
  (hgeomseq : ∃ r : ℂ, (Complex.log b / Complex.log a) = r ∧ 
                       (Complex.log c / Complex.log b) = r ∧ 
                       (Complex.log a / Complex.log c) = r) :
  ∃ r : ℂ, r^3 = 1 ∧ r ≠ 1 ∧ 
    ((Complex.log b / Complex.log a) = r ∧ 
     (Complex.log c / Complex.log b) = r ∧ 
     (Complex.log a / Complex.log c) = r) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_geometric_log_ratio_l1690_169016


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1690_169019

/-- Represents a soccer team with 11 players -/
def SoccerTeam := Fin 11

/-- The number of ways to arrange players from two soccer teams in a line
    such that no two adjacent players are from the same team -/
def valid_arrangements : ℕ :=
  2 * (Nat.factorial 11) ^ 2

/-- Theorem stating that the number of valid arrangements is correct -/
theorem valid_arrangements_count :
  valid_arrangements = 2 * (Nat.factorial 11) ^ 2 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1690_169019


namespace NUMINAMATH_CALUDE_prime_pythagorean_inequality_l1690_169025

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (hp : Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
sorry

end NUMINAMATH_CALUDE_prime_pythagorean_inequality_l1690_169025


namespace NUMINAMATH_CALUDE_right_triangle_test_l1690_169039

theorem right_triangle_test : 
  -- Option A
  (3 : ℝ)^2 + 4^2 = 5^2 ∧
  -- Option B
  (1 : ℝ)^2 + 2^2 = (Real.sqrt 5)^2 ∧
  -- Option C
  (2 : ℝ)^2 + (2 * Real.sqrt 3)^2 ≠ 3^2 ∧
  -- Option D
  (1 : ℝ)^2 + (Real.sqrt 3)^2 = 2^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_test_l1690_169039


namespace NUMINAMATH_CALUDE_f_order_l1690_169049

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

-- State the theorem
theorem f_order (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
  sorry

end NUMINAMATH_CALUDE_f_order_l1690_169049


namespace NUMINAMATH_CALUDE_equation_solution_l1690_169067

theorem equation_solution (b : ℝ) (hb : b ≠ 0) :
  (0 : ℝ)^2 + 9*b^2 = (3*b - 0)^2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1690_169067


namespace NUMINAMATH_CALUDE_square_of_sum_l1690_169048

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l1690_169048


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1690_169034

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  (∃ (q₁ : ℕ), 4351 = q₁ * n + 8) ∧
  (∃ (q₂ : ℕ), 5161 = q₂ * n + 10) ∧
  (∃ (q₃ : ℕ), 6272 = q₃ * n + 12) ∧
  (∃ (q₄ : ℕ), 7383 = q₄ * n + 14) ∧
  ∀ (m : ℕ), m > 0 →
    (∃ (r₁ : ℕ), 4351 = r₁ * m + 8) →
    (∃ (r₂ : ℕ), 5161 = r₂ * m + 10) →
    (∃ (r₃ : ℕ), 6272 = r₃ * m + 12) →
    (∃ (r₄ : ℕ), 7383 = r₄ * m + 14) →
    m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1690_169034


namespace NUMINAMATH_CALUDE_equal_bisecting_diagonals_implies_rectangle_bisecting_diagonals_implies_parallelogram_rhombus_equal_diagonals_implies_square_l1690_169002

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem equal_bisecting_diagonals_implies_rectangle 
  (q : Quadrilateral) 
  (h1 : has_equal_diagonals q) 
  (h2 : diagonals_bisect_each_other q) : 
  is_rectangle q := by sorry

-- Theorem 2
theorem bisecting_diagonals_implies_parallelogram 
  (q : Quadrilateral) 
  (h : diagonals_bisect_each_other q) : 
  is_parallelogram q := by sorry

-- Theorem 3
theorem rhombus_equal_diagonals_implies_square 
  (q : Quadrilateral) 
  (h1 : is_rhombus q) 
  (h2 : has_equal_diagonals q) : 
  is_square q := by sorry

end NUMINAMATH_CALUDE_equal_bisecting_diagonals_implies_rectangle_bisecting_diagonals_implies_parallelogram_rhombus_equal_diagonals_implies_square_l1690_169002


namespace NUMINAMATH_CALUDE_inequality_theorem_l1690_169089

theorem inequality_theorem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hmn : m > n) (hn2 : n ≥ 2) :
  ((1 + 1 / m : ℝ) ^ m > (1 + 1 / n : ℝ) ^ n) ∧
  ((1 + 1 / m : ℝ) ^ (m + 1) < (1 + 1 / n : ℝ) ^ (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1690_169089


namespace NUMINAMATH_CALUDE_ceiling_times_self_182_l1690_169082

theorem ceiling_times_self_182 :
  ∃! (x : ℝ), ⌈x⌉ * x = 182 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ceiling_times_self_182_l1690_169082


namespace NUMINAMATH_CALUDE_power_of_three_l1690_169058

theorem power_of_three (a b : ℕ+) (h : 3^(a : ℕ) * 3^(b : ℕ) = 81) :
  (3^(a : ℕ))^(b : ℕ) = 3^4 := by
sorry

end NUMINAMATH_CALUDE_power_of_three_l1690_169058


namespace NUMINAMATH_CALUDE_leftover_value_is_correct_l1690_169060

/-- Calculate the value of leftover coins after combining and rolling --/
def leftover_value (james_quarters james_dimes emily_quarters emily_dimes : ℕ)
  (quarters_per_roll dimes_per_roll : ℕ) : ℚ :=
  let total_quarters := james_quarters + emily_quarters
  let total_dimes := james_dimes + emily_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

/-- The main theorem --/
theorem leftover_value_is_correct :
  leftover_value 65 134 103 229 40 50 = 33 / 10 :=
by sorry

end NUMINAMATH_CALUDE_leftover_value_is_correct_l1690_169060


namespace NUMINAMATH_CALUDE_simon_lego_count_l1690_169004

theorem simon_lego_count (kent_legos : ℕ) (bruce_extra : ℕ) (simon_percentage : ℚ) :
  kent_legos = 40 →
  bruce_extra = 20 →
  simon_percentage = 1/5 →
  (kent_legos + bruce_extra) * (1 + simon_percentage) = 72 :=
by sorry

end NUMINAMATH_CALUDE_simon_lego_count_l1690_169004


namespace NUMINAMATH_CALUDE_fraction_difference_simplification_l1690_169007

theorem fraction_difference_simplification : 
  ∃ q : ℕ+, (2022 : ℚ) / 2021 - 2021 / 2022 = (4043 : ℚ) / q ∧ Nat.gcd 4043 q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_simplification_l1690_169007


namespace NUMINAMATH_CALUDE_marble_problem_l1690_169006

theorem marble_problem (x : ℝ) : 
  x + 3*x + 6*x + 24*x = 156 → x = 156/34 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1690_169006


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l1690_169064

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem largest_digit_divisible_by_6 :
  ∀ N : ℕ, N ≤ 9 →
    (is_divisible_by_6 (2345 * 10 + N) → N ≤ 4) ∧
    (is_divisible_by_6 (2345 * 10 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l1690_169064


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1690_169033

/-- A linear equation in two variables x and y of the form mx + 3y = 4x - 1 -/
def linear_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x + 3 * y = 4 * x - 1

/-- The condition for the equation to be linear in two variables -/
def is_linear_in_two_variables (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y, linear_equation m x y ↔ a * x + b * y = c

theorem linear_equation_condition (m : ℝ) :
  is_linear_in_two_variables m ↔ m ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1690_169033


namespace NUMINAMATH_CALUDE_ellipse_equation_and_max_slope_product_l1690_169027

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a line in the form x = my + 1 -/
structure Line where
  m : ℝ

/-- Calculates the product of slopes of the three sides of a triangle formed by
    two points on an ellipse and a fixed point -/
def slope_product (e : Ellipse) (l : Line) (p : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_equation_and_max_slope_product 
  (e : Ellipse) (p : ℝ × ℝ) (h_p_on_ellipse : p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1)
  (h_p : p = (1, 3/2)) (h_eccentricity : Real.sqrt (e.a^2 - e.b^2) / e.a = 1/2) :
  (∃ (e' : Ellipse), e'.a = 2 ∧ e'.b = Real.sqrt 3 ∧
    (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / e'.a^2 + y^2 / e'.b^2 = 1)) ∧
  (∃ (max_t : ℝ), max_t = 9/64 ∧
    ∀ (l : Line), slope_product e' l p ≤ max_t) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_max_slope_product_l1690_169027


namespace NUMINAMATH_CALUDE_cafeteria_bill_theorem_l1690_169012

/-- The total cost of a cafeteria order for three people -/
def cafeteria_cost (coffee_price ice_cream_price cake_price : ℕ) : ℕ :=
  let mell_order := 2 * coffee_price + cake_price
  let friend_order := 2 * coffee_price + cake_price + ice_cream_price
  mell_order + 2 * friend_order

/-- Theorem stating the total cost for Mell and her friends' cafeteria order -/
theorem cafeteria_bill_theorem :
  cafeteria_cost 4 3 7 = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_bill_theorem_l1690_169012


namespace NUMINAMATH_CALUDE_computer_price_theorem_l1690_169068

/-- The sticker price of the computer -/
def sticker_price : ℝ := 750

/-- The price at store A after discount and rebate -/
def price_A (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at store B after discount -/
def price_B (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem computer_price_theorem :
  price_A sticker_price = price_B sticker_price - 25 :=
by sorry

end NUMINAMATH_CALUDE_computer_price_theorem_l1690_169068


namespace NUMINAMATH_CALUDE_b_initial_investment_l1690_169028

/-- Represents the business investment scenario --/
structure BusinessInvestment where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment
  initial_duration : ℕ  -- Initial duration in months
  a_withdrawal : ℕ  -- Amount A withdraws after initial duration
  b_addition : ℕ  -- Amount B adds after initial duration
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit_share : ℕ  -- A's share of the profit

/-- Calculates the total investment of A --/
def total_investment_a (bi : BusinessInvestment) : ℕ :=
  bi.a_initial * bi.initial_duration + (bi.a_initial - bi.a_withdrawal) * (12 - bi.initial_duration)

/-- Calculates the total investment of B --/
def total_investment_b (bi : BusinessInvestment) : ℕ :=
  bi.b_initial * bi.initial_duration + (bi.b_initial + bi.b_addition) * (12 - bi.initial_duration)

/-- Theorem stating that given the conditions, B's initial investment was 4000 Rs --/
theorem b_initial_investment (bi : BusinessInvestment) 
  (h1 : bi.a_initial = 3000)
  (h2 : bi.initial_duration = 8)
  (h3 : bi.a_withdrawal = 1000)
  (h4 : bi.b_addition = 1000)
  (h5 : bi.total_profit = 840)
  (h6 : bi.a_profit_share = 320)
  : bi.b_initial = 4000 := by
  sorry

end NUMINAMATH_CALUDE_b_initial_investment_l1690_169028


namespace NUMINAMATH_CALUDE_luke_new_cards_l1690_169098

/-- The number of new baseball cards Luke had --/
def new_cards (cards_per_page old_cards total_pages : ℕ) : ℕ :=
  cards_per_page * total_pages - old_cards

/-- Theorem stating that Luke had 3 new cards --/
theorem luke_new_cards : new_cards 3 9 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_luke_new_cards_l1690_169098


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l1690_169065

theorem soda_bottle_difference (regular_soda : ℕ) (diet_soda : ℕ)
  (h1 : regular_soda = 60)
  (h2 : diet_soda = 19) :
  regular_soda - diet_soda = 41 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l1690_169065


namespace NUMINAMATH_CALUDE_log_inequality_condition_l1690_169026

theorem log_inequality_condition (a b : ℝ) : 
  (∀ a b, Real.log a > Real.log b → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l1690_169026


namespace NUMINAMATH_CALUDE_daisy_spending_difference_l1690_169024

def breakfast_muffin1_price : ℚ := 2
def breakfast_muffin2_price : ℚ := 3
def breakfast_coffee1_price : ℚ := 4
def breakfast_coffee2_discount : ℚ := 0.5
def lunch_soup_price : ℚ := 3.75
def lunch_salad_price : ℚ := 5.75
def lunch_lemonade_price : ℚ := 1
def lunch_service_charge_percent : ℚ := 10

def breakfast_total : ℚ := breakfast_muffin1_price + breakfast_muffin2_price + breakfast_coffee1_price + (breakfast_coffee1_price - breakfast_coffee2_discount)

def lunch_subtotal : ℚ := lunch_soup_price + lunch_salad_price + lunch_lemonade_price

def lunch_total : ℚ := lunch_subtotal + (lunch_subtotal * lunch_service_charge_percent / 100)

theorem daisy_spending_difference : lunch_total - breakfast_total = -0.95 := by
  sorry

end NUMINAMATH_CALUDE_daisy_spending_difference_l1690_169024


namespace NUMINAMATH_CALUDE_min_value_theorem_l1690_169066

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1690_169066


namespace NUMINAMATH_CALUDE_derivative_of_even_function_l1690_169003

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f is even
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem derivative_of_even_function (hf : IsEven f) :
  ∀ x, (deriv f) (-x) = -(deriv f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_l1690_169003


namespace NUMINAMATH_CALUDE_opposite_of_three_l1690_169029

theorem opposite_of_three : (-(3 : ℤ)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1690_169029


namespace NUMINAMATH_CALUDE_cube_difference_positive_l1690_169001

theorem cube_difference_positive (a b : ℝ) : a > b → a^3 - b^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_positive_l1690_169001


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l1690_169093

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the initial number of cookies -/
def initialCookies : ℕ := 60

/-- Represents the initial number of brownies -/
def initialBrownies : ℕ := 10

/-- Represents the number of cookies eaten per day -/
def cookiesPerDay : ℕ := 3

/-- Represents the number of brownies eaten per day -/
def browniesPerDay : ℕ := 1

/-- Calculates the remaining cookies after a week -/
def remainingCookies : ℕ := initialCookies - daysInWeek * cookiesPerDay

/-- Calculates the remaining brownies after a week -/
def remainingBrownies : ℕ := initialBrownies - daysInWeek * browniesPerDay

/-- Theorem stating the difference between remaining cookies and brownies after a week -/
theorem cookie_brownie_difference :
  remainingCookies - remainingBrownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l1690_169093


namespace NUMINAMATH_CALUDE_carlton_outfit_combinations_l1690_169091

theorem carlton_outfit_combinations 
  (button_up_shirts : ℕ) 
  (sweater_vests : ℕ) 
  (ties : ℕ) 
  (h1 : button_up_shirts = 4)
  (h2 : sweater_vests = 3 * button_up_shirts)
  (h3 : ties = 2 * sweater_vests) : 
  button_up_shirts * sweater_vests * ties = 1152 := by
  sorry

end NUMINAMATH_CALUDE_carlton_outfit_combinations_l1690_169091


namespace NUMINAMATH_CALUDE_min_production_time_l1690_169057

/-- Represents the production process of ceramic items -/
structure CeramicProduction where
  shapingTime : ℕ := 15
  dryingTime : ℕ := 10
  firingTime : ℕ := 30
  totalItems : ℕ := 75
  totalWorkers : ℕ := 13

/-- Calculates the production time for a given stage -/
def stageTime (itemsPerWorker : ℕ) (timePerItem : ℕ) : ℕ :=
  (itemsPerWorker + 1) * timePerItem

/-- Theorem stating the minimum production time for ceramic items -/
theorem min_production_time (prod : CeramicProduction) :
  ∃ (shapers firers : ℕ),
    shapers + firers = prod.totalWorkers ∧
    (∀ (s f : ℕ),
      s + f = prod.totalWorkers →
      max (stageTime (prod.totalItems / s) prod.shapingTime)
          (stageTime (prod.totalItems / f) prod.firingTime)
      ≥ 325) :=
sorry

end NUMINAMATH_CALUDE_min_production_time_l1690_169057


namespace NUMINAMATH_CALUDE_coin_flips_l1690_169038

/-- The number of times a coin is flipped -/
def n : ℕ := sorry

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads on the first 4 flips and not heads on the last flip -/
def p_event : ℚ := 1/32

theorem coin_flips : 
  p_heads = 1/2 → 
  p_event = (p_heads ^ 4) * ((1 - p_heads) ^ 1) * (p_heads ^ (n - 5)) → 
  n = 9 := by sorry

end NUMINAMATH_CALUDE_coin_flips_l1690_169038


namespace NUMINAMATH_CALUDE_equilibrium_force_l1690_169056

/-- Given three forces in a 2D plane, prove that a specific fourth force is required for equilibrium. -/
theorem equilibrium_force (f₁ f₂ f₃ f₄ : ℝ × ℝ) : 
  f₁ = (-2, -1) → f₂ = (-3, 2) → f₃ = (4, -3) →
  (f₁.1 + f₂.1 + f₃.1 + f₄.1 = 0 ∧ f₁.2 + f₂.2 + f₃.2 + f₄.2 = 0) →
  f₄ = (1, 2) := by
sorry

end NUMINAMATH_CALUDE_equilibrium_force_l1690_169056


namespace NUMINAMATH_CALUDE_g_range_l1690_169083

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 2)

theorem g_range :
  Set.range g = {y : ℝ | y ≠ -21} := by sorry

end NUMINAMATH_CALUDE_g_range_l1690_169083


namespace NUMINAMATH_CALUDE_expression_evaluation_l1690_169050

theorem expression_evaluation :
  let sin30 : Real := 1/2
  4 * (Real.sqrt 3 + Real.sqrt 7) / (5 * Real.sqrt (3 + sin30)) = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1690_169050


namespace NUMINAMATH_CALUDE_minimum_h_10_l1690_169095

def IsTenuous (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y > 2 * y.val ^ 2

def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_h_10 (h : ℕ+ → ℤ) 
  (tenuous : IsTenuous h) 
  (min_sum : ∀ g : ℕ+ → ℤ, IsTenuous g → SumH g ≥ SumH h) : 
  h ⟨10, Nat.succ_pos 9⟩ ≥ 137 := by
  sorry

end NUMINAMATH_CALUDE_minimum_h_10_l1690_169095


namespace NUMINAMATH_CALUDE_remainder_problem_l1690_169074

theorem remainder_problem :
  {x : ℕ | x < 100 ∧ x % 7 = 3 ∧ x % 9 = 4} = {31, 94} := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1690_169074


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1690_169071

theorem complex_fraction_simplification :
  (Complex.I : ℂ) / (1 - Complex.I) = -1/2 + Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1690_169071


namespace NUMINAMATH_CALUDE_parabola_focus_l1690_169046

theorem parabola_focus (p : ℝ) :
  4 * p = 1/4 → (0, 1/(16 : ℝ)) = (0, p) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1690_169046


namespace NUMINAMATH_CALUDE_cosine_inequality_l1690_169054

theorem cosine_inequality (θ : ℝ) : 5 + 8 * Real.cos θ + 4 * Real.cos (2 * θ) + Real.cos (3 * θ) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l1690_169054


namespace NUMINAMATH_CALUDE_max_distance_ellipse_circle_l1690_169040

/-- The maximum distance between a point on the ellipse x²/9 + y² = 1
    and a point on the circle (x-4)² + y² = 1 is 8 -/
theorem max_distance_ellipse_circle : 
  ∃ (max_dist : ℝ),
    max_dist = 8 ∧
    ∀ (P Q : ℝ × ℝ),
      (P.1^2 / 9 + P.2^2 = 1) →
      ((Q.1 - 4)^2 + Q.2^2 = 1) →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_circle_l1690_169040


namespace NUMINAMATH_CALUDE_smallest_constant_for_inequality_l1690_169020

/-- The smallest possible real constant C such that 
    |x^3 + y^3 + z^3 + 1| ≤ C|x^5 + y^5 + z^5 + 1| 
    holds for all real x, y, z satisfying x + y + z = -1 -/
theorem smallest_constant_for_inequality :
  ∃ (C : ℝ), C = 1539 / 1449 ∧
  (∀ (x y z : ℝ), x + y + z = -1 →
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1|) ∧
  (∀ (C' : ℝ), C' < C →
    ∃ (x y z : ℝ), x + y + z = -1 ∧
      |x^3 + y^3 + z^3 + 1| > C' * |x^5 + y^5 + z^5 + 1|) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_for_inequality_l1690_169020


namespace NUMINAMATH_CALUDE_part_1_part_2_part_3_l1690_169036

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Part 1
theorem part_1 (a : ℝ) (h : circle_C a (a+1)) : 
  let P : ℝ × ℝ := (a, a+1)
  ∃ (d s : ℝ), d = Real.sqrt 40 ∧ s = 1/3 ∧ 
    d^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ∧
    s = (P.2 - Q.2) / (P.1 - Q.1) := by sorry

-- Part 2
theorem part_2 (M : ℝ × ℝ) (h : circle_C M.1 M.2) :
  2 * Real.sqrt 2 ≤ Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ∧
  Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ 6 * Real.sqrt 2 := by sorry

-- Part 3
theorem part_3 (m n : ℝ) (h : m^2 + n^2 - 4*m - 14*n + 45 = 0) :
  2 - Real.sqrt 3 ≤ (n - 3) / (m + 2) ∧ (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_part_3_l1690_169036


namespace NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l1690_169009

/-- Price of a single rose in yuan -/
def rose_price : ℝ := sorry

/-- Price of a single carnation in yuan -/
def carnation_price : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is greater than 24 yuan -/
axiom condition1 : 6 * rose_price + 3 * carnation_price > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * rose_price + 5 * carnation_price < 22

/-- Theorem: The price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations :
  2 * rose_price > 3 * carnation_price := by sorry

end NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l1690_169009


namespace NUMINAMATH_CALUDE_negation_of_p_is_existential_l1690_169073

-- Define the set of even numbers
def A : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 2 * k}

-- Define the proposition p
def p : Prop := ∀ x : ℤ, (2 * x) ∈ A

-- Theorem statement
theorem negation_of_p_is_existential :
  ¬p ↔ ∃ x : ℤ, (2 * x) ∉ A := by sorry

end NUMINAMATH_CALUDE_negation_of_p_is_existential_l1690_169073


namespace NUMINAMATH_CALUDE_cone_base_radius_l1690_169079

/-- Given a circle of radius 16 divided into 4 equal parts, if one part forms the lateral surface of a cone, then the radius of the cone's base is 4. -/
theorem cone_base_radius (r : ℝ) (h1 : r = 16) (h2 : r > 0) : 
  (2 * Real.pi * r) / 4 = 2 * Real.pi * 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1690_169079


namespace NUMINAMATH_CALUDE_remainder_theorem_l1690_169051

theorem remainder_theorem (n : ℤ) (h : n % 13 = 3) : (5 * n - 11) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1690_169051


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l1690_169047

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day n days after the given day -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

/-- The main theorem -/
theorem tomorrow_is_saturday 
  (h : addDays (addDays DayOfWeek.Wednesday 2) 5 = DayOfWeek.Monday) : 
  nextDay (addDays DayOfWeek.Wednesday 2) = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_tomorrow_is_saturday_l1690_169047


namespace NUMINAMATH_CALUDE_spelling_contest_total_questions_l1690_169092

/-- Represents a participant in the spelling contest -/
structure Participant where
  name : String
  round1_correct : ℕ
  round1_wrong : ℕ
  round2_correct : ℕ
  round2_wrong : ℕ
  round3_correct : ℕ
  round3_wrong : ℕ

/-- Calculates the total number of questions for a participant -/
def totalQuestions (p : Participant) : ℕ :=
  p.round1_correct + p.round1_wrong +
  p.round2_correct + p.round2_wrong +
  p.round3_correct + p.round3_wrong

/-- The spelling contest -/
def spellingContest : Prop :=
  let drew : Participant := {
    name := "Drew"
    round1_correct := 20
    round1_wrong := 6
    round2_correct := 24
    round2_wrong := 9
    round3_correct := 28
    round3_wrong := 14
  }
  let carla : Participant := {
    name := "Carla"
    round1_correct := 14
    round1_wrong := 2 * drew.round1_wrong
    round2_correct := 21
    round2_wrong := 8
    round3_correct := 22
    round3_wrong := 10
  }
  let blake : Participant := {
    name := "Blake"
    round1_correct := 0
    round1_wrong := 0
    round2_correct := 18
    round2_wrong := 11
    round3_correct := 15
    round3_wrong := 16
  }
  
  -- Conditions
  (∀ p : Participant, (p.round1_correct : ℚ) / (p.round1_correct + p.round1_wrong) ≥ 0.7) ∧
  (∀ p : Participant, (p.round2_correct : ℚ) / (p.round2_correct + p.round2_wrong) ≥ 0.7) ∧
  (∀ p : Participant, (p.round3_correct : ℚ) / (p.round3_correct + p.round3_wrong) ≥ 0.7) ∧
  (∀ p : Participant, ((p.round1_correct + p.round2_correct) : ℚ) / (p.round1_correct + p.round1_wrong + p.round2_correct + p.round2_wrong) ≥ 0.75) ∧
  
  -- Theorem to prove
  (totalQuestions drew + totalQuestions carla + totalQuestions blake = 248)

theorem spelling_contest_total_questions : spellingContest := by sorry

end NUMINAMATH_CALUDE_spelling_contest_total_questions_l1690_169092


namespace NUMINAMATH_CALUDE_same_color_probability_l1690_169000

/-- The probability of drawing two balls of the same color from a bag containing
    8 green balls, 5 red balls, and 7 blue balls, with replacement. -/
theorem same_color_probability :
  let total_balls : ℕ := 8 + 5 + 7
  let p_green : ℚ := 8 / total_balls
  let p_red : ℚ := 5 / total_balls
  let p_blue : ℚ := 7 / total_balls
  let p_same_color : ℚ := p_green ^ 2 + p_red ^ 2 + p_blue ^ 2
  p_same_color = 117 / 200 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1690_169000


namespace NUMINAMATH_CALUDE_work_completion_time_l1690_169032

/-- Represents the time taken to complete a work when one worker is assisted by two others on alternate days -/
def time_to_complete (time_a time_b time_c : ℝ) : ℝ :=
  2 * 4

/-- Theorem stating that if A can do a work in 11 days, B in 20 days, and C in 55 days,
    and A is assisted by B and C on alternate days, then the work can be completed in 8 days -/
theorem work_completion_time (time_a time_b time_c : ℝ)
  (ha : time_a = 11)
  (hb : time_b = 20)
  (hc : time_c = 55) :
  time_to_complete time_a time_b time_c = 8 := by
  sorry

#eval time_to_complete 11 20 55

end NUMINAMATH_CALUDE_work_completion_time_l1690_169032


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1690_169015

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a + b ≠ 0) 
  (h2 : a - 2*b ≠ 0) 
  (h3 : a^2 - b^2 ≠ 0) 
  (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1690_169015


namespace NUMINAMATH_CALUDE_intersects_x_axis_once_l1690_169081

/-- A function f(x) = (k-3)x^2 + 2x + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- The condition for a quadratic function to have exactly one root -/
def has_one_root (k : ℝ) : Prop :=
  (k = 3) ∨ (4 * (k - 3) * 1 = 2^2)

theorem intersects_x_axis_once (k : ℝ) :
  (∃! x, f k x = 0) ↔ has_one_root k := by sorry

end NUMINAMATH_CALUDE_intersects_x_axis_once_l1690_169081


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1690_169030

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1690_169030


namespace NUMINAMATH_CALUDE_adams_friends_strawberries_l1690_169069

/-- The number of strawberries Adam's friends ate -/
def friends_strawberries (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Adam's friends ate 2 strawberries -/
theorem adams_friends_strawberries :
  friends_strawberries 35 33 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adams_friends_strawberries_l1690_169069


namespace NUMINAMATH_CALUDE_garden_fencing_theorem_l1690_169080

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

theorem garden_fencing_theorem :
  ∀ (garden : RectangularGarden),
    garden.length = 50 →
    garden.length = 2 * garden.width →
    perimeter garden = 150 := by
  sorry

end NUMINAMATH_CALUDE_garden_fencing_theorem_l1690_169080


namespace NUMINAMATH_CALUDE_binomial_sum_27_mod_9_l1690_169023

def binomial_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => Nat.choose n k)

theorem binomial_sum_27_mod_9 :
  (binomial_sum 27 - Nat.choose 27 0) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_27_mod_9_l1690_169023


namespace NUMINAMATH_CALUDE_laptop_price_proof_l1690_169053

theorem laptop_price_proof (original_price : ℝ) : 
  (0.7 * original_price - (0.8 * original_price - 70) = 20) → 
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l1690_169053


namespace NUMINAMATH_CALUDE_mode_best_for_market_share_l1690_169022

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Mean
  | Median
  | Mode
  | Variance

/-- Represents a shoe factory -/
structure ShoeFactory where
  survey_data : List Nat  -- List of shoe sizes from the survey

/-- Determines the most appropriate statistical measure for increasing market share -/
def best_measure_for_market_share (factory : ShoeFactory) : StatisticalMeasure :=
  StatisticalMeasure.Mode

/-- Theorem stating that the mode is the most appropriate measure for increasing market share -/
theorem mode_best_for_market_share (factory : ShoeFactory) :
  best_measure_for_market_share factory = StatisticalMeasure.Mode := by
  sorry


end NUMINAMATH_CALUDE_mode_best_for_market_share_l1690_169022


namespace NUMINAMATH_CALUDE_same_color_probability_l1690_169018

/-- Represents the number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- Represents the total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- Represents the number of shoes to select -/
def shoes_to_select : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability : 
  (num_pairs : ℚ) / (total_shoes.choose shoes_to_select) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1690_169018


namespace NUMINAMATH_CALUDE_soccer_score_theorem_l1690_169017

/-- Represents the scores of a soccer player -/
structure SoccerScores where
  game6 : ℕ
  game7 : ℕ
  game8 : ℕ
  game9 : ℕ
  first6GamesTotal : ℕ
  game10 : ℕ

/-- The minimum number of points scored in the 10th game -/
def minGame10Score (s : SoccerScores) : Prop :=
  s.game10 = 13

/-- The given conditions of the problem -/
def problemConditions (s : SoccerScores) : Prop :=
  s.game6 = 18 ∧
  s.game7 = 25 ∧
  s.game8 = 15 ∧
  s.game9 = 22 ∧
  (s.first6GamesTotal + s.game6 + s.game7 + s.game8 + s.game9 + s.game10) / 10 >
    (s.first6GamesTotal + s.game6) / 6 ∧
  (s.first6GamesTotal + s.game6 + s.game7 + s.game8 + s.game9 + s.game10) / 10 > 20

theorem soccer_score_theorem (s : SoccerScores) :
  problemConditions s → minGame10Score s := by
  sorry

end NUMINAMATH_CALUDE_soccer_score_theorem_l1690_169017


namespace NUMINAMATH_CALUDE_same_solution_implies_m_equals_9_l1690_169075

theorem same_solution_implies_m_equals_9 :
  ∀ y m : ℝ,
  (y + 3 * m = 32) ∧ (y - 4 = 1) →
  m = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_equals_9_l1690_169075


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1690_169010

/-- Given a point P in 3D space, this function returns the point symmetric to P with respect to the y-axis --/
def symmetry_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, -z)

/-- Theorem stating that the point symmetric to (2, -3, -5) with respect to the y-axis is (-2, -3, 5) --/
theorem symmetry_of_point :
  symmetry_y_axis (2, -3, -5) = (-2, -3, 5) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1690_169010


namespace NUMINAMATH_CALUDE_buddy_cards_bought_l1690_169088

/-- Represents the number of baseball cards Buddy has on each day --/
structure BuddyCards where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Represents the number of cards Buddy bought on Wednesday and Thursday --/
structure CardsBought where
  wednesday : ℕ
  thursday : ℕ

/-- The theorem statement --/
theorem buddy_cards_bought (cards : BuddyCards) (bought : CardsBought) : 
  cards.monday = 30 →
  cards.tuesday = cards.monday / 2 →
  cards.thursday = 32 →
  bought.thursday = cards.tuesday / 3 →
  cards.wednesday = cards.tuesday + bought.wednesday →
  cards.thursday = cards.wednesday + bought.thursday →
  bought.wednesday = 12 := by
  sorry


end NUMINAMATH_CALUDE_buddy_cards_bought_l1690_169088


namespace NUMINAMATH_CALUDE_satellite_sensor_ratio_l1690_169087

theorem satellite_sensor_ratio :
  ∀ (S : ℝ) (N : ℝ),
    S > 0 →
    N > 0 →
    S = 0.2 * S + 24 * N →
    N / (0.2 * S) = 1 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_satellite_sensor_ratio_l1690_169087


namespace NUMINAMATH_CALUDE_unique_rational_root_l1690_169077

/-- The polynomial function we're examining -/
def f (x : ℚ) : ℚ := 3 * x^4 - 4 * x^3 - 10 * x^2 + 6 * x + 3

/-- A rational number is a root of f if f(x) = 0 -/
def is_root (x : ℚ) : Prop := f x = 0

/-- The statement that 1/3 is the only rational root of f -/
theorem unique_rational_root : 
  (is_root (1/3)) ∧ (∀ x : ℚ, is_root x → x = 1/3) := by sorry

end NUMINAMATH_CALUDE_unique_rational_root_l1690_169077


namespace NUMINAMATH_CALUDE_min_triangle_area_l1690_169094

open Complex

-- Define the equation solutions
def solutions : Set ℂ := {z : ℂ | (z - 3)^10 = 32}

-- Define the property that solutions form a regular decagon
def is_regular_decagon (s : Set ℂ) : Prop := sorry

-- Define the function to calculate the area of a triangle given three complex points
def triangle_area (a b c : ℂ) : ℝ := sorry

-- Theorem statement
theorem min_triangle_area (h : is_regular_decagon solutions) :
  ∃ (a b c : ℂ), a ∈ solutions ∧ b ∈ solutions ∧ c ∈ solutions ∧
  (∀ (x y z : ℂ), x ∈ solutions → y ∈ solutions → z ∈ solutions →
    triangle_area a b c ≤ triangle_area x y z) ∧
  triangle_area a b c = 2 * Real.sin (18 * π / 180) * Real.sin (36 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l1690_169094


namespace NUMINAMATH_CALUDE_equation_solution_l1690_169014

theorem equation_solution : 
  ∀ x : ℝ, (3*x + 2)*(x + 3) = x + 3 ↔ x = -3 ∨ x = -1/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1690_169014


namespace NUMINAMATH_CALUDE_parallelogram_iff_opposite_sides_equal_l1690_169041

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def opposite_sides_equal (q : Quadrilateral) : Prop :=
  (q.vertices 0 = q.vertices 2) ∧ (q.vertices 1 = q.vertices 3)

def is_parallelogram (q : Quadrilateral) : Prop :=
  ∃ (v : ℝ × ℝ), (q.vertices 1 - q.vertices 0 = v) ∧ (q.vertices 2 - q.vertices 3 = v) ∧
                 (q.vertices 3 - q.vertices 0 = v) ∧ (q.vertices 2 - q.vertices 1 = v)

theorem parallelogram_iff_opposite_sides_equal (q : Quadrilateral) :
  is_parallelogram q ↔ opposite_sides_equal q := by sorry

end NUMINAMATH_CALUDE_parallelogram_iff_opposite_sides_equal_l1690_169041


namespace NUMINAMATH_CALUDE_vinegar_left_is_60_l1690_169085

/-- Represents the pickle-making scenario with given supplies and constraints. -/
structure PickleSupplies where
  jars : ℕ
  cucumbers : ℕ
  vinegar : ℕ
  pickles_per_cucumber : ℕ
  pickles_per_jar : ℕ
  vinegar_per_jar : ℕ

/-- Calculates the amount of vinegar left after making pickles. -/
def vinegar_left (supplies : PickleSupplies) : ℕ :=
  let jar_capacity := supplies.jars * supplies.pickles_per_jar
  let cucumber_capacity := supplies.cucumbers * supplies.pickles_per_cucumber
  let vinegar_capacity := supplies.vinegar / supplies.vinegar_per_jar * supplies.pickles_per_jar
  let pickles_made := min jar_capacity (min cucumber_capacity vinegar_capacity)
  let jars_used := (pickles_made + supplies.pickles_per_jar - 1) / supplies.pickles_per_jar
  supplies.vinegar - jars_used * supplies.vinegar_per_jar

/-- Theorem stating that given the specific supplies and constraints, 60 ounces of vinegar are left. -/
theorem vinegar_left_is_60 :
  vinegar_left {
    jars := 4,
    cucumbers := 10,
    vinegar := 100,
    pickles_per_cucumber := 6,
    pickles_per_jar := 12,
    vinegar_per_jar := 10
  } = 60 := by
  sorry

end NUMINAMATH_CALUDE_vinegar_left_is_60_l1690_169085


namespace NUMINAMATH_CALUDE_math_expressions_equality_l1690_169063

theorem math_expressions_equality : 
  (∃ (a b c d : ℝ), 
    a = (Real.sqrt 5 - (Real.sqrt 3 + Real.sqrt 15) / (Real.sqrt 6 * Real.sqrt 2)) ∧
    b = ((Real.sqrt 48 - 4 * Real.sqrt (1/8)) - (3 * Real.sqrt (1/3) - 2 * Real.sqrt 0.5)) ∧
    c = ((3 + Real.sqrt 5) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2) ∧
    d = ((- Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5)) ∧
    a = -1 ∧
    b = 3 * Real.sqrt 3 ∧
    c = 2 * Real.sqrt 3 ∧
    d = -3 - Real.sqrt 5) := by
  sorry

#check math_expressions_equality

end NUMINAMATH_CALUDE_math_expressions_equality_l1690_169063


namespace NUMINAMATH_CALUDE_sequence_286_ends_l1690_169076

/-- A sequence of seven positive integers where each number differs by one from its neighbors -/
def ValidSequence (a b c d e f g : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧
  (b = a + 1 ∨ b = a - 1) ∧
  (c = b + 1 ∨ c = b - 1) ∧
  (d = c + 1 ∨ d = c - 1) ∧
  (e = d + 1 ∨ e = d - 1) ∧
  (f = e + 1 ∨ f = e - 1) ∧
  (g = f + 1 ∨ g = f - 1)

theorem sequence_286_ends (a b c d e f g : ℕ) :
  ValidSequence a b c d e f g →
  a + b + c + d + e + f + g = 2017 →
  (a = 286 ∨ g = 286) ∧ b ≠ 286 ∧ c ≠ 286 ∧ d ≠ 286 ∧ e ≠ 286 ∧ f ≠ 286 :=
by sorry

end NUMINAMATH_CALUDE_sequence_286_ends_l1690_169076


namespace NUMINAMATH_CALUDE_remainder_17_pow_53_mod_7_l1690_169013

theorem remainder_17_pow_53_mod_7 : 17^53 % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_53_mod_7_l1690_169013


namespace NUMINAMATH_CALUDE_sqrt_sum_of_powers_l1690_169061

theorem sqrt_sum_of_powers : Real.sqrt (5^4 + 5^4 + 5^4 + 2^4) = Real.sqrt 1891 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_of_powers_l1690_169061


namespace NUMINAMATH_CALUDE_rose_bush_cost_l1690_169045

def total_cost : ℕ := 4100
def num_rose_bushes : ℕ := 20
def gardener_hourly_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def soil_volume : ℕ := 100
def soil_cost_per_unit : ℕ := 5

theorem rose_bush_cost : 
  (total_cost - (gardener_hourly_rate * gardener_hours_per_day * gardener_days) - 
   (soil_volume * soil_cost_per_unit)) / num_rose_bushes = 150 := by
  sorry

end NUMINAMATH_CALUDE_rose_bush_cost_l1690_169045


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1690_169005

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = -7*x ↔ x = 0 ∨ x = -7 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1690_169005


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1690_169099

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  A ∈ Set.Ioo 0 π →
  B ∈ Set.Ioo 0 π →
  b * Real.sin A + a * Real.cos B = 0 →
  B = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1690_169099


namespace NUMINAMATH_CALUDE_ladybugs_without_spots_l1690_169008

def total_ladybugs : ℕ := 67082
def spotted_ladybugs : ℕ := 12170

theorem ladybugs_without_spots : 
  total_ladybugs - spotted_ladybugs = 54912 :=
by sorry

end NUMINAMATH_CALUDE_ladybugs_without_spots_l1690_169008
