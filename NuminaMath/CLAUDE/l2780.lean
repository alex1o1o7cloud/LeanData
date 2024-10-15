import Mathlib

namespace NUMINAMATH_CALUDE_largest_B_term_l2780_278023

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the expansion of (1+0.1)^500 -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem largest_B_term : 
  ∀ k : ℕ, k ≤ 500 → k ≠ 45 → B 45 > B k := by sorry

end NUMINAMATH_CALUDE_largest_B_term_l2780_278023


namespace NUMINAMATH_CALUDE_S_4_S_n_l2780_278040

-- Define N(n) as the largest odd factor of n
def N (n : ℕ+) : ℕ := sorry

-- Define S(n) as the sum of N(k) from k=1 to 2^n
def S (n : ℕ) : ℕ := sorry

-- Theorem for S(4)
theorem S_4 : S 4 = 86 := by sorry

-- Theorem for S(n)
theorem S_n (n : ℕ) : S n = (4^n + 2) / 3 := by sorry

end NUMINAMATH_CALUDE_S_4_S_n_l2780_278040


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2780_278099

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 1

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (-1, 0)

/-- Theorem: The tangent line to y = x^2 + x + 1 passing through (-1, 0) is x - y + 1 = 0 -/
theorem tangent_line_equation :
  ∃ (x₀ : ℝ), 
    let y₀ := f x₀
    let m := f' x₀
    (P.1 - x₀) * m = P.2 - y₀ ∧
    ∀ (x y : ℝ), y = m * (x - x₀) + y₀ ↔ x - y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2780_278099


namespace NUMINAMATH_CALUDE_custom_op_M_T_l2780_278029

def custom_op (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

theorem custom_op_M_T :
  custom_op M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end NUMINAMATH_CALUDE_custom_op_M_T_l2780_278029


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2780_278016

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns the triple (a, b, c) -/
def quadraticCoefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem quadratic_equation_coefficients :
  quadraticCoefficients (fun x => x^2 - x) = (1, -1, 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2780_278016


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2780_278038

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 50 = 0 → a^4 - 101*a = 2550 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2780_278038


namespace NUMINAMATH_CALUDE_sarah_wallet_ones_l2780_278050

/-- Represents the contents of Sarah's wallet -/
structure Wallet where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The wallet satisfies the given conditions -/
def valid_wallet (w : Wallet) : Prop :=
  w.ones + w.twos + w.fives = 50 ∧
  w.ones + 2 * w.twos + 5 * w.fives = 146

theorem sarah_wallet_ones :
  ∃ w : Wallet, valid_wallet w ∧ w.ones = 14 := by
  sorry

end NUMINAMATH_CALUDE_sarah_wallet_ones_l2780_278050


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l2780_278081

def numKnights : ℕ := 30
def chosenKnights : ℕ := 4

def prob_adjacent_knights : ℚ :=
  1 - (Nat.choose (numKnights - chosenKnights + 1) (chosenKnights - 1) : ℚ) /
      (Nat.choose numKnights chosenKnights : ℚ)

theorem adjacent_knights_probability :
  prob_adjacent_knights = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l2780_278081


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l2780_278018

theorem floor_of_expression_equals_eight :
  ⌊(2005^3 : ℝ) / (2003 * 2004) - (2003^3 : ℝ) / (2004 * 2005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l2780_278018


namespace NUMINAMATH_CALUDE_tommy_wheel_count_l2780_278070

/-- The number of wheels on each truck -/
def truck_wheels : ℕ := 4

/-- The number of wheels on each car -/
def car_wheels : ℕ := 4

/-- The number of trucks Tommy saw -/
def trucks_seen : ℕ := 12

/-- The number of cars Tommy saw -/
def cars_seen : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := truck_wheels * trucks_seen + car_wheels * cars_seen

theorem tommy_wheel_count : total_wheels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_wheel_count_l2780_278070


namespace NUMINAMATH_CALUDE_triangle_properties_l2780_278039

open Real

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (t.a * cos t.C + t.c * cos t.A = 2 * t.b * cos t.A) →
  (t.A = π / 3) ∧
  (t.a = Real.sqrt 7 ∧ t.b = 2 →
    (1/2 * t.b * t.c * sin t.A = (3 * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2780_278039


namespace NUMINAMATH_CALUDE_solution_difference_l2780_278011

theorem solution_difference (a b : ℝ) : 
  a ≠ b ∧ 
  (6 * a - 18) / (a^2 + 3 * a - 18) = a + 3 ∧
  (6 * b - 18) / (b^2 + 3 * b - 18) = b + 3 ∧
  a > b →
  a - b = 3 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2780_278011


namespace NUMINAMATH_CALUDE_gcd_cube_plus_sixteen_and_plus_four_l2780_278046

theorem gcd_cube_plus_sixteen_and_plus_four (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 4^2) (n + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_sixteen_and_plus_four_l2780_278046


namespace NUMINAMATH_CALUDE_carpet_width_l2780_278096

/-- Proves that given a room 15 meters long and 6 meters wide, carpeted at a cost of 30 paise per meter for a total of Rs. 36, the width of the carpet used is 800 centimeters. -/
theorem carpet_width (room_length : ℝ) (room_breadth : ℝ) (carpet_cost_paise : ℝ) (total_cost_rupees : ℝ) :
  room_length = 15 →
  room_breadth = 6 →
  carpet_cost_paise = 30 →
  total_cost_rupees = 36 →
  ∃ (carpet_width : ℝ), carpet_width = 800 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l2780_278096


namespace NUMINAMATH_CALUDE_magnitude_of_linear_combination_l2780_278077

/-- Given two planar unit vectors with a right angle between them, 
    prove that the magnitude of 3 times the first vector plus 4 times the second vector is 5. -/
theorem magnitude_of_linear_combination (m n : ℝ × ℝ) : 
  ‖m‖ = 1 → ‖n‖ = 1 → m • n = 0 → ‖3 • m + 4 • n‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_linear_combination_l2780_278077


namespace NUMINAMATH_CALUDE_gcd_of_256_180_720_l2780_278047

theorem gcd_of_256_180_720 : Nat.gcd 256 (Nat.gcd 180 720) = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_720_l2780_278047


namespace NUMINAMATH_CALUDE_sphere_equation_l2780_278019

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a sphere in 3D space -/
def Sphere (center : Point3D) (radius : ℝ) : Set Point3D :=
  {p : Point3D | (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = radius^2}

/-- Theorem: The equation (x - x₀)² + (y - y₀)² + (z - z₀)² = r² represents a sphere
    with center (x₀, y₀, z₀) and radius r in a three-dimensional Cartesian coordinate system -/
theorem sphere_equation (center : Point3D) (radius : ℝ) :
  Sphere center radius = {p : Point3D | (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = radius^2} := by
  sorry

end NUMINAMATH_CALUDE_sphere_equation_l2780_278019


namespace NUMINAMATH_CALUDE_pebble_difference_l2780_278020

/-- Represents the number of pebbles thrown by each person -/
structure PebbleCount where
  candy : ℚ
  lance : ℚ
  sandy : ℚ

/-- The pebble throwing scenario -/
def pebble_scenario (p : PebbleCount) : Prop :=
  p.lance = p.candy + 10 ∧ 
  5 * p.candy = 2 * p.lance ∧
  4 * p.candy = 2 * p.sandy

theorem pebble_difference (p : PebbleCount) 
  (h : pebble_scenario p) : 
  p.lance + p.sandy - p.candy = 30 := by
  sorry

#check pebble_difference

end NUMINAMATH_CALUDE_pebble_difference_l2780_278020


namespace NUMINAMATH_CALUDE_sin_symmetry_l2780_278044

theorem sin_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x + π / 3)
  let g (x : ℝ) := f (x - π / 12)
  ∀ t, g ((-π / 12) + t) = g ((-π / 12) - t) :=
by sorry

end NUMINAMATH_CALUDE_sin_symmetry_l2780_278044


namespace NUMINAMATH_CALUDE_book_collection_ratio_l2780_278075

theorem book_collection_ratio : ∀ (L S : ℕ), 
  L + S = 3000 →  -- Total books
  S = 600 →       -- Susan's books
  L / S = 4       -- Ratio of Lidia's to Susan's books
  := by sorry

end NUMINAMATH_CALUDE_book_collection_ratio_l2780_278075


namespace NUMINAMATH_CALUDE_algorithm_design_properties_algorithm_not_endless_algorithm_not_unique_correct_statement_about_algorithms_l2780_278004

/-- Represents the properties of an algorithm -/
structure Algorithm where
  finite : Bool
  clearlyDefined : Bool
  nonUnique : Bool
  simple : Bool
  convenient : Bool
  operable : Bool

/-- Defines the correct properties of an algorithm according to computer science -/
def correctAlgorithmProperties : Algorithm :=
  { finite := true
  , clearlyDefined := true
  , nonUnique := true
  , simple := true
  , convenient := true
  , operable := true }

/-- Theorem stating that algorithms should be designed to be simple, convenient, and operable -/
theorem algorithm_design_properties :
  (a : Algorithm) → a = correctAlgorithmProperties → a.simple ∧ a.convenient ∧ a.operable :=
by sorry

/-- Theorem stating that an algorithm cannot run endlessly -/
theorem algorithm_not_endless :
  (a : Algorithm) → a = correctAlgorithmProperties → a.finite :=
by sorry

/-- Theorem stating that there can be multiple algorithms for a task -/
theorem algorithm_not_unique :
  (a : Algorithm) → a = correctAlgorithmProperties → a.nonUnique :=
by sorry

/-- Main theorem proving that the statement about algorithm design properties is correct -/
theorem correct_statement_about_algorithms :
  ∃ (a : Algorithm), a = correctAlgorithmProperties ∧
    (a.simple ∧ a.convenient ∧ a.operable) ∧
    a.finite ∧
    a.nonUnique :=
by sorry

end NUMINAMATH_CALUDE_algorithm_design_properties_algorithm_not_endless_algorithm_not_unique_correct_statement_about_algorithms_l2780_278004


namespace NUMINAMATH_CALUDE_shawn_score_shawn_score_is_six_l2780_278072

theorem shawn_score (points_per_basket : ℕ) (matthew_points : ℕ) (total_baskets : ℕ) : ℕ :=
  let matthew_baskets := matthew_points / points_per_basket
  let shawn_baskets := total_baskets - matthew_baskets
  let shawn_points := shawn_baskets * points_per_basket
  shawn_points

theorem shawn_score_is_six :
  shawn_score 3 9 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shawn_score_shawn_score_is_six_l2780_278072


namespace NUMINAMATH_CALUDE_distributive_property_only_true_l2780_278090

open Real

theorem distributive_property_only_true : ∀ b x y : ℝ,
  (b * (x + y) = b * x + b * y) ∧
  (b^(x + y) ≠ b^x + b^y) ∧
  (log (x + y) ≠ log x + log y) ∧
  (log x / log y ≠ log x - log y) ∧
  (b * (x / y) ≠ b * x / (b * y)) :=
by sorry

end NUMINAMATH_CALUDE_distributive_property_only_true_l2780_278090


namespace NUMINAMATH_CALUDE_trajectory_of_shared_focus_l2780_278015

/-- Given a parabola and a hyperbola sharing a focus, prove the trajectory of (m,n) -/
theorem trajectory_of_shared_focus (n m : ℝ) : 
  n < 0 → 
  (∃ (x y : ℝ), y^2 = 2*n*x) → 
  (∃ (x y : ℝ), x^2/4 - y^2/m^2 = 1) → 
  (∃ (f : ℝ × ℝ), f ∈ {p : ℝ × ℝ | p.1^2/(2*n) = p.2^2/m^2}) →
  n^2/16 - m^2/4 = 1 ∧ n < 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_shared_focus_l2780_278015


namespace NUMINAMATH_CALUDE_matrix_equality_l2780_278054

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![20/3, 4/3], ![-8/3, 8/3]]) : 
  B * A = ![![20/3, 4/3], ![-8/3, 8/3]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_l2780_278054


namespace NUMINAMATH_CALUDE_a_range_l2780_278008

theorem a_range (a : ℝ) (ha : a > 0) 
  (h : ∀ x : ℝ, x > 0 → 9*x + a^2/x ≥ a^2 + 8) : 
  2 ≤ a ∧ a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_a_range_l2780_278008


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l2780_278065

theorem parabola_intersection_difference (a b c d : ℝ) : 
  (∀ x, 3 * x^2 - 6 * x + 6 = -2 * x^2 - 4 * x + 6 → x = a ∨ x = c) →
  (3 * a^2 - 6 * a + 6 = -2 * a^2 - 4 * a + 6) →
  (3 * c^2 - 6 * c + 6 = -2 * c^2 - 4 * c + 6) →
  c ≥ a →
  c - a = 2/5 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l2780_278065


namespace NUMINAMATH_CALUDE_orange_percentage_l2780_278017

/-- Given a box of fruit with initial oranges and kiwis, and additional kiwis added,
    calculate the percentage of oranges in the final mixture. -/
theorem orange_percentage
  (initial_oranges : ℕ)
  (initial_kiwis : ℕ)
  (added_kiwis : ℕ)
  (h1 : initial_oranges = 24)
  (h2 : initial_kiwis = 30)
  (h3 : added_kiwis = 26) :
  (initial_oranges : ℚ) / (initial_oranges + initial_kiwis + added_kiwis) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_orange_percentage_l2780_278017


namespace NUMINAMATH_CALUDE_curve_satisfies_conditions_l2780_278033

/-- The curve that satisfies the given conditions -/
def curve (x y : ℝ) : Prop := x * y = 4

/-- The tangent line to the curve at point (x,y) -/
def tangent_line (x y : ℝ) : Set (ℝ × ℝ) :=
  {(t, s) | s - y = -(y / x) * (t - x)}

theorem curve_satisfies_conditions :
  -- The curve passes through (1,4)
  curve 1 4 ∧
  -- For any point (x,y) on the curve, the tangent line intersects
  -- the x-axis at (2x,0) and the y-axis at (0,2y)
  ∀ x y : ℝ, x > 0 → y > 0 → curve x y →
    (2*x, 0) ∈ tangent_line x y ∧ (0, 2*y) ∈ tangent_line x y :=
by sorry


end NUMINAMATH_CALUDE_curve_satisfies_conditions_l2780_278033


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l2780_278057

theorem smallest_c_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ c : ℝ, c > 0 → c * |x^(2/3) - y^(2/3)| + (x*y)^(1/3) ≥ (x^(2/3) + y^(2/3))/2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ ((1/2 - ε) * |x^(2/3) - y^(2/3)| + (x*y)^(1/3) < (x^(2/3) + y^(2/3))/2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l2780_278057


namespace NUMINAMATH_CALUDE_pizza_pooling_benefit_l2780_278056

/-- Represents a square pizza with side length and price --/
structure Pizza where
  side : ℕ
  price : ℕ

/-- Calculates the area of a square pizza --/
def pizzaArea (p : Pizza) : ℕ := p.side * p.side

/-- Calculates the number of pizzas that can be bought with a given amount of money --/
def pizzaCount (p : Pizza) (money : ℕ) : ℕ := money / p.price

/-- The small pizza option --/
def smallPizza : Pizza := { side := 6, price := 10 }

/-- The large pizza option --/
def largePizza : Pizza := { side := 9, price := 20 }

/-- The amount of money each friend has --/
def individualMoney : ℕ := 30

/-- The total amount of money when pooled --/
def pooledMoney : ℕ := 2 * individualMoney

theorem pizza_pooling_benefit :
  pizzaArea largePizza * pizzaCount largePizza pooledMoney -
  2 * (pizzaArea smallPizza * pizzaCount smallPizza individualMoney) = 135 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pooling_benefit_l2780_278056


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l2780_278067

theorem simplify_trig_fraction (x : Real) :
  let u := Real.sin (x/2) * (Real.cos (x/2) + Real.sin (x/2))
  (2 - Real.sin x + Real.cos x) / (2 + Real.sin x - Real.cos x) = (3 - 2*u) / (1 + 2*u) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l2780_278067


namespace NUMINAMATH_CALUDE_license_plate_increase_l2780_278010

-- Define the number of possible letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters and digits in old and new license plates
def old_num_letters : ℕ := 2
def old_num_digits : ℕ := 3
def new_num_letters : ℕ := 2
def new_num_digits : ℕ := 4

-- Calculate the number of possible old and new license plates
def num_old_plates : ℕ := num_letters^old_num_letters * num_digits^old_num_digits
def num_new_plates : ℕ := num_letters^new_num_letters * num_digits^new_num_digits

-- Theorem: The ratio of new to old license plates is 10
theorem license_plate_increase : num_new_plates / num_old_plates = 10 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2780_278010


namespace NUMINAMATH_CALUDE_strawberry_cost_l2780_278053

theorem strawberry_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 301 → 
  7 * J * N / 100 = 196 / 100 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_cost_l2780_278053


namespace NUMINAMATH_CALUDE_no_real_roots_iff_b_positive_l2780_278082

/-- The polynomial has no real roots if and only if b is positive -/
theorem no_real_roots_iff_b_positive (b : ℝ) : 
  (∀ x : ℝ, x^4 + b*x^3 - 2*x^2 + b*x + 2 ≠ 0) ↔ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_b_positive_l2780_278082


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2780_278032

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/5
  let r : ℚ := 1/5
  let n : ℕ := 7
  geometric_sum a r n = 78124/312500 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2780_278032


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l2780_278086

/-- Represents a 9x9 checkerboard with numbers from 1 to 81 -/
def Checkerboard : Type := Fin 9 → Fin 9 → Nat

/-- The number at position (i, j) on the checkerboard -/
def number_at (board : Checkerboard) (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The sum of the numbers in the four corners of the checkerboard -/
def corner_sum (board : Checkerboard) : Nat :=
  number_at board 0 0 +
  number_at board 0 8 +
  number_at board 8 0 +
  number_at board 8 8

/-- The theorem stating that the sum of the numbers in the four corners is 164 -/
theorem corner_sum_is_164 (board : Checkerboard) : corner_sum board = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l2780_278086


namespace NUMINAMATH_CALUDE_high_school_students_l2780_278042

theorem high_school_students (m j : ℕ) : 
  m = 4 * j →  -- Maria's school has 4 times as many students as Javier's
  m + j = 2500 →  -- Total students in both schools
  m = 2000 :=  -- Prove that Maria's school has 2000 students
by
  sorry

end NUMINAMATH_CALUDE_high_school_students_l2780_278042


namespace NUMINAMATH_CALUDE_smallest_q_value_l2780_278024

theorem smallest_q_value (p q : ℕ+) 
  (h1 : (72 : ℚ) / 487 < p.val / q.val)
  (h2 : p.val / q.val < (18 : ℚ) / 121) :
  ∀ (q' : ℕ+), ((72 : ℚ) / 487 < p.val / q'.val ∧ p.val / q'.val < (18 : ℚ) / 121) → q.val ≤ q'.val →
  q.val = 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_q_value_l2780_278024


namespace NUMINAMATH_CALUDE_final_salary_correct_l2780_278021

/-- Calculates the final salary after a series of changes -/
def calculate_final_salary (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) (deduction : ℝ) : ℝ :=
  let salary_after_raise := initial_salary * (1 + raise_percentage)
  let salary_after_cut := salary_after_raise * (1 - cut_percentage)
  salary_after_cut - deduction

/-- Theorem stating that the final salary matches the expected value -/
theorem final_salary_correct (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) (deduction : ℝ) 
    (h1 : initial_salary = 3000)
    (h2 : raise_percentage = 0.1)
    (h3 : cut_percentage = 0.15)
    (h4 : deduction = 100) :
  calculate_final_salary initial_salary raise_percentage cut_percentage deduction = 2705 := by
  sorry

end NUMINAMATH_CALUDE_final_salary_correct_l2780_278021


namespace NUMINAMATH_CALUDE_only_2015_could_be_hexadecimal_l2780_278013

def is_hexadecimal_digit (d : Char) : Bool :=
  ('0' <= d && d <= '9') || ('A' <= d && d <= 'F')

def could_be_hexadecimal (n : Nat) : Bool :=
  n.repr.all is_hexadecimal_digit

theorem only_2015_could_be_hexadecimal :
  (could_be_hexadecimal 66 = false) ∧
  (could_be_hexadecimal 108 = false) ∧
  (could_be_hexadecimal 732 = false) ∧
  (could_be_hexadecimal 2015 = true) :=
by sorry

end NUMINAMATH_CALUDE_only_2015_could_be_hexadecimal_l2780_278013


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2780_278002

/-- A triangle with an inscribed rectangle -/
structure InscribedRectangle where
  /-- The height of the triangle -/
  triangle_height : ℝ
  /-- The base of the triangle -/
  triangle_base : ℝ
  /-- The width of the inscribed rectangle -/
  rectangle_width : ℝ
  /-- The length of the inscribed rectangle -/
  rectangle_length : ℝ
  /-- The width of the rectangle is one-third of its length -/
  width_is_third_of_length : rectangle_width = rectangle_length / 3
  /-- The rectangle is inscribed in the triangle -/
  rectangle_inscribed : rectangle_length ≤ triangle_base

/-- The area of the inscribed rectangle given the triangle's dimensions -/
def rectangle_area (r : InscribedRectangle) : ℝ :=
  r.rectangle_width * r.rectangle_length

/-- Theorem: The area of the inscribed rectangle is 675/64 square inches -/
theorem inscribed_rectangle_area (r : InscribedRectangle)
    (h1 : r.triangle_height = 9)
    (h2 : r.triangle_base = 15) :
    rectangle_area r = 675 / 64 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2780_278002


namespace NUMINAMATH_CALUDE_average_difference_l2780_278091

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2780_278091


namespace NUMINAMATH_CALUDE_valid_pairs_l2780_278051

def is_valid_pair (m n : ℕ+) : Prop :=
  (3^m.val + 1) % (m.val * n.val) = 0 ∧ (3^n.val + 1) % (m.val * n.val) = 0

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ 
    ((m = 1 ∧ n = 1) ∨ 
     (m = 1 ∧ n = 2) ∨ 
     (m = 1 ∧ n = 4) ∨ 
     (m = 2 ∧ n = 1) ∨ 
     (m = 4 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2780_278051


namespace NUMINAMATH_CALUDE_group_size_is_correct_l2780_278071

/-- The number of people in a group where:
  1. The average weight increases by 2.5 kg when a new person joins.
  2. The person being replaced weighs 45 kg.
  3. The new person weighs 65 kg.
-/
def group_size : ℕ := 8

/-- The weight of the person being replaced -/
def original_weight : ℝ := 45

/-- The weight of the new person joining the group -/
def new_weight : ℝ := 65

/-- The increase in average weight when the new person joins -/
def average_increase : ℝ := 2.5

theorem group_size_is_correct : 
  (new_weight - original_weight) = (average_increase * group_size) :=
sorry

end NUMINAMATH_CALUDE_group_size_is_correct_l2780_278071


namespace NUMINAMATH_CALUDE_impossible_c_value_l2780_278052

theorem impossible_c_value (a b c : ℤ) : 
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + c*x - 8) → c ≠ 4 := by
sorry

end NUMINAMATH_CALUDE_impossible_c_value_l2780_278052


namespace NUMINAMATH_CALUDE_red_to_yellow_ratio_l2780_278043

/-- Represents the number of mugs of each color in Hannah's collection. -/
structure MugCollection where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  other : ℕ

/-- Checks if a mug collection satisfies Hannah's conditions. -/
def isValidCollection (m : MugCollection) : Prop :=
  m.red + m.blue + m.yellow + m.other = 40 ∧
  m.blue = 3 * m.red ∧
  m.yellow = 12 ∧
  m.other = 4

/-- Theorem stating that for any valid mug collection, the ratio of red to yellow mugs is 1:2. -/
theorem red_to_yellow_ratio (m : MugCollection) (h : isValidCollection m) :
  m.red * 2 = m.yellow := by sorry

end NUMINAMATH_CALUDE_red_to_yellow_ratio_l2780_278043


namespace NUMINAMATH_CALUDE_truck_filling_problem_l2780_278003

/-- A problem about filling a truck with stone blocks -/
theorem truck_filling_problem 
  (truck_capacity : ℕ) 
  (initial_workers : ℕ) 
  (work_rate : ℕ) 
  (initial_work_time : ℕ) 
  (total_time : ℕ)
  (h1 : truck_capacity = 6000)
  (h2 : initial_workers = 2)
  (h3 : work_rate = 250)
  (h4 : initial_work_time = 4)
  (h5 : total_time = 6)
  : ∃ (joined_workers : ℕ),
    (initial_workers * work_rate * initial_work_time) + 
    ((initial_workers + joined_workers) * work_rate * (total_time - initial_work_time)) = 
    truck_capacity ∧ joined_workers = 6 := by
  sorry


end NUMINAMATH_CALUDE_truck_filling_problem_l2780_278003


namespace NUMINAMATH_CALUDE_power_function_through_point_l2780_278066

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/3)

-- State the theorem
theorem power_function_through_point (h : f 27 = 3) : f 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2780_278066


namespace NUMINAMATH_CALUDE_unique_N_exists_l2780_278036

theorem unique_N_exists : ∃! N : ℝ, 
  ∃ a b c : ℝ, 
    a + b + c = 120 ∧
    a + 8 = N ∧
    8 * b = N ∧
    c / 8 = N := by
  sorry

end NUMINAMATH_CALUDE_unique_N_exists_l2780_278036


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2780_278073

theorem inheritance_calculation (x : ℝ) : 
  let after_charity := 0.95 * x
  let federal_tax := 0.25 * after_charity
  let after_federal := after_charity - federal_tax
  let state_tax := 0.12 * after_federal
  federal_tax + state_tax = 15000 → x = 46400 := by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2780_278073


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2780_278060

theorem no_solution_for_equation : ¬∃ (a b : ℕ), 2 * a^2 + 1 = 4 * b^2 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2780_278060


namespace NUMINAMATH_CALUDE_unique_number_property_l2780_278048

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l2780_278048


namespace NUMINAMATH_CALUDE_rectangle_dimensions_theorem_l2780_278064

def rectangle_dimensions (w : ℝ) : Prop :=
  let l := w + 3
  let perimeter := 2 * (w + l)
  let area := w * l
  perimeter = 2 * area ∧ w > 0 ∧ l > 0 → w = 1 ∧ l = 4

theorem rectangle_dimensions_theorem :
  ∃ w : ℝ, rectangle_dimensions w := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_theorem_l2780_278064


namespace NUMINAMATH_CALUDE_geometric_series_relation_l2780_278000

/-- Given two infinite geometric series:
    Series I with first term a₁ = 12 and common ratio r₁ = 1/3
    Series II with first term a₂ = 12 and common ratio r₂ = (4+n)/12
    If the sum of Series II is five times the sum of Series I, then n = 152 -/
theorem geometric_series_relation (n : ℝ) : 
  let a₁ : ℝ := 12
  let r₁ : ℝ := 1/3
  let a₂ : ℝ := 12
  let r₂ : ℝ := (4+n)/12
  (a₁ / (1 - r₁) = a₂ / (1 - r₂) / 5) → n = 152 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l2780_278000


namespace NUMINAMATH_CALUDE_students_with_puppies_and_parrots_l2780_278012

theorem students_with_puppies_and_parrots 
  (total_students : ℕ) 
  (puppy_percentage : ℚ) 
  (parrot_percentage : ℚ) 
  (h1 : total_students = 40)
  (h2 : puppy_percentage = 80 / 100)
  (h3 : parrot_percentage = 25 / 100) :
  ⌊(total_students : ℚ) * puppy_percentage * parrot_percentage⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_with_puppies_and_parrots_l2780_278012


namespace NUMINAMATH_CALUDE_unique_solution_l2780_278087

def equation1 (x y : ℝ) : Prop := 3 * x + 4 * y = 26

def equation2 (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 1)^2) + Real.sqrt ((x - 10)^2 + (y - 5)^2) = 10

theorem unique_solution :
  ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2 ∧ p = (6, 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2780_278087


namespace NUMINAMATH_CALUDE_janes_mean_score_l2780_278045

def janes_scores : List ℝ := [98, 97, 92, 85, 93]

theorem janes_mean_score :
  (janes_scores.sum / janes_scores.length : ℝ) = 93 := by
  sorry

end NUMINAMATH_CALUDE_janes_mean_score_l2780_278045


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2780_278089

theorem quadratic_function_property (a b : ℝ) : 
  let f := λ x : ℝ => x^2 + a*x + b
  (f 1 = 0) → (f 2 = 0) → (f (-1) = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2780_278089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2780_278049

/-- An arithmetic sequence with positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first three terms of the sequence is 15 -/
def sum_first_three (seq : ArithmeticSequence) : Prop :=
  seq.a 1 + seq.a 2 + seq.a 3 = 15

/-- The product of the first three terms of the sequence is 80 -/
def product_first_three (seq : ArithmeticSequence) : Prop :=
  seq.a 1 * seq.a 2 * seq.a 3 = 80

/-- Theorem: If the sum of the first three terms is 15 and their product is 80,
    then the sum of the 11th, 12th, and 13th terms is 135 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h_sum : sum_first_three seq) (h_product : product_first_three seq) :
  seq.a 11 + seq.a 12 + seq.a 13 = 135 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2780_278049


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2780_278092

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := -1
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ = (-5 + Real.sqrt 29) / 2 ∧
  x₂ = (-5 - Real.sqrt 29) / 2 ∧
  a * x₁^2 + b * x₁ + c = 0 ∧
  a * x₂^2 + b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2780_278092


namespace NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l2780_278014

/-- Given a hyperbola with equation 9x^2 - 16y^2 = 144, 
    its asymptotic lines are y = ± 3/4 x -/
theorem hyperbola_asymptotic_lines :
  let hyperbola := {(x, y) : ℝ × ℝ | 9 * x^2 - 16 * y^2 = 144}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = 3/4 * x ∨ y = -3/4 * x}
  asymptotic_lines = {(x, y) : ℝ × ℝ | ∃ (t : ℝ), t ≠ 0 ∧ (t*x, t*y) ∈ hyperbola} :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l2780_278014


namespace NUMINAMATH_CALUDE_triple_application_of_f_l2780_278088

def f (p : ℝ) : ℝ := 2 * p + 20

theorem triple_application_of_f :
  ∃ p : ℝ, f (f (f p)) = -4 ∧ p = -18 := by
  sorry

end NUMINAMATH_CALUDE_triple_application_of_f_l2780_278088


namespace NUMINAMATH_CALUDE_grant_total_earnings_l2780_278059

/-- Grant's earnings over four months as a freelance math worker -/
def grant_earnings (X Y Z W : ℕ) : ℕ :=
  let month1 := X
  let month2 := 3 * X + Y
  let month3 := 2 * month2 - Z
  let month4 := (month1 + month2 + month3) / 3 + W
  month1 + month2 + month3 + month4

/-- Theorem stating Grant's total earnings over four months -/
theorem grant_total_earnings :
  grant_earnings 350 30 20 50 = 5810 := by
  sorry

end NUMINAMATH_CALUDE_grant_total_earnings_l2780_278059


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_9x_l2780_278025

theorem factorization_x_squared_minus_9x (x : ℝ) : x^2 - 9*x = x*(x - 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_9x_l2780_278025


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_iff_strictly_increasing_l2780_278083

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The condition that a_{n+2} > a_n for all positive integers n -/
def Condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) > a n

/-- The sequence is strictly increasing -/
def StrictlyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_condition_iff_strictly_increasing
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  Condition a ↔ StrictlyIncreasing a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_iff_strictly_increasing_l2780_278083


namespace NUMINAMATH_CALUDE_marble_count_theorem_l2780_278068

/-- Represents the total number of marbles in a bag given the ratio of colors and the number of green marbles -/
def total_marbles (red blue green yellow : ℕ) (green_count : ℕ) : ℕ :=
  (red + blue + green + yellow) * green_count / green

/-- Theorem stating that given the specific ratio and number of green marbles, the total is 120 -/
theorem marble_count_theorem :
  total_marbles 1 3 2 4 24 = 120 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_theorem_l2780_278068


namespace NUMINAMATH_CALUDE_smallest_cube_ending_in_388_l2780_278027

def is_cube_ending_in_388 (n : ℕ) : Prop := n^3 % 1000 = 388

theorem smallest_cube_ending_in_388 : 
  (∃ (n : ℕ), is_cube_ending_in_388 n) ∧ 
  (∀ (m : ℕ), m < 16 → ¬is_cube_ending_in_388 m) ∧ 
  is_cube_ending_in_388 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_in_388_l2780_278027


namespace NUMINAMATH_CALUDE_second_half_speed_l2780_278080

/-- Represents the speed of a car during a trip -/
structure TripSpeed where
  average : ℝ
  firstHalf : ℝ
  secondHalf : ℝ

/-- Theorem stating the speed of the car in the second half of the trip -/
theorem second_half_speed (trip : TripSpeed) (h1 : trip.average = 60) (h2 : trip.firstHalf = 75) :
  trip.secondHalf = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_half_speed_l2780_278080


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l2780_278097

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y, a * x + 2 * y + 1 = 0) →
  (∀ x y, x + y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂, a * x₁ + 2 * y₁ + 1 = 0 ∧ x₂ + y₂ - 2 = 0 → 
    (y₂ - y₁) * (x₂ - x₁) = -(x₂ - x₁) * (y₂ - y₁)) →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l2780_278097


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l2780_278094

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_composite_reverse : 
  ∃ (p : ℕ), is_two_digit p ∧ Nat.Prime p ∧ 
  ¬(Nat.Prime (reverse_digits p)) ∧
  ∀ (q : ℕ), is_two_digit q → Nat.Prime q → 
  ¬(Nat.Prime (reverse_digits q)) → p ≤ q ∧ p = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l2780_278094


namespace NUMINAMATH_CALUDE_density_difference_of_cubes_l2780_278061

theorem density_difference_of_cubes (m₁ : ℝ) (a₁ : ℝ) (m₁_pos : m₁ > 0) (a₁_pos : a₁ > 0) :
  let m₂ := 0.75 * m₁
  let a₂ := 1.25 * a₁
  let ρ₁ := m₁ / (a₁^3)
  let ρ₂ := m₂ / (a₂^3)
  (ρ₁ - ρ₂) / ρ₁ = 0.616 := by
sorry

end NUMINAMATH_CALUDE_density_difference_of_cubes_l2780_278061


namespace NUMINAMATH_CALUDE_odd_prime_expression_factors_l2780_278093

theorem odd_prime_expression_factors (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hab : a < b) : 
  (Finset.filter (· ∣ a^3 * b) (Finset.range (a^3 * b + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_expression_factors_l2780_278093


namespace NUMINAMATH_CALUDE_all_propositions_false_l2780_278001

theorem all_propositions_false : ∃ a b : ℝ,
  (a > b ∧ a^2 ≤ b^2) ∧
  (a^2 > b^2 ∧ a ≤ b) ∧
  (a > b ∧ b/a ≥ 1) ∧
  (a > b ∧ 1/a ≥ 1/b) := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2780_278001


namespace NUMINAMATH_CALUDE_reflection_of_P_l2780_278006

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -5)

theorem reflection_of_P :
  reflect_y P = (-3, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_P_l2780_278006


namespace NUMINAMATH_CALUDE_k_domain_l2780_278030

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 3) + 1 / (x^2 + 3) + 1 / (x^3 + 3)

def domain_k : Set ℝ := {x | x ≠ -3 ∧ x ≠ -Real.rpow 3 (1/3)}

theorem k_domain :
  {x : ℝ | ∃ y, k x = y} = domain_k :=
sorry

end NUMINAMATH_CALUDE_k_domain_l2780_278030


namespace NUMINAMATH_CALUDE_initial_oranges_count_l2780_278074

/-- Proves that the initial number of oranges in a bowl is 20, given the specified conditions. -/
theorem initial_oranges_count (apples : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  apples = 14 → removed_oranges = 14 → apple_percentage = 7/10 → 
  ∃ initial_oranges : ℕ, 
    initial_oranges = 20 ∧ 
    (apples : ℚ) / ((apples : ℚ) + (initial_oranges - removed_oranges : ℚ)) = apple_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l2780_278074


namespace NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l2780_278026

/-- Sequence definition -/
def a : ℕ → ℕ → ℕ
  | m, 0 => 0
  | m, 1 => 1
  | m, k + 2 => m * a m (k + 1) - a m k

/-- Main theorem -/
theorem consecutive_terms_iff_equation (m : ℕ) :
  ∀ x y : ℕ, x^2 - m*x*y + y^2 = 1 ↔ ∃ k : ℕ, x = a m k ∧ y = a m (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l2780_278026


namespace NUMINAMATH_CALUDE_percentage_equality_l2780_278063

theorem percentage_equality (x : ℝ) : (90 / 100 * 600 = 50 / 100 * x) → x = 1080 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2780_278063


namespace NUMINAMATH_CALUDE_boat_distance_proof_l2780_278055

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

theorem boat_distance_proof (boat_speed stream_speed time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 3) :
  distance_downstream boat_speed stream_speed time = 63 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l2780_278055


namespace NUMINAMATH_CALUDE_two_true_propositions_l2780_278005

theorem two_true_propositions (p q : Prop) (h : p ∧ q) :
  (p ∨ q) ∧ p ∧ ¬(¬q) ∧ ¬((¬p) ∨ (¬q)) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l2780_278005


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l2780_278078

/-- Given an arithmetic sequence where a₁ = -3 and a₂ = 1, prove that a₆ = 17 -/
theorem sixth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℤ), 
    a 1 = -3 → 
    a 2 = 1 → 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → 
    a 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l2780_278078


namespace NUMINAMATH_CALUDE_expression_equals_59_l2780_278069

theorem expression_equals_59 (a b c : ℝ) (ha : a = 17) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)) = 59 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_59_l2780_278069


namespace NUMINAMATH_CALUDE_cube_face_sum_l2780_278058

theorem cube_face_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f = 1729) → 
  (a + b + c + d + e + f = 39) := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l2780_278058


namespace NUMINAMATH_CALUDE_outstanding_student_distribution_l2780_278041

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 126 ways to distribute 10 indistinguishable objects
    into 6 distinguishable containers, with each container receiving at least one object. -/
theorem outstanding_student_distribution : distribute 10 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_outstanding_student_distribution_l2780_278041


namespace NUMINAMATH_CALUDE_pizza_cut_area_theorem_l2780_278095

/-- Represents a circular pizza -/
structure Pizza where
  area : ℝ
  radius : ℝ

/-- Represents a cut on the pizza -/
structure Cut where
  distance_from_center : ℝ

/-- Theorem: Given a circular pizza with area 4 μ² cut into 4 parts by two perpendicular
    straight cuts each at a distance of 50 cm from the center, the sum of the areas of
    two opposite pieces is equal to 1.5 μ² -/
theorem pizza_cut_area_theorem (p : Pizza) (c : Cut) :
  p.area = 4 →
  c.distance_from_center = 0.5 →
  ∃ (piece1 piece2 : ℝ), piece1 + piece2 = 1.5 ∧ 
    piece1 + piece2 = (p.area - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_pizza_cut_area_theorem_l2780_278095


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2780_278007

/-- Represents a repeating decimal with a 4-digit repetend -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d) / 9999

theorem repeating_decimal_subtraction :
  RepeatingDecimal 2 3 4 5 - RepeatingDecimal 6 7 8 9 - RepeatingDecimal 1 2 3 4 = -5678 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2780_278007


namespace NUMINAMATH_CALUDE_defective_units_percentage_l2780_278076

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.05)
  (h2 : total_shipped_defective_ratio = 0.0035) :
  ∃ (defective_ratio : Real),
    defective_ratio = 0.07 ∧
    shipped_defective_ratio * defective_ratio = total_shipped_defective_ratio :=
by sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l2780_278076


namespace NUMINAMATH_CALUDE_inverse_proposition_l2780_278085

-- Define the original proposition
def corresponding_angles_equal : Prop := sorry

-- Define the inverse proposition
def equal_angles_corresponding : Prop := sorry

-- Theorem stating that equal_angles_corresponding is the inverse of corresponding_angles_equal
theorem inverse_proposition : 
  (corresponding_angles_equal → equal_angles_corresponding) ∧ 
  (equal_angles_corresponding → corresponding_angles_equal) := by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2780_278085


namespace NUMINAMATH_CALUDE_circles_intersection_common_chord_l2780_278034

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

theorem circles_intersection_common_chord :
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) →
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersection_common_chord_l2780_278034


namespace NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l2780_278098

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def parallel (a b : E) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_equality_sufficient_not_necessary 
  (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a = b → (‖a‖ = ‖b‖ ∧ parallel a b)) ∧ 
  ∃ (c d : E), ‖c‖ = ‖d‖ ∧ parallel c d ∧ c ≠ d := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_sufficient_not_necessary_l2780_278098


namespace NUMINAMATH_CALUDE_no_geometric_subsequence_of_three_l2780_278022

theorem no_geometric_subsequence_of_three (a : ℕ → ℤ) :
  (∀ n, a n = 3^n - 2^n) →
  ¬ ∃ r s t : ℕ, r < s ∧ s < t ∧ ∃ b : ℚ, b ≠ 0 ∧
    (a s : ℚ) / (a r : ℚ) = b ∧ (a t : ℚ) / (a s : ℚ) = b :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_subsequence_of_three_l2780_278022


namespace NUMINAMATH_CALUDE_red_squares_less_than_half_l2780_278062

/-- Represents a cube with side length 3, composed of 27 unit cubes -/
structure LargeCube where
  total_units : Nat
  red_units : Nat
  blue_units : Nat
  side_length : Nat

/-- Calculates the total number of visible unit squares on the surface of the large cube -/
def total_surface_squares (cube : LargeCube) : Nat :=
  6 * (cube.side_length * cube.side_length)

/-- Calculates the maximum number of red unit squares that can be visible on the surface -/
def max_red_surface_squares (cube : LargeCube) : Nat :=
  (cube.side_length - 1) * (cube.side_length - 1) * 3 + (cube.side_length - 1) * 3 * 2 + 8 * 3

/-- Theorem stating that the maximum number of red squares on the surface is less than half the total -/
theorem red_squares_less_than_half (cube : LargeCube) 
  (h1 : cube.total_units = 27)
  (h2 : cube.red_units = 9)
  (h3 : cube.blue_units = 18)
  (h4 : cube.side_length = 3)
  : max_red_surface_squares cube < (total_surface_squares cube) / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_squares_less_than_half_l2780_278062


namespace NUMINAMATH_CALUDE_freds_baseball_cards_l2780_278028

/-- Fred's baseball card problem -/
theorem freds_baseball_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 40 → cards_bought = 22 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 18 := by
  sorry

end NUMINAMATH_CALUDE_freds_baseball_cards_l2780_278028


namespace NUMINAMATH_CALUDE_fraction_comparison_l2780_278035

theorem fraction_comparison (n : ℕ) (hn : n > 0) :
  (n + 1 : ℝ) ^ (n + 3) / (n + 3 : ℝ) ^ (n + 1) > n ^ (n + 2) / (n + 2 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2780_278035


namespace NUMINAMATH_CALUDE_book_sale_profit_l2780_278037

/-- Calculates the percent profit for a book sale given the cost, markup percentage, and discount percentage. -/
theorem book_sale_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  cost = 50 ∧ markup_percent = 30 ∧ discount_percent = 10 →
  (((cost * (1 + markup_percent / 100)) * (1 - discount_percent / 100) - cost) / cost) * 100 = 17 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_l2780_278037


namespace NUMINAMATH_CALUDE_arnel_friends_count_l2780_278009

/-- Represents the pencil sharing problem --/
def pencil_sharing (num_boxes : ℕ) (pencils_per_box : ℕ) (kept_pencils : ℕ) (pencils_per_friend : ℕ) : ℕ :=
  let total_pencils := num_boxes * pencils_per_box
  let shared_pencils := total_pencils - kept_pencils
  shared_pencils / pencils_per_friend

/-- Proves that Arnel shared pencils with 5 friends --/
theorem arnel_friends_count :
  pencil_sharing 10 5 10 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arnel_friends_count_l2780_278009


namespace NUMINAMATH_CALUDE_parrot_fraction_l2780_278031

theorem parrot_fraction (p t : ℝ) : 
  p + t = 1 →                     -- Total fraction of birds
  (2/3 : ℝ) * p + (1/4 : ℝ) * t = (1/2 : ℝ) →  -- Male birds equation
  p = (3/5 : ℝ) := by             -- Fraction of parrots
sorry

end NUMINAMATH_CALUDE_parrot_fraction_l2780_278031


namespace NUMINAMATH_CALUDE_basketball_team_selection_count_l2780_278079

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 7 starters from a team of 16 players,
    including a set of 4 quadruplets, where exactly 3 of the quadruplets
    must be in the starting lineup -/
def basketball_team_selection : ℕ :=
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  (choose quadruplets quadruplets_in_lineup) *
  (choose (total_players - quadruplets) (starters - quadruplets_in_lineup))

theorem basketball_team_selection_count :
  basketball_team_selection = 1980 := by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_count_l2780_278079


namespace NUMINAMATH_CALUDE_book_ratio_l2780_278084

/-- The number of books Pete read last year -/
def P : ℕ := sorry

/-- The number of books Matt read last year -/
def M : ℕ := sorry

/-- Pete doubles his reading this year -/
axiom pete_doubles : P * 2 = 300 - P

/-- Matt reads 50% more this year -/
axiom matt_increases : M * 3/2 = 75

/-- Pete read 300 books across both years -/
axiom pete_total : P + P * 2 = 300

/-- Matt read 75 books in his second year -/
axiom matt_second_year : M * 3/2 = 75

/-- The ratio of books Pete read last year to books Matt read last year is 2:1 -/
theorem book_ratio : P / M = 2 := by sorry

end NUMINAMATH_CALUDE_book_ratio_l2780_278084
