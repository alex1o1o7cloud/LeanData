import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_locus_l3966_396608

/-- Given two fixed points F₁ and F₂ on the x-axis, and a point M such that
    the sum of its distances to F₁ and F₂ is constant, prove that the locus of M
    is an ellipse with F₁ and F₂ as foci. -/
theorem ellipse_locus (F₁ F₂ M : ℝ × ℝ) (d : ℝ) :
  F₁ = (-4, 0) →
  F₂ = (4, 0) →
  d = 10 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) +
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = d →
  M.1^2 / 25 + M.2^2 / 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_locus_l3966_396608


namespace NUMINAMATH_CALUDE_bag_problem_l3966_396640

/-- The number of red balls in the bag -/
def red_balls (a : ℕ) : ℕ := a + 1

/-- The number of yellow balls in the bag -/
def yellow_balls (a : ℕ) : ℕ := a

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls (a : ℕ) : ℕ := red_balls a + yellow_balls a + blue_balls

/-- The score earned by drawing a red ball -/
def red_score : ℕ := 1

/-- The score earned by drawing a yellow ball -/
def yellow_score : ℕ := 2

/-- The score earned by drawing a blue ball -/
def blue_score : ℕ := 3

/-- The expected value of the score when drawing a ball -/
def expected_value : ℚ := 5 / 3

theorem bag_problem (a : ℕ) :
  (a = 2) ∧
  (let p : ℚ := (Nat.choose 3 1 * Nat.choose 2 2 + Nat.choose 3 2 * Nat.choose 1 1) / Nat.choose 6 3
   p = 3 / 10) :=
by sorry

end NUMINAMATH_CALUDE_bag_problem_l3966_396640


namespace NUMINAMATH_CALUDE_passing_percentage_is_25_percent_l3966_396666

/-- The percentage of total marks needed to pass a test -/
def passing_percentage (pradeep_score : ℕ) (failed_by : ℕ) (max_marks : ℕ) : ℚ :=
  (pradeep_score + failed_by : ℚ) / max_marks * 100

/-- Theorem stating that the passing percentage is 25% given the problem conditions -/
theorem passing_percentage_is_25_percent :
  passing_percentage 185 25 840 = 25 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_25_percent_l3966_396666


namespace NUMINAMATH_CALUDE_K_set_equals_target_set_l3966_396691

/-- The set of natural numbers K satisfying the given conditions for a fixed h = 2^r -/
def K_set (r : ℕ) : Set ℕ :=
  {K : ℕ | ∃ (m n : ℕ), m > 1 ∧ Odd m ∧
    K ∣ (m^(2^r) - 1) ∧
    K ∣ (n^((m^(2^r) - 1) / K) + 1)}

/-- The set of numbers of the form 2^(r+s) * t where t is odd -/
def target_set (r : ℕ) : Set ℕ :=
  {K : ℕ | ∃ (s t : ℕ), K = 2^(r+s) * t ∧ Odd t}

/-- The main theorem stating that K_set equals target_set for any non-negative integer r -/
theorem K_set_equals_target_set (r : ℕ) : K_set r = target_set r := by
  sorry

end NUMINAMATH_CALUDE_K_set_equals_target_set_l3966_396691


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3966_396641

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 5 = 0 ∧         -- divisible by 5
  n % 3 ≠ 0 ∧         -- not divisible by 3
  n % 4 ≠ 0 ∧         -- not divisible by 4
  (97 * n) % 2 = 0 ∧  -- 97 times is even
  n / 10 ≥ 6 ∧        -- tens digit not less than 6
  n = 70              -- the number is 70
  := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3966_396641


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l3966_396606

/-- The number of bowl colors -/
def numBowls : ℕ := 5

/-- The number of glass colors -/
def numGlasses : ℕ := 4

/-- The total number of possible pairings without restrictions -/
def totalPairings : ℕ := numBowls * numGlasses

/-- The number of restricted pairings (purple bowl with green glass) -/
def restrictedPairings : ℕ := 1

/-- The number of valid pairings -/
def validPairings : ℕ := totalPairings - restrictedPairings

theorem bowl_glass_pairings :
  validPairings = 19 :=
sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l3966_396606


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3966_396644

theorem trig_expression_simplification (α β : Real) :
  (Real.sin (α + β))^2 - Real.sin α^2 - Real.sin β^2 /
  ((Real.sin (α + β))^2 - Real.cos α^2 - Real.cos β^2) = 
  -Real.tan α * Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3966_396644


namespace NUMINAMATH_CALUDE_sibling_age_sum_l3966_396661

/-- Given the ages of three siblings, prove that the sum of the younger and older siblings' ages is correct. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = 10 → 
  juliet = maggie + 3 → 
  ralph = juliet + 2 → 
  maggie + ralph = 19 := by
sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l3966_396661


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3966_396653

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3966_396653


namespace NUMINAMATH_CALUDE_max_distance_to_line_l3966_396605

/-- The maximum distance from the point (1, 1) to the line x*cos(θ) + y*sin(θ) = 2 is 2 + √2 -/
theorem max_distance_to_line : 
  let P : ℝ × ℝ := (1, 1)
  let line (θ : ℝ) (x y : ℝ) := x * Real.cos θ + y * Real.sin θ = 2
  ∃ (d : ℝ), d = 2 + Real.sqrt 2 ∧ 
    ∀ (θ : ℝ), d ≥ Real.sqrt ((P.1 * Real.cos θ + P.2 * Real.sin θ - 2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l3966_396605


namespace NUMINAMATH_CALUDE_stamp_arrangement_exists_l3966_396693

/-- Represents the quantity of each stamp denomination -/
def stamp_quantities : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- Represents the value of each stamp denomination -/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- A function to calculate the number of unique stamp arrangements -/
def count_stamp_arrangements (quantities : List Nat) (values : List Nat) (target : Nat) : Nat :=
  sorry

/-- Theorem stating that there exists a positive number of unique arrangements -/
theorem stamp_arrangement_exists :
  ∃ n : Nat, n > 0 ∧ count_stamp_arrangements stamp_quantities stamp_values 15 = n :=
sorry

end NUMINAMATH_CALUDE_stamp_arrangement_exists_l3966_396693


namespace NUMINAMATH_CALUDE_area_ratio_abc_xyz_l3966_396668

-- Define points as pairs of real numbers
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (2, 0)
def B : Point := (8, 12)
def C : Point := (14, 0)
def X : Point := (6, 0)
def Y : Point := (8, 4)
def Z : Point := (10, 0)

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem area_ratio_abc_xyz :
  (triangleArea X Y Z) / (triangleArea A B C) = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_area_ratio_abc_xyz_l3966_396668


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3966_396659

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ n : ℕ, n = 8 ∧ D = n * (n - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3966_396659


namespace NUMINAMATH_CALUDE_triangle_inequality_l3966_396694

theorem triangle_inequality (a b c : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h4 : a * b + b * c + c * a = 18) : 
  1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3 > 1 / (a + b + c - 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3966_396694


namespace NUMINAMATH_CALUDE_hostel_problem_l3966_396663

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 32

/-- The number of days the provisions would last if 50 men left -/
def reduced_days : ℕ := 40

/-- The number of men that left the hostel -/
def men_left : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_problem :
  initial_men = 250 ∧
  (initial_days : ℚ) * initial_men = reduced_days * (initial_men - men_left) :=
sorry

end NUMINAMATH_CALUDE_hostel_problem_l3966_396663


namespace NUMINAMATH_CALUDE_sector_area_l3966_396665

/-- Given a circular sector with perimeter 10 and central angle 3 radians, its area is 6 -/
theorem sector_area (r : ℝ) (perimeter : ℝ) (central_angle : ℝ) : 
  perimeter = 10 → central_angle = 3 → perimeter = 2 * r + central_angle * r → 
  (1/2) * r^2 * central_angle = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3966_396665


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l3966_396685

theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) : ∃ b : ℝ, b^2 = 2*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l3966_396685


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3966_396656

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3966_396656


namespace NUMINAMATH_CALUDE_QED_product_l3966_396602

theorem QED_product (Q E D : ℂ) : 
  Q = 5 + 2*I ∧ E = I ∧ D = 5 - 2*I → Q * E * D = 29 * I :=
by sorry

end NUMINAMATH_CALUDE_QED_product_l3966_396602


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3966_396635

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = -p.1 + 1}
def N : Set (ℝ × ℝ) := {p | p.2 = p.1 - 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(1, 0)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3966_396635


namespace NUMINAMATH_CALUDE_equation_solution_l3966_396695

theorem equation_solution : ∃! x : ℚ, (x - 35) / 3 = (3 * x + 10) / 8 ∧ x = -310 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3966_396695


namespace NUMINAMATH_CALUDE_range_of_m_l3966_396614

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≥ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≥ 0

def not_p (x : ℝ) : Prop := -2 < x ∧ x < 10

def not_q (x m : ℝ) : Prop := 1 - m < x ∧ x < 1 + m

theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧
    (∀ x : ℝ, not_q x m → not_p x) ∧
    (∃ x : ℝ, not_p x ∧ ¬(not_q x m))) ↔
  (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3966_396614


namespace NUMINAMATH_CALUDE_child_support_calculation_l3966_396620

def child_support_owed (base_salary : List ℝ) (bonuses : List ℝ) (rates : List ℝ) (paid : ℝ) : ℝ :=
  let incomes := List.zipWith (· + ·) base_salary bonuses
  let owed := List.sum (List.zipWith (· * ·) incomes rates)
  owed - paid

theorem child_support_calculation : 
  let base_salary := [30000, 30000, 30000, 36000, 36000, 36000, 36000]
  let bonuses := [2000, 3000, 4000, 5000, 6000, 7000, 8000]
  let rates := [0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25]
  let paid := 1200
  child_support_owed base_salary bonuses rates paid = 75150 := by
  sorry

#eval child_support_owed 
  [30000, 30000, 30000, 36000, 36000, 36000, 36000]
  [2000, 3000, 4000, 5000, 6000, 7000, 8000]
  [0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25]
  1200

end NUMINAMATH_CALUDE_child_support_calculation_l3966_396620


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3966_396616

theorem pure_imaginary_complex_number (b : ℝ) : 
  let z : ℂ := (1 + b * Complex.I) * (2 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I ∧ y ≠ 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3966_396616


namespace NUMINAMATH_CALUDE_bill_donut_purchase_l3966_396618

/-- The number of ways to distribute donuts among types with constraints -/
def donut_combinations (total_donuts : ℕ) (num_types : ℕ) (min_types : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the specific case for Bill's donut purchase -/
theorem bill_donut_purchase :
  donut_combinations 8 5 4 = 425 :=
sorry

end NUMINAMATH_CALUDE_bill_donut_purchase_l3966_396618


namespace NUMINAMATH_CALUDE_number_ordering_l3966_396682

theorem number_ordering : (2 : ℝ)^27 < 10^9 ∧ 10^9 < 5^13 := by sorry

end NUMINAMATH_CALUDE_number_ordering_l3966_396682


namespace NUMINAMATH_CALUDE_equal_interest_rates_l3966_396633

/-- Proves that given two accounts with equal investments, if one account has an interest rate of 10%
    and both accounts earn the same interest at the end of the year, then the interest rate of the
    other account is also 10%. -/
theorem equal_interest_rates
  (investment : ℝ)
  (rate1 rate2 : ℝ)
  (h1 : investment > 0)
  (h2 : rate2 = 0.1)
  (h3 : investment * rate1 = investment * rate2) :
  rate1 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_equal_interest_rates_l3966_396633


namespace NUMINAMATH_CALUDE_perimeter_formula_and_maximum_l3966_396650

noncomputable section

open Real

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  BC : ℝ
  x : ℝ  -- Angle B
  y : ℝ  -- Perimeter
  h_A : A = π / 3
  h_BC : BC = 2 * sqrt 3
  h_x_pos : x > 0
  h_x_upper : x < 2 * π / 3

/-- Perimeter function -/
def perimeter (t : Triangle) : ℝ := 6 * sin t.x + 2 * sqrt 3 * cos t.x + 2 * sqrt 3

theorem perimeter_formula_and_maximum (t : Triangle) :
  t.y = perimeter t ∧ t.y ≤ 6 * sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_perimeter_formula_and_maximum_l3966_396650


namespace NUMINAMATH_CALUDE_parabola_symmetry_problem_l3966_396677

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * x^2

/-- The problem statement -/
theorem parabola_symmetry_problem (A B : ParabolaPoint) (m : ℝ) 
  (h_symmetric : ∃ (t : ℝ), (A.x + B.x) / 2 = t ∧ (A.y + B.y) / 2 = t + m)
  (h_product : A.x * B.x = -1/2) :
  m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_problem_l3966_396677


namespace NUMINAMATH_CALUDE_A_intersect_B_l3966_396692

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | x < 3}

theorem A_intersect_B : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3966_396692


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_21_l3966_396631

theorem smallest_multiple_of_5_and_21 : ∃ b : ℕ+, 
  (∀ k : ℕ+, 5 ∣ k ∧ 21 ∣ k → b ≤ k) ∧ 5 ∣ b ∧ 21 ∣ b ∧ b = 105 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_21_l3966_396631


namespace NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3966_396679

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3966_396679


namespace NUMINAMATH_CALUDE_baker_theorem_l3966_396601

def baker_problem (initial_cakes sold_cakes additional_cakes : ℕ) : Prop :=
  initial_cakes - sold_cakes + additional_cakes = 111

theorem baker_theorem : baker_problem 110 75 76 := by
  sorry

end NUMINAMATH_CALUDE_baker_theorem_l3966_396601


namespace NUMINAMATH_CALUDE_function_inequality_l3966_396613

theorem function_inequality 
  (f : Real → Real) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) 
  (h_ineq : ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → 
    (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) 
  (u v w : Real) 
  (h_order : 0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1) : 
  ((w - v) / (w - u)) * f u + ((v - u) / (w - u)) * f w ≤ f v + 2 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3966_396613


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_of_binomial_expansion_l3966_396629

theorem fourth_term_coefficient_of_binomial_expansion :
  let n : ℕ := 7
  let k : ℕ := 3
  let coef : ℕ := n.choose k * 2^k
  coef = 280 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_of_binomial_expansion_l3966_396629


namespace NUMINAMATH_CALUDE_new_student_weight_l3966_396696

theorem new_student_weight (n : ℕ) (w_avg : ℝ) (w_new_avg : ℝ) (w_new : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new_avg = 27.1 →
  n * w_avg + w_new = (n + 1) * w_new_avg →
  w_new = 1 := by
sorry

end NUMINAMATH_CALUDE_new_student_weight_l3966_396696


namespace NUMINAMATH_CALUDE_sum_of_roots_l3966_396698

theorem sum_of_roots (x : ℝ) : 
  (x^2 + 2023*x = 2024) → 
  (∃ y : ℝ, y^2 + 2023*y = 2024 ∧ x + y = -2023) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3966_396698


namespace NUMINAMATH_CALUDE_binomial_divisibility_l3966_396651

theorem binomial_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  p^2 ∣ (Nat.choose (2*p - 1) (p - 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l3966_396651


namespace NUMINAMATH_CALUDE_a_6_equals_11_l3966_396675

/-- Given a sequence {aₙ} where Sₙ is the sum of its first n terms -/
def S (n : ℕ) : ℕ := n^2 + 1

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Proof that the 6th term of the sequence is 11 -/
theorem a_6_equals_11 : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_a_6_equals_11_l3966_396675


namespace NUMINAMATH_CALUDE_expression_equals_two_l3966_396697

theorem expression_equals_two : 2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3966_396697


namespace NUMINAMATH_CALUDE_r_plus_s_value_l3966_396634

/-- The line equation y = -5/3 * x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is where the line crosses the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line crosses the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- T is a point on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 4 * abs ((P.1 * s - r * P.2) / 2)

/-- Main theorem: Given the conditions, r + s = 10.5 -/
theorem r_plus_s_value (r s : ℝ) 
  (h1 : line_equation r s) 
  (h2 : T_on_PQ r s) 
  (h3 : area_condition r s) : 
  r + s = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_r_plus_s_value_l3966_396634


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3966_396625

/-- A quadratic equation kx^2 + 3x - 1 = 0 has real roots if and only if k ≥ -9/4 and k ≠ 0 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9/4 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3966_396625


namespace NUMINAMATH_CALUDE_not_p_and_not_q_is_false_l3966_396670

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 > 0

-- Theorem statement
theorem not_p_and_not_q_is_false : ¬(¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_is_false_l3966_396670


namespace NUMINAMATH_CALUDE_percentage_of_filled_seats_l3966_396643

/-- Given a hall with 600 seats and 240 vacant seats, prove that 60% of the seats were filled. -/
theorem percentage_of_filled_seats (total_seats : ℕ) (vacant_seats : ℕ) : 
  total_seats = 600 → vacant_seats = 240 → 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 = 60) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_filled_seats_l3966_396643


namespace NUMINAMATH_CALUDE_expression_value_l3966_396622

theorem expression_value (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3966_396622


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3966_396662

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3966_396662


namespace NUMINAMATH_CALUDE_team_formations_count_l3966_396615

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to form a team of 3 teachers from 4 female and 5 male teachers,
    with the condition that the team must include both male and female teachers -/
def teamFormations : ℕ :=
  choose 5 1 * choose 4 2 + choose 5 2 * choose 4 1

theorem team_formations_count :
  teamFormations = 70 := by sorry

end NUMINAMATH_CALUDE_team_formations_count_l3966_396615


namespace NUMINAMATH_CALUDE_total_bottles_l3966_396654

theorem total_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 9) (h2 : diet_soda = 8) : 
  regular_soda + diet_soda = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_l3966_396654


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l3966_396684

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ (N.image Prod.fst) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l3966_396684


namespace NUMINAMATH_CALUDE_bedroom_set_final_price_l3966_396619

def original_price : ℝ := 2000
def gift_cards : ℝ := 200
def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10

def final_price : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  price_after_second_discount - gift_cards

theorem bedroom_set_final_price :
  final_price = 1330 := by sorry

end NUMINAMATH_CALUDE_bedroom_set_final_price_l3966_396619


namespace NUMINAMATH_CALUDE_parabola_c_value_l3966_396627

/-- A parabola passing through two specific points has a unique c-value -/
theorem parabola_c_value (b : ℝ) :
  ∃! c : ℝ, (2^2 + 2*b + c = 20) ∧ ((-2)^2 + (-2)*b + c = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3966_396627


namespace NUMINAMATH_CALUDE_inequality_proof_l3966_396699

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3966_396699


namespace NUMINAMATH_CALUDE_robe_savings_l3966_396683

def repair_cost : ℕ := 10
def initial_savings : ℕ := 630

def corner_light_cost (repair : ℕ) : ℕ := 2 * repair
def brake_disk_cost (light : ℕ) : ℕ := 3 * light

def total_expenses (repair : ℕ) : ℕ :=
  repair + corner_light_cost repair + 2 * brake_disk_cost (corner_light_cost repair)

theorem robe_savings : 
  initial_savings + total_expenses repair_cost = 780 :=
sorry

end NUMINAMATH_CALUDE_robe_savings_l3966_396683


namespace NUMINAMATH_CALUDE_fraction_2021_2019_position_l3966_396612

def sequence_position (m n : ℕ) : ℕ :=
  let k := m + n
  let previous_terms := (k - 1) * (k - 2) / 2
  let current_group_position := m
  previous_terms + current_group_position

theorem fraction_2021_2019_position :
  sequence_position 2021 2019 = 8159741 :=
by sorry

end NUMINAMATH_CALUDE_fraction_2021_2019_position_l3966_396612


namespace NUMINAMATH_CALUDE_blue_balls_count_l3966_396678

theorem blue_balls_count (total : ℕ) (green blue yellow white : ℕ) : 
  green = total / 4 →
  blue = total / 8 →
  yellow = total / 12 →
  white = 26 →
  total = green + blue + yellow + white →
  blue = 6 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l3966_396678


namespace NUMINAMATH_CALUDE_fifteen_percent_of_900_is_135_l3966_396610

theorem fifteen_percent_of_900_is_135 : ∃ x : ℝ, x * 0.15 = 135 ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_900_is_135_l3966_396610


namespace NUMINAMATH_CALUDE_club_female_count_l3966_396687

theorem club_female_count (total : ℕ) (difference : ℕ) (female : ℕ) : 
  total = 82 →
  difference = 6 →
  female = total / 2 + difference / 2 →
  female = 44 := by
sorry

end NUMINAMATH_CALUDE_club_female_count_l3966_396687


namespace NUMINAMATH_CALUDE_x_minus_y_equals_18_l3966_396632

theorem x_minus_y_equals_18 (x y : ℤ) (h1 : x + y = 10) (h2 : x = 14) : x - y = 18 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_18_l3966_396632


namespace NUMINAMATH_CALUDE_quadrilateral_angle_proof_l3966_396637

theorem quadrilateral_angle_proof (A B C D : ℝ) : 
  A + B = 180 →
  C = D →
  A = 85 →
  B + C + D = 180 →
  D = 42.5 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_proof_l3966_396637


namespace NUMINAMATH_CALUDE_female_workers_count_l3966_396688

/-- Represents the number of workers of each type and their wages --/
structure WorkforceData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the total daily wage for all workers --/
def total_daily_wage (data : WorkforceData) : ℕ :=
  data.male_workers * data.male_wage +
  data.female_workers * data.female_wage +
  data.child_workers * data.child_wage

/-- Calculates the total number of workers --/
def total_workers (data : WorkforceData) : ℕ :=
  data.male_workers + data.female_workers + data.child_workers

/-- Theorem stating that the number of female workers is 15 --/
theorem female_workers_count (data : WorkforceData)
  (h1 : data.male_workers = 20)
  (h2 : data.child_workers = 5)
  (h3 : data.male_wage = 25)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 21)
  (h7 : (total_daily_wage data) / (total_workers data) = data.average_wage) :
  data.female_workers = 15 :=
sorry

end NUMINAMATH_CALUDE_female_workers_count_l3966_396688


namespace NUMINAMATH_CALUDE_negative_abs_equals_opposite_l3966_396689

theorem negative_abs_equals_opposite (x : ℝ) : x < 0 → |x| = -x := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_equals_opposite_l3966_396689


namespace NUMINAMATH_CALUDE_right_triangle_rotation_volumes_l3966_396623

theorem right_triangle_rotation_volumes 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (K₁ K₂ K₃ : ℝ),
    K₁ = (2/3) * a * b^2 * Real.pi ∧
    K₂ = (2/3) * a^2 * b * Real.pi ∧
    K₃ = (2/3) * (a^2 * b^2) / c * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_volumes_l3966_396623


namespace NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_condition_b_implies_parallelogram_l3966_396648

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

/-- Definition of a parallelogram --/
def is_parallelogram {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.A - q.D = q.B - q.C

/-- Theorem: If opposite sides of a quadrilateral are equal, it is a parallelogram --/
theorem opposite_sides_equal_implies_parallelogram 
  {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  q.A - q.D = q.B - q.C → q.A - q.B = q.D - q.C → is_parallelogram q :=
by sorry

/-- Main theorem: If AD=BC and AB=DC, then ABCD is a parallelogram --/
theorem condition_b_implies_parallelogram 
  {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  q.A - q.D = q.B - q.C → q.A - q.B = q.D - q.C → is_parallelogram q :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_condition_b_implies_parallelogram_l3966_396648


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3966_396639

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 7 * x - y - 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3966_396639


namespace NUMINAMATH_CALUDE_cubic_expression_factorization_l3966_396673

theorem cubic_expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_factorization_l3966_396673


namespace NUMINAMATH_CALUDE_ratio_between_zero_and_one_l3966_396686

theorem ratio_between_zero_and_one : 
  let A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
  let B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by sorry

end NUMINAMATH_CALUDE_ratio_between_zero_and_one_l3966_396686


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_range_l3966_396658

theorem function_inequality_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 6 ≥ 0) → m ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_range_l3966_396658


namespace NUMINAMATH_CALUDE_betty_age_l3966_396617

/-- Given the ages of Albert, Mary, and Betty, prove that Betty is 4 years old. -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 8) : 
  betty = 4 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l3966_396617


namespace NUMINAMATH_CALUDE_remaining_tickets_l3966_396669

def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

theorem remaining_tickets :
  tickets_from_whack_a_mole + tickets_from_skee_ball - tickets_spent_on_hat = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_l3966_396669


namespace NUMINAMATH_CALUDE_square_root_calculations_l3966_396646

theorem square_root_calculations : 
  (Real.sqrt 3)^2 = 3 ∧ Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculations_l3966_396646


namespace NUMINAMATH_CALUDE_delta_fourth_order_zero_l3966_396660

def u (n : ℕ) : ℕ := n^3 + 2*n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
| f => λ n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
| 0 => id
| k + 1 => Δ ∘ iteratedΔ k

theorem delta_fourth_order_zero (n : ℕ) : 
  ∀ k : ℕ, (∀ n : ℕ, iteratedΔ k u n = 0) ↔ k ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_delta_fourth_order_zero_l3966_396660


namespace NUMINAMATH_CALUDE_negation_equivalence_l3966_396624

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3966_396624


namespace NUMINAMATH_CALUDE_common_chord_equation_l3966_396630

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : (ℝ × ℝ) → Prop
  circle2 : (ℝ × ℝ) → Prop

/-- The equation of a line in a plane -/
structure Line where
  equation : (ℝ × ℝ) → Prop

/-- The common chord of two intersecting circles -/
def commonChord (circles : TwoCircles) : Line :=
  sorry

/-- Combining equations of two circles -/
def combineEquations (circles : TwoCircles) : (ℝ × ℝ) → Prop :=
  sorry

/-- Eliminating quadratic terms from an equation -/
def eliminateQuadraticTerms (eq : (ℝ × ℝ) → Prop) : (ℝ × ℝ) → Prop :=
  sorry

/-- Theorem: The equation of the common chord of two intersecting circles
    is obtained by eliminating the quadratic terms after combining
    the equations of the two circles -/
theorem common_chord_equation (circles : TwoCircles) :
  (commonChord circles).equation =
  eliminateQuadraticTerms (combineEquations circles) :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3966_396630


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3966_396647

theorem line_inclination_angle (x y : ℝ) :
  x + Real.sqrt 3 * y - 1 = 0 →
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧ Real.tan θ = -1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3966_396647


namespace NUMINAMATH_CALUDE_goods_lost_percentage_l3966_396676

/-- Calculates the percentage of goods lost during theft given the profit percentage and loss percentage -/
theorem goods_lost_percentage (profit_percent : ℝ) (loss_percent : ℝ) : 
  profit_percent = 10 → loss_percent = 34 → 
  (100 + profit_percent) * (100 - loss_percent) / 100 = 66 * (100 + profit_percent) / 100 := by
  sorry

#check goods_lost_percentage

end NUMINAMATH_CALUDE_goods_lost_percentage_l3966_396676


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3966_396600

-- First expression
theorem simplify_expression_1 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 0) :
  (4 * x^2) / (x^2 - y^2) / (x / (x + y)) = 4 * x / (x - y) := by sorry

-- Second expression
theorem simplify_expression_2 (m : ℝ) (h : m ≠ 1) :
  m / (m - 1) - 1 = 1 / (m - 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3966_396600


namespace NUMINAMATH_CALUDE_convex_quad_interior_point_inequality_l3966_396672

/-- A convex quadrilateral with an interior point and parallel lines -/
structure ConvexQuadWithInteriorPoint where
  /-- The area of the convex quadrilateral ABCD -/
  T : ℝ
  /-- The area of quadrilateral AEPH -/
  t₁ : ℝ
  /-- The area of quadrilateral PFCG -/
  t₂ : ℝ
  /-- The areas are non-negative -/
  h₁ : 0 ≤ T
  h₂ : 0 ≤ t₁
  h₃ : 0 ≤ t₂

/-- The inequality holds for any convex quadrilateral with an interior point -/
theorem convex_quad_interior_point_inequality (q : ConvexQuadWithInteriorPoint) :
  Real.sqrt q.t₁ + Real.sqrt q.t₂ ≤ Real.sqrt q.T :=
sorry

end NUMINAMATH_CALUDE_convex_quad_interior_point_inequality_l3966_396672


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3966_396681

theorem inequality_equivalence (x y : ℝ) : 
  (y - 2*x < Real.sqrt (4*x^2 - 4*x + 1)) ↔ 
  ((x < 1/2 ∧ y < 1) ∨ (x ≥ 1/2 ∧ y < 4*x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3966_396681


namespace NUMINAMATH_CALUDE_smallest_n_congruent_to_neg_2023_mod_9_l3966_396645

theorem smallest_n_congruent_to_neg_2023_mod_9 : 
  ∃ n : ℕ, 
    (4 ≤ n ∧ n ≤ 12) ∧ 
    n ≡ -2023 [ZMOD 9] ∧
    (∀ m : ℕ, (4 ≤ m ∧ m ≤ 12) ∧ m ≡ -2023 [ZMOD 9] → n ≤ m) ∧
    n = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruent_to_neg_2023_mod_9_l3966_396645


namespace NUMINAMATH_CALUDE_total_amount_in_euros_l3966_396636

/-- Represents the distribution of shares among w, x, y, and z -/
structure ShareDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the exchange rate from dollars to euros -/
def exchange_rate : ℝ := 0.85

/-- Defines the share ratios relative to w -/
def share_ratios : ShareDistribution := {
  w := 1,
  x := 0.75,
  y := 0.5,
  z := 0.25
}

/-- Theorem stating the total amount in euros given the conditions -/
theorem total_amount_in_euros : 
  ∀ (shares : ShareDistribution),
  shares.w * exchange_rate = 15 →
  (shares.w + shares.x + shares.y + shares.z) * exchange_rate = 37.5 :=
by
  sorry

#check total_amount_in_euros

end NUMINAMATH_CALUDE_total_amount_in_euros_l3966_396636


namespace NUMINAMATH_CALUDE_min_value_of_m_l3966_396626

theorem min_value_of_m (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - x^2 - x + (-(a^3) + a^2 + a) = (x - a) * (x - b) * (x - c)) →
  ∀ m : ℝ, (∀ x : ℝ, x^3 - x^2 - x + m = (x - a) * (x - b) * (x - c)) → 
  m ≥ -5/27 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_m_l3966_396626


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sides_tangent_product_l3966_396642

/-- 
For a triangle with sides forming an arithmetic sequence, 
the product of 3 and the tangents of half the smallest and largest angles equals 1.
-/
theorem triangle_arithmetic_sides_tangent_product (a b c : ℝ) (α β γ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- angles are positive
  α + β + γ = π →  -- sum of angles in a triangle
  a + c = 2 * b →  -- arithmetic sequence condition
  α ≤ β ∧ β ≤ γ →  -- α is smallest, γ is largest
  3 * Real.tan (α / 2) * Real.tan (γ / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sides_tangent_product_l3966_396642


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3966_396638

/-- Given a polar coordinate equation r = 3, prove it represents a circle with radius 3 centered at the origin in Cartesian coordinates. -/
theorem polar_to_cartesian_circle (x y : ℝ) : 
  (∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = 3 * Real.sin θ) ↔ x^2 + y^2 = 9 := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3966_396638


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l3966_396607

theorem greatest_prime_factor_of_210 : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ 210 ∧ ∀ (q : ℕ), q.Prime → q ∣ 210 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l3966_396607


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l3966_396674

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ digit_sum n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ digit_sum m = 27 → m ≤ n :=
by
  use 999
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l3966_396674


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3966_396611

theorem polynomial_coefficient_sum :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (∀ x : ℝ, x + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                     a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                     a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a + a₂ + a₃ + a₄ + a₅ + a₆ + a₈ = 510 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3966_396611


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3966_396664

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
    (f a + f b) * (f c + f d) = (a + b) * (c + d)

/-- The main theorem stating that any function satisfying the equation
    must be either the identity function or its reciprocal -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, x > 0 → f x > 0) 
    (heq : SatisfiesEquation f) :
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3966_396664


namespace NUMINAMATH_CALUDE_range_of_a_l3966_396680

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = B a) → a < -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3966_396680


namespace NUMINAMATH_CALUDE_smallest_y_squared_value_l3966_396652

/-- Represents an isosceles trapezoid EFGH with a tangent circle -/
structure IsoscelesTrapezoidWithTangentCircle where
  EF : ℝ
  GH : ℝ
  y : ℝ
  is_isosceles : EF > GH
  tangent_circle : Bool

/-- The smallest possible value of y^2 in the given configuration -/
def smallest_y_squared (t : IsoscelesTrapezoidWithTangentCircle) : ℝ := sorry

/-- Theorem stating the smallest possible value of y^2 -/
theorem smallest_y_squared_value 
  (t : IsoscelesTrapezoidWithTangentCircle) 
  (h1 : t.EF = 102) 
  (h2 : t.GH = 26) 
  (h3 : t.tangent_circle = true) : 
  smallest_y_squared t = 1938 := by sorry

end NUMINAMATH_CALUDE_smallest_y_squared_value_l3966_396652


namespace NUMINAMATH_CALUDE_roller_skate_attendance_l3966_396603

/-- The number of wheels on the floor when all people skated -/
def total_wheels : ℕ := 320

/-- The number of roller skates per person -/
def skates_per_person : ℕ := 2

/-- The number of wheels per roller skate -/
def wheels_per_skate : ℕ := 2

/-- The number of people who showed up to roller skate -/
def num_people : ℕ := total_wheels / (skates_per_person * wheels_per_skate)

theorem roller_skate_attendance : num_people = 80 := by
  sorry

end NUMINAMATH_CALUDE_roller_skate_attendance_l3966_396603


namespace NUMINAMATH_CALUDE_negation_equivalence_l3966_396649

theorem negation_equivalence (x : ℝ) :
  ¬(x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3966_396649


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l3966_396609

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := Complex.mk (3*m - 2) (m - 1)

/-- z is purely imaginary if and only if m = 2/3 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 2/3 := by
  sorry

/-- z lies in the fourth quadrant if and only if 2/3 < m < 1 -/
theorem z_in_fourth_quadrant (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (2/3 < m ∧ m < 1) := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l3966_396609


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l3966_396655

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a - 2 * i) * i^2013 = b - i →
  a^2 + b^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l3966_396655


namespace NUMINAMATH_CALUDE_knights_round_table_l3966_396628

theorem knights_round_table (n : ℕ) (h : n = 25) :
  let total_arrangements := n * (n + 1) * (n + 2) / 6
  let non_adjacent_arrangements := n * (n - 3) * (n - 4) / 2
  (total_arrangements - non_adjacent_arrangements : ℚ) / total_arrangements = 11 / 46 :=
by sorry

end NUMINAMATH_CALUDE_knights_round_table_l3966_396628


namespace NUMINAMATH_CALUDE_origin_is_solution_l3966_396604

/-- The equation defining the set of points -/
def equation (x y : ℝ) : Prop :=
  x^2 * (y + y^2) = y^3 + x^4

/-- Theorem stating that (0, 0) is a solution to the equation -/
theorem origin_is_solution : equation 0 0 := by
  sorry

end NUMINAMATH_CALUDE_origin_is_solution_l3966_396604


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3966_396667

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ 2 ∧ (x^3 - 3*x^2)/(x^2 - 4*x + 4) + x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3966_396667


namespace NUMINAMATH_CALUDE_cake_after_four_trips_l3966_396621

/-- The fraction of cake remaining after a given number of trips to the pantry -/
def cakeRemaining (trips : ℕ) : ℚ :=
  (1 : ℚ) / 2^trips

/-- The theorem stating that after 4 trips, 1/16 of the cake remains -/
theorem cake_after_four_trips :
  cakeRemaining 4 = (1 : ℚ) / 16 := by
  sorry

#eval cakeRemaining 4

end NUMINAMATH_CALUDE_cake_after_four_trips_l3966_396621


namespace NUMINAMATH_CALUDE_prob_different_suits_enlarged_deck_l3966_396690

/-- A deck of cards with five suits -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h1 : total_cards = num_suits * cards_per_suit)
  (h2 : num_suits = 5)

/-- The probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  (d.total_cards - d.cards_per_suit) / (d.total_cards - 1)

/-- The main theorem -/
theorem prob_different_suits_enlarged_deck :
  ∃ d : Deck, d.total_cards = 65 ∧ prob_different_suits d = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_enlarged_deck_l3966_396690


namespace NUMINAMATH_CALUDE_diagram3_illustrates_inflation_l3966_396671

/-- Represents a diagram showing economic data over time -/
structure EconomicDiagram where
  prices : ℕ → ℝ
  time : ℕ

/-- Definition of inflation -/
def is_inflation (d : EconomicDiagram) : Prop :=
  ∀ t₁ t₂, t₁ < t₂ → d.prices t₁ < d.prices t₂

/-- Diagram №3 from the problem -/
def diagram3 : EconomicDiagram :=
  sorry

/-- Theorem stating that Diagram №3 illustrates inflation -/
theorem diagram3_illustrates_inflation : is_inflation diagram3 := by
  sorry

end NUMINAMATH_CALUDE_diagram3_illustrates_inflation_l3966_396671


namespace NUMINAMATH_CALUDE_max_a_value_l3966_396657

/-- Given integers a and b satisfying the conditions, the maximum value of a is 23 -/
theorem max_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a ≤ 23 ∧ ∃ (a₀ b₀ : ℤ), a₀ > b₀ ∧ b₀ > 0 ∧ a₀ + b₀ + a₀ * b₀ = 143 ∧ a₀ = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l3966_396657
