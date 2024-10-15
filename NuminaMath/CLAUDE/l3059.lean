import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3059_305987

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3059_305987


namespace NUMINAMATH_CALUDE_mod_seventeen_problem_l3059_305996

theorem mod_seventeen_problem (n : ℕ) (h1 : n < 17) (h2 : (2 * n) % 17 = 1) :
  (3^n)^2 % 17 - 3 % 17 = 13 % 17 := by
  sorry

end NUMINAMATH_CALUDE_mod_seventeen_problem_l3059_305996


namespace NUMINAMATH_CALUDE_log_equality_l3059_305955

theorem log_equality : Real.log 16 / Real.log 4096 = Real.log 4 / Real.log 64 := by sorry

end NUMINAMATH_CALUDE_log_equality_l3059_305955


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3059_305921

theorem smaller_number_proof (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : min a b = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3059_305921


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l3059_305981

theorem solution_set_reciprocal_inequality (x : ℝ) :
  {x : ℝ | 1 / x > 3} = Set.Ioo 0 (1 / 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l3059_305981


namespace NUMINAMATH_CALUDE_total_money_l3059_305983

theorem total_money (brad_money : ℚ) (josh_money : ℚ) (doug_money : ℚ) : 
  josh_money = 2 * brad_money →
  josh_money = (3 / 4) * doug_money →
  doug_money = 32 →
  brad_money + josh_money + doug_money = 68 := by
sorry

end NUMINAMATH_CALUDE_total_money_l3059_305983


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l3059_305911

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def chemistry_marks : ℕ := 97
def biology_marks : ℕ := 95
def average_marks : ℚ := 93
def num_subjects : ℕ := 5

theorem physics_marks_calculation :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 82 :=
by sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l3059_305911


namespace NUMINAMATH_CALUDE_birds_in_tree_l3059_305976

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29) 
  (h2 : final_birds = 42) : 
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3059_305976


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_l3059_305963

/-- Three circles inscribed in a corner -/
structure InscribedCircles where
  r : ℝ  -- radius of small circle
  a : ℝ  -- distance from center of small circle to corner vertex
  x : ℝ  -- radius of medium circle
  y : ℝ  -- radius of large circle

/-- Conditions for the inscribed circles -/
def valid_inscribed_circles (c : InscribedCircles) : Prop :=
  c.r > 0 ∧ c.a > c.r ∧ c.x > c.r ∧ c.y > c.x

/-- Theorem stating the radii of medium and large circles -/
theorem inscribed_circles_radii (c : InscribedCircles) 
  (h : valid_inscribed_circles c) : 
  c.x = c.a * c.r / (c.a - c.r) ∧ 
  c.y = c.a^2 * c.r / (c.a - c.r)^2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_l3059_305963


namespace NUMINAMATH_CALUDE_triangle_properties_l3059_305914

/-- Triangle represented by three points in 2D space -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Calculate the squared distance between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Check if a triangle is a right triangle -/
def isRightTriangle (t : Triangle) : Prop :=
  let a := distanceSquared t.p1 t.p2
  let b := distanceSquared t.p2 t.p3
  let c := distanceSquared t.p3 t.p1
  (a + b = c) ∨ (b + c = a) ∨ (c + a = b)

/-- Triangle A -/
def triangleA : Triangle :=
  { p1 := (0, 0), p2 := (3, 4), p3 := (0, 8) }

/-- Triangle B -/
def triangleB : Triangle :=
  { p1 := (3, 4), p2 := (10, 4), p3 := (3, 0) }

theorem triangle_properties :
  ¬(isRightTriangle triangleA) ∧
  (isRightTriangle triangleB) ∧
  (distanceSquared triangleB.p1 triangleB.p2 = 65 ∨
   distanceSquared triangleB.p2 triangleB.p3 = 65 ∨
   distanceSquared triangleB.p3 triangleB.p1 = 65) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3059_305914


namespace NUMINAMATH_CALUDE_donny_apple_purchase_cost_l3059_305915

-- Define the prices of apples
def small_apple_price : ℚ := 1.5
def medium_apple_price : ℚ := 2
def big_apple_price : ℚ := 3

-- Define the number of apples Donny bought
def small_apples_bought : ℕ := 6
def medium_apples_bought : ℕ := 6
def big_apples_bought : ℕ := 8

-- Calculate the total cost
def total_cost : ℚ := 
  small_apple_price * small_apples_bought +
  medium_apple_price * medium_apples_bought +
  big_apple_price * big_apples_bought

-- Theorem statement
theorem donny_apple_purchase_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_donny_apple_purchase_cost_l3059_305915


namespace NUMINAMATH_CALUDE_selling_price_is_180_l3059_305900

/-- Calculates the selling price per machine to break even -/
def selling_price_per_machine (cost_parts : ℕ) (cost_patent : ℕ) (num_machines : ℕ) : ℕ :=
  (cost_parts + cost_patent) / num_machines

/-- Theorem: The selling price per machine is $180 -/
theorem selling_price_is_180 :
  selling_price_per_machine 3600 4500 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_180_l3059_305900


namespace NUMINAMATH_CALUDE_painted_cubes_count_l3059_305951

/-- Represents the number of smaller cubes with a given number of painted faces -/
structure PaintedCubes :=
  (three : ℕ)
  (two : ℕ)
  (one : ℕ)

/-- Calculates the number of smaller cubes with different numbers of painted faces
    when a large cube is cut into smaller cubes -/
def countPaintedCubes (large_edge : ℕ) (small_edge : ℕ) : PaintedCubes :=
  sorry

/-- Theorem stating the correct number of painted smaller cubes for the given problem -/
theorem painted_cubes_count :
  countPaintedCubes 8 2 = PaintedCubes.mk 8 24 24 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l3059_305951


namespace NUMINAMATH_CALUDE_no_all_permutations_perfect_squares_l3059_305986

/-- A function that checks if a natural number has all non-zero digits -/
def allDigitsNonZero (n : ℕ) : Prop := sorry

/-- A function that generates all permutations of digits of a natural number -/
def digitPermutations (n : ℕ) : Set ℕ := sorry

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := sorry

theorem no_all_permutations_perfect_squares :
  ∀ n : ℕ, n ≥ 10 → allDigitsNonZero n →
    ∃ m ∈ digitPermutations n, ¬ isPerfectSquare m :=
sorry

end NUMINAMATH_CALUDE_no_all_permutations_perfect_squares_l3059_305986


namespace NUMINAMATH_CALUDE_waiter_earnings_proof_l3059_305994

/-- Calculates the waiter's earnings from tips given the total number of customers,
    number of non-tipping customers, and the tip amount from each tipping customer. -/
def waiterEarnings (totalCustomers nonTippingCustomers tipAmount : ℕ) : ℕ :=
  (totalCustomers - nonTippingCustomers) * tipAmount

/-- Proves that the waiter's earnings are $27 given the specific conditions -/
theorem waiter_earnings_proof :
  waiterEarnings 7 4 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_waiter_earnings_proof_l3059_305994


namespace NUMINAMATH_CALUDE_job_applicant_age_range_l3059_305998

/-- The maximum number of different integer ages within a range defined by
    an average age and a number of standard deviations. -/
def max_different_ages (average_age : ℕ) (std_dev : ℕ) (num_std_devs : ℕ) : ℕ :=
  2 * num_std_devs * std_dev + 1

/-- Theorem stating that for the given problem parameters, 
    the maximum number of different ages is 41. -/
theorem job_applicant_age_range : 
  max_different_ages 40 10 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_job_applicant_age_range_l3059_305998


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3059_305934

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3059_305934


namespace NUMINAMATH_CALUDE_negation_of_exists_ln_positive_l3059_305966

theorem negation_of_exists_ln_positive :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_ln_positive_l3059_305966


namespace NUMINAMATH_CALUDE_chord_bisected_at_P_l3059_305933

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop := 5*x - 3*y - 13 = 0

-- Theorem statement
theorem chord_bisected_at_P :
  ∀ (A B : ℝ × ℝ),
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  chord_equation A.1 A.2 →
  chord_equation B.1 B.2 →
  chord_equation P.1 P.2 →
  (A.1 + B.1) / 2 = P.1 ∧
  (A.2 + B.2) / 2 = P.2 :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_at_P_l3059_305933


namespace NUMINAMATH_CALUDE_math_exam_total_points_l3059_305901

/-- The total number of points in a math exam, given the scores of three students and the number of mistakes made by one of them. -/
theorem math_exam_total_points (bryan_score jen_score sammy_score : ℕ) (sammy_mistakes : ℕ) : 
  bryan_score = 20 →
  jen_score = bryan_score + 10 →
  sammy_score = jen_score - 2 →
  sammy_mistakes = 7 →
  bryan_score + (jen_score - bryan_score) + (sammy_score - jen_score + 2) + sammy_mistakes = 35 :=
by sorry

end NUMINAMATH_CALUDE_math_exam_total_points_l3059_305901


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3059_305969

def f (x : ℝ) : ℝ := 5*x^6 - 3*x^4 + 6*x^3 - 8*x + 10

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, f = λ x => (3*x - 9) * q x + 3550 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3059_305969


namespace NUMINAMATH_CALUDE_cups_per_girl_l3059_305947

/-- Given a class with students, boys, and girls, prove the number of cups each girl brought. -/
theorem cups_per_girl (total_students : ℕ) (num_boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ)
  (h1 : total_students = 30)
  (h2 : num_boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_cups = 90)
  (h5 : total_students = num_boys + 2 * num_boys) :
  (total_cups - num_boys * cups_per_boy) / (total_students - num_boys) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cups_per_girl_l3059_305947


namespace NUMINAMATH_CALUDE_trailing_zeros_of_nine_to_999_plus_one_l3059_305945

theorem trailing_zeros_of_nine_to_999_plus_one :
  ∃ n : ℕ, (9^999 + 1 : ℕ) = 10 * n ∧ (9^999 + 1 : ℕ) % 100 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_nine_to_999_plus_one_l3059_305945


namespace NUMINAMATH_CALUDE_triangle_properties_l3059_305944

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of the specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.B = π/3)
  (h3 : Real.cos t.A = 2 * Real.sqrt 7 / 7) :
  t.c = 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3059_305944


namespace NUMINAMATH_CALUDE_min_value_expression_l3059_305965

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1 / x^2 + 1 / y^2 + 1 / (x * y) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3059_305965


namespace NUMINAMATH_CALUDE_exponent_simplification_l3059_305973

theorem exponent_simplification (x : ℝ) : (x^5 * x^3) * x^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3059_305973


namespace NUMINAMATH_CALUDE_button_sequence_l3059_305917

theorem button_sequence (a : ℕ → ℕ) : 
  a 1 = 1 →                 -- First term is 1
  (∀ n : ℕ, a (n + 1) = 3 * a n) →  -- Common ratio is 3
  a 6 = 243 →               -- Sixth term is 243
  a 5 = 81 :=               -- Prove fifth term is 81
by sorry

end NUMINAMATH_CALUDE_button_sequence_l3059_305917


namespace NUMINAMATH_CALUDE_rob_has_five_nickels_l3059_305916

/-- Represents the number of coins of each type Rob has -/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents for a given CoinCount -/
def totalValueInCents (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Proves that Rob has 5 nickels given the conditions -/
theorem rob_has_five_nickels :
  ∃ (robsCoins : CoinCount),
    robsCoins.quarters = 7 ∧
    robsCoins.dimes = 3 ∧
    robsCoins.pennies = 12 ∧
    totalValueInCents robsCoins = 242 ∧
    robsCoins.nickels = 5 := by
  sorry


end NUMINAMATH_CALUDE_rob_has_five_nickels_l3059_305916


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_AB_l3059_305928

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume the circles intersect at A and B
axiom intersect_at_A : circle1 A.1 A.2 ∧ circle2 A.1 A.2
axiom intersect_at_B : circle1 B.1 B.2 ∧ circle2 B.1 B.2

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_AB :
  ∀ x y : ℝ, perpendicular_bisector x y ↔ 
  (x - A.1) * (B.1 - A.1) + (y - A.2) * (B.2 - A.2) = 0 ∧
  (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = 
  ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_AB_l3059_305928


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_l3059_305941

theorem triangle_side_ratio_sum (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : c^2 = a^2 + b^2 - a*b) :
  (a / (b + c) + b / (a + c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_l3059_305941


namespace NUMINAMATH_CALUDE_ravi_money_l3059_305940

theorem ravi_money (ravi giri kiran : ℚ) : 
  (ravi / giri = 6 / 7) →
  (giri / kiran = 6 / 15) →
  (kiran = 105) →
  ravi = 36 := by
sorry

end NUMINAMATH_CALUDE_ravi_money_l3059_305940


namespace NUMINAMATH_CALUDE_price_reduction_doubles_profit_l3059_305997

-- Define the initial conditions
def initial_purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_daily_sales : ℝ := 30
def sales_increase_per_yuan : ℝ := 3

-- Define the profit function
def profit (price_reduction : ℝ) : ℝ :=
  let new_price := initial_selling_price - price_reduction
  let new_sales := initial_daily_sales + sales_increase_per_yuan * price_reduction
  (new_price - initial_purchase_price) * new_sales

-- Theorem statement
theorem price_reduction_doubles_profit :
  ∃ (price_reduction : ℝ), 
    price_reduction = 30 ∧ 
    profit price_reduction = 2 * profit 0 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_doubles_profit_l3059_305997


namespace NUMINAMATH_CALUDE_tangent_fifteen_degree_ratio_l3059_305985

theorem tangent_fifteen_degree_ratio :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_fifteen_degree_ratio_l3059_305985


namespace NUMINAMATH_CALUDE_bridge_lamps_l3059_305939

/-- The number of lamps on a bridge -/
def numLamps (bridgeLength : ℕ) (lampSpacing : ℕ) : ℕ :=
  bridgeLength / lampSpacing + 1

theorem bridge_lamps :
  let bridgeLength : ℕ := 30
  let lampSpacing : ℕ := 5
  numLamps bridgeLength lampSpacing = 7 := by
  sorry

end NUMINAMATH_CALUDE_bridge_lamps_l3059_305939


namespace NUMINAMATH_CALUDE_unique_prime_sum_10003_l3059_305948

/-- A function that returns the number of ways to write a given natural number as the sum of two primes -/
def countPrimeSumWays (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there is exactly one way to write 10003 as the sum of two primes -/
theorem unique_prime_sum_10003 : countPrimeSumWays 10003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10003_l3059_305948


namespace NUMINAMATH_CALUDE_lcm_gcd_difference_nineteen_l3059_305912

theorem lcm_gcd_difference_nineteen (a b : ℕ+) :
  Nat.lcm a b - Nat.gcd a b = 19 →
  ((a = 1 ∧ b = 20) ∨ (a = 20 ∧ b = 1)) ∨
  ((a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4)) ∨
  ((a = 19 ∧ b = 38) ∨ (a = 38 ∧ b = 19)) :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_difference_nineteen_l3059_305912


namespace NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_lunch_box_l3059_305905

theorem min_students_with_blue_eyes_and_lunch_box
  (total_students : ℕ)
  (blue_eyes : ℕ)
  (lunch_box : ℕ)
  (h1 : total_students = 35)
  (h2 : blue_eyes = 20)
  (h3 : lunch_box = 22)
  : ∃ (both : ℕ), both ≥ 7 ∧ both ≤ min blue_eyes lunch_box :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_with_blue_eyes_and_lunch_box_l3059_305905


namespace NUMINAMATH_CALUDE_problem_solution_l3059_305972

theorem problem_solution (x : ℝ) : 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3059_305972


namespace NUMINAMATH_CALUDE_range_of_a_l3059_305927

open Set Real

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a}

-- Define propositions p and q
def p (a : ℝ) : Prop := (1 : ℝ) ∈ A a
def q (a : ℝ) : Prop := (2 : ℝ) ∈ A a

-- Theorem statement
theorem range_of_a (a : ℝ) 
  (h1 : a > 0) 
  (h2 : p a ∨ q a) 
  (h3 : ¬(p a ∧ q a)) : 
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3059_305927


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3059_305926

def f (x : ℝ) := x^4 - x

theorem tangent_point_coordinates :
  ∀ m n : ℝ,
  (∃ k : ℝ, f m = n ∧ 4 * m^3 - 1 = 3) →
  m = 1 ∧ n = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3059_305926


namespace NUMINAMATH_CALUDE_pentagonal_dodecahedron_properties_l3059_305906

/-- A polyhedron with pentagonal faces -/
structure PentagonalPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Euler's formula for polyhedra -/
axiom eulers_formula {p : PentagonalPolyhedron} : p.vertices - p.edges + p.faces = 2

/-- Each face is a pentagon -/
axiom pentagonal_faces {p : PentagonalPolyhedron} : p.edges * 2 = p.faces * 5

/-- Theorem: A polyhedron with 12 pentagonal faces has 30 edges and 20 vertices -/
theorem pentagonal_dodecahedron_properties :
  ∃ (p : PentagonalPolyhedron), p.faces = 12 ∧ p.edges = 30 ∧ p.vertices = 20 :=
sorry

end NUMINAMATH_CALUDE_pentagonal_dodecahedron_properties_l3059_305906


namespace NUMINAMATH_CALUDE_page_lines_increase_l3059_305980

theorem page_lines_increase (original_lines : ℕ) 
  (h1 : (110 : ℝ) = 0.8461538461538461 * original_lines) 
  (h2 : original_lines + 110 = 240) : 
  (original_lines + 110 : ℕ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l3059_305980


namespace NUMINAMATH_CALUDE_range_of_a_l3059_305964

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (m : ℝ),
    x₁^2 - m*x₁ - 1 = 0 ∧
    x₂^2 - m*x₂ - 1 = 0 ∧
    a^2 + 4*a - 3 ≤ |x₁ - x₂|

def q (a : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 + 2*x + a < 0

-- Define the theorem
theorem range_of_a :
  ∀ (a : ℝ), (p a ∨ q a) ∧ ¬(p a ∧ q a) → a = 1 ∨ a < -5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3059_305964


namespace NUMINAMATH_CALUDE_gmat_question_percentages_l3059_305960

theorem gmat_question_percentages
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 85)
  (h2 : neither_correct = 5)
  (h3 : both_correct = 60)
  : ∃ (second_correct : ℝ), second_correct = 70 :=
by sorry

end NUMINAMATH_CALUDE_gmat_question_percentages_l3059_305960


namespace NUMINAMATH_CALUDE_fourth_power_plus_64_solutions_l3059_305984

theorem fourth_power_plus_64_solutions :
  let solutions : Set ℂ := {2 + 2*I, -2 - 2*I, -2 + 2*I, 2 - 2*I}
  ∀ z : ℂ, z^4 + 64 = 0 ↔ z ∈ solutions :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_power_plus_64_solutions_l3059_305984


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l3059_305937

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (2*m + 3)*x + m^2

-- Define the condition for distinct real roots
def has_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

-- Part 1: Range of m
theorem range_of_m (m : ℝ) (h : has_distinct_real_roots m) : m > -3/4 := by
  sorry

-- Part 2: Value of m when 1/x₁ + 1/x₂ = 1
theorem value_of_m (m : ℝ) (h1 : has_distinct_real_roots m)
  (h2 : ∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ 1/x₁ + 1/x₂ = 1) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l3059_305937


namespace NUMINAMATH_CALUDE_red_lucky_stars_count_l3059_305992

theorem red_lucky_stars_count 
  (blue_count : ℕ) 
  (yellow_count : ℕ) 
  (red_count : ℕ) 
  (total_count : ℕ) 
  (pick_probability : ℚ) :
  blue_count = 20 →
  yellow_count = 15 →
  total_count = blue_count + yellow_count + red_count →
  pick_probability = 1/2 →
  (red_count : ℚ) / (total_count : ℚ) = pick_probability →
  red_count = 35 := by
sorry

end NUMINAMATH_CALUDE_red_lucky_stars_count_l3059_305992


namespace NUMINAMATH_CALUDE_purse_wallet_cost_difference_l3059_305974

theorem purse_wallet_cost_difference (wallet_cost purse_cost : ℕ) : 
  wallet_cost = 22 →
  purse_cost < 4 * wallet_cost →
  wallet_cost + purse_cost = 107 →
  4 * wallet_cost - purse_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_purse_wallet_cost_difference_l3059_305974


namespace NUMINAMATH_CALUDE_system_solutions_l3059_305919

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^4 + (7/2)*x^2*y + 2*y^3 = 0
def equation2 (x y : ℝ) : Prop := 4*x^2 + 7*x*y + 2*y^3 = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (2, -1), (-11/2, -11/2)}

-- Theorem stating that the solution set contains all and only solutions to the system
theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3059_305919


namespace NUMINAMATH_CALUDE_train_crossing_time_l3059_305923

/-- Proves the time it takes for a train to cross a signal post given its length and the time it takes to cross a bridge -/
theorem train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (bridge_crossing_time : ℝ) :
  train_length = 600 →
  bridge_length = 18000 →
  bridge_crossing_time = 1200 →
  (train_length / (bridge_length / bridge_crossing_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3059_305923


namespace NUMINAMATH_CALUDE_g_composition_of_three_l3059_305978

def g (n : ℤ) : ℤ :=
  if n < 5 then n^2 + 2*n - 1 else 2*n + 3

theorem g_composition_of_three : g (g (g 3)) = 65 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l3059_305978


namespace NUMINAMATH_CALUDE_high_school_student_count_l3059_305991

theorem high_school_student_count :
  let total_students : ℕ := 325
  let glasses_percentage : ℚ := 40 / 100
  let non_glasses_count : ℕ := 195
  (1 - glasses_percentage) * total_students = non_glasses_count :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_student_count_l3059_305991


namespace NUMINAMATH_CALUDE_ae_length_l3059_305967

-- Define the points
variable (A B C D E : Point)

-- Define the shapes
def is_isosceles_trapezoid (A B C E : Point) : Prop := sorry

def is_rectangle (A C D E : Point) : Prop := sorry

-- Define the lengths
def length (P Q : Point) : ℝ := sorry

-- State the theorem
theorem ae_length 
  (h1 : is_isosceles_trapezoid A B C E)
  (h2 : is_rectangle A C D E)
  (h3 : length A B = 10)
  (h4 : length E C = 20) :
  length A E = 20 := by sorry

end NUMINAMATH_CALUDE_ae_length_l3059_305967


namespace NUMINAMATH_CALUDE_cubic_three_roots_l3059_305959

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

/-- The derivative of f with respect to x -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The second derivative of f with respect to x -/
def f'' (x : ℝ) : ℝ := 6*x

/-- The value of f at x = 1 -/
def f_at_1 (a : ℝ) : ℝ := -2 - a

/-- The value of f at x = -1 -/
def f_at_neg_1 (a : ℝ) : ℝ := 2 - a

/-- Theorem: The cubic function f(x) = x^3 - 3x - a has three distinct real roots 
    if and only if a is in the open interval (-2, 2) -/
theorem cubic_three_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_l3059_305959


namespace NUMINAMATH_CALUDE_sum_of_special_sequence_l3059_305957

/-- Given positive real numbers a and b that form an arithmetic sequence with -2,
    and can also form a geometric sequence after rearrangement, prove their sum is 5 -/
theorem sum_of_special_sequence (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ d : ℝ, (a = b + d ∧ b = -2 + d) ∨ (b = a + d ∧ a = -2 + d) ∨ (a = -2 + d ∧ -2 = b + d)) →
  (∃ r : ℝ, r ≠ 0 ∧ ((a = b * r ∧ b = -2 * r) ∨ (b = a * r ∧ a = -2 * r) ∨ (a = -2 * r ∧ -2 = b * r))) →
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_sequence_l3059_305957


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3059_305908

theorem fraction_power_equality : (72000 ^ 4 : ℝ) / (24000 ^ 4) = 81 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3059_305908


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3059_305909

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 827 * n ≡ 1369 * n [ZMOD 36] ∧ ∀ (m : ℕ), m > 0 → 827 * m ≡ 1369 * m [ZMOD 36] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3059_305909


namespace NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l3059_305961

/-- A real quadratic trinomial -/
def QuadraticTrinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_roots_of_special_quadratic 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, QuadraticTrinomial a b c (x^3 + x) ≥ QuadraticTrinomial a b c (x^2 + 1)) →
  (-b / a = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l3059_305961


namespace NUMINAMATH_CALUDE_fourth_to_sixth_ratio_l3059_305949

structure MathClasses where
  fourth_level : ℕ
  sixth_level : ℕ
  seventh_level : ℕ
  total_students : ℕ

def MathClasses.valid (c : MathClasses) : Prop :=
  c.fourth_level = c.sixth_level ∧
  c.seventh_level = 2 * c.fourth_level ∧
  c.sixth_level = 40 ∧
  c.total_students = 520

theorem fourth_to_sixth_ratio (c : MathClasses) (h : c.valid) :
  c.fourth_level = c.sixth_level :=
by sorry

end NUMINAMATH_CALUDE_fourth_to_sixth_ratio_l3059_305949


namespace NUMINAMATH_CALUDE_contest_ranking_l3059_305993

theorem contest_ranking (A B C D : ℝ) 
  (sum_equal : A + B = C + D)
  (interchange : C + A > D + B)
  (bob_highest : B > A + D)
  (nonnegative : A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0) :
  B > A ∧ A > C ∧ C > D := by
  sorry

end NUMINAMATH_CALUDE_contest_ranking_l3059_305993


namespace NUMINAMATH_CALUDE_equation_solutions_l3059_305995

theorem equation_solutions :
  ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3059_305995


namespace NUMINAMATH_CALUDE_total_cookies_eq_eaten_plus_left_l3059_305979

/-- The number of cookies Mom made initially -/
def total_cookies : ℕ := sorry

/-- The number of cookies eaten by Julie and Matt -/
def cookies_eaten : ℕ := 9

/-- The number of cookies left after Julie and Matt ate -/
def cookies_left : ℕ := 23

/-- Theorem stating that the total number of cookies is the sum of eaten and left cookies -/
theorem total_cookies_eq_eaten_plus_left : 
  total_cookies = cookies_eaten + cookies_left := by sorry

end NUMINAMATH_CALUDE_total_cookies_eq_eaten_plus_left_l3059_305979


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l3059_305910

def f (x : ℝ) := x^2 - 2*x - 1

theorem sum_of_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l3059_305910


namespace NUMINAMATH_CALUDE_middle_school_eight_total_games_l3059_305918

/-- Represents a basketball conference -/
structure BasketballConference where
  numTeams : ℕ
  intraConferenceGamesPerPair : ℕ
  nonConferenceGamesPerTeam : ℕ

/-- Calculate the total number of games in a season for a given basketball conference -/
def totalGamesInSeason (conf : BasketballConference) : ℕ :=
  let intraConferenceGames := conf.numTeams.choose 2 * conf.intraConferenceGamesPerPair
  let nonConferenceGames := conf.numTeams * conf.nonConferenceGamesPerTeam
  intraConferenceGames + nonConferenceGames

/-- The "Middle School Eight" basketball conference -/
def middleSchoolEight : BasketballConference :=
  { numTeams := 8
  , intraConferenceGamesPerPair := 2
  , nonConferenceGamesPerTeam := 4 }

theorem middle_school_eight_total_games :
  totalGamesInSeason middleSchoolEight = 88 := by
  sorry


end NUMINAMATH_CALUDE_middle_school_eight_total_games_l3059_305918


namespace NUMINAMATH_CALUDE_hilt_garden_border_l3059_305922

/-- The number of rocks needed to complete the border -/
def total_rocks : ℕ := 125

/-- The number of rocks Mrs. Hilt already has -/
def current_rocks : ℕ := 64

/-- The number of additional rocks Mrs. Hilt needs -/
def additional_rocks : ℕ := total_rocks - current_rocks

theorem hilt_garden_border :
  additional_rocks = 61 := by sorry

end NUMINAMATH_CALUDE_hilt_garden_border_l3059_305922


namespace NUMINAMATH_CALUDE_fifth_term_is_nine_l3059_305935

-- Define the sequence and its sum
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem fifth_term_is_nine : a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_nine_l3059_305935


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l3059_305989

theorem fractional_exponent_simplification (a : ℝ) (ha : a > 0) :
  a^2 * Real.sqrt a = a^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l3059_305989


namespace NUMINAMATH_CALUDE_owner_short_percentage_l3059_305946

/-- Calculates the percentage of tank price the owner is short of after selling goldfish --/
def percentage_short_of_tank_price (goldfish_buy_price goldfish_sell_price tank_cost : ℚ) 
                                   (goldfish_sold : ℕ) : ℚ :=
  let profit_per_goldfish := goldfish_sell_price - goldfish_buy_price
  let total_profit := profit_per_goldfish * goldfish_sold
  let amount_short := tank_cost - total_profit
  (amount_short / tank_cost) * 100

/-- Proves that the owner is short of 45% of the tank price --/
theorem owner_short_percentage (goldfish_buy_price goldfish_sell_price tank_cost : ℚ) 
                               (goldfish_sold : ℕ) :
  goldfish_buy_price = 25/100 →
  goldfish_sell_price = 75/100 →
  tank_cost = 100 →
  goldfish_sold = 110 →
  percentage_short_of_tank_price goldfish_buy_price goldfish_sell_price tank_cost goldfish_sold = 45 :=
by
  sorry

#eval percentage_short_of_tank_price (25/100) (75/100) 100 110

end NUMINAMATH_CALUDE_owner_short_percentage_l3059_305946


namespace NUMINAMATH_CALUDE_cube_root_26_approximation_l3059_305932

theorem cube_root_26_approximation (ε : ℝ) (h : ε > 0) : 
  ∃ (x : ℝ), |x - (3 - 1/27)| < ε ∧ x^3 = 26 :=
sorry

end NUMINAMATH_CALUDE_cube_root_26_approximation_l3059_305932


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3059_305968

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3059_305968


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3059_305953

/-- An isosceles, obtuse triangle with one angle 80% larger than a right angle has two smallest angles measuring 9 degrees each. -/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  a = b →  -- isosceles condition
  c = 90 + 0.8 * 90 →  -- largest angle is 80% larger than right angle
  a = 9 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3059_305953


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3059_305962

theorem complex_equation_solution (z : ℂ) :
  z * Complex.I = 1 → z = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3059_305962


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3059_305907

theorem solution_set_quadratic_inequality :
  let S : Set ℝ := {x | x^2 + x - 2 ≥ 0}
  S = {x : ℝ | x ≤ -2 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3059_305907


namespace NUMINAMATH_CALUDE_polynomial_root_theorem_l3059_305988

theorem polynomial_root_theorem (a b : ℚ) :
  let f : ℝ → ℝ := fun x ↦ x^3 + a*x + b
  (f (4 - 2*Real.sqrt 5) = 0) →
  (∃ r : ℤ, f r = 0) →
  (∃ r : ℤ, f r = 0 ∧ r = -8) := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_theorem_l3059_305988


namespace NUMINAMATH_CALUDE_expected_weekly_rain_l3059_305903

/-- Represents the possible weather outcomes for a day -/
inductive Weather
  | Sun
  | Rain3
  | Rain8

/-- The probability distribution of weather outcomes -/
def weather_prob : Weather → ℝ
  | Weather.Sun => 0.3
  | Weather.Rain3 => 0.4
  | Weather.Rain8 => 0.3

/-- The amount of rain for each weather outcome -/
def rain_amount : Weather → ℝ
  | Weather.Sun => 0
  | Weather.Rain3 => 3
  | Weather.Rain8 => 8

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- Expected value of rain for a single day -/
def expected_daily_rain : ℝ :=
  (weather_prob Weather.Sun * rain_amount Weather.Sun) +
  (weather_prob Weather.Rain3 * rain_amount Weather.Rain3) +
  (weather_prob Weather.Rain8 * rain_amount Weather.Rain8)

/-- Theorem: The expected value of the total amount of rain for seven days is 25.2 inches -/
theorem expected_weekly_rain :
  (days_in_week : ℝ) * expected_daily_rain = 25.2 := by
  sorry


end NUMINAMATH_CALUDE_expected_weekly_rain_l3059_305903


namespace NUMINAMATH_CALUDE_choose_one_book_from_specific_shelf_l3059_305954

/-- Represents a bookshelf with Chinese books on the upper shelf and math books on the lower shelf -/
structure Bookshelf :=
  (chinese_books : ℕ)
  (math_books : ℕ)

/-- Calculates the number of ways to choose one book from the bookshelf -/
def ways_to_choose_one_book (shelf : Bookshelf) : ℕ :=
  shelf.chinese_books + shelf.math_books

/-- Theorem stating that for a bookshelf with 5 Chinese books and 4 math books,
    the number of ways to choose one book is 9 -/
theorem choose_one_book_from_specific_shelf :
  let shelf : Bookshelf := ⟨5, 4⟩
  ways_to_choose_one_book shelf = 9 := by sorry

end NUMINAMATH_CALUDE_choose_one_book_from_specific_shelf_l3059_305954


namespace NUMINAMATH_CALUDE_smaller_pack_size_l3059_305977

/-- Represents the number of eggs in a package -/
structure EggPackage where
  size : ℕ

/-- Represents a purchase of eggs -/
structure EggPurchase where
  totalEggs : ℕ
  largePacks : ℕ
  smallPacks : ℕ
  largePackSize : ℕ
  smallPackSize : ℕ

/-- Defines a valid egg purchase -/
def isValidPurchase (p : EggPurchase) : Prop :=
  p.totalEggs = p.largePacks * p.largePackSize + p.smallPacks * p.smallPackSize

/-- Theorem: Given the conditions, the size of the smaller pack must be 24 eggs -/
theorem smaller_pack_size (p : EggPurchase) 
    (h1 : p.totalEggs = 79)
    (h2 : p.largePacks = 5)
    (h3 : p.largePackSize = 11)
    (h4 : isValidPurchase p) :
    p.smallPackSize = 24 := by
  sorry

#check smaller_pack_size

end NUMINAMATH_CALUDE_smaller_pack_size_l3059_305977


namespace NUMINAMATH_CALUDE_complex_magnitude_l3059_305904

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 + Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3059_305904


namespace NUMINAMATH_CALUDE_line_AB_equation_l3059_305920

/-- Triangle ABC with given coordinates and line equations -/
structure Triangle where
  B : ℝ × ℝ
  C : ℝ × ℝ
  line_AC : ℝ → ℝ → ℝ
  altitude_A_AB : ℝ → ℝ → ℝ

/-- The equation of line AB in the given triangle -/
def line_AB (t : Triangle) : ℝ → ℝ → ℝ :=
  fun x y => 3 * (x - 3) - 2 * (y - 4)

/-- Theorem stating that the equation of line AB is correct -/
theorem line_AB_equation (t : Triangle) 
  (hB : t.B = (3, 4))
  (hC : t.C = (5, 2))
  (hAC : t.line_AC = fun x y => x - 4*y + 3)
  (hAlt : t.altitude_A_AB = fun x y => 2*x + 3*y - 16) :
  line_AB t = fun x y => 3 * (x - 3) - 2 * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_line_AB_equation_l3059_305920


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3059_305936

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  (2 * i / (1 - i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3059_305936


namespace NUMINAMATH_CALUDE_min_value_and_points_l3059_305929

theorem min_value_and_points (x y : ℝ) :
  (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 ≥ 1/6 ∧
  (∃ x y : ℝ, (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 = 1/6 ∧ 
   x = 5/2 ∧ y = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_points_l3059_305929


namespace NUMINAMATH_CALUDE_taehyung_current_age_l3059_305982

/-- Taehyung's age this year -/
def taehyung_age : ℕ := 9

/-- Taehyung's uncle's age this year -/
def uncle_age : ℕ := taehyung_age + 17

/-- The sum of Taehyung's and his uncle's ages four years later -/
def sum_ages_later : ℕ := (taehyung_age + 4) + (uncle_age + 4)

theorem taehyung_current_age :
  taehyung_age = 9 ∧ uncle_age = taehyung_age + 17 ∧ sum_ages_later = 43 :=
by sorry

end NUMINAMATH_CALUDE_taehyung_current_age_l3059_305982


namespace NUMINAMATH_CALUDE_line_perpendicular_theorem_l3059_305938

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_theorem
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : contained a α)
  (h4 : perpendicularLP b β)
  (h5 : parallel α β) :
  perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_theorem_l3059_305938


namespace NUMINAMATH_CALUDE_soda_cans_problem_l3059_305952

/-- The number of cans Tim initially had -/
def initial_cans : ℕ := 22

/-- The number of cans Jeff took -/
def cans_taken : ℕ := 6

/-- The number of cans Tim had after Jeff took some -/
def cans_after_taken : ℕ := initial_cans - cans_taken

/-- The number of cans Tim bought -/
def cans_bought : ℕ := cans_after_taken / 2

/-- The final number of cans Tim had -/
def final_cans : ℕ := 24

theorem soda_cans_problem :
  cans_after_taken + cans_bought = final_cans :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_problem_l3059_305952


namespace NUMINAMATH_CALUDE_sixth_root_unity_product_l3059_305924

theorem sixth_root_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_unity_product_l3059_305924


namespace NUMINAMATH_CALUDE_paula_tickets_needed_l3059_305943

/-- Represents the number of times Paula wants to ride each attraction -/
structure RideFrequencies where
  goKarts : Nat
  bumperCars : Nat
  rollerCoaster : Nat
  ferrisWheel : Nat

/-- Represents the ticket cost for each attraction -/
structure TicketCosts where
  goKarts : Nat
  bumperCars : Nat
  rollerCoaster : Nat
  ferrisWheel : Nat

/-- Calculates the total number of tickets needed based on ride frequencies and ticket costs -/
def totalTicketsNeeded (freq : RideFrequencies) (costs : TicketCosts) : Nat :=
  freq.goKarts * costs.goKarts +
  freq.bumperCars * costs.bumperCars +
  freq.rollerCoaster * costs.rollerCoaster +
  freq.ferrisWheel * costs.ferrisWheel

/-- Theorem stating that Paula needs 52 tickets in total -/
theorem paula_tickets_needed :
  let frequencies : RideFrequencies := {
    goKarts := 2,
    bumperCars := 4,
    rollerCoaster := 3,
    ferrisWheel := 1
  }
  let costs : TicketCosts := {
    goKarts := 4,
    bumperCars := 5,
    rollerCoaster := 7,
    ferrisWheel := 3
  }
  totalTicketsNeeded frequencies costs = 52 := by
  sorry

end NUMINAMATH_CALUDE_paula_tickets_needed_l3059_305943


namespace NUMINAMATH_CALUDE_whitewashing_cost_l3059_305958

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_height : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 3

theorem whitewashing_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := num_windows * (window_height * window_width)
  let whitewash_area := wall_area - door_area - window_area
  whitewash_area * cost_per_sqft = 2718 :=
by sorry

end NUMINAMATH_CALUDE_whitewashing_cost_l3059_305958


namespace NUMINAMATH_CALUDE_abs_nonnegative_rational_l3059_305930

theorem abs_nonnegative_rational (x : ℚ) : |x| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_nonnegative_rational_l3059_305930


namespace NUMINAMATH_CALUDE_equation_solutions_l3059_305925

theorem equation_solutions (x : ℝ) : 
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8 ↔ 
  x = 7 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3059_305925


namespace NUMINAMATH_CALUDE_kindergarten_group_divisibility_l3059_305971

theorem kindergarten_group_divisibility (n : ℕ) (a : ℕ) (h1 : n = 3 * a / 2) 
  (h2 : a % 2 = 0) (h3 : a % 4 = 0) : n % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_group_divisibility_l3059_305971


namespace NUMINAMATH_CALUDE_expression_evaluation_l3059_305956

theorem expression_evaluation :
  |(-Real.sqrt 2)| + (-2023)^(0 : ℕ) - 2 * Real.sin (45 * π / 180) - (1/2)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3059_305956


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l3059_305970

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs (z + Complex.I) = 2) :
  ∃ (max_val : ℝ), max_val = 4 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs (w + Complex.I) = 2 →
    Complex.abs ((w - (2 - Complex.I))^2 * (w - Complex.I)) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l3059_305970


namespace NUMINAMATH_CALUDE_largest_number_problem_l3059_305975

theorem largest_number_problem (A B C : ℝ) 
  (sum_eq : A + B + C = 50)
  (first_eq : A = 2 * B - 43)
  (third_eq : C = (1/2) * A + 5) :
  max A (max B C) = B ∧ B = 27.375 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l3059_305975


namespace NUMINAMATH_CALUDE_no_nonzero_triple_sum_zero_l3059_305990

theorem no_nonzero_triple_sum_zero :
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a = b + c ∧ b = c + a ∧ c = a + b ∧
    a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_triple_sum_zero_l3059_305990


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l3059_305942

-- Define variables for time spent on each activity
def trumpet_time : ℕ := 40
def running_time : ℕ := trumpet_time / 2
def basketball_time : ℕ := running_time / 2

-- Theorem to prove
theorem kenny_basketball_time : basketball_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l3059_305942


namespace NUMINAMATH_CALUDE_second_friend_is_nina_l3059_305913

structure Friend where
  hasChild : Bool
  name : String
  childName : String

def isNinotchka (name : String) : Bool :=
  name = "Nina" || name = "Ninotchka"

theorem second_friend_is_nina (friend1 friend2 : Friend) :
  friend2.hasChild = true →
  friend2.childName = friend2.name →
  isNinotchka friend2.childName →
  friend2.name = "Nina" :=
by
  sorry

end NUMINAMATH_CALUDE_second_friend_is_nina_l3059_305913


namespace NUMINAMATH_CALUDE_expression_simplification_l3059_305950

theorem expression_simplification (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = -2/3 * x - 2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3059_305950


namespace NUMINAMATH_CALUDE_student_number_calculation_l3059_305902

theorem student_number_calculation (x : ℤ) (h : x = 63) : (x * 4) - 142 = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_calculation_l3059_305902


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3059_305999

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  (X^5 + 3*X^3 + 1 : Polynomial ℝ) = (X + 1)^2 * q + r ∧
  r.degree < 2 ∧
  r = 5*X + 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3059_305999


namespace NUMINAMATH_CALUDE_compare_complex_fractions_l3059_305931

theorem compare_complex_fractions : 
  1 / ((123^2 - 4) * 1375) > (7 / (5 * 9150625)) - (1 / (605 * 125^2)) := by
  sorry

end NUMINAMATH_CALUDE_compare_complex_fractions_l3059_305931
