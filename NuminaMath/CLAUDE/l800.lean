import Mathlib

namespace NUMINAMATH_CALUDE_geometry_problem_l800_80074

-- Define the points
def M : ℝ × ℝ := (2, -2)
def N : ℝ × ℝ := (4, 4)
def P : ℝ × ℝ := (2, -3)

-- Define the equations
def perpendicular_bisector (x y : ℝ) : Prop := x + 3*y - 6 = 0
def parallel_line (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem geometry_problem :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2) ∧
  (∀ x y : ℝ, parallel_line x y ↔ 
    (y - P.2) = ((N.2 - M.2) / (N.1 - M.1)) * (x - P.1)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_problem_l800_80074


namespace NUMINAMATH_CALUDE_min_sum_a_b_l800_80089

theorem min_sum_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a * 2 + b * 3 - a * b = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * 2 + y * 3 - x * y = 0 → a + b ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l800_80089


namespace NUMINAMATH_CALUDE_factorial_quotient_trailing_zeros_l800_80056

def trailing_zeros (n : ℕ) : ℕ := sorry

def factorial (n : ℕ) : ℕ := sorry

theorem factorial_quotient_trailing_zeros :
  trailing_zeros (factorial 2018 / (factorial 30 * factorial 11)) = 493 := by sorry

end NUMINAMATH_CALUDE_factorial_quotient_trailing_zeros_l800_80056


namespace NUMINAMATH_CALUDE_different_solution_D_same_solution_B_same_solution_C_l800_80072

-- Define the reference equation
def reference_equation (x : ℚ) : Prop := x - 3 = 3 * x + 4

-- Define the equations from options B, C, and D
def equation_B (x : ℚ) : Prop := 1 / (x + 3) + 2 = 0
def equation_C (x a : ℚ) : Prop := (a^2 + 1) * (x - 3) = (3 * x + 4) * (a^2 + 1)
def equation_D (x : ℚ) : Prop := (7 * x - 4) * (x - 1) = (5 * x - 11) * (x - 1)

-- Theorem stating that D has a different solution set
theorem different_solution_D :
  ∃ x : ℚ, equation_D x ∧ ¬(reference_equation x) :=
sorry

-- Theorems stating that B and C have the same solution as the reference equation
theorem same_solution_B :
  ∀ x : ℚ, equation_B x ↔ reference_equation x :=
sorry

theorem same_solution_C :
  ∀ x a : ℚ, equation_C x a ↔ reference_equation x :=
sorry

end NUMINAMATH_CALUDE_different_solution_D_same_solution_B_same_solution_C_l800_80072


namespace NUMINAMATH_CALUDE_descent_time_is_50_seconds_l800_80088

/-- Represents the descent scenario on an escalator -/
structure EscalatorDescent where
  /-- Time taken to walk down stationary escalator (in seconds) -/
  stationary_time : ℝ
  /-- Time taken to walk down moving escalator (in seconds) -/
  moving_time : ℝ
  /-- Duration of escalator stoppage (in seconds) -/
  stop_duration : ℝ

/-- Calculates the total descent time for the given scenario -/
def total_descent_time (descent : EscalatorDescent) : ℝ :=
  sorry

/-- Theorem stating that the total descent time is 50 seconds -/
theorem descent_time_is_50_seconds (descent : EscalatorDescent) 
  (h1 : descent.stationary_time = 80)
  (h2 : descent.moving_time = 40)
  (h3 : descent.stop_duration = 20) :
  total_descent_time descent = 50 := by
  sorry

end NUMINAMATH_CALUDE_descent_time_is_50_seconds_l800_80088


namespace NUMINAMATH_CALUDE_odd_divisibility_l800_80050

theorem odd_divisibility (n : ℕ) (h : Odd (94 * n)) :
  ∃ k : ℕ, n * (n - 1) ^ ((n - 1) ^ n + 1) + n = k * ((n - 1) ^ n + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisibility_l800_80050


namespace NUMINAMATH_CALUDE_reflected_point_spherical_coordinates_l800_80037

/-- Given a point P with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    this function returns the spherical coordinates of the point Q(-x, y, z) -/
def spherical_coordinates_of_reflected_point (x y z ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that if a point P has rectangular coordinates (x, y, z) and 
    spherical coordinates (3, 5π/6, π/4), then the point Q(-x, y, z) has 
    spherical coordinates (3, π/6, π/4) -/
theorem reflected_point_spherical_coordinates 
  (x y z : Real) 
  (h1 : x = 3 * Real.sin (π/4) * Real.cos (5*π/6))
  (h2 : y = 3 * Real.sin (π/4) * Real.sin (5*π/6))
  (h3 : z = 3 * Real.cos (π/4)) :
  spherical_coordinates_of_reflected_point x y z 3 (5*π/6) (π/4) = (3, π/6, π/4) := by
  sorry

end NUMINAMATH_CALUDE_reflected_point_spherical_coordinates_l800_80037


namespace NUMINAMATH_CALUDE_range_of_a_l800_80071

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- Define the set of x that satisfies p
def P : Set ℝ := {x | p x}

-- Define the set of x that satisfies q for a given a
def Q (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x a)) ↔
  (∀ a : ℝ, a ∈ Set.Ici 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l800_80071


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l800_80046

/-- A geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term formula for the geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l800_80046


namespace NUMINAMATH_CALUDE_jake_viewing_time_l800_80086

/-- Calculates the number of hours Jake watched on Friday given his viewing schedule for the week --/
theorem jake_viewing_time (hours_per_day : ℕ) (show_length : ℕ) : 
  hours_per_day = 24 →
  show_length = 52 →
  let monday := hours_per_day / 2
  let tuesday := 4
  let wednesday := hours_per_day / 4
  let mon_to_wed := monday + tuesday + wednesday
  let thursday := mon_to_wed / 2
  let mon_to_thu := mon_to_wed + thursday
  19 = show_length - mon_to_thu := by sorry


end NUMINAMATH_CALUDE_jake_viewing_time_l800_80086


namespace NUMINAMATH_CALUDE_least_three_digit_with_product_12_l800_80069

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_product_12_l800_80069


namespace NUMINAMATH_CALUDE_smallest_number_properties_l800_80083

/-- The smallest number that is divisible by 18 and 30 and is a perfect square -/
def smallest_number : ℕ := 900

/-- Predicate to check if a number is divisible by both 18 and 30 -/
def divisible_by_18_and_30 (n : ℕ) : Prop := n % 18 = 0 ∧ n % 30 = 0

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem smallest_number_properties :
  divisible_by_18_and_30 smallest_number ∧
  is_perfect_square smallest_number ∧
  ∀ n : ℕ, n < smallest_number → ¬(divisible_by_18_and_30 n ∧ is_perfect_square n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_properties_l800_80083


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l800_80093

theorem basketball_shot_probability (a b c : ℝ) : 
  a ∈ (Set.Ioo 0 1) → 
  b ∈ (Set.Ioo 0 1) → 
  c ∈ (Set.Ioo 0 1) → 
  3 * a + 2 * b = 1 → 
  a * b ≤ 1 / 24 := by
sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l800_80093


namespace NUMINAMATH_CALUDE_f_properties_l800_80031

def f (x : ℝ) : ℝ := 1 - |x - x^2|

theorem f_properties :
  (∀ x, f x ≤ 1) ∧
  (f 0 = 1 ∧ f 1 = 1) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 1 - x + x^2) ∧
  (∀ x, (x < 0 ∨ x > 1) → f x = 1 + x - x^2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l800_80031


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l800_80070

theorem stamp_collection_problem (C K A : ℕ) : 
  C > 2 * K ∧ 
  K = A / 2 ∧ 
  C + K + A = 930 ∧ 
  A = 370 → 
  C - 2 * K = 5 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l800_80070


namespace NUMINAMATH_CALUDE_function_identity_l800_80042

theorem function_identity (f : ℕ → ℕ) : 
  (∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) → 
  (∀ n : ℕ, f n = n) := by sorry

end NUMINAMATH_CALUDE_function_identity_l800_80042


namespace NUMINAMATH_CALUDE_fixed_point_range_l800_80097

/-- A function f: ℝ → ℝ has a fixed point if there exists an x such that f(x) = x -/
def HasFixedPoint (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = x

/-- The quadratic function f(x) = x^2 + x + a -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + x + a

theorem fixed_point_range (a : ℝ) :
  HasFixedPoint (f a) → a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_range_l800_80097


namespace NUMINAMATH_CALUDE_smartphone_demand_l800_80065

theorem smartphone_demand (d p : ℝ) (k : ℝ) :
  (d * p = k) →  -- Demand is inversely proportional to price
  (30 * 600 = k) →  -- 30 customers purchase at $600
  (20 * 900 = k) →  -- 20 customers purchase at $900
  True :=
by
  sorry

end NUMINAMATH_CALUDE_smartphone_demand_l800_80065


namespace NUMINAMATH_CALUDE_storage_tubs_cost_l800_80060

/-- The total cost of storage tubs -/
def total_cost (large_count : ℕ) (small_count : ℕ) (large_price : ℕ) (small_price : ℕ) : ℕ :=
  large_count * large_price + small_count * small_price

/-- Theorem: The total cost of 3 large tubs at $6 each and 6 small tubs at $5 each is $48 -/
theorem storage_tubs_cost :
  total_cost 3 6 6 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_storage_tubs_cost_l800_80060


namespace NUMINAMATH_CALUDE_min_vertices_for_perpendicular_diagonals_l800_80020

theorem min_vertices_for_perpendicular_diagonals : 
  (∀ k : ℕ, k < 28 → ¬(∃ m : ℕ, 2 * m = k ∧ m * (m - 1)^2 / 2 ≥ 1000)) ∧ 
  (∃ m : ℕ, 2 * m = 28 ∧ m * (m - 1)^2 / 2 ≥ 1000) := by
  sorry

end NUMINAMATH_CALUDE_min_vertices_for_perpendicular_diagonals_l800_80020


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l800_80075

theorem smallest_n_for_candy (n : ℕ) : 
  (∀ m : ℕ, m > 0 → (25 * m) % 10 = 0 ∧ (25 * m) % 18 = 0 ∧ (25 * m) % 20 = 0 → m ≥ n) →
  (25 * n) % 10 = 0 ∧ (25 * n) % 18 = 0 ∧ (25 * n) % 20 = 0 →
  n = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l800_80075


namespace NUMINAMATH_CALUDE_rational_equation_solution_l800_80012

theorem rational_equation_solution :
  ∃ x : ℚ, (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9/4 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l800_80012


namespace NUMINAMATH_CALUDE_sum_of_first_and_third_l800_80068

theorem sum_of_first_and_third (A B C : ℝ) : 
  A + B + C = 330 → 
  A = 2 * B → 
  C = A / 3 → 
  B = 90 → 
  A + C = 240 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_and_third_l800_80068


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l800_80078

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the point P as the intersection of C₁ and C₂ in the first quadrant
def P : ℝ × ℝ := sorry

-- State that P satisfies both C₁ and C₂
axiom P_on_C₁ : C₁ P.1 P.2
axiom P_on_C₂ : C₂ P.1 P.2

-- State that P is in the first quadrant
axiom P_first_quadrant : P.1 > 0 ∧ P.2 > 0

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := sorry

-- Theorem stating the distance from P to the left focus is 4
theorem distance_to_left_focus :
  Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l800_80078


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l800_80029

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l800_80029


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l800_80027

-- Problem 1
theorem problem_1 : -1^2 - |(-2)| + (1/3 - 3/4) * 12 = -8 := by sorry

-- Problem 2
theorem problem_2 :
  ∃ (x y : ℚ), (x / 2 - (y + 1) / 3 = 1) ∧ (3 * x + 2 * y = 10) ∧ (x = 3) ∧ (y = 1/2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l800_80027


namespace NUMINAMATH_CALUDE_right_triangle_sides_l800_80051

/-- A right triangle with perimeter 60 and altitude to hypotenuse 12 has sides 15, 20, and 25. -/
theorem right_triangle_sides (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a + b + c = 60 →
  h = 12 →
  a^2 + b^2 = c^2 →
  a * b = 2 * h * c →
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l800_80051


namespace NUMINAMATH_CALUDE_unique_divisible_number_l800_80055

theorem unique_divisible_number : ∃! (x y z u v : ℕ),
  (x < 10 ∧ y < 10 ∧ z < 10 ∧ u < 10 ∧ v < 10) ∧
  (x * 10^9 + 6 * 10^8 + 1 * 10^7 + y * 10^6 + 0 * 10^5 + 6 * 10^4 + 4 * 10^3 + z * 10^2 + u * 10 + v) % 61875 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l800_80055


namespace NUMINAMATH_CALUDE_new_average_weight_l800_80016

theorem new_average_weight (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 6 →
  a = 78 →
  (b + c + d + e) / 4 = 79 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l800_80016


namespace NUMINAMATH_CALUDE_pet_shop_dogs_count_l800_80062

/-- Given a pet shop with dogs, cats, and bunnies, where the ratio of dogs to cats to bunnies
    is 7 : 7 : 8, and the total number of dogs and bunnies is 330, prove that there are 154 dogs. -/
theorem pet_shop_dogs_count : ℕ → ℕ → ℕ → Prop :=
  fun dogs cats bunnies =>
    (dogs : ℚ) / cats = 1 →
    (dogs : ℚ) / bunnies = 7 / 8 →
    dogs + bunnies = 330 →
    dogs = 154

/-- Proof of the pet_shop_dogs_count theorem -/
lemma prove_pet_shop_dogs_count : ∃ dogs cats bunnies, pet_shop_dogs_count dogs cats bunnies :=
  sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_count_l800_80062


namespace NUMINAMATH_CALUDE_eve_age_proof_l800_80038

/-- Adam's current age -/
def adam_age : ℕ := 9

/-- Eve's current age -/
def eve_age : ℕ := 14

/-- Theorem stating Eve's age based on the given conditions -/
theorem eve_age_proof :
  (adam_age < eve_age) ∧
  (eve_age + 1 = 3 * (adam_age - 4)) ∧
  (adam_age = 9) →
  eve_age = 14 := by
sorry

end NUMINAMATH_CALUDE_eve_age_proof_l800_80038


namespace NUMINAMATH_CALUDE_arrange_13_blue_5_red_l800_80035

/-- The number of ways to arrange blue and red balls with constraints -/
def arrange_balls (blue_balls red_balls : ℕ) : ℕ :=
  Nat.choose (blue_balls - red_balls + 1 + red_balls) (red_balls + 1)

/-- Theorem: Arranging 13 blue balls and 5 red balls with constraints yields 2002 ways -/
theorem arrange_13_blue_5_red :
  arrange_balls 13 5 = 2002 := by
  sorry

#eval arrange_balls 13 5

end NUMINAMATH_CALUDE_arrange_13_blue_5_red_l800_80035


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l800_80030

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1215 :=
by sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l800_80030


namespace NUMINAMATH_CALUDE_unique_right_triangle_l800_80098

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (3,4,5) forms a right triangle --/
theorem unique_right_triangle :
  (¬ isRightTriangle 2 3 4) ∧
  (¬ isRightTriangle 3 4 6) ∧
  (isRightTriangle 3 4 5) ∧
  (¬ isRightTriangle 4 5 6) :=
by sorry

#check unique_right_triangle

end NUMINAMATH_CALUDE_unique_right_triangle_l800_80098


namespace NUMINAMATH_CALUDE_student_photo_count_l800_80077

theorem student_photo_count :
  ∀ (m n : ℕ),
    m > 0 →
    n > 0 →
    m + 4 = n - 1 →  -- First rearrangement condition
    m + 3 = n - 2 →  -- Second rearrangement condition
    m * n = 24 :=    -- Total number of students
by
  sorry

end NUMINAMATH_CALUDE_student_photo_count_l800_80077


namespace NUMINAMATH_CALUDE_complex_division_result_l800_80006

/-- Given that z = a^2 - 1 + (1 + a)i where a ∈ ℝ is a purely imaginary number,
    prove that z / (2 + i) = 2/5 + 4/5 * i -/
theorem complex_division_result (a : ℝ) (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = a^2 - 1 + (1 + a) * i →
  z.re = 0 →
  z / (2 + i) = 2/5 + 4/5 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_result_l800_80006


namespace NUMINAMATH_CALUDE_earnings_ratio_l800_80052

theorem earnings_ratio (mork_rate mindy_rate combined_rate : ℝ) 
  (h1 : mork_rate = 0.30)
  (h2 : mindy_rate = 0.20)
  (h3 : combined_rate = 0.225) : 
  ∃ (m k : ℝ), m > 0 ∧ k > 0 ∧ 
    (mindy_rate * m + mork_rate * k) / (m + k) = combined_rate ∧ 
    m / k = 3 := by
  sorry

end NUMINAMATH_CALUDE_earnings_ratio_l800_80052


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l800_80014

theorem absolute_value_inequality_solution_range :
  (∃ (x : ℝ), |x - 5| + |x - 3| < m) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l800_80014


namespace NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l800_80079

-- Define a polygon as a type
class Polygon (P : Type)

-- Define a hexagon as a specific type of polygon
class Hexagon (H : Type) extends Polygon H

-- Define the sum of exterior angles for a polygon
def sum_of_exterior_angles (P : Type) [Polygon P] : ℝ := 360

-- Theorem statement
theorem hexagon_exterior_angles_sum (H : Type) [Hexagon H] :
  sum_of_exterior_angles H = 360 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l800_80079


namespace NUMINAMATH_CALUDE_mango_purchase_l800_80064

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The amount of grapes purchased in kg -/
def grape_amount : ℕ := 8

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 1055

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℕ := (total_paid - grape_amount * grape_price) / mango_price

theorem mango_purchase : mango_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_l800_80064


namespace NUMINAMATH_CALUDE_archer_fish_count_l800_80092

/-- The total number of fish Archer caught in a day -/
def total_fish (first_round : ℕ) (second_round_increase : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := first_round + second_round_increase
  let third_round := second_round + (third_round_percentage * second_round) / 100
  first_round + second_round + third_round

/-- Theorem stating that Archer caught 60 fish in total -/
theorem archer_fish_count : total_fish 8 12 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_archer_fish_count_l800_80092


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangency_l800_80019

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- A circle with center (0, 3) and parameter m -/
structure Circle (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + y^2 - 6*y + m = 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The asymptote of a hyperbola -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) := sorry

/-- Tangency condition between a line and a circle -/
def is_tangent (l : Set (ℝ × ℝ)) (c : Circle m) : Prop := sorry

theorem hyperbola_circle_tangency 
  (a b : ℝ) 
  (h : Hyperbola a b) 
  (m : ℝ) 
  (c : Circle m) :
  eccentricity h = 3 →
  is_tangent (asymptote h) c →
  m = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangency_l800_80019


namespace NUMINAMATH_CALUDE_range_of_a_l800_80076

-- Define the propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*a*m + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ x^2 / (m - 1) + y^2 / (2 - m - c) = 1

-- Define the theorem
theorem range_of_a : 
  (∀ m a : ℝ, ¬(q m) → ¬(p m a)) ∧ 
  (∃ m a : ℝ, ¬(q m) ∧ p m a) → 
  {a : ℝ | 1/3 ≤ a ∧ a ≤ 3/8} = {a : ℝ | ∃ m : ℝ, p m a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l800_80076


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l800_80049

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  1/x + 1/y + 1/z ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l800_80049


namespace NUMINAMATH_CALUDE_sqrt_one_plus_a_squared_is_quadratic_radical_l800_80022

/-- A function is a quadratic radical if it's the square root of an expression 
    that yields a real number for all real values of its variable. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(1 + a²) is a quadratic radical. -/
theorem sqrt_one_plus_a_squared_is_quadratic_radical :
  is_quadratic_radical (fun a => Real.sqrt (1 + a^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_sqrt_one_plus_a_squared_is_quadratic_radical_l800_80022


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l800_80023

theorem largest_integer_with_remainder : ∃ n : ℕ, n = 94 ∧ 
  (∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n) ∧ 
  n < 100 ∧ 
  n % 6 = 4 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l800_80023


namespace NUMINAMATH_CALUDE_chang_e_3_descent_time_l800_80057

/-- Represents the descent phase of Chang'e 3 --/
structure DescentPhase where
  initial_altitude : ℝ  -- in kilometers
  final_altitude : ℝ    -- in meters
  playback_time_initial : ℕ  -- in seconds
  total_video_duration : ℕ   -- in seconds

/-- Calculates the time spent in the descent phase --/
def descent_time (d : DescentPhase) : ℕ :=
  114  -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the descent time for the given conditions is 114 seconds --/
theorem chang_e_3_descent_time :
  let d : DescentPhase := {
    initial_altitude := 2.4,
    final_altitude := 100,
    playback_time_initial := 30 * 60 + 28,  -- 30 minutes and 28 seconds
    total_video_duration := 2 * 60 * 60 + 10 * 60 + 48  -- 2 hours, 10 minutes, and 48 seconds
  }
  descent_time d = 114 := by sorry

end NUMINAMATH_CALUDE_chang_e_3_descent_time_l800_80057


namespace NUMINAMATH_CALUDE_square_areas_sum_l800_80047

theorem square_areas_sum (a : ℝ) (h1 : a > 0) (h2 : (a + 4)^2 - a^2 = 80) : 
  a^2 + (a + 4)^2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_square_areas_sum_l800_80047


namespace NUMINAMATH_CALUDE_dots_on_line_l800_80011

/-- The number of dots drawn on a line of given length at given intervals, excluding the beginning and end points. -/
def numDots (lineLength : ℕ) (interval : ℕ) : ℕ :=
  if interval = 0 then 0
  else (lineLength - interval) / interval

theorem dots_on_line (lineLength : ℕ) (interval : ℕ) 
  (h1 : lineLength = 30) 
  (h2 : interval = 5) : 
  numDots lineLength interval = 5 := by
  sorry

end NUMINAMATH_CALUDE_dots_on_line_l800_80011


namespace NUMINAMATH_CALUDE_photos_sum_equals_total_l800_80015

/-- The total number of photos collected by Tom, Tim, and Paul -/
def total_photos : ℕ := 152

/-- Tom's photos -/
def tom_photos : ℕ := 38

/-- Tim's photos -/
def tim_photos : ℕ := total_photos - 100

/-- Paul's photos -/
def paul_photos : ℕ := tim_photos + 10

/-- Theorem stating that the sum of individual photos equals the total photos -/
theorem photos_sum_equals_total : 
  tom_photos + tim_photos + paul_photos = total_photos := by sorry

end NUMINAMATH_CALUDE_photos_sum_equals_total_l800_80015


namespace NUMINAMATH_CALUDE_king_spade_then_spade_probability_l800_80048

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (spades : Nat)
  (king_of_spades : Nat)

/-- The probability of drawing a King of Spades followed by any Spade from a standard 52-card deck -/
def probability_king_spade_then_spade (d : Deck) : Rat :=
  (d.king_of_spades : Rat) / d.total_cards * (d.spades - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a King of Spades followed by any Spade 
    from a standard 52-card deck is 1/221 -/
theorem king_spade_then_spade_probability :
  probability_king_spade_then_spade ⟨52, 13, 1⟩ = 1 / 221 := by
  sorry

end NUMINAMATH_CALUDE_king_spade_then_spade_probability_l800_80048


namespace NUMINAMATH_CALUDE_circle_area_difference_l800_80043

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let c2 : ℝ := 30
  let area1 := π * r1^2
  let r2 := c2 / (2 * π)
  let area2 := π * r2^2
  area1 - area2 = (225 * (4 * π^2 - 1)) / π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l800_80043


namespace NUMINAMATH_CALUDE_team_c_score_l800_80063

/-- Given a trivia game with three teams, prove that Team C's score is 4 points. -/
theorem team_c_score (team_a team_b team_c total : ℕ) : 
  team_a = 2 → team_b = 9 → total = 15 → team_a + team_b + team_c = total → team_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_team_c_score_l800_80063


namespace NUMINAMATH_CALUDE_circplus_comm_circplus_not_scalar_mult_circplus_zero_circplus_self_circplus_pos_l800_80039

-- Define the ⊕ operation
def circplus (x y : ℝ) : ℝ := |x - y|^2

-- Theorem statements
theorem circplus_comm (x y : ℝ) : circplus x y = circplus y x := by sorry

theorem circplus_not_scalar_mult (x y : ℝ) : 
  2 * (circplus x y) ≠ circplus (2 * x) (2 * y) := by sorry

theorem circplus_zero (x : ℝ) : circplus x 0 = x^2 := by sorry

theorem circplus_self (x : ℝ) : circplus x x = 0 := by sorry

theorem circplus_pos (x y : ℝ) : x ≠ y → circplus x y > 0 := by sorry

end NUMINAMATH_CALUDE_circplus_comm_circplus_not_scalar_mult_circplus_zero_circplus_self_circplus_pos_l800_80039


namespace NUMINAMATH_CALUDE_equivalent_discount_l800_80021

theorem equivalent_discount (original_price : ℝ) 
  (first_discount second_discount : ℝ) 
  (h1 : first_discount = 0.3) 
  (h2 : second_discount = 0.2) :
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  let equivalent_discount := 1 - (final_price / original_price)
  equivalent_discount = 0.44 := by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l800_80021


namespace NUMINAMATH_CALUDE_correct_average_after_adjustments_l800_80010

theorem correct_average_after_adjustments (n : ℕ) (initial_avg : ℚ) 
  (error1 : ℚ) (wrong_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 17 →
  wrong_num = 13 →
  correct_num = 31 →
  (n : ℚ) * initial_avg - error1 - wrong_num + correct_num = n * 40.3 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_adjustments_l800_80010


namespace NUMINAMATH_CALUDE_range_of_fraction_l800_80009

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  ∃ (x : ℝ), -2 < x ∧ x < -1/2 ∧ (∃ (a' b' : ℝ), 1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < -1 ∧ x = a' / b') ∧
  (∀ (y : ℝ), (∃ (a' b' : ℝ), 1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < -1 ∧ y = a' / b') → -2 < y ∧ y < -1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l800_80009


namespace NUMINAMATH_CALUDE_round_trip_combinations_l800_80081

def num_flights_A_to_B : ℕ := 2
def num_flights_B_to_A : ℕ := 3

theorem round_trip_combinations : num_flights_A_to_B * num_flights_B_to_A = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_combinations_l800_80081


namespace NUMINAMATH_CALUDE_max_value_theorem_l800_80091

theorem max_value_theorem (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_eq : x^2 - 3*x*y + 4*y^2 = 9) :
  x^2 + 3*x*y + 4*y^2 ≤ 63 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 - 3*x₀*y₀ + 4*y₀^2 = 9 ∧ x₀^2 + 3*x₀*y₀ + 4*y₀^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l800_80091


namespace NUMINAMATH_CALUDE_complex_number_simplification_l800_80003

theorem complex_number_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  3*i * (2 - 5*i) - (4 - 7*i) = 11 + 13*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l800_80003


namespace NUMINAMATH_CALUDE_algebraic_grid_difference_l800_80008

/-- Represents a 3x3 grid of algebraic expressions -/
structure AlgebraicGrid (α : Type) [Ring α] where
  grid : Matrix (Fin 3) (Fin 3) α

/-- Checks if all rows, columns, and diagonals have the same sum -/
def isValidGrid {α : Type} [Ring α] (g : AlgebraicGrid α) : Prop :=
  let rowSum (i : Fin 3) := g.grid i 0 + g.grid i 1 + g.grid i 2
  let colSum (j : Fin 3) := g.grid 0 j + g.grid 1 j + g.grid 2 j
  let diag1Sum := g.grid 0 0 + g.grid 1 1 + g.grid 2 2
  let diag2Sum := g.grid 0 2 + g.grid 1 1 + g.grid 2 0
  ∀ i j : Fin 3, rowSum i = colSum j ∧ rowSum i = diag1Sum ∧ rowSum i = diag2Sum

theorem algebraic_grid_difference {α : Type} [CommRing α] (x : α) (M N : α) :
  let g : AlgebraicGrid α := {
    grid := λ i j =>
      if i = 0 ∧ j = 0 then M
      else if i = 0 ∧ j = 2 then x^2 - x - 1
      else if i = 1 ∧ j = 2 then x
      else if i = 2 ∧ j = 0 then x^2 - x
      else if i = 2 ∧ j = 1 then x - 1
      else if i = 2 ∧ j = 2 then N
      else 0  -- Other entries are not specified
  }
  isValidGrid g →
  M - N = -2*x^2 + 4*x :=
by
  sorry

end NUMINAMATH_CALUDE_algebraic_grid_difference_l800_80008


namespace NUMINAMATH_CALUDE_inequality_proof_l800_80026

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) :
  a > 1 ∧ 1 > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l800_80026


namespace NUMINAMATH_CALUDE_angle_b_value_l800_80067

-- Define the angles
variable (a b c : ℝ)

-- Define the conditions
axiom straight_line : a + b + c = 180
axiom ratio_b_a : b = 2 * a
axiom ratio_c_b : c = 3 * b

-- Theorem to prove
theorem angle_b_value : b = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_value_l800_80067


namespace NUMINAMATH_CALUDE_platform_length_l800_80073

/-- The length of a platform crossed by two trains moving in opposite directions -/
theorem platform_length 
  (x y : ℝ) -- lengths of trains A and B in meters
  (p q : ℝ) -- speeds of trains A and B in km/h
  (t : ℝ) -- time taken to cross the platform in seconds
  (h_positive : x > 0 ∧ y > 0 ∧ p > 0 ∧ q > 0 ∧ t > 0) -- All values are positive
  : ∃ (L : ℝ), L = (p + q) * (5 * t / 18) - (x + y) :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l800_80073


namespace NUMINAMATH_CALUDE_pond_side_length_l800_80085

/-- Represents the dimensions and pond of a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ
  pond_side : ℝ

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the remaining area after building a square pond -/
def remaining_area (g : Garden) : ℝ := garden_area g - g.pond_side ^ 2

/-- Theorem stating the side length of the pond given the conditions -/
theorem pond_side_length (g : Garden) 
  (h1 : g.length = 15)
  (h2 : g.width = 10)
  (h3 : remaining_area g = (garden_area g) / 2) :
  g.pond_side = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pond_side_length_l800_80085


namespace NUMINAMATH_CALUDE_no_real_m_for_single_root_l800_80059

theorem no_real_m_for_single_root : 
  ¬∃ (m : ℝ), (∀ (x : ℝ), x^2 + (4*m+2)*x + m = 0 ↔ x = -2*m-1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_m_for_single_root_l800_80059


namespace NUMINAMATH_CALUDE_closed_mul_l800_80024

structure SpecialSet (S : Set ℝ) : Prop where
  one_mem : (1 : ℝ) ∈ S
  closed_sub : ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S
  closed_inv : ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

theorem closed_mul {S : Set ℝ} (h : SpecialSet S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_closed_mul_l800_80024


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l800_80041

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 30 * x + k = 0) ↔ (k = 9 ∨ k = 15) := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l800_80041


namespace NUMINAMATH_CALUDE_periodic_even_function_extension_l800_80080

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = Real.log (1 - x) / Real.log (1/2)) :
  ∀ x ∈ Set.Ioo 1 2, f x = Real.log (x - 1) / Real.log (1/2) := by
sorry

end NUMINAMATH_CALUDE_periodic_even_function_extension_l800_80080


namespace NUMINAMATH_CALUDE_larger_number_proof_l800_80099

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 7 * S + 15) :
  L = 1590 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l800_80099


namespace NUMINAMATH_CALUDE_inequality_proof_l800_80096

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l800_80096


namespace NUMINAMATH_CALUDE_fraction_equality_l800_80034

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (4*x - y) / (3*x + 2*y) = 3) : 
  (3*x - 2*y) / (4*x + y) = 31/23 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l800_80034


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l800_80013

theorem square_difference_of_sum_and_difference (x y : ℝ) 
  (h_sum : x + y = 20) (h_diff : x - y = 8) : x^2 - y^2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l800_80013


namespace NUMINAMATH_CALUDE_different_amounts_eq_127_l800_80032

/-- Represents the number of coins of each denomination --/
structure CoinCounts where
  jiao_1 : Nat
  jiao_5 : Nat
  yuan_1 : Nat
  yuan_5 : Nat

/-- Calculates the number of different non-zero amounts that can be paid with the given coins --/
def differentAmounts (coins : CoinCounts) : Nat :=
  sorry

/-- The specific coin counts given in the problem --/
def problemCoins : CoinCounts :=
  { jiao_1 := 1
  , jiao_5 := 2
  , yuan_1 := 5
  , yuan_5 := 2 }

/-- Theorem stating that the number of different non-zero amounts is 127 --/
theorem different_amounts_eq_127 : differentAmounts problemCoins = 127 :=
  sorry

end NUMINAMATH_CALUDE_different_amounts_eq_127_l800_80032


namespace NUMINAMATH_CALUDE_hexagon_area_equals_six_l800_80002

/-- Given an equilateral triangle with area 4 and a regular hexagon with the same perimeter,
    prove that the area of the hexagon is 6. -/
theorem hexagon_area_equals_six (s t : ℝ) : 
  s > 0 → t > 0 → -- Positive side lengths
  3 * s = 6 * t → -- Equal perimeters
  s^2 * Real.sqrt 3 / 4 = 4 → -- Triangle area
  6 * (t^2 * Real.sqrt 3 / 4) = 6 := by
sorry


end NUMINAMATH_CALUDE_hexagon_area_equals_six_l800_80002


namespace NUMINAMATH_CALUDE_cos_330_degrees_l800_80040

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l800_80040


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l800_80036

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l800_80036


namespace NUMINAMATH_CALUDE_platform_length_l800_80017

/-- Given a train of length 1200 m that takes 120 sec to pass a tree and 150 sec to pass a platform, 
    prove that the length of the platform is 300 m. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_tree : ℝ) 
  (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 150) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 300 := by
sorry


end NUMINAMATH_CALUDE_platform_length_l800_80017


namespace NUMINAMATH_CALUDE_pizza_combinations_l800_80053

theorem pizza_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l800_80053


namespace NUMINAMATH_CALUDE_octagon_area_l800_80028

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
theorem octagon_area (r : ℝ) (h : r = 3) : 
  let octagon_area := 8 * (1/2 * r^2 * Real.sin (π/4))
  octagon_area = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l800_80028


namespace NUMINAMATH_CALUDE_average_marks_proof_l800_80025

-- Define the marks for each subject
def physics : ℝ := 125
def chemistry : ℝ := 15
def mathematics : ℝ := 55

-- Define the conditions
theorem average_marks_proof :
  -- Average of all three subjects is 65
  (physics + chemistry + mathematics) / 3 = 65 ∧
  -- Average of physics and mathematics is 90
  (physics + mathematics) / 2 = 90 ∧
  -- Average of physics and chemistry is 70
  (physics + chemistry) / 2 = 70 ∧
  -- Physics marks are 125
  physics = 125 →
  -- Prove that chemistry is the subject that averages 70 with physics
  (physics + chemistry) / 2 = 70 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_proof_l800_80025


namespace NUMINAMATH_CALUDE_mode_and_median_of_game_scores_l800_80001

def game_scores : List Int := [20, 18, 23, 17, 20, 20, 18]

def mode (l : List Int) : Int := sorry

def median (l : List Int) : Int := sorry

theorem mode_and_median_of_game_scores :
  mode game_scores = 20 ∧ median game_scores = 20 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_game_scores_l800_80001


namespace NUMINAMATH_CALUDE_parabola_c_value_l800_80004

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, -5), and passing through (-1, -4) -/
def Parabola (a b c : ℚ) : Prop :=
  ∀ x y : ℚ, y = a * x^2 + b * x + c →
  (∃ t : ℚ, y = a * (x + 3)^2 - 5) ∧  -- vertex form
  (-4 : ℚ) = a * (-1 + 3)^2 - 5       -- passes through (-1, -4)

/-- The value of c for the given parabola is -11/4 -/
theorem parabola_c_value (a b c : ℚ) (h : Parabola a b c) : c = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l800_80004


namespace NUMINAMATH_CALUDE_inequality_solution_l800_80044

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, ((x - p) * (x - q)) / (x - r) ≥ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2))
  (h2 : p < q) : 
  p + 2*q + 3*r = 78 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l800_80044


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l800_80018

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l800_80018


namespace NUMINAMATH_CALUDE_largest_base3_3digit_in_base10_l800_80000

/-- The largest three-digit number in base 3 -/
def largest_base3_3digit : ℕ := 2 * 3^2 + 2 * 3^1 + 2 * 3^0

/-- Theorem: The largest three-digit number in base 3, when converted to base 10, equals 26 -/
theorem largest_base3_3digit_in_base10 : largest_base3_3digit = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_base3_3digit_in_base10_l800_80000


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l800_80058

theorem largest_n_satisfying_inequality : ∃ (n : ℕ),
  (∃ (x : Fin n → ℝ), (∀ (i j : Fin n), i < j → 
    (1 + x i * x j)^2 ≤ 0.99 * (1 + (x i)^2) * (1 + (x j)^2))) ∧
  (∀ (m : ℕ), m > n → 
    ¬∃ (y : Fin m → ℝ), ∀ (i j : Fin m), i < j → 
      (1 + y i * y j)^2 ≤ 0.99 * (1 + (y i)^2) * (1 + (y j)^2)) ∧
  n = 31 :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l800_80058


namespace NUMINAMATH_CALUDE_pencil_packing_problem_l800_80084

theorem pencil_packing_problem :
  ∃ (a k m : ℤ),
    200 ≤ a ∧ a ≤ 300 ∧
    a % 10 = 7 ∧
    a % 12 = 9 ∧
    a = 60 * m + 57 ∧
    (a = 237 ∨ a = 297) :=
by sorry

end NUMINAMATH_CALUDE_pencil_packing_problem_l800_80084


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_l800_80045

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Represents the dimensions of the wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build the wall -/
def blocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  let oddRowBlocks := wall.length / 2
  let evenRowBlocks := oddRowBlocks + 1
  let numRows := wall.height / block.height
  let oddRows := numRows / 2
  let evenRows := numRows - oddRows
  oddRows * oddRowBlocks + evenRows * evenRowBlocks

/-- The theorem stating the smallest number of blocks needed -/
theorem smallest_number_of_blocks 
  (wall : WallDimensions)
  (block : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 8)
  (h3 : block.height = 1)
  (h4 : block.length = 2 ∨ block.length = 1)
  (h5 : wall.length % 2 = 0) -- Ensures wall is even on the ends
  : blocksNeeded wall block = 484 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_l800_80045


namespace NUMINAMATH_CALUDE_shaded_area_recursive_square_division_l800_80082

theorem shaded_area_recursive_square_division (r : ℝ) (h1 : r = 1/16) (h2 : 0 < r) (h3 : r < 1) :
  (1/4) * (1 / (1 - r)) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_recursive_square_division_l800_80082


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l800_80094

theorem square_plus_inverse_square (x : ℝ) (h : x - 3/x = 2) : x^2 + 9/x^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l800_80094


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l800_80033

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : average_all = 5/2)
  (h3 : childless_families = 2) :
  (total_families * average_all) / (total_families - childless_families) = 3 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l800_80033


namespace NUMINAMATH_CALUDE_existence_of_triple_l800_80005

theorem existence_of_triple (n : ℕ) :
  let A := Finset.range (2^(n+1))
  ∀ S : Finset ℕ, S ⊆ A → S.card = 2*n + 1 →
    ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (b * c : ℝ) < 2 * (a^2 : ℝ) ∧ 2 * (a^2 : ℝ) < 4 * (b * c : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_triple_l800_80005


namespace NUMINAMATH_CALUDE_balloon_count_l800_80087

theorem balloon_count (initial : Real) (given : Real) (total : Real) 
  (h1 : initial = 7.0)
  (h2 : given = 5.0)
  (h3 : total = initial + given) :
  total = 12.0 := by
sorry

end NUMINAMATH_CALUDE_balloon_count_l800_80087


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l800_80054

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5789 * 10 + N) % 6 = 0 → N ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l800_80054


namespace NUMINAMATH_CALUDE_tilly_bag_cost_l800_80007

/-- Calculates the cost per bag for Tilly's business --/
def cost_per_bag (num_bags : ℕ) (selling_price : ℚ) (total_profit : ℚ) : ℚ :=
  (num_bags * selling_price - total_profit) / num_bags

/-- Proves that the cost per bag is $7 given the problem conditions --/
theorem tilly_bag_cost :
  cost_per_bag 100 10 300 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tilly_bag_cost_l800_80007


namespace NUMINAMATH_CALUDE_train_speed_l800_80090

/-- The speed of a train given the time to pass an electric pole and a platform -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 15 →
  platform_length = 380 →
  platform_time = 52.99696024318054 →
  ∃ (speed : ℝ), abs (speed - 36.0037908) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l800_80090


namespace NUMINAMATH_CALUDE_equation_solutions_l800_80066

theorem equation_solutions : 
  {x : ℝ | Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6} = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l800_80066


namespace NUMINAMATH_CALUDE_trapezoid_area_l800_80061

/-- The area of a trapezoid with sum of bases 36 cm and height 15 cm is 270 square centimeters. -/
theorem trapezoid_area (base_sum : ℝ) (height : ℝ) (h1 : base_sum = 36) (h2 : height = 15) :
  (base_sum * height) / 2 = 270 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l800_80061


namespace NUMINAMATH_CALUDE_chessboard_decomposition_l800_80095

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  white_squares : Nat
  black_squares : Nat

/-- Represents a decomposition of the chessboard -/
def Decomposition := List Rectangle

/-- Checks if a decomposition is valid according to the given conditions -/
def is_valid_decomposition (d : Decomposition) : Prop :=
  d.all (λ r => r.white_squares = r.black_squares) ∧
  d.length > 0 ∧
  (List.zip d (List.tail d)).all (λ (r1, r2) => r1.white_squares < r2.white_squares) ∧
  (d.map (λ r => r.white_squares + r.black_squares)).sum = 64

/-- The main theorem to be proved -/
theorem chessboard_decomposition :
  (∃ (d : Decomposition), is_valid_decomposition d ∧ d.length = 7) ∧
  (∀ (d : Decomposition), is_valid_decomposition d → d.length ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_chessboard_decomposition_l800_80095
