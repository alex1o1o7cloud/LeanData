import Mathlib

namespace simplify_exponents_l2611_261102

theorem simplify_exponents (a b : ℝ) : (a^4 * a^3) * (b^2 * b^5) = a^7 * b^7 := by
  sorry

end simplify_exponents_l2611_261102


namespace otimes_self_otimes_self_l2611_261124

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem: For any real number a, (a ⊗ a) ⊗ (a ⊗ a) = 0 -/
theorem otimes_self_otimes_self (a : ℝ) : otimes (otimes a a) (otimes a a) = 0 := by
  sorry

end otimes_self_otimes_self_l2611_261124


namespace price_comparison_l2611_261106

theorem price_comparison (a : ℝ) (h : a > 0) : a * (1.1^5) * (0.9^5) < a := by
  sorry

end price_comparison_l2611_261106


namespace units_digit_of_7_power_1000_l2611_261192

theorem units_digit_of_7_power_1000 : (7^(10^3)) % 10 = 1 := by
  sorry

end units_digit_of_7_power_1000_l2611_261192


namespace convex_pentagon_side_comparison_l2611_261150

/-- A circle in which pentagons are inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A convex pentagon inscribed in a circle -/
structure ConvexPentagon (c : Circle) where
  vertices : Fin 5 → ℝ × ℝ
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = c.radius^2
  convex : sorry  -- Additional condition to ensure convexity

/-- The side length of a regular pentagon inscribed in a circle -/
def regularPentagonSideLength (c : Circle) : ℝ := sorry

/-- The side lengths of a convex pentagon -/
def pentagonSideLengths (c : Circle) (p : ConvexPentagon c) : Fin 5 → ℝ := sorry

theorem convex_pentagon_side_comparison (c : Circle) (p : ConvexPentagon c) :
  ∃ i : Fin 5, pentagonSideLengths c p i ≤ regularPentagonSideLength c := by sorry

end convex_pentagon_side_comparison_l2611_261150


namespace james_muffins_count_l2611_261168

def arthur_muffins : ℝ := 115.0
def baking_ratio : ℝ := 12.0

theorem james_muffins_count : 
  arthur_muffins / baking_ratio = 9.5833 := by sorry

end james_muffins_count_l2611_261168


namespace triangle_properties_l2611_261190

/-- Given a triangle ABC with the following properties:
    - a, b, c are sides opposite to angles A, B, C respectively
    - a = 2√3
    - A = π/3
    - Area S = 2√3
    - sin(C-B) = sin(2B) - sin(A)
    Prove the properties of sides b, c and the shape of the triangle -/
theorem triangle_properties (a b c A B C S : Real) : 
  a = 2 * Real.sqrt 3 →
  A = π / 3 →
  S = 2 * Real.sqrt 3 →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  Real.sin (C - B) = Real.sin (2*B) - Real.sin A →
  ((b = 2 ∧ c = 4) ∨ (b = 4 ∧ c = 2)) ∧
  (B = π / 2 ∨ C = B) := by
  sorry

end triangle_properties_l2611_261190


namespace part1_part2_l2611_261134

-- Define the operation
def star_op (a b : ℚ) : ℚ := (a * b) / (a + b)

-- Part 1: Prove the specific calculation
theorem part1 : star_op (-3) (-1/3) = -3/10 := by sorry

-- Part 2: Prove when the operation is undefined
theorem part2 (a b : ℚ) : 
  a + b = 0 → ¬ ∃ (q : ℚ), star_op a b = q := by sorry

end part1_part2_l2611_261134


namespace largest_non_sum_of_composites_l2611_261191

/-- A number is composite if it has more than two distinct positive divisors. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A natural number can be expressed as the sum of two composite numbers. -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ a + b = n

/-- 11 is the largest natural number that cannot be expressed as the sum of two composite numbers. -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l2611_261191


namespace opposite_numbers_l2611_261113

theorem opposite_numbers : ((-5)^2 : ℤ) = -(-5^2) :=
sorry

end opposite_numbers_l2611_261113


namespace min_value_complex_expression_l2611_261154

variable (a b c : ℤ)
variable (ω : ℂ)

theorem min_value_complex_expression (h1 : a * b * c = 60)
                                     (h2 : ω ≠ 1)
                                     (h3 : ω^3 = 1) :
  ∃ (min : ℝ), min = Real.sqrt 3 ∧
    ∀ (x y z : ℤ), x * y * z = 60 →
      Complex.abs (↑x + ↑y * ω + ↑z * ω^2) ≥ min :=
by sorry

end min_value_complex_expression_l2611_261154


namespace intersection_A_B_l2611_261196

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_A_B_l2611_261196


namespace cycling_route_length_l2611_261153

/-- The total length of a rectangular cycling route -/
def total_length (upper_horizontal : ℝ) (left_vertical : ℝ) : ℝ :=
  2 * (upper_horizontal + left_vertical)

/-- Theorem: The total length of the cycling route is 52 km -/
theorem cycling_route_length :
  let upper_horizontal := 4 + 7 + 2
  let left_vertical := 6 + 7
  total_length upper_horizontal left_vertical = 52 := by
  sorry

end cycling_route_length_l2611_261153


namespace condition1_condition2_man_work_twice_boy_work_l2611_261139

/-- The daily work done by a man -/
def M : ℝ := sorry

/-- The daily work done by a boy -/
def B : ℝ := sorry

/-- The total work to be done -/
def total_work : ℝ := sorry

/-- First condition: 12 men and 16 boys can do the work in 5 days -/
theorem condition1 : 5 * (12 * M + 16 * B) = total_work := sorry

/-- Second condition: 13 men and 24 boys can do the work in 4 days -/
theorem condition2 : 4 * (13 * M + 24 * B) = total_work := sorry

/-- Theorem to prove: The daily work done by a man is twice that of a boy -/
theorem man_work_twice_boy_work : M = 2 * B := by sorry

end condition1_condition2_man_work_twice_boy_work_l2611_261139


namespace triangle_properties_l2611_261100

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Define triangle ABC
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  -- Cosine law
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Statement A
  (a / Real.cos A = b / Real.sin B → A = π/4) ∧
  -- Statement D
  (A < π/2 ∧ B < π/2 ∧ C < π/2 → Real.sin A + Real.sin B > Real.cos A + Real.cos B) :=
by sorry

end triangle_properties_l2611_261100


namespace square_sum_equals_three_l2611_261162

theorem square_sum_equals_three (a b : ℝ) (h : a^4 + b^4 = a^2 - 2*a^2*b^2 + b^2 + 6) : 
  a^2 + b^2 = 3 := by
sorry

end square_sum_equals_three_l2611_261162


namespace min_value_theorem_l2611_261145

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) :
  (1 / a + 1 / b ≥ 2 * Real.sqrt 2) ∧ (b / a^3 + a / b^3 ≥ 4) := by
  sorry

end min_value_theorem_l2611_261145


namespace eighth_square_fully_shaded_l2611_261108

/-- Represents the number of shaded squares and total squares in the nth diagram -/
def squarePattern (n : ℕ) : ℕ := n^2

/-- The fraction of shaded squares in the nth diagram -/
def shadedFraction (n : ℕ) : ℚ := squarePattern n / squarePattern n

theorem eighth_square_fully_shaded :
  shadedFraction 8 = 1 := by sorry

end eighth_square_fully_shaded_l2611_261108


namespace similar_triangle_perimeter_l2611_261135

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter (small large : Triangle) :
  small.isIsosceles ∧
  small.a = 15 ∧ small.b = 15 ∧ small.c = 6 ∧
  small.isSimilar large ∧
  large.c = 18 →
  large.perimeter = 108 := by
  sorry

end similar_triangle_perimeter_l2611_261135


namespace hall_width_to_length_ratio_l2611_261188

/-- Represents a rectangular hall -/
structure RectangularHall where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular hall -/
def HallProperties (hall : RectangularHall) : Prop :=
  hall.width > 0 ∧ 
  hall.length > 0 ∧
  hall.width * hall.length = 128 ∧ 
  hall.length - hall.width = 8

theorem hall_width_to_length_ratio 
  (hall : RectangularHall) 
  (h : HallProperties hall) : 
  hall.width / hall.length = 1 / 2 := by
  sorry

end hall_width_to_length_ratio_l2611_261188


namespace frank_candy_bags_l2611_261136

/-- The number of bags Frank used to store his candy -/
def num_bags (total_candy : ℕ) (candy_per_bag : ℕ) : ℕ :=
  total_candy / candy_per_bag

/-- Theorem: Frank used 26 bags to store his candy -/
theorem frank_candy_bags : num_bags 858 33 = 26 := by
  sorry

end frank_candy_bags_l2611_261136


namespace purchase_cost_l2611_261111

/-- The cost of a single pencil in dollars -/
def pencil_cost : ℚ := 2.5

/-- The cost of a single pen in dollars -/
def pen_cost : ℚ := 3.5

/-- The number of pencils bought -/
def num_pencils : ℕ := 38

/-- The number of pens bought -/
def num_pens : ℕ := 56

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := pencil_cost * num_pencils + pen_cost * num_pens

theorem purchase_cost : total_cost = 291 := by sorry

end purchase_cost_l2611_261111


namespace diameter_segments_length_l2611_261112

theorem diameter_segments_length (r : ℝ) (chord_length : ℝ) :
  r = 6 ∧ chord_length = 10 →
  ∃ (a b : ℝ), a + b = 2 * r ∧ a * b = (chord_length / 2) ^ 2 ∧
  a = 6 - Real.sqrt 11 ∧ b = 6 + Real.sqrt 11 := by
  sorry

end diameter_segments_length_l2611_261112


namespace integer_roots_of_polynomial_l2611_261123

/-- Represents a polynomial of degree 4 with rational coefficients -/
structure Polynomial4 where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : Polynomial4) (x : ℝ) : Prop :=
  x^4 + p.a * x^2 + p.b * x + p.c = 0

theorem integer_roots_of_polynomial (p : Polynomial4) :
  isRoot p (2 - Real.sqrt 5) →
  (∃ (r₁ r₂ : ℤ), isRoot p (r₁ : ℝ) ∧ isRoot p (r₂ : ℝ)) →
  ∃ (r : ℤ), isRoot p (r : ℝ) ∧ r = -2 :=
sorry

end integer_roots_of_polynomial_l2611_261123


namespace equation_solution_l2611_261119

theorem equation_solution (x : ℝ) : 
  x^6 - 22*x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5)/2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5)/2) :=
sorry

end equation_solution_l2611_261119


namespace rectangular_prism_diagonal_l2611_261194

theorem rectangular_prism_diagonal : 
  let a : ℝ := 12
  let b : ℝ := 24
  let c : ℝ := 15
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  diagonal = 3 * Real.sqrt 105 := by
  sorry

end rectangular_prism_diagonal_l2611_261194


namespace penelope_candy_count_l2611_261114

/-- Given a ratio of M&M candies to Starbursts candies and a number of Starbursts,
    calculate the number of M&M candies. -/
def calculate_mm_candies (mm_ratio : ℕ) (starburst_ratio : ℕ) (starburst_count : ℕ) : ℕ :=
  (starburst_count / starburst_ratio) * mm_ratio

/-- Theorem stating that given the specific ratio and Starburst count,
    the number of M&M candies is 25. -/
theorem penelope_candy_count :
  calculate_mm_candies 5 3 15 = 25 := by
  sorry

end penelope_candy_count_l2611_261114


namespace min_value_sum_l2611_261109

theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : 1/p + 1/q + 1/r = 1) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 1 → p + q + r ≤ x + y + z ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 1 ∧ a + b + c = 9 := by
  sorry

end min_value_sum_l2611_261109


namespace customers_stayed_behind_l2611_261166

theorem customers_stayed_behind (initial_customers : ℕ) 
  (h1 : initial_customers = 11) 
  (stayed : ℕ) 
  (left : ℕ) 
  (h2 : left = stayed + 5) 
  (h3 : stayed + left = initial_customers) : 
  stayed = 3 := by
  sorry

end customers_stayed_behind_l2611_261166


namespace inequality_solution_set_l2611_261125

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, (a * x) / (x - 1) < 1 ↔ (x < b ∨ x > 3)) →
  (a * 3) / (3 - 1) = 1 →
  a - b = -1/3 := by
    sorry

end inequality_solution_set_l2611_261125


namespace math_problem_proof_l2611_261121

theorem math_problem_proof (b m n : ℕ) (B C : ℝ) (D : ℝ) :
  b = 4 →
  m = 1 →
  n = 1 →
  (b^m)^n + b^(m+n) = 20 →
  2^20 = B^10 →
  B > 0 →
  Real.sqrt ((20 * B + 45) / C) = C →
  D = C * Real.sin (30 * π / 180) →
  A = 20 ∧ B = 4 ∧ C = 5 ∧ D = 2.5 :=
by sorry

end math_problem_proof_l2611_261121


namespace zias_club_size_l2611_261137

/-- Represents the number of people with one coin -/
def one_coin_people : ℕ := 7

/-- Represents the angle of the smallest sector in degrees -/
def smallest_sector : ℕ := 35

/-- Represents the angle increment between sectors in degrees -/
def angle_increment : ℕ := 10

/-- Calculates the total number of sectors in the pie chart -/
def total_sectors : ℕ := 6

/-- Represents the total angle of a full circle in degrees -/
def full_circle : ℕ := 360

/-- Theorem: The number of people in Zia's club is 72 -/
theorem zias_club_size : 
  (full_circle / (smallest_sector / one_coin_people) : ℕ) = 72 := by
  sorry

end zias_club_size_l2611_261137


namespace max_value_of_expression_l2611_261157

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
sorry

end max_value_of_expression_l2611_261157


namespace crazy_silly_school_movies_l2611_261175

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 11

/-- The difference between the number of movies and books -/
def movie_book_difference : ℕ := 6

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := num_books + movie_book_difference

theorem crazy_silly_school_movies :
  num_movies = 17 := by
  sorry

end crazy_silly_school_movies_l2611_261175


namespace equation_solution_exists_l2611_261120

theorem equation_solution_exists : ∃ (x y z : ℕ+), x + y + z + 4 = 10 := by
  sorry

end equation_solution_exists_l2611_261120


namespace min_value_quadratic_l2611_261171

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 + 6*x ≥ -9) ∧ (∃ x, x^2 + 6*x = -9) := by
  sorry

end min_value_quadratic_l2611_261171


namespace intersection_implies_a_equals_one_l2611_261130

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem intersection_implies_a_equals_one :
  ∀ a : ℝ, (A ∩ B a = {3}) → a = 1 := by
sorry

end intersection_implies_a_equals_one_l2611_261130


namespace system_solution_l2611_261146

def solution_set := {x : ℝ | 0 < x ∧ x < 1}

theorem system_solution : 
  {x : ℝ | x * (x + 2) > 0 ∧ |x| < 1} = solution_set :=
by sorry

end system_solution_l2611_261146


namespace terrell_hike_distance_l2611_261143

theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end terrell_hike_distance_l2611_261143


namespace salad_dressing_composition_l2611_261122

theorem salad_dressing_composition (vinegar_p : ℝ) (oil_p : ℝ) (vinegar_q : ℝ) (oil_q : ℝ) 
  (ratio_p : ℝ) (ratio_q : ℝ) (vinegar_new : ℝ) :
  vinegar_p = 0.3 →
  vinegar_p + oil_p = 1 →
  vinegar_q = 0.1 →
  oil_q = 0.9 →
  ratio_p = 0.1 →
  ratio_q = 0.9 →
  ratio_p + ratio_q = 1 →
  vinegar_new = 0.12 →
  ratio_p * vinegar_p + ratio_q * vinegar_q = vinegar_new →
  oil_p = 0.7 := by
sorry

end salad_dressing_composition_l2611_261122


namespace sqrt2_irrational_sqrt2_approximation_no_exact_rational_sqrt2_l2611_261199

-- Define √2 as an irrational number
noncomputable def sqrt2 : ℝ := Real.sqrt 2

-- Statement that √2 is irrational
theorem sqrt2_irrational : Irrational sqrt2 := sorry

-- Statement that √2 can be approximated by rationals
theorem sqrt2_approximation :
  ∀ ε > 0, ∃ p q : ℤ, q ≠ 0 ∧ |((p : ℝ) / q)^2 - 2| < ε := sorry

-- Statement that no rational number exactly equals √2
theorem no_exact_rational_sqrt2 :
  ¬∃ p q : ℤ, q ≠ 0 ∧ ((p : ℝ) / q)^2 = 2 := sorry

end sqrt2_irrational_sqrt2_approximation_no_exact_rational_sqrt2_l2611_261199


namespace equation_solution_l2611_261110

theorem equation_solution : ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 := by
  sorry

end equation_solution_l2611_261110


namespace units_digit_of_100_factorial_l2611_261140

theorem units_digit_of_100_factorial (n : ℕ) : n = 100 → n.factorial % 10 = 0 := by sorry

end units_digit_of_100_factorial_l2611_261140


namespace kite_area_is_102_l2611_261138

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : Int
  y : Int

/-- Represents a kite shape -/
structure Kite where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Calculate the area of a kite given its vertices -/
def kiteArea (k : Kite) : Int :=
  sorry

/-- The kite in the problem -/
def problemKite : Kite := {
  p1 := { x := 0, y := 10 }
  p2 := { x := 6, y := 14 }
  p3 := { x := 12, y := 10 }
  p4 := { x := 6, y := 0 }
}

theorem kite_area_is_102 : kiteArea problemKite = 102 := by
  sorry

end kite_area_is_102_l2611_261138


namespace intersecting_lines_k_value_l2611_261141

/-- Two lines intersecting on the y-axis -/
structure IntersectingLines where
  k : ℝ
  line1 : ℝ → ℝ → ℝ := fun x y => 2*x + 3*y - k
  line2 : ℝ → ℝ → ℝ := fun x y => x - k*y + 12
  intersect_on_y_axis : ∃ y, line1 0 y = 0 ∧ line2 0 y = 0

/-- The value of k for intersecting lines -/
theorem intersecting_lines_k_value (l : IntersectingLines) : l.k = 6 ∨ l.k = -6 := by
  sorry

end intersecting_lines_k_value_l2611_261141


namespace ab_and_a_reciprocal_b_relationship_l2611_261148

theorem ab_and_a_reciprocal_b_relationship (a b : ℝ) (h : a * b ≠ 0) :
  ¬(∀ a b, a * b > 1 → a > 1 / b) ∧ 
  ¬(∀ a b, a > 1 / b → a * b > 1) ∧
  ¬(∀ a b, a * b > 1 ↔ a > 1 / b) :=
by sorry

end ab_and_a_reciprocal_b_relationship_l2611_261148


namespace intersecting_lines_k_value_l2611_261180

/-- Given two lines p and q that intersect at a point, prove the value of k -/
theorem intersecting_lines_k_value (k : ℝ) : 
  let p : ℝ → ℝ := λ x => -2 * x + 3
  let q : ℝ → ℝ := λ x => k * x + 9
  (p 6 = -9) ∧ (q 6 = -9) → k = -3 := by
  sorry

end intersecting_lines_k_value_l2611_261180


namespace trackball_mice_count_l2611_261131

theorem trackball_mice_count (total : ℕ) (wireless_ratio optical_ratio : ℚ) : 
  total = 80 →
  wireless_ratio = 1/2 →
  optical_ratio = 1/4 →
  (wireless_ratio + optical_ratio + (1 - wireless_ratio - optical_ratio)) = 1 →
  ↑total * (1 - wireless_ratio - optical_ratio) = 20 :=
by sorry

end trackball_mice_count_l2611_261131


namespace prove_january_salary_l2611_261126

def january_salary (feb mar apr may : ℕ) : Prop :=
  let jan := 32000 - (feb + mar + apr)
  (feb + mar + apr + may) / 4 = 8100 ∧
  (jan + feb + mar + apr) / 4 = 8000 ∧
  may = 6500 →
  jan = 6100

theorem prove_january_salary :
  ∀ (feb mar apr may : ℕ),
  january_salary feb mar apr may :=
by
  sorry

end prove_january_salary_l2611_261126


namespace matrix_inverse_scalar_multiple_l2611_261156

-- Define the matrix A
def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 3; 5, d]

-- State the theorem
theorem matrix_inverse_scalar_multiple
  (d k : ℝ) :
  (A d)⁻¹ = k • (A d) →
  d = -2 ∧ k = 1/19 :=
sorry

end matrix_inverse_scalar_multiple_l2611_261156


namespace max_value_of_A_l2611_261169

theorem max_value_of_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 := by
  sorry

end max_value_of_A_l2611_261169


namespace repeating_decimal_to_fraction_l2611_261184

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  x.nonRepeating + x.repeating / (1 - (1 / 10 ^ x.repeatingDigits))

theorem repeating_decimal_to_fraction :
  let x : RepeatingDecimal := { nonRepeating := 7/10, repeating := 36/100, repeatingDigits := 2 }
  x.toRational = 27 / 37 := by
  sorry

end repeating_decimal_to_fraction_l2611_261184


namespace complex_subtraction_simplification_l2611_261187

theorem complex_subtraction_simplification :
  (7 : ℂ) - 5*I - ((3 : ℂ) - 7*I) = (4 : ℂ) + 2*I :=
by sorry

end complex_subtraction_simplification_l2611_261187


namespace right_angled_triangle_set_l2611_261165

theorem right_angled_triangle_set : ∃ (a b c : ℝ), 
  (a = Real.sqrt 2 ∧ b = Real.sqrt 3 ∧ c = Real.sqrt 5) ∧ 
  a^2 + b^2 = c^2 ∧ 
  (∀ (x y z : ℝ), 
    ((x = Real.sqrt 3 ∧ y = 2 ∧ z = Real.sqrt 5) ∨ 
     (x = 3 ∧ y = 4 ∧ z = 5) ∨ 
     (x = 1 ∧ y = 2 ∧ z = 3)) → 
    x^2 + y^2 ≠ z^2) :=
by sorry

end right_angled_triangle_set_l2611_261165


namespace min_operations_cube_l2611_261118

/-- Represents a rhombus configuration --/
structure RhombusConfig :=
  (n : ℕ)
  (rhombuses : ℕ)

/-- Represents a rearrangement operation --/
inductive RearrangementOp
  | insert
  | remove

/-- The minimum number of operations to transform the configuration --/
def min_operations (config : RhombusConfig) : ℕ :=
  config.n^3

/-- Theorem stating that the minimum number of operations is n³ --/
theorem min_operations_cube (config : RhombusConfig) 
  (h1 : config.rhombuses = 3 * config.n^2) :
  min_operations config = config.n^3 := by
  sorry

#check min_operations_cube

end min_operations_cube_l2611_261118


namespace investment_comparison_l2611_261170

def initial_investment : ℝ := 200

def delta_year1_change : ℝ := 1.10
def delta_year2_change : ℝ := 0.90

def echo_year1_change : ℝ := 0.70
def echo_year2_change : ℝ := 1.50

def foxtrot_year1_change : ℝ := 1.00
def foxtrot_year2_change : ℝ := 0.95

def final_delta : ℝ := initial_investment * delta_year1_change * delta_year2_change
def final_echo : ℝ := initial_investment * echo_year1_change * echo_year2_change
def final_foxtrot : ℝ := initial_investment * foxtrot_year1_change * foxtrot_year2_change

theorem investment_comparison : final_foxtrot < final_delta ∧ final_delta < final_echo := by
  sorry

end investment_comparison_l2611_261170


namespace pencil_cost_l2611_261115

/-- Given that 150 pencils cost $45, prove that 3200 pencils cost $960 -/
theorem pencil_cost (box_size : ℕ) (box_cost : ℚ) (target_quantity : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  target_quantity = 3200 →
  (target_quantity : ℚ) * (box_cost / box_size) = 960 :=
by
  sorry

end pencil_cost_l2611_261115


namespace smallest_difference_l2611_261151

def Digits : Finset Nat := {0, 2, 4, 5, 7}

def is_valid_arrangement (a b c d x y z : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  x ∈ Digits ∧ y ∈ Digits ∧ z ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ x ∧ a ≠ y ∧ a ≠ z ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ x ∧ b ≠ y ∧ b ≠ z ∧
  c ≠ d ∧ c ≠ x ∧ c ≠ y ∧ c ≠ z ∧
  d ≠ x ∧ d ≠ y ∧ d ≠ z ∧
  x ≠ y ∧ x ≠ z ∧
  y ≠ z ∧
  a ≠ 0 ∧ x ≠ 0

def difference (a b c d x y z : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d - (100 * x + 10 * y + z)

theorem smallest_difference :
  ∀ a b c d x y z,
    is_valid_arrangement a b c d x y z →
    difference a b c d x y z ≥ 1325 :=
by sorry

end smallest_difference_l2611_261151


namespace fifty_bees_honey_production_l2611_261129

/-- The amount of honey (in grams) produced by a given number of bees in 50 days -/
def honey_production (num_bees : ℕ) : ℕ :=
  num_bees * 1

theorem fifty_bees_honey_production :
  honey_production 50 = 50 := by sorry

end fifty_bees_honey_production_l2611_261129


namespace opposite_of_negative_2023_l2611_261147

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = 2023 := by
  sorry

end opposite_of_negative_2023_l2611_261147


namespace dennis_teaching_years_l2611_261177

/-- Given the teaching years of Virginia, Adrienne, and Dennis, prove that Dennis has taught for 46 years. -/
theorem dennis_teaching_years 
  (total : ℕ) 
  (h_total : total = 102)
  (h_virginia_adrienne : ∃ (a : ℕ), virginia = a + 9)
  (h_virginia_dennis : ∃ (d : ℕ), virginia = d - 9)
  (h_sum : virginia + adrienne + dennis = total)
  : dennis = 46 := by
  sorry

end dennis_teaching_years_l2611_261177


namespace vitamin_a_content_l2611_261173

/-- The amount of Vitamin A in a single pill, in mg -/
def vitamin_a_per_pill : ℝ := 50

/-- The recommended daily serving of Vitamin A, in mg -/
def daily_recommended : ℝ := 200

/-- The number of pills needed for the weekly recommended amount -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem vitamin_a_content :
  vitamin_a_per_pill = daily_recommended * (days_per_week : ℝ) / (pills_per_week : ℝ) := by
  sorry

end vitamin_a_content_l2611_261173


namespace map_scale_l2611_261189

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (map_km : ℝ) (actual_cm : ℝ)
  (h1 : map_cm = 15)
  (h2 : map_km = 90)
  (h3 : actual_cm = 20) :
  (actual_cm / map_cm) * map_km = 120 := by
  sorry

end map_scale_l2611_261189


namespace distance_from_origin_l2611_261142

theorem distance_from_origin (x : ℝ) : 
  |x| = Real.sqrt 5 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by sorry

end distance_from_origin_l2611_261142


namespace permutations_mod_1000_l2611_261104

/-- The number of characters in the string -/
def n : ℕ := 16

/-- The number of A's in the string -/
def num_a : ℕ := 4

/-- The number of B's in the string -/
def num_b : ℕ := 5

/-- The number of C's in the string -/
def num_c : ℕ := 4

/-- The number of D's in the string -/
def num_d : ℕ := 3

/-- The number of positions where A's cannot be placed -/
def no_a_positions : ℕ := 5

/-- The number of positions where B's cannot be placed -/
def no_b_positions : ℕ := 5

/-- The number of positions where C's and D's cannot be placed -/
def no_cd_positions : ℕ := 6

/-- The function that calculates the number of permutations satisfying the conditions -/
def permutations : ℕ :=
  (Nat.choose no_cd_positions num_d) *
  (Nat.choose (no_cd_positions - num_d) (num_c - (no_cd_positions - num_d))) *
  (Nat.choose no_a_positions num_b) *
  (Nat.choose no_b_positions num_a)

theorem permutations_mod_1000 :
  permutations ≡ 75 [MOD 1000] := by sorry

end permutations_mod_1000_l2611_261104


namespace train_length_calculation_l2611_261103

/-- Conversion factor from km/hr to m/s -/
def kmhr_to_ms : ℚ := 5 / 18

/-- Calculate the length of a train given its speed in km/hr and crossing time in seconds -/
def train_length (speed : ℚ) (time : ℚ) : ℚ :=
  speed * kmhr_to_ms * time

/-- The cumulative length of two trains -/
def cumulative_length (speed1 speed2 time1 time2 : ℚ) : ℚ :=
  train_length speed1 time1 + train_length speed2 time2

theorem train_length_calculation (speed1 speed2 time1 time2 : ℚ) :
  speed1 = 27 ∧ speed2 = 45 ∧ time1 = 20 ∧ time2 = 30 →
  cumulative_length speed1 speed2 time1 time2 = 525 := by
  sorry

end train_length_calculation_l2611_261103


namespace min_quotient_value_l2611_261116

def is_valid_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a = c + 1 ∧
  b = d + 1

def number_value (a b c d : ℕ) : ℕ :=
  1000 * a + 100 * b + 10 * c + d

def digit_sum (a b c d : ℕ) : ℕ :=
  a + b + c + d

def quotient (a b c d : ℕ) : ℚ :=
  (number_value a b c d : ℚ) / (digit_sum a b c d : ℚ)

theorem min_quotient_value :
  ∀ a b c d : ℕ, is_valid_number a b c d →
  quotient a b c d ≥ 192.67 :=
sorry

end min_quotient_value_l2611_261116


namespace fourth_term_of_arithmetic_sequence_l2611_261182

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem fourth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 0 = 12 →
  a 5 = 47 →
  a 3 = 29.5 := by
sorry

end fourth_term_of_arithmetic_sequence_l2611_261182


namespace ourDie_expected_value_l2611_261178

/-- Represents the four-sided die with its probabilities and winnings --/
structure UnusualDie where
  side1_prob : ℚ
  side1_win : ℚ
  side2_prob : ℚ
  side2_win : ℚ
  side3_prob : ℚ
  side3_win : ℚ
  side4_prob : ℚ
  side4_win : ℚ

/-- The specific unusual die described in the problem --/
def ourDie : UnusualDie :=
  { side1_prob := 1/4
  , side1_win := 2
  , side2_prob := 1/4
  , side2_win := 4
  , side3_prob := 1/3
  , side3_win := -6
  , side4_prob := 1/6
  , side4_win := 0 }

/-- Calculates the expected value of rolling the die --/
def expectedValue (d : UnusualDie) : ℚ :=
  d.side1_prob * d.side1_win +
  d.side2_prob * d.side2_win +
  d.side3_prob * d.side3_win +
  d.side4_prob * d.side4_win

/-- Theorem stating that the expected value of rolling ourDie is -1/2 --/
theorem ourDie_expected_value :
  expectedValue ourDie = -1/2 := by
  sorry


end ourDie_expected_value_l2611_261178


namespace village_leadership_choices_l2611_261163

/-- The number of members in the village -/
def villageSize : ℕ := 16

/-- The number of deputy mayors -/
def numDeputyMayors : ℕ := 3

/-- The number of council members per deputy mayor -/
def councilMembersPerDeputy : ℕ := 3

/-- The total number of council members -/
def totalCouncilMembers : ℕ := numDeputyMayors * councilMembersPerDeputy

/-- The number of ways to choose the village leadership -/
def leadershipChoices : ℕ := 
  villageSize * 
  (villageSize - 1) * 
  (villageSize - 2) * 
  (villageSize - 3) * 
  Nat.choose (villageSize - 4) councilMembersPerDeputy * 
  Nat.choose (villageSize - 4 - councilMembersPerDeputy) councilMembersPerDeputy * 
  Nat.choose (villageSize - 4 - 2 * councilMembersPerDeputy) councilMembersPerDeputy

theorem village_leadership_choices : 
  leadershipChoices = 154828800 := by sorry

end village_leadership_choices_l2611_261163


namespace ravenswood_gnomes_remaining_l2611_261186

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken by the forest owner -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_ravenswood_gnomes : ℕ := 48

theorem ravenswood_gnomes_remaining :
  remaining_ravenswood_gnomes = 
    (ravenswood_ratio * westerville_gnomes) - 
    (ravenswood_ratio * westerville_gnomes * taken_percentage).floor := by
  sorry

end ravenswood_gnomes_remaining_l2611_261186


namespace speed_conversion_correct_l2611_261117

/-- Conversion factor from km/h to m/s -/
def kmh_to_ms : ℝ := 0.277778

/-- Given speed in km/h -/
def speed_kmh : ℝ := 84

/-- Equivalent speed in m/s -/
def speed_ms : ℝ := speed_kmh * kmh_to_ms

theorem speed_conversion_correct : 
  ∃ ε > 0, |speed_ms - 23.33| < ε :=
sorry

end speed_conversion_correct_l2611_261117


namespace quadratic_roots_condition_l2611_261152

theorem quadratic_roots_condition (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  n < -6 ∨ n > 6 := by
sorry

end quadratic_roots_condition_l2611_261152


namespace tricycle_wheels_count_l2611_261101

theorem tricycle_wheels_count :
  ∀ (tricycle_wheels : ℕ),
    3 * 2 + 4 * tricycle_wheels + 7 * 1 = 25 →
    tricycle_wheels = 3 :=
by
  sorry

end tricycle_wheels_count_l2611_261101


namespace replaced_girl_weight_l2611_261183

theorem replaced_girl_weight
  (n : ℕ)
  (original_average : ℝ)
  (new_average : ℝ)
  (new_girl_weight : ℝ)
  (h1 : n = 25)
  (h2 : new_average = original_average + 1)
  (h3 : new_girl_weight = 80) :
  ∃ (replaced_weight : ℝ),
    replaced_weight = new_girl_weight - n * (new_average - original_average) ∧
    replaced_weight = 55 := by
  sorry

end replaced_girl_weight_l2611_261183


namespace bouquet_count_l2611_261172

/-- The number of narcissus flowers available -/
def narcissus : ℕ := 75

/-- The number of chrysanthemums available -/
def chrysanthemums : ℕ := 90

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The total number of bouquets that can be made -/
def total_bouquets : ℕ := (narcissus / flowers_per_bouquet) + (chrysanthemums / flowers_per_bouquet)

theorem bouquet_count : total_bouquets = 33 := by
  sorry

end bouquet_count_l2611_261172


namespace cubic_yard_to_cubic_inches_l2611_261132

-- Define the conversion factor
def inches_per_yard : ℕ := 36

-- Theorem statement
theorem cubic_yard_to_cubic_inches :
  (inches_per_yard ^ 3 : ℕ) = 46656 :=
sorry

end cubic_yard_to_cubic_inches_l2611_261132


namespace divisors_of_1728_power_1728_l2611_261197

theorem divisors_of_1728_power_1728 :
  ∃! n : ℕ, n = (Finset.filter
    (fun d => (Finset.filter (fun x => x ∣ d) (Finset.range (d + 1))).card = 1728)
    (Finset.filter (fun x => x ∣ 1728^1728) (Finset.range (1728^1728 + 1)))).card :=
by sorry

end divisors_of_1728_power_1728_l2611_261197


namespace printer_ink_problem_l2611_261149

/-- The problem of calculating the additional money needed for printer inks --/
theorem printer_ink_problem (initial_amount : ℕ) (black_cost red_cost yellow_cost : ℕ)
  (black_quantity red_quantity yellow_quantity : ℕ) : 
  initial_amount = 50 →
  black_cost = 11 →
  red_cost = 15 →
  yellow_cost = 13 →
  black_quantity = 2 →
  red_quantity = 3 →
  yellow_quantity = 2 →
  (black_cost * black_quantity + red_cost * red_quantity + yellow_cost * yellow_quantity) - initial_amount = 43 := by
  sorry

#check printer_ink_problem

end printer_ink_problem_l2611_261149


namespace operation_result_l2611_261161

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem operation_result : 
  op (op Element.three Element.one) (op Element.four Element.two) = Element.two := by
  sorry

end operation_result_l2611_261161


namespace set_operations_theorem_l2611_261193

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem set_operations_theorem :
  (Set.compl A ∪ B = {x | x < 5}) ∧
  (A ∩ Set.compl B = {x | x ≥ 5}) := by sorry

end set_operations_theorem_l2611_261193


namespace light_year_scientific_notation_l2611_261176

def light_year : ℝ := 9500000000000

theorem light_year_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), light_year = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 12 ∧ a = 9.5 :=
by sorry

end light_year_scientific_notation_l2611_261176


namespace cookie_average_l2611_261159

theorem cookie_average (packages : List Nat) 
  (h1 : packages = [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]) : 
  (packages.sum : Rat) / packages.length = 27/2 := by
  sorry

end cookie_average_l2611_261159


namespace suv_coupe_price_ratio_l2611_261198

theorem suv_coupe_price_ratio 
  (coupe_price : ℝ) 
  (commission_rate : ℝ) 
  (total_commission : ℝ) 
  (h1 : coupe_price = 30000)
  (h2 : commission_rate = 0.02)
  (h3 : total_commission = 1800)
  (h4 : ∃ x : ℝ, commission_rate * (coupe_price + x * coupe_price) = total_commission) :
  ∃ x : ℝ, x * coupe_price = 2 * coupe_price := by
sorry

end suv_coupe_price_ratio_l2611_261198


namespace isosceles_triangle_perimeter_l2611_261181

theorem isosceles_triangle_perimeter (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) →
  (∃ (base leg : ℝ), 
    (base^2 - 6*base + 8 = 0) ∧ 
    (leg^2 - 6*leg + 8 = 0) ∧
    (base ≠ leg) ∧
    (base + 2*leg = 10)) :=
by sorry

end isosceles_triangle_perimeter_l2611_261181


namespace congruence_problem_l2611_261179

theorem congruence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n < 25 ∧ -175 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end congruence_problem_l2611_261179


namespace fourth_grade_students_l2611_261185

/-- Calculates the final number of students in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 43 -/
theorem fourth_grade_students : final_student_count 4 3 42 = 43 := by
  sorry

end fourth_grade_students_l2611_261185


namespace problem_statement_l2611_261133

theorem problem_statement (a b x y : ℝ) 
  (sum_ab : a + b = 2)
  (sum_xy : x + y = 2)
  (product_sum : a * x + b * y = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
sorry

end problem_statement_l2611_261133


namespace father_son_age_difference_l2611_261155

/-- Proves that a father is 25 years older than his son given the problem conditions -/
theorem father_son_age_difference :
  ∀ (father_age son_age : ℕ),
    father_age > son_age →
    son_age = 23 →
    father_age + 2 = 2 * (son_age + 2) →
    father_age - son_age = 25 := by
  sorry

end father_son_age_difference_l2611_261155


namespace trig_expression_equality_l2611_261105

theorem trig_expression_equality (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π)  -- α is in the second quadrant
  (h2 : Real.sin (π/2 + α) = -Real.sqrt 5 / 5) :
  (Real.cos α ^ 3 + Real.sin α) / Real.cos (α - π/4) = 9 * Real.sqrt 2 / 5 := by
  sorry

end trig_expression_equality_l2611_261105


namespace siblings_total_age_l2611_261174

/-- Given the age ratio of Halima, Beckham, and Michelle as 4:3:7, and the age difference
    between Halima and Beckham as 9 years, prove that the total age of the three siblings
    is 126 years. -/
theorem siblings_total_age
  (halima_ratio : ℕ) (beckham_ratio : ℕ) (michelle_ratio : ℕ)
  (age_ratio : halima_ratio = 4 ∧ beckham_ratio = 3 ∧ michelle_ratio = 7)
  (age_difference : ℕ) (halima_beckham_diff : age_difference = 9)
  : ∃ (x : ℕ), 
    halima_ratio * x - beckham_ratio * x = age_difference ∧
    halima_ratio * x + beckham_ratio * x + michelle_ratio * x = 126 := by
  sorry

end siblings_total_age_l2611_261174


namespace cube_surface_area_increase_l2611_261127

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_area := 6 * s^2
  let new_edge := 1.6 * s
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 1.56 := by
sorry

end cube_surface_area_increase_l2611_261127


namespace ribbon_division_l2611_261128

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) (ribbon_per_box : ℚ) :
  total_ribbon = 5 / 8 →
  num_boxes = 5 →
  ribbon_per_box = total_ribbon / num_boxes →
  ribbon_per_box = 1 / 8 :=
by sorry

end ribbon_division_l2611_261128


namespace perpendicular_line_through_point_l2611_261160

/-- Given a line L1 with equation x - y + 2 = 0 and a point P (1, 0),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x + y - 1 = 0 -/
theorem perpendicular_line_through_point (L1 L2 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (L1 = {(x, y) | x - y + 2 = 0}) →
  (P = (1, 0)) →
  (L2 = {(x, y) | (x, y) ∈ L2 ∧ (∀ (a b : ℝ × ℝ), a ∈ L1 → b ∈ L1 → (a.1 - b.1) * (P.1 - x) + (a.2 - b.2) * (P.2 - y) = 0)}) →
  (L2 = {(x, y) | x + y - 1 = 0}) :=
by sorry

end perpendicular_line_through_point_l2611_261160


namespace five_digit_divisible_by_36_l2611_261107

theorem five_digit_divisible_by_36 (n : ℕ) : 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a b : ℕ, n = a * 10000 + 1000 + 200 + 30 + b) ∧  -- form ⬜123⬜
  (n % 36 = 0) →  -- divisible by 36
  (n = 11232 ∨ n = 61236) := by
sorry

end five_digit_divisible_by_36_l2611_261107


namespace log_range_l2611_261164

def log_defined (a : ℝ) : Prop :=
  a - 2 > 0 ∧ a - 2 ≠ 1 ∧ 5 - a > 0

theorem log_range : 
  {a : ℝ | log_defined a} = {a : ℝ | (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5)} :=
by sorry

end log_range_l2611_261164


namespace shopkeeper_loss_percentage_l2611_261167

theorem shopkeeper_loss_percentage
  (profit_rate : ℝ)
  (theft_rate : ℝ)
  (h_profit : profit_rate = 0.1)
  (h_theft : theft_rate = 0.2) :
  let selling_price := 1 + profit_rate
  let remaining_goods := 1 - theft_rate
  let cost_price_remaining := remaining_goods
  let selling_price_remaining := selling_price * remaining_goods
  let loss := theft_rate
  loss / cost_price_remaining = 0.25 := by sorry

end shopkeeper_loss_percentage_l2611_261167


namespace alicia_tax_deduction_l2611_261158

/-- Calculates the tax deduction in cents given an hourly wage in dollars and a tax rate percentage. -/
def tax_deduction_cents (hourly_wage : ℚ) (tax_rate_percent : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate_percent / 100)

/-- Proves that Alicia's tax deduction is 50 cents per hour. -/
theorem alicia_tax_deduction :
  tax_deduction_cents 25 2 = 50 := by
  sorry

#eval tax_deduction_cents 25 2

end alicia_tax_deduction_l2611_261158


namespace system_solutions_l2611_261144

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  4 * x^2 / (1 + 4 * x^2) = y ∧
  4 * y^2 / (1 + 4 * y^2) = z ∧
  4 * z^2 / (1 + 4 * z^2) = x

-- Theorem statement
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end system_solutions_l2611_261144


namespace fraction_multiplication_l2611_261195

theorem fraction_multiplication : (2 : ℚ) / 3 * 3 / 5 * 4 / 7 * 5 / 8 = 1 / 7 := by
  sorry

end fraction_multiplication_l2611_261195
