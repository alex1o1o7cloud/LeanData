import Mathlib

namespace NUMINAMATH_CALUDE_positive_difference_l2781_278137

theorem positive_difference (x y w : ℝ) 
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hw : 0.5 < w ∧ w < 1) : 
  w - y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_l2781_278137


namespace NUMINAMATH_CALUDE_binomial_12_6_l2781_278181

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_6_l2781_278181


namespace NUMINAMATH_CALUDE_parabola_vertex_l2781_278178

/-- The parabola defined by y = -2(x-2)^2 - 5 has vertex (2, -5) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * (x - 2)^2 - 5 → (2, -5) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2781_278178


namespace NUMINAMATH_CALUDE_area_ratio_is_9_32_l2781_278119

-- Define the triangle XYZ
structure Triangle :=
  (XY YZ XZ : ℝ)

-- Define the points M, N, O
structure Points (t : Triangle) :=
  (p q r : ℝ)
  (p_pos : p > 0)
  (q_pos : q > 0)
  (r_pos : r > 0)
  (sum_eq : p + q + r = 3/4)
  (sum_sq_eq : p^2 + q^2 + r^2 = 1/2)

-- Define the function to calculate the ratio of areas
def areaRatio (t : Triangle) (pts : Points t) : ℝ :=
  -- The actual calculation of the ratio would go here
  sorry

-- State the theorem
theorem area_ratio_is_9_32 (t : Triangle) (pts : Points t) 
  (h1 : t.XY = 12) (h2 : t.YZ = 16) (h3 : t.XZ = 20) : 
  areaRatio t pts = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_9_32_l2781_278119


namespace NUMINAMATH_CALUDE_sum_inequality_l2781_278112

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2781_278112


namespace NUMINAMATH_CALUDE_cat_catches_rat_l2781_278165

/-- The time (in hours) it takes for the cat to catch the rat after it starts chasing -/
def catchTime : ℝ := 4

/-- The average speed of the cat in km/h -/
def catSpeed : ℝ := 90

/-- The average speed of the rat in km/h -/
def ratSpeed : ℝ := 36

/-- The time (in hours) the cat waits before chasing the rat -/
def waitTime : ℝ := 6

theorem cat_catches_rat : 
  catchTime * catSpeed = (catchTime + waitTime) * ratSpeed :=
by sorry

end NUMINAMATH_CALUDE_cat_catches_rat_l2781_278165


namespace NUMINAMATH_CALUDE_max_gangsters_is_35_l2781_278190

/-- Represents a gang in Chicago -/
structure Gang :=
  (id : Nat)

/-- Represents a gangster in Chicago -/
structure Gangster :=
  (id : Nat)

/-- The total number of gangs in Chicago -/
def totalGangs : Nat := 36

/-- Represents the conflict relation between gangs -/
def inConflict : Gang → Gang → Prop := sorry

/-- Represents the membership of a gangster in a gang -/
def isMember : Gangster → Gang → Prop := sorry

/-- All gangsters belong to multiple gangs -/
axiom multiple_membership (g : Gangster) : ∃ (g1 g2 : Gang), g1 ≠ g2 ∧ isMember g g1 ∧ isMember g g2

/-- Any two gangsters belong to different sets of gangs -/
axiom different_memberships (g1 g2 : Gangster) : g1 ≠ g2 → ∃ (gang : Gang), (isMember g1 gang ∧ ¬isMember g2 gang) ∨ (isMember g2 gang ∧ ¬isMember g1 gang)

/-- No gangster belongs to two gangs that are in conflict -/
axiom no_conflict_membership (g : Gangster) (gang1 gang2 : Gang) : isMember g gang1 → isMember g gang2 → ¬inConflict gang1 gang2

/-- Each gang not including a gangster is in conflict with some gang including that gangster -/
axiom conflict_with_member_gang (g : Gangster) (gang1 : Gang) : ¬isMember g gang1 → ∃ (gang2 : Gang), isMember g gang2 ∧ inConflict gang1 gang2

/-- The maximum number of gangsters in Chicago -/
def maxGangsters : Nat := 35

/-- Theorem: The maximum number of gangsters in Chicago is 35 -/
theorem max_gangsters_is_35 : ∀ (gangsters : Finset Gangster), gangsters.card ≤ maxGangsters :=
  sorry

end NUMINAMATH_CALUDE_max_gangsters_is_35_l2781_278190


namespace NUMINAMATH_CALUDE_solve_for_a_l2781_278199

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 2

-- State the theorem
theorem solve_for_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 8 * x + 4 * a) → 
  deriv (f a) 2 = 20 → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l2781_278199


namespace NUMINAMATH_CALUDE_evan_book_difference_l2781_278124

/-- Represents the number of books Evan owns at different points in time -/
structure EvanBooks where
  twoYearsAgo : ℕ
  current : ℕ
  inFiveYears : ℕ

/-- The conditions of Evan's book collection -/
def evanBookConditions (books : EvanBooks) : Prop :=
  books.twoYearsAgo = 200 ∧
  books.current = books.twoYearsAgo - 40 ∧
  books.inFiveYears = 860

/-- The theorem stating the difference between Evan's books in five years
    and five times his current number of books -/
theorem evan_book_difference (books : EvanBooks) 
  (h : evanBookConditions books) : 
  books.inFiveYears - (5 * books.current) = 60 := by
  sorry

end NUMINAMATH_CALUDE_evan_book_difference_l2781_278124


namespace NUMINAMATH_CALUDE_smallest_a_is_54_l2781_278152

/-- A polynomial with three positive integer roots -/
structure PolynomialWithIntegerRoots where
  a : ℤ
  b : ℤ
  roots : Fin 3 → ℤ
  roots_positive : ∀ i, 0 < roots i
  polynomial_property : ∀ x, x^3 - a*x^2 + b*x - 30030 = (x - roots 0) * (x - roots 1) * (x - roots 2)

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a : ℤ := 54

/-- Theorem stating that 54 is the smallest possible value of a -/
theorem smallest_a_is_54 (p : PolynomialWithIntegerRoots) : 
  p.a ≥ smallest_a ∧ ∃ (q : PolynomialWithIntegerRoots), q.a = smallest_a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_54_l2781_278152


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l2781_278123

theorem percentage_of_percentage (x : ℝ) : (10 / 100) * ((50 / 100) * 500) = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l2781_278123


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2781_278116

theorem fraction_sum_equality : (18 : ℚ) / 45 - 3 / 8 + 1 / 9 = 49 / 360 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2781_278116


namespace NUMINAMATH_CALUDE_two_digit_multiplication_l2781_278174

theorem two_digit_multiplication (a b c d : ℕ) :
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →
  ((b = d ∧ a + c = 10) ∨ (a = c ∧ b + d = 10) ∨ (c = d ∧ a + b = 10)) →
  (10 * a + b) * (10 * c + d) = 
    (if b = d ∧ a + c = 10 then 100 * (a^2 + a) + b * d
     else if a = c ∧ b + d = 10 then 100 * a * c + 100 * b + b^2
     else 100 * a * c + 100 * c + b * c) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_l2781_278174


namespace NUMINAMATH_CALUDE_function_property_l2781_278162

theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2781_278162


namespace NUMINAMATH_CALUDE_max_squares_on_8x8_board_l2781_278192

/-- Represents a checkerboard --/
structure Checkerboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a straight line on a checkerboard --/
structure Line :=
  (board : Checkerboard)

/-- Returns the maximum number of squares a line can pass through on a checkerboard --/
def max_squares_passed (l : Line) : Nat :=
  l.board.rows + l.board.cols - 1

/-- Theorem: The maximum number of squares a straight line can pass through on an 8x8 checkerboard is 15 --/
theorem max_squares_on_8x8_board :
  ∀ (l : Line), l.board = Checkerboard.mk 8 8 → max_squares_passed l = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_squares_on_8x8_board_l2781_278192


namespace NUMINAMATH_CALUDE_odd_function_product_nonpositive_l2781_278161

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_product_nonpositive_l2781_278161


namespace NUMINAMATH_CALUDE_product_digit_sum_l2781_278196

/-- The number of 9's in the factor that, when multiplied by 9, 
    produces a number whose digits sum to 1111 -/
def k : ℕ := 124

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The factor consisting of k 9's -/
def factor (k : ℕ) : ℕ :=
  10^k - 1

theorem product_digit_sum :
  sum_of_digits (9 * factor k) = 1111 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2781_278196


namespace NUMINAMATH_CALUDE_correct_total_crayons_l2781_278131

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 41

/-- The number of crayons Sam added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after Sam's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem correct_total_crayons : total_crayons = 53 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_crayons_l2781_278131


namespace NUMINAMATH_CALUDE_equal_area_triangles_l2781_278150

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 13 13 10 = triangleArea 13 13 24 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l2781_278150


namespace NUMINAMATH_CALUDE_square_diagonal_length_l2781_278130

theorem square_diagonal_length (A : ℝ) (d : ℝ) : 
  A = 392 → d = 28 → d^2 = 2 * A :=
by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l2781_278130


namespace NUMINAMATH_CALUDE_purchases_per_customer_l2781_278184

/-- Given a parking lot scenario, prove that each customer makes exactly one purchase. -/
theorem purchases_per_customer (num_cars : ℕ) (customers_per_car : ℕ) 
  (sports_sales : ℕ) (music_sales : ℕ) 
  (h1 : num_cars = 10) 
  (h2 : customers_per_car = 5) 
  (h3 : sports_sales = 20) 
  (h4 : music_sales = 30) : 
  (sports_sales + music_sales) / (num_cars * customers_per_car) = 1 :=
by sorry

end NUMINAMATH_CALUDE_purchases_per_customer_l2781_278184


namespace NUMINAMATH_CALUDE_mode_median_mean_relationship_l2781_278160

def dataset : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode (data : List ℕ) : ℕ := sorry

def median (data : List ℕ) : ℚ := sorry

def mean (data : List ℕ) : ℚ := sorry

theorem mode_median_mean_relationship :
  let m := mode dataset
  let med := median dataset
  let μ := mean dataset
  (m : ℚ) > med ∧ med > μ := by sorry

end NUMINAMATH_CALUDE_mode_median_mean_relationship_l2781_278160


namespace NUMINAMATH_CALUDE_dave_tickets_remaining_l2781_278148

theorem dave_tickets_remaining (initial_tickets used_tickets : ℕ) :
  initial_tickets = 127 →
  used_tickets = 84 →
  initial_tickets - used_tickets = 43 :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_remaining_l2781_278148


namespace NUMINAMATH_CALUDE_no_equilateral_right_triangle_no_equilateral_obtuse_triangle_l2781_278166

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) = 180

-- Define triangle types
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∧ t.angles 1 = t.angles 2

def Triangle.isRight (t : Triangle) : Prop :=
  t.angles 0 = 90 ∨ t.angles 1 = 90 ∨ t.angles 2 = 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angles 0 > 90 ∨ t.angles 1 > 90 ∨ t.angles 2 > 90

-- Theorem stating that equilateral right triangles cannot exist
theorem no_equilateral_right_triangle :
  ∀ t : Triangle, ¬(t.isEquilateral ∧ t.isRight) :=
sorry

-- Theorem stating that equilateral obtuse triangles cannot exist
theorem no_equilateral_obtuse_triangle :
  ∀ t : Triangle, ¬(t.isEquilateral ∧ t.isObtuse) :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_right_triangle_no_equilateral_obtuse_triangle_l2781_278166


namespace NUMINAMATH_CALUDE_contrapositive_square_inequality_l2781_278142

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x^2 > y^2) → ¬(x > y)) ↔ (x^2 ≤ y^2 → x ≤ y) := by sorry

end NUMINAMATH_CALUDE_contrapositive_square_inequality_l2781_278142


namespace NUMINAMATH_CALUDE_total_marbles_count_l2781_278179

theorem total_marbles_count (num_jars : ℕ) (marbles_per_jar : ℕ) : 
  num_jars = 16 →
  marbles_per_jar = 5 →
  (∃ (num_clay_pots : ℕ), num_jars = 2 * num_clay_pots) →
  (∃ (total_marbles : ℕ), 
    total_marbles = num_jars * marbles_per_jar + 
                    (num_jars / 2) * (3 * marbles_per_jar) ∧
    total_marbles = 200) :=
by
  sorry

#check total_marbles_count

end NUMINAMATH_CALUDE_total_marbles_count_l2781_278179


namespace NUMINAMATH_CALUDE_geometry_propositions_l2781_278101

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom h₁ : p₁
axiom h₂ : ¬p₂
axiom h₃ : ¬p₃
axiom h₄ : p₄

-- Theorem to prove
theorem geometry_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2781_278101


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l2781_278115

/-- The function f(x) = x³ + x - 16 -/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_lines_theorem :
  /- Tangent lines with slope 4 -/
  (∃ x₀ y₀ : ℝ, f x₀ = y₀ ∧ f' x₀ = 4 ∧ (4*x₀ - y₀ - 18 = 0 ∨ 4*x₀ - y₀ - 14 = 0)) ∧
  /- Tangent line at point (2, -6) -/
  (f 2 = -6 ∧ f' 2 = 13 ∧ 13*2 - (-6) - 32 = 0) ∧
  /- Tangent line passing through origin -/
  (∃ x₀ : ℝ, f x₀ = f' x₀ * (-x₀) ∧ f' x₀ = 13) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l2781_278115


namespace NUMINAMATH_CALUDE_susan_board_game_movement_l2781_278193

theorem susan_board_game_movement (total_spaces : ℕ) (first_move : ℕ) (third_move : ℕ) (sent_back : ℕ) (remaining_spaces : ℕ) : 
  total_spaces = 48 →
  first_move = 8 →
  third_move = 6 →
  sent_back = 5 →
  remaining_spaces = 37 →
  first_move + third_move + remaining_spaces + sent_back = total_spaces →
  ∃ (second_move : ℕ), second_move = 28 := by
sorry

end NUMINAMATH_CALUDE_susan_board_game_movement_l2781_278193


namespace NUMINAMATH_CALUDE_not_divisible_by_97_l2781_278167

theorem not_divisible_by_97 (k : ℤ) : (99^3 - 99) % k = 0 → k ≠ 97 := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_97_l2781_278167


namespace NUMINAMATH_CALUDE_same_parity_min_max_l2781_278122

/-- A set of elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def min_element (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def max_element (A : Set ℤ) : ℤ := sorry

/-- A predicate to check if a number is even -/
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_min_max : 
  is_even (min_element A_P) ↔ is_even (max_element A_P) := by sorry

end NUMINAMATH_CALUDE_same_parity_min_max_l2781_278122


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2781_278138

theorem square_difference_given_sum_and_product (m n : ℝ) 
  (h1 : m + n = 6) (h2 : m * n = 4) : (m - n)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2781_278138


namespace NUMINAMATH_CALUDE_lewis_harvest_earnings_l2781_278141

/-- Calculates the total earnings during harvest season --/
def harvest_earnings (regular_weekly : ℕ) (overtime_weekly : ℕ) (weeks : ℕ) : ℕ :=
  (regular_weekly + overtime_weekly) * weeks

/-- Theorem stating Lewis's total earnings during harvest season --/
theorem lewis_harvest_earnings :
  harvest_earnings 28 939 1091 = 1055497 := by
  sorry

end NUMINAMATH_CALUDE_lewis_harvest_earnings_l2781_278141


namespace NUMINAMATH_CALUDE_division_equation_proof_l2781_278177

theorem division_equation_proof : (320 : ℝ) / (54 + 26) = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_equation_proof_l2781_278177


namespace NUMINAMATH_CALUDE_power_division_l2781_278168

theorem power_division (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2781_278168


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2781_278111

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -c ∧ s₁ * s₂ = a ∧
   3 * s₁ + 3 * s₂ = -a ∧ 9 * s₁ * s₂ = b) →
  b / c = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2781_278111


namespace NUMINAMATH_CALUDE_percentage_problem_l2781_278102

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 3200 →
  0.1 * N = (P / 100) * 650 + 190 →
  P = 20 :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l2781_278102


namespace NUMINAMATH_CALUDE_tillys_star_ratio_l2781_278169

/-- Proves that given the conditions of Tilly's star counting, the ratio of stars to the west to stars to the east is 6:1 -/
theorem tillys_star_ratio :
  ∀ (stars_east stars_west : ℕ),
    stars_east = 120 →
    (∃ k : ℕ, stars_west = k * stars_east) →
    stars_east + stars_west = 840 →
    stars_west / stars_east = 6 := by
  sorry

end NUMINAMATH_CALUDE_tillys_star_ratio_l2781_278169


namespace NUMINAMATH_CALUDE_square_garden_side_length_l2781_278149

/-- Given a square garden with a perimeter of 112 meters, prove that the length of each side is 28 meters. -/
theorem square_garden_side_length :
  ∀ (side_length : ℝ),
  (4 * side_length = 112) →
  side_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_square_garden_side_length_l2781_278149


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2781_278109

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 + x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2781_278109


namespace NUMINAMATH_CALUDE_triangle_theorem_l2781_278156

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3 ∧
  t.a + t.c = 6 ∧
  (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2) = (1/2) * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.B = π/3 ∧ (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2781_278156


namespace NUMINAMATH_CALUDE_decreasing_cubic_condition_l2781_278128

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

-- Define what it means for a function to be decreasing on ℝ
def IsDecreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x > g y

-- Theorem statement
theorem decreasing_cubic_condition (a : ℝ) :
  IsDecreasing (f a) → a < -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_condition_l2781_278128


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2781_278186

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_eight_ten :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2781_278186


namespace NUMINAMATH_CALUDE_debate_school_ratio_l2781_278164

/-- The number of students in the third school -/
def third_school : ℕ := 200

/-- The number of students in the second school -/
def second_school : ℕ := third_school + 40

/-- The total number of students who shook the mayor's hand -/
def total_students : ℕ := 920

/-- The number of students in the first school -/
def first_school : ℕ := total_students - second_school - third_school

/-- The ratio of students in the first school to students in the second school -/
def school_ratio : ℚ := first_school / second_school

theorem debate_school_ratio : school_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_debate_school_ratio_l2781_278164


namespace NUMINAMATH_CALUDE_polynomial_division_l2781_278146

def p (x : ℝ) : ℝ := x^5 + 3*x^4 - 28*x^3 + 45*x^2 - 58*x + 24
def d (x : ℝ) : ℝ := x - 3
def q (x : ℝ) : ℝ := x^4 + 6*x^3 - 10*x^2 + 15*x - 13
def r : ℝ := -15

theorem polynomial_division :
  ∀ x : ℝ, p x = d x * q x + r :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2781_278146


namespace NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l2781_278100

theorem r_fourth_plus_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l2781_278100


namespace NUMINAMATH_CALUDE_multiply_polynomial_l2781_278183

theorem multiply_polynomial (x : ℝ) : 
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_l2781_278183


namespace NUMINAMATH_CALUDE_pentagon_rectangle_intersection_angle_l2781_278171

-- Define the structure of our problem
structure PentagonWithRectangles where
  -- Regular pentagon
  pentagon_angle : ℝ
  -- Right angles from rectangles
  right_angle1 : ℝ
  right_angle2 : ℝ
  -- Reflex angle
  reflex_angle : ℝ
  -- The angle we're solving for
  x : ℝ

-- Define our theorem
theorem pentagon_rectangle_intersection_angle 
  (p : PentagonWithRectangles) 
  (h1 : p.pentagon_angle = 108)
  (h2 : p.right_angle1 = 90)
  (h3 : p.right_angle2 = 90)
  (h4 : p.reflex_angle = 198)
  (h5 : p.pentagon_angle + p.right_angle1 + p.right_angle2 + p.reflex_angle + p.x = 540) :
  p.x = 54 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_intersection_angle_l2781_278171


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l2781_278143

-- Define the function f(x) for part (1)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the function f(x) for part (2) with parameter a
def f_with_a (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part (1)
theorem solution_set_of_inequality (x : ℝ) :
  f x ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 := by sorry

-- Part (2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_with_a a x ≥ 2) → (a = -1 ∨ a = 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l2781_278143


namespace NUMINAMATH_CALUDE_factor_expression_l2781_278147

theorem factor_expression (x : ℝ) : x * (x - 4) + 2 * (x - 4) = (x + 2) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2781_278147


namespace NUMINAMATH_CALUDE_rower_upstream_speed_man_rowing_upstream_speed_l2781_278182

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed -/
theorem rower_upstream_speed (v_still : ℝ) (v_downstream : ℝ) : 
  v_still > 0 → v_downstream > v_still → 
  2 * v_still - v_downstream = v_still - (v_downstream - v_still) := by
  sorry

/-- The specific problem instance -/
theorem man_rowing_upstream_speed : 
  let v_still : ℝ := 33
  let v_downstream : ℝ := 40
  2 * v_still - v_downstream = 26 := by
  sorry

end NUMINAMATH_CALUDE_rower_upstream_speed_man_rowing_upstream_speed_l2781_278182


namespace NUMINAMATH_CALUDE_tan_period_l2781_278103

/-- The period of tan(3x/4) is 4π/3 -/
theorem tan_period (x : ℝ) : 
  (fun x => Real.tan ((3 : ℝ) * x / 4)) = (fun x => Real.tan ((3 : ℝ) * (x + 4 * Real.pi / 3) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_tan_period_l2781_278103


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2781_278114

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2781_278114


namespace NUMINAMATH_CALUDE_area_depends_on_arc_length_l2781_278113

-- Define the unit circle
def unitCircle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point on the unit circle with positive coordinates
def PointOnCircle (p : ℝ × ℝ) : Prop :=
  p ∈ unitCircle ∧ p.1 > 0 ∧ p.2 > 0

-- Define the projection points
def X₁ (x : ℝ × ℝ) : ℝ × ℝ := (x.1, 0)
def X₂ (x : ℝ × ℝ) : ℝ × ℝ := (0, x.2)

-- Define the area of region XYY₁X₁
def areaXYY₁X₁ (x y : ℝ × ℝ) : ℝ := sorry

-- Define the area of region XYY₂X₂
def areaXYY₂X₂ (x y : ℝ × ℝ) : ℝ := sorry

-- Define the angle subtended by arc XY at the center
def arcAngle (x y : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem area_depends_on_arc_length (x y : ℝ × ℝ) 
  (hx : PointOnCircle x) (hy : PointOnCircle y) :
  areaXYY₁X₁ x y + areaXYY₂X₂ x y = arcAngle x y := by
  sorry

end NUMINAMATH_CALUDE_area_depends_on_arc_length_l2781_278113


namespace NUMINAMATH_CALUDE_question_1_l2781_278139

theorem question_1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) :
  a^2 + 3/2 * b - 5 = -2 := by sorry

end NUMINAMATH_CALUDE_question_1_l2781_278139


namespace NUMINAMATH_CALUDE_x_plus_3y_equals_1_l2781_278104

theorem x_plus_3y_equals_1 (x y : ℝ) (h1 : x + y = 19) (h2 : x + 2*y = 10) : x + 3*y = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_3y_equals_1_l2781_278104


namespace NUMINAMATH_CALUDE_equality_theorem_l2781_278140

theorem equality_theorem (a b c d e f : ℝ) 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (∀ k : ℝ, 
    (a + b + c + (d+k) + (e+k) + (f+k) = d + e + f + (a+k) + (b+k) + (c+k)) ∧
    (a^2 + b^2 + c^2 + (d+k)^2 + (e+k)^2 + (f+k)^2 = d^2 + e^2 + f^2 + (a+k)^2 + (b+k)^2 + (c+k)^2) ∧
    (a^3 + b^3 + c^3 + (d+k)^3 + (e+k)^3 + (f+k)^3 = d^3 + e^3 + f^3 + (a+k)^3 + (b+k)^3 + (c+k)^3)) ∧
  (∀ k : ℝ, k ≠ 0 → 
    a^4 + b^4 + c^4 + (d+k)^4 + (e+k)^4 + (f+k)^4 ≠ d^4 + e^4 + f^4 + (a+k)^4 + (b+k)^4 + (c+k)^4) :=
by sorry

end NUMINAMATH_CALUDE_equality_theorem_l2781_278140


namespace NUMINAMATH_CALUDE_simplify_expression_l2781_278176

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (343 : ℝ) ^ (1/3 : ℝ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2781_278176


namespace NUMINAMATH_CALUDE_power_sum_negative_two_l2781_278194

theorem power_sum_negative_two : (-2)^2009 + (-2)^2010 = 2^2009 := by sorry

end NUMINAMATH_CALUDE_power_sum_negative_two_l2781_278194


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2781_278163

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 25 ∧ x - y = 7 → x * y = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2781_278163


namespace NUMINAMATH_CALUDE_candy_eaten_count_l2781_278144

/-- Represents the number of candy pieces collected and eaten by Travis and his brother -/
structure CandyCount where
  initial : ℕ
  remaining : ℕ
  eaten : ℕ

/-- Theorem stating that the difference between initial and remaining candy count equals the eaten count -/
theorem candy_eaten_count (c : CandyCount) (h1 : c.initial = 68) (h2 : c.remaining = 60) :
  c.eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_eaten_count_l2781_278144


namespace NUMINAMATH_CALUDE_triangular_pyramid_volume_l2781_278106

/-- The volume of a triangular pyramid formed by intersecting a right prism with a plane --/
theorem triangular_pyramid_volume 
  (a α β φ : ℝ) 
  (ha : a > 0)
  (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π)
  (hαβ : α + β < π)
  (hφ : 0 < φ ∧ φ < π/2) :
  ∃ V : ℝ, V = (a^3 * Real.sin α^2 * Real.sin β^2 * Real.tan φ) / (6 * Real.sin (α + β)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_volume_l2781_278106


namespace NUMINAMATH_CALUDE_f_monotonic_intervals_fixed_and_extremum_point_condition_no_two_distinct_extrema_fixed_points_l2781_278197

/-- The function f(x) = x³ + ax² + bx + 3 -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 3

/-- A point x₀ is a fixed point of f if f(x₀) = x₀ -/
def is_fixed_point (a b x₀ : ℝ) : Prop := f a b x₀ = x₀

/-- A point x₀ is an extremum point of f if f'(x₀) = 0 -/
def is_extremum_point (a b x₀ : ℝ) : Prop :=
  3*x₀^2 + 2*a*x₀ + b = 0

theorem f_monotonic_intervals (b : ℝ) :
  (b ≥ 0 → StrictMono (f 0 b)) ∧
  (b < 0 → StrictMonoOn (f 0 b) {x | x < -Real.sqrt (-b/3) ∨ x > Real.sqrt (-b/3)}) :=
sorry

theorem fixed_and_extremum_point_condition :
  ∃ x₀ : ℝ, is_fixed_point 0 (-3) x₀ ∧ is_extremum_point 0 (-3) x₀ :=
sorry

theorem no_two_distinct_extrema_fixed_points :
  ¬∃ a b x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    is_fixed_point a b x₁ ∧ is_extremum_point a b x₁ ∧
    is_fixed_point a b x₂ ∧ is_extremum_point a b x₂ :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_intervals_fixed_and_extremum_point_condition_no_two_distinct_extrema_fixed_points_l2781_278197


namespace NUMINAMATH_CALUDE_maria_water_bottles_l2781_278121

theorem maria_water_bottles (initial bottles_bought bottles_remaining : ℕ) 
  (h1 : initial = 14)
  (h2 : bottles_bought = 45)
  (h3 : bottles_remaining = 51) :
  initial - (bottles_remaining - bottles_bought) = 8 := by
  sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l2781_278121


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2781_278157

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x| + |x - 1| < a)) → a ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2781_278157


namespace NUMINAMATH_CALUDE_house_rent_fraction_l2781_278159

theorem house_rent_fraction (salary : ℝ) (house_rent food conveyance left : ℝ) : 
  food = (3/10) * salary →
  conveyance = (1/8) * salary →
  food + conveyance = 3400 →
  left = 1400 →
  house_rent = salary - (food + conveyance + left) →
  house_rent / salary = 2/5 := by
sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l2781_278159


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2781_278136

theorem quadratic_roots_property (f g : ℝ) : 
  (3 * f^2 + 5 * f - 8 = 0) → 
  (3 * g^2 + 5 * g - 8 = 0) → 
  (f - 2) * (g - 2) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2781_278136


namespace NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l2781_278133

/-- Given inverse proportionality of pressure and volume at constant temperature,
    calculate the new pressure when the volume changes. -/
theorem gas_pressure_volume_relationship
  (initial_volume initial_pressure new_volume : ℝ)
  (h_positive : initial_volume > 0 ∧ initial_pressure > 0 ∧ new_volume > 0)
  (h_inverse_prop : ∀ (v p : ℝ), v > 0 → p > 0 → v * p = initial_volume * initial_pressure) :
  let new_pressure := (initial_volume * initial_pressure) / new_volume
  new_pressure = 2 ∧ initial_volume = 2.28 ∧ initial_pressure = 5 ∧ new_volume = 5.7 := by
sorry

end NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l2781_278133


namespace NUMINAMATH_CALUDE_smallest_n_with_2323_divisible_l2781_278108

def count_divisible (n : ℕ) : ℕ :=
  (n / 2) + (n / 23) - 2 * (n / 46)

theorem smallest_n_with_2323_divisible : ∃ (n : ℕ), n > 0 ∧ count_divisible n = 2323 ∧ ∀ m < n, count_divisible m ≠ 2323 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_2323_divisible_l2781_278108


namespace NUMINAMATH_CALUDE_birthday_money_possibility_l2781_278187

theorem birthday_money_possibility (x y : ℕ) : ∃ (a : ℕ), 
  a < 10 ∧ 
  (x * y) % 10 = a ∧ 
  ((x + 1) * (y + 1)) % 10 = a ∧ 
  ((x + 2) * (y + 2)) % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_birthday_money_possibility_l2781_278187


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2781_278125

/-- Given two lines in the xy-plane, this function returns true if they are parallel --/
def are_parallel (m1 : ℝ) (m2 : ℝ) : Prop := m1 = m2

/-- Given a point (x, y) and a line equation y = mx + b, this function returns true if the point lies on the line --/
def point_on_line (x : ℝ) (y : ℝ) (m : ℝ) (b : ℝ) : Prop := y = m * x + b

theorem parallel_line_through_point (x0 y0 : ℝ) : 
  ∃ (m b : ℝ), 
    are_parallel m 2 ∧ 
    point_on_line x0 y0 m b ∧ 
    m = 2 ∧ 
    b = -5 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2781_278125


namespace NUMINAMATH_CALUDE_fifty_ring_squares_l2781_278154

/-- Calculate the number of squares in the nth ring around a 2x1 rectangle --/
def ring_squares (n : ℕ) : ℕ :=
  let outer_width := 2 + 2 * n
  let outer_height := 1 + 2 * n
  let inner_width := 2 + 2 * (n - 1)
  let inner_height := 1 + 2 * (n - 1)
  outer_width * outer_height - inner_width * inner_height

/-- The 50th ring around a 2x1 rectangle contains 402 unit squares --/
theorem fifty_ring_squares : ring_squares 50 = 402 := by
  sorry

#eval ring_squares 50  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_fifty_ring_squares_l2781_278154


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l2781_278117

theorem coefficient_x_squared_in_binomial_expansion :
  let binomial := (X - 1 / X : Polynomial ℚ)^6
  (binomial.coeff 2) = 15 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l2781_278117


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2781_278118

theorem absolute_value_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ ((-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2781_278118


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2781_278172

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = 5) : z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2781_278172


namespace NUMINAMATH_CALUDE_machine_working_time_l2781_278189

/-- The number of shirts made by the machine -/
def total_shirts : ℕ := 12

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 2

/-- The time the machine was working in minutes -/
def working_time : ℕ := total_shirts / shirts_per_minute

theorem machine_working_time : working_time = 6 := by sorry

end NUMINAMATH_CALUDE_machine_working_time_l2781_278189


namespace NUMINAMATH_CALUDE_isabel_bouquets_l2781_278126

/-- Given the total number of flowers, flowers per bouquet, and wilted flowers,
    calculate the maximum number of full bouquets that can be made. -/
def max_bouquets (total : ℕ) (per_bouquet : ℕ) (wilted : ℕ) : ℕ :=
  (total - wilted) / per_bouquet

/-- Theorem stating that given 132 total flowers, 11 flowers per bouquet,
    and 16 wilted flowers, the maximum number of full bouquets is 10. -/
theorem isabel_bouquets :
  max_bouquets 132 11 16 = 10 := by
  sorry

end NUMINAMATH_CALUDE_isabel_bouquets_l2781_278126


namespace NUMINAMATH_CALUDE_solve_for_y_l2781_278185

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2781_278185


namespace NUMINAMATH_CALUDE_melted_spheres_radius_l2781_278145

theorem melted_spheres_radius (n : ℕ) (r : ℝ) (R : ℝ) :
  n = 12 →
  r = 2 →
  (4 / 3) * Real.pi * R^3 = n * ((4 / 3) * Real.pi * r^3) →
  R = (96 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_melted_spheres_radius_l2781_278145


namespace NUMINAMATH_CALUDE_square_of_97_l2781_278173

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l2781_278173


namespace NUMINAMATH_CALUDE_circle_sequence_theorem_circle_sequence_theorem_proof_l2781_278158

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a structure for a circle
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

-- Define a function to check if a circle passes through two points
def passesThrough (c : Circle) (p1 p2 : Point) : Prop :=
  (c.center.x - p1.x)^2 + (c.center.y - p1.y)^2 = c.radius^2 ∧
  (c.center.x - p2.x)^2 + (c.center.y - p2.y)^2 = c.radius^2

-- Define a function to check if two circles are tangent
def areTangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x)^2 + (c1.center.y - c2.center.y)^2 = (c1.radius + c2.radius)^2

-- Define the main theorem
theorem circle_sequence_theorem (t : Triangle) 
  (C1 C2 C3 C4 C5 C6 C7 : Circle) : Prop :=
  passesThrough C1 t.A t.B ∧
  passesThrough C2 t.B t.C ∧ areTangent C1 C2 ∧
  passesThrough C3 t.C t.A ∧ areTangent C2 C3 ∧
  passesThrough C4 t.A t.B ∧ areTangent C3 C4 ∧
  passesThrough C5 t.B t.C ∧ areTangent C4 C5 ∧
  passesThrough C6 t.C t.A ∧ areTangent C5 C6 ∧
  passesThrough C7 t.A t.B ∧ areTangent C6 C7
  →
  C7 = C1

-- The proof would go here
theorem circle_sequence_theorem_proof : ∀ t C1 C2 C3 C4 C5 C6 C7, 
  circle_sequence_theorem t C1 C2 C3 C4 C5 C6 C7 :=
sorry

end NUMINAMATH_CALUDE_circle_sequence_theorem_circle_sequence_theorem_proof_l2781_278158


namespace NUMINAMATH_CALUDE_smallest_n_value_l2781_278129

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 
  (∀ m : ℕ, m > 70 ∧ 70 ∣ (21 * m) → m ≥ N) → N = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2781_278129


namespace NUMINAMATH_CALUDE_product_of_numbers_l2781_278110

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 194) : x * y = -25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2781_278110


namespace NUMINAMATH_CALUDE_price_decrease_approx_16_67_percent_l2781_278151

/-- Calculates the percent decrease between two prices -/
def percentDecrease (oldPrice newPrice : ℚ) : ℚ :=
  (oldPrice - newPrice) / oldPrice * 100

/-- The original price per pack -/
def originalPricePerPack : ℚ := 9 / 6

/-- The promotional price per pack -/
def promotionalPricePerPack : ℚ := 10 / 8

/-- Theorem stating that the percent decrease in price per pack is approximately 16.67% -/
theorem price_decrease_approx_16_67_percent :
  abs (percentDecrease originalPricePerPack promotionalPricePerPack - 100 * (1 / 6)) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_approx_16_67_percent_l2781_278151


namespace NUMINAMATH_CALUDE_imaginary_part_of_4_plus_3i_l2781_278191

theorem imaginary_part_of_4_plus_3i :
  Complex.im (4 + 3*Complex.I) = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_4_plus_3i_l2781_278191


namespace NUMINAMATH_CALUDE_max_area_triangle_l2781_278105

/-- Given a triangle ABC where angle B equals angle C and 7a² + b² + c² = 4√3,
    the maximum possible area of the triangle is √5/5. -/
theorem max_area_triangle (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
    (h2 : 7 * a^2 + b^2 + c^2 = 4 * Real.sqrt 3)
    (h3 : b = c) : 
    ∃ (S : ℝ), S = Real.sqrt 5 / 5 ∧ 
    ∀ (A : ℝ), A = 1/2 * a * b * Real.sqrt (1 - (a / (2 * b))^2) → A ≤ S :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l2781_278105


namespace NUMINAMATH_CALUDE_amoeba_growth_l2781_278135

/-- The population of amoebas after a given number of 10-minute increments -/
def amoeba_population (initial_population : ℕ) (increments : ℕ) : ℕ :=
  initial_population * (3 ^ increments)

/-- Theorem: The amoeba population after 1 hour (6 increments) is 36450 -/
theorem amoeba_growth : amoeba_population 50 6 = 36450 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_growth_l2781_278135


namespace NUMINAMATH_CALUDE_fraction_of_apples_sold_l2781_278134

/-- Proves that the fraction of apples sold is 1/2 given the initial quantities and conditions --/
theorem fraction_of_apples_sold
  (initial_oranges : ℕ)
  (initial_apples : ℕ)
  (orange_fraction_sold : ℚ)
  (total_fruits_left : ℕ)
  (h1 : initial_oranges = 40)
  (h2 : initial_apples = 70)
  (h3 : orange_fraction_sold = 1/4)
  (h4 : total_fruits_left = 65)
  : (initial_apples - (total_fruits_left - (initial_oranges - initial_oranges * orange_fraction_sold))) / initial_apples = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_apples_sold_l2781_278134


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l2781_278170

/-- The volume of a 60% salt solution needed to mix with 1 liter of pure water to create a 20% salt solution -/
def salt_solution_volume : ℝ := 0.5

/-- The concentration of salt in the original solution -/
def original_concentration : ℝ := 0.6

/-- The concentration of salt in the final mixture -/
def final_concentration : ℝ := 0.2

/-- The volume of pure water added -/
def pure_water_volume : ℝ := 1

theorem salt_solution_mixture :
  salt_solution_volume * original_concentration = 
  (pure_water_volume + salt_solution_volume) * final_concentration :=
sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l2781_278170


namespace NUMINAMATH_CALUDE_sigma_odd_implies_perfect_square_l2781_278180

/-- The number of positive divisors of a natural number -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: If the number of positive divisors of a natural number is odd, then the number is a perfect square -/
theorem sigma_odd_implies_perfect_square (N : ℕ) : 
  Odd (sigma N) → ∃ m : ℕ, N = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sigma_odd_implies_perfect_square_l2781_278180


namespace NUMINAMATH_CALUDE_ellipse_parabola_focus_l2781_278120

theorem ellipse_parabola_focus (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) →
  (∃ k : ℝ, ∀ x y : ℝ, x^2 = 8*y → y = k) →
  (n^2 - m^2 = 4) →
  (Real.sqrt (n^2 - m^2) / n = 1/2) →
  m - n = 2 * Real.sqrt 3 - 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_focus_l2781_278120


namespace NUMINAMATH_CALUDE_inspection_team_selection_l2781_278107

theorem inspection_team_selection 
  (total_employees : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (team_size : ℕ) 
  (h1 : total_employees = 15)
  (h2 : men = 10)
  (h3 : women = 5)
  (h4 : team_size = 6)
  (h5 : men + women = total_employees)
  (h6 : 2 * women = men) : 
  Nat.choose men 4 * Nat.choose women 2 = 
  (number_of_ways_to_select_team : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inspection_team_selection_l2781_278107


namespace NUMINAMATH_CALUDE_songs_added_l2781_278188

theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) 
  (h1 : initial = 30) 
  (h2 : deleted = 8) 
  (h3 : final = 32) : 
  final - (initial - deleted) = 10 := by
  sorry

end NUMINAMATH_CALUDE_songs_added_l2781_278188


namespace NUMINAMATH_CALUDE_max_value_abc_l2781_278195

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → x^2 + y^2 + z^2 = 1 → 
  a * b * Real.sqrt 3 + 2 * a * c ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧ 
  a₀ * b₀ * Real.sqrt 3 + 2 * a₀ * c₀ = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2781_278195


namespace NUMINAMATH_CALUDE_journey_distance_l2781_278198

/-- Proves that given a journey of 9 hours, partly on foot at 4 km/hr for 16 km,
    and partly on bicycle at 9 km/hr, the total distance traveled is 61 km. -/
theorem journey_distance (total_time foot_speed bike_speed foot_distance : ℝ) :
  total_time = 9 ∧
  foot_speed = 4 ∧
  bike_speed = 9 ∧
  foot_distance = 16 →
  ∃ (bike_distance : ℝ),
    foot_distance / foot_speed + bike_distance / bike_speed = total_time ∧
    foot_distance + bike_distance = 61 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2781_278198


namespace NUMINAMATH_CALUDE_john_car_profit_l2781_278175

/-- Calculates the profit John made from fixing and racing a car. -/
theorem john_car_profit (initial_cost discount prize_money percentage_kept : ℚ) :
  initial_cost = 20000 →
  discount = 20 / 100 →
  prize_money = 70000 →
  percentage_kept = 90 / 100 →
  (prize_money * percentage_kept) - (initial_cost * (1 - discount)) = 47000 := by
sorry


end NUMINAMATH_CALUDE_john_car_profit_l2781_278175


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l2781_278155

theorem mariela_get_well_cards (hospital_cards : ℕ) (home_cards : ℕ) 
  (h1 : hospital_cards = 403) (h2 : home_cards = 287) : 
  hospital_cards + home_cards = 690 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l2781_278155


namespace NUMINAMATH_CALUDE_denominator_value_l2781_278153

theorem denominator_value (p q x : ℚ) : 
  p / q = 4 / 5 → 
  4 / 7 + (2 * q - p) / x = 1 → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_denominator_value_l2781_278153


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2781_278132

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 2 * x^2 + 6 * x - 15) =
  x^3 + 2 * x^2 + 3 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2781_278132


namespace NUMINAMATH_CALUDE_triangle_properties_l2781_278127

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧
  (t.c - 2 * t.a) * Real.cos t.B + t.b * Real.cos t.C = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 3 ∧ Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2781_278127
