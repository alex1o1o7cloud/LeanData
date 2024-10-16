import Mathlib

namespace NUMINAMATH_CALUDE_base_k_equality_l3260_326035

theorem base_k_equality (k : ℕ) : k^2 + 3*k + 2 = 30 → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_equality_l3260_326035


namespace NUMINAMATH_CALUDE_cereal_box_bowls_l3260_326083

/-- Given a cereal box with the following properties:
  * Each spoonful contains 4 clusters of oats
  * Each bowl has 25 spoonfuls of cereal
  * Each box contains 500 clusters of oats
  Prove that there are 5 bowlfuls of cereal in each box. -/
theorem cereal_box_bowls (clusters_per_spoon : ℕ) (spoons_per_bowl : ℕ) (clusters_per_box : ℕ)
  (h1 : clusters_per_spoon = 4)
  (h2 : spoons_per_bowl = 25)
  (h3 : clusters_per_box = 500) :
  clusters_per_box / (clusters_per_spoon * spoons_per_bowl) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_bowls_l3260_326083


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_AB_l3260_326001

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (-5, 1)

-- Define the line segment AB
def segment_AB : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • A + t • B}

-- Define the perpendicular bisector equation
def perp_bisector_eq (x y : ℝ) : Prop := 3 * x + y + 4 = 0

-- Theorem statement
theorem perpendicular_bisector_of_AB :
  ∀ p : ℝ × ℝ, perp_bisector_eq p.1 p.2 ↔ 
    (dist p A = dist p B ∧ 
     ∀ q : ℝ × ℝ, q ∈ segment_AB → dist p q ≤ dist p A) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_AB_l3260_326001


namespace NUMINAMATH_CALUDE_hash_3_8_l3260_326046

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem to prove
theorem hash_3_8 : hash 3 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_hash_3_8_l3260_326046


namespace NUMINAMATH_CALUDE_base8_digit_product_l3260_326005

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product :
  productOfList (toBase8 8127) = 1764 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_l3260_326005


namespace NUMINAMATH_CALUDE_function_transformation_l3260_326017

/-- Given a function f(x) = 3sin(2x + φ) where φ ∈ (0, π/2), if the graph of f(x) is translated
    left by π/6 units and is symmetric about the y-axis, then f(x) = 3sin(2x + π/6). -/
theorem function_transformation (φ : Real) (h1 : φ > 0) (h2 : φ < π/2) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (2 * x + φ)
  let g : ℝ → ℝ := λ x ↦ f (x + π/6)
  (∀ x, g x = g (-x)) →  -- Symmetry about y-axis
  (∀ x, f x = 3 * Real.sin (2 * x + π/6)) := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3260_326017


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3260_326080

theorem circle_intersection_range (a : ℝ) : 
  (∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    ((p.1 - a)^2 + (p.2 - a)^2 = 4) ∧
    ((q.1 - a)^2 + (q.2 - a)^2 = 4) ∧
    (p.1^2 + p.2^2 = 4) ∧
    (q.1^2 + q.2^2 = 4)) →
  (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3260_326080


namespace NUMINAMATH_CALUDE_sarahs_age_l3260_326013

theorem sarahs_age (ana mark billy sarah : ℝ) 
  (h1 : sarah = 3 * mark - 4)
  (h2 : mark = billy + 4)
  (h3 : billy = ana / 2)
  (h4 : ∃ (years : ℝ), ana + years = 15) :
  sarah = 30.5 := by
sorry

end NUMINAMATH_CALUDE_sarahs_age_l3260_326013


namespace NUMINAMATH_CALUDE_prob_two_private_teams_prob_distribution_ξ_expectation_ξ_l3260_326065

/-- The number of guided tour teams -/
def guided_teams : ℕ := 6

/-- The number of private tour teams -/
def private_teams : ℕ := 3

/-- The total number of teams -/
def total_teams : ℕ := guided_teams + private_teams

/-- The number of draws with replacement -/
def num_draws : ℕ := 4

/-- The random variable representing the number of private teams drawn -/
def ξ : ℕ → ℝ := sorry

/-- The probability of drawing two private tour teams when selecting two numbers at a time -/
theorem prob_two_private_teams : 
  (Nat.choose private_teams 2 : ℚ) / (Nat.choose total_teams 2) = 1 / 12 := by sorry

/-- The probability distribution of ξ -/
theorem prob_distribution_ξ : 
  (ξ 0 = 16 / 81) ∧ 
  (ξ 1 = 32 / 81) ∧ 
  (ξ 2 = 8 / 27) ∧ 
  (ξ 3 = 8 / 81) ∧ 
  (ξ 4 = 1 / 81) := by sorry

/-- The mathematical expectation of ξ -/
theorem expectation_ξ : 
  (0 * ξ 0 + 1 * ξ 1 + 2 * ξ 2 + 3 * ξ 3 + 4 * ξ 4 : ℝ) = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_two_private_teams_prob_distribution_ξ_expectation_ξ_l3260_326065


namespace NUMINAMATH_CALUDE_coat_cost_l3260_326070

def weekly_savings : ℕ := 25
def weeks_of_saving : ℕ := 6
def bill_fraction : ℚ := 1 / 3
def dad_contribution : ℕ := 70

theorem coat_cost : 
  weekly_savings * weeks_of_saving - 
  (weekly_savings * weeks_of_saving : ℚ) * bill_fraction +
  dad_contribution = 170 := by sorry

end NUMINAMATH_CALUDE_coat_cost_l3260_326070


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3260_326053

/-- Given positive real numbers m and n, and perpendicular vectors (m, 1) and (1, n-1),
    the minimum value of 1/m + 2/n is 3 + 2√2. -/
theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
    (h_perp : m * 1 + 1 * (n - 1) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * 1 + 1 * (y - 1) = 0 → 1/x + 2/y ≥ 1/m + 2/n) →
  1/m + 2/n = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3260_326053


namespace NUMINAMATH_CALUDE_sqrt_three_product_l3260_326008

theorem sqrt_three_product : 5 * Real.sqrt 3 * (2 * Real.sqrt 3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_product_l3260_326008


namespace NUMINAMATH_CALUDE_same_distance_different_time_l3260_326088

/-- Calculates the required average speed for a rider to cover the same distance as another rider in a different time. -/
theorem same_distance_different_time 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 5) : 
  (joann_speed * joann_time) / fran_time = 12 := by
  sorry

#check same_distance_different_time

end NUMINAMATH_CALUDE_same_distance_different_time_l3260_326088


namespace NUMINAMATH_CALUDE_abs_y_bound_l3260_326049

theorem abs_y_bound (x y : ℝ) (h1 : |x + y| < 1/3) (h2 : |2*x - y| < 1/6) : |y| < 5/18 := by
  sorry

end NUMINAMATH_CALUDE_abs_y_bound_l3260_326049


namespace NUMINAMATH_CALUDE_toy_store_restocking_l3260_326036

theorem toy_store_restocking (initial_games : ℕ) (sold_games : ℕ) (final_games : ℕ)
  (h1 : initial_games = 95)
  (h2 : sold_games = 68)
  (h3 : final_games = 74) :
  final_games - (initial_games - sold_games) = 47 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_restocking_l3260_326036


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3260_326072

-- Define the sets A and B
def A : Set ℝ := {x | 3*x^2 - 14*x + 16 ≤ 0}
def B : Set ℝ := {x | (3*x - 7) / x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 7/3 < x ∧ x ≤ 8/3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3260_326072


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3260_326041

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations for parallel and perpendicular
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3260_326041


namespace NUMINAMATH_CALUDE_A_eq_B_l3260_326042

/-- A coloring of points in the plane -/
structure Coloring (n : ℕ+) where
  color : ℕ → ℕ → Bool
  valid : ∀ x y x' y', x' ≤ x → y' ≤ y → x + y ≤ n → color x y = false → color x' y' = false

/-- The number of ways to choose n blue points with distinct x-coordinates -/
def A (n : ℕ+) (c : Coloring n) : ℕ := sorry

/-- The number of ways to choose n blue points with distinct y-coordinates -/
def B (n : ℕ+) (c : Coloring n) : ℕ := sorry

/-- The main theorem: A = B for any valid coloring -/
theorem A_eq_B (n : ℕ+) (c : Coloring n) : A n c = B n c := by sorry

end NUMINAMATH_CALUDE_A_eq_B_l3260_326042


namespace NUMINAMATH_CALUDE_intersection_M_N_l3260_326006

def M : Set ℝ := {x | (x + 1) * (x - 3) < 0}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3260_326006


namespace NUMINAMATH_CALUDE_n_range_l3260_326024

/-- The function f(x) with parameters m and n -/
def f (m n x : ℝ) : ℝ := m * x^2 - (5 * m + n) * x + n

/-- Theorem stating the range of n given the conditions -/
theorem n_range :
  ∀ n : ℝ,
  (∃ m : ℝ, -2 < m ∧ m < -1 ∧
    ∃ x : ℝ, 3 < x ∧ x < 5 ∧ f m n x = 0) →
  0 < n ∧ n ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_n_range_l3260_326024


namespace NUMINAMATH_CALUDE_bisected_right_triangle_angles_l3260_326051

/-- A right triangle with a bisected right angle -/
structure BisectedRightTriangle where
  /-- The measure of the first acute angle -/
  α : Real
  /-- The measure of the second acute angle -/
  β : Real
  /-- The right angle is 90 degrees -/
  right_angle : α + β = 90
  /-- The angle bisector divides the right angle into two 45-degree angles -/
  bisector_angle : Real
  bisector_property : bisector_angle = 45
  /-- The ratio of angles formed by the angle bisector and the hypotenuse is 7:11 -/
  hypotenuse_angles : Real × Real
  hypotenuse_angles_ratio : hypotenuse_angles.1 / hypotenuse_angles.2 = 7 / 11
  hypotenuse_angles_sum : hypotenuse_angles.1 + hypotenuse_angles.2 = 180 - bisector_angle

/-- The theorem stating the angles of the triangle given the conditions -/
theorem bisected_right_triangle_angles (t : BisectedRightTriangle) : 
  t.α = 65 ∧ t.β = 25 ∧ t.α + t.β = 90 := by
  sorry

end NUMINAMATH_CALUDE_bisected_right_triangle_angles_l3260_326051


namespace NUMINAMATH_CALUDE_mini_quiz_true_false_count_l3260_326020

/-- The number of true-false questions in the mini-quiz. -/
def n : ℕ := 3

/-- The number of multiple-choice questions. -/
def m : ℕ := 2

/-- The number of answer choices for each multiple-choice question. -/
def k : ℕ := 4

/-- The total number of ways to write the answer key. -/
def total_ways : ℕ := 96

theorem mini_quiz_true_false_count :
  (2^n - 2) * k^m = total_ways ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_mini_quiz_true_false_count_l3260_326020


namespace NUMINAMATH_CALUDE_zoo_layout_l3260_326043

theorem zoo_layout (tiger_enclosures zebra_enclosures giraffe_enclosures : ℕ)
  (tigers_per_enclosure : ℕ) (zebras_per_enclosure giraffes_per_enclosure : ℕ)
  (total_animals : ℕ) :
  tiger_enclosures = 4 →
  zebra_enclosures = 2 * tiger_enclosures →
  giraffe_enclosures = 3 * zebra_enclosures →
  zebras_per_enclosure = 10 →
  giraffes_per_enclosure = 2 →
  total_animals = 144 →
  total_animals = tiger_enclosures * tigers_per_enclosure +
                  zebra_enclosures * zebras_per_enclosure +
                  giraffe_enclosures * giraffes_per_enclosure →
  tigers_per_enclosure = 4 := by
sorry

end NUMINAMATH_CALUDE_zoo_layout_l3260_326043


namespace NUMINAMATH_CALUDE_sams_dimes_proof_l3260_326031

/-- The number of dimes Sam's dad gave him -/
def dimes_from_dad (initial_dimes final_dimes : ℕ) : ℕ :=
  final_dimes - initial_dimes

theorem sams_dimes_proof (initial_dimes final_dimes : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16) : 
  dimes_from_dad initial_dimes final_dimes = 7 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_proof_l3260_326031


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3260_326045

theorem quadratic_equation_solution :
  ∀ x : ℝ, (x - 6) * (x + 2) = 0 ↔ x = 6 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3260_326045


namespace NUMINAMATH_CALUDE_roots_transformation_l3260_326033

theorem roots_transformation (p q r : ℂ) : 
  (p^3 - 4*p^2 + 5*p + 2 = 0) ∧ 
  (q^3 - 4*q^2 + 5*q + 2 = 0) ∧ 
  (r^3 - 4*r^2 + 5*r + 2 = 0) →
  ((3*p)^3 - 12*(3*p)^2 + 45*(3*p) + 54 = 0) ∧
  ((3*q)^3 - 12*(3*q)^2 + 45*(3*q) + 54 = 0) ∧
  ((3*r)^3 - 12*(3*r)^2 + 45*(3*r) + 54 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l3260_326033


namespace NUMINAMATH_CALUDE_circle_equation_diameter_circle_equation_points_line_l3260_326015

-- Define points and line
def P₁ : ℝ × ℝ := (4, 9)
def P₂ : ℝ × ℝ := (6, 3)
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)
def l (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem for the first circle
theorem circle_equation_diameter (x y : ℝ) : 
  (x - 5)^2 + (y - 6)^2 = 10 ↔ 
  ∃ (t : ℝ), (x, y) = (1 - t) • P₁ + t • P₂ ∧ 0 ≤ t ∧ t ≤ 1 :=
sorry

-- Theorem for the second circle
theorem circle_equation_points_line (x y : ℝ) :
  x^2 + y^2 + 2*x + 4*y - 5 = 0 ↔
  (∃ (cx cy : ℝ), (x - cx)^2 + (y - cy)^2 = ((x - 2)^2 + (y + 3)^2) ∧
                  (x - (-2))^2 + (y - (-5))^2 = ((x - 2)^2 + (y + 3)^2) ∧
                  l cx cy) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_diameter_circle_equation_points_line_l3260_326015


namespace NUMINAMATH_CALUDE_smallest_multiple_in_sequence_l3260_326012

theorem smallest_multiple_in_sequence (a : ℕ) : 
  (∀ i ∈ Finset.range 16, ∃ k : ℕ, a + 3 * i = 3 * k) →
  (6 * a + 3 * (0 + 1 + 2 + 3 + 4 + 5) = 5 * a + 3 * (11 + 12 + 13 + 14 + 15)) →
  a = 150 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_in_sequence_l3260_326012


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3260_326028

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3260_326028


namespace NUMINAMATH_CALUDE_remainder_problem_l3260_326076

theorem remainder_problem (n : ℕ) : 
  n % 12 = 22 → 
  ((n % 34) + (n % 12)) % 12 = 10 → 
  n % 34 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3260_326076


namespace NUMINAMATH_CALUDE_interest_earned_proof_l3260_326093

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem interest_earned_proof (principal : ℝ) :
  compound_interest principal 0.08 2 = 19828.80 →
  19828.80 - principal = 2828.80 := by
  sorry

#check interest_earned_proof

end NUMINAMATH_CALUDE_interest_earned_proof_l3260_326093


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3260_326023

theorem inequality_system_solution (x : ℝ) :
  (x - 3 * (x - 2) ≥ 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3260_326023


namespace NUMINAMATH_CALUDE_area_closed_region_l3260_326094

/-- The area of the closed region formed by f(x) and g(x) over one period -/
theorem area_closed_region (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (a * x) + Real.cos (a * x)
  let g : ℝ → ℝ := λ x ↦ Real.sqrt (a^2 + 1)
  let period : ℝ := 2 * Real.pi / a
  ∃ (area : ℝ), area = period * Real.sqrt (a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_area_closed_region_l3260_326094


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l3260_326014

theorem geometric_progression_solution : 
  ∃! x : ℝ, (30 + x)^2 = (15 + x) * (60 + x) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l3260_326014


namespace NUMINAMATH_CALUDE_triangle_area_l3260_326011

/-- Given a triangle with one side of length 2, a median to this side of length 1,
    and the sum of the other two sides equal to 1 + √3,
    prove that the area of the triangle is √3/2. -/
theorem triangle_area (a b c : ℝ) (h1 : c = 2) (h2 : a + b = 1 + Real.sqrt 3)
  (h3 : ∃ (m : ℝ), m = 1 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) :
  (a * b) / 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3260_326011


namespace NUMINAMATH_CALUDE_a_neq_1_necessary_not_sufficient_for_a_squared_neq_1_l3260_326007

theorem a_neq_1_necessary_not_sufficient_for_a_squared_neq_1 :
  (∀ a : ℝ, a^2 ≠ 1 → a ≠ 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ a^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_neq_1_necessary_not_sufficient_for_a_squared_neq_1_l3260_326007


namespace NUMINAMATH_CALUDE_min_clicks_to_one_color_l3260_326073

/-- Represents a chessboard -/
def Chessboard := Fin 98 → Fin 98 → Bool

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Fin 98 × Fin 98
  bottom_right : Fin 98 × Fin 98

/-- Applies a click to a rectangle on the chessboard -/
def applyClick (board : Chessboard) (rect : Rectangle) : Chessboard :=
  sorry

/-- Checks if the entire board is one color -/
def isOneColor (board : Chessboard) : Bool :=
  sorry

/-- Initial chessboard with alternating colors -/
def initialBoard : Chessboard :=
  sorry

/-- Theorem: The minimum number of clicks to make the chessboard one color is 98 -/
theorem min_clicks_to_one_color :
  ∀ (clicks : List Rectangle),
    isOneColor (clicks.foldl applyClick initialBoard) →
    clicks.length ≥ 98 :=
  sorry

end NUMINAMATH_CALUDE_min_clicks_to_one_color_l3260_326073


namespace NUMINAMATH_CALUDE_rectangle_area_l3260_326002

/-- Given a rectangle with length twice its width and width of 5 inches, prove its area is 50 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 5 → length = 2 * width → width * length = 50 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3260_326002


namespace NUMINAMATH_CALUDE_friend_walking_rates_l3260_326092

theorem friend_walking_rates (trail_length : ℝ) (p_distance : ℝ) 
  (hp : trail_length = 33)
  (hpd : p_distance = 18) :
  let q_distance := trail_length - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_friend_walking_rates_l3260_326092


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3260_326082

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that given the collinearity of the points (4, -10), (-b + 4, 6), and (3b + 6, 4),
    the value of b must be -16/31 -/
theorem collinear_points_b_value :
  ∀ b : ℚ, collinear 4 (-10) (-b + 4) 6 (3*b + 6) 4 → b = -16/31 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3260_326082


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3260_326099

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| - 2*x + y = 1) 
  (h2 : x - |y| + y = 8) : 
  x + y = 17 ∨ x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3260_326099


namespace NUMINAMATH_CALUDE_M_subset_N_l3260_326075

-- Define the sets M and N
def M : Set ℝ := {α | ∃ k : ℤ, α = k * 90 ∨ α = k * 180 + 45}
def N : Set ℝ := {α | ∃ k : ℤ, α = k * 45}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3260_326075


namespace NUMINAMATH_CALUDE_tank_egg_difference_l3260_326085

/-- The number of eggs Tank gathered in the first round -/
def tank_first : ℕ := 160

/-- The number of eggs Tank gathered in the second round -/
def tank_second : ℕ := 30

/-- The number of eggs Emma gathered in the first round -/
def emma_first : ℕ := tank_first - 10

/-- The number of eggs Emma gathered in the second round -/
def emma_second : ℕ := 60

/-- The total number of eggs collected by all 8 people -/
def total_eggs : ℕ := 400

theorem tank_egg_difference :
  tank_first - tank_second = 130 ∧
  emma_second = 2 * tank_second ∧
  tank_first > tank_second ∧
  tank_first = emma_first + 10 ∧
  tank_first + emma_first + tank_second + emma_second = total_eggs :=
by sorry

end NUMINAMATH_CALUDE_tank_egg_difference_l3260_326085


namespace NUMINAMATH_CALUDE_james_argument_friends_l3260_326061

/-- Calculates the number of friends James got into an argument with -/
def friends_in_argument (initial_friends new_friends final_friends : ℕ) : ℕ :=
  initial_friends + new_friends - final_friends

theorem james_argument_friends :
  friends_in_argument 20 1 19 = 1 := by
  sorry

end NUMINAMATH_CALUDE_james_argument_friends_l3260_326061


namespace NUMINAMATH_CALUDE_triangle_prime_count_l3260_326029

def is_prime (n : ℕ) : Prop := sorry

def count_primes (a b : ℕ) : ℕ := sorry

def triangle_sides_valid (n : ℕ) : Prop :=
  let side1 := Real.log 16 / Real.log 8
  let side2 := Real.log 128 / Real.log 8
  let side3 := Real.log n / Real.log 8
  side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1

theorem triangle_prime_count :
  ∀ n : ℕ, 
    n > 0 → 
    is_prime n → 
    triangle_sides_valid n →
    ∃ (count : ℕ), count = count_primes 9 4095 := by
  sorry

end NUMINAMATH_CALUDE_triangle_prime_count_l3260_326029


namespace NUMINAMATH_CALUDE_oplus_k_oplus_k_l3260_326059

-- Define the ⊕ operation
def oplus (x y : ℝ) : ℝ := x^3 - 2*y + x

-- Theorem statement
theorem oplus_k_oplus_k (k : ℝ) : oplus k (oplus k k) = -k^3 + 3*k := by
  sorry

end NUMINAMATH_CALUDE_oplus_k_oplus_k_l3260_326059


namespace NUMINAMATH_CALUDE_baxter_spent_105_l3260_326027

/-- The cost of peanuts per pound -/
def cost_per_pound : ℕ := 3

/-- The minimum purchase requirement in pounds -/
def minimum_purchase : ℕ := 15

/-- The amount Baxter purchased over the minimum, in pounds -/
def over_minimum : ℕ := 20

/-- Calculates the total amount Baxter spent on peanuts -/
def baxter_spent : ℕ := cost_per_pound * (minimum_purchase + over_minimum)

/-- Proves that Baxter spent $105 on peanuts -/
theorem baxter_spent_105 : baxter_spent = 105 := by
  sorry

end NUMINAMATH_CALUDE_baxter_spent_105_l3260_326027


namespace NUMINAMATH_CALUDE_carpet_shampooing_time_l3260_326074

theorem carpet_shampooing_time 
  (jason_rate : ℝ) 
  (tom_rate : ℝ) 
  (h1 : jason_rate = 1 / 3) 
  (h2 : tom_rate = 1 / 6) : 
  1 / (jason_rate + tom_rate) = 2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shampooing_time_l3260_326074


namespace NUMINAMATH_CALUDE_kindergarten_count_l3260_326032

/-- Given the ratio of boys to girls and girls to teachers in a kindergarten,
    along with the number of boys, prove the total number of students and teachers. -/
theorem kindergarten_count (boys girls teachers : ℕ) : 
  (boys : ℚ) / girls = 3 / 4 →
  (girls : ℚ) / teachers = 5 / 2 →
  boys = 18 →
  boys + girls + teachers = 53 := by
sorry

end NUMINAMATH_CALUDE_kindergarten_count_l3260_326032


namespace NUMINAMATH_CALUDE_circular_film_radius_l3260_326081

/-- The radius of a circular film formed by a liquid --/
theorem circular_film_radius 
  (volume : ℝ) 
  (thickness : ℝ) 
  (radius : ℝ) 
  (h1 : volume = 320) 
  (h2 : thickness = 0.05) 
  (h3 : π * radius^2 * thickness = volume) : 
  radius = Real.sqrt (6400 / π) := by
sorry

end NUMINAMATH_CALUDE_circular_film_radius_l3260_326081


namespace NUMINAMATH_CALUDE_crayon_difference_l3260_326037

theorem crayon_difference (willy_crayons lucy_crayons : ℕ) 
  (h1 : willy_crayons = 1400) 
  (h2 : lucy_crayons = 290) : 
  willy_crayons - lucy_crayons = 1110 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l3260_326037


namespace NUMINAMATH_CALUDE_min_value_fraction_l3260_326067

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/a + 2/b ≤ 1/x + 2/y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 2/y = (3 + 2 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3260_326067


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3260_326019

/-- Given a quadratic equation (m-2)x^2 + 3x - m^2 - m + 6 = 0 where one root is 0,
    prove that m = -3 is the only valid solution. -/
theorem quadratic_root_zero (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 3 * x - m^2 - m + 6 = 0 ↔ x = 0 ∨ x = (m^2 + m - 6) / (2 * m - 4)) →
  m - 2 ≠ 0 →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3260_326019


namespace NUMINAMATH_CALUDE_red_card_events_mutually_exclusive_not_opposite_l3260_326097

-- Define the set of cards
inductive Card : Type
  | Black : Card
  | Red : Card
  | White : Card

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsRed (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem statement
theorem red_card_events_mutually_exclusive_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsRed d)) ∧
  -- The events are not opposite (i.e., it's possible for neither to occur)
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsRed d) :=
sorry

end NUMINAMATH_CALUDE_red_card_events_mutually_exclusive_not_opposite_l3260_326097


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3260_326066

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = 2x² -/
def f (x : ℝ) : ℝ := 2 * x^2

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3260_326066


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l3260_326086

theorem intersection_implies_m_value (m : ℝ) : 
  let A : Set ℝ := {1, m-2}
  let B : Set ℝ := {2, 3}
  A ∩ B = {2} → m = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l3260_326086


namespace NUMINAMATH_CALUDE_p_current_age_is_fifteen_l3260_326009

/-- Given the age ratios of two people P and Q at different times, 
    prove that P's current age is 15 years. -/
theorem p_current_age_is_fifteen :
  ∀ (p q : ℕ),
  (p - 3) / (q - 3) = 4 / 3 →
  (p + 6) / (q + 6) = 7 / 6 →
  p = 15 := by
  sorry

end NUMINAMATH_CALUDE_p_current_age_is_fifteen_l3260_326009


namespace NUMINAMATH_CALUDE_can_lids_per_box_l3260_326062

theorem can_lids_per_box 
  (initial_lids : ℕ) 
  (num_boxes : ℕ) 
  (total_lids : ℕ) 
  (h1 : initial_lids = 14) 
  (h2 : num_boxes = 3) 
  (h3 : total_lids = 53) : 
  (total_lids - initial_lids) / num_boxes = 13 := by
sorry

end NUMINAMATH_CALUDE_can_lids_per_box_l3260_326062


namespace NUMINAMATH_CALUDE_fraction_transformation_l3260_326048

theorem fraction_transformation (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 8 : ℚ) / (d + 8) = (1 : ℚ) / 3 →
  d = 25 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3260_326048


namespace NUMINAMATH_CALUDE_scooter_initial_value_l3260_326095

/-- Proves that the initial value of a scooter is 40000 given the depreciation rate and final value after 3 years -/
theorem scooter_initial_value (depreciation_rate : ℚ) (final_value : ℚ) : 
  depreciation_rate = 3/4 →
  final_value = 16875 →
  (depreciation_rate^3 * 40000 : ℚ) = final_value := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l3260_326095


namespace NUMINAMATH_CALUDE_expression_evaluation_l3260_326039

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3260_326039


namespace NUMINAMATH_CALUDE_absolute_value_sin_sqrt_calculation_l3260_326063

theorem absolute_value_sin_sqrt_calculation :
  |(-3 : ℝ)| + 2 * Real.sin (30 * π / 180) - Real.sqrt 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sin_sqrt_calculation_l3260_326063


namespace NUMINAMATH_CALUDE_shortest_tree_height_l3260_326090

/-- Given three trees with specified height relationships, prove the height of the shortest tree -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 →
  middle = 2/3 * tallest →
  shortest = 1/2 * middle →
  shortest = 50 := by
sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l3260_326090


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3260_326052

/-- A hyperbola is defined by its standard equation and properties -/
structure Hyperbola where
  /-- The standard equation of the hyperbola: y²/a² - x²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The hyperbola passes through a given point -/
  passes_through : ℝ × ℝ → Prop
  /-- The asymptotic equations of the hyperbola -/
  asymptotic_equations : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop)

/-- Theorem: Given a hyperbola that passes through (√3, 4) with asymptotic equations 2x ± y = 0,
    its standard equation is y²/4 - x² = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  h.passes_through (Real.sqrt 3, 4) ∧
  h.asymptotic_equations = ((fun x y => 2*x = y), (fun x y => 2*x = -y)) →
  h.equation = fun x y => y^2/4 - x^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3260_326052


namespace NUMINAMATH_CALUDE_sine_angle_plus_pi_half_l3260_326087

theorem sine_angle_plus_pi_half (α : Real) : 
  (∃ r : Real, r > 0 ∧ -1 = r * Real.cos α ∧ Real.sqrt 3 = r * Real.sin α) →
  Real.sin (α + π/2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sine_angle_plus_pi_half_l3260_326087


namespace NUMINAMATH_CALUDE_arctan_tan_sum_equals_angle_l3260_326038

theorem arctan_tan_sum_equals_angle (θ : Real) : 
  θ ≥ 0 ∧ θ ≤ π / 2 → Real.arctan (Real.tan θ + 3 * Real.tan (π / 12)) = θ := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_sum_equals_angle_l3260_326038


namespace NUMINAMATH_CALUDE_problem_solution_l3260_326016

/-- Given that 4x^5 + 3x^3 - 2x + 1 + g(x) = 7x^3 - 5x^2 + 4x - 3,
    prove that g(x) = -4x^5 + 4x^3 - 5x^2 + 6x - 4 -/
theorem problem_solution (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 4*x^5 + 3*x^3 - 2*x + 1 + g x = 7*x^3 - 5*x^2 + 4*x - 3) : 
  g x = -4*x^5 + 4*x^3 - 5*x^2 + 6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3260_326016


namespace NUMINAMATH_CALUDE_f_simplification_f_value_at_specific_angle_l3260_326060

noncomputable def f (θ : ℝ) : ℝ :=
  (Real.sin (θ + 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 - θ) * Real.cos (θ + 3 * Real.pi)) /
  (Real.cos (-Real.pi / 2 - θ) * Real.sin (-3 * Real.pi / 2 - θ))

theorem f_simplification (θ : ℝ) : f θ = -Real.cos θ := by sorry

theorem f_value_at_specific_angle (θ : ℝ) (h : Real.sin (θ - Real.pi / 6) = 3 / 5) :
  f (θ + Real.pi / 3) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_at_specific_angle_l3260_326060


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_l3260_326021

theorem inscribed_circle_triangle (r : ℝ) (a b c : ℝ) :
  r = 3 →
  a + b = 7 →
  a = 3 →
  b = 4 →
  c^2 = a^2 + b^2 →
  (a + r)^2 + (b + r)^2 = c^2 →
  (a, b, c) = (3, 4, 5) ∨ (a, b, c) = (4, 3, 5) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_triangle_l3260_326021


namespace NUMINAMATH_CALUDE_maritime_silk_road_analysis_l3260_326058

/-- Represents the Maritime Silk Road -/
structure MaritimeSilkRoad where
  economic_exchange : Bool
  cultural_exchange : Bool

/-- Represents the discussion method used -/
structure DiscussionMethod where
  theory_of_two_points : Bool
  theory_of_emphasis : Bool

/-- Represents the viewpoints in the discussion -/
inductive Viewpoint
  | economy_first
  | culture_first

/-- Theorem stating the analysis of the Maritime Silk Road discussion -/
theorem maritime_silk_road_analysis 
  (msr : MaritimeSilkRoad) 
  (method : DiscussionMethod) 
  (viewpoints : List Viewpoint) :
  msr.economic_exchange = true →
  msr.cultural_exchange = true →
  method.theory_of_two_points = true →
  method.theory_of_emphasis = true →
  viewpoints.length > 1 →
  (∃ (analysis : Bool), 
    analysis = true ↔ 
      (∃ (social_existence_consciousness : Bool) (culture_economy : Bool),
        social_existence_consciousness = true ∧ 
        culture_economy = true)) :=
by sorry

end NUMINAMATH_CALUDE_maritime_silk_road_analysis_l3260_326058


namespace NUMINAMATH_CALUDE_fixed_fee_is_9_39_l3260_326022

/-- Represents a cloud storage service billing system -/
structure CloudStorageBilling where
  fixed_fee : ℝ
  feb_usage_fee : ℝ
  feb_total : ℝ
  mar_total : ℝ

/-- The cloud storage billing satisfies the given conditions -/
def satisfies_conditions (bill : CloudStorageBilling) : Prop :=
  bill.feb_total = bill.fixed_fee + bill.feb_usage_fee ∧
  bill.mar_total = bill.fixed_fee + 3 * bill.feb_usage_fee ∧
  bill.feb_total = 15.80 ∧
  bill.mar_total = 28.62

/-- The fixed monthly fee is 9.39 given the conditions -/
theorem fixed_fee_is_9_39 (bill : CloudStorageBilling) 
  (h : satisfies_conditions bill) : bill.fixed_fee = 9.39 := by
  sorry

end NUMINAMATH_CALUDE_fixed_fee_is_9_39_l3260_326022


namespace NUMINAMATH_CALUDE_parallel_lines_problem_l3260_326055

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def num_parallelograms (n : ℕ) (m : ℕ) : ℕ := n.choose 2 * m.choose 2

/-- The theorem statement -/
theorem parallel_lines_problem (n : ℕ) :
  num_parallelograms n 8 = 420 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_problem_l3260_326055


namespace NUMINAMATH_CALUDE_angle_function_simplification_l3260_326044

theorem angle_function_simplification (α : Real) 
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.cos (α + π / 2) = -1 / 5) :
  (Real.tan (α - π) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.cos (-α - π) * Real.tan (π + α)) = -2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_function_simplification_l3260_326044


namespace NUMINAMATH_CALUDE_system_solution_l3260_326040

theorem system_solution (a b : ℝ) : 
  (2 * 5 + b = a) ∧ (5 - 2 * b = 3) → a = 11 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l3260_326040


namespace NUMINAMATH_CALUDE_difference_max_min_both_l3260_326050

/-- The total number of students at the university -/
def total_students : ℕ := 2500

/-- The number of students studying German -/
def german_students : ℕ → Prop :=
  λ g => 1750 ≤ g ∧ g ≤ 1875

/-- The number of students studying Russian -/
def russian_students : ℕ → Prop :=
  λ r => 625 ≤ r ∧ r ≤ 875

/-- The number of students studying both German and Russian -/
def both_languages (g r b : ℕ) : Prop :=
  g + r - b = total_students

/-- The minimum number of students studying both languages -/
def min_both (m : ℕ) : Prop :=
  ∃ g r, german_students g ∧ russian_students r ∧ both_languages g r m ∧
  ∀ b, (∃ g' r', german_students g' ∧ russian_students r' ∧ both_languages g' r' b) → m ≤ b

/-- The maximum number of students studying both languages -/
def max_both (M : ℕ) : Prop :=
  ∃ g r, german_students g ∧ russian_students r ∧ both_languages g r M ∧
  ∀ b, (∃ g' r', german_students g' ∧ russian_students r' ∧ both_languages g' r' b) → b ≤ M

theorem difference_max_min_both :
  ∃ m M, min_both m ∧ max_both M ∧ M - m = 375 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_l3260_326050


namespace NUMINAMATH_CALUDE_sum_of_digits_square_22222_l3260_326054

/-- The sum of the digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The square of 22222 -/
def square_22222 : ℕ := 22222 * 22222

theorem sum_of_digits_square_22222 : sum_of_digits square_22222 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_22222_l3260_326054


namespace NUMINAMATH_CALUDE_ratio_to_two_l3260_326069

theorem ratio_to_two (x : ℝ) : (x / 2 = 150 / 1) → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_two_l3260_326069


namespace NUMINAMATH_CALUDE_parabola_slope_relation_l3260_326089

/-- Parabola struct representing y^2 = 2px --/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Theorem statement --/
theorem parabola_slope_relation 
  (C : Parabola) 
  (A : Point)
  (B : Point)
  (P M N : Point)
  (h_A : A.y^2 = 2 * C.p * A.x)
  (h_A_x : A.x = 1)
  (h_B : B.x = -C.p/2 ∧ B.y = 0)
  (h_AB : (A.x - B.x)^2 + (A.y - B.y)^2 = 8)
  (h_P : P.y^2 = 2 * C.p * P.x ∧ P.y = 2)
  (h_M : M.y^2 = 2 * C.p * M.x)
  (h_N : N.y^2 = 2 * C.p * N.x)
  (k₁ k₂ k₃ : ℝ)
  (h_k₁ : k₁ ≠ 0)
  (h_k₂ : k₂ ≠ 0)
  (h_k₃ : k₃ ≠ 0)
  (h_PM : (M.y - P.y) = k₁ * (M.x - P.x))
  (h_PN : (N.y - P.y) = k₂ * (N.x - P.x))
  (h_MN : (N.y - M.y) = k₃ * (N.x - M.x)) :
  1/k₁ + 1/k₂ - 1/k₃ = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_slope_relation_l3260_326089


namespace NUMINAMATH_CALUDE_two_n_squares_implies_n_squares_l3260_326004

theorem two_n_squares_implies_n_squares (n : ℕ) 
  (h : ∃ (k m : ℤ), 2 * n = k^2 + m^2) : 
  ∃ (a b : ℚ), n = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_two_n_squares_implies_n_squares_l3260_326004


namespace NUMINAMATH_CALUDE_range_of_a_l3260_326047

theorem range_of_a (a : ℝ) : 
  (∀ x, 0 < x ∧ x < 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x, ¬(0 < x ∧ x < 1) ∧ (x - a) * (x - (a + 2)) ≤ 0) → 
  -1 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3260_326047


namespace NUMINAMATH_CALUDE_students_not_in_biology_l3260_326068

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) : 
  total_students = 880 → 
  biology_percentage = 30 / 100 →
  (total_students : ℚ) * (1 - biology_percentage) = 616 :=
by sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l3260_326068


namespace NUMINAMATH_CALUDE_equation_holds_when_b_plus_c_is_ten_l3260_326064

theorem equation_holds_when_b_plus_c_is_ten (a b c : ℕ) : 
  a > 0 → a < 10 → b > 0 → b < 10 → c > 0 → c < 10 → b + c = 10 → 
  (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a^2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_when_b_plus_c_is_ten_l3260_326064


namespace NUMINAMATH_CALUDE_roommate_difference_l3260_326026

theorem roommate_difference (bob_roommates john_roommates : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) : 
  john_roommates - 2 * bob_roommates = 5 := by
  sorry

end NUMINAMATH_CALUDE_roommate_difference_l3260_326026


namespace NUMINAMATH_CALUDE_cube_split_l3260_326077

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

theorem cube_split (m : ℕ) (h1 : m > 1) :
  (∃ k : ℕ, k > 0 ∧ k < m ∧ nth_odd (triangular_number k + 1) = 59) →
  m = 8 := by sorry

end NUMINAMATH_CALUDE_cube_split_l3260_326077


namespace NUMINAMATH_CALUDE_paint_needed_for_smaller_statues_l3260_326096

/-- The height of the original statue in feet -/
def original_height : ℝ := 12

/-- The height of each smaller statue in feet -/
def smaller_height : ℝ := 2

/-- The number of smaller statues -/
def num_statues : ℕ := 720

/-- The amount of paint in pints needed for the original statue -/
def paint_for_original : ℝ := 1

/-- The amount of paint needed for all smaller statues -/
def paint_for_all_statues : ℝ := 20

theorem paint_needed_for_smaller_statues :
  (num_statues : ℝ) * paint_for_original * (smaller_height / original_height) ^ 2 = paint_for_all_statues :=
sorry

end NUMINAMATH_CALUDE_paint_needed_for_smaller_statues_l3260_326096


namespace NUMINAMATH_CALUDE_orlando_weight_gain_l3260_326003

theorem orlando_weight_gain (x : ℝ) : 
  x + (2 * x + 2) + ((1 / 2) * (2 * x + 2) - 3) = 20 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_orlando_weight_gain_l3260_326003


namespace NUMINAMATH_CALUDE_smallest_c_value_l3260_326056

/-- Given a function y = a * cos(b * x + c), where a, b, and c are positive constants,
    and the graph reaches its maximum at x = 0, prove that the smallest possible value of c is 0. -/
theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) →
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3260_326056


namespace NUMINAMATH_CALUDE_salary_problem_l3260_326018

theorem salary_problem (A B : ℝ) 
  (h1 : A + B = 4000)
  (h2 : A * 0.05 = B * 0.15) :
  A = 3000 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l3260_326018


namespace NUMINAMATH_CALUDE_number_calculations_l3260_326079

/-- The number that is 17 more than 5 times X -/
def number_more_than_5x (x : ℝ) : ℝ := 5 * x + 17

/-- The number that is less than 5 times 22 by Y -/
def number_less_than_5_times_22 (y : ℝ) : ℝ := 22 * 5 - y

theorem number_calculations (x y : ℝ) : 
  (number_more_than_5x x = 5 * x + 17) ∧ 
  (number_less_than_5_times_22 y = 22 * 5 - y) :=
by sorry

end NUMINAMATH_CALUDE_number_calculations_l3260_326079


namespace NUMINAMATH_CALUDE_swimmers_speed_l3260_326034

/-- Swimmer's speed in still water given time, distance, and current speed -/
theorem swimmers_speed (time : ℝ) (distance : ℝ) (current_speed : ℝ) 
  (h1 : time = 2.5)
  (h2 : distance = 5)
  (h3 : current_speed = 2) :
  ∃ v : ℝ, v = 4 ∧ time = distance / (v - current_speed) :=
by sorry

end NUMINAMATH_CALUDE_swimmers_speed_l3260_326034


namespace NUMINAMATH_CALUDE_triangle_passing_theorem_l3260_326078

/-- A triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- Whether a triangle can pass through another triangle -/
def can_pass_through (t1 t2 : Triangle) : Prop := sorry

theorem triangle_passing_theorem (T Q : Triangle) 
  (h_T_area : Triangle.area T < 4)
  (h_Q_area : Triangle.area Q = 3) :
  can_pass_through T Q := by sorry

end NUMINAMATH_CALUDE_triangle_passing_theorem_l3260_326078


namespace NUMINAMATH_CALUDE_pizza_order_count_l3260_326071

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) :
  total_slices / slices_per_pizza = 21 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_count_l3260_326071


namespace NUMINAMATH_CALUDE_widget_production_difference_l3260_326091

/-- Calculates the difference in widget production between Monday and Tuesday --/
theorem widget_production_difference (w t : ℕ) (hw : w = 2 * t) :
  w * t - (w + 4) * (t - 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_widget_production_difference_l3260_326091


namespace NUMINAMATH_CALUDE_total_ingredients_for_batches_l3260_326010

/-- The amount of flour needed for one batch of cookies (in cups) -/
def flour_per_batch : ℝ := 4

/-- The amount of sugar needed for one batch of cookies (in cups) -/
def sugar_per_batch : ℝ := 1.5

/-- The number of batches we want to make -/
def num_batches : ℕ := 8

/-- Theorem: The total amount of flour and sugar combined needed for 8 batches is 44 cups -/
theorem total_ingredients_for_batches : 
  (flour_per_batch + sugar_per_batch) * num_batches = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_ingredients_for_batches_l3260_326010


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_times_three_l3260_326098

theorem radical_conjugate_sum_times_three : 
  let x := 15 - Real.sqrt 500
  let y := 15 + Real.sqrt 500
  3 * (x + y) = 90 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_times_three_l3260_326098


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3260_326025

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (4 - 3*i) / i
  Complex.im z = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3260_326025


namespace NUMINAMATH_CALUDE_num_correct_statements_is_zero_l3260_326084

/-- Represents a programming statement --/
inductive Statement
  | Input (vars : List String)
  | Output (expr : String)
  | Assignment (lhs : String) (rhs : String)

/-- Checks if an input statement is correct --/
def isValidInput (s : Statement) : Bool :=
  match s with
  | Statement.Input vars => vars.length > 0 && vars.all (fun v => v.length > 0)
  | _ => false

/-- Checks if an output statement is correct --/
def isValidOutput (s : Statement) : Bool :=
  match s with
  | Statement.Output expr => expr.startsWith "PRINT"
  | _ => false

/-- Checks if an assignment statement is correct --/
def isValidAssignment (s : Statement) : Bool :=
  match s with
  | Statement.Assignment lhs rhs => lhs.length > 0 && !lhs.toList.head!.isDigit && !rhs.contains '='
  | _ => false

/-- Checks if a statement is correct --/
def isValidStatement (s : Statement) : Bool :=
  isValidInput s || isValidOutput s || isValidAssignment s

/-- The list of statements to check --/
def statements : List Statement :=
  [Statement.Input ["a;", "b;", "c"],
   Statement.Output "A=4",
   Statement.Assignment "3" "B",
   Statement.Assignment "A" "B=-2"]

/-- Theorem: The number of correct statements is 0 --/
theorem num_correct_statements_is_zero : 
  (statements.filter isValidStatement).length = 0 := by
  sorry


end NUMINAMATH_CALUDE_num_correct_statements_is_zero_l3260_326084


namespace NUMINAMATH_CALUDE_ab_four_necessary_not_sufficient_l3260_326000

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  b : ℝ

/-- The condition that the slopes are equal -/
def slopes_equal (l : TwoLines) : Prop :=
  l.a * l.b = 4

/-- The condition that the lines are parallel -/
def are_parallel (l : TwoLines) : Prop :=
  (2 * l.b = l.a * 2) ∧ ¬(2 * (l.b - 2) = l.a * (-1))

/-- The main theorem: ab = 4 is necessary but not sufficient for parallelism -/
theorem ab_four_necessary_not_sufficient :
  (∀ l : TwoLines, are_parallel l → slopes_equal l) ∧
  ¬(∀ l : TwoLines, slopes_equal l → are_parallel l) :=
sorry

end NUMINAMATH_CALUDE_ab_four_necessary_not_sufficient_l3260_326000


namespace NUMINAMATH_CALUDE_set_operations_proof_l3260_326057

def U := Set ℝ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}

def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem set_operations_proof :
  (Set.compl B = {x : ℝ | x < 2 ∨ x ≥ 5}) ∧
  (A ∩ (Set.compl B) = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (5 ≤ x ∧ x < 6)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_proof_l3260_326057


namespace NUMINAMATH_CALUDE_a_six_plus_seven_l3260_326030

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the sequence a_n
def a (n : ℕ) : ℝ := f n

-- State the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_periodic (x : ℝ) : f (x + 3) = f x
axiom f_neg_two : f (-2) = -3

-- State the theorem
theorem a_six_plus_seven : a f 6 + a f 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_six_plus_seven_l3260_326030
