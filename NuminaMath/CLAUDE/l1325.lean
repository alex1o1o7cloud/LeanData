import Mathlib

namespace min_value_theorem_l1325_132594

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) :
  1 / (a + 2) + 2 / (b + 1) ≥ 9 / 10 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2 * b₀ = 6 ∧ 1 / (a₀ + 2) + 2 / (b₀ + 1) = 9 / 10 := by
  sorry

end min_value_theorem_l1325_132594


namespace marble_arrangement_count_l1325_132501

/-- The number of ways to arrange 7 red marbles and n blue marbles in a row,
    where n is the maximum number of blue marbles that can be arranged such that
    the number of adjacent same-color pairs equals the number of adjacent different-color pairs -/
def M : ℕ := sorry

/-- The maximum number of blue marbles that can be arranged with 7 red marbles
    such that the number of adjacent same-color pairs equals the number of adjacent different-color pairs -/
def n : ℕ := sorry

/-- The theorem stating that M modulo 1000 equals 716 -/
theorem marble_arrangement_count : M % 1000 = 716 := by sorry

end marble_arrangement_count_l1325_132501


namespace tan_beta_value_l1325_132571

theorem tan_beta_value (α β : ℝ) 
  (h1 : Real.tan (π - α) = -(1 / 5))
  (h2 : Real.tan (α - β) = 1 / 3) : 
  Real.tan β = -(1 / 8) := by
  sorry

end tan_beta_value_l1325_132571


namespace skateboard_cost_l1325_132547

theorem skateboard_cost (total_toys : ℝ) (toy_cars : ℝ) (toy_trucks : ℝ) 
  (h1 : total_toys = 25.62)
  (h2 : toy_cars = 14.88)
  (h3 : toy_trucks = 5.86) :
  total_toys - toy_cars - toy_trucks = 4.88 := by
  sorry

end skateboard_cost_l1325_132547


namespace three_digit_rounding_l1325_132566

theorem three_digit_rounding (A : ℕ) : 
  (100 ≤ A * 100 + 76) ∧ (A * 100 + 76 < 1000) ∧ 
  ((A * 100 + 76) / 100 * 100 = 700) → A = 7 := by
  sorry

end three_digit_rounding_l1325_132566


namespace intersection_and_union_when_m_equals_3_subset_of_complement_iff_m_in_range_l1325_132500

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < m + 2}
def B : Set ℝ := {x | -4 < x ∧ x < 4}

-- Theorem for part (I)
theorem intersection_and_union_when_m_equals_3 :
  (A 3 ∩ B = {x | 1 < x ∧ x < 4}) ∧
  (A 3 ∪ B = {x | -4 < x ∧ x < 5}) := by
  sorry

-- Theorem for part (II)
theorem subset_of_complement_iff_m_in_range :
  ∀ m : ℝ, A m ⊆ Bᶜ ↔ m ≤ -6 ∨ m ≥ 6 := by
  sorry

end intersection_and_union_when_m_equals_3_subset_of_complement_iff_m_in_range_l1325_132500


namespace system_solution_ratio_l1325_132507

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 2*z = 0 →
  x*z / (y^2) = 10 := by
sorry

end system_solution_ratio_l1325_132507


namespace range_of_a_l1325_132536

theorem range_of_a (P Q : Prop) (h_or : P ∨ Q) (h_not_and : ¬(P ∧ Q))
  (h_P : P ↔ ∀ x : ℝ, x^2 - 2*x > a)
  (h_Q : Q ↔ ∃ x : ℝ, x^2 + 2*a*x + 2 = 0) :
  (-2 < a ∧ a < -1) ∨ (a ≥ 1) :=
sorry

end range_of_a_l1325_132536


namespace dilation_determinant_l1325_132583

theorem dilation_determinant (D : Matrix (Fin 3) (Fin 3) ℝ) 
  (h1 : D = Matrix.diagonal (λ _ => (5 : ℝ))) 
  (h2 : ∀ (i j : Fin 3), i ≠ j → D i j = 0) : 
  Matrix.det D = 125 := by
  sorry

end dilation_determinant_l1325_132583


namespace subset_A_l1325_132557

def A : Set ℝ := {x | x > -1}

theorem subset_A : {0} ⊆ A := by sorry

end subset_A_l1325_132557


namespace bob_has_ten_candies_l1325_132560

/-- The number of candies Bob has after trick-or-treating. -/
def bob_candies (mary_candies sue_candies john_candies sam_candies total_candies : ℕ) : ℕ :=
  total_candies - (mary_candies + sue_candies + john_candies + sam_candies)

/-- Theorem stating that Bob has 10 candies given the conditions from the problem. -/
theorem bob_has_ten_candies :
  bob_candies 5 20 5 10 50 = 10 := by sorry

end bob_has_ten_candies_l1325_132560


namespace sum_divisible_by_nine_l1325_132530

theorem sum_divisible_by_nine : 
  ∃ k : ℕ, 8230 + 8231 + 8232 + 8233 + 8234 + 8235 = 9 * k := by
  sorry

end sum_divisible_by_nine_l1325_132530


namespace truncated_cube_edge_count_l1325_132568

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  -- The number of original cube edges
  original_edges : ℕ := 12
  -- The number of corners (vertices) in the original cube
  corners : ℕ := 8
  -- The number of edges in each pentagonal face created by truncation
  pentagonal_edges : ℕ := 5
  -- Condition that cutting planes do not intersect within the cube
  non_intersecting_cuts : Prop

/-- The number of edges in a truncated cube -/
def edge_count (tc : TruncatedCube) : ℕ :=
  tc.original_edges + (tc.corners * tc.pentagonal_edges) / 2

/-- Theorem stating that a truncated cube has 32 edges -/
theorem truncated_cube_edge_count (tc : TruncatedCube) :
  edge_count tc = 32 := by
  sorry

#check truncated_cube_edge_count

end truncated_cube_edge_count_l1325_132568


namespace room_perimeter_l1325_132561

theorem room_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 12 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 16 := by
  sorry

end room_perimeter_l1325_132561


namespace regions_in_circle_l1325_132538

/-- The number of regions created by radii and concentric circles inside a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (num_radii : ℕ) (num_concentric_circles : ℕ) 
  (h1 : num_radii = 16) (h2 : num_concentric_circles = 10) : 
  num_regions num_radii num_concentric_circles = 176 := by
  sorry

#eval num_regions 16 10  -- Should output 176

end regions_in_circle_l1325_132538


namespace arithmetic_calculation_l1325_132555

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end arithmetic_calculation_l1325_132555


namespace solve_equation_l1325_132545

theorem solve_equation (x : ℝ) (h1 : x > 5) 
  (h2 : Real.sqrt (x - 3 * Real.sqrt (x - 5)) + 3 = Real.sqrt (x + 3 * Real.sqrt (x - 5)) - 3) : 
  x = 41 := by
  sorry

end solve_equation_l1325_132545


namespace collinearity_condition_l1325_132527

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points are collinear if the area of the triangle formed by them is zero -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

theorem collinearity_condition (n : ℝ) : 
  let A : Point := ⟨1, 1⟩
  let B : Point := ⟨4, 0⟩
  let C : Point := ⟨0, n⟩
  collinear A B C ↔ n = 4/3 := by
  sorry

end collinearity_condition_l1325_132527


namespace christinas_driving_time_l1325_132562

theorem christinas_driving_time 
  (total_distance : ℝ) 
  (christina_speed : ℝ) 
  (friend_speed : ℝ) 
  (friend_time : ℝ) 
  (h1 : total_distance = 210)
  (h2 : christina_speed = 30)
  (h3 : friend_speed = 40)
  (h4 : friend_time = 3) :
  (total_distance - friend_speed * friend_time) / christina_speed * 60 = 180 :=
by sorry

end christinas_driving_time_l1325_132562


namespace shaggy_seed_count_l1325_132598

/-- Represents the number of seeds Shaggy ate -/
def shaggy_seeds : ℕ := 54

/-- Represents the total number of seeds -/
def total_seeds : ℕ := 60

/-- Represents the ratio of Shaggy's berry eating speed to Fluffball's -/
def berry_speed_ratio : ℕ := 6

/-- Represents the ratio of Shaggy's seed eating speed to Fluffball's -/
def seed_speed_ratio : ℕ := 3

/-- Represents the ratio of berries Shaggy ate to Fluffball -/
def berry_ratio : ℕ := 2

theorem shaggy_seed_count : 
  50 < total_seeds ∧ 
  total_seeds < 65 ∧ 
  berry_speed_ratio = 6 ∧ 
  seed_speed_ratio = 3 ∧ 
  berry_ratio = 2 → 
  shaggy_seeds = 54 := by sorry

end shaggy_seed_count_l1325_132598


namespace complex_equation_solution_l1325_132559

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 1 + Complex.I) → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l1325_132559


namespace dividend_calculation_l1325_132533

theorem dividend_calculation (divisor quotient remainder : Int) 
  (h1 : divisor = 800)
  (h2 : quotient = 594)
  (h3 : remainder = -968) :
  divisor * quotient + remainder = 474232 := by
  sorry

end dividend_calculation_l1325_132533


namespace tan_product_pi_ninths_l1325_132589

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end tan_product_pi_ninths_l1325_132589


namespace a_bounds_circle_D_equation_l1325_132565

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line L
def line_L (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the function a
def a (x y : ℝ) : ℝ := y - x

-- Theorem for the maximum and minimum values of a on circle C
theorem a_bounds :
  (∃ x y : ℝ, circle_C x y ∧ a x y = 2 * Real.sqrt 2 + 1) ∧
  (∃ x y : ℝ, circle_C x y ∧ a x y = 1 - 2 * Real.sqrt 2) ∧
  (∀ x y : ℝ, circle_C x y → 1 - 2 * Real.sqrt 2 ≤ a x y ∧ a x y ≤ 2 * Real.sqrt 2 + 1) :=
sorry

-- Define circle D
def circle_D (center_x center_y x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 9

-- Theorem for the equation of circle D
theorem circle_D_equation :
  ∃ center_x center_y : ℝ,
    line_L center_x center_y ∧
    ((∀ x y : ℝ, circle_D center_x center_y x y ↔ (x - 3)^2 + (y + 1)^2 = 9) ∨
     (∀ x y : ℝ, circle_D center_x center_y x y ↔ (x + 2)^2 + (y - 4)^2 = 9)) ∧
    (∃ x y : ℝ, circle_C x y ∧ (x - center_x)^2 + (y - center_y)^2 = 25) :=
sorry

end a_bounds_circle_D_equation_l1325_132565


namespace max_two_match_winners_100_l1325_132539

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)

/-- Represents the number of matches a participant has won -/
def wins (t : Tournament) (p : ℕ) : ℕ := sorry

/-- The number of participants who have won exactly two matches -/
def two_match_winners (t : Tournament) : ℕ := sorry

/-- The maximum possible number of two-match winners -/
def max_two_match_winners (t : Tournament) : ℕ := sorry

theorem max_two_match_winners_100 :
  ∀ t : Tournament, t.participants = 100 → max_two_match_winners t = 49 := by
  sorry

end max_two_match_winners_100_l1325_132539


namespace geometry_books_shelf_filling_l1325_132514

/-- Represents the number of books that fill a shelf. -/
structure ShelfFilling where
  algebra : ℕ
  geometry : ℕ

/-- Represents the properties of the book arrangement problem. -/
structure BookArrangement where
  P : ℕ  -- Total number of algebra books
  Q : ℕ  -- Total number of geometry books
  X : ℕ  -- Number of algebra books that fill the shelf
  Y : ℕ  -- Number of geometry books that fill the shelf

/-- The main theorem about the number of geometry books (Z) that fill the shelf. -/
theorem geometry_books_shelf_filling 
  (arr : BookArrangement) 
  (fill1 : ShelfFilling)
  (fill2 : ShelfFilling)
  (h1 : fill1.algebra = arr.X ∧ fill1.geometry = arr.Y)
  (h2 : fill2.algebra = 2 * fill2.geometry)
  (h3 : arr.P + 2 * arr.Q = arr.X + 2 * arr.Y) :
  ∃ Z : ℕ, Z = (arr.P + 2 * arr.Q) / 2 ∧ 
             Z * 2 = arr.P + 2 * arr.Q ∧
             fill2.geometry = Z :=
by sorry

end geometry_books_shelf_filling_l1325_132514


namespace minuend_calculation_l1325_132570

theorem minuend_calculation (subtrahend difference : ℝ) 
  (h1 : subtrahend = 1.34)
  (h2 : difference = 3.66) : 
  subtrahend + difference = 5 := by
  sorry

end minuend_calculation_l1325_132570


namespace xiao_ming_reading_progress_l1325_132511

/-- Calculates the starting page for the 6th day of reading -/
def starting_page_6th_day (total_pages book_pages_per_day days_read : ℕ) : ℕ :=
  book_pages_per_day * days_read + 1

/-- Proves that the starting page for the 6th day is 301 -/
theorem xiao_ming_reading_progress : starting_page_6th_day 500 60 5 = 301 := by
  sorry

end xiao_ming_reading_progress_l1325_132511


namespace lincoln_county_population_l1325_132590

/-- The number of cities in the County of Lincoln -/
def num_cities : ℕ := 25

/-- The lower bound of the average population -/
def lower_bound : ℕ := 5200

/-- The upper bound of the average population -/
def upper_bound : ℕ := 5700

/-- The average population of the cities -/
def avg_population : ℚ := (lower_bound + upper_bound) / 2

/-- The total population of all cities -/
def total_population : ℕ := 136250

theorem lincoln_county_population :
  (num_cities : ℚ) * avg_population = total_population := by
  sorry

end lincoln_county_population_l1325_132590


namespace matthew_score_proof_l1325_132575

def basket_value : ℕ := 3
def total_baskets : ℕ := 5
def shawn_points : ℕ := 6

def matthew_points : ℕ := 9

theorem matthew_score_proof :
  matthew_points = basket_value * total_baskets - shawn_points :=
by sorry

end matthew_score_proof_l1325_132575


namespace angle_C_is_30_degrees_ab_range_when_c_is_1_l1325_132554

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < Real.pi / 2
  acute_B : 0 < B ∧ B < Real.pi / 2
  acute_C : 0 < C ∧ C < Real.pi / 2
  tan_C_eq : Real.tan C = (a * b) / (a^2 + b^2 - c^2)

/-- Theorem stating that if tan C = (ab) / (a² + b² - c²) in an acute triangle, then C = 30° -/
theorem angle_C_is_30_degrees (t : AcuteTriangle) : t.C = Real.pi / 6 := by
  sorry

/-- Theorem stating that if c = 1 and tan C = (ab) / (a² + b² - 1) in an acute triangle, 
    then 2√3 < ab ≤ 2 + √3 -/
theorem ab_range_when_c_is_1 (t : AcuteTriangle) (h : t.c = 1) : 
  2 * Real.sqrt 3 < t.a * t.b ∧ t.a * t.b ≤ 2 + Real.sqrt 3 := by
  sorry

end angle_C_is_30_degrees_ab_range_when_c_is_1_l1325_132554


namespace arithmetic_mean_after_removal_l1325_132544

/-- Given a set of 60 numbers with an arithmetic mean of 42, 
    prove that removing 50 and 60 results in a new arithmetic mean of 41.5 -/
theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) : 
  S.card = 60 → 
  x ∈ S → 
  y ∈ S → 
  x = 50 → 
  y = 60 → 
  (S.sum id) / S.card = 42 → 
  ((S.sum id) - x - y) / (S.card - 2) = 41.5 := by
sorry

end arithmetic_mean_after_removal_l1325_132544


namespace max_enclosed_area_l1325_132592

/-- Represents an infinite chessboard -/
structure InfiniteChessboard where

/-- Represents a closed non-self-intersecting polygonal line on the chessboard -/
structure PolygonalLine where
  chessboard : InfiniteChessboard
  is_closed : Bool
  is_non_self_intersecting : Bool
  along_cell_sides : Bool

/-- Represents the area enclosed by a polygonal line -/
def EnclosedArea (line : PolygonalLine) : ℕ := sorry

/-- Counts the number of black cells inside a polygonal line -/
def BlackCellsCount (line : PolygonalLine) : ℕ := sorry

/-- Theorem stating the maximum area enclosed by a polygonal line -/
theorem max_enclosed_area (line : PolygonalLine) (k : ℕ) 
  (h1 : line.is_closed = true)
  (h2 : line.is_non_self_intersecting = true)
  (h3 : line.along_cell_sides = true)
  (h4 : BlackCellsCount line = k) :
  EnclosedArea line ≤ 4 * k + 1 := by sorry

end max_enclosed_area_l1325_132592


namespace victor_percentage_l1325_132573

def max_marks : ℕ := 500
def victor_marks : ℕ := 460

theorem victor_percentage : 
  (victor_marks : ℚ) / max_marks * 100 = 92 := by sorry

end victor_percentage_l1325_132573


namespace percentage_of_b_l1325_132597

/-- Given that 12 is 6% of a, a certain percentage of b is 6, and c equals b / a,
    prove that the percentage of b is 6 / (200 * c) * 100 -/
theorem percentage_of_b (a b c : ℝ) (h1 : 0.06 * a = 12) (h2 : ∃ p, p * b = 6) (h3 : c = b / a) :
  ∃ p, p * b = 6 ∧ p * 100 = 6 / (200 * c) * 100 := by
  sorry

end percentage_of_b_l1325_132597


namespace sum_of_fourth_powers_of_roots_l1325_132535

theorem sum_of_fourth_powers_of_roots (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) → 
  (q^3 - 2*q^2 + 3*q - 4 = 0) → 
  (r^3 - 2*r^2 + 3*r - 4 = 0) → 
  p^4 + q^4 + r^4 = 18 := by
sorry

end sum_of_fourth_powers_of_roots_l1325_132535


namespace sum_equals_3000_length_conversion_l1325_132503

-- Problem 1
theorem sum_equals_3000 : 1361 + 972 + 639 + 28 = 3000 := by sorry

-- Problem 2
theorem length_conversion :
  ∀ (meters decimeters centimeters : ℕ),
    meters * 10 + decimeters - (centimeters / 10) = 91 →
    9 * 10 + 9 - (80 / 10) = 91 := by sorry

end sum_equals_3000_length_conversion_l1325_132503


namespace line_circle_intersection_l1325_132552

/-- Given a line and a circle that intersect, prove the value of the line's slope --/
theorem line_circle_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 - A.2 + 3 = 0) ∧ 
    (a * B.1 - B.2 + 3 = 0) ∧ 
    ((A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧ 
    ((B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  a = 0 := by
sorry

end line_circle_intersection_l1325_132552


namespace corner_sum_6x12_board_l1325_132522

/-- Represents a rectangular board filled with consecutive numbers -/
structure NumberBoard where
  rows : Nat
  cols : Nat
  total_numbers : Nat

/-- Returns the number at a given position on the board -/
def NumberBoard.number_at (board : NumberBoard) (row : Nat) (col : Nat) : Nat :=
  (row - 1) * board.cols + col

/-- Theorem stating that the sum of corner numbers on a 6x12 board is 146 -/
theorem corner_sum_6x12_board :
  let board : NumberBoard := ⟨6, 12, 72⟩
  (board.number_at 1 1) + (board.number_at 1 12) +
  (board.number_at 6 1) + (board.number_at 6 12) = 146 := by
  sorry

#check corner_sum_6x12_board

end corner_sum_6x12_board_l1325_132522


namespace congruence_problem_l1325_132546

theorem congruence_problem (x : ℤ) :
  (4 * x + 9) % 20 = 3 → (3 * x + 15) % 20 = 10 := by
  sorry

end congruence_problem_l1325_132546


namespace three_zeros_condition_l1325_132534

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Theorem: For f(x) to have exactly 3 zeros, 'a' must be in the range (-∞, -3) -/
theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end three_zeros_condition_l1325_132534


namespace complete_square_quadratic_l1325_132541

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 14*x + 24 = 0 ↔ (x + c)^2 = d ∧ d = 25 := by
sorry

end complete_square_quadratic_l1325_132541


namespace centroid_perpendicular_triangle_area_l1325_132584

/-- Given a triangle ABC with sides a, b, c, and area S, prove that the area of the triangle 
    formed by the bases of perpendiculars dropped from the centroid to the sides of ABC 
    is equal to (4/9) * (a² + b² + c²) / (a² * b² * c²) * S³ -/
theorem centroid_perpendicular_triangle_area 
  (a b c : ℝ) 
  (S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : S > 0) : 
  ∃ (S_new : ℝ), S_new = (4/9) * (a^2 + b^2 + c^2) / (a^2 * b^2 * c^2) * S^3 := by
  sorry

end centroid_perpendicular_triangle_area_l1325_132584


namespace shortest_side_is_10_area_is_integer_l1325_132548

/-- Represents a triangle with integer side lengths and area --/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  sum_eq_perimeter : a + b + c = 48
  one_side_eq_21 : a = 21 ∨ b = 21 ∨ c = 21

/-- The shortest side of the triangle is 10 --/
theorem shortest_side_is_10 (t : IntegerTriangle) : 
  min t.a (min t.b t.c) = 10 := by
  sorry

/-- The area of the triangle is an integer --/
theorem area_is_integer (t : IntegerTriangle) : 
  ∃ (s : ℕ), 4 * t.area = s * s * (t.a + t.b - t.c) * (t.a + t.c - t.b) * (t.b + t.c - t.a) := by
  sorry

end shortest_side_is_10_area_is_integer_l1325_132548


namespace least_zogs_for_dropping_beats_eating_l1325_132577

theorem least_zogs_for_dropping_beats_eating :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, k > 0 → k < n → k * (k + 1) ≤ 15 * k) → 15 * 15 < 15 * (15 + 1) :=
sorry

end least_zogs_for_dropping_beats_eating_l1325_132577


namespace triangle_count_theorem_l1325_132517

/-- The number of triangles formed by selecting three non-collinear points from a set of points on a triangle -/
def num_triangles (a b c : ℕ) : ℕ :=
  let total_points := 3 + a + b + c
  let total_combinations := (total_points.choose 3)
  let collinear_combinations := (a + 2).choose 3 + (b + 2).choose 3 + (c + 2).choose 3
  total_combinations - collinear_combinations

/-- Theorem stating that the number of triangles formed in the given configuration is 357 -/
theorem triangle_count_theorem : num_triangles 2 3 7 = 357 := by
  sorry

end triangle_count_theorem_l1325_132517


namespace base8_digit_product_12345_l1325_132582

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Computes the product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

/-- The product of the digits in the base 8 representation of 12345 (base 10) is 0 --/
theorem base8_digit_product_12345 :
  listProduct (toBase8 12345) = 0 := by
  sorry

end base8_digit_product_12345_l1325_132582


namespace angle_measure_problem_l1325_132579

/-- Given two supplementary angles C and D, where C is 5 times D, prove that the measure of angle C is 150°. -/
theorem angle_measure_problem (C D : ℝ) : 
  C + D = 180 →  -- C and D are supplementary
  C = 5 * D →    -- C is 5 times D
  C = 150 :=     -- The measure of angle C is 150°
by
  sorry

end angle_measure_problem_l1325_132579


namespace sum_of_divisors_540_has_4_prime_factors_l1325_132518

-- Define the number we're working with
def n : ℕ := 540

-- Define the sum of positive divisors function
noncomputable def sum_of_divisors (m : ℕ) : ℕ := sorry

-- Define a function to count distinct prime factors
noncomputable def count_distinct_prime_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_divisors_540_has_4_prime_factors :
  count_distinct_prime_factors (sum_of_divisors n) = 4 := by sorry

end sum_of_divisors_540_has_4_prime_factors_l1325_132518


namespace percentage_in_at_least_two_trips_l1325_132558

/-- Represents the percentage of students who went on a specific trip -/
structure TripParticipation where
  threeDay : Rat
  twoDay : Rat
  oneDay : Rat

/-- Represents the percentage of students who participated in multiple trips -/
structure MultipleTrips where
  threeDayAndOneDay : Rat
  twoDayAndOther : Rat

/-- Calculates the percentage of students who participated in at least two trips -/
def percentageInAtLeastTwoTrips (tp : TripParticipation) (mt : MultipleTrips) : Rat :=
  mt.threeDayAndOneDay + mt.twoDayAndOther

/-- Main theorem: The percentage of students who participated in at least two trips is 22% -/
theorem percentage_in_at_least_two_trips :
  ∀ (tp : TripParticipation) (mt : MultipleTrips),
  tp.threeDay = 25/100 ∧
  tp.twoDay = 10/100 ∧
  mt.threeDayAndOneDay = 65/100 * tp.threeDay ∧
  mt.twoDayAndOther = 60/100 * tp.twoDay →
  percentageInAtLeastTwoTrips tp mt = 22/100 := by
  sorry

end percentage_in_at_least_two_trips_l1325_132558


namespace train_length_train_length_proof_l1325_132528

/-- The length of a train given specific crossing times -/
theorem train_length (t_man : ℝ) (t_platform : ℝ) (l_platform : ℝ) : ℝ :=
  let train_length := (t_platform * l_platform) / (t_platform - t_man)
  186

/-- The train passes a stationary point in 8 seconds -/
def time_passing_man : ℝ := 8

/-- The train crosses a platform in 20 seconds -/
def time_crossing_platform : ℝ := 20

/-- The length of the platform is 279 meters -/
def platform_length : ℝ := 279

theorem train_length_proof :
  train_length time_passing_man time_crossing_platform platform_length = 186 := by
  sorry

end train_length_train_length_proof_l1325_132528


namespace sugar_measurement_l1325_132540

theorem sugar_measurement (sugar_needed : ℚ) (cup_capacity : ℚ) : 
  sugar_needed = 3 + 3 / 4 ∧ cup_capacity = 1 / 3 → 
  ↑(Int.ceil ((sugar_needed / cup_capacity) : ℚ)) = 12 :=
by sorry

end sugar_measurement_l1325_132540


namespace complex_number_modulus_l1325_132508

theorem complex_number_modulus : 
  let z : ℂ := 2 / (1 + Complex.I) + (1 - Complex.I)^2
  Complex.abs z = Real.sqrt 10 := by
sorry

end complex_number_modulus_l1325_132508


namespace only_common_term_is_one_l1325_132537

def x : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem only_common_term_is_one : ∀ n : ℕ, x n = y n ↔ n = 0 := by sorry

end only_common_term_is_one_l1325_132537


namespace shaded_squares_in_six_by_six_grid_l1325_132599

/-- Represents a grid with a given size and number of unshaded squares per row -/
structure Grid where
  size : Nat
  unshadedPerRow : Nat

/-- Calculates the number of shaded squares in the grid -/
def shadedSquares (g : Grid) : Nat :=
  g.size * (g.size - g.unshadedPerRow)

theorem shaded_squares_in_six_by_six_grid :
  ∀ (g : Grid), g.size = 6 → g.unshadedPerRow = 1 → shadedSquares g = 30 := by
  sorry

end shaded_squares_in_six_by_six_grid_l1325_132599


namespace square_sum_equality_l1325_132574

theorem square_sum_equality (y : ℝ) : 
  (y - 2)^2 + 2*(y - 2)*(5 + y) + (5 + y)^2 = (2*y + 3)^2 := by
  sorry

end square_sum_equality_l1325_132574


namespace modified_triangle_property_unbounded_l1325_132529

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- A function that checks if a set of 10 consecutive integers contains a right triangle -/
def has_right_triangle (start : ℕ) : Prop :=
  ∃ (a b c : ℕ), start ≤ a ∧ a < b ∧ b < c ∧ c < start + 10 ∧ is_right_triangle a b c

/-- The main theorem stating that for any k ≥ 10, the set {5, 6, ..., k} 
    satisfies the modified triangle property for all 10-element subsets -/
theorem modified_triangle_property_unbounded (k : ℕ) (h : k ≥ 10) :
  ∀ (n : ℕ), 5 ≤ n ∧ n ≤ k - 9 → has_right_triangle n :=
sorry

end modified_triangle_property_unbounded_l1325_132529


namespace completing_square_equivalence_l1325_132551

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x - 1 = 0

-- Define the completed square form
def completed_square (x : ℝ) : Prop := (x - 2)^2 = 5

-- Theorem stating that the completed square form is equivalent to the original equation
theorem completing_square_equivalence :
  ∀ x : ℝ, original_equation x ↔ completed_square x :=
by sorry

end completing_square_equivalence_l1325_132551


namespace cosine_fourth_minus_sine_fourth_l1325_132542

theorem cosine_fourth_minus_sine_fourth (θ : ℝ) : 
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by
  sorry

end cosine_fourth_minus_sine_fourth_l1325_132542


namespace quadratic_roots_preservation_l1325_132588

theorem quadratic_roots_preservation (p q α : ℝ) 
  (h1 : ∃ x : ℝ, x^2 + p*x + q = 0) 
  (h2 : 0 < α) (h3 : α ≤ 1) : 
  ∃ y : ℝ, α*y^2 + p*y + q = 0 :=
sorry

end quadratic_roots_preservation_l1325_132588


namespace property_P_theorems_seq_012_has_property_P_l1325_132572

/-- Definition of a sequence with property P -/
def has_property_P (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 3 ∧
  (∀ i j, 1 ≤ i → i ≤ j → j ≤ k → (∃ n ≤ k, a n = a j + a i ∨ a n = a j - a i)) ∧
  (∀ i, 1 ≤ i → i < k → a i < a (i + 1)) ∧
  0 ≤ a 1

theorem property_P_theorems (a : ℕ → ℝ) (k : ℕ) (h : has_property_P a k) :
  (∀ i ≤ k, a k - a i ∈ Set.range (fun n => a n)) ∧
  (k ≥ 5 → ∃ d : ℝ, ∀ i < k, a (i + 1) - a i = d) :=
by sorry

/-- The sequence 0, 1, 2 has property P -/
theorem seq_012_has_property_P :
  has_property_P (fun n => if n = 1 then 0 else if n = 2 then 1 else 2) 3 :=
by sorry

end property_P_theorems_seq_012_has_property_P_l1325_132572


namespace preimage_of_4_3_l1325_132581

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem preimage_of_4_3 :
  ∃ (p : ℝ × ℝ), f p = (4, 3) ∧ p = (2, 1) := by
sorry

end preimage_of_4_3_l1325_132581


namespace conditions_sufficient_not_necessary_l1325_132567

theorem conditions_sufficient_not_necessary (m : ℝ) (h : m > 0) :
  (∀ x y a : ℝ, |x - a| < m ∧ |y - a| < m → |x - y| < 2*m) ∧
  (∃ x y a : ℝ, |x - y| < 2*m ∧ (|x - a| ≥ m ∨ |y - a| ≥ m)) :=
by sorry

end conditions_sufficient_not_necessary_l1325_132567


namespace linear_function_not_in_third_quadrant_l1325_132523

/-- A linear function that does not pass through the third quadrant -/
structure LinearFunctionNotInThirdQuadrant where
  k : ℝ
  b : ℝ
  not_in_third_quadrant : ∀ x y : ℝ, y = k * x + b → ¬(x < 0 ∧ y < 0)

/-- Theorem: For a linear function y = kx + b that does not pass through the third quadrant,
    k is negative and b is non-negative -/
theorem linear_function_not_in_third_quadrant
  (f : LinearFunctionNotInThirdQuadrant) : f.k < 0 ∧ f.b ≥ 0 := by
  sorry

end linear_function_not_in_third_quadrant_l1325_132523


namespace quadratic_inequality_l1325_132550

theorem quadratic_inequality (m : ℝ) :
  (∀ x : ℝ, m * x^2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m ≤ 0) :=
by sorry

end quadratic_inequality_l1325_132550


namespace inverse_of_proposition_l1325_132549

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x > 2 → x > 1

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop := x > 1 → x > 2

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_proposition :
  (∀ x, original_proposition x) ↔ (∀ x, inverse_proposition x) :=
sorry

end inverse_of_proposition_l1325_132549


namespace log_inequality_l1325_132509

/-- Given a = log_3(2), b = log_2(3), and c = log_(1/2)(5), prove that c < a < b -/
theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 2 / Real.log 3)
  (hb : b = Real.log 3 / Real.log 2)
  (hc : c = Real.log 5 / Real.log (1/2)) :
  c < a ∧ a < b := by
  sorry

end log_inequality_l1325_132509


namespace train_passing_jogger_time_train_passes_jogger_in_25_seconds_l1325_132586

/-- Time taken for a train to pass a jogger -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (train_length initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time taken for the train to pass the jogger is 25 seconds -/
theorem train_passes_jogger_in_25_seconds : 
  train_passing_jogger_time 9 45 100 150 = 25 := by
  sorry

end train_passing_jogger_time_train_passes_jogger_in_25_seconds_l1325_132586


namespace two_digit_sum_divisible_by_11_four_digit_divisible_by_11_l1325_132531

-- Problem 1
theorem two_digit_sum_divisible_by_11 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : ℤ, (10 * a + b) + (10 * b + a) = 11 * k :=
sorry

-- Problem 2
theorem four_digit_divisible_by_11 (m n : ℕ) (h1 : m < 10) (h2 : n < 10) :
  ∃ k : ℤ, 1000 * m + 100 * n + 10 * n + m = 11 * k :=
sorry

end two_digit_sum_divisible_by_11_four_digit_divisible_by_11_l1325_132531


namespace consecutive_squares_determinant_l1325_132585

theorem consecutive_squares_determinant (n : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    (n + (i.val * 3 + j.val : ℕ))^2
  Matrix.det M = -6^3 := by
  sorry

end consecutive_squares_determinant_l1325_132585


namespace parallelogram_network_l1325_132513

theorem parallelogram_network (first_set : ℕ) (total_parallelograms : ℕ) 
  (h1 : first_set = 8) 
  (h2 : total_parallelograms = 784) : 
  ∃ (second_set : ℕ), 
    second_set > 0 ∧ 
    (first_set - 1) * (second_set - 1) = total_parallelograms := by
  sorry

end parallelogram_network_l1325_132513


namespace find_genuine_stacks_l1325_132524

/-- Represents a stack of coins -/
structure CoinStack :=
  (count : Nat)
  (hasOddCoin : Bool)

/-- Represents the result of weighing two stacks -/
inductive WeighResult
  | Equal
  | Unequal

/-- Represents the state of the coin stacks -/
structure CoinStacks :=
  (stack1 : CoinStack)
  (stack2 : CoinStack)
  (stack3 : CoinStack)
  (stack4 : CoinStack)

/-- Represents a weighing action -/
def weigh (s1 s2 : CoinStack) : WeighResult :=
  if s1.hasOddCoin = s2.hasOddCoin then WeighResult.Equal else WeighResult.Unequal

/-- The main theorem -/
theorem find_genuine_stacks 
  (stacks : CoinStacks)
  (h1 : stacks.stack1.count = 5)
  (h2 : stacks.stack2.count = 6)
  (h3 : stacks.stack3.count = 7)
  (h4 : stacks.stack4.count = 19)
  (h5 : (stacks.stack1.hasOddCoin || stacks.stack2.hasOddCoin || stacks.stack3.hasOddCoin || stacks.stack4.hasOddCoin) ∧ 
        (¬stacks.stack1.hasOddCoin ∨ ¬stacks.stack2.hasOddCoin ∨ ¬stacks.stack3.hasOddCoin ∨ ¬stacks.stack4.hasOddCoin)) :
  ∃ (s1 s2 : CoinStack), s1 ∈ [stacks.stack1, stacks.stack2, stacks.stack3, stacks.stack4] ∧ 
                         s2 ∈ [stacks.stack1, stacks.stack2, stacks.stack3, stacks.stack4] ∧ 
                         s1 ≠ s2 ∧ 
                         ¬s1.hasOddCoin ∧ 
                         ¬s2.hasOddCoin := by
  sorry


end find_genuine_stacks_l1325_132524


namespace fraction_simplification_l1325_132505

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (a - b) / (2*a*b - b^2 - a^2) = 1 / (b - a) := by sorry

end fraction_simplification_l1325_132505


namespace percentage_subtraction_l1325_132578

theorem percentage_subtraction (original : ℝ) (incorrect_subtraction : ℝ) (difference : ℝ) : 
  original = 200 →
  incorrect_subtraction = 25 →
  difference = 25 →
  let incorrect_result := original - incorrect_subtraction
  let correct_result := incorrect_result - difference
  let percentage := (original - correct_result) / original * 100
  percentage = 25 := by sorry

end percentage_subtraction_l1325_132578


namespace circles_externally_tangent_l1325_132593

/-- Two circles are externally tangent if and only if the distance between their centers
    is equal to the sum of their radii. -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

/-- The theorem stating that two circles with radii 2 and 3, whose centers are 5 units apart,
    are externally tangent. -/
theorem circles_externally_tangent :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let d : ℝ := 5
  externally_tangent r₁ r₂ d :=
by
  sorry


end circles_externally_tangent_l1325_132593


namespace age_ratio_theorem_l1325_132563

/-- Represents the ages of John and Emily -/
structure Ages where
  john : ℕ
  emily : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.john - 3 = 5 * (ages.emily - 3)) ∧
  (ages.john - 7 = 6 * (ages.emily - 7))

/-- The theorem to be proved -/
theorem age_ratio_theorem (ages : Ages) :
  problem_conditions ages →
  ∃ x : ℕ, x = 17 ∧ (ages.john + x) / (ages.emily + x) = 3 :=
by
  sorry


end age_ratio_theorem_l1325_132563


namespace product_of_largest_primes_l1325_132569

/-- The largest one-digit prime -/
def largest_one_digit_prime : ℕ := 7

/-- The second largest one-digit prime -/
def second_largest_one_digit_prime : ℕ := 5

/-- The largest three-digit prime -/
def largest_three_digit_prime : ℕ := 997

/-- Theorem stating that the product of the two largest one-digit primes
    and the largest three-digit prime is 34895 -/
theorem product_of_largest_primes :
  largest_one_digit_prime * second_largest_one_digit_prime * largest_three_digit_prime = 34895 := by
  sorry

end product_of_largest_primes_l1325_132569


namespace at_least_one_less_than_one_l1325_132502

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  min a (min b c) < 1 := by
  sorry

end at_least_one_less_than_one_l1325_132502


namespace cell_diameter_scientific_notation_l1325_132564

/-- Expresses a given number in scientific notation -/
def scientificNotation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem cell_diameter_scientific_notation :
  scientificNotation 0.00065 = (6.5, -4) := by sorry

end cell_diameter_scientific_notation_l1325_132564


namespace sarah_apples_l1325_132521

theorem sarah_apples (boxes : ℕ) (apples_per_box : ℕ) (h1 : boxes = 7) (h2 : apples_per_box = 7) :
  boxes * apples_per_box = 49 := by
  sorry

end sarah_apples_l1325_132521


namespace inequality_solution_set_l1325_132506

theorem inequality_solution_set :
  let f : ℝ → ℝ := λ x ↦ 2 * x
  let integral_value : ℝ := ∫ x in (0:ℝ)..1, f x
  {x : ℝ | |x - 2| > integral_value} = Set.Ioi 3 ∪ Set.Iio 1 :=
by sorry

end inequality_solution_set_l1325_132506


namespace flour_scoops_l1325_132595

/-- Given a bag of flour, the amount needed for a recipe, and the size of a measuring cup,
    calculate the number of scoops to remove from the bag. -/
def scoop_count (bag_size : ℚ) (recipe_amount : ℚ) (measure_size : ℚ) : ℚ :=
  (bag_size - recipe_amount) / measure_size

theorem flour_scoops :
  let bag_size : ℚ := 8
  let recipe_amount : ℚ := 6
  let measure_size : ℚ := 1/4
  scoop_count bag_size recipe_amount measure_size = 8 := by sorry

end flour_scoops_l1325_132595


namespace thirty_sided_polygon_diagonals_l1325_132526

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by
  sorry

#eval num_diagonals 30  -- This will evaluate to 405

end thirty_sided_polygon_diagonals_l1325_132526


namespace chicken_problem_l1325_132520

/-- The problem of calculating the difference in number of chickens bought by John and Ray -/
theorem chicken_problem (chicken_cost : ℕ) (john_extra : ℕ) (ray_less : ℕ) (ray_chickens : ℕ) :
  chicken_cost = 3 →
  john_extra = 15 →
  ray_less = 18 →
  ray_chickens = 10 →
  (john_extra + ray_less + ray_chickens * chicken_cost) / chicken_cost - ray_chickens = 11 :=
by sorry

end chicken_problem_l1325_132520


namespace heron_height_calculation_l1325_132580

theorem heron_height_calculation (a b c : ℝ) (h : ℝ) :
  a = 20 ∧ b = 99 ∧ c = 101 →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  h = 2 * area / b →
  h = 20 := by
  sorry

end heron_height_calculation_l1325_132580


namespace inequality_solution_l1325_132510

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end inequality_solution_l1325_132510


namespace isosceles_triangle_smallest_angle_isosceles_triangle_smallest_angle_proof_l1325_132591

/-- An isosceles triangle with one angle 40% larger than a right angle has two smallest angles measuring 27°. -/
theorem isosceles_triangle_smallest_angle : ℝ → Prop :=
  fun x =>
    let right_angle := 90
    let large_angle := 1.4 * right_angle
    let sum_of_angles := 180
    x > 0 ∧ 
    x < large_angle ∧ 
    2 * x + large_angle = sum_of_angles →
    x = 27

/-- Proof of the theorem -/
theorem isosceles_triangle_smallest_angle_proof : isosceles_triangle_smallest_angle 27 := by
  sorry

end isosceles_triangle_smallest_angle_isosceles_triangle_smallest_angle_proof_l1325_132591


namespace complement_A_intersect_B_l1325_132553

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end complement_A_intersect_B_l1325_132553


namespace rhombus_area_in_hexagon_l1325_132596

/-- A regular hexagon -/
structure RegularHexagon where
  area : ℝ

/-- The total area of rhombuses that can be formed within a regular hexagon -/
def total_rhombus_area (h : RegularHexagon) : ℝ :=
  sorry

/-- Theorem: In a regular hexagon with area 80, the total area of rhombuses is 45 -/
theorem rhombus_area_in_hexagon (h : RegularHexagon) 
  (h_area : h.area = 80) : total_rhombus_area h = 45 :=
  sorry

end rhombus_area_in_hexagon_l1325_132596


namespace birds_left_in_cage_l1325_132587

/-- The number of birds initially in the cage -/
def initial_birds : ℕ := 19

/-- The number of birds taken out of the cage -/
def birds_taken_out : ℕ := 10

/-- Theorem stating that the number of birds left in the cage is 9 -/
theorem birds_left_in_cage : initial_birds - birds_taken_out = 9 := by
  sorry

end birds_left_in_cage_l1325_132587


namespace max_square_field_size_l1325_132532

/-- The maximum size of a square field that can be fully fenced given the specified conditions -/
theorem max_square_field_size (wire_cost : ℝ) (budget : ℝ) : 
  wire_cost = 30 → 
  budget = 120000 → 
  (budget / wire_cost : ℝ) < 4000 → 
  (budget / wire_cost / 4 : ℝ) ^ 2 = 1000000 := by
  sorry

end max_square_field_size_l1325_132532


namespace chord_length_square_of_quarter_circle_l1325_132576

/-- Given a circular sector with central angle 90° and radius 10 cm,
    the square of the chord length connecting the arc endpoints is 200 cm². -/
theorem chord_length_square_of_quarter_circle (r : ℝ) (h : r = 10) :
  let chord_length_square := 2 * r^2
  chord_length_square = 200 := by sorry

end chord_length_square_of_quarter_circle_l1325_132576


namespace tangent_line_range_l1325_132525

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line is equal to the radius of the circle. -/
def is_tangent_line (m n : ℝ) : Prop :=
  1 = |(m + 1) + (n + 1) - 2| / Real.sqrt ((m + 1)^2 + (n + 1)^2)

/-- The range of m + n when the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1 -/
theorem tangent_line_range (m n : ℝ) :
  is_tangent_line m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
by sorry

end tangent_line_range_l1325_132525


namespace plane_perpendicularity_l1325_132504

/-- Two different lines in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem plane_perpendicularity (m n : Line3D) (α β : Plane3D) 
  (h1 : m ≠ n) (h2 : α ≠ β) (h3 : perpendicular m α) (h4 : parallel m β) :
  perpendicular_planes α β :=
sorry

end plane_perpendicularity_l1325_132504


namespace notebook_cost_l1325_132519

theorem notebook_cost (notebook_cost pen_cost : ℚ) 
  (total_cost : notebook_cost + pen_cost = 5/2)
  (price_difference : notebook_cost = pen_cost + 2) :
  notebook_cost = 9/4 := by
  sorry

end notebook_cost_l1325_132519


namespace marathon_theorem_l1325_132543

def marathon_problem (total_miles : ℝ) (day1_percent : ℝ) (day3_miles : ℝ) : Prop :=
  let day1_miles := total_miles * day1_percent / 100
  let remaining_miles := total_miles - day1_miles
  let day2_miles := total_miles - day1_miles - day3_miles
  (day2_miles / remaining_miles) * 100 = 50

theorem marathon_theorem : 
  marathon_problem 70 20 28 := by
  sorry

end marathon_theorem_l1325_132543


namespace complex_real_implies_a_eq_neg_one_l1325_132512

theorem complex_real_implies_a_eq_neg_one (a : ℝ) :
  (Complex.I : ℂ) * (a + 1 : ℝ) = (0 : ℂ) → a = -1 := by
  sorry

end complex_real_implies_a_eq_neg_one_l1325_132512


namespace apple_difference_is_twenty_l1325_132515

/-- The number of apples Cecile bought -/
def cecile_apples : ℕ := 15

/-- The total number of apples bought by Diane and Cecile -/
def total_apples : ℕ := 50

/-- The number of apples Diane bought -/
def diane_apples : ℕ := total_apples - cecile_apples

/-- Diane bought more apples than Cecile -/
axiom diane_bought_more : diane_apples > cecile_apples

/-- The difference between the number of apples Diane and Cecile bought -/
def apple_difference : ℕ := diane_apples - cecile_apples

theorem apple_difference_is_twenty : apple_difference = 20 :=
sorry

end apple_difference_is_twenty_l1325_132515


namespace certain_number_proof_l1325_132516

theorem certain_number_proof (x : ℕ) (certain_number : ℕ) : 
  (certain_number = 3 * x + 36) → (x = 4) → (certain_number = 48) := by
  sorry

end certain_number_proof_l1325_132516


namespace complex_equation_solution_l1325_132556

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l1325_132556
