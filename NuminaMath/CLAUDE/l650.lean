import Mathlib

namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l650_65058

def polynomial (x : ℤ) : ℤ := x^3 - 5*x^2 - 8*x + 24

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  (∀ x : ℤ, is_root x ↔ (x = -3 ∨ x = 2 ∨ x = 4)) ∨
  (∀ x : ℤ, ¬is_root x) := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l650_65058


namespace NUMINAMATH_CALUDE_market_qualified_product_probability_l650_65038

theorem market_qualified_product_probability :
  let market_share_A : ℝ := 0.8
  let market_share_B : ℝ := 0.2
  let qualification_rate_A : ℝ := 0.75
  let qualification_rate_B : ℝ := 0.8
  market_share_A * qualification_rate_A + market_share_B * qualification_rate_B = 0.76 :=
by sorry

end NUMINAMATH_CALUDE_market_qualified_product_probability_l650_65038


namespace NUMINAMATH_CALUDE_max_value_constraint_l650_65029

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  x + 2*y + 2*z ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l650_65029


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_l650_65025

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_with_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ 
             digits_product n = 12 ∧
             (∀ (m : ℕ), is_two_digit m → digits_product m = 12 → m ≤ n) ∧
             n = 62 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_l650_65025


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l650_65068

noncomputable section

-- Define the vector space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points of the pentagon and extended points
variable (A B C D E A' B' C' D' E' : V)

-- Define the conditions
variable (h1 : A' - A = A - B)
variable (h2 : B' - B = B - C)
variable (h3 : C' - C = C - D)
variable (h4 : D' - D = D - E)
variable (h5 : E' - E = E - A)

-- State the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (2/31 : ℝ) • B' + (4/31 : ℝ) • C' + (8/31 : ℝ) • D' + (16/31 : ℝ) • E' :=
sorry

end NUMINAMATH_CALUDE_pentagon_reconstruction_l650_65068


namespace NUMINAMATH_CALUDE_price_reduction_rate_l650_65026

theorem price_reduction_rate (original_price final_price : ℝ) 
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  (h3 : ∃ x : ℝ, final_price = original_price * (1 - x)^2) :
  ∃ x : ℝ, final_price = original_price * (1 - x)^2 ∧ x = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_rate_l650_65026


namespace NUMINAMATH_CALUDE_problem_statement_l650_65045

theorem problem_statement (a b m n : ℝ) : 
  a * m^2001 + b * n^2001 = 3 →
  a * m^2002 + b * n^2002 = 7 →
  a * m^2003 + b * n^2003 = 24 →
  a * m^2004 + b * n^2004 = 102 →
  m^2 * (n - 1) = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l650_65045


namespace NUMINAMATH_CALUDE_lizas_peanut_butter_cookies_l650_65062

/-- Given the conditions of Liza's cookie-making scenario, prove that she used 2/5 of the remaining butter for peanut butter cookies. -/
theorem lizas_peanut_butter_cookies (total_butter : ℝ) (remaining_butter : ℝ) (peanut_butter_fraction : ℝ) :
  total_butter = 10 →
  remaining_butter = total_butter / 2 →
  2 = remaining_butter - peanut_butter_fraction * remaining_butter - (1 / 3) * (remaining_butter - peanut_butter_fraction * remaining_butter) →
  peanut_butter_fraction = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lizas_peanut_butter_cookies_l650_65062


namespace NUMINAMATH_CALUDE_polynomial_equality_l650_65049

theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, p x + (2*x^6 + 4*x^4 + 6*x^2) = 8*x^4 + 27*x^3 + 33*x^2 + 15*x + 5) →
  (∀ x, p x = -2*x^6 + 4*x^4 + 27*x^3 + 27*x^2 + 15*x + 5) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l650_65049


namespace NUMINAMATH_CALUDE_frame_purchase_remaining_money_l650_65086

theorem frame_purchase_remaining_money 
  (budget : ℝ) 
  (initial_frame_price_increase : ℝ) 
  (smaller_frame_price_ratio : ℝ) :
  budget = 60 →
  initial_frame_price_increase = 0.2 →
  smaller_frame_price_ratio = 3/4 →
  budget - (budget * (1 + initial_frame_price_increase) * smaller_frame_price_ratio) = 6 := by
  sorry

end NUMINAMATH_CALUDE_frame_purchase_remaining_money_l650_65086


namespace NUMINAMATH_CALUDE_diana_bottle_caps_l650_65047

theorem diana_bottle_caps (initial final eaten : ℕ) : 
  final = 61 → eaten = 4 → initial = final + eaten := by sorry

end NUMINAMATH_CALUDE_diana_bottle_caps_l650_65047


namespace NUMINAMATH_CALUDE_smallest_number_remainder_l650_65067

theorem smallest_number_remainder (n : ℕ) : 
  (n = 197) → 
  (∀ m : ℕ, m < n → m % 13 ≠ 2 ∨ m % 16 ≠ 5) → 
  n % 13 = 2 → 
  n % 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_remainder_l650_65067


namespace NUMINAMATH_CALUDE_spade_calculation_l650_65053

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : (spade 8 5) + (spade 3 (spade 6 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l650_65053


namespace NUMINAMATH_CALUDE_cone_apex_angle_l650_65075

theorem cone_apex_angle (α β : ℝ) : 
  β = Real.arcsin (1/4) →
  2 * α = Real.arcsin (2 * Real.sin β) + β →
  2 * α = π/6 + Real.arcsin (1/4) :=
by sorry

end NUMINAMATH_CALUDE_cone_apex_angle_l650_65075


namespace NUMINAMATH_CALUDE_circle_coloring_exists_l650_65001

/-- A point on a circle --/
structure CirclePoint where
  angle : Real

/-- A color (red or blue) --/
inductive Color
  | Red
  | Blue

/-- A coloring function for points on a circle --/
def ColoringFunction := CirclePoint → Color

/-- Predicate to check if three points form a right-angled triangle inscribed in the circle --/
def IsRightTriangle (p1 p2 p3 : CirclePoint) : Prop :=
  -- We assume this predicate exists and is correctly defined
  sorry

theorem circle_coloring_exists :
  ∃ (f : ColoringFunction),
    ∀ (p1 p2 p3 : CirclePoint),
      IsRightTriangle p1 p2 p3 →
        (f p1 ≠ f p2) ∨ (f p1 ≠ f p3) ∨ (f p2 ≠ f p3) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_coloring_exists_l650_65001


namespace NUMINAMATH_CALUDE_polynomial_expansion_l650_65036

theorem polynomial_expansion (x : ℝ) : 
  (7 * x^2 + 5 - 3 * x) * (4 * x^3) = 28 * x^5 - 12 * x^4 + 20 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l650_65036


namespace NUMINAMATH_CALUDE_percentage_calculation_l650_65003

theorem percentage_calculation (P : ℝ) : 
  (0.16 * (P / 100) * 93.75 = 6) → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l650_65003


namespace NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l650_65017

theorem max_perfect_squares_pairwise_products (a b : ℕ) (h : a ≠ b) :
  let products := {a * (a + 2), b * (b + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y : ℕ, x = y ^ 2) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y : ℕ, x = y ^ 2) → s.card ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l650_65017


namespace NUMINAMATH_CALUDE_polygon_with_150_degree_angles_is_12_gon_l650_65005

theorem polygon_with_150_degree_angles_is_12_gon (n : ℕ) 
  (h : n ≥ 3) 
  (interior_angle : ℝ) 
  (h_angle : interior_angle = 150) 
  (h_sum : (n - 2) * 180 = n * interior_angle) : n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_150_degree_angles_is_12_gon_l650_65005


namespace NUMINAMATH_CALUDE_connected_vertices_probability_is_correct_l650_65087

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of choosing at least two connected vertices when selecting three random vertices -/
def connected_vertices_probability (d : RegularDodecahedron) : ℚ :=
  9 / 19

/-- Theorem stating the probability of choosing at least two connected vertices -/
theorem connected_vertices_probability_is_correct (d : RegularDodecahedron) :
  connected_vertices_probability d = 9 / 19 := by
  sorry


end NUMINAMATH_CALUDE_connected_vertices_probability_is_correct_l650_65087


namespace NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l650_65027

/-- Given a line segment AB extended to point P such that AP:PB = 7:5,
    prove that the position vector of P can be expressed as 
    P = (5/12)A + (7/12)B -/
theorem extended_line_segment_vector_representation 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h : (dist A P) / (dist P B) = 7 / 5) : -- AP:PB = 7:5
  ∃ (t u : ℝ), t = 5/12 ∧ u = 7/12 ∧ P = t • A + u • B :=
by sorry

end NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l650_65027


namespace NUMINAMATH_CALUDE_coin_flip_configurations_l650_65065

theorem coin_flip_configurations (n : ℕ) (h : n = 10) : 
  (Finset.range n).card + (n.choose 2) = 46 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_configurations_l650_65065


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l650_65057

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_3402 :
  largest_perfect_square_factor 3402 = 81 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l650_65057


namespace NUMINAMATH_CALUDE_pairwise_product_inequality_l650_65097

theorem pairwise_product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * Real.rpow (x^3 + y^3 + z^3) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_product_inequality_l650_65097


namespace NUMINAMATH_CALUDE_base_six_conversion_and_addition_l650_65095

def base_six_to_ten (n : Nat) : Nat :=
  (n % 10) + 6 * ((n / 10) % 10) + 36 * (n / 100)

theorem base_six_conversion_and_addition :
  let base_six_num := 214
  let base_ten_num := base_six_to_ten base_six_num
  base_ten_num = 82 ∧ base_ten_num + 15 = 97 := by
  sorry

end NUMINAMATH_CALUDE_base_six_conversion_and_addition_l650_65095


namespace NUMINAMATH_CALUDE_remainder_proof_l650_65033

theorem remainder_proof : ∃ r : ℕ, r < 33 ∧ r < 8 ∧ 266 % 33 = r ∧ 266 % 8 = r :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l650_65033


namespace NUMINAMATH_CALUDE_parallel_planes_from_skew_parallel_lines_l650_65044

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_skew_parallel_lines 
  (m n : Line) (α β : Plane) :
  skew m n →
  parallel_line_plane m α →
  parallel_line_plane n α →
  parallel_line_plane m β →
  parallel_line_plane n β →
  parallel_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_skew_parallel_lines_l650_65044


namespace NUMINAMATH_CALUDE_forgotten_digit_probability_l650_65061

theorem forgotten_digit_probability : 
  let total_digits : ℕ := 10
  let max_attempts : ℕ := 2
  let favorable_outcomes : ℕ := (total_digits - 1) + (total_digits - 1)
  let total_outcomes : ℕ := total_digits * (total_digits - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_digit_probability_l650_65061


namespace NUMINAMATH_CALUDE_max_value_quarter_l650_65048

def f (a b : ℕ) : ℚ := (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_quarter (a b : ℕ) (ha : 2 ≤ a ∧ a ≤ 8) (hb : 2 ≤ b ∧ b ≤ 8) :
  f a b ≤ 1/4 := by
  sorry

#eval f 2 2  -- To check the function definition

end NUMINAMATH_CALUDE_max_value_quarter_l650_65048


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l650_65085

theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side : ℝ) (rect_width : ℝ),
    pentagon_side * 5 = 60 →
    rect_width * 6 = 40 →
    pentagon_side / rect_width = 9 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l650_65085


namespace NUMINAMATH_CALUDE_corn_acreage_l650_65077

/-- Given a total of 1034 acres of land divided among beans, wheat, and corn
    in the ratio of 5:2:4, prove that the number of acres used for corn is 376. -/
theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
    (h1 : total_land = 1034)
    (h2 : beans_ratio = 5)
    (h3 : wheat_ratio = 2)
    (h4 : corn_ratio = 4) :
    (corn_ratio * total_land) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry


end NUMINAMATH_CALUDE_corn_acreage_l650_65077


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l650_65050

/-- A quadrilateral with coordinates of its four vertices -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.A.1 - q.C.1, q.A.2 - q.C.2), (q.B.1 - q.D.1, q.B.2 - q.D.2))

/-- Check if the diagonals of a quadrilateral are equal -/
def equal_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  d1.1^2 + d1.2^2 = d2.1^2 + d2.2^2

/-- Check if the diagonals of a quadrilateral bisect each other -/
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let mid1 := ((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2)
  let mid2 := ((q.B.1 + q.D.1) / 2, (q.B.2 + q.D.2) / 2)
  mid1 = mid2

/-- Check if the diagonals of a quadrilateral are perpendicular -/
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  d1.1 * d2.1 + d1.2 * d2.2 = 0

/-- Check if all sides of a quadrilateral are equal -/
def equal_sides (q : Quadrilateral) : Prop :=
  let side1 := (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2
  let side2 := (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2
  let side3 := (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2
  let side4 := (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2
  side1 = side2 ∧ side2 = side3 ∧ side3 = side4

/-- A quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ diagonals_bisect q

/-- A quadrilateral is a rhombus -/
def is_rhombus (q : Quadrilateral) : Prop :=
  equal_sides q

/-- A quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ perpendicular_diagonals q

theorem quadrilateral_properties :
  (∀ q : Quadrilateral, equal_diagonals q ∧ diagonals_bisect q → is_rectangle q) ∧
  ¬(∀ q : Quadrilateral, perpendicular_diagonals q → is_rhombus q) ∧
  (∀ q : Quadrilateral, equal_diagonals q ∧ perpendicular_diagonals q → is_square q) ∧
  (∀ q : Quadrilateral, equal_sides q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l650_65050


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l650_65016

theorem triangle_third_side_length : ∀ (x : ℝ),
  (x > 0 ∧ 5 + 9 > x ∧ x + 5 > 9 ∧ x + 9 > 5) → x = 8 ∨ (x < 8 ∨ x > 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l650_65016


namespace NUMINAMATH_CALUDE_store_customer_ratio_l650_65098

theorem store_customer_ratio : 
  let non_holiday_rate : ℚ := 175  -- customers per hour during non-holiday season
  let holiday_total : ℕ := 2800    -- total customers during holiday season
  let holiday_hours : ℕ := 8       -- number of hours during holiday season
  let holiday_rate : ℚ := holiday_total / holiday_hours  -- customers per hour during holiday season
  holiday_rate / non_holiday_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_store_customer_ratio_l650_65098


namespace NUMINAMATH_CALUDE_total_books_in_class_l650_65021

theorem total_books_in_class (num_tables : ℕ) (books_per_table_ratio : ℚ) : 
  num_tables = 500 →
  books_per_table_ratio = 2 / 5 →
  (num_tables : ℚ) * books_per_table_ratio * num_tables = 100000 :=
by sorry

end NUMINAMATH_CALUDE_total_books_in_class_l650_65021


namespace NUMINAMATH_CALUDE_problem_solution_l650_65066

theorem problem_solution : 
  let tan60 := Real.sqrt 3
  |Real.sqrt 2 - Real.sqrt 3| - tan60 + 1 / Real.sqrt 2 = -(Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l650_65066


namespace NUMINAMATH_CALUDE_chord_arrangement_count_l650_65072

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to connect 4 points with 3 chords such that each chord intersects the other two. -/
def fourPointConnections : ℕ := 8

/-- The number of ways to connect 5 points with 3 chords such that exactly two chords share a common endpoint and the remaining chord intersects these two. -/
def fivePointConnections : ℕ := 5

/-- The total number of ways to arrange three intersecting chords among 20 points on a circle. -/
def totalChordArrangements : ℕ := 
  choose 20 3 + choose 20 4 * fourPointConnections + 
  choose 20 5 * fivePointConnections + choose 20 6

theorem chord_arrangement_count : totalChordArrangements = 156180 := by
  sorry

end NUMINAMATH_CALUDE_chord_arrangement_count_l650_65072


namespace NUMINAMATH_CALUDE_friday_to_thursday_ratio_is_two_to_one_l650_65040

/-- Represents the daily sales of ground beef in kilograms -/
structure DailySales where
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Theorem stating the ratio of Friday to Thursday sales is 2:1 -/
theorem friday_to_thursday_ratio_is_two_to_one (sales : DailySales) : 
  sales.thursday = 210 →
  sales.saturday = 130 →
  sales.sunday = sales.saturday / 2 →
  sales.thursday + sales.friday + sales.saturday + sales.sunday = 825 →
  sales.friday / sales.thursday = 2 := by
  sorry

#check friday_to_thursday_ratio_is_two_to_one

end NUMINAMATH_CALUDE_friday_to_thursday_ratio_is_two_to_one_l650_65040


namespace NUMINAMATH_CALUDE_probability_mathematics_letter_l650_65074

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_mathematics_letter : 
  (unique_letters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_mathematics_letter_l650_65074


namespace NUMINAMATH_CALUDE_inequality_range_real_inequality_range_unit_interval_l650_65020

-- Define the inequality function
def inequality (k x : ℝ) : Prop :=
  (k * x^2 + k * x + 4) / (x^2 + x + 1) > 1

-- Theorem for the first part of the problem
theorem inequality_range_real : 
  (∀ x : ℝ, inequality k x) ↔ k ∈ Set.Icc 1 13 := by sorry

-- Theorem for the second part of the problem
theorem inequality_range_unit_interval :
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → inequality k x) ↔ k ∈ Set.Ioi (-1/2) := by sorry

end NUMINAMATH_CALUDE_inequality_range_real_inequality_range_unit_interval_l650_65020


namespace NUMINAMATH_CALUDE_right_angled_projection_l650_65060

structure Plane where
  α : Type

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def Triangle (A B C : Point) := True

def RightAngledTriangle (A B C : Point) := Triangle A B C

def IsInPlane (p : Point) (α : Plane) : Prop := sorry

def IsOutsidePlane (p : Point) (α : Plane) : Prop := sorry

def Projection (p : Point) (α : Plane) : Point := sorry

def IsOn (p : Point) (A B : Point) : Prop := sorry

theorem right_angled_projection 
  (α : Plane) (A B C C1 : Point) : 
  RightAngledTriangle A B C →
  IsInPlane A α →
  IsInPlane B α →
  IsOutsidePlane C α →
  C1 = Projection C α →
  ¬IsOn C1 A B →
  RightAngledTriangle A B C1 := by sorry

end NUMINAMATH_CALUDE_right_angled_projection_l650_65060


namespace NUMINAMATH_CALUDE_max_ab_value_l650_65073

def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : ∃ (ε : ℝ), ε ≠ 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → 
    f a b x ≤ f a b 1 ∨ f a b x ≥ f a b 1) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∃ (ε : ℝ), ε ≠ 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → 
      f a' b' x ≤ f a' b' 1 ∨ f a' b' x ≥ f a' b' 1) → 
    a' * b' ≤ a * b) →
  a * b = 9 := by sorry

end NUMINAMATH_CALUDE_max_ab_value_l650_65073


namespace NUMINAMATH_CALUDE_vector_computation_l650_65037

theorem vector_computation : 
  (4 : ℝ) • (![2, -9] : Fin 2 → ℝ) - (3 : ℝ) • (![(-1), -6] : Fin 2 → ℝ) = ![11, -18] :=
by sorry

end NUMINAMATH_CALUDE_vector_computation_l650_65037


namespace NUMINAMATH_CALUDE_smallest_piece_length_l650_65012

/-- Given a rod of length 120 cm cut into three pieces proportional to 3, 5, and 7,
    the length of the smallest piece is 24 cm. -/
theorem smallest_piece_length :
  let total_length : ℝ := 120
  let ratio_sum : ℝ := 3 + 5 + 7
  let smallest_ratio : ℝ := 3
  smallest_ratio * (total_length / ratio_sum) = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_piece_length_l650_65012


namespace NUMINAMATH_CALUDE_prob_three_students_same_group_l650_65052

/-- The probability that three specific students are assigned to the same group
    when 800 students are randomly assigned to 4 equal-sized groups -/
theorem prob_three_students_same_group :
  let total_students : ℕ := 800
  let num_groups : ℕ := 4
  let group_size : ℕ := total_students / num_groups
  -- Assuming each group has equal size
  (∀ g : Fin num_groups, (group_size : ℚ) = total_students / num_groups)
  →
  (probability_same_group : ℚ) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_students_same_group_l650_65052


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l650_65004

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l650_65004


namespace NUMINAMATH_CALUDE_movie_expense_ratio_l650_65088

/-- Proves the ratio of movie expenses to weekly allowance -/
theorem movie_expense_ratio (weekly_allowance : ℚ) (car_wash_earning : ℚ) (final_amount : ℚ) :
  weekly_allowance = 10 →
  car_wash_earning = 6 →
  final_amount = 11 →
  (weekly_allowance - (final_amount - car_wash_earning)) / weekly_allowance = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_movie_expense_ratio_l650_65088


namespace NUMINAMATH_CALUDE_garden_ratio_l650_65035

/-- A rectangular garden with given perimeter and length has a specific length-to-width ratio -/
theorem garden_ratio (perimeter length width : ℝ) : 
  perimeter = 300 →
  length = 100 →
  perimeter = 2 * length + 2 * width →
  length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l650_65035


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l650_65091

theorem cookie_jar_problem (x : ℕ) : 
  (x - 1 = (x + 5) / 2) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l650_65091


namespace NUMINAMATH_CALUDE_philips_banana_groups_l650_65063

theorem philips_banana_groups :
  let total_bananas : ℕ := 392
  let bananas_per_group : ℕ := 2
  let num_groups : ℕ := total_bananas / bananas_per_group
  num_groups = 196 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_groups_l650_65063


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l650_65056

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 1 + a 2 + a 3 = 1)
  (h3 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 9) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l650_65056


namespace NUMINAMATH_CALUDE_min_value_theorem_l650_65010

/-- A geometric sequence with positive terms satisfying the given conditions -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n) ∧
  a 3 = a 2 + 2 * a 1

/-- The existence of terms satisfying the product condition -/
def ExistTerms (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m * a n = 64 * (a 1)^2

/-- The theorem statement -/
theorem min_value_theorem (a : ℕ → ℝ) 
  (h1 : GeometricSequence a) 
  (h2 : ExistTerms a) : 
  ∀ m n : ℕ, a m * a n = 64 * (a 1)^2 → 1 / m + 9 / n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l650_65010


namespace NUMINAMATH_CALUDE_bills_score_l650_65093

theorem bills_score (john sue bill : ℕ) 
  (h1 : bill = john + 20)
  (h2 : bill * 2 = sue)
  (h3 : john + bill + sue = 160) :
  bill = 45 := by
sorry

end NUMINAMATH_CALUDE_bills_score_l650_65093


namespace NUMINAMATH_CALUDE_odd_function_zero_value_l650_65064

/-- A function f is odd if f(-x) = -f(x) for all x in ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- For any odd function f defined on ℝ, f(0) = 0 -/
theorem odd_function_zero_value (f : ℝ → ℝ) (h : IsOdd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_l650_65064


namespace NUMINAMATH_CALUDE_cattle_train_speed_l650_65054

/-- The speed of the cattle train in mph -/
def cattle_speed : ℝ := 56

/-- The time difference in hours between the cattle train's departure and the diesel train's departure -/
def time_difference : ℝ := 6

/-- The duration in hours that the diesel train traveled -/
def diesel_travel_time : ℝ := 12

/-- The speed difference in mph between the cattle train and the diesel train -/
def speed_difference : ℝ := 33

/-- The total distance in miles between the two trains after the diesel train's travel -/
def total_distance : ℝ := 1284

theorem cattle_train_speed :
  cattle_speed * (time_difference + diesel_travel_time) +
  (cattle_speed - speed_difference) * diesel_travel_time = total_distance :=
sorry

end NUMINAMATH_CALUDE_cattle_train_speed_l650_65054


namespace NUMINAMATH_CALUDE_product_equals_three_halves_l650_65080

theorem product_equals_three_halves : 12 * 0.5 * 4 * 0.0625 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_three_halves_l650_65080


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l650_65028

theorem fraction_zero_implies_a_equals_two (a : ℝ) : 
  (a^2 - 4) / (a + 2) = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l650_65028


namespace NUMINAMATH_CALUDE_inequality_solution_set_l650_65083

theorem inequality_solution_set (x : ℝ) :
  (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l650_65083


namespace NUMINAMATH_CALUDE_function_properties_l650_65024

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  is_odd f ∧ has_period f 40 := by sorry

end NUMINAMATH_CALUDE_function_properties_l650_65024


namespace NUMINAMATH_CALUDE_ellipse_x_axis_iff_l650_65039

/-- Defines an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (k : ℝ) : Prop :=
  0 < k ∧ k < 2 ∧ ∀ x y : ℝ, x^2 / 2 + y^2 / k = 1 → 
    ∃ c : ℝ, c > 0 ∧ c < 1 ∧
      ∀ p : ℝ × ℝ, (p.1 - c)^2 + p.2^2 + (p.1 + c)^2 + p.2^2 = 2

/-- The condition 0 < k < 2 is necessary and sufficient for the equation
    x^2/2 + y^2/k = 1 to represent an ellipse with foci on the x-axis -/
theorem ellipse_x_axis_iff (k : ℝ) : is_ellipse_x_axis k ↔ (0 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_x_axis_iff_l650_65039


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l650_65014

theorem oliver_candy_boxes (initial_boxes final_boxes : ℕ) : 
  initial_boxes = 8 → final_boxes = 6 → initial_boxes + final_boxes = 14 :=
by sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l650_65014


namespace NUMINAMATH_CALUDE_smallest_possible_S_l650_65015

/-- The number of faces on each die -/
def num_faces : ℕ := 8

/-- The target sum we're comparing to -/
def target_sum : ℕ := 3000

/-- The function to calculate the smallest possible value of S -/
def smallest_S (n : ℕ) : ℕ := 9 * n - target_sum

/-- The theorem stating the smallest possible value of S -/
theorem smallest_possible_S :
  ∃ (n : ℕ), 
    (n * num_faces ≥ target_sum) ∧ 
    (∀ m : ℕ, m < n → m * num_faces < target_sum) ∧
    (smallest_S n = 375) := by
  sorry

#check smallest_possible_S

end NUMINAMATH_CALUDE_smallest_possible_S_l650_65015


namespace NUMINAMATH_CALUDE_marble_count_l650_65000

theorem marble_count (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r := by sorry

end NUMINAMATH_CALUDE_marble_count_l650_65000


namespace NUMINAMATH_CALUDE_term_degree_le_poly_degree_l650_65002

/-- A polynomial of degree 6 -/
def Polynomial6 : Type := ℕ → ℚ

/-- The degree of a polynomial -/
def degree (p : Polynomial6) : ℕ := 6

/-- A term of a polynomial -/
def Term : Type := ℕ × ℚ

/-- The degree of a term -/
def termDegree (t : Term) : ℕ := t.1

theorem term_degree_le_poly_degree (p : Polynomial6) (t : Term) : 
  termDegree t ≤ degree p := by sorry

end NUMINAMATH_CALUDE_term_degree_le_poly_degree_l650_65002


namespace NUMINAMATH_CALUDE_odd_function_property_l650_65084

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x - 3) = f (x + 2))
  (h_value : f 1 = 2) :
  f 2011 - f 2010 = 2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l650_65084


namespace NUMINAMATH_CALUDE_max_value_constraint_l650_65078

theorem max_value_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 6 * b < 110) :
  a * b * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l650_65078


namespace NUMINAMATH_CALUDE_functional_equation_solution_l650_65046

-- Define the function type
def ContinuousFunction (α : Type*) := α → ℝ

-- State the theorem
theorem functional_equation_solution
  (f : ContinuousFunction ℝ)
  (h_cont : Continuous f)
  (h_domain : ∀ x : ℝ, x > 0 → f x ≠ 0)
  (h_eq : ∀ x y : ℝ, x > 0 → y > 0 →
    f (x + 1/x) + f (y + 1/y) = f (x + 1/y) + f (y + 1/x)) :
  ∃ c d : ℝ, ∀ x : ℝ, x > 0 → f x = c * x + d :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l650_65046


namespace NUMINAMATH_CALUDE_greater_number_proof_l650_65032

theorem greater_number_proof (a b : ℝ) (h_sum : a + b = 40) (h_diff : a - b = 2) (h_greater : a > b) : a = 21 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l650_65032


namespace NUMINAMATH_CALUDE_quadratic_roots_sixth_power_sum_l650_65070

theorem quadratic_roots_sixth_power_sum (r s : ℝ) : 
  r^2 - 3 * r * Real.sqrt 2 + 4 = 0 ∧ 
  s^2 - 3 * s * Real.sqrt 2 + 4 = 0 → 
  r^6 + s^6 = 648 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sixth_power_sum_l650_65070


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l650_65008

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem stating that f is monotonically decreasing on (-∞, 1]
theorem f_monotone_decreasing :
  MonotoneOn f (Set.Iic 1) := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l650_65008


namespace NUMINAMATH_CALUDE_point_equidistant_from_origin_and_A_l650_65006

/-- Given a point P(x, y) that is 17 units away from both the origin O(0,0) and point A(16,0),
    prove that the coordinates of P must be either (8, 15) or (8, -15). -/
theorem point_equidistant_from_origin_and_A : ∀ x y : ℝ,
  (x^2 + y^2 = 17^2) →
  ((x - 16)^2 + y^2 = 17^2) →
  ((x = 8 ∧ y = 15) ∨ (x = 8 ∧ y = -15)) :=
by sorry

end NUMINAMATH_CALUDE_point_equidistant_from_origin_and_A_l650_65006


namespace NUMINAMATH_CALUDE_wanda_eating_theorem_l650_65031

/-- Pascal's triangle up to row n -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Check if a number is odd -/
def isOdd (n : ℕ) : Bool :=
  sorry

/-- Count odd numbers in Pascal's triangle up to row n -/
def countOddNumbers (triangle : List (List ℕ)) : ℕ :=
  sorry

/-- Check if a path in Pascal's triangle satisfies the no-sum condition -/
def validPath (path : List ℕ) : Bool :=
  sorry

/-- Main theorem -/
theorem wanda_eating_theorem :
  ∃ (path : List ℕ), 
    (path.length > 100000) ∧ 
    (∀ n ∈ path, n ∈ (PascalTriangle 2011).join) ∧
    (∀ n ∈ path, isOdd n) ∧
    validPath path :=
  sorry

end NUMINAMATH_CALUDE_wanda_eating_theorem_l650_65031


namespace NUMINAMATH_CALUDE_parabola_transformation_l650_65007

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ

/-- Shifts a parabola vertically -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { equation := λ x => p.equation x + shift }

/-- Shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { equation := λ x => p.equation (x - shift) }

theorem parabola_transformation (p : Parabola) (h : p.equation = λ x => 2 * x^2) :
  (horizontal_shift (vertical_shift p 3) 1).equation = λ x => 2 * (x - 1)^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l650_65007


namespace NUMINAMATH_CALUDE_multiplier_can_be_greater_than_one_l650_65055

theorem multiplier_can_be_greater_than_one (a b : ℚ) (h : a * b ≤ b) : 
  ∃ (a : ℚ), a * b ≤ b ∧ a > 1 :=
sorry

end NUMINAMATH_CALUDE_multiplier_can_be_greater_than_one_l650_65055


namespace NUMINAMATH_CALUDE_store_sales_l650_65081

theorem store_sales (dvd_count : ℕ) (dvd_cd_ratio : ℚ) : 
  dvd_count = 168 → dvd_cd_ratio = 1.6 → dvd_count + (dvd_count / dvd_cd_ratio).floor = 273 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_l650_65081


namespace NUMINAMATH_CALUDE_johnson_family_seating_l650_65051

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem johnson_family_seating (num_boys num_girls : ℕ) 
  (h1 : num_boys = 5) 
  (h2 : num_girls = 4) 
  (h3 : num_boys + num_girls = 9) : 
  factorial (num_boys + num_girls) - factorial num_boys * factorial num_girls = 359760 :=
sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l650_65051


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l650_65034

def M : Set ℝ := {x : ℝ | x^2 - 3*x - 4 ≥ 0}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

theorem intersection_complement_M_and_N :
  (Mᶜ ∩ N) = {x : ℝ | 1 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l650_65034


namespace NUMINAMATH_CALUDE_calculate_expression_l650_65059

theorem calculate_expression : -2^3 / (-2) + (-2)^2 * (-5) = -16 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l650_65059


namespace NUMINAMATH_CALUDE_fraction_equality_l650_65079

theorem fraction_equality : (4 + 14) / (7 + 14) = 6 / 7 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l650_65079


namespace NUMINAMATH_CALUDE_middle_seat_is_A_l650_65076

/-- Represents the position of a person in the train -/
inductive Position
| first
| second
| third
| fourth
| fifth

/-- Represents a person -/
inductive Person
| A
| B
| C
| D
| E

/-- The seating arrangement in the train -/
def SeatingArrangement := Person → Position

theorem middle_seat_is_A (arrangement : SeatingArrangement) : 
  (arrangement Person.D = Position.fifth) →
  (arrangement Person.A = Position.fourth ∧ arrangement Person.E = Position.second) ∨
  (arrangement Person.A = Position.third ∧ arrangement Person.E = Position.second) →
  (arrangement Person.B = Position.first ∨ arrangement Person.B = Position.second) →
  (arrangement Person.B ≠ arrangement Person.C ∧ 
   arrangement Person.A ≠ arrangement Person.C ∧
   arrangement Person.E ≠ arrangement Person.C) →
  arrangement Person.A = Position.third :=
by sorry

end NUMINAMATH_CALUDE_middle_seat_is_A_l650_65076


namespace NUMINAMATH_CALUDE_hike_length_is_four_l650_65071

/-- Represents the hike details -/
structure Hike where
  initial_water : ℝ
  duration : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_part_consumption_rate : ℝ

/-- Calculates the length of the hike in miles -/
def hike_length (h : Hike) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, the hike length is 4 miles -/
theorem hike_length_is_four (h : Hike) 
  (h_initial : h.initial_water = 10)
  (h_duration : h.duration = 2)
  (h_remaining : h.remaining_water = 2)
  (h_leak : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 3)
  (h_first_part : h.first_part_consumption_rate = 1) :
  hike_length h = 4 := by
  sorry

end NUMINAMATH_CALUDE_hike_length_is_four_l650_65071


namespace NUMINAMATH_CALUDE_total_digits_in_books_l650_65092

/-- Calculate the number of digits used to number pages in a book -/
def digitsInBook (pages : ℕ) : ℕ :=
  let singleDigitPages := min pages 9
  let doubleDigitPages := min (pages - 9) 90
  let tripleDigitPages := min (pages - 99) 900
  let quadrupleDigitPages := max (pages - 999) 0
  singleDigitPages * 1 +
  doubleDigitPages * 2 +
  tripleDigitPages * 3 +
  quadrupleDigitPages * 4

/-- The total number of digits used to number pages in the collection of books -/
def totalDigits : ℕ :=
  digitsInBook 450 + digitsInBook 675 + digitsInBook 1125 + digitsInBook 2430

theorem total_digits_in_books :
  totalDigits = 15039 := by sorry

end NUMINAMATH_CALUDE_total_digits_in_books_l650_65092


namespace NUMINAMATH_CALUDE_max_market_women_eight_market_women_l650_65030

def farthings_in_2s_2_1_4d : ℕ := 105

theorem max_market_women (n : ℕ) : n ∣ farthings_in_2s_2_1_4d → n ≤ 8 :=
sorry

theorem eight_market_women : ∃ (s : Finset ℕ), s.card = 8 ∧ ∀ n ∈ s, n ∣ farthings_in_2s_2_1_4d :=
sorry

end NUMINAMATH_CALUDE_max_market_women_eight_market_women_l650_65030


namespace NUMINAMATH_CALUDE_higher_interest_rate_theorem_l650_65043

/-- Given a principal amount, two interest rates, and a time period,
    calculate the difference in interest earned between the two rates. -/
def interest_difference (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  principal * rate1 * time - principal * rate2 * time

theorem higher_interest_rate_theorem (R : ℝ) :
  interest_difference 5000 (R / 100) (12 / 100) 2 = 600 → R = 18 := by
  sorry

end NUMINAMATH_CALUDE_higher_interest_rate_theorem_l650_65043


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l650_65018

theorem sum_of_coefficients (g h i j k : ℤ) : 
  (∀ y : ℝ, 1000 * y^3 + 27 = (g * y + h) * (i * y^2 + j * y + k)) →
  g + h + i + j + k = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l650_65018


namespace NUMINAMATH_CALUDE_total_amount_spent_l650_65090

theorem total_amount_spent (num_pens num_pencils : ℕ) 
                           (avg_pen_price avg_pencil_price : ℚ) : 
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 14 →
  avg_pencil_price = 2 →
  (num_pens : ℚ) * avg_pen_price + (num_pencils : ℚ) * avg_pencil_price = 570 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l650_65090


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l650_65041

def polynomial (x : ℝ) : ℝ := x^4 - 6*x^3 + 9*x^2 + 18*x - 24

theorem sum_of_absolute_roots : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x : ℝ, polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    |r₁| + |r₂| + |r₃| + |r₄| = 6 + 2 * Real.sqrt 6 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l650_65041


namespace NUMINAMATH_CALUDE_limit_of_a_sequence_l650_65022

def a (n : ℕ) : ℚ := n / (3 * n - 1)

theorem limit_of_a_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_a_sequence_l650_65022


namespace NUMINAMATH_CALUDE_unique_pair_sum_28_l650_65019

theorem unique_pair_sum_28 :
  ∃! (a b : ℕ), a ≠ b ∧ a > 11 ∧ b > 11 ∧ a + b = 28 ∧
  (Even a ∨ Even b) ∧
  (∀ (c d : ℕ), c ≠ d ∧ c > 11 ∧ d > 11 ∧ c + d = 28 ∧ (Even c ∨ Even d) → (c = a ∧ d = b) ∨ (c = b ∧ d = a)) ∧
  a = 12 ∧ b = 16 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_sum_28_l650_65019


namespace NUMINAMATH_CALUDE_find_number_l650_65096

theorem find_number : ∃! x : ℝ, 0.8 * x + 20 = x := by
  sorry

end NUMINAMATH_CALUDE_find_number_l650_65096


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l650_65094

def p (x : ℝ) : Prop := |x - 4| > 2

def q (x : ℝ) : Prop := x > 1

def not_p (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 6

theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, not_p x → q x) ∧ ¬(∀ x, q x → not_p x) := by sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l650_65094


namespace NUMINAMATH_CALUDE_sugar_consumption_calculation_l650_65069

/-- Given a price increase and consumption change, calculate the initial consumption -/
theorem sugar_consumption_calculation 
  (price_increase : ℝ) 
  (expenditure_increase : ℝ) 
  (new_consumption : ℝ) 
  (h1 : price_increase = 0.32)
  (h2 : expenditure_increase = 0.10)
  (h3 : new_consumption = 25) :
  ∃ (initial_consumption : ℝ), 
    initial_consumption = 75 ∧ 
    (1 + price_increase) * new_consumption = (1 + expenditure_increase) * initial_consumption :=
by sorry

end NUMINAMATH_CALUDE_sugar_consumption_calculation_l650_65069


namespace NUMINAMATH_CALUDE_leftmost_digit_in_base9_is_5_l650_65023

/-- Represents a number in base-3 as a list of digits -/
def Base3Number := List Nat

/-- Converts a base-3 number to its decimal (base-10) representation -/
def toDecimal (n : Base3Number) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its base-9 representation -/
def toBase9 (n : Nat) : List Nat :=
  sorry

/-- Gets the leftmost digit of a list of digits -/
def leftmostDigit (digits : List Nat) : Nat :=
  digits.head!

/-- The given base-3 number -/
def givenNumber : Base3Number :=
  [1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2]

theorem leftmost_digit_in_base9_is_5 :
  leftmostDigit (toBase9 (toDecimal givenNumber)) = 5 :=
sorry

end NUMINAMATH_CALUDE_leftmost_digit_in_base9_is_5_l650_65023


namespace NUMINAMATH_CALUDE_functions_are_odd_l650_65013

-- Define the property for functions f and g
def has_property (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (g x) = g (f x) ∧ f (g x) = -x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem functions_are_odd (f g : ℝ → ℝ) (h : has_property f g) :
  is_odd f ∧ is_odd g :=
sorry

end NUMINAMATH_CALUDE_functions_are_odd_l650_65013


namespace NUMINAMATH_CALUDE_custom_op_example_l650_65099

-- Define the custom operation
def custom_op (A B : ℕ) : ℕ := (A + 3) * (B - 2)

-- State the theorem
theorem custom_op_example : custom_op 12 17 = 225 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l650_65099


namespace NUMINAMATH_CALUDE_eden_initial_bears_count_l650_65082

/-- Represents the number of stuffed bears Eden had initially --/
def eden_initial_bears : ℕ := 10

/-- Represents the total number of stuffed bears Daragh had initially --/
def daragh_initial_bears : ℕ := 20

/-- Represents the number of favorite bears Daragh took out --/
def daragh_favorite_bears : ℕ := 8

/-- Represents the number of Daragh's sisters --/
def number_of_sisters : ℕ := 3

/-- Represents the number of stuffed bears Eden has now --/
def eden_current_bears : ℕ := 14

theorem eden_initial_bears_count :
  eden_initial_bears =
    eden_current_bears -
    ((daragh_initial_bears - daragh_favorite_bears) / number_of_sisters) :=
by
  sorry

#eval eden_initial_bears

end NUMINAMATH_CALUDE_eden_initial_bears_count_l650_65082


namespace NUMINAMATH_CALUDE_fraction_multiplication_l650_65011

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l650_65011


namespace NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l650_65009

/-- Represents a stratified sampling setup -/
structure StratifiedSampling where
  population : Type
  strata : Type
  num_layers : ℕ
  stratification : population → strata

/-- The probability of an individual being sampled in stratified sampling -/
def sample_probability (ss : StratifiedSampling) (individual : ss.population) : ℝ :=
  sorry

/-- Theorem stating that the sample probability is independent of the number of layers and stratification -/
theorem stratified_sampling_equal_probability (ss : StratifiedSampling) 
  (individual1 individual2 : ss.population) :
  sample_probability ss individual1 = sample_probability ss individual2 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l650_65009


namespace NUMINAMATH_CALUDE_square_fencing_cost_l650_65089

/-- The cost of fencing one side of a square -/
def cost_per_side : ℕ := 69

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing a square -/
def total_cost : ℕ := cost_per_side * num_sides

theorem square_fencing_cost : total_cost = 276 := by
  sorry

end NUMINAMATH_CALUDE_square_fencing_cost_l650_65089


namespace NUMINAMATH_CALUDE_prime_divisor_property_l650_65042

theorem prime_divisor_property (n k : ℕ) (h1 : n > 1) 
  (h2 : ∀ d : ℕ, d ∣ n → (d + k) ∣ n ∨ (d - k) ∣ n) : 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_property_l650_65042
