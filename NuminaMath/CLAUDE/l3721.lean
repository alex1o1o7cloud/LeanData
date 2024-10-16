import Mathlib

namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l3721_372164

/-- A parabola with equation y = x^2 + 4 -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- A hyperbola with equation y^2 - mx^2 = 1, where m is a parameter -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Two curves are tangent if they intersect at exactly one point -/
def are_tangent (curve1 curve2 : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2

theorem parabola_hyperbola_tangency (m : ℝ) :
  are_tangent (parabola) (hyperbola m) → m = 8 + 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l3721_372164


namespace NUMINAMATH_CALUDE_solve_digit_equation_l3721_372151

theorem solve_digit_equation (a b d v t r : ℕ) : 
  a + b = v →
  v + d = t →
  t + a = r →
  b + d + r = 18 →
  1 ≤ a ∧ a ≤ 9 →
  1 ≤ b ∧ b ≤ 9 →
  1 ≤ d ∧ d ≤ 9 →
  1 ≤ v ∧ v ≤ 9 →
  1 ≤ t ∧ t ≤ 9 →
  1 ≤ r ∧ r ≤ 9 →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_solve_digit_equation_l3721_372151


namespace NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l3721_372138

theorem interior_angle_sum_regular_polygon (n : ℕ) (h : n > 2) :
  let exterior_angle : ℝ := 20
  let interior_angle_sum : ℝ := (n - 2) * 180
  (360 / exterior_angle = n) →
  interior_angle_sum = 2880 :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l3721_372138


namespace NUMINAMATH_CALUDE_remainder_calculation_l3721_372158

theorem remainder_calculation (a b r : ℕ) 
  (h1 : a - b = 1200)
  (h2 : a = 1495)
  (h3 : a = 5 * b + r)
  (h4 : r < b) : 
  r = 20 := by
sorry

end NUMINAMATH_CALUDE_remainder_calculation_l3721_372158


namespace NUMINAMATH_CALUDE_equal_area_polygons_equidecomposable_l3721_372140

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define the concept of area for a polygon
noncomputable def area (P : Polygon) : ℝ := sorry

-- Define equidecomposability
def equidecomposable (P Q : Polygon) : Prop := sorry

-- Theorem statement
theorem equal_area_polygons_equidecomposable (P Q : Polygon) :
  area P = area Q → equidecomposable P Q := by sorry

end NUMINAMATH_CALUDE_equal_area_polygons_equidecomposable_l3721_372140


namespace NUMINAMATH_CALUDE_tangent_line_at_2_g_unique_minimum_l3721_372116

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x) + 1 / x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f x - 1 / x - a * Real.log x

-- Statement 1: Tangent line equation
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ 23 * x - 36 * y + 20 = 0 :=
sorry

-- Statement 2: Unique minimum point of g
theorem g_unique_minimum (a : ℝ) (h : a > 0) :
  ∃! x, x > 0 ∧ ∀ y, y > 0 → g a y ≥ g a x :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_g_unique_minimum_l3721_372116


namespace NUMINAMATH_CALUDE_problem_solution_l3721_372106

theorem problem_solution (p q : ℕ) (hp : p < 30) (hq : q < 30) (h_eq : p + q + p * q = 119) :
  p + q = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3721_372106


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3721_372150

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 28) :
  let side_length := face_perimeter / 4
  side_length ^ 3 = 343 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3721_372150


namespace NUMINAMATH_CALUDE_equal_squares_with_difference_one_l3721_372195

theorem equal_squares_with_difference_one :
  ∃ (a b : ℝ), a = b + 1 ∧ a^2 = b^2 :=
by sorry

end NUMINAMATH_CALUDE_equal_squares_with_difference_one_l3721_372195


namespace NUMINAMATH_CALUDE_larger_number_problem_l3721_372103

theorem larger_number_problem (x y : ℚ) : 
  (5 * y = 6 * x) → 
  (x + y = 42) → 
  (y > x) →
  y = 252 / 11 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3721_372103


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_product_and_gcd_l3721_372184

theorem two_digit_numbers_with_product_and_gcd 
  (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100) 
  (h3 : a * b = 1728) 
  (h4 : Nat.gcd a b = 12) : 
  (a = 36 ∧ b = 48) ∨ (a = 48 ∧ b = 36) := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_product_and_gcd_l3721_372184


namespace NUMINAMATH_CALUDE_winners_make_zeros_largest_c_winners_make_zeros_optimal_c_l3721_372122

/-- Represents a game state in Winners Make Zeros --/
structure GameState where
  m : ℕ
  n : ℕ

/-- Determines if a given game state is a winning position --/
def is_winning_position (state : GameState) : Prop :=
  sorry

/-- The largest valid choice for c that results in a winning position --/
def largest_winning_c : ℕ :=
  999

theorem winners_make_zeros_largest_c :
  ∀ c : ℕ,
    c > largest_winning_c →
    c > 0 ∧
    2007777 - c * 2007 ≥ 0 →
    ¬is_winning_position ⟨2007777 - c * 2007, 2007⟩ :=
by sorry

theorem winners_make_zeros_optimal_c :
  largest_winning_c > 0 ∧
  2007777 - largest_winning_c * 2007 ≥ 0 ∧
  is_winning_position ⟨2007777 - largest_winning_c * 2007, 2007⟩ :=
by sorry

end NUMINAMATH_CALUDE_winners_make_zeros_largest_c_winners_make_zeros_optimal_c_l3721_372122


namespace NUMINAMATH_CALUDE_four_digit_perfect_squares_l3721_372144

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

theorem four_digit_perfect_squares :
  (∀ n : ℕ, is_four_digit n ∧ all_even_digits n ∧ ∃ k, n = k^2 ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464) ∧
  (¬ ∃ n : ℕ, is_four_digit n ∧ all_odd_digits n ∧ ∃ k, n = k^2) :=
sorry

end NUMINAMATH_CALUDE_four_digit_perfect_squares_l3721_372144


namespace NUMINAMATH_CALUDE_base_10_to_base_5_88_l3721_372149

def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_10_to_base_5_88 : to_base_5 88 = [3, 2, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_5_88_l3721_372149


namespace NUMINAMATH_CALUDE_prism_minimum_characteristics_l3721_372124

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  base_edges : ℕ
  height : ℝ
  height_pos : height > 0

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ := p.base_edges + 2

/-- The number of edges in a prism -/
def num_edges (p : Prism) : ℕ := 3 * p.base_edges

/-- The number of lateral edges in a prism -/
def num_lateral_edges (p : Prism) : ℕ := p.base_edges

/-- The number of vertices in a prism -/
def num_vertices (p : Prism) : ℕ := 2 * p.base_edges

/-- Theorem about the minimum characteristics of a prism -/
theorem prism_minimum_characteristics :
  (∀ p : Prism, num_faces p ≥ 5) ∧
  (∃ p : Prism, num_faces p = 5 ∧
                num_edges p = 9 ∧
                num_lateral_edges p = 3 ∧
                num_vertices p = 6) := by sorry

end NUMINAMATH_CALUDE_prism_minimum_characteristics_l3721_372124


namespace NUMINAMATH_CALUDE_polynomial_existence_l3721_372123

theorem polynomial_existence : 
  ∃ (P : ℝ → ℝ → ℝ → ℝ), ∀ (t : ℝ), P (t^1993) (t^1994) (t + t^1995) = t := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l3721_372123


namespace NUMINAMATH_CALUDE_custom_op_two_five_l3721_372193

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_five_l3721_372193


namespace NUMINAMATH_CALUDE_min_value_rational_function_l3721_372159

theorem min_value_rational_function (x : ℝ) (h : x > 6) :
  (x^2 + 12*x) / (x - 6) ≥ 30 ∧
  ((x^2 + 12*x) / (x - 6) = 30 ↔ x = 12) :=
by sorry

end NUMINAMATH_CALUDE_min_value_rational_function_l3721_372159


namespace NUMINAMATH_CALUDE_dot_product_theorem_l3721_372141

def a : ℝ × ℝ := (1, 3)

theorem dot_product_theorem (b : ℝ × ℝ) 
  (h1 : Real.sqrt ((b.1 - 1)^2 + (b.2 - 3)^2) = Real.sqrt 10)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 2) :
  (2 * a.1 + b.1) * (a.1 - b.1) + (2 * a.2 + b.2) * (a.2 - b.2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l3721_372141


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_l3721_372162

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric -/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Checks if a polygon is inside a triangle -/
def isInsideTriangle (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Checks if a polygon is a hexagon -/
def isHexagon (p : Polygon) : Prop := sorry

/-- Checks if the vertices of a polygon divide the sides of a triangle into three equal parts -/
def verticesDivideSides (p : Polygon) (t : Triangle) : Prop := sorry

theorem largest_centrally_symmetric_polygon (t : Triangle) :
  ∃ (p : Polygon),
    isCentrallySymmetric p ∧
    isInsideTriangle p t ∧
    isHexagon p ∧
    verticesDivideSides p t ∧
    area p = (2/3) * triangleArea t ∧
    ∀ (q : Polygon),
      isCentrallySymmetric q → isInsideTriangle q t →
      area q ≤ area p :=
sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_l3721_372162


namespace NUMINAMATH_CALUDE_largest_special_number_l3721_372135

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem largest_special_number : 
  ∀ n : ℕ, n < 200 → is_perfect_square n → n % 3 = 0 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_l3721_372135


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l3721_372171

theorem a_gt_b_necessary_not_sufficient (a b c : ℝ) :
  (∀ c ≠ 0, a * c^2 > b * c^2 → a > b) ∧
  (∃ c, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l3721_372171


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_bought_14_apples_l3721_372117

theorem bridget_apples : ℕ → Prop :=
  fun total : ℕ =>
    let remaining_after_ann : ℕ := total / 2
    let remaining_after_cassie : ℕ := remaining_after_ann - 3
    remaining_after_cassie = 4 → total = 14

-- The proof
theorem bridget_bought_14_apples : bridget_apples 14 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_bought_14_apples_l3721_372117


namespace NUMINAMATH_CALUDE_power_function_sum_l3721_372136

/-- A power function passing through (4, 2) has k + a = 3/2 --/
theorem power_function_sum (k a : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ f : ℝ → ℝ, f x = k * x^a) → 
  k * 4^a = 2 → 
  k + a = 3/2 := by sorry

end NUMINAMATH_CALUDE_power_function_sum_l3721_372136


namespace NUMINAMATH_CALUDE_hiking_committee_selection_l3721_372185

theorem hiking_committee_selection (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_hiking_committee_selection_l3721_372185


namespace NUMINAMATH_CALUDE_non_fiction_count_is_six_l3721_372143

def fiction_count : ℕ := 5
def total_cases : ℕ := 150

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem non_fiction_count_is_six :
  ∃ (n : ℕ), n > 0 ∧ 
    (choose fiction_count 2) * (choose n 2) = total_cases ∧
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_non_fiction_count_is_six_l3721_372143


namespace NUMINAMATH_CALUDE_square_difference_identity_l3721_372113

theorem square_difference_identity (x : ℝ) : (x + 1)^2 - x^2 = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3721_372113


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3721_372155

def a : ℝ × ℝ := (1, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3721_372155


namespace NUMINAMATH_CALUDE_percentage_reduction_price_increase_l3721_372133

-- Define the original price
def original_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the initial profit per kilogram
def initial_profit : ℝ := 10

-- Define the initial daily sales
def initial_sales : ℝ := 500

-- Define the sales decrease rate
def sales_decrease_rate : ℝ := 20

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Theorem for the percentage reduction
theorem percentage_reduction (x : ℝ) : 
  original_price * (1 - x)^2 = final_price → x = 0.2 := by sorry

-- Theorem for the price increase
theorem price_increase (x : ℝ) :
  (initial_profit + x) * (initial_sales - sales_decrease_rate * x) = target_profit →
  x > 0 →
  ∀ y, y > 0 → 
  (initial_profit + y) * (initial_sales - sales_decrease_rate * y) = target_profit →
  x ≤ y →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_percentage_reduction_price_increase_l3721_372133


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l3721_372167

/-- The total number of watermelon seeds Yeon, Gwi, and Bom have together -/
def total_seeds (bom gwi yeon : ℕ) : ℕ := bom + gwi + yeon

/-- Theorem stating the total number of watermelon seeds -/
theorem watermelon_seeds_count :
  ∀ (bom gwi yeon : ℕ),
  bom = 300 →
  gwi = bom + 40 →
  yeon = 3 * gwi →
  total_seeds bom gwi yeon = 1660 :=
by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_count_l3721_372167


namespace NUMINAMATH_CALUDE_problem_statement_l3721_372139

theorem problem_statement (a b c d x : ℤ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : x = -1)  -- x is the largest negative integer
  : x^2 - (a + b - c * d)^2012 + (-c * d)^2011 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3721_372139


namespace NUMINAMATH_CALUDE_exactly_two_non_congruent_triangles_l3721_372182

/-- A triangle with integer side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The perimeter of a triangle --/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Triangle inequality --/
def is_valid_triangle (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Non-congruent triangles --/
def are_non_congruent (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of valid triangles with perimeter 12 --/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | perimeter t = 12 ∧ is_valid_triangle t}

/-- The theorem to be proved --/
theorem exactly_two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ valid_triangles ∧
    t2 ∈ valid_triangles ∧
    are_non_congruent t1 t2 ∧
    ∀ (t3 : IntTriangle),
      t3 ∈ valid_triangles →
      (t3 = t1 ∨ t3 = t2 ∨ ¬(are_non_congruent t1 t3 ∧ are_non_congruent t2 t3)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_non_congruent_triangles_l3721_372182


namespace NUMINAMATH_CALUDE_ryan_marbles_count_l3721_372102

/-- The number of friends Ryan shares his marbles with -/
def num_friends : ℕ := 9

/-- The number of marbles each friend receives -/
def marbles_per_friend : ℕ := 8

/-- Ryan's total number of marbles -/
def total_marbles : ℕ := num_friends * marbles_per_friend

theorem ryan_marbles_count : total_marbles = 72 := by
  sorry

end NUMINAMATH_CALUDE_ryan_marbles_count_l3721_372102


namespace NUMINAMATH_CALUDE_max_profit_at_84_l3721_372189

/-- The defect rate as a function of daily output --/
def defect_rate (x : ℕ) : ℚ :=
  if x ≤ 94 then 1 / (96 - x) else 2/3

/-- The daily profit as a function of daily output and profit per qualified instrument --/
def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if x ≤ 94 
  then (x - 3*x / (2*(96 - x))) * A
  else 0

/-- Theorem: The daily profit is maximized when the daily output is 84 --/
theorem max_profit_at_84 (A : ℚ) (h : A > 0) :
  ∀ x : ℕ, x ≥ 1 → daily_profit 84 A ≥ daily_profit x A :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_84_l3721_372189


namespace NUMINAMATH_CALUDE_sum_of_roots_l3721_372148

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3721_372148


namespace NUMINAMATH_CALUDE_ara_height_after_growth_l3721_372157

theorem ara_height_after_growth (initial_height : ℝ) (shea_growth_rate : ℝ) (ara_growth_fraction : ℝ) (shea_final_height : ℝ) :
  initial_height * (1 + shea_growth_rate) = shea_final_height →
  shea_growth_rate = 0.25 →
  ara_growth_fraction = 1 / 3 →
  shea_final_height = 70 →
  initial_height * (1 + ara_growth_fraction * shea_growth_rate) = 60.67 := by
sorry

end NUMINAMATH_CALUDE_ara_height_after_growth_l3721_372157


namespace NUMINAMATH_CALUDE_min_value_and_range_l3721_372191

theorem min_value_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + 3*b^2 = 3) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m) ∧ 
             m = 4) ∧
  (∀ x : ℝ, (2 * |x - 1| + |x| ≥ Real.sqrt 5 * a + b) ↔ (x ≤ -2/3 ∨ x ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_range_l3721_372191


namespace NUMINAMATH_CALUDE_extra_chairs_added_l3721_372154

/-- The number of extra chairs added to a wedding seating arrangement -/
theorem extra_chairs_added (rows : ℕ) (chairs_per_row : ℕ) (total_chairs : ℕ) : 
  rows = 7 → chairs_per_row = 12 → total_chairs = 95 → 
  total_chairs - (rows * chairs_per_row) = 11 := by
  sorry

end NUMINAMATH_CALUDE_extra_chairs_added_l3721_372154


namespace NUMINAMATH_CALUDE_abs_lt_one_iff_square_lt_one_l3721_372170

theorem abs_lt_one_iff_square_lt_one (x : ℝ) : |x| < 1 ↔ x^2 < 1 := by sorry

end NUMINAMATH_CALUDE_abs_lt_one_iff_square_lt_one_l3721_372170


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3721_372196

theorem quadratic_no_real_roots : ¬ ∃ (x : ℝ), x^2 + x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3721_372196


namespace NUMINAMATH_CALUDE_soccer_ball_weight_l3721_372147

theorem soccer_ball_weight (soccer_ball_weight bicycle_weight : ℝ) : 
  5 * soccer_ball_weight = 3 * bicycle_weight →
  2 * bicycle_weight = 60 →
  soccer_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_weight_l3721_372147


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3721_372112

/-- A regular hexagon inscribed in a square with specific properties -/
structure InscribedHexagon where
  square_perimeter : ℝ
  square_side_length : ℝ
  hexagon_side_length : ℝ
  hexagon_area : ℝ
  perimeter_constraint : square_perimeter = 160
  side_length_relation : square_side_length = square_perimeter / 4
  hexagon_side_relation : hexagon_side_length = square_side_length / 2
  area_formula : hexagon_area = 3 * Real.sqrt 3 / 2 * hexagon_side_length ^ 2

/-- The theorem stating the area of the inscribed hexagon -/
theorem inscribed_hexagon_area (h : InscribedHexagon) : h.hexagon_area = 600 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3721_372112


namespace NUMINAMATH_CALUDE_shark_sightings_total_l3721_372132

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := sorry

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- Cape May has 8 less than double the number of shark sightings of Daytona Beach -/
axiom cape_may_relation : cape_may_sightings = 2 * daytona_beach_sightings - 8

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := cape_may_sightings + daytona_beach_sightings

theorem shark_sightings_total : total_sightings = 40 := by
  sorry

end NUMINAMATH_CALUDE_shark_sightings_total_l3721_372132


namespace NUMINAMATH_CALUDE_arthurs_walk_distance_l3721_372137

/-- Represents Arthur's walk in blocks -/
structure ArthursWalk where
  east : ℕ
  north : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the total distance of Arthur's walk in miles -/
def total_distance (walk : ArthursWalk) : ℚ :=
  (walk.east + walk.north + walk.south + walk.west) * (1 / 4)

/-- Theorem: Arthur's specific walk equals 6.5 miles -/
theorem arthurs_walk_distance :
  let walk : ArthursWalk := { east := 8, north := 10, south := 3, west := 5 }
  total_distance walk = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_walk_distance_l3721_372137


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3721_372108

theorem pentagon_rectangle_ratio : 
  let pentagon_perimeter : ℝ := 100
  let rectangle_perimeter : ℝ := 100
  let pentagon_side := pentagon_perimeter / 5
  let rectangle_width := rectangle_perimeter / 6
  pentagon_side / rectangle_width = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3721_372108


namespace NUMINAMATH_CALUDE_min_sum_distances_to_lines_l3721_372119

/-- The minimum sum of distances from a point on the parabola y² = 4x to two specific lines -/
theorem min_sum_distances_to_lines : 
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line1 := {P : ℝ × ℝ | 4 * P.1 - 3 * P.2 + 6 = 0}
  let line2 := {P : ℝ × ℝ | P.1 = -1}
  let dist_to_line1 (P : ℝ × ℝ) := |4 * P.1 - 3 * P.2 + 6| / Real.sqrt (4^2 + (-3)^2)
  let dist_to_line2 (P : ℝ × ℝ) := |P.1 + 1|
  ∃ (min_dist : ℝ), min_dist = 2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ parabola → 
      dist_to_line1 P + dist_to_line2 P ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_to_lines_l3721_372119


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3721_372156

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 4) - y^2 / (k + 4) = 1

-- Theorem statement
theorem hyperbola_k_range (k : ℝ) :
  is_hyperbola k → k < -4 ∨ k > 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3721_372156


namespace NUMINAMATH_CALUDE_ceiling_floor_square_zero_l3721_372187

theorem ceiling_floor_square_zero : 
  (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ))^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_square_zero_l3721_372187


namespace NUMINAMATH_CALUDE_orange_juice_students_l3721_372186

/-- Given a school survey where 50% of students chose apple juice and 30% chose orange juice,
    with 120 students choosing apple juice, prove that 72 students chose orange juice. -/
theorem orange_juice_students (total : ℕ) (apple : ℕ) (orange : ℕ)
  (h1 : apple = 120)
  (h2 : apple = total * 50 / 100)
  (h3 : orange = total * 30 / 100) :
  orange = 72 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_students_l3721_372186


namespace NUMINAMATH_CALUDE_articles_bought_l3721_372166

/-- The number of articles bought at the cost price -/
def X : ℝ := sorry

/-- The cost price of each article -/
def C : ℝ := sorry

/-- The selling price of each article -/
def S : ℝ := sorry

/-- The gain percent -/
def gain_percent : ℝ := 8.695652173913043

theorem articles_bought (h1 : X * C = 46 * S) 
                        (h2 : gain_percent = ((S - C) / C) * 100) : 
  X = 50 := by sorry

end NUMINAMATH_CALUDE_articles_bought_l3721_372166


namespace NUMINAMATH_CALUDE_cylinder_cone_lateral_area_ratio_l3721_372183

/-- The ratio of lateral surface areas of a cylinder and a cone with equal slant heights and base radii -/
theorem cylinder_cone_lateral_area_ratio 
  (r : ℝ) -- base radius
  (l : ℝ) -- slant height
  (h_pos_r : r > 0)
  (h_pos_l : l > 0) :
  (2 * π * r * l) / (π * r * l) = 2 := by
  sorry

#check cylinder_cone_lateral_area_ratio

end NUMINAMATH_CALUDE_cylinder_cone_lateral_area_ratio_l3721_372183


namespace NUMINAMATH_CALUDE_residue_of_8_1234_mod_13_l3721_372175

theorem residue_of_8_1234_mod_13 : (8 : ℤ)^1234 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_8_1234_mod_13_l3721_372175


namespace NUMINAMATH_CALUDE_count_eight_digit_numbers_seven_different_is_correct_l3721_372134

/-- The number of 8-digit numbers where exactly 7 digits are all different -/
def count_eight_digit_numbers_seven_different : ℕ := 5080320

/-- Theorem stating that the count of 8-digit numbers where exactly 7 digits are all different is 5080320 -/
theorem count_eight_digit_numbers_seven_different_is_correct :
  count_eight_digit_numbers_seven_different = 5080320 := by sorry

end NUMINAMATH_CALUDE_count_eight_digit_numbers_seven_different_is_correct_l3721_372134


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3721_372153

/-- The logarithm function to base 10 -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The equation has exactly one solution iff a = 4 or a < 0 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, log10 (a * x) = 2 * log10 (x + 1) ∧ a * x > 0 ∧ x + 1 > 0) ↔ 
  (a = 4 ∨ a < 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3721_372153


namespace NUMINAMATH_CALUDE_b_alone_time_l3721_372145

/-- The time it takes for A and B together to complete the task -/
def time_AB : ℝ := 3

/-- The time it takes for B and C together to complete the task -/
def time_BC : ℝ := 6

/-- The time it takes for A and C together to complete the task -/
def time_AC : ℝ := 4.5

/-- The rate at which A completes the task -/
def rate_A : ℝ := sorry

/-- The rate at which B completes the task -/
def rate_B : ℝ := sorry

/-- The rate at which C completes the task -/
def rate_C : ℝ := sorry

theorem b_alone_time (h1 : rate_A + rate_B = 1 / time_AB)
                     (h2 : rate_B + rate_C = 1 / time_BC)
                     (h3 : rate_A + rate_C = 1 / time_AC) :
  1 / rate_B = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_b_alone_time_l3721_372145


namespace NUMINAMATH_CALUDE_maddie_purchase_cost_l3721_372100

/-- Calculates the total cost of Maddie's beauty product purchase --/
def calculate_total_cost (
  palette_price : ℝ)
  (palette_count : ℕ)
  (palette_discount : ℝ)
  (lipstick_price : ℝ)
  (lipstick_count : ℕ)
  (hair_color_price : ℝ)
  (hair_color_count : ℕ)
  (hair_color_discount : ℝ)
  (sales_tax_rate : ℝ) : ℝ :=
  let palette_cost := palette_price * palette_count * (1 - palette_discount)
  let lipstick_cost := lipstick_price * (lipstick_count - 1)
  let hair_color_cost := hair_color_price * hair_color_count * (1 - hair_color_discount)
  let subtotal := palette_cost + lipstick_cost + hair_color_cost
  let total := subtotal * (1 + sales_tax_rate)
  total

/-- Theorem stating that the total cost of Maddie's purchase is $58.64 --/
theorem maddie_purchase_cost :
  calculate_total_cost 15 3 0.2 2.5 4 4 3 0.1 0.08 = 58.64 := by
  sorry

end NUMINAMATH_CALUDE_maddie_purchase_cost_l3721_372100


namespace NUMINAMATH_CALUDE_equation_solution_range_l3721_372118

theorem equation_solution_range (k : ℝ) : 
  (∃! x : ℝ, x > 0 ∧ (x^2 + k*x + 3) / (x - 1) = 3*x + k) ↔ 
  (k = -33/8 ∨ k = -4 ∨ k ≥ -3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3721_372118


namespace NUMINAMATH_CALUDE_combination_permutation_problem_l3721_372192

-- Define the combination function
def C (n k : ℕ) : ℕ := n.choose k

-- Define the permutation function
def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- State the theorem
theorem combination_permutation_problem (n : ℕ) :
  C n 2 * A 2 2 = 42 → n.factorial / (3 * (n - 3).factorial) = 35 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_problem_l3721_372192


namespace NUMINAMATH_CALUDE_all_choose_same_house_probability_l3721_372198

/-- The probability that all 3 persons choose the same house when there are 3 houses
    and each person independently chooses a house with equal probability. -/
theorem all_choose_same_house_probability :
  let num_houses : ℕ := 3
  let num_persons : ℕ := 3
  let prob_choose_house : ℚ := 1 / 3
  (num_houses * (prob_choose_house ^ num_persons)) = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_all_choose_same_house_probability_l3721_372198


namespace NUMINAMATH_CALUDE_irrational_root_sum_squares_l3721_372115

theorem irrational_root_sum_squares (a b c : ℤ) (r : ℝ) 
  (h1 : a * r^2 + b * r + c = 0)
  (h2 : a * c ≠ 0) : 
  Irrational (Real.sqrt (r^2 + c^2)) :=
by sorry

end NUMINAMATH_CALUDE_irrational_root_sum_squares_l3721_372115


namespace NUMINAMATH_CALUDE_cantaloupes_left_total_l3721_372199

/-- The total number of cantaloupes left after each person's changes -/
def total_cantaloupes_left (fred_initial fred_eaten tim_initial tim_lost susan_initial susan_given nancy_initial nancy_traded : ℕ) : ℕ :=
  (fred_initial - fred_eaten) + (tim_initial - tim_lost) + (susan_initial - susan_given) + (nancy_initial - nancy_traded)

/-- Theorem stating the total number of cantaloupes left is 138 -/
theorem cantaloupes_left_total :
  total_cantaloupes_left 38 4 44 7 57 10 25 5 = 138 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_left_total_l3721_372199


namespace NUMINAMATH_CALUDE_car_wash_group_composition_l3721_372169

theorem car_wash_group_composition (total : ℕ) (girls : ℕ) : 
  girls = (2 * total : ℚ) / 5 →    -- Initially 40% of the group are girls
  ((girls : ℚ) - 2) / total = 3 / 10 →   -- After changes, 30% of the group are girls
  girls = 8 := by
sorry

end NUMINAMATH_CALUDE_car_wash_group_composition_l3721_372169


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3721_372130

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3721_372130


namespace NUMINAMATH_CALUDE_haley_cousins_count_l3721_372131

/-- The number of origami papers Haley has to give away -/
def total_papers : ℕ := 48

/-- The number of origami papers each cousin receives -/
def papers_per_cousin : ℕ := 8

/-- Haley's number of cousins -/
def num_cousins : ℕ := total_papers / papers_per_cousin

theorem haley_cousins_count : num_cousins = 6 := by
  sorry

end NUMINAMATH_CALUDE_haley_cousins_count_l3721_372131


namespace NUMINAMATH_CALUDE_cookie_count_l3721_372172

theorem cookie_count (bundles_per_box : ℕ) (cookies_per_bundle : ℕ) (num_boxes : ℕ) : 
  bundles_per_box = 9 → cookies_per_bundle = 7 → num_boxes = 13 →
  bundles_per_box * cookies_per_bundle * num_boxes = 819 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l3721_372172


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l3721_372188

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l3721_372188


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l3721_372177

/-- Given an angle α that satisfies α = 45° + k · 180° where k is an integer,
    the terminal side of α falls in either the first or third quadrant. -/
theorem terminal_side_quadrant (k : ℤ) (α : Real) 
  (h : α = 45 + k * 180) : 
  (0 < α % 360 ∧ α % 360 < 90) ∨ (180 < α % 360 ∧ α % 360 < 270) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l3721_372177


namespace NUMINAMATH_CALUDE_choose_product_equals_8400_l3721_372146

theorem choose_product_equals_8400 : Nat.choose 10 3 * Nat.choose 8 4 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_choose_product_equals_8400_l3721_372146


namespace NUMINAMATH_CALUDE_defective_pen_count_l3721_372142

theorem defective_pen_count (total_pens : ℕ) (prob_non_defective : ℚ) : 
  total_pens = 12 →
  prob_non_defective = 6/11 →
  (∃ (non_defective : ℕ), 
    (non_defective : ℚ) / total_pens * ((non_defective - 1) : ℚ) / (total_pens - 1) = prob_non_defective ∧
    total_pens - non_defective = 1) :=
by sorry

end NUMINAMATH_CALUDE_defective_pen_count_l3721_372142


namespace NUMINAMATH_CALUDE_cos_theta_value_l3721_372107

theorem cos_theta_value (θ : Real) 
  (h : (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) = 1/2) : 
  Real.cos θ = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_value_l3721_372107


namespace NUMINAMATH_CALUDE_median_divides_triangle_equally_l3721_372163

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- A median of a triangle -/
def median (t : Triangle) (vertex : Point) : Point := sorry

/-- Theorem: A median of a triangle divides the triangle into two triangles of equal area -/
theorem median_divides_triangle_equally (t : Triangle) (vertex : Point) :
  let m := median t vertex
  triangleArea ⟨t.A, t.B, m⟩ = triangleArea ⟨t.A, t.C, m⟩ ∨
  triangleArea ⟨t.B, t.C, m⟩ = triangleArea ⟨t.A, t.C, m⟩ ∨
  triangleArea ⟨t.A, t.B, m⟩ = triangleArea ⟨t.B, t.C, m⟩ :=
sorry

end NUMINAMATH_CALUDE_median_divides_triangle_equally_l3721_372163


namespace NUMINAMATH_CALUDE_books_not_sold_percentage_l3721_372197

def initial_stock : ℕ := 800
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem books_not_sold_percentage :
  percentage_not_sold = 66 := by sorry

end NUMINAMATH_CALUDE_books_not_sold_percentage_l3721_372197


namespace NUMINAMATH_CALUDE_smallest_n_for_shared_vertex_triangles_l3721_372120

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for a monochromatic triangle in a two-colored complete graph -/
def MonochromaticTriangle (n : ℕ) (c : TwoColoring n) (v₁ v₂ v₃ : Fin n) : Prop :=
  v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧
  c v₁ v₂ = c v₂ v₃ ∧ c v₂ v₃ = c v₁ v₃

/-- Predicate for two monochromatic triangles sharing exactly one vertex -/
def SharedVertexTriangles (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ (v₁ v₂ v₃ v₄ v₅ : Fin n),
    MonochromaticTriangle n c v₁ v₂ v₃ ∧
    MonochromaticTriangle n c v₁ v₄ v₅ ∧
    v₂ ≠ v₄ ∧ v₂ ≠ v₅ ∧ v₃ ≠ v₄ ∧ v₃ ≠ v₅

/-- The main theorem: 9 is the smallest n such that any two-coloring of K_n contains two monochromatic triangles sharing exactly one vertex -/
theorem smallest_n_for_shared_vertex_triangles :
  (∀ c : TwoColoring 9, SharedVertexTriangles 9 c) ∧
  (∀ m : ℕ, m < 9 → ∃ c : TwoColoring m, ¬SharedVertexTriangles m c) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_shared_vertex_triangles_l3721_372120


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3721_372176

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 4 + a 7 = 39)
  (h_sum2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3721_372176


namespace NUMINAMATH_CALUDE_walking_distance_ratio_l3721_372121

/-- The ratio of walking distances given different walking speeds and times -/
theorem walking_distance_ratio 
  (your_speed : ℝ) 
  (harris_speed : ℝ) 
  (harris_time : ℝ) 
  (your_time : ℝ) 
  (h1 : your_speed = 2 * harris_speed) 
  (h2 : harris_time = 2) 
  (h3 : your_time = 3) : 
  (your_speed * your_time) / (harris_speed * harris_time) = 3 := by
sorry

end NUMINAMATH_CALUDE_walking_distance_ratio_l3721_372121


namespace NUMINAMATH_CALUDE_six_friends_assignment_l3721_372126

/-- The number of ways to assign friends to rooms -/
def assignment_ways (n : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose n 3 * 1 * Nat.factorial 3

/-- Theorem stating the number of ways to assign 6 friends to 6 rooms -/
theorem six_friends_assignment :
  assignment_ways 6 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_six_friends_assignment_l3721_372126


namespace NUMINAMATH_CALUDE_imoProof_l3721_372181

theorem imoProof (a b : ℕ) (ha : a = 18) (hb : b = 1) : 
  ¬ (7 ∣ (a * b * (a + b))) ∧ 
  (7^7 ∣ ((a + b)^7 - a^7 - b^7)) := by
sorry

end NUMINAMATH_CALUDE_imoProof_l3721_372181


namespace NUMINAMATH_CALUDE_circle_points_condition_l3721_372127

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (2, 1)

-- Define the condition for a point being inside or outside the circle
def is_inside_circle (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 < 0
def is_outside_circle (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 > 0

-- Theorem statement
theorem circle_points_condition (a : ℝ) :
  (is_inside_circle point_A.1 point_A.2 a ∧ is_outside_circle point_B.1 point_B.2 a) ∨
  (is_outside_circle point_A.1 point_A.2 a ∧ is_inside_circle point_B.1 point_B.2 a) →
  -4 < a ∧ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_condition_l3721_372127


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3721_372174

theorem pipe_filling_time (fill_rate : ℝ → ℝ → ℝ) (time : ℝ → ℝ → ℝ) :
  (fill_rate 3 8 = 1) →
  (∀ n t, fill_rate n t * t = 1) →
  (time 2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3721_372174


namespace NUMINAMATH_CALUDE_total_cost_of_pipes_l3721_372129

def copper_length : ℝ := 10
def plastic_length : ℝ := copper_length + 5
def cost_per_meter : ℝ := 4

theorem total_cost_of_pipes : copper_length * cost_per_meter + plastic_length * cost_per_meter = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_pipes_l3721_372129


namespace NUMINAMATH_CALUDE_calvin_sequence_base_9_sum_l3721_372125

def calvin_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * calvin_sequence n + 5

def to_base_9 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 9) :: aux (m / 9)
    aux n

def sum_digits (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem calvin_sequence_base_9_sum :
  sum_digits (to_base_9 (calvin_sequence 10)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_calvin_sequence_base_9_sum_l3721_372125


namespace NUMINAMATH_CALUDE_max_value_at_negative_one_l3721_372114

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem max_value_at_negative_one (a b : ℝ) :
  (∀ x, f a b x ≤ 0) ∧
  (f a b (-1) = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ → f a b x < f a b (-1) + ε) →
  a + b = 11 :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_negative_one_l3721_372114


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l3721_372101

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l3721_372101


namespace NUMINAMATH_CALUDE_solve_equation_l3721_372105

theorem solve_equation (x : ℝ) (h : 9 - (4/x) = 7 + (8/x)) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3721_372105


namespace NUMINAMATH_CALUDE_product_inspection_l3721_372194

def total_products : ℕ := 100
def non_defective : ℕ := 98
def defective : ℕ := 2
def selected : ℕ := 3

theorem product_inspection :
  (Nat.choose total_products selected = 161700) ∧
  (Nat.choose defective 1 * Nat.choose non_defective (selected - 1) = 9506) ∧
  (Nat.choose total_products selected - Nat.choose non_defective selected = 9604) ∧
  (Nat.choose defective 1 * Nat.choose non_defective (selected - 1) * Nat.factorial selected = 57036) :=
by sorry

end NUMINAMATH_CALUDE_product_inspection_l3721_372194


namespace NUMINAMATH_CALUDE_original_number_proof_l3721_372110

theorem original_number_proof (h1 : 213 * 16 = 3408) 
  (h2 : ∃ x, x * 21.3 = 34.080000000000005) : 
  ∃ x, x * 21.3 = 34.080000000000005 ∧ x = 1.6 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l3721_372110


namespace NUMINAMATH_CALUDE_min_odd_integers_l3721_372128

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 30)
  (sum_abcd : a + b + c + d = 45)
  (sum_all : a + b + c + d + e + f = 62) :
  ∃ (odd_count : ℕ), 
    odd_count ≥ 2 ∧ 
    (∃ (odd_integers : Finset ℤ), 
      odd_integers.card = odd_count ∧
      odd_integers ⊆ {a, b, c, d, e, f} ∧
      ∀ x ∈ odd_integers, Odd x) ∧
    ∀ (other_odd_count : ℕ),
      other_odd_count < odd_count →
      ¬∃ (other_odd_integers : Finset ℤ),
        other_odd_integers.card = other_odd_count ∧
        other_odd_integers ⊆ {a, b, c, d, e, f} ∧
        ∀ x ∈ other_odd_integers, Odd x :=
sorry

end NUMINAMATH_CALUDE_min_odd_integers_l3721_372128


namespace NUMINAMATH_CALUDE_f_seven_equals_neg_seventeen_l3721_372111

/-- Given a function f(x) = a*x^7 + b*x^3 + c*x - 5 where a, b, and c are constants,
    if f(-7) = 7, then f(7) = -17 -/
theorem f_seven_equals_neg_seventeen 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5) 
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_neg_seventeen_l3721_372111


namespace NUMINAMATH_CALUDE_corner_square_length_l3721_372179

/-- Given a rectangular sheet of dimensions 48 m x 36 m, if squares of side length x
    are cut from each corner to form an open box with volume 5120 m³, then x = 8 meters. -/
theorem corner_square_length (x : ℝ) : 
  x > 0 ∧ x < 24 ∧ x < 18 →
  (48 - 2*x) * (36 - 2*x) * x = 5120 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_corner_square_length_l3721_372179


namespace NUMINAMATH_CALUDE_max_value_of_f_l3721_372190

def f (x : ℝ) : ℝ := -4 * x^2 + 10 * x

theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3721_372190


namespace NUMINAMATH_CALUDE_object_ends_on_left_l3721_372161

/-- Represents the sides of a square --/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left

/-- Represents the vertices of a regular octagon --/
inductive OctagonVertex
  | Bottom
  | BottomLeft
  | Left
  | TopLeft
  | Top
  | TopRight
  | Right
  | BottomRight

/-- The number of sides a square rolls to reach the leftmost position from the bottom --/
def numRolls : Nat := 4

/-- The angle of rotation for each roll of the square --/
def rotationPerRoll : Int := 135

/-- Function to calculate the final position of an object on a square
    after rolling around an octagon --/
def finalPosition (initialSide : SquareSide) (rolls : Nat) : SquareSide :=
  sorry

/-- Theorem stating that an object initially on the right side of the square
    will end up on the left side after rolling to the leftmost position --/
theorem object_ends_on_left :
  finalPosition SquareSide.Right numRolls = SquareSide.Left :=
  sorry

end NUMINAMATH_CALUDE_object_ends_on_left_l3721_372161


namespace NUMINAMATH_CALUDE_imaginary_power_l3721_372173

theorem imaginary_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_l3721_372173


namespace NUMINAMATH_CALUDE_vertical_strips_count_l3721_372178

/-- Represents a grid rectangle with a hole -/
structure GridRectangleWithHole where
  outer_perimeter : ℕ
  hole_perimeter : ℕ
  horizontal_strips : ℕ

/-- Theorem: Given a grid rectangle with a hole, if cutting horizontally yields 20 strips,
    then cutting vertically yields 21 strips -/
theorem vertical_strips_count
  (rect : GridRectangleWithHole)
  (h_outer : rect.outer_perimeter = 50)
  (h_hole : rect.hole_perimeter = 32)
  (h_horizontal : rect.horizontal_strips = 20) :
  ∃ (vertical_strips : ℕ), vertical_strips = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_strips_count_l3721_372178


namespace NUMINAMATH_CALUDE_inequality_proof_l3721_372109

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp : 0 < p) 
  (hpa : p ≤ a) (hpb : p ≤ b) (hpc : p ≤ c) (hpd : p ≤ d) (hpe : p ≤ e)
  (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 
    25 + 6 * (Real.sqrt (q/p) - Real.sqrt (p/q))^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3721_372109


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l3721_372160

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the ellipse C
def ellipse_C (x y θ : ℝ) : Prop := x = 2 * Real.sqrt 3 * Real.cos θ ∧ y = Real.sqrt 3 * Real.sin θ

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the equation of the ellipse we want to prove
def target_ellipse (x y : ℝ) : Prop := x^2 / 45 + y^2 / 36 = 1

-- State the theorem
theorem shortest_major_axis_ellipse :
  ∃ (M : ℝ × ℝ), 
    line_l M.1 M.2 ∧ 
    (∀ (E : ℝ × ℝ → Prop), 
      (∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
        (∀ (x y : ℝ), E (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
        E M ∧ 
        (∀ (x y : ℝ), E (x, y) → 
          Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) + 
          Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2 * a)) →
      (∀ (x y : ℝ), E (x, y) → target_ellipse x y)) :=
sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l3721_372160


namespace NUMINAMATH_CALUDE_goat_average_price_l3721_372165

/-- The average price of a goat given the total cost of cows and goats, and the average price of a cow -/
theorem goat_average_price
  (total_cost : ℕ)
  (num_cows : ℕ)
  (num_goats : ℕ)
  (cow_avg_price : ℕ)
  (h1 : total_cost = 1400)
  (h2 : num_cows = 2)
  (h3 : num_goats = 8)
  (h4 : cow_avg_price = 460) :
  (total_cost - num_cows * cow_avg_price) / num_goats = 60 := by
  sorry

end NUMINAMATH_CALUDE_goat_average_price_l3721_372165


namespace NUMINAMATH_CALUDE_married_student_percentage_l3721_372180

theorem married_student_percentage
  (total : ℝ)
  (total_positive : total > 0)
  (male_percentage : ℝ)
  (male_percentage_def : male_percentage = 0.7)
  (married_male_fraction : ℝ)
  (married_male_fraction_def : married_male_fraction = 1 / 7)
  (single_female_fraction : ℝ)
  (single_female_fraction_def : single_female_fraction = 1 / 3) :
  (male_percentage * married_male_fraction * total +
   (1 - male_percentage) * (1 - single_female_fraction) * total) / total = 0.3 := by
sorry

end NUMINAMATH_CALUDE_married_student_percentage_l3721_372180


namespace NUMINAMATH_CALUDE_smallest_discount_value_l3721_372104

theorem smallest_discount_value : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → 
    (1 - (m : ℝ) / 100 ≥ (1 - 0.20)^2 ∨ 
     1 - (m : ℝ) / 100 ≥ (1 - 0.15)^3 ∨ 
     1 - (m : ℝ) / 100 ≥ (1 - 0.30) * (1 - 0.10))) ∧ 
  (1 - (n : ℝ) / 100 < (1 - 0.20)^2 ∧ 
   1 - (n : ℝ) / 100 < (1 - 0.15)^3 ∧ 
   1 - (n : ℝ) / 100 < (1 - 0.30) * (1 - 0.10)) ∧ 
  n = 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_discount_value_l3721_372104


namespace NUMINAMATH_CALUDE_roberta_shopping_l3721_372152

def shopping_trip (initial_amount bag_price_difference : ℕ) : Prop :=
  let shoe_price := 45
  let bag_price := shoe_price - bag_price_difference
  let lunch_price := bag_price / 4
  let total_expenses := shoe_price + bag_price + lunch_price
  let money_left := initial_amount - total_expenses
  money_left = 78

theorem roberta_shopping :
  shopping_trip 158 17 := by
  sorry

end NUMINAMATH_CALUDE_roberta_shopping_l3721_372152


namespace NUMINAMATH_CALUDE_twenty_points_sixty_assignments_l3721_372168

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_needed (points : ℕ) : ℕ :=
  let groups := (points + 3) / 4
  (groups * (groups + 1) * 2) / 2

/-- The point system increases every 4 points -/
axiom point_system_increment (n : ℕ) : assignments_needed (4 * n) = 2 * n * (n + 1)

/-- The goal is to prove that 20 points require 60 assignments -/
theorem twenty_points_sixty_assignments : assignments_needed 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_twenty_points_sixty_assignments_l3721_372168
