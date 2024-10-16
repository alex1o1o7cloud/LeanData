import Mathlib

namespace NUMINAMATH_CALUDE_tailor_buttons_l4182_418231

/-- The number of buttons purchased by a tailor -/
def total_buttons (green : ℕ) : ℕ :=
  let yellow := green + 10
  let blue := green - 5
  let red := 2 * (yellow + blue)
  let white := red + green
  let black := red - green
  green + yellow + blue + red + white + black

/-- Theorem: The tailor purchased 1385 buttons -/
theorem tailor_buttons : total_buttons 90 = 1385 := by
  sorry

end NUMINAMATH_CALUDE_tailor_buttons_l4182_418231


namespace NUMINAMATH_CALUDE_Q_subset_complement_P_l4182_418232

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > 1}

-- Define the complement of P in the real numbers
def CₘP : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_subset_complement_P : Q ⊆ CₘP := by sorry

end NUMINAMATH_CALUDE_Q_subset_complement_P_l4182_418232


namespace NUMINAMATH_CALUDE_equality_check_l4182_418227

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  (-(3 * 2)^2 ≠ -3 * 2^2) ∧ 
  (-|2^3| ≠ |-2^3|) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l4182_418227


namespace NUMINAMATH_CALUDE_sheet_length_is_30_l4182_418219

/-- Represents the dimensions and usage of a typist's sheet. -/
structure TypistSheet where
  width : ℝ
  length : ℝ
  sideMargin : ℝ
  topBottomMargin : ℝ
  usagePercentage : ℝ

/-- Calculates the length of a typist's sheet given the specifications. -/
def calculateSheetLength (sheet : TypistSheet) : ℝ :=
  sheet.length

/-- Theorem stating that the length of the sheet is 30 cm under the given conditions. -/
theorem sheet_length_is_30 (sheet : TypistSheet)
    (h1 : sheet.width = 20)
    (h2 : sheet.sideMargin = 2)
    (h3 : sheet.topBottomMargin = 3)
    (h4 : sheet.usagePercentage = 64)
    (h5 : (sheet.width - 2 * sheet.sideMargin) * (sheet.length - 2 * sheet.topBottomMargin) = 
          sheet.usagePercentage / 100 * sheet.width * sheet.length) :
    calculateSheetLength sheet = 30 := by
  sorry

#check sheet_length_is_30

end NUMINAMATH_CALUDE_sheet_length_is_30_l4182_418219


namespace NUMINAMATH_CALUDE_prob_at_least_one_l4182_418271

theorem prob_at_least_one (P₁ P₂ : ℝ) 
  (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) 
  (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) : 
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_l4182_418271


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4182_418260

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 1 * a 3 + 2 * a 2 * a 3 + a 1 * a 5 = 16 →
  a 2 + a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4182_418260


namespace NUMINAMATH_CALUDE_airplane_passenger_ratio_l4182_418235

/-- Given an airplane with 80 passengers, of which 30 are men, prove that the ratio of men to women is 3:5. -/
theorem airplane_passenger_ratio :
  let total_passengers : ℕ := 80
  let num_men : ℕ := 30
  let num_women : ℕ := total_passengers - num_men
  (num_men : ℚ) / (num_women : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passenger_ratio_l4182_418235


namespace NUMINAMATH_CALUDE_complex_modulus_example_l4182_418242

theorem complex_modulus_example : Complex.abs (3 - 10*I) = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l4182_418242


namespace NUMINAMATH_CALUDE_solution_value_l4182_418288

/-- The function F as defined in the problem -/
def F (a b c : ℚ) : ℚ := a * b^3 + c

/-- Theorem stating that -5/19 is the solution to the equation -/
theorem solution_value : ∃ a : ℚ, F a 3 8 = F a 2 3 ∧ a = -5/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l4182_418288


namespace NUMINAMATH_CALUDE_cricket_bat_price_l4182_418248

theorem cricket_bat_price (cost_price_A : ℝ) (profit_A_percent : ℝ) (profit_B_percent : ℝ) : 
  cost_price_A = 156 →
  profit_A_percent = 20 →
  profit_B_percent = 25 →
  let selling_price_B := cost_price_A * (1 + profit_A_percent / 100)
  let selling_price_C := selling_price_B * (1 + profit_B_percent / 100)
  selling_price_C = 234 :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l4182_418248


namespace NUMINAMATH_CALUDE_third_term_expansion_l4182_418208

-- Define i as the imaginary unit
axiom i : ℂ
axiom i_squared : i * i = -1

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem third_term_expansion :
  let n : ℕ := 6
  let r : ℕ := 2
  (binomial n r : ℂ) * (1 : ℂ)^(n - r) * i^r = -15 := by sorry

end NUMINAMATH_CALUDE_third_term_expansion_l4182_418208


namespace NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_l_l4182_418253

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define the line l
def l (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := 2*x - 3*y + 3 = 0

-- Theorem statement
theorem line_through_circle_center_perpendicular_to_l :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y, C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 4) ∧ 
    (result_line x₀ y₀) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ ∧ l x₂ y₂ ∧ x₁ ≠ x₂ → 
      (y₂ - y₁) * (x₀ - x₁) = -(x₂ - x₁) * (y₀ - y₁)) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_l_l4182_418253


namespace NUMINAMATH_CALUDE_g_of_2_l4182_418261

def g (x : ℝ) : ℝ := x^2 - 3*x + 1

theorem g_of_2 : g 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_l4182_418261


namespace NUMINAMATH_CALUDE_percentage_difference_l4182_418251

theorem percentage_difference (w u y z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l4182_418251


namespace NUMINAMATH_CALUDE_find_set_B_l4182_418295

open Set

def U : Set ℕ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℕ := {1, 2, 3}

theorem find_set_B : 
  ∃! B : Set ℕ, (U \ (A ∩ B) = {1, 2, 4, 5, 6, 7}) ∧ B ⊆ U ∧ B = {3, 4, 5} :=
sorry

end NUMINAMATH_CALUDE_find_set_B_l4182_418295


namespace NUMINAMATH_CALUDE_stratified_sample_grade10_l4182_418220

theorem stratified_sample_grade10 (total_sample : ℕ) (grade12 : ℕ) (grade11 : ℕ) (grade10 : ℕ) :
  total_sample = 50 →
  grade12 = 750 →
  grade11 = 850 →
  grade10 = 900 →
  (grade10 * total_sample) / (grade12 + grade11 + grade10) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_grade10_l4182_418220


namespace NUMINAMATH_CALUDE_area_of_two_sectors_l4182_418209

theorem area_of_two_sectors (r : ℝ) (angle : ℝ) : 
  r = 10 → angle = π / 4 → 2 * (angle / (2 * π)) * (π * r^2) = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_two_sectors_l4182_418209


namespace NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l4182_418202

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Theorem: The coordinates of point A(2,3) with respect to the x-axis are (2,-3) -/
theorem coordinates_wrt_x_axis :
  let A : Point := ⟨2, 3⟩
  reflectAcrossXAxis A = ⟨2, -3⟩ := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l4182_418202


namespace NUMINAMATH_CALUDE_volunteers_distribution_l4182_418265

/-- The number of ways to distribute n volunteers into k schools,
    with each school receiving at least one volunteer. -/
def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 75 volunteers into 3 schools,
    with each school receiving at least one volunteer, is equal to 150. -/
theorem volunteers_distribution :
  distribute_volunteers 75 3 = 150 := by sorry

end NUMINAMATH_CALUDE_volunteers_distribution_l4182_418265


namespace NUMINAMATH_CALUDE_bus_passengers_l4182_418222

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 4 → 
  (initial_students : ℚ) * (2/3)^num_stops = 320/27 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l4182_418222


namespace NUMINAMATH_CALUDE_composition_fraction_l4182_418259

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_fraction : f (g (f 3)) / g (f (g 3)) = 59 / 35 := by
  sorry

end NUMINAMATH_CALUDE_composition_fraction_l4182_418259


namespace NUMINAMATH_CALUDE_unique_number_property_l4182_418237

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l4182_418237


namespace NUMINAMATH_CALUDE_evaluate_expression_l4182_418281

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4182_418281


namespace NUMINAMATH_CALUDE_point_on_graph_l4182_418257

/-- A point (x, y) lies on the graph of y = -6/x if and only if xy = -6 -/
def lies_on_graph (x y : ℝ) : Prop := x * y = -6

/-- The function f(x) = -6/x -/
noncomputable def f (x : ℝ) : ℝ := -6 / x

theorem point_on_graph : lies_on_graph 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l4182_418257


namespace NUMINAMATH_CALUDE_total_marks_calculation_l4182_418268

theorem total_marks_calculation (obtained_marks : ℝ) (percentage : ℝ) (total_marks : ℝ) : 
  obtained_marks = 450 → percentage = 90 → obtained_marks = (percentage / 100) * total_marks → 
  total_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l4182_418268


namespace NUMINAMATH_CALUDE_subtract_and_multiply_l4182_418221

theorem subtract_and_multiply :
  (5/6 - 1/3) * 3/4 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_and_multiply_l4182_418221


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4182_418230

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4182_418230


namespace NUMINAMATH_CALUDE_sum_of_squares_101_to_200_l4182_418236

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_101_to_200 :
  sum_of_squares 200 - sum_of_squares 100 = 2348350 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_101_to_200_l4182_418236


namespace NUMINAMATH_CALUDE_divisibility_by_six_l4182_418217

theorem divisibility_by_six (n : ℕ) : 6 ∣ (n^3 - 7*n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l4182_418217


namespace NUMINAMATH_CALUDE_calculate_expression_l4182_418210

theorem calculate_expression : |-3| + 8 / (-2) + Real.sqrt 16 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4182_418210


namespace NUMINAMATH_CALUDE_money_distribution_l4182_418269

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 400)
  (ac_sum : a + c = 300)
  (bc_sum : b + c = 150) :
  c = 50 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l4182_418269


namespace NUMINAMATH_CALUDE_triangle_inequality_with_area_l4182_418256

/-- Triangle inequality theorem for sides and area -/
theorem triangle_inequality_with_area (a b c S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : S > 0)
  (h5 : S = Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_area_l4182_418256


namespace NUMINAMATH_CALUDE_sams_water_buckets_l4182_418205

-- Define the initial amount of water
def initial_water : Real := 1

-- Define the additional amount of water
def additional_water : Real := 8.8

-- Define the total amount of water
def total_water : Real := initial_water + additional_water

-- Theorem statement
theorem sams_water_buckets : total_water = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_sams_water_buckets_l4182_418205


namespace NUMINAMATH_CALUDE_sin_alpha_value_l4182_418293

theorem sin_alpha_value (α β : Real) (a b : Fin 2 → Real) :
  a 0 = Real.cos α ∧ a 1 = Real.sin α ∧
  b 0 = Real.cos β ∧ b 1 = Real.sin β ∧
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 2 * Real.sqrt 5 / 5 ∧
  0 < α ∧ α < Real.pi / 2 ∧
  -Real.pi / 2 < β ∧ β < 0 ∧
  Real.cos (5 * Real.pi / 2 - β) = -5 / 13 →
  Real.sin α = 33 / 65 := by sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l4182_418293


namespace NUMINAMATH_CALUDE_lawn_length_is_70_l4182_418289

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  roadWidth : ℝ
  costPerSquareMeter : ℝ
  totalCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width

/-- Theorem stating the length of the lawn given specific conditions -/
theorem lawn_length_is_70 (l : LawnWithRoads) 
  (h1 : l.width = 50)
  (h2 : l.roadWidth = 10)
  (h3 : l.costPerSquareMeter = 3)
  (h4 : l.totalCost = 3600)
  (h5 : roadArea l = l.totalCost / l.costPerSquareMeter) :
  l.length = 70 := by
  sorry


end NUMINAMATH_CALUDE_lawn_length_is_70_l4182_418289


namespace NUMINAMATH_CALUDE_special_triangle_cosine_l4182_418286

/-- A triangle with consecutive integer side lengths where the middle angle is 1.5 times the smallest angle -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n
  side2 : ℕ := n + 1
  side3 : ℕ := n + 2
  smallest_angle : ℝ
  middle_angle : ℝ
  largest_angle : ℝ
  angle_sum : middle_angle = 1.5 * smallest_angle
  angle_total : smallest_angle + middle_angle + largest_angle = Real.pi

/-- The cosine of the smallest angle in a SpecialTriangle is 53/60 -/
theorem special_triangle_cosine (t : SpecialTriangle) : 
  Real.cos t.smallest_angle = 53 / 60 := by sorry

end NUMINAMATH_CALUDE_special_triangle_cosine_l4182_418286


namespace NUMINAMATH_CALUDE_norma_laundry_problem_l4182_418264

theorem norma_laundry_problem (t : ℕ) (s : ℕ) : 
  s = 2 * t →                    -- Twice as many sweaters as T-shirts
  (t + s) - (t + 3) = 15 →       -- 15 items missing
  t = 9 :=                       -- Prove that t (number of T-shirts) is 9
by sorry

end NUMINAMATH_CALUDE_norma_laundry_problem_l4182_418264


namespace NUMINAMATH_CALUDE_abc_inequalities_l4182_418255

theorem abc_inequalities (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l4182_418255


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l4182_418267

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 202 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 202 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l4182_418267


namespace NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l4182_418211

theorem consecutive_four_product_plus_one_is_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l4182_418211


namespace NUMINAMATH_CALUDE_polygon_with_five_triangles_has_fourteen_diagonals_l4182_418246

/-- A polygon is a shape with a finite number of straight sides. -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 0

/-- The number of triangles formed when drawing diagonals from one vertex of a polygon. -/
def triangles_from_vertex (p : Polygon) : ℕ := p.sides - 2

/-- The number of diagonals in a polygon. -/
def num_diagonals (p : Polygon) : ℕ := p.sides * (p.sides - 3) / 2

/-- Theorem: A polygon that can be divided into at most 5 triangles by drawing diagonals from one vertex has 14 diagonals. -/
theorem polygon_with_five_triangles_has_fourteen_diagonals (p : Polygon) 
  (h : triangles_from_vertex p ≤ 5) : 
  num_diagonals p = 14 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_five_triangles_has_fourteen_diagonals_l4182_418246


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l4182_418263

theorem book_arrangement_count : ℕ → ℕ → ℕ
  | 4, 6 => 17280
  | _, _ => 0

/-- The number of ways to arrange math and history books with specific constraints -/
theorem book_arrangement_proof (m h : ℕ) (hm : m = 4) (hh : h = 6) :
  book_arrangement_count m h = 4 * 3 * 2 * Nat.factorial h :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l4182_418263


namespace NUMINAMATH_CALUDE_average_problem_l4182_418262

theorem average_problem (a b c d P : ℝ) :
  (a + b + c + d) / 4 = 8 →
  (a + b + c + d + P) / 5 = P →
  P = 8 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l4182_418262


namespace NUMINAMATH_CALUDE_pizza_order_cost_is_185_l4182_418298

/-- Represents the cost calculation for a pizza order with special offers --/
def pizza_order_cost (
  large_pizza_price : ℚ)
  (medium_pizza_price : ℚ)
  (small_pizza_price : ℚ)
  (topping_price : ℚ)
  (drink_price : ℚ)
  (garlic_bread_price : ℚ)
  (triple_cheese_count : ℕ)
  (triple_cheese_toppings : ℕ)
  (meat_lovers_count : ℕ)
  (meat_lovers_toppings : ℕ)
  (veggie_delight_count : ℕ)
  (veggie_delight_toppings : ℕ)
  (drink_count : ℕ)
  (garlic_bread_count : ℕ) : ℚ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * large_pizza_price + triple_cheese_count * triple_cheese_toppings * topping_price
  let meat_lovers_cost := ((meat_lovers_count + 2) / 3) * medium_pizza_price + meat_lovers_count * meat_lovers_toppings * topping_price
  let veggie_delight_cost := ((veggie_delight_count * 3) / 5) * small_pizza_price + veggie_delight_count * veggie_delight_toppings * topping_price
  let drink_and_bread_cost := drink_count * drink_price + max 0 (garlic_bread_count - drink_count) * garlic_bread_price
  triple_cheese_cost + meat_lovers_cost + veggie_delight_cost + drink_and_bread_cost

/-- Theorem stating that the given order costs $185 --/
theorem pizza_order_cost_is_185 :
  pizza_order_cost 10 8 5 (5/2) 2 4 6 2 4 3 10 1 8 5 = 185 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_cost_is_185_l4182_418298


namespace NUMINAMATH_CALUDE_line_through_point_l4182_418247

theorem line_through_point (k : ℚ) : 
  (2 * k * (-3/2) - 3 * 4 = k + 2) → k = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l4182_418247


namespace NUMINAMATH_CALUDE_climb_10_stairs_l4182_418292

/-- Function representing the number of ways to climb n stairs -/
def climb_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | n + 4 => climb_ways (n + 3) + climb_ways (n + 2) + climb_ways n

/-- Theorem stating that there are 151 ways to climb 10 stairs -/
theorem climb_10_stairs : climb_ways 10 = 151 := by
  sorry

end NUMINAMATH_CALUDE_climb_10_stairs_l4182_418292


namespace NUMINAMATH_CALUDE_train_crossing_time_l4182_418216

theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 90 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4182_418216


namespace NUMINAMATH_CALUDE_wang_heng_birth_date_l4182_418244

theorem wang_heng_birth_date :
  ∃! (year month : ℕ),
    1901 ≤ year ∧ year ≤ 2000 ∧
    1 ≤ month ∧ month ≤ 12 ∧
    (month * 2 + 5) * 50 + year - 250 = 2088 ∧
    year = 1988 ∧
    month = 1 := by
  sorry

end NUMINAMATH_CALUDE_wang_heng_birth_date_l4182_418244


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l4182_418214

theorem quadratic_equation_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ -x₁^2 - 3*x₁ + 3 = 0 ∧ -x₂^2 - 3*x₂ + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l4182_418214


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4182_418272

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 4 < 0} = {x : ℝ | -1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4182_418272


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4182_418278

/-- The line equation ax + 2by - 2 = 0 -/
def line_equation (a b x y : ℝ) : Prop := a * x + 2 * b * y - 2 = 0

/-- The circle equation x^2 + y^2 - 4x - 2y - 8 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The line bisects the circumference of the circle -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a b x y → circle_equation x y

theorem min_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_bisect : line_bisects_circle a b) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4182_418278


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l4182_418241

/-- A quadratic function passing through given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_1 : a * (-3)^2 + b * (-3) + c = 0
  point_2 : a * (-2)^2 + b * (-2) + c = -3
  point_3 : a * (-1)^2 + b * (-1) + c = -4
  point_4 : c = -3

/-- Statements about the quadratic function -/
def statements (f : QuadraticFunction) : Fin 4 → Prop
  | 0 => f.a * f.c < 0
  | 1 => ∀ x > 1, ∀ y > x, f.a * y^2 + f.b * y + f.c > f.a * x^2 + f.b * x + f.c
  | 2 => f.a * (-4)^2 + (f.b - 4) * (-4) + f.c = 0
  | 3 => ∀ x, -1 < x → x < 0 → f.a * x^2 + (f.b - 1) * x + f.c + 3 > 0

/-- The main theorem -/
theorem quadratic_function_theorem (f : QuadraticFunction) :
  ∃ (S : Finset (Fin 4)), S.card = 2 ∧ (∀ i, i ∈ S ↔ statements f i) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l4182_418241


namespace NUMINAMATH_CALUDE_current_batting_average_l4182_418200

/-- Represents a cricket player's batting statistics -/
structure BattingStats where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculates the batting average -/
def batting_average (stats : BattingStats) : ℚ :=
  stats.total_runs / stats.matches_played

/-- The theorem statement -/
theorem current_batting_average 
  (current_stats : BattingStats)
  (next_match_runs : ℕ)
  (new_average : ℚ)
  (h1 : current_stats.matches_played = 6)
  (h2 : batting_average 
    ⟨current_stats.matches_played + 1, current_stats.total_runs + next_match_runs⟩ = new_average)
  (h3 : next_match_runs = 78)
  (h4 : new_average = 54)
  : batting_average current_stats = 50 := by
  sorry

end NUMINAMATH_CALUDE_current_batting_average_l4182_418200


namespace NUMINAMATH_CALUDE_factorial_not_divisible_by_square_l4182_418291

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem factorial_not_divisible_by_square (n : ℕ) :
  ¬((n - 1).factorial % (n^2) = 0) ↔ 
  (n = 8 ∨ n = 9 ∨ 
   (∃ p : ℕ, is_prime p ∧ (n = p ∨ n = 2*p))) :=
sorry

end NUMINAMATH_CALUDE_factorial_not_divisible_by_square_l4182_418291


namespace NUMINAMATH_CALUDE_shell_collection_division_l4182_418218

theorem shell_collection_division (lino_morning : ℝ) (maria_morning : ℝ) 
  (lino_afternoon : ℝ) (maria_afternoon : ℝ) 
  (h1 : lino_morning = 292.5) 
  (h2 : maria_morning = 375.25)
  (h3 : lino_afternoon = 324.75)
  (h4 : maria_afternoon = 419.3) : 
  (lino_morning + lino_afternoon + maria_morning + maria_afternoon) / 2 = 705.9 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_division_l4182_418218


namespace NUMINAMATH_CALUDE_tuition_calculation_l4182_418250

theorem tuition_calculation (discount : ℕ) (total_cost : ℕ) : 
  discount = 15 → total_cost = 75 → (total_cost + discount) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_tuition_calculation_l4182_418250


namespace NUMINAMATH_CALUDE_cookie_combinations_theorem_l4182_418287

/-- The number of different combinations of cookies that can be purchased
    given the specified conditions. -/
def cookieCombinations : ℕ := 34

/-- The number of types of cookies available. -/
def cookieTypes : ℕ := 4

/-- The total number of cookies to be purchased. -/
def totalCookies : ℕ := 8

/-- The minimum number of each type of cookie to be purchased. -/
def minEachType : ℕ := 1

theorem cookie_combinations_theorem :
  (cookieTypes = 4) →
  (totalCookies = 8) →
  (minEachType = 1) →
  (cookieCombinations = 34) := by sorry

end NUMINAMATH_CALUDE_cookie_combinations_theorem_l4182_418287


namespace NUMINAMATH_CALUDE_beatrice_on_beach_probability_l4182_418249

theorem beatrice_on_beach_probability :
  let p_beach : ℝ := 1/2  -- Probability of Béatrice being on the beach
  let p_tennis : ℝ := 1/4  -- Probability of Béatrice being on the tennis court
  let p_cafe : ℝ := 1/4  -- Probability of Béatrice being in the cafe
  let p_not_found_beach : ℝ := 1/2  -- Probability of not finding Béatrice if she's on the beach
  let p_not_found_tennis : ℝ := 1/3  -- Probability of not finding Béatrice if she's on the tennis court
  let p_not_found_cafe : ℝ := 0  -- Probability of not finding Béatrice if she's in the cafe

  let p_beach_and_not_found : ℝ := p_beach * p_not_found_beach
  let p_tennis_and_not_found : ℝ := p_tennis * p_not_found_tennis
  let p_cafe_and_not_found : ℝ := p_cafe * p_not_found_cafe
  let p_not_found_total : ℝ := p_beach_and_not_found + p_tennis_and_not_found + p_cafe_and_not_found

  p_beach_and_not_found / p_not_found_total = 3/5 :=
by
  sorry

#check beatrice_on_beach_probability

end NUMINAMATH_CALUDE_beatrice_on_beach_probability_l4182_418249


namespace NUMINAMATH_CALUDE_perimeter_folded_square_l4182_418240

/-- Given a square ABCD with side length 2, where A is folded to meet BC at A' such that A'C = 1/2,
    the perimeter of triangle A'BD is (3 + √17)/2 + 2√2. -/
theorem perimeter_folded_square (A B C D A' : ℝ × ℝ) : 
  (∀ (X Y : ℝ × ℝ), ‖X - Y‖ = 2 → (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A)) →
  A'.1 = B.1 + 3/2 →
  A'.2 = B.2 →
  C.1 = B.1 + 2 →
  C.2 = B.2 →
  ‖A' - C‖ = 1/2 →
  ‖A' - B‖ + ‖B - D‖ + ‖D - A'‖ = (3 + Real.sqrt 17) / 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_folded_square_l4182_418240


namespace NUMINAMATH_CALUDE_total_cost_is_55_l4182_418215

/-- The total cost of two pairs of shoes, where the first pair costs $22 and the second pair is 50% more expensive than the first pair. -/
def total_cost : ℝ :=
  let first_pair_cost : ℝ := 22
  let second_pair_cost : ℝ := first_pair_cost * 1.5
  first_pair_cost + second_pair_cost

/-- Theorem stating that the total cost of the two pairs of shoes is $55. -/
theorem total_cost_is_55 : total_cost = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_55_l4182_418215


namespace NUMINAMATH_CALUDE_four_digit_count_is_900_l4182_418283

/-- The count of four-digit positive integers with thousands digit 3 and non-zero hundreds digit -/
def four_digit_count : ℕ :=
  let thousands_digit := 3
  let hundreds_choices := 9  -- 1 to 9
  let tens_choices := 10     -- 0 to 9
  let ones_choices := 10     -- 0 to 9
  hundreds_choices * tens_choices * ones_choices

theorem four_digit_count_is_900 : four_digit_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_count_is_900_l4182_418283


namespace NUMINAMATH_CALUDE_farm_ploughing_rate_l4182_418273

/-- Given a farm field and ploughing conditions, calculate the required daily ploughing rate to finish on time -/
theorem farm_ploughing_rate (total_area planned_rate actual_rate extra_days left_area : ℕ) : 
  total_area = 720 ∧ 
  actual_rate = 85 ∧ 
  extra_days = 2 ∧ 
  left_area = 40 →
  (total_area : ℚ) / ((total_area - left_area) / actual_rate - extra_days) = 120 := by
  sorry

end NUMINAMATH_CALUDE_farm_ploughing_rate_l4182_418273


namespace NUMINAMATH_CALUDE_largest_value_l4182_418258

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 1 ∧ y - 1 = z + 5 ∧ z + 5 = w - 2) :
  w ≥ x ∧ w ≥ y ∧ w ≥ z := by sorry

end NUMINAMATH_CALUDE_largest_value_l4182_418258


namespace NUMINAMATH_CALUDE_fold_line_length_squared_fold_line_theorem_l4182_418233

/-- Represents an equilateral triangle with side length 15 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 15

/-- Represents the folded triangle -/
structure FoldedTriangle extends EquilateralTriangle where
  fold_distance : ℝ
  is_valid_fold : fold_distance = 11

/-- The theorem stating the square of the fold line length -/
theorem fold_line_length_squared (t : FoldedTriangle) : ℝ :=
  2174209 / 78281

/-- The main theorem to be proved -/
theorem fold_line_theorem (t : FoldedTriangle) : 
  fold_line_length_squared t = 2174209 / 78281 := by
  sorry

end NUMINAMATH_CALUDE_fold_line_length_squared_fold_line_theorem_l4182_418233


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l4182_418252

theorem complex_arithmetic_equation : 
  -4^2 * ((1 - 7) / 6)^3 + ((-5)^3 - 3) / (-2)^3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l4182_418252


namespace NUMINAMATH_CALUDE_tims_toads_l4182_418266

theorem tims_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ) 
  (h1 : jim_toads = tim_toads + 20)
  (h2 : sarah_toads = 2 * jim_toads)
  (h3 : sarah_toads = 100) : 
  tim_toads = 30 := by
sorry

end NUMINAMATH_CALUDE_tims_toads_l4182_418266


namespace NUMINAMATH_CALUDE_find_q_l4182_418274

-- Define the polynomial Q(x)
def Q (x p q d : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- Define the properties of the polynomial
def polynomial_properties (p q d : ℝ) : Prop :=
  let zeros_sum := -p
  let zeros_product := -d
  let coefficients_sum := 1 + p + q + d
  (zeros_sum / 3 = zeros_product) ∧ (zeros_product = coefficients_sum)

-- Theorem statement
theorem find_q : 
  ∀ p q d : ℝ, polynomial_properties p q d → d = 5 → q = -26 :=
by sorry

end NUMINAMATH_CALUDE_find_q_l4182_418274


namespace NUMINAMATH_CALUDE_lunch_cakes_count_l4182_418234

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := sorry

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served -/
def total_cakes : ℕ := 14

/-- Theorem stating that the number of cakes served during lunch today is 5 -/
theorem lunch_cakes_count : lunch_cakes = 5 := by
  sorry

#check lunch_cakes_count

end NUMINAMATH_CALUDE_lunch_cakes_count_l4182_418234


namespace NUMINAMATH_CALUDE_sector_central_angle_l4182_418239

theorem sector_central_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  l + 2 * r = 6 →
  (1 / 2) * l * r = 2 →
  α = l / r →
  α = 1 ∨ α = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4182_418239


namespace NUMINAMATH_CALUDE_vector_collinearity_l4182_418270

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- The problem statement -/
theorem vector_collinearity (m : ℝ) :
  collinear (m + 3, 2) (m, 1) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l4182_418270


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l4182_418277

theorem gcd_of_three_numbers :
  Nat.gcd 13642 (Nat.gcd 19236 34176) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l4182_418277


namespace NUMINAMATH_CALUDE_sphere_hemisphere_cone_volume_ratio_l4182_418228

/-- The ratio of the volume of a sphere to the combined volume of a hemisphere and a cone -/
theorem sphere_hemisphere_cone_volume_ratio (r : ℝ) (hr : r > 0) : 
  (4 / 3 * π * r^3) / ((1 / 2 * 4 / 3 * π * (3 * r)^3) + (1 / 3 * π * r^2 * (2 * r))) = 1 / 14 := by
  sorry

#check sphere_hemisphere_cone_volume_ratio

end NUMINAMATH_CALUDE_sphere_hemisphere_cone_volume_ratio_l4182_418228


namespace NUMINAMATH_CALUDE_collinear_points_sum_l4182_418280

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), p2 = p1 + t • (p3 - p1) ∧ p3 = p1 + s • (p3 - p1)

/-- If the points (2,a,b), (a,3,b), and (a,b,4) are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

#check collinear_points_sum

end NUMINAMATH_CALUDE_collinear_points_sum_l4182_418280


namespace NUMINAMATH_CALUDE_b_plus_c_positive_l4182_418238

theorem b_plus_c_positive (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_c_positive_l4182_418238


namespace NUMINAMATH_CALUDE_bounded_sequence_periodic_l4182_418223

/-- A bounded sequence of integers satisfying the given recurrence relation -/
def BoundedSequence (a : ℕ → ℤ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, |a n| ≤ M ∧
  ∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) * a (n-4)) / (a (n-1) * a (n-2) + a (n-3) + a (n-4))

/-- Definition of a periodic sequence -/
def IsPeriodic (a : ℕ → ℤ) (l : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ k ≥ l, a (k + T) = a k

/-- The main theorem -/
theorem bounded_sequence_periodic (a : ℕ → ℤ) (h : BoundedSequence a) :
  ∃ l : ℕ, IsPeriodic a l := by sorry

end NUMINAMATH_CALUDE_bounded_sequence_periodic_l4182_418223


namespace NUMINAMATH_CALUDE_prime_sum_10_product_21_l4182_418275

theorem prime_sum_10_product_21 (p q : ℕ) : 
  Prime p → Prime q → p ≠ q → p + q = 10 → p * q = 21 := by sorry

end NUMINAMATH_CALUDE_prime_sum_10_product_21_l4182_418275


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l4182_418229

theorem right_triangle_consecutive_even_sides (a b c : ℕ) : 
  (∃ x : ℕ, a = x - 2 ∧ b = x ∧ c = x + 2) →  -- sides are consecutive even numbers
  (a^2 + b^2 = c^2) →                        -- right-angled triangle
  c = 10                                     -- hypotenuse length is 10
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l4182_418229


namespace NUMINAMATH_CALUDE_inequality_proof_l4182_418203

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a / c < b / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4182_418203


namespace NUMINAMATH_CALUDE_duck_weight_calculation_l4182_418254

theorem duck_weight_calculation (num_ducks : ℕ) (cost_per_duck : ℚ) (selling_price_per_pound : ℚ) (profit : ℚ) : 
  num_ducks = 30 →
  cost_per_duck = 10 →
  selling_price_per_pound = 5 →
  profit = 300 →
  (profit + num_ducks * cost_per_duck) / (selling_price_per_pound * num_ducks) = 4 := by
sorry

end NUMINAMATH_CALUDE_duck_weight_calculation_l4182_418254


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4182_418282

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (50 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (50 * π / 180)) =
  Real.tan (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4182_418282


namespace NUMINAMATH_CALUDE_intersection_point_of_parabolas_l4182_418206

-- Define the parabolas
def C₁ (x y : ℝ) : Prop :=
  (x - (Real.sqrt 2 - 1))^2 = 2 * (y - 1)^2

def C₂ (a b x y : ℝ) : Prop :=
  x^2 - a*y + x + 2*b = 0

-- Define the perpendicular tangents condition
def perpendicularTangents (a : ℝ) (x y : ℝ) : Prop :=
  (2*y - 2) * (2*y - a) = -1

-- Theorem statement
theorem intersection_point_of_parabolas
  (a b : ℝ) (h : ∃ x y, C₁ x y ∧ C₂ a b x y ∧ perpendicularTangents a x y) :
  ∃ x y, C₁ x y ∧ C₂ a b x y ∧ x = Real.sqrt 2 - 1/2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_parabolas_l4182_418206


namespace NUMINAMATH_CALUDE_parabola_chords_fixed_point_and_isosceles_triangle_l4182_418299

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define a chord on the parabola passing through A
def chord_through_A (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ parabola point_A.1 point_A.2

-- Define perpendicularity of two chords
def perpendicular_chords (P Q : ℝ × ℝ) : Prop :=
  (P.1 - point_A.1) * (Q.1 - point_A.1) + (P.2 - point_A.2) * (Q.2 - point_A.2) = 0

-- Define the point T
def point_T : ℝ × ℝ := (5, -2)

-- Define a line passing through a point
def line_through_point (P Q : ℝ × ℝ) (T : ℝ × ℝ) : Prop :=
  (T.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (T.1 - P.1)

-- Define an isosceles triangle
def isosceles_triangle (P Q : ℝ × ℝ) : Prop :=
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 = (Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2

-- Theorem statement
theorem parabola_chords_fixed_point_and_isosceles_triangle
  (P Q : ℝ × ℝ)
  (h1 : chord_through_A P)
  (h2 : chord_through_A Q)
  (h3 : perpendicular_chords P Q)
  (h4 : line_through_point P Q point_T) :
  (∀ R : ℝ × ℝ, chord_through_A R → perpendicular_chords P R → line_through_point P R point_T) ∧
  (∃! R : ℝ × ℝ, chord_through_A R ∧ perpendicular_chords P R ∧ isosceles_triangle P R) :=
sorry

end NUMINAMATH_CALUDE_parabola_chords_fixed_point_and_isosceles_triangle_l4182_418299


namespace NUMINAMATH_CALUDE_problem_solution_l4182_418279

theorem problem_solution (a : ℚ) : a + a/3 + a/4 = 11/4 → a = 33/19 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4182_418279


namespace NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_sufficient_not_necessary_l4182_418297

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem 1
theorem range_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

-- Theorem 2
theorem range_a_for_sufficient_not_necessary :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_sufficient_not_necessary_l4182_418297


namespace NUMINAMATH_CALUDE_sum_of_cubes_equality_l4182_418212

def original_equation (x : ℝ) : Prop :=
  x * Real.rpow x (1/3) + 4*x - 9 * Real.rpow x (1/3) + 2 = 0

def transformed_equation (y : ℝ) : Prop :=
  y^4 + 4*y^3 - 9*y + 2 = 0

def roots_original : Set ℝ :=
  {x : ℝ | original_equation x ∧ x ≥ 0}

def roots_transformed : Set ℝ :=
  {y : ℝ | transformed_equation y ∧ y ≥ 0}

theorem sum_of_cubes_equality :
  ∀ (x₁ x₂ x₃ x₄ : ℝ) (y₁ y₂ y₃ y₄ : ℝ),
  roots_original = {x₁, x₂, x₃, x₄} →
  roots_transformed = {y₁, y₂, y₃, y₄} →
  x₁^3 + x₂^3 + x₃^3 + x₄^3 = y₁^9 + y₂^9 + y₃^9 + y₄^9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equality_l4182_418212


namespace NUMINAMATH_CALUDE_trailingZerosOfSquareMinusFactorial_l4182_418226

-- Define the number we're working with
def n : ℕ := 999999

-- Define the factorial function
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

-- Define a function to count trailing zeros
def countTrailingZeros (x : ℕ) : ℕ :=
  if x = 0 then 0
  else if x % 10 = 0 then 1 + countTrailingZeros (x / 10)
  else 0

-- Theorem statement
theorem trailingZerosOfSquareMinusFactorial :
  countTrailingZeros (n^2 - factorial 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trailingZerosOfSquareMinusFactorial_l4182_418226


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l4182_418225

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l4182_418225


namespace NUMINAMATH_CALUDE_mobile_plan_comparison_l4182_418276

/-- Represents the monthly cost in yuan for a mobile phone plan -/
def monthly_cost (rental : ℝ) (rate : ℝ) (duration : ℝ) : ℝ :=
  rental + rate * duration

/-- The monthly rental fee for Global Call in yuan -/
def global_call_rental : ℝ := 50

/-- The per-minute call rate for Global Call in yuan -/
def global_call_rate : ℝ := 0.4

/-- The monthly rental fee for Shenzhouxing in yuan -/
def shenzhouxing_rental : ℝ := 0

/-- The per-minute call rate for Shenzhouxing in yuan -/
def shenzhouxing_rate : ℝ := 0.6

/-- The breakeven point in minutes where both plans cost the same -/
def breakeven_point : ℝ := 250

theorem mobile_plan_comparison :
  ∀ duration : ℝ,
    duration > breakeven_point →
      monthly_cost global_call_rental global_call_rate duration <
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration ∧
    duration < breakeven_point →
      monthly_cost global_call_rental global_call_rate duration >
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration ∧
    duration = breakeven_point →
      monthly_cost global_call_rental global_call_rate duration =
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration :=
by
  sorry

end NUMINAMATH_CALUDE_mobile_plan_comparison_l4182_418276


namespace NUMINAMATH_CALUDE_locus_of_points_l4182_418290

noncomputable def circle_locus (d : ℝ) : Set (ℝ × ℝ) :=
  if 0 < d ∧ d < 0.5 then {p : ℝ × ℝ | p.1^2 + p.2^2 = 1 - 2*d}
  else if d = 0.5 then {(0, 0)}
  else ∅

theorem locus_of_points (k : Set (ℝ × ℝ)) (O P Q : ℝ × ℝ) (d : ℝ) :
  (k = {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = 1}) →
  (O = (0, 0)) →
  (P = (-1, 0)) →
  (Q = (1, 0)) →
  (∀ e f : ℝ, 
    (abs (e - f) = d) → 
    (∀ E₁ F₁ F₂ : ℝ × ℝ,
      (E₁ ∈ k ∧ E₁.1 = e) →
      ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (E₁.1 - P.1)^2 + (E₁.2 - P.2)^2) →
      (F₁.1 = f) →
      ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (E₁.1 - P.1)^2 + (E₁.2 - P.2)^2) →
      (F₂.1 = f) →
      (F₁ ∈ circle_locus d ∧ F₂ ∈ circle_locus d))) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_points_l4182_418290


namespace NUMINAMATH_CALUDE_count_even_factors_l4182_418294

def n : ℕ := 2^3 * 3^2 * 7^1 * 11^1

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 36 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l4182_418294


namespace NUMINAMATH_CALUDE_odd_functions_sufficient_not_necessary_l4182_418284

-- Define the real-valued functions
variable (f g h : ℝ → ℝ)

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the relationship between f, g, and h
def FunctionsRelated (f g h : ℝ → ℝ) : Prop := ∀ x, h x = f x * g x

-- Theorem statement
theorem odd_functions_sufficient_not_necessary :
  (∀ f g h, FunctionsRelated f g h → (IsOdd f ∧ IsOdd g → IsEven h)) ∧
  (∃ f g h, FunctionsRelated f g h ∧ IsEven h ∧ ¬(IsOdd f ∧ IsOdd g)) :=
sorry

end NUMINAMATH_CALUDE_odd_functions_sufficient_not_necessary_l4182_418284


namespace NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l4182_418201

theorem cubic_equation_sum_of_cubes (a b c : ℝ) : 
  (a - Real.rpow 20 (1/3 : ℝ)) * (a - Real.rpow 70 (1/3 : ℝ)) * (a - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  (b - Real.rpow 20 (1/3 : ℝ)) * (b - Real.rpow 70 (1/3 : ℝ)) * (b - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  (c - Real.rpow 20 (1/3 : ℝ)) * (c - Real.rpow 70 (1/3 : ℝ)) * (c - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  a ≠ b → b ≠ c → a ≠ c →
  a^3 + b^3 + c^3 = 260.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l4182_418201


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l4182_418224

theorem least_k_for_inequality : ∃ k : ℤ, (∀ j : ℤ, 0.00010101 * (10 : ℝ)^j > 10 → k ≤ j) ∧ 0.00010101 * (10 : ℝ)^k > 10 ∧ k = 6 :=
by sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l4182_418224


namespace NUMINAMATH_CALUDE_monotonicity_intervals_l4182_418285

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * a * x^2 + (2*a^2 + a - 1) * x + 3

theorem monotonicity_intervals (a : ℝ) :
  (a = 2 → ∀ x y, x < y → f a x < f a y) ∧
  (a < 2 → (∀ x y, x < y ∧ y < 2*a - 1 → f a x < f a y) ∧
           (∀ x y, 2*a - 1 < x ∧ x < y ∧ y < a + 1 → f a x > f a y) ∧
           (∀ x y, a + 1 < x ∧ x < y → f a x < f a y)) ∧
  (a > 2 → (∀ x y, x < y ∧ y < a + 1 → f a x < f a y) ∧
           (∀ x y, a + 1 < x ∧ x < y ∧ y < 2*a - 1 → f a x > f a y) ∧
           (∀ x y, 2*a - 1 < x ∧ x < y → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_l4182_418285


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l4182_418243

/-- A hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ
  /-- The hyperbola has a horizontal axis -/
  horizontal_axis : Prop

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ := 
  fun x ↦ -2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = 4)
  (h3 : h.horizontal_axis) :
  other_asymptote h = fun x ↦ -2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l4182_418243


namespace NUMINAMATH_CALUDE_fifth_power_sum_l4182_418207

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l4182_418207


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l4182_418204

theorem ratio_of_x_intercepts (s t : ℝ) : 
  (5 = 2 * s + 5) →  -- First line equation at x-intercept
  (5 = 7 * t + 5) →  -- Second line equation at x-intercept
  s / t = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l4182_418204


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l4182_418213

theorem fraction_ratio_equality : ∃ (x y : ℚ), 
  (x / y) / (7 / 15) = ((5 / 3) / ((2 / 3) - (1 / 4))) / ((1 / 3 + 1 / 6) / (1 / 2 - 1 / 3)) ∧
  x / y = 28 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l4182_418213


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l4182_418245

/-- The modulus for our congruences -/
def m : ℕ := 9

/-- First congruence equation -/
def eq1 (x y : ℤ) : Prop := y ≡ 2 * x + 3 [ZMOD m]

/-- Second congruence equation -/
def eq2 (x y : ℤ) : Prop := y ≡ 7 * x + 6 [ZMOD m]

/-- The x-coordinate of the intersection point -/
def intersection_x : ℤ := 3

theorem intersection_point_x_coordinate :
  ∃ y : ℤ, eq1 intersection_x y ∧ eq2 intersection_x y :=
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l4182_418245


namespace NUMINAMATH_CALUDE_number_difference_proof_l4182_418296

theorem number_difference_proof :
  ∃ x : ℚ, (1 / 3 : ℚ) * x - (1 / 4 : ℚ) * x = 3 ∧ x = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l4182_418296
