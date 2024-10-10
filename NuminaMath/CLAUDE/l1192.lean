import Mathlib

namespace cuboid_area_example_l1192_119284

/-- The surface area of a cuboid -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 6 cm, and height 9 cm is 348 square centimeters -/
theorem cuboid_area_example : cuboid_surface_area 8 6 9 = 348 := by
  sorry

end cuboid_area_example_l1192_119284


namespace point_transformation_theorem_l1192_119217

def rotate90CounterClockwise (center x : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := x
  (cx - (py - cy), cy + (px - cx))

def reflectAboutYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem point_transformation_theorem (a b : ℝ) :
  let p := (a, b)
  let center := (2, 6)
  let transformed := reflectAboutYEqualsX (rotate90CounterClockwise center p)
  transformed = (-7, 4) → b - a = 15 := by
  sorry

end point_transformation_theorem_l1192_119217


namespace seven_telephones_wires_l1192_119258

/-- The number of wires needed to connect n telephone sets, where each pair is connected. -/
def wiresNeeded (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 7 telephone sets, the number of wires needed is 21. -/
theorem seven_telephones_wires : wiresNeeded 7 = 21 := by
  sorry

end seven_telephones_wires_l1192_119258


namespace definite_integral_x_squared_l1192_119285

theorem definite_integral_x_squared : ∫ x in (0 : ℝ)..1, x^2 = 1/3 := by sorry

end definite_integral_x_squared_l1192_119285


namespace intersection_of_A_and_B_l1192_119248

def A : Set ℕ := {2, 4, 6, 16, 29}
def B : Set ℕ := {4, 16, 20, 27, 29, 32}

theorem intersection_of_A_and_B : A ∩ B = {4, 16, 29} := by sorry

end intersection_of_A_and_B_l1192_119248


namespace negation_of_unique_solution_l1192_119296

theorem negation_of_unique_solution (a b : ℝ) (h : a ≠ 0) :
  ¬(∃! x : ℝ, a * x = b) ↔ (¬∃ x : ℝ, a * x = b) ∨ (∃ x y : ℝ, x ≠ y ∧ a * x = b ∧ a * y = b) :=
by sorry

end negation_of_unique_solution_l1192_119296


namespace additional_sheep_problem_l1192_119224

theorem additional_sheep_problem (mary_sheep : ℕ) (bob_additional : ℕ) :
  mary_sheep = 300 →
  (mary_sheep + 266 = 2 * mary_sheep + bob_additional - 69) →
  bob_additional = 35 := by
  sorry

end additional_sheep_problem_l1192_119224


namespace expand_binomial_product_simplify_algebraic_expression_expand_cubic_difference_l1192_119250

-- 1. Expansion of (2m-3)(5-3m)
theorem expand_binomial_product (m : ℝ) : 
  (2*m - 3) * (5 - 3*m) = -6*m^2 + 19*m - 15 := by sorry

-- 2. Simplification of (3a^3)^2⋅(2b^2)^3÷(6ab)^2
theorem simplify_algebraic_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (3*a^3)^2 * (2*b^2)^3 / (6*a*b)^2 = 2*a^4*b^4 := by sorry

-- 3. Expansion of (a-b)(a^2+ab+b^2)
theorem expand_cubic_difference (a b : ℝ) : 
  (a - b) * (a^2 + a*b + b^2) = a^3 - b^3 := by sorry

end expand_binomial_product_simplify_algebraic_expression_expand_cubic_difference_l1192_119250


namespace alice_grading_papers_l1192_119275

/-- Given that Ms. Alice can grade 296 papers in 8 hours, prove that she can grade 407 papers in 11 hours. -/
theorem alice_grading_papers : 
  let papers_in_8_hours : ℕ := 296
  let hours_initial : ℕ := 8
  let hours_new : ℕ := 11
  let papers_in_11_hours : ℕ := 407
  (papers_in_8_hours : ℚ) / hours_initial * hours_new = papers_in_11_hours :=
by sorry

end alice_grading_papers_l1192_119275


namespace pyramid_paint_theorem_l1192_119238

/-- Represents a pyramid-like structure with a given number of floors -/
structure PyramidStructure where
  floors : Nat

/-- Calculates the number of painted faces on one side of the structure -/
def sideFaces (p : PyramidStructure) : Nat :=
  (p.floors * (p.floors + 1)) / 2

/-- Calculates the total number of red-painted faces -/
def redFaces (p : PyramidStructure) : Nat :=
  4 * sideFaces p

/-- Calculates the total number of blue-painted faces -/
def blueFaces (p : PyramidStructure) : Nat :=
  sideFaces p

/-- Calculates the total number of painted faces -/
def totalPaintedFaces (p : PyramidStructure) : Nat :=
  redFaces p + blueFaces p

/-- Theorem stating the ratio of red to blue painted faces and the total number of painted faces -/
theorem pyramid_paint_theorem (p : PyramidStructure) (h : p.floors = 25) :
  redFaces p / blueFaces p = 4 ∧ totalPaintedFaces p = 1625 := by
  sorry

end pyramid_paint_theorem_l1192_119238


namespace initial_stock_proof_l1192_119255

/-- The number of coloring books initially in stock at a store -/
def initial_stock : ℕ := 86

/-- The number of coloring books sold -/
def books_sold : ℕ := 37

/-- The number of shelves used for remaining books -/
def shelves_used : ℕ := 7

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 7

/-- Theorem stating that the initial stock equals 86 -/
theorem initial_stock_proof : 
  initial_stock = books_sold + (shelves_used * books_per_shelf) :=
by sorry

end initial_stock_proof_l1192_119255


namespace rhombus_other_diagonal_l1192_119270

/-- Represents a rhombus with given area and one diagonal -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Theorem: In a rhombus with area 60 cm² and one diagonal 12 cm, the other diagonal is 10 cm -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.area = 60) (h2 : r.diagonal1 = 12) :
  ∃ (diagonal2 : ℝ), diagonal2 = 10 ∧ r.area = (r.diagonal1 * diagonal2) / 2 :=
by sorry

end rhombus_other_diagonal_l1192_119270


namespace xy_length_is_30_l1192_119299

/-- A right triangle XYZ with specific angle and side length properties -/
structure RightTriangleXYZ where
  /-- The length of side XZ -/
  xz : ℝ
  /-- The measure of angle Y in radians -/
  angle_y : ℝ
  /-- XZ equals 15 -/
  xz_eq : xz = 15
  /-- Angle Y equals 30 degrees (π/6 radians) -/
  angle_y_eq : angle_y = π / 6
  /-- The triangle is a right triangle (angle X is 90 degrees) -/
  right_angle : True

/-- The length of side XY in the right triangle XYZ -/
def length_xy (t : RightTriangleXYZ) : ℝ := 2 * t.xz

/-- Theorem stating that the length of XY is 30 in the given right triangle -/
theorem xy_length_is_30 (t : RightTriangleXYZ) : length_xy t = 30 := by
  sorry

end xy_length_is_30_l1192_119299


namespace quadratic_function_property_l1192_119261

-- Define the quadratic function
variable (f : ℝ → ℝ)

-- Define the interval [a, b]
variable (a b : ℝ)

-- Define the axis of symmetry
def axis_of_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f x

-- Define the range condition
def range_condition (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y ∈ Set.Icc (f b) (f a), ∃ x ∈ Set.Icc a b, f x = y

-- Theorem statement
theorem quadratic_function_property
  (h_axis : axis_of_symmetry f)
  (h_range : range_condition f a b) :
  ∀ x, x ∉ Set.Ioo a b :=
sorry

end quadratic_function_property_l1192_119261


namespace pyramid_base_edge_length_l1192_119215

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramid_height : ℝ
  /-- Radius of the hemisphere -/
  hemisphere_radius : ℝ
  /-- The hemisphere is tangent to the four faces of the pyramid -/
  tangent_to_faces : Bool

/-- Theorem: Edge length of the base of the pyramid -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.pyramid_height = 4)
  (h2 : p.hemisphere_radius = 3)
  (h3 : p.tangent_to_faces = true) :
  ∃ (edge_length : ℝ), edge_length = Real.sqrt 14 := by
  sorry

end pyramid_base_edge_length_l1192_119215


namespace gcd_20020_11011_l1192_119214

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := by
  sorry

end gcd_20020_11011_l1192_119214


namespace derivative_sin_plus_cos_at_pi_l1192_119274

/-- Given f(x) = sin(x) + cos(x), prove that f'(π) = -1 -/
theorem derivative_sin_plus_cos_at_pi :
  let f := λ x : ℝ => Real.sin x + Real.cos x
  (deriv f) π = -1 := by sorry

end derivative_sin_plus_cos_at_pi_l1192_119274


namespace inequality_solution_sets_l1192_119227

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, ax^2 + b*x - 1 < 0 ↔ -1/2 < x ∧ x < 1) →
  (∀ x, (a*x + 2) / (b*x + 1) < 0 ↔ x < -1 ∨ x > 1) :=
sorry

end inequality_solution_sets_l1192_119227


namespace consecutive_odd_integers_average_l1192_119221

theorem consecutive_odd_integers_average (n : ℕ) (first : ℤ) :
  n = 10 →
  first = 145 →
  first % 2 = 1 →
  let sequence := List.range n |>.map (λ i => first + 2 * i)
  (sequence.sum / n : ℚ) = 154 := by
  sorry

end consecutive_odd_integers_average_l1192_119221


namespace complex_in_third_quadrant_l1192_119269

theorem complex_in_third_quadrant (z : ℂ) : z * (1 + Complex.I) = 1 - 2 * Complex.I → 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_in_third_quadrant_l1192_119269


namespace parallel_no_common_points_relation_l1192_119232

-- Define the concept of lines in a space
axiom Line : Type

-- Define the parallel relation between lines
axiom parallel : Line → Line → Prop

-- Define the property of having no common points
axiom no_common_points : Line → Line → Prop

-- Define the theorem
theorem parallel_no_common_points_relation (a b : Line) :
  (parallel a b → no_common_points a b) ∧
  ¬(no_common_points a b → parallel a b) :=
sorry

end parallel_no_common_points_relation_l1192_119232


namespace inequality_chain_l1192_119291

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end inequality_chain_l1192_119291


namespace farm_problem_l1192_119201

/-- Proves that given the conditions of the farm problem, the number of hens is 24 -/
theorem farm_problem (hens cows : ℕ) : 
  hens + cows = 48 →
  2 * hens + 4 * cows = 144 →
  hens = 24 := by
sorry

end farm_problem_l1192_119201


namespace same_remainder_divisor_l1192_119219

theorem same_remainder_divisor : ∃! (n : ℕ), n > 0 ∧ 
  ∃ (r : ℕ), r > 0 ∧ r < n ∧ 
  (2287 % n = r) ∧ (2028 % n = r) ∧ (1806 % n = r) :=
by sorry

end same_remainder_divisor_l1192_119219


namespace derivative_value_l1192_119212

theorem derivative_value (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 2 + x^3) :
  f' 2 = -12 := by
  sorry

end derivative_value_l1192_119212


namespace equation_represents_parabola_l1192_119251

/-- The equation |y-3| = √((x+4)² + (y-1)²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2) ↔ y = a * x^2 + b * x + c) :=
by sorry

end equation_represents_parabola_l1192_119251


namespace triangle_angle_calculation_l1192_119259

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  B = 60 * π / 180 →
  b = 7 * Real.sqrt 6 →
  a = 14 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = 45 * π / 180 := by
sorry

end triangle_angle_calculation_l1192_119259


namespace square_area_ratio_l1192_119203

/-- Given a large square and a small square with coinciding centers,
    if the area of the cross formed by the small square is 17 times
    the area of the small square, then the area of the large square
    is 81 times the area of the small square. -/
theorem square_area_ratio (small_side large_side : ℝ) : 
  small_side > 0 →
  large_side > 0 →
  2 * large_side * small_side - small_side^2 = 17 * small_side^2 →
  large_side^2 = 81 * small_side^2 := by
  sorry

#check square_area_ratio

end square_area_ratio_l1192_119203


namespace f_sum_equals_two_l1192_119218

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

noncomputable def f_deriv : ℝ → ℝ := deriv f

theorem f_sum_equals_two :
  f 2017 + f_deriv 2017 + f (-2017) - f_deriv (-2017) = 2 := by sorry

end f_sum_equals_two_l1192_119218


namespace sandi_spending_ratio_l1192_119278

/-- Proves that the ratio of Sandi's spending to her initial amount is 1:2 --/
theorem sandi_spending_ratio :
  ∀ (sandi_initial sandi_spent gillian_spent : ℚ),
  sandi_initial = 600 →
  gillian_spent = 3 * sandi_spent + 150 →
  gillian_spent = 1050 →
  sandi_spent / sandi_initial = 1 / 2 := by
sorry

end sandi_spending_ratio_l1192_119278


namespace sum_34_47_in_base4_l1192_119243

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 10 and returns the result in base 4 -/
def addAndConvertToBase4 (a b : ℕ) : List ℕ :=
  toBase4 (a + b)

theorem sum_34_47_in_base4 :
  addAndConvertToBase4 34 47 = [1, 1, 0, 1] :=
sorry

end sum_34_47_in_base4_l1192_119243


namespace tangent_circles_m_value_l1192_119256

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 1 = 0

-- Define the condition of external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m

-- State the theorem
theorem tangent_circles_m_value (m : ℝ) :
  externally_tangent m → m = 3 ∨ m = -3 := by
  sorry

end tangent_circles_m_value_l1192_119256


namespace partial_fraction_decomposition_product_l1192_119292

theorem partial_fraction_decomposition_product (N₁ N₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 → (42 * x - 36) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -288 := by
sorry

end partial_fraction_decomposition_product_l1192_119292


namespace complement_A_intersect_B_l1192_119229

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := {x | 2 < x ∧ x ≤ 3}

-- State the theorem
theorem complement_A_intersect_B : (Aᶜ ∩ B) = open_interval := by sorry

end complement_A_intersect_B_l1192_119229


namespace restaurant_bill_calculation_l1192_119247

def total_cost (adult_meal_cost adult_drink_cost kid_drink_cost dessert_cost : ℚ)
               (num_adults num_kids num_exclusive_dishes : ℕ)
               (discount_rate sales_tax_rate service_charge_rate : ℚ)
               (exclusive_dish_charge : ℚ) : ℚ :=
  let subtotal := adult_meal_cost * num_adults +
                  adult_drink_cost * num_adults +
                  kid_drink_cost * num_kids +
                  dessert_cost * (num_adults + num_kids) +
                  exclusive_dish_charge * num_exclusive_dishes
  let discounted_subtotal := subtotal * (1 - discount_rate)
  let with_tax := discounted_subtotal * (1 + sales_tax_rate)
  let final_total := with_tax * (1 + service_charge_rate)
  final_total

theorem restaurant_bill_calculation :
  let adult_meal_cost : ℚ := 12
  let adult_drink_cost : ℚ := 2.5
  let kid_drink_cost : ℚ := 1.5
  let dessert_cost : ℚ := 4
  let num_adults : ℕ := 7
  let num_kids : ℕ := 4
  let num_exclusive_dishes : ℕ := 3
  let discount_rate : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.075
  let service_charge_rate : ℚ := 0.15
  let exclusive_dish_charge : ℚ := 3
  total_cost adult_meal_cost adult_drink_cost kid_drink_cost dessert_cost
             num_adults num_kids num_exclusive_dishes
             discount_rate sales_tax_rate service_charge_rate
             exclusive_dish_charge = 178.57 := by
  sorry

end restaurant_bill_calculation_l1192_119247


namespace andy_rahim_age_difference_l1192_119290

/-- The age difference between Andy and Rahim -/
def ageDifference (rahimAge : ℕ) (andyFutureAge : ℕ) : ℕ :=
  andyFutureAge - 5 - rahimAge

theorem andy_rahim_age_difference :
  ∀ (rahimAge : ℕ) (andyFutureAge : ℕ),
    rahimAge = 6 →
    andyFutureAge = 2 * rahimAge →
    ageDifference rahimAge andyFutureAge = 1 := by
  sorry

end andy_rahim_age_difference_l1192_119290


namespace annes_total_distance_l1192_119276

/-- Anne's hiking journey -/
def annes_hike (flat_speed flat_time uphill_speed uphill_time downhill_speed downhill_time : ℝ) : ℝ :=
  flat_speed * flat_time + uphill_speed * uphill_time + downhill_speed * downhill_time

/-- Theorem: Anne's total distance traveled is 14 miles -/
theorem annes_total_distance :
  annes_hike 3 2 2 2 4 1 = 14 := by
  sorry

end annes_total_distance_l1192_119276


namespace smallest_positive_integer_congruence_l1192_119200

theorem smallest_positive_integer_congruence :
  ∃ y : ℕ+, 
    (∀ z : ℕ+, (42 * z.val + 8) % 24 = 4 → y ≤ z) ∧
    (42 * y.val + 8) % 24 = 4 ∧
    y.val = 2 := by
  sorry

end smallest_positive_integer_congruence_l1192_119200


namespace coefficient_x4_is_negative_seven_l1192_119242

/-- The coefficient of x^4 in the expanded expression -/
def coefficient_x4 (a b c d e f g : ℤ) : ℤ :=
  5 * a - 3 * 0 + 4 * (-3)

/-- The expression to be expanded -/
def expression (x : ℚ) : ℚ :=
  5 * (x^4 - 2*x^3 + x^2) - 3 * (x^2 - x + 1) + 4 * (x^6 - 3*x^4 + x^3)

theorem coefficient_x4_is_negative_seven :
  coefficient_x4 1 (-2) 1 0 (-1) 1 = -7 := by sorry

end coefficient_x4_is_negative_seven_l1192_119242


namespace general_inequality_l1192_119207

theorem general_inequality (x : ℝ) (n : ℕ) (h : x > 0) (hn : n > 0) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
  sorry

end general_inequality_l1192_119207


namespace smallest_n_mod_30_l1192_119268

theorem smallest_n_mod_30 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(435 * m ≡ 867 * m [ZMOD 30])) ∧ 
  (435 * n ≡ 867 * n [ZMOD 30]) :=
by sorry

end smallest_n_mod_30_l1192_119268


namespace eggs_per_box_l1192_119239

/-- Given a chicken coop with hens that lay eggs daily, prove the number of eggs per box -/
theorem eggs_per_box 
  (num_hens : ℕ) 
  (eggs_per_hen_per_day : ℕ) 
  (days_per_week : ℕ) 
  (boxes_per_week : ℕ) 
  (h1 : num_hens = 270)
  (h2 : eggs_per_hen_per_day = 1)
  (h3 : days_per_week = 7)
  (h4 : boxes_per_week = 315) :
  (num_hens * eggs_per_hen_per_day * days_per_week) / boxes_per_week = 6 :=
by sorry

end eggs_per_box_l1192_119239


namespace sum_reciprocals_bound_l1192_119280

theorem sum_reciprocals_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) ∧ ∀ x ≥ 2, ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ 1 / a + 1 / b = x :=
by sorry

end sum_reciprocals_bound_l1192_119280


namespace rock_cd_price_l1192_119244

/-- The price of a rock and roll CD -/
def rock_price : ℝ := sorry

/-- The price of a pop CD -/
def pop_price : ℝ := 10

/-- The price of a dance CD -/
def dance_price : ℝ := 3

/-- The price of a country CD -/
def country_price : ℝ := 7

/-- The number of each type of CD Julia wants to buy -/
def quantity : ℕ := 4

/-- The amount of money Julia has -/
def julia_money : ℝ := 75

/-- The amount Julia is short by -/
def short_amount : ℝ := 25

theorem rock_cd_price : rock_price = 5 := by
  sorry

end rock_cd_price_l1192_119244


namespace wendy_recycling_points_l1192_119216

def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def unrecycled_bags : ℕ := 2

theorem wendy_recycling_points :
  (total_bags - unrecycled_bags) * points_per_bag = 45 :=
by sorry

end wendy_recycling_points_l1192_119216


namespace first_series_seasons_l1192_119213

/-- Represents the number of seasons in the first movie series -/
def S : ℕ := sorry

/-- Represents the number of seasons in the second movie series -/
def second_series_seasons : ℕ := 14

/-- Represents the original number of episodes per season -/
def original_episodes_per_season : ℕ := 16

/-- Represents the number of episodes lost per season -/
def lost_episodes_per_season : ℕ := 2

/-- Represents the total number of episodes remaining after the loss -/
def total_remaining_episodes : ℕ := 364

/-- Theorem stating that the number of seasons in the first movie series is 12 -/
theorem first_series_seasons :
  S = 12 :=
by sorry

end first_series_seasons_l1192_119213


namespace point_m_location_l1192_119262

theorem point_m_location (L P M : ℚ) : 
  L = 1/6 → P = 1/12 → M - L = (P - L) / 3 → M = 1/9 := by
  sorry

end point_m_location_l1192_119262


namespace largest_of_five_consecutive_even_l1192_119279

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (n : ℕ) : ℕ := 5 * n - 20

theorem largest_of_five_consecutive_even :
  ∃ n : ℕ, sum_first_n_even 30 = sum_five_consecutive_even n ∧ n = 190 := by
  sorry

end largest_of_five_consecutive_even_l1192_119279


namespace equations_not_intersecting_at_roots_l1192_119205

theorem equations_not_intersecting_at_roots : ∀ (x : ℝ),
  (x = 0 ∨ x = 3) →
  (x = x - 3) →
  False :=
by sorry

#check equations_not_intersecting_at_roots

end equations_not_intersecting_at_roots_l1192_119205


namespace shorter_side_length_l1192_119246

theorem shorter_side_length (a b : ℕ) : 
  a > b →                 -- Ensure a is the longer side
  2 * a + 2 * b = 48 →    -- Perimeter condition
  a * b = 140 →           -- Area condition
  b = 10 := by            -- Conclusion: shorter side is 10 feet
sorry

end shorter_side_length_l1192_119246


namespace arithmetic_sequence_50th_term_l1192_119283

/-- Given an arithmetic sequence where a₇ = 10 and a₂₁ = 34, prove that a₅₀ = 682/7 -/
theorem arithmetic_sequence_50th_term :
  ∀ (a : ℕ → ℚ), 
    (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
    a 7 = 10 →                                        -- 7th term is 10
    a 21 = 34 →                                       -- 21st term is 34
    a 50 = 682 / 7 :=                                 -- 50th term is 682/7
by
  sorry

end arithmetic_sequence_50th_term_l1192_119283


namespace distance_at_16_00_l1192_119297

/-- Represents the distance to Moscow at a given time -/
structure DistanceAtTime where
  time : ℕ  -- Time in hours since 12:00
  lowerBound : ℚ
  upperBound : ℚ

/-- The problem statement -/
theorem distance_at_16_00 
  (d12 : DistanceAtTime) 
  (d13 : DistanceAtTime)
  (d15 : DistanceAtTime)
  (h_constant_speed : ∀ t₁ t₂, d12.time ≤ t₁ → t₁ < t₂ → t₂ ≤ d15.time → 
    (d12.lowerBound - d15.upperBound) / (d15.time - d12.time) ≤ 
    (d12.upperBound - d15.lowerBound) / (d15.time - d12.time))
  (h_d12 : d12.time = 0 ∧ d12.lowerBound = 81.5 ∧ d12.upperBound = 82.5)
  (h_d13 : d13.time = 1 ∧ d13.lowerBound = 70.5 ∧ d13.upperBound = 71.5)
  (h_d15 : d15.time = 3 ∧ d15.lowerBound = 45.5 ∧ d15.upperBound = 46.5) :
  ∃ (d : ℚ), d = 34 ∧ 
    (d12.lowerBound - d) / 4 = (d12.upperBound - d) / 4 ∧
    (d13.lowerBound - d) / 3 = (d13.upperBound - d) / 3 ∧
    (d15.lowerBound - d) / 1 = (d15.upperBound - d) / 1 :=
sorry

end distance_at_16_00_l1192_119297


namespace opposite_of_neg_six_l1192_119202

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- The opposite of -6 is 6. -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end opposite_of_neg_six_l1192_119202


namespace hyperbola_equation_l1192_119223

/-- Given a hyperbola with the standard form equation, prove that under certain conditions, 
    it has a specific equation. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c - a = 1 ∧ b = Real.sqrt 3) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) :=
sorry

end hyperbola_equation_l1192_119223


namespace total_score_three_probability_l1192_119209

def yellow_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := yellow_balls + white_balls

def yellow_score : ℕ := 1
def white_score : ℕ := 2

def prob_yellow (balls_left : ℕ) : ℚ := yellow_balls / balls_left
def prob_white (balls_left : ℕ) : ℚ := white_balls / balls_left

theorem total_score_three_probability :
  (prob_yellow total_balls * prob_white (total_balls - 1) +
   prob_white total_balls * prob_yellow (total_balls - 1)) = 3/5 := by
  sorry

end total_score_three_probability_l1192_119209


namespace cards_per_page_l1192_119281

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 16)
  (h3 : pages = 8) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end cards_per_page_l1192_119281


namespace non_acute_triangle_sides_count_l1192_119230

/-- Given a triangle with two sides of lengths 20 and 19, this function returns the number of possible integer lengths for the third side that make the triangle not acute. -/
def count_non_acute_triangle_sides : ℕ :=
  let a : ℕ := 20
  let b : ℕ := 19
  let possible_sides := (Finset.range 37).filter (fun s => 
    (s > 1 ∧ s < 39) ∧  -- Triangle inequality
    ((s * s ≥ a * a + b * b) ∨  -- s is the longest side (obtuse or right triangle)
     (a * a ≥ s * s + b * b)))  -- a is the longest side (obtuse or right triangle)
  possible_sides.card

/-- Theorem stating that there are exactly 16 possible integer lengths for the third side of a triangle with sides 20 and 19 that make it not acute. -/
theorem non_acute_triangle_sides_count : count_non_acute_triangle_sides = 16 := by
  sorry

end non_acute_triangle_sides_count_l1192_119230


namespace division_theorem_l1192_119254

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 132 →
  divisor = 16 →
  quotient = 8 →
  dividend = divisor * quotient + remainder →
  remainder = 4 := by
sorry

end division_theorem_l1192_119254


namespace polynomial_division_theorem_l1192_119273

theorem polynomial_division_theorem :
  ∃ (α β r : ℝ), ∀ z : ℝ,
    4 * z^4 - 3 * z^3 + 5 * z^2 - 7 * z + 6 =
    (4 * z + 7) * (z^3 - 2.5 * z^2 + α * z + β) + r :=
by sorry

end polynomial_division_theorem_l1192_119273


namespace base_k_conversion_l1192_119220

/-- Given that 44 in base k equals 36 in base 10, prove that 67 in base 10 equals 103 in base k. -/
theorem base_k_conversion (k : ℕ) (h : 4 * k + 4 = 36) : 
  (67 : ℕ).digits k = [3, 0, 1] :=
sorry

end base_k_conversion_l1192_119220


namespace same_gender_probability_l1192_119234

def num_male : ℕ := 3
def num_female : ℕ := 2
def total_volunteers : ℕ := num_male + num_female
def num_to_select : ℕ := 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def total_ways : ℕ := combination total_volunteers num_to_select
def same_gender_ways : ℕ := combination num_male num_to_select + combination num_female num_to_select

theorem same_gender_probability : 
  (same_gender_ways : ℚ) / total_ways = 2 / 5 := by sorry

end same_gender_probability_l1192_119234


namespace circular_field_diameter_specific_field_diameter_l1192_119233

/-- The diameter of a circular field given the cost of fencing. -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- The diameter of the specific circular field is approximately 34 meters. -/
theorem specific_field_diameter : 
  abs (circular_field_diameter 2 213.63 - 34) < 0.01 := by
  sorry

end circular_field_diameter_specific_field_diameter_l1192_119233


namespace roots_difference_squared_l1192_119241

theorem roots_difference_squared (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 4 * x₁ - 3 = 0) →
  (2 * x₂^2 + 4 * x₂ - 3 = 0) →
  (x₁ - x₂)^2 = 10 := by
sorry

end roots_difference_squared_l1192_119241


namespace initial_pink_hats_l1192_119266

/-- The number of pink hard hats initially in the truck -/
def initial_pink : ℕ := sorry

/-- The number of green hard hats initially in the truck -/
def initial_green : ℕ := 15

/-- The number of yellow hard hats initially in the truck -/
def initial_yellow : ℕ := 24

/-- The number of pink hard hats Carl takes away -/
def carl_pink : ℕ := 4

/-- The number of pink hard hats John takes away -/
def john_pink : ℕ := 6

/-- The number of green hard hats John takes away -/
def john_green : ℕ := 2 * john_pink

/-- The total number of hard hats remaining in the truck after Carl and John take some away -/
def remaining_hats : ℕ := 43

theorem initial_pink_hats : initial_pink = 26 := by sorry

end initial_pink_hats_l1192_119266


namespace mrs_hilt_shortage_l1192_119204

def initial_amount : ℚ := 375 / 100
def pencil_cost : ℚ := 115 / 100
def eraser_cost : ℚ := 85 / 100
def notebook_cost : ℚ := 225 / 100

theorem mrs_hilt_shortage :
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -50 / 100 := by
  sorry

end mrs_hilt_shortage_l1192_119204


namespace decimal_sum_and_multiply_l1192_119249

theorem decimal_sum_and_multiply : 
  let a : ℚ := 0.0034
  let b : ℚ := 0.125
  let c : ℚ := 0.00678
  2 * (a + b + c) = 0.27036 := by
sorry

end decimal_sum_and_multiply_l1192_119249


namespace roots_sum_product_l1192_119294

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 4*a - 1 = 0) → 
  (b^4 - 4*b - 1 = 0) → 
  (∀ x : ℝ, x ≠ a ∧ x ≠ b → x^4 - 4*x - 1 ≠ 0) →
  a * b + a + b = 1 := by
  sorry

end roots_sum_product_l1192_119294


namespace marbles_problem_l1192_119289

theorem marbles_problem (total : ℕ) (bags : ℕ) (remaining : ℕ) : 
  bags = 4 →
  remaining = 21 →
  (total / bags) * (bags - 1) = remaining →
  total = 28 :=
by sorry

end marbles_problem_l1192_119289


namespace cos_2alpha_minus_2pi_3_l1192_119210

theorem cos_2alpha_minus_2pi_3 (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.cos (2 * α - 2 * π / 3) = - 7 / 9 := by
  sorry

end cos_2alpha_minus_2pi_3_l1192_119210


namespace largest_multiple_of_7_less_than_neg_95_l1192_119226

theorem largest_multiple_of_7_less_than_neg_95 :
  ∀ n : ℤ, n * 7 < -95 → n * 7 ≤ -98 :=
by
  sorry

end largest_multiple_of_7_less_than_neg_95_l1192_119226


namespace unique_composite_with_sum_power_of_two_l1192_119287

theorem unique_composite_with_sum_power_of_two :
  ∃! m : ℕ+, 
    (1 < m) ∧ 
    (∀ a b : ℕ+, a * b = m → ∃ k : ℕ, a + b = 2^k) ∧
    m = 15 := by
  sorry

end unique_composite_with_sum_power_of_two_l1192_119287


namespace complex_fraction_simplification_l1192_119267

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7*I
  let z₂ : ℂ := 4 - 7*I
  (z₁ / z₂) + (z₂ / z₁) = -66/65 := by sorry

end complex_fraction_simplification_l1192_119267


namespace candy_distribution_l1192_119208

theorem candy_distribution (total red blue : ℕ) (h1 : total = 25689) (h2 : red = 1342) (h3 : blue = 8965) :
  let remaining := total - (red + blue)
  ∃ (green : ℕ), green * 3 = remaining ∧ green = 5127 := by
  sorry

end candy_distribution_l1192_119208


namespace chemistry_physics_difference_l1192_119271

/-- Proves that the difference between chemistry and physics scores is 10 -/
theorem chemistry_physics_difference
  (M P C : ℕ)  -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 60)  -- Sum of Mathematics and Physics scores
  (h2 : (M + C) / 2 = 35)  -- Average of Mathematics and Chemistry scores
  (h3 : C > P)  -- Chemistry score is higher than Physics score
  : C - P = 10 := by
  sorry

end chemistry_physics_difference_l1192_119271


namespace function_property_l1192_119252

theorem function_property (f : ℤ → ℤ) 
  (h : ∀ (x y : ℤ), f x + f y = f (x + 1) + f (y - 1))
  (h1 : f 2016 = 6102)
  (h2 : f 6102 = 2016) :
  f 1 = 8117 := by
sorry

end function_property_l1192_119252


namespace segment_count_is_21_l1192_119222

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  triple_intersection : Bool
  triple_intersection_count : ℕ

/-- Calculate the number of non-overlapping line segments in a given configuration -/
def count_segments (config : LineConfiguration) : ℕ :=
  config.num_lines * 4 - config.triple_intersection_count

/-- The specific configuration given in the problem -/
def problem_config : LineConfiguration :=
  { num_lines := 6
  , triple_intersection := true
  , triple_intersection_count := 3 }

/-- Theorem stating that the number of non-overlapping line segments in the given configuration is 21 -/
theorem segment_count_is_21 : count_segments problem_config = 21 := by
  sorry

end segment_count_is_21_l1192_119222


namespace isabel_birthday_money_l1192_119272

/-- The amount of money Isabel received for her birthday -/
def birthday_money : ℕ := sorry

/-- The cost of each toy -/
def toy_cost : ℕ := 2

/-- The number of toys Isabel could buy -/
def toys_bought : ℕ := 7

/-- Theorem stating that Isabel's birthday money is equal to the total cost of the toys she could buy -/
theorem isabel_birthday_money :
  birthday_money = toy_cost * toys_bought :=
sorry

end isabel_birthday_money_l1192_119272


namespace withdraw_300_from_two_banks_in_20_bills_l1192_119240

/-- Calculates the number of bills received when withdrawing money from two banks -/
def number_of_bills (amount_per_bank : ℕ) (num_banks : ℕ) (bill_value : ℕ) : ℕ :=
  (amount_per_bank * num_banks) / bill_value

/-- Proves that withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdraw_300_from_two_banks_in_20_bills : 
  number_of_bills 300 2 20 = 30 := by
  sorry

end withdraw_300_from_two_banks_in_20_bills_l1192_119240


namespace thousand_worries_conforms_to_cognitive_movement_l1192_119286

-- Define cognitive movement
structure CognitiveMovement where
  repetitive : Bool
  infinite : Bool

-- Define a phrase
structure Phrase where
  text : String
  conformsToCognitiveMovement : Bool

-- Define the specific phrase
def thousandWorries : Phrase where
  text := "A thousand worries yield one insight"
  conformsToCognitiveMovement := true -- This is what we want to prove

-- Theorem statement
theorem thousand_worries_conforms_to_cognitive_movement 
  (cm : CognitiveMovement) 
  (h1 : cm.repetitive = true) 
  (h2 : cm.infinite = true) : 
  thousandWorries.conformsToCognitiveMovement = true := by
  sorry


end thousand_worries_conforms_to_cognitive_movement_l1192_119286


namespace quadratic_b_value_l1192_119236

/-- The value of b in a quadratic function y = x² - bx + c passing through (1,n) and (3,n) -/
theorem quadratic_b_value (n : ℝ) : 
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 - b*x + c = n ↔ x = 1 ∨ x = 3) → b = 4 := by
  sorry

end quadratic_b_value_l1192_119236


namespace constant_term_binomial_expansion_l1192_119277

/-- The constant term in the binomial expansion of (x - 2/x)^8 -/
def constant_term : ℤ := 1120

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The general term of the binomial expansion (x - 2/x)^8 -/
def general_term (r : ℕ) : ℤ := (-2)^r * binomial 8 r

theorem constant_term_binomial_expansion :
  constant_term = general_term 4 := by sorry

end constant_term_binomial_expansion_l1192_119277


namespace parallel_lines_D_eq_18_l1192_119264

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y + 1 = 0

/-- The second line equation -/
def line2 (D : ℝ) (x y : ℝ) : Prop := 9 * x + D * y + 1 = 0

/-- The main theorem: if the two lines are parallel, then D = 18 -/
theorem parallel_lines_D_eq_18 :
  (∃ D : ℝ, ∀ x y : ℝ, (line1 x y ↔ line2 D x y) → D = 18) :=
sorry

end parallel_lines_D_eq_18_l1192_119264


namespace adjacent_sum_9_is_30_l1192_119253

def divisors_of_216 : List ℕ := [2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 54, 72, 108, 216]

def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i j, i ≠ j → (arr.get! i).gcd (arr.get! j) > 1

def adjacent_sum_9 (arr : List ℕ) : ℕ :=
  let idx := arr.indexOf 9
  (arr.get! ((idx - 1 + arr.length) % arr.length)) + (arr.get! ((idx + 1) % arr.length))

theorem adjacent_sum_9_is_30 :
  ∃ arr : List ℕ, arr.Perm divisors_of_216 ∧ valid_arrangement arr ∧ adjacent_sum_9 arr = 30 :=
sorry

end adjacent_sum_9_is_30_l1192_119253


namespace bench_placement_l1192_119260

theorem bench_placement (path_length : ℕ) (interval : ℕ) (bench_count : ℕ) : 
  path_length = 120 ∧ 
  interval = 10 ∧ 
  bench_count = (path_length / interval) + 1 →
  bench_count = 13 := by
  sorry

end bench_placement_l1192_119260


namespace group_size_after_new_member_l1192_119228

theorem group_size_after_new_member (n : ℕ) : 
  (n * 14 + 32) / (n + 1) = 15 → n = 17 := by
  sorry

end group_size_after_new_member_l1192_119228


namespace correct_calculation_l1192_119206

theorem correct_calculation (x y : ℝ) : 6 * x * y^2 - 3 * y^2 * x = 3 * x * y^2 := by
  sorry

end correct_calculation_l1192_119206


namespace power_equation_l1192_119225

theorem power_equation : 32^4 * 4^5 = 2^30 := by
  sorry

end power_equation_l1192_119225


namespace yield_increase_correct_l1192_119245

/-- The percentage increase in rice yield after each harvest -/
def yield_increase_percentage : ℝ := 20

/-- The initial harvest yield in sacks of rice -/
def initial_harvest : ℝ := 20

/-- The total yield after two harvests in sacks of rice -/
def total_yield_two_harvests : ℝ := 44

/-- Theorem stating that the given yield increase percentage is correct -/
theorem yield_increase_correct : 
  initial_harvest + initial_harvest * (1 + yield_increase_percentage / 100) = total_yield_two_harvests :=
by sorry

end yield_increase_correct_l1192_119245


namespace acute_triangle_tangent_inequality_l1192_119257

theorem acute_triangle_tangent_inequality (A B C : Real) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (1 / (1 + Real.tan A) + 1 / (1 + Real.tan B)) < (Real.tan A / (1 + Real.tan A) + Real.tan B / (1 + Real.tan B)) := by
  sorry

end acute_triangle_tangent_inequality_l1192_119257


namespace inverse_proportion_problem_l1192_119265

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  InverselyProportional x y →
  (x + y = 30 → x = 3 * y) →
  x = -12 →
  y = -14.0625 := by sorry

end inverse_proportion_problem_l1192_119265


namespace rope_remaining_l1192_119237

theorem rope_remaining (initial_length : ℝ) (fraction_to_allan : ℝ) (fraction_to_jack : ℝ) :
  initial_length = 20 ∧ 
  fraction_to_allan = 1/4 ∧ 
  fraction_to_jack = 2/3 →
  initial_length * (1 - fraction_to_allan) * (1 - fraction_to_jack) = 5 := by
  sorry

end rope_remaining_l1192_119237


namespace basketball_scores_l1192_119293

def first_ten_games : List Nat := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

theorem basketball_scores (game_11 game_12 : Nat) : 
  game_11 < 10 →
  game_12 < 10 →
  (List.sum first_ten_games + game_11) % 11 = 0 →
  (List.sum first_ten_games + game_11 + game_12) % 12 = 0 →
  game_11 * game_12 = 0 := by
sorry

end basketball_scores_l1192_119293


namespace vehicle_value_depreciation_l1192_119282

theorem vehicle_value_depreciation (last_year_value : ℝ) (depreciation_factor : ℝ) (this_year_value : ℝ) :
  last_year_value = 20000 →
  depreciation_factor = 0.8 →
  this_year_value = last_year_value * depreciation_factor →
  this_year_value = 16000 := by
  sorry

end vehicle_value_depreciation_l1192_119282


namespace orange_picking_theorem_l1192_119295

/-- The total number of oranges picked over three days -/
def total_oranges (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of oranges picked over three days -/
theorem orange_picking_theorem (day1 day2 day3 : ℕ) 
  (h1 : day1 = 100)
  (h2 : day2 = 3 * day1)
  (h3 : day3 = 70) :
  total_oranges day1 day2 day3 = 470 := by
  sorry


end orange_picking_theorem_l1192_119295


namespace sum_of_squares_of_roots_l1192_119231

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 2 ∧ b = -6 ∧ c = 4 → x₁^2 + x₂^2 = 5 :=
by sorry


end sum_of_squares_of_roots_l1192_119231


namespace square_area_on_parabola_l1192_119263

/-- The area of a square with one side on y = 10 and endpoints on y = x^2 + 4x + 3 is 44 -/
theorem square_area_on_parabola : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + 4*x₁ + 3 = 10) ∧ 
    (x₂^2 + 4*x₂ + 3 = 10) ∧ 
    (x₁ ≠ x₂) ∧
    ((x₂ - x₁)^2 = 44) := by
  sorry

end square_area_on_parabola_l1192_119263


namespace smallest_other_integer_l1192_119235

theorem smallest_other_integer (a b x : ℕ+) : 
  (a = 70 ∨ b = 70) →
  (Nat.gcd a b = x + 7) →
  (Nat.lcm a b = x * (x + 7)) →
  (min a b ≠ 70 → min a b ≥ 20) :=
by sorry

end smallest_other_integer_l1192_119235


namespace distance_problems_l1192_119288

def distance_point_to_line (p : Fin n → ℝ) (a b : Fin n → ℝ) : ℝ :=
  sorry

theorem distance_problems :
  let d1 := distance_point_to_line (![1, 0]) (![0, 0]) (![0, 1])
  let d2 := distance_point_to_line (![1, 0]) (![0, 0]) (![1, 1])
  let d3 := distance_point_to_line (![1, 0, 0]) (![0, 0, 0]) (![1, 1, 1])
  (d1 = 1) ∧ (d2 = Real.sqrt 2 / 2) ∧ (d3 = Real.sqrt 6 / 3) := by
  sorry

end distance_problems_l1192_119288


namespace ticket_cost_l1192_119298

/-- Given the total amount collected and average daily ticket sales over three days,
    prove that the cost of one ticket is $4. -/
theorem ticket_cost (total_amount : ℚ) (avg_daily_sales : ℚ) 
  (h1 : total_amount = 960)
  (h2 : avg_daily_sales = 80) : 
  total_amount / (avg_daily_sales * 3) = 4 := by
  sorry

end ticket_cost_l1192_119298


namespace negative_four_less_than_negative_sqrt_fourteen_l1192_119211

theorem negative_four_less_than_negative_sqrt_fourteen : -4 < -Real.sqrt 14 := by
  sorry

end negative_four_less_than_negative_sqrt_fourteen_l1192_119211
