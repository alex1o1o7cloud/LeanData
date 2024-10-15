import Mathlib

namespace NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l1151_115131

theorem mashed_potatoes_suggestion (bacon : ℕ) (tomatoes : ℕ) (total : ℕ) 
  (h1 : bacon = 374) 
  (h2 : tomatoes = 128) 
  (h3 : total = 826) :
  total - (bacon + tomatoes) = 324 :=
by sorry

end NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l1151_115131


namespace NUMINAMATH_CALUDE_paper_I_max_mark_l1151_115150

/-- The maximum mark for Paper I -/
def max_mark : ℕ := 262

/-- The passing percentage for Paper I -/
def passing_percentage : ℚ := 65 / 100

/-- The marks scored by the candidate -/
def scored_marks : ℕ := 112

/-- The marks by which the candidate failed -/
def failed_by : ℕ := 58

/-- Theorem stating that the maximum mark for Paper I is 262 -/
theorem paper_I_max_mark :
  (↑max_mark * passing_percentage).floor = scored_marks + failed_by :=
sorry

end NUMINAMATH_CALUDE_paper_I_max_mark_l1151_115150


namespace NUMINAMATH_CALUDE_cristina_photos_l1151_115171

theorem cristina_photos (total_slots : ℕ) (john_photos sarah_photos clarissa_photos : ℕ) 
  (h1 : total_slots = 40)
  (h2 : john_photos = 10)
  (h3 : sarah_photos = 9)
  (h4 : clarissa_photos = 14)
  (h5 : ∃ (cristina_photos : ℕ), cristina_photos + john_photos + sarah_photos + clarissa_photos = total_slots) :
  ∃ (cristina_photos : ℕ), cristina_photos = 7 := by
sorry

end NUMINAMATH_CALUDE_cristina_photos_l1151_115171


namespace NUMINAMATH_CALUDE_problem_solution_l1151_115180

theorem problem_solution (x y z : ℝ) (hx : x = 7) (hy : y = -2) (hz : z = 4) :
  ((x - 2*y)^y) / z = 1 / 484 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1151_115180


namespace NUMINAMATH_CALUDE_conjugate_complex_equation_l1151_115145

/-- Two complex numbers are conjugates if their real parts are equal and their imaginary parts are opposites -/
def are_conjugates (a b : ℂ) : Prop := a.re = b.re ∧ a.im = -b.im

/-- The main theorem -/
theorem conjugate_complex_equation (a b : ℂ) :
  are_conjugates a b → (a + b)^2 - 3 * a * b * I = 4 - 6 * I →
  ((a = 1 + I ∧ b = 1 - I) ∨
   (a = -1 - I ∧ b = -1 + I) ∨
   (a = 1 - I ∧ b = 1 + I) ∨
   (a = -1 + I ∧ b = -1 - I)) :=
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_equation_l1151_115145


namespace NUMINAMATH_CALUDE_pq_length_is_1098_over_165_l1151_115191

/-- The line y = (5/3)x --/
def line1 (x y : ℝ) : Prop := y = (5/3) * x

/-- The line y = (5/12)x --/
def line2 (x y : ℝ) : Prop := y = (5/12) * x

/-- The midpoint of two points --/
def is_midpoint (mx my px py qx qy : ℝ) : Prop :=
  mx = (px + qx) / 2 ∧ my = (py + qy) / 2

/-- The squared distance between two points --/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

theorem pq_length_is_1098_over_165 :
  ∀ (px py qx qy : ℝ),
    line1 px py →
    line2 qx qy →
    is_midpoint 10 8 px py qx qy →
    distance_squared px py qx qy = (1098/165)^2 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_is_1098_over_165_l1151_115191


namespace NUMINAMATH_CALUDE_patty_weight_factor_l1151_115103

/-- Given:
  - Robbie weighs 100 pounds
  - Patty was initially x times as heavy as Robbie
  - Patty lost 235 pounds
  - After weight loss, Patty weighs 115 pounds more than Robbie
Prove that x = 4.5 -/
theorem patty_weight_factor (x : ℝ) 
  (robbie_weight : ℝ) (patty_weight_loss : ℝ) (patty_final_difference : ℝ)
  (h1 : robbie_weight = 100)
  (h2 : patty_weight_loss = 235)
  (h3 : patty_final_difference = 115)
  (h4 : x * robbie_weight - patty_weight_loss = robbie_weight + patty_final_difference) :
  x = 4.5 := by
sorry

end NUMINAMATH_CALUDE_patty_weight_factor_l1151_115103


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1151_115138

theorem average_speed_calculation (distance : ℝ) (time : ℝ) (average_speed : ℝ) :
  distance = 210 →
  time = 4.5 →
  average_speed = distance / time →
  average_speed = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1151_115138


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1151_115104

theorem solve_linear_equation :
  ∃ x : ℝ, 45 - 3 * x = 12 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1151_115104


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1151_115184

theorem isosceles_triangle (a b c : ℝ) (α β γ : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β) →
  α = β := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1151_115184


namespace NUMINAMATH_CALUDE_twentieth_digit_of_half_power_twenty_l1151_115165

theorem twentieth_digit_of_half_power_twenty (n : ℕ) : n = 20 → 
  ∃ (x : ℚ), x = (1/2)^20 ∧ 
  (∃ (a b : ℕ), x = a / (10^n) ∧ x < (a + 1) / (10^n) ∧ a % 10 = 1) :=
sorry

end NUMINAMATH_CALUDE_twentieth_digit_of_half_power_twenty_l1151_115165


namespace NUMINAMATH_CALUDE_dice_product_probability_composite_probability_l1151_115187

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of possible outcomes when rolling a 6-sided die -/
def dieOutcomes : Finset ℕ := sorry

/-- The set of all possible outcomes when rolling 4 dice -/
def allOutcomes : Finset (ℕ × ℕ × ℕ × ℕ) := sorry

/-- The product of the numbers in a 4-tuple -/
def product (t : ℕ × ℕ × ℕ × ℕ) : ℕ := sorry

/-- The set of outcomes that result in a non-composite product -/
def nonCompositeOutcomes : Finset (ℕ × ℕ × ℕ × ℕ) := sorry

theorem dice_product_probability :
  (Finset.card nonCompositeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 13 / 1296 :=
sorry

theorem composite_probability :
  1 - (Finset.card nonCompositeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 1283 / 1296 :=
sorry

end NUMINAMATH_CALUDE_dice_product_probability_composite_probability_l1151_115187


namespace NUMINAMATH_CALUDE_smallest_cube_box_for_pyramid_l1151_115107

theorem smallest_cube_box_for_pyramid (pyramid_height base_length base_width : ℝ) 
  (h_height : pyramid_height = 15)
  (h_base_length : base_length = 9)
  (h_base_width : base_width = 12) :
  let box_side := max pyramid_height (max base_length base_width)
  (box_side ^ 3 : ℝ) = 3375 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_box_for_pyramid_l1151_115107


namespace NUMINAMATH_CALUDE_census_objects_eq_population_l1151_115101

/-- The entirety of objects under investigation in a census -/
def census_objects : Type := Unit

/-- The term "population" in statistical context -/
def population : Type := Unit

/-- Theorem stating that census objects are equivalent to population -/
theorem census_objects_eq_population : census_objects ≃ population := sorry

end NUMINAMATH_CALUDE_census_objects_eq_population_l1151_115101


namespace NUMINAMATH_CALUDE_intersection_product_l1151_115109

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 10*y + 21 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 52 = 0

/-- Intersection point of the two circles -/
def intersection_point (x y : ℝ) : Prop :=
  circle1 x y ∧ circle2 x y

/-- The theorem stating that the product of all coordinates of intersection points is 189 -/
theorem intersection_product : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁ * y₁ * x₂ * y₂ = 189 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_l1151_115109


namespace NUMINAMATH_CALUDE_region_is_lower_left_l1151_115179

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Define what it means to be on the lower left side of the line
def lower_left_side (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Theorem statement
theorem region_is_lower_left : 
  ∀ (x y : ℝ), region x y → lower_left_side x y :=
sorry

end NUMINAMATH_CALUDE_region_is_lower_left_l1151_115179


namespace NUMINAMATH_CALUDE_divisibility_by_37_l1151_115198

def N (x y : ℕ) : ℕ := 300070003 + 1000000 * x + 100 * y

theorem divisibility_by_37 :
  ∀ x y : ℕ, x ≤ 9 ∧ y ≤ 9 →
  (37 ∣ N x y ↔ (x = 8 ∧ y = 1) ∨ (x = 4 ∧ y = 4) ∨ (x = 0 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l1151_115198


namespace NUMINAMATH_CALUDE_residue_11_2048_mod_17_l1151_115116

theorem residue_11_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_residue_11_2048_mod_17_l1151_115116


namespace NUMINAMATH_CALUDE_ab_value_l1151_115106

theorem ab_value (a b : ℝ) (h1 : a * Real.exp a = Real.exp 2) (h2 : Real.log (b / Real.exp 1) = Real.exp 3 / b) : a * b = Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1151_115106


namespace NUMINAMATH_CALUDE_range_of_f_l1151_115151

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 18) / Real.log (1/3)

theorem range_of_f :
  Set.range f = Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1151_115151


namespace NUMINAMATH_CALUDE_vector_from_origin_to_line_l1151_115102

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line -/
def givenLine : ParametricLine where
  x := λ t => 4 * t + 2
  y := λ t => t + 2

/-- Check if a vector is parallel to another vector -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Check if a point lies on the given line -/
def liesOnLine (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p.1 = givenLine.x t ∧ p.2 = givenLine.y t

theorem vector_from_origin_to_line :
  liesOnLine (6, 3) ∧ isParallel (6, 3) (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_from_origin_to_line_l1151_115102


namespace NUMINAMATH_CALUDE_zeros_imply_a_range_l1151_115105

/-- The function h(x) = ax² - x - ln(x) has two distinct zeros -/
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
  a * x₁^2 - x₁ - Real.log x₁ = 0 ∧
  a * x₂^2 - x₂ - Real.log x₂ = 0

/-- If h(x) has two distinct zeros, then 0 < a < 1 -/
theorem zeros_imply_a_range (a : ℝ) (h : a ≠ 0) :
  has_two_distinct_zeros a → 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_zeros_imply_a_range_l1151_115105


namespace NUMINAMATH_CALUDE_largest_undefined_x_l1151_115173

theorem largest_undefined_x : 
  let f (x : ℝ) := 10 * x^2 - 30 * x + 20
  ∃ (max : ℝ), f max = 0 ∧ ∀ x, f x = 0 → x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_largest_undefined_x_l1151_115173


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1151_115190

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1151_115190


namespace NUMINAMATH_CALUDE_prob_first_success_third_trial_l1151_115115

/-- Probability of first success on third trial in a geometric distribution -/
theorem prob_first_success_third_trial (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  let q := 1 - p
  (q ^ 2) * p = p * (1 - p) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_first_success_third_trial_l1151_115115


namespace NUMINAMATH_CALUDE_school_start_time_proof_l1151_115143

structure SchoolCommute where
  normalTime : ℕ
  redLightStops : ℕ
  redLightTime : ℕ
  constructionTime : ℕ
  departureTime : Nat × Nat
  lateMinutes : ℕ

def schoolStartTime (c : SchoolCommute) : Nat × Nat :=
  sorry

theorem school_start_time_proof (c : SchoolCommute) 
  (h1 : c.normalTime = 30)
  (h2 : c.redLightStops = 4)
  (h3 : c.redLightTime = 3)
  (h4 : c.constructionTime = 10)
  (h5 : c.departureTime = (7, 15))
  (h6 : c.lateMinutes = 7) :
  schoolStartTime c = (8, 0) :=
by
  sorry

end NUMINAMATH_CALUDE_school_start_time_proof_l1151_115143


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1151_115133

theorem lcm_of_ratio_and_hcf (a b : ℕ+) :
  (a : ℚ) / b = 14 / 21 →
  Nat.gcd a b = 28 →
  Nat.lcm a b = 1176 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1151_115133


namespace NUMINAMATH_CALUDE_point_d_coordinates_l1151_115162

/-- Given a line segment AB with endpoints A(-3, 2) and B(5, 10), and a point D on AB
    such that AD = 2DB, and the slope of AB is 1, prove that the coordinates of D are (7/3, 22/3). -/
theorem point_d_coordinates :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (5, 10)
  let D : ℝ × ℝ := (x, y)
  ∀ x y : ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B) →  -- D is on segment AB
    (x - (-3))^2 + (y - 2)^2 = 4 * ((5 - x)^2 + (10 - y)^2) →  -- AD = 2DB
    (10 - 2) / (5 - (-3)) = 1 →  -- Slope of AB is 1
    D = (7/3, 22/3) :=
by
  sorry

end NUMINAMATH_CALUDE_point_d_coordinates_l1151_115162


namespace NUMINAMATH_CALUDE_min_c_over_d_l1151_115137

theorem min_c_over_d (x C D : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)
  (eq1 : x^2 + 1/x^2 = C) (eq2 : x + 1/x = D) : 
  ∀ y : ℝ, y > 0 → y^2 + 1/y^2 = C → y + 1/y = D → C / D ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_c_over_d_l1151_115137


namespace NUMINAMATH_CALUDE_furniture_purchase_price_l1151_115112

theorem furniture_purchase_price 
  (marked_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (purchase_price : ℝ) : 
  marked_price = 132 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.1 ∧ 
  marked_price * (1 - discount_rate) = purchase_price * (1 + profit_rate) → 
  purchase_price = 108 := by
sorry

end NUMINAMATH_CALUDE_furniture_purchase_price_l1151_115112


namespace NUMINAMATH_CALUDE_staff_avg_age_l1151_115118

def robotics_camp (total_members : ℕ) (overall_avg_age : ℝ)
  (num_girls num_boys num_adults num_staff : ℕ)
  (avg_age_girls avg_age_boys avg_age_adults : ℝ) : Prop :=
  total_members = 50 ∧
  overall_avg_age = 20 ∧
  num_girls = 22 ∧
  num_boys = 18 ∧
  num_adults = 5 ∧
  num_staff = 5 ∧
  avg_age_girls = 18 ∧
  avg_age_boys = 19 ∧
  avg_age_adults = 30

theorem staff_avg_age
  (h : robotics_camp 50 20 22 18 5 5 18 19 30) :
  (50 * 20 - (22 * 18 + 18 * 19 + 5 * 30)) / 5 = 22.4 :=
by sorry

end NUMINAMATH_CALUDE_staff_avg_age_l1151_115118


namespace NUMINAMATH_CALUDE_tan_sum_eq_two_l1151_115156

theorem tan_sum_eq_two (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_eq_two_l1151_115156


namespace NUMINAMATH_CALUDE_sqrt_sum_geq_product_sum_l1151_115155

theorem sqrt_sum_geq_product_sum {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : Real.sqrt x + Real.sqrt y + Real.sqrt z ≥ x * y + y * z + z * x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_geq_product_sum_l1151_115155


namespace NUMINAMATH_CALUDE_expand_expression_l1151_115163

theorem expand_expression (x y : ℝ) : (3*x - 5) * (4*y + 20) = 12*x*y + 60*x - 20*y - 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1151_115163


namespace NUMINAMATH_CALUDE_convention_handshakes_eq_680_l1151_115157

/-- Represents the number of handshakes at a twins and quadruplets convention --/
def convention_handshakes : ℕ := by
  -- Define the number of twin sets and quadruplet sets
  let twin_sets : ℕ := 8
  let quad_sets : ℕ := 5

  -- Calculate total number of twins and quadruplets
  let total_twins : ℕ := twin_sets * 2
  let total_quads : ℕ := quad_sets * 4

  -- Calculate handshakes among twins
  let twin_handshakes : ℕ := (total_twins * (total_twins - 2)) / 2

  -- Calculate handshakes among quadruplets
  let quad_handshakes : ℕ := (total_quads * (total_quads - 4)) / 2

  -- Calculate cross handshakes between twins and quadruplets
  let cross_handshakes : ℕ := total_twins * (2 * total_quads / 3)

  -- Sum all handshakes
  exact twin_handshakes + quad_handshakes + cross_handshakes

/-- Theorem stating that the total number of handshakes is 680 --/
theorem convention_handshakes_eq_680 : convention_handshakes = 680 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_eq_680_l1151_115157


namespace NUMINAMATH_CALUDE_stock_price_calculation_l1151_115168

/-- Calculates the final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

/-- Theorem stating that given the specific conditions, the final stock price is $151.2 -/
theorem stock_price_calculation :
  final_stock_price 120 0.8 0.3 = 151.2 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l1151_115168


namespace NUMINAMATH_CALUDE_nonparallel_side_length_l1151_115158

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- Radius of the circle -/
  r : ℝ
  /-- Length of each parallel side -/
  a : ℝ
  /-- Length of each non-parallel side -/
  x : ℝ
  /-- The trapezoid is inscribed in the circle -/
  inscribed : True
  /-- The parallel sides are equal -/
  parallel_equal : True
  /-- The non-parallel sides are equal -/
  nonparallel_equal : True

/-- Theorem stating the length of non-parallel sides in the specific trapezoid -/
theorem nonparallel_side_length (t : InscribedTrapezoid) 
  (h1 : t.r = 300) 
  (h2 : t.a = 150) : 
  t.x = 300 := by
  sorry

end NUMINAMATH_CALUDE_nonparallel_side_length_l1151_115158


namespace NUMINAMATH_CALUDE_convex_polygons_contain_center_l1151_115113

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a convex polygon
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool

-- Define a function to check if a point is inside a polygon
def is_inside (p : ℝ × ℝ) (poly : ConvexPolygon) : Prop :=
  sorry

-- Define a function to check if a polygon is inside a square
def is_inside_square (poly : ConvexPolygon) (sq : Square) : Prop :=
  sorry

-- Theorem statement
theorem convex_polygons_contain_center 
  (sq : Square) 
  (poly1 poly2 poly3 : ConvexPolygon) 
  (h1 : is_inside_square poly1 sq)
  (h2 : is_inside_square poly2 sq)
  (h3 : is_inside_square poly3 sq) :
  is_inside sq.center poly1 ∧ is_inside sq.center poly2 ∧ is_inside sq.center poly3 :=
sorry

end NUMINAMATH_CALUDE_convex_polygons_contain_center_l1151_115113


namespace NUMINAMATH_CALUDE_range_of_a_l1151_115111

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 2)*x + 1 ≥ 0) → 0 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1151_115111


namespace NUMINAMATH_CALUDE_candy_problem_l1151_115197

def candy_remaining (initial : ℕ) (day : ℕ) : ℚ :=
  match day with
  | 0 => initial
  | 1 => initial / 2
  | 2 => initial / 2 * (1 / 3)
  | 3 => initial / 2 * (1 / 3) * (1 / 4)
  | 4 => initial / 2 * (1 / 3) * (1 / 4) * (1 / 5)
  | 5 => initial / 2 * (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6)
  | _ => 0

theorem candy_problem (initial : ℕ) :
  candy_remaining initial 5 = 1 ↔ initial = 720 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l1151_115197


namespace NUMINAMATH_CALUDE_elite_cheaper_at_min_shirts_l1151_115108

/-- Elite T-Shirt Company's pricing structure -/
def elite_cost (n : ℕ) : ℚ := 30 + 8 * n

/-- Omega T-Shirt Company's pricing structure -/
def omega_cost (n : ℕ) : ℚ := 10 + 12 * n

/-- The minimum number of shirts for which Elite is cheaper than Omega -/
def min_shirts_for_elite : ℕ := 6

theorem elite_cheaper_at_min_shirts :
  elite_cost min_shirts_for_elite < omega_cost min_shirts_for_elite ∧
  ∀ k : ℕ, k < min_shirts_for_elite → elite_cost k ≥ omega_cost k :=
by sorry

end NUMINAMATH_CALUDE_elite_cheaper_at_min_shirts_l1151_115108


namespace NUMINAMATH_CALUDE_total_plants_l1151_115167

def garden_problem (basil oregano thyme rosemary : ℕ) : Prop :=
  oregano = 2 * basil + 2 ∧
  thyme = 3 * basil - 3 ∧
  rosemary = (basil + thyme) / 2 ∧
  basil = 5 ∧
  basil + oregano + thyme + rosemary ≤ 50

theorem total_plants (basil oregano thyme rosemary : ℕ) :
  garden_problem basil oregano thyme rosemary →
  basil + oregano + thyme + rosemary = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_total_plants_l1151_115167


namespace NUMINAMATH_CALUDE_platform_length_l1151_115175

/-- The length of the platform given a train's characteristics and crossing times. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 45)
  (h3 : time_pole = 18) :
  let speed := train_length / time_pole
  let total_distance := speed * time_platform
  train_length + (total_distance - train_length) = 450 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1151_115175


namespace NUMINAMATH_CALUDE_sum_smallest_largest_primes_1_to_50_l1151_115188

theorem sum_smallest_largest_primes_1_to_50 :
  (∃ p q : ℕ, 
    Prime p ∧ Prime q ∧
    1 < p ∧ p ≤ 50 ∧
    1 < q ∧ q ≤ 50 ∧
    (∀ r : ℕ, Prime r ∧ 1 < r ∧ r ≤ 50 → p ≤ r ∧ r ≤ q) ∧
    p + q = 49) :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_primes_1_to_50_l1151_115188


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l1151_115140

-- Define a 1962-digit number
def is_1962_digit_number (n : ℕ) : Prop :=
  10^1961 ≤ n ∧ n < 10^1962

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

-- Define the property of being divisible by 9
def divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

-- State the theorem
theorem sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9 
  (n : ℕ) (a b c : ℕ) : 
  is_1962_digit_number n → 
  divisible_by_9 n → 
  a = sum_of_digits n → 
  b = sum_of_digits a → 
  c = sum_of_digits b → 
  c = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l1151_115140


namespace NUMINAMATH_CALUDE_set_M_membership_l1151_115127

def M : Set ℕ := {x : ℕ | (1 : ℚ) / (x - 2 : ℚ) ≤ 0}

theorem set_M_membership :
  1 ∈ M ∧ 2 ∉ M ∧ 3 ∉ M ∧ 4 ∉ M :=
by sorry

end NUMINAMATH_CALUDE_set_M_membership_l1151_115127


namespace NUMINAMATH_CALUDE_emily_calculation_l1151_115146

theorem emily_calculation (n : ℕ) : n = 42 → (n + 1)^2 = n^2 + 85 → (n - 1)^2 = n^2 - 83 := by
  sorry

end NUMINAMATH_CALUDE_emily_calculation_l1151_115146


namespace NUMINAMATH_CALUDE_point_on_circle_x_value_l1151_115170

/-- A circle in the xy-plane with diameter endpoints (-3,0) and (21,0) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  h1 : center = (9, 0)
  h2 : radius = 12

/-- A point on the circle -/
structure PointOnCircle (c : Circle) where
  x : ℝ
  y : ℝ
  h : (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem point_on_circle_x_value (c : Circle) (p : PointOnCircle c) (h : p.y = 12) :
  p.x = 9 := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_x_value_l1151_115170


namespace NUMINAMATH_CALUDE_cone_volume_l1151_115174

/-- Given a cone with base radius 3 and lateral surface area 15π, its volume is 12π. -/
theorem cone_volume (r h : ℝ) : 
  r = 3 → 
  π * r * (r^2 + h^2).sqrt = 15 * π → 
  (1/3) * π * r^2 * h = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1151_115174


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1151_115134

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of marbles -/
def total_marbles : ℕ := 16

/-- The number of colored marbles -/
def colored_marbles : ℕ := 4

/-- The number of non-colored marbles -/
def non_colored_marbles : ℕ := total_marbles - colored_marbles

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 5

/-- The number of colored marbles to be chosen -/
def chosen_colored : ℕ := 2

/-- The number of non-colored marbles to be chosen -/
def chosen_non_colored : ℕ := chosen_marbles - chosen_colored

theorem marble_selection_ways :
  choose colored_marbles chosen_colored * choose non_colored_marbles chosen_non_colored = 1320 :=
sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1151_115134


namespace NUMINAMATH_CALUDE_calcium_oxide_weight_l1151_115169

-- Define atomic weights
def atomic_weight_Ca : Real := 40.08
def atomic_weight_O : Real := 16.00

-- Define the compound
structure Compound where
  calcium : Nat
  oxygen : Nat

-- Define molecular weight calculation
def molecular_weight (c : Compound) : Real :=
  c.calcium * atomic_weight_Ca + c.oxygen * atomic_weight_O

-- Theorem to prove
theorem calcium_oxide_weight :
  molecular_weight { calcium := 1, oxygen := 1 } = 56.08 := by
  sorry

end NUMINAMATH_CALUDE_calcium_oxide_weight_l1151_115169


namespace NUMINAMATH_CALUDE_tan_2_implies_sum_23_10_l1151_115147

theorem tan_2_implies_sum_23_10 (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ ^ 2 = 23 / 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_2_implies_sum_23_10_l1151_115147


namespace NUMINAMATH_CALUDE_distance_calculation_l1151_115186

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 24

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's and Brad's start times in hours -/
def time_difference : ℝ := 1

/-- Total time Maxwell walks before meeting Brad in hours -/
def total_time : ℝ := 3

theorem distance_calculation :
  distance_between_homes = maxwell_speed * total_time + brad_speed * (total_time - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l1151_115186


namespace NUMINAMATH_CALUDE_expression_factorization_l1151_115136

theorem expression_factorization (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1151_115136


namespace NUMINAMATH_CALUDE_f_injective_f_property_inverse_f_512_l1151_115117

/-- A function satisfying f(5) = 2 and f(2x) = 2f(x) for all x -/
def f : ℝ → ℝ :=
  sorry

/-- f is injective -/
theorem f_injective : Function.Injective f :=
  sorry

/-- The inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem f_property (x : ℝ) : f (2 * x) = 2 * f x :=
  sorry

axiom f_5 : f 5 = 2

/-- The main theorem: f⁻¹(512) = 1280 -/
theorem inverse_f_512 : f_inv 512 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_f_injective_f_property_inverse_f_512_l1151_115117


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l1151_115194

theorem shaded_fraction_of_rectangle : 
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 20
  let total_area : ℝ := rectangle_length * rectangle_width
  let quarter_area : ℝ := total_area / 4
  let shaded_area : ℝ := quarter_area / 2
  shaded_area / total_area = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l1151_115194


namespace NUMINAMATH_CALUDE_probability_theorem_l1151_115139

/-- The probability of selecting exactly one high-quality item and one defective item
    from a set of 4 high-quality items and 1 defective item, when two items are randomly selected. -/
def probability_one_high_quality_one_defective : ℚ := 2 / 5

/-- The number of high-quality items -/
def num_high_quality : ℕ := 4

/-- The number of defective items -/
def num_defective : ℕ := 1

/-- The total number of items -/
def total_items : ℕ := num_high_quality + num_defective

/-- The number of items to be selected -/
def items_to_select : ℕ := 2

/-- Theorem stating that the probability of selecting exactly one high-quality item
    and one defective item is 2/5 -/
theorem probability_theorem :
  probability_one_high_quality_one_defective =
    (num_high_quality.choose 1 * num_defective.choose 1 : ℚ) /
    (total_items.choose items_to_select : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1151_115139


namespace NUMINAMATH_CALUDE_divisor_problem_l1151_115176

theorem divisor_problem (d : ℕ) (h_pos : d > 0) :
  1200 % d = 3 ∧ 1640 % d = 2 ∧ 1960 % d = 7 → d = 9 ∨ d = 21 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1151_115176


namespace NUMINAMATH_CALUDE_expression_simplification_l1151_115159

theorem expression_simplification :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1151_115159


namespace NUMINAMATH_CALUDE_remainder_theorem_l1151_115181

theorem remainder_theorem (x : ℤ) : x % 63 = 25 → x % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1151_115181


namespace NUMINAMATH_CALUDE_f_minimum_when_a_is_one_f_nonnegative_iff_a_ge_one_l1151_115195

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 2 / x + a * x - a - 2

theorem f_minimum_when_a_is_one :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, f 1 x ≥ min :=
sorry

theorem f_nonnegative_iff_a_ge_one :
  ∀ a > 0, (∀ x ∈ Set.Icc 1 3, f a x ≥ 0) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_when_a_is_one_f_nonnegative_iff_a_ge_one_l1151_115195


namespace NUMINAMATH_CALUDE_cookie_jar_theorem_l1151_115161

def cookie_jar_problem (initial_amount doris_spent : ℕ) : Prop :=
  let martha_spent := doris_spent / 2
  let total_spent := doris_spent + martha_spent
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 15

theorem cookie_jar_theorem :
  cookie_jar_problem 24 6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_theorem_l1151_115161


namespace NUMINAMATH_CALUDE_tarantulas_needed_l1151_115172

/-- The number of legs for each animal type --/
def legs_per_chimp : ℕ := 4
def legs_per_lion : ℕ := 4
def legs_per_lizard : ℕ := 4
def legs_per_tarantula : ℕ := 8

/-- The number of animals already seen --/
def chimps_seen : ℕ := 12
def lions_seen : ℕ := 8
def lizards_seen : ℕ := 5

/-- The total number of legs Borgnine wants to see --/
def total_legs_goal : ℕ := 1100

/-- Theorem: The number of tarantulas needed to reach the total legs goal --/
theorem tarantulas_needed : 
  (chimps_seen * legs_per_chimp + 
   lions_seen * legs_per_lion + 
   lizards_seen * legs_per_lizard + 
   125 * legs_per_tarantula) = total_legs_goal :=
by sorry

end NUMINAMATH_CALUDE_tarantulas_needed_l1151_115172


namespace NUMINAMATH_CALUDE_ice_cream_volume_l1151_115135

/-- The volume of ice cream on a cone -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ)
  (h_cone_pos : h_cone > 0)
  (r_pos : r > 0)
  (h_cylinder_pos : h_cylinder > 0) :
  let v_cone := (1 / 3) * π * r^2 * h_cone
  let v_cylinder := π * r^2 * h_cylinder
  let v_hemisphere := (2 / 3) * π * r^3
  v_cylinder + v_hemisphere = 14.25 * π :=
by
  sorry

#check ice_cream_volume 10 1.5 2

end NUMINAMATH_CALUDE_ice_cream_volume_l1151_115135


namespace NUMINAMATH_CALUDE_least_trees_required_l1151_115149

theorem least_trees_required (n : ℕ) : 
  (n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n) → 
  (∀ m : ℕ, m > 0 ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m → n ≤ m) → 
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_least_trees_required_l1151_115149


namespace NUMINAMATH_CALUDE_lcm_of_1428_and_924_l1151_115124

theorem lcm_of_1428_and_924 : Nat.lcm 1428 924 = 15708 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_1428_and_924_l1151_115124


namespace NUMINAMATH_CALUDE_rectangular_prism_ratio_l1151_115153

/-- In a rectangular prism with edges a ≤ b ≤ c, if a:b = b:c = c:√(a² + b²), 
    then (a/b)² = (√5 - 1)/2 -/
theorem rectangular_prism_ratio (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) 
  (h_ratio : a / b = b / c ∧ b / c = c / Real.sqrt (a^2 + b^2)) :
  (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_ratio_l1151_115153


namespace NUMINAMATH_CALUDE_graphic_artist_pages_sum_l1151_115152

theorem graphic_artist_pages_sum (n : ℕ) (a₁ d : ℝ) : 
  n = 15 ∧ a₁ = 3 ∧ d = 2 → 
  (n / 2 : ℝ) * (2 * a₁ + (n - 1) * d) = 255 := by
  sorry

end NUMINAMATH_CALUDE_graphic_artist_pages_sum_l1151_115152


namespace NUMINAMATH_CALUDE_even_sum_probability_l1151_115189

theorem even_sum_probability (wheel1_even_prob wheel2_even_prob : ℚ) 
  (h1 : wheel1_even_prob = 3 / 5)
  (h2 : wheel2_even_prob = 1 / 2) : 
  wheel1_even_prob * wheel2_even_prob + (1 - wheel1_even_prob) * (1 - wheel2_even_prob) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l1151_115189


namespace NUMINAMATH_CALUDE_sheet_length_is_30_l1151_115182

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

end NUMINAMATH_CALUDE_sheet_length_is_30_l1151_115182


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l1151_115183

/-- The number of candies Susan bought on Tuesday -/
def tuesday_candies : ℕ := 3

/-- The number of candies Susan bought on Thursday -/
def thursday_candies : ℕ := 5

/-- The number of candies Susan bought on Friday -/
def friday_candies : ℕ := 2

/-- The number of candies Susan has left -/
def remaining_candies : ℕ := 4

/-- The total number of candies Susan bought -/
def total_candies : ℕ := tuesday_candies + thursday_candies + friday_candies

theorem susan_ate_six_candies : total_candies - remaining_candies = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l1151_115183


namespace NUMINAMATH_CALUDE_round_table_gender_divisibility_l1151_115122

theorem round_table_gender_divisibility (n : ℕ) : 
  (∃ k : ℕ, k = n / 2 ∧ k = n - k) → 
  (∃ m : ℕ, n = 4 * m) :=
by sorry

end NUMINAMATH_CALUDE_round_table_gender_divisibility_l1151_115122


namespace NUMINAMATH_CALUDE_tuna_distribution_l1151_115192

theorem tuna_distribution (total_customers : ℕ) (tuna_count : ℕ) (tuna_weight : ℕ) (customers_without_fish : ℕ) :
  total_customers = 100 →
  tuna_count = 10 →
  tuna_weight = 200 →
  customers_without_fish = 20 →
  (tuna_count * tuna_weight) / (total_customers - customers_without_fish) = 25 := by
  sorry

end NUMINAMATH_CALUDE_tuna_distribution_l1151_115192


namespace NUMINAMATH_CALUDE_count_polynomials_l1151_115142

/-- A function to determine if an expression is a polynomial -/
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "x^2+2" => true
  | "1/a+4" => false
  | "3ab^2/7" => true
  | "ab/c" => false
  | "-5x" => true
  | "0" => true
  | _ => false

/-- The list of expressions to check -/
def expressions : List String := ["x^2+2", "1/a+4", "3ab^2/7", "ab/c", "-5x", "0"]

/-- Theorem stating that there are exactly 4 polynomial expressions in the given list -/
theorem count_polynomials : 
  (expressions.filter is_polynomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_polynomials_l1151_115142


namespace NUMINAMATH_CALUDE_sum_of_coordinates_B_l1151_115129

/-- Given that M(3,7) is the midpoint of AB and A(9,3), prove that the sum of B's coordinates is 8 -/
theorem sum_of_coordinates_B (A B M : ℝ × ℝ) : 
  A = (9, 3) → M = (3, 7) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_B_l1151_115129


namespace NUMINAMATH_CALUDE_quadratic_unique_root_l1151_115128

/-- A function that represents the quadratic equation (m-4)x^2 - 2mx - m - 6 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 4) * x^2 - 2 * m * x - m - 6

/-- Condition for the quadratic function to have exactly one root -/
def has_unique_root (m : ℝ) : Prop :=
  (∃ x : ℝ, f m x = 0) ∧ (∀ x y : ℝ, f m x = 0 → f m y = 0 → x = y)

theorem quadratic_unique_root (m : ℝ) :
  has_unique_root m → m = -4 ∨ m = 3 ∨ m = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_unique_root_l1151_115128


namespace NUMINAMATH_CALUDE_gcd_1729_867_l1151_115166

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l1151_115166


namespace NUMINAMATH_CALUDE_abs_z_equals_one_l1151_115132

theorem abs_z_equals_one (z : ℂ) (h : (1 - 2*I)^2 / z = 4 - 3*I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_one_l1151_115132


namespace NUMINAMATH_CALUDE_trip_time_change_l1151_115123

/-- Calculates the time required for a trip given the original time, original speed, and new speed -/
def new_trip_time (original_time : ℚ) (original_speed : ℚ) (new_speed : ℚ) : ℚ :=
  (original_time * original_speed) / new_speed

theorem trip_time_change (original_time : ℚ) (original_speed : ℚ) (new_speed : ℚ) 
  (h1 : original_time = 16/3)
  (h2 : original_speed = 80)
  (h3 : new_speed = 50) :
  ∃ (ε : ℚ), abs (new_trip_time original_time original_speed new_speed - 853/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_trip_time_change_l1151_115123


namespace NUMINAMATH_CALUDE_ball_probability_l1151_115148

theorem ball_probability (p_red_yellow p_red_white : ℝ) 
  (h1 : p_red_yellow = 0.4) 
  (h2 : p_red_white = 0.9) : 
  ∃ (p_red p_yellow p_white : ℝ), 
    p_red + p_yellow = p_red_yellow ∧ 
    p_red + p_white = p_red_white ∧ 
    p_yellow + p_white = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1151_115148


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1151_115160

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1^2 + 4*x1 - 2 = 0 ∧ x2^2 + 4*x2 - 2 = 0 ∧ x1 = -2 + Real.sqrt 6 ∧ x2 = -2 - Real.sqrt 6) ∧
  (∃ y1 y2 : ℝ, 2*y1^2 - 3*y1 + 1 = 0 ∧ 2*y2^2 - 3*y2 + 1 = 0 ∧ y1 = 1/2 ∧ y2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1151_115160


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1151_115141

theorem probability_of_white_ball 
  (prob_red : ℝ) 
  (prob_black : ℝ) 
  (h1 : prob_red = 0.4) 
  (h2 : prob_black = 0.25) 
  (h3 : prob_red + prob_black + (1 - prob_red - prob_black) = 1) :
  1 - prob_red - prob_black = 0.35 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1151_115141


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1151_115193

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x, x^2 + 2*x > 0 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1151_115193


namespace NUMINAMATH_CALUDE_real_number_inequality_l1151_115154

theorem real_number_inequality (x : Fin 8 → ℝ) (h : ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ i j, i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (π / 7) := by
  sorry

end NUMINAMATH_CALUDE_real_number_inequality_l1151_115154


namespace NUMINAMATH_CALUDE_original_group_size_l1151_115144

theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : ℕ :=
  let original_size := 42
  let work_amount := original_size * initial_days
  have h1 : work_amount = (original_size - absent_men) * final_days := by sorry
  have h2 : initial_days = 12 := by sorry
  have h3 : absent_men = 6 := by sorry
  have h4 : final_days = 14 := by sorry
  original_size

#check original_group_size

end NUMINAMATH_CALUDE_original_group_size_l1151_115144


namespace NUMINAMATH_CALUDE_subset_gcd_property_l1151_115121

theorem subset_gcd_property (A : Finset ℕ) 
  (h1 : A ⊆ Finset.range 2007)
  (h2 : A.card = 1004) :
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (Nat.gcd a b ∣ c)) ∧
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(Nat.gcd a b ∣ c)) := by
sorry

end NUMINAMATH_CALUDE_subset_gcd_property_l1151_115121


namespace NUMINAMATH_CALUDE_smallest_x_divisible_by_3_5_11_l1151_115196

theorem smallest_x_divisible_by_3_5_11 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → 107 * 151 * y % 3 = 0 ∧ 107 * 151 * y % 5 = 0 ∧ 107 * 151 * y % 11 = 0 → x ≤ y) ∧
  107 * 151 * x % 3 = 0 ∧ 107 * 151 * x % 5 = 0 ∧ 107 * 151 * x % 11 = 0 ∧
  x = 165 := by
  sorry

-- Additional definitions to match the problem conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m > 1 → m < n → n % m ≠ 0

axiom prime_107 : is_prime 107
axiom prime_151 : is_prime 151
axiom prime_3 : is_prime 3
axiom prime_5 : is_prime 5
axiom prime_11 : is_prime 11

end NUMINAMATH_CALUDE_smallest_x_divisible_by_3_5_11_l1151_115196


namespace NUMINAMATH_CALUDE_pipe_fill_time_l1151_115119

/-- Represents the time (in hours) it takes for a pipe to fill a tank without a leak -/
def fill_time : ℝ := 6

/-- Represents the time (in hours) it takes for the pipe to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 8

/-- Represents the time (in hours) it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 24

/-- Proves that the time it takes for the pipe to fill the tank without the leak is 6 hours -/
theorem pipe_fill_time : 
  (1 / fill_time - 1 / leak_empty_time) * fill_time_with_leak = 1 :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l1151_115119


namespace NUMINAMATH_CALUDE_min_points_per_player_l1151_115120

theorem min_points_per_player 
  (num_players : ℕ) 
  (total_points : ℕ) 
  (max_individual_points : ℕ) 
  (h1 : num_players = 12)
  (h2 : total_points = 100)
  (h3 : max_individual_points = 23) :
  ∃ (min_points : ℕ), 
    min_points = 7 ∧ 
    (∃ (scores : List ℕ), 
      scores.length = num_players ∧ 
      scores.sum = total_points ∧
      (∀ s ∈ scores, s ≥ min_points) ∧
      (∃ s ∈ scores, s = max_individual_points) ∧
      (∀ s ∈ scores, s ≤ max_individual_points)) :=
by sorry

end NUMINAMATH_CALUDE_min_points_per_player_l1151_115120


namespace NUMINAMATH_CALUDE_family_probability_l1151_115130

theorem family_probability (p_boy p_girl : ℝ) (h1 : p_boy = 1 / 2) (h2 : p_girl = 1 / 2) :
  let p_at_least_one_each := 1 - (p_boy ^ 4 + p_girl ^ 4)
  p_at_least_one_each = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_family_probability_l1151_115130


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l1151_115199

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  restTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (poolLength : ℝ) (duration : ℝ) (swimmer1 : Swimmer) (swimmer2 : Swimmer) : ℕ :=
  sorry

/-- The main theorem --/
theorem swimmers_pass_count :
  let poolLength : ℝ := 120
  let duration : ℝ := 15 * 60  -- 15 minutes in seconds
  let swimmer1 : Swimmer := { speed := 4, restTime := 30 }
  let swimmer2 : Swimmer := { speed := 3, restTime := 0 }
  countPasses poolLength duration swimmer1 swimmer2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l1151_115199


namespace NUMINAMATH_CALUDE_find_y_l1151_115125

theorem find_y : ∃ y : ℕ, y^3 * 6^4 / 432 = 5184 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1151_115125


namespace NUMINAMATH_CALUDE_jack_has_forty_dollars_l1151_115114

/-- Represents the cost of a pair of socks in dollars -/
def sock_cost : ℚ := 9.5

/-- Represents the cost of the soccer shoes in dollars -/
def shoe_cost : ℚ := 92

/-- Represents the additional amount Jack needs in dollars -/
def additional_needed : ℚ := 71

/-- Calculates Jack's initial money based on the given costs and additional amount needed -/
def jack_initial_money : ℚ :=
  2 * sock_cost + shoe_cost - additional_needed

/-- Theorem stating that Jack's initial money is $40 -/
theorem jack_has_forty_dollars :
  jack_initial_money = 40 := by sorry

end NUMINAMATH_CALUDE_jack_has_forty_dollars_l1151_115114


namespace NUMINAMATH_CALUDE_some_magical_beings_are_enchanting_creatures_l1151_115177

-- Define the sets
variable (W : Set α) -- Wizards
variable (M : Set α) -- Magical beings
variable (E : Set α) -- Enchanting creatures

-- Define the conditions
variable (h1 : W ⊆ M) -- All wizards are magical beings
variable (h2 : ∃ x, x ∈ E ∩ W) -- Some enchanting creatures are wizards

-- State the theorem
theorem some_magical_beings_are_enchanting_creatures :
  ∃ x, x ∈ M ∩ E := by sorry

end NUMINAMATH_CALUDE_some_magical_beings_are_enchanting_creatures_l1151_115177


namespace NUMINAMATH_CALUDE_barbara_candies_left_l1151_115100

/-- The number of candies Barbara has left after using some -/
def candies_left (initial : Float) (used : Float) : Float :=
  initial - used

/-- Theorem: If Barbara initially has 18.0 candies and uses 9.0 candies,
    then the number of candies she has left is 9.0. -/
theorem barbara_candies_left :
  candies_left 18.0 9.0 = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_left_l1151_115100


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1151_115110

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ 
  (a > -2 ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1151_115110


namespace NUMINAMATH_CALUDE_g_of_negative_four_l1151_115126

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem g_of_negative_four : g (-4) = -18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_negative_four_l1151_115126


namespace NUMINAMATH_CALUDE_chosen_number_calculation_l1151_115178

theorem chosen_number_calculation (x : ℕ) (h : x = 30) : x * 8 - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_calculation_l1151_115178


namespace NUMINAMATH_CALUDE_apartment_building_occupancy_l1151_115185

theorem apartment_building_occupancy :
  let total_floors : ℕ := 12
  let full_floors : ℕ := total_floors / 2
  let half_capacity_floors : ℕ := total_floors - full_floors
  let apartments_per_floor : ℕ := 10
  let people_per_apartment : ℕ := 4
  let people_per_full_floor : ℕ := apartments_per_floor * people_per_apartment
  let people_per_half_floor : ℕ := people_per_full_floor / 2
  let total_people : ℕ := full_floors * people_per_full_floor + half_capacity_floors * people_per_half_floor
  total_people = 360 := by
  sorry

end NUMINAMATH_CALUDE_apartment_building_occupancy_l1151_115185


namespace NUMINAMATH_CALUDE_trapezium_circle_radius_l1151_115164

/-- Represents a trapezium PQRS with a circle tangent to all sides -/
structure TrapeziumWithCircle where
  -- Length of PQ and SR
  side_length : ℝ
  -- Area of the trapezium
  area : ℝ
  -- Assertion that SP is parallel to RQ
  sp_parallel_rq : Prop
  -- Assertion that all sides are tangent to the circle
  all_sides_tangent : Prop

/-- The radius of the circle in a trapezium with given properties -/
def circle_radius (t : TrapeziumWithCircle) : ℝ :=
  12

/-- Theorem stating that for a trapezium with given properties, the radius of the inscribed circle is 12 -/
theorem trapezium_circle_radius 
  (t : TrapeziumWithCircle) 
  (h1 : t.side_length = 25)
  (h2 : t.area = 600) :
  circle_radius t = 12 := by sorry

end NUMINAMATH_CALUDE_trapezium_circle_radius_l1151_115164
