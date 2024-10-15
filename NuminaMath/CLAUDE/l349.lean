import Mathlib

namespace NUMINAMATH_CALUDE_total_is_255_l349_34941

/-- Represents the ratio of money distribution among three people -/
structure MoneyRatio :=
  (a b c : ℕ)

/-- Calculates the total amount of money given a ratio and the first person's share -/
def totalAmount (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  let multiplier := firstShare / ratio.a
  multiplier * (ratio.a + ratio.b + ratio.c)

/-- Theorem stating that for the given ratio and first share, the total amount is 255 -/
theorem total_is_255 (ratio : MoneyRatio) (h1 : ratio.a = 3) (h2 : ratio.b = 5) (h3 : ratio.c = 9) 
    (h4 : totalAmount ratio 45 = 255) : totalAmount ratio 45 = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_is_255_l349_34941


namespace NUMINAMATH_CALUDE_uncle_bradley_bill_change_l349_34994

theorem uncle_bradley_bill_change (total_amount : ℕ) (small_bill_denom : ℕ) (total_bills : ℕ) :
  total_amount = 1000 →
  small_bill_denom = 50 →
  total_bills = 13 →
  ∃ (large_bill_denom : ℕ),
    (3 * total_amount / 10 / small_bill_denom + (total_amount - 3 * total_amount / 10) / large_bill_denom = total_bills) ∧
    large_bill_denom = 100 := by
  sorry

#check uncle_bradley_bill_change

end NUMINAMATH_CALUDE_uncle_bradley_bill_change_l349_34994


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l349_34925

-- Define variables
variable (a b : ℝ)
variable (m n : ℕ)

-- Define the condition that the terms are like terms
def are_like_terms : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ * a^(2*m) * b = k₂ * a^4 * b^n

-- Theorem statement
theorem like_terms_exponent_sum :
  are_like_terms a b m n → m + n = 3 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l349_34925


namespace NUMINAMATH_CALUDE_weight_difference_l349_34971

-- Define the weights as natural numbers
def sam_weight : ℕ := 105
def peter_weight : ℕ := 65

-- Define Tyler's weight based on Peter's weight
def tyler_weight : ℕ := 2 * peter_weight

-- Theorem to prove
theorem weight_difference : tyler_weight - sam_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l349_34971


namespace NUMINAMATH_CALUDE_other_number_is_twenty_l349_34909

theorem other_number_is_twenty (x y : ℤ) 
  (sum_eq : 3 * x + 2 * y = 145) 
  (one_is_35 : x = 35 ∨ y = 35) : 
  (x ≠ 35 → x = 20) ∧ (y ≠ 35 → y = 20) :=
sorry

end NUMINAMATH_CALUDE_other_number_is_twenty_l349_34909


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l349_34960

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 2 + 1
  (1 - 1 / x) / ((x^2 - 2*x + 1) / x^2) = 1 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l349_34960


namespace NUMINAMATH_CALUDE_income_calculation_l349_34989

/-- Given a person's income and expenditure ratio, and their savings amount, 
    calculate their income. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : 
  income_ratio = 5 → expenditure_ratio = 4 → savings = 3200 → 
  income_ratio * savings / (income_ratio - expenditure_ratio) = 16000 := by
  sorry

#check income_calculation

end NUMINAMATH_CALUDE_income_calculation_l349_34989


namespace NUMINAMATH_CALUDE_smallest_v_in_consecutive_cubes_sum_l349_34929

theorem smallest_v_in_consecutive_cubes_sum (w x y u v : ℕ) :
  w < x ∧ x < y ∧ y < u ∧ u < v →
  (∃ a, w = a^3) ∧ (∃ b, x = b^3) ∧ (∃ c, y = c^3) ∧ (∃ d, u = d^3) ∧ (∃ e, v = e^3) →
  w^3 + x^3 + y^3 + u^3 = v^3 →
  v ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_v_in_consecutive_cubes_sum_l349_34929


namespace NUMINAMATH_CALUDE_arrangements_count_l349_34919

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of arrangements for 7 students in a row, 
    where one student (A) must be in the center and 
    two students (B and C) must stand together -/
def arrangements : ℕ := 192

/-- Theorem stating that the number of arrangements is 192 -/
theorem arrangements_count : 
  (∀ (n : ℕ), n = total_students → 
   ∃ (center : Fin n) (together : Fin n → Fin n → Prop),
   (∀ (i j : Fin n), together i j ↔ together j i) ∧
   (∃! (pair : Fin n × Fin n), together pair.1 pair.2) ∧
   (center = ⟨(n - 1) / 2, by sorry⟩) →
   (arrangements = 192)) :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l349_34919


namespace NUMINAMATH_CALUDE_set_A_properties_l349_34927

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (2 ∈ A) ∧
  (-2 ∈ A) ∧
  (A = {-2, 2}) ∧
  (∅ ⊆ A) := by
sorry

end NUMINAMATH_CALUDE_set_A_properties_l349_34927


namespace NUMINAMATH_CALUDE_min_value_theorem_l349_34965

/-- An even function f defined on ℝ -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m + 1| - 2

/-- The property of f being an even function -/
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem min_value_theorem (m : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  is_even_function (f m) →
  f m a + f m (2 * b) = m →
  (∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 9/5) →
  ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l349_34965


namespace NUMINAMATH_CALUDE_oranges_picked_sum_l349_34937

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 14

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_sum :
  total_oranges = 55 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_sum_l349_34937


namespace NUMINAMATH_CALUDE_a_greater_than_zero_when_a_greater_than_b_l349_34956

theorem a_greater_than_zero_when_a_greater_than_b (a b : ℝ) 
  (h1 : a^2 > b^2) (h2 : a > b) : a > 0 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_zero_when_a_greater_than_b_l349_34956


namespace NUMINAMATH_CALUDE_geometry_angle_probability_l349_34959

def geometry_letters : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def angle_letters : Finset Char := {'A', 'N', 'G', 'L', 'E'}

theorem geometry_angle_probability : 
  (geometry_letters ∩ angle_letters).card / geometry_letters.card = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometry_angle_probability_l349_34959


namespace NUMINAMATH_CALUDE_parabola_directrix_l349_34916

/-- Given a parabola with equation y = ax² and directrix y = -1, prove that a = 1/4 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 → (∃ k : ℝ, y = -1/4/k ∧ k = a)) → 
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l349_34916


namespace NUMINAMATH_CALUDE_income_distribution_equation_l349_34982

/-- Represents a person's income distribution --/
structure IncomeDistribution where
  total : ℝ
  children_percent : ℝ
  wife_percent : ℝ
  orphan_percent : ℝ
  remaining : ℝ

/-- Theorem stating the relationship between income and its distribution --/
theorem income_distribution_equation (d : IncomeDistribution) 
  (h1 : d.children_percent = 0.1)
  (h2 : d.wife_percent = 0.2)
  (h3 : d.orphan_percent = 0.1)
  (h4 : d.remaining = 500) :
  d.total - (2 * d.children_percent * d.total + 
             d.wife_percent * d.total + 
             d.orphan_percent * (d.total - (2 * d.children_percent * d.total + d.wife_percent * d.total))) = 
  d.remaining := by
  sorry

#eval 500 / 0.54  -- This will output the approximate total income

end NUMINAMATH_CALUDE_income_distribution_equation_l349_34982


namespace NUMINAMATH_CALUDE_shaded_area_of_circle_with_perpendicular_diameters_l349_34904

theorem shaded_area_of_circle_with_perpendicular_diameters (r : ℝ) (h : r = 4) :
  let circle_area := π * r^2
  let quarter_circle_area := circle_area / 4
  let right_triangle_area := r^2 / 2
  let shaded_area := 2 * quarter_circle_area + 2 * right_triangle_area
  shaded_area = 16 + 8 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_circle_with_perpendicular_diameters_l349_34904


namespace NUMINAMATH_CALUDE_problem_solution_l349_34939

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + 2*x - 3

-- Define the set M
def M : Set ℝ := {x | f x ≤ -1}

-- Theorem statement
theorem problem_solution :
  (M = {x : ℝ | x ≤ 0}) ∧
  (∀ x ∈ M, x * (f x)^2 - x^2 * (f x) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l349_34939


namespace NUMINAMATH_CALUDE_line_equation_conversion_l349_34935

/-- Given a line in the form (3, 7) · ((x, y) - (-2, 4)) = 0, 
    prove that its slope-intercept form y = mx + b 
    has m = -3/7 and b = 22/7 -/
theorem line_equation_conversion :
  ∀ (x y : ℝ), 
  (3 : ℝ) * (x + 2) + (7 : ℝ) * (y - 4) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = -3/7 ∧ b = 22/7 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_conversion_l349_34935


namespace NUMINAMATH_CALUDE_rectangular_field_area_l349_34996

/-- The area of a rectangular field with length 1.2 meters and width three-fourths of the length is 1.08 square meters. -/
theorem rectangular_field_area : 
  let length : ℝ := 1.2
  let width : ℝ := (3/4) * length
  let area : ℝ := length * width
  area = 1.08 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l349_34996


namespace NUMINAMATH_CALUDE_inequality_solution_set_l349_34901

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 + 2*(m+1)*x + 9*m + 4) < 0) ↔ m < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l349_34901


namespace NUMINAMATH_CALUDE_x_less_than_one_iff_x_abs_x_less_than_one_l349_34984

theorem x_less_than_one_iff_x_abs_x_less_than_one (x : ℝ) : x < 1 ↔ x * |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_one_iff_x_abs_x_less_than_one_l349_34984


namespace NUMINAMATH_CALUDE_grain_mixture_pricing_l349_34970

/-- Calculates the selling price of a grain given its cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the grain mixture problem -/
theorem grain_mixture_pricing
  (wheat_weight : ℚ) (wheat_price : ℚ) (wheat_profit : ℚ)
  (rice_weight : ℚ) (rice_price : ℚ) (rice_profit : ℚ)
  (barley_weight : ℚ) (barley_price : ℚ) (barley_profit : ℚ)
  (h_wheat_weight : wheat_weight = 30)
  (h_wheat_price : wheat_price = 11.5)
  (h_wheat_profit : wheat_profit = 30)
  (h_rice_weight : rice_weight = 20)
  (h_rice_price : rice_price = 14.25)
  (h_rice_profit : rice_profit = 25)
  (h_barley_weight : barley_weight = 15)
  (h_barley_price : barley_price = 10)
  (h_barley_profit : barley_profit = 35) :
  let total_weight := wheat_weight + rice_weight + barley_weight
  let total_selling_price := sellingPrice (wheat_weight * wheat_price) wheat_profit +
                             sellingPrice (rice_weight * rice_price) rice_profit +
                             sellingPrice (barley_weight * barley_price) barley_profit
  total_selling_price / total_weight = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_grain_mixture_pricing_l349_34970


namespace NUMINAMATH_CALUDE_area_union_rotated_triangle_l349_34931

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The centroid of a triangle -/
def Triangle.centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Rotation of a point around another point by 180 degrees -/
def rotate180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The union of two regions -/
def unionArea (area1 : ℝ) (area2 : ℝ) : ℝ := sorry

theorem area_union_rotated_triangle (t : Triangle) :
  let m := t.centroid
  let t' := Triangle.mk t.a t.b t.c t.h_positive
  unionArea t.area t'.area = t.area := by sorry

end NUMINAMATH_CALUDE_area_union_rotated_triangle_l349_34931


namespace NUMINAMATH_CALUDE_geometric_sequence_exists_l349_34934

theorem geometric_sequence_exists : ∃ (a r : ℝ), 
  a ≠ 0 ∧ r ≠ 0 ∧ 
  a * r^2 = 3 ∧
  a * r^4 = 27 ∧
  a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_exists_l349_34934


namespace NUMINAMATH_CALUDE_sphere_surface_area_and_volume_l349_34944

/-- Given a sphere with diameter 18 inches, prove its surface area and volume -/
theorem sphere_surface_area_and_volume :
  let diameter : ℝ := 18
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  let volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  surface_area = 324 * Real.pi ∧ volume = 972 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_sphere_surface_area_and_volume_l349_34944


namespace NUMINAMATH_CALUDE_coin_rotation_theorem_l349_34992

/-- 
  Represents the number of degrees a coin rotates when rolling around another coin.
  
  coinA : The rolling coin
  coinB : The stationary coin
  radiusRatio : The ratio of coinB's radius to coinA's radius
  rotationDegrees : The number of degrees coinA rotates around its center
-/
def coinRotation (coinA coinB : ℝ) (radiusRatio : ℝ) (rotationDegrees : ℝ) : Prop :=
  coinA > 0 ∧ 
  coinB > 0 ∧ 
  radiusRatio = 2 ∧ 
  rotationDegrees = 3 * 360

theorem coin_rotation_theorem (coinA coinB radiusRatio rotationDegrees : ℝ) :
  coinRotation coinA coinB radiusRatio rotationDegrees →
  rotationDegrees = 1080 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_rotation_theorem_l349_34992


namespace NUMINAMATH_CALUDE_square_root_problem_l349_34969

theorem square_root_problem (n : ℝ) (x : ℝ) (h1 : n > 0) (h2 : Real.sqrt n = x + 3) (h3 : Real.sqrt n = 2*x - 6) :
  x = 1 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l349_34969


namespace NUMINAMATH_CALUDE_max_score_in_range_score_2079_is_in_range_l349_34983

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_score_in_range :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → score x ≤ score 2079 :=
by sorry

theorem score_2079 : score 2079 = 30 :=
by sorry

theorem is_in_range : 2017 ≤ 2079 ∧ 2079 ≤ 2117 :=
by sorry

end NUMINAMATH_CALUDE_max_score_in_range_score_2079_is_in_range_l349_34983


namespace NUMINAMATH_CALUDE_juggling_show_balls_l349_34957

/-- The number of balls needed for a juggling show -/
def balls_needed (jugglers : ℕ) (balls_per_juggler : ℕ) : ℕ :=
  jugglers * balls_per_juggler

/-- Theorem: 378 jugglers each juggling 6 balls require 2268 balls in total -/
theorem juggling_show_balls : balls_needed 378 6 = 2268 := by
  sorry

end NUMINAMATH_CALUDE_juggling_show_balls_l349_34957


namespace NUMINAMATH_CALUDE_three_solutions_cubic_l349_34951

theorem three_solutions_cubic (n : ℕ+) (x y : ℤ) 
  (h : x^3 - 3*x*y^2 + y^3 = n) : 
  ∃ (a b c d e f : ℤ), 
    (a^3 - 3*a*b^2 + b^3 = n) ∧ 
    (c^3 - 3*c*d^2 + d^3 = n) ∧ 
    (e^3 - 3*e*f^2 + f^3 = n) ∧ 
    (a ≠ c ∨ b ≠ d) ∧ 
    (a ≠ e ∨ b ≠ f) ∧ 
    (c ≠ e ∨ d ≠ f) := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_cubic_l349_34951


namespace NUMINAMATH_CALUDE_bookcase_length_in_feet_l349_34947

/-- Converts inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem bookcase_length_in_feet :
  inches_to_feet 48 = 4 :=
by sorry

end NUMINAMATH_CALUDE_bookcase_length_in_feet_l349_34947


namespace NUMINAMATH_CALUDE_one_large_one_small_capacity_l349_34967

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- Represents the capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of 3 large trucks and 4 small trucks is 22 tons -/
axiom condition1 : 3 * large_truck_capacity + 4 * small_truck_capacity = 22

/-- The total capacity of 2 large trucks and 6 small trucks is 23 tons -/
axiom condition2 : 2 * large_truck_capacity + 6 * small_truck_capacity = 23

/-- Theorem: One large truck and one small truck can transport 6.5 tons together -/
theorem one_large_one_small_capacity : 
  large_truck_capacity + small_truck_capacity = 6.5 := by sorry

end NUMINAMATH_CALUDE_one_large_one_small_capacity_l349_34967


namespace NUMINAMATH_CALUDE_ellipse_product_l349_34912

/-- A point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- An ellipse defined by its center, major axis, minor axis, and a focus -/
structure Ellipse :=
  (center : Point)
  (majorAxis : ℝ)
  (minorAxis : ℝ)
  (focus : Point)

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Diameter of the incircle of a right triangle -/
def incircleDiameter (leg1 leg2 hypotenuse : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_product (e : Ellipse) :
  distance e.center e.focus = 8 →
  incircleDiameter e.minorAxis 8 e.majorAxis = 4 →
  e.majorAxis * e.minorAxis = 240 := by sorry

end NUMINAMATH_CALUDE_ellipse_product_l349_34912


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l349_34949

/-- Given a circle with equation (x-2)^2 + y^2 = 4, prove its center and radius -/
theorem circle_center_and_radius :
  let equation := (fun (x y : ℝ) => (x - 2)^2 + y^2 = 4)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, 0) ∧ radius = 2 ∧
    ∀ (x y : ℝ), equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l349_34949


namespace NUMINAMATH_CALUDE_fraction_simplification_l349_34954

theorem fraction_simplification (x : ℝ) : (x - 2) / 4 - (3 * x + 1) / 3 = (-9 * x - 10) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l349_34954


namespace NUMINAMATH_CALUDE_fourth_term_value_l349_34955

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem fourth_term_value : a 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_value_l349_34955


namespace NUMINAMATH_CALUDE_min_shift_for_sine_overlap_l349_34999

theorem min_shift_for_sine_overlap (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + π / 3)) →
  (∀ x, g x = Real.sin (2 * x)) →
  (∀ x, f x = g (x + π / 6)) →
  (∀ x, f x = g (x + φ)) →
  φ > 0 →
  φ ≥ π / 6 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_sine_overlap_l349_34999


namespace NUMINAMATH_CALUDE_hyperbola_equation_l349_34918

/-- Represents a hyperbola with focus on the y-axis -/
structure Hyperbola where
  transverse_axis_length : ℝ
  focal_length : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / (h.transverse_axis_length/2)^2) - (x^2 / ((h.focal_length/2)^2 - (h.transverse_axis_length/2)^2)) = 1

theorem hyperbola_equation (h : Hyperbola) 
  (h_transverse : h.transverse_axis_length = 6)
  (h_focal : h.focal_length = 10) :
  ∀ x y : ℝ, standard_equation h x y ↔ y^2/9 - x^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l349_34918


namespace NUMINAMATH_CALUDE_curve_and_tangent_line_l349_34906

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the line l1
def l1 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the property of point P
def P_property (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 3)^2 + y^2 = 4 * ((x - 3)^2 + y^2)

-- Define the minimization condition
def min_distance (Q M : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (mx, my) := M
  l1 qx qy ∧ C mx my ∧
  ∀ M' : ℝ × ℝ, C M'.1 M'.2 → (qx - mx)^2 + (qy - my)^2 ≤ (qx - M'.1)^2 + (qy - M'.2)^2

-- State the theorem
theorem curve_and_tangent_line :
  (∀ P : ℝ × ℝ, P_property P → C P.1 P.2) ∧
  (∀ Q M : ℝ × ℝ, min_distance Q M → (M.1 = 1 ∨ M.2 = -4)) :=
sorry

end NUMINAMATH_CALUDE_curve_and_tangent_line_l349_34906


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l349_34953

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  shaded_squares : ℕ
  half_shaded_squares : ℕ

/-- Calculates the fraction of the quilt block that is shaded -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  (q.shaded_squares + q.half_shaded_squares / 2) / q.total_squares

/-- Theorem stating that the given quilt block has 3/8 of its area shaded -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := {
    total_squares := 16,
    shaded_squares := 4,
    half_shaded_squares := 4
  }
  shaded_fraction q = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l349_34953


namespace NUMINAMATH_CALUDE_margarita_run_distance_l349_34975

/-- Proves that Margarita ran 18 feet given the conditions of the long jump event -/
theorem margarita_run_distance (ricciana_run : ℝ) (ricciana_jump : ℝ) (margarita_total : ℝ) :
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_total = ricciana_run + ricciana_jump + 1 →
  ∃ (margarita_run : ℝ) (margarita_jump : ℝ),
    margarita_jump = 2 * ricciana_jump - 1 ∧
    margarita_total = margarita_run + margarita_jump ∧
    margarita_run = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_margarita_run_distance_l349_34975


namespace NUMINAMATH_CALUDE_repeating_decimal_35_sum_l349_34903

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 99

theorem repeating_decimal_35_sum : 
  ∀ a b : ℕ, 
  a > 0 → b > 0 →
  RepeatingDecimal 35 = a / b →
  Nat.gcd a b = 1 →
  a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_35_sum_l349_34903


namespace NUMINAMATH_CALUDE_cycle_selling_price_l349_34917

/-- The selling price of a cycle after applying successive discounts -/
def selling_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem: The selling price of a cycle originally priced at Rs. 3,600, 
    after applying successive discounts of 15%, 10%, and 5%, is equal to Rs. 2,616.30 -/
theorem cycle_selling_price :
  selling_price 3600 0.15 0.10 0.05 = 2616.30 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l349_34917


namespace NUMINAMATH_CALUDE_cosine_identity_73_47_l349_34950

theorem cosine_identity_73_47 :
  let α : Real := 73 * π / 180
  let β : Real := 47 * π / 180
  (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos α) * (Real.cos β) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_73_47_l349_34950


namespace NUMINAMATH_CALUDE_correct_operation_l349_34961

theorem correct_operation : 
  (5 * Real.sqrt 3 - 2 * Real.sqrt 3 ≠ 3) ∧ 
  (2 * Real.sqrt 2 * 3 * Real.sqrt 2 ≠ 6) ∧ 
  (3 * Real.sqrt 3 / Real.sqrt 3 = 3) ∧ 
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l349_34961


namespace NUMINAMATH_CALUDE_pamela_sugar_amount_l349_34977

/-- The amount of sugar Pamela spilled in ounces -/
def sugar_spilled : ℝ := 5.2

/-- The amount of sugar Pamela has left in ounces -/
def sugar_left : ℝ := 4.6

/-- The initial amount of sugar Pamela bought in ounces -/
def initial_sugar : ℝ := sugar_spilled + sugar_left

theorem pamela_sugar_amount : initial_sugar = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_pamela_sugar_amount_l349_34977


namespace NUMINAMATH_CALUDE_max_min_difference_z_l349_34943

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l349_34943


namespace NUMINAMATH_CALUDE_solve_equation_l349_34936

theorem solve_equation : (45 : ℚ) / (8 - 3/4) = 180/29 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l349_34936


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_eq_neg_two_l349_34915

/-- If the simplified result of (3x+2)(3x+a) does not contain a linear term of x, then a = -2 -/
theorem no_linear_term_implies_a_eq_neg_two (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (3*x + 2) * (3*x + a) = b*x^2 + c) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_eq_neg_two_l349_34915


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l349_34968

/-- Given a parallelogram with area 44 cm² and height 11 cm, its base length is 4 cm. -/
theorem parallelogram_base_length (area : ℝ) (height : ℝ) (base : ℝ) 
    (h1 : area = 44) 
    (h2 : height = 11) 
    (h3 : area = base * height) : base = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l349_34968


namespace NUMINAMATH_CALUDE_car_travel_distance_l349_34913

/-- Given a car that travels 300 miles on 10 gallons of gas, 
    prove that it will travel 450 miles on 15 gallons of gas. -/
theorem car_travel_distance (miles : ℝ) (gallons : ℝ) 
  (h1 : miles = 300) (h2 : gallons = 10) :
  (miles / gallons) * 15 = 450 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l349_34913


namespace NUMINAMATH_CALUDE_initial_bananas_count_l349_34946

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 70

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left_on_tree : ℕ := 100

/-- The number of bananas remaining in Raj's basket -/
def bananas_in_basket : ℕ := 2 * bananas_eaten

/-- The total number of bananas Raj cut from the tree -/
def bananas_cut : ℕ := bananas_eaten + bananas_in_basket

/-- The initial number of bananas on the tree -/
def initial_bananas : ℕ := bananas_cut + bananas_left_on_tree

theorem initial_bananas_count : initial_bananas = 310 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l349_34946


namespace NUMINAMATH_CALUDE_inequality_proof_l349_34922

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l349_34922


namespace NUMINAMATH_CALUDE_roots_of_equation_l349_34979

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | f x = 0} = {2, 3, -2} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l349_34979


namespace NUMINAMATH_CALUDE_certain_number_calculation_l349_34976

theorem certain_number_calculation (x y : ℝ) : 
  0.12 / x * 2 = y → x = 0.1 → y = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l349_34976


namespace NUMINAMATH_CALUDE_infinite_solutions_l349_34948

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 2 * x - 3 * y = 5
def equation2 (x y : ℝ) : Prop := 4 * x - 6 * y = 10

-- Theorem stating that the system has infinitely many solutions
theorem infinite_solutions :
  ∃ (f : ℝ → ℝ × ℝ), ∀ (t : ℝ),
    let (x, y) := f t
    equation1 x y ∧ equation2 x y :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l349_34948


namespace NUMINAMATH_CALUDE_probability_ace_spade_three_correct_l349_34914

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Nat := 52

/-- Number of Aces in a standard deck -/
def NumAces : Nat := 4

/-- Number of Spades in a standard deck -/
def NumSpades : Nat := 13

/-- Number of 3s in a standard deck -/
def NumThrees : Nat := 4

/-- Probability of drawing an Ace as the first card, a Spade as the second card,
    and a 3 as the third card when dealing three cards at random from a standard deck -/
def probability_ace_spade_three : ℚ :=
  17 / 11050

theorem probability_ace_spade_three_correct :
  probability_ace_spade_three = 17 / 11050 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_spade_three_correct_l349_34914


namespace NUMINAMATH_CALUDE_triangleAreaSum_form_l349_34986

/-- The sum of areas of all triangles with vertices on a 2 by 3 by 4 rectangular box -/
def triangleAreaSum : ℝ := sorry

/-- The number of vertices of a rectangular box -/
def vertexCount : ℕ := 8

/-- The dimensions of the rectangular box -/
def boxDimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0  -- This line is just to satisfy Lean's totality requirement

/-- Theorem stating the form of the sum of triangle areas -/
theorem triangleAreaSum_form :
  ∃ (k p : ℝ), triangleAreaSum = 168 + k * Real.sqrt p :=
sorry

end NUMINAMATH_CALUDE_triangleAreaSum_form_l349_34986


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l349_34990

theorem inverse_proportion_problem (x y : ℝ) (h : x * y = 12) :
  x = 5 → y = 2.4 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l349_34990


namespace NUMINAMATH_CALUDE_employee_earnings_theorem_l349_34932

/-- Calculates the total earnings for an employee based on their work schedule and pay rates -/
def calculate_earnings (task_a_rate : ℚ) (task_b_rate : ℚ) (overtime_multiplier : ℚ) 
                       (commission_rate : ℚ) (task_a_hours : List ℚ) (task_b_hours : List ℚ) : ℚ :=
  let task_a_total_hours := task_a_hours.sum
  let task_b_total_hours := task_b_hours.sum
  let task_a_regular_hours := min task_a_total_hours 40
  let task_a_overtime_hours := max (task_a_total_hours - 40) 0
  let task_a_earnings := task_a_regular_hours * task_a_rate + 
                         task_a_overtime_hours * task_a_rate * overtime_multiplier
  let task_b_earnings := task_b_total_hours * task_b_rate
  let total_before_commission := task_a_earnings + task_b_earnings
  let commission := if task_b_total_hours ≥ 10 then total_before_commission * commission_rate else 0
  total_before_commission + commission

/-- Theorem stating that the employee's earnings for the given work schedule and pay rates equal $2211 -/
theorem employee_earnings_theorem :
  let task_a_rate : ℚ := 30
  let task_b_rate : ℚ := 40
  let overtime_multiplier : ℚ := 1.5
  let commission_rate : ℚ := 0.1
  let task_a_hours : List ℚ := [6, 6, 6, 12, 12]
  let task_b_hours : List ℚ := [4, 4, 4, 3, 3]
  calculate_earnings task_a_rate task_b_rate overtime_multiplier commission_rate task_a_hours task_b_hours = 2211 := by
  sorry

end NUMINAMATH_CALUDE_employee_earnings_theorem_l349_34932


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l349_34972

theorem davids_chemistry_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 72)
  (h2 : math_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : average_marks = 62.6)
  (h6 : (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks : ℚ) / 5 = average_marks) :
  chemistry_marks = 62 :=
by sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l349_34972


namespace NUMINAMATH_CALUDE_fraction_value_l349_34942

theorem fraction_value (a b c : ℤ) 
  (eq1 : a + b = 20) 
  (eq2 : b + c = 22) 
  (eq3 : c + a = 2022) : 
  (a - b) / (c - a) = 1000 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l349_34942


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l349_34940

theorem average_of_eleven_numbers 
  (n : ℕ) 
  (first_six_avg : ℚ) 
  (last_six_avg : ℚ) 
  (sixth_number : ℚ) 
  (h1 : n = 11) 
  (h2 : first_six_avg = 98) 
  (h3 : last_six_avg = 65) 
  (h4 : sixth_number = 318) : 
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l349_34940


namespace NUMINAMATH_CALUDE_rectangles_in_6x6_grid_l349_34962

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in a grid of size n x n -/
def rectangles_in_grid (n : ℕ) : ℕ := (choose_two n) ^ 2

/-- Theorem: In a 6x6 grid, the number of rectangles is 225 -/
theorem rectangles_in_6x6_grid : rectangles_in_grid 6 = 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_6x6_grid_l349_34962


namespace NUMINAMATH_CALUDE_doll_distribution_theorem_l349_34973

def distribute_dolls (n_dolls : ℕ) (n_houses : ℕ) : ℕ :=
  let choose_pair := n_dolls.choose 2
  let choose_house := n_houses
  let arrange_rest := (n_dolls - 2).factorial
  choose_pair * choose_house * arrange_rest

theorem doll_distribution_theorem :
  distribute_dolls 7 6 = 15120 :=
sorry

end NUMINAMATH_CALUDE_doll_distribution_theorem_l349_34973


namespace NUMINAMATH_CALUDE_solution_set_nonempty_iff_a_gt_one_l349_34930

theorem solution_set_nonempty_iff_a_gt_one :
  ∀ a : ℝ, (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_iff_a_gt_one_l349_34930


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l349_34966

theorem restaurant_bill_calculation
  (check_amount : ℝ)
  (tax_rate : ℝ)
  (tip : ℝ)
  (h1 : check_amount = 15)
  (h2 : tax_rate = 0.20)
  (h3 : tip = 2) :
  check_amount + check_amount * tax_rate + tip = 20 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l349_34966


namespace NUMINAMATH_CALUDE_journey_distance_l349_34905

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = 224 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l349_34905


namespace NUMINAMATH_CALUDE_power_of_two_sum_l349_34958

theorem power_of_two_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l349_34958


namespace NUMINAMATH_CALUDE_range_of_m_l349_34945

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 15 / 2) * Real.sin (Real.pi * x)

theorem range_of_m (x₀ : ℝ) (h₁ : x₀ ∈ Set.Ioo (-1) 1)
  (h₂ : ∀ x : ℝ, f x ≤ f x₀)
  (h₃ : ∃ m : ℝ, x₀^2 + (f x₀)^2 < m^2) :
  ∃ m : ℝ, m ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l349_34945


namespace NUMINAMATH_CALUDE_correct_equation_for_john_scenario_l349_34988

/-- Represents a driving scenario with a stop -/
structure DrivingScenario where
  speed_before_stop : ℝ
  stop_duration : ℝ
  speed_after_stop : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The given driving scenario -/
def john_scenario : DrivingScenario :=
  { speed_before_stop := 60
  , stop_duration := 0.5
  , speed_after_stop := 80
  , total_distance := 200
  , total_time := 4 }

/-- The equation representing the driving scenario -/
def scenario_equation (s : DrivingScenario) (t : ℝ) : Prop :=
  s.speed_before_stop * t + s.speed_after_stop * (s.total_time - s.stop_duration - t) = s.total_distance

/-- Theorem stating that the equation correctly represents John's driving scenario -/
theorem correct_equation_for_john_scenario :
  ∀ t, scenario_equation john_scenario t ↔ 60 * t + 80 * (7/2 - t) = 200 :=
sorry

end NUMINAMATH_CALUDE_correct_equation_for_john_scenario_l349_34988


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l349_34993

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem consecutive_odd_numbers_equation (N₁ N₂ N₃ : ℤ) : 
  is_odd N₁ ∧ is_odd N₂ ∧ is_odd N₃ ∧ 
  N₂ = N₁ + 2 ∧ N₃ = N₂ + 2 ∧
  N₁ = 9 →
  N₁ ≠ 3 * N₃ + 16 + 4 * N₂ :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l349_34993


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l349_34987

theorem min_value_cos_sin (x : ℝ) : 
  3 * Real.cos x - 4 * Real.sin x ≥ -5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y - 4 * Real.sin y = -5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l349_34987


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l349_34995

/-- Given vectors a and b in ℝ², prove that the magnitude of 2a+b is √13 -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) :
  a = (-2, 1) →
  b = (1, 0) →
  Real.sqrt ((2 * a.1 + b.1)^2 + (2 * a.2 + b.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l349_34995


namespace NUMINAMATH_CALUDE_right_triangle_identification_l349_34908

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle 6 7 8) ∧
  ¬(is_right_triangle 3 4 6) ∧
  ¬(is_right_triangle 7 12 15) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l349_34908


namespace NUMINAMATH_CALUDE_max_value_of_f_l349_34985

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l349_34985


namespace NUMINAMATH_CALUDE_puzzle_cost_calculation_l349_34978

def puzzle_cost (initial_money savings comic_cost final_money : ℕ) : ℕ :=
  initial_money + savings - comic_cost - final_money

theorem puzzle_cost_calculation :
  puzzle_cost 8 13 2 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_cost_calculation_l349_34978


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l349_34920

theorem last_digit_sum_powers : 
  (1993^2002 + 1995^2002) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l349_34920


namespace NUMINAMATH_CALUDE_gcd_of_squares_l349_34926

theorem gcd_of_squares : Nat.gcd (101^2 + 202^2 + 303^2) (100^2 + 201^2 + 304^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l349_34926


namespace NUMINAMATH_CALUDE_tic_tac_toe_winnings_l349_34911

theorem tic_tac_toe_winnings
  (total_games : ℕ)
  (tied_games : ℕ)
  (net_loss : ℤ)
  (h_total : total_games = 100)
  (h_tied : tied_games = 40)
  (h_loss : net_loss = 30)
  (h_win_value : ℤ)
  (h_tie_value : ℤ)
  (h_lose_value : ℤ)
  (h_win_val : h_win_value = 1)
  (h_tie_val : h_tie_value = 0)
  (h_lose_val : h_lose_value = -2)
  : ∃ (won_games : ℕ),
    won_games = 30 ∧
    won_games + tied_games + (total_games - won_games - tied_games) = total_games ∧
    h_win_value * won_games + h_tie_value * tied_games + h_lose_value * (total_games - won_games - tied_games) = -net_loss :=
by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_winnings_l349_34911


namespace NUMINAMATH_CALUDE_circle_theorem_l349_34907

-- Define the circle and points
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def O : ℝ × ℝ := (0, 0)
def r : ℝ := 52

-- Define the points based on the given conditions
theorem circle_theorem (A B : ℝ × ℝ) (P Q : ℝ × ℝ) :
  A ∈ Circle O r →
  B ∈ Circle O r →
  P.1 = 28 ∧ P.2 = 0 →
  (Q.1 - A.1)^2 + (Q.2 - A.2)^2 = 15^2 →
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 15^2 →
  ∃ t : ℝ, Q = (t * B.1, t * B.2) →
  (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = 11^2 :=
by sorry


end NUMINAMATH_CALUDE_circle_theorem_l349_34907


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l349_34902

theorem cubic_expression_equality : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l349_34902


namespace NUMINAMATH_CALUDE_team_selection_ways_l349_34900

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the required team size
def team_size : ℕ := 8

-- Define the number of quadruplets that must be in the team
def required_quadruplets : ℕ := 2

-- Define the number of non-quadruplet players
def non_quadruplet_players : ℕ := total_players - num_quadruplets

-- Define the number of additional players needed after selecting quadruplets
def additional_players : ℕ := team_size - required_quadruplets

-- Theorem statement
theorem team_selection_ways : 
  (Nat.choose num_quadruplets required_quadruplets) * 
  (Nat.choose non_quadruplet_players additional_players) = 18018 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_l349_34900


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l349_34924

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_of_integers 10 30 + count_even_integers 10 30 = 431 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l349_34924


namespace NUMINAMATH_CALUDE_total_tabs_is_322_l349_34991

def browser1_windows : ℕ := 4
def browser1_tabs_per_window : ℕ := 10

def browser2_windows : ℕ := 5
def browser2_tabs_per_window : ℕ := 12

def browser3_windows : ℕ := 6
def browser3_tabs_per_window : ℕ := 15

def browser4_windows : ℕ := browser1_windows
def browser4_tabs_per_window : ℕ := browser1_tabs_per_window + 5

def browser5_windows : ℕ := browser2_windows
def browser5_tabs_per_window : ℕ := browser2_tabs_per_window - 2

def browser6_windows : ℕ := 3
def browser6_tabs_per_window : ℕ := browser3_tabs_per_window / 2

def total_tabs : ℕ := 
  browser1_windows * browser1_tabs_per_window +
  browser2_windows * browser2_tabs_per_window +
  browser3_windows * browser3_tabs_per_window +
  browser4_windows * browser4_tabs_per_window +
  browser5_windows * browser5_tabs_per_window +
  browser6_windows * browser6_tabs_per_window

theorem total_tabs_is_322 : total_tabs = 322 := by
  sorry

end NUMINAMATH_CALUDE_total_tabs_is_322_l349_34991


namespace NUMINAMATH_CALUDE_color_coat_drying_time_l349_34923

/-- Represents the drying time for nail polish coats -/
structure NailPolishDryingTime where
  base_coat : ℕ
  color_coat : ℕ
  top_coat : ℕ
  total_time : ℕ

/-- Theorem: Given the conditions of Jane's nail polish application,
    prove that each color coat takes 3 minutes to dry -/
theorem color_coat_drying_time (t : NailPolishDryingTime)
  (h1 : t.base_coat = 2)
  (h2 : t.top_coat = 5)
  (h3 : t.total_time = 13)
  (h4 : t.total_time = t.base_coat + 2 * t.color_coat + t.top_coat) :
  t.color_coat = 3 := by
  sorry

end NUMINAMATH_CALUDE_color_coat_drying_time_l349_34923


namespace NUMINAMATH_CALUDE_sequence_sum_property_l349_34952

theorem sequence_sum_property (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = n^2 + n) →
  (∀ n : ℕ, a n = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l349_34952


namespace NUMINAMATH_CALUDE_sum_squares_35_consecutive_divisible_by_35_l349_34980

theorem sum_squares_35_consecutive_divisible_by_35 (n : ℕ+) :
  ∃ k : ℤ, (((n + 35) * (n + 36) * (2 * (n + 35) + 1)) / 6 -
            (n * (n + 1) * (2 * n + 1)) / 6) = 35 * k :=
sorry

end NUMINAMATH_CALUDE_sum_squares_35_consecutive_divisible_by_35_l349_34980


namespace NUMINAMATH_CALUDE_percentage_problem_l349_34910

theorem percentage_problem :
  let total := 500
  let unknown_percentage := 50
  let given_percentage := 10
  let result := 25
  (given_percentage / 100) * (unknown_percentage / 100) * total = result :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l349_34910


namespace NUMINAMATH_CALUDE_min_red_chips_l349_34938

/-- Represents a box of colored chips -/
structure ChipBox where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if a ChipBox satisfies the given conditions -/
def isValidChipBox (box : ChipBox) : Prop :=
  box.blue ≥ box.white / 2 ∧
  box.blue ≤ box.red / 3 ∧
  box.white + box.blue ≥ 55

/-- The theorem stating the minimum number of red chips -/
theorem min_red_chips (box : ChipBox) :
  isValidChipBox box → box.red ≥ 57 := by
  sorry

end NUMINAMATH_CALUDE_min_red_chips_l349_34938


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l349_34981

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt 2 / Real.sqrt 110 ∨ m ≥ Real.sqrt 2 / Real.sqrt 110}

/-- The equation of the line with slope m and y-intercept 3 -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) :
  m ∈ possible_slopes ↔
  ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

#check line_intersects_ellipse

end NUMINAMATH_CALUDE_line_intersects_ellipse_l349_34981


namespace NUMINAMATH_CALUDE_largest_undefined_x_l349_34921

theorem largest_undefined_x (f : ℝ → ℝ) :
  (∀ x, f x = (x + 2) / (10 * x^2 - 85 * x + 10)) →
  (∃ x, 10 * x^2 - 85 * x + 10 = 0) →
  (∀ x, 10 * x^2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_undefined_x_l349_34921


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l349_34997

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l349_34997


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l349_34964

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The main theorem -/
theorem complex_fraction_equals_neg_i : (1 - i) / (1 + i) = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l349_34964


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l349_34963

theorem magnitude_of_complex_power : 
  Complex.abs ((2 + 2*Complex.I)^(3+3)) = 512 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l349_34963


namespace NUMINAMATH_CALUDE_golden_ratio_system_solution_l349_34928

theorem golden_ratio_system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 2 * x * Real.sqrt (x + 1) - y * (y + 1) = 1)
  (eq2 : 2 * y * Real.sqrt (y + 1) - z * (z + 1) = 1)
  (eq3 : 2 * z * Real.sqrt (z + 1) - x * (x + 1) = 1) :
  x = (1 + Real.sqrt 5) / 2 ∧ y = (1 + Real.sqrt 5) / 2 ∧ z = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_golden_ratio_system_solution_l349_34928


namespace NUMINAMATH_CALUDE_rational_square_decomposition_l349_34933

theorem rational_square_decomposition (r : ℚ) :
  ∃ (S : Set (ℚ × ℚ)), (Set.Infinite S) ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^2 + y^2 = r^2) :=
sorry

end NUMINAMATH_CALUDE_rational_square_decomposition_l349_34933


namespace NUMINAMATH_CALUDE_car_production_total_l349_34974

/-- The number of cars produced in North America -/
def north_america_cars : ℕ := 3884

/-- The number of cars produced in Europe -/
def europe_cars : ℕ := 2871

/-- The total number of cars produced -/
def total_cars : ℕ := north_america_cars + europe_cars

theorem car_production_total : total_cars = 6755 := by
  sorry

end NUMINAMATH_CALUDE_car_production_total_l349_34974


namespace NUMINAMATH_CALUDE_drawer_is_translation_l349_34998

-- Define the possible transformations
inductive Transformation
  | DrawerMovement
  | MagnifyingGlassEffect
  | ClockHandMovement
  | MirrorReflection

-- Define the properties of a translation
def isTranslation (t : Transformation) : Prop :=
  match t with
  | Transformation.DrawerMovement => true
  | _ => false

-- Theorem statement
theorem drawer_is_translation :
  ∀ t : Transformation, isTranslation t ↔ t = Transformation.DrawerMovement :=
by sorry

end NUMINAMATH_CALUDE_drawer_is_translation_l349_34998
