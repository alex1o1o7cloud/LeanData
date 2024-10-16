import Mathlib

namespace NUMINAMATH_CALUDE_max_planes_for_15_points_l1083_108372

/-- The maximum number of planes determined by n points in space -/
def max_planes (n : ℕ) : ℕ := Nat.choose n 3

/-- The number of points in space -/
def num_points : ℕ := 15

theorem max_planes_for_15_points : max_planes num_points = 455 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_for_15_points_l1083_108372


namespace NUMINAMATH_CALUDE_pumpkin_pies_sold_l1083_108335

/-- Represents the number of pumpkin pies sold -/
def pumpkin_pies : ℕ := sorry

/-- The number of slices in a pumpkin pie -/
def pumpkin_slices : ℕ := 8

/-- The price of a pumpkin pie slice in cents -/
def pumpkin_price : ℕ := 500

/-- The number of slices in a custard pie -/
def custard_slices : ℕ := 6

/-- The price of a custard pie slice in cents -/
def custard_price : ℕ := 600

/-- The number of custard pies sold -/
def custard_pies : ℕ := 5

/-- The total revenue in cents -/
def total_revenue : ℕ := 34000

theorem pumpkin_pies_sold :
  pumpkin_pies * pumpkin_slices * pumpkin_price +
  custard_pies * custard_slices * custard_price = total_revenue →
  pumpkin_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_pies_sold_l1083_108335


namespace NUMINAMATH_CALUDE_min_touches_theorem_l1083_108359

/-- Represents the minimal number of touches required to turn on all lamps in an n×n grid -/
def minTouches (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else n^2

/-- Theorem stating the minimal number of touches required for an n×n grid of lamps -/
theorem min_touches_theorem (n : ℕ) :
  (∀ (grid : Fin n → Fin n → Bool), ∃ (touches : Fin n → Fin n → Bool),
    (∀ i j, touches i j → (∀ k, grid i k = !grid i k ∧ grid k j = !grid k j)) →
    (∀ i j, grid i j = true)) →
  minTouches n = if n % 2 = 1 then n else n^2 :=
sorry

end NUMINAMATH_CALUDE_min_touches_theorem_l1083_108359


namespace NUMINAMATH_CALUDE_new_person_weight_l1083_108377

/-- 
Given a group of 8 people where one person weighing 65 kg is replaced by a new person,
and the average weight of the group increases by 2.5 kg, prove that the weight of the new person is 85 kg.
-/
theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  leaving_weight = 65 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + leaving_weight = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1083_108377


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_m_range_l1083_108341

theorem quadratic_distinct_roots_m_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   m * x^2 + (2*m + 1) * x + m = 0 ∧ 
   m * y^2 + (2*m + 1) * y + m = 0) ↔ 
  (m > -1/4 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_m_range_l1083_108341


namespace NUMINAMATH_CALUDE_m_range_if_f_increasing_l1083_108384

/-- Piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + 2*m*x - 2 else 1 + Real.log x

/-- Theorem stating that if f is increasing, then m is in [1, 2] -/
theorem m_range_if_f_increasing (m : ℝ) :
  (∀ x y, x < y → f m x < f m y) → m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_if_f_increasing_l1083_108384


namespace NUMINAMATH_CALUDE_evaluate_expression_l1083_108351

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1083_108351


namespace NUMINAMATH_CALUDE_l_shape_perimeter_l1083_108385

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the L-shape configuration -/
structure LShape where
  vertical : Rectangle
  horizontal : Rectangle
  overlap : ℝ

/-- Calculates the perimeter of the L-shape -/
def LShape.perimeter (l : LShape) : ℝ :=
  l.vertical.perimeter + l.horizontal.perimeter - 2 * l.overlap

theorem l_shape_perimeter :
  let l : LShape := {
    vertical := { width := 3, height := 6 },
    horizontal := { width := 4, height := 2 },
    overlap := 1
  }
  l.perimeter = 28 := by sorry

end NUMINAMATH_CALUDE_l_shape_perimeter_l1083_108385


namespace NUMINAMATH_CALUDE_f_properties_l1083_108324

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| - |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a ≤ 0) :
  -- Part 1: Solution set when a = 0
  (a = 0 → {x : ℝ | f 0 x < 1} = {x : ℝ | 0 < x ∧ x < 2}) ∧
  -- Part 2: Range of a when triangle area > 3/2
  (∃ (x y : ℝ), x < y ∧ 
    (1/2 * (y - x) * (max (f a x) (f a y))) > 3/2 → a < -1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1083_108324


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1083_108333

theorem trigonometric_equation_solution (t : ℝ) : 
  5.43 * Real.cos (22 * π / 180 - t) * Real.cos (82 * π / 180 - t) + 
  Real.cos (112 * π / 180 - t) * Real.cos (172 * π / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) ↔ 
  (∃ k : ℤ, t = 2 * π * k ∨ t = π / 2 * (4 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1083_108333


namespace NUMINAMATH_CALUDE_cube_net_in_square_l1083_108398

-- Define a square
structure Square where
  side_length : ℝ

-- Define a cube net
structure CubeNet where
  faces : Finset (Square)
  face_count : Nat

-- Define the problem
theorem cube_net_in_square :
  ∃ (large_square : Square) (cube_net : CubeNet),
    large_square.side_length = 3 ∧
    cube_net.face_count = 6 ∧
    ∀ (face : Square), face ∈ cube_net.faces → face.side_length = 1 ∧
    -- The condition that the cube net fits within the large square
    -- is represented by this placeholder
    (cube_net_fits_in_square : Prop) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_net_in_square_l1083_108398


namespace NUMINAMATH_CALUDE_fantasy_creatures_gala_handshakes_l1083_108361

-- Define the number of gremlins and imps
def num_gremlins : ℕ := 30
def num_imps : ℕ := 20

-- Define the number of imps each imp shakes hands with
def imp_imp_handshakes : ℕ := 5

-- Calculate the number of handshakes between gremlins
def gremlin_gremlin_handshakes : ℕ := num_gremlins * (num_gremlins - 1) / 2

-- Calculate the number of handshakes between imps
def imp_imp_total_handshakes : ℕ := num_imps * imp_imp_handshakes / 2

-- Calculate the number of handshakes between gremlins and imps
def gremlin_imp_handshakes : ℕ := num_gremlins * num_imps

-- Define the total number of handshakes
def total_handshakes : ℕ := gremlin_gremlin_handshakes + imp_imp_total_handshakes + gremlin_imp_handshakes

-- Theorem statement
theorem fantasy_creatures_gala_handshakes : total_handshakes = 1085 := by
  sorry

end NUMINAMATH_CALUDE_fantasy_creatures_gala_handshakes_l1083_108361


namespace NUMINAMATH_CALUDE_perfect_square_plus_492_l1083_108393

theorem perfect_square_plus_492 (n : ℕ) : 
  (∃ k : ℕ, n^2 + 492 = k^2) → (n = 122 ∨ n = 38) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_plus_492_l1083_108393


namespace NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_equality_l1083_108370

theorem min_trig_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  3 * Real.cos θ + 2 / Real.sin θ + 2 * Real.sqrt 2 * Real.tan θ ≥ 7 * Real.sqrt 2 / 2 :=
by sorry

theorem min_trig_expression_equality :
  3 * Real.cos (Real.pi / 4) + 2 / Real.sin (Real.pi / 4) + 2 * Real.sqrt 2 * Real.tan (Real.pi / 4) = 7 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_equality_l1083_108370


namespace NUMINAMATH_CALUDE_complex_abs_value_l1083_108317

theorem complex_abs_value : Complex.abs (-3 + (8/5) * Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_value_l1083_108317


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1083_108343

theorem triangle_perimeter (x : ℕ+) : 
  (1 < x) ∧ (x < 5) ∧ (1 + x > 4) ∧ (x + 4 > 1) ∧ (4 + 1 > x) → 
  1 + x + 4 = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1083_108343


namespace NUMINAMATH_CALUDE_min_ab_min_2a_3b_l1083_108394

-- Define the conditions
def positive_reals (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Define the constraint equation
def constraint (a b : ℝ) : Prop := a + b - a * b = 0

-- Theorem for the minimum value of ab
theorem min_ab (a b : ℝ) (h1 : positive_reals a b) (h2 : constraint a b) :
  a * b ≥ 4 ∧ (a * b = 4 ↔ a = 2 ∧ b = 2) :=
sorry

-- Theorem for the minimum value of 2a + 3b
theorem min_2a_3b (a b : ℝ) (h1 : positive_reals a b) (h2 : constraint a b) :
  2 * a + 3 * b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_ab_min_2a_3b_l1083_108394


namespace NUMINAMATH_CALUDE_fraction_equality_l1083_108349

theorem fraction_equality (x : ℝ) (f : ℝ) (h1 : x > 0) (h2 : x = 0.4166666666666667) 
  (h3 : f * x = (25/216) * (1/x)) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1083_108349


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1083_108364

theorem max_value_of_expression (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x + y = 8) :
  (∀ a b : ℝ, a ≥ 1 → b ≥ 1 → a + b = 8 → 
    |Real.sqrt (x - 1/y) + Real.sqrt (y - 1/x)| ≥ |Real.sqrt (a - 1/b) + Real.sqrt (b - 1/a)|) ∧
  |Real.sqrt (x - 1/y) + Real.sqrt (y - 1/x)| ≤ Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1083_108364


namespace NUMINAMATH_CALUDE_hotel_cost_calculation_l1083_108329

def trip_expenses (savings flight_cost food_cost remaining : ℕ) : Prop :=
  ∃ hotel_cost : ℕ, 
    savings = flight_cost + food_cost + hotel_cost + remaining

theorem hotel_cost_calculation (savings flight_cost food_cost remaining : ℕ) 
  (h : trip_expenses savings flight_cost food_cost remaining) : 
  ∃ hotel_cost : ℕ, hotel_cost = 800 ∧ trip_expenses 6000 1200 3000 1000 := by
  sorry

end NUMINAMATH_CALUDE_hotel_cost_calculation_l1083_108329


namespace NUMINAMATH_CALUDE_min_value_theorem_l1083_108331

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  (x + 6) / Real.sqrt (x - 2) ≥ 4 * Real.sqrt 2 ∧
  ((x + 6) / Real.sqrt (x - 2) = 4 * Real.sqrt 2 ↔ x = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1083_108331


namespace NUMINAMATH_CALUDE_lucy_additional_distance_l1083_108318

/-- The length of the field in kilometers -/
def field_length : ℚ := 24

/-- The fraction of the field that Mary ran -/
def mary_fraction : ℚ := 3/8

/-- The fraction of Mary's distance that Edna ran -/
def edna_fraction : ℚ := 2/3

/-- The fraction of Edna's distance that Lucy ran -/
def lucy_fraction : ℚ := 5/6

/-- Mary's running distance in kilometers -/
def mary_distance : ℚ := field_length * mary_fraction

/-- Edna's running distance in kilometers -/
def edna_distance : ℚ := mary_distance * edna_fraction

/-- Lucy's running distance in kilometers -/
def lucy_distance : ℚ := edna_distance * lucy_fraction

theorem lucy_additional_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucy_additional_distance_l1083_108318


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1083_108356

theorem circle_area_ratio (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h_diameter : 2 * r = 0.2 * (2 * s)) : 
  (π * r^2) / (π * s^2) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1083_108356


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1083_108307

/-- The length of the longest side of a triangle with vertices at (3,3), (8,9), and (9,3) is √61 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ),
  a = (3, 3) ∧ b = (8, 9) ∧ c = (9, 3) ∧
  (max (dist a b) (max (dist b c) (dist c a)))^2 = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1083_108307


namespace NUMINAMATH_CALUDE_interior_perimeter_is_20_l1083_108325

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_width : ℝ
  outer_height : ℝ
  border_width : ℝ
  frame_area : ℝ

/-- Calculates the sum of the lengths of the four interior edges of a frame -/
def interior_perimeter (f : Frame) : ℝ :=
  2 * ((f.outer_width - 2 * f.border_width) + (f.outer_height - 2 * f.border_width))

/-- Theorem stating that for a frame with given dimensions, the interior perimeter is 20 inches -/
theorem interior_perimeter_is_20 (f : Frame) 
  (h_outer_width : f.outer_width = 8)
  (h_outer_height : f.outer_height = 10)
  (h_border_width : f.border_width = 2)
  (h_frame_area : f.frame_area = 52) :
  interior_perimeter f = 20 := by
  sorry

#check interior_perimeter_is_20

end NUMINAMATH_CALUDE_interior_perimeter_is_20_l1083_108325


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_reciprocal_bounds_l1083_108328

/-- The maximum and minimum values of the sum of reciprocals of distances from the center to two perpendicular points on an ellipse -/
theorem ellipse_perpendicular_points_sum_reciprocal_bounds
  (a b : ℝ) (ha : 0 < b) (hab : b < a)
  (P Q : ℝ × ℝ)
  (hP : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1)
  (hQ : (Q.1 / a) ^ 2 + (Q.2 / b) ^ 2 = 1)
  (hPOQ : (P.1 * Q.1 + P.2 * Q.2) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt (Q.1^2 + Q.2^2)) = 0) :
  (a + b) / (a * b) ≤ 1 / Real.sqrt (P.1^2 + P.2^2) + 1 / Real.sqrt (Q.1^2 + Q.2^2) ∧
  1 / Real.sqrt (P.1^2 + P.2^2) + 1 / Real.sqrt (Q.1^2 + Q.2^2) ≤ Real.sqrt (2 * (a^2 + b^2)) / (a * b) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_reciprocal_bounds_l1083_108328


namespace NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l1083_108369

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop := k * (1 - k) < 0

-- State the theorem
theorem k_negative_sufficient_not_necessary :
  (∀ k : ℝ, k < 0 → is_hyperbola k) ∧
  (∃ k : ℝ, is_hyperbola k ∧ ¬(k < 0)) :=
sorry

end NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l1083_108369


namespace NUMINAMATH_CALUDE_sector_chord_length_l1083_108332

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, 
    its chord length is 2sin(1) cm. -/
theorem sector_chord_length 
  (r : ℝ) 
  (α : ℝ) 
  (h_area : (1/2) * α * r^2 = 1) 
  (h_perim : 2*r + α*r = 4) : 
  2 * r * Real.sin (α/2) = 2 * Real.sin 1 := by
sorry

end NUMINAMATH_CALUDE_sector_chord_length_l1083_108332


namespace NUMINAMATH_CALUDE_wendy_extraction_cost_l1083_108312

/-- The cost of a dental cleaning in dollars -/
def cleaning_cost : ℕ := 70

/-- The cost of a dental filling in dollars -/
def filling_cost : ℕ := 120

/-- The number of fillings Wendy had -/
def num_fillings : ℕ := 2

/-- The total cost of Wendy's dental bill in dollars -/
def total_bill : ℕ := 5 * filling_cost

/-- The cost of Wendy's tooth extraction in dollars -/
def extraction_cost : ℕ := total_bill - (cleaning_cost + num_fillings * filling_cost)

theorem wendy_extraction_cost : extraction_cost = 290 := by
  sorry

end NUMINAMATH_CALUDE_wendy_extraction_cost_l1083_108312


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1083_108326

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n with a_3 = 2 and a_5 = 8, prove that a_7 = 32 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a3 : a 3 = 2) 
    (h_a5 : a 5 = 8) : 
  a 7 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1083_108326


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1083_108353

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 12) ∣ (n^3 + 105) ∧ 
  ∀ (m : ℕ), m > n → m > 0 → ¬((m + 12) ∣ (m^3 + 105)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1083_108353


namespace NUMINAMATH_CALUDE_unique_solution_l1083_108396

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (h : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 13 ∧ y = 11 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1083_108396


namespace NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l1083_108383

-- 1. Prove that 9999×2222+3333×3334 = 33330000
theorem problem1 : 9999 * 2222 + 3333 * 3334 = 33330000 := by sorry

-- 2. Prove that 96%×25+0.75+0.25 = 25
theorem problem2 : (96 / 100) * 25 + 0.75 + 0.25 = 25 := by sorry

-- 3. Prove that 5/8 + 7/10 + 3/8 + 3/10 = 2
theorem problem3 : 5/8 + 7/10 + 3/8 + 3/10 = 2 := by sorry

-- 4. Prove that 3.7 × 6/5 - 2.2 ÷ 5/6 = 1.8
theorem problem4 : 3.7 * (6/5) - 2.2 / (5/6) = 1.8 := by sorry

end NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l1083_108383


namespace NUMINAMATH_CALUDE_triangle_third_side_exists_l1083_108346

theorem triangle_third_side_exists : ∃ x : ℕ, 
  3 ≤ x ∧ x ≤ 7 ∧ 
  (x + 3 > 5) ∧ (x + 5 > 3) ∧ (3 + 5 > x) ∧
  (x > 5 - 3) ∧ (x < 5 + 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_exists_l1083_108346


namespace NUMINAMATH_CALUDE_not_necessarily_divisible_by_twenty_l1083_108381

theorem not_necessarily_divisible_by_twenty (k : ℤ) (n : ℤ) : 
  n = k * (k + 1) * (k + 2) → (∃ m : ℤ, n = 5 * m) → 
  ¬(∀ (k : ℤ), ∃ (m : ℤ), n = 20 * m) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_divisible_by_twenty_l1083_108381


namespace NUMINAMATH_CALUDE_largest_x_value_l1083_108389

theorem largest_x_value (x : ℝ) : 
  x ≠ 7 → 
  ((x^2 - 5*x - 84) / (x - 7) = 2 / (x + 6)) → 
  x ≤ -5 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_l1083_108389


namespace NUMINAMATH_CALUDE_circles_intersect_l1083_108368

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the distance between centers
def distance_between_centers : ℝ := 2

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > abs (radius1 - radius2) ∧
  distance_between_centers < radius1 + radius2 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l1083_108368


namespace NUMINAMATH_CALUDE_hoopit_hands_l1083_108323

/-- Represents the number of toes on each hand of a Hoopit -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of toes on each hand of a Neglart -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands each Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that each Hoopit has 4 hands -/
theorem hoopit_hands : 
  ∃ (h : ℕ), h = 4 ∧ 
  hoopit_students * h * hoopit_toes_per_hand + 
  neglart_students * neglart_hands * neglart_toes_per_hand = total_toes :=
sorry

end NUMINAMATH_CALUDE_hoopit_hands_l1083_108323


namespace NUMINAMATH_CALUDE_fruit_bowl_problem_l1083_108320

/-- Proves that given a bowl with 14 apples and an unknown number of oranges,
    if removing 15 oranges results in apples being 70% of the remaining fruit,
    then the initial number of oranges was 21. -/
theorem fruit_bowl_problem (initial_oranges : ℕ) : 
  (14 : ℝ) / ((14 : ℝ) + (initial_oranges - 15)) = 0.7 → initial_oranges = 21 :=
by sorry

end NUMINAMATH_CALUDE_fruit_bowl_problem_l1083_108320


namespace NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l1083_108327

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := sorry

theorem prob_same_length_regular_hexagon :
  prob_same_length = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l1083_108327


namespace NUMINAMATH_CALUDE_largest_square_area_l1083_108357

theorem largest_square_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a^2 + b^2 + c^2 = 450 →  -- sum of square areas
  a^2 = 100 →  -- area of square on AB
  c^2 = 225 :=  -- area of largest square (on BC)
by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l1083_108357


namespace NUMINAMATH_CALUDE_m_value_after_subtraction_l1083_108371

theorem m_value_after_subtraction (M : ℝ) : 
  (25 / 100 : ℝ) * M = (55 / 100 : ℝ) * 2500 → 
  M - (10 / 100 : ℝ) * M = 4950 := by
  sorry

end NUMINAMATH_CALUDE_m_value_after_subtraction_l1083_108371


namespace NUMINAMATH_CALUDE_tournament_participants_perfect_square_l1083_108386

-- Define the tournament structure
structure ChessTournament where
  masters : ℕ
  grandmasters : ℕ

-- Define the property that each participant scored half their points against masters
def halfPointsAgainstMasters (t : ChessTournament) : Prop :=
  let totalParticipants := t.masters + t.grandmasters
  (t.masters * (t.masters - 1) + t.grandmasters * (t.grandmasters - 1)) / 2 = t.masters * t.grandmasters

-- Theorem statement
theorem tournament_participants_perfect_square (t : ChessTournament) 
  (h : halfPointsAgainstMasters t) : 
  ∃ n : ℕ, (t.masters + t.grandmasters) = n^2 :=
sorry

end NUMINAMATH_CALUDE_tournament_participants_perfect_square_l1083_108386


namespace NUMINAMATH_CALUDE_equation_solution_l1083_108302

/-- Custom multiplication operation for real numbers -/
def custom_mul (a b : ℝ) : ℝ := a^2 - b^2

/-- Theorem stating the solution to the equation -/
theorem equation_solution :
  ∃ x : ℝ, custom_mul (x + 2) 5 = (x - 5) * (5 + x) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1083_108302


namespace NUMINAMATH_CALUDE_julian_celine_ratio_is_one_l1083_108319

/-- The number of erasers collected by Celine -/
def celine_erasers : ℕ := 10

/-- The number of erasers collected by Julian -/
def julian_erasers : ℕ := celine_erasers

/-- The total number of erasers collected -/
def total_erasers : ℕ := 35

/-- The ratio of erasers collected by Julian to Celine -/
def julian_to_celine_ratio : ℚ := julian_erasers / celine_erasers

theorem julian_celine_ratio_is_one : julian_to_celine_ratio = 1 := by
  sorry

end NUMINAMATH_CALUDE_julian_celine_ratio_is_one_l1083_108319


namespace NUMINAMATH_CALUDE_calculate_expression_l1083_108310

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1083_108310


namespace NUMINAMATH_CALUDE_symmetric_line_is_symmetric_l1083_108388

/-- The point of symmetry -/
def P : ℝ × ℝ := (2, -1)

/-- The equation of the original line: 3x - y - 4 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - y - 4 = 0

/-- The equation of the symmetric line: 3x - y - 7 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 3 * x - y - 7 = 0

/-- Definition of symmetry with respect to a point -/
def is_symmetric (line1 line2 : (ℝ → ℝ → Prop)) (p : ℝ × ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ),
    line1 x1 y1 → line2 x2 y2 →
    (x1 + x2) / 2 = p.1 ∧ (y1 + y2) / 2 = p.2

/-- The main theorem: the symmetric_line is symmetric to the original_line with respect to P -/
theorem symmetric_line_is_symmetric :
  is_symmetric original_line symmetric_line P :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_is_symmetric_l1083_108388


namespace NUMINAMATH_CALUDE_percentage_problem_l1083_108392

theorem percentage_problem (P : ℝ) : P = 20 → (1600 * P / 100 = 650 * P / 100 + 190) := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1083_108392


namespace NUMINAMATH_CALUDE_order_cost_is_43_l1083_108354

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of fries in dollars -/
def fries_cost : ℕ := 2

/-- The number of sandwiches ordered -/
def num_sandwiches : ℕ := 3

/-- The number of sodas ordered -/
def num_sodas : ℕ := 7

/-- The number of fries ordered -/
def num_fries : ℕ := 5

/-- The total cost of the order -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas + fries_cost * num_fries

theorem order_cost_is_43 : total_cost = 43 := by
  sorry

end NUMINAMATH_CALUDE_order_cost_is_43_l1083_108354


namespace NUMINAMATH_CALUDE_symmetry_implies_coordinates_l1083_108390

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given points A(a,-2) and B(3,b) are symmetric with respect to the origin, prove a = -3 and b = 2 -/
theorem symmetry_implies_coordinates (a b : ℝ) 
  (h : symmetric_wrt_origin a (-2) 3 b) : a = -3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_coordinates_l1083_108390


namespace NUMINAMATH_CALUDE_compound_hydrogen_atoms_l1083_108397

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_hydrogen_atoms :
  ∀ (c : Compound),
    c.carbon = 4 →
    c.oxygen = 2 →
    molecularWeight c 12.01 16.00 1.008 = 88 →
    c.hydrogen = 8 := by
  sorry

end NUMINAMATH_CALUDE_compound_hydrogen_atoms_l1083_108397


namespace NUMINAMATH_CALUDE_inequality_proof_l1083_108352

def f (x a : ℝ) : ℝ := |x + a| + 2 * |x - 1|

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ x ∈ Set.Icc 1 2, f x a > x^2 - b + 1) :
  (a + 1/2)^2 + (b + 1/2)^2 > 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1083_108352


namespace NUMINAMATH_CALUDE_min_value_theorem_l1083_108305

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 1 → (a + 1) * (b + 1) / (a * b) ≥ (x + 1) * (y + 1) / (x * y)) ∧
  (x + 1) * (y + 1) / (x * y) = 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1083_108305


namespace NUMINAMATH_CALUDE_unique_bounded_sequence_l1083_108373

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) = (a n + a (n - 1)) / Nat.gcd (a n) (a (n - 1))

def is_bounded_sequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M

theorem unique_bounded_sequence :
  ∀ a : ℕ → ℕ, is_valid_sequence a → is_bounded_sequence a →
    ∀ n : ℕ, a n = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_bounded_sequence_l1083_108373


namespace NUMINAMATH_CALUDE_sixth_number_in_sequence_l1083_108339

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n

theorem sixth_number_in_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_sum : a 2 + a 3 = 24) :
  a 6 = 128 := by
sorry

end NUMINAMATH_CALUDE_sixth_number_in_sequence_l1083_108339


namespace NUMINAMATH_CALUDE_die_probabilities_order_l1083_108358

def is_less_than_2 (n : ℕ) : Bool := n < 2

def is_greater_than_2 (n : ℕ) : Bool := n > 2

def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

def prob_less_than_2 : ℚ := 1 / 6

def prob_greater_than_2 : ℚ := 2 / 3

def prob_odd : ℚ := 1 / 2

theorem die_probabilities_order :
  prob_less_than_2 < prob_odd ∧ prob_odd < prob_greater_than_2 :=
sorry

end NUMINAMATH_CALUDE_die_probabilities_order_l1083_108358


namespace NUMINAMATH_CALUDE_first_three_consecutive_fives_l1083_108345

/-- The sequence of digits formed by concatenating natural numbers -/
def digitSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => sorry  -- Definition of the sequence

/-- The function that returns the digit at a given position in the sequence -/
def digitAt (position : ℕ) : ℕ := sorry

/-- Theorem stating the positions of the first occurrence of three consecutive '5' digits -/
theorem first_three_consecutive_fives :
  ∃ (start : ℕ), start = 100 ∧
    digitAt start = 5 ∧
    digitAt (start + 1) = 5 ∧
    digitAt (start + 2) = 5 ∧
    (∀ (pos : ℕ), pos < start → ¬(digitAt pos = 5 ∧ digitAt (pos + 1) = 5 ∧ digitAt (pos + 2) = 5)) :=
  sorry


end NUMINAMATH_CALUDE_first_three_consecutive_fives_l1083_108345


namespace NUMINAMATH_CALUDE_library_purchase_theorem_l1083_108382

-- Define the types of books
inductive BookType
| SocialScience
| Children

-- Define the price function
def price : BookType → ℕ
| BookType.SocialScience => 40
| BookType.Children => 20

-- Define the total cost function
def totalCost (ss_count : ℕ) (c_count : ℕ) : ℕ :=
  ss_count * price BookType.SocialScience + c_count * price BookType.Children

-- Define the valid purchase plan predicate
def isValidPurchasePlan (ss_count : ℕ) (c_count : ℕ) : Prop :=
  ss_count + c_count ≥ 70 ∧
  c_count = ss_count + 20 ∧
  totalCost ss_count c_count ≤ 2000

-- State the theorem
theorem library_purchase_theorem :
  (totalCost 20 40 = 1600) ∧
  (20 * price BookType.SocialScience = 30 * price BookType.Children + 200) ∧
  (∀ ss_count c_count : ℕ, isValidPurchasePlan ss_count c_count ↔ 
    (ss_count = 25 ∧ c_count = 45) ∨ (ss_count = 26 ∧ c_count = 46)) :=
sorry

end NUMINAMATH_CALUDE_library_purchase_theorem_l1083_108382


namespace NUMINAMATH_CALUDE_problem_solution_l1083_108300

theorem problem_solution : 
  (3 * Real.sqrt 18 / Real.sqrt 2 + Real.sqrt 12 * Real.sqrt 3 = 15) ∧
  ((2 + Real.sqrt 6)^2 - (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 8 + 4 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1083_108300


namespace NUMINAMATH_CALUDE_f_properties_l1083_108367

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 * 2^(-x)
  else if x < 0 then 3 * 2^x - 2^(-x)
  else 0

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x < 0, f x = 3 * 2^x - 2^(-x)) ∧  -- f(x) for x < 0
  f 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_properties_l1083_108367


namespace NUMINAMATH_CALUDE_abs_x_squared_lt_x_solution_set_l1083_108330

theorem abs_x_squared_lt_x_solution_set :
  {x : ℝ | |x| * |x| < x} = {x : ℝ | (0 < x ∧ x < 1) ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_abs_x_squared_lt_x_solution_set_l1083_108330


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1083_108395

/-- Given a cone with slant height 13 cm and height 12 cm, its lateral surface area is 65π cm². -/
theorem cone_lateral_surface_area (s h r : ℝ) : 
  s = 13 → h = 12 → s^2 = h^2 + r^2 → (π * r * s : ℝ) = 65 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1083_108395


namespace NUMINAMATH_CALUDE_train_problem_l1083_108308

/-- 
Given two trains departing simultaneously from points A and B towards each other,
this theorem proves the speeds of the trains and the distance between A and B.
-/
theorem train_problem (p q t : ℝ) (hp : p > 0) (hq : q > 0) (ht : t > 0) :
  ∃ (x y z : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- Speeds and distance are positive
    (p / y = (z - p) / x) ∧  -- Trains meet at distance p from B
    (t * y = q + z - p) ∧   -- Second train's position after t hours
    (t * (x + y) = 2 * z) ∧ -- Total distance traveled by both trains after t hours
    (x = (4 * p - 2 * q) / t) ∧ -- Speed of first train
    (y = 2 * p / t) ∧           -- Speed of second train
    (z = 3 * p - q)             -- Distance between A and B
  := by sorry

end NUMINAMATH_CALUDE_train_problem_l1083_108308


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l1083_108366

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem difference_divisible_by_nine (N : ℕ) :
  ∃ k : ℤ, N - (sum_of_digits N) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l1083_108366


namespace NUMINAMATH_CALUDE_profit_decrease_for_one_loom_l1083_108315

/-- Represents the profit decrease when one loom breaks down for a month -/
def profit_decrease (num_looms : ℕ) (total_sales : ℕ) (manufacturing_expenses : ℕ) (establishment_charges : ℕ) : ℕ :=
  let sales_per_loom := total_sales / num_looms
  let manufacturing_per_loom := manufacturing_expenses / num_looms
  let establishment_per_loom := establishment_charges / num_looms
  sales_per_loom - manufacturing_per_loom - establishment_per_loom

/-- Theorem stating the profit decrease when one loom breaks down for a month -/
theorem profit_decrease_for_one_loom :
  profit_decrease 125 500000 150000 75000 = 2200 := by
  sorry

#eval profit_decrease 125 500000 150000 75000

end NUMINAMATH_CALUDE_profit_decrease_for_one_loom_l1083_108315


namespace NUMINAMATH_CALUDE_negation_equivalence_l1083_108313

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1083_108313


namespace NUMINAMATH_CALUDE_triangle_inequality_l1083_108399

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_S : 2 * S = a + b + c) : 
  a^n / (b + c) + b^n / (c + a) + c^n / (a + b) ≥ (2/3)^(n-2) * S^(n-1) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1083_108399


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1083_108309

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 7 ∧ 
  (∃ n : ℕ, c = 2 * n + 1) ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1083_108309


namespace NUMINAMATH_CALUDE_initial_tree_height_l1083_108314

/-- Represents the growth of a tree over time -/
def TreeGrowth (initial_height growth_rate years final_height : ℝ) : Prop :=
  initial_height + growth_rate * years = final_height

theorem initial_tree_height : 
  ∃ (h : ℝ), TreeGrowth h 5 5 29 ∧ h = 4 := by sorry

end NUMINAMATH_CALUDE_initial_tree_height_l1083_108314


namespace NUMINAMATH_CALUDE_preimage_of_three_one_l1083_108303

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem stating that (1, 1) is the pre-image of (3, 1) under f -/
theorem preimage_of_three_one :
  f (1, 1) = (3, 1) ∧ ∀ p : ℝ × ℝ, f p = (3, 1) → p = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_three_one_l1083_108303


namespace NUMINAMATH_CALUDE_equation_has_real_root_l1083_108379

theorem equation_has_real_root (K : ℝ) (h : K ≠ 0) :
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l1083_108379


namespace NUMINAMATH_CALUDE_rectangle_area_l1083_108321

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1083_108321


namespace NUMINAMATH_CALUDE_min_value_theorem_l1083_108306

/-- A monotonically increasing function on ℝ of the form f(x) = a^x + b -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a^x + b

/-- The theorem stating the minimum value of the expression -/
theorem min_value_theorem (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : f a b 1 = 3) :
  (4 / (a - 1) + 1 / b) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1083_108306


namespace NUMINAMATH_CALUDE_function_minimum_value_l1083_108336

theorem function_minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ((x^2) / (y - 2) + (y^2) / (x - 2)) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_value_l1083_108336


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l1083_108387

theorem matching_shoes_probability (n : ℕ) (h : n = 100) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 199 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l1083_108387


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l1083_108301

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℚ) (new_person_weight : ℚ) :
  initial_people = 6 →
  initial_avg_weight = 160 →
  new_person_weight = 97 →
  let total_weight : ℚ := initial_people * initial_avg_weight + new_person_weight
  let new_people : ℕ := initial_people + 1
  let new_avg_weight : ℚ := total_weight / new_people
  new_avg_weight = 151 := by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l1083_108301


namespace NUMINAMATH_CALUDE_distance_between_shores_is_600_l1083_108348

/-- Represents the distance between two shores --/
def distance_between_shores : ℝ := sorry

/-- Represents the distance of the first meeting point from shore A --/
def first_meeting_distance : ℝ := 500

/-- Represents the distance of the second meeting point from shore B --/
def second_meeting_distance : ℝ := 300

/-- Theorem stating that the distance between shores A and B is 600 yards --/
theorem distance_between_shores_is_600 :
  distance_between_shores = 600 :=
sorry

end NUMINAMATH_CALUDE_distance_between_shores_is_600_l1083_108348


namespace NUMINAMATH_CALUDE_distance_between_mum_and_turbo_l1083_108347

/-- The distance between Usain's mum and Turbo when Usain has run 100 meters -/
theorem distance_between_mum_and_turbo (usain_speed mum_speed turbo_speed : ℝ) : 
  usain_speed = 2 * mum_speed →
  mum_speed = 5 * turbo_speed →
  usain_speed > 0 →
  (100 / usain_speed) * mum_speed - (100 / usain_speed) * turbo_speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_mum_and_turbo_l1083_108347


namespace NUMINAMATH_CALUDE_g_is_odd_and_f_negative_two_l1083_108337

/-- The function f(x) -/
noncomputable def f (x m n : ℝ) : ℝ := (2^x - 2^(-x)) * m + (x^3 + x) * n + x^2 - 1

/-- The function g(x) -/
noncomputable def g (x m n : ℝ) : ℝ := (2^x - 2^(-x)) * m + (x^3 + x) * n

theorem g_is_odd_and_f_negative_two (m n : ℝ) :
  (∀ x, g (-x) m n = -g x m n) ∧ (f 2 m n = 8 → f (-2) m n = -2) :=
sorry

end NUMINAMATH_CALUDE_g_is_odd_and_f_negative_two_l1083_108337


namespace NUMINAMATH_CALUDE_book_purchase_theorem_l1083_108365

/-- Represents the unit price of type A books -/
def price_A : ℕ := 30

/-- Represents the unit price of type B books -/
def price_B : ℕ := 20

/-- The total number of books to be purchased -/
def total_books : ℕ := 40

/-- The maximum budget for book purchase -/
def max_budget : ℕ := 980

/-- Theorem stating the correctness of book prices and maximum number of type A books -/
theorem book_purchase_theorem :
  (price_A = price_B + 10) ∧
  (3 * price_A + 2 * price_B = 130) ∧
  (∀ a : ℕ, a ≤ total_books → 
    a * price_A + (total_books - a) * price_B ≤ max_budget →
    a ≤ 18) ∧
  (18 * price_A + (total_books - 18) * price_B ≤ max_budget) :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_theorem_l1083_108365


namespace NUMINAMATH_CALUDE_caseys_corn_rows_l1083_108391

/-- Represents the problem of calculating the number of corn plant rows Casey can water --/
theorem caseys_corn_rows :
  let pump_rate : ℚ := 3  -- gallons per minute
  let pump_time : ℕ := 25  -- minutes
  let plants_per_row : ℕ := 15
  let water_per_plant : ℚ := 1/2  -- gallons
  let num_pigs : ℕ := 10
  let water_per_pig : ℚ := 4  -- gallons
  let num_ducks : ℕ := 20
  let water_per_duck : ℚ := 1/4  -- gallons
  
  let total_water : ℚ := pump_rate * pump_time
  let water_for_animals : ℚ := num_pigs * water_per_pig + num_ducks * water_per_duck
  let water_for_plants : ℚ := total_water - water_for_animals
  let num_plants : ℚ := water_for_plants / water_per_plant
  let num_rows : ℚ := num_plants / plants_per_row
  
  num_rows = 4 := by sorry

end NUMINAMATH_CALUDE_caseys_corn_rows_l1083_108391


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l1083_108374

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x * (7 - x) ≥ 12
def inequality2 (x : ℝ) : Prop := x^2 > 2 * (x - 1)

-- Define the solution sets
def solution_set1 : Set ℝ := {x | 3 ≤ x ∧ x ≤ 4}
def solution_set2 : Set ℝ := Set.univ

-- Theorem statements
theorem inequality1_solution : {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem inequality2_solution : {x : ℝ | inequality2 x} = solution_set2 := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l1083_108374


namespace NUMINAMATH_CALUDE_percentage_of_m_l1083_108334

theorem percentage_of_m (j k l m : ℝ) : 
  (1.25 * j = 0.25 * k) →
  (1.5 * k = 0.5 * l) →
  (∃ p, 1.75 * l = p / 100 * m) →
  (0.2 * m = 7 * j) →
  (∃ p, 1.75 * l = p / 100 * m ∧ p = 75) := by
sorry

end NUMINAMATH_CALUDE_percentage_of_m_l1083_108334


namespace NUMINAMATH_CALUDE_pentagon_angle_Q_measure_l1083_108304

-- Define the sum of angles in a pentagon
def pentagon_angle_sum : ℝ := 540

-- Define the known angles
def angle1 : ℝ := 130
def angle2 : ℝ := 90
def angle3 : ℝ := 110
def angle4 : ℝ := 115

-- Define the relation between Q and R
def Q_R_relation (Q R : ℝ) : Prop := Q = 2 * R

-- Theorem statement
theorem pentagon_angle_Q_measure :
  ∀ Q R : ℝ,
  Q_R_relation Q R →
  angle1 + angle2 + angle3 + angle4 + Q + R = pentagon_angle_sum →
  Q = 63.33 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_angle_Q_measure_l1083_108304


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1083_108338

/-- A quadratic function passing through three given points -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_proof (a b c : ℝ) :
  (quadratic_function a b c 1 = 5) ∧
  (quadratic_function a b c 0 = 3) ∧
  (quadratic_function a b c (-1) = -3) →
  (∀ x, quadratic_function a b c x = -2 * x^2 + 4 * x + 3) ∧
  (∃ x y, x = 1 ∧ y = 5 ∧ ∀ t, quadratic_function a b c t ≤ quadratic_function a b c x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1083_108338


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l1083_108340

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication_result : i^2 * (1 + i) = -1 - i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l1083_108340


namespace NUMINAMATH_CALUDE_patsy_appetizers_l1083_108375

def guests : ℕ := 30
def appetizers_per_guest : ℕ := 6
def deviled_eggs_dozens : ℕ := 3
def pigs_in_blanket_dozens : ℕ := 2
def kebabs_dozens : ℕ := 2
def appetizers_per_dozen : ℕ := 12

theorem patsy_appetizers : 
  (guests * appetizers_per_guest - 
   (deviled_eggs_dozens + pigs_in_blanket_dozens + kebabs_dozens) * appetizers_per_dozen) / 
   appetizers_per_dozen = 8 := by
sorry

end NUMINAMATH_CALUDE_patsy_appetizers_l1083_108375


namespace NUMINAMATH_CALUDE_farmer_duck_sales_l1083_108376

/-- A farmer sells ducks and chickens, buys a wheelbarrow, and resells it. -/
theorem farmer_duck_sales
  (duck_price : ℕ)
  (chicken_price : ℕ)
  (chicken_count : ℕ)
  (duck_count : ℕ)
  (wheelbarrow_profit : ℕ)
  (h1 : duck_price = 10)
  (h2 : chicken_price = 8)
  (h3 : chicken_count = 5)
  (h4 : wheelbarrow_profit = 60)
  (h5 : (duck_price * duck_count + chicken_price * chicken_count) / 2 = wheelbarrow_profit / 2) :
  duck_count = 2 := by
sorry


end NUMINAMATH_CALUDE_farmer_duck_sales_l1083_108376


namespace NUMINAMATH_CALUDE_expression_factorization_l1083_108380

theorem expression_factorization (x : ℝ) :
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1083_108380


namespace NUMINAMATH_CALUDE_girls_in_school_l1083_108344

/-- The number of girls in a school given stratified sampling information -/
theorem girls_in_school (total : ℕ) (sample : ℕ) (girl_sample : ℕ) 
  (h_total : total = 1600)
  (h_sample : sample = 200)
  (h_girl_sample : girl_sample = 95)
  (h_ratio : (girl_sample : ℚ) / sample = (↑girls : ℚ) / total) :
  girls = 760 :=
sorry

end NUMINAMATH_CALUDE_girls_in_school_l1083_108344


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1083_108311

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, (x = a ∧ y = b) ∨ (x = b ∧ y = a) → x * y = k

theorem inverse_proportion_problem (a b : ℝ) :
  InverselyProportional a b →
  (∃ a₀ b₀ : ℝ, a₀ + b₀ = 60 ∧ a₀ = 3 * b₀ ∧ InverselyProportional a₀ b₀) →
  (a = -12 → b = -225/4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1083_108311


namespace NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l1083_108363

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_zero_at_seven_fifths : g (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l1083_108363


namespace NUMINAMATH_CALUDE_billy_initial_dandelions_l1083_108378

/-- The number of dandelions Billy picked initially -/
def billy_initial : ℕ := sorry

/-- The number of dandelions George picked initially -/
def george_initial : ℕ := sorry

/-- The number of additional dandelions each person picked -/
def additional_picks : ℕ := 10

/-- The average number of dandelions picked -/
def average_picks : ℕ := 34

theorem billy_initial_dandelions :
  billy_initial = 36 ∧
  george_initial = billy_initial / 3 ∧
  (billy_initial + george_initial + 2 * additional_picks) / 2 = average_picks :=
sorry

end NUMINAMATH_CALUDE_billy_initial_dandelions_l1083_108378


namespace NUMINAMATH_CALUDE_palm_meadows_beds_l1083_108316

theorem palm_meadows_beds (total_rooms : ℕ) (rooms_with_fewer_beds : ℕ) (beds_in_other_rooms : ℕ) (total_beds : ℕ) :
  total_rooms = 13 →
  rooms_with_fewer_beds = 8 →
  total_rooms - rooms_with_fewer_beds = 5 →
  beds_in_other_rooms = 3 →
  total_beds = 31 →
  (rooms_with_fewer_beds * 2) + ((total_rooms - rooms_with_fewer_beds) * beds_in_other_rooms) = total_beds :=
by
  sorry

end NUMINAMATH_CALUDE_palm_meadows_beds_l1083_108316


namespace NUMINAMATH_CALUDE_intercepted_arc_is_sixty_degrees_l1083_108355

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a circle
structure Circle where
  radius : ℝ
  radius_positive : radius > 0

-- Define the relationship between the triangle and circle
def circle_radius_equals_triangle_height (t : EquilateralTriangle) (c : Circle) : Prop :=
  c.radius = t.side * Real.sqrt 3 / 2

-- Define the arc intercepted by the sides of the triangle
def intercepted_arc (t : EquilateralTriangle) (c : Circle) : ℝ := sorry

-- Theorem statement
theorem intercepted_arc_is_sixty_degrees 
  (t : EquilateralTriangle) (c : Circle) 
  (h : circle_radius_equals_triangle_height t c) : 
  intercepted_arc t c = 60 := by sorry

end NUMINAMATH_CALUDE_intercepted_arc_is_sixty_degrees_l1083_108355


namespace NUMINAMATH_CALUDE_cos_50_cos_20_plus_sin_50_sin_20_l1083_108362

theorem cos_50_cos_20_plus_sin_50_sin_20 :
  Real.cos (50 * π / 180) * Real.cos (20 * π / 180) + Real.sin (50 * π / 180) * Real.sin (20 * π / 180) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cos_50_cos_20_plus_sin_50_sin_20_l1083_108362


namespace NUMINAMATH_CALUDE_area_relationship_l1083_108360

/-- A circle circumscribed about a right triangle with sides 12, 35, and 37 -/
structure CircumscribedTriangle where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The area of the largest non-triangular region -/
  C : ℝ
  /-- The sum of the areas of the two smaller non-triangular regions -/
  A_plus_B : ℝ
  /-- The radius is half of the hypotenuse -/
  radius_eq : radius = 37 / 2
  /-- The largest non-triangular region is a semicircle -/
  C_eq : C = π * radius^2 / 2
  /-- The sum of all regions equals the circle's area -/
  area_eq : A_plus_B + 210 + C = π * radius^2

/-- The relationship between the areas of the non-triangular regions -/
theorem area_relationship (t : CircumscribedTriangle) : t.A_plus_B + 210 = t.C := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l1083_108360


namespace NUMINAMATH_CALUDE_intersection_A_B_values_a_b_l1083_108342

-- Define sets A and B
def A : Set ℝ := {x | 4 - x^2 > 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 1} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | 2*x^2 + a*x + b < 0}

-- Theorem for the values of a and b
theorem values_a_b : 
  ∃ a b : ℝ, quadratic_inequality a b = B ∧ a = 4 ∧ b = -6 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_values_a_b_l1083_108342


namespace NUMINAMATH_CALUDE_pages_per_comic_l1083_108322

theorem pages_per_comic (total_pages : ℕ) (initial_comics : ℕ) (final_comics : ℕ)
  (h1 : total_pages = 150)
  (h2 : initial_comics = 5)
  (h3 : final_comics = 11) :
  total_pages / (final_comics - initial_comics) = 25 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_comic_l1083_108322


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1083_108350

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -5033 [ZMOD 12] := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1083_108350
