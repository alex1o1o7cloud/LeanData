import Mathlib

namespace uncle_jerry_tomatoes_l2951_295117

theorem uncle_jerry_tomatoes (day1 day2 total : ℕ) 
  (h1 : day2 = day1 + 50)
  (h2 : day1 + day2 = total)
  (h3 : total = 290) : 
  day1 = 120 := by
sorry

end uncle_jerry_tomatoes_l2951_295117


namespace hexagon_semicircles_area_l2951_295132

/-- The area of the region inside a regular hexagon with side length 4, 
    but outside eight semicircles (where each semicircle's diameter 
    coincides with each side of the hexagon) -/
theorem hexagon_semicircles_area : 
  let s : ℝ := 4 -- side length of the hexagon
  let r : ℝ := s / 2 -- radius of each semicircle
  let hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area : ℝ := 8 * (Real.pi * r^2 / 2)
  hexagon_area - semicircle_area = 24 * Real.sqrt 3 - 16 * Real.pi :=
by sorry

end hexagon_semicircles_area_l2951_295132


namespace remainder_of_binary_div_8_l2951_295138

/-- The binary representation of the number --/
def binary_num : List Bool := [true, true, false, true, true, true, false, false, true, false, true, true]

/-- Convert a binary number to decimal --/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Get the last three digits of a binary number --/
def last_three_digits (binary : List Bool) : List Bool :=
  binary.reverse.take 3

theorem remainder_of_binary_div_8 :
  binary_to_decimal (last_three_digits binary_num) % 8 = 3 := by
  sorry

end remainder_of_binary_div_8_l2951_295138


namespace shaded_cubes_count_l2951_295186

/-- Represents a 3x3x3 cube made up of smaller cubes -/
structure LargeCube :=
  (small_cubes : Fin 3 → Fin 3 → Fin 3 → Bool)

/-- Represents the shading pattern on a face of the large cube -/
inductive FaceShading
  | FourCorners
  | LShape

/-- Represents the shading of opposite faces -/
structure OppositeShading :=
  (face1 : FaceShading)
  (face2 : FaceShading)

/-- The shading pattern for all three pairs of opposite faces -/
def cube_shading : Fin 3 → OppositeShading :=
  λ _ => { face1 := FaceShading.FourCorners, face2 := FaceShading.LShape }

/-- Counts the number of smaller cubes with at least one face shaded -/
def count_shaded_cubes (c : LargeCube) (shading : Fin 3 → OppositeShading) : Nat :=
  sorry

theorem shaded_cubes_count :
  ∀ c : LargeCube,
  count_shaded_cubes c cube_shading = 17 :=
sorry

end shaded_cubes_count_l2951_295186


namespace shopping_total_proof_l2951_295133

def toy_count : ℕ := 5
def toy_price : ℚ := 3
def toy_discount : ℚ := 0.20

def book_count : ℕ := 3
def book_price : ℚ := 8
def book_discount : ℚ := 0.15

def shirt_count : ℕ := 2
def shirt_price : ℚ := 12
def shirt_discount : ℚ := 0.25

def total_paid : ℚ := 50.40

theorem shopping_total_proof :
  (toy_count : ℚ) * toy_price * (1 - toy_discount) +
  (book_count : ℚ) * book_price * (1 - book_discount) +
  (shirt_count : ℚ) * shirt_price * (1 - shirt_discount) = total_paid :=
by sorry

end shopping_total_proof_l2951_295133


namespace factory_bulb_reliability_l2951_295194

theorem factory_bulb_reliability 
  (factory_x_reliability : ℝ) 
  (factory_x_supply : ℝ) 
  (total_reliability : ℝ) 
  (h1 : factory_x_reliability = 0.59) 
  (h2 : factory_x_supply = 0.60) 
  (h3 : total_reliability = 0.62) :
  let factory_y_supply := 1 - factory_x_supply
  let factory_y_reliability := (total_reliability - factory_x_supply * factory_x_reliability) / factory_y_supply
  factory_y_reliability = 0.665 := by
sorry

end factory_bulb_reliability_l2951_295194


namespace equation_solution_l2951_295155

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 6*x * Real.sqrt (x + 5) - 52
  ∃ (x₁ x₂ : ℝ), x₁ = (9 - Real.sqrt 5) / 2 ∧ x₂ = (9 + Real.sqrt 5) / 2 ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end equation_solution_l2951_295155


namespace parabola_tangent_and_circle_l2951_295139

/-- Given a parabola y = x^2 and point P (1, -1), this theorem proves:
    1. The x-coordinates of the tangent points M and N, where x₁ < x₂, are x₁ = 1 - √2 and x₂ = 1 + √2.
    2. The area of a circle with center P tangent to line MN is 16π/5. -/
theorem parabola_tangent_and_circle (x₁ x₂ : ℝ) :
  let P : ℝ × ℝ := (1, -1)
  let T₀ : ℝ → ℝ := λ x => x^2
  let is_tangent (x : ℝ) := T₀ x = (x - 1)^2 - 1 ∧ 2*x = (x^2 + 1) / (x - 1)
  x₁ < x₂ ∧ is_tangent x₁ ∧ is_tangent x₂ →
  (x₁ = 1 - Real.sqrt 2 ∧ x₂ = 1 + Real.sqrt 2) ∧
  (π * ((2 * 1 + 1 + 1) / Real.sqrt (4 + 1))^2 = 16 * π / 5) := by
sorry

end parabola_tangent_and_circle_l2951_295139


namespace chemical_quantity_problem_l2951_295157

theorem chemical_quantity_problem (x : ℤ) : 
  532 * x - 325 * x = 1065430 → x = 5148 := by sorry

end chemical_quantity_problem_l2951_295157


namespace division_theorem_l2951_295172

theorem division_theorem (M q : ℤ) (h : M = 54 * q + 37) :
  ∃ (k : ℤ), M = 18 * k + 1 ∧ k = 3 * q + 2 := by
  sorry

end division_theorem_l2951_295172


namespace f_of_3_eq_3_l2951_295142

/-- The function f satisfying the given equation for all x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The main equation defining f -/
axiom f_eq (x : ℝ) : (x^(3^5 - 1) - 1) * f x = (x + 1) * (x^2 + 1) * (x^3 + 1) * (x^(3^4) + 1) - 1

/-- Theorem stating that f(3) = 3 -/
theorem f_of_3_eq_3 : f 3 = 3 := by sorry

end f_of_3_eq_3_l2951_295142


namespace delta_value_l2951_295130

theorem delta_value : ∀ Δ : ℤ, 4 * 3 = Δ - 6 → Δ = 18 := by
  sorry

end delta_value_l2951_295130


namespace perpendicular_line_tangent_cubic_l2951_295156

/-- Given a line ax - by - 2 = 0 perpendicular to the tangent of y = x^3 at (1,1), prove a/b = -1/3 -/
theorem perpendicular_line_tangent_cubic (a b : ℝ) : 
  (∀ x y : ℝ, a * x - b * y - 2 = 0 → 
    (x - 1) * (3 * (1 : ℝ)^2) + (y - 1) = 0) → 
  a / b = -1/3 := by
sorry

end perpendicular_line_tangent_cubic_l2951_295156


namespace consecutive_draws_count_l2951_295115

/-- The number of ways to draw 4 consecutively numbered balls from a set of 20 balls. -/
def consecutiveDraws : ℕ := 17

/-- The total number of balls in the bin. -/
def totalBalls : ℕ := 20

/-- The number of balls to be drawn. -/
def ballsDrawn : ℕ := 4

theorem consecutive_draws_count :
  consecutiveDraws = totalBalls - ballsDrawn + 1 :=
by sorry

end consecutive_draws_count_l2951_295115


namespace die_roll_probability_l2951_295159

theorem die_roll_probability : 
  let p : ℚ := 1/3  -- probability of rolling a number divisible by 3
  let n : ℕ := 8    -- number of rolls
  1 - (1 - p)^n = 6305/6561 := by
sorry

end die_roll_probability_l2951_295159


namespace largest_number_problem_l2951_295119

theorem largest_number_problem (a b c : ℝ) : 
  a < b → b < c →
  a + b + c = 100 →
  c - b = 8 →
  b - a = 4 →
  c = 40 := by
sorry

end largest_number_problem_l2951_295119


namespace fractional_equation_solution_l2951_295196

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 1) / x = 2 / 3 ∧ x = -3 :=
by
  -- Proof goes here
  sorry

end fractional_equation_solution_l2951_295196


namespace circle_equation_through_point_l2951_295177

/-- The equation of a circle with center (1, 0) passing through (1, -1) -/
theorem circle_equation_through_point :
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let equation (x y : ℝ) := (x - center.1)^2 + (y - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2
  ∀ x y : ℝ, equation x y ↔ (x - 1)^2 + y^2 = 1 :=
by
  sorry


end circle_equation_through_point_l2951_295177


namespace inequality_system_sum_l2951_295151

theorem inequality_system_sum (a b : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x + 2*a > 4 ∧ 2*x < b)) →
  a + b = 6 := by
sorry

end inequality_system_sum_l2951_295151


namespace certain_number_proof_l2951_295181

theorem certain_number_proof : ∃ x : ℝ, x * 16 = 3408 ∧ 16 * 21.3 = 340.8 → x = 213 := by
  sorry

end certain_number_proof_l2951_295181


namespace f_properties_l2951_295185

def f (x : ℝ) : ℝ := -x - x^3

theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end f_properties_l2951_295185


namespace triangle_y_coordinate_l2951_295183

/-- Given a triangle with vertices (-1, 0), (7, y), and (7, -4), if its area is 32, then y = 4 -/
theorem triangle_y_coordinate (y : ℝ) : 
  let vertices := [(-1, 0), (7, y), (7, -4)]
  let area := (1/2 : ℝ) * |(-1 * (y - (-4)) + 7 * ((-4) - 0) + 7 * (0 - y))|
  area = 32 → y = 4 := by sorry

end triangle_y_coordinate_l2951_295183


namespace min_value_trig_expression_l2951_295179

theorem min_value_trig_expression :
  ∃ (x : ℝ), ∀ (y : ℝ),
    (Real.sin y)^8 + (Real.cos y)^8 + 1
    ≤ ((Real.sin x)^8 + (Real.cos x)^8 + 1) / ((Real.sin x)^6 + (Real.cos x)^6 + 1)
    ∧ ((Real.sin x)^8 + (Real.cos x)^8 + 1) / ((Real.sin x)^6 + (Real.cos x)^6 + 1) = 1/2 := by
  sorry

end min_value_trig_expression_l2951_295179


namespace smallest_sum_of_squares_with_diff_217_l2951_295135

theorem smallest_sum_of_squares_with_diff_217 :
  ∃ (x y : ℕ), 
    x^2 - y^2 = 217 ∧
    ∀ (a b : ℕ), a^2 - b^2 = 217 → x^2 + y^2 ≤ a^2 + b^2 ∧
    x^2 + y^2 = 505 :=
by sorry

end smallest_sum_of_squares_with_diff_217_l2951_295135


namespace billion_to_scientific_notation_l2951_295166

def billion : ℝ := 10^9

theorem billion_to_scientific_notation :
  let value : ℝ := 27.58 * billion
  value = 2.758 * 10^10 := by sorry

end billion_to_scientific_notation_l2951_295166


namespace min_sum_squares_l2951_295102

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

end min_sum_squares_l2951_295102


namespace circle_sum_of_center_and_radius_l2951_295192

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*y + 81 = -y^2 + 14*x

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_of_center_and_radius :
  ∃ a b r, circle_center_radius a b r ∧ a + b + r = 15 + 4 * Real.sqrt 2 :=
sorry

end circle_sum_of_center_and_radius_l2951_295192


namespace solution_set_for_a_eq_1_a_range_for_interval_containment_l2951_295188

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part (1)
theorem solution_set_for_a_eq_1 :
  let a := 1
  ∃ (S : Set ℝ), S = {x | f a x ≥ g x} ∧ S = Set.Icc (-1) ((Real.sqrt 17 - 1) / 2) :=
sorry

-- Theorem for part (2)
theorem a_range_for_interval_containment :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≥ g x) → a ∈ Set.Icc (-1) 1 :=
sorry

end solution_set_for_a_eq_1_a_range_for_interval_containment_l2951_295188


namespace complex_fraction_calculation_l2951_295129

theorem complex_fraction_calculation : 
  ∃ ε > 0, |((9/20 : ℚ) - 11/30 + 13/42 - 15/56 + 17/72) * 120 - (1/3) / (1/4) - 42| < ε :=
by sorry

end complex_fraction_calculation_l2951_295129


namespace arithmetic_calculation_l2951_295107

theorem arithmetic_calculation : (21 / (6 + 1 - 4)) * 5 = 35 := by
  sorry

end arithmetic_calculation_l2951_295107


namespace reciprocal_of_negative_twenty_eight_l2951_295147

theorem reciprocal_of_negative_twenty_eight :
  (1 : ℚ) / (-28 : ℚ) = -1 / 28 := by
  sorry

end reciprocal_of_negative_twenty_eight_l2951_295147


namespace no_common_multiple_in_factors_of_600_l2951_295114

theorem no_common_multiple_in_factors_of_600 : 
  ∀ n : ℕ, n ∣ 600 → ¬(30 ∣ n ∧ 42 ∣ n ∧ 56 ∣ n) :=
by
  sorry

end no_common_multiple_in_factors_of_600_l2951_295114


namespace problem_solution_l2951_295126

theorem problem_solution (y : ℝ) (hy : y ≠ 0) : (9 * y)^18 = (27 * y)^9 → y = 1/3 := by
  sorry

end problem_solution_l2951_295126


namespace p_squared_plus_18_composite_l2951_295167

theorem p_squared_plus_18_composite (p : ℕ) (hp : Prime p) : ¬ Prime (p^2 + 18) := by
  sorry

end p_squared_plus_18_composite_l2951_295167


namespace quiz_points_l2951_295125

theorem quiz_points (n : ℕ) (total : ℕ) (r : ℕ) (h1 : n = 12) (h2 : total = 8190) (h3 : r = 2) :
  let first_question_points := total / (r^n - 1)
  let fifth_question_points := first_question_points * r^4
  fifth_question_points = 32 := by
sorry

end quiz_points_l2951_295125


namespace exists_valid_selection_l2951_295169

/-- A vertex of a polygon with two distinct numbers -/
structure Vertex :=
  (num1 : ℕ)
  (num2 : ℕ)
  (distinct : num1 ≠ num2)

/-- A convex 100-gon with two numbers at each vertex -/
def Polygon := Fin 100 → Vertex

/-- A selection of numbers from the vertices -/
def Selection := Fin 100 → ℕ

/-- Predicate to check if a selection is valid (no adjacent vertices have the same number) -/
def ValidSelection (p : Polygon) (s : Selection) : Prop :=
  ∀ i : Fin 100, s i ≠ s (i + 1)

/-- Theorem stating that for any 100-gon with two distinct numbers at each vertex,
    there exists a valid selection of numbers -/
theorem exists_valid_selection (p : Polygon) :
  ∃ s : Selection, ValidSelection p s ∧ ∀ i : Fin 100, s i = (p i).num1 ∨ s i = (p i).num2 :=
sorry

end exists_valid_selection_l2951_295169


namespace largest_gold_coins_distribution_l2951_295100

theorem largest_gold_coins_distribution (n : ℕ) : 
  (n % 13 = 3) →  -- Condition: 3 people receive an extra coin after equal distribution
  (n < 150) →     -- Condition: Total coins less than 150
  (∀ m : ℕ, (m % 13 = 3) ∧ (m < 150) → m ≤ n) →  -- n is the largest number satisfying conditions
  n = 146 :=      -- Conclusion: The largest number of coins is 146
by sorry

end largest_gold_coins_distribution_l2951_295100


namespace shortest_horizontal_distance_l2951_295120

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - x - 6

/-- Theorem stating the shortest horizontal distance -/
theorem shortest_horizontal_distance :
  ∃ (x₁ x₂ : ℝ),
    f x₁ = 6 ∧
    f x₂ = -6 ∧
    ∀ (y₁ y₂ : ℝ),
      f y₁ = 6 →
      f y₂ = -6 →
      |x₁ - x₂| ≤ |y₁ - y₂| ∧
      |x₁ - x₂| = 3 :=
sorry

end shortest_horizontal_distance_l2951_295120


namespace inequality_proof_l2951_295170

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end inequality_proof_l2951_295170


namespace parallelepiped_volume_l2951_295182

def v1 : ℝ × ℝ × ℝ := (1, 3, 4)
def v2 (k : ℝ) : ℝ × ℝ × ℝ := (2, k, 1)
def v3 (k : ℝ) : ℝ × ℝ × ℝ := (1, 1, k)

def volume (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  (a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2) -
  (a3 * b2 * c1 + a1 * b3 * c2 + a2 * b1 * c3)

theorem parallelepiped_volume (k : ℝ) :
  k > 0 ∧ volume v1 (v2 k) (v3 k) = 12 ↔
  k = 5 + Real.sqrt 26 ∨ k = 5 + Real.sqrt 2 ∨ k = 5 - Real.sqrt 2 := by
  sorry

end parallelepiped_volume_l2951_295182


namespace cone_volume_l2951_295146

/-- The volume of a cone with base radius 1 and slant height 2 is (√3/3)π -/
theorem cone_volume (r h l : ℝ) : 
  r = 1 → l = 2 → h^2 + r^2 = l^2 → (1/3 * π * r^2 * h) = (Real.sqrt 3 / 3) * π :=
by sorry

end cone_volume_l2951_295146


namespace orchard_area_distribution_l2951_295123

/-- Represents an orange orchard with flat and hilly land. -/
structure Orchard where
  total_area : ℝ
  flat_area : ℝ
  hilly_area : ℝ
  sampled_flat : ℝ
  sampled_hilly : ℝ

/-- Checks if the orchard satisfies the given conditions. -/
def is_valid_orchard (o : Orchard) : Prop :=
  o.total_area = 120 ∧
  o.flat_area + o.hilly_area = o.total_area ∧
  o.sampled_flat + o.sampled_hilly = 10 ∧
  o.sampled_hilly = 2 * o.sampled_flat + 1

/-- Theorem stating the correct distribution of flat and hilly land in the orchard. -/
theorem orchard_area_distribution (o : Orchard) (h : is_valid_orchard o) :
  o.flat_area = 36 ∧ o.hilly_area = 84 := by
  sorry

end orchard_area_distribution_l2951_295123


namespace angle_PQ_A1BD_l2951_295187

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D

/-- Represents a line in 3D space -/
structure Line where
  direction : Point3D

/-- Calculates the reflection of a point with respect to a plane -/
def reflect_point_plane (p : Point3D) (plane : Plane) : Point3D :=
  sorry

/-- Calculates the reflection of a point with respect to a line -/
def reflect_point_line (p : Point3D) (line : Line) : Point3D :=
  sorry

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (line : Line) (plane : Plane) : ℝ :=
  sorry

theorem angle_PQ_A1BD (cube : Cube) : 
  let C1BD : Plane := sorry
  let B1D : Line := sorry
  let A1BD : Plane := sorry
  let P : Point3D := reflect_point_plane cube.A C1BD
  let Q : Point3D := reflect_point_line cube.A B1D
  let PQ : Line := { direction := sorry }
  Real.sin (angle_line_plane PQ A1BD) = 2 * Real.sqrt 3 / 5 := by
  sorry

end angle_PQ_A1BD_l2951_295187


namespace unit_digit_of_seven_to_fourteen_l2951_295162

theorem unit_digit_of_seven_to_fourteen (n : ℕ) : n = 7^14 → n % 10 = 9 := by
  sorry

end unit_digit_of_seven_to_fourteen_l2951_295162


namespace boxes_sold_l2951_295121

theorem boxes_sold (initial boxes_left : ℕ) (h : initial ≥ boxes_left) :
  initial - boxes_left = initial - boxes_left :=
by sorry

end boxes_sold_l2951_295121


namespace circle_equation_l2951_295116

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_tangent_to_line (C : Circle) (a b c : ℝ) : Prop :=
  let (x, y) := C.center
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2) = C.radius

def is_tangent_to_x_axis (C : Circle) : Prop :=
  C.center.2 = C.radius

-- State the theorem
theorem circle_equation (C : Circle) :
  C.radius = 1 →
  is_in_first_quadrant C.center →
  is_tangent_to_line C 4 (-3) 0 →
  is_tangent_to_x_axis C →
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 1 ↔ (x, y) ∈ {p | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2} :=
by sorry

end circle_equation_l2951_295116


namespace mode_invariant_under_single_removal_l2951_295110

def dataset : List ℕ := [5, 6, 8, 8, 8, 1, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_invariant_under_single_removal (d : ℕ) :
  d ∈ dataset → mode (dataset.erase d) = mode dataset := by
  sorry

end mode_invariant_under_single_removal_l2951_295110


namespace xy_inequality_l2951_295178

theorem xy_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = x + y + 3) :
  (x + y ≥ 6) ∧ (x * y ≥ 9) := by
  sorry

end xy_inequality_l2951_295178


namespace choose_three_from_fifteen_l2951_295175

theorem choose_three_from_fifteen : Nat.choose 15 3 = 455 := by
  sorry

end choose_three_from_fifteen_l2951_295175


namespace max_product_arithmetic_mean_l2951_295104

theorem max_product_arithmetic_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : 2 = (2 * a + b) / 2) : 
  a * b ≤ 2 ∧ (a * b = 2 ↔ b = 2 ∧ a = 1) := by
  sorry

end max_product_arithmetic_mean_l2951_295104


namespace sum_squares_inequality_l2951_295124

theorem sum_squares_inequality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 := by
sorry

end sum_squares_inequality_l2951_295124


namespace probability_two_green_bottles_l2951_295127

/-- The probability of selecting 2 green bottles out of 4 green bottles and 38 black bottles -/
theorem probability_two_green_bottles (green_bottles : ℕ) (black_bottles : ℕ) : 
  green_bottles = 4 → black_bottles = 38 → 
  (Nat.choose green_bottles 2 : ℚ) / (Nat.choose (green_bottles + black_bottles) 2) = 1 / 143.5 :=
by sorry

end probability_two_green_bottles_l2951_295127


namespace debby_water_bottles_l2951_295171

def water_bottle_problem (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) : Prop :=
  let days : ℕ := (initial_bottles - remaining_bottles) / bottles_per_day
  days = 1

theorem debby_water_bottles :
  water_bottle_problem 301 144 157 :=
sorry

end debby_water_bottles_l2951_295171


namespace power_of_product_equals_product_of_powers_l2951_295112

theorem power_of_product_equals_product_of_powers (b : ℝ) : (2 * b^2)^3 = 8 * b^6 := by
  sorry

end power_of_product_equals_product_of_powers_l2951_295112


namespace find_m_value_l2951_295128

/-- Given functions f and g, and a condition on their values at x = 3, 
    prove that the parameter m in g equals -11/3 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) 
    (hf : ∀ x, f x = 3 * x^2 + 2 / x - 1)
    (hg : ∀ x, g x = 2 * x^2 - m)
    (h_diff : f 3 - g 3 = 5) : 
  m = -11/3 := by sorry

end find_m_value_l2951_295128


namespace inequality_iff_quadratic_nonpositive_l2951_295199

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), x < y → y < z →
    f y - ((z - y) / (z - x) * f x + (y - x) / (z - x) * f z) ≤
    f ((x + z) / 2) - (f x + f z) / 2

/-- The set of quadratic functions with non-positive leading coefficient -/
def QuadraticNonPositive (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≤ 0 ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

/-- Main theorem: A function satisfies the inequality if and only if it's quadratic with non-positive leading coefficient -/
theorem inequality_iff_quadratic_nonpositive (f : ℝ → ℝ) :
  SatisfiesInequality f ↔ QuadraticNonPositive f :=
sorry

end inequality_iff_quadratic_nonpositive_l2951_295199


namespace raft_journey_time_l2951_295111

/-- Represents the journey of a steamboat between two cities -/
structure SteamboatJourney where
  speed : ℝ  -- Speed of the steamboat
  current : ℝ  -- Speed of the river current
  time_ab : ℝ  -- Time from A to B
  time_ba : ℝ  -- Time from B to A

/-- Calculates the time taken by a raft to travel from A to B -/
def raft_time (journey : SteamboatJourney) : ℝ :=
  60  -- The actual calculation is omitted and replaced with the result

/-- Theorem stating the raft journey time given steamboat journey details -/
theorem raft_journey_time (journey : SteamboatJourney) 
  (h1 : journey.time_ab = 10)
  (h2 : journey.time_ba = 15)
  (h3 : journey.speed > 0)
  (h4 : journey.current > 0) :
  raft_time journey = 60 := by
  sorry


end raft_journey_time_l2951_295111


namespace sum_of_symmetric_roots_l2951_295136

theorem sum_of_symmetric_roots (f : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, f (-x) = f x) 
  (h_roots : ∃! (s : Finset ℝ), s.card = 2009 ∧ ∀ x ∈ s, f x = 0) : 
  ∃ (s : Finset ℝ), s.card = 2009 ∧ (∀ x ∈ s, f x = 0) ∧ (s.sum id = 0) := by
sorry

end sum_of_symmetric_roots_l2951_295136


namespace simplify_fraction_l2951_295197

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end simplify_fraction_l2951_295197


namespace quadratic_roots_condition_l2951_295163

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem stating the conditions for f(p) = f(q) = 0 -/
theorem quadratic_roots_condition (p q : ℝ) :
  (f p q p = 0 ∧ f p q q = 0) ↔ 
  ((p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2)) :=
sorry

end quadratic_roots_condition_l2951_295163


namespace distribute_three_letters_four_mailboxes_l2951_295160

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: Distributing 3 letters into 4 mailboxes results in 4^3 ways -/
theorem distribute_three_letters_four_mailboxes : 
  distribute 3 4 = 4^3 := by sorry

end distribute_three_letters_four_mailboxes_l2951_295160


namespace correct_subtraction_l2951_295154

theorem correct_subtraction (x : ℤ) (h : x - 21 = 52) : x - 40 = 33 := by
  sorry

end correct_subtraction_l2951_295154


namespace meeting_time_correct_l2951_295191

/-- Represents the time in hours after 7:00 AM -/
def time_after_seven (hours minutes : ℕ) : ℚ :=
  hours + minutes / 60

/-- The problem setup -/
structure TravelProblem where
  julia_speed : ℚ
  mark_speed : ℚ
  total_distance : ℚ
  mark_departure_time : ℚ

/-- The solution to the problem -/
def meeting_time (p : TravelProblem) : ℚ :=
  (p.total_distance + p.mark_speed * p.mark_departure_time) / (p.julia_speed + p.mark_speed)

/-- The theorem statement -/
theorem meeting_time_correct (p : TravelProblem) : 
  p.julia_speed = 15 ∧ 
  p.mark_speed = 20 ∧ 
  p.total_distance = 85 ∧ 
  p.mark_departure_time = 0.75 →
  meeting_time p = time_after_seven 2 51 := by
  sorry

#eval time_after_seven 2 51

end meeting_time_correct_l2951_295191


namespace six_people_circular_table_l2951_295148

/-- The number of distinct circular permutations of n elements -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- Two seating arrangements are considered the same if one is a rotation of the other -/
axiom rotation_equivalence : ∀ n : ℕ, n > 0 → circularPermutations n = (n - 1).factorial

theorem six_people_circular_table : circularPermutations 6 = 120 := by
  sorry

end six_people_circular_table_l2951_295148


namespace factor_expression_l2951_295103

theorem factor_expression (z : ℝ) : 75 * z^23 + 225 * z^46 = 75 * z^23 * (1 + 3 * z^23) := by
  sorry

end factor_expression_l2951_295103


namespace selection_theorem_l2951_295168

/-- The number of ways to select 3 people from 4 male and 3 female students, ensuring both genders are represented. -/
def selection_ways (male_count female_count : ℕ) (total_selected : ℕ) : ℕ :=
  Nat.choose (male_count + female_count) total_selected -
  Nat.choose male_count total_selected -
  Nat.choose female_count total_selected

/-- Theorem stating that the number of ways to select 3 people from 4 male and 3 female students, ensuring both genders are represented, is 30. -/
theorem selection_theorem :
  selection_ways 4 3 3 = 30 := by
  sorry

end selection_theorem_l2951_295168


namespace problem_statement_l2951_295141

theorem problem_statement (a b : ℝ) (h : a - 3*b = 3) : 
  (a + 2*b) - (2*a - b) = -3 := by
sorry

end problem_statement_l2951_295141


namespace max_value_of_trig_function_l2951_295143

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x => (1/5) * Real.sin (x + π/3) + Real.cos (x - π/6)
  ∃ M : ℝ, M = 6/5 ∧ ∀ x : ℝ, f x ≤ M ∧ ∃ x₀ : ℝ, f x₀ = M :=
by
  sorry

end max_value_of_trig_function_l2951_295143


namespace triangle_tangent_sum_inequality_l2951_295144

/-- For any acute-angled triangle ABC with perimeter p and inradius r,
    the sum of the tangents of its angles is greater than or equal to
    the ratio of its perimeter to twice its inradius. -/
theorem triangle_tangent_sum_inequality (A B C : ℝ) (p r : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute angles
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  p > 0 ∧ r > 0 →          -- Positive perimeter and inradius
  Real.tan A + Real.tan B + Real.tan C ≥ p / (2 * r) := by
sorry

end triangle_tangent_sum_inequality_l2951_295144


namespace largest_three_digit_divisible_by_4_and_5_l2951_295190

theorem largest_three_digit_divisible_by_4_and_5 : ∀ n : ℕ,
  n ≤ 999 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 → n ≤ 980 := by
  sorry

#check largest_three_digit_divisible_by_4_and_5

end largest_three_digit_divisible_by_4_and_5_l2951_295190


namespace weaving_time_approx_l2951_295180

/-- The time taken to weave a certain amount of cloth, given the weaving rate and total time -/
def weaving_time (rate : Real) (total_time : Real) : Real :=
  total_time

theorem weaving_time_approx :
  let rate := 1.14  -- meters per second
  let total_time := 45.6140350877193  -- seconds
  ∃ ε > 0, |weaving_time rate total_time - 45.614| < ε :=
sorry

end weaving_time_approx_l2951_295180


namespace triangle_lines_correct_l2951_295137

/-- Triangle with vertices A(-5,0), B(3,-3), and C(0,2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle : Triangle := { A := (-5, 0), B := (3, -3), C := (0, 2) }

/-- The equation of the line containing side BC -/
def line_BC : LineEquation := { a := 5, b := 3, c := -6 }

/-- The equation of the line containing the altitude from A to side BC -/
def altitude_A : LineEquation := { a := 5, b := 2, c := 25 }

theorem triangle_lines_correct (t : Triangle) (bc : LineEquation) (alt : LineEquation) :
  t = triangle → bc = line_BC → alt = altitude_A := by sorry

end triangle_lines_correct_l2951_295137


namespace wickets_before_match_l2951_295101

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : BowlingStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: The cricketer had taken 85 wickets before the match -/
theorem wickets_before_match (stats : BowlingStats) : 
  stats.average = 12.4 →
  newAverage stats 5 26 = 12 →
  stats.wickets = 85 := by
  sorry

#check wickets_before_match

end wickets_before_match_l2951_295101


namespace problem_1_problem_2_l2951_295198

def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

theorem problem_1 : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

theorem problem_2 : ∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end problem_1_problem_2_l2951_295198


namespace transform_is_shift_l2951_295109

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define the transformation
def transform (g : RealFunction) : RealFunction :=
  λ x => g (x - 2) + 3

-- State the theorem
theorem transform_is_shift (g : RealFunction) :
  ∀ x y, transform g x = y ↔ g (x - 2) = y - 3 :=
sorry

end transform_is_shift_l2951_295109


namespace stamps_ratio_l2951_295176

def stamps_problem (bert ernie peggy : ℕ) : Prop :=
  bert = 4 * ernie ∧
  ∃ k : ℕ, ernie = k * peggy ∧
  peggy = 75 ∧
  bert = peggy + 825

theorem stamps_ratio (bert ernie peggy : ℕ) 
  (h : stamps_problem bert ernie peggy) : ernie / peggy = 3 := by
  sorry

end stamps_ratio_l2951_295176


namespace faculty_reduction_l2951_295158

theorem faculty_reduction (initial_faculty : ℕ) : 
  (initial_faculty : ℝ) * 0.85 * 0.75 = 180 → 
  initial_faculty = 282 :=
by
  sorry

end faculty_reduction_l2951_295158


namespace circle_tangent_to_line_l2951_295195

theorem circle_tangent_to_line (m : ℝ) (hm : m ≥ 0) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = m}
  let line := {(x, y) : ℝ × ℝ | x + y = Real.sqrt (2 * m)}
  ∃ (p : ℝ × ℝ), p ∈ circle ∧ p ∈ line ∧
    ∀ (q : ℝ × ℝ), q ∈ circle → q ∈ line → q = p :=
by
  sorry

end circle_tangent_to_line_l2951_295195


namespace apple_tree_problem_l2951_295149

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 11

/-- The number of apples picked from the tree -/
def apples_picked : ℕ := 7

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 2

/-- The number of apples currently on the tree -/
def current_apples : ℕ := 6

theorem apple_tree_problem :
  initial_apples - apples_picked + new_apples = current_apples :=
by sorry

end apple_tree_problem_l2951_295149


namespace parabola_distance_property_l2951_295108

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define the condition for Q
def Q_condition (Q P : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (px, py) := P
  (qx - 2, qy) = -4 * (px - 2, py)

-- Theorem statement
theorem parabola_distance_property (Q P : ℝ × ℝ) :
  directrix P.1 →
  parabola Q.1 Q.2 →
  Q_condition Q P →
  Real.sqrt ((Q.1 - 2)^2 + Q.2^2) = 20 := by sorry

end parabola_distance_property_l2951_295108


namespace num_regions_correct_l2951_295118

/-- The number of regions formed by n lines in a plane, where no two lines are parallel and no three lines are concurrent. -/
def num_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- Theorem stating that num_regions correctly calculates the number of regions. -/
theorem num_regions_correct (n : ℕ) :
  num_regions n = n * (n + 1) / 2 + 1 :=
by sorry

end num_regions_correct_l2951_295118


namespace solution_approximation_l2951_295173

/-- The solution to the equation (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5 is approximately 0.01689 -/
theorem solution_approximation : ∃ x : ℝ, 
  (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5 ∧ 
  abs (x - 0.01689) < 0.00001 := by
  sorry

end solution_approximation_l2951_295173


namespace geometric_series_sum_6_terms_l2951_295153

def geometricSeriesSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_6_terms :
  let a : ℚ := 2
  let r : ℚ := 1/3
  let n : ℕ := 6
  geometricSeriesSum a r n = 2184/729 := by
sorry

end geometric_series_sum_6_terms_l2951_295153


namespace ones_digit_of_9_pow_47_l2951_295189

-- Define a function to get the ones digit of an integer
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem stating that the ones digit of 9^47 is 9
theorem ones_digit_of_9_pow_47 : ones_digit (9^47) = 9 := by
  sorry

end ones_digit_of_9_pow_47_l2951_295189


namespace investment_growth_l2951_295106

/-- The monthly interest rate for an investment that grows from $300 to $363 in 2 months -/
def monthly_interest_rate : ℝ :=
  0.1

theorem investment_growth (initial_investment : ℝ) (final_amount : ℝ) (months : ℕ) :
  initial_investment = 300 →
  final_amount = 363 →
  months = 2 →
  final_amount = initial_investment * (1 + monthly_interest_rate) ^ months :=
by sorry

end investment_growth_l2951_295106


namespace g_form_l2951_295184

-- Define polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_g_prod : ∀ x, f (g x) = f x * g x
axiom g_3_eq_50 : g 3 = 50

-- Define the theorem
theorem g_form : g = fun x ↦ x^2 + 20*x - 20 :=
sorry

end g_form_l2951_295184


namespace no_real_roots_l2951_295105

theorem no_real_roots : ∀ x : ℝ, x^2 + 3*x + 5 ≠ 0 := by
  sorry

end no_real_roots_l2951_295105


namespace helicopter_rental_cost_l2951_295165

/-- Calculates the total cost of helicopter rental given the specified conditions -/
theorem helicopter_rental_cost : 
  let hours_per_day : ℕ := 2
  let num_days : ℕ := 3
  let rate_day1 : ℚ := 85
  let rate_day2 : ℚ := 75
  let rate_day3 : ℚ := 65
  let discount_rate : ℚ := 0.05
  let cost_before_discount : ℚ := hours_per_day * (rate_day1 + rate_day2 + rate_day3)
  let discount : ℚ := discount_rate * cost_before_discount
  let total_cost : ℚ := cost_before_discount - discount
  total_cost = 427.5 := by sorry

end helicopter_rental_cost_l2951_295165


namespace two_thirds_of_number_is_36_l2951_295150

theorem two_thirds_of_number_is_36 (x : ℚ) : (2 : ℚ) / 3 * x = 36 → x = 54 := by
  sorry

end two_thirds_of_number_is_36_l2951_295150


namespace parallel_vectors_x_value_l2951_295145

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v1 v2 : Fin 2 → ℝ) : Prop :=
  v1 0 * v2 1 = v1 1 * v2 0

/-- The problem statement -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  are_parallel (λ i => if i = 0 then 1 else 2) (λ i => if i = 0 then 2*x else -3) →
  x = -3/4 := by
  sorry

end parallel_vectors_x_value_l2951_295145


namespace opposite_of_abs_neg_pi_l2951_295193

theorem opposite_of_abs_neg_pi : -(|-π|) = -π := by
  sorry

end opposite_of_abs_neg_pi_l2951_295193


namespace pairball_playing_time_l2951_295174

theorem pairball_playing_time (num_children : ℕ) (total_time : ℕ) : 
  num_children = 6 →
  total_time = 90 →
  ∃ (time_per_child : ℕ), 
    time_per_child * num_children = 2 * total_time ∧
    time_per_child = 30 :=
by sorry

end pairball_playing_time_l2951_295174


namespace trig_expression_simplification_l2951_295152

theorem trig_expression_simplification (x : Real) :
  x = π / 4 →
  (1 + Real.sin (x + π / 4) - Real.cos (x + π / 4)) / 
  (1 + Real.sin (x + π / 4) + Real.cos (x + π / 4)) = 1 := by
  sorry

end trig_expression_simplification_l2951_295152


namespace derivative_of_f_l2951_295113

noncomputable def f (x : ℝ) := Real.cos (x^2 + x)

theorem derivative_of_f (x : ℝ) :
  deriv f x = -(2 * x + 1) * Real.sin (x^2 + x) := by sorry

end derivative_of_f_l2951_295113


namespace product_of_sum_and_sum_of_reciprocals_ge_four_l2951_295140

theorem product_of_sum_and_sum_of_reciprocals_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1/a + 1/b) ≥ 4 := by
  sorry

end product_of_sum_and_sum_of_reciprocals_ge_four_l2951_295140


namespace minimum_guests_proof_l2951_295161

-- Define the total food consumed
def total_food : ℝ := 319

-- Define the maximum individual consumption limits
def max_meat : ℝ := 1.5
def max_vegetables : ℝ := 0.3
def max_dessert : ℝ := 0.2

-- Define the consumption ratio
def meat_ratio : ℝ := 3
def vegetables_ratio : ℝ := 1
def dessert_ratio : ℝ := 1

-- Define the minimum number of guests
def min_guests : ℕ := 160

-- Theorem statement
theorem minimum_guests_proof :
  ∃ (guests : ℕ), guests ≥ min_guests ∧
  (guests : ℝ) * (max_meat + max_vegetables + max_dessert) ≥ total_food ∧
  ∀ (g : ℕ), g < guests →
    (g : ℝ) * (max_meat + max_vegetables + max_dessert) < total_food :=
sorry

end minimum_guests_proof_l2951_295161


namespace rain_stop_time_l2951_295131

def rain_duration (start_time : ℕ) (day1_duration : ℕ) : ℕ → ℕ
  | 1 => day1_duration
  | 2 => day1_duration + 2
  | 3 => 2 * (day1_duration + 2)
  | _ => 0

theorem rain_stop_time (start_time : ℕ) (day1_duration : ℕ) :
  start_time = 7 ∧ 
  (rain_duration start_time day1_duration 1 + 
   rain_duration start_time day1_duration 2 + 
   rain_duration start_time day1_duration 3 = 46) →
  start_time + day1_duration = 17 := by
  sorry

end rain_stop_time_l2951_295131


namespace ratio_of_x_to_y_l2951_295134

theorem ratio_of_x_to_y (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1/5 * x) / (1/6 * y) = 18/25 := by
  sorry

end ratio_of_x_to_y_l2951_295134


namespace binomial_coefficient_equality_l2951_295164

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 := by sorry

end binomial_coefficient_equality_l2951_295164


namespace ship_lighthouse_distance_l2951_295122

/-- The distance between a ship and a lighthouse given specific sailing conditions -/
theorem ship_lighthouse_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (angle_A : ℝ) 
  (angle_B : ℝ) : 
  speed = 15 → 
  time = 4 → 
  angle_A = 60 * π / 180 → 
  angle_B = 15 * π / 180 → 
  ∃ (d : ℝ), d = 800 * Real.sqrt 3 - 240 ∧ 
    d = Real.sqrt ((speed * time * (Real.cos angle_B - Real.cos angle_A) / (Real.sin angle_A - Real.sin angle_B))^2 + 
                   (speed * time * (Real.sin angle_B * Real.cos angle_A - Real.sin angle_A * Real.cos angle_B) / (Real.sin angle_A - Real.sin angle_B))^2) := by
  sorry

end ship_lighthouse_distance_l2951_295122
