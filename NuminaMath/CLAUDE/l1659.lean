import Mathlib

namespace NUMINAMATH_CALUDE_total_removed_volume_is_one_forty_eighth_l1659_165921

/-- A unit cube with corners cut off such that each face forms a regular hexagon -/
structure ModifiedCube where
  /-- The original cube is a unit cube -/
  is_unit_cube : Bool
  /-- Each face of the modified cube forms a regular hexagon -/
  faces_are_hexagons : Bool

/-- The volume of a single removed triangular pyramid -/
def single_pyramid_volume (cube : ModifiedCube) : ℝ :=
  sorry

/-- The total number of removed triangular pyramids -/
def num_pyramids : ℕ := 8

/-- The total volume of all removed triangular pyramids -/
def total_removed_volume (cube : ModifiedCube) : ℝ :=
  (single_pyramid_volume cube) * (num_pyramids : ℝ)

/-- Theorem: The total volume of removed triangular pyramids is 1/48 -/
theorem total_removed_volume_is_one_forty_eighth (cube : ModifiedCube) :
  cube.is_unit_cube ∧ cube.faces_are_hexagons →
  total_removed_volume cube = 1 / 48 :=
sorry

end NUMINAMATH_CALUDE_total_removed_volume_is_one_forty_eighth_l1659_165921


namespace NUMINAMATH_CALUDE_girls_fraction_in_class_l1659_165964

theorem girls_fraction_in_class (T G B : ℚ) (h1 : T > 0) (h2 : G > 0) (h3 : B > 0)
  (h4 : T = G + B) (h5 : B / G = 5 / 3) :
  ∃ X : ℚ, X * G = (1 / 4) * T ∧ X = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_girls_fraction_in_class_l1659_165964


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1659_165985

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 1 → 
    b = 3 → 
    c^2 = a^2 + b^2 → 
    c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1659_165985


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1659_165988

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - x₁ - 2 = 0 ∧ x₂^2 - x₂ - 2 = 0 ∧ x₁ = 2 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, 2*y₁^2 + 2*y₁ - 1 = 0 ∧ 2*y₂^2 + 2*y₂ - 1 = 0 ∧ 
    y₁ = (-1 + Real.sqrt 3) / 2 ∧ y₂ = (-1 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1659_165988


namespace NUMINAMATH_CALUDE_stone_exit_and_return_velocity_range_l1659_165934

/-- 
Theorem: Stone Exit and Return Velocity Range

For a stone thrown upwards in a well with the following properties:
- Well depth: h = 10 meters
- Cover cycle: opens for 1 second, closes for 1 second
- Stone thrown 0.5 seconds before cover opens
- Acceleration due to gravity: g = 10 m/s²

The initial velocities V for which the stone will exit the well and fall back onto the cover
are in the range (85/6, 33/2) ∪ (285/14, 45/2).
-/
theorem stone_exit_and_return_velocity_range (h g τ : ℝ) (V : ℝ) : 
  h = 10 → 
  g = 10 → 
  τ = 1 → 
  (V ∈ Set.Ioo (85/6) (33/2) ∪ Set.Ioo (285/14) (45/2)) ↔ 
  (∃ t : ℝ, 
    t > 0 ∧ 
    V * t - (1/2) * g * t^2 ≥ h ∧
    ∃ t' : ℝ, t' > t ∧ V * t' - (1/2) * g * t'^2 = 0 ∧
    (∃ n : ℕ, t' = (2*n + 3/2) * τ ∨ t' = (2*n + 7/2) * τ)) :=
by sorry

end NUMINAMATH_CALUDE_stone_exit_and_return_velocity_range_l1659_165934


namespace NUMINAMATH_CALUDE_expansion_term_count_l1659_165984

/-- The number of terms in a polynomial -/
def num_terms (p : Polynomial ℚ) : ℕ := sorry

/-- The expansion of the product of two polynomials -/
def expand_product (p q : Polynomial ℚ) : Polynomial ℚ := sorry

theorem expansion_term_count :
  let p := X + Y + Z
  let q := U + V + W + X
  num_terms (expand_product p q) = 12 := by sorry

end NUMINAMATH_CALUDE_expansion_term_count_l1659_165984


namespace NUMINAMATH_CALUDE_y_properties_l1659_165902

/-- A function y(x) composed of two directly proportional components -/
def y (x : ℝ) (k₁ k₂ : ℝ) : ℝ := k₁ * (x - 3) + k₂ * (x^2 + 1)

/-- Theorem stating the properties of the function y(x) -/
theorem y_properties :
  ∃ (k₁ k₂ : ℝ),
    (y 0 k₁ k₂ = -2) ∧
    (y 1 k₁ k₂ = 4) ∧
    (∀ x, y x k₁ k₂ = 4*x^2 + 2*x - 2) ∧
    (y (-1) k₁ k₂ = 0) ∧
    (y (1/2) k₁ k₂ = 0) := by
  sorry

#check y_properties

end NUMINAMATH_CALUDE_y_properties_l1659_165902


namespace NUMINAMATH_CALUDE_team_a_min_wins_l1659_165917

theorem team_a_min_wins (total_games : ℕ) (lost_games : ℕ) (min_points : ℕ) 
  (win_points draw_points lose_points : ℕ) :
  total_games = 5 →
  lost_games = 1 →
  min_points = 7 →
  win_points = 3 →
  draw_points = 1 →
  lose_points = 0 →
  ∃ (won_games : ℕ),
    won_games ≥ 2 ∧
    won_games + lost_games ≤ total_games ∧
    won_games * win_points + (total_games - won_games - lost_games) * draw_points > min_points :=
by sorry

end NUMINAMATH_CALUDE_team_a_min_wins_l1659_165917


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1659_165957

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, (m - 3) * x^(|m| + 2) + 2 * x - 7 = 0 ↔ ∃ a b c, a ≠ 0 ∧ a * x^2 + b * x + c = 0) ↔ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1659_165957


namespace NUMINAMATH_CALUDE_t_greater_than_a_squared_l1659_165994

/-- An equilateral triangle with a point on one of its sides -/
structure EquilateralTriangleWithPoint where
  a : ℝ  -- Side length of the equilateral triangle
  x : ℝ  -- Distance from A to P on side AB
  h1 : 0 < a  -- Side length is positive
  h2 : 0 ≤ x ∧ x ≤ a  -- P is on side AB

/-- The expression t = AP^2 + PB^2 + CP^2 -/
def t (triangle : EquilateralTriangleWithPoint) : ℝ :=
  let a := triangle.a
  let x := triangle.x
  x^2 + (a - x)^2 + (a^2 - a*x + x^2)

/-- Theorem: t is always greater than a^2 -/
theorem t_greater_than_a_squared (triangle : EquilateralTriangleWithPoint) :
  t triangle > triangle.a^2 := by
  sorry

end NUMINAMATH_CALUDE_t_greater_than_a_squared_l1659_165994


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1659_165996

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_x : x > 1) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x - 1) / (2 * x))) : 
  Real.tan θ = -x * Real.sqrt (1 - 1 / x^2) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1659_165996


namespace NUMINAMATH_CALUDE_horner_rule_f_3_l1659_165935

/-- Horner's Rule evaluation of a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 1]

theorem horner_rule_f_3 :
  f 3 = horner_eval f_coeffs 3 ∧ horner_eval f_coeffs 3 = 1642 :=
sorry

end NUMINAMATH_CALUDE_horner_rule_f_3_l1659_165935


namespace NUMINAMATH_CALUDE_quadratic_roots_shift_l1659_165910

theorem quadratic_roots_shift (a b c : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (y : ℝ) := a * y^2 + (b - 2*a) * y + (a - b + c)
  ∀ (x y : ℝ), f x = 0 ∧ g y = 0 → y = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_shift_l1659_165910


namespace NUMINAMATH_CALUDE_problem_book_solution_l1659_165922

/-- The number of problems solved by Taeyeon and Yura -/
def total_problems_solved (taeyeon_per_day : ℕ) (taeyeon_days : ℕ) (yura_per_day : ℕ) (yura_days : ℕ) : ℕ :=
  taeyeon_per_day * taeyeon_days + yura_per_day * yura_days

/-- Theorem stating that Taeyeon and Yura solved 262 problems in total -/
theorem problem_book_solution :
  total_problems_solved 16 7 25 6 = 262 := by
  sorry

end NUMINAMATH_CALUDE_problem_book_solution_l1659_165922


namespace NUMINAMATH_CALUDE_garage_cleanup_l1659_165981

theorem garage_cleanup (total_trips : ℕ) (jean_extra_trips : ℕ) (total_capacity : ℝ) (actual_weight : ℝ) 
  (h1 : total_trips = 40)
  (h2 : jean_extra_trips = 6)
  (h3 : total_capacity = 8000)
  (h4 : actual_weight = 7850) : 
  let bill_trips := (total_trips - jean_extra_trips) / 2
  let jean_trips := bill_trips + jean_extra_trips
  let avg_weight := actual_weight / total_trips
  jean_trips = 23 ∧ avg_weight = 196.25 := by
  sorry

end NUMINAMATH_CALUDE_garage_cleanup_l1659_165981


namespace NUMINAMATH_CALUDE_advance_tickets_sold_l1659_165900

/-- Proves that the number of advance tickets sold is 20 given the ticket prices, total tickets sold, and total receipts. -/
theorem advance_tickets_sold (advance_price same_day_price total_tickets total_receipts : ℕ) 
  (h1 : advance_price = 20)
  (h2 : same_day_price = 30)
  (h3 : total_tickets = 60)
  (h4 : total_receipts = 1600) : 
  ∃ (advance_tickets : ℕ), 
    advance_tickets * advance_price + (total_tickets - advance_tickets) * same_day_price = total_receipts ∧ 
    advance_tickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_advance_tickets_sold_l1659_165900


namespace NUMINAMATH_CALUDE_activity_popularity_order_l1659_165918

-- Define the activities
inductive Activity
  | dodgeball
  | natureWalk
  | painting

-- Define the popularity fraction for each activity
def popularity (a : Activity) : Rat :=
  match a with
  | Activity.dodgeball => 13/40
  | Activity.natureWalk => 8/25
  | Activity.painting => 9/20

-- Define a function to compare two activities based on their popularity
def morePopular (a b : Activity) : Prop :=
  popularity a > popularity b

-- Theorem stating the correct order of activities
theorem activity_popularity_order :
  morePopular Activity.painting Activity.dodgeball ∧
  morePopular Activity.dodgeball Activity.natureWalk :=
by
  sorry

#check activity_popularity_order

end NUMINAMATH_CALUDE_activity_popularity_order_l1659_165918


namespace NUMINAMATH_CALUDE_apples_in_basket_after_removal_l1659_165977

/-- Given a total number of apples and baskets, and a number of apples removed from each basket,
    calculate the number of apples remaining in each basket. -/
def applesPerBasket (totalApples : ℕ) (numBaskets : ℕ) (applesRemoved : ℕ) : ℕ :=
  (totalApples / numBaskets) - applesRemoved

/-- Theorem stating that for the given problem, each basket contains 9 apples after removal. -/
theorem apples_in_basket_after_removal :
  applesPerBasket 128 8 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_after_removal_l1659_165977


namespace NUMINAMATH_CALUDE_cylinder_volume_l1659_165926

/-- The volume of a cylinder given water displacement measurements -/
theorem cylinder_volume
  (initial_water_level : ℝ)
  (final_water_level : ℝ)
  (cylinder_min_marking : ℝ)
  (cylinder_max_marking : ℝ)
  (h1 : initial_water_level = 30)
  (h2 : final_water_level = 35)
  (h3 : cylinder_min_marking = 15)
  (h4 : cylinder_max_marking = 45) :
  let water_displaced := final_water_level - initial_water_level
  let cylinder_marking_range := cylinder_max_marking - cylinder_min_marking
  let submerged_proportion := (final_water_level - cylinder_min_marking) / cylinder_marking_range
  cylinder_marking_range / (final_water_level - cylinder_min_marking) * water_displaced = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1659_165926


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1659_165978

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : s > 0)
  (h4 : s + 2*x = 3*s) -- The outer boundary is 3 times the inner square side
  (h5 : 2*y = s) -- The shorter side spans half the inner square side
  : x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1659_165978


namespace NUMINAMATH_CALUDE_solid_max_volume_l1659_165986

/-- The side length of each cube in centimeters -/
def cube_side_length : ℝ := 3

/-- The number of cubes in the base layer -/
def base_layer_cubes : ℕ := 4 * 4

/-- The number of cubes in the second layer -/
def second_layer_cubes : ℕ := 2 * 2

/-- The total number of cubes in the solid -/
def total_cubes : ℕ := base_layer_cubes + second_layer_cubes

/-- The volume of a single cube in cubic centimeters -/
def single_cube_volume : ℝ := cube_side_length ^ 3

/-- The maximum volume of the solid in cubic centimeters -/
def max_volume : ℝ := (total_cubes : ℝ) * single_cube_volume

theorem solid_max_volume : max_volume = 540 := by sorry

end NUMINAMATH_CALUDE_solid_max_volume_l1659_165986


namespace NUMINAMATH_CALUDE_dihedral_angle_at_apex_is_45_degrees_l1659_165938

/-- A regular square pyramid with coinciding centers of inscribed and circumscribed spheres -/
structure RegularSquarePyramid where
  /-- The centers of the inscribed and circumscribed spheres coincide -/
  coinciding_centers : Bool

/-- The dihedral angle at the apex of the pyramid -/
def dihedral_angle_at_apex (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem: The dihedral angle at the apex of a regular square pyramid 
    with coinciding centers of inscribed and circumscribed spheres is 45° -/
theorem dihedral_angle_at_apex_is_45_degrees (p : RegularSquarePyramid) 
    (h : p.coinciding_centers = true) : 
    dihedral_angle_at_apex p = 45 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_at_apex_is_45_degrees_l1659_165938


namespace NUMINAMATH_CALUDE_volume_of_sphere_containing_pyramid_l1659_165965

/-- Regular triangular pyramid with base on sphere -/
structure RegularTriangularPyramid where
  /-- Base edge length -/
  baseEdge : ℝ
  /-- Volume of the pyramid -/
  volume : ℝ
  /-- Radius of the circumscribed sphere -/
  sphereRadius : ℝ

/-- Theorem: Volume of sphere containing regular triangular pyramid -/
theorem volume_of_sphere_containing_pyramid (p : RegularTriangularPyramid) 
  (h1 : p.baseEdge = 2 * Real.sqrt 3)
  (h2 : p.volume = Real.sqrt 3) :
  (4 / 3) * Real.pi * p.sphereRadius ^ 3 = (20 * Real.sqrt 5 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_containing_pyramid_l1659_165965


namespace NUMINAMATH_CALUDE_squad_size_problem_l1659_165975

theorem squad_size_problem (total : ℕ) (transfer : ℕ) 
  (h1 : total = 146) (h2 : transfer = 11) : 
  (∃ (first second : ℕ), 
    first + second = total ∧ 
    first - transfer = second + transfer ∧
    first = 84 ∧ 
    second = 62) := by
  sorry

end NUMINAMATH_CALUDE_squad_size_problem_l1659_165975


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_threshold_l1659_165952

theorem sum_of_numbers_ge_threshold : 
  let numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℝ := 1.1
  (numbers.filter (λ x => x ≥ threshold)).sum = 3.9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_threshold_l1659_165952


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l1659_165951

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℕ) :
  total_customers = 7 →
  non_tipping_customers = 4 →
  total_tips = 27 →
  (total_tips / (total_customers - non_tipping_customers) : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_calculation_l1659_165951


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_six_l1659_165940

theorem sum_reciprocals_equals_six (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) :
  1 / a + 1 / b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_six_l1659_165940


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1659_165929

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + 
                   Real.sin (35 * π / 180) + Real.sin (45 * π / 180) + 
                   Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + 
                   Real.sin (75 * π / 180) + Real.sin (85 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (15 * π / 180) * Real.cos (25 * π / 180)
  numerator / denominator = 4 * Real.sin (50 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1659_165929


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1659_165992

theorem min_value_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2) ≥ 4 := by
  sorry

theorem min_value_attainable : ∃ x y : ℝ, (x*y - 2)^2 + (x^2 + y^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1659_165992


namespace NUMINAMATH_CALUDE_sqrt_eight_simplification_l1659_165928

theorem sqrt_eight_simplification : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_simplification_l1659_165928


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1659_165989

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive sides
  a + b + c = 40 →  -- perimeter condition
  (1/2) * a * b = 30 →  -- area condition
  a^2 + b^2 = c^2 →  -- right triangle (Pythagorean theorem)
  c = 18.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1659_165989


namespace NUMINAMATH_CALUDE_BA_equals_AB_l1659_165995

-- Define the matrices A and B
variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- Define the given conditions
def condition1 : Prop := A + B = A * B
def condition2 : Prop := A * B = !![12, -6; 9, -3]

-- State the theorem
theorem BA_equals_AB (h1 : condition1 A B) (h2 : condition2 A B) : 
  B * A = !![12, -6; 9, -3] := by sorry

end NUMINAMATH_CALUDE_BA_equals_AB_l1659_165995


namespace NUMINAMATH_CALUDE_locus_is_apollonian_circle_l1659_165983

/-- An Apollonian circle is the locus of points with a constant ratio of distances to two fixed points. -/
def ApollonianCircle (A B : ℝ × ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {M | dist A M / dist M B = k}

/-- The locus of points M satisfying |AM| : |MB| = k ≠ 1, where A and B are fixed points, is an Apollonian circle. -/
theorem locus_is_apollonian_circle (A B : ℝ × ℝ) (k : ℝ) (h : k ≠ 1) :
  {M : ℝ × ℝ | dist A M / dist M B = k} = ApollonianCircle A B k := by
  sorry

end NUMINAMATH_CALUDE_locus_is_apollonian_circle_l1659_165983


namespace NUMINAMATH_CALUDE_lg_sqrt_sum_l1659_165944

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sqrt_sum : lg (Real.sqrt 5) + lg (Real.sqrt 20) = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_sqrt_sum_l1659_165944


namespace NUMINAMATH_CALUDE_taxi_driver_theorem_l1659_165909

def taxi_trips : List Int := [5, -3, 6, -7, 6, -2, -5, 4, 6, -8]

theorem taxi_driver_theorem :
  (taxi_trips.take 7).sum = 0 ∧ taxi_trips.sum = 2 := by sorry

end NUMINAMATH_CALUDE_taxi_driver_theorem_l1659_165909


namespace NUMINAMATH_CALUDE_office_canteen_chairs_l1659_165931

/-- The number of round tables in the office canteen -/
def num_round_tables : ℕ := 2

/-- The number of rectangular tables in the office canteen -/
def num_rectangular_tables : ℕ := 2

/-- The number of chairs per round table -/
def chairs_per_round_table : ℕ := 6

/-- The number of chairs per rectangular table -/
def chairs_per_rectangular_table : ℕ := 7

/-- The total number of chairs in the office canteen -/
def total_chairs : ℕ := num_round_tables * chairs_per_round_table + num_rectangular_tables * chairs_per_rectangular_table

theorem office_canteen_chairs : total_chairs = 26 := by
  sorry

end NUMINAMATH_CALUDE_office_canteen_chairs_l1659_165931


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l1659_165967

/-- The volume of a sphere with two cylindrical holes drilled into it -/
theorem bowling_ball_volume 
  (sphere_diameter : ℝ) 
  (hole1_depth hole1_diameter : ℝ) 
  (hole2_depth hole2_diameter : ℝ) 
  (h1 : sphere_diameter = 24)
  (h2 : hole1_depth = 6)
  (h3 : hole1_diameter = 3)
  (h4 : hole2_depth = 6)
  (h5 : hole2_diameter = 4) : 
  (4 / 3 * π * (sphere_diameter / 2) ^ 3) - 
  (π * (hole1_diameter / 2) ^ 2 * hole1_depth) - 
  (π * (hole2_diameter / 2) ^ 2 * hole2_depth) = 2266.5 * π := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l1659_165967


namespace NUMINAMATH_CALUDE_third_month_sale_l1659_165927

def average_sale : ℕ := 3500
def number_of_months : ℕ := 6
def sale_month1 : ℕ := 3435
def sale_month2 : ℕ := 3920
def sale_month4 : ℕ := 4230
def sale_month5 : ℕ := 3560
def sale_month6 : ℕ := 2000

theorem third_month_sale :
  let total_sales := average_sale * number_of_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 3855 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l1659_165927


namespace NUMINAMATH_CALUDE_recurrence_equals_explicit_l1659_165966

def recurrence_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 5
  | 1 => 10
  | n + 2 => 5 * recurrence_sequence (n + 1) - 6 * recurrence_sequence n + 2 * (n + 2) - 3

def explicit_form (n : ℕ) : ℤ :=
  2^(n + 1) + 3^n + n + 2

theorem recurrence_equals_explicit : ∀ n : ℕ, recurrence_sequence n = explicit_form n :=
  sorry

end NUMINAMATH_CALUDE_recurrence_equals_explicit_l1659_165966


namespace NUMINAMATH_CALUDE_valid_numbers_l1659_165925

def is_valid_number (n : ℕ) : Prop :=
  523000 ≤ n ∧ n ≤ 523999 ∧ n % 7 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 523152 ∨ n = 523656 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1659_165925


namespace NUMINAMATH_CALUDE_range_of_m_l1659_165947

/-- Statement p: For any real number x, the inequality x^2 - 2x + m ≥ 0 always holds -/
def statement_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

/-- Statement q: The equation (x^2)/(m-3) - (y^2)/m = 1 represents a hyperbola with foci on the x-axis -/
def statement_q (m : ℝ) : Prop :=
  m > 3 ∧ ∀ x y : ℝ, x^2 / (m - 3) - y^2 / m = 1

/-- The range of m when "p ∨ q" is true and "p ∧ q" is false -/
theorem range_of_m :
  ∀ m : ℝ, (statement_p m ∨ statement_q m) ∧ ¬(statement_p m ∧ statement_q m) →
  1 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1659_165947


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_168_l1659_165912

/-- The sum of an arithmetic sequence with first term 3, common difference 2, and 12 terms -/
def arithmetic_sum : ℕ := 
  let a₁ : ℕ := 3  -- first term
  let d : ℕ := 2   -- common difference
  let n : ℕ := 12  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating that the sum of the arithmetic sequence is 168 -/
theorem arithmetic_sum_equals_168 : arithmetic_sum = 168 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_168_l1659_165912


namespace NUMINAMATH_CALUDE_translation_motions_l1659_165904

/-- Represents a type of motion. -/
inductive Motion
  | Swing
  | VerticalElevator
  | PlanetMovement
  | ConveyorBelt

/-- Determines if a given motion is a translation. -/
def isTranslation (m : Motion) : Prop :=
  match m with
  | Motion.VerticalElevator => True
  | Motion.ConveyorBelt => True
  | _ => False

/-- The theorem stating which motions are translations. -/
theorem translation_motions :
  (∀ m : Motion, isTranslation m ↔ (m = Motion.VerticalElevator ∨ m = Motion.ConveyorBelt)) :=
by sorry

end NUMINAMATH_CALUDE_translation_motions_l1659_165904


namespace NUMINAMATH_CALUDE_betty_order_cost_l1659_165959

/-- The total cost of Betty's order -/
def total_cost (slipper_price lipstick_price hair_color_price sunglasses_price tshirt_price : ℚ) 
  (slipper_qty lipstick_qty hair_color_qty sunglasses_qty tshirt_qty : ℕ) : ℚ :=
  slipper_price * slipper_qty + 
  lipstick_price * lipstick_qty + 
  hair_color_price * hair_color_qty + 
  sunglasses_price * sunglasses_qty + 
  tshirt_price * tshirt_qty

/-- The theorem stating that Betty's total order cost is $110.25 -/
theorem betty_order_cost : 
  total_cost 2.5 1.25 3 5.75 12.25 6 4 8 3 4 = 110.25 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_cost_l1659_165959


namespace NUMINAMATH_CALUDE_difference_of_squares_l1659_165905

theorem difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1659_165905


namespace NUMINAMATH_CALUDE_value_of_a_l1659_165906

theorem value_of_a (a b c d : ℤ) 
  (eq1 : a = b + 3)
  (eq2 : b = c + 6)
  (eq3 : c = d + 15)
  (eq4 : d = 50) : 
  a = 74 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1659_165906


namespace NUMINAMATH_CALUDE_inequality_of_positive_numbers_l1659_165973

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 * b + a * b^2 ≤ a^3 + b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_numbers_l1659_165973


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1659_165993

theorem cubic_equation_unique_solution :
  ∃! (x : ℤ), x^3 + (x+1)^3 + (x+2)^3 = (x+3)^3 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l1659_165993


namespace NUMINAMATH_CALUDE_missing_score_l1659_165939

theorem missing_score (scores : List ℕ) (mean : ℚ) : 
  scores = [73, 83, 86, 73] ∧ 
  mean = 79.2 ∧ 
  (scores.sum + (missing : ℕ)) / 5 = mean → 
  missing = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_score_l1659_165939


namespace NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_three_l1659_165913

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_ten_x_plus_three_l1659_165913


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1659_165916

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1659_165916


namespace NUMINAMATH_CALUDE_karlson_candies_theorem_l1659_165961

/-- Represents the maximum number of candies Karlson can eat -/
def max_candies (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that the maximum number of candies Karlson can eat with 31 initial ones is 465 -/
theorem karlson_candies_theorem :
  max_candies 31 = 465 := by
  sorry

#eval max_candies 31

end NUMINAMATH_CALUDE_karlson_candies_theorem_l1659_165961


namespace NUMINAMATH_CALUDE_dividend_calculation_l1659_165974

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 9) : 
  divisor * quotient + remainder = 162 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1659_165974


namespace NUMINAMATH_CALUDE_experts_win_probability_l1659_165970

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expert_score : ℕ := 3

/-- The current score of Viewers -/
def viewer_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
theorem experts_win_probability : 
  p^4 + 4 * p^3 * q = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_l1659_165970


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1659_165972

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (a b : Line) (α β : Plane)
  (diff_lines : a ≠ b)
  (diff_planes : α ≠ β)
  (h1 : parallel_planes α β)
  (h2 : contained_in a α) :
  parallel_line_plane a β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1659_165972


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l1659_165982

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

/-- The theorem stating that if a/cos(B) = b/cos(A) in a triangle, 
    then the triangle is either isosceles or right-angled. -/
theorem triangle_isosceles_or_right_angled (t : Triangle) 
  (h : t.a / Real.cos t.B = t.b / Real.cos t.A) : 
  (t.A = t.B) ∨ (t.C = π/2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l1659_165982


namespace NUMINAMATH_CALUDE_song_book_cost_l1659_165942

def trumpet_cost : ℝ := 149.16
def music_tool_cost : ℝ := 9.98
def total_spent : ℝ := 163.28

theorem song_book_cost :
  total_spent - (trumpet_cost + music_tool_cost) = 4.14 := by sorry

end NUMINAMATH_CALUDE_song_book_cost_l1659_165942


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1659_165919

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = (34 * total_votes) / 100 ∧ 
    rival_votes = candidate_votes + 640 ∧
    candidate_votes + rival_votes = total_votes) →
  total_votes = 2000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1659_165919


namespace NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l1659_165960

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cube_side = 216 - 54 * π := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l1659_165960


namespace NUMINAMATH_CALUDE_coffee_conference_theorem_l1659_165933

/-- Represents the number of participants who went for coffee -/
def coffee_goers (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (remaining : ℕ), 
    remaining > 0 ∧ 
    remaining < n ∧ 
    remaining % 2 = 0 ∧ 
    k = n - remaining}

/-- The theorem stating the possible number of coffee goers -/
theorem coffee_conference_theorem :
  coffee_goers 14 = {6, 8, 10, 12} :=
sorry


end NUMINAMATH_CALUDE_coffee_conference_theorem_l1659_165933


namespace NUMINAMATH_CALUDE_world_expo_ticket_sales_l1659_165932

theorem world_expo_ticket_sales :
  let regular_price : ℕ := 200
  let concession_price : ℕ := 120
  let total_tickets : ℕ := 1200
  let total_revenue : ℕ := 216000
  ∃ (regular_tickets concession_tickets : ℕ),
    regular_tickets + concession_tickets = total_tickets ∧
    regular_tickets * regular_price + concession_tickets * concession_price = total_revenue ∧
    regular_tickets = 900 ∧
    concession_tickets = 300 := by
sorry

end NUMINAMATH_CALUDE_world_expo_ticket_sales_l1659_165932


namespace NUMINAMATH_CALUDE_division_problem_l1659_165990

theorem division_problem (k : ℕ) (h : k = 14) : 56 / k = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1659_165990


namespace NUMINAMATH_CALUDE_tangent_line_circle_min_sum_l1659_165963

theorem tangent_line_circle_min_sum (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
               (x - 1)^2 + (y - 1)^2 = 1 ∧
               ∀ a b : ℝ, (x - a)^2 + (y - b)^2 ≤ 1 → 
                          (m + 1) * a + (n + 1) * b - 2 ≠ 0) →
  (∀ p q : ℝ, p > 0 → q > 0 → 
    (∃ x y : ℝ, (p + 1) * x + (q + 1) * y - 2 = 0 ∧ 
                (x - 1)^2 + (y - 1)^2 = 1 ∧
                ∀ a b : ℝ, (x - a)^2 + (y - b)^2 ≤ 1 → 
                           (p + 1) * a + (q + 1) * b - 2 ≠ 0) →
    m + n ≤ p + q) →
  m + n = 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_min_sum_l1659_165963


namespace NUMINAMATH_CALUDE_junk_mail_total_l1659_165997

/-- Calculates the total number of junk mail pieces a mailman should give --/
theorem junk_mail_total (houses_per_block : ℕ) (num_blocks : ℕ) (mail_per_house : ℕ) : 
  houses_per_block = 50 → num_blocks = 3 → mail_per_house = 45 → 
  houses_per_block * num_blocks * mail_per_house = 6750 := by
  sorry

#check junk_mail_total

end NUMINAMATH_CALUDE_junk_mail_total_l1659_165997


namespace NUMINAMATH_CALUDE_birthday_cookies_l1659_165954

theorem birthday_cookies (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) :
  friends = 7 →
  packages = 5 →
  cookies_per_package = 36 →
  (packages * cookies_per_package) / (friends + 1) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_cookies_l1659_165954


namespace NUMINAMATH_CALUDE_researchers_distribution_l1659_165976

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with at least one object in each box. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of researchers -/
def num_researchers : ℕ := 4

/-- The theorem stating that the number of ways to distribute 4 researchers
    to 3 schools, with at least one researcher in each school, is 36. -/
theorem researchers_distribution :
  distribute num_researchers num_schools = 36 := by sorry

end NUMINAMATH_CALUDE_researchers_distribution_l1659_165976


namespace NUMINAMATH_CALUDE_tomatoes_sold_to_maxwell_l1659_165915

/-- Calculates the amount of tomatoes sold to Mrs. Maxwell -/
theorem tomatoes_sold_to_maxwell 
  (total_harvest : ℝ) 
  (sold_to_wilson : ℝ) 
  (not_sold : ℝ) 
  (h1 : total_harvest = 245.5)
  (h2 : sold_to_wilson = 78)
  (h3 : not_sold = 42) :
  total_harvest - sold_to_wilson - not_sold = 125.5 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_sold_to_maxwell_l1659_165915


namespace NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l1659_165948

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 6 * Complex.I) + Complex.abs (z - 5) = 7) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30 / Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l1659_165948


namespace NUMINAMATH_CALUDE_cheyenne_pots_count_l1659_165907

/-- The number of clay pots Cheyenne made -/
def total_pots : ℕ := 80

/-- The fraction of pots that cracked -/
def cracked_fraction : ℚ := 2/5

/-- The revenue from selling the remaining pots -/
def revenue : ℕ := 1920

/-- The price of each clay pot -/
def price_per_pot : ℕ := 40

/-- Theorem stating that the number of clay pots Cheyenne made is 80 -/
theorem cheyenne_pots_count :
  total_pots = 80 ∧
  cracked_fraction = 2/5 ∧
  revenue = 1920 ∧
  price_per_pot = 40 ∧
  (1 - cracked_fraction) * total_pots * price_per_pot = revenue :=
by sorry

end NUMINAMATH_CALUDE_cheyenne_pots_count_l1659_165907


namespace NUMINAMATH_CALUDE_bus_assignment_count_l1659_165955

def num_boys : ℕ := 6
def num_girls : ℕ := 4
def num_buses : ℕ := 5
def attendants_per_bus : ℕ := 2

theorem bus_assignment_count : 
  (Nat.choose num_buses 3) * 
  (Nat.factorial num_boys / (Nat.factorial attendants_per_bus ^ 3)) * 
  (Nat.factorial num_girls / (Nat.factorial attendants_per_bus ^ 2)) * 
  (1 / Nat.factorial 3) * 
  (1 / Nat.factorial 2) * 
  Nat.factorial num_buses = 54000 := by
sorry

end NUMINAMATH_CALUDE_bus_assignment_count_l1659_165955


namespace NUMINAMATH_CALUDE_nectar_water_content_l1659_165901

/-- The percentage of water in honey -/
def honey_water_percentage : ℝ := 25

/-- The weight of flower-nectar needed to produce 1 kg of honey -/
def nectar_weight : ℝ := 1.5

/-- The weight of honey produced from nectar_weight of flower-nectar -/
def honey_weight : ℝ := 1

/-- The percentage of water in flower-nectar -/
def nectar_water_percentage : ℝ := 50

theorem nectar_water_content :
  nectar_water_percentage = 50 :=
sorry

end NUMINAMATH_CALUDE_nectar_water_content_l1659_165901


namespace NUMINAMATH_CALUDE_function_constancy_l1659_165979

def is_constant {α : Type*} (f : α → ℕ) : Prop :=
  ∀ x y, f x = f y

theorem function_constancy (f : ℤ × ℤ → ℕ) 
  (h : ∀ (x y : ℤ), 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_constant f := by
  sorry

end NUMINAMATH_CALUDE_function_constancy_l1659_165979


namespace NUMINAMATH_CALUDE_total_books_total_books_specific_l1659_165936

theorem total_books (stu_books : ℕ) (albert_multiplier : ℕ) : ℕ :=
  let albert_books := albert_multiplier * stu_books
  stu_books + albert_books

theorem total_books_specific : total_books 9 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_books_total_books_specific_l1659_165936


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l1659_165924

/-- Proves that adding 300 mL of solution Y to 100 mL of solution X
    creates a solution that is 25% alcohol by volume. -/
theorem alcohol_mixture_proof 
  (x_conc : Real) -- Concentration of alcohol in solution X
  (y_conc : Real) -- Concentration of alcohol in solution Y
  (x_vol : Real)  -- Volume of solution X
  (y_vol : Real)  -- Volume of solution Y to be added
  (h1 : x_conc = 0.10) -- Solution X is 10% alcohol
  (h2 : y_conc = 0.30) -- Solution Y is 30% alcohol
  (h3 : x_vol = 100)   -- We start with 100 mL of solution X
  (h4 : y_vol = 300)   -- We add 300 mL of solution Y
  : (x_conc * x_vol + y_conc * y_vol) / (x_vol + y_vol) = 0.25 := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l1659_165924


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1659_165911

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ (y : ℝ), y = 4 * x * (3 - 2 * x) ∧ ∀ (z : ℝ), z = 4 * x * (3 - 2 * x) → z ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1659_165911


namespace NUMINAMATH_CALUDE_area_swept_specific_triangle_l1659_165950

/-- Represents a triangle with sides and height -/
structure Triangle where
  bc : ℝ
  ab : ℝ
  ad : ℝ

/-- Calculates the area swept by a triangle moving upward -/
def area_swept (t : Triangle) (speed : ℝ) (time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area swept by the specific triangle -/
theorem area_swept_specific_triangle :
  let t : Triangle := { bc := 6, ab := 5, ad := 4 }
  area_swept t 3 2 = 66 := by sorry

end NUMINAMATH_CALUDE_area_swept_specific_triangle_l1659_165950


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1659_165968

theorem complex_magnitude_product : Complex.abs ((7 - 4*I) * (3 + 11*I)) = Real.sqrt 8450 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1659_165968


namespace NUMINAMATH_CALUDE_sector_area_l1659_165930

/-- The area of a sector with a central angle of 60° in a circle passing through two given points -/
theorem sector_area (P Q : ℝ × ℝ) (h : P = (2, -2) ∧ Q = (8, 6)) : 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (1/6 : ℝ) * π * r^2 = 50*π/3 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1659_165930


namespace NUMINAMATH_CALUDE_shaded_area_is_45_l1659_165920

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  bottomLeft : Point
  side : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  bottomRight : Point
  base : ℝ
  height : ℝ

/-- Calculates the area of the shaded region formed by the intersection of a square and a right triangle -/
def shadedArea (square : Square) (triangle : RightTriangle) : ℝ :=
  sorry

/-- Theorem stating that the shaded area is 45 square units given the specified conditions -/
theorem shaded_area_is_45 :
  ∀ (square : Square) (triangle : RightTriangle),
    square.bottomLeft = Point.mk 12 0 →
    square.side = 12 →
    triangle.bottomRight = Point.mk 12 0 →
    triangle.base = 12 →
    triangle.height = 9 →
    shadedArea square triangle = 45 :=
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_45_l1659_165920


namespace NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l1659_165949

theorem freshmen_in_liberal_arts 
  (total_students : ℝ) 
  (freshmen_ratio : ℝ) 
  (psych_majors_ratio : ℝ) 
  (freshmen_psych_lib_arts_ratio : ℝ) 
  (h1 : freshmen_ratio = 0.4)
  (h2 : psych_majors_ratio = 0.5)
  (h3 : freshmen_psych_lib_arts_ratio = 0.1) :
  (freshmen_psych_lib_arts_ratio * total_students) / (psych_majors_ratio * (freshmen_ratio * total_students)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l1659_165949


namespace NUMINAMATH_CALUDE_savings_calculation_l1659_165969

theorem savings_calculation (savings : ℚ) (tv_cost : ℚ) 
  (h1 : tv_cost = 150)
  (h2 : (1 : ℚ) / 4 * savings = tv_cost) : 
  savings = 600 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1659_165969


namespace NUMINAMATH_CALUDE_shifted_data_invariants_l1659_165923

variable {n : ℕ}
variable (X Y : Fin n → ℝ)
variable (c : ℝ)

def is_shifted (X Y : Fin n → ℝ) (c : ℝ) : Prop :=
  ∀ i, Y i = X i + c

def standard_deviation (X : Fin n → ℝ) : ℝ := sorry

def range (X : Fin n → ℝ) : ℝ := sorry

theorem shifted_data_invariants (h : is_shifted X Y c) (h_nonzero : c ≠ 0) :
  standard_deviation Y = standard_deviation X ∧ range Y = range X := by sorry

end NUMINAMATH_CALUDE_shifted_data_invariants_l1659_165923


namespace NUMINAMATH_CALUDE_minimum_buses_needed_l1659_165999

def students : ℕ := 535
def bus_capacity : ℕ := 45

theorem minimum_buses_needed : 
  ∃ (n : ℕ), n * bus_capacity ≥ students ∧ 
  ∀ (m : ℕ), m * bus_capacity ≥ students → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_minimum_buses_needed_l1659_165999


namespace NUMINAMATH_CALUDE_cars_sold_first_day_l1659_165914

theorem cars_sold_first_day (total : ℕ) (second_day : ℕ) (third_day : ℕ)
  (h1 : total = 57)
  (h2 : second_day = 16)
  (h3 : third_day = 27) :
  total - second_day - third_day = 14 := by
  sorry

end NUMINAMATH_CALUDE_cars_sold_first_day_l1659_165914


namespace NUMINAMATH_CALUDE_division_sum_theorem_l1659_165903

theorem division_sum_theorem (dividend : ℕ) (divisor : ℕ) (h1 : dividend = 54) (h2 : divisor = 9) :
  dividend / divisor + dividend + divisor = 69 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l1659_165903


namespace NUMINAMATH_CALUDE_farmer_milk_production_l1659_165937

/-- Calculates the total milk production for a given number of cows over a week -/
def totalMilkProduction (numCows : ℕ) (milkPerDay : ℕ) : ℕ :=
  numCows * milkPerDay * 7

/-- Proves that 52 cows producing 5 liters of milk per day will produce 1820 liters in a week -/
theorem farmer_milk_production :
  totalMilkProduction 52 5 = 1820 := by
  sorry

#eval totalMilkProduction 52 5

end NUMINAMATH_CALUDE_farmer_milk_production_l1659_165937


namespace NUMINAMATH_CALUDE_fair_coin_same_side_five_tosses_l1659_165946

/-- A fair coin is a coin with equal probability of landing on either side -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a sequence of independent events -/
def prob_sequence (p : ℝ) (n : ℕ) : ℝ := p ^ n

/-- The number of tosses -/
def num_tosses : ℕ := 5

/-- Theorem: The probability of a fair coin landing on the same side for 5 tosses is 1/32 -/
theorem fair_coin_same_side_five_tosses (p : ℝ) (h : fair_coin p) :
  prob_sequence p num_tosses = 1 / 32 := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_same_side_five_tosses_l1659_165946


namespace NUMINAMATH_CALUDE_matthew_crackers_left_l1659_165971

/-- Calculates the number of crackers Matthew has left after distributing them to friends and the friends eating some. -/
def crackers_left (initial_crackers : ℕ) (num_friends : ℕ) (crackers_eaten_per_friend : ℕ) : ℕ :=
  let distributed_crackers := initial_crackers - 1
  let crackers_per_friend := distributed_crackers / num_friends
  let remaining_with_friends := (crackers_per_friend - crackers_eaten_per_friend) * num_friends
  1 + remaining_with_friends

/-- Proves that Matthew has 11 crackers left given the initial conditions. -/
theorem matthew_crackers_left :
  crackers_left 23 2 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_left_l1659_165971


namespace NUMINAMATH_CALUDE_upstream_distance_is_96_l1659_165956

/-- Represents the boat's journey on a river -/
structure RiverJourney where
  boatSpeed : ℝ
  riverSpeed : ℝ
  downstreamDistance : ℝ
  downstreamTime : ℝ
  upstreamTime : ℝ

/-- Calculates the upstream distance for a given river journey -/
def upstreamDistance (journey : RiverJourney) : ℝ :=
  (journey.boatSpeed - journey.riverSpeed) * journey.upstreamTime

/-- Theorem stating that for the given conditions, the upstream distance is 96 km -/
theorem upstream_distance_is_96 (journey : RiverJourney) 
  (h1 : journey.boatSpeed = 14)
  (h2 : journey.downstreamDistance = 200)
  (h3 : journey.downstreamTime = 10)
  (h4 : journey.upstreamTime = 12)
  (h5 : journey.downstreamDistance = (journey.boatSpeed + journey.riverSpeed) * journey.downstreamTime) :
  upstreamDistance journey = 96 := by
  sorry

end NUMINAMATH_CALUDE_upstream_distance_is_96_l1659_165956


namespace NUMINAMATH_CALUDE_max_m_and_min_sum_l1659_165980

theorem max_m_and_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ m : ℝ, (3 / a + 1 / b ≥ m / (a + 3 * b)) → m ≤ 12) ∧
  (a + 2 * b + 2 * a * b = 8 → a + 2 * b ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_max_m_and_min_sum_l1659_165980


namespace NUMINAMATH_CALUDE_least_value_with_specific_remainders_l1659_165962

theorem least_value_with_specific_remainders :
  ∃ (N : ℕ), 
    N > 0 ∧
    N % 6 = 5 ∧
    N % 5 = 4 ∧
    N % 4 = 3 ∧
    N % 3 = 2 ∧
    N % 2 = 1 ∧
    (∀ (M : ℕ), M > 0 ∧ 
      M % 6 = 5 ∧
      M % 5 = 4 ∧
      M % 4 = 3 ∧
      M % 3 = 2 ∧
      M % 2 = 1 → M ≥ N) ∧
    N = 59 :=
by sorry

end NUMINAMATH_CALUDE_least_value_with_specific_remainders_l1659_165962


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1659_165945

theorem trigonometric_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1659_165945


namespace NUMINAMATH_CALUDE_yellow_pencils_count_l1659_165953

/-- Represents a grid of colored pencils -/
structure PencilGrid :=
  (size : ℕ)
  (perimeter_color : String)
  (inside_color : String)

/-- Calculates the number of pencils of the inside color in the grid -/
def count_inside_pencils (grid : PencilGrid) : ℕ :=
  grid.size * grid.size - (4 * grid.size - 4)

/-- The theorem to be proved -/
theorem yellow_pencils_count (grid : PencilGrid) 
  (h1 : grid.size = 10)
  (h2 : grid.perimeter_color = "red")
  (h3 : grid.inside_color = "yellow") :
  count_inside_pencils grid = 64 := by
  sorry

end NUMINAMATH_CALUDE_yellow_pencils_count_l1659_165953


namespace NUMINAMATH_CALUDE_fixed_fee_is_7_42_l1659_165991

/-- Represents the billing structure and usage for an online service provider -/
structure BillingInfo where
  fixedFee : ℝ
  hourlyCharge : ℝ
  decemberUsage : ℝ
  januaryUsage : ℝ

/-- Calculates the total bill based on fixed fee, hourly charge, and usage -/
def calculateBill (info : BillingInfo) (usage : ℝ) : ℝ :=
  info.fixedFee + info.hourlyCharge * usage

/-- Theorem stating that under given conditions, the fixed monthly fee is $7.42 -/
theorem fixed_fee_is_7_42 (info : BillingInfo) :
  calculateBill info info.decemberUsage = 12.48 →
  calculateBill info info.januaryUsage = 17.54 →
  info.januaryUsage = 2 * info.decemberUsage →
  info.fixedFee = 7.42 := by
  sorry

#eval (7.42 : Float)

end NUMINAMATH_CALUDE_fixed_fee_is_7_42_l1659_165991


namespace NUMINAMATH_CALUDE_eccentricity_range_l1659_165987

/-- An ellipse with center O and endpoint A of its major axis -/
structure Ellipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  eccentricity : ℝ

/-- The condition that there is no point P on the ellipse such that ∠OPA = π/2 -/
def noRightAngle (e : Ellipse) : Prop :=
  ∀ p : ℝ × ℝ, p ≠ e.center → p ≠ (e.center.1 + e.majorAxis, e.center.2) →
    (p.1 - e.center.1)^2 + (p.2 - e.center.2)^2 = e.majorAxis^2 * (1 - e.eccentricity^2) →
    (p.1 - e.center.1) * (p.1 - (e.center.1 + e.majorAxis)) +
    (p.2 - e.center.2) * p.2 ≠ 0

/-- The theorem stating the range of eccentricity -/
theorem eccentricity_range (e : Ellipse) :
  0 < e.eccentricity ∧ e.eccentricity < 1 ∧ noRightAngle e →
  0 < e.eccentricity ∧ e.eccentricity ≤ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_range_l1659_165987


namespace NUMINAMATH_CALUDE_jose_profit_share_l1659_165908

def calculate_share (investment : ℕ) (duration : ℕ) (total_ratio : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_ratio

theorem jose_profit_share 
  (tom_investment : ℕ) (tom_duration : ℕ)
  (jose_investment : ℕ) (jose_duration : ℕ)
  (total_profit : ℕ) :
  tom_investment = 30000 →
  tom_duration = 12 →
  jose_investment = 45000 →
  jose_duration = 10 →
  total_profit = 45000 →
  calculate_share jose_investment jose_duration 
    (tom_investment * tom_duration + jose_investment * jose_duration) 
    total_profit = 25000 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l1659_165908


namespace NUMINAMATH_CALUDE_find_number_l1659_165943

/-- Given two positive integers with specific LCM and HCF, prove one number given the other -/
theorem find_number (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 30) (h3 : B = 150) :
  A = 462 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1659_165943


namespace NUMINAMATH_CALUDE_sum_of_decimal_and_fraction_l1659_165941

theorem sum_of_decimal_and_fraction : 7.31 + (1 / 5 : ℚ) = 7.51 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimal_and_fraction_l1659_165941


namespace NUMINAMATH_CALUDE_acute_inclination_implies_ab_negative_l1659_165998

-- Define a line with coefficients a, b, and c
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of having an acute angle of inclination
def hasAcuteInclination (l : Line) : Prop :=
  0 < -l.a / l.b ∧ -l.a / l.b < 1

-- Theorem statement
theorem acute_inclination_implies_ab_negative (l : Line) :
  hasAcuteInclination l → l.a * l.b < 0 := by
  sorry

end NUMINAMATH_CALUDE_acute_inclination_implies_ab_negative_l1659_165998


namespace NUMINAMATH_CALUDE_salary_after_changes_l1659_165958

def initial_salary : ℝ := 3000
def raise_percentage : ℝ := 0.15
def cut_percentage : ℝ := 0.25

theorem salary_after_changes : 
  initial_salary * (1 + raise_percentage) * (1 - cut_percentage) = 2587.5 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_changes_l1659_165958
