import Mathlib

namespace NUMINAMATH_CALUDE_cube_occupation_percentage_l4104_410440

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along a given dimension -/
def cubesFit (dimension cubeSide : ℕ) : ℕ :=
  dimension / cubeSide

/-- Calculates the volume occupied by cubes in the box -/
def occupiedVolume (d : BoxDimensions) (cubeSide : ℕ) : ℕ :=
  (cubesFit d.length cubeSide) * (cubesFit d.width cubeSide) * (cubesFit d.height cubeSide) * (cubeSide ^ 3)

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a 
    8x7x12 inch box is equal to 4/7 -/
theorem cube_occupation_percentage :
  let boxDim : BoxDimensions := { length := 8, width := 7, height := 12 }
  let cubeSide : ℕ := 4
  (occupiedVolume boxDim cubeSide : ℚ) / (boxVolume boxDim : ℚ) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_occupation_percentage_l4104_410440


namespace NUMINAMATH_CALUDE_asymptotes_of_specific_hyperbola_l4104_410492

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

/-- The specific hyperbola we're interested in -/
def specific_hyperbola : Hyperbola where
  a := 1
  b := 2
  h_pos := by simp

theorem asymptotes_of_specific_hyperbola :
  ∀ x y : ℝ, asymptote_equation specific_hyperbola x y ↔ (y = 2*x ∨ y = -2*x) :=
sorry

end NUMINAMATH_CALUDE_asymptotes_of_specific_hyperbola_l4104_410492


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_coefficients_l4104_410473

def f (x : ℝ) : ℝ := 2 * x^2 - x + 5

def g (x : ℝ) : ℝ := f (x - 7) + 3

theorem quadratic_shift_sum_coefficients :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 86) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_coefficients_l4104_410473


namespace NUMINAMATH_CALUDE_trapezoid_sides_l4104_410426

/-- A rectangular trapezoid with an inscribed circle -/
structure RectangularTrapezoid (r : ℝ) where
  /-- The radius of the inscribed circle -/
  radius : r > 0
  /-- The shorter base of the trapezoid -/
  short_base : ℝ
  /-- The longer base of the trapezoid -/
  long_base : ℝ
  /-- One of the non-parallel sides of the trapezoid -/
  side1 : ℝ
  /-- The other non-parallel side of the trapezoid -/
  side2 : ℝ
  /-- The shorter base is equal to 4r/3 -/
  short_base_eq : short_base = 4*r/3
  /-- The circle is inscribed, so one non-parallel side equals the diameter -/
  side1_eq_diameter : side1 = 2*r
  /-- Property of trapezoids with an inscribed circle -/
  inscribed_circle_property : side1 + long_base = short_base + side2

/-- Theorem: The sides of the rectangular trapezoid with an inscribed circle of radius r 
    and shorter base 4r/3 are 4r, 10r/3, and 2r -/
theorem trapezoid_sides (r : ℝ) (t : RectangularTrapezoid r) : 
  t.short_base = 4*r/3 ∧ t.long_base = 10*r/3 ∧ t.side1 = 2*r ∧ t.side2 = 8*r/3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l4104_410426


namespace NUMINAMATH_CALUDE_eliana_fuel_cost_l4104_410446

/-- The amount Eliana spent on fuel in a week -/
def fuel_cost (refill_cost : ℕ) (refill_count : ℕ) : ℕ :=
  refill_cost * refill_count

/-- Proof that Eliana spent $63 on fuel this week -/
theorem eliana_fuel_cost :
  fuel_cost 21 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_eliana_fuel_cost_l4104_410446


namespace NUMINAMATH_CALUDE_composition_ratio_l4104_410465

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 151 / 121 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l4104_410465


namespace NUMINAMATH_CALUDE_min_value_sum_l4104_410455

theorem min_value_sum (a b : ℝ) (h : a^2 + 2*b^2 = 6) : 
  ∃ (m : ℝ), m = -3 ∧ ∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l4104_410455


namespace NUMINAMATH_CALUDE_min_value_theorem_l4104_410497

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3 * y = 1 → 1 / a + 2 / b ≤ 1 / x + 2 / y) ∧
  1 / a + 2 / b = 3 * Real.sqrt 6 + 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4104_410497


namespace NUMINAMATH_CALUDE_arithmetic_simplifications_l4104_410454

theorem arithmetic_simplifications :
  (∀ (a b c : Rat), a / 16 - b / 16 + c / 16 = (a - b + c) / 16) ∧
  (∀ (d e f : Rat), d / 12 - e / 12 + f / 12 = (d - e + f) / 12) ∧
  (∀ (g h i j k l m : Nat), g + h + i + j + k + l + m = 736) ∧
  (∀ (n p q r : Rat), n - p / 9 - q / 9 + (1 + r / 99) = 2 + r / 99) →
  5 / 16 - 3 / 16 + 7 / 16 = 9 / 16 ∧
  3 / 12 - 4 / 12 + 6 / 12 = 5 / 12 ∧
  64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 ∧
  2 - 8 / 9 - 1 / 9 + (1 + 98 / 99) = 2 + 98 / 99 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_simplifications_l4104_410454


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l4104_410458

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (P : ConvexPolyhedron) : ℕ :=
  (P.vertices.choose 2) - P.edges - (2 * P.quadrilateral_faces)

/-- Theorem stating the number of space diagonals in the specific polyhedron -/
theorem specific_polyhedron_space_diagonals :
  ∃ (P : ConvexPolyhedron),
    P.vertices = 26 ∧
    P.edges = 60 ∧
    P.faces = 36 ∧
    P.triangular_faces = 24 ∧
    P.quadrilateral_faces = 12 ∧
    space_diagonals P = 241 := by
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l4104_410458


namespace NUMINAMATH_CALUDE_equation_solution_l4104_410434

theorem equation_solution (y : ℝ) (x : ℝ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l4104_410434


namespace NUMINAMATH_CALUDE_jenny_jellybeans_l4104_410457

/-- The fraction of jellybeans remaining after eating 25% -/
def remainingFraction : ℝ := 0.75

/-- The number of days that passed -/
def days : ℕ := 3

/-- The number of jellybeans remaining after 3 days -/
def remainingJellybeans : ℕ := 27

/-- The original number of jellybeans in Jenny's jar -/
def originalJellybeans : ℕ := 64

theorem jenny_jellybeans :
  (remainingFraction ^ days) * (originalJellybeans : ℝ) = remainingJellybeans := by
  sorry

end NUMINAMATH_CALUDE_jenny_jellybeans_l4104_410457


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base5_max_digit_sum_attainable_l4104_410468

/-- Given a positive integer n, returns the sum of its digits in base 5 representation -/
def sumOfDigitsBase5 (n : ℕ) : ℕ := sorry

/-- The greatest possible sum of digits in base 5 for integers less than 3139 -/
def maxDigitSum : ℕ := 16

theorem greatest_digit_sum_base5 :
  ∀ n : ℕ, n > 0 ∧ n < 3139 → sumOfDigitsBase5 n ≤ maxDigitSum :=
by sorry

theorem max_digit_sum_attainable :
  ∃ n : ℕ, n > 0 ∧ n < 3139 ∧ sumOfDigitsBase5 n = maxDigitSum :=
by sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base5_max_digit_sum_attainable_l4104_410468


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l4104_410414

theorem right_triangle_perimeter_equals_area :
  ∀ a b c : ℕ,
    a ≤ b → b ≤ c →
    a^2 + b^2 = c^2 →
    a + b + c = (a * b) / 2 →
    ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l4104_410414


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l4104_410467

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (h : edge_length = Real.sqrt 3) : 
  let cube_diagonal := Real.sqrt (3 * edge_length ^ 2)
  let sphere_radius := cube_diagonal / 2
  4 * Real.pi * sphere_radius ^ 2 = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l4104_410467


namespace NUMINAMATH_CALUDE_two_points_with_area_three_l4104_410481

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x/4 + y/3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- Intersection points of the line and ellipse -/
structure IntersectionPoints where
  A : PointOnEllipse
  B : PointOnEllipse
  on_line_A : line A.x A.y
  on_line_B : line B.x B.y

/-- Area of a triangle given three points -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem two_points_with_area_three (intersections : IntersectionPoints) :
  ∃! (points : Finset PointOnEllipse),
    points.card = 2 ∧
    ∀ P ∈ points,
      triangleArea (P.x, P.y) (intersections.A.x, intersections.A.y) (intersections.B.x, intersections.B.y) = 3 :=
sorry

end NUMINAMATH_CALUDE_two_points_with_area_three_l4104_410481


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4104_410494

theorem geometric_series_sum : 
  let s := ∑' k, (3^k : ℝ) / (9^k - 1)
  s = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4104_410494


namespace NUMINAMATH_CALUDE_tessellating_nonagon_angles_l4104_410438

/-- A nonagon that tessellates the plane and can be decomposed into seven triangles -/
structure TessellatingNonagon where
  /-- The vertices of the nonagon -/
  vertices : Fin 9 → ℝ × ℝ
  /-- The nonagon tessellates the plane -/
  tessellates : sorry
  /-- The nonagon can be decomposed into seven triangles -/
  decomposable : sorry
  /-- Some sides of the nonagon form rhombuses with equal side lengths -/
  has_rhombuses : sorry

/-- The angles of a tessellating nonagon -/
def nonagon_angles (n : TessellatingNonagon) : Fin 9 → ℝ := sorry

/-- Theorem stating the angles of the tessellating nonagon -/
theorem tessellating_nonagon_angles (n : TessellatingNonagon) :
  nonagon_angles n = ![105, 60, 195, 195, 195, 15, 165, 165, 165] := by sorry

end NUMINAMATH_CALUDE_tessellating_nonagon_angles_l4104_410438


namespace NUMINAMATH_CALUDE_alice_acorn_price_l4104_410487

/-- Given the conditions of Alice and Bob's acorn purchases, prove that Alice paid $15 for each acorn. -/
theorem alice_acorn_price (alice_acorns : ℕ) (bob_price : ℝ) (alice_bob_ratio : ℝ) : 
  alice_acorns = 3600 → 
  bob_price = 6000 → 
  alice_bob_ratio = 9 → 
  (alice_bob_ratio * bob_price) / alice_acorns = 15 := by
sorry

end NUMINAMATH_CALUDE_alice_acorn_price_l4104_410487


namespace NUMINAMATH_CALUDE_non_integer_factors_integer_products_l4104_410442

theorem non_integer_factors_integer_products :
  ∃ (a b c : ℝ),
    (¬ ∃ (n : ℤ), a = n) ∧
    (¬ ∃ (n : ℤ), b = n) ∧
    (¬ ∃ (n : ℤ), c = n) ∧
    (∃ (m : ℤ), a * b = m) ∧
    (∃ (m : ℤ), b * c = m) ∧
    (∃ (m : ℤ), c * a = m) ∧
    (∃ (m : ℤ), a * b * c = m) :=
by sorry

end NUMINAMATH_CALUDE_non_integer_factors_integer_products_l4104_410442


namespace NUMINAMATH_CALUDE_pencil_distribution_l4104_410475

theorem pencil_distribution (total_pencils : ℕ) (pencils_per_student : ℕ) (h1 : total_pencils = 42) (h2 : pencils_per_student = 3) :
  total_pencils / pencils_per_student = 14 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4104_410475


namespace NUMINAMATH_CALUDE_ultra_squarish_exists_l4104_410488

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def digits_nonzero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

def first_three_digits (n : ℕ) : ℕ :=
  (n / 10000) % 1000

def middle_two_digits (n : ℕ) : ℕ :=
  (n / 100) % 100

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem ultra_squarish_exists :
  ∃ M : ℕ,
    1000000 ≤ M ∧ M < 10000000 ∧
    is_perfect_square M ∧
    digits_nonzero M ∧
    is_perfect_square (first_three_digits M) ∧
    is_perfect_square (middle_two_digits M) ∧
    is_perfect_square (last_two_digits M) :=
  sorry

end NUMINAMATH_CALUDE_ultra_squarish_exists_l4104_410488


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l4104_410448

theorem gcd_of_powers_of_two : Nat.gcd (2^1502 - 1) (2^1513 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l4104_410448


namespace NUMINAMATH_CALUDE_office_network_connections_l4104_410401

/-- Represents a network of switches with their connections -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem stating that a network of 30 switches, each connected to 4 others, has 60 connections -/
theorem office_network_connections :
  let network : SwitchNetwork := { num_switches := 30, connections_per_switch := 4 }
  total_connections network = 60 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l4104_410401


namespace NUMINAMATH_CALUDE_blakes_initial_money_l4104_410478

/-- Blake's grocery shopping problem -/
theorem blakes_initial_money (orange_cost apples_cost mangoes_cost change : ℕ) 
  (h1 : orange_cost = 40)
  (h2 : apples_cost = 50)
  (h3 : mangoes_cost = 60)
  (h4 : change = 150) : 
  orange_cost + apples_cost + mangoes_cost + change = 300 := by
  sorry

#check blakes_initial_money

end NUMINAMATH_CALUDE_blakes_initial_money_l4104_410478


namespace NUMINAMATH_CALUDE_f_properties_l4104_410432

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3) + 1

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 5 * Real.pi / 12 → f x₁ < f x₂) ∧
  (∀ x₁ x₂ x₃, x₁ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               x₂ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               x₃ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               f x₁ + f x₃ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4104_410432


namespace NUMINAMATH_CALUDE_factorization_equality_l4104_410495

theorem factorization_equality (m n : ℝ) : 4 * m^3 * n - 16 * m * n^3 = 4 * m * n * (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4104_410495


namespace NUMINAMATH_CALUDE_cube_equation_solution_l4104_410484

theorem cube_equation_solution (x y : ℝ) : x^3 - 8*y^3 = 0 ↔ x = 2*y := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l4104_410484


namespace NUMINAMATH_CALUDE_mary_saw_36_snakes_l4104_410451

/-- The total number of snakes Mary saw -/
def total_snakes (breeding_balls : ℕ) (snakes_per_ball : ℕ) (additional_pairs : ℕ) : ℕ :=
  breeding_balls * snakes_per_ball + additional_pairs * 2

/-- Theorem stating that Mary saw 36 snakes in total -/
theorem mary_saw_36_snakes :
  total_snakes 3 8 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_mary_saw_36_snakes_l4104_410451


namespace NUMINAMATH_CALUDE_green_marble_fraction_l4104_410447

theorem green_marble_fraction (total : ℝ) (h1 : total > 0) : 
  let initial_green := (1/4) * total
  let initial_yellow := total - initial_green
  let new_green := 3 * initial_green
  let new_total := new_green + initial_yellow
  new_green / new_total = 1/2 := by sorry

end NUMINAMATH_CALUDE_green_marble_fraction_l4104_410447


namespace NUMINAMATH_CALUDE_basketball_team_selection_l4104_410499

def num_players : ℕ := 12
def team_size : ℕ := 6
def captain_count : ℕ := 1

theorem basketball_team_selection :
  (num_players.choose captain_count) * ((num_players - captain_count).choose (team_size - captain_count)) = 5544 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l4104_410499


namespace NUMINAMATH_CALUDE_euler_family_mean_age_is_11_l4104_410476

def euler_family_mean_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

theorem euler_family_mean_age_is_11 :
  let ages := [8, 8, 8, 13, 13, 16]
  euler_family_mean_age ages = 11 := by
sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_is_11_l4104_410476


namespace NUMINAMATH_CALUDE_square_ends_in_001_l4104_410429

theorem square_ends_in_001 (x : ℤ) : 
  x^2 ≡ 1 [ZMOD 1000] → 
  (x ≡ 1 [ZMOD 500] ∨ x ≡ -1 [ZMOD 500] ∨ x ≡ 249 [ZMOD 500] ∨ x ≡ -249 [ZMOD 500]) :=
by sorry

end NUMINAMATH_CALUDE_square_ends_in_001_l4104_410429


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_36_l4104_410464

theorem smallest_non_factor_product_of_36 (x y : ℕ+) : 
  x ≠ y →
  x ∣ 36 →
  y ∣ 36 →
  ¬(x * y ∣ 36) →
  (∀ a b : ℕ+, a ≠ b → a ∣ 36 → b ∣ 36 → ¬(a * b ∣ 36) → x * y ≤ a * b) →
  x * y = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_36_l4104_410464


namespace NUMINAMATH_CALUDE_max_value_f_one_l4104_410422

/-- Given a function f(x) = x^2 + abx + a + 2b where f(0) = 4,
    the maximum value of f(1) is 7. -/
theorem max_value_f_one (a b : ℝ) :
  let f := fun x : ℝ => x^2 + a*b*x + a + 2*b
  f 0 = 4 →
  (∀ x : ℝ, f 1 ≤ 7) ∧ (∃ x : ℝ, f 1 = 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_one_l4104_410422


namespace NUMINAMATH_CALUDE_additive_function_is_scalar_multiple_l4104_410483

/-- A function from rationals to rationals satisfying the given additive property -/
def AdditiveFunction (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

/-- The theorem stating that any additive function on rationals is a scalar multiple -/
theorem additive_function_is_scalar_multiple :
  ∀ f : ℚ → ℚ, AdditiveFunction f → ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_additive_function_is_scalar_multiple_l4104_410483


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l4104_410490

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l4104_410490


namespace NUMINAMATH_CALUDE_xy_sum_product_l4104_410462

theorem xy_sum_product (x y : ℝ) (h1 : x + y = 2 * Real.sqrt 3) (h2 : x * y = Real.sqrt 6) :
  x^2 * y + x * y^2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_product_l4104_410462


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l4104_410496

/-- Calculates the total number of birds in a pet store with specific cage arrangements. -/
theorem pet_store_bird_count : 
  let total_cages : ℕ := 9
  let parrots_per_mixed_cage : ℕ := 2
  let parakeets_per_mixed_cage : ℕ := 3
  let cockatiels_per_mixed_cage : ℕ := 1
  let parakeets_per_special_cage : ℕ := 5
  let special_cage_frequency : ℕ := 3

  let special_cages : ℕ := total_cages / special_cage_frequency
  let mixed_cages : ℕ := total_cages - special_cages

  let total_parrots : ℕ := mixed_cages * parrots_per_mixed_cage
  let total_parakeets : ℕ := (mixed_cages * parakeets_per_mixed_cage) + (special_cages * parakeets_per_special_cage)
  let total_cockatiels : ℕ := mixed_cages * cockatiels_per_mixed_cage

  let total_birds : ℕ := total_parrots + total_parakeets + total_cockatiels
  
  total_birds = 51 := by sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l4104_410496


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_seven_l4104_410411

theorem no_solution_iff_m_equals_seven (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) ≠ (x - m) / (x - 8)) ↔ m = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_seven_l4104_410411


namespace NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l4104_410437

theorem inscribed_circles_area_ratio (s : ℝ) (h : s > 0) : 
  let square_side := s
  let large_circle_radius := s / 2
  let triangle_side := s * (Real.sqrt 3) / 2
  let small_circle_radius := s * (Real.sqrt 3) / 12
  (π * (small_circle_radius ^ 2)) / (square_side ^ 2) = π / 48 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l4104_410437


namespace NUMINAMATH_CALUDE_second_reduction_percentage_l4104_410421

/-- Given two successive price reductions, where the first is 25% and the combined effect
    is equivalent to a single 47.5% reduction, proves that the second reduction is 30%. -/
theorem second_reduction_percentage (P : ℝ) (x : ℝ) 
  (h1 : P > 0)  -- Assume positive initial price
  (h2 : 0 ≤ x ∧ x ≤ 1)  -- Second reduction percentage is between 0 and 1
  (h3 : (1 - x) * (P - 0.25 * P) = P - 0.475 * P)  -- Combined reduction equation
  : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_l4104_410421


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l4104_410406

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l4104_410406


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l4104_410489

def is_valid_assignment (w h i t e a r p c n : Nat) : Prop :=
  w < 10 ∧ h < 10 ∧ i < 10 ∧ t < 10 ∧ e < 10 ∧ a < 10 ∧ r < 10 ∧ p < 10 ∧ c < 10 ∧ n < 10 ∧
  w ≠ h ∧ w ≠ i ∧ w ≠ t ∧ w ≠ e ∧ w ≠ a ∧ w ≠ r ∧ w ≠ p ∧ w ≠ c ∧ w ≠ n ∧
  h ≠ i ∧ h ≠ t ∧ h ≠ e ∧ h ≠ a ∧ h ≠ r ∧ h ≠ p ∧ h ≠ c ∧ h ≠ n ∧
  i ≠ t ∧ i ≠ e ∧ i ≠ a ∧ i ≠ r ∧ i ≠ p ∧ i ≠ c ∧ i ≠ n ∧
  t ≠ e ∧ t ≠ a ∧ t ≠ r ∧ t ≠ p ∧ t ≠ c ∧ t ≠ n ∧
  e ≠ a ∧ e ≠ r ∧ e ≠ p ∧ e ≠ c ∧ e ≠ n ∧
  a ≠ r ∧ a ≠ p ∧ a ≠ c ∧ a ≠ n ∧
  r ≠ p ∧ r ≠ c ∧ r ≠ n ∧
  p ≠ c ∧ p ≠ n ∧
  c ≠ n

def white_plus_water_equals_picnic (w h i t e a r p c n : Nat) : Prop :=
  10000 * w + 1000 * h + 100 * i + 10 * t + e +
  10000 * w + 1000 * a + 100 * t + 10 * e + r =
  100000 * p + 10000 * i + 1000 * c + 100 * n + 10 * i + c

theorem cryptarithmetic_solution :
  ∃! (w h i t e a r p c n : Nat),
    is_valid_assignment w h i t e a r p c n ∧
    white_plus_water_equals_picnic w h i t e a r p c n ∧
    100000 * p + 10000 * i + 1000 * c + 100 * n + 10 * i + c = 169069 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l4104_410489


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l4104_410424

theorem least_n_with_gcd_conditions : 
  ∃ n : ℕ, n > 500 ∧ 
    Nat.gcd 42 (n + 80) = 14 ∧ 
    Nat.gcd (n + 42) 80 = 40 ∧
    (∀ m : ℕ, m > 500 → 
      Nat.gcd 42 (m + 80) = 14 → 
      Nat.gcd (m + 42) 80 = 40 → 
      n ≤ m) ∧
    n = 638 :=
by sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l4104_410424


namespace NUMINAMATH_CALUDE_not_always_equal_l4104_410435

theorem not_always_equal : ∃ (a b : ℝ), 3 * (a + b) ≠ 3 * a + b := by
  sorry

end NUMINAMATH_CALUDE_not_always_equal_l4104_410435


namespace NUMINAMATH_CALUDE_car_not_speeding_l4104_410423

/-- Braking distance function -/
def braking_distance (x : ℝ) : ℝ := 0.01 * x + 0.002 * x^2

/-- Speed limit in km/h -/
def speed_limit : ℝ := 120

/-- Measured braking distance in meters -/
def measured_distance : ℝ := 26.5

/-- Theorem: There exists a speed less than the speed limit that results in the measured braking distance -/
theorem car_not_speeding : ∃ x : ℝ, x < speed_limit ∧ braking_distance x = measured_distance := by
  sorry


end NUMINAMATH_CALUDE_car_not_speeding_l4104_410423


namespace NUMINAMATH_CALUDE_count_theorem_l4104_410412

/-- The count of positive integers less than 3000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  -- Definition placeholder
  0

/-- The upper bound of the considered range -/
def upper_bound : ℕ := 3000

/-- Predicate to check if a number has at most three different digits -/
def has_at_most_three_digits (n : ℕ) : Prop :=
  -- Definition placeholder
  True

theorem count_theorem :
  count_numbers_with_at_most_three_digits = 891 :=
sorry


end NUMINAMATH_CALUDE_count_theorem_l4104_410412


namespace NUMINAMATH_CALUDE_lcm_48_180_l4104_410452

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l4104_410452


namespace NUMINAMATH_CALUDE_viviana_vanilla_chips_l4104_410443

/-- Given the conditions about chocolate and vanilla chips, prove that Viviana has 20 vanilla chips. -/
theorem viviana_vanilla_chips 
  (viviana_chocolate : ℕ) 
  (viviana_vanilla : ℕ) 
  (susana_chocolate : ℕ) 
  (susana_vanilla : ℕ) 
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : susana_chocolate = 25)
  (h4 : viviana_chocolate + susana_chocolate + viviana_vanilla + susana_vanilla = 90) :
  viviana_vanilla = 20 := by
sorry

end NUMINAMATH_CALUDE_viviana_vanilla_chips_l4104_410443


namespace NUMINAMATH_CALUDE_half_vector_AB_l4104_410491

/-- Given two points A and B in a 2D plane, prove that half of the vector from A to B is (2, 1) -/
theorem half_vector_AB (A B : ℝ × ℝ) (h1 : A = (-1, 0)) (h2 : B = (3, 2)) :
  (1 / 2 : ℝ) • (B.1 - A.1, B.2 - A.2) = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_half_vector_AB_l4104_410491


namespace NUMINAMATH_CALUDE_smallest_solution_l4104_410415

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 27) / x = 15

-- State the theorem
theorem smallest_solution :
  ∃ (s : ℝ), s = 1 - Real.sqrt 10 ∧
  equation s ∧
  (∀ (x : ℝ), equation x → x ≥ s) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l4104_410415


namespace NUMINAMATH_CALUDE_chenny_spoons_l4104_410407

/-- Given the following:
  * Chenny bought 9 plates at $2 each
  * Spoons cost $1.50 each
  * The total paid for plates and spoons is $24
  Prove that Chenny bought 4 spoons -/
theorem chenny_spoons (num_plates : ℕ) (price_plate : ℚ) (price_spoon : ℚ) (total_paid : ℚ) :
  num_plates = 9 →
  price_plate = 2 →
  price_spoon = 3/2 →
  total_paid = 24 →
  (total_paid - num_plates * price_plate) / price_spoon = 4 :=
by sorry

end NUMINAMATH_CALUDE_chenny_spoons_l4104_410407


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4104_410449

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + 1/y^2 + 1) * (x^2 + 1/y^2 - 1000) + (y^2 + 1/x^2 + 1) * (y^2 + 1/x^2 - 1000) ≥ -498998 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
  (x^2 + 1/y^2 + 1) * (x^2 + 1/y^2 - 1000) + (y^2 + 1/x^2 + 1) * (y^2 + 1/x^2 - 1000) = -498998 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4104_410449


namespace NUMINAMATH_CALUDE_absolute_value_at_zero_l4104_410416

-- Define a fourth-degree polynomial with real coefficients
def fourthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x => a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem absolute_value_at_zero (a b c d e : ℝ) :
  let g := fourthDegreePolynomial a b c d e
  (|g 1| = 16 ∧ |g 3| = 16 ∧ |g 4| = 16 ∧ |g 5| = 16 ∧ |g 6| = 16 ∧ |g 7| = 16) →
  |g 0| = 54 := by
  sorry


end NUMINAMATH_CALUDE_absolute_value_at_zero_l4104_410416


namespace NUMINAMATH_CALUDE_f_range_l4104_410419

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 + 2 * Real.sin x + 3 * Real.cos x ^ 2 - 6) / (Real.sin x - 1)

theorem f_range : 
  ∀ (y : ℝ), (∃ (x : ℝ), Real.sin x ≠ 1 ∧ f x = y) ↔ (0 ≤ y ∧ y < 8) := by
  sorry

end NUMINAMATH_CALUDE_f_range_l4104_410419


namespace NUMINAMATH_CALUDE_night_crew_ratio_l4104_410453

theorem night_crew_ratio (D N : ℚ) (h1 : D > 0) (h2 : N > 0) : 
  (N * (3/4)) / (D + N * (3/4)) = 1/3 → N/D = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_night_crew_ratio_l4104_410453


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l4104_410409

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- Perpendicular bisector equation -/
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

/-- Theorem stating that the perpendicular bisector of AB is 3x - y - 9 = 0 -/
theorem perpendicular_bisector_of_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l4104_410409


namespace NUMINAMATH_CALUDE_fraction_simplification_l4104_410408

theorem fraction_simplification (a : ℝ) (h : a ≠ 0) : (a^2 - 1) / a + 1 / a = a := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4104_410408


namespace NUMINAMATH_CALUDE_equation_solutions_l4104_410460

theorem equation_solutions :
  (∃ x : ℝ, 4 * x + 3 = 5 * x - 1 ∧ x = 4) ∧
  (∃ x : ℝ, 4 * (x - 1) = 1 - x ∧ x = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4104_410460


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_l4104_410404

/-- Given a polynomial x^3 - 2023x + n with integer roots p, q, and r, 
    prove that the sum of their absolute values is 84 -/
theorem sum_of_abs_roots (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) → 
  |p| + |q| + |r| = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_l4104_410404


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l4104_410485

/-- Prove that a jogger is 250 meters ahead of a train's engine given the specified conditions -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5/18) →
  train_speed = 45 * (5/18) →
  train_length = 120 →
  passing_time = 37 →
  (train_speed - jogger_speed) * passing_time - train_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l4104_410485


namespace NUMINAMATH_CALUDE_max_investment_at_lower_rate_l4104_410456

theorem max_investment_at_lower_rate
  (total_investment : ℝ)
  (lower_rate : ℝ)
  (higher_rate : ℝ)
  (min_interest : ℝ)
  (h1 : total_investment = 25000)
  (h2 : lower_rate = 0.07)
  (h3 : higher_rate = 0.12)
  (h4 : min_interest = 2450)
  : ∃ (x : ℝ), x ≤ 11000 ∧
    x + (total_investment - x) = total_investment ∧
    lower_rate * x + higher_rate * (total_investment - x) ≥ min_interest ∧
    ∀ (y : ℝ), y > x →
      lower_rate * y + higher_rate * (total_investment - y) < min_interest :=
by sorry


end NUMINAMATH_CALUDE_max_investment_at_lower_rate_l4104_410456


namespace NUMINAMATH_CALUDE_max_value_theorem_l4104_410461

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2/2 = 1) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 + y'^2/2 = 1 →
    x' * Real.sqrt (1 + y'^2) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4104_410461


namespace NUMINAMATH_CALUDE_chosen_numbers_sum_l4104_410430

theorem chosen_numbers_sum (S : Finset ℕ) : 
  S.card = 5 ∧ 
  S ⊆ Finset.range 9 ∧ 
  S.sum id = ((Finset.range 9).sum id - S.sum id) / 2 → 
  S.sum id = 15 := by sorry

end NUMINAMATH_CALUDE_chosen_numbers_sum_l4104_410430


namespace NUMINAMATH_CALUDE_division_problem_l4104_410425

theorem division_problem (number quotient remainder divisor : ℕ) : 
  number = quotient * divisor + remainder →
  divisor = 163 →
  quotient = 76 →
  remainder = 13 →
  number = 12401 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l4104_410425


namespace NUMINAMATH_CALUDE_mobile_price_change_l4104_410479

theorem mobile_price_change (initial_price : ℝ) (decrease_percent : ℝ) : 
  (initial_price * 1.4 * (1 - decrease_percent / 100) = initial_price * 1.18999999999999993) →
  decrease_percent = 15 := by
sorry

end NUMINAMATH_CALUDE_mobile_price_change_l4104_410479


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l4104_410428

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer given the direction. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Represents the conditions of the swimming problem. -/
structure SwimmingProblem where
  downstreamDistance : ℝ
  upstreamDistance : ℝ
  time : ℝ

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 6 km/h. -/
theorem swimmer_speed_in_still_water (p : SwimmingProblem)
  (h1 : p.downstreamDistance = 72)
  (h2 : p.upstreamDistance = 36)
  (h3 : p.time = 9)
  : ∃ (s : SwimmerSpeeds),
    effectiveSpeed s true * p.time = p.downstreamDistance ∧
    effectiveSpeed s false * p.time = p.upstreamDistance ∧
    s.swimmer = 6 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l4104_410428


namespace NUMINAMATH_CALUDE_jason_final_pears_l4104_410420

def initial_pears : ℕ := 46
def pears_given_to_keith : ℕ := 47
def pears_received_from_mike : ℕ := 12

theorem jason_final_pears :
  (if initial_pears ≥ pears_given_to_keith
   then initial_pears - pears_given_to_keith
   else 0) + pears_received_from_mike = 12 := by
  sorry

end NUMINAMATH_CALUDE_jason_final_pears_l4104_410420


namespace NUMINAMATH_CALUDE_power_sum_of_i_l4104_410480

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^67 + i^101 = -i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l4104_410480


namespace NUMINAMATH_CALUDE_count_four_digit_with_three_is_1000_l4104_410474

/-- The count of four-digit positive integers with the thousands digit 3 -/
def count_four_digit_with_three : ℕ :=
  (List.range 10).length * (List.range 10).length * (List.range 10).length

/-- Theorem: The count of four-digit positive integers with the thousands digit 3 is 1000 -/
theorem count_four_digit_with_three_is_1000 :
  count_four_digit_with_three = 1000 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_with_three_is_1000_l4104_410474


namespace NUMINAMATH_CALUDE_eleven_rays_max_regions_l4104_410482

/-- The maximum number of regions into which n rays can split a plane -/
def max_regions (n : ℕ) : ℕ := (n^2 - n + 2) / 2

/-- Theorem: 11 rays can split a plane into a maximum of 56 regions -/
theorem eleven_rays_max_regions : max_regions 11 = 56 := by
  sorry

end NUMINAMATH_CALUDE_eleven_rays_max_regions_l4104_410482


namespace NUMINAMATH_CALUDE_youngest_child_age_l4104_410459

/-- Represents the age of the youngest child in a group of 5 children -/
def youngest_age (total_age : ℕ) : ℕ :=
  (total_age - 20) / 5

/-- Theorem stating that if the sum of ages of 5 children born at 2-year intervals is 50,
    then the age of the youngest child is 6 years -/
theorem youngest_child_age :
  youngest_age 50 = 6 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l4104_410459


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l4104_410486

theorem quadratic_discriminant : 
  let a : ℚ := 5
  let b : ℚ := 5 + (1 / 5)
  let c : ℚ := 1 / 5
  let discriminant := b^2 - 4*a*c
  discriminant = 576 / 25 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l4104_410486


namespace NUMINAMATH_CALUDE_star_difference_l4104_410463

-- Define the ⋆ operation
def star (x y : ℤ) : ℤ := x * y - 3 * x + 1

-- Theorem statement
theorem star_difference : star 5 3 - star 3 5 = -6 := by sorry

end NUMINAMATH_CALUDE_star_difference_l4104_410463


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l4104_410410

theorem shaded_fraction_of_rectangle (length width : ℝ) (shaded_quarter : ℝ) :
  length = 15 →
  width = 20 →
  shaded_quarter = (1 / 4) * (length * width) →
  shaded_quarter = (1 / 5) * (length * width) →
  shaded_quarter / (length * width) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l4104_410410


namespace NUMINAMATH_CALUDE_rearranged_number_bounds_l4104_410498

/-- Given a natural number B, returns the number A obtained by moving the last digit of B to the first position --/
def rearrange_digits (B : ℕ) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

/-- Checks if two natural numbers are coprime --/
def are_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Theorem stating the largest and smallest possible values of A given the conditions on B --/
theorem rearranged_number_bounds :
  ∀ B : ℕ,
  B > 222222222 →
  are_coprime B 18 →
  ∃ A : ℕ,
  A = rearrange_digits B ∧
  A ≤ 999999998 ∧
  A ≥ 122222224 ∧
  (∀ A' : ℕ, A' = rearrange_digits B → A' ≤ 999999998 ∧ A' ≥ 122222224) :=
sorry

end NUMINAMATH_CALUDE_rearranged_number_bounds_l4104_410498


namespace NUMINAMATH_CALUDE_winter_wheat_harvest_scientific_notation_l4104_410466

theorem winter_wheat_harvest_scientific_notation :
  239000000 = 2.39 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_winter_wheat_harvest_scientific_notation_l4104_410466


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l4104_410427

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if a = 1, C = 60°, and c = √3, then A = π/6 -/
theorem triangle_angle_calculation (a b c A B C : Real) : 
  a = 1 → C = π / 3 → c = Real.sqrt 3 → A = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l4104_410427


namespace NUMINAMATH_CALUDE_simplify_expression_l4104_410450

theorem simplify_expression : (81 * (10 ^ 12)) / (9 * (10 ^ 4)) = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4104_410450


namespace NUMINAMATH_CALUDE_engineer_designer_ratio_l4104_410441

/-- Represents the ratio of engineers to designers in a team -/
structure TeamRatio where
  engineers : ℕ
  designers : ℕ

/-- Proves that the ratio of engineers to designers is 2:1 given the average ages -/
theorem engineer_designer_ratio (team_avg : ℝ) (engineer_avg : ℝ) (designer_avg : ℝ) 
    (h1 : team_avg = 52) (h2 : engineer_avg = 48) (h3 : designer_avg = 60) : 
    ∃ (ratio : TeamRatio), ratio.engineers = 2 ∧ ratio.designers = 1 := by
  sorry

#check engineer_designer_ratio

end NUMINAMATH_CALUDE_engineer_designer_ratio_l4104_410441


namespace NUMINAMATH_CALUDE_quadratic_completion_square_constant_term_value_l4104_410433

theorem quadratic_completion_square (x : ℝ) : 
  x^2 - 8*x + 3 = (x - 4)^2 - 13 :=
by sorry

theorem constant_term_value : 
  ∃ (a h : ℝ), ∀ (x : ℝ), x^2 - 8*x + 3 = a*(x - h)^2 - 13 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_constant_term_value_l4104_410433


namespace NUMINAMATH_CALUDE_fraction_chain_l4104_410493

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by sorry

end NUMINAMATH_CALUDE_fraction_chain_l4104_410493


namespace NUMINAMATH_CALUDE_video_game_earnings_l4104_410477

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem video_game_earnings : 
  (total_games - non_working_games) * price_per_game = 30 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l4104_410477


namespace NUMINAMATH_CALUDE_shaded_area_circle_with_inscribed_square_l4104_410445

/-- The area of the shaded region in a circle with radius 2, where the unshaded region forms an inscribed square -/
theorem shaded_area_circle_with_inscribed_square :
  let circle_radius : ℝ := 2
  let circle_area := π * circle_radius^2
  let inscribed_square_side := 2 * circle_radius
  let inscribed_square_area := inscribed_square_side^2
  let unshaded_area := inscribed_square_area / 2
  let shaded_area := circle_area - unshaded_area
  shaded_area = 4 * π - 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circle_with_inscribed_square_l4104_410445


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l4104_410405

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l4104_410405


namespace NUMINAMATH_CALUDE_chocolate_box_count_l4104_410402

def chocolate_problem (total_bars : ℕ) : Prop :=
  let bar_cost : ℕ := 3
  let unsold_bars : ℕ := 4
  let revenue : ℕ := 9
  (total_bars - unsold_bars) * bar_cost = revenue

theorem chocolate_box_count : ∃ (n : ℕ), chocolate_problem n ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_count_l4104_410402


namespace NUMINAMATH_CALUDE_pizza_both_toppings_l4104_410418

/-- Represents a pizza with cheese and olive toppings -/
structure Pizza where
  total_slices : ℕ
  cheese_slices : ℕ
  olive_slices : ℕ
  both_toppings : ℕ

/-- Theorem: Given the conditions, prove that 7 slices have both cheese and olives -/
theorem pizza_both_toppings (p : Pizza) 
  (h1 : p.total_slices = 24)
  (h2 : p.cheese_slices = 15)
  (h3 : p.olive_slices = 16)
  (h4 : p.total_slices = p.both_toppings + (p.cheese_slices - p.both_toppings) + (p.olive_slices - p.both_toppings)) :
  p.both_toppings = 7 := by
  sorry

#check pizza_both_toppings

end NUMINAMATH_CALUDE_pizza_both_toppings_l4104_410418


namespace NUMINAMATH_CALUDE_fibonacci_polynomial_property_l4104_410472

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the theorem
theorem fibonacci_polynomial_property (p : ℝ → ℝ) :
  (∀ k ∈ Finset.range 991, p (k + 992) = fib (k + 992)) →
  p 1983 = fib 1083 - 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_polynomial_property_l4104_410472


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4104_410400

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4104_410400


namespace NUMINAMATH_CALUDE_permutation_combination_sum_l4104_410403

/-- Permutation of n elements taken r at a time -/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then Nat.factorial n / Nat.factorial (n - r) else 0

/-- Combination of n elements taken r at a time -/
def combination (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)) else 0

theorem permutation_combination_sum : 3 * (permutation 3 2) + 2 * (combination 4 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_sum_l4104_410403


namespace NUMINAMATH_CALUDE_ship_distance_constant_l4104_410439

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircular path -/
structure SemicircularPath where
  center : Point
  radius : ℝ

/-- Represents the ship's journey -/
structure ShipJourney where
  path1 : SemicircularPath
  path2 : SemicircularPath

/-- Represents the ship's position along its journey -/
structure ShipPosition where
  t : ℝ  -- Time parameter (0 ≤ t ≤ 2)
  isOnFirstPath : Bool

/-- Distance function for the ship's position -/
def distance (journey : ShipJourney) (pos : ShipPosition) : ℝ :=
  if pos.isOnFirstPath then journey.path1.radius else journey.path2.radius

theorem ship_distance_constant (journey : ShipJourney) :
  ∀ t1 t2 : ℝ, 0 ≤ t1 ∧ t1 ≤ 1 → 0 ≤ t2 ∧ t2 ≤ 1 →
    distance journey { t := t1, isOnFirstPath := true } =
    distance journey { t := t2, isOnFirstPath := true } ∧
  ∀ t3 t4 : ℝ, 1 < t3 ∧ t3 ≤ 2 → 1 < t4 ∧ t4 ≤ 2 →
    distance journey { t := t3, isOnFirstPath := false } =
    distance journey { t := t4, isOnFirstPath := false } ∧
  journey.path1.radius ≠ journey.path2.radius →
    ∃ t5 t6 : ℝ, 0 ≤ t5 ∧ t5 ≤ 1 ∧ 1 < t6 ∧ t6 ≤ 2 ∧
      distance journey { t := t5, isOnFirstPath := true } ≠
      distance journey { t := t6, isOnFirstPath := false } :=
by
  sorry

end NUMINAMATH_CALUDE_ship_distance_constant_l4104_410439


namespace NUMINAMATH_CALUDE_count_special_numbers_l4104_410444

def is_odd (n : Nat) : Bool := n % 2 = 1

def is_even (n : Nat) : Bool := n % 2 = 0

def digits : List Nat := [1, 2, 3, 4, 5]

def is_valid_number (n : List Nat) : Bool :=
  n.length = 5 ∧ 
  n.toFinset.card = 5 ∧
  n.all (λ d => d ∈ digits) ∧
  (∃ i, i ∈ [1, 2, 3] ∧ 
    is_odd (n.get! i) ∧ 
    is_even (n.get! (i-1)) ∧ 
    is_even (n.get! (i+1)))

theorem count_special_numbers :
  (List.filter is_valid_number (List.permutations digits)).length = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l4104_410444


namespace NUMINAMATH_CALUDE_probability_of_a_l4104_410431

theorem probability_of_a (p_a p_b : ℝ) (h_pb : p_b = 2/5)
  (h_independent : p_a * p_b = 0.22857142857142856) :
  p_a = 0.5714285714285714 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_a_l4104_410431


namespace NUMINAMATH_CALUDE_absolute_value_equals_negation_implies_nonpositive_l4104_410469

theorem absolute_value_equals_negation_implies_nonpositive (a : ℝ) :
  |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negation_implies_nonpositive_l4104_410469


namespace NUMINAMATH_CALUDE_underdog_wins_probability_l4104_410413

def best_of_five_probability (p : ℚ) : ℚ :=
  (p^5) + 5 * (p^4) * (1 - p) + 10 * (p^3) * ((1 - p)^2)

theorem underdog_wins_probability :
  best_of_five_probability (1/3) = 17/81 := by
  sorry

end NUMINAMATH_CALUDE_underdog_wins_probability_l4104_410413


namespace NUMINAMATH_CALUDE_select_captains_l4104_410471

theorem select_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_select_captains_l4104_410471


namespace NUMINAMATH_CALUDE_center_of_mass_theorem_l4104_410470

/-- Represents the center of mass coordinates for an n × n chessboard -/
structure CenterOfMass where
  x : ℚ
  y : ℚ

/-- Calculates the center of mass for the sum-1 rule -/
def centerOfMassSumRule (n : ℕ) : CenterOfMass :=
  { x := ((n + 1) * (7 * n - 1)) / (12 * n),
    y := ((n + 1) * (7 * n - 1)) / (12 * n) }

/-- Calculates the center of mass for the product rule -/
def centerOfMassProductRule (n : ℕ) : CenterOfMass :=
  { x := (2 * n + 1) / 3,
    y := (2 * n + 1) / 3 }

/-- Theorem stating the correctness of the center of mass calculations -/
theorem center_of_mass_theorem (n : ℕ) :
  (centerOfMassSumRule n).x = ((n + 1) * (7 * n - 1)) / (12 * n) ∧
  (centerOfMassSumRule n).y = ((n + 1) * (7 * n - 1)) / (12 * n) ∧
  (centerOfMassProductRule n).x = (2 * n + 1) / 3 ∧
  (centerOfMassProductRule n).y = (2 * n + 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_center_of_mass_theorem_l4104_410470


namespace NUMINAMATH_CALUDE_sequence_properties_l4104_410436

/-- Geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- Arithmetic sequence with positive common difference -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ ∀ n, b (n + 1) = b n + d

theorem sequence_properties (a b : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_arith : arithmetic_sequence b)
  (h_eq3 : a 3 = b 3)
  (h_eq7 : a 7 = b 7) :
  a 5 < b 5 ∧ a 1 > b 1 ∧ a 9 > b 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l4104_410436


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l4104_410417

/-- The remaining volume of a cube after removing two perpendicular cylindrical sections. -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) 
  (h_cube_side : cube_side = 6)
  (h_cylinder_radius : cylinder_radius = 1) :
  cube_side ^ 3 - 2 * π * cylinder_radius ^ 2 * cube_side = 216 - 12 * π := by
  sorry

#check remaining_cube_volume

end NUMINAMATH_CALUDE_remaining_cube_volume_l4104_410417
