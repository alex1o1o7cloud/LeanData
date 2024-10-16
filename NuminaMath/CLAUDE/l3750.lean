import Mathlib

namespace NUMINAMATH_CALUDE_solution_y_initial_weight_l3750_375038

/-- Proves that the initial weight of solution Y is 8 kg given the problem conditions --/
theorem solution_y_initial_weight :
  ∀ (W : ℝ),
  (W > 0) →
  (0.20 * W = W * 0.20) →
  (0.25 * W = 0.20 * W + 0.4) →
  W = 8 := by
sorry

end NUMINAMATH_CALUDE_solution_y_initial_weight_l3750_375038


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l3750_375081

theorem fraction_sum_proof (x A B : ℚ) : 
  (5*x - 11) / (2*x^2 + x - 6) = A / (x + 2) + B / (2*x - 3) → 
  A = 3 ∧ B = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l3750_375081


namespace NUMINAMATH_CALUDE_smallest_z_for_cube_equation_l3750_375040

theorem smallest_z_for_cube_equation : 
  (∃ (w x y z : ℕ), 
    w < x ∧ x < y ∧ y < z ∧
    w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
    w^3 + x^3 + y^3 = 2 * z^3) ∧
  (∀ (w x y z : ℕ),
    w < x → x < y → y < z →
    w + 1 = x → x + 1 = y → y + 1 = z →
    w^3 + x^3 + y^3 = 2 * z^3 →
    z ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_for_cube_equation_l3750_375040


namespace NUMINAMATH_CALUDE_charlie_horns_l3750_375077

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- The problem statement -/
theorem charlie_horns (charlie carli : Instruments) : 
  charlie.flutes = 1 →
  charlie.harps = 1 →
  carli.flutes = 2 * charlie.flutes →
  carli.horns = charlie.horns / 2 →
  carli.harps = 0 →
  charlie.flutes + charlie.horns + charlie.harps + 
    carli.flutes + carli.horns + carli.harps = 7 →
  charlie.horns = 2 := by
  sorry

#check charlie_horns

end NUMINAMATH_CALUDE_charlie_horns_l3750_375077


namespace NUMINAMATH_CALUDE_gcd_18_30_45_l3750_375024

theorem gcd_18_30_45 : Nat.gcd 18 (Nat.gcd 30 45) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_45_l3750_375024


namespace NUMINAMATH_CALUDE_fifteen_consecutive_naturals_l3750_375062

theorem fifteen_consecutive_naturals (N : ℕ) : 
  (N < 81 ∧ 
   ∀ k : ℕ, (N < k ∧ k < 81) → (k - N ≤ 15)) ∧ 
  (∃ m : ℕ, N < m ∧ m < 81 ∧ m - N = 15) →
  N = 66 := by
sorry

end NUMINAMATH_CALUDE_fifteen_consecutive_naturals_l3750_375062


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3750_375071

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 1) :
  1/x + 1/y + 1/z ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3750_375071


namespace NUMINAMATH_CALUDE_rainwater_farm_chickens_l3750_375085

/-- Represents the number of animals on Mr. Rainwater's farm -/
structure FarmAnimals where
  cows : Nat
  goats : Nat
  chickens : Nat
  ducks : Nat

/-- Defines the conditions for Mr. Rainwater's farm -/
def valid_farm (f : FarmAnimals) : Prop :=
  f.cows = 9 ∧
  f.goats = 4 * f.cows ∧
  f.goats = 2 * f.chickens ∧
  f.ducks = (3 * f.chickens) / 2 ∧
  (f.ducks - 2 * f.chickens) % 3 = 0 ∧
  f.goats + f.chickens + f.ducks ≤ 100

theorem rainwater_farm_chickens :
  ∀ f : FarmAnimals, valid_farm f → f.chickens = 18 :=
sorry

end NUMINAMATH_CALUDE_rainwater_farm_chickens_l3750_375085


namespace NUMINAMATH_CALUDE_arrangements_count_l3750_375042

def number_of_people : ℕ := 7
def number_of_gaps : ℕ := number_of_people - 1

theorem arrangements_count :
  (number_of_people - 2).factorial * number_of_gaps.choose 2 = 3600 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l3750_375042


namespace NUMINAMATH_CALUDE_circle_center_travel_distance_l3750_375033

-- Define the triangle
def triangle_sides : (ℝ × ℝ × ℝ) := (5, 12, 13)

-- Define the circle radius
def circle_radius : ℝ := 2

-- Define the function to calculate the perimeter of the inscribed triangle
def inscribed_triangle_perimeter (sides : ℝ × ℝ × ℝ) (radius : ℝ) : ℝ :=
  let (a, b, c) := sides
  (a - 2 * radius) + (b - 2 * radius) + (c - 2 * radius)

-- Theorem statement
theorem circle_center_travel_distance :
  inscribed_triangle_perimeter triangle_sides circle_radius = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_travel_distance_l3750_375033


namespace NUMINAMATH_CALUDE_tan_beta_value_l3750_375078

theorem tan_beta_value (α β : ℝ) 
  (h1 : Real.tan (π - α) = -(1 / 5))
  (h2 : Real.tan (α - β) = 1 / 3) : 
  Real.tan β = -(1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3750_375078


namespace NUMINAMATH_CALUDE_distance_to_line_l3750_375017

/-- Given a line l with slope k passing through point A(0,2), and a normal vector n to l,
    prove that for any point B satisfying |n⋅AB| = |n|, the distance from B to l is 1. -/
theorem distance_to_line (k : ℝ) (n : ℝ × ℝ) (B : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 2)
  let l := {(x, y) : ℝ × ℝ | y - 2 = k * x}
  n.1 = -k ∧ n.2 = 1 →  -- n is a normal vector to l
  |n.1 * (B.1 - A.1) + n.2 * (B.2 - A.2)| = Real.sqrt (n.1^2 + n.2^2) →
  Real.sqrt ((B.1 - 0)^2 + (B.2 - (k * B.1 + 2))^2) / Real.sqrt (1 + k^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_line_l3750_375017


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3750_375070

/-- An isosceles triangle with congruent sides of length 8 and perimeter 25 has a base of length 9. -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 8
    let perimeter := 25
    (2 * congruent_side + base = perimeter) →
    base = 9

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3750_375070


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l3750_375058

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l3750_375058


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l3750_375037

theorem coconut_grove_problem (x : ℝ) 
  (h1 : (x + 2) * 40 + x * 120 + (x - 2) * 180 = 100 * (3 * x)) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l3750_375037


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l3750_375010

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

/-- Theorem: The number of diagonals in a convex pentadecagon is 90 -/
theorem pentadecagon_diagonals : 
  num_diagonals pentadecagon_sides = 90 := by sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l3750_375010


namespace NUMINAMATH_CALUDE_normas_cards_l3750_375051

/-- Proves that Norma's total number of cards is 158.0 given the initial and found amounts -/
theorem normas_cards (initial_cards : Real) (found_cards : Real) 
  (h1 : initial_cards = 88.0) 
  (h2 : found_cards = 70.0) : 
  initial_cards + found_cards = 158.0 := by
sorry

end NUMINAMATH_CALUDE_normas_cards_l3750_375051


namespace NUMINAMATH_CALUDE_expand_product_l3750_375029

theorem expand_product (x : ℝ) : (5*x + 3) * (2*x^2 + 4) = 10*x^3 + 6*x^2 + 20*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3750_375029


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l3750_375096

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The main theorem -/
theorem unique_four_digit_square : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ 
  is_perfect_square n ∧
  is_perfect_square (reverse n) ∧
  is_perfect_square (n / reverse n) ∧
  n = 9801 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l3750_375096


namespace NUMINAMATH_CALUDE_sum_2012_terms_eq_4021_l3750_375004

/-- A sequence where each term (after the second) is the sum of its previous and next terms -/
def SpecialSequence (a₀ a₁ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | n + 2 => SpecialSequence a₀ a₁ (n + 1) + SpecialSequence a₀ a₁ n

/-- The sum of the first n terms of the special sequence -/
def SequenceSum (a₀ a₁ : ℤ) (n : ℕ) : ℤ :=
  (List.range n).map (SpecialSequence a₀ a₁) |>.sum

theorem sum_2012_terms_eq_4021 :
  SequenceSum 2010 2011 2012 = 4021 := by
  sorry

end NUMINAMATH_CALUDE_sum_2012_terms_eq_4021_l3750_375004


namespace NUMINAMATH_CALUDE_expression_simplification_l3750_375046

theorem expression_simplification :
  ((3 + 4 + 5 + 7) / 3) + ((3 * 6 + 9) / 4) = 157 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3750_375046


namespace NUMINAMATH_CALUDE_problem_solution_l3750_375079

def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem problem_solution (a b : ℝ) : 
  M ∪ N a b = Set.univ ∧ M ∩ N a b = Set.Ioc 2009 2010 → a = -2009 ∧ b = -2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3750_375079


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3750_375086

-- Define the side lengths of the isosceles triangle
def side_a : ℝ := 9
def side_b : ℝ := sorry  -- This will be either 3 or 5

-- Define the equation for side_b
axiom side_b_equation : side_b^2 - 8*side_b + 15 = 0

-- Define the perimeter of the triangle
def perimeter : ℝ := 2*side_a + side_b

-- Theorem statement
theorem isosceles_triangle_perimeter :
  perimeter = 19 ∨ perimeter = 21 ∨ perimeter = 23 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3750_375086


namespace NUMINAMATH_CALUDE_sum_of_facing_angles_l3750_375043

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ
  is_isosceles : vertex_angle > 0 ∧ vertex_angle < 180

-- Define the configuration of two isosceles triangles
structure TwoTrianglesConfig where
  triangle1 : IsoscelesTriangle
  triangle2 : IsoscelesTriangle
  distance : ℝ
  same_base_line : Bool
  facing_equal_sides : Bool

-- Theorem statement
theorem sum_of_facing_angles (config : TwoTrianglesConfig) :
  config.triangle1 = config.triangle2 →
  config.triangle1.vertex_angle = 40 →
  config.distance = 4 →
  config.same_base_line = true →
  config.facing_equal_sides = true →
  (180 - config.triangle1.vertex_angle) + (180 - config.triangle2.vertex_angle) = 80 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_facing_angles_l3750_375043


namespace NUMINAMATH_CALUDE_garment_fraction_theorem_l3750_375098

theorem garment_fraction_theorem (bikini_fraction trunks_fraction : ℝ) 
  (h1 : bikini_fraction = 0.38)
  (h2 : trunks_fraction = 0.25) : 
  bikini_fraction + trunks_fraction = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_garment_fraction_theorem_l3750_375098


namespace NUMINAMATH_CALUDE_kyle_car_payment_l3750_375055

def monthly_income : ℝ := 3200

def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def gas_maintenance : ℝ := 350

def other_expenses : ℝ := rent + utilities + retirement_savings + groceries + insurance + miscellaneous + gas_maintenance

def car_payment : ℝ := monthly_income - other_expenses

theorem kyle_car_payment :
  car_payment = 350 := by sorry

end NUMINAMATH_CALUDE_kyle_car_payment_l3750_375055


namespace NUMINAMATH_CALUDE_team_selection_count_l3750_375027

/-- The number of ways to select a team of 6 people from a group of 7 boys and 9 girls, with at least 2 boys -/
def selectTeam (boys girls : ℕ) : ℕ := 
  (Nat.choose boys 2 * Nat.choose girls 4) +
  (Nat.choose boys 3 * Nat.choose girls 3) +
  (Nat.choose boys 4 * Nat.choose girls 2) +
  (Nat.choose boys 5 * Nat.choose girls 1) +
  (Nat.choose boys 6 * Nat.choose girls 0)

/-- Theorem stating that the number of ways to select the team is 7042 -/
theorem team_selection_count : selectTeam 7 9 = 7042 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l3750_375027


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3750_375028

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 36 = 0) ↔ m = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3750_375028


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l3750_375001

/-- Given a curve y = x^2 - 3x, if there exists a point where the tangent line
    has a slope of 1, then the x-coordinate of this point is 2. -/
theorem tangent_point_x_coordinate (x : ℝ) : 
  (∃ y : ℝ, y = x^2 - 3*x ∧ (deriv (fun x => x^2 - 3*x)) x = 1) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l3750_375001


namespace NUMINAMATH_CALUDE_lines_always_parallel_l3750_375006

/-- A linear function f(x) = kx + b -/
def f (k b x : ℝ) : ℝ := k * x + b

/-- Line l₁ represented by y = f(x) -/
def l₁ (k b : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f k b x}

/-- Line l₂ defined as y - y₀ = f(x) - f(x₀) -/
def l₂ (k b x₀ y₀ : ℝ) : Set (ℝ × ℝ) := {(x, y) | y - y₀ = f k b x - f k b x₀}

/-- Point P -/
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

theorem lines_always_parallel (k b x₀ y₀ : ℝ) 
  (h : P x₀ y₀ ∉ l₁ k b) : 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    ((x, y) ∈ l₁ k b ↔ y = k * x + m) ∧ 
    ((x, y) ∈ l₂ k b x₀ y₀ ↔ y = k * x + (y₀ - k * x₀)) :=
sorry

end NUMINAMATH_CALUDE_lines_always_parallel_l3750_375006


namespace NUMINAMATH_CALUDE_F_symmetry_l3750_375076

def F (a b c d : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x^5 + c * x^3 + d * x - 6

theorem F_symmetry (a b c d : ℝ) :
  F a b c d (-2) = 10 → F a b c d 2 = -22 := by sorry

end NUMINAMATH_CALUDE_F_symmetry_l3750_375076


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3750_375036

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (b : ℝ) (x : ℝ) : Prop := x < 1 ∨ x > b

-- Define the new quadratic function
def g (a b c : ℝ) (x : ℝ) : ℝ := x^2 - b * (a + c) * x + 4 * c

theorem quadratic_inequality_theorem (a b : ℝ) :
  (∀ x, f a x > 0 ↔ solution_set b x) →
  (a = 1 ∧ b = 2) ∧
  (∀ c x, 
    (c > 1 → (g a b c x > 0 ↔ x < 2 ∨ x > 2 * c)) ∧
    (c = 1 → (g a b c x > 0 ↔ x ≠ 2)) ∧
    (c < 1 → (g a b c x > 0 ↔ x > 2 ∨ x < 2 * c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3750_375036


namespace NUMINAMATH_CALUDE_soccer_ball_purchase_l3750_375015

theorem soccer_ball_purchase (first_batch_cost second_batch_cost : ℕ) 
  (unit_price_difference : ℕ) :
  first_batch_cost = 800 →
  second_batch_cost = 1560 →
  unit_price_difference = 2 →
  ∃ (first_batch_quantity second_batch_quantity : ℕ) 
    (first_unit_price second_unit_price : ℕ),
    first_batch_quantity * first_unit_price = first_batch_cost ∧
    second_batch_quantity * second_unit_price = second_batch_cost ∧
    second_batch_quantity = 2 * first_batch_quantity ∧
    first_unit_price = second_unit_price + unit_price_difference ∧
    first_batch_quantity + second_batch_quantity = 30 :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_purchase_l3750_375015


namespace NUMINAMATH_CALUDE_rectangle_width_calculation_l3750_375061

theorem rectangle_width_calculation (big_length : ℝ) (small_area : ℝ) :
  big_length = 40 →
  small_area = 200 →
  ∃ (big_width : ℝ),
    big_width = 20 ∧
    small_area = (big_length / 2) * (big_width / 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_calculation_l3750_375061


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3750_375063

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 64 * π → 2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3750_375063


namespace NUMINAMATH_CALUDE_inscribed_radius_theorem_l3750_375035

/-- A configuration of spheres in a triangular pyramid -/
structure SpherePyramid where
  /-- The number of spheres in the pyramid -/
  num_spheres : ℕ
  /-- The radius of each sphere in the pyramid -/
  sphere_radius : ℝ
  /-- The radius of the sphere circumscribing the pyramid -/
  circumscribing_radius : ℝ
  /-- The radius of the sphere inscribed at the center of the pyramid -/
  inscribed_radius : ℝ
  /-- Each sphere touches at least three others -/
  touches_at_least_three : Prop
  /-- The inscribed sphere touches six identical spheres -/
  inscribed_touches_six : Prop
  /-- The number of spheres is exactly ten -/
  sphere_count : num_spheres = 10
  /-- The circumscribing radius is 5√2 + 5 -/
  circumscribing_radius_value : circumscribing_radius = 5 * Real.sqrt 2 + 5

/-- Theorem stating the relationship between the inscribed and circumscribing radii -/
theorem inscribed_radius_theorem (p : SpherePyramid) : 
  p.inscribed_radius = Real.sqrt 6 - 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_radius_theorem_l3750_375035


namespace NUMINAMATH_CALUDE_remainder_theorem_l3750_375026

theorem remainder_theorem (P Q Q' R R' a b c : ℕ) 
  (h1 : P = a * Q + R) 
  (h2 : Q = (b + c) * Q' + R') : 
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3750_375026


namespace NUMINAMATH_CALUDE_distance_between_points_l3750_375050

theorem distance_between_points : Real.sqrt ((0 - 6)^2 + (18 - 0)^2) = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3750_375050


namespace NUMINAMATH_CALUDE_skateboard_price_after_discounts_l3750_375069

/-- Calculates the final price of an item after two consecutive percentage discounts -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem: The final price of a $150 skateboard after 40% and 25% discounts is $67.50 -/
theorem skateboard_price_after_discounts :
  final_price 150 0.4 0.25 = 67.5 := by
  sorry

#eval final_price 150 0.4 0.25

end NUMINAMATH_CALUDE_skateboard_price_after_discounts_l3750_375069


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l3750_375097

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ((perimeter_inches + 11) / 12 : ℕ)

/-- Theorem stating that for a 4x6 inch picture, quadrupled and with a 3-inch border, 9 feet of framing is needed. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l3750_375097


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l3750_375041

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_derivative_at_negative_one (a b : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l3750_375041


namespace NUMINAMATH_CALUDE_opposite_of_three_l3750_375025

theorem opposite_of_three : (-(3 : ℤ)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3750_375025


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3750_375039

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (6 * x) / (2 * y + z) + (3 * y) / (x + 2 * z) + (9 * z) / (x + y) ≥ 83 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
    (6 * x) / (2 * y + z) + (3 * y) / (x + 2 * z) + (9 * z) / (x + y) < 83 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3750_375039


namespace NUMINAMATH_CALUDE_emma_money_theorem_l3750_375030

def emma_money_problem (initial_amount furniture_cost fraction_to_anna : ℚ) : Prop :=
  let remaining_after_furniture := initial_amount - furniture_cost
  let amount_to_anna := fraction_to_anna * remaining_after_furniture
  let final_amount := remaining_after_furniture - amount_to_anna
  final_amount = 400

theorem emma_money_theorem :
  emma_money_problem 2000 400 (3/4) := by
  sorry

end NUMINAMATH_CALUDE_emma_money_theorem_l3750_375030


namespace NUMINAMATH_CALUDE_triangle_circles_theorem_l3750_375032

/-- Represents a triangular arrangement of circles -/
structure TriangularArrangement where
  total_circles : ℕ
  longest_side_length : ℕ
  shorter_side_rows : List ℕ

/-- Calculates the number of ways to choose three consecutive circles along the longest side -/
def longest_side_choices (arr : TriangularArrangement) : ℕ :=
  (arr.longest_side_length * (arr.longest_side_length + 1)) / 2

/-- Calculates the number of ways to choose three consecutive circles along a shorter side -/
def shorter_side_choices (arr : TriangularArrangement) : ℕ :=
  arr.shorter_side_rows.sum

/-- Calculates the total number of ways to choose three consecutive circles in any direction -/
def total_choices (arr : TriangularArrangement) : ℕ :=
  longest_side_choices arr + 2 * shorter_side_choices arr

/-- The main theorem stating that for the given arrangement, there are 57 ways to choose three consecutive circles -/
theorem triangle_circles_theorem (arr : TriangularArrangement) 
  (h1 : arr.total_circles = 33)
  (h2 : arr.longest_side_length = 6)
  (h3 : arr.shorter_side_rows = [4, 4, 4, 3, 2, 1]) :
  total_choices arr = 57 := by
  sorry


end NUMINAMATH_CALUDE_triangle_circles_theorem_l3750_375032


namespace NUMINAMATH_CALUDE_additional_oil_purchased_l3750_375022

/-- Proves that a 30% price reduction allows purchasing 9 more kgs of oil with a budget of 900 Rs. --/
theorem additional_oil_purchased (budget : ℝ) (reduced_price : ℝ) (reduction_percentage : ℝ) : 
  budget = 900 →
  reduced_price = 30 →
  reduction_percentage = 0.3 →
  ⌊budget / reduced_price - budget / (reduced_price / (1 - reduction_percentage))⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_additional_oil_purchased_l3750_375022


namespace NUMINAMATH_CALUDE_sector_arc_length_l3750_375074

-- Define the sector
def Sector (area : ℝ) (angle : ℝ) : Type :=
  {r : ℝ // area = (1/2) * r^2 * angle}

-- Define the theorem
theorem sector_arc_length 
  (s : Sector 4 2) : 
  s.val * 2 = 4 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3750_375074


namespace NUMINAMATH_CALUDE_average_price_is_52_cents_l3750_375023

/-- Represents the fruit selection problem -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  oranges_removed : ℕ

/-- Calculates the average price of fruits kept -/
def average_price_kept (fs : FruitSelection) : ℚ :=
  sorry

/-- Theorem stating the average price of fruits kept is 52 cents -/
theorem average_price_is_52_cents (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 20)
  (h4 : fs.initial_avg_price = 56/100)
  (h5 : fs.oranges_removed = 10) :
  average_price_kept fs = 52/100 :=
by sorry

end NUMINAMATH_CALUDE_average_price_is_52_cents_l3750_375023


namespace NUMINAMATH_CALUDE_quadratic_root_sum_cubes_equals_sum_l3750_375048

theorem quadratic_root_sum_cubes_equals_sum (k : ℚ) : 
  (∃ a b : ℚ, (4 * a^2 + 5 * a + k = 0) ∧ 
               (4 * b^2 + 5 * b + k = 0) ∧ 
               (a ≠ b) ∧
               (a^3 + b^3 = a + b)) ↔ 
  (k = 9/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_cubes_equals_sum_l3750_375048


namespace NUMINAMATH_CALUDE_seth_candy_bars_l3750_375059

theorem seth_candy_bars (max_candy_bars : ℕ) (seth_candy_bars : ℕ) : 
  max_candy_bars = 24 →
  seth_candy_bars = 3 * max_candy_bars + 6 →
  seth_candy_bars = 78 :=
by sorry

end NUMINAMATH_CALUDE_seth_candy_bars_l3750_375059


namespace NUMINAMATH_CALUDE_expression_value_l3750_375083

theorem expression_value (x y : ℝ) 
  (eq1 : x - y = -2)
  (eq2 : 2 * x + y = -1) :
  (x - y)^2 - (x - 2*y) * (x + 2*y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3750_375083


namespace NUMINAMATH_CALUDE_runner_problem_l3750_375005

theorem runner_problem (v : ℝ) (h : v > 0) :
  (40 / v = 20 / v + 8) → (40 / (v / 2) = 16) := by
  sorry

end NUMINAMATH_CALUDE_runner_problem_l3750_375005


namespace NUMINAMATH_CALUDE_log7_10_approximation_l3750_375053

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_5_approx : ℝ := 0.699

-- Define a tolerance for approximation
def tolerance : ℝ := 0.001

-- Theorem statement
theorem log7_10_approximation :
  let log10_7 := log10_5_approx + log10_2_approx
  abs (Real.log 10 / Real.log 7 - 33 / 10) < tolerance := by
  sorry

end NUMINAMATH_CALUDE_log7_10_approximation_l3750_375053


namespace NUMINAMATH_CALUDE_triangle_theorem_l3750_375095

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def triangleCondition (t : Triangle) : Prop :=
  t.b * (Real.sin (t.C / 2))^2 + t.c * (Real.sin (t.B / 2))^2 = t.a / 2

theorem triangle_theorem (t : Triangle) (h : triangleCondition t) :
  (t.b + t.c = 2 * t.a) ∧ (t.A ≤ Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3750_375095


namespace NUMINAMATH_CALUDE_complex_additive_inverse_l3750_375094

theorem complex_additive_inverse (m : ℝ) : 
  let z : ℂ := (1 - m * I) / (1 - 2 * I)
  (∃ (a : ℝ), z = a - a * I) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_additive_inverse_l3750_375094


namespace NUMINAMATH_CALUDE_unique_grid_solution_l3750_375082

/-- Represents a 3x3 grid with some fixed values and variables A, B, C, D -/
structure Grid :=
  (A B C D : ℕ)

/-- Checks if two numbers are adjacent in the grid -/
def adjacent (x y : ℕ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 5) ∨
  (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 5) ∨ (x = 4 ∧ y = 7) ∨ (x = 5 ∧ y = 6) ∨
  (x = 5 ∧ y = 8) ∨ (x = 6 ∧ y = 9) ∨ (x = 7 ∧ y = 8) ∨ (x = 8 ∧ y = 9) ∨
  (y = 1 ∧ x = 2) ∨ (y = 1 ∧ x = 4) ∨ (y = 2 ∧ x = 3) ∨ (y = 2 ∧ x = 5) ∨
  (y = 3 ∧ x = 6) ∨ (y = 4 ∧ x = 5) ∨ (y = 4 ∧ x = 7) ∨ (y = 5 ∧ x = 6) ∨
  (y = 5 ∧ x = 8) ∨ (y = 6 ∧ x = 9) ∨ (y = 7 ∧ x = 8) ∨ (y = 8 ∧ x = 9)

/-- The main theorem to prove -/
theorem unique_grid_solution :
  ∀ (g : Grid),
    (g.A ≠ 1 ∧ g.A ≠ 3 ∧ g.A ≠ 5 ∧ g.A ≠ 7 ∧ g.A ≠ 9) →
    (g.B ≠ 1 ∧ g.B ≠ 3 ∧ g.B ≠ 5 ∧ g.B ≠ 7 ∧ g.B ≠ 9) →
    (g.C ≠ 1 ∧ g.C ≠ 3 ∧ g.C ≠ 5 ∧ g.C ≠ 7 ∧ g.C ≠ 9) →
    (g.D ≠ 1 ∧ g.D ≠ 3 ∧ g.D ≠ 5 ∧ g.D ≠ 7 ∧ g.D ≠ 9) →
    (∀ (x y : ℕ), adjacent x y → x + y < 12) →
    (g.A = 8 ∧ g.B = 6 ∧ g.C = 4 ∧ g.D = 2) :=
by sorry


end NUMINAMATH_CALUDE_unique_grid_solution_l3750_375082


namespace NUMINAMATH_CALUDE_tangent_inequality_l3750_375008

theorem tangent_inequality (α β : Real) 
  (h1 : 0 < α) (h2 : α ≤ π/4) (h3 : 0 < β) (h4 : β ≤ π/4) : 
  Real.sqrt (Real.tan α * Real.tan β) ≤ Real.tan ((α + β)/2) ∧ 
  Real.tan ((α + β)/2) ≤ (Real.tan α + Real.tan β)/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_inequality_l3750_375008


namespace NUMINAMATH_CALUDE_fiftieth_term_l3750_375073

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem fiftieth_term (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 50 = 99 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_l3750_375073


namespace NUMINAMATH_CALUDE_open_box_volume_l3750_375018

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length sheet_width cut_size : ℕ)
  (h1 : sheet_length = 40)
  (h2 : sheet_width = 30)
  (h3 : cut_size = 8)
  : (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 2688 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l3750_375018


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l3750_375084

/-- An arithmetic sequence of natural numbers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + (n - 1) * d

theorem arithmetic_contains_geometric (a d : ℕ) (h : d > 0) :
  ∃ (r : ℚ) (f : ℕ → ℕ), 
    (∀ n, f n < f (n + 1)) ∧ 
    (∀ n, ArithmeticSequence a d (f n) * r = ArithmeticSequence a d (f (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l3750_375084


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3750_375075

theorem square_difference_theorem (x y : ℚ) 
  (h1 : x + 2 * y = 5 / 9) 
  (h2 : x - 2 * y = 1 / 9) : 
  x^2 - 4 * y^2 = 5 / 81 := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3750_375075


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_is_two_l3750_375099

theorem sum_of_x_coordinates_is_two :
  let f (x : ℝ) := |x^2 - 4*x + 3|
  let g (x : ℝ) := 7 - 2*x
  ∃ (x₁ x₂ : ℝ), (f x₁ = g x₁) ∧ (f x₂ = g x₂) ∧ (x₁ + x₂ = 2) ∧
    (∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_is_two_l3750_375099


namespace NUMINAMATH_CALUDE_composite_surface_area_is_39_l3750_375016

/-- The surface area of a composite object formed by three cylinders -/
def composite_surface_area (π : ℝ) (h : ℝ) (r₁ r₂ r₃ : ℝ) : ℝ :=
  (2 * π * r₁ * h + π * r₁^2) +
  (2 * π * r₂ * h + π * r₂^2) +
  (2 * π * r₃ * h + π * r₃^2) +
  π * r₁^2 + π * r₂^2 + π * r₃^2

/-- The surface area of the composite object is 39 square meters -/
theorem composite_surface_area_is_39 :
  composite_surface_area 3 1 1.5 1 0.5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_composite_surface_area_is_39_l3750_375016


namespace NUMINAMATH_CALUDE_simplify_expression_l3750_375091

theorem simplify_expression (a b : ℝ) :
  (15*a + 45*b) + (21*a + 32*b) - (12*a + 40*b) = 24*a + 37*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3750_375091


namespace NUMINAMATH_CALUDE_saras_quarters_l3750_375045

theorem saras_quarters (current_quarters borrowed_quarters : ℕ) 
  (h1 : current_quarters = 512)
  (h2 : borrowed_quarters = 271) :
  current_quarters + borrowed_quarters = 783 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l3750_375045


namespace NUMINAMATH_CALUDE_absolute_difference_inequality_l3750_375090

theorem absolute_difference_inequality (x : ℝ) :
  |2*x - 4| - |3*x + 9| < 1 ↔ x < -3 ∨ x > -6/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_l3750_375090


namespace NUMINAMATH_CALUDE_exchange_10_dollars_equals_1200_yen_l3750_375054

/-- The exchange rate from US dollars to Japanese yen -/
def exchange_rate : ℝ := 120

/-- The amount of US dollars to be exchanged -/
def dollars_to_exchange : ℝ := 10

/-- The function that calculates the amount of yen received for a given amount of dollars -/
def exchange (dollars : ℝ) : ℝ := dollars * exchange_rate

theorem exchange_10_dollars_equals_1200_yen :
  exchange dollars_to_exchange = 1200 := by
  sorry

end NUMINAMATH_CALUDE_exchange_10_dollars_equals_1200_yen_l3750_375054


namespace NUMINAMATH_CALUDE_min_both_composers_l3750_375067

theorem min_both_composers (total : ℕ) (mozart : ℕ) (beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : mozart = 95)
  (h_beethoven : beethoven = 80)
  : ∃ (both : ℕ), both ≥ mozart + beethoven - total ∧ both = 40 :=
sorry

end NUMINAMATH_CALUDE_min_both_composers_l3750_375067


namespace NUMINAMATH_CALUDE_quartic_real_root_condition_l3750_375088

theorem quartic_real_root_condition (p q : ℝ) :
  (∃ x : ℝ, x^4 + p * x^2 + q = 0) →
  p^2 ≥ 4 * q ∧
  ¬(∀ p q : ℝ, p^2 ≥ 4 * q → ∃ x : ℝ, x^4 + p * x^2 + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quartic_real_root_condition_l3750_375088


namespace NUMINAMATH_CALUDE_fraction_simplification_l3750_375064

theorem fraction_simplification (b y : ℝ) :
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 + 2*y^2) = 
  2*b^2 / (b^2 + 2*y^2)^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3750_375064


namespace NUMINAMATH_CALUDE_relationship_abc_l3750_375021

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.6 0.4)
  (hb : b = Real.rpow 0.4 0.6)
  (hc : c = Real.rpow 0.4 0.4) :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3750_375021


namespace NUMINAMATH_CALUDE_investment_calculation_l3750_375047

/-- Calculates the investment amount given share details and dividend received -/
theorem investment_calculation (face_value premium dividend_rate total_dividend : ℚ) : 
  face_value = 100 →
  premium = 20 / 100 →
  dividend_rate = 5 / 100 →
  total_dividend = 600 →
  (total_dividend / (face_value * dividend_rate)) * (face_value * (1 + premium)) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l3750_375047


namespace NUMINAMATH_CALUDE_systematic_sampling_questionnaire_C_l3750_375020

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_C (total_population : ℕ) 
  (sample_size : ℕ) (first_number : ℕ) : 
  total_population = 960 →
  sample_size = 32 →
  first_number = 9 →
  (960 - 750) / (960 / 32) = 7 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_questionnaire_C_l3750_375020


namespace NUMINAMATH_CALUDE_largest_n_inequality_l3750_375034

theorem largest_n_inequality : ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 < (3/2 : ℚ) ↔ n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_largest_n_inequality_l3750_375034


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3750_375013

theorem unique_integer_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x)^2 - x = 2016 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3750_375013


namespace NUMINAMATH_CALUDE_sum_double_factorial_divisible_l3750_375072

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

theorem sum_double_factorial_divisible :
  (double_factorial 1985 + double_factorial 1986) % 1987 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_double_factorial_divisible_l3750_375072


namespace NUMINAMATH_CALUDE_train_length_l3750_375012

/-- Given a train that crosses two platforms of different lengths at different times, 
    this theorem proves the length of the train. -/
theorem train_length 
  (platform1_length : ℝ) 
  (platform1_time : ℝ) 
  (platform2_length : ℝ) 
  (platform2_time : ℝ) 
  (h1 : platform1_length = 120)
  (h2 : platform1_time = 15)
  (h3 : platform2_length = 250)
  (h4 : platform2_time = 20) :
  ∃ train_length : ℝ, 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧ 
    train_length = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3750_375012


namespace NUMINAMATH_CALUDE_problem_solution_l3750_375052

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 3|

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 5) ↔ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3750_375052


namespace NUMINAMATH_CALUDE_cosine_A_in_special_triangle_l3750_375011

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem cosine_A_in_special_triangle (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)  -- Sum of angles in a triangle
  (h2 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)  -- Positive side lengths
  (h3 : Real.sin t.A / 4 = Real.sin t.B / 5)  -- Given ratio
  (h4 : Real.sin t.B / 5 = Real.sin t.C / 6)  -- Given ratio
  : Real.cos t.A = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_A_in_special_triangle_l3750_375011


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3750_375089

theorem inequality_solution_set : 
  ¬(∀ x : ℝ, -3 * x > 9 ↔ x < -3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3750_375089


namespace NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l3750_375007

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ  -- Vertex
  F : ℝ × ℝ  -- Focus

/-- A point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : sorry  -- Condition for the point to be on the parabola

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_vertex_distance 
  (p : Parabola) 
  (A : ParabolaPoint p) 
  (h1 : distance A.point p.F = 18) 
  (h2 : distance A.point p.V = 19) : 
  distance p.F p.V = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l3750_375007


namespace NUMINAMATH_CALUDE_exists_four_digit_with_eleven_multiple_permutation_l3750_375087

/-- A permutation of the digits of a number -/
def isDigitPermutation (a b : ℕ) : Prop := sorry

/-- Check if a number is between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem exists_four_digit_with_eleven_multiple_permutation :
  ∃ n : ℕ, isFourDigit n ∧ ∃ m : ℕ, isDigitPermutation n m ∧ m % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_four_digit_with_eleven_multiple_permutation_l3750_375087


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3750_375049

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  (x - 2)^2 - (x - 3) * (x + 3) = -4 * x + 13 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 + 2*x) / (x^2 - 1) / (x + 1 + (2*x + 1) / (x - 1)) = 1 / (x + 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3750_375049


namespace NUMINAMATH_CALUDE_sum_of_binomial_coeffs_l3750_375002

-- Define the binomial coefficient
def binomial_coeff (n m : ℕ) : ℕ := sorry

-- State the combinatorial identity
axiom combinatorial_identity (n m : ℕ) : 
  binomial_coeff (n + 1) m = binomial_coeff n (m - 1) + binomial_coeff n m

-- State the theorem to be proved
theorem sum_of_binomial_coeffs :
  binomial_coeff 7 4 + binomial_coeff 7 5 + binomial_coeff 8 6 = binomial_coeff 9 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binomial_coeffs_l3750_375002


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3750_375031

theorem quadratic_root_relation (p : ℝ) : 
  (∃ a : ℝ, a ≠ 0 ∧ (a^2 + p*a + 18 = 0) ∧ ((2*a)^2 + p*(2*a) + 18 = 0)) ↔ 
  (p = 9 ∨ p = -9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3750_375031


namespace NUMINAMATH_CALUDE_solve_rope_problem_l3750_375066

def rope_problem (x : ℝ) : Prop :=
  let known_ropes := [8, 20, 7]
  let total_ropes := 6
  let knot_loss := 1.2
  let final_length := 35
  let num_knots := total_ropes - 1
  let total_knot_loss := num_knots * knot_loss
  final_length + total_knot_loss = (known_ropes.sum + 3 * x)

theorem solve_rope_problem :
  ∃ x : ℝ, rope_problem x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_solve_rope_problem_l3750_375066


namespace NUMINAMATH_CALUDE_distance_A_l3750_375068

-- Define the points
def A : ℝ × ℝ := (0, 11)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the condition that AA' and BB' intersect at C
def intersect_at_C (A' B' : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    C = (t₁ * A'.1 + (1 - t₁) * A.1, t₁ * A'.2 + (1 - t₁) * A.2) ∧
    C = (t₂ * B'.1 + (1 - t₂) * B.1, t₂ * B'.2 + (1 - t₂) * B.2)

-- Main theorem
theorem distance_A'B'_is_2_26 :
  ∃ A' B' : ℝ × ℝ, 
    line_y_eq_x A' ∧ 
    line_y_eq_x B' ∧ 
    intersect_at_C A' B' ∧ 
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2.26 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_l3750_375068


namespace NUMINAMATH_CALUDE_polynomial_properties_l3750_375014

def f (x : ℝ) : ℝ := 8*x^7 + 5*x^6 + 3*x^4 + 2*x + 1

theorem polynomial_properties :
  (f 2 = 1397) ∧
  (f (-1) = -1) ∧
  (∃ c : ℝ, c ∈ Set.Icc (-1) 2 ∧ f c = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l3750_375014


namespace NUMINAMATH_CALUDE_wine_equation_correct_l3750_375060

/-- Represents the value of clear wine in terms of grain -/
def clear_wine_value : ℝ := 10

/-- Represents the value of turbid wine in terms of grain -/
def turbid_wine_value : ℝ := 3

/-- Represents the total amount of grain available -/
def total_grain : ℝ := 30

/-- Represents the total amount of wine obtained -/
def total_wine : ℝ := 5

/-- Theorem stating that the equation 10x + 3(5-x) = 30 correctly represents
    the relationship between clear wine, turbid wine, and total grain value -/
theorem wine_equation_correct (x : ℝ) :
  0 ≤ x ∧ x ≤ total_wine →
  clear_wine_value * x + turbid_wine_value * (total_wine - x) = total_grain :=
by sorry

end NUMINAMATH_CALUDE_wine_equation_correct_l3750_375060


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3750_375019

/-- A function representing quadratic variation of y with respect to x -/
def quadratic_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem quadratic_symmetry (k : ℝ) :
  quadratic_variation k 5 = 25 →
  quadratic_variation k (-5) = 25 := by
  sorry

#check quadratic_symmetry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3750_375019


namespace NUMINAMATH_CALUDE_square_root_squared_l3750_375056

theorem square_root_squared : (Real.sqrt 930249)^2 = 930249 := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l3750_375056


namespace NUMINAMATH_CALUDE_books_sold_l3750_375003

/-- Given that Tom initially had 5 books, bought 38 new books, and now has 39 books in total,
    prove that the number of books Tom sold is 4. -/
theorem books_sold (initial_books : ℕ) (new_books : ℕ) (total_books : ℕ) (sold_books : ℕ) : 
  initial_books = 5 → new_books = 38 → total_books = 39 → 
  initial_books - sold_books + new_books = total_books →
  sold_books = 4 := by
sorry

end NUMINAMATH_CALUDE_books_sold_l3750_375003


namespace NUMINAMATH_CALUDE_ordering_abc_l3750_375065

theorem ordering_abc (a b c : ℝ) : 
  a = -(5/4) * Real.log (4/5) →
  b = Real.exp (1/4) / 4 →
  c = 1/3 →
  a < b ∧ b < c := by
sorry

end NUMINAMATH_CALUDE_ordering_abc_l3750_375065


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3750_375000

/-- The area of a square with diagonal length 12√2 cm is 144 cm² -/
theorem square_area_from_diagonal : ∀ s : ℝ,
  s > 0 →
  s * s * 2 = (12 * Real.sqrt 2) ^ 2 →
  s * s = 144 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3750_375000


namespace NUMINAMATH_CALUDE_plastic_rings_total_weight_l3750_375044

theorem plastic_rings_total_weight 
  (orange : ℝ) (purple : ℝ) (white : ℝ) (blue : ℝ) (red : ℝ) (green : ℝ)
  (h_orange : orange = 0.08)
  (h_purple : purple = 0.33)
  (h_white : white = 0.42)
  (h_blue : blue = 0.59)
  (h_red : red = 0.24)
  (h_green : green = 0.16) :
  orange + purple + white + blue + red + green = 1.82 := by
sorry

end NUMINAMATH_CALUDE_plastic_rings_total_weight_l3750_375044


namespace NUMINAMATH_CALUDE_difference_of_squares_25_7_l3750_375057

theorem difference_of_squares_25_7 : (25 + 7)^2 - (25 - 7)^2 = 700 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_25_7_l3750_375057


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_four_l3750_375009

def expression (n : ℕ) : ℤ :=
  7 * (n - 3)^4 - n^2 + 12*n - 30

theorem largest_n_multiple_of_four :
  ∀ n : ℕ, n < 100000 →
    (4 ∣ expression n) →
    n ≤ 99999 ∧
    (4 ∣ expression 99999) ∧
    99999 < 100000 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_four_l3750_375009


namespace NUMINAMATH_CALUDE_solution_value_l3750_375092

theorem solution_value (a b : ℝ) : 
  (2 * a + b = 3) → (6 * a + 3 * b - 1 = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3750_375092


namespace NUMINAMATH_CALUDE_complex_number_property_l3750_375093

theorem complex_number_property (b : ℝ) : 
  let z : ℂ := (2 - b * I) / (1 + 2 * I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l3750_375093


namespace NUMINAMATH_CALUDE_jason_pears_count_l3750_375080

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 105 - (47 + 12)

/-- The total number of pears picked -/
def total_pears : ℕ := 105

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 47

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 12

theorem jason_pears_count : jason_pears = 46 := by sorry

end NUMINAMATH_CALUDE_jason_pears_count_l3750_375080
