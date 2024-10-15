import Mathlib

namespace NUMINAMATH_CALUDE_all_calculations_incorrect_l3925_392554

theorem all_calculations_incorrect : 
  (-|-3| ≠ 3) ∧ 
  (∀ a b : ℝ, (a + b)^2 ≠ a^2 + b^2) ∧ 
  (∀ a : ℝ, a ≠ 0 → a^3 * a^4 ≠ a^12) ∧ 
  (|-3^2| ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_all_calculations_incorrect_l3925_392554


namespace NUMINAMATH_CALUDE_x_equals_three_l3925_392511

theorem x_equals_three (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_three_l3925_392511


namespace NUMINAMATH_CALUDE_diamond_six_three_l3925_392594

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem diamond_six_three : diamond 6 3 = 18 := by sorry

end NUMINAMATH_CALUDE_diamond_six_three_l3925_392594


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3925_392575

/-- Triangle ABC with given vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from a point to a line -/
def altitude (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem about the triangle ABC -/
theorem triangle_abc_properties :
  let t : Triangle := { A := (-2, 4), B := (-3, -1), C := (1, 3) }
  let alt_B_AC : ℝ → ℝ := altitude t.B (fun x => x - 1)  -- Line AC: y = x - 1
  ∀ x y, alt_B_AC x = y ↔ x + y - 2 = 0 ∧ triangleArea t = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3925_392575


namespace NUMINAMATH_CALUDE_f_less_than_g_for_n_ge_5_l3925_392503

theorem f_less_than_g_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : n^2 + n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_f_less_than_g_for_n_ge_5_l3925_392503


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3925_392583

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * Real.pi * r ^ 2 = 16 * Real.pi) →
  (4 / 3 * Real.pi * r ^ 3 = 32 / 3 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3925_392583


namespace NUMINAMATH_CALUDE_product_of_specific_integers_l3925_392559

theorem product_of_specific_integers : 
  ∃ (a b : ℤ), 
    a = 32 ∧ 
    b = 3125 ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    a * b = 100000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_integers_l3925_392559


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l3925_392588

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l3925_392588


namespace NUMINAMATH_CALUDE_polynomial_identity_l3925_392539

theorem polynomial_identity : ∀ x : ℝ, 
  (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3925_392539


namespace NUMINAMATH_CALUDE_certain_number_value_l3925_392520

theorem certain_number_value (x p n : ℕ) (h1 : x > 0) (h2 : Prime p) 
  (h3 : ∃ k : ℕ, Prime k ∧ Even k ∧ x = k * n * p) (h4 : x ≥ 44) 
  (h5 : ∀ y, y > 0 → y < x → ¬∃ k : ℕ, Prime k ∧ Even k ∧ y = k * n * p) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l3925_392520


namespace NUMINAMATH_CALUDE_number_calculation_l3925_392586

theorem number_calculation (n : ℝ) : 
  (0.20 * 0.45 * 0.60 * 0.75 * n = 283.5) → n = 7000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3925_392586


namespace NUMINAMATH_CALUDE_trigonometric_ratio_proof_trigonometric_expression_simplification_l3925_392542

theorem trigonometric_ratio_proof (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

theorem trigonometric_expression_simplification (α : Real) :
  (Real.sin (π/2 + α) * Real.cos (5*π/2 - α) * Real.tan (-π + α)) /
  (Real.tan (7*π - α) * Real.sin (π + α)) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_proof_trigonometric_expression_simplification_l3925_392542


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_350_by_175_percent_l3925_392521

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (percentage / 100) * initial = initial * (1 + percentage / 100) := by sorry

theorem increase_350_by_175_percent :
  350 + (175 / 100) * 350 = 962.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_350_by_175_percent_l3925_392521


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3925_392510

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ k : ℕ, k < 9 → ¬(11 ∣ (11002 + k))) ∧ (11 ∣ (11002 + 9)) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3925_392510


namespace NUMINAMATH_CALUDE_johns_arcade_spending_l3925_392527

theorem johns_arcade_spending (total_allowance : ℚ) 
  (remaining_after_toy_store : ℚ) (toy_store_fraction : ℚ) 
  (h1 : total_allowance = 9/4)
  (h2 : remaining_after_toy_store = 3/5)
  (h3 : toy_store_fraction = 1/3) : 
  ∃ (arcade_fraction : ℚ), 
    arcade_fraction = 3/5 ∧ 
    remaining_after_toy_store = (1 - arcade_fraction) * total_allowance * (1 - toy_store_fraction) :=
by sorry

end NUMINAMATH_CALUDE_johns_arcade_spending_l3925_392527


namespace NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_one_mod_seventeen_l3925_392578

theorem smallest_five_digit_negative_congruent_to_one_mod_seventeen :
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 17] → n ≥ -10011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_one_mod_seventeen_l3925_392578


namespace NUMINAMATH_CALUDE_jihye_wallet_money_l3925_392529

/-- The total amount of money in Jihye's wallet -/
def total_money (note_value : ℕ) (note_count : ℕ) (coin_value : ℕ) : ℕ :=
  note_value * note_count + coin_value

/-- Theorem stating the total amount of money in Jihye's wallet -/
theorem jihye_wallet_money : total_money 1000 2 560 = 2560 := by
  sorry

end NUMINAMATH_CALUDE_jihye_wallet_money_l3925_392529


namespace NUMINAMATH_CALUDE_num_lines_eq_60_l3925_392515

def coefficients : Finset ℕ := {1, 3, 5, 7, 9}

/-- The number of different lines formed by the equation Ax + By + C = 0,
    where A, B, and C are distinct elements from the set {1, 3, 5, 7, 9} -/
def num_lines : ℕ :=
  (coefficients.card) * (coefficients.card - 1) * (coefficients.card - 2)

theorem num_lines_eq_60 : num_lines = 60 := by
  sorry

end NUMINAMATH_CALUDE_num_lines_eq_60_l3925_392515


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3925_392517

def digits : Finset Nat := {1, 4, 7, 9}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

def largest_number : Nat := 97
def smallest_number : Nat := 14

theorem two_digit_number_difference :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number - smallest_number = 83 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3925_392517


namespace NUMINAMATH_CALUDE_simplify_fraction_l3925_392518

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * y) / (-(x^2 * y)) = -2 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3925_392518


namespace NUMINAMATH_CALUDE_gcd_g_x_l3925_392597

def g (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(7*x+4)^2*(8*x+5)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 360 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 120 := by
sorry

end NUMINAMATH_CALUDE_gcd_g_x_l3925_392597


namespace NUMINAMATH_CALUDE_projection_area_eq_projection_length_l3925_392502

/-- A cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- A plane onto which the cube is projected -/
class ProjectionPlane

/-- A line perpendicular to the projection plane -/
class PerpendicularLine (P : ProjectionPlane)

/-- The area of the projection of a cube onto a plane -/
noncomputable def projection_area (cube : UnitCube) (P : ProjectionPlane) : ℝ :=
  sorry

/-- The length of the projection of a cube onto a line perpendicular to the projection plane -/
noncomputable def projection_length (cube : UnitCube) (P : ProjectionPlane) (L : PerpendicularLine P) : ℝ :=
  sorry

/-- Theorem stating that the area of the projection of a unit cube onto a plane
    is equal to the length of its projection onto a perpendicular line -/
theorem projection_area_eq_projection_length
  (cube : UnitCube) (P : ProjectionPlane) (L : PerpendicularLine P) :
  projection_area cube P = projection_length cube P L :=
sorry

end NUMINAMATH_CALUDE_projection_area_eq_projection_length_l3925_392502


namespace NUMINAMATH_CALUDE_smallest_prime_sum_l3925_392596

def digit_set : Set Nat := {1, 2, 3, 5}

def is_valid_prime (p : Nat) (used_digits : Set Nat) : Prop :=
  Nat.Prime p ∧ 
  (p % 10) ∈ digit_set ∧
  (p % 10) ∉ used_digits ∧
  (∀ d ∈ digit_set, d ≠ p % 10 → ¬ (∃ k, p / 10^k % 10 = d))

def valid_prime_triple (p q r : Nat) : Prop :=
  is_valid_prime p ∅ ∧
  is_valid_prime q {p % 10} ∧
  is_valid_prime r {p % 10, q % 10}

theorem smallest_prime_sum :
  ∀ p q r, valid_prime_triple p q r → p + q + r ≥ 71 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_l3925_392596


namespace NUMINAMATH_CALUDE_f_is_linear_l3925_392555

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that f(x) = mx + b for all x ∈ ℝ -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- The function f(x) = -x -/
def f : ℝ → ℝ := fun x ↦ -x

/-- Theorem: The function f(x) = -x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_f_is_linear_l3925_392555


namespace NUMINAMATH_CALUDE_circle_equation_l3925_392580

/-- A circle with center on y = 3x and tangent to x-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 3 * center.1
  tangent_to_x_axis : center.2 = radius

/-- The line 2x + y - 10 = 0 -/
def intercepting_line (x y : ℝ) : Prop := 2 * x + y - 10 = 0

/-- The chord intercepted by the line has length 4 -/
def chord_length (c : TangentCircle) : ℝ := 4

theorem circle_equation (c : TangentCircle) 
  (h : ∃ (x y : ℝ), intercepting_line x y ∧ 
       ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
       ((x - c.center.1)^2 + (y - c.center.2)^2 = (chord_length c / 2)^2)) :
  ((c.center.1 = 1 ∧ c.center.2 = 3 ∧ c.radius = 3) ∨
   (c.center.1 = -6 ∧ c.center.2 = -18 ∧ c.radius = 18)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3925_392580


namespace NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l3925_392508

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- A point (x,y) is inside the ellipse if the left side of the equation is negative -/
def inside_ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 < 0

theorem origin_inside_ellipse_iff_k_range :
  ∀ k : ℝ, inside_ellipse k 0 0 ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l3925_392508


namespace NUMINAMATH_CALUDE_converse_of_proposition_l3925_392504

theorem converse_of_proposition (a b : ℝ) : 
  (∀ x y : ℝ, x ≥ y → x^3 ≥ y^3) → 
  (∀ x y : ℝ, x^3 ≥ y^3 → x ≥ y) :=
sorry

end NUMINAMATH_CALUDE_converse_of_proposition_l3925_392504


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3925_392592

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
theorem distance_to_y_axis (A : ℝ × ℝ) : 
  A.1 = -3 → A.2 = 4 → |A.1| = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3925_392592


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3925_392543

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27) ≤ 1 / (6*Real.sqrt 3 + 6) :=
sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27) = 1 / (6*Real.sqrt 3 + 6) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3925_392543


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l3925_392519

/-- The distance between the center of a circle and a point in polar coordinates -/
theorem distance_circle_center_to_point 
  (ρ : ℝ → ℝ) -- Radius function for the circle
  (θ : ℝ) -- Angle parameter
  (r : ℝ) -- Radius of point D
  (φ : ℝ) -- Angle of point D
  (h1 : ∀ θ, ρ θ = 2 * Real.sin θ) -- Circle equation
  (h2 : r = 1) -- Radius of point D
  (h3 : φ = Real.pi) -- Angle of point D
  : Real.sqrt 2 = Real.sqrt ((0 - r * Real.cos φ)^2 + (1 - r * Real.sin φ)^2) :=
sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l3925_392519


namespace NUMINAMATH_CALUDE_integer_quotient_problem_l3925_392505

theorem integer_quotient_problem (x y : ℤ) :
  1996 * x + y / 96 = x + y →
  x / y = 1 / 2016 ∨ y / x = 2016 := by
sorry

end NUMINAMATH_CALUDE_integer_quotient_problem_l3925_392505


namespace NUMINAMATH_CALUDE_largest_integral_y_l3925_392553

theorem largest_integral_y : ∃ y : ℤ, y = 4 ∧ 
  (∀ z : ℤ, (1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 7/11 → z ≤ y) ∧
  (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11 :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_y_l3925_392553


namespace NUMINAMATH_CALUDE_popcorn_probability_l3925_392591

theorem popcorn_probability (white_ratio : ℚ) (yellow_ratio : ℚ) 
  (white_pop_prob : ℚ) (yellow_pop_prob : ℚ) :
  white_ratio = 3/4 →
  yellow_ratio = 1/4 →
  white_pop_prob = 1/3 →
  yellow_pop_prob = 3/4 →
  let white_and_pop := white_ratio * white_pop_prob
  let yellow_and_pop := yellow_ratio * yellow_pop_prob
  let total_pop := white_and_pop + yellow_and_pop
  (white_and_pop / total_pop) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_probability_l3925_392591


namespace NUMINAMATH_CALUDE_apple_purchase_cost_l3925_392500

/-- Represents a purchase option for apples -/
structure AppleOption where
  count : ℕ
  price : ℕ

/-- Calculates the total cost of purchasing apples -/
def totalCost (option1 : AppleOption) (option2 : AppleOption) (count1 : ℕ) (count2 : ℕ) : ℕ :=
  option1.price * count1 + option2.price * count2

/-- Calculates the total number of apples purchased -/
def totalApples (option1 : AppleOption) (option2 : AppleOption) (count1 : ℕ) (count2 : ℕ) : ℕ :=
  option1.count * count1 + option2.count * count2

theorem apple_purchase_cost (option1 : AppleOption) (option2 : AppleOption) :
  option1.count = 4 →
  option1.price = 15 →
  option2.count = 7 →
  option2.price = 25 →
  ∃ (count1 count2 : ℕ),
    count1 = count2 ∧
    totalApples option1 option2 count1 count2 = 28 ∧
    totalCost option1 option2 count1 count2 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_cost_l3925_392500


namespace NUMINAMATH_CALUDE_trig_identity_l3925_392536

theorem trig_identity : 
  (Real.cos (12 * π / 180) - Real.cos (18 * π / 180) * Real.sin (60 * π / 180)) / 
  Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3925_392536


namespace NUMINAMATH_CALUDE_olympic_triathlon_distance_l3925_392531

theorem olympic_triathlon_distance :
  ∀ (cycling running swimming : ℝ),
  cycling = 4 * running →
  swimming = (3 / 80) * cycling →
  running - swimming = 8.5 →
  cycling + running + swimming = 51.5 := by
sorry

end NUMINAMATH_CALUDE_olympic_triathlon_distance_l3925_392531


namespace NUMINAMATH_CALUDE_inverse_g_84_l3925_392567

theorem inverse_g_84 (g : ℝ → ℝ) (h : ∀ x, g x = 3 * x^3 + 3) :
  g 3 = 84 ∧ (∀ y, g y = 84 → y = 3) :=
sorry

end NUMINAMATH_CALUDE_inverse_g_84_l3925_392567


namespace NUMINAMATH_CALUDE_triangle_acute_angled_l3925_392566

theorem triangle_acute_angled (a b c : ℝ) 
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sides_relation : a^4 + b^4 = c^4) : 
  c^2 < a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_acute_angled_l3925_392566


namespace NUMINAMATH_CALUDE_x_plus_y_between_52_and_53_l3925_392522

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  y = 4 * (floor x) + 2 ∧
  y = 5 * (floor (x - 3)) + 7 ∧
  ∀ n : ℤ, x ≠ n

-- Theorem statement
theorem x_plus_y_between_52_and_53 (x y : ℝ) 
  (h : problem_conditions x y) : 
  52 < x + y ∧ x + y < 53 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_between_52_and_53_l3925_392522


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3925_392589

theorem polynomial_expansion (x : ℝ) : 
  (x^3 - 3*x + 3) * (x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3925_392589


namespace NUMINAMATH_CALUDE_sum_of_digits_9_pow_1001_l3925_392590

theorem sum_of_digits_9_pow_1001 : ∃ (n : ℕ), 
  (9^1001 : ℕ) % 100 = n ∧ (n / 10 + n % 10 = 9) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9_pow_1001_l3925_392590


namespace NUMINAMATH_CALUDE_icosahedron_cube_relation_l3925_392558

/-- Given a cube with edge length a and an inscribed icosahedron, 
    m is the length of the line segment connecting two vertices 
    of the icosahedron on a face of the cube -/
def icosahedron_in_cube (a m : ℝ) : Prop :=
  a > 0 ∧ m > 0 ∧ a^2 - a*m - m^2 = 0

/-- Theorem stating the relationship between the cube's edge length 
    and the distance between icosahedron vertices on a face -/
theorem icosahedron_cube_relation {a m : ℝ} 
  (h : icosahedron_in_cube a m) : a^2 - a*m - m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_cube_relation_l3925_392558


namespace NUMINAMATH_CALUDE_great_pyramid_tallest_duration_l3925_392585

/-- Represents the dimensions and historical facts about the Great Pyramid of Giza -/
structure GreatPyramid where
  height : ℕ
  width : ℕ
  year_built : Int
  year_surpassed : Int
  height_above_500 : height = 500 + 20
  width_relation : width = height + 234
  sum_height_width : height + width = 1274
  built_BC : year_built < 0
  surpassed_AD : year_surpassed > 0

/-- Theorem stating the duration for which the Great Pyramid was the tallest structure -/
theorem great_pyramid_tallest_duration (p : GreatPyramid) : 
  p.year_surpassed - p.year_built = 3871 :=
sorry

end NUMINAMATH_CALUDE_great_pyramid_tallest_duration_l3925_392585


namespace NUMINAMATH_CALUDE_triangle_theorem_l3925_392546

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C)
  (h2 : t.b + t.c = Real.sqrt 2 * t.a)
  (h3 : t.a * t.b * Real.sin t.A / 2 = Real.sqrt 3 / 12) : 
  t.A = Real.pi / 3 ∧ t.a = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3925_392546


namespace NUMINAMATH_CALUDE_expression_positivity_l3925_392526

theorem expression_positivity (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) :
  5*x^2 + 5*y^2 + 5*z^2 + 6*x*y - 8*x*z - 8*y*z > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_positivity_l3925_392526


namespace NUMINAMATH_CALUDE_zhong_is_symmetrical_l3925_392509

/-- A Chinese character is represented as a structure with left and right sides -/
structure ChineseCharacter where
  left : String
  right : String

/-- A function to check if a character is symmetrical -/
def isSymmetrical (c : ChineseCharacter) : Prop :=
  c.left = c.right

/-- The Chinese character "中" -/
def zhong : ChineseCharacter :=
  { left := "|", right := "|" }

/-- Theorem stating that "中" is symmetrical -/
theorem zhong_is_symmetrical : isSymmetrical zhong := by
  sorry


end NUMINAMATH_CALUDE_zhong_is_symmetrical_l3925_392509


namespace NUMINAMATH_CALUDE_adam_strawberries_l3925_392563

/-- The number of strawberries Adam had left -/
def strawberries_left : ℕ := 33

/-- The number of strawberries Adam ate -/
def strawberries_eaten : ℕ := 2

/-- The initial number of strawberries Adam picked -/
def initial_strawberries : ℕ := strawberries_left + strawberries_eaten

theorem adam_strawberries : initial_strawberries = 35 := by
  sorry

end NUMINAMATH_CALUDE_adam_strawberries_l3925_392563


namespace NUMINAMATH_CALUDE_candies_to_remove_for_even_distribution_l3925_392535

def total_candies : ℕ := 24
def num_sisters : ℕ := 4

theorem candies_to_remove_for_even_distribution :
  (total_candies % num_sisters = 0) ∧
  (total_candies / num_sisters * num_sisters = total_candies) :=
by sorry

end NUMINAMATH_CALUDE_candies_to_remove_for_even_distribution_l3925_392535


namespace NUMINAMATH_CALUDE_coefficient_sum_l3925_392525

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10 + a₁₁*(x+1)^11) →
  a₁ + a₂ + a₁₁ = 781 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l3925_392525


namespace NUMINAMATH_CALUDE_circle_area_triple_radius_l3925_392577

theorem circle_area_triple_radius (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A := by sorry

end NUMINAMATH_CALUDE_circle_area_triple_radius_l3925_392577


namespace NUMINAMATH_CALUDE_nonnegative_rational_function_l3925_392540

theorem nonnegative_rational_function (x : ℝ) :
  (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_rational_function_l3925_392540


namespace NUMINAMATH_CALUDE_class_composition_l3925_392556

theorem class_composition (num_boys : ℕ) (avg_boys avg_girls avg_class : ℚ) :
  num_boys = 12 →
  avg_boys = 84 →
  avg_girls = 92 →
  avg_class = 86 →
  ∃ (num_girls : ℕ), 
    (num_boys : ℚ) * avg_boys + (num_girls : ℚ) * avg_girls = 
    ((num_boys : ℚ) + (num_girls : ℚ)) * avg_class ∧
    num_girls = 4 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l3925_392556


namespace NUMINAMATH_CALUDE_zaras_estimate_bound_l3925_392587

theorem zaras_estimate_bound (x y ε : ℝ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x - y < ε) 
  (h4 : ε > 0) : 
  (x + 2*ε) - (y - ε) < 2*ε := by
sorry

end NUMINAMATH_CALUDE_zaras_estimate_bound_l3925_392587


namespace NUMINAMATH_CALUDE_equation_solution_l3925_392568

theorem equation_solution :
  ∀ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3925_392568


namespace NUMINAMATH_CALUDE_not_in_fourth_quadrant_l3925_392570

/-- A linear function defined by y = 3x + 2 -/
def linear_function (x : ℝ) : ℝ := 3 * x + 2

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the linear function y = 3x + 2 does not pass through the fourth quadrant -/
theorem not_in_fourth_quadrant :
  ∀ x : ℝ, ¬(fourth_quadrant x (linear_function x)) :=
by sorry

end NUMINAMATH_CALUDE_not_in_fourth_quadrant_l3925_392570


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l3925_392550

/-- Given an angle α whose terminal side passes through the point P(-4m, 3m) where m < 0,
    prove that 2sin(α) + cos(α) = -2/5 -/
theorem angle_terminal_side_point (m : ℝ) (α : ℝ) (h1 : m < 0) 
  (h2 : Real.cos α = 4 * m / (5 * abs m)) (h3 : Real.sin α = 3 * m / (5 * abs m)) :
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l3925_392550


namespace NUMINAMATH_CALUDE_blue_balloons_most_l3925_392541

/-- Represents the color of a balloon -/
inductive BalloonColor
  | Red
  | Blue
  | Yellow

/-- Counts the number of balloons of a given color -/
def count_balloons (color : BalloonColor) : ℕ :=
  match color with
  | BalloonColor.Red => 6
  | BalloonColor.Blue => 12
  | BalloonColor.Yellow => 6

theorem blue_balloons_most : 
  (∀ c : BalloonColor, c ≠ BalloonColor.Blue → count_balloons BalloonColor.Blue > count_balloons c) ∧ 
  count_balloons BalloonColor.Red + count_balloons BalloonColor.Blue + count_balloons BalloonColor.Yellow = 24 ∧
  count_balloons BalloonColor.Blue = count_balloons BalloonColor.Red + 6 ∧
  count_balloons BalloonColor.Red = 24 / 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloons_most_l3925_392541


namespace NUMINAMATH_CALUDE_newborn_count_l3925_392513

theorem newborn_count (total_children : ℕ) (toddlers : ℕ) : 
  total_children = 40 →
  toddlers = 6 →
  total_children = 5 * toddlers + toddlers + (total_children - 5 * toddlers - toddlers) →
  (total_children - 5 * toddlers - toddlers) = 4 :=
by sorry

end NUMINAMATH_CALUDE_newborn_count_l3925_392513


namespace NUMINAMATH_CALUDE_closest_point_on_line_l3925_392564

/-- The point on the line y = 2x - 1 that is closest to (3, 4) is (13/5, 21/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 2 * x - 1 → 
  (x - 3)^2 + (y - 4)^2 ≥ (13/5 - 3)^2 + (21/5 - 4)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l3925_392564


namespace NUMINAMATH_CALUDE_last_round_probability_l3925_392545

/-- A tournament with the given conditions -/
structure Tournament (n : ℕ) where
  num_players : ℕ := 2^(n+1)
  num_rounds : ℕ := n+1
  pairing : Unit  -- Represents the pairing process
  pushover_game : Unit  -- Represents the Pushover game

/-- The probability of two specific players facing each other in the last round -/
def face_probability (t : Tournament n) : ℚ :=
  (2^n - 1) / 8^n

/-- Theorem stating the probability of players 1 and 2^n facing each other in the last round -/
theorem last_round_probability (n : ℕ) (h : n > 0) :
  ∀ (t : Tournament n), face_probability t = (2^n - 1) / 8^n :=
sorry

end NUMINAMATH_CALUDE_last_round_probability_l3925_392545


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l3925_392506

theorem cube_root_of_eight (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l3925_392506


namespace NUMINAMATH_CALUDE_age_equation_solution_l3925_392514

/-- Given a person's current age of 50, prove that the equation
    5 * (A + 5) - 5 * (A - X) = A is satisfied when X = 5. -/
theorem age_equation_solution :
  let A : ℕ := 50
  let X : ℕ := 5
  5 * (A + 5) - 5 * (A - X) = A :=
by sorry

end NUMINAMATH_CALUDE_age_equation_solution_l3925_392514


namespace NUMINAMATH_CALUDE_range_of_f_l3925_392573

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≤ -Real.sqrt 2 ∨ y ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3925_392573


namespace NUMINAMATH_CALUDE_quadratic_vertex_value_and_range_l3925_392584

/-- The quadratic function y = ax^2 + 2ax + a -/
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + a

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x (a : ℝ) : ℝ := -1

theorem quadratic_vertex_value_and_range (a : ℝ) :
  f a (vertex_x a) = 0 ∧ f a (vertex_x a) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_value_and_range_l3925_392584


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l3925_392523

-- Define the conditions P and Q
def P (x : ℝ) : Prop := |x - 2| < 3
def Q (x : ℝ) : Prop := x^2 - 8*x + 15 < 0

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) := by
  sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l3925_392523


namespace NUMINAMATH_CALUDE_four_squares_power_of_two_l3925_392537

def count_four_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 0

theorem four_squares_power_of_two (n : ℕ) :
  count_four_squares n = (Nat.card {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a^2 + b^2 + c^2 + d^2 = 2^n}) :=
sorry

end NUMINAMATH_CALUDE_four_squares_power_of_two_l3925_392537


namespace NUMINAMATH_CALUDE_inequality_proof_l3925_392532

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^2/(1+x^2) + y^2/(1+y^2) + z^2/(1+z^2) = 2) : 
  x/(1+x^2) + y/(1+y^2) + z/(1+z^2) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3925_392532


namespace NUMINAMATH_CALUDE_julie_hourly_rate_l3925_392593

/-- Calculates the hourly rate given the following conditions:
  * Hours worked per day
  * Days worked per week
  * Monthly salary when missing one day of work
  * Number of weeks in a month
-/
def calculate_hourly_rate (hours_per_day : ℕ) (days_per_week : ℕ) 
  (monthly_salary_missing_day : ℕ) (weeks_per_month : ℕ) : ℚ :=
  let total_hours := hours_per_day * days_per_week * weeks_per_month - hours_per_day
  monthly_salary_missing_day / total_hours

theorem julie_hourly_rate :
  calculate_hourly_rate 8 6 920 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_julie_hourly_rate_l3925_392593


namespace NUMINAMATH_CALUDE_min_value_product_l3925_392544

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 9) :
  x^4 * y^3 * z^2 ≥ 1/3456 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    1/x₀ + 1/y₀ + 1/z₀ = 9 ∧ 
    x₀^4 * y₀^3 * z₀^2 = 1/3456 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l3925_392544


namespace NUMINAMATH_CALUDE_chess_pieces_arrangement_l3925_392599

theorem chess_pieces_arrangement (total : ℕ) 
  (h1 : ∃ inner : ℕ, total = inner + 60)
  (h2 : ∃ outer : ℕ, 60 = outer + 32) : 
  total = 80 := by
sorry

end NUMINAMATH_CALUDE_chess_pieces_arrangement_l3925_392599


namespace NUMINAMATH_CALUDE_total_signup_combinations_l3925_392507

/-- The number of ways for one person to sign up -/
def signup_options : ℕ := 2

/-- The number of people signing up -/
def num_people : ℕ := 3

/-- Theorem: The total number of different ways for three people to sign up, 
    each with two independent choices, is 8 -/
theorem total_signup_combinations : signup_options ^ num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_signup_combinations_l3925_392507


namespace NUMINAMATH_CALUDE_probability_5_heart_ace_l3925_392561

/-- Represents a standard deck of 52 playing cards. -/
def StandardDeck : ℕ := 52

/-- Represents the number of 5s in a standard deck. -/
def NumberOf5s : ℕ := 4

/-- Represents the number of hearts in a standard deck. -/
def NumberOfHearts : ℕ := 13

/-- Represents the number of Aces in a standard deck. -/
def NumberOfAces : ℕ := 4

/-- Theorem stating the probability of drawing a 5 as the first card, 
    a heart as the second card, and an Ace as the third card from a standard 52-card deck. -/
theorem probability_5_heart_ace : 
  (NumberOf5s : ℚ) / StandardDeck * 
  NumberOfHearts / (StandardDeck - 1) * 
  NumberOfAces / (StandardDeck - 2) = 1 / 650 := by
  sorry

end NUMINAMATH_CALUDE_probability_5_heart_ace_l3925_392561


namespace NUMINAMATH_CALUDE_ball_probability_l3925_392574

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 20)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 37)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3925_392574


namespace NUMINAMATH_CALUDE_real_part_of_inverse_l3925_392530

theorem real_part_of_inverse (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_l3925_392530


namespace NUMINAMATH_CALUDE_blocks_remaining_l3925_392547

theorem blocks_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 78 → used = 19 → remaining = initial - used → remaining = 59 := by
  sorry

end NUMINAMATH_CALUDE_blocks_remaining_l3925_392547


namespace NUMINAMATH_CALUDE_probability_of_event_b_l3925_392565

theorem probability_of_event_b 
  (prob_a : ℝ) 
  (prob_a_and_b : ℝ) 
  (prob_neither_a_nor_b : ℝ) 
  (h1 : prob_a = 0.20)
  (h2 : prob_a_and_b = 0.15)
  (h3 : prob_neither_a_nor_b = 0.5499999999999999) :
  ∃ (prob_b : ℝ), prob_b = 0.40 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_event_b_l3925_392565


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l3925_392549

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 6
  h_b : b = 8
  h_c : c = 10
  h_right : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def longest_altitudes_sum (t : RightTriangle) : ℝ := t.a + t.b

theorem longest_altitudes_sum_is_14 (t : RightTriangle) :
  longest_altitudes_sum t = 14 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l3925_392549


namespace NUMINAMATH_CALUDE_bus_purchase_problem_l3925_392551

-- Define the variables
variable (a b : ℝ)
variable (x : ℝ)  -- Number of A model buses

-- Define the conditions
def total_buses : ℝ := 10
def fuel_savings_A : ℝ := 2.4
def fuel_savings_B : ℝ := 2
def price_difference : ℝ := 2
def model_cost_difference : ℝ := 6
def total_fuel_savings : ℝ := 22.4

-- State the theorem
theorem bus_purchase_problem :
  (a - b = price_difference) →
  (3 * b - 2 * a = model_cost_difference) →
  (fuel_savings_A * x + fuel_savings_B * (total_buses - x) = total_fuel_savings) →
  (a = 120 ∧ b = 100 ∧ x = 6 ∧ a * x + b * (total_buses - x) = 1120) := by
  sorry

end NUMINAMATH_CALUDE_bus_purchase_problem_l3925_392551


namespace NUMINAMATH_CALUDE_budget_theorem_l3925_392533

/-- Represents a budget with three categories in a given ratio -/
structure Budget where
  ratio_1 : ℕ
  ratio_2 : ℕ
  ratio_3 : ℕ
  amount_2 : ℚ

/-- Calculates the total amount allocated in a budget -/
def total_amount (b : Budget) : ℚ :=
  (b.ratio_1 + b.ratio_2 + b.ratio_3) * (b.amount_2 / b.ratio_2)

/-- Theorem stating that for a budget with ratio 5:4:1 and $720 allocated to the second category,
    the total amount is $1800 -/
theorem budget_theorem (b : Budget) 
  (h1 : b.ratio_1 = 5)
  (h2 : b.ratio_2 = 4)
  (h3 : b.ratio_3 = 1)
  (h4 : b.amount_2 = 720) :
  total_amount b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_budget_theorem_l3925_392533


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3925_392581

theorem simultaneous_equations_solution (m : ℝ) :
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3925_392581


namespace NUMINAMATH_CALUDE_candle_burn_time_l3925_392557

/-- Proves that given a candle that lasts 8 nights when burned for 1 hour per night, 
    if 6 candles are used over 24 nights, then the average burn time per night is 2 hours. -/
theorem candle_burn_time 
  (candle_duration : ℕ) 
  (burn_time_per_night : ℕ) 
  (num_candles : ℕ) 
  (total_nights : ℕ) 
  (h1 : candle_duration = 8)
  (h2 : burn_time_per_night = 1)
  (h3 : num_candles = 6)
  (h4 : total_nights = 24) :
  (candle_duration * burn_time_per_night * num_candles) / total_nights = 2 := by
  sorry

end NUMINAMATH_CALUDE_candle_burn_time_l3925_392557


namespace NUMINAMATH_CALUDE_camping_items_l3925_392571

theorem camping_items (total_items : ℕ) 
  (tent_stakes : ℕ) 
  (drink_mix : ℕ) 
  (water_bottles : ℕ) 
  (food_cans : ℕ) : 
  total_items = 32 → 
  drink_mix = 2 * tent_stakes → 
  water_bottles = tent_stakes + 2 → 
  food_cans * 2 = tent_stakes → 
  tent_stakes + drink_mix + water_bottles + food_cans = total_items → 
  tent_stakes = 6 := by
sorry

end NUMINAMATH_CALUDE_camping_items_l3925_392571


namespace NUMINAMATH_CALUDE_max_soccer_balls_buyable_l3925_392528

/-- The cost of 6 soccer balls in yuan -/
def cost_of_six_balls : ℕ := 168

/-- The number of balls in a set -/
def balls_in_set : ℕ := 6

/-- The amount of money available to spend in yuan -/
def available_money : ℕ := 500

/-- The maximum number of soccer balls that can be bought -/
def max_balls_bought : ℕ := 17

theorem max_soccer_balls_buyable :
  (cost_of_six_balls * max_balls_bought) / balls_in_set ≤ available_money ∧
  (cost_of_six_balls * (max_balls_bought + 1)) / balls_in_set > available_money :=
by sorry

end NUMINAMATH_CALUDE_max_soccer_balls_buyable_l3925_392528


namespace NUMINAMATH_CALUDE_theater_occupancy_l3925_392560

theorem theater_occupancy (total_chairs : ℕ) (total_people : ℕ) : 
  (3 * total_people = 5 * (4 * total_chairs / 5)) →  -- Three-fifths of people occupy four-fifths of chairs
  (total_chairs - (4 * total_chairs / 5) = 5) →      -- 5 chairs are empty
  (total_people = 33) :=                             -- Total people is 33
by
  sorry

#check theater_occupancy

end NUMINAMATH_CALUDE_theater_occupancy_l3925_392560


namespace NUMINAMATH_CALUDE_equal_radii_of_intersecting_triangles_l3925_392534

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  vertices : Fin 3 → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Configuration of two intersecting triangles -/
structure IntersectingTriangles where
  triangle1 : TriangleWithInscribedCircle
  triangle2 : TriangleWithInscribedCircle
  smallTriangles : Fin 6 → TriangleWithInscribedCircle
  hexagon : Set (ℝ × ℝ)

/-- The theorem stating that the radii of the inscribed circles of the two original triangles are equal -/
theorem equal_radii_of_intersecting_triangles (config : IntersectingTriangles) 
  (h : ∀ i j : Fin 6, (config.smallTriangles i).radius = (config.smallTriangles j).radius) :
  config.triangle1.radius = config.triangle2.radius :=
sorry

end NUMINAMATH_CALUDE_equal_radii_of_intersecting_triangles_l3925_392534


namespace NUMINAMATH_CALUDE_fraction_calculation_l3925_392598

theorem fraction_calculation : (2 / 8 : ℚ) + (4 / 16 : ℚ) * (3 / 9 : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3925_392598


namespace NUMINAMATH_CALUDE_gross_profit_percentage_l3925_392501

theorem gross_profit_percentage (sales_price gross_profit : ℝ) 
  (h1 : sales_price = 91)
  (h2 : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 := by
sorry

end NUMINAMATH_CALUDE_gross_profit_percentage_l3925_392501


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3925_392516

theorem simplify_and_evaluate (x y : ℤ) (A B : ℤ) (h1 : A = 2*x + y) (h2 : B = 2*x - y) (h3 : x = -1) (h4 : y = 2) :
  (A^2 - B^2) * (x - 2*y) = 80 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3925_392516


namespace NUMINAMATH_CALUDE_equation_solution_l3925_392512

theorem equation_solution : ∃! x : ℚ, (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3925_392512


namespace NUMINAMATH_CALUDE_sine_inequality_l3925_392552

theorem sine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 (Real.pi / 2) → 
  y ∈ Set.Icc 0 Real.pi → 
  Real.sin (x + y) ≥ Real.sin x - Real.sin y := by
sorry

end NUMINAMATH_CALUDE_sine_inequality_l3925_392552


namespace NUMINAMATH_CALUDE_taxi_speed_taxi_speed_is_45_l3925_392548

/-- The speed of a taxi that overtakes a bus under specific conditions. -/
theorem taxi_speed : ℝ → Prop :=
  fun v =>
    (∀ (bus_distance : ℝ),
      bus_distance = 4 * (v - 30) →  -- Distance covered by bus in 4 hours
      bus_distance + 2 * (v - 30) = 2 * v) →  -- Taxi covers bus distance in 2 hours
    v = 45

/-- Proof of the taxi speed theorem. -/
theorem taxi_speed_is_45 : taxi_speed 45 := by
  sorry

end NUMINAMATH_CALUDE_taxi_speed_taxi_speed_is_45_l3925_392548


namespace NUMINAMATH_CALUDE_work_completion_time_l3925_392569

/-- Given that two workers A and B can complete a work together in a certain number of days,
    and worker A can complete the work alone in a certain number of days,
    this function calculates the number of days worker B needs to complete the work alone. -/
def days_for_b_alone (days_together : ℚ) (days_a_alone : ℚ) : ℚ :=
  (days_together * days_a_alone) / (days_a_alone - days_together)

/-- Theorem stating that if A and B together can complete a work in 4 days,
    and A alone can complete the same work in 12 days,
    then B alone can complete the work in 6 days. -/
theorem work_completion_time :
  days_for_b_alone 4 12 = 6 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l3925_392569


namespace NUMINAMATH_CALUDE_hannah_easter_eggs_l3925_392538

theorem hannah_easter_eggs (total : ℕ) (h : total = 63) :
  ∃ (helen : ℕ) (hannah : ℕ),
    hannah = 2 * helen ∧
    hannah + helen = total ∧
    hannah = 42 := by
  sorry

end NUMINAMATH_CALUDE_hannah_easter_eggs_l3925_392538


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3925_392582

theorem tan_alpha_plus_pi_fourth (x y : ℝ) (α : ℝ) : 
  (x < 0 ∧ y > 0) →  -- terminal side in second quadrant
  (3 * x + 4 * y = 0) →  -- m ⊥ OA
  (Real.tan α = -3/4) →  -- derived from m ⊥ OA
  Real.tan (α + π/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3925_392582


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l3925_392579

-- Define set A
def A : Set ℝ := {x | (x + 2) / (x - 3) < 0}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l3925_392579


namespace NUMINAMATH_CALUDE_log_cos_acute_angle_l3925_392524

theorem log_cos_acute_angle (A m n : ℝ) : 
  0 < A → A < π/2 →
  Real.log (1 + Real.sin A) = m →
  Real.log (1 / (1 - Real.sin A)) = n →
  Real.log (Real.cos A) = (1/2) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_log_cos_acute_angle_l3925_392524


namespace NUMINAMATH_CALUDE_difference_y_coordinates_l3925_392576

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 4, n + k) on this line,
    the value of k is 2. -/
theorem difference_y_coordinates (m n k : ℝ) : 
  (m = 2*n + 5) → (m + 4 = 2*(n + k) + 5) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_y_coordinates_l3925_392576


namespace NUMINAMATH_CALUDE_valid_triples_are_solutions_l3925_392562

def is_valid_triple (x y z : ℕ+) : Prop :=
  ∃ (n : ℤ), (Real.sqrt (2005 / (x + y : ℝ)) + 
              Real.sqrt (2005 / (y + z : ℝ)) + 
              Real.sqrt (2005 / (z + x : ℝ))) = n

def is_solution_triple (x y z : ℕ+) : Prop :=
  (x = 2005 * 2 ∧ y = 2005 * 2 ∧ z = 2005 * 14) ∨
  (x = 2005 * 2 ∧ y = 2005 * 14 ∧ z = 2005 * 2) ∨
  (x = 2005 * 14 ∧ y = 2005 * 2 ∧ z = 2005 * 2)

theorem valid_triples_are_solutions (x y z : ℕ+) :
  is_valid_triple x y z ↔ is_solution_triple x y z := by
  sorry

end NUMINAMATH_CALUDE_valid_triples_are_solutions_l3925_392562


namespace NUMINAMATH_CALUDE_train_length_l3925_392595

/-- Calculates the length of a train given the time it takes to cross a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) :
  bridge_length = 1500 →
  bridge_time = 70 →
  post_time = 20 →
  ∃ (train_length : ℝ),
    train_length / post_time = (train_length + bridge_length) / bridge_time ∧
    train_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3925_392595


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l3925_392572

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w u : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hv : M.mulVec v = ![2, 6])
  (hw : M.mulVec w = ![3, -5])
  (hu : M.mulVec u = ![-1, 4]) :
  M.mulVec (2 • v - w + 4 • u) = ![-3, 33] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l3925_392572
