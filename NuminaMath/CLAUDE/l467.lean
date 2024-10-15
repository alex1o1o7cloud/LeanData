import Mathlib

namespace NUMINAMATH_CALUDE_candy_bar_cost_l467_46792

/-- The cost of a single candy bar given Carl's earnings and purchasing power -/
theorem candy_bar_cost (weekly_earnings : ℚ) (weeks : ℕ) (bars_bought : ℕ) : 
  weekly_earnings = 3/4 ∧ weeks = 4 ∧ bars_bought = 6 → 
  (weekly_earnings * weeks) / bars_bought = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l467_46792


namespace NUMINAMATH_CALUDE_max_trig_function_ratio_l467_46754

/-- Given a function f(x) = 3sin(x) + 4cos(x) that attains its maximum value when x = θ,
    prove that (sin(2θ) + cos²(θ) + 1) / cos(2θ) = 65/7 -/
theorem max_trig_function_ratio (θ : Real) 
    (h : ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 3 * Real.sin θ + 4 * Real.cos θ) :
    (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_trig_function_ratio_l467_46754


namespace NUMINAMATH_CALUDE_equal_coffee_and_milk_consumed_l467_46738

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the drinking and refilling process --/
def drinkAndRefill (contents : CupContents) (amount : ℚ) : CupContents :=
  let remainingCoffee := contents.coffee * (1 - amount)
  let remainingMilk := contents.milk * (1 - amount)
  { coffee := remainingCoffee,
    milk := 1 - remainingCoffee }

/-- The main theorem stating that equal amounts of coffee and milk are consumed --/
theorem equal_coffee_and_milk_consumed :
  let initial := { coffee := 1, milk := 0 }
  let step1 := drinkAndRefill initial (1/6)
  let step2 := drinkAndRefill step1 (1/3)
  let step3 := drinkAndRefill step2 (1/2)
  let finalDrink := 1 - step3.coffee - step3.milk
  1 - initial.coffee + finalDrink = 1 := by sorry

end NUMINAMATH_CALUDE_equal_coffee_and_milk_consumed_l467_46738


namespace NUMINAMATH_CALUDE_set_equality_l467_46703

theorem set_equality (M : Set ℕ) : M ∪ {1} = {1, 2, 3} → M = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l467_46703


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l467_46762

theorem compare_negative_fractions : -3/4 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l467_46762


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l467_46789

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x > 0, x^2 - x ≤ Real.exp x - a*x - 1) →
  a ≤ Real.exp 1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l467_46789


namespace NUMINAMATH_CALUDE_expression_simplification_l467_46765

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ((2 * x - 3) / (x - 2) - 1) / ((x^2 - 2*x + 1) / (x - 2)) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l467_46765


namespace NUMINAMATH_CALUDE_unique_intersection_implies_r_equals_three_l467_46776

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- State the theorem
theorem unique_intersection_implies_r_equals_three 
  (r : ℝ) 
  (h_r_pos : r > 0) 
  (h_unique : ∃! p, p ∈ A ∩ B r) : 
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_r_equals_three_l467_46776


namespace NUMINAMATH_CALUDE_farm_animals_l467_46766

theorem farm_animals (total_legs total_animals : ℕ) 
  (h_legs : total_legs = 38)
  (h_animals : total_animals = 12)
  (h_positive : total_legs > 0 ∧ total_animals > 0) :
  ∃ (chickens sheep : ℕ),
    chickens + sheep = total_animals ∧
    2 * chickens + 4 * sheep = total_legs ∧
    chickens = 5 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l467_46766


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l467_46702

theorem angle_sum_is_pi_over_two 
  (α β γ : Real) 
  (h_sin_α : Real.sin α = 1/3)
  (h_sin_β : Real.sin β = 1/(3*Real.sqrt 11))
  (h_sin_γ : Real.sin γ = 3/Real.sqrt 11)
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_acute_γ : 0 < γ ∧ γ < π/2) :
  α + β + γ = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l467_46702


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l467_46778

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a*b + b*c + a*c = 50) : 
  a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l467_46778


namespace NUMINAMATH_CALUDE_probability_no_brown_is_51_310_l467_46713

def total_balls : ℕ := 32
def brown_balls : ℕ := 14
def non_brown_balls : ℕ := total_balls - brown_balls

def probability_no_brown : ℚ := (Nat.choose non_brown_balls 3 : ℚ) / (Nat.choose total_balls 3 : ℚ)

theorem probability_no_brown_is_51_310 : probability_no_brown = 51 / 310 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_brown_is_51_310_l467_46713


namespace NUMINAMATH_CALUDE_different_remainders_l467_46742

theorem different_remainders (a b c p : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (hp : Nat.Prime p) (hsum : p = a * b + b * c + a * c) : 
  (a ^ 2 % p ≠ b ^ 2 % p ∧ a ^ 2 % p ≠ c ^ 2 % p ∧ b ^ 2 % p ≠ c ^ 2 % p) ∧
  (a ^ 3 % p ≠ b ^ 3 % p ∧ a ^ 3 % p ≠ c ^ 3 % p ∧ b ^ 3 % p ≠ c ^ 3 % p) := by
  sorry

end NUMINAMATH_CALUDE_different_remainders_l467_46742


namespace NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_l467_46747

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m^2 - m - 2 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

-- Theorem for part (I)
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

-- Theorem for part (II)
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_l467_46747


namespace NUMINAMATH_CALUDE_polynomial_simplification_l467_46763

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + r^2 + 5 * r - 4) - (r^3 + 3 * r^2 + 7 * r - 2) = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l467_46763


namespace NUMINAMATH_CALUDE_solid_color_not_yellow_percentage_l467_46783

-- Define the total percentage of marbles
def total_percentage : ℝ := 100

-- Define the percentage of solid color marbles
def solid_color_percentage : ℝ := 90

-- Define the percentage of solid yellow marbles
def solid_yellow_percentage : ℝ := 5

-- Theorem to prove
theorem solid_color_not_yellow_percentage :
  solid_color_percentage - solid_yellow_percentage = 85 := by
  sorry

end NUMINAMATH_CALUDE_solid_color_not_yellow_percentage_l467_46783


namespace NUMINAMATH_CALUDE_triangle_side_length_l467_46751

theorem triangle_side_length (a b c : ℝ) (A : ℝ) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  S = (3 * Real.sqrt 3) / 2 →  -- Area of the triangle
  b + c = 3 * Real.sqrt 3 →  -- Sum of sides b and c
  S = (1 / 2) * b * c * Real.sin A →  -- Area formula
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →  -- Law of cosines
  a = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l467_46751


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l467_46707

/-- In a Cartesian coordinate system, the coordinates of a point (2, -3) with respect to the origin are (2, -3) -/
theorem point_coordinates_wrt_origin :
  let point : ℝ × ℝ := (2, -3)
  point = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l467_46707


namespace NUMINAMATH_CALUDE_max_a_value_l467_46757

-- Define the line equation
def line_equation (m : ℚ) (x : ℚ) : ℚ := m * x + 3

-- Define the condition for not passing through lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℕ, 0 < x → x ≤ 50 → ¬ ∃ y : ℤ, line_equation m x = y

-- Define the theorem
theorem max_a_value : 
  (∀ m : ℚ, 1/2 < m → m < 26/51 → no_lattice_points m) ∧
  ¬(∀ m : ℚ, 1/2 < m → m < 26/51 + 1/10000 → no_lattice_points m) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l467_46757


namespace NUMINAMATH_CALUDE_commute_speed_ratio_l467_46760

/-- Proves that the ratio of speeds for a commuter is 2:1 given specific conditions -/
theorem commute_speed_ratio 
  (distance : ℝ) 
  (total_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : distance = 28) 
  (h2 : total_time = 6) 
  (h3 : return_speed = 14) : 
  return_speed / ((2 * distance) / total_time - return_speed) = 2 := by
  sorry

#check commute_speed_ratio

end NUMINAMATH_CALUDE_commute_speed_ratio_l467_46760


namespace NUMINAMATH_CALUDE_sum_of_k_values_l467_46726

theorem sum_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : a^2 / (1 - b) = k)
  (h_eq2 : b^2 / (1 - c) = k)
  (h_eq3 : c^2 / (1 - a) = k) :
  ∃ k1 k2 : ℝ, k = k1 ∨ k = k2 ∧ k1 + k2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l467_46726


namespace NUMINAMATH_CALUDE_quadratic_sum_l467_46795

/-- A quadratic function passing through (1, 3) and (2, 12) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := λ x ↦ p * x^2 + q * x + r

/-- The theorem stating that p + q + 3r = -5 for the given quadratic function -/
theorem quadratic_sum (p q r : ℝ) : 
  (QuadraticFunction p q r 1 = 3) → 
  (QuadraticFunction p q r 2 = 12) → 
  p + q + 3 * r = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l467_46795


namespace NUMINAMATH_CALUDE_hannahs_quarters_l467_46718

def is_valid_quarter_count (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧
  n % 7 = 3 ∧
  n % 8 = 3

theorem hannahs_quarters :
  ∀ n : ℕ, is_valid_quarter_count n ↔ (n = 171 ∨ n = 339) :=
by sorry

end NUMINAMATH_CALUDE_hannahs_quarters_l467_46718


namespace NUMINAMATH_CALUDE_simplify_expression_l467_46701

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (343 : ℝ) ^ (1/3 : ℝ) = 35 := by
  sorry

-- Additional definitions to match the problem conditions
def condition1 : (625 : ℝ) = 5^4 := by sorry
def condition2 : (343 : ℝ) = 7^3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l467_46701


namespace NUMINAMATH_CALUDE_range_of_a_l467_46720

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.exp x + Real.log a / a > Real.log x / a) ↔ a > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l467_46720


namespace NUMINAMATH_CALUDE_trapezoid_area_l467_46740

-- Define a trapezoid
structure Trapezoid :=
  (smaller_base : ℝ)
  (adjacent_angle : ℝ)
  (diagonal_angle : ℝ)

-- Define the area function for a trapezoid
def area (t : Trapezoid) : ℝ := sorry

-- Theorem statement
theorem trapezoid_area (t : Trapezoid) :
  t.smaller_base = 2 ∧
  t.adjacent_angle = 135 ∧
  t.diagonal_angle = 150 →
  area t = 2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l467_46740


namespace NUMINAMATH_CALUDE_ellipse_properties_l467_46735

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- Given ellipse properties, prove eccentricity and equation -/
theorem ellipse_properties (e : Ellipse) 
  (h1 : e.a = (3/2) * e.b)  -- Ratio of major to minor axis
  (h2 : e.c = 2)            -- Focus at (0, -2)
  : e.c / e.a = Real.sqrt 5 / 3 ∧   -- Eccentricity
    ∀ x y : ℝ, (y^2 / (36/5) + x^2 / (16/5) = 1) ↔ 
    (y^2 / e.b^2 + x^2 / e.a^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l467_46735


namespace NUMINAMATH_CALUDE_inequality_proof_l467_46715

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : a + b + Real.sqrt 2 * c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l467_46715


namespace NUMINAMATH_CALUDE_opposite_numbers_cube_inequality_l467_46724

theorem opposite_numbers_cube_inequality (a b : ℝ) (h1 : a = -b) (h2 : a ≠ 0) : a^3 ≠ b^3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_cube_inequality_l467_46724


namespace NUMINAMATH_CALUDE_polynomial_equation_l467_46739

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 + 1
def g (x : ℝ) : ℝ := -x^4 + 5*x^2 - 4

theorem polynomial_equation :
  (∀ x, f x + g x = 2*x^2 - 3) →
  (∀ x, g x = -x^4 + 5*x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_l467_46739


namespace NUMINAMATH_CALUDE_only_one_correct_proposition_l467_46785

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Line → Prop)
variable (para_line : Line → Line → Prop)
variable (para_line_plane : Line → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (a b c : Line) (α β : Plane)

-- State the theorem
theorem only_one_correct_proposition :
  (¬(∀ (a b c : Line) (α : Plane), 
    subset a α → subset b α → perp c a → perp c b → perp_line_plane c α)) ∧
  (¬(∀ (a b : Line) (α : Plane),
    subset b α → para_line a b → para_line_plane a α)) ∧
  (¬(∀ (a b : Line) (α β : Plane),
    para_line_plane a α → intersect α β b → para_line a b)) ∧
  (∀ (a b : Line) (α : Plane),
    perp_line_plane a α → perp_line_plane b α → para_line a b) ∧
  (¬(∀ (a b c : Line) (α β : Plane),
    ((subset a α → subset b α → perp c a → perp c b → perp_line_plane c α) ∨
     (subset b α → para_line a b → para_line_plane a α) ∨
     (para_line_plane a α → intersect α β b → para_line a b)) ∧
    (perp_line_plane a α → perp_line_plane b α → para_line a b))) :=
by sorry

end NUMINAMATH_CALUDE_only_one_correct_proposition_l467_46785


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l467_46727

/-- Given a parallelogram with adjacent sides of lengths 3s and 4s units forming a 30-degree angle,
    if the area is 18√3 square units, then s = 3^(3/4). -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →  -- Ensuring s is positive for physical meaning
  (3 * s) * (4 * s) * Real.sin (π / 6) = 18 * Real.sqrt 3 →
  s = 3 ^ (3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_side_length_l467_46727


namespace NUMINAMATH_CALUDE_sin_negative_three_pi_fourths_l467_46768

theorem sin_negative_three_pi_fourths :
  Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_three_pi_fourths_l467_46768


namespace NUMINAMATH_CALUDE_smallest_integer_a_l467_46752

theorem smallest_integer_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), 
    Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) ↔ 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_a_l467_46752


namespace NUMINAMATH_CALUDE_sheep_problem_l467_46730

theorem sheep_problem (n : ℕ) (h1 : n > 0) :
  let total := n * n
  let remainder := total % 10
  let elder_share := total - remainder
  let younger_share := remainder
  (remainder < 10 ∧ elder_share % 20 = 10) →
  (elder_share + younger_share + 2) / 2 = (elder_share + 2) / 2 ∧
  (elder_share + younger_share + 2) / 2 = (younger_share + 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_sheep_problem_l467_46730


namespace NUMINAMATH_CALUDE_tripod_height_theorem_l467_46788

/-- Represents a tripod with three legs -/
structure Tripod where
  leg_length : ℝ
  original_height : ℝ
  broken_leg_length : ℝ

/-- Calculates the new height of a tripod after one leg is shortened -/
def new_height (t : Tripod) : ℝ :=
  sorry

/-- Expresses the new height as a fraction m / √n -/
def height_fraction (t : Tripod) : ℚ × ℕ :=
  sorry

theorem tripod_height_theorem (t : Tripod) 
  (h_leg : t.leg_length = 5)
  (h_height : t.original_height = 4)
  (h_broken : t.broken_leg_length = 4) :
  let (m, n) := height_fraction t
  ⌊m + Real.sqrt n⌋ = 183 :=
sorry

end NUMINAMATH_CALUDE_tripod_height_theorem_l467_46788


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l467_46744

theorem roots_sum_and_product : ∃ (r₁ r₂ : ℚ),
  (∀ x, (3 * x + 2) * (x - 5) + (3 * x + 2) * (x - 8) = 0 ↔ x = r₁ ∨ x = r₂) ∧
  r₁ + r₂ = 35 / 6 ∧
  r₁ * r₂ = -13 / 3 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l467_46744


namespace NUMINAMATH_CALUDE_relationship_abc_l467_46722

theorem relationship_abc (a b c : ℝ) (ha : a = (0.4 : ℝ)^2) (hb : b = 2^(0.4 : ℝ)) (hc : c = Real.log 2 / Real.log 0.4) :
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_relationship_abc_l467_46722


namespace NUMINAMATH_CALUDE_store_a_advantage_l467_46782

/-- The original price of each computer in yuan -/
def original_price : ℝ := 6000

/-- The cost of buying computers from Store A -/
def cost_store_a (x : ℝ) : ℝ := original_price + (0.75 * original_price) * (x - 1)

/-- The cost of buying computers from Store B -/
def cost_store_b (x : ℝ) : ℝ := 0.8 * original_price * x

/-- Theorem stating when it's more advantageous to buy from Store A -/
theorem store_a_advantage (x : ℝ) : x > 5 → cost_store_a x < cost_store_b x := by
  sorry

#check store_a_advantage

end NUMINAMATH_CALUDE_store_a_advantage_l467_46782


namespace NUMINAMATH_CALUDE_zero_has_square_and_cube_root_l467_46716

/-- A number x is a square root of y if x * x = y -/
def is_square_root (x y : ℝ) : Prop := x * x = y

/-- A number x is a cube root of y if x * x * x = y -/
def is_cube_root (x y : ℝ) : Prop := x * x * x = y

/-- 0 has both a square root and a cube root -/
theorem zero_has_square_and_cube_root :
  ∃ (x y : ℝ), is_square_root x 0 ∧ is_cube_root y 0 :=
sorry

end NUMINAMATH_CALUDE_zero_has_square_and_cube_root_l467_46716


namespace NUMINAMATH_CALUDE_min_value_theorem_l467_46705

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  ∃ (m : ℝ), m = 4 ∧ ∀ x y, x > 0 → y > 0 → x + 1/y = 2 → 2/x + 2*y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l467_46705


namespace NUMINAMATH_CALUDE_money_sharing_l467_46723

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 2 * (total / 13) →
  ben = 3 * (total / 13) →
  carlos = 8 * (total / 13) →
  ben = 60 →
  total = 260 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l467_46723


namespace NUMINAMATH_CALUDE_total_siblings_weight_l467_46786

def antonio_weight : ℕ := 50
def sister_weight_diff : ℕ := 12
def antonio_backpack : ℕ := 5
def sister_backpack : ℕ := 3
def marco_weight : ℕ := 30
def stuffed_animal : ℕ := 2

theorem total_siblings_weight :
  (antonio_weight + (antonio_weight - sister_weight_diff) + marco_weight) +
  (antonio_backpack + sister_backpack + stuffed_animal) = 128 := by
  sorry

end NUMINAMATH_CALUDE_total_siblings_weight_l467_46786


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l467_46708

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 1 ∧ 3*x - 2*y = 7 ∧ x = 2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l467_46708


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l467_46756

theorem circle_area_from_circumference (k : ℝ) : 
  let circumference := 18 * Real.pi
  let radius := circumference / (2 * Real.pi)
  let area := k * Real.pi
  area = Real.pi * radius^2 → k = 81 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l467_46756


namespace NUMINAMATH_CALUDE_perpetually_alive_configurations_l467_46770

/-- Represents the state of a cell: alive or dead -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- Counts the number of alive neighbors for a cell -/
def countAliveNeighbors (grid : Grid m n) (i : Fin m) (j : Fin n) : ℕ :=
  sorry

/-- Updates the state of a single cell based on its neighbors -/
def updateCell (grid : Grid m n) (i : Fin m) (j : Fin n) : CellState :=
  sorry

/-- Updates the entire grid for one time step -/
def updateGrid (grid : Grid m n) : Grid m n :=
  sorry

/-- Checks if a grid has at least one alive cell -/
def hasAliveCell (grid : Grid m n) : Prop :=
  sorry

/-- Checks if a grid configuration is perpetually alive -/
def isPerpetuallyAlive (initialGrid : Grid m n) : Prop :=
  ∀ t : ℕ, hasAliveCell ((updateGrid^[t]) initialGrid)

/-- The main theorem: for all pairs (m, n) except (1,1), (1,3), and (3,1),
    there exists a perpetually alive configuration -/
theorem perpetually_alive_configurations (m n : ℕ) 
  (h : (m, n) ≠ (1, 1) ∧ (m, n) ≠ (1, 3) ∧ (m, n) ≠ (3, 1)) :
  ∃ (initialGrid : Grid m n), isPerpetuallyAlive initialGrid :=
sorry

end NUMINAMATH_CALUDE_perpetually_alive_configurations_l467_46770


namespace NUMINAMATH_CALUDE_proportion_sum_l467_46721

theorem proportion_sum (a b c d : ℚ) 
  (h1 : a/b = 3/2) 
  (h2 : c/d = 3/2) 
  (h3 : b + d ≠ 0) : 
  (a + c) / (b + d) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_proportion_sum_l467_46721


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l467_46732

-- Define the original length of the rectangle
def original_length : ℝ := 140

-- Define the length increase factor
def length_increase : ℝ := 1.30

-- Define the width decrease factor
def width_decrease : ℝ := 0.8230769230769231

-- Define the approximate width we want to prove
def approximate_width : ℝ := 130.91

-- Theorem statement
theorem rectangle_width_proof :
  ∃ (original_width : ℝ),
    (original_length * original_width = original_length * length_increase * original_width * width_decrease) ∧
    (abs (original_width - approximate_width) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l467_46732


namespace NUMINAMATH_CALUDE_existence_of_another_max_sequence_l467_46794

/-- Represents a sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Counts the number of occurrences of a sequence in a circular strip -/
def countOccurrences (strip : BinarySequence) (seq : BinarySequence) : ℕ := sorry

theorem existence_of_another_max_sequence 
  (n : ℕ) 
  (h_n : n > 5) 
  (strip : BinarySequence) 
  (h_strip : strip.length > n) 
  (M : ℕ) 
  (h_M_max : ∀ seq : BinarySequence, seq.length = n → countOccurrences strip seq ≤ M) 
  (seq_max : BinarySequence) 
  (h_seq_max : seq_max = [true, true] ++ List.replicate (n - 2) false) 
  (h_M_reached : countOccurrences strip seq_max = M) 
  (seq_min : BinarySequence) 
  (h_seq_min : seq_min = List.replicate (n - 2) false ++ [true, true]) 
  (h_min_reached : ∀ seq : BinarySequence, seq.length = n → 
    countOccurrences strip seq ≥ countOccurrences strip seq_min) :
  ∃ (seq : BinarySequence), seq.length = n ∧ seq ≠ seq_max ∧ countOccurrences strip seq = M :=
sorry

end NUMINAMATH_CALUDE_existence_of_another_max_sequence_l467_46794


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l467_46759

theorem complex_fraction_simplification :
  (1 + 2 * Complex.I) / (1 - 2 * Complex.I) = -(3/5 : ℂ) + (4/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l467_46759


namespace NUMINAMATH_CALUDE_bulbs_per_pack_l467_46787

/-- The number of bulbs Sean needs to replace in each room --/
def bedroom_bulbs : ℕ := 2
def bathroom_bulbs : ℕ := 1
def kitchen_bulbs : ℕ := 1
def basement_bulbs : ℕ := 4

/-- The total number of bulbs Sean needs to replace in the rooms --/
def room_bulbs : ℕ := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

/-- The number of bulbs Sean needs to replace in the garage --/
def garage_bulbs : ℕ := room_bulbs / 2

/-- The total number of bulbs Sean needs to replace --/
def total_bulbs : ℕ := room_bulbs + garage_bulbs

/-- The number of packs Sean will buy --/
def num_packs : ℕ := 6

/-- Theorem: The number of bulbs in each pack is 2 --/
theorem bulbs_per_pack : total_bulbs / num_packs = 2 := by
  sorry

end NUMINAMATH_CALUDE_bulbs_per_pack_l467_46787


namespace NUMINAMATH_CALUDE_spending_ratio_theorem_l467_46769

/-- Represents David's wages from last week -/
def last_week_wages : ℝ := 1

/-- Percentage spent on recreation last week -/
def last_week_recreation_percent : ℝ := 0.20

/-- Percentage spent on transportation last week -/
def last_week_transportation_percent : ℝ := 0.10

/-- Percentage reduction in wages this week -/
def wage_reduction_percent : ℝ := 0.30

/-- Percentage spent on recreation this week -/
def this_week_recreation_percent : ℝ := 0.25

/-- Percentage spent on transportation this week -/
def this_week_transportation_percent : ℝ := 0.15

/-- The ratio of this week's combined spending to last week's is approximately 0.9333 -/
theorem spending_ratio_theorem : 
  let last_week_total := (last_week_recreation_percent + last_week_transportation_percent) * last_week_wages
  let this_week_wages := (1 - wage_reduction_percent) * last_week_wages
  let this_week_total := (this_week_recreation_percent + this_week_transportation_percent) * this_week_wages
  abs ((this_week_total / last_week_total) - 0.9333) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_spending_ratio_theorem_l467_46769


namespace NUMINAMATH_CALUDE_triangle_inequality_max_l467_46798

theorem triangle_inequality_max (a b c x y z : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  a * y * z + b * z * x + c * x * y ≤ 
    (a * b * c) / (-a^2 - b^2 - c^2 + 2 * (a * b + b * c + c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_max_l467_46798


namespace NUMINAMATH_CALUDE_cos_pi_minus_theta_point_l467_46704

theorem cos_pi_minus_theta_point (θ : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos θ = 4 ∧ r * Real.sin θ = -3) →
  Real.cos (Real.pi - θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_theta_point_l467_46704


namespace NUMINAMATH_CALUDE_outgoing_roads_different_colors_l467_46771

/-- Represents a color of a street -/
inductive Color
| Red
| Blue
| Green

/-- Represents an intersection in the city -/
structure Intersection where
  streets : Fin 3 → Color
  different_colors : streets 0 ≠ streets 1 ∧ streets 1 ≠ streets 2 ∧ streets 0 ≠ streets 2

/-- Represents the city with its intersections and outgoing roads -/
structure City where
  intersections : Set Intersection
  outgoing_roads : Fin 3 → Color

/-- The theorem stating that the outgoing roads have different colors -/
theorem outgoing_roads_different_colors (city : City) : 
  city.outgoing_roads 0 ≠ city.outgoing_roads 1 ∧ 
  city.outgoing_roads 1 ≠ city.outgoing_roads 2 ∧ 
  city.outgoing_roads 0 ≠ city.outgoing_roads 2 := by
  sorry

end NUMINAMATH_CALUDE_outgoing_roads_different_colors_l467_46771


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l467_46767

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 15 = (x + 3) * k) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l467_46767


namespace NUMINAMATH_CALUDE_fraction_subtraction_l467_46784

theorem fraction_subtraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  1 / x - 1 / (x - 1) = -1 / (x^2 - x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l467_46784


namespace NUMINAMATH_CALUDE_cat_dog_food_difference_l467_46750

/-- Represents the number of packages of cat food Adam bought. -/
def cat_food_packages : ℕ := 15

/-- Represents the number of packages of dog food Adam bought. -/
def dog_food_packages : ℕ := 10

/-- Represents the number of cans in each package of cat food. -/
def cans_per_cat_package : ℕ := 12

/-- Represents the number of cans in each package of dog food. -/
def cans_per_dog_package : ℕ := 8

/-- Theorem stating the difference between the total number of cans of cat food and dog food. -/
theorem cat_dog_food_difference :
  cat_food_packages * cans_per_cat_package - dog_food_packages * cans_per_dog_package = 100 := by
  sorry

end NUMINAMATH_CALUDE_cat_dog_food_difference_l467_46750


namespace NUMINAMATH_CALUDE_boatsman_speed_calculation_l467_46764

/-- The speed of the boatsman in still water -/
def boatsman_speed : ℝ := 7

/-- The speed of the river -/
def river_speed : ℝ := 3

/-- The distance between the two destinations -/
def distance : ℝ := 40

/-- The time difference between upstream and downstream travel -/
def time_difference : ℝ := 6

theorem boatsman_speed_calculation :
  (distance / (boatsman_speed - river_speed) - distance / (boatsman_speed + river_speed) = time_difference) ∧
  (boatsman_speed > river_speed) :=
sorry

end NUMINAMATH_CALUDE_boatsman_speed_calculation_l467_46764


namespace NUMINAMATH_CALUDE_probability_log3_integer_l467_46736

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The count of four-digit numbers that are powers of 3. -/
def CountPowersOfThree : ℕ := 2

/-- The total count of four-digit numbers. -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The probability of a randomly chosen four-digit number being a power of 3. -/
def ProbabilityPowerOfThree : ℚ := CountPowersOfThree / TotalFourDigitNumbers

theorem probability_log3_integer :
  ProbabilityPowerOfThree = 1 / 4500 := by
  sorry

end NUMINAMATH_CALUDE_probability_log3_integer_l467_46736


namespace NUMINAMATH_CALUDE_utility_value_sets_l467_46743

theorem utility_value_sets (A B : Set α) (h : B ⊆ A) : A ∪ B = A := by
  sorry

end NUMINAMATH_CALUDE_utility_value_sets_l467_46743


namespace NUMINAMATH_CALUDE_population_size_l467_46700

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem population_size (birth_rate death_rate net_growth_rate : ℚ) : 
  birth_rate = 32 →
  death_rate = 11 →
  net_growth_rate = 21 / 1000 →
  (birth_rate - death_rate) / 1000 = net_growth_rate →
  1000 = (birth_rate - death_rate) / net_growth_rate :=
by sorry

end NUMINAMATH_CALUDE_population_size_l467_46700


namespace NUMINAMATH_CALUDE_fraction_simplification_l467_46761

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l467_46761


namespace NUMINAMATH_CALUDE_prob_empty_mailbox_is_five_ninths_l467_46734

/-- The number of different greeting cards -/
def num_cards : ℕ := 4

/-- The number of different mailboxes -/
def num_mailboxes : ℕ := 3

/-- The probability of at least one mailbox being empty when cards are randomly placed -/
def prob_empty_mailbox : ℚ := 5/9

/-- Theorem stating that the probability of at least one empty mailbox is 5/9 -/
theorem prob_empty_mailbox_is_five_ninths :
  prob_empty_mailbox = 5/9 :=
sorry

end NUMINAMATH_CALUDE_prob_empty_mailbox_is_five_ninths_l467_46734


namespace NUMINAMATH_CALUDE_product_of_repeated_digits_l467_46725

def number_of_3s : ℕ := 25
def number_of_6s : ℕ := 25

def number_of_2s : ℕ := 24
def number_of_7s : ℕ := 24

def first_number : ℕ := (3 * (10^number_of_3s - 1)) / 9
def second_number : ℕ := (6 * (10^number_of_6s - 1)) / 9

def result : ℕ := (2 * 10^49 + 10^48 + 7 * (10^24 - 1) / 9) * 10 + 8

theorem product_of_repeated_digits :
  first_number * second_number = result := by sorry

end NUMINAMATH_CALUDE_product_of_repeated_digits_l467_46725


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l467_46717

theorem isosceles_triangle_condition (a b : ℝ) (A B : ℝ) : 
  0 < a → 0 < b → 0 < A → A < π → 0 < B → B < π →
  a * Real.cos B = b * Real.cos A → A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l467_46717


namespace NUMINAMATH_CALUDE_oil_mixture_price_l467_46774

theorem oil_mixture_price (x y z : ℝ) 
  (volume_constraint : x + y + z = 23.5)
  (cost_constraint : 55 * x + 70 * y + 82 * z = 65 * 23.5) :
  (55 * x + 70 * y + 82 * z) / (x + y + z) = 65 := by
  sorry

end NUMINAMATH_CALUDE_oil_mixture_price_l467_46774


namespace NUMINAMATH_CALUDE_susan_cloth_bags_l467_46711

/-- Calculates the number of cloth bags Susan brought to carry peaches. -/
def number_of_cloth_bags (total_peaches knapsack_peaches : ℕ) : ℕ :=
  let cloth_bag_peaches := 2 * knapsack_peaches
  (total_peaches - knapsack_peaches) / cloth_bag_peaches

/-- Proves that Susan brought 2 cloth bags given the problem conditions. -/
theorem susan_cloth_bags :
  number_of_cloth_bags (5 * 12) 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_cloth_bags_l467_46711


namespace NUMINAMATH_CALUDE_grocer_coffee_stock_l467_46777

/-- The amount of coffee initially in stock -/
def initial_stock : ℝ := 400

/-- The percentage of decaffeinated coffee in the initial stock -/
def initial_decaf_percent : ℝ := 0.20

/-- The amount of additional coffee purchased -/
def additional_coffee : ℝ := 100

/-- The percentage of decaffeinated coffee in the additional purchase -/
def additional_decaf_percent : ℝ := 0.60

/-- The final percentage of decaffeinated coffee after the purchase -/
def final_decaf_percent : ℝ := 0.28000000000000004

theorem grocer_coffee_stock :
  (initial_decaf_percent * initial_stock + additional_decaf_percent * additional_coffee) / 
  (initial_stock + additional_coffee) = final_decaf_percent := by
  sorry

end NUMINAMATH_CALUDE_grocer_coffee_stock_l467_46777


namespace NUMINAMATH_CALUDE_parabola_shift_right_one_unit_l467_46775

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_right_one_unit :
  let original := Parabola.mk (-1/2) 0 0
  let shifted := shift_parabola original 1
  shifted = Parabola.mk (-1/2) 1 (-1/2) := by sorry

end NUMINAMATH_CALUDE_parabola_shift_right_one_unit_l467_46775


namespace NUMINAMATH_CALUDE_video_game_sales_l467_46749

/-- Calculates the money earned from selling working video games. -/
def money_earned (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that given 10 total games, 8 non-working games, and a price of $6 per working game,
    the total money earned is $12. -/
theorem video_game_sales : money_earned 10 8 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_l467_46749


namespace NUMINAMATH_CALUDE_painted_faces_count_correct_l467_46753

/-- Represents the count of painted faces for unit cubes cut from a larger cube. -/
structure PaintedFacesCount where
  one_face : ℕ
  two_faces : ℕ
  three_faces : ℕ

/-- 
  Given a cube with side length a (where a is a natural number greater than 2),
  calculates the number of unit cubes with exactly one, two, and three faces painted
  when the cube is cut into unit cubes.
-/
def count_painted_faces (a : ℕ) : PaintedFacesCount :=
  { one_face := 6 * (a - 2)^2,
    two_faces := 12 * (a - 2),
    three_faces := 8 }

/-- Theorem stating the correct count of painted faces for unit cubes. -/
theorem painted_faces_count_correct (a : ℕ) (h : a > 2) :
  count_painted_faces a = { one_face := 6 * (a - 2)^2,
                            two_faces := 12 * (a - 2),
                            three_faces := 8 } := by
  sorry

end NUMINAMATH_CALUDE_painted_faces_count_correct_l467_46753


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l467_46729

/-- A parabola is defined by its equation and opening direction -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  opens_downward : Bool

/-- The focus of a parabola is a point in the plane -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola := {
  equation := fun x y => y = -1/4 * x^2,
  opens_downward := true
}

/-- Theorem stating that the focus of our parabola is at (0, -1) -/
theorem focus_of_our_parabola : focus our_parabola = (0, -1) := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l467_46729


namespace NUMINAMATH_CALUDE_satisfying_function_is_identity_l467_46773

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (f 1 = 1) ∧
  (∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_satisfying_function_is_identity_l467_46773


namespace NUMINAMATH_CALUDE_remaining_distance_proof_l467_46797

def total_distance : ℝ := 369
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2

theorem remaining_distance_proof :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time) = 121 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_proof_l467_46797


namespace NUMINAMATH_CALUDE_probability_two_segments_longer_than_one_l467_46755

/-- The probability of exactly two segments being longer than 1 when a line segment 
    of length 3 is divided into three parts by randomly selecting two points -/
theorem probability_two_segments_longer_than_one (total_length : ℝ) 
  (h_total_length : total_length = 3) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_probability_two_segments_longer_than_one_l467_46755


namespace NUMINAMATH_CALUDE_g_4_cubed_eq_16_l467_46710

/-- Given two functions f and g satisfying certain conditions, prove that [g(4)]^3 = 16 -/
theorem g_4_cubed_eq_16
  (f g : ℝ → ℝ)
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^3)
  (h3 : g 16 = 16) :
  (g 4)^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_g_4_cubed_eq_16_l467_46710


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l467_46791

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + 3 - k

-- Define the condition for two distinct real roots
def has_two_distinct_roots (k : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ quadratic α k = 0 ∧ quadratic β k = 0

-- Define the relationship between k and the roots
def root_relationship (k α β : ℝ) : Prop :=
  k^2 = α * β + 3*k

-- Theorem statement
theorem quadratic_root_theorem (k : ℝ) :
  has_two_distinct_roots k → (∃ α β : ℝ, root_relationship k α β) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l467_46791


namespace NUMINAMATH_CALUDE_proposition_relationship_l467_46799

theorem proposition_relationship (a b : ℝ) : 
  ¬(((a + b ≠ 4) → (a ≠ 1 ∧ b ≠ 3)) ∧ ((a ≠ 1 ∧ b ≠ 3) → (a + b ≠ 4))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l467_46799


namespace NUMINAMATH_CALUDE_events_B_C_mutually_exclusive_not_complementary_l467_46741

-- Define the sample space
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {n ∈ Ω | n % 2 = 1}
def B : Set Nat := {n ∈ Ω | n ≤ 2}
def C : Set Nat := {n ∈ Ω | n ≥ 4}

-- Theorem statement
theorem events_B_C_mutually_exclusive_not_complementary :
  (B ∩ C = ∅) ∧ (B ∪ C ≠ Ω) :=
sorry

end NUMINAMATH_CALUDE_events_B_C_mutually_exclusive_not_complementary_l467_46741


namespace NUMINAMATH_CALUDE_initial_girls_count_l467_46706

theorem initial_girls_count (b g : ℕ) : 
  (2 * (g - 15) = b) →
  (5 * (b - 45) = g - 15) →
  g = 40 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l467_46706


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l467_46737

theorem unique_solution_for_equation :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ∧ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l467_46737


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l467_46728

def M : Set ℝ := {x | (x + 3) * (x - 5) > 0}

def P (a : ℝ) : Set ℝ := {x | x^2 + (a - 8) * x - 8 * a ≤ 0}

def target_set : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_not_necessary_condition :
  (∀ a, a = 0 → M ∩ P a = target_set) ∧
  ¬(∀ a, M ∩ P a = target_set → a = 0) := by sorry

theorem necessary_not_sufficient_condition :
  (∀ a, M ∩ P a = target_set → a ≤ 3) ∧
  ¬(∀ a, a ≤ 3 → M ∩ P a = target_set) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l467_46728


namespace NUMINAMATH_CALUDE_extremum_sum_l467_46746

/-- A function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_sum (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_extremum_sum_l467_46746


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l467_46719

theorem quadratic_equation_roots (k : ℝ) :
  let f := fun x : ℝ => x^2 + (2*k - 1)*x + k^2
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0) →
  (k < 1/4 ∧
   (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 = 0 → f x2 = 0 → x1 + x2 + x1*x2 - 1 = 0 → k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l467_46719


namespace NUMINAMATH_CALUDE_mike_marbles_l467_46712

theorem mike_marbles (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 8 → given = 4 → remaining = initial - given → remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_marbles_l467_46712


namespace NUMINAMATH_CALUDE_rectangle_length_l467_46793

/-- Given a rectangle with perimeter 700 and breadth 100, its length is 250. -/
theorem rectangle_length (perimeter breadth length : ℝ) : 
  perimeter = 700 →
  breadth = 100 →
  perimeter = 2 * (length + breadth) →
  length = 250 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l467_46793


namespace NUMINAMATH_CALUDE_increasing_decreasing_functions_exist_l467_46758

-- Define a function that is increasing on one interval and decreasing on another
def has_increasing_decreasing_intervals (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∧
    (∀ x y, c < x ∧ x < y ∧ y < d → f y < f x)

-- Theorem stating that such functions exist
theorem increasing_decreasing_functions_exist :
  ∃ f : ℝ → ℝ, has_increasing_decreasing_intervals f :=
sorry

end NUMINAMATH_CALUDE_increasing_decreasing_functions_exist_l467_46758


namespace NUMINAMATH_CALUDE_existence_of_triangle_with_divisible_side_lengths_l467_46733

/-- Given an odd prime p, a positive integer n, and 8 distinct points with integer coordinates
    on a circle of diameter p^n, there exists a triangle formed by three of these points
    such that the square of its side lengths is divisible by p^(n+1). -/
theorem existence_of_triangle_with_divisible_side_lengths
  (p : ℕ) (n : ℕ) (h_p_prime : Nat.Prime p) (h_p_odd : Odd p) (h_n_pos : 0 < n)
  (points : Fin 8 → ℤ × ℤ)
  (h_distinct : Function.Injective points)
  (h_on_circle : ∀ i : Fin 8, (points i).1^2 + (points i).2^2 = (p^n)^2) :
  ∃ i j k : Fin 8, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ m : ℕ, (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2) * m = p^(n+1)) ∧
    (∃ m : ℕ, (((points j).1 - (points k).1)^2 + ((points j).2 - (points k).2)^2) * m = p^(n+1)) ∧
    (∃ m : ℕ, (((points k).1 - (points i).1)^2 + ((points k).2 - (points i).2)^2) * m = p^(n+1)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_triangle_with_divisible_side_lengths_l467_46733


namespace NUMINAMATH_CALUDE_equidistant_point_l467_46731

/-- The distance between two points in a 2D plane -/
def distance (x1 y1 x2 y2 : ℚ) : ℚ :=
  ((x2 - x1)^2 + (y2 - y1)^2).sqrt

/-- The point C with coordinates (3, 0) -/
def C : ℚ × ℚ := (3, 0)

/-- The point D with coordinates (5, 6) -/
def D : ℚ × ℚ := (5, 6)

/-- The y-coordinate of the point on the y-axis -/
def y : ℚ := 13/3

theorem equidistant_point : 
  distance 0 y C.1 C.2 = distance 0 y D.1 D.2 := by sorry

end NUMINAMATH_CALUDE_equidistant_point_l467_46731


namespace NUMINAMATH_CALUDE_least_common_denominator_l467_46779

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 11))))) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l467_46779


namespace NUMINAMATH_CALUDE_function_properties_l467_46745

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x + 1

theorem function_properties :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧
  (f 1 1 = 2) ∧
  (∀ (a : ℝ), (∃! (x : ℝ), x > Real.exp (-3) ∧ f a x = 0) ↔ 
    (a ≤ 2 / Real.exp 3 ∨ a = 1 / Real.exp 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l467_46745


namespace NUMINAMATH_CALUDE_alley_width_l467_46748

/-- The width of a narrow alley given a ladder's length and angles -/
theorem alley_width (b : ℝ) (h_b_pos : b > 0) : ∃ w : ℝ,
  w = b * (1 + Real.sqrt 3) / 2 ∧
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x = b * Real.cos (π / 3) ∧
    y = b * Real.cos (π / 6) ∧
    w = x + y :=
by sorry

end NUMINAMATH_CALUDE_alley_width_l467_46748


namespace NUMINAMATH_CALUDE_triangle_area_l467_46772

/-- Given a triangle ABC with cos A = 4/5 and AB · AC = 8, prove that its area is 3 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let cosA := 4/5
  let dotProduct := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  dotProduct = 8 →
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
          Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * 
          Real.sqrt (1 - cosA^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l467_46772


namespace NUMINAMATH_CALUDE_store_inventory_problem_l467_46781

/-- Represents the inventory of a store selling pomelos and watermelons -/
structure StoreInventory where
  pomelos : ℕ
  watermelons : ℕ

/-- Represents the daily sales of pomelos and watermelons -/
structure DailySales where
  pomelos : ℕ
  watermelons : ℕ

/-- The theorem statement for the store inventory problem -/
theorem store_inventory_problem 
  (initial : StoreInventory)
  (sales : DailySales)
  (days : ℕ) :
  initial.watermelons = 3 * initial.pomelos →
  sales.pomelos = 20 →
  sales.watermelons = 30 →
  days = 3 →
  initial.watermelons - days * sales.watermelons = 
    4 * (initial.pomelos - days * sales.pomelos) - 26 →
  initial.pomelos = 176 := by
  sorry


end NUMINAMATH_CALUDE_store_inventory_problem_l467_46781


namespace NUMINAMATH_CALUDE_cubic_root_sum_l467_46790

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  (a*b)/c + (b*c)/a + (c*a)/b = 49/6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l467_46790


namespace NUMINAMATH_CALUDE_stamp_exchange_problem_l467_46796

theorem stamp_exchange_problem (petya_stamps : ℕ) (kolya_stamps : ℕ) : 
  kolya_stamps = petya_stamps + 5 →
  (0.76 * kolya_stamps + 0.2 * petya_stamps : ℝ) = ((0.8 * petya_stamps + 0.24 * kolya_stamps : ℝ) - 1) →
  petya_stamps = 45 ∧ kolya_stamps = 50 := by
sorry

end NUMINAMATH_CALUDE_stamp_exchange_problem_l467_46796


namespace NUMINAMATH_CALUDE_pm2_5_diameter_scientific_notation_l467_46709

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The diameter of PM2.5 particulate matter in meters -/
def pm2_5_diameter : ℝ := 0.0000025

/-- The scientific notation representation of the PM2.5 diameter -/
def pm2_5_scientific : ScientificNotation :=
  { coefficient := 2.5
    exponent := -6
    valid := by sorry }

theorem pm2_5_diameter_scientific_notation :
  pm2_5_diameter = pm2_5_scientific.coefficient * (10 : ℝ) ^ pm2_5_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_pm2_5_diameter_scientific_notation_l467_46709


namespace NUMINAMATH_CALUDE_jo_stair_climbing_l467_46780

/-- Number of ways to climb n stairs with 1, 2, or 3 steps at a time -/
def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| n + 3 => f (n + 2) + f (n + 1) + f n

/-- Number of ways to climb n stairs, finishing with a 3-step -/
def g (n : ℕ) : ℕ := if n < 3 then 0 else f (n - 3)

theorem jo_stair_climbing :
  g 8 = 13 := by sorry

end NUMINAMATH_CALUDE_jo_stair_climbing_l467_46780


namespace NUMINAMATH_CALUDE_hotel_expenditure_l467_46714

theorem hotel_expenditure (n : ℕ) (m : ℕ) (individual_cost : ℕ) (extra_cost : ℕ) 
  (h1 : n = 9)
  (h2 : m = 8)
  (h3 : individual_cost = 12)
  (h4 : extra_cost = 8) :
  m * individual_cost + (individual_cost + (m * individual_cost + individual_cost + extra_cost) / n) = 117 := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l467_46714
