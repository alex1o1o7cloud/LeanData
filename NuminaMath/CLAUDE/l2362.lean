import Mathlib

namespace NUMINAMATH_CALUDE_certain_value_proof_l2362_236205

theorem certain_value_proof (N : ℝ) (h : 0.4 * N = 420) : (1/4) * (1/3) * (2/5) * N = 35 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l2362_236205


namespace NUMINAMATH_CALUDE_simplify_expression_l2362_236238

theorem simplify_expression : 
  (Real.sqrt 6 - Real.sqrt 18) * Real.sqrt (1/3) + 2 * Real.sqrt 6 = Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2362_236238


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l2362_236281

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 15, 
    prove that the sum of the x-coordinate and y-coordinate of its center is 7. -/
theorem circle_center_coordinate_sum :
  ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = 10) ∧ h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l2362_236281


namespace NUMINAMATH_CALUDE_production_line_b_units_l2362_236216

/-- Represents a production line -/
inductive ProductionLine
| A
| B
| C

/-- Represents the production data for a factory -/
structure FactoryProduction where
  total_units : ℕ
  sampling_ratio : ProductionLine → ℕ
  stratified_sampling : Bool

/-- Calculates the number of units produced by a specific production line -/
def units_produced (fp : FactoryProduction) (line : ProductionLine) : ℕ :=
  (fp.total_units * fp.sampling_ratio line) / (fp.sampling_ratio ProductionLine.A + fp.sampling_ratio ProductionLine.B + fp.sampling_ratio ProductionLine.C)

theorem production_line_b_units
  (fp : FactoryProduction)
  (h1 : fp.total_units = 5000)
  (h2 : fp.sampling_ratio ProductionLine.A = 1)
  (h3 : fp.sampling_ratio ProductionLine.B = 2)
  (h4 : fp.sampling_ratio ProductionLine.C = 2)
  (h5 : fp.stratified_sampling = true) :
  units_produced fp ProductionLine.B = 2000 := by
  sorry

end NUMINAMATH_CALUDE_production_line_b_units_l2362_236216


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_simplify_fraction_3_l2362_236240

-- Part 1
theorem simplify_fraction_1 (x : ℝ) (h : x ≠ 1) :
  (3 * x + 2) / (x - 1) - 5 / (x - 1) = 3 :=
by sorry

-- Part 2
theorem simplify_fraction_2 (a : ℝ) (h : a ≠ 3) :
  (a^2) / (a^2 - 6*a + 9) / (a / (a - 3)) = a / (a - 3) :=
by sorry

-- Part 3
theorem simplify_fraction_3 (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ -4) :
  (x - 4) / (x + 3) / (x - 3 - 7 / (x + 3)) = 1 / (x + 4) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_simplify_fraction_3_l2362_236240


namespace NUMINAMATH_CALUDE_equation_solutions_no_other_solutions_l2362_236294

/-- Definition of the factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The main theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x y z : ℕ, 2^x + 3^y + 7 = factorial z ↔ (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

/-- Auxiliary theorem: There are no other solutions -/
theorem no_other_solutions :
  ∀ x y z : ℕ, 2^x + 3^y + 7 = factorial z →
  (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_no_other_solutions_l2362_236294


namespace NUMINAMATH_CALUDE_rectangular_field_with_pond_l2362_236286

theorem rectangular_field_with_pond (l w : ℝ) : 
  l = 2 * w →                    -- length is double the width
  l * w = 8 * 49 →               -- area of field is 8 times area of pond (7^2 = 49)
  l = 28 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_with_pond_l2362_236286


namespace NUMINAMATH_CALUDE_positive_sum_product_iff_l2362_236248

theorem positive_sum_product_iff (a b : ℝ) : (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_product_iff_l2362_236248


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2362_236279

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, (5 * x^2 + 7 = 2 * x - 6) ↔ (x = p + q * I ∨ x = p - q * I)) →
  p + q^2 = 69/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2362_236279


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2362_236213

theorem sum_of_coefficients (a b c d e : ℤ) : 
  (∀ x : ℚ, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9 →
  a + b + c + d + e = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2362_236213


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2362_236293

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2362_236293


namespace NUMINAMATH_CALUDE_backpack_cost_l2362_236224

/-- The cost of backpacks with discount and monogramming -/
theorem backpack_cost (original_price : ℝ) (discount_percent : ℝ) (monogram_fee : ℝ) (quantity : ℕ) :
  original_price = 20 →
  discount_percent = 20 →
  monogram_fee = 12 →
  quantity = 5 →
  quantity * (original_price * (1 - discount_percent / 100) + monogram_fee) = 140 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_l2362_236224


namespace NUMINAMATH_CALUDE_age_difference_l2362_236266

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove that Mehki is 10 years older than Jordyn. -/
theorem age_difference (mehki_age jordyn_age zrinka_age : ℕ) 
  (h1 : jordyn_age = 2 * zrinka_age)
  (h2 : zrinka_age = 6)
  (h3 : mehki_age = 22) :
  mehki_age - jordyn_age = 10 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2362_236266


namespace NUMINAMATH_CALUDE_complex_sixth_root_of_negative_eight_l2362_236218

theorem complex_sixth_root_of_negative_eight :
  {z : ℂ | z^6 = -8} = {Complex.I * Real.rpow 2 (1/3), -Complex.I * Real.rpow 2 (1/3)} := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_root_of_negative_eight_l2362_236218


namespace NUMINAMATH_CALUDE_solve_equation_l2362_236260

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2362_236260


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2362_236269

theorem max_sum_of_factors (heart club : ℕ) (h : heart * club = 48) :
  ∃ (a b : ℕ), a * b = 48 ∧ a + b ≤ heart + club ∧ a + b = 49 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2362_236269


namespace NUMINAMATH_CALUDE_election_majority_l2362_236270

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 455 →
  winning_percentage = 70 / 100 →
  ⌊(2 * winning_percentage - 1) * total_votes⌋ = 182 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l2362_236270


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l2362_236285

theorem probability_at_least_one_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 5 →
  white_balls = 4 →
  (1 - (red_balls / total_balls * (red_balls - 1) / (total_balls - 1))) = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l2362_236285


namespace NUMINAMATH_CALUDE_original_triangle_area_l2362_236282

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with an area of 256 square feet, 
    the area of the original triangle is 16 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 4 * original →  -- The dimensions are quadrupled
  new^2 = 256 →         -- The area of the new triangle is 256 square feet
  original^2 = 16 :=    -- The area of the original triangle is 16 square feet
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l2362_236282


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_is_17_l2362_236264

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

structure DistancesToFaces where
  ABC : ℝ
  ABD : ℝ
  ACD : ℝ
  BCD : ℝ

def ABCD : Tetrahedron := sorry

def X : Point := sorry
def Y : Point := sorry

def distances_X : DistancesToFaces := {
  ABC := 14,
  ABD := 11,
  ACD := 29,
  BCD := 8
}

def distances_Y : DistancesToFaces := {
  ABC := 15,
  ABD := 13,
  ACD := 25,
  BCD := 11
}

def inscribed_sphere_radius (t : Tetrahedron) : ℝ := sorry

theorem inscribed_sphere_radius_is_17 :
  inscribed_sphere_radius ABCD = 17 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_is_17_l2362_236264


namespace NUMINAMATH_CALUDE_abs_sum_inequality_range_l2362_236235

theorem abs_sum_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) → a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_range_l2362_236235


namespace NUMINAMATH_CALUDE_greatest_integer_under_sqrt_l2362_236237

theorem greatest_integer_under_sqrt (N : ℤ) : 
  (∀ k : ℤ, k ≤ Real.sqrt (2007^2 - 20070 + 31) → k ≤ N) ∧ 
  N ≤ Real.sqrt (2007^2 - 20070 + 31) →
  N = 2002 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_under_sqrt_l2362_236237


namespace NUMINAMATH_CALUDE_initial_plant_ratio_l2362_236241

/-- Represents the number and types of plants in Roxy's garden -/
structure Garden where
  flowering : ℕ
  fruiting : ℕ

/-- Represents the transactions of buying and giving away plants -/
structure Transactions where
  bought_flowering : ℕ
  bought_fruiting : ℕ
  given_flowering : ℕ
  given_fruiting : ℕ

/-- Calculates the final number of plants after transactions -/
def final_plants (initial : Garden) (trans : Transactions) : ℕ :=
  initial.flowering + initial.fruiting + trans.bought_flowering + trans.bought_fruiting - 
  trans.given_flowering - trans.given_fruiting

/-- Theorem stating the initial ratio of fruiting to flowering plants -/
theorem initial_plant_ratio (initial : Garden) (trans : Transactions) :
  initial.flowering = 7 ∧ 
  trans.bought_flowering = 3 ∧ 
  trans.bought_fruiting = 2 ∧
  trans.given_flowering = 1 ∧
  trans.given_fruiting = 4 ∧
  final_plants initial trans = 21 →
  initial.fruiting = 2 * initial.flowering :=
by
  sorry


end NUMINAMATH_CALUDE_initial_plant_ratio_l2362_236241


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l2362_236220

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l2362_236220


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2362_236259

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  Nat.choose (total_balls + num_boxes - 1) (num_boxes - 1)

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes
    with at least one ball in each box -/
def distribute_balls_no_empty (total_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  distribute_balls (total_balls - num_boxes) num_boxes

theorem seven_balls_three_boxes :
  distribute_balls_no_empty 7 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2362_236259


namespace NUMINAMATH_CALUDE_large_cube_edge_is_one_meter_l2362_236280

/-- The edge length of a cubical box that can contain a given number of smaller cubes -/
def large_cube_edge_length (small_cube_edge : ℝ) (num_small_cubes : ℝ) : ℝ :=
  (small_cube_edge^3 * num_small_cubes)^(1/3)

/-- Theorem: The edge length of a cubical box that can contain 999.9999999999998 cubes 
    with 10 cm edge length is 1 meter -/
theorem large_cube_edge_is_one_meter :
  large_cube_edge_length 0.1 999.9999999999998 = 1 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_edge_is_one_meter_l2362_236280


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2362_236255

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c (-1) = -1)
  (h2 : f a b c 0 = -7/4)
  (h3 : f a b c 1 = -2)
  (h4 : f a b c 2 = -7/4) :
  (f a b c 3 = -1) ∧
  (∀ x, f a b c x ≥ -2) ∧
  (f a b c 1 = -2) ∧
  (∀ x₁ x₂, -1 < x₁ → x₁ < 0 → 1 < x₂ → x₂ < 2 → f a b c x₁ > f a b c x₂) ∧
  (∀ x, 0 ≤ x → x ≤ 5 → -2 ≤ f a b c x ∧ f a b c x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2362_236255


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2362_236271

theorem solve_exponential_equation :
  ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2362_236271


namespace NUMINAMATH_CALUDE_angle_AOB_is_right_angle_l2362_236298

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 3*x

-- Define a line passing through (3,0)
def line_through_3_0 (t : ℝ) (x y : ℝ) : Prop := x = t*y + 3

-- Define the intersection points
def intersection_points (t : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_3_0 t x₁ y₁ ∧ line_through_3_0 t x₂ y₂

-- Theorem statement
theorem angle_AOB_is_right_angle (t : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points t x₁ y₁ x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_angle_AOB_is_right_angle_l2362_236298


namespace NUMINAMATH_CALUDE_weight_problem_l2362_236276

theorem weight_problem (a b c d e : ℝ) : 
  ((a + b + c) / 3 = 84) →
  ((a + b + c + d) / 4 = 80) →
  (e = d + 3) →
  ((b + c + d + e) / 4 = 79) →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l2362_236276


namespace NUMINAMATH_CALUDE_function_identity_l2362_236202

theorem function_identity (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
    ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2362_236202


namespace NUMINAMATH_CALUDE_digit_sum_in_multiplication_l2362_236258

theorem digit_sum_in_multiplication (c d a b : ℕ) : 
  c < 10 → d < 10 → a < 10 → b < 10 →
  (30 + c) * (10 * d + 4) = 100 * a + 10 * b + 8 →
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_in_multiplication_l2362_236258


namespace NUMINAMATH_CALUDE_tangent_sum_l2362_236290

theorem tangent_sum (x y a b : Real) 
  (h1 : Real.sin (2 * x) + Real.sin (2 * y) = a)
  (h2 : Real.cos (2 * x) + Real.cos (2 * y) = b)
  (h3 : a^2 + b^2 ≤ 4)
  (h4 : a^2 + b^2 + 2*b ≠ 0) :
  Real.tan x + Real.tan y = 4 * a / (a^2 + b^2 + 2*b) :=
sorry

end NUMINAMATH_CALUDE_tangent_sum_l2362_236290


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2362_236254

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) :
  5 * x^2 - 7 * x - (3 * x^2 - 2 * (-x^2 + 4 * x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2362_236254


namespace NUMINAMATH_CALUDE_triangle_side_length_l2362_236267

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area condition
  (B = π/3) →  -- 60° in radians
  (a^2 + c^2 = 3*a*c) →  -- Given condition
  (b = 2 * Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2362_236267


namespace NUMINAMATH_CALUDE_fractions_product_one_l2362_236222

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def irreducible (n d : ℕ) : Prop := Nat.gcd n d = 1

def valid_fraction (n d : ℕ) : Prop :=
  n ∈ S ∧ d ∈ S ∧ n ≠ d ∧ irreducible n d

theorem fractions_product_one :
  ∃ (n₁ d₁ n₂ d₂ n₃ d₃ : ℕ),
    valid_fraction n₁ d₁ ∧
    valid_fraction n₂ d₂ ∧
    valid_fraction n₃ d₃ ∧
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₁ ≠ d₂ ∧ n₁ ≠ d₃ ∧
    n₂ ≠ n₃ ∧ n₂ ≠ d₁ ∧ n₂ ≠ d₃ ∧
    n₃ ≠ d₁ ∧ n₃ ≠ d₂ ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧
    d₂ ≠ d₃ ∧
    (n₁ : ℚ) / d₁ * (n₂ : ℚ) / d₂ * (n₃ : ℚ) / d₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractions_product_one_l2362_236222


namespace NUMINAMATH_CALUDE_inequality_proof_l2362_236215

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h : (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2*a + b + c) * (2*b + c + d) * (2*c + d + a) * (2*d + a + b) * (a*b*c*d)^2 ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2362_236215


namespace NUMINAMATH_CALUDE_yoojeongs_marbles_l2362_236273

theorem yoojeongs_marbles (marbles_given : ℕ) (marbles_left : ℕ) :
  marbles_given = 8 →
  marbles_left = 24 →
  marbles_given + marbles_left = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_yoojeongs_marbles_l2362_236273


namespace NUMINAMATH_CALUDE_log_equation_solution_l2362_236246

-- Define the logarithm function
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x = 17 ∧ log2 ((3*x + 9) / (5*x - 3)) + log2 ((5*x - 3) / (x - 2)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2362_236246


namespace NUMINAMATH_CALUDE_quadratic_linear_common_solution_l2362_236230

theorem quadratic_linear_common_solution
  (a d : ℝ) (x₁ x₂ e : ℝ) 
  (ha : a ≠ 0)
  (hd : d ≠ 0)
  (hx : x₁ ≠ x₂)
  (h_common : d * x₁ + e = 0)
  (h_unique : ∃! x, a * (x - x₁) * (x - x₂) + d * x + e = 0) :
  a * (x₂ - x₁) = d := by
sorry

end NUMINAMATH_CALUDE_quadratic_linear_common_solution_l2362_236230


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2362_236239

/-- Given that 55 cows eat 55 bags of husk in 55 days, prove that one cow will eat one bag of husk in 55 days. -/
theorem cow_husk_consumption (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 55) 
  (h2 : num_bags = 55) 
  (h3 : num_days = 55) : 
  num_days = 55 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2362_236239


namespace NUMINAMATH_CALUDE_abc_equals_314_l2362_236227

/-- Represents a base-5 number with two digits -/
def BaseFiveNumber (tens : Nat) (ones : Nat) : Nat :=
  5 * tens + ones

/-- Proposition: Given the conditions, ABC = 314 -/
theorem abc_equals_314 
  (A B C : Nat) 
  (h1 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
  (h2 : A < 5 ∧ B < 5 ∧ C < 5)
  (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h4 : BaseFiveNumber A B + C = BaseFiveNumber C 0)
  (h5 : BaseFiveNumber A B + BaseFiveNumber B A = BaseFiveNumber C C) :
  100 * A + 10 * B + C = 314 :=
sorry

end NUMINAMATH_CALUDE_abc_equals_314_l2362_236227


namespace NUMINAMATH_CALUDE_annual_loss_is_14400_l2362_236250

/-- The number of yellow balls in the box -/
def yellow_balls : ℕ := 3

/-- The number of white balls in the box -/
def white_balls : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := yellow_balls + white_balls

/-- The number of balls drawn in each attempt -/
def drawn_balls : ℕ := 3

/-- The reward for drawing 3 balls of the same color (in yuan) -/
def same_color_reward : ℚ := 5

/-- The payment for drawing 3 balls of different colors (in yuan) -/
def diff_color_payment : ℚ := 1

/-- The number of people drawing balls per day -/
def people_per_day : ℕ := 100

/-- The number of days in a year for this calculation -/
def days_per_year : ℕ := 360

/-- The probability of drawing 3 balls of the same color -/
def prob_same_color : ℚ := 1 / 10

/-- The probability of drawing 3 balls of different colors -/
def prob_diff_color : ℚ := 9 / 10

/-- The expected earnings per person (in yuan) -/
def expected_earnings_per_person : ℚ := 
  prob_same_color * same_color_reward - prob_diff_color * diff_color_payment

/-- The daily earnings (in yuan) -/
def daily_earnings : ℚ := expected_earnings_per_person * people_per_day

/-- Theorem: The annual loss is 14400 yuan -/
theorem annual_loss_is_14400 : 
  -daily_earnings * days_per_year = 14400 := by sorry

end NUMINAMATH_CALUDE_annual_loss_is_14400_l2362_236250


namespace NUMINAMATH_CALUDE_eccentricity_product_range_l2362_236299

/-- An ellipse and a hyperbola with common foci -/
structure ConicPair where
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Intersection point
  e₁ : ℝ      -- Eccentricity of ellipse
  e₂ : ℝ      -- Eccentricity of hyperbola

/-- The conditions given in the problem -/
def satisfies_conditions (pair : ConicPair) : Prop :=
  pair.F₁.1 < 0 ∧ pair.F₂.1 > 0 ∧  -- Foci on x-axis, centered at origin
  pair.P.1 > 0 ∧ pair.P.2 > 0 ∧    -- P in first quadrant
  ‖pair.P - pair.F₁‖ = ‖pair.P - pair.F₂‖ ∧  -- Isosceles triangle
  ‖pair.P - pair.F₁‖ = 10 ∧        -- |PF₁| = 10
  pair.e₁ > 0 ∧ pair.e₂ > 0        -- Positive eccentricities

theorem eccentricity_product_range (pair : ConicPair) 
  (h : satisfies_conditions pair) : 
  pair.e₁ * pair.e₂ > 1/3 ∧ 
  ∀ M, ∃ pair', satisfies_conditions pair' ∧ pair'.e₁ * pair'.e₂ > M :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_product_range_l2362_236299


namespace NUMINAMATH_CALUDE_equation_system_solution_l2362_236253

theorem equation_system_solution (x y : ℝ) :
  (2 * x^2 + 6 * x + 4 * y + 2 = 0) →
  (3 * x + y + 4 = 0) →
  (y^2 + 17 * y - 11 = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2362_236253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l2362_236206

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁ = 1 and a₁, a₃, a₁₃ form a geometric sequence,
    prove that the minimum value of (2S_n + 16) / (a_n + 3) is 4,
    where S_n is the sum of the first n terms of {a_n}. -/
theorem arithmetic_sequence_minimum_value (d : ℝ) (n : ℕ) :
  d ≠ 0 →
  let a : ℕ → ℝ := λ k => 1 + (k - 1) * d
  let S : ℕ → ℝ := λ k => k * (a 1 + a k) / 2
  (a 3)^2 = (a 1) * (a 13) →
  (∀ k : ℕ, (2 * S k + 16) / (a k + 3) ≥ 4) ∧
  (∃ k : ℕ, (2 * S k + 16) / (a k + 3) = 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l2362_236206


namespace NUMINAMATH_CALUDE_negation_equivalence_l2362_236251

-- Define the universe of discourse
variable (Student : Type)

-- Define the property of being patient
variable (isPatient : Student → Prop)

-- Statement (6): All students are patient
def allStudentsPatient : Prop := ∀ s : Student, isPatient s

-- Statement (5): At least one student is impatient
def oneStudentImpatient : Prop := ∃ s : Student, ¬(isPatient s)

-- Theorem: Statement (5) is equivalent to the negation of statement (6)
theorem negation_equivalence : oneStudentImpatient Student isPatient ↔ ¬(allStudentsPatient Student isPatient) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2362_236251


namespace NUMINAMATH_CALUDE_square_value_preserving_shifted_square_value_preserving_l2362_236210

-- Define a "value-preserving" interval
def is_value_preserving (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

-- Theorem for f(x) = x^2
theorem square_value_preserving :
  ∀ a b : ℝ, is_value_preserving (fun x => x^2) a b ↔ a = 0 ∧ b = 1 := by sorry

-- Theorem for g(x) = x^2 + m
theorem shifted_square_value_preserving :
  ∀ m : ℝ, m ≠ 0 →
  (∃ a b : ℝ, is_value_preserving (fun x => x^2 + m) a b) ↔
  (m ∈ Set.Icc (-1) (-3/4) ∪ Set.Ioo 0 (1/4)) := by sorry

end NUMINAMATH_CALUDE_square_value_preserving_shifted_square_value_preserving_l2362_236210


namespace NUMINAMATH_CALUDE_eightieth_digit_is_one_l2362_236295

def sequence_digit (n : ℕ) : ℕ :=
  if n ≤ 102 then
    let num := 60 - ((n - 1) / 2)
    if n % 2 = 0 then num % 10 else (num / 10) % 10
  else
    sorry -- Handle single-digit numbers if needed

theorem eightieth_digit_is_one :
  sequence_digit 80 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eightieth_digit_is_one_l2362_236295


namespace NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l2362_236289

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

theorem trailing_zeros_30_factorial :
  trailingZeros (factorial 30) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l2362_236289


namespace NUMINAMATH_CALUDE_maximize_f_l2362_236284

-- Define the function f
def f (a b c x y z : ℝ) : ℝ := a * x + b * y + c * z

-- State the theorem
theorem maximize_f (a b c : ℝ) :
  (∀ x y z : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ -5 ≤ y ∧ y ≤ 5 ∧ -5 ≤ z ∧ z ≤ 5) →
  f a b c 3 1 1 > f a b c 2 1 1 →
  f a b c 2 2 3 > f a b c 2 3 4 →
  f a b c 3 3 4 > f a b c 3 3 3 →
  ∀ x y z : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ -5 ≤ y ∧ y ≤ 5 ∧ -5 ≤ z ∧ z ≤ 5 →
  f a b c x y z ≤ f a b c 5 (-5) 5 :=
by sorry

end NUMINAMATH_CALUDE_maximize_f_l2362_236284


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l2362_236204

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem eighth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 4/3) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ ((a₂ - a₁) : ℚ) 8 = 19/3 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l2362_236204


namespace NUMINAMATH_CALUDE_circle_fixed_points_l2362_236233

theorem circle_fixed_points (m : ℝ) :
  let circle := λ (x y : ℝ) => x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2
  circle 1 1 = 0 ∧ circle (1/5) (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_fixed_points_l2362_236233


namespace NUMINAMATH_CALUDE_answer_key_problem_l2362_236277

theorem answer_key_problem (total_ways : ℕ) (tf_questions : ℕ) (mc_questions : ℕ) : 
  total_ways = 384 → 
  tf_questions = 3 → 
  mc_questions = 3 → 
  (∃ (n : ℕ), total_ways = 6 * n^mc_questions) →
  (∃ (n : ℕ), n = 4 ∧ total_ways = 6 * n^mc_questions) := by
sorry

end NUMINAMATH_CALUDE_answer_key_problem_l2362_236277


namespace NUMINAMATH_CALUDE_roses_to_tulips_ratio_l2362_236229

/-- Represents the number of flowers of each type in the shop -/
structure FlowerShop where
  carnations : ℕ
  violets : ℕ
  tulips : ℕ
  roses : ℕ

/-- Conditions for the flower shop inventory -/
def validFlowerShop (shop : FlowerShop) : Prop :=
  shop.violets = shop.carnations / 3 ∧
  shop.tulips = shop.violets / 4 ∧
  shop.carnations = 2 * (shop.carnations + shop.violets + shop.tulips + shop.roses) / 3

/-- Theorem stating that in a valid flower shop, the ratio of roses to tulips is 1:1 -/
theorem roses_to_tulips_ratio (shop : FlowerShop) (h : validFlowerShop shop) :
  shop.roses = shop.tulips := by
  sorry

#check roses_to_tulips_ratio

end NUMINAMATH_CALUDE_roses_to_tulips_ratio_l2362_236229


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2362_236283

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2362_236283


namespace NUMINAMATH_CALUDE_sports_club_members_l2362_236226

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 48)
  (hT : T = 46)
  (hBoth : Both = 21)
  (hNeither : Neither = 7) :
  (B + T) - Both + Neither = 80 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l2362_236226


namespace NUMINAMATH_CALUDE_grass_seed_bags_l2362_236263

theorem grass_seed_bags (lawn_length lawn_width coverage_per_bag extra_coverage : ℕ) 
  (h1 : lawn_length = 22)
  (h2 : lawn_width = 36)
  (h3 : coverage_per_bag = 250)
  (h4 : extra_coverage = 208) :
  (lawn_length * lawn_width + extra_coverage) / coverage_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_bags_l2362_236263


namespace NUMINAMATH_CALUDE_valid_sampling_interval_l2362_236292

def total_population : ℕ := 102
def removed_individuals : ℕ := 2
def sampling_interval : ℕ := 10

theorem valid_sampling_interval :
  (total_population - removed_individuals) % sampling_interval = 0 := by
  sorry

end NUMINAMATH_CALUDE_valid_sampling_interval_l2362_236292


namespace NUMINAMATH_CALUDE_expression_value_l2362_236209

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 3) :
  c * d + m - (a + b) / m = 4 ∨ c * d + m - (a + b) / m = -2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2362_236209


namespace NUMINAMATH_CALUDE_max_comic_books_l2362_236236

theorem max_comic_books (available : ℚ) (cost_per_book : ℚ) (max_books : ℕ) : 
  available = 12.5 ∧ cost_per_book = 1.15 → max_books = 10 ∧ 
  (max_books : ℚ) * cost_per_book ≤ available ∧
  ∀ n : ℕ, (n : ℚ) * cost_per_book ≤ available → n ≤ max_books :=
by sorry

end NUMINAMATH_CALUDE_max_comic_books_l2362_236236


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2362_236228

-- Define the sample space
def SampleSpace := Finset (Fin 6 × Fin 6)

-- Define the events
def event_W (s : SampleSpace) : Prop := sorry
def event_1 (s : SampleSpace) : Prop := sorry
def event_2 (s : SampleSpace) : Prop := sorry

-- Define mutually exclusive
def mutually_exclusive (A B : SampleSpace → Prop) : Prop :=
  ∀ s : SampleSpace, ¬(A s ∧ B s)

-- Define complementary
def complementary (A B : SampleSpace → Prop) : Prop :=
  ∀ s : SampleSpace, (A s ∨ B s) ∧ ¬(A s ∧ B s)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_W event_1 ∧
  mutually_exclusive event_W event_2 ∧
  ¬complementary event_W event_1 ∧
  ¬complementary event_W event_2 :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2362_236228


namespace NUMINAMATH_CALUDE_set_B_equals_one_two_three_four_l2362_236262

def A : Set Int := {-3, -2, -1, 1, 2, 3, 4}

def f (a : Int) : Int := Int.natAbs a

def B : Set Int := f '' A

theorem set_B_equals_one_two_three_four : B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_B_equals_one_two_three_four_l2362_236262


namespace NUMINAMATH_CALUDE_geoff_total_spending_l2362_236211

/-- Geoff's spending on sneakers over three days -/
def geoff_spending (x : ℝ) : ℝ := x + 4*x + 5*x

/-- Theorem: Geoff's total spending over three days equals 10 times his Monday spending -/
theorem geoff_total_spending (x : ℝ) : geoff_spending x = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_geoff_total_spending_l2362_236211


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2362_236261

/-- An isosceles triangle with specific side lengths -/
structure IsoscelesTriangle where
  /-- Length of equal sides PQ and PR -/
  side : ℝ
  /-- Length of base QR -/
  base : ℝ
  /-- side is positive -/
  side_pos : side > 0
  /-- base is positive -/
  base_pos : base > 0

/-- The area of an isosceles triangle with side length 13 and base length 10 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle)
  (h_side : t.side = 13)
  (h_base : t.base = 10) :
  let height := Real.sqrt (t.side ^ 2 - (t.base / 2) ^ 2)
  (1 / 2 : ℝ) * t.base * height = 60 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2362_236261


namespace NUMINAMATH_CALUDE_point_coordinates_l2362_236296

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define the distance to x-axis
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

-- Define the distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

-- State the theorem
theorem point_coordinates :
  in_fourth_quadrant P ∧
  distance_to_x_axis P = 1 ∧
  distance_to_y_axis P = 2 →
  P = (2, -1) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l2362_236296


namespace NUMINAMATH_CALUDE_parabola_equation_l2362_236247

/-- A parabola with focus (0,1) and vertex at (0,0) has the standard equation x^2 = 4y -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (0, 1)
  let vertex : ℝ × ℝ := (0, 0)
  let p : ℝ := focus.2 - vertex.2
  (x^2 = 4*y) ↔ (
    (∀ (x' y' : ℝ), (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - (focus.2 - p))^2) ∧
    vertex = (0, 0) ∧
    focus = (0, 1)
  ) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2362_236247


namespace NUMINAMATH_CALUDE_farthest_vertex_distance_l2362_236201

/-- Given a rectangle ABCD with area 48 and diagonal 10, and a point O such that OB = OD = 13,
    the distance from O to the farthest vertex of the rectangle is 7√(29/5). -/
theorem farthest_vertex_distance (A B C D O : ℝ × ℝ) : 
  let area := abs ((B.1 - A.1) * (D.2 - A.2))
  let diagonal := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let OB_dist := Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2)
  let OD_dist := Real.sqrt ((O.1 - D.1)^2 + (O.2 - D.2)^2)
  area = 48 ∧ diagonal = 10 ∧ OB_dist = 13 ∧ OD_dist = 13 →
  max (Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2))
      (max (Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2))
           (max (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2))
                (Real.sqrt ((O.1 - D.1)^2 + (O.2 - D.2)^2))))
  = 7 * Real.sqrt (29/5) :=
by
  sorry


end NUMINAMATH_CALUDE_farthest_vertex_distance_l2362_236201


namespace NUMINAMATH_CALUDE_shirt_markup_l2362_236287

theorem shirt_markup (P : ℝ) (h : 2 * P - 1.8 * P = 5) : 1.8 * P = 45 := by
  sorry

end NUMINAMATH_CALUDE_shirt_markup_l2362_236287


namespace NUMINAMATH_CALUDE_breadth_is_ten_l2362_236234

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 20 * breadth
  length_eq : length = breadth + 10

/-- The breadth of a rectangular plot with the given properties is 10 meters -/
theorem breadth_is_ten (plot : RectangularPlot) : plot.breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_breadth_is_ten_l2362_236234


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l2362_236265

theorem sum_of_reciprocals_bound {α β k : ℝ} (hα : α > 0) (hβ : β > 0) (hk : k > 0)
  (hαβ : α ≠ β) (hfα : |Real.log α| = k) (hfβ : |Real.log β| = k) :
  1 / α + 1 / β > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l2362_236265


namespace NUMINAMATH_CALUDE_weekday_hourly_brew_l2362_236214

/-- Represents a coffee shop's brewing schedule and output -/
structure CoffeeShop where
  weekdayHourlyBrew : ℕ
  weekendTotalBrew : ℕ
  dailyHours : ℕ
  weeklyTotalBrew : ℕ

/-- Theorem stating the number of coffee cups brewed per hour on a weekday -/
theorem weekday_hourly_brew (shop : CoffeeShop) 
  (h1 : shop.dailyHours = 5)
  (h2 : shop.weekendTotalBrew = 120)
  (h3 : shop.weeklyTotalBrew = 370) :
  shop.weekdayHourlyBrew = 10 := by
  sorry

end NUMINAMATH_CALUDE_weekday_hourly_brew_l2362_236214


namespace NUMINAMATH_CALUDE_complex_arithmetic_equalities_l2362_236252

theorem complex_arithmetic_equalities :
  (16 / (-2)^3 - (-1/2)^3 * (-4) + 2.5 = 0) ∧
  ((-1)^2022 + |(-2)^2 + 4| - (1/2 - 1/4 + 1/8) * (-24) = 10) := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equalities_l2362_236252


namespace NUMINAMATH_CALUDE_star_equation_two_roots_l2362_236231

/-- Custom binary operation on real numbers -/
def star (a b : ℝ) : ℝ := a * b^2 - b

/-- Theorem stating the condition for the equation 1※x = k to have two distinct real roots -/
theorem star_equation_two_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ star 1 x₁ = k ∧ star 1 x₂ = k) ↔ k > -1/4 :=
sorry

end NUMINAMATH_CALUDE_star_equation_two_roots_l2362_236231


namespace NUMINAMATH_CALUDE_memory_card_capacity_l2362_236221

/-- Proves that a memory card with capacity for 3,000 pictures of 8 megabytes
    can hold 4,000 pictures of 6 megabytes -/
theorem memory_card_capacity 
  (initial_count : Nat) 
  (initial_size : Nat) 
  (new_size : Nat) 
  (h1 : initial_count = 3000)
  (h2 : initial_size = 8)
  (h3 : new_size = 6) :
  (initial_count * initial_size) / new_size = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_memory_card_capacity_l2362_236221


namespace NUMINAMATH_CALUDE_mean_home_runs_l2362_236275

def total_players : ℕ := 12
def players_with_5 : ℕ := 3
def players_with_7 : ℕ := 5
def players_with_9 : ℕ := 3
def players_with_11 : ℕ := 1

def total_home_runs : ℕ := 
  5 * players_with_5 + 7 * players_with_7 + 9 * players_with_9 + 11 * players_with_11

theorem mean_home_runs : 
  (total_home_runs : ℚ) / total_players = 88 / 12 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l2362_236275


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2362_236232

theorem sin_cos_identity (x : ℝ) : 
  (Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x))^2 = 2 - 2 * Real.cos ((2 / 3) * Real.pi - x) ↔ 
  (∃ n : ℤ, x = (2 * Real.pi / 5) * ↑n) ∨ 
  (∃ k : ℤ, x = (2 * Real.pi / 9) * (3 * ↑k + 1)) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2362_236232


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2362_236212

/-- Given a positive geometric sequence {a_n} where a_2, a_3/2, a_1 form an arithmetic sequence,
    prove that (a_4 + a_5) / (a_3 + a_4) = (1 + √5) / 2 -/
theorem geometric_arithmetic_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 2 - a 3 / 2 = a 3 / 2 - a 1) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2362_236212


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l2362_236256

theorem rectangle_dimension_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let L' := 1.3 * L
  let A := L * B
  let A' := 1.885 * A
  ∃ p : ℝ, p > 0 ∧ L' * (B * (1 + p / 100)) = A' ∧ p = 45 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l2362_236256


namespace NUMINAMATH_CALUDE_steven_owes_jeremy_l2362_236207

theorem steven_owes_jeremy (rate : ℚ) (rooms : ℚ) (amount_owed : ℚ) : 
  rate = 9/4 → rooms = 8/5 → amount_owed = rate * rooms → amount_owed = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_steven_owes_jeremy_l2362_236207


namespace NUMINAMATH_CALUDE_linear_function_increasing_condition_l2362_236274

/-- A linear function y = (2m-1)x + 1 -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (2*m - 1)*x + 1

theorem linear_function_increasing_condition 
  (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : y₁ < y₂)
  (h3 : y₁ = linear_function m x₁)
  (h4 : y₂ = linear_function m x₂) :
  m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_condition_l2362_236274


namespace NUMINAMATH_CALUDE_constant_function_l2362_236219

variable (f : ℝ → ℝ)

theorem constant_function
  (h1 : Continuous f')
  (h2 : f 0 = 0)
  (h3 : ∀ x, |f' x| ≤ |f x|) :
  ∃ c, ∀ x, f x = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_l2362_236219


namespace NUMINAMATH_CALUDE_not_always_int_greater_than_decimal_l2362_236243

-- Define a decimal as a structure with an integer part and a fractional part
structure Decimal where
  integerPart : Int
  fractionalPart : Rat
  fractionalPart_lt_one : fractionalPart < 1

-- Define the comparison between an integer and a decimal
def intGreaterThanDecimal (n : Int) (d : Decimal) : Prop :=
  n > d.integerPart + d.fractionalPart

-- Theorem statement
theorem not_always_int_greater_than_decimal :
  ¬ ∀ (n : Int) (d : Decimal), intGreaterThanDecimal n d :=
sorry

end NUMINAMATH_CALUDE_not_always_int_greater_than_decimal_l2362_236243


namespace NUMINAMATH_CALUDE_system_solution_l2362_236268

theorem system_solution :
  ∃ (x y : ℚ), 
    (3 * (x + y) - 4 * (x - y) = 5) ∧
    ((x + y) / 2 + (x - y) / 6 = 0) ∧
    (x = -1/3) ∧ (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2362_236268


namespace NUMINAMATH_CALUDE_final_card_count_l2362_236244

def baseball_card_problem (initial_cards : ℕ) (maria_takes : ℕ → ℕ) (peter_takes : ℕ) (paul_multiplies : ℕ → ℕ) : ℕ :=
  let after_maria := initial_cards - maria_takes initial_cards
  let after_peter := after_maria - peter_takes
  paul_multiplies after_peter

theorem final_card_count :
  baseball_card_problem 15 (fun n => (n + 1) / 2) 1 (fun n => 3 * n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_final_card_count_l2362_236244


namespace NUMINAMATH_CALUDE_completing_square_sum_l2362_236200

theorem completing_square_sum (d e f : ℤ) : 
  d > 0 ∧ 
  (∀ x : ℝ, 25 * x^2 + 30 * x - 24 = 0 ↔ (d * x + e)^2 = f) → 
  d + e + f = 41 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2362_236200


namespace NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l2362_236272

/-- Represents a cricketer's performance --/
structure CricketerPerformance where
  innings : ℕ
  scoreLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the average score after the last inning --/
def averageAfterLastInning (c : CricketerPerformance) : ℚ :=
  let previousAverage := (c.innings - 1 : ℚ) * (c.averageIncrease + (c.scoreLastInning : ℚ) / c.innings)
  (previousAverage + c.scoreLastInning) / c.innings

/-- Theorem stating the cricketer's average after the 22nd inning --/
theorem cricketer_average_after_22nd_inning 
  (c : CricketerPerformance)
  (h1 : c.innings = 22)
  (h2 : c.scoreLastInning = 134)
  (h3 : c.averageIncrease = 7/2) :
  averageAfterLastInning c = 121/2 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l2362_236272


namespace NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l2362_236242

theorem sin_arccos_eight_seventeenths : 
  Real.sin (Real.arccos (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l2362_236242


namespace NUMINAMATH_CALUDE_stick_length_ratio_l2362_236257

/-- Proves that the ratio of the second stick to the first stick is 2:1 given the conditions of the problem -/
theorem stick_length_ratio (stick2 : ℝ) 
  (h1 : 3 + stick2 + (stick2 - 1) = 14) : 
  stick2 / 3 = 2 := by sorry

end NUMINAMATH_CALUDE_stick_length_ratio_l2362_236257


namespace NUMINAMATH_CALUDE_triplets_shirts_l2362_236225

/-- The number of shirts Hazel, Razel, and Gazel have in total -/
def total_shirts (hazel razel gazel : ℕ) : ℕ := hazel + razel + gazel

/-- Theorem stating the total number of shirts given the conditions -/
theorem triplets_shirts : 
  ∀ (hazel razel gazel : ℕ),
  hazel = 6 →
  razel = 2 * hazel →
  gazel = razel / 2 - 1 →
  total_shirts hazel razel gazel = 23 := by
sorry

end NUMINAMATH_CALUDE_triplets_shirts_l2362_236225


namespace NUMINAMATH_CALUDE_odd_divisors_iff_perfect_square_l2362_236203

/-- A number is a perfect square if and only if it has an odd number of divisors -/
theorem odd_divisors_iff_perfect_square (n : ℕ) : 
  Odd (Nat.card {d : ℕ | d ∣ n}) ↔ ∃ k : ℕ, n = k^2 := by
  sorry


end NUMINAMATH_CALUDE_odd_divisors_iff_perfect_square_l2362_236203


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l2362_236297

theorem degree_to_radian_conversion (π : Real) :
  (180 : Real) = π → (300 : Real) * π / 180 = 5 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l2362_236297


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2362_236223

theorem base_conversion_problem :
  ∀ c d : ℕ,
  c < 10 → d < 10 →
  (5 * 6^2 + 2 * 6^1 + 4 * 6^0 = 2 * 10^2 + c * 10^1 + d * 10^0) →
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2362_236223


namespace NUMINAMATH_CALUDE_temperature_conversion_l2362_236291

theorem temperature_conversion (C F : ℝ) : 
  C = (4/7) * (F - 40) → C = 28 → F = 89 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2362_236291


namespace NUMINAMATH_CALUDE_range_of_t_l2362_236249

-- Define the solution set
def solution_set : Set ℤ := {1, 2, 3}

-- Define the inequality condition
def inequality_condition (t : ℝ) (x : ℤ) : Prop :=
  |3 * (x : ℝ) + t| < 4

-- Define the main theorem
theorem range_of_t :
  ∀ t : ℝ,
  (∀ x : ℤ, x ∈ solution_set ↔ inequality_condition t x) →
  -7 < t ∧ t < -5 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l2362_236249


namespace NUMINAMATH_CALUDE_dante_sold_coconuts_l2362_236288

/-- The number of coconuts Paolo has -/
def paolo_coconuts : ℕ := 14

/-- The number of coconuts Dante has relative to Paolo -/
def dante_multiplier : ℕ := 3

/-- The number of coconuts Dante has left after selling -/
def dante_coconuts_left : ℕ := 32

/-- The number of coconuts Dante sold -/
def dante_sold : ℕ := dante_multiplier * paolo_coconuts - dante_coconuts_left

theorem dante_sold_coconuts : dante_sold = 10 := by
  sorry

end NUMINAMATH_CALUDE_dante_sold_coconuts_l2362_236288


namespace NUMINAMATH_CALUDE_probability_of_middle_position_l2362_236208

theorem probability_of_middle_position (n : ℕ) (h : n = 3) :
  (2 : ℚ) / (n.factorial : ℚ) = (1 : ℚ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_middle_position_l2362_236208


namespace NUMINAMATH_CALUDE_albert_betty_age_ratio_l2362_236245

/-- Represents the ages of Albert, Mary, and Betty -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.mary = ages.albert - 22 ∧
  ages.betty = 11

/-- The theorem to prove -/
theorem albert_betty_age_ratio (ages : Ages) :
  age_conditions ages → (ages.albert : ℚ) / ages.betty = 4 := by
  sorry

#check albert_betty_age_ratio

end NUMINAMATH_CALUDE_albert_betty_age_ratio_l2362_236245


namespace NUMINAMATH_CALUDE_tree_planting_equation_l2362_236278

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  (180 / x - 180 / (1.5 * x) = 2) ↔ 
  (∃ (planned_trees actual_trees : ℝ),
    planned_trees = 180 / x ∧
    actual_trees = 180 / (1.5 * x) ∧
    planned_trees - actual_trees = 2 ∧
    180 / x > 2) := by sorry

end NUMINAMATH_CALUDE_tree_planting_equation_l2362_236278


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_not_sufficient_condition_l2362_236217

theorem quadratic_inequality_necessary_condition :
  ∀ x : ℝ, x^2 - 2*x - 3 < 0 → -2 < x ∧ x < 3 :=
by sorry

theorem not_sufficient_condition :
  ∃ x : ℝ, -2 < x ∧ x < 3 ∧ ¬(x^2 - 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_not_sufficient_condition_l2362_236217
