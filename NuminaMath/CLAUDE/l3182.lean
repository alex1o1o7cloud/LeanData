import Mathlib

namespace NUMINAMATH_CALUDE_power_sum_equality_l3182_318227

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3182_318227


namespace NUMINAMATH_CALUDE_supplement_of_supplement_35_l3182_318259

/-- The supplement of an angle is the angle that, when added to the original angle, forms a straight angle (180 degrees). -/
def supplement (angle : ℝ) : ℝ := 180 - angle

/-- Theorem: The supplement of the supplement of a 35-degree angle is 35 degrees. -/
theorem supplement_of_supplement_35 :
  supplement (supplement 35) = 35 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_supplement_35_l3182_318259


namespace NUMINAMATH_CALUDE_intersection_M_N_l3182_318269

def M : Set ℝ := {y | ∃ x, y = x^2}

def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

theorem intersection_M_N :
  (M.prod Set.univ) ∩ N = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ Real.sqrt 2 ∧ p.2 = p.1^2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3182_318269


namespace NUMINAMATH_CALUDE_point_set_is_hyperbola_l3182_318299

-- Define the set of points (x, y) based on the given parametric equations
def point_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≠ 0 ∧ p.1 = (2 * t + 1) / t ∧ p.2 = (t - 2) / t}

-- Theorem stating that the point_set forms a hyperbola
theorem point_set_is_hyperbola : 
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ 
    (∀ p : ℝ × ℝ, p ∈ point_set ↔ 
      a * p.1 * p.1 + b * p.1 * p.2 + c * p.2 * p.2 + d * p.1 + e * p.2 + f = 0) ∧
    b * b - 4 * a * c > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_set_is_hyperbola_l3182_318299


namespace NUMINAMATH_CALUDE_equation_solution_l3182_318207

theorem equation_solution : ∃! x : ℚ, (x - 30) / 3 = (3 * x + 4) / 8 ∧ x = -252 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3182_318207


namespace NUMINAMATH_CALUDE_marksman_probability_l3182_318283

theorem marksman_probability (p10 p9 p8 : ℝ) 
  (h1 : p10 = 0.20)
  (h2 : p9 = 0.30)
  (h3 : p8 = 0.10) :
  1 - (p10 + p9 + p8) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_marksman_probability_l3182_318283


namespace NUMINAMATH_CALUDE_triangle_equations_l3182_318264

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)  -- Side lengths
  (A B C : ℝ)  -- Angles in radians
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- Side lengths are positive
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)  -- Angles are positive
  (h7 : A + B + C = π)  -- Sum of angles is π

-- Define the theorem
theorem triangle_equations (t : Triangle) (h : t.A = π/3) :
  t.a * Real.sin t.C - Real.sqrt 3 * t.c * Real.cos t.A = 0 ∧
  Real.tan (t.A + t.B) * (1 - Real.tan t.A * Real.tan t.B) = (Real.sqrt 3 * t.c) / (t.a * Real.cos t.B) ∧
  Real.sqrt 3 * t.b * Real.sin t.A - t.a * Real.cos t.C = (t.c + t.b) * Real.cos t.A :=
by sorry

end NUMINAMATH_CALUDE_triangle_equations_l3182_318264


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3182_318238

theorem x_value_from_fraction_equality (x y : ℝ) :
  x ≠ 1 →
  y^2 + 3*y - 3 ≠ 0 →
  (x / (x - 1) = (y^2 + 3*y - 2) / (y^2 + 3*y - 3)) →
  x = (y^2 + 3*y - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3182_318238


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3182_318205

theorem sufficient_not_necessary_condition : 
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧ 
  (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3182_318205


namespace NUMINAMATH_CALUDE_two_x_plus_three_equals_nine_l3182_318251

theorem two_x_plus_three_equals_nine (x : ℝ) (h : x = 3) : 2 * x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_x_plus_three_equals_nine_l3182_318251


namespace NUMINAMATH_CALUDE_investment_growth_l3182_318275

theorem investment_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) (final_amount : ℝ) :
  initial_investment = 400 →
  interest_rate = 0.12 →
  years = 5 →
  final_amount = 705.03 →
  initial_investment * (1 + interest_rate) ^ years = final_amount :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l3182_318275


namespace NUMINAMATH_CALUDE_k_value_l3182_318253

def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

theorem k_value : ∀ k : ℕ, A k ∪ B = {1, 2, 3, 5} → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3182_318253


namespace NUMINAMATH_CALUDE_maddy_graduation_time_l3182_318242

/-- The number of semesters Maddy needs to be in college -/
def semesters_needed (total_credits : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ) : ℕ :=
  total_credits / (credits_per_class * classes_per_semester)

/-- Proof that Maddy needs 8 semesters to graduate -/
theorem maddy_graduation_time :
  semesters_needed 120 3 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maddy_graduation_time_l3182_318242


namespace NUMINAMATH_CALUDE_product_of_three_digit_numbers_l3182_318289

theorem product_of_three_digit_numbers : ∃ (I K S : Nat), 
  (I ≠ 0 ∧ K ≠ 0 ∧ S ≠ 0) ∧  -- non-zero digits
  (I ≠ K ∧ K ≠ S ∧ I ≠ S) ∧  -- distinct digits
  (I < 10 ∧ K < 10 ∧ S < 10) ∧  -- single digits
  ((100 * I + 10 * K + S) * (100 * K + 10 * S + I) = 100602) ∧  -- product
  (100602 % 10 = S) ∧  -- ends with S
  (100602 / 100 = I * 10 + K) ∧  -- after removing zeros, IKS remains
  (S = 2 ∧ K = 6 ∧ I = 1)  -- specific values that satisfy the conditions
:= by sorry

end NUMINAMATH_CALUDE_product_of_three_digit_numbers_l3182_318289


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a3_l3182_318297

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 - a 1 = 1 →
  (∀ q : ℝ, q > 0 → a 3 ≤ (a 1) * q^2) →
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a3_l3182_318297


namespace NUMINAMATH_CALUDE_rectangle_width_l3182_318257

/-- Given a rectangular area with a known area and length, prove that its width is 7 feet. -/
theorem rectangle_width (area : ℝ) (length : ℝ) (h1 : area = 35) (h2 : length = 5) :
  area / length = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3182_318257


namespace NUMINAMATH_CALUDE_total_balls_in_bag_l3182_318278

/-- The number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  green : ℕ
  yellow : ℕ
  red : ℕ
  purple : ℕ

/-- The probability of choosing a ball that is neither red nor purple -/
def prob_not_red_or_purple (bag : BagContents) : ℚ :=
  (bag.white + bag.green + bag.yellow : ℚ) / (bag.white + bag.green + bag.yellow + bag.red + bag.purple)

/-- The theorem stating the total number of balls in the bag -/
theorem total_balls_in_bag (bag : BagContents) 
  (h1 : bag.white = 10)
  (h2 : bag.green = 30)
  (h3 : bag.yellow = 10)
  (h4 : bag.red = 47)
  (h5 : bag.purple = 3)
  (h6 : prob_not_red_or_purple bag = 1/2) :
  bag.white + bag.green + bag.yellow + bag.red + bag.purple = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_in_bag_l3182_318278


namespace NUMINAMATH_CALUDE_sarahs_pool_depth_is_five_l3182_318262

/-- The depth of Sarah's pool in feet -/
def sarahs_pool_depth : ℝ := 5

/-- The depth of John's pool in feet -/
def johns_pool_depth : ℝ := 15

/-- Theorem stating that Sarah's pool depth is 5 feet -/
theorem sarahs_pool_depth_is_five :
  sarahs_pool_depth = 5 ∧
  johns_pool_depth = 2 * sarahs_pool_depth + 5 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_pool_depth_is_five_l3182_318262


namespace NUMINAMATH_CALUDE_xy_divides_x_squared_plus_2y_minus_1_l3182_318250

theorem xy_divides_x_squared_plus_2y_minus_1 (x y : ℕ+) :
  (x * y) ∣ (x^2 + 2*y - 1) ↔ 
  ((x = 3 ∧ y = 8) ∨ 
   (x = 5 ∧ y = 8) ∨ 
   (x = 1) ∨ 
   (∃ n : ℕ+, x = 2*n - 1 ∧ y = n)) := by
sorry

end NUMINAMATH_CALUDE_xy_divides_x_squared_plus_2y_minus_1_l3182_318250


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l3182_318272

theorem odd_square_minus_one_div_eight (a : ℤ) (h : ∃ k : ℤ, a = 2 * k + 1) :
  ∃ m : ℤ, a^2 - 1 = 8 * m :=
by sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l3182_318272


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_210_l3182_318224

theorem last_three_digits_of_7_to_210 : 7^210 % 1000 = 599 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_210_l3182_318224


namespace NUMINAMATH_CALUDE_inventory_problem_l3182_318225

/-- The inventory problem -/
theorem inventory_problem
  (ties : ℕ) (belts : ℕ) (black_shirts : ℕ) (white_shirts : ℕ) (hats : ℕ) (socks : ℕ)
  (h_ties : ties = 34)
  (h_belts : belts = 40)
  (h_black_shirts : black_shirts = 63)
  (h_white_shirts : white_shirts = 42)
  (h_hats : hats = 25)
  (h_socks : socks = 80)
  : let jeans := (2 * (black_shirts + white_shirts)) / 3
    let scarves := (ties + belts) / 2
    let jackets := hats + hats / 5
    jeans - (scarves + jackets) = 3 := by
  sorry

end NUMINAMATH_CALUDE_inventory_problem_l3182_318225


namespace NUMINAMATH_CALUDE_toms_seashells_l3182_318260

/-- Calculates the number of unbroken seashells Tom had left after three days of collecting and giving some away. -/
theorem toms_seashells (day1_total day1_broken day2_total day2_broken day3_total day3_broken given_away : ℕ) 
  (h1 : day1_total = 7)
  (h2 : day1_broken = 4)
  (h3 : day2_total = 12)
  (h4 : day2_broken = 5)
  (h5 : day3_total = 15)
  (h6 : day3_broken = 8)
  (h7 : given_away = 3) :
  day1_total - day1_broken + day2_total - day2_broken + day3_total - day3_broken - given_away = 14 := by
  sorry


end NUMINAMATH_CALUDE_toms_seashells_l3182_318260


namespace NUMINAMATH_CALUDE_arrangement_theorem_l3182_318273

/-- The number of ways to arrange 2 teachers and 5 students in a row,
    with the teachers adjacent but not at the ends. -/
def arrangement_count : ℕ := 960

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects,
    where order matters. -/
def permutations_of_k (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem arrangement_theorem :
  arrangement_count = 
    2 * permutations_of_k 5 2 * permutations 4 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l3182_318273


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3182_318213

theorem polynomial_factorization (x : ℝ) : x^4 + 16 = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3182_318213


namespace NUMINAMATH_CALUDE_max_value_of_y_l3182_318244

def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_value_of_y :
  ∃ (α : ℝ), α = 3 ∧ ∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_y_l3182_318244


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3182_318240

theorem point_in_third_quadrant :
  let A : ℝ × ℝ := (Real.sin (2014 * π / 180), Real.cos (2014 * π / 180))
  A.1 < 0 ∧ A.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3182_318240


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l3182_318294

/-- Represents an ellipse with axes of symmetry on the coordinate axes -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > 0 ∧ b > 0 ∧ a ≠ b

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_from_conditions :
  ∀ e : Ellipse,
    e.a + e.b = 9 →  -- Sum of semi-axes is 9 (half of 18)
    e.a^2 - e.b^2 = 9 →  -- Focal distance squared is 9 (6^2 / 4)
    (∀ x y : ℝ, ellipse_equation e x y ↔ 
      (x^2 / 25 + y^2 / 16 = 1 ∨ x^2 / 16 + y^2 / 25 = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l3182_318294


namespace NUMINAMATH_CALUDE_three_lines_two_intersections_l3182_318285

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x - y + 8 = 0
def line3 (a x y : ℝ) : Prop := a*x + 3*y - 5 = 0

-- Define what it means for two points to be distinct
def distinct (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

-- Define what it means for a point to be on a line
def on_line1 (p : ℝ × ℝ) : Prop := line1 p.1 p.2
def on_line2 (p : ℝ × ℝ) : Prop := line2 p.1 p.2
def on_line3 (a : ℝ) (p : ℝ × ℝ) : Prop := line3 a p.1 p.2

-- Theorem statement
theorem three_lines_two_intersections (a : ℝ) :
  (∃ p1 p2 : ℝ × ℝ, distinct p1 p2 ∧ 
    on_line1 p1 ∧ on_line1 p2 ∧ 
    on_line2 p1 ∧ on_line2 p2 ∧ 
    on_line3 a p1 ∧ on_line3 a p2 ∧
    (∀ p3 : ℝ × ℝ, on_line1 p3 ∧ on_line2 p3 ∧ on_line3 a p3 → p3 = p1 ∨ p3 = p2)) →
  a = 3 ∨ a = -6 :=
sorry

end NUMINAMATH_CALUDE_three_lines_two_intersections_l3182_318285


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3182_318215

theorem quadratic_no_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3182_318215


namespace NUMINAMATH_CALUDE_cake_distribution_l3182_318290

theorem cake_distribution (n : ℕ) (most least : ℚ) : 
  most = 1/11 → least = 1/14 → (∀ x, least ≤ x ∧ x ≤ most) → 
  (n : ℚ) * least ≤ 1 ∧ 1 ≤ (n : ℚ) * most → n = 12 ∨ n = 13 := by
  sorry

#check cake_distribution

end NUMINAMATH_CALUDE_cake_distribution_l3182_318290


namespace NUMINAMATH_CALUDE_brown_sugar_amount_l3182_318268

-- Define the amount of white sugar used
def white_sugar : ℝ := 0.25

-- Define the additional amount of brown sugar compared to white sugar
def additional_brown_sugar : ℝ := 0.38

-- Theorem stating the amount of brown sugar used
theorem brown_sugar_amount : 
  white_sugar + additional_brown_sugar = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_brown_sugar_amount_l3182_318268


namespace NUMINAMATH_CALUDE_merchant_profit_l3182_318281

theorem merchant_profit (C S : ℝ) (h : C > 0) (h1 : 18 * C = 16 * S) : 
  (S - C) / C * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l3182_318281


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_097_billion_l3182_318277

theorem scientific_notation_of_1_097_billion :
  ∃ (a : ℝ) (n : ℤ), 1.097e9 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.097 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_097_billion_l3182_318277


namespace NUMINAMATH_CALUDE_projection_magnitude_l3182_318243

def a : Fin 2 → ℝ := ![1, -1]
def b : Fin 2 → ℝ := ![2, -1]

theorem projection_magnitude :
  ‖((a + b) • a / (a • a)) • a‖ = (5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_projection_magnitude_l3182_318243


namespace NUMINAMATH_CALUDE_florist_fertilizer_l3182_318202

def fertilizer_problem (daily_amount : ℕ) (regular_days : ℕ) (extra_amount : ℕ) : Prop :=
  let regular_total := daily_amount * regular_days
  let final_day_amount := daily_amount + extra_amount
  let total_amount := regular_total + final_day_amount
  total_amount = 45

theorem florist_fertilizer :
  fertilizer_problem 3 12 6 := by
  sorry

end NUMINAMATH_CALUDE_florist_fertilizer_l3182_318202


namespace NUMINAMATH_CALUDE_calculation_result_quadratic_solution_l3182_318228

-- Problem 1
theorem calculation_result : Real.sqrt 9 + |1 - Real.sqrt 2| + ((-8 : ℝ) ^ (1/3)) - Real.sqrt 2 = 0 := by
  sorry

-- Problem 2
theorem quadratic_solution (x : ℝ) (h : 4 * x^2 - 16 = 0) : x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_quadratic_solution_l3182_318228


namespace NUMINAMATH_CALUDE_quadrilateral_equal_area_implies_midpoint_l3182_318270

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (E : Point) (quad : Quadrilateral) : Prop := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A B : Point) : Prop := sorry

theorem quadrilateral_equal_area_implies_midpoint 
  (quad : Quadrilateral) (E : Point) :
  isInside E quad →
  (triangleArea E quad.A quad.B = triangleArea E quad.B quad.C) ∧
  (triangleArea E quad.B quad.C = triangleArea E quad.C quad.D) ∧
  (triangleArea E quad.C quad.D = triangleArea E quad.D quad.A) →
  (isMidpoint E quad.A quad.C) ∨ (isMidpoint E quad.B quad.D) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_equal_area_implies_midpoint_l3182_318270


namespace NUMINAMATH_CALUDE_tan_double_angle_l3182_318239

theorem tan_double_angle (α : Real) (h : 3 * Real.cos α + Real.sin α = 0) :
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3182_318239


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3182_318287

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3182_318287


namespace NUMINAMATH_CALUDE_count_values_for_sum_20_main_theorem_l3182_318265

def count_integer_values (n : ℕ) : ℕ :=
  (Finset.filter (fun d => n % d = 0) (Finset.range (n + 1))).card

theorem count_values_for_sum_20 :
  count_integer_values 20 = 6 :=
sorry

theorem main_theorem :
  ∃ (S : Finset ℤ),
    S.card = 6 ∧
    ∀ (a b c : ℕ),
      a > 0 → b > 0 → c > 0 →
      a + b + c = 20 →
      (a + b : ℤ) / (c : ℤ) ∈ S :=
sorry

end NUMINAMATH_CALUDE_count_values_for_sum_20_main_theorem_l3182_318265


namespace NUMINAMATH_CALUDE_average_expenditure_feb_to_july_l3182_318282

/-- Calculates the average expenditure for February to July given the conditions -/
theorem average_expenditure_feb_to_july 
  (avg_jan_to_june : ℝ) 
  (expenditure_jan : ℝ) 
  (expenditure_july : ℝ) 
  (h1 : avg_jan_to_june = 4200)
  (h2 : expenditure_jan = 1200)
  (h3 : expenditure_july = 1500) :
  (6 * avg_jan_to_june - expenditure_jan + expenditure_july) / 6 = 4250 := by
  sorry

#check average_expenditure_feb_to_july

end NUMINAMATH_CALUDE_average_expenditure_feb_to_july_l3182_318282


namespace NUMINAMATH_CALUDE_firefighter_water_delivery_time_l3182_318248

/-- Proves that 5 firefighters can deliver 4000 gallons of water in 40 minutes -/
theorem firefighter_water_delivery_time :
  let water_needed : ℕ := 4000
  let firefighters : ℕ := 5
  let water_per_minute_per_hose : ℕ := 20
  let total_water_per_minute : ℕ := firefighters * water_per_minute_per_hose
  water_needed / total_water_per_minute = 40 := by
  sorry

end NUMINAMATH_CALUDE_firefighter_water_delivery_time_l3182_318248


namespace NUMINAMATH_CALUDE_outermost_to_innermost_ratio_l3182_318266

/-- A sequence of alternating inscribed squares and circles -/
structure SquareCircleSequence where
  S1 : Real  -- Side length of innermost square
  C1 : Real  -- Diameter of circle inscribing S1
  S2 : Real  -- Side length of square inscribing C1
  C2 : Real  -- Diameter of circle inscribing S2
  S3 : Real  -- Side length of square inscribing C2
  C3 : Real  -- Diameter of circle inscribing S3
  S4 : Real  -- Side length of outermost square

/-- Properties of the SquareCircleSequence -/
axiom sequence_properties (seq : SquareCircleSequence) :
  seq.C1 = seq.S1 * Real.sqrt 2 ∧
  seq.S2 = seq.C1 ∧
  seq.C2 = seq.S2 * Real.sqrt 2 ∧
  seq.S3 = seq.C2 ∧
  seq.C3 = seq.S3 * Real.sqrt 2 ∧
  seq.S4 = seq.C3

/-- The ratio of the outermost square's side length to the innermost square's side length is 2√2 -/
theorem outermost_to_innermost_ratio (seq : SquareCircleSequence) :
  seq.S4 / seq.S1 = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_outermost_to_innermost_ratio_l3182_318266


namespace NUMINAMATH_CALUDE_circle_radius_l3182_318222

theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 - 2*x + 4*y = 0 → ∃ (h k r : ℝ), r = Real.sqrt 5 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l3182_318222


namespace NUMINAMATH_CALUDE_min_k_good_is_two_l3182_318291

/-- A function f: ℕ+ → ℕ+ is k-good if for all m ≠ n in ℕ+, (f(m)+n, f(n)+m) ≤ k -/
def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

/-- The minimum k for which a k-good function exists is 2 -/
theorem min_k_good_is_two :
  (∃ k : ℕ, k > 0 ∧ ∃ f : ℕ+ → ℕ+, IsKGood k f) ∧
  (∀ k : ℕ, k > 0 → (∃ f : ℕ+ → ℕ+, IsKGood k f) → k ≥ 2) :=
by sorry

#check min_k_good_is_two

end NUMINAMATH_CALUDE_min_k_good_is_two_l3182_318291


namespace NUMINAMATH_CALUDE_eraser_ratio_is_two_to_one_l3182_318209

-- Define the number of erasers for each person
def tanya_total : ℕ := 20
def hanna_total : ℕ := 4

-- Define the number of red erasers Tanya has
def tanya_red : ℕ := tanya_total / 2

-- Define Rachel's erasers in terms of Tanya's red erasers
def rachel_total : ℕ := tanya_red / 2 - 3

-- Define the ratio of Hanna's erasers to Rachel's erasers
def eraser_ratio : ℚ := hanna_total / rachel_total

-- Theorem to prove
theorem eraser_ratio_is_two_to_one :
  eraser_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_eraser_ratio_is_two_to_one_l3182_318209


namespace NUMINAMATH_CALUDE_suit_price_calculation_l3182_318217

theorem suit_price_calculation (original_price : ℝ) : 
  (original_price * 1.2 * 0.8 = 144) → original_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l3182_318217


namespace NUMINAMATH_CALUDE_equation_D_is_correct_l3182_318271

theorem equation_D_is_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_equation_D_is_correct_l3182_318271


namespace NUMINAMATH_CALUDE_distance_calculation_l3182_318255

/-- Conversion factor from meters to kilometers -/
def meters_to_km : ℝ := 1000

/-- Distance from Xiaoqing's home to the park in meters -/
def total_distance : ℝ := 6000

/-- Distance Xiaoqing has already walked in meters -/
def walked_distance : ℝ := 1200

/-- Theorem stating the conversion of total distance to kilometers and the remaining distance to the park -/
theorem distance_calculation :
  (total_distance / meters_to_km = 6) ∧
  (total_distance - walked_distance = 4800) := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l3182_318255


namespace NUMINAMATH_CALUDE_pair_probability_after_removal_l3182_318296

/-- Represents a deck of cards -/
structure Deck :=
  (size : ℕ)
  (num_fives : ℕ)
  (num_threes : ℕ)

/-- Calculates the number of ways to choose 2 cards from a deck -/
def choose_two (d : Deck) : ℕ := Nat.choose d.size 2

/-- Calculates the number of ways to form pairs in a deck -/
def num_pairs (d : Deck) : ℕ := d.num_fives * Nat.choose 5 2 + d.num_threes * Nat.choose 3 2

/-- The probability of selecting a pair from the deck -/
def pair_probability (d : Deck) : ℚ := (num_pairs d : ℚ) / (choose_two d : ℚ)

theorem pair_probability_after_removal :
  let d : Deck := ⟨46, 4, 2⟩
  pair_probability d = 46 / 1035 :=
sorry

end NUMINAMATH_CALUDE_pair_probability_after_removal_l3182_318296


namespace NUMINAMATH_CALUDE_value_of_a_l3182_318230

theorem value_of_a : ∃ (a : ℝ), 
  (∃ (x y : ℝ), 2*x + y = 3*a ∧ x - 2*y = 9*a ∧ x + 3*y = 24) → 
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3182_318230


namespace NUMINAMATH_CALUDE_group_size_l3182_318216

/-- 
Given a group of people with men, women, and children, where:
- The number of men is twice the number of women
- The number of women is 3 times the number of children
- The number of children is 30

Prove that the total number of people in the group is 300.
-/
theorem group_size (children women men : ℕ) 
  (h1 : men = 2 * women) 
  (h2 : women = 3 * children) 
  (h3 : children = 30) : 
  children + women + men = 300 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l3182_318216


namespace NUMINAMATH_CALUDE_equation_solution_l3182_318288

theorem equation_solution : ∃! x : ℚ, (3 / 20 + 3 / x = 8 / x + 1 / 15) ∧ x = 60 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3182_318288


namespace NUMINAMATH_CALUDE_average_monthly_balance_l3182_318267

def monthly_balances : List ℝ := [200, 300, 250, 350, 300]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 280 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l3182_318267


namespace NUMINAMATH_CALUDE_square_pyramid_components_l3182_318261

/-- The number of rows in the square pyramid -/
def num_rows : ℕ := 10

/-- The number of unit rods in the first row -/
def first_row_rods : ℕ := 4

/-- The number of additional rods in each subsequent row -/
def additional_rods_per_row : ℕ := 4

/-- Calculate the total number of unit rods in the pyramid -/
def total_rods (n : ℕ) : ℕ :=
  first_row_rods * n * (n + 1) / 2

/-- Calculate the number of internal connectors -/
def internal_connectors (n : ℕ) : ℕ :=
  4 * (n * (n - 1) / 2)

/-- Calculate the number of vertical connectors -/
def vertical_connectors (n : ℕ) : ℕ :=
  4 * (n - 1)

/-- The total number of connectors -/
def total_connectors (n : ℕ) : ℕ :=
  internal_connectors n + vertical_connectors n

/-- The main theorem: proving the total number of unit rods and connectors -/
theorem square_pyramid_components :
  total_rods num_rows + total_connectors num_rows = 436 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_components_l3182_318261


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3182_318252

/-- Theorem: For a right circular cone and a sphere with the same radius,
    if the volume of the cone is one-third that of the sphere,
    then the ratio of the altitude of the cone to its base radius is 4/3. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3) * ((4 / 3) * π * r^3) = (1 / 3) * π * r^2 * h →
  h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3182_318252


namespace NUMINAMATH_CALUDE_zeros_count_theorem_l3182_318206

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_unique_zero_in_interval (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f c = 0 ∧ ∀ x, x ∈ Set.Icc a b → f x = 0 → x = c

def count_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem zeros_count_theorem (f : ℝ → ℝ) :
  is_even_function f →
  (∀ x, f (5 + x) = f (5 - x)) →
  has_unique_zero_in_interval f 0 5 1 →
  count_zeros_in_interval f (-2012) 2012 = 806 :=
sorry

end NUMINAMATH_CALUDE_zeros_count_theorem_l3182_318206


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3182_318235

/-- Represents a right triangle with acute angles in the ratio 5:4 -/
structure RightTriangle where
  /-- First acute angle in degrees -/
  angle1 : ℝ
  /-- Second acute angle in degrees -/
  angle2 : ℝ
  /-- The triangle is a right triangle -/
  is_right_triangle : angle1 + angle2 = 90
  /-- The ratio of acute angles is 5:4 -/
  angle_ratio : angle1 / angle2 = 5 / 4

/-- Theorem: In a right triangle where the ratio of acute angles is 5:4,
    the measures of these angles are 50° and 40° -/
theorem right_triangle_acute_angles (t : RightTriangle) : 
  t.angle1 = 50 ∧ t.angle2 = 40 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3182_318235


namespace NUMINAMATH_CALUDE_function_inequality_l3182_318214

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x < (deriv^[2] f) x) : 
  f 1 > ℯ * f 0 ∧ f 2019 > ℯ^2019 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3182_318214


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l3182_318245

-- Define the basic geometric objects
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_to_line_are_parallel
  (α β : Plane) (m : Line) (h_diff : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β := by sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l3182_318245


namespace NUMINAMATH_CALUDE_max_y_value_l3182_318246

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : 
  ∀ (z : ℤ), z * x + 3 * x + 2 * z ≠ -4 ∨ z ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l3182_318246


namespace NUMINAMATH_CALUDE_prob_white_glow_pop_is_12_21_l3182_318256

/-- Represents the color of a kernel -/
inductive KernelColor
| White
| Yellow

/-- Represents the properties of kernels in the bag -/
structure KernelProperties where
  totalWhite : Rat
  totalYellow : Rat
  whiteGlow : Rat
  yellowGlow : Rat
  whiteGlowPop : Rat
  yellowGlowPop : Rat

/-- The given properties of the kernels in the bag -/
def bagProperties : KernelProperties :=
  { totalWhite := 3/4
  , totalYellow := 1/4
  , whiteGlow := 1/2
  , yellowGlow := 3/4
  , whiteGlowPop := 1/2
  , yellowGlowPop := 3/4
  }

/-- The probability that a randomly selected kernel that glows and pops is white -/
def probWhiteGlowPop (props : KernelProperties) : Rat :=
  let whiteGlowPop := props.totalWhite * props.whiteGlow * props.whiteGlowPop
  let yellowGlowPop := props.totalYellow * props.yellowGlow * props.yellowGlowPop
  whiteGlowPop / (whiteGlowPop + yellowGlowPop)

/-- Theorem stating that the probability of selecting a white kernel that glows and pops is 12/21 -/
theorem prob_white_glow_pop_is_12_21 :
  probWhiteGlowPop bagProperties = 12/21 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_glow_pop_is_12_21_l3182_318256


namespace NUMINAMATH_CALUDE_ceiling_floor_expression_l3182_318204

theorem ceiling_floor_expression : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_expression_l3182_318204


namespace NUMINAMATH_CALUDE_fibonacci_fifth_divisible_by_five_l3182_318274

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_fifth_divisible_by_five (k : ℕ) :
  5 ∣ fibonacci (5 * k) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_fifth_divisible_by_five_l3182_318274


namespace NUMINAMATH_CALUDE_erased_number_proof_l3182_318284

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n ∧ x ≥ 1 →
  (n * (n + 1) / 2 - x) / (n - 1) = 614 / 17 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l3182_318284


namespace NUMINAMATH_CALUDE_median_and_mode_of_S_l3182_318221

/-- The set of data --/
def S : Finset ℕ := {6, 7, 4, 7, 5, 2}

/-- Definition of median for a finite set of natural numbers --/
def median (s : Finset ℕ) : ℚ := sorry

/-- Definition of mode for a finite set of natural numbers --/
def mode (s : Finset ℕ) : ℕ := sorry

theorem median_and_mode_of_S :
  median S = 5.5 ∧ mode S = 7 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_of_S_l3182_318221


namespace NUMINAMATH_CALUDE_contractor_male_workers_l3182_318231

/-- Represents the number of male workers employed by the contractor. -/
def male_workers : ℕ := sorry

/-- Represents the number of female workers employed by the contractor. -/
def female_workers : ℕ := 15

/-- Represents the number of child workers employed by the contractor. -/
def child_workers : ℕ := 5

/-- Represents the daily wage of a male worker in Rupees. -/
def male_wage : ℕ := 35

/-- Represents the daily wage of a female worker in Rupees. -/
def female_wage : ℕ := 20

/-- Represents the daily wage of a child worker in Rupees. -/
def child_wage : ℕ := 8

/-- Represents the average daily wage paid by the contractor in Rupees. -/
def average_wage : ℕ := 26

/-- Theorem stating that the number of male workers employed by the contractor is 20. -/
theorem contractor_male_workers :
  (male_workers * male_wage + female_workers * female_wage + child_workers * child_wage) /
  (male_workers + female_workers + child_workers) = average_wage →
  male_workers = 20 := by
  sorry

end NUMINAMATH_CALUDE_contractor_male_workers_l3182_318231


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3182_318279

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {3, 4, 5}

theorem complement_union_theorem :
  (Set.compl A ∩ U) ∪ B = {1, 3, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3182_318279


namespace NUMINAMATH_CALUDE_irrational_sum_product_theorem_l3182_318200

theorem irrational_sum_product_theorem (a : ℝ) (h : Irrational a) :
  ∃ (b b' : ℝ), Irrational b ∧ Irrational b' ∧
    (¬ Irrational (a + b)) ∧
    (¬ Irrational (a * b')) ∧
    (Irrational (a * b)) ∧
    (Irrational (a + b')) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sum_product_theorem_l3182_318200


namespace NUMINAMATH_CALUDE_quiz_answer_key_l3182_318208

theorem quiz_answer_key (n : ℕ) : 
  (14 * n^2 = 224) → n = 4 :=
by
  sorry

#check quiz_answer_key

end NUMINAMATH_CALUDE_quiz_answer_key_l3182_318208


namespace NUMINAMATH_CALUDE_tony_temp_day5_l3182_318286

-- Define the illnesses and their effects
structure Illness where
  duration : ℕ
  tempChange : ℤ
  startDay : ℕ

-- Define Tony's normal temperature and fever threshold
def normalTemp : ℕ := 95
def feverThreshold : ℕ := 100

-- Define the illnesses
def illnessA : Illness := ⟨7, 10, 1⟩
def illnessB : Illness := ⟨5, 4, 3⟩
def illnessC : Illness := ⟨3, -2, 5⟩

-- Function to calculate temperature change on a given day
def tempChangeOnDay (day : ℕ) : ℤ :=
  let baseChange := 
    (if day ≥ illnessA.startDay then illnessA.tempChange else 0) +
    (if day ≥ illnessB.startDay then 
      (if day ≥ illnessA.startDay then 2 * illnessB.tempChange else illnessB.tempChange)
    else 0) +
    (if day ≥ illnessC.startDay then illnessC.tempChange else 0)
  let synergisticEffect := if day = 5 then -3 else 0
  baseChange + synergisticEffect

-- Theorem to prove
theorem tony_temp_day5 : 
  (normalTemp : ℤ) + tempChangeOnDay 5 = 108 ∧ 
  (normalTemp : ℤ) + tempChangeOnDay 5 - feverThreshold = 8 := by
  sorry

end NUMINAMATH_CALUDE_tony_temp_day5_l3182_318286


namespace NUMINAMATH_CALUDE_three_sevenths_decomposition_l3182_318293

theorem three_sevenths_decomposition :
  3 / 7 = 1 / 8 + 1 / 56 + 1 / 9 + 1 / 72 := by
  sorry

#check three_sevenths_decomposition

end NUMINAMATH_CALUDE_three_sevenths_decomposition_l3182_318293


namespace NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l3182_318258

/-- The set V_n of integers of the form 1 + kn where k ≥ 1 -/
def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, k ≥ 1 ∧ m = 1 + k * n}

/-- A number is indecomposable in V_n if it can't be expressed as a product of two numbers from V_n -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → m ≠ p * q

/-- Two lists of natural numbers are considered different if they are not permutations of each other -/
def DifferentFactorizations (l1 l2 : List ℕ) : Prop :=
  ¬(l1.Perm l2)

theorem existence_of_multiple_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ l1 l2 : List ℕ,
      (∀ x ∈ l1, Indecomposable n x) ∧
      (∀ x ∈ l2, Indecomposable n x) ∧
      (r = l1.prod) ∧
      (r = l2.prod) ∧
      DifferentFactorizations l1 l2 :=
sorry


end NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l3182_318258


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l3182_318211

/-- Represents the sample sizes for three districts -/
structure DistrictSamples where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ

/-- Given a population divided into three districts with a ratio of 2:3:5,
    and a maximum sample size of 60 for any district,
    prove that the total sample size is 120. -/
theorem stratified_sampling_size :
  ∀ (s : DistrictSamples),
  (s.d1 : ℚ) / 2 = s.d2 / 3 ∧
  (s.d1 : ℚ) / 2 = s.d3 / 5 ∧
  s.d3 ≤ 60 ∧
  s.d3 = 60 →
  s.d1 + s.d2 + s.d3 = 120 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_size_l3182_318211


namespace NUMINAMATH_CALUDE_train_passing_time_l3182_318241

/-- The time taken for a slower train to pass the driver of a faster train -/
theorem train_passing_time (length : ℝ) (speed_fast speed_slow : ℝ) :
  length = 500 →
  speed_fast = 45 →
  speed_slow = 30 →
  let relative_speed := speed_fast + speed_slow
  let relative_speed_ms := relative_speed * 1000 / 3600
  let time := length / relative_speed_ms
  ∃ ε > 0, |time - 24| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3182_318241


namespace NUMINAMATH_CALUDE_blue_pick_fraction_l3182_318229

def guitar_pick_collection (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : Prop :=
  red + blue + yellow = total ∧ red = total / 2 ∧ blue = 12 ∧ yellow = 6

theorem blue_pick_fraction 
  (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h : guitar_pick_collection total red blue yellow) : 
  blue = total / 3 := by
sorry

end NUMINAMATH_CALUDE_blue_pick_fraction_l3182_318229


namespace NUMINAMATH_CALUDE_lindas_furniture_spending_l3182_318254

theorem lindas_furniture_spending (savings : ℚ) (tv_cost : ℚ) (furniture_fraction : ℚ) :
  savings = 840 →
  tv_cost = 210 →
  furniture_fraction * savings + tv_cost = savings →
  furniture_fraction = 3/4 := by
sorry

end NUMINAMATH_CALUDE_lindas_furniture_spending_l3182_318254


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3182_318263

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3182_318263


namespace NUMINAMATH_CALUDE_sin_X_in_right_triangle_l3182_318203

-- Define the right triangle XYZ
def RightTriangle (X Y Z : ℝ) : Prop :=
  0 < X ∧ 0 < Y ∧ 0 < Z ∧ X^2 + Y^2 = Z^2

-- State the theorem
theorem sin_X_in_right_triangle :
  ∀ X Y Z : ℝ,
  RightTriangle X Y Z →
  X = 8 →
  Z = 17 →
  Real.sin (Real.arcsin (X / Z)) = 8 / 17 :=
by sorry

end NUMINAMATH_CALUDE_sin_X_in_right_triangle_l3182_318203


namespace NUMINAMATH_CALUDE_tomatoes_count_l3182_318280

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The difference between the number of students who suggested mashed potatoes
    and the number of students who suggested tomatoes -/
def difference : ℕ := 65

/-- The number of students who suggested adding tomatoes -/
def tomatoes : ℕ := mashed_potatoes - difference

theorem tomatoes_count : tomatoes = 79 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_count_l3182_318280


namespace NUMINAMATH_CALUDE_salary_problem_l3182_318223

/-- The average monthly salary of employees in an organization -/
def average_salary (num_employees : ℕ) (total_salary : ℕ) : ℚ :=
  total_salary / num_employees

/-- The problem statement -/
theorem salary_problem (initial_total_salary : ℕ) :
  let num_employees : ℕ := 20
  let manager_salary : ℕ := 3300
  let new_average : ℚ := average_salary (num_employees + 1) (initial_total_salary + manager_salary)
  let initial_average : ℚ := average_salary num_employees initial_total_salary
  new_average = initial_average + 100 →
  initial_average = 1200 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l3182_318223


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3182_318276

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (D : ℝ), D = 2 * Real.sqrt 3 ∧
  (∀ (x y : ℝ), 2 * x^2 + 2 * y^2 + 3 ≥ D * (x + y)) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), 2 * x^2 + 2 * y^2 + 3 ≥ D' * (x + y)) → D' ≤ D) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3182_318276


namespace NUMINAMATH_CALUDE_congruence_from_power_difference_l3182_318233

theorem congruence_from_power_difference (a b : ℕ+) (h : a^b.val - b^a.val = 1008) :
  a ≡ b [ZMOD 1008] := by
  sorry

end NUMINAMATH_CALUDE_congruence_from_power_difference_l3182_318233


namespace NUMINAMATH_CALUDE_project_workers_needed_l3182_318201

/-- Represents a construction project with workers -/
structure Project where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completionRatio : ℚ
  
/-- Calculates the minimum number of workers needed to complete the project on schedule -/
def minWorkersNeeded (p : Project) : ℕ :=
  sorry

/-- The theorem stating the minimum number of workers needed for the specific project -/
theorem project_workers_needed :
  let p : Project := {
    totalDays := 40,
    elapsedDays := 10,
    initialWorkers := 10,
    completionRatio := 2/5
  }
  minWorkersNeeded p = 5 := by sorry

end NUMINAMATH_CALUDE_project_workers_needed_l3182_318201


namespace NUMINAMATH_CALUDE_prism_surface_area_l3182_318247

/-- A right rectangular prism with integer dimensions -/
structure RectPrism where
  l : ℕ
  w : ℕ
  h : ℕ
  l_ne_w : l ≠ w
  w_ne_h : w ≠ h
  h_ne_l : h ≠ l

/-- The processing fee calculation function -/
def processingFee (p : RectPrism) : ℚ :=
  0.3 * p.l + 0.4 * p.w + 0.5 * p.h

/-- The surface area calculation function -/
def surfaceArea (p : RectPrism) : ℕ :=
  2 * (p.l * p.w + p.l * p.h + p.w * p.h)

/-- The main theorem -/
theorem prism_surface_area (p : RectPrism) :
  (∃ (σ₁ σ₂ σ₃ σ₄ : Equiv.Perm (Fin 3)),
    3 * (σ₁.toFun 0 : ℕ) + 4 * (σ₁.toFun 1 : ℕ) + 5 * (σ₁.toFun 2 : ℕ) = 81 ∧
    3 * (σ₂.toFun 0 : ℕ) + 4 * (σ₂.toFun 1 : ℕ) + 5 * (σ₂.toFun 2 : ℕ) = 81 ∧
    3 * (σ₃.toFun 0 : ℕ) + 4 * (σ₃.toFun 1 : ℕ) + 5 * (σ₃.toFun 2 : ℕ) = 87 ∧
    3 * (σ₄.toFun 0 : ℕ) + 4 * (σ₄.toFun 1 : ℕ) + 5 * (σ₄.toFun 2 : ℕ) = 87) →
  surfaceArea p = 276 := by
  sorry


end NUMINAMATH_CALUDE_prism_surface_area_l3182_318247


namespace NUMINAMATH_CALUDE_area_of_removed_triangles_l3182_318295

theorem area_of_removed_triangles (side_length : ℝ) (hypotenuse : ℝ) : 
  side_length = 16 → hypotenuse = 8 → 
  4 * (1/2 * (hypotenuse^2 / 2)) = 64 := by
  sorry

end NUMINAMATH_CALUDE_area_of_removed_triangles_l3182_318295


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l3182_318237

/-- A function that moves the last digit of a number to the beginning -/
def moveLastDigitToFront (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let remainingDigits := n / 10
  lastDigit * 10^5 + remainingDigits

/-- A predicate that checks if a number becomes an integer multiple when its last digit is moved to the front -/
def isValidNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, moveLastDigitToFront n = k * n

/-- The set of all valid six-digit numbers -/
def validNumbers : Finset ℕ :=
  {142857, 102564, 128205, 153846, 179487, 205128, 230769}

/-- The main theorem stating that validNumbers contains all and only the six-digit numbers
    that become an integer multiple when the last digit is moved to the beginning -/
theorem valid_numbers_characterization :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 →
    (n ∈ validNumbers ↔ isValidNumber n) := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l3182_318237


namespace NUMINAMATH_CALUDE_power_of_product_l3182_318218

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l3182_318218


namespace NUMINAMATH_CALUDE_raspberry_pie_degrees_is_45_l3182_318220

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The total number of students in Mandy's class -/
def total_students : ℕ := 48

/-- The number of students preferring chocolate pie -/
def chocolate_preference : ℕ := 18

/-- The number of students preferring apple pie -/
def apple_preference : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_preference : ℕ := 8

/-- Calculate the number of degrees for raspberry pie in the pie chart -/
def raspberry_pie_degrees : ℚ :=
  let remaining_students := total_students - (chocolate_preference + apple_preference + blueberry_preference)
  let raspberry_preference := remaining_students / 2
  (raspberry_preference : ℚ) / total_students * full_circle

/-- Theorem stating that the number of degrees for raspberry pie is 45° -/
theorem raspberry_pie_degrees_is_45 : raspberry_pie_degrees = 45 := by
  sorry

end NUMINAMATH_CALUDE_raspberry_pie_degrees_is_45_l3182_318220


namespace NUMINAMATH_CALUDE_min_weighings_is_three_l3182_318232

/-- Represents a collection of coins with two adjacent lighter coins. -/
structure CoinCollection where
  n : ℕ
  light_weight : ℕ
  heavy_weight : ℕ
  
/-- Represents a weighing operation on a subset of coins. -/
def Weighing (cc : CoinCollection) (subset : Finset ℕ) : ℕ := sorry

/-- The minimum number of weighings required to identify the two lighter coins. -/
def min_weighings (cc : CoinCollection) : ℕ := sorry

/-- Theorem stating that the minimum number of weighings is 3 for any valid coin collection. -/
theorem min_weighings_is_three (cc : CoinCollection) 
  (h1 : cc.n ≥ 2) 
  (h2 : cc.light_weight = 9) 
  (h3 : cc.heavy_weight = 10) :
  min_weighings cc = 3 := by sorry

end NUMINAMATH_CALUDE_min_weighings_is_three_l3182_318232


namespace NUMINAMATH_CALUDE_cookie_sales_problem_l3182_318236

/-- Represents the number of boxes of cookies sold -/
structure CookieSales where
  chocolate : ℕ
  plain : ℕ

/-- Represents the price of cookies in cents -/
def CookiePrice : ℕ × ℕ := (125, 75)

theorem cookie_sales_problem (sales : CookieSales) : 
  sales.chocolate + sales.plain = 1585 →
  125 * sales.chocolate + 75 * sales.plain = 158675 →
  sales.plain = 789 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_problem_l3182_318236


namespace NUMINAMATH_CALUDE_curve_tangent_theorem_l3182_318226

/-- A curve defined by y = x² + ax + b -/
def curve (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2*x + a

/-- The tangent line at x = 0 -/
def tangent_at_zero (a b : ℝ) : ℝ → ℝ := λ x ↦ 3*x - b + 1

theorem curve_tangent_theorem (a b : ℝ) :
  (∀ x, tangent_at_zero a b x = 3*x - (curve a b 0) + 1) →
  curve_derivative a 0 = 3 →
  a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_theorem_l3182_318226


namespace NUMINAMATH_CALUDE_fabric_cost_difference_l3182_318298

/-- The amount of fabric Kenneth bought in ounces -/
def kenneth_fabric : ℝ := 700

/-- The price per ounce of fabric in dollars -/
def price_per_oz : ℝ := 40

/-- The amount of fabric Nicholas bought in ounces -/
def nicholas_fabric : ℝ := 6 * kenneth_fabric

/-- The total cost of Kenneth's fabric in dollars -/
def kenneth_cost : ℝ := kenneth_fabric * price_per_oz

/-- The total cost of Nicholas's fabric in dollars -/
def nicholas_cost : ℝ := nicholas_fabric * price_per_oz

/-- The difference in cost between Nicholas's and Kenneth's fabric purchases -/
theorem fabric_cost_difference : nicholas_cost - kenneth_cost = 140000 := by
  sorry

end NUMINAMATH_CALUDE_fabric_cost_difference_l3182_318298


namespace NUMINAMATH_CALUDE_newspaper_coupon_free_tickets_l3182_318210

/-- Represents the amusement park scenario --/
structure AmusementPark where
  ferris_wheel_cost : ℝ
  roller_coaster_cost : ℝ
  multiple_ride_discount : ℝ
  tickets_bought : ℝ

/-- Calculates the number of free tickets from the newspaper coupon --/
def free_tickets (park : AmusementPark) : ℝ :=
  park.ferris_wheel_cost + park.roller_coaster_cost - park.multiple_ride_discount - park.tickets_bought

/-- Theorem stating that the number of free tickets is 1 given the specific conditions --/
theorem newspaper_coupon_free_tickets :
  ∀ (park : AmusementPark),
    park.ferris_wheel_cost = 2 →
    park.roller_coaster_cost = 7 →
    park.multiple_ride_discount = 1 →
    park.tickets_bought = 7 →
    free_tickets park = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_newspaper_coupon_free_tickets_l3182_318210


namespace NUMINAMATH_CALUDE_max_sum_of_exponents_l3182_318219

theorem max_sum_of_exponents (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) :
  x + y ≤ -2 ∧ ∃ (a b : ℝ), (2 : ℝ)^a + (2 : ℝ)^b = 1 ∧ a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_exponents_l3182_318219


namespace NUMINAMATH_CALUDE_anne_twice_sister_height_l3182_318292

/-- Represents the heights of Anne, her sister, and Bella -/
structure Heights where
  anne : ℝ
  sister : ℝ
  bella : ℝ

/-- The conditions of the problem -/
def HeightConditions (h : Heights) : Prop :=
  ∃ (n : ℝ),
    h.anne = n * h.sister ∧
    h.bella = 3 * h.anne ∧
    h.anne = 80 ∧
    h.bella - h.sister = 200

/-- The theorem stating that under the given conditions, 
    Anne's height is twice her sister's height -/
theorem anne_twice_sister_height (h : Heights) 
  (hc : HeightConditions h) : h.anne = 2 * h.sister := by
  sorry

end NUMINAMATH_CALUDE_anne_twice_sister_height_l3182_318292


namespace NUMINAMATH_CALUDE_complete_square_transformation_l3182_318249

theorem complete_square_transformation (x : ℝ) : 
  x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_transformation_l3182_318249


namespace NUMINAMATH_CALUDE_expand_expression_l3182_318212

theorem expand_expression (y : ℝ) : 5 * (y + 3) * (y - 2) * (y + 1) = 5 * y^3 + 10 * y^2 - 25 * y - 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3182_318212


namespace NUMINAMATH_CALUDE_investment_growth_l3182_318234

/-- Calculates the total amount after compound interest is applied --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth :
  let initial_investment : ℝ := 300
  let monthly_rate : ℝ := 0.1
  let months : ℕ := 2
  compound_interest initial_investment monthly_rate months = 363 := by
sorry

end NUMINAMATH_CALUDE_investment_growth_l3182_318234
