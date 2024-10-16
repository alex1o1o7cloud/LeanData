import Mathlib

namespace NUMINAMATH_CALUDE_magician_payment_calculation_l2348_234845

/-- The total amount paid to a magician given their hourly rate, daily hours, and number of weeks worked -/
def magician_payment (hourly_rate : ℕ) (daily_hours : ℕ) (weeks : ℕ) : ℕ :=
  hourly_rate * daily_hours * 7 * weeks

/-- Theorem stating that a magician charging $60 per hour, working 3 hours daily for 2 weeks, earns $2520 -/
theorem magician_payment_calculation :
  magician_payment 60 3 2 = 2520 := by
  sorry

#eval magician_payment 60 3 2

end NUMINAMATH_CALUDE_magician_payment_calculation_l2348_234845


namespace NUMINAMATH_CALUDE_percentage_of_green_leaves_l2348_234896

/-- Given a collection of leaves with known properties, prove the percentage of green leaves. -/
theorem percentage_of_green_leaves 
  (total_leaves : ℕ) 
  (brown_percentage : ℚ) 
  (yellow_leaves : ℕ) 
  (h1 : total_leaves = 25)
  (h2 : brown_percentage = 1/5)
  (h3 : yellow_leaves = 15) :
  (total_leaves - (brown_percentage * total_leaves).num - yellow_leaves : ℚ) / total_leaves = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_green_leaves_l2348_234896


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2348_234843

/-- Given a function f: ℝ → ℝ with a tangent line y = -x + 6 at x=2, 
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 2 + (deriv f 2) * (x - 2) = -x + 6) : 
    f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2348_234843


namespace NUMINAMATH_CALUDE_parametric_eq_normal_l2348_234854

/-- The parametric equation of a plane -/
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2*s - 3*t, 4 - s + 2*t, 1 - 3*s - t)

/-- The normal form equation of a plane -/
def plane_normal (x y z : ℝ) : Prop :=
  5*x + 11*y + 7*z - 61 = 0

/-- Theorem stating that the parametric and normal form equations represent the same plane -/
theorem parametric_eq_normal :
  ∀ (x y z : ℝ), (∃ (s t : ℝ), plane_parametric s t = (x, y, z)) ↔ plane_normal x y z :=
by sorry

end NUMINAMATH_CALUDE_parametric_eq_normal_l2348_234854


namespace NUMINAMATH_CALUDE_internet_cost_comparison_l2348_234888

/-- Cost calculation for dial-up internet access -/
def dialup_cost (hours : ℝ) : ℝ := 4.2 * hours

/-- Cost calculation for monthly subscription -/
def subscription_cost : ℝ := 130 - 25

/-- The number of hours where both methods cost the same -/
def equal_cost_hours : ℝ := 25

theorem internet_cost_comparison :
  /- Part 1: Prove that costs are equal at 25 hours -/
  dialup_cost equal_cost_hours = subscription_cost ∧
  /- Part 2: Prove that subscription is cheaper for 30 hours -/
  dialup_cost 30 > subscription_cost := by
  sorry

#check internet_cost_comparison

end NUMINAMATH_CALUDE_internet_cost_comparison_l2348_234888


namespace NUMINAMATH_CALUDE_range_of_a_in_linear_program_l2348_234830

/-- The range of values for a given the specified constraints and maximum point -/
theorem range_of_a_in_linear_program (x y a : ℝ) : 
  (1 ≤ x + y) → (x + y ≤ 4) → 
  (-2 ≤ x - y) → (x - y ≤ 2) → 
  (a > 0) →
  (∀ x' y', (1 ≤ x' + y') → (x' + y' ≤ 4) → (-2 ≤ x' - y') → (x' - y' ≤ 2) → 
    (a * x' + y' ≤ a * x + y)) →
  (x = 3 ∧ y = 1) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_linear_program_l2348_234830


namespace NUMINAMATH_CALUDE_rectangular_shape_perimeter_and_area_l2348_234801

/-- A rectangular shape composed of 5 cm segments -/
structure RectangularShape where
  segmentLength : ℝ
  length : ℝ
  height : ℝ

/-- Calculate the perimeter of the rectangular shape -/
def perimeter (shape : RectangularShape) : ℝ :=
  2 * (shape.length + shape.height)

/-- Calculate the area of the rectangular shape -/
def area (shape : RectangularShape) : ℝ :=
  shape.length * shape.height

theorem rectangular_shape_perimeter_and_area 
  (shape : RectangularShape)
  (h1 : shape.segmentLength = 5)
  (h2 : shape.length = 45)
  (h3 : shape.height = 30) :
  perimeter shape = 200 ∧ area shape = 725 := by
  sorry

#check rectangular_shape_perimeter_and_area

end NUMINAMATH_CALUDE_rectangular_shape_perimeter_and_area_l2348_234801


namespace NUMINAMATH_CALUDE_forty_ab_over_c_value_l2348_234806

theorem forty_ab_over_c_value (a b c : ℝ) 
  (eq1 : 4 * a = 5 * b)
  (eq2 : 5 * b = 30)
  (eq3 : a + b + c = 15) :
  40 * a * b / c = 1200 := by
  sorry

end NUMINAMATH_CALUDE_forty_ab_over_c_value_l2348_234806


namespace NUMINAMATH_CALUDE_height_of_congruent_triangles_l2348_234802

/-- Triangle congruence relation -/
def CongruentTriangles (t1 t2 : Type) : Prop := sorry

/-- Area of a triangle -/
def TriangleArea (t : Type) : ℝ := sorry

/-- Height of a triangle on a given side -/
def TriangleHeight (t : Type) (side : ℝ) : ℝ := sorry

/-- Side length of a triangle -/
def TriangleSide (t : Type) (side : String) : ℝ := sorry

theorem height_of_congruent_triangles 
  (ABC DEF : Type) 
  (h_cong : CongruentTriangles ABC DEF) 
  (h_side : TriangleSide ABC "AB" = TriangleSide DEF "DE" ∧ TriangleSide ABC "AB" = 4) 
  (h_area : TriangleArea DEF = 10) :
  TriangleHeight ABC (TriangleSide ABC "AB") = 5 := by
  sorry

end NUMINAMATH_CALUDE_height_of_congruent_triangles_l2348_234802


namespace NUMINAMATH_CALUDE_locus_of_vertex_A_l2348_234814

def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  B = (-6, 0) ∧ C = (6, 0)

def angle_condition (A B C : ℝ) : Prop :=
  Real.sin B - Real.sin C = (1/2) * Real.sin A

def locus_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 27 = 1 ∧ x < -3

theorem locus_of_vertex_A (A B C : ℝ × ℝ) (angleA angleB angleC : ℝ) :
  triangle_ABC A B C →
  angle_condition angleA angleB angleC →
  locus_equation A.1 A.2 :=
sorry

end NUMINAMATH_CALUDE_locus_of_vertex_A_l2348_234814


namespace NUMINAMATH_CALUDE_star_seven_three_l2348_234897

-- Define the ⋆ operation
def star (x y : ℤ) : ℤ := 2 * x - 4 * y

-- State the theorem
theorem star_seven_three : star 7 3 = 2 := by sorry

end NUMINAMATH_CALUDE_star_seven_three_l2348_234897


namespace NUMINAMATH_CALUDE_strawberry_problem_l2348_234865

theorem strawberry_problem (initial : Float) (eaten : Float) (remaining : Float) :
  initial = 78.0 → eaten = 42.0 → remaining = initial - eaten → remaining = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_problem_l2348_234865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2348_234858

/-- An arithmetic sequence with first term 0 and non-zero common difference -/
structure ArithmeticSequence where
  d : ℝ
  hd : d ≠ 0
  a : ℕ → ℝ
  h_init : a 1 = 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  ∃ m : ℕ, seq.a m = seq.a 1 + seq.a 2 + seq.a 3 + seq.a 4 + seq.a 5 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2348_234858


namespace NUMINAMATH_CALUDE_angle_DAB_is_54_degrees_l2348_234850

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- A pentagon defined by five points -/
structure Pentagon :=
  (B : Point) (C : Point) (D : Point) (E : Point) (G : Point)

/-- The measure of an angle in degrees -/
def angle_measure (p q r : Point) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a triangle is isosceles -/
def is_isosceles (t : Triangle) : Prop :=
  distance t.C t.A = distance t.C t.B

/-- Checks if a pentagon is regular -/
def is_regular_pentagon (p : Pentagon) : Prop := sorry

/-- Theorem: In an isosceles triangle with a regular pentagon constructed on one side,
    the angle DAB measures 54 degrees -/
theorem angle_DAB_is_54_degrees 
  (t : Triangle) 
  (p : Pentagon) 
  (h1 : is_isosceles t) 
  (h2 : is_regular_pentagon p)
  (h3 : p.B = t.B ∧ p.C = t.C)
  (D : Point) 
  : angle_measure D t.A t.B = 54 := by sorry

end NUMINAMATH_CALUDE_angle_DAB_is_54_degrees_l2348_234850


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2348_234844

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2348_234844


namespace NUMINAMATH_CALUDE_f_ln_2_equals_neg_1_l2348_234846

-- Define the base of natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_ln_2_equals_neg_1 
  (h_monotonic : Monotone f)
  (h_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - e) :
  f (Real.log 2) = -1 := by sorry

end NUMINAMATH_CALUDE_f_ln_2_equals_neg_1_l2348_234846


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2348_234822

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2348_234822


namespace NUMINAMATH_CALUDE_bob_alice_difference_l2348_234882

/-- The difference in final amounts between two investors, given their initial investment
    and respective returns. -/
def investment_difference (initial_investment : ℕ) (alice_multiplier bob_multiplier : ℕ) : ℕ :=
  (initial_investment * bob_multiplier + initial_investment) - (initial_investment * alice_multiplier)

/-- Theorem stating that given the problem conditions, Bob ends up with $8000 more than Alice. -/
theorem bob_alice_difference : investment_difference 2000 2 5 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_bob_alice_difference_l2348_234882


namespace NUMINAMATH_CALUDE_cable_cost_calculation_l2348_234852

/-- Calculates the total cost of cable for a neighborhood given the following parameters:
* Number of east-west and north-south streets
* Length of east-west and north-south streets
* Cable required per mile of street
* Cost of regular cable for east-west and north-south streets
* Number of intersections and cost per intersection
* Number of streets requiring higher grade cable and its cost
-/
def total_cable_cost (
  num_ew_streets : ℕ
  ) (num_ns_streets : ℕ
  ) (len_ew_street : ℝ
  ) (len_ns_street : ℝ
  ) (cable_per_mile : ℝ
  ) (cost_ew_cable : ℝ
  ) (cost_ns_cable : ℝ
  ) (num_intersections : ℕ
  ) (cost_per_intersection : ℝ
  ) (num_hg_ew_streets : ℕ
  ) (num_hg_ns_streets : ℕ
  ) (cost_hg_cable : ℝ
  ) : ℝ :=
  let regular_ew_cost := (num_ew_streets : ℝ) * len_ew_street * cable_per_mile * cost_ew_cable
  let regular_ns_cost := (num_ns_streets : ℝ) * len_ns_street * cable_per_mile * cost_ns_cable
  let hg_ew_cost := (num_hg_ew_streets : ℝ) * len_ew_street * cable_per_mile * cost_hg_cable
  let hg_ns_cost := (num_hg_ns_streets : ℝ) * len_ns_street * cable_per_mile * cost_hg_cable
  let intersection_cost := (num_intersections : ℝ) * cost_per_intersection
  regular_ew_cost + regular_ns_cost + hg_ew_cost + hg_ns_cost + intersection_cost

theorem cable_cost_calculation :
  total_cable_cost 18 10 2 4 5 2500 3500 20 5000 3 2 4000 = 1530000 := by
  sorry

end NUMINAMATH_CALUDE_cable_cost_calculation_l2348_234852


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2348_234877

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2348_234877


namespace NUMINAMATH_CALUDE_total_volume_of_stacked_boxes_l2348_234848

/-- The volume of a single box in cubic centimeters -/
def single_box_volume : ℝ := 30

/-- The number of horizontal rows of boxes -/
def horizontal_rows : ℕ := 7

/-- The number of vertical rows of boxes -/
def vertical_rows : ℕ := 5

/-- The number of floors of boxes -/
def floors : ℕ := 3

/-- The total number of boxes -/
def total_boxes : ℕ := horizontal_rows * vertical_rows * floors

/-- The theorem stating the total volume of stacked boxes -/
theorem total_volume_of_stacked_boxes :
  (single_box_volume * total_boxes : ℝ) = 3150 := by sorry

end NUMINAMATH_CALUDE_total_volume_of_stacked_boxes_l2348_234848


namespace NUMINAMATH_CALUDE_bottle_caps_distribution_l2348_234838

theorem bottle_caps_distribution (num_children : ℕ) (total_caps : ℕ) (caps_per_child : ℕ) :
  num_children = 9 →
  total_caps = 45 →
  total_caps = num_children * caps_per_child →
  caps_per_child = 5 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_distribution_l2348_234838


namespace NUMINAMATH_CALUDE_tea_trader_profit_percentage_tea_trader_profit_is_35_percent_l2348_234821

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage 
  (tea1_weight : ℝ) (tea1_cost : ℝ) 
  (tea2_weight : ℝ) (tea2_cost : ℝ) 
  (sale_price : ℝ) : ℝ :=
  let total_weight := tea1_weight + tea2_weight
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_sale := total_weight * sale_price
  let profit := total_sale - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage

/-- Proves that the profit percentage is 35% for the given scenario --/
theorem tea_trader_profit_is_35_percent : 
  tea_trader_profit_percentage 80 15 20 20 21.6 = 35 := by
  sorry

end NUMINAMATH_CALUDE_tea_trader_profit_percentage_tea_trader_profit_is_35_percent_l2348_234821


namespace NUMINAMATH_CALUDE_log_stack_sum_l2348_234871

/-- Given an arithmetic sequence with 11 terms, starting at 5 and ending at 15,
    prove that the sum of all terms is 110. -/
theorem log_stack_sum : 
  let n : ℕ := 11  -- number of terms
  let a : ℕ := 5   -- first term
  let l : ℕ := 15  -- last term
  n * (a + l) / 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2348_234871


namespace NUMINAMATH_CALUDE_minimize_distance_sum_l2348_234816

/-- Given points P and Q in the xy-plane, and R on the line segment PQ, 
    prove that R(2, -1/9) minimizes the sum of distances PR + RQ -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) : 
  P = (-3, -4) → 
  Q = (6, 3) → 
  R.1 = 2 → 
  R.2 = -1/9 → 
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • P + t • Q) →
  ∀ (S : ℝ × ℝ), (∃ (u : ℝ), 0 ≤ u ∧ u ≤ 1 ∧ S = (1 - u) • P + u • Q) →
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≤ 
    Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) + Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_minimize_distance_sum_l2348_234816


namespace NUMINAMATH_CALUDE_cloth_trimming_l2348_234809

theorem cloth_trimming (x : ℝ) : 
  x > 0 → 
  (x - 4) * (x - 3) = 120 → 
  x = 12 :=
by sorry

end NUMINAMATH_CALUDE_cloth_trimming_l2348_234809


namespace NUMINAMATH_CALUDE_rectangle_area_l2348_234875

theorem rectangle_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 * b) / (b^2 * a) = 5/8 →
  (a + 6) * (b + 6) - a * b = 114 →
  a * b = 40 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2348_234875


namespace NUMINAMATH_CALUDE_regular_polygon_angle_sum_l2348_234825

/-- For a regular polygon with n sides, if the sum of its interior angles
    is 4 times the sum of its exterior angles, then n = 10 -/
theorem regular_polygon_angle_sum (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_sum_l2348_234825


namespace NUMINAMATH_CALUDE_mice_problem_l2348_234818

theorem mice_problem (x : ℕ) : 
  (x / 2 : ℕ) * 2 = x ∧ 
  ((x - x / 2) / 3 : ℕ) * 3 = x - x / 2 ∧
  (((x - x / 2) - (x - x / 2) / 3) / 4 : ℕ) * 4 = (x - x / 2) - (x - x / 2) / 3 ∧
  ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) / 5 : ℕ) * 5 = 
    ((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4 ∧
  ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) - 
    ((((x - x / 2) - (x - x / 2) / 3) - ((x - x / 2) - (x - x / 2) / 3) / 4) / 5)) = 
    (x - x / 2) / 3 + 2 →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_mice_problem_l2348_234818


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2348_234826

theorem min_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x + 8*y + 1 = 0) :
  ∃ (m : ℝ), m = -2*Real.sqrt 2 - 1 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 - 2*a + 8*b + 1 = 0 → m ≤ a + 2*b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2348_234826


namespace NUMINAMATH_CALUDE_range_of_expression_l2348_234840

theorem range_of_expression (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_one : x + y + z = 1) :
  -1/8 ≤ (z - x) * (z - y) ∧ (z - x) * (z - y) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l2348_234840


namespace NUMINAMATH_CALUDE_tennis_players_count_l2348_234874

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : neither = 5)
  (h4 : both = 3)
  : ∃ tennis : ℕ, tennis = 18 ∧ 
    total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l2348_234874


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2004_l2348_234861

/-- The sum of the tens digit and the units digit in the decimal representation of 9^2004 is 7. -/
theorem sum_of_tens_and_units_digits_of_9_pow_2004 : ∃ n : ℕ, 9^2004 = 100 * n + 61 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2004_l2348_234861


namespace NUMINAMATH_CALUDE_basic_computer_price_l2348_234847

theorem basic_computer_price 
  (total_price : ℝ) 
  (enhanced_computer_diff : ℝ) 
  (printer_ratio : ℝ) :
  total_price = 2500 →
  enhanced_computer_diff = 500 →
  printer_ratio = 1/4 →
  ∃ (basic_computer : ℝ) (printer : ℝ),
    basic_computer + printer = total_price ∧
    printer = printer_ratio * (basic_computer + enhanced_computer_diff + printer) ∧
    basic_computer = 1750 :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l2348_234847


namespace NUMINAMATH_CALUDE_xy_difference_squared_l2348_234817

theorem xy_difference_squared (x y : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x - y = -2) : 
  x^2 * y - x * y^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_xy_difference_squared_l2348_234817


namespace NUMINAMATH_CALUDE_g_sum_property_l2348_234855

def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + 5

theorem g_sum_property (d e f : ℝ) :
  g d e f 20 = 7 → g d e f 20 + g d e f (-20) = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l2348_234855


namespace NUMINAMATH_CALUDE_m_range_l2348_234887

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∀ x : ℝ, x^2 + (m-3)*x + 1 ≠ 0

def Q (m : ℝ) : Prop := ∃ a b : ℝ, a > b ∧ a^2 + b^2 = m-1 ∧
  ∀ x y : ℝ, x^2 + y^2/(m-1) = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, (¬(P m) → False) ∧ ((P m ∧ Q m) → False)) →
  {m : ℝ | 1 < m ∧ m ≤ 2} = {m : ℝ | ∃ x : ℝ, m = x ∧ 1 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2348_234887


namespace NUMINAMATH_CALUDE_chain_rule_with_local_injectivity_l2348_234851

/-- Given two differentiable functions f and g, with f having a local injectivity property,
    prove that their composition is differentiable and satisfies the chain rule. -/
theorem chain_rule_with_local_injectivity 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (x₀ : ℝ) 
  (hf : DifferentiableAt ℝ f x₀)
  (hg : DifferentiableAt ℝ g (f x₀))
  (hU : ∃ U : Set ℝ, IsOpen U ∧ x₀ ∈ U ∧ ∀ x ∈ U, x ≠ x₀ → f x ≠ f x₀) :
  DifferentiableAt ℝ (g ∘ f) x₀ ∧ 
  deriv (g ∘ f) x₀ = deriv g (f x₀) * deriv f x₀ :=
by sorry

end NUMINAMATH_CALUDE_chain_rule_with_local_injectivity_l2348_234851


namespace NUMINAMATH_CALUDE_randys_trip_length_l2348_234862

theorem randys_trip_length :
  ∀ (total : ℚ),
  (total / 3 : ℚ) + 20 + (total / 5 : ℚ) = total →
  total = 300 / 7 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2348_234862


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2348_234823

theorem arithmetic_calculation : 1^2 + (2 * 3)^3 - 4^2 + Real.sqrt 9 = 204 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2348_234823


namespace NUMINAMATH_CALUDE_expression_value_at_nine_l2348_234883

theorem expression_value_at_nine :
  let x : ℝ := 9
  let f (x : ℝ) := (x^9 - 27*x^6 + 19683) / (x^6 - 27)
  f x = 492804 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_nine_l2348_234883


namespace NUMINAMATH_CALUDE_good_number_count_and_gcd_l2348_234842

def is_good_number (n : ℕ) : Prop :=
  n ≤ 2012 ∧ n % 9 = 6

theorem good_number_count_and_gcd :
  (∃ (S : Finset ℕ), (∀ n, n ∈ S ↔ is_good_number n) ∧ S.card = 223) ∧
  (∃ d : ℕ, d > 0 ∧ (∀ n, is_good_number n → d ∣ n) ∧
    ∀ m, m > 0 → (∀ n, is_good_number n → m ∣ n) → m ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_good_number_count_and_gcd_l2348_234842


namespace NUMINAMATH_CALUDE_fraction_ordering_l2348_234892

theorem fraction_ordering : 16/13 < 21/17 ∧ 21/17 < 20/15 := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2348_234892


namespace NUMINAMATH_CALUDE_difference_of_squares_l2348_234894

theorem difference_of_squares (m n : ℝ) : (-m - n) * (-m + n) = (-m)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2348_234894


namespace NUMINAMATH_CALUDE_peach_count_l2348_234837

theorem peach_count (initial : ℕ) (picked : ℕ) (total : ℕ) : 
  initial = 34 → picked = 52 → total = initial + picked → total = 86 := by
sorry

end NUMINAMATH_CALUDE_peach_count_l2348_234837


namespace NUMINAMATH_CALUDE_decreasing_function_positive_l2348_234800

/-- A decreasing function satisfying a specific condition is always positive -/
theorem decreasing_function_positive
  (f : ℝ → ℝ) (hf : Monotone (fun x ↦ -f x))
  (hf' : ∀ x, ∃ f'x, HasDerivAt f f'x x ∧ f x / f'x + x < 1) :
  ∀ x, f x > 0 := by
sorry

end NUMINAMATH_CALUDE_decreasing_function_positive_l2348_234800


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_x_l2348_234864

theorem inequality_holds_for_all_x : ∀ x : ℝ, x + 2 < x + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_x_l2348_234864


namespace NUMINAMATH_CALUDE_assignment_plans_count_l2348_234835

/-- The number of students --/
def total_students : ℕ := 6

/-- The number of tasks --/
def total_tasks : ℕ := 4

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of students who cannot be assigned to a specific task --/
def restricted_students : ℕ := 2

/-- Calculates the total number of different assignment plans --/
def total_assignment_plans : ℕ := 
  (total_students.factorial / (total_students - selected_students).factorial) - 
  2 * ((total_students - 1).factorial / (total_students - selected_students).factorial)

/-- Theorem stating the total number of different assignment plans --/
theorem assignment_plans_count : total_assignment_plans = 240 := by
  sorry

end NUMINAMATH_CALUDE_assignment_plans_count_l2348_234835


namespace NUMINAMATH_CALUDE_square_areas_equal_l2348_234898

/-- Represents the configuration of squares and circles -/
structure SquareCircleConfig where
  circle_radius : ℝ
  num_small_squares : ℕ

/-- Calculates the area of the larger square -/
def larger_square_area (config : SquareCircleConfig) : ℝ :=
  4 * config.circle_radius ^ 2

/-- Calculates the total area of the smaller squares -/
def total_small_squares_area (config : SquareCircleConfig) : ℝ :=
  config.num_small_squares * (2 * config.circle_radius) ^ 2

/-- Theorem stating that the area of the larger square is equal to the total area of the smaller squares -/
theorem square_areas_equal (config : SquareCircleConfig) 
    (h1 : config.circle_radius = 3)
    (h2 : config.num_small_squares = 4) : 
  larger_square_area config = total_small_squares_area config ∧ 
  larger_square_area config = 144 := by
  sorry

#eval larger_square_area { circle_radius := 3, num_small_squares := 4 }
#eval total_small_squares_area { circle_radius := 3, num_small_squares := 4 }

end NUMINAMATH_CALUDE_square_areas_equal_l2348_234898


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2348_234878

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 1 ≤ 0) ↔ (∃ x : ℝ, x^2 - 3*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2348_234878


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l2348_234873

/-- For an arithmetic sequence {a_n} with sum S_n = 2n - 1, prove the common ratio is 2 -/
theorem arithmetic_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = 2 * n - 1) 
  (h2 : ∀ n, S n = n * a 1) : 
  (a 2) / (a 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l2348_234873


namespace NUMINAMATH_CALUDE_factors_of_504_l2348_234849

def number_of_positive_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_504 : number_of_positive_factors 504 = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_504_l2348_234849


namespace NUMINAMATH_CALUDE_gabby_makeup_set_savings_l2348_234859

/-- Proves that Gabby needs $10 more to buy the makeup set -/
theorem gabby_makeup_set_savings (makeup_cost initial_savings mom_contribution : ℕ) 
  (h1 : makeup_cost = 65)
  (h2 : initial_savings = 35)
  (h3 : mom_contribution = 20) :
  makeup_cost - initial_savings - mom_contribution = 10 := by
  sorry

end NUMINAMATH_CALUDE_gabby_makeup_set_savings_l2348_234859


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2348_234893

theorem sqrt_meaningful_range (m : ℝ) : 
  (∃ (x : ℝ), x^2 = m + 3) ↔ m ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2348_234893


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2348_234836

theorem arithmetic_calculation : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2348_234836


namespace NUMINAMATH_CALUDE_sam_investment_result_l2348_234819

/-- Calculates the final amount of an investment given initial conditions and interest rates --/
def calculate_investment (initial_investment : ℝ) (first_rate : ℝ) (first_years : ℕ) 
  (multiplier : ℝ) (second_rate : ℝ) : ℝ :=
  let first_phase := initial_investment * (1 + first_rate) ^ first_years
  let second_phase := first_phase * multiplier
  let final_amount := second_phase * (1 + second_rate)
  final_amount

/-- Theorem stating the final amount of Sam's investment --/
theorem sam_investment_result : 
  calculate_investment 10000 0.20 3 3 0.15 = 59616 := by
  sorry

#eval calculate_investment 10000 0.20 3 3 0.15

end NUMINAMATH_CALUDE_sam_investment_result_l2348_234819


namespace NUMINAMATH_CALUDE_tea_mixture_price_is_153_l2348_234829

/-- Calculates the price of a tea mixture given the prices of three tea varieties and their mixing ratio. -/
def tea_mixture_price (p1 p2 p3 : ℚ) (r1 r2 r3 : ℚ) : ℚ :=
  (p1 * r1 + p2 * r2 + p3 * r3) / (r1 + r2 + r3)

/-- Theorem stating that the price of a specific tea mixture is 153. -/
theorem tea_mixture_price_is_153 :
  tea_mixture_price 126 135 175.5 1 1 2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_is_153_l2348_234829


namespace NUMINAMATH_CALUDE_certain_number_equation_l2348_234820

theorem certain_number_equation (x : ℝ) : (40 * 30 + (x + 8) * 3) / 5 = 1212 ↔ x = 1612 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2348_234820


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2348_234866

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 + 5*x + 8)*(x + 2) - (x + 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x + 2)*(x + 6) = 
  6*x^3 + 21*x^2 + 18*x := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2348_234866


namespace NUMINAMATH_CALUDE_chromosome_variation_identification_l2348_234839

-- Define the structure for a genetic condition
structure GeneticCondition where
  name : String
  chromosomeAffected : Nat
  variationType : String

-- Define the statements
def statement1 : GeneticCondition := ⟨"cri-du-chat syndrome", 5, "partial deletion"⟩
def statement2 := "free combination of non-homologous chromosomes during meiosis"
def statement3 := "chromosomal exchange between synapsed homologous chromosomes"
def statement4 : GeneticCondition := ⟨"Down syndrome", 21, "extra chromosome"⟩

-- Define what constitutes a chromosome variation
def isChromosomeVariation (condition : GeneticCondition) : Prop :=
  condition.variationType = "partial deletion" ∨ condition.variationType = "extra chromosome"

-- Theorem to prove
theorem chromosome_variation_identification :
  (isChromosomeVariation statement1 ∧ isChromosomeVariation statement4) ∧
  (¬ isChromosomeVariation ⟨"", 0, statement2⟩ ∧ ¬ isChromosomeVariation ⟨"", 0, statement3⟩) := by
  sorry


end NUMINAMATH_CALUDE_chromosome_variation_identification_l2348_234839


namespace NUMINAMATH_CALUDE_minimum_g_5_l2348_234895

def Tenuous (f : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 ∧ y > 0 → f x + f y > y^2

def SumOfG (g : ℕ → ℤ) : ℤ :=
  (List.range 10).map (λ i => g (i + 1)) |>.sum

theorem minimum_g_5 (g : ℕ → ℤ) (h_tenuous : Tenuous g) 
    (h_min : ∀ g' : ℕ → ℤ, Tenuous g' → SumOfG g ≤ SumOfG g') : 
  g 5 ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_minimum_g_5_l2348_234895


namespace NUMINAMATH_CALUDE_exists_range_sum_and_even_count_611_l2348_234889

/-- Sum of integers from a to b (inclusive) -/
def sum_range (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Count of even integers in range [a, b] -/
def count_even (a b : ℤ) : ℤ :=
  if a % 2 = 0 && b % 2 = 0 then
    (b - a) / 2 + 1
  else
    (b - a + 1) / 2

theorem exists_range_sum_and_even_count_611 :
  ∃ a b : ℤ, sum_range a b + count_even a b = 611 :=
sorry

end NUMINAMATH_CALUDE_exists_range_sum_and_even_count_611_l2348_234889


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2348_234853

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2348_234853


namespace NUMINAMATH_CALUDE_min_value_of_h_neg_infinity_to_zero_l2348_234810

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function h(x) defined in terms of f(x) and g(x) -/
def h (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x ^ 3 - b * g x - 2

theorem min_value_of_h_neg_infinity_to_zero 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : IsOdd f) (hg : IsOdd g)
  (hmax : ∃ x > 0, ∀ y > 0, h f g a b y ≤ h f g a b x ∧ h f g a b x = 5) :
  ∃ x < 0, ∀ y < 0, h f g a b y ≥ h f g a b x ∧ h f g a b x = -9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_h_neg_infinity_to_zero_l2348_234810


namespace NUMINAMATH_CALUDE_minimum_score_exists_l2348_234812

/-- Represents the scores of the four people who took the math test. -/
structure TestScores where
  marty : ℕ
  others : Fin 3 → ℕ

/-- The proposition that Marty's score is the minimum to conclude others scored below average. -/
def IsMinimumScore (scores : TestScores) : Prop :=
  scores.marty = 61 ∧
  (∀ i : Fin 3, scores.others i < 20) ∧
  (∀ s : TestScores, s.marty < 61 → 
    ∃ i : Fin 3, s.others i ≥ 20 ∨ (s.marty + (Finset.sum Finset.univ s.others)) / 4 ≠ 20)

/-- The theorem stating that there exists a score distribution satisfying the conditions. -/
theorem minimum_score_exists : ∃ scores : TestScores, IsMinimumScore scores ∧ 
  (scores.marty + (Finset.sum Finset.univ scores.others)) / 4 = 20 := by
  sorry

#check minimum_score_exists

end NUMINAMATH_CALUDE_minimum_score_exists_l2348_234812


namespace NUMINAMATH_CALUDE_chef_cakes_l2348_234870

/-- Given a total number of eggs, eggs put in the fridge, and eggs needed per cake,
    calculate the number of cakes that can be made. -/
def cakes_made (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

/-- Prove that given 60 total eggs, with 10 eggs put in the fridge,
    and 5 eggs needed for one cake, the number of cakes the chef can make is 10. -/
theorem chef_cakes :
  cakes_made 60 10 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_chef_cakes_l2348_234870


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2348_234860

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 4 * n = k ^ 2) ∧ 
    (∃ (m : ℕ), 3 * n = m ^ 3)) ∧
  (∀ (n : ℕ), n > 0 ∧ n < 144 → 
    ¬(∃ (k : ℕ), 4 * n = k ^ 2) ∨ 
    ¬(∃ (m : ℕ), 3 * n = m ^ 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2348_234860


namespace NUMINAMATH_CALUDE_rice_price_calculation_l2348_234857

def initial_amount : ℝ := 500
def wheat_flour_price : ℝ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℝ := 150
def soda_quantity : ℕ := 1
def rice_quantity : ℕ := 2
def remaining_balance : ℝ := 235

theorem rice_price_calculation : 
  ∃ (rice_price : ℝ), 
    initial_amount - 
    (rice_price * rice_quantity + 
     wheat_flour_price * wheat_flour_quantity + 
     soda_price * soda_quantity) = remaining_balance ∧ 
    rice_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_rice_price_calculation_l2348_234857


namespace NUMINAMATH_CALUDE_complex_equality_implies_sum_l2348_234831

theorem complex_equality_implies_sum (a b : ℝ) :
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (2 + Complex.I) * (1 - b * Complex.I) = a + Complex.I →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_sum_l2348_234831


namespace NUMINAMATH_CALUDE_coin_value_difference_l2348_234869

/-- Represents the total number of coins Alice has -/
def total_coins : ℕ := 3030

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the total value in cents given the number of dimes -/
def total_value (dimes : ℕ) : ℕ :=
  dime_value * dimes + nickel_value * (total_coins - dimes)

/-- Represents the constraint that Alice has at least three times as many nickels as dimes -/
def nickel_constraint (dimes : ℕ) : Prop :=
  3 * dimes ≤ total_coins - dimes

theorem coin_value_difference :
  ∃ (max_dimes min_dimes : ℕ),
    nickel_constraint max_dimes ∧
    nickel_constraint min_dimes ∧
    (∀ d, nickel_constraint d → total_value d ≤ total_value max_dimes) ∧
    (∀ d, nickel_constraint d → total_value min_dimes ≤ total_value d) ∧
    total_value max_dimes - total_value min_dimes = 3780 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l2348_234869


namespace NUMINAMATH_CALUDE_smaller_cubes_count_l2348_234811

theorem smaller_cubes_count (large_volume : ℝ) (small_volume : ℝ) (surface_area_diff : ℝ) :
  large_volume = 125 →
  small_volume = 1 →
  surface_area_diff = 600 →
  (((6 * small_volume^(2/3)) * (large_volume / small_volume)) - (6 * large_volume^(2/3))) = surface_area_diff →
  (large_volume / small_volume) = 125 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cubes_count_l2348_234811


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l2348_234890

theorem complex_in_first_quadrant (a : ℝ) : 
  (((1 : ℂ) + a * Complex.I) / ((2 : ℂ) - Complex.I)).re > 0 ∧
  (((1 : ℂ) + a * Complex.I) / ((2 : ℂ) - Complex.I)).im > 0 →
  -1/2 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l2348_234890


namespace NUMINAMATH_CALUDE_intersection_A_B_l2348_234886

-- Define set A
def A : Set ℝ := {x | (x - 2) * (2 * x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | x < 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1/2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2348_234886


namespace NUMINAMATH_CALUDE_product_expansion_l2348_234891

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (7 / x^2 + 7*x - 7/x) = 3 / x^2 + 3*x - 3/x := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2348_234891


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2348_234879

theorem quadratic_factorization_sum (p q r : ℤ) : 
  (∀ x, x^2 + 16*x + 63 = (x + p) * (x + q)) →
  (∀ x, x^2 - 15*x + 56 = (x - q) * (x - r)) →
  p + q + r = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2348_234879


namespace NUMINAMATH_CALUDE_unoccupied_volume_of_cube_l2348_234813

/-- The volume of a cube not occupied by five spheres --/
theorem unoccupied_volume_of_cube (π : Real) : 
  let cube_edge : Real := 2
  let sphere_radius : Real := 1
  let cube_volume : Real := cube_edge ^ 3
  let sphere_volume : Real := (4 / 3) * π * sphere_radius ^ 3
  let total_sphere_volume : Real := 5 * sphere_volume
  cube_volume - total_sphere_volume = 8 - (20 / 3) * π := by sorry

end NUMINAMATH_CALUDE_unoccupied_volume_of_cube_l2348_234813


namespace NUMINAMATH_CALUDE_max_sides_cube_cross_section_l2348_234885

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where

/-- A plane is a flat, two-dimensional surface that extends infinitely far. -/
structure Plane where

/-- A cross-section is the intersection of a plane with a three-dimensional object. -/
def CrossSection (c : Cube) (p : Plane) : Set (ℝ × ℝ × ℝ) := sorry

/-- The number of sides in a polygon. -/
def NumberOfSides (polygon : Set (ℝ × ℝ × ℝ)) : ℕ := sorry

/-- The maximum number of sides in any cross-section of a cube is 6. -/
theorem max_sides_cube_cross_section (c : Cube) : 
  ∀ p : Plane, NumberOfSides (CrossSection c p) ≤ 6 ∧ 
  ∃ p : Plane, NumberOfSides (CrossSection c p) = 6 :=
sorry

end NUMINAMATH_CALUDE_max_sides_cube_cross_section_l2348_234885


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l2348_234884

theorem tan_alpha_minus_pi_over_four (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan β = 1/3) :
  Real.tan (α - π/4) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l2348_234884


namespace NUMINAMATH_CALUDE_quadrilateral_existence_l2348_234815

theorem quadrilateral_existence : ∃ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
  d = 2 * a ∧
  a + b + c + d = 2 ∧
  a + b + c > d ∧
  a + b + d > c ∧
  a + c + d > b ∧
  b + c + d > a := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_existence_l2348_234815


namespace NUMINAMATH_CALUDE_valid_numbers_l2348_234863

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n ≤ 999999) ∧  -- six-digit number
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 100000 + 2014 * 10 + b) ∧  -- formed by adding digits to 2014
  n % 36 = 0  -- divisible by 36

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {220140, 720144, 320148} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2348_234863


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2348_234881

theorem two_numbers_problem (x y : ℝ) : 
  x^2 + y^2 = 45/4 ∧ x - y = x * y → 
  (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2348_234881


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l2348_234808

theorem largest_number_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (b = (5 * a) / 3) →
  (c = (7 * a) / 3) →
  (c - a = 32) →
  c = 56 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l2348_234808


namespace NUMINAMATH_CALUDE_complex_number_problem_l2348_234807

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  z₂ = 4 + 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2348_234807


namespace NUMINAMATH_CALUDE_total_sales_correct_l2348_234803

/-- Calculates the total amount of money made from selling bracelets and necklaces. -/
def calculate_total_sales (
  bracelet_price : ℕ)
  (bracelet_discount_price : ℕ)
  (necklace_price : ℕ)
  (necklace_discount_price : ℕ)
  (regular_bracelets_sold : ℕ)
  (discounted_bracelets_sold : ℕ)
  (regular_necklaces_sold : ℕ)
  (discounted_necklace_sets_sold : ℕ) : ℕ :=
  (regular_bracelets_sold * bracelet_price) +
  (discounted_bracelets_sold / 2 * bracelet_discount_price) +
  (regular_necklaces_sold * necklace_price) +
  (discounted_necklace_sets_sold * necklace_discount_price)

theorem total_sales_correct :
  calculate_total_sales 5 8 10 25 12 12 8 2 = 238 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_correct_l2348_234803


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2348_234804

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x + 18 = 0

-- Define the roots of the equation
def root1 : ℝ := 3
def root2 : ℝ := 6

-- Define the isosceles triangle formed by the roots
def isosceles_triangle (a b : ℝ) : Prop :=
  (quadratic_equation a ∧ quadratic_equation b) ∧
  ((a = root1 ∧ b = root2) ∨ (a = root2 ∧ b = root1))

-- State the theorem
theorem isosceles_triangle_perimeter :
  ∀ a b : ℝ, isosceles_triangle a b → a + 2*b = 15 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2348_234804


namespace NUMINAMATH_CALUDE_algebra_test_average_l2348_234834

theorem algebra_test_average (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (total_average : ℚ) (male_average : ℚ) :
  total_students = male_students + female_students →
  total_students = 36 →
  male_students = 8 →
  female_students = 28 →
  total_average = 90 →
  male_average = 83 →
  (total_students : ℚ) * total_average = 
    (male_students : ℚ) * male_average + (female_students : ℚ) * ((3240 - 664 : ℚ) / 28) :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2348_234834


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2348_234828

theorem book_arrangement_count : 
  let total_books : ℕ := 7
  let identical_math_books : ℕ := 3
  let identical_physics_books : ℕ := 2
  let distinct_books : ℕ := total_books - identical_math_books - identical_physics_books
  ↑(Nat.factorial total_books) / (↑(Nat.factorial identical_math_books) * ↑(Nat.factorial identical_physics_books)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2348_234828


namespace NUMINAMATH_CALUDE_intersection_complement_sets_l2348_234805

open Set

theorem intersection_complement_sets (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let M : Set ℝ := {x | b < x ∧ x < (a + b) / 2}
  let N : Set ℝ := {x | Real.sqrt (a * b) < x ∧ x < a}
  M ∩ (Nᶜ) = {x | b < x ∧ x ≤ Real.sqrt (a * b)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_sets_l2348_234805


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l2348_234867

theorem ticket_price_possibilities : 
  ∃! n : ℕ, n > 0 ∧ 
    (∀ x : ℕ, x > 0 → (72 % x = 0 ∧ 90 % x = 0 ∧ 108 % x = 0) ↔ x ∈ Finset.range n) :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l2348_234867


namespace NUMINAMATH_CALUDE_R_properties_l2348_234899

noncomputable def R (x : ℝ) : ℝ :=
  x^2 + 1/x^2 + (1-x)^2 + 1/(1-x)^2 + x^2/(1-x)^2 + (x-1)^2/x^2

theorem R_properties :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = R (1/x)) ∧
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = R (1-x)) ∧
  ¬ (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = c) :=
by sorry

end NUMINAMATH_CALUDE_R_properties_l2348_234899


namespace NUMINAMATH_CALUDE_flash_catch_up_distance_l2348_234856

theorem flash_catch_up_distance
  (v a x y : ℝ) -- v: Ace's speed, a: Flash's acceleration, x: Flash's initial speed multiplier, y: initial distance behind
  (hx : x > 1)
  (ha : a > 0) :
  let d := y + x * v * (-(x - 1) * v + Real.sqrt ((x - 1)^2 * v^2 + 2 * a * y)) / a
  let t := (-(x - 1) * v + Real.sqrt ((x - 1)^2 * v^2 + 2 * a * y)) / a
  d = y + x * v * t + (1/2) * a * t^2 ∧
  d = v * t :=
by sorry

end NUMINAMATH_CALUDE_flash_catch_up_distance_l2348_234856


namespace NUMINAMATH_CALUDE_model_c_net_change_l2348_234841

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_increase (price : ℝ) (increase : ℝ) : ℝ :=
  price * (1 + increase)

def model_c_price : ℝ := 2000

def model_c_discount1 : ℝ := 0.20
def model_c_increase : ℝ := 0.20
def model_c_discount2 : ℝ := 0.05

theorem model_c_net_change :
  let price1 := apply_discount model_c_price model_c_discount1
  let price2 := apply_increase price1 model_c_increase
  let price3 := apply_discount price2 model_c_discount2
  price3 - model_c_price = -176 := by
  sorry

end NUMINAMATH_CALUDE_model_c_net_change_l2348_234841


namespace NUMINAMATH_CALUDE_polynomial_equality_l2348_234827

theorem polynomial_equality (h : ℝ → ℝ) : 
  (∀ x : ℝ, 7 * x^4 - 4 * x^3 + x + h x = 5 * x^3 - 7 * x + 6) →
  (∀ x : ℝ, h x = -7 * x^4 + 9 * x^3 - 8 * x + 6) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2348_234827


namespace NUMINAMATH_CALUDE_house_painting_cost_is_1900_l2348_234880

/-- The cost of painting a house given the contributions of three individuals. -/
def housePaintingCost (judsonContribution : ℕ) : ℕ :=
  let kennyContribution := judsonContribution + judsonContribution / 5
  let camiloContribution := kennyContribution + 200
  judsonContribution + kennyContribution + camiloContribution

/-- Theorem stating that the total cost of painting the house is $1900. -/
theorem house_painting_cost_is_1900 : housePaintingCost 500 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_is_1900_l2348_234880


namespace NUMINAMATH_CALUDE_vector_magnitude_l2348_234872

def problem (m n : ℝ × ℝ) : Prop :=
  let ⟨mx, my⟩ := m
  let ⟨nx, ny⟩ := n
  (mx * nx + my * ny = 0) ∧  -- m perpendicular to n
  (m.1 - 2 * n.1 = 11) ∧     -- x-component of m - 2n = 11
  (m.2 - 2 * n.2 = -2) ∧     -- y-component of m - 2n = -2
  (mx^2 + my^2 = 25)         -- |m| = 5

theorem vector_magnitude (m n : ℝ × ℝ) :
  problem m n → n.1^2 + n.2^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2348_234872


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_area_l2348_234868

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with semi-major axis 7 and semi-minor axis 2√6 -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 49 + p.y^2 / 24 = 1

/-- The two foci of the ellipse -/
def F₁ : Point := ⟨-5, 0⟩
def F₂ : Point := ⟨5, 0⟩

/-- Checks if lines PF₁ and PF₂ are perpendicular -/
def areLinesPerpendicular (p : Point) : Prop :=
  (p.y / (p.x + 5)) * (p.y / (p.x - 5)) = -1

/-- Calculates the area of triangle PF₁F₂ -/
def triangleArea (p : Point) : ℝ :=
  5 * |p.y|

/-- Main theorem -/
theorem ellipse_perpendicular_area (p : Point) :
  isOnEllipse p → areLinesPerpendicular p → triangleArea p = 24 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_area_l2348_234868


namespace NUMINAMATH_CALUDE_value_of_S_6_l2348_234832

theorem value_of_S_6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12196 := by
  sorry

end NUMINAMATH_CALUDE_value_of_S_6_l2348_234832


namespace NUMINAMATH_CALUDE_part_one_part_two_l2348_234824

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3

-- Part I
theorem part_one (a b : ℝ) (h : solution_set a b) : a + b = 11 := by
  sorry

-- Part II
def g (b c x : ℝ) := -x^2 + b*x + c

theorem part_two (b c : ℝ) (h1 : b = 6) 
  (h2 : ∀ x, g b c x ≤ 0) : c ≤ -9 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2348_234824


namespace NUMINAMATH_CALUDE_percentage_problem_l2348_234876

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 640 = 0.20 * 650 + 190 → P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2348_234876


namespace NUMINAMATH_CALUDE_triangle_construction_cases_l2348_234833

/-- A triangle with side lengths a, b, c and angles A, B, C. --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The height of a triangle from vertex A to side BC. --/
def height_A (t : Triangle) : ℝ := sorry

/-- The height of a triangle from vertex C to side AB. --/
def height_C (t : Triangle) : ℝ := sorry

/-- Constructs triangles given side AB, height CC₁, and angle A. --/
def construct_ABC_CC1_A (c : ℝ) (h : ℝ) (α : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height CC₁, and angle C. --/
def construct_ABC_CC1_C (c : ℝ) (h : ℝ) (γ : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle A. --/
def construct_ABC_AA1_A (c : ℝ) (h : ℝ) (α : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle B. --/
def construct_ABC_AA1_B (c : ℝ) (h : ℝ) (β : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle C. --/
def construct_ABC_AA1_C (c : ℝ) (h : ℝ) (γ : ℝ) : Set Triangle := sorry

/-- The total number of distinct triangles that can be constructed from all cases. --/
def total_distinct_triangles : ℕ := sorry

theorem triangle_construction_cases :
  ∀ (c h α β γ : ℝ),
    c > 0 → h > 0 → 0 < α < π → 0 < β < π → 0 < γ < π →
    total_distinct_triangles = 11 := by sorry

end NUMINAMATH_CALUDE_triangle_construction_cases_l2348_234833
