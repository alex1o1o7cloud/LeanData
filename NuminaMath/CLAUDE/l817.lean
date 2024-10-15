import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l817_81738

theorem system_solution (x y z : ℝ) : 
  x^2 + y^2 + z^2 = 1 ∧ x^3 + y^3 + z^3 = 1 → 
  (x = 1 ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = 1 ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l817_81738


namespace NUMINAMATH_CALUDE_max_value_is_80_l817_81788

structure Rock :=
  (weight : ℕ)
  (value : ℕ)

def rock_types : List Rock := [
  ⟨6, 20⟩,
  ⟨3, 9⟩,
  ⟨2, 4⟩
]

def max_weight : ℕ := 24

def min_available : ℕ := 10

def optimal_value (rocks : List Rock) (max_w : ℕ) (min_avail : ℕ) : ℕ :=
  sorry

theorem max_value_is_80 :
  optimal_value rock_types max_weight min_available = 80 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_80_l817_81788


namespace NUMINAMATH_CALUDE_third_wednesday_not_22nd_l817_81725

def is_third_wednesday (day : ℕ) : Prop :=
  ∃ (first_wednesday : ℕ), 
    1 ≤ first_wednesday ∧ 
    first_wednesday ≤ 7 ∧ 
    day = first_wednesday + 14

theorem third_wednesday_not_22nd : 
  ¬ is_third_wednesday 22 :=
sorry

end NUMINAMATH_CALUDE_third_wednesday_not_22nd_l817_81725


namespace NUMINAMATH_CALUDE_bus_network_property_l817_81731

-- Define the type for bus stops
variable {V : Type}

-- Define the "can be reached from" relation
def can_reach (G : V → V → Prop) (x y : V) : Prop := G x y

-- Define the "comes after" relation
def comes_after (G : V → V → Prop) (x y : V) : Prop :=
  ∀ z, can_reach G z x → can_reach G z y ∧ ∀ w, can_reach G y w → can_reach G x w

-- State the theorem
theorem bus_network_property (G : V → V → Prop) 
  (h : ∀ x y : V, x ≠ y → (can_reach G x y ↔ comes_after G x y)) :
  ∀ a b : V, a ≠ b → (can_reach G a b ∨ can_reach G b a) ∧ ¬(can_reach G a b ∧ can_reach G b a) :=
sorry

end NUMINAMATH_CALUDE_bus_network_property_l817_81731


namespace NUMINAMATH_CALUDE_length_of_DH_l817_81764

-- Define the triangle and points
structure Triangle :=
  (A B C D E F G H : ℝ × ℝ)

-- Define the properties of the triangle and points
def EquilateralTriangle (t : Triangle) : Prop :=
  let d := Real.sqrt 3
  t.A = (0, 0) ∧ t.B = (2, 0) ∧ t.C = (1, d)

def PointsOnSides (t : Triangle) : Prop :=
  ∃ x y z w : ℝ,
    0 ≤ x ∧ x ≤ 2 ∧
    0 ≤ y ∧ y ≤ 2 ∧
    0 ≤ z ∧ z ≤ 2 ∧
    0 ≤ w ∧ w ≤ 2 ∧
    t.D = (x, 0) ∧
    t.F = (y, 0) ∧
    t.E = (1 - z/2, z * Real.sqrt 3 / 2) ∧
    t.G = (1 - w/2, w * Real.sqrt 3 / 2)

def ParallelLines (t : Triangle) : Prop :=
  (t.E.2 - t.D.2) / (t.E.1 - t.D.1) = Real.sqrt 3 ∧
  (t.G.2 - t.F.2) / (t.G.1 - t.F.1) = Real.sqrt 3

def SpecificLengths (t : Triangle) : Prop :=
  t.D.1 - t.A.1 = 0.5 ∧
  Real.sqrt ((t.E.1 - t.D.1)^2 + (t.E.2 - t.D.2)^2) = 1 ∧
  t.F.1 - t.D.1 = 0.5 ∧
  Real.sqrt ((t.G.1 - t.F.1)^2 + (t.G.2 - t.F.2)^2) = 1 ∧
  t.B.1 - t.F.1 = 0.5

def ParallelDH (t : Triangle) : Prop :=
  ∃ k : ℝ, t.H = (k * t.C.1 + (1 - k) * t.A.1, k * t.C.2 + (1 - k) * t.A.2)

-- State the theorem
theorem length_of_DH (t : Triangle) :
  EquilateralTriangle t →
  PointsOnSides t →
  ParallelLines t →
  SpecificLengths t →
  ParallelDH t →
  Real.sqrt ((t.H.1 - t.D.1)^2 + (t.H.2 - t.D.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_length_of_DH_l817_81764


namespace NUMINAMATH_CALUDE_identity_function_unique_l817_81729

def PositiveInt := {n : ℤ // n > 0}

def DivisibilityCondition (f : PositiveInt → PositiveInt) : Prop :=
  ∀ a b : PositiveInt, (a.val - (f b).val) ∣ (a.val * (f a).val - b.val * (f b).val)

theorem identity_function_unique :
  ∀ f : PositiveInt → PositiveInt,
    DivisibilityCondition f →
    ∀ x : PositiveInt, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_identity_function_unique_l817_81729


namespace NUMINAMATH_CALUDE_sean_total_apples_l817_81702

def initial_apples : ℕ := 9
def apples_per_day : ℕ := 8
def days : ℕ := 5

theorem sean_total_apples :
  initial_apples + apples_per_day * days = 49 := by
  sorry

end NUMINAMATH_CALUDE_sean_total_apples_l817_81702


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l817_81708

/-- Isosceles triangle type -/
structure IsoscelesTriangle where
  /-- Base length of the isosceles triangle -/
  base : ℝ
  /-- Length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- Height of the isosceles triangle -/
  height : ℝ
  /-- Condition: base and side are positive -/
  base_pos : 0 < base
  side_pos : 0 < side
  /-- Condition: height is positive -/
  height_pos : 0 < height
  /-- Condition: triangle inequality -/
  triangle_ineq : base < 2 * side

/-- Theorem: Given a perimeter and a height, an isosceles triangle exists -/
theorem isosceles_triangle_exists (perimeter : ℝ) (height : ℝ) 
  (perimeter_pos : 0 < perimeter) (height_pos : 0 < height) : 
  ∃ (t : IsoscelesTriangle), t.base + 2 * t.side = perimeter ∧ t.height = height := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l817_81708


namespace NUMINAMATH_CALUDE_max_current_speed_is_26_l817_81735

/-- The maximum possible integer value for the river current speed --/
def max_current_speed : ℕ := 26

/-- The speed at which Mumbo runs --/
def mumbo_speed : ℕ := 11

/-- The speed at which Yumbo walks --/
def yumbo_speed : ℕ := 6

/-- Represents the travel scenario described in the problem --/
structure TravelScenario where
  x : ℝ  -- distance from origin to Mumbo's raft storage
  y : ℝ  -- distance from origin to Yumbo's raft storage
  v : ℕ  -- speed of the river current

/-- Condition that Yumbo arrives earlier than Mumbo --/
def yumbo_arrives_earlier (s : TravelScenario) : Prop :=
  s.y / yumbo_speed < s.x / mumbo_speed + (s.x + s.y) / s.v

/-- Main theorem stating that 26 is the maximum possible current speed --/
theorem max_current_speed_is_26 :
  ∀ s : TravelScenario,
    s.x > 0 ∧ s.y > 0 ∧ s.x < s.y ∧ s.v ≥ 6 ∧ yumbo_arrives_earlier s
    → s.v ≤ max_current_speed :=
by sorry

#check max_current_speed_is_26

end NUMINAMATH_CALUDE_max_current_speed_is_26_l817_81735


namespace NUMINAMATH_CALUDE_opposite_values_imply_result_l817_81734

theorem opposite_values_imply_result (a b : ℝ) : 
  |a + 2| = -(b - 3)^2 → a^b + 3*(a - b) = -23 := by
  sorry

end NUMINAMATH_CALUDE_opposite_values_imply_result_l817_81734


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l817_81712

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_subsequence : (a 2) ^ 2 = a 1 * a 6
  sum_condition : 2 * a 1 + a 2 = 1

/-- The main theorem stating the explicit formula for the nth term -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 5/3 - n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l817_81712


namespace NUMINAMATH_CALUDE_statue_cost_l817_81721

theorem statue_cost (selling_price : ℚ) (profit_percentage : ℚ) (original_cost : ℚ) : 
  selling_price = 670 ∧ 
  profit_percentage = 25 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) → 
  original_cost = 536 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l817_81721


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l817_81751

theorem opposite_of_negative_one_third :
  -((-1 : ℚ) / 3) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l817_81751


namespace NUMINAMATH_CALUDE_no_infinite_sequence_positive_integers_l817_81730

theorem no_infinite_sequence_positive_integers :
  ¬ ∃ (a : ℕ → ℕ+), ∀ (n : ℕ), (a (n-1))^2 ≥ 2 * (a n) * (a (n+2)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_positive_integers_l817_81730


namespace NUMINAMATH_CALUDE_solution_set_implies_m_equals_two_l817_81742

theorem solution_set_implies_m_equals_two (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + a^2 < 0 ↔ 1 < x ∧ x < m) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_equals_two_l817_81742


namespace NUMINAMATH_CALUDE_division_value_problem_l817_81761

theorem division_value_problem (x : ℝ) : (4 / x) * 12 = 8 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l817_81761


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l817_81737

theorem snow_leopard_arrangement (n : ℕ) (h : n = 7) : 
  2 * Nat.factorial (n - 2) = 240 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l817_81737


namespace NUMINAMATH_CALUDE_milk_ratio_l817_81714

def total_cartons : ℕ := 24
def regular_cartons : ℕ := 3

theorem milk_ratio :
  let chocolate_cartons := total_cartons - regular_cartons
  (chocolate_cartons : ℚ) / regular_cartons = 7 / 1 :=
by sorry

end NUMINAMATH_CALUDE_milk_ratio_l817_81714


namespace NUMINAMATH_CALUDE_complex_number_equality_l817_81705

theorem complex_number_equality (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 3) ∧ 
  Complex.abs (z - 1) = Complex.abs (z - Complex.I) →
  z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l817_81705


namespace NUMINAMATH_CALUDE_initial_distance_between_cars_l817_81785

/-- 
Given two cars A and B traveling in the same direction:
- Car A's speed is 58 mph
- Car B's speed is 50 mph
- After 6 hours, Car A is 8 miles ahead of Car B
Prove that the initial distance between Car A and Car B is 40 miles
-/
theorem initial_distance_between_cars (speed_A speed_B time_elapsed final_distance : ℝ) 
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : time_elapsed = 6)
  (h4 : final_distance = 8) :
  speed_A * time_elapsed - speed_B * time_elapsed - final_distance = 40 :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_between_cars_l817_81785


namespace NUMINAMATH_CALUDE_factor_ab_squared_minus_25a_l817_81748

theorem factor_ab_squared_minus_25a (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_ab_squared_minus_25a_l817_81748


namespace NUMINAMATH_CALUDE_total_apples_correct_l817_81715

/-- The number of apples Bill picked from the orchard -/
def total_apples : ℕ := 56

/-- The number of children Bill has -/
def num_children : ℕ := 2

/-- The number of apples each child takes for teachers -/
def apples_per_child : ℕ := 3

/-- The number of teachers each child gives apples to -/
def num_teachers : ℕ := 2

/-- The number of pies Jill bakes -/
def num_pies : ℕ := 2

/-- The number of apples used per pie -/
def apples_per_pie : ℕ := 10

/-- The number of apples Bill has left -/
def apples_left : ℕ := 24

/-- Theorem stating that the total number of apples Bill picked is correct -/
theorem total_apples_correct :
  total_apples = 
    num_children * apples_per_child * num_teachers +
    num_pies * apples_per_pie +
    apples_left :=
by sorry

end NUMINAMATH_CALUDE_total_apples_correct_l817_81715


namespace NUMINAMATH_CALUDE_initial_fee_calculation_l817_81766

/-- The initial fee for a taxi trip, given the rate per segment and total charge for a specific distance. -/
theorem initial_fee_calculation (rate_per_segment : ℝ) (total_charge : ℝ) (distance : ℝ) : 
  rate_per_segment = 0.35 →
  distance = 3.6 →
  total_charge = 5.65 →
  ∃ (initial_fee : ℝ), initial_fee = 2.50 ∧ 
    total_charge = initial_fee + (distance / (2/5)) * rate_per_segment :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_calculation_l817_81766


namespace NUMINAMATH_CALUDE_change_received_l817_81747

/-- The change received when buying a football and baseball with given costs and payment amount -/
theorem change_received (football_cost baseball_cost payment : ℚ) : 
  football_cost = 9.14 →
  baseball_cost = 6.81 →
  payment = 20 →
  payment - (football_cost + baseball_cost) = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l817_81747


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l817_81709

/-- A quadratic polynomial with a common root property -/
structure QuadraticPolynomial where
  P : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c
  common_root : ∃ t : ℝ, P t = 0 ∧ P (P (P t)) = 0

/-- 
For any quadratic polynomial P(x) where P(x) and P(P(P(x))) have a common root, 
P(0)P(1) = 0
-/
theorem quadratic_polynomial_property (p : QuadraticPolynomial) : 
  p.P 0 * p.P 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l817_81709


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l817_81783

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark correct_mark : ℚ) : 
  n = 10 → 
  initial_average = 100 → 
  wrong_mark = 90 → 
  correct_mark = 10 → 
  (n * initial_average - (wrong_mark - correct_mark)) / n = 92 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l817_81783


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l817_81707

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_to_plane : Plane → Plane → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (P Q R : Plane) 
  (h1 : parallel_to_plane P R) 
  (h2 : parallel_to_plane Q R) : 
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel 
  (l1 l2 : Line) 
  (P : Plane) 
  (h1 : perpendicular_to_plane l1 P) 
  (h2 : perpendicular_to_plane l2 P) : 
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l817_81707


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_unique_positive_integer_solution_l817_81794

/-- The quadratic equation x^2 - 2x + 2m - 1 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*x + 2*m - 1 = 0

/-- m is a positive integer -/
def is_positive_integer (m : ℝ) : Prop :=
  m > 0 ∧ ∃ n : ℕ, m = n

theorem quadratic_equation_roots (m : ℝ) :
  has_real_roots m ↔ m ≤ 1 :=
sorry

theorem unique_positive_integer_solution (m : ℝ) :
  is_positive_integer m ∧ has_real_roots m →
  m = 1 ∧ ∃ x : ℝ, x = 1 ∧ x^2 - 2*x + 2*m - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_unique_positive_integer_solution_l817_81794


namespace NUMINAMATH_CALUDE_share_multiple_problem_l817_81782

theorem share_multiple_problem (total a b c x : ℚ) : 
  total = 880 →
  c = 160 →
  a + b + c = total →
  4 * a = 5 * b →
  4 * a = x * c →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_share_multiple_problem_l817_81782


namespace NUMINAMATH_CALUDE_ginger_mat_straw_ratio_l817_81720

/-- Given the conditions for Ginger's mat weaving, prove the ratio of green to orange straws per mat -/
theorem ginger_mat_straw_ratio :
  let red_per_mat : ℕ := 20
  let orange_per_mat : ℕ := 30
  let total_mats : ℕ := 10
  let total_straws : ℕ := 650
  let green_per_mat : ℕ := (total_straws - red_per_mat * total_mats - orange_per_mat * total_mats) / total_mats
  green_per_mat * 2 = orange_per_mat := by
  sorry

#check ginger_mat_straw_ratio

end NUMINAMATH_CALUDE_ginger_mat_straw_ratio_l817_81720


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l817_81763

/-- 
Given a triangle XYZ:
- ext_angle_x is the exterior angle at vertex X
- angle_y is the angle at vertex Y
- angle_z is the angle at vertex Z

This theorem states that if the exterior angle at X is 150° and the angle at Y is 140°, 
then the angle at Z must be 110°.
-/
theorem triangle_angle_calculation 
  (ext_angle_x angle_y angle_z : ℝ) 
  (h1 : ext_angle_x = 150)
  (h2 : angle_y = 140) :
  angle_z = 110 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l817_81763


namespace NUMINAMATH_CALUDE_horner_method_result_l817_81700

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_result : f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_result_l817_81700


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l817_81717

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l817_81717


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_six_l817_81762

def polynomial (x : ℝ) : ℝ :=
  -5 * (x^5 - 2*x^3 + x) + 8 * (x^5 + x^3 - 3) - 3 * (3*x^5 + x^3 + 2)

theorem leading_coefficient_is_negative_six :
  ∃ (p : ℝ → ℝ), ∀ x, ∃ (r : ℝ), polynomial x = -6 * x^5 + r ∧ (∀ y, |y| ≥ 1 → |r| ≤ |y|^5 * |-6 * x^5|) :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_six_l817_81762


namespace NUMINAMATH_CALUDE_prime_solution_equation_l817_81759

theorem prime_solution_equation : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l817_81759


namespace NUMINAMATH_CALUDE_product_of_x_and_y_l817_81723

theorem product_of_x_and_y (x y : ℝ) : 
  (-3 * x + 4 * y = 28) → (3 * x - 2 * y = 8) → x * y = 264 :=
by
  sorry


end NUMINAMATH_CALUDE_product_of_x_and_y_l817_81723


namespace NUMINAMATH_CALUDE_brenda_sally_meeting_distance_l817_81755

theorem brenda_sally_meeting_distance 
  (track_length : ℝ) 
  (sally_extra_distance : ℝ) 
  (h1 : track_length = 300)
  (h2 : sally_extra_distance = 100) :
  let first_meeting_distance := (track_length / 2 + sally_extra_distance) / 2
  first_meeting_distance = 150 := by
  sorry

end NUMINAMATH_CALUDE_brenda_sally_meeting_distance_l817_81755


namespace NUMINAMATH_CALUDE_fraction_sum_bound_l817_81795

theorem fraction_sum_bound (a b c : ℕ) (h : (1 : ℚ) / a + 1 / b + 1 / c < 1) :
  (1 : ℚ) / a + 1 / b + 1 / c < 41 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_bound_l817_81795


namespace NUMINAMATH_CALUDE_negation_of_false_is_true_l817_81760

theorem negation_of_false_is_true (p q : Prop) 
  (hp : p) (hq : ¬q) : ¬q := by sorry

end NUMINAMATH_CALUDE_negation_of_false_is_true_l817_81760


namespace NUMINAMATH_CALUDE_problem_solution_l817_81769

def f (a x : ℝ) : ℝ := a * x^2 + x - a

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ≤ -2 ∨ x ≥ 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x > -2*x^2 - 3*x + 1 - 2*a) ↔ a > 2) ∧
  (∀ a : ℝ, a < 0 →
    ((-1/2 < a ∧ a < 0 ∧ ∀ x : ℝ, f a x > 1 ↔ 1 < x ∧ x < -(a+1)/a) ∨
     (a = -1/2 ∧ ∀ x : ℝ, ¬(f a x > 1)) ∨
     (a < -1/2 ∧ ∀ x : ℝ, f a x > 1 ↔ -(a+1)/a < x ∧ x < 1))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l817_81769


namespace NUMINAMATH_CALUDE_jake_final_bitcoin_count_l817_81701

/-- Calculates the final number of bitcoins Jake has after a series of transactions -/
def final_bitcoin_count (initial : ℕ) (first_donation : ℕ) (second_donation : ℕ) : ℕ :=
  let after_first_donation := initial - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  after_tripling - second_donation

/-- Theorem stating that Jake ends up with 80 bitcoins -/
theorem jake_final_bitcoin_count :
  final_bitcoin_count 80 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_bitcoin_count_l817_81701


namespace NUMINAMATH_CALUDE_nail_color_percentage_difference_l817_81786

theorem nail_color_percentage_difference :
  let total_nails : ℕ := 20
  let purple_nails : ℕ := 6
  let blue_nails : ℕ := 8
  let striped_nails : ℕ := total_nails - purple_nails - blue_nails
  let blue_percentage : ℚ := blue_nails / total_nails * 100
  let striped_percentage : ℚ := striped_nails / total_nails * 100
  blue_percentage - striped_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_nail_color_percentage_difference_l817_81786


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_divisibility_of_four_consecutive_integers_optimality_of_twelve_l817_81758

/-- The greatest whole number that must be a divisor of the product of any four consecutive positive integers is 12. -/
theorem greatest_divisor_four_consecutive_integers : ℕ :=
  let f : ℕ → ℕ := λ n => n * (n + 1) * (n + 2) * (n + 3)
  12

theorem divisibility_of_four_consecutive_integers (n : ℕ) :
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

theorem optimality_of_twelve (m : ℕ) :
  (∀ n : ℕ, m ∣ (n * (n + 1) * (n + 2) * (n + 3))) → m ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_divisibility_of_four_consecutive_integers_optimality_of_twelve_l817_81758


namespace NUMINAMATH_CALUDE_fourth_side_length_l817_81741

/-- A rhombus inscribed in a circle -/
structure InscribedRhombus where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The length of three sides of the rhombus
  side_length : ℝ
  -- Assumption that the rhombus is actually inscribed in the circle
  is_inscribed : True

/-- Theorem: In a rhombus inscribed in a circle with radius 100√2, 
    if three sides have length 100, then the fourth side also has length 100 -/
theorem fourth_side_length (r : InscribedRhombus) 
    (h1 : r.radius = 100 * Real.sqrt 2) 
    (h2 : r.side_length = 100) : 
  r.side_length = 100 := by
  sorry


end NUMINAMATH_CALUDE_fourth_side_length_l817_81741


namespace NUMINAMATH_CALUDE_flag_design_count_l817_81744

/-- The number of colors available for the flag -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 4

/-- A function that calculates the number of possible flag designs -/
def flag_designs (n : ℕ) (k : ℕ) : ℕ :=
  if n = 1 then k
  else k * (k - 1)^(n - 1)

/-- Theorem stating that the number of possible flag designs is 24 -/
theorem flag_design_count :
  flag_designs num_stripes num_colors = 24 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l817_81744


namespace NUMINAMATH_CALUDE_greatest_m_value_l817_81789

theorem greatest_m_value (p m : ℕ) (hp : Nat.Prime p) 
  (heq : p * (p + m) + 2 * p = (m + 2)^3) : 
  m ≤ 28 ∧ ∃ (p' m' : ℕ), Nat.Prime p' ∧ p' * (p' + 28) + 2 * p' = (28 + 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_m_value_l817_81789


namespace NUMINAMATH_CALUDE_trig_identities_l817_81771

theorem trig_identities (θ : Real) (h : Real.sin (θ - π/3) = 1/3) :
  (Real.sin (θ + 2*π/3) = -1/3) ∧ (Real.cos (θ - 5*π/6) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l817_81771


namespace NUMINAMATH_CALUDE_right_triangle_area_l817_81757

theorem right_triangle_area (α : Real) (hypotenuse : Real) :
  α = 30 * π / 180 →
  hypotenuse = 20 →
  ∃ (area : Real), area = 50 * Real.sqrt 3 ∧
    area = (1 / 2) * (hypotenuse / 2) * (hypotenuse / 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l817_81757


namespace NUMINAMATH_CALUDE_max_xy_min_inverse_sum_l817_81710

-- Define the conditions
variable (x y : ℝ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x + 4*y = 4)

-- Theorem for the maximum value of xy
theorem max_xy : ∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → xy ≤ 1 := by
  sorry

-- Theorem for the minimum value of 1/x + 2/y
theorem min_inverse_sum : ∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → 1/x + 2/y ≥ (9 + 4*Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_min_inverse_sum_l817_81710


namespace NUMINAMATH_CALUDE_stock_change_is_negative_4_375_percent_l817_81780

/-- The overall percent change in a stock value after three days of fluctuations -/
def stock_percent_change : ℝ := by
  -- Define the daily changes
  let day1_change : ℝ := 0.85  -- 15% decrease
  let day2_change : ℝ := 1.25  -- 25% increase
  let day3_change : ℝ := 0.90  -- 10% decrease

  -- Calculate the overall change
  let overall_change : ℝ := day1_change * day2_change * day3_change

  -- Calculate the percent change
  exact (overall_change - 1) * 100

/-- Theorem stating that the overall percent change in the stock is -4.375% -/
theorem stock_change_is_negative_4_375_percent : 
  stock_percent_change = -4.375 := by
  sorry

end NUMINAMATH_CALUDE_stock_change_is_negative_4_375_percent_l817_81780


namespace NUMINAMATH_CALUDE_additional_amount_is_three_l817_81791

/-- The minimum purchase amount required for free delivery -/
def min_purchase : ℝ := 18

/-- The cost of a quarter-pounder burger -/
def burger_cost : ℝ := 3.20

/-- The cost of large fries -/
def fries_cost : ℝ := 1.90

/-- The cost of a milkshake -/
def milkshake_cost : ℝ := 2.40

/-- The number of each item Danny ordered -/
def quantity : ℕ := 2

/-- The total cost of Danny's current order -/
def order_total : ℝ := quantity * burger_cost + quantity * fries_cost + quantity * milkshake_cost

/-- The additional amount needed for free delivery -/
def additional_amount : ℝ := min_purchase - order_total

theorem additional_amount_is_three :
  additional_amount = 3 :=
by sorry

end NUMINAMATH_CALUDE_additional_amount_is_three_l817_81791


namespace NUMINAMATH_CALUDE_wrapping_paper_usage_l817_81711

theorem wrapping_paper_usage
  (total_paper : ℚ)
  (small_presents : ℕ)
  (large_presents : ℕ)
  (h1 : total_paper = 5 / 12)
  (h2 : small_presents = 4)
  (h3 : large_presents = 2)
  (h4 : small_presents + large_presents = 6) :
  ∃ (small_paper large_paper : ℚ),
    small_paper * small_presents + large_paper * large_presents = total_paper ∧
    large_paper = 2 * small_paper ∧
    small_paper = 5 / 96 ∧
    large_paper = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_usage_l817_81711


namespace NUMINAMATH_CALUDE_twelve_chairs_subsets_l817_81736

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets containing at least four adjacent chairs
    for n chairs arranged in a circle -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle,
    the number of subsets containing at least four adjacent chairs is 1701 -/
theorem twelve_chairs_subsets :
  subsets_with_adjacent_chairs n = 1701 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_subsets_l817_81736


namespace NUMINAMATH_CALUDE_no_leg_longer_than_both_l817_81778

-- Define two right triangles
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem no_leg_longer_than_both (t1 t2 : RightTriangle) 
  (h : t1.hypotenuse = t2.hypotenuse) : 
  ¬(t1.leg1 > t2.leg1 ∧ t1.leg1 > t2.leg2) ∨ 
  ¬(t1.leg2 > t2.leg1 ∧ t1.leg2 > t2.leg2) :=
sorry

end NUMINAMATH_CALUDE_no_leg_longer_than_both_l817_81778


namespace NUMINAMATH_CALUDE_polyhedron_20_faces_l817_81718

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The Euler characteristic for polyhedra -/
def euler_characteristic (p : Polyhedron) : ℕ :=
  p.vertices - p.edges + p.faces

/-- Theorem: A polyhedron with 20 triangular faces has 30 edges and 12 vertices -/
theorem polyhedron_20_faces (p : Polyhedron) 
  (h_faces : p.faces = 20) 
  (h_triangular : p.edges * 2 = p.faces * 3) 
  (h_euler : euler_characteristic p = 2) : 
  p.edges = 30 ∧ p.vertices = 12 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_20_faces_l817_81718


namespace NUMINAMATH_CALUDE_M_equals_interval_l817_81727

/-- The set of real numbers m for which there exists an x in (-1, 1) satisfying x^2 - x - m = 0 -/
def M : Set ℝ := {m : ℝ | ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

/-- The theorem stating that M is equal to [-1/4, 2) -/
theorem M_equals_interval : M = Set.Icc (-1/4) 2 := by
  sorry

end NUMINAMATH_CALUDE_M_equals_interval_l817_81727


namespace NUMINAMATH_CALUDE_book_selection_theorem_l817_81743

/-- Given the number of books in each language, calculates the number of ways to select two books. -/
def book_selection (japanese : ℕ) (english : ℕ) (chinese : ℕ) :
  (ℕ × ℕ × ℕ) :=
  let different_languages := japanese * english + japanese * chinese + english * chinese
  let same_language := japanese * (japanese - 1) / 2 + english * (english - 1) / 2 + chinese * (chinese - 1) / 2
  let total := (japanese + english + chinese) * (japanese + english + chinese - 1) / 2
  (different_languages, same_language, total)

/-- Theorem stating the correct number of ways to select books given the specified quantities. -/
theorem book_selection_theorem :
  book_selection 5 7 10 = (155, 76, 231) := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l817_81743


namespace NUMINAMATH_CALUDE_three_number_ratio_sum_l817_81777

theorem three_number_ratio_sum (a b c : ℝ) : 
  (a : ℝ) > 0 ∧ b = 2 * a ∧ c = 4 * a ∧ a^2 + b^2 + c^2 = 1701 →
  a + b + c = 63 := by
sorry

end NUMINAMATH_CALUDE_three_number_ratio_sum_l817_81777


namespace NUMINAMATH_CALUDE_books_read_first_week_l817_81767

/-- The number of books read in the first week of a 7-week reading plan -/
def books_first_week (total_books : ℕ) (second_week : ℕ) (later_weeks : ℕ) : ℕ :=
  total_books - second_week - (later_weeks * 5)

theorem books_read_first_week :
  books_first_week 54 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_read_first_week_l817_81767


namespace NUMINAMATH_CALUDE_where_is_waldo_books_l817_81793

/-- The number of "Where's Waldo?" books published -/
def num_books : ℕ := 15

/-- The number of puzzles in each "Where's Waldo?" book -/
def puzzles_per_book : ℕ := 30

/-- The time in minutes to solve one puzzle -/
def time_per_puzzle : ℕ := 3

/-- The total time in minutes to solve all puzzles -/
def total_time : ℕ := 1350

/-- Theorem stating that the number of "Where's Waldo?" books is correct -/
theorem where_is_waldo_books :
  num_books = total_time / (puzzles_per_book * time_per_puzzle) :=
by sorry

end NUMINAMATH_CALUDE_where_is_waldo_books_l817_81793


namespace NUMINAMATH_CALUDE_toms_bowling_score_l817_81704

theorem toms_bowling_score (tom jerry : ℕ) : 
  tom = jerry + 30 → 
  (tom + jerry) / 2 = 90 → 
  tom = 105 := by
sorry

end NUMINAMATH_CALUDE_toms_bowling_score_l817_81704


namespace NUMINAMATH_CALUDE_fraction_proof_l817_81724

theorem fraction_proof (t k : ℚ) (f : ℚ) 
  (h1 : t = f * (k - 32))
  (h2 : t = 105)
  (h3 : k = 221) :
  f = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_proof_l817_81724


namespace NUMINAMATH_CALUDE_dani_pants_count_l817_81728

/-- Calculate the final number of pants after receiving a certain number of pairs each year for a given period. -/
def final_pants_count (initial_pants : ℕ) (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) : ℕ :=
  initial_pants + pairs_per_year * pants_per_pair * years

/-- Theorem stating that Dani will have 90 pants after 5 years -/
theorem dani_pants_count : final_pants_count 50 4 2 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_count_l817_81728


namespace NUMINAMATH_CALUDE_total_rejection_is_0_75_percent_l817_81774

-- Define the rejection rates and inspection proportion
def john_rejection_rate : ℝ := 0.005
def jane_rejection_rate : ℝ := 0.009
def jane_inspection_proportion : ℝ := 0.625

-- Define the total rejection percentage
def total_rejection_percentage : ℝ :=
  jane_rejection_rate * jane_inspection_proportion +
  john_rejection_rate * (1 - jane_inspection_proportion)

-- Theorem statement
theorem total_rejection_is_0_75_percent :
  total_rejection_percentage = 0.0075 := by
  sorry

#eval total_rejection_percentage

end NUMINAMATH_CALUDE_total_rejection_is_0_75_percent_l817_81774


namespace NUMINAMATH_CALUDE_average_time_is_five_l817_81756

/-- Colin's running times for each mile -/
def mile_times : List ℕ := [6, 5, 5, 4]

/-- Total number of miles run -/
def total_miles : ℕ := mile_times.length

/-- Calculates the average time per mile -/
def average_time_per_mile : ℚ :=
  (mile_times.sum : ℚ) / total_miles

/-- Theorem: The average time per mile is 5 minutes -/
theorem average_time_is_five : average_time_per_mile = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_time_is_five_l817_81756


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l817_81772

theorem average_of_five_numbers : 
  let numbers : List ℕ := [8, 9, 10, 11, 12]
  (numbers.sum / numbers.length : ℚ) = 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l817_81772


namespace NUMINAMATH_CALUDE_total_segment_length_l817_81726

-- Define the grid dimensions
def grid_width : ℕ := 5
def grid_height : ℕ := 6

-- Define the number of unit squares
def total_squares : ℕ := 30

-- Define the lengths of the six line segments
def segment_lengths : List ℕ := [5, 1, 4, 2, 3, 3]

-- Theorem statement
theorem total_segment_length :
  grid_width = 5 ∧ 
  grid_height = 6 ∧ 
  total_squares = 30 ∧ 
  segment_lengths = [5, 1, 4, 2, 3, 3] →
  List.sum segment_lengths = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_segment_length_l817_81726


namespace NUMINAMATH_CALUDE_ant_borya_position_l817_81790

/-- Represents a point on the coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Generates the nth point in the spiral sequence -/
def spiral_point (n : Nat) : Point :=
  sorry

/-- The starting point of the sequence -/
def P₀ : Point := { x := 0, y := 0 }

/-- The second point in the sequence -/
def P₁ : Point := { x := 1, y := 0 }

/-- The spiral sequence of points -/
def P : Nat → Point
  | 0 => P₀
  | 1 => P₁
  | n + 2 => spiral_point (n + 2)

theorem ant_borya_position : P 1557 = { x := 20, y := 17 } := by
  sorry

end NUMINAMATH_CALUDE_ant_borya_position_l817_81790


namespace NUMINAMATH_CALUDE_pool_filling_times_l817_81768

theorem pool_filling_times (t₁ t₂ : ℝ) : 
  (t₁ > 0 ∧ t₂ > 0) →  -- Ensure positive times
  (1 / t₁ + 1 / t₂ = 1 / 2.4) →  -- Combined filling rate
  (t₂ / (4 * t₁) + t₁ / (4 * t₂) = 11 / 24) →  -- Fraction filled by individual operations
  (t₁ = 4 ∧ t₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_times_l817_81768


namespace NUMINAMATH_CALUDE_line_equation_l817_81749

theorem line_equation (slope_angle : Real) (y_intercept : Real) :
  slope_angle = Real.pi / 4 → y_intercept = 2 →
  ∃ f : Real → Real, f = λ x => x + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_l817_81749


namespace NUMINAMATH_CALUDE_xy_value_l817_81776

theorem xy_value (x y : ℝ) (h : Real.sqrt (x - 1) + (y - 2)^2 = 0) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l817_81776


namespace NUMINAMATH_CALUDE_polygon_area_bounds_l817_81739

/-- Represents a polygon in 2D space -/
structure Polygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents the projections of a polygon -/
structure Projections where
  x_axis : ℝ
  y_axis : ℝ
  bisector_1_3 : ℝ
  bisector_2_4 : ℝ

/-- Given a polygon, return its projections -/
def get_projections (p : Polygon) : Projections :=
  sorry

/-- Calculate the area of a polygon -/
def area (p : Polygon) : ℝ :=
  sorry

/-- Check if a polygon is convex -/
def is_convex (p : Polygon) : Prop :=
  sorry

theorem polygon_area_bounds (M : Polygon) 
    (h_proj : get_projections M = Projections.mk 4 5 (3 * Real.sqrt 2) (4 * Real.sqrt 2)) : 
  (area M ≤ 17.5) ∧ (is_convex M → area M ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_bounds_l817_81739


namespace NUMINAMATH_CALUDE_angle_between_vectors_l817_81753

def vector1 : Fin 2 → ℝ := ![2, 5]
def vector2 : Fin 2 → ℝ := ![-3, 7]

theorem angle_between_vectors (v1 v2 : Fin 2 → ℝ) :
  v1 = vector1 → v2 = vector2 →
  Real.arccos ((v1 0 * v2 0 + v1 1 * v2 1) /
    (Real.sqrt (v1 0 ^ 2 + v1 1 ^ 2) * Real.sqrt (v2 0 ^ 2 + v2 1 ^ 2))) =
  45 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l817_81753


namespace NUMINAMATH_CALUDE_polynomial_simplification_l817_81765

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 2 * q^3 + 3 * q^2 - 7 * q + 9) + (5 * q^3 - 8 * q^2 + 6 * q - 1) =
  4 * q^4 + 3 * q^3 - 5 * q^2 - q + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l817_81765


namespace NUMINAMATH_CALUDE_trigonometric_identities_l817_81787

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (2 * sin α - cos α) / (sin α + 2 * cos α) = 3/4 ∧
  2 * sin α^2 - sin α * cos α + cos α^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l817_81787


namespace NUMINAMATH_CALUDE_thread_length_calculation_l817_81722

theorem thread_length_calculation (original_length : ℝ) (additional_fraction : ℝ) : 
  original_length = 12 →
  additional_fraction = 3/4 →
  original_length + (additional_fraction * original_length) = 21 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_calculation_l817_81722


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l817_81706

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 1501 * C - 3003 * A = 6006)
  (eq2 : 1501 * B + 4504 * A = 7507)
  (eq3 : A + B = 1) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l817_81706


namespace NUMINAMATH_CALUDE_chef_apple_pies_l817_81752

theorem chef_apple_pies (total pies : ℕ) (pecan pumpkin apple : ℕ) : 
  total = 13 → pecan = 4 → pumpkin = 7 → total = apple + pecan + pumpkin → apple = 2 := by
  sorry

end NUMINAMATH_CALUDE_chef_apple_pies_l817_81752


namespace NUMINAMATH_CALUDE_grid_paths_6_5_l817_81792

/-- The number of distinct paths on a grid from (0,0) to (m,n) using only right and up moves -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

theorem grid_paths_6_5 : gridPaths 6 5 = 462 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_6_5_l817_81792


namespace NUMINAMATH_CALUDE_unique_triangle_solution_l817_81770

theorem unique_triangle_solution (a b : ℝ) (A : ℝ) (ha : a = 30) (hb : b = 25) (hA : A = 150 * π / 180) :
  ∃! (c : ℝ) (B C : ℝ), 
    0 < c ∧ 0 < B ∧ 0 < C ∧
    a / Real.sin A = b / Real.sin B ∧
    b / Real.sin B = c / Real.sin C ∧
    A + B + C = π := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_solution_l817_81770


namespace NUMINAMATH_CALUDE_geometric_mean_of_2_and_8_l817_81732

theorem geometric_mean_of_2_and_8 : 
  ∃ (b : ℝ), b^2 = 2 * 8 ∧ (b = 4 ∨ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_2_and_8_l817_81732


namespace NUMINAMATH_CALUDE_inequality_proof_l817_81750

open BigOperators

theorem inequality_proof (n : ℕ) (δ : ℝ) (a b : ℕ → ℝ) 
  (h_pos_a : ∀ i ∈ Finset.range (n + 1), a i > 0)
  (h_pos_b : ∀ i ∈ Finset.range (n + 1), b i > 0)
  (h_delta : ∀ i ∈ Finset.range n, b (i + 1) - b i ≥ δ)
  (h_delta_pos : δ > 0)
  (h_sum_a : ∑ i in Finset.range n, a i = 1) :
  ∑ i in Finset.range n, (i + 1 : ℝ) * (∏ j in Finset.range (i + 1), (a j * b j)) ^ (1 / (i + 1 : ℝ)) / (b (i + 1) * b i) < 1 / δ :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l817_81750


namespace NUMINAMATH_CALUDE_committee_count_12_5_l817_81773

/-- The number of ways to choose a committee of size k from a group of n people -/
def committeeCount (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the entire group -/
def groupSize : ℕ := 12

/-- The size of the committee to be chosen -/
def committeeSize : ℕ := 5

/-- Theorem stating that the number of ways to choose a 5-person committee from 12 people is 792 -/
theorem committee_count_12_5 : 
  committeeCount groupSize committeeSize = 792 := by sorry

end NUMINAMATH_CALUDE_committee_count_12_5_l817_81773


namespace NUMINAMATH_CALUDE_sector_central_angle_l817_81784

/-- Given a sector with radius 2 and area 4, its central angle (in absolute value) is 2 radians -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 4) :
  |2 * area / r^2| = 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l817_81784


namespace NUMINAMATH_CALUDE_factorial_division_l817_81779

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l817_81779


namespace NUMINAMATH_CALUDE_three_equal_perimeter_triangles_l817_81799

def stick_lengths : List ℕ := [2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 9]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def forms_triangle (lengths : List ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  is_triangle a b c ∧
  a + b + c = 14

theorem three_equal_perimeter_triangles :
  ∃ (t1 t2 t3 : List ℕ),
    t1 ⊆ stick_lengths ∧
    t2 ⊆ stick_lengths ∧
    t3 ⊆ stick_lengths ∧
    t1 ∩ t2 = ∅ ∧ t2 ∩ t3 = ∅ ∧ t3 ∩ t1 = ∅ ∧
    forms_triangle t1 ∧
    forms_triangle t2 ∧
    forms_triangle t3 :=
  sorry

end NUMINAMATH_CALUDE_three_equal_perimeter_triangles_l817_81799


namespace NUMINAMATH_CALUDE_project_update_lcm_l817_81775

theorem project_update_lcm : Nat.lcm 5 (Nat.lcm 9 (Nat.lcm 10 13)) = 1170 := by
  sorry

end NUMINAMATH_CALUDE_project_update_lcm_l817_81775


namespace NUMINAMATH_CALUDE_rhombus_existence_condition_l817_81797

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  /-- The perimeter of the rhombus -/
  k : ℝ
  /-- The sum of the diagonals of the rhombus -/
  u : ℝ
  /-- The perimeter is positive -/
  k_pos : k > 0
  /-- The sum of diagonals is positive -/
  u_pos : u > 0

/-- The condition for the existence of a rhombus given its perimeter and sum of diagonals -/
theorem rhombus_existence_condition (r : Rhombus) : 
  Real.sqrt 2 * r.u ≤ r.k ∧ r.k < 2 * r.u :=
by sorry

end NUMINAMATH_CALUDE_rhombus_existence_condition_l817_81797


namespace NUMINAMATH_CALUDE_reciprocals_from_product_l817_81745

theorem reciprocals_from_product (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_reciprocals_from_product_l817_81745


namespace NUMINAMATH_CALUDE_tank_fill_time_l817_81703

/-- Proves that the time required to fill 3/4 of a 4000-gallon tank at a rate of 10 gallons per hour is 300 hours. -/
theorem tank_fill_time (tank_capacity : ℝ) (fill_rate : ℝ) (fill_fraction : ℝ) (fill_time : ℝ) :
  tank_capacity = 4000 →
  fill_rate = 10 →
  fill_fraction = 3/4 →
  fill_time = (fill_fraction * tank_capacity) / fill_rate →
  fill_time = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l817_81703


namespace NUMINAMATH_CALUDE_f_minimum_and_inequality_l817_81740

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem f_minimum_and_inequality :
  (∃ (x_min : ℝ), ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = 1) ∧
  (∀ (x : ℝ), x > 0 → x * (Real.exp x) * f x + (x * Real.exp x - 1) * Real.log x - Real.exp x + 1/2 > 0) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_and_inequality_l817_81740


namespace NUMINAMATH_CALUDE_geologists_can_reach_station_l817_81746

/-- Represents the problem of geologists traveling to a station. -/
structure GeologistsProblem where
  totalDistance : ℝ
  timeLimit : ℝ
  motorcycleSpeed : ℝ
  walkingSpeed : ℝ
  numberOfGeologists : ℕ

/-- Checks if the geologists can reach the station within the time limit. -/
def canReachStation (problem : GeologistsProblem) : Prop :=
  ∃ (strategy : Unit), 
    let twoGeologistsTime := problem.totalDistance / problem.motorcycleSpeed
    let walkingTime := problem.totalDistance / problem.walkingSpeed
    let meetingTime := (problem.totalDistance - problem.walkingSpeed) / (problem.motorcycleSpeed + problem.walkingSpeed)
    let returnTime := (problem.totalDistance - problem.walkingSpeed * meetingTime) / problem.motorcycleSpeed
    twoGeologistsTime ≤ problem.timeLimit ∧ 
    walkingTime ≤ problem.timeLimit ∧
    meetingTime + returnTime ≤ problem.timeLimit

/-- The specific problem instance. -/
def geologistsProblem : GeologistsProblem :=
  { totalDistance := 60
  , timeLimit := 3
  , motorcycleSpeed := 50
  , walkingSpeed := 5
  , numberOfGeologists := 3 }

/-- Theorem stating that the geologists can reach the station within the time limit. -/
theorem geologists_can_reach_station : canReachStation geologistsProblem := by
  sorry


end NUMINAMATH_CALUDE_geologists_can_reach_station_l817_81746


namespace NUMINAMATH_CALUDE_inequality_proof_l817_81716

theorem inequality_proof (x y : ℝ) (n k : ℕ) 
  (h1 : x > y) (h2 : y > 0) (h3 : n > k) :
  (x^k - y^k)^n < (x^n - y^n)^k := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l817_81716


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l817_81713

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 15 → 
  ∀ x, x^2 - 16*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l817_81713


namespace NUMINAMATH_CALUDE_complex_equation_sum_l817_81796

theorem complex_equation_sum (m n : ℝ) : 
  (m + n * Complex.I) * (4 - 2 * Complex.I) = 3 * Complex.I + 5 → m + n = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l817_81796


namespace NUMINAMATH_CALUDE_initial_money_calculation_l817_81798

theorem initial_money_calculation (initial_amount : ℝ) : 
  (initial_amount / 2 - (initial_amount / 2) / 2 = 51) → initial_amount = 204 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l817_81798


namespace NUMINAMATH_CALUDE_three_intersection_points_l817_81781

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 8

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by sorry

end NUMINAMATH_CALUDE_three_intersection_points_l817_81781


namespace NUMINAMATH_CALUDE_intersection_equality_l817_81754

theorem intersection_equality (a : ℝ) : 
  (∀ x, (1 < x ∧ x < 7) ∧ (a + 1 < x ∧ x < 2*a + 5) ↔ (3 < x ∧ x < 7)) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_l817_81754


namespace NUMINAMATH_CALUDE_sin_theta_value_l817_81733

theorem sin_theta_value (θ : Real) 
  (h1 : 5 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l817_81733


namespace NUMINAMATH_CALUDE_tan_two_implies_fraction_four_fifths_l817_81719

theorem tan_two_implies_fraction_four_fifths (θ : Real) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_fraction_four_fifths_l817_81719
