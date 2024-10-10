import Mathlib

namespace pure_imaginary_complex_fraction_l3575_357550

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I * (((a + 3 * Complex.I) / (1 - 2 * Complex.I)).im) = ((a + 3 * Complex.I) / (1 - 2 * Complex.I))) → a = 6 := by
  sorry

end pure_imaginary_complex_fraction_l3575_357550


namespace larry_road_trip_money_l3575_357554

theorem larry_road_trip_money (initial_money : ℝ) : 
  initial_money * (1 - 0.04 - 0.30) - 52 = 368 → 
  initial_money = 636.36 := by
sorry

end larry_road_trip_money_l3575_357554


namespace speed_difference_20_l3575_357576

/-- The speed equation for a subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- Theorem stating that the speed difference between 5 and 3 seconds is 20 km/h -/
theorem speed_difference_20 : speed 5 - speed 3 = 20 := by sorry

end speed_difference_20_l3575_357576


namespace smallest_chord_length_l3575_357518

/-- The circle equation x^2 + y^2 + 4x - 6y + 4 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y + 4 = 0

/-- The line equation mx - y + 1 = 0 -/
def line_equation (m x y : ℝ) : Prop :=
  m*x - y + 1 = 0

/-- The chord length for a given m -/
noncomputable def chord_length (m : ℝ) : ℝ :=
  sorry  -- Definition of chord length in terms of m

theorem smallest_chord_length :
  ∃ (m : ℝ), ∀ (m' : ℝ), chord_length m ≤ chord_length m' ∧ m = 1 :=
by sorry

end smallest_chord_length_l3575_357518


namespace spring_scale_reading_comparison_l3575_357535

/-- Represents the angular velocity of Earth's rotation -/
def earth_angular_velocity : ℝ := sorry

/-- Represents the radius of the Earth at the equator -/
def earth_equator_radius : ℝ := sorry

/-- Represents the acceleration due to gravity at the equator -/
def gravity_equator : ℝ := sorry

/-- Represents the acceleration due to gravity at the poles -/
def gravity_pole : ℝ := sorry

/-- Calculates the centrifugal force at the equator for an object of mass m -/
def centrifugal_force (m : ℝ) : ℝ :=
  m * earth_angular_velocity^2 * earth_equator_radius

/-- Calculates the apparent weight of an object at the equator -/
def apparent_weight_equator (m : ℝ) : ℝ :=
  m * gravity_equator - centrifugal_force m

/-- Calculates the apparent weight of an object at the pole -/
def apparent_weight_pole (m : ℝ) : ℝ :=
  m * gravity_pole

theorem spring_scale_reading_comparison (m : ℝ) (m_pos : m > 0) :
  apparent_weight_pole m > apparent_weight_equator m :=
by
  sorry

end spring_scale_reading_comparison_l3575_357535


namespace radical_simplification_l3575_357567

theorem radical_simplification : 
  Real.sqrt ((16^12 + 8^14) / (16^5 + 8^16 + 2^24)) = 2^11 * Real.sqrt (65/17) := by
  sorry

end radical_simplification_l3575_357567


namespace age_ratio_problem_l3575_357565

/-- Given Mike's current age m and Dan's current age d, prove that the number of years
    until their age ratio is 3:2 is 97, given the initial conditions. -/
theorem age_ratio_problem (m d : ℕ) (h1 : m - 3 = 4 * (d - 3)) (h2 : m - 8 = 5 * (d - 8)) :
  ∃ x : ℕ, x = 97 ∧ (m + x : ℚ) / (d + x) = 3 / 2 :=
sorry

end age_ratio_problem_l3575_357565


namespace circle_arc_angle_l3575_357507

theorem circle_arc_angle (E AB BC CD AD : ℝ) : 
  E = 40 →
  AB = BC →
  BC = CD →
  AB + BC + CD + AD = 360 →
  (AB - AD) / 2 = E →
  ∃ (ACD : ℝ), ACD = 15 := by
sorry

end circle_arc_angle_l3575_357507


namespace complement_of_A_in_U_l3575_357581

def U : Set ℝ := {x | Real.exp x > 1}
def A : Set ℝ := {x | x > 1}

theorem complement_of_A_in_U : Set.compl A ∩ U = Set.Ioo 0 1 ∪ Set.singleton 1 := by sorry

end complement_of_A_in_U_l3575_357581


namespace smallest_semicircle_area_l3575_357530

/-- Given a right-angled triangle with semicircles on each side, prove that the smallest semicircle has area 144 -/
theorem smallest_semicircle_area (x : ℝ) : 
  x > 0 ∧ x^2 < 180 ∧ 3*x < 180 ∧ x^2 + 3*x = 180 → x^2 = 144 := by
  sorry

end smallest_semicircle_area_l3575_357530


namespace roger_money_theorem_l3575_357563

def roger_money_problem (initial_amount gift_amount spent_amount : ℕ) : Prop :=
  initial_amount + gift_amount - spent_amount = 19

theorem roger_money_theorem :
  roger_money_problem 16 28 25 := by
  sorry

end roger_money_theorem_l3575_357563


namespace prob_boy_girl_twins_l3575_357506

/-- The probability of twins being born -/
def prob_twins : ℚ := 3 / 250

/-- The probability of twins being identical, given that they are twins -/
def prob_identical_given_twins : ℚ := 1 / 3

/-- The probability of twins being fraternal, given that they are twins -/
def prob_fraternal_given_twins : ℚ := 1 - prob_identical_given_twins

/-- The probability of fraternal twins being a boy and a girl -/
def prob_boy_girl_given_fraternal : ℚ := 1 / 2

/-- The theorem stating the probability of a pregnant woman giving birth to boy-girl twins -/
theorem prob_boy_girl_twins : 
  prob_twins * prob_fraternal_given_twins * prob_boy_girl_given_fraternal = 1 / 250 := by
  sorry

end prob_boy_girl_twins_l3575_357506


namespace rhombus_area_from_overlapping_strips_l3575_357516

/-- The area of a rhombus formed by two overlapping strips -/
theorem rhombus_area_from_overlapping_strips (β : Real) (h : β ≠ 0) : 
  let strip_width : Real := 2
  let diagonal1 : Real := strip_width
  let diagonal2 : Real := strip_width / Real.sin β
  let rhombus_area : Real := (1 / 2) * diagonal1 * diagonal2
  rhombus_area = 2 / Real.sin β :=
by sorry

end rhombus_area_from_overlapping_strips_l3575_357516


namespace intersection_points_sum_l3575_357583

/-- The quadratic function f(x) = (x+2)(x-4) -/
def f (x : ℝ) : ℝ := (x + 2) * (x - 4)

/-- The function g(x) = -f(x) -/
def g (x : ℝ) : ℝ := -f x

/-- The function h(x) = f(-x) -/
def h (x : ℝ) : ℝ := f (-x)

/-- The number of intersection points between y=f(x) and y=g(x) -/
def a : ℕ := 2

/-- The number of intersection points between y=f(x) and y=h(x) -/
def b : ℕ := 1

theorem intersection_points_sum : 10 * a + b = 21 := by
  sorry

end intersection_points_sum_l3575_357583


namespace geometric_sum_l3575_357568

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  a 8 + a 12 = 24 := by
sorry

end geometric_sum_l3575_357568


namespace actual_speed_is_22_5_l3575_357536

/-- Proves that the actual average speed is 22.5 mph given the conditions of the problem -/
theorem actual_speed_is_22_5 (v t : ℝ) (h : v > 0) (h' : t > 0) :
  (v * t = (v + 37.5) * (3/8 * t)) → v = 22.5 := by
  sorry

end actual_speed_is_22_5_l3575_357536


namespace odd_number_pattern_l3575_357524

/-- Represents the number of odd numbers in a row of the pattern -/
def row_length (n : ℕ) : ℕ := 2 * n - 1

/-- Calculates the sum of odd numbers up to the nth row -/
def sum_to_row (n : ℕ) : ℕ := n^2

/-- Represents the nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- The problem statement -/
theorem odd_number_pattern :
  let total_previous_rows := sum_to_row 20
  let position_in_row := 6
  nth_odd (total_previous_rows + position_in_row) = 811 := by
  sorry

end odd_number_pattern_l3575_357524


namespace integral_identity_l3575_357519

theorem integral_identity : ∫ (x : ℝ) in -Real.arctan (1/3)..0, (3 * Real.tan x + 1) / (2 * Real.sin (2*x) - 5 * Real.cos (2*x) + 1) = (1/4) * Real.log (6/5) := by
  sorry

end integral_identity_l3575_357519


namespace max_ab_for_tangent_circle_l3575_357599

/-- Given a line l: x + 2y = 0 tangent to a circle C: (x-a)² + (y-b)² = 5,
    where the center (a,b) of C is above l, the maximum value of ab is 25/8 -/
theorem max_ab_for_tangent_circle (a b : ℝ) : 
  (∀ x y : ℝ, (x + 2*y = 0) → ((x - a)^2 + (y - b)^2 = 5)) →  -- tangency condition
  (a + 2*b > 0) →  -- center above the line
  (∀ a' b' : ℝ, (∀ x y : ℝ, (x + 2*y = 0) → ((x - a')^2 + (y - b')^2 = 5)) → 
                (a' + 2*b' > 0) → 
                a * b ≤ a' * b') →
  a * b = 25/8 := by sorry

end max_ab_for_tangent_circle_l3575_357599


namespace product_sum_theorem_l3575_357586

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  a*b + b*c + a*c = 131 := by
sorry

end product_sum_theorem_l3575_357586


namespace spade_evaluation_l3575_357527

-- Define the ♠ operation
def spade (a b : ℚ) : ℚ := (3 * a + b) / (a + b)

-- Theorem statement
theorem spade_evaluation :
  spade (spade 5 (spade 3 6)) 1 = 17 / 7 := by
  sorry

end spade_evaluation_l3575_357527


namespace perpendicular_lines_l3575_357515

def line1 (a : ℝ) (x y : ℝ) : Prop := 2*x + a*y = 0

def line2 (a : ℝ) (x y : ℝ) : Prop := x - (a+1)*y = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ → line2 a x₂ y₂ → 
    (x₁ * x₂ + y₁ * y₂ = 0 ∨ (x₁ = 0 ∧ y₁ = 0) ∨ (x₂ = 0 ∧ y₂ = 0))

theorem perpendicular_lines (a : ℝ) : perpendicular a → (a = -2 ∨ a = 1) := by
  sorry

end perpendicular_lines_l3575_357515


namespace geometric_sequence_178th_term_l3575_357579

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_178th_term :
  let a₁ := 5
  let a₂ := -20
  let r := a₂ / a₁
  geometric_sequence a₁ r 178 = -5 * 4^177 := by
  sorry

end geometric_sequence_178th_term_l3575_357579


namespace area_of_quadrilateral_ABCD_l3575_357534

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length 1 -/
def UnitCube : Set Point3D :=
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 1}

/-- The diagonal vertices of the cube -/
def A : Point3D := ⟨0, 0, 0⟩
def C : Point3D := ⟨1, 1, 1⟩

/-- The midpoints of two opposite edges not containing A or C -/
def B : Point3D := ⟨0.5, 0, 1⟩
def D : Point3D := ⟨0.5, 1, 0⟩

/-- The plane passing through A, B, C, and D -/
def InterceptingPlane : Set Point3D :=
  {p | ∃ (s t : ℝ), p = ⟨s, t, 1 - s - t⟩ ∧ 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ t ∧ t ≤ 1}

/-- The quadrilateral ABCD formed by the intersection of the plane and the cube -/
def QuadrilateralABCD : Set Point3D :=
  UnitCube ∩ InterceptingPlane

/-- The area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point3D) : ℝ := sorry

theorem area_of_quadrilateral_ABCD :
  quadrilateralArea A B C D = Real.sqrt 6 / 2 := by sorry

end area_of_quadrilateral_ABCD_l3575_357534


namespace plant_arrangement_count_l3575_357526

/-- The number of ways to arrange plants in a row -/
def arrangePlants (basil tomato pepper : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial basil * Nat.factorial tomato * Nat.factorial pepper

theorem plant_arrangement_count :
  arrangePlants 4 4 3 = 20736 := by
  sorry

end plant_arrangement_count_l3575_357526


namespace triangle_theorem_l3575_357587

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the condition for an acute-angled triangle
def isAcuteAngled (t : Triangle) : Prop := sorry

-- Define the point P on AC
def pointOnAC (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

-- Define the condition 2AP = BC
def conditionAP (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

-- Define points X and Y symmetric to P with respect to A and C
def symmetricPoints (t : Triangle) (P X Y : ℝ × ℝ) : Prop := sorry

-- Define the condition BX = BY
def equalDistances (t : Triangle) (X Y : ℝ × ℝ) : Prop := sorry

-- Define the angle BCA
def angleBCA (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_theorem (t : Triangle) (P X Y : ℝ × ℝ) :
  isAcuteAngled t →
  pointOnAC t P →
  conditionAP t P →
  symmetricPoints t P X Y →
  equalDistances t X Y →
  angleBCA t = 60 := by
  sorry

end triangle_theorem_l3575_357587


namespace min_value_sum_squares_l3575_357590

theorem min_value_sum_squares (x y z : ℝ) (h : 4*x + 3*y + 12*z = 1) :
  x^2 + y^2 + z^2 ≥ 1/169 :=
by sorry

end min_value_sum_squares_l3575_357590


namespace average_weight_decrease_l3575_357548

theorem average_weight_decrease (initial_count : ℕ) (initial_avg : ℝ) (new_weight : ℝ) : 
  initial_count = 20 → 
  initial_avg = 57 → 
  new_weight = 48 → 
  let new_avg := (initial_count * initial_avg + new_weight) / (initial_count + 1)
  initial_avg - new_avg = 0.43 := by sorry

end average_weight_decrease_l3575_357548


namespace max_value_implies_a_values_l3575_357584

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- State the theorem
theorem max_value_implies_a_values (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 2) →
  a = 2 ∨ a = -1 :=
sorry

end max_value_implies_a_values_l3575_357584


namespace area_of_quadrilateral_l3575_357574

/-- Given a quadrilateral ABED where ABE and BED are right triangles sharing base BE,
    with AB = 15, BE = 20, and ED = 25, prove that the area of ABED is 400. -/
theorem area_of_quadrilateral (A B E D : ℝ × ℝ) : 
  let triangle_area (a b : ℝ) := (a * b) / 2
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15^2 →  -- AB = 15
  (B.1 - E.1)^2 + (B.2 - E.2)^2 = 20^2 →  -- BE = 20
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = 25^2 →  -- ED = 25
  (A.1 - E.1) * (B.2 - E.2) = (A.2 - E.2) * (B.1 - E.1) →  -- ABE is right-angled at B
  (B.1 - E.1) * (D.2 - E.2) = (B.2 - E.2) * (D.1 - E.1) →  -- BED is right-angled at E
  triangle_area 15 20 + triangle_area 20 25 = 400 := by
    sorry

end area_of_quadrilateral_l3575_357574


namespace complement_M_intersect_N_l3575_357514

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}

-- Define the set N
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^x ∧ 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem complement_M_intersect_N : (Set.univ \ M) ∩ N = Set.Icc 2 4 := by sorry

end complement_M_intersect_N_l3575_357514


namespace incandescent_bulbs_count_l3575_357571

/-- Represents the waterfall system and power consumption --/
structure WaterfallSystem where
  water_flow : ℝ  -- m³/s
  waterfall_height : ℝ  -- m
  turbine_efficiency : ℝ
  dynamo_efficiency : ℝ
  transmission_efficiency : ℝ
  num_motors : ℕ
  power_per_motor : ℝ  -- horsepower
  motor_efficiency : ℝ
  num_arc_lamps : ℕ
  arc_lamp_voltage : ℝ  -- V
  arc_lamp_current : ℝ  -- A
  incandescent_bulb_power : ℝ  -- W

/-- Calculates the number of incandescent bulbs that can be powered --/
def calculate_incandescent_bulbs (system : WaterfallSystem) : ℕ :=
  sorry

/-- Theorem stating the number of incandescent bulbs that can be powered --/
theorem incandescent_bulbs_count (system : WaterfallSystem) 
  (h1 : system.water_flow = 8)
  (h2 : system.waterfall_height = 5)
  (h3 : system.turbine_efficiency = 0.8)
  (h4 : system.dynamo_efficiency = 0.9)
  (h5 : system.transmission_efficiency = 0.95)
  (h6 : system.num_motors = 5)
  (h7 : system.power_per_motor = 10)
  (h8 : system.motor_efficiency = 0.85)
  (h9 : system.num_arc_lamps = 24)
  (h10 : system.arc_lamp_voltage = 40)
  (h11 : system.arc_lamp_current = 10)
  (h12 : system.incandescent_bulb_power = 55) :
  calculate_incandescent_bulbs system = 3920 :=
sorry

end incandescent_bulbs_count_l3575_357571


namespace rectangle_ratio_l3575_357502

/-- Given a configuration of squares and a rectangle, prove the ratio of rectangle's length to width -/
theorem rectangle_ratio (s : ℝ) (h1 : s > 0) : 
  let large_square_side := 3 * s
  let small_square_side := s
  let rectangle_width := s
  let rectangle_length := large_square_side
  (rectangle_length / rectangle_width : ℝ) = 3 := by
sorry

end rectangle_ratio_l3575_357502


namespace bill_receives_26_l3575_357547

/-- Given a sum of money M to be divided among Allan, Bill, and Carol, prove that Bill receives $26 --/
theorem bill_receives_26 (M : ℚ) : 
  (∃ (allan_share bill_share carol_share : ℚ),
    -- Allan's share
    allan_share = 1 + (1/3) * (M - 1) ∧
    -- Bill's share
    bill_share = 6 + (1/3) * (M - allan_share - 6) ∧
    -- Carol's share
    carol_share = M - allan_share - bill_share ∧
    -- Carol receives $40
    carol_share = 40 ∧
    -- Bill's share is $26
    bill_share = 26) :=
by sorry

end bill_receives_26_l3575_357547


namespace smallest_n_for_candy_purchase_l3575_357505

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (m : ℕ), m > 0 ∧ 
    9 ∣ m ∧ 10 ∣ m ∧ 20 ∣ m ∧ m = 30 * k) → n ≤ k) ∧
  (∃ (m : ℕ), m > 0 ∧ 9 ∣ m ∧ 10 ∣ m ∧ 20 ∣ m ∧ m = 30 * n) :=
by sorry

end smallest_n_for_candy_purchase_l3575_357505


namespace inverse_g_at_167_l3575_357525

def g (x : ℝ) : ℝ := 5 * x^5 + 7

theorem inverse_g_at_167 : g⁻¹ 167 = 2 := by sorry

end inverse_g_at_167_l3575_357525


namespace range_of_m_l3575_357513

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x m : ℝ) : Prop := x^2 + x + m - m^2 > 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (m > 1) →
  (∀ x : ℝ, (¬(p x) → ¬(q x m)) ∧ ∃ y : ℝ, ¬(p y) ∧ (q y m)) →
  m ≥ 3 :=
sorry

end range_of_m_l3575_357513


namespace power_of_seven_l3575_357517

theorem power_of_seven (k : ℕ) : (7 : ℝ) ^ (4 * k + 2) = 784 → (7 : ℝ) ^ k = 2 := by
  sorry

end power_of_seven_l3575_357517


namespace range_of_fraction_l3575_357566

theorem range_of_fraction (x y : ℝ) 
  (h1 : x - 2*y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3/2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by sorry

end range_of_fraction_l3575_357566


namespace composition_equation_solution_l3575_357559

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 5
  let φ : ℝ → ℝ := λ x ↦ 5 * x + 4
  ∀ x : ℝ, δ (φ x) = 9 → x = -3/5 :=
by sorry

end composition_equation_solution_l3575_357559


namespace quadratic_root_implies_d_l3575_357533

theorem quadratic_root_implies_d (d : ℚ) : 
  (∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = -7 + Real.sqrt 15 ∨ x = -7 - Real.sqrt 15) →
  d = 181 / 8 := by
  sorry

end quadratic_root_implies_d_l3575_357533


namespace mary_screws_problem_l3575_357509

theorem mary_screws_problem (initial_screws : Nat) (buy_multiplier : Nat) (num_sections : Nat) : 
  initial_screws = 8 → buy_multiplier = 2 → num_sections = 4 → 
  (initial_screws + initial_screws * buy_multiplier) / num_sections = 6 := by
sorry

end mary_screws_problem_l3575_357509


namespace max_sum_with_length_constraint_l3575_357564

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) --/
def length (n : ℕ) : ℕ := sorry

theorem max_sum_with_length_constraint :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ length x + length y = 16 ∧
  ∀ (a b : ℕ), a > 1 → b > 1 → length a + length b = 16 → a + 3 * b ≤ x + 3 * y ∧
  x + 3 * y = 98305 := by sorry

end max_sum_with_length_constraint_l3575_357564


namespace lg_expression_equals_two_l3575_357555

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_two :
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 := by
  sorry

end lg_expression_equals_two_l3575_357555


namespace laptop_price_l3575_357549

def original_price : ℝ → Prop :=
  fun x => (0.80 * x - 50 = 0.70 * x - 30) ∧ (x > 0)

theorem laptop_price : ∃ x, original_price x ∧ x = 200 := by
  sorry

end laptop_price_l3575_357549


namespace student_scores_l3575_357511

theorem student_scores (M P C : ℝ) 
  (h1 : C = P + 20) 
  (h2 : (M + C) / 2 = 30) : 
  M + P = 40 := by
sorry

end student_scores_l3575_357511


namespace golden_rectangle_perimeter_l3575_357582

/-- A golden rectangle is a rectangle where the ratio of its width to its length is (√5 - 1) / 2 -/
def is_golden_rectangle (width length : ℝ) : Prop :=
  width / length = (Real.sqrt 5 - 1) / 2

/-- The perimeter of a rectangle given its width and length -/
def rectangle_perimeter (width length : ℝ) : ℝ :=
  2 * (width + length)

theorem golden_rectangle_perimeter :
  ∀ width length : ℝ,
  is_golden_rectangle width length →
  (width = Real.sqrt 5 - 1 ∨ length = Real.sqrt 5 - 1) →
  rectangle_perimeter width length = 4 ∨ rectangle_perimeter width length = 2 * Real.sqrt 5 + 2 :=
by sorry

end golden_rectangle_perimeter_l3575_357582


namespace complement_intersection_equals_set_l3575_357546

theorem complement_intersection_equals_set (U M N : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  M = {1, 3, 4} →
  N = {2, 4, 5} →
  (U \ M) ∩ N = {2, 5} := by sorry

end complement_intersection_equals_set_l3575_357546


namespace midpoint_of_fractions_l3575_357521

theorem midpoint_of_fractions :
  let a := 1 / 7
  let b := 1 / 9
  let midpoint := (a + b) / 2
  midpoint = 8 / 63 := by sorry

end midpoint_of_fractions_l3575_357521


namespace truncated_pyramid_sphere_area_relation_l3575_357528

/-- Given a regular n-gonal truncated pyramid circumscribed around a sphere:
    S1 is the area of the base surface,
    S2 is the area of the lateral surface,
    S is the total surface area,
    σ is the area of the polygon whose vertices are the tangential points of the sphere and the lateral faces of the pyramid.
    This theorem states that σS = 4S1S2 cos²(π/n). -/
theorem truncated_pyramid_sphere_area_relation (n : ℕ) (S1 S2 S σ : ℝ) :
  n ≥ 3 →
  S1 > 0 →
  S2 > 0 →
  S = S1 + S2 →
  σ > 0 →
  σ * S = 4 * S1 * S2 * (Real.cos (π / n : ℝ))^2 := by
  sorry

end truncated_pyramid_sphere_area_relation_l3575_357528


namespace base_b_divisibility_l3575_357552

theorem base_b_divisibility (b : ℤ) : 
  let diff := 2 * b^3 + 2 * b - (2 * b^2 + 2 * b + 1)
  (b = 8 ∧ ¬(diff % 3 = 0)) ∨
  ((b = 3 ∨ b = 4 ∨ b = 6 ∨ b = 7) ∧ (diff % 3 = 0)) := by
  sorry

end base_b_divisibility_l3575_357552


namespace lawrence_average_work_hours_l3575_357538

def lawrence_work_hours (full_days : ℕ) (partial_days : ℕ) (full_day_hours : ℝ) (partial_day_hours : ℝ) : ℝ :=
  (full_days : ℝ) * full_day_hours + (partial_days : ℝ) * partial_day_hours

theorem lawrence_average_work_hours :
  let total_days : ℕ := 5
  let full_days : ℕ := 3
  let partial_days : ℕ := 2
  let full_day_hours : ℝ := 8
  let partial_day_hours : ℝ := 5.5
  let total_hours := lawrence_work_hours full_days partial_days full_day_hours partial_day_hours
  total_hours / total_days = 7 := by
sorry

end lawrence_average_work_hours_l3575_357538


namespace unique_square_divisible_by_three_in_range_l3575_357500

theorem unique_square_divisible_by_three_in_range : ∃! x : ℕ,
  (∃ n : ℕ, x = n^2) ∧
  x % 3 = 0 ∧
  60 < x ∧ x < 130 :=
by
  -- The proof goes here
  sorry

end unique_square_divisible_by_three_in_range_l3575_357500


namespace small_months_not_remainder_l3575_357553

/-- The number of months in a year -/
def total_months : ℕ := 12

/-- The number of big months in a year -/
def big_months : ℕ := 7

/-- The number of small months in a year -/
def small_months : ℕ := 4

/-- February is a special case and not counted as either big or small -/
def february_special : Prop := True

theorem small_months_not_remainder :
  small_months ≠ total_months - big_months :=
by sorry

end small_months_not_remainder_l3575_357553


namespace two_in_S_l3575_357508

def S : Set ℕ := {0, 1, 2}

theorem two_in_S : 2 ∈ S := by sorry

end two_in_S_l3575_357508


namespace rectangle_dimension_solution_l3575_357540

theorem rectangle_dimension_solution :
  ∃! x : ℚ, (3 * x - 4 > 0) ∧ 
             (2 * x + 7 > 0) ∧ 
             ((3 * x - 4) * (2 * x + 7) = 18 * x - 10) ∧
             x = 3 / 2 := by
  sorry

end rectangle_dimension_solution_l3575_357540


namespace corner_divisions_l3575_357522

/-- A corner made up of 3 squares -/
structure Corner :=
  (squares : Fin 3 → Square)

/-- Represents a division of the corner into equal parts -/
structure Division :=
  (parts : ℕ)
  (is_equal : Bool)

/-- Checks if a division of the corner into n parts is possible and equal -/
def is_valid_division (c : Corner) (n : ℕ) : Prop :=
  ∃ (d : Division), d.parts = n ∧ d.is_equal = true

/-- Theorem stating that the corner can be divided into 2, 3, and 4 equal parts -/
theorem corner_divisions (c : Corner) :
  (is_valid_division c 2) ∧ 
  (is_valid_division c 3) ∧ 
  (is_valid_division c 4) :=
sorry

end corner_divisions_l3575_357522


namespace min_value_at_six_l3575_357593

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

/-- Theorem stating that f(x) has a minimum value when x = 6 -/
theorem min_value_at_six :
  ∀ x : ℝ, f x ≥ f 6 :=
by sorry

end min_value_at_six_l3575_357593


namespace arccos_one_half_equals_pi_third_l3575_357580

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l3575_357580


namespace product_mod_seven_l3575_357504

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end product_mod_seven_l3575_357504


namespace alpha_values_l3575_357578

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^3 - 1) = 5 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 8 ∨ α = -Complex.I * Real.sqrt 8 :=
sorry

end alpha_values_l3575_357578


namespace equation_solutions_l3575_357591

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 1)^2 - 9 = 0 ↔ x = 5/2 ∨ x = -1/2) ∧
  (∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = 7 ∨ x = -1) := by
  sorry

end equation_solutions_l3575_357591


namespace apartment_fraction_sum_l3575_357598

theorem apartment_fraction_sum : 
  let one_bedroom : ℝ := 0.12
  let two_bedroom : ℝ := 0.26
  let three_bedroom : ℝ := 0.38
  let four_bedroom : ℝ := 0.24
  one_bedroom + two_bedroom + three_bedroom = 0.76 :=
by sorry

end apartment_fraction_sum_l3575_357598


namespace shell_difference_l3575_357572

theorem shell_difference (perfect_shells broken_shells non_spiral_perfect : ℕ) 
  (h1 : perfect_shells = 17)
  (h2 : broken_shells = 52)
  (h3 : non_spiral_perfect = 12) :
  broken_shells / 2 - (perfect_shells - non_spiral_perfect) = 21 := by
  sorry

end shell_difference_l3575_357572


namespace second_month_sale_is_11860_l3575_357520

/-- Represents the sales data for a grocer over 6 months -/
structure GrocerSales where
  first_month : ℕ
  third_month : ℕ
  fourth_month : ℕ
  fifth_month : ℕ
  sixth_month : ℕ
  average_sale : ℕ

/-- Calculates the sale in the second month given the sales data -/
def second_month_sale (sales : GrocerSales) : ℕ :=
  6 * sales.average_sale - (sales.first_month + sales.third_month + sales.fourth_month + sales.fifth_month + sales.sixth_month)

/-- Theorem stating that the second month sale is 11860 given the specific sales data -/
theorem second_month_sale_is_11860 :
  let sales : GrocerSales := {
    first_month := 5420,
    third_month := 6350,
    fourth_month := 6500,
    fifth_month := 6200,
    sixth_month := 8270,
    average_sale := 6400
  }
  second_month_sale sales = 11860 := by
  sorry

end second_month_sale_is_11860_l3575_357520


namespace multiplier_problem_l3575_357544

theorem multiplier_problem (n : ℝ) (m : ℝ) (h1 : n = 3) (h2 : m * n = 3 * n + 12) : m = 7 := by
  sorry

end multiplier_problem_l3575_357544


namespace miami_ny_temp_difference_l3575_357560

/-- Represents the temperatures of three cities and their relationships -/
structure CityTemperatures where
  new_york : ℝ
  miami : ℝ
  san_diego : ℝ
  ny_temp_is_80 : new_york = 80
  miami_cooler_than_sd : miami = san_diego - 25
  average_temp : (new_york + miami + san_diego) / 3 = 95

/-- The temperature difference between Miami and New York -/
def temp_difference (ct : CityTemperatures) : ℝ :=
  ct.miami - ct.new_york

/-- Theorem stating that the temperature difference between Miami and New York is 10 degrees -/
theorem miami_ny_temp_difference (ct : CityTemperatures) : temp_difference ct = 10 := by
  sorry

end miami_ny_temp_difference_l3575_357560


namespace union_of_A_and_B_l3575_357597

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x < 2} := by sorry

end union_of_A_and_B_l3575_357597


namespace base_nine_digits_of_2048_l3575_357531

theorem base_nine_digits_of_2048 : ∃ n : ℕ, n > 0 ∧ 9^(n-1) ≤ 2048 ∧ 2048 < 9^n :=
by sorry

end base_nine_digits_of_2048_l3575_357531


namespace binomial_15_12_times_3_l3575_357589

theorem binomial_15_12_times_3 : 3 * (Nat.choose 15 12) = 2730 := by
  sorry

end binomial_15_12_times_3_l3575_357589


namespace max_value_sqrt_sum_l3575_357543

theorem max_value_sqrt_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧ Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z) ≤ m ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 2 ∧
  Real.sqrt x₀ + Real.sqrt (2 * y₀) + Real.sqrt (3 * z₀) = m :=
by
  sorry

end max_value_sqrt_sum_l3575_357543


namespace volunteer_arrangements_l3575_357594

/-- The number of ways to arrange n people among k exits, with each exit having at least one person. -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of permutations of r items chosen from n items. -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

theorem volunteer_arrangements :
  arrangements 5 4 = choose 5 2 * permutations 3 3 ∧ 
  arrangements 5 4 = 240 := by sorry

end volunteer_arrangements_l3575_357594


namespace social_media_to_phone_ratio_l3575_357569

/-- Represents the daily phone usage in hours -/
def daily_phone_usage : ℝ := 8

/-- Represents the weekly social media usage in hours -/
def weekly_social_media : ℝ := 28

/-- Represents the number of days in a week -/
def days_in_week : ℝ := 7

/-- Theorem stating that the ratio of daily social media usage to total daily phone usage is 1:2 -/
theorem social_media_to_phone_ratio :
  (weekly_social_media / days_in_week) / daily_phone_usage = 1 / 2 := by
  sorry

end social_media_to_phone_ratio_l3575_357569


namespace candy_box_count_candy_box_theorem_l3575_357537

theorem candy_box_count : ℝ → Prop :=
  fun x =>
    let day1_eaten := 0.2 * x + 16
    let day1_remaining := x - day1_eaten
    let day2_eaten := 0.3 * day1_remaining + 20
    let day2_remaining := day1_remaining - day2_eaten
    let day3_eaten := 0.75 * day2_remaining + 30
    day3_eaten = day2_remaining ∧ x = 270

theorem candy_box_theorem : ∃ x : ℝ, candy_box_count x :=
  sorry

end candy_box_count_candy_box_theorem_l3575_357537


namespace quadratic_function_properties_l3575_357588

-- Define the function f(x) = -x^2 + bx + c
def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Theorem statement
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c 0 = -3) →
  (f b c (-6) = -3) →
  (b = -6 ∧ c = -3) ∧
  (∀ x : ℝ, -4 ≤ x ∧ x ≤ 0 → f b c x ≤ 6) ∧
  (∃ x : ℝ, -4 ≤ x ∧ x ≤ 0 ∧ f b c x = 6) :=
by sorry

end quadratic_function_properties_l3575_357588


namespace remaining_area_is_27_l3575_357585

/-- Represents the square grid --/
def Grid := Fin 6 → Fin 6 → Bool

/-- The area of a single cell in square centimeters --/
def cellArea : ℝ := 1

/-- The total area of the square in square centimeters --/
def totalArea : ℝ := 36

/-- The area of the dark grey triangles in square centimeters --/
def darkGreyArea : ℝ := 3

/-- The area of the light grey triangles in square centimeters --/
def lightGreyArea : ℝ := 6

/-- The total area of removed triangles in square centimeters --/
def removedArea : ℝ := darkGreyArea + lightGreyArea

/-- Theorem: The area of the remaining shape after cutting out triangles is 27 square cm --/
theorem remaining_area_is_27 : totalArea - removedArea = 27 := by
  sorry

end remaining_area_is_27_l3575_357585


namespace number_placement_theorem_l3575_357503

-- Define the function f that maps integer coordinates to natural numbers
def f : ℤ × ℤ → ℕ := fun (x, y) => Nat.gcd (Int.natAbs x) (Int.natAbs y)

-- Define the property that every natural number appears at some point
def surjective (f : ℤ × ℤ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), f (x, y) = n

-- Define the property of periodicity along a line
def periodic_along_line (f : ℤ × ℤ → ℕ) (a b c : ℤ) : Prop :=
  c ≠ 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℤ), (a * x₁ + b * y₁ = c) ∧ (a * x₂ + b * y₂ = c) ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  ∃ (dx dy : ℤ), ∀ (x y : ℤ), 
    (a * x + b * y = c) → f (x, y) = f (x + dx, y + dy)

theorem number_placement_theorem :
  surjective f ∧ 
  (∀ (a b c : ℤ), periodic_along_line f a b c) :=
sorry

end number_placement_theorem_l3575_357503


namespace expand_polynomial_product_l3575_357541

theorem expand_polynomial_product : 
  ∀ x : ℝ, (3*x^2 - 2*x + 4) * (4*x^2 + 3*x - 6) = 12*x^4 + x^3 - 8*x^2 + 24*x - 24 := by
  sorry

end expand_polynomial_product_l3575_357541


namespace seating_theorem_l3575_357561

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange a pair of people -/
def pairArrangements : ℕ := 2

/-- The number of ways to seat 10 people around a round table with two specific people next to each other -/
def seatingArrangements : ℕ := roundTableArrangements 9 * pairArrangements

theorem seating_theorem : seatingArrangements = 80640 := by sorry

end seating_theorem_l3575_357561


namespace only_prime_three_squared_plus_eight_prime_l3575_357556

theorem only_prime_three_squared_plus_eight_prime :
  ∀ p : ℕ, Prime p ∧ Prime (p^2 + 8) → p = 3 :=
by sorry

end only_prime_three_squared_plus_eight_prime_l3575_357556


namespace least_positive_angle_theta_l3575_357539

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0 ∧ 
   Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin θ ∧
   ∀ φ, φ > 0 ∧ Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin φ → θ ≤ φ) →
  θ = 70 * Real.pi / 180 := by
sorry

end least_positive_angle_theta_l3575_357539


namespace sum_of_cubes_l3575_357545

theorem sum_of_cubes (a b c d e : ℝ) 
  (sum_zero : a + b + c + d + e = 0)
  (sum_products : a*b*c + a*b*d + a*b*e + a*c*d + a*c*e + a*d*e + b*c*d + b*c*e + b*d*e + c*d*e = 2008) :
  a^3 + b^3 + c^3 + d^3 + e^3 = -12048 := by
sorry

end sum_of_cubes_l3575_357545


namespace rehab_centers_multiple_l3575_357529

/-- The number of rehabilitation centers visited by each person and the total visited --/
structure RehabCenters where
  lisa : ℕ
  jude : ℕ
  han : ℕ
  jane : ℕ
  total : ℕ

/-- The conditions of the problem --/
def problem_conditions (rc : RehabCenters) : Prop :=
  rc.lisa = 6 ∧
  rc.jude = rc.lisa / 2 ∧
  rc.han = 2 * rc.jude - 2 ∧
  rc.total = 27 ∧
  rc.jane = rc.total - (rc.lisa + rc.jude + rc.han)

/-- The theorem to be proved --/
theorem rehab_centers_multiple (rc : RehabCenters) 
  (h : problem_conditions rc) : ∃ x : ℕ, x = 2 ∧ rc.jane = x * rc.han + 6 := by
  sorry

end rehab_centers_multiple_l3575_357529


namespace modulus_of_complex_number_l3575_357558

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  ∃ (z : ℂ), z = i^2017 / (1 + i) ∧ Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_complex_number_l3575_357558


namespace rectangle_perimeter_equals_22_l3575_357557

-- Define the triangle
def triangle_side_a : ℝ := 5
def triangle_side_b : ℝ := 12
def triangle_side_c : ℝ := 13

-- Define the rectangle
def rectangle_width : ℝ := 5

-- Theorem statement
theorem rectangle_perimeter_equals_22 :
  let triangle_area := (1/2) * triangle_side_a * triangle_side_b
  let rectangle_length := triangle_area / rectangle_width
  2 * (rectangle_width + rectangle_length) = 22 :=
by
  sorry

end rectangle_perimeter_equals_22_l3575_357557


namespace parallel_line_equation_l3575_357575

/-- A line passing through (1,1) and parallel to x+2y+2016=0 has equation x+2y-3=0 -/
theorem parallel_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∃ c : ℝ, l = {(x, y) | x + 2*y + c = 0}) →
  ((1, 1) ∈ l) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x + 2*y + 2016 = 0 → False)) →
  l = {(x, y) | x + 2*y - 3 = 0} :=
by sorry

end parallel_line_equation_l3575_357575


namespace fraction_simplification_l3575_357570

theorem fraction_simplification : (4 : ℚ) / (2 - 4 / 5) = 10 / 3 := by
  sorry

end fraction_simplification_l3575_357570


namespace speed_conversion_proof_l3575_357596

/-- Converts a speed from meters per second to kilometers per hour. -/
def convert_mps_to_kmh (speed_mps : ℚ) : ℚ :=
  speed_mps * 3.6

/-- Proves that converting 17/36 m/s to km/h results in 1.7 km/h. -/
theorem speed_conversion_proof :
  convert_mps_to_kmh (17/36) = 1.7 := by
  sorry

end speed_conversion_proof_l3575_357596


namespace cubic_root_sum_square_l3575_357577

theorem cubic_root_sum_square (p q r t : ℝ) : 
  (p^3 - 6*p^2 + 8*p - 1 = 0) →
  (q^3 - 6*q^2 + 8*q - 1 = 0) →
  (r^3 - 6*r^2 + 8*r - 1 = 0) →
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) →
  (t^4 - 12*t^2 - 8*t = -4) := by
  sorry

end cubic_root_sum_square_l3575_357577


namespace polynomial_division_theorem_l3575_357512

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^5 - 24*x^4 + 5*x^3 + 15*x^2 - 18*x + 12 = 
  (x - 3) * (x^5 + 5*x^4 - 9*x^3 - 22*x^2 - 51*x - 171) - 501 := by
  sorry

end polynomial_division_theorem_l3575_357512


namespace problem_statement_l3575_357510

theorem problem_statement (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 20 * x + 40) / (x - 3)) →
  P + Q = 50 := by
  sorry

end problem_statement_l3575_357510


namespace salary_percentage_is_120_percent_l3575_357573

/-- The percentage of one employee's salary compared to another -/
def salary_percentage (total_salary n_salary : ℚ) : ℚ :=
  ((total_salary - n_salary) / n_salary) * 100

/-- Proof that the salary percentage is 120% given the conditions -/
theorem salary_percentage_is_120_percent 
  (total_salary : ℚ) 
  (n_salary : ℚ) 
  (h1 : total_salary = 594) 
  (h2 : n_salary = 270) : 
  salary_percentage total_salary n_salary = 120 := by
  sorry

end salary_percentage_is_120_percent_l3575_357573


namespace tangent_lines_perpendicular_range_l3575_357542

/-- Given two curves and their tangent lines, prove the range of parameter a -/
theorem tangent_lines_perpendicular_range (a : ℝ) : 
  ∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 3/2 ∧
  let f₁ (x : ℝ) := (a * x - 1) * Real.exp x
  let f₂ (x : ℝ) := (1 - x) * Real.exp (-x)
  let k₁ := (a * x₀ + a - 1) * Real.exp x₀
  let k₂ := (x₀ - 2) * Real.exp (-x₀)
  k₁ * k₂ = -1 →
  1 ≤ a ∧ a ≤ 3/2 :=
sorry

end tangent_lines_perpendicular_range_l3575_357542


namespace marble_difference_l3575_357523

theorem marble_difference (red_marbles : ℕ) (red_bags : ℕ) (blue_marbles : ℕ) (blue_bags : ℕ)
  (h1 : red_marbles = 288)
  (h2 : red_bags = 12)
  (h3 : blue_marbles = 243)
  (h4 : blue_bags = 9)
  (h5 : red_bags ≠ 0)
  (h6 : blue_bags ≠ 0) :
  blue_marbles / blue_bags - red_marbles / red_bags = 3 :=
by
  sorry

end marble_difference_l3575_357523


namespace katies_journey_distance_l3575_357532

/-- The total distance of Katie's journey to the island -/
def total_distance (leg1 leg2 leg3 : ℕ) : ℕ :=
  leg1 + leg2 + leg3

/-- Theorem stating that the total distance of Katie's journey is 436 miles -/
theorem katies_journey_distance :
  total_distance 132 236 68 = 436 := by
  sorry

end katies_journey_distance_l3575_357532


namespace abc_inequality_l3575_357562

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 11/6 * c < a + b ∧ a + b < 2 * c)
  (h2 : 3/2 * a < b + c ∧ b + c < 5/3 * a)
  (h3 : 5/2 * b < a + c ∧ a + c < 11/4 * b) :
  b < c ∧ c < a := by
  sorry

end abc_inequality_l3575_357562


namespace largest_number_l3575_357501

/-- Represents a number with a finite or repeating decimal expansion -/
structure DecimalNumber where
  integerPart : ℕ
  finitePart : List ℕ
  repeatingPart : List ℕ

/-- The set of numbers to compare -/
def numberSet : Set DecimalNumber := {
  ⟨8, [1, 2, 3, 5], []⟩,
  ⟨8, [1, 2, 3], [5]⟩,
  ⟨8, [1, 2, 3], [4, 5]⟩,
  ⟨8, [1, 2], [3, 4, 5]⟩,
  ⟨8, [1], [2, 3, 4, 5]⟩
}

/-- Converts a DecimalNumber to a real number -/
def toReal (d : DecimalNumber) : ℝ :=
  sorry

/-- Compares two DecimalNumbers -/
def greaterThan (a b : DecimalNumber) : Prop :=
  toReal a > toReal b

/-- Theorem stating that 8.123̅5 is the largest number in the set -/
theorem largest_number (n : DecimalNumber) :
  n ∈ numberSet →
  greaterThan ⟨8, [1, 2, 3], [5]⟩ n ∨ n = ⟨8, [1, 2, 3], [5]⟩ :=
  sorry

end largest_number_l3575_357501


namespace cube_volume_increase_l3575_357551

theorem cube_volume_increase (a : ℝ) (ha : a > 0) :
  (2 * a)^3 - a^3 = 7 * a^3 := by sorry

end cube_volume_increase_l3575_357551


namespace geometric_sequence_sum_l3575_357595

/-- Given a geometric sequence {a_n} where a_2 = 2 and a_5 = 1/4,
    prove that the sum a_1*a_2 + a_2*a_3 + ... + a_5*a_6 equals 341/32. -/
theorem geometric_sequence_sum (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2) →  -- geometric sequence property
  a 2 = 2 →
  a 5 = 1/4 →
  (a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 : ℚ) = 341/32 :=
by sorry

end geometric_sequence_sum_l3575_357595


namespace peter_total_spending_l3575_357592

/-- The cost of one shirt in dollars -/
def shirt_cost : ℚ := 10

/-- The cost of one pair of pants in dollars -/
def pants_cost : ℚ := 6

/-- The number of shirts Peter bought -/
def peter_shirts : ℕ := 5

/-- The number of pairs of pants Peter bought -/
def peter_pants : ℕ := 2

/-- The number of shirts Jessica bought -/
def jessica_shirts : ℕ := 2

/-- The total cost of Jessica's purchase in dollars -/
def jessica_total : ℚ := 20

theorem peter_total_spending :
  peter_shirts * shirt_cost + peter_pants * pants_cost = 62 :=
sorry

end peter_total_spending_l3575_357592
