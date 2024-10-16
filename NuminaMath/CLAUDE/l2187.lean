import Mathlib

namespace NUMINAMATH_CALUDE_power_sum_equals_40_l2187_218729

theorem power_sum_equals_40 : (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 + 2^4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_40_l2187_218729


namespace NUMINAMATH_CALUDE_complex_number_location_l2187_218705

theorem complex_number_location : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (Complex.I : ℂ) / (3 + Complex.I) = ⟨x, y⟩ := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2187_218705


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2187_218783

theorem algebraic_expression_equality (x y : ℝ) (h : x + 2*y = 2) : 1 - 2*x - 4*y = -3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2187_218783


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2187_218721

theorem quadratic_roots_condition (a : ℝ) : 
  (a ∈ Set.Ici 2 → ∃ x : ℝ, x^2 - a*x + 1 = 0) ∧ 
  (∃ a : ℝ, a ∉ Set.Ici 2 ∧ ∃ x : ℝ, x^2 - a*x + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2187_218721


namespace NUMINAMATH_CALUDE_girls_in_class_l2187_218789

theorem girls_in_class (total_students : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ)
  (h1 : total_students = 35)
  (h2 : girls_ratio = 3)
  (h3 : boys_ratio = 4) :
  (girls_ratio * total_students) / (girls_ratio + boys_ratio) = 15 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l2187_218789


namespace NUMINAMATH_CALUDE_cups_in_smaller_purchase_is_40_l2187_218796

/-- The cost of a single paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of a single paper cup -/
def cup_cost : ℝ := sorry

/-- The number of cups in the smaller purchase -/
def cups_in_smaller_purchase : ℕ := sorry

/-- The total cost of 100 plates and 200 cups is $6.00 -/
axiom total_cost_large : 100 * plate_cost + 200 * cup_cost = 6

/-- The total cost of 20 plates and the unknown number of cups is $1.20 -/
axiom total_cost_small : 20 * plate_cost + cups_in_smaller_purchase * cup_cost = 1.2

theorem cups_in_smaller_purchase_is_40 : cups_in_smaller_purchase = 40 := by sorry

end NUMINAMATH_CALUDE_cups_in_smaller_purchase_is_40_l2187_218796


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l2187_218733

/-- Represents a sculpture made of unit cubes -/
structure Sculpture where
  totalCubes : Nat
  layer1 : Nat
  layer2 : Nat
  layer3 : Nat
  layer4 : Nat

/-- Calculate the exposed surface area of the sculpture -/
def exposedSurfaceArea (s : Sculpture) : Nat :=
  5 * s.layer1 + 4 * s.layer2 + s.layer3 + 3 * s.layer4

/-- The main theorem stating the exposed surface area of the specific sculpture -/
theorem sculpture_surface_area :
  ∃ (s : Sculpture),
    s.totalCubes = 20 ∧
    s.layer1 = 1 ∧
    s.layer2 = 4 ∧
    s.layer3 = 9 ∧
    s.layer4 = 6 ∧
    exposedSurfaceArea s = 48 := by
  sorry


end NUMINAMATH_CALUDE_sculpture_surface_area_l2187_218733


namespace NUMINAMATH_CALUDE_sector_area_theorem_l2187_218750

-- Define the circular sector
def circular_sector (central_angle : Real) (arc_length : Real) : Real × Real :=
  (central_angle, arc_length)

-- Theorem statement
theorem sector_area_theorem (s : Real × Real) :
  s = circular_sector (120 * π / 180) (6 * π) →
  (1/3) * π * ((6 * π) / ((1/3) * 2 * π))^2 = 27 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_theorem_l2187_218750


namespace NUMINAMATH_CALUDE_exponential_function_property_l2187_218726

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, a^x ≤ a^2) ∧ 
  (∀ x ∈ Set.Icc 1 2, a^x ≥ a^1) ∧
  (a^2 - a^1 = a / 2) →
  a = 1/2 ∨ a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l2187_218726


namespace NUMINAMATH_CALUDE_symmetric_points_range_l2187_218711

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

-- Define the theorem
theorem symmetric_points_range (a : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ ∃ y : ℝ, f a x = -g y) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l2187_218711


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l2187_218755

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem: It is false that all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (for completeness, not directly used in the proof)
def is_convex (s : Square) : Prop := true
def has_four_right_angles (s : Square) : Prop := true
def has_equal_diagonals (s : Square) : Prop := true
def similar (s1 s2 : Square) : Prop := true

end NUMINAMATH_CALUDE_not_all_squares_congruent_l2187_218755


namespace NUMINAMATH_CALUDE_exterior_angle_sum_is_360_l2187_218765

/-- A convex polygon with n sides and equilateral triangles attached to each side -/
structure ConvexPolygonWithTriangles where
  n : ℕ  -- number of sides of the original polygon
  [n_pos : Fact (n > 0)]

/-- The sum of exterior angles of a convex polygon with attached equilateral triangles -/
def exterior_angle_sum (p : ConvexPolygonWithTriangles) : ℝ :=
  360

/-- Theorem: The sum of all exterior angles in a convex polygon with attached equilateral triangles is 360° -/
theorem exterior_angle_sum_is_360 (p : ConvexPolygonWithTriangles) :
  exterior_angle_sum p = 360 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_sum_is_360_l2187_218765


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_plus_c_range_l2187_218746

/- Define a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/- Define the condition √3a*sin(C) + a*cos(C) = c + b -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b

/- Theorem 1: If the condition holds, then angle A = 60° -/
theorem angle_A_is_60_degrees (t : Triangle) (h : condition t) : t.A = 60 * π / 180 := by
  sorry

/- Theorem 2: If a = √3 and the condition holds, then √3 < b + c ≤ 2√3 -/
theorem b_plus_c_range (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : condition t) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_plus_c_range_l2187_218746


namespace NUMINAMATH_CALUDE_intersection_M_N_l2187_218723

def M : Set ℕ := {1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2187_218723


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2187_218732

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*k*x + k^2 - k - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (k = 5 → x₁*x₂^2 + x₁^2*x₂ = 190) ∧
  (x₁ - 3*x₂ = 2 → k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2187_218732


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2187_218791

/-- The eccentricity of a hyperbola with equation x²/5 - y²/4 = 1 is 3√5/5 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := Real.sqrt 5
  let b : ℝ := 2
  let c : ℝ := 3
  let e : ℝ := c / a
  (∀ x y : ℝ, x^2 / 5 - y^2 / 4 = 1) →
  e = 3 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2187_218791


namespace NUMINAMATH_CALUDE_roller_coaster_rides_l2187_218727

def initial_tickets : ℕ := 287
def spent_tickets : ℕ := 134
def earned_tickets : ℕ := 32
def cost_per_ride : ℕ := 17

theorem roller_coaster_rides : 
  (initial_tickets - spent_tickets + earned_tickets) / cost_per_ride = 10 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_rides_l2187_218727


namespace NUMINAMATH_CALUDE_power_relation_l2187_218764

theorem power_relation (x m n : ℝ) (hm : x^m = 3) (hn : x^n = 5) :
  x^(2*m - 3*n) = 9/125 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2187_218764


namespace NUMINAMATH_CALUDE_ellipse_range_l2187_218758

theorem ellipse_range (m n : ℝ) : 
  (m^2 / 3 + n^2 / 8 = 1) → 
  ∃ x : ℝ, x = Real.sqrt 3 * m ∧ -3 ≤ x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_range_l2187_218758


namespace NUMINAMATH_CALUDE_sprocket_production_rate_l2187_218794

/-- Represents the production rate of a machine in sprockets per hour -/
structure MachineRate where
  rate : ℝ

/-- Represents the time taken by a machine to produce a certain number of sprockets -/
structure MachineTime where
  time : ℝ

def machineA : MachineRate := sorry
def machineP : MachineTime := sorry
def machineQ : MachineTime := sorry

theorem sprocket_production_rate :
  -- Machine P and Q each manufacture 330 sprockets
  (machineP.time * machineA.rate = 330) →
  (machineQ.time * (machineA.rate * 1.1) = 330) →
  -- Machine P takes 10 hours longer than Machine Q
  (machineP.time = machineQ.time + 10) →
  -- Machine Q produces 10% more sprockets per hour than Machine A
  (machineA.rate * 1.1 = machineA.rate + 0.1 * machineA.rate) →
  -- Prove that Machine A produces 3 sprockets per hour
  machineA.rate = 3 := by
    sorry

end NUMINAMATH_CALUDE_sprocket_production_rate_l2187_218794


namespace NUMINAMATH_CALUDE_floors_per_house_l2187_218749

/-- The number of floors in each house given the building conditions -/
theorem floors_per_house 
  (builders_per_floor : ℕ)
  (days_per_floor : ℕ)
  (daily_wage : ℕ)
  (total_builders : ℕ)
  (total_houses : ℕ)
  (total_cost : ℕ)
  (h1 : builders_per_floor = 3)
  (h2 : days_per_floor = 30)
  (h3 : daily_wage = 100)
  (h4 : total_builders = 6)
  (h5 : total_houses = 5)
  (h6 : total_cost = 270000) :
  total_cost / (total_builders * daily_wage * days_per_floor * total_houses) = 3 :=
sorry

end NUMINAMATH_CALUDE_floors_per_house_l2187_218749


namespace NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_64_l2187_218762

theorem fourth_root_16_times_sixth_root_64 : (16 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_64_l2187_218762


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2187_218770

/-- Represents the number of pies with specific ingredients -/
structure PieCount where
  total : ℕ
  chocolate : ℕ
  marshmallow : ℕ
  cayenne : ℕ
  salted_soy_nut : ℕ

/-- Conditions for the pie problem -/
def pie_conditions (p : PieCount) : Prop :=
  p.total = 48 ∧
  p.chocolate = (5 * p.total) / 8 ∧
  p.marshmallow = (3 * p.total) / 4 ∧
  p.cayenne = (2 * p.total) / 3 ∧
  p.salted_soy_nut = p.total / 4 ∧
  p.salted_soy_nut ≤ p.marshmallow

/-- The theorem stating the maximum number of pies without any of the mentioned ingredients -/
theorem max_pies_without_ingredients (p : PieCount) 
  (h : pie_conditions p) : 
  p.total - max p.chocolate (max p.marshmallow (max p.cayenne p.salted_soy_nut)) ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2187_218770


namespace NUMINAMATH_CALUDE_sequence_properties_l2187_218738

def sequence_a (n : ℕ) : ℝ := (n + 1 : ℝ) * 2^(n - 1)

def partial_sum (n : ℕ) : ℝ := n * 2^n - 2^n

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n) :
  (∀ n : ℕ, n > 0 → a n / 2^n - a (n-1) / 2^(n-1) = 1/2) ∧
  (∀ n : ℕ, n > 0 → a n = sequence_a n) ∧
  (∀ n : ℕ, n > 0 → S n = partial_sum n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2187_218738


namespace NUMINAMATH_CALUDE_julias_initial_money_l2187_218772

theorem julias_initial_money (initial_money : ℝ) : 
  initial_money / 2 - (initial_money / 2) / 4 = 15 → initial_money = 40 := by
  sorry

end NUMINAMATH_CALUDE_julias_initial_money_l2187_218772


namespace NUMINAMATH_CALUDE_boys_share_calculation_l2187_218757

/-- Proves that in a family with a given boy-to-girl ratio and total children, 
    if a certain amount is shared among the boys, each boy receives the calculated amount. -/
theorem boys_share_calculation 
  (total_children : ℕ) 
  (boy_ratio girl_ratio : ℕ) 
  (total_money : ℕ) 
  (h1 : total_children = 180) 
  (h2 : boy_ratio = 5) 
  (h3 : girl_ratio = 7) 
  (h4 : total_money = 3900) :
  total_money / (total_children * boy_ratio / (boy_ratio + girl_ratio)) = 52 := by
sorry


end NUMINAMATH_CALUDE_boys_share_calculation_l2187_218757


namespace NUMINAMATH_CALUDE_average_weight_increase_l2187_218768

theorem average_weight_increase (W : ℝ) : 
  let original_average := (W + 45) / 10
  let new_average := (W + 75) / 10
  new_average - original_average = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2187_218768


namespace NUMINAMATH_CALUDE_tanika_cracker_sales_l2187_218707

theorem tanika_cracker_sales (saturday_sales : ℕ) : 
  saturday_sales = 60 → 
  (saturday_sales + (saturday_sales + saturday_sales / 2)) = 150 := by
sorry

end NUMINAMATH_CALUDE_tanika_cracker_sales_l2187_218707


namespace NUMINAMATH_CALUDE_cube_sum_equality_l2187_218799

theorem cube_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^3 + b^3 + c^3 - 3*a*b*c = 0) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l2187_218799


namespace NUMINAMATH_CALUDE_range_of_fraction_l2187_218712

theorem range_of_fraction (a b : ℝ) 
  (ha : -6 < a ∧ a < 8) 
  (hb : 2 < b ∧ b < 3) : 
  -3 < a/b ∧ a/b < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2187_218712


namespace NUMINAMATH_CALUDE_quadratic_roots_relations_l2187_218773

theorem quadratic_roots_relations (a : ℝ) :
  let x₁ : ℝ := (1 + Real.sqrt (5 - 4*a)) / 2
  let x₂ : ℝ := (1 - Real.sqrt (5 - 4*a)) / 2
  (x₁*x₂ + x₁ + x₂ - a = 0) ∧ (x₁*x₂ - a*(x₁ + x₂) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relations_l2187_218773


namespace NUMINAMATH_CALUDE_approx_value_of_625_power_l2187_218782

theorem approx_value_of_625_power (ε : Real) (hε : ε > 0) :
  ∃ (x : Real), abs (x - ((625 : Real)^(0.2 : Real) * (625 : Real)^(0.12 : Real))) < ε ∧
                 abs (x - 17.15) < ε :=
sorry

end NUMINAMATH_CALUDE_approx_value_of_625_power_l2187_218782


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l2187_218792

/-- Given a point C with coordinates (3, y), when reflected over the x-axis to point D,
    the sum of all coordinate values of C and D is equal to 6. -/
theorem reflection_sum_coordinates (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l2187_218792


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2187_218735

-- Define the ellipse
def ellipse (x y b : ℝ) : Prop := x^2 / 4 + y^2 / b^2 = 1

-- Define the constraint on b
def b_constraint (b : ℝ) : Prop := 0 < b ∧ b < 2

-- Define the maximum value of |BF_2| + |AF_2|
def max_focal_sum (b : ℝ) : Prop := ∃ (A B F_2 : ℝ × ℝ), 
  ∀ (P Q : ℝ × ℝ), dist P F_2 + dist Q F_2 ≤ dist A F_2 + dist B F_2 ∧ 
  dist A F_2 + dist B F_2 = 5

-- Theorem statement
theorem ellipse_focal_property (b : ℝ) :
  b_constraint b →
  (∀ x y, ellipse x y b → max_focal_sum b) →
  b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2187_218735


namespace NUMINAMATH_CALUDE_complex_product_modulus_l2187_218761

theorem complex_product_modulus : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l2187_218761


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l2187_218728

theorem quadratic_inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → x^2 + a*x + 9 ≥ 0) → a ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l2187_218728


namespace NUMINAMATH_CALUDE_largest_number_problem_l2187_218759

theorem largest_number_problem (a b c : ℝ) : 
  a < b ∧ b < c →
  a + b + c = 72 →
  c - b = 5 →
  b - a = 8 →
  c = 30 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2187_218759


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l2187_218724

theorem rectangle_ratio_theorem (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≤ b) :
  let d := Real.sqrt (a^2 + b^2)
  let k := a / b
  (a / b = (a + 2*b) / d) → (k^4 - 3*k^2 - 4*k - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l2187_218724


namespace NUMINAMATH_CALUDE_jared_current_age_l2187_218795

/-- Represents a person's age at different points in time -/
structure PersonAge where
  current : ℕ
  twoYearsAgo : ℕ
  fiveYearsLater : ℕ

/-- The problem statement -/
theorem jared_current_age (tom : PersonAge) (jared : PersonAge) : 
  tom.fiveYearsLater = 30 →
  jared.twoYearsAgo = 2 * tom.twoYearsAgo →
  jared.current = 48 := by
  sorry

end NUMINAMATH_CALUDE_jared_current_age_l2187_218795


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l2187_218771

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem point_in_fourth_quadrant_y_negative (p : Point) 
  (h : p.x = 5) (h₂ : fourth_quadrant p) : p.y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l2187_218771


namespace NUMINAMATH_CALUDE_father_eats_four_papayas_l2187_218701

/-- The number of papayas Jake eats in one week -/
def jake_papayas : ℕ := 3

/-- The number of papayas Jake's brother eats in one week -/
def brother_papayas : ℕ := 5

/-- The number of weeks Jake is planning for -/
def weeks : ℕ := 4

/-- The total number of papayas Jake needs to buy for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of papayas Jake's father eats in one week -/
def father_papayas : ℕ := (total_papayas - (jake_papayas + brother_papayas) * weeks) / weeks

theorem father_eats_four_papayas : father_papayas = 4 := by
  sorry

end NUMINAMATH_CALUDE_father_eats_four_papayas_l2187_218701


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l2187_218702

theorem unique_six_digit_number : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  n / 100000 = 1 ∧
  3 * n = (n % 100000) * 10 + 1 ∧
  n = 142857 := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l2187_218702


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l2187_218780

theorem quadratic_inequality_always_true (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l2187_218780


namespace NUMINAMATH_CALUDE_correct_calculation_l2187_218753

theorem correct_calculation (x : ℝ) (h : x / 15 = 6) : 15 * x = 1350 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2187_218753


namespace NUMINAMATH_CALUDE_count_sum_generating_sets_l2187_218703

/-- A set of 9 positive integers that can generate all sums from 1 to 500 -/
def SumGeneratingSet : Type := { A : Finset ℕ // A.card = 9 ∧ ∀ n ≤ 500, ∃ B ⊆ A, (B.sum id) = n }

/-- The count of SumGeneratingSet -/
def CountSumGeneratingSets : ℕ := sorry

theorem count_sum_generating_sets :
  CountSumGeneratingSets = 74 := by sorry

end NUMINAMATH_CALUDE_count_sum_generating_sets_l2187_218703


namespace NUMINAMATH_CALUDE_not_always_prime_l2187_218763

theorem not_always_prime : ∃ n : ℕ, ¬ Nat.Prime (n^2 - n + 11) := by
  sorry

end NUMINAMATH_CALUDE_not_always_prime_l2187_218763


namespace NUMINAMATH_CALUDE_platform_length_l2187_218720

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 300 →
  platform_cross_time = 27 →
  pole_cross_time = 18 →
  (train_length + (train_length / pole_cross_time * platform_cross_time - train_length)) = 450 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2187_218720


namespace NUMINAMATH_CALUDE_certain_number_problem_l2187_218719

theorem certain_number_problem (x y : ℝ) (h1 : x = 180) 
  (h2 : 0.25 * x = 0.10 * y - 5) : y = 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2187_218719


namespace NUMINAMATH_CALUDE_triangle_count_is_36_l2187_218737

/-- The number of triangles with integer side lengths where the longest side is 11 -/
def count_triangles : ℕ :=
  (Finset.range 11).sum (λ x => (Finset.Ico x 11).card)

/-- Theorem stating that the count of such triangles is 36 -/
theorem triangle_count_is_36 : count_triangles = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_36_l2187_218737


namespace NUMINAMATH_CALUDE_right_triangle_cube_sides_l2187_218709

theorem right_triangle_cube_sides : ∃ (x : ℝ), 
  let a := x^3
  let b := x^3 - x
  let c := x^3 + x
  a^2 + b^2 = c^2 ∧ a = 8 ∧ b = 6 ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cube_sides_l2187_218709


namespace NUMINAMATH_CALUDE_stream_speed_proof_l2187_218714

/-- Proves that the speed of a stream is 5 km/hr given the conditions of a boat's travel --/
theorem stream_speed_proof (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 25 →
  downstream_distance = 120 →
  downstream_time = 4 →
  ∃ stream_speed : ℝ, stream_speed = 5 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

#check stream_speed_proof

end NUMINAMATH_CALUDE_stream_speed_proof_l2187_218714


namespace NUMINAMATH_CALUDE_lost_in_mountains_second_group_size_l2187_218704

theorem lost_in_mountains (initial_people : ℕ) (initial_days : ℕ) (days_after_sharing : ℕ) : ℕ :=
  let initial_portions := initial_people * initial_days
  let remaining_portions := initial_portions - initial_people
  let total_people := initial_people + (remaining_portions / (days_after_sharing + 1) - initial_people)
  remaining_portions / (days_after_sharing + 1) - initial_people

theorem second_group_size :
  lost_in_mountains 9 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lost_in_mountains_second_group_size_l2187_218704


namespace NUMINAMATH_CALUDE_concert_revenue_calculation_l2187_218743

def ticket_revenue (student_price : ℕ) (non_student_price : ℕ) (total_tickets : ℕ) (student_tickets : ℕ) : ℕ :=
  let non_student_tickets := total_tickets - student_tickets
  student_price * student_tickets + non_student_price * non_student_tickets

theorem concert_revenue_calculation :
  ticket_revenue 9 11 2000 520 = 20960 := by
  sorry

end NUMINAMATH_CALUDE_concert_revenue_calculation_l2187_218743


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l2187_218715

/-- 
Given an arithmetic sequence {a_n} with non-zero common difference d, 
where a_1 = 2d, if a_k is the geometric mean of a_1 and a_{2k+1}, then k = 3.
-/
theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) - a n = d) →
  a 1 = 2 * d →
  a k ^ 2 = a 1 * a (2 * k + 1) →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l2187_218715


namespace NUMINAMATH_CALUDE_product_value_l2187_218708

theorem product_value : 
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l2187_218708


namespace NUMINAMATH_CALUDE_dogsled_race_speed_l2187_218741

/-- Proves that given a 300-mile course, if one team (A) finishes 3 hours faster than another team (T)
    and has an average speed 5 mph greater, then the slower team's (T) average speed is 20 mph. -/
theorem dogsled_race_speed (course_length : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  course_length = 300 →
  time_difference = 3 →
  speed_difference = 5 →
  ∃ (speed_T : ℝ) (time_T : ℝ) (time_A : ℝ),
    course_length = speed_T * time_T ∧
    course_length = (speed_T + speed_difference) * (time_T - time_difference) ∧
    speed_T = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_l2187_218741


namespace NUMINAMATH_CALUDE_thirteenth_result_l2187_218766

theorem thirteenth_result (results : List ℝ) 
  (h1 : results.length = 25)
  (h2 : results.sum / 25 = 19)
  (h3 : (results.take 12).sum / 12 = 14)
  (h4 : (results.drop 13).sum / 12 = 17) :
  results[12] = 103 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_result_l2187_218766


namespace NUMINAMATH_CALUDE_hunter_always_catches_grasshopper_l2187_218775

/-- A point in the 2D integer plane -/
structure Point where
  x : Int
  y : Int

/-- The grasshopper's trajectory -/
structure Trajectory where
  start : Point
  jump : Point

/-- Spiral search function that returns the nth point in the spiral -/
def spiralSearch (n : Nat) : Point :=
  sorry

/-- Predicate to check if a point is on a trajectory at a given time -/
def onTrajectory (p : Point) (t : Trajectory) (time : Nat) : Prop :=
  p.x = t.start.x + t.jump.x * time ∧ p.y = t.start.y + t.jump.y * time

theorem hunter_always_catches_grasshopper :
  ∀ (t : Trajectory), ∃ (time : Nat), onTrajectory (spiralSearch time) t time :=
sorry

end NUMINAMATH_CALUDE_hunter_always_catches_grasshopper_l2187_218775


namespace NUMINAMATH_CALUDE_fruits_given_to_jane_l2187_218767

/- Define the initial number of each type of fruit -/
def plums : ℕ := 25
def guavas : ℕ := 30
def apples : ℕ := 36
def oranges : ℕ := 20
def bananas : ℕ := 15

/- Define the total number of fruits Jacqueline had initially -/
def initial_fruits : ℕ := plums + guavas + apples + oranges + bananas

/- Define the number of fruits Jacqueline had left -/
def fruits_left : ℕ := 38

/- Theorem: The number of fruits Jacqueline gave Jane is equal to 
   the difference between her initial fruits and the fruits left -/
theorem fruits_given_to_jane : 
  initial_fruits - fruits_left = 88 := by sorry

end NUMINAMATH_CALUDE_fruits_given_to_jane_l2187_218767


namespace NUMINAMATH_CALUDE_no_valid_number_l2187_218700

theorem no_valid_number : ¬∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧  -- 3-digit number
  (∃ (x : ℕ), x < 10 ∧ n = 520 + x) ∧  -- in the form 52x where x is a digit
  (n % 6 = 0) ∧  -- divisible by 6
  (n % 10 = 6)  -- last digit is 6
  := by sorry

end NUMINAMATH_CALUDE_no_valid_number_l2187_218700


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_squares_divisible_by_two_l2187_218779

theorem sum_of_three_consecutive_odd_squares_divisible_by_two (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, 3 * n^2 + 8 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_squares_divisible_by_two_l2187_218779


namespace NUMINAMATH_CALUDE_new_tax_rate_is_32_percent_l2187_218793

/-- Calculates the new tax rate given the original rate, income, and differential savings -/
def calculate_new_tax_rate (original_rate : ℚ) (income : ℚ) (differential_savings : ℚ) : ℚ :=
  (original_rate * income - differential_savings) / income

/-- Theorem stating that the new tax rate is 32% given the problem conditions -/
theorem new_tax_rate_is_32_percent :
  let original_rate : ℚ := 42 / 100
  let income : ℚ := 42400
  let differential_savings : ℚ := 4240
  calculate_new_tax_rate original_rate income differential_savings = 32 / 100 := by
  sorry

#eval calculate_new_tax_rate (42 / 100) 42400 4240

end NUMINAMATH_CALUDE_new_tax_rate_is_32_percent_l2187_218793


namespace NUMINAMATH_CALUDE_fourth_pile_magazines_l2187_218718

def magazine_sequence (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 4
  else if n = 3 then 6
  else if n = 5 then 13
  else 0  -- For other values, we don't have information

def difference_sequence (n : ℕ) : ℕ :=
  magazine_sequence (n + 1) - magazine_sequence n

theorem fourth_pile_magazines :
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 3 → difference_sequence (n + 1) = difference_sequence n + 1) →
  magazine_sequence 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fourth_pile_magazines_l2187_218718


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l2187_218786

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -7 + 5*I
  let z₂ : ℂ := 5 - 3*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l2187_218786


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l2187_218739

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
   x₁^2 + a*x₁ + 1 = b ∧ x₂^2 + a*x₂ + 1 = b) →
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l2187_218739


namespace NUMINAMATH_CALUDE_surface_area_upper_bound_l2187_218717

/-- A convex broken line in 3D space -/
structure ConvexBrokenLine where
  points : List (Real × Real × Real)
  is_convex : Bool
  length : Real

/-- The surface area generated by rotating a convex broken line around an axis -/
def surface_area_of_rotation (line : ConvexBrokenLine) (axis : Real × Real × Real) : Real :=
  sorry

/-- The theorem stating the upper bound of the surface area of rotation -/
theorem surface_area_upper_bound (line : ConvexBrokenLine) (axis : Real × Real × Real) :
  surface_area_of_rotation line axis ≤ Real.pi * line.length ^ 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_upper_bound_l2187_218717


namespace NUMINAMATH_CALUDE_team_selection_count_l2187_218744

def boys := 10
def girls := 10
def team_size := 8
def min_boys := 3

def select_team (b g : ℕ) (t m : ℕ) : ℕ :=
  (Nat.choose b 3 * Nat.choose g 5) +
  (Nat.choose b 4 * Nat.choose g 4) +
  (Nat.choose b 5 * Nat.choose g 3) +
  (Nat.choose b 6 * Nat.choose g 2) +
  (Nat.choose b 7 * Nat.choose g 1) +
  (Nat.choose b 8 * Nat.choose g 0)

theorem team_selection_count :
  select_team boys girls team_size min_boys = 114275 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l2187_218744


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2187_218713

theorem solve_linear_equation (y : ℚ) (h : -3 * y - 9 = 6 * y + 3) : y = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2187_218713


namespace NUMINAMATH_CALUDE_f_neg_l2187_218722

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_pos : ∀ x > 0, f x = -x * (1 + x)

-- Theorem to prove
theorem f_neg : ∀ x < 0, f x = -x * (1 - x) := by sorry

end NUMINAMATH_CALUDE_f_neg_l2187_218722


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2187_218774

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2187_218774


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2187_218748

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 12 is √3 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2187_218748


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2187_218742

/-- Represents a partitioned triangle with four regions -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the fourth triangle -/
  area4 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- Theorem stating that given the areas of the four triangles, 
    the area of the quadrilateral is 18 -/
theorem quadrilateral_area 
  (t : PartitionedTriangle) 
  (h1 : t.area1 = 5) 
  (h2 : t.area2 = 9) 
  (h3 : t.area3 = 24/5) 
  (h4 : t.area4 = 9) : 
  t.areaQuad = 18 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_l2187_218742


namespace NUMINAMATH_CALUDE_deer_families_stayed_l2187_218710

theorem deer_families_stayed (total : ℕ) (moved_out : ℕ) (h1 : total = 79) (h2 : moved_out = 34) :
  total - moved_out = 45 := by
  sorry

end NUMINAMATH_CALUDE_deer_families_stayed_l2187_218710


namespace NUMINAMATH_CALUDE_sqrt_65_bound_l2187_218740

theorem sqrt_65_bound (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_bound_l2187_218740


namespace NUMINAMATH_CALUDE_largest_c_for_inequality_l2187_218731

theorem largest_c_for_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, c = |Real.log (a / b)| ∧
  (∀ x α : ℝ, 0 < |x| → |x| ≤ c → 0 < α → α < 1 →
    a^α * b^(1-α) ≤ a * (Real.sinh (α*x) / Real.sinh x) + b * (Real.sinh ((1-α)*x) / Real.sinh x)) ∧
  (∀ c' : ℝ, c' > c →
    ∃ x α : ℝ, 0 < |x| ∧ |x| ≤ c' ∧ 0 < α ∧ α < 1 ∧
      a^α * b^(1-α) > a * (Real.sinh (α*x) / Real.sinh x) + b * (Real.sinh ((1-α)*x) / Real.sinh x)) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_inequality_l2187_218731


namespace NUMINAMATH_CALUDE_yellow_ball_fraction_l2187_218788

theorem yellow_ball_fraction (total : ℝ) (h : total > 0) :
  let initial_green := (4/7) * total
  let initial_yellow := total - initial_green
  let new_yellow := 3 * initial_yellow
  let new_green := initial_green * (3/2)
  let new_total := new_yellow + new_green
  new_yellow / new_total = 3/5 := by
sorry

end NUMINAMATH_CALUDE_yellow_ball_fraction_l2187_218788


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_f_non_positive_iff_m_eq_one_l2187_218784

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem f_monotonicity_and_range (m : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∨
  (∃ c, 0 < c ∧ 
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < c → f m x₁ < f m x₂) ∧
    (∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂)) :=
sorry

theorem f_non_positive_iff_m_eq_one (m : ℝ) :
  (∀ x, 0 < x → f m x ≤ 0) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_f_non_positive_iff_m_eq_one_l2187_218784


namespace NUMINAMATH_CALUDE_lights_on_200_7_11_l2187_218798

/-- The number of lights that are on after the switching operation -/
def lights_on (total_lights : ℕ) (interval1 interval2 : ℕ) : ℕ :=
  (total_lights / interval1 + total_lights / interval2) -
  2 * (total_lights / (interval1 * interval2))

/-- Theorem stating the number of lights on after the switching operation -/
theorem lights_on_200_7_11 :
  lights_on 200 7 11 = 44 := by
sorry

end NUMINAMATH_CALUDE_lights_on_200_7_11_l2187_218798


namespace NUMINAMATH_CALUDE_pocket_probability_change_l2187_218778

-- Define the initial state of the pocket
def initial_red_balls : ℕ := 4
def initial_white_balls : ℕ := 8

-- Define the number of balls removed/added
def balls_changed : ℕ := 6

-- Define the final probability of drawing a red ball
def final_red_probability : ℚ := 5/6

-- Theorem statement
theorem pocket_probability_change :
  let total_balls : ℕ := initial_red_balls + initial_white_balls
  let new_red_balls : ℕ := initial_red_balls + balls_changed
  let new_total_balls : ℕ := total_balls
  (new_red_balls : ℚ) / new_total_balls = final_red_probability := by
  sorry

end NUMINAMATH_CALUDE_pocket_probability_change_l2187_218778


namespace NUMINAMATH_CALUDE_words_with_consonants_l2187_218769

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The number of consonants in the alphabet -/
def consonant_count : ℕ := 4

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels -/
def vowel_only_words : ℕ := vowel_count ^ word_length

theorem words_with_consonants :
  total_words - vowel_only_words = 7744 :=
sorry

end NUMINAMATH_CALUDE_words_with_consonants_l2187_218769


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2187_218797

/-- Given a sequence a_n where a_1 = 2 and {1 + a_n} forms a geometric sequence
    with common ratio 3, prove that a_4 = 80. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, (1 + a (n + 1)) = 3 * (1 + a n)) →
  a 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2187_218797


namespace NUMINAMATH_CALUDE_max_prob_div_by_10_min_nonzero_prob_div_by_10_l2187_218706

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ
  h : length > 0

/-- The probability of a number in the segment being divisible by 10 -/
def prob_div_by_10 (s : Segment) : ℚ :=
  (s.length.divisors.filter (· % 10 = 0)).card / s.length

/-- The maximum probability of a number in any segment being divisible by 10 is 1 -/
theorem max_prob_div_by_10 : ∃ s : Segment, prob_div_by_10 s = 1 :=
  sorry

/-- The minimum non-zero probability of a number in any segment being divisible by 10 is 1/19 -/
theorem min_nonzero_prob_div_by_10 : 
  ∀ s : Segment, prob_div_by_10 s ≠ 0 → prob_div_by_10 s ≥ 1/19 :=
  sorry

end NUMINAMATH_CALUDE_max_prob_div_by_10_min_nonzero_prob_div_by_10_l2187_218706


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2187_218725

theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4) →  -- two sides are 4cm and 8cm
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 20 :=  -- perimeter is 20cm
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2187_218725


namespace NUMINAMATH_CALUDE_cistern_fill_time_proof_l2187_218716

/-- The time it takes to fill the cistern when both taps are opened simultaneously -/
def simultaneous_fill_time : ℝ := 7.2

/-- The time it takes to empty the cistern using the second tap -/
def empty_time : ℝ := 9

/-- The time it takes to fill the cistern using only the first tap -/
def fill_time : ℝ := 4

/-- Theorem stating that the fill_time is correct given the other conditions -/
theorem cistern_fill_time_proof :
  (1 / fill_time) - (1 / empty_time) = (1 / simultaneous_fill_time) :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_proof_l2187_218716


namespace NUMINAMATH_CALUDE_equipment_cost_proof_l2187_218756

/-- The number of players on the team -/
def num_players : ℕ := 16

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 15.20

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 6.80

/-- The total cost of equipment for all players -/
def total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)

theorem equipment_cost_proof : total_cost = 752 := by
  sorry

end NUMINAMATH_CALUDE_equipment_cost_proof_l2187_218756


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2187_218790

theorem unique_solution_exists : ∃! (x : ℝ), x > 0 ∧ (Int.floor x) * x + x^2 = 93 ∧ ∀ (ε : ℝ), ε > 0 → |x - 7.10| < ε := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2187_218790


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l2187_218736

/-- Given two parallel vectors a and b, where a = (6, 8) and b = (sinα, cosα), prove that tanα = 3/4 -/
theorem parallel_vectors_tan_alpha (α : Real) : 
  let a : Fin 2 → Real := ![6, 8]
  let b : Fin 2 → Real := ![Real.sin α, Real.cos α]
  (∃ (k : Real), k ≠ 0 ∧ (∀ i, a i = k * b i)) → 
  Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l2187_218736


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2187_218730

theorem quadratic_roots_condition (α β γ p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  (α*x₁^2 + β*x₁ + γ = α*x₂^2 + β*x₂ + γ) →
  (p^2 = 4*q ∨ p = -β/α) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2187_218730


namespace NUMINAMATH_CALUDE_g_critical_points_l2187_218760

noncomputable def g (x : ℝ) : ℝ :=
  if -3 < x ∧ x ≤ 0 then -x - 3
  else if 0 < x ∧ x ≤ 2 then x - 3
  else if 2 < x ∧ x ≤ 3 then x^2 - 4*x + 6
  else 0  -- Default value for x outside the defined range

def is_critical_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → f y ≤ f x

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ δ > 0, ∀ y, |y - x| < δ → f x ≤ f y

theorem g_critical_points :
  is_critical_point g 0 ∧ 
  is_critical_point g 2 ∧
  is_local_minimum g 2 :=
sorry

end NUMINAMATH_CALUDE_g_critical_points_l2187_218760


namespace NUMINAMATH_CALUDE_stone_reduction_moves_l2187_218787

theorem stone_reduction_moves (n : ℕ) (h : n = 2005) : 
  ∃ (m : ℕ), m = 11 ∧ 
  (∀ (k : ℕ), k < m → 2^(m - k) ≥ n) ∧
  (2^(m - m) < n) :=
sorry

end NUMINAMATH_CALUDE_stone_reduction_moves_l2187_218787


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2187_218754

/-- Given a line Ax + By + C = 0 where AC < 0 and BC < 0, the line does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (A B C : ℝ) (hAC : A * C < 0) (hBC : B * C < 0) :
  ∃ (x y : ℝ), A * x + B * y + C = 0 ∧ (x ≤ 0 ∧ y ≤ 0 → False) :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2187_218754


namespace NUMINAMATH_CALUDE_bus_truck_meeting_time_l2187_218751

theorem bus_truck_meeting_time 
  (initial_distance : ℝ) 
  (truck_speed : ℝ) 
  (bus_speed : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 8)
  (h2 : truck_speed = 60)
  (h3 : bus_speed = 40)
  (h4 : final_distance = 78) :
  (final_distance - initial_distance) / (truck_speed + bus_speed) = 0.7 := by
sorry

end NUMINAMATH_CALUDE_bus_truck_meeting_time_l2187_218751


namespace NUMINAMATH_CALUDE_coin_value_equality_l2187_218777

theorem coin_value_equality (n : ℕ) : 
  (20 * 25 + 10 * 10 = 10 * 25 + n * 10) → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l2187_218777


namespace NUMINAMATH_CALUDE_intersection_implies_k_value_l2187_218752

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The x-coordinate of the intersection point -/
def x_intersect : ℝ := 2

/-- The y-coordinate of the intersection point -/
def y_intersect : ℝ := 13

/-- Line p with equation y = 5x + 3 -/
def p : Line := { slope := 5, intercept := 3 }

/-- Line q with equation y = kx + 7, where k is to be determined -/
def q (k : ℝ) : Line := { slope := k, intercept := 7 }

/-- Theorem stating that if lines p and q intersect at (2, 13), then k = 3 -/
theorem intersection_implies_k_value :
  y_intersect = p.slope * x_intersect + p.intercept ∧
  y_intersect = (q k).slope * x_intersect + (q k).intercept →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_k_value_l2187_218752


namespace NUMINAMATH_CALUDE_probability_of_rolling_seven_l2187_218781

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := sides * sides

/-- The number of ways to roll a sum of 7 with two dice -/
def waysToRollSeven : ℕ := 6

/-- The probability of rolling a sum of 7 with two fair 6-sided dice -/
theorem probability_of_rolling_seven :
  (waysToRollSeven : ℚ) / totalOutcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_of_rolling_seven_l2187_218781


namespace NUMINAMATH_CALUDE_rationalization_sum_l2187_218745

/-- Represents a cube root expression of the form (a * ∛b) / c --/
structure CubeRootExpression where
  a : ℤ
  b : ℕ
  c : ℕ
  c_pos : c > 0
  b_not_perfect_cube : ∀ (p : ℕ), Prime p → ¬(p^3 ∣ b)

/-- Rationalizes the denominator of 5 / (3 * ∛7) --/
def rationalize_denominator : CubeRootExpression :=
  { a := 5
    b := 49
    c := 21
    c_pos := by sorry
    b_not_perfect_cube := by sorry }

/-- The sum of a, b, and c in the rationalized expression --/
def sum_of_parts (expr : CubeRootExpression) : ℤ :=
  expr.a + expr.b + expr.c

theorem rationalization_sum :
  sum_of_parts rationalize_denominator = 75 := by sorry

end NUMINAMATH_CALUDE_rationalization_sum_l2187_218745


namespace NUMINAMATH_CALUDE_complement_of_union_l2187_218747

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2187_218747


namespace NUMINAMATH_CALUDE_min_value_expression_l2187_218785

theorem min_value_expression (u v : ℝ) : 
  (u - v)^2 + (Real.sqrt (4 - u^2) - 2*v - 5)^2 ≥ 9 - 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2187_218785


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_l2187_218776

theorem seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : remaining_seashells = 27) : 
  initial_seashells - remaining_seashells = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_l2187_218776


namespace NUMINAMATH_CALUDE_grover_boxes_l2187_218734

/-- Represents the number of face masks in each box -/
def masks_per_box : ℕ := 20

/-- Represents the cost of each box in dollars -/
def cost_per_box : ℚ := 15

/-- Represents the selling price of each face mask in dollars -/
def price_per_mask : ℚ := 5/4  -- $1.25

/-- Represents the total profit in dollars -/
def total_profit : ℚ := 15

/-- Calculates the revenue from selling one box of face masks -/
def revenue_per_box : ℚ := masks_per_box * price_per_mask

/-- Calculates the profit from selling one box of face masks -/
def profit_per_box : ℚ := revenue_per_box - cost_per_box

/-- Theorem: Given the conditions, Grover bought 3 boxes of face masks -/
theorem grover_boxes : 
  ∃ (n : ℕ), n * profit_per_box = total_profit ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_grover_boxes_l2187_218734
