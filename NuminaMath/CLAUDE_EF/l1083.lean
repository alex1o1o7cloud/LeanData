import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_volume_equals_566_6_pi_l1083_108393

/-- The volume of a truncated cone -/
noncomputable def volume_truncated_cone (R r h : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

/-- The volume of a cylinder -/
noncomputable def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The total volume of a composite solid consisting of a truncated cone topped with a cylinder -/
noncomputable def volume_composite (R_cone r_cone h_cone h_cylinder : ℝ) : ℝ :=
  volume_truncated_cone R_cone r_cone h_cone + volume_cylinder r_cone h_cylinder

theorem composite_volume_equals_566_6_pi :
  volume_composite 10 5 8 4 = 566.6666666666666 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_volume_equals_566_6_pi_l1083_108393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distances_l1083_108330

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length = 1

/-- A point on the surface of the cube -/
structure CubePoint (c : Cube) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = c.edge_length) ∨ 
               (y = 0 ∨ y = c.edge_length) ∨ 
               (z = 0 ∨ z = c.edge_length)

/-- The shortest distance between two points on the cube surface -/
noncomputable def shortest_distance (c : Cube) (p q : CubePoint c) : ℝ := sorry

/-- Vertex A of the cube -/
def vertex_A (c : Cube) : CubePoint c where
  x := 0
  y := 0
  z := 0
  on_surface := by simp

/-- Vertex B of the cube -/
def vertex_B (c : Cube) : CubePoint c where
  x := c.edge_length
  y := 0
  z := 0
  on_surface := by simp

/-- Vertex D of the cube -/
def vertex_D (c : Cube) : CubePoint c where
  x := c.edge_length
  y := c.edge_length
  z := c.edge_length
  on_surface := by simp

/-- Point M on edge AB -/
noncomputable def point_M (c : Cube) : CubePoint c where
  x := c.edge_length / 2
  y := 0
  z := 0
  on_surface := by simp

/-- Point N on edge BC -/
noncomputable def point_N (c : Cube) : CubePoint c where
  x := c.edge_length
  y := c.edge_length / 2
  z := 0
  on_surface := by simp

theorem shortest_distances (c : Cube) : 
  (shortest_distance c (vertex_A c) (vertex_B c) = 1) ∧
  (shortest_distance c (point_M c) (point_N c) = 2) ∧
  (shortest_distance c (vertex_A c) (vertex_D c) = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distances_l1083_108330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_theorem_l1083_108344

/-- Calculates the height of a tree in inches after growing by a percentage -/
noncomputable def tree_height_in_inches (initial_height_feet : ℝ) (growth_percentage : ℝ) : ℝ :=
  let final_height_feet := initial_height_feet * (1 + growth_percentage / 100)
  final_height_feet * 12

/-- Theorem: A tree initially 10 feet tall that grows 50% taller is now 180 inches tall -/
theorem tree_growth_theorem : tree_height_in_inches 10 50 = 180 := by
  -- Unfold the definition of tree_height_in_inches
  unfold tree_height_in_inches
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_theorem_l1083_108344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1083_108339

/-- The race distance in yards -/
def d : ℝ := sorry

/-- The speed of runner X -/
def x : ℝ := sorry

/-- The speed of runner Y -/
def y : ℝ := sorry

/-- The speed of runner Z -/
def z : ℝ := sorry

/-- Runner X finishes 25 yards ahead of runner Y -/
axiom x_beats_y : d / x = (d - 25) / y

/-- Runner Y finishes 15 yards ahead of runner Z -/
axiom y_beats_z : d / y = (d - 15) / z

/-- Runner X finishes 35 yards ahead of runner Z -/
axiom x_beats_z : d / x = (d - 35) / z

/-- The race distance is 75 yards -/
theorem race_distance : d = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1083_108339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l1083_108395

noncomputable def scores : List ℝ := [91, 89, 88, 90, 92]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem variance_of_scores :
  variance scores = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l1083_108395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_max_volume_l1083_108336

/-- The volume of the water tank as a function of the cut size x -/
noncomputable def volume (x : ℝ) : ℝ := -1/2 * x^3 + 60 * x^2

/-- The derivative of the volume function -/
noncomputable def volume_derivative (x : ℝ) : ℝ := -3/2 * x^2 + 120 * x

theorem water_tank_max_volume :
  ∃ (x : ℝ),
    0 < x ∧ x < 120 ∧
    (∀ y, 0 < y → y < 120 → volume y ≤ volume x) ∧
    x = 80 ∧
    volume x = 128000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_max_volume_l1083_108336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1083_108338

theorem calculation_proofs :
  ((3 / 8 - 1 / 6 - 2 / 3) * 24 = -11) ∧
  (1 / 2 - (-3)^2 + |(-2)| = -13/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1083_108338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1083_108345

noncomputable section

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  A = π / 3 →
  a = 3 →
  Real.sin B + Real.sin C = 2 * Real.sqrt 3 * Real.sin B * Real.sin C →
  -- Part 1
  (1 / b + 1 / c = 1) ∧
  -- Part 2
  (b = Real.sqrt 6 →
   ∃ D : ℝ,
     -- D is on the extension of CA and BD ⊥ BC
     D > 0 ∧
     -- AD = 2√6 + 3√2
     D = 2 * Real.sqrt 6 + 3 * Real.sqrt 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1083_108345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1083_108304

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of line l₁ -/
noncomputable def slope_l₁ (m : ℝ) : ℝ := -(m + 1) / (1 - m)

/-- The slope of line l₂ -/
noncomputable def slope_l₂ (m : ℝ) : ℝ := -(m - 1) / (2*m + 1)

/-- m = 0 is a sufficient but not necessary condition for l₁ to be perpendicular to l₂ -/
theorem sufficient_not_necessary :
  (∃ m : ℝ, m = 0 → perpendicular (slope_l₁ m) (slope_l₂ m)) ∧
  (∃ m : ℝ, perpendicular (slope_l₁ m) (slope_l₂ m) ∧ m ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1083_108304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_16_to_binary_digits_l1083_108312

theorem base_16_to_binary_digits : ∃ (n : ℕ), n = 19 ∧ 
  (∀ k : ℕ, (77777 : ℕ) = k → 
   2^(n-1) ≤ k ∧ k < 2^n) := by
  sorry

#check base_16_to_binary_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_16_to_binary_digits_l1083_108312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1083_108333

theorem sin_2alpha_value (α : ℝ) 
  (h : Matrix.det ![![Real.sin α, Real.cos α], ![2, 1]] = 0) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1083_108333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_surface_area_formula_l1083_108397

/-- A rectangular cuboid with edge lengths a, b, and c. -/
structure RectangularCuboid where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The sum of the lengths of three edges emerging from one vertex. -/
def RectangularCuboid.edgeSum (cuboid : RectangularCuboid) : ℝ :=
  cuboid.a + cuboid.b + cuboid.c

/-- The length of the space diagonal. -/
noncomputable def RectangularCuboid.spaceDiagonal (cuboid : RectangularCuboid) : ℝ :=
  Real.sqrt (cuboid.a^2 + cuboid.b^2 + cuboid.c^2)

/-- The surface area of the cuboid. -/
def RectangularCuboid.surfaceArea (cuboid : RectangularCuboid) : ℝ :=
  2 * (cuboid.a * cuboid.b + cuboid.a * cuboid.c + cuboid.b * cuboid.c)

/-- Theorem: The surface area of a rectangular cuboid is equal to p^2 - q^2,
    where p is the sum of the lengths of three edges emerging from one vertex,
    and q is the length of the space diagonal. -/
theorem cuboid_surface_area_formula (cuboid : RectangularCuboid) (p q : ℝ)
    (h1 : cuboid.edgeSum = p)
    (h2 : cuboid.spaceDiagonal = q) :
    cuboid.surfaceArea = p^2 - q^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_surface_area_formula_l1083_108397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_eq_u_poly_sum_of_coefficients_is_six_l1083_108365

/-- Sequence u_n defined by the given recurrence relation -/
def u : ℕ → ℝ
  | 0 => 6  -- Added case for 0
  | 1 => 6
  | n + 1 => u n + (5 + 2 * (n - 1))

/-- The polynomial representation of u_n -/
def u_poly (n : ℕ) : ℝ := n^2 + 2*n + 3

/-- Theorem stating that u_n equals its polynomial representation -/
theorem u_eq_u_poly : ∀ n : ℕ, u n = u_poly n := by
  sorry

/-- Theorem proving the sum of coefficients of u_n's polynomial representation is 6 -/
theorem sum_of_coefficients_is_six :
  ∃ (a b c : ℝ), (∀ n : ℕ, u n = a * n^2 + b * n + c) ∧ (a + b + c = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_eq_u_poly_sum_of_coefficients_is_six_l1083_108365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_l1083_108378

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp x

theorem odd_function_implies_a_eq_neg_one :
  (∀ x : ℝ, f a x = -(f a (-x))) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_l1083_108378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_ticket_price_l1083_108377

/-- Proves that the price of a child's ticket is $1.50 given the specified conditions -/
theorem theater_ticket_price 
  (total_seats : ℕ) 
  (adult_price : ℚ) 
  (total_income : ℚ) 
  (num_children : ℕ) 
  (h1 : total_seats = 200)
  (h2 : adult_price = 3)
  (h3 : total_income = 510)
  (h4 : num_children = 60)
  : ∃ (child_price : ℚ), 
    child_price * num_children + adult_price * (total_seats - num_children) = total_income ∧ 
    child_price = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_ticket_price_l1083_108377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_odd_l1083_108319

theorem expression_is_odd (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : c > 0) :
  Odd (7^a + 3*(b-1)*c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_odd_l1083_108319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l1083_108358

def article_A_purchase : ℕ := 15
def article_B_purchase : ℕ := 20
def article_C_purchase : ℕ := 30

def article_A_cost : ℚ := 25
def article_B_cost : ℚ := 40
def article_C_cost : ℚ := 55

def article_A_sell : ℕ := 12
def article_B_sell : ℕ := 18
def article_C_sell : ℕ := 25

def article_A_price : ℚ := 38
def article_B_price : ℚ := 50
def article_C_price : ℚ := 65

def total_cost : ℚ := 
  article_A_purchase * article_A_cost + 
  article_B_purchase * article_B_cost + 
  article_C_purchase * article_C_cost

def total_revenue : ℚ := 
  article_A_sell * article_A_price + 
  article_B_sell * article_B_price + 
  article_C_sell * article_C_price

def profit : ℚ := total_revenue - total_cost

def profit_percentage : ℚ := (profit / total_cost) * 100

theorem dealer_profit_percentage : 
  abs (profit_percentage - 5.52) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l1083_108358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_m_value_l1083_108337

/-- Given points P(-2,m) and Q(m,4), if the line PQ is perpendicular to the line x+y+1=0, then m = 1 -/
theorem perpendicular_line_m_value (m : ℝ) : 
  (((m - 4) / (-2 - m)) * (-1) = -1) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_m_value_l1083_108337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1083_108399

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 2 * x^2 + 1

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := (1 + x) * Real.exp x - 4 * x

-- Theorem statement
theorem tangent_line_at_zero_one :
  ∃ (m b : ℝ), 
    (∀ x, m * x + b = f 0 + f' 0 * (x - 0)) ∧
    f 0 = 1 ∧
    m = 1 ∧
    b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1083_108399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1083_108392

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_positive : d > 0) :
  increasing_sequence a ∧
  ¬ increasing_sequence (λ n => n * a n) ∧
  ¬ increasing_sequence (λ n => a n / n) ∧
  increasing_sequence (λ n => a n + 3 * n * d) :=
sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1083_108392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_decomposition_and_cube_decomposition_l1083_108317

/-- Represents the sum of the first n odd numbers -/
def sumFirstOddNumbers (n : ℕ) : ℕ :=
  List.range n |>.map (fun i => 2 * i + 1) |>.sum

/-- Represents the smallest number in the decomposition of m^3 -/
def smallestInCubeDecomposition (m : ℕ) : ℕ :=
  2 * m - 1

theorem power_two_decomposition_and_cube_decomposition :
  (sumFirstOddNumbers 5 = 5^2) ∧
  (∃ m : ℕ, m > 0 ∧ smallestInCubeDecomposition m = 21 ∧ m = 5) := by
  sorry

#eval sumFirstOddNumbers 5  -- To check the result
#eval smallestInCubeDecomposition 5  -- To check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_decomposition_and_cube_decomposition_l1083_108317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_theorem_sin_B_theorem_l1083_108305

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A = Real.pi/3 ∧ t.a > 0 ∧ t.b > 0 ∧ t.c > 0

-- Part 1: Maximum area
noncomputable def max_area (t : Triangle) : ℝ := 3 * Real.sqrt 3 / 4

theorem max_area_theorem (t : Triangle) 
  (h : triangle_conditions t) (ha : t.a = Real.sqrt 3) : 
  ∃ (area : ℝ), area ≤ max_area t ∧ 
  ∀ (other_area : ℝ), other_area ≤ area := by sorry

-- Part 2: Value of sin B
noncomputable def sin_B_value (t : Triangle) : ℝ := (Real.sqrt 39 + Real.sqrt 3) / 8

theorem sin_B_theorem (t : Triangle) 
  (h : triangle_conditions t) (hc : t.c = t.a / 2) : 
  Real.sin t.B = sin_B_value t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_theorem_sin_B_theorem_l1083_108305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1083_108349

/-- The inclination angle of the line cos(π/6)x + sin(π/6)y + 2 = 0 is 2π/3 -/
theorem line_inclination_angle :
  ∃ θ : ℝ, θ = 2*π/3 ∧ 
    (let slope := -(Real.cos (π/6) / Real.sin (π/6));
     θ = Real.arctan slope + π) :=
by
  -- We'll use existence introduction and prove the two conjuncts
  use 2*π/3
  constructor
  
  -- First conjunct: θ = 2π/3
  · rfl

  -- Second conjunct: θ = arctan(slope) + π
  · 
    -- Define the slope
    let slope := -(Real.cos (π/6) / Real.sin (π/6))
    
    -- The main proof goes here
    -- We would need to show that arctan(slope) + π = 2π/3
    -- This involves trigonometric calculations
    sorry -- Placeholder for the detailed proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1083_108349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_existence_l1083_108326

/-- Given polynomials P and Q, and a polynomial R such that 
    P(x) - P(y) = R(x, y)(Q(x) - Q(y)) for all x and y,
    there exists a polynomial S such that P(x) = S(Q(x)) for all x. -/
theorem polynomial_composition_existence 
  {K : Type*} [Field K] (P Q : Polynomial K) :
  (∃ R : Polynomial K × Polynomial K, ∀ x y : K, 
    P.eval x - P.eval y = (R.1.eval x * R.2.eval y) * (Q.eval x - Q.eval y)) →
  ∃ S : Polynomial K, ∀ x : K, P.eval x = S.eval (Q.eval x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_existence_l1083_108326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vivian_total_break_time_l1083_108362

/-- Represents the duration of breaks in minutes -/
structure BreakSchedule where
  morning_warmup : ℕ
  morning_recess1 : ℕ
  morning_recess2 : ℕ
  lunch_break : ℕ
  lunch_transition : ℕ
  afternoon_recess1 : ℕ
  afternoon_recess2 : ℕ
  monday_assembly : ℕ
  friday_tutoring : ℕ
  wednesday_class_meeting : ℕ

/-- Calculates the total break time for the week given a break schedule -/
def totalBreakTime (schedule : BreakSchedule) : ℕ :=
  let daily_total := schedule.morning_warmup + schedule.morning_recess1 + schedule.morning_recess2 +
                     schedule.lunch_break + schedule.lunch_transition +
                     schedule.afternoon_recess1 + schedule.afternoon_recess2
  let weekly_total := daily_total * 5
  let wednesday_adjustment := schedule.wednesday_class_meeting - schedule.afternoon_recess1
  weekly_total + wednesday_adjustment + schedule.monday_assembly + schedule.friday_tutoring

/-- The break schedule for Vivian's students -/
def vivianSchedule : BreakSchedule :=
  { morning_warmup := 10
  , morning_recess1 := 15
  , morning_recess2 := 15
  , lunch_break := 25
  , lunch_transition := 5
  , afternoon_recess1 := 20
  , afternoon_recess2 := 10
  , monday_assembly := 30
  , friday_tutoring := 45
  , wednesday_class_meeting := 35 }

/-- Theorem stating that the total break time for Vivian's students is 590 minutes per week -/
theorem vivian_total_break_time :
  totalBreakTime vivianSchedule = 590 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vivian_total_break_time_l1083_108362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_implies_negative_k_l1083_108332

/-- A function representing an inverse proportion --/
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x => k / x

/-- Property that a function increases as x increases in each quadrant --/
def increases_in_each_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (x₁ < 0 ∧ x₂ < 0) ∨ (x₁ > 0 ∧ x₂ > 0) → f x₁ < f x₂

/-- Theorem stating that if an inverse proportion function increases in each quadrant, then k < 0 --/
theorem inverse_proportion_increases_implies_negative_k (k : ℝ) :
  increases_in_each_quadrant (inverse_proportion k) → k < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_implies_negative_k_l1083_108332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_parameterized_line_l1083_108313

/-- Given a line l with parameterized equation x = a + t, y = b + t,
    prove that the distance between point P1(a+t1, b+t1) and point P(a,b) is √2|t1| -/
theorem distance_on_parameterized_line (a b t1 : ℝ) :
  let P := (a, b)
  let P1 := (a + t1, b + t1)
  Real.sqrt ((P1.1 - P.1)^2 + (P1.2 - P.2)^2) = Real.sqrt 2 * |t1| := by
  sorry

#check distance_on_parameterized_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_parameterized_line_l1083_108313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sequence_sum_ratio_l1083_108380

/-- A geometric sequence with positive terms where a₃, a₅, -a₄ form an arithmetic sequence -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  is_positive : ∀ n, a n > 0
  is_geometric : ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n
  special_condition : ∃ d : ℝ, a 5 - a 3 = d ∧ a 3 - (-a 4) = d

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : SpecialGeometricSequence) (n : ℕ) : ℝ :=
  let q := Classical.choose seq.is_geometric
  (seq.a 1) * (1 - q^n) / (1 - q)

/-- The main theorem -/
theorem special_geometric_sequence_sum_ratio
  (seq : SpecialGeometricSequence) :
  sum_n seq 6 / sum_n seq 3 = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sequence_sum_ratio_l1083_108380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_problem_l1083_108387

/-- Triangle type -/
structure Triangle where
  -- Define the properties of a triangle here
  mk :: -- You can add necessary fields

/-- Similarity relation between triangles -/
def Similar (t1 t2 : Triangle) : Prop := sorry

/-- Corresponding sides of similar triangles -/
def CorrespondingSides (t1 t2 : Triangle) (s1 s2 : ℝ) : Prop := sorry

/-- Corresponding angle bisectors of similar triangles -/
def CorrespondingAngleBisectors (t1 t2 : Triangle) (b1 b2 : ℝ) : Prop := sorry

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- 
Given two similar triangles with corresponding sides of lengths 3 and 2,
prove that if the sum of their corresponding angle bisectors is 15
and the difference in their areas is 6, then the lengths of the
angle bisectors are 9 and 6, and the areas of the triangles are 54/5 and 24/5.
-/
theorem similar_triangles_problem 
  (t1 t2 : Triangle) 
  (h_similar : Similar t1 t2)
  (h_sides : ∃ (s1 s2 : ℝ), CorrespondingSides t1 t2 s1 s2 ∧ s1 = 3 ∧ s2 = 2)
  (h_bisectors : ∃ (b1 b2 : ℝ), CorrespondingAngleBisectors t1 t2 b1 b2 ∧ b1 + b2 = 15)
  (h_area_diff : area t1 - area t2 = 6) :
  ∃ (b1 b2 : ℝ), CorrespondingAngleBisectors t1 t2 b1 b2 ∧ 
    b1 = 9 ∧ b2 = 6 ∧ area t1 = 54/5 ∧ area t2 = 24/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_problem_l1083_108387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_625_l1083_108384

theorem power_of_625 : (625 : ℝ) ^ (12 / 100) * (625 : ℝ) ^ (8 / 100) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_625_l1083_108384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1083_108381

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1083_108381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l1083_108314

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a set of 5 points in a plane -/
def FivePoints := Fin 5 → Point

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

/-- Predicate to check if no three points in the set are collinear -/
def NoThreeCollinear (points : FivePoints) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areCollinear (points i) (points j) (points k)

/-- The number of intersection points of perpendiculars -/
noncomputable def NumIntersections (points : FivePoints) : ℕ := sorry

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersections (points : FivePoints) (h : NoThreeCollinear points) :
  NumIntersections points = 310 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l1083_108314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1083_108355

noncomputable def f (x : ℝ) : ℝ :=
  (1 + (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + x) / (1 - 3 * x))) /
  (1 - 3 * (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + x) / (1 - 3 * x)))

noncomputable def g (x : ℝ) : ℝ :=
  (1 - 3 * (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + x) / (1 - 3 * x))) /
  (1 + (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + x) / (1 - 3 * x)))

theorem inequality_solution_set (x : ℝ) :
  f x < g x ↔ (x < -1 ∨ (0 < x ∧ x < 1 ∧ x ≠ 1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1083_108355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1083_108361

/-- The solution set of the inequality ax^2 + (a-1)x - 1 > 0 -/
noncomputable def SolutionSet (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Ioi (-1)
  else if a > 0 then Set.Ioi (-1) ∪ Set.Ioi (1/a)
  else if a < -1 then Set.Ioo (-1) (1/a)
  else if -1 < a ∧ a < 0 then Set.Ioo (1/a) (-1)
  else ∅

theorem inequality_theorem (a : ℝ) :
  (∀ x, a * x^2 + (a - 1) * x - 1 > 0 → x ∈ SolutionSet a) ∧
  (a * (-a)^2 + (a - 1) * (-a) - 1 > 0 → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1083_108361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_dodecagon_area_ratio_l1083_108354

/-- Predicate indicating that a real number represents the area of a regular dodecagon. -/
def is_regular_dodecagon (area : ℝ) : Prop := sorry

/-- Predicate indicating that a real number represents the area of a regular hexagon. -/
def is_regular_hexagon (area : ℝ) : Prop := sorry

/-- Predicate indicating that the hexagon with area Q is inscribed in the dodecagon with area P
    by connecting every second vertex of the dodecagon. -/
def is_inscribed_hexagon_in_dodecagon (P Q : ℝ) : Prop := sorry

/-- The ratio of the area of a regular hexagon inscribed in a regular dodecagon
    (formed by connecting every second vertex) to the area of the dodecagon. -/
theorem hexagon_dodecagon_area_ratio :
  ∀ (P Q : ℝ),
  P > 0 →
  Q > 0 →
  is_regular_dodecagon P →
  is_regular_hexagon Q →
  is_inscribed_hexagon_in_dodecagon P Q →
  Q / P = (2 * Real.sqrt 3 - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_dodecagon_area_ratio_l1083_108354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_budget_l1083_108357

/-- Represents the taxi fare structure in Metropolis City -/
structure TaxiFare where
  initial_fare : ℚ
  initial_distance : ℚ
  additional_rate : ℚ
  additional_distance : ℚ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare : TaxiFare) (distance : ℚ) : ℚ :=
  fare.initial_fare + max 0 (distance - fare.initial_distance) / fare.additional_distance * fare.additional_rate

/-- Theorem: The maximum distance that can be traveled with a $15 budget (including a $3 tip) is 3.5 miles -/
theorem max_distance_for_budget (fare : TaxiFare) (budget : ℚ) (tip : ℚ) : 
  fare.initial_fare = 3 ∧ 
  fare.initial_distance = 1/2 ∧ 
  fare.additional_rate = 3/10 ∧ 
  fare.additional_distance = 1/10 ∧
  budget = 15 ∧ 
  tip = 3 →
  ∃ (distance : ℚ), distance = 7/2 ∧ calculate_fare fare distance + tip = budget :=
by sorry

#eval calculate_fare { initial_fare := 3, initial_distance := 1/2, additional_rate := 3/10, additional_distance := 1/10 } (7/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_budget_l1083_108357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_one_over_one_equals_one_l1083_108310

theorem sqrt_one_over_one_equals_one : Real.sqrt 1 / 1 = 1 := by
  rw [Real.sqrt_one, div_one]

#check sqrt_one_over_one_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_one_over_one_equals_one_l1083_108310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_third_l1083_108346

noncomputable section

def Point := ℝ × ℝ

theorem sin_alpha_plus_pi_third (α : ℝ) (P : Point) :
  (P.1 = 1 ∧ P.2 = Real.sqrt 3) →  -- Point P(1, √3) is on the terminal side of angle α
  Real.sin (α + π/3) = Real.sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_third_l1083_108346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_recursive_formula_l1083_108386

def x : ℕ → ℕ
  | 0 => 2  -- Adding case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => if n % 2 = 0 then x (n + 2) + x (n + 1) else x (n + 2) + 2 * x (n + 1)

theorem x_recursive_formula (n : ℕ) :
  (n ≥ 4 ∧ n % 2 = 0 → x n = 4 * x (n - 2) - 2 * x (n - 4)) ∧
  (n ≥ 3 ∧ n % 2 = 1 → x n = 4 * x (n - 2) - 2 * x (n - 4)) := by
  sorry

#eval x 5  -- Just to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_recursive_formula_l1083_108386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emery_reading_time_l1083_108396

-- Define the reading times for Emery and Serena
variable (emery_time serena_time : ℝ)

-- Define the conditions
axiom emery_faster : emery_time = (1/5) * serena_time
axiom average_time : (emery_time + serena_time) / 2 = 60

-- Theorem to prove
theorem emery_reading_time : emery_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emery_reading_time_l1083_108396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_l1083_108353

/-- The function f(x) = -x + 1/(2x) -/
noncomputable def f (x : ℝ) : ℝ := -x + 1/(2*x)

/-- Theorem stating that f is an odd function -/
theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

/-- Theorem stating that f is decreasing on (0, +∞) -/
theorem f_is_decreasing : ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 > f x2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_l1083_108353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_natural_satisfies_conditions_l1083_108375

theorem no_natural_satisfies_conditions : ¬ ∃ (n : ℕ), (9 ∣ (2*n - 5)) ∧ (15 ∣ (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_natural_satisfies_conditions_l1083_108375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1083_108351

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x + Real.pi/4))^2

theorem function_properties (a b : ℝ) (h1 : a = f (Real.log 5)) (h2 : b = f (Real.log (1/5))) :
  (a + b = 1) ∧ (a - b = Real.sin (2 * Real.log 5)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1083_108351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l1083_108328

open Real

-- Define the function f
noncomputable def f (x : ℝ) := Real.sin x + 5 * x

-- Define the theorem
theorem alpha_range (α : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f x = Real.sin x + 5 * x) →
  f (1 - α) + f (1 - α^2) < 0 →
  α ∈ Set.Ioo 1 (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l1083_108328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l1083_108388

noncomputable def small_circle_circumference : ℝ := 18 * Real.pi
def radius_difference : ℝ := 10

theorem annulus_area : 
  let r := small_circle_circumference / (2 * Real.pi)
  let R := r + radius_difference
  Real.pi * R^2 - Real.pi * r^2 = 280 * Real.pi := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l1083_108388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_and_q_is_false_l1083_108343

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) * (1 + x))
noncomputable def g (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

-- Define evenness for a function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the propositions
def p : Prop := is_even f
def q : Prop := is_even g

-- The theorem to prove
theorem not_p_and_q_is_false : ¬((¬p) ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_and_q_is_false_l1083_108343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inside_triangle_l1083_108307

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a median
def median (t : Triangle) (vertex : Fin 3) : ℝ × ℝ := sorry

-- Define the centroid
def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define a point being inside a triangle
def is_inside_triangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

-- Theorem stating that the centroid is always inside the triangle
theorem centroid_inside_triangle (t : Triangle) : 
  is_inside_triangle (centroid t) t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inside_triangle_l1083_108307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ninth_game_score_l1083_108374

def basketball_problem (scores : List ℕ) (eight_game_avg : ℚ) (nine_game_avg : ℚ) : Prop :=
  -- First four games scores
  scores = [18, 22, 15, 20] ∧
  -- Average after eight games is higher than after four games
  eight_game_avg > (18 + 22 + 15 + 20) / 4 ∧
  -- Average after nine games is greater than 19
  nine_game_avg > 19 ∧
  -- The ninth game score is the minimum possible
  ∀ (ninth_score : ℕ),
    (List.sum scores + (8 * eight_game_avg).ceil + ninth_score) / 9 > 19 →
    ninth_score ≥ 21

theorem min_ninth_game_score (scores : List ℕ) (eight_game_avg : ℚ) (nine_game_avg : ℚ)
  (h : basketball_problem scores eight_game_avg nine_game_avg) :
  ∃ (ninth_score : ℕ), ninth_score = 21 ∧
    (List.sum scores + (8 * eight_game_avg).ceil + ninth_score) / 9 > 19 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ninth_game_score_l1083_108374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_min_value_of_f_inequality_proof_l1083_108316

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log (f x - 2)

-- Theorem for the domain of g
theorem domain_of_g : Set ℝ := { x | 0.5 < x ∧ x < 2.5 }

-- Theorem for the minimum value of f
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 1 := by
  sorry

-- Theorem for the inequality
theorem inequality_proof (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_min_value_of_f_inequality_proof_l1083_108316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1083_108347

def total_silverware : ℕ := 24
def forks : ℕ := 8
def spoons : ℕ := 7
def knives : ℕ := 5
def teaspoons : ℕ := 4
def pieces_to_choose : ℕ := 4

noncomputable def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose forks 2 * Nat.choose spoons 1 * Nat.choose knives 1) / Nat.choose total_silverware pieces_to_choose

theorem probability_theorem :
  probability_two_forks_one_spoon_one_knife = 196 / 2530 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1083_108347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_two_l1083_108379

/-- The motion equation of an object -/
noncomputable def s (t : ℝ) : ℝ := t^2 + 3/t

/-- The velocity of the object -/
noncomputable def v (t : ℝ) : ℝ := deriv s t

/-- Theorem: The velocity of the object at t = 2 is 13/4 -/
theorem velocity_at_two : v 2 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_two_l1083_108379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_g_lower_bound_tight_g_upper_bound_tight_l1083_108341

/-- The function g defined for positive real numbers a, b, c -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a + 2*b) + b / (b + 2*c) + c / (c + 2*a)

/-- Theorem stating the bounds of g(a,b,c) for positive real numbers a, b, c -/
theorem g_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  3/2 ≤ g a b c ∧ g a b c ≤ 3 := by
  sorry

/-- Theorem stating that the lower bound 3/2 is tight -/
theorem g_lower_bound_tight :
  ∀ ε > 0, ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ g a b c < 3/2 + ε := by
  sorry

/-- Theorem stating that the upper bound 3 is tight -/
theorem g_upper_bound_tight :
  ∀ ε > 0, ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ g a b c > 3 - ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_g_lower_bound_tight_g_upper_bound_tight_l1083_108341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l1083_108325

-- Define set M
def M : Set ℝ := {x | |x + 2| ≤ 1}

-- Define set N
def N : Set ℝ := {x | ∃ a : ℝ, x = 2 * Real.sin a}

-- Theorem statement
theorem union_of_M_and_N : M ∪ N = Set.Icc (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l1083_108325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_is_three_l1083_108311

-- Define the logarithm property
def log_property (b : ℝ) : Prop := Real.logb b 729 = 6

-- Theorem stating that 3 is the base satisfying the log property
theorem log_base_is_three : ∃ (b : ℝ), log_property b ∧ b = 3 := by
  -- Introduce the witness
  use 3
  
  constructor
  
  -- Prove the log property
  · unfold log_property
    -- Here we would need to prove that Real.logb 3 729 = 6
    sorry
  
  -- Prove that b = 3
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_is_three_l1083_108311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1083_108389

def sequenceA (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 2
  else sequenceA (n - 1) + 3 * (n - 1)

theorem sequence_formula (n : ℕ) :
  n ≥ 1 → sequenceA n = 2 + (3 * n * (n - 1)) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1083_108389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1083_108323

noncomputable section

-- Define the vectors and function
def OA (a ω φ x : Real) : Real × Real := (2 * a * (Real.cos ((ω * x + φ) / 2))^2, 1)
def OB (a ω φ x : Real) : Real × Real := (1, Real.sqrt 3 * a * Real.sin (ω * x + φ) - a)
def f (a ω φ x : Real) : Real := (OA a ω φ x).1 * (OB a ω φ x).1 + (OA a ω φ x).2 * (OB a ω φ x).2

-- State the theorem
theorem function_properties 
  (a ω φ : Real) 
  (ha : a ≠ 0) 
  (hω : ω > 0) 
  (hφ : 0 < φ ∧ φ < Real.pi / 2) :
  -- The function f can be simplified to the given form
  (∀ x, f a ω φ x = 2 * a * Real.sin (2 * x + Real.pi / 3)) ∧
  -- The distance between adjacent highest points is π
  (∃ T, T = Real.pi ∧ ∀ x, f a ω φ (x + T) = f a ω φ x) ∧
  -- The graph has a symmetrical axis at x = π/12
  (∀ x, f a ω φ (Real.pi / 6 - x) = f a ω φ (Real.pi / 6 + x)) ∧
  -- Monotonically increasing intervals when a > 0
  (a > 0 → ∀ k : Int, ∀ x, 
    k * Real.pi - 5 * Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 12 → 
    ∀ y, x ≤ y → f a ω φ x ≤ f a ω φ y) ∧
  -- Maximum and minimum values on [0, π/2]
  (∃ b : Real, 
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 1 ω φ x + b ≤ 2) ∧
    (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f 1 ω φ x + b = 2) ∧
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 1 ω φ x + b ≥ -Real.sqrt 3) ∧
    (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f 1 ω φ x + b = -Real.sqrt 3) ∧
    b = -1 + Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1083_108323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_us_avg_sq_feet_per_person_approx_l1083_108342

/-- The population of the United States in 2020 (in millions) -/
noncomputable def us_population : ℝ := 331

/-- The total area of the United States (in square miles) -/
noncomputable def us_area : ℝ := 3796742

/-- The number of square feet in one square mile -/
noncomputable def sq_feet_per_sq_mile : ℝ := 5280^2

/-- The average number of square feet per person in the United States -/
noncomputable def avg_sq_feet_per_person : ℝ := (us_area * sq_feet_per_sq_mile) / (us_population * 1000000)

/-- Theorem stating that the average number of square feet per person in the United States is approximately 320,000 -/
theorem us_avg_sq_feet_per_person_approx :
  abs (avg_sq_feet_per_person - 320000) < 1000 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_us_avg_sq_feet_per_person_approx_l1083_108342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_siding_cost_l1083_108359

/-- Calculates the cost of siding needed for Sandy's daughter's playhouse --/
theorem sandys_siding_cost :
  let wall_width : ℚ := 8
  let wall_height : ℚ := 10
  let roof_width : ℚ := 8
  let roof_height : ℚ := 5
  let siding_width : ℚ := 10
  let siding_height : ℚ := 12
  let cost_per_section : ℚ := 30.50

  let wall_area := wall_width * wall_height
  let roof_area := 2 * roof_width * roof_height
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let sections_needed := ⌈(total_area / siding_area)⌉

  sections_needed * cost_per_section = 61 := by
  sorry

#eval (⌈(160 : ℚ) / 120⌉ : ℚ) * 30.50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_siding_cost_l1083_108359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l1083_108390

/-- Sum of digits function -/
noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Absolute value function for integers -/
def abs_int (z : ℤ) : ℕ := Int.natAbs z

/-- Condition for the polynomial -/
def satisfies_condition (P : IntPolynomial) : Prop :=
  ∀ x y : ℕ, sum_of_digits x = sum_of_digits y →
    sum_of_digits (abs_int (P x)) = sum_of_digits (abs_int (P y))

/-- Characterization of polynomials satisfying the condition -/
theorem polynomial_characterization (P : IntPolynomial) :
  satisfies_condition P ↔
    ∃ k c : ℕ, c < 10^k ∧
      ((∀ x, P x = (10^k : ℤ) * x + c) ∨
       (∀ x, P x = -((10^k : ℤ) * x + c))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l1083_108390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_altitude_is_10_l1083_108367

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
noncomputable def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude / 2

/-- Theorem: The altitude of the small cone cut off from the given frustum is 10 cm -/
theorem small_cone_altitude_is_10 (f : Frustum) 
  (h_altitude : f.altitude = 20)
  (h_lower_base : f.lower_base_area = 324 * Real.pi)
  (h_upper_base : f.upper_base_area = 36 * Real.pi) :
  small_cone_altitude f = 10 := by
  sorry

#check small_cone_altitude_is_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_altitude_is_10_l1083_108367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_problem_l1083_108394

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a₁ : ℝ  -- First term
  d : ℝ   -- Common difference

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (seq.a₁ + seq.nthTerm n) / 2

theorem weaving_problem (seq : ArithmeticSequence) 
  (h1 : seq.sumFirstN 7 = 28)
  (h2 : seq.nthTerm 2 + seq.nthTerm 5 + seq.nthTerm 8 = 15) :
  seq.nthTerm 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_problem_l1083_108394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1083_108315

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- Main theorem -/
theorem ellipse_triangle_area
  (G : Ellipse)
  (l : Line)
  (P A B : Point)
  (h_ecc : eccentricity G = Real.sqrt 6 / 3)
  (h_focus : Point.mk (2 * Real.sqrt 2) 0 ∈ {p : Point | p.x^2 / G.a^2 + p.y^2 / G.b^2 = 1})
  (h_slope : l.m = 1)
  (h_intersect : A ∈ {p : Point | p.x^2 / G.a^2 + p.y^2 / G.b^2 = 1} ∧
                 B ∈ {p : Point | p.x^2 / G.a^2 + p.y^2 / G.b^2 = 1} ∧
                 A.y = l.m * A.x + l.c ∧ B.y = l.m * B.x + l.c)
  (h_P : P = Point.mk (-3) 2)
  (h_isosceles : distance P A = distance P B) :
  triangleArea (distance A B) (abs (P.y - (l.m * P.x + l.c)) / Real.sqrt (1 + l.m^2)) = 9/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1083_108315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_ratio_l1083_108348

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.tan x - 1) + Real.sqrt (3 - Real.tan x)

-- Define the domain condition
def domain_condition (x : ℝ) : Prop := 1 ≤ Real.tan x ∧ Real.tan x ≤ 3

-- Theorem statement
theorem function_extrema_and_ratio :
  ∃ (M N : ℝ),
    (∀ x, domain_condition x → f x ≤ M) ∧
    (∀ x, domain_condition x → N ≤ f x) ∧
    (∃ x₁ x₂, domain_condition x₁ ∧ domain_condition x₂ ∧ f x₁ = M ∧ f x₂ = N) ∧
    M = 2 ∧ N = Real.sqrt 2 ∧ M / N = Real.sqrt 2 :=
by
  sorry

#check function_extrema_and_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_ratio_l1083_108348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l1083_108340

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define points A and B on the left branch of the hyperbola
variable (A B : ℝ × ℝ)

-- Axioms
axiom A_on_hyperbola : hyperbola A.1 A.2
axiom B_on_hyperbola : hyperbola B.1 B.2
axiom A_B_F₁_collinear : ∃ (t : ℝ), A.1 = t * F₁.1 ∧ A.2 = t * F₁.2 ∧
                                    B.1 = t * F₁.1 ∧ B.2 = t * F₁.2

-- Define the perimeter of triangle F₂AB
noncomputable def perimeter (A B : ℝ × ℝ) : ℝ := dist A F₂ + dist B F₂ + dist A B

-- Theorem to prove
theorem min_perimeter : 
  ∃ (min_perim : ℝ), min_perim = 10 ∧ 
  ∀ (A B : ℝ × ℝ), hyperbola A.1 A.2 → hyperbola B.1 B.2 → 
  (∃ (t : ℝ), A.1 = t * F₁.1 ∧ A.2 = t * F₁.2 ∧ B.1 = t * F₁.1 ∧ B.2 = t * F₁.2) →
  perimeter A B ≥ min_perim :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l1083_108340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_prize_amount_l1083_108306

def fix_cost : ℝ := 20000
def discount_rate : ℝ := 0.20
def kept_prize_rate : ℝ := 0.90
def total_made : ℝ := 47000

theorem race_prize_amount :
  let discounted_cost : ℝ := fix_cost * (1 - discount_rate)
  let prize_kept : ℝ := total_made - discounted_cost
  let prize_amount : ℝ := prize_kept / kept_prize_rate
  ∃ ε > 0, |prize_amount - 34444.44| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_prize_amount_l1083_108306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_nitrate_moles_l1083_108352

/-- Represents a chemical substance -/
structure Substance where
  name : String

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List (Substance × ℕ)
  products : List (Substance × ℕ)

/-- The number of moles of a substance in a reaction -/
def moles (s : Substance) (r : Reaction) : ℕ :=
  (r.reactants ++ r.products).filter (λ p => p.1.name = s.name) |>.map Prod.snd |>.sum

theorem silver_nitrate_moles (agno3 hcl agcl hno3 : Substance)
    (r : Reaction)
    (h1 : r.reactants = [(agno3, 1), (hcl, 1)])
    (h2 : r.products = [(agcl, 1), (hno3, 1)])
    (h3 : moles hcl r = 2)
    (h4 : moles hno3 r = 2) :
  moles agno3 r = moles hno3 r := by
  sorry

#check silver_nitrate_moles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_nitrate_moles_l1083_108352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_rate_approximation_l1083_108308

noncomputable def investment1 : ℝ := 2200
noncomputable def rate1 : ℝ := 0.05
noncomputable def investment2 : ℝ := 1100
noncomputable def rate2 : ℝ := 0.08

noncomputable def totalInvestment : ℝ := investment1 + investment2
noncomputable def totalAnnualIncome : ℝ := investment1 * rate1 + investment2 * rate2

noncomputable def equivalentRate : ℝ := totalAnnualIncome / totalInvestment

theorem equivalent_rate_approximation :
  ∃ ε > 0, abs (equivalentRate - 0.06) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_rate_approximation_l1083_108308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_weighted_average_l1083_108309

def num_students : Nat := 11

def weights : Fin num_students → ℝ
  | ⟨0, _⟩ => 1.2
  | ⟨1, _⟩ => 1.6
  | ⟨2, _⟩ => 1.8
  | ⟨3, _⟩ => 1.4
  | ⟨4, _⟩ => 1
  | ⟨5, _⟩ => 1.5
  | ⟨6, _⟩ => 2
  | ⟨7, _⟩ => 1.3
  | ⟨8, _⟩ => 1.7
  | ⟨9, _⟩ => 1.9
  | ⟨10, _⟩ => 1.1
  | ⟨n+11, h⟩ => by simp [num_students] at h

def sum_weights : ℝ := (Finset.univ.sum weights)

theorem doubled_weighted_average 
  (marks : Fin num_students → ℝ) 
  (h : (Finset.univ.sum (λ i => weights i * marks i)) / sum_weights = 36) :
  (Finset.univ.sum (λ i => weights i * (2 * marks i))) / sum_weights = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_weighted_average_l1083_108309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_two_zero_points_l1083_108368

/-- The function f(x) = a(x - 2e) · ln(x) + 1 has exactly two zero points if and only if a ∈ (-∞, 0) ∪ (1/e, +∞) -/
theorem function_two_zero_points (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧
    a * (x - 2 * Real.exp 1) * Real.log x + 1 = 0 ∧
    a * (y - 2 * Real.exp 1) * Real.log y + 1 = 0 ∧
    ∀ z : ℝ, z > 0 → z ≠ x → z ≠ y →
      a * (z - 2 * Real.exp 1) * Real.log z + 1 ≠ 0) ↔
  (a < 0 ∨ a > 1 / Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_two_zero_points_l1083_108368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_contribution_is_21000_l1083_108329

/-- Calculates the partner's contribution given the initial investment, time periods, and profit ratio --/
def partnerContribution (initialInvestment : ℚ) (totalMonths : ℕ) (partnerJoinMonth : ℕ) (profitRatio : ℚ × ℚ) : ℚ :=
  let partnerMonths := totalMonths - partnerJoinMonth
  let initialInvestorShare := profitRatio.1
  let partnerShare := profitRatio.2
  (initialInvestment * totalMonths * partnerShare) / (initialInvestorShare * partnerMonths)

/-- Theorem stating that B's contribution is 21000 given the problem conditions --/
theorem b_contribution_is_21000 :
  partnerContribution 3500 12 9 (2, 3) = 21000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_contribution_is_21000_l1083_108329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_cos_eq_zero_solution_l1083_108331

theorem sin_sqrt3_cos_eq_zero_solution :
  ∃! x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ Real.sin x + Real.sqrt 3 * Real.cos x = 0 ∧ x = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_cos_eq_zero_solution_l1083_108331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1083_108324

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  perimeter : Real
  area : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.perimeter = Real.sqrt 2 + 1)
  (h2 : Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C)
  (h3 : t.area = (1/6) * Real.sin t.C) : 
  t.c = 1 ∧ t.C = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1083_108324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_flat_speed_l1083_108376

/-- Cyclist's journey parameters --/
structure CyclistJourney where
  total_distance : ℝ
  total_time : ℝ
  flat_distance_ratio : ℝ
  hill_speed_reduction : ℝ

/-- Calculate the speed on the flat part of the journey in km/h --/
noncomputable def flat_speed (journey : CyclistJourney) : ℝ :=
  let flat_distance := journey.total_distance * journey.flat_distance_ratio
  let flat_time := journey.total_time * journey.flat_distance_ratio
  (flat_distance / flat_time) * (60 / 1000)

/-- Theorem stating the cyclist's speed on the flat part --/
theorem cyclist_flat_speed :
  let journey : CyclistJourney := {
    total_distance := 1080,
    total_time := 12,
    flat_distance_ratio := 0.5,
    hill_speed_reduction := 0.3
  }
  flat_speed journey = 5.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_flat_speed_l1083_108376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1083_108371

theorem exponential_inequality (x : ℝ) : (2 : ℝ)^x + (2 : ℝ)^(-x) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1083_108371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_in_list_l1083_108350

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (lst : List Int) : List Int :=
  lst.filter (λ x => x > 0)

def range (lst : List Int) : Int :=
  match lst.maximum?, lst.minimum? with
  | some max, some min => max - min
  | _, _ => 0

theorem range_of_positive_integers_in_list :
  let K := consecutive_integers (-3) 10
  range (positive_integers K) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_in_list_l1083_108350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1083_108391

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi/4) = Real.sqrt 2
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos (θ - Real.pi/4)

-- Define points M and N on C₁ and C₂ respectively
def M (x y : ℝ) : Prop := x - y + 2 = 0
def N (x y θ : ℝ) : Prop := x = Real.sqrt 2 / 2 + Real.cos θ ∧ y = Real.sqrt 2 / 2 + Real.sin θ

-- State the theorem
theorem min_distance_C₁_C₂ : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    M x₁ y₁ → (∃ θ₂, N x₂ y₂ θ₂) → 
    ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧ 
    ∀ (x₁' y₁' x₂' y₂' : ℝ), 
      M x₁' y₁' → (∃ θ₂', N x₂' y₂' θ₂') → 
      Real.sqrt ((x₁' - x₂')^2 + (y₁' - y₂')^2) ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1083_108391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l1083_108356

theorem certain_number_proof : ∃ (x : ℕ), 
  Real.sqrt ((((x + 10) * 2) / 2) ^ 3) - 2 = 270 / 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l1083_108356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1083_108300

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * (2 - a) / x + (a + 2) * log x - a * x - 2

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 4 * b * x - 1/4

theorem function_inequality (a : ℝ) (b : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (∀ x₁ : ℝ, 0 < x₁ ∧ x₁ ≤ exp 1 → ∃ x₂ : ℝ, 0 < x₂ ∧ x₂ ≤ 2 ∧ f 1 x₁ ≥ g b x₂) →
  b ≥ sqrt 3 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1083_108300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1083_108318

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

noncomputable def combined_rate (rate_a rate_b : ℝ) : ℝ := rate_a + rate_b

theorem work_completion_time 
  (total_days : ℝ) 
  (combined_days : ℝ) 
  (h1 : total_days = 6) 
  (h2 : combined_days = 15/4) :
  ∃ (days_a : ℝ), 
    work_rate days_a + (work_rate total_days - work_rate days_a) = work_rate combined_days ∧ 
    days_a = 6 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1083_108318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l1083_108334

-- Statement 1
def K_squared_relation (A B : Type) (K_squared : ℝ) : Prop :=
  ∀ (k₁ k₂ : ℝ), k₁ < k₂ → (K_squared = k₁ → relation A B) → (K_squared = k₂ → relation A B)
where
  relation : Type → Type → Prop := sorry

-- Statement 2
def exponential_regression (c k : ℝ) : Prop :=
  ∃ (y : ℝ → ℝ) (x : ℝ),
    (y = λ x ↦ c * Real.exp (k * x)) ∧
    ((λ x ↦ Real.log (y x)) = λ x ↦ k * x + Real.log c) ∧
    c = Real.exp 4 ∧ k = 0.3

-- Statement 3
def linear_regression (b a x_bar y_bar : ℝ) : Prop :=
  b = 2 ∧ x_bar = 1 ∧ y_bar = 3 → a = 1

theorem all_statements_correct :
  (∃ (A B : Type) (K_squared : ℝ), K_squared_relation A B K_squared) ∧
  (∃ c k : ℝ, exponential_regression c k) ∧
  (∃ b a x_bar y_bar : ℝ, linear_regression b a x_bar y_bar) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l1083_108334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_on_hyperbola_l1083_108363

def possible_points : List (ℕ × ℕ) :=
  [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

def on_hyperbola (point : ℕ × ℕ) : Bool :=
  point.2 = 6 / point.1

def count_on_hyperbola (points : List (ℕ × ℕ)) : ℕ :=
  (points.filter on_hyperbola).length

theorem probability_on_hyperbola :
  (count_on_hyperbola possible_points : ℚ) / possible_points.length = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_on_hyperbola_l1083_108363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1083_108360

theorem function_equality : 
  (∀ x : ℝ, x = (x^3)^(1/3)) ∧ 
  (∃ x : ℝ, x ≠ 1 ∧ (x^2 - 1) / (x - 1) ≠ x + 1) ∧
  (∀ x : ℝ, x ≠ 0 → |x| / x = if x > 0 then 1 else -1) ∧
  (∀ x : ℝ, |x - 1| = |x - 1|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1083_108360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_trigonometry_l1083_108385

noncomputable section

-- Define constants
def m : ℝ := 2 * Real.sin (18 * Real.pi / 180)

-- Define n as a function of m
def n (m : ℝ) : ℝ := 4 - m^2

-- Theorem statement
theorem golden_ratio_trigonometry (m n : ℝ) : 
  m = 2 * Real.sin (18 * Real.pi / 180) →
  n = 4 - m^2 →
  (m + Real.sqrt n) / Real.sin (63 * Real.pi / 180) = 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_trigonometry_l1083_108385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1083_108327

-- Define the equation and its roots
def equation (x θ : ℝ) : Prop :=
  x^2 + x / Real.tan θ - 1 / Real.sin θ = 0

-- Define the circle
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the line passing through two points
def line_through_points (a b x y : ℝ) : Prop :=
  y - a^2 = ((b^2 - a^2) / (b - a)) * (x - a)

-- Main theorem
theorem line_tangent_to_circle (θ a b : ℝ) :
  a ≠ b →
  equation a θ →
  equation b θ →
  ∃ (x y : ℝ), unit_circle x y ∧ line_through_points a b x y ∧
    ∀ (x' y' : ℝ), unit_circle x' y' → line_through_points a b x' y' → (x, y) = (x', y') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1083_108327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1083_108383

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a + 2) ^ x < (a + 2) ^ y

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, (p a ∧ q a → False) ∧ (p a ∨ q a)) → 
  {a : ℝ | a ≤ -1 ∨ a > 1} = Set.univ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1083_108383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1083_108301

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4) + Real.cos (Real.pi / 4 - x)

theorem f_max_value : ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1083_108301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_T_n_l1083_108373

/-- Definition of T_n as the minimum value of the sum -/
noncomputable def T (n : ℕ+) : ℝ :=
  Real.sqrt (n.val ^ 6 + 529)

/-- The main theorem stating that 8 is the unique positive integer n for which T_n is an integer -/
theorem unique_integer_T_n : ∃! (n : ℕ+), ∃ (m : ℕ), T n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_T_n_l1083_108373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_correct_smallest_n_is_correct_no_smaller_n_l1083_108364

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The smallest natural number n such that the number of trailing zeros in (n+20)!
    is exactly 2020 more than the number of trailing zeros in n! -/
def smallestN : ℕ := 5^2017 - 20

theorem smallest_n_correct :
  ∃ n, trailingZeros (n + 20) = trailingZeros n + 2020 ∧
       ∀ m < n, trailingZeros (m + 20) ≠ trailingZeros m + 2020 :=
sorry

theorem smallest_n_is_correct : 
  trailingZeros (smallestN + 20) = trailingZeros smallestN + 2020 :=
sorry

theorem no_smaller_n :
  ∀ m < smallestN, trailingZeros (m + 20) ≠ trailingZeros m + 2020 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_correct_smallest_n_is_correct_no_smaller_n_l1083_108364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_condition_l1083_108303

/-- The area of a regular polygon with n sides inscribed in a circle of radius R --/
noncomputable def regularPolygonArea (n : ℕ) (R : ℝ) : ℝ :=
  1/2 * (n : ℝ) * R^2 * Real.sin (2 * Real.pi / (n : ℝ))

/-- Theorem: If the area of a regular polygon with n sides inscribed in a circle of radius R is 4R^2, then n = 24 --/
theorem regular_polygon_area_condition (n : ℕ) (R : ℝ) (h : R > 0) :
  regularPolygonArea n R = 4 * R^2 → n = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_condition_l1083_108303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_park_area_l1083_108369

/-- Represents the scale of the map in miles per inch -/
noncomputable def scale : ℝ := 500 / 2

/-- The length of the short diagonal on the map in inches -/
def map_diagonal : ℝ := 10

/-- The actual length of the short diagonal in miles -/
noncomputable def actual_diagonal : ℝ := map_diagonal * scale

/-- The area of the rhombus-shaped park in square miles -/
noncomputable def park_area : ℝ := (Real.sqrt 3 / 2) * actual_diagonal^2

theorem rhombus_park_area : park_area = 3125000 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_park_area_l1083_108369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_second_draw_is_one_third_prob_same_color_is_seven_fifteenths_n_is_five_when_prob_two_red_is_one_twentyfirst_l1083_108370

-- Define the type for balls
inductive Ball : Type where
  | Red : Ball
  | Green : Ball
deriving DecidableEq

-- Define the bag contents
def bag : Multiset Ball :=
  Multiset.replicate 2 Ball.Red + Multiset.replicate 4 Ball.Green

-- Define the probability of drawing a red ball on the second draw
def prob_red_second_draw (b : Multiset Ball) : ℚ :=
  let total := Multiset.card b
  let red := Multiset.count Ball.Red b
  (red * (total - 1)) / (total * (total - 1))

-- Define the probability of drawing two balls of the same color
def prob_same_color (b : Multiset Ball) : ℚ :=
  let total := Multiset.card b
  let red := Multiset.count Ball.Red b
  let green := Multiset.count Ball.Green b
  (red * (red - 1) + green * (green - 1)) / (total * (total - 1))

-- Define the probability of drawing two red balls
def prob_two_red (red : ℕ) (green : ℕ) : ℚ :=
  let total := red + green
  (red * (red - 1)) / (total * (total - 1))

theorem prob_red_second_draw_is_one_third :
  prob_red_second_draw bag = 1/3 := by sorry

theorem prob_same_color_is_seven_fifteenths :
  prob_same_color bag = 7/15 := by sorry

theorem n_is_five_when_prob_two_red_is_one_twentyfirst :
  ∀ n : ℕ, prob_two_red 2 n = 1/21 → n = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_second_draw_is_one_third_prob_same_color_is_seven_fifteenths_n_is_five_when_prob_two_red_is_one_twentyfirst_l1083_108370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_people_example_l1083_108372

/-- Given a number of boats and total people, calculates the average number of people per boat. -/
noncomputable def average_people_per_boat (num_boats : ℝ) (total_people : ℝ) : ℝ :=
  total_people / num_boats

/-- Theorem stating that with 3.0 boats and 5.0 people, the average is 5.0 / 3.0 people per boat. -/
theorem average_people_example : average_people_per_boat 3.0 5.0 = 5.0 / 3.0 := by
  -- Unfold the definition of average_people_per_boat
  unfold average_people_per_boat
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_people_example_l1083_108372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_16_l1083_108302

-- Define the line l in polar form
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi/3) = 6

-- Define the circle C in parametric form
def circle_param (x y θ : ℝ) : Prop := x = 10 * Real.cos θ ∧ y = 10 * Real.sin θ

-- Define the line l in rectangular form
def line_rect (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 12 = 0

-- Define the circle C in standard form
def circle_standard (x y : ℝ) : Prop := x^2 + y^2 = 100

-- Theorem statement
theorem chord_length_is_16 :
  ∀ x y θ ρ,
  line_polar ρ θ →
  circle_param x y θ →
  line_rect x y →
  circle_standard x y →
  ∃ chord_length, chord_length = 16 ∧ 
    chord_length^2 = 4 * (100 - 36) := by
  sorry

#check chord_length_is_16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_16_l1083_108302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_negative_two_l1083_108321

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the invertibility of f
axiom f_invertible : Function.Injective f

-- Define the conditions
axiom condition1 : f 0 = 2
axiom condition2 : f 2 = 4

-- State the theorem
theorem a_minus_b_equals_negative_two : 
  ∃ (a b : ℝ), f a = b ∧ f b = 4 ∧ a - b = -2 :=
by
  -- Provide a and b
  use 0, 2
  -- Prove the conditions
  constructor
  · exact condition1
  constructor
  · exact condition2
  · -- Prove a - b = -2
    norm_num

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_negative_two_l1083_108321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisible_by_15_l1083_108320

/-- The largest prime with 2023 digits -/
noncomputable def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2023 digits -/
axiom p_digits : (Nat.digits 10 p).length = 2023

/-- p is the largest prime with 2023 digits -/
axiom p_largest : ∀ q : ℕ, Nat.Prime q → (Nat.digits 10 q).length = 2023 → q ≤ p

theorem smallest_k_divisible_by_15 : ∃ m : ℕ, p^2 - 1 = 15 * m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisible_by_15_l1083_108320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_ray_trigonometric_equality_l1083_108382

/-- Given that the terminal side of angle α lies on the ray 3x + 4y = 0 (x < 0),
    prove that (sin(π-α)cos(3π+α)tanα) / (cos(-α)sin(π+α)) = -3/4 -/
theorem angle_on_ray_trigonometric_equality (α : ℝ) 
    (h : ∃ (x y : ℝ), x < 0 ∧ 3*x + 4*y = 0 ∧ (Real.cos α = x / Real.sqrt (x^2 + y^2)) ∧ (Real.sin α = y / Real.sqrt (x^2 + y^2))) :
    (Real.sin (π - α) * Real.cos (3*π + α) * Real.tan α) / (Real.cos (-α) * Real.sin (π + α)) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_ray_trigonometric_equality_l1083_108382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1083_108335

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * x * Real.log x + b

-- Define the derivative of f(x)
noncomputable def f_deriv (a x : ℝ) : ℝ := a * (1 + Real.log x)

-- Theorem statement
theorem tangent_line_sum (a b : ℝ) :
  (f_deriv a 1 = 2) →    -- Slope of tangent line at x=1 is 2
  (f a b 1 = 2) →        -- f(1) = 2 (y-intercept of tangent line)
  a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1083_108335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_range_of_a_l1083_108322

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + x + a = 0}

-- Define the range of a
def range_of_a : Set ℝ := {-6} ∪ Set.Ioi (1/4)

-- Theorem statement
theorem subset_implies_range_of_a :
  ∀ a : ℝ, (B a ⊆ A) ↔ a ∈ range_of_a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_range_of_a_l1083_108322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_point_d_l1083_108366

def A : ℂ := 2 + Complex.I
def B : ℂ := 4 + 3*Complex.I
def C : ℂ := 3 + 5*Complex.I

def is_parallelogram (a b c d : ℂ) : Prop :=
  (a + c = b + d) ∨ (a + d = b + c) ∨ (a + b = c + d)

theorem parallelogram_point_d :
  ∃ (D : ℂ), is_parallelogram A B C D ∧ (D = 3 - Complex.I ∨ D = 1 + 3*Complex.I ∨ D = 5 + 7*Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_point_d_l1083_108366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1083_108398

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1083_108398
