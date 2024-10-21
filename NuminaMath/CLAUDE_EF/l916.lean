import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_graph_representation_l916_91650

/-- Represents the movement pattern of a participant in the race -/
inductive MovementPattern
  | ConstantSpeed
  | SprintRestPattern

/-- Represents a participant in the race -/
structure Participant where
  name : String
  movement : MovementPattern

/-- Represents the graph of a participant's distance over time -/
inductive GraphType
  | SteadyRise
  | PausedRise

/-- Represents the complete race scenario -/
structure RaceScenario where
  rabbit : Participant
  snail : Participant

/-- Determines the correct graph representation for a given participant -/
def correct_graph_for_participant (p : Participant) : GraphType :=
  match p.movement with
  | MovementPattern.ConstantSpeed => GraphType.SteadyRise
  | MovementPattern.SprintRestPattern => GraphType.PausedRise

/-- Represents the correct graph representation for the entire race scenario -/
def correct_graph_representation (scenario : RaceScenario) : Prop :=
  correct_graph_for_participant scenario.snail = GraphType.SteadyRise ∧
  correct_graph_for_participant scenario.rabbit = GraphType.PausedRise

/-- Theorem stating the correct graph representation for the given race scenario -/
theorem race_graph_representation (scenario : RaceScenario) :
  scenario.rabbit.name = "rabbit" ∧
  scenario.rabbit.movement = MovementPattern.SprintRestPattern ∧
  scenario.snail.name = "snail" ∧
  scenario.snail.movement = MovementPattern.ConstantSpeed →
  correct_graph_representation scenario :=
by
  intro h
  unfold correct_graph_representation
  apply And.intro
  · simp [correct_graph_for_participant, h.right.right]
  · simp [correct_graph_for_participant, h.right.left]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_graph_representation_l916_91650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_implies_b_equals_2008_l916_91637

theorem three_roots_implies_b_equals_2008 (a b : ℝ) (h1 : a ≠ 0) :
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, abs (abs (x - a) - b) = 2008) →
  b = 2008 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_implies_b_equals_2008_l916_91637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_leg_is_5_sqrt_3_l916_91679

/-- A right triangle with one leg of 15 inches and the angle opposite that leg being 60° -/
structure SpecialTriangle where
  /-- The length of the first leg -/
  leg1 : ℝ
  /-- The angle opposite the first leg in radians -/
  angle1 : ℝ
  /-- The angle opposite the second leg in radians -/
  angle2 : ℝ
  /-- The triangle is right-angled -/
  is_right : angle1 + angle2 + π/2 = π
  /-- The first leg is 15 inches long -/
  leg1_length : leg1 = 15
  /-- The angle opposite the first leg is 60° (π/3 radians) -/
  angle1_measure : angle1 = π/3

/-- The length of the other leg in the special triangle -/
noncomputable def other_leg_length (t : SpecialTriangle) : ℝ := 5 * Real.sqrt 3

/-- Theorem stating that the length of the other leg is 5√3 inches -/
theorem other_leg_is_5_sqrt_3 (t : SpecialTriangle) : 
  other_leg_length t = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_leg_is_5_sqrt_3_l916_91679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l916_91628

/-- The main theorem to prove -/
theorem lcm_problem : Nat.lcm (Nat.lcm (Nat.lcm 12 16) (Nat.lcm 18 24)) (Nat.lcm (Nat.lcm 12 16) (Nat.lcm 18 24)) = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l916_91628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_selling_price_theorem_l916_91634

/-- The selling price of a stock before brokerage, given the cash realized after brokerage and the brokerage rate. -/
noncomputable def selling_price_before_brokerage (cash_realized : ℝ) (brokerage_rate : ℝ) : ℝ :=
  cash_realized / (1 - brokerage_rate)

/-- Theorem stating that given a cash realized amount of 120.50 after a 1/4% brokerage,
    the selling price before brokerage is approximately 120.90. -/
theorem stock_selling_price_theorem :
  let cash_realized : ℝ := 120.50
  let brokerage_rate : ℝ := 1 / 400
  let selling_price := selling_price_before_brokerage cash_realized brokerage_rate
  |selling_price - 120.90| < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_selling_price_theorem_l916_91634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_curve_tangent_line_l916_91606

-- Define the quadratic curve
def curve (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 4

-- Define the derivative of the curve
def curve_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

-- Define the slope of line K
def slope_K (b : ℝ) : ℝ := 2 * (curve_derivative b 1)

-- Define the slope of line y = x + c
def slope_parallel : ℝ := 1

theorem quadratic_curve_tangent_line (b : ℝ) :
  (∃ K : ℝ → ℝ, 
    (K 0 = 0) ∧  -- K passes through origin
    (∀ x, K x = (slope_K b) * x) ∧  -- K has slope twice that of tangent at x = 1
    (slope_K b = slope_parallel))  -- K is parallel to y = x + c
  → 
  (b = -3/2 ∧ ∃ K : ℝ → ℝ, ∀ x, K x = x) :=
by
  intro h
  sorry

#check quadratic_curve_tangent_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_curve_tangent_line_l916_91606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l916_91692

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 14 * Real.sqrt x - 15 * x^(1/3) + 2

-- Define the point of tangency
def x₀ : ℝ := 1

-- Define the proposed tangent line
def tangent_line (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem tangent_line_at_x₀ :
  (deriv f x₀) * (x - x₀) + f x₀ = tangent_line x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l916_91692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_deviations_l916_91696

/-- Represents the standard deviation of exam scores. -/
noncomputable def standard_deviation (mean : ℝ) (score_above : ℝ) (deviations_above : ℝ) : ℝ :=
  (score_above - mean) / deviations_above

/-- Calculates the number of standard deviations a score is below the mean. -/
noncomputable def deviations_below (mean : ℝ) (score : ℝ) (std_dev : ℝ) : ℝ :=
  (mean - score) / std_dev

theorem exam_score_deviations (mean : ℝ) (score_above : ℝ) (score_below : ℝ) 
    (deviations_above : ℝ) (h_mean : mean = 76) (h_above : score_above = 100) 
    (h_below : score_below = 60) (h_dev_above : deviations_above = 3) : 
    deviations_below mean score_below (standard_deviation mean score_above deviations_above) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_deviations_l916_91696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l916_91661

theorem power_equation_solution :
  ∃! x : ℝ, (1/5 : ℝ)^x * (1/4 : ℝ)^2 = 1/(10^4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l916_91661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersections_l916_91600

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

-- Define the polar equation for C₁
noncomputable def C₁_polar (θ : ℝ) : ℝ :=
  4 * Real.sin θ

-- Define the polar equation for C₂
noncomputable def C₂_polar (θ : ℝ) : ℝ :=
  8 * Real.sin θ

-- Theorem statement
theorem distance_between_intersections :
  let A := C₁_polar (π/3)
  let B := C₂_polar (π/3)
  |A - B| = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersections_l916_91600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l916_91678

variable (a b c x x₀ x₁ x₂ : ℝ)

def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_function_properties
  (ha : a > 0)
  (hroots : f a b c x₁ - x₁ = 0 ∧ f a b c x₂ - x₂ = 0)
  (horder : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a)
  (hsymmetry : ∀ x, f a b c (2*x₀ - x) = f a b c x) :
  (∀ x, 0 < x ∧ x < x₁ → x < f a b c x ∧ f a b c x < x₁) ∧
  x₀ < x₁/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l916_91678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comets_tail_area_is_8pi_minus_9_l916_91684

noncomputable section

/-- The area of a quarter circle with radius r -/
def quarterCircleArea (r : ℝ) : ℝ := Real.pi * r^2 / 4

/-- The area of a quarter square with side s -/
def quarterSquareArea (s : ℝ) : ℝ := s^2 / 4

/-- The comet's tail region area -/
def cometsTailArea : ℝ :=
  quarterCircleArea 6 - (quarterCircleArea 2 + quarterSquareArea 6)

theorem comets_tail_area_is_8pi_minus_9 :
  cometsTailArea = 8 * Real.pi - 9 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comets_tail_area_is_8pi_minus_9_l916_91684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_two_l916_91690

theorem cube_root_sum_equals_two (y : ℝ) (hy : y > 0) 
  (h : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
  y^6 = 116/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_two_l916_91690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inclination_l916_91626

/-- The angle of inclination of the tangent line to the curve y = x³ - 2x + m at x = 1 is 45°. -/
theorem tangent_line_inclination (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 2*x + m
  let f' : ℝ → ℝ := λ x => 3*x^2 - 2
  let slope : ℝ := f' 1
  Real.arctan slope = π/4 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inclination_l916_91626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parts_area_not_all_less_than_four_ninths_l916_91602

/-- An isosceles right triangle with a point on its hypotenuse and perpendiculars to the legs -/
structure IsoscelesRightTriangleWithPoint where
  /-- Side length of the equal legs -/
  a : ℝ
  /-- Point P divides the hypotenuse into segments of length x and y -/
  x : ℝ
  y : ℝ
  /-- Conditions -/
  h_positive : 0 < a
  h_hypotenuse : x + y = a * Real.sqrt 2

/-- The areas of the three parts created by the perpendiculars -/
noncomputable def areas (t : IsoscelesRightTriangleWithPoint) : Fin 3 → ℝ
| 0 => t.x * t.y / (2 * t.a)  -- Area of triangle AMP
| 1 => t.x * t.y / (2 * t.a)  -- Area of triangle BNP
| 2 => t.x * t.y / t.a^2      -- Area of rectangle PMNQ

/-- The original triangle's area -/
noncomputable def originalArea (t : IsoscelesRightTriangleWithPoint) : ℝ := t.a^2 / 2

/-- Theorem: It's impossible for each part to have an area less than 4/9 of the original triangle's area -/
theorem parts_area_not_all_less_than_four_ninths (t : IsoscelesRightTriangleWithPoint) :
  ¬(∀ i : Fin 3, areas t i < (4/9) * originalArea t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parts_area_not_all_less_than_four_ninths_l916_91602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_zeros_l916_91656

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x) - 1

def has_exactly_three_zeros (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ b ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧
    ∀ x, a ≤ x ∧ x ≤ b ∧ f x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem cosine_zeros (ω : ℝ) :
  ω > 0 → (has_exactly_three_zeros (f ω) 0 (2 * Real.pi) ↔ 2 ≤ ω ∧ ω < 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_zeros_l916_91656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_when_slope_negative_one_line_equation_when_triangle_area_minimized_l916_91697

noncomputable section

-- Define the line l
def line_l (k : ℝ) : ℝ → ℝ := λ x ↦ k * (x - 2) + 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the x-intercept of line l
noncomputable def x_intercept (k : ℝ) : ℝ := 2 - 1/k

-- Define the y-intercept of line l
noncomputable def y_intercept (k : ℝ) : ℝ := 1 - 2*k

-- Define the area of triangle AOB
noncomputable def area_triangle (k : ℝ) : ℝ := (1/2) * (x_intercept k) * (y_intercept k)

-- Define the area of the circumcircle of triangle AOB
noncomputable def area_circumcircle (k : ℝ) : ℝ := Real.pi * ((x_intercept k)^2 + (y_intercept k)^2) / 4

theorem circumcircle_area_when_slope_negative_one :
  area_circumcircle (-1) = 9*Real.pi/2 := by
  sorry

theorem line_equation_when_triangle_area_minimized :
  ∃ k : ℝ, k < 0 ∧ 
  (∀ m : ℝ, m < 0 → area_triangle k ≤ area_triangle m) ∧
  (line_l k) = (λ x ↦ -1/2 * x + 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_when_slope_negative_one_line_equation_when_triangle_area_minimized_l916_91697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_one_l916_91601

/-- The circle represented by curve C1 -/
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The line represented by curve C2 -/
def line_C2 (x y : ℝ) : Prop := x + y + 2 * Real.sqrt 2 - 1 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 0)

/-- The distance from a point to a line -/
noncomputable def distPointToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem shortest_distance_is_one :
  ∃ (p q : ℝ × ℝ), circle_C1 p.1 p.2 ∧ line_C2 q.1 q.2 ∧
    (∀ (p' q' : ℝ × ℝ), circle_C1 p'.1 p'.2 → line_C2 q'.1 q'.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2)) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_one_l916_91601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_pyramid_volume_l916_91627

/-- A regular quadrilateral pyramid with base side length a -/
structure RegularQuadPyramid where
  a : ℝ
  h : a > 0

/-- The volume of a regular quadrilateral pyramid -/
noncomputable def volume (p : RegularQuadPyramid) : ℝ :=
  (p.a^3 / 6) * Real.sqrt (Real.sqrt 5 + 1)

/-- The plane angle at the vertex of the pyramid -/
noncomputable def plane_angle_at_vertex (p : RegularQuadPyramid) : ℝ := sorry

/-- The angle between the slant edge and the base of the pyramid -/
noncomputable def angle_slant_edge_with_base (p : RegularQuadPyramid) : ℝ := sorry

/-- The theorem stating the volume of a regular quadrilateral pyramid
    with the given condition on angles -/
theorem regular_quad_pyramid_volume
  (p : RegularQuadPyramid)
  (h : plane_angle_at_vertex p = angle_slant_edge_with_base p) :
  volume p = (p.a^3 / 6) * Real.sqrt (Real.sqrt 5 + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_pyramid_volume_l916_91627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l916_91694

-- Define IsTriangle and IsOpposite as needed
def IsTriangle (A B C : Real) : Prop := sorry
def IsOpposite (side angle1 angle2 angle3 : Real) : Prop := sorry

theorem triangle_cosine_proof (A B C : Real) (a b c : Real) : 
  -- Triangle ABC exists
  IsTriangle A B C →
  -- a, b, c are sides opposite to angles A, B, C
  IsOpposite a A B C →
  IsOpposite b B A C →
  IsOpposite c C A B →
  -- sin A, sin B, sin C form an arithmetic sequence
  2 * Real.sin B = Real.sin A + Real.sin C →
  -- a = 2c
  a = 2 * c →
  -- Conclusion: cos A = -1/4
  Real.cos A = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l916_91694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repair_cost_calculation_l916_91622

/-- Proves that the repair cost is $200 given the initial cost, selling price, and gain percent --/
theorem repair_cost_calculation (initial_cost selling_price : ℕ) (gain_percent : ℚ) :
  initial_cost = 800 →
  selling_price = 1200 →
  gain_percent = 20 / 100 →
  ∃ (repair_cost : ℕ),
    repair_cost = 200 ∧
    selling_price = initial_cost + repair_cost + Int.floor (gain_percent * (initial_cost + repair_cost)) :=
by
  sorry

#check repair_cost_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repair_cost_calculation_l916_91622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_income_greater_than_average_l916_91698

-- Define incomes as real numbers
noncomputable def income_C : ℝ := 100  -- We assume C's income as base
noncomputable def income_A : ℝ := income_C * 1.2
noncomputable def income_B : ℝ := income_A * 1.25
noncomputable def income_D : ℝ := income_B * 0.85
noncomputable def income_E : ℝ := income_C * 1.1

-- Calculate average income of A, C, D, and E
noncomputable def avg_income : ℝ := (income_A + income_C + income_D + income_E) / 4

-- Calculate the percentage difference
noncomputable def percentage_difference : ℝ := (income_B - avg_income) / avg_income * 100

-- Theorem statement
theorem b_income_greater_than_average : 
  31.14 < percentage_difference ∧ percentage_difference < 31.16 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_income_greater_than_average_l916_91698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l916_91642

noncomputable def a (n : ℕ) : ℝ := n * (n + 1)

noncomputable def S (n : ℕ) : ℝ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def b (n : ℕ) : ℝ := Real.sqrt (a n)

noncomputable def c (n : ℕ) : ℝ := b (n + 1) - b n

theorem sequence_properties :
  (a 2 = 6) ∧
  (∀ n : ℕ, 3 * S n = (n + 1) * a n + n * (n + 1)) ∧
  (∀ n : ℕ, a n = n * (n + 1)) ∧
  (∀ n : ℕ, c (n + 1) < c n) ∧
  (∀ n : ℕ, 1 < c n ∧ c n ≤ Real.sqrt 6 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l916_91642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_below_x_axis_is_one_third_l916_91617

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram -/
noncomputable def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram below the x-axis -/
noncomputable def areaBelowXAxis (p : Parallelogram) : ℝ := sorry

/-- The probability of a point in the parallelogram being below the x-axis -/
noncomputable def probBelowXAxis (p : Parallelogram) : ℝ :=
  areaBelowXAxis p / parallelogramArea p

/-- The specific parallelogram PQRS from the problem -/
def PQRS : Parallelogram :=
  { P := { x := 4, y := 4 },
    Q := { x := -2, y := -2 },
    R := { x := -8, y := -2 },
    S := { x := 2, y := 4 } }

theorem prob_below_x_axis_is_one_third :
  probBelowXAxis PQRS = 1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_below_x_axis_is_one_third_l916_91617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_minimum_value_l916_91667

-- Define the function y(x)
noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)

-- State the theorem
theorem y_minimum_value :
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → y x ≥ 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_minimum_value_l916_91667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l916_91680

theorem tan_sum_problem (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 15)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 20) : 
  Real.tan (x + y) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l916_91680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_set_size_l916_91633

theorem first_set_size (pass_rate_all : ℚ) 
  (h_pass_rate : pass_rate_all = 88666666666666667 / 1000000000000000) : ℕ :=
  let x : ℕ := 40
  let first_set : ℕ := x
  let second_set : ℕ := 50
  let third_set : ℕ := 60
  let pass_rate_first : ℚ := 1
  let pass_rate_second : ℚ := 9/10
  let pass_rate_third : ℚ := 4/5
  let total_students : ℕ := first_set + second_set + third_set
  let total_passed : ℚ := first_set * pass_rate_first + 
                          second_set * pass_rate_second + 
                          third_set * pass_rate_third
  have h : pass_rate_all = total_passed / total_students := by sorry
  x

#check first_set_size

/- Proof goes here -/
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_set_size_l916_91633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l916_91659

theorem polynomial_identity (P : Polynomial ℝ) 
  (h1 : P.eval 0 = 0) 
  (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) : 
  P = Polynomial.X :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l916_91659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_is_24_l916_91677

/-- Represents the speed of a boat in still water and the speed of a stream. -/
structure BoatProblem where
  boat_speed : ℝ  -- Speed of the boat in still water
  stream_speed : ℝ  -- Speed of the stream

/-- The conditions of the problem are satisfied. -/
def satisfies_conditions (p : BoatProblem) : Prop :=
  p.stream_speed = 8 ∧
  2 * (p.boat_speed - p.stream_speed) = p.boat_speed + p.stream_speed

/-- The theorem stating that if the conditions are satisfied, the boat's speed in still water is 24 kmph. -/
theorem boat_speed_is_24 (p : BoatProblem) :
  satisfies_conditions p → p.boat_speed = 24 := by
  sorry

#check boat_speed_is_24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_is_24_l916_91677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l916_91665

/-- The area of an equilateral triangle with side length 2 is √3 -/
theorem equilateral_triangle_area : 
  let side_length : ℝ := 2
  let triangle_area := (Real.sqrt 3 / 4) * side_length^2
  triangle_area = Real.sqrt 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l916_91665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_at_start_l916_91682

/-- Represents a circular track divided into quarters -/
structure CircularTrack where
  circumference : ℚ
  quarters : Fin 4 → String

/-- Represents a runner on the track -/
structure Runner where
  distance_run : ℚ

/-- Determines the quarter where the runner stops -/
def stop_quarter (track : CircularTrack) (runner : Runner) : String :=
  if runner.distance_run % track.circumference = 0 then track.quarters 0 else "Not at a quarter boundary"

theorem runner_stops_at_start (track : CircularTrack) (runner : Runner) 
  (h1 : track.circumference > 0)
  (h2 : runner.distance_run > 0)
  (h3 : runner.distance_run % track.circumference = 0) :
  stop_quarter track runner = track.quarters 0 := by
  unfold stop_quarter
  rw [if_pos h3]

#check runner_stops_at_start

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_at_start_l916_91682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waterloo_street_problem_l916_91688

def is_arithmetic_sequence (s : List Nat) : Prop :=
  s.length > 1 ∧ ∃ d, ∀ i, i + 1 < s.length → s[i + 1]! - s[i]! = d

def has_no_repeated_digits (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem waterloo_street_problem :
  ∃ (house_numbers : List Nat),
    house_numbers.length = 14 ∧
    (∀ n ∈ house_numbers, 500 ≤ n ∧ n ≤ 599) ∧
    is_arithmetic_sequence house_numbers ∧
    (house_numbers.filter (λ n => n % 2 = 0)).length = 7 ∧
    (house_numbers.filter (λ n => n % 2 = 1)).length = 7 ∧
    555 ∈ house_numbers ∧
    (∀ n ∈ house_numbers, n ≠ 555 → has_no_repeated_digits n) →
    house_numbers.minimum? = some 506 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waterloo_street_problem_l916_91688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_negative_values_l916_91673

-- Define the property of the function f
def has_property (f : ℚ → ℚ) :=
  (∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b) ∧
  (∀ p : ℕ, Nat.Prime p → f p = p)

-- State the theorem
theorem function_property_implies_negative_values
  (f : ℚ → ℚ) (h : has_property f) :
  f (5/12) < 0 ∧ f (8/15) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_negative_values_l916_91673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_number_is_803_l916_91619

def sequenceNumbers : List ℕ := [11, 23, 47, 83, 131, 191, 263, 347, 443]

def differences (seq : List ℕ) : List ℕ :=
  List.zipWith (·-·) (seq.tail) seq

def next_difference (diffs : List ℕ) : ℕ :=
  diffs.getLast?.getD 0 + 12

def next_number (seq : List ℕ) (diffs : List ℕ) : ℕ :=
  seq.getLast?.getD 0 + next_difference diffs

theorem tenth_number_is_803 :
  next_number (sequenceNumbers ++ [551, 671]) (differences (sequenceNumbers ++ [551, 671])) = 803 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_number_is_803_l916_91619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l916_91638

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 - Real.sqrt 2 / 2 * t, 2 + Real.sqrt 2 / 2 * t)

-- Define the circle C in polar form
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- State the theorem
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧
  ∀ (θ : ℝ) (t : ℝ),
    let (x, y) := line_l t
    let ρ := circle_C θ
    (x - ρ * Real.cos θ) ^ 2 + (y - ρ * Real.sin θ) ^ 2 ≥ d ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l916_91638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_14_l916_91689

def divisors_of_252 : List Nat :=
  [2, 3, 4, 6, 7, 9, 14, 18, 21, 28, 36, 42, 63, 84, 126, 252]

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i j, (i + 1) % arr.length = j % arr.length →
    ∃ k > 1, k ∣ arr[i]! ∧ k ∣ arr[j]!

theorem adjacent_sum_14 (arr : List Nat) :
  arr.toFinset = divisors_of_252.toFinset →
  is_valid_arrangement arr →
  14 ∈ arr →
  ∃ i j, arr[i]! = 14 ∧
         (i + 1) % arr.length = j % arr.length ∧
         (j + 1) % arr.length = ((i - 1 + arr.length) % arr.length) ∧
         arr[j]! + arr[(i - 1 + arr.length) % arr.length]! = 70 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_14_l916_91689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l916_91699

/-- Maria's current age -/
def M : ℕ := sorry

/-- Carl's current age -/
def C : ℕ := sorry

/-- Maria is twelve years older than Carl -/
axiom age_difference : M = C + 12

/-- Ten years from now, Maria will be three times as old as Carl was six years ago -/
axiom future_relation : M + 10 = 3 * (C - 6)

/-- The sum of Maria and Carl's current ages is 52 -/
theorem sum_of_ages : M + C = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l916_91699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_equivalence_l916_91644

/-- Conversion from spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

/-- Theorem stating the equivalence of the given spherical and rectangular coordinates -/
theorem spherical_to_rectangular_equivalence :
  spherical_to_rectangular 4 (π/6) (π/4) = (2*Real.sqrt 3, Real.sqrt 2, 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_equivalence_l916_91644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_open_1_closed_2_l916_91674

-- Define set A
def A : Set ℝ := {x | Real.rpow 2 x ≤ 4}

-- Define set B (domain of log(x - 1))
def B : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem A_intersect_B_equals_open_1_closed_2 : A ∩ B = Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_open_1_closed_2_l916_91674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_l916_91668

theorem determinant_zero_implies_sum (x y : ℝ) : 
  x ≠ y →
  Matrix.det !![2, 5, 10; 4, x, y; 4, y, x] = 0 →
  x + y = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_l916_91668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l916_91643

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  a > 0 → b > 0 → c > 0 →
  Real.sin C / (2 * Real.sin A - Real.sin C) = (b^2 - a^2 - c^2) / (c^2 - a^2 - b^2) →
  Real.sin (2 * A + π / 6) + 1 / 2 = (1 + Real.sqrt 3) / 2 →
  C = 7 * π / 12 ∨ C = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l916_91643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l916_91653

/-- The asymptotic line equation of the hyperbola 3x^2 - y^2 = 1 -/
theorem hyperbola_asymptote (x y : ℝ) :
  (3 * x^2 - y^2 = 1) → (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l916_91653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_l916_91603

theorem power_of_two_divisibility (k : ℕ) :
  (∀ n : ℕ, n > 0 → ∃ m : ℕ, 2^((k-1)*n+1) * (Nat.factorial (k*n)) / (Nat.factorial n) = m) ↔
  ∃ a : ℕ, k = 2^a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_l916_91603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_rate_l916_91608

/-- The time it takes for worker a to complete the work alone -/
noncomputable def time_a : ℝ := 3

/-- The time it takes for workers b and c together to complete the work -/
noncomputable def time_bc : ℝ := 2

/-- The time it takes for workers a and b together to complete the work -/
noncomputable def time_ab : ℝ := 2

/-- The total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- The work rate of worker c -/
noncomputable def work_rate_c : ℝ := 1 / 3

theorem c_work_rate :
  work_rate_c = total_work / time_a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_rate_l916_91608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_three_consecutive_not_square_sum_squares_six_consecutive_not_square_sum_squares_eleven_consecutive_is_square_l916_91646

-- Part 1: Sum of squares of 3 consecutive integers
theorem sum_squares_three_consecutive_not_square (n : ℕ) : 
  ¬ ∃ m : ℕ, 3 * n^2 + 2 = m^2 := by sorry

-- Part 2: Sum of squares of 6 consecutive integers
theorem sum_squares_six_consecutive_not_square (n : ℤ) : 
  ¬ ∃ m : ℤ, 6 * n^2 + 6 * n + 19 = m^2 := by sorry

-- Part 3: Sum of squares of 11 consecutive integers starting from 18
def sum_squares_from_18_to_28 : ℕ := 
  List.range 11
  |>.map (fun i => (i + 18)^2)
  |>.sum

theorem sum_squares_eleven_consecutive_is_square : 
  sum_squares_from_18_to_28 = 77^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_three_consecutive_not_square_sum_squares_six_consecutive_not_square_sum_squares_eleven_consecutive_is_square_l916_91646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l916_91648

theorem recurring_decimal_to_fraction : (3 / 10 : ℚ) + (5 / 11 : ℚ) = 83 / 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l916_91648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_first_four_terms_sum_l916_91687

theorem geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) (h : r ≠ 1) :
  (Finset.range n).sum (λ i => a * r ^ i) = a * (1 - r^n) / (1 - r) :=
by sorry

theorem first_four_terms_sum :
  let a : ℚ := 1
  let r : ℚ := 1/3
  let n : ℕ := 4
  (Finset.range n).sum (λ i => a * r^i) = 40/27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_first_four_terms_sum_l916_91687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_of_system_l916_91630

/-- The system of equations -/
noncomputable def system (x y : ℝ) : Prop :=
  x^3 + 3*y^3 = 11 ∧ x^2*y + x*y^2 = 6

/-- The ratio function -/
noncomputable def ratio (x y : ℝ) : ℝ := x / y

/-- The theorem statement -/
theorem min_ratio_of_system :
  ∃ (x₀ y₀ : ℝ), system x₀ y₀ ∧
  ∀ (x y : ℝ), system x y → ratio x y ≥ (-1 - Real.sqrt 217) / 12 := by
  sorry

#check min_ratio_of_system

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_of_system_l916_91630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_is_150_l916_91664

-- Define the side lengths of the squares
noncomputable def small_side : ℝ := 2
noncomputable def medium_side : ℝ := 4
noncomputable def large_side : ℝ := 8

-- Define the number of medium squares
def num_medium_squares : ℕ := 3

-- Define the areas of different parts
noncomputable def area_largest_square : ℝ := large_side ^ 2
noncomputable def area_medium_squares : ℝ := num_medium_squares * medium_side ^ 2
noncomputable def area_smallest_square : ℝ := small_side ^ 2
noncomputable def area_term1 : ℝ := 4 * 4 / 2 * 2
noncomputable def area_term2 : ℝ := 2 * 2 / 2
noncomputable def area_term3 : ℝ := (large_side + small_side) * 2 / 2 * 2

-- Theorem statement
theorem total_area_is_150 :
  area_largest_square + area_medium_squares + area_smallest_square + area_term1 + area_term2 + area_term3 = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_is_150_l916_91664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sufficient_not_necessary_l916_91649

/-- The circle C centered at (3,0) with radius 3 -/
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + p.2^2 = 9}

/-- The line l passing through (0,5) with slope k -/
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + 5}

/-- The condition for a line to be tangent to the circle -/
def is_tangent (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∩ circle_C ∧ ∀ q : ℝ × ℝ, q ∈ l ∩ circle_C → q = p

/-- The theorem stating that slope -8/15 is sufficient but not necessary for tangency -/
theorem slope_sufficient_not_necessary :
  (is_tangent (line_l (-8/15))) ∧
  ¬(∀ k : ℝ, is_tangent (line_l k) → k = -8/15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sufficient_not_necessary_l916_91649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_and_inequality_l916_91681

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
noncomputable def g (c d x : ℝ) : ℝ := Real.exp x * (c*x + d)

-- Define the theorem
theorem curve_properties_and_inequality 
  (a b c d : ℝ) : 
  (f a b 0 = 2) →  -- f passes through (0, 2)
  (g c d 0 = 2) →  -- g passes through (0, 2)
  ((deriv (f a b)) 0 = 4) →  -- f has slope 4 at x = 0
  ((deriv (g c d)) 0 = 4) →  -- g has slope 4 at x = 0
  (a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∧  -- Part 1 of the conclusion
  (∀ k, (∀ x ≥ -2, f a b x ≤ k * g c d x) ↔ (1 ≤ k ∧ k ≤ Real.exp 2)) -- Part 2 of the conclusion
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_and_inequality_l916_91681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_a_ne_neg_two_l916_91676

/-- Two lines in ℝ³ represented by their parametric equations -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines in ℝ³ are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, 
    (l1.point.1 + t * l1.direction.1 ≠ l2.point.1 + u * l2.direction.1) ∨
    (l1.point.2.1 + t * l1.direction.2.1 ≠ l2.point.2.1 + u * l2.direction.2.1) ∨
    (l1.point.2.2 + t * l1.direction.2.2 ≠ l2.point.2.2 + u * l2.direction.2.2)

theorem lines_skew_iff_a_ne_neg_two (a : ℝ) : 
  are_skew 
    ⟨(1, 2, a), (3, 4, 5)⟩ 
    ⟨(5, 3, 1), (6, 3, 2)⟩ 
  ↔ a ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_a_ne_neg_two_l916_91676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_inclination_l916_91636

/-- Given a line with equation x + y - 2 = 0, prove its slope and angle of inclination -/
theorem line_slope_and_inclination :
  ∃ (m θ : ℝ),
    (∀ x y : ℝ, x + y - 2 = 0 → y = m * x + 2) ∧  -- Slope-intercept form
    m = -1 ∧                                      -- Slope
    θ = 3 * Real.pi / 4 ∧                         -- Angle of inclination
    Real.tan θ = m ∧                              -- Tangent of angle equals slope
    0 ≤ θ ∧ θ < Real.pi                           -- Angle in [0, π)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_inclination_l916_91636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_equation_l916_91632

/-- Represents the annual average growth rate of education funds investment -/
def x : ℝ := sorry

/-- The education funds investment in 2014 (in millions of yuan) -/
def investment_2014 : ℝ := 2500

/-- The education funds investment in 2016 (in millions of yuan) -/
def investment_2016 : ℝ := 3500

/-- Theorem stating the relationship between investments and growth rate -/
theorem investment_growth_equation :
  investment_2014 * (1 + x)^2 = investment_2016 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_equation_l916_91632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_3142_l916_91641

def digits : List Nat := [1, 2, 3, 4]

def is_valid_number (n : Nat) : Bool :=
  n ≥ 1000 && n ≤ 9999 && List.all (n.repr.data) (λ c => c.toNat - 48 ∈ digits)

def valid_numbers : List Nat :=
  (List.range 9000).filter (λ n => is_valid_number (n + 1000))

theorem position_of_3142 :
  valid_numbers.indexOf 3142 = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_3142_l916_91641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l916_91686

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The y-coordinate of the midpoint of C and D -/
noncomputable def y_midpoint : ℝ := 6 - 3 * Real.sqrt 3

/-- The theorem statement -/
theorem ellipse_theorem (e : Ellipse) 
  (h_ecc : e.eccentricity = 1/2) 
  (h_mid : y_midpoint = (Real.sqrt 3 / 2) * e.a) :
  (∃ (x y : ℝ), x^2/4 + y^2/3 = 1) ∧
  (∃ (M N N' : ℝ × ℝ), 
    let F := (1, 0)
    let k := (N.2 - F.2) / (N.1 - F.1)
    (4, 0) ∈ Set.range (λ t : ℝ => (t, (M.2 - N'.2)/(M.1 - N'.1) * (t - M.1) + M.2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l916_91686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_equals_22032_l916_91610

/-- The function b(p) is defined as the unique positive integer k such that |k - √p| < 1/2 -/
noncomputable def b (p : ℕ+) : ℕ+ :=
  ⟨(Int.floor (Real.sqrt p.val + 1/2)).toNat + 1, sorry⟩

/-- The sum of b(p) for p from 1 to 1000 -/
noncomputable def S : ℕ := (Finset.range 1000).sum (fun i => (b ⟨i+1, sorry⟩).val)

/-- The main theorem stating that S equals 22032 -/
theorem sum_b_equals_22032 : S = 22032 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_equals_22032_l916_91610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_approx_9_25_l916_91666

/-- Function F as defined in the problem -/
noncomputable def F (a b c d e : ℝ) : ℝ := a^b + (c * d) / e

/-- The theorem statement -/
theorem exists_x_approx_9_25 :
  ∃ x : ℝ, F 2 (x - 2) 4 11 2 = 174 ∧ |x - 9.25| < 0.01 := by
  sorry

#check exists_x_approx_9_25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_approx_9_25_l916_91666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l916_91663

/-- The circle C in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The line l in the problem -/
def problem_line (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

/-- Point A through which line l passes -/
def point_A : ℝ × ℝ := (2, -2)

/-- Theorem stating the y-intercept of line l -/
theorem line_y_intercept :
  ∃ (m b : ℝ),
    (∀ x y, problem_line m b x y → (x = point_A.1 ∧ y = point_A.2) → True) ∧
    (∀ x y, problem_circle x y → 
      ∀ m' b', (∀ x' y', problem_line m' b' x' y' → (x' = point_A.1 ∧ y' = point_A.2) → True) →
        ∃ d, ∀ x'' y'', problem_line m' b' x'' y'' → 
          (|m'*x + b' - y| : ℝ) / Real.sqrt (m'^2 + 1) ≤ d) ∧
    b = -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l916_91663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_false_bag_l916_91670

/-- A structure representing a balance with two pans and a pointer. -/
structure Balance where
  reading : ℚ → ℚ → ℚ
  reading_def : ∀ L R, reading L R = R - L

/-- A structure representing a collection of coin bags. -/
structure CoinBags where
  n : ℕ
  n_ge_2 : n ≥ 2
  coins_per_bag : ℕ
  coins_per_bag_def : coins_per_bag = n * (n - 1) / 2 + 1
  real_weight : ℚ
  false_weight : ℚ
  false_weight_diff : false_weight ≠ real_weight

/-- Function to perform the first weighing. -/
noncomputable def first_weighing (b : Balance) (c : CoinBags) : ℚ := 
  b.reading (c.n * (c.n - 1) / 2 * c.real_weight) 
            ((c.n - 1) * c.n / 2 * c.real_weight)

/-- Function to perform the second weighing. -/
noncomputable def second_weighing (b : Balance) (c : CoinBags) : ℚ := 
  b.reading (c.n * (c.n - 1) / 2 * c.real_weight) 
            ((c.n - 1) * c.n / 2 * c.real_weight)

/-- Theorem stating that the bag with false coins can be identified. -/
theorem identify_false_bag (b : Balance) (c : CoinBags) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ c.n ∧ 
  i = (c.n * (first_weighing b c)) / (second_weighing b c + first_weighing b c) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_false_bag_l916_91670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l916_91615

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) > 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality a x ∧ x > 1 ∧ x < 2}

-- Theorem statement
theorem inequality_solution_implies_a_value :
  ∀ a : ℝ, solution_set a = Set.Ioo 1 2 → a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l916_91615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_pair_after_removal_l916_91669

/-- Represents a deck of cards after a pair has been removed -/
structure DeckAfterRemoval where
  threeCardSet : Nat  -- Number of sets with 3 cards
  fourCardSets : Nat  -- Number of sets with 4 cards

/-- Calculate the probability of selecting a matching pair from the deck -/
def probabilityOfPair (deck : DeckAfterRemoval) : Rat :=
  let totalCards := deck.threeCardSet * 3 + deck.fourCardSets * 4
  let totalWays := Nat.choose totalCards 2
  let pairsFromThree := deck.threeCardSet * Nat.choose 3 2
  let pairsFromFour := deck.fourCardSets * Nat.choose 4 2
  (pairsFromThree + pairsFromFour : Nat) / totalWays

/-- The main theorem to prove -/
theorem probability_of_pair_after_removal :
  let deck : DeckAfterRemoval := { threeCardSet := 1, fourCardSets := 9 }
  probabilityOfPair deck = 57 / 703 := by
  sorry

#eval probabilityOfPair { threeCardSet := 1, fourCardSets := 9 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_pair_after_removal_l916_91669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_satisfying_inequality_l916_91695

theorem greatest_x_satisfying_inequality : 
  ∃ (x : ℤ), (3.71 * (10 : ℝ)^x) / (6.52 * (10 : ℝ)^(x-3)) < 10230 ∧ 
  ∀ (y : ℤ), y > x → (3.71 * (10 : ℝ)^y) / (6.52 * (10 : ℝ)^(y-3)) ≥ 10230 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_satisfying_inequality_l916_91695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_transitive_perpendicular_parallel_implies_perpendicular_planes_perpendicular_parallel_line_plane_parallel_lines_plane_implies_parallel_line_plane_l916_91616

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ˡ " => parallel_lines
local infix:50 " ∥ᵖ " => parallel_planes
local infix:50 " ∥ˡᵖ " => parallel_line_plane
local infix:50 " ⊥ˡᵖ " => perpendicular_line_plane
local infix:50 " ⊥ᵖ " => perpendicular_planes
local infix:50 " ⊂ " => line_in_plane

-- Theorem 1
theorem parallel_planes_transitive (α β γ : Plane) :
  α ∥ᵖ β → α ∥ᵖ γ → β ∥ᵖ γ := by sorry

-- Theorem 2
theorem perpendicular_parallel_implies_perpendicular_planes (m : Line) (α β : Plane) :
  m ⊥ˡᵖ α → m ∥ˡᵖ β → α ⊥ᵖ β := by sorry

-- Additional theorems for completeness (not required by the original problem)
theorem perpendicular_parallel_line_plane (m : Line) (α β : Plane) :
  α ⊥ᵖ β → m ∥ˡᵖ α → m ⊥ˡᵖ β := by sorry

theorem parallel_lines_plane_implies_parallel_line_plane (m n : Line) (α : Plane) :
  m ∥ˡ n → n ⊂ α → m ∥ˡᵖ α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_transitive_perpendicular_parallel_implies_perpendicular_planes_perpendicular_parallel_line_plane_parallel_lines_plane_implies_parallel_line_plane_l916_91616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_30_60_90_triangles_l916_91657

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  long_leg : ℝ
  short_leg : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The theorem to be proved -/
theorem area_of_overlapping_30_60_90_triangles :
  ∀ (t : Triangle30_60_90),
    t.hypotenuse = 10 →
    t.long_leg = 5 * Real.sqrt 3 →
    t.short_leg = 5 →
    triangleArea t.short_leg t.long_leg = (25 * Real.sqrt 3) / 2 := by
  intro t hyp_hypotenuse hyp_long_leg hyp_short_leg
  unfold triangleArea
  rw [hyp_long_leg, hyp_short_leg]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_30_60_90_triangles_l916_91657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_99_percent_composite_l916_91605

/-- Sequence defined by a(n) = 10^n + 1 -/
def a (n : ℕ) : ℕ := 10^n + 1

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

theorem at_least_99_percent_composite :
  ∃ (S : Finset ℕ), S.card ≥ 1980 ∧ S ⊆ Finset.range 2000 ∧ ∀ i ∈ S, IsComposite (a i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_99_percent_composite_l916_91605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l916_91683

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos (2 * x)

-- State the theorem about the range of the function
theorem f_range :
  (∀ x : ℝ, f x ≥ -2 ∧ f x ≤ 9/8) ∧
  (∃ x : ℝ, f x = -2) ∧
  (∃ x : ℝ, f x = 9/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l916_91683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_tbone_steaks_expected_value_filet_steaks_l916_91612

/-- Represents the types of steaks -/
inductive SteakType
| Filet
| Ribeye
| Sirloin
| TBone

/-- Represents the distribution of steak types in 100 boxes -/
def steak_distribution : List (SteakType × ℕ) :=
  [(SteakType.Filet, 20), (SteakType.Ribeye, 30), (SteakType.Sirloin, 20), (SteakType.TBone, 30)]

/-- The total number of boxes -/
def total_boxes : ℕ := 100

/-- The number of boxes selected using stratified sampling -/
def stratified_sample_size : ℕ := 10

/-- The number of boxes selected from the stratified sample -/
def final_sample_size : ℕ := 4

/-- The number of boxes for calculating expected value -/
def expected_value_sample_size : ℕ := 3

/-- Theorem for the probability of selecting exactly 2 T-bone steaks -/
theorem probability_two_tbone_steaks :
  let p := (Nat.choose 3 2 * Nat.choose 7 2) / Nat.choose 10 4
  p = 3 / 10 := by sorry

/-- Theorem for the expected value of filet steaks in 3 boxes -/
theorem expected_value_filet_steaks :
  let p : Rat := 20 / total_boxes
  let expected_value : Rat := expected_value_sample_size * p
  expected_value = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_tbone_steaks_expected_value_filet_steaks_l916_91612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barry_finishes_first_l916_91609

noncomputable section

variable (x y : ℝ)

-- Define the areas of lawns
def barry_lawn_area : ℝ := x
def stacy_lawn_area : ℝ := x / 3
def tom_lawn_area : ℝ := x / 4

-- Define the mowing rates
def barry_mowing_rate : ℝ := y
def stacy_mowing_rate : ℝ := y / 4
def tom_mowing_rate : ℝ := y / 8

-- Calculate mowing times
def barry_mowing_time : ℝ := barry_lawn_area x / barry_mowing_rate y
def stacy_mowing_time : ℝ := stacy_lawn_area x / stacy_mowing_rate y
def tom_mowing_time : ℝ := tom_lawn_area x / tom_mowing_rate y

-- Theorem statement
theorem barry_finishes_first (hx : x > 0) (hy : y > 0) :
  barry_mowing_time x y < stacy_mowing_time x y ∧ barry_mowing_time x y < tom_mowing_time x y :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barry_finishes_first_l916_91609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_platform_passengers_l916_91613

theorem railway_platform_passengers (initial_passengers : ℕ) : 
  (initial_passengers - initial_passengers / 10 - 
   (initial_passengers - initial_passengers / 10) / 7 - 
   (initial_passengers - initial_passengers / 10 - 
    (initial_passengers - initial_passengers / 10) / 7) / 5 = 216) → 
  initial_passengers = 350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_platform_passengers_l916_91613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraiser_bronze_families_correct_l916_91658

def fundraiser_bronze_families 
  (total_goal : ℕ) 
  (bronze_donation silver_donation gold_donation : ℕ) 
  (silver_families gold_families : ℕ) 
  (final_day_need : ℕ) : ℕ :=
  let total_raised := total_goal - final_day_need
  let silver_contribution := silver_families * silver_donation
  let gold_contribution := gold_families * gold_donation
  let bronze_contribution := total_raised - (silver_contribution + gold_contribution)
  bronze_contribution / bronze_donation

theorem fundraiser_bronze_families_correct
  (total_goal : ℕ) 
  (bronze_donation silver_donation gold_donation : ℕ) 
  (silver_families gold_families : ℕ) 
  (final_day_need : ℕ) :
  fundraiser_bronze_families total_goal bronze_donation silver_donation gold_donation silver_families gold_families final_day_need = 10 :=
by
  -- The proof goes here
  sorry

#eval fundraiser_bronze_families 750 25 50 100 7 1 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraiser_bronze_families_correct_l916_91658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixture_l916_91624

/-- Given initial and added salt solutions, calculate the resulting salt concentration -/
noncomputable def resulting_concentration (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) : ℝ :=
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
  (initial_volume + added_volume)

/-- Theorem: Adding 25 ounces of 40% salt solution to 50 ounces of 10% salt solution 
    results in a 20% salt solution -/
theorem salt_solution_mixture : 
  resulting_concentration 50 0.1 25 0.4 = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixture_l916_91624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_larry_likeable_two_digits_l916_91640

/-- A number is Larry-likeable if it's divisible by 4 -/
def is_larry_likeable (n : Nat) : Prop := n % 4 = 0

/-- The set of two-digit numbers (from 00 to 99) -/
def two_digit_numbers : Set Nat := {n : Nat | 0 ≤ n ∧ n ≤ 99}

/-- The set of Larry-likeable two-digit numbers -/
def larry_likeable_two_digits : Set Nat := {n ∈ two_digit_numbers | is_larry_likeable n}

theorem count_larry_likeable_two_digits : 
  Finset.card (Finset.filter (fun n => n % 4 = 0) (Finset.range 100)) = 25 := by
  sorry

#eval Finset.card (Finset.filter (fun n => n % 4 = 0) (Finset.range 100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_larry_likeable_two_digits_l916_91640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_a_l916_91621

theorem nearest_integer_a (x : ℚ) : 
  x = 19 / 15 + 19 / 3 → 
  ∃ n : ℤ, n = 8 ∧ ∀ m : ℤ, |x - n| ≤ |x - m| :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_a_l916_91621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_variance_properties_l916_91629

def isEqualVarianceSequence (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

theorem equal_variance_properties :
  (-- Statement 1
   isEqualVarianceSequence (fun n => (-1)^n)) ∧
  (-- Statement 2
   ∀ a : ℕ → ℝ, isEqualVarianceSequence a → 
   ∃ d : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = d) ∧
  (-- Statement 3
   ∀ a : ℕ → ℝ, isEqualVarianceSequence a → 
   ∀ k : ℕ, k > 0 → isEqualVarianceSequence (fun n => a (k * n))) ∧
  (-- Statement 4
   ∀ a : ℕ → ℝ, isEqualVarianceSequence a → 
   (∃ d : ℝ, ∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = d) → 
   ∃ c : ℝ, ∀ n : ℕ, a n = c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_variance_properties_l916_91629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l916_91631

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmh : ℝ)
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : train_speed_kmh = 72) :
  (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 20 := by
  -- Convert speed from km/h to m/s
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  
  -- Calculate total distance
  have total_distance : ℝ := train_length + bridge_length
  
  -- Calculate crossing time
  have crossing_time : ℝ := total_distance / train_speed_ms
  
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l916_91631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_monotonic_decreasing_condition_l916_91639

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Theorem 1: Monotonically increasing on ℝ
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 :=
sorry

-- Theorem 2: Monotonically decreasing on (-1, 1)
theorem monotonic_decreasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 1, StrictMonoOn (fun x => -(f a x)) (Set.Ioo (-1) 1)) ↔ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_monotonic_decreasing_condition_l916_91639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_passengers_for_coverage_l916_91611

/-- Represents a bus route with n stations -/
def BusRoute (n : ℕ) := Fin n

/-- Represents a passenger's journey from station i to station j -/
structure Passenger (n : ℕ) where
  board : BusRoute n
  deboard : BusRoute n
  h : board.val < deboard.val

/-- Checks if a set of passengers covers all required pairs -/
def covers_all_pairs (n : ℕ) (passengers : Finset (Passenger n)) : Prop :=
  ∀ (i₁ i₂ j₁ j₂ : BusRoute n),
    i₁.val < i₂.val → i₂.val < j₁.val → j₁.val < j₂.val →
    (∃ p ∈ passengers, (p.board = i₁ ∧ p.deboard = j₁) ∨ (p.board = i₂ ∧ p.deboard = j₂))

theorem min_passengers_for_coverage :
  ∃ (passengers : Finset (Passenger 2018)),
    passengers.card = 1009 ∧
    covers_all_pairs 2018 passengers ∧
    ∀ (smaller_set : Finset (Passenger 2018)),
      smaller_set.card < 1009 →
      ¬(covers_all_pairs 2018 smaller_set) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_passengers_for_coverage_l916_91611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l916_91672

/-- The function g(x) = x^2 ln(x) -/
noncomputable def g (x : ℝ) : ℝ := x^2 * Real.log x

/-- The derivative of g(x) -/
noncomputable def g_deriv (x : ℝ) : ℝ := x * (2 * Real.log x + 1)

/-- The theorem stating that the angle of inclination of the tangent line to g(x) at x = 1 is π/4 -/
theorem tangent_angle_at_one :
  Real.arctan (g_deriv 1) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l916_91672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l916_91675

/-- The parabola y^2 = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line 4x - 3y + 11 = 0 -/
def Line (x y : ℝ) : Prop := 4*x - 3*y + 11 = 0

/-- Distance from a point (x, y) to the axis of symmetry x = 1 -/
def d1 (x y : ℝ) : ℝ := |x - 1|

/-- Distance from a point (x, y) to the line 4x - 3y + 11 = 0 -/
noncomputable def d2 (x y : ℝ) : ℝ := |4*x - 3*y + 11| / Real.sqrt 25

theorem min_distance_sum :
  ∃ (min : ℝ), min = 3 ∧
  ∀ (x y : ℝ), Parabola x y →
  d1 x y + d2 x y ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l916_91675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_range_l916_91651

/-- A function f: ℝ → ℝ is monotonically increasing on an interval I if
    for all x, y ∈ I with x < y, we have f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ I → y ∈ I → x < y → f x < f y

/-- The function f(x) = ax² + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

/-- The interval (-∞, 4) -/
def I : Set ℝ := Set.Iio 4

/-- The theorem stating the range of a for which f is monotonically increasing on I -/
theorem f_monotone_range :
  {a : ℝ | MonotonicallyIncreasing (f a) I} = Set.Icc (-1/4) 0 := by
  sorry

#check f_monotone_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_range_l916_91651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l916_91671

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℚ
  rate1 : ℚ
  rate2 : ℚ
  amount1 : ℚ
  amount2 : ℚ

/-- Calculates the average interest rate for an investment -/
def averageRate (inv : Investment) : ℚ :=
  (inv.rate1 * inv.amount1 + inv.rate2 * inv.amount2) / inv.total

theorem investment_average_rate :
  ∀ (inv : Investment),
    inv.total = 6000 ∧
    inv.rate1 = 3/100 ∧
    inv.rate2 = 5/100 ∧
    inv.amount1 + inv.amount2 = inv.total ∧
    inv.rate1 * inv.amount1 = inv.rate2 * inv.amount2 →
    averageRate inv = 375/10000 := by
  sorry

#eval (375 : ℚ) / 10000  -- To display the result as 0.0375

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l916_91671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l916_91647

-- Define the function f(x) = log₃x + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 3

-- State the theorem
theorem root_in_interval :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

-- Additional lemmas to support the main theorem
lemma f_continuous : Continuous f := by
  sorry

lemma f_strictly_increasing : StrictMono f := by
  sorry

lemma f_neg_at_two : f 2 < 0 := by
  sorry

lemma f_pos_at_three : f 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l916_91647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_equality_l916_91614

theorem log_power_equality (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Real.log 2 / Real.log a = m) → (Real.log a / Real.log 3 = n) → (a ^ (m + 2 * n) = 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_equality_l916_91614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_a_range_l916_91645

/-- The function f(x) defined on (-1, 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a/2

/-- Theorem stating that if f(x) is positive for all x in (-1, 1), then a is in (0, 2] -/
theorem f_positive_implies_a_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-1) 1, f a x > 0) → a ∈ Set.Ioc 0 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_a_range_l916_91645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2x_4y_l916_91618

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The point A(0,4) -/
def A : Point := ⟨0, 4⟩

/-- The point B(-2,0) -/
def B : Point := ⟨-2, 0⟩

/-- A point P(x,y) equidistant from A and B -/
def P (x y : ℝ) : Point := ⟨x, y⟩

theorem min_value_2x_4y : 
    ∀ (P : Point), distance P A = distance P B → 
    2^P.x + 4^P.y ≥ 4 * Real.sqrt 2 ∧ 
    ∃ (P : Point), distance P A = distance P B ∧ 
      2^P.x + 4^P.y = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2x_4y_l916_91618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l916_91604

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.cos (x + Real.pi / 2)

theorem problem_solution (α β : ℝ) 
  (h1 : α ∈ Set.Ioo (-Real.pi/2) 0) 
  (h2 : β ∈ Set.Ioo (-Real.pi/2) 0)
  (h3 : f (α + 3*Real.pi/4) = -3*Real.sqrt 2/5)
  (h4 : f (Real.pi/4 - β) = -5*Real.sqrt 2/13) : 
  f (Real.pi/12) = Real.sqrt 2/2 ∧ Real.cos (α + β) = 16/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l916_91604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l916_91620

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 1/a|

-- Theorem statement
theorem f_properties (a : ℝ) (ha : a ≠ 0) :
  -- Part 1: Solution set for a = 2
  (a = 2 → {x : ℝ | f 2 x > 3} = {x : ℝ | x < -11/4 ∨ x > 1/4}) ∧
  -- Part 2: Lower bound for f(m) + f(-1/m)
  (∀ m : ℝ, f a m + f a (-1/m) ≥ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l916_91620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plain_area_calculation_l916_91652

theorem plain_area_calculation (total_area difference : ℝ) : ℝ :=
  let x := 200
  have h1 : total_area = 350 := by sorry
  have h2 : difference = 50 := by sorry
  have h3 : x + (x - difference) = total_area := by sorry
  x

#check plain_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plain_area_calculation_l916_91652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_unique_l916_91625

/-- A monic cubic polynomial with specific properties -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, p = fun x ↦ x^3 + a*x^2 + b*x + c) ∧  -- monic cubic
  (p 0 = 1) ∧                                        -- p(0) = 1
  (∀ x : ℝ, (deriv p) x = 0 → p x = 0)               -- zeroes of p' are zeroes of p

/-- The theorem stating the unique form of the special polynomial -/
theorem special_polynomial_unique (p : ℝ → ℝ) :
  special_polynomial p → p = fun x ↦ (x + 1)^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_unique_l916_91625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_side_is_18_l916_91662

/-- Represents the dimensions of the tofu -/
structure TofuDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the side length of the largest square that can be cut from the tofu three times -/
def largestSquareSide (dimensions : TofuDimensions) : ℕ :=
  min dimensions.width (dimensions.length / 3)

theorem largest_square_side_is_18 (tofu : TofuDimensions) 
    (h1 : tofu.length = 54)
    (h2 : tofu.width = 20) : 
  largestSquareSide tofu = 18 := by
  sorry

#eval largestSquareSide { length := 54, width := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_side_is_18_l916_91662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_4_neg4_6_l916_91655

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y > 0 then Real.pi/2
           else if y < 0 then 3*Real.pi/2
           else 0  -- undefined, but we need to return something
  (r, θ, z)

theorem rectangular_to_cylindrical_4_neg4_6 :
  let (r, θ, z) := rectangular_to_cylindrical 4 (-4) 6
  r = 4 * Real.sqrt 2 ∧ θ = 7*Real.pi/4 ∧ z = 6 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_4_neg4_6_l916_91655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_ln_2x_l916_91654

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x)

-- State the theorem
theorem derivative_of_ln_2x (x : ℝ) (h : x ≠ 0) : 
  deriv f x = 1 / x := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_ln_2x_l916_91654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l916_91691

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- State the theorem
theorem max_value_f :
  ∃ (x : ℝ), π / 4 ≤ x ∧ x ≤ π / 2 ∧
  f x = 3 / 2 ∧
  ∀ (y : ℝ), π / 4 ≤ y ∧ y ≤ π / 2 → f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l916_91691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_neg_i_l916_91693

theorem complex_fraction_equals_neg_i : 
  (1 - Complex.I) / (1 + Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_neg_i_l916_91693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_in_square_l916_91685

/-- Represents a square with side length in centimeters -/
structure Square where
  side_length : ℝ

/-- Represents a circle with diameter in millimeters -/
structure Circle where
  diameter : ℝ

/-- Function to convert centimeters to millimeters -/
noncomputable def cm_to_mm (x : ℝ) : ℝ := 10 * x

/-- Function to convert millimeters to meters -/
noncomputable def mm_to_m (x : ℝ) : ℝ := x / 1000

/-- Theorem stating that it's possible to cut out circles from a 10 cm square
    such that the sum of their diameters exceeds 5 meters -/
theorem circles_in_square (s : Square) (h : s.side_length = 10) :
  ∃ (circles : List Circle), 
    (∀ c ∈ circles, c.diameter ≤ cm_to_mm s.side_length) ∧ 
    (mm_to_m (List.sum (List.map Circle.diameter circles)) > 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_in_square_l916_91685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_l916_91607

-- Define the angle α in the third quadrant
variable (α : Real)
axiom α_in_third_quadrant : Real.pi < α ∧ α < 3*Real.pi/2

-- Define the function f
noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (3*Real.pi/2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

-- Theorem 1: Simplification of f(α)
theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

-- Theorem 2: Value of f(α) when cos(α - 3π/2) = 1/5
theorem f_value_when_cos_condition 
  (h : Real.cos (α - 3*Real.pi/2) = 1/5) : f α = 2 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_l916_91607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_theorem_l916_91635

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a telephone number in the format ABC-DEF-GHIJ -/
structure PhoneNumber where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  G : Digit
  H : Digit
  I : Digit
  J : Digit

/-- Checks if three digits are consecutive -/
def areConsecutive (a b c : Digit) : Prop :=
  ∃ (x : Digit), (a.val = x.val ∧ b.val = x.val + 1 ∧ c.val = x.val + 2) ∨
                 (a.val = x.val + 1 ∧ b.val = x.val ∧ c.val = x.val + 2) ∨
                 (a.val = x.val + 1 ∧ b.val = x.val + 2 ∧ c.val = x.val) ∨
                 (a.val = x.val + 2 ∧ b.val = x.val ∧ c.val = x.val + 1) ∨
                 (a.val = x.val + 2 ∧ b.val = x.val + 1 ∧ c.val = x.val)

/-- Checks if a digit is odd -/
def isOdd (d : Digit) : Prop := d.val % 2 = 1

/-- Checks if four digits are consecutive odd numbers -/
def areConsecutiveOdd (a b c d : Digit) : Prop :=
  isOdd a ∧ isOdd b ∧ isOdd c ∧ isOdd d ∧
  ∃ (x : Digit), (a.val = x.val ∧ b.val = x.val + 2 ∧ c.val = x.val + 4 ∧ d.val = x.val + 6)

/-- Main theorem: Given the conditions, A must equal 6 -/
theorem phone_number_theorem (pn : PhoneNumber) : 
  (pn.A.val > pn.B.val ∧ pn.B.val > pn.C.val) →
  (pn.D.val > pn.E.val ∧ pn.E.val > pn.F.val) →
  (pn.G.val > pn.H.val ∧ pn.H.val > pn.I.val ∧ pn.I.val > pn.J.val) →
  areConsecutive pn.D pn.E pn.F →
  areConsecutiveOdd pn.G pn.H pn.I pn.J →
  pn.A.val + pn.B.val + pn.C.val = 12 →
  pn.A ≠ pn.B ∧ pn.A ≠ pn.C ∧ pn.B ≠ pn.C →
  pn.A = ⟨6, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_theorem_l916_91635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fraction_l916_91660

noncomputable section

-- Define the function f
def f (a b c x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * a * x^2 + 2 * b * x + c

-- State the theorem
theorem range_of_fraction (a b c : ℝ) :
  (∀ x : ℝ, Differentiable ℝ (fun x ↦ f a b c x)) →
  (∃ x₁ ∈ Set.Ioo 0 1, IsLocalMax (f a b c) x₁) →
  (∃ x₂ ∈ Set.Ioo 1 2, IsLocalMin (f a b c) x₂) →
  ∃ y ∈ Set.Ioo (1/4) 1, y = (b - 2) / (a - 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fraction_l916_91660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_tangent_circle_l916_91623

noncomputable section

structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

def Ellipse.foci (e : Ellipse) : ℝ × ℝ :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  (-c, c)

theorem ellipse_properties (e : Ellipse) 
  (h₁ : (e.foci.2 - e.foci.1) = 2)
  (h₂ : e.equation 1 (3/2)) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ e.eccentricity = 1/2 := by
  sorry

theorem tangent_circle 
  (e : Ellipse)
  (h₁ : e.a = 2 ∧ e.b = Real.sqrt 3)
  (l : ℝ → ℝ)
  (A B : ℝ × ℝ)
  (h₂ : e.equation A.1 A.2 ∧ e.equation B.1 B.2)
  (h₃ : ∃ t : ℝ, A.1 = l A.2 ∧ B.1 = l B.2)
  (h₄ : (1/2) * |A.2 - B.2| * 2 = 12 * Real.sqrt 2 / 7) :
  ∃ (x₀ y₀ : ℝ), ∀ (x y : ℝ),
    (x - 1)^2 + y^2 = 2 ↔ (x - x₀)^2 + (y - y₀)^2 = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_tangent_circle_l916_91623
