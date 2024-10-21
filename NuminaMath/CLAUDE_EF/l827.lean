import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_closed_interval_neg_one_one_l827_82742

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 2))

-- Define the domain A of f
def A : Set ℝ := {x | x < -1 ∨ x > 2}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a^2 + a > 0}

-- State the theorem
theorem a_in_closed_interval_neg_one_one 
  (h : A ∪ B a = B a) : 
  a ∈ Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_closed_interval_neg_one_one_l827_82742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l827_82774

/-- The angle between two 2D vectors -/
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

/-- Theorem: The angle between vectors (1,-2) and (3,1) is arccos(√2/10) -/
theorem angle_between_specific_vectors :
  angle_between (1, -2) (3, 1) = Real.arccos (Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l827_82774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_when_y_is_4_l827_82789

-- Define the functions h and k
noncomputable def k (y : ℝ) : ℝ := 40 / (y + 5)
noncomputable def h (y : ℝ) : ℝ := 4 * (k y)⁻¹

-- State the theorem
theorem h_equals_20_when_y_is_4 :
  ∃! y : ℝ, h y = 20 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_when_y_is_4_l827_82789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_greater_than_f_l827_82727

open Real

/-- The function f(x) = x - 1/x -/
noncomputable def f (x : ℝ) : ℝ := x - 1/x

/-- The function g(x) = 1 + x ln x -/
noncomputable def g (x : ℝ) : ℝ := 1 + x * log x

/-- Theorem stating that g(x) > f(x) for all x > 0 -/
theorem g_greater_than_f : ∀ x > 0, g x > f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_greater_than_f_l827_82727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_integer_solutions_l827_82731

theorem quadratic_inequality_integer_solutions :
  (Finset.filter (fun x => x^2 - 10*x + 16 > 0) (Finset.range 201)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_integer_solutions_l827_82731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_annual_health_insurance_cost_l827_82701

/-- Calculates the government contribution percentage based on annual income -/
noncomputable def income_contribution (annual_income : ℝ) : ℝ :=
  if annual_income < 10000 then 0.90
  else if annual_income ≤ 25000 then 0.75
  else if annual_income ≤ 40000 then 0.50
  else if annual_income ≤ 55000 then 0.35
  else if annual_income ≤ 70000 then 0.20
  else 0.10

/-- Calculates the additional government contribution percentage based on age -/
def age_contribution (age : ℕ) : ℝ :=
  if age > 55 then 0.10
  else if age > 45 then 0.05
  else 0

/-- Calculates the additional government contribution percentage based on family size -/
def family_size_contribution (family_size : ℕ) : ℝ :=
  if family_size ≥ 3 then 0.10
  else if family_size = 2 then 0.05
  else 0

/-- Calculates the annual health insurance cost -/
noncomputable def annual_health_insurance_cost (
  monthly_plan_price : ℝ)
  (hourly_wage : ℝ)
  (weekly_work_hours : ℝ)
  (age : ℕ)
  (family_size : ℕ) : ℝ :=
  let annual_income := hourly_wage * weekly_work_hours * 4 * 12
  let total_contribution := income_contribution annual_income +
                            age_contribution age +
                            family_size_contribution family_size
  let monthly_cost := monthly_plan_price * (1 - total_contribution)
  monthly_cost * 12

/-- Theorem: Bill's annual health insurance cost is $2400 -/
theorem bills_annual_health_insurance_cost :
  annual_health_insurance_cost 500 25 30 38 3 = 2400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_annual_health_insurance_cost_l827_82701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_reordering_bound_l827_82773

/-- A vector in ℝ² -/
def Vector2D := ℝ × ℝ

/-- The zero vector in ℝ² -/
def zeroVector : Vector2D := (0, 0)

/-- The magnitude (length) of a vector -/
noncomputable def magnitude (v : Vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- The sum of a list of vectors -/
def vectorSum (vs : List Vector2D) : Vector2D :=
  vs.foldl (fun acc v => (acc.1 + v.1, acc.2 + v.2)) zeroVector

theorem vector_reordering_bound (n : ℕ) (vs : List Vector2D) 
  (h1 : vs.length = n)
  (h2 : ∀ v ∈ vs, magnitude v = 1)
  (h3 : vectorSum vs = zeroVector) :
  ∃ perm : List Vector2D, 
    (Multiset.ofList perm = Multiset.ofList vs) ∧ 
    (∀ k, k ≤ n → magnitude (vectorSum (perm.take k)) ≤ Real.sqrt 5) := by
  sorry

#check vector_reordering_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_reordering_bound_l827_82773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_length_l827_82759

theorem circle_segment_length 
  (C : ℝ → ℝ → Prop) 
  (A B : ℝ × ℝ) :
  (∃ r : ℝ, ∀ x y : ℝ, C x y ↔ (x - r)^2 + (y - r)^2 = r^2) →  -- Circle definition
  (∃ r : ℝ, 2 * Real.pi * r = 18 * Real.pi) →  -- Circumference condition
  (∀ x y : ℝ, C x y → (x - A.1)^2 + (y - A.2)^2 ≤ (B.1 - A.1)^2 + (B.2 - A.2)^2) →  -- AB is diameter
  (∃ p : ℝ × ℝ, C p.1 p.2 ∧ 
    Real.arccos ((p.1 - A.1) * (B.1 - p.1) + (p.2 - A.2) * (B.2 - p.2)) / 
    (((p.1 - A.1)^2 + (p.2 - A.2)^2)^(1/2) * ((B.1 - p.1)^2 + (B.2 - p.2)^2)^(1/2)) = Real.pi / 6) →  -- Angle ACB is 30°
  ∃ D : ℝ × ℝ, C D.1 D.2 ∧ 
    ((D.1 - B.1)^2 + (D.2 - B.2)^2)^(1/2) = 9 * (2 - 3^(1/2))^(1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_length_l827_82759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_M_l827_82762

noncomputable def M : Set ℝ := {x : ℝ | x ≤ 3 * Real.sqrt 3}
noncomputable def a : ℝ := 2 * Real.sqrt 6

theorem a_in_M : a ∈ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_M_l827_82762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_negative_six_over_thirtyseven_l827_82711

def my_sequence (n : ℕ) : ℚ :=
  if n % 2 = 1 then
    (n : ℚ) / (2 + (List.range n).sum)
  else
    -(n : ℚ) / (2 + (List.range n).sum)

theorem sixth_term_is_negative_six_over_thirtyseven :
  my_sequence 6 = -6 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_negative_six_over_thirtyseven_l827_82711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l827_82770

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

def line_equation (x y : ℝ) : Prop := 3*x + 4*y + 4 = 0

noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2)

theorem distance_center_to_line :
  ∃ (x₀ y₀ : ℝ),
    (∀ x y, circle_equation x y ↔ (x - x₀)^2 + (y - y₀)^2 = (x₀ - 1)^2 + (y₀ - 2)^2) →
    distance_point_to_line x₀ y₀ 3 4 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l827_82770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l827_82703

/-- Represents a 5x5 grid where X's can be placed -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if three X's are aligned in a given direction -/
def three_aligned (g : Grid) (dx dy : Int) : Prop :=
  ∃ x y, g x y ∧ g (x + dx) (y + dy) ∧ g (x + 2*dx) (y + 2*dy)

/-- Checks if a grid configuration is valid (no three X's aligned) -/
def valid_grid (g : Grid) : Prop :=
  ¬(three_aligned g 1 0 ∨ three_aligned g 0 1 ∨ 
    three_aligned g 1 1 ∨ three_aligned g 1 (-1))

/-- Counts the number of X's in a grid -/
def count_x (g : Grid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 5)) (λ x => 
    Finset.sum (Finset.univ : Finset (Fin 5)) (λ y => 
      if g x y then 1 else 0)))

/-- Theorem stating that 11 is the maximum number of X's that can be placed -/
theorem max_x_placement :
  (∃ g : Grid, valid_grid g ∧ count_x g = 11) ∧
  (∀ g : Grid, valid_grid g → count_x g ≤ 11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l827_82703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_120_x_intercept_2_parallel_lines_distance_l827_82748

-- Define the line l₁: ax + y + 2 = 0
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 2 = 0

-- Define the line l₂: 2x - y + 1 = 0
def line_l2 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem 1: If the slope angle of l₁ is 120°, then a = √3
theorem slope_angle_120 (a : ℝ) :
  (∃ x y, line_l1 a x y ∧ Real.tan (120 * π / 180) = -a) → a = Real.sqrt 3 := by
  sorry

-- Theorem 2: If the x-intercept of l₁ is 2, then a = -1
theorem x_intercept_2 (a : ℝ) :
  (∃ x, line_l1 a x 0 ∧ x = 2) → a = -1 := by
  sorry

-- Theorem 3: If l₁ is parallel to l₂, then the distance between them is (3√5)/5
theorem parallel_lines_distance (a : ℝ) :
  (∀ x y, line_l1 a x y ↔ line_l2 x y) →
  (∃ d, d = (3 * Real.sqrt 5) / 5 ∧
    ∀ x₁ y₁ x₂ y₂, line_l1 a x₁ y₁ → line_l2 x₂ y₂ →
      d = |((2 * x₂ - y₂ + 1) - (a * x₁ + y₁ + 2))| / Real.sqrt (a^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_120_x_intercept_2_parallel_lines_distance_l827_82748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_increases_l827_82712

/-- Represents the round trip time of a ferry -/
noncomputable def roundTripTime (S V Vw : ℝ) : ℝ :=
  S / (V + Vw) + S / (V - Vw)

/-- Theorem: The round trip time increases when the water flow speed increases -/
theorem round_trip_time_increases
  (S V Vw1 Vw2 : ℝ)
  (h1 : S > 0) (h2 : V > 0) (h3 : 0 ≤ Vw1) (h4 : Vw1 < V) (h5 : Vw1 < Vw2) (h6 : Vw2 < V) :
  roundTripTime S V Vw1 < roundTripTime S V Vw2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_increases_l827_82712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_zero_count_l827_82745

-- Define f₀(x)
def f₀ (x : ℝ) : ℝ := x + |x - 100| - |x + 100|

-- Define fₙ(x) recursively
def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => f₀ x
  | n + 1 => |f n x| - 1

-- Theorem statement
theorem f_100_zero_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f 100 x = 0) ∧ Finset.card S = 301 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_zero_count_l827_82745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_calculation_l827_82722

def total_bowls : ℕ := 110
def cost_per_bowl : ℚ := 10
def sold_bowls : ℕ := 100
def selling_price : ℚ := 14

def total_cost : ℚ := total_bowls * cost_per_bowl
def total_revenue : ℚ := sold_bowls * selling_price
def gain : ℚ := total_revenue - total_cost
def percentage_gain : ℚ := (gain / total_cost) * 100

theorem percentage_gain_calculation :
  ∃ (ε : ℚ), abs (percentage_gain - 27.27) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_calculation_l827_82722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_5_range_of_t_for_solution_l827_82779

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = Set.Icc (-3.5) 1.5 := by sorry

-- Theorem for the range of t
theorem range_of_t_for_solution :
  {t : ℝ | ∃ x : ℝ, t^2 + 3*t > f x} = Set.Iio (-4) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_5_range_of_t_for_solution_l827_82779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washing_machine_payment_l827_82738

theorem washing_machine_payment
  (part_payment : ℝ)
  (part_payment_percentage : ℝ)
  (sales_tax_rate : ℝ)
  (discount_rate : ℝ)
  (h1 : part_payment = 650)
  (h2 : part_payment_percentage = 0.15)
  (h3 : sales_tax_rate = 0.07)
  (h4 : discount_rate = 0.10) :
  (part_payment / part_payment_percentage * (1 - discount_rate) * (1 + sales_tax_rate) - part_payment) = 3523 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_washing_machine_payment_l827_82738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculations_l827_82713

-- Define the angles α and β as noncomputable
noncomputable def α : ℝ := Real.arctan (-4/3)
noncomputable def β : ℝ := Real.arctan (3/4)

-- State the theorem
theorem angle_calculations :
  (Real.tan α / (Real.sin (Real.pi - α) - Real.cos (Real.pi/2 + α)) = -5/6) ∧
  (β ∈ Set.Icc Real.pi (3/2 * Real.pi)) ∧
  (Real.cos (2*α - β) = 4/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculations_l827_82713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_specific_chord_length_l827_82700

/-- The ellipse equation 4x^2 + y^2 = 1 -/
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

/-- The line equation y = x + m -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The length of the chord intercepted by the ellipse on the line -/
noncomputable def chord_length (m : ℝ) : ℝ := 2 * Real.sqrt 10 / 5

theorem ellipse_line_intersection (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
by sorry

theorem specific_chord_length (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = chord_length m) →
  m = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_specific_chord_length_l827_82700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_mixture_theorem_l827_82726

noncomputable def antifreeze_mixture (volume1 : ℝ) (concentration1 : ℝ) (volume2 : ℝ) (concentration2 : ℝ) : ℝ :=
  (volume1 * concentration1 + volume2 * concentration2) / (volume1 + volume2)

theorem antifreeze_mixture_theorem : 
  antifreeze_mixture 4 0.05 8 0.20 = 0.15 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_mixture_theorem_l827_82726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_loop_coverage_l827_82715

-- Define a closed curve in 2D space
def ClosedCurve : Type := ℝ → ℝ × ℝ

-- Define the length of a curve
noncomputable def curveLength (c : ClosedCurve) : ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem string_loop_coverage (c : ClosedCurve) (L : ℝ) 
  (h1 : curveLength c = 2 * L) (h2 : L > 0) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ Set.range c → distance center p ≤ L / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_loop_coverage_l827_82715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l827_82702

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  R : ℝ  -- Radius of the outer circle
  r : ℝ  -- Radius of the inner circle
  h : R > r

/-- A point on the outer circle of the annulus. -/
def OuterPoint (a : Annulus) := { p : ℝ × ℝ // p.1^2 + p.2^2 = a.R^2 }

/-- A tangent line from a point on the outer circle to the inner circle. -/
def Tangent (a : Annulus) (p : OuterPoint a) := 
  { q : ℝ × ℝ // (q.1 - p.val.1)^2 + (q.2 - p.val.2)^2 = a.R^2 - a.r^2 }

/-- The length of the tangent line. -/
noncomputable def TangentLength (a : Annulus) (p : OuterPoint a) (t : Tangent a p) : ℝ :=
  Real.sqrt ((t.val.1 - p.val.1)^2 + (t.val.2 - p.val.2)^2)

/-- The theorem stating that the area of an annulus is π times the square of the tangent length. -/
theorem annulus_area (a : Annulus) (p : OuterPoint a) (t : Tangent a p) :
  (a.R^2 - a.r^2) * π = (TangentLength a p t)^2 * π := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l827_82702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_at_sample_point_l827_82767

/-- Regression equation: ŷ = 0.85x - 85.7 -/
def regression_equation (x : ℝ) : ℝ := 0.85 * x - 85.7

/-- Sample point x-coordinate -/
def sample_x : ℝ := 165

/-- Sample point y-coordinate -/
def sample_y : ℝ := 57

/-- Residual calculation -/
def calculate_residual (x y : ℝ) : ℝ := y - regression_equation x

theorem residual_at_sample_point :
  calculate_residual sample_x sample_y = 2.45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_at_sample_point_l827_82767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_cost_is_620_l827_82786

-- Define the original cost
noncomputable def original_cost : ℚ := 775

-- Define the discount percentage
def discount_percentage : ℚ := 20

-- Define the function to calculate the new cost after discount
noncomputable def new_cost (original : ℚ) (discount : ℚ) : ℚ :=
  original * (1 - discount / 100)

-- Theorem statement
theorem discounted_cost_is_620 :
  new_cost original_cost discount_percentage = 620 := by
  -- Unfold the definitions
  unfold new_cost
  unfold original_cost
  unfold discount_percentage
  -- Simplify the expression
  simp [mul_sub, mul_div_cancel']
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_cost_is_620_l827_82786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_sum_l827_82719

noncomputable section

-- Define the parabola P
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1/4)

-- Define the three given intersection points
def p1 : ℝ × ℝ := (-3, 9)
def p2 : ℝ × ℝ := (1, 1)
def p3 : ℝ × ℝ := (4, 16)

-- Define the Euclidean distance function
noncomputable def euclidean_distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- State the theorem
theorem intersection_points_distance_sum :
  ∃ (p4 : ℝ × ℝ),
    p4 ∈ P ∧
    p4 ≠ p1 ∧ p4 ≠ p2 ∧ p4 ≠ p3 ∧
    euclidean_distance focus p1 +
    euclidean_distance focus p2 +
    euclidean_distance focus p3 +
    euclidean_distance focus p4 = 31 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_sum_l827_82719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l827_82736

def lottery_numbers : Finset ℕ := Finset.range 10

theorem lottery_probability : 
  let total_draws := (Finset.powerset lottery_numbers).filter (λ s => Finset.card s = 6)
  let winning_numbers : Finset ℕ := {1, 2, 3, 5, 7, 8}
  let winning_draws := (Finset.powerset lottery_numbers).filter (λ s => 
    Finset.card s = 6 ∧ (Finset.card (s ∩ winning_numbers) ≥ 5))
  (Finset.card winning_draws : ℚ) / Finset.card total_draws = 5 / 42 :=
by
  sorry

#eval Finset.card (Finset.powerset (Finset.range 10))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l827_82736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_distance_on_foot_l827_82720

/-- A journey with two modes of transportation -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  foot_speed : ℝ
  bicycle_speed : ℝ

/-- Calculate the distance traveled on foot for a given journey -/
noncomputable def distance_on_foot (j : Journey) : ℝ :=
  let x := j.total_distance - (j.bicycle_speed * (j.total_time - j.total_distance / j.foot_speed))
  x / (1 - j.foot_speed / j.bicycle_speed)

/-- Theorem stating that for the specific journey described, the distance on foot is 16 km -/
theorem specific_journey_distance_on_foot :
  let j : Journey := {
    total_distance := 61,
    total_time := 9,
    foot_speed := 4,
    bicycle_speed := 9
  }
  distance_on_foot j = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_distance_on_foot_l827_82720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_game_proof_l827_82751

/-- Represents the probabilities of hitting each score on the target -/
def hit_probabilities : List ℚ := [1/5, 1/4, 1/6, 1/8, 1/10, 1/12]

/-- Represents the scores corresponding to each probability -/
def scores : List ℚ := [10, 9, 8, 7, 6, 1]

/-- The payout for hitting a score less than 6 -/
def low_score_payout : ℚ := 17/10

/-- Calculates the expected value of the game -/
def expected_value (miss_payment : ℚ) : ℚ :=
  (List.sum (List.zipWith (· * ·) hit_probabilities scores) + low_score_payout * (1/12)) -
  miss_payment * (1 - List.sum hit_probabilities)

/-- Theorem stating that the game is fair when B pays 96 forints for a miss -/
theorem fair_game_proof :
  expected_value 96 = 0 := by
  sorry

#eval expected_value 96

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_game_proof_l827_82751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_after_e_l827_82795

-- Define the function f(x) = x / ln(x)
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- Theorem statement
theorem f_monotone_increasing_after_e :
  StrictMonoOn f (Set.Ioi (Real.exp 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_after_e_l827_82795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l827_82787

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the tangent lines
def line1 (x : ℝ) : Prop := x = 2
def line2 (x y : ℝ) : Prop := 5*x - 12*y + 26 = 0

-- Define what it means for a line to be tangent to the circle
def is_tangent (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧
  ∀ (x' y' : ℝ), line x' y' → (circle_eq x' y' → x' = x ∧ y' = y)

-- Theorem statement
theorem tangent_lines_to_circle :
  (is_tangent (λ x y => line1 x) ∧ is_tangent line2) ∧
  (∀ (line : ℝ → ℝ → Prop),
    (∃ (x y : ℝ), line x y ∧ x = P.1 ∧ y = P.2) →
    is_tangent line →
    (∀ (x y : ℝ), line x y ↔ (line1 x ∨ line2 x y))) :=
sorry

#check tangent_lines_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l827_82787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l827_82764

/-- The length of a train in meters -/
noncomputable def train_length : ℝ := 750

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 90

/-- The time taken to cross the platform in seconds -/
noncomputable def crossing_time : ℝ := 60

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

/-- Theorem stating the relationship between train length, speed, and crossing time -/
theorem train_length_proof :
  train_length = train_speed * km_per_hr_to_m_per_s * crossing_time / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l827_82764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l827_82739

theorem trigonometric_problem (a β : ℝ)
  (h1 : Real.cos a = 1 / 7)
  (h2 : Real.cos (a - β) = 13 / 14)
  (h3 : 0 < β)
  (h4 : β < a)
  (h5 : a < π / 2) :
  Real.tan (2 * a) = -8 * Real.sqrt 3 / 47 ∧ β = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l827_82739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_cube_probabilities_l827_82749

/-- A cube constructed from 27 dice --/
structure LargeCube where
  small_cubes : Fin 27 → Dice
  visible_cubes : Finset (Fin 27)
  visible_cubes_count : visible_cubes.card = 26

/-- A single die --/
structure Dice where
  faces : Fin 6 → Nat
  face_values : ∀ f : Fin 6, faces f ∈ Finset.range 7 \ {0}

/-- The probability of exactly 25 faces showing six on the surface --/
noncomputable def prob_25_sixes (c : LargeCube) : ℝ :=
  62 / (6^6 * 3^12 * 2^8 : ℝ)

/-- The probability of at least one face showing one on the surface --/
noncomputable def prob_at_least_one_one (c : LargeCube) : ℝ :=
  1 - (5^6 : ℝ) / (2^2 * 3^18)

/-- The expected number of sixes showing on the surface --/
def expected_sixes : ℝ := 9

/-- The expected sum of numbers on the faces showing on the surface --/
def expected_sum : ℝ := 189

/-- The expected number of different numbers on the faces showing on the surface --/
noncomputable def expected_different_numbers : ℝ :=
  6 - (5^6 : ℝ) / (2 * 3^17)

/-- Main theorem combining all parts --/
theorem large_cube_probabilities (c : LargeCube) :
  (prob_25_sixes c = 31 / (2^13 * 3^18 : ℝ)) ∧
  (prob_at_least_one_one c > 0.99998) ∧
  (expected_sixes = 9) ∧
  (expected_sum = 189) ∧
  (expected_different_numbers > 5.99) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_cube_probabilities_l827_82749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_square_properties_l827_82756

/-- The side length of the equilateral triangle -/
noncomputable def triangle_side : ℝ := 10

/-- The height of an equilateral triangle -/
noncomputable def triangle_height (s : ℝ) : ℝ := (Real.sqrt 3 / 2) * s

/-- The side length of the square -/
noncomputable def square_side : ℝ := triangle_height triangle_side

/-- The perimeter of the triangle -/
noncomputable def triangle_perimeter : ℝ := 3 * triangle_side

/-- The area of the square -/
noncomputable def square_area : ℝ := square_side ^ 2

theorem triangle_and_square_properties :
  triangle_perimeter = 30 ∧ square_area = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_square_properties_l827_82756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_three_points_is_axiom_l827_82796

-- Define the concept of a point in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the concept of a plane
structure Plane where
  -- A plane can be defined by a point and a normal vector
  point : Point3D
  normal : Point3D

-- Define what it means for points to be collinear
def areCollinear (p q r : Point3D) : Prop := sorry

-- Define membership for Point3D in Plane
instance : Membership Point3D Plane where
  mem p plane := sorry

-- Define the axiom
axiom plane_through_three_points (p q r : Point3D) : 
  ¬areCollinear p q r → ∃! (plane : Plane), p ∈ plane ∧ q ∈ plane ∧ r ∈ plane

-- Theorem to prove
theorem plane_through_three_points_is_axiom : 
  ∀ (p q r : Point3D), ¬areCollinear p q r → 
    ∃! (plane : Plane), p ∈ plane ∧ q ∈ plane ∧ r ∈ plane :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_three_points_is_axiom_l827_82796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_MF1F2_is_right_triangle_l827_82747

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1
def C2 (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the foci F1 and F2
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the intersection point M
noncomputable def M : ℝ × ℝ := (Real.sqrt (3/2), Real.sqrt (1/2))

-- Theorem statement
theorem triangle_MF1F2_is_right_triangle :
  C1 M.1 M.2 ∧ C2 M.1 M.2 →
  let d1 := Real.sqrt ((M.1 - F1.1)^2 + (M.2 - F1.2)^2)
  let d2 := Real.sqrt ((M.1 - F2.1)^2 + (M.2 - F2.2)^2)
  let d3 := Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)
  d1^2 + d2^2 = d3^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_MF1F2_is_right_triangle_l827_82747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l827_82740

/-- Circle C in the xy-plane -/
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Line l in the xy-plane -/
def lineL (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 3 = 0

/-- Point Q -/
def Q : ℝ × ℝ := (3, 0)

/-- Distance between two points in the plane -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersection_distance_difference :
  ∃ (P₁ P₂ : ℝ × ℝ),
    circleC P₁.1 P₁.2 ∧ circleC P₂.1 P₂.2 ∧
    lineL P₁.1 P₁.2 ∧ lineL P₂.1 P₂.2 ∧
    P₁ ≠ P₂ ∧
    |distance P₁ Q - distance P₂ Q| = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l827_82740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l827_82734

-- Define the points
def A' : ℝ × ℝ := (0, 0)
def B' : ℝ × ℝ := (2, 3)
def C' : ℝ × ℝ := (5, 4)
def D' : ℝ × ℝ := (6, 1)

-- Define the quadrilateral
def quadrilateral : List (ℝ × ℝ) := [A', B', C', D']

-- Define the function to calculate the area of a quadrilateral
noncomputable def area (q : List (ℝ × ℝ)) : ℝ := sorry

-- Define the line passing through A' that cuts the quadrilateral into equal areas
noncomputable def cutting_line : ℝ → ℝ := sorry

-- Define the intersection point of the cutting line and C'D'
noncomputable def intersection_point : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_point_sum : 
  let p' : ℤ := 25
  let q' : ℤ := 6
  let r' : ℤ := 13
  let s' : ℤ := 2
  area [A', B', intersection_point] = area [intersection_point, C', D'] ∧
  p' + q' + r' + s' = 46 := by
  sorry

#check intersection_point_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l827_82734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_divides_f_l827_82783

def f (n : ℤ) : ℤ := n^2 + 5*n + 23

theorem smallest_prime_divides_f :
  ∃ (n : ℤ), (17 : ℤ) ∣ f n ∧
  ∀ (p : ℕ), p < 17 → Nat.Prime p → ∀ (m : ℤ), ¬((p : ℤ) ∣ f m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_divides_f_l827_82783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_edge_length_l827_82768

/-- Represents a sequence of alternating cubes and spheres -/
structure ContainerSequence where
  n : ℕ  -- Total number of containers
  s₁ : ℝ  -- Edge length of the largest cube

/-- The edge length of the nth cube in the sequence -/
noncomputable def edge_length (seq : ContainerSequence) (n : ℕ) : ℝ :=
  seq.s₁ / (Real.sqrt 3) ^ (n - 1)

/-- The main theorem -/
theorem smallest_cube_edge_length (seq : ContainerSequence) 
  (h1 : seq.n = 9)
  (h2 : seq.s₁ > 0) : 
  edge_length seq 5 = seq.s₁ / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_edge_length_l827_82768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l827_82753

/-- Reflection of a point (x,y) across the line y = mx + b --/
noncomputable def reflect (x y m b : ℝ) : ℝ × ℝ :=
  let x' := (x * (1 - m^2) + 2 * m * (y - b)) / (1 + m^2)
  let y' := (2 * m * x - (1 - m^2) * y + 2 * b) / (1 + m^2)
  (x', y')

/-- The theorem stating that the reflection of (2,3) across y = mx + b is (8,-1),
    and m + b = -5 --/
theorem reflection_sum (m b : ℝ) :
  reflect 2 3 m b = (8, -1) → m + b = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l827_82753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_tu_river_correct_l827_82772

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents a survey scenario -/
structure Scenario where
  description : String
  suggestedMethod : SurveyMethod

/-- Determines if a census survey is appropriate for a given scenario -/
def isCensusAppropriate (s : Scenario) : Prop :=
  (s.description = "energy-saving lamps service life") → False ∧
  (s.description = "heights of classmates") → True ∧
  (s.description = "Tu River water quality") → False ∧
  (s.description = "city-wide students' bedtime") → False

/-- Determines if a sampling survey is appropriate for a given scenario -/
def isSamplingAppropriate (s : Scenario) : Prop :=
  (s.description = "energy-saving lamps service life") → True ∧
  (s.description = "heights of classmates") → False ∧
  (s.description = "Tu River water quality") → True ∧
  (s.description = "city-wide students' bedtime") → True

/-- The list of scenarios to be evaluated -/
def scenarios : List Scenario :=
  [⟨"energy-saving lamps service life", SurveyMethod.Census⟩,
   ⟨"heights of classmates", SurveyMethod.Sampling⟩,
   ⟨"Tu River water quality", SurveyMethod.Sampling⟩,
   ⟨"city-wide students' bedtime", SurveyMethod.Census⟩]

/-- Theorem stating that only the Tu River water quality scenario correctly suggests a sampling survey -/
theorem only_tu_river_correct :
  ∃! s, s ∈ scenarios ∧ s.suggestedMethod = SurveyMethod.Sampling ∧ isSamplingAppropriate s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_tu_river_correct_l827_82772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_equals_negative_eleven_l827_82741

/-- Given an angle α whose terminal side is in the second quadrant and passes through
    the point (-4, 3), prove that (sin α - 2cos α) / (sin α + cos α) = -11 -/
theorem angle_expression_equals_negative_eleven (α : ℝ) :
  (α > π / 2 ∧ α < π) →  -- Angle in second quadrant
  (Real.sin α = 3 / 5 ∧ Real.cos α = -4 / 5) →  -- Point (-4, 3) on terminal side
  (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -11 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_equals_negative_eleven_l827_82741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_7_value_l827_82755

mutual
  def a : ℕ → ℚ
    | 0 => 1
    | (n + 1) => (a n ^ 2 + 1) / b n

  def b : ℕ → ℚ
    | 0 => 5
    | (n + 1) => (b n ^ 2 + 1) / a n
end

theorem b_7_value : b 7 = 5 ^ 1094 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_7_value_l827_82755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_enclosed_area_volume_of_revolution_l827_82791

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, 0 ≤ t ∧ t ≤ Real.pi/2 ∧ x = Real.sin t ∧ y = Real.sin (2*t)}

-- Theorem for the equation of the curve
theorem curve_equation (x y : ℝ) :
  (x, y) ∈ C → y = 2 * x * Real.sqrt (1 - x^2) := by sorry

-- Theorem for the area enclosed by the curve and x-axis
theorem enclosed_area :
  (∫ (x : ℝ) in Set.Icc 0 1, 2 * x * Real.sqrt (1 - x^2)) = 2/3 := by sorry

-- Theorem for the volume of the solid of revolution
theorem volume_of_revolution :
  (2 * Real.pi * ∫ (x : ℝ) in Set.Icc 0 1, x * (2 * x * Real.sqrt (1 - x^2))) = 8 * Real.pi / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_enclosed_area_volume_of_revolution_l827_82791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l827_82758

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the point P on the hyperbola in the first quadrant
def P (x₀ y₀ : ℝ) : Prop := hyperbola x₀ y₀ ∧ first_quadrant x₀ y₀

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define λ₁ and λ₂
noncomputable def lambda₁ (x₀ y₀ : ℝ) : ℝ := (2 + x₀) / (2 - (-12 - 7*x₀) / (4*x₀ + 7))
noncomputable def lambda₂ (x₀ y₀ : ℝ) : ℝ := (2 - x₀) / (2 - (7*x₀ - 12) / (4*x₀ - 7))

-- Define the slope k_MN
noncomputable def k_MN (x₀ y₀ : ℝ) : ℝ := -x₀ / (21 * y₀)

theorem hyperbola_properties (x₀ y₀ : ℝ) (h : P x₀ y₀) :
  lambda₁ x₀ y₀ + lambda₂ x₀ y₀ = -14 ∧ 
  ∀ k, k = k_MN x₀ y₀ → k < -Real.sqrt 3 / 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l827_82758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laboratory_painting_l827_82721

/-- Calculates the painting area and paint needed for a laboratory --/
theorem laboratory_painting (length width height : ℝ) 
  (excluded_area paint_per_sqm : ℝ) :
  length = 12 ∧ width = 8 ∧ height = 6 ∧ 
  excluded_area = 28.4 ∧ paint_per_sqm = 0.2 →
  (2 * (length * width + width * height + height * length) - length * width - excluded_area = 307.6) ∧
  ((2 * (length * width + width * height + height * length) - length * width - excluded_area) * paint_per_sqm = 61.52) := by
  sorry

#check laboratory_painting

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laboratory_painting_l827_82721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ratio_l827_82744

open BigOperators Finset

def num_balls : ℕ := 24
def num_bins : ℕ := 6

def distribution_A : List ℕ := [3, 4, 4, 5, 5, 5]
def distribution_B : List ℕ := [4, 4, 4, 4, 4, 4]

noncomputable def probability_A : ℚ := 
  (Nat.choose num_bins 1) * (Nat.choose (num_bins - 1) 2) * 
  (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5)) / 
  (num_bins ^ num_balls)

noncomputable def probability_B : ℚ := 
  (Nat.factorial num_balls / (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4)) / 
  (num_bins ^ num_balls)

theorem probability_ratio :
  probability_A / probability_B = 5 / 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ratio_l827_82744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_rotation_l827_82771

/-- A geometrical figure with n vertices where an arrow is rotated by x degrees at each vertex -/
structure GeometricalFigure where
  n : ℕ
  x : ℚ

/-- The number of full revolutions completed by the arrow -/
noncomputable def revolutions (fig : GeometricalFigure) : ℚ :=
  (fig.n : ℚ) * fig.x / 360

theorem arrow_rotation (fig : GeometricalFigure) :
  fig.n = 9 ∧ revolutions fig = 5/2 → fig.x = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_rotation_l827_82771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_seven_digit_count_l827_82797

theorem base_seven_digit_count :
  let base : ℕ := 7
  let max_num : ℕ := 2401
  let excluded_digits : Finset ℕ := {4, 5, 6}
  let count_with_excluded_digits := 
    (Finset.range max_num).filter (λ n ↦ ∃ d ∈ excluded_digits, d ∈ (n.digits base))
  ↑count_with_excluded_digits.card = 2145 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_seven_digit_count_l827_82797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hailstone_properties_l827_82718

def hailstone_sequence (k : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 =>
    let a_n := hailstone_sequence k n
    if a_n % 2 = 0 then a_n / 2 else a_n + k

theorem hailstone_properties (k : ℕ) :
  (k = 5 → hailstone_sequence k 5 = 4) ∧
  (Odd k → ∀ n, hailstone_sequence k n ≤ 2 * k) ∧
  (Even k → ∀ n, hailstone_sequence k n < hailstone_sequence k (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hailstone_properties_l827_82718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l827_82705

theorem tan_beta_value (α β : ℝ) (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
  (h2 : Real.tan (α + β) = 1 / 7) (h3 : π / 2 < α ∧ α < π) :
  Real.tan β = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l827_82705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l827_82782

/-- An odd function f with specific properties -/
def OddFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log x - a * x) ∧
  (a > (1 : ℝ) / 2) ∧
  (∃ y, y ∈ Set.Ioo (-2) 0 ∧ f y = 1 ∧ ∀ x ∈ Set.Ioo (-2) 0, f x ≥ 1)

/-- Theorem stating that for an odd function f with given properties, a = 1 -/
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) (h : OddFunction f a) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l827_82782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_student_score_fifth_student_score_exists_l827_82766

theorem fifth_student_score (scores : List ℝ) (average : ℝ) (x : ℝ) : 
  scores.length = 4 →
  scores = [75, 85, 90, 95] →
  average = 85 →
  (scores.sum + x) / 5 = average →
  x = 80 := by
  sorry

theorem fifth_student_score_exists : ∃ x : ℝ,
  ([75, 85, 90, 95].sum + x) / 5 = 85 ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_student_score_fifth_student_score_exists_l827_82766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_line_configuration_l827_82778

/-- A type representing a line in a plane -/
def Line : Type := ℝ → ℝ → Prop

/-- A type representing a point in a plane -/
def Point : Type := ℝ × ℝ

/-- A function that checks if two lines intersect at a point -/
def intersect (l1 l2 : Line) (p : Point) : Prop :=
  l1 p.1 p.2 ∧ l2 p.1 p.2

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : List Line
  intersection_points : List Point
  valid_intersection : ∀ p, p ∈ intersection_points → ∃ l1 l2, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ intersect l1 l2 p
  all_intersections : ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ∃ p, p ∈ intersection_points ∧ intersect l1 l2 p

/-- The existence of a configuration with 5 lines and 6 intersection points -/
axiom five_line_configuration : ∃ c : LineConfiguration, c.lines.length = 5 ∧ c.intersection_points.length = 6

/-- The theorem to be proved -/
theorem eight_line_configuration : ∃ c : LineConfiguration, c.lines.length = 8 ∧ c.intersection_points.length = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_line_configuration_l827_82778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_regular_hexagonal_pyramid_l827_82717

/-- A regular hexagonal pyramid with vertices on a sphere of radius 3 --/
structure RegularHexagonalPyramid where
  -- The distance from the center of the sphere to the base
  x : ℝ
  -- Constraint that x is between 0 and 3
  x_range : 0 < x ∧ x < 3

/-- The volume of a regular hexagonal pyramid --/
noncomputable def volume (p : RegularHexagonalPyramid) : ℝ :=
  (Real.sqrt 3 / 2) * (-p.x^3 - 3*p.x^2 + 9*p.x + 27)

/-- The maximum volume of a regular hexagonal pyramid with vertices on a sphere of radius 3 --/
theorem max_volume_regular_hexagonal_pyramid :
  ∃ (p : RegularHexagonalPyramid), ∀ (q : RegularHexagonalPyramid), volume p ≥ volume q ∧ volume p = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_regular_hexagonal_pyramid_l827_82717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_square_function_inequality_l827_82777

/-- A function that is odd and equals x^2 for non-negative x -/
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem odd_square_function_inequality (t : ℝ) :
  (∀ x ∈ Set.Icc t (t + 2), f (x + t) ≥ 2 * f x) → t ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_square_function_inequality_l827_82777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l827_82724

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan x - (1 / Real.tan x)

-- State the theorem
theorem period_of_f :
  ∀ x : ℝ, f (x + π/2) = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l827_82724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l827_82730

theorem sin_cos_relation (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l827_82730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l827_82780

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

-- Theorem statement
theorem value_of_a (a : ℝ) : 
  (∀ x, f' a x = -(f' a (-x))) → a = -1 := by
  intro h
  -- Apply the hypothesis to x = 0
  have h0 := h 0
  -- Simplify the equation
  simp [f'] at h0
  -- Solve for a
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l827_82780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_equals_1824_l827_82798

-- Define the enthalpies of formation (in kJ/mol)
def enthalpy_formation_NH3 : ℚ := -46
def enthalpy_formation_H2SO4 : ℚ := -814
def enthalpy_formation_NH4_2SO4 : ℚ := -909

-- Define the balanced equation coefficients
def coeff_NH3 : ℕ := 2
def coeff_H2SO4 : ℕ := 1
def coeff_NH4_2SO4 : ℕ := 1

-- Define the number of moles of Ammonia reacting
def moles_NH3_reacting : ℚ := 4

-- Define the function to calculate enthalpy change of reaction
noncomputable def enthalpy_change_reaction : ℚ :=
  coeff_NH4_2SO4 * enthalpy_formation_NH4_2SO4 -
  (coeff_NH3 * enthalpy_formation_NH3 + coeff_H2SO4 * enthalpy_formation_H2SO4)

-- Define the function to calculate total heat released
noncomputable def total_heat_released : ℚ :=
  -(moles_NH3_reacting / coeff_NH3) * enthalpy_change_reaction

-- Theorem statement
theorem heat_released_equals_1824 :
  total_heat_released = 1824 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_equals_1824_l827_82798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_divisible_by_seven_l827_82754

/-- A function that returns the first four digits of a positive integer -/
def firstFourDigits (n : ℕ) : Fin 10 × Fin 10 × Fin 10 × Fin 10 := sorry

/-- A function that checks if a positive integer starts with the digits 1137 -/
def startsWith1137 (n : ℕ) : Prop :=
  firstFourDigits n = (1, 1, 3, 7)

/-- A function that returns all permutations of the digits of a positive integer -/
def digitPermutations (n : ℕ) : Set ℕ := sorry

/-- The main theorem -/
theorem exists_permutation_divisible_by_seven (n : ℕ) 
  (h : startsWith1137 n) : 
  ∃ m ∈ digitPermutations n, m % 7 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_divisible_by_seven_l827_82754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_values_l827_82750

-- Define the function f(x) = √(-x² + 2x + 8) + 2
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 8) + 2

-- Define the line g(x) = kx + 3
def g (k : ℝ) (x : ℝ) : ℝ := k*x + 3

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem intersection_distance_implies_k_values 
  (k : ℝ) 
  (x₁ x₂ : ℝ) 
  (h₁ : f x₁ = g k x₁) 
  (h₂ : f x₂ = g k x₂) 
  (h₃ : x₁ ≠ x₂) 
  (h₄ : distance x₁ (f x₁) x₂ (f x₂) = 12 * Real.sqrt 5 / 5) :
  k = 2 ∨ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_values_l827_82750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l827_82714

open Real

/-- Given vectors OA, OB, OC, and point P, prove tan α = 4/3 and a trigonometric expression. -/
theorem vector_problem (α : ℝ) (P : ℝ × ℝ) :
  let OA : ℝ × ℝ := (sin α, 1)
  let OB : ℝ × ℝ := (cos α, 0)
  let OC : ℝ × ℝ := (-sin α, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  let BP : ℝ × ℝ := (P.1 - OB.1, P.2 - OB.2)
  -- P is on line AB with AB = BP
  (AB = BP) →
  -- O, P, and C are collinear
  (∃ (k : ℝ), P = (k * OC.1, k * OC.2)) →
  -- Part I
  (tan α = 4/3) ∧
  -- Part II
  ((sin (2*α) + sin α) / (2*cos (2*α) + 2*sin α^2 + cos α) + sin (2*α) = 172/75) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l827_82714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_triangle_area_l827_82710

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The focus of the parabola -/
def focus : Point := ⟨4, 0⟩

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- A line with slope 1 passing through the focus -/
def line (t : ℝ) : Point := ⟨t + 4, t⟩

/-- Calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem parabola_equation_and_triangle_area (para : Parabola) :
  (para.p = 8) ∧
  (∃ A B : Point,
    (A.y^2 = 16 * A.x) ∧
    (B.y^2 = 16 * B.x) ∧
    (∃ t₁ t₂ : ℝ, A = line t₁ ∧ B = line t₂) ∧
    (area_triangle origin A B = 32 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_triangle_area_l827_82710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_780_l827_82725

theorem prime_divisors_of_780 : 
  (Finset.filter (λ p => Nat.Prime p ∧ p ∣ 780) (Finset.range 781)).card = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_780_l827_82725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digit_differences_eq_495_l827_82716

/-- The first digit of a natural number -/
def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

/-- The last digit of a natural number -/
def last_digit (n : ℕ) : ℕ :=
  n % 10

/-- The sum of the differences between the first and last digits
    of all natural numbers from 1 to 999 -/
def sum_of_digit_differences : ℕ :=
  (Finset.range 1000).sum (λ n => (first_digit n - last_digit n))

theorem sum_of_digit_differences_eq_495 :
  sum_of_digit_differences = 495 := by
  sorry

#eval sum_of_digit_differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digit_differences_eq_495_l827_82716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abs_power_sum_l827_82784

theorem cube_root_abs_power_sum : (8 : ℝ) ^ (1/3) + |(-5)| + (-1) ^ 2023 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abs_power_sum_l827_82784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_unit_interval_l827_82788

/-- The function f(x) = x^2 + 2x^(-1) -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

/-- Theorem: f is decreasing on the interval (0,1] -/
theorem f_decreasing_on_unit_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≥ f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_unit_interval_l827_82788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_houses_to_build_l827_82776

/-- Calculates the remaining houses to be built given the total contracted,
    fraction built in first half, and additional houses built later. -/
theorem remaining_houses_to_build
  (total_contracted : ℕ)
  (first_half_fraction : ℚ)
  (additional_built : ℕ)
  (h1 : total_contracted = 2000)
  (h2 : first_half_fraction = 3/5)
  (h3 : additional_built = 300) :
  total_contracted - (first_half_fraction * ↑total_contracted).floor - additional_built = 500 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_houses_to_build_l827_82776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_congruence_l827_82799

theorem infinite_solutions_congruence (a b c : ℕ) (ha : a > 0) (hc : c > 0) :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ (∀ x ∈ S, (a^x + x) % c = b % c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_congruence_l827_82799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_reciprocal_at_one_l827_82709

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem derivative_of_reciprocal_at_one :
  deriv f 1 = -1 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_reciprocal_at_one_l827_82709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_value_l827_82732

/-- Given unit vectors e₁ and e₂ with an angle of 2π/3 between them, 
    if (e₁ - 2e₂) ⟂ (ke₁ + e₂), then k = 5/4 -/
theorem perpendicular_vectors_k_value 
  (e₁ e₂ : ℝ × ℝ) 
  (h_unit_e₁ : ‖e₁‖ = 1) 
  (h_unit_e₂ : ‖e₂‖ = 1) 
  (h_angle : e₁ • e₂ = -1/2) 
  (h_perp : (e₁ - 2 • e₂) • (k • e₁ + e₂) = 0) : 
  k = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_value_l827_82732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revolutions_l827_82775

-- Define constants
def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5  -- 6 inches = 0.5 feet
def gear_ratio : ℝ := 2
def front_wheel_revolutions : ℝ := 50

-- Theorem statement
theorem back_wheel_revolutions : 
  (2 * π * front_wheel_radius * front_wheel_revolutions * gear_ratio) / (2 * π * back_wheel_radius) = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revolutions_l827_82775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l827_82733

/-- Represents a mixture of water and hydrochloric acid -/
structure Mixture where
  volume : ℝ
  water_percentage : ℝ
  hcl_percentage : ℝ

/-- Calculate the water volume in the mixture -/
noncomputable def Mixture.water_volume (m : Mixture) : ℝ :=
  m.volume * m.water_percentage / 100

/-- Calculate the hydrochloric acid volume in the mixture -/
noncomputable def Mixture.hcl_volume (m : Mixture) : ℝ :=
  m.volume * m.hcl_percentage / 100

/-- The theorem to be proved -/
theorem mixture_composition 
  (original : Mixture)
  (h_original_volume : original.volume = 300)
  (h_original_water : original.water_percentage = 60)
  (h_sum_to_100 : original.water_percentage + original.hcl_percentage = 100)
  (h_added_water : ℝ)
  (h_added_water_volume : h_added_water = 100)
  (new : Mixture)
  (h_new_volume : new.volume = original.volume + h_added_water)
  (h_new_water : new.water_percentage = 70)
  (h_new_hcl : new.hcl_percentage = 30)
  (h_water_conservation : new.water_volume = original.water_volume + h_added_water)
  (h_hcl_conservation : new.hcl_volume = original.hcl_volume) :
  original.hcl_percentage = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l827_82733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_correct_l827_82743

/-- The sequence a_n defined by the given recurrence relation -/
def a : ℕ → ℝ
  | 0 => 2  -- Define a value for n = 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) + 3 * 2^(n + 1)

/-- The proposed general term formula for a_n -/
def a_formula (n : ℕ) : ℝ := (3 * n - 1) * 2^(n - 1)

/-- Theorem stating that the formula matches the sequence for all n ≥ 1 -/
theorem a_formula_correct (n : ℕ) (h : n ≥ 1) : a n = a_formula n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_correct_l827_82743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l827_82761

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := Set.Ioo (-2 : ℝ) 2

def A_real : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A_real ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l827_82761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_turns_back_to_original_direction_l827_82735

/-- Represents a turn, where the angle is in degrees and the direction is either left or right -/
structure Turn where
  angle : ℝ
  isLeft : Bool

/-- Represents a sequence of two turns -/
structure TwoTurns where
  first : Turn
  second : Turn

/-- 
  Given two turns, if they result in the car moving in its original direction,
  then the turns must be equal in magnitude but opposite in direction
-/
theorem two_turns_back_to_original_direction (turns : TwoTurns) :
  (turns.first.angle = turns.second.angle ∧ turns.first.isLeft ≠ turns.second.isLeft) ↔
  (turns.first.angle - turns.second.angle) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_turns_back_to_original_direction_l827_82735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_area_decrease_l827_82765

theorem towel_area_decrease (L W : ℝ) : 
  let original_area := L * W
  let bleached_length := 0.8 * L
  let bleached_width := 0.9 * W
  let bleached_area := bleached_length * bleached_width
  let percentage_decrease := (original_area - bleached_area) / original_area * 100
  percentage_decrease = 28 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_area_decrease_l827_82765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_third_f_theta_minus_pi_sixth_l827_82760

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 2 * sin (x - π / 12)

-- Theorem for part (1)
theorem f_pi_third : f (π / 3) = 1 := by sorry

-- Theorem for part (2)
theorem f_theta_minus_pi_sixth (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 (π / 2)) (h2 : sin θ = 4 / 5) : 
  f (θ - π / 6) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_third_f_theta_minus_pi_sixth_l827_82760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cube_edge_length_l827_82723

/-- Given a square-based straight pyramid with height m and base area A,
    if a cube is placed on the base of this pyramid such that its four vertices
    touch the pyramid's slant edges, then the length of the cube's edge is
    (m * sqrt(A)) / (m + sqrt(A)). -/
theorem pyramid_cube_edge_length (m A : ℝ) (hm : m > 0) (hA : A > 0) :
  let x := (m * Real.sqrt A) / (m + Real.sqrt A)
  ∃ (x : ℝ), x > 0 ∧ x^2 * (m + Real.sqrt A)^2 = A * m^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cube_edge_length_l827_82723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_formula_l827_82792

/-- Represents a tetrahedron with an inscribed sphere -/
structure Tetrahedron where
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Areas of the four faces -/
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := (1/3) * (t.S₁ + t.S₂ + t.S₃ + t.S₄) * t.r

/-- Theorem stating the volume formula for a tetrahedron -/
theorem tetrahedron_volume_formula (t : Tetrahedron) :
  volume t = (1/3) * (t.S₁ + t.S₂ + t.S₃ + t.S₄) * t.r := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_formula_l827_82792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_range_l827_82769

noncomputable def f (x : ℝ) : ℝ := -1/3 * x^2 + 2

def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

def range : Set ℝ := {y | ∃ x ∈ domain, f x = y}

theorem parabola_range :
  range = {y | -19/3 ≤ y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_range_l827_82769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_equations_correct_l827_82781

/-- Represents the original price per bottle in yuan -/
def x : Real := Real.mk 0  -- Placeholder value, will be solved for

/-- The price of one box of bottles in yuan -/
def box_price : ℝ := 36

/-- The discount factor applied to each bottle during the promotion -/
def discount_factor : ℝ := 0.9

/-- Theorem stating that the given equations correctly represent the promotion -/
theorem promotion_equations_correct :
  (discount_factor * (box_price + 2 * x) = box_price) ∧
  ((box_price / x) * discount_factor = box_price / (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_equations_correct_l827_82781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_point_2021_l827_82737

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (-2, -1)
def D : ℝ × ℝ := (-1, 1)
def P : ℝ × ℝ := (0, 2)

def rotate_180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

def rotation_sequence (n : ℕ) (point : ℝ × ℝ) : ℝ × ℝ :=
  match n % 4 with
  | 0 => rotate_180 D point
  | 1 => rotate_180 A point
  | 2 => rotate_180 B point
  | _ => rotate_180 C point

def final_point : ℕ → ℝ × ℝ
  | 0 => P
  | n + 1 => rotation_sequence n (final_point n)

theorem final_point_2021 :
  final_point 2021 = (-2020, 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_point_2021_l827_82737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_both_pipes_l827_82794

/-- Represents the time it takes to fill a pool using two pipes -/
noncomputable def fill_time (slower_pipe_time : ℝ) (faster_pipe_rate : ℝ) : ℝ :=
  1 / (1 / slower_pipe_time + faster_pipe_rate / slower_pipe_time)

/-- Theorem: Given the conditions, the fill time for both pipes is 4 hours -/
theorem fill_time_both_pipes :
  let slower_pipe_time : ℝ := 9
  let faster_pipe_rate : ℝ := 1.25
  fill_time slower_pipe_time faster_pipe_rate = 4 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval fill_time 9 1.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_both_pipes_l827_82794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l827_82728

noncomputable section

-- Define the inverse proportion function
def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m - 3) / x

-- Define the condition for passing through first and third quadrants
def passes_through_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

-- Theorem statement
theorem inverse_proportion_quadrants (m : ℝ) :
  passes_through_first_and_third_quadrants (inverse_proportion m) ↔ m > 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l827_82728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_purchase_amount_l827_82707

/-- The price of rice per pound -/
def rice_price : ℚ := 11/10

/-- The price of oats per pound -/
def oats_price : ℚ := 1/2

/-- The total amount of rice and oats bought in pounds -/
def total_amount : ℚ := 30

/-- The total cost of the purchase -/
def total_cost : ℚ := 47/2

/-- The amount of rice bought in pounds -/
def rice_amount : ℚ := (total_cost - oats_price * total_amount) / (rice_price - oats_price)

theorem rice_purchase_amount :
  (⌊rice_amount * 10 + 1/2⌋ : ℚ) / 10 = 71/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_purchase_amount_l827_82707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l827_82752

/-- Function to calculate the number of zeros before the first non-zero digit in a fraction -/
noncomputable def number_of_zeros_before_first_nonzero (q : ℚ) : ℕ :=
  sorry -- Implementation details omitted for brevity

/-- Theorem stating the relationship between the fraction and the number of zeros -/
theorem zeros_before_first_nonzero_digit (n m : ℕ) :
  let fraction := (1 : ℚ) / (2^n * 5^m)
  let decimal_places := m
  let leading_nonzero_digits := 2^(4-n)
  (0 < leading_nonzero_digits) ∧ (leading_nonzero_digits < 10) →
  (number_of_zeros_before_first_nonzero fraction = decimal_places - 1) :=
by sorry

/-- The specific problem instance -/
example : number_of_zeros_before_first_nonzero ((1 : ℚ) / (2^3 * 5^7)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l827_82752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_AOB_l827_82757

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculate the area of a triangle given two points in polar coordinates and the pole -/
noncomputable def triangleArea (a : PolarPoint) (b : PolarPoint) : ℝ :=
  (1/2) * a.r * b.r * Real.sin (b.θ - a.θ)

theorem triangle_area_AOB :
  let a : PolarPoint := { r := 2, θ := π/6 }
  let b : PolarPoint := { r := 4, θ := π/3 }
  triangleArea a b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_AOB_l827_82757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l827_82704

-- Define the quadrilateral
def Quadrilateral : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2 * p.2 ≤ 4 ∧ 3 * p.1 + 2 * p.2 ≥ 6 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Quadrilateral ∧ q ∈ Quadrilateral ∧
    ∀ (r s : ℝ × ℝ), r ∈ Quadrilateral → s ∈ Quadrilateral →
      distance p q ≥ distance r s ∧
      distance p q = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l827_82704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_one_proof_l827_82793

def numeral : ℚ := 135.21

def digit_appears_twice (d : ℕ) (n : ℚ) : Prop :=
  (n.num.natAbs.digits 10).count d = 2

def place_value (d : ℕ) (n : ℚ) (i : ℕ) : ℚ :=
  10 ^ ((n.num.natAbs.digits 10).length - i)

theorem digit_one_proof :
  digit_appears_twice 1 numeral ∧
  ∃ i j, i ≠ j ∧ 
    place_value 1 numeral i - place_value 1 numeral j = 99.9 := by
  sorry

#check digit_one_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_one_proof_l827_82793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_wrestling_qualifiers_l827_82785

/-- Represents the state of a tournament after a round -/
structure TournamentState where
  no_loss : ℕ  -- Number of participants with no losses
  one_loss : ℕ -- Number of participants with one loss

/-- Defines the rules and structure of the arm-wrestling tournament -/
def ArmWrestlingTournament :=
  { n : ℕ // n = 896 }

/-- Simulates a round of the tournament -/
def next_round (state : TournamentState) : TournamentState :=
  { no_loss := state.no_loss / 2,
    one_loss := state.no_loss / 2 + state.one_loss / 2 }

/-- Determines if the tournament has ended -/
def is_tournament_end (state : TournamentState) : Prop :=
  state.no_loss = 1 ∧ state.one_loss < state.no_loss

/-- The main theorem stating the number of qualifying athletes -/
theorem arm_wrestling_qualifiers (tournament : ArmWrestlingTournament) :
  ∃ (rounds : ℕ) (final_state : TournamentState),
    final_state = (Nat.iterate next_round rounds { no_loss := tournament.val, one_loss := 0 }) ∧
    is_tournament_end final_state ∧
    final_state.no_loss + final_state.one_loss = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_wrestling_qualifiers_l827_82785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_constraint_l827_82790

theorem function_extrema_constraint (a : ℝ) : 
  (∀ x : ℝ, (Real.sin x - a)^2 + 1 ≤ (1 - a)^2 + 1) ∧ 
  (∃ x : ℝ, (Real.sin x - a)^2 + 1 = (Real.sin x - Real.sin x)^2 + 1) →
  -1 ≤ a ∧ a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_constraint_l827_82790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_circle_circumference_l827_82708

-- Define the necessary functions
def is_rectangle_inscribed_in_circle (width height : ℝ) (circle : Set (ℝ × ℝ)) : Prop := sorry

def circle_circumference (circle : Set (ℝ × ℝ)) : ℝ := sorry

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle : Set (ℝ × ℝ)),
    rectangle_width = 9 →
    rectangle_height = 12 →
    is_rectangle_inscribed_in_circle rectangle_width rectangle_height circle →
    circle_circumference circle = 15 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_circle_circumference_l827_82708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_five_digits_l827_82706

/-- A nonzero digit is a natural number between 1 and 9 inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The first number in the sum -/
def num1 : ℕ := 9876

/-- The second number in the sum, parameterized by a nonzero digit A -/
def num2 (A : NonzeroDigit) : ℕ := 400 + 10 * A.val + 2

/-- The third number in the sum, parameterized by a nonzero digit B -/
def num3 (B : NonzeroDigit) : ℕ := 500 + 50 * B.val + 1

/-- The sum of the three numbers -/
def total_sum (A B : NonzeroDigit) : ℕ := num1 + num2 A + num3 B

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log n 10).succ

/-- The main theorem: the sum always has 5 digits -/
theorem sum_has_five_digits (A B : NonzeroDigit) :
  num_digits (total_sum A B) = 5 := by
  sorry

#eval num_digits (total_sum ⟨1, by norm_num⟩ ⟨1, by norm_num⟩)
#eval num_digits (total_sum ⟨9, by norm_num⟩ ⟨9, by norm_num⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_five_digits_l827_82706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l827_82763

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 6 * x + Real.pi / 3)

theorem dot_product_theorem (A B C : ℝ × ℝ) :
  -2 < A.1 ∧ A.1 < 10 →
  f A.1 = 0 →
  A.2 = 0 →
  f B.1 = B.2 →
  f C.1 = C.2 →
  (B.1 - A.1) / (B.2 - A.2) = (C.1 - A.1) / (C.2 - A.2) →
  (B.1 + C.1, B.2 + C.2) • (A.1, A.2) = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l827_82763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_unit_greater_than_tens_is_36_l827_82729

/-- A function that returns true if a two-digit number has its unit digit greater than its ten's digit -/
def unit_greater_than_tens (n : Nat) : Bool :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 10 > n / 10

/-- The count of two-digit numbers where the unit digit is greater than the ten's digit -/
def count_unit_greater_than_tens : Nat :=
  (List.range 90).filter (unit_greater_than_tens ∘ (· + 10)) |>.length

theorem count_unit_greater_than_tens_is_36 :
  count_unit_greater_than_tens = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_unit_greater_than_tens_is_36_l827_82729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_ratio_l827_82746

theorem sphere_radius_ratio (V₁ V₂ r₁ r₂ : ℝ) (h₁ : V₁ = 512 * Real.pi) (h₂ : V₂ = 128 * Real.pi) 
  (h₃ : V₂ / V₁ = (r₂ / r₁)^3) : r₂ / r₁ = 1 / (2 * (4 : ℝ) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_ratio_l827_82746
