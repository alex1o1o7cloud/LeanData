import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_production_theorem_l6_679

/-- The sum of a geometric sequence with first term a, common ratio r, and n terms -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The number of canoes produced in the first month -/
def initialProduction : ℝ := 5

/-- The ratio of production increase each month -/
def productionRatio : ℝ := 3

/-- The number of months of production -/
def numberOfMonths : ℕ := 7

/-- The total number of canoes produced over the given period -/
noncomputable def totalProduction : ℝ := geometricSum initialProduction productionRatio numberOfMonths

theorem canoe_production_theorem : totalProduction = 5465 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_production_theorem_l6_679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l6_618

/-- The time needed for q to complete the work alone -/
noncomputable def q_time (W : ℝ) (P Q R : ℝ → ℝ) : ℝ :=
  W / (Q W)

theorem work_completion_time
  (W : ℝ)
  (P Q R : ℝ → ℝ)
  (h1 : P W = Q W + R W)
  (h2 : P W + Q W = W / 10)
  (h3 : R W = W / 50) :
  q_time W P Q R = 12.5 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l6_618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l6_641

/-- Circle C: x^2 + y^2 - 8y + 12 = 0 -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

/-- Line l: ax + y + 2a = 0 -/
def lineL (a x y : ℝ) : Prop := a*x + y + 2*a = 0

/-- The line is tangent to the circle if and only if a = -3/4 -/
theorem line_tangent_to_circle :
  ∀ a : ℝ, (∀ x y : ℝ, lineL a x y → circleC x y) ↔ a = -3/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l6_641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l6_668

theorem hyperbola_parabola_intersection (a p : ℝ) : 
  a > 0 → 
  p > 0 → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1 ∧ y = (Real.sqrt 3 / a) * x ∧ x = 2 ∧ y = Real.sqrt 3) →
  (∃ (x : ℝ), x^2 / a^2 - 0^2 / 3 = 1 ∧ 0^2 = 2 * p * x) →
  p = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l6_668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l6_693

noncomputable def f (x : ℝ) : ℝ := x * (2015 + Real.log x)

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  deriv f x₀ = 2016 → x₀ = 1 := by
  intro h_deriv
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l6_693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_characterization_l6_674

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 ∧ x ≤ 2 then Real.exp (x * Real.log 2) - 1
  else if x < 0 ∧ x ≥ -2 then -(Real.exp (-x * Real.log 2) - 1)
  else 0

-- Define the properties of f
axiom f_odd : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f (-x) = -f x
axiom f_range : Set.range f = Set.Icc (-3 : ℝ) 3

-- Define the function g
def g (m : ℝ) : ℝ → ℝ := fun x => x^2 - 2*x + m

theorem m_range_characterization (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, g m x₂ = f x₁) ↔
  m ∈ Set.Icc (-5 : ℝ) (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_characterization_l6_674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l6_653

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) := {(x, y) : ℝ × ℝ | x + m * y + 6 = 0}
def l₂ (m : ℝ) := {(x, y) : ℝ × ℝ | (m - 2) * x + 3 * y + 2 * m = 0}

-- Define the conditions for intersection, perpendicularity, and parallelism
def intersect (m : ℝ) : Prop := m ≠ -1 ∧ m ≠ 3
def perpendicular (m : ℝ) : Prop := m = 1 / 2
def parallel (m : ℝ) : Prop := m = -1

-- Theorem statement
theorem line_relations :
  ∀ m : ℝ,
  ((∃ p, p ∈ l₁ m ∧ p ∈ l₂ m) ↔ intersect m) ∧
  ((∀ p q : ℝ × ℝ, p ≠ q → p ∈ l₁ m → q ∈ l₁ m →
    ∀ r s : ℝ × ℝ, r ≠ s → r ∈ l₂ m → s ∈ l₂ m →
      ((p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0)) ↔ perpendicular m) ∧
  ((∀ p q : ℝ × ℝ, p ≠ q → p ∈ l₁ m → q ∈ l₁ m →
    ∀ r s : ℝ × ℝ, r ≠ s → r ∈ l₂ m → s ∈ l₂ m →
      ((p.1 - q.1) * (s.2 - r.2) = (p.2 - q.2) * (s.1 - r.1))) ↔ parallel m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l6_653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtractions_252_72_l6_616

/-- Counts the number of subtraction operations needed to make two numbers equal -/
def count_subtractions (a b : ℕ) : ℕ :=
  match a, b with
  | 0, _ => b
  | _, 0 => a
  | a + 1, b + 1 => count_subtractions a b

/-- Theorem stating that the number of subtraction operations for 252 and 72 is 4 -/
theorem subtractions_252_72 : count_subtractions 252 72 = 4 := by
  sorry

#eval count_subtractions 252 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtractions_252_72_l6_616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l6_655

/-- Geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio
  (a : ℝ) (q : ℝ) (hq : q ≠ 0) (ha : a ≠ 0) :
  let S := geometric_sum a q
  let seq := geometric_sequence a q
  2 * S 3 = 2 * seq 1 + seq 2 →
  q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l6_655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpressionSimplification_l6_632

-- Define the expression as noncomputable
noncomputable def trigExpression (x y z : Real) : Real :=
  (Real.sin x + Real.cos y * Real.sin z) / (Real.cos x - Real.sin y * Real.sin z)

-- State the theorem
theorem trigExpressionSimplification :
  trigExpression (7 * π / 180) (15 * π / 180) (8 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpressionSimplification_l6_632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_special_angle_l6_605

/-- 
Given an angle θ with vertex at the origin, initial side on the positive x-axis,
and terminal side with slope 2 in the second quadrant, prove that cos θ = √5 / 5
-/
theorem cos_theta_special_angle (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π/2))  -- θ is in the first quadrant
  (h2 : Real.tan θ = 2) :  -- slope of terminal side is 2
  Real.cos θ = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_special_angle_l6_605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abhay_speed_l6_636

-- Define the total distance
noncomputable def total_distance : ℝ := 30

-- Define Abhay's speed as a variable
variable (speed_A : ℝ)

-- Define Sameer's speed in terms of Abhay's
noncomputable def speed_S (speed_A : ℝ) : ℝ := total_distance / (total_distance / speed_A - 2)

-- State the theorem
theorem abhay_speed (speed_A : ℝ) :
  (total_distance / speed_A = total_distance / (speed_S speed_A) + 2) ∧
  (total_distance / (2 * speed_A) = total_distance / (speed_S speed_A) - 1) →
  speed_A = 10 := by
  sorry

#check abhay_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abhay_speed_l6_636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gambler_winning_percentage_l6_617

theorem gambler_winning_percentage 
  (initial_games : ℕ) 
  (initial_win_rate : ℚ)
  (additional_games : ℕ) 
  (new_win_rate : ℚ) :
  initial_games = 20 →
  initial_win_rate = 2/5 →
  additional_games = 20 →
  new_win_rate = 4/5 →
  (((initial_win_rate * initial_games).num + (new_win_rate * additional_games).num : ℚ) / 
   (initial_games + additional_games) = 3/5) :=
by
  intro h1 h2 h3 h4
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gambler_winning_percentage_l6_617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_in_terms_of_x_l6_624

theorem tan_double_angle_in_terms_of_x (γ x : ℝ) (h1 : 0 < γ) (h2 : γ < π / 2) (h3 : x > 0)
  (h4 : Real.sin γ = Real.sqrt ((2 * x - 1) / (4 * x))) :
  Real.tan (2 * γ) = Real.sqrt (4 * x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_in_terms_of_x_l6_624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l6_619

theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → b = 1 → a = Real.sqrt 3 → 
  0 < A ∧ A < π → 0 < B ∧ B < π → 0 < C ∧ C < π →
  A + B + C = π →
  c * Real.sin A = a * Real.sin C →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l6_619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_square_l6_613

-- Define the square
structure Square where
  side : ℝ

-- Define the circle
structure Circle where
  radius : ℝ

-- Define the tangency condition
def IsTangent (c : Circle) (s : Square) : Prop := sorry

-- Define the segment cutting condition
def CutsSegment (c : Circle) (s : Square) (length : ℝ) : Prop := sorry

-- Main theorem
theorem circle_radius_in_square (c : Circle) (s : Square) :
  IsTangent c s →
  CutsSegment c s 4 →
  CutsSegment c s 2 →
  CutsSegment c s 1 →
  c.radius = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_square_l6_613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_ratio_theorem_l6_614

/-- Represents a truncated square pyramid with an inscribed rectangular parallelepiped -/
structure TruncatedPyramidWithParallelepiped where
  α : ℝ  -- Angle between pyramid's slant edge and base plane
  β : ℝ  -- Angle between parallelepiped's diagonal and its base
  hα_pos : 0 < α
  hα_lt_pi_div_two : α < π / 2
  hβ_pos : 0 < β
  hβ_lt_pi_div_two : β < π / 2

/-- Calculates the ratio of lateral surface areas (pyramid to parallelepiped) -/
noncomputable def lateral_surface_area_ratio (p : TruncatedPyramidWithParallelepiped) : ℝ :=
  (Real.sin (p.α + p.β) * Real.sqrt (2 * (1 + Real.sin p.α ^ 2))) / (2 * Real.sin p.α ^ 2 * Real.cos p.β)

/-- Theorem stating the ratio of lateral surface areas -/
theorem lateral_surface_area_ratio_theorem (p : TruncatedPyramidWithParallelepiped) :
  lateral_surface_area_ratio p = (Real.sin (p.α + p.β) * Real.sqrt (2 * (1 + Real.sin p.α ^ 2))) / (2 * Real.sin p.α ^ 2 * Real.cos p.β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_ratio_theorem_l6_614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l6_628

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 5 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y - 9 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 9| / Real.sqrt 2

-- State the theorem
theorem distance_difference :
  ∃ (max_dist min_dist : ℝ),
    (∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≤ max_dist) ∧
    (∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≥ min_dist) ∧
    (max_dist - min_dist = 2 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l6_628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_equilateral_right_prism_l6_635

/-- The maximum volume of a right prism with equilateral triangular bases -/
theorem max_volume_equilateral_right_prism :
  ∀ s h : ℝ,
  s > 0 → h > 0 →
  (Real.sqrt 3 / 4) * s^2 + 2 * s * h = 27 →
  (Real.sqrt 3 / 4) * s^2 * h ≤ 27 * Real.sqrt 3 / (Real.sqrt 3 + 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_equilateral_right_prism_l6_635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_sock_count_l6_647

/-- The number of socks John has after throwing away old socks and buying new ones -/
def final_sock_count (initial_socks : ℕ) (throw_away_percent : ℚ) (buy_fraction : ℚ) : ℕ :=
  let remaining_socks := initial_socks - (initial_socks * throw_away_percent).ceil.toNat
  remaining_socks + (remaining_socks * buy_fraction).ceil.toNat

/-- Theorem stating that John ends up with 62 socks -/
theorem john_sock_count : final_sock_count 68 (455 / 1000) (2 / 3) = 62 := by
  sorry

#eval final_sock_count 68 (455 / 1000) (2 / 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_sock_count_l6_647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l6_677

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) + Real.sqrt (15 - 3*x)

-- State the theorem
theorem f_range : 
  Set.range f = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l6_677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_necessary_not_sufficient_l6_629

-- Define Necessary and Sufficient as propositions
def Necessary (P Q : Prop) : Prop := Q → P
def Sufficient (P Q : Prop) : Prop := P → Q

theorem equality_condition_necessary_not_sufficient 
  (a₁ a₂ b₁ b₂ : ℝ) 
  (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) (hb₁ : b₁ ≠ 0) (hb₂ : b₂ ≠ 0)
  (M : Set ℝ) (hM : M = {x | a₁ * x + b₁ < 0})
  (N : Set ℝ) (hN : N = {x | a₂ * x + b₂ < 0}) :
  (Necessary (a₁ / a₂ = b₁ / b₂) (M = N)) ∧ 
  (¬ Sufficient (a₁ / a₂ = b₁ / b₂) (M = N)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_necessary_not_sufficient_l6_629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ticket_cost_l6_652

def ticket_cost (normal_price : ℝ) (num_tickets : ℕ) (discount_percent : ℝ) (service_fee_percent : ℝ) : ℝ :=
  let discounted_price := normal_price * (1 - discount_percent)
  let total_before_fee := discounted_price * (num_tickets : ℝ)
  total_before_fee * (1 + service_fee_percent)

def scalper_cost (normal_price : ℝ) (num_tickets : ℕ) (price_increase_percent : ℝ) (flat_discount : ℝ) (service_fee_percent : ℝ) : ℝ :=
  let increased_price := normal_price * (1 + price_increase_percent)
  let total_before_discount := increased_price * (num_tickets : ℝ)
  let total_after_discount := total_before_discount - flat_discount
  total_after_discount * (1 + service_fee_percent)

theorem total_ticket_cost (normal_price : ℝ) :
  normal_price = 50 →
  ticket_cost normal_price 4 0 0.12 +
  scalper_cost normal_price 6 1.75 20 0.12 +
  ticket_cost normal_price 1 0.4 0.12 +
  ticket_cost normal_price 1 0.25 0.12 +
  ticket_cost normal_price 1 0.5 0.12 +
  ticket_cost normal_price 1 0.8 0 +
  ticket_cost normal_price 1 0.9 0 = 1246 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ticket_cost_l6_652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equivalence_l6_620

theorem fraction_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^(-2 : ℤ) * b^(-2 : ℤ)) / (a^(-2 : ℤ) + b^(-2 : ℤ)) = (a^2 * b^2) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equivalence_l6_620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_M_given_N_l6_686

-- Define the number of classes and destinations
def num_classes : ℕ := 4
def num_destinations : ℕ := 4

-- Define the events (using propositions instead of strings)
def event_M : Prop := True  -- Placeholder for the actual event
def event_N : Prop := True  -- Placeholder for the actual event

-- Define the number of outcomes for event N
def n_N : ℕ := num_destinations * (num_destinations - 1)^(num_classes - 1)

-- Define the number of outcomes for event M ∩ N
def n_MN : ℕ := Nat.factorial num_classes

-- Define the conditional probability P(M|N)
noncomputable def P_M_given_N : ℚ := n_MN / n_N

-- State the theorem
theorem prob_M_given_N : P_M_given_N = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_M_given_N_l6_686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l6_642

/-- Given an ellipse C with center at the origin, foci on the coordinate axis,
    and a point M on the line y = 3/2x in the first quadrant such that:
    1. The projection of M on the x-axis is the right focus F₂.
    2. MF₁ • MF₂ = 9/4
    Prove that the equation of the ellipse is x²/4 + y²/3 = 1. -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) (M F₁ F₂ : ℝ × ℝ) :
  (0, 0) ∈ C →  -- Center at origin
  (∃ c, F₁ = (-c, 0) ∧ F₂ = (c, 0)) →  -- Foci on x-axis
  M.2 = 3/2 * M.1 →  -- M on line y = 3/2x
  M.1 = F₂.1 →  -- Projection of M on x-axis is F₂
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 9/4 →  -- MF₁ • MF₂ = 9/4
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/4 + y^2/3 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l6_642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l6_680

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^3 / Real.log 3 + Real.log x / Real.log 9 = 4 →
  x = 3^(8/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l6_680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l6_696

-- Define the function f(x) = ln x - 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x

-- Define the zero point x₀
noncomputable def x₀ : ℝ := Real.exp 1

-- Theorem statement
theorem zero_point_in_interval :
  ∃ (k : ℤ), f x₀ = 0 ∧ x₀ ∈ Set.Ico (k : ℝ) ((k : ℝ) + 1) ∧ k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l6_696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l6_650

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the equations
def median_CM_eq (x y : ℝ) : Prop := 6 * x + 10 * y - 59 = 0
def angle_bisector_B_eq (x y : ℝ) : Prop := x - 4 * y + 10 = 0

-- Define the theorem
theorem triangle_properties (ABC : Triangle) :
  ABC.A = (3, -1) →
  (∀ x y, median_CM_eq x y ↔ ∃ t : ℝ, (x, y) = (t * ABC.B.fst + (1 - t) * ABC.C.fst, t * ABC.B.snd + (1 - t) * ABC.C.snd)) →
  (∀ x y, angle_bisector_B_eq x y ↔ ∃ t : ℝ, (x, y) = (t * ABC.B.fst + (1 - t) * ABC.A.fst, t * ABC.B.snd + (1 - t) * ABC.A.snd)) →
  ABC.B = (10, 5) ∧
  (∀ x y, (∃ t : ℝ, (x, y) = (t * ABC.B.fst + (1 - t) * ABC.C.fst, t * ABC.B.snd + (1 - t) * ABC.C.snd)) ↔ 2 * x + 9 * y - 65 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l6_650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l6_678

noncomputable def f (x : ℝ) : ℝ := (3 / Real.pi) ^ (x^2 + 2*x - 3)

theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x < y → x > -1 → f y < f x :=
by
  intros x y hxy hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l6_678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_19_12_l6_637

theorem binomial_19_12 (h1 : Nat.choose 18 11 = 31824) (h2 : Nat.choose 18 12 = 18564) :
  Nat.choose 19 12 = 50388 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_19_12_l6_637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l6_651

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (1 - x) else x * (1 + x)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x ∈ Set.Icc (-1/2) (1/2) → y ∈ Set.Icc (-1/2) (1/2) → x ≤ y → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l6_651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_winning_situations_l6_644

/-- Represents a lottery ticket type -/
inductive TicketType
  | FirstPrize
  | SecondPrize
  | ThirdPrize
  | NonWinning
deriving Fintype, DecidableEq

/-- Represents a customer -/
inductive Customer
  | A
  | B
  | C
  | D
deriving Fintype, DecidableEq

/-- The total number of tickets in the lottery -/
def totalTickets : Nat := 8

/-- The number of tickets each customer draws -/
def ticketsPerCustomer : Nat := 2

/-- The number of customers -/
def numberOfCustomers : Nat := 4

/-- The number of different winning situations in the lottery -/
def winningStituations : Nat := 60

theorem lottery_winning_situations :
  (Fintype.card { t : TicketType // t ≠ TicketType.NonWinning } = 3) →
  (Fintype.card { t : TicketType // t = TicketType.NonWinning } = 5) →
  totalTickets = 8 →
  ticketsPerCustomer = 2 →
  numberOfCustomers = 4 →
  winningStituations = 60 := by
    intro h1 h2 h3 h4 h5
    sorry

#check lottery_winning_situations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_winning_situations_l6_644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l6_658

noncomputable def f (x a b : ℝ) : ℝ := (2*x + b) / (x + a)

theorem function_properties :
  ∀ a b : ℝ,
  (f 0 a b = 3/2) →
  (f (-1) a b = 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f x a b = (2*x + 3) / (x + 2)) ∧
  (∀ x y : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f x a b < f y a b) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f x a b ≥ 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f x a b ≤ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l6_658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_percentage_l6_670

/-- Calculates the profit percentage given the selling price and profit -/
noncomputable def profit_percentage (selling_price : ℝ) (profit : ℝ) : ℝ :=
  (profit / (selling_price - profit)) * 100

theorem cricket_bat_profit_percentage :
  let selling_price : ℝ := 850
  let profit : ℝ := 230
  abs (profit_percentage selling_price profit - 37.10) < 0.01 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_percentage_l6_670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_function_k_range_l6_664

/-- A function is closed if it's monotonic and there exists an interval [a,b] in its domain
    such that the range of f on [a,b] is [a,b] -/
def IsClosed' (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  (Monotone f ∨ StrictAntiOn f D) ∧
  ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.image f (Set.Icc a b) = Set.Icc a b

/-- The function f(x) = k + √x where k < 0 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k + Real.sqrt x

theorem closed_function_k_range (k : ℝ) (h : k < 0) :
  IsClosed' (f k) (Set.Ioi 0) → -1/4 < k ∧ k < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_function_k_range_l6_664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equations_classification_l6_600

-- Definition of a linear equation in one variable
def is_linear_equation_in_one_variable (eq : String) : Prop :=
  ∃ (a b : ℚ) (x : String), a ≠ 0 ∧ eq = s!"{a}{x} + {b} = 0"

-- Equations from the problem
def eq1 : String := "4x - 3 = x"
def eq2 : String := "3x(x-2) = 1"
def eq3 : String := "1 - 2a = 2a + 1"
def eq4 : String := "3a^2 = 5"
def eq5 : String := "(2x+4)/3 = 3x - 2"
def eq6 : String := "x + 1 = 1/x"
def eq7 : String := "2x - 6y = 3x - 1"
def eq8 : String := "x = 1"

-- Theorem to prove
theorem linear_equations_classification :
  is_linear_equation_in_one_variable eq1 ∧
  is_linear_equation_in_one_variable eq3 ∧
  is_linear_equation_in_one_variable eq5 ∧
  is_linear_equation_in_one_variable eq8 ∧
  ¬is_linear_equation_in_one_variable eq2 ∧
  ¬is_linear_equation_in_one_variable eq4 ∧
  ¬is_linear_equation_in_one_variable eq6 ∧
  ¬is_linear_equation_in_one_variable eq7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equations_classification_l6_600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_form_zero_area_triangle_l6_656

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- Define the roots of the polynomial
def roots : Set ℝ := {x | f x = 0}

-- Assume there are exactly three distinct real roots
axiom three_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ roots = {a, b, c}

-- Define the area of a triangle given three side lengths
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem roots_form_zero_area_triangle :
  ∃ (a b c : ℝ), a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧ triangle_area a b c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_form_zero_area_triangle_l6_656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_score_is_133_l6_688

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  average : ℚ
  highLowDiff : ℕ
  avgExcludingHighLow : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highestScore (stats : BatsmanStats) : ℚ :=
  (stats.average * stats.innings + stats.highLowDiff) / 2

/-- Theorem stating that the highest score is 133 given the specific conditions -/
theorem highest_score_is_133 (stats : BatsmanStats) 
  (h_innings : stats.innings = 46)
  (h_average : stats.average = 58)
  (h_diff : stats.highLowDiff = 150)
  (h_avg_excl : stats.avgExcludingHighLow = 58) :
  highestScore stats = 133 := by
  sorry

#eval highestScore ⟨46, 58, 150, 58⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_score_is_133_l6_688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l6_672

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem f_properties :
  ∀ x y : ℝ, x ≠ 1 → y ≠ 1 →
  (1 < x → 1 < y → x < y → f x > f y) ∧
  (3 ≤ x → x ≤ 5 → f x ≤ 2) ∧
  (3 ≤ x → x ≤ 5 → f x ≥ 3/2) ∧
  f 3 = 2 ∧
  f 5 = 3/2 := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l6_672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l6_687

/-- The eccentricity of an ellipse with equation x^2/2 + y = 1 --/
noncomputable def eccentricity_of_ellipse : ℝ := Real.sqrt 2 / 2

/-- The equation of the ellipse --/
def ellipse_equation (x y : ℝ) : Prop := x^2/2 + y = 1

theorem ellipse_eccentricity :
  ∃ (e : ℝ), e = eccentricity_of_ellipse ∧
  ∀ (x y : ℝ), ellipse_equation x y →
    e = Real.sqrt (1 - (y^2 / (x^2/2 + y^2))) := by
  sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l6_687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l6_639

def line1 (x y : ℝ) : Prop := x + y - 1 = 0

def line2 (x y : ℝ) : Prop := 2 * x + 2 * y + 1 = 0

noncomputable def distance_between_parallel_lines (a b c d e f : ℝ) : ℝ :=
  abs (c - f) / Real.sqrt (a^2 + b^2)

theorem distance_between_given_lines :
  distance_between_parallel_lines 1 1 (-1) 2 2 1 = 3 * Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l6_639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_is_six_l6_654

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- Left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- Dot product of vectors OP and FP -/
def dot_product (P : ℝ × ℝ) : ℝ := 
  let (x, y) := P
  x^2 + x + y^2

/-- Theorem: Maximum value of dot product is 6 -/
theorem max_dot_product_is_six : 
  ∃ M : ℝ, M = 6 ∧ 
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 → dot_product P ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_is_six_l6_654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_linear_combination_l6_611

theorem smallest_linear_combination (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  let m := p * q - p - q + 1
  (∀ n ≥ m, ∃ x y : ℕ, n = p * x + q * y) ∧
  (∀ k < m, ∃ n ≥ k, ∀ x y : ℕ, n ≠ p * x + q * y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_linear_combination_l6_611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_jumping_time_is_sixteen_thirds_sum_m_n_is_nineteen_l6_612

/-- A dodecagon with vertices numbered from 1 to 12 -/
def Dodecagon := Fin 12

/-- The positions of the three frogs -/
structure FrogPositions where
  frog1 : Dodecagon
  frog2 : Dodecagon
  frog3 : Dodecagon

/-- The initial positions of the frogs -/
def initialPositions : FrogPositions where
  frog1 := ⟨3, by norm_num⟩  -- Corresponds to A_4
  frog2 := ⟨7, by norm_num⟩  -- Corresponds to A_8
  frog3 := ⟨11, by norm_num⟩ -- Corresponds to A_12

/-- Function to check if two frogs are on the same vertex -/
def twoFrogsOnSameVertex (positions : FrogPositions) : Prop :=
  positions.frog1 = positions.frog2 ∨ positions.frog2 = positions.frog3 ∨ positions.frog3 = positions.frog1

/-- The expected number of minutes until the frogs stop jumping -/
def expectedJumpingTime : ℚ := 16/3

/-- Theorem stating that the expected jumping time is 16/3 -/
theorem expected_jumping_time_is_sixteen_thirds :
  expectedJumpingTime = 16/3 := by rfl

/-- The sum of m and n in the problem statement -/
def m_plus_n : ℕ := 19

/-- Theorem stating that m + n = 19 -/
theorem sum_m_n_is_nineteen :
  m_plus_n = 19 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_jumping_time_is_sixteen_thirds_sum_m_n_is_nineteen_l6_612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_at_150th_position_l6_601

def repeating_pattern : List Char := ['X', 'Y', 'Z']

theorem letter_at_150th_position (n : Nat) :
  n = 150 →
  (repeating_pattern.get! ((n - 1) % repeating_pattern.length)) = 'Z' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_at_150th_position_l6_601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_origin_l6_646

/-- Probability function for reaching (0,0) from (x,y) -/
noncomputable def P : ℕ → ℕ → ℚ := sorry

/-- The particle starts at (6,6) -/
def start_point : ℕ × ℕ := (6, 6)

/-- Movement probabilities -/
def move_prob : ℚ := 1/4

/-- Axiom: Base case for P(0,0) -/
axiom P_origin : P 0 0 = 1

/-- Axiom: Base case for P(x,0) and P(0,y) where x,y > 0 -/
axiom P_axis (x y : ℕ) : (x > 0 ∨ y > 0) → P x 0 = 0 ∧ P 0 y = 0

/-- Axiom: Recursive definition of P for x,y ≥ 1 -/
axiom P_recursive (x y : ℕ) :
  x ≥ 1 → y ≥ 1 →
  P x y = move_prob * P (x-1) y + move_prob * P x (y-1) +
          move_prob * P (x-1) (y-1) + move_prob * P (x-2) (y-1)

/-- The main theorem: P(6,6) is equal to m/(4^n) -/
theorem prob_reach_origin :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ ¬(4 ∣ m) ∧ P 6 6 = m / (4^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_origin_l6_646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_B_is_quadratic_l6_669

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation √2x² = √3x -/
noncomputable def equation_B (x : ℝ) : ℝ := Real.sqrt 2 * x^2 - Real.sqrt 3 * x

/-- Theorem: The equation √2x² = √3x is a quadratic equation -/
theorem equation_B_is_quadratic : is_quadratic_equation equation_B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_B_is_quadratic_l6_669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_function_min_value_trig_function_l6_649

-- Statement B
theorem max_value_function (x : ℝ) (h : x < 1/2) :
  ∃ y : ℝ, y = 2*x + 1/(2*x - 1) ∧ y ≤ -1 ∧ ∃ x₀ : ℝ, x₀ < 1/2 ∧ 2*x₀ + 1/(2*x₀ - 1) = -1 :=
sorry

-- Statement D
theorem min_value_trig_function :
  ∃ y : ℝ, (∀ x : ℝ, y ≤ 1/(Real.sin x)^2 + 4/(Real.cos x)^2) ∧ y = 9 ∧
  ∃ x₀ : ℝ, 1/(Real.sin x₀)^2 + 4/(Real.cos x₀)^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_function_min_value_trig_function_l6_649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_in_range_l6_676

theorem factors_in_range : ∃ n : ℕ, 
  n = (Finset.filter (λ x : ℕ ↦ 
    2000 ≤ x ∧ x < 3000 ∧ 
    x % 18 = 0 ∧ x % 24 = 0 ∧ x % 32 = 0) 
    (Finset.range 3000)).card ∧ 
  n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_in_range_l6_676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_proof_l6_604

/-- Calculates the surface area of a rectangular tank -/
noncomputable def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost per square foot of insulation -/
noncomputable def cost_per_square_foot (total_cost surface_area : ℝ) : ℝ := total_cost / surface_area

theorem insulation_cost_proof (l w h total_cost : ℝ) 
  (hl : l = 6) (hw : w = 3) (hh : h = 2) (htc : total_cost = 1440) :
  cost_per_square_foot total_cost (surface_area l w h) = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cost_per_square_foot 1440 (surface_area 6 3 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_proof_l6_604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l6_682

/-- The function f(x) = e^(3x) - x^3 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (3 * x) - x^3

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := 3 * Real.exp (3 * x) - 3 * x^2

/-- The slope of the tangent line at x = 0 -/
noncomputable def m : ℝ := f' 0

/-- The y-intercept of the tangent line -/
noncomputable def b : ℝ := f 0 - m * 0

theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * x + b ↔ 3 * x - y + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l6_682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l6_615

noncomputable section

open Real

def f (ω : ℝ) (x : ℝ) : ℝ :=
  (sin (ω * x))^2 + 2 * sqrt 3 * cos (ω * x) * sin (ω * x) +
  sin (ω * x + π/4) * sin (ω * x - π/4)

theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + π) = f ω x) :
  ω = 1 ∧
  ∀ x ∈ Set.Ioo 0 π,
    (x ∈ Set.Ioc 0 (π/3) ∨ x ∈ Set.Ico (5*π/6) π) ↔
    ∀ y ∈ Set.Ioo 0 π, y < x → f ω y < f ω x :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l6_615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_ratio_l6_610

theorem cylinder_surface_area_ratio (V : ℝ) (h r R H : ℝ) (hV : V > 0) :
  (π * r^2 * h = 2 * V) →
  (2 * π * r^2 + 2 * π * r * h ≥ 3 * (4 * π * V^2)^(1/3)) →
  (2 * π * R^2 = π * R * H) →
  H / R = 2 := by
  sorry

#check cylinder_surface_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_ratio_l6_610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l6_621

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 1)

-- State the theorem about the range of f
theorem range_of_f : 
  Set.range f = Set.Icc (0 : ℝ) (4/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l6_621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_calculation_l6_603

-- Define the spade operation
noncomputable def spade (a b : ℝ) : ℝ := (3 * a^2 / b) - (b^2 / a)

-- Theorem statement
theorem spade_calculation :
  let x := spade 2 3
  let y := spade 4 x
  abs (spade y 5 - 5541.3428) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_calculation_l6_603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l6_634

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the line l in polar form
def line_l_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 1

-- Define the line l in rectangular form
noncomputable def line_l_rect (x y : ℝ) : Prop := x + y - Real.sqrt 2 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ : ℝ, curve_C θ₁ = A ∧ curve_C θ₂ = B ∧
  line_l_rect A.1 A.2 ∧ line_l_rect B.1 B.2

-- Define point P on y-axis
noncomputable def point_P : ℝ × ℝ := (0, Real.sqrt 2)

-- State the theorem
theorem curve_line_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  line_l_rect (curve_C 0).1 (curve_C 0).2 →
  (A.1 - point_P.1) * (B.1 - point_P.1) = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l6_634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abc_is_11_l6_685

-- Define the piecewise function f
noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then a * x + 2
  else if x = 0 then a * b
  else b * x + c

-- State the theorem
theorem sum_of_abc_is_11 (a b c : ℕ) :
  f a b c 3 = 8 ∧ f a b c 0 = 6 ∧ f a b c (-3) = -15 →
  a + b + c = 11 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abc_is_11_l6_685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_l6_625

theorem sin_sum_angles (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →  -- α is acute
  0 < β ∧ β < Real.pi/2 →  -- β is acute
  Real.cos α = 12/13 → 
  Real.cos (2*α + β) = 3/5 → 
  Real.sin (α + β) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_l6_625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_approx_30_39_l6_690

/-- Represents a rectangular field with square ponds -/
structure RectField where
  width : ℝ
  length : ℝ
  pond1_side : ℝ
  pond2_side : ℝ
  pond3_side : ℝ

/-- The length of the field is thrice its width -/
def length_is_thrice_width (f : RectField) : Prop :=
  f.length = 3 * f.width

/-- The combined area of the ponds is 1/4 of the field's area -/
def ponds_area_is_quarter_field (f : RectField) : Prop :=
  (f.pond1_side ^ 2 + f.pond2_side ^ 2 + f.pond3_side ^ 2) = (1/4) * (f.length * f.width)

/-- The sides of the ponds are 6m, 5m, and 4m respectively -/
def pond_sizes (f : RectField) : Prop :=
  f.pond1_side = 6 ∧ f.pond2_side = 5 ∧ f.pond3_side = 4

/-- The main theorem stating the length of the field -/
theorem field_length_is_approx_30_39 (f : RectField) 
  (h1 : length_is_thrice_width f)
  (h2 : ponds_area_is_quarter_field f)
  (h3 : pond_sizes f) : 
  ∃ ε > 0, abs (f.length - 30.39) < ε := by
  sorry

#check field_length_is_approx_30_39

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_approx_30_39_l6_690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_11_l6_633

theorem three_digit_numbers_divisible_by_11 : 
  (Finset.filter (fun n => n % 11 = 0) (Finset.range 900 ∪ Finset.range 100)).card = 81 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_11_l6_633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_conversion_l6_666

/-- Converts polar coordinates (r, θ) to rectangular coordinates (x, y) -/
noncomputable def polar_to_rect (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given polar coordinates -/
noncomputable def given_polar : ℝ × ℝ := (7, Real.pi / 3)

/-- The expected rectangular coordinates -/
noncomputable def expected_rect : ℝ × ℝ := (3.5, 7 * Real.sqrt 3 / 2)

theorem polar_to_rect_conversion :
  polar_to_rect given_polar.1 given_polar.2 = expected_rect := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_conversion_l6_666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_probability_three_heads_one_tail_l6_602

/-- The probability of getting exactly three heads and one tail when tossing four coins simultaneously -/
noncomputable def probability_three_heads_one_tail : ℝ := 1 / 4

/-- The number of coins tossed -/
def num_coins : ℕ := 4

/-- The number of heads we're looking for -/
def num_heads : ℕ := 3

/-- The probability of getting a head on a single coin toss -/
noncomputable def prob_head : ℝ := 1 / 2

/-- The probability of getting a tail on a single coin toss -/
noncomputable def prob_tail : ℝ := 1 - prob_head

/-- Theorem stating the probability of getting exactly three heads and one tail -/
theorem prove_probability_three_heads_one_tail :
  probability_three_heads_one_tail =
    (Nat.choose num_coins num_heads : ℝ) * prob_head ^ num_heads * prob_tail ^ (num_coins - num_heads) :=
by
  sorry

#check prove_probability_three_heads_one_tail

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_probability_three_heads_one_tail_l6_602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_arc_circumference_l6_663

noncomputable section

-- Define the necessary types and functions
variable (Point : Type)
variable (on_circle : Point → Point → ℝ → Prop)
variable (angle : Point → Point → Point → ℝ)
variable (arc_length : Point → Point → Point → ℝ)

/-- Theorem: Circumference of the major arc AB given angle ACB = 40° on a circle with radius 12 -/
theorem major_arc_circumference (A B C O : Point) (r : ℝ) 
  (h1 : r = 12) 
  (h2 : on_circle A O r ∧ on_circle B O r ∧ on_circle C O r) 
  (h3 : angle A C B = 40 * π / 180) : 
  arc_length O A B = 192 * π / 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_arc_circumference_l6_663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l6_643

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x, f (x + 1) = 2 * f x) → (∀ x, f x = 2^x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l6_643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_tangent_to_BD_l6_623

-- Define the points
variable (A B C D H S T : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define right angles
def is_right_angle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the foot of perpendicular
def is_foot_of_perpendicular (H A B D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a point lying on a line segment
def lies_on_segment (P A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a point lying inside a triangle
def lies_inside_triangle (P A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define angle difference
noncomputable def angle_difference (A B C D E F : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define perpendicular bisector
def perpendicular_bisector (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define a line
def line_through (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Theorem statement
theorem circumcircle_tangent_to_BD 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_right_angle A B C)
  (h3 : is_right_angle A D C)
  (h4 : is_foot_of_perpendicular H A B D)
  (h5 : lies_on_segment S A B)
  (h6 : lies_on_segment T A D)
  (h7 : lies_inside_triangle H S C T)
  (h8 : angle_difference S H C B S C = 90)
  (h9 : angle_difference T H C D T C = 90) :
  ∃ P, P ∈ perpendicular_bisector H S ∩ perpendicular_bisector H T ∩ line_through A H :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_tangent_to_BD_l6_623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l6_698

def b : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => b (n + 2) + b (n + 1)

theorem fibonacci_series_sum :
  (∑' n : ℕ, b n / 3^(n + 1)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l6_698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l6_681

/-- 
Proves that the point (-3, π/8) in polar coordinates is equivalent 
to the point (3, 9π/8) in standard polar coordinate representation.
-/
theorem polar_coordinate_equivalence :
  let original_r : ℝ := -3
  let original_θ : ℝ := π / 8
  let standard_r : ℝ := 3
  let standard_θ : ℝ := 9 * π / 8
  (original_r * Real.cos original_θ = standard_r * Real.cos standard_θ ∧
   original_r * Real.sin original_θ = standard_r * Real.sin standard_θ) ∧ 
  standard_r > 0 ∧ 
  0 ≤ standard_θ ∧ 
  standard_θ < 2 * π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l6_681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limited_growth_functions_l6_626

/-- Definition of a limited growth function -/
def is_limited_growth (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ x, f (x + a) ≤ f x + b

/-- Function 1: f₁(x) = x² + x + 1 -/
def f₁ : ℝ → ℝ := λ x ↦ x^2 + x + 1

/-- Function 2: f₂(x) = √|x| -/
noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.sqrt (abs x)

/-- Function 3: f₃(x) = sin(x²) -/
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.sin (x^2)

/-- Theorem stating which functions are limited growth functions -/
theorem limited_growth_functions :
  ¬ is_limited_growth f₁ ∧ is_limited_growth f₂ ∧ is_limited_growth f₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limited_growth_functions_l6_626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_box_perimeter_volume_l6_671

theorem cubic_box_perimeter_volume (a : ℕ+) : 
  (12 : ℝ) * a.val = 3 * (a.val : ℝ) ^ 3 ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_box_perimeter_volume_l6_671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_11_consecutive_throws_l6_622

/-- Represents a die with a given number of sides -/
structure Die where
  sides : ℕ
  h : sides > 0

/-- Calculates the probability of rolling a specific sum with two dice -/
def probSumTwoDice (d1 d2 : Die) (sum : ℕ) : ℚ :=
  let totalOutcomes := d1.sides * d2.sides
  let favorableOutcomes := (Finset.range d1.sides).filter (λ x ↦ x + 1 ≤ sum ∧ sum - x ≤ d2.sides) |>.card
  favorableOutcomes / totalOutcomes

/-- The probability of rolling a sum of 11 on two consecutive throws
    with an 8-sided die and a 10-sided die -/
theorem prob_sum_11_consecutive_throws :
  let d8 : Die := ⟨8, by norm_num⟩
  let d10 : Die := ⟨10, by norm_num⟩
  (probSumTwoDice d8 d10 11) * (probSumTwoDice d8 d10 11) = 1 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_11_consecutive_throws_l6_622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l6_673

theorem divisors_multiple_of_five (n : ℕ) (hn : n = 5400) :
  (Finset.filter (λ d ↦ d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l6_673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l6_659

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 3 * x + 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + x * Real.exp x - 3

-- Theorem statement
theorem tangent_line_at_zero_one :
  ∃ (a b c : ℝ), 
    (∀ x y, y = f x → (x = 0 ∧ y = 1) → a * x + b * y + c = 0) ∧
    a = 2 ∧ b = 1 ∧ c = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l6_659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_intersection_l6_694

theorem triangle_vector_intersection (A B C F G Q : ℝ × ℝ × ℝ) : 
  (∃ (k : ℝ), F = B + k • (C - B) ∧ k > 1 ∧ (k - 1) / 1 = 4 / 1) →
  (∃ (t : ℝ), G = A + t • (C - A) ∧ 0 < t ∧ t < 1 ∧ t / (1 - t) = 3 / 2) →
  (∃ (s r : ℝ), Q = A + s • (F - A) ∧ Q = B + r • (G - B)) →
  Q = (8/28) • A + (-5/28) • B + (20/28) • C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_intersection_l6_694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l6_627

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def second_diagonal (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

/-- Theorem: For a rhombus with area 75 cm² and one diagonal of 10 cm, 
    the length of the other diagonal is 15 cm -/
theorem rhombus_diagonal_length :
  let r : Rhombus := { area := 75, diagonal1 := 10 }
  second_diagonal r = 15 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l6_627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_well_defined_up_to_largest_m_x_undefined_after_largest_m_largest_m_is_maximal_l6_683

/-- A sequence defined by the recurrence relation xₙ = xₙ₋₂ - 1/xₙ₋₁ for n ≥ 3,
    with initial conditions x₁ = 5 and x₂ = 401. -/
noncomputable def x : ℕ → ℝ
  | 0 => 5  -- Handle the zero case
  | 1 => 5
  | 2 => 401
  | n + 3 => x (n + 1) - 1 / x (n + 2)

/-- The largest value of m for which the sequence x is well-defined. -/
def largest_m : ℕ := 2007

theorem x_well_defined_up_to_largest_m :
  ∀ n ≤ largest_m, x n ≠ 0 ∧
  (n < largest_m → x (n + 1) * x n = 2007 - (n + 1)) :=
by sorry

theorem x_undefined_after_largest_m :
  x largest_m = 0 ∨ x (largest_m - 1) = 0 :=
by sorry

theorem largest_m_is_maximal :
  ∀ m > largest_m, ¬(∀ n ≤ m, x n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_well_defined_up_to_largest_m_x_undefined_after_largest_m_largest_m_is_maximal_l6_683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_at_pi_over_2_l6_607

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * (x - Real.pi / 6))

theorem g_value_at_pi_over_2 : g (Real.pi / 2) = -1 / 2 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [Real.cos_sub, Real.cos_two_mul, Real.sin_two_mul]
  -- The rest of the proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_at_pi_over_2_l6_607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_line_l6_697

-- Define the circle
def is_in_circle (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define the line passing through P with slope m
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Define the area difference function (this would be complex to implement, so we'll use a placeholder)
noncomputable def area_difference (m : ℝ) : ℝ := sorry

-- State the theorem
theorem max_area_difference_line :
  ∃ (m : ℝ), (∀ m' : ℝ, area_difference m ≥ area_difference m') ∧
             (∀ x y : ℝ, line_through_P m x y ↔ y = -x + 2) :=
by
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_line_l6_697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_percentage_of_pentagon_l6_695

/-- The ratio of the area of an equilateral triangle to the area of a pentagon formed by 
    placing the triangle on top of a square (sharing one side) -/
noncomputable def triangle_to_pentagon_area_ratio : ℝ :=
  (4 * Real.sqrt 3) / 13

/-- A pentagon formed by placing an equilateral triangle on top of a square, 
    such that one side of the triangle is a side of the square -/
structure Pentagon where
  side : ℝ
  side_positive : side > 0

theorem triangle_area_percentage_of_pentagon (p : Pentagon) :
  let square_area := p.side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * p.side ^ 2
  let pentagon_area := square_area + triangle_area
  triangle_area / pentagon_area = triangle_to_pentagon_area_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_percentage_of_pentagon_l6_695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_all_real_range_a_range_is_closed_interval_l6_645

/-- The function f(x) defined as the logarithm of a quadratic expression -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + 2 * x + 1)

/-- Theorem stating the range of 'a' for which f(x) is defined for all real x -/
theorem range_of_a_for_all_real_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

/-- Corollary: The range of 'a' is a closed interval [0, 1] -/
theorem a_range_is_closed_interval :
  {a : ℝ | ∀ x : ℝ, ∃ y : ℝ, f a x = y} = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_all_real_range_a_range_is_closed_interval_l6_645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_b_speed_l6_684

/-- Proves that student B's speed is 12 km/h given the problem conditions --/
theorem student_b_speed (distance speed_ratio time_difference : ℝ) : ℝ :=
  let student_b_speed : ℝ := 12
  have h1 : distance = 12 := by sorry
  have h2 : speed_ratio = 1.2 := by sorry
  have h3 : time_difference = 1/6 := by sorry
  have h4 : distance / student_b_speed - time_difference = distance / (speed_ratio * student_b_speed) := by sorry
  student_b_speed

#check student_b_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_b_speed_l6_684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_envelope_needs_extra_postage_l6_675

-- Define the envelope structure
structure Envelope where
  length : ℚ
  height : ℚ

-- Define the condition for extra postage
def needsExtraPostage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.2 || ratio > 2.8

-- Define the list of envelopes
def envelopes : List Envelope := [
  ⟨7, 5⟩,  -- Envelope A
  ⟨10, 4⟩, -- Envelope B
  ⟨8, 8⟩,  -- Envelope C
  ⟨14, 5⟩  -- Envelope D
]

-- Theorem statement
theorem one_envelope_needs_extra_postage :
  (envelopes.filter needsExtraPostage).length = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_envelope_needs_extra_postage_l6_675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l6_660

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), Real.cos x > Real.sin x) ↔
  (∀ x ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), Real.cos x ≤ Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l6_660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_theorem_l6_692

/-- The function we want to minimize and maximize -/
def f (x y : ℝ) : ℝ := |x^2 - 2*x*y|

/-- The maximum value of f(x,y) for x in [0,2] given a fixed y -/
noncomputable def g (y : ℝ) : ℝ := ⨆ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2), f x y

/-- The minimum value of g(y) over all real y -/
noncomputable def min_max_value : ℝ := ⨅ (y : ℝ), g y

/-- The theorem stating that the minimum-maximum value is 4 -/
theorem min_max_theorem : min_max_value = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_theorem_l6_692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l6_662

/-- Triangle ABC with vertices A(4, -6), B(-4, 0), and C(-1, 4) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
noncomputable def givenTriangle : Triangle :=
  { A := (4, -6)
  , B := (-4, 0)
  , C := (-1, 4)
  }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The perpendicular bisector of side BC -/
noncomputable def perpendicularBisectorBC (t : Triangle) : LineEquation :=
  { a := 3
  , b := 4
  , c := -1/2
  }

/-- The median from vertex A to side BC -/
noncomputable def medianFromA (t : Triangle) : LineEquation :=
  { a := 7
  , b := 1
  , c := 3
  }

theorem triangle_properties (t : Triangle) (h : t = givenTriangle) :
  (perpendicularBisectorBC t).a = 3 ∧
  (perpendicularBisectorBC t).b = 4 ∧
  (perpendicularBisectorBC t).c = -1/2 ∧
  (medianFromA t).a = 7 ∧
  (medianFromA t).b = 1 ∧
  (medianFromA t).c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l6_662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_one_over_98_squared_l6_609

/-- Represents the decimal expansion of a rational number -/
def DecimalExpansion (q : ℚ) : ℕ → ℕ := sorry

/-- Returns true if the decimal expansion of q is repeating -/
def isRepeating (q : ℚ) : Prop := sorry

/-- Returns the length of the repeating period in the decimal expansion of q -/
def periodLength (q : ℚ) : ℕ := sorry

/-- Returns the sum of digits in one period of the repeating decimal expansion of q -/
def sumOfDigitsInPeriod (q : ℚ) : ℕ := sorry

theorem sum_of_digits_one_over_98_squared :
  let q : ℚ := 1 / (98 * 98)
  isRepeating q ∧ sumOfDigitsInPeriod q = 916 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_one_over_98_squared_l6_609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sale_result_l6_665

/-- Represents the result of selling two items with given profit and loss percentages -/
noncomputable def net_result (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : ℝ :=
  let cost_price1 := selling_price / (1 + profit_percent / 100)
  let cost_price2 := selling_price / (1 - loss_percent / 100)
  2 * selling_price - (cost_price1 + cost_price2)

/-- Theorem stating that selling two vases for $2.40 each, with 25% profit on one and 15% loss on the other, results in a 6 cent gain -/
theorem vase_sale_result :
  net_result 2.40 25 15 = 0.06 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval net_result 2.40 25 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sale_result_l6_665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l6_638

-- Define the sector
structure Sector where
  r : ℝ
  θ : ℝ

-- Define the perimeter of the sector
noncomputable def perimeter (s : Sector) : ℝ := 2 * s.r + s.θ * s.r

-- Define the area of the sector
noncomputable def area (s : Sector) : ℝ := 1/2 * s.θ * s.r^2

-- Theorem statement
theorem sector_max_area :
  ∀ s : Sector, perimeter s = 20 →
  area s ≤ area { r := 5, θ := 2 } :=
by
  sorry

#check sector_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l6_638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l6_661

-- Define a function f with domain (-1, 1)
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Ioo (-1) 1

-- Define g(x) = f(2x-1)
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ∈ dom_f} = Set.Ioo 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l6_661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_DBCE_l6_648

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : A.1 = C.1 ∧ A.2 = C.2

-- Define the similarity relation
def Similar (t1 t2 : Triangle) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (t : Triangle) : ℝ := sorry

-- Define the area of a trapezoid
def TrapezoidArea (D B C E : ℝ × ℝ) : ℝ := sorry

theorem area_of_trapezoid_DBCE 
  (ABC : Triangle)
  (smallestTriangles : Finset Triangle)
  (D B C E : ℝ × ℝ) :
  (∀ t ∈ smallestTriangles, Similar t ABC) →
  (∀ t ∈ smallestTriangles, TriangleArea t = 1) →
  smallestTriangles.card = 9 →
  TriangleArea ABC = 45 →
  TrapezoidArea D B C E = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_DBCE_l6_648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l6_631

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | n + 1 => (3 * sequence_a n - 4) / (9 * sequence_a n + 15)

theorem sequence_a_general_term (n : ℕ) (h : n > 0) : 
  sequence_a n = (49 - 22 * n) / (33 * n - 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l6_631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l6_691

def number_of_distributions (n : ℕ) (k : ℕ) : ℕ :=
  -- Number of ways to distribute n identical objects into k distinct non-empty groups
  sorry

theorem balls_in_boxes (n k : ℕ) :
  (n ≥ k) →
  (number_of_distributions n k) = Nat.choose (n - 1) (k - 1) :=
by
  sorry

-- The specific problem instance
example : number_of_distributions 7 4 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l6_691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_true_q_false_l6_640

-- Define a as a real number
variable (a : ℝ)

-- Define proposition p
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def prop_q : Prop := ∃ x : ℝ, Real.sin x + Real.cos x = 2

-- Theorem stating that p is true and q is false
theorem p_true_q_false (a : ℝ) : prop_p a ∧ ¬prop_q :=
by
  constructor
  · -- Proof that p is true
    intro x
    -- The proof for p would go here, but we'll use sorry for now
    sorry
  · -- Proof that q is false
    intro h
    cases h with
    | intro x hx =>
      -- The proof for ¬q would go here, but we'll use sorry for now
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_true_q_false_l6_640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_pass_through_correct_l6_689

/-- The probability of a car passing through intersection k on a highway with n intersections -/
def probability_pass_through (n : ℕ) (k : ℕ) : ℚ :=
  (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

/-- The probability of a car entering the highway from any of the first k roads -/
def probability_enter (n : ℕ) (k : ℕ) : ℚ := k / n

/-- The probability of a car exiting the highway at intersection k -/
def probability_exit (n : ℕ) (k : ℕ) : ℚ := 1 / n

/-- The probability of a car exiting the highway before or at intersection k -/
def probability_exit_before (n : ℕ) (k : ℕ) : ℚ := k / n

/-- Theorem stating the probability of a car passing through intersection k -/
theorem probability_pass_through_correct (n : ℕ) (k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  probability_pass_through n k =
    (probability_enter n k + probability_exit n k * (1 - probability_exit_before n k)) +
    ((n - k + 1) / n * (1 - probability_exit_before n k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_pass_through_correct_l6_689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_calculation_l6_657

/-- The ratio of shares between b, c, and d -/
def share_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 3/2
  | 2 => 1/2

/-- The share of c in rupees -/
def c_share : ℚ := 40

/-- The total amount to be divided -/
def total_amount : ℚ := c_share * (share_ratio 0 + share_ratio 1 + share_ratio 2) / share_ratio 1

theorem total_amount_calculation :
  ∃ (ε : ℚ), abs (total_amount - 80.01) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_calculation_l6_657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l6_608

/-- The function f(x) = (3x^2 + 8x + 12) / (3x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 8*x + 12) / (3*x + 4)

/-- The proposed oblique asymptote function -/
noncomputable def g (x : ℝ) : ℝ := x + 4/3

/-- Theorem: The oblique asymptote of f(x) is y = x + 4/3 -/
theorem oblique_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l6_608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l6_699

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and subset relations
variable (parallel : Line → Line → Prop)
variable (parallelPlane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Notation for parallelism and subset
local notation:50 a " ∥ " b:50 => parallel a b
local notation:50 a " ∥ " α:50 => parallelPlane a α
local notation:50 b " ⊂ " α:50 => subset b α

-- Theorem statement
theorem all_propositions_false 
  (a b : Line) (α : Plane) : 
  (¬ ∀ (a b : Line) (α : Plane), (a ∥ b ∧ b ⊂ α) → a ∥ α) ∧ 
  (¬ ∀ (a b : Line) (α : Plane), (a ∥ b ∧ b ∥ α) → a ∥ α) ∧ 
  (¬ ∀ (a b : Line) (α : Plane), (a ∥ α ∧ b ∥ α) → a ∥ b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l6_699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cassinis_identity_l6_667

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Cassini's Identity -/
theorem cassinis_identity (n : ℕ) (hn : n > 0) : fib (n + 1) * fib (n - 1) - fib n ^ 2 = (-1 : ℤ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cassinis_identity_l6_667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1_value_l6_630

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^3 + x + 2

-- Define the arithmetic sequence a_n
noncomputable def a (a₁ : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * (1/2 : ℝ)

-- State the theorem
theorem a_1_value :
  ∃ a₁ : ℝ, 
    (f (a a₁ 1) + f (a a₁ 2) + f (a a₁ 3) + f (a a₁ 4) + f (a a₁ 5) + f (a a₁ 6) = 18) ∧
    (a₁ = -1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1_value_l6_630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l6_606

def fourth_quadrant (α : ℝ) : Prop := 
  0 < α ∧ α < Real.pi/2 ∧ Real.sin α < 0 ∧ Real.cos α > 0

theorem expression_value (α : ℝ) (h : fourth_quadrant α) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) + Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = -2 / Real.sin α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l6_606
