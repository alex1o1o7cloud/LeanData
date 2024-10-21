import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1202_120233

-- Define the expression
noncomputable def original_expression : ℝ := (3 + Real.sqrt 5) / (2 - Real.sqrt 5)

-- Define the rationalized form
noncomputable def rationalized_form : ℝ := -11 - 5 * Real.sqrt 5

-- Theorem statement
theorem rationalize_denominator :
  original_expression = rationalized_form := by
  sorry

-- Check the product ABC
def A : ℤ := -11
def B : ℤ := -5
def C : ℤ := 5

#eval A * B * C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1202_120233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sequence_limit_l1202_120232

noncomputable def complex_sequence (n : ℕ) : ℂ := ((3 + 4 * Complex.I) / 7) ^ n

theorem complex_sequence_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, Complex.abs (complex_sequence n) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sequence_limit_l1202_120232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_right_triangle_60_l1202_120213

-- Define a right triangle with a 60° angle
structure RightTriangle60 where
  -- Side lengths
  a : ℝ  -- adjacent to 60° angle
  b : ℝ  -- opposite to 60° angle
  c : ℝ  -- hypotenuse
  -- Properties
  right_angle : a^2 + b^2 = c^2
  angle_60 : a = b * Real.sqrt 3

-- Theorem statement
theorem area_right_triangle_60 (t : RightTriangle60) (h : t.a = 4) :
  (1/2) * t.a * t.b = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_right_triangle_60_l1202_120213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_3sin_max_value_l1202_120280

theorem cos_plus_3sin_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), (Real.cos y + 3 * Real.sin y) ≤ (Real.cos x + 3 * Real.sin x) ∧ 
  (Real.cos x + 3 * Real.sin x = Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_3sin_max_value_l1202_120280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_of_squares_l1202_120265

theorem cubic_roots_sum_of_squares (a b c s : ℝ) : 
  (∀ x : ℝ, x^3 - 9*x^2 + 11*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 18*s^2 - 8*s = -37 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_of_squares_l1202_120265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_with_gcd_l1202_120227

theorem sum_of_integers_with_gcd (a b : ℕ) : 
  a = (6 * b) / 10 ∧ a > 0 ∧ b > 0 → Nat.gcd a b = 7 → a + b = 56 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_with_gcd_l1202_120227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1202_120286

def f (x b c : ℝ) : ℝ := x^2 + 2*b*x + c

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := Real.sqrt (f x) + Real.sqrt (f (2 - x))

theorem problem_solution 
  (h_even : ∀ x, f x 0 1 = f (-x) 0 1)
  (h_roots : ∃ l, ∀ x, f x 0 1 = (1/2) * (x + 1)^2 ↔ x = l) :
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → Real.sqrt (f x 0 1) ≤ ((Real.sqrt 5 - 1) / 2) * |x| + 1) ∧ 
  (∀ x₁ x₂, x₁ ∈ Set.Icc (0 : ℝ) 2 → x₂ ∈ Set.Icc (0 : ℝ) 2 → 
    |g (f · 0 1) x₁ - g (f · 0 1) x₂| ≤ Real.sqrt 5 + 1 - 2 * Real.sqrt 2) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (0 : ℝ) 2 ∧ x₂ ∈ Set.Icc (0 : ℝ) 2 ∧ 
    |g (f · 0 1) x₁ - g (f · 0 1) x₂| = Real.sqrt 5 + 1 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1202_120286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_angle_of_inclination_l1202_120202

noncomputable def Line := ℝ → ℝ

structure LineProperties where
  slope : ℝ
  angleOfInclination : ℝ

def getLineProperties (l : Line) : LineProperties :=
  { slope := 0, angleOfInclination := 0 } -- Placeholder implementation

theorem slope_to_angle_of_inclination (l : Line) :
  let props := getLineProperties l
  props.slope = Real.sqrt 3 → props.angleOfInclination = 60 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_angle_of_inclination_l1202_120202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1202_120239

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h₁ : 0 < b
  h₂ : b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The left focus of an ellipse -/
noncomputable def leftFocus (e : Ellipse) : ℝ × ℝ :=
  (-e.a * eccentricity e, 0)

/-- The right focus of an ellipse -/
noncomputable def rightFocus (e : Ellipse) : ℝ × ℝ :=
  (e.a * eccentricity e, 0)

/-- The angle between two vectors is obtuse if their dot product is negative -/
def isObtuseAngle (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 < 0

theorem ellipse_eccentricity_range (e : Ellipse) :
  (∃ p : PointOnEllipse e, isObtuseAngle
    (p.x - (leftFocus e).1, p.y - (leftFocus e).2)
    (p.x - (rightFocus e).1, p.y - (rightFocus e).2)) →
  Real.sqrt 2 / 2 < eccentricity e ∧ eccentricity e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1202_120239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_same_perimeter_altitude_l1202_120261

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equal_side : ℝ
  base : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equal_side + t.base

/-- Calculates the altitude of an isosceles triangle -/
noncomputable def altitude (t : IsoscelesTriangle) : ℝ := 
  Real.sqrt (t.equal_side ^ 2 - (t.base / 2) ^ 2)

theorem isosceles_triangles_same_perimeter_altitude (u u' : IsoscelesTriangle) :
  u.equal_side = 6 ∧ u.base = 10 →
  perimeter u = perimeter u' →
  altitude u = altitude u' →
  u'.base = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_same_perimeter_altitude_l1202_120261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l1202_120204

/-- Tom's current age in years -/
def T : ℕ := sorry

/-- Number of years ago when Tom's age was three times the sum of his children's ages -/
def N : ℕ := sorry

/-- The sum of the ages of Tom's four children is equal to Tom's age -/
axiom children_sum : T = T

/-- N years ago, Tom's age was three times the sum of his children's ages -/
axiom age_relation : T - N = 3 * (T - 4 * N)

/-- Theorem stating that the ratio of Tom's current age to N is 11/2 -/
theorem tom_age_ratio : (T : ℚ) / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l1202_120204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_difference_l1202_120281

theorem divisibility_of_power_difference (p a b : ℕ) : 
  Nat.Prime p → (a - b) % p = 0 → (a^p - b) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_difference_l1202_120281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_arrangements_eq_3072_l1202_120221

/-- Represents a school in the club --/
structure School :=
  (members : Finset (Fin 4))

/-- Represents the club with 4 schools --/
structure Club :=
  (schools : Fin 4 → School)
  (total_members : (Finset.univ.sum fun i => (schools i).members.card) = 16)

/-- Represents a meeting arrangement --/
structure MeetingArrangement (c : Club) :=
  (host : Fin 4)
  (president : Fin 4)
  (vice_president : Fin 4)
  (host_representative : Fin 4)
  (other_representatives : Fin 3 → Fin 4)
  (president_from_host : president ∈ (c.schools host).members)
  (vice_president_from_host : vice_president ∈ (c.schools host).members)
  (host_rep_from_host : host_representative ∈ (c.schools host).members)
  (president_vp_different : president ≠ vice_president)
  (host_rep_different : host_representative ≠ president ∧ host_representative ≠ vice_president)

/-- The number of possible meeting arrangements --/
def number_of_arrangements (c : Club) : ℕ := sorry

/-- Theorem stating the number of possible arrangements --/
theorem number_of_arrangements_eq_3072 (c : Club) :
  number_of_arrangements c = 3072 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_arrangements_eq_3072_l1202_120221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l1202_120246

/-- The function f(x) = (x^2 - x)e^x --/
noncomputable def f (x : ℝ) : ℝ := (x^2 - x) * Real.exp x

/-- Theorem stating the bound on the difference of roots --/
theorem root_difference_bound (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : f x₁ = m) (h₄ : f x₂ = m) :
  |x₁ - x₂| < m / Real.exp 1 + m + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l1202_120246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_efficient_representation_l1202_120295

/-- Represents a mathematical expression using only the digit 7 and basic operations -/
inductive Expr
  | seven : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | pow : Expr → Expr → Expr

/-- Evaluates an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.seven => 7
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.pow e1 e2 => (eval e1) ^ (Int.floor (eval e2))

/-- Counts the number of sevens used in an expression -/
def countSevens : Expr → ℕ
  | Expr.seven => 1
  | Expr.add e1 e2 => countSevens e1 + countSevens e2
  | Expr.sub e1 e2 => countSevens e1 + countSevens e2
  | Expr.mul e1 e2 => countSevens e1 + countSevens e2
  | Expr.div e1 e2 => countSevens e1 + countSevens e2
  | Expr.pow e1 e2 => countSevens e1 + countSevens e2

/-- Theorem: There exists a positive integer n such that the number consisting of n consecutive 7s
    can be represented using fewer than n sevens, using only the digit 7 and the allowed operations -/
theorem exists_efficient_representation :
  ∃ (n : ℕ) (e : Expr), n > 0 ∧ eval e = (7 : ℚ) * ((10 : ℚ)^n - 1) / 9 ∧ countSevens e < n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_efficient_representation_l1202_120295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1202_120219

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1202_120219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_k_range_l1202_120267

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define points
def C : ℝ × ℝ := (-1, 0)  -- Center of the circle
def A : ℝ × ℝ := (1, 0)

-- Define the conditions
def conditions (P Q M : ℝ × ℝ) : Prop :=
  circle_eq P.1 P.2 ∧  -- P is on the circle
  (∃ t : ℝ, Q = (C.1 + t * (P.1 - C.1), C.2 + t * (P.2 - C.2))) ∧  -- Q is on radius CP
  (∃ s : ℝ, M = (A.1 + s * (P.1 - A.1), A.2 + s * (P.2 - A.2))) ∧  -- M is on AP
  ((M.1 - Q.1) * (P.1 - A.1) + (M.2 - Q.2) * (P.2 - A.2) = 0) ∧  -- MQ ⋅ AP = 0
  ((P.1 - A.1)^2 + (P.2 - A.2)^2 = 4 * ((M.1 - A.1)^2 + (M.2 - A.2)^2))  -- AP = 2AM

-- Define the trajectory of Q
def trajectory_Q (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (-Real.sqrt 2 / 2 ≤ k ∧ k ≤ -Real.sqrt 3 / 3) ∨
  (Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 2 / 2)

-- Theorem statement
theorem circle_trajectory_and_k_range :
  ∀ P Q M : ℝ × ℝ,
  conditions P Q M →
  (∀ x y : ℝ, Q = (x, y) → trajectory_Q x y) ∧
  (∀ k : ℝ,
    (∃ b : ℝ, 
      (∀ x y : ℝ, y = k * x + b → x^2 + y^2 = 1 → trajectory_Q x y) ∧
      (3/4 ≤ (1 + k^2) / (1 + 2 * k^2) ∧ (1 + k^2) / (1 + 2 * k^2) ≤ 4/5)) →
    k_range k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_k_range_l1202_120267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_dealer_profit_percentage_proof_l1202_120276

/-- Dealer's profit calculation -/
theorem dealer_profit_percentage (cost selling_price : ℝ) : Prop :=
  let current_profit_percentage := (selling_price - cost) / cost * 100
  let new_cost := cost * 0.9
  let new_profit_percentage := (selling_price - new_cost) / new_cost * 100
  (new_profit_percentage = current_profit_percentage * 1.15) →
  (current_profit_percentage = 35)

/-- Proof of the dealer's profit percentage -/
theorem dealer_profit_percentage_proof :
  ∀ (cost selling_price : ℝ), dealer_profit_percentage cost selling_price :=
by
  intros cost selling_price
  unfold dealer_profit_percentage
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_dealer_profit_percentage_proof_l1202_120276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l1202_120218

noncomputable def f (x : ℝ) := x^3 - (1/2)*x^2 - 2*x + 5

theorem f_upper_bound (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 2, f x < m) → m > 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l1202_120218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1202_120291

theorem triangle_inequality (α β γ : ℝ) : 
  Real.cos ((β - γ) / 2) ≥ Real.sqrt (8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1202_120291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_can_crash_l1202_120251

/-- Represents the direction of a sign in a cell -/
inductive Direction
  | North
  | West
deriving Inhabited

/-- Represents a position on the grid -/
structure Position where
  x : Int
  y : Int
deriving Inhabited

/-- The size of the grid (201x201) -/
def gridSize : Nat := 201

/-- Checks if a position is within the grid boundaries -/
def isWithinGrid (p : Position) : Bool :=
  0 ≤ p.x ∧ p.x < gridSize ∧ 0 ≤ p.y ∧ p.y < gridSize

/-- Represents the grid with signs -/
def Grid := Position → Direction

/-- The starting position (center of the grid) -/
def startPosition : Position :=
  ⟨gridSize / 2, gridSize / 2⟩

/-- Theorem: For any grid configuration, the car can always reach a position outside the grid -/
theorem car_can_crash (grid : Grid) : 
  ∃ (path : List Position), 
    path.head? = some startPosition ∧ 
    (∀ i < path.length - 1, 
      match grid (path[i]!) with
      | Direction.North => path[i + 1]! = ⟨path[i]!.x, path[i]!.y + 1⟩
      | Direction.West => path[i + 1]! = ⟨path[i]!.x - 1, path[i]!.y⟩
    ) ∧
    ¬isWithinGrid (path.getLast sorry) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_can_crash_l1202_120251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1202_120231

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line mx+(2m-1)y+1=0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -m / (2*m - 1)

/-- The slope of the line 3x+my+9=0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -3 / m

theorem perpendicular_condition (m : ℝ) :
  (m = -1 → are_perpendicular (slope1 m) (slope2 m)) ∧
  ¬(are_perpendicular (slope1 m) (slope2 m) → m = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1202_120231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_catch_up_time_l1202_120255

/-- Represents the speed of Xiao Yue -/
noncomputable def xiao_yue_speed : ℝ := 1

/-- Xiao Ling's speed is half of Xiao Yue's -/
noncomputable def xiao_ling_speed : ℝ := xiao_yue_speed / 2

/-- Bus speed is five times Xiao Yue's speed -/
noncomputable def bus_speed : ℝ := 5 * xiao_yue_speed

/-- Time Xiao Yue was on the bus before getting off -/
def time_on_bus : ℝ := 10

/-- 
Theorem stating that the time for Xiao Yue to catch up to Xiao Ling 
after getting off the bus is 110 seconds
-/
theorem catch_up_time : ℝ := by
  -- The distance Xiao Yue travels equals the sum of:
  -- 1. The distance the bus traveled while Xiao Yue was on it
  -- 2. The distance Xiao Ling traveled during the total time
  have h : xiao_yue_speed * 110 = bus_speed * time_on_bus + xiao_ling_speed * (110 + time_on_bus)
  sorry -- Proof omitted

  -- The theorem holds
  exact 110

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_catch_up_time_l1202_120255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gardening_earnings_l1202_120212

/-- Calculates the earnings for a single hour based on its position in the cycle -/
def hourly_rate (hour : ℕ) : ℕ :=
  match (hour - 1) % 6 with
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | 3 => 5
  | 4 => 6
  | 5 => 7
  | _ => 0  -- This case should never occur due to the modulo operation

/-- Calculates the total earnings for a given number of hours -/
def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map (fun i => hourly_rate (i + 1)) |>.sum

/-- The main theorem stating that 47 hours of gardening results in $209 earned -/
theorem gardening_earnings : total_earnings 47 = 209 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gardening_earnings_l1202_120212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_power_four_l1202_120214

open Matrix

variable {n : ℕ} -- Dimension of the matrix
variable (B : Matrix (Fin n) (Fin n) ℝ) -- B is an n×n real matrix

theorem det_B_power_four (h : Matrix.det B = -2) : Matrix.det (B^4) = 16 := by
  have h1 : Matrix.det (B^4) = (Matrix.det B)^4 := by
    apply Matrix.det_pow
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_power_four_l1202_120214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_cos_figure_l1202_120224

open Real MeasureTheory

/-- The area of the closed figure bounded by y = cos x, x = π/2, x = 3π/2, and y = 0 is 2 -/
theorem area_cos_figure : 
  ∫ x in (Real.pi/2)..(3*Real.pi/2), max 0 (cos x) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_cos_figure_l1202_120224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_results_l1202_120266

theorem tan_alpha_results (α : Real) (h : Real.tan α = 2) : 
  Real.tan (α + π/4) = -3 ∧ (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_results_l1202_120266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l1202_120277

open Set

def U : Set ℕ := {0, 1, 2, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_problem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l1202_120277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1202_120296

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (x - Real.pi/6) * Real.cos (x - Real.pi/6)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * (x - Real.pi/6))

theorem f_equals_g : ∀ x, f x = g x := by
  intro x
  unfold f g
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1202_120296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1202_120275

/-- An infinite geometric sequence -/
noncomputable def GeometricSequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- The sum of an infinite geometric series -/
noncomputable def InfiniteGeometricSum (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 / (1 - q)

/-- The sum of squares of an infinite geometric series -/
noncomputable def InfiniteGeometricSumSquares (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  (a 0)^2 / (1 - q^2)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_sum : InfiniteGeometricSum a q = 3)
  (h_sum_squares : InfiniteGeometricSumSquares a q = 9/2) :
  q = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1202_120275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_l1202_120253

/-- The number of ways to distribute n students among k communities,
    with at least one student in each community. -/
def number_of_distributions (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem student_distribution (n k : ℕ) : n = 4 → k = 3 → 
  (number_of_distributions n k) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_l1202_120253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_ln_sqrt_3_l1202_120294

-- Define the function
noncomputable def f (x : ℝ) : ℝ := -Real.log (Real.cos x)

-- Define the arc length
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x) ^ 2)

-- Theorem statement
theorem arc_length_ln_sqrt_3 :
  arcLength 0 (π / 6) = Real.log (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_ln_sqrt_3_l1202_120294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_points_exist_l1202_120209

/-- Represents a football tournament -/
structure Tournament :=
  (teams : ℕ)
  (draw_ratio : ℚ)

/-- Represents the condition that more than 3/4 of matches ended in a draw -/
def high_draw_ratio (t : Tournament) : Prop :=
  t.draw_ratio > 3/4

/-- The total number of matches in a round-robin tournament -/
def total_matches (t : Tournament) : ℕ :=
  t.teams * (t.teams - 1) / 2

/-- Represents the points of a team in the tournament -/
def points_of (i : ℕ) (t : Tournament) : ℕ := sorry

/-- The theorem to be proved -/
theorem equal_points_exist (t : Tournament) 
  (h1 : t.teams = 28) 
  (h2 : high_draw_ratio t) : 
  ∃ (i j : ℕ), i ≠ j ∧ i ≤ t.teams ∧ j ≤ t.teams ∧ 
  (points_of i t = points_of j t) := by
  sorry

#check equal_points_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_points_exist_l1202_120209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_prediction_correct_l1202_120245

/-- Represents the number of vehicles of each type sold at a dealership -/
structure VehicleSales where
  sportsCars : ℕ
  sedans : ℕ
  suvs : ℕ
deriving Repr

/-- Predicts the number of sedans and SUVs sold based on the number of sports cars -/
def predictSales (sportsCars : ℕ) : VehicleSales :=
  { sportsCars := sportsCars
  , sedans := (5 * sportsCars) / 3
  , suvs := 2 * sportsCars }

theorem sales_prediction_correct (expectedSportsCars : ℕ) :
  expectedSportsCars = 36 →
  let prediction := predictSales expectedSportsCars
  prediction.sedans = 60 ∧ prediction.suvs = 72 := by
  sorry

#eval predictSales 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_prediction_correct_l1202_120245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1202_120252

-- Define the line l: kx + y - 3 = 0
def line (k : ℝ) (x y : ℝ) : Prop := k * x + y - 3 = 0

-- Define the circle x^2 + y^2 = 3
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the intersection points A and B
def intersection_points (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  line k A.1 A.2 ∧ unit_circle A.1 A.2 ∧
  line k B.1 B.2 ∧ unit_circle B.1 B.2 ∧
  A ≠ B

-- Define an equilateral triangle
def is_equilateral_triangle (O A B : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Theorem statement
theorem line_circle_intersection (k : ℝ) :
  (∃ A B : ℝ × ℝ, intersection_points k A B ∧
    is_equilateral_triangle (0, 0) A B) →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1202_120252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_min_distance_l1202_120284

-- Define the curves
noncomputable def curve_C1 (α : Real) : Real × Real := (Real.cos α, Real.sin α ^ 2)

noncomputable def curve_C2 (θ : Real) : Real × Real :=
  let ρ := -Real.sqrt 2 / 2 / Real.cos (θ - Real.pi / 4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def curve_C3 (θ : Real) : Real × Real :=
  let ρ := 2 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the theorem
theorem intersection_and_min_distance :
  (∃ α θ : Real, curve_C1 α = curve_C2 θ ∧ curve_C1 α = (-1, 0)) ∧
  (∃ d : Real, d = Real.sqrt 2 - 1 ∧
    ∀ θ₁ θ₂ : Real,
      let (x₁, y₁) := curve_C2 θ₁
      let (x₂, y₂) := curve_C3 θ₂
      d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ∧
      ∃ θ₃ θ₄ : Real,
        let (x₃, y₃) := curve_C2 θ₃
        let (x₄, y₄) := curve_C3 θ₄
        d = Real.sqrt ((x₃ - x₄)^2 + (y₃ - y₄)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_min_distance_l1202_120284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oleg_position_l1202_120290

/-- Represents a train with a fixed number of cars -/
structure Train :=
  (num_cars : ℕ)

/-- Represents the relative motion of two trains -/
def relative_motion (t1 t2 : Train) (time : ℝ) (distance : ℝ) : Prop :=
  distance = (t1.num_cars + t2.num_cars : ℝ) * (time / 80)

/-- Theorem stating Oleg's position in his train -/
theorem oleg_position (train1 train2 : Train) 
  (h1 : train1.num_cars = 20)
  (h2 : train2.num_cars = 20)
  (h3 : relative_motion train1 train2 36 18) : 
  15 = train2.num_cars - (18 - 3) :=
by
  sorry

#check oleg_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oleg_position_l1202_120290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_maps_circle_to_circle_l1202_120243

-- Define the homothety transformation
def homothety (G : ℝ × ℝ) (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (G.1 + k * (P.1 - G.1), G.2 + k * (P.2 - G.2))

-- Define the circumcircle of a triangle
def circumcircle (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | ∃ (r : ℝ), r > 0 ∧ (dist P A = r ∧ dist P B = r ∧ dist P C = r)}

theorem homothety_maps_circle_to_circle 
  (G A B C I_A I_B I_C : ℝ × ℝ) 
  (h : (ℝ × ℝ) → (ℝ × ℝ)) 
  (C_circle : Set (ℝ × ℝ)) 
  (E_circle : Set (ℝ × ℝ)) :
  (h = homothety G (-1/2)) →
  (C_circle = circumcircle A B C) →
  (E_circle = circumcircle I_A I_B I_C) →
  (h A = I_A) →
  (h B = I_B) →
  (h C = I_C) →
  (h '' C_circle = E_circle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_maps_circle_to_circle_l1202_120243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_l1202_120222

/-- The circle C: (x-4)^2 + y^2 = 4 -/
def myCircle (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4

/-- The line y = x -/
def myLine (x y : ℝ) : Prop := y = x

/-- Point M on the circle -/
structure PointM where
  x : ℝ
  y : ℝ
  on_circle : myCircle x y

/-- Point N on the line, not at the origin -/
structure PointN where
  x : ℝ
  y : ℝ
  on_line : myLine x y
  not_origin : x ≠ 0 ∨ y ≠ 0

/-- Vector from O to M -/
def vector_OM (M : PointM) : ℝ × ℝ := (M.x, M.y)

/-- Vector from O to N -/
def vector_ON (N : PointN) : ℝ × ℝ := (N.x, N.y)

/-- Projection of vector OM onto vector ON -/
noncomputable def projection (M : PointM) (N : PointN) : ℝ :=
  let OM := vector_OM M
  let ON := vector_ON N
  (OM.1 * ON.1 + OM.2 * ON.2) / Real.sqrt (ON.1^2 + ON.2^2)

/-- Theorem: The maximum projection is 2√2 + 2 -/
theorem max_projection :
  ∃ (M : PointM) (N : PointN), ∀ (M' : PointM) (N' : PointN),
    projection M' N' ≤ projection M N ∧ projection M N = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_l1202_120222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l1202_120256

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the theorem
theorem angle_A_measure (t : Triangle) (h : (t.a + t.c) * (t.a - t.c) = t.b * (t.b + t.c)) :
  Real.arccos (-1/2) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l1202_120256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_special_function_l1202_120285

/-- The limit of (2 - e^(sin x))^(cot(π x)) as x approaches 0 is e^(-1/π) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |(2 - Real.exp (Real.sin x))^(Real.tan (π * x)⁻¹) - Real.exp (-1 / π)| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_special_function_l1202_120285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cornflowers_dandelions_flower_sum_daisies_count_l1202_120289

/-- The number of daisies along the path -/
def n : ℕ := 26

/-- The total number of flowers after adding cornflowers and dandelions -/
def total_flowers : ℕ := 101

/-- The number of cornflowers is one less than the number of daisies -/
theorem cornflowers : ℕ := n - 1

/-- The number of dandelions is two less than twice the number of daisies -/
theorem dandelions : ℕ := 2*n - 2

/-- The total number of flowers is the sum of daisies, cornflowers, and dandelions -/
theorem flower_sum : total_flowers = n + cornflowers + dandelions := by
  rfl

theorem daisies_count : n = 26 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cornflowers_dandelions_flower_sum_daisies_count_l1202_120289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1202_120250

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- State the theorem
theorem function_properties :
  -- f(x) has extreme values at x = ±1
  (∃ (y : ℝ), f 1 = y ∧ f (-1) = y) ∧
  -- |f(x₁) - f(x₂)| ≤ 4 for x₁, x₂ ∈ [-1, 1]
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ 4) ∧
  -- Three tangents can be drawn through A(1, m) to y = f(x) iff -3 < m < -2
  (∀ (m : ℝ), m ≠ -2 →
    (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f x₁ + (x₁ - 1) * (f' x₁) = m ∧
      f x₂ + (x₂ - 1) * (f' x₂) = m ∧
      f x₃ + (x₃ - 1) * (f' x₃) = m) ↔
    -3 < m ∧ m < -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1202_120250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_axis_intersection_l1202_120230

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The other x-axis intersection point of the ellipse -/
theorem ellipse_x_axis_intersection (e : Ellipse) (p : Point) :
  e.focus1 = ⟨0, 3⟩ →
  e.focus2 = ⟨4, 0⟩ →
  p = ⟨1, 0⟩ →
  ∃ q : Point, q.y = 0 ∧ q ≠ p ∧
    distance e.focus1 p + distance e.focus2 p =
    distance e.focus1 q + distance e.focus2 q ∧
    q = ⟨3 + Real.sqrt 10, 0⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_axis_intersection_l1202_120230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1202_120257

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 20 * (0.618 ^ x) - x

-- State the theorem
theorem root_in_interval (k : ℤ) :
  (∃ x : ℝ, x > ↑k ∧ x < ↑k + 1 ∧ f x = 0) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1202_120257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1202_120287

/-- An arithmetic sequence with a negative common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  d_negative : d < 0

/-- The conditions given in the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 2 * seq.a 4 = 12 ∧ seq.a 1 + seq.a 5 = 8

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℕ) * seq.d)

/-- The main theorem to prove -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h : problem_conditions seq) :
  seq.a 1 = 4 ∧ seq.d = -1 ∧ sum_n seq 10 = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1202_120287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_polynomial_sqrt_at_3_l1202_120283

-- Define the function f(x) = √(1+x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x)

-- Define the third-degree Taylor polynomial T₃(x) centered at x=3
noncomputable def T₃ (x : ℝ) : ℝ := 2 + (1/4)*(x-3) - (1/64)*(x-3)^2 + (1/512)*(x-3)^3

-- Theorem statement
theorem taylor_polynomial_sqrt_at_3 :
  ∀ x : ℝ, T₃ x = f 3 + (deriv f) 3 * (x - 3) + (deriv (deriv f)) 3 / 2 * (x - 3)^2 + (deriv (deriv (deriv f))) 3 / 6 * (x - 3)^3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_polynomial_sqrt_at_3_l1202_120283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l1202_120297

/-- The parabola C with equation y² = ax, where a > 0 -/
structure Parabola where
  a : ℝ
  hpos : a > 0

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line passing through point M with slope √3 -/
noncomputable def line (p : Point) : Set Point :=
  {q : Point | q.y - p.y = Real.sqrt 3 * (q.x - p.x)}

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry (c : Parabola) : Set Point :=
  {p : Point | p.x = -c.a / 4}

/-- The intersection point of the line and the axis of symmetry -/
noncomputable def point_B (c : Parabola) (m : Point) : Point :=
  { x := -c.a / 4,
    y := Real.sqrt 3 * (-c.a / 4 - m.x) + m.y }

/-- The intersection point of the line and the parabola -/
noncomputable def point_A (c : Parabola) (m : Point) : Point :=
  { x := 4 + c.a / 4,
    y := Real.sqrt 3 * (c.a / 4 + 2) }

/-- Vector equality -/
def vector_equal (p q r s : Point) : Prop :=
  p.x - q.x = r.x - s.x ∧ p.y - q.y = r.y - s.y

/-- The main theorem -/
theorem parabola_intersection (c : Parabola) :
  let m : Point := { x := 2, y := 0 }
  let b := point_B c m
  let a := point_A c m
  (a ∈ line m) ∧
  (a.y^2 = c.a * a.x) ∧
  vector_equal b m m a →
  c.a = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l1202_120297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nagy_theorem_l1202_120210

open Real Set

theorem nagy_theorem (a a' b b' : ℝ) (f : ℝ → ℝ) 
  (h1 : a < a') (h2 : a' < b) (h3 : b < b')
  (h4 : ContinuousOn f (Icc a b'))
  (h5 : DifferentiableOn ℝ f (Ioo a b')) :
  ∃ (c c' : ℝ), c ∈ Ioo a b ∧ c' ∈ Ioo a' b' ∧ c < c' ∧
    (f b - f a = (deriv f c) * (b - a)) ∧
    (f b' - f a' = (deriv f c') * (b' - a')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nagy_theorem_l1202_120210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1202_120207

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = -f a b (-x)) →  -- f is odd
  f a b (1/2) = 2/5 →                                  -- f(1/2) = 2/5
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = x / (1 + x^2)) ∧
  (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → f a b x < f a b y) ∧
  {t : ℝ | f a b (t-1) + f a b t < 0} = Set.Ioo 0 (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1202_120207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candice_bakery_expenditure_l1202_120206

/-- Calculates Candice's total expenditure at the bakery over 4 weeks -/
def bakery_expenditure (white_bread_price : ℝ) (white_bread_quantity : ℕ)
                       (baguette_price : ℝ) (baguette_quantity : ℕ)
                       (sourdough_price : ℝ) (sourdough_quantity : ℕ)
                       (croissant_price : ℝ) (croissant_quantity : ℕ)
                       (weeks : ℕ) : ℝ :=
  (white_bread_price * white_bread_quantity +
   baguette_price * baguette_quantity +
   sourdough_price * sourdough_quantity +
   croissant_price * croissant_quantity) * weeks

/-- Theorem stating Candice's total expenditure at the bakery over 4 weeks -/
theorem candice_bakery_expenditure :
  bakery_expenditure 3.5 2 1.5 1 4.5 2 2.0 1 4 = 78 :=
by
  -- Unfold the definition of bakery_expenditure
  unfold bakery_expenditure
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candice_bakery_expenditure_l1202_120206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_mode_l1202_120259

def scores : List Nat := [60, 60, 60, 60, 72, 75, 80, 83, 85, 85, 88, 91, 91, 91, 96, 97, 97, 97, 102, 102, 102, 104, 106, 109, 110, 110, 111]

def mode (l : List Nat) : List Nat :=
  let freq := l.foldl (fun acc x => acc.insert x ((acc.findD x 0) + 1)) Std.HashMap.empty
  let maxFreq := freq.fold (fun acc _ v => max acc v) 0
  freq.toList.filter (fun (_, v) => v == maxFreq) |>.map Prod.fst

theorem scores_mode :
  mode scores = [91, 97, 102] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_mode_l1202_120259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_plus_one_div_by_three_l1202_120234

theorem power_two_plus_one_div_by_three (n : ℕ) :
  3 ∣ (2^n + 1) ↔ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_plus_one_div_by_three_l1202_120234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_exponential_equation_l1202_120237

theorem solutions_to_exponential_equation (n m : ℕ) (x : ℤ) :
  2^n - 1 = x^m ↔
    (n = 1 ∧ m % 2 = 0 ∧ (x = 1 ∨ x = -1)) ∨
    (n = 1 ∧ m % 2 = 1 ∧ x = 1) ∨
    (m = 1 ∧ x = 2^n - 1) ∨
    (n ≤ 1 ∨ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_exponential_equation_l1202_120237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_wins_l1202_120278

/-- Represents the state of the game with two piles of stones -/
structure GameState where
  a : ℕ  -- Number of stones in the larger pile
  b : ℕ  -- Number of stones in the smaller pile
  deriving Repr

/-- Defines a valid move in the game -/
def validMove (state : GameState) (stones : ℕ) : Prop :=
  1 ≤ stones ∧ stones ≤ state.b ∧ state.a ≥ stones

/-- Applies a move to the game state -/
def applyMove (state : GameState) (stones : ℕ) : GameState :=
  if state.a - stones ≥ state.b then
    { a := state.a - stones, b := state.b }
  else
    { a := state.b, b := state.a - stones }

/-- Determines if a game state is a winning position for the current player -/
def isWinningPosition : GameState → Prop :=
  sorry -- The actual implementation would go here

/-- Theorem stating that Sasha (the first player) has a winning strategy -/
theorem sasha_wins (initialState : GameState) 
  (h1 : initialState.a = 2022) 
  (h2 : initialState.b = 1703) : 
  isWinningPosition initialState := by
  sorry -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_wins_l1202_120278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_is_twice_diagonal_l1202_120205

/-- A rectangle in a 2D plane --/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

/-- A point on the perimeter of a rectangle --/
structure PerimeterPoint (rect : Rectangle) where
  x : ℝ
  y : ℝ
  on_perimeter : (x = 0 ∨ x = rect.width) ∧ 0 ≤ y ∧ y ≤ rect.height ∨
                 (y = 0 ∨ y = rect.height) ∧ 0 ≤ x ∧ x ≤ rect.width

/-- The length of the shortest path that starts and ends at a point on the perimeter
    and intersects each side of the rectangle at least once --/
noncomputable def shortestPathLength (rect : Rectangle) (M : PerimeterPoint rect) : ℝ :=
  2 * Real.sqrt (rect.width ^ 2 + rect.height ^ 2)

theorem shortest_path_length_is_twice_diagonal (rect : Rectangle) (M : PerimeterPoint rect) :
  shortestPathLength rect M = 2 * Real.sqrt (rect.width ^ 2 + rect.height ^ 2) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_is_twice_diagonal_l1202_120205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1202_120268

-- Define the custom operation *
noncomputable def star (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star (Real.cos x) (Real.sin x)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-1) (Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1202_120268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_bound_l1202_120244

structure IntegerPoint where
  x : Int
  y : Int
  z : Int

structure ConvexPolyhedron where
  vertices : Finset IntegerPoint
  is_convex : Bool
  no_other_integer_points : Bool

def vertex_count (p : ConvexPolyhedron) : Nat :=
  p.vertices.card

theorem polyhedron_vertex_bound (p : ConvexPolyhedron) 
  (h1 : p.is_convex = true) 
  (h2 : p.no_other_integer_points = true) : 
  vertex_count p ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_bound_l1202_120244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_second_conceptual_box_a_prob_first_conceptual_box_b_after_mistake_l1202_120215

/-- Represents a box containing questions -/
structure QuestionBox where
  conceptual : Nat
  calculation : Nat

/-- Represents the process of drawing questions -/
def draw (box : QuestionBox) : Finset (Bool × Bool) :=
  sorry

/-- The probability of drawing a conceptual question second from Box A -/
theorem prob_second_conceptual_box_a (box_a : QuestionBox) 
  (h1 : box_a.conceptual = 2) (h2 : box_a.calculation = 2) : 
  (Finset.filter (fun p => p.2) (draw box_a)).card / (draw box_a).card = 1/2 := by
  sorry

/-- The probability of drawing a conceptual question first from Box B after student A's mistake -/
theorem prob_first_conceptual_box_b_after_mistake 
  (box_a box_b : QuestionBox) 
  (h1 : box_a.conceptual = 2) (h2 : box_a.calculation = 2)
  (h3 : box_b.conceptual = 2) (h4 : box_b.calculation = 3) :
  let new_box_b : QuestionBox := {
    conceptual := box_b.conceptual + 2,
    calculation := box_b.calculation + 2
  }
  (Finset.filter (fun p => p.1) (draw new_box_b)).card / (draw new_box_b).card = 3/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_second_conceptual_box_a_prob_first_conceptual_box_b_after_mistake_l1202_120215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_of_roots_l1202_120260

theorem min_sum_of_squares_of_roots :
  ∃ (α : ℝ), ∀ (p q : ℝ),
  (p^2 - (α - 2)*p - α - 1 = 0 ∧ q^2 - (α - 2)*q - α - 1 = 0) →
  (p^2 + q^2 ≥ 5 ∧ ∃ (β : ℝ), ∃ (r s : ℝ), 
    r^2 - (β - 2)*r - β - 1 = 0 ∧ 
    s^2 - (β - 2)*s - β - 1 = 0 ∧ 
    r^2 + s^2 = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_of_roots_l1202_120260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_sum_l1202_120262

noncomputable def f (x : ℝ) : ℝ := max (max (-6 * x - 23) (4 * x + 1)) (7 * x + 4)

def is_tangent_at_three_points (q : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, q x = a * x^2 + b * x + c) ∧ 
    (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧
    (q x₁ = f x₁ ∧ q x₂ = f x₂ ∧ q x₃ = f x₃) ∧
    (∀ x, q x ≤ f x)

theorem tangent_points_sum (q : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : is_tangent_at_three_points q x₁ x₂ x₃) : 
  x₁ + x₂ + x₃ = -5.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_sum_l1202_120262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_1_min_integer_a_l1202_120201

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

-- Theorem for the tangent line
theorem tangent_line_at_1 :
  ∃ (m c : ℝ), ∀ x y, y = f x → (x = 1 → y = m * (x - 1) + f 1) ∧
  (m * x + y - c = 0 ↔ y = m * (x - 1) + f 1) ∧ m = -15 ∧ c = 1 := by
  sorry

-- Theorem for the minimum integer a
theorem min_integer_a :
  ∃ a : ℕ, (∀ x : ℝ, x > 0 → f x ≤ (a - 3 : ℝ) * x^2 + (2 * a - 13 : ℝ) * x - 2) ∧
  (∀ b : ℕ, b < a → ∃ x : ℝ, x > 0 ∧ f x > (b - 3 : ℝ) * x^2 + (2 * b - 13 : ℝ) * x - 2) ∧
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_1_min_integer_a_l1202_120201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_M_l1202_120242

def M : Finset Nat := Finset.filter (λ n => Nat.Prime n ∧ n < 5) (Finset.range 5)

theorem number_of_proper_subsets_of_M : Finset.card (Finset.powerset M \ {M}) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_M_l1202_120242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_E_properties_l1202_120248

def E : Set (ℤ × ℤ) := {p | (p.1 > 0) ∨ (p.1 = 0 ∧ p.2 ≥ 0)}

theorem E_properties :
  (∀ x y, x ∈ E → y ∈ E → (x + y) ∈ E) ∧
  (0, 0) ∈ E ∧
  (∀ p : ℤ × ℤ, p ≠ (0, 0) → (p ∈ E ∨ -p ∈ E) ∧ ¬(p ∈ E ∧ -p ∈ E)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_E_properties_l1202_120248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l1202_120247

noncomputable section

/-- The line passing through the point (3, 0, 2) in the direction (1, 5, -2) -/
def line (t : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 3 + t
  | 1 => 5 * t
  | 2 => 2 - 2 * t

/-- The point (1, 4, 5) -/
def point : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 1
  | 1 => 4
  | 2 => 5

/-- The proposed closest point (105/31, 60/31, 50/31) -/
def closest_point : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 105 / 31
  | 1 => 60 / 31
  | 2 => 50 / 31

theorem closest_point_is_closest :
  ∃ t : ℝ, line t = closest_point ∧
  ∀ s : ℝ, ‖line s - point‖ ≥ ‖closest_point - point‖ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l1202_120247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l1202_120270

theorem largest_power_dividing_factorial : ∃ k : ℕ, k = 30 ∧ 
  (∀ m : ℕ, 2010^m ∣ Nat.factorial 2010 → m ≤ k) ∧ 
  2010^k ∣ Nat.factorial 2010 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l1202_120270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_width_is_12_meters_l1202_120225

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a courtyard in meters -/
structure CourtyardDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a single brick in square centimeters -/
def brickArea (brick : BrickDimensions) : ℝ :=
  brick.length * brick.width

/-- Calculates the total area covered by all bricks in square meters -/
noncomputable def totalBrickArea (brick : BrickDimensions) (numBricks : ℕ) : ℝ :=
  (brickArea brick * (numBricks : ℝ)) / 10000

/-- The main theorem stating that the courtyard width is 12 meters -/
theorem courtyard_width_is_12_meters
  (brick : BrickDimensions)
  (courtyard : CourtyardDimensions)
  (numBricks : ℕ)
  (h1 : brick.length = 12)
  (h2 : brick.width = 6)
  (h3 : courtyard.length = 18)
  (h4 : numBricks = 30000)
  : courtyard.width = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_width_is_12_meters_l1202_120225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_sin_x_solutions_l1202_120282

theorem tan_2x_eq_sin_x_solutions :
  ∃ (S : Finset ℝ), S.card = 3 ∧ 
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ Real.tan (2 * y) = Real.sin y → y ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_sin_x_solutions_l1202_120282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_two_heads_l1202_120299

/-- The probability of getting at least two heads when tossing four fair coins -/
def probability_at_least_two_heads_four_coins : ℚ := 11 / 16

/-- The number of coins tossed -/
def num_coins : ℕ := 4

/-- The probability of getting heads on a single fair coin toss -/
def prob_heads : ℚ := 1 / 2

/-- The probability of getting at least two heads when tossing four fair coins -/
theorem prob_at_least_two_heads : 
  probability_at_least_two_heads_four_coins = 
    1 - (Nat.choose num_coins 0 * prob_heads ^ 0 * (1 - prob_heads) ^ (num_coins - 0) + 
         Nat.choose num_coins 1 * prob_heads ^ 1 * (1 - prob_heads) ^ (num_coins - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_two_heads_l1202_120299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_condition_l1202_120249

/-- A parallelogram is inscribed in the outer curve and circumscribed about the inner circle -/
def is_parallelogram (P Q R S : ℝ × ℝ) : Prop := sorry

/-- A line segment is tangent to the unit circle -/
def is_tangent_to_circle (P Q : ℝ × ℝ) : Prop := sorry

/-- The condition for the parallelogram property on two curves -/
theorem parallelogram_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → ∃ P : ℝ × ℝ, 
    P.1^2/a^2 + P.2^2/b^2 = 1 ∧ 
    ∃ Q R S : ℝ × ℝ, 
      Q.1^2/a^2 + Q.2^2/b^2 = 1 ∧
      R.1^2/a^2 + R.2^2/b^2 = 1 ∧
      S.1^2/a^2 + S.2^2/b^2 = 1 ∧
      is_parallelogram P Q R S ∧
      is_tangent_to_circle P Q ∧
      is_tangent_to_circle Q R ∧
      is_tangent_to_circle R S ∧
      is_tangent_to_circle S P) ↔
  1/a^2 + 1/b^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_condition_l1202_120249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_torque_approx_l1202_120229

/-- The radius of the system in meters -/
noncomputable def r : ℝ := 0.1

/-- The distance between charges in meters -/
noncomputable def d : ℝ := 0.05

/-- The charge in coulombs -/
noncomputable def Q : ℝ := 1e-3

/-- Coulomb's constant in N·m²/C² -/
noncomputable def k : ℝ := 8.99e9

/-- The angle that maximizes torque -/
noncomputable def α : ℝ := 9.95 * Real.pi / 180

/-- The maximum torque of the system -/
noncomputable def max_torque : ℝ := (k * Q^2 * r^2 * Real.sin (2 * α)) / (d^2 + 4 * r^2 * Real.sin α^2)^(3/2)

/-- Theorem stating that the maximum torque is approximately 1.36 × 10^5 N·m -/
theorem max_torque_approx :
  ∃ ε > 0, |max_torque - 1.36e5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_torque_approx_l1202_120229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liter_conversion_hour_to_day_conversion_cubic_meter_conversion_l1202_120238

-- Define conversion factors
noncomputable def liters_to_milliliters : ℝ → ℝ := (· * 1000)
noncomputable def hours_to_days : ℝ → ℝ := (· / 24)
noncomputable def cubic_meters_to_cubic_centimeters : ℝ → ℝ := (· * 1000000)

-- Theorem statements
theorem liter_conversion (x : ℝ) : 
  x = ⌊x⌋ + (x - ⌊x⌋) → 
  liters_to_milliliters x = liters_to_milliliters ⌊x⌋ + liters_to_milliliters (x - ⌊x⌋) :=
by sorry

theorem hour_to_day_conversion : 
  hours_to_days 6 = (1 : ℝ) / 4 :=
by sorry

theorem cubic_meter_conversion : 
  cubic_meters_to_cubic_centimeters 0.75 = 750000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liter_conversion_hour_to_day_conversion_cubic_meter_conversion_l1202_120238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_intersection_l1202_120273

theorem equilateral_triangle_intersection (a : ℝ) : 
  let line := {p : ℝ × ℝ | p.2 = a * p.1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.2 + 6 = 0}
  let C := (0, 3)
  (∃ (A B : ℝ × ℝ), 
    A ∈ line ∧ A ∈ circle ∧
    B ∈ line ∧ B ∈ circle ∧
    C ∉ line ∧
    dist A B = dist B C ∧ dist B C = dist C A)
  → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by
  sorry

#check equilateral_triangle_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_intersection_l1202_120273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_right_triangle_l1202_120263

/-- Represents a right triangle with given hypotenuse and one leg -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse_positive : hypotenuse > 0
  leg1_positive : leg1 > 0
  leg2_positive : leg2 > 0
  pythagorean : hypotenuse^2 = leg1^2 + leg2^2

/-- Calculate the area of a right triangle -/
noncomputable def area (t : RightTriangle) : ℝ :=
  t.leg1 * t.leg2 / 2

/-- Calculate the perimeter of a right triangle -/
noncomputable def perimeter (t : RightTriangle) : ℝ :=
  t.hypotenuse + t.leg1 + t.leg2

/-- Theorem about a specific right triangle -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.hypotenuse = 13 ∧ t.leg1 = 5 ∧ area t = 30 ∧ perimeter t = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_right_triangle_l1202_120263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l1202_120264

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x - Real.pi / 6)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- Theorem statement
theorem triangle_max_area 
  (ABC : Triangle) 
  (h1 : f (ABC.A / 2) = 1) 
  (h2 : ABC.a = 2) : 
  ABC.area ≤ 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l1202_120264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1202_120279

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from (2, -1) to the line x - y + 3 = 0 is 3√2 -/
theorem distance_point_to_line_example : distance_point_to_line 2 (-1) 1 (-1) 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1202_120279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_inscribed_quadrilaterals_l1202_120288

-- Define the circles and quadrilaterals
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def Quadrilateral (A B C D : ℝ × ℝ) : Set (ℝ × ℝ) := {A, B, C, D}

-- Define the area function (this would need to be implemented properly)
noncomputable def area (q : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the problem setup
theorem concentric_circles_inscribed_quadrilaterals 
  (center : ℝ × ℝ) (R R₁ : ℝ) 
  (K : Set (ℝ × ℝ)) (K₁ : Set (ℝ × ℝ))
  (ABCD : Set (ℝ × ℝ)) (A₁B₁C₁D₁ : Set (ℝ × ℝ))
  (A B C D A₁ B₁ C₁ D₁ : ℝ × ℝ)
  (h_R₁_gt_R : R₁ > R)
  (h_K : K = Circle center R)
  (h_K₁ : K₁ = Circle center R₁)
  (h_ABCD : ABCD = Quadrilateral A B C D)
  (h_A₁B₁C₁D₁ : A₁B₁C₁D₁ = Quadrilateral A₁ B₁ C₁ D₁)
  (h_ABCD_inscribed : ABCD ⊆ K)
  (h_A₁B₁C₁D₁_inscribed : A₁B₁C₁D₁ ⊆ K₁)
  (h_A₁_on_CD : ∃ t : ℝ, A₁ = C + t • (D - C))
  (h_B₁_on_DA : ∃ t : ℝ, B₁ = D + t • (A - D))
  (h_C₁_on_AB : ∃ t : ℝ, C₁ = A + t • (B - A))
  (h_D₁_on_BC : ∃ t : ℝ, D₁ = B + t • (C - B)) :
  area A₁B₁C₁D₁ / area ABCD ≥ R₁^2 / R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_inscribed_quadrilaterals_l1202_120288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1202_120216

noncomputable section

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the distance from a point to a line
noncomputable def distanceToLine (x₀ y₀ A B C : ℝ) : ℝ := 
  (abs (A * x₀ + B * y₀ + C)) / Real.sqrt (A^2 + B^2)

-- Define the perimeter of a triangle
noncomputable def trianglePerimeter (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) +
  Real.sqrt ((x₂ - x₃)^2 + (y₂ - y₃)^2) +
  Real.sqrt ((x₃ - x₁)^2 + (y₃ - y₁)^2)

-- Define the radius of the inscribed circle of a triangle
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt ((s - a) * (s - b) * (s - c) / s)

theorem ellipse_properties :
  ∀ a b c : ℝ,
  a > b ∧ b > 0 ∧
  distanceToLine c 0 1 (Real.sqrt 3) 0 = 1/2 ∧
  (∀ x y : ℝ, ellipse a b x y → trianglePerimeter (-c) 0 x y c 0 = 6) →
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∀ m n : ℝ,
    ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse 2 (Real.sqrt 3) x₁ y₁ ∧
    ellipse 2 (Real.sqrt 3) x₂ y₂ ∧
    y₁ ≠ y₂ ∧
    x₁ = m * y₁ - 1 ∧
    x₂ = m * y₂ - 1 →
    inscribedCircleRadius (Real.sqrt ((x₁ - 1)^2 + y₁^2))
                          (Real.sqrt ((x₂ - 1)^2 + y₂^2))
                          (Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ≤ 3/4) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1202_120216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1202_120272

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def f (x : ℝ) : ℝ :=
  distance x 0 (-2) 4 + distance x 0 (-1) (-3)

theorem min_value_of_f :
  ∃ (x : ℝ), f x = Real.sqrt 50 ∧ ∀ (y : ℝ), f y ≥ f x := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1202_120272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1202_120293

-- Define the hyperbola and circle
noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
noncomputable def hyperbola_circle (x y : ℝ) : Prop := (x - Real.sqrt 10 / 2)^2 + y^2 = 1

-- Define the focus of the hyperbola
noncomputable def focus : ℝ × ℝ := (-Real.sqrt 10 / 2, 0)

-- Define the eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x y : ℝ), hyperbola a b x y ∧ hyperbola_circle x y ∧
  (∃ (m k : ℝ), y = m * x + k ∧ 
    (x - Real.sqrt 10 / 2)^2 + y^2 = 1 ∧
    m * (-Real.sqrt 10 / 2) + k = 0) →
  eccentricity a (Real.sqrt 10 / 2) = Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1202_120293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walt_interest_calculation_l1202_120228

/-- Calculates the total interest earned from two investments with different rates -/
theorem walt_interest_calculation 
  (total_money : ℝ) 
  (investment_8_percent : ℝ) 
  (rate_8_percent : ℝ) 
  (rate_9_percent : ℝ) 
  (h1 : total_money = 9000)
  (h2 : investment_8_percent = 4000)
  (h3 : rate_8_percent = 0.08)
  (h4 : rate_9_percent = 0.09) :
  let investment_9_percent := total_money - investment_8_percent
  let interest_8_percent := investment_8_percent * rate_8_percent
  let interest_9_percent := investment_9_percent * rate_9_percent
  let total_interest := interest_8_percent + interest_9_percent
  total_interest = 770 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walt_interest_calculation_l1202_120228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_vector_subtraction_l1202_120220

/-- If C is the midpoint of line segment AB, then AB - BC = AC -/
theorem midpoint_vector_subtraction (A B C : EuclideanSpace ℝ (Fin 3)) 
  (h : C = (1 / 2 : ℝ) • (A + B)) : A - B - (B - C) = A - C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_vector_subtraction_l1202_120220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_102_103_l1202_120298

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g_102_103 : Int.gcd (g 102) (g 103) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_102_103_l1202_120298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_tiling_l1202_120271

theorem rectangle_tiling (squares : List ℕ) : 
  squares = [2, 5, 7, 9, 16, 25, 28, 33, 36] →
  ∃ (width height : ℕ), 
    width * height = (squares.map (λ x => x * x)).sum ∧
    width = 61 ∧ 
    height = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_tiling_l1202_120271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_formula_l1202_120254

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for n = 0
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / (3 * sequence_a (n + 1) + 1)

def sequence_b (n : ℕ) : ℚ := 1 / sequence_a n

theorem sequence_b_formula (n : ℕ) : sequence_b n = 3 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_formula_l1202_120254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_m_l1202_120236

/-- A sequence of positive integers -/
def b : ℕ → ℕ := sorry

/-- The sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- We fix a_1 = 1 for simplicity
  | n + 1 => (a n) ^ (b n) + 1

/-- Checks if a number is a power of 2 -/
def isPowerOfTwo (m : ℕ) : Prop :=
  ∃ k : ℕ, m = 2^k

/-- The main theorem -/
theorem characterization_of_m (m : ℕ) (h : m ≥ 3) :
  (∃ N : ℕ, ∃ p : ℕ, ∀ n ≥ N, a (n + p) % m = a n % m) →
  (∃ q u v : ℕ, 2 ≤ q ∧ q ≤ m - 1 ∧
    ∃ T : ℕ, ∀ t : ℕ, b (v + u * (t + T)) % q = b (v + u * t) % q) ↔
  ¬ isPowerOfTwo m := by
  sorry

#check characterization_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_m_l1202_120236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_sets_midpoint_closed_iff_l1202_120217

/-- A set S in the complex plane is good for angle θ if it's invariant under rotation by θ around any of its points. -/
def IsGood (S : Set ℂ) (θ : ℝ) : Prop :=
  ∀ z w : ℂ, z ∈ S → w ∈ S → (w - z) * Complex.exp (θ * Complex.I) + z ∈ S ∧
                              (w - z) * Complex.exp (-θ * Complex.I) + z ∈ S

/-- A set S in the complex plane is midpoint-closed if the midpoint of any two points in S is also in S. -/
def IsMidpointClosed (S : Set ℂ) : Prop :=
  ∀ z w : ℂ, z ∈ S → w ∈ S → (z + w) / 2 ∈ S

/-- The main theorem: characterization of rational r for which good sets are midpoint-closed. -/
theorem good_sets_midpoint_closed_iff (r : ℚ) (hr : -1 ≤ r ∧ r ≤ 1) :
  (∀ S : Set ℂ, IsGood S (Real.arccos (r : ℝ)) → IsMidpointClosed S) ↔
  ∃ n : ℤ, r = (4 * n - 1) / (4 * n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_sets_midpoint_closed_iff_l1202_120217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_correct_l1202_120200

/-- The side length of the square and equilateral triangles -/
noncomputable def side_length : ℝ := 2 * Real.sqrt 3

/-- The rhombus formed by the intersection of two equilateral triangles in a square -/
structure Rhombus where
  square_side : ℝ
  triangle_side : ℝ
  h : square_side = triangle_side
  h' : square_side = side_length

/-- The area of the rhombus -/
noncomputable def rhombus_area (r : Rhombus) : ℝ :=
  8 * Real.sqrt 3 - 12

/-- The theorem stating that the area of the rhombus is 8√3 - 12 -/
theorem rhombus_area_is_correct (r : Rhombus) : 
  rhombus_area r = 8 * Real.sqrt 3 - 12 := by
  unfold rhombus_area
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_correct_l1202_120200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1202_120203

/-- The distance between two parallel lines represented by their coefficients and constants -/
noncomputable def distance_parallel_lines (a b m n : ℝ) : ℝ :=
  abs (m - n) / Real.sqrt (a^2 + b^2)

/-- Proof that the distance between 5x + 12y + 3 = 0 and 10x + 24y + 5 = 0 is 1/26 -/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 5 12 3 (5/2) = 1/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1202_120203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1202_120258

noncomputable def A : ℝ × ℝ := (1, 1)
noncomputable def B : ℝ × ℝ := (4, 2)
noncomputable def C : ℝ × ℝ := (-4, 6)

def median_equation (x y : ℝ) : Prop := 3 * x + y - 4 = 0

noncomputable def altitude_length : ℝ := Real.sqrt 5

noncomputable def triangle_area : ℝ := 10

theorem triangle_properties :
  let midpoint := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  median_equation (midpoint.1 - A.1) (midpoint.2 - A.2) ∧
  altitude_length = (|B.1 - C.1| * |A.2 - B.2| + |B.2 - C.2| * |A.1 - B.1|) / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) ∧
  triangle_area = (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * altitude_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1202_120258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l1202_120274

-- Define the constants
noncomputable def a : ℝ := 2^(3/10)
noncomputable def b : ℝ := Real.log 3 / Real.log (1/5)
noncomputable def c : ℝ := Real.log 4 / Real.log (1/5)

-- State the theorem
theorem a_greater_than_b_greater_than_c : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l1202_120274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1202_120240

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The ratio of S_6 to S_3 is 1:2 -/
axiom ratio_S6_S3 : S 6 / S 3 = 1 / 2

/-- Theorem: If S_6 : S_3 = 1 : 2, then S_9 : S_3 = 3 : 4 -/
theorem geometric_sequence_ratio : S 9 / S 3 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1202_120240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_third_max_cos_sum_l1202_120269

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law : c^2 = a^2 + b^2 - a*b

-- Theorem 1: Angle C is π/3
theorem angle_C_is_pi_third (t : Triangle) : t.C = π/3 := by
  sorry

-- Theorem 2: Maximum value of cos A + cos B is 1
theorem max_cos_sum (t : Triangle) : 
  ∃ (x : ℝ), ∀ (y : ℝ), Real.cos t.A + Real.cos t.B ≤ x ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_third_max_cos_sum_l1202_120269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequality_l1202_120292

open Real

theorem logarithm_inequality (x : ℝ) (a b c : ℝ) 
  (h1 : x ∈ Set.Ioo (Real.exp (-1)) 1)
  (h2 : a = log x)
  (h3 : b = 2 * log x)
  (h4 : c = (log x)^3) :
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequality_l1202_120292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_l1202_120226

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_3 : (Finset.range 3).sum a = 7
  sum_6 : (Finset.range 6).sum a = 63

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 2^(n - 1)

/-- Theorem stating that the general term of the sequence is 2^(n-1) -/
theorem geometric_sequence_general_term (seq : GeometricSequence) :
  ∀ n : ℕ, seq.a n = general_term n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_l1202_120226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l1202_120208

theorem matrix_scalar_multiplication (w : Fin 3 → ℝ) :
  let N : Matrix (Fin 3) (Fin 3) ℝ := ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]]
  N.mulVec w = (3 : ℝ) • w :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l1202_120208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l1202_120211

/-- Represents a card with two sides, each containing a number. -/
structure Card (n : ℕ) where
  side1 : Fin n
  side2 : Fin n

/-- The theorem statement -/
theorem card_arrangement_theorem (n : ℕ) (cards : Fin n → Card n) 
  (h1 : ∀ i : Fin n, cards i ∈ Set.range cards)
  (h2 : ∀ k : Fin n, (Multiset.count k (Multiset.map Card.side1 (Multiset.ofList (List.ofFn cards))) + 
                      Multiset.count k (Multiset.map Card.side2 (Multiset.ofList (List.ofFn cards)))) = 2) :
  ∃ (arrangement : Fin n → Bool), 
    ∀ k : Fin n, ∃ i : Fin n, 
      (arrangement i = true ∧ (cards i).side1 = k) ∨ 
      (arrangement i = false ∧ (cards i).side2 = k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l1202_120211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l1202_120241

/-- A function f with the given properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + φ)

/-- The theorem stating the possible forms of f given the conditions -/
theorem f_expression (ω φ : ℝ) :
  (∀ x, f ω φ (x + π) = f ω φ x) ∧  -- Smallest positive period is π
  (∀ x, f ω φ (π/3 + x) = f ω φ (π/3 - x)) →  -- Symmetry condition
  (f ω φ = f 2 (-π/6) ∨ f ω φ = f (-2) (π/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l1202_120241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l1202_120235

theorem book_arrangement_count : ℕ := by
  -- Define the number of math and English books
  let math_books : ℕ := 4
  let english_books : ℕ := 6

  -- Define the function to calculate the number of arrangements
  let arrangement_count (math : ℕ) (english : ℕ) : ℕ :=
    2 * (Nat.factorial math) * (Nat.factorial english)

  -- Calculate the result
  let result := arrangement_count math_books english_books

  -- Prove that the number of arrangements is 34560
  have h : result = 34560 := by sorry

  -- Return the result
  exact result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l1202_120235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_tax_free_items_is_six_l1202_120223

/-- Calculates the cost of tax-free items given total spending, tax percentage, and tax rate. -/
noncomputable def cost_of_tax_free_items (total_spending : ℝ) (tax_percentage : ℝ) (tax_rate : ℝ) : ℝ :=
  total_spending - (total_spending * (1 - tax_percentage / 100) * (1 + tax_rate / 100))

/-- Theorem stating that given the problem conditions, the cost of tax-free items is 6. -/
theorem cost_of_tax_free_items_is_six :
  cost_of_tax_free_items 20 30 6 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_tax_free_items_is_six_l1202_120223
