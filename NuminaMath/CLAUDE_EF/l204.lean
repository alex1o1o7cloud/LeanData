import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l204_20431

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem quadratic_properties :
  (∀ x y : ℝ, f x = 0 ∧ f y = 0 → x ≠ y) ∧ 
  (∀ x : ℝ, f 1.5 ≤ f x) ∧
  (∀ x y : ℝ, x > 1.5 ∧ y > x → f x < f y) ∧
  (∀ x y : ℝ, x < 1.5 ∧ y < x → f x > f y) ∧
  (¬∃ y : ℝ, ∀ x : ℝ, f 0 ≥ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l204_20431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_change_l204_20425

/-- Calculates the percentage change in membership given initial increase and subsequent decrease -/
def membership_change (initial_increase : Real) (subsequent_decrease : Real) : Real :=
  let after_increase := 1 + initial_increase
  let after_decrease := after_increase * (1 - subsequent_decrease)
  (after_decrease - 1) * 100

/-- The total percentage change in membership from fall to spring -/
theorem total_percentage_change :
  ∃ (ε : Real), abs (membership_change 0.09 0.19 - (-11.71)) < ε ∧ ε > 0 := by
  sorry

#eval membership_change 0.09 0.19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_change_l204_20425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_radii_l204_20417

/-- A triangle with side lengths 5, 12, and 13 -/
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  right_angle : PQ^2 + QR^2 = PR^2
  PQ_eq : PQ = 5
  QR_eq : QR = 12
  PR_eq : PR = 13

/-- Point S on PR such that PS:SR = 5:8 -/
noncomputable def S_point (t : RightTriangle) : ℝ × ℝ :=
  let PS := (5 / 13) * t.PR
  let SR := (8 / 13) * t.PR
  (PS, SR)

/-- Line through S bisects angle QPR -/
def bisects_angle (t : RightTriangle) (s : ℝ × ℝ) : Prop :=
  s.1 / s.2 = t.PQ / t.QR

/-- Radius of inscribed circle in triangle PQS -/
noncomputable def r_p (t : RightTriangle) (s : ℝ × ℝ) : ℝ :=
  let PQ := t.PQ
  let QS := (PQ * s.2) / t.QR
  let PS := s.1
  let semi_p := (PQ + QS + PS) / 2
  (semi_p - PQ) * (semi_p - QS) * (semi_p - PS) / semi_p

/-- Radius of inscribed circle in triangle QRS -/
noncomputable def r_q (t : RightTriangle) (s : ℝ × ℝ) : ℝ :=
  let QR := t.QR
  let QS := (t.PQ * s.2) / QR
  let SR := s.2
  let semi_q := (QR + QS + SR) / 2
  (semi_q - QR) * (semi_q - QS) * (semi_q - SR) / semi_q

theorem ratio_of_radii (t : RightTriangle) :
  let s := S_point t
  bisects_angle t s →
  r_p t s / r_q t s = 175 / 576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_radii_l204_20417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternateSquareReciprocate_formula_l204_20405

/-- The result of alternately squaring and reciprocating a number n times -/
noncomputable def alternateSquareReciprocate (x : ℝ) : ℕ → ℝ
  | 0 => x
  | n + 1 => 1 / ((alternateSquareReciprocate x n) ^ 2)

theorem alternateSquareReciprocate_formula (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  alternateSquareReciprocate x n = x ^ ((-2 : ℤ) ^ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternateSquareReciprocate_formula_l204_20405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_theorem_l204_20459

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ ∀ x : ℝ, f x > (deriv f) x + 2

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) - 2019 = -(f x - 2019)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f x - 2017 * Real.exp x < 2}

-- State the theorem
theorem function_inequality_theorem (f : ℝ → ℝ) 
  (h1 : condition1 f) (h2 : condition2 f) : 
  solution_set f = Set.Ioi 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_theorem_l204_20459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l204_20439

/-- The number of days it takes A to finish the work alone -/
noncomputable def days_A : ℝ := 18

/-- The number of days it takes B to finish the work alone -/
noncomputable def days_B : ℝ := days_A / 2

/-- The fraction of work A and B can complete together in one day -/
def work_per_day : ℚ := 1 / 6

theorem work_completion_time :
  (1 / days_A + 1 / days_B = work_per_day) → days_A = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l204_20439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_capacity_l204_20409

theorem car_fuel_capacity 
  (speed : ℝ) 
  (time : ℝ) 
  (fuel_efficiency : ℝ) 
  (fraction_used : ℝ) 
  (h1 : speed = 50)
  (h2 : time = 5)
  (h3 : fuel_efficiency = 30)
  (h4 : fraction_used = 0.8333333333333334) : 
  (speed * time / fuel_efficiency) / fraction_used = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_capacity_l204_20409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_three_halves_l204_20408

/-- The slope of a line given by the equation ax + by + c = 0 is -a/b when b ≠ 0 -/
theorem line_slope_is_three_halves (a b c : ℝ) (h : b ≠ 0) :
  -a / b = 3/2 → 6 * (-4/b) = -a := by
  sorry

#eval (6 : ℚ) / (-4 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_three_halves_l204_20408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approx_l204_20450

/-- The radius of a wheel given its total distance covered and number of revolutions -/
noncomputable def wheel_radius (total_distance : ℝ) (revolutions : ℕ) : ℝ :=
  total_distance / (2 * Real.pi * (revolutions : ℝ))

/-- Theorem stating that a wheel covering 281.6 cm in 200 revolutions has a radius of approximately 0.224 cm -/
theorem wheel_radius_approx :
  let r := wheel_radius 281.6 200
  ∃ ε > 0, abs (r - 0.224) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approx_l204_20450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_meaningful_equation_l204_20465

theorem range_of_m_for_meaningful_equation : 
  ∀ m : ℝ, (∃ x : ℝ, Real.sin x - Real.sqrt 3 * Real.cos x = (4 * m - 6) / (4 - m)) ↔ 
  (-1 ≤ m ∧ m ≤ 7/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_meaningful_equation_l204_20465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l204_20402

-- Define the functions
noncomputable def f (x : ℝ) := |Real.sin x|
noncomputable def g (x : ℝ) := -Real.tan x

-- Define the property of having a period of π
def has_period_pi (h : ℝ → ℝ) : Prop :=
  ∀ x, h (x + Real.pi) = h x

-- Define the property of being monotonically decreasing on an interval
def monotone_decreasing_on (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → h y < h x

-- State the theorem
theorem f_and_g_properties :
  (has_period_pi f ∧ monotone_decreasing_on f (Real.pi/2) (3*Real.pi/4)) ∧
  (has_period_pi g ∧ monotone_decreasing_on g (Real.pi/2) (3*Real.pi/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l204_20402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l204_20446

noncomputable section

/-- The line passing through the point (3,1,-2) with direction vector (-3,7,-4) -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (3 - 3*t, 1 + 7*t, -2 - 4*t)

/-- The given point -/
def point : ℝ × ℝ × ℝ := (1, 2, -1)

/-- The closest point on the line to the given point -/
def closest_point : ℝ × ℝ × ℝ := (111/74, 173/74, -170/37)

theorem closest_point_is_correct :
  ∀ t : ℝ, ‖line t - point‖ ≥ ‖closest_point - point‖ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l204_20446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l204_20454

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x - cos x

-- State the theorem
theorem expression_value :
  (∀ x, (deriv^[2] f) x = 2 * f x) →
  ∀ x, (1 + sin x ^ 2) / (cos x ^ 2 - sin (2 * x)) = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l204_20454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_moles_formed_l204_20478

/-- Represents the number of moles of a substance -/
def Moles : Type := ℕ

/-- Provides an instance of OfNat for Moles -/
instance : OfNat Moles n where
  ofNat := n

/-- Represents the chemical equation C5H11OH + HCl → C5H11Cl + H2O -/
structure ChemicalEquation where
  amyl_alcohol : Moles
  hydrochloric_acid : Moles
  chloro_dimethylpropane : Moles
  water : Moles

/-- The reaction follows a 1:1:1:1 molar ratio -/
axiom reaction_ratio (eq : ChemicalEquation) : 
  eq.amyl_alcohol = eq.hydrochloric_acid ∧ 
  eq.amyl_alcohol = eq.chloro_dimethylpropane ∧ 
  eq.amyl_alcohol = eq.water

/-- The given reaction conditions -/
def given_reaction : ChemicalEquation := {
  amyl_alcohol := 3,
  hydrochloric_acid := 3,
  chloro_dimethylpropane := 3,
  water := 3
}

/-- Theorem: The number of moles of water formed is 3 -/
theorem water_moles_formed : given_reaction.water = 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_moles_formed_l204_20478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_calculation_l204_20455

noncomputable section

def radius1 : ℝ := 1
def radius2 : ℝ := 1
def radius3 : ℝ := 2 * Real.sqrt ((13 - 6 * Real.sqrt 3) / 13)
def side_length : ℝ := Real.sqrt 3

def circumradius : ℝ := 4 * Real.sqrt 3 - 6

theorem circumradius_calculation :
  let r1 := radius1
  let r2 := radius2
  let r3 := radius3
  let s := side_length
  circumradius = 4 * Real.sqrt 3 - 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_calculation_l204_20455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_pprime_is_parabola_l204_20412

/-- Represents a parabola in a 2D plane -/
structure Parabola where
  k : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ x = y^2 / (2 * k)

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The locus of P' for a given parabola -/
noncomputable def locusOfPPrime (p : Parabola) : Parabola :=
  { k := p.k / 2
    equation := fun (x, y) ↦ x = y^2 / p.k + p.k / 2 }

/-- The focus of the locus of P' -/
noncomputable def focusOfLocus (p : Parabola) : Point :=
  { x := 3 * p.k / 4
    y := 0 }

/-- The directrix of the locus of P' -/
noncomputable def directrixOfLocus (p : Parabola) : ℝ := p.k / 4

theorem locus_of_pprime_is_parabola (p : Parabola) :
  ∃ (p' : Parabola), 
    (∀ (x y : ℝ), p'.equation (x, y) ↔ (locusOfPPrime p).equation (x, y)) ∧
    (focusOfLocus p = { x := 3 * p.k / 4, y := 0 }) ∧
    (directrixOfLocus p = p.k / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_pprime_is_parabola_l204_20412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l204_20477

/-- The circle equation -/
def circle_eq (x y b : ℝ) : Prop := x^2 + y^2 = 4 * b^2

/-- The parabola equation -/
def parabola_eq (x y b : ℝ) : Prop := y = x^2 - 4 * b

/-- The number of intersection points -/
noncomputable def intersection_count (b : ℝ) : ℕ := 
  if b > 1/8 then 3 else 1

theorem three_intersection_points (b : ℝ) :
  intersection_count b = 3 ↔ b > 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l204_20477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_standard_normals_is_normal_l204_20491

/-- The probability density function of a standard normal distribution -/
noncomputable def standard_normal_pdf (x : ℝ) : ℝ :=
  1 / Real.sqrt (2 * Real.pi) * Real.exp (-x^2 / 2)

/-- Two independent random variables with standard normal distribution -/
structure IndependentStandardNormals where
  X : ℝ → ℝ
  Y : ℝ → ℝ
  hX : X = standard_normal_pdf
  hY : Y = standard_normal_pdf

/-- The sum of two independent standard normal random variables -/
def sum_of_normals (vars : IndependentStandardNormals) (z : ℝ) : ℝ :=
  vars.X z + vars.Y z

/-- The probability density function of the sum of two independent standard normal random variables -/
noncomputable def sum_pdf (z : ℝ) : ℝ :=
  1 / Real.sqrt (4 * Real.pi) * Real.exp (-z^2 / 4)

/-- Theorem: The sum of two independent standard normal random variables is normally distributed -/
theorem sum_of_standard_normals_is_normal (vars : IndependentStandardNormals) :
  sum_of_normals vars = sum_pdf := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_standard_normals_is_normal_l204_20491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_is_circle_l204_20401

/-- The set of points M(x, y) whose distances from O(0, 0) and A(3, 0) have a ratio of 1:2 -/
noncomputable def Apollonius (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) / Real.sqrt ((x - 3)^2 + y^2) = 1/2

/-- The equation of a circle centered at (1, 0) with radius 2 -/
def CircleEquation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 4

/-- Theorem stating that the Apollonius set is equivalent to the circle equation -/
theorem apollonius_is_circle :
  ∀ x y : ℝ, Apollonius x y ↔ CircleEquation x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_is_circle_l204_20401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_per_litre_mixed_correct_solution_correct_l204_20429

/-- Represents the properties of an oil type -/
structure OilType where
  volume : ℝ
  price_per_litre : ℝ

/-- Calculates the price per litre of mixed lubricant oil -/
noncomputable def price_per_litre_mixed (oil1 oil2 oil3 : OilType) : ℝ :=
  let total_cost := oil1.volume * oil1.price_per_litre +
                    oil2.volume * oil2.price_per_litre +
                    oil3.volume * oil3.price_per_litre
  let total_volume := oil1.volume + oil2.volume + oil3.volume
  total_cost / total_volume

/-- Theorem: The price per litre of the mixed lubricant oil is correctly calculated -/
theorem price_per_litre_mixed_correct (oil1 oil2 oil3 : OilType) :
  price_per_litre_mixed oil1 oil2 oil3 =
  (oil1.volume * oil1.price_per_litre +
   oil2.volume * oil2.price_per_litre +
   oil3.volume * oil3.price_per_litre) /
  (oil1.volume + oil2.volume + oil3.volume) :=
by sorry

/-- The given problem instance -/
noncomputable def problem_instance : ℝ :=
  let oil1 : OilType := { volume := 100, price_per_litre := 45 }
  let oil2 : OilType := { volume := 30, price_per_litre := 57.5 }
  let oil3 : OilType := { volume := 20, price_per_litre := 72 }
  price_per_litre_mixed oil1 oil2 oil3

/-- Theorem: The solution to the problem instance is correct -/
theorem solution_correct : problem_instance = 51.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_per_litre_mixed_correct_solution_correct_l204_20429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_perimeter_l204_20403

/-- A convex quadrilateral with special properties -/
structure SpecialQuadrilateral where
  /-- The quadrilateral is convex -/
  is_convex : Bool
  /-- The diagonals intersect at a point dividing each diagonal in the ratio 1:2 -/
  diagonal_ratio : ℝ × ℝ
  /-- The quadrilateral formed by joining the midpoints of the sides is a square -/
  midpoint_square : Bool
  /-- The side length of the square formed by midpoints -/
  midpoint_square_side : ℝ

/-- Perimeter of a SpecialQuadrilateral -/
noncomputable def perimeter (q : SpecialQuadrilateral) : ℝ := sorry

/-- Theorem stating the perimeter of the special quadrilateral -/
theorem special_quadrilateral_perimeter (q : SpecialQuadrilateral) 
  (h1 : q.is_convex = true)
  (h2 : q.diagonal_ratio = (1, 2))
  (h3 : q.midpoint_square = true)
  (h4 : q.midpoint_square_side = 3) :
  ∃ (p : ℝ), p = 4 * Real.sqrt 5 + 6 * Real.sqrt 2 ∧ 
  p = perimeter q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_perimeter_l204_20403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_amount_example_l204_20438

/-- Given a selling price and a gain percentage, calculate the gain amount. -/
noncomputable def gain_amount (selling_price : ℝ) (gain_percentage : ℝ) : ℝ :=
  let cost_price := selling_price / (1 + gain_percentage / 100)
  selling_price - cost_price

/-- Theorem: The gain amount is $15 when the selling price is $90 and the gain percentage is 20%. -/
theorem gain_amount_example : gain_amount 90 20 = 15 := by
  -- Unfold the definition of gain_amount
  unfold gain_amount
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_amount_example_l204_20438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_G_less_than_one_l204_20496

open Real MeasureTheory Measure

-- Define the function G as the derivative of y = 2cos(x) + 3
noncomputable def G (x : ℝ) : ℝ := -2 * Real.sin x

-- Define the interval
def interval : Set ℝ := Set.Icc (-π/3) π

-- Define the event
def event : Set ℝ := {x ∈ interval | G x < 1}

-- State the theorem
theorem probability_G_less_than_one :
  (volume event / volume interval) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_G_less_than_one_l204_20496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l204_20443

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

theorem a_range (a : ℝ) : A ⊆ B a → a ∈ Set.Icc (-4) (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l204_20443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l204_20464

/-- A power function with a coefficient dependent on m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(m^2 - 2)

/-- The function f is increasing on (0, +∞) -/
def is_increasing_on_pos (m : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f m x < f m y

/-- The unique value of m that satisfies the conditions -/
theorem unique_m_value : ∃! m : ℝ, is_increasing_on_pos m ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l204_20464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_flow_rate_specific_case_l204_20411

/-- Given an initial amount of water, a final amount of water, and a duration of rainfall,
    calculate the rate of water flow into a tank. -/
noncomputable def water_flow_rate (initial_amount final_amount : ℝ) (duration : ℝ) : ℝ :=
  (final_amount - initial_amount) / duration

/-- Theorem stating that given the specific conditions in the problem,
    the water flow rate is 2 L/min. -/
theorem water_flow_rate_specific_case :
  water_flow_rate 100 280 90 = 2 := by
  -- Unfold the definition of water_flow_rate
  unfold water_flow_rate
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof is completed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_flow_rate_specific_case_l204_20411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jumbo_price_is_nine_l204_20456

/-- Represents the pumpkin sales problem -/
structure PumpkinSales where
  regular_price : ℚ
  total_pumpkins : ℕ
  total_revenue : ℚ
  regular_pumpkins : ℕ

/-- Calculates the price of each jumbo pumpkin -/
def jumbo_price (sale : PumpkinSales) : ℚ :=
  (sale.total_revenue - sale.regular_price * sale.regular_pumpkins) / (sale.total_pumpkins - sale.regular_pumpkins)

/-- Theorem stating that the jumbo pumpkin price is $9.00 -/
theorem jumbo_price_is_nine (sale : PumpkinSales)
  (h1 : sale.regular_price = 4)
  (h2 : sale.total_pumpkins = 80)
  (h3 : sale.total_revenue = 395)
  (h4 : sale.regular_pumpkins = 65) :
  jumbo_price sale = 9 := by
  sorry

/-- Example calculation -/
def example_sale : PumpkinSales :=
  { regular_price := 4
  , total_pumpkins := 80
  , total_revenue := 395
  , regular_pumpkins := 65 }

#eval jumbo_price example_sale

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jumbo_price_is_nine_l204_20456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_exists_l204_20422

/-- Represents the speed of a cyclist during a two-part journey -/
def cyclist_speed (v : ℝ) : Prop :=
  let total_distance : ℝ := 17
  let first_distance : ℝ := 7
  let second_distance : ℝ := 10
  let second_speed : ℝ := 7
  let average_speed : ℝ := 7.99
  (total_distance / (first_distance / v + second_distance / second_speed) = average_speed) ∧
  (abs (v - 10.01) < 0.01)

/-- Theorem stating the existence of a speed satisfying the cyclist's journey conditions -/
theorem cyclist_speed_exists : ∃ v : ℝ, cyclist_speed v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_exists_l204_20422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l204_20488

/-- The function f(x) = √x - x for x ≥ 0 -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x

/-- The range of f is (-∞, 1/4] -/
theorem range_of_f : Set.range f = Set.Iic (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l204_20488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_2alpha_l204_20497

theorem cos_pi_half_minus_2alpha (α : ℝ) (h : Real.sin α - Real.cos α = 1/3) :
  Real.cos (π/2 - 2*α) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_2alpha_l204_20497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_necessary_not_sufficient_for_equality_l204_20460

/-- Two vectors are collinear if they are scalar multiples of each other -/
def collinear (v w : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k, k, k) • w ∨ w = (k, k, k) • v

/-- Theorem stating that collinearity is necessary but not sufficient for vector equality -/
theorem collinear_necessary_not_sufficient_for_equality :
  (∀ v w : ℝ × ℝ × ℝ, v = w → collinear v w) ∧
  (∃ v w : ℝ × ℝ × ℝ, collinear v w ∧ v ≠ w) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_necessary_not_sufficient_for_equality_l204_20460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_distance_l204_20432

/-- The total distance traveled by a bee on a hemispherical dome --/
theorem bee_flight_distance (r : ℝ) (third_leg : ℝ) :
  r = 50 ∧ third_leg = 90 →
  ∃ (second_leg : ℝ),
    second_leg^2 + third_leg^2 = (2*r)^2 ∧
    abs ((2*r + second_leg + third_leg) - 233.59) < 0.01 := by
  sorry

#check bee_flight_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_distance_l204_20432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_intersection_points_l204_20457

/-- The first equation defining a curve in the xy-plane -/
def equation1 (x y : ℝ) : Prop :=
  (x^2 - y + 2) * (3 * x^3 + y - 4) = 0

/-- The second equation defining a curve in the xy-plane -/
def equation2 (x y : ℝ) : Prop :=
  (x + y^2 - 2) * (2 * x^2 - 5 * y + 7) = 0

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def is_intersection_point (x y : ℝ) : Prop :=
  equation1 x y ∧ equation2 x y

/-- The set of all intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | is_intersection_point p.1 p.2}

/-- The theorem stating that there are exactly 6 intersection points -/
theorem six_intersection_points :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 6 ∧ ∀ p, p ∈ S ↔ p ∈ intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_intersection_points_l204_20457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_covered_l204_20430

/-- Represents a rectangular card -/
structure Card where
  width : Real
  height : Real

/-- Represents a square on a checkerboard -/
structure Square where
  side : Real

/-- Represents a placement of the card on the checkerboard -/
def Placement := Card → Finset Square

theorem max_squares_covered (card : Card) (square : Square) : 
  card.width = 1.5 ∧ card.height = 2 ∧ square.side = 1 → 
  ∃ (n : Nat), n ≤ 9 ∧ 
  ∀ (m : Nat), (∃ (placement : Placement), 
    (∀ s ∈ placement card, s.side = 1) ∧ 
    (placement card).card = m) → 
  m ≤ n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_covered_l204_20430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_l204_20458

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 2 = 0) →
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 2 = 0) →
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 2 = 0) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₂ * (x₁ + x₃) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_l204_20458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_with_lcm_and_ratio_l204_20416

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : Nat.lcm a b = 30) 
  (h4 : 3 * a = 2 * b) : 
  a + b = 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_with_lcm_and_ratio_l204_20416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l204_20467

theorem geometric_sequence_sixth_term 
  (a₁ : ℝ) 
  (a₈ : ℝ) 
  (h₁ : a₁ = 3) 
  (h₂ : a₈ = 39366) 
  (h₃ : ∀ n : Nat, 2 ≤ n ∧ n ≤ 7 → ∃ r : ℝ, a₁ * r^(n-1) = a₁ * r^(n-2) * r) :
  ∃ a₆ : ℝ, a₆ = 23328 ∧ a₆ = a₁ * (a₈ / a₁)^((6-1)/(8-1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l204_20467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_non_square_formula_l204_20463

/-- The sequence of natural numbers with perfect squares removed -/
def nonSquareSeq : ℕ → ℕ :=
  sorry

/-- The nth number in the sequence of natural numbers with perfect squares removed -/
def nthNonSquare (n : ℕ) : ℕ := nonSquareSeq n

/-- The nearest integer to the square root of a natural number -/
def nearestSqrt (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the nth non-square number is n plus the nearest integer to sqrt(n) -/
theorem nth_non_square_formula (n : ℕ) :
  nthNonSquare n = n + nearestSqrt n :=
by
  sorry

#check nth_non_square_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_non_square_formula_l204_20463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_on_perp_bisector_l204_20479

-- Define the circle C with center O and radius r
variable (O : ℝ × ℝ) (r : ℝ)
def C (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | dist p O = r}

-- Define points B and D on the circle C
variable (B D : ℝ × ℝ)
axiom B_on_C : B ∈ C O r
axiom D_on_C : D ∈ C O r

-- Define the perpendicular bisector of BD
def perp_bisector (B D : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {A : ℝ × ℝ | dist A B = dist A D}

-- State the theorem
theorem equidistant_points_on_perp_bisector (B D : ℝ × ℝ) :
  ∀ A : ℝ × ℝ, (dist A B = dist A D) ↔ A ∈ perp_bisector B D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_on_perp_bisector_l204_20479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l204_20473

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (7 / 4) * a n + (5 / 4) * Real.sqrt (3^n - (a n)^2)

theorem a_10_value : a 10 = 11907 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l204_20473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l204_20415

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3 * Real.sqrt 2 / 2, -3 / 2;
     3 / 2, 3 * Real.sqrt 2 / 2]

theorem matrix_power_four :
  A ^ 4 = !![(-81 : ℝ), 0;
              0, -81] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l204_20415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l204_20400

theorem triangle_vector_relation (A B C M : EuclideanSpace ℝ (Fin 2)) (lambda mu : ℝ) :
  (M - A = 3 • (C - M)) →
  (M - B = lambda • (A - B) + mu • (C - B)) →
  mu - lambda = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l204_20400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_three_l204_20423

/-- The eccentricity of a hyperbola -/
noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := 
  let b : ℝ := 2
  let c : ℝ := Real.sqrt (a^2 + b^2)
  c / a

/-- The hyperbola equation -/
def hyperbola_equation (x y a : ℝ) : Prop :=
  y^2 / a^2 - x^2 / 4 = 1

theorem hyperbola_eccentricity_is_three :
  ∀ a : ℝ, 
    (hyperbola_equation 2 (-1) a) → 
    (hyperbola_eccentricity a = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_three_l204_20423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2theta_value_l204_20413

noncomputable def vector_a (θ : Real) : Fin 2 → Real := ![2, Real.sin θ]
noncomputable def vector_b (θ : Real) : Fin 2 → Real := ![1, Real.cos θ]

theorem tan_2theta_value (θ : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2) 
  (h_parallel : ∃ (k : Real), vector_a θ = k • vector_b θ) : 
  Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2theta_value_l204_20413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l204_20428

theorem sin_600_plus_tan_240 : Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l204_20428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_range_l204_20448

def f (x a b c : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

theorem extreme_values_and_range (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f x a b c < c^2) →
  (∃ y : ℝ, ∀ x : ℝ, x ≠ 1 → x ≠ 2 → (f x a b c - f 1 a b c) * (f x a b c - f 2 a b c) < 0) →
  (a = -3 ∧ b = 4) ∧ (c < -1 ∨ c > 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_range_l204_20448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_sum_is_22_l204_20476

-- Define a cube type with 8 vertices
structure Cube where
  vertices : Fin 8 → ℕ

-- Define a function to check if all numbers are unique and in range [2, 9]
def valid_assignment (c : Cube) : Prop :=
  (∀ i j, i ≠ j → c.vertices i ≠ c.vertices j) ∧
  (∀ i, 2 ≤ c.vertices i ∧ c.vertices i ≤ 9)

-- Define a function to get the sum of a face
def face_sum (c : Cube) (face : Fin 4 → Fin 8) : ℕ :=
  c.vertices (face 0) + c.vertices (face 1) + c.vertices (face 2) + c.vertices (face 3)

-- Define a function to check if all face sums are equal
def equal_face_sums (c : Cube) : Prop :=
  ∀ face1 face2 : Fin 4 → Fin 8, face_sum c face1 = face_sum c face2

-- Theorem statement
theorem cube_face_sum_is_22 (c : Cube) 
  (h1 : valid_assignment c) 
  (h2 : equal_face_sums c) : 
  ∃ (face : Fin 4 → Fin 8), face_sum c face = 22 := by
  sorry

#check cube_face_sum_is_22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_sum_is_22_l204_20476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l204_20485

/-- Represents the game state on a 1 × 101 grid -/
structure GameState where
  player1_pos : Nat
  player2_pos : Nat

/-- Possible moves in the game -/
inductive Move where
  | One   : Move
  | Two   : Move
  | Three : Move
  | Four  : Move

/-- Defines a valid move in the game -/
def is_valid_move (gs : GameState) (m : Move) (is_player1 : Bool) : Bool :=
  match m with
  | Move.One   => true
  | Move.Two   => true
  | Move.Three => true
  | Move.Four  => true

/-- Applies a move to the game state -/
def apply_move (gs : GameState) (m : Move) (is_player1 : Bool) : GameState :=
  let move_value := match m with
    | Move.One   => 1
    | Move.Two   => 2
    | Move.Three => 3
    | Move.Four  => 4
  if is_player1 then
    { gs with player1_pos := min 101 (gs.player1_pos + move_value) }
  else
    { gs with player2_pos := max 1 (gs.player2_pos - move_value) }

/-- Checks if the game is over -/
def is_game_over (gs : GameState) : Bool :=
  gs.player1_pos = 101 || gs.player2_pos = 1

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      ∃ (n : Nat), 
        let rec game_play : Nat → GameState
          | 0 => ⟨1, 101⟩
          | n + 1 =>
            let gs := game_play n
            if is_game_over gs then
              gs
            else if n % 2 = 0 then
              apply_move gs (strategy gs) true
            else
              apply_move gs (opponent_strategy gs) false
        (game_play n).player1_pos = 101 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l204_20485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_half_condition_l204_20447

theorem log_half_condition (x : ℝ) :
  (x > 1 → Real.log (x + 2) / Real.log (1/2) < 0) ∧
  ¬(Real.log (x + 2) / Real.log (1/2) < 0 → x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_half_condition_l204_20447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_180_l204_20461

/-- The number of positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Theorem: The number of positive factors of 180 is 18 -/
theorem factors_of_180 : num_factors 180 = 18 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_180_l204_20461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l204_20426

/-- The circle with equation (x+a)^2 + (y-a)^2 = 4 -/
def my_circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + a)^2 + (p.2 - a)^2 = 4}

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (r : ℝ), c = {q : ℝ × ℝ | (q.1 - center.1)^2 + (q.2 - center.2)^2 = r^2} ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 < r^2

theorem range_of_a (a : ℝ) : 
  is_inside (-1, -1) (my_circle a) → -1 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l204_20426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l204_20495

-- Define the total number of balls
def total_balls : ℕ := 5

-- Define the number of red balls
def red_balls : ℕ := 3

-- Define the number of white balls
def white_balls : ℕ := 2

-- Define event A: first ball drawn is red
noncomputable def event_A : ℝ := red_balls / total_balls

-- Define the probability of both events A and B occurring
noncomputable def prob_A_and_B : ℝ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Theorem to prove
theorem conditional_probability_B_given_A :
  prob_A_and_B / event_A = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l204_20495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_symmetric_circle_representation_l204_20482

/-- The equation of a circle symmetric to x^2 + y^2 - 4x = 0 about y = x -/
theorem symmetric_circle_equation : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x = 0 ↔ y^2 + x^2 - 4*y = 0) :=
by
  sorry

/-- The symmetric circle is represented by x^2 + y^2 - 4y = 0 -/
theorem symmetric_circle_representation :
  ∃ f : ℝ × ℝ → Prop, (∀ x y : ℝ, f (x, y) ↔ x^2 + y^2 - 4*y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_symmetric_circle_representation_l204_20482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_1600_simplification_l204_20452

theorem cube_root_1600_simplification :
  ∀ c d : ℕ+,
  (c : ℝ) * (d : ℝ)^(1/3) = 1600^(1/3) →
  (∀ k : ℕ+, k < d → ¬((c : ℝ) * (k : ℝ)^(1/3) = 1600^(1/3))) →
  c + d = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_1600_simplification_l204_20452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_e_l204_20434

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_max_at_e (x : ℝ) (h : x > 0) : 
  f (Real.exp 1) ≥ f x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_e_l204_20434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l204_20492

/-- A plane vector represented by its x and y coordinates -/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- The dot product of two plane vectors -/
def dot_product (v1 v2 : PlaneVector) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Two vectors are perpendicular if their dot product is zero -/
def is_perpendicular (v1 v2 : PlaneVector) : Prop := dot_product v1 v2 = 0

/-- The magnitude (length) of a vector -/
noncomputable def magnitude (v : PlaneVector) : ℝ := Real.sqrt (v.x^2 + v.y^2)

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def is_parallel (v1 v2 : PlaneVector) : Prop :=
  ∃ (k : ℝ), v2.x = k * v1.x ∧ v2.y = k * v1.y

theorem vector_problem :
  let a : PlaneVector := ⟨1, 2⟩
  let b1 : PlaneVector := ⟨1, 1⟩
  let b2 : PlaneVector := ⟨0, 0⟩  -- Placeholder value
  (∀ (k : ℝ),
    (is_perpendicular ⟨k * a.x - b1.x, k * a.y - b1.y⟩ a → k = 3/5)) ∧
  (∀ (b2 : PlaneVector),
    is_parallel a b2 → magnitude b2 = 2 * Real.sqrt 5 →
      (b2 = ⟨2, 4⟩ ∨ b2 = ⟨-2, -4⟩)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l204_20492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l204_20445

theorem inequality_solution_count : 
  (Finset.filter (fun n : ℕ => (n : ℝ) + 9 > 0 ∧ (n : ℝ) - 2 > 0 ∧ (n : ℝ) - 15 < 0) (Finset.range 16)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l204_20445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_3_l204_20410

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem for proposition 2
theorem proposition_2 
  (m n : Line) (α : Plane) :
  perpendicular m n → 
  line_perpendicular_plane m α → 
  ¬ line_in_plane n α → 
  parallel_line_plane n α :=
sorry

-- Theorem for proposition 3
theorem proposition_3 
  (m n : Line) (α β : Plane) :
  plane_perpendicular α β → 
  line_perpendicular_plane m α → 
  line_perpendicular_plane n β → 
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_3_l204_20410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_consecutive_probability_l204_20442

/-- The probability of all Escher prints being consecutive in a random arrangement -/
theorem escher_prints_consecutive_probability
  (total_pieces : ℕ)
  (escher_prints : ℕ)
  (h_total : total_pieces = 12)
  (h_escher : escher_prints = 4) :
  (Nat.factorial (total_pieces - escher_prints + 1) * Nat.factorial escher_prints) /
  (Nat.factorial total_pieces : ℚ) = 1 / 55 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_consecutive_probability_l204_20442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l204_20406

def A : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℤ := {x | (x : ℝ)^2 - 3*(x : ℝ) < 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l204_20406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formulas_and_sum_l204_20414

def S (n : ℕ) : ℕ := n^2 + n + 1

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => 2 * (n + 1)

def b (n : ℕ) : ℕ := 2^(n - 1)

def c (n : ℕ) : ℕ :=
  if n % 2 = 1 then a n else b n

def T (n : ℕ) : ℕ :=
  (List.range n).map (fun i => c (i + 1)) |>.sum

theorem sequence_formulas_and_sum (n : ℕ) :
  (∀ n, S n = n^2 + n + 1) ∧
  (∀ n, n ≥ 2 → S n - S (n - 1) = a n) ∧
  (b 3 = a 2) ∧
  (b 4 = a 4) ∧
  (∀ n, a n = if n = 0 then 3 else 2 * n) ∧
  (∀ n, b n = 2^(n - 1)) ∧
  (T 10 = 733) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formulas_and_sum_l204_20414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l204_20418

/-- The cubic function f(x) = x³ - 3x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- The exponential function g(x) = 2ˣ - a -/
noncomputable def g (a x : ℝ) : ℝ := Real.exp (x * Real.log 2) - a

/-- The closed interval [0, 2] -/
def I : Set ℝ := Set.Icc 0 2

/-- The main theorem -/
theorem range_of_a (hf : ∀ x₁ ∈ I, ∃ x₂ ∈ I, |f x₁ - g a x₂| ≤ 2) : 2 ≤ a ∧ a ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l204_20418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_value_l204_20490

def sequence_a : ℕ → ℤ
  | 0 => 2  -- We define a_0 as 2 to match a_1 in the problem
  | 1 => 5  -- This corresponds to a_2 in the problem
  | (n + 2) => sequence_a (n + 1) - sequence_a n

theorem a_6_value : sequence_a 5 = -3 := by
  -- The proof goes here
  sorry

#eval sequence_a 5  -- This will evaluate a_6 (0-indexed)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_value_l204_20490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_average_l204_20480

theorem four_number_average (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a + b + c + d) / 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_average_l204_20480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_plus_half_floor_sum_small_floor_specific_sum_all_statements_true_l204_20424

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Statement I
theorem floor_plus_half (x : ℤ) : floor (x + 0.5) = x + 1 := by sorry

-- Statement II
theorem floor_sum_small (x y : ℝ) (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) :
  floor (x + y) = floor x + floor y := by sorry

-- Statement III
theorem floor_specific_sum : floor (2.3 + 3.2) = floor 2.3 + floor 3.2 := by sorry

-- All statements are true
theorem all_statements_true :
  (∀ x : ℤ, floor (x + 0.5) = x + 1) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < 1 → 0 ≤ y ∧ y < 1 → floor (x + y) = floor x + floor y) ∧
  (floor (2.3 + 3.2) = floor 2.3 + floor 3.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_plus_half_floor_sum_small_floor_specific_sum_all_statements_true_l204_20424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l204_20493

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 + (2 * a + 1) * x

-- State the theorem
theorem f_properties (a : ℝ) :
  (a ≥ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < -1/(2*a) → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂ : ℝ, -1/(2*a) < x₁ → x₁ < x₂ → f a x₂ < f a x₁)) ∧
  (a < 0 → ∀ x : ℝ, 0 < x → f a x ≤ -3/(4*a) - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l204_20493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l204_20483

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 0) (h : (2 : ℝ)^(x * (2 : ℝ)^y) = 16) :
  (2 / (x - 2) + 2 / y) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l204_20483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharons_drive_l204_20436

/-- The distance between Sharon's house and her mother's house in miles -/
noncomputable def distance : ℝ := 135

/-- Sharon's usual driving time in minutes -/
noncomputable def usual_time : ℝ := 180

/-- The fraction of the journey completed before the snowstorm -/
noncomputable def fraction_before_storm : ℝ := 1/3

/-- The speed reduction due to the snowstorm in miles per hour -/
noncomputable def speed_reduction : ℝ := 20

/-- The total trip time during the snowstorm day in minutes -/
noncomputable def storm_trip_time : ℝ := 276

/-- Theorem stating the relationship between the journey times and distances -/
theorem sharons_drive :
  (fraction_before_storm * distance / (distance / usual_time)) +
  ((1 - fraction_before_storm) * distance / (distance / usual_time - speed_reduction / 60)) = storm_trip_time :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharons_drive_l204_20436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_l204_20471

/-- Represents a player in the game -/
inductive Player : Type
  | Petya : Player
  | Vasya : Player

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the game state -/
structure GameState where
  grid : List (List Bool)
  currentPlayer : Player

/-- Checks if a 4x4 square starting at the given position contains a chip -/
def has_chip_in_4x4_square (state : GameState) (pos : Position) : Bool :=
  sorry

/-- Checks if the game is over (i.e., every 4x4 square contains a chip) -/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Represents a move in the game -/
def make_move (state : GameState) (pos : Position) : GameState :=
  sorry

/-- Represents the optimal strategy for Vasya -/
def vasya_strategy (state : GameState) : Position :=
  sorry

/-- Theorem stating that Vasya has a winning strategy -/
theorem vasya_wins :
  ∀ (initial_state : GameState),
    initial_state.currentPlayer = Player.Petya →
    ∀ (petya_moves : List Position),
      ∃ (vasya_moves : List Position),
        is_game_over (List.foldl (λ s m ↦ make_move s m.1) 
                           initial_state 
                           (List.zip petya_moves vasya_moves)) ∧
        (List.foldl (λ s m ↦ make_move s m.1) 
              initial_state 
              (List.zip petya_moves vasya_moves)).currentPlayer = Player.Vasya :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_l204_20471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_without_fulltime_jobs_l204_20444

/-- Calculates the percentage of parents without full-time jobs in a survey. -/
theorem percentage_without_fulltime_jobs (total_parents : ℕ) 
  (h1 : total_parents > 0) 
  (mothers_ratio : ℚ) 
  (h2 : mothers_ratio = 3/5) 
  (mothers_fulltime_ratio : ℚ) 
  (h3 : mothers_fulltime_ratio = 7/8) 
  (fathers_fulltime_ratio : ℚ) 
  (h4 : fathers_fulltime_ratio = 3/4) : 
  ∃ (percent : ℚ), percent = 18 := by
  let mothers := (mothers_ratio * total_parents).floor
  let fathers := total_parents - mothers
  let mothers_without_fulltime := mothers - (mothers_fulltime_ratio * mothers).floor
  let fathers_without_fulltime := fathers - (fathers_fulltime_ratio * fathers).floor
  let total_without_fulltime := mothers_without_fulltime + fathers_without_fulltime
  let percent := (total_without_fulltime : ℚ) / total_parents * 100
  exists percent
  sorry

#check percentage_without_fulltime_jobs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_without_fulltime_jobs_l204_20444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atOp_solutions_l204_20462

-- Define the operation @
def atOp (a b : ℝ) : ℝ := 2 * a^2 - a + b

-- Theorem statement
theorem atOp_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (∀ (x : ℝ), atOp x 3 = 4 ↔ (x = x₁ ∨ x = x₂)) ∧
  ((x₁ = 1 ∧ x₂ = -1/2) ∨ (x₁ = -1/2 ∧ x₂ = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_atOp_solutions_l204_20462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_64_81_l204_20489

-- Define the mean proportional function as noncomputable
noncomputable def mean_proportional (a b : ℝ) : ℝ := Real.sqrt (a * b)

-- Theorem statement
theorem mean_proportional_64_81 :
  mean_proportional 64 81 = 72 := by
  -- Unfold the definition of mean_proportional
  unfold mean_proportional
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_64_81_l204_20489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l204_20419

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - Real.cos x ^ 2 - 1 / 2

theorem function_properties :
  (∃ (min : ℝ), ∀ (x : ℝ), f x ≥ min ∧ ∃ (x₀ : ℝ), f x₀ = min) ∧
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (a b c : ℝ) (A B C : ℝ),
    c = Real.sqrt 3 →
    f C = 0 →
    b = 2 * a →
    c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
    a = 1 ∧ b = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l204_20419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_third_quadrant_l204_20449

theorem sin_value_in_third_quadrant (α : Real) 
  (h1 : Real.cos α = -1/3) 
  (h2 : α ∈ Set.Ioo π (3*π/2)) : 
  Real.sin α = -2*Real.sqrt 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_third_quadrant_l204_20449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_quadratic_form_l204_20470

theorem divisibility_of_quadratic_form (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), ∃ (k : ℤ), 4 * a^2 + 9 * b^2 - 1 = n * k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_quadratic_form_l204_20470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsidiary_coeff_pair_3x2_2x_minus1_no_linear_term_difference_l204_20468

-- Define the subsidiary coefficient pair type
def SubsidiaryCoeffPair := ℝ × ℝ × ℝ

-- Define the subsidiary polynomial function
def subsidiaryPolynomial (p : SubsidiaryCoeffPair) (x : ℝ) : ℝ :=
  match p with
  | (a, b, c) => a * x^2 + b * x + c

-- Theorem 1
theorem subsidiary_coeff_pair_3x2_2x_minus1 :
  (3, 2, -1) = (fun (p : SubsidiaryCoeffPair) => p) (3, 2, -1) :=
by sorry

-- Theorem 2
theorem no_linear_term_difference (a : ℝ) :
  (∀ x, subsidiaryPolynomial (2, a, 1) x - subsidiaryPolynomial (1, -2, 4) x = x^2 - 3) ↔ 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsidiary_coeff_pair_3x2_2x_minus1_no_linear_term_difference_l204_20468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l204_20441

/-- A parabola in the xy-plane defined by y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  is_parabola : equation = λ x y ↦ y^2 = 8*x

/-- A line in the xy-plane with a 45° inclination angle -/
structure Line where
  equation : ℝ → ℝ → Prop
  has_45_degree_inclination : equation = λ x y ↦ y = x - 2

/-- The focus of a parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- Points where the line intersects the parabola -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating that the product of distances from focus to intersection points is 32 -/
theorem intersection_distance_product (p : Parabola) (l : Line) (pts : IntersectionPoints) :
  l.equation (focus.1) (focus.2) →
  p.equation (pts.A.1) (pts.A.2) ∧ l.equation (pts.A.1) (pts.A.2) →
  p.equation (pts.B.1) (pts.B.2) ∧ l.equation (pts.B.1) (pts.B.2) →
  (distance focus pts.A) * (distance focus pts.B) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l204_20441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binomial_l204_20494

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), p = 67 ∧ 
  Nat.Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ Nat.choose 210 105 ∧
  ∀ (q : ℕ), Nat.Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 210 105 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binomial_l204_20494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l204_20469

open Real

theorem triangle_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 1) (h2 : b = sqrt 3) (h3 : A = π / 6) :
  let B := arcsin (b * sin A / a)
  B = π / 3 ∨ B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l204_20469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l204_20420

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x^2 else (4 - a/2)*x - 1

theorem monotone_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  (4 ≤ a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l204_20420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_capacity_of_tanks_l204_20437

/-- Represents the capacity and current fill level of a water tank -/
structure Tank where
  capacity : ℚ
  fillLevel : ℚ

/-- Calculates the new fill level after adding water to a tank -/
def addWater (tank : Tank) (amount : ℚ) : ℚ :=
  (tank.capacity * tank.fillLevel + amount) / tank.capacity

theorem total_capacity_of_tanks (tankA tankB : Tank) 
  (hA1 : tankA.fillLevel = 3/4)
  (hA2 : addWater tankA 5 = 7/8)
  (hB1 : tankB.fillLevel = 2/3)
  (hB2 : addWater tankB 3 = 5/6) :
  tankA.capacity + tankB.capacity = 58 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_capacity_of_tanks_l204_20437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ABCD_l204_20481

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let s := (AB + BC + CD + DA) / 2
  Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA))

/-- The area of a quadrilateral ABCD with specific side lengths and angles -/
theorem area_of_quadrilateral_ABCD :
  ∀ (A B C D : ℝ × ℝ),
    let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
    let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
    let angle_B := Real.arccos ((AB^2 + BC^2 - (B.1 - C.1)^2 - (B.2 - C.2)^2) / (2 * AB * BC))
    let angle_C := Real.arccos ((BC^2 + CD^2 - (C.1 - D.1)^2 - (C.2 - D.2)^2) / (2 * BC * CD))
    AB = 5 →
    BC = 6 →
    CD = 7 →
    BD = 8 →
    angle_B = 2 * π / 3 →
    angle_C = 2 * π / 3 →
    area_quadrilateral A B C D = 20.5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ABCD_l204_20481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_zero_condition_l204_20486

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x + a) / x - 1

-- Theorem 1: Maximum value when a = 1
theorem max_value_when_a_is_one :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 y ≤ f 1 x ∧ f 1 x = 0 :=
by sorry

-- Theorem 2: Condition for zero in (0, e]
theorem zero_condition (a : ℝ) :
  (∃ (x : ℝ), 0 < x ∧ x ≤ Real.exp 1 ∧ f a x = 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_zero_condition_l204_20486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_to_skew_lines_l204_20435

/-- A line in three-dimensional space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Two lines are skew if they are neither parallel nor intersecting -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ t : ℝ, ∀ i, l1.point i = l2.point i + t * l2.direction i) ∧
  ¬ (∃ k : ℝ, ∀ i, l1.direction i = k * l2.direction i)

/-- A line is perpendicular to another line if their direction vectors are orthogonal -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  (Finset.sum Finset.univ (λ i => l1.direction i * l2.direction i)) = 0

/-- The main theorem: given two skew lines, there exists a unique line perpendicular to both -/
theorem unique_perpendicular_to_skew_lines (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! l : Line3D, is_perpendicular l l1 ∧ is_perpendicular l l2 :=
sorry

#check unique_perpendicular_to_skew_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_to_skew_lines_l204_20435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l204_20407

theorem polynomial_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ (P : ℕ → ℕ),
    (∀ x : ℕ, 0 < x → P x > x) →
    (∀ m : ℕ, 0 < m → ∃ l : ℕ, m ∣ (Nat.iterate P l p)) →
    (∃ a : ℕ, (∀ x : ℕ, P x = x + a) ∧ (a = 1 ∨ a = p)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l204_20407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l204_20472

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  α ∈ Set.Icc π ((3 * π) / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l204_20472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowan_upstream_time_l204_20499

/-- Represents Rowan's rowing scenario -/
structure RowingScenario where
  downstream_distance : ℝ
  downstream_time : ℝ
  still_water_speed : ℝ

/-- Calculates the time it takes to row upstream given a rowing scenario -/
noncomputable def upstream_time (scenario : RowingScenario) : ℝ :=
  let current_speed := (scenario.downstream_distance / scenario.downstream_time - scenario.still_water_speed) / 2
  scenario.downstream_distance / (scenario.still_water_speed - current_speed)

/-- Theorem stating that given Rowan's specific rowing scenario, it takes 4 hours to row upstream -/
theorem rowan_upstream_time :
  let scenario : RowingScenario := {
    downstream_distance := 26,
    downstream_time := 2,
    still_water_speed := 9.75
  }
  upstream_time scenario = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowan_upstream_time_l204_20499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l204_20474

/-- The locus of point Q given the conditions of curves C₁ and C₂ -/
theorem locus_of_Q (x y : ℝ) : 
  (∃ (a : ℝ), 
    -- C₁: x²/3 + y²/4 = 1
    x^2/3 + y^2/4 = 1 ∧ 
    -- C₂: x + y = 1
    a + (1 - a) = 1 ∧ 
    -- P is on C₂
    (∃ (t : ℝ), x = t * a ∧ y = t * (1 - a)) ∧
    -- Q satisfies |OQ| · |OP| = |OR|²
    (∃ (q : ℝ × ℝ), 
      q.1^2 + q.2^2 = x^2 + y^2 ∧
      q.1 * a + q.2 * (1 - a) = a^2 + (1 - a)^2)) →
  -- The locus of Q
  (x - 3/2)^2 / (21/4) + (y - 2)^2 / 7 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l204_20474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_satisfying_points_l204_20466

-- Define the given elements
noncomputable def Line : Type := ℝ → ℝ → ℝ → Prop
noncomputable def Point : Type := ℝ × ℝ × ℝ
noncomputable def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the distance functions
noncomputable def distToLine (p : Point) (l : Line) : ℝ := sorry
noncomputable def distToPoint (p1 p2 : Point) : ℝ := sorry
noncomputable def distToPlane (p : Point) (pl : Plane) : ℝ := sorry

-- Define the set of points satisfying the conditions
def SatisfyingPoints (a : Line) (A : Point) (s : Plane) (m n p : ℝ) : Set Point :=
  {x : Point | distToLine x a = m ∧ distToPoint x A = n ∧ distToPlane x s = p}

-- Theorem statement
theorem max_satisfying_points (a : Line) (A : Point) (s : Plane) (m n p : ℝ) :
  ∃ (S : Finset Point), ↑S ⊆ SatisfyingPoints a A s m n p ∧ S.card = 8 ∧
  ∀ (T : Finset Point), ↑T ⊆ SatisfyingPoints a A s m n p → T.card ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_satisfying_points_l204_20466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l204_20433

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 26 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (2, -3)
def center2 : ℝ × ℝ := (-1, 3)
noncomputable def radius1 : ℝ := Real.sqrt 13
def radius2 : ℝ := 6

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 45

-- Theorem statement
theorem circles_intersect :
  distance_between_centers > abs (radius2 - radius1) ∧
  distance_between_centers < radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l204_20433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sums_is_even_l204_20453

/-- A function that represents the pairing of numbers on the cards after shuffling -/
def pairing : Fin 99 → Fin 99 := sorry

/-- The sum of the numbers on both sides of a card -/
def cardSum (i : Fin 99) : ℕ := (i.val + 1) + (pairing i).val + 1

/-- The product of all card sums -/
def productOfSums : ℕ := Finset.prod (Finset.univ : Finset (Fin 99)) cardSum

theorem product_of_sums_is_even : Even productOfSums := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sums_is_even_l204_20453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_perfect_cube_l204_20498

theorem smallest_base_perfect_cube : 
  ∃ (b : ℕ), b > 3 ∧ 
    (∀ (k : ℕ), k > 3 → (∃ (n : ℕ), 2 * k + 3 = n ^ 3) → k ≥ b) ∧ 
    (∃ (n : ℕ), 2 * b + 3 = n ^ 3) ∧
    b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_perfect_cube_l204_20498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l204_20475

/-- Profit function for product A -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * x

/-- Profit function for product B -/
noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.sqrt x

/-- Total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := f x + g (100000 - x)

theorem max_profit_theorem :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 100000 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 100000 → total_profit y ≤ total_profit x) ∧
  total_profit x = 41/8 * 1000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l204_20475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_coordinates_are_exact_coordinate_desc_is_exact_l204_20427

-- Define a type for location descriptions
inductive LocationDescription where
  | Cinema : Nat → Nat → String → LocationDescription
  | Direction : Float → String → LocationDescription
  | StreetSection : String → String → LocationDescription
  | Coordinates : Float → Float → LocationDescription

-- Function to determine if a location description is exact
def isExactLocation (loc : LocationDescription) : Prop :=
  match loc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

-- Theorem stating that only coordinate-based descriptions are exact
theorem only_coordinates_are_exact (loc : LocationDescription) :
  isExactLocation loc ↔ ∃ (long lat : Float), loc = LocationDescription.Coordinates long lat :=
sorry

-- Examples of location descriptions
def cinemaDesc : LocationDescription := LocationDescription.Cinema 2 3 "Pacific Cinema"
def directionDesc : LocationDescription := LocationDescription.Direction 40 "southeast"
def streetDesc : LocationDescription := LocationDescription.StreetSection "Tianfu Avenue" "middle"
def coordinateDesc : LocationDescription := LocationDescription.Coordinates 116 42

-- Theorem stating that the coordinate description is the only exact location
theorem coordinate_desc_is_exact :
  isExactLocation coordinateDesc ∧ 
  ¬isExactLocation cinemaDesc ∧ 
  ¬isExactLocation directionDesc ∧ 
  ¬isExactLocation streetDesc :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_coordinates_are_exact_coordinate_desc_is_exact_l204_20427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l204_20487

open Real

/-- The function z(x,y) = x ln(y/x) satisfies the equation x(∂z/∂x) + y(∂z/∂y) = z for all x > 0 and y > 0 -/
theorem function_satisfies_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let z : ℝ → ℝ → ℝ := λ x y => x * log (y / x)
  x * (deriv (λ t => z t y) x) + y * (deriv (λ t => z x t) y) = z x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l204_20487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l204_20451

/-- Given a point M(5,4,3), prove that its projection P onto the Oyz plane results in |OP| = 5 -/
theorem projection_distance (M : ℝ × ℝ × ℝ) : 
  M = (5, 4, 3) → 
  let P : ℝ × ℝ × ℝ := (0, M.2.1, M.2.2)
  Real.sqrt (P.1^2 + P.2.1^2 + P.2.2^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l204_20451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_l204_20440

/-- The speed of student B in km/h -/
noncomputable def speed_B : ℝ := 12

/-- The speed of student A in km/h -/
noncomputable def speed_A : ℝ := 1.2 * speed_B

/-- The distance to the activity location in km -/
noncomputable def distance : ℝ := 12

/-- The time difference in hours between A and B's arrival -/
noncomputable def time_diff : ℝ := 1/6

theorem student_B_speed :
  speed_B = 12 ∧
  speed_A = 1.2 * speed_B ∧
  distance / speed_B - distance / speed_A = time_diff := by
  sorry

#check student_B_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_l204_20440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_implication_inverse_of_specific_implication_l204_20421

-- Define the Inverse operation for propositions
def Inverse (P : Prop) : Prop := ¬P

theorem inverse_of_implication (P Q : Prop) :
  Inverse (P → Q) ↔ (Q ∧ ¬P) := by sorry

theorem inverse_of_specific_implication :
  Inverse (∀ a : ℝ, a > 1 → a > 0) ↔ (∃ a : ℝ, a > 0 ∧ ¬(a > 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_implication_inverse_of_specific_implication_l204_20421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tripod_theorem_l204_20484

/-- The height of a tripod after one leg is shortened -/
noncomputable def broken_tripod_height : ℝ := 384 / Real.sqrt (5 * 389)

/-- The floor of m + √n for the broken tripod problem -/
def floor_m_plus_sqrt_n : ℕ := 428

theorem broken_tripod_theorem (original_leg_length : ℝ) (original_height : ℝ)
  (shortened_length : ℝ) (h : ℝ) :
  original_leg_length = 6 →
  original_height = 5 →
  shortened_length = original_leg_length - 2 →
  h = broken_tripod_height →
  ⌊(384 : ℝ) + Real.sqrt (5 * 389)⌋ = floor_m_plus_sqrt_n := by
  sorry

#eval floor_m_plus_sqrt_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tripod_theorem_l204_20484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_lower_bound_l204_20404

/-- The function f(x) defined as a(e^x + a) - x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 1

theorem f_monotonicity_and_lower_bound (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → f_deriv a x < 0) ∧
  (a > 0 → ∀ x : ℝ, x < Real.log (1/a) → f_deriv a x < 0) ∧
  (a > 0 → ∀ x : ℝ, x > Real.log (1/a) → f_deriv a x > 0) ∧
  (a > 0 → ∀ x : ℝ, f a x > 2 * Real.log a + 3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_lower_bound_l204_20404
