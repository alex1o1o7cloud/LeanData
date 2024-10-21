import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1190_119050

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ x, f ((-Real.pi / 6) + x) = f ((-Real.pi / 6) - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1190_119050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_tangent_inverse_l1190_119065

theorem least_positive_tangent_inverse (a b : ℝ) (x : ℝ) :
  (Real.tan x = a / b) →
  (Real.tan (2 * x) = b / (a + b)) →
  (∃ k : ℝ, x = Real.arctan k ∧ k > 0 ∧ ∀ k' > 0, x = Real.arctan k' → k ≤ k') →
  (∃ k : ℝ, x = Real.arctan k ∧ k = 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_tangent_inverse_l1190_119065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_100_factorial_l1190_119030

-- Define a function to calculate the last two nonzero digits of a factorial
def lastTwoNonzeroDigits (n : Nat) : Nat :=
  n.factorial % 100

-- Theorem statement
theorem last_two_nonzero_digits_100_factorial :
  lastTwoNonzeroDigits 100 = 76 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_100_factorial_l1190_119030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fourteen_l1190_119003

theorem repeating_decimal_fourteen : ∃ (x : ℚ), x = 14 / 99 ∧ x = 0.14 + 0.0014 / (1 - 1/100) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fourteen_l1190_119003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l1190_119090

theorem tan_half_angle (α : Real) (h1 : π/2 < α ∧ α < π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 2/10) : Real.tan (α/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l1190_119090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1190_119094

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g x = -g (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1190_119094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_difference_l1190_119026

/-- The list price of Laptop Y in dollars -/
def list_price : ℚ := 59.99

/-- The discount offered by Tech Bargains in dollars -/
def tech_bargains_discount : ℚ := 15

/-- The discount percentage offered by Digital Deal -/
def digital_deal_discount_percent : ℚ := 30

/-- The price at Tech Bargains in dollars -/
def tech_bargains_price : ℚ := list_price - tech_bargains_discount

/-- The price at Digital Deal in dollars -/
def digital_deal_price : ℚ := list_price * (1 - digital_deal_discount_percent / 100)

/-- The price difference in cents -/
def price_difference_cents : ℤ := 
  ⌊(tech_bargains_price - digital_deal_price) * 100⌋

theorem laptop_price_difference :
  price_difference_cents = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_difference_l1190_119026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_square_condition_l1190_119046

def x : ℕ → ℕ
  | 0 => 0  -- Adding a case for 0
  | 1 => 1
  | 2 => 4
  | (n + 3) => 4 * x (n + 2) - x (n + 1)

theorem sequence_perfect_square_condition :
  ∃! m : ℕ, ∀ n : ℕ, ∃ k : ℕ, 3 * (x n)^2 + m = k^2 ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_square_condition_l1190_119046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_correct_l1190_119018

/-- A move is either horizontal (permuting elements within rows) or vertical (permuting elements within columns) -/
inductive Move
| horizontal
| vertical

/-- The minimum number of moves required to achieve any permutation on an m×n table -/
def min_moves (m n : ℕ) : ℕ :=
  if m = 1 ∨ n = 1 then 1 else 3

/-- Predicate to check if a list of moves achieves a given permutation on an m×n table -/
def moves_achieve_permutation (m n : ℕ) (moves : List Move) (perm : Fin (m * n) → Fin (m * n)) : Prop :=
  sorry

/-- Theorem: The minimum number of moves required to achieve any permutation on an m×n table,
    such that some permutation cannot be achieved in fewer moves, is given by min_moves -/
theorem min_moves_correct (m n : ℕ) :
  (∀ (perm : Fin (m * n) → Fin (m * n)), ∃ (moves : List Move), moves.length ≤ min_moves m n ∧
    moves_achieve_permutation m n moves perm) ∧
  (∃ (perm : Fin (m * n) → Fin (m * n)), ∀ (moves : List Move),
    moves.length < min_moves m n → ¬ moves_achieve_permutation m n moves perm) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_correct_l1190_119018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orbit_dense_in_interval_l1190_119033

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 (Real.sqrt 2 / 2) then x + (2 - Real.sqrt 2) / 2
  else if x ∈ Set.Ico (Real.sqrt 2 / 2) 1 then x - Real.sqrt 2 / 2
  else x  -- Default case to make the function total

-- Define the n-th iteration of f
noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

-- The main theorem
theorem orbit_dense_in_interval :
  ∀ (a b : ℝ), 0 < a → a < b → b < 1 →
  ∃ (x : ℝ) (n : ℕ), x ∈ Set.Ioo a b ∧ f_iter n x ∈ Set.Ioo a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orbit_dense_in_interval_l1190_119033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l1190_119087

/-- The volume of a regular triangular pyramid -/
noncomputable def triangular_pyramid_volume (R φ : ℝ) : ℝ :=
  (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)

/-- Theorem: The volume of a regular triangular pyramid circumscribed by a sphere of radius R
    and with a dihedral angle φ at the vertex is (8/27) * R³ * sin²(φ/2) * (1 + 2 cos φ) -/
theorem regular_triangular_pyramid_volume (R φ : ℝ) (h1 : R > 0) (h2 : 0 < φ ∧ φ < Real.pi) :
  ∃ (V : ℝ), V = triangular_pyramid_volume R φ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l1190_119087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tg_plus_ctg_values_l1190_119012

noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

noncomputable def cosec (x : ℝ) : ℝ := 1 / Real.sin x

noncomputable def tg (x : ℝ) : ℝ := Real.tan x

noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

theorem tg_plus_ctg_values (x : ℝ) :
  sec x - cosec x = Real.sqrt 15 → tg x + ctg x = -3 ∨ tg x + ctg x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tg_plus_ctg_values_l1190_119012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_overtime_pay_l1190_119084

/-- Represents Lewis's earnings during the harvest season -/
structure HarvestEarnings where
  regularPayPerWeek : ℚ
  harvestWeeks : ℕ
  totalEarnings : ℚ

/-- Calculates Lewis's overtime pay per week during the harvest season -/
def overtimePayPerWeek (e : HarvestEarnings) : ℚ :=
  (e.totalEarnings - e.regularPayPerWeek * e.harvestWeeks) / e.harvestWeeks

/-- Theorem stating that Lewis's overtime pay per week is approximately $939.27 -/
theorem lewis_overtime_pay (e : HarvestEarnings)
    (h1 : e.regularPayPerWeek = 28)
    (h2 : e.harvestWeeks = 1091)
    (h3 : e.totalEarnings = 1054997) :
    ‖overtimePayPerWeek e - 939.27‖ < 1/100 := by
  sorry

#eval overtimePayPerWeek { regularPayPerWeek := 28, harvestWeeks := 1091, totalEarnings := 1054997 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_overtime_pay_l1190_119084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_number_l1190_119007

def markOutProcess (n : ℕ) : ℕ := 
  if n ≤ 150 then
    let rec process (remaining : List ℕ) (skip : ℕ) : ℕ :=
      match remaining with
      | [] => 0  -- This case should never occur if the process is correct
      | [x] => x
      | _ => process (remaining.filter (fun x => (remaining.indexOf x + 1) % (skip + 2) ≠ 0)) (skip + 1)
    process (List.range n) 1
  else 0

theorem last_remaining_number :
  markOutProcess 150 = 128 := by
  sorry

#eval markOutProcess 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_number_l1190_119007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1190_119032

open Real

theorem triangle_problem (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (cos α, sin α)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  AC.1 * BC.1 + AC.2 * BC.2 = -1 →
  (2 * sin α ^ 2 + sin (2 * α)) / (1 + tan α) = -9/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1190_119032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_motion_l1190_119061

/-- Represents a crank-slider mechanism -/
structure CrankSlider where
  crank_length : ℝ
  connecting_rod_length : ℝ
  am_ratio : ℝ
  angular_velocity : ℝ

/-- Equations of motion for point M -/
noncomputable def equations_of_motion (cs : CrankSlider) (t : ℝ) : ℝ × ℝ :=
  (cs.crank_length * (1 - cs.am_ratio) * Real.cos (cs.angular_velocity * t),
   cs.connecting_rod_length * cs.am_ratio * Real.sin (cs.angular_velocity * t))

/-- Velocity of point M -/
noncomputable def velocity_of_m (cs : CrankSlider) (t : ℝ) : ℝ :=
  cs.connecting_rod_length * cs.am_ratio * cs.angular_velocity *
    Real.sqrt (1 + (cs.crank_length / cs.connecting_rod_length)^2 * (1 / cs.am_ratio - 1)^2 *
      Real.cos (cs.angular_velocity * t)^2)

theorem crank_slider_motion (cs : CrankSlider)
    (h1 : cs.crank_length = 90)
    (h2 : cs.connecting_rod_length = 90)
    (h3 : cs.am_ratio = 2/3)
    (h4 : cs.angular_velocity = 10) :
  ∀ t : ℝ,
    equations_of_motion cs t = (30 * Real.cos (10 * t), 150 * Real.sin (10 * t)) ∧
    velocity_of_m cs t = 300 * Real.sqrt (1 + 24 * Real.cos (10 * t)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_motion_l1190_119061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1190_119081

-- Define the complex number z
noncomputable def z : ℂ := Complex.I

-- Define the condition (1-i)z = 1+i
axiom z_condition : (1 - Complex.I) * z = 1 + Complex.I

-- Define the point corresponding to |z|-i
noncomputable def point : ℂ := Complex.abs z - Complex.I

-- Theorem to prove
theorem point_in_fourth_quadrant :
  point.re > 0 ∧ point.im < 0 := by
  sorry

-- Additional lemma to show that z = i
lemma z_is_i : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1190_119081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_is_specific_vector_l1190_119039

noncomputable def line (x : ℝ) : ℝ := (3/2) * x - 1

noncomputable def vector_on_line (a : ℝ) : ℝ × ℝ := (a, line a)

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  ((dot_product / magnitude_squared) * w.1, (dot_product / magnitude_squared) * w.2)

theorem constant_projection_is_specific_vector :
  ∀ w : ℝ × ℝ, (∃ p : ℝ × ℝ, ∀ a : ℝ, projection (vector_on_line a) w = p) →
  (∃ p : ℝ × ℝ, ∀ a : ℝ, projection (vector_on_line a) w = p ∧ p = (6/13, -4/13)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_is_specific_vector_l1190_119039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_to_2i_l1190_119005

open Complex

theorem smallest_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I))) :
  Complex.abs (z + 2*I) ≥ (7:ℝ)/2 ∧ ∃ w : ℂ, Complex.abs (w^2 + 9) = Complex.abs (w * (w + 3*I)) ∧ Complex.abs (w + 2*I) = (7:ℝ)/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_to_2i_l1190_119005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_zero_l1190_119014

/-- The function f(x) defined on the closed interval [0,π] -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x + 5 * a / 8 - 5 / 2

/-- Theorem stating the existence of a such that f(x) has a maximum value of 0 on [0,π] -/
theorem exists_a_max_zero :
  ∃ a : ℝ, a = 3 / 2 ∧ 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ π → f a x ≤ 0) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ π ∧ f a x = 0) :=
by
  -- We'll use a = 3/2 as our witness
  use 3 / 2
  constructor
  · -- Prove a = 3/2
    rfl
  constructor
  · -- Prove ∀ x : ℝ, 0 ≤ x ∧ x ≤ π → f a x ≤ 0
    sorry
  · -- Prove ∃ x : ℝ, 0 ≤ x ∧ x ≤ π ∧ f a x = 0
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_zero_l1190_119014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_value_l1190_119020

/-- A function f is symmetric about the origin if f(-x) = -f(x) for all x in the domain of f -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (|x-2|+a)/√(4-x²) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (abs (x - 2) + a) / Real.sqrt (4 - x^2)

theorem symmetry_implies_value (a : ℝ) :
  SymmetricAboutOrigin (f a) → f a (a/2) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_value_l1190_119020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l1190_119062

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_x_coordinate (a : ℝ) :
  (∀ x, f' a x = - f' a (-x)) →  -- f' is an odd function
  (∃ x, f' a x = 3/2) →          -- there exists a point with slope 3/2
  (∃ x, f' a x = 3/2 ∧ x = Real.log 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l1190_119062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_cd_product_l1190_119008

/-- An equilateral triangle with vertices at (0,0), (c, 17), and (d, 53) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ
  is_equilateral : (c^2 + 17^2 = d^2 + 53^2) ∧ (c^2 + 17^2 = c^2 + d^2 - 2*c*d*Real.cos (2*π/3))

/-- The product of c and d in the equilateral triangle equals 16011/9 -/
theorem equilateral_triangle_cd_product (t : EquilateralTriangle) : t.c * t.d = 16011 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_cd_product_l1190_119008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l1190_119093

/-- Represents the principal amount in Rupees -/
def principal (p : ℝ) : Prop := true

/-- Calculates simple interest for a given principal, rate, and time -/
def simple_interest (p r t : ℝ) : ℝ := p * r * t

/-- Calculates compound interest for a given principal, rate, number of compounds per year, and time -/
noncomputable def compound_interest (p r n t : ℝ) : ℝ := p * ((1 + r / n) ^ (n * t) - 1)

/-- The theorem stating the principal amount that satisfies the given conditions -/
theorem principal_amount : ∃ p : ℝ, 
  principal p ∧ 
  (compound_interest p 0.1 2 1 - simple_interest p 0.1 1 = 3) ∧
  p = 1200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l1190_119093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l1190_119068

-- Define the sample correlation coefficient
noncomputable def sample_correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define the strength of linear correlation
noncomputable def linear_correlation_strength (r : ℝ) : ℝ := sorry

-- Define the mean of a list of real numbers
noncomputable def mean (l : List ℝ) : ℝ := sorry

-- Define a point on a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a linear regression model
structure LinearRegressionModel where
  slope : ℝ
  intercept : ℝ

-- Define a residual plot
noncomputable def residual_plot (model : LinearRegressionModel) (data : List Point) : List ℝ := sorry

-- Define the width of a residual plot band
noncomputable def residual_plot_band_width (residuals : List ℝ) : ℝ := sorry

-- Define the goodness of fit for a model
noncomputable def model_fit_quality (width : ℝ) : ℝ := sorry

theorem correlation_coefficient_properties
  (x y : List ℝ) (model : LinearRegressionModel) :
  let r := sample_correlation_coefficient x y
  let x_mean := mean x
  let y_mean := mean y
  let data := List.zipWith (λ x y => Point.mk x y) x y
  let residuals := residual_plot model data
  let band_width := residual_plot_band_width residuals
  -- As |r| approaches 1, linear correlation strengthens
  (∀ ε > 0, ∃ δ > 0, ∀ r', |r'| > 1 - δ → linear_correlation_strength r' > 1 - ε) ∧
  -- As |r| approaches 0, linear correlation weakens
  (∀ ε > 0, ∃ δ > 0, ∀ r', |r'| < δ → linear_correlation_strength r' < ε) ∧
  -- The point (x̄, ȳ) lies on the regression line
  (model.slope * x_mean + model.intercept = y_mean) ∧
  -- A narrower residual plot band indicates better model fit
  (∀ width₁ width₂, width₁ < width₂ → model_fit_quality width₁ > model_fit_quality width₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l1190_119068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_b_l1190_119060

def a : Fin 3 → ℝ := ![-1, -3, 2]
def b : Fin 3 → ℝ := ![1, 2, 0]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

theorem dot_product_a_b :
  dot_product a b = -7 := by
  simp [dot_product, a, b]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_a_b_l1190_119060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l1190_119043

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
noncomputable def line1 : Line2D :=
  { point := (2, 2),
    direction := (3, -4) }

/-- The second line -/
noncomputable def line2 : Line2D :=
  { point := (4, -6),
    direction := (5, 3) }

/-- A point lies on a line if it satisfies the line's parametric equation -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

/-- The intersection point of the two lines -/
noncomputable def intersectionPoint : ℝ × ℝ := (160/29, -160/29)

/-- Theorem stating that the intersection point lies on both lines and is unique -/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 → p = intersectionPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l1190_119043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_price_is_30_l1190_119071

/-- Represents the price structure and conditions of a waterpark admission system. -/
structure WaterparkAdmission where
  adult_price : ℝ
  kid_price : ℝ
  discount_rate : ℝ
  soda_price : ℝ
  total_paid : ℝ
  num_adults : ℕ
  num_kids : ℕ

/-- Calculates the total cost of admission before discount. -/
def total_cost_before_discount (w : WaterparkAdmission) : ℝ :=
  w.adult_price * (w.num_adults : ℝ) + w.kid_price * (w.num_kids : ℝ)

/-- Calculates the discounted total cost of admission. -/
def discounted_total_cost (w : WaterparkAdmission) : ℝ :=
  total_cost_before_discount w * (1 - w.discount_rate)

/-- Theorem stating that the adult admission price is $30 given the problem conditions. -/
theorem adult_ticket_price_is_30 (w : WaterparkAdmission) :
  w.adult_price = 30 ∧
  w.kid_price = w.adult_price / 2 ∧
  w.discount_rate = 0.2 ∧
  w.soda_price = 5 ∧
  w.total_paid = 197 ∧
  w.num_adults = 6 ∧
  w.num_kids = 4 →
  w.adult_price = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_price_is_30_l1190_119071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1190_119051

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (15, 0)

-- Define the initial slopes of the lines
def slope_A : ℝ := 2
def slope_C : ℝ := -2

-- Define the rotation rate
noncomputable def rotation_rate : ℝ := 10 * (Real.pi / 180)  -- 10 degrees per second in radians

-- Define a function to represent a rotating line
noncomputable def rotating_line (point : ℝ × ℝ) (initial_slope : ℝ) (time : ℝ) : 
  ℝ → ℝ := λ x ↦ sorry

-- Define the triangle formed by the intersections
noncomputable def triangle_area (time : ℝ) : ℝ := sorry

-- State the theorem
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 0.5 ∧ 
  ∀ (t : ℝ), triangle_area t ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1190_119051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l1190_119088

theorem compute_expression (a b c : ℚ) : 
  a = 4/7 → b = 5/6 → c = 3/8 → (a^(-3 : ℤ) * b^2) * c^(-1 : ℤ) = 343/346 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l1190_119088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_identification_l1190_119006

-- Define the four functions
def f₁ (x : ℝ) : ℝ := (x - 1)^2
noncomputable def f₂ (x : ℝ) : ℝ := Real.sqrt 2 * x^2 - 1
def f₃ (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1
def f₄ (x : ℝ) : ℝ := (x + 1)^2 - x^2

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- State the theorem
theorem quadratic_identification :
  is_quadratic f₁ ∧ is_quadratic f₂ ∧ is_quadratic f₃ ∧ ¬is_quadratic f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_identification_l1190_119006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_theorem_l1190_119097

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

-- Define a line segment
def Segment (A B : Point) : Set Point :=
  {P : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y)⟩}

-- Define a square
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point
  is_square : Prop -- placeholder for conditions that ABCD forms a square

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Main theorem
theorem square_intersection_theorem 
  (A B M : Point)
  (square_AM : Square)
  (square_MB : Square)
  (circle_AM : Circle)
  (circle_MB : Circle)
  (N : Point)
  (h1 : M ∈ Segment A B)
  (h2 : square_AM.A = A ∧ square_AM.B = M)
  (h3 : square_MB.A = M ∧ square_MB.B = B)
  (h4 : N ∈ {P : Point | (P.x - circle_AM.center.x)^2 + (P.y - circle_AM.center.y)^2 = circle_AM.radius^2} ∧
        N ∈ {P : Point | (P.x - circle_MB.center.x)^2 + (P.y - circle_MB.center.y)^2 = circle_MB.radius^2})
  (h5 : Prop) -- placeholder for condition that squares are on the same side of AB
  : 
  (∃ D : Point, D = square_MB.D ∧ N ∈ Segment A D) ∧ 
  (∃ angle : ℝ, angle = Real.pi / 2) -- placeholder for right angle condition
  := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_theorem_l1190_119097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluation_probabilities_l1190_119041

/-- Represents the evaluation outcome for a group of teachers -/
structure EvaluationOutcome where
  total_teachers : Nat
  it_experts : Nat
  selected_teachers : Nat
  it_expert_pass_prob : ℚ
  other_pass_prob : ℚ

/-- Defines the conditions of the evaluation -/
def school_evaluation : EvaluationOutcome :=
  { total_teachers := 10
  , it_experts := 1
  , selected_teachers := 3
  , it_expert_pass_prob := 3/4
  , other_pass_prob := 1/2
  }

/-- The probability of exactly two teachers passing the evaluation -/
noncomputable def prob_two_pass (e : EvaluationOutcome) : ℚ :=
  sorry

/-- The probability of the school passing the evaluation -/
noncomputable def prob_school_pass (e : EvaluationOutcome) : ℚ :=
  sorry

/-- The probability of the IT expert being selected given the school passed -/
noncomputable def prob_it_expert_given_pass (e : EvaluationOutcome) : ℚ :=
  sorry

/-- Main theorem stating the probabilities to be proved -/
theorem evaluation_probabilities :
  prob_two_pass school_evaluation = 63/160 ∧
  prob_it_expert_given_pass school_evaluation = 15/43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluation_probabilities_l1190_119041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marias_workday_end_l1190_119045

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : TimeOfDay) : Int :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : TimeOfDay) (m : Int) : TimeOfDay :=
  let totalMinutes : Int := t.hours * 60 + t.minutes + m
  { hours := (totalMinutes / 60 % 24).toNat,
    minutes := (totalMinutes % 60).toNat,
    valid := by sorry }

theorem marias_workday_end 
  (start : TimeOfDay)
  (lunch : TimeOfDay)
  (workHours : Nat)
  (lunchDuration : Nat)
  (h1 : start.hours = 7 ∧ start.minutes = 25)
  (h2 : lunch.hours = 12 ∧ lunch.minutes = 0)
  (h3 : workHours = 9)
  (h4 : lunchDuration = 1) :
  let endTime := addMinutes lunch (lunchDuration * 60 + workHours * 60 - timeDifference start lunch)
  endTime.hours = 17 ∧ endTime.minutes = 25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marias_workday_end_l1190_119045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_A_l1190_119047

-- Define the set A
def A : Finset Char := {'a', 'b', 'c'}

-- Theorem stating that the number of proper subsets of A is 7
theorem proper_subsets_of_A : (Finset.powerset A).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_A_l1190_119047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1190_119010

noncomputable def f (x : ℝ) := 2 * Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.cos (2 * x) ^ 2 - 3

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
    T = π / 2 ∧
    (∀ x ∈ Set.Icc (π / 16) (3 * π / 16), f x ≥ -Real.sqrt 2 / 2 - 1) ∧
    f (3 * π / 16) = -Real.sqrt 2 / 2 - 1 ∧
    (∀ x ∈ Set.Icc (π / 16) (3 * π / 16), f x = -Real.sqrt 2 / 2 - 1 → x = 3 * π / 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1190_119010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_primes_with_prime_remainder_l1190_119000

/-- Counts the number of prime numbers between 30 and 80 (inclusive) 
    that have a prime remainder when divided by 12. -/
theorem count_primes_with_prime_remainder : 
  (Finset.filter (fun p => Nat.Prime p ∧ 
                           30 ≤ p ∧ 
                           p ≤ 80 ∧ 
                           Nat.Prime (p % 12)) (Finset.range 81)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_primes_with_prime_remainder_l1190_119000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l1190_119023

theorem parallel_vectors_tan_alpha (α : ℝ) : 
  let a : Fin 2 → ℝ := ![3, 4]
  let b : Fin 2 → ℝ := ![Real.sin α, Real.cos α]
  (a 0 * b 1 = a 1 * b 0) → Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l1190_119023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1190_119028

/-- An acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  side_angle_relation : a * Real.sin ((A + C) / 2) = b * Real.sin A
  c_eq_one : c = 1

theorem acute_triangle_properties (t : AcuteTriangle) :
  t.B = π/3 ∧ Real.sqrt 3/8 < (1/2 * t.a * t.b * Real.sin t.C) ∧ 
  (1/2 * t.a * t.b * Real.sin t.C) < Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1190_119028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_equation_l1190_119092

def B : ℂ := 3 - 2*Complex.I
def Q : ℂ := -5
def R : ℂ := 2*Complex.I
def T : ℂ := -1 + 5*Complex.I

theorem complex_sum_equation : B - Q + R + T = -3 + 5*Complex.I := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_equation_l1190_119092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_A_to_BCA₁_plane_l1190_119072

-- Define the prism
structure Prism where
  a : ℝ
  α : ℝ
  H : ℝ

-- Define the distance function
noncomputable def distance_to_plane (p : Prism) : ℝ :=
  (p.a * p.H * Real.sin p.α) / Real.sqrt (p.a^2 * Real.sin p.α^2 + p.H^2)

-- Theorem statement
theorem distance_from_A_to_BCA₁_plane (p : Prism) :
  distance_to_plane p = (p.a * p.H * Real.sin p.α) / Real.sqrt (p.a^2 * Real.sin p.α^2 + p.H^2) :=
by
  -- Unfold the definition of distance_to_plane
  unfold distance_to_plane
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_A_to_BCA₁_plane_l1190_119072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1190_119037

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- A cube whose vertices are at the midpoints of a regular dodecahedron's edges -/
structure MidpointCube (P : RegularDodecahedron) where

/-- The volume of a regular dodecahedron -/
noncomputable def volumeDodecahedron (P : RegularDodecahedron) : ℝ :=
  (15 + 7 * Real.sqrt 5) * P.sideLength ^ 3 / 4

/-- The volume of a cube whose vertices are at the midpoints of a regular dodecahedron's edges -/
noncomputable def volumeMidpointCube (P : RegularDodecahedron) (Q : MidpointCube P) : ℝ :=
  (P.sideLength ^ 3 * Real.sqrt (8 * (5 - Real.sqrt 5) ^ 3)) / 8

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (P : RegularDodecahedron) (Q : MidpointCube P) :
  volumeDodecahedron P / volumeMidpointCube P Q = (15 + 7 * Real.sqrt 5) / Real.sqrt (8 * (5 - Real.sqrt 5) ^ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1190_119037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_sequence_theorem_optimization_theorem_geometry_theorem_l1190_119089

def vector_problem (a b c : ℝ × ℝ) : Prop :=
  let a := (1, -3)
  let b := (-2, 6)
  let angle := 60 * Real.pi / 180
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) = Real.cos angle ∧
  c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = -10 →
  c.1^2 + c.2^2 = 40

theorem vector_theorem : ∀ c : ℝ × ℝ, vector_problem c c c := by
  sorry

-- Sequence problem
def sequence_problem (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 0 ∧
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
  a 2017 = 0

theorem sequence_theorem : ∀ a : ℕ → ℤ, sequence_problem a := by
  sorry

-- Optimization problem
def optimization_problem (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y = 1 →
  (2 / x + x / y) ≥ 2 + 2 * Real.sqrt 2

theorem optimization_theorem : ∀ x y : ℝ, optimization_problem x y := by
  sorry

-- Geometry problem
def geometry_problem (R r : ℝ) : Prop :=
  R^2 = 29 / 4 ∧ r = 1 →
  4 * Real.pi * r^2 + 4 * Real.pi * R^2 = 33 * Real.pi

theorem geometry_theorem : ∀ R r : ℝ, geometry_problem R r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_sequence_theorem_optimization_theorem_geometry_theorem_l1190_119089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_properties_l1190_119035

-- Define the set T of all non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation ◇
def diamond (x y : ℝ) : ℝ := 3 * x * y

-- Theorem stating the properties of ◇
theorem diamond_properties :
  (∀ x y, x ∈ T → y ∈ T → diamond x y = diamond y x) ∧ 
  (∃ x y z, x ∈ T ∧ y ∈ T ∧ z ∈ T ∧ diamond x (diamond y z) ≠ diamond (diamond x y) z) ∧
  (∀ x, x ∈ T → diamond x (1/3) = x ∧ diamond (1/3) x = x) ∧
  (∀ x, x ∈ T → ∃ y, y ∈ T ∧ diamond x y = 1/3 ∧ diamond y x = 1/3) ∧
  (∀ x, x ∈ T → diamond x (1/(3*x)) = 1 ∧ diamond (1/(3*x)) x = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_properties_l1190_119035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l1190_119034

def CardSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 8, 9}

def IsValidArrangement (arrangement : List ℕ) : Prop :=
  arrangement.toFinset = CardSet ∧
  ∀ i : Fin (arrangement.length - 1),
    (10 * arrangement[i.val] + arrangement[i.val + 1]) % 7 = 0

theorem no_valid_arrangement : ¬ ∃ arrangement : List ℕ, IsValidArrangement arrangement := by
  sorry

#check no_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l1190_119034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bamboo_probability_l1190_119098

def bamboo_lengths : List Float := [2.5, 2.6, 2.7, 2.8, 2.9]

def count_pairs_with_difference (lengths : List Float) (diff : Float) : Nat :=
  lengths.foldl (λ count x => count + (lengths.filter (λ y => Float.abs (y - x) == diff)).length) 0 / 2

def total_pairs (n : Nat) : Nat := n * (n - 1) / 2

theorem bamboo_probability :
  let n := bamboo_lengths.length
  let desired_pairs := count_pairs_with_difference bamboo_lengths 0.3
  let total_pairs := total_pairs n
  (desired_pairs : Rat) / total_pairs = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bamboo_probability_l1190_119098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1190_119066

noncomputable def arithmeticSequenceSum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties (a₁ d : ℝ) (h : d ≠ 0) :
  (∃ M : ℝ, ∀ n : ℕ, arithmeticSequenceSum a₁ d n ≤ M ↔ d < 0) ∧
  (∀ n : ℕ+, 0 < arithmeticSequenceSum a₁ d n → 
    ∀ m : ℕ+, m < n → arithmeticSequenceSum a₁ d m < arithmeticSequenceSum a₁ d n) ∧
  ¬(∀ n : ℕ+, (∀ m : ℕ+, m < n → 
    arithmeticSequenceSum a₁ d m < arithmeticSequenceSum a₁ d n) → 
      0 < arithmeticSequenceSum a₁ d n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1190_119066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_running_time_is_four_l1190_119021

/-- Represents the jogging scenario with given parameters -/
structure JoggingScenario where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  reducedSpeed : ℝ

/-- Calculates the time spent running energetically -/
noncomputable def energeticRunningTime (scenario : JoggingScenario) : ℝ :=
  (scenario.totalDistance - scenario.reducedSpeed * scenario.totalTime) /
  (scenario.initialSpeed - scenario.reducedSpeed)

/-- Theorem stating that the energetic running time is 4 hours for the given scenario -/
theorem energetic_running_time_is_four :
  let scenario : JoggingScenario := {
    totalDistance := 64,
    totalTime := 8,
    initialSpeed := 10,
    reducedSpeed := 6
  }
  energeticRunningTime scenario = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_running_time_is_four_l1190_119021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_digits_l1190_119031

def digit_sequence : ℕ → ℕ 
  | n => (n - 1) % 6 + 1

def remove_every_nth (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ 
  | k => seq (k + (k / (n - 1)))

def final_sequence : ℕ → ℕ :=
  remove_every_nth 5 (remove_every_nth 3 (remove_every_nth 2 digit_sequence))

theorem sum_of_specific_digits :
  final_sequence 3047 + final_sequence 3048 + final_sequence 3049 = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_digits_l1190_119031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_multiples_of_two_and_three_l1190_119063

theorem count_non_multiples_of_two_and_three : 
  ∃ (S : Finset ℕ), 
    S = Finset.range 51 \ {0} ∧ 
    (S.filter (λ n ↦ n % 2 ≠ 0 ∧ n % 3 ≠ 0)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_multiples_of_two_and_three_l1190_119063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_valid_z_is_18_l1190_119019

def is_valid_z (z : ℕ) : Bool :=
  z < 10 && (36000 + z * 100 + 72) % 6 = 0 && (36000 + z * 100 + 72) % 4 = 0

def sum_valid_z : ℕ := (List.range 10).filter is_valid_z |>.sum

theorem sum_valid_z_is_18 : sum_valid_z = 18 := by
  sorry

#eval sum_valid_z

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_valid_z_is_18_l1190_119019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1190_119055

theorem solve_exponential_equation :
  ∃ x : ℚ, (3 : ℝ) ^ (2 * (x : ℝ)) = Real.sqrt 27 ∧ x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1190_119055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1190_119080

theorem equation_solution (x : ℝ) : 
  8.487 * (Real.sin x ^ 2 - Real.tan x ^ 2) / (Real.cos x ^ 2 - (1 / Real.tan x) ^ 2) - 
  Real.tan x ^ 6 + Real.tan x ^ 4 - Real.tan x ^ 2 = 0 →
  ∃ k : ℤ, x = π / 4 * (2 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1190_119080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1190_119027

theorem cos_double_angle_special_case (α : Real) :
  Real.sin (π + α) = -1/2 →
  π/2 < α ∧ α < π →
  Real.cos (2*α) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1190_119027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_specific_rhombus_l1190_119069

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
noncomputable def inscribed_circle_radius (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / (4 * Real.sqrt ((d1/2)^2 + (d2/2)^2))

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals 8 and 18 is 36/√97 -/
theorem inscribed_circle_radius_for_specific_rhombus :
  inscribed_circle_radius 8 18 = 36 / Real.sqrt 97 := by
  -- Expand the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_specific_rhombus_l1190_119069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_multiple_l1190_119024

/-- Given two distinct positive integers a and b, and a positive integer n,
    there exists a finite sequence of operations that results in a multiple of n
    appearing on the blackboard. Each operation adds gcd(x, y) + lcm(x, y) to the board
    for some x and y already on the board. -/
theorem blackboard_multiple (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hn : n > 0) :
  ∃ (k : ℕ) (seq : Fin (k + 2) → ℕ), 
    (seq 0 = a ∧ seq 1 = b) ∧
    (∀ i : Fin (k + 2), i.val ≥ 2 → 
      ∃ (x y : Fin (k + 2)), 
        seq i = Nat.gcd (seq x) (seq y) + Nat.lcm (seq x) (seq y) ∧ 
        seq x ≠ seq y) ∧
    (∃ (i : Fin (k + 2)), n ∣ seq i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_multiple_l1190_119024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_product_l1190_119048

/-- Regular octagon in the complex plane -/
structure RegularOctagon where
  vertices : Fin 8 → ℂ
  is_regular : ∀ i j : Fin 8, Complex.abs (vertices i - vertices j) = Complex.abs (vertices 0 - vertices 1)
  q1_position : vertices 0 = 2
  q5_position : vertices 4 = 6
  center_at_4 : ∀ i : Fin 8, vertices i - 4 = (vertices i - 4)^8

/-- The product of complex numbers representing the vertices of a regular octagon -/
def octagon_product (O : RegularOctagon) : ℂ :=
  Finset.prod (Finset.univ : Finset (Fin 8)) (λ i => O.vertices i)

/-- Theorem: The product of complex numbers representing the vertices of a regular octagon 
    with Q₁ at (2,0) and Q₅ at (6,0) is equal to 65535 -/
theorem regular_octagon_product (O : RegularOctagon) : 
  octagon_product O = 65535 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_product_l1190_119048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_m_values_l1190_119036

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The theorem stating the possible values of m when |PQ| is minimized -/
theorem min_distance_m_values (m n a : ℝ) :
  ellipse m n →
  a > 0 →
  a ≠ Real.sqrt 3 →
  (∀ x y, ellipse x y → distance m n a 0 ≤ distance x y a 0) →
  m = Real.sqrt 3 ∨ m = 3*a := by
  sorry

#check min_distance_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_m_values_l1190_119036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_part_trip_average_speed_l1190_119004

/-- Calculates the average speed of a two-part trip -/
noncomputable def average_speed (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  total_distance / (distance1 / speed1 + (total_distance - distance1) / speed2)

/-- Theorem: The average speed of a 60 km trip with first half at 60 km/h and second half at 30 km/h is 40 km/h -/
theorem two_part_trip_average_speed :
  average_speed 60 30 60 30 = 40 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_part_trip_average_speed_l1190_119004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_potatoes_l1190_119013

theorem initial_potatoes (initial_tomatoes picked_potatoes remaining_total initial_potatoes : ℕ) 
  (h1 : initial_tomatoes = 175)
  (h2 : picked_potatoes = 172)
  (h3 : remaining_total = 80)
  (h4 : initial_tomatoes + (initial_potatoes - picked_potatoes) = remaining_total) :
  initial_potatoes = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_potatoes_l1190_119013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_length_is_800cm_l1190_119078

/-- Calculates the length of a wall given its height, width, number of bricks, and brick dimensions. -/
noncomputable def wall_length (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℝ) 
                (brick_length brick_width brick_height : ℝ) : ℝ :=
  (num_bricks * brick_length * brick_width * brick_height) / (wall_height * wall_width)

/-- Theorem stating that the length of the wall is 800 cm given the specified conditions. -/
theorem wall_length_is_800cm : 
  wall_length 600 2 2909.090909090909 5 11 6 = 800 := by
  -- The proof goes here
  sorry

#eval (2909.090909090909 * 5 * 11 * 6) / (600 * 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_length_is_800cm_l1190_119078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1190_119040

theorem trigonometric_identity : 
  Real.sin (π / 18)⁻¹ - 3 * Real.cos (2 * π / 9) = 
  4 * Real.sqrt (2 - Real.sqrt 3) / 3 - 3 * Real.sqrt 3 + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1190_119040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cube_volume_l1190_119091

/-- The volume of an ice cube after melting for three hours -/
noncomputable def volume_after_melting (initial_volume : ℝ) : ℝ :=
  initial_volume * (1 - 3/4) * (1 - 3/4) * (1 - 2/3)

/-- Theorem stating the original volume of the ice cube -/
theorem ice_cube_volume : ∃ (v : ℝ), volume_after_melting v = 0.4 ∧ v = 19.2 := by
  -- We'll use 19.2 as our witness for v
  use 19.2
  -- Split the goal into two parts
  apply And.intro
  · -- Prove that volume_after_melting 19.2 = 0.4
    -- We'll rely on computation for this part
    norm_num [volume_after_melting]
  · -- Prove that 19.2 = 19.2 (trivial)
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cube_volume_l1190_119091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_AlF3_l1190_119011

/-- The molecular weight of 7 moles of AlF3 -/
theorem molecular_weight_AlF3 (atomic_weight_Al atomic_weight_F : ℝ) 
  (h1 : atomic_weight_Al = 26.98)
  (h2 : atomic_weight_F = 19.00) : ℝ :=
by
  -- Define the molecular weight of AlF3
  let molecular_weight_AlF3 := atomic_weight_Al + 3 * atomic_weight_F
  
  -- Calculate the weight of 7 moles of AlF3
  let weight_7_moles := 7 * molecular_weight_AlF3
  
  -- Return the result
  exact weight_7_moles

#check molecular_weight_AlF3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_AlF3_l1190_119011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_pi_l1190_119058

noncomputable section

-- Define the functions
def f1 (x : ℝ) : ℝ := Real.sin (x / 2)
def f2 (x : ℝ) : ℝ := Real.tan (2 * x)
def f3 (x : ℝ) : ℝ := |Real.sin (2 * x)|
def f4 (x : ℝ) : ℝ := 5 + Real.cos (2 * x)

end noncomputable section

-- Define the concept of a periodic function
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Theorem statement
theorem periodic_function_pi :
  (isPeriodic f4 π) ∧ 
  (¬ isPeriodic f1 π) ∧ 
  (¬ isPeriodic f2 π) ∧ 
  (¬ isPeriodic f3 π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_pi_l1190_119058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_l1190_119057

theorem stratified_sampling 
  (total_students : ℕ) 
  (girls : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 125) 
  (h2 : girls = 50) 
  (h3 : sample_size = 25) :
  let boys := total_students - girls
  let boys_to_select := (boys * sample_size) / total_students
  let prob_boy_selected := sample_size / total_students
  (boys_to_select = 15) ∧ (prob_boy_selected = 1/5) := by
  sorry

#check stratified_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_l1190_119057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inverse_point_l1190_119067

/-- The base of an exponential function satisfying certain conditions -/
noncomputable def a : ℝ := sorry

/-- The exponential function with base a -/
noncomputable def f (x : ℝ) : ℝ := a^x

/-- The inverse function of f -/
noncomputable def f_inverse : ℝ → ℝ := sorry

theorem exponential_inverse_point : 
  a > 0 ∧ a ≠ 1 ∧ f_inverse 9 = 2 → a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inverse_point_l1190_119067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_applicability_l1190_119075

def square_difference_formula (a b : ℝ) : ℝ := a^2 - b^2

def expression_A (x y : ℝ) : ℝ := (-4*x + 3*y) * (4*x + 3*y)
def expression_B (x y : ℝ) : ℝ := (4*x - 3*y) * (3*y - 4*x)
def expression_C (x y : ℝ) : ℝ := (-4*x + 3*y) * (-4*x - 3*y)
def expression_D (x y : ℝ) : ℝ := (4*x + 3*y) * (4*x - 3*y)

theorem square_difference_applicability :
  (∃ a b : ℝ, ∀ x y : ℝ, expression_A x y = square_difference_formula a b) ∧
  (∃ a b : ℝ, ∀ x y : ℝ, expression_C x y = square_difference_formula a b) ∧
  (∃ a b : ℝ, ∀ x y : ℝ, expression_D x y = square_difference_formula a b) ∧
  ¬(∃ a b : ℝ, ∀ x y : ℝ, expression_B x y = square_difference_formula a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_applicability_l1190_119075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_translation_symmetry_l1190_119086

/-- The phase shift of a cosine function that results in symmetry about the origin after translation -/
def phase_shift (φ : ℝ) : Prop :=
  ∃ k : ℤ, φ = (5 * Real.pi) / 6 + k * Real.pi

/-- The original cosine function -/
noncomputable def original_func (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

/-- The translated cosine function -/
noncomputable def translated_func (x φ : ℝ) : ℝ := original_func (x - Real.pi / 6) φ

/-- Symmetry about the origin for the translated function -/
def symmetric_about_origin (φ : ℝ) : Prop :=
  ∀ x : ℝ, translated_func x φ = -translated_func (-x) φ

theorem cosine_translation_symmetry (φ : ℝ) :
  symmetric_about_origin φ → phase_shift φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_translation_symmetry_l1190_119086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_cube_net_l1190_119016

/- The problem is about identifying the correct net of a cube -/

-- Define a type for the answer choices
inductive CubeNet
  | A
  | B
  | C
  | D

-- Theorem stating that C is the correct net of a cube
theorem correct_cube_net : CubeNet.C = CubeNet.C := by
  -- The proof is trivial as we're just asserting equality
  rfl

#check correct_cube_net

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_cube_net_l1190_119016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_roots_l1190_119073

/-- A monic polynomial of degree n with real coefficients -/
def MonicPolynomial (n : ℕ) := {p : Polynomial ℝ // p.leadingCoeff = 1 ∧ p.natDegree = n}

/-- The sum of squares of the roots of a polynomial -/
noncomputable def sumSquaresRoots (p : Polynomial ℝ) : ℝ :=
  (p.roots.map (λ r => r^2)).sum

/-- The theorem stating the minimum value of the sum of squares of roots -/
theorem min_sum_squares_roots {n : ℕ} (p : MonicPolynomial n) 
  (h : p.val.coeff (n-1) = p.val.coeff (n-2)) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (q : MonicPolynomial n), 
    q.val.coeff (n-1) = q.val.coeff (n-2) → m ≤ sumSquaresRoots q.val :=
sorry

#check min_sum_squares_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_roots_l1190_119073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1190_119079

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + 2 / (x - 1)

-- Define the domain
def domain : Set ℝ := { x | 2 ≤ x ∧ x < 4 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | 5/3 < y ∧ y ≤ 3 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1190_119079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_is_196_l1190_119099

def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3

def sequence_set : Set ℕ :=
  {n : ℕ | satisfies_conditions n}

theorem sum_of_max_min_is_196 :
  ∃ (a_min a_max : ℕ),
    a_min ∈ sequence_set ∧
    a_max ∈ sequence_set ∧
    (∀ n, n ∈ sequence_set → a_min ≤ n ∧ n ≤ a_max) ∧
    a_min + a_max = 196 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_is_196_l1190_119099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_96_l1190_119054

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def area_enclosed : ℝ := 96

/-- The equation defining the graph -/
def graph_equation (x y : ℝ) : Prop := abs x + abs (3 * y) = 12

/-- Theorem stating that the area enclosed by the graph is 96 square units -/
theorem area_enclosed_is_96 : area_enclosed = 96 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_96_l1190_119054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_opposite_sign_l1190_119095

theorem max_opposite_sign (a b c d e f : ℤ) : 
  ∃ (n : ℕ), n ≤ 5 ∧ 
  (a * b * c * d * e * f > 0 → n = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + 
                                   (if d < 0 then 1 else 0) + (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) ∧
  (a * b * c * d * e * f < 0 → n = (if a > 0 then 1 else 0) + (if b > 0 then 1 else 0) + (if c > 0 then 1 else 0) + 
                                   (if d > 0 then 1 else 0) + (if e > 0 then 1 else 0) + (if f > 0 then 1 else 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_opposite_sign_l1190_119095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_315_degrees_csc_315_degrees_result_l1190_119049

theorem csc_315_degrees : Real.sin (315 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

theorem csc_315_degrees_result : (Real.sin (315 * π / 180))⁻¹ = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_315_degrees_csc_315_degrees_result_l1190_119049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_l1190_119042

/-- The slope angle of a line given by the equation y + 1 = 0 is 0 degrees. -/
theorem slope_angle_horizontal_line : 
  ∃ (slope_angle : Set (ℝ × ℝ) → ℝ),
  let line := {(x, y) : ℝ × ℝ | y + 1 = 0}
  slope_angle line = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_l1190_119042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_equality_l1190_119074

/-- In a triangle ABC, the product of the sines of half the angles equals 1/8 
    if and only if all sides are equal -/
theorem triangle_sine_product_equality (A B C a b c : ℝ) : 
  -- Given: ABC is a triangle
  (A + B + C = π) →
  -- Side lengths relate to angles (law of sines)
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- The theorem
  (Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) = 1/8 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_equality_l1190_119074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1190_119017

/-- Represents a parabola with equation y² = 6x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → Prop

/-- Represents a line with a given slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The intersection points of a line and a parabola -/
def intersection (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {(x, y) | p.equation x y ∧ y = l.slope * x + l.intercept}

theorem parabola_chord_length 
  (p : Parabola) 
  (l : Line) 
  (h1 : p.equation = λ x y => y^2 = 6*x)
  (h2 : p.focus = (3/2, 0))
  (h3 : l.slope = 1)
  (h4 : l.intercept = -3/2) :
  let points := intersection p l
  ∃ A B : ℝ × ℝ, A ∈ points ∧ B ∈ points ∧ ‖A - B‖ = 12 := by
  sorry

#check parabola_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1190_119017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_OE_dot_OF_l1190_119082

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 6

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := x^2 = 8*y

-- Define the common tangent line l
def l (y : ℝ) : Prop := y = -2

-- Define the common intersection point
noncomputable def intersection_point : ℝ × ℝ := (Real.sqrt 2, -2)

-- Define the dot product of OE and OF
noncomputable def OE_dot_OF (t : ℝ) : ℝ := 128 / (16 + t^2) - 6

theorem range_of_OE_dot_OF :
  ∀ t : ℝ, -6 < OE_dot_OF t ∧ OE_dot_OF t ≤ 2 :=
by
  intro t
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_OE_dot_OF_l1190_119082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_l1190_119076

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + x)^2 - 2 * Real.log (1 + x)

-- Define the equation
def equation (x a : ℝ) : Prop := f x = x^2 + x + a

-- Theorem statement
theorem root_range (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc 0 2 ∧ y ∈ Set.Icc 0 2 ∧ x ≠ y ∧ equation x a ∧ equation y a) ∧
  (∀ z : ℝ, z ∈ Set.Icc 0 2 → equation z a → (z = x ∨ z = y)) →
  a ∈ Set.Ioo (2 - 2 * Real.log 2) (3 - 2 * Real.log 3 + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_l1190_119076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_domain_l1190_119096

theorem square_root_domain (b : ℝ) : ∃ (x : ℝ), x^2 = b - 3 ↔ b ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_domain_l1190_119096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_are_perfect_cubes_l1190_119070

noncomputable def my_sequence (n : ℕ) : ℚ :=
  (10^(3*n+2) + 7 * 10^(2*n+1) + 1) / 3

theorem my_sequence_are_perfect_cubes :
  ∀ n : ℕ, ∃ m : ℚ, my_sequence n = m^3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_are_perfect_cubes_l1190_119070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_prism_volume_l1190_119022

-- Define the cone parameters
variable (l : ℝ) -- slant height of the cone
variable (α : ℝ) -- angle between slant height and base plane

-- Define the volume function for the inscribed prism
noncomputable def prism_volume (l : ℝ) (α : ℝ) : ℝ :=
  (18 * Real.sqrt 7 / 49) * l^3 * (Real.cos α)^3 * (Real.tan (α/2))^3

-- Theorem statement
theorem inscribed_prism_volume (l : ℝ) (α : ℝ) (h_l : l > 0) (h_α : 0 < α ∧ α < π/2) :
  prism_volume l α = (18 * Real.sqrt 7 / 49) * l^3 * (Real.cos α)^3 * (Real.tan (α/2))^3 := by
  -- Unfold the definition of prism_volume
  unfold prism_volume
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_prism_volume_l1190_119022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1190_119059

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  passes_origin : a * 0^2 + b * 0 + c = 0
  symmetry_axis : -b / (2 * a) = -1/2
  equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = x ∧ 
                ∀ y : ℝ, a * y^2 + b * y + c = y → y = x

/-- The function g(x) derived from f(x) -/
def g (f : QuadraticFunction) (lambda : ℝ) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c - (1 + 2 * lambda) * x

/-- Main theorem statement -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (∀ x : ℝ, f.a * x^2 + f.b * x + f.c = x^2 + x) ∧
  (∀ lambda : ℝ, (∀ x : ℝ, g f lambda x ≥ -1) ↔ -1 ≤ lambda ∧ lambda ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1190_119059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_one_implies_b_half_l1190_119044

/-- Given a function g(x) = bx^3 + bx - 3 where b is a real number,
    if g(g(1)) = 1, then b = 1/2 -/
theorem g_composition_equals_one_implies_b_half (b : ℝ) : 
  (let g := λ x => b * x^3 + b * x - 3 
   g (g 1) = 1) → b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_one_implies_b_half_l1190_119044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_four_zeros_l1190_119015

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_def : ∀ x : ℝ, x ≥ 0 → f x = Real.log (x^2 - 3*x + 3) / Real.log 10

-- Define what it means for a function to have exactly n zeros
def has_exactly_n_zeros (g : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ (S : Finset ℝ), (∀ x ∈ S, g x = 0) ∧ 
                    (∀ x : ℝ, g x = 0 → x ∈ S) ∧ 
                    (Finset.card S = n)

-- The theorem to prove
theorem f_has_four_zeros : has_exactly_n_zeros f 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_four_zeros_l1190_119015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_craftee_and_best_purchase_l1190_119025

/-- Calculates the change and discounts for a purchase at Craftee And Best store -/
theorem craftee_and_best_purchase
  (wooden_toy_price : ℚ)
  (hat_price : ℚ)
  (tax_rate : ℚ)
  (wooden_toy_discount_rate : ℚ)
  (hat_discount_rate : ℚ)
  (wooden_toys_bought : ℕ)
  (hats_bought : ℕ)
  (paid_amount : ℚ)
  (h1 : wooden_toy_price = 20)
  (h2 : hat_price = 10)
  (h3 : tax_rate = 0.08)
  (h4 : wooden_toy_discount_rate = 0.15)
  (h5 : hat_discount_rate = 0.10)
  (h6 : wooden_toys_bought = 4)
  (h7 : hats_bought = 5)
  (h8 : paid_amount = 250) :
  ∃ (change wooden_toys_discount hats_discount : ℚ),
    change = 127.96 ∧
    wooden_toys_discount = 12 ∧
    hats_discount = 5 ∧
    let wooden_toys_cost := wooden_toy_price * wooden_toys_bought
    let hats_cost := hat_price * hats_bought
    let total_cost_before_discounts := wooden_toys_cost + hats_cost
    let total_discounts := wooden_toys_discount + hats_discount
    let total_cost_after_discounts := total_cost_before_discounts - total_discounts
    let tax_amount := tax_rate * total_cost_after_discounts
    let total_cost_after_discounts_and_taxes := total_cost_after_discounts + tax_amount
    change = paid_amount - total_cost_after_discounts_and_taxes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_craftee_and_best_purchase_l1190_119025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_first_exponent_l1190_119029

theorem base_of_first_exponent (x a b : ℕ) (hx : x > 0) (ha : a > 0) (hb : b > 0) :
  (x^a : ℕ) * (9^(3*a - 1) : ℕ) = 2^7 * (3^b : ℕ) →
  a = 7 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_first_exponent_l1190_119029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sin_value_l1190_119077

theorem arithmetic_sequence_sin_value (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (sum_condition : a 2 + a 6 = (3/2) * Real.pi) : 
  Real.sin (2 * a 4 - Real.pi / 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sin_value_l1190_119077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_approx_l1190_119009

/-- The volume of a wedge that represents one-third of a cylinder -/
noncomputable def wedge_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem wedge_volume_approx :
  let r : ℝ := 7
  let h : ℝ := 9
  let π_approx : ℝ := 3.14
  abs (wedge_volume r h - 461.58) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_approx_l1190_119009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l1190_119085

/-- The length of the arc of the curve given by ρ = 6(1 + sin φ) in polar coordinates,
    where -π/2 ≤ φ ≤ 0, is equal to 12(2 - √2). -/
theorem arc_length_polar_curve :
  ∫ φ in (-π/2)..0, Real.sqrt ((6 * (1 + Real.sin φ))^2 + (6 * Real.cos φ)^2) = 12 * (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l1190_119085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l1190_119083

noncomputable def square_center : ℝ × ℝ := (4, 5)
def square_area : ℝ := 16
def dilation_center : ℝ × ℝ := (0, 0)
def dilation_factor : ℝ := 3

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1)^2 + (p.2)^2)

def dilate_point (p : ℝ × ℝ) (center : ℝ × ℝ) (factor : ℝ) : ℝ × ℝ :=
  (center.1 + factor * (p.1 - center.1), center.2 + factor * (p.2 - center.2))

theorem farthest_vertex_after_dilation :
  ∃ (vertices : List (ℝ × ℝ)),
    (∀ v ∈ vertices, distance_from_origin (dilate_point v dilation_center dilation_factor) ≤ 
      distance_from_origin (dilate_point (18, 21) dilation_center dilation_factor)) ∧
    (List.length vertices = 4) ∧
    (∀ v ∈ vertices, distance_from_origin v = Real.sqrt (square_area / 2)) ∧
    (List.sum (List.map Prod.fst vertices) / 4 = square_center.1) ∧
    (List.sum (List.map Prod.snd vertices) / 4 = square_center.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l1190_119083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1190_119064

/-- Parabola C defined by parametric equations -/
noncomputable def C (t : ℝ) : ℝ × ℝ := (2 * t^2, 2 * t)

/-- Point M on parabola C -/
noncomputable def M : ℝ → ℝ × ℝ := C

/-- Point P as midpoint of OM -/
noncomputable def P (t : ℝ) : ℝ × ℝ := ((M t).1 / 2, (M t).2 / 2)

/-- Theorem: Trajectory of P is y^2 = x -/
theorem trajectory_of_P :
  ∀ t : ℝ, (P t).2^2 = (P t).1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1190_119064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_one_neg_two_l1190_119052

/-- If the terminal side of angle α passes through point P(1, -2) in a rectangular coordinate system, then sin α = -2/√5 -/
theorem sin_alpha_for_point_one_neg_two (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -2) →
  Real.sin α = -2 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_one_neg_two_l1190_119052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l1190_119038

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

/-- Theorem: The area of triangle ABC with given vertices is 1.5 -/
theorem triangle_abc_area :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-2, 5)
  let C : ℝ × ℝ := (4, -2)
  triangle_area A B C = 1.5 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l1190_119038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_23_permutations_divisible_by_37_l1190_119053

/-- A six-digit number type -/
def SixDigitNumber := Nat

/-- Predicate to check if a number has six different non-zero digits -/
def has_six_different_nonzero_digits (n : SixDigitNumber) : Prop :=
  ∃ (a b c d e f : Nat), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    n = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f

/-- Predicate to check if a number is divisible by 37 -/
def is_divisible_by_37 (n : SixDigitNumber) : Prop :=
  ∃ k : Nat, n = 37 * k

/-- Theorem statement -/
theorem at_least_23_permutations_divisible_by_37 
  (N : SixDigitNumber) 
  (h1 : has_six_different_nonzero_digits N) 
  (h2 : is_divisible_by_37 N) : 
  ∃ (permutations : Finset SixDigitNumber), 
    (∀ p ∈ permutations, is_divisible_by_37 p) ∧ 
    (∀ p ∈ permutations, has_six_different_nonzero_digits p) ∧
    (∀ p ∈ permutations, p ≠ N) ∧
    permutations.card ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_23_permutations_divisible_by_37_l1190_119053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celebration_day_l1190_119002

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day of the week that is n days after a given day -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

theorem celebration_day (birthday : DayOfWeek) (h : birthday = DayOfWeek.Friday) :
  dayAfter birthday 1499 = DayOfWeek.Sunday := by
  sorry

#eval dayAfter DayOfWeek.Friday 1499

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celebration_day_l1190_119002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_relation_l1190_119056

/-- Given that the equation 4^x - 4^(-x) = 2 cos(ax) has 2007 roots,
    prove that the equation 4^x + 4^(-x) = 2 cos(ax) + 4 has 4014 roots. -/
theorem root_count_relation (a : ℝ) :
  (∃! (s : Set ℝ), s.Finite ∧ Set.ncard s = 2007 ∧ ∀ x ∈ s, 4^x - 4^(-x) = 2 * Real.cos (a * x)) →
  ∃! (t : Set ℝ), t.Finite ∧ Set.ncard t = 4014 ∧ ∀ x ∈ t, 4^x + 4^(-x) = 2 * Real.cos (a * x) + 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_relation_l1190_119056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_10_25_l1190_119001

/-- Represents the battery life of a portable device -/
structure BatteryLife where
  standbyTime : ℝ  -- Total battery life in standby mode (hours)
  activeTime : ℝ   -- Total battery life in active use (hours)
  totalUsedTime : ℝ -- Total time device has been on since full charge (hours)
  activeUsedTime : ℝ -- Time device was actively used (hours)

/-- Calculates the remaining battery life in standby mode -/
noncomputable def remainingStandbyTime (b : BatteryLife) : ℝ :=
  let standbyRate := 1 / b.standbyTime
  let activeRate := 1 / b.activeTime
  let standbyUsedTime := b.totalUsedTime - b.activeUsedTime
  let batteryUsed := standbyRate * standbyUsedTime + activeRate * b.activeUsedTime
  let batteryRemaining := 1 - batteryUsed
  batteryRemaining / standbyRate

/-- Theorem: Given the specified conditions, the remaining battery life in standby mode is 10.25 hours -/
theorem remaining_battery_life_is_10_25 (b : BatteryLife) 
    (h1 : b.standbyTime = 30)
    (h2 : b.activeTime = 4)
    (h3 : b.totalUsedTime = 10)
    (h4 : b.activeUsedTime = 1.5) :
  remainingStandbyTime b = 10.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_10_25_l1190_119001
