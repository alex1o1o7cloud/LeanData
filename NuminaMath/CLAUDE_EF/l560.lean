import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_non_monotonic_f_l560_56091

open Real

-- Define the derivative of f(x)
noncomputable def f_derivative (x k : ℝ) : ℝ := exp x + k^2 / exp x - 1 / k

-- Define the property of f(x) not being monotonic
def not_monotonic (f : ℝ → ℝ) : Prop := ∃ x y z : ℝ, x < y ∧ y < z ∧ f x < f y ∧ f y > f z

-- Theorem statement
theorem k_range_for_non_monotonic_f (f : ℝ → ℝ) (k : ℝ) 
  (h_derivative : ∀ x, deriv f x = f_derivative x k)
  (h_not_monotonic : not_monotonic f) :
  0 < k ∧ k < sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_non_monotonic_f_l560_56091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_height_of_buildings_l560_56024

/-- The height of the tallest building in feet -/
noncomputable def tallest : ℝ := 100

/-- The height of the second tallest building -/
noncomputable def second : ℝ := tallest / 2

/-- The height of the third tallest building -/
noncomputable def third : ℝ := second / 3

/-- The height of the fourth tallest building -/
noncomputable def fourth : ℝ := third / 4

/-- The height of the fifth tallest building -/
noncomputable def fifth : ℝ := (2 / 5) * fourth

/-- The height of the sixth tallest building -/
noncomputable def sixth : ℝ := (3 / 4) * fifth

/-- The theorem stating the total height of all buildings -/
theorem total_height_of_buildings :
  tallest + second + third + fourth + fifth + sixth = 173.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_height_of_buildings_l560_56024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_18m4_l560_56012

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

noncomputable def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_18m4 (m : ℕ) (h_odd : is_odd m) (h_div : num_divisors m = 13) :
  num_divisors (18 * m^4) = 294 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_18m4_l560_56012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l560_56034

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo (Real.pi/2) Real.pi) 
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α + Real.pi/4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l560_56034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_equals_one_l560_56016

theorem subset_implies_m_equals_one (m : ℝ) : 
  (({3, m^2} : Set ℝ) ⊆ ({-1, 3, 2*m-1} : Set ℝ)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_equals_one_l560_56016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_reading_percentage_l560_56044

/-- Proves that the percentage of remaining pages read in the second week is 30% -/
theorem second_week_reading_percentage 
  (total_pages : ℕ)
  (first_week_pages : ℕ)
  (third_week_pages : ℕ)
  (h1 : total_pages = 600)
  (h2 : first_week_pages = total_pages / 2)
  (h3 : third_week_pages = 210)
  (h4 : first_week_pages + (total_pages - first_week_pages - third_week_pages) + third_week_pages = total_pages) :
  (total_pages - first_week_pages - third_week_pages : ℚ) / (total_pages - first_week_pages) = 3/10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_reading_percentage_l560_56044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_mu_squared_l560_56068

/-- The hyperbola C defined by x²/a² - y²/b² = 1 -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1}

/-- The right branch of the hyperbola -/
def RightBranch (a b : ℝ) : Set (ℝ × ℝ) :=
  {p ∈ Hyperbola a b | p.1 > 0}

theorem min_lambda_mu_squared (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∀ P ∈ RightBranch a b,
  ∀ lambda mu : ℝ,
  P.1 = (lambda + mu) * a ∧ P.2 = (lambda - mu) * b →
  (∀ lambda' mu' : ℝ, lambda' ^ 2 + mu' ^ 2 ≥ lambda ^ 2 + mu ^ 2) →
  lambda ^ 2 + mu ^ 2 = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_mu_squared_l560_56068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recoloring_periodic_after_1280_steps_l560_56011

/-- Represents the color of a point -/
inductive Color
| Red
| Green
deriving Repr, DecidableEq

/-- Represents the coloring of all points on the circle -/
def Coloring := Vector Color 2561

/-- Recolors a single point based on its neighbors -/
def recolorPoint (left right center : Color) : Color :=
  if left = center ∧ right = center then center else
    match center with
    | Color.Red => Color.Green
    | Color.Green => Color.Red

/-- Performs one step of recoloring for all points -/
def recolorStep (col : Coloring) : Coloring :=
  Vector.ofFn (fun i =>
    recolorPoint (col.get ((i - 1 + 2561) % 2561))
                 (col.get ((i + 1) % 2561))
                 (col.get i))

/-- Performs n steps of recoloring -/
def recolorNSteps (n : Nat) (col : Coloring) : Coloring :=
  match n with
  | 0 => col
  | n + 1 => recolorNSteps n (recolorStep col)

theorem recoloring_periodic_after_1280_steps :
  ∀ (initialColoring : Coloring),
    recolorNSteps 1280 initialColoring = recolorNSteps 1282 initialColoring := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recoloring_periodic_after_1280_steps_l560_56011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l560_56040

-- Define the circle equation
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the cosine function
def cos_func (x y : ℝ) : Prop := y = Real.cos x

-- Define an intersection point
def intersection_point (p : ℝ × ℝ) : Prop := my_circle p.1 p.2 ∧ cos_func p.1 p.2

-- Theorem statement
theorem max_intersections :
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (S : Set (ℝ × ℝ)), (∀ p ∈ S, intersection_point p) → 
  ∃ (f : S → Fin n), Function.Injective f) ∧
  (∃ (S : Set (ℝ × ℝ)), (∀ p ∈ S, intersection_point p) ∧ 
  ∃ (f : Fin n → S), Function.Bijective f) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l560_56040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l560_56089

-- Define the concept of a "new stationary point"
def new_stationary_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = (deriv f) x

-- Define the functions
def g (x : ℝ) : ℝ := 2 * x

noncomputable def h (x : ℝ) : ℝ := Real.log x

def φ (x : ℝ) : ℝ := x^3

-- State the theorem
theorem new_stationary_points_order 
  (a b c : ℝ) 
  (ha : new_stationary_point g a)
  (hb : new_stationary_point h b)
  (hc : new_stationary_point φ c)
  (hb_pos : b > 0) -- Ensuring b is positive for ln x to be defined
  (hc_nonzero : c ≠ 0) -- Given condition for φ(x)
  : c > b ∧ b > a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l560_56089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravelling_rate_is_half_l560_56081

/-- Represents the dimensions and cost of a rectangular plot with a gravel path --/
structure PlotWithPath where
  length : ℝ
  width : ℝ
  pathWidth : ℝ
  totalCost : ℝ

/-- Calculates the area of the gravel path --/
noncomputable def pathArea (p : PlotWithPath) : ℝ :=
  p.length * p.width - (p.length - 2 * p.pathWidth) * (p.width - 2 * p.pathWidth)

/-- Calculates the rate per square meter for gravelling --/
noncomputable def gravellingRate (p : PlotWithPath) : ℝ :=
  p.totalCost / pathArea p

/-- Theorem stating that for the given plot dimensions and cost, the gravelling rate is 0.5 --/
theorem gravelling_rate_is_half :
  let p : PlotWithPath := {
    length := 110,
    width := 65,
    pathWidth := 2.5,
    totalCost := 425
  }
  gravellingRate p = 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravelling_rate_is_half_l560_56081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_1_complex_calculation_2_l560_56076

-- Complex numbers are already defined in Mathlib, so we don't need to redefine them
-- We can use the existing Complex type

-- Theorem statements
theorem complex_calculation_1 :
  (1 + 2*Complex.I) + (3 - 4*Complex.I) - (5 + 6*Complex.I) = -1 - 8*Complex.I := by
  -- Proof steps would go here
  sorry

theorem complex_calculation_2 :
  (1 - Complex.I)*(-1 + Complex.I) + (-1 + Complex.I) = -1 + 3*Complex.I := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_1_complex_calculation_2_l560_56076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arch_width_for_given_height_optimal_arch_dimensions_tunnel_design_theorem_l560_56000

/-- Represents the dimensions and constraints of a tunnel with a half-elliptical cross-section -/
structure TunnelDimensions where
  laneWidth : ℝ
  vehicleHeightLimit : ℝ
  tunnelLength : ℝ
  minArchHeight : ℝ

/-- Calculates the arch width given the arch height for a tunnel with specified dimensions -/
noncomputable def archWidth (td : TunnelDimensions) (archHeight : ℝ) : ℝ :=
  2 * (td.laneWidth / 2) * Real.sqrt (archHeight / (archHeight - td.vehicleHeightLimit))

/-- Calculates the volume of earthwork for the tunnel -/
noncomputable def tunnelVolume (td : TunnelDimensions) (archHeight : ℝ) : ℝ :=
  (Real.pi / 4) * archWidth td archHeight * archHeight * td.tunnelLength

/-- Theorem stating the arch width for a given arch height -/
theorem arch_width_for_given_height (td : TunnelDimensions) 
  (h : td.minArchHeight = 6) :
  abs (archWidth td td.minArchHeight - 33.3) < 0.1 := by sorry

/-- Theorem for the optimal arch dimensions to minimize earthwork volume -/
theorem optimal_arch_dimensions (td : TunnelDimensions) 
  (h : td.minArchHeight = 6) :
  ∃ (optimalHeight optimalWidth : ℝ),
    optimalHeight ≥ td.minArchHeight ∧
    abs (optimalHeight - 6.4) < 0.1 ∧
    abs (optimalWidth - 31.1) < 0.1 ∧
    ∀ (height : ℝ), height ≥ td.minArchHeight →
      tunnelVolume td optimalHeight ≤ tunnelVolume td height := by sorry

/-- The main theorem combining both parts of the problem -/
theorem tunnel_design_theorem (td : TunnelDimensions) 
  (h1 : td.laneWidth = 22) 
  (h2 : td.vehicleHeightLimit = 4.5) 
  (h3 : td.tunnelLength = 25000) 
  (h4 : td.minArchHeight = 6) :
  (abs (archWidth td td.minArchHeight - 33.3) < 0.1) ∧
  (∃ (optimalHeight optimalWidth : ℝ),
    optimalHeight ≥ td.minArchHeight ∧
    abs (optimalHeight - 6.4) < 0.1 ∧
    abs (optimalWidth - 31.1) < 0.1 ∧
    ∀ (height : ℝ), height ≥ td.minArchHeight →
      tunnelVolume td optimalHeight ≤ tunnelVolume td height) := by
  constructor
  · exact arch_width_for_given_height td h4
  · exact optimal_arch_dimensions td h4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arch_width_for_given_height_optimal_arch_dimensions_tunnel_design_theorem_l560_56000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l560_56019

/-- Definition of the ellipse -/
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the focus -/
def right_focus (x y : ℝ) : Prop := x = 3 ∧ y = 0

/-- Definition of eccentricity -/
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 3 / 2

/-- Definition of line intersecting ellipse -/
def line_intersects_ellipse (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

/-- Definition of midpoint -/
def is_midpoint (m : ℝ × ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  m = ((x1 + x2) / 2, (y1 + y2) / 2)

/-- Definition of circle passing through origin -/
noncomputable def circle_passes_origin (m n : ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2)
  (m.1 / 2)^2 + (m.2 / 2)^2 = (d / 2)^2

theorem ellipse_theorem (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (k : ℝ) :
  (∀ x y, is_ellipse a b x y) →
  (∃ x y, right_focus x y) →
  eccentricity e →
  (∃ x1 y1 x2 y2, line_intersects_ellipse k x1 y1 ∧ line_intersects_ellipse k x2 y2) →
  (∃ m n, is_midpoint m x1 y1 3 0 ∧ is_midpoint n x2 y2 3 0 ∧ circle_passes_origin m n) →
  (a^2 = 12 ∧ b^2 = 3 ∧ (k = Real.sqrt 2 / 4 ∨ k = -Real.sqrt 2 / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l560_56019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l560_56092

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log ((x^2 + a*x + b) / (x^2 + x + 1)) / Real.log 3

theorem function_properties (a b : ℝ) :
  (f a b 0 = 0 ∧
   (∀ x y : ℝ, 1 ≤ x → x < y → f a b x < f a b y) ∧
   (∀ x : ℝ, f a b x ≤ 1) ∧
   (∃ x : ℝ, f a b x = 1)) ↔
  (a = -1 ∧ b = 1) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l560_56092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l560_56062

noncomputable def f (a b c x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem function_properties
  (a b c : ℝ)
  (h1 : f a b c 0 = 1)
  (h2 : f a b c (Real.pi / 2) = 1)
  (h3 : b > 0)
  (h4 : ∀ x ∈ Set.Ioo 0 (Real.pi / 2), |f a b c x| ≤ 2) :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + 5 * Real.pi / 4),
    ∀ y ∈ Set.Icc (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + 5 * Real.pi / 4),
    x ≤ y → f a b c x ≥ f a b c y) ∧
  a ∈ Set.Icc (-Real.sqrt 2) (4 + 3 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l560_56062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l560_56006

-- Define the line x - y = 0
def line (x y : ℝ) : Prop := x = y

-- Define the circle x^2 + (y - 6)^2 = 2
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Statement to prove
theorem min_distance_line_circle :
  ∃ (x1 y1 x2 y2 : ℝ),
    line x1 y1 ∧ circle_eq x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      line x3 y3 → circle_eq x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l560_56006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_signal_post_l560_56082

/-- The time (in seconds) it takes for a train to pass a signal post -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem: A 400-meter long train traveling at 60 km/hour takes approximately 24 seconds to pass a signal post -/
theorem train_passing_signal_post :
  ∃ ε > 0, |train_passing_time 400 60 - 24| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_signal_post_l560_56082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_chord_count_l560_56030

/-- Predicate to represent that a chord is tangent to the inner circle -/
def chord_tangent_to_inner_circle : ℕ → Prop := sorry

/-- Theorem about the number of chords in concentric circles -/
theorem concentric_circles_chord_count (angle_ABC : ℝ) (n : ℕ) : 
  angle_ABC = 75 →
  (∀ i : ℕ, i < n → chord_tangent_to_inner_circle i) →
  (105 * n) % 360 = 0 →
  n = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_chord_count_l560_56030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l560_56027

-- Define the original garden dimensions
noncomputable def original_length : ℝ := 60
noncomputable def original_width : ℝ := 20

-- Define the perimeter (same for both shapes)
noncomputable def perimeter : ℝ := 2 * (original_length + original_width)

-- Define the side length of the new square garden
noncomputable def new_side : ℝ := perimeter / 4

-- Define the areas of the original and new gardens
noncomputable def original_area : ℝ := original_length * original_width
noncomputable def new_area : ℝ := new_side ^ 2

-- Theorem statement
theorem garden_area_increase :
  new_area - original_area = 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l560_56027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l560_56039

theorem trig_identity (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π/2)) :
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = -2 / Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l560_56039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_seventeen_halves_l560_56002

/-- The area of a quadrilateral given by four points in 2D space -/
noncomputable def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AD := (D.1 - A.1, D.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let BD := (D.1 - B.1, D.2 - B.2)
  let areaABD := (1/2) * abs (AB.1 * AD.2 - AB.2 * AD.1)
  let areaBCD := (1/2) * abs (BC.1 * BD.2 - BC.2 * BD.1)
  areaABD + areaBCD

theorem quadrilateral_area_is_seventeen_halves :
  quadrilateralArea (0, 0) (2, 3) (5, 0) (3, -2) = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_seventeen_halves_l560_56002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curved_shape_max_area_l560_56043

/-- The area of the curved shape as a function of x -/
noncomputable def curvedShapeArea (x : ℝ) : ℝ := x / Real.pi - ((4 + Real.pi) * x^2) / (2 * Real.pi^2)

/-- The maximum area of the curved shape -/
noncomputable def maxCurvedShapeArea : ℝ := 1 / (2 * (Real.pi + 4))

theorem curved_shape_max_area :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 1 → curvedShapeArea y ≤ curvedShapeArea x) ∧
  curvedShapeArea x = maxCurvedShapeArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curved_shape_max_area_l560_56043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radicals_l560_56066

-- Define the expression
noncomputable def expression : ℝ := Real.sqrt 72 + Real.sqrt 32

-- Define the simplified form
noncomputable def simplified_form : ℝ := 10 * Real.sqrt 2

-- Theorem statement
theorem simplify_radicals : expression = simplified_form := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radicals_l560_56066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_scores_l560_56058

-- Define the variables
variable (n : ℕ)  -- number of judges
variable (S : ℝ)  -- total score
variable (x : ℝ)  -- highest score
variable (y : ℝ)  -- lowest score

-- Define the conditions
axiom avg_all : S / n = 9.64
axiom avg_without_highest : (S - x) / (n - 1) = 9.60
axiom avg_without_lowest : (S - y) / (n - 1) = 9.68
axiom max_score : ∀ score : ℝ, score ≤ 10
axiom min_score : ∀ score : ℝ, score ≥ 0

-- State the theorem
theorem competition_scores :
  y = 2 ∧ n = 49 := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_scores_l560_56058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_differences_equals_51204_l560_56014

def S : Finset ℕ := Finset.range 12

def positivesDifferences (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  (S.product S).filter (fun p => p.1 > p.2)

theorem sum_of_differences_equals_51204 :
  (positivesDifferences S).sum (fun p => 2^p.1 - 2^p.2) = 51204 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_differences_equals_51204_l560_56014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_3π_div_4_l560_56035

/-- Parametric curve in R² --/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric curve --/
noncomputable def givenCurve : ParametricCurve where
  x := λ t => 4 * Real.sqrt 2 * (Real.cos t) ^ 3
  y := λ t => Real.sqrt 2 * (Real.sin t) ^ 3

/-- The line x = 2 --/
def boundaryLine (x : ℝ) : Prop := x = 2

/-- The region of interest --/
def region (p : ℝ × ℝ) : Prop :=
  ∃ t, p.1 = givenCurve.x t ∧ p.2 = givenCurve.y t ∧ p.1 ≥ 2

/-- The area of the region --/
noncomputable def areaOfRegion : ℝ := sorry

theorem area_equals_3π_div_4 : areaOfRegion = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_3π_div_4_l560_56035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2CPD_value_l560_56028

/-- Helper function to calculate the angle between three points -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given five equally spaced points on a line and a point P satisfying certain angle conditions,
    prove that sin(2∠CPD) has a specific value. -/
theorem sin_2CPD_value (A B C D E P : ℝ × ℝ) : 
  (∃ k : ℝ, k > 0 ∧ B.1 - A.1 = k ∧ C.1 - B.1 = k ∧ D.1 - C.1 = k ∧ E.1 - D.1 = k) →
  (A.2 = B.2 ∧ B.2 = C.2 ∧ C.2 = D.2 ∧ D.2 = E.2) →
  Real.cos (angle A P D) = 3/5 →
  Real.cos (angle B P E) = 2/5 →
  Real.sin (2 * angle C P D) = (64 + 24 * Real.sqrt 21) / 125 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2CPD_value_l560_56028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metro_journey_theorem_l560_56088

/-- Represents a station in the metro system -/
structure Station where
  color : ℕ

/-- Represents a segment between two stations -/
structure Segment where
  start : Station
  finish : Station

/-- Represents the metro system -/
structure MetroSystem where
  stations : List Station
  segments : List Segment
  num_lines : ℕ

/-- Represents a train's journey -/
structure Journey where
  start : Station
  finish : Station
  duration : ℕ

/-- Helper function to determine if a train is at a given segment at a given time -/
def train_at_segment (metro : MetroSystem) (j : Journey) (s : Segment) (t : ℕ) : Prop :=
  sorry

/-- Main theorem: If a train travels for an even number of minutes in a valid metro system,
    it must pass through all segments to reach a station of a different color -/
theorem metro_journey_theorem (metro : MetroSystem) (j : Journey) :
  metro.num_lines = 3 →
  j.duration % 2 = 0 →
  j.start.color ≠ j.finish.color →
  ∀ s : Segment, s ∈ metro.segments →
    ∃ t : ℕ, t < j.duration ∧ train_at_segment metro j s t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metro_journey_theorem_l560_56088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_laborers_l560_56017

/-- Proves that 3 laborers were absent given the conditions of the problem -/
theorem absent_laborers (total_work : ℕ) (original_laborers : ℕ) (original_days : ℕ) (actual_days : ℕ) (absent_laborers : ℕ)
  (h1 : total_work = original_laborers * original_days)
  (h2 : original_laborers = 7)
  (h3 : original_days = 8)
  (h4 : actual_days = 14)
  (h5 : total_work = (original_laborers - absent_laborers) * actual_days) :
  absent_laborers = 3 := by
  sorry

#check absent_laborers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_laborers_l560_56017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_two_lines_l560_56026

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point1 : Point3D
  point2 : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Function to calculate the distance between a line and a point -/
noncomputable def distanceLineToPoint (l : Line3D) (p : Point3D) : ℝ := sorry

/-- Function to check if a line is perpendicular to a plane -/
def isPerpendicularToPlane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Function to check if a point is the midpoint of two other points -/
def isMidpoint (m : Point3D) (p1 : Point3D) (p2 : Point3D) : Prop := sorry

/-- Function to check if a point divides a line segment in a given ratio -/
def divideLineInRatio (k : Point3D) (c : Point3D) (d : Point3D) (ratio : ℚ) : Prop := sorry

/-- Function to calculate the distance between two lines -/
noncomputable def distanceLinesToLine (l1 l2 : Line3D) : ℝ := sorry

/-- Theorem statement -/
theorem line_equidistant_from_two_lines 
  (t : Tetrahedron) 
  (planeABC : Plane3D) 
  (M N K : Point3D) :
  isPerpendicularToPlane (Line3D.mk t.C t.D) planeABC →
  isMidpoint M t.D t.B →
  isMidpoint N t.A t.B →
  divideLineInRatio K t.C t.D (1/3) →
  distanceLinesToLine (Line3D.mk t.C N) (Line3D.mk t.A M) = 
  distanceLinesToLine (Line3D.mk t.C N) (Line3D.mk t.B K) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_two_lines_l560_56026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_solid_l560_56078

/-- A rectangular solid with given face areas -/
structure RectangularSolid where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area1_pos : area1 > 0
  area2_pos : area2 > 0
  area3_pos : area3 > 0

/-- The volume of a rectangular solid -/
noncomputable def volume (s : RectangularSolid) : ℝ :=
  Real.sqrt (s.area1 * s.area2 * s.area3)

/-- Theorem: The volume of a rectangular solid with face areas √2, √3, and √6 is √6 -/
theorem volume_of_specific_solid :
  ∃ (s : RectangularSolid),
    s.area1 = Real.sqrt 2 ∧
    s.area2 = Real.sqrt 3 ∧
    s.area3 = Real.sqrt 6 ∧
    volume s = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_solid_l560_56078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l560_56056

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (4 + x^2)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f a b x = -f a b (-x)) →  -- f is odd on (-2,2)
  f a b 1 = -2/5 →                                     -- f(1) = -2/5
  (a = -2 ∧ b = 0) ∧                                   -- Part 1
  (∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → f a b x > f a b y) ∧  -- Part 2
  (∀ m : ℝ, f a b (m-2) + f a b (m^2-m) > 0 ↔ 0 < m ∧ m < Real.sqrt 2) :=  -- Part 3
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l560_56056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l560_56051

noncomputable def system_equations (x y a : ℝ) : Prop :=
  x^2 + y^2 = 10 * (x * Real.cos a + y * Real.sin a) ∧
  x^2 + y^2 = 10 * (x * Real.sin (3 * a) + y * Real.cos (3 * a))

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem system_solution (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    system_equations x₁ y₁ a ∧
    system_equations x₂ y₂ a ∧
    distance x₁ y₁ x₂ y₂ = 8) ↔
  (∃ k : ℤ, a = π / 8 + k * π / 2) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l560_56051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_sum_l560_56032

theorem chord_cosine_sum (r : ℝ) (γ δ : ℝ) : 
  0 < r → 
  0 < γ → 
  0 < δ → 
  γ + δ < π → 
  5^2 = 2 * r^2 * (1 - Real.cos γ) → 
  7^2 = 2 * r^2 * (1 - Real.cos δ) → 
  12^2 = 2 * r^2 * (1 - Real.cos (γ + δ)) → 
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧ Real.cos γ = p / q ∧ Nat.Coprime p q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_sum_l560_56032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l560_56005

-- Define the angle in radians (48 degrees)
noncomputable def angle : ℝ := 48 * Real.pi / 180

-- State the theorem
theorem largest_expression :
  0 < Real.cos angle ∧ 
  Real.cos angle < Real.sin angle ∧ 
  Real.sin angle < 1 ∧
  Real.tan angle = Real.sin angle / Real.cos angle →
  (Real.tan angle + 1 / Real.tan angle > Real.sin angle + Real.cos angle) ∧
  (Real.tan angle + 1 / Real.tan angle > Real.tan angle + Real.cos angle) ∧
  (Real.tan angle + 1 / Real.tan angle > 1 / Real.tan angle + Real.sin angle) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l560_56005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_sum_gt_one_l560_56052

-- Define the function g
noncomputable def g (x m : ℝ) : ℝ := Real.log x + 1 / (2 * x) - m

-- State the theorem
theorem zero_points_sum_gt_one {m : ℝ} {x₁ x₂ : ℝ} 
  (h1 : g x₁ m = 0) 
  (h2 : g x₂ m = 0) 
  (h3 : x₁ < x₂) 
  (h4 : x₁ > 0) 
  (h5 : x₂ > 0) : 
  x₁ + x₂ > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_sum_gt_one_l560_56052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_truncated_pyramid_volume_l560_56015

/-- A regular hexagonal truncated pyramid inscribed in a sphere -/
structure HexagonalTruncatedPyramid (R : ℝ) where
  /-- The radius of the sphere -/
  radius : R > 0
  /-- The plane of the lower base passes through the center of the sphere -/
  lower_base_through_center : True
  /-- The lateral edge forms an angle of 60° with the plane of the base -/
  lateral_edge_angle : True

/-- The volume of a regular hexagonal truncated pyramid inscribed in a sphere -/
noncomputable def volume (R : ℝ) (pyramid : HexagonalTruncatedPyramid R) : ℝ :=
  (21 * R^3) / 16

/-- Theorem: The volume of the inscribed regular hexagonal truncated pyramid -/
theorem hexagonal_truncated_pyramid_volume (R : ℝ) (pyramid : HexagonalTruncatedPyramid R) :
  volume R pyramid = (21 * R^3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_truncated_pyramid_volume_l560_56015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l560_56098

noncomputable def options : List ℝ := [3000, 3100, 3200, 3300, 3400]

noncomputable def target : ℝ := 529 / 0.17

theorem closest_to_target :
  (options.argmin (λ x => |x - target|)) = some 3100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l560_56098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_line_segment_l560_56079

noncomputable def complex_midpoint (z₁ z₂ : ℂ) : ℂ := (z₁ + z₂) / 2

theorem midpoint_of_specific_line_segment :
  let z₁ : ℂ := -11 + 3*Complex.I
  let z₂ : ℂ := 3 - 7*Complex.I
  complex_midpoint z₁ z₂ = -4 - 2*Complex.I :=
by
  -- Unfold the definition of complex_midpoint
  unfold complex_midpoint
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_line_segment_l560_56079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l560_56061

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 4 * (Real.sin x)^3 * Real.cos x

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = π / 2 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (x : ℝ), f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l560_56061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_domino_tiling_iff_odd_l560_56083

/-- Represents a square grid with one corner removed -/
structure CornerRemovedGrid (n : ℕ) where
  size : n > 1

/-- Represents a domino placement on the grid -/
inductive DominoPlacement (n : ℕ)
  | horizontal : Fin n → Fin (n - 1) → DominoPlacement n
  | vertical : Fin (n - 1) → Fin n → DominoPlacement n

/-- A valid tiling of the grid with dominoes -/
def ValidTiling (n : ℕ) := List (DominoPlacement n)

/-- Counts the number of horizontal dominoes in a tiling -/
def countHorizontal (n : ℕ) (tiling : ValidTiling n) : ℕ := sorry

/-- Counts the number of vertical dominoes in a tiling -/
def countVertical (n : ℕ) (tiling : ValidTiling n) : ℕ := sorry

/-- The main theorem to be proved -/
theorem equal_domino_tiling_iff_odd (n : ℕ) (h : n > 1) :
  (∃ (tiling : ValidTiling n), countHorizontal n tiling = countVertical n tiling) ↔
  (n % 2 = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_domino_tiling_iff_odd_l560_56083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_average_gpa_is_93_l560_56029

/-- The average GPA for the school, given the GPAs of 6th, 7th, and 8th graders -/
noncomputable def average_school_gpa (gpa_6th gpa_7th gpa_8th : ℝ) : ℝ :=
  (gpa_6th + gpa_7th + gpa_8th) / 3

/-- Theorem stating that the average GPA for the school is 93 -/
theorem school_average_gpa_is_93 :
  let gpa_6th : ℝ := 93
  let gpa_7th : ℝ := gpa_6th + 2
  let gpa_8th : ℝ := 91
  average_school_gpa gpa_6th gpa_7th gpa_8th = 93 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_average_gpa_is_93_l560_56029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_c_closest_to_1600_in_mirror_l560_56071

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour : ℕ)
  (minute : ℕ)

/-- Represents the orientation of a clock -/
inductive ClockOrientation
  | Normal
  | Reversed

/-- A function that determines how close a clock is to 16:00 -/
noncomputable def closenessTo1600 (c : Clock) (o : ClockOrientation) : ℝ :=
  sorry

/-- The set of clocks A, B, C, D -/
def clocks : List Clock :=
  sorry

/-- Theorem stating that Clock C is closest to 16:00 when viewed in a mirror -/
theorem clock_c_closest_to_1600_in_mirror :
  let mirrorClocks := clocks.map (λ c => (c, ClockOrientation.Reversed))
  ∃ (c : Clock), c ∈ clocks ∧
    (∀ (other : Clock), other ∈ clocks →
      closenessTo1600 c ClockOrientation.Reversed ≤ closenessTo1600 other ClockOrientation.Reversed) ∧
    c = (clocks.get? 2).get sorry
  := by sorry

#check clock_c_closest_to_1600_in_mirror

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_c_closest_to_1600_in_mirror_l560_56071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_costs_85_cents_l560_56059

/-- The cost of a burger in cents -/
def burger_cost : ℕ → Prop := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ → Prop := sorry

/-- The cost of fries in cents -/
def fries_cost : ℕ → Prop := sorry

/-- Alice's purchase condition -/
axiom alice_purchase (b s f : ℕ) :
  burger_cost b → soda_cost s → fries_cost f →
  3 * b + 2 * s + f = 420

/-- Bill's purchase condition -/
axiom bill_purchase (b s f : ℕ) :
  burger_cost b → soda_cost s → fries_cost f →
  2 * b + s + 2 * f = 340

/-- The theorem stating the cost of a burger is 85 cents -/
theorem burger_costs_85_cents :
  ∃ (b s f : ℕ), burger_cost b ∧ soda_cost s ∧ fries_cost f ∧ b = 85 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_costs_85_cents_l560_56059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_m_value_l560_56080

noncomputable section

/-- Line l with parametric equations -/
def line_l (t m : ℝ) : ℝ × ℝ := ((Real.sqrt 3 / 2) * t + m, t / 2)

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 - 4 * Real.cos θ = 0 ∧ ρ ≥ 0

/-- Curve C in rectangular coordinates -/
def curve_C_rect (x y : ℝ) : Prop := y^2 = 4 * x

/-- Intersection points of line l and curve C -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t m ∧ curve_C_rect p.1 p.2}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Main theorem -/
theorem intersection_distance_implies_m_value (m : ℝ) :
  (∃ A B, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ distance A B = 16) → m = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_m_value_l560_56080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_32_over_3_l560_56072

/-- The area of a square inscribed in the ellipse (x²/4) + (y²/8) = 1, 
    with its sides parallel to the coordinate axes -/
noncomputable def inscribedSquareArea : ℝ := 32 / 3

/-- The equation of the ellipse -/
def ellipseEquation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 8 = 1

/-- Theorem: The area of the inscribed square is 32/3 -/
theorem inscribed_square_area_is_32_over_3 : 
  ∃ (s : ℝ), s > 0 ∧ 
  ellipseEquation s s ∧
  ellipseEquation (-s) s ∧
  ellipseEquation s (-s) ∧
  ellipseEquation (-s) (-s) ∧
  (2 * s)^2 = inscribedSquareArea :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_32_over_3_l560_56072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_theorem_l560_56045

theorem constant_function_theorem (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f)
  (h1' : Differentiable ℝ (deriv f))
  (h2 : ∀ x : ℝ, (deriv f x)^2 + deriv (deriv f) x ≤ 0) : 
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_theorem_l560_56045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l560_56042

-- Definition of associated equation
def is_associated_equation (eq : Real → Prop) (ineq : Real → Prop) : Prop :=
  ∃ x, eq x ∧ ineq x

-- Part 1
theorem part1 : is_associated_equation (λ x ↦ x - (3*x + 1) = -5) 
  (λ x ↦ -x + 2 > x - 5 ∧ 3*x - 1 > -x + 2) := by sorry

-- Part 2
theorem part2 : ∃ (eq : Real → Prop), is_associated_equation eq 
  (λ x ↦ x - 2 < 1 ∧ 1 + x > -x + 2) ∧ eq 1 := by sorry

-- Part 3
theorem part3 : ∀ m : Real, 
  (∃ (eq1 eq2 : Real → Prop), 
    is_associated_equation eq1 (λ x ↦ x < 2*x - m ∧ x - 2 ≤ m) ∧
    is_associated_equation eq2 (λ x ↦ x < 2*x - m ∧ x - 2 ≤ m) ∧
    eq1 (1/2) ∧ eq2 2) →
  0 ≤ m ∧ m < 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l560_56042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l560_56095

noncomputable section

open Real

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : Real.cos B = 5/13) 
  (h2 : Real.cos C = 4/5) 
  (h3 : c = 1) 
  (h4 : 0 < A ∧ A < π) 
  (h5 : 0 < B ∧ B < π) 
  (h6 : 0 < C ∧ C < π) 
  (h7 : A + B + C = π) 
  (h8 : Real.sin A / a = Real.sin B / b) 
  (h9 : Real.sin B / b = Real.sin C / c) 
  : a = 21/13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l560_56095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_l560_56097

-- Define the width of the room
def width : ℚ := 375/100

-- Define the total cost of paving
def total_cost : ℚ := 16500

-- Define the rate of paving per square meter
def rate_per_sqm : ℚ := 800

-- Define the length of the room
noncomputable def length : ℚ := total_cost / rate_per_sqm / width

-- Theorem stating that the length of the room is 5.5 meters
theorem room_length : length = 11/2 := by
  -- Unfold the definition of length
  unfold length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_l560_56097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_program_count_l560_56077

def courses : Finset String := 
  {"English", "Algebra", "Geometry", "History", "Art", "Latin", "Science", "Music"}

def math_courses : Finset String := 
  {"Algebra", "Geometry"}

def required_courses : Finset String := 
  {"English", "History"}

theorem valid_program_count :
  (Finset.filter (fun program : Finset String => 
    program ⊆ courses ∧ 
    program.card = 5 ∧
    required_courses ⊆ program ∧
    (∃ c ∈ math_courses, c ∈ program)) (Finset.powerset courses)).card = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_program_count_l560_56077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_with_specific_roots_and_difference_l560_56075

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_specific_roots_and_difference 
  (f g : ℝ → ℝ) (s : ℝ) :
  (∀ x, ∃ a b c d, f x = x^3 + a*x^2 + b*x + c ∧ g x = x^3 + a*x^2 + b*x + d) →  -- f and g are cubic polynomials
  (f (s + 2) = 0 ∧ f (s + 6) = 0) →  -- two roots of f
  (g (s + 4) = 0 ∧ g (s + 10) = 0) →  -- two roots of g
  (∀ x, f x - g x = 2 * s) →  -- constant difference between f and g
  s = 32 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_with_specific_roots_and_difference_l560_56075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_eddy_freddy_l560_56064

-- Define the parameters
noncomputable def distance_AB : ℝ := 600
noncomputable def distance_AC : ℝ := 360
noncomputable def time_Eddy : ℝ := 3
noncomputable def time_Freddy : ℝ := 4

-- Define the average speeds
noncomputable def speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def speed_Freddy : ℝ := distance_AC / time_Freddy

-- Theorem statement
theorem speed_ratio_eddy_freddy :
  (speed_Eddy / speed_Freddy) = 20 / 9 := by
  -- Expand the definitions
  unfold speed_Eddy speed_Freddy
  -- Simplify the expression
  simp [distance_AB, distance_AC, time_Eddy, time_Freddy]
  -- Prove the equality
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_eddy_freddy_l560_56064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l560_56001

/-- Given two vectors a and b in ℝ³, if a is parallel to b, then the sum of the second and third components of b is 1/2. -/
theorem parallel_vectors_sum (a b : ℝ × ℝ × ℝ) (h : ∃ (k : ℝ), a = k • b) :
  a = (2, -1, 2) → b.1 = 1 → b.2.1 + b.2.2 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l560_56001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_annual_return_l560_56033

-- Define the initial stock price and profit
noncomputable def initial_price : ℝ := 5000
noncomputable def profit : ℝ := 400

-- Define the annual return calculation
noncomputable def annual_return (initial_price profit : ℝ) : ℝ :=
  (profit / initial_price) * 100

-- Theorem statement
theorem stock_annual_return :
  annual_return initial_price profit = 8 := by
  -- Unfold the definition of annual_return
  unfold annual_return
  -- Simplify the expression
  simp [initial_price, profit]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_annual_return_l560_56033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_kite_l560_56007

/-- Parabola of the form y = ax^2 + c -/
structure Parabola1 where
  a : ℝ
  c : ℝ

/-- Parabola of the form y = c - bx^2 -/
structure Parabola2 where
  b : ℝ
  c : ℝ

/-- A kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola1
  p2 : Parabola2

def intersectsAxesInFourPoints (k : Kite) : Prop :=
  k.p1.a < 0 ∧ k.p2.b > 0

noncomputable def kiteArea (k : Kite) : ℝ :=
  3 * (k.p2.c - k.p1.c) * (Real.sqrt (k.p2.c / k.p2.b))

theorem parabola_intersection_kite (k : Kite) :
  k.p1.c = 3 ∧ k.p2.c = 9 ∧
  intersectsAxesInFourPoints k ∧
  kiteArea k = 18 →
  k.p1.a + k.p2.b = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_kite_l560_56007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sugar_l560_56053

/-- Represents the amount of flour in pounds -/
def flour : ℝ → ℝ := sorry

/-- Represents the amount of sugar in pounds -/
def sugar : ℝ → ℝ := sorry

/-- Represents the amount of oats in pounds -/
def oats : ℝ → ℝ := sorry

/-- The conditions given in the problem -/
def conditions (s : ℝ) : Prop :=
  flour s ≥ 4 + 2 * sugar s ∧
  flour s ≤ 3 * sugar s ∧
  oats s = flour s + sugar s

/-- The theorem stating that the minimum amount of sugar is 4 pounds -/
theorem min_sugar : 
  ∀ s : ℝ, conditions s → sugar s ≥ 4 ∧ conditions 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sugar_l560_56053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l560_56013

/-- The area of a triangle given three points in 2D space -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (1/2) * abs ((x₁*y₂ + x₂*y₃ + x₃*y₁) - (y₁*x₂ + y₂*x₃ + y₃*x₁))

/-- A point is on the line x + y = 9 -/
def isOnLine (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 = 9

theorem triangle_PQR_area :
  ∀ (R : ℝ × ℝ), isOnLine R → triangleArea (1, 2) (-2, 5) R = 9 :=
by
  intro R hR
  sorry

#check triangle_PQR_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l560_56013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_l560_56063

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of XXZ + YXZ + Z given X, Y, and Z are digits -/
def sum (X Y Z : Digit) : ℕ :=
  (100 * X.val + 10 * X.val + Z.val) + (100 * Y.val + 10 * X.val + Z.val) + Z.val

/-- Predicate to check if three digits are different -/
def are_different (X Y Z : Digit) : Prop :=
  X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z

theorem largest_sum :
  ∃ (X Y Z : Digit), are_different X Y Z ∧ 
    sum X Y Z = 987 ∧
    ∀ (A B C : Digit), are_different A B C → sum A B C ≤ 987 := by
  sorry

#eval sum ⟨8, sorry⟩ ⟨1, sorry⟩ ⟨7, sorry⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_l560_56063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l560_56084

/-- Given a triangle ABC with the specified conditions, prove the values of sin A, B, b, and c. -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sin (π/2 + A) = 11/14 →
  Real.cos (π - B) = -1/2 →
  a = 5 →
  a / Real.sin A = b / Real.sin B →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.sin A = 5*Real.sqrt 3/14 ∧ 
  B = π/3 ∧ 
  b = 7 ∧ 
  c = 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l560_56084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_difference_l560_56009

theorem max_angle_difference (x y : Real) (h1 : 0 < y) (h2 : y ≤ x) (h3 : x < π/2) 
  (h4 : Real.tan x = 3 * Real.tan y) :
  ∃ (max_diff : Real), max_diff = π/6 ∧ x - y ≤ max_diff ∧ 
  ∃ (x' y' : Real), x' - y' = max_diff ∧ 0 < y' ∧ y' ≤ x' ∧ x' < π/2 ∧ 
  Real.tan x' = 3 * Real.tan y' :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_difference_l560_56009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_domains_equal_l560_56004

-- Define the domain for both functions
noncomputable def Domain : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x / x
noncomputable def g (x : ℝ) : ℝ := x^0

-- Theorem statement
theorem f_equiv_g : 
  ∀ x ∈ Domain, f x = g x := by
  sorry

-- Additional theorem to show that the domains are equal
theorem domains_equal :
  ∀ x : ℝ, x ∈ Domain ↔ x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_domains_equal_l560_56004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_number_l560_56003

def number : ℕ := 304300000000

noncomputable def scientific_notation (n : ℕ) : ℝ × ℤ :=
  let s := toString n
  let l := s.length
  let m := (n : ℝ) / (10 ^ (l - 1) : ℝ)
  (m, l - 1)

theorem scientific_notation_of_number :
  scientific_notation number = (3.043, 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_number_l560_56003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l560_56038

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = (ax+b) / (1+x^2) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem odd_function_value (a b : ℝ) :
  IsOdd (f a b) → f a b (1/2) = 2/5 → f a b 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l560_56038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l560_56036

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x^2

-- State the theorem
theorem a_lower_bound 
  (h : ∀ (b x : ℝ), b ≤ 0 → e < x → x ≤ e^2 → f a b x ≥ x) : 
  a ≥ e^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l560_56036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_x_axis_symmetric_about_line_x_eq_2_symmetric_about_y_axis_l560_56048

-- Define a positive real number a that is not equal to 1
variable (a : ℝ) (ha : a > 0 ∧ a ≠ 1)

-- Define the functions
noncomputable def f1 (x : ℝ) := a^x - 1
noncomputable def g1 (x : ℝ) := -a^x + 1

noncomputable def f2 (x : ℝ) := a^(x - 2)
noncomputable def g2 (x : ℝ) := a^(2 - x)

noncomputable def f3 (x : ℝ) := a^(x + 2)
noncomputable def g3 (x : ℝ) := a^(2 - x)

-- Theorem statements
theorem symmetric_about_x_axis (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x : ℝ, f1 a x = -(g1 a x) := by sorry

theorem symmetric_about_line_x_eq_2 (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x : ℝ, f2 a x = g2 a (4 - x) := by sorry

theorem symmetric_about_y_axis (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x : ℝ, f3 a x = g3 a (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_x_axis_symmetric_about_line_x_eq_2_symmetric_about_y_axis_l560_56048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_k_for_integer_b_k_l560_56057

-- Define the sequence b_n
noncomputable def b : ℕ → ℝ
  | 0 => 2
  | (n + 1) => b n + (Real.log 7)⁻¹ * Real.log ((2 * ↑n + 3) / (2 * ↑n + 2))

-- Define the property for b_k to be an integer
def is_integer_b_k (k : ℕ) : Prop := ∃ m : ℤ, b k = ↑m

-- Theorem statement
theorem least_integer_k_for_integer_b_k :
  ∃ k : ℕ, k > 1 ∧ is_integer_b_k k ∧ ∀ j : ℕ, 1 < j ∧ j < k → ¬ is_integer_b_k j :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_k_for_integer_b_k_l560_56057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cube_difference_divisibility_l560_56021

theorem odd_cube_difference_divisibility (a b : ℤ) (n : ℕ)
  (ha : Odd a) (hb : Odd b) (hn : n > 0) :
  (2^n ∣ (a - b)) ↔ (2^n ∣ (a^3 - b^3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cube_difference_divisibility_l560_56021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_interpolation_l560_56090

-- Define the polynomial p(x) as noncomputable
noncomputable def p (a b c x : ℝ) : ℝ :=
  (a^4 * (x - b) * (x - c)) / ((a - b) * (a - c)) +
  (b^4 * (x - a) * (x - c)) / ((b - a) * (b - c)) +
  (c^4 * (x - a) * (x - b)) / ((c - a) * (c - b))

-- State the theorem
theorem quadratic_interpolation (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  p a b c a = a^4 ∧ p a b c b = b^4 ∧ p a b c c = c^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_interpolation_l560_56090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_multiplication_division_l560_56094

theorem percentage_error_multiplication_division (x : ℝ) (h : x ≠ 0) : 
  (|(x * 10 - x / 10)| / (x * 10)) * 100 = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_multiplication_division_l560_56094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l560_56037

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 8 - y^2 / 4 = 1

-- Define the eccentricity
noncomputable def eccentricity (C : (ℝ → ℝ → Prop)) : ℝ := Real.sqrt 6 / 2

-- Define the asymptotes
def asymptotes (C : (ℝ → ℝ → Prop)) : Set (ℝ → ℝ) := 
  {λ x => (Real.sqrt 2 / 2) * x, λ x => -(Real.sqrt 2 / 2) * x}

theorem hyperbola_properties :
  let C := hyperbola_C
  eccentricity C = Real.sqrt 6 / 2 ∧
  asymptotes C = {λ x => (Real.sqrt 2 / 2) * x, λ x => -(Real.sqrt 2 / 2) * x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l560_56037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_prime_twelfth_prime_l560_56055

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- State the given condition
theorem sixth_prime : nthPrime 6 = 13 := by sorry

-- State the theorem to be proved
theorem twelfth_prime : nthPrime 12 = 37 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_prime_twelfth_prime_l560_56055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_plus_linear_l560_56074

theorem divisibility_of_power_plus_linear (k : ℕ+) (a : ℤ) 
  (h : a ≡ 3 [ZMOD 8]) : 
  ∃ m : ℕ+, (2^(k:ℕ) : ℤ) ∣ (a^(m:ℕ) + a + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_power_plus_linear_l560_56074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_distance_proof_l560_56093

noncomputable def C1 (α : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos α, Real.sqrt 2 + Real.sqrt 2 * Real.sin α)

noncomputable def C2 (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, 2 * Real.sqrt 2 + 2 * Real.sqrt 2 * Real.sin α)

noncomputable def C1_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin θ

noncomputable def C2_polar (θ : ℝ) : ℝ := 4 * Real.sqrt 2 * Real.sin θ

theorem curve_and_distance_proof :
  (∀ α : ℝ, C2 α = (2 * (C1 α).1, 2 * (C1 α).2)) ∧
  (C2_polar (π/4) - C1_polar (π/4) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_distance_proof_l560_56093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_closed_form_l560_56022

def mySequence (n : ℕ) : ℚ :=
  if n = 0 then 1 else 2 * mySequence (n - 1) + 2

theorem mySequence_closed_form (n : ℕ) :
  mySequence n = 3 * 2^n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_closed_form_l560_56022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_disorder_given_positive_l560_56041

/-- The probability of having the disorder -/
noncomputable def p_disorder : ℝ := 1 / 200

/-- The probability of not having the disorder -/
noncomputable def p_no_disorder : ℝ := 1 - p_disorder

/-- The true positive rate (sensitivity) of the test -/
def sensitivity : ℝ := 1

/-- The false positive rate of the test -/
def false_positive_rate : ℝ := 0.05

/-- The probability of testing positive -/
noncomputable def p_positive : ℝ := sensitivity * p_disorder + false_positive_rate * p_no_disorder

/-- The main theorem: probability of having the disorder given a positive test result -/
theorem prob_disorder_given_positive : 
  (sensitivity * p_disorder) / p_positive = 20 / 219 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_disorder_given_positive_l560_56041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_sixths_pi_plus_alpha_l560_56060

theorem sin_five_sixths_pi_plus_alpha (α : ℝ) 
  (h : Real.cos (π / 3 + α) = 1 / 3) : 
  Real.sin (5 * π / 6 + α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_sixths_pi_plus_alpha_l560_56060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_or_isosceles_l560_56054

theorem triangle_right_or_isosceles (α β γ : ℝ) 
  (triangle_angles : α + β + γ = Real.pi) 
  (angle_condition : Real.sin α / Real.sin β = Real.cos β / Real.cos α) : 
  α = β ∨ γ = Real.pi/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_or_isosceles_l560_56054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l560_56099

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 1}
def B : Set ℝ := {-2, -1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l560_56099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tileable_iff_not_one_two_four_l560_56046

/-- A tile is either 5×5 or 1×3 -/
inductive Tile
  | big : Tile
  | small : Tile

/-- A tiling is a function from board positions to optional tiles -/
def Tiling (N : ℕ) := Fin N → Fin N → Option Tile

/-- A tiling is valid if it covers the entire board without overlaps -/
def is_valid_tiling (N : ℕ) (t : Tiling N) : Prop :=
  ∀ i j : Fin N, ∃! tile : Tile, 
    (tile = Tile.big ∧ i.val < N - 4 ∧ j.val < N - 4 ∧ 
      (∀ x y, x < 5 → y < 5 → t ⟨i.val + x, sorry⟩ ⟨j.val + y, sorry⟩ = some Tile.big)) ∨
    (tile = Tile.small ∧ j.val < N - 2 ∧
      (∀ y, y < 3 → t i ⟨j.val + y, sorry⟩ = some Tile.small))

/-- An N×N board is tileable if there exists a valid tiling for it -/
def is_tileable (N : ℕ) : Prop :=
  ∃ t : Tiling N, is_valid_tiling N t

/-- Main theorem: An N×N board is tileable if and only if N is not 1, 2, or 4 -/
theorem tileable_iff_not_one_two_four (N : ℕ) :
  is_tileable N ↔ N ≠ 1 ∧ N ≠ 2 ∧ N ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tileable_iff_not_one_two_four_l560_56046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_apple_count_smallest_valid_apple_count_is_200_l560_56049

def is_valid_apple_count (n : ℕ) : Prop :=
  n > 2 ∧ n % 9 = 2 ∧ n % 10 = 2 ∧ n % 11 = 2

theorem smallest_valid_apple_count :
  ∃ (m : ℕ), is_valid_apple_count m ∧ ∀ (n : ℕ), is_valid_apple_count n → m ≤ n :=
by
  -- The proof would go here
  sorry

-- Remove the #eval line as it's causing the DecidablePred error
-- Instead, we can use a theorem to state the result

theorem smallest_valid_apple_count_is_200 :
  ∃ (m : ℕ), is_valid_apple_count m ∧ m = 200 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_apple_count_smallest_valid_apple_count_is_200_l560_56049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brennan_file_download_l560_56096

theorem brennan_file_download (first_round : ℕ) : first_round = 800 :=
  let kept_first_round := (30 : ℕ) * first_round / 100
  let second_round := 400
  let kept_second_round := (2 : ℕ) * second_round / 5
  let total_kept := 400
  have h : kept_first_round + kept_second_round = total_kept := by
    sorry
  by
    sorry

#check brennan_file_download

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brennan_file_download_l560_56096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_31_36_l560_56070

-- Define the function g
noncomputable def g (a b : ℝ) : ℝ :=
  if a + b ≤ 5 then
    (3 * a * b - a + 4) / (3 * a)
  else
    (3 * a * b - b - 5) / (-3 * b)

-- State the theorem
theorem g_sum_equals_31_36 : g 3 2 + g 2 4 = 31 / 36 := by
  -- Evaluate g(3,2)
  have h1 : g 3 2 = 19 / 9 := by
    unfold g
    simp [if_pos]
    norm_num
    
  -- Evaluate g(2,4)
  have h2 : g 2 4 = -5 / 4 := by
    unfold g
    simp [if_neg]
    norm_num
    
  -- Add the results
  calc
    g 3 2 + g 2 4 = 19 / 9 + (-5 / 4) := by rw [h1, h2]
    _             = 31 / 36           := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_31_36_l560_56070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l560_56025

/-- The distance from the center of the circle x^2 + y^2 - 2x + 2y = 0 to the line y = x + 1 is 3√2/2 -/
theorem distance_circle_center_to_line : 
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 / 2 ∧
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y = 0 → (x - 1)^2 + (y + 1)^2 = 2) ∧
    (∀ x y : ℝ, y = x + 1 → -x + y - 1 = 0) ∧
    d = |(- x₀ + y₀ - 1)| / Real.sqrt 2 ∧
    x₀ = 1 ∧ y₀ = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l560_56025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_problem_l560_56020

/-- The length of a wire stretched between three poles -/
noncomputable def wire_length (d1 d2 h1 h2 h3 : ℝ) : ℝ :=
  Real.sqrt (d1^2 + (h2 - h1)^2) + Real.sqrt (d2^2 + (h2 - h3)^2)

/-- Theorem stating the length of the wire in the given problem -/
theorem wire_length_problem : 
  wire_length 16 18 8 22 10 = Real.sqrt 452 + Real.sqrt 468 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_problem_l560_56020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_conditions_l560_56050

/-- A triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  t.C = 90 ∨ t.a^2 + t.b^2 = t.c^2

theorem right_triangle_conditions (t : Triangle) :
  (t.C = 90) →
  (t.A + t.B = t.C) →
  (∃ k : ℝ, t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k) →
  isRightTriangle t ∧
  ¬(∃ k : ℝ, t.A = 3*k ∧ t.B = 4*k ∧ t.C = 5*k → isRightTriangle t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_conditions_l560_56050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_thousand_six_hundred_scientific_notation_l560_56008

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Define the property of scientific notation
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = scientific_notation a n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem statement
theorem eight_thousand_six_hundred_scientific_notation :
  is_scientific_notation 8600 8.6 3 := by
  sorry

#check eight_thousand_six_hundred_scientific_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_thousand_six_hundred_scientific_notation_l560_56008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_determines_k_l560_56069

/-- The value of k for which the lines y = 7x + 6, y = -3x + 9, and y = 4x + k intersect at the same point -/
noncomputable def k : ℝ := 69/10

/-- The x-coordinate of the intersection point -/
noncomputable def x : ℝ := 3/10

/-- The y-coordinate of the intersection point -/
noncomputable def y : ℝ := 81/10

/-- The first line: y = 7x + 6 -/
def line1 (x : ℝ) : ℝ := 7 * x + 6

/-- The second line: y = -3x + 9 -/
def line2 (x : ℝ) : ℝ := -3 * x + 9

/-- The third line: y = 4x + k -/
noncomputable def line3 (x : ℝ) : ℝ := 4 * x + k

theorem intersection_point_determines_k :
  line1 x = y ∧ line2 x = y ∧ line3 x = y → k = 69/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_determines_k_l560_56069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l560_56067

theorem distance_point_to_line (a : ℝ) (h1 : a > 0) : 
  (|a - 2 + 3| / Real.sqrt 2 = 1) → (a = Real.sqrt 2 - 1) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l560_56067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equation_l560_56023

/-- The projection of vector v onto vector u is given by (v ⬝ u / ∥u∥²) * u -/
noncomputable def projection (v u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * u.1 + v.2 * u.2) / (u.1^2 + u.2^2)
  (scalar * u.1, scalar * u.2)

theorem projection_equation (c : ℝ) : 
  projection (-3, c) (3, 2) = (21/13, 14/13) → c = 8 := by
  sorry

#check projection_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equation_l560_56023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_square_roots_equals_364_315_l560_56018

/-- Represents a circle in the upper half-plane tangent to the x-axis -/
structure Circle where
  radius : ℚ
  deriving Repr

/-- Represents a layer of circles -/
def Layer := List Circle

/-- Calculates the radius of a new circle tangent to two given circles -/
noncomputable def newRadius (r₁ r₂ : ℚ) : ℚ := (r₁ * r₂) / ((r₁.sqrt + r₂.sqrt) ^ 2)

/-- Constructs the initial layer L₀ -/
def L₀ : Layer := [
  { radius := 10000 },
  { radius := 11025 },
  { radius := 12100 }
]

/-- Constructs layer Lₖ given the previous layers -/
def constructLayer (prevLayers : List Layer) : Layer := sorry

/-- Constructs all layers up to L₅ -/
def allLayers : List Layer := sorry

/-- The set S of all circles in layers L₀ to L₅ -/
def S : List Circle := sorry

/-- Calculates the sum of 1/√r(C) for all circles C in S -/
noncomputable def sumInverseSquareRoots (circles : List Circle) : ℚ := sorry

/-- The main theorem to prove -/
theorem sum_inverse_square_roots_equals_364_315 :
  sumInverseSquareRoots S = 364 / 315 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_square_roots_equals_364_315_l560_56018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_g_nonnegative_implies_a_leq_one_l560_56085

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

-- State the theorem
theorem f_plus_g_nonnegative_implies_a_leq_one (a : ℝ) :
  (∀ x > 0, f x + g a x ≥ 0) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_g_nonnegative_implies_a_leq_one_l560_56085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_vertex_C_l560_56073

/-- Given a triangle ABC where AB = 2 and AD is a median with length 1.5,
    prove that the locus of vertex C is a circle with radius 3 and
    center on AB at a distance 4 from B -/
theorem locus_of_vertex_C (A B C D : ℝ × ℝ) : 
  (norm (A - B) = 2) →
  (D = (B + C) / 2) →
  (norm (A - D) = 3/2) →
  ∃ (O : ℝ × ℝ), (O.1 = -2 ∧ O.2 = 0) ∧ norm (C - O) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_vertex_C_l560_56073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l560_56087

theorem floor_ceil_fraction_square : ⌊⌈(24/5 : ℚ)^2⌉ + 11/3⌋ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l560_56087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_point_l560_56086

/-- Given a circle with center O and radius r, and a line l with distance d > r from O,
    prove that A(0, √(d² - r²)) forms a constant angle ∠MAN for any M and N on l
    such that the circle with diameter MN is tangent to the given circle externally. -/
theorem constant_angle_point (d r : ℝ) (h : d > r) :
  let O : ℝ × ℝ := (0, d)
  let A : ℝ × ℝ := (0, Real.sqrt (d^2 - r^2))
  ∀ (M N : ℝ × ℝ),
    M.2 = 0 ∧ N.2 = 0 →  -- M and N are on the x-axis (line l)
    (∃ s : ℝ, (M.1 - s)^2 + d^2 = (r + s)^2 ∧
              (N.1 + s)^2 + d^2 = (r + s)^2) →  -- Circle with diameter MN is tangent to the given circle
    ∃ θ : ℝ, ∀ (M' N' : ℝ × ℝ),
      M'.2 = 0 ∧ N'.2 = 0 →
      (∃ s' : ℝ, (M'.1 - s')^2 + d^2 = (r + s')^2 ∧
                 (N'.1 + s')^2 + d^2 = (r + s')^2) →
      (A.1 - M.1) * (N.2 - M.2) - (A.2 - M.2) * (N.1 - M.1) =
      (A.1 - M'.1) * (N'.2 - M'.2) - (A.2 - M'.2) * (N'.1 - M'.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_point_l560_56086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l560_56047

-- Define the ellipse C
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the slope of a line through the origin
noncomputable def line_slope (x y : ℝ) : ℝ := y / x

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ellipse a b 2 0 ∧ eccentricity a b = 1/2 →
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧
    line_slope x1 y1 * line_slope x2 y2 = -3/4 →
    (a = 2 ∧ b = Real.sqrt 3) ∧
    ∃ (x3 y3 : ℝ),
      ellipse a b x3 y3 ∧
      (3*x1 - x2) / (x3 - x2) = (3*y1 - y2) / (y3 - y2) ∧
      (3*x1 - x2) / (x3 - x2) = 5 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l560_56047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_polygon_existence_l560_56010

/-- Check if a list of real numbers can form a polygon -/
def is_polygon (sides : List ℝ) : Prop :=
  ∀ (i : Fin sides.length), sides[i] < (sides.sum - sides[i])

/-- Theorem stating the existence of a set of 100 sticks that can form a polygon,
    but no proper subset of it can form a polygon -/
theorem stick_polygon_existence : ∃ (sticks : Finset ℝ), 
  (Finset.card sticks = 100) ∧ 
  (∀ (s : Finset ℝ), s ⊆ sticks → Finset.card s < 100 → ¬ is_polygon (Finset.toList s)) ∧
  (is_polygon (Finset.toList sticks)) := by
  sorry

#check stick_polygon_existence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_polygon_existence_l560_56010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_of_divisibility_l560_56031

theorem square_ratio_of_divisibility (a b : ℕ) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_of_divisibility_l560_56031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l560_56065

/-- A right triangle with legs a and b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Volume of a cone with radius r and height h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem right_triangle_hypotenuse (t : RightTriangle) 
  (h1 : cone_volume t.b t.a = 640 * Real.pi)
  (h2 : cone_volume t.a t.b = 1536 * Real.pi) :
  Real.sqrt (t.a^2 + t.b^2) = 32.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l560_56065
