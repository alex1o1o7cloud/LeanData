import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l281_28163

/-- The speed of a train given its length and time to cross a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem: A 360-meter long train crossing a point in 6 seconds has a speed of 216 km/h -/
theorem train_speed_calculation :
  train_speed 360 6 = 216 := by
  -- Expand the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l281_28163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_sixth_game_l281_28155

-- Define a fair coin toss
noncomputable def fairCoinToss : ℝ := 1 / 2

-- Define the probability of winning when starting first
noncomputable def probWinStartingFirst : ℝ := 2 / 3

-- Define the probability of winning the nth game
noncomputable def probWinNthGame (n : ℕ) : ℝ :=
  (1 + (-1)^(n + 1) / 3^n) / 2

-- State the theorem
theorem prob_win_sixth_game :
  probWinNthGame 6 = 1 / 2 * (1 - 1 / 729) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_sixth_game_l281_28155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_min_b_value_l281_28117

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1
def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + 4

-- Theorem for the monotonicity of f
theorem f_monotonicity (a : ℝ) :
  ∃ (I : Set ℝ), Set.Nonempty I ∧ 
  (∀ x ∈ I, MonotoneOn (f a) I) ∧
  (∃ J : Set ℝ, Set.Nonempty J ∧ J ≠ I ∧ ∀ x ∈ J, ¬MonotoneOn (f a) J) := by
  sorry

-- Theorem for the minimum value of b
theorem min_b_value :
  ∃ b : ℝ, b = 17/8 ∧
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f (1/4) x₁ ≥ g b x₂) ∧
  (∀ b' < b, ∃ x₁ ∈ Set.Ioo 0 2, ∀ x₂ ∈ Set.Icc 1 2, f (1/4) x₁ < g b' x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_min_b_value_l281_28117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_change_l281_28166

/-- Given a regression equation y = 2 - 2.5x, prove that when x increases by 2 units, y decreases by 5 units. -/
theorem regression_change (x y : ℝ) :
  y = 2 - 2.5 * x →
  (2 - 2.5 * (x + 2)) - y = -5 := by
  intro h
  calc
    (2 - 2.5 * (x + 2)) - y
      = (2 - 2.5 * x - 5) - y := by ring
    _ = (2 - 2.5 * x - 5) - (2 - 2.5 * x) := by rw [h]
    _ = -5 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_change_l281_28166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l281_28100

theorem trigonometric_identity (γ : ℝ) : 
  3 * Real.tan γ ^ 2 + 3 * (1 / Real.tan γ) ^ 2 + 2 / Real.sin γ ^ 2 + 2 / Real.cos γ ^ 2 = 19 →
  Real.sin γ ^ 4 - Real.sin γ ^ 2 = -1/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l281_28100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_promotional_cost_l281_28137

/-- Profit function for a promotional event -/
noncomputable def profit (x : ℝ) : ℝ := 25 - (36 / (x + 3) + x)

/-- Theorem stating the optimal promotional cost for maximum profit -/
theorem optimal_promotional_cost (a : ℝ) (h_a : a > 0) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ ∀ y : ℝ, 0 ≤ y ∧ y ≤ a → profit x ≥ profit y) →
  ((a ≥ 3 → ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, 0 ≤ y ∧ y ≤ a → profit x ≥ profit y) ∧
   (a < 3 → ∃ x : ℝ, x = a ∧ ∀ y : ℝ, 0 ≤ y ∧ y ≤ a → profit x ≥ profit y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_promotional_cost_l281_28137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_integral_path_independent_l281_28107

/-- The vector field F(x,y) = (4x + 8y + 5, 9x + 8) -/
def F (x y : ℝ) : ℝ × ℝ := (4*x + 8*y + 5, 9*x + 8)

/-- Point O -/
def O : ℝ × ℝ := (0, 0)

/-- Point A -/
def A : ℝ × ℝ := (4, 5)

/-- Point C -/
def C : ℝ × ℝ := (0, 5)

/-- The parabola y = (5/16)x^2 -/
noncomputable def parabola (x : ℝ) : ℝ := (5/16) * x^2

/-- Line integral along the direct path OA -/
def integral_OA : ℝ := 262

/-- Line integral along the path OCA -/
def integral_OCA : ℝ := 252

/-- Line integral along the parabola OA -/
noncomputable def integral_parabola : ℝ := 796/3

theorem line_integral_path_independent :
  integral_OA = 262 ∧ integral_OCA = 252 ∧ integral_parabola = 796/3 →
  ∃ (v : ℝ), v = integral_OA ∧ v = integral_OCA ∧ v = integral_parabola := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_integral_path_independent_l281_28107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_neither_sufficient_nor_necessary_l281_28185

-- Define a geometric sequence
def geometric_sequence (a₀ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n ↦ a₀ * q^n

-- Define a decreasing sequence
def is_decreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s (n + 1) < s n

-- Theorem statement
theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a₀ q : ℝ, 0 < q ∧ q < 1 → is_decreasing (geometric_sequence a₀ q)) ∧
  ¬(∀ a₀ q : ℝ, is_decreasing (geometric_sequence a₀ q) → (0 < q ∧ q < 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_neither_sufficient_nor_necessary_l281_28185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l281_28127

/-- The function f(x) = (7x^2 - 4) / (4x^2 + 7x - 5) -/
noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 4) / (4 * x^2 + 7 * x - 5)

/-- The horizontal asymptote of f(x) is y = 7/4 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x - 7/4| < ε :=
by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l281_28127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_line_l281_28143

open Real

-- Define the point P in polar coordinates
noncomputable def P : ℝ × ℝ := (2, π/3)

-- Define the line equation in polar form
def line_equation (ρ θ : ℝ) : Prop := ρ * (cos θ + Real.sqrt 3 * sin θ) = 6

-- Define the distance function
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  let x := p.1 * cos p.2
  let y := p.1 * sin p.2
  abs (x + Real.sqrt 3 * y - 6) / Real.sqrt 4

-- Theorem statement
theorem distance_from_P_to_line :
  distance_to_line P = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_line_l281_28143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l281_28169

def g (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem functional_equation_solutions (a b : ℝ) :
  ∃ (f : ℝ → ℝ), (∀ x, f (g a b x) = g a b (f x)) ∧
  ((a = 1 ∧ b = 0) ∨
   (a = 1 ∧ b ≠ 0 ∧ ∃ c, ∀ x, f x = x + c) ∨
   (a ≠ 1 ∧ a ≠ -1 ∧ ∃ l, ∀ x, f x = b/(1-a) + l * (x - b/(1-a))) ∨
   (a = -1 ∧ ∀ x, f (-x) = -f x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l281_28169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_tetrahedron_with_all_labels_l281_28114

/-- Represents a tetrahedron with vertices labeled 1, 2, 3, 4 -/
structure OriginalTetrahedron where
  vertices : Fin 4 → Fin 4
  is_valid : ∀ i : Fin 4, vertices i = i

/-- Represents a small tetrahedron resulting from the division -/
structure SmallTetrahedron where
  vertices : Fin 4 → Fin 4

/-- Represents the division of the original tetrahedron into smaller ones -/
structure TetrahedronDivision where
  original : OriginalTetrahedron
  small_tetrahedra : Set SmallTetrahedron
  labeling_rule : SmallTetrahedron → Fin 4 → Fin 4

/-- The main theorem stating the existence of a small tetrahedron with vertices labeled 1, 2, 3, 4 -/
theorem exists_small_tetrahedron_with_all_labels (division : TetrahedronDivision) :
  ∃ t ∈ division.small_tetrahedra, ∀ i : Fin 4, ∃ j : Fin 4, division.labeling_rule t i = j ∧
    (∀ k : Fin 4, k ≠ i → division.labeling_rule t i ≠ division.labeling_rule t k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_tetrahedron_with_all_labels_l281_28114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usb_available_space_l281_28199

/-- Represents the capacity of a USB drive in gigabytes -/
def UsbCapacity : ℚ := 16

/-- Represents the percentage of the USB drive that is already used -/
def UsedPercentage : ℚ := 50

/-- Calculates the available space on a USB drive given its total capacity and the percentage used -/
def availableSpace (capacity : ℚ) (usedPercentage : ℚ) : ℚ :=
  capacity * (1 - usedPercentage / 100)

/-- Theorem stating that the available space on a 16 GB USB drive with 50% used is 8 GB -/
theorem usb_available_space :
  availableSpace UsbCapacity UsedPercentage = 8 := by
  -- Unfold the definitions and simplify
  unfold availableSpace UsbCapacity UsedPercentage
  -- Perform the calculation
  simp [Rat.mul_def, Rat.sub_def, Rat.div_def]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_usb_available_space_l281_28199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l281_28182

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Definition of a point on the ellipse -/
def on_ellipse (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- Definition of the right focus of the ellipse -/
noncomputable def right_focus (E : Ellipse) : ℝ × ℝ :=
  (Real.sqrt (E.a^2 - E.b^2), 0)

/-- Definition of an isosceles right triangle -/
def is_isosceles_right_triangle (A B F : ℝ × ℝ) : Prop :=
  let AF := (F.1 - A.1, F.2 - A.2)
  let BF := (F.1 - B.1, F.2 - B.2)
  AF.1^2 + AF.2^2 = BF.1^2 + BF.2^2 ∧ 
  AF.1 * BF.1 + AF.2 * BF.2 = 0

/-- The main theorem -/
theorem ellipse_eccentricity (E : Ellipse) :
  (∃ t : ℝ, ∃ A B : ℝ × ℝ, 
    on_ellipse E A.1 A.2 ∧ 
    on_ellipse E B.1 B.2 ∧ 
    A.2 = t ∧ B.2 = t ∧
    is_isosceles_right_triangle A B (right_focus E)) →
  Real.sqrt (E.a^2 - E.b^2) / E.a = Real.sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l281_28182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l281_28145

-- Define the function f(x) = x + sin x
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorem
theorem range_of_f_on_interval :
  Set.range (fun x ↦ f x) ∩ Set.Icc 0 (2 * Real.pi) = Set.Icc 0 (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l281_28145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l281_28126

-- Define the ellipse C
def ellipse_C (b : ℝ) (x y : ℝ) : Prop := x^2 / 4 + y^2 / b^2 = 1

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Theorem statement
theorem ellipse_eccentricity (b : ℝ) (h1 : 0 < b) (h2 : b < 2) :
  (∀ x y, ellipse_C b x y) → (2 * b = 2) → eccentricity 2 b = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l281_28126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l281_28181

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x + Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_max_min_on_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x ≤ (3 + Real.sqrt 2) / 2 ∧
    1 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l281_28181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_weight_calculation_l281_28179

/-- Nancy's weight calculation problem -/
theorem nancy_weight_calculation (W R E : ℝ) : 
  W = R - 2 →
  0.05 * (W + 2) = E →
  R - E = 64 →
  ∃ (ε : ℝ), abs (W - 65.37) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_weight_calculation_l281_28179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_approximation_l281_28171

theorem z_value_approximation (x y z : ℝ) (hx : x = 100.48) (hy : y = 100.70) (hxz : x * z = y ^ 2) :
  ∃ ε > 0, |z - 100.92| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_approximation_l281_28171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_plus_cos_l281_28192

theorem min_value_sin_plus_cos :
  ∃ m : Real, m = Real.sqrt 2 ∧ ∀ x : Real, 0 < x ∧ x < Real.pi → Real.sin x + Real.cos x ≥ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_plus_cos_l281_28192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_unchanged_l281_28161

noncomputable section

def rectangle_length : ℝ := 40
def rectangle_width : ℝ := 20
def shed_area : ℝ := 100

def rectangle_area : ℝ := rectangle_length * rectangle_width
def rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)

def square_side : ℝ := rectangle_perimeter / 4
def square_area : ℝ := square_side * square_side

def usable_square_area : ℝ := square_area - shed_area

theorem garden_area_unchanged :
  usable_square_area = rectangle_area :=
by
  -- Expand definitions
  unfold usable_square_area square_area square_side rectangle_perimeter rectangle_area
  -- Simplify
  simp [rectangle_length, rectangle_width, shed_area]
  -- The proof is completed by computation
  norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_unchanged_l281_28161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l281_28150

/-- The area of a triangle with vertices (2,1,0), (5,3,2), and (11,7,4) is √22 -/
theorem triangle_area : ∃ area : ℝ, area = Real.sqrt 22 := by
  -- Define the vertices
  let u : Fin 3 → ℝ := ![2, 1, 0]
  let v : Fin 3 → ℝ := ![5, 3, 2]
  let w : Fin 3 → ℝ := ![11, 7, 4]

  -- Calculate the area using the cross product method
  let area : ℝ := (1/2) * Real.sqrt (
    (v 1 - u 1) * (w 2 - u 2) - (v 2 - u 2) * (w 1 - u 1) ^ 2 +
    (v 2 - u 2) * (w 0 - u 0) - (v 0 - u 0) * (w 2 - u 2) ^ 2 +
    (v 0 - u 0) * (w 1 - u 1) - (v 1 - u 1) * (w 0 - u 0) ^ 2
  )

  -- Prove that the calculated area equals √22
  use area
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l281_28150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_difference_and_total_l281_28101

def planned_daily_production : ℕ := 100

def production_differences : List ℤ := [-1, 3, -2, 4, 7, -5, -10]

def actual_production (diff : ℤ) : ℕ := planned_daily_production + diff.toNat

theorem production_difference_and_total :
  (∃ max min : ℕ,
    max ∈ production_differences.map actual_production ∧
    min ∈ production_differences.map actual_production ∧
    (∀ x ∈ production_differences.map actual_production, x ≤ max) ∧
    (∀ x ∈ production_differences.map actual_production, min ≤ x) ∧
    max - min = 17) ∧
  (planned_daily_production * 7 + production_differences.sum = 696) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_difference_and_total_l281_28101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l281_28156

theorem tan_ratio_from_sin_sum_diff (α β : ℝ) 
  (h1 : Real.sin (α + β) = 1/5) 
  (h2 : Real.sin (α - β) = 3/5) : 
  Real.tan α / Real.tan β = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l281_28156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_pq_l281_28172

theorem right_triangle_pq (P Q R : Real) (tanP : ℝ) (PR PQ : ℝ) :
  P + Q + R = Real.pi →
  R = Real.pi / 2 →
  tanP = 3 / 2 →
  PR = 6 →
  tanP = PQ / PR →
  PQ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_pq_l281_28172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_equation_l281_28196

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are negatively correlated -/
def negatively_correlated (x y : ℝ → ℝ) : Prop :=
  ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ > y t₂

/-- Calculates the mean of a sample -/
noncomputable def sample_mean (sample : List ℝ) : ℝ :=
  (sample.sum) / (sample.length : ℝ)

/-- Checks if a point lies on a linear regression line -/
def point_on_line (x y : ℝ) (reg : LinearRegression) : Prop :=
  y = reg.slope * x + reg.intercept

theorem correct_regression_equation
  (x y : ℝ → ℝ)
  (h_corr : negatively_correlated x y)
  (h_mean_x : ∃ sample_x : List ℝ, sample_mean sample_x = 3)
  (h_mean_y : ∃ sample_y : List ℝ, sample_mean sample_y = 3.5)
  : point_on_line 3 3.5 { slope := -2, intercept := 9.5 } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_equation_l281_28196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l281_28159

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 2| / Real.sqrt 2

-- Theorem statement
theorem max_distance_ellipse_to_line :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 5 + Real.sqrt 2 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≤ max_dist ∧
    ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = max_dist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l281_28159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_l281_28102

theorem induction_step (k : ℕ) :
  let f (n : ℕ) := (Finset.range n).prod (λ i => n + i + 1)
  let g (n : ℕ) := 2^n * (Finset.range n).prod (λ i => 2 * i + 1)
  f (k + 1) = f k * (2 * (2 * k + 1)) ∧ g (k + 1) = g k * (2 * (2 * k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_l281_28102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_inequality_l281_28191

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x + 2| - 5)

-- Define the domain A
def A : Set ℝ := {x | x ≤ -4 ∨ x ≥ 1}

-- Define the set B
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem domain_and_inequality :
  (∀ x, x ∈ A ↔ f x ∈ Set.univ) ∧
  (∀ a b, a ∈ (B ∩ (Aᶜ)) → b ∈ (B ∩ (Aᶜ)) → |a + b| / 2 < |1 + a * b / 4|) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_inequality_l281_28191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l281_28195

theorem average_difference : ∃ (x : ℝ), x = 7 := by
  let set1 := [20, 40, 60]
  let set2 := [10, 70, 19]
  let avg1 := (set1.sum) / (set1.length : ℝ)
  let avg2 := (set2.sum) / (set2.length : ℝ)
  have h : avg1 - avg2 = 7 := by
    -- Proof steps would go here
    sorry
  exact ⟨avg1 - avg2, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l281_28195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l281_28190

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + n * (n - 1) / 2 * d

-- Theorem statement
theorem arithmetic_sequence_sum_property (a₁ d : ℝ) :
  S a₁ d 6 - a₁ = 15 → S a₁ d 7 = 21 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l281_28190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_m_l281_28186

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1 - (x^2 - 3*x + 3) * Real.exp x

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The theorem stating the result of the problem -/
theorem tangent_point_m (m : ℝ) : 
  (∃! (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    (g t₁ - (-4) = (deriv g t₁) * (t₁ - m)) ∧ 
    (g t₂ - (-4) = (deriv g t₂) * (t₂ - m))) →
  m = -1 ∨ m = 7/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_m_l281_28186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_number_properties_l281_28152

theorem basic_number_properties :
  (- (-6) = 6) ∧
  (abs (-8) = 8) ∧
  ((-3 : ℚ)⁻¹ = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_number_properties_l281_28152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equilateral_triangle_l281_28148

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ

-- Define a function to calculate the sum of distances
noncomputable def sum_distances_to_midpoints (t : EquilateralTriangle) : ℝ :=
  3 * (t.side_length * Real.sqrt 3) / 4

-- Theorem statement
theorem sum_distances_equilateral_triangle :
  ∀ t : EquilateralTriangle, t.side_length = 3 →
  sum_distances_to_midpoints t = (9 * Real.sqrt 3) / 4 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equilateral_triangle_l281_28148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l281_28168

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l281_28168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_fewer_heads_12_flips_l281_28164

/-- The number of coin flips -/
noncomputable def n : ℕ := 12

/-- The probability of getting heads on a single flip -/
noncomputable def p : ℝ := 1/2

/-- The probability of getting fewer heads than tails in n fair coin flips -/
noncomputable def prob_fewer_heads (n : ℕ) (p : ℝ) : ℝ :=
  1/2 * (1 - (n.choose (n/2)) * p^(n/2) * (1-p)^(n/2))

theorem prob_fewer_heads_12_flips :
  prob_fewer_heads n p = 1586/4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_fewer_heads_12_flips_l281_28164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l281_28184

theorem trigonometric_problem (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2 ∧ π/2 < β ∧ β < π)
  (h2 : Real.cos (β - π/4) = 1/3)
  (h3 : Real.sin (α + β) = 4/5) :
  Real.sin (2*β) = -7/9 ∧
  (fun x => Real.cos x - Real.sin x) α = (16 - 3*Real.sqrt 2) / 15 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l281_28184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_with_discount_l281_28174

/-- The final cost of pens with a bulk discount -/
theorem pen_cost_with_discount (dozen_cost : ℚ) (quantity : ℚ) (discount_rate : ℚ) : 
  dozen_cost = 18 ∧ 
  quantity = 5/2 ∧ 
  discount_rate = 15/100 ∧ 
  quantity ≥ 2 →
  dozen_cost * quantity * (1 - discount_rate) = 3825/100 := by
  intro h
  sorry

#check pen_cost_with_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_with_discount_l281_28174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_period_f_monotone_increasing_l281_28120

noncomputable def P (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x) + 1, 1)
noncomputable def Q (x : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + 1)
def O : ℝ × ℝ := (0, 0)

noncomputable def f (x : ℝ) : ℝ := (P x - O).1 * (Q x - O).1 + (P x - O).2 * (Q x - O).2

theorem f_expression (x : ℝ) : f x = 2 * Real.sin (2 * x + Real.pi / 6) + 2 := by sorry

theorem f_period : ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧ T = Real.pi := by sorry

theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_period_f_monotone_increasing_l281_28120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_spherical_shell_moment_of_inertia_l281_28154

/-- Predicate to define what it means to be a moment of inertia for a thin spherical shell -/
def is_moment_of_inertia (Θ M R : ℝ) : Prop :=
  Θ ≥ 0 ∧  -- Moment of inertia is non-negative
  Θ = (2/3) * M * R^2 ∧  -- The formula for a thin spherical shell
  ∀ (k : ℝ), k > 1 → Θ < k * M * R^2  -- Upper bound property

/-- The moment of inertia of a thin spherical shell -/
theorem thin_spherical_shell_moment_of_inertia 
  (M R : ℝ) (h₁ : M > 0) (h₂ : R > 0) : 
  ∃ Θ : ℝ, Θ = (2/3) * M * R^2 ∧ 
  is_moment_of_inertia Θ M R := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_spherical_shell_moment_of_inertia_l281_28154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l281_28113

/-- The measure of the smaller angle formed by the hour-hand and minute-hand of a clock at a given time -/
noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_angle : ℝ := (hours % 12 : ℝ) * 30 + (minutes : ℝ) * 0.5
  let minute_angle : ℝ := (minutes : ℝ) * 6
  min (abs (hour_angle - minute_angle)) (360 - abs (hour_angle - minute_angle))

/-- Theorem stating that the measure of the smaller angle at 7:15 is 127.5° -/
theorem clock_angle_at_7_15 : clock_angle 7 15 = 127.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l281_28113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l281_28129

/-- The function f(x) = sin(2x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

/-- The period of f is π -/
noncomputable def period : ℝ := Real.pi

/-- The smallest positive period of f(x) = sin(2x + π/3) is π -/
theorem smallest_positive_period_of_f :
  ∀ (x : ℝ), f (x + period) = f x ∧
  ∀ (p : ℝ), 0 < p → p < period → ∃ (x : ℝ), f (x + p) ≠ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l281_28129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_theorem_l281_28112

/-- Parabola with equation x² = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- Circle with center (a, b) and radius 3/2 -/
def Circle (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - a)^2 + (p.2 - b)^2 = 9/4}

/-- Line with slope m passing through (0, √2) -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = m * p.1 + Real.sqrt 2}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (2, 0)

/-- Helper function to calculate the area of a triangle formed by a line and a parabola -/
noncomputable def area_triangle (l : Set (ℝ × ℝ)) (p : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Implementation details omitted

/-- Theorem stating the minimum area of triangle PAB and the equation of line l -/
theorem parabola_circle_tangent_theorem
  (a b : ℝ)
  (h1 : (a, b) ∈ Parabola)  -- Center of circle is on the parabola
  (h2 : (0, 0) ∈ Circle a b)  -- Circle passes through origin
  (h3 : ∃ (x : ℝ), (x, -1/4) ∈ Circle a b)  -- Circle is tangent to directrix y = -1/4
  :
  (∃ (m : ℝ),
    (∀ (m' : ℝ), area_triangle (Line m) Parabola ≤ area_triangle (Line m') Parabola) ∧
    (Line m = Line (-Real.sqrt 6 / 3)) ∧
    (area_triangle (Line m) Parabola = 9 * Real.sqrt 3 / 4)) :=
by
  sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_theorem_l281_28112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l281_28104

-- Define the parabola and circle
def parabola (x y : ℝ) : Prop := y^2 = 2*x
def circle_eq (x y : ℝ) : Prop := (x-3)^2 + y^2 = 8

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ circle_eq x₁ y₁ ∧
  parabola x₂ y₂ ∧ circle_eq x₂ y₂ ∧
  x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0 ∧
  y₁ < y₂

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Theorem statement
theorem intersection_line_equation (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points x₁ y₁ x₂ y₂ →
  ∀ x y, (∃ t : ℝ, x = (1-t)*x₁ + t*x₂ ∧ y = (1-t)*y₁ + t*y₂) →
  line_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l281_28104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l281_28124

/-- The time (in seconds) it takes for a train to cross a platform -/
noncomputable def time_to_cross (train_length platform_length : ℝ) (train_speed : ℝ) : ℝ :=
  (train_length + platform_length) / (train_speed * 1000 / 3600)

/-- Theorem stating the time it takes for the train to cross the platform -/
theorem train_crossing_time :
  let train_length := (470 : ℝ)
  let platform_length := (520 : ℝ)
  let train_speed := (55 : ℝ)
  abs (time_to_cross train_length platform_length train_speed - 64.8) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l281_28124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_problem_l281_28188

-- Define α as noncomputable
noncomputable def α : ℝ := (-1 + Real.sqrt 29) / 2

-- Define the set of coin denominations
def coin_denominations : Set ℝ := {1} ∪ {α^k | k : ℕ}

-- Function to represent a coin combination
def coin_combination := ℕ → ℕ

-- Predicate to check if a combination is valid (at most 6 of each denomination)
def is_valid_combination (c : coin_combination) : Prop :=
  ∀ k, c k ≤ 6

-- Function to calculate the sum of a combination
noncomputable def combination_sum (c : coin_combination) : ℝ :=
  c 0 + ∑' k, c (k + 1) * α^k

-- Theorem statement
theorem coin_problem :
  ∀ n : ℕ, ∃ c : coin_combination,
    is_valid_combination c ∧ combination_sum c = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_problem_l281_28188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_integral_equation_l281_28139

open MeasureTheory Interval Real Set

theorem no_solution_for_integral_equation (α : ℝ) (h : 1/2 < α) :
  ¬∃ f : Icc 0 1 → ℝ, ∀ x : ℝ, x ∈ Icc 0 1 →
    f ⟨x, by sorry⟩ = 1 + α * ∫ t in x..1, f ⟨t, by sorry⟩ * f ⟨t - x, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_integral_equation_l281_28139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_g_l281_28193

/-- A function that is periodic with period 24 -/
noncomputable def f : ℝ → ℝ := sorry

/-- The property that f is periodic with period 24 -/
axiom f_periodic (x : ℝ) : f x = f (x - 24)

/-- The function g defined in terms of f -/
noncomputable def g (x : ℝ) : ℝ := f (x / 4)

/-- The theorem stating that 96 is the smallest positive period of g -/
theorem smallest_period_of_g :
  ∀ b : ℝ, b > 0 → (∀ x : ℝ, g (x - b) = g x) → b ≥ 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_g_l281_28193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_increase_l281_28132

/-- Given a circle with radius R, when the central angle increases by 1°,
    the corresponding increase in arc length is πR/180 -/
theorem arc_length_increase (R : ℝ) (h : R > 0) :
  (π * R) / 180 = (2 * π * R * 1) / 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_increase_l281_28132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_angle_is_60_degrees_l281_28144

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 5)^2 + (y - 1)^2 = 2

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define a point on y = x
structure PointOnLine where
  x : ℝ
  y : ℝ
  on_line : line_y_eq_x x y

-- Define tangent lines
def is_tangent_line (l : ℝ → ℝ → Prop) (p : PointOnLine) : Prop :=
  ∃ (x y : ℝ), l x y ∧ circle_eq x y ∧ 
  ∀ (x' y' : ℝ), l x' y' → ¬(circle_eq x' y') ∨ (x' = x ∧ y' = y)

-- Define symmetry about y = x
def symmetric_about_y_eq_x (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), l₁ x y ↔ l₂ y x

-- Placeholder for angle between two lines
noncomputable def angle_between (l₁ l₂ : ℝ → ℝ → Prop) : ℝ := sorry

-- Main theorem
theorem tangent_lines_angle_is_60_degrees 
  (p : PointOnLine) 
  (l₁ l₂ : ℝ → ℝ → Prop) 
  (h₁ : is_tangent_line l₁ p) 
  (h₂ : is_tangent_line l₂ p) 
  (h_sym : symmetric_about_y_eq_x l₁ l₂) : 
  angle_between l₁ l₂ = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_angle_is_60_degrees_l281_28144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l281_28108

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l281_28108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l281_28119

-- Define the properties of function f
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- Define the theorem
theorem range_of_x (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f 1} = Set.Ioo 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l281_28119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l281_28109

noncomputable def f (x : ℝ) : ℝ := |x - 1/2| + |x + 1/2|

def M : Set ℝ := {x | f x < 2}

theorem f_properties :
  (M = Set.Ioo (-1 : ℝ) 1) ∧
  (∀ (a b : ℝ), a ∈ M → b ∈ M → |a + b| < |1 + a * b|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l281_28109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l281_28118

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / x - 1

theorem f_max_value (x : ℝ) (h : x ≤ -2) :
  f x ≤ -11/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l281_28118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l281_28197

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x-1)
def domain_f_x_minus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_f_2x_minus_1 (h : ∀ x, f (x - 1) ∈ domain_f_x_minus_1 ↔ x ∈ domain_f_x_minus_1) :
  ∀ x, f (2*x - 1) ∈ domain_f_x_minus_1 ↔ x ∈ Set.Icc (-1) (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l281_28197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_divides_area_in_half_l281_28131

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the L-shaped region -/
def LShapedRegion : List Point := [
  ⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩
]

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- Calculates the total area of the L-shaped region -/
def totalArea : ℝ := 22

/-- Checks if a line through the origin with given slope divides the area in half -/
def dividesAreaInHalf (slope : ℝ) : Prop :=
  ∃ (intersectionPoint : Point),
    intersectionPoint.y = slope * intersectionPoint.x ∧
    2 * (triangleArea ⟨0, 0⟩ ⟨0, 4⟩ intersectionPoint + 6) = totalArea

/-- The main theorem stating that the slope 3/8 divides the area in half -/
theorem slope_divides_area_in_half :
  dividesAreaInHalf (3/8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_divides_area_in_half_l281_28131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_1994_is_110_l281_28180

/-- Model for bird population growth --/
structure BirdPopulation where
  /-- Constant of proportionality --/
  k : ℝ
  /-- Population function --/
  pop : ℕ → ℝ

/-- Properties of the bird population model --/
def ValidBirdPopulation (bp : BirdPopulation) : Prop :=
  ∀ n, bp.pop (n + 2) - bp.pop n = bp.k * bp.pop (n + 1)

/-- Theorem: Given the conditions, the population in 1994 is 110 --/
theorem population_1994_is_110 :
  ∃ bp : BirdPopulation,
    ValidBirdPopulation bp ∧
    bp.pop 1992 = 50 ∧
    bp.pop 1993 = 80 ∧
    bp.pop 1995 = 162 ∧
    bp.pop 1994 = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_1994_is_110_l281_28180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l281_28128

noncomputable def f (a b x : ℝ) : ℝ := (3 * x + b) / (a^2 + 4)

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f a b x = -f a b (-x)) →
  (f a b 1 = 3/5) →
  (a = 1 ∨ a = -1) ∧ (b = 0) ∧
  (∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → f a b x < f a b y) ∧
  (∀ m, m ∈ Set.Ioo (Real.sqrt 2 - 1) 1 →
    f a b (2*m - 2) + f a b (m^2 + 1) > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l281_28128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_max_inequality_l281_28176

/-- Given complex numbers z₁, z₂, z₃ satisfying certain conditions, 
    prove that the maximum of specific expressions is at least 2016. -/
theorem complex_max_inequality (z₁ z₂ z₃ : ℂ) : 
  ({Complex.abs (z₁ + z₂ + z₃), Complex.abs (-z₁ + z₂ + z₃), 
    Complex.abs (z₁ - z₂ + z₃), Complex.abs (z₁ + z₂ - z₃)} : Set ℝ) = {98, 84, 42, 28} →
  max 
    (Complex.abs (z₁^2 * (2 - z₂^2) - z₃^2))
    (max 
      (Complex.abs (z₂^2 * (2 - z₃^2) - z₁^2))
      (Complex.abs (2 * z₃^2 * (z₁^2 + 1) - z₂^2))) ≥ 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_max_inequality_l281_28176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l281_28153

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^3) + f (y^3) = (x + y) * f (x^2) + f (y^2) - f (x*y)) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l281_28153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l281_28178

/-- Represents a square on the board -/
structure Square where
  row : Nat
  col : Nat
  isBlack : Bool

/-- Represents the triangular board -/
def Board := List (List Square)

/-- Checks if a move is valid according to the rules -/
def isValidMove (board : Board) (fromSquare toSquare : Square) (stepNumber : Nat) : Bool :=
  sorry

/-- Counts the number of valid 7-step paths from P to Q -/
def countValidPaths (board : Board) (p q : Square) : Nat :=
  sorry

/-- The main theorem stating that there are 56 valid paths -/
theorem valid_paths_count (board : Board) (p q : Square) :
  p.row = 0 ∧ p.isBlack = true ∧ q.row = 6 ∧ q.isBlack = true →
  countValidPaths board p q = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l281_28178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_equal_implies_diff_less_one_ceiling_sum_le_sum_ceiling_l281_28106

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Property ②
theorem ceiling_equal_implies_diff_less_one (x₁ x₂ : ℝ) :
  ceiling x₁ = ceiling x₂ → x₁ - x₂ < 1 :=
by sorry

-- Property ③
theorem ceiling_sum_le_sum_ceiling (x₁ x₂ : ℝ) :
  ceiling (x₁ + x₂) ≤ ceiling x₁ + ceiling x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_equal_implies_diff_less_one_ceiling_sum_le_sum_ceiling_l281_28106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_exchange_theorem_l281_28167

/-- Represents a girl in the envelope exchange problem -/
structure Girl :=
  (sent : ℕ)
  (received : ℕ)

/-- The envelope exchange problem -/
theorem envelope_exchange_theorem 
  (girls : Finset Girl) 
  (h_count : girls.card = 10)
  (h_initial : ∀ g, g ∈ girls → g.sent + g.received = 10)
  (h_unique_final : ∀ g₁ g₂, g₁ ∈ girls → g₂ ∈ girls → g₁ ≠ g₂ → g₁.received ≠ g₂.received)
  (h_conservation : (girls.sum (λ g => g.sent)) = (girls.sum (λ g => g.received))) :
  ∃ g ∈ girls, g.received > g.sent :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_exchange_theorem_l281_28167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l281_28121

noncomputable section

def m (x : Real) : Real × Real := (Real.sin x, -1)
def n (x : Real) : Real × Real := (Real.sqrt 3 * Real.cos x, -1/2)

def f (x : Real) : Real := (m x + n x).fst * (m x).fst + (m x + n x).snd * (m x).snd

def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧  -- A is acute
  a = 2 * Real.sqrt 3 ∧
  c = 4 ∧
  ∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ f A

theorem problem_solution :
  ∀ (A B C : Real) (a b c : Real),
  triangle_ABC A B C a b c →
  (∀ (k : Int), ∀ x ∈ Set.Icc (k * Real.pi + Real.pi/3) (k * Real.pi + 5*Real.pi/6), 
    ∀ y ∈ Set.Icc (k * Real.pi + Real.pi/3) (k * Real.pi + 5*Real.pi/6), 
    x ≤ y → f y ≤ f x) ∧
  A = Real.pi/3 ∧
  b = 2 ∧
  1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l281_28121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_divisible_by_seven_l281_28149

/-- The base of the number system -/
def base : ℕ := 15

/-- The number of digits in the given number -/
def num_digits : ℕ := 14

/-- The function that generates the coefficients of the number -/
def coefficient (n : ℕ) : ℕ :=
  if n ≤ 9 then 9 - n
  else 14 - n

/-- The number in base 15 -/
def number : ℕ :=
  Finset.sum (Finset.range num_digits) (fun k => coefficient k * base ^ k)

/-- The theorem stating that the number is divisible by 7 -/
theorem number_divisible_by_seven : 7 ∣ number := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_divisible_by_seven_l281_28149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_exist_and_unique_l281_28198

/-- A line that is simultaneously tangent to two curves -/
structure TangentLine where
  slope : ℝ
  intercept : ℝ

/-- The two curves to which the line is tangent -/
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp (x - 2)
noncomputable def curve2 (x : ℝ) : ℝ := Real.log x

/-- A function to check if a line is tangent to a curve at a given point -/
def is_tangent (line : TangentLine) (curve : ℝ → ℝ) (point : ℝ × ℝ) : Prop :=
  let (x, y) := point
  y = curve x ∧ line.slope = (deriv curve x)

/-- The main theorem stating the existence and uniqueness of the two tangent lines -/
theorem tangent_lines_exist_and_unique :
  ∃! (l1 l2 : TangentLine),
    (∃ (x1 : ℝ), is_tangent l1 curve1 (x1, curve1 x1)) ∧
    (∃ (x2 : ℝ), is_tangent l1 curve2 (x2, curve2 x2)) ∧
    (∃ (x3 : ℝ), is_tangent l2 curve1 (x3, curve1 x3)) ∧
    (∃ (x4 : ℝ), is_tangent l2 curve2 (x4, curve2 x4)) ∧
    l1 = { slope := 1, intercept := -1 } ∧
    l2 = { slope := 1 / Real.exp 1, intercept := 0 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_exist_and_unique_l281_28198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperback_ratio_l281_28116

/-- Represents the collection of books owned by Thabo -/
structure BookCollection where
  paperback_fiction : ℕ
  paperback_nonfiction : ℕ
  hardcover_nonfiction : ℕ

/-- The conditions of Thabo's book collection -/
def thabo_collection : BookCollection :=
  let hardcover_nonfiction := 25
  let paperback_nonfiction := hardcover_nonfiction + 20
  let paperback_fiction := 160 - paperback_nonfiction - hardcover_nonfiction
  ⟨paperback_fiction, paperback_nonfiction, hardcover_nonfiction⟩

/-- The theorem stating the ratio of paperback fiction to paperback nonfiction books -/
theorem paperback_ratio : 
  (thabo_collection.paperback_fiction : ℚ) / thabo_collection.paperback_nonfiction = 2 := by
  -- Unfold the definition of thabo_collection
  unfold thabo_collection
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

#check paperback_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperback_ratio_l281_28116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l281_28135

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 + 1

theorem f_range :
  Set.Icc (-2 : ℝ) (Real.sqrt 3) = Set.image f (Set.Icc (-π/3) (π/4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l281_28135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_triangle_area_l281_28133

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B = Real.pi/3 ∧ Real.cos t.A = 4/5 ∧ t.b = Real.sqrt 3

-- Theorem for sin C
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.C = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry

-- Theorem for area of triangle ABC
theorem triangle_area (t : Triangle) (h : triangle_conditions t) :
  (1/2) * t.a * t.b * Real.sin t.C = (9 * Real.sqrt 3 + 36) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_triangle_area_l281_28133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_value_l281_28175

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

theorem common_tangent_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = g x₀ ∧ 
    (deriv (f a)) x₀ = (deriv g) x₀) →
  a = Real.exp 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_value_l281_28175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_formula_l281_28136

/-- The area of a regular octagon inscribed in a circle with radius r -/
noncomputable def regular_octagon_area (r : ℝ) : ℝ := 2 * r^2 * Real.sqrt 2

/-- Theorem: The area of a regular octagon inscribed in a circle with radius r is 2r²√2 -/
theorem regular_octagon_area_formula (r : ℝ) (h : r > 0) :
  regular_octagon_area r = 2 * r^2 * Real.sqrt 2 := by
  -- Proof goes here
  sorry

#check regular_octagon_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_formula_l281_28136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_bounds_in_range_l281_28140

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Statement for the smallest positive period
theorem smallest_positive_period : ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
  (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧ p = Real.pi := by sorry

-- Statement for the range when 0 < x ≤ π/3
theorem range_in_interval : 
  ∀ y ∈ Set.range (fun x => f x), 2 ≤ y ∧ y ≤ 3 := by sorry

-- Statement that 2 and 3 are in the range
theorem bounds_in_range : 
  ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ ≤ Real.pi / 3 ∧ 0 < x₂ ∧ x₂ ≤ Real.pi / 3 ∧ 
  f x₁ = 2 ∧ f x₂ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_bounds_in_range_l281_28140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_relatively_prime_integers_l281_28173

theorem sum_of_relatively_prime_integers (a b : ℕ+) : 
  Nat.Coprime a b → 
  (a : ℝ) / b = ∑' n, ((2 * n - 1 : ℝ) / 2^n + (2 * n : ℝ) / 3^n) → 
  a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_relatively_prime_integers_l281_28173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_concentration_is_34_375_percent_l281_28183

noncomputable def vessel1_capacity : ℝ := 3
noncomputable def vessel1_alcohol_percentage : ℝ := 25
noncomputable def vessel2_capacity : ℝ := 5
noncomputable def vessel2_alcohol_percentage : ℝ := 40
noncomputable def total_liquid : ℝ := 8
noncomputable def final_vessel_capacity : ℝ := 10

noncomputable def alcohol_in_vessel1 : ℝ := vessel1_capacity * (vessel1_alcohol_percentage / 100)
noncomputable def alcohol_in_vessel2 : ℝ := vessel2_capacity * (vessel2_alcohol_percentage / 100)
noncomputable def total_alcohol : ℝ := alcohol_in_vessel1 + alcohol_in_vessel2

theorem new_concentration_is_34_375_percent :
  (total_alcohol / total_liquid) * 100 = 34.375 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_concentration_is_34_375_percent_l281_28183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l281_28162

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The first term of the geometric sequence -/
def a₁ : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q : ℝ := sorry

/-- Theorem: If S₈ = 2 and S₂₄ = 14, then S₂₀₁₆ = 2^253 - 2 -/
theorem geometric_sequence_sum :
  S 8 = 2 →
  S 24 = 14 →
  S 2016 = 2^253 - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l281_28162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l281_28142

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + 2*x else Real.log (x + 1)

-- State the theorem
theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, |f x| ≥ a * x) → a ∈ Set.Icc (-2) 0 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l281_28142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l281_28151

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 10 cm, 
    and a height of 10 cm, is 150 square centimeters -/
theorem trapezium_area_example : trapeziumArea 20 10 10 = 150 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l281_28151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l281_28170

theorem union_cardinality : 
  let A : Finset ℕ := {4, 5, 6}
  let B : Finset ℕ := {2, 3, 4}
  Finset.card (A ∪ B) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l281_28170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_radius_proof_l281_28158

/-- The equation of the cookie's boundary -/
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6*x + 24*y

/-- The radius of the cookie -/
noncomputable def cookie_radius : ℝ := Real.sqrt 117

theorem cookie_radius_proof :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_boundary x y ↔ (x - h)^2 + (y - k)^2 = cookie_radius^2 :=
by
  -- Introduce the center coordinates
  let h := 3
  let k := 12
  
  -- Prove the existence of h and k
  use h, k
  
  -- Prove the equivalence for all x and y
  intro x y
  
  -- Expand the definitions and simplify
  simp [cookie_boundary, cookie_radius]
  
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_radius_proof_l281_28158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_deviation_10_gt_100_l281_28194

/-- The number of coin tosses in the first experiment -/
def n₁ : ℕ := 10

/-- The number of coin tosses in the second experiment -/
def n₂ : ℕ := 100

/-- The probability of heads for a fair coin -/
noncomputable def p : ℝ := 1/2

/-- The absolute deviation for n coin tosses -/
noncomputable def abs_deviation (n : ℕ) (m : ℕ) : ℝ :=
  |m / n - p|

/-- The expected value of the absolute deviation for n coin tosses -/
noncomputable def E_abs_deviation (n : ℕ) : ℝ :=
  sorry

/-- The theorem stating that the expected absolute deviation for 10 tosses
    is greater than for 100 tosses -/
theorem abs_deviation_10_gt_100 :
  E_abs_deviation n₁ > E_abs_deviation n₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_deviation_10_gt_100_l281_28194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tower_height_l281_28134

-- Define the radius of the base sphere
def base_radius : ℝ := 100

-- Define the structure of the tower
structure Tower where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  h₁ : 0 < r₁ ∧ r₁ ≤ base_radius
  h₂ : 0 < r₂ ∧ r₂ ≤ r₁
  h₃ : 0 < r₃ ∧ r₃ ≤ r₂

-- Define the height function for the tower
noncomputable def tower_height (t : Tower) : ℝ :=
  base_radius + (base_radius^2 - t.r₁^2).sqrt + 
  (t.r₁^2 - t.r₂^2).sqrt + (t.r₂^2 - t.r₃^2).sqrt + t.r₃

-- Theorem statement
theorem max_tower_height :
  ∀ t : Tower, tower_height t ≤ 300 :=
by
  sorry

#check max_tower_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tower_height_l281_28134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l281_28123

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1/2 then x/3
  else if 1/2 < x ∧ x < 1 then 2*x^3/(x+1)
  else 0  -- undefined outside [0,1]

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := a*x - a/2 + 3

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a > 0)
  (h2 : ∀ (x₁ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ 1 →
    ∃ (x₂ : ℝ), 0 ≤ x₂ ∧ x₂ ≤ 1/2 ∧ f x₁ = g a x₂) :
  a ≥ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l281_28123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_to_arrive_at_5pm_l281_28160

/-- Represents the time of day in hours and minutes -/
structure Time where
  hours : Int
  minutes : Int
  valid : 0 ≤ minutes ∧ minutes < 60 := by sorry

/-- Calculates the difference between two times in hours -/
noncomputable def timeDiff (t1 t2 : Time) : ℝ :=
  (t2.hours - t1.hours : ℝ) + (t2.minutes - t1.minutes : ℝ) / 60

/-- Represents Cátia's bicycle journey -/
structure Journey where
  departureTime : Time
  speed : ℝ
  distance : ℝ

theorem speed_to_arrive_at_5pm (j1 j2 : Journey) (h1 : j1.speed = 20)
    (h2 : j1.departureTime = ⟨15, 45, by sorry⟩)
    (h3 : j2.speed = 10)
    (h4 : timeDiff j1.departureTime ⟨16, 30, by sorry⟩ = j1.distance / j1.speed)
    (h5 : timeDiff j1.departureTime ⟨17, 15, by sorry⟩ = j2.distance / j2.speed)
    (h6 : j1.distance = j2.distance) :
    ∃ j3 : Journey, j3.speed = 12 ∧ 
    timeDiff j1.departureTime ⟨17, 0, by sorry⟩ = j3.distance / j3.speed ∧
    j3.distance = j1.distance := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_to_arrive_at_5pm_l281_28160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l281_28122

-- Define the function f(x) = x ln(1+x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log (1 + x)

-- State the theorem
theorem f_properties :
  -- f(x) is defined on (-1, +∞)
  (∀ x : ℝ, x > -1 → f x ∈ Set.univ) ∧
  -- f(x) is monotonically increasing in (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- The derivative of f(x) at x = 0 is 0
  (deriv f 0 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l281_28122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_factorial_mod_seventeen_l281_28130

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => i + 1)

/-- Theorem: 13! ≡ 9 (mod 17) -/
theorem thirteen_factorial_mod_seventeen : 
  factorial 13 % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_factorial_mod_seventeen_l281_28130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_height_is_1700_l281_28105

/-- The temperature decrease rate per 100m elevation increase -/
noncomputable def temp_decrease_rate : ℝ := 0.7

/-- The temperature at the summit of the mountain -/
noncomputable def summit_temp : ℝ := 14.1

/-- The temperature at the base of the mountain -/
noncomputable def base_temp : ℝ := 26

/-- The height of the mountain relative to its base -/
noncomputable def mountain_height : ℝ := (base_temp - summit_temp) / temp_decrease_rate * 100

theorem mountain_height_is_1700 : mountain_height = 1700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_height_is_1700_l281_28105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l281_28138

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x - 2 * Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  Set.range (f ∘ (fun x => x * π - π / 12)) = Set.Icc (-1) 2 := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l281_28138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_p_plus_q_equals_one_l281_28141

/-- The probability of making an even number of errors in a six-step process -/
noncomputable def q (p : ℝ) : ℝ := (1 + (2*p - 1)^6) / 2

/-- The theorem stating that there are exactly two values of p where p + q(p) = 1 -/
theorem two_solutions_for_p_plus_q_equals_one :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ p ∈ s, 0 ≤ p ∧ p ≤ 1 ∧ p + q p = 1) ∧
  (∀ p, 0 ≤ p ∧ p ≤ 1 ∧ p + q p = 1 → p ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_p_plus_q_equals_one_l281_28141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l281_28111

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  (cos B) / b + (cos C) / c = (Real.sqrt 3 * sin A) / (3 * sin C) →
  cos B + Real.sqrt 3 * sin B = 2 →
  b = Real.sqrt 3 ∧
  (∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 / 4 ∧
    ∀ (a' c' : ℝ), a' > 0 → c' > 0 →
      a' * c' * sin B / 2 ≤ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l281_28111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expr_l281_28115

-- Define the function
def f (x : ℝ) : ℝ := -2 * x + 8

-- Define the point M
def M : ℝ × ℝ → Prop := λ p => p.2 = f p.1

-- Define the range of x₁
def x_range : Set ℝ := Set.Icc 2 5

-- Define the expression (y₁+1)/(x₁+1)
noncomputable def expr (p : ℝ × ℝ) : ℝ := (p.2 + 1) / (p.1 + 1)

-- Theorem statement
theorem range_of_expr :
  ∀ p : ℝ × ℝ, M p → p.1 ∈ x_range → 
  expr p ∈ Set.Icc (-1/6) (5/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expr_l281_28115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l281_28177

noncomputable def seriesSum (x : ℝ) : ℝ := ∑' n, n * x^n

theorem infinite_series_sum :
  seriesSum (1/20) = 20/361 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l281_28177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_transport_equation_l281_28103

/-- The amount of goods (in kg) that machine A transports per hour -/
def x : ℝ := sorry

/-- The amount of goods (in kg) that machine B transports per hour -/
noncomputable def y : ℝ := x + 60

/-- The time it takes for machine A to transport 500 kg -/
noncomputable def time_A : ℝ := 500 / x

/-- The time it takes for machine B to transport 800 kg -/
noncomputable def time_B : ℝ := 800 / y

theorem machine_transport_equation : 
  500 / x = 800 / (x + 60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_transport_equation_l281_28103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_approx_l281_28189

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure Floor where
  breadth : ℝ
  length : ℝ
  paintCost : ℝ
  paintRate : ℝ

/-- The length is 200% more than the breadth, and the painting cost is 484 at a rate of 3 per square meter. -/
def floorConditions (f : Floor) : Prop :=
  f.length = 3 * f.breadth ∧
  f.paintCost = 484 ∧
  f.paintRate = 3

/-- The theorem stating the length of the floor given the conditions. -/
theorem floor_length_approx (f : Floor) (h : floorConditions f) :
  ∃ ε > 0, |f.length - 21.99| < ε := by
  sorry

#check floor_length_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_approx_l281_28189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_increase_after_price_decrease_l281_28157

/-- Proves that a 25% price decrease with a sales increase ratio of 4 results in a 50% revenue increase -/
theorem revenue_increase_after_price_decrease (P U : ℝ) (hP : P > 0) (hU : U > 0) : 
  let price_decrease_percent : ℝ := 25
  let sales_increase_ratio : ℝ := 4
  let new_price : ℝ := P * (1 - price_decrease_percent / 100)
  let sales_increase_percent : ℝ := price_decrease_percent * sales_increase_ratio
  let new_units : ℝ := U * (1 + sales_increase_percent / 100)
  let original_revenue : ℝ := P * U
  let new_revenue : ℝ := new_price * new_units
  new_revenue = 1.5 * original_revenue :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_increase_after_price_decrease_l281_28157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_new_alloy_l281_28187

/-- Represents an alloy with its total weight and composition ratios -/
structure Alloy where
  weight : ℚ
  ratio1 : ℚ
  ratio2 : ℚ

/-- Calculates the amount of the second component in an alloy -/
def secondComponentAmount (a : Alloy) : ℚ :=
  (a.ratio2 / (a.ratio1 + a.ratio2)) * a.weight

/-- Theorem: The amount of tin in the new alloy is 44 kg -/
theorem tin_amount_in_new_alloy (alloyA alloyB : Alloy)
  (hA : alloyA = { weight := 60, ratio1 := 3, ratio2 := 2 })
  (hB : alloyB = { weight := 100, ratio1 := 1, ratio2 := 4 }) :
  secondComponentAmount alloyA + secondComponentAmount alloyB = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_new_alloy_l281_28187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l281_28125

noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then x^2 + 2^x - 1
  else if x < 0 then -((-x)^2 + 2^(-x) - 1)
  else 0

theorem solution_set_of_inequality (h_odd : ∀ x, f (-x) = -f x) :
  {x : ℝ | f x + 7 < 0} = Set.Ioi (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l281_28125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lengths_lower_bound_l281_28147

/-- A structure representing a line segment -/
structure Segment where
  start : ℝ
  finish : ℝ

/-- The set M is the union of k pairwise disjoint segments on a line -/
def M (k : ℕ) : Set Segment := sorry

/-- The property that any segment of length ≤ 1 can be positioned on the line with endpoints in M -/
def can_position (M : Set Segment) : Prop :=
  ∀ s : Segment, s.finish - s.start ≤ 1 → ∃ t : ℝ, ({ start := t + s.start, finish := t + s.finish } : Segment) ∈ M

/-- The sum of lengths of segments in M -/
noncomputable def sum_lengths (M : Set Segment) : ℝ := sorry

/-- Theorem stating that the sum of lengths of segments in M is at least 1/k -/
theorem sum_lengths_lower_bound (k : ℕ) (h : k > 0) :
  can_position (M k) → sum_lengths (M k) ≥ 1 / k := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lengths_lower_bound_l281_28147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_max_ab_value_l281_28110

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + Real.exp (x - 1)

-- Theorem 1
theorem f_has_one_zero :
  ∃! x, f (-1) 1 x = 0 :=
sorry

-- Theorem 2
theorem max_ab_value (a b : ℝ) (h : a ≠ 0) :
  (∀ x, f a b x ≤ Real.exp (x - 1) + x + 1) →
  a * b ≤ (1 / 2) * Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_max_ab_value_l281_28110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l281_28165

/-- Given a class with boys and girls, prove that the ratio of boys to girls is 6/5 -/
theorem boys_to_girls_ratio (a b : ℚ) (ha : a > 0) (hb : b > 0) : 
  (75.5 * a + 81 * b) / (a + b) = 78 →
  a / b = 6/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l281_28165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zach_filling_time_l281_28146

/-- The time Zach spent filling water balloons --/
def zach_time : ℕ → ℕ := λ _ => 40

/-- The total number of water balloons --/
def total_balloons : ℕ := 170

/-- The number of water balloons that popped --/
def popped_balloons : ℕ := 10

/-- Max's rate of filling water balloons (balloons per minute) --/
def max_rate : ℕ := 2

/-- Zach's rate of filling water balloons (balloons per minute) --/
def zach_rate : ℕ := 3

/-- The time Max spent filling water balloons (in minutes) --/
def max_time : ℕ := 30

theorem zach_filling_time :
  zach_time total_balloons = 40 :=
by
  sorry

#eval zach_time total_balloons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zach_filling_time_l281_28146
