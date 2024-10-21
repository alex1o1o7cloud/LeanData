import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_10_factorial_greater_than_9_factorial_l102_10258

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem divisors_of_10_factorial_greater_than_9_factorial : 
  (Finset.filter (fun d => d > factorial 9 ∧ (factorial 10) % d = 0) (Finset.range (factorial 10 + 1))).card = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_10_factorial_greater_than_9_factorial_l102_10258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sin_less_f_cos_l102_10298

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the angles α and β
def α : ℝ := sorry
def β : ℝ := sorry

-- Axioms based on the given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, x ≤ y → y ≤ 0 → f y ≤ f x
axiom acute_angles : 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2
axiom triangle_inequality : α + β > Real.pi/2

-- Theorem to prove
theorem f_sin_less_f_cos : f (Real.sin α) < f (Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sin_less_f_cos_l102_10298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_I_l102_10278

-- Define the function f(x) = 4x - x^4
def f (x : ℝ) : ℝ := 4 * x - x^4

-- Define the interval [-1, 2]
def I : Set ℝ := Set.Icc (-1) 2

theorem max_min_f_on_I :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f b ≤ f x) ∧
  a = 1 ∧ b = 2 := by
  sorry

#check max_min_f_on_I

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_I_l102_10278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_space_length_is_twelve_l102_10204

/-- The length of an ad space that satisfies the given conditions -/
noncomputable def ad_space_length : ℝ :=
  let num_companies : ℕ := 3
  let ads_per_company : ℕ := 10
  let ad_width : ℝ := 5
  let cost_per_sqft : ℝ := 60
  let total_cost : ℝ := 108000
  total_cost / (num_companies * ads_per_company * ad_width * cost_per_sqft)

/-- Theorem stating that the ad space length is 12 feet -/
theorem ad_space_length_is_twelve : ad_space_length = 12 := by
  -- Unfold the definition of ad_space_length
  unfold ad_space_length
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_space_length_is_twelve_l102_10204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_l102_10230

/-- Given f(x) = x^2 and g(x) is a polynomial such that f(g(x)) = 9x^2 - 6x + 1,
    prove that g(x) = 3x - 1 or g(x) = -3x + 1 -/
theorem polynomial_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2) →
  (∃ n : ℕ, ∀ x, ∃ p : Polynomial ℝ, g x = p.eval x ∧ p.degree ≤ n) →
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_composition_l102_10230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_grid_division_l102_10241

theorem square_grid_division (m n k : ℕ) (h : m^2 = n * k) :
  ∃ (d m₁ n₁ : ℕ), 
    m = m₁ * d ∧ 
    n = n₁ * d ∧ 
    Nat.Coprime m₁ n₁ ∧ 
    m₁ ∣ k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_grid_division_l102_10241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l102_10291

def M₀ : ℝ × ℝ × ℝ := (-2, 3, 5)
def M₁ : ℝ × ℝ × ℝ := (-1, 2, 4)
def M₂ : ℝ × ℝ × ℝ := (-1, -2, -4)
def M₃ : ℝ × ℝ × ℝ := (3, 0, -1)

def plane_equation (x y z : ℝ) : ℝ := x - 8*y + 4*z + 1

theorem distance_to_plane :
  let d := |plane_equation M₀.1 M₀.2.1 M₀.2.2| / Real.sqrt (1^2 + (-8)^2 + 4^2)
  d = 5/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l102_10291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l102_10266

theorem angle_sum_theorem (w x y z k : ℝ) : 
  w > 0 → x > 0 → y > 0 → z > 0 →
  (Real.cos w) * (Real.cos x) * (Real.cos y) * (Real.cos z) ≠ 0 →
  w + x + y + z = 2 * Real.pi →
  3 * (Real.tan w) = k * (1 + 1 / (Real.cos w)) →
  4 * (Real.tan x) = k * (1 + 1 / (Real.cos x)) →
  5 * (Real.tan y) = k * (1 + 1 / (Real.cos y)) →
  6 * (Real.tan z) = k * (1 + 1 / (Real.cos z)) →
  k = Real.sqrt 19 := by
  sorry

#check angle_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l102_10266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l102_10248

/-- The circle O -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The ellipse E -/
def ellipse_E (x y : ℝ) : Prop := y^2/3 + x^2/2 = 1

/-- The slope of a tangent line from a point (x₀, y₀) to the ellipse -/
def tangent_slope (x₀ y₀ k : ℝ) : Prop :=
  (2 - x₀^2) * k^2 + 2 * k * x₀ * y₀ - (y₀^2 - 3) = 0

theorem tangent_slopes_product (x₀ y₀ k₁ k₂ : ℝ) :
  circle_O x₀ y₀ →
  tangent_slope x₀ y₀ k₁ →
  tangent_slope x₀ y₀ k₂ →
  k₁ * k₂ = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l102_10248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_arc_length_l102_10254

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 2 * x

-- Define the circle P
noncomputable def circle_P (x y : ℝ) : Prop :=
  (x - Real.sqrt 2)^2 + y^2 = 36

-- Define the point M
noncomputable def point_M : ℝ × ℝ := (-Real.sqrt 2, 1)

-- Theorem statement
theorem min_arc_length :
  ∃ (r : ℝ),
    (∀ x y, ellipse x y → r = 6) ∧
    (∀ x y, parabola x y → circle_P x y) →
    (∃ θ : ℝ, 0 < θ ∧ θ < π ∧
      (∀ l : ℝ → ℝ → Prop, 
        (l (point_M.1) (point_M.2) → 
          ∃ arc_length, arc_length = r * θ ∧
          (∀ other_arc_length, other_arc_length ≥ arc_length) ∧
          arc_length = 4 * π))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_arc_length_l102_10254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_poly_satisfies_conditions_l102_10246

theorem no_integer_poly_satisfies_conditions : 
  ¬∃ (f : Polynomial ℤ), (Polynomial.eval 7 f = 11) ∧ (Polynomial.eval 11 f = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_poly_satisfies_conditions_l102_10246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l102_10244

open Real

-- Define the triangle ABC
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the main theorem
theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c) 
  (h_acute : A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2)
  (h_relation : c - b = 2 * b * Real.cos A) :
  A = 2 * B ∧ 
  Real.sqrt 2 < a / b ∧ a / b < Real.sqrt 3 ∧
  5 * Real.sqrt 3 / 3 < 1 / Real.tan B - 1 / Real.tan A + 2 * Real.sin A ∧ 
  1 / Real.tan B - 1 / Real.tan A + 2 * Real.sin A < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l102_10244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_is_four_l102_10283

-- Define the wall dimensions
noncomputable def wall_width : ℝ := sorry
noncomputable def wall_height : ℝ := 6 * wall_width
noncomputable def wall_length : ℝ := 7 * wall_height

-- Define the volume of the wall
noncomputable def wall_volume : ℝ := wall_width * wall_height * wall_length

-- Theorem statement
theorem wall_width_is_four :
  wall_volume = 16128 → wall_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_is_four_l102_10283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_internal_point_l102_10294

open Real

theorem area_of_triangle_with_internal_point (S₁ S₂ S₃ S : ℝ) 
  (h_pos : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0) :
  S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃)^2 :=
by
  -- We assume the result is true based on the geometric reasoning provided
  -- A formal proof would require more advanced geometry in Lean
  sorry

#check area_of_triangle_with_internal_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_internal_point_l102_10294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l102_10219

def data : List Int := [27, 28, 29, 29, 29, 30, 30, 30, 31, 31, 42, 43, 45, 46, 48, 51, 51, 51, 52, 53, 61, 64, 65, 68, 69]

def mode (l : List Int) : Int := sorry

def median (l : List Int) : Int := sorry

theorem median_mode_difference :
  let m := median data
  let d := mode data
  abs (m - d) = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l102_10219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l102_10238

/-- Given a magnification factor and the diameter of a magnified image,
    calculate the actual diameter of the object. -/
noncomputable def actualDiameter (magnificationFactor : ℝ) (magnifiedDiameter : ℝ) : ℝ :=
  magnifiedDiameter / magnificationFactor

/-- Theorem stating that for a magnification factor of 1000 and a magnified image
    diameter of 0.3 cm, the actual diameter is 0.0003 cm. -/
theorem tissue_diameter_calculation :
  actualDiameter 1000 0.3 = 0.0003 := by
  -- Unfold the definition of actualDiameter
  unfold actualDiameter
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l102_10238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_regular_triangular_pyramid_l102_10276

/-- The area of a cross-section in a regular triangular pyramid -/
theorem cross_section_area_regular_triangular_pyramid 
  (H : ℝ) -- height of the pyramid
  (α : ℝ) -- angle between lateral face and base
  (h1 : H > 0) -- height is positive
  (h2 : 0 < α ∧ α < π / 2) -- angle is between 0 and π/2
  : 
  ∃ (A : ℝ), -- area of the cross-section
    A = (1 / 2) * H^2 * Real.sqrt 3 * Real.tan (π/2 - α) * Real.sqrt (1 + 16 * (Real.tan (π/2 - α))^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_regular_triangular_pyramid_l102_10276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_with_nines_l102_10277

def count_nines_in_last_k_digits (n : ℕ) (k : ℕ) : ℕ :=
  (String.toList (toString n)).reverse.take k
    |> List.filter (· = '9')
    |> List.length

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, ∃ m : ℕ, 
    (2^n % 10^k = m) ∧ 
    (count_nines_in_last_k_digits m k ≥ k / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_with_nines_l102_10277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l102_10288

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.cos C = -(1/8 : ℝ) →
  Real.sin B = (2/3 : ℝ) * Real.sin C →
  c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l102_10288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equals_power_l102_10284

theorem sqrt_sum_equals_power : Real.sqrt 2016 + Real.sqrt 56 = (14 : ℝ)^(3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equals_power_l102_10284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l102_10211

-- Define the function f(x) = 3^x + 3x - 8
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3) + 3*x - 8

-- State the theorem
theorem root_in_interval :
  Continuous f →
  (∀ x ∈ Set.Ioo 1 2, f x ∈ Set.univ) →
  f 2 > 0 →
  f 1.5 < 0 →
  f 1.75 > 0 →
  ∃ x ∈ Set.Ioo 1.5 1.75, f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l102_10211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_from_point_on_terminal_side_l102_10263

theorem trig_values_from_point_on_terminal_side (θ : Real) (x : Real) 
  (h1 : x ≠ 0)
  (h2 : ∃ P : Real × Real, P = (x, -2) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ)
  (h3 : Real.cos θ = x / 3) :
  Real.sin θ = -2/3 ∧ Real.tan θ = 2 * Real.sqrt 5 / 5 ∨ Real.tan θ = -2 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_from_point_on_terminal_side_l102_10263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l102_10208

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

noncomputable def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < a + 1}

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2*x + Real.pi/3) + 1

theorem problem_solution :
  (∃ S : Set ℝ, S = {a : ℝ | ∀ x : ℝ, x ∈ B a → x ∈ A} ∧ S = Set.Ici 0) ∧
  (∀ a : ℝ, ∃ T : Set ℝ, T = {x₀ : ℝ | f a x₀ ∈ A} ∧
    T = ⋃ (k : ℤ), {x₀ : ℝ | k*Real.pi - Real.pi/4 < x₀ ∧ x₀ < k*Real.pi - Real.pi/12} ∪
                   {x₀ : ℝ | k*Real.pi + Real.pi/4 < x₀ ∧ x₀ < k*Real.pi + 5*Real.pi/12}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l102_10208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_l102_10297

/-- The radius of a circular cookie made from scrap dough -/
theorem scrap_cookie_radius 
  (square_side : ℝ)
  (large_cookie_radius : ℝ)
  (small_cookie_radius : ℝ)
  (num_large_cookies : ℝ)
  (num_small_cookies : ℝ)
  (h1 : square_side = 6)
  (h2 : large_cookie_radius = 1)
  (h3 : small_cookie_radius = 0.5)
  (h4 : num_large_cookies = 4)
  (h5 : num_small_cookies = 5) :
  Real.sqrt (square_side^2 - π * (num_large_cookies * large_cookie_radius^2 + num_small_cookies * small_cookie_radius^2)) = Real.sqrt 30.75 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_l102_10297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_approx_10_53_l102_10237

/-- Represents the frequency distribution of grades in Ms. Rivera's history class -/
def grade_distribution : List (String × Nat) := [
  ("90% - 100%", 6),
  ("80% - 89%", 7),
  ("70% - 79%", 10),
  ("60% - 69%", 8),
  ("50% - 59%", 4),
  ("Below 50%", 3)
]

/-- Calculates the total number of students in the class -/
def total_students : Nat :=
  grade_distribution.foldr (fun (_, count) acc => count + acc) 0

/-- Calculates the number of students in the 50%-59% range -/
def students_in_range : Nat :=
  (grade_distribution.filter (fun (range, _) => range == "50% - 59%")).head!.2

/-- Calculates the percentage of students in the 50%-59% range -/
noncomputable def percentage_in_range : Float :=
  (students_in_range.toFloat / total_students.toFloat) * 100

/-- Theorem stating that the percentage of students in the 50%-59% range is approximately 10.53% -/
theorem percentage_approx_10_53 :
  (Float.round (percentage_in_range * 100) / 100) = 10.53 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_approx_10_53_l102_10237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l102_10240

noncomputable section

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define a line
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the right vertex of the hyperbola
def right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

-- Main theorem
theorem hyperbola_intersection_theorem 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (p q : ℝ × ℝ) (hp : hyperbola a b p.1 p.2) (hq : hyperbola a b q.1 q.2)
  (hpq : p ≠ q) (hnv : p ≠ right_vertex a ∧ q ≠ right_vertex a)
  (hm : (right_vertex a).1 - (p.1 + q.1) / 2 = (q.1 - p.1) / 4) :
  (∃ k m : ℝ, line k m p.1 p.2 ∧ line k m q.1 q.2) →
  (let k₁ := (q.2 - p.2) / (q.1 - p.1);
   let k₂ := ((p.2 + q.2) / 2) / ((p.1 + q.1) / 2);
   k₁ * k₂ = b^2 / a^2) ∧
  (∃ x : ℝ, x = a * (a^2 + b^2) / (a^2 - b^2) ∧
   line ((q.2 - p.2) / (q.1 - p.1)) (q.2 - ((q.2 - p.2) / (q.1 - p.1)) * q.1) x 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l102_10240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l102_10218

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem root_in_interval :
  ∃ x : ℝ, x > (1/4 : ℝ) ∧ x < (1/2 : ℝ) ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l102_10218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_large_chips_proof_l102_10253

/-- The maximum number of large chips in a box of 72 chips, where the number of small chips
    exceeds the number of large chips by a prime number. -/
def max_large_chips : ℕ := 35

/-- Proof that the maximum number of large chips is 35 -/
theorem max_large_chips_proof :
  let total_chips : ℕ := 72
  let large_chips : ℕ → ℕ := λ p => (total_chips - p) / 2
  let small_chips : ℕ → ℕ := λ p => total_chips - large_chips p
  (∃ p : ℕ, Nat.Prime p ∧ 
    small_chips p = large_chips p + p ∧
    ∀ q : ℕ, Nat.Prime q → large_chips q ≤ large_chips p) ∧
  max_large_chips = 35 := by
  sorry

#eval max_large_chips

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_large_chips_proof_l102_10253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l102_10255

theorem complex_multiplication :
  (2 + 2 * Complex.I) * (1 - 2 * Complex.I) = 6 - 2 * Complex.I := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l102_10255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_race_distance_l102_10299

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  headStart : ℝ

/-- The race setup -/
structure RaceSetup where
  b : Runner
  a : Runner
  c : Runner
  hb : b.speed = 1
  ha : a.speed = 4 * b.speed
  hc : c.speed = 2 * b.speed
  haa : a.headStart = -69
  hcc : c.headStart = -25

/-- The finish time for a runner given the race distance -/
noncomputable def finishTime (runner : Runner) (distance : ℝ) : ℝ :=
  (distance - runner.headStart) / runner.speed

/-- The theorem stating the optimal race distance -/
theorem optimal_race_distance (setup : RaceSetup) :
  ∃ (d : ℝ), d > 0 ∧
    finishTime setup.a d = finishTime setup.b d ∧
    finishTime setup.b d = finishTime setup.c d ∧
    d = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_race_distance_l102_10299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_fifth_term_l102_10243

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_fifth_term (a₁ d : ℝ) :
  a₁ = 2 →
  arithmetic_sum a₁ d 3 = 12 →
  arithmetic_sequence a₁ d 5 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_fifth_term_l102_10243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l102_10274

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (cos (8*x) + 3*cos (4*x) + 3*cos (2*x) = 8*cos x * (cos (3*x))^3 - 0.5) ↔ 
  (∃ k : ℤ, x = (π * (6*k + 1)) / 30 ∨ x = (π * (6*k - 1)) / 30) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l102_10274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l102_10268

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem max_value_of_y :
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧
  y x = 13 ∧
  ∀ (z : ℝ), 1 ≤ z ∧ z ≤ 9 → y z ≤ y x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l102_10268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_proof_l102_10272

/-- The radius of a semicircle sharing a base with an isosceles triangle -/
noncomputable def semicircle_radius (total_perimeter : ℝ) (triangle_side : ℝ) : ℝ :=
  94 / (Real.pi + 4)

theorem semicircle_radius_proof 
  (total_perimeter : ℝ) 
  (triangle_side : ℝ) 
  (h1 : total_perimeter = 162)
  (h2 : triangle_side = 34) :
  semicircle_radius total_perimeter triangle_side = 94 / (Real.pi + 4) := by
  -- Unfold the definition of semicircle_radius
  unfold semicircle_radius
  -- The left-hand side is now exactly equal to the right-hand side
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_proof_l102_10272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_AB_AC_is_zero_l102_10245

/-- The cosine of the angle between vectors AB and AC is 0, given points A(-2, 1, 1), B(2, 3, -2), and C(0, 0, 3) -/
theorem cosine_angle_AB_AC_is_zero :
  let A : Fin 3 → ℝ := ![(-2), 1, 1]
  let B : Fin 3 → ℝ := ![2, 3, (-2)]
  let C : Fin 3 → ℝ := ![0, 0, 3]
  let AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
  let AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]
  let dot_product := (AB 0 * AC 0) + (AB 1 * AC 1) + (AB 2 * AC 2)
  let magnitude_AB := Real.sqrt ((AB 0)^2 + (AB 1)^2 + (AB 2)^2)
  let magnitude_AC := Real.sqrt ((AC 0)^2 + (AC 1)^2 + (AC 2)^2)
  dot_product / (magnitude_AB * magnitude_AC) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_AB_AC_is_zero_l102_10245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_derivatives_correct_l102_10286

-- Define the functions
noncomputable def f1 (x : ℝ) := Real.cos x
noncomputable def f2 (x : ℝ) := -1 / Real.sqrt x
noncomputable def f3 (x : ℝ) := 1 / x^2
def f4 : ℝ → ℝ := Function.const ℝ 3

-- State the theorem
theorem all_derivatives_correct :
  (∀ x, deriv f1 x = -Real.sin x) ∧
  (∀ x, x > 0 → deriv f2 x = 1 / (2 * x * Real.sqrt x)) ∧
  (deriv f3 3 = -2 / 27) ∧
  (∀ x, deriv f4 x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_derivatives_correct_l102_10286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_equals_100x_implies_x_equals_50_over_101_l102_10290

theorem average_equals_100x_implies_x_equals_50_over_101 :
  let n : ℕ := 99
  let sum_1_to_99 : ℕ := n * (n + 1) / 2
  ∀ x : ℚ, (sum_1_to_99 + x) / 100 = 100 * x → x = 50 / 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_equals_100x_implies_x_equals_50_over_101_l102_10290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_from_hexagons_l102_10295

/-- The area of an equilateral triangle formed by connecting the centers of three adjacent regular hexagons with side length 1 is 3√3. -/
theorem area_triangle_from_hexagons (side_length : ℝ) (h_side : side_length = 1) :
  let center_distance := 2 * side_length
  let triangle_side := center_distance
  let semiperimeter := (3 * triangle_side) / 2
  let inradius := side_length
  semiperimeter * inradius = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_from_hexagons_l102_10295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_proof_l102_10262

theorem trigonometric_identity_proof (t : ℝ) (k m n : ℕ+) :
  (1 + 2 * Real.sin t) * (1 + 2 * Real.cos t) = 9/4 →
  (1 - 2 * Real.sin t) * (1 - 2 * Real.cos t) = m/n - Real.sqrt k →
  Nat.Coprime m.val n.val →
  k = 11 ∧ m = 27 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_proof_l102_10262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_equals_210_l102_10257

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  second_term : a 2 = 7
  fourth_term : a 4 = 15

/-- Sum of the first n terms of an arithmetic sequence. -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem to prove. -/
theorem sum_10_equals_210 (seq : ArithmeticSequence) : sum_n seq 10 = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_equals_210_l102_10257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l102_10201

-- Define the sign function
noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1
  else if a = 0 then 0
  else -1

-- Define the conditions for x, y, and z
def condition (t : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := t
  x = 3000 - 3001 * sign (y + z + 3) ∧
  y = 3000 - 3001 * sign (x + z + 3) ∧
  z = 3000 - 3001 * sign (x + y + 3)

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 3 ∧ ∀ t, t ∈ s ↔ condition t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l102_10201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_price_is_250_l102_10222

/-- The price Hendricks paid for the guitar -/
noncomputable def hendricks_price : ℝ := 200

/-- The percentage discount Hendricks received compared to Gerald's price -/
noncomputable def discount_percentage : ℝ := 20

/-- Gerald's price for the guitar -/
noncomputable def gerald_price : ℝ := hendricks_price / (1 - discount_percentage / 100)

/-- Theorem stating that Gerald's price is $250 given the conditions -/
theorem gerald_price_is_250 : gerald_price = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_price_is_250_l102_10222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l102_10231

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (2 * Real.pi * x)

-- State the theorem
theorem solutions_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, -1 ≤ x ∧ x ≤ 1) ∧
                    (∀ x ∈ S, g (g (g x)) = g x) ∧
                    (∀ x, -1 ≤ x ∧ x ≤ 1 ∧ g (g (g x)) = g x → x ∈ S) ∧
                    (Finset.card S = 68) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l102_10231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_sum_l102_10203

-- Define an equilateral triangle ABC with side length 1
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1

-- Define points D, E, F on sides BC, CA, AB respectively
def PointsOnSides (A B C D E F : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 0 ≤ t₃ ∧ t₃ ≤ 1 ∧
  D = (1 - t₁) • B + t₁ • C ∧
  E = (1 - t₂) • C + t₂ • A ∧
  F = (1 - t₃) • A + t₃ • B

-- Define the ratio condition
def RatioCondition (D E F : ℝ × ℝ) : Prop :=
  dist D E / 20 = dist E F / 22 ∧ dist E F / 22 = dist F D / 38

-- Define points X, Y, Z on lines BC, CA, AB respectively
def PointsOnLines (A B C X Y Z : ℝ × ℝ) : Prop :=
  ∃ s₁ s₂ s₃ : ℝ,
  X = (1 - s₁) • B + s₁ • C ∧
  Y = (1 - s₂) • C + s₂ • A ∧
  Z = (1 - s₃) • A + s₃ • B

-- Define perpendicularity conditions
def PerpendicularityConditions (D E F X Y Z : ℝ × ℝ) : Prop :=
  ((X.1 - Y.1) * (D.1 - E.1) + (X.2 - Y.2) * (D.2 - E.2) = 0) ∧
  ((Y.1 - Z.1) * (E.1 - F.1) + (Y.2 - Z.2) * (E.2 - F.2) = 0) ∧
  ((Z.1 - X.1) * (F.1 - D.1) + (Z.2 - X.2) * (F.2 - D.2) = 0)

-- Define the area of a triangle
noncomputable def TriangleArea (P Q R : ℝ × ℝ) : ℝ :=
  abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2

-- The main theorem
theorem triangle_areas_sum (A B C D E F X Y Z : ℝ × ℝ) :
  Triangle A B C →
  PointsOnSides A B C D E F →
  RatioCondition D E F →
  PointsOnLines A B C X Y Z →
  PerpendicularityConditions D E F X Y Z →
  (1 / TriangleArea D E F + 1 / TriangleArea X Y Z) = (97 * Real.sqrt 2 + 40 * Real.sqrt 3) / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_sum_l102_10203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l102_10261

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) : ℝ := Real.cos (2/3 * (x - Real.pi/4) - Real.pi/4)

theorem g_monotone_increasing :
  ∀ φ : ℝ, -Real.pi/2 < φ ∧ φ < 0 →
  (∀ x : ℝ, f x φ = f (Real.pi/4 - x) φ) →
  (∀ x y : ℝ, -Real.pi/2 ≤ x ∧ x < y ∧ y ≤ Real.pi/2 → g x < g y) :=
by
  sorry

#check g_monotone_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l102_10261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l102_10232

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def satisfiesCondition (t : Triangle) : Prop :=
  t.a * (Real.cos t.B) = t.b * (Real.cos t.A)

-- Define isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Theorem statement
theorem triangle_shape (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesCondition t) : 
  isIsosceles t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l102_10232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equation_solutions_l102_10215

theorem complex_modulus_equation_solutions :
  ∃! (s : Finset ℝ), (∀ d ∈ s, Complex.abs (1/3 - d * Complex.I) = 2/3) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equation_solutions_l102_10215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l102_10281

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + (Real.log x - 1) / (Real.log a) + 1

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l102_10281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_value_l102_10216

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 - 3*x) / (5*x - 1)

-- State the theorem
theorem unreachable_value :
  ∀ x : ℝ, x ≠ (1/5) → f x ≠ (-3/5) := by
  -- Proof goes here
  sorry

-- Example usage (optional)
#check unreachable_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_value_l102_10216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l102_10225

-- Define the curves
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ :=
  (3 + 2 * Real.cos θ, 4 + 2 * Real.sin θ)

noncomputable def C₂ (φ : ℝ) : ℝ × ℝ :=
  (Real.cos φ, Real.sin φ)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_distance_between_curves :
  ∃ (M : ℝ), M = 8 ∧ 
  (∀ (θ φ : ℝ), distance (C₁ θ) (C₂ φ) ≤ M) ∧
  (∃ (θ₀ φ₀ : ℝ), distance (C₁ θ₀) (C₂ φ₀) = M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l102_10225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_center_l102_10289

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the line
def my_line (x y a : ℝ) : Prop := 2*x - y + a = 0

-- Define the center of the circle
def my_center : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem chord_through_center (a : ℝ) :
  my_line my_center.1 my_center.2 a → a = -1 :=
by
  intro h
  -- Proof steps would go here
  sorry

#check chord_through_center

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_center_l102_10289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_is_sixteen_l102_10214

noncomputable def average (numbers : List ℝ) : ℝ := (numbers.sum) / numbers.length

theorem fourth_number_is_sixteen
  (numbers : List ℝ)
  (h1 : numbers.length = 4)
  (h2 : average numbers = 20)
  (h3 : 3 ∈ numbers)
  (h4 : 33 ∈ numbers)
  (h5 : 28 ∈ numbers)
  : ∃ x, x ∈ numbers ∧ x = 16 := by
  sorry

#check fourth_number_is_sixteen

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_is_sixteen_l102_10214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l102_10252

theorem profit_percentage_calculation (cost_price marked_price : ℝ) 
  (h1 : cost_price = 47.50)
  (h2 : marked_price = 65)
  (h3 : marked_price > cost_price) : 
  (((0.95 * marked_price - cost_price) / cost_price) * 100 = 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l102_10252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_in_interval_l102_10271

noncomputable def f (x : Real) : Real := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem max_value_f_in_interval :
  ∃ (x : Real), π/4 ≤ x ∧ x ≤ π/2 ∧
  f x = 3/2 ∧
  ∀ (y : Real), π/4 ≤ y ∧ y ≤ π/2 → f y ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_in_interval_l102_10271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l102_10264

noncomputable section

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line l
def l (k m x y : ℝ) : Prop := y = k*x + m

-- Define a point on the ellipse
def on_ellipse (x y : ℝ) : Prop := C x y

-- Define the intersection points A and B
def intersection_points (k m x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l k m x₁ y₁ ∧ l k m x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the condition OA + OB = λOQ
def vector_condition (x₁ y₁ x₂ y₂ xq yq lambda : ℝ) : Prop :=
  x₁ + x₂ = lambda * xq ∧ y₁ + y₂ = lambda * yq

-- Define the area of triangle ABO
def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  abs (x₁ * y₂ - x₂ * y₁) / 2

theorem ellipse_intersection_theorem :
  ∀ (k m x₁ y₁ x₂ y₂ xq yq lambda : ℝ),
    intersection_points k m x₁ y₁ x₂ y₂ →
    on_ellipse xq yq →
    vector_condition x₁ y₁ x₂ y₂ xq yq lambda →
    (-2 < lambda ∧ lambda < 2) ∧
    (triangle_area x₁ y₁ x₂ y₂ ≤ Real.sqrt 2 / 2) ∧
    (triangle_area x₁ y₁ x₂ y₂ = Real.sqrt 2 / 2 ↔ lambda = Real.sqrt 2 ∨ lambda = -Real.sqrt 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l102_10264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_produced_from_methane_and_oxygen_l102_10296

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℕ

instance : HMul ℕ Moles Moles where
  hMul n m := ⟨n * m.value⟩

instance : OfNat Moles n where
  ofNat := ⟨n⟩

/-- Represents the chemical reaction CH₄ + 2O₂ → CO₂ + 2H₂O -/
def methane_oxygen_reaction (methane oxygen : Moles) : Moles :=
  2 * methane

/-- Theorem stating that 3 moles of Methane and 6 moles of Oxygen produce 6 moles of Water -/
theorem water_produced_from_methane_and_oxygen :
  let methane : Moles := 3
  let oxygen : Moles := 6
  methane_oxygen_reaction methane oxygen = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_produced_from_methane_and_oxygen_l102_10296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l102_10205

-- Define the function f(x) = x + 1/x
noncomputable def f (x : ℝ) := x + 1/x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo (-2 : ℝ) 0 ∧
  (∀ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 0 → f x ≤ f c) ∧
  f c = -2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l102_10205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_largest_smallest_l102_10292

def array : List (List Nat) := [
  [12, 7, 9, 5, 6],
  [14, 8, 16, 14, 10],
  [10, 4, 9, 7, 11],
  [15, 5, 18, 13, 3],
  [9, 3, 6, 11, 4]
]

def is_largest_in_column (n : Nat) (col : Nat) : Prop :=
  ∀ row, row < array.length → n ≥ (array.get! row).get! col

def is_smallest_in_row (n : Nat) (row : Nat) : Prop :=
  ∀ col, col < (array.get! row).length → n ≤ (array.get! row).get! col

theorem unique_largest_smallest :
  ∃! n, ∃ row col, 
    (array.get! row).get! col = n ∧
    is_largest_in_column n col ∧
    is_smallest_in_row n row ∧
    n = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_largest_smallest_l102_10292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l102_10256

noncomputable def f (x : ℝ) := Real.sin (2 * x) - Real.cos (2 * x)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧
  (∀ x, f (π/4 + x) = f (π/4 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l102_10256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l102_10206

-- Define the function f as noncomputable due to its dependency on Real.log
noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

-- State the theorem
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, f x a = f (-x) a) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l102_10206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l102_10235

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a ^ 2 + h.b ^ 2)

/-- Checks if a point lies on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y / x = h.b / h.a

theorem hyperbola_equation (h : Hyperbola) :
  focal_length h = 5 →
  on_asymptote h 2 1 →
  h.a ^ 2 = 20 ∧ h.b ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l102_10235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l102_10259

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x else Real.log (x + 1)

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, f (2 - x^2) > f x ↔ -2 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l102_10259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_sum_is_20_l102_10234

-- Define the points
def point1 : ℝ × ℝ := (1, 7)
def point2 : ℝ × ℝ := (13, 16)
def point3 (k : ℤ) : ℝ × ℝ := (5, k)

-- Define the function to calculate the area of a triangle
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem min_area_sum_is_20 :
  ∃ (k1 k2 : ℤ), k1 ≠ k2 ∧
  (∀ (k : ℤ), triangleArea point1 point2 (point3 k) ≥ triangleArea point1 point2 (point3 k1)) ∧
  (∀ (k : ℤ), triangleArea point1 point2 (point3 k) ≥ triangleArea point1 point2 (point3 k2)) ∧
  k1 + k2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_sum_is_20_l102_10234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_is_one_l102_10273

theorem root_difference_is_one (p : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - p*x + (p^2 - 1)/4
  let roots := { x : ℝ | f x = 0 }
  ∃ (r s : ℝ), r ∈ roots ∧ s ∈ roots ∧ r ≥ s ∧ r - s = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_is_one_l102_10273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_theorem_l102_10242

theorem positive_difference_theorem : |((8^2 + 8^2) / 8 : ℚ) - ((8^2 * 8^2) / 8 : ℚ)| = 496 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_theorem_l102_10242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_A_star_B_l102_10202

def A : Finset ℕ := {1, 3, 5, 7}
def B : Finset ℕ := {2, 3, 5}

def star_operation (X Y : Finset ℕ) : Finset ℕ := X \ Y

theorem number_of_subsets_of_A_star_B : 
  Finset.card (Finset.powerset (star_operation A B)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_A_star_B_l102_10202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_monotonicity_l102_10287

theorem binomial_expansion_monotonicity 
  (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (∀ k : ℕ, k < n → Nat.choose n k * a^(n-k) * b^k > Nat.choose n (k+1) * a^(n-(k+1)) * b^(k+1)) ↔ 
    a > n * b ∧ 
  ((∀ k : ℕ, k < n → Nat.choose n k * a^(n-k) * b^k < Nat.choose n (k+1) * a^(n-(k+1)) * b^(k+1)) ↔ 
    a < b / n) := by
  sorry

#check binomial_expansion_monotonicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_monotonicity_l102_10287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l102_10251

-- Define the function h
noncomputable def h : ℝ → ℝ := sorry

-- Define the properties of h
axiom h_property : h (-1.5) = 3 ∧ h 3.5 = 3

-- Define the uniqueness of the points
axiom unique_points : ∀ x y : ℝ, x - y = 5 ∧ h x = h y → (x = 3.5 ∧ y = -1.5) ∨ (x = -1.5 ∧ y = 3.5)

-- Theorem statement
theorem intersection_point_sum :
  ∃ x y : ℝ, h x = h (x - 5) ∧ x + y = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l102_10251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l102_10236

/-- Given a triangle ABC with point D on the extension of BC and E the midpoint of AD,
    prove that if BC = 2CD and AE = lambda*AB + (3/4)AC, then lambda = -1/4 -/
theorem triangle_vector_relation (A B C D E : EuclideanSpace ℝ (Fin 2)) (lambda : ℝ) :
  (∃ t : ℝ, D = B + t • (C - B) ∧ t > 1) →  -- D is on the extension line of BC
  C - B = 2 • (D - C) →  -- BC = 2CD
  E = A + (1/2) • (D - A) →  -- E is midpoint of AD
  E - A = lambda • (B - A) + (3/4) • (C - A) →  -- AE = lambda*AB + (3/4)AC
  lambda = -1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l102_10236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l102_10213

-- Define the solid T
def T : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p; |x| + |y| + |z| ≤ 2}

-- State the theorem
theorem volume_of_T : MeasureTheory.volume T = 32 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l102_10213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_lock_combinations_l102_10249

theorem suitcase_lock_combinations : Fintype.card {s : Fin 4 → Fin 10 | Function.Injective s} = 5040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_lock_combinations_l102_10249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_quantity_is_10_l102_10293

/-- Represents a mixture of pure ghee and vanaspati -/
structure GheeMixture where
  total : ℝ
  pure_ghee : ℝ
  vanaspati : ℝ

/-- The original mixture before adding pure ghee -/
def original_mixture (x : ℝ) : GheeMixture :=
  { total := x
  , pure_ghee := 0.6 * x
  , vanaspati := 0.4 * x }

/-- The amount of pure ghee added to the mixture -/
def added_pure_ghee : ℝ := 10

/-- The final mixture after adding pure ghee -/
def final_mixture (x : ℝ) : GheeMixture :=
  { total := x + added_pure_ghee
  , pure_ghee := 0.6 * x + added_pure_ghee
  , vanaspati := 0.4 * x }

/-- The theorem stating the original quantity of the mixture -/
theorem original_quantity_is_10 :
  ∃ x, (final_mixture x).vanaspati / (final_mixture x).total = 0.2 ∧ x = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_quantity_is_10_l102_10293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_construction_and_intersection_l102_10228

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space using a point and a direction vector -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Define membership for Point3D in Line3D -/
def Point3D.mem_line (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk 
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

instance : Membership Point3D Line3D where
  mem := Point3D.mem_line

/-- Define membership for Point3D in Sphere -/
def Point3D.mem_sphere (p : Point3D) (s : Sphere) : Prop :=
  (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 = s.radius^2

instance : Membership Point3D Sphere where
  mem := Point3D.mem_sphere

/-- Theorem: Given two points and a line, there exists a sphere with its center on the line
    that passes through both points, and the line intersects the sphere at two points -/
theorem sphere_construction_and_intersection 
  (A B : Point3D) (L : Line3D) :
  ∃ (S : Sphere) (P1 P2 : Point3D),
    (S.center ∈ L) ∧ 
    (A ∈ S) ∧ 
    (B ∈ S) ∧
    (P1 ∈ L) ∧ 
    (P2 ∈ L) ∧
    (P1 ∈ S) ∧ 
    (P2 ∈ S) ∧
    (P1 ≠ P2) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_construction_and_intersection_l102_10228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_distance_ratio_l102_10275

theorem incenter_distance_ratio (a b c : ℝ) (A : ℝ) (r : ℝ) :
  let p := (a + b + c) / 2
  let d₁ := r / Real.sin (A / 2)
  d₁^2 / ((p - a) / a) = 2 * r * a / Real.sin A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_distance_ratio_l102_10275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latus_rectum_parabola_l102_10239

/-- Predicate to check if a point (x, y) is on the latus rectum of the parabola x^2 = -y -/
def IsLatusRectum (x y : ℝ) : Prop :=
  x^2 = -y ∧ y = 1/4

/-- The equation of the latus rectum for the parabola x^2 = -y is y = 1/4 -/
theorem latus_rectum_parabola :
  ∀ (x y : ℝ), x^2 = -y → (y = 1/4 ↔ IsLatusRectum x y) :=
by
  intros x y h
  constructor
  · intro h_y
    exact ⟨h, h_y⟩
  · intro h_latus
    exact h_latus.2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_latus_rectum_parabola_l102_10239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_of_three_over_twentysix_l102_10217

/-- The decimal representation of 3/26 -/
def decimal_rep : ℚ := 3 / 26

/-- The length of the repeating sequence in the decimal representation of 3/26 -/
def repeat_length : ℕ := 6

/-- The repeating sequence in the decimal representation of 3/26 -/
def repeat_seq : List ℕ := [1, 5, 3, 8, 4, 6]

/-- The position of the digit we're looking for within the repeating sequence -/
def target_position : ℕ := 99 % repeat_length

theorem hundredth_digit_of_three_over_twentysix :
  (repeat_seq.get? target_position).isSome ∧
  (repeat_seq.get? target_position).get! = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_of_three_over_twentysix_l102_10217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l102_10285

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * (x + Real.pi / 6))

theorem axis_of_symmetry :
  ∃ (k : ℤ), f (-Real.pi / 6 + k * Real.pi / 2) = f (-Real.pi / 6 - k * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l102_10285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l102_10267

/-- Triangle PQR with angle bisector PS -/
structure TrianglePQR where
  /-- Side lengths -/
  p : ℝ
  q : ℝ
  r : ℝ
  /-- Segments of side QR created by angle bisector PS -/
  x : ℝ
  y : ℝ
  /-- p is positive -/
  hp_pos : 0 < p
  /-- q is positive -/
  hq_pos : 0 < q
  /-- r is positive -/
  hr_pos : 0 < r
  /-- x is positive -/
  hx_pos : 0 < x
  /-- y is positive -/
  hy_pos : 0 < y
  /-- PS is angle bisector -/
  h_bisector : x / q = y / r
  /-- S is on QR -/
  h_on_side : x + y = p
  /-- Given ratio condition -/
  h_ratio : p / (q + r) = 3 / 5

/-- The main theorem -/
theorem angle_bisector_theorem (t : TrianglePQR) : t.x / t.y = t.q / t.r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l102_10267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_triangular_pyramid_l102_10210

/-- The radius of a sphere circumscribed around a triangular pyramid -/
theorem sphere_radius_triangular_pyramid 
  (d α β : ℝ) 
  (h_d : d > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_β : 0 < β ∧ β < π) : 
  ∃ (R : ℝ), R = (d / (2 * Real.cos (α/2)^2)) * Real.sqrt (Real.cos (α/2)^4 + (1 / Real.tan β)^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_triangular_pyramid_l102_10210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l102_10282

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * Real.cos x - 1) + Real.sqrt (49 - x^2)

def domain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y}

theorem f_domain : domain f = 
  {x : ℝ | (-7 ≤ x ∧ x < -5*Real.pi/3) ∨ 
           (-Real.pi/3 < x ∧ x < Real.pi/3) ∨ 
           (5*Real.pi/3 < x ∧ x ≤ 7)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l102_10282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_seven_l102_10250

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two numbers are consecutive -/
def consecutive (a b : Nat) : Prop := a + 1 = b ∨ b + 1 = a

/-- Check if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Check if the grid satisfies the consecutive number condition -/
def satisfiesConsecutiveCondition (g : Grid) : Prop :=
  ∀ i j k l, consecutive (g i j) (g k l) →
    sharesEdge (i, j) (k, l)

/-- Sum of corner numbers in the grid -/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Check if the grid contains all numbers from 1 to 9 -/
def containsAllNumbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j, g i j = n.val + 1

theorem center_is_seven (g : Grid)
  (h1 : satisfiesConsecutiveCondition g)
  (h2 : cornerSum g = 22)
  (h3 : containsAllNumbers g) :
  g 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_seven_l102_10250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_range_l102_10221

-- Define a geometric sequence
noncomputable def GeometricSequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a₁ * q^(n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def GeometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- Main theorem
theorem geometric_sequence_ratio_range
  (a₁ : ℝ)
  (q : ℝ)
  (h_positive : ∀ n : ℕ, 0 < GeometricSequence a₁ q n)
  (h_sum_inequality : ∀ n : ℕ+, GeometricSum a₁ q (2*n) < 3 * GeometricSum a₁ q n) :
  0 < q ∧ q ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_range_l102_10221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l102_10280

theorem calculate_expressions :
  ((Real.sqrt 5)^2 + Real.sqrt ((-3)^2) - Real.sqrt 18 * Real.sqrt (1/2) = 5) ∧
  ((Real.sqrt 5 - Real.sqrt 2)^2 + (2 + Real.sqrt 3) * (2 - Real.sqrt 3) = 8 - 2 * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l102_10280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_configurations_l102_10229

-- Define the cube configuration
def CubeConfiguration := Fin 8 → Bool

-- Define the group of rotations
def RotationGroup := Fin 24

-- Function to count fixed points under a rotation
def countFixedPoints (r : RotationGroup) : ℕ := sorry

-- Function to calculate the total number of fixed points
def totalFixedPoints : ℕ := sorry

-- The number of distinct configurations
noncomputable def distinctConfigurations : ℚ :=
  (totalFixedPoints : ℚ) / 24

-- Theorem stating the number of distinct configurations
theorem num_distinct_configurations :
  Int.floor distinctConfigurations = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_configurations_l102_10229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l102_10269

-- Definition of property P
def has_property_P (a : ℕ → ℝ) (k n₀ : ℕ) (d : ℝ) :=
  ∀ n ≥ n₀, a (n + k) - a n = d

-- Main theorem
theorem sequence_properties
  (a b c : ℕ → ℝ)  -- Sequences a, b, c
  (i j : ℕ)        -- Natural numbers i, j
  (d₁ d₂ : ℝ)      -- Real numbers d₁, d₂
  (h_P : has_property_P a 3 2 0)  -- a has property P(3,2,0)
  (h_a2 : a 2 = 3)
  (h_a4 : a 4 = 5)
  (h_a678 : a 6 + a 7 + a 8 = 18)
  (h_arith : ∀ n, b (n + 1) - b n = b 2 - b 1)  -- b is arithmetic
  (h_geom : ∃ q > 0, ∀ n, c (n + 1) = q * c n)  -- c is geometric with positive ratio
  (h_b1c3 : b 1 = c 3 ∧ b 1 = 2)
  (h_b3c1 : b 3 = c 1 ∧ b 3 = 8)
  (h_a_def : ∀ n, a n = b n + c n)
  (h_Pi : has_property_P a i 2 d₁)
  (h_Pj : has_property_P a j 2 d₂)
  (h_i_lt_j : i < j)
  (h_coprime : Nat.Coprime i j) :
  (a 3 = 10) ∧
  (¬ ∀ n ≥ 1, a (n + 2) - a n = 0) ∧
  (has_property_P a (j - i) (i + 2) ((j - i : ℝ) / i * d₁)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l102_10269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_CP_CQ_zero_l102_10270

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -x + 4

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the intersection points P and Q
noncomputable axiom P : ℝ × ℝ
noncomputable axiom Q : ℝ × ℝ

-- Assume P and Q are on both the line and the circle
axiom P_on_line : line_l P.1 P.2
axiom P_on_circle : circle_C P.1 P.2
axiom Q_on_line : line_l Q.1 Q.2
axiom Q_on_circle : circle_C Q.1 Q.2

-- Define vectors CP and CQ
noncomputable def CP : ℝ × ℝ := (P.1 - center.1, P.2 - center.2)
noncomputable def CQ : ℝ × ℝ := (Q.1 - center.1, Q.2 - center.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_CP_CQ_zero : dot_product CP CQ = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_CP_CQ_zero_l102_10270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l102_10224

open Real

/-- The series ∑ (n≥0) [n! * (19/7)^n / (n + 1)^n] converges. -/
theorem series_convergence : 
  ∃ (L : ℝ), HasSum (λ n : ℕ ↦ (n.factorial : ℝ) * (19/7)^n / (n + 1 : ℝ)^n) L :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l102_10224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l102_10223

noncomputable def a : ℝ := sorry

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := a^x

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x * Real.log 2 + y - 1 = 0

-- Theorem statement
theorem tangent_line_implies_a_value :
  (∀ x y, y = curve x → x = 0 → tangent_line x y) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l102_10223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_fifth_root_l102_10200

theorem power_of_two_product_fifth_root : (2^10 * 2^15 : ℕ) = 32^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_fifth_root_l102_10200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degenerate_ellipse_b_l102_10227

/-- The equation of a conic section -/
def conic_equation (b : ℝ) (x y : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 6 * y + b = 0

/-- A degenerate ellipse is a single point -/
def is_degenerate_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, f p.1 p.2

/-- The value of b for which the conic is a degenerate ellipse -/
theorem degenerate_ellipse_b : 
  ∃! b : ℝ, is_degenerate_ellipse (conic_equation b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degenerate_ellipse_b_l102_10227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_l102_10220

-- Define ω as a complex number
variable (ω : ℂ)

-- Define the conditions
axiom omega_cube : ω^3 = 1
axiom omega_not_one : ω ≠ 1

-- Define z₁ and z₂ as complex numbers
variable (z₁ z₂ : ℂ)

-- Define the third point
def z₃ (ω z₁ z₂ : ℂ) : ℂ := -ω * z₁ - ω^2 * z₂

-- Define what it means for three complex numbers to form an equilateral triangle
def is_equilateral_triangle (a b c : ℂ) : Prop :=
  Complex.abs (b - a) = Complex.abs (c - b) ∧
  Complex.abs (c - b) = Complex.abs (a - c)

-- Theorem statement
theorem equilateral_triangle (ω z₁ z₂ : ℂ) 
  (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  is_equilateral_triangle z₁ z₂ (z₃ ω z₁ z₂) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_l102_10220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l102_10209

/-- The set of all real numbers except 1 -/
noncomputable def RealStarSet : Set ℝ := {x : ℝ | x ≠ 1}

/-- The function f: ℝ* → ℝ -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2007*x - 6028) / (3*(x - 1))

/-- The theorem stating that f satisfies the functional equation -/
theorem f_satisfies_equation :
  ∀ x ∈ RealStarSet, x + f x + 2 * f ((x + 2009) / (x - 1)) = 2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l102_10209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_assignment_l102_10233

/-- Represents the three individuals --/
inductive Person
| aleshin
| belyaev
| belkin

/-- Represents the three professions --/
inductive Profession
| architect
| accountant
| archaeologist

/-- Represents the three cities --/
inductive City
| belgorod
| bryansk
| astrakhan

/-- Assigns a profession to each person --/
def profession_assignment : Person → Profession := sorry

/-- Assigns a city to each person --/
def city_assignment : Person → City := sorry

/-- Condition 1: Belkin rarely visits Belgorod --/
axiom belkin_not_in_belgorod : city_assignment Person.belkin ≠ City.belgorod

/-- Condition 2: For two individuals, their profession and city start with the same letter as their surname --/
axiom matching_initials :
  ∃ (p1 p2 : Person), p1 ≠ p2 ∧
  ((p1 = Person.aleshin ∧ profession_assignment p1 = Profession.archaeologist ∧ city_assignment p1 = City.astrakhan) ∨
   (p1 = Person.belyaev ∧ profession_assignment p1 = Profession.architect ∧ city_assignment p1 = City.belgorod)) ∧
  ((p2 = Person.aleshin ∧ profession_assignment p2 = Profession.archaeologist ∧ city_assignment p2 = City.astrakhan) ∨
   (p2 = Person.belyaev ∧ profession_assignment p2 = Profession.architect ∧ city_assignment p2 = City.belgorod))

/-- Condition 3: The architect's wife is Belkin's younger sister --/
axiom architect_not_belkin : profession_assignment Person.belkin ≠ Profession.architect

/-- The main theorem: Prove that the assignments are unique and correct --/
theorem unique_assignment :
  (profession_assignment Person.aleshin = Profession.archaeologist ∧ city_assignment Person.aleshin = City.astrakhan) ∧
  (profession_assignment Person.belyaev = Profession.architect ∧ city_assignment Person.belyaev = City.belgorod) ∧
  (profession_assignment Person.belkin = Profession.accountant ∧ city_assignment Person.belkin = City.bryansk) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_assignment_l102_10233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_in_third_quadrant_l102_10212

theorem sine_of_angle_in_third_quadrant (α : Real) : 
  (α > Real.pi ∧ α < 3 * Real.pi / 2) →  -- α is in the third quadrant
  Real.tan α = 3 → 
  Real.sin α = -(3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_in_third_quadrant_l102_10212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l102_10247

/-- The circle C in Cartesian coordinates -/
def circleC (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

/-- The line l in Cartesian coordinates -/
def lineL (x y : ℝ) : Prop := x + y = 2

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 2| / Real.sqrt 2

theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circleC x y ∧ 
  (∀ (x' y' : ℝ), circleC x' y' → distance_to_line x y ≥ distance_to_line x' y') ∧
  distance_to_line x y = (3 * Real.sqrt 2) / 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l102_10247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l102_10207

/-- Represents a parabola in the Cartesian plane -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The focus of a parabola -/
noncomputable def focus (par : Parabola) : Point :=
  ⟨par.p/2, 0⟩

theorem parabola_chord_length 
  (par : Parabola)
  (h_dist : distance (Point.mk 4 (Real.sqrt (8*par.p))) (focus par) = 5)
  : 
  (par.p = 2) ∧
  (∃ A B : Point, 
    par.eq A.x A.y ∧ 
    par.eq B.x B.y ∧ 
    A.y = 2*A.x - 3 ∧ 
    B.y = 2*B.x - 3 ∧
    distance A B = Real.sqrt 35) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l102_10207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_dried_grapes_is_20_l102_10260

-- Define the weight of fresh grapes
noncomputable def fresh_grapes_weight : ℝ := 40

-- Define the water content percentage in fresh grapes
noncomputable def fresh_water_percentage : ℝ := 90

-- Define the weight of dried grapes
noncomputable def dried_grapes_weight : ℝ := 5

-- Define the function to calculate the percentage of water in dried grapes
noncomputable def water_percentage_in_dried_grapes : ℝ :=
  let solid_content := fresh_grapes_weight * (1 - fresh_water_percentage / 100)
  let water_in_dried := dried_grapes_weight - solid_content
  (water_in_dried / dried_grapes_weight) * 100

-- Theorem statement
theorem water_percentage_in_dried_grapes_is_20 :
  water_percentage_in_dried_grapes = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_dried_grapes_is_20_l102_10260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l102_10265

/-- Helper function to represent a point being a tangent point of a set at another point -/
def is_tangent_point (S : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) : Prop := sorry

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Helper function to calculate the area of a quadrilateral given four points -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Given two parabolas C₁ and C₂, prove properties about their intersections and tangent lines. -/
theorem parabola_properties (p : ℝ) (h_p : p > 0) : 
  ∃ (M A B C P Q : ℝ × ℝ),
    let C₁ := {(x, y) : ℝ × ℝ | x^2 = 2*p*y}
    let C₂ := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
    let O := (0, 0)
    -- M is the intersection of C₁ and C₂ (other than O)
    M ∈ C₁ ∧ M ∈ C₂ ∧ M ≠ O
    -- A, B, C are intersections of tangent lines with axes
    ∧ (∃ t : ℝ, A = (t, 0) ∧ is_tangent_point C₁ M A)
    ∧ (∃ t : ℝ, B = (t, 0) ∧ is_tangent_point C₂ M B)
    ∧ (∃ t : ℝ, C = (0, t) ∧ is_tangent_point C₂ M C)
    -- P and Q are intersections of y = x + 1 with C₁
    ∧ P ∈ C₁ ∧ Q ∈ C₁
    ∧ P.2 = P.1 + 1 ∧ Q.2 = Q.1 + 1
    ∧ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 6
    -- Prove the following
    → (P.1 * Q.1 + P.2 * Q.2 = -1)  -- OP · OQ = -1
    ∧ (area_triangle B O C / area_quadrilateral A O C M = 1/2)  -- Area ratio
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l102_10265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l102_10279

/-- Represents a 3x3 grid where each cell can contain an X or be empty -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if a given grid has no more than one X in each small square -/
def valid_placement (g : Grid) : Prop :=
  ∀ i j, g i j → ∀ i' j', g i' j' → (i = i' ∧ j = j') ∨ (i ≠ i' ∨ j ≠ j')

/-- Checks if a given grid has no three X's in a row horizontally -/
def no_horizontal_three (g : Grid) : Prop :=
  ∀ i, ¬(g i 0 ∧ g i 1 ∧ g i 2)

/-- Checks if a given grid has no three X's in a row vertically -/
def no_vertical_three (g : Grid) : Prop :=
  ∀ j, ¬(g 0 j ∧ g 1 j ∧ g 2 j)

/-- Counts the number of X's in a given grid -/
def count_x (g : Grid) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => 
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j => 
      if g i j then 1 else 0))

/-- The main theorem stating that the maximum number of X's that can be placed on a 3x3 grid
    under the given conditions is 5 -/
theorem max_x_placement :
  (∃ g : Grid, valid_placement g ∧ no_horizontal_three g ∧ no_vertical_three g ∧ count_x g = 5) ∧
  (∀ g : Grid, valid_placement g → no_horizontal_three g → no_vertical_three g → count_x g ≤ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_placement_l102_10279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_multiple_of_4_l102_10226

/-- A fair 10-sided die -/
def decahedral_die : Finset ℕ := Finset.range 10

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event.filter (λ x => x ∈ die)).card / die.card

/-- The event of rolling a multiple of 4 on a decahedral die -/
def multiple_of_4_decahedral : Finset ℕ := Finset.filter (λ x => x % 4 = 0) decahedral_die

/-- The event of rolling a 4 on a six-sided die -/
def roll_4_six_sided : Finset ℕ := Finset.filter (λ x => x = 4) six_sided_die

/-- The probability that the product of rolls from a decahedral die and a six-sided die is a multiple of 4 -/
theorem prob_product_multiple_of_4 :
  prob multiple_of_4_decahedral decahedral_die +
  prob roll_4_six_sided six_sided_die * (1 - prob multiple_of_4_decahedral decahedral_die) =
  1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_multiple_of_4_l102_10226
