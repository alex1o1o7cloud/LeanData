import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_alpha_l818_81846

theorem max_tan_alpha (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = Real.sin α / Real.sin β) :
  ∃ (max_tan_α : ℝ), max_tan_α = Real.sqrt 2 / 4 ∧ 
    ∀ θ, 0 < θ ∧ θ < π/2 → Real.tan θ ≤ max_tan_α :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_alpha_l818_81846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_solution_l818_81844

/-- Represents the nested square root function with y levels of nesting -/
noncomputable def A (y : ℕ) (x : ℝ) : ℝ :=
  match y with
  | 0 => x
  | y+1 => Real.sqrt (x + A y x)

/-- The problem statement -/
theorem nested_sqrt_solution :
  ∀ n m : ℤ, A 1964 (n : ℝ) = m → n = 0 ∧ m = 0 :=
by
  sorry

#check nested_sqrt_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_solution_l818_81844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l818_81809

/-- A function f(x) that is monotonically increasing on R -/
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - 2 * x^2 + c * x

/-- The condition that f is monotonically increasing on R -/
def is_monotone (a c : ℝ) : Prop := Monotone (f a c)

/-- The expression we want to minimize -/
noncomputable def expr_to_minimize (a c : ℝ) : ℝ := a / (c^2 + 4) + c / (a^2 + 4)

/-- The theorem stating the minimum value of the expression -/
theorem min_value_theorem (a c : ℝ) (h1 : is_monotone a c) (h2 : a * c ≤ 4) :
  expr_to_minimize a c ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l818_81809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_set_characterization_l818_81853

def is_closed_set (M : Set ℂ) : Prop :=
  ∀ x y, x ∈ M → y ∈ M → x * y ∈ M ∧ x^2 ∈ M

theorem closed_set_characterization (M : Set ℂ) (h : M = {α, β, γ}) :
  is_closed_set M → M = {(-1 : ℂ), 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_set_characterization_l818_81853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_approx_two_years_l818_81869

/-- Calculates the number of years for a compound interest investment --/
noncomputable def calculate_investment_years (P : ℝ) (r : ℝ) (n : ℝ) (CI : ℝ) : ℝ :=
  let A := P + CI
  Real.log (A / P) / (n * Real.log (1 + r / n))

/-- Theorem stating the investment duration is approximately 2 years --/
theorem investment_duration_approx_two_years :
  let P : ℝ := 50000
  let r : ℝ := 0.04
  let n : ℝ := 2
  let CI : ℝ := 4121.608
  let t := calculate_investment_years P r n CI
  Int.floor t = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_approx_two_years_l818_81869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_polynomial_forms_l818_81881

def f (x : ℝ) : ℝ := x^2

theorem h_polynomial_forms 
  (h : ℝ → ℝ) 
  (h_poly : Polynomial ℝ)
  (h_eq : ∀ x, f (h x) = 9*x^2 - 6*x + 1) :
  (∀ x, h x = 3*x - 1) ∨ (∀ x, h x = -3*x + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_polynomial_forms_l818_81881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_triangle_abc_l818_81830

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x)^2, Real.sqrt 3)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.sin (2 * x))

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Theorem for the center of symmetry
theorem center_of_symmetry (k : ℤ) :
  ∃ y, f (k * Real.pi / 2 - Real.pi / 12) = y ∧
       f (-k * Real.pi / 2 + Real.pi / 12) = y :=
by sorry

-- Theorem for triangle ABC
theorem triangle_abc (A B C : ℝ) (a b c : ℝ) :
  f C = 3 →
  c = 1 →
  a * b = 2 * Real.sqrt 3 →
  a > b →
  a = 2 ∧ b = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_triangle_abc_l818_81830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_condition_l818_81823

-- Define the function
noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^(a^2 + 1) + 2 * x + 3

-- State the theorem
theorem quadratic_function_condition (a : ℝ) :
  (∀ x, ∃ p q r : ℝ, f a x = p * x^2 + q * x + r) →
  a = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_condition_l818_81823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_eq_neg200_l818_81892

/-- The coefficient of x³y³ in the expansion of (3x+y)(x-2y)⁵ -/
def coefficient_x3y3 : ℤ :=
  let binomial := fun (n k : ℕ) => Int.ofNat (Nat.choose n k)
  let term1 := 3 * binomial 5 3 * (-2)^3
  let term2 := binomial 5 2 * (-2)^2
  term1 + term2

/-- Theorem stating that the coefficient of x³y³ in (3x+y)(x-2y)⁵ is -200 -/
theorem coefficient_x3y3_eq_neg200 : coefficient_x3y3 = -200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_eq_neg200_l818_81892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l818_81845

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 + 3 * Real.sin x ^ 4 * Real.cos x ^ 2 + Real.cos x ^ 6

-- State the theorem about the range of g
theorem range_of_g :
  Set.range g = Set.Icc (11 / 27) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l818_81845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_tangent_circles_l818_81839

-- Define the circles and triangle
def circle_small : ℝ := 1
def circle_large : ℝ := 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_tangent_to_circles (t : Triangle) : Prop := sorry
def has_congruent_sides (t : Triangle) : Prop := sorry

-- Define the area function for a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area_with_tangent_circles 
  (t : Triangle) 
  (h1 : is_tangent_to_circles t) 
  (h2 : has_congruent_sides t) : 
  Real.sqrt 2 * 16 = area t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_tangent_circles_l818_81839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l818_81857

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Checks if a line passes through the focus -/
def passesThroughFocus (A B : Point) : Prop :=
  (B.y - focus.y) * (A.x - focus.x) = (A.y - focus.y) * (B.x - focus.x)

/-- Calculates the distance between two points -/
noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

/-- Main theorem -/
theorem parabola_chord_length (A B : Point) 
  (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (hFocus : passesThroughFocus A B) 
  (hSum : A.x + B.x = 6) : 
  distance A B = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l818_81857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l818_81812

/-- A circle M with center on the line y = -x + 2 and tangent to the lines x - y = 0 and x - y + 4 = 0 has the standard equation x^2 + (y-2)^2 = 2 -/
theorem circle_equation (M : Set (ℝ × ℝ)) : 
  (∃ (c : ℝ × ℝ), c.2 = -c.1 + 2 ∧ c ∈ M) →  -- center on y = -x + 2
  (∀ (p : ℝ × ℝ), p ∈ M → (p.1 - p.2 = 0 → (∀ (q : ℝ × ℝ), q ∈ M → p = q))) →  -- tangent to x - y = 0
  (∀ (p : ℝ × ℝ), p ∈ M → (p.1 - p.2 + 4 = 0 → (∀ (q : ℝ × ℝ), q ∈ M → p = q))) →  -- tangent to x - y + 4 = 0
  (∀ (x y : ℝ), (x, y) ∈ M ↔ x^2 + (y-2)^2 = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l818_81812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_relation_holds_l818_81868

noncomputable def x (t : ℝ) : ℝ := t^(3/(t-1))
noncomputable def y (t : ℝ) : ℝ := t^((t+1)/(t-1))

theorem no_relation_holds (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) :
  (y t)^(x t) ≠ (x t)^(y t) ∧
  (x t)^(x t) ≠ (y t)^(y t) ∧
  (x t)^((y t)^(x t)) ≠ (y t)^((x t)^(y t)) ∧
  (x t)^(y t) ≠ (y t)^(x t) :=
by
  sorry

#check no_relation_holds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_relation_holds_l818_81868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l818_81864

noncomputable def expansion (x : ℝ) (n : ℕ) := (x^(1/3) - 1/(2*x^(1/3)))^n

def binomial_ratio (n : ℕ) : ℚ := (n.choose 4 : ℚ) / (n.choose 2 : ℚ)

noncomputable def general_term (n r : ℕ) (x : ℝ) : ℝ := 
  n.choose r * (-1/2)^r * x^((n - 2*r : ℤ)/3)

theorem expansion_properties (x : ℝ) (n : ℕ) :
  binomial_ratio n = 14/3 →
  (n = 10 ∧ 
   general_term 10 2 x = 45/4 * x^2 ∧
   (∀ r, r ∈ ({2, 5, 8} : Set ℕ) ↔ (10 - 2*r) % 3 = 0)) := by
  sorry

#eval binomial_ratio 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l818_81864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_final_distance_l818_81877

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the traveler's journey --/
def travelerJourney : List (ℝ × ℝ) :=
  [(0, 18), (-11, 0), (0, -6), (6, 0)]

/-- Calculates the final position of the traveler --/
def finalPosition (journey : List (ℝ × ℝ)) : Point :=
  let finalCoords := journey.foldl (fun (acc : ℝ × ℝ) (step : ℝ × ℝ) => 
    (acc.1 + step.1, acc.2 + step.2)) (0, 0)
  { x := finalCoords.1, y := finalCoords.2 }

/-- The main theorem to prove --/
theorem traveler_final_distance : 
  distance (Point.mk 0 0) (finalPosition travelerJourney) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_final_distance_l818_81877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_linear_intersection_l818_81800

/-- Quadratic function -/
def y₁ (a : ℝ) (x : ℝ) : ℝ := a * x^2

/-- Linear function -/
def y₂ (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Distance from point to line y = -1 -/
def distToLine (x y : ℝ) : ℝ := y + 1

/-- Distance between two points -/
noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Main theorem -/
theorem quadratic_linear_intersection 
  (a f k b : ℝ) 
  (h₁ : f > 0)
  (h₂ : ∀ x : ℝ, distToLine x (y₁ a x) = dist x (y₁ a x) 0 f)
  (h₃ : y₂ k b 0 = f) :
  (f = 1 ∧ a = 1/4 ∧ b = 1) ∧ 
  (k = 1 → ∃ x₁ x₂ : ℝ, y₁ a x₁ = y₂ k b x₁ ∧ y₁ a x₂ = y₂ k b x₂ ∧ 
    (y₁ a x₁ + 1) * (y₁ a x₂ + 1) = 8) ∧
  (∃ p q : ℝ, (∀ x₁ x₂ : ℝ, y₁ a x₁ = y₂ k b x₁ ∧ y₁ a x₂ = y₂ k b x₂ → 
    (y₁ a x₁ + 1) * (y₁ a x₂ + 1) = p ∧ 
    Real.sqrt (x₁^2 * (1 + (a * x₁)^2/16)) * Real.sqrt (x₂^2 * (1 + (a * x₂)^2/16)) = q) ∧
    q = Real.sqrt (4 * p + 9)) := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_linear_intersection_l818_81800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_correct_l818_81824

noncomputable def p (x : ℝ) : ℝ := -10/3 * x^2 + 20/3 * x + 10

theorem p_is_correct :
  (∀ x : ℝ, x = 3 ∨ x = -1 → p x = 0) ∧
  (∃ a b c : ℝ, ∀ x : ℝ, p x = a * x^2 + b * x + c) ∧
  p 2 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_correct_l818_81824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_is_four_l818_81802

/-- The function f(x) = 2 + 2x / (x^2 + 1) -/
noncomputable def f (x : ℝ) : ℝ := 2 + 2 * x / (x^2 + 1)

/-- M is the maximum value of f(x) -/
noncomputable def M : ℝ := sSup (Set.range f)

/-- m is the minimum value of f(x) -/
noncomputable def m : ℝ := sInf (Set.range f)

/-- Theorem: The sum of the maximum and minimum values of f(x) is 4 -/
theorem sum_of_max_and_min_is_four : M + m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_is_four_l818_81802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_age_problem_l818_81825

/-- Maxwell's age problem -/
theorem maxwell_age_problem (maxwell_age : ℕ) (sister_age : ℕ) : 
  sister_age = 2 →
  maxwell_age + 2 = 2 * (sister_age + 2) →
  maxwell_age = 6 := by
  intro h1 h2
  -- Substitute sister_age with 2 in h2
  rw [h1] at h2
  -- Simplify the right side of h2
  simp at h2
  -- Solve the equation
  linarith

#check maxwell_age_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_age_problem_l818_81825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_13_l818_81837

theorem trigonometric_identity_13 :
  let d : ℝ := 2 * Real.pi / 13
  (Real.sin (4 * d) * Real.sin (8 * d) * Real.sin (12 * d) * Real.sin (16 * d) * Real.sin (20 * d)) /
  (Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_13_l818_81837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_minimum_triple_sum_minimum_achievable_l818_81834

open BigOperators

def triple_sum (x y : ℤ) : ℕ :=
  ∑ i in Finset.range 10, ∑ j in Finset.range 10, ∑ k in Finset.range 10,
    Int.natAbs ((k + 1) * (x + y - 10 * (i + 1)) * (3 * x - 6 * y - 36 * (j + 1)) * (19 * x + 95 * y - 95 * (k + 1)))

theorem triple_sum_minimum (x y : ℤ) : triple_sum x y ≥ 2394000000 := by
  sorry

theorem triple_sum_minimum_achievable : ∃ x y : ℤ, triple_sum x y = 2394000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_minimum_triple_sum_minimum_achievable_l818_81834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_volume_theorem_sum_m_n_p_theorem_l818_81813

/-- The volume of points inside or within one unit of a rectangular parallelepiped -/
noncomputable def extended_parallelepiped_volume (l w h : ℝ) : ℝ :=
  let original_volume := l * w * h
  let outward_volume := 2 * (l * w + l * h + w * h)
  let edge_volume := Real.pi * (l + w + h)
  let vertex_volume := 4 * Real.pi / 3
  original_volume + outward_volume + edge_volume + vertex_volume

/-- The dimensions of the parallelepiped -/
def l : ℝ := 2
def w : ℝ := 3
def h : ℝ := 4

/-- The theorem stating the volume of the extended parallelepiped -/
theorem extended_parallelepiped_volume_theorem :
  extended_parallelepiped_volume l w h = (228 + 31 * Real.pi) / 3 :=
by sorry

/-- The sum of m, n, and p in the fractional form (m + nπ) / p -/
def m_plus_n_plus_p : ℕ := 262

/-- Theorem stating that m + n + p equals 262 -/
theorem sum_m_n_p_theorem :
  m_plus_n_plus_p = 262 :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_volume_theorem_sum_m_n_p_theorem_l818_81813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_minimum_value_f_greater_than_one_l818_81817

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

-- Define the domain of x
def domain : Set ℝ := { x | x ≥ 1 }

-- Theorem 1: Monotonicity of f when a = -2
theorem f_monotone_increasing :
  ∀ x y, x ∈ domain → y ∈ domain → x < y → f (-2) x < f (-2) y :=
sorry

-- Theorem 2: Minimum value of f when a = -2
theorem f_minimum_value :
  ∀ x, x ∈ domain → f (-2) x ≥ f (-2) 1 ∧ f (-2) 1 = 0 :=
sorry

-- Theorem 3: Condition for f(x) > 1
theorem f_greater_than_one :
  (∀ x, x ∈ domain → f a x > 1) ↔ a > -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_minimum_value_f_greater_than_one_l818_81817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_walk_time_difference_l818_81878

/-- Represents Angelina's walking scenario -/
structure WalkingScenario where
  v : ℝ  -- Speed from home to grocery in meters per second
  d1 : ℝ  -- Distance from home to grocery in meters
  d2 : ℝ  -- Distance from grocery to gym in meters

/-- Calculates the time difference between two walks -/
noncomputable def timeDifference (s : WalkingScenario) : ℝ :=
  s.d1 / s.v - s.d2 / (2 * s.v)

/-- Theorem stating the time difference in Angelina's scenario -/
theorem angelina_walk_time_difference (s : WalkingScenario) 
  (h1 : s.d1 = 250) 
  (h2 : s.d2 = 360) 
  (h3 : s.v > 0) : 
  timeDifference s = 70 / s.v := by
  sorry

#check angelina_walk_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_walk_time_difference_l818_81878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_kite_coefficients_sum_l818_81833

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Calculates the x-intercepts of a parabola -/
def x_intercepts (p : Parabola) : Set ℝ :=
  {x | p.a * x^2 + p.c = 0}

/-- Calculates the y-intercept of a parabola -/
def y_intercept (p : Parabola) : ℝ := p.c

/-- Represents the kite formed by the intersections of two parabolas with the axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- Calculates the perimeter of the kite -/
noncomputable def kite_perimeter (k : Kite) : ℝ :=
  sorry  -- Actual calculation would go here

theorem parabola_kite_coefficients_sum (a b : ℝ) :
  let p1 : Parabola := ⟨a, 3⟩
  let p2 : Parabola := ⟨-b, 6⟩
  let k : Kite := ⟨p1, p2⟩
  (x_intercepts p1 = ∅) ∧
  (x_intercepts p2 ≠ ∅) ∧
  kite_perimeter k = 20 →
  a + b = 0.26 := by
  sorry

#check parabola_kite_coefficients_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_kite_coefficients_sum_l818_81833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l818_81896

/-- Calculates the interest rate at which A lent money to B -/
noncomputable def calculate_interest_rate (principal : ℝ) (b_lending_rate : ℝ) (time : ℝ) (b_gain : ℝ) : ℝ :=
  let interest_from_c := principal * b_lending_rate * time
  let interest_to_a := interest_from_c - b_gain
  (interest_to_a / (principal * time)) * 100

/-- Proves that the interest rate at which A lent money to B is 10% per annum -/
theorem interest_rate_is_ten_percent 
  (principal : ℝ) 
  (b_lending_rate : ℝ) 
  (time : ℝ) 
  (b_gain : ℝ) 
  (h1 : principal = 4000)
  (h2 : b_lending_rate = 0.115)
  (h3 : time = 3)
  (h4 : b_gain = 180) :
  calculate_interest_rate principal b_lending_rate time b_gain = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l818_81896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_properties_l818_81874

/-- Given a parabola and an ellipse with specific conditions, prove properties about the ellipse equation and a ratio involving perpendicular lines. -/
theorem parabola_ellipse_properties :
  ∀ (a b : ℝ) (B : ℝ × ℝ),
    a > b → b > 0 →
    (let (x₀, y₀) := B
     y₀^2 = 4 * x₀ ∧                    -- B is on the parabola y^2 = 4x
     x₀^2 / a^2 + y₀^2 / b^2 = 1 ∧      -- B is on the ellipse x^2/a^2 + y^2/b^2 = 1
     x₀ > 0 ∧ y₀ > 0 ∧                  -- B is in the first quadrant
     Real.sqrt ((x₀ - 1)^2 + y₀^2) = 5/3 -- |BF| = 5/3, where F is (1, 0)
    ) →
    (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∀ (m : ℝ), 
      let TF := 3 * Real.sqrt (1 + m^2)
      let PQ := 12 * (m^2 + 1) / (3 * m^2 + 4)
      TF / PQ ≥ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_properties_l818_81874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l818_81871

/-- Calculates the total amount after compound interest --/
noncomputable def total_amount (principal : ℝ) (rate : ℝ) (time : ℝ) (compounding : ℝ) : ℝ :=
  principal * (1 + rate / compounding) ^ (compounding * time)

/-- Theorem stating the total amount after compound interest --/
theorem compound_interest_problem (principal : ℝ) :
  let rate : ℝ := 0.08
  let time : ℝ := 2
  let compounding : ℝ := 1
  let interest : ℝ := 2828.80
  principal > 0 →
  interest = total_amount principal rate time compounding - principal →
  total_amount principal rate time compounding = 19828.80 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l818_81871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_perfect_negative_line_l818_81848

/-- Sample data type -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ

/-- Sample correlation coefficient -/
noncomputable def sampleCorrelationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem: If all points lie on y = -3x + 1, then correlation coefficient is -1 -/
theorem correlation_coefficient_perfect_negative_line (data : SampleData) 
  (h_n : data.n ≥ 2)
  (h_x_not_all_equal : ∃ (i j : Fin data.n), data.x i ≠ data.x j)
  (h_on_line : ∀ (i : Fin data.n), data.y i = -3 * data.x i + 1) :
  sampleCorrelationCoefficient data = -1 := by
  sorry

#check correlation_coefficient_perfect_negative_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_perfect_negative_line_l818_81848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l818_81803

/-- The line y = 3x - 1 -/
def line (x : ℝ) : ℝ := 3 * x - 1

/-- The point we're measuring distance from -/
def external_point : ℝ × ℝ := (4, -2)

/-- The proposed closest point on the line -/
def closest_point : ℝ × ℝ := (-0.5, -2.5)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem closest_point_on_line :
  closest_point.2 = line closest_point.1 ∧
  ∀ p : ℝ × ℝ, p.2 = line p.1 →
    distance closest_point external_point ≤ distance p external_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l818_81803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_symmetric_origin_l818_81818

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log ((x - 2) / (x + 2))

-- Define the domain of f
def domain (x : ℝ) : Prop := x < -2 ∨ x > 2

-- Theorem: f is an odd function
theorem f_is_odd : ∀ x, domain x → f (-x) = -f x := by
  sorry

-- Theorem: The graph of f is symmetric with respect to the origin
theorem f_symmetric_origin : ∀ x, domain x → ∃ y, f x = y ∧ f (-x) = -y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_symmetric_origin_l818_81818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l818_81838

-- Define the trapezoid ABCD
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_parallel_cd : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)
  ac_perp_cd : (C.1 - A.1) * (D.1 - C.1) + (C.2 - A.2) * (D.2 - C.2) = 0

-- Define the properties of the trapezoid
def trapezoid_properties (t : Trapezoid) : Prop :=
  let cd_length := Real.sqrt ((t.D.1 - t.C.1)^2 + (t.D.2 - t.C.2)^2)
  let tan_d := (t.A.2 - t.D.2) / (t.D.1 - t.A.1)
  let tan_b := (t.C.2 - t.B.2) / (t.B.1 - t.C.1)
  cd_length = 15 ∧ tan_d = 3/4 ∧ tan_b = 3/5

-- Define the length of BC
noncomputable def bc_length (t : Trapezoid) : ℝ :=
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)

-- Theorem statement
theorem trapezoid_bc_length (t : Trapezoid) (h : trapezoid_properties t) :
  ∃ ε > 0, |bc_length t - 21.864| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l818_81838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l818_81894

theorem function_inequality (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : DifferentiableOn ℝ f (Set.Ici a))
  (h2 : DifferentiableOn ℝ g (Set.Ici a))
  (h3 : f a = g a)
  (h4 : ∀ x > a, deriv f x > deriv g x) :
  ∀ x > a, f x > g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l818_81894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_equidistant_l818_81850

/-- A triangle in a 2D plane -/
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

/-- The angle bisector of an angle in a triangle -/
noncomputable def angle_bisector (t : Triangle) (vertex : Fin 3) : 
  Subspace ℝ (EuclideanSpace ℝ (Fin 2)) := sorry

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) := sorry

/-- The distance from a point to a line -/
noncomputable def dist_to_line (p : EuclideanSpace ℝ (Fin 2)) 
  (l : Subspace ℝ (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- The line through two points -/
noncomputable def line_through (p q : EuclideanSpace ℝ (Fin 2)) : 
  Subspace ℝ (EuclideanSpace ℝ (Fin 2)) := sorry

theorem incenter_equidistant (t : Triangle) :
  let i := incenter t
  let d1 := dist_to_line i (line_through t.A t.B)
  let d2 := dist_to_line i (line_through t.B t.C)
  let d3 := dist_to_line i (line_through t.C t.A)
  d1 = d2 ∧ d2 = d3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_equidistant_l818_81850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_of_intersecting_circles_l818_81893

/-- Given two circles x^2 + y^2 = 9 and (x-2)^2 + (y-1)^2 = 16 intersecting at points A and B,
    the equation of line AB is 2x + y + 1 = 0 -/
theorem line_equation_of_intersecting_circles :
  ∀ (A B : ℝ × ℝ),
  (A.1^2 + A.2^2 = 9) →
  (B.1^2 + B.2^2 = 9) →
  ((A.1 - 2)^2 + (A.2 - 1)^2 = 16) →
  ((B.1 - 2)^2 + (B.2 - 1)^2 = 16) →
  A ≠ B →
  ∃ (x y : ℝ), 2*x + y + 1 = 0 ∧ (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_of_intersecting_circles_l818_81893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_correct_l818_81847

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_O2 (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 9

-- Define the common chord length
noncomputable def common_chord_length : ℝ := Real.sqrt 65 / 2

-- Theorem statement
theorem common_chord_length_is_correct :
  ∃ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y ∧
  ∀ (x' y' : ℝ), circle_O1 x' y' ∧ circle_O2 x' y' →
    ((x - x')^2 + (y - y')^2 : ℝ) ≤ common_chord_length^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_correct_l818_81847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_lamp_c_on_l818_81805

-- Define the lamps
inductive Lamp
| A
| B
| C

-- Define the switches
inductive Switch
| Red
| Blue
| Green

-- Define the state of a lamp (on or off)
def LampState := Bool

-- Define the state of all lamps
def LampStates := Lamp → LampState

-- Define the effect of a switch on the lamp states
def applySwitch (s : Switch) (state : LampStates) : LampStates :=
  match s with
  | Switch.Red => λ l => match l with
    | Lamp.A | Lamp.B => !state l
    | Lamp.C => state l
  | Switch.Blue => λ l => match l with
    | Lamp.B | Lamp.C => !state l
    | Lamp.A => state l
  | Switch.Green => λ l => match l with
    | Lamp.A | Lamp.C => !state l
    | Lamp.B => state l

-- Define the initial state where all lamps are on
def initialState : LampStates := λ _ => true

-- Define a sequence of switch presses
def switchSequence : List Switch := []

-- Define the function to apply a sequence of switches
def applySequence (seq : List Switch) (state : LampStates) : LampStates :=
  seq.foldl (λ s switch => applySwitch switch s) state

-- Define the final state after all switch presses
def finalState : LampStates :=
  applySequence switchSequence (applySwitch Switch.Red initialState)

-- Define the count of red switch presses
def redSwitchCount : Nat := 8

-- Theorem statement
theorem only_lamp_c_on :
  (redSwitchCount = 8) →
  (switchSequence.length = 19) →
  (finalState Lamp.A = false) ∧
  (finalState Lamp.B = false) ∧
  (finalState Lamp.C = true) :=
by sorry

#check only_lamp_c_on

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_lamp_c_on_l818_81805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_15_seconds_l818_81810

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that the time taken for the train to cross the platform is approximately 15 seconds -/
theorem train_crossing_time_approx_15_seconds :
  ∃ ε > 0, |train_crossing_time 120 130.02 60 - 15| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_15_seconds_l818_81810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_quarter_f_solution_set_l818_81836

-- Define the max function
noncomputable def max' (x y : ℝ) : ℝ := if x ≥ y then x else y

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := max' (a^x - a) (-Real.log x / Real.log a)

-- Theorem for the first part
theorem f_sum_quarter (a : ℝ) (h1 : a = 1/4) :
  f a 2 + f a (1/2) = 3/4 := by sorry

-- Theorem for the second part
theorem f_solution_set (a : ℝ) (h1 : a > 1) :
  ∀ x > 0, f a x ≥ 2 ↔ (x ≤ 1/a^2 ∨ x ≥ Real.log (a + 2) / Real.log a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_quarter_f_solution_set_l818_81836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_downstream_distance_l818_81808

/-- Represents the scenario of Julie's rowing trip -/
structure RowingTrip where
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the downstream distance for a given rowing trip -/
noncomputable def downstream_distance (trip : RowingTrip) : ℝ :=
  let boat_speed := trip.upstream_distance / trip.upstream_time + trip.stream_speed
  (boat_speed + trip.stream_speed) * trip.downstream_time

/-- Theorem stating that Julie rowed 36 km downstream -/
theorem julie_downstream_distance :
  let trip := RowingTrip.mk 32 4 4 0.5
  downstream_distance trip = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_downstream_distance_l818_81808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_sin_max_cos_squared_plus_sin_max_attainable_l818_81854

theorem cos_squared_plus_sin_max (x : ℝ) : Real.cos x ^ 2 + Real.sin x ≤ 5 / 4 :=
by sorry

theorem cos_squared_plus_sin_max_attainable : ∃ x : ℝ, Real.cos x ^ 2 + Real.sin x = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_sin_max_cos_squared_plus_sin_max_attainable_l818_81854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_in_interval_l818_81889

theorem whole_numbers_in_interval : ∃ (n : ℕ), n = (Int.floor (3 * Real.pi) - Int.ceil (7/4 : ℚ) + 1) ∧ n = 8 := by
  -- We use Int.floor and Int.ceil instead of ⌊⌋ and ⌈⌉ for better compatibility
  -- We cast 7/4 to ℚ (rational numbers) to ensure correct handling
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_in_interval_l818_81889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l818_81899

/-- Given that in the expansion of (2x + 1/(2x))^(2n), the coefficient of x^2 is 224,
    prove that the coefficient of 1/x^2 is 14 -/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ r : ℕ, r ≤ 2*n ∧ Nat.choose (2*n) r * 2^(2*n - 2*r) = 224 ∧ 2*n - 2*r = 2) →
  (∃ s : ℕ, s ≤ 2*n ∧ Nat.choose (2*n) s * 2^(2*n - 2*s) = 14 ∧ 2*n = 2*s - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l818_81899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_disinfected_area_l818_81819

/- Represents the number of barrels of type A disinfectant -/
def x : ℕ := 22

/- Represents the total cost of purchasing disinfectants -/
def y (x : ℕ) : ℕ := 100 * x + 6000

/- Represents the total area that can be disinfected -/
def S (x : ℕ) : ℕ := 1000 * x + 30000

/- The maximum number of type A barrels that can be purchased within budget -/
def max_x : ℕ := 22

theorem max_disinfected_area :
  ∀ x : ℕ, x ≤ max_x → x > 0 → x < 30 →
  y x ≤ 8200 →
  S x ≤ 52000 ∧ (∃ (x : ℕ), x ≤ max_x ∧ x > 0 ∧ x < 30 ∧ S x = 52000) := by
  sorry

#check max_disinfected_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_disinfected_area_l818_81819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_to_sin_cos_l818_81863

theorem tan_to_sin_cos (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.tan α = 2) :
  Real.sin α = 2 / Real.sqrt 5 ∧ Real.cos α = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_to_sin_cos_l818_81863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_l818_81832

/-- Triangle PQR with vertices P(0, 10), Q(4, 0), and R(10, 0) -/
structure TrianglePQR where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (4, 0)
  R : ℝ × ℝ := (10, 0)

/-- Line segment PQ -/
def linePQ (t : TrianglePQR) : Set (ℝ × ℝ) :=
  {p | ∃ l : ℝ, 0 ≤ l ∧ l ≤ 1 ∧ p = ((1 - l) • t.P.1 + l • t.Q.1, (1 - l) • t.P.2 + l • t.Q.2)}

/-- Line segment PR -/
def linePR (t : TrianglePQR) : Set (ℝ × ℝ) :=
  {p | ∃ l : ℝ, 0 ≤ l ∧ l ≤ 1 ∧ p = ((1 - l) • t.P.1 + l • t.R.1, (1 - l) • t.P.2 + l • t.R.2)}

/-- Horizontal line y = s -/
def horizontalLine (s : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = s}

/-- Point V where horizontal line intersects PQ -/
noncomputable def pointV (t : TrianglePQR) (s : ℝ) : ℝ × ℝ :=
  (4 - 2*s/5, s)

/-- Point W where horizontal line intersects PR -/
def pointW (s : ℝ) : ℝ × ℝ :=
  (10 - s, s)

/-- Area of triangle PVW -/
noncomputable def areaPVW (t : TrianglePQR) (s : ℝ) : ℝ :=
  (1/2) * (6 - 3*s/5) * (10 - s)

/-- Main theorem -/
theorem horizontal_line_intersection (t : TrianglePQR) (s : ℝ) :
  (pointV t s ∈ linePQ t) ∧
  (pointW s ∈ linePR t) ∧
  (pointV t s ∈ horizontalLine s) ∧
  (pointW s ∈ horizontalLine s) ∧
  (areaPVW t s = 15) →
  s = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_l818_81832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circle_centers_l818_81862

/-- The greatest possible distance between the centers of two circles with 5-inch diameters
    placed within a 15-inch by 18-inch rectangle, without extending beyond the rectangle's boundaries. -/
theorem max_distance_between_circle_centers : 
  ∃ (max_distance : ℝ), max_distance = Real.sqrt 269 := by
  let rectangle_width : ℝ := 15
  let rectangle_height : ℝ := 18
  let circle_diameter : ℝ := 5
  let circle_radius : ℝ := circle_diameter / 2
  let max_horizontal_distance : ℝ := rectangle_width - 2 * circle_radius
  let max_vertical_distance : ℝ := rectangle_height - 2 * circle_radius
  let max_distance : ℝ := Real.sqrt (max_horizontal_distance ^ 2 + max_vertical_distance ^ 2)
  
  exists max_distance
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_circle_centers_l818_81862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l818_81860

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else k * Real.exp x / x

theorem range_of_k (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f k x = y) ↔ (k > 0 ∧ k ≤ 2 / Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l818_81860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_from_second_shop_l818_81870

/-- Prove that the number of books bought from the second shop is 60 -/
theorem books_from_second_shop :
  ∀ (first_shop_books : ℕ) (first_shop_cost second_shop_cost avg_price : ℚ),
  first_shop_books = 32 →
  first_shop_cost = 1500 →
  second_shop_cost = 340 →
  avg_price = 20 →
  (first_shop_cost + second_shop_cost) / avg_price - first_shop_books = 60 :=
by
  intros first_shop_books first_shop_cost second_shop_cost avg_price h1 h2 h3 h4
  -- Convert first_shop_books to ℚ for consistent arithmetic
  have h5 : (first_shop_cost + second_shop_cost) / avg_price - ↑first_shop_books = 60
  · sorry
  exact h5


end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_from_second_shop_l818_81870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l818_81841

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - (b / a)^2)

/-- The right focus of an ellipse. -/
noncomputable def rightFocus (e : Ellipse a b) : ℝ × ℝ := (a * eccentricity e, 0)

/-- The upper vertex of an ellipse. -/
def upperVertex (e : Ellipse a b) : ℝ × ℝ := (0, b)

/-- The left vertex of an ellipse. -/
def leftVertex (e : Ellipse a b) : ℝ × ℝ := (-a, 0)

/-- The perpendicular bisector of a line segment passes through a point. -/
def perpendicularBisectorPassesThrough (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let slope := (p2.2 - p1.2) / (p2.1 - p1.1)
  let perpSlope := -1 / slope
  perpSlope * (p3.1 - midpoint.1) = p3.2 - midpoint.2

theorem ellipse_eccentricity (a b : ℝ) (e : Ellipse a b) :
  perpendicularBisectorPassesThrough (upperVertex e) (rightFocus e) (leftVertex e) →
  eccentricity e = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l818_81841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_points_and_fixed_points_l818_81815

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define point P on line l
def point_P (b : ℝ) : ℝ × ℝ := (2*b, b)

-- Define the length of PA
noncomputable def length_PA (b : ℝ) : ℝ := sorry

-- Define the equation of circle N
def circle_N (x y b : ℝ) : Prop := 
  (x - b)^2 + (y - (b + 4)/2)^2 = (4*b^2 + (4 - b)^2)/4

-- Define the length of AB
noncomputable def length_AB (b : ℝ) : ℝ := 
  4 * Real.sqrt (1 - 4 / (5*b^2 - 8*b + 16))

theorem circle_tangent_points_and_fixed_points :
  (∃ b₁ b₂ : ℝ, 
    (length_PA b₁ = 2 * Real.sqrt 3 ∧ point_P b₁ = (0, 0)) ∧
    (length_PA b₂ = 2 * Real.sqrt 3 ∧ point_P b₂ = (16/5, 8/5))) ∧
  (∀ b : ℝ, line_l (point_P b).1 (point_P b).2 → 
    circle_N 0 4 b ∧ circle_N (8/5) (4/5) b) ∧
  (∃ b : ℝ, ∀ b' : ℝ, length_AB b ≤ length_AB b' ∧ length_AB b = Real.sqrt 11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_points_and_fixed_points_l818_81815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_roots_coincide_l818_81822

-- Define the type for quadratic polynomials
def QuadraticPolynomial := ℝ → ℝ

-- Define the property of having a repeated root
def has_repeated_root (p : QuadraticPolynomial) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, p x = 0 ↔ x = r

-- Define the addition of two quadratic polynomials
def add_poly (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  λ x ↦ p x + q x

-- Theorem statement
theorem repeated_roots_coincide (P Q : QuadraticPolynomial) :
  has_repeated_root P →
  has_repeated_root Q →
  has_repeated_root (add_poly P Q) →
  ∃ r : ℝ, (∀ x : ℝ, P x = 0 ↔ x = r) ∧
           (∀ x : ℝ, Q x = 0 ↔ x = r) ∧
           (∀ x : ℝ, add_poly P Q x = 0 ↔ x = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_roots_coincide_l818_81822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_triangular_prism_l818_81867

/-- Represents a right prism with triangular bases -/
structure TriangularPrism where
  a : ℝ  -- side length of the base triangle
  b : ℝ  -- side length of the base triangle
  h : ℝ  -- height of the prism
  θ : ℝ  -- angle between sides a and b

/-- The sum of areas of three mutually adjacent faces -/
noncomputable def adjacentFacesArea (p : TriangularPrism) : ℝ :=
  p.a * p.h + p.b * p.h + 1/2 * p.a * p.b * Real.sin p.θ

/-- The volume of the triangular prism -/
noncomputable def volume (p : TriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h * Real.sin p.θ

/-- Theorem stating the maximum volume of the prism -/
theorem max_volume_of_triangular_prism :
  ∀ p : TriangularPrism, adjacentFacesArea p = 36 → volume p ≤ 27 :=
by sorry

#check max_volume_of_triangular_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_triangular_prism_l818_81867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_square_areas_l818_81873

noncomputable def triangle_side : ℝ := 12

noncomputable def triangle_height : ℝ := triangle_side * Real.sqrt 3 / 2

noncomputable def circle_radius : ℝ := triangle_height / Real.sqrt 3

noncomputable def square_side : ℝ := triangle_height

noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

noncomputable def square_area : ℝ := square_side ^ 2

theorem circle_and_square_areas :
  circle_area = 36 * Real.pi ∧ square_area = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_square_areas_l818_81873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_monotonicity_l818_81821

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

theorem extreme_values_and_monotonicity :
  (∃ (x_min x_max : ℝ), x_min ∈ Set.Icc (-4 : ℝ) 6 ∧ x_max ∈ Set.Icc (-4 : ℝ) 6 ∧
    (∀ x, x ∈ Set.Icc (-4 : ℝ) 6 → f (-2) x_min ≤ f (-2) x ∧ f (-2) x ≤ f (-2) x_max) ∧
    f (-2) x_min = -1 ∧ f (-2) x_max = 35) ∧
  (∀ x y, x ∈ Set.Icc (-4 : ℝ) 6 → y ∈ Set.Icc (-4 : ℝ) 6 → x < y → f 4 x < f 4 y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_monotonicity_l818_81821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_fraction_of_digit_sum_4_l818_81879

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

def count_primes (s : Finset ℕ) : ℕ := sorry

theorem prime_fraction_of_digit_sum_4 :
  let S := Finset.filter (λ n ↦ n ≤ 1000 ∧ digit_sum n = 4) (Finset.range 1001)
  (count_primes S : ℚ) / (S.card : ℚ) = 4 / 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_fraction_of_digit_sum_4_l818_81879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_two_l818_81856

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp (2 * x) - Real.exp (a * x)) * Real.cos x

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_implies_a_equals_negative_two :
  ∀ a : ℝ, is_odd (f a) → a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_two_l818_81856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_problem_solution_l818_81806

/-- Represents a partner in the investment partnership -/
structure Partner where
  investment : ℚ
  profit : ℚ
  duration : ℚ

/-- The partnership problem setup -/
structure PartnershipProblem where
  P : Partner
  Q : Partner
  investment_ratio : P.investment / Q.investment = 7 / 5
  profit_ratio : P.profit / Q.profit = 7 / 11
  P_duration : P.duration = 5

theorem partnership_problem_solution (prob : PartnershipProblem) : prob.Q.duration = 55 := by
  sorry

#check partnership_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_problem_solution_l818_81806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_decreasing_implies_a_less_than_three_halves_l818_81882

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x) * Real.exp x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x^2 + (2 - a)*x - a) * Real.exp x

-- State the theorem
theorem not_decreasing_implies_a_less_than_three_halves (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f_derivative a x > 0) → a < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_decreasing_implies_a_less_than_three_halves_l818_81882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pentagon_pebbles_l818_81880

def P : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | n+2 => P (n+1) + 3*(n+2) - 2

theorem tenth_pentagon_pebbles : P 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pentagon_pebbles_l818_81880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transform_l818_81826

-- Define the points in the coordinate plane
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (20, 0)
def P : ℝ × ℝ := (30, 0)
def Q : ℝ × ℝ := (30, 20)
def R : ℝ × ℝ := (50, 0)

-- Define the rotation parameters
variable (n x y : ℝ)

-- Helper function (not part of the problem, but needed for the statement)
noncomputable def rotatePoint (p : ℝ × ℝ) (angle : ℝ) (center : ℝ × ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem rotation_transform (h1 : 0 < n) (h2 : n < 180) 
  (h3 : rotatePoint A n (x, y) = P)
  (h4 : rotatePoint B n (x, y) = Q)
  (h5 : rotatePoint C n (x, y) = R) :
  n + x + y = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transform_l818_81826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_drift_theorem_l818_81811

/-- Represents the time taken by a raft to drift downstream given steamboat travel times -/
noncomputable def raft_drift_time (downstream_time upstream_time : ℝ) : ℝ :=
  (downstream_time * upstream_time) / (upstream_time - downstream_time) * 2

/-- Theorem stating that for a river where a steamboat takes 3 days downstream
    and 4 days upstream, a raft will take 24 days to drift downstream -/
theorem raft_drift_theorem :
  raft_drift_time 3 4 = 24 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_drift_theorem_l818_81811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_from_equation_triangle_is_isosceles_right_l818_81828

/-- A triangle with side lengths satisfying a specific equation is isosceles right. -/
theorem isosceles_right_triangle_from_equation 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_lengths_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h : Real.sqrt (c^2 - a^2 - b^2) + |a - b| = 0) : 
  a = b ∧ c^2 = a^2 + b^2 := by
  sorry

/-- Definition of an isosceles right triangle -/
def IsoscelesRightTriangle (a b c : ℝ) : Prop :=
  a = b ∧ c^2 = a^2 + b^2

/-- The main theorem stating that a triangle satisfying the given equation is isosceles right. -/
theorem triangle_is_isosceles_right 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_lengths_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h : Real.sqrt (c^2 - a^2 - b^2) + |a - b| = 0) : 
  IsoscelesRightTriangle a b c := by
  apply isosceles_right_triangle_from_equation a b c triangle_inequality side_lengths_positive h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_from_equation_triangle_is_isosceles_right_l818_81828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_maximizes_relationship_l818_81849

-- Define the concept of a regression equation
def RegressionEquation (y x : ℝ → ℝ) : Prop := sorry

-- Define the concept of a real relationship between variables
def RealRelationship (y x : ℝ → ℝ) : Prop := sorry

-- Define the concept of maximizing a relationship
def MaximizedRelationship (r : Prop) : Prop := sorry

-- Theorem stating that a regression equation represents the maximized real relationship
theorem regression_equation_maximizes_relationship (y x : ℝ → ℝ) :
  RegressionEquation y x → MaximizedRelationship (RealRelationship y x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_equation_maximizes_relationship_l818_81849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_functions_imply_n_equals_one_l818_81861

theorem equal_functions_imply_n_equals_one (n : ℝ) : 
  (λ (x : ℝ) => x^3 - 3*x^2 + n) 2 = (λ (x : ℝ) => 2*x^3 - 6*x^2 + 5*n) 2 → n = 1 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_functions_imply_n_equals_one_l818_81861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l818_81801

theorem angle_relation (α β γ φ : ℝ) 
  (h1 : Real.sin α + 7 * Real.sin β = 4 * (Real.sin γ + 2 * Real.sin φ))
  (h2 : Real.cos α + 7 * Real.cos β = 4 * (Real.cos γ + 2 * Real.cos φ)) :
  2 * Real.cos (α - φ) = 7 * Real.cos (β - γ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l818_81801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_mork_income_ratio_l818_81831

/-- Represents the income ratio between Mindy and Mork -/
def income_ratio (n : ℝ) : Prop := True

/-- Mork's tax rate -/
def mork_tax_rate : ℝ := 0.40

/-- Mindy's tax rate -/
def mindy_tax_rate : ℝ := 0.30

/-- Combined tax rate for Mork and Mindy -/
def combined_tax_rate : ℝ := 0.32

/-- Theorem stating the income ratio between Mindy and Mork -/
theorem mindy_mork_income_ratio :
  ∀ n : ℝ, income_ratio n →
    (mork_tax_rate + mindy_tax_rate * n) / (1 + n) = combined_tax_rate →
    n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mindy_mork_income_ratio_l818_81831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_always_negative_sum_l818_81804

noncomputable def f (x : ℝ) : ℝ := x - 1/x

theorem m_range_for_always_negative_sum :
  ∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f (m * x) + m * f x < 0) ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_always_negative_sum_l818_81804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_income_calculation_l818_81886

/-- Calculate the annual income from dividend given investment details -/
theorem dividend_income_calculation 
  (investment : ℝ) 
  (face_value : ℝ) 
  (quoted_price : ℝ) 
  (dividend_rate : ℝ) : ℝ :=
by
  let num_shares := ⌊investment / quoted_price⌋
  let annual_income := num_shares * face_value * dividend_rate
  have h1 : investment = 4455 := by sorry
  have h2 : face_value = 10 := by sorry
  have h3 : quoted_price = 8.25 := by sorry
  have h4 : dividend_rate = 0.12 := by sorry
  sorry

#check dividend_income_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_income_calculation_l818_81886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l818_81843

noncomputable def f (x : ℝ) : ℝ := 
  if x < 1 then 3 * x - 1 else 2 * x^2

def solution_set : Set ℝ := { a | f (f a) = 2 * (f a)^2 }

theorem solution_set_characterization : 
  solution_set = Set.Ici (2/3) ∪ {1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l818_81843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_and_normal_l818_81865

noncomputable def curve (t : ℝ) : ℝ × ℝ := (2 * Real.tan t, 2 * Real.sin t ^ 2 + Real.sin (2 * t))

noncomputable def t₀ : ℝ := Real.pi / 4

noncomputable def tangent_line (x : ℝ) : ℝ := (1/2) * x + 1

noncomputable def normal_line (x : ℝ) : ℝ := -2 * x + 6

theorem curve_tangent_and_normal :
  let (x₀, y₀) := curve t₀
  let m := (Real.sin t₀ * Real.cos t₀ ^ 3 + Real.cos (2 * t₀) * Real.cos t₀ ^ 2) / (1 / Real.cos t₀ ^ 2)
  (∀ x, tangent_line x = y₀ + m * (x - x₀)) ∧
  (∀ x, normal_line x = y₀ - (1/m) * (x - x₀)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_and_normal_l818_81865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l818_81807

/-- The sum of the infinite series 1 + 2²(1/999) + 3²(1/999)² + 4²(1/999)³ + ... -/
noncomputable def infiniteSeries : ℝ := ∑' n, (n + 1)^2 * (1/999)^n

/-- The theorem stating that the infinite series sum equals 997005/996004 -/
theorem infiniteSeriesSum : infiniteSeries = 997005/996004 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l818_81807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l818_81884

/-- The eccentricity of a hyperbola given its equation and a condition on its asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let Q : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 4*x + 6*y = 0
  let center : ℝ × ℝ := (2, -3)
  let asymptote : ℝ → ℝ := λ x => b/a * x
  (asymptote (center.1) = center.2) →
  ∃ e : ℝ, e = Real.sqrt 13 / 2 ∧ e^2 = 1 + (b/a)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l818_81884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l818_81887

/-- The rotation function -/
noncomputable def f (z : ℂ) : ℂ := ((-2 + 2 * Complex.I * Real.sqrt 3) * z + (5 * Real.sqrt 3 - 9 * Complex.I)) / 3

/-- The fixed point of the rotation -/
noncomputable def d : ℂ := (7 * Real.sqrt 3) / 37 - (35 * Complex.I) / 37

/-- Theorem stating that d is the fixed point of the rotation function f -/
theorem rotation_fixed_point : f d = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l818_81887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_N_power_10_mod_7_equals_1_l818_81888

theorem probability_N_power_10_mod_7_equals_1 :
  ∃ (S : Finset ℕ),
    S.card = 2023 ∧
    (Finset.filter (fun n => (n^10) % 7 = 1) S).card / S.card = 3/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_N_power_10_mod_7_equals_1_l818_81888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_spot_location_l818_81890

noncomputable def bridge1 : ℝ := 25
noncomputable def bridge2 : ℝ := 85
noncomputable def fishing_spot_ratio : ℝ := 2/3

theorem fishing_spot_location :
  bridge1 + fishing_spot_ratio * (bridge2 - bridge1) = 65 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_spot_location_l818_81890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l818_81895

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.sqrt (8 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 1 ≤ x ∧ x ≤ 8} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l818_81895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l818_81851

/-- 
Given positive integers x and y, prove that if:
1. x % y = 7
2. x / y = 86.1
3. x + y is prime
Then y = 70
-/
theorem problem_solution (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 7)
  (h2 : (x : ℝ) / (y : ℝ) = 86.1)
  (h3 : Nat.Prime (x + y)) :
  y = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l818_81851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l818_81835

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 1

theorem function_properties :
  (∀ x : ℝ, f x ≤ 1) ∧
  (∀ x : ℝ, f x ≥ 5/6) ∧
  (f 0 = 1) ∧
  (f 1 = 5/6) ∧
  ((∃ c : ℝ, c = 3/4 ∧ (∀ x : ℝ, f (3/2) + c * (x - 3/2) = c * x - 1/8)) ∨
   (∀ x : ℝ, f (3/2) + 0 * (x - 3/2) = 1)) ∧
  (∫ x in (0 : ℝ)..(3/2 : ℝ), (1 - f x) = 9/64) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l818_81835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_range_l818_81875

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 + x - 2 * Real.log x + a

/-- f(x) has exactly one zero in the interval (0, 2) -/
def has_unique_zero (a : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 2 ∧ f a x = 0

/-- The theorem stating the range of a for which f(x) has exactly one zero in (0, 2) -/
theorem f_unique_zero_range :
  ∀ a : ℝ, has_unique_zero a ↔ (a = -3/2 ∨ a ≤ 2 * Real.log 2 - 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_range_l818_81875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l818_81816

theorem candy_distribution (n : Nat) (h : n = 2010) : 
  (Finset.filter (fun k => ∃ m : Fin n, k = (m * (m + 1) / 2) % n) (Finset.range n)).card = 408 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l818_81816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_unique_k_characterization_l818_81852

/-- The number of elements in {k+1, k+2, ..., 2k} with exactly 3 ones in binary representation -/
def f (k : ℕ+) : ℕ := sorry

theorem existence_of_k (m : ℕ+) : ∃ k : ℕ+, f k = m := by sorry

theorem unique_k_characterization :
  {m : ℕ+ | ∃! k : ℕ+, f k = m} = {m : ℕ+ | ∃ s : ℕ, s ≥ 2 ∧ m = s * (s - 1) / 2 + 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_unique_k_characterization_l818_81852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l818_81872

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3/2) * x^2 - 3 * Real.log x

-- Define the theorem
theorem max_m_value (m : ℤ) :
  (∀ x > 1, f (x * Real.log x + 2*x - 1) > f (↑m * (x - 1))) →
  m ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l818_81872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l818_81820

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the geometric sequence property for a_1, a_3, a_6
def geometric_property (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, a 3 = a 1 * r ∧ a 6 = a 3 * r

-- Define the b sequence
def b_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 0 ∧ ∀ n ≥ 2, |b n - b (n - 1)| = 2^(a n)

theorem arithmetic_sequence_proof 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : a 2 = 5)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_property a) :
  (∀ n, a n = n + 3) ∧
  (∀ b, b_sequence b a → 
    (b 3 ∈ ({-96, -32, 32, 96} : Set ℝ) ∧
     (∀ k, b k = 2116 → k ≥ 7) ∧
     (∃ k, b k = 2116 ∧ k = 7))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l818_81820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l818_81827

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℂ)

theorem matrix_properties (h : A^2 + B^2 = 2 * A * B) :
  (Matrix.det (A * B - B * A) = 0) ∧
  (Matrix.rank (A - B) = 1 → A * B = B * A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l818_81827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_Y_value_l818_81858

/-- Right triangle XYZ with given side lengths -/
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XY ^ 2 + XZ ^ 2 = YZ ^ 2

/-- The tangent of angle Y in the right triangle -/
noncomputable def tan_Y (t : RightTriangle) : ℝ := t.XZ / t.XY

theorem tan_Y_value (t : RightTriangle) 
  (h1 : t.XY = 24) 
  (h2 : t.YZ = 25) : 
  tan_Y t = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_Y_value_l818_81858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l818_81898

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (Real.sqrt 3 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ := f x φ + (deriv (f · φ)) x

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, g x φ = -g (-x) φ) : φ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l818_81898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l818_81842

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) - 1

-- Define the theorem
theorem triangle_side_length 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : f B = 0)
  (h2 : a * c * Real.cos B = 3/2)
  (h3 : a + c = 4)
  : b = Real.sqrt 7 := by
  sorry

#check triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l818_81842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_ellipse_to_line_l818_81885

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 4*x - 5*y + 40 = 0

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |4*x - 5*y + 40| / Real.sqrt 41

/-- Theorem stating the minimal distance from the ellipse to the line -/
theorem minimal_distance_ellipse_to_line :
  ∃ (x y : ℝ), ellipse x y ∧ 
  (∀ (x' y' : ℝ), ellipse x' y' → distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = 15 / Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_ellipse_to_line_l818_81885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_b_mod_6_l818_81840

theorem remainder_b_mod_6 (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : b % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_b_mod_6_l818_81840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_irrational_l818_81814

theorem sqrt_2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_irrational_l818_81814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_quadrilateral_theorem_l818_81866

structure Quadrilateral (A B C D : ℝ × ℝ) : Prop where
  -- Define the quadrilateral ABCD

structure Circle (O : ℝ × ℝ) (r : ℝ) : Prop where
  -- Define the circle with center O and radius r

def on_circle (P : ℝ × ℝ) (c : Circle O r) : Prop :=
  -- Define what it means for a point to be on a circle
  sorry

def intersect (c : Circle O r) (l : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  -- Define what it means for a circle to intersect a line at a point
  sorry

def chord_length (P Q : ℝ × ℝ) : ℝ :=
  -- Define the length of a chord
  sorry

def angle (P Q R : ℝ × ℝ) : ℝ :=
  -- Define an angle
  sorry

def line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  -- Define a line through two points
  sorry

theorem circle_quadrilateral_theorem 
  (A B C D : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) (E F : ℝ × ℝ) 
  (quad : Quadrilateral A B C D) (c : Circle O r) :
  on_circle B c → on_circle C c → on_circle D c →
  intersect c (line A D) E →
  intersect c (line A B) F →
  chord_length B F = chord_length F E →
  chord_length F E = chord_length E D →
  chord_length B C = chord_length C D →
  angle D A B = π / 2 →
  angle O B C = 3 * π / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_quadrilateral_theorem_l818_81866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_zero_l818_81859

/-- Rational Man's track -/
noncomputable def rational_track (t : ℝ) : ℝ × ℝ := (Real.cos t, Real.sin t)

/-- Irrational Man's track -/
noncomputable def irrational_track (t : ℝ) : ℝ × ℝ := (-1 + 3 * Real.cos (t / 2), 3 * Real.sin (t / 2))

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The shortest distance between Rational Man's track and Irrational Man's track is 0 -/
theorem shortest_distance_is_zero :
  ∃ (t₁ t₂ : ℝ), distance (rational_track t₁) (irrational_track t₂) = 0 := by
  sorry

#check shortest_distance_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_zero_l818_81859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l818_81829

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem g_properties :
  (∀ x, g x = 2 * Real.cos (2 * x)) ∧
  (∀ x, g (-x) = g x) ∧
  (∀ x, g (x + Real.pi) = g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l818_81829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_writing_concepts_l818_81897

/-- This theorem is a placeholder for the speech writing concepts discussed. -/
theorem speech_writing_concepts : True := by
  trivial

#check speech_writing_concepts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_writing_concepts_l818_81897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_leq_is_forall_gt_l818_81891

theorem negation_of_exists_leq_is_forall_gt :
  (¬ ∃ m : ℝ, (3 : ℝ)^m ≤ 0) ↔ (∀ m : ℝ, (3 : ℝ)^m > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_leq_is_forall_gt_l818_81891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_mile_ride_cost_l818_81876

-- Define the parameters of the taxi ride
def base_fare : ℚ := 2
def cost_per_mile : ℚ := 3/10
def distance : ℚ := 12
def discount_threshold : ℚ := 10
def discount_rate : ℚ := 1/10

-- Define the function to calculate the taxi fare
def taxi_fare (base : ℚ) (rate : ℚ) (miles : ℚ) (threshold : ℚ) (discount : ℚ) : ℚ :=
  let total_before_discount := base + rate * miles
  if miles > threshold then
    total_before_discount * (1 - discount)
  else
    total_before_discount

-- Theorem to prove
theorem twelve_mile_ride_cost :
  taxi_fare base_fare cost_per_mile distance discount_threshold discount_rate = 63/25 := by
  sorry

#eval taxi_fare base_fare cost_per_mile distance discount_threshold discount_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_mile_ride_cost_l818_81876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density_increase_after_compaction_l818_81855

/-- Density is mass divided by volume -/
noncomputable def density (mass volume : ℝ) : ℝ := mass / volume

theorem density_increase_after_compaction 
  (m : ℝ) (v1 v2 : ℝ) (ρ1 ρ2 : ℝ) 
  (h1 : v2 = 0.8 * v1) 
  (h2 : ρ1 = density m v1) 
  (h3 : ρ2 = density m v2) :
  ρ2 = 1.25 * ρ1 := by
  sorry

#check density_increase_after_compaction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_density_increase_after_compaction_l818_81855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_circular_motion_of_triangle_vertex_l818_81883

/-- Represents a point moving uniformly along a circle -/
structure CircularMotion where
  center : ℂ
  radius : ℝ
  angularVelocity : ℝ
  initialAngle : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : ℂ
  B : ℂ
  C : ℂ

/-- The theorem statement -/
theorem uniform_circular_motion_of_triangle_vertex 
  (motionA motionB : CircularMotion)
  (triangle : EquilateralTriangle)
  (t : ℝ)
  (hA : triangle.A = motionA.center + motionA.radius * Complex.exp (Complex.I * (motionA.angularVelocity * t + motionA.initialAngle)))
  (hB : triangle.B = motionB.center + motionB.radius * Complex.exp (Complex.I * (motionB.angularVelocity * t + motionB.initialAngle)))
  (hEqualVelocity : motionA.angularVelocity = motionB.angularVelocity)
  (hClockwise : motionA.angularVelocity > 0)
  (hEquilateral : 
    Complex.abs (triangle.B - triangle.A) = Complex.abs (triangle.C - triangle.B) ∧ 
    Complex.abs (triangle.C - triangle.B) = Complex.abs (triangle.A - triangle.C)) :
  ∃ (motionC : CircularMotion), 
    triangle.C = motionC.center + motionC.radius * Complex.exp (Complex.I * (motionC.angularVelocity * t + motionC.initialAngle)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_circular_motion_of_triangle_vertex_l818_81883
